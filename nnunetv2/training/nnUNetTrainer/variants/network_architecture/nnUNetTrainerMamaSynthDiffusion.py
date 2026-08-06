"""Conditional residual-diffusion trainers for pre-to-post synthesis.

Instead of predicting the post-minus-pre delta in a single forward pass, these
trainers learn a denoising diffusion model over that residual, conditioned on the
pre-contrast slice.

How it is put together:
  * The residual ``r0 = post - pre`` is divided by ``residual_scale`` so it has
    roughly unit variance and the cosine noise schedule's signal-to-noise ratio
    lands in a sensible range. The scale is estimated once from the data and
    written to disk, so it survives a resume unchanged.
  * The network predicts the noise (epsilon parametrization) and is trained with a
    masked MSE against the noise that was actually added.
  * Conditioning: the noisy residual and the pre-contrast slice enter as separate
    input channels.
  * Timestep conditioning: a sinusoidal embedding runs through an MLP and is
    injected into every encoder and decoder block as a per-channel scale and shift.
  * Training draws one random timestep per sample per iteration and takes a single
    backward pass.
  * Sampling is deterministic DDIM over a small number of steps.

Stage widths and strides come from the plans architecture, so the memory footprint
matches the U-Net the other trainers use.

Note for inference: a checkpoint from these trainers is not a single-pass translate
network. A predictor has to call ``MamaSynthDiffusionUNet.ddim_sample(cond=pre, ...)``
and form ``post = pre + residual_scale * sample``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerMamaSynth import (
    nnUNetTrainerMamaSynthTranslation,
)


def _cosine_alphas_cumprod(num_timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine noise schedule, returned as the cumulative product of alphas.

    Follows Nichol and Dhariwal: the cumulative product is defined directly from a
    cosine curve, the implied betas are clamped away from 0 and 1, and the product is
    rebuilt from them.
    """
    steps = num_timesteps + 1
    x = torch.linspace(0, num_timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = betas.clamp(1e-8, 0.999)
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0).to(torch.float32)


def _timestep_embedding(
    timesteps: torch.Tensor, dim: int, max_period: int = 10000
) -> torch.Tensor:
    """Sinusoidal embedding of timestep indices, mapping [B] to [B, dim]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=timesteps.device)
        / max(half, 1)
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    # Odd embedding widths leave one slot unfilled by the cos/sin pair.
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def _group_count(channels: int) -> int:
    """Pick the largest group count up to 32 that divides ``channels``."""
    for g in (32, 16, 8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class FiLMResBlock(nn.Module):
    """Residual block of two norm-activation-convolution pairs.

    The time embedding is projected to a per-channel scale and shift and applied after
    the second normalization, so timestep information reaches every block.
    """

    def __init__(self, in_channels: int, out_channels: int, temb_dim: int):
        """Build the two convolutions, the conditioning projection and the skip path.

        The skip is a 1x1 convolution when the channel count changes, otherwise identity.
        """
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        # Projects the time embedding to a scale and a shift per output channel.
        self.film = nn.Linear(temb_dim, 2 * out_channels)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        """Apply both convolutions with the time-conditioned modulation between them."""
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.film(F.silu(temb)).chunk(2, dim=1)
        h = self.norm2(h)
        h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(F.silu(h))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    """Self-attention over spatial positions, used only at low resolutions."""

    def __init__(self, channels: int):
        """Build the fused query/key/value projection and the output projection."""
        super().__init__()
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.scale = channels**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attend over all spatial positions and add the result to the input."""
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        q = q.reshape(b, c, h * w).permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w).permute(0, 2, 1)
        attn = torch.softmax((q @ k) * self.scale, dim=-1)
        out = (attn @ v).permute(0, 2, 1).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    """Halve the spatial resolution with a strided convolution."""

    def __init__(self, in_channels: int, out_channels: int):
        """Build the stride-2 convolution."""
        super().__init__()
        self.op = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the strided convolution."""
        return self.op(x)


class Upsample(nn.Module):
    """Double the spatial resolution by nearest-neighbour interpolation plus a convolution."""

    def __init__(self, in_channels: int, out_channels: int):
        """Build the post-interpolation convolution."""
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Interpolate to twice the size, then convolve."""
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class MamaSynthDiffusionUNet(nn.Module):
    """Conditional noise-prediction U-Net with time conditioning at every block.

    ``forward(x_t, cond, t)`` takes the noisy residual ``[B, 1, H, W]``, the
    conditioning images ``[B, Ccond, H, W]`` and the timestep indices ``[B]``, and
    returns the predicted noise ``[B, 1, H, W]``.
    """

    def __init__(
        self,
        cond_channels: int,
        features_per_stage,
        strides,
        num_res_blocks: int = 2,
        attn_resolution: int = 16,
        patch_size: int = 512,
        num_timesteps: int = 1000,
        schedule: str = "cosine",
    ):
        """Assemble the encoder, decoder and noise schedule from the plans architecture.

        A stage downsamples when its stride along the first axis is 2. Attention is added
        at stages whose resolution has fallen to ``attn_resolution`` or below. The output
        convolution is zero-initialised, so the model starts by predicting no noise.
        """
        super().__init__()
        features_per_stage = [int(f) for f in features_per_stage]
        # A stage downsamples when its stride along the first spatial axis is 2.
        downsample_flags = [int(s[0]) == 2 for s in strides]
        n_stages = len(features_per_stage)

        self.num_timesteps = int(num_timesteps)
        self.out_channels = 1
        self.cond_channels = int(cond_channels)

        temb_dim = features_per_stage[0] * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(features_per_stage[0], temb_dim),
            nn.SiLU(),
            nn.Linear(temb_dim, temb_dim),
        )
        self.sinusoidal_dim = features_per_stage[0]

        resolutions = []
        res = patch_size
        for i in range(n_stages):
            if i > 0 and downsample_flags[i]:
                res = res // 2
            resolutions.append(res)
        # Attention is only affordable once the feature map has shrunk far enough.
        use_attn = [r <= attn_resolution for r in resolutions]

        in_ch = self.out_channels + self.cond_channels
        self.in_conv = nn.Conv2d(in_ch, features_per_stage[0], kernel_size=3, padding=1)

        self.enc_downsamplers = nn.ModuleList()
        self.enc_stages = nn.ModuleList()
        prev = features_per_stage[0]
        for i in range(n_stages):
            out = features_per_stage[i]
            if i > 0 and downsample_flags[i]:
                self.enc_downsamplers.append(Downsample(prev, out))
                stage_in = out
            else:
                self.enc_downsamplers.append(nn.Identity())
                stage_in = prev
            blocks = nn.ModuleList()
            cur = stage_in
            for _ in range(num_res_blocks):
                blocks.append(FiLMResBlock(cur, out, temb_dim))
                cur = out
                if use_attn[i]:
                    blocks.append(AttnBlock(out))
            self.enc_stages.append(blocks)
            prev = out

        self.dec_upsamplers = nn.ModuleList()
        self.dec_stages = nn.ModuleList()
        for i in reversed(range(n_stages - 1)):
            skip_ch = features_per_stage[i]
            up_in = features_per_stage[i + 1]
            self.dec_upsamplers.append(Upsample(up_in, skip_ch))
            blocks = nn.ModuleList()
            # The first block of each decoder stage sees the upsampled features and the skip
            # connection concatenated.
            cur = skip_ch * 2
            out = features_per_stage[i]
            for _ in range(num_res_blocks):
                blocks.append(FiLMResBlock(cur, out, temb_dim))
                cur = out
                if use_attn[i]:
                    blocks.append(AttnBlock(out))
            self.dec_stages.append(blocks)

        self.out_norm = nn.GroupNorm(
            _group_count(features_per_stage[0]), features_per_stage[0]
        )
        self.out_conv = nn.Conv2d(
            features_per_stage[0], self.out_channels, kernel_size=3, padding=1
        )
        # Zero-initialise the output layer so the model starts out predicting no noise,
        # which keeps the first training steps stable.
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

        if schedule != "cosine":
            raise ValueError(f"Unsupported schedule: {schedule}")
        alphas_cumprod = _cosine_alphas_cumprod(self.num_timesteps)
        self.register_buffer("alphas_cumprod", alphas_cumprod, persistent=True)

    def forward(
        self, x_t: torch.Tensor, cond: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Predict the noise in ``x_t`` given the conditioning images and the timestep.

        The noisy residual and conditioning are concatenated channel-wise, run through the
        encoder while retaining one skip per stage, and decoded from the bottleneck with
        those skips concatenated back in.
        """
        temb = self.time_mlp(_timestep_embedding(t, self.sinusoidal_dim).to(x_t.dtype))
        h = self.in_conv(torch.cat([x_t, cond], dim=1))
        skips = []
        for downsampler, blocks in zip(self.enc_downsamplers, self.enc_stages):
            h = downsampler(h)
            for block in blocks:
                h = block(h, temb) if isinstance(block, FiLMResBlock) else block(h)
            skips.append(h)
        # The last encoder output is the bottleneck; decoding walks back up from there,
        # pairing each stage with its stored skip.
        h = skips[-1]
        for upsampler, blocks, skip in zip(
            self.dec_upsamplers, self.dec_stages, reversed(skips[:-1])
        ):
            h = upsampler(h)
            h = torch.cat([h, skip], dim=1)
            for block in blocks:
                h = block(h, temb) if isinstance(block, FiLMResBlock) else block(h)
        return self.out_conv(F.silu(self.out_norm(h)))

    def q_sample(
        self, r0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Add noise to a clean residual at timestep ``t`` (the forward diffusion step)."""
        acp = self.alphas_cumprod[t].to(r0.dtype)[:, None, None, None]
        return acp.sqrt() * r0 + (1.0 - acp).sqrt() * noise

    def predict_x0_from_eps(
        self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor
    ) -> torch.Tensor:
        """Recover the clean residual implied by a noisy sample and a noise prediction."""
        acp = self.alphas_cumprod[t].to(x_t.dtype)[:, None, None, None]
        return (x_t - (1.0 - acp).sqrt() * eps) / acp.sqrt().clamp_min(1e-8)

    @torch.no_grad()
    def ddim_sample(
        self,
        cond: torch.Tensor,
        num_steps: int = 16,
        x0_clip: float | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample a residual deterministically with DDIM (eta = 0).

        Starts from Gaussian noise and walks a descending, evenly spaced subset of the
        training timesteps. At each step the network's noise prediction gives an estimate
        of the clean residual, which is re-noised to the next timestep; the final estimate
        is returned. ``x0_clip``, when set, bounds that estimate at every step.
        """
        b, _, h, w = cond.shape
        device = cond.device
        x = torch.randn(b, self.out_channels, h, w, device=device, generator=generator)
        x = x.to(cond.dtype)
        step_indices = (
            torch.linspace(0, self.num_timesteps - 1, num_steps, device=device)
            .round()
            .long()
        )
        step_indices = torch.unique(step_indices)
        seq = list(step_indices.tolist())[::-1]
        for i, t_cur in enumerate(seq):
            t_batch = torch.full((b,), int(t_cur), device=device, dtype=torch.long)
            eps = self.forward(x, cond, t_batch)
            x0 = self.predict_x0_from_eps(x, t_batch, eps)
            if x0_clip is not None:
                x0 = x0.clamp(-x0_clip, x0_clip)
            t_next = seq[i + 1] if i + 1 < len(seq) else None
            if t_next is None:
                x = x0
            else:
                acp_next = self.alphas_cumprod[t_next].to(x.dtype)
                x = acp_next.sqrt() * x0 + (1.0 - acp_next).sqrt() * eps
        return x


class nnUNetTrainerMamaSynthDiffusion(nnUNetTrainerMamaSynthTranslation):
    """Conditional residual diffusion trainer with cosine schedule and time conditioning.

    Trains the denoiser on the normalized residual conditioned on the pre-contrast
    slice. The reconstruction figure used for checkpoint selection is computed from
    actual DDIM samples rather than from a one-step estimate, so it reflects what
    inference will produce.
    """

    batch_size_override = 16

    num_train_timesteps = 1000
    # Diffusion hyperparameters; subclasses override these.
    num_sampling_steps = 16
    num_res_blocks = 2
    attn_resolution = 16
    x0_clip = None
    residual_scale_estimation_batches = 20
    _residual_scale_default = 1.0

    def __init__(
        self, plans, configuration, fold, dataset_json, device=torch.device("cuda")
    ):
        """Lower the learning rate and initialise the residual scale."""
        super().__init__(plans, configuration, fold, dataset_json, device)
        # Noise prediction trains more stably at a lower learning rate than the regression
        # trainers' default.
        self.initial_lr = 1e-4
        self.residual_scale = float(self._residual_scale_default)

    @staticmethod
    def build_network_architecture(
        plans_manager,
        configuration_manager,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision=True,
    ):
        """Build the diffusion U-Net using the stage widths and strides from the plans."""
        arch = configuration_manager.network_arch_init_kwargs
        patch = configuration_manager.patch_size
        return MamaSynthDiffusionUNet(
            cond_channels=num_input_channels,
            features_per_stage=arch["features_per_stage"],
            strides=arch["strides"],
            num_res_blocks=nnUNetTrainerMamaSynthDiffusion.num_res_blocks,
            attn_resolution=nnUNetTrainerMamaSynthDiffusion.attn_resolution,
            patch_size=int(patch[0]),
            num_timesteps=nnUNetTrainerMamaSynthDiffusion.num_train_timesteps,
            schedule="cosine",
        )

    def _net(self) -> MamaSynthDiffusionUNet:
        """Return the underlying network, unwrapping the DDP container if present."""
        net = self.network
        return net.module if hasattr(net, "module") else net

    def set_deep_supervision_enabled(self, enabled: bool):
        """No-op: the diffusion network has no deep-supervision heads to toggle."""
        return

    def on_train_start(self):
        """Run the base start-up, then make sure the residual scale is available."""
        super().on_train_start()
        self._ensure_residual_scale()

    def _ensure_residual_scale(self):
        """Load the residual scale from disk, or estimate it once and persist it.

        The estimate is the root-mean-square of the residual over acquired pixels, taken
        over a handful of training batches. Only rank zero computes it; the others wait at
        a barrier and read the file it wrote.
        """
        path = Path(self.output_folder_base) / "diffusion_residual_scale.json"
        if path.is_file():
            with path.open("r", encoding="utf-8") as handle:
                self.residual_scale = float(json.load(handle)["residual_scale"])
            self.print_to_log_file(
                f"Loaded residual_scale={self.residual_scale:.6f} from {path}"
            )
            return

        if self.local_rank == 0:
            sq_sum = 0.0
            count = 0.0
            for _ in range(self.residual_scale_estimation_batches):
                batch = next(self.dataloader_train)
                target = batch["target"]
                if isinstance(target, list):
                    target = target[0]
                mask = batch["acquired_image_mask"].to(target.dtype)
                sq_sum += float((target.square() * mask).sum().item())
                count += float(mask.sum().item())
            rms = math.sqrt(sq_sum / max(count, 1.0))
            self.residual_scale = max(rms, 1e-6)
            with path.open("w", encoding="utf-8") as handle:
                json.dump({"residual_scale": self.residual_scale}, handle)
            self.print_to_log_file(
                f"Estimated residual_scale={self.residual_scale:.6f} (masked RMS of residual over "
                f"{self.residual_scale_estimation_batches} batches); saved to {path}"
            )

        if self.is_ddp:
            import torch.distributed as dist

            dist.barrier()
            with path.open("r", encoding="utf-8") as handle:
                self.residual_scale = float(json.load(handle)["residual_scale"])

    def train_step(self, batch: dict) -> dict:
        """Take one denoising step: noise the residual at a random timestep and regress it.

        Alongside the training loss, a cheap single-step reconstruction estimate is logged
        for monitoring, together with the error of leaving the input unchanged.
        """
        from nnunetv2.utilities.helpers import dummy_context

        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"].to(self.device, non_blocking=True)
        acquired_image_mask = batch["acquired_image_mask"].to(
            self.device, non_blocking=True
        )
        net = self._net()

        self.optimizer.zero_grad(set_to_none=True)
        with (
            torch.autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            r0 = target / self.residual_scale
            t = torch.randint(
                0, net.num_timesteps, (data.shape[0],), device=self.device
            )
            noise = torch.randn_like(r0)
            x_t = net.q_sample(r0, t, noise)
            eps_hat = net(x_t, data, t)
            loss = self._masked_mse(eps_hat, noise, acquired_image_mask)
            # Reconstruction monitor only: one denoising step from a single random timestep, so
            # it is noisy but costs nothing extra.
            with torch.no_grad():
                x0 = net.predict_x0_from_eps(x_t, t, eps_hat)
                pred_post = data + x0 * self.residual_scale
                target_post = data + target
                recon_mse = self._masked_mse(
                    pred_post, target_post, acquired_image_mask
                )
                baseline_mse = self._masked_mse(data, target_post, acquired_image_mask)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {
            "loss": loss.detach().cpu().numpy(),
            "mse": recon_mse.detach().cpu().numpy(),
            "baseline_mse": baseline_mse.detach().cpu().numpy(),
        }

    def validation_step(self, batch: dict) -> dict:
        """Score one batch by both noise prediction and a full DDIM reconstruction."""
        from nnunetv2.utilities.helpers import dummy_context

        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"].to(self.device, non_blocking=True)
        acquired_image_mask = batch["acquired_image_mask"].to(
            self.device, non_blocking=True
        )
        net = self._net()

        with (
            torch.autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            r0 = target / self.residual_scale
            t = torch.randint(
                0, net.num_timesteps, (data.shape[0],), device=self.device
            )
            noise = torch.randn_like(r0)
            x_t = net.q_sample(r0, t, noise)
            eps_hat = net(x_t, data, t)
            eps_loss = self._masked_mse(eps_hat, noise, acquired_image_mask)
            # Full sampling loop, so the recorded reconstruction error matches what inference
            # would produce.
            sample = net.ddim_sample(
                data, num_steps=self.num_sampling_steps, x0_clip=self.x0_clip
            )
            pred_post = data + sample * self.residual_scale
            target_post = data + target
            recon_mse = self._masked_mse(pred_post, target_post, acquired_image_mask)
            baseline_mse = self._masked_mse(data, target_post, acquired_image_mask)

        return {
            "loss": eps_loss.detach().cpu().numpy(),
            "mse": recon_mse.detach().cpu().numpy(),
            "baseline_mse": baseline_mse.detach().cpu().numpy(),
        }

    def _save_example_batch_png(self, num_examples: int = 2):
        """Write a panel of input, reference, sampled output, both deltas and the error.

        Overrides the base version, which calls the network directly; here the output has
        to come from a sampling loop.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from batchgenerators.utilities.file_and_folder_operations import join

        batch = next(self.dataloader_train)
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"].to(self.device, non_blocking=True)
        acquired_image_mask = batch["acquired_image_mask"].to(
            self.device, non_blocking=True
        )
        keys = batch.get("keys", [""] * data.shape[0])
        net = self._net()

        was_training = self.network.training
        self.network.eval()
        with torch.no_grad():
            with (
                torch.autocast(self.device.type, enabled=True)
                if self.device.type == "cuda"
                else torch.autocast("cpu", enabled=False)
            ):
                sample = net.ddim_sample(
                    data, num_steps=self.num_sampling_steps, x0_clip=self.x0_clip
                )
        if was_training:
            self.network.train()

        pre = data.detach().cpu().float().numpy()
        target_delta = target.detach().cpu().float().numpy()
        pred_delta = (sample * self.residual_scale).detach().cpu().float().numpy()
        target_post = pre + target_delta
        pred_post = pre + pred_delta
        support = acquired_image_mask.detach().cpu().float().numpy() > 0.5
        err = np.abs(pred_post - target_post)
        n = min(num_examples, data.shape[0])

        titles = [
            "input pre-contrast",
            "reference post-contrast",
            "generated post (DDIM)",
            "reference delta",
            "generated delta",
            "abs post error",
        ]
        fig, axes = plt.subplots(n, 6, figsize=(18, 3 * n), squeeze=False)
        for i in range(n):
            images = [
                pre[i, 0],
                target_post[i, 0],
                pred_post[i, 0],
                target_delta[i, 0],
                pred_delta[i, 0],
                err[i, 0],
            ]
            for ax, image, title in zip(axes[i], images, titles):
                ax.imshow(np.where(support[i, 0], image, np.nan), cmap="gray")
                ax.set_title(title, fontsize=8)
                ax.axis("off")
            axes[i, 0].set_ylabel(str(keys[i]), fontsize=7)
        fig.suptitle(
            f"epoch {self.current_epoch} | DDIM {self.num_sampling_steps} steps | "
            f"residual_scale={self.residual_scale:.4f}",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(join(self.output_folder, "translation_examples.png"), dpi=100)
        plt.close(fig)


class nnUNetTrainerMamaSynthDiffusion_BS_32(nnUNetTrainerMamaSynthDiffusion):
    """Diffusion trainer at batch size 32."""

    batch_size_override = 32


class nnUNetTrainerMamaSynthDiffusion_BS_8(nnUNetTrainerMamaSynthDiffusion):
    """Diffusion trainer at batch size 8."""

    batch_size_override = 8


class nnUNetTrainerMamaSynthDiffusion_BS_32_Sample32(
    nnUNetTrainerMamaSynthDiffusion_BS_32
):
    """Batch size 32 with 32 sampling steps: slower validation, finer samples."""

    num_sampling_steps = 32


class nnUNetTrainerMamaSynthDiffusion_BS_16(nnUNetTrainerMamaSynthDiffusion):
    """Diffusion trainer at batch size 16."""

    batch_size_override = 16


class nnUNetTrainerMamaSynthDiffusion_BS_56(nnUNetTrainerMamaSynthDiffusion):
    """Diffusion trainer at batch size 56."""

    batch_size_override = 56
