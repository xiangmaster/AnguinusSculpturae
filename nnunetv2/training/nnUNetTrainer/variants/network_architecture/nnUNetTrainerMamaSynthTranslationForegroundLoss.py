"""Pre-to-post translation trainers whose loss is confined to the tissue foreground.

Built on the LPIPS+MSE fine-tune trainer, but the loss region is the tissue itself
rather than the whole acquired field of view.

Data comes from two places:

* pre, post and the lesion mask used for slice sampling come from the standard
  translation dataset;
* a per-case binary foreground mask comes from a second dataset directory, loaded
  with the same crop placement as pre and post so it stays pixel-aligned.

The loss has two parts:

* MSE over ``foreground * acquired`` only. Because the masked MSE divides by the
  mask sum, this is a true foreground-only mean: background pixels get neither
  gradient nor weight, rather than being averaged in as zeros.
* LPIPS on the composite that inference actually produces, namely the predicted
  foreground feathered onto the pre-contrast background along a signed-distance
  ramp, compared against the real post. Blending against the real background
  avoids the artificial hard edge that plain masking would introduce.

The network still predicts a residual over the whole canvas; only the region the
loss looks at changes. Launch with::

    nnUNetv2_train_mamasynth_translate <Dataset625> <fold> \
        -tr nnUNetTrainerMamaSynthTranslationLPIPS_BS_32_MAEFinetune_ForegroundLoss \
        --foreground-dir <foreground dataset> -pretrained_weights <mae.pth>
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import (
    NonDetMultiThreadedAugmenter,
)
from batchgenerators.dataloading.single_threaded_augmenter import (
    SingleThreadedAugmenter,
)
from threadpoolctl import threadpool_limits

from nnunetv2.training.mamasynth.data_loader import (
    MamaSynthTranslationFixedValDataLoader,
    MamaSynthTranslationSliceDataLoader,
)
from nnunetv2.training.mamasynth.dataset import (
    BACKGROUND_LABEL,
    BREAST_LABEL,
    LESION_LABEL,
    TARGET_SIZE,
    MamaSynthCase,
    _centered_placement,
    _place_into_canvas,
    _random_placement,
    open_b2nd,
    resample_slice,
    zscore_with_stats,
)
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerMamaSynth import (
    nnUNetTrainerMamaSynthTranslationLPIPS_BS_32_MAEFinetune,
)
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA

FOREGROUND_SUFFIX = "_foreground_mask.b2nd"


def _build_blend_weight_np(mask_bool: np.ndarray, feather: float) -> np.ndarray:
    """Turn a binary foreground mask into a soft blend weight in [0, 1].

    Mirrors the blend used at inference time: a signed-distance ramp that is 1 well
    inside the foreground, 0 well outside, and linear across plus or minus ``feather``
    pixels of the boundary. A non-positive feather returns the hard mask unchanged.
    """
    if feather is None or feather <= 0:
        return mask_bool.astype(np.float32)
    from scipy import ndimage

    dist_in = ndimage.distance_transform_edt(mask_bool)
    dist_out = ndimage.distance_transform_edt(~mask_bool)
    signed = dist_in - dist_out
    weight = np.clip(0.5 + signed / (2.0 * float(feather)), 0.0, 1.0)
    return weight.astype(np.float32)


def load_pre_post_delta_foreground_slice(
    case: MamaSynthCase,
    slice_idx: int,
    mean: float,
    std: float,
    foreground_path: Path,
    crop_center_yx: Tuple[float, float] | None = None,
) -> tuple:
    """Load one training slice together with all three masks, mutually aligned.

    Extends the standard pre/post slice loader with the breast label and the external
    foreground mask, placing every array with the same crop so they stay pixel-aligned.
    The lesion and breast labels are the raw, disjoint anatomical labels, meaning the
    breast label already excludes the lesion. Returns ``(pre, delta, post,
    acquired_image_mask, lesion_mask, breast_mask, foreground_mask)``. Passing
    ``crop_center_yx`` selects a deterministic centred crop instead of a random one.
    """
    pre = np.asarray(open_b2nd(case.pre_path)[slice_idx], dtype=np.float32)
    post = np.asarray(open_b2nd(case.post_path)[slice_idx], dtype=np.float32)
    raw_mask = np.asarray(open_b2nd(case.mask_path)[slice_idx], dtype=np.uint8)
    lesion_mask = (raw_mask == LESION_LABEL).astype(np.int16)
    breast_mask = (raw_mask == BREAST_LABEL).astype(np.int16)
    foreground_mask = (
        np.asarray(open_b2nd(Path(foreground_path))[slice_idx], dtype=np.uint8) > 0
    ).astype(np.int16)

    pre = zscore_with_stats(
        resample_slice(pre, case.spacing_y, case.spacing_x, is_mask=False), mean, std
    )
    post = zscore_with_stats(
        resample_slice(post, case.spacing_y, case.spacing_x, is_mask=False), mean, std
    )
    lesion_mask = resample_slice(
        lesion_mask, case.spacing_y, case.spacing_x, is_mask=True
    ).astype(np.int16)
    breast_mask = resample_slice(
        breast_mask, case.spacing_y, case.spacing_x, is_mask=True
    ).astype(np.int16)
    foreground_mask = resample_slice(
        foreground_mask, case.spacing_y, case.spacing_x, is_mask=True
    ).astype(np.int16)
    acquired_image_mask = np.ones(pre.shape, dtype=np.float32)

    if crop_center_yx is None:
        placement_yx = (
            _random_placement(pre.shape[0], TARGET_SIZE),
            _random_placement(pre.shape[1], TARGET_SIZE),
        )
    else:
        placement_yx = (
            _centered_placement(pre.shape[0], TARGET_SIZE, crop_center_yx[0]),
            _centered_placement(pre.shape[1], TARGET_SIZE, crop_center_yx[1]),
        )
    pre = _place_into_canvas(pre.astype(np.float32), placement_yx, fill=0.0)
    post = _place_into_canvas(post.astype(np.float32), placement_yx, fill=0.0)
    lesion_mask = _place_into_canvas(lesion_mask, placement_yx, fill=BACKGROUND_LABEL)
    breast_mask = _place_into_canvas(breast_mask, placement_yx, fill=BACKGROUND_LABEL)
    foreground_mask = _place_into_canvas(
        foreground_mask, placement_yx, fill=BACKGROUND_LABEL
    )
    acquired_image_mask = _place_into_canvas(
        acquired_image_mask, placement_yx, fill=0.0
    )
    delta = post - pre
    return (
        np.ascontiguousarray(pre),
        np.ascontiguousarray(delta.astype(np.float32)),
        np.ascontiguousarray(post),
        np.ascontiguousarray(acquired_image_mask.astype(np.float32)),
        np.ascontiguousarray(lesion_mask),
        np.ascontiguousarray(breast_mask),
        np.ascontiguousarray(foreground_mask),
    )


class _ForegroundDirMixin:
    """Shared foreground-mask path lookup and per-sample feathering."""

    feather: float = 4.0
    foreground_dir: Path

    def _foreground_path(self, case: MamaSynthCase) -> Path:
        """Return the foreground-mask file belonging to a case."""
        return self.foreground_dir / f"{case.case_id}{FOREGROUND_SUFFIX}"

    def _soft_weight(self, foreground_t: torch.Tensor) -> torch.Tensor:
        """Feather a hard foreground tensor into a soft weight of the same shape."""
        foreground_bool = foreground_t[0].detach().cpu().numpy() > 0.5
        soft = _build_blend_weight_np(foreground_bool, self.feather)
        return torch.from_numpy(soft)[None].float()


class MamaSynthTranslationForegroundLossSliceDataLoader(
    _ForegroundDirMixin, MamaSynthTranslationSliceDataLoader
):
    """Training loader that also emits the hard and feathered foreground masks.

    Slice sampling stays lesion-centric and is inherited unchanged; only the extra
    masks are added to each batch.
    """

    def __init__(
        self,
        cases: List[MamaSynthCase],
        batch_size: int,
        pre_mean: float,
        pre_std: float,
        foreground_dir: Path,
        feather: float = 4.0,
        transforms=None,
    ):
        """Record the foreground-mask directory and the feather width."""
        super().__init__(cases, batch_size, pre_mean, pre_std, transforms=transforms)
        self.foreground_dir = Path(foreground_dir)
        self.feather = float(feather)

    def _sample_foreground_example(self, max_tries: int = 8) -> tuple:
        """Draw one slice, preferring a draw whose crop still contains lesion voxels.

        After ``max_tries`` unsuccessful draws the last one is returned regardless, so
        sampling always makes progress.
        """
        result = None
        for _ in range(max_tries):
            case, slice_idx = self._foreground_slice()
            out = load_pre_post_delta_foreground_slice(
                case,
                slice_idx,
                self.pre_mean,
                self.pre_std,
                self._foreground_path(case),
            )
            result = (*out, f"{case.case_id}_s{slice_idx}")
            # Index 4 is the lesion mask: prefer crops that kept the lesion.
            if out[4].any():
                return result
        return result

    def generate_train_batch(self) -> dict:
        """Assemble one augmented batch of images, masks and blend weights.

        The image pair and all three masks are stacked into a single image/segmentation
        pair so the spatial transform moves them together, then split apart again. Masks
        are re-binarised after interpolation and intersected with the acquired region, and
        the delta is recomputed from the transformed pre and post rather than transformed
        directly.
        """
        data_all = None
        target_all = None
        post_all = None
        acquired_image_mask_all = None
        lesion_mask_all = None
        breast_mask_all = None
        foreground_mask_all = None
        foreground_soft_all = None
        keys: List[str] = []
        with torch.no_grad(), threadpool_limits(limits=1, user_api=None):
            for j in range(self.batch_size):
                pre, delta, post, acquired, lesion, breast, foreground, key = (
                    self._sample_foreground_example()
                )
                keys.append(key)

                image = torch.from_numpy(np.stack((pre, post), axis=0)).float()
                seg = torch.from_numpy(
                    np.stack((acquired, lesion > 0, breast > 0, foreground > 0), axis=0)
                ).float()
                if self.transforms is not None:
                    transformed = self.transforms(image=image, segmentation=seg)
                    image, seg = transformed["image"], transformed["segmentation"]

                pre_t = image[0:1].float()
                post_t = image[1:2].float()
                # Recompute the delta from the transformed images rather than transforming it, so
                # it stays exactly consistent with them.
                delta_t = (post_t - pre_t).float()
                acquired_t = (seg[0:1].float() > 0.5).float()
                lesion_t = ((seg[1:2].float() > 0.5) & (acquired_t > 0.5)).float()
                breast_t = ((seg[2:3].float() > 0.5) & (acquired_t > 0.5)).float()
                foreground_t = ((seg[3:4].float() > 0.5) & (acquired_t > 0.5)).float()
                soft_t = self._soft_weight(foreground_t)

                if data_all is None:
                    data_all = torch.empty(
                        (self.batch_size, *pre_t.shape), dtype=torch.float32
                    )
                    target_all = torch.empty(
                        (self.batch_size, *delta_t.shape), dtype=torch.float32
                    )
                    post_all = torch.empty(
                        (self.batch_size, *post_t.shape), dtype=torch.float32
                    )
                    acquired_image_mask_all = torch.empty(
                        (self.batch_size, *acquired_t.shape), dtype=torch.float32
                    )
                    lesion_mask_all = torch.empty(
                        (self.batch_size, *lesion_t.shape), dtype=torch.float32
                    )
                    breast_mask_all = torch.empty(
                        (self.batch_size, *breast_t.shape), dtype=torch.float32
                    )
                    foreground_mask_all = torch.empty(
                        (self.batch_size, *foreground_t.shape), dtype=torch.float32
                    )
                    foreground_soft_all = torch.empty(
                        (self.batch_size, *soft_t.shape), dtype=torch.float32
                    )
                data_all[j] = pre_t
                target_all[j] = delta_t
                post_all[j] = post_t
                acquired_image_mask_all[j] = acquired_t
                lesion_mask_all[j] = lesion_t
                breast_mask_all[j] = breast_t
                foreground_mask_all[j] = foreground_t
                foreground_soft_all[j] = soft_t

        return {
            "data": data_all,
            "target": target_all,
            "post": post_all,
            "acquired_image_mask": acquired_image_mask_all,
            "lesion_mask": lesion_mask_all,
            "breast_mask": breast_mask_all,
            "foreground_mask": foreground_mask_all,
            "foreground_soft": foreground_soft_all,
            "keys": keys,
        }


class MamaSynthTranslationForegroundLossFixedValDataLoader(
    _ForegroundDirMixin, MamaSynthTranslationFixedValDataLoader
):
    """Fixed validation window, extended with the foreground mask and blend weight."""

    def __init__(
        self,
        cases: List[MamaSynthCase],
        batch_size: int,
        pre_mean: float,
        pre_std: float,
        foreground_dir: Path,
        feather: float = 4.0,
    ):
        """Record the foreground-mask directory and the feather width."""
        super().__init__(cases, batch_size, pre_mean, pre_std)
        self.foreground_dir = Path(foreground_dir)
        self.feather = float(feather)

    def generate_train_batch(self) -> dict:
        """Emit the next fixed batch of validation slices, wrapping at the end.

        No augmentation is applied: every slice uses its stored centred crop so the
        validation set is identical from epoch to epoch.
        """
        end = min(self._cursor + self.batch_size, len(self.samples))
        batch_samples = self.samples[self._cursor : end]
        self._cursor = 0 if end == len(self.samples) else end

        data_all = []
        target_all = []
        post_all = []
        acquired_image_mask_all = []
        lesion_mask_all = []
        breast_mask_all = []
        foreground_mask_all = []
        foreground_soft_all = []
        keys: List[str] = []
        with torch.no_grad(), threadpool_limits(limits=1, user_api=None):
            for case, slice_idx, crop_center_yx in batch_samples:
                pre, delta, post, acquired, lesion, breast, foreground = (
                    load_pre_post_delta_foreground_slice(
                        case,
                        slice_idx,
                        self.pre_mean,
                        self.pre_std,
                        self._foreground_path(case),
                        crop_center_yx=crop_center_yx,
                    )
                )
                acquired_t = torch.from_numpy(acquired)[None].float()
                breast_t = (
                    (torch.from_numpy(breast)[None].float() > 0.5) & (acquired_t > 0.5)
                ).float()
                foreground_t = (
                    (torch.from_numpy(foreground)[None].float() > 0.5)
                    & (acquired_t > 0.5)
                ).float()
                data_all.append(torch.from_numpy(pre)[None].float())
                target_all.append(torch.from_numpy(delta)[None].float())
                post_all.append(torch.from_numpy(post)[None].float())
                acquired_image_mask_all.append(acquired_t)
                lesion_mask_all.append(torch.from_numpy(lesion > 0)[None].float())
                breast_mask_all.append(breast_t)
                foreground_mask_all.append(foreground_t)
                foreground_soft_all.append(self._soft_weight(foreground_t))
                keys.append(f"{case.case_id}_s{slice_idx}")

        return {
            "data": torch.stack(data_all),
            "target": torch.stack(target_all),
            "post": torch.stack(post_all),
            "acquired_image_mask": torch.stack(acquired_image_mask_all),
            "lesion_mask": torch.stack(lesion_mask_all),
            "breast_mask": torch.stack(breast_mask_all),
            "foreground_mask": torch.stack(foreground_mask_all),
            "foreground_soft": torch.stack(foreground_soft_all),
            "keys": keys,
        }


class nnUNetTrainerMamaSynthTranslationLPIPS_BS_32_MAEFinetune_ForegroundLoss(
    nnUNetTrainerMamaSynthTranslationLPIPS_BS_32_MAEFinetune
):
    """LPIPS+MSE fine-tune trainer with the loss restricted to the tissue foreground.

    MSE is a pure foreground mean, so the background is ignored outright. LPIPS is
    taken on the feathered composite of the prediction over the pre-contrast
    background, which is the image the inference script assembles.
    """

    # Width in pixels of the signed-distance seam used for the composite; matches the
    # feather the inference script blends with.
    lpips_composite_feather = 4.0

    def _resolve_foreground_dir(self) -> Path:
        """Locate the foreground-mask dataset from the environment.

        Raises if the variable is unset or does not point at a directory, since the
        trainer cannot run without those masks.
        """
        directory = os.environ.get("MAMASYNTH_FOREGROUND_DIR")
        if not directory:
            raise RuntimeError(
                "ForegroundLoss trainer needs the foreground-mask dataset. Pass --foreground-dir "
                "(e.g. .../Dataset627_foreground); MAMASYNTH_FOREGROUND_DIR is not set."
            )
        path = Path(directory)
        if not path.is_dir():
            raise NotADirectoryError(
                f"MAMASYNTH_FOREGROUND_DIR is not a directory: {path}"
            )
        return path

    def get_dataloaders(self):
        """Build the random training loader and the fixed validation loader.

        Both are pointed at the foreground-mask directory; the validation loader's length
        sets the number of validation iterations per epoch.
        """
        foreground_dir = self._resolve_foreground_dir()
        patch_size = self.configuration_manager.patch_size
        rotation_for_DA, _, _, mirror_axes = (
            self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        )
        tr_transforms = self.get_translation_training_transforms(
            patch_size, rotation_for_DA, mirror_axes
        )

        tr_keys, val_keys = self.do_split()
        tr_cases = [c for c in self.cases if c.case_id in set(tr_keys)]
        val_cases = [c for c in self.cases if c.case_id in set(val_keys)]
        dl_tr = MamaSynthTranslationForegroundLossSliceDataLoader(
            tr_cases,
            self.batch_size,
            self.pre_stats["mean"],
            self.pre_stats["std"],
            foreground_dir,
            feather=self.lpips_composite_feather,
            transforms=tr_transforms,
        )
        dl_val = MamaSynthTranslationForegroundLossFixedValDataLoader(
            val_cases,
            self.batch_size,
            self.pre_stats["mean"],
            self.pre_stats["std"],
            foreground_dir,
            feather=self.lpips_composite_feather,
        )
        self.fixed_validation_loader = dl_val
        self.num_val_iterations_per_epoch = dl_val.num_batches
        self.print_to_log_file(
            f"Foreground-loss training. Foreground masks: {foreground_dir} "
            f"(feather {self.lpips_composite_feather}px). "
            f"Fixed validation: {len(dl_val.samples)} slices in {dl_val.num_batches} batches"
        )

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(
                data_loader=dl_tr,
                transform=None,
                num_processes=allowed_num_processes,
                num_cached=max(6, allowed_num_processes // 2),
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )
        mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        _ = next(mt_gen_train)
        return mt_gen_train, mt_gen_val

    def _lpips_preprocess_nomask(self, image: torch.Tensor) -> torch.Tensor:
        """Scale and replicate to three channels, without zeroing anything out.

        The usual preprocessing nulls synthetic padding; the composite already carries the
        real pre-contrast background there, so nothing needs masking.
        """
        image = (
            image.clamp(-self.lpips_clip_sigma, self.lpips_clip_sigma)
            / self.lpips_clip_sigma
        )
        return image.expand(-1, 3, -1, -1)

    def _lpips_loss_composite(
        self, composite_pred: torch.Tensor, target_post: torch.Tensor
    ) -> torch.Tensor:
        """Perceptual distance between the blended composite and the real post image.

        The metric object accumulates state across calls, so it is reset on both sides of
        the evaluation.
        """
        model = self._get_lpips_model()
        pred = self._lpips_preprocess_nomask(composite_pred)
        target = self._lpips_preprocess_nomask(target_post)
        model.reset()
        loss = model(pred, target)
        model.reset()
        return loss

    def _mse_regions(
        self,
        acquired_image_mask: torch.Tensor,
        foreground_mask: torch.Tensor,
        breast_mask: torch.Tensor,
        lesion_mask: torch.Tensor,
    ):
        """Return the ``(mask, weight)`` pairs whose weighted sum forms the MSE term.

        Weights sum to one so the magnitude of the MSE, and therefore its balance against
        the perceptual term, does not change when subclasses split the region up. This
        base version uses the foreground as a single region.
        """
        return [(foreground_mask * acquired_image_mask, 1.0)]

    def _combined_mse(self, prediction, target, regions) -> torch.Tensor:
        """Sum the per-region masked MSEs with their weights."""
        total = None
        for mask, weight in regions:
            term = float(weight) * self._masked_mse(prediction, target, mask)
            total = term if total is None else total + term
        return total

    def _foreground_step(self, batch: dict):
        """Shared forward pass for training and validation.

        Adds the predicted residual to the input, scores it region-wise against the real
        post, blends it onto the pre-contrast background for the perceptual term, and also
        returns the error of leaving the input unchanged as a reference point. Returns
        ``(mse_loss, lpips_loss, loss, baseline_mse)``.
        """
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"].to(self.device, non_blocking=True)
        acquired_image_mask = batch["acquired_image_mask"].to(
            self.device, non_blocking=True
        )
        foreground_mask = batch["foreground_mask"].to(self.device, non_blocking=True)
        foreground_soft = batch["foreground_soft"].to(self.device, non_blocking=True)
        breast_mask = batch["breast_mask"].to(self.device, non_blocking=True)
        lesion_mask = batch["lesion_mask"].to(self.device, non_blocking=True)

        output = self.network(data)
        prediction_post = data + output
        target_post = data + target
        regions = self._mse_regions(
            acquired_image_mask, foreground_mask, breast_mask, lesion_mask
        )
        mse_loss = self._combined_mse(prediction_post, target_post, regions)
        composite_pred = (
            foreground_soft * prediction_post + (1.0 - foreground_soft) * data
        )
        lpips_loss = self._lpips_loss_composite(composite_pred, target_post)
        loss = mse_loss + self.lpips_weight * lpips_loss
        baseline_mse = self._combined_mse(data, target_post, regions)
        return mse_loss, lpips_loss, loss, baseline_mse

    def train_step(self, batch: dict) -> dict:
        """Run the shared forward pass and take one clipped optimizer step."""
        from nnunetv2.utilities.helpers import dummy_context

        self.optimizer.zero_grad(set_to_none=True)
        with (
            torch.autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            mse_loss, lpips_loss, loss, baseline_mse = self._foreground_step(batch)

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
            "mse": mse_loss.detach().cpu().numpy(),
            "baseline_mse": baseline_mse.detach().cpu().numpy(),
            "lpips": lpips_loss.detach().cpu().numpy(),
        }

    def validation_step(self, batch: dict) -> dict:
        """Run the shared forward pass without updating the weights."""
        from nnunetv2.utilities.helpers import dummy_context

        with (
            torch.autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            mse_loss, lpips_loss, loss, baseline_mse = self._foreground_step(batch)
        return {
            "loss": loss.detach().cpu().numpy(),
            "mse": mse_loss.detach().cpu().numpy(),
            "baseline_mse": baseline_mse.detach().cpu().numpy(),
            "lpips": lpips_loss.detach().cpu().numpy(),
        }


class nnUNetTrainerMamaSynthTranslationLPIPS_BS_32_MAEFinetune_ForegroundBreastLoss(
    nnUNetTrainerMamaSynthTranslationLPIPS_BS_32_MAEFinetune_ForegroundLoss
):
    """Splits the MSE across two disjoint regions instead of one.

    The regions are the foreground outside the breast and the breast itself, lesion
    included. The lesion lies inside the breast, which lies inside the foreground, but
    every pixel is scored in exactly one region, so nothing is counted twice. The
    weights sum to one, keeping the overall MSE magnitude equal to the single-region
    parent. The perceptual term is unchanged.
    """

    # Weights for (foreground outside breast, breast). Must sum to 1 so the MSE keeps
    # the same magnitude as the single-region parent.
    weight_foreground = 0.5
    weight_breast = 0.5

    def _mse_regions(
        self, acquired_image_mask, foreground_mask, breast_mask, lesion_mask
    ):
        """Split the foreground into breast and non-breast tissue."""
        acquired = acquired_image_mask
        # The stored breast label excludes the lesion, so add it back to get the whole
        # breast.
        breast_full = ((breast_mask + lesion_mask) > 0.5).float()
        foreground_outer = (
            (foreground_mask > 0.5) & ~(breast_full > 0.5)
        ).float() * acquired
        breast_region = breast_full * acquired
        return [
            (foreground_outer, self.weight_foreground),
            (breast_region, self.weight_breast),
        ]


class nnUNetTrainerMamaSynthTranslationLPIPS_BS_32_MAEFinetune_ForegroundBreastLesionLoss(
    nnUNetTrainerMamaSynthTranslationLPIPS_BS_32_MAEFinetune_ForegroundLoss
):
    """Splits the MSE across three disjoint regions.

    The regions are the foreground outside the breast, the breast excluding the lesion,
    and the lesion. Every pixel is scored once, and the weights sum to one so the
    overall MSE magnitude matches the single-region parent. The perceptual term is
    unchanged.
    """

    # Weights for (foreground outside breast, breast shell, lesion). Must sum to 1.
    weight_foreground = 1.0 / 3.0
    weight_breast = 1.0 / 3.0
    weight_lesion = 1.0 / 3.0

    def _mse_regions(
        self, acquired_image_mask, foreground_mask, breast_mask, lesion_mask
    ):
        """Split the foreground into non-breast tissue, breast shell and lesion."""
        acquired = acquired_image_mask
        lesion_bool = lesion_mask > 0.5
        # Already disjoint from the lesion, so it is the breast shell as stored.
        breast_bool = breast_mask > 0.5
        foreground_bool = foreground_mask > 0.5
        lesion_region = lesion_bool.float() * acquired
        breast_shell = (breast_bool & ~lesion_bool).float() * acquired
        foreground_outer = (
            foreground_bool & ~breast_bool & ~lesion_bool
        ).float() * acquired
        return [
            (foreground_outer, self.weight_foreground),
            (breast_shell, self.weight_breast),
            (lesion_region, self.weight_lesion),
        ]
