from __future__ import annotations

from typing import List

import numpy as np
import torch
from batchgenerators.dataloading.data_loader import DataLoader
from threadpoolctl import threadpool_limits

from nnunetv2.training.mamasynth.dataset import (
    MamaSynthCase,
    lesion_slice_indices,
    load_pre_and_mask_slice,
    load_pre_post_delta_slice,
    peak_lesion_slice_and_center,
)


class MamaSynthSliceDataLoader(DataLoader):
    """Samples random 512x512 pre-contrast slices from blosc2 volumes and applies nnU-Net transforms.

    Output matches ``nnUNetDataLoader``: ``{'data': (B, 1, 512, 512), 'target': tensor | list, 'keys': [...]}``
    so the standard nnU-Net training/validation loop and deep supervision are reused unchanged. Foreground
    oversampling targets the (rare) lesion label rather than any foreground, since breast is present almost
    everywhere.
    """

    def __init__(
        self,
        cases: List[MamaSynthCase],
        batch_size: int,
        oversample_foreground_percent: float = 0.0,
        transforms=None,
        probabilistic_oversampling: bool = False,
    ):
        super().__init__(cases, batch_size, 1, None, True, False, True, None)
        self.cases = cases
        self.indices = list(range(len(cases)))
        self.oversample_foreground_percent = oversample_foreground_percent
        self.transforms = transforms
        self.probabilistic_oversampling = probabilistic_oversampling
        self._no_lesion: set = set()
        # Global slice index: cumulative slice counts so a random draw is uniform over ALL slices
        # in the dataset (not uniform over cases). O(#cases) memory.
        self._cumulative_slices = np.cumsum([c.n_slices for c in cases], dtype=np.int64)
        self._total_slices = int(self._cumulative_slices[-1])

    def _draw_global_slice(self) -> tuple:
        # Uniform over the global slice pool: pick a global slice id, map it back to (case, local slice).
        global_slice = int(np.random.randint(self._total_slices))
        case_idx = int(
            np.searchsorted(self._cumulative_slices, global_slice, side="right")
        )
        previous = 0 if case_idx == 0 else int(self._cumulative_slices[case_idx - 1])
        return self.cases[case_idx], global_slice - previous

    def _foreground_slice(self, max_tries: int = 128) -> tuple:
        # Sampling strategy: draw a slice globally-uniformly and redraw if it has no foreground
        # (no lesion). This yields uniform sampling over the set of lesion-containing slices.
        for _ in range(max_tries):
            case, slice_idx = self._draw_global_slice()
            if slice_idx in lesion_slice_indices(str(case.mask_path)):
                return case, slice_idx
        return (
            self._lesion_slice()
        )  # guaranteed-foreground fallback (should rarely trigger)

    def _lesion_slice(self, max_tries: int = 64) -> tuple:
        for _ in range(max_tries):
            case_idx = np.random.randint(len(self.cases))
            if case_idx in self._no_lesion:
                continue
            case = self.cases[case_idx]
            slices = lesion_slice_indices(str(case.mask_path))
            if slices:
                return case, int(np.random.choice(slices))
            self._no_lesion.add(case_idx)
        return (
            self._draw_global_slice()
        )  # last resort; should never happen with lesion data

    def _sample_foreground_example(self, max_tries: int = 8) -> tuple:
        # Draw a lesion-containing slice, load+crop it, and redraw if the actual 512x512 crop ended up
        # without foreground (the random crop can clip the lesion out of large-FOV slices).
        pre = mask = None
        for _ in range(max_tries):
            case, slice_idx = self._foreground_slice()
            pre, mask = load_pre_and_mask_slice(case, slice_idx)
            if mask.any():  # foreground (lesion) present in the cropped slice
                return pre, mask, f"{case.case_id}_s{slice_idx}"
        return (
            pre,
            mask,
            f"{case.case_id}_s{slice_idx}",
        )  # accept last attempt (very rare)

    def generate_train_batch(self) -> dict:
        data_all = None
        seg_all = None
        keys: List[str] = []
        with torch.no_grad(), threadpool_limits(limits=1, user_api=None):
            for j in range(self.batch_size):
                pre, mask, key = (
                    self._sample_foreground_example()
                )  # every slice contains foreground (lesion)
                keys.append(key)
                data = torch.from_numpy(pre)[None].float()  # (1, 512, 512)
                seg = torch.from_numpy(mask)[None].to(torch.int16)  # (1, 512, 512)

                if self.transforms is not None:
                    transformed = self.transforms(
                        **{"image": data, "segmentation": seg}
                    )
                    data, seg = transformed["image"], transformed["segmentation"]

                if data_all is None:
                    data_all = torch.empty(
                        (self.batch_size, *data.shape), dtype=torch.float32
                    )
                data_all[j] = data

                if isinstance(
                    seg, list
                ):  # deep supervision -> list of downsampled targets
                    if seg_all is None:
                        seg_all = [
                            torch.empty((self.batch_size, *s.shape), dtype=s.dtype)
                            for s in seg
                        ]
                    for level, s in enumerate(seg):
                        seg_all[level][j] = s
                else:
                    if seg_all is None:
                        seg_all = torch.empty(
                            (self.batch_size, *seg.shape), dtype=seg.dtype
                        )
                    seg_all[j] = seg

        return {"data": data_all, "target": seg_all, "keys": keys}


class MamaSynthTranslationSliceDataLoader(DataLoader):
    """Samples lesion-containing pre/post slice pairs for pre-to-post image translation.

    The network sees the normalized pre slice and predicts normalized-space delta ``post - pre``.
    ``acquired_image_mask`` is one for all pixels from the scan, including anatomical label 0, and zero
    only for synthetic canvas padding. ``lesion_mask`` contains anatomical label 1 only. MSE, LPIPS,
    GAN, and whole-image Curia terms exclude padding through the acquired-image mask; SSIM, Dice, and
    Curia lesion emphasis use the lesion mask.
    """

    def __init__(
        self,
        cases: List[MamaSynthCase],
        batch_size: int,
        pre_mean: float,
        pre_std: float,
        transforms=None,
    ):
        super().__init__(cases, batch_size, 1, None, True, False, True, None)
        self.cases = cases
        self.indices = list(range(len(cases)))
        self.pre_mean = float(pre_mean)
        self.pre_std = float(pre_std)
        self.transforms = transforms
        self._no_lesion: set = set()
        self._cumulative_slices = np.cumsum([c.n_slices for c in cases], dtype=np.int64)
        self._total_slices = int(self._cumulative_slices[-1])

    def _draw_global_slice(self) -> tuple:
        global_slice = int(np.random.randint(self._total_slices))
        case_idx = int(
            np.searchsorted(self._cumulative_slices, global_slice, side="right")
        )
        previous = 0 if case_idx == 0 else int(self._cumulative_slices[case_idx - 1])
        return self.cases[case_idx], global_slice - previous

    def _foreground_slice(self, max_tries: int = 128) -> tuple:
        for _ in range(max_tries):
            case, slice_idx = self._draw_global_slice()
            if slice_idx in lesion_slice_indices(str(case.mask_path)):
                return case, slice_idx
        return self._lesion_slice()

    def _lesion_slice(self, max_tries: int = 64) -> tuple:
        for _ in range(max_tries):
            case_idx = np.random.randint(len(self.cases))
            if case_idx in self._no_lesion:
                continue
            case = self.cases[case_idx]
            slices = lesion_slice_indices(str(case.mask_path))
            if slices:
                return case, int(np.random.choice(slices))
            self._no_lesion.add(case_idx)
        return self._draw_global_slice()

    def _sample_foreground_example(self, max_tries: int = 8) -> tuple:
        pre = delta = post = acquired_image_mask = lesion_mask = None
        key = ""
        for _ in range(max_tries):
            case, slice_idx = self._foreground_slice()
            pre, delta, post, acquired_image_mask, lesion_mask = (
                load_pre_post_delta_slice(case, slice_idx, self.pre_mean, self.pre_std)
            )
            key = f"{case.case_id}_s{slice_idx}"
            if lesion_mask.any():
                return pre, delta, post, acquired_image_mask, lesion_mask, key
        return pre, delta, post, acquired_image_mask, lesion_mask, key

    def generate_train_batch(self) -> dict:
        data_all = None
        target_all = None
        post_all = None
        acquired_image_mask_all = None
        lesion_mask_all = None
        keys: List[str] = []
        with torch.no_grad(), threadpool_limits(limits=1, user_api=None):
            for j in range(self.batch_size):
                pre, delta, post, acquired_image_mask, lesion_mask, key = (
                    self._sample_foreground_example()
                )
                keys.append(key)

                image = torch.from_numpy(np.stack((pre, post), axis=0)).float()
                seg = torch.from_numpy(
                    np.stack((acquired_image_mask, lesion_mask > 0), axis=0)
                ).float()
                if self.transforms is not None:
                    transformed = self.transforms(
                        **{"image": image, "segmentation": seg}
                    )
                    image, seg = transformed["image"], transformed["segmentation"]

                pre_t = image[0:1].float()
                post_t = image[1:2].float()
                delta_t = (post_t - pre_t).float()
                acquired_image_mask_t = (seg[0:1].float() > 0.5).float()
                lesion_mask_t = (
                    (seg[1:2].float() > 0.5) & (acquired_image_mask_t > 0.5)
                ).float()

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
                        (self.batch_size, *acquired_image_mask_t.shape),
                        dtype=torch.float32,
                    )
                    lesion_mask_all = torch.empty(
                        (self.batch_size, *lesion_mask_t.shape), dtype=torch.float32
                    )
                data_all[j] = pre_t
                target_all[j] = delta_t
                post_all[j] = post_t
                acquired_image_mask_all[j] = acquired_image_mask_t
                lesion_mask_all[j] = lesion_mask_t

        return {
            "data": data_all,
            "target": target_all,
            "post": post_all,
            "acquired_image_mask": acquired_image_mask_all,
            "lesion_mask": lesion_mask_all,
            "keys": keys,
        }


class MamaSynthTranslationFixedValDataLoader(MamaSynthTranslationSliceDataLoader):
    """Fixed five-slice validation window centered on each case's maximum-lesion slice."""

    def __init__(
        self,
        cases: List[MamaSynthCase],
        batch_size: int,
        pre_mean: float,
        pre_std: float,
    ):
        super().__init__(cases, batch_size, pre_mean, pre_std, transforms=None)
        self.samples = []
        for case in cases:
            peak_slice, crop_center_yx = peak_lesion_slice_and_center(
                str(case.mask_path), case.spacing_y, case.spacing_x
            )
            if case.n_slices < 5:
                print(
                    f"Skipping {case.case_id} from validation: only {case.n_slices} slices, "
                    f"cannot build a 5-slice window."
                )
                continue
            # Center the 5-slice window on the peak slice, clamping inward so the window stays in
            # bounds for cases whose peak lesion sits within 2 slices of a volume edge.
            center = int(np.clip(peak_slice, 2, case.n_slices - 3))
            self.samples.extend(
                (case, slice_idx, crop_center_yx)
                for slice_idx in range(center - 2, center + 3)
            )
        if not self.samples:
            raise ValueError("Validation split contains no usable lesion cases.")
        self.num_batches = int(np.ceil(len(self.samples) / self.batch_size))
        self._cursor = 0

    def reset(self):
        self._cursor = 0

    def generate_train_batch(self) -> dict:
        end = min(self._cursor + self.batch_size, len(self.samples))
        batch_samples = self.samples[self._cursor : end]
        self._cursor = 0 if end == len(self.samples) else end

        data_all = []
        target_all = []
        post_all = []
        acquired_image_mask_all = []
        lesion_mask_all = []
        keys: List[str] = []
        with torch.no_grad(), threadpool_limits(limits=1, user_api=None):
            for case, slice_idx, crop_center_yx in batch_samples:
                pre, delta, post, acquired_image_mask, lesion_mask = (
                    load_pre_post_delta_slice(
                        case,
                        slice_idx,
                        self.pre_mean,
                        self.pre_std,
                        crop_center_yx=crop_center_yx,
                    )
                )
                data_all.append(torch.from_numpy(pre)[None].float())
                target_all.append(torch.from_numpy(delta)[None].float())
                post_all.append(torch.from_numpy(post)[None].float())
                acquired_image_mask_all.append(
                    torch.from_numpy(acquired_image_mask)[None].float()
                )
                lesion_mask_all.append(torch.from_numpy(lesion_mask > 0)[None].float())
                keys.append(f"{case.case_id}_s{slice_idx}")

        return {
            "data": torch.stack(data_all),
            "target": torch.stack(target_all),
            "post": torch.stack(post_all),
            "acquired_image_mask": torch.stack(acquired_image_mask_all),
            "lesion_mask": torch.stack(lesion_mask_all),
            "keys": keys,
        }
