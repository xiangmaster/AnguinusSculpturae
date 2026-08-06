from __future__ import annotations

from typing import List

import numpy as np
import torch
from batchgenerators.dataloading.data_loader import DataLoader
from threadpoolctl import threadpool_limits

from nnunetv2.training.mamasynth.mae_dataset import MamaSynthMAECase, load_image_slice


class MamaSynthMAESliceDataLoader(DataLoader):
    """Samples random 512x512 image slices (any modality) for masked-autoencoder pretraining.

    Slices are drawn uniformly over ALL slices in the dataset (not uniform over volumes): the
    cumulative slice counts map a single global random index back to (volume, local slice), so a
    volume with twice as many slices is twice as likely to be drawn. There is no foreground/lesion
    logic - the MAE dataset has no masks. Spatial augmentation is applied to the image with the
    acquired-image mask carried along as a segmentation channel so canvas padding tracks rotation /
    scaling border fills.

    Output: ``{'data': (B, 1, 512, 512), 'acquired_image_mask': (B, 1, 512, 512), 'keys': [...]}``.
    """

    def __init__(self, cases: List[MamaSynthMAECase], batch_size: int, transforms=None):
        super().__init__(cases, batch_size, 1, None, True, False, True, None)
        self.cases = cases
        self.indices = list(range(len(cases)))
        self.transforms = transforms
        self._cumulative_slices = np.cumsum([c.n_slices for c in cases], dtype=np.int64)
        self._total_slices = int(self._cumulative_slices[-1])

    def _draw_global_slice(self) -> tuple:
        global_slice = int(np.random.randint(self._total_slices))
        case_idx = int(
            np.searchsorted(self._cumulative_slices, global_slice, side="right")
        )
        previous = 0 if case_idx == 0 else int(self._cumulative_slices[case_idx - 1])
        return self.cases[case_idx], global_slice - previous

    def generate_train_batch(self) -> dict:
        data_all = None
        acquired_all = None
        keys: List[str] = []
        with torch.no_grad(), threadpool_limits(limits=1, user_api=None):
            for j in range(self.batch_size):
                case, slice_idx = self._draw_global_slice()
                image, acquired = load_image_slice(case, slice_idx)
                keys.append(f"{case.volume_id}_s{slice_idx}")

                img_t = torch.from_numpy(image)[None].float()  # (1, 512, 512)
                seg_t = torch.from_numpy(acquired)[None].float()  # (1, 512, 512)
                if self.transforms is not None:
                    transformed = self.transforms(
                        **{"image": img_t, "segmentation": seg_t}
                    )
                    img_t, seg_t = transformed["image"], transformed["segmentation"]
                acquired_t = (seg_t.float() > 0.5).float()

                if data_all is None:
                    data_all = torch.empty(
                        (self.batch_size, *img_t.shape), dtype=torch.float32
                    )
                    acquired_all = torch.empty(
                        (self.batch_size, *acquired_t.shape), dtype=torch.float32
                    )
                data_all[j] = img_t
                acquired_all[j] = acquired_t

        return {"data": data_all, "acquired_image_mask": acquired_all, "keys": keys}


class MamaSynthMAEFixedValDataLoader(MamaSynthMAESliceDataLoader):
    """Deterministic validation: one centered slice (centered crop) per validation volume.

    The sample list is fixed at construction (capped to ``max_samples`` to bound validation time;
    the cap is reported by the trainer, never silent). A sequential cursor walks the list so each
    epoch sees the identical slices; combined with a per-sample mask seed in the trainer this makes
    the reconstruction MSE strictly comparable across epochs.
    """

    def __init__(
        self,
        cases: List[MamaSynthMAECase],
        batch_size: int,
        max_samples: int = 1000,
    ):
        super().__init__(cases, batch_size, transforms=None)
        self.samples = [(case, case.n_slices // 2) for case in cases]
        self.num_volumes = len(cases)
        self.capped = len(self.samples) > max_samples
        if self.capped:
            self.samples = self.samples[:max_samples]
        if not self.samples:
            raise ValueError("Validation split contains no usable volumes.")
        self.num_batches = int(np.ceil(len(self.samples) / self.batch_size))
        self._cursor = 0

    def reset(self):
        self._cursor = 0

    def generate_train_batch(self) -> dict:
        end = min(self._cursor + self.batch_size, len(self.samples))
        batch_samples = self.samples[self._cursor : end]
        self._cursor = 0 if end == len(self.samples) else end

        data_all = []
        acquired_all = []
        keys: List[str] = []
        with torch.no_grad(), threadpool_limits(limits=1, user_api=None):
            for case, slice_idx in batch_samples:
                # Centered crop so the same anatomy is scored every epoch (default is random).
                image, acquired = load_image_slice(case, slice_idx, center_crop=True)
                data_all.append(torch.from_numpy(image)[None].float())
                acquired_all.append(torch.from_numpy(acquired)[None].float())
                keys.append(f"{case.volume_id}_s{slice_idx}")

        return {
            "data": torch.stack(data_all),
            "acquired_image_mask": torch.stack(acquired_all),
            "keys": keys,
        }
