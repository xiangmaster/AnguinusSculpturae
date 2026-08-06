# MAMA-SYNTH dataset builders

The scripts that convert 3D NIfTI cohorts into the three blosc2 datasets the trainers read
(`Dataset625_Pre_Seg`, `Dataset626_MAE`, `Dataset627_foreground`), plus the dataset-wide intensity
statistics used to normalize pre and post consistently.

Every path is a required argument, so each run states where it reads and writes.

Run them as modules:

```bash
python -m nnunetv2.preprocessing.mamasynth.<script> --help
```

---

## 1. `Dataset625_Pre_Seg` — pre + post volumes and the lesion/breast mask

Drives translation training and the breast and lesion segmenters.

```bash
python -m nnunetv2.preprocessing.mamasynth.create_dataset625_pre_seg_blosc2 \
    --raw_root    <raw_root> \
    --output_root <blosc2_root> \
    --workers 8 --clevel 3
```

Reads:

```
<raw_root>/images/<case>_0000.nii.gz     # pre
<raw_root>/images/<case>_0001.nii.gz     # post
<raw_root>/masks/<case>.nii.gz           # lesion = 1, breast = 2
```

Writes `<blosc2_root>/Dataset625_Pre_Seg/` with `<case>_pre.b2nd`, `<case>_post.b2nd`,
`<case>_mask.b2nd` and `manifest.csv`. Keeps the slices carrying the breast label. Arrays are
`(N, H, W)` with `chunks=(1, H, W)`, so one chunk is exactly one 2D slice. Each array carries vlmeta
(`case_id`, `through_axis_3d`, `spacing_3d`, `slice_spacing_2d`, `retained_indices_3d`,
`dataset_source`) — `retained_indices_3d` is what step 3 reuses to stay aligned. A case whose post
volume is absent or shape-mismatched is written with pre and mask, and the manifest records why.

Also takes `--dataset_name`, `--cases` (whitelist), `--limit`, `--overwrite`, `--dry_run`.

## 2. `Dataset626_MAE` — unlabeled slices for masked-autoencoder pretraining

```bash
python -m nnunetv2.preprocessing.mamasynth.create_dataset626_mae_blosc2 \
    --source_root <ssl_datasets_root> \
    --output_dir  <blosc2_root>/Dataset626_MAE \
    --workers 8
```

Walks `--source_root` recursively for image NIfTIs, treating each volume independently and skipping
directories named `masks`. Each volume becomes one `.b2nd` slice stack under a collision-safe id
(`<relative-path>__<volume>__<hash>.b2nd`), with vlmeta recording the source path, spacing, selected
axis and original slice indices. Needs no masks and no pairing, so it runs independently of step 1.

## 3. `Dataset627_foreground` — pre volumes plus binary tissue-support masks

Drives the foreground segmenter. Runs after step 1, reusing the slice axis and
`retained_indices_3d` from each `<case>_pre.b2nd` so the masks line up slice-for-slice.

```bash
python -m nnunetv2.preprocessing.mamasynth.create_dataset627_foreground_blosc2 \
    --pre_blosc2_dir       <blosc2_root>/Dataset625_Pre_Seg \
    --foreground_mask_root <foreground_masks> \
    --output_dir           <blosc2_root>/Dataset627_foreground \
    --workers 4
```

`--foreground_mask_root` holds the 3D masks, laid out as `<chunk>/<source>__<rest>.nii.gz`. They were
generated over the pre images with
[Foreground-and-Anonymization-Area-Segmentation](https://github.com/MIC-DKFZ/Foreground-and-Anonymization-Area-Segmentation),
taking its anatomical-foreground output.

### Matching masks to case IDs

How much of a mask filename survives into the Dataset625 case ID depends on the cohort, so
`case_ids_from_foreground_mask` offers every form and keeps whichever exists in the dataset:

| Mask filename | Dataset625 case ID | Cohorts |
|---|---|---|
| `MAMA-MIA__Breast_MRI_001` | `Breast_MRI_001` | most: MAMA-MIA, ISPY, ACRIN, Duke, TCGA, QIN, … |
| `fastMRI_breast__001` | `fastMRI_breast_001` | fastMRI (keeps the source prefix) |
| `odelia_CAM__ODELIA_BRAID1_0158_1` | `CAM_ODELIA_BRAID1_0158_1` | ODELIA: `CAM`, `MHA`, `RSH`, `RUMC`, `UKA`, `UMCU` |
| `<stem>` | `<stem>` | filenames with no `__` separator |

On the full dataset this resolves a mask for all 7555 Dataset625 cases in one pass — 6518 via the
first form, 296 fastMRI, 741 ODELIA. Two masks claiming the same case ID raise rather than pick one.

## 4. `training_pre_stats.json` — dataset-wide pre-contrast statistics

Translation training normalizes pre and post with one shared mean/std rather than per case. This
computes them with Welford's algorithm over the pre-contrast volumes:

```bash
python -m nnunetv2.preprocessing.mamasynth.compute_dataset_stats \
    --image_dir   <dce_image_dir> \
    --output_path <blosc2_root>/training_pre_stats.json
```

Reads `image_dir/<patient_id>/<patient_id>_<phase_index>.nii.gz`, treating the lowest phase index as
pre. Writes `{"mean", "std", "n_voxels", "n_patients"}`, passed to training via `--pre-stats`.

---

## `splits_final.json`

Hand-built and kept next to each dataset directory; the loaders read it as given.

It follows the **MAMA-MIA** folds — Duke, ISPY1, ISPY2 and NACT cases. All NACT cases are included;
for the other cohorts the folds cover a subset. Each fold is then extended with the **remaining
timepoints of the same patients**, which matters most for ISPY1 and ISPY2, since MAMA-MIA itself uses
only the first timepoint. Keeping every timepoint of a patient inside one fold is what stops the same
patient appearing on both sides of a split.
