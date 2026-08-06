# MAMA-SYNTH — 2D pre-to-post contrast synthesis

This repository holds the code for our MAMA-SYNTH challenge entry: generating a synthetic
post-contrast breast-MRI slice from a pre-contrast slice, together with the segmentation models the
synthesis pipeline depends on.

It is a fork of [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet), used purely as a training and
inference framework. Everything specific to this work lives in the `*MamaSynth*` modules listed under
[Repository map](#repository-map); the rest of the tree is upstream nnU-Net and is not used here.

**Everything is 2D.** All models are slice-wise, the only configuration is `2d`, and the plans
identifier is `mamaSynthPlans`. Slices are placed on a fixed 512x512 canvas; the region outside the
acquired image is zero padding, and every loss and metric excludes it. No 3D or cascade configuration
is trained or supported.

---

## The final configuration

The pipeline described below is the one used for the final submission: two fold-ensembled
translation models composited along the breast and tissue boundaries, followed by a lesion
enhancement step.

Trained weights are published separately — see [Weights](#weights).

### What it does, per slice

Two translation models cover different parts of the image, and two segmentations decide where each
applies:

```
image_synth  = mean over folds of the outside-breast translation model
lesion_synth = mean over folds of the inside-breast translation model
stitched     = breast_soft     * lesion_synth + (1 - breast_soft)     * image_synth
final        = foreground_soft * stitched     + (1 - foreground_soft) * pre
final        = final * (1 + (gain - 1) * lesion_soft)
```

`breast_soft` and `foreground_soft` are signed-distance ramps across the mask boundaries, so the
transitions are gradual rather than hard steps. Outside the tissue support the real pre-contrast
image is kept unchanged. The lesion region comes from the fold-averaged lesion segmentation, with the
threshold lowered per slice until at least one in-breast voxel passes; scaling the intensity that is
already there preserves the lesion's internal texture.

### Models it loads

| Slot | Trainer | Checkpoint | Folds |
|---|---|---|---|
| `image` — outside-breast translation | `nnUNetTrainerMamaSynthTranslationLPIPS_BS_48_MAEFinetune_ep_200` | `checkpoint_best` | 0–3 |
| `lesion_image` — inside-breast translation | `nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceTverskyA02B08_LPIPS05_BS_48_MAEFinetune` | `checkpoint_best_dice` | 0–3 |
| `lesion_seg` — lesion segmentation | `nnUNetTrainerMamaSynthLesionTverskyA02B08_BS_64_epoch_1000` | `checkpoint_best` | 0–3 |
| `breast` — breast segmentation | `nnUNetTrainerMamaSynthBreast_BS_32` | `checkpoint_final` | 0 |
| `foreground` — tissue-support segmentation | `nnUNetTrainerMamaSynthForeground_BS_32` | `checkpoint_final` | 0 |

Composite settings: `feather 4`, `lesion_feather 2`, `lesion_gain 1.25`, `lesion_threshold 0.5`,
`lesion_step 0.05`, `air_from pre`, flip TTA on. Each has a matching flag on the prediction entry
point, listed under [Inference](#inference).

### Weights

The trained checkpoints are not stored in this repository. They are published on Hugging Face:

> **<https://huggingface.co/Bubenpo/AnguinusSculpturae>**

```bash
hf download Bubenpo/AnguinusSculpturae --local-dir mama-synth-weights
```

5.6 GB, the 14 networks of the table above. They are the checkpoints the submitted container bundles,
verbatim except that optimizer and grad-scaler state is stripped — inference reads only
`network_weights`, so predictions are unchanged. Per-fold files are renamed to a uniform
`checkpoint.pth` and grouped one directory per slot:

```
assets/
├── image/        {plans.json, dataset.json, fold_0..3/checkpoint.pth}
├── lesion_image/ {plans.json, dataset.json, fold_0..3/checkpoint.pth}
├── lesion_seg/   {plans.json, dataset.json, fold_0..3/checkpoint.pth}
├── breast/       {plans.json, dataset.json, checkpoint.pth}
└── foreground/   {plans.json, dataset.json, checkpoint.pth}
```

Every slot keeps its own `plans.json` and `dataset.json`: each network is rebuilt from them before
the weights are loaded, so they are not optional.

---

## Install

Install PyTorch for your hardware first, then this package in editable mode:

```bash
pip install -e .
```

This installs the `nnUNetv2_train_mamasynth*` and `nnUNetv2_predict_mamasynth*` entry points used
below.

Evaluation runs in a **separate environment** against the challenge evaluation
script from [`mama-research/mama-synth`](https://github.com/mama-research/mama-synth/) — see
[Evaluation](#evaluation).

---

## Data

Three blosc2 datasets are expected. The scripts that build them live under
[`nnunetv2/preprocessing/mamasynth/`](nnunetv2/preprocessing/mamasynth/) and are documented in
[its README](nnunetv2/preprocessing/mamasynth/README.md). They read per-case 3D NIfTI volumes and
write the per-case `.b2nd` slice stacks the data loaders consume.

| Dataset | Contents | Used for |
|---|---|---|
| `Dataset625_Pre_Seg` | pre + post volumes, mask with lesion = 1 and breast = 2 | translation, breast and lesion segmenters |
| `Dataset626_MAE` | unlabeled slices | masked-autoencoder pretraining |
| `Dataset627_foreground` | pre volumes + binary tissue-support masks | foreground segmenter |

Each directory holds per-case `.b2nd` arrays, a `manifest.csv`, and a `splits_final.json` defining
the folds. Slices are read, resampled and normalized on the fly by the data loaders, so **no nnU-Net
preprocessing step is run** and `nnUNet_preprocessed` is never populated.

`splits_final.json` is hand-built and read as given. It follows the MAMA-MIA folds — Duke, ISPY1,
ISPY2 and NACT — with all NACT cases and a subset of the others, each fold then extended with the
remaining timepoints of the same patients (mainly ISPY1 and ISPY2, since MAMA-MIA uses only the first
timepoint). Holding every timepoint of a patient in one fold keeps that patient off both sides of a
split.

Translation training also needs `training_pre_stats.json`, the dataset-wide intensity mean and
standard deviation used to normalize pre and post consistently across cases; it is written by
`compute_dataset_stats.py` — see
[the preprocessing README](nnunetv2/preprocessing/mamasynth/README.md).

The launcher passes all paths to the trainers through environment variables
(`MAMASYNTH_DATASET_DIR`, `MAMASYNTH_PRE_STATS_PATH`, `MAMASYNTH_SEGMENTATION_MODEL_DIR`,
`MAMASYNTH_FOREGROUND_DIR`, …), so start training through the entry points below rather than calling
a trainer directly.

> `--pre-stats` and `--seg-model-dir` fall back to `$MAMASYNTH_PRE_STATS_PATH` and
> `$MAMASYNTH_SEGMENTATION_MODEL_DIR`, so exporting those once lets you omit the flags. With neither
> the variable nor the flag given, the launcher stops and says which one is missing.

---

## Preprocessing

Documented in **[`nnunetv2/preprocessing/mamasynth/README.md`](nnunetv2/preprocessing/mamasynth/README.md)**:
the command for each of the three datasets, the input layouts they read, the `.b2nd` and vlmeta
contract they write, the per-cohort mask-to-case-ID mapping, and `training_pre_stats.json`.

---

## Training

Order matters: the segmenters are needed while the translation models train, and the MAE checkpoint
initialises them.

**1. MAE pretraining** — one run, reused by every translation model.

```bash
nnUNetv2_train_mamasynth_mae <Dataset626_MAE> 0 -tr nnUNetTrainerMamaSynthMAE_BS_64_ep_1000
```

**2. Segmentation models.**

```bash
# breast (label 2 merged with the lesion)
nnUNetv2_train_mamasynth <Dataset625_Pre_Seg> 0 -tr nnUNetTrainerMamaSynthBreast_BS_32

# lesion, from the pre-contrast image
nnUNetv2_train_mamasynth <Dataset625_Pre_Seg> <fold> \
    -tr nnUNetTrainerMamaSynthLesionTverskyA02B08_BS_64_epoch_1000

# tissue support, from its own dataset
nnUNetv2_train_mamasynth_foreground <Dataset627_foreground> 0 \
    -tr nnUNetTrainerMamaSynthForeground_BS_32
```

**3. Translation models**, initialised from the MAE checkpoint. The inside-breast model's objective
includes a term computed through the frozen lesion segmenter, so `--seg-model-dir` must point at the
model trained in step 2.

```bash
# outside-breast model
nnUNetv2_train_mamasynth_translate <Dataset625_Pre_Seg> <fold> \
    -tr nnUNetTrainerMamaSynthTranslationLPIPS_BS_48_MAEFinetune_ep_200 \
    -pretrained_weights <mae_checkpoint.pth> \
    --pre-stats <training_pre_stats.json>

# inside-breast model
nnUNetv2_train_mamasynth_translate <Dataset625_Pre_Seg> <fold> \
    -tr nnUNetTrainerMamaSynthTranslationLPIPSSSIMDiceTverskyA02B08_LPIPS05_BS_48_MAEFinetune \
    -pretrained_weights <mae_checkpoint.pth> \
    --pre-stats <training_pre_stats.json> \
    --seg-model-dir <segmentation_model_root>
```

Train folds 0–3 of the three ensembled slots to reproduce the submission. Results land under
`nnUNet_results/Dataset625_Pre_Seg/<trainer>__mamaSynthPlans__2d/fold_<n>/`.

Each run writes `progress_*.png` plots and a `translation_examples.png` preview
(input | reference | generated | both deltas | error) refreshed every epoch.

---

## Inference

Reproducing the submitted pipeline on a folder of `.mha` / `.tif` slices, against the published
weights (`W` is the `assets/` directory from [Weights](#weights)):

```bash
W=mama-synth-weights/assets

nnUNetv2_predict_mamasynth_translate_foreground_stitched_lesion_folds_synthfolds \
  -i  <input_dir> \
  -o  <output_dir> \
  --image-dir        $W/image        --image-folds        0 1 2 3 --image-checkpoint-name        checkpoint.pth \
  --lesion-image-dir $W/lesion_image --lesion-image-folds 0 1 2 3 --lesion-image-checkpoint-name checkpoint.pth \
  --lesion-seg-dir   $W/lesion_seg   --lesion-seg-folds   0 1 2 3 --lesion-seg-checkpoint-name   checkpoint.pth \
  --breast-weights     $W/breast/checkpoint.pth \
  --plans-breast       $W/breast/plans.json     --dataset-json-breast     $W/breast/dataset.json \
  --foreground-weights $W/foreground/checkpoint.pth \
  --plans-foreground   $W/foreground/plans.json --dataset-json-foreground $W/foreground/dataset.json \
  --feather 4 --lesion-feather 2 --lesion-gain 1.25 --lesion-threshold 0.5 \
  --dont-norm -device cuda
```

`--plans-breast` and `--plans-foreground` are needed only because those two slots are flat. A run
directory is otherwise inferred as the checkpoint's grandparent, so `image/fold_0/checkpoint.pth`
finds `image/plans.json` by itself; a bare `breast/checkpoint.pth` would look one level too high.

**Straight from a local training run** the same flags take the results directories instead, and the
`--plans-*` arguments can be dropped since `nnUNet_results/<dataset>/<trainer>__mamaSynthPlans__2d/`
already holds `plans.json` and `dataset.json` next to its `fold_<n>/`. Use the original checkpoint
names from the [table above](#models-it-loads) — `checkpoint_best.pth` for `--image-dir` and
`--lesion-seg-dir`, `checkpoint_best_dice.pth` for `--lesion-image-dir`, and
`fold_0/checkpoint_final.pth` for the two segmenters.

`--dont-norm` is correct when the inputs are already z-scored, as the challenge validation inputs
are; the output then stays in that same space. For raw inputs, drop it for per-slice normalization or
pass `--pre-stats` for dataset-wide statistics. The segmenters always z-score each slice internally,
independent of this flag.

`--save-masks`, `--save-intermediates` and `--save-lesion-mask` write the masks, the soft weights and
the two intermediate syntheses next to each output, which is the quickest way to see where a result
went wrong.

Simpler variants of the same pipeline are available as separate entry points, each dropping one
stage: `..._foreground_stitched_lesion_folds` (single-fold translations), `..._foreground_stitched_lesion`
(single fold throughout), `..._foreground_stitched` (no lesion step), `..._stitched` (no tissue
compositing) and `nnUNetv2_predict_mamasynth_translate` (one model, no compositing). Use them to
isolate the contribution of each stage.

---

## Evaluation

Scoring uses the challenge evaluation script from the MAMA-SYNTH challenge repository:

> **<https://github.com/mama-research/mama-synth/>**

`src/evaluation/evaluate.py` there is the Grand Challenge entry point. Run it in its own environment against a folder of predicted slices; it reports MSE, LPIPS, SSIM_tumor, FRD, Dice,
HD95 and the AUROC metrics.

---

## Repository map

Only these paths are specific to this work.

| Path | Contents |
|---|---|
| `nnunetv2/training/nnUNetTrainer/variants/network_architecture/nnUNetTrainerMamaSynth*.py` | all trainers: translation, MAE, breast, lesion, foreground, diffusion |
| `nnunetv2/preprocessing/mamasynth/` | dataset builders: source NIfTI to the three blosc2 datasets, plus pre-contrast statistics |
| `nnunetv2/training/mamasynth/` | blosc2 dataset readers, slice data loaders, plans |
| `nnunetv2/inference/predict_mamasynth_*.py` | prediction and compositing entry points |
| `nnunetv2/run/run_training_mamasynth.py` | training launchers; sets the environment the trainers read |

The main trainer file is large because it holds one class per configuration tried. The base classes
are `nnUNetTrainerMamaSynthTranslation` (masked pixel error) and its subclasses, which add perceptual,
structural, segmentation-based, feature-matching and adversarial terms; most other classes only
change weights, batch size or schedule.

---

## Contact

For questions, issues, or collaborations, please contact:

📧 benjamin.hamm@dkfz-heidelberg.de

 <img src="documentation/assets/dkfz_logo.png" height="70px" />
