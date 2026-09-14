# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Detect and classify harmful brain activity in electroencephalography (EEG) data: seizure (SZ), generalized periodic discharges (GPD), lateralized periodic discharges (LPD), lateralized rhythmic delta activity (LRDA), generalized rhythmic delta activity (GRDA), or "other".

## Metric
Kullback Liebler divergence between the predicted probability and the observed target.

## Submission Format
For each `eeg_id` in the test set, you must predict a probability for each of the `vote` columns. The file should contain a header and have the following format:

```
eeg_id,seizure_vote,lpd_vote,gpd_vote,lrda_vote,grda_vote,other_vote\
0,0.166,0.166,0.167,0.167,0.167,0.167\
1,0.166,0.166,0.167,0.167,0.167,0.167\
etc.
```

Your total predicted probabilities for each row must sum to one or your submission will fail.

## Dataset
**train.csv** Metadata for the train set. The expert annotators reviewed 50 second long EEG samples plus matched spectrograms covering 10 a minute window centered at the same time and labeled the central 10 seconds. Many of these samples overlapped and have been consolidated. `train.csv` provides the metadata that allows you to extract the original subsets that the raters annotated.

- `eeg_id` - A unique identifier for the entire EEG recording.
- `eeg_sub_id` - An ID for the specific 50 second long subsample this row's labels apply to.
- `eeg_label_offset_seconds` - The time between the beginning of the consolidated EEG and this subsample.
- `spectrogram_id` - A unique identifier for the entire EEG recording.
- `spectrogram_sub_id` - An ID for the specific 10 minute subsample this row's labels apply to.
- `spectogram_label_offset_seconds` - The time between the beginning of the consolidated spectrogram and this subsample.
- `label_id` - An ID for this set of labels.
- `patient_id` - An ID for the patient who donated the data.
- `expert_consensus` - The consensus annotator label. Provided for convenience only.
- `[seizure/lpd/gpd/lrda/grda/other]_vote` - The count of annotator votes for a given brain activity class. The full names of the activity classes are as follows: `lpd`: lateralized periodic discharges, `gpd`: generalized periodic discharges, `lrd`: lateralized rhythmic delta activity, and `grda`: generalized rhythmic delta activity . A detailed explanations of these patterns is [available here.](https://www.acns.org/UserFiles/file/ACNSStandardizedCriticalCareEEGTerminology_rev2021.pdf)

**test.csv** Metadata for the test set. As there are no overlapping samples in the test set, many columns in the train metadata don't apply.

- `eeg_id`
- `spectrogram_id`
- `patient_id`

**sample_submission.csv**

- `eeg_id`
- `[seizure/lpd/gpd/lrda/grda/other]_vote` - The target columns. Your predictions must be probabilities. Note that the test samples had between 3 and 20 annotators.

**train_eegs/** EEG data from one or more overlapping samples. Use the metadata in train.csv to select specific annotated subsets. The column names are [the names of the individual electrode locations for EEG leads](https://en.wikipedia.org/wiki/10%E2%80%9320_system_%28EEG%29), with one exception. The EKG column is for an electrocardiogram lead that records data from the heart. All of the EEG data (for both train and test) was collected at a frequency of 200 samples per second.

**test_eegs/** Exactly 50 seconds of EEG data.

train_spectrograms/ Spectrograms assembled EEG data. Use the metadata in train.csv to select specific annotated subsets. The column names indicate the frequency in hertz and the recording regions of the EEG electrodes. The latter are abbreviated as LL = left lateral; RL = right lateral; LP = left parasagittal; RP = right parasagittal.

**test_spectrograms/** Spectrograms assembled using exactly 10 minutes of EEG data.

**example_figures/** Larger copies of the example case images used on the overview tab.

# 2. Python version

3.12

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
librosa==0.11.0
mne==1.10.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sklearn-pandas==2.2.0
timm==1.0.19
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (166 lines)
            example_figures.zip (14.8 MB)
            sample_submission.csv (9851 lines)
            sample_submission.csv.zip (19.0 kB)
            test.csv (9851 lines)
            test.csv.zip (22.8 kB)
            test_eegs.zip (1.5 GB)
            test_spectrograms.zip (346.5 MB)
            train.csv (96951 lines)
            train.csv.zip (1.6 MB)
            train_eegs.zip (14.4 GB)
            train_spectrograms.zip (3.2 GB)
            example_figures/
                Sample01.pdf (914.3 kB)
                Sample02.pdf (703.4 kB)
                ... and 18 other files
            hms-harmful-brain-activity-classification/
                description.md (166 lines)
                example_figures.zip (14.8 MB)
                ... and 10 other files
                example_figures/
                    Sample01.pdf (914.3 kB)
                    Sample02.pdf (703.4 kB)
                    ... and 18 other files
                hms-harmful-brain-activity-classification/
                test_eegs/
                    1001717358.parquet (3.1 MB)
                    1003353736.parquet (972.5 kB)
                    ... and 1691 other files
                test_spectrograms/
                    1002209002.parquet (713.8 kB)
                    1005228554.parquet (648.5 kB)
                    ... and 1112 other files
                train_eegs/
                    1000913311.parquet (980.2 kB)
                    1001369401.parquet (1.2 MB)
                    ... and 15394 other files
                train_spectrograms/
                    1000086677.parquet (564.7 kB)
                    1000189855.parquet (672.7 kB)
                    ... and 10022 other files
            test_eegs/
                1001717358.parquet (3.1 MB)
                1003353736.parquet (972.5 kB)
                ... and 1691 other files
            test_spectrograms/
                1002209002.parquet (713.8 kB)
                1005228554.parquet (648.5 kB)
                ... and 1112 other files
            train_eegs/
                1000913311.parquet (980.2 kB)
                1001369401.parquet (1.2 MB)
                ... and 15394 other files
            train_spectrograms/
                1000086677.parquet (564.7 kB)
                1000189855.parquet (672.7 kB)
                ... and 10022 other files
        input/
            description.md (166 lines)
            example_figures.zip (14.8 MB)
            sample_submission.csv (9851 lines)
            sample_submission.csv.zip (19.0 kB)
            test.csv (9851 lines)
            test.csv.zip (22.8 kB)
            test_eegs.zip (1.5 GB)
            test_spectrograms.zip (346.5 MB)
            train.csv (96951 lines)
            train.csv.zip (1.6 MB)
            train_eegs.zip (14.4 GB)
            train_spectrograms.zip (3.2 GB)
            example_figures/
                Sample01.pdf (914.3 kB)
                Sample02.pdf (703.4 kB)
                ... and 18 other files
            hms-harmful-brain-activity-classification/
                description.md (166 lines)
                example_figures.zip (14.8 MB)
                ... and 10 other files
                example_figures/
                    Sample01.pdf (914.3 kB)
                    Sample02.pdf (703.4 kB)
                    ... and 18 other files
                hms-harmful-brain-activity-classification/
                test_eegs/
                    1001717358.parquet (3.1 MB)
                    1003353736.parquet (972.5 kB)
                    ... and 1691 other files
                test_spectrograms/
                    1002209002.parquet (713.8 kB)
                    1005228554.parquet (648.5 kB)
                    ... and 1112 other files
                train_eegs/
                    1000913311.parquet (980.2 kB)
                    1001369401.parquet (1.2 MB)
                    ... and 15394 other files
                train_spectrograms/
                    1000086677.parquet (564.7 kB)
                    1000189855.parquet (672.7 kB)
                    ... and 10022 other files
            test_eegs/
                1001717358.parquet (3.1 MB)
                1003353736.parquet (972.5 kB)
                ... and 1691 other files
            test_spectrograms/
                1002209002.parquet (713.8 kB)
                1005228554.parquet (648.5 kB)
                ... and 1112 other files
            train_eegs/
                1000913311.parquet (980.2 kB)
                1001369401.parquet (1.2 MB)
                ... and 15394 other files
            train_spectrograms/
                1000086677.parquet (564.7 kB)
                1000189855.parquet (672.7 kB)
                ... and 10022 other files
        working/
            hms-harmful-brain-activity-classification/
                description.md (166 lines)
                example_figures.zip (14.8 MB)
                ... and 10 other files
                example_figures/
                    Sample01.pdf (914.3 kB)
                    Sample02.pdf (703.4 kB)
                    ... and 18 other files
                hms-harmful-brain-activity-classification/
                test_eegs/
                    1001717358.parquet (3.1 MB)
                    1003353736.parquet (972.5 kB)
                    ... and 1691 other files
                test_spectrograms/
                    1002209002.parquet (713.8 kB)
                    1005228554.parquet (648.5 kB)
                    ... and 1112 other files
                train_eegs/
                    1000913311.parquet (980.2 kB)
                    1001369401.parquet (1.2 MB)
                    ... and 15394 other files
                train_spectrograms/
                    1000086677.parquet (564.7 kB)
                    1000189855.parquet (672.7 kB)
                    ... and 10022 other files
```

-> data/hms-harmful-brain-activity-classification/sample_submission.csv has 9850 rows and 7 columns.
The columns are: eeg_id, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote

-> data/hms-harmful-brain-activity-classification/test.csv has 9850 rows and 3 columns.
The columns are: spectrogram_id, eeg_id, patient_id

-> data/hms-harmful-brain-activity-classification/train.csv has 96950 rows and 15 columns.
The columns are: eeg_id, eeg_sub_id, eeg_label_offset_seconds, spectrogram_id, spectrogram_sub_id, spectrogram_label_offset_seconds, label_id, patient_id, expert_consensus, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote

-> data/sample_submission.csv has 9850 rows and 7 columns.
The columns are: eeg_id, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote

-> data/test.csv has 9850 rows and 3 columns.
The columns are: spectrogram_id, eeg_id, patient_id

-> data/train.csv has 96950 rows and 15 columns.
The columns are: eeg_id, eeg_sub_id, eeg_label_offset_seconds, spectrogram_id, spectrogram_sub_id, spectrogram_label_offset_seconds, label_id, patient_id, expert_consensus, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote

-> (stopped after 10 files for performance)

# 5. Target score

0.4611364180524799

# 6. Current score

1.39779

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I first fix the runtime blocker by making the weight loading robust: if the expected Kaggle input folders/files don’t exist, the code fall back to a valid uniform-probability submission rather than crashing. I also fix a logic bug in `brain_lead` (RP was duplicated instead of using RR) and remove references to undefined variables during inference-time EEG slicing. Finally, I guarantee the submission has the correct 6 target columns, correct row count/order, and probabilities that sum to 1.0 per row, so a valid `submission.csv` is always produced end-to-end.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower is better) is far from the target (0.4611), and the most likely reason is that you are effectively submitting near-uniform probabilities because the external weight files in `/kaggle/input/hms-baseline` and `/kaggle/input/hms-eeg` are not available in your environment. The minimal, score-relevant fix is to load and ensemble the provided competition baseline weights *only if they exist*, and otherwise keep your safe uniform fallback. I also ensure inference uses `autocast` on CUDA to fit memory (without changing model logic) and add a tiny numerical stabilization by normalizing after averaging folds (already present) and guaranteeing float64 normalization for KL safety. These changes preserve your architecture/inference semantics while making it actually use learned weights when present, which should move KL much closer to the target band.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower is better) is far above the target (0.4611), and the most likely cause is that you’re averaging probabilities from mismatched/unavailable checkpoints: the code tries to load weights that either don’t exist or don’t match the defined model heads, so it effectively falls back to weak/uniform-like predictions. I make weight loading robust to common Kaggle checkpoint formats (`state_dict`, `model`, `net`) and ensure we only ensemble checkpoints that actually load cleanly into the model, otherwise skip them. I also keep inference semantics identical but add a numerically-stable “logit-ensemble” (average logits then softmax) which is typically better for KL than averaging already-softmaxed probabilities, without changing architecture or training. Finally, I keep the same submission schema/paths and still guarantee a valid CSV even if no checkpoints are usable.'
- What this solution (achieved 1.40995) has done: 'Your KL (1.40995, lower is better) is far above the target (0.4611), and the most likely cause is that the notebook isn’t actually using any good checkpoints (most weight paths don’t exist), so it falls back to near-uniform predictions. I make the code discover and load any `.pth` checkpoints that are actually present inside the provided competition dataset folder, instead of relying on hardcoded non-existent `/kaggle/input/hms-baseline` and `/kaggle/input/hms-eeg` paths. To better match typical training/inference for these EfficientNet checkpoints without changing the architecture, I also apply the standard EfficientNet input normalization for spectrogram images (and keep EEG path untouched), which usually improves calibration and KL. Finally, I keep your current robust logit-ensembling + softmax and still guarantee a valid submission CSV even if no checkpoints are found.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995; lower is better) is much worse than the target (0.4611), and the main score issue is that the code is likely not actually using any meaningful checkpoints and/or is discovering irrelevant `.pth` files (including ones that don’t match your defined heads), leading to near-uniform predictions. I make checkpoint discovery/load stricter by only accepting files whose names match the expected baseline pattern (fold/epoch/val_loss) and by requiring a sane number of loaded keys, so you don’t ensemble random incompatible weights. I also fix a spectrogram slicing bug (`r` never increments) so the 4 region images are not identical, which should materially improve predictions without changing the model or training logic. Finally, I keep your existing logit-ensembling + row-normalization and still guarantee a valid `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995; lower is better) is far above the target (0.4611), and the most likely reason is that you’re still effectively not using meaningful checkpoints (none exist in this dataset) and/or your inference-time spectrogram region slicing is not matching the common baseline layout. To move the score down toward the target with minimal logic change, I (1) extend checkpoint discovery to also accept the common `best*.pth` / `*fold*.pth` naming patterns and search `/kaggle/input` so any provided checkpoints are actually found and loaded, and (2) fix the spectrogram crop to use the correct frequency axis (rows) with a single consistent 300-row window while keeping the 4 region columns as-is (this corrects a silent feature bug without changing the model). I also keep your existing logit-ensembling + softmax + row-normalization (good for KL) and still guarantee a valid uniform fallback submission if no checkpoints are available.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995; lower is better) is far above the target (0.4611), and the most likely reason is that you’re still effectively making near-uniform predictions because no usable checkpoints are actually being found/loaded. I make the checkpoint discovery explicitly look inside the provided dataset folder for common checkpoint names and also accept `.pt/.bin` (not only `.pth`), then load only those that pass your existing key-match sanity checks. This is a minimal, score-relevant change that preserves your model/inference logic but increases the chance you actually use learned weights (which should reduce KL toward the target). I also make DataLoader more robust by setting `persistent_workers` only when workers>0 and adding `torch.inference_mode()` (same semantics as no_grad, faster/safer) without changing outputs.'
- What this solution (achieved 1.40995) has done: 'Your KL is far above target because you’re still effectively not using any trained checkpoints in this environment (so predictions are close to uniform), and additionally the spectrogram input pipeline is mismatched to the EfficientNet-B5 `in_chans=3` model (you currently produce 4 channels then slice/re-pack in a way that silently drops one region). I make checkpoint discovery actually find usable fold checkpoints inside the provided dataset by allowing broader filename patterns but still validating load success, so you ensemble real weights when present. I also fix the spectrogram tensor to be true 3-channel as expected by `NetSpec` by stacking 3 region images (keeping the same crop/log steps), which should materially reduce KL without changing the model architecture/training loop. Finally, I keep your existing logit-ensemble + softmax + strict row-normalization to preserve submission validity and KL stability.'
- What this solution (achieved 1.40694) has done: 'Your current KL (1.40995, lower is better) is far above the target (0.4611), and the code is still very likely producing weak/uniform-like predictions because it isn’t actually using any trained weights from this environment. I make checkpoint discovery actually locate and use real HMS baseline fold checkpoints by downloading the public baseline weights directly from `timm` (the same backbone you already use) when no external `.pth` are present, without changing your model architecture or inference loop. I also fix a silent spectrogram-axis mixup (your crop is on the wrong axis), which materially hurts the spectrogram features while preserving the same “3 region channels + log + clip” core logic. Finally, I keep your existing logit-ensembling + softmax + strict row-normalization so the submission remains valid and KL-stable.'
- What this solution (achieved 1.39779) has done: 'Your current KL (1.40694; lower is better) is far above the target (0.4611), and the biggest likely issue is that when no trained competition checkpoints are present you fall back to a random classification head (even if the EfficientNet backbone is pretrained), which tends to produce poorly calibrated probabilities. I keep your exact model architectures and inference flow, but replace the “pretrained backbone + random head” fallback with a safer prior: class probabilities equal to the training label distribution (computed from train.csv), which is typically much closer to the public/private distribution than uniform/random and should reduce KL. I also slightly strengthen numerical safety by always clipping + renormalizing probabilities after softmax/averaging (same semantics, just avoids rare NaNs/zeros). Everything still runs end-to-end and always writes a valid `submission.csv`.'
- What this solution (achieved 1.39779) has done: 'Your current KL (1.39779, lower is better) is still far from the target (0.4611), so we should make a small, score-relevant improvement without changing your model architectures or inference flow. The biggest likely issue is that the test EEG/spectrogram parquet files are stored in multiple possible Kaggle input roots, and your dataset hard-codes only one path; if that path is missing in your runtime, inference crash or silently force fallback behavior (train prior) for a bad score. I make the dataset resolve file paths robustly (try both `/kaggle/input/...` and `/kaggle/data/...` competition roots, plus the nested folder), and I ensure we only use train-prior fallback when *no* model predictions were produced, not because of avoidable file-path misses. This keeps core logic identical (same preprocessing, models, checkpoint loading, ensembling), but should materially reduce KL if your environment actually contains the baseline checkpoints and parquets under a different root.'
- What this solution (achieved 1.39779) has done: 'Your current KL (1.39779; lower is better) is still far above the target, and the most likely remaining issue is that inference never finds/loads real competition checkpoints, so you’re effectively submitting the (weak) train-prior for all rows. I make one minimal, score-relevant improvement: broaden checkpoint discovery to also look for common baseline weight filenames that don’t include “val_loss” (e.g., `fold0.pth`, `best_fold0.pth`, etc.), and classify EEG-vs-SPEC checkpoints using both filename and parent folder names to avoid misrouting. This preserves your model architectures and inference/ensemble semantics, but increases the chance that real pretrained fold checkpoints present in the dataset are actually loaded, which should reduce KL toward the target. Submission formatting and probability normalization remain unchanged and still guarantee a valid `submission.csv`.'
- What this solution (achieved 1.39779) has done: 'I make one score-relevant fix in your spectrogram preprocessing: the HMS spectrogram parquet columns are grouped by 4 regions (LL/RL/LP/RP) per frequency bin, but your current code slices `region*100:(region+1)*100`, which mis-aligns regions and discards 3/4 of the information. I change the crop to correctly pick each region by striding every 4th column (region index 0..2 kept to preserve your `in_chans=3` model), keeping the same log/clip/transpose/normalization core logic. This should meaningfully reduce KL (toward your 0.461 target) without changing model architecture, training loops, or loss. Everything else (checkpoint discovery/loading, logit-ensembling, softmax, row-normalization, submission writing) stays the same.'

# 9. Code solution

## === cell 0
import random
import cv2
import json
import numpy as np
import copy
import pandas as pd
import torch
import gc

import albumentations as A
import os
import re
import librosa
import pickle
import timm
from tqdm import tqdm
import mne

import torchaudio
import torch.nn as nn
from torch.utils.data import DataLoader



## === cell 1
CFG = {
    "batch_size": 32,
    "num_worker": 4,
    "data": "/kaggle/input/hms-harmful-brain-activity-classification/test.csv",
    "train_csv": "/kaggle/input/hms-harmful-brain-activity-classification/train.csv",
    "weights_spec": [
        "/kaggle/input/hms-baseline/fold0_epoch_4_val_loss_0.549755.pth",
        "/kaggle/input/hms-baseline/fold1_epoch_4_val_loss_0.524753.pth",
        "/kaggle/input/hms-baseline/fold2_epoch_4_val_loss_0.511551.pth",
        "/kaggle/input/hms-baseline/fold3_epoch_4_val_loss_0.548698.pth",
        "/kaggle/input/hms-baseline/fold4_epoch_3_val_loss_0.666013.pth",
        "/kaggle/input/hms-harmful-brain-activity-classification/fold0_epoch_4_val_loss_0.549755.pth",
        "/kaggle/input/hms-harmful-brain-activity-classification/fold1_epoch_4_val_loss_0.524753.pth",
        "/kaggle/input/hms-harmful-brain-activity-classification/fold2_epoch_4_val_loss_0.511551.pth",
        "/kaggle/input/hms-harmful-brain-activity-classification/fold3_epoch_4_val_loss_0.548698.pth",
        "/kaggle/input/hms-harmful-brain-activity-classification/fold4_epoch_3_val_loss_0.666013.pth",
    ],
    "weights_eeg": [
        "/kaggle/input/hms-eeg/fold0_epoch_4_val_loss_0.584720.pth",
        "/kaggle/input/hms-eeg/fold1_epoch_4_val_loss_0.590993.pth",
        "/kaggle/input/hms-eeg/fold2_epoch_4_val_loss_0.603758.pth",
        "/kaggle/input/hms-eeg/fold3_epoch_4_val_loss_0.580059.pth",
        "/kaggle/input/hms-eeg/fold4_epoch_4_val_loss_0.545819.pth",
        "/kaggle/input/hms-harmful-brain-activity-classification/fold0_epoch_4_val_loss_0.584720.pth",
        "/kaggle/input/hms-harmful-brain-activity-classification/fold1_epoch_4_val_loss_0.590993.pth",
        "/kaggle/input/hms-harmful-brain-activity-classification/fold2_epoch_4_val_loss_0.603758.pth",
        "/kaggle/input/hms-harmful-brain-activity-classification/fold3_epoch_4_val_loss_0.580059.pth",
        "/kaggle/input/hms-harmful-brain-activity-classification/fold4_epoch_4_val_loss_0.545819.pth",
    ],
    "flip": True,
}

TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]


def _existing_weight_paths(paths):
    return [p for p in paths if isinstance(p, str) and os.path.exists(p)]


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for k in ("state_dict", "model", "net", "model_state_dict", "weights"):
            v = ckpt_obj.get(k, None)
            if isinstance(v, dict):
                ckpt_obj = v
                break
    if not isinstance(ckpt_obj, dict):
        raise TypeError("Checkpoint is not a state_dict/dict-like object.")
    cleaned = {}
    for k, v in ckpt_obj.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        cleaned[nk] = v
    return cleaned


def _safe_load_weights(model, weight_path, device):
    """
    Keep: robust weight loading across common checkpoint wrappers.
    """
    try:
        ckpt = torch.load(weight_path, map_location=device)
        sd = _extract_state_dict(ckpt)

        model_keys = set(model.state_dict().keys())
        sd_keys = set(sd.keys())
        intersect = len(model_keys & sd_keys)

        if intersect < 0.25 * len(model_keys):
            return False, f"Too few matching keys ({intersect}/{len(model_keys)})"

        missing, unexpected = model.load_state_dict(sd, strict=False)

        if len(missing) > 0.75 * len(model_keys):
            return False, f"Too many missing keys ({len(missing)}/{len(model_keys)})"
        return True, ""
    except Exception as e:
        return False, str(e)


def _discover_ckpt_weights(search_roots):
    """
    Keep: broad discovery. This is only useful if there are actual checkpoints present.
    """
    exts = (".pth", ".pt", ".bin")
    found = []
    for root in search_roots:
        if not root or not os.path.exists(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith(exts):
                    found.append(os.path.join(dirpath, fn))
    return sorted(set(found))


def _is_expected_baseline_ckpt(path):
    """
    Keep: accept additional common HMS baseline checkpoint names.
    """
    bn = os.path.basename(path).lower()
    if re.search(
        r"fold\d+_epoch_?\d+_val[_\-]?loss[_\-]?[0-9.]+(\.pth|\.pt|\.bin)$", bn
    ):
        return True
    if re.search(r"(best|final|last|checkpoint)([_\-].*)?(\.pth|\.pt|\.bin)$", bn):
        return True
    if re.search(r"(fold|kf|split)[_-]?\d+.*(\.pth|\.pt|\.bin)$", bn):
        return True
    if ("efficientnet" in bn or "effnet" in bn) and bn.endswith(
        (".pth", ".pt", ".bin")
    ):
        return True
    if re.fullmatch(r"fold\d+(\.pth|\.pt|\.bin)$", bn):
        return True
    if re.fullmatch(r"(best|final|last)[_-]?fold\d+(\.pth|\.pt|\.bin)$", bn):
        return True
    return False


def _compute_train_prior(train_csv_path, targets, eps=1e-12):
    try:
        if not (isinstance(train_csv_path, str) and os.path.exists(train_csv_path)):
            raise FileNotFoundError(train_csv_path)
        df = pd.read_csv(train_csv_path, usecols=targets)
        votes = df[targets].to_numpy(dtype=np.float64)
        row_sums = votes.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums <= 0, 1.0, row_sums)
        probs = votes / row_sums
        prior = probs.mean(axis=0)
        prior = np.clip(prior, eps, None)
        prior = prior / prior.sum()
        return prior.astype(np.float32), True
    except Exception:
        prior = np.full((len(targets),), 1.0 / len(targets), dtype=np.float32)
        return prior, False


def _resolve_existing_path(candidates):
    for p in candidates:
        if isinstance(p, str) and os.path.exists(p):
            return p
    return None


def _competition_roots():
    return [
        "/kaggle/input/hms-harmful-brain-activity-classification",
        "/kaggle/data/hms-harmful-brain-activity-classification",
        "/kaggle/input/hms-harmful-brain-activity-classification/hms-harmful-brain-activity-classification",
        "/kaggle/data/hms-harmful-brain-activity-classification/hms-harmful-brain-activity-classification",
    ]


def _resolve_eeg_parquet(eeg_id):
    cands = [f"{root}/test_eegs/{eeg_id}.parquet" for root in _competition_roots()]
    p = _resolve_existing_path(cands)
    if p is None:
        raise FileNotFoundError(
            f"EEG parquet not found for eeg_id={eeg_id}. Tried: {cands[:2]} ..."
        )
    return p


def _resolve_spec_parquet(spec_id):
    cands = [
        f"{root}/test_spectrograms/{spec_id}.parquet" for root in _competition_roots()
    ]
    p = _resolve_existing_path(cands)
    if p is None:
        raise FileNotFoundError(
            f"Spectrogram parquet not found for spectrogram_id={spec_id}. Tried: {cands[:2]} ..."
        )
    return p


def _looks_like_eeg_ckpt(path: str) -> bool:
    """
    Keep: improve EEG-vs-SPEC checkpoint routing.
    """
    p = path.lower()
    bn = os.path.basename(p)
    if "eeg" in bn or "wave" in bn:
        return True
    if "/eeg" in p or "eeg_" in p or "_eeg" in p:
        return True
    if "neteeg" in bn or "transform" in bn:
        return True
    return False




## === cell 2
class AlaskaDataIter:
    def __init__(self, df, training_flag=False, shuffle=False, use_eeg=False):
        self.training_flag = training_flag
        self.shuffle = shuffle
        self.raw_data_set_size = None
        self.df = df

        self.train_trans = A.Compose([A.HorizontalFlip(p=0.5)])

        TARS = {"Seizure": 0, "LPD": 1, "GPD": 2, "LRDA": 3, "GRDA": 4, "Other": 5}
        self.TARS2 = {x: y for y, x in TARS.items()}

        self.eeg_nms = [
            "Fp1",
            "F3",
            "C3",
            "P3",
            "F7",
            "T3",
            "T5",
            "O1",
            "Fz",
            "Cz",
            "Pz",
            "Fp2",
            "F4",
            "C4",
            "P4",
            "F8",
            "T4",
            "T6",
            "O2",
            "EKG",
        ]

        self.LL = ["Fp1", "F7", "T3", "T5", "O1"]
        self.RR = ["Fp2", "F8", "T4", "T6", "O2"]
        self.LP = ["Fp1", "F3", "C3", "P3", "O1"]
        self.RP = ["Fp2", "F4", "C4", "P4", "O2"]

        self.leads_dict = {value: index for index, value in enumerate(self.eeg_nms)}
        self.use_eeg = use_eeg

    def __getitem__(self, item):
        return self.single_map_func(self.df.iloc[item], self.training_flag)

    def __len__(self):
        return len(self.df)

    def brain_lead(self, waves):
        waves = copy.deepcopy(waves)
        brain_leads = [self.LL, self.LP, self.RP, self.RR]

        leads = []
        for combine in brain_leads:
            for i in range(len(combine) - 1):
                tmp_lead = (
                    waves[self.leads_dict[combine[i]]]
                    - waves[self.leads_dict[combine[i + 1]]]
                )
                leads.append(tmp_lead)

        data = np.concatenate([leads], axis=0)
        return data

    def _normalize_for_efficientnet(self, x_chw):
        x = x_chw.astype(np.float32, copy=False)
        mn = np.min(x)
        mx = np.max(x)
        if mx > mn:
            x = (x - mn) / (mx - mn)
        else:
            x = np.zeros_like(x, dtype=np.float32)
        return x

    def single_map_func(self, dp, is_training):
        if self.use_eeg:
            eeg_path = _resolve_eeg_parquet(int(dp["eeg_id"]))
            eeg = pd.read_parquet(eeg_path)

            offset = 0
            eeg = eeg.iloc[offset * 200 : (offset + 50) * 200]

            waves = eeg.values
            waves = np.transpose(waves, axes=[1, 0])

            for i in range(waves.shape[0]):
                m = np.nanmean(waves[i])
                if np.isnan(waves[i]).mean() < 1:
                    waves[i] = np.nan_to_num(waves[i], nan=m)
                else:
                    waves[i] = 0

            waves = np.array(waves, dtype=np.float64)
            waves = mne.filter.filter_data(waves, 200, 0, 20, verbose=False)
            waves = self.brain_lead(waves)
            data = waves
            return data.astype(np.float32)
        else:
            spec_path = _resolve_spec_parquet(int(dp["spectrogram_id"]))
            spec = pd.read_parquet(spec_path)
            spec = spec.values[:, 1:]  # drop the first (time) column

            images = []
            for region in range(3):  # keep 3 channels to match NetSpec(in_chans=3)
                img = spec[:300, region::4]  # (300, F) region-specific columns
                img = np.clip(img, np.exp(-4), np.exp(8))
                img = np.log(img)
                img = np.nan_to_num(img, nan=0.0)
                img = img.T  # (F, 300)
                images.append(img)

            images = np.stack(images, -1)  # (F,300,3)
            data = np.transpose(images, [2, 0, 1]).astype(np.float32)  # (3,F,300)
            data = self._normalize_for_efficientnet(data)
            return data.astype(np.float32)




## === cell 3
class NetSpec(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.model = timm.create_model("efficientnet_b5", pretrained=False, in_chans=3)
        self.fc = nn.Linear(2048, 6, bias=True)
        self.dropout = nn.Dropout(0.5)
        self.avg = nn.AdaptiveAvgPool2d(1)

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x):
        bs = x.size(0)
        x = (x - self.mean) / self.std
        x = self.model.forward_features(x)
        x = self.avg(x)
        x = x.view(bs, -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x




## === cell 4
class Transform(nn.Module):
    def __init__(self):
        super().__init__()
        self.wave_transform = torchaudio.transforms.Spectrogram(
            n_fft=1024, hop_length=50, power=1
        )
        self.am2db = torchaudio.transforms.AmplitudeToDB(stype="magnitude", top_db=80)
        self.resizer = nn.UpsamplingBilinear2d(size=[160, 320])

    def forward(self, x):
        image = self.wave_transform(x)
        image = self.am2db(image)
        n, c, h, w = image.size()
        image = image[:, :, : int(20 / 100 * h + 30), :]
        image = torch.reshape(image, shape=[n, 4, -1, w])
        return image


class NetEeg(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.preprocess = Transform()
        self.model = timm.create_model("efficientnet_b5", pretrained=False, in_chans=4)
        self.fc = nn.Linear(2048, 6, bias=True)
        self.dropout = nn.Dropout(0.5)
        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        bs = x.size(0)
        x = self.preprocess(x)
        x = self.model.forward_features(x)
        x = self.avg(x)
        x = x.view(bs, -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x




## === cell 5
def inference_function_logits(test_loader, model, device):
    model.eval()
    logits = []
    use_amp = device.type == "cuda"
    with torch.inference_mode():
        with tqdm(test_loader, unit="test_batch", desc="Inference") as tqdm_test_loader:
            for X in tqdm_test_loader:
                X = X.to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    y = model(X)
                logits.append(y.to("cpu").float().numpy())
    return {"logits": np.concatenate(logits, axis=0)}


def _row_normalize(p, eps=1e-12):
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, None)
    s = p.sum(axis=1, keepdims=True)
    s = np.where(s <= 0, 1.0, s)
    p = p / s
    return p.astype(np.float32)


def _softmax_np(x, eps=1e-12):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(x)
    ex = np.clip(ex, eps, None)
    return _row_normalize(ex, eps=eps)




## === cell 6
test_df = pd.read_csv(CFG["data"])
test_df.head(5)



## === cell 7
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

eeg_weight_paths = _existing_weight_paths(CFG["weights_eeg"])
spec_weight_paths = _existing_weight_paths(CFG["weights_spec"])

if len(eeg_weight_paths) == 0 and len(spec_weight_paths) == 0:
    discovered = _discover_ckpt_weights(
        [
            "/kaggle/input/hms-harmful-brain-activity-classification",
            "/kaggle/input",
            "/kaggle/data/hms-harmful-brain-activity-classification",
            "/kaggle/data",
        ]
    )
    discovered = [p for p in discovered if _is_expected_baseline_ckpt(p)]

    disc_eeg = [p for p in discovered if _looks_like_eeg_ckpt(p)]
    disc_spec = [p for p in discovered if p not in set(disc_eeg)]
    eeg_weight_paths = disc_eeg
    spec_weight_paths = disc_spec

predictions = None
all_source_logits = []

loaded_eeg = 0
loaded_spec = 0

_loader_kwargs = dict(
    batch_size=CFG["batch_size"],
    num_workers=CFG["num_worker"],
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
    drop_last=False,
)
if CFG["num_worker"] > 0:
    _loader_kwargs["persistent_workers"] = True

fallback_used = {"train_prior": False}

train_prior, train_prior_ok = _compute_train_prior(CFG.get("train_csv", ""), TARGETS)

if len(eeg_weight_paths) > 0:
    fold_logits = []
    test_dataset = AlaskaDataIter(
        test_df, training_flag=False, shuffle=False, use_eeg=True
    )
    test_loader = DataLoader(test_dataset, **_loader_kwargs)

    for model_weight in eeg_weight_paths:
        model = NetEeg()
        ok, msg = _safe_load_weights(model, model_weight, device)
        if not ok:
            del model
            gc.collect()
            continue

        model.to(device)
        prediction_dict = inference_function_logits(test_loader, model, device)
        fold_logits.append(prediction_dict["logits"])
        loaded_eeg += 1

        del model, prediction_dict
        torch.cuda.empty_cache()
        gc.collect()

    if len(fold_logits) > 0:
        eeg_logits = np.mean(np.stack(fold_logits, axis=0), axis=0)
        all_source_logits.append(eeg_logits)

if len(spec_weight_paths) > 0:
    fold_logits = []
    test_dataset = AlaskaDataIter(
        test_df, training_flag=False, shuffle=False, use_eeg=False
    )
    test_loader = DataLoader(test_dataset, **_loader_kwargs)

    for model_weight in spec_weight_paths:
        model = NetSpec()
        ok, msg = _safe_load_weights(model, model_weight, device)
        if not ok:
            del model
            gc.collect()
            continue

        model.to(device)
        prediction_dict = inference_function_logits(test_loader, model, device)
        fold_logits.append(prediction_dict["logits"])
        loaded_spec += 1

        del model, prediction_dict
        torch.cuda.empty_cache()
        gc.collect()

    if len(fold_logits) > 0:
        spec_logits = np.mean(np.stack(fold_logits, axis=0), axis=0)
        all_source_logits.append(spec_logits)

if len(all_source_logits) > 0:
    logits = np.mean(np.stack(all_source_logits, axis=0), axis=0)
    predictions = _softmax_np(logits).astype(np.float32)
else:
    n = len(test_df)
    predictions = np.tile(train_prior.reshape(1, -1), (n, 1)).astype(np.float32)
    fallback_used["train_prior"] = True

if (
    predictions.ndim != 2
    or predictions.shape[0] != len(test_df)
    or predictions.shape[1] != len(TARGETS)
):
    n = len(test_df)
    predictions = np.tile(train_prior.reshape(1, -1), (n, 1)).astype(np.float32)
    fallback_used["train_prior"] = True

predictions = _row_normalize(predictions)



## === cell 8
sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
sub[TARGETS] = predictions
sub.to_csv("submission.csv", index=False)
print(f"Submission shape: {sub.shape}")
print(sub.head())
print(
    f"Row sums (min/mean/max): {sub[TARGETS].sum(1).min():.6f} / {sub[TARGETS].sum(1).mean():.6f} / {sub[TARGETS].sum(1).max():.6f}"
)
print(
    f"Found EEG weights: {len(eeg_weight_paths)} | Found SPEC weights: {len(spec_weight_paths)} | Loaded EEG: {loaded_eeg} | Loaded SPEC: {loaded_spec}"
)
print(
    f"Fallback used (train_prior): {fallback_used} | train_prior_ok: {train_prior_ok}"
)
print("Example weight paths (first 5 EEG/spec):")
print("EEG:", eeg_weight_paths[:5])
print("SPEC:", spec_weight_paths[:5])
print("Train prior:", dict(zip(TARGETS, train_prior.tolist())))
