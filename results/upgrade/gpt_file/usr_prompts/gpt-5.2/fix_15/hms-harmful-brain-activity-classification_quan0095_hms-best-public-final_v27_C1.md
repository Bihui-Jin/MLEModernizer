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

No external packages required in the script and installed.

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

0.2854517255666261

# 6. Current score

1.4093

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the hard runtime blockers so the notebook can run in a CPU-only Kaggle environment: (1) remove the unconditional CUDA usage by selecting `cuda` only if available, (2) make weight loading device-agnostic via `map_location`, and (3) eliminate dependencies on missing external model-weight datasets by falling back to a valid uniform-probability submission if those `/kaggle/input/...` paths are not present. I also fix a latent bug in `stft_spec_from_eeg` where `signal` was used before import, and remove extremely verbose per-sample printing that would otherwise time out. Finally, I ensure the produced `submission.csv` has the exact required columns, includes all `eeg_id` from `test.csv`, and each row sums to 1.'
- What this solution (achieved 1.40995) has done: 'Your current score is far worse than the target (lower is better), and the main reason is that you’re effectively submitting near-uniform probabilities because the intended pretrained weights are missing, so inference is skipped. The smallest change that legitimately moves the score toward the target is to ensure the notebook can actually load model weights by pointing to a weights directory that exists in this dataset (or, if none exist, keep the uniform fallback). I add an automatic search for `.pth` files under `/kaggle/input/` and load those that match your `Net` architecture, while keeping the same model, preprocessing, and inference logic. I also make sure we only spend time generating features if we actually run inference (otherwise we keep the fast uniform submission path).'
- What this solution (achieved 1.40995) has done: 'Your current score is much worse than the target (lower is better), and the main cause is that you’re still often falling back to near-uniform predictions because no compatible pretrained `.pth` weights are actually being found/loaded. I make two minimal, score-relevant fixes: (1) broaden and harden weight discovery to search for *any* `.pth` under `/kaggle/input` and then only keep those that successfully `load_state_dict`, and (2) ensure `eeg_id` is consistently treated as a string everywhere so dictionary lookups don’t silently miss and revert to uniform. This preserves your exact model/preprocessing/inference logic; it only increases the chance you actually run inference with real weights (which should move KL down toward the target). The rest of the pipeline (feature extraction, model forward, softmax, submission normalization) remains unchanged.'
- What this solution (achieved 1.40995) has done: 'Your current score is far worse than the target because you’re still effectively producing near-uniform predictions whenever compatible weights aren’t found/loaded, so the smallest reliable way to move KL down is to (1) make weight loading succeed more often for common checkpoint formats (raw `state_dict`, `{"state_dict":...}`, `{"model":...}`) and (2) ensure we don’t accidentally reject compatible checkpoints due to prefixes like `module.`. I keep your exact model architecture and inference flow, but add a minimal “robust state_dict extraction + prefix stripping” loader and broaden discovery to include `.pt` as well as `.pth`, then actually use every successfully loaded checkpoint (up to 5) for your existing ensemble. This should materially reduce the gap toward the target without changing evaluation semantics or training approach (there is still no training here). The submission formatting and probability normalization remain unchanged.'
- What this solution (achieved 1.40995) has done: 'Your score is far worse than the target (lower is better), which strongly suggests the submission is still mostly near-uniform due to missing/failed checkpoint loading or missing predictions for many `eeg_id`. I make the smallest changes that increase the chance of producing non-uniform predictions: (1) harden checkpoint loading to correctly handle common key prefixes and DDP wrappers (including nested prefixes) and ignore non-tensor entries, and (2) ensure `eeg_id` keys coming from the DataLoader are decoded reliably (they can arrive as tensors/arrays), preventing silent dictionary misses that revert to uniform. These changes preserve your exact model architecture, preprocessing, and inference logic; they only reduce accidental uniform fallbacks and should move KL substantially down toward the target. The submission writing and probability normalization remain unchanged and still guarantee valid rows summing to 1.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower is better) is far from the target (0.28545), and the most likely reason is still that inference is effectively failing (no compatible weights loaded) and/or many `eeg_id` predictions are silently missing and replaced by uniform probabilities. I keep your exact model + feature pipeline, but make two minimal, score-relevant fixes: (1) ensure `result_7` keys *exactly* match the `eeg_id` strings from `sample_submission.csv` by normalizing IDs consistently and avoiding odd tensor/list stringification, and (2) broaden checkpoint discovery/loading slightly to include `.bin` and handle `ema_state_dict` plus `module.model.`-style nested prefixes so more real checkpoints load successfully. These changes should increase the fraction of non-uniform predictions and move KL down toward the target without changing architecture, preprocessing, or inference semantics. The submission format and per-row normalization remain unchanged to guarantee validity.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower is better) is far worse than the target (0.28545), and the most likely cause is still that you end up with mostly-uniform predictions because no compatible checkpoints are actually being loaded (so `result_7` is empty or sparse). I make two minimal, score-relevant changes: (1) explicitly search for weights not only in `/kaggle/input` but also in common working locations, and prefer HMS-related filenames to increase the chance of loading something compatible; and (2) add a strict coverage check so we only run inference when features exist and, after inference, we warn if too many `eeg_id` are missing (helping detect silent mismatches) while keeping the same fallback behavior. This preserves your model architecture, preprocessing, and inference logic; it only increases the chance you actually use real weights for most test rows, which should move KL down toward the target. The submission formatting and per-row probability normalization remain unchanged to guarantee a valid CSV.'
- What this solution (achieved 1.40995) has done: 'Your KL is far worse than the target, so we should improve (decrease) it with minimal, score-relevant changes. Right now inference is likely running with partially/incorrectly loaded checkpoints (because `strict=False` can silently accept mismatched heads/backbones) and/or with checkpoints that don’t actually correspond to your `vit_small_patch14_reg4_dinov2...` architecture, yielding near-random predictions close to uniform. I keep the exact same model and feature pipeline, but make weight loading *stricter*: only accept checkpoints that (a) contain enough matching tensor keys and (b) have the expected `head.*` shapes, otherwise skip them and fall back (uniform) rather than using bad weights. Additionally, I stop importing `librosa` (unused) to avoid potential environment issues/time, and I make submission-row alignment safer by always keying predictions by the `eeg_id` list from `sample_submission.csv`.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower is better) is far from the target (0.28545), and the main score driver here is that many/most rows are still effectively getting uniform predictions due to missing or unusable checkpoints. I make weight loading more reliable without changing your model or feature pipeline by (1) enabling safe CPU inference (so you can actually run the ViT on Kaggle CPU if no GPU is available) and (2) accepting compatible checkpoints only when they match the head shapes while also handling nested key prefixes more robustly. Additionally, I ensure `spec_arr` preprocessing matches common HMS spectrogram handling by converting to log-domain with clipping (analogous to what you already do in `__getitem__`), which typically stabilizes the features and reduces KL without changing the overall approach. These changes are minimal, preserve your architecture/inference semantics, and mainly aim to avoid the uniform-fallback behavior and stabilize inputs so the score moves down toward the target.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far from the target (0.28545), and the biggest likely driver is that you still end up with many uniform (fallback) rows and/or low-quality predictions from running on CPU with a very large ViT input size. I make two minimal, score-relevant changes that preserve the same model and inference semantics: (1) ensure feature extraction/inference operate on one row per `eeg_id` (test has unique ids, but this also prevents accidental duplicates and stabilizes coverage), and (2) adjust only the inference image size and batch size to a lighter but still valid resolution (224) so CPU inference can complete reliably for all rows, reducing missing predictions and moving KL downward toward the target. I also add a tiny epsilon-smoothing to the final probabilities to reduce KL penalties from overconfident near-zeros while keeping rows normalized and the same softmax-based approach. The rest of your architecture, preprocessing functions, ensemble logic, and submission formatting remain unchanged.'
- What this solution (achieved 1.4093) has done: 'Your KL is far above the target (lower is better), so we should reduce it with the smallest, safest changes that increase the chance you actually produce meaningful (non-uniform) predictions for all test rows. The biggest score risk in your current pipeline is that you often end up with no usable checkpoints and therefore skip feature extraction/inference, yielding near-uniform predictions (very high KL). I make weight discovery prefer checkpoints that match your exact backbone string and “twostage” naming, and I make feature extraction run when either weights exist or cached features already exist (so reruns don’t silently fall back). Finally, I add a tiny per-row prior blend using the train vote distribution (a legitimate calibration for KL) to reduce penalty from pathological/overconfident outputs while keeping the same softmax-based inference and submission semantics.'
- What this solution (achieved 1.4093) has done: 'Your current KL (1.4093, lower is better) is far above the target (0.28545), and the dominant likely cause is that you’re still producing many near-uniform predictions (either because inference runs on CPU too slowly/incompletely, or because `avail_w` still doesn’t load anything compatible). I keep your exact model/feature logic, but make two minimal, score-relevant reliability changes: (1) enforce full prediction coverage by doing a lightweight “per-sample fallback inference” for any missing `eeg_id` whenever at least one checkpoint is loaded (so we don’t silently fill with uniform), and (2) slightly increase the prior-blend (a legitimate calibration for KL) only when we detect any missing predictions to reduce KL damage from remaining weak rows. These changes don’t alter the architecture, losses, or preprocessing; they only reduce the amount of uniform fallback and stabilize probabilities toward the expected class distribution. The script still run end-to-end and always write a valid `submission.csv` with rows summing to 1.'
- What this solution (achieved 1.4093) has done: 'Your KL is far above the target (lower is better), so we need a small, legitimate improvement without changing your model/feature pipeline. The biggest likely issue is that your CPU inference is not finishing reliably for all test rows (or is so slow that coverage drops and many rows fall back to uniform), which keeps KL very high. I keep the exact same architecture and feature extraction, but (1) make the DataLoader deterministic and more robust (no worker-related partial loads), and (2) reduce repeated model forward passes by batching the 3 raw_10s views into one concatenated forward per model (same computation, just fewer Python/model calls), improving the chance you get full coverage and non-uniform predictions for all ids within the time limit. Finally, I keep your existing probability smoothing/prior blend but ensure we never accidentally leave NaNs/Infs after resizing/normalization (which can silently poison predictions toward uniform/garbage and worsen KL).'
- What this solution (achieved 1.4093) has done: 'Your KL is far above the target (lower is better), so we should reduce it with minimal, score-relevant changes that keep your exact model/feature logic intact. The biggest likely cause is that you still end up with mostly-uniform predictions because no compatible checkpoints are actually being found/loaded in this environment, so I make weight discovery explicitly search the dataset folders you *do* have (`/kaggle/input/hms-harmful-brain-activity-classification` and siblings) and accept more realistic naming patterns (not only “twostage”). Next, I make the checkpoint-compatibility test less likely to reject good weights by relaxing the required presence of the auxiliary heads (head1–head4) while still enforcing the main head shape, so we can use checkpoints trained with only the main head instead of falling back to uniform. Finally, I keep your existing prior-blend/smoothing but only apply the stronger prior blend when we detect low coverage, so calibration helps KL without masking real predictions when coverage is good.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import random
import warnings
import os

warnings.filterwarnings("ignore")



## === cell 1
DEBUG = False



## === cell 2
NAMES = ["LL", "LP", "RP", "RR"]

FEATS = [
    ["Fp1", "F7", "T3", "T5", "O1"],
    ["Fp1", "F3", "C3", "P3", "O1"],
    ["Fp2", "F8", "T4", "T6", "O2"],
    ["Fp2", "F4", "C4", "P4", "O2"],
]



## === cell 3
from scipy.signal import butter, lfilter



## === cell 4
from scipy import signal


def stft_spec_from_eeg(parquet_path):
    EEG_LENGTH = 50
    eeg = pd.read_parquet(parquet_path)

    time_temp = 0
    time_start = round(time_temp + (50 - EEG_LENGTH) / 2 * 200)
    time_stop = round(time_temp + (50 + EEG_LENGTH) / 2 * 200)

    eeg = eeg.iloc[time_start:time_stop]

    list_eeg = list()
    for k in range(4):
        COLS = FEATS[k]
        img = np.zeros((128, 142, 4), dtype="float32")
        for kk in range(4):
            eeg_1 = eeg[COLS[kk]]
            mean_value = eeg_1.mean()
            eeg_1.fillna(value=mean_value, inplace=True)
            eeg_1 = eeg_1.values

            eeg_2 = eeg[COLS[kk + 1]]
            mean_value = eeg_2.mean()
            eeg_2.fillna(value=mean_value, inplace=True)
            eeg_2 = eeg_2.values

            new_eeg = eeg_1 - eeg_2
            del eeg_1
            del eeg_2
            fs = 200
            nperseg = 70
            noverlap = 0
            f, t, spec = signal.spectrogram(
                new_eeg, fs, nperseg=nperseg, noverlap=noverlap, nfft=256
            )

            spec = np.abs(spec)
            spec = np.log1p(spec).astype("float32")

            img[:, :, kk] += spec[:128, :]
        img = np.concatenate(
            (img[:, :, 0], img[:, :, 1], img[:, :, 2], img[:, :, 3]), 1
        )
        list_eeg.append(img)
    img = np.concatenate(list_eeg, 0)
    img /= 2.0
    return img




## === cell 5
NAMES = ["LL", "LP", "RP", "RR"]

SFREQ = 200

filter_range = [0.5, 40]

RAW_FEATS = {
    "LL": ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1"],
    "RL": ["Fp2-F8", "F8-T4", "T4-T6", "T6-O2"],
    "LP": ["Fp1-F3", "F3-C3", "C3-P3", "P3-O1"],
    "RP": ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2"],
}

b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")


def raw10seeg_from_eeg(parquet_path, eeg_id):
    EEG_LENGTH = 10
    raw_eeg = pd.read_parquet(parquet_path)

    time_temp = 0
    time_start = round(time_temp + (50 - EEG_LENGTH) / 2 * 200)
    time_stop = round(time_temp + (50 + EEG_LENGTH) / 2 * 200)

    eeg_default = raw_eeg.loc[time_start : (time_stop - 1), :].reset_index(drop=True)
    list_eeg = list()
    for region in RAW_FEATS.keys():
        eeg = np.zeros((len(RAW_FEATS[region]), eeg_default.shape[0]), dtype=np.float32)
        for chan_i, chan in enumerate(RAW_FEATS[region]):
            eeg_1 = eeg_default.loc[:, chan.split("-")[0]]
            mean_value = eeg_1.mean()
            eeg_1.fillna(value=mean_value, inplace=True)
            eeg_1 = eeg_1.values

            eeg_2 = eeg_default.loc[:, chan.split("-")[1]]
            mean_value = eeg_2.mean()
            eeg_2.fillna(value=mean_value, inplace=True)
            eeg_2 = eeg_2.values

            new_eeg = eeg_1 - eeg_2
            new_eeg = signal.filtfilt(b, a, new_eeg, axis=0)
            new_eeg = np.clip(new_eeg, -1024, 1024)
            eeg[chan_i, :] = new_eeg

        eeg = np.reshape(eeg, (4, 200, EEG_LENGTH))
        eeg = np.concatenate(
            [eeg[0, :, :], eeg[1, :, :], eeg[2, :, :], eeg[3, :, :]], 1
        )
        list_eeg.append(eeg)

    eeg_c = np.concatenate(list_eeg, 1)
    eeg_c /= 104

    time_temp = 0
    time_start = round(time_temp + 18 * 200)
    time_stop = round(time_temp + 28 * 200)

    eeg_default = raw_eeg.loc[time_start : (time_stop - 1), :].reset_index(drop=True)
    list_eeg = list()
    for region in RAW_FEATS.keys():
        eeg = np.zeros((len(RAW_FEATS[region]), eeg_default.shape[0]), dtype=np.float32)
        for chan_i, chan in enumerate(RAW_FEATS[region]):
            eeg_1 = eeg_default.loc[:, chan.split("-")[0]]
            mean_value = eeg_1.mean()
            eeg_1.fillna(value=mean_value, inplace=True)
            eeg_1 = eeg_1.values

            eeg_2 = eeg_default.loc[:, chan.split("-")[1]]
            mean_value = eeg_2.mean()
            eeg_2.fillna(value=mean_value, inplace=True)
            eeg_2 = eeg_2.values

            new_eeg = eeg_1 - eeg_2
            new_eeg = signal.filtfilt(b, a, new_eeg, axis=0)
            new_eeg = np.clip(new_eeg, -1024, 1024)
            eeg[chan_i, :] = new_eeg

        eeg = np.reshape(eeg, (4, 200, EEG_LENGTH))
        eeg = np.concatenate(
            [eeg[0, :, :], eeg[1, :, :], eeg[2, :, :], eeg[3, :, :]], 1
        )
        list_eeg.append(eeg)

    eeg_l = np.concatenate(list_eeg, 1)
    eeg_l /= 104

    time_temp = 0
    time_start = round(time_temp + 22 * 200)
    time_stop = round(time_temp + 32 * 200)

    eeg_default = raw_eeg.loc[time_start : (time_stop - 1), :].reset_index(drop=True)
    list_eeg = list()
    for region in RAW_FEATS.keys():
        eeg = np.zeros((len(RAW_FEATS[region]), eeg_default.shape[0]), dtype=np.float32)
        for chan_i, chan in enumerate(RAW_FEATS[region]):
            eeg_1 = eeg_default.loc[:, chan.split("-")[0]]
            mean_value = eeg_1.mean()
            eeg_1.fillna(value=mean_value, inplace=True)
            eeg_1 = eeg_1.values

            eeg_2 = eeg_default.loc[:, chan.split("-")[1]]
            mean_value = eeg_2.mean()
            eeg_2.fillna(value=mean_value, inplace=True)
            eeg_2 = eeg_2.values

            new_eeg = eeg_1 - eeg_2
            new_eeg = signal.filtfilt(b, a, new_eeg, axis=0)
            new_eeg = np.clip(new_eeg, -1024, 1024)
            eeg[chan_i, :] = new_eeg

        eeg = np.reshape(eeg, (4, 200, EEG_LENGTH))
        eeg = np.concatenate(
            [eeg[0, :, :], eeg[1, :, :], eeg[2, :, :], eeg[3, :, :]], 1
        )
        list_eeg.append(eeg)

    eeg_r = np.concatenate(list_eeg, 1)
    eeg_r /= 104

    return eeg_l, eeg_c, eeg_r


def raw50seeg_from_eeg(parquet_path):
    EEG_LENGTH = 50
    raw_eeg = pd.read_parquet(parquet_path)
    time_temp = 0
    time_start = round(time_temp + (50 - EEG_LENGTH) / 2 * 200)
    time_stop = round(time_temp + (50 + EEG_LENGTH) / 2 * 200)

    eeg_default = raw_eeg.loc[time_start : (time_stop - 1), :].reset_index(drop=True)
    list_eeg = list()
    for region in RAW_FEATS.keys():

        eeg = np.zeros((len(RAW_FEATS[region]), eeg_default.shape[0]), dtype=np.float32)
        for chan_i, chan in enumerate(RAW_FEATS[region]):
            eeg_1 = eeg_default.loc[:, chan.split("-")[0]]
            mean_value = eeg_1.mean()
            eeg_1.fillna(value=mean_value, inplace=True)
            eeg_1 = eeg_1.values

            eeg_2 = eeg_default.loc[:, chan.split("-")[1]]
            mean_value = eeg_2.mean()
            eeg_2.fillna(value=mean_value, inplace=True)
            eeg_2 = eeg_2.values

            new_eeg = eeg_1 - eeg_2
            del eeg_1
            del eeg_2
            new_eeg = signal.filtfilt(b, a, new_eeg, axis=0)
            new_eeg = np.clip(new_eeg, -1024, 1024).astype("float32")
            eeg[chan_i, :] = new_eeg

        eeg = np.reshape(eeg, (4, 200, EEG_LENGTH))
        eeg = np.concatenate(
            (eeg[0, :, :], eeg[1, :, :], eeg[2, :, :], eeg[3, :, :]), 1
        )
        list_eeg.append(eeg)

    eeg = np.concatenate(list_eeg, 1)
    eeg /= 104

    return eeg




## === cell 6
CLASSES = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
N_CLASSES = len(CLASSES)

if DEBUG == True:
    test = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    )[:40]
    SPEC_PATH = (
        "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/"
    )
    EEG_PATH = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"
else:
    test = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
    )
    SPEC_PATH = (
        "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
    )
    EEG_PATH = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"

test["eeg_id"] = test["eeg_id"].astype(str)
test["spectrogram_id"] = test["spectrogram_id"].astype(str)

test = test.drop_duplicates(subset=["eeg_id"], keep="first").reset_index(drop=True)

print(test.shape)

spec_directory_path = "spec_spectrograms/"
if not os.path.exists(spec_directory_path):
    os.makedirs(spec_directory_path)

eeg_directory_path = "eeg_spectrograms/"
if not os.path.exists(eeg_directory_path):
    os.makedirs(eeg_directory_path)

raw_10s_directory_path = "eeg_10s_raws/"
if not os.path.exists(raw_10s_directory_path):
    os.makedirs(raw_10s_directory_path)

raw_50s_directory_path = "eeg_50s_raws/"
if not os.path.exists(raw_50s_directory_path):
    os.makedirs(raw_50s_directory_path)

from joblib import Parallel, delayed

EEG_IDS = test.eeg_id.unique()




## === cell 7
class Config:
    seed = 2024
    num_folds = 5




## === cell 8
def seed_everything(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


seed_everything(Config.seed)



## === cell 9
import timm



## === cell 10
import torch.utils.data as data
import torchvision



## === cell 11
from torch.utils.data import DataLoader



## === cell 12
import gc



## === cell 13
from skimage.transform import resize


class ImageFolder(data.Dataset):
    def __init__(self, df, test_imgsize):
        super().__init__()
        df["eeg_id"] = df["eeg_id"]
        self.spec_data_path = spec_directory_path
        self.eeg_data_path = eeg_directory_path
        self.raw_50s_data_path = raw_50s_directory_path
        self.raw_10s_data_path = raw_10s_directory_path
        self.df = df.reset_index(drop=True)
        self.test_imgsize = test_imgsize

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.loc[index]
        eeg_id = str(row.eeg_id)
        spec_image_path = os.path.join(self.spec_data_path, eeg_id + ".npy")
        eeg_image_path = os.path.join(self.eeg_data_path, eeg_id + ".npy")
        raw_50s_image_path = os.path.join(self.raw_50s_data_path, eeg_id + ".npy")
        raw_10s_l_image_path = os.path.join(self.raw_10s_data_path, eeg_id + "_l.npy")
        raw_10s_c_image_path = os.path.join(self.raw_10s_data_path, eeg_id + "_c.npy")
        raw_10s_r_image_path = os.path.join(self.raw_10s_data_path, eeg_id + "_r.npy")

        spec_img = np.load(spec_image_path).astype("float32")
        raw_50s_img = np.load(raw_50s_image_path).astype("float32")
        raw_10s_l_img = np.load(raw_10s_l_image_path).astype("float32")
        raw_10s_c_img = np.load(raw_10s_c_image_path).astype("float32")
        raw_10s_r_img = np.load(raw_10s_r_image_path).astype("float32")
        eeg_img = np.load(eeg_image_path).astype("float32")

        eeg_img = resize(eeg_img, self.test_imgsize)
        spec_img = resize(spec_img, self.test_imgsize)
        raw_10s_l_img = resize(raw_10s_l_img, self.test_imgsize)
        raw_10s_c_img = resize(raw_10s_c_img, self.test_imgsize)
        raw_10s_r_img = resize(raw_10s_r_img, self.test_imgsize)
        raw_50s_img = resize(raw_50s_img, self.test_imgsize)

        eeg_img = np.expand_dims(eeg_img, -1)
        spec_img = np.expand_dims(spec_img, -1)
        raw_50s_img = np.expand_dims(raw_50s_img, -1)
        raw_10s_l_img = np.expand_dims(raw_10s_l_img, -1)
        raw_10s_c_img = np.expand_dims(raw_10s_c_img, -1)
        raw_10s_r_img = np.expand_dims(raw_10s_r_img, -1)

        eps = 1e-6
        spec_img = np.clip(spec_img, np.exp(-4), np.exp(8))
        spec_img = np.log(spec_img)
        spec_img = np.nan_to_num(spec_img, nan=0.0, posinf=0.0, neginf=0.0)

        img_mean = spec_img.mean(axis=(0, 1))
        img_std = spec_img.std(axis=(0, 1))
        spec_img = (spec_img - img_mean) / (img_std + eps)
        spec_img = np.nan_to_num(spec_img, nan=0.0, posinf=0.0, neginf=0.0)

        eeg_img = np.nan_to_num(eeg_img, nan=0.0, posinf=0.0, neginf=0.0)
        raw_50s_img = np.nan_to_num(raw_50s_img, nan=0.0, posinf=0.0, neginf=0.0)
        raw_10s_l_img = np.nan_to_num(raw_10s_l_img, nan=0.0, posinf=0.0, neginf=0.0)
        raw_10s_c_img = np.nan_to_num(raw_10s_c_img, nan=0.0, posinf=0.0, neginf=0.0)
        raw_10s_r_img = np.nan_to_num(raw_10s_r_img, nan=0.0, posinf=0.0, neginf=0.0)

        return (
            spec_img,
            eeg_img,
            raw_50s_img,
            raw_10s_l_img,
            raw_10s_c_img,
            raw_10s_r_img,
            eeg_id,
        )




## === cell 14
class Net(nn.Module):
    def __init__(self, back_bone, device_id):
        super().__init__()
        self.spec_model = timm.create_model(
            back_bone, num_classes=6, pretrained=False, in_chans=1
        )
        self.eeg_model = timm.create_model(
            back_bone, num_classes=6, pretrained=False, in_chans=1
        )
        self.raw_50s_model = timm.create_model(
            back_bone, num_classes=6, pretrained=False, in_chans=1
        )
        self.raw_10s_model = timm.create_model(
            back_bone, num_classes=6, pretrained=False, in_chans=1
        )

        self.device_id = device_id

        self.spec_model.fc_norm = nn.Identity()
        self.spec_model.head_drop = nn.Identity()
        self.spec_model.head = nn.Identity()

        self.eeg_model.fc_norm = nn.Identity()
        self.eeg_model.head_drop = nn.Identity()
        self.eeg_model.head = nn.Identity()

        self.raw_50s_model.fc_norm = nn.Identity()
        self.raw_50s_model.head_drop = nn.Identity()
        self.raw_50s_model.head = nn.Identity()

        self.raw_10s_model.fc_norm = nn.Identity()
        self.raw_10s_model.head_drop = nn.Identity()
        self.raw_10s_model.head = nn.Identity()

        self.head = nn.Linear(384 * 4, 6)
        self.head1 = nn.Linear(384, 6)
        self.head2 = nn.Linear(384, 6)
        self.head3 = nn.Linear(384, 6)
        self.head4 = nn.Linear(384, 6)

    def forward(self, spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_imgs):
        spec_imgs = spec_imgs.transpose(1, 2).transpose(1, 3).contiguous()
        eeg_imgs = eeg_imgs.transpose(1, 2).transpose(1, 3).contiguous()
        raw_50s_imgs = raw_50s_imgs.transpose(1, 2).transpose(1, 3).contiguous()
        raw_10s_imgs = raw_10s_imgs.transpose(1, 2).transpose(1, 3).contiguous()

        spec_feature = self.spec_model.forward_features(spec_imgs)[:, 0]
        eeg_feature = self.eeg_model.forward_features(eeg_imgs)[:, 0]
        raw_50s_feature = self.raw_50s_model.forward_features(raw_50s_imgs)[:, 0]
        raw_10s_feature = self.raw_10s_model.forward_features(raw_10s_imgs)[:, 0]

        feature = torch.cat(
            (spec_feature, eeg_feature, raw_50s_feature, raw_10s_feature), 1
        )
        logits = self.head(feature)
        logits_1 = self.head1(spec_feature)
        logits_2 = self.head2(eeg_feature)
        logits_3 = self.head3(raw_50s_feature)
        logits_4 = self.head4(raw_10s_feature)

        return logits, logits_1, logits_2, logits_3, logits_4




## === cell 15
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


def _existing_weights(paths):
    return [p for p in paths if os.path.exists(p)]


def _discover_model_weights(
    roots=(
        "/kaggle/input",
        "/kaggle/input/hms-harmful-brain-activity-classification",
        "/kaggle/data",
        "/kaggle/working",
    ),
    must_contain=tuple(),
    prefer_contain=tuple(),
    limit=400,
):
    exts = (".pth", ".pt", ".bin")
    found = []
    for root in roots:
        if not os.path.exists(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                low = fn.lower()
                if low.endswith(exts):
                    if must_contain and not all(k in low for k in must_contain):
                        continue
                    found.append(os.path.join(dirpath, fn))

    def _rank(p):
        low = os.path.basename(p).lower()
        score = 0
        for key, w in [
            ("vit_small_patch14", 200),
            ("dinov2", 120),
            ("reg4", 80),
            ("lvd142m", 60),
            ("twostage", 50),
            ("stage2", 30),
            ("hms", 20),
            ("harmful", 10),
            ("fold", 5),
            ("vit", 3),
        ]:
            if key in low:
                score += w
        if prefer_contain and all(k in low for k in prefer_contain):
            score += 1000
        return (-score, p)

    found = sorted(found, key=_rank)
    return found[:limit]


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for k in (
            "state_dict",
            "model",
            "model_state_dict",
            "net",
            "weights",
            "ema_state_dict",
        ):
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return ckpt_obj[k]
    return ckpt_obj


def _strip_prefix_from_state_dict(sd):
    if not isinstance(sd, dict):
        return sd
    prefixes = (
        "module.",
        "model.",
        "net.",
        "encoder.",
        "module.model.",
        "module.net.",
        "module.encoder.",
        "student.",
        "teacher.",
        "backbone.",
        "module.backbone.",
    )
    out = {}
    for k, v in sd.items():
        nk = k
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if nk.startswith(p):
                    nk = nk[len(p) :]
                    changed = True
        out[nk] = v
    return out


def _tensor_only_state_dict(sd):
    if not isinstance(sd, dict):
        return sd
    return {k: v for k, v in sd.items() if torch.is_tensor(v)}


def _is_compatible_checkpoint(model: nn.Module, state: dict, min_key_match_ratio=0.50):
    if not isinstance(state, dict) or len(state) == 0:
        return False

    model_sd = model.state_dict()
    model_keys = set(model_sd.keys())
    state_keys = set(state.keys())
    overlap = model_keys & state_keys
    ratio = len(overlap) / max(1, len(model_keys))

    for k in ("head.weight", "head.bias"):
        if k not in state or k not in model_sd:
            return False
        if tuple(state[k].shape) != tuple(model_sd[k].shape):
            return False

    for k in (
        "head1.weight",
        "head2.weight",
        "head3.weight",
        "head4.weight",
        "head1.bias",
        "head2.bias",
        "head3.bias",
        "head4.bias",
    ):
        if k in state and k in model_sd:
            if tuple(state[k].shape) != tuple(model_sd[k].shape):
                return False

    return ratio >= min_key_match_ratio


model_weights_stage2_small = [
    "/kaggle/input/hms-stage2/fold_0_raw_50_10_bestlb_twostage.pth",
    "/kaggle/input/hms-stage2/fold_1_raw_50_10_bestlb_twostage.pth",
    "/kaggle/input/hms-stage2/fold_2_raw_50_10_bestlb_twostage.pth",
    "/kaggle/input/hms-stage2/fold_3_raw_50_10_bestlb_twostage.pth",
    "/kaggle/input/hms-stage2/fold_4_raw_50_10_bestlb_twostage.pth",
]

avail_w = _existing_weights(model_weights_stage2_small)
if len(avail_w) == 0:
    auto_w = _discover_model_weights(
        must_contain=tuple(),
        prefer_contain=("vit_small_patch14", "dinov2"),
        limit=400,
    )
    if len(auto_w) == 0:
        auto_w = _discover_model_weights(
            must_contain=("hms",),
            prefer_contain=tuple(),
            limit=400,
        )

    if len(auto_w) > 0:
        print("Auto-discovered weight candidates (first ones):")
        for p in auto_w[:15]:
            print(" ", p)
        avail_w = auto_w




## === cell 16
def save(row):
    eeg_id = str(row["eeg_id"])
    spec_id = str(row["spectrogram_id"])
    spec = pd.read_parquet(f"{SPEC_PATH}{spec_id}.parquet")

    spec_arr = spec.values[:, 1:].T.astype("float32")  # (Hz, Time) = (400, 300)
    split_spec_arr = spec_arr[:, 0:300]

    split_spec_arr = np.clip(split_spec_arr, np.exp(-4), np.exp(8)).astype("float32")
    split_spec_arr = np.log(split_spec_arr).astype("float32")
    split_spec_arr = np.nan_to_num(
        split_spec_arr, nan=0.0, posinf=0.0, neginf=0.0
    ).astype("float32")

    np.save(f"{spec_directory_path}{eeg_id}", split_spec_arr)

    img_l, img_c, img_r = raw10seeg_from_eeg(f"{EEG_PATH}{eeg_id}.parquet", eeg_id)
    np.save(f"{raw_10s_directory_path}{eeg_id}_l", img_l)
    np.save(f"{raw_10s_directory_path}{eeg_id}_c", img_c)
    np.save(f"{raw_10s_directory_path}{eeg_id}_r", img_r)

    img = raw50seeg_from_eeg(f"{EEG_PATH}{eeg_id}.parquet")
    np.save(f"{raw_50s_directory_path}{eeg_id}", img)

    img = stft_spec_from_eeg(f"{EEG_PATH}{eeg_id}.parquet")
    np.save(f"{eeg_directory_path}{eeg_id}", img)


def _features_ready_for_any(df):
    if len(df) == 0:
        return False
    eeg_id0 = str(df.iloc[0]["eeg_id"])
    needed = [
        os.path.join(spec_directory_path, eeg_id0 + ".npy"),
        os.path.join(eeg_directory_path, eeg_id0 + ".npy"),
        os.path.join(raw_50s_directory_path, eeg_id0 + ".npy"),
        os.path.join(raw_10s_directory_path, eeg_id0 + "_l.npy"),
        os.path.join(raw_10s_directory_path, eeg_id0 + "_c.npy"),
        os.path.join(raw_10s_directory_path, eeg_id0 + "_r.npy"),
    ]
    return all(os.path.exists(p) for p in needed)


if (len(avail_w) > 0) or _features_ready_for_any(test):
    _ = Parallel(n_jobs=4)(delayed(save)(row) for index, row in test.iterrows())
else:
    print(
        "No usable weights found and no cached features; skipping feature extraction and producing uniform submission."
    )




## === cell 17
def _to_str_id(x):
    if isinstance(x, (list, tuple)):
        if len(x) == 1:
            return _to_str_id(x[0])
        return str(x[0])
    if isinstance(x, str):
        return x
    if isinstance(x, bytes):
        return x.decode("utf-8")
    if torch.is_tensor(x):
        if x.numel() == 1:
            return str(x.detach().cpu().item())
        return str(x.detach().cpu().view(-1)[0].item())
    if isinstance(x, np.ndarray):
        if x.shape == ():
            return str(x.item())
        if x.size == 1:
            return str(x.reshape(-1)[0].item())
        return str(x.reshape(-1)[0].item())
    return str(x)


result_7 = {}
vit_models = []

if len(avail_w) > 0 and _features_ready_for_any(test):
    for w in avail_w:
        try:
            model = Net("vit_small_patch14_reg4_dinov2.lvd142m", device).to(device)
            ckpt = torch.load(w, map_location=device)
            state = _extract_state_dict(ckpt)
            state = _strip_prefix_from_state_dict(state)
            state = _tensor_only_state_dict(state)

            if not _is_compatible_checkpoint(model, state, min_key_match_ratio=0.50):
                print("Skipping likely-incompatible weight (shape/key mismatch):", w)
                del model
                continue

            missing, unexpected = model.load_state_dict(state, strict=False)
            model.eval()
            vit_models.append(model)
            print(
                "Loaded:",
                w,
                "| missing:",
                len(missing),
                "| unexpected:",
                len(unexpected),
            )
            if len(vit_models) >= 5:
                break
        except Exception as e:
            print(
                "Skipping incompatible weight:",
                w,
                "| Error:",
                type(e).__name__,
                str(e)[:200],
            )
            try:
                del model
            except Exception:
                pass

    if len(vit_models) > 0:
        test_imgsize = (224, 224)

        test_data = ImageFolder(test, test_imgsize)
        test_loader = DataLoader(
            test_data,
            batch_size=4,
            pin_memory=False,
            num_workers=0,
            drop_last=False,
        )

        with torch.no_grad():
            for (
                spec_imgs,
                eeg_imgs,
                raw_50s_imgs,
                raw_10s_l_imgs,
                raw_10s_c_imgs,
                raw_10s_r_imgs,
                eeg_ids,
            ) in test_loader:
                spec_imgs = spec_imgs.to(device).float()
                eeg_imgs = eeg_imgs.to(device).float()
                raw_50s_imgs = raw_50s_imgs.to(device).float()
                raw_10s_l_imgs = raw_10s_l_imgs.to(device).float()
                raw_10s_c_imgs = raw_10s_c_imgs.to(device).float()
                raw_10s_r_imgs = raw_10s_r_imgs.to(device).float()

                B = spec_imgs.shape[0]
                spec3 = torch.cat([spec_imgs, spec_imgs, spec_imgs], dim=0)
                eeg3 = torch.cat([eeg_imgs, eeg_imgs, eeg_imgs], dim=0)
                raw50_3 = torch.cat([raw_50s_imgs, raw_50s_imgs, raw_50s_imgs], dim=0)
                raw10_3 = torch.cat(
                    [raw_10s_l_imgs, raw_10s_c_imgs, raw_10s_r_imgs], dim=0
                )

                ensemble_probs = torch.zeros((B, 6), device=device)
                for model in vit_models:
                    logits_all, _, _, _, _ = model(spec3, eeg3, raw50_3, raw10_3)
                    probs_all = logits_all.softmax(dim=1)  # (3B,6)
                    probs_l = probs_all[0:B]
                    probs_c = probs_all[B : 2 * B]
                    probs_r = probs_all[2 * B : 3 * B]
                    ensemble_probs += (probs_l + probs_c + probs_r) / 3.0

                ensemble_probs /= max(1, len(vit_models))
                ensemble_probs = ensemble_probs.detach().cpu().numpy()

                for j in range(len(eeg_ids)):
                    eeg_id = _to_str_id(eeg_ids[j])
                    result_7[eeg_id] = ensemble_probs[j].astype("float64")

        coverage = len(result_7) / max(1, len(test))
        print(f"Prediction coverage: {len(result_7)}/{len(test)} = {coverage:.3f}")
        if coverage < 0.95:
            print(
                "WARNING: Low prediction coverage; some eeg_id will require fallback handling."
            )
    else:
        print(
            "No compatible weights loaded; result_7 will remain empty (uniform fallback)."
        )
else:
    if len(avail_w) == 0:
        print("No weight paths available; skipping model inference for result_7.")
    else:
        print(
            "Weights exist but features are not available; skipping inference (uniform fallback)."
        )



## === cell 18
for model in vit_models:
    del model
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()



## === cell 19
result_8 = {}
result_5 = {}
result_4 = {}



## === cell 20
sub = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
)
sub["eeg_id"] = sub["eeg_id"].astype(str)
sub_ids = sub["eeg_id"].tolist()

preds = np.zeros((len(sub_ids), N_CLASSES), dtype=np.float64)

have_any = len(result_7) > 0
missing_ids = []

if have_any:
    for i, eeg_id in enumerate(sub_ids):
        if eeg_id in result_7:
            preds[i] = result_7[eeg_id]
        else:
            missing_ids.append(eeg_id)
            preds[i] = 1.0 / N_CLASSES
    print("Missing eeg_id predictions:", len(missing_ids), "/", len(sub_ids))
else:
    preds[:] = 1.0 / N_CLASSES

if have_any and (len(missing_ids) > 0) and _features_ready_for_any(test):
    try:
        fallback_model = None
        for w in avail_w:
            try:
                m = Net("vit_small_patch14_reg4_dinov2.lvd142m", device).to(device)
                ckpt = torch.load(w, map_location=device)
                state = _tensor_only_state_dict(
                    _strip_prefix_from_state_dict(_extract_state_dict(ckpt))
                )
                if _is_compatible_checkpoint(m, state, min_key_match_ratio=0.50):
                    m.load_state_dict(state, strict=False)
                    m.eval()
                    fallback_model = m
                    break
                else:
                    del m
            except Exception:
                try:
                    del m
                except Exception:
                    pass

        if fallback_model is not None:
            miss_df = (
                test[test["eeg_id"].isin(missing_ids)].copy().reset_index(drop=True)
            )
            test_imgsize = (224, 224)
            miss_data = ImageFolder(miss_df, test_imgsize)
            miss_loader = DataLoader(
                miss_data,
                batch_size=1,
                pin_memory=False,
                num_workers=0,
                drop_last=False,
            )

            miss_preds = {}
            with torch.no_grad():
                for (
                    spec_imgs,
                    eeg_imgs,
                    raw_50s_imgs,
                    raw_10s_l_imgs,
                    raw_10s_c_imgs,
                    raw_10s_r_imgs,
                    eeg_ids,
                ) in miss_loader:
                    spec_imgs = spec_imgs.to(device).float()
                    eeg_imgs = eeg_imgs.to(device).float()
                    raw_50s_imgs = raw_50s_imgs.to(device).float()
                    raw_10s_l_imgs = raw_10s_l_imgs.to(device).float()
                    raw_10s_c_imgs = raw_10s_c_imgs.to(device).float()
                    raw_10s_r_imgs = raw_10s_r_imgs.to(device).float()

                    logits_l, _, _, _, _ = fallback_model(
                        spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_l_imgs
                    )
                    logits_c, _, _, _, _ = fallback_model(
                        spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_c_imgs
                    )
                    logits_r, _, _, _, _ = fallback_model(
                        spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_r_imgs
                    )
                    probs = (
                        logits_l.softmax(dim=1)
                        + logits_c.softmax(dim=1)
                        + logits_r.softmax(dim=1)
                    ) / 3.0
                    eeg_id = _to_str_id(eeg_ids[0])
                    miss_preds[eeg_id] = (
                        probs.detach().cpu().numpy()[0].astype("float64")
                    )

            fill_ct = 0
            for i, eeg_id in enumerate(sub_ids):
                if eeg_id in miss_preds:
                    preds[i] = miss_preds[eeg_id]
                    fill_ct += 1
            print("Filled missing ids via fallback single-sample inference:", fill_ct)

            del fallback_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        else:
            print(
                "Fallback model could not be loaded; keeping uniform for missing ids."
            )
    except Exception as e:
        print(
            "Fallback inference failed; keeping uniform for missing ids.",
            type(e).__name__,
            str(e)[:200],
        )

train_df = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
)
vote_arr = train_df[CLASSES].values.astype(np.float64)
prior = vote_arr.sum(axis=0)
prior = prior / prior.sum()
prior = prior.reshape(1, -1)

preds = np.clip(preds, 1e-12, 1.0)
smooth_eps = 5e-4
preds = (1.0 - smooth_eps) * preds + smooth_eps * (1.0 / N_CLASSES)

prior_blend = 0.02
if have_any:
    cov = 1.0 - (len(missing_ids) / max(1, len(sub_ids)))
    if cov < 0.90:
        prior_blend = 0.06
    elif cov < 0.98:
        prior_blend = 0.04

preds = (1.0 - prior_blend) * preds + prior_blend * prior
preds = np.nan_to_num(
    preds, nan=1.0 / N_CLASSES, posinf=1.0 / N_CLASSES, neginf=1.0 / N_CLASSES
)
preds = preds / preds.sum(axis=1, keepdims=True)

sub[CLASSES] = preds
sub.to_csv("submission.csv", index=False)
print(sub.head())
print(
    "Saved submission.csv with shape:", sub.shape, "| have_any_predictions:", have_any
)



## === cell 21
if DEBUG == False:
    import shutil

    for p in [
        "spec_spectrograms",
        "eeg_spectrograms",
        "eeg_50s_raws",
        "eeg_10s_raws",
        "squeezeformer",
    ]:
        if os.path.exists(p):
            shutil.rmtree(p, ignore_errors=True)
