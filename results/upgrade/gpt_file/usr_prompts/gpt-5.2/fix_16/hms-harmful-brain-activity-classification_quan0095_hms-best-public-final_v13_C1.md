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

0.2875132578344483

# 6. Current score

0.88599

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the immediate crash by removing the dependency on missing external model weight files and replacing it with a safe, deterministic fallback that still produces valid probability predictions. I also remove the unnecessary `librosa` import (not installed by default in this environment) to prevent an earlier import-time failure. To keep the core pipeline intact, I preserve the same feature extraction and dataset code, but make the inference stage robust: if weights are unavailable, it generate calibrated uniform predictions (valid for KL metric and submission constraints) rather than erroring out. Finally, I ensure `result_5` is always defined and that the written `submission.csv` has the exact required columns and row-wise probabilities summing to 1.'
- What this solution (achieved 1.40995) has done: 'The timeout is dominated by the huge, repeated feature extraction step that reads/parses thousands of parquet files and runs `signal.spectrogram`/`filtfilt` multiple times per EEG, plus slow per-sample `skimage.resize` in the DataLoader. I keep the same feature definitions and model logic, but make extraction provably equivalent faster by (1) reading each EEG parquet only once per id (instead of 4x), (2) vectorizing channel differencing and NaN fill, and (3) using process-based parallelism with safe worker counts. For training/inference, I keep the same model/loops but speed up input pipeline by using `torch.nn.functional.interpolate` (GPU/CPU-optimized) instead of `skimage.resize`, and by enabling pinned memory + multiple DataLoader workers without changing data order/semantics. These changes reduce constant factors substantially while preserving the algorithm and outputs up to negligible floating-point differences.'
- What this solution (achieved 1.40629) has done: 'The timeout is dominated by the expensive per-`eeg_id` preprocessing step (reading parquet + multiple `filtfilt` + multiple `spectrogram` + multiple resizes + writing 6 `.npy` files) executed for ~30k IDs (train+val+test) due to `need_all` concatenation; the training loop itself then becomes secondary. I keep the exact same models, loss, and training semantics, but remove the unnecessary train/val preprocessing by restricting feature materialization to **test only** (submission only needs test features), which is provably equivalent for producing `submission.csv` and preserves evaluation semantics for inference. For the remaining test preprocessing, I also avoid reloading the same EEG parquet multiple times inside each job by caching derived channel-index pairs and use faster thread-based parallelism for I/O-bound parquet/resize work (no algorithm changes). Finally, I keep determinism settings and avoid any approximations/precision changes.'
- What this solution (achieved 1.41937) has done: 'Your current score (1.406, lower-is-better) is far worse than the target (0.287), so we should improve predictions without changing the core model/features/training (none) and while keeping changes minimal. The biggest issue is that you are running randomly initialized ViT models (no weights), so outputs are essentially noise; the least invasive way to move KL toward the target is to remove that randomness by using a stable, label-free prior-only predictor derived from train vote distributions. We keep all preprocessing/inference code intact, but add a simple gate: if no valid model checkpoint is present, skip heavy inference and write a prior-based submission (optionally lightly mixed with uniform) with correct normalization. This should drastically reduce KL versus random logits while remaining fully legitimate (no leakage) and producing a valid submission quickly and deterministically.'
- What this solution (achieved 0.80158) has done: 'I fix the crash in cell 13 by building `test_pid_map` with a correctly aligned index (use the deduplicated dataframe’s own `eeg_id` column, not the full `test` column). This is a pure bug fix that unblocks writing `submission.csv` and does not change any model/feature logic. I also add a small safety normalization when inserting `result_5` predictions to guarantee each row sums to 1 (required by the competition) even if a checkpoint produces slightly unnormalized probabilities. Everything else (feature extraction, model, inference flow, prior fallback) stays the same.'
- What this solution (achieved 0.8252) has done: 'Your current score (0.80158, lower-is-better) is far worse than the target (0.2875), and the biggest limiter is that “model inference” is effectively random because the code may load an unrelated checkpoint (any .pth/.pt/.bin found) with `strict=False`. To move KL closer to the target with minimal logic changes, I (1) stop using arbitrary checkpoints unless they look compatible, and (2) strengthen the already-legitimate, label-free prior fallback by using a better-calibrated mixture of patient prior and global prior, plus a tiny uniform smoothing to avoid extreme probabilities. This keeps your feature extraction, model architecture, and inference loop intact when a real compatible checkpoint exists, but avoids harmful random predictions when it doesn’t. Finally, the submission normalization is kept and made numerically safe so every row sums to 1.'
- What this solution (achieved 0.90594) has done: 'Your current score (0.8252, lower-is-better) is still far from the target (0.2875), and the main remaining lever without changing the model/features/training is improving the **fallback** distribution used when checkpoints are missing/ineffective. I keep all feature extraction/model code intact, but replace the current global/patient-only prior with a better-calibrated **test-time mixture**: (1) patient prior if available, (2) a spectrogram-level prior keyed by `spectrogram_id` (computed from train, no leakage), and (3) the global prior, then apply the same safe smoothing/normalization. This is a minimal change localized to prediction post-processing and typically reduces KL a lot versus using only patient/global priors. The submission writing stays identical and still guarantees row-wise probabilities sum to 1.'
- What this solution (achieved 0.88599) has done: 'Your current score (0.90594, lower-is-better) is far worse than the target (0.2875), and since the model is almost certainly running without a truly trained/compatible checkpoint, the biggest gain with minimal change is improving the *fallback* probabilities (which dominate your submission). I keep your whole preprocessing/model code intact, but change the fallback prior from “sum votes then normalize” to the *expected label distribution under the competition’s KL target construction*: normalize votes to probabilities **per eeg_id first**, then average those per-eeg probabilities when building global/patient/spectrogram priors. I also keep your existing smoothing/mixing but slightly reduce the final global-mix/uniform-mix so the improved priors aren’t washed out, while still guaranteeing no zeros and row sums of 1. This should move KL substantially toward your target without changing architecture, features, loops, or inference semantics.'

# 9. Code solution

## === cell 0
import os
import warnings
import random
import gc

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

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
from scipy import signal




## === cell 4
SFREQ = 200
filter_range = [0.5, 40]
b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")

RAW_FEATS = {
    "LL": ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1"],
    "RL": ["Fp2-F8", "F8-T4", "T4-T6", "T6-O2"],
    "LP": ["Fp1-F3", "F3-C3", "C3-P3", "P3-O1"],
    "RP": ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2"],
}

_ALL_EEG_COLS = sorted(
    {c for group in FEATS for c in group}
    | {c for r in RAW_FEATS.values() for ch in r for c in ch.split("-")}
)

_EEG_CACHE = {}
_SPEC_CACHE = {}
_MAX_EEG_CACHE_ITEMS = 16
_MAX_SPEC_CACHE_ITEMS = 32


def _cache_put(cache, key, value, max_items):
    cache[key] = value
    if len(cache) > max_items:
        try:
            cache.pop(next(iter(cache)))
        except Exception:
            cache.clear()


def _load_eeg_array(parquet_path):
    arr_col = _EEG_CACHE.get(parquet_path)
    if arr_col is not None:
        return arr_col
    df = pd.read_parquet(parquet_path, columns=_ALL_EEG_COLS)
    arr = df.to_numpy(dtype=np.float32, copy=False)
    if np.isnan(arr).any():
        col_means = np.nanmean(arr, axis=0)
        inds = np.where(np.isnan(arr))
        arr[inds] = col_means[inds[1]]
    col_index = {c: i for i, c in enumerate(df.columns)}
    out = (arr, col_index)
    _cache_put(_EEG_CACHE, parquet_path, out, _MAX_EEG_CACHE_ITEMS)
    return out


def stft_spec_from_eeg_array(eeg_arr, col_index):
    EEG_LENGTH = 50
    time_start = round((50 - EEG_LENGTH) / 2 * 200)
    time_stop = round((50 + EEG_LENGTH) / 2 * 200)
    eeg_arr = eeg_arr[time_start:time_stop]

    list_eeg = []
    fs = 200
    nperseg = 70
    noverlap = 0

    for k in range(4):
        COLS = FEATS[k]
        img = np.zeros((128, 142, 4), dtype=np.float32)
        for kk in range(4):
            x1 = eeg_arr[:, col_index[COLS[kk]]]
            x2 = eeg_arr[:, col_index[COLS[kk + 1]]]
            new_eeg = x1 - x2
            f, t, spec = signal.spectrogram(
                new_eeg, fs, nperseg=nperseg, noverlap=noverlap, nfft=256
            )
            spec = np.log1p(np.abs(spec)).astype(np.float32, copy=False)
            img[:, :, kk] += spec[:128, :]
        img = np.concatenate(
            (img[:, :, 0], img[:, :, 1], img[:, :, 2], img[:, :, 3]), axis=1
        )
        list_eeg.append(img)
    img = np.concatenate(list_eeg, axis=0)
    img /= 2.0
    return img


def _build_raw_feat_indices(col_index):
    idx_pairs = {}
    for region, chans in RAW_FEATS.items():
        idx1 = np.fromiter(
            (col_index[ch.split("-")[0]] for ch in chans), dtype=np.int32
        )
        idx2 = np.fromiter(
            (col_index[ch.split("-")[1]] for ch in chans), dtype=np.int32
        )
        idx_pairs[region] = (idx1, idx2)
    return idx_pairs


def _raw10s_block(eeg_arr, idx_pairs, time_start, time_stop, EEG_LENGTH=10):
    eeg_default = eeg_arr[time_start:time_stop]  # (T, C)
    list_eeg = []
    for region, (idx1, idx2) in idx_pairs.items():
        diff = eeg_default[:, idx1] - eeg_default[:, idx2]  # (T, 4)
        diff = signal.filtfilt(b, a, diff, axis=0)
        diff = np.clip(diff, -1024, 1024).astype(np.float32, copy=False)  # (T, 4)
        eeg = diff.T  # (4, T)

        eeg = np.reshape(eeg, (4, 200, EEG_LENGTH))
        eeg = np.concatenate(
            [eeg[0, :, :], eeg[1, :, :], eeg[2, :, :], eeg[3, :, :]], axis=1
        )
        list_eeg.append(eeg)
    eeg_c = np.concatenate(list_eeg, axis=1)
    eeg_c /= 104.0
    return eeg_c


def raw10seeg_from_eeg_array(eeg_arr, col_index):
    idx_pairs = _build_raw_feat_indices(col_index)

    EEG_LENGTH = 10
    t0 = round((50 - EEG_LENGTH) / 2 * 200)
    t1 = round((50 + EEG_LENGTH) / 2 * 200)
    eeg_c = _raw10s_block(eeg_arr, idx_pairs, t0, t1, EEG_LENGTH=EEG_LENGTH)

    t0 = round(18 * 200)
    t1 = round(28 * 200)
    eeg_l = _raw10s_block(eeg_arr, idx_pairs, t0, t1, EEG_LENGTH=EEG_LENGTH)

    t0 = round(22 * 200)
    t1 = round(32 * 200)
    eeg_r = _raw10s_block(eeg_arr, idx_pairs, t0, t1, EEG_LENGTH=EEG_LENGTH)
    return eeg_l, eeg_c, eeg_r


def raw50seeg_from_eeg_array(eeg_arr, col_index):
    EEG_LENGTH = 50
    time_start = round((50 - EEG_LENGTH) / 2 * 200)
    time_stop = round((50 + EEG_LENGTH) / 2 * 200)
    eeg_default = eeg_arr[time_start:time_stop]

    idx_pairs = _build_raw_feat_indices(col_index)

    list_eeg = []
    for region, (idx1, idx2) in idx_pairs.items():
        diff = eeg_default[:, idx1] - eeg_default[:, idx2]  # (T, 4)
        diff = signal.filtfilt(b, a, diff, axis=0)
        diff = np.clip(diff, -1024, 1024).astype(np.float32, copy=False)  # (T, 4)
        eeg = diff.T  # (4, T)

        eeg = np.reshape(eeg, (4, 200, EEG_LENGTH))
        eeg = np.concatenate(
            (eeg[0, :, :], eeg[1, :, :], eeg[2, :, :], eeg[3, :, :]), axis=1
        )
        list_eeg.append(eeg)
    eeg = np.concatenate(list_eeg, axis=1)
    eeg /= 104.0
    return eeg




## === cell 5
CLASSES = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
N_CLASSES = len(CLASSES)

train = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/train.csv")
test = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/test.csv")

TRAIN_SPEC_PATH = (
    "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/"
)
TRAIN_EEG_PATH = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"
TEST_SPEC_PATH = (
    "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
)
TEST_EEG_PATH = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"

print("train:", train.shape, "test:", test.shape)

spec_directory_path = "spec_spectrograms/"
eeg_directory_path = "eeg_spectrograms/"
raw_10s_directory_path = "eeg_10s_raws/"
raw_50s_directory_path = "eeg_50s_raws/"

for p in [
    spec_directory_path,
    eeg_directory_path,
    raw_10s_directory_path,
    raw_50s_directory_path,
]:
    if not os.path.exists(p):
        os.makedirs(p)




## === cell 6
from joblib import Parallel, delayed

_PRECOMP_IMG_SIZE = (224, 224)

try:
    import cv2

    def _resize2d_np_bilinear(arr2d, out_hw=_PRECOMP_IMG_SIZE):
        h, w = out_hw
        out = cv2.resize(
            np.asarray(arr2d, dtype=np.float32),
            (w, h),
            interpolation=cv2.INTER_LINEAR,
        )
        return out.astype(np.float32, copy=False)

except Exception:

    def _resize2d_np_bilinear(arr2d, out_hw=_PRECOMP_IMG_SIZE):
        t = torch.from_numpy(np.asarray(arr2d)).unsqueeze(0).unsqueeze(0)  # 1,1,H,W
        t = F.interpolate(t, size=out_hw, mode="bilinear", align_corners=False)
        return (
            t.squeeze(0).squeeze(0).contiguous().numpy().astype(np.float32, copy=False)
        )


_IDXPAIR_CACHE = {}
_MAX_IDXPAIR_CACHE_ITEMS = 64


def _get_idx_pairs_for_eeg(parquet_path, col_index):
    v = _IDXPAIR_CACHE.get(parquet_path)
    if v is not None:
        return v
    v = _build_raw_feat_indices(col_index)
    _cache_put(_IDXPAIR_CACHE, parquet_path, v, _MAX_IDXPAIR_CACHE_ITEMS)
    return v


def save_from_paths(eeg_id, spec_id, SPEC_PATH, EEG_PATH):
    eeg_id = str(eeg_id)
    spec_id = str(spec_id)

    spec_out = f"{spec_directory_path}{eeg_id}.npy"
    eeg_out = f"{eeg_directory_path}{eeg_id}.npy"
    raw50_out = f"{raw_50s_directory_path}{eeg_id}.npy"
    raw10_l_out = f"{raw_10s_directory_path}{eeg_id}_l.npy"
    raw10_c_out = f"{raw_10s_directory_path}{eeg_id}_c.npy"
    raw10_r_out = f"{raw_10s_directory_path}{eeg_id}_r.npy"

    if (
        os.path.exists(spec_out)
        and os.path.exists(eeg_out)
        and os.path.exists(raw50_out)
        and os.path.exists(raw10_l_out)
        and os.path.exists(raw10_c_out)
        and os.path.exists(raw10_r_out)
    ):
        return

    spec_path = f"{SPEC_PATH}{spec_id}.parquet"
    spec_df = _SPEC_CACHE.get(spec_path)
    if spec_df is None:
        spec_df = pd.read_parquet(spec_path)
        _cache_put(_SPEC_CACHE, spec_path, spec_df, _MAX_SPEC_CACHE_ITEMS)

    spec_arr = spec_df.values[:, 1:].T.astype("float32", copy=False)
    split_spec_arr = spec_arr[:, 0:300]
    split_spec_arr = _resize2d_np_bilinear(split_spec_arr, _PRECOMP_IMG_SIZE)
    np.save(f"{spec_directory_path}{eeg_id}", split_spec_arr)

    eeg_parquet = f"{EEG_PATH}{eeg_id}.parquet"
    eeg_arr, col_index = _load_eeg_array(eeg_parquet)
    idx_pairs = _get_idx_pairs_for_eeg(eeg_parquet, col_index)

    EEG_LENGTH = 10
    t0 = round(18 * 200)
    t1 = round(28 * 200)
    img_l = _raw10s_block(eeg_arr, idx_pairs, t0, t1, EEG_LENGTH=EEG_LENGTH)
    t0 = round((50 - EEG_LENGTH) / 2 * 200)
    t1 = round((50 + EEG_LENGTH) / 2 * 200)
    img_c = _raw10s_block(eeg_arr, idx_pairs, t0, t1, EEG_LENGTH=EEG_LENGTH)
    t0 = round(22 * 200)
    t1 = round(32 * 200)
    img_r = _raw10s_block(eeg_arr, idx_pairs, t0, t1, EEG_LENGTH=EEG_LENGTH)

    np.save(
        f"{raw_10s_directory_path}{eeg_id}_l",
        _resize2d_np_bilinear(img_l, _PRECOMP_IMG_SIZE),
    )
    np.save(
        f"{raw_10s_directory_path}{eeg_id}_c",
        _resize2d_np_bilinear(img_c, _PRECOMP_IMG_SIZE),
    )
    np.save(
        f"{raw_10s_directory_path}{eeg_id}_r",
        _resize2d_np_bilinear(img_r, _PRECOMP_IMG_SIZE),
    )

    img50 = raw50seeg_from_eeg_array(eeg_arr, col_index)
    np.save(
        f"{raw_50s_directory_path}{eeg_id}",
        _resize2d_np_bilinear(img50, _PRECOMP_IMG_SIZE),
    )

    imgstft = stft_spec_from_eeg_array(eeg_arr, col_index)
    np.save(
        f"{eeg_directory_path}{eeg_id}",
        _resize2d_np_bilinear(imgstft, _PRECOMP_IMG_SIZE),
    )




## === cell 7
class Config:
    seed = 2024
    num_folds = 5




## === cell 8
def seed_everything(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


seed_everything(Config.seed)




## === cell 9
import timm
import torch.utils.data as data
from torch.utils.data import DataLoader




## === cell 10
class ImageFolder(data.Dataset):
    def __init__(self, df, test_imgsize, targets=None):
        super().__init__()
        self.spec_data_path = spec_directory_path
        self.eeg_data_path = eeg_directory_path
        self.raw_50s_data_path = raw_50s_directory_path
        self.raw_10s_data_path = raw_10s_directory_path
        self.df = df.reset_index(drop=True)
        self.test_imgsize = test_imgsize
        self.targets = targets

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _np_load_f32(path):
        arr = np.load(path, mmap_mode="r")
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        return arr

    def __getitem__(self, index):
        row = self.df.loc[index]
        eeg_id = str(row.eeg_id)

        spec_img = self._np_load_f32(os.path.join(self.spec_data_path, eeg_id + ".npy"))
        raw_50s_img = self._np_load_f32(
            os.path.join(self.raw_50s_data_path, eeg_id + ".npy")
        )
        raw_10s_l_img = self._np_load_f32(
            os.path.join(self.raw_10s_data_path, eeg_id + "_l.npy")
        )
        raw_10s_c_img = self._np_load_f32(
            os.path.join(self.raw_10s_data_path, eeg_id + "_c.npy")
        )
        raw_10s_r_img = self._np_load_f32(
            os.path.join(self.raw_10s_data_path, eeg_id + "_r.npy")
        )
        eeg_img = self._np_load_f32(os.path.join(self.eeg_data_path, eeg_id + ".npy"))

        eeg_img = torch.from_numpy(np.asarray(eeg_img)).unsqueeze(-1)
        spec_img = torch.from_numpy(np.asarray(spec_img)).unsqueeze(-1)
        raw_10s_l_img = torch.from_numpy(np.asarray(raw_10s_l_img)).unsqueeze(-1)
        raw_10s_c_img = torch.from_numpy(np.asarray(raw_10s_c_img)).unsqueeze(-1)
        raw_10s_r_img = torch.from_numpy(np.asarray(raw_10s_r_img)).unsqueeze(-1)
        raw_50s_img = torch.from_numpy(np.asarray(raw_50s_img)).unsqueeze(-1)

        eps = 1e-6
        spec_img = torch.clamp(spec_img, min=float(np.exp(-4)), max=float(np.exp(8)))
        spec_img = torch.log(spec_img)
        spec_img = torch.nan_to_num(spec_img, nan=0.0)

        img_mean = spec_img.mean(dim=(0, 1), keepdim=True)
        img_std = spec_img.std(dim=(0, 1), keepdim=True)
        spec_img = (spec_img - img_mean) / (img_std + eps)

        if self.targets is None:
            return (
                spec_img,
                eeg_img,
                raw_50s_img,
                raw_10s_l_img,
                raw_10s_c_img,
                raw_10s_r_img,
                eeg_id,
            )
        else:
            y = self.targets[index].astype(np.float32)
            return (
                spec_img,
                eeg_img,
                raw_50s_img,
                raw_10s_l_img,
                raw_10s_c_img,
                raw_10s_r_img,
                y,
            )




## === cell 11
class Net(nn.Module):
    def __init__(self, back_bone, device_id):
        super().__init__()
        self.spec_model = timm.create_model(
            "vit_small_patch14_reg4_dinov2.lvd142m",
            num_classes=6,
            pretrained=False,
            in_chans=1,
            img_size=None,
            dynamic_img_size=True,
        )
        self.eeg_model = timm.create_model(
            "vit_small_patch14_reg4_dinov2.lvd142m",
            num_classes=6,
            pretrained=False,
            in_chans=1,
            img_size=None,
            dynamic_img_size=True,
        )
        self.raw_50s_model = timm.create_model(
            back_bone,
            num_classes=6,
            pretrained=False,
            in_chans=1,
            img_size=None,
            dynamic_img_size=True,
        )
        self.raw_10s_model = timm.create_model(
            back_bone,
            num_classes=6,
            pretrained=False,
            in_chans=1,
            img_size=None,
            dynamic_img_size=True,
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

        self.head = nn.Linear(384 * 2 + 768 * 2, 6)
        self.head1 = nn.Linear(384, 6)
        self.head2 = nn.Linear(384, 6)
        self.head3 = nn.Linear(768, 6)
        self.head4 = nn.Linear(768, 6)

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




## === cell 12
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

vote_sum_by_eeg = train.groupby("eeg_id")[CLASSES].sum()
vote_prob_by_eeg = vote_sum_by_eeg.div(
    vote_sum_by_eeg.sum(axis=1).replace(0, np.nan), axis=0
).fillna(1.0 / N_CLASSES)

global_prior = vote_prob_by_eeg.mean(axis=0).values.astype(np.float32)
global_prior = global_prior / global_prior.sum()

eeg_to_patient = train.drop_duplicates("eeg_id")[["eeg_id", "patient_id"]].set_index(
    "eeg_id"
)["patient_id"]
vote_prob_by_eeg_with_patient = vote_prob_by_eeg.join(eeg_to_patient, how="left")
vote_prob_by_patient = vote_prob_by_eeg_with_patient.groupby("patient_id")[
    CLASSES
].mean()

eeg_to_spec = train.drop_duplicates("eeg_id")[["eeg_id", "spectrogram_id"]].set_index(
    "eeg_id"
)["spectrogram_id"]
vote_prob_by_eeg_with_spec = vote_prob_by_eeg.join(eeg_to_spec, how="left")
vote_prob_by_spec = vote_prob_by_eeg_with_spec.groupby("spectrogram_id")[CLASSES].mean()


def _dirichlet_smooth(p, strength=0.30):
    p = np.asarray(p, dtype=np.float32)
    p = np.clip(p, 1e-12, 1e12)
    p = p / p.sum()
    u = np.full_like(p, 1.0 / p.size)
    p = (1.0 - strength) * p + strength * u
    p = np.clip(p, 1e-8, 1.0)
    return p / p.sum()


global_prior = _dirichlet_smooth(global_prior, strength=0.20)

patient_prior_map = {}
for pid, row in vote_prob_by_patient.iterrows():
    p = row.values.astype(np.float32)
    if not np.isfinite(p).all() or p.sum() <= 0:
        continue
    patient_prior_map[int(pid)] = _dirichlet_smooth(p / p.sum(), strength=0.20)

spec_prior_map = {}
for sid, row in vote_prob_by_spec.iterrows():
    p = row.values.astype(np.float32)
    if not np.isfinite(p).all() or p.sum() <= 0:
        continue
    try:
        spec_prior_map[int(sid)] = _dirichlet_smooth(p / p.sum(), strength=0.20)
    except Exception:
        continue

print("Global prior:", dict(zip(CLASSES, global_prior.tolist())))
print("Patient priors available:", len(patient_prior_map))
print("Spectrogram priors available:", len(spec_prior_map))


def _find_any_checkpoint(root="/kaggle/input"):
    exts = (".pth", ".pt", ".bin")
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(exts):
                return os.path.join(dirpath, fn)
    return None


def _checkpoint_looks_compatible(ckpt_path: str) -> bool:
    try:
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict):
            return False
        keys = list(state.keys())
        expected_prefixes = (
            "spec_model.",
            "eeg_model.",
            "raw_50s_model.",
            "raw_10s_model.",
            "head.",
        )
        return any(any(k.startswith(p) for p in expected_prefixes) for k in keys)
    except Exception:
        return False


ckpt_path = _find_any_checkpoint("/kaggle/input")
USE_MODEL_INFERENCE = ckpt_path is not None and _checkpoint_looks_compatible(ckpt_path)
print("Checkpoint found:", ckpt_path)
print("Checkpoint looks compatible:", USE_MODEL_INFERENCE)

result_5 = {}

if USE_MODEL_INFERENCE:
    test_unique = test.drop_duplicates("eeg_id")[["eeg_id", "spectrogram_id"]].copy()
    test_unique["eeg_id"] = test_unique["eeg_id"].astype(str)
    test_unique["spectrogram_id"] = test_unique["spectrogram_id"].astype(str)

    rows = [
        (str(r.eeg_id), str(r.spectrogram_id), TEST_SPEC_PATH, TEST_EEG_PATH)
        for r in test_unique.itertuples(index=False)
    ]

    n_jobs = min(8, os.cpu_count() or 1)
    _ = Parallel(n_jobs=n_jobs, prefer="threads", batch_size=16)(
        delayed(save_from_paths)(eeg_id, spec_id, sp, ep)
        for (eeg_id, spec_id, sp, ep) in rows
    )

    model = Net("vit_base_patch14_reg4_dinov2.lvd142m", device_id=0).to(device)

    try:
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict):
            model.load_state_dict(state, strict=False)
        print("Loaded checkpoint with strict=False.")
    except Exception as e:
        print("Checkpoint load failed; falling back to prior-only. Error:", repr(e))
        USE_MODEL_INFERENCE = False

if USE_MODEL_INFERENCE:
    model.eval()

    test_inf_df = test.drop_duplicates("eeg_id")[
        ["eeg_id", "spectrogram_id", "patient_id"]
    ].copy()
    test_inf_df["eeg_id"] = test_inf_df["eeg_id"].astype(str)

    test_data = ImageFolder(test_inf_df, (224, 224), targets=None)
    num_workers = min(4, os.cpu_count() or 1)
    pin = torch.cuda.is_available()

    test_loader = DataLoader(
        test_data,
        batch_size=16,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        in_order=True,
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
            spec_imgs = spec_imgs.to(device, non_blocking=True).float()
            eeg_imgs = eeg_imgs.to(device, non_blocking=True).float()
            raw_50s_imgs = raw_50s_imgs.to(device, non_blocking=True).float()
            raw_10s_l_imgs = raw_10s_l_imgs.to(device, non_blocking=True).float()
            raw_10s_c_imgs = raw_10s_c_imgs.to(device, non_blocking=True).float()
            raw_10s_r_imgs = raw_10s_r_imgs.to(device, non_blocking=True).float()

            logits_l, _, _, _, _ = model(
                spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_l_imgs
            )
            logits_c, _, _, _, _ = model(
                spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_c_imgs
            )
            logits_r, _, _, _, _ = model(
                spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_r_imgs
            )
            probs = (
                logits_l.softmax(1) + logits_c.softmax(1) + logits_r.softmax(1)
            ) / 3.0
            probs = probs.detach().cpu().numpy().astype(np.float32, copy=False)

            probs = np.clip(probs, 1e-8, 1e8)
            probs = probs / probs.sum(axis=1, keepdims=True)

            for j in range(len(eeg_ids)):
                result_5[str(eeg_ids[j])] = probs[j]

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()




## === cell 13
sample_sub = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
)
for c in CLASSES:
    if c not in sample_sub.columns:
        raise ValueError(f"Missing required submission column: {c}")

_test_dedup = test.drop_duplicates("eeg_id")[
    ["eeg_id", "patient_id", "spectrogram_id"]
].copy()
_test_dedup["eeg_id"] = _test_dedup["eeg_id"].astype(str)
test_pid_map = _test_dedup.set_index("eeg_id")["patient_id"].to_dict()
test_sid_map = _test_dedup.set_index("eeg_id")["spectrogram_id"].to_dict()


def _get_fallback_prior_for_eeg(eeg_id: str) -> np.ndarray:
    pid = test_pid_map.get(eeg_id, None)
    if pid is not None:
        pid = int(pid)
    sid = test_sid_map.get(eeg_id, None)
    if sid is not None:
        try:
            sid = int(sid)
        except Exception:
            sid = None

    p_patient = patient_prior_map.get(pid, None)
    p_spec = spec_prior_map.get(sid, None)

    w_patient = 0.70 if p_patient is not None else 0.0
    w_spec = 0.20 if p_spec is not None else 0.0
    w_global = 1.0 - w_patient - w_spec

    out = w_global * global_prior.copy()
    if p_patient is not None:
        out += w_patient * p_patient
    if p_spec is not None:
        out += w_spec * p_spec
    out = np.clip(out, 1e-8, 1e8)
    out = out / out.sum()
    return out.astype(np.float32, copy=False)


pred_mat = np.zeros((len(sample_sub), 6), dtype=np.float32)
missing = 0
for i, eeg_id in enumerate(sample_sub["eeg_id"].astype(str).values):
    if eeg_id in result_5:
        pred_mat[i] = result_5[eeg_id]
    else:
        missing += 1
        pred_mat[i] = _get_fallback_prior_for_eeg(eeg_id)

if len(result_5) == 0:
    print(
        "No model predictions available; writing patient+spectrogram+global prior submission."
    )
    for i, eeg_id in enumerate(sample_sub["eeg_id"].astype(str).values):
        pred_mat[i] = _get_fallback_prior_for_eeg(eeg_id)
    missing = len(sample_sub)

w_global_mix = 0.03
pred_mat = (1.0 - w_global_mix) * pred_mat + w_global_mix * global_prior[None, :]

u_strength = 0.01
pred_mat = (1.0 - u_strength) * pred_mat + u_strength * (1.0 / N_CLASSES)

pred_mat = np.clip(pred_mat, 1e-8, 1e8)
pred_mat = pred_mat / pred_mat.sum(axis=1, keepdims=True)

sub = sample_sub.copy()
sub[CLASSES] = pred_mat
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv. Missing preds filled with priors:", missing)
print(sub.head())




## === cell 14
if DEBUG == False:
    for p in [
        "/kaggle/working/spec_spectrograms/",
        "/kaggle/working/eeg_spectrograms/",
        "/kaggle/working/eeg_50s_raws/",
        "/kaggle/working/eeg_10s_raws/",
        "/kaggle/working/squeezeformer",
    ]:
        if os.path.exists(p):
            try:
                import shutil

                shutil.rmtree(p)
            except Exception as e:
                print("Cleanup failed for", p, "with", repr(e))
