# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.2986156254854332

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the immediate runtime blocker by correcting the bad model weight path (it currently has a double leading slash) and by adding a safe fallback that still produces valid probabilities if the weight file isn’t available in this environment. I also ensure `result_5` is always defined so the submission cell can’t crash, and I keep the prediction semantics unchanged (softmax probabilities averaged across L/C/R). Finally, I make the submission `eeg_id` dtype consistent with how keys are stored and always renormalize probabilities to sum to 1 to satisfy Kaggle’s submission validator.'
- What this solution (achieved 1.40995) has done: 'The timeout is dominated by heavy per-file preprocessing (reading ~10k parquet files, computing multiple spectrograms + multiple filtfilt passes, and then resizing six arrays per EEG using torch interpolate) plus redundant disk I/O. I keep the exact same features/logic but remove unnecessary intermediate saves, cache existence checks to avoid repeated filesystem calls, and (most importantly) batch the 6 resize operations into one torch interpolate call per EEG to reduce overhead drastically. I also speed up parquet reading by selecting only needed columns (both for spectrogram and EEG) and avoid repeated DataFrame indexing work in the raw feature extractor while preserving identical math. Finally, I tune parallelism/batching to reduce joblib overhead and add `inference_mode()` for a small inference speedup without changing outputs.'
- What this solution (achieved 1.40995) has done: 'The timeout is dominated by CPU feature generation and resizing for 9,850 test EEGs: each sample computes multiple spectrograms/filters and then performs six 518×518 bilinear resizes, all currently done via pandas-heavy paths and re-reading full parquet columns. I keep the exact same features/models/inference semantics, but remove redundant work by (1) reading only the needed spectrogram time window directly (first 300 columns) instead of loading the whole parquet, (2) eliminating pandas overhead inside the EEG feature functions by converting once to a single NumPy matrix and reusing it for raw10/raw50/STFT, (3) reusing index mappings and avoiding repeated dict construction per file, and (4) resizing the 6 images in one batched `torch.nn.functional.interpolate` call (same bilinear math) instead of 6 separate calls. These changes are provably equivalent (same data slices, same transforms, same interpolation mode/params) and primarily cut constant factors and I/O, helping the script finish within 600 seconds.'
- What this solution (achieved 1.40995) has done: 'I fix the `FileNotFoundError` by making feature generation verify that all resized `.npy` files exist for every `eeg_id`, and if any are missing, regenerate them serially (this avoids silent failures from the parallel step). I also make `save_one` return and log failures more explicitly in `DEBUG` mode, and I make the dataset robust by regenerating missing cached files on-the-fly as a last resort (keeps core logic identical, just prevents crashes). This should both produce a valid `submission.csv` end-to-end and improve the score vs. the current uniform-fallback rows (which are likely causing the 1.40995). Model/inference semantics (3-crop L/C/R average of softmax outputs) are unchanged.'

# 9. Code solution

## === cell 0
import os
import gc
import random
import warnings

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
def stft_spec_from_eeg_np(X_stft: np.ndarray):
    """
    X_stft: numpy array shaped (T, n_stft_cols) for columns _ALL_STFT_COLS in that order.
    Preserves original stft_spec_from_eeg_df logic; only avoids pandas overhead.
    """
    EEG_LENGTH = 50

    time_temp = 0
    time_start = round(time_temp + (50 - EEG_LENGTH) / 2 * 200)
    time_stop = round(time_temp + (50 + EEG_LENGTH) / 2 * 200)
    X_stft = X_stft[time_start:time_stop]

    if np.isnan(X_stft).any():
        col_means = np.nanmean(X_stft, axis=0)
        inds = np.where(np.isnan(X_stft))
        X_stft[inds] = col_means[inds[1]]

    list_eeg = []
    fs = 200
    nperseg = 70
    noverlap = 0

    for k in range(4):
        idx = _FEATS_IDXS[k]
        X = X_stft[:, idx].astype(np.float32, copy=False)  # (T,5)
        diffs = (X[:, :-1] - X[:, 1:]).T  # (4,T)

        img = np.zeros((128, 142, 4), dtype="float32")
        for kk in range(4):
            new_eeg = diffs[kk]
            f, t, spec = signal.spectrogram(
                new_eeg, fs, nperseg=nperseg, noverlap=noverlap, nfft=256
            )
            spec = np.abs(spec)
            spec = np.log1p(spec).astype("float32", copy=False)
            img[:, :, kk] += spec[:128, :]

        img = np.concatenate(
            (img[:, :, 0], img[:, :, 1], img[:, :, 2], img[:, :, 3]), 1
        )
        list_eeg.append(img)

    img = np.concatenate(list_eeg, 0)
    img /= 2.0
    return img


SFREQ = 200
filter_range = [0.5, 40]

RAW_FEATS = {
    "LL": ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1"],
    "RL": ["Fp2-F8", "F8-T4", "T4-T6", "T6-O2"],
    "LP": ["Fp1-F3", "F3-C3", "C3-P3", "P3-O1"],
    "RP": ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2"],
}

sos = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass", output="sos")

_RAW_REGION_PRECOMP = {}
_ALL_RAW_COLS = []
_all_raw_set = set()
for region, chans in RAW_FEATS.items():
    c1_list = [c.split("-")[0] for c in chans]
    c2_list = [c.split("-")[1] for c in chans]
    cols = list(dict.fromkeys(c1_list + c2_list))
    _RAW_REGION_PRECOMP[region] = (cols, c1_list, c2_list)
    for c in cols:
        if c not in _all_raw_set:
            _all_raw_set.add(c)
            _ALL_RAW_COLS.append(c)

_ALL_STFT_COLS = list(dict.fromkeys([c for grp in FEATS for c in grp]))
_EEG_READ_COLS = list(dict.fromkeys(_ALL_RAW_COLS + _ALL_STFT_COLS))

_RAW_COL_IDX = {c: i for i, c in enumerate(_ALL_RAW_COLS)}
_STFT_COL_IDX = {c: i for i, c in enumerate(_ALL_STFT_COLS)}
_FEATS_IDXS = [[_STFT_COL_IDX[c] for c in FEATS[k]] for k in range(4)]


def _raw_segment_from_eeg_np(
    X_all: np.ndarray, time_start: int, time_stop: int, EEG_LENGTH: int
):
    Xseg = X_all[time_start:time_stop]  # view; shape (200*EEG_LENGTH, ncols)

    list_eeg = []
    for _, (_, c1_list, c2_list) in _RAW_REGION_PRECOMP.items():
        A = Xseg[:, [_RAW_COL_IDX[c] for c in c1_list]].T  # (4,T)
        B = Xseg[:, [_RAW_COL_IDX[c] for c in c2_list]].T  # (4,T)

        eeg = A - B  # (4,T)
        eeg = signal.sosfiltfilt(sos, eeg, axis=1)
        eeg = np.clip(eeg, -1024, 1024).astype(np.float32, copy=False)

        eeg = np.reshape(eeg, (4, 200, EEG_LENGTH))
        eeg = np.concatenate(
            [eeg[0, :, :], eeg[1, :, :], eeg[2, :, :], eeg[3, :, :]], 1
        )
        list_eeg.append(eeg)

    eeg_out = np.concatenate(list_eeg, 1)
    eeg_out /= 104
    return eeg_out


def _nanfill_inplace(X_all: np.ndarray):
    if np.isnan(X_all).any():
        col_means = np.nanmean(X_all, axis=0)
        inds = np.where(np.isnan(X_all))
        X_all[inds] = col_means[inds[1]]


def raw10seeg_from_eeg_np(X_raw: np.ndarray):
    EEG_LENGTH = 10

    time_temp = 0
    time_start = round(time_temp + (50 - EEG_LENGTH) / 2 * 200)
    time_stop = round(time_temp + (50 + EEG_LENGTH) / 2 * 200)
    eeg_c = _raw_segment_from_eeg_np(X_raw, time_start, time_stop, EEG_LENGTH)

    time_temp = 0
    time_start = round(time_temp + 18 * 200)
    time_stop = round(time_temp + 28 * 200)
    eeg_l = _raw_segment_from_eeg_np(X_raw, time_start, time_stop, EEG_LENGTH)

    time_temp = 0
    time_start = round(time_temp + 22 * 200)
    time_stop = round(time_temp + 32 * 200)
    eeg_r = _raw_segment_from_eeg_np(X_raw, time_start, time_stop, EEG_LENGTH)

    return eeg_l, eeg_c, eeg_r


def raw50seeg_from_eeg_np(X_raw: np.ndarray):
    EEG_LENGTH = 50
    time_temp = 0
    time_start = round(time_temp + (50 - EEG_LENGTH) / 2 * 200)
    time_stop = round(time_temp + (50 + EEG_LENGTH) / 2 * 200)
    return _raw_segment_from_eeg_np(X_raw, time_start, time_stop, EEG_LENGTH)




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

print(test.shape)

resized_directory_path = "resized_518/"
os.makedirs(resized_directory_path, exist_ok=True)

from joblib import Parallel, delayed




## === cell 6
def _resized_cached(eeg_id: str):
    base = resized_directory_path
    return (
        os.path.exists(f"{base}{eeg_id}_spec.npy")
        and os.path.exists(f"{base}{eeg_id}_eeg.npy")
        and os.path.exists(f"{base}{eeg_id}_raw50.npy")
        and os.path.exists(f"{base}{eeg_id}_raw10l.npy")
        and os.path.exists(f"{base}{eeg_id}_raw10c.npy")
        and os.path.exists(f"{base}{eeg_id}_raw10r.npy")
    )


@torch.inference_mode()
def _resize6(arrs2d, out_hw=(518, 518)) -> list:
    x = np.stack(arrs2d, axis=0).astype(np.float32, copy=False)  # (6,H,W)
    xt = torch.from_numpy(x).unsqueeze(1)  # (6,1,H,W)
    yt = F.interpolate(xt, size=out_hw, mode="bilinear", align_corners=False)
    y = yt.squeeze(1).cpu().numpy().astype("float32", copy=False)  # (6,H,W)
    return [y[i] for i in range(y.shape[0])]


def _resize6_save(
    eeg_id: str,
    split_spec_arr: np.ndarray,
    imgstft: np.ndarray,
    img50: np.ndarray,
    img_l: np.ndarray,
    img_c: np.ndarray,
    img_r: np.ndarray,
    out_hw=(518, 518),
):
    base = resized_directory_path
    r0, r1, r2, r3, r4, r5 = _resize6(
        [split_spec_arr, imgstft, img50, img_l, img_c, img_r], out_hw=out_hw
    )
    np.save(f"{base}{eeg_id}_spec.npy", r0)
    np.save(f"{base}{eeg_id}_eeg.npy", r1)
    np.save(f"{base}{eeg_id}_raw50.npy", r2)
    np.save(f"{base}{eeg_id}_raw10l.npy", r3)
    np.save(f"{base}{eeg_id}_raw10c.npy", r4)
    np.save(f"{base}{eeg_id}_raw10r.npy", r5)


def _read_split_spec_arr(spec_path: str) -> np.ndarray:
    spec = pd.read_parquet(spec_path)
    arr = spec.values[:, 1:301].T.astype("float32", copy=False)
    return arr


def save_one(eeg_id, spec_id, out_hw=(518, 518), verbose=False):
    """
    Bugfix: make feature generation robust so missing cached files don't crash inference.
    - Returns 1 only when all 6 npy outputs exist after processing.
    - If it fails, we return 0 so a later serial retry can regenerate.
    """
    eeg_id = str(eeg_id)
    spec_id = str(spec_id)

    if _resized_cached(eeg_id):
        return 1

    try:
        split_spec_arr = _read_split_spec_arr(f"{SPEC_PATH}{spec_id}.parquet")

        eeg_df = pd.read_parquet(f"{EEG_PATH}{eeg_id}.parquet", columns=_EEG_READ_COLS)
        X_all = eeg_df.to_numpy(dtype=np.float32, copy=True)

        raw_idx = [eeg_df.columns.get_loc(c) for c in _ALL_RAW_COLS]
        stft_idx = [eeg_df.columns.get_loc(c) for c in _ALL_STFT_COLS]

        X_raw = X_all[:, raw_idx]
        _nanfill_inplace(X_raw)

        X_stft = X_all[:, stft_idx]

        img_l, img_c, img_r = raw10seeg_from_eeg_np(X_raw)
        img50 = raw50seeg_from_eeg_np(X_raw)
        imgstft = stft_spec_from_eeg_np(X_stft)

        _resize6_save(
            eeg_id, split_spec_arr, imgstft, img50, img_l, img_c, img_r, out_hw=out_hw
        )
        return 1 if _resized_cached(eeg_id) else 0
    except Exception as e:
        if verbose or DEBUG:
            print(f"[save_one] failed eeg_id={eeg_id} spec_id={spec_id} err={repr(e)}")
        return 0


uniq = test.drop_duplicates("eeg_id")[["eeg_id", "spectrogram_id"]].reset_index(
    drop=True
)

n_jobs = min(8, os.cpu_count() or 4)
ok_flags = Parallel(n_jobs=n_jobs, backend="loky", batch_size=128, prefer="processes")(
    delayed(save_one)(row.eeg_id, row.spectrogram_id)
    for row in uniq.itertuples(index=False)
)

failed_idx = [i for i, f in enumerate(ok_flags) if f != 1]
if len(failed_idx) > 0:
    if DEBUG:
        print("Parallel feature generation had failures:", len(failed_idx))
    for i in failed_idx:
        row = uniq.iloc[i]
        _ = save_one(row.eeg_id, row.spectrogram_id, verbose=DEBUG)

missing = []
for row in uniq.itertuples(index=False):
    if not _resized_cached(str(row.eeg_id)):
        missing.append(str(row.eeg_id))
if len(missing) > 0:
    print(
        f"WARNING: {len(missing)} eeg_ids still missing cached features; "
        f"they will be handled with uniform probabilities during submission."
    )
else:
    print("All cached feature files present for inference.")




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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(Config.seed)



## === cell 9
import timm
import torch.utils.data as data
from torch.utils.data import DataLoader




## === cell 10
class ImageFolder(data.Dataset):
    def __init__(self, df, test_imgsize):
        super().__init__()
        self.resized_path = resized_directory_path
        self.df = df.reset_index(drop=True)
        self.test_imgsize = tuple(test_imgsize)

        self._id2spec = {
            str(r.eeg_id): str(r.spectrogram_id)
            for r in df.drop_duplicates("eeg_id")[
                ["eeg_id", "spectrogram_id"]
            ].itertuples(index=False)
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        """
        Bugfix: if any cached file is missing (e.g., parallel save failed),
        regenerate for this eeg_id on-the-fly to prevent DataLoader FileNotFoundError.
        Core feature logic is unchanged (calls the same save_one()).
        """
        row = self.df.loc[index]
        eeg_id = str(row.eeg_id)

        if not _resized_cached(eeg_id):
            spec_id = self._id2spec.get(eeg_id)
            if spec_id is not None:
                _ = save_one(eeg_id, spec_id, verbose=False)

        spec_fp = os.path.join(self.resized_path, eeg_id + "_spec.npy")
        eeg_fp = os.path.join(self.resized_path, eeg_id + "_eeg.npy")
        raw50_fp = os.path.join(self.resized_path, eeg_id + "_raw50.npy")
        raw10l_fp = os.path.join(self.resized_path, eeg_id + "_raw10l.npy")
        raw10c_fp = os.path.join(self.resized_path, eeg_id + "_raw10c.npy")
        raw10r_fp = os.path.join(self.resized_path, eeg_id + "_raw10r.npy")

        spec_img = np.load(spec_fp, mmap_mode="r")
        eeg_img = np.load(eeg_fp, mmap_mode="r")
        raw_50s_img = np.load(raw50_fp, mmap_mode="r")
        raw_10s_l_img = np.load(raw10l_fp, mmap_mode="r")
        raw_10s_c_img = np.load(raw10c_fp, mmap_mode="r")
        raw_10s_r_img = np.load(raw10r_fp, mmap_mode="r")

        eeg_img = eeg_img[..., None]
        spec_img = spec_img[..., None]
        raw_50s_img = raw_50s_img[..., None]
        raw_10s_l_img = raw_10s_l_img[..., None]
        raw_10s_c_img = raw_10s_c_img[..., None]
        raw_10s_r_img = raw_10s_r_img[..., None]

        eps = 1e-6
        spec_img = np.clip(spec_img, np.exp(-4), np.exp(8))
        spec_img = np.log(spec_img)
        spec_img = np.nan_to_num(spec_img, nan=0.0)

        img_mean = spec_img.mean(axis=(0, 1))
        img_std = spec_img.std(axis=(0, 1))
        spec_img = (spec_img - img_mean) / (img_std + eps)

        return (
            spec_img,
            eeg_img,
            raw_50s_img,
            raw_10s_l_img,
            raw_10s_c_img,
            raw_10s_r_img,
            eeg_id,
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
        )
        self.eeg_model = timm.create_model(
            "vit_small_patch14_reg4_dinov2.lvd142m",
            num_classes=6,
            pretrained=False,
            in_chans=1,
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
device = "cuda:0" if torch.cuda.is_available() else "cpu"

vit_models = []

model_weights = [
    "/kaggle/input/hms_multi_modals/pytorch/vit_base/1/hms_multi_modals.pth"
]
model_types = ["vit_base"]

for i in range(len(model_types)):
    if model_types[i] == "vit_base":
        print("Requested weights:", model_weights[i])
        model = Net("vit_base_patch14_reg4_dinov2.lvd142m", device).to(device)

        if os.path.exists(model_weights[i]):
            state = torch.load(model_weights[i], map_location=device)
            model.load_state_dict(state, strict=True)
            model.eval()
            vit_models.append(model)
            print("Loaded custom checkpoint.")
        else:
            print(
                "WARNING: custom weight file not found. Falling back to timm pretrained backbone weights."
            )
            model.spec_model = timm.create_model(
                "vit_small_patch14_reg4_dinov2.lvd142m",
                num_classes=6,
                pretrained=True,
                in_chans=1,
            ).to(device)
            model.eeg_model = timm.create_model(
                "vit_small_patch14_reg4_dinov2.lvd142m",
                num_classes=6,
                pretrained=True,
                in_chans=1,
            ).to(device)
            model.raw_50s_model = timm.create_model(
                "vit_base_patch14_reg4_dinov2.lvd142m",
                num_classes=6,
                pretrained=True,
                in_chans=1,
            ).to(device)
            model.raw_10s_model = timm.create_model(
                "vit_base_patch14_reg4_dinov2.lvd142m",
                num_classes=6,
                pretrained=True,
                in_chans=1,
            ).to(device)

            for m in [
                model.spec_model,
                model.eeg_model,
                model.raw_50s_model,
                model.raw_10s_model,
            ]:
                m.fc_norm = nn.Identity()
                m.head_drop = nn.Identity()
                m.head = nn.Identity()

            model.eval()
            vit_models.append(model)

test_data = ImageFolder(test, (518, 518))

num_workers = min(4, os.cpu_count() or 2)
test_loader = DataLoader(
    test_data,
    batch_size=32,
    pin_memory=torch.cuda.is_available(),
    num_workers=num_workers,
    drop_last=False,
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
)

result_5 = {}

with torch.inference_mode():
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

        if len(vit_models) == 0:
            ensemble_probs = torch.full(
                (spec_imgs.shape[0], 6), 1.0 / 6.0, device=device
            )
        else:
            ensemble_probs = torch.zeros((spec_imgs.shape[0], 6), device=device)
            for model in vit_models:
                model.eval()
                logits_l, _, _, _, _ = model(
                    spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_l_imgs
                )
                logits_c, _, _, _, _ = model(
                    spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_c_imgs
                )
                logits_r, _, _, _, _ = model(
                    spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_r_imgs
                )
                probs_l = logits_l.softmax(dim=1)
                probs_c = logits_c.softmax(dim=1)
                probs_r = logits_r.softmax(dim=1)
                ensemble_probs += (probs_l + probs_c + probs_r) / 3.0
            ensemble_probs /= len(vit_models)

        ensemble_probs_np = (
            ensemble_probs.detach().cpu().numpy().astype(np.float64, copy=False)
        )
        for eeg_id, p in zip(list(eeg_ids), ensemble_probs_np):
            eeg_id = str(eeg_id)
            prev = result_5.get(eeg_id)
            if prev is None:
                result_5[eeg_id] = p.copy()
            else:
                prev += p



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1769249155.py in <cell line: 0>()
     78 # Note: core inference logic unchanged; only made upstream caching robust to avoid missing-file crashes.
     79 with torch.inference_mode():
---> 80     for (
     81         spec_imgs,
     82         eeg_imgs,

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1478                 del self._task_info[idx]
   1479                 self._rcvd_idx += 1
-> 1480                 return self._process_data(data)
   1481 
   1482     def _try_put_index(self):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _process_data(self, data)
   1503         self._try_put_index()
   1504         if isinstance(data, ExceptionWrapper):
-> 1505             data.reraise()
   1506         return data
   1507 

/usr/local/lib/python3.11/dist-packages/torch/_utils.py in reraise(self)
    731             # instantiate since we don't know how to
    732             raise RuntimeError(msg) from None
--> 733         raise exception
    734 
    735 

FileNotFoundError: Caught FileNotFoundError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
            ~~~~~~~~~~~~^^^^^
  File "/tmp/ipykernel_55/3438895045.py", line 41, in __getitem__
    spec_img = np.load(spec_fp, mmap_mode="r")
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/numpy/lib/npyio.py", line 427, in load
    fid = stack.enter_context(open(os_fspath(file), "rb"))
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: 'resized_518/2578018731_spec.npy'


## === cell 13
for model in vit_models:
    del model
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()



## === cell 14
sample_sub = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
)

rows = []
for eeg_id in sample_sub["eeg_id"].astype(str).tolist():
    if eeg_id in result_5:
        r = result_5[eeg_id]
    else:
        r = np.ones((6,), dtype=np.float64) / 6.0
    rows.append([eeg_id, r[0], r[1], r[2], r[3], r[4], r[5]])

df = pd.DataFrame(rows, columns=["eeg_id"] + CLASSES)

probs = df[CLASSES].to_numpy(dtype=np.float64)
probs = np.clip(probs, 1e-12, None)
probs = probs / probs.sum(axis=1, keepdims=True)
df[CLASSES] = probs

df.to_csv("submission.csv", index=False)
print(df.head())
print("Saved submission.csv with shape:", df.shape)



## === cell 15
if DEBUG == False:
    import shutil

    for p in [
        "/kaggle/working/resized_518/",
    ]:
        if os.path.exists(p):
            shutil.rmtree(p, ignore_errors=True)
