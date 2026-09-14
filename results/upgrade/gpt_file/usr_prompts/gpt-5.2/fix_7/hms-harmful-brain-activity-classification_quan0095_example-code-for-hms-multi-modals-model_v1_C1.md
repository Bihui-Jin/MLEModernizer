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
def stft_spec_from_eeg_df(eeg: pd.DataFrame):
    EEG_LENGTH = 50

    time_temp = 0
    time_start = round(time_temp + (50 - EEG_LENGTH) / 2 * 200)
    time_stop = round(time_temp + (50 + EEG_LENGTH) / 2 * 200)
    eeg = eeg.iloc[time_start:time_stop]

    list_eeg = []
    fs = 200
    nperseg = 70
    noverlap = 0
    for k in range(4):
        COLS = FEATS[k]  # 5 cols -> 4 diffs
        X = eeg[COLS].to_numpy(dtype=np.float32, copy=True)  # (T,5)
        col_means = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        if inds[0].size:
            X[inds] = col_means[inds[1]]
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

b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")


_RAW_REGION_PRECOMP = {}
_ALL_RAW_COLS = set()
for region, chans in RAW_FEATS.items():
    c1_list = [c.split("-")[0] for c in chans]
    c2_list = [c.split("-")[1] for c in chans]
    cols = list(dict.fromkeys(c1_list + c2_list))
    _RAW_REGION_PRECOMP[region] = (cols, c1_list, c2_list)
    _ALL_RAW_COLS.update(cols)

_ALL_STFT_COLS = list(dict.fromkeys([c for grp in FEATS for c in grp]))


def _raw_segment_from_eeg_np(
    X_all: np.ndarray, col_idx: dict, time_start: int, time_stop: int, EEG_LENGTH: int
):
    Xseg = X_all[time_start:time_stop]  # view; shape (200*EEG_LENGTH, ncols)

    list_eeg = []
    for region, (cols, c1_list, c2_list) in _RAW_REGION_PRECOMP.items():
        A = Xseg[:, [col_idx[c] for c in c1_list]].T  # (4,T)
        B = Xseg[:, [col_idx[c] for c in c2_list]].T  # (4,T)

        eeg = A - B  # (4,T)
        eeg = signal.filtfilt(b, a, eeg, axis=1)
        eeg = np.clip(eeg, -1024, 1024).astype(np.float32, copy=False)

        eeg = np.reshape(eeg, (4, 200, EEG_LENGTH))
        eeg = np.concatenate(
            [eeg[0, :, :], eeg[1, :, :], eeg[2, :, :], eeg[3, :, :]], 1
        )
        list_eeg.append(eeg)

    eeg_out = np.concatenate(list_eeg, 1)
    eeg_out /= 104
    return eeg_out


def raw10seeg_from_eeg_df(raw_eeg: pd.DataFrame):
    EEG_LENGTH = 10

    X_all = raw_eeg[list(_ALL_RAW_COLS)].to_numpy(dtype=np.float32, copy=True)
    col_means = np.nanmean(X_all, axis=0)
    inds = np.where(np.isnan(X_all))
    if inds[0].size:
        X_all[inds] = col_means[inds[1]]
    cols = list(_ALL_RAW_COLS)
    col_idx = {c: i for i, c in enumerate(cols)}

    time_temp = 0
    time_start = round(time_temp + (50 - EEG_LENGTH) / 2 * 200)
    time_stop = round(time_temp + (50 + EEG_LENGTH) / 2 * 200)
    eeg_c = _raw_segment_from_eeg_np(X_all, col_idx, time_start, time_stop, EEG_LENGTH)

    time_temp = 0
    time_start = round(time_temp + 18 * 200)
    time_stop = round(time_temp + 28 * 200)
    eeg_l = _raw_segment_from_eeg_np(X_all, col_idx, time_start, time_stop, EEG_LENGTH)

    time_temp = 0
    time_start = round(time_temp + 22 * 200)
    time_stop = round(time_temp + 32 * 200)
    eeg_r = _raw_segment_from_eeg_np(X_all, col_idx, time_start, time_stop, EEG_LENGTH)

    return eeg_l, eeg_c, eeg_r


def raw50seeg_from_eeg_df(raw_eeg: pd.DataFrame):
    EEG_LENGTH = 50
    X_all = raw_eeg[list(_ALL_RAW_COLS)].to_numpy(dtype=np.float32, copy=True)
    col_means = np.nanmean(X_all, axis=0)
    inds = np.where(np.isnan(X_all))
    if inds[0].size:
        X_all[inds] = col_means[inds[1]]
    cols = list(_ALL_RAW_COLS)
    col_idx = {c: i for i, c in enumerate(cols)}

    time_temp = 0
    time_start = round(time_temp + (50 - EEG_LENGTH) / 2 * 200)
    time_stop = round(time_temp + (50 + EEG_LENGTH) / 2 * 200)
    return _raw_segment_from_eeg_np(X_all, col_idx, time_start, time_stop, EEG_LENGTH)




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

spec_directory_path = "spec_spectrograms/"
os.makedirs(spec_directory_path, exist_ok=True)

eeg_directory_path = "eeg_spectrograms/"
os.makedirs(eeg_directory_path, exist_ok=True)

raw_10s_directory_path = "eeg_10s_raws/"
os.makedirs(raw_10s_directory_path, exist_ok=True)

raw_50s_directory_path = "eeg_50s_raws/"
os.makedirs(raw_50s_directory_path, exist_ok=True)

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
    tensors = [
        torch.from_numpy(split_spec_arr),
        torch.from_numpy(imgstft),
        torch.from_numpy(img50),
        torch.from_numpy(img_l),
        torch.from_numpy(img_c),
        torch.from_numpy(img_r),
    ]
    xt = torch.stack(tensors, dim=0).unsqueeze(1)  # (6,1,H,W)
    xt = F.interpolate(xt, size=out_hw, mode="bilinear", align_corners=False)
    arr = xt.squeeze(1).cpu().numpy().astype("float32", copy=False)

    base = resized_directory_path
    np.save(f"{base}{eeg_id}_spec", arr[0])
    np.save(f"{base}{eeg_id}_eeg", arr[1])
    np.save(f"{base}{eeg_id}_raw50", arr[2])
    np.save(f"{base}{eeg_id}_raw10l", arr[3])
    np.save(f"{base}{eeg_id}_raw10c", arr[4])
    np.save(f"{base}{eeg_id}_raw10r", arr[5])


def save_one(eeg_id, spec_id, out_hw=(518, 518)):
    eeg_id = str(eeg_id)
    spec_id = str(spec_id)

    if _resized_cached(eeg_id):
        return

    spec = pd.read_parquet(f"{SPEC_PATH}{spec_id}.parquet")
    spec_arr = spec.values[:, 1:].T.astype("float32", copy=False)  # (Hz, Time)
    split_spec_arr = spec_arr[:, 0:300]

    eeg_cols = list(dict.fromkeys(list(_ALL_RAW_COLS) + _ALL_STFT_COLS))
    eeg_df = pd.read_parquet(f"{EEG_PATH}{eeg_id}.parquet", columns=eeg_cols)

    img_l, img_c, img_r = raw10seeg_from_eeg_df(eeg_df)
    img50 = raw50seeg_from_eeg_df(eeg_df)
    imgstft = stft_spec_from_eeg_df(eeg_df)

    _resize6_save(
        eeg_id, split_spec_arr, imgstft, img50, img_l, img_c, img_r, out_hw=out_hw
    )


uniq = test.drop_duplicates("eeg_id")[["eeg_id", "spectrogram_id"]].reset_index(
    drop=True
)

n_jobs = min(4, os.cpu_count() or 4)
_ = Parallel(n_jobs=n_jobs, backend="loky", batch_size=32)(
    delayed(save_one)(row.eeg_id, row.spectrogram_id)
    for row in uniq.itertuples(index=False)
)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
_RemoteTraceback                          Traceback (most recent call last)
_RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/joblib/externals/loky/process_executor.py", line 490, in _process_worker
    r = call_item()
        ^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/joblib/externals/loky/process_executor.py", line 291, in __call__
    return self.fn(*self.args, **self.kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/joblib/parallel.py", line 607, in __call__
    return [func(*args, **kwargs) for func, args, kwargs in self.items]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/joblib/parallel.py", line 607, in <listcomp>
    return [func(*args, **kwargs) for func, args, kwargs in self.items]
            ^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_55/3851630160.py", line 70, in save_one
  File "/tmp/ipykernel_55/3851630160.py", line 37, in _resize6_save
RuntimeError: stack expects each tensor to be equal size, but got [400, 300] at entry 0 and [512, 568] at entry 1
"""

The above exception was the direct cause of the following exception:

RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/3851630160.py in <cell line: 0>()
     79 # Speed: fewer joblib tasks overhead via larger batch_size; cap workers to avoid IO thrash.
     80 n_jobs = min(4, os.cpu_count() or 4)
---> 81 _ = Parallel(n_jobs=n_jobs, backend="loky", batch_size=32)(
     82     delayed(save_one)(row.eeg_id, row.spectrogram_id)
     83     for row in uniq.itertuples(index=False)

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in __call__(self, iterable)
   2070         next(output)
   2071 
-> 2072         return output if self.return_generator else list(output)
   2073 
   2074     def __repr__(self):

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _get_outputs(self, iterator, pre_dispatch)
   1680 
   1681             with self._backend.retrieval_context():
-> 1682                 yield from self._retrieve()
   1683 
   1684         except GeneratorExit:

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _retrieve(self)
   1782             # worker traceback.
   1783             if self._aborting:
-> 1784                 self._raise_error_fast()
   1785                 break
   1786 

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _raise_error_fast(self)
   1857         # called directly or if the generator is gc'ed.
   1858         if error_job is not None:
-> 1859             error_job.get_result(self.timeout)
   1860 
   1861     def _warn_exit_early(self):

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in get_result(self, timeout)
    756             # callback thread, and is stored internally. It's just waiting to
    757             # be returned.
--> 758             return self._return_or_raise()
    759 
    760         # For other backends, the main thread needs to run the retrieval step.

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _return_or_raise(self)
    771         try:
    772             if self.status == TASK_ERROR:
--> 773                 raise self._result
    774             return self._result
    775         finally:

RuntimeError: stack expects each tensor to be equal size, but got [400, 300] at entry 0 and [512, 568] at entry 1

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

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.loc[index]
        eeg_id = str(row.eeg_id)

        spec_img = np.load(
            os.path.join(self.resized_path, eeg_id + "_spec.npy")
        ).astype("float32", copy=False)
        eeg_img = np.load(os.path.join(self.resized_path, eeg_id + "_eeg.npy")).astype(
            "float32", copy=False
        )
        raw_50s_img = np.load(
            os.path.join(self.resized_path, eeg_id + "_raw50.npy")
        ).astype("float32", copy=False)
        raw_10s_l_img = np.load(
            os.path.join(self.resized_path, eeg_id + "_raw10l.npy")
        ).astype("float32", copy=False)
        raw_10s_c_img = np.load(
            os.path.join(self.resized_path, eeg_id + "_raw10c.npy")
        ).astype("float32", copy=False)
        raw_10s_r_img = np.load(
            os.path.join(self.resized_path, eeg_id + "_raw10r.npy")
        ).astype("float32", copy=False)

        eeg_img = np.expand_dims(eeg_img, -1)
        spec_img = np.expand_dims(spec_img, -1)
        raw_50s_img = np.expand_dims(raw_50s_img, -1)
        raw_10s_l_img = np.expand_dims(raw_10s_l_img, -1)
        raw_10s_c_img = np.expand_dims(raw_10s_c_img, -1)
        raw_10s_r_img = np.expand_dims(raw_10s_r_img, -1)

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
    prefetch_factor=2 if num_workers > 0 else None,
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
/tmp/ipykernel_55/553796898.py in <cell line: 0>()
     78 # Speed: inference_mode is stricter than no_grad and reduces overhead, same outputs.
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
  File "/tmp/ipykernel_55/1259760951.py", line 15, in __getitem__
    spec_img = np.load(
               ^^^^^^^^
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
        "/kaggle/working/spec_spectrograms/",
        "/kaggle/working/eeg_spectrograms/",
        "/kaggle/working/eeg_50s_raws/",
        "/kaggle/working/eeg_10s_raws/",
        "/kaggle/working/resized_518/",
        "/kaggle/working/squeezeformer",
    ]:
        if os.path.exists(p):
            shutil.rmtree(p, ignore_errors=True)
