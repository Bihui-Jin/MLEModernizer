# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

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
from scipy.signal import butter, lfilter  # noqa: F401
from scipy import signal




## === cell 4
def _df_to_filled_np(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Return float32 array [T, C] with NaNs filled by per-column mean (pandas mean ignores NaNs)."""
    arr = df[cols].to_numpy(dtype=np.float32, copy=True)
    means = np.nanmean(arr, axis=0)
    inds = np.where(np.isnan(arr))
    if inds[0].size:
        arr[inds] = means[inds[1]]
    return arr


_STFT_UNIQUE_COLS = sorted({c for cols in FEATS for c in cols})
_STFT_COL_IDX = {c: i for i, c in enumerate(_STFT_UNIQUE_COLS)}


def stft_spec_from_eeg_df(eeg: pd.DataFrame):
    EEG_LENGTH = 50
    time_temp = 0
    time_start = round(time_temp + (50 - EEG_LENGTH) / 2 * 200)
    time_stop = round(time_temp + (50 + EEG_LENGTH) / 2 * 200)

    eeg = eeg.iloc[time_start:time_stop]

    list_eeg = list()
    fs = 200
    nperseg = 70
    noverlap = 0

    eeg_np = _df_to_filled_np(eeg, _STFT_UNIQUE_COLS)

    for k in range(4):
        COLS = FEATS[k]
        img = np.zeros((128, 142, 4), dtype="float32")
        for kk in range(4):
            eeg_1 = eeg_np[:, _STFT_COL_IDX[COLS[kk]]]
            eeg_2 = eeg_np[:, _STFT_COL_IDX[COLS[kk + 1]]]
            new_eeg = eeg_1 - eeg_2

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


def stft_spec_from_eeg(parquet_path):
    eeg = pd.read_parquet(parquet_path)
    return stft_spec_from_eeg_df(eeg)




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

_RAW_UNIQUE_COLS = sorted(
    {c for chans in RAW_FEATS.values() for ch in chans for c in ch.split("-")}
)
_RAW_COL_IDX = {c: i for i, c in enumerate(_RAW_UNIQUE_COLS)}
_RAW_PAIR_IDXS = {
    region: [
        (_RAW_COL_IDX[ch.split("-")[0]], _RAW_COL_IDX[ch.split("-")[1]]) for ch in chans
    ]
    for region, chans in RAW_FEATS.items()
}


def _raw_window_to_region_tensor_from_np(eeg_np: np.ndarray, EEG_LENGTH: int):
    list_eeg = []

    for region in RAW_FEATS.keys():
        pair_idxs = _RAW_PAIR_IDXS[region]  # 4 pairs
        i1 = np.fromiter((p[0] for p in pair_idxs), dtype=np.int64, count=4)
        i2 = np.fromiter((p[1] for p in pair_idxs), dtype=np.int64, count=4)

        diffs = eeg_np[:, i1] - eeg_np[:, i2]  # [T, 4]
        diffs = signal.filtfilt(b, a, diffs, axis=0)
        diffs = np.clip(diffs, -1024, 1024).astype(np.float32, copy=False)  # [T, 4]

        eeg = diffs.T  # [4, T]
        eeg = np.reshape(eeg, (4, 200, EEG_LENGTH))
        eeg = np.concatenate(
            [eeg[0, :, :], eeg[1, :, :], eeg[2, :, :], eeg[3, :, :]], 1
        )
        list_eeg.append(eeg)

    eeg_out = np.concatenate(list_eeg, 1)
    eeg_out /= 104
    return eeg_out


def _raw_window_to_region_tensor(eeg_default: pd.DataFrame, EEG_LENGTH: int):
    eeg_np = _df_to_filled_np(eeg_default, _RAW_UNIQUE_COLS)  # [T, C]
    return _raw_window_to_region_tensor_from_np(eeg_np, EEG_LENGTH)


def raw10seeg_from_eeg_df(raw_eeg: pd.DataFrame):
    EEG_LENGTH = 10
    time_temp = 0

    base_start = round(time_temp + (50 - 50) / 2 * 200)
    base_stop = round(time_temp + (50 + 50) / 2 * 200)
    base_np = _df_to_filled_np(
        raw_eeg.iloc[base_start:base_stop], _RAW_UNIQUE_COLS
    )  # [10000, C]

    time_start = round(time_temp + (50 - EEG_LENGTH) / 2 * 200)
    time_stop = round(time_temp + (50 + EEG_LENGTH) / 2 * 200)
    eeg_c = _raw_window_to_region_tensor_from_np(
        base_np[time_start:time_stop], EEG_LENGTH
    )

    time_start = round(time_temp + 18 * 200)
    time_stop = round(time_temp + 28 * 200)
    eeg_l = _raw_window_to_region_tensor_from_np(
        base_np[time_start:time_stop], EEG_LENGTH
    )

    time_start = round(time_temp + 22 * 200)
    time_stop = round(time_temp + 32 * 200)
    eeg_r = _raw_window_to_region_tensor_from_np(
        base_np[time_start:time_stop], EEG_LENGTH
    )

    return eeg_l, eeg_c, eeg_r


def raw10seeg_from_eeg(parquet_path, eeg_id):
    raw_eeg = pd.read_parquet(parquet_path)
    return raw10seeg_from_eeg_df(raw_eeg)


def raw50seeg_from_eeg_df(raw_eeg: pd.DataFrame):
    EEG_LENGTH = 50
    time_temp = 0
    time_start = round(time_temp + (50 - EEG_LENGTH) / 2 * 200)
    time_stop = round(time_temp + (50 + EEG_LENGTH) / 2 * 200)

    eeg_default = raw_eeg.iloc[time_start:time_stop]
    eeg_np = _df_to_filled_np(eeg_default, _RAW_UNIQUE_COLS)  # [T, C]
    return _raw_window_to_region_tensor_from_np(eeg_np, EEG_LENGTH)


def raw50seeg_from_eeg(parquet_path):
    raw_eeg = pd.read_parquet(parquet_path)
    return raw50seeg_from_eeg_df(raw_eeg)




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

if DEBUG is True:
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

cache224_directory_path = "cache_224/"
os.makedirs(cache224_directory_path, exist_ok=True)
cache518_directory_path = "cache_518/"
os.makedirs(cache518_directory_path, exist_ok=True)

from joblib import Parallel, delayed

EEG_IDS = test.eeg_id.unique()


def save(row):
    eeg_id = str(row["eeg_id"])
    spec_id = row["spectrogram_id"]

    spec = pd.read_parquet(f"{SPEC_PATH}{spec_id}.parquet")
    split_spec_arr = (
        spec.iloc[:300, 1:].to_numpy(dtype=np.float32, copy=False).T
    )  # (Hz, Time<=300)
    np.save(f"{spec_directory_path}{eeg_id}", split_spec_arr)

    eeg_df = pd.read_parquet(f"{EEG_PATH}{eeg_id}.parquet")

    img_l, img_c, img_r = raw10seeg_from_eeg_df(eeg_df)
    np.save(f"{raw_10s_directory_path}{eeg_id}_l", img_l)
    np.save(f"{raw_10s_directory_path}{eeg_id}_c", img_c)
    np.save(f"{raw_10s_directory_path}{eeg_id}_r", img_r)

    img = raw50seeg_from_eeg_df(eeg_df)
    np.save(f"{raw_50s_directory_path}{eeg_id}", img)

    img = stft_spec_from_eeg_df(eeg_df)
    np.save(f"{eeg_directory_path}{eeg_id}", img)


def _needs_test_features(eeg_id: str):
    return not (
        os.path.exists(f"{spec_directory_path}{eeg_id}.npy")
        and os.path.exists(f"{eeg_directory_path}{eeg_id}.npy")
        and os.path.exists(f"{raw_50s_directory_path}{eeg_id}.npy")
        and os.path.exists(f"{raw_10s_directory_path}{eeg_id}_l.npy")
        and os.path.exists(f"{raw_10s_directory_path}{eeg_id}_c.npy")
        and os.path.exists(f"{raw_10s_directory_path}{eeg_id}_r.npy")
    )


test_records = test.to_dict("records")
todo = [row for row in test_records if _needs_test_features(str(row["eeg_id"]))]

CPU = os.cpu_count() or 4
N_JOBS = min(
    8, max(1, CPU)
)  # threads: safe to use more for I/O + scipy/pyarrow internals

if len(todo):
    _ = Parallel(n_jobs=N_JOBS, prefer="threads", batch_size=8)(
        delayed(save)(row) for row in todo
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



## === cell 10
import torch.utils.data as data
from torch.utils.data import DataLoader



## === cell 11
from skimage.transform import resize



## === cell 12
import shutil



## === cell 13
pass




## === cell 14
def _resize_and_norm_all(
    spec_img,
    eeg_img,
    raw_50s_img,
    raw_10s_l_img,
    raw_10s_c_img,
    raw_10s_r_img,
    out_hw,
):
    eeg_img = resize(eeg_img, out_hw, preserve_range=True, anti_aliasing=True).astype(
        "float32"
    )
    spec_img = resize(spec_img, out_hw, preserve_range=True, anti_aliasing=True).astype(
        "float32"
    )
    raw_10s_l_img = resize(
        raw_10s_l_img, out_hw, preserve_range=True, anti_aliasing=True
    ).astype("float32")
    raw_10s_c_img = resize(
        raw_10s_c_img, out_hw, preserve_range=True, anti_aliasing=True
    ).astype("float32")
    raw_10s_r_img = resize(
        raw_10s_r_img, out_hw, preserve_range=True, anti_aliasing=True
    ).astype("float32")
    raw_50s_img = resize(
        raw_50s_img, out_hw, preserve_range=True, anti_aliasing=True
    ).astype("float32")

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

    img_mean = eeg_img.mean(axis=(0, 1))
    img_std = eeg_img.std(axis=(0, 1))
    eeg_img = (eeg_img - img_mean) / (img_std + eps)

    img_mean = spec_img.mean(axis=(0, 1))
    img_std = spec_img.std(axis=(0, 1))
    spec_img = (spec_img - img_mean) / (img_std + eps)

    return (
        spec_img.astype("float32", copy=False),
        eeg_img.astype("float32", copy=False),
        raw_50s_img.astype("float32", copy=False),
        raw_10s_l_img.astype("float32", copy=False),
        raw_10s_c_img.astype("float32", copy=False),
        raw_10s_r_img.astype("float32", copy=False),
    )


def _cache_paths(base_dir: str, eeg_id: str):
    return (
        os.path.join(base_dir, f"{eeg_id}_spec.npy"),
        os.path.join(base_dir, f"{eeg_id}_eeg.npy"),
        os.path.join(base_dir, f"{eeg_id}_raw50.npy"),
        os.path.join(base_dir, f"{eeg_id}_raw10l.npy"),
        os.path.join(base_dir, f"{eeg_id}_raw10c.npy"),
        os.path.join(base_dir, f"{eeg_id}_raw10r.npy"),
    )


class CachedTest518Folder(data.Dataset):
    def __init__(self, df):
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.df["eeg_id"] = self.df["eeg_id"].astype(str)
        self.cache_dir = cache518_directory_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        eeg_id = self.df.loc[index, "eeg_id"]
        p_spec, p_eeg, p_raw50, p_l, p_c, p_r = _cache_paths(self.cache_dir, eeg_id)
        spec_img = np.load(p_spec, mmap_mode="r")
        eeg_img = np.load(p_eeg, mmap_mode="r")
        raw_50s_img = np.load(p_raw50, mmap_mode="r")
        raw_10s_l_img = np.load(p_l, mmap_mode="r")
        raw_10s_c_img = np.load(p_c, mmap_mode="r")
        raw_10s_r_img = np.load(p_r, mmap_mode="r")
        return (
            spec_img,
            eeg_img,
            raw_50s_img,
            raw_10s_l_img,
            raw_10s_c_img,
            raw_10s_r_img,
            eeg_id,
        )


class TestOnTheFlyFolder(data.Dataset):
    def __init__(self, df, test_imgsize, max_cache_items=32):
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.df["eeg_id"] = self.df["eeg_id"].astype(str)
        self.test_imgsize = tuple(test_imgsize)

        self._max_cache_items = int(max_cache_items)
        self._cache = {}  # eeg_id -> tuple(arrays)
        self._cache_order = []  # simple LRU

    def __len__(self):
        return len(self.df)

    def _lru_get(self, key):
        v = self._cache.get(key, None)
        if v is None:
            return None
        try:
            self._cache_order.remove(key)
        except ValueError:
            pass
        self._cache_order.append(key)
        return v

    def _lru_put(self, key, value):
        if key in self._cache:
            self._cache[key] = value
            try:
                self._cache_order.remove(key)
            except ValueError:
                pass
            self._cache_order.append(key)
            return
        self._cache[key] = value
        self._cache_order.append(key)
        if len(self._cache_order) > self._max_cache_items:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)

    def __getitem__(self, index):
        row = self.df.loc[index]
        eeg_id = row.eeg_id
        spec_id = row.spectrogram_id

        cached = self._lru_get(eeg_id)
        if cached is not None:
            (
                spec_img,
                eeg_img,
                raw_50s_img,
                raw_10s_l_img,
                raw_10s_c_img,
                raw_10s_r_img,
            ) = cached
            return (
                spec_img,
                eeg_img,
                raw_50s_img,
                raw_10s_l_img,
                raw_10s_c_img,
                raw_10s_r_img,
                eeg_id,
            )

        spec = pd.read_parquet(f"{SPEC_PATH}{spec_id}.parquet")
        split_spec_arr = spec.iloc[:300, 1:].to_numpy(dtype=np.float32, copy=False).T

        eeg_df = pd.read_parquet(f"{EEG_PATH}{eeg_id}.parquet")
        raw_10s_l, raw_10s_c, raw_10s_r = raw10seeg_from_eeg_df(eeg_df)
        raw_50s = raw50seeg_from_eeg_df(eeg_df)
        eeg_stft = stft_spec_from_eeg_df(eeg_df)

        spec_img, eeg_img, raw_50s_img, raw_10s_l_img, raw_10s_c_img, raw_10s_r_img = (
            _resize_and_norm_all(
                split_spec_arr,
                eeg_stft,
                raw_50s,
                raw_10s_l,
                raw_10s_c,
                raw_10s_r,
                self.test_imgsize,
            )
        )

        self._lru_put(
            eeg_id,
            (
                spec_img,
                eeg_img,
                raw_50s_img,
                raw_10s_l_img,
                raw_10s_c_img,
                raw_10s_r_img,
            ),
        )

        return (
            spec_img,
            eeg_img,
            raw_50s_img,
            raw_10s_l_img,
            raw_10s_c_img,
            raw_10s_r_img,
            eeg_id,
        )


class ImageFolder(data.Dataset):
    def __init__(self, df, test_imgsize, mode="test"):
        super().__init__()
        df = df.copy()
        df["eeg_id"] = df["eeg_id"]
        self.spec_data_path = spec_directory_path
        self.eeg_data_path = eeg_directory_path
        self.raw_50s_data_path = raw_50s_directory_path
        self.raw_10s_data_path = raw_10s_directory_path
        self.df = df.reset_index(drop=True)
        self.test_imgsize = tuple(test_imgsize)
        self.mode = mode

        if self.test_imgsize == (224, 224):
            self.cache_dir = cache224_directory_path
        elif self.test_imgsize == (518, 518):
            self.cache_dir = cache518_directory_path
        else:
            self.cache_dir = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.loc[index]
        eeg_id = str(row.eeg_id)

        if self.cache_dir is not None:
            p_spec, p_eeg, p_raw50, p_l, p_c, p_r = _cache_paths(self.cache_dir, eeg_id)
            if (
                os.path.exists(p_spec)
                and os.path.exists(p_eeg)
                and os.path.exists(p_raw50)
                and os.path.exists(p_l)
                and os.path.exists(p_c)
                and os.path.exists(p_r)
            ):
                spec_img = np.load(p_spec, mmap_mode="r")
                eeg_img = np.load(p_eeg, mmap_mode="r")
                raw_50s_img = np.load(p_raw50, mmap_mode="r")
                raw_10s_l_img = np.load(p_l, mmap_mode="r")
                raw_10s_c_img = np.load(p_c, mmap_mode="r")
                raw_10s_r_img = np.load(p_r, mmap_mode="r")

                if self.mode == "train":
                    y = row[CLASSES].values.astype(np.float32)
                    y = np.clip(y, 0.0, None)
                    s = float(y.sum())
                    if s <= 0:
                        y[:] = 1.0 / len(CLASSES)
                    else:
                        y = y / s
                    return (
                        spec_img,
                        eeg_img,
                        raw_50s_img,
                        raw_10s_l_img,
                        raw_10s_c_img,
                        raw_10s_r_img,
                        y,
                        eeg_id,
                    )
                return (
                    spec_img,
                    eeg_img,
                    raw_50s_img,
                    raw_10s_l_img,
                    raw_10s_c_img,
                    raw_10s_r_img,
                    eeg_id,
                )

        spec_image_path = os.path.join(self.spec_data_path, eeg_id + ".npy")
        eeg_image_path = os.path.join(self.eeg_data_path, eeg_id + ".npy")
        raw_50s_image_path = os.path.join(self.raw_50s_data_path, eeg_id + ".npy")
        raw_10s_l_image_path = os.path.join(self.raw_10s_data_path, eeg_id + "_l.npy")
        raw_10s_c_image_path = os.path.join(self.raw_10s_data_path, eeg_id + "_c.npy")
        raw_10s_r_image_path = os.path.join(self.raw_10s_data_path, eeg_id + "_r.npy")

        spec_img = np.load(spec_image_path, mmap_mode="r")
        raw_50s_img = np.load(raw_50s_image_path, mmap_mode="r")
        raw_10s_l_img = np.load(raw_10s_l_image_path, mmap_mode="r")
        raw_10s_c_img = np.load(raw_10s_c_image_path, mmap_mode="r")
        raw_10s_r_img = np.load(raw_10s_r_image_path, mmap_mode="r")
        eeg_img = np.load(eeg_image_path, mmap_mode="r")

        spec_img, eeg_img, raw_50s_img, raw_10s_l_img, raw_10s_c_img, raw_10s_r_img = (
            _resize_and_norm_all(
                spec_img,
                eeg_img,
                raw_50s_img,
                raw_10s_l_img,
                raw_10s_c_img,
                raw_10s_r_img,
                self.test_imgsize,
            )
        )

        if self.mode == "train":
            y = row[CLASSES].values.astype(np.float32)
            y = np.clip(y, 0.0, None)
            s = float(y.sum())
            if s <= 0:
                y[:] = 1.0 / len(CLASSES)
            else:
                y = y / s
            return (
                spec_img,
                eeg_img,
                raw_50s_img,
                raw_10s_l_img,
                raw_10s_c_img,
                raw_10s_r_img,
                y,
                eeg_id,
            )

        return (
            spec_img,
            eeg_img,
            raw_50s_img,
            raw_10s_l_img,
            raw_10s_c_img,
            raw_10s_r_img,
            eeg_id,
        )




## === cell 15
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

        self.head = nn.Linear(768 * 2 + 384 * 2, 6)
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
        return logits




## === cell 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

train_df = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
)
train_df["eeg_id"] = train_df["eeg_id"].astype(str)

agg = train_df.groupby("eeg_id", as_index=False)[
    CLASSES + ["patient_id", "spectrogram_id"]
].agg({**{c: "sum" for c in CLASSES}, "patient_id": "first", "spectrogram_id": "first"})

patients = agg["patient_id"].astype(int).values
uniq_pat = np.unique(patients)
rng = np.random.default_rng(Config.seed)
rng.shuffle(uniq_pat)
n_val = max(1, int(0.1 * len(uniq_pat)))
val_pat = set(uniq_pat[:n_val])

trn_agg = agg[~agg["patient_id"].isin(val_pat)].reset_index(drop=True)
val_agg = agg[agg["patient_id"].isin(val_pat)].reset_index(drop=True)
print("Train eeg_ids:", len(trn_agg), "Val eeg_ids:", len(val_agg))

SPEC_PATH_TRAIN = (
    "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/"
)
EEG_PATH_TRAIN = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"


def save_train_row(row):
    eeg_id = str(row["eeg_id"])
    spec_id = row["spectrogram_id"]

    if not os.path.exists(f"{spec_directory_path}{eeg_id}.npy"):
        spec = pd.read_parquet(f"{SPEC_PATH_TRAIN}{spec_id}.parquet")
        split_spec_arr = spec.iloc[:300, 1:].to_numpy(dtype=np.float32, copy=False).T
        np.save(f"{spec_directory_path}{eeg_id}", split_spec_arr)

    need_eeg = (
        (not os.path.exists(f"{raw_10s_directory_path}{eeg_id}_l.npy"))
        or (not os.path.exists(f"{raw_10s_directory_path}{eeg_id}_c.npy"))
        or (not os.path.exists(f"{raw_10s_directory_path}{eeg_id}_r.npy"))
        or (not os.path.exists(f"{raw_50s_directory_path}{eeg_id}.npy"))
        or (not os.path.exists(f"{eeg_directory_path}{eeg_id}.npy"))
    )
    if need_eeg:
        eeg_df = pd.read_parquet(f"{EEG_PATH_TRAIN}{eeg_id}.parquet")

        if not os.path.exists(f"{raw_10s_directory_path}{eeg_id}_l.npy"):
            img_l, img_c, img_r = raw10seeg_from_eeg_df(eeg_df)
            np.save(f"{raw_10s_directory_path}{eeg_id}_l", img_l)
            np.save(f"{raw_10s_directory_path}{eeg_id}_c", img_c)
            np.save(f"{raw_10s_directory_path}{eeg_id}_r", img_r)

        if not os.path.exists(f"{raw_50s_directory_path}{eeg_id}.npy"):
            img = raw50seeg_from_eeg_df(eeg_df)
            np.save(f"{raw_50s_directory_path}{eeg_id}", img)

        if not os.path.exists(f"{eeg_directory_path}{eeg_id}.npy"):
            img = stft_spec_from_eeg_df(eeg_df)
            np.save(f"{eeg_directory_path}{eeg_id}", img)


MAX_TRAIN_EEGS = 1800 if not DEBUG else 40
trn_use = trn_agg.head(MAX_TRAIN_EEGS).copy()
val_use = val_agg.head(max(200, int(0.15 * MAX_TRAIN_EEGS))).copy()

_ = Parallel(n_jobs=N_JOBS, prefer="threads", batch_size=8)(
    delayed(save_train_row)(row) for row in trn_use.to_dict("records")
)
_ = Parallel(n_jobs=N_JOBS, prefer="threads", batch_size=8)(
    delayed(save_train_row)(row) for row in val_use.to_dict("records")
)


def _build_cache_for_df(df: pd.DataFrame, out_hw, cache_dir: str):
    def _one(eeg_id: str):
        eeg_id = str(eeg_id)
        p_spec, p_eeg, p_raw50, p_l, p_c, p_r = _cache_paths(cache_dir, eeg_id)
        if (
            os.path.exists(p_spec)
            and os.path.exists(p_eeg)
            and os.path.exists(p_raw50)
            and os.path.exists(p_l)
            and os.path.exists(p_c)
            and os.path.exists(p_r)
        ):
            return

        spec_img = np.load(
            os.path.join(spec_directory_path, eeg_id + ".npy"), mmap_mode="r"
        )
        eeg_img = np.load(
            os.path.join(eeg_directory_path, eeg_id + ".npy"), mmap_mode="r"
        )
        raw_50s_img = np.load(
            os.path.join(raw_50s_directory_path, eeg_id + ".npy"), mmap_mode="r"
        )
        raw_10s_l_img = np.load(
            os.path.join(raw_10s_directory_path, eeg_id + "_l.npy"), mmap_mode="r"
        )
        raw_10s_c_img = np.load(
            os.path.join(raw_10s_directory_path, eeg_id + "_c.npy"), mmap_mode="r"
        )
        raw_10s_r_img = np.load(
            os.path.join(raw_10s_directory_path, eeg_id + "_r.npy"), mmap_mode="r"
        )

        spec_img, eeg_img, raw_50s_img, raw_10s_l_img, raw_10s_c_img, raw_10s_r_img = (
            _resize_and_norm_all(
                spec_img,
                eeg_img,
                raw_50s_img,
                raw_10s_l_img,
                raw_10s_c_img,
                raw_10s_r_img,
                out_hw,
            )
        )

        np.save(p_spec, spec_img)
        np.save(p_eeg, eeg_img)
        np.save(p_raw50, raw_50s_img)
        np.save(p_l, raw_10s_l_img)
        np.save(p_c, raw_10s_c_img)
        np.save(p_r, raw_10s_r_img)

    ids = df["eeg_id"].astype(str).values
    _ = Parallel(n_jobs=N_JOBS, prefer="threads", batch_size=64)(
        delayed(_one)(eeg_id) for eeg_id in ids
    )


train_imgsize = (224, 224)
_build_cache_for_df(trn_use, train_imgsize, cache224_directory_path)
_build_cache_for_df(val_use, train_imgsize, cache224_directory_path)

train_data = ImageFolder(trn_use, train_imgsize, mode="train")
val_data = ImageFolder(val_use, train_imgsize, mode="train")

pin = torch.cuda.is_available()
train_loader = DataLoader(
    train_data,
    batch_size=8,
    shuffle=True,
    num_workers=min(4, os.cpu_count() or 4),
    drop_last=True,
    pin_memory=pin,
    persistent_workers=True,
    prefetch_factor=2,
)
val_loader = DataLoader(
    val_data,
    batch_size=8,
    shuffle=False,
    num_workers=min(4, os.cpu_count() or 4),
    drop_last=False,
    pin_memory=pin,
    persistent_workers=True,
    prefetch_factor=2,
)

model = Net("vit_base_patch14_reg4_dinov2.lvd142m", str(device)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)


def soft_ce_loss(logits, targets):
    logp = F.log_softmax(logits, dim=1)
    return -(targets * logp).sum(dim=1).mean()


model.train()
total_steps = 180  # fixed budget
step = 0
for epoch in range(1, 4):
    for batch in train_loader:
        (
            spec_imgs,
            eeg_imgs,
            raw_50s_imgs,
            raw_10s_l_imgs,
            raw_10s_c_imgs,
            raw_10s_r_imgs,
            y,
            eeg_ids,
        ) = batch
        spec_imgs = spec_imgs.to(device, non_blocking=True).float()
        eeg_imgs = eeg_imgs.to(device, non_blocking=True).float()
        raw_50s_imgs = raw_50s_imgs.to(device, non_blocking=True).float()
        raw_10s_l_imgs = raw_10s_l_imgs.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)
        logits = model(spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_l_imgs)
        loss = soft_ce_loss(logits, y)
        loss.backward()
        optimizer.step()

        step += 1
        if step >= total_steps:
            break
    if step >= total_steps:
        break

model.eval()
with torch.no_grad():
    val_losses = []
    for batch in val_loader:
        (
            spec_imgs,
            eeg_imgs,
            raw_50s_imgs,
            raw_10s_l_imgs,
            raw_10s_c_imgs,
            raw_10s_r_imgs,
            y,
            eeg_ids,
        ) = batch
        spec_imgs = spec_imgs.to(device, non_blocking=True).float()
        eeg_imgs = eeg_imgs.to(device, non_blocking=True).float()
        raw_50s_imgs = raw_50s_imgs.to(device, non_blocking=True).float()
        raw_10s_l_imgs = raw_10s_l_imgs.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).float()
        logits = model(spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_l_imgs)
        val_losses.append(float(soft_ce_loss(logits, y).detach().cpu().item()))
    print("Val soft-CE:", float(np.mean(val_losses)) if len(val_losses) else None)



## === cell 17
pass



## === cell 18
test_df = test.copy()
test_df["eeg_id"] = test_df["eeg_id"].astype(str)

_build_cache_for_df(test_df[["eeg_id"]], (518, 518), cache518_directory_path)

test_data = CachedTest518Folder(test_df[["eeg_id"]])

pin = torch.cuda.is_available()
test_loader = DataLoader(
    test_data,
    batch_size=8,
    pin_memory=pin,
    num_workers=min(4, os.cpu_count() or 4),
    drop_last=False,
    persistent_workers=True,
    prefetch_factor=4,
)

result_3 = {}

if "model" not in globals():
    model = Net("vit_base_patch14_reg4_dinov2.lvd142m", str(device)).to(device)

model.eval()
with torch.no_grad():
    for batch_idx, (
        spec_imgs,
        eeg_imgs,
        raw_50s_imgs,
        raw_10s_l_imgs,
        raw_10s_c_imgs,
        raw_10s_r_imgs,
        eeg_ids,
    ) in enumerate(test_loader):
        spec_imgs = spec_imgs.to(device, non_blocking=True).float()
        eeg_imgs = eeg_imgs.to(device, non_blocking=True).float()
        raw_50s_imgs = raw_50s_imgs.to(device, non_blocking=True).float()
        raw_10s_l_imgs = raw_10s_l_imgs.to(device, non_blocking=True).float()
        raw_10s_c_imgs = raw_10s_c_imgs.to(device, non_blocking=True).float()
        raw_10s_r_imgs = raw_10s_r_imgs.to(device, non_blocking=True).float()

        logits_l = model(spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_l_imgs)
        logits_c = model(spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_c_imgs)
        logits_r = model(spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_r_imgs)

        probs = (
            logits_l.softmax(dim=1) + logits_c.softmax(dim=1) + logits_r.softmax(dim=1)
        ) / 3.0
        probs = probs.detach().cpu().numpy()

        for j in range(len(eeg_ids)):
            eeg_id = str(eeg_ids[j])
            if eeg_id not in result_3:
                result_3[eeg_id] = np.zeros((6,), dtype=np.float64)
            result_3[eeg_id] += probs[j].astype(np.float64)



## === cell 19
if "model" in globals():
    del model
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()



## === cell 20
sub = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
)
preds = np.zeros((len(sub), 6), dtype=np.float64)

uniform = np.ones(6, dtype=np.float64) / 6.0

for i, eeg_id in enumerate(sub["eeg_id"].astype(str).values):
    if eeg_id in result_3 and np.isfinite(result_3[eeg_id]).all():
        p = result_3[eeg_id].copy()
    else:
        p = uniform.copy()
    p = np.clip(p, 1e-12, None)
    p = p / p.sum()
    preds[i] = p

sub[CLASSES] = preds.astype(np.float32)
sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print(
    "Row sums (min/mean/max):",
    preds.sum(axis=1).min(),
    preds.sum(axis=1).mean(),
    preds.sum(axis=1).max(),
)



## === cell 21
if DEBUG is False:
    shutil.rmtree("/kaggle/working/spec_spectrograms/", ignore_errors=True)
    shutil.rmtree("/kaggle/working/eeg_spectrograms/", ignore_errors=True)
    shutil.rmtree("/kaggle/working/eeg_50s_raws/", ignore_errors=True)
    shutil.rmtree("/kaggle/working/eeg_10s_raws/", ignore_errors=True)
    shutil.rmtree("/kaggle/working/cache_224/", ignore_errors=True)
    shutil.rmtree("/kaggle/working/cache_518/", ignore_errors=True)
    shutil.rmtree("/kaggle/working/squeezeformer", ignore_errors=True)
