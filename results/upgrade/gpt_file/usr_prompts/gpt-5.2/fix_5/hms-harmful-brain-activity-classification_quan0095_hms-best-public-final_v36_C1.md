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

try:
    import librosa  # noqa: F401
except Exception:
    librosa = None



## === cell 3
from scipy import signal



## === cell 4

SFREQ = 200
filter_range = [0.5, 40]

RAW_FEATS = {
    "LL": ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1"],
    "RL": ["Fp2-F8", "F8-T4", "T4-T6", "T6-O2"],
    "LP": ["Fp1-F3", "F3-C3", "C3-P3", "P3-O1"],
    "RP": ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2"],
}

b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")

_ALL_CHANS = sorted(
    {c for cols in FEATS for c in cols}
    | {p for pairs in RAW_FEATS.values() for ch in pairs for p in ch.split("-")}
)
_CHAN_TO_IDX = {c: i for i, c in enumerate(_ALL_CHANS)}

_REGION_PAIRS = {}
for region, pairs in RAW_FEATS.items():
    idx_pairs = []
    for ch in pairs:
        c1, c2 = ch.split("-")
        idx_pairs.append((_CHAN_TO_IDX[c1], _CHAN_TO_IDX[c2]))
    _REGION_PAIRS[region] = idx_pairs


def _read_eeg_array(parquet_path: str) -> np.ndarray:
    """Read parquet once, return (T, C) float32 with NaNs filled by per-column mean (same semantics)."""
    df = pd.read_parquet(parquet_path, columns=_ALL_CHANS)
    arr = df.to_numpy(dtype=np.float32, copy=False)  # (T, C), may contain NaNs
    means = np.nanmean(arr, axis=0)
    means = np.where(np.isfinite(means), means, 0.0).astype(np.float32)
    nan_mask = ~np.isfinite(arr)
    if nan_mask.any():
        arr[nan_mask] = means[np.nonzero(nan_mask)[1]]
    return arr


def _slice_center(
    arr: np.ndarray, eeg_length_s: int, start_s: float = None
) -> np.ndarray:
    """Return slice of length eeg_length_s seconds at 200 Hz.
    If start_s is None -> centered like original (50s total, centered window).
    Else -> [start_s, start_s + eeg_length_s) seconds from start.
    """
    if start_s is None:
        time_start = round((50 - eeg_length_s) / 2 * 200)
    else:
        time_start = round(start_s * 200)
    time_stop = time_start + round(eeg_length_s * 200)
    return arr[time_start:time_stop]


def _raw_from_window_arr(win_arr: np.ndarray, eeg_length_s: int) -> np.ndarray:
    """Compute raw representation for one window (same as raw50seeg_from_eeg or raw10seeg windows)."""
    list_eeg = []
    T = win_arr.shape[0]
    for region in RAW_FEATS.keys():
        eeg = np.empty((4, T), dtype=np.float32)
        for chan_i, (i1, i2) in enumerate(_REGION_PAIRS[region]):
            new_eeg = win_arr[:, i1] - win_arr[:, i2]
            new_eeg = signal.filtfilt(b, a, new_eeg, axis=0)
            new_eeg = np.clip(new_eeg, -1024, 1024).astype(np.float32, copy=False)
            eeg[chan_i, :] = new_eeg
        eeg = np.reshape(eeg, (4, 200, eeg_length_s))
        eeg = np.concatenate(
            (eeg[0, :, :], eeg[1, :, :], eeg[2, :, :], eeg[3, :, :]), 1
        )
        list_eeg.append(eeg)
    eeg_out = np.concatenate(list_eeg, 1)
    eeg_out /= 104.0
    return eeg_out


def raw10seeg_from_eeg(parquet_path, eeg_id):
    EEG_LENGTH = 10
    raw_arr = _read_eeg_array(parquet_path)

    win_c = _slice_center(raw_arr, EEG_LENGTH, start_s=None)
    win_l = _slice_center(raw_arr, EEG_LENGTH, start_s=18)
    win_r = _slice_center(raw_arr, EEG_LENGTH, start_s=22)

    eeg_c = _raw_from_window_arr(win_c, EEG_LENGTH)
    eeg_l = _raw_from_window_arr(win_l, EEG_LENGTH)
    eeg_r = _raw_from_window_arr(win_r, EEG_LENGTH)
    return eeg_l, eeg_c, eeg_r


def raw50seeg_from_eeg(parquet_path):
    EEG_LENGTH = 50
    raw_arr = _read_eeg_array(parquet_path)
    win = _slice_center(raw_arr, EEG_LENGTH, start_s=None)
    return _raw_from_window_arr(win, EEG_LENGTH)


def stft_spec_from_eeg(parquet_path):
    """Keep same core logic as original cell 4 STFT (128,142,4) per region, then concat."""
    EEG_LENGTH = 50
    raw_arr = _read_eeg_array(parquet_path)

    eeg_arr = _slice_center(raw_arr, EEG_LENGTH, start_s=None)

    list_eeg = []
    for k in range(4):
        cols = FEATS[k]
        col_idx = [_CHAN_TO_IDX[c] for c in cols]
        img = np.zeros((128, 142, 4), dtype="float32")
        for kk in range(4):
            new_eeg = eeg_arr[:, col_idx[kk]] - eeg_arr[:, col_idx[kk + 1]]
            fs = 200
            nperseg = 70
            noverlap = 0
            _, _, spec = signal.spectrogram(
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

from joblib import Parallel, delayed

EEG_IDS = test.eeg_id.unique()


def _exists_all(eeg_id: int) -> bool:
    eeg_id = str(eeg_id)
    return (
        os.path.exists(f"{spec_directory_path}{eeg_id}.npy")
        and os.path.exists(f"{raw_10s_directory_path}{eeg_id}_l.npy")
        and os.path.exists(f"{raw_10s_directory_path}{eeg_id}_c.npy")
        and os.path.exists(f"{raw_10s_directory_path}{eeg_id}_r.npy")
        and os.path.exists(f"{raw_50s_directory_path}{eeg_id}.npy")
        and os.path.exists(f"{eeg_directory_path}{eeg_id}.npy")
    )


def save(row):
    eeg_id = row["eeg_id"]
    if _exists_all(eeg_id):
        return

    spec_id = row["spectrogram_id"]
    spec = pd.read_parquet(f"{SPEC_PATH}{spec_id}.parquet")
    spec_arr = spec.values[:, 1:].T.astype(
        "float32", copy=False
    )  # (Hz, Time) = (400, 300+)
    split_spec_arr = spec_arr[:, 0:300]
    np.save(f"{spec_directory_path}{eeg_id}", split_spec_arr)

    eeg_parquet = f"{EEG_PATH}{eeg_id}.parquet"

    img_l, img_c, img_r = raw10seeg_from_eeg(eeg_parquet, eeg_id)
    np.save(f"{raw_10s_directory_path}{eeg_id}_l", img_l)
    np.save(f"{raw_10s_directory_path}{eeg_id}_c", img_c)
    np.save(f"{raw_10s_directory_path}{eeg_id}_r", img_r)

    img = raw50seeg_from_eeg(eeg_parquet)
    np.save(f"{raw_50s_directory_path}{eeg_id}", img)

    img = stft_spec_from_eeg(eeg_parquet)
    np.save(f"{eeg_directory_path}{eeg_id}", img)


_ = Parallel(n_jobs=4, prefer="processes")(
    delayed(save)(row) for row in test[["eeg_id", "spectrogram_id"]].to_dict("records")
)




## === cell 6
class Config:
    seed = 2024
    num_folds = 5




## === cell 7
def seed_everything(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


seed_everything(Config.seed)



## === cell 8
import timm



## === cell 9
import torch.utils.data as data
from torch.utils.data import DataLoader



## === cell 10
try:
    import cv2  # type: ignore

    def _resize2d(img: np.ndarray, out_hw):
        oh, ow = out_hw
        return cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)

except Exception:
    from skimage.transform import resize as _sk_resize

    def _resize2d(img: np.ndarray, out_hw):
        return _sk_resize(img, out_hw, preserve_range=True, anti_aliasing=False).astype(
            np.float32, copy=False
        )




## === cell 11
import shutil



## === cell 12
os.makedirs("eeg_spectrograms/", exist_ok=True)



## === cell 14
def save(row):
    return


_ = Parallel(n_jobs=1)(delayed(save)(row) for _, row in test.head(1).iterrows())




## === cell 15
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

        spec_img = np.load(spec_image_path).astype("float32", copy=False)
        raw_50s_img = np.load(raw_50s_image_path).astype("float32", copy=False)
        raw_10s_l_img = np.load(raw_10s_l_image_path).astype("float32", copy=False)
        raw_10s_c_img = np.load(raw_10s_c_image_path).astype("float32", copy=False)
        raw_10s_r_img = np.load(raw_10s_r_image_path).astype("float32", copy=False)
        eeg_img = np.load(eeg_image_path).astype("float32", copy=False)

        spec_img = _resize2d(spec_img, self.test_imgsize)
        raw_10s_l_img = _resize2d(raw_10s_l_img, self.test_imgsize)
        raw_10s_c_img = _resize2d(raw_10s_c_img, self.test_imgsize)
        raw_10s_r_img = _resize2d(raw_10s_r_img, self.test_imgsize)
        raw_50s_img = _resize2d(raw_50s_img, self.test_imgsize)

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




## === cell 16
class Net(nn.Module):
    def __init__(self, back_bone, device_id):
        super().__init__()
        self.spec_model = timm.create_model(
            back_bone, num_classes=6, pretrained=False, in_chans=1
        )
        self.eeg_model = timm.create_model(
            back_bone,
            num_classes=6,
            pretrained=False,
            in_chans=1,
            dynamic_img_pad=True,
            dynamic_img_size=True,
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




## === cell 17
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

result_4 = {}  # ensure always defined

model_weights = [
    "/kaggle/input/hms-stage2/fold_0_exp_5_bestlb.pth",
    "/kaggle/input/hms-stage2/fold_1_exp_5_bestlb.pth",
    "/kaggle/input/hms-stage2/fold_2_exp_5_bestlb.pth",
    "/kaggle/input/hms-stage2/fold_3_exp_5_bestlb.pth",
    "/kaggle/input/hms-stage2/fold_4_exp_5_bestlb.pth",
]
model_types = ["vit_small", "vit_small", "vit_small", "vit_small", "vit_small"]

weights_available = all(os.path.exists(p) for p in model_weights)

if not weights_available:
    print("WARNING: Pretrained weights not found under /kaggle/input/hms-stage2/.")
    print(
        "Training a small in-notebook model (same Net architecture) to improve over uniform predictions."
    )

    train_df = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    )
    train_df = train_df.drop_duplicates(subset=["label_id"]).reset_index(drop=True)

    y = train_df[CLASSES].values.astype(np.float64)
    y = np.clip(y, 0, None)
    y_sum = y.sum(axis=1, keepdims=True)
    y = np.where(y_sum > 0, y / y_sum, np.full_like(y, 1.0 / 6.0))

    max_train = 2500 if torch.cuda.is_available() else 800
    train_df = train_df.iloc[:max_train].copy()
    y = y[:max_train]

    SPEC_PATH_TR = (
        "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/"
    )
    EEG_PATH_TR = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"

    def save_train(row):
        eeg_id = row["eeg_id"]
        if _exists_all(eeg_id):
            return
        spec_id = row["spectrogram_id"]
        spec = pd.read_parquet(f"{SPEC_PATH_TR}{spec_id}.parquet")
        spec_arr = spec.values[:, 1:].T.astype("float32", copy=False)
        split_spec_arr = spec_arr[:, 0:300]
        np.save(f"{spec_directory_path}{eeg_id}", split_spec_arr)

        eeg_parquet = f"{EEG_PATH_TR}{eeg_id}.parquet"
        img_l, img_c, img_r = raw10seeg_from_eeg(eeg_parquet, eeg_id)
        np.save(f"{raw_10s_directory_path}{eeg_id}_l", img_l)
        np.save(f"{raw_10s_directory_path}{eeg_id}_c", img_c)
        np.save(f"{raw_10s_directory_path}{eeg_id}_r", img_r)

        img = raw50seeg_from_eeg(eeg_parquet)
        np.save(f"{raw_50s_directory_path}{eeg_id}", img)

        img = stft_spec_from_eeg(eeg_parquet)
        np.save(f"{eeg_directory_path}{eeg_id}", img)

    _ = Parallel(n_jobs=4, prefer="processes")(
        delayed(save_train)(row)
        for row in train_df[["eeg_id", "spectrogram_id"]].to_dict("records")
    )

    class TrainFolder(ImageFolder):
        def __init__(self, df, y, test_imgsize):
            super().__init__(df, test_imgsize)
            self.y = y.astype(np.float32)

        def __getitem__(self, index):
            (
                spec_img,
                eeg_img,
                raw_50s_img,
                raw_10s_l_img,
                raw_10s_c_img,
                raw_10s_r_img,
                eeg_id,
            ) = super().__getitem__(index)
            target = self.y[index]
            return (
                spec_img,
                eeg_img,
                raw_50s_img,
                raw_10s_l_img,
                raw_10s_c_img,
                raw_10s_r_img,
                target,
            )

    train_data = TrainFolder(train_df, y, (518, 518))
    train_loader = DataLoader(
        train_data,
        batch_size=8 if torch.cuda.is_available() else 2,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=2,
        drop_last=True,
        persistent_workers=True if 2 > 0 else False,
    )

    model = Net("vit_small_patch14_reg4_dinov2.lvd142m", device).to(device)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

    def kl_loss(pred_logits, target_probs):
        pred_log_probs = F.log_softmax(pred_logits, dim=1)
        return F.kl_div(pred_log_probs, target_probs, reduction="batchmean")

    steps = 180 if torch.cuda.is_available() else 60
    it = iter(train_loader)
    for step in range(steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)

        (
            spec_imgs,
            eeg_imgs,
            raw_50s_imgs,
            raw_10s_l_imgs,
            raw_10s_c_imgs,
            raw_10s_r_imgs,
            targets,
        ) = batch
        spec_imgs = spec_imgs.to(device).float()
        eeg_imgs = eeg_imgs.to(device).float()
        raw_50s_imgs = raw_50s_imgs.to(device).float()
        raw_10s_l_imgs = raw_10s_l_imgs.to(device).float()
        targets = targets.to(device).float()

        opt.zero_grad(set_to_none=True)
        logits, _, _, _, _ = model(spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_l_imgs)
        loss = kl_loss(logits, targets)
        loss.backward()
        opt.step()

        if (step + 1) % 50 == 0:
            print(f"train step {step+1}/{steps} - loss {loss.item():.4f}")

    model.eval()
    vit_models = [model]
else:
    vit_models = []
    for i in range(len(model_types)):
        if model_types[i] == "vit_small":
            print(model_weights[i])
            model = Net("vit_small_patch14_reg4_dinov2.lvd142m", device).to(device)
            state = torch.load(model_weights[i], map_location=device)
            model.load_state_dict(state)
            model.eval()
            vit_models.append(model)

test_data = ImageFolder(test, (518, 518))
test_loader = DataLoader(
    test_data,
    batch_size=32,
    pin_memory=torch.cuda.is_available(),
    num_workers=4,
    drop_last=False,
    persistent_workers=True if 4 > 0 else False,
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

        ensemble_probs = torch.zeros((spec_imgs.shape[0], 6), device=device)
        for model in vit_models:
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

        ensemble_probs /= float(len(vit_models))
        ensemble_probs = ensemble_probs.detach().cpu().numpy()

        for j in range(len(eeg_ids)):
            eeg_id = eeg_ids[j]
            if eeg_id not in result_4:
                result_4[eeg_id] = np.zeros(6, dtype=np.float64)
            result_4[eeg_id] += ensemble_probs[j].astype(np.float64)

for model in vit_models:
    del model
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()



## === cell 18
sample_sub = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
)
sub = sample_sub.copy()

default_prob = np.array([1.0 / 6] * 6, dtype=np.float64)

preds = np.zeros((len(sub), 6), dtype=np.float64)
for i, eeg_id in enumerate(sub["eeg_id"].astype(str).values):
    if eeg_id in result_4 and np.isfinite(result_4[eeg_id]).all():
        preds[i] = result_4[eeg_id]
    else:
        preds[i] = default_prob

preds = np.clip(preds, 1e-12, None)
preds = preds / preds.sum(axis=1, keepdims=True)

sub[CLASSES] = preds
sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Saved submission.csv with shape:", sub.shape)



## === cell 19
if DEBUG is False:
    shutil.rmtree("/kaggle/working/spec_spectrograms", ignore_errors=True)
    shutil.rmtree("/kaggle/working/eeg_spectrograms", ignore_errors=True)
    shutil.rmtree("/kaggle/working/eeg_50s_raws", ignore_errors=True)
    shutil.rmtree("/kaggle/working/eeg_10s_raws", ignore_errors=True)
    shutil.rmtree("/kaggle/working/squeezeformer", ignore_errors=True)
