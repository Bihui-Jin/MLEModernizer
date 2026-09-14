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

    eeg_arr, col_index = _load_eeg_array(f"{EEG_PATH}{eeg_id}.parquet")

    img_l, img_c, img_r = raw10seeg_from_eeg_array(eeg_arr, col_index)
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

vote_sum = train.groupby("eeg_id")[CLASSES].sum()
vote_dist = vote_sum.div(vote_sum.sum(axis=1), axis=0).astype(np.float32)

train_unique = train.drop_duplicates("eeg_id")[
    ["eeg_id", "spectrogram_id", "patient_id"]
].copy()
train_unique["eeg_id"] = train_unique["eeg_id"].astype(str)

vote_dist_df = vote_dist.reset_index()
vote_dist_df["eeg_id"] = vote_dist_df["eeg_id"].astype(str)
train_unique = train_unique.merge(
    vote_dist_df, on="eeg_id", how="left", validate="one_to_one"
)
if train_unique[CLASSES].isna().any().any():
    raise RuntimeError("Found NaNs in merged targets; eeg_id alignment failed.")

targets = train_unique[CLASSES].values.astype(np.float32)

patients = train_unique["patient_id"].values
uniq_p = np.unique(patients)
rng = np.random.RandomState(Config.seed)
rng.shuffle(uniq_p)
val_patients = set(uniq_p[: max(1, int(0.1 * len(uniq_p)))])
is_val = np.array([p in val_patients for p in patients])

train_df = train_unique.loc[
    ~is_val, ["eeg_id", "spectrogram_id", "patient_id"]
].reset_index(drop=True)
val_df = train_unique.loc[
    is_val, ["eeg_id", "spectrogram_id", "patient_id"]
].reset_index(drop=True)
train_y = targets[~is_val]
val_y = targets[is_val]

print("unique eegs:", len(train_unique), "train:", len(train_df), "val:", len(val_df))

train_need = train_df[["eeg_id", "spectrogram_id"]].copy()
val_need = val_df[["eeg_id", "spectrogram_id"]].copy()
test_unique = test.drop_duplicates("eeg_id")[["eeg_id", "spectrogram_id"]].copy()
need_all = pd.concat([train_need, val_need, test_unique], axis=0, ignore_index=True)
need_all["eeg_id"] = need_all["eeg_id"].astype(str)
need_all["spectrogram_id"] = need_all["spectrogram_id"].astype(str)
need_all = need_all.drop_duplicates("eeg_id", keep="first")

test_ids_set = set(test_unique["eeg_id"].astype(str).values)


def _path_tuple(eeg_id, spec_id):
    if eeg_id in test_ids_set:
        return eeg_id, spec_id, TEST_SPEC_PATH, TEST_EEG_PATH
    else:
        return eeg_id, spec_id, TRAIN_SPEC_PATH, TRAIN_EEG_PATH


rows = [
    _path_tuple(str(r.eeg_id), str(r.spectrogram_id))
    for r in need_all.itertuples(index=False)
]

n_jobs = min(4, os.cpu_count() or 1)
_ = Parallel(n_jobs=n_jobs, prefer="processes", batch_size=4)(
    delayed(save_from_paths)(eeg_id, spec_id, sp, ep)
    for (eeg_id, spec_id, sp, ep) in rows
)

img_size = (224, 224)

train_ds = ImageFolder(train_df, img_size, targets=train_y)
val_ds = ImageFolder(val_df, img_size, targets=val_y)

num_workers = min(4, os.cpu_count() or 1)
pin = torch.cuda.is_available()

train_loader = DataLoader(
    train_ds,
    batch_size=8,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True,
    pin_memory=pin,
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
    in_order=True,
)
val_loader = DataLoader(
    val_ds,
    batch_size=8,
    shuffle=False,
    num_workers=num_workers,
    drop_last=False,
    pin_memory=pin,
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
    in_order=True,
)

model = Net("vit_base_patch14_reg4_dinov2.lvd142m", device_id=0).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3)


def kldiv_loss_from_logits(logits, target_probs):
    logp = F.log_softmax(logits, dim=1)
    target = torch.clamp(target_probs, 1e-8, 1.0)
    target = target / target.sum(dim=1, keepdim=True)
    return F.kl_div(logp, target, reduction="batchmean")


epochs = 2
best_val = float("inf")
best_state = None

for ep in range(1, epochs + 1):
    model.train()
    tr_loss = 0.0
    ntr = 0
    for (
        spec_imgs,
        eeg_imgs,
        raw_50s_imgs,
        raw_10s_l_imgs,
        raw_10s_c_imgs,
        raw_10s_r_imgs,
        y,
    ) in train_loader:
        spec_imgs = spec_imgs.to(device, non_blocking=True).float()
        eeg_imgs = eeg_imgs.to(device, non_blocking=True).float()
        raw_50s_imgs = raw_50s_imgs.to(device, non_blocking=True).float()
        raw_10s_l_imgs = raw_10s_l_imgs.to(device, non_blocking=True).float()
        raw_10s_c_imgs = raw_10s_c_imgs.to(device, non_blocking=True).float()
        raw_10s_r_imgs = raw_10s_r_imgs.to(device, non_blocking=True).float()
        y = torch.as_tensor(y, device=device, dtype=torch.float32)

        optimizer.zero_grad(set_to_none=True)

        logits_l, _, _, _, _ = model(spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_l_imgs)
        logits_c, _, _, _, _ = model(spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_c_imgs)
        logits_r, _, _, _, _ = model(spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_r_imgs)
        logits = (logits_l + logits_c + logits_r) / 3.0

        loss = kldiv_loss_from_logits(logits, y)
        loss.backward()
        optimizer.step()

        tr_loss += float(loss.item()) * spec_imgs.size(0)
        ntr += spec_imgs.size(0)

    model.eval()
    va_loss = 0.0
    nva = 0
    with torch.no_grad():
        for (
            spec_imgs,
            eeg_imgs,
            raw_50s_imgs,
            raw_10s_l_imgs,
            raw_10s_c_imgs,
            raw_10s_r_imgs,
            y,
        ) in val_loader:
            spec_imgs = spec_imgs.to(device, non_blocking=True).float()
            eeg_imgs = eeg_imgs.to(device, non_blocking=True).float()
            raw_50s_imgs = raw_50s_imgs.to(device, non_blocking=True).float()
            raw_10s_l_imgs = raw_10s_l_imgs.to(device, non_blocking=True).float()
            raw_10s_c_imgs = raw_10s_c_imgs.to(device, non_blocking=True).float()
            raw_10s_r_imgs = raw_10s_r_imgs.to(device, non_blocking=True).float()
            y = torch.as_tensor(y, device=device, dtype=torch.float32)

            logits_l, _, _, _, _ = model(
                spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_l_imgs
            )
            logits_c, _, _, _, _ = model(
                spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_c_imgs
            )
            logits_r, _, _, _, _ = model(
                spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_r_imgs
            )
            logits = (logits_l + logits_c + logits_r) / 3.0

            loss = kldiv_loss_from_logits(logits, y)
            va_loss += float(loss.item()) * spec_imgs.size(0)
            nva += spec_imgs.size(0)

    tr_loss /= max(1, ntr)
    va_loss /= max(1, nva)
    print(f"epoch {ep}/{epochs} train_kld={tr_loss:.5f} val_kld={va_loss:.5f}")

    if va_loss < best_val:
        best_val = va_loss
        best_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }

if best_state is not None:
    model.load_state_dict(best_state, strict=True)

train_prior = vote_sum.sum(axis=0).values.astype(np.float32)
train_prior = train_prior / train_prior.sum()
train_prior = np.clip(train_prior, 1e-8, 1.0)
train_prior = train_prior / train_prior.sum()
print("Train prior:", dict(zip(CLASSES, train_prior.tolist())))



## === cell 13
test_inf_df = test.drop_duplicates("eeg_id")[
    ["eeg_id", "spectrogram_id", "patient_id"]
].copy()
test_inf_df["eeg_id"] = test_inf_df["eeg_id"].astype(str)

test_data = ImageFolder(test_inf_df, img_size, targets=None)
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

result_5 = {}

model.eval()
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

        logits_l, _, _, _, _ = model(spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_l_imgs)
        logits_c, _, _, _, _ = model(spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_c_imgs)
        logits_r, _, _, _, _ = model(spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_r_imgs)
        probs = (logits_l.softmax(1) + logits_c.softmax(1) + logits_r.softmax(1)) / 3.0
        probs = probs.detach().cpu().numpy().astype(np.float32, copy=False)

        for j in range(len(eeg_ids)):
            result_5[str(eeg_ids[j])] = probs[j]



## === cell 14
del model
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

sample_sub = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
)
for c in CLASSES:
    if c not in sample_sub.columns:
        raise ValueError(f"Missing required submission column: {c}")

pred_mat = np.zeros((len(sample_sub), 6), dtype=np.float32)
missing = 0
for i, eeg_id in enumerate(sample_sub["eeg_id"].astype(str).values):
    if eeg_id in result_5:
        pred_mat[i] = result_5[eeg_id]
    else:
        missing += 1
        pred_mat[i] = np.ones(6, dtype=np.float32) / 6.0

alpha = 0.15
pred_mat = (1.0 - alpha) * pred_mat + alpha * train_prior[None, :]

pred_mat = np.clip(pred_mat, 1e-8, 1e8)
pred_mat = pred_mat / pred_mat.sum(axis=1, keepdims=True)

sub = sample_sub.copy()
sub[CLASSES] = pred_mat
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv. Missing preds filled with uniform:", missing)
print(sub.head())



## === cell 15
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
