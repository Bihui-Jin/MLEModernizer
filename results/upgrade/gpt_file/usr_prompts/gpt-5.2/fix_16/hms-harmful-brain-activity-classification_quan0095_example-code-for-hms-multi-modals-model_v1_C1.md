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
os.environ.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 2) // 2)))
os.environ.setdefault("MKL_NUM_THREADS", str(max(1, (os.cpu_count() or 2) // 2)))



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
    Xseg = X_all[time_start:time_stop]

    list_eeg = []
    for _, (_, c1_list, c2_list) in _RAW_REGION_PRECOMP.items():
        A = Xseg[:, [_RAW_COL_IDX[c] for c in c1_list]].T  # (4,T)
        B = Xseg[:, [_RAW_COL_IDX[c] for c in c2_list]].T  # (4,T)

        eeg = A - B
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




## === cell 6
def _resized_cached(eeg_id: str):
    return False


def save_one(eeg_id, spec_id, out_hw=(518, 518), verbose=False):
    return 0


uniq = test.drop_duplicates("eeg_id")[["eeg_id", "spectrogram_id"]].reset_index(
    drop=True
)
print("Skipping precompute/caching. Unique test eeg_ids:", len(uniq))




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
def _resize2d_to_hw_scipy(arr2d: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    x = np.asarray(arr2d, dtype=np.float32)
    out_h, out_w = out_hw
    y = signal.resample(x, out_w, axis=1)
    y = signal.resample(y, out_h, axis=0)
    return np.asarray(y, dtype=np.float32)


def _spec_from_parquet(spec_path: str) -> np.ndarray:
    spec = pd.read_parquet(spec_path)
    return spec.values[:, 1:301].T.astype("float32", copy=False)  # (300, T)


from collections import OrderedDict


class _SpecImageCache:
    def __init__(self, spec_path_prefix: str, out_hw=(518, 518), max_items: int = 4096):
        self.spec_path_prefix = spec_path_prefix
        self.out_hw = tuple(out_hw)
        self._cache = OrderedDict()
        self._eps = 1e-6
        self.max_items = int(max_items)

    def get(self, spectrogram_id: str) -> np.ndarray:
        img = self._cache.get(spectrogram_id)
        if img is not None:
            self._cache.move_to_end(spectrogram_id)
            return img

        arr = _spec_from_parquet(f"{self.spec_path_prefix}{spectrogram_id}.parquet")
        img = _resize2d_to_hw_scipy(arr, self.out_hw)[..., None]  # (H,W,1)

        img = np.clip(img, np.exp(-4), np.exp(8))
        img = np.log(img)
        img = np.nan_to_num(img, nan=0.0)
        img_mean = img.mean(axis=(0, 1))
        img_std = img.std(axis=(0, 1))
        img = (img - img_mean) / (img_std + self._eps)
        img = img.astype(np.float32, copy=False)

        self._cache[spectrogram_id] = img
        if len(self._cache) > self.max_items:
            self._cache.popitem(last=False)
        return img


_SPEC_CACHE = _SpecImageCache(SPEC_PATH, out_hw=(518, 518))



## === cell 11
_unique_spec_ids = test["spectrogram_id"].astype(str).unique().tolist()
_SPEC_PRECOMP = {}
for sid in _unique_spec_ids:
    _SPEC_PRECOMP[sid] = _SPEC_CACHE.get(sid)
_SPEC_CACHE._cache.clear()
gc.collect()


class ImageFolder(data.Dataset):
    def __init__(self, df, test_imgsize, spec_cache: _SpecImageCache):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.test_imgsize = tuple(test_imgsize)
        H, W = self.test_imgsize

        self.spec_cache = spec_cache
        self._fallback = np.zeros((H, W, 1), dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        eeg_id = str(row.eeg_id)
        spec_id = str(row.spectrogram_id)

        spec_img = _SPEC_PRECOMP.get(spec_id)
        if spec_img is None:
            spec_img = self.spec_cache.get(spec_id)

        if DEBUG:
            eeg_df = pd.read_parquet(
                f"{EEG_PATH}{eeg_id}.parquet", columns=_EEG_READ_COLS
            )
            X_all = eeg_df.to_numpy(dtype=np.float32, copy=True)
            raw_idx = [eeg_df.columns.get_loc(c) for c in _ALL_RAW_COLS]
            stft_idx = [eeg_df.columns.get_loc(c) for c in _ALL_STFT_COLS]
            X_raw = X_all[:, raw_idx]
            _nanfill_inplace(X_raw)
            X_stft = X_all[:, stft_idx]
            img_l, img_c, img_r = raw10seeg_from_eeg_np(X_raw)
            img50 = raw50seeg_from_eeg_np(X_raw)
            imgstft = stft_spec_from_eeg_np(X_stft)

            eeg_img = _resize2d_to_hw_scipy(imgstft, self.test_imgsize)[..., None]
            raw_50s_img = _resize2d_to_hw_scipy(img50, self.test_imgsize)[..., None]
            raw_10s_l_img = _resize2d_to_hw_scipy(img_l, self.test_imgsize)[..., None]
            raw_10s_c_img = _resize2d_to_hw_scipy(img_c, self.test_imgsize)[..., None]
            raw_10s_r_img = _resize2d_to_hw_scipy(img_r, self.test_imgsize)[..., None]
        else:
            eeg_img = self._fallback
            raw_50s_img = self._fallback
            raw_10s_l_img = self._fallback
            raw_10s_c_img = self._fallback
            raw_10s_r_img = self._fallback

        return (
            spec_img,
            eeg_img,
            raw_50s_img,
            raw_10s_l_img,
            raw_10s_c_img,
            raw_10s_r_img,
            eeg_id,
        )




## === cell 12
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




## === cell 13
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

test_data = ImageFolder(test, (518, 518), _SPEC_CACHE)

num_workers = 0
test_loader = DataLoader(
    test_data,
    batch_size=32,
    pin_memory=torch.cuda.is_available(),
    num_workers=num_workers,
    drop_last=False,
    persistent_workers=False,
)

sample_sub = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
)
eeg_ids_all = sample_sub["eeg_id"].astype(str).to_numpy()
idx_map = {eid: i for i, eid in enumerate(eeg_ids_all)}
out = np.full((len(eeg_ids_all), 6), 1.0 / 6.0, dtype=np.float64)

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
            raw10_stack = torch.cat(
                [raw_10s_l_imgs, raw_10s_c_imgs, raw_10s_r_imgs], dim=0
            )
            bs = spec_imgs.shape[0]
            spec_stack = (
                spec_imgs.unsqueeze(0)
                .expand(3, -1, -1, -1, -1)
                .reshape(3 * bs, *spec_imgs.shape[1:])
                .contiguous()
            )
            eeg_stack = (
                eeg_imgs.unsqueeze(0)
                .expand(3, -1, -1, -1, -1)
                .reshape(3 * bs, *eeg_imgs.shape[1:])
                .contiguous()
            )
            raw50_stack = (
                raw_50s_imgs.unsqueeze(0)
                .expand(3, -1, -1, -1, -1)
                .reshape(3 * bs, *raw_50s_imgs.shape[1:])
                .contiguous()
            )

            for model in vit_models:
                model.eval()
                logits_stack, _, _, _, _ = model(
                    spec_stack, eeg_stack, raw50_stack, raw10_stack
                )
                probs_stack = logits_stack.softmax(dim=1)

                probs_l = probs_stack[0:bs]
                probs_c = probs_stack[bs : 2 * bs]
                probs_r = probs_stack[2 * bs : 3 * bs]
                ensemble_probs += (probs_l + probs_c + probs_r) / 3.0

            ensemble_probs /= len(vit_models)

        ensemble_probs_np = (
            ensemble_probs.detach().cpu().numpy().astype(np.float64, copy=False)
        )
        for eid, p in zip(eeg_ids, ensemble_probs_np):
            j = idx_map.get(str(eid))
            if j is not None:
                out[j] = p



## === cell 14
for model in vit_models:
    del model
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()



## === cell 15
out = np.clip(out, 1e-12, None)
out = out / out.sum(axis=1, keepdims=True)

df = pd.DataFrame(out, columns=CLASSES)
df.insert(0, "eeg_id", eeg_ids_all)

df.to_csv("submission.csv", index=False)
print(df.head())
print("Saved submission.csv with shape:", df.shape)



## === cell 16
if DEBUG == False:
    import shutil

    for p in [
        "/kaggle/working/resized_518/",
    ]:
        if os.path.exists(p):
            shutil.rmtree(p, ignore_errors=True)
