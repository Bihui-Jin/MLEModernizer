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

albumentations==2.0.8
geopandas==0.14.4
librosa==0.11.0
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
PyWavelets==1.8.0
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

# 5. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os



## === cell 1
import albumentations as A
import gc
import librosa
import matplotlib.pyplot as plt
import math
import multiprocessing
import pywt
import random
import time
import timm
import torch
import torch.nn as nn

from albumentations.pytorch import ToTensorV2
from glob import glob
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Dict, List, Optional

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using", torch.cuda.device_count(), "GPU(s)")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

torch.set_num_threads(max(1, min(os.cpu_count() or 1, 8)))
torch.set_num_interop_threads(max(1, min(os.cpu_count() or 1, 8)))




## === cell 2
class config:
    model1 = "resnet50d"
    model2 = "vit_base_patch16_224"
    model3 = "resnet34d"
    epoch = 10
    lr = 1e-3
    batchsize = 32
    splits = 5
    momentum = 0.9
    MAX_GRAD_NORM = 1e7
    WEIGHT_DECAY = 0.01
    device = "cpu"
    FOLDS = 5
    AMP = True


class paths:
    preloadedeeg = "/kaggle/input/brain-eeg-spectrograms/eeg_specs.npy"
    train_eeg_dir = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs"
    train_spec_dir = (
        "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms"
    )
    train_csv = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    test_csv = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
    test_eeg = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs"
    test_spec = (
        "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms"
    )
    out = "/kaggle/working/"




## === cell 3
USE_WAVELET = None

NAMES = ["LL", "LP", "RP", "RR"]

FEATS = [
    ["Fp1", "F7", "T3", "T5", "O1"],
    ["Fp1", "F3", "C3", "P3", "O1"],
    ["Fp2", "F8", "T4", "T6", "O2"],
    ["Fp2", "F4", "C4", "P4", "O2"],
]


def maddest(d, axis: int = None):
    """
    Denoise function.
    """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def denoise(x: np.ndarray, wavelet: str = "haar", level: int = 1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1 / 0.6745) * maddest(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="hard") for i in coeff[1:])
    output = pywt.waverec(coeff, wavelet, mode="per")
    return output


_MEL_FBANK_CACHE = {}


def _get_mel_fbank(sr, n_fft, n_mels, fmin, fmax):
    key = (sr, n_fft, n_mels, fmin, fmax)
    fb = _MEL_FBANK_CACHE.get(key)
    if fb is None:
        fb = librosa.filters.mel(
            sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
        )
        _MEL_FBANK_CACHE[key] = fb
    return fb


def spectrogram_from_eeg(parquet_path, display=False):
    eeg = pd.read_parquet(parquet_path)
    middle = (len(eeg) - 10_000) // 2
    eeg = eeg.iloc[middle : middle + 10_000]

    img = np.zeros((128, 256, 4), dtype="float32")

    if display:
        plt.figure(figsize=(10, 7))
    signals = []
    for k in range(4):
        COLS = FEATS[k]
        for kk in range(4):
            x = eeg[COLS[kk]].values - eeg[COLS[kk + 1]].values

            m = np.nanmean(x)
            if np.isnan(x).mean() < 1:
                x = np.nan_to_num(x, nan=m)
            else:
                x[:] = 0

            if USE_WAVELET:
                x = denoise(x, wavelet=USE_WAVELET)
            signals.append(x)

            sr = 200
            hop_length = len(x) // 256
            n_fft = 1024
            n_mels = 128
            fmin = 0
            fmax = 20
            win_length = 128

            mel_fbank = _get_mel_fbank(sr, n_fft, n_mels, fmin, fmax)
            mel_spec = librosa.feature.melspectrogram(
                y=x,
                sr=sr,
                hop_length=hop_length,
                n_fft=n_fft,
                n_mels=n_mels,
                fmin=fmin,
                fmax=fmax,
                win_length=win_length,
                htk=True,
                norm="slaney",
                mel_basis=mel_fbank,
            )

            width = (mel_spec.shape[1] // 32) * 32
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[
                :, :width
            ]

            mel_spec_db = (mel_spec_db + 40) / 40
            img[:, :, k] += mel_spec_db

        img[:, :, k] /= 4.0

        if display:
            plt.subplot(2, 2, k + 1)
            plt.imshow(img[:, :, k], aspect="auto", origin="lower")
            plt.title(f"Spectrogram {NAMES[k]}")

    if display:
        plt.show()
    return img




## === cell 4
df = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/test.csv")




## === cell 5
def _eeg_to_mel_item(fn: str):
    sp = spectrogram_from_eeg(os.path.join(paths.test_eeg, fn))
    name = int(fn.split(".")[0])
    return name, np.array(sp)


test_eeg_files = [f for f in os.listdir(paths.test_eeg) if f.endswith(".parquet")]
all_eegs = {}
n_proc = max(1, min(multiprocessing.cpu_count(), 8))  # cap to avoid oversubscription
with multiprocessing.get_context("fork").Pool(processes=n_proc) as pool:
    for name, sp in tqdm(
        pool.imap_unordered(_eeg_to_mel_item, test_eeg_files, chunksize=8),
        total=len(test_eeg_files),
        desc="Loading test EEGs -> mel images (parallel)",
    ):
        all_eegs[name] = sp



## === cell 6
all_eegs  # will display dict summary in notebook environments



## === cell 7
test_df = pd.read_csv(paths.test_csv)
print(f"Test dataframe shape is: {test_df.shape}")
test_df.head()




## === cell 8
def _load_spec_item(fn: str):
    sp = pd.read_parquet(os.path.join(paths.test_spec, fn))
    name = int(fn.split(".")[0])
    return name, np.array(sp)


test_spec_files = [f for f in os.listdir(paths.test_spec) if f.endswith(".parquet")]
all_spectrograms = {}
with multiprocessing.get_context("fork").Pool(processes=n_proc) as pool:
    for name, sp in tqdm(
        pool.imap_unordered(_load_spec_item, test_spec_files, chunksize=16),
        total=len(test_spec_files),
        desc="Loading test spectrograms (parallel)",
    ):
        all_spectrograms[name] = sp



## === cell 9
all_spectrograms  # will display dict summary in notebook environments



## === cell 10
TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
targets = TARGETS  # keep original variable name expectation

train_df_raw = pd.read_csv(paths.train_csv)
train_df_raw[TARGETS] = train_df_raw[TARGETS].astype(np.float32)
train_df_raw["total_votes"] = train_df_raw[TARGETS].sum(axis=1).astype(np.float32)
train_df_raw["spec_r"] = (
    (train_df_raw["spectrogram_label_offset_seconds"].astype(np.float32) * 2.0)
    .round()
    .astype(np.int32)
)

train_df = (
    train_df_raw.sort_values(["eeg_id", "total_votes"], ascending=[True, False])
    .groupby("eeg_id", as_index=False)
    .agg(
        {
            "spectrogram_id": "first",
            "patient_id": "first",
            "spec_r": "median",
            "seizure_vote": "sum",
            "lpd_vote": "sum",
            "gpd_vote": "sum",
            "lrda_vote": "sum",
            "grda_vote": "sum",
            "other_vote": "sum",
        }
    )
)
train_df["spec_r"] = train_df["spec_r"].round().astype(np.int32)
print("Train consolidated shape:", train_df.shape)
print(train_df.head())




## === cell 11
def load_parquet_arrays_by_ids(
    dir_path: str, ids: np.ndarray, desc: str
) -> dict[int, np.ndarray]:
    out = {}
    ids = np.asarray(ids, dtype=np.int64)
    fns = [f"{int(i)}.parquet" for i in ids]
    for fn in tqdm(fns, desc=desc):
        path = os.path.join(dir_path, fn)
        if os.path.exists(path):
            out[int(fn.split(".")[0])] = np.array(pd.read_parquet(path))
    return out


train_spec_ids = train_df["spectrogram_id"].astype(int).unique()
train_specs = load_parquet_arrays_by_ids(
    paths.train_spec_dir, train_spec_ids, "Loading train spectrograms (subset)"
)

train_eeg_ids = train_df["eeg_id"].astype(int).unique()
train_eeg_files = [
    f"{int(i)}.parquet"
    for i in train_eeg_ids
    if os.path.exists(os.path.join(paths.train_eeg_dir, f"{int(i)}.parquet"))
]


def _train_eeg_to_mel_item(fn: str):
    key = int(fn.split(".")[0])
    sp = spectrogram_from_eeg(os.path.join(paths.train_eeg_dir, fn))
    return key, np.array(sp)


train_eegs = {}
with multiprocessing.get_context("fork").Pool(processes=n_proc) as pool:
    for key, sp in tqdm(
        pool.imap_unordered(_train_eeg_to_mel_item, train_eeg_files, chunksize=8),
        total=len(train_eeg_files),
        desc="Loading train EEGs -> mel images (subset, parallel)",
    ):
        train_eegs[key] = sp

print("Loaded train_specs:", len(train_specs), "train_eegs:", len(train_eegs))




## === cell 12
class CustomDataset:
    def __init__(
        self,
        traindf: pd.DataFrame,
        config,
        mode: str = "train",
        specs: Optional[dict[int, np.ndarray]] = None,
        eegs: Optional[dict[int, np.ndarray]] = None,
    ):
        self.traindf = traindf.reset_index(drop=True)
        self.specs = specs
        self.eeg = eegs
        self.mode = mode
        self._x_cache: dict[int, np.ndarray] = {}  # eeg_id -> (3,512,512) float32

    def __len__(self):
        return len(self.traindf)

    def _get_spec_region_img(
        self, spec_arr: np.ndarray, r: int, region: int
    ) -> np.ndarray:
        """
        Returns a (100,256) float32 normalized image for one region (freq x time).
        """
        if spec_arr.ndim != 2:
            spec_arr = np.squeeze(spec_arr)
            if spec_arr.ndim != 2:
                spec_arr = spec_arr.reshape(spec_arr.shape[0], -1)

        tmax, fmax = spec_arr.shape[0], spec_arr.shape[1]
        r = int(np.clip(r, 0, max(0, tmax - 1)))
        t_end = min(tmax, r + 300)

        f_start = region * 100
        f_end = min(fmax, (region + 1) * 100)
        if f_start >= fmax:
            raw = np.zeros((100, max(1, t_end - r)), dtype=np.float32)
        else:
            raw = spec_arr[r:t_end, f_start:f_end].T.astype(np.float32, copy=False)

        raw = np.clip(raw, np.exp(-4), np.exp(8))
        raw = np.log(raw)

        ep = 1e-6
        mu = np.nanmean(raw)
        std = np.nanstd(raw)
        raw = (raw - mu) / (std + ep)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

        freq_target = 100
        time_target = 256

        if raw.shape[0] < freq_target:
            pad = np.zeros((freq_target - raw.shape[0], raw.shape[1]), dtype=np.float32)
            raw = np.concatenate([raw, pad], axis=0)
        elif raw.shape[0] > freq_target:
            raw = raw[:freq_target, :]

        if raw.shape[1] < time_target:
            pad = np.zeros((raw.shape[0], time_target - raw.shape[1]), dtype=np.float32)
            raw = np.concatenate([raw, pad], axis=1)
        elif raw.shape[1] > time_target:
            raw = raw[:, :time_target]

        return raw

    def _build_x(self, row) -> np.ndarray:
        X = np.zeros((128, 256, 8), dtype=np.float32)

        if self.mode == "test":
            r = 0
        else:
            r = int(row.get("spec_r", 0))

        spec_arr = self.specs[int(row.spectrogram_id)]
        for region in range(4):
            img_region = self._get_spec_region_img(spec_arr, r, region)  # (100,256)
            X[14:-14, :, region] = img_region / 2.0

        eeg_img = self.eeg[int(row.eeg_id)]  # (128,256,4)
        X[:, :, 4:] = eeg_img

        spectograms = [X[:, :, i : i + 1] for i in range(4)]
        spectograms = np.concatenate(spectograms, axis=0)  # (512,256,1)

        eegs = [X[:, :, i : i + 1] for i in range(4, 8)]
        eegs = np.concatenate(eegs, axis=0)  # (512,256,1)

        x = np.concatenate([spectograms, eegs], axis=1)  # (512,512,1)
        x = np.concatenate([x, x, x], axis=2)  # (512,512,3)
        x = np.transpose(x, (2, 0, 1))  # (3,512,512)
        return x.astype(np.float32, copy=False)

    def __getitem__(self, idx):
        row = self.traindf.iloc[idx]
        eeg_id = int(row.eeg_id)

        x = self._x_cache.get(eeg_id)
        if x is None:
            x = self._build_x(row)
            self._x_cache[eeg_id] = x

        if self.mode != "test":
            v = row[targets].values.astype(np.float32, copy=False)
            v_sum = float(np.sum(v))
            if v_sum > 0:
                y = v / v_sum
            else:
                y = np.full(6, 1.0 / 6.0, dtype=np.float32)
        else:
            y = np.zeros(6, dtype=np.float32)

        return {"data": x, "target": y}




## === cell 13
customdataset = CustomDataset(
    test_df, config, mode="test", specs=all_spectrograms, eegs=all_eegs
)



## === cell 14
customdataset[0]



## === cell 15
test_df.iloc[0].spectrogram_id




## === cell 16
def fast_collate_fn(batch):
    data = np.stack([b["data"] for b in batch], axis=0)  # (B,3,512,512)
    target = np.stack([b["target"] for b in batch], axis=0)  # (B,6)
    data_t = torch.from_numpy(data)
    target_t = torch.from_numpy(target)
    return {"data": data_t, "target": target_t}


def make_loader(ds, batch_size, shuffle, num_workers):
    num_workers = int(max(0, num_workers))
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
        collate_fn=fast_collate_fn,
        drop_last=False,
    )


test_loader = make_loader(
    customdataset,
    batch_size=config.batchsize,
    shuffle=False,
    num_workers=min(8, multiprocessing.cpu_count()),
)
X = customdataset[0]["data"]
y = customdataset[0]["target"]
y




## === cell 17
class Custommodel(nn.Module):
    def __init__(self, config, numclass: int = 6):
        super(Custommodel, self).__init__()
        self.model = timm.create_model(
            config.model1,
            pretrained=False,
            drop_rate=0.1,
            drop_path_rate=0.2,
        )
        self.features = nn.Sequential(*list(self.model.children())[:-2])
        self.customlayer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.model.num_features, numclass),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.customlayer(x)
        return x




## === cell 18
def inference_function(test_loader, model, device):
    model.eval()
    softmax = nn.Softmax(dim=1)
    preds = []
    with torch.inference_mode():
        for batch in tqdm(test_loader, desc="Inference", leave=False):
            x = batch["data"].to(device, non_blocking=True)
            if device.type == "cuda":
                x = x.to(memory_format=torch.channels_last)
            ypred = model(x)
            ypred = softmax(ypred)
            preds.append(ypred.detach().cpu().numpy())
    return {"predictions": np.concatenate(preds, axis=0)}




## === cell 19
_TEST_DATASET_ONCE = CustomDataset(
    test_df, config, mode="test", specs=all_spectrograms, eegs=all_eegs
)
_TEST_LOADER_ONCE = make_loader(
    _TEST_DATASET_ONCE,
    batch_size=config.batchsize,
    shuffle=False,
    num_workers=min(8, multiprocessing.cpu_count()),
)


def predict_from_weight_dir(weight_dir: str, model_ctor, model_name: str):
    if not os.path.isdir(weight_dir):
        print(f"Weight dir not found, skipping {model_name}: {weight_dir}")
        return None

    weight_files = sorted(
        [f for f in os.listdir(weight_dir) if f.endswith((".pt", ".pth", ".bin"))]
    )
    if len(weight_files) == 0:
        print(f"No weight files found in {weight_dir}, skipping {model_name}")
        return None

    preds_list = []
    testloader = _TEST_LOADER_ONCE

    for wf in weight_files:
        wpath = os.path.join(weight_dir, wf)
        try:
            dd = torch.load(wpath, map_location="cpu")
        except Exception as e:
            print(f"Failed to load {wpath} ({e}), skipping.")
            continue

        model = model_ctor()
        if isinstance(dd, dict) and "model" in dd:
            state = dd["model"]
        else:
            state = dd
        try:
            model.load_state_dict(state, strict=True)
        except Exception as e:
            print(f"State dict mismatch for {wpath} ({e}), skipping.")
            continue

        model.to(device)
        if device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)
        pred = inference_function(testloader, model, device)["predictions"]
        preds_list.append(pred)

        del model, dd, state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(preds_list) == 0:
        print(f"No usable weights loaded for {model_name}.")
        return None

    preds = np.mean(np.stack(preds_list, axis=0), axis=0)
    return preds


predictions = predict_from_weight_dir(
    "/kaggle/input/resnet5010ep2",
    model_ctor=lambda: Custommodel(config),
    model_name="resnet50d",
)




## === cell 20
class Custommodelkk(nn.Module):
    def __init__(self, config, numclass: int = 6):
        super(Custommodelkk, self).__init__()
        self.model = timm.create_model(
            config.model3,
            pretrained=False,
            drop_rate=0.1,
            drop_path_rate=0.2,
        )
        self.features = nn.Sequential(*list(self.model.children())[:-2])
        self.customlayer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.model.num_features, numclass),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.customlayer(x)
        return x




## === cell 21
predictions3 = predict_from_weight_dir(
    "/kaggle/input/resnet34d2",
    model_ctor=lambda: Custommodelkk(config),
    model_name="resnet34d",
)



## === cell 22
predictions  # possibly None if no weights



## === cell 23
from torchvision.transforms import transforms

tras = transforms.Compose([transforms.Resize((224, 224))])


class Custommodel2(nn.Module):
    def __init__(self, config, transform, numclass: int = 6):
        super(Custommodel2, self).__init__()
        self.model = timm.create_model(
            config.model2,
            pretrained=False,
        )
        self.model.head = nn.Linear(self.model.head.in_features, numclass)
        self.transform = transform

    def forward(self, x):
        x = self.transform(x)
        x = self.model(x)
        return x




## === cell 24
predictions2 = predict_from_weight_dir(
    "/kaggle/input/visiontransformer",
    model_ctor=lambda: Custommodel2(config, tras),
    model_name="vit_base_patch16_224",
)




## === cell 25
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)


def make_patient_folds(
    df_in: pd.DataFrame, n_folds: int = 5, seed: int = 42
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pats = df_in["patient_id"].astype(int).values
    uniq = np.unique(pats)
    rng.shuffle(uniq)
    fold_map = {}
    for i, p in enumerate(uniq):
        fold_map[p] = i % n_folds
    return np.array([fold_map[p] for p in pats], dtype=np.int32)


folds = make_patient_folds(train_df, n_folds=config.FOLDS, seed=42)
train_df = train_df.copy()
train_df["fold"] = folds


def row_normalize(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, eps, None)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs


def train_one_fold(fold: int):
    trn = train_df[train_df["fold"] != fold].reset_index(drop=True)
    val = train_df[train_df["fold"] == fold].reset_index(drop=True)

    ds_trn = CustomDataset(
        trn, config, mode="train", specs=train_specs, eegs=train_eegs
    )
    ds_val = CustomDataset(
        val, config, mode="train", specs=train_specs, eegs=train_eegs
    )

    nw = min(8, multiprocessing.cpu_count())
    dl_trn = make_loader(
        ds_trn, batch_size=config.batchsize, shuffle=True, num_workers=nw
    )
    dl_val = make_loader(
        ds_val, batch_size=config.batchsize, shuffle=False, num_workers=nw
    )

    model = Custommodel(config).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    opt = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.WEIGHT_DECAY
    )

    criterion = nn.KLDivLoss(reduction="batchmean")
    logsoftmax = nn.LogSoftmax(dim=1)

    scaler = torch.cuda.amp.GradScaler(enabled=(config.AMP and device.type == "cuda"))

    for ep in range(config.epoch):
        model.train()
        for batch in dl_trn:
            x = batch["data"].to(device, non_blocking=True)
            y = batch["target"].to(device, non_blocking=True, dtype=torch.float32)

            if device.type == "cuda":
                x = x.to(memory_format=torch.channels_last)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(
                enabled=(config.AMP and device.type == "cuda")
            ):
                logits = model(x)
                logp = logsoftmax(logits)
                loss = criterion(logp, y)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            scaler.step(opt)
            scaler.update()

        model.eval()
        val_losses = []
        with torch.inference_mode():
            for batch in dl_val:
                x = batch["data"].to(device, non_blocking=True)
                y = batch["target"].to(device, non_blocking=True, dtype=torch.float32)
                if device.type == "cuda":
                    x = x.to(memory_format=torch.channels_last)
                logits = model(x)
                logp = logsoftmax(logits)
                val_losses.append(float(criterion(logp, y).detach().cpu()))
        print(
            f"Fold {fold} Epoch {ep+1}/{config.epoch} val_KL: {np.mean(val_losses):.5f}"
        )

    return model


have_external = any(
    isinstance(p, np.ndarray) for p in [predictions, predictions2, predictions3]
)
predictions_cv = None
if not have_external:
    fold_models = []
    for f in range(config.FOLDS):
        print("Training fold", f)
        m = train_one_fold(f)
        fold_models.append(m)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    testdataset_cv = _TEST_DATASET_ONCE
    testloader_cv = _TEST_LOADER_ONCE

    cv_preds = []
    for m in fold_models:
        p = inference_function(testloader_cv, m, device)["predictions"]
        cv_preds.append(p.astype(np.float64))
    predictions_cv = np.mean(np.stack(cv_preds, axis=0), axis=0)




## === cell 26
def train_prior_from_votes(train_csv_path: str) -> np.ndarray:
    tr = pd.read_csv(train_csv_path, usecols=TARGETS)
    vote_sum = tr[TARGETS].sum(axis=0).values.astype(np.float64)
    vote_sum = np.clip(vote_sum, 1e-12, None)
    prior = vote_sum / vote_sum.sum()
    return prior


n_test = len(test_df)

available = [
    p
    for p in [predictions_cv, predictions, predictions2, predictions3]
    if isinstance(p, np.ndarray)
]
if len(available) == 0:
    prior = train_prior_from_votes(paths.train_csv)  # shape (6,)
    finalpred = np.tile(prior[None, :], (n_test, 1))
else:
    finalpred = np.zeros_like(available[0], dtype=np.float64)
    for p in available:
        finalpred += p.astype(np.float64)
    finalpred /= len(available)

finalpred = row_normalize(finalpred)



## === cell 27
finalpred



## === cell 28
sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
sub[TARGETS] = finalpred.astype(np.float32)
sub.to_csv("submission.csv", index=False)
print(f"Submission shape: {sub.shape}")
print(sub.head())



## === cell 29
print(
    "Row sums (min/mean/max):",
    sub[TARGETS].sum(axis=1).min(),
    sub[TARGETS].sum(axis=1).mean(),
    sub[TARGETS].sum(axis=1).max(),
)
print("Total sum:", float(np.sum(finalpred)))
