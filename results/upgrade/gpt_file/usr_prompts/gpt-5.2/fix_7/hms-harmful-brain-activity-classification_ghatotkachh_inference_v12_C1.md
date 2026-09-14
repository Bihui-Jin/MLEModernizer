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

import albumentations as A  # kept (import side-effects / compatibility)
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

from albumentations.pytorch import ToTensorV2  # kept
from glob import glob
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using", torch.cuda.device_count(), "GPU(s)")

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




## === cell 1
class config:
    model = "resnet50d"
    epoch = 2  # unchanged (only used if no pretrained weights exist)
    lr = 1e-3
    batchsize = 32
    splits = 5
    momentum = 0.9
    MAX_GRAD_NORM = 1e7
    WEIGHT_DECAY = 0.01
    device = "cpu"
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


TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]



## === cell 2
USE_WAVELET = None

NAMES = ["LL", "LP", "RP", "RR"]

FEATS = [
    ["Fp1", "F7", "T3", "T5", "O1"],
    ["Fp1", "F3", "C3", "P3", "O1"],
    ["Fp2", "F8", "T4", "T6", "O2"],
    ["Fp2", "F4", "C4", "P4", "O2"],
]


def maddest(d, axis: int = None):
    """Denoise function."""
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def denoise(x: np.ndarray, wavelet: str = "haar", level: int = 1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1 / 0.6745) * maddest(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="hard") for i in coeff[1:])
    output = pywt.waverec(coeff, wavelet, mode="per")
    return output


_MEL_KW = dict(sr=200, n_fft=1024, n_mels=128, fmin=0, fmax=20)
_MEL_FB = librosa.filters.mel(**_MEL_KW).astype(np.float32)


def _melspectrogram_fast(
    y: np.ndarray, hop_length: int, win_length: int = 128
) -> np.ndarray:
    S = librosa.core.stft(
        y=y,
        n_fft=_MEL_KW["n_fft"],
        hop_length=hop_length,
        win_length=win_length,
        window="hann",
        center=True,
        pad_mode="reflect",
    )
    P = (np.abs(S) ** 2).astype(np.float32)
    return _MEL_FB @ P  # (n_mels, t)


def spectrogram_from_eeg(parquet_path, display=False):
    cols = []
    for k in range(4):
        cols.extend(FEATS[k])
    cols = sorted(set(cols))
    eeg = pd.read_parquet(parquet_path, columns=cols)
    middle = (len(eeg) - 10_000) // 2
    eeg = eeg.iloc[middle : middle + 10_000]

    img = np.zeros((128, 256, 4), dtype="float32")

    if display:
        plt.figure(figsize=(10, 7))
    signals = []
    eps = 1e-10
    for k in range(4):
        COLS = FEATS[k]

        for kk in range(4):
            a = eeg[COLS[kk]].to_numpy(copy=False)
            b = eeg[COLS[kk + 1]].to_numpy(copy=False)
            x = a - b

            m = np.nanmean(x)
            if np.isnan(x).mean() < 1:
                x = np.nan_to_num(x, nan=m)
            else:
                x = np.zeros_like(x)

            if USE_WAVELET:
                x = denoise(x, wavelet=USE_WAVELET)
            signals.append(x)

            mel_spec = _melspectrogram_fast(
                y=x.astype(np.float32, copy=False),
                hop_length=len(x) // 256,
                win_length=128,
            )

            width = (mel_spec.shape[1] // 32) * 32
            mel_spec = mel_spec[:, :width].astype(np.float32, copy=False)

            mel_spec_db = np.log(np.maximum(mel_spec, eps)).astype(
                np.float32, copy=False
            )
            mel_spec_db = (mel_spec_db + 40) / 40
            img[:, :, k] += mel_spec_db

        img[:, :, k] /= 4.0

        if display:
            plt.subplot(2, 2, k + 1)
            plt.imshow(img[:, :, k], aspect="auto", origin="lower")
            plt.title(f"Spectrogram {NAMES[k]}")

    if display:
        plt.show()
        plt.figure(figsize=(10, 5))
        offset = 0
        for k in range(4):
            if k > 0:
                offset -= signals[3 - k].min()
            plt.plot(range(10_000), signals[k] + offset, label=NAMES[3 - k])
            offset += signals[3 - k].max()
        plt.legend()
        plt.title("Signals")
        plt.show()
        print()
        print("#" * 25)
        print()

    return img




## === cell 3
test_df = pd.read_csv(paths.test_csv)
print(f"Test dataframe shape is: {test_df.shape}")
test_df.head()



## === cell 4
try:
    _MP_CTX = multiprocessing.get_context("fork")
except ValueError:
    _MP_CTX = multiprocessing.get_context("spawn")


def _eeg_worker(fn: str):
    fp = os.path.join(paths.test_eeg, fn)
    name = int(fn.split(".")[0])
    sp = spectrogram_from_eeg(fp)
    return name, np.asarray(sp, dtype=np.float32)


test_eeg_files = sorted(os.listdir(paths.test_eeg))
all_eegs = {}

nproc = max(1, min(multiprocessing.cpu_count(), 8))  # cap to avoid overhead/ram spikes
chunksize = 32 if len(test_eeg_files) >= 1024 else 16
with _MP_CTX.Pool(processes=nproc) as pool:
    for name, arr in tqdm(
        pool.imap_unordered(_eeg_worker, test_eeg_files, chunksize=chunksize),
        total=len(test_eeg_files),
        desc="Loading test EEGs (parallel)",
    ):
        all_eegs[name] = arr

print("Loaded test EEG specs:", len(all_eegs))



## === cell 5
all_spectrograms = {}
for fn in tqdm(sorted(os.listdir(paths.test_spec)), desc="Loading test spectrograms"):
    name = int(fn.split(".")[0])
    sp = pd.read_parquet(os.path.join(paths.test_spec, fn))
    arr = sp.to_numpy(copy=False)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    all_spectrograms[name] = arr

print("Loaded test spectrogram parquet arrays:", len(all_spectrograms))



## === cell 6
weights_dir = "/kaggle/input/resnet50"
USE_PRETRAINED = os.path.isdir(weights_dir) and len(os.listdir(weights_dir)) > 0
print("USE_PRETRAINED:", USE_PRETRAINED)




## === cell 7
def _safe_patch_100x256(img2d: np.ndarray) -> np.ndarray:
    """
    Bugfix: ensure the spectrogram patch we assign into X[14:-14,:,region]
    is always shape (100, 256), even if slicing yields different sizes.
    This preserves the same normalization logic but removes runtime crashes.
    """
    img2d = np.asarray(img2d, dtype=np.float32)
    img2d = np.nan_to_num(img2d, nan=0.0, posinf=0.0, neginf=0.0)

    if img2d.ndim != 2:
        img2d = img2d.reshape(img2d.shape[0], -1)

    target_h, target_w = 100, 256
    h, w = img2d.shape

    if h <= 1 or w <= 1:
        return np.zeros((target_h, target_w), dtype=np.float32)

    if (h, w) == (target_h, target_w):
        return img2d

    x_old = np.linspace(0.0, 1.0, num=w, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num=target_w, dtype=np.float32)
    tmp = np.vstack(
        [np.interp(x_new, x_old, img2d[i]).astype(np.float32) for i in range(h)]
    )

    y_old = np.linspace(0.0, 1.0, num=h, dtype=np.float32)
    y_new = np.linspace(0.0, 1.0, num=target_h, dtype=np.float32)
    out = np.vstack(
        [np.interp(y_new, y_old, tmp[:, j]).astype(np.float32) for j in range(target_w)]
    ).T

    return out.astype(np.float32)




## === cell 8
class SpecPatchCache:
    def __init__(self, specs: Dict[int, np.ndarray]):
        self.specs = specs
        self._cache: Dict[Tuple[int, int], np.ndarray] = {}

    def get_patches(self, spec_id: int, r: int) -> np.ndarray:
        key = (int(spec_id), int(r))
        out = self._cache.get(key, None)
        if out is not None:
            return out

        spec_arr = self.specs.get(int(spec_id), None)
        patches = np.zeros((4, 100, 256), dtype=np.float32)
        if spec_arr is None:
            self._cache[key] = patches
            return patches

        t_len = spec_arr.shape[0]
        r0 = int(np.clip(r, 0, max(0, t_len - 300)))
        ep = 1e-6

        for region in range(4):
            blk = spec_arr[
                r0 : r0 + 300, region * 100 : (region + 1) * 100
            ]  # (300,100)
            img = blk.T  # (100,300)
            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)
            mu = np.nanmean(img)
            std = np.nanstd(img)
            img = (img - mu) / (std + ep)
            img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

            if img.shape[1] >= 256:
                start = (img.shape[1] - 256) // 2
                patch = img[:, start : start + 256]
            else:
                patch = _safe_patch_100x256(img)

            patches[region] = (patch / 2.0).astype(np.float32, copy=False)

        self._cache[key] = patches
        return patches


class CustomDataset:
    def __init__(
        self,
        traindf: pd.DataFrame,
        config,
        mode: str = "train",
        specs: Dict[int, np.ndarray] = None,
        eegs: Dict[int, np.ndarray] = None,
        patch_cache: Optional[SpecPatchCache] = None,
    ):
        self.traindf = traindf
        self.specs = specs if specs is not None else all_spectrograms
        self.eeg = eegs if eegs is not None else all_eegs
        self.mode = mode

        self._eeg_id = traindf["eeg_id"].to_numpy(np.int64, copy=False)
        self._spec_id = traindf["spectrogram_id"].to_numpy(np.int64, copy=False)
        if mode != "test":
            self._r = (
                (
                    traindf["min"].to_numpy(np.int32, copy=False)
                    + traindf["max"].to_numpy(np.int32, copy=False)
                )
                // 4
            ).astype(np.int32)
            self._y = traindf[TARGETS].to_numpy(np.float32, copy=False)
        else:
            self._r = None
            self._y = None

        self._patch_cache = (
            patch_cache if patch_cache is not None else SpecPatchCache(self.specs)
        )

        self._data_cache: List[torch.Tensor] = [None] * len(self.traindf)
        if self.mode == "test":
            for i in tqdm(range(len(self.traindf)), desc="Precomputing test tensors"):
                self._data_cache[i] = self._build_x3(i)
        else:
            for i in tqdm(
                range(len(self.traindf)), desc=f"Precomputing {mode} tensors"
            ):
                self._data_cache[i] = self._build_x3(i)

    def _build_x3(self, idx: int) -> torch.Tensor:
        X = np.zeros((128, 256, 8), dtype=np.float32)
        if self.mode == "test":
            r = 0
        else:
            r = int(self._r[idx])

        eeg_id = int(self._eeg_id[idx])
        spec_id = int(self._spec_id[idx])

        img_eeg = self.eeg.get(eeg_id, None)
        if img_eeg is None:
            img_eeg = np.zeros((128, 256, 4), dtype=np.float32)

        patches = self._patch_cache.get_patches(spec_id, r)  # (4,100,256)

        X[14:-14, :, 0:4] = patches.transpose(1, 2, 0)  # (100,256,4)
        X[:, :, 4:8] = img_eeg

        spec = np.transpose(X[:, :, 0:4], (2, 0, 1)).reshape(4 * 128, 256)  # (512,256)
        eeg = np.transpose(X[:, :, 4:8], (2, 0, 1)).reshape(4 * 128, 256)  # (512,256)
        x2 = np.concatenate([spec, eeg], axis=1)  # (512,512)
        x3 = np.stack([x2, x2, x2], axis=0).astype(
            np.float32, copy=False
        )  # (3,512,512)
        return torch.from_numpy(x3)

    def __len__(self):
        return len(self.traindf)

    def __getitem__(self, idx):
        x3 = self._data_cache[idx]
        if self.mode != "test":
            y = self._y[idx]
        else:
            y = np.zeros(6, dtype=np.float32)
        return {"data": x3, "target": y}




## === cell 9
def _seed_worker(worker_id):
    seed = 42 + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


num_workers = 0
pin = torch.cuda.is_available()

test_patch_cache = SpecPatchCache(all_spectrograms)
customdataset = CustomDataset(
    test_df,
    config,
    mode="test",
    specs=all_spectrograms,
    eegs=all_eegs,
    patch_cache=test_patch_cache,
)
_ = customdataset[0]

test_loader = DataLoader(
    customdataset,
    batch_size=config.batchsize,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin,
    persistent_workers=False,
)

if not USE_PRETRAINED:
    train_df_full = pd.read_csv(paths.train_csv)

    train_df_full["min"] = (
        train_df_full["spectrogram_label_offset_seconds"] * 2
    ).astype(
        np.int32
    )  # 0.5s steps
    train_df_full["max"] = train_df_full["min"] + 600  # 300 rows window

    y_counts = train_df_full[TARGETS].values.astype(np.float64)
    y_prob = y_counts / np.clip(y_counts.sum(axis=1, keepdims=True), 1.0, None)
    train_df_full[TARGETS] = y_prob.astype(np.float32)

    rng = np.random.RandomState(42)
    unique_patients = train_df_full["patient_id"].unique()
    rng.shuffle(unique_patients)
    n_valid = max(1, int(0.1 * len(unique_patients)))
    valid_patients = set(unique_patients[:n_valid])

    train_df = train_df_full[
        ~train_df_full["patient_id"].isin(valid_patients)
    ].reset_index(drop=True)
    valid_df = train_df_full[
        train_df_full["patient_id"].isin(valid_patients)
    ].reset_index(drop=True)

    max_train_rows = 8000
    max_valid_rows = 1500
    if len(train_df) > max_train_rows:
        train_df = train_df.sample(n=max_train_rows, random_state=42).reset_index(
            drop=True
        )
    if len(valid_df) > max_valid_rows:
        valid_df = valid_df.sample(n=max_valid_rows, random_state=42).reset_index(
            drop=True
        )

    print("Train/valid rows:", len(train_df), len(valid_df))

    needed_train_spec_ids = set(
        train_df["spectrogram_id"].tolist() + valid_df["spectrogram_id"].tolist()
    )
    needed_train_eeg_ids = set(
        train_df["eeg_id"].tolist() + valid_df["eeg_id"].tolist()
    )

    train_spectrograms = {}
    for sid in tqdm(
        sorted(list(needed_train_spec_ids))[:3000],
        desc="Loading train spectrograms (subset)",
    ):
        fp = os.path.join(paths.train_spec_dir, f"{sid}.parquet")
        if os.path.exists(fp):
            sp = pd.read_parquet(fp)
            arr = sp.to_numpy(copy=False)
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32, copy=False)
            train_spectrograms[int(sid)] = arr

    def _train_eeg_worker(eid: int):
        fp = os.path.join(paths.train_eeg_dir, f"{eid}.parquet")
        if not os.path.exists(fp):
            return None
        return int(eid), np.asarray(spectrogram_from_eeg(fp), dtype=np.float32)

    train_eegs = {}
    train_eeg_ids_subset = sorted(list(needed_train_eeg_ids))[:3000]
    with _MP_CTX.Pool(processes=max(1, min(multiprocessing.cpu_count(), 8))) as pool:
        for out in tqdm(
            pool.imap_unordered(
                _train_eeg_worker, train_eeg_ids_subset, chunksize=chunksize
            ),
            total=len(train_eeg_ids_subset),
            desc="Loading train EEGs (subset, parallel)",
        ):
            if out is None:
                continue
            eid, arr = out
            train_eegs[eid] = arr

    print(
        "Loaded train specs:",
        len(train_spectrograms),
        "Loaded train eeg specs:",
        len(train_eegs),
    )

    train_patch_cache = SpecPatchCache(train_spectrograms)
    train_dataset = CustomDataset(
        train_df,
        config,
        mode="train",
        specs=train_spectrograms,
        eegs=train_eegs,
        patch_cache=train_patch_cache,
    )
    valid_dataset = CustomDataset(
        valid_df,
        config,
        mode="train",
        specs=train_spectrograms,
        eegs=train_eegs,
        patch_cache=train_patch_cache,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batchsize,
        shuffle=True,
        num_workers=0,
        pin_memory=pin,
        persistent_workers=False,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batchsize,
        shuffle=False,
        num_workers=0,
        pin_memory=pin,
        persistent_workers=False,
    )




## === cell 10
class Custommodel(nn.Module):
    def __init__(self, config, numclass: int = 6):
        super(Custommodel, self).__init__()
        self.model = timm.create_model(
            config.model,
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




## === cell 11
def inference_function(test_loader, model, device):
    model.eval()
    softmax = nn.Softmax(dim=1)
    preds = []
    for batch in tqdm(test_loader, desc="Inference"):
        x = batch["data"].to(device, non_blocking=True)
        with torch.no_grad():
            ypred = model(x)
            ypred = softmax(ypred)
        preds.append(ypred.detach().cpu().numpy())
    return {"predictions": np.concatenate(preds, axis=0)}


def train_one_epoch(train_loader, model, optimizer, device, scaler=None):
    model.train()
    for batch in tqdm(train_loader, desc="Train", leave=False):
        x = batch["data"].to(device, non_blocking=True)
        y = batch["target"]
        if not torch.is_tensor(y):
            y = torch.as_tensor(y, dtype=torch.float32, device=device)
        else:
            y = y.to(device=device, dtype=torch.float32, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
                log_probs = torch.log_softmax(logits, dim=1)
                loss = -(y * log_probs).sum(dim=1).mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            log_probs = torch.log_softmax(logits, dim=1)
            loss = -(y * log_probs).sum(dim=1).mean()
            loss.backward()
            optimizer.step()


def valid_kld(valid_loader, model, device):
    model.eval()
    losses = []
    for batch in tqdm(valid_loader, desc="Valid", leave=False):
        x = batch["data"].to(device, non_blocking=True)
        y = batch["target"]
        if not torch.is_tensor(y):
            y = torch.as_tensor(y, dtype=torch.float32, device=device)
        else:
            y = y.to(device=device, dtype=torch.float32, non_blocking=True)

        with torch.no_grad():
            logits = model(x)
            log_probs = torch.log_softmax(logits, dim=1)
            loss = -(y * log_probs).sum(dim=1).mean()
        losses.append(loss.detach().cpu().item())
    return float(np.mean(losses)) if len(losses) else float("nan")




## === cell 12
predictions = None

if USE_PRETRAINED:
    preds_list = []
    for fn in sorted(os.listdir(weights_dir)):
        dd = torch.load(os.path.join(weights_dir, fn), map_location="cpu")
        model = Custommodel(config)
        model.load_state_dict(dd["model"], strict=True)
        model.to(device)
        with torch.inference_mode():
            pred_dict = inference_function(test_loader, model, device)
        preds_list.append(pred_dict["predictions"])
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    preds_arr = np.array(preds_list)
    if preds_arr.shape[0] == 5:
        w = np.array([2, 4, 5, 3, 1], dtype=np.float32)
        predictions = np.average(preds_arr, weights=w, axis=0)
    else:
        predictions = preds_arr.mean(axis=0)
else:
    model = Custommodel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.WEIGHT_DECAY
    )
    scaler = torch.amp.GradScaler(enabled=(torch.cuda.is_available() and config.AMP))

    for ep in range(config.epoch):
        train_one_epoch(train_loader, model, optimizer, device, scaler=scaler)
        vkld = valid_kld(valid_loader, model, device)
        print(f"Epoch {ep+1}/{config.epoch} valid_KL_like: {vkld:.5f}")

    with torch.inference_mode():
        pred_dict = inference_function(test_loader, model, device)
    predictions = pred_dict["predictions"]

predictions = np.asarray(predictions, dtype=np.float64)
predictions = np.clip(predictions, 1e-8, 1.0)
predictions = predictions / predictions.sum(axis=1, keepdims=True)
assert predictions.shape == (len(test_df), 6)



## === cell 13
sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
sub[TARGETS] = predictions.astype(np.float32)
sub.to_csv("submission.csv", index=False)
print(f"Submission shape: {sub.shape}")
print(sub.head())
print(
    "Row-sum check (min/max):",
    sub[TARGETS].sum(axis=1).min(),
    sub[TARGETS].sum(axis=1).max(),
)
