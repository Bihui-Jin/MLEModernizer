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

# 5. Target score

0.516782941336992

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'I fix the pipeline so it runs end-to-end and always writes a valid `submission.csv` with the required columns and per-row probabilities summing to 1. The main runtime blocker is the missing checkpoint directory `/kaggle/input/resnet34newcode`, so I add a safe fallback to output a valid baseline distribution (using train label priors) when no weights are found, instead of crashing. I also fix dataset bugs that would break inference or slow it severely (undefined `targets`, excessive `print`, and incorrect spectrogram windowing using nonexistent `min/max` columns). Finally, I ensure the prediction array shape matches `(n_test, 6)` and is normalized row-wise before saving.'
- What this solution (achieved 1.40012) has done: 'Your current score is far above the target (lower is better), and it looks like you’re still in the “no checkpoints found → train-prior baseline” path, which is a very weak predictor for this metric. To move the score down toward the target without changing the core model/inference semantics, I (1) fix a correctness bug in `CustomDataset.__getitem__` where it can still index `self.specs[row.spectrogram_id]` even when missing (causing silent failure risks and inconsistent behavior), (2) add a safe fallback to use the provided `sample_submission.csv` distribution when checkpoints are missing (this is usually closer to the competition label distribution than a raw train-row mean), and (3) blend (convex-combine) the train prior with the sample-submission prior to reduce overconfident mismatch while keeping valid probabilities. These are minimal, low-risk changes that should improve KL divergence versus the current prior-only baseline.'

# 9. Code solution

## === cell 0
import os
import gc
import math
import random
import time
import multiprocessing
from glob import glob
from typing import Dict, List

import numpy as np
import pandas as pd

import albumentations as A
import librosa
import matplotlib.pyplot as plt
import pywt
import timm
import torch
import torch.nn as nn

from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, get_worker_info
from tqdm import tqdm

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using", torch.cuda.device_count(), "GPU(s)")




## === cell 1
class config:
    MODEL = "resnet34d"
    model2 = "tiny_vit_21m_512"
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




## === cell 2
USE_WAVELET = None

NAMES = ["LL", "LP", "RP", "RR"]

FEATS = [
    ["Fp1", "F7", "T3", "T5", "O1"],
    ["Fp1", "F3", "C3", "P3", "O1"],
    ["Fp2", "F8", "T4", "T6", "O2"],
    ["Fp2", "F4", "C4", "P4", "O2"],
]

TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
targets = TARGETS  # dataset uses lowercase `targets`


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


_SR = 200
_NFFT = 1024
_NMELS = 128
_FMIN = 0
_FMAX = 20
_WIN_LENGTH = 128
_HOP_LENGTH = 10000 // 256  # 39, identical to original

_MEL_BASIS = librosa.filters.mel(
    sr=_SR, n_fft=_NFFT, n_mels=_NMELS, fmin=_FMIN, fmax=_FMAX, htk=False, norm="slaney"
).astype(np.float32)
_WINDOW = librosa.filters.get_window("hann", _WIN_LENGTH, fftbins=True).astype(
    np.float32
)

_TORCH_MEL = torch.from_numpy(_MEL_BASIS)  # (128, 513)
_TORCH_WIN = torch.from_numpy(_WINDOW)  # (128,)


def _mel_spectrogram_power_torch(
    y: np.ndarray, torch_device: torch.device
) -> np.ndarray:
    yt = torch.as_tensor(y, dtype=torch.float32, device=torch_device)
    win = _TORCH_WIN.to(torch_device)
    mel = _TORCH_MEL.to(torch_device)
    D = torch.stft(
        yt,
        n_fft=_NFFT,
        hop_length=_HOP_LENGTH,
        win_length=_WIN_LENGTH,
        window=win,
        center=True,
        pad_mode="constant",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    S = D.abs() ** 2  # power=2
    M = mel @ S  # (128, T)
    return M.detach().cpu().numpy().astype(np.float32, copy=False)


def spectrogram_from_eeg(parquet_path, display=False):
    use_cols = sorted({c for cols in FEATS for c in cols})
    eeg = pd.read_parquet(parquet_path, columns=use_cols)

    middle = (len(eeg) - 10_000) // 2
    eeg = eeg.iloc[middle : middle + 10_000]

    img = np.zeros((128, 256, 4), dtype="float32")

    if display:
        plt.figure(figsize=(10, 7))
    signals = []
    stft_device = device if torch.cuda.is_available() else torch.device("cpu")

    for k in range(4):
        COLS = FEATS[k]
        cols_arr = eeg[COLS].to_numpy(dtype=np.float32, copy=False)  # shape (10000,5)

        for kk in range(4):
            x = cols_arr[:, kk] - cols_arr[:, kk + 1]

            m = np.nanmean(x)
            if np.isnan(x).mean() < 1:
                x = np.nan_to_num(x, nan=m)
            else:
                x = x.copy()
                x[:] = 0

            if USE_WAVELET:
                x = denoise(x, wavelet=USE_WAVELET)
            signals.append(x)

            mel_spec = _mel_spectrogram_power_torch(x, stft_device)

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
        plt.figure(figsize=(10, 5))
        offset = 0
        for k in range(4):
            if k > 0:
                offset -= signals[3 - k].min()
            plt.plot(range(10_000), signals[k] + offset, label=NAMES[3 - k])
            offset += signals[3 - k].max()
        plt.legend()
        plt.title("EEG Signals")
        plt.show()
        print()
        print("#" * 25)
        print()

    return img




## === cell 3
test_df = pd.read_csv(paths.test_csv)
test_df.head()



## === cell 4
test_eeg_paths = {
    int(fn[:-8]): os.path.join(paths.test_eeg, fn)
    for fn in os.listdir(paths.test_eeg)
    if fn.endswith(".parquet")
}
print("Test EEG paths:", len(test_eeg_paths))

all_eegs = {}



## === cell 5
print(f"Test dataframe shape is: {test_df.shape}")
test_df.head()



## === cell 6
test_spec_paths = {
    int(fn[:-8]): os.path.join(paths.test_spec, fn)
    for fn in os.listdir(paths.test_spec)
    if fn.endswith(".parquet")
}
print("Spectrogram paths:", len(test_spec_paths))



## === cell 7
from collections import OrderedDict

_EEG_IMG_CACHE_DIR = os.path.join(paths.out, "eeg_img_cache_npz")
os.makedirs(_EEG_IMG_CACHE_DIR, exist_ok=True)


def _eeg_cache_file(eeg_id: int) -> str:
    return os.path.join(_EEG_IMG_CACHE_DIR, f"{int(eeg_id)}.npz")


class CustomDataset:
    def __init__(
        self,
        traindf: pd.DataFrame,
        config,
        mode: str = "train",
        specs=None,
        eegs=None,
        spec_paths=None,
        eeg_paths=None,
        eeg_cache_size: int = 256,
        spec_cache_size: int = 256,
        disk_cache_eeg: bool = True,
    ):
        self.traindf = traindf
        self.specs = specs  # optional eager dict cache
        self.spec_paths = spec_paths  # optional lazy path dict
        self.eeg = {} if eegs is None else eegs  # optional eager dict cache
        self.eeg_paths = eeg_paths  # optional lazy path dict
        self.mode = mode

        self._spec_cache = OrderedDict()
        self._eeg_cache = OrderedDict()
        self._spec_cache_size = int(spec_cache_size)
        self._eeg_cache_size = int(eeg_cache_size)
        self._disk_cache_eeg = bool(disk_cache_eeg)

    def __len__(self):
        return len(self.traindf)

    def _lru_get(self, cache: OrderedDict, key):
        v = cache.get(key, None)
        if v is not None:
            cache.move_to_end(key)
        return v

    def _lru_put(self, cache: OrderedDict, key, value, max_size: int):
        cache[key] = value
        cache.move_to_end(key)
        if len(cache) > max_size:
            cache.popitem(last=False)

    def _get_spec(self, spectrogram_id: int):
        if self.specs is not None:
            return self.specs.get(spectrogram_id, None)
        if self.spec_paths is None:
            return None
        sp = self._lru_get(self._spec_cache, spectrogram_id)
        if sp is not None:
            return sp
        fp = self.spec_paths.get(spectrogram_id, None)
        if fp is None:
            return None
        sp = pd.read_parquet(fp).to_numpy(copy=False)
        self._lru_put(self._spec_cache, spectrogram_id, sp, self._spec_cache_size)
        return sp

    def _get_eeg_img(self, eeg_id: int):
        if self.eeg is not None:
            v = self.eeg.get(eeg_id, None)
            if v is not None:
                return v

        if self.eeg_paths is None:
            return None

        v = self._lru_get(self._eeg_cache, eeg_id)
        if v is not None:
            return v

        if self._disk_cache_eeg:
            cf = _eeg_cache_file(eeg_id)
            if os.path.exists(cf):
                try:
                    v = np.load(cf)["img"].astype(np.float32, copy=False)
                    self._lru_put(self._eeg_cache, eeg_id, v, self._eeg_cache_size)
                    return v
                except Exception:
                    pass  # fall back to recompute if cache is corrupted

        fp = self.eeg_paths.get(eeg_id, None)
        if fp is None:
            return None

        v = np.array(spectrogram_from_eeg(fp), dtype=np.float32)

        if self._disk_cache_eeg:
            cf = _eeg_cache_file(eeg_id)
            tmp = cf + f".tmp.{os.getpid()}"
            try:
                np.savez_compressed(tmp, img=v)
                os.replace(tmp, cf)
            except Exception:
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except Exception:
                    pass

        self._lru_put(self._eeg_cache, eeg_id, v, self._eeg_cache_size)
        return v

    def __getitem__(self, idx):
        X = np.zeros((128, 256, 8), dtype="float32")
        row = self.traindf.iloc[idx]

        spec = self._get_spec(int(row.spectrogram_id))
        r = 0
        if spec is not None:
            if spec.shape[0] >= 300:
                r = max(0, (spec.shape[0] - 300) // 2)
            else:
                r = 0

            for region in range(4):
                img = spec[
                    r : r + 300, region * 100 : (region + 1) * 100
                ].T  # (100,300)
                img = np.clip(img, np.exp(-4), np.exp(8))
                img = np.log(img)

                ep = 1e-6
                mu = np.nanmean(img)
                std = np.nanstd(img)
                img = (img - mu) / (std + ep)
                img = np.nan_to_num(img, nan=0.0)

                X[14:-14, :, region] = img[:, 22:-22] / 2.0

        eeg_img = self._get_eeg_img(int(row.eeg_id))
        if eeg_img is None:
            eeg_img = np.zeros((128, 256, 4), dtype="float32")
        X[:, :, 4:] = eeg_img

        X_t = torch.from_numpy(X)  # float32, (128,256,8)
        spec_t = X_t[:, :, 0:4].permute(2, 0, 1).reshape(4 * 128, 256)  # (512,256)
        eeg_t = X_t[:, :, 4:8].permute(2, 0, 1).reshape(4 * 128, 256)  # (512,256)
        x = torch.cat([spec_t, eeg_t], dim=1).unsqueeze(0)  # (1,512,512)
        x = x.expand(3, -1, -1).contiguous()  # exact same values as repeat(3,1,1)

        if self.mode != "test":
            y = row[targets].values.astype(np.float32)
        else:
            y = np.zeros(6, dtype="float32")

        return {"data": x, "target": y}




## === cell 8
customdataset = CustomDataset(
    test_df,
    config,
    mode="test",
    specs=None,
    eegs=all_eegs,
    spec_paths=test_spec_paths,
    eeg_paths=test_eeg_paths,
    eeg_cache_size=512,
    spec_cache_size=512,
    disk_cache_eeg=True,
)
sample = customdataset[0]
print(sample["data"].shape, sample["target"].shape)



## === cell 9
_loader_workers = min(8, os.cpu_count() or 2)
test_loader = DataLoader(
    customdataset,
    batch_size=config.batchsize,
    shuffle=False,
    num_workers=_loader_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(_loader_workers > 0),
    prefetch_factor=4 if _loader_workers > 0 else None,
)




## === cell 10
class CustomModel(nn.Module):
    def __init__(self, config, num_classes: int = 6, pretrained: bool = True):
        super(CustomModel, self).__init__()
        self.USE_KAGGLE_SPECTROGRAMS = True
        self.USE_EEG_SPECTROGRAMS = True
        self.model = timm.create_model(
            config.MODEL,
            pretrained=False,
            drop_rate=0.1,
            drop_path_rate=0.2,
        )
        self.features = nn.Sequential(*list(self.model.children())[:-2])
        self.custom_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.custom_layers(x)
        return x




## === cell 11
def inference_function(test_loader, model, device):
    model.eval()
    softmax = nn.Softmax(dim=1)
    preds = []
    with torch.inference_mode():
        for batch in tqdm(test_loader, desc="Infer"):
            x = batch["data"].to(device, non_blocking=True)
            ypred = model(x)
            ypred = softmax(ypred)
            preds.append(ypred.detach().cpu().numpy())
    return {"predictions": np.concatenate(preds, axis=0)}




## === cell 12
def make_soft_targets(df_: pd.DataFrame) -> np.ndarray:
    votes = df_[TARGETS].to_numpy(dtype=np.float64)
    probs = votes / np.clip(votes.sum(axis=1, keepdims=True), 1e-12, None)
    return probs.astype(np.float32)


def patient_split(df_: pd.DataFrame, val_frac: float = 0.1, seed: int = 42):
    rng = np.random.RandomState(seed)
    patients = df_["patient_id"].unique()
    rng.shuffle(patients)
    n_val = max(1, int(len(patients) * val_frac))
    val_pat = set(patients[:n_val])
    tr_idx = df_.index[~df_["patient_id"].isin(val_pat)].to_numpy()
    va_idx = df_.index[df_["patient_id"].isin(val_pat)].to_numpy()
    return tr_idx, va_idx


def train_one_model(
    train_df: pd.DataFrame,
    specs: dict,
    eegs: dict,
    device,
    epochs: int = 1,
    lr: float = 1e-3,
    batch_size: int = 32,
):
    model = CustomModel(config).to(device)

    criterion = nn.KLDivLoss(reduction="batchmean")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY
    )

    tr_idx, va_idx = patient_split(train_df, val_frac=0.08, seed=42)
    tr_df = train_df.loc[tr_idx].reset_index(drop=True)
    va_df = train_df.loc[va_idx].reset_index(drop=True)

    tr_probs = make_soft_targets(tr_df)
    va_probs = make_soft_targets(va_df)
    for j, c in enumerate(TARGETS):
        tr_df[c] = tr_probs[:, j]
        va_df[c] = va_probs[:, j]

    train_ds = CustomDataset(
        tr_df, config, mode="train", specs=specs, eegs=eegs, disk_cache_eeg=True
    )
    val_ds = CustomDataset(
        va_df, config, mode="train", specs=specs, eegs=eegs, disk_cache_eeg=True
    )

    _w = min(8, os.cpu_count() or 2)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=_w,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(_w > 0),
        prefetch_factor=2 if _w > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=_w,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(_w > 0),
        prefetch_factor=2 if _w > 0 else None,
    )

    for ep in range(epochs):
        model.train()
        tr_loss = 0.0
        n_tr = 0
        for batch in tqdm(train_loader, desc=f"Train epoch {ep+1}/{epochs}"):
            x = batch["data"].to(device, non_blocking=True)
            t = batch["target"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            log_probs = torch.log_softmax(logits, dim=1)
            loss = criterion(log_probs, t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            optimizer.step()

            bs = x.size(0)
            tr_loss += loss.item() * bs
            n_tr += bs

        model.eval()
        va_loss = 0.0
        n_va = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Val epoch {ep+1}/{epochs}"):
                x = batch["data"].to(device, non_blocking=True)
                t = batch["target"].to(device, non_blocking=True)
                logits = model(x)
                log_probs = torch.log_softmax(logits, dim=1)
                loss = criterion(log_probs, t)
                bs = x.size(0)
                va_loss += loss.item() * bs
                n_va += bs

        print(
            f"Epoch {ep+1}: train_KL={tr_loss/max(1,n_tr):.5f} val_KL={va_loss/max(1,n_va):.5f}"
        )

    return model


ckpt_dir = "/kaggle/input/resnet34newcode"
predictions = None

if os.path.isdir(ckpt_dir):
    ckpt_files = [
        f for f in os.listdir(ckpt_dir) if os.path.isfile(os.path.join(ckpt_dir, f))
    ]
else:
    ckpt_files = []

if len(ckpt_files) > 0:
    fold_preds = []
    testdataset = CustomDataset(
        test_df,
        config,
        mode="test",
        specs=None,
        eegs=all_eegs,
        spec_paths=test_spec_paths,
        eeg_paths=test_eeg_paths,
        eeg_cache_size=512,
        spec_cache_size=512,
        disk_cache_eeg=True,
    )
    _w = min(8, os.cpu_count() or 2)
    testloader = DataLoader(
        testdataset,
        batch_size=config.batchsize,
        shuffle=False,
        num_workers=_w,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(_w > 0),
        prefetch_factor=4 if _w > 0 else None,
    )

    for fn in sorted(ckpt_files):
        dd = torch.load(os.path.join(ckpt_dir, fn), map_location="cpu")
        model = CustomModel(config)
        state = dd["model"] if isinstance(dd, dict) and "model" in dd else dd
        model.load_state_dict(state, strict=True)
        model.to(device)

        pred_dict = inference_function(testloader, model, device)
        fold_preds.append(pred_dict["predictions"])
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    predictions = np.mean(np.stack(fold_preds, axis=0), axis=0)
else:
    train_meta = pd.read_csv(paths.train_csv)

    train_eeg_paths = {
        int(fn[:-8]): os.path.join(paths.train_eeg_dir, fn)
        for fn in os.listdir(paths.train_eeg_dir)
        if fn.endswith(".parquet")
    }
    train_spec_paths = {
        int(fn[:-8]): os.path.join(paths.train_spec_dir, fn)
        for fn in os.listdir(paths.train_spec_dir)
        if fn.endswith(".parquet")
    }

    train_eegs_cache = {}
    train_specs_cache = None  # use lazy spec_paths instead of dict

    def train_one_model_lazypaths(
        train_df: pd.DataFrame,
        spec_paths: dict,
        eeg_paths: dict,
        device,
        epochs: int = 1,
        lr: float = 1e-3,
        batch_size: int = 32,
    ):
        model = CustomModel(config).to(device)
        criterion = nn.KLDivLoss(reduction="batchmean")
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY
        )

        tr_idx, va_idx = patient_split(train_df, val_frac=0.08, seed=42)
        tr_df = train_df.loc[tr_idx].reset_index(drop=True)
        va_df = train_df.loc[va_idx].reset_index(drop=True)

        tr_probs = make_soft_targets(tr_df)
        va_probs = make_soft_targets(va_df)
        for j, c in enumerate(TARGETS):
            tr_df[c] = tr_probs[:, j]
            va_df[c] = va_probs[:, j]

        train_ds = CustomDataset(
            tr_df,
            config,
            mode="train",
            specs=None,
            eegs={},
            spec_paths=spec_paths,
            eeg_paths=eeg_paths,
            eeg_cache_size=256,
            spec_cache_size=256,
            disk_cache_eeg=True,
        )
        val_ds = CustomDataset(
            va_df,
            config,
            mode="train",
            specs=None,
            eegs={},
            spec_paths=spec_paths,
            eeg_paths=eeg_paths,
            eeg_cache_size=256,
            spec_cache_size=256,
            disk_cache_eeg=True,
        )

        _w = min(8, os.cpu_count() or 2)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=_w,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(_w > 0),
            prefetch_factor=2 if _w > 0 else None,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=_w,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(_w > 0),
            prefetch_factor=2 if _w > 0 else None,
        )

        for ep in range(epochs):
            model.train()
            tr_loss = 0.0
            n_tr = 0
            for batch in tqdm(train_loader, desc=f"Train epoch {ep+1}/{epochs}"):
                x = batch["data"].to(device, non_blocking=True)
                t = batch["target"].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                log_probs = torch.log_softmax(logits, dim=1)
                loss = criterion(log_probs, t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
                optimizer.step()

                bs = x.size(0)
                tr_loss += loss.item() * bs
                n_tr += bs

            model.eval()
            va_loss = 0.0
            n_va = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Val epoch {ep+1}/{epochs}"):
                    x = batch["data"].to(device, non_blocking=True)
                    t = batch["target"].to(device, non_blocking=True)
                    logits = model(x)
                    log_probs = torch.log_softmax(logits, dim=1)
                    loss = criterion(log_probs, t)
                    bs = x.size(0)
                    va_loss += loss.item() * bs
                    n_va += bs

            print(
                f"Epoch {ep+1}: train_KL={tr_loss/max(1,n_tr):.5f} val_KL={va_loss/max(1,n_va):.5f}"
            )

        return model

    local_epochs = 1
    model = train_one_model_lazypaths(
        train_df=train_meta,
        spec_paths=train_spec_paths,
        eeg_paths=train_eeg_paths,
        device=device,
        epochs=local_epochs,
        lr=config.lr,
        batch_size=config.batchsize,
    )

    testdataset = CustomDataset(
        test_df,
        config,
        mode="test",
        specs=None,
        eegs=all_eegs,
        spec_paths=test_spec_paths,
        eeg_paths=test_eeg_paths,
        eeg_cache_size=512,
        spec_cache_size=512,
        disk_cache_eeg=True,
    )
    _w = min(8, os.cpu_count() or 2)
    testloader = DataLoader(
        testdataset,
        batch_size=config.batchsize,
        shuffle=False,
        num_workers=_w,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(_w > 0),
        prefetch_factor=4 if _w > 0 else None,
    )
    pred_dict = inference_function(testloader, model, device)
    predictions = pred_dict["predictions"]

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("Predictions shape:", predictions.shape)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_54/2757371692.py in <cell line: 0>()
    288 
    289     local_epochs = 1
--> 290     model = train_one_model_lazypaths(
    291         train_df=train_meta,
    292         spec_paths=train_spec_paths,

/tmp/ipykernel_54/2757371692.py in train_one_model_lazypaths(train_df, spec_paths, eeg_paths, device, epochs, lr, batch_size)
    251             tr_loss = 0.0
    252             n_tr = 0
--> 253             for batch in tqdm(train_loader, desc=f"Train epoch {ep+1}/{epochs}"):
    254                 x = batch["data"].to(device, non_blocking=True)
    255                 t = batch["target"].to(device, non_blocking=True)

/usr/local/lib/python3.11/dist-packages/tqdm/std.py in __iter__(self)
   1179 
   1180         try:
-> 1181             for obj in iterable:
   1182                 yield obj
   1183                 # Update and possibly print the progressbar.

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

RuntimeError: Caught RuntimeError in DataLoader worker process 0.
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
  File "/tmp/ipykernel_54/3926276919.py", line 145, in __getitem__
    eeg_img = self._get_eeg_img(int(row.eeg_id))
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_54/3926276919.py", line 100, in _get_eeg_img
    v = np.array(spectrogram_from_eeg(fp), dtype=np.float32)
                 ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_54/3728642458.py", line 124, in spectrogram_from_eeg
    mel_spec = _mel_spectrogram_power_torch(x, stft_device)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_54/3728642458.py", line 65, in _mel_spectrogram_power_torch
    yt = torch.as_tensor(y, dtype=torch.float32, device=torch_device)
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/cuda/__init__.py", line 305, in _lazy_init
    raise RuntimeError(
RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method


## === cell 13
predictions = np.asarray(predictions, dtype=np.float64)
if predictions.ndim != 2 or predictions.shape[1] != 6:
    raise ValueError(f"predictions must be (n_test, 6), got {predictions.shape}")

predictions = np.clip(predictions, 1e-12, None)
predictions = predictions / predictions.sum(axis=1, keepdims=True)

sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
sub[TARGETS] = predictions.astype(np.float32)
sub.to_csv("submission.csv", index=False)

print(f"Submission shape: {sub.shape}")
print(
    "Row sums min/max:", sub[TARGETS].sum(axis=1).min(), sub[TARGETS].sum(axis=1).max()
)
sub.head()



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_54/3481819918.py in <cell line: 0>()
      1 predictions = np.asarray(predictions, dtype=np.float64)
      2 if predictions.ndim != 2 or predictions.shape[1] != 6:
----> 3     raise ValueError(f"predictions must be (n_test, 6), got {predictions.shape}")
      4 
      5 predictions = np.clip(predictions, 1e-12, None)

ValueError: predictions must be (n_test, 6), got ()

## === cell 14
print("Total sum of all probabilities:", float(np.sum(predictions)))
print("First row:", sub.iloc[0].to_dict())

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/745516380.py in <cell line: 0>()
      1 print("Total sum of all probabilities:", float(np.sum(predictions)))
----> 2 print("First row:", sub.iloc[0].to_dict())

NameError: name 'sub' is not defined
