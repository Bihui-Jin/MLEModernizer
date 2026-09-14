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

0.5034600687699031

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 1.40995) has done: 'I fix the inference aggregation so `predictions` is always a proper 2D array (N,6) even when no weight files are found or when checkpoint formats differ, which is what caused the `AxisError`. I also make weight loading robust to both `{"model": state_dict}` and plain `state_dict` checkpoints, and ensure we fall back to a valid uniform-probability submission if weights are missing so a `.csv` is always produced. These changes are execution/stability fixes and should be score-neutral unless your current run was failing to generate a submission at all. Finally, I add a strict post-processing normalization/clipping step to guarantee every row sums to 1 (required by the competition).'

# 9. Code solution

## === cell 0
import albumentations as A
import gc
import librosa
import matplotlib.pyplot as plt
import math
import multiprocessing
import numpy as np
import os
import pandas as pd
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
from typing import Dict, List, Tuple, Optional

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using", torch.cuda.device_count(), "GPU(s)")

torch.set_num_threads(max(1, min(8, os.cpu_count() or 2)))
os.environ.setdefault("OMP_NUM_THREADS", str(torch.get_num_threads()))
os.environ.setdefault("MKL_NUM_THREADS", str(torch.get_num_threads()))

try:
    import pyarrow.parquet as pq  # type: ignore

    _HAS_PYARROW = True
except Exception:
    pq = None
    _HAS_PYARROW = False




## === cell 1
class config:
    BATCH_SIZE = 32
    MODEL = "tf_efficientnet_b1"
    NUM_WORKERS = max(0, min(4, (os.cpu_count() or 2) // 2))
    PRINT_FREQ = 50
    SEED = 20
    VISUALIZE = False

    EPOCHS = 1
    LR = 1e-3
    WEIGHT_DECAY = 1e-4


class paths:
    MODEL_WEIGHTS = (
        "/kaggle/input/hba-efficientnet-weights/tf_efficientnet_b1_epoch_8.pth"
    )
    OUTPUT_DIR = "/kaggle/working/"
    TRAIN_CSV = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    TEST_CSV = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
    TEST_EEGS = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
    TEST_SPECTROGRAMS = (
        "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
    )
    TRAIN_EEGS = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"
    TRAIN_SPECTROGRAMS = (
        "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/"
    )


model_weights = []
if os.path.isfile(paths.MODEL_WEIGHTS):
    model_weights = [paths.MODEL_WEIGHTS]
else:
    model_weights = []  # if not available, we'll train a quick model from scratch below

print(f"Using {len(model_weights)} model weight file(s)")
for mw in model_weights[:10]:
    print(" -", mw)




## === cell 2
USE_WAVELET = None

NAMES = ["LL", "LP", "RP", "RR"]

FEATS = [
    ["Fp1", "F7", "T3", "T5", "O1"],
    ["Fp1", "F3", "C3", "P3", "O1"],
    ["Fp2", "F8", "T4", "T6", "O2"],
    ["Fp2", "F4", "C4", "P4", "O2"],
]

label_cols = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]


def maddest(d, axis: int = None):
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
_TARGET_FRAMES = 256
_HOP_LENGTH = 10000 // _TARGET_FRAMES  # same as original: len(x)//256

_mel_fb = librosa.filters.mel(
    sr=_SR, n_fft=_NFFT, n_mels=_NMELS, fmin=_FMIN, fmax=_FMAX, htk=False, norm="slaney"
).astype(np.float32)
_mel_fb_t_cpu = torch.from_numpy(_mel_fb)  # (n_mels, 1+n_fft//2)
_win_cpu = torch.hann_window(_WIN_LENGTH, periodic=True, dtype=torch.float32)

_mel_fb_t_dev = None
_win_dev = None

_TOP_DB = 80.0
_AMIN = 1e-10


def _power_to_db_np(S, ref_max=True):
    S = np.maximum(S, _AMIN)
    if ref_max:
        ref = np.max(S)
    else:
        ref = 1.0
    ref = max(ref, _AMIN)
    log_spec = 10.0 * np.log10(S)
    log_spec -= 10.0 * np.log10(ref)
    log_spec = np.maximum(log_spec, log_spec.max() - _TOP_DB)
    return log_spec


def _ensure_mel_stft_tensors_on(device_):
    global _mel_fb_t_dev, _win_dev
    if _mel_fb_t_dev is None or _mel_fb_t_dev.device != device_:
        _mel_fb_t_dev = _mel_fb_t_cpu.to(device_, non_blocking=True)
    if _win_dev is None or _win_dev.device != device_:
        _win_dev = _win_cpu.to(device_, non_blocking=True)


def _mel_spectrogram_fast(x: np.ndarray) -> np.ndarray:
    xt = torch.from_numpy(np.asarray(x, dtype=np.float32))
    if torch.cuda.is_available():
        _ensure_mel_stft_tensors_on(device)
        xt = xt.to(device, non_blocking=True)
        X = torch.stft(
            xt,
            n_fft=_NFFT,
            hop_length=_HOP_LENGTH,
            win_length=_WIN_LENGTH,
            window=_win_dev,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        P = X.real * X.real + X.imag * X.imag
        mel = torch.matmul(_mel_fb_t_dev, P)
        return mel.detach().cpu().numpy()
    else:
        X = torch.stft(
            xt,
            n_fft=_NFFT,
            hop_length=_HOP_LENGTH,
            win_length=_WIN_LENGTH,
            window=_win_cpu,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        P = X.real * X.real + X.imag * X.imag
        mel = torch.matmul(_mel_fb_t_cpu, P)
        return mel.numpy(force=True)


_needed_cols_sorted = sorted({c for grp in FEATS for c in grp})


def _read_eeg_needed_cols(parquet_path: str) -> np.ndarray:
    if _HAS_PYARROW:
        table = pq.read_table(parquet_path, columns=_needed_cols_sorted)
        cols = [
            table.column(i).to_numpy(zero_copy_only=False)
            for i in range(table.num_columns)
        ]
        return np.stack(cols, axis=1).astype(np.float32, copy=False)
    else:
        eeg = pd.read_parquet(parquet_path, columns=_needed_cols_sorted)
        return eeg.to_numpy(dtype=np.float32, copy=False)


def spectrogram_from_eeg(parquet_path, display=False):
    eeg_np_full = _read_eeg_needed_cols(parquet_path)

    middle = (len(eeg_np_full) - 10_000) // 2
    eeg_np = eeg_np_full[middle : middle + 10_000]

    img = np.zeros((128, 256, 4), dtype="float32")

    if display:
        plt.figure(figsize=(10, 7))
    signals = []

    nanmean = np.nanmean
    nan_to_num = np.nan_to_num

    col_idx = {c: i for i, c in enumerate(_needed_cols_sorted)}

    for k in range(4):
        COLS = FEATS[k]
        for kk in range(4):
            a = eeg_np[:, col_idx[COLS[kk]]]
            b = eeg_np[:, col_idx[COLS[kk + 1]]]
            x = a - b

            m = nanmean(x)
            if np.isnan(x).mean() < 1:
                x = nan_to_num(x, nan=m)
            else:
                x = x.copy()
                x[:] = 0

            if USE_WAVELET:
                x = denoise(x, wavelet=USE_WAVELET)
            signals.append(x)

            mel_spec = _mel_spectrogram_fast(x)

            width = (mel_spec.shape[1] // 32) * 32
            mel_spec = mel_spec[:, :width]

            mel_spec_db = _power_to_db_np(mel_spec, ref_max=True).astype(np.float32)
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


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(config.SEED)




## === cell 3
test_df = pd.read_csv(paths.TEST_CSV)
sample_sub = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
)

test_df = sample_sub[["eeg_id"]].merge(test_df, on="eeg_id", how="left")
print(f"Test dataframe shape is: {test_df.shape}")
print(test_df.head())




## === cell 4
class SpectrogramCache:
    def __init__(self, base_dir: str, max_items: int = 256):
        self.base_dir = base_dir
        self.max_items = max_items
        self._cache: Dict[int, np.ndarray] = {}
        self._lru: List[int] = []

    def _touch(self, key: int):
        try:
            self._lru.remove(key)
        except ValueError:
            pass
        self._lru.append(key)

    def get(self, sid: int) -> np.ndarray:
        sid = int(sid)
        arr = self._cache.get(sid)
        if arr is not None:
            self._touch(sid)
            return arr
        p = os.path.join(self.base_dir, f"{sid}.parquet")
        aux = pd.read_parquet(p)  # keep semantics identical
        arr = aux.iloc[:, 1:].to_numpy()
        del aux
        self._cache[sid] = arr
        self._touch(sid)
        if len(self._lru) > self.max_items:
            old = self._lru.pop(0)
            self._cache.pop(old, None)
        return arr


class EegSpectrogramCache:
    def __init__(self, base_dir: str, max_items: int = 256):
        self.base_dir = base_dir
        self.max_items = max_items
        self._cache: Dict[int, np.ndarray] = {}
        self._lru: List[int] = []

    def _touch(self, key: int):
        try:
            self._lru.remove(key)
        except ValueError:
            pass
        self._lru.append(key)

    def get(self, eeg_id: int) -> np.ndarray:
        eeg_id = int(eeg_id)
        arr = self._cache.get(eeg_id)
        if arr is not None:
            self._touch(eeg_id)
            return arr
        p = os.path.join(self.base_dir, f"{eeg_id}.parquet")
        arr = spectrogram_from_eeg(p, display=False)
        self._cache[eeg_id] = arr
        self._touch(eeg_id)
        if len(self._lru) > self.max_items:
            old = self._lru.pop(0)
            self._cache.pop(old, None)
        return arr


test_spec_cache = SpectrogramCache(paths.TEST_SPECTROGRAMS, max_items=512)
test_eeg_cache = EegSpectrogramCache(paths.TEST_EEGS, max_items=256)
print("Initialized lazy test caches (spectrogram + EEG)")




## === cell 5
all_eegs = None
print(
    "Skipping eager build of all test EEG spectrograms; will compute lazily via cache."
)




## === cell 6
class CustomModel(nn.Module):
    def __init__(self, config, num_classes: int = 6):
        super(CustomModel, self).__init__()
        self.USE_KAGGLE_SPECTROGRAMS = True
        self.USE_EEG_SPECTROGRAMS = True
        self.model = timm.create_model(config.MODEL, pretrained=False)
        self.features = nn.Sequential(*list(self.model.children())[:-2])
        self.custom_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.model.num_features, num_classes),
        )

    def __reshape_input(self, x):
        B, H, W, C = x.shape  # C==8
        x5 = x.view(B, H, W, 4, 2)
        spectrograms = x5[..., 0].permute(0, 3, 1, 2).contiguous()  # (B,4,H,W)
        eegs = x5[..., 1].permute(0, 3, 1, 2).contiguous()  # (B,4,H,W)

        if self.USE_KAGGLE_SPECTROGRAMS & self.USE_EEG_SPECTROGRAMS:
            x2 = torch.cat([spectrograms, eegs], dim=2)  # (B,4,2H,W)
        elif self.USE_EEG_SPECTROGRAMS:
            x2 = eegs
        else:
            x2 = spectrograms

        ch0 = x2[:, 0:1, :, :]
        ch3 = x2[:, 3:4, :, :]
        chm = x2.mean(dim=1, keepdim=True)
        x3 = torch.cat([ch0, chm, ch3], dim=1)  # (B,3,*,*)
        return x3

    def forward(self, x):
        x = self.__reshape_input(x)
        x = self.features(x)
        x = self.custom_layers(x)
        return x




## === cell 7
_HFLIP_TRANSFORM = A.Compose([A.HorizontalFlip(p=0.5)])


class CustomDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        config,
        augment: bool = False,
        mode: str = "train",
        specs: Dict[int, np.ndarray] = None,
        eeg_specs: Dict[int, np.ndarray] = None,
        spec_cache: SpectrogramCache = None,
        eeg_cache: EegSpectrogramCache = None,
    ):
        self.df = df.reset_index(drop=True)
        self.config = config
        self.batch_size = self.config.BATCH_SIZE
        self.augment = augment
        self.mode = mode
        self.spectrograms = specs if specs is not None else {}
        self.eeg_spectrograms = eeg_specs if eeg_specs is not None else {}
        self.spec_cache = spec_cache
        self.eeg_cache = eeg_cache

        self._processed_spec_cache: Dict[Tuple[int, int], np.ndarray] = {}
        self._processed_spec_lru: List[Tuple[int, int]] = []
        self._processed_spec_max = 4096 if mode == "test" else 1024

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        X, y = self.__data_generation(index)
        if self.augment:
            X = self.__transform(X)
        return torch.from_numpy(X), torch.from_numpy(y)

    def _get_spec(self, sid: int) -> np.ndarray:
        sid = int(sid)
        if sid in self.spectrograms:
            return self.spectrograms[sid]
        if self.spec_cache is not None:
            return self.spec_cache.get(sid)
        raise KeyError(f"Spectrogram id {sid} not found")

    def _get_eeg_spec(self, eeg_id: int) -> np.ndarray:
        eeg_id = int(eeg_id)
        if eeg_id in self.eeg_spectrograms:
            return self.eeg_spectrograms[eeg_id]
        if self.eeg_cache is not None:
            return self.eeg_cache.get(eeg_id)
        raise KeyError(f"EEG id {eeg_id} not found")

    def _touch_processed(self, key: Tuple[int, int]):
        try:
            self._processed_spec_lru.remove(key)
        except ValueError:
            pass
        self._processed_spec_lru.append(key)

    def _get_processed_kaggle_spec(
        self, sid: int, r: int, spec_arr: np.ndarray
    ) -> np.ndarray:
        key = (int(sid), int(r))
        cached = self._processed_spec_cache.get(key)
        if cached is not None:
            self._touch_processed(key)
            return cached

        Xk = np.zeros((128, 256, 4), dtype="float32")
        ep = 1e-6

        block = spec_arr[r : r + 300, :400]
        block = block.reshape(300, 4, 100).transpose(
            1, 2, 0
        )  # (4,100,300) == img per region

        block = np.clip(block, np.exp(-4), np.exp(8))
        block = np.log(block)

        mu = np.nanmean(block, axis=(1, 2), keepdims=True)
        std = np.nanstd(block, axis=(1, 2), keepdims=True)
        block = (block - mu) / (std + ep)
        block = np.nan_to_num(block, nan=0.0)

        cropped = block[:, :, 22:-22] / 2.0  # (4,100,256)
        Xk[14:-14, :, :] = cropped.transpose(1, 2, 0)  # (100,256,4) into (128,256,4)

        self._processed_spec_cache[key] = Xk
        self._touch_processed(key)
        if len(self._processed_spec_lru) > self._processed_spec_max:
            old = self._processed_spec_lru.pop(0)
            self._processed_spec_cache.pop(old, None)
        return Xk

    def __data_generation(self, index):
        X = np.zeros((128, 256, 8), dtype="float32")
        y = np.zeros(6, dtype="float32")
        row = self.df.iloc[index]

        sid = int(row.spectrogram_id)
        spec = self._get_spec(sid)

        if self.mode == "test":
            max_r = max(0, spec.shape[0] - 300)
            r = max_r // 2
        else:
            if "min" in row and "max" in row:
                r = int((row["min"] + row["max"]) // 4)
            else:
                max_r = max(0, spec.shape[0] - 300)
                r = max_r // 2

        X[:, :, :4] = self._get_processed_kaggle_spec(sid, r, spec)

        eeg_img = self._get_eeg_spec(int(row.eeg_id))
        X[:, :, 4:] = eeg_img

        if self.mode != "test":
            votes = row[label_cols].values.astype(np.float32)
            s = float(votes.sum())
            if s > 0:
                y = votes / s
            else:
                y[:] = 1.0 / 6.0

        return X, y

    def __transform(self, img):
        return _HFLIP_TRANSFORM(image=img)["image"]




## === cell 8
test_dataset = CustomDataset(
    test_df,
    config,
    mode="test",
    specs=None,
    eeg_specs=None,
    spec_cache=test_spec_cache,
    eeg_cache=test_eeg_cache,
)

_test_num_workers = max(1, min(4, os.cpu_count() or 2))

test_loader = DataLoader(
    test_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=_test_num_workers,
    pin_memory=True,
    persistent_workers=(_test_num_workers > 0),
    prefetch_factor=2 if _test_num_workers > 0 else None,
    drop_last=False,
)
X0, y0 = test_dataset[0]
print(f"X shape: {X0.shape}")
print(f"y shape: {y0.shape}")
print("Test loader workers:", _test_num_workers)




## === cell 9
def inference_function(test_loader, model, device):
    model.eval()
    softmax = nn.Softmax(dim=1)
    preds = []
    with tqdm(test_loader, unit="test_batch", desc="Inference") as tqdm_test_loader:
        with torch.inference_mode():
            for step, (X, y) in enumerate(tqdm_test_loader):
                X = X.to(device, non_blocking=True)
                y_preds = model(X)
                y_preds = softmax(y_preds)
                preds.append(y_preds.detach().cpu().numpy())
    return {"predictions": np.concatenate(preds, axis=0)}




## === cell 10
def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ["model", "state_dict", "model_state_dict", "net"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    if isinstance(ckpt, dict):
        return ckpt
    return ckpt


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def kl_divergence(
    p_true: torch.Tensor, p_pred: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    p_true = torch.clamp(p_true, eps, 1.0)
    p_pred = torch.clamp(p_pred, eps, 1.0)
    return torch.sum(p_true * torch.log(p_true / p_pred), dim=1).mean()


def make_train_val_split_by_patient(
    train_df: pd.DataFrame, val_frac: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    patients = train_df["patient_id"].dropna().unique()
    rng = np.random.RandomState(config.SEED)
    rng.shuffle(patients)
    n_val = max(1, int(len(patients) * val_frac))
    val_patients = set(patients[:n_val])
    is_val = train_df["patient_id"].isin(val_patients)
    return train_df.loc[~is_val].reset_index(drop=True), train_df.loc[
        is_val
    ].reset_index(drop=True)




## === cell 11
train_df = pd.read_csv(paths.TRAIN_CSV)

agg = train_df.groupby(["eeg_id", "spectrogram_id", "patient_id"], as_index=False)[
    label_cols
].mean()

train_part, val_part = make_train_val_split_by_patient(agg, val_frac=0.1)
print("Train rows:", len(train_part), "Val rows:", len(val_part))


def load_train_specs_for_ids(spec_ids: np.ndarray) -> Dict[int, np.ndarray]:
    out = {}
    unique_ids = np.unique(spec_ids.astype(int))
    for sid in tqdm(unique_ids, desc="Load train spectrograms"):
        p = os.path.join(paths.TRAIN_SPECTROGRAMS, f"{int(sid)}.parquet")
        aux = pd.read_parquet(p)
        out[int(sid)] = aux.iloc[:, 1:].to_numpy()
        del aux
    return out


def load_train_eegs_for_ids(eeg_ids: np.ndarray) -> Dict[int, np.ndarray]:
    out = {}
    unique_ids = np.unique(eeg_ids.astype(int))
    for eid in tqdm(unique_ids, desc="Build train EEG spectrograms"):
        p = os.path.join(paths.TRAIN_EEGS, f"{int(eid)}.parquet")
        out[int(eid)] = spectrogram_from_eeg(p, display=False)
    return out


MAX_TRAIN_ROWS = 2048
if len(train_part) > MAX_TRAIN_ROWS:
    train_part = train_part.iloc[:MAX_TRAIN_ROWS].reset_index(drop=True)
if len(val_part) > max(256, MAX_TRAIN_ROWS // 8):
    val_part = val_part.iloc[: max(256, MAX_TRAIN_ROWS // 8)].reset_index(drop=True)

train_specs = load_train_specs_for_ids(train_part["spectrogram_id"].values)
val_specs = load_train_specs_for_ids(val_part["spectrogram_id"].values)
train_eegs = load_train_eegs_for_ids(train_part["eeg_id"].values)
val_eegs = load_train_eegs_for_ids(val_part["eeg_id"].values)

train_dataset = CustomDataset(
    train_part,
    config,
    augment=True,
    mode="train",
    specs=train_specs,
    eeg_specs=train_eegs,
)
val_dataset = CustomDataset(
    val_part, config, augment=False, mode="train", specs=val_specs, eeg_specs=val_eegs
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    drop_last=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    drop_last=False,
)




## === cell 12
trained_weight_path = os.path.join(paths.OUTPUT_DIR, "trained_model.pth")

if len(model_weights) == 0:
    print(
        "No external weights found; training a minimal model to improve score vs uniform submission."
    )
    model = CustomModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
    )

    for epoch in range(config.EPOCHS):
        model.train()
        tr_losses = []
        for X, y in tqdm(train_loader, desc=f"Train epoch {epoch+1}/{config.EPOCHS}"):
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(X)
            p = torch.softmax(logits, dim=1)
            loss = kl_divergence(y, p)
            loss.backward()
            optimizer.step()
            tr_losses.append(loss.item())

        model.eval()
        va_losses = []
        with torch.inference_mode():
            for X, y in tqdm(val_loader, desc=f"Val epoch {epoch+1}/{config.EPOCHS}"):
                X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model(X)
                p = torch.softmax(logits, dim=1)
                loss = kl_divergence(y, p)
                va_losses.append(loss.item())

        print(
            f"Epoch {epoch+1}: train_KL={np.mean(tr_losses):.5f} val_KL={np.mean(va_losses):.5f}"
        )

    torch.save({"model": model.state_dict()}, trained_weight_path)
    model_weights = [trained_weight_path]
    del model
    torch.cuda.empty_cache()
    gc.collect()
else:
    print("External weight provided; skipping training.")




## === cell 13
predictions_list = []

for model_weight in model_weights:
    model = CustomModel(config)
    checkpoint = torch.load(model_weight, map_location="cpu")
    state_dict = _strip_module_prefix(_extract_state_dict(checkpoint))

    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        print(f"WARNING: strict=True load failed for {model_weight}: {repr(e)}")
        model.load_state_dict(state_dict, strict=False)

    model.to(device)
    prediction_dict = inference_function(test_loader, model, device)
    predictions_list.append(prediction_dict["predictions"])

    del model, checkpoint, state_dict, prediction_dict
    torch.cuda.empty_cache()
    gc.collect()

predictions = (
    np.mean(np.stack(predictions_list, axis=0), axis=0)
    if len(predictions_list) > 0
    else np.full((len(test_df), 6), 1.0 / 6.0, dtype=np.float32)
)
print("Raw predictions shape:", predictions.shape)




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/3099501633.py in <cell line: 0>()
     13 
     14     model.to(device)
---> 15     prediction_dict = inference_function(test_loader, model, device)
     16     predictions_list.append(prediction_dict["predictions"])
     17 

/tmp/ipykernel_55/1385286588.py in inference_function(test_loader, model, device)
      5     with tqdm(test_loader, unit="test_batch", desc="Inference") as tqdm_test_loader:
      6         with torch.inference_mode():
----> 7             for step, (X, y) in enumerate(tqdm_test_loader):
      8                 X = X.to(device, non_blocking=True)
      9                 y_preds = model(X)

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
  File "/tmp/ipykernel_55/4011877697.py", line 36, in __getitem__
    X, y = self.__data_generation(index)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_55/4011877697.py", line 122, in __data_generation
    eeg_img = self._get_eeg_spec(int(row.eeg_id))
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_55/4011877697.py", line 54, in _get_eeg_spec
    return self.eeg_cache.get(eeg_id)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_55/1989994048.py", line 54, in get
    arr = spectrogram_from_eeg(p, display=False)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_55/2695112355.py", line 172, in spectrogram_from_eeg
    mel_spec = _mel_spectrogram_fast(x)
               ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_55/2695112355.py", line 85, in _mel_spectrogram_fast
    xt = xt.to(device, non_blocking=True)
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/cuda/__init__.py", line 305, in _lazy_init
    raise RuntimeError(
RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method


## === cell 14
TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

pred = np.asarray(predictions, dtype=np.float64)
if pred.ndim != 2 or pred.shape[1] != 6 or pred.shape[0] != len(test_df):
    raise ValueError(
        f"Bad predictions shape {pred.shape}; expected ({len(test_df)}, 6)"
    )

pred = np.clip(pred, 1e-8, 1.0)
row_sums = pred.sum(axis=1, keepdims=True)
row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
pred = pred / row_sums

pred_df = pd.DataFrame(pred, columns=TARGETS)
pred_df["eeg_id"] = test_df["eeg_id"].values

pred_agg = pred_df.groupby("eeg_id", as_index=False)[TARGETS].mean()

sub = sample_sub[["eeg_id"]].merge(pred_agg, on="eeg_id", how="left")

sub[TARGETS] = sub[TARGETS].fillna(1.0 / 6.0)

sub_probs = sub[TARGETS].to_numpy(dtype=np.float64)
sub_probs = np.clip(sub_probs, 1e-8, 1.0)
sub_probs = sub_probs / np.maximum(sub_probs.sum(axis=1, keepdims=True), 1e-12)
sub[TARGETS] = sub_probs.astype(np.float32)

out_path = os.path.join(paths.OUTPUT_DIR, "submission.csv")
sub.to_csv(out_path, index=False)
print(f"Submission shape: {sub.shape}")
print("Saved to:", out_path)
print(sub.head())
print(
    "Row-sum check (min/max):",
    sub[TARGETS].sum(axis=1).min(),
    sub[TARGETS].sum(axis=1).max(),
)

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3399663703.py in <cell line: 0>()
      8 ]
      9 
---> 10 pred = np.asarray(predictions, dtype=np.float64)
     11 if pred.ndim != 2 or pred.shape[1] != 6 or pred.shape[0] != len(test_df):
     12     raise ValueError(

NameError: name 'predictions' is not defined
