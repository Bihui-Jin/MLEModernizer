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
import os
import gc
import math
import time
import random
import multiprocessing
from glob import glob
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict

import numpy as np
import pandas as pd

import albumentations as A
from albumentations.pytorch import ToTensorV2

import librosa  # kept imported to preserve environment parity; no longer used in hot path
import matplotlib.pyplot as plt

import pywt
import timm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using", torch.cuda.device_count(), "GPU(s)")


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

torch.set_num_threads(1)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")




## === cell 1
class config:
    model = "resnet50d"
    epoch = 2  # keep identical to provided script
    lr = 1e-3
    batchsize = 16
    splits = 5
    momentum = 0.9
    MAX_GRAD_NORM = 1e7
    WEIGHT_DECAY = 0.01
    device = "cuda" if torch.cuda.is_available() else "cpu"
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

_EEG_COLS_NEEDED = sorted({c for group in FEATS for c in group})

_SR = 200
_N_FFT = 1024
_N_MELS = 128
_WIN_LENGTH = 128
_FMIN = 0.0
_FMAX = 20.0

_WINDOW_CPU = torch.hann_window(_WIN_LENGTH, periodic=True, dtype=torch.float32)

_MEL_FB_CPU = torch.from_numpy(
    librosa.filters.mel(
        sr=_SR,
        n_fft=_N_FFT,
        n_mels=_N_MELS,
        fmin=_FMIN,
        fmax=_FMAX,
        htk=True,
        norm="slaney",
    ).astype(np.float32)
)  # (n_mels, 1+n_fft//2)

_MEL_FB_GPU = None
_WINDOW_GPU = None
if torch.cuda.is_available():
    _MEL_FB_GPU = _MEL_FB_CPU.to("cuda", non_blocking=True)
    _WINDOW_GPU = _WINDOW_CPU.to("cuda", non_blocking=True)

_EPS = 1e-10


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


def _torch_mel_db_norm(x_np: np.ndarray, hop_length: int) -> np.ndarray:
    """
    Compute mel spectrogram (power) and convert to dB using ref=max, matching librosa.power_to_db,
    then apply the same scaling: (db + 40) / 40.

    Returns float32 array shape (128, T).
    """
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        x = torch.from_numpy(x_np.astype(np.float32, copy=False)).to(
            "cuda", non_blocking=True
        )
        window = _WINDOW_GPU
        mel_fb = _MEL_FB_GPU
    else:
        x = torch.from_numpy(x_np.astype(np.float32, copy=False))
        window = _WINDOW_CPU
        mel_fb = _MEL_FB_CPU

    spec = torch.stft(
        x,
        n_fft=_N_FFT,
        hop_length=hop_length,
        win_length=_WIN_LENGTH,
        window=window,
        center=True,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    power = spec.real * spec.real + spec.imag * spec.imag  # (freq, time)
    mel = mel_fb @ power  # (n_mels, time)

    mel = torch.clamp(mel, min=_EPS)
    maxv = torch.max(mel)
    mel_db = 10.0 * torch.log10(mel) - 10.0 * torch.log10(torch.clamp(maxv, min=_EPS))

    mel_db = mel_db.to(dtype=torch.float32).detach().cpu().numpy()
    mel_db = (mel_db + 40.0) / 40.0
    return mel_db.astype(np.float32, copy=False)


def spectrogram_from_eeg(parquet_path, display=False):
    eeg_df = pd.read_parquet(parquet_path, columns=_EEG_COLS_NEEDED, engine="pyarrow")
    n = len(eeg_df)
    middle = (n - 10_000) // 2
    if middle < 0:
        middle = 0
    eeg_df = eeg_df.iloc[middle : middle + 10_000]

    eeg_arr = eeg_df.to_numpy(dtype=np.float32, copy=False)  # shape (T, C)
    col2idx = {c: i for i, c in enumerate(_EEG_COLS_NEEDED)}

    img = np.zeros((128, 256, 4), dtype="float32")

    if display:
        plt.figure(figsize=(10, 7))
    signals = []
    hop_length = max(1, len(eeg_arr) // 256)

    for k in range(4):
        COLS = FEATS[k]

        for kk in range(4):
            a = eeg_arr[:, col2idx[COLS[kk]]]
            b = eeg_arr[:, col2idx[COLS[kk + 1]]]
            x = a - b

            m = np.nanmean(x)
            if np.isnan(x).mean() < 1:
                x = np.nan_to_num(x, nan=float(m), copy=False)
            else:
                x = np.zeros_like(x, dtype=np.float32)

            if USE_WAVELET:
                x = denoise(x, wavelet=USE_WAVELET).astype(np.float32, copy=False)

            signals.append(x)

            mel_spec_db = _torch_mel_db_norm(x, hop_length=hop_length)  # (128, T)

            width = (mel_spec_db.shape[1] // 32) * 32
            mel_spec_db = mel_spec_db[:, :width]

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
df = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/test.csv")




## === cell 4
train_df_full = pd.read_csv(paths.train_csv)
test_df = pd.read_csv(paths.test_csv)
print(f"Train dataframe shape is: {train_df_full.shape}")
print(f"Test dataframe shape is: {test_df.shape}")

train_votes = train_df_full[TARGETS].values.astype(np.float32)
train_votes = np.clip(train_votes, 0.0, None)
train_probs = train_votes / np.clip(train_votes.sum(axis=1, keepdims=True), 1e-6, None)
train_df_full[TARGETS] = train_probs

patients = train_df_full["patient_id"].values
unique_patients = np.unique(patients)
rng = np.random.default_rng(42)
rng.shuffle(unique_patients)
val_patients = set(unique_patients[: max(1, int(0.1 * len(unique_patients)))])
is_val = train_df_full["patient_id"].isin(val_patients)
train_df = train_df_full.loc[~is_val].reset_index(drop=True)
val_df = train_df_full.loc[is_val].reset_index(drop=True)
print(f"Train/Val split: {train_df.shape} / {val_df.shape}")

max_train_rows = 12000
if len(train_df) > max_train_rows:
    train_df = train_df.sample(max_train_rows, random_state=42).reset_index(drop=True)
max_val_rows = 2000
if len(val_df) > max_val_rows:
    val_df = val_df.sample(max_val_rows, random_state=42).reset_index(drop=True)

needed_test_eeg_ids = test_df["eeg_id"].astype(int).unique().tolist()
needed_test_spec_ids = test_df["spectrogram_id"].astype(int).unique().tolist()

needed_train_eeg_ids = []
needed_train_spec_ids = []




## === cell 5
def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _eeg_worker_to_npy(args: Tuple[int, str, str]):
    eeg_id, parquet_path, out_dir = args
    out_path = os.path.join(out_dir, f"{int(eeg_id)}.npy")
    if os.path.exists(out_path):
        return int(eeg_id), out_path
    sp = spectrogram_from_eeg(parquet_path)
    np.save(out_path, np.asarray(sp, dtype=np.float32), allow_pickle=False)
    return int(eeg_id), out_path


def build_eeg_cache_dir(eeg_dir: str, eeg_ids: list[int], out_dir: str):
    _ensure_dir(out_dir)
    eeg_ids_set = set(map(int, eeg_ids))

    fn_map = {
        int(fn.split(".")[0]): fn
        for fn in os.listdir(eeg_dir)
        if fn.endswith(".parquet")
    }
    tasks = []
    for eeg_id in eeg_ids_set:
        out_path = os.path.join(out_dir, f"{int(eeg_id)}.npy")
        if os.path.exists(out_path):
            continue
        fn = fn_map.get(int(eeg_id), None)
        if fn is None:
            continue
        tasks.append((int(eeg_id), os.path.join(eeg_dir, fn), out_dir))

    if tasks:
        n_workers = min(max(1, (os.cpu_count() or 2) - 1), 12)
        chunksize = 16 if len(tasks) > 256 else 8
        if n_workers <= 1 or len(tasks) <= 8:
            for t in tqdm(
                tasks, desc=f"Building EEG mels ({os.path.basename(eeg_dir)})"
            ):
                _eeg_worker_to_npy(t)
        else:
            with multiprocessing.get_context("spawn").Pool(processes=n_workers) as pool:
                for _ in tqdm(
                    pool.imap_unordered(_eeg_worker_to_npy, tasks, chunksize=chunksize),
                    total=len(tasks),
                    desc=f"Building EEG mels ({os.path.basename(eeg_dir)})",
                ):
                    pass

    return {
        int(eeg_id): os.path.join(out_dir, f"{int(eeg_id)}.npy")
        for eeg_id in eeg_ids_set
    }


test_eeg_cache_dir = os.path.join(paths.out, "eeg_cache_test")
train_eeg_cache_dir = os.path.join(paths.out, "eeg_cache_train")

all_eegs_test = build_eeg_cache_dir(
    paths.test_eeg, needed_test_eeg_ids, test_eeg_cache_dir
)
all_eegs_train = {}  # lazy




## === cell 6
def _spec_to_img(spec: np.ndarray) -> np.ndarray:
    Xspec = np.zeros((128, 256, 4), dtype=np.float32)
    n_time = spec.shape[0]
    win = 300
    r = max(0, (n_time - win) // 2)
    r = min(r, max(0, n_time - win))

    ep = 1e-6

    block = spec[r : r + win, :400].T.astype(np.float32, copy=False)  # (400,300)
    block = block.reshape(4, 100, win)  # (4,100,300)

    block = np.clip(block, np.exp(-4), np.exp(8))
    block = np.log(block)

    mu = np.nanmean(block, axis=(1, 2), keepdims=True)
    std = np.nanstd(block, axis=(1, 2), keepdims=True)
    block = (block - mu) / (std + ep)
    block = np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0)

    Xspec[14:-14, :, :] = (block[:, :, 22:-22] / 2.0).transpose(1, 2, 0)
    return Xspec


def _specimg_worker_to_npy(args: Tuple[int, str, str]):
    sid, parquet_path, out_dir = args
    out_path = os.path.join(out_dir, f"{int(sid)}.npy")
    if os.path.exists(out_path):
        return int(sid), out_path
    sp = pd.read_parquet(parquet_path, engine="pyarrow").values.astype(
        np.float32, copy=False
    )
    img = _spec_to_img(sp)
    np.save(out_path, img.astype(np.float32, copy=False), allow_pickle=False)
    return int(sid), out_path


def build_specimg_cache_dir(spec_dir: str, spec_ids: list[int], out_dir: str):
    _ensure_dir(out_dir)
    spec_ids_set = set(map(int, spec_ids))

    fn_map = {
        int(fn.split(".")[0]): fn
        for fn in os.listdir(spec_dir)
        if fn.endswith(".parquet")
    }
    tasks = []
    for sid in spec_ids_set:
        out_path = os.path.join(out_dir, f"{int(sid)}.npy")
        if os.path.exists(out_path):
            continue
        fn = fn_map.get(int(sid), None)
        if fn is None:
            continue
        tasks.append((int(sid), os.path.join(spec_dir, fn), out_dir))

    if tasks:
        n_workers = min(max(1, (os.cpu_count() or 2) - 1), 12)
        chunksize = 32 if len(tasks) > 512 else 16
        if n_workers <= 1 or len(tasks) <= 16:
            for t in tqdm(
                tasks, desc=f"Precomputing spec images ({os.path.basename(spec_dir)})"
            ):
                _specimg_worker_to_npy(t)
        else:
            with multiprocessing.get_context("spawn").Pool(processes=n_workers) as pool:
                for _ in tqdm(
                    pool.imap_unordered(
                        _specimg_worker_to_npy, tasks, chunksize=chunksize
                    ),
                    total=len(tasks),
                    desc=f"Precomputing spec images ({os.path.basename(spec_dir)})",
                ):
                    pass

    return {int(sid): os.path.join(out_dir, f"{int(sid)}.npy") for sid in spec_ids_set}


test_specimg_cache_dir = os.path.join(paths.out, "specimg_cache_test")
train_specimg_cache_dir = os.path.join(paths.out, "specimg_cache_train")

all_specimgs_test = build_specimg_cache_dir(
    paths.test_spec, needed_test_spec_ids, test_specimg_cache_dir
)
all_specimgs_train = {}  # lazy

all_spectrograms_test = None
all_spectrograms_train = None




## === cell 7
class _LRUCache:
    def __init__(self, max_items: int = 2048):
        self.max_items = int(max_items)
        self._d = OrderedDict()

    def get(self, k):
        v = self._d.get(k, None)
        if v is not None:
            self._d.move_to_end(k)
        return v

    def put(self, k, v):
        self._d[k] = v
        self._d.move_to_end(k)
        if len(self._d) > self.max_items:
            self._d.popitem(last=False)


_EEG_IMG_CACHE = _LRUCache(max_items=1024)
_SPEC_IMG_CACHE = _LRUCache(max_items=2048)

_train_eeg_fn_map = {
    int(fn.split(".")[0]): os.path.join(paths.train_eeg_dir, fn)
    for fn in os.listdir(paths.train_eeg_dir)
    if fn.endswith(".parquet")
}
_train_spec_fn_map = {
    int(fn.split(".")[0]): os.path.join(paths.train_spec_dir, fn)
    for fn in os.listdir(paths.train_spec_dir)
    if fn.endswith(".parquet")
}


class CustomDataset:
    def __init__(
        self,
        traindf: pd.DataFrame,
        config,
        mode: str = "train",
        specs: dict[int, np.ndarray] = None,  # kept for compatibility
        eegs: dict[
            int, str
        ] = None,  # id->.npy path (test precomputed) or empty (train lazy)
        specimgs: dict[
            int, str
        ] = None,  # id->.npy path (test precomputed) or empty (train lazy)
    ):
        self.traindf = traindf
        self.specs = specs
        self.specimgs = specimgs or {}
        self.eeg = eegs or {}
        self.mode = mode

    def __len__(self):
        return len(self.traindf)

    def _get_spec_img(self, sid: int) -> np.ndarray:
        cached = _SPEC_IMG_CACHE.get(sid)
        if cached is not None:
            return cached

        if sid in self.specimgs:
            arr = np.load(
                self.specimgs[sid]
            )  # fully load; faster than mmap for small arrays + avoids per-access overhead
        else:
            p = _train_spec_fn_map.get(sid, None)
            if p is None:
                arr = np.zeros((128, 256, 4), dtype=np.float32)
            else:
                sp = pd.read_parquet(p, engine="pyarrow").values.astype(
                    np.float32, copy=False
                )
                arr = _spec_to_img(sp)

        arr = np.asarray(arr, dtype=np.float32)
        _SPEC_IMG_CACHE.put(sid, arr)
        return arr

    def _get_eeg_img(self, eeg_id: int) -> np.ndarray:
        cached = _EEG_IMG_CACHE.get(eeg_id)
        if cached is not None:
            return cached

        if eeg_id in self.eeg:
            arr = np.load(self.eeg[eeg_id])  # fully load
        else:
            p = _train_eeg_fn_map.get(eeg_id, None)
            if p is None:
                arr = np.zeros((128, 256, 4), dtype=np.float32)
            else:
                arr = spectrogram_from_eeg(p)

        arr = np.asarray(arr, dtype=np.float32)
        _EEG_IMG_CACHE.put(eeg_id, arr)
        return arr

    def __getitem__(self, idx):
        row = self.traindf.iloc[idx]

        sid = int(row.spectrogram_id)
        spec4 = self._get_spec_img(sid)  # (128,256,4)

        eeg_id = int(row.eeg_id)
        eeg_img = self._get_eeg_img(eeg_id)  # (128,256,4)

        x = torch.empty((3, 512, 512), dtype=torch.float32)

        s = torch.from_numpy(spec4).permute(2, 0, 1).contiguous()  # (4,128,256)
        s = s.reshape(512, 256)  # (512,256)
        e = torch.from_numpy(eeg_img).permute(2, 0, 1).contiguous()  # (4,128,256)
        e = e.reshape(512, 256)

        x0 = torch.cat([s, e], dim=1).contiguous()  # (512,512)
        x[0].copy_(x0)
        x[1].copy_(x0)
        x[2].copy_(x0)

        y = np.zeros(6, dtype="float32")
        if self.mode != "test" and all(t in self.traindf.columns for t in TARGETS):
            y = row[TARGETS].values.astype(np.float32)

        return {"data": x, "target": torch.from_numpy(y).float()}




## === cell 8
train_dataset = CustomDataset(
    train_df,
    config,
    mode="train",
    specs=all_spectrograms_train,
    eegs=all_eegs_train,
    specimgs=all_specimgs_train,
)
val_dataset = CustomDataset(
    val_df,
    config,
    mode="train",
    specs=all_spectrograms_train,
    eegs=all_eegs_train,
    specimgs=all_specimgs_train,
)
test_dataset = CustomDataset(
    test_df,
    config,
    mode="test",
    specs=all_spectrograms_test,
    eegs=all_eegs_test,
    specimgs=all_specimgs_test,
)


def _loader_workers():
    if not torch.cuda.is_available():
        return 0
    return min(4, max(1, (os.cpu_count() or 2) // 2))


num_workers = _loader_workers()
prefetch_factor = 2 if num_workers > 0 else None
persistent_workers = bool(num_workers > 0)

train_loader = DataLoader(
    train_dataset,
    batch_size=config.batchsize,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=persistent_workers,
    prefetch_factor=prefetch_factor,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=config.batchsize,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=persistent_workers,
    prefetch_factor=prefetch_factor,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=config.batchsize,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=persistent_workers,
    prefetch_factor=prefetch_factor,
)




## === cell 9
class Custommodel(nn.Module):
    def __init__(self, config, numclass: int = 6):
        super(Custommodel, self).__init__()
        self.model = timm.create_model(
            config.model,
            pretrained=False,
            drop_rate=0.1,
            drop_path_rate=0.2,
            num_classes=0,  # make a pure feature extractor in timm-safe way
        )
        self.customlayer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.model.num_features, numclass),
        )

    def forward(self, x):
        x = self.model(x)
        x = self.customlayer(x)
        return x




## === cell 10
def train_one_epoch(model, loader, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0
    n = 0
    for batch in tqdm(loader, desc="Train", leave=False):
        x = batch["data"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda",
            enabled=(device.type == "cuda" and config.AMP),
            dtype=torch.float16,
        ):
            logits = model(x)
            log_probs = torch.log_softmax(logits, dim=1)
            loss = -(y * log_probs).sum(dim=1).mean()

        if scaler is not None and device.type == "cuda" and config.AMP:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        n += bs
    return total_loss / max(n, 1)


@torch.no_grad()
def valid_one_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    n = 0
    for batch in tqdm(loader, desc="Valid", leave=False):
        x = batch["data"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)
        logits = model(x)
        log_probs = torch.log_softmax(logits, dim=1)
        loss = -(y * log_probs).sum(dim=1).mean()
        bs = x.size(0)
        total_loss += loss.item() * bs
        n += bs
    return total_loss / max(n, 1)




## === cell 11
model = Custommodel(config).to(device)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=config.lr, weight_decay=config.WEIGHT_DECAY
)
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and config.AMP))

best_val = float("inf")
best_state = None

for ep in range(config.epoch):
    tr_loss = train_one_epoch(model, train_loader, optimizer, device, scaler=scaler)
    va_loss = valid_one_epoch(model, val_loader, device)
    print(
        f"Epoch {ep+1}/{config.epoch} | train_loss={tr_loss:.5f} val_loss={va_loss:.5f}"
    )
    if va_loss < best_val:
        best_val = va_loss
        best_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }

if best_state is not None:
    model.load_state_dict(best_state, strict=True)




## === cell 12
def inference_function(test_loader, model, device):
    model.eval()
    softmax = nn.Softmax(dim=1)
    preds = []
    for batch in tqdm(test_loader, desc="Infer", leave=False):
        x = batch["data"].to(device, non_blocking=True)
        with torch.no_grad():
            ypred = model(x)
            ypred = softmax(ypred)
        preds.append(ypred.detach().cpu().numpy())
    return {"predictions": np.concatenate(preds, axis=0)}




## === cell 13
pred_dict = inference_function(test_loader, model, device)
predictions = pred_dict["predictions"]




## === cell 14
predictions




## === cell 15
sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})

preds = np.asarray(predictions, dtype=np.float32)
if preds.ndim != 2 or preds.shape[0] != len(sub) or preds.shape[1] != 6:
    preds = np.full((len(sub), 6), 1.0 / 6.0, dtype=np.float32)

preds = np.clip(preds, 1e-8, 1.0)
preds = preds / preds.sum(axis=1, keepdims=True)

sub[TARGETS] = preds
sub.to_csv("submission.csv", index=False)
print(f"Submission shape: {sub.shape}")
sub.head()
