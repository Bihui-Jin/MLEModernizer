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

0.5760566692629098

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 1.40995) has done: 'I fix the missing model-weights path error by falling back to a valid, always-available baseline when `/kaggle/input/resnet5010ep` doesn’t exist, so the notebook always produces a submission. I also fix multiple runtime/logic issues in the dataset: undefined `targets`, wrong indexing into spectrogram arrays, the noisy `print(r)`, and shape handling so batches are consistent. Finally, I enforce probability normalization (sum to 1) and ensure the submission columns match `sample_submission.csv`, producing a valid `submission.csv` end-to-end within the Kaggle environment.'

# 9. Code solution

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

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")




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

    FIX: Force CPU only (no .to('cuda')) to avoid CUDA init inside DataLoader workers.
    """
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
    return {
        int(eeg_id): os.path.join(out_dir, f"{int(eeg_id)}.npy")
        for eeg_id in eeg_ids_set
    }


test_eeg_cache_dir = os.path.join(paths.out, "eeg_cache_test")
train_eeg_cache_dir = os.path.join(paths.out, "eeg_cache_train")
_ensure_dir(test_eeg_cache_dir)
_ensure_dir(train_eeg_cache_dir)

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
    return {int(sid): os.path.join(out_dir, f"{int(sid)}.npy") for sid in spec_ids_set}


test_specimg_cache_dir = os.path.join(paths.out, "specimg_cache_test")
train_specimg_cache_dir = os.path.join(paths.out, "specimg_cache_train")
_ensure_dir(test_specimg_cache_dir)
_ensure_dir(train_specimg_cache_dir)

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

_test_eeg_fn_map = {
    int(fn.split(".")[0]): os.path.join(paths.test_eeg, fn)
    for fn in os.listdir(paths.test_eeg)
    if fn.endswith(".parquet")
}
_test_spec_fn_map = {
    int(fn.split(".")[0]): os.path.join(paths.test_spec, fn)
    for fn in os.listdir(paths.test_spec)
    if fn.endswith(".parquet")
}


class CustomDataset:
    def __init__(
        self,
        traindf: pd.DataFrame,
        config,
        mode: str = "train",
        specs: dict[int, np.ndarray] = None,  # kept for compatibility
        eegs: dict[int, str] = None,  # id->.npy path (cache path)
        specimgs: dict[int, str] = None,  # id->.npy path (cache path)
    ):
        self.traindf = traindf
        self.specs = specs
        self.specimgs = specimgs or {}
        self.eeg = eegs or {}
        self.mode = mode

        self._eeg_ids = self.traindf["eeg_id"].to_numpy(dtype=np.int64, copy=False)
        self._spec_ids = self.traindf["spectrogram_id"].to_numpy(
            dtype=np.int64, copy=False
        )
        self._has_targets = (self.mode != "test") and all(
            t in self.traindf.columns for t in TARGETS
        )
        if self._has_targets:
            self._targets = self.traindf[TARGETS].to_numpy(dtype=np.float32, copy=False)
        else:
            self._targets = None

    def __len__(self):
        return len(self.traindf)

    def _load_or_build_spec_img(self, sid: int) -> np.ndarray:
        cached = _SPEC_IMG_CACHE.get(sid)
        if cached is not None:
            return cached

        out_path = self.specimgs.get(sid, None)
        if out_path is not None and os.path.exists(out_path):
            arr = np.load(out_path)
        else:
            p = (
                _train_spec_fn_map.get(sid)
                if self.mode != "test"
                else _test_spec_fn_map.get(sid)
            )
            if p is None:
                arr = np.zeros((128, 256, 4), dtype=np.float32)
            else:
                sp = pd.read_parquet(p, engine="pyarrow").values.astype(
                    np.float32, copy=False
                )
                arr = _spec_to_img(sp)
            if out_path is not None:
                try:
                    np.save(
                        out_path, np.asarray(arr, dtype=np.float32), allow_pickle=False
                    )
                except Exception:
                    pass

        arr = np.asarray(arr, dtype=np.float32)
        _SPEC_IMG_CACHE.put(sid, arr)
        return arr

    def _load_or_build_eeg_img(self, eeg_id: int) -> np.ndarray:
        cached = _EEG_IMG_CACHE.get(eeg_id)
        if cached is not None:
            return cached

        out_path = self.eeg.get(eeg_id, None)
        if out_path is not None and os.path.exists(out_path):
            arr = np.load(out_path)
        else:
            p = (
                _train_eeg_fn_map.get(eeg_id)
                if self.mode != "test"
                else _test_eeg_fn_map.get(eeg_id)
            )
            if p is None:
                arr = np.zeros((128, 256, 4), dtype=np.float32)
            else:
                arr = spectrogram_from_eeg(p)
            if out_path is not None:
                try:
                    np.save(
                        out_path, np.asarray(arr, dtype=np.float32), allow_pickle=False
                    )
                except Exception:
                    pass

        arr = np.asarray(arr, dtype=np.float32)
        _EEG_IMG_CACHE.put(eeg_id, arr)
        return arr

    def __getitem__(self, idx):
        sid = int(self._spec_ids[idx])
        spec4 = self._load_or_build_spec_img(sid)  # (128,256,4)

        eeg_id = int(self._eeg_ids[idx])
        eeg_img = self._load_or_build_eeg_img(eeg_id)  # (128,256,4)

        s = np.transpose(spec4, (2, 0, 1)).reshape(512, 256)  # view/copy as needed
        e = np.transpose(eeg_img, (2, 0, 1)).reshape(512, 256)
        x0 = np.concatenate((s, e), axis=1).astype(np.float32, copy=False)  # (512,512)

        x = torch.from_numpy(x0).unsqueeze(0).repeat(3, 1, 1)  # (3,512,512)

        if self._has_targets:
            y = torch.from_numpy(self._targets[idx]).float()
        else:
            y = torch.zeros(6, dtype=torch.float32)

        return {"data": x, "target": y}




## === cell 8
train_dataset = CustomDataset(
    train_df,
    config,
    mode="train",
    specs=all_spectrograms_train,
    eegs=all_eegs_train,  # train remains lazy (computed on-demand + LRU)
    specimgs=all_specimgs_train,  # train remains lazy
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
    eegs=all_eegs_test,  # points to cache paths; built on-demand
    specimgs=all_specimgs_test,  # points to cache paths; built on-demand
)


def _loader_workers():
    return 0


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

if torch.cuda.is_available():
    try:
        model = torch.compile(model, mode="reduce-overhead")
    except Exception as e:
        print("torch.compile skipped:", repr(e))

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




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
TorchRuntimeError                         Traceback (most recent call last)
/tmp/ipykernel_55/2849527436.py in <cell line: 0>()
     16 
     17 for ep in range(config.epoch):
---> 18     tr_loss = train_one_epoch(model, train_loader, optimizer, device, scaler=scaler)
     19     va_loss = valid_one_epoch(model, val_loader, device)
     20     print(

/tmp/ipykernel_55/2496015898.py in train_one_epoch(model, loader, optimizer, device, scaler)
     13             dtype=torch.float16,
     14         ):
---> 15             logits = model(x)
     16             log_probs = torch.log_softmax(logits, dim=1)
     17             loss = -(y * log_probs).sum(dim=1).mean()

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/eval_frame.py in _fn(*args, **kwargs)
    572 
    573             try:
--> 574                 return fn(*args, **kwargs)
    575             finally:
    576                 # Restore the dynamic layer stack depth if necessary.

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in __call__(self, frame, cache_entry, frame_state)
   1378         with compile_lock, _disable_current_modes():
   1379             # skip=1: skip this frame
-> 1380             return self._torchdynamo_orig_callable(
   1381                 frame, cache_entry, self.hooks, frame_state, skip=1
   1382             )

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in __call__(self, frame, cache_entry, hooks, frame_state, skip)
   1162         counters["frames"]["total"] += 1
   1163         try:
-> 1164             result = self._inner_convert(
   1165                 frame, cache_entry, hooks, frame_state, skip=skip + 1
   1166             )

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in __call__(self, frame, cache_entry, hooks, frame_state, skip)
    545 
    546         with compile_context(CompileContext(compile_id)):
--> 547             return _compile(
    548                 frame.f_code,
    549                 frame.f_globals,

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in _compile(code, globals, locals, builtins, closure, compiler_fn, one_graph, export, export_constraints, hooks, cache_entry, cache_size, frame, frame_state, compile_id, skip)
    984         guarded_code = None
    985         try:
--> 986             guarded_code = compile_inner(code, one_graph, hooks, transform)
    987 
    988             # NB: We only put_code_state in success case.  Success case here

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in compile_inner(code, one_graph, hooks, transform)
    713             stack.enter_context(torch._dynamo.callback_handler.install_callbacks())
    714             stack.enter_context(CompileTimeInstructionCounter.record())
--> 715             return _compile_inner(code, one_graph, hooks, transform)
    716 
    717         return None  # dead, but see https://github.com/python/mypy/issues/7577

/usr/local/lib/python3.11/dist-packages/torch/_utils_internal.py in wrapper_function(*args, **kwargs)
     93 
     94             if not StrobelightCompileTimeProfiler.enabled:
---> 95                 return function(*args, **kwargs)
     96 
     97             return StrobelightCompileTimeProfiler.profile_compile_time(

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in _compile_inner(code, one_graph, hooks, transform)
    748             CompileContext.get().attempt = attempt
    749             try:
--> 750                 out_code = transform_code_object(code, transform)
    751                 break
    752             except exc.RestartAnalysis as e:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/bytecode_transformation.py in transform_code_object(code, transformations, safe)
   1359     propagate_line_nums(instructions)
   1360 
-> 1361     transformations(instructions, code_options)
   1362     return clean_and_assemble_instructions(instructions, keys, code_options)[1]
   1363 

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in _fn(*args, **kwargs)
    229             exit_stack.enter_context(torch_function_mode_stack_state_mgr)
    230             try:
--> 231                 return fn(*args, **kwargs)
    232             finally:
    233                 cleanup.close()

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in transform(instructions, code_options)
    660         try:
    661             with tracing(tracer.output.tracing_context), tracer.set_current_tx():
--> 662                 tracer.run()
    663         except exc.UnspecializeRestartAnalysis:
    664             speculation_log.clear()

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in run(self)
   2866 
   2867     def run(self):
-> 2868         super().run()
   2869 
   2870     def should_compile_partial_graph(self):

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in run(self)
   1050             try:
   1051                 self.output.push_tx(self)
-> 1052                 while self.step():
   1053                     pass
   1054             except TensorifyScalarRestartAnalysis:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in step(self)
    960 
    961         try:
--> 962             self.dispatch_table[inst.opcode](self, inst)
    963             return not self.output.should_exit
    964         except TensorifyScalarRestartAnalysis:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in wrapper(self, inst)
    657                 return handle_graph_break(self, inst, speculation.reason)
    658             try:
--> 659                 return inner_fn(self, inst)
    660             except Unsupported as excp:
    661                 if self.generic_context_manager_depth > 0:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in CALL(self, inst)
   2339     @break_graph_if_unsupported(push=1)
   2340     def CALL(self, inst):
-> 2341         self._call(inst)
   2342 
   2343     def COPY(self, inst):

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in _call(self, inst, call_kw)
   2333             # if call_function fails, need to set kw_names to None, otherwise
   2334             # a subsequent call may have self.kw_names set to an old value
-> 2335             self.call_function(fn, args, kwargs)
   2336         finally:
   2337             self.kw_names = None

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in call_function(self, fn, args, kwargs)
    895         if inner_fn and callable(inner_fn) and is_forbidden(inner_fn):
    896             raise AssertionError(f"Attempt to trace forbidden callable {inner_fn}")
--> 897         self.push(fn.call_function(self, args, kwargs))  # type: ignore[arg-type]
    898 
    899     def inline_user_function_return(self, fn, args, kwargs):

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/variables/lazy.py in realize_and_forward(self, *args, **kwargs)
    168         self: LazyVariableTracker, *args: Any, **kwargs: Any
    169     ) -> Any:
--> 170         return getattr(self.realize(), name)(*args, **kwargs)
    171 
    172     return realize_and_forward

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/variables/nn_module.py in call_function(self, tx, args, kwargs)
    912         )
    913         with ctx:
--> 914             return variables.UserFunctionVariable(fn, source=source).call_function(
    915                 tx, [self] + list(args), kwargs
    916             )

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/variables/functions.py in call_function(self, tx, args, kwargs)
    315                 with torch._dynamo.side_effects.allow_side_effects_under_checkpoint(tx):
    316                     return super().call_function(tx, args, kwargs)
--> 317         return super().call_function(tx, args, kwargs)
    318 
    319 

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/variables/functions.py in call_function(self, tx, args, kwargs)
    116         kwargs: "Dict[str, VariableTracker]",
    117     ) -> "VariableTracker":
--> 118         return tx.inline_user_function_return(self, [*self.self_args(), *args], kwargs)
    119 
    120     def call_hasattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in inline_user_function_return(self, fn, args, kwargs)
    901         A call to some user defined function by inlining it.
    902         """
--> 903         return InliningInstructionTranslator.inline_call(self, fn, args, kwargs)
    904 
    905     def get_line_of_code_header(self, lineno=None):

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in inline_call(cls, parent, func, args, kwargs)
   3070     def inline_call(cls, parent, func, args, kwargs):
   3071         with patch.dict(counters, {"unimplemented": counters["inline_call"]}):
-> 3072             return cls.inline_call_(parent, func, args, kwargs)
   3073 
   3074     @staticmethod

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in inline_call_(parent, func, args, kwargs)
   3196         try:
   3197             with strict_ctx:
-> 3198                 tracer.run()
   3199         except exc.ObservedException as e:
   3200             msg = f"Observed exception DURING INLING {code} : {e}"

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in run(self)
   1050             try:
   1051                 self.output.push_tx(self)
-> 1052                 while self.step():
   1053                     pass
   1054             except TensorifyScalarRestartAnalysis:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in step(self)
    960 
    961         try:
--> 962             self.dispatch_table[inst.opcode](self, inst)
    963             return not self.output.should_exit
    964         except TensorifyScalarRestartAnalysis:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in wrapper(self, inst)
    657                 return handle_graph_break(self, inst, speculation.reason)
    658             try:
--> 659                 return inner_fn(self, inst)
    660             except Unsupported as excp:
    661                 if self.generic_context_manager_depth > 0:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in CALL(self, inst)
   2339     @break_graph_if_unsupported(push=1)
   2340     def CALL(self, inst):
-> 2341         self._call(inst)
   2342 
   2343     def COPY(self, inst):

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in _call(self, inst, call_kw)
   2333             # if call_function fails, need to set kw_names to None, otherwise
   2334             # a subsequent call may have self.kw_names set to an old value
-> 2335             self.call_function(fn, args, kwargs)
   2336         finally:
   2337             self.kw_names = None

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in call_function(self, fn, args, kwargs)
    895         if inner_fn and callable(inner_fn) and is_forbidden(inner_fn):
    896             raise AssertionError(f"Attempt to trace forbidden callable {inner_fn}")
--> 897         self.push(fn.call_function(self, args, kwargs))  # type: ignore[arg-type]
    898 
    899     def inline_user_function_return(self, fn, args, kwargs):

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/variables/nn_module.py in call_function(self, tx, args, kwargs)
    912         )
    913         with ctx:
--> 914             return variables.UserFunctionVariable(fn, source=source).call_function(
    915                 tx, [self] + list(args), kwargs
    916             )

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/variables/functions.py in call_function(self, tx, args, kwargs)
    315                 with torch._dynamo.side_effects.allow_side_effects_under_checkpoint(tx):
    316                     return super().call_function(tx, args, kwargs)
--> 317         return super().call_function(tx, args, kwargs)
    318 
    319 

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/variables/functions.py in call_function(self, tx, args, kwargs)
    116         kwargs: "Dict[str, VariableTracker]",
    117     ) -> "VariableTracker":
--> 118         return tx.inline_user_function_return(self, [*self.self_args(), *args], kwargs)
    119 
    120     def call_hasattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in inline_user_function_return(self, fn, args, kwargs)
    901         A call to some user defined function by inlining it.
    902         """
--> 903         return InliningInstructionTranslator.inline_call(self, fn, args, kwargs)
    904 
    905     def get_line_of_code_header(self, lineno=None):

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in inline_call(cls, parent, func, args, kwargs)
   3070     def inline_call(cls, parent, func, args, kwargs):
   3071         with patch.dict(counters, {"unimplemented": counters["inline_call"]}):
-> 3072             return cls.inline_call_(parent, func, args, kwargs)
   3073 
   3074     @staticmethod

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in inline_call_(parent, func, args, kwargs)
   3196         try:
   3197             with strict_ctx:
-> 3198                 tracer.run()
   3199         except exc.ObservedException as e:
   3200             msg = f"Observed exception DURING INLING {code} : {e}"

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in run(self)
   1050             try:
   1051                 self.output.push_tx(self)
-> 1052                 while self.step():
   1053                     pass
   1054             except TensorifyScalarRestartAnalysis:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in step(self)
    960 
    961         try:
--> 962             self.dispatch_table[inst.opcode](self, inst)
    963             return not self.output.should_exit
    964         except TensorifyScalarRestartAnalysis:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in wrapper(self, inst)
    657                 return handle_graph_break(self, inst, speculation.reason)
    658             try:
--> 659                 return inner_fn(self, inst)
    660             except Unsupported as excp:
    661                 if self.generic_context_manager_depth > 0:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in CALL(self, inst)
   2339     @break_graph_if_unsupported(push=1)
   2340     def CALL(self, inst):
-> 2341         self._call(inst)
   2342 
   2343     def COPY(self, inst):

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in _call(self, inst, call_kw)
   2333             # if call_function fails, need to set kw_names to None, otherwise
   2334             # a subsequent call may have self.kw_names set to an old value
-> 2335             self.call_function(fn, args, kwargs)
   2336         finally:
   2337             self.kw_names = None

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in call_function(self, fn, args, kwargs)
    895         if inner_fn and callable(inner_fn) and is_forbidden(inner_fn):
    896             raise AssertionError(f"Attempt to trace forbidden callable {inner_fn}")
--> 897         self.push(fn.call_function(self, args, kwargs))  # type: ignore[arg-type]
    898 
    899     def inline_user_function_return(self, fn, args, kwargs):

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/variables/torch.py in call_function(self, tx, args, kwargs)
    951             fake_out_shape = kwargs["out"].proxy.node.meta["example_value"].shape
    952 
--> 953         tensor_variable = wrap_fx_proxy(
    954             tx=tx,
    955             proxy=tx.output.create_proxy(

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/variables/builder.py in wrap_fx_proxy(tx, proxy, example_value, subclass_type, **options)
   2151     }
   2152     if subclass_type is None:
-> 2153         return wrap_fx_proxy_cls(target_cls=TensorVariable, **kwargs)
   2154     else:
   2155         result = wrap_fx_proxy_cls(target_cls=TensorWithTFOverrideVariable, **kwargs)

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/variables/builder.py in wrap_fx_proxy_cls(target_cls, tx, proxy, example_value, subclass_type, **options)
   2217 ):
   2218     if example_value is None:
-> 2219         return _wrap_fx_proxy(
   2220             target_cls, tx, proxy, example_value, subclass_type, **options
   2221         )

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/variables/builder.py in _wrap_fx_proxy(target_cls, tx, proxy, example_value, subclass_type, **options)
   2313         # only allow_non_graph_fake in this instance because we handle the non-fake
   2314         # cases properly below.
-> 2315         example_value = get_fake_value(proxy.node, tx, allow_non_graph_fake=True)
   2316 
   2317     return handle_traced_output(

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/utils.py in get_fake_value(node, tx, allow_non_graph_fake)
   2534             unimplemented(f"TypeError {node.target}: {cause}")
   2535 
-> 2536         raise TorchRuntimeError(str(e)).with_traceback(e.__traceback__) from None
   2537 
   2538     if not allow_non_graph_fake:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/utils.py in get_fake_value(node, tx, allow_non_graph_fake)
   2469     try:
   2470         with tx.fake_mode, enable_python_dispatcher():
-> 2471             ret_val = wrap_fake_exception(
   2472                 lambda: run_node(tx.output, node, args, kwargs, nnmodule)
   2473             )

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/utils.py in wrap_fake_exception(fn)
   2015 def wrap_fake_exception(fn):
   2016     try:
-> 2017         return fn()
   2018     except UnsupportedFakeTensorException as e:
   2019         from .exc import unimplemented

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/utils.py in <lambda>()
   2470         with tx.fake_mode, enable_python_dispatcher():
   2471             ret_val = wrap_fake_exception(
-> 2472                 lambda: run_node(tx.output, node, args, kwargs, nnmodule)
   2473             )
   2474     except Unsupported:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/utils.py in run_node(tracer, node, args, kwargs, nnmodule)
   2602             unimplemented(make_error_message(e), from_exc=e)
   2603         except Exception as e:
-> 2604             raise RuntimeError(make_error_message(e)).with_traceback(
   2605                 e.__traceback__
   2606             ) from e

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/utils.py in run_node(tracer, node, args, kwargs, nnmodule)
   2584         try:
   2585             if op == "call_function":
-> 2586                 return node.target(*args, **kwargs)
   2587             elif op == "call_method":
   2588                 return getattr(args[0], node.target)(*args[1:], **kwargs)

/usr/local/lib/python3.11/dist-packages/torch/utils/_stats.py in wrapper(*args, **kwargs)
     19             simple_call_counter[fn.__qualname__] = 0
     20         simple_call_counter[fn.__qualname__] = simple_call_counter[fn.__qualname__] + 1
---> 21         return fn(*args, **kwargs)
     22     return wrapper

/usr/local/lib/python3.11/dist-packages/torch/_subclasses/fake_tensor.py in __torch_dispatch__(self, func, types, args, kwargs)
   1274         ), func
   1275         try:
-> 1276             return self.dispatch(func, types, args, kwargs)
   1277         except TypeError:
   1278             log.exception("fake tensor raised TypeError")

/usr/local/lib/python3.11/dist-packages/torch/_subclasses/fake_tensor.py in dispatch(self, func, types, args, kwargs)
   1814 
   1815         if self.cache_enabled:
-> 1816             return self._cached_dispatch_impl(func, types, args, kwargs)
   1817         else:
   1818             return self._dispatch_impl(func, types, args, kwargs)

/usr/local/lib/python3.11/dist-packages/torch/_subclasses/fake_tensor.py in _cached_dispatch_impl(self, func, types, args, kwargs)
   1375             else:
   1376                 self._validate_cache_key(func, args, kwargs)
-> 1377                 output = self._dispatch_impl(func, types, args, kwargs)
   1378                 entry = self._make_cache_entry(state, key, func, args, kwargs, output)
   1379                 key.strip_shape_env()

/usr/local/lib/python3.11/dist-packages/torch/_subclasses/fake_tensor.py in _dispatch_impl(self, func, types, args, kwargs)
   2288                 with self:
   2289                     return maybe_propagate_real_tensors(
-> 2290                         decomposition_table[func](*args, **kwargs)
   2291                     )
   2292 

/usr/local/lib/python3.11/dist-packages/torch/_prims_common/wrappers.py in _fn(out, *args, **kwargs)
    289                 result = fn(*args, is_out=(out is not None), **kwargs)  # type: ignore[arg-type]
    290             else:
--> 291                 result = fn(*args, **kwargs)
    292             assert (
    293                 isinstance(result, TensorLike)

/usr/local/lib/python3.11/dist-packages/torch/_decomp/decompositions.py in inner(*args, **kwargs)
     81                 return x
     82 
---> 83         r = f(*tree_map(increase_prec, args), **tree_map(increase_prec, kwargs))
     84         if compute_dtype_only:
     85             return r

/usr/local/lib/python3.11/dist-packages/torch/_decomp/decompositions.py in addmm(self, mat1, mat2, beta, alpha)
   1448         beta = int(beta)
   1449         alpha = int(alpha)
-> 1450     out = alpha * torch.mm(mat1, mat2)
   1451     if beta == 0:
   1452         return out

/usr/local/lib/python3.11/dist-packages/torch/utils/_stats.py in wrapper(*args, **kwargs)
     19             simple_call_counter[fn.__qualname__] = 0
     20         simple_call_counter[fn.__qualname__] = simple_call_counter[fn.__qualname__] + 1
---> 21         return fn(*args, **kwargs)
     22     return wrapper

/usr/local/lib/python3.11/dist-packages/torch/_subclasses/fake_tensor.py in __torch_dispatch__(self, func, types, args, kwargs)
   1274         ), func
   1275         try:
-> 1276             return self.dispatch(func, types, args, kwargs)
   1277         except TypeError:
   1278             log.exception("fake tensor raised TypeError")

/usr/local/lib/python3.11/dist-packages/torch/_subclasses/fake_tensor.py in dispatch(self, func, types, args, kwargs)
   1814 
   1815         if self.cache_enabled:
-> 1816             return self._cached_dispatch_impl(func, types, args, kwargs)
   1817         else:
   1818             return self._dispatch_impl(func, types, args, kwargs)

/usr/local/lib/python3.11/dist-packages/torch/_subclasses/fake_tensor.py in _cached_dispatch_impl(self, func, types, args, kwargs)
   1375             else:
   1376                 self._validate_cache_key(func, args, kwargs)
-> 1377                 output = self._dispatch_impl(func, types, args, kwargs)
   1378                 entry = self._make_cache_entry(state, key, func, args, kwargs, output)
   1379                 key.strip_shape_env()

/usr/local/lib/python3.11/dist-packages/torch/_subclasses/fake_tensor.py in _dispatch_impl(self, func, types, args, kwargs)
   2382         try:
   2383             with in_kernel_invocation_manager(self):
-> 2384                 r = func(*args, **kwargs)
   2385         except NotImplementedError as not_implemented_error:
   2386             return maybe_run_unsafe_fallback(not_implemented_error)

/usr/local/lib/python3.11/dist-packages/torch/_ops.py in __call__(self, *args, **kwargs)
    721     # that are named "self". This way, all the aten ops can be called by kwargs.
    722     def __call__(self, /, *args, **kwargs):
--> 723         return self._op(*args, **kwargs)
    724 
    725     # Use positional-only argument to avoid naming collision with aten ops arguments

/usr/local/lib/python3.11/dist-packages/torch/_prims_common/wrappers.py in _fn(out, *args, **kwargs)
    289                 result = fn(*args, is_out=(out is not None), **kwargs)  # type: ignore[arg-type]
    290             else:
--> 291                 result = fn(*args, **kwargs)
    292             assert (
    293                 isinstance(result, TensorLike)

/usr/local/lib/python3.11/dist-packages/torch/_meta_registrations.py in meta_mm(a, b)
   2125     N, M1 = a.shape
   2126     M2, P = b.shape
-> 2127     torch._check(
   2128         M1 == M2,
   2129         lambda: f"a and b must have same reduction dim, but got [{N}, {M1}] X [{M2}, {P}].",

/usr/local/lib/python3.11/dist-packages/torch/__init__.py in _check(cond, message)
   1654             message. Default: ``None``
   1655     """
-> 1656     _check_with(RuntimeError, cond, message)
   1657 
   1658 

/usr/local/lib/python3.11/dist-packages/torch/__init__.py in _check_with(error_type, cond, message)
   1636         message_evaluated = str(message())
   1637 
-> 1638     raise error_type(message_evaluated)
   1639 
   1640 

TorchRuntimeError: Failed running call_function <built-in function linear>(*(FakeTensor(..., device='cuda:0', size=(1, 1), dtype=torch.float16,
           grad_fn=<MeanBackward1>), Parameter(FakeTensor(..., device='cuda:0', size=(6, 2048), requires_grad=True)), Parameter(FakeTensor(..., device='cuda:0', size=(6,), requires_grad=True))), **{}):
a and b must have same reduction dim, but got [1, 1] X [2048, 6].

from user code:
   File "/tmp/ipykernel_55/175391509.py", line 19, in forward
    x = self.customlayer(x)
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/container.py", line 250, in forward
    input = module(input)
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)

Set TORCH_LOGS="+dynamo" and TORCHDYNAMO_VERBOSE=1 for more information


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True


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




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
TorchRuntimeError                         Traceback (most recent call last)
/tmp/ipykernel_55/1734492889.py in <cell line: 0>()
----> 1 pred_dict = inference_function(test_loader, model, device)
      2 predictions = pred_dict["predictions"]
      3 
      4 

/tmp/ipykernel_55/1934144940.py in inference_function(test_loader, model, device)
      6         x = batch["data"].to(device, non_blocking=True)
      7         with torch.no_grad():
----> 8             ypred = model(x)
      9             ypred = softmax(ypred)
     10         preds.append(ypred.detach().cpu().numpy())

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/eval_frame.py in _fn(*args, **kwargs)
    572 
    573             try:
--> 574                 return fn(*args, **kwargs)
    575             finally:
    576                 # Restore the dynamic layer stack depth if necessary.

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in __call__(self, frame, cache_entry, frame_state)
   1378         with compile_lock, _disable_current_modes():
   1379             # skip=1: skip this frame
-> 1380             return self._torchdynamo_orig_callable(
   1381                 frame, cache_entry, self.hooks, frame_state, skip=1
   1382             )

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in __call__(self, frame, cache_entry, hooks, frame_state, skip)
   1162         counters["frames"]["total"] += 1
   1163         try:
-> 1164             result = self._inner_convert(
   1165                 frame, cache_entry, hooks, frame_state, skip=skip + 1
   1166             )

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in __call__(self, frame, cache_entry, hooks, frame_state, skip)
    545 
    546         with compile_context(CompileContext(compile_id)):
--> 547             return _compile(
    548                 frame.f_code,
    549                 frame.f_globals,

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in _compile(code, globals, locals, builtins, closure, compiler_fn, one_graph, export, export_constraints, hooks, cache_entry, cache_size, frame, frame_state, compile_id, skip)
    984         guarded_code = None
    985         try:
--> 986             guarded_code = compile_inner(code, one_graph, hooks, transform)
    987 
    988             # NB: We only put_code_state in success case.  Success case here

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in compile_inner(code, one_graph, hooks, transform)
    713             stack.enter_context(torch._dynamo.callback_handler.install_callbacks())
    714             stack.enter_context(CompileTimeInstructionCounter.record())
--> 715             return _compile_inner(code, one_graph, hooks, transform)
    716 
    717         return None  # dead, but see https://github.com/python/mypy/issues/7577

/usr/local/lib/python3.11/dist-packages/torch/_utils_internal.py in wrapper_function(*args, **kwargs)
     93 
     94             if not StrobelightCompileTimeProfiler.enabled:
---> 95                 return function(*args, **kwargs)
     96 
     97             return StrobelightCompileTimeProfiler.profile_compile_time(

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in _compile_inner(code, one_graph, hooks, transform)
    748             CompileContext.get().attempt = attempt
    749             try:
--> 750                 out_code = transform_code_object(code, transform)
    751                 break
    752             except exc.RestartAnalysis as e:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/bytecode_transformation.py in transform_code_object(code, transformations, safe)
   1359     propagate_line_nums(instructions)
   1360 
-> 1361     transformations(instructions, code_options)
   1362     return clean_and_assemble_instructions(instructions, keys, code_options)[1]
   1363 

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in _fn(*args, **kwargs)
    229             exit_stack.enter_context(torch_function_mode_stack_state_mgr)
    230             try:
--> 231                 return fn(*args, **kwargs)
    232             finally:
    233                 cleanup.close()

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py in transform(instructions, code_options)
    660         try:
    661             with tracing(tracer.output.tracing_context), tracer.set_current_tx():
--> 662                 tracer.run()
    663         except exc.UnspecializeRestartAnalysis:
    664             speculation_log.clear()

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in run(self)
   2866 
   2867     def run(self):
-> 2868         super().run()
   2869 
   2870     def should_compile_partial_graph(self):

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in run(self)
   1050             try:
   1051                 self.output.push_tx(self)
-> 1052                 while self.step():
   1053                     pass
   1054             except TensorifyScalarRestartAnalysis:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in step(self)
    960 
    961         try:
--> 962             self.dispatch_table[inst.opcode](self, inst)
    963             return not self.output.should_exit
    964         except TensorifyScalarRestartAnalysis:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in wrapper(self, inst)
    657                 return handle_graph_break(self, inst, speculation.reason)
    658             try:
--> 659                 return inner_fn(self, inst)
    660             except Unsupported as excp:
    661                 if self.generic_context_manager_depth > 0:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in CALL(self, inst)
   2339     @break_graph_if_unsupported(push=1)
   2340     def CALL(self, inst):
-> 2341         self._call(inst)
   2342 
   2343     def COPY(self, inst):

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in _call(self, inst, call_kw)
   2333             # if call_function fails, need to set kw_names to None, otherwise
   2334             # a subsequent call may have self.kw_names set to an old value
-> 2335             self.call_function(fn, args, kwargs)
   2336         finally:
   2337             self.kw_names = None

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in call_function(self, fn, args, kwargs)
    895         if inner_fn and callable(inner_fn) and is_forbidden(inner_fn):
    896             raise AssertionError(f"Attempt to trace forbidden callable {inner_fn}")
--> 897         self.push(fn.call_function(self, args, kwargs))  # type: ignore[arg-type]
    898 
    899     def inline_user_function_return(self, fn, args, kwargs):

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/variables/lazy.py in realize_and_forward(self, *args, **kwargs)
    168         self: LazyVariableTracker, *args: Any, **kwargs: Any
    169     ) -> Any:
--> 170         return getattr(self.realize(), name)(*args, **kwargs)
    171 
    172     return realize_and_forward

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/variables/nn_module.py in call_function(self, tx, args, kwargs)
    912         )
    913         with ctx:
--> 914             return variables.UserFunctionVariable(fn, source=source).call_function(
    915                 tx, [self] + list(args), kwargs
    916             )

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/variables/functions.py in call_function(self, tx, args, kwargs)
    315                 with torch._dynamo.side_effects.allow_side_effects_under_checkpoint(tx):
    316                     return super().call_function(tx, args, kwargs)
--> 317         return super().call_function(tx, args, kwargs)
    318 
    319 

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/variables/functions.py in call_function(self, tx, args, kwargs)
    116         kwargs: "Dict[str, VariableTracker]",
    117     ) -> "VariableTracker":
--> 118         return tx.inline_user_function_return(self, [*self.self_args(), *args], kwargs)
    119 
    120     def call_hasattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in inline_user_function_return(self, fn, args, kwargs)
    901         A call to some user defined function by inlining it.
    902         """
--> 903         return InliningInstructionTranslator.inline_call(self, fn, args, kwargs)
    904 
    905     def get_line_of_code_header(self, lineno=None):

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in inline_call(cls, parent, func, args, kwargs)
   3070     def inline_call(cls, parent, func, args, kwargs):
   3071         with patch.dict(counters, {"unimplemented": counters["inline_call"]}):
-> 3072             return cls.inline_call_(parent, func, args, kwargs)
   3073 
   3074     @staticmethod

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in inline_call_(parent, func, args, kwargs)
   3196         try:
   3197             with strict_ctx:
-> 3198                 tracer.run()
   3199         except exc.ObservedException as e:
   3200             msg = f"Observed exception DURING INLING {code} : {e}"

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in run(self)
   1050             try:
   1051                 self.output.push_tx(self)
-> 1052                 while self.step():
   1053                     pass
   1054             except TensorifyScalarRestartAnalysis:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in step(self)
    960 
    961         try:
--> 962             self.dispatch_table[inst.opcode](self, inst)
    963             return not self.output.should_exit
    964         except TensorifyScalarRestartAnalysis:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in wrapper(self, inst)
    657                 return handle_graph_break(self, inst, speculation.reason)
    658             try:
--> 659                 return inner_fn(self, inst)
    660             except Unsupported as excp:
    661                 if self.generic_context_manager_depth > 0:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in CALL(self, inst)
   2339     @break_graph_if_unsupported(push=1)
   2340     def CALL(self, inst):
-> 2341         self._call(inst)
   2342 
   2343     def COPY(self, inst):

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in _call(self, inst, call_kw)
   2333             # if call_function fails, need to set kw_names to None, otherwise
   2334             # a subsequent call may have self.kw_names set to an old value
-> 2335             self.call_function(fn, args, kwargs)
   2336         finally:
   2337             self.kw_names = None

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in call_function(self, fn, args, kwargs)
    895         if inner_fn and callable(inner_fn) and is_forbidden(inner_fn):
    896             raise AssertionError(f"Attempt to trace forbidden callable {inner_fn}")
--> 897         self.push(fn.call_function(self, args, kwargs))  # type: ignore[arg-type]
    898 
    899     def inline_user_function_return(self, fn, args, kwargs):

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/variables/nn_module.py in call_function(self, tx, args, kwargs)
    912         )
    913         with ctx:
--> 914             return variables.UserFunctionVariable(fn, source=source).call_function(
    915                 tx, [self] + list(args), kwargs
    916             )

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/variables/functions.py in call_function(self, tx, args, kwargs)
    315                 with torch._dynamo.side_effects.allow_side_effects_under_checkpoint(tx):
    316                     return super().call_function(tx, args, kwargs)
--> 317         return super().call_function(tx, args, kwargs)
    318 
    319 

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/variables/functions.py in call_function(self, tx, args, kwargs)
    116         kwargs: "Dict[str, VariableTracker]",
    117     ) -> "VariableTracker":
--> 118         return tx.inline_user_function_return(self, [*self.self_args(), *args], kwargs)
    119 
    120     def call_hasattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in inline_user_function_return(self, fn, args, kwargs)
    901         A call to some user defined function by inlining it.
    902         """
--> 903         return InliningInstructionTranslator.inline_call(self, fn, args, kwargs)
    904 
    905     def get_line_of_code_header(self, lineno=None):

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in inline_call(cls, parent, func, args, kwargs)
   3070     def inline_call(cls, parent, func, args, kwargs):
   3071         with patch.dict(counters, {"unimplemented": counters["inline_call"]}):
-> 3072             return cls.inline_call_(parent, func, args, kwargs)
   3073 
   3074     @staticmethod

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in inline_call_(parent, func, args, kwargs)
   3196         try:
   3197             with strict_ctx:
-> 3198                 tracer.run()
   3199         except exc.ObservedException as e:
   3200             msg = f"Observed exception DURING INLING {code} : {e}"

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in run(self)
   1050             try:
   1051                 self.output.push_tx(self)
-> 1052                 while self.step():
   1053                     pass
   1054             except TensorifyScalarRestartAnalysis:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in step(self)
    960 
    961         try:
--> 962             self.dispatch_table[inst.opcode](self, inst)
    963             return not self.output.should_exit
    964         except TensorifyScalarRestartAnalysis:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in wrapper(self, inst)
    657                 return handle_graph_break(self, inst, speculation.reason)
    658             try:
--> 659                 return inner_fn(self, inst)
    660             except Unsupported as excp:
    661                 if self.generic_context_manager_depth > 0:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in CALL(self, inst)
   2339     @break_graph_if_unsupported(push=1)
   2340     def CALL(self, inst):
-> 2341         self._call(inst)
   2342 
   2343     def COPY(self, inst):

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in _call(self, inst, call_kw)
   2333             # if call_function fails, need to set kw_names to None, otherwise
   2334             # a subsequent call may have self.kw_names set to an old value
-> 2335             self.call_function(fn, args, kwargs)
   2336         finally:
   2337             self.kw_names = None

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py in call_function(self, fn, args, kwargs)
    895         if inner_fn and callable(inner_fn) and is_forbidden(inner_fn):
    896             raise AssertionError(f"Attempt to trace forbidden callable {inner_fn}")
--> 897         self.push(fn.call_function(self, args, kwargs))  # type: ignore[arg-type]
    898 
    899     def inline_user_function_return(self, fn, args, kwargs):

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/variables/torch.py in call_function(self, tx, args, kwargs)
    951             fake_out_shape = kwargs["out"].proxy.node.meta["example_value"].shape
    952 
--> 953         tensor_variable = wrap_fx_proxy(
    954             tx=tx,
    955             proxy=tx.output.create_proxy(

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/variables/builder.py in wrap_fx_proxy(tx, proxy, example_value, subclass_type, **options)
   2151     }
   2152     if subclass_type is None:
-> 2153         return wrap_fx_proxy_cls(target_cls=TensorVariable, **kwargs)
   2154     else:
   2155         result = wrap_fx_proxy_cls(target_cls=TensorWithTFOverrideVariable, **kwargs)

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/variables/builder.py in wrap_fx_proxy_cls(target_cls, tx, proxy, example_value, subclass_type, **options)
   2217 ):
   2218     if example_value is None:
-> 2219         return _wrap_fx_proxy(
   2220             target_cls, tx, proxy, example_value, subclass_type, **options
   2221         )

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/variables/builder.py in _wrap_fx_proxy(target_cls, tx, proxy, example_value, subclass_type, **options)
   2313         # only allow_non_graph_fake in this instance because we handle the non-fake
   2314         # cases properly below.
-> 2315         example_value = get_fake_value(proxy.node, tx, allow_non_graph_fake=True)
   2316 
   2317     return handle_traced_output(

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/utils.py in get_fake_value(node, tx, allow_non_graph_fake)
   2534             unimplemented(f"TypeError {node.target}: {cause}")
   2535 
-> 2536         raise TorchRuntimeError(str(e)).with_traceback(e.__traceback__) from None
   2537 
   2538     if not allow_non_graph_fake:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/utils.py in get_fake_value(node, tx, allow_non_graph_fake)
   2469     try:
   2470         with tx.fake_mode, enable_python_dispatcher():
-> 2471             ret_val = wrap_fake_exception(
   2472                 lambda: run_node(tx.output, node, args, kwargs, nnmodule)
   2473             )

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/utils.py in wrap_fake_exception(fn)
   2015 def wrap_fake_exception(fn):
   2016     try:
-> 2017         return fn()
   2018     except UnsupportedFakeTensorException as e:
   2019         from .exc import unimplemented

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/utils.py in <lambda>()
   2470         with tx.fake_mode, enable_python_dispatcher():
   2471             ret_val = wrap_fake_exception(
-> 2472                 lambda: run_node(tx.output, node, args, kwargs, nnmodule)
   2473             )
   2474     except Unsupported:

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/utils.py in run_node(tracer, node, args, kwargs, nnmodule)
   2602             unimplemented(make_error_message(e), from_exc=e)
   2603         except Exception as e:
-> 2604             raise RuntimeError(make_error_message(e)).with_traceback(
   2605                 e.__traceback__
   2606             ) from e

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/utils.py in run_node(tracer, node, args, kwargs, nnmodule)
   2584         try:
   2585             if op == "call_function":
-> 2586                 return node.target(*args, **kwargs)
   2587             elif op == "call_method":
   2588                 return getattr(args[0], node.target)(*args[1:], **kwargs)

/usr/local/lib/python3.11/dist-packages/torch/utils/_stats.py in wrapper(*args, **kwargs)
     19             simple_call_counter[fn.__qualname__] = 0
     20         simple_call_counter[fn.__qualname__] = simple_call_counter[fn.__qualname__] + 1
---> 21         return fn(*args, **kwargs)
     22     return wrapper

/usr/local/lib/python3.11/dist-packages/torch/_subclasses/fake_tensor.py in __torch_dispatch__(self, func, types, args, kwargs)
   1274         ), func
   1275         try:
-> 1276             return self.dispatch(func, types, args, kwargs)
   1277         except TypeError:
   1278             log.exception("fake tensor raised TypeError")

/usr/local/lib/python3.11/dist-packages/torch/_subclasses/fake_tensor.py in dispatch(self, func, types, args, kwargs)
   1814 
   1815         if self.cache_enabled:
-> 1816             return self._cached_dispatch_impl(func, types, args, kwargs)
   1817         else:
   1818             return self._dispatch_impl(func, types, args, kwargs)

/usr/local/lib/python3.11/dist-packages/torch/_subclasses/fake_tensor.py in _cached_dispatch_impl(self, func, types, args, kwargs)
   1375             else:
   1376                 self._validate_cache_key(func, args, kwargs)
-> 1377                 output = self._dispatch_impl(func, types, args, kwargs)
   1378                 entry = self._make_cache_entry(state, key, func, args, kwargs, output)
   1379                 key.strip_shape_env()

/usr/local/lib/python3.11/dist-packages/torch/_subclasses/fake_tensor.py in _dispatch_impl(self, func, types, args, kwargs)
   2288                 with self:
   2289                     return maybe_propagate_real_tensors(
-> 2290                         decomposition_table[func](*args, **kwargs)
   2291                     )
   2292 

/usr/local/lib/python3.11/dist-packages/torch/_prims_common/wrappers.py in _fn(out, *args, **kwargs)
    289                 result = fn(*args, is_out=(out is not None), **kwargs)  # type: ignore[arg-type]
    290             else:
--> 291                 result = fn(*args, **kwargs)
    292             assert (
    293                 isinstance(result, TensorLike)

/usr/local/lib/python3.11/dist-packages/torch/_decomp/decompositions.py in inner(*args, **kwargs)
     81                 return x
     82 
---> 83         r = f(*tree_map(increase_prec, args), **tree_map(increase_prec, kwargs))
     84         if compute_dtype_only:
     85             return r

/usr/local/lib/python3.11/dist-packages/torch/_decomp/decompositions.py in addmm(self, mat1, mat2, beta, alpha)
   1448         beta = int(beta)
   1449         alpha = int(alpha)
-> 1450     out = alpha * torch.mm(mat1, mat2)
   1451     if beta == 0:
   1452         return out

/usr/local/lib/python3.11/dist-packages/torch/utils/_stats.py in wrapper(*args, **kwargs)
     19             simple_call_counter[fn.__qualname__] = 0
     20         simple_call_counter[fn.__qualname__] = simple_call_counter[fn.__qualname__] + 1
---> 21         return fn(*args, **kwargs)
     22     return wrapper

/usr/local/lib/python3.11/dist-packages/torch/_subclasses/fake_tensor.py in __torch_dispatch__(self, func, types, args, kwargs)
   1274         ), func
   1275         try:
-> 1276             return self.dispatch(func, types, args, kwargs)
   1277         except TypeError:
   1278             log.exception("fake tensor raised TypeError")

/usr/local/lib/python3.11/dist-packages/torch/_subclasses/fake_tensor.py in dispatch(self, func, types, args, kwargs)
   1814 
   1815         if self.cache_enabled:
-> 1816             return self._cached_dispatch_impl(func, types, args, kwargs)
   1817         else:
   1818             return self._dispatch_impl(func, types, args, kwargs)

/usr/local/lib/python3.11/dist-packages/torch/_subclasses/fake_tensor.py in _cached_dispatch_impl(self, func, types, args, kwargs)
   1375             else:
   1376                 self._validate_cache_key(func, args, kwargs)
-> 1377                 output = self._dispatch_impl(func, types, args, kwargs)
   1378                 entry = self._make_cache_entry(state, key, func, args, kwargs, output)
   1379                 key.strip_shape_env()

/usr/local/lib/python3.11/dist-packages/torch/_subclasses/fake_tensor.py in _dispatch_impl(self, func, types, args, kwargs)
   2382         try:
   2383             with in_kernel_invocation_manager(self):
-> 2384                 r = func(*args, **kwargs)
   2385         except NotImplementedError as not_implemented_error:
   2386             return maybe_run_unsafe_fallback(not_implemented_error)

/usr/local/lib/python3.11/dist-packages/torch/_ops.py in __call__(self, *args, **kwargs)
    721     # that are named "self". This way, all the aten ops can be called by kwargs.
    722     def __call__(self, /, *args, **kwargs):
--> 723         return self._op(*args, **kwargs)
    724 
    725     # Use positional-only argument to avoid naming collision with aten ops arguments

/usr/local/lib/python3.11/dist-packages/torch/_prims_common/wrappers.py in _fn(out, *args, **kwargs)
    289                 result = fn(*args, is_out=(out is not None), **kwargs)  # type: ignore[arg-type]
    290             else:
--> 291                 result = fn(*args, **kwargs)
    292             assert (
    293                 isinstance(result, TensorLike)

/usr/local/lib/python3.11/dist-packages/torch/_meta_registrations.py in meta_mm(a, b)
   2125     N, M1 = a.shape
   2126     M2, P = b.shape
-> 2127     torch._check(
   2128         M1 == M2,
   2129         lambda: f"a and b must have same reduction dim, but got [{N}, {M1}] X [{M2}, {P}].",

/usr/local/lib/python3.11/dist-packages/torch/__init__.py in _check(cond, message)
   1654             message. Default: ``None``
   1655     """
-> 1656     _check_with(RuntimeError, cond, message)
   1657 
   1658 

/usr/local/lib/python3.11/dist-packages/torch/__init__.py in _check_with(error_type, cond, message)
   1636         message_evaluated = str(message())
   1637 
-> 1638     raise error_type(message_evaluated)
   1639 
   1640 

TorchRuntimeError: Failed running call_function <built-in function linear>(*(FakeTensor(..., device='cuda:0', size=(1, 1)), Parameter(FakeTensor(..., device='cuda:0', size=(6, 2048), requires_grad=True)), Parameter(FakeTensor(..., device='cuda:0', size=(6,), requires_grad=True))), **{}):
a and b must have same reduction dim, but got [1, 1] X [2048, 6].

from user code:
   File "/tmp/ipykernel_55/175391509.py", line 19, in forward
    x = self.customlayer(x)
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/container.py", line 250, in forward
    input = module(input)
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)

Set TORCH_LOGS="+dynamo" and TORCHDYNAMO_VERBOSE=1 for more information


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True


## === cell 14
sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})

preds = np.asarray(predictions, dtype=np.float32)
if preds.ndim != 2 or preds.shape[0] != len(sub) or preds.shape[1] != 6:
    preds = np.full((len(sub), 6), 1.0 / 6.0, dtype=np.float32)

preds = np.clip(preds, 1e-8, 1.0)
preds = preds / preds.sum(axis=1, keepdims=True)

sub[TARGETS] = preds
sub.to_csv("submission.csv", index=False)
print(f"Submission shape: {sub.shape}")
print(sub.head())

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/532074912.py in <cell line: 0>()
      1 sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
      2 
----> 3 preds = np.asarray(predictions, dtype=np.float32)
      4 if preds.ndim != 2 or preds.shape[0] != len(sub) or preds.shape[1] != 6:
      5     preds = np.full((len(sub), 6), 1.0 / 6.0, dtype=np.float32)

NameError: name 'predictions' is not defined
