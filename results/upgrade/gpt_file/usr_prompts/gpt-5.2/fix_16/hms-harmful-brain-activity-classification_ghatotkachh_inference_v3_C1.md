# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.6538911800803626

# 6. Current score

1.13663

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the two blockers that prevent a valid `submission.csv` from being created: the missing model directory (so inference never runs) and the shape/column mismatch when writing predictions. To keep core logic intact while ensuring end-to-end execution, I add a safe fallback that produces a valid probabilistic submission (uniform distribution) when no pretrained weights are found in `/kaggle/input`. I also remove unintended debug printing and fix undefined variables (`targets` vs `TARGETS`, missing `min/max` for test) that would otherwise break dataset indexing. Finally, I enforce that each prediction row sums to 1 (required by the competition) and that the submission columns exactly match `sample_submission.csv`.'
- What this solution (achieved 1.40995) has done: 'I fix the dataset tensor shape bug that currently crashes `__getitem__` by removing the invalid reshape and instead building a proper 3-channel image tensor that matches what a ResNet expects (3×H×W), while keeping the same underlying features (spectrogram + EEG-derived spectrogram) and normalization logic. Since no pretrained weights are available in the provided `/kaggle/input` paths, I add a safe fallback that generates a valid probabilistic submission (uniform distribution) so the notebook always produces `submission.csv` end-to-end. Finally, I ensure probabilities are strictly positive, clipped, and normalized to sum to 1 per row, and that the submission columns exactly match `sample_submission.csv` ordering to avoid Kaggle format failures.'
- What this solution (achieved 1.40995) has done: 'The timeout is dominated by heavy per-sample feature extraction inside `CustomDataset.__getitem__` (parquet I/O + librosa STFT per EEG) and by inadvertently training 5 folds × 10 epochs when no external weights are found. I (1) hard-disable the training fallback so the notebook always runs inference-only (no change to inference semantics), (2) keep the exact same feature math but precompute/cache the full test features once per `eeg_id`/`spectrogram_id` and reuse them across all model weight files, and (3) reduce DataLoader overhead by returning already-built tensors and using a faster collate while keeping determinism. These changes are provably equivalent for test-time outputs (same inputs to the model) and cut runtime by eliminating redundant computation and any training. If no weights exist, it still falls back to uniform predictions as before.'
- What this solution (achieved 0.90562) has done: 'Your current score (1.40995, lower-is-better) suggests the model is effectively not using meaningful learned weights on Kaggle, most likely because `find_weight_files()` doesn’t locate any actual pretrained models in your environment and you fall back to uniform predictions. The smallest change that should materially move KL divergence toward your target is to enable a lightweight, patient-folded training fallback on a small subset of the training data using the exact same model, inputs, and loss semantics (KL on vote-distributions), then run inference with that trained checkpoint. To keep runtime under 600s and preserve core logic, this training is capped to 1 epoch, uses the same feature pipeline (cached), and only triggers when no external weights are found. Finally, submission writing remains unchanged, still enforcing positive probabilities and row-sum=1.'
- What this solution (achieved 1.40995) has done: 'Your current KL score (0.90562, lower-is-better) is still above the target (0.65389), so we should cautiously improve model quality without changing the core model/feature/loss logic. The biggest minimal win here is to make the fallback training actually learn from *all* training rows (not a tiny patient subset), while keeping the same architecture, KL loss, and single-epoch approach, and keeping runtime under 600s by caching only what’s needed. We also fix a silent correctness issue in feature extraction: your spectrogram slice uses `r` in 0..300 but the spectrogram time axis is ~600, so the correct indexing is `t0=2*r` and `t1=2*r+300` (consistent with how you derived `min/max`), which should materially improve training signal while preserving the same semantics. Finally, we keep submission normalization/clipping exactly as required.'
- What this solution (achieved 1.40995) has done: 'We fix the training fallback crash by making cached `.npy` loads robust to partially-written/corrupted files (the `mmap length is greater than file size` error) by retrying a normal load and recomputing the cache when needed. This is a correctness/stability fix that preserves the exact feature logic while allowing the 1-epoch fallback training to complete, which should improve KL from the uniform/near-uniform behavior toward your target. We also make cache writes atomic (write to temp then `os.replace`) to prevent future partial files when multiple DataLoader workers compute the same cache entry. Finally, we keep the submission formatting/normalization unchanged to ensure a valid `submission.csv` is always produced.'
- What this solution (achieved 1.13663) has done: 'I fix the cache writer that currently crashes with `FileNotFoundError` by making the `.npy` atomic save function compatible with how `np.save` appends the `.npy` suffix, ensuring the temp file actually exists before `os.replace`. This is a pure runtime/stability bugfix that preserves feature/model logic and unblocks both test feature caching and the 1-epoch training fallback, which should improve KL from your current 1.40995 toward the target band. I also add a defensive `os.makedirs(..., exist_ok=True)` inside the saver so directory creation can’t race under multiprocessing. No other modeling/training semantics are changed; submission formatting and row-sum normalization remain as-is.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

import librosa
import albumentations as A
import gc
import matplotlib.pyplot as plt
import math
import multiprocessing as mp
import random
import time
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from albumentations.pytorch import ToTensorV2
from glob import glob
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Dict, List, Tuple


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using", torch.cuda.device_count(), "GPU(s); device:", device)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True




## === cell 1
class config:
    model = "resnet18d"
    epoch = 10
    lr = 1e-3
    batchsize = 32
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


model_path = os.path.join(paths.out, "trained_model.pth")



## === cell 2
USE_WAVELET = None

NAMES = ["LL", "LP", "RP", "RR"]

FEATS = [
    ["Fp1", "F7", "T3", "T5", "O1"],
    ["Fp1", "F3", "C3", "P3", "O1"],
    ["Fp2", "F8", "T4", "T6", "O2"],
    ["Fp2", "F4", "C4", "P4", "O2"],
]

EEG_NEEDED_COLS = sorted(set(sum(FEATS, [])))


_SR = 200
_N_FFT = 1024
_WIN_LENGTH = 128
_N_MELS = 128
_FMIN = 0
_FMAX = 20

_MEL_BASIS = librosa.filters.mel(
    sr=_SR, n_fft=_N_FFT, n_mels=_N_MELS, fmin=_FMIN, fmax=_FMAX
).astype(np.float32)


def _mel_spec_db_equiv(x: np.ndarray, hop_length: int) -> np.ndarray:
    D = librosa.stft(
        y=x,
        n_fft=_N_FFT,
        hop_length=hop_length,
        win_length=_WIN_LENGTH,
        center=True,
        window="hann",
        pad_mode="constant",
    )
    S = (np.abs(D) ** 2).astype(np.float32, copy=False)  # power spectrogram
    M = _MEL_BASIS @ S  # mel power
    mel_spec_db = librosa.power_to_db(M, ref=np.max).astype(np.float32, copy=False)
    return mel_spec_db


def spectrogram_from_eeg(parquet_path, display=False):
    eeg = pd.read_parquet(parquet_path, columns=EEG_NEEDED_COLS)
    middle = (len(eeg) - 10_000) // 2
    eeg = eeg.iloc[middle : middle + 10_000]

    img = np.zeros((128, 256, 4), dtype="float32")

    if display:
        plt.figure(figsize=(10, 7))
    signals = []
    hop = 10_000 // 256

    eeg_np = {c: eeg[c].to_numpy(copy=False) for c in EEG_NEEDED_COLS}

    for k in range(4):
        COLS = FEATS[k]

        for kk in range(4):
            x = eeg_np[COLS[kk]] - eeg_np[COLS[kk + 1]]
            x = np.asarray(x, dtype=np.float32)

            m = np.nanmean(x)
            if np.isnan(x).mean() < 1:
                x = np.nan_to_num(x, nan=m)
            else:
                x[:] = 0

            signals.append(x)

            mel_spec_db = _mel_spec_db_equiv(x, hop_length=hop)
            width = (mel_spec_db.shape[1] // 32) * 32
            mel_spec_db = mel_spec_db[:, :width]

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
df = pd.read_csv(paths.test_csv)
df.head()



## === cell 4
CACHE_DIR = os.path.join(paths.out, "cache_hms")
os.makedirs(CACHE_DIR, exist_ok=True)

EEG_CACHE_DIR = os.path.join(CACHE_DIR, "eeg_specs_per_file")
SPEC_CACHE_DIR = os.path.join(CACHE_DIR, "spectrograms_per_file")
SPEC_PROC_CACHE_DIR = os.path.join(CACHE_DIR, "spectrograms_processed_100x256x4")
TEST_X_CACHE_DIR = os.path.join(CACHE_DIR, "test_model_input_3x128x256_per_row")
TRAIN_X_CACHE_DIR = os.path.join(
    CACHE_DIR, "train_model_input_3x128x256_per_row"
)  # training cache
os.makedirs(EEG_CACHE_DIR, exist_ok=True)
os.makedirs(SPEC_CACHE_DIR, exist_ok=True)
os.makedirs(SPEC_PROC_CACHE_DIR, exist_ok=True)
os.makedirs(TEST_X_CACHE_DIR, exist_ok=True)
os.makedirs(TRAIN_X_CACHE_DIR, exist_ok=True)


def _eeg_cache_file(eeg_id: int) -> str:
    return os.path.join(EEG_CACHE_DIR, f"eeg_{int(eeg_id)}.npy")


def _spec_cache_file(spec_id: int) -> str:
    return os.path.join(SPEC_CACHE_DIR, f"spec_{int(spec_id)}.npy")


def _spec_proc_cache_file(spec_id: int) -> str:
    return os.path.join(SPEC_PROC_CACHE_DIR, f"specproc_{int(spec_id)}.npy")


def _test_x_cache_file(eeg_id: int, spec_id: int) -> str:
    return os.path.join(TEST_X_CACHE_DIR, f"x_e{int(eeg_id)}_s{int(spec_id)}.npy")


def _train_x_cache_file(eeg_id: int, spec_id: int, r: int) -> str:
    return os.path.join(
        TRAIN_X_CACHE_DIR, f"x_e{int(eeg_id)}_s{int(spec_id)}_r{int(r)}.npy"
    )


def _atomic_save_npy(path: str, arr: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_base = f"{path}.tmp_{os.getpid()}_{time.time_ns()}"
    np.save(tmp_base, arr)
    tmp_path = tmp_base + ".npy"
    os.replace(tmp_path, path)


def _robust_load_npy(path: str) -> np.ndarray:
    try:
        return np.load(path, allow_pickle=False, mmap_mode="r")
    except ValueError:
        return np.load(path, allow_pickle=False)
    except Exception:
        raise


def get_eeg_spec(eeg_id: int, eeg_dir: str) -> np.ndarray:
    """Load EEG-derived spectrogram (128,256,4) from per-file cache or compute once."""
    fn = _eeg_cache_file(eeg_id)
    if os.path.isfile(fn):
        try:
            return _robust_load_npy(fn)
        except Exception:
            try:
                os.remove(fn)
            except OSError:
                pass

    parquet_path = os.path.join(eeg_dir, f"{int(eeg_id)}.parquet")
    arr = np.asarray(spectrogram_from_eeg(parquet_path), dtype=np.float32)
    _atomic_save_npy(fn, arr)
    return arr


def get_spectrogram(spec_id: int, spec_dir: str) -> np.ndarray:
    """Load spectrogram parquet array from per-file cache or read once."""
    fn = _spec_cache_file(spec_id)
    if os.path.isfile(fn):
        try:
            return _robust_load_npy(fn)
        except Exception:
            try:
                os.remove(fn)
            except OSError:
                pass

    parquet_path = os.path.join(spec_dir, f"{int(spec_id)}.parquet")
    df = pd.read_parquet(parquet_path)
    arr = np.asarray(df.to_numpy(copy=False))
    _atomic_save_npy(fn, arr)
    return arr


def _pad_or_crop_2d(a: np.ndarray, out_h: int, out_w: int, pad_value: float = 0.0):
    a = np.asarray(a)
    h, w = a.shape
    oh, ow = out_h, out_w
    out = np.full((oh, ow), pad_value, dtype=a.dtype)
    hh = min(h, oh)
    ww = min(w, ow)
    out[:hh, :ww] = a[:hh, :ww]
    return out


def preprocess_spec_regions_from_full_spec(sp_full: np.ndarray) -> np.ndarray:
    """
    Returns float32 array shaped (100,256,4) corresponding to X[14:-14,:,0:4] content
    when r=0. Keeps operations identical to the original pipeline.
    """
    out = np.zeros((100, 256, 4), dtype=np.float32)
    t0, t1 = (0, 300)  # r=0 for test
    for region in range(4):
        f0, f1 = region * 100, (region + 1) * 100
        img = sp_full[t0:t1, f0:f1].T  # (100,300)
        if img.ndim != 2:
            img = np.zeros((100, 300), dtype=np.float32)
        img = _pad_or_crop_2d(img.astype(np.float32, copy=False), 100, 300, 0.0)

        img = np.clip(img, np.exp(-4), np.exp(8))
        img = np.log(img)

        ep = 1e-6
        mu = np.nanmean(img)
        std = np.nanstd(img)
        img = (img - mu) / (std + ep)
        img = np.nan_to_num(img, nan=0.0)

        img_c = img[:, 22:-22]
        img_c = _pad_or_crop_2d(img_c, 100, 256, 0.0)

        out[:, :, region] = img_c.astype(np.float32, copy=False) / 2.0
    return out


class LazyFeatureStore:
    """Small in-memory memoization + persistent .npy cache on disk."""

    def __init__(self, eeg_dir: str, spec_dir: str, max_mem_items: int = 512):
        self.eeg_dir = eeg_dir
        self.spec_dir = spec_dir
        self.max_mem_items = max_mem_items
        self._eeg_mem: Dict[int, np.ndarray] = {}
        self._spec_mem: Dict[int, np.ndarray] = {}
        self._specproc_mem: Dict[int, np.ndarray] = {}
        self._eeg_keys: List[int] = []
        self._spec_keys: List[int] = []
        self._specproc_keys: List[int] = []

    def _memo_put(self, dct, keys, k, v):
        if k in dct:
            return
        dct[k] = v
        keys.append(k)
        if len(keys) > self.max_mem_items:
            old = keys.pop(0)
            dct.pop(old, None)

    def eeg(self, eeg_id: int) -> np.ndarray:
        eeg_id = int(eeg_id)
        v = self._eeg_mem.get(eeg_id)
        if v is not None:
            return v
        v = get_eeg_spec(eeg_id, self.eeg_dir)
        self._memo_put(self._eeg_mem, self._eeg_keys, eeg_id, v)
        return v

    def spec(self, spec_id: int) -> np.ndarray:
        spec_id = int(spec_id)
        v = self._spec_mem.get(spec_id)
        if v is not None:
            return v
        v = get_spectrogram(spec_id, self.spec_dir)
        self._memo_put(self._spec_mem, self._spec_keys, spec_id, v)
        return v

    def spec_processed_r0(self, spec_id: int) -> np.ndarray:
        spec_id = int(spec_id)
        v = self._specproc_mem.get(spec_id)
        if v is not None:
            return v
        fn = _spec_proc_cache_file(spec_id)
        if os.path.isfile(fn):
            try:
                v = _robust_load_npy(fn)
                self._memo_put(self._specproc_mem, self._specproc_keys, spec_id, v)
                return v
            except Exception:
                try:
                    os.remove(fn)
                except OSError:
                    pass

        sp = self.spec(spec_id)
        v = preprocess_spec_regions_from_full_spec(sp)
        _atomic_save_npy(fn, v)
        self._memo_put(self._specproc_mem, self._specproc_keys, spec_id, v)
        return v


test_store = LazyFeatureStore(paths.test_eeg, paths.test_spec, max_mem_items=256)



## === cell 5
test_df = pd.read_csv(paths.test_csv)
print(f"Test dataframe shape is: {test_df.shape}")
test_df.head()



## === cell 6
TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]


class CustomDataset:
    def __init__(
        self,
        traindf: pd.DataFrame,
        config,
        mode: str = "train",
        store: LazyFeatureStore | None = None,
        eeg_dir: str | None = None,
        spec_dir: str | None = None,
    ):
        self.traindf = traindf.reset_index(drop=True)
        self.mode = mode

        self.eeg_ids = self.traindf["eeg_id"].to_numpy(dtype=np.int64, copy=False)
        self.spec_ids = self.traindf["spectrogram_id"].to_numpy(
            dtype=np.int64, copy=False
        )

        if ("min" in self.traindf.columns) and ("max" in self.traindf.columns):
            self.r_arr = (
                (self.traindf["min"].to_numpy() + self.traindf["max"].to_numpy()) // 4
            ).astype(np.int64)
        else:
            self.r_arr = None

        if self.mode != "test":
            self.targets = self.traindf[TARGETS].to_numpy(dtype=np.float32, copy=False)
        else:
            self.targets = None

        if store is None:
            if eeg_dir is None or spec_dir is None:
                raise ValueError(
                    "Provide either store=LazyFeatureStore or (eeg_dir, spec_dir)."
                )
            store = LazyFeatureStore(eeg_dir, spec_dir, max_mem_items=256)
        self.store = store

    def __len__(self):
        return len(self.traindf)

    def __getitem__(self, idx):
        eeg_id = int(self.eeg_ids[idx])
        spec_id = int(self.spec_ids[idx])

        if self.mode == "test":
            cache_fn = _test_x_cache_file(eeg_id, spec_id)
            if os.path.isfile(cache_fn):
                try:
                    x_np = _robust_load_npy(cache_fn)
                    x = torch.from_numpy(np.asarray(x_np, dtype=np.float32))
                    y = np.zeros(6, dtype="float32")
                    return {"data": x, "target": y}
                except Exception:
                    try:
                        os.remove(cache_fn)
                    except OSError:
                        pass

        X = np.zeros((128, 256, 8), dtype="float32")

        if self.mode == "test":
            r = 0
        else:
            r = int(self.r_arr[idx]) if self.r_arr is not None else 0

        if self.mode != "test":
            cache_fn_tr = _train_x_cache_file(eeg_id, spec_id, r)
            if os.path.isfile(cache_fn_tr):
                try:
                    x_np = _robust_load_npy(cache_fn_tr)
                    x = torch.from_numpy(np.asarray(x_np, dtype=np.float32))
                    y = self.targets[idx]
                    return {"data": x, "target": y}
                except Exception:
                    try:
                        os.remove(cache_fn_tr)
                    except OSError:
                        pass

        if r == 0:
            spec_proc = self.store.spec_processed_r0(spec_id)  # (100,256,4)
            X[14:-14, :, 0:4] = np.asarray(spec_proc, dtype=np.float32)
        else:
            sp = self.store.spec(spec_id)  # numpy array (time x features)
            for region in range(4):
                t0, t1 = 2 * r, 2 * r + 300
                f0, f1 = region * 100, (region + 1) * 100

                img = sp[t0:t1, f0:f1].T  # expected (100,300) before cropping

                if img.ndim != 2:
                    img = np.zeros((100, 300), dtype=np.float32)
                img = _pad_or_crop_2d(img.astype(np.float32, copy=False), 100, 300, 0.0)

                img = np.clip(img, np.exp(-4), np.exp(8))
                img = np.log(img)

                ep = 1e-6
                mu = np.nanmean(img)
                std = np.nanstd(img)
                img = (img - mu) / (std + ep)
                img = np.nan_to_num(img, nan=0.0)

                img_c = img[:, 22:-22]  # (100,256) if width==300
                img_c = _pad_or_crop_2d(img_c, 100, 256, 0.0)

                X[14:-14, :, region] = img_c / 2.0

        eeg_img = self.store.eeg(eeg_id)  # (128,256,4)
        X[:, :, 4:] = np.asarray(eeg_img, dtype=np.float32)

        Xt = torch.from_numpy(X)  # (128,256,8)
        spect = Xt[:, :, 0:4].permute(2, 0, 1)  # (4,128,256)
        eegs = Xt[:, :, 4:8].permute(2, 0, 1)  # (4,128,256)

        spect_m = spect.mean(dim=0, keepdim=True)  # (1,128,256)
        eeg_m = eegs.mean(dim=0, keepdim=True)  # (1,128,256)
        diff = spect_m - eeg_m  # (1,128,256)
        x = torch.cat([spect_m, eeg_m, diff], dim=0).to(torch.float32)  # (3,128,256)

        if self.mode != "test":
            y = self.targets[idx]
            _atomic_save_npy(_train_x_cache_file(eeg_id, spec_id, r), x.numpy())
        else:
            y = np.zeros(6, dtype="float32")
            _atomic_save_npy(_test_x_cache_file(eeg_id, spec_id), x.numpy())

        return {"data": x, "target": y}




## === cell 7
customdataset = CustomDataset(test_df, config, mode="test", store=test_store)
sample_item = customdataset[0]
print(sample_item["data"].shape, sample_item["target"].shape)



## === cell 8
from torch.utils.data import DataLoader


def fast_collate(batch):
    data = torch.stack([b["data"] for b in batch], dim=0)
    target = torch.from_numpy(np.stack([b["target"] for b in batch], axis=0))
    return {"data": data, "target": target}


_loader_workers = min(8, max(0, mp.cpu_count() - 1))
test_loader = DataLoader(
    customdataset,
    batch_size=config.batchsize,
    shuffle=False,
    num_workers=_loader_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(_loader_workers > 0),
    prefetch_factor=4 if _loader_workers > 0 else None,
    collate_fn=fast_collate,
)
X0 = customdataset[0]["data"]
y0 = customdataset[0]["target"]
print(X0.shape, y0)




## === cell 9
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




## === cell 10
def inference_function(test_loader, model, device):
    model.eval()
    softmax = nn.Softmax(dim=1)
    preds = []
    use_amp = torch.cuda.is_available() and bool(getattr(config, "AMP", False))
    for batch in tqdm(test_loader, desc="Inference"):
        x = batch["data"].to(device, non_blocking=True)
        with torch.no_grad():
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    ypred = model(x)
            else:
                ypred = model(x)
            ypred = softmax(ypred)
        preds.append(ypred.detach().cpu().numpy())
    return {"predictions": np.concatenate(preds, axis=0)}




## === cell 11
def add_min_max_offsets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    r = np.clip(
        (df["spectrogram_label_offset_seconds"].to_numpy(np.int64) // 2), 0, 300
    )
    df["min"] = (4 * r).astype(np.int64)
    df["max"] = (4 * r).astype(np.int64)
    return df


def make_patient_folds(
    train_df: pd.DataFrame, n_folds: int = 5, seed: int = 42
) -> pd.DataFrame:
    df = train_df.copy()
    patients = df["patient_id"].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(patients)
    fold_map = {p: i % n_folds for i, p in enumerate(patients)}
    df["fold"] = df["patient_id"].map(fold_map).astype(np.int64)
    return df


def targets_to_prob(y_votes: torch.Tensor) -> torch.Tensor:
    y = y_votes.to(torch.float32)
    y = y / (y.sum(dim=1, keepdim=True) + 1e-6)
    y = torch.clamp(y, 1e-6, 1.0)
    y = y / y.sum(dim=1, keepdim=True)
    return y


def find_weight_files():
    candidate_dirs = [
        "/kaggle/input/10ep5foldresent18",
        "/kaggle/input/resent18models1ep",
    ]
    weight_files = []
    for d in candidate_dirs:
        if os.path.isdir(d):
            for fn in os.listdir(d):
                if fn.endswith((".pth", ".pt", ".bin")):
                    weight_files.append(os.path.join(d, fn))
    return sorted(weight_files)


weight_files = find_weight_files()
print(f"Found {len(weight_files)} candidate weight files in /kaggle/input.")

if len(weight_files) == 0 and os.path.isfile(model_path):
    weight_files = [model_path]
    print(f"Using local model_path weights: {model_path}")

ENABLE_TRAIN_FALLBACK = len(weight_files) == 0




## === cell 12
def kl_divergence_loss_from_logits(
    logits: torch.Tensor, target_votes: torch.Tensor
) -> torch.Tensor:
    p = F.log_softmax(logits, dim=1)  # log q
    t = targets_to_prob(target_votes)  # p (target distribution)
    return F.kl_div(p, t, reduction="batchmean")


def train_one_epoch(train_loader, model, optimizer, device):
    model.train()
    use_amp = torch.cuda.is_available() and bool(getattr(config, "AMP", False))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    running = 0.0
    n = 0
    for batch in tqdm(train_loader, desc="Train(1 epoch)"):
        x = batch["data"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
                loss = kl_divergence_loss_from_logits(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = kl_divergence_loss_from_logits(logits, y)
            loss.backward()
            optimizer.step()

        bs = x.size(0)
        running += float(loss.detach().cpu()) * bs
        n += bs
    return running / max(n, 1)


def maybe_train_fallback_and_get_weight_file() -> str | None:
    if not ENABLE_TRAIN_FALLBACK:
        return None

    t0 = time.time()
    train_df = pd.read_csv(paths.train_csv)

    train_df = add_min_max_offsets(train_df)
    train_df = make_patient_folds(train_df, n_folds=config.FOLDS, seed=42)

    tr_df = train_df[train_df["fold"] != 0].reset_index(drop=True)
    if len(tr_df) < 1000:
        tr_df = train_df.reset_index(drop=True)

    train_store = LazyFeatureStore(
        paths.train_eeg_dir, paths.train_spec_dir, max_mem_items=128
    )
    tr_ds = CustomDataset(tr_df, config, mode="train", store=train_store)

    cache_n = min(8000, len(tr_ds))
    for i in tqdm(range(cache_n), desc="Caching train features (moderate subset)"):
        _ = tr_ds[i]

    _wk = min(4, max(0, mp.cpu_count() - 1))
    tr_loader = DataLoader(
        tr_ds,
        batch_size=config.batchsize,
        shuffle=True,
        num_workers=_wk,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(_wk > 0),
        prefetch_factor=2 if _wk > 0 else None,
        collate_fn=fast_collate,
        drop_last=False,
    )

    model = Custommodel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.WEIGHT_DECAY
    )

    loss = train_one_epoch(tr_loader, model, optimizer, device)
    print(f"Fallback training loss (KL) after 1 epoch: {loss:.6f}")

    ckpt = {"model": model.state_dict()}
    torch.save(ckpt, model_path)
    print(
        f"Saved fallback trained weights to: {model_path} (elapsed {time.time()-t0:.1f}s)"
    )

    del model, optimizer, tr_loader, tr_ds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model_path


trained_path = maybe_train_fallback_and_get_weight_file()
if trained_path is not None:
    weight_files = [trained_path]

if len(weight_files) == 0:
    print(
        "No usable weights found and training fallback did not run; will use uniform predictions."
    )



## === cell 13
predictions = None

if len(weight_files) == 0:
    n = len(test_df)
    predictions = np.full((n, 6), 1.0 / 6.0, dtype=np.float32)
    print("No usable weights found; using uniform fallback predictions.")
else:
    fold_preds = []

    testdataset = CustomDataset(test_df, config, mode="test", store=test_store)
    for i in tqdm(range(len(testdataset)), desc="Caching test features"):
        _ = testdataset[i]

    _inf_workers = min(8, max(0, mp.cpu_count() - 1))
    testloader = DataLoader(
        testdataset,
        batch_size=config.batchsize,
        shuffle=False,
        num_workers=_inf_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(_inf_workers > 0),
        prefetch_factor=4 if _inf_workers > 0 else None,
        collate_fn=fast_collate,
    )

    for wf in weight_files:
        dd = torch.load(wf, map_location="cpu")
        model = Custommodel(config)

        state = dd["model"] if isinstance(dd, dict) and ("model" in dd) else dd
        model.load_state_dict(state, strict=True)

        model.to(device)
        pred_dict = inference_function(testloader, model, device)
        fold_preds.append(pred_dict["predictions"])

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    predictions = np.mean(np.stack(fold_preds, axis=0), axis=0)
    print("Predictions computed from weights; shape:", predictions.shape)



## === cell 14
predictions = np.asarray(predictions, dtype=np.float32)
if predictions.ndim != 2 or predictions.shape[1] != 6:
    raise ValueError(f"predictions must have shape (N,6), got {predictions.shape}")

predictions = np.clip(predictions, 1e-6, 1.0)
predictions = predictions / predictions.sum(axis=1, keepdims=True)

print(
    "Row sum min/max:",
    float(predictions.sum(axis=1).min()),
    float(predictions.sum(axis=1).max()),
)

sample_sub_path = (
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
)
sample_sub = pd.read_csv(sample_sub_path)
sub = sample_sub.copy()

if len(sub) != len(predictions):
    raise ValueError(
        f"Submission rows {len(sub)} != predictions rows {len(predictions)}"
    )

sub[TARGETS] = predictions.astype(np.float32)

out_path = os.path.join(paths.out, "submission.csv")
sub.to_csv(out_path, index=False)
print(f"Saved submission to: {out_path}")
print(f"Submission shape: {sub.shape}")
print(sub.head())
