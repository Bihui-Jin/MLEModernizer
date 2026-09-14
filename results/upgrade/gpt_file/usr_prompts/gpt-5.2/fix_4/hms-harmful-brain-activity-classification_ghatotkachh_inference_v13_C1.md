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
from typing import Dict, List

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

            mel_spec = librosa.feature.melspectrogram(
                y=x,
                sr=200,
                hop_length=len(x) // 256,
                n_fft=1024,
                n_mels=128,
                fmin=0,
                fmax=20,
                win_length=128,
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
needed_train_eeg_ids = (
    pd.concat([train_df["eeg_id"], val_df["eeg_id"]]).astype(int).unique().tolist()
)
needed_train_spec_ids = (
    pd.concat([train_df["spectrogram_id"], val_df["spectrogram_id"]])
    .astype(int)
    .unique()
    .tolist()
)




## === cell 5
def _eeg_worker(args):
    eeg_id, parquet_path = args
    sp = spectrogram_from_eeg(parquet_path)
    return int(eeg_id), np.array(sp, dtype=np.float32)


def build_eeg_cache(eeg_dir: str, eeg_ids: list[int], cache_path: str):
    eeg_ids_set = set(map(int, eeg_ids))
    all_eegs_local = {}
    if os.path.exists(cache_path):
        all_eegs_local = np.load(cache_path, allow_pickle=True).item()
        missing = [i for i in eeg_ids_set if i not in all_eegs_local]
    else:
        missing = list(eeg_ids_set)

    if len(missing) > 0:
        fn_map = {
            int(fn.split(".")[0]): fn
            for fn in os.listdir(eeg_dir)
            if fn.endswith(".parquet")
        }
        tasks = []
        for eeg_id in missing:
            fn = fn_map.get(int(eeg_id), None)
            if fn is None:
                continue
            tasks.append((int(eeg_id), os.path.join(eeg_dir, fn)))

        n_workers = min(max(1, (os.cpu_count() or 2) // 2), 8)
        if n_workers <= 1 or len(tasks) <= 4:
            for t in tqdm(
                tasks, desc=f"Building EEG mels ({os.path.basename(eeg_dir)})"
            ):
                k, v = _eeg_worker(t)
                all_eegs_local[k] = v
        else:
            with multiprocessing.get_context("spawn").Pool(processes=n_workers) as pool:
                for k, v in tqdm(
                    pool.imap_unordered(_eeg_worker, tasks, chunksize=4),
                    total=len(tasks),
                    desc=f"Building EEG mels ({os.path.basename(eeg_dir)})",
                ):
                    all_eegs_local[k] = v

        np.save(cache_path, all_eegs_local, allow_pickle=True)
    return all_eegs_local


test_eeg_cache_path = os.path.join(paths.out, "all_test_eeg_specs.npy")
train_eeg_cache_path = os.path.join(paths.out, "all_train_eeg_specs.npy")

all_eegs_test = build_eeg_cache(
    paths.test_eeg, needed_test_eeg_ids, test_eeg_cache_path
)
all_eegs_train = build_eeg_cache(
    paths.train_eeg_dir, needed_train_eeg_ids, train_eeg_cache_path
)




## === cell 6
def _spec_to_img(spec: np.ndarray) -> np.ndarray:
    Xspec = np.zeros((128, 256, 4), dtype=np.float32)
    n_time = spec.shape[0]
    win = 300
    r = max(0, (n_time - win) // 2)
    r = min(r, max(0, n_time - win))

    ep = 1e-6
    for region in range(4):
        c0, c1 = region * 100, (region + 1) * 100
        img = spec[r : r + win, c0:c1].T  # (100,300)
        img = np.clip(img, np.exp(-4), np.exp(8))
        img = np.log(img)
        mu = np.nanmean(img)
        std = np.nanstd(img)
        img = (img - mu) / (std + ep)
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        Xspec[14:-14, :, region] = img[:, 22:-22] / 2.0
    return Xspec


def build_spec_cache(spec_dir: str, spec_ids: list[int], cache_path: str):
    spec_ids_set = set(map(int, spec_ids))
    all_specs_local = {}
    if os.path.exists(cache_path):
        all_specs_local = np.load(cache_path, allow_pickle=True).item()
        missing = [i for i in spec_ids_set if i not in all_specs_local]
    else:
        missing = list(spec_ids_set)

    if len(missing) > 0:
        fn_map = {
            int(fn.split(".")[0]): fn
            for fn in os.listdir(spec_dir)
            if fn.endswith(".parquet")
        }
        for sid in tqdm(
            missing, desc=f"Loading spectrograms ({os.path.basename(spec_dir)})"
        ):
            fn = fn_map.get(int(sid), None)
            if fn is None:
                continue
            sp = pd.read_parquet(os.path.join(spec_dir, fn))
            all_specs_local[int(sid)] = sp.values.astype(np.float32, copy=False)
        np.save(cache_path, all_specs_local, allow_pickle=True)
    return all_specs_local


def build_spec_img_cache(spec_dir: str, spec_ids: list[int], cache_path: str):
    spec_ids_set = set(map(int, spec_ids))
    all_imgs_local = {}
    if os.path.exists(cache_path):
        all_imgs_local = np.load(cache_path, allow_pickle=True).item()
        missing = [i for i in spec_ids_set if i not in all_imgs_local]
    else:
        missing = list(spec_ids_set)

    if len(missing) > 0:
        fn_map = {
            int(fn.split(".")[0]): fn
            for fn in os.listdir(spec_dir)
            if fn.endswith(".parquet")
        }
        for sid in tqdm(
            missing, desc=f"Precomputing spec images ({os.path.basename(spec_dir)})"
        ):
            fn = fn_map.get(int(sid), None)
            if fn is None:
                continue
            sp = pd.read_parquet(os.path.join(spec_dir, fn)).values.astype(
                np.float32, copy=False
            )
            all_imgs_local[int(sid)] = _spec_to_img(sp)
        np.save(cache_path, all_imgs_local, allow_pickle=True)
    return all_imgs_local


test_spec_cache_path = os.path.join(paths.out, "all_test_specs.npy")
train_spec_cache_path = os.path.join(paths.out, "all_train_specs.npy")

all_spectrograms_test = build_spec_cache(
    paths.test_spec, needed_test_spec_ids, test_spec_cache_path
)
all_spectrograms_train = build_spec_cache(
    paths.train_spec_dir, needed_train_spec_ids, train_spec_cache_path
)

test_spec_img_cache_path = os.path.join(paths.out, "all_test_spec_imgs.npy")
train_spec_img_cache_path = os.path.join(paths.out, "all_train_spec_imgs.npy")
all_specimgs_test = build_spec_img_cache(
    paths.test_spec, needed_test_spec_ids, test_spec_img_cache_path
)
all_specimgs_train = build_spec_img_cache(
    paths.train_spec_dir, needed_train_spec_ids, train_spec_img_cache_path
)




## === cell 7
class CustomDataset:
    def __init__(
        self,
        traindf: pd.DataFrame,
        config,
        mode: str = "train",
        specs: dict[int, np.ndarray] = None,  # raw specs (kept for compatibility)
        eegs: dict[int, np.ndarray] = None,
        specimgs: dict[int, np.ndarray] = None,  # precomputed (128,256,4)
    ):
        self.traindf = traindf
        self.specs = specs
        self.specimgs = specimgs
        self.eeg = eegs
        self.mode = mode

    def __len__(self):
        return len(self.traindf)

    def __getitem__(self, idx):
        row = self.traindf.iloc[idx]

        X = np.zeros((128, 256, 8), dtype="float32")

        sid = int(row.spectrogram_id)
        if self.specimgs is not None and sid in self.specimgs:
            X[:, :, :4] = self.specimgs[sid]
        else:
            spec = self.specs[sid]
            X[:, :, :4] = _spec_to_img(spec)

        eeg_img = self.eeg[int(row.eeg_id)]
        X[:, :, 4:] = eeg_img

        X = torch.from_numpy(X)  # (128,256,8)
        spectograms = torch.cat(
            [X[:, :, i : i + 1] for i in range(4)], dim=0
        )  # (512,256,1)
        eegs = torch.cat(
            [X[:, :, i : i + 1] for i in range(4, 8)], dim=0
        )  # (512,256,1)

        x = torch.cat([spectograms, eegs], dim=1)  # (512,512,1)
        x = torch.cat([x, x, x], dim=2)  # (512,512,3)
        x = x.permute(2, 0, 1).contiguous()  # (3,512,512)

        y = np.zeros(6, dtype="float32")
        if self.mode != "test" and all(t in self.traindf.columns for t in TARGETS):
            y = row[TARGETS].values.astype(np.float32)

        return {"data": x.float(), "target": torch.from_numpy(y).float()}




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
