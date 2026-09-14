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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os



## === cell 1
import librosa
import albumentations as A
import gc
import matplotlib.pyplot as plt
import math
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
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




## === cell 2
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




## === cell 3
USE_WAVELET = None

NAMES = ["LL", "LP", "RP", "RR"]

FEATS = [
    ["Fp1", "F7", "T3", "T5", "O1"],
    ["Fp1", "F3", "C3", "P3", "O1"],
    ["Fp2", "F8", "T4", "T6", "O2"],
    ["Fp2", "F4", "C4", "P4", "O2"],
]

EEG_NEEDED_COLS = sorted(set(sum(FEATS, [])))


def spectrogram_from_eeg(parquet_path, display=False):
    eeg = pd.read_parquet(parquet_path, columns=EEG_NEEDED_COLS)
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
        plt.title("Signals")
        plt.show()
        print()
        print("#" * 25)
        print()

    return img




## === cell 4
df = pd.read_csv(paths.test_csv)
df.head()



## === cell 5
CACHE_DIR = os.path.join(paths.out, "cache_hms")
os.makedirs(CACHE_DIR, exist_ok=True)

EEG_CACHE_DIR = os.path.join(CACHE_DIR, "eeg_specs_per_file")
SPEC_CACHE_DIR = os.path.join(CACHE_DIR, "spectrograms_per_file")
os.makedirs(EEG_CACHE_DIR, exist_ok=True)
os.makedirs(SPEC_CACHE_DIR, exist_ok=True)


def _eeg_cache_file(eeg_id: int) -> str:
    return os.path.join(EEG_CACHE_DIR, f"eeg_{int(eeg_id)}.npy")


def _spec_cache_file(spec_id: int) -> str:
    return os.path.join(SPEC_CACHE_DIR, f"spec_{int(spec_id)}.npy")


def get_eeg_spec(eeg_id: int, eeg_dir: str) -> np.ndarray:
    """Load EEG-derived spectrogram (128,256,4) from per-file cache or compute once."""
    fn = _eeg_cache_file(eeg_id)
    if os.path.isfile(fn):
        return np.load(fn, allow_pickle=False)
    parquet_path = os.path.join(eeg_dir, f"{int(eeg_id)}.parquet")
    arr = np.asarray(spectrogram_from_eeg(parquet_path), dtype=np.float32)
    np.save(fn, arr)
    return arr


def get_spectrogram(spec_id: int, spec_dir: str) -> np.ndarray:
    """Load spectrogram parquet array from per-file cache or read once."""
    fn = _spec_cache_file(spec_id)
    if os.path.isfile(fn):
        return np.load(fn, allow_pickle=False)
    parquet_path = os.path.join(spec_dir, f"{int(spec_id)}.parquet")
    arr = np.asarray(pd.read_parquet(parquet_path))
    np.save(fn, arr)
    return arr


class LazyFeatureStore:
    """Small in-memory memoization + persistent .npy cache on disk."""

    def __init__(self, eeg_dir: str, spec_dir: str, max_mem_items: int = 512):
        self.eeg_dir = eeg_dir
        self.spec_dir = spec_dir
        self.max_mem_items = max_mem_items
        self._eeg_mem: Dict[int, np.ndarray] = {}
        self._spec_mem: Dict[int, np.ndarray] = {}
        self._eeg_keys: List[int] = []
        self._spec_keys: List[int] = []

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


test_store = LazyFeatureStore(paths.test_eeg, paths.test_spec, max_mem_items=256)


def _precompute_one(args):
    kind, _id, _dir = args
    if kind == "eeg":
        _ = get_eeg_spec(_id, _dir)
    else:
        _ = get_spectrogram(_id, _dir)
    return 1


def precompute_test_caches(
    test_df: pd.DataFrame, eeg_dir: str, spec_dir: str, workers: int
):
    eeg_ids = test_df["eeg_id"].to_numpy(dtype=np.int64, copy=False)
    spec_ids = test_df["spectrogram_id"].to_numpy(dtype=np.int64, copy=False)

    eeg_unique = np.unique(eeg_ids)
    spec_unique = np.unique(spec_ids)

    eeg_todo = [
        int(x) for x in eeg_unique if not os.path.isfile(_eeg_cache_file(int(x)))
    ]
    spec_todo = [
        int(x) for x in spec_unique if not os.path.isfile(_spec_cache_file(int(x)))
    ]

    print(
        f"Cache warmup needed: EEG {len(eeg_todo)}/{len(eeg_unique)}; SPEC {len(spec_todo)}/{len(spec_unique)}"
    )

    if len(eeg_todo) == 0 and len(spec_todo) == 0:
        return

    tasks = [("spec", sid, spec_dir) for sid in spec_todo] + [
        ("eeg", eid, eeg_dir) for eid in eeg_todo
    ]

    workers = int(
        max(1, min(workers, mp.cpu_count() // 2 if mp.cpu_count() > 2 else 1))
    )
    chunksize = max(1, len(tasks) // (workers * 8))  # amortize mp overhead

    t0 = time.time()
    with mp.get_context("fork").Pool(processes=workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(_precompute_one, tasks, chunksize=chunksize),
            total=len(tasks),
            desc="Precomputing test cache",
        ):
            pass
    print(f"Cache warmup done in {time.time()-t0:.1f}s")




## === cell 6
test_df = pd.read_csv(paths.test_csv)
print(f"Test dataframe shape is: {test_df.shape}")
test_df.head()



## === cell 7
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
        X = np.zeros((128, 256, 8), dtype="float32")

        eeg_id = int(self.eeg_ids[idx])
        spec_id = int(self.spec_ids[idx])

        if self.mode == "test":
            r = 0
        else:
            r = int(self.r_arr[idx]) if self.r_arr is not None else 0

        sp = self.store.spec(spec_id)
        for region in range(4):
            img = sp[r : r + 300, region * 100 : (region + 1) * 100].T

            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)

            ep = 1e-6
            mu = np.nanmean(img)
            std = np.nanstd(img)
            img = (img - mu) / (std + ep)
            img = np.nan_to_num(img, nan=0.0)

            X[14:-14, :, region] = img[:, 22:-22] / 2.0

        eeg_img = self.store.eeg(eeg_id)
        X[:, :, 4:] = eeg_img

        X = torch.tensor(X)

        spectograms = [X[:, :, i : i + 1] for i in range(4)]
        spectograms = torch.cat(spectograms, dim=0)

        eegs = [X[:, :, i : i + 1] for i in range(4, 8)]
        eegs = torch.cat(eegs, dim=0)

        x = torch.cat([spectograms, eegs], dim=1)
        x = torch.cat([x, x, x], dim=2)
        x = x.permute(2, 0, 1)

        if self.mode != "test":
            y = self.targets[idx]
        else:
            y = np.zeros(6, dtype="float32")

        return {"data": x, "target": y}




## === cell 8
customdataset = CustomDataset(test_df, config, mode="test", store=test_store)
sample_item = customdataset[0]
sample_item["data"].shape, sample_item["target"].shape



## === cell 9
_pre_workers = min(8, max(1, mp.cpu_count() - 1))
precompute_test_caches(test_df, paths.test_eeg, paths.test_spec, workers=_pre_workers)

from torch.utils.data import DataLoader

_loader_workers = min(4, max(0, mp.cpu_count() - 1))
test_loader = DataLoader(
    customdataset,
    batch_size=config.batchsize,
    shuffle=False,
    num_workers=_loader_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(_loader_workers > 0),
    prefetch_factor=2 if _loader_workers > 0 else None,
)
X0 = customdataset[0]["data"]
y0 = customdataset[0]["target"]
X0.shape, y0




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




## === cell 12
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
print(f"Found {len(weight_files)} candidate weight files.")

model_path = os.path.join(paths.out, "trained_model.pth")



## === cell 13
predictions = None

if len(weight_files) == 0 and os.path.isfile(model_path):
    weight_files = [model_path]

if len(weight_files) == 0:
    raise RuntimeError(
        "No pretrained weight files found in expected /kaggle/input dirs, and no local trained_model.pth present. "
        "Training from scratch is disabled to meet the 600s timeout requirement."
    )

fold_preds = []
testdataset = CustomDataset(test_df, config, mode="test", store=test_store)
_inf_workers = min(4, max(0, mp.cpu_count() - 1))
testloader = DataLoader(
    testdataset,
    batch_size=config.batchsize,
    shuffle=False,
    num_workers=_inf_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(_inf_workers > 0),
    prefetch_factor=2 if _inf_workers > 0 else None,
)

for wf in weight_files:
    dd = torch.load(wf, map_location="cpu")
    model = Custommodel(config)

    state = dd["model"] if isinstance(dd, dict) and ("model" in dd) else dd
    model.load_state_dict(state, strict=True)

    model.to(device)
    pred_dict = inference_function(testloader, model, device)
    fold_preds.append(pred_dict["predictions"])

predictions = np.mean(np.stack(fold_preds, axis=0), axis=0)
predictions.shape



## === cell 14
predictions = np.asarray(predictions)
if predictions.ndim != 2 or predictions.shape[1] != 6:
    raise ValueError(f"predictions must have shape (N,6), got {predictions.shape}")

row_sums = predictions.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
predictions = predictions / row_sums

predictions = np.clip(predictions, 1e-6, 1.0)
predictions = predictions / predictions.sum(axis=1, keepdims=True)

predictions[:2], predictions.sum(axis=1).min(), predictions.sum(axis=1).max()



## === cell 15
sample_sub_path = (
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
)
sample_sub = pd.read_csv(sample_sub_path)
sub = sample_sub.copy()
sub[TARGETS] = predictions.astype(np.float32)

out_path = os.path.join(paths.out, "submission.csv")
sub.to_csv(out_path, index=False)
print(f"Saved submission to: {out_path}")
print(f"Submission shape: {sub.shape}")
sub.head()
