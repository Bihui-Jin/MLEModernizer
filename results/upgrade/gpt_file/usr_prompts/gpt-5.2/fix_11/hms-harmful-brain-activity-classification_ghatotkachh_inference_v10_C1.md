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

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

torch.backends.cudnn.benchmark = True




## === cell 2
class config:
    model = "resnet50d"
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


TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]



## === cell 3
import torchaudio
import torchaudio.functional as AF

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


_COLS_NEEDED = sorted({c for group in FEATS for c in group})
_COL_TO_IDX = {c: i for i, c in enumerate(_COLS_NEEDED)}
_FEATS_IDX = [[_COL_TO_IDX[c] for c in group] for group in FEATS]

_SR = 200
_N_FFT = 1024
_N_MELS = 128
_FMIN = 0
_FMAX = 20
_WIN_LENGTH = 128

_MEL_TRANSFORM = None


def _power_to_db_np(S: np.ndarray) -> np.ndarray:
    S = np.maximum(S, 1e-10)
    ref = np.max(S)
    ref = max(ref, 1e-10)
    return 10.0 * np.log10(S) - 10.0 * np.log10(ref)


def _init_mel_transform():
    global _MEL_TRANSFORM
    _MEL_TRANSFORM = None  # hop_length depends on signal length; cached per-hop below.


def _get_mel_transform(hop_length: int):
    global _MEL_TRANSFORM
    if (
        _MEL_TRANSFORM is None
        or getattr(_MEL_TRANSFORM, "_hop_length", None) != hop_length
    ):
        mt = torchaudio.transforms.MelSpectrogram(
            sample_rate=_SR,
            n_fft=_N_FFT,
            win_length=_WIN_LENGTH,
            hop_length=hop_length,
            f_min=_FMIN,
            f_max=_FMAX,
            n_mels=_N_MELS,
            power=2.0,
            center=True,
            pad_mode="reflect",
            norm=None,
            mel_scale="htk",
        )
        mt._hop_length = hop_length
        _MEL_TRANSFORM = mt
    return _MEL_TRANSFORM


def spectrogram_from_eeg(parquet_path, display=False):
    eeg = pd.read_parquet(parquet_path, columns=_COLS_NEEDED)
    arr = eeg.to_numpy(dtype=np.float32, copy=False)  # (T, C)
    n = arr.shape[0]
    middle = (n - 10_000) // 2
    arr = arr[middle : middle + 10_000]

    img = np.zeros((128, 256, 4), dtype=np.float32)

    hop_length = len(arr) // 256  # identical to original

    nanmean = np.nanmean
    isnan = np.isnan
    nan_to_num = np.nan_to_num

    mel_transform = _get_mel_transform(hop_length)

    for k in range(4):
        idxs = _FEATS_IDX[k]
        acc = img[:, :, k]  # view

        xs0 = arr[:, idxs[0]] - arr[:, idxs[1]]
        xs1 = arr[:, idxs[1]] - arr[:, idxs[2]]
        xs2 = arr[:, idxs[2]] - arr[:, idxs[3]]
        xs3 = arr[:, idxs[3]] - arr[:, idxs[4]]

        xs = (xs0, xs1, xs2, xs3)
        xs_fixed = []
        for x in xs:
            m = nanmean(x)
            if isnan(x).mean() < 1:
                x = nan_to_num(x, nan=m)
            else:
                x = x.copy()
                x[:] = 0
            if USE_WAVELET:
                x = denoise(x, wavelet=USE_WAVELET)
            xs_fixed.append(x.astype(np.float32, copy=False))

        xst = torch.from_numpy(np.stack(xs_fixed, axis=0))  # (4, T)
        with torch.inference_mode():
            mel = mel_transform(xst)  # (4, 128, time)
        mel_np = mel.numpy()

        t = mel_np.shape[2]
        width = (t // 32) * 32
        mel_np = mel_np[:, :, :width]  # (4,128,width)

        mel_db0 = (
            _power_to_db_np(mel_np[0]).astype(np.float32, copy=False) + 40.0
        ) / 40.0
        mel_db1 = (
            _power_to_db_np(mel_np[1]).astype(np.float32, copy=False) + 40.0
        ) / 40.0
        mel_db2 = (
            _power_to_db_np(mel_np[2]).astype(np.float32, copy=False) + 40.0
        ) / 40.0
        mel_db3 = (
            _power_to_db_np(mel_np[3]).astype(np.float32, copy=False) + 40.0
        ) / 40.0

        acc += mel_db0
        acc += mel_db1
        acc += mel_db2
        acc += mel_db3
        acc /= 4.0

    return img




## === cell 4
test_df = pd.read_csv(paths.test_csv)
print(f"Test dataframe shape is: {test_df.shape}")
test_df.head()




## === cell 5
def _mp_context():
    try:
        return multiprocessing.get_context("fork")
    except ValueError:
        return multiprocessing.get_context("spawn")


def _pool_init():
    _init_mel_transform()


def _build_one_eeg_spec(args):
    fpath, fid = args
    sp = spectrogram_from_eeg(fpath)
    return fid, sp


eeg_cache_path = os.path.join(paths.out, "test_eeg_specs_cache_v2.npz")

eeg_ids = None
eeg_vals = None
eeg_id_to_idx = None

if os.path.exists(eeg_cache_path):
    loaded = np.load(eeg_cache_path, allow_pickle=False, mmap_mode="r")
    eeg_ids = loaded["ids"].astype(np.int64, copy=False)
    eeg_vals = loaded["vals"]  # (N,128,256,4)
else:
    eeg_files = [
        e
        for e in os.scandir(paths.test_eeg)
        if e.is_file() and e.name.endswith(".parquet")
    ]
    eeg_files.sort(key=lambda e: e.name)
    tasks = [(e.path, int(e.name.split(".")[0])) for e in eeg_files]

    cpu = os.cpu_count() or 2
    max_workers = min(8, max(2, cpu - 1))

    n_tasks = len(tasks)
    eeg_ids = np.empty((n_tasks,), dtype=np.int64)
    eeg_vals = np.empty((n_tasks, 128, 256, 4), dtype=np.float32)

    ctx = _mp_context()
    with ctx.Pool(
        processes=max_workers, initializer=_pool_init, maxtasksperchild=400
    ) as pool:
        for i, (fid, sp) in enumerate(
            tqdm(
                pool.imap(_build_one_eeg_spec, tasks, chunksize=128),
                total=n_tasks,
                desc="Building EEG spectrograms",
            )
        ):
            eeg_ids[i] = int(fid)
            eeg_vals[i] = sp

    np.savez_compressed(eeg_cache_path, ids=eeg_ids, vals=eeg_vals)

eeg_id_to_idx = {int(k): i for i, k in enumerate(eeg_ids)}
len(eeg_id_to_idx), int(eeg_ids[0])



## === cell 6
spec_cols = [str(i) for i in range(400)]


def _load_one_spec(args):
    fpath, fid = args
    sp = pd.read_parquet(fpath, columns=spec_cols)
    return fid, sp.to_numpy(dtype=np.float32, copy=False)


spec_cache_path = os.path.join(paths.out, "test_spectrograms_cache_v2.npz")

spec_ids = None
spec_vals = None
spec_id_to_idx = None

if os.path.exists(spec_cache_path):
    loaded = np.load(spec_cache_path, allow_pickle=False, mmap_mode="r")
    spec_ids = loaded["ids"].astype(np.int64, copy=False)
    spec_vals = loaded["vals"]  # (N, T, 400)
else:
    spec_files = [
        e
        for e in os.scandir(paths.test_spec)
        if e.is_file() and e.name.endswith(".parquet")
    ]
    spec_files.sort(key=lambda e: e.name)
    spec_tasks = [(e.path, int(e.name.split(".")[0])) for e in spec_files]

    cpu = os.cpu_count() or 2
    max_workers = min(8, max(2, cpu - 1))

    n_tasks = len(spec_tasks)
    spec_ids = np.empty((n_tasks,), dtype=np.int64)

    _fid0, _arr0 = _load_one_spec(spec_tasks[0])
    T = _arr0.shape[0]
    spec_vals = np.empty((n_tasks, T, 400), dtype=np.float32)
    spec_ids[0] = int(_fid0)
    spec_vals[0] = _arr0

    ctx = _mp_context()
    with ctx.Pool(processes=max_workers, maxtasksperchild=600) as pool:
        it = pool.imap(_load_one_spec, spec_tasks[1:], chunksize=256)
        for i, (fid, sp) in enumerate(
            tqdm(it, total=n_tasks - 1, desc="Loading test spectrograms")
        ):
            j = i + 1
            spec_ids[j] = int(fid)
            spec_vals[j] = sp

    np.savez_compressed(spec_cache_path, ids=spec_ids, vals=spec_vals)

spec_id_to_idx = {int(k): i for i, k in enumerate(spec_ids)}
len(spec_id_to_idx), int(spec_ids[0])



## === cell 7
from functools import lru_cache


class CustomDataset:
    def __init__(
        self,
        traindf: pd.DataFrame,
        config,
        mode: str = "train",
        specs=None,  # (spec_ids, spec_vals, spec_id_to_idx)
        eegs=None,  # (eeg_ids, eeg_vals, eeg_id_to_idx)
    ):
        self.traindf = traindf.reset_index(drop=True)
        self.mode = mode

        self.spec_ids_arr, self.spec_vals_arr, self.spec_id_to_idx = specs
        self.eeg_ids_arr, self.eeg_vals_arr, self.eeg_id_to_idx = eegs

        self._eeg_ids = self.traindf["eeg_id"].to_numpy(dtype=np.int64, copy=False)
        self._spec_ids = self.traindf["spectrogram_id"].to_numpy(
            dtype=np.int64, copy=False
        )
        self._has_minmax = ("min" in self.traindf.columns) and (
            "max" in self.traindf.columns
        )
        if self._has_minmax:
            self._minv = self.traindf["min"].to_numpy(copy=False)
            self._maxv = self.traindf["max"].to_numpy(copy=False)
        else:
            self._minv = None
            self._maxv = None

        if self.mode != "test":
            self._targets = self.traindf[TARGETS].to_numpy(dtype=np.float32, copy=False)
        else:
            self._targets = None

        self._spec_block_cache_test = None
        if self.mode == "test":
            uniq = np.unique(self._spec_ids)
            cache = {}
            for sid in tqdm(
                uniq, desc="Precomputing normalized spec blocks (test)", leave=False
            ):
                cache[int(sid)] = self._compute_spec_block(int(sid), 0)
            self._spec_block_cache_test = cache

    def __len__(self):
        return len(self.traindf)

    def _get_spec(self, spectrogram_id: int) -> np.ndarray:
        return self.spec_vals_arr[self.spec_id_to_idx[spectrogram_id]]

    def _get_eeg_img(self, eeg_id: int) -> np.ndarray:
        return self.eeg_vals_arr[self.eeg_id_to_idx[eeg_id]]

    def _compute_spec_block(self, spectrogram_id: int, r: int):
        sp = self._get_spec(spectrogram_id)
        r0 = max(0, min(r, max(0, sp.shape[0] - 300)))

        block = sp[r0 : r0 + 300, :400]  # (300,400)
        block = block.reshape(300, 4, 100).transpose(1, 2, 0)  # (4,100,300)

        block = np.clip(block, np.exp(-4), np.exp(8))
        block = np.log(block)

        ep = 1e-6
        mu = np.nanmean(block, axis=(1, 2), keepdims=True)
        std = np.nanstd(block, axis=(1, 2), keepdims=True)
        block = (block - mu) / (std + ep)
        block = np.nan_to_num(block, nan=0.0)

        block = (
            block[:, :, 22:-22].astype(np.float32, copy=False)
        ) / 2.0  # (4,100,256)
        return [block[i] for i in range(4)]

    @lru_cache(maxsize=4096)
    def _cached_spec_block(self, spectrogram_id: int, r: int):
        return self._compute_spec_block(spectrogram_id, r)

    def __getitem__(self, idx):
        X = np.zeros((128, 256, 8), dtype=np.float32)

        if self.mode == "test" or not self._has_minmax:
            r = 0
        else:
            r = int((self._minv[idx] + self._maxv[idx]) // 4)

        sid = int(self._spec_ids[idx])
        if self.mode == "test" and self._spec_block_cache_test is not None:
            blocks = self._spec_block_cache_test[sid]
        else:
            blocks = self._cached_spec_block(sid, r)

        X_spec = X[:, :, :4]
        for region in range(4):
            X_spec[14:-14, :, region] = blocks[region]

        eeg_img = self._get_eeg_img(int(self._eeg_ids[idx]))
        X[:, :, 4:] = eeg_img

        spec_mean = X_spec.mean(axis=2)
        eeg_mean = X[:, :, 4:].mean(axis=2)
        all_mean = X.mean(axis=2)
        x = np.stack([spec_mean, eeg_mean, all_mean], axis=0).astype(
            np.float32, copy=False
        )

        if self.mode != "test":
            y = self._targets[idx]
        else:
            y = np.zeros(6, dtype=np.float32)

        return {"data": torch.from_numpy(x), "target": torch.from_numpy(y)}




## === cell 8
customdataset = CustomDataset(
    test_df,
    config,
    mode="test",
    specs=(spec_ids, spec_vals, spec_id_to_idx),
    eegs=(eeg_ids, eeg_vals, eeg_id_to_idx),
)
sample = customdataset[0]
sample["data"].shape, sample["target"].shape



## === cell 9
cpu = os.cpu_count() or 2
if torch.cuda.is_available():
    loader_workers = min(6, max(2, cpu // 3))
else:
    loader_workers = min(6, max(2, cpu // 2))

test_loader = DataLoader(
    customdataset,
    batch_size=config.batchsize,
    shuffle=False,
    num_workers=loader_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(loader_workers > 0),
    prefetch_factor=2 if loader_workers > 0 else None,
)
batch = next(iter(test_loader))
batch["data"].shape




## === cell 10
class Custommodel(nn.Module):
    def __init__(self, config, numclass: int = 6):
        super(Custommodel, self).__init__()
        self.model = timm.create_model(
            config.model,
            pretrained=False,
            drop_rate=0.1,
            drop_path_rate=0.2,
            in_chans=3,
            num_classes=0,
        )
        self.features = self.model
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
    use_amp = torch.cuda.is_available() and config.AMP

    with torch.inference_mode():
        for batch in tqdm(test_loader, desc="Inference", leave=False):
            x = batch["data"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                ypred = model(x)
                ypred = softmax(ypred)
            preds.append(ypred.detach().cpu().numpy())
    predictions = np.concatenate(preds, axis=0)
    return {"predictions": predictions}




## === cell 12
weights_dir = "/kaggle/input/resnet50"
SKIP_TRAIN = os.path.isdir(weights_dir)

if not SKIP_TRAIN:
    train_df = pd.read_csv(paths.train_csv)

    y_votes = train_df[TARGETS].values.astype(np.float32)
    y_sum = np.clip(y_votes.sum(axis=1, keepdims=True), 1e-6, None)
    train_df[TARGETS] = y_votes / y_sum

    needed_eeg_ids = set(train_df["eeg_id"].unique().tolist())
    needed_spec_ids = set(train_df["spectrogram_id"].unique().tolist())

    def load_parquet_dict(folder, ids_set, desc):
        out = {}
        files = [
            e for e in os.scandir(folder) if e.is_file() and e.name.endswith(".parquet")
        ]
        files.sort(key=lambda e: e.name)
        for e in tqdm(files, desc=desc):
            fid = int(e.name.split(".")[0])
            if fid not in ids_set:
                continue
            out[fid] = pd.read_parquet(e.path).to_numpy(dtype=np.float32, copy=False)
        return out

    def load_eegspec_dict(folder, ids_set, desc):
        out = {}
        files = [
            e for e in os.scandir(folder) if e.is_file() and e.name.endswith(".parquet")
        ]
        files.sort(key=lambda e: e.name)
        for e in tqdm(files, desc=desc):
            fid = int(e.name.split(".")[0])
            if fid not in ids_set:
                continue
            sp = spectrogram_from_eeg(e.path)
            out[fid] = np.array(sp, dtype=np.float32)
        return out

    train_spectrograms = load_parquet_dict(
        paths.train_spec_dir, needed_spec_ids, "Loading train spectrograms"
    )
    train_eegs = load_eegspec_dict(
        paths.train_eeg_dir, needed_eeg_ids, "Building train EEG spectrograms"
    )

    rng = np.random.default_rng(42)
    patients = train_df["patient_id"].unique()
    rng.shuffle(patients)
    val_patients = set(patients[: max(1, int(0.1 * len(patients)))])
    is_val = train_df["patient_id"].isin(val_patients)

    tr_df = train_df.loc[~is_val].reset_index(drop=True)
    va_df = train_df.loc[is_val].reset_index(drop=True)

    train_dataset = CustomDataset(
        tr_df,
        config,
        mode="train",
        specs=(
            np.array(list(train_spectrograms.keys()), dtype=np.int64),
            np.stack(
                [train_spectrograms[k] for k in sorted(train_spectrograms.keys())],
                axis=0,
            ),
            {int(k): i for i, k in enumerate(sorted(train_spectrograms.keys()))},
        ),
        eegs=(
            np.array(list(train_eegs.keys()), dtype=np.int64),
            np.stack([train_eegs[k] for k in sorted(train_eegs.keys())], axis=0),
            {int(k): i for i, k in enumerate(sorted(train_eegs.keys()))},
        ),
    )
    val_dataset = CustomDataset(
        va_df,
        config,
        mode="train",
        specs=(
            np.array(list(train_spectrograms.keys()), dtype=np.int64),
            np.stack(
                [train_spectrograms[k] for k in sorted(train_spectrograms.keys())],
                axis=0,
            ),
            {int(k): i for i, k in enumerate(sorted(train_spectrograms.keys()))},
        ),
        eegs=(
            np.array(list(train_eegs.keys()), dtype=np.int64),
            np.stack([train_eegs[k] for k in sorted(train_eegs.keys())], axis=0),
            {int(k): i for i, k in enumerate(sorted(train_eegs.keys()))},
        ),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batchsize,
        shuffle=True,
        num_workers=loader_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=(loader_workers > 0),
        prefetch_factor=2 if loader_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batchsize,
        shuffle=False,
        num_workers=loader_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(loader_workers > 0),
        prefetch_factor=2 if loader_workers > 0 else None,
    )

    model = Custommodel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.WEIGHT_DECAY
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=(torch.cuda.is_available() and config.AMP)
    )

    def train_one_epoch(model, loader):
        model.train()
        total = 0.0
        n = 0
        for batch in tqdm(loader, desc="Train", leave=False):
            x = batch["data"].to(device, non_blocking=True)
            y = batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(
                enabled=(torch.cuda.is_available() and config.AMP)
            ):
                logits = model(x)
                logp = torch.log_softmax(logits, dim=1)
                loss = -(y * logp).sum(dim=1).mean()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach().cpu()) * x.size(0)
            n += x.size(0)
        return total / max(1, n)

    @torch.no_grad()
    def eval_epoch(model, loader):
        model.eval()
        total = 0.0
        n = 0
        for batch in tqdm(loader, desc="Val", leave=False):
            x = batch["data"].to(device, non_blocking=True)
            y = batch["target"].to(device, non_blocking=True)
            logits = model(x)
            logp = torch.log_softmax(logits, dim=1)
            loss = -(y * logp).sum(dim=1).mean()
            total += float(loss.detach().cpu()) * x.size(0)
            n += x.size(0)
        return total / max(1, n)

    for ep in range(config.epoch):
        tr_loss = train_one_epoch(model, train_loader)
        va_loss = eval_epoch(model, val_loader)
        print(
            f"Epoch {ep+1}/{config.epoch} - train_loss: {tr_loss:.5f}  val_loss: {va_loss:.5f}"
        )

    del train_loader, val_loader, train_dataset, val_dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
else:
    model = None  # will not be used if weights exist



## === cell 13
predictions = None

if os.path.isdir(weights_dir):
    fold_preds = []
    weight_files = sorted(os.listdir(weights_dir))
    for wf in weight_files:
        ckpt_path = os.path.join(weights_dir, wf)
        dd = torch.load(ckpt_path, map_location="cpu")
        m = Custommodel(config)
        state = dd.get("model", dd.get("state_dict", dd))
        m.load_state_dict(state, strict=False)
        m.to(device)

        prediction_dict = inference_function(test_loader, m, device)
        fold_preds.append(prediction_dict["predictions"])

        del m
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    predictions = np.mean(np.stack(fold_preds, axis=0), axis=0)
else:
    prediction_dict = inference_function(test_loader, model, device)
    predictions = prediction_dict["predictions"]

predictions.shape



## === cell 14
predictions = np.clip(predictions, 1e-7, 1.0).astype(np.float64)
predictions = predictions / predictions.sum(axis=1, keepdims=True)

assert predictions.shape[0] == len(test_df)
assert predictions.shape[1] == 6
row_sums = predictions.sum(axis=1)
print("Row sum min/max:", row_sums.min(), row_sums.max())



## === cell 15
sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
sub[TARGETS] = predictions
out_path = os.path.join(paths.out, "submission.csv")
sub.to_csv(out_path, index=False)
print(f"Submission shape: {sub.shape}")
print("Saved to:", out_path)
sub.head()
