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

0.5626450365366341

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 1.40995) has done: 'I fix the two blockers preventing a valid submission: (1) missing pretrained weight directory `/kaggle/input/resnet50` by adding a safe fallback that still produces predictions, and (2) the dataset pipeline issues (undefined `targets`, missing `min/max` columns in test, and a debug `print` that would spam/slow execution). I also correct a shape/channel construction bug so the model receives a proper 3-channel image tensor `(B,3,H,W)` and ensure probabilities are normalized to sum to 1 per row (required for the KL metric). Finally, I make file reading robust for `.parquet` spectrogram/eeg files and keep runtime within limits by avoiding redundant dataset construction and removing accidental heavy prints.'

# 9. Code solution

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




## === cell 1
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




## === cell 2
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


def _init_mel_transform():
    global _MEL_TRANSFORM
    _MEL_TRANSFORM = None  # cached per hop_length via attribute check below


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

    hop_length = len(arr) // 256  # identical to original
    mel_transform = _get_mel_transform(hop_length)

    nanmean = np.nanmean
    isnan = np.isnan
    nan_to_num = np.nan_to_num

    sigs = np.empty((16, arr.shape[0]), dtype=np.float32)
    for k in range(4):
        idxs = _FEATS_IDX[k]
        xs0 = arr[:, idxs[0]] - arr[:, idxs[1]]
        xs1 = arr[:, idxs[1]] - arr[:, idxs[2]]
        xs2 = arr[:, idxs[2]] - arr[:, idxs[3]]
        xs3 = arr[:, idxs[3]] - arr[:, idxs[4]]
        xs = (xs0, xs1, xs2, xs3)

        for j, x in enumerate(xs):
            m = nanmean(x)
            if isnan(x).mean() < 1:
                x = nan_to_num(x, nan=m)
            else:
                x = x.copy()
                x[:] = 0
            if USE_WAVELET:
                x = denoise(x, wavelet=USE_WAVELET)
            sigs[k * 4 + j] = x.astype(np.float32, copy=False)

    xst = torch.from_numpy(sigs)  # (16, T)
    with torch.inference_mode():
        mel = mel_transform(xst)  # (16, 128, time)
    mel_np = mel.numpy()

    t = mel_np.shape[2]
    width = (t // 32) * 32
    mel_np = mel_np[:, :, :width]  # (16, 128, width)

    mel_np = (10.0 * np.log10(np.maximum(mel_np, 1e-10))).astype(np.float32, copy=False)
    mel_np = mel_np.reshape(4, 4, 128, width)  # (region, diff, mel, time)

    mel_np = mel_np - np.max(mel_np, axis=(2, 3), keepdims=True)
    mel_np = (mel_np + 40.0) / 40.0  # (4,4,128,width)
    img = (
        mel_np.mean(axis=1)
        .transpose(2, 3, 0)
        .reshape(128, width, 4)
        .astype(np.float32, copy=False)
    )

    if img.shape[1] != 256:
        if img.shape[1] > 256:
            img = img[:, :256, :]
        else:
            pad = 256 - img.shape[1]
            img = np.pad(
                img, ((0, 0), (0, pad), (0, 0)), mode="constant", constant_values=0.0
            )

    return img




## === cell 3
test_df = pd.read_csv(paths.test_csv)
print(f"Test dataframe shape is: {test_df.shape}")
test_df.head()




## === cell 4
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


eeg_ids_path = os.path.join(paths.out, "test_eeg_ids_cache_v3.npy")
eeg_vals_path = os.path.join(paths.out, "test_eeg_vals_cache_v3.npy")

eeg_ids = None
eeg_vals = None
eeg_id_to_idx = None

if os.path.exists(eeg_ids_path) and os.path.exists(eeg_vals_path):
    eeg_ids = np.load(eeg_ids_path, mmap_mode="r")
    eeg_vals = np.load(eeg_vals_path, mmap_mode="r")  # (N,128,256,4)
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
        processes=max_workers, initializer=_pool_init, maxtasksperchild=2000
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

    np.save(eeg_ids_path, eeg_ids, allow_pickle=False)
    np.save(eeg_vals_path, eeg_vals, allow_pickle=False)

eeg_id_to_idx = {int(k): i for i, k in enumerate(np.asarray(eeg_ids))}
len(eeg_id_to_idx), int(np.asarray(eeg_ids)[0])




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
RemoteTraceback                           Traceback (most recent call last)
RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/usr/lib/python3.11/multiprocessing/pool.py", line 125, in worker
    result = (True, func(*args, **kwds))
                    ^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.11/multiprocessing/pool.py", line 48, in mapstar
    return list(map(*args))
           ^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_55/3525368011.py", line 15, in _build_one_eeg_spec
    sp = spectrogram_from_eeg(fpath)
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_55/1525232993.py", line 125, in spectrogram_from_eeg
    .transpose(2, 3, 0)
     ^^^^^^^^^^^^^^^^^^
numpy.exceptions.AxisError: axis 3 is out of bounds for array of dimension 3
"""

The above exception was the direct cause of the following exception:

AxisError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3525368011.py in <cell line: 0>()
     47         processes=max_workers, initializer=_pool_init, maxtasksperchild=2000
     48     ) as pool:
---> 49         for i, (fid, sp) in enumerate(
     50             tqdm(
     51                 pool.imap(_build_one_eeg_spec, tasks, chunksize=128),

/usr/local/lib/python3.11/dist-packages/tqdm/std.py in __iter__(self)
   1179 
   1180         try:
-> 1181             for obj in iterable:
   1182                 yield obj
   1183                 # Update and possibly print the progressbar.

/usr/lib/python3.11/multiprocessing/pool.py in <genexpr>(.0)
    421                     result._set_length
    422                 ))
--> 423             return (item for chunk in result for item in chunk)
    424 
    425     def imap_unordered(self, func, iterable, chunksize=1):

/usr/lib/python3.11/multiprocessing/pool.py in next(self, timeout)
    871         if success:
    872             return value
--> 873         raise value
    874 
    875     __next__ = next                    # XXX

AxisError: axis 3 is out of bounds for array of dimension 3

## === cell 5
spec_cols = [str(i) for i in range(400)]


def _load_one_spec(args):
    fpath, fid = args
    sp = pd.read_parquet(fpath, columns=spec_cols)
    return fid, sp.to_numpy(dtype=np.float32, copy=False)


spec_ids_path = os.path.join(paths.out, "test_spec_ids_cache_v3.npy")
spec_vals_path = os.path.join(paths.out, "test_spec_vals_cache_v3.npy")

spec_ids = None
spec_vals = None
spec_id_to_idx = None

if os.path.exists(spec_ids_path) and os.path.exists(spec_vals_path):
    spec_ids = np.load(spec_ids_path, mmap_mode="r")
    spec_vals = np.load(spec_vals_path, mmap_mode="r")
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
    with ctx.Pool(processes=max_workers, maxtasksperchild=4000) as pool:
        it = pool.imap(_load_one_spec, spec_tasks[1:], chunksize=256)
        for i, (fid, sp) in enumerate(
            tqdm(it, total=n_tasks - 1, desc="Loading test spectrograms")
        ):
            j = i + 1
            spec_ids[j] = int(fid)
            spec_vals[j] = sp

    np.save(spec_ids_path, spec_ids, allow_pickle=False)
    np.save(spec_vals_path, spec_vals, allow_pickle=False)

spec_id_to_idx = {int(k): i for i, k in enumerate(np.asarray(spec_ids))}
len(spec_id_to_idx), int(np.asarray(spec_ids)[0])




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ArrowInvalid                              Traceback (most recent call last)
/tmp/ipykernel_55/1488182890.py in <cell line: 0>()
     33     spec_ids = np.empty((n_tasks,), dtype=np.int64)
     34 
---> 35     _fid0, _arr0 = _load_one_spec(spec_tasks[0])
     36     T = _arr0.shape[0]
     37     spec_vals = np.empty((n_tasks, T, 400), dtype=np.float32)

/tmp/ipykernel_55/1488182890.py in _load_one_spec(args)
      4 def _load_one_spec(args):
      5     fpath, fid = args
----> 6     sp = pd.read_parquet(fpath, columns=spec_cols)
      7     return fid, sp.to_numpy(dtype=np.float32, copy=False)
      8 

/usr/local/lib/python3.11/dist-packages/pandas/io/parquet.py in read_parquet(path, engine, columns, storage_options, use_nullable_dtypes, dtype_backend, filesystem, filters, **kwargs)
    665     check_dtype_backend(dtype_backend)
    666 
--> 667     return impl.read(
    668         path,
    669         columns=columns,

/usr/local/lib/python3.11/dist-packages/pandas/io/parquet.py in read(self, path, columns, filters, use_nullable_dtypes, dtype_backend, storage_options, filesystem, **kwargs)
    272         )
    273         try:
--> 274             pa_table = self.api.parquet.read_table(
    275                 path_or_handle,
    276                 columns=columns,

/usr/local/lib/python3.11/dist-packages/pyarrow/parquet/core.py in read_table(source, columns, use_threads, schema, use_pandas_metadata, read_dictionary, memory_map, buffer_size, partitioning, filesystem, filters, use_legacy_dataset, ignore_prefixes, pre_buffer, coerce_int96_timestamp_unit, decryption_properties, thrift_string_size_limit, thrift_container_size_limit, page_checksum_verification)
   1841         )
   1842 
-> 1843     return dataset.read(columns=columns, use_threads=use_threads,
   1844                         use_pandas_metadata=use_pandas_metadata)
   1845 

/usr/local/lib/python3.11/dist-packages/pyarrow/parquet/core.py in read(self, columns, use_threads, use_pandas_metadata)
   1483                 )
   1484 
-> 1485         table = self._dataset.to_table(
   1486             columns=columns, filter=self._filter_expression,
   1487             use_threads=use_threads

/usr/local/lib/python3.11/dist-packages/pyarrow/_dataset.pyx in pyarrow._dataset.Dataset.to_table()

/usr/local/lib/python3.11/dist-packages/pyarrow/_dataset.pyx in pyarrow._dataset.Dataset.scanner()

/usr/local/lib/python3.11/dist-packages/pyarrow/_dataset.pyx in pyarrow._dataset.Scanner.from_dataset()

/usr/local/lib/python3.11/dist-packages/pyarrow/_dataset.pyx in pyarrow._dataset.Scanner._make_scan_options()

/usr/local/lib/python3.11/dist-packages/pyarrow/_dataset.pyx in pyarrow._dataset._populate_builder()

/usr/local/lib/python3.11/dist-packages/pyarrow/error.pxi in pyarrow.lib.check_status()

ArrowInvalid: No match for FieldRef.Name(0) in time: int64
LL_0.59: float
LL_0.78: float
LL_0.98: float
LL_1.17: float
LL_1.37: float
LL_1.56: float
LL_1.76: float
LL_1.95: float
LL_2.15: float
LL_2.34: float
LL_2.54: float
LL_2.73: float
LL_2.93: float
LL_3.13: float
LL_3.32: float
LL_3.52: float
LL_3.71: float
LL_3.91: float
LL_4.1: float
LL_4.3: float
LL_4.49: float
LL_4.69: float
LL_4.88: float
LL_5.08: float
LL_5.27: float
LL_5.47: float
LL_5.66: float
LL_5.86: float
LL_6.05: float
LL_6.25: float
LL_6.45: float
LL_6.64: float
LL_6.84: float
LL_7.03: float
LL_7.23: float
LL_7.42: float
LL_7.62: float
LL_7.81: float
LL_8.01: float
LL_8.2: float
LL_8.4: float
LL_8.59: float
LL_8.79: float
LL_8.98: float
LL_9.18: float
LL_9.38: float
LL_9.57: float
LL_9.77: float
LL_9.96: float
LL_10.16: float
LL_10.35: float
LL_10.55: float
LL_10.74: float
LL_10.94: float
LL_11.13: float
LL_11.33: float
LL_11.52: float
LL_11.72: float
LL_11.91: float
LL_12.11: float
LL_12.3: float
LL_12.5: float
LL_12.7: float
LL_12.89: float
LL_13.09: float
LL_13.28: float
LL_13.48: float
LL_13.67: float
LL_13.87: float
LL_14.06: float
LL_14.26: float
LL_14.45: float
LL_14.65: float
LL_14.84: float
LL_15.04: float
LL_15.23: float
LL_15.43: float
LL_15.63: float
LL_15.82: float
LL_16.02: float
LL_16.21: float
LL_16.41: float
LL_16.6: float
LL_16.8: float
LL_16.99: float
LL_17.19: float
LL_17.38: float
LL_17.58: float
LL_17.77: float
LL_17.97: float
LL_18.16: float
LL_18.36: float
LL_18.55: float
LL_18.75: float
LL_18.95: float
LL_19.14: float
LL_19.34: float
LL_19.53: float
LL_19.73: float
LL_19.92: float
RL_0.59: float
RL_0.78: float
RL_0.98: float
RL_1.17: float
RL_1.37: float
RL_1.56: float
RL_1.76: float
RL_1.95: float
RL_2.15: float
RL_2.34: float
RL_2.54: float
RL_2.73: float
RL_2.93: float
RL_3.13: float
RL_3.32: float
RL_3.52: float
RL_3.71: float
RL_3.91: float
RL_4.1: float
RL_4.3: float
RL_4.49: float
RL_4.69: float
RL_4.88: float
RL_5.08: float
RL_5.27: float
RL_5.47: float
RL_5.66: float
RL_5.86: float
RL_6.05: float
RL_6.25: float
RL_6.45: float
RL_6.64: float
RL_6.84: float
RL_7.03: float
RL_7.23: float
RL_7.42: float
RL_7.62: float
RL_7.81: float
RL_8.01: float
RL_8.2: float
RL_8.4: float
RL_8.59: float
RL_8.79: float
RL_8.98: float
RL_9.18: float
RL_9.38: float
RL_9.57: float
RL_9.77: float
RL_9.96: float
RL_10.16: float
RL_10.35: float
RL_10.55: float
RL_10.74: float
RL_10.94: float
RL_11.13: float
RL_11.33: float
RL_11.52: float
RL_11.72: float
RL_11.91: float
RL_12.11: float
RL_12.3: float
RL_12.5: float
RL_12.7: float
RL_12.89: float
RL_13.09: float
RL_13.28: float
RL_13.48: float
RL_13.67: float
RL_13.87: float
RL_14.06: float
RL_14.26: float
RL_14.45: float
RL_14.65: float
RL_14.84: float
RL_15.04: float
RL_15.23: float
RL_15.43: float
RL_15.63: float
RL_15.82: float
RL_16.02: float
RL_16.21: float
RL_16.41: float
RL_16.6: float
RL_16.8: float
RL_16.99: float
RL_17.19: float
RL_17.38: float
RL_17.58: float
RL_17.77: float
RL_17.97: float
RL_18.16: float
RL_18.36: float
RL_18.55: float
RL_18.75: float
RL_18.95: float
RL_19.14: float
RL_19.34: float
RL_19.53: float
RL_19.73: float
RL_19.92: float
LP_0.59: float
LP_0.78: float
LP_0.98: float
LP_1.17: float
LP_1.37: float
LP_1.56: float
LP_1.76: float
LP_1.95: float
LP_2.15: float
LP_2.34: float
LP_2.54: float
LP_2.73: float
LP_2.93: float
LP_3.13: float
LP_3.32: float
LP_3.52: float
LP_3.71: float
LP_3.91: float
LP_4.1: float
LP_4.3: float
LP_4.49: float
LP_4.69: float
LP_4.88: float
LP_5.08: float
LP_5.27: float
LP_5.47: float
LP_5.66: float
LP_5.86: float
LP_6.05: float
LP_6.25: float
LP_6.45: float
LP_6.64: float
LP_6.84: float
LP_7.03: float
LP_7.23: float
LP_7.42: float
LP_7.62: float
LP_7.81: float
LP_8.01: float
LP_8.2: float
LP_8.4: float
LP_8.59: float
LP_8.79: float
LP_8.98: float
LP_9.18: float
LP_9.38: float
LP_9.57: float
LP_9.77: float
LP_9.96: float
LP_10.16: float
LP_10.35: float
LP_10.55: float
LP_10.74: float
LP_10.94: float
LP_11.13: float
LP_11.33: float
LP_11.52: float
LP_11.72: float
LP_11.91: float
LP_12.11: float
LP_12.3: float
LP_12.5: float
LP_12.7: float
LP_12.89: float
LP_13.09: float
LP_13.28: float
LP_13.48: float
LP_13.67: float
LP_13.87: float
LP_14.06: float
LP_14.26: float
LP_14.45: float
LP_14.65: float
LP_14.84: float
LP_15.04: float
LP_15.23: float
LP_15.43: float
LP_15.63: float
LP_15.82: float
LP_16.02: float
LP_16.21: float
LP_16.41: float
LP_16.6: float
LP_16.8: float
LP_16.99: float
LP_17.19: float
LP_17.38: float
LP_17.58: float
LP_17.77: float
LP_17.97: float
LP_18.16: float
LP_18.36: float
LP_18.55: float
LP_18.75: float
LP_18.95: float
LP_19.14: float
LP_19.34: float
LP_19.53: float
LP_19.73: float
LP_19.92: float
RP_0.59: float
RP_0.78: float
RP_0.98: float
RP_1.17: float
RP_1.37: float
RP_1.56: float
RP_1.76: float
RP_1.95: float
RP_2.15: float
RP_2.34: float
RP_2.54: float
RP_2.73: float
RP_2.93: float
RP_3.13: float
RP_3.32: float
RP_3.52: float
RP_3.71: float
RP_3.91: float
RP_4.1: float
RP_4.3: float
RP_4.49: float
RP_4.69: float
RP_4.88: float
RP_5.08: float
RP_5.27: float
RP_5.47: float
RP_5.66: float
RP_5.86: float
RP_6.05: float
RP_6.25: float
RP_6.45: float
RP_6.64: float
RP_6.84: float
RP_7.03: float
RP_7.23: float
RP_7.42: float
RP_7.62: float
RP_7.81: float
RP_8.01: float
RP_8.2: float
RP_8.4: float
RP_8.59: float
RP_8.79: float
RP_8.98: float
RP_9.18: float
RP_9.38: float
RP_9.57: float
RP_9.77: float
RP_9.96: float
RP_10.16: float
RP_10.35: float
RP_10.55: float
RP_10.74: float
RP_10.94: float
RP_11.13: float
RP_11.33: float
RP_11.52: float
RP_11.72: float
RP_11.91: float
RP_12.11: float
RP_12.3: float
RP_12.5: float
RP_12.7: float
RP_12.89: float
RP_13.09: float
RP_13.28: float
RP_13.48: float
RP_13.67: float
RP_13.87: float
RP_14.06: float
RP_14.26: float
RP_14.45: float
RP_14.65: float
RP_14.84: float
RP_15.04: float
RP_15.23: float
RP_15.43: float
RP_15.63: float
RP_15.82: float
RP_16.02: float
RP_16.21: float
RP_16.41: float
RP_16.6: float
RP_16.8: float
RP_16.99: float
RP_17.19: float
RP_17.38: float
RP_17.58: float
RP_17.77: float
RP_17.97: float
RP_18.16: float
RP_18.36: float
RP_18.55: float
RP_18.75: float
RP_18.95: float
RP_19.14: float
RP_19.34: float
RP_19.53: float
RP_19.73: float
RP_19.92: float
__fragment_index: int32
__batch_index: int32
__last_in_fragment: bool
__filename: string

## === cell 6
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

        self._zero_target = torch.zeros(6, dtype=torch.float32)

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
        X_spec[14:-14, :, 0] = blocks[0]
        X_spec[14:-14, :, 1] = blocks[1]
        X_spec[14:-14, :, 2] = blocks[2]
        X_spec[14:-14, :, 3] = blocks[3]

        eeg_img = self._get_eeg_img(int(self._eeg_ids[idx]))
        X[:, :, 4:] = eeg_img

        spec_mean = X_spec.mean(axis=2)
        eeg_mean = X[:, :, 4:].mean(axis=2)
        all_mean = X.mean(axis=2)
        x = np.stack((spec_mean, eeg_mean, all_mean), axis=0).astype(
            np.float32, copy=False
        )

        if self.mode != "test":
            y = torch.from_numpy(self._targets[idx])
        else:
            y = self._zero_target

        return {"data": torch.from_numpy(x), "target": y}




## === cell 7
customdataset = CustomDataset(
    test_df,
    config,
    mode="test",
    specs=(spec_ids, spec_vals, spec_id_to_idx),
    eegs=(eeg_ids, eeg_vals, eeg_id_to_idx),
)
sample = customdataset[0]
sample["data"].shape, sample["target"].shape




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/161207376.py in <cell line: 0>()
----> 1 customdataset = CustomDataset(
      2     test_df,
      3     config,
      4     mode="test",
      5     specs=(spec_ids, spec_vals, spec_id_to_idx),

/tmp/ipykernel_55/2593486103.py in __init__(self, traindf, config, mode, specs, eegs)
     44                 uniq, desc="Precomputing normalized spec blocks (test)", leave=False
     45             ):
---> 46                 cache[int(sid)] = self._compute_spec_block(int(sid), 0)
     47             self._spec_block_cache_test = cache
     48 

/tmp/ipykernel_55/2593486103.py in _compute_spec_block(self, spectrogram_id, r)
     60 
     61     def _compute_spec_block(self, spectrogram_id: int, r: int):
---> 62         sp = self._get_spec(spectrogram_id)
     63         r0 = max(0, min(r, max(0, sp.shape[0] - 300)))
     64 

/tmp/ipykernel_55/2593486103.py in _get_spec(self, spectrogram_id)
     54 
     55     def _get_spec(self, spectrogram_id: int) -> np.ndarray:
---> 56         return self.spec_vals_arr[self.spec_id_to_idx[spectrogram_id]]
     57 
     58     def _get_eeg_img(self, eeg_id: int) -> np.ndarray:

TypeError: 'NoneType' object is not subscriptable

## === cell 8
cpu = os.cpu_count() or 2
if torch.cuda.is_available():
    loader_workers = min(8, max(2, cpu // 2))
else:
    loader_workers = min(8, max(2, cpu // 2))

test_loader = DataLoader(
    customdataset,
    batch_size=config.batchsize,
    shuffle=False,
    num_workers=loader_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(loader_workers > 0),
    prefetch_factor=4 if loader_workers > 0 else None,
)
batch = next(iter(test_loader))
batch["data"].shape




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3571757925.py in <cell line: 0>()
      7 
      8 test_loader = DataLoader(
----> 9     customdataset,
     10     batch_size=config.batchsize,
     11     shuffle=False,

NameError: name 'customdataset' is not defined

## === cell 9
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




## === cell 10
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




## === cell 11
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
        prefetch_factor=4 if loader_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batchsize,
        shuffle=False,
        num_workers=loader_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(loader_workers > 0),
        prefetch_factor=4 if loader_workers > 0 else None,
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




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
AxisError                                 Traceback (most recent call last)
/tmp/ipykernel_55/361129867.py in <cell line: 0>()
     42         paths.train_spec_dir, needed_spec_ids, "Loading train spectrograms"
     43     )
---> 44     train_eegs = load_eegspec_dict(
     45         paths.train_eeg_dir, needed_eeg_ids, "Building train EEG spectrograms"
     46     )

/tmp/ipykernel_55/361129867.py in load_eegspec_dict(folder, ids_set, desc)
     35             if fid not in ids_set:
     36                 continue
---> 37             sp = spectrogram_from_eeg(e.path)
     38             out[fid] = np.array(sp, dtype=np.float32)
     39         return out

/tmp/ipykernel_55/1525232993.py in spectrogram_from_eeg(parquet_path, display)
    123     img = (
    124         mel_np.mean(axis=1)
--> 125         .transpose(2, 3, 0)
    126         .reshape(128, width, 4)
    127         .astype(np.float32, copy=False)

AxisError: axis 3 is out of bounds for array of dimension 3

## === cell 12
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




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2187513871.py in <cell line: 0>()
     22     predictions = np.mean(np.stack(fold_preds, axis=0), axis=0)
     23 else:
---> 24     prediction_dict = inference_function(test_loader, model, device)
     25     predictions = prediction_dict["predictions"]
     26 

NameError: name 'test_loader' is not defined

## === cell 13
predictions = np.clip(predictions, 1e-7, 1.0).astype(np.float64)
predictions = predictions / predictions.sum(axis=1, keepdims=True)

assert predictions.shape[0] == len(test_df)
assert predictions.shape[1] == 6
row_sums = predictions.sum(axis=1)
print("Row sum min/max:", row_sums.min(), row_sums.max())




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1491836415.py in <cell line: 0>()
----> 1 predictions = np.clip(predictions, 1e-7, 1.0).astype(np.float64)
      2 predictions = predictions / predictions.sum(axis=1, keepdims=True)
      3 
      4 assert predictions.shape[0] == len(test_df)
      5 assert predictions.shape[1] == 6

/usr/local/lib/python3.11/dist-packages/numpy/core/fromnumeric.py in clip(a, a_min, a_max, out, **kwargs)
   2167 
   2168     """
-> 2169     return _wrapfunc(a, 'clip', a_min, a_max, out=out, **kwargs)
   2170 
   2171 

/usr/local/lib/python3.11/dist-packages/numpy/core/fromnumeric.py in _wrapfunc(obj, method, *args, **kwds)
     54     bound = getattr(obj, method, None)
     55     if bound is None:
---> 56         return _wrapit(obj, method, *args, **kwds)
     57 
     58     try:

/usr/local/lib/python3.11/dist-packages/numpy/core/fromnumeric.py in _wrapit(obj, method, *args, **kwds)
     43     except AttributeError:
     44         wrap = None
---> 45     result = getattr(asarray(obj), method)(*args, **kwds)
     46     if wrap:
     47         if not isinstance(result, mu.ndarray):

/usr/local/lib/python3.11/dist-packages/numpy/core/_methods.py in _clip(a, min, max, out, **kwargs)
     97         return um.maximum(a, min, out=out, **kwargs)
     98     else:
---> 99         return um.clip(a, min, max, out=out, **kwargs)
    100 
    101 def _mean(a, axis=None, dtype=None, out=None, keepdims=False, *, where=True):

TypeError: '>=' not supported between instances of 'NoneType' and 'float'

## === cell 14
sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
sub[TARGETS] = predictions
out_path = os.path.join(paths.out, "submission.csv")
sub.to_csv(out_path, index=False)
print(f"Submission shape: {sub.shape}")
print("Saved to:", out_path)
sub.head()

## --- ERROR in outputing the csv:
Invalid submission: Submission probabilities must add to 1 for each row
