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

0.5859539902913767

# 6. Current score

1.41937

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I (1) fix the missing external model-weight directories by falling back to a safe, score-neutral baseline predictor when those folders aren’t present, so the notebook always produces a valid `submission.csv`. I (2) remove runtime/logic errors in the dataset (`targets` undefined, invalid `min/max` usage for test, and noisy `print(r)`), and make spectrogram slicing robust to different parquet shapes. I (3) ensure predictions are always a proper `(n_test, 6)` array, clipped and row-normalized to sum to 1 to satisfy the submission constraints. These changes keep the core model definitions/inference semantics intact, but guarantee end-to-end execution and a valid CSV.'
- What this solution (achieved 1.41937) has done: 'Most of the timeout comes from extremely heavy, repeated CPU feature extraction: building mel-spectrogram EEG images for **all** test EEG parquet files (and many train EEGs) with `librosa` inside Python loops. The fastest correctness-preserving fix is to avoid doing work that the submission never uses: skip all training-data loading/training entirely when external weights are present, and also avoid generating unused artifacts (like the extra `df` load and notebook display cells). For inference, keep the same model logic but speed up I/O by (1) using `pyarrow` parquet reads with only needed columns, (2) thread-based parallelism for parquet+librosa (avoids fork overhead / pickling large objects), and (3) precomputing constants used inside the spectrogram loop. These changes keep the same inputs/outputs and model semantics; they only remove redundant computation and reduce overhead.'
- What this solution (achieved 1.41937) has done: 'I fix the two runtime blockers that prevent end-to-end execution: (1) `librosa.feature.melspectrogram` no longer accepts `mel_basis`, so we must pass the correct argument name (`mel_filters`) while keeping the same cached mel filterbank logic, and (2) the test EEG image dictionary was keyed by parquet filename IDs, but the dataset indexes by `eeg_id` from `test.csv`, so we must build `all_eegs` in `test_df` order keyed by `eeg_id` to avoid `KeyError`. I also remove the hard failure when external weight folders are missing and instead fall back to a stable, score-improving (vs uniform) class-prior predictor computed from train vote totals, which is still metric-aligned and legitimate. These changes keep the model/inference logic intact when weights exist, but guarantee a valid `submission.csv` is always produced. Lastly, I keep the existing row-normalization to satisfy the submission constraint that probabilities sum to 1.'

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

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

torch.set_num_threads(max(1, min(os.cpu_count() or 1, 8)))
torch.set_num_interop_threads(max(1, min(os.cpu_count() or 1, 8)))

_CAN_COMPILE = hasattr(torch, "compile")
_COMPILE_MODE = "reduce-overhead"




## === cell 1
class config:
    model1 = "resnet50d"
    model2 = "vit_base_patch16_224"
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

_SR = 200
_NFFT = 1024
_NMELS = 128
_FMIN = 0
_FMAX = 20
_WINLEN = 128
_MEL_FBANK_CACHE = {}
_MEL_SPEC_WIDTH = 256  # target time bins; matches original hop_length=len(x)//256


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


def _get_mel_fbank(sr, n_fft, n_mels, fmin, fmax):
    key = (sr, n_fft, n_mels, fmin, fmax)
    fb = _MEL_FBANK_CACHE.get(key)
    if fb is None:
        fb = librosa.filters.mel(
            sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
        )
        _MEL_FBANK_CACHE[key] = fb
    return fb


def spectrogram_from_eeg(parquet_path, display=False):
    need_cols = sorted(set(sum(FEATS, [])))
    eeg = pd.read_parquet(parquet_path, columns=need_cols, engine="pyarrow")
    for c in eeg.columns:
        if eeg[c].dtype != np.float32:
            eeg[c] = eeg[c].astype(np.float32, copy=False)

    middle = (len(eeg) - 10_000) // 2
    eeg = eeg.iloc[middle : middle + 10_000]

    img = np.zeros((128, 256, 4), dtype="float32")

    if display:
        plt.figure(figsize=(10, 7))

    mel_fbank = _get_mel_fbank(_SR, _NFFT, _NMELS, _FMIN, _FMAX)

    for k in range(4):
        COLS = FEATS[k]
        for kk in range(4):
            a = eeg[COLS[kk]].to_numpy()
            b = eeg[COLS[kk + 1]].to_numpy()
            x = a - b

            m = np.nanmean(x)
            if np.isnan(x).mean() < 1:
                x = np.nan_to_num(x, nan=m)
            else:
                x[:] = 0

            if USE_WAVELET:
                x = denoise(x, wavelet=USE_WAVELET)

            hop_length = max(1, len(x) // _MEL_SPEC_WIDTH)

            mel_spec = librosa.feature.melspectrogram(
                y=x,
                sr=_SR,
                hop_length=hop_length,
                n_fft=_NFFT,
                n_mels=_NMELS,
                fmin=_FMIN,
                fmax=_FMAX,
                win_length=_WINLEN,
                htk=True,
                norm="slaney",
                mel_filters=mel_fbank,
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
    return img




## === cell 3
df = None



## === cell 4
from concurrent.futures import ThreadPoolExecutor, as_completed

test_df = pd.read_csv(paths.test_csv)
print(f"Test dataframe shape is: {test_df.shape}")
print(test_df.head())


def _eeg_to_mel_item_by_eeg_id(eeg_id: int):
    parquet_path = os.path.join(paths.test_eeg, f"{int(eeg_id)}.parquet")
    sp = spectrogram_from_eeg(parquet_path)
    return int(eeg_id), np.array(sp, dtype=np.float32, copy=False)


all_eegs = {}
n_workers = max(1, min((os.cpu_count() or 1), 8))
with ThreadPoolExecutor(max_workers=n_workers) as ex:
    futs = [
        ex.submit(_eeg_to_mel_item_by_eeg_id, int(eid))
        for eid in test_df["eeg_id"].values
    ]
    for fut in tqdm(
        as_completed(futs),
        total=len(futs),
        desc="Loading test EEGs -> mel images (threads, keyed by eeg_id)",
    ):
        eid, sp = fut.result()
        all_eegs[eid] = sp

print(f"all_eegs loaded: {len(all_eegs)}")




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/879358623.py in <cell line: 0>()
     26         desc="Loading test EEGs -> mel images (threads, keyed by eeg_id)",
     27     ):
---> 28         eid, sp = fut.result()
     29         all_eegs[eid] = sp
     30 

/usr/lib/python3.11/concurrent/futures/_base.py in result(self, timeout)
    447                     raise CancelledError()
    448                 elif self._state == FINISHED:
--> 449                     return self.__get_result()
    450 
    451                 self._condition.wait(timeout)

/usr/lib/python3.11/concurrent/futures/_base.py in __get_result(self)
    399         if self._exception:
    400             try:
--> 401                 raise self._exception
    402             finally:
    403                 # Break a reference cycle with the exception in self._exception

/usr/lib/python3.11/concurrent/futures/thread.py in run(self)
     56 
     57         try:
---> 58             result = self.fn(*self.args, **self.kwargs)
     59         except BaseException as exc:
     60             self.future.set_exception(exc)

/tmp/ipykernel_55/879358623.py in _eeg_to_mel_item_by_eeg_id(eeg_id)
     10 def _eeg_to_mel_item_by_eeg_id(eeg_id: int):
     11     parquet_path = os.path.join(paths.test_eeg, f"{int(eeg_id)}.parquet")
---> 12     sp = spectrogram_from_eeg(parquet_path)
     13     return int(eeg_id), np.array(sp, dtype=np.float32, copy=False)
     14 

/tmp/ipykernel_55/3451993901.py in spectrogram_from_eeg(parquet_path, display)
     81 
     82             # Bugfix: librosa>=0.11 uses 'mel_filters' instead of legacy/invalid 'mel_basis'
---> 83             mel_spec = librosa.feature.melspectrogram(
     84                 y=x,
     85                 sr=_SR,

/usr/local/lib/python3.11/dist-packages/librosa/feature/spectral.py in melspectrogram(y, sr, S, n_fft, hop_length, win_length, window, center, pad_mode, power, **kwargs)
   2146 
   2147     # Build a Mel filter
-> 2148     mel_basis = filters.mel(sr=sr, n_fft=n_fft, **kwargs)
   2149 
   2150     melspec: np.ndarray = np.einsum("...ft,mf->...mt", S, mel_basis, optimize=True)

TypeError: mel() got an unexpected keyword argument 'mel_filters'

## === cell 5
def _load_spec_item(fn: str):
    sp = pd.read_parquet(os.path.join(paths.test_spec, fn), engine="pyarrow")
    name = int(fn.split(".")[0])
    arr = sp.to_numpy(copy=False)
    return name, arr


test_spec_files = [f for f in os.listdir(paths.test_spec) if f.endswith(".parquet")]
all_spectrograms = {}

with ThreadPoolExecutor(max_workers=n_workers) as ex:
    futs = [ex.submit(_load_spec_item, fn) for fn in test_spec_files]
    for fut in tqdm(
        as_completed(futs), total=len(futs), desc="Loading test spectrograms (threads)"
    ):
        name, sp = fut.result()
        all_spectrograms[name] = sp

print(f"all_spectrograms loaded: {len(all_spectrograms)}")



## === cell 6
TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
targets = TARGETS  # keep original variable name expectation

train_df_raw = None
train_df = None



## === cell 7
train_specs = None
train_eegs = None




## === cell 8
class CustomDataset:
    def __init__(
        self,
        traindf: pd.DataFrame,
        config,
        mode: str = "train",
        specs: Optional[dict[int, np.ndarray]] = None,
        eegs: Optional[dict[int, np.ndarray]] = None,
    ):
        self.traindf = traindf.reset_index(drop=True)
        self.specs = specs
        self.eeg = eegs
        self.mode = mode
        self._x_cache: dict[int, np.ndarray] = {}  # eeg_id -> (3,512,512) float32

    def __len__(self):
        return len(self.traindf)

    def _get_spec_region_img(
        self, spec_arr: np.ndarray, r: int, region: int
    ) -> np.ndarray:
        """Returns a (100,256) float32 normalized image for one region (freq x time)."""
        if spec_arr.ndim != 2:
            spec_arr = np.squeeze(spec_arr)
            if spec_arr.ndim != 2:
                spec_arr = spec_arr.reshape(spec_arr.shape[0], -1)

        tmax, fmax = spec_arr.shape[0], spec_arr.shape[1]
        r = int(np.clip(r, 0, max(0, tmax - 1)))
        t_end = min(tmax, r + 300)

        f_start = region * 100
        f_end = min(fmax, (region + 1) * 100)
        if f_start >= fmax:
            raw = np.zeros((100, max(1, t_end - r)), dtype=np.float32)
        else:
            raw = spec_arr[r:t_end, f_start:f_end].T.astype(np.float32, copy=False)

        raw = np.clip(raw, np.exp(-4), np.exp(8))
        raw = np.log(raw)

        ep = 1e-6
        mu = np.nanmean(raw)
        std = np.nanstd(raw)
        raw = (raw - mu) / (std + ep)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

        freq_target = 100
        time_target = 256

        if raw.shape[0] < freq_target:
            pad = np.zeros((freq_target - raw.shape[0], raw.shape[1]), dtype=np.float32)
            raw = np.concatenate([raw, pad], axis=0)
        elif raw.shape[0] > freq_target:
            raw = raw[:freq_target, :]

        if raw.shape[1] < time_target:
            pad = np.zeros((raw.shape[0], time_target - raw.shape[1]), dtype=np.float32)
            raw = np.concatenate([raw, pad], axis=1)
        elif raw.shape[1] > time_target:
            raw = raw[:, :time_target]

        return raw

    def _build_x(self, row) -> np.ndarray:
        if self.mode == "test":
            r = 0
        else:
            r = int(row.get("spec_r", 0))

        x = np.empty((3, 512, 512), dtype=np.float32)

        spec_arr = self.specs[int(row.spectrogram_id)]
        for region in range(4):
            img_region = self._get_spec_region_img(spec_arr, r, region)  # (100,256)
            y0 = 128 * region + 14
            y1 = 128 * region + 114
            x[:, 128 * region : 128 * (region + 1), 0:256] = 0.0
            val = (img_region / 2.0).astype(np.float32, copy=False)
            x[0, y0:y1, 0:256] = val
            x[1, y0:y1, 0:256] = val
            x[2, y0:y1, 0:256] = val

        eeg_img = self.eeg[int(row.eeg_id)].astype(
            np.float32, copy=False
        )  # (128,256,4)
        for ch in range(4):
            y0 = 128 * ch
            y1 = 128 * (ch + 1)
            val = eeg_img[:, :, ch]
            x[0, y0:y1, 256:512] = val
            x[1, y0:y1, 256:512] = val
            x[2, y0:y1, 256:512] = val

        return x

    def __getitem__(self, idx):
        row = self.traindf.iloc[idx]
        eeg_id = int(row.eeg_id)

        x = self._x_cache.get(eeg_id)
        if x is None:
            x = self._build_x(row)
            self._x_cache[eeg_id] = x

        if self.mode != "test":
            v = row[targets].values.astype(np.float32, copy=False)
            v_sum = float(np.sum(v))
            if v_sum > 0:
                y = v / v_sum
            else:
                y = np.full(6, 1.0 / 6.0, dtype=np.float32)
        else:
            y = np.zeros(6, dtype=np.float32)

        return {"data": x, "target": y}




## === cell 9
customdataset = CustomDataset(
    test_df, config, mode="test", specs=all_spectrograms, eegs=all_eegs
)

item0 = customdataset[0]
print("Sample shapes:", item0["data"].shape, item0["target"].shape)
print(
    "First spectrogram_id:",
    int(test_df.iloc[0].spectrogram_id),
    "First eeg_id:",
    int(test_df.iloc[0].eeg_id),
)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/2933457113.py in <cell line: 0>()
      3 )
      4 
----> 5 item0 = customdataset[0]
      6 print("Sample shapes:", item0["data"].shape, item0["target"].shape)
      7 print(

/tmp/ipykernel_55/2164538461.py in __getitem__(self, idx)
    101         x = self._x_cache.get(eeg_id)
    102         if x is None:
--> 103             x = self._build_x(row)
    104             self._x_cache[eeg_id] = x
    105 

/tmp/ipykernel_55/2164538461.py in _build_x(self, row)
     82             x[2, y0:y1, 0:256] = val
     83 
---> 84         eeg_img = self.eeg[int(row.eeg_id)].astype(
     85             np.float32, copy=False
     86         )  # (128,256,4)

KeyError: 2578018731

## === cell 10
def fast_collate_fn(batch):
    data = np.stack([b["data"] for b in batch], axis=0)  # (B,3,512,512)
    target = np.stack([b["target"] for b in batch], axis=0)  # (B,6)
    data_t = torch.from_numpy(data)
    target_t = torch.from_numpy(target)
    return {"data": data_t, "target": target_t}


def make_loader(ds, batch_size, shuffle, num_workers):
    num_workers = int(max(0, num_workers))
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=fast_collate_fn,
        drop_last=False,
    )


test_loader = make_loader(
    customdataset,
    batch_size=config.batchsize,
    shuffle=False,
    num_workers=0,
)
X = customdataset[0]["data"]
y = customdataset[0]["target"]
print("y0:", y)




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/2729765899.py in <cell line: 0>()
     28     num_workers=0,
     29 )
---> 30 X = customdataset[0]["data"]
     31 y = customdataset[0]["target"]
     32 print("y0:", y)

/tmp/ipykernel_55/2164538461.py in __getitem__(self, idx)
    101         x = self._x_cache.get(eeg_id)
    102         if x is None:
--> 103             x = self._build_x(row)
    104             self._x_cache[eeg_id] = x
    105 

/tmp/ipykernel_55/2164538461.py in _build_x(self, row)
     82             x[2, y0:y1, 0:256] = val
     83 
---> 84         eeg_img = self.eeg[int(row.eeg_id)].astype(
     85             np.float32, copy=False
     86         )  # (128,256,4)

KeyError: 2578018731

## === cell 11
class Custommodel(nn.Module):
    def __init__(self, config, numclass: int = 6):
        super(Custommodel, self).__init__()
        self.model = timm.create_model(
            config.model1,
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




## === cell 12
def inference_function(test_loader, model, device):
    model.eval()
    softmax = nn.Softmax(dim=1)
    preds = []
    with torch.inference_mode():
        for batch in tqdm(test_loader, desc="Inference", leave=False):
            x = batch["data"].to(device, non_blocking=True)
            if device.type == "cuda":
                x = x.to(memory_format=torch.channels_last)
            ypred = model(x)
            ypred = softmax(ypred)
            preds.append(ypred.detach().cpu().numpy())
    return {"predictions": np.concatenate(preds, axis=0)}




## === cell 13
_TEST_DATASET_ONCE = CustomDataset(
    test_df, config, mode="test", specs=all_spectrograms, eegs=all_eegs
)
_TEST_LOADER_ONCE = make_loader(
    _TEST_DATASET_ONCE,
    batch_size=config.batchsize,
    shuffle=False,
    num_workers=0,
)


def predict_from_weight_dir(weight_dir: str, model_ctor, model_name: str):
    if not os.path.isdir(weight_dir):
        print(f"Weight dir not found, skipping {model_name}: {weight_dir}")
        return None

    weight_files = sorted(
        [f for f in os.listdir(weight_dir) if f.endswith((".pt", ".pth", ".bin"))]
    )
    if len(weight_files) == 0:
        print(f"No weight files found in {weight_dir}, skipping {model_name}")
        return None

    preds_list = []
    testloader = _TEST_LOADER_ONCE

    for wf in weight_files:
        wpath = os.path.join(weight_dir, wf)
        try:
            dd = torch.load(wpath, map_location="cpu")
        except Exception as e:
            print(f"Failed to load {wpath} ({e}), skipping.")
            continue

        model = model_ctor()
        if isinstance(dd, dict) and "model" in dd:
            state = dd["model"]
        else:
            state = dd
        try:
            model.load_state_dict(state, strict=True)
        except Exception as e:
            print(f"State dict mismatch for {wpath} ({e}), skipping.")
            continue

        model.to(device)
        if device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)

        if _CAN_COMPILE:
            try:
                model = torch.compile(model, mode=_COMPILE_MODE, fullgraph=False)
            except Exception:
                pass

        pred = inference_function(testloader, model, device)["predictions"]
        preds_list.append(pred)

        del model, dd, state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(preds_list) == 0:
        print(f"No usable weights loaded for {model_name}.")
        return None

    preds = np.mean(np.stack(preds_list, axis=0), axis=0)
    return preds


predictions = predict_from_weight_dir(
    "/kaggle/input/resnet5010ep2",
    model_ctor=lambda: Custommodel(config),
    model_name="resnet50d",
)




## === cell 14
class Custommodelkk(nn.Module):
    def __init__(self, config, numclass: int = 6):
        super(Custommodelkk, self).__init__()
        self.model = timm.create_model(
            config.model3,
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


predictions3 = predict_from_weight_dir(
    "/kaggle/input/resnet34d2",
    model_ctor=lambda: Custommodelkk(config),
    model_name="resnet34d",
)

print("predictions resnet50d:", None if predictions is None else predictions.shape)



## === cell 15
from torchvision.transforms import transforms

tras = transforms.Compose([transforms.Resize((224, 224))])


class Custommodel2(nn.Module):
    def __init__(self, config, transform, numclass: int = 6):
        super(Custommodel2, self).__init__()
        self.model = timm.create_model(
            config.model2,
            pretrained=False,
        )
        self.model.head = nn.Linear(self.model.head.in_features, numclass)
        self.transform = transform

    def forward(self, x):
        x = self.transform(x)
        x = self.model(x)
        return x


predictions2 = predict_from_weight_dir(
    "/kaggle/input/visiontransformer",
    model_ctor=lambda: Custommodel2(config, tras),
    model_name="vit_base_patch16_224",
)




## === cell 16
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)


def make_patient_folds(
    df_in: pd.DataFrame, n_folds: int = 5, seed: int = 42
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pats = df_in["patient_id"].astype(int).values
    uniq = np.unique(pats)
    rng.shuffle(uniq)
    fold_map = {}
    for i, p in enumerate(uniq):
        fold_map[p] = i % n_folds
    return np.array([fold_map[p] for p in pats], dtype=np.int32)


def row_normalize(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, eps, None)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs


predictions_cv = None




## === cell 17
def train_prior_from_votes(train_csv_path: str) -> np.ndarray:
    tr = pd.read_csv(train_csv_path, usecols=TARGETS)
    vote_sum = tr[TARGETS].sum(axis=0).values.astype(np.float64)
    vote_sum = np.clip(vote_sum, 1e-12, None)
    prior = vote_sum / vote_sum.sum()
    return prior


n_test = len(test_df)

available = [
    p
    for p in [predictions_cv, predictions, predictions2, predictions3]
    if isinstance(p, np.ndarray)
]
if len(available) == 0:
    prior = train_prior_from_votes(paths.train_csv)  # shape (6,)
    finalpred = np.tile(prior[None, :], (n_test, 1))
else:
    finalpred = np.zeros_like(available[0], dtype=np.float64)
    for p in available:
        finalpred += p.astype(np.float64)
    finalpred /= len(available)

finalpred = row_normalize(finalpred)
print("finalpred:", finalpred.shape, finalpred.dtype)



## === cell 18
sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
sub[TARGETS] = finalpred.astype(np.float32)
sub.to_csv("submission.csv", index=False)
print(f"Submission shape: {sub.shape}")
print(sub.head())

print(
    "Row sums (min/mean/max):",
    sub[TARGETS].sum(axis=1).min(),
    sub[TARGETS].sum(axis=1).mean(),
    sub[TARGETS].sum(axis=1).max(),
)
print("Total sum:", float(np.sum(finalpred)))
