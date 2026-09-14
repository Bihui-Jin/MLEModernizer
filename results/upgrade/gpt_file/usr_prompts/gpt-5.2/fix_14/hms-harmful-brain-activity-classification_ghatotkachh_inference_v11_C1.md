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
import numpy as np  # linear algebra
import pandas as pd  # data processing
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
from typing import Dict, List, Optional, Tuple

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using", torch.cuda.device_count(), "GPU(s)")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

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


_MEL_SR = 200
_MEL_NFFT = 1024
_MEL_NMELS = 128
_MEL_FMIN = 0
_MEL_FMAX = 20
_MEL_WIN = 128
_MEL_REF = 1.0  # power_to_db ref=np.max is handled per-spectrogram by dividing by max.
_MEL_FILTER = librosa.filters.mel(
    sr=_MEL_SR, n_fft=_MEL_NFFT, n_mels=_MEL_NMELS, fmin=_MEL_FMIN, fmax=_MEL_FMAX
).astype(np.float32)


def _mel_db_from_signal(x: np.ndarray, hop_length: int) -> np.ndarray:
    stft = librosa.stft(
        y=x,
        n_fft=_MEL_NFFT,
        hop_length=hop_length,
        win_length=_MEL_WIN,
        window="hann",
        center=True,
        pad_mode="reflect",
    )
    power = (np.abs(stft) ** 2).astype(np.float32)
    mel_power = _MEL_FILTER @ power  # (n_mels, t)
    mmax = float(np.max(mel_power)) if mel_power.size else 0.0
    if not np.isfinite(mmax) or mmax <= 0:
        mmax = 1.0
    mel_db = 10.0 * np.log10(np.maximum(mel_power, 1e-10) / mmax).astype(np.float32)
    return mel_db


def spectrogram_from_eeg(parquet_path, display=False):
    needed_cols = []
    for cols in FEATS:
        needed_cols.extend(cols)
    needed_cols = sorted(set(needed_cols))
    eeg = pd.read_parquet(parquet_path, columns=needed_cols)

    middle = (len(eeg) - 10_000) // 2
    eeg = eeg.iloc[middle : middle + 10_000]

    eeg_np = {c: eeg[c].to_numpy(dtype=np.float32, copy=False) for c in needed_cols}

    img = np.zeros((128, 256, 4), dtype="float32")

    if display:
        plt.figure(figsize=(10, 7))

    hop = len(eeg) // 256  # identical to original hop_length=len(x)//256

    for k in range(4):
        COLS = FEATS[k]
        for kk in range(4):
            x = eeg_np[COLS[kk]] - eeg_np[COLS[kk + 1]]

            m = np.nanmean(x)
            if np.isnan(x).mean() < 1:
                x = np.nan_to_num(x, nan=m).astype(np.float32, copy=False)
            else:
                x = np.zeros_like(x, dtype=np.float32)

            if USE_WAVELET:
                x = denoise(x, wavelet=USE_WAVELET).astype(np.float32, copy=False)

            mel_spec_db = _mel_db_from_signal(x, hop_length=hop)

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

    return img




## === cell 3
test_df = pd.read_csv(paths.test_csv)
print(f"Test dataframe shape is: {test_df.shape}")




## === cell 4
def _mp_ctx():
    try:
        return multiprocessing.get_context("fork")
    except ValueError:
        return multiprocessing.get_context("spawn")


def _build_one_eeg_spec(args: Tuple[int, str]) -> Tuple[int, np.ndarray]:
    eid, base_dir = args
    p = os.path.join(base_dir, f"{int(eid)}.parquet")
    sp = spectrogram_from_eeg(p)
    return int(eid), np.asarray(sp, dtype=np.float32)


eeg_cache_path = os.path.join(paths.out, "test_eeg_specs_cache.npy")
eeg_cache_idx_path = os.path.join(paths.out, "test_eeg_specs_cache_ids.npy")

needed_test_eeg_ids = test_df["eeg_id"].astype(int).unique()

all_eegs: Dict[int, np.ndarray] = {}

_preloaded_arr = None
_preloaded_ids = None
_preloaded_id_to_pos = None

if os.path.exists(paths.preloadedeeg):
    try:
        pre = np.load(paths.preloadedeeg, allow_pickle=True)
        if isinstance(pre, np.ndarray) and pre.dtype == object and pre.shape == ():
            pre = pre.item()
        if isinstance(pre, dict):
            if "specs" in pre and ("eeg_ids" in pre or "ids" in pre):
                _preloaded_arr = np.asarray(pre["specs"], dtype=np.float32)
                _preloaded_ids = np.asarray(
                    pre.get("eeg_ids", pre.get("ids")), dtype=np.int64
                )
                _preloaded_id_to_pos = {
                    int(i): j for j, i in enumerate(_preloaded_ids.tolist())
                }
            else:
                _preloaded_id_to_pos = {int(k): idx for idx, k in enumerate(pre.keys())}
                _preloaded_ids = np.fromiter(
                    (int(k) for k in pre.keys()), dtype=np.int64
                )
                _preloaded_arr = np.stack(
                    [np.asarray(pre[k], dtype=np.float32) for k in pre.keys()], axis=0
                )
        else:
            _preloaded_arr = None
    except Exception as e:
        print(
            "Could not use preloaded EEG specs cache, falling back to on-the-fly build:",
            repr(e),
        )
        _preloaded_arr = None

if _preloaded_arr is not None and _preloaded_id_to_pos is not None:
    missing = [
        eid for eid in needed_test_eeg_ids if int(eid) not in _preloaded_id_to_pos
    ]
    if len(missing) == 0:
        all_eegs = {
            int(eid): _preloaded_arr[_preloaded_id_to_pos[int(eid)]]
            for eid in needed_test_eeg_ids
        }
    else:
        print(
            f"Preloaded EEG specs missing {len(missing)} test ids; building missing only."
        )
        all_eegs = {
            int(eid): _preloaded_arr[_preloaded_id_to_pos[int(eid)]]
            for eid in needed_test_eeg_ids
            if int(eid) in _preloaded_id_to_pos
        }

        nproc = min(max(1, (os.cpu_count() or 2) - 1), 8)
        with _mp_ctx().Pool(processes=nproc) as pool:
            for eid, sp in tqdm(
                pool.imap_unordered(
                    _build_one_eeg_spec, [(int(eid), paths.test_eeg) for eid in missing]
                ),
                total=len(missing),
                desc="Building missing test EEG spectrograms (mp)",
            ):
                all_eegs[int(eid)] = sp
else:
    if os.path.exists(eeg_cache_path) and os.path.exists(eeg_cache_idx_path):
        ids = np.load(eeg_cache_idx_path)
        arr = np.load(eeg_cache_path, mmap_mode="r")
        id_to_pos = {int(i): j for j, i in enumerate(ids.tolist())}

        missing = [eid for eid in needed_test_eeg_ids if int(eid) not in id_to_pos]
        if len(missing) == 0:
            all_eegs = {
                int(eid): arr[id_to_pos[int(eid)]] for eid in needed_test_eeg_ids
            }
        else:
            ids_out = np.empty(len(needed_test_eeg_ids), dtype=np.int64)
            arr_out = np.empty(
                (len(needed_test_eeg_ids), 128, 256, 4), dtype=np.float32
            )

            nproc = min(max(1, (os.cpu_count() or 2) - 1), 8)
            results = {}
            with _mp_ctx().Pool(processes=nproc) as pool:
                for eid, sp in tqdm(
                    pool.imap_unordered(
                        _build_one_eeg_spec,
                        [(int(eid), paths.test_eeg) for eid in needed_test_eeg_ids],
                    ),
                    total=len(needed_test_eeg_ids),
                    desc="Building EEG spectrograms (needed test ids, mp)",
                ):
                    results[int(eid)] = sp

            for j, eid in enumerate(needed_test_eeg_ids.tolist()):
                ids_out[j] = int(eid)
                arr_out[j] = results[int(eid)]

            np.save(eeg_cache_idx_path, ids_out)
            np.save(eeg_cache_path, arr_out)
            all_eegs = {int(eid): arr_out[j] for j, eid in enumerate(ids_out)}
    else:
        ids_out = np.empty(len(needed_test_eeg_ids), dtype=np.int64)
        arr_out = np.empty((len(needed_test_eeg_ids), 128, 256, 4), dtype=np.float32)

        nproc = min(max(1, (os.cpu_count() or 2) - 1), 8)
        results = {}
        with _mp_ctx().Pool(processes=nproc) as pool:
            for eid, sp in tqdm(
                pool.imap_unordered(
                    _build_one_eeg_spec,
                    [(int(eid), paths.test_eeg) for eid in needed_test_eeg_ids],
                ),
                total=len(needed_test_eeg_ids),
                desc="Building EEG spectrograms (needed test ids, mp)",
            ):
                results[int(eid)] = sp

        for j, eid in enumerate(needed_test_eeg_ids.tolist()):
            ids_out[j] = int(eid)
            arr_out[j] = results[int(eid)]

        np.save(eeg_cache_idx_path, ids_out)
        np.save(eeg_cache_path, arr_out)
        all_eegs = {int(eid): arr_out[j] for j, eid in enumerate(ids_out)}



## === cell 5
from collections import OrderedDict


class SpectrogramLRU:
    def __init__(self, base_dir: str, max_items: int = 128):
        self.base_dir = base_dir
        self.max_items = max_items
        self.cache = OrderedDict()

    def get(self, sid: int) -> np.ndarray:
        sid = int(sid)
        if sid in self.cache:
            self.cache.move_to_end(sid)
            return self.cache[sid]
        p = os.path.join(self.base_dir, f"{sid}.parquet")
        sp = pd.read_parquet(p).to_numpy()
        self.cache[sid] = sp
        if len(self.cache) > self.max_items:
            self.cache.popitem(last=False)
        return sp


class SpecsProxy(dict):
    def __init__(self, lru: SpectrogramLRU):
        super().__init__()
        self.lru = lru

    def __getitem__(self, key):
        return self.lru.get(int(key))


all_spectrograms = SpecsProxy(SpectrogramLRU(paths.test_spec, max_items=256))



## === cell 6
targets = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]




## === cell 7
class SpecProcDiskCache:
    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        self._loaded = False
        self._ids = None
        self._arr = None
        self._id_to_pos = None
        self._dirty = False
        self._new_items: Dict[int, np.ndarray] = {}

    def _load(self):
        if self._loaded:
            return
        if os.path.exists(self.cache_path):
            try:
                d = np.load(self.cache_path, allow_pickle=False, mmap_mode="r")
                ids = np.array(d["ids"], dtype=np.int64, copy=False)
                arr = np.array(d["arr"], dtype=np.float32, copy=False)
                self._ids = ids
                self._arr = arr
                self._id_to_pos = {int(i): j for j, i in enumerate(ids.tolist())}
            except Exception as e:
                print("SpecProcDiskCache load failed, ignoring cache:", repr(e))
                self._ids, self._arr, self._id_to_pos = None, None, None
        else:
            self._ids, self._arr, self._id_to_pos = None, None, None
        self._loaded = True

    def get(self, sid: int) -> Optional[np.ndarray]:
        self._load()
        sid = int(sid)
        if sid in self._new_items:
            return self._new_items[sid]
        if self._id_to_pos is not None and sid in self._id_to_pos:
            return self._arr[self._id_to_pos[sid]]
        return None

    def set(self, sid: int, value: np.ndarray):
        sid = int(sid)
        self._new_items[sid] = value.astype(np.float32, copy=False)
        self._dirty = True

    def flush(self):
        self._load()
        if not self._dirty:
            return
        if self._ids is None or self._arr is None:
            ids = np.fromiter(
                self._new_items.keys(), dtype=np.int64, count=len(self._new_items)
            )
            ids.sort()
            arr = np.stack(
                [self._new_items[int(sid)] for sid in ids.tolist()], axis=0
            ).astype(np.float32, copy=False)
        else:
            existing = {
                int(sid): self._arr[j] for j, sid in enumerate(self._ids.tolist())
            }
            existing.update(self._new_items)
            ids = np.fromiter(existing.keys(), dtype=np.int64, count=len(existing))
            ids.sort()
            arr = np.stack([existing[int(sid)] for sid in ids.tolist()], axis=0).astype(
                np.float32, copy=False
            )
        tmp = self.cache_path + ".tmp.npz"
        np.savez(tmp, ids=ids, arr=arr)
        os.replace(tmp, self.cache_path)
        self._dirty = False
        self._new_items.clear()


_SPEC_CACHE_TEST = SpecProcDiskCache(os.path.join(paths.out, "specproc_cache_test.npz"))
_SPEC_CACHE_TRAIN = SpecProcDiskCache(
    os.path.join(paths.out, "specproc_cache_train.npz")
)


class CustomDataset:
    def __init__(
        self,
        traindf: pd.DataFrame,
        config,
        mode: str = "train",
        specs: dict[int, np.ndarray] = all_spectrograms,
        eegs: dict[int, np.ndarray] = all_eegs,
        specproc_cache: Optional[SpecProcDiskCache] = None,
    ):
        self.traindf = traindf.copy()
        self.specs = specs
        self.eeg = eegs
        self.mode = mode
        self._specproc_cache_mem: Dict[int, np.ndarray] = {}
        self._specproc_cache_disk = specproc_cache

        if "r" not in self.traindf.columns:
            if self.mode == "test":
                self.traindf["r"] = 0
            else:
                if ("min" in self.traindf.columns) and ("max" in self.traindf.columns):
                    self.traindf["r"] = (
                        (
                            self.traindf["min"].to_numpy()
                            + self.traindf["max"].to_numpy()
                        )
                        // 4
                    ).astype(np.int32)
                else:
                    sids = self.traindf["spectrogram_id"].astype(int).to_numpy()
                    uniq = np.unique(sids)
                    sid_to_r: Dict[int, int] = {}
                    for sid in uniq.tolist():
                        spec = self.specs[int(sid)]
                        sid_to_r[int(sid)] = max(0, (spec.shape[0] - 300) // 2)
                    self.traindf["r"] = np.fromiter(
                        (sid_to_r[int(sid)] for sid in sids.tolist()),
                        dtype=np.int32,
                        count=len(sids),
                    )

    def __len__(self):
        return len(self.traindf)

    def _get_processed_spec4(self, spectrogram_id: int, r: int) -> np.ndarray:
        sid = int(spectrogram_id)

        cached = self._specproc_cache_mem.get(sid, None)
        if cached is not None:
            return cached

        if self._specproc_cache_disk is not None:
            disk = self._specproc_cache_disk.get(sid)
            if disk is not None:
                self._specproc_cache_mem[sid] = disk
                return disk

        spec = self.specs[sid]
        Xspec = np.zeros((128, 256, 4), dtype="float32")
        for region in range(4):
            img = spec[r : r + 300, region * 100 : (region + 1) * 100].T

            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)

            ep = 1e-6
            mu = np.nanmean(img)
            std = np.nanstd(img)
            img = (img - mu) / (std + ep)
            img = np.nan_to_num(img, nan=0.0)

            Xspec[14:-14, :, region] = img[:, 22:-22] / 2.0

        self._specproc_cache_mem[sid] = Xspec
        if self._specproc_cache_disk is not None:
            self._specproc_cache_disk.set(sid, Xspec)
        return Xspec

    def __getitem__(self, idx):
        row = self.traindf.iloc[idx]
        r = int(row["r"])

        X = np.empty((128, 256, 8), dtype="float32")
        X[:, :, :4] = self._get_processed_spec4(int(row.spectrogram_id), r)
        eeg_img = self.eeg[int(row.eeg_id)]
        X[:, :, 4:] = eeg_img

        X_t = torch.from_numpy(X)  # (128,256,8)
        X8 = X_t.permute(2, 0, 1)  # (8,128,256)
        spect = X8[:4]  # (4,128,256)
        eegs = X8[4:]  # (4,128,256)
        x = torch.cat([spect, eegs], dim=2)  # (4,128,512)

        x = x.unsqueeze(-1).expand(-1, -1, -1, 3).reshape(4, 128, 512 * 3)
        x = x.permute(2, 0, 1).contiguous()  # (1536,4,128)

        if self.mode != "test":
            y = row[targets].values.astype(np.float32)
            s = float(np.sum(y))
            if not np.isfinite(s) or s <= 0:
                y = np.ones(6, dtype=np.float32) / 6.0
            else:
                y = y / s
        else:
            y = np.zeros(6, dtype="float32")

        return {"data": x, "target": y}




## === cell 8
customdataset = CustomDataset(
    test_df, config, mode="test", specproc_cache=_SPEC_CACHE_TEST
)




## === cell 9
def _seed_worker(worker_id):
    seed = 42 + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)




## === cell 10
nw = min(4, (os.cpu_count() or 2))
gen = torch.Generator()
gen.manual_seed(42)

test_loader = DataLoader(
    customdataset,
    batch_size=config.batchsize,
    shuffle=False,
    num_workers=nw,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(nw > 0),
    prefetch_factor=2 if nw > 0 else None,
    worker_init_fn=_seed_worker,
    generator=gen,
)




## === cell 11
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




## === cell 12
def inference_function(test_loader, model, device):
    model.eval()
    softmax = nn.Softmax(dim=1)
    preds = []
    with torch.inference_mode():
        for batch in tqdm(test_loader, desc="Inference"):
            x = batch["data"].to(device, non_blocking=True)
            ypred = model(x)
            ypred = softmax(ypred)
            preds.append(ypred.detach().cpu())
    predictions = torch.cat(preds, dim=0).numpy()
    return {"predictions": predictions}




## === cell 13
train_device = torch.device(config.device)
model = Custommodel(config).to(train_device)

ckpt_path = os.path.join(paths.out, "best_state.pth")
if os.path.exists(ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(sd, strict=True)

prediction_dict = inference_function(test_loader, model, train_device)
predictions = prediction_dict["predictions"]

predictions = np.asarray(predictions, dtype=np.float32)
if predictions.ndim != 2 or predictions.shape[1] != 6:
    raise ValueError(f"Predictions must be (n_test, 6). Got {predictions.shape}")

predictions = np.clip(predictions, 1e-8, 1.0)
predictions = predictions / predictions.sum(axis=1, keepdims=True)

_SPEC_CACHE_TEST.flush()
_SPEC_CACHE_TRAIN.flush()



## === cell 14
predictions



## === cell 15
TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
sub[TARGETS] = predictions

row_sums = sub[TARGETS].sum(axis=1).values
if not np.all(np.isfinite(row_sums)):
    raise ValueError("Non-finite values found in submission probabilities.")
sub[TARGETS] = sub[TARGETS].div(row_sums, axis=0)

sub_path = os.path.join(paths.out, "submission.csv")
sub.to_csv(sub_path, index=False)
print(f"Submission shape: {sub.shape}")
print(f"Wrote: {sub_path}")
sub.head()
