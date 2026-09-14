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

0.6289474645234481

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.4102) has done: 'I (1) remove the dependency on the missing `/kaggle/input/10ep5foldresent18` weights by falling back to an untrained model inference that still produces valid probabilities, so the notebook always runs end-to-end and yields a `submission.csv`. I (2) fix dataset bugs that currently prevent correct tensor construction/inference (undefined `targets`, invalid `min/max` usage for test, and a `print` inside `__getitem__` that would spam/slow the dataloader). I (3) ensure predictions have the exact `(n_test, 6)` shape and are normalized to sum to 1 per row (required by the competition), preventing the “Columns must be same length as key” error. These changes are execution-stability focused (current score not yielded), while preserving the original model architecture and inference semantics.'
- What this solution (achieved 1.38832) has done: 'Your current score is much worse than the target (lower is better), and the main reason is that you’re effectively doing untrained inference because the external weight directory is missing, so predictions are close to random and yield high KL. The smallest legitimate improvement (without changing the model/feature core logic) is to train the exact same model briefly on the provided `train.csv` using the same input pipeline, then run inference on test. To keep it within the 600s budget, this patch trains on a small, deterministic subset of unique `eeg_id`s (still legitimate, no leakage) and uses a single pass of epochs with KL-aligned soft targets (vote proportions). The submission formatting and probability normalization remain unchanged to avoid invalid submissions.'
- What this solution (achieved 1.40264) has done: 'Your current score (1.38832, lower-is-better) is far from the target (0.62895), so we should improve generalization with minimal, metric-aligned changes while keeping the same model and feature pipeline. The biggest issue is that training uses raw vote counts as soft targets, while the competition evaluates against vote *proportions*; we fix this by normalizing targets inside the Dataset (so both train/inference semantics stay consistent) and keep the KLDiv objective unchanged. To reduce overfitting and move the score down toward the target, we also do a tiny patient-wise split (no leakage) and only train on the train split, selecting the best of the 2 epochs by validation KL (no early stopping—still exactly 2 epochs). Finally, we keep the submission formatting identical and ensure per-row probabilities sum to 1.'
- What this solution (achieved 0.83431) has done: 'The timeout is dominated by heavy, single-threaded preprocessing: (1) building EEG-derived spectrograms for all 1,692 test EEG parquet files using `librosa` four times per region (16 mel-spectrogram calls per file), and (2) loading every test spectrogram parquet into a Python dict even though only a small crop is needed per sample. To preserve core logic and outputs, the optimized script keeps the exact same transformations but removes redundant work by caching per-file results, using fast direct parquet column reads, precomputing the mel filter once, and computing the mel-spectrogram via an equivalent STFT→mel→dB pipeline (same parameters) without `librosa.feature.melspectrogram` overhead. It also avoids materializing all test spectrograms in memory by adding a tiny on-demand cache (same data, same crop), and speeds up DataLoader throughput with `persistent_workers` + limited workers while keeping determinism. These changes reduce wall time drastically while keeping the same model, training loop, loss, and feature semantics (only negligible FP differences).'

# 9. Code solution

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

try:
    from scipy.signal import stft as _scipy_stft
except Exception:
    _scipy_stft = None

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


_HANN_WIN = np.hanning(_MEL_WIN).astype(np.float32)


def _mel_db_from_signal(x: np.ndarray, hop_length: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return np.zeros((_MEL_NMELS, 0), dtype=np.float32)

    if _scipy_stft is None:
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
    else:
        pad = _MEL_NFFT // 2
        xpad = np.pad(x, (pad, pad), mode="reflect")
        _, _, Zxx = _scipy_stft(
            xpad,
            fs=_MEL_SR,
            window=_HANN_WIN,
            nperseg=_MEL_WIN,
            noverlap=_MEL_WIN - hop_length,
            nfft=_MEL_NFFT,
            detrend=False,
            return_onesided=True,
            boundary=None,
            padded=False,
            axis=-1,
        )
        power = (np.abs(Zxx) ** 2).astype(np.float32)

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


class LazyEEGSpecs:
    def __init__(
        self,
        base_dir: str,
        preloaded_path: Optional[str] = None,
        max_items: int = 256,
    ):
        from collections import OrderedDict

        self.base_dir = base_dir
        self.max_items = max_items
        self.cache = OrderedDict()

        self._arr = None
        self._id_to_pos = None
        if preloaded_path is not None and os.path.exists(preloaded_path):
            try:
                pre = np.load(preloaded_path, allow_pickle=True)
                if (
                    isinstance(pre, np.ndarray)
                    and pre.dtype == object
                    and pre.shape == ()
                ):
                    pre = pre.item()
                if isinstance(pre, dict):
                    if "specs" in pre and ("eeg_ids" in pre or "ids" in pre):
                        arr = np.asarray(pre["specs"], dtype=np.float32)
                        ids = np.asarray(
                            pre.get("eeg_ids", pre.get("ids")), dtype=np.int64
                        )
                        self._arr = arr
                        self._id_to_pos = {
                            int(i): j for j, i in enumerate(ids.tolist())
                        }
                    else:
                        keys = [int(k) for k in pre.keys()]
                        self._arr = np.stack(
                            [np.asarray(pre[k], dtype=np.float32) for k in pre.keys()],
                            axis=0,
                        )
                        self._id_to_pos = {int(k): j for j, k in enumerate(keys)}
            except Exception as e:
                print(
                    "Could not use preloaded EEG specs; will build on-demand. Reason:",
                    repr(e),
                )
                self._arr = None
                self._id_to_pos = None

    def get(self, eid: int) -> np.ndarray:
        from collections import OrderedDict

        eid = int(eid)
        if eid in self.cache:
            self.cache.move_to_end(eid)
            return self.cache[eid]

        if self._id_to_pos is not None and eid in self._id_to_pos:
            sp = self._arr[self._id_to_pos[eid]]
        else:
            p = os.path.join(self.base_dir, f"{eid}.parquet")
            sp = np.asarray(spectrogram_from_eeg(p), dtype=np.float32)

        self.cache[eid] = sp
        if len(self.cache) > self.max_items:
            self.cache.popitem(last=False)
        return sp

    def __getitem__(self, key):
        return self.get(int(key))


all_eegs = LazyEEGSpecs(
    paths.test_eeg, preloaded_path=paths.preloadedeeg, max_items=256
)



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


def _build_x(x_t: torch.Tensor) -> torch.Tensor:
    spect = x_t[:4]  # (4,128,256)
    eegs = x_t[4:]  # (4,128,256)
    x0 = torch.cat([spect, eegs], dim=2)  # (4,128,512)
    x = x0.permute(2, 0, 1).contiguous()  # (512,4,128)
    x = x.repeat_interleave(3, dim=0)  # (1536,4,128)
    return x


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

        self._eeg_ids = self.traindf["eeg_id"].astype(np.int64).to_numpy()
        self._spec_ids = self.traindf["spectrogram_id"].astype(np.int64).to_numpy()
        self._r = self.traindf["r"].astype(np.int32).to_numpy()
        if self.mode != "test":
            self._y = self.traindf[targets].to_numpy(dtype=np.float32, copy=True)
        else:
            self._y = None

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
        r = int(self._r[idx])
        spec_id = int(self._spec_ids[idx])
        eeg_id = int(self._eeg_ids[idx])

        X = np.empty((128, 256, 8), dtype="float32")
        X[:, :, :4] = self._get_processed_spec4(spec_id, r)
        X[:, :, 4:] = self.eeg[eeg_id]

        x_t = torch.from_numpy(X).permute(2, 0, 1).contiguous()  # (8,128,256)
        x = _build_x(x_t)

        if self.mode != "test":
            y = self._y[idx].astype(np.float32, copy=False)
            s = float(np.sum(y))
            if not np.isfinite(s) or s <= 0:
                y = np.ones(6, dtype=np.float32) / 6.0
            else:
                y = y / s
        else:
            y = np.zeros(6, dtype="float32")

        return {"data": x, "target": y}




## === cell 8
def _process_one_sid(args: Tuple[int, str]) -> Tuple[int, np.ndarray]:
    sid, base_dir = args
    p = os.path.join(base_dir, f"{int(sid)}.parquet")
    spec = pd.read_parquet(p).to_numpy()

    r = max(
        0, (spec.shape[0] - 300) // 2
    )  # matches dataset's 'r' computation for test (center crop)
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

    return int(sid), Xspec.astype(np.float32, copy=False)


def precompute_test_specproc_cache(
    test_df: pd.DataFrame, disk_cache: SpecProcDiskCache
):
    sids = test_df["spectrogram_id"].astype(np.int64).unique()
    missing = []
    for sid in sids.tolist():
        if disk_cache.get(int(sid)) is None:
            missing.append(int(sid))

    if not missing:
        return

    nproc = min(max(1, (os.cpu_count() or 2) - 1), 8)
    ctx = _mp_ctx()
    with ctx.Pool(processes=nproc) as pool:
        for sid, xspec in tqdm(
            pool.imap_unordered(
                _process_one_sid, [(int(sid), paths.test_spec) for sid in missing]
            ),
            total=len(missing),
            desc="Precomputing processed test spectrograms (mp)",
        ):
            disk_cache.set(int(sid), xspec)

    disk_cache.flush()


precompute_test_specproc_cache(test_df, _SPEC_CACHE_TEST)



## === cell 9
customdataset = CustomDataset(
    test_df, config, mode="test", specproc_cache=_SPEC_CACHE_TEST
)




## === cell 10
def _seed_worker(worker_id):
    seed = 42 + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)




## === cell 11
nw = min(8, (os.cpu_count() or 2))
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




## === cell 12
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




## === cell 13
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




## === cell 14
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



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/11822994.py in <cell line: 0>()
      7     model.load_state_dict(sd, strict=True)
      8 
----> 9 prediction_dict = inference_function(test_loader, model, train_device)
     10 predictions = prediction_dict["predictions"]
     11 

/tmp/ipykernel_55/3326450697.py in inference_function(test_loader, model, device)
      6         for batch in tqdm(test_loader, desc="Inference"):
      7             x = batch["data"].to(device, non_blocking=True)
----> 8             ypred = model(x)
      9             ypred = softmax(ypred)
     10             preds.append(ypred.detach().cpu())

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

/tmp/ipykernel_55/1758622680.py in forward(self, x)
     16 
     17     def forward(self, x):
---> 18         x = self.features(x)
     19         x = self.customlayer(x)
     20         return x

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

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/container.py in forward(self, input)
    248     def forward(self, input):
    249         for module in self:
--> 250             input = module(input)
    251         return input
    252 

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

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/container.py in forward(self, input)
    248     def forward(self, input):
    249         for module in self:
--> 250             input = module(input)
    251         return input
    252 

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

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/conv.py in forward(self, input)
    552 
    553     def forward(self, input: Tensor) -> Tensor:
--> 554         return self._conv_forward(input, self.weight, self.bias)
    555 
    556 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/conv.py in _conv_forward(self, input, weight, bias)
    547                 self.groups,
    548             )
--> 549         return F.conv2d(
    550             input, weight, bias, self.stride, self.padding, self.dilation, self.groups
    551         )

RuntimeError: Given groups=1, weight of size [32, 3, 3, 3], expected input[32, 1536, 4, 128] to have 3 channels, but got 1536 channels instead

## === cell 15
predictions



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1633991585.py in <cell line: 0>()
----> 1 predictions
      2 

NameError: name 'predictions' is not defined

## === cell 16
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

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1160474943.py in <cell line: 0>()
      9 
     10 sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
---> 11 sub[TARGETS] = predictions
     12 
     13 row_sums = sub[TARGETS].sum(axis=1).values

NameError: name 'predictions' is not defined
