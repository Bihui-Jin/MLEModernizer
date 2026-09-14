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

0.5642164438714241

# 6. Current score

1.41934

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I fix the missing model-weights path so inference can run even when `/kaggle/input/resnet50` doesn’t exist, by falling back to a deterministic, competition-valid baseline probability distribution derived from the training vote priors. I also remove a couple of runtime blockers in the dataset (`targets` undefined, noisy `print(r)`, and a bad `.to(device)` on numpy targets) while preserving the same feature construction logic. Finally, I ensure predictions are the correct shape `(n_test, 6)` and are normalized to sum to 1 per row so the submission is accepted and produces `submission.csv`.'
- What this solution (achieved 1.41937) has done: 'Main bottlenecks are (1) building EEG-derived spectrograms for every test EEG upfront (1691× heavy STFT/mel work) and (2) repeated Parquet→pandas conversions inside per-sample feature building. To stay within 600s without changing model/training/inference semantics, the optimized script loads and caches test EEG specs and test spectrogram patches on-demand (LRU) so only actually-accessed items are computed/decoded, and replaces slow pandas conversions with direct PyArrow-to-NumPy paths. DataLoader settings are tuned to avoid multiprocessing overhead with PyArrow and to reuse caches efficiently, while preserving determinism and identical feature math. The model, weights ensembling, input tensor construction, and probability post-processing remain the same.'
- What this solution (achieved 1.41937) has done: 'Your current score (1.41937, lower-is-better) is far worse than the target (0.5642), and the biggest reason is that you’re usually falling back to a global vote-prior baseline because `/kaggle/input/resnet50` doesn’t exist, so the CNN never contributes meaningful signal. To move the score toward the target with minimal semantic change, I (1) auto-discover and load pretrained weight files from common Kaggle input locations (including your dataset folder), and only if none are found keep the existing prior fallback; (2) ensure inference uses the same `channels_last` + `autocast` settings deterministically to match how weights were likely trained; and (3) keep your existing feature construction, dataset, and submission formatting unchanged. These changes should improve score substantially (toward the target band) while preserving your core pipeline and staying within runtime constraints.'
- What this solution (achieved 1.41934) has done: 'Your score is far above the target (1.419 → 0.564, lower is better), and the main reason is that your “pretrained” branch still typically never finds compatible weights, so you submit the weak global-prior fallback. I make the smallest change that materially improves score: broaden weight discovery (recursive), then load only checkpoints whose tensors actually match your `Custommodel` state_dict (so you don’t silently “use” irrelevant files), and prefer the best-matching ones; if none match, keep your existing prior fallback. Additionally, I ensure test-time input uses `channels_last` consistently (matching your model conversion) and add a tiny, metric-safe probability smoothing (epsilon) only after ensembling to avoid overconfident zeros that can hurt KL. These changes preserve your feature construction, model architecture, and inference semantics while making it much more likely the real CNN predictions are used.'
- What this solution (achieved 1.41934) has done: 'Your current score is far worse than the target (1.419 → 0.564, lower is better), and the most likely cause is still that you’re submitting the weak global prior because no compatible weights are ever found. I make weight loading more tolerant to common checkpoint formats (e.g., `{'model_state_dict': ...}` and keys prefixed with `model.`) and relax the “near-exact match” filter to “best available match above a reasonable threshold”, so you actually use a real model when any relevant weights exist. I also ensure the model input uses contiguous `channels_last` (often required for best GPU throughput/consistency with how timm backbones are trained) and keep the same post-processing normalization so rows always sum to 1. These are minimal changes that preserve your feature construction, architecture, and inference semantics, but should materially reduce KL toward your target if any usable weights exist in the environment.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

import albumentations as A  # kept (import side-effects / compatibility)
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

from albumentations.pytorch import ToTensorV2  # kept
from glob import glob
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using", torch.cuda.device_count(), "GPU(s)")

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import pyarrow as pa
import pyarrow.parquet as pq




## === cell 1
class config:
    model = "resnet50d"
    epoch = 2  # unchanged (only used if no pretrained weights exist)
    lr = 1e-3
    batchsize = 32
    splits = 5
    momentum = 0.9
    MAX_GRAD_NORM = 1e7
    WEIGHT_DECAY = 0.01
    device = "cpu"
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
    """Denoise function."""
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def denoise(x: np.ndarray, wavelet: str = "haar", level: int = 1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1 / 0.6745) * maddest(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="hard") for i in coeff[1:])
    output = pywt.waverec(coeff, wavelet, mode="per")
    return output


_MEL_KW = dict(sr=200, n_fft=1024, n_mels=128, fmin=0, fmax=20)
_MEL_FB = librosa.filters.mel(**_MEL_KW).astype(np.float32)

_TORCH_MEL_FB: Optional[torch.Tensor] = None
_TORCH_WIN: Optional[torch.Tensor] = None


def _ensure_torch_audio_kernels(dev: torch.device):
    global _TORCH_MEL_FB, _TORCH_WIN
    if _TORCH_MEL_FB is None or _TORCH_MEL_FB.device != dev:
        _TORCH_MEL_FB = torch.from_numpy(_MEL_FB).to(device=dev)
    if _TORCH_WIN is None or _TORCH_WIN.device != dev:
        _TORCH_WIN = torch.hann_window(128, periodic=True, device=dev)


def _melspectrogram_fast_torch(
    y: np.ndarray, hop_length: int, win_length: int = 128
) -> np.ndarray:
    in_worker = multiprocessing.current_process().name != "MainProcess"
    if torch.cuda.is_available() and (not in_worker):
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")

    _ensure_torch_audio_kernels(dev)

    yt = torch.from_numpy(y.astype(np.float32, copy=False)).to(device=dev)
    pad = _MEL_KW["n_fft"] // 2
    yt = torch.nn.functional.pad(
        yt[None, None, :], (pad, pad), mode="reflect"
    ).squeeze()

    S = torch.stft(
        yt,
        n_fft=_MEL_KW["n_fft"],
        hop_length=int(hop_length),
        win_length=int(win_length),
        window=_TORCH_WIN,
        center=False,  # already padded above
        return_complex=True,
    )
    P = (S.abs() ** 2).to(dtype=torch.float32)  # (freq, time)
    mel = _TORCH_MEL_FB @ P  # (n_mels, time)
    return mel.detach().cpu().numpy()


def _read_eeg_parquet(parquet_path: str, columns: List[str]) -> np.ndarray:
    tbl = pq.read_table(parquet_path, columns=columns)
    cols_np = [
        tbl.column(i).to_numpy(zero_copy_only=False) for i in range(tbl.num_columns)
    ]
    arr = np.column_stack(cols_np)
    return arr


def spectrogram_from_eeg(parquet_path, display=False):
    cols = []
    for k in range(4):
        cols.extend(FEATS[k])
    cols = sorted(set(cols))

    eeg_arr = _read_eeg_parquet(parquet_path, columns=cols)  # (T, C)
    col_index = {c: i for i, c in enumerate(cols)}

    middle = (eeg_arr.shape[0] - 10_000) // 2
    eeg_arr = eeg_arr[middle : middle + 10_000]

    img = np.zeros((128, 256, 4), dtype="float32")

    if display:
        plt.figure(figsize=(10, 7))
    signals = []
    eps = 1e-10
    for k in range(4):
        COLS = FEATS[k]

        for kk in range(4):
            a = eeg_arr[:, col_index[COLS[kk]]]
            b = eeg_arr[:, col_index[COLS[kk + 1]]]
            x = a - b

            m = np.nanmean(x)
            if np.isnan(x).mean() < 1:
                x = np.nan_to_num(x, nan=m)
            else:
                x = np.zeros_like(x)

            if USE_WAVELET:
                x = denoise(x, wavelet=USE_WAVELET)
            signals.append(x)

            mel_spec = _melspectrogram_fast_torch(
                y=x.astype(np.float32, copy=False),
                hop_length=len(x) // 256,
                win_length=128,
            )

            width = (mel_spec.shape[1] // 32) * 32
            mel_spec = mel_spec[:, :width].astype(np.float32, copy=False)

            mel_spec_db = np.log(np.maximum(mel_spec, eps)).astype(
                np.float32, copy=False
            )
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
test_df = pd.read_csv(paths.test_csv)
print(f"Test dataframe shape is: {test_df.shape}")
test_df.head()



## === cell 4
try:
    _MP_CTX = multiprocessing.get_context("spawn")
except Exception:
    _MP_CTX = None


def _mp_init_worker():
    pass


class _LRUEEGSpecCache:
    def __init__(self, eeg_dir: str, max_items: int = 128):
        self.eeg_dir = eeg_dir
        self.max_items = int(max_items)
        self._cache: Dict[int, np.ndarray] = {}
        self._order: List[int] = []

    def get(self, eeg_id: int) -> np.ndarray:
        eid = int(eeg_id)
        arr = self._cache.get(eid, None)
        if arr is not None:
            try:
                self._order.remove(eid)
            except ValueError:
                pass
            self._order.append(eid)
            return arr

        fp = os.path.join(self.eeg_dir, f"{eid}.parquet")
        if not os.path.exists(fp):
            arr = np.zeros((128, 256, 4), dtype=np.float32)
        else:
            arr = np.asarray(spectrogram_from_eeg(fp), dtype=np.float32)

        self._cache[eid] = arr
        self._order.append(eid)
        if len(self._order) > self.max_items:
            old = self._order.pop(0)
            self._cache.pop(old, None)
        return arr


needed_test_eeg_ids = np.unique(test_df["eeg_id"].to_numpy(np.int64, copy=False))
all_eegs: Dict[int, np.ndarray] = (
    {}
)  # kept for compatibility; left empty (on-demand cache used instead)
test_eeg_cache = _LRUEEGSpecCache(paths.test_eeg, max_items=256)

print(
    "Will compute/load test EEG specs on-demand (LRU), not upfront. Unique eeg_id:",
    len(needed_test_eeg_ids),
)



## === cell 5
all_spectrograms: Dict[int, np.ndarray] = {}
print("Skipping eager test spectrogram preload; will use on-demand LRU cache.")




## === cell 6
def _discover_weight_files() -> List[str]:
    candidates = [
        "/kaggle/input/resnet50",
        "/kaggle/input/hms-harmful-brain-activity-classification/resnet50",
        "/kaggle/input/hms-harmful-brain-activity-classification",
        "/kaggle/input",
    ]
    exts = (".pt", ".pth", ".bin")
    found: List[str] = []

    for base in candidates:
        if not os.path.exists(base):
            continue
        for pattern in (os.path.join(base, f"*{ext}") for ext in exts):
            found.extend([fp for fp in glob(pattern) if os.path.isfile(fp)])
        for pattern in (os.path.join(base, "*", f"*{ext}") for ext in exts):
            found.extend([fp for fp in glob(pattern) if os.path.isfile(fp)])
        for pattern in (os.path.join(base, "*", "*", f"*{ext}") for ext in exts):
            found.extend([fp for fp in glob(pattern) if os.path.isfile(fp)])
        for pattern in (os.path.join(base, "*", "*", "*", f"*{ext}") for ext in exts):
            found.extend([fp for fp in glob(pattern) if os.path.isfile(fp)])

    found = sorted(set(found))
    return found


weight_files = _discover_weight_files()
print("Discovered weight files:", len(weight_files))
if len(weight_files) > 0:
    print("First few weight files:", weight_files[:10])




## === cell 7
def _safe_patch_100x256(img2d: np.ndarray) -> np.ndarray:
    """
    Ensure the spectrogram patch we assign into X[14:-14,:,region]
    is always shape (100, 256). Uses linear interpolation (separable)
    but fully vectorized for speed.
    """
    img2d = np.asarray(img2d, dtype=np.float32)
    img2d = np.nan_to_num(img2d, nan=0.0, posinf=0.0, neginf=0.0)

    if img2d.ndim != 2:
        img2d = img2d.reshape(img2d.shape[0], -1)

    target_h, target_w = 100, 256
    h, w = img2d.shape

    if h <= 1 or w <= 1:
        return np.zeros((target_h, target_w), dtype=np.float32)
    if (h, w) == (target_h, target_w):
        return img2d

    x_new = np.linspace(0, w - 1, target_w, dtype=np.float32)
    x0 = np.floor(x_new).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    wx = x_new - x0
    tmp = img2d[:, x0] * (1.0 - wx)[None, :] + img2d[:, x1] * wx[None, :]

    y_new = np.linspace(0, h - 1, target_h, dtype=np.float32)
    y0 = np.floor(y_new).astype(np.int32)
    y1 = np.minimum(y0 + 1, h - 1)
    wy = y_new - y0
    out = tmp[y0, :] * (1.0 - wy)[:, None] + tmp[y1, :] * wy[:, None]

    return out.astype(np.float32, copy=False)




## === cell 8
class _LRUSpecArrayCache:
    def __init__(self, spec_dir: str, max_items: int = 128):
        self.spec_dir = spec_dir
        self.max_items = int(max_items)
        self._cache: Dict[int, np.ndarray] = {}
        self._order: List[int] = []

    def get(self, spec_id: int) -> Optional[np.ndarray]:
        sid = int(spec_id)
        arr = self._cache.get(sid, None)
        if arr is not None:
            try:
                self._order.remove(sid)
            except ValueError:
                pass
            self._order.append(sid)
            return arr

        fp = os.path.join(self.spec_dir, f"{sid}.parquet")
        if not os.path.exists(fp):
            return None

        tbl = pq.read_table(fp)
        cols_np = [
            tbl.column(i).to_numpy(zero_copy_only=False) for i in range(tbl.num_columns)
        ]
        arr = np.column_stack(cols_np)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)

        self._cache[sid] = arr
        self._order.append(sid)
        if len(self._order) > self.max_items:
            old = self._order.pop(0)
            self._cache.pop(old, None)
        return arr


class SpecPatchCache:
    def __init__(
        self,
        specs: Optional[Dict[int, np.ndarray]] = None,
        spec_dir: Optional[str] = None,
        max_spec_items: int = 128,
    ):
        self.specs = specs if specs is not None else {}
        self._cache: Dict[Tuple[int, int], np.ndarray] = {}
        self._lru = (
            _LRUSpecArrayCache(spec_dir, max_items=max_spec_items) if spec_dir else None
        )

    def _get_spec_arr(self, spec_id: int) -> Optional[np.ndarray]:
        sid = int(spec_id)
        if self.specs is not None and sid in self.specs:
            return self.specs[sid]
        if self._lru is not None:
            return self._lru.get(sid)
        return None

    def get_patches(self, spec_id: int, r: int) -> np.ndarray:
        key = (int(spec_id), int(r))
        out = self._cache.get(key, None)
        if out is not None:
            return out

        spec_arr = self._get_spec_arr(int(spec_id))
        patches = np.zeros((4, 100, 256), dtype=np.float32)
        if spec_arr is None:
            self._cache[key] = patches
            return patches

        t_len = spec_arr.shape[0]
        r0 = int(np.clip(r, 0, max(0, t_len - 300)))
        ep = 1e-6

        for region in range(4):
            blk = spec_arr[
                r0 : r0 + 300, region * 100 : (region + 1) * 100
            ]  # (300,100)
            img = blk.T  # (100,300)
            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)
            mu = np.nanmean(img)
            std = np.nanstd(img)
            img = (img - mu) / (std + ep)
            img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

            if img.shape[1] >= 256:
                start = (img.shape[1] - 256) // 2
                patch = img[:, start : start + 256]
            else:
                patch = _safe_patch_100x256(img)

            patches[region] = (patch / 2.0).astype(np.float32, copy=False)

        self._cache[key] = patches
        return patches


class CustomDataset:
    def __init__(
        self,
        traindf: pd.DataFrame,
        config,
        mode: str = "train",
        specs: Optional[Dict[int, np.ndarray]] = None,
        eegs: Optional[Dict[int, np.ndarray]] = None,
        patch_cache: Optional[SpecPatchCache] = None,
        eeg_cache: Optional["_LRUEEGSpecCache"] = None,
    ):
        self.traindf = traindf
        self.specs = specs if specs is not None else all_spectrograms
        self.eeg = eegs if eegs is not None else all_eegs
        self._eeg_cache = eeg_cache  # may be None
        self.mode = mode

        self._eeg_id = traindf["eeg_id"].to_numpy(np.int64, copy=False)
        self._spec_id = traindf["spectrogram_id"].to_numpy(np.int64, copy=False)
        if mode != "test":
            self._r = (
                (
                    traindf["min"].to_numpy(np.int32, copy=False)
                    + traindf["max"].to_numpy(np.int32, copy=False)
                )
                // 4
            ).astype(np.int32)
            self._y = traindf[TARGETS].to_numpy(np.float32, copy=False)
        else:
            self._r = None
            self._y = None

        self._patch_cache = (
            patch_cache if patch_cache is not None else SpecPatchCache(self.specs)
        )

        self._data_cache: List[Optional[torch.Tensor]] = [None] * len(self.traindf)
        if self.mode != "test":
            for i in tqdm(
                range(len(self.traindf)), desc=f"Precomputing {mode} tensors"
            ):
                self._data_cache[i] = self._build_x3(i)

    def _get_eeg_img(self, eeg_id: int) -> np.ndarray:
        if self._eeg_cache is not None:
            return self._eeg_cache.get(int(eeg_id))
        img = self.eeg.get(int(eeg_id), None)
        if img is None:
            return np.zeros((128, 256, 4), dtype=np.float32)
        return img

    def _build_x3(self, idx: int) -> torch.Tensor:
        X = np.zeros((128, 256, 8), dtype=np.float32)
        r = 0 if self.mode == "test" else int(self._r[idx])

        eeg_id = int(self._eeg_id[idx])
        spec_id = int(self._spec_id[idx])

        img_eeg = self._get_eeg_img(eeg_id)
        patches = self._patch_cache.get_patches(spec_id, r)  # (4,100,256)

        X[14:-14, :, 0:4] = patches.transpose(1, 2, 0)  # (100,256,4)
        X[:, :, 4:8] = img_eeg

        spec = np.transpose(X[:, :, 0:4], (2, 0, 1)).reshape(512, 256)
        eeg = np.transpose(X[:, :, 4:8], (2, 0, 1)).reshape(512, 256)
        x2 = np.concatenate([spec, eeg], axis=1).astype(
            np.float32, copy=False
        )  # (512,512)
        x3 = np.broadcast_to(x2, (3,) + x2.shape).copy()
        return torch.from_numpy(x3)

    def __len__(self):
        return len(self.traindf)

    def __getitem__(self, idx):
        x3 = self._data_cache[idx]
        if x3 is None:
            x3 = self._build_x3(idx)
            self._data_cache[idx] = x3
        if self.mode != "test":
            y = self._y[idx]
        else:
            y = np.zeros(6, dtype=np.float32)
        return {"data": x3, "target": y}




## === cell 9
def _seed_worker(worker_id):
    seed = 42 + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


num_workers = 0
pin = torch.cuda.is_available()

test_patch_cache = SpecPatchCache(
    specs=None,  # do not use eager dict
    spec_dir=paths.test_spec,
    max_spec_items=256,
)

customdataset = CustomDataset(
    test_df,
    config,
    mode="test",
    specs=None,
    eegs=all_eegs,  # kept (empty), actual fetch via eeg_cache
    patch_cache=test_patch_cache,
    eeg_cache=test_eeg_cache,
)
_ = customdataset[0]

test_loader = DataLoader(
    customdataset,
    batch_size=config.batchsize,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin,
    persistent_workers=False,
)

train_loader = None
valid_loader = None
train_df_full = None



## === cell 10
TRAIN_IF_NO_PRETRAINED = False

if (False) and TRAIN_IF_NO_PRETRAINED:
    pass




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

    n = len(test_loader.dataset)
    out = np.empty((n, 6), dtype=np.float32)
    start = 0

    use_amp = torch.cuda.is_available()
    for batch in tqdm(test_loader, desc="Inference"):
        x = batch["data"].to(device, non_blocking=True)
        if torch.cuda.is_available():
            x = x.to(memory_format=torch.channels_last).contiguous(
                memory_format=torch.channels_last
            )

        with torch.no_grad():
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    ypred = model(x)
                    ypred = softmax(ypred)
            else:
                ypred = model(x)
                ypred = softmax(ypred)
        b = ypred.shape[0]
        out[start : start + b] = ypred.detach().cpu().numpy()
        start += b

    return {"predictions": out}


def train_one_epoch(train_loader, model, optimizer, device, scaler=None):
    model.train()
    for batch in tqdm(train_loader, desc="Train", leave=False):
        x = batch["data"].to(device, non_blocking=True)
        y = batch["target"]
        if not torch.is_tensor(y):
            y = torch.as_tensor(y, dtype=torch.float32, device=device)
        else:
            y = y.to(device=device, dtype=torch.float32, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
                log_probs = torch.log_softmax(logits, dim=1)
                loss = -(y * log_probs).sum(dim=1).mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            log_probs = torch.log_softmax(logits, dim=1)
            loss = -(y * log_probs).sum(dim=1).mean()
            loss.backward()
            optimizer.step()


def valid_kld(valid_loader, model, device):
    model.eval()
    losses = []
    for batch in tqdm(valid_loader, desc="Valid", leave=False):
        x = batch["data"].to(device, non_blocking=True)
        y = batch["target"]
        if not torch.is_tensor(y):
            y = torch.as_tensor(y, dtype=torch.float32, device=device)
        else:
            y = y.to(device=device, dtype=torch.float32, non_blocking=True)

        with torch.no_grad():
            logits = model(x)
            log_probs = torch.log_softmax(logits, dim=1)
            loss = -(y * log_probs).sum(dim=1).mean()
        losses.append(loss.detach().cpu().item())
    return float(np.mean(losses)) if len(losses) else float("nan")




## === cell 13
predictions = None


def _to_channels_last_if_cuda(model: nn.Module):
    if torch.cuda.is_available():
        return model.to(memory_format=torch.channels_last)
    return model


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for k in ("state_dict", "model", "model_state_dict", "net", "weights"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj


def _strip_prefixes(state: dict) -> dict:
    if not isinstance(state, dict):
        return state

    out = state
    if any(k.startswith("module.") for k in out.keys()):
        out = {k.replace("module.", "", 1): v for k, v in out.items()}

    if any(k.startswith("model.") for k in out.keys()):
        out = {k.replace("model.", "", 1): v for k, v in out.items()}

    if any(k.startswith("backbone.") for k in out.keys()):
        out = {k.replace("backbone.", "", 1): v for k, v in out.items()}

    return out


def _score_state_compat(model: nn.Module, state: dict) -> int:
    msd = model.state_dict()
    if not isinstance(state, dict):
        return 0
    score = 0
    for k, v in state.items():
        if k in msd and hasattr(v, "shape") and msd[k].shape == v.shape:
            score += 1
    return score


candidate_files = []
best_compat = 0
if len(weight_files) > 0:
    probe_model = Custommodel(config)
    msd = probe_model.state_dict()
    total_keys = len(msd)

    for fp in weight_files:
        try:
            dd = torch.load(fp, map_location="cpu")
            st = _strip_prefixes(_extract_state_dict(dd))
            compat = _score_state_compat(probe_model, st)
            if compat > 0:
                candidate_files.append((compat, fp))
                if compat > best_compat:
                    best_compat = compat
        except Exception:
            continue

    del probe_model, msd
    gc.collect()

candidate_files = sorted(candidate_files, key=lambda x: (-x[0], x[1]))
usable_weight_files: List[str] = []
if len(candidate_files) > 0:
    thr_abs = int(0.70 * max(1, best_compat))
    thr_rel = int(0.90 * max(1, best_compat))
    thr = max(thr_abs, thr_rel)
    usable_weight_files = [fp for compat, fp in candidate_files if compat >= thr][:5]

USE_PRETRAINED = len(usable_weight_files) > 0

print(
    "Weight candidates (any compat):", len(candidate_files), "best_compat:", best_compat
)
print("Usable compatible weight files:", len(usable_weight_files))
if USE_PRETRAINED:
    print("First few usable weight files:", usable_weight_files[:5])

if USE_PRETRAINED:
    preds_list = []
    for fp in usable_weight_files:
        dd = torch.load(fp, map_location="cpu")
        model = Custommodel(config)
        state = _strip_prefixes(_extract_state_dict(dd))
        model.load_state_dict(state, strict=True)
        model.to(device)
        model = _to_channels_last_if_cuda(model)
        with torch.inference_mode():
            pred_dict = inference_function(test_loader, model, device)
        preds_list.append(pred_dict["predictions"])
        del model, dd, state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    preds_arr = np.array(preds_list)
    if preds_arr.shape[0] == 5:
        w = np.array([2, 4, 5, 3, 1], dtype=np.float32)
        predictions = np.average(preds_arr, weights=w, axis=0)
    else:
        predictions = preds_arr.mean(axis=0)

else:
    if (
        TRAIN_IF_NO_PRETRAINED
        and (train_loader is not None)
        and (valid_loader is not None)
    ):
        model = Custommodel(config).to(device)
        model = _to_channels_last_if_cuda(model)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.WEIGHT_DECAY
        )
        scaler = torch.amp.GradScaler(
            enabled=(torch.cuda.is_available() and config.AMP)
        )

        for ep in range(config.epoch):
            train_one_epoch(train_loader, model, optimizer, device, scaler=scaler)
            vkld = valid_kld(valid_loader, model, device)
            print(f"Epoch {ep+1}/{config.epoch} valid_KL_like: {vkld:.5f}")

        with torch.inference_mode():
            pred_dict = inference_function(test_loader, model, device)
        predictions = pred_dict["predictions"]
    else:
        train_df_full = pd.read_csv(paths.train_csv, usecols=TARGETS)
        y_counts = train_df_full[TARGETS].values.astype(np.float64)
        prior = y_counts.sum(axis=0)
        prior = prior / np.clip(prior.sum(), 1.0, None)
        predictions = np.tile(prior[None, :], (len(test_df), 1)).astype(np.float32)

predictions = np.asarray(predictions, dtype=np.float64)
predictions = np.nan_to_num(
    predictions, nan=1.0 / 6.0, posinf=1.0 / 6.0, neginf=1.0 / 6.0
)
predictions = np.clip(predictions, 1e-12, 1.0)
eps = 1e-4
predictions = (1.0 - 6 * eps) * predictions + eps
predictions = predictions / predictions.sum(axis=1, keepdims=True)
assert predictions.shape == (len(test_df), 6)



## === cell 14
sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
sub[TARGETS] = predictions.astype(np.float32)
sub.to_csv("submission.csv", index=False)
print(f"Submission shape: {sub.shape}")
print(sub.head())
print(
    "Row-sum check (min/max):",
    sub[TARGETS].sum(axis=1).min(),
    sub[TARGETS].sum(axis=1).max(),
)
