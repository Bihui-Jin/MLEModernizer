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

cudf-polars-cu12==25.6.0
geopandas==0.14.4
librosa==0.11.0
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
polars==1.25.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scipy==1.15.3
sklearn-pandas==2.2.0
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

0.3874020518640224

# 6. Current score

1.22283

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.4475) has done: 'I first make the script robust to the missing `/kaggle/input/hms-models/` dependency by providing a tiny fallback `EegModel` with the same “returns log-probabilities” interface, so inference can run end-to-end. Next, I fix the reshape crash by enforcing the expected fixed-length time axis (4×4×2500) via safe pad/crop right before the reshape, which preserves the existing preprocessing/model semantics. I also stop printing per-sample predictions (it time out) and ensure predictions are valid probabilities (non-negative, sum-to-1) so the submission passes Kaggle checks. Finally, I write a correct `submission.csv` with the required columns and row alignment.'
- What this solution (achieved 1.4301) has done: 'Your current score is far worse than the target (lower is better), and the dominant reason is that you’re effectively using an untrained fallback model for test-time inference when `/kaggle/input/hms-models/` isn’t available, producing near-uniform/poorly-informed probabilities. To move the score substantially toward the target while keeping the same preprocessing + model interface (log-probs -> exp -> normalize), I add a minimal “train-on-metadata” step that trains the same fallback `EegModel` on a small subset of `train_eegs/` using your existing `compute_eeg_from_file` features and KLDivLoss with soft labels derived from the vote columns. This preserves your core logic (same input tensor shape 4×4×2500, same forward returning log-probabilities, same inference pipeline), but replaces random weights with learned ones so predictions are meaningfully calibrated for the competition metric. I keep it lightweight (few epochs, capped number of training samples) to finish within 600s and still produce `submission.csv` in the correct format.'
- What this solution (achieved 1.40109) has done: 'Main bottlenecks are per-sample parquet reads + heavy SciPy `filtfilt` in preprocessing, plus Python-loop overhead in caching and stacking. I keep the exact same feature computation and model inference, but speed it up by (1) reading EEG parquet files with `pyarrow` into NumPy much faster than Polars for this small fixed column set, (2) replacing the slow generic `bin_array(...).mean(...)` with an equivalent reshape+pad implementation specialized for 1D downsampling by 4, (3) vectorizing chain construction to avoid repeated column-to-numpy conversions and redundant temporaries, and (4) making batching/inference more efficient (set `inference_mode`, enable cuDNN benchmarking, ensure pinned-memory path is used correctly). These changes preserve identical preprocessing semantics (same butterworth coefficients + filtfilt, same padding mode behavior, same normalization, same model outputs), only reducing overhead and constant factors to fit under 600 seconds.'
- What this solution (achieved 1.22283) has done: 'Your current score is much worse than the target (lower is better), so we should improve model quality while preserving your existing preprocessing and inference semantics. The biggest issue is that the fallback `EegModel` is extremely underpowered (global average pooling over a single conv), so even with your minimal training it can’t learn useful patterns; I minimally strengthen the fallback model while keeping the same input/output interface (4×4×2500 -> log-probs) and the same KLDivLoss training loop. I also make the fallback training deterministic and slightly more aligned to the leaderboard metric by adding a tiny “label smoothing via Dirichlet prior” (you already do alpha=0.5; we keep that) and adding gradient clipping for stability (doesn’t change the approach, just prevents occasional divergence). No changes are made to your EEG preprocessing, the log-prob -> exp -> normalize prediction pipeline, file paths, or submission format.'

# 9. Code solution

## === cell 0
from tqdm.auto import tqdm
import pandas as pd
import polars as pl
import numpy as np
import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import Optional, Tuple, List, Dict, Any



## === cell 1
DATA_DIR = "/kaggle/input/hms-harmful-brain-activity-classification/"
SPEC_DIR = os.path.join(DATA_DIR, "test_spectrograms/")
EEG_DIR = os.path.join(DATA_DIR, "test_eegs/")
TRAIN_EEG_DIR = os.path.join(DATA_DIR, "train_eegs/")

df_train = pl.read_csv(os.path.join(DATA_DIR, "train.csv"))
df_test = pl.read_csv(os.path.join(DATA_DIR, "test.csv"))
sample_submission = pl.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

df_test.head()



## === cell 2
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print()

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = (
        True  # safe for fixed input shapes; does not change math/outputs
    )
    torch.backends.cuda.matmul.allow_tf32 = False  # keep numerics stable
    torch.backends.cudnn.allow_tf32 = False




## === cell 3
def load_model(path: str, model: nn.Module) -> nn.Module:
    ckpt = torch.load(path, map_location=torch.device("cpu"))
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    model.load_state_dict(state, strict=False)
    return model




## === cell 4
import sys
import glob

try:
    sys.path.append("/kaggle/input/hms-models/")
    from eeg_cnn_rnn_w1 import EegModel  # type: ignore

    CurrModel = EegModel
    HAS_EXTERNAL_MODELS = True
except Exception as e:
    HAS_EXTERNAL_MODELS = False

    class EegModel(nn.Module):
        def __init__(self, n_classes: int = 6):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.head = nn.Linear(64, n_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            z = self.features(x).flatten(1)
            logits = self.head(z)
            return torch.log_softmax(logits, dim=-1)

    CurrModel = EegModel

print("External models available:", HAS_EXTERNAL_MODELS)



## === cell 5
ensembles = []

fold_dirs = ["/kaggle/input/hms-models/new_fold/send_kaggle/*"]

models = []
if HAS_EXTERNAL_MODELS:
    for fold_dir in fold_dirs:
        for fold_path in glob.glob(fold_dir):
            print("Loading:", fold_path)
            model = CurrModel()
            model = load_model(fold_path, model)
            model = model.to(device)
            models.append(model)

if len(models) == 0:
    model = CurrModel().to(device)
    models = [model]

len(models)



## === cell 6
import polars as pl
import librosa
import numpy as np

from scipy.ndimage import convolve

KERNEL = np.array([-1, -1, -1, 0, 1, 1, 1])


def compute_spec(eeg: np.ndarray) -> np.ndarray:
    spectrogram = librosa.stft(
        eeg,
        n_fft=1024,
        hop_length=39,
        win_length=256,
        window="hann",
        center=True,
        pad_mode="constant",
        out=None,
        dtype=None,
    )
    spectrogram = librosa.power_to_db(np.abs(spectrogram) ** 2, ref=np.max).astype(
        np.float32
    )
    spectrogram = (spectrogram + 80) / 80
    spectrogram = spectrogram**2
    return spectrogram[:256][::2, ::2]


def spec(eeg: np.ndarray) -> np.ndarray:
    s = compute_spec(eeg)
    eeg2 = convolve(eeg, KERNEL)
    s = s + compute_spec(eeg2)
    return s / 2


def compute_chain(df_eeg: pl.DataFrame) -> np.ndarray:
    Fp1 = df_eeg["Fp1"].to_numpy()
    Fp2 = df_eeg["Fp2"].to_numpy()
    F3 = df_eeg["F3"].to_numpy()
    F4 = df_eeg["F4"].to_numpy()
    F7 = df_eeg["F7"].to_numpy()
    F8 = df_eeg["F8"].to_numpy()
    C3 = df_eeg["C3"].to_numpy()
    C4 = df_eeg["C4"].to_numpy()
    P3 = df_eeg["P3"].to_numpy()
    P4 = df_eeg["P4"].to_numpy()
    T3 = df_eeg["T3"].to_numpy()
    T4 = df_eeg["T4"].to_numpy()
    T5 = df_eeg["T5"].to_numpy()
    T6 = df_eeg["T6"].to_numpy()
    O1 = df_eeg["O1"].to_numpy()
    O2 = df_eeg["O2"].to_numpy()

    ll = np.stack([(spec(Fp1 - F7), spec(F7 - T3), spec(T3 - T5), spec(T5 - O1))])
    lp = np.stack([(spec(Fp1 - F3), spec(F3 - C3), spec(C3 - P3), spec(P3 - O1))])
    rp = np.stack([(spec(Fp2 - F4), spec(F4 - C4), spec(C4 - P4), spec(P4 - O2))])
    rl = np.stack([(spec(Fp2 - F8), spec(F8 - T4), spec(T4 - T6), spec(T6 - O2))])

    chain = np.stack([ll, lp, rp, rl])[:, 0]
    return chain


def preprocess_chain(chain: np.ndarray) -> np.ndarray:
    return chain


def compute_spec_from_file(filepath: str) -> np.ndarray:
    df_eeg = pl.read_parquet(filepath).fill_null(0)
    chain = compute_chain(df_eeg)
    chain = preprocess_chain(chain)
    return chain




## === cell 7
import numpy as np
import math

from typing import Union, Tuple, List


def bin_array(
    array,
    bin_size,
    axis=-1,
    pad_dir="symmetric",
    mode="edge",
    return_padding=False,
    **padding_kwargs,
) -> Union[np.ndarray, Tuple[np.ndarray, List]]:
    if axis == -1:
        axis = array.ndim - 1

    curr_len = array.shape[axis]
    n_bins = math.ceil(curr_len / bin_size)
    new_len = n_bins * bin_size

    new_shape = list(array.shape)
    new_shape[axis] = n_bins
    new_shape.insert(axis + 1, bin_size)

    padding = [(0, 0)] * array.ndim
    if curr_len != new_len:
        if pad_dir == "left":
            pad_l = new_len - curr_len
            pad_r = 0
        elif pad_dir == "right":
            pad_l = 0
            pad_r = new_len - curr_len
        else:
            pad_l = (new_len - curr_len) // 2
            pad_r = (new_len - curr_len) - pad_l

        padding[axis] = (pad_l, pad_r)
        array = np.pad(array, padding, mode=mode, **padding_kwargs)

    array = array.reshape(new_shape)

    if return_padding:
        return array, padding
    return array




## === cell 8
from scipy.signal import welch
from scipy.stats import linregress
from scipy.signal import butter, filtfilt

import pyarrow.parquet as pq

_BUTTER_CACHE: Dict[
    Tuple[int, str, Tuple[float, ...]], Tuple[np.ndarray, np.ndarray]
] = {}


def butter_filter(eeg_data, fs=200, cutoff_freq=22, order=4, btype="lowpass"):
    key = (order, btype, tuple(np.atleast_1d(cutoff_freq).astype(float).tolist()))
    ba = _BUTTER_CACHE.get(key)
    if ba is None:
        b, a = butter(
            N=order, Wn=np.asarray(cutoff_freq) / (0.5 * fs), btype=btype, analog=False
        )
        _BUTTER_CACHE[key] = (b, a)
    else:
        b, a = ba
    return filtfilt(b, a, eeg_data)


def _bin_mean_1d_reflect_symmetric(x: np.ndarray, bin_size: int = 4) -> np.ndarray:
    n = x.shape[0]
    n_bins = (n + bin_size - 1) // bin_size
    new_len = n_bins * bin_size
    if new_len != n:
        pad_total = new_len - n
        pad_l = pad_total // 2
        pad_r = pad_total - pad_l
        x = np.pad(x, (pad_l, pad_r), mode="reflect")
    return x.reshape(n_bins, bin_size).mean(axis=1)


def compute_eeg(eeg: np.ndarray) -> np.ndarray:
    eeg = butter_filter(eeg, cutoff_freq=np.array([0.25, 50]), btype="bandpass")
    eeg = _bin_mean_1d_reflect_symmetric(eeg, bin_size=4)
    return eeg


_EEG_COLS = [
    "Fp1",
    "Fp2",
    "F3",
    "F4",
    "F7",
    "F8",
    "C3",
    "C4",
    "P3",
    "P4",
    "T3",
    "T4",
    "T5",
    "T6",
    "O1",
    "O2",
]
_EEG_COL2IDX = {c: i for i, c in enumerate(_EEG_COLS)}

_PAIR_A = np.array(
    [
        _EEG_COL2IDX["Fp1"],
        _EEG_COL2IDX["F7"],
        _EEG_COL2IDX["T3"],
        _EEG_COL2IDX["T5"],
        _EEG_COL2IDX["Fp1"],
        _EEG_COL2IDX["F3"],
        _EEG_COL2IDX["C3"],
        _EEG_COL2IDX["P3"],
        _EEG_COL2IDX["Fp2"],
        _EEG_COL2IDX["F4"],
        _EEG_COL2IDX["C4"],
        _EEG_COL2IDX["P4"],
        _EEG_COL2IDX["Fp2"],
        _EEG_COL2IDX["F8"],
        _EEG_COL2IDX["T4"],
        _EEG_COL2IDX["T6"],
    ],
    dtype=np.int64,
)
_PAIR_B = np.array(
    [
        _EEG_COL2IDX["F7"],
        _EEG_COL2IDX["T3"],
        _EEG_COL2IDX["T5"],
        _EEG_COL2IDX["O1"],
        _EEG_COL2IDX["F3"],
        _EEG_COL2IDX["C3"],
        _EEG_COL2IDX["P3"],
        _EEG_COL2IDX["O1"],
        _EEG_COL2IDX["F4"],
        _EEG_COL2IDX["C4"],
        _EEG_COL2IDX["P4"],
        _EEG_COL2IDX["O2"],
        _EEG_COL2IDX["F8"],
        _EEG_COL2IDX["T4"],
        _EEG_COL2IDX["T6"],
        _EEG_COL2IDX["O2"],
    ],
    dtype=np.int64,
)


def _read_eeg_matrix_parquet(filepath: str) -> np.ndarray:
    table = pq.read_table(filepath, columns=_EEG_COLS)
    mat = table.to_pandas(types_mapper=None).to_numpy(copy=False)
    mat = np.asarray(mat, dtype=np.float32)
    np.nan_to_num(mat, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return mat


def compute_eeg_chain(df_eeg: pl.DataFrame) -> np.ndarray:
    Fp1 = df_eeg["Fp1"].to_numpy()
    Fp2 = df_eeg["Fp2"].to_numpy()
    F3 = df_eeg["F3"].to_numpy()
    F4 = df_eeg["F4"].to_numpy()
    F7 = df_eeg["F7"].to_numpy()
    F8 = df_eeg["F8"].to_numpy()
    C3 = df_eeg["C3"].to_numpy()
    C4 = df_eeg["C4"].to_numpy()
    P3 = df_eeg["P3"].to_numpy()
    P4 = df_eeg["P4"].to_numpy()
    T3 = df_eeg["T3"].to_numpy()
    T4 = df_eeg["T4"].to_numpy()
    T5 = df_eeg["T5"].to_numpy()
    T6 = df_eeg["T6"].to_numpy()
    O1 = df_eeg["O1"].to_numpy()
    O2 = df_eeg["O2"].to_numpy()

    ll = np.stack(
        [
            (
                compute_eeg(Fp1 - F7),
                compute_eeg(F7 - T3),
                compute_eeg(T3 - T5),
                compute_eeg(T5 - O1),
            )
        ]
    )
    lp = np.stack(
        [
            (
                compute_eeg(Fp1 - F3),
                compute_eeg(F3 - C3),
                compute_eeg(C3 - P3),
                compute_eeg(P3 - O1),
            )
        ]
    )
    rp = np.stack(
        [
            (
                compute_eeg(Fp2 - F4),
                compute_eeg(F4 - C4),
                compute_eeg(C4 - P4),
                compute_eeg(P4 - O2),
            )
        ]
    )
    rl = np.stack(
        [
            (
                compute_eeg(Fp2 - F8),
                compute_eeg(F8 - T4),
                compute_eeg(T4 - T6),
                compute_eeg(T6 - O2),
            )
        ]
    )

    chain = np.stack([ll, lp, rp, rl])[:, 0]
    return chain


def compute_eeg_from_file(filepath: str) -> np.ndarray:
    mat = _read_eeg_matrix_parquet(filepath)  # [T, 16]
    diffs = mat[:, _PAIR_A] - mat[:, _PAIR_B]
    out = np.empty((16, (diffs.shape[0] + 3) // 4), dtype=np.float32)
    for i in range(16):
        out[i] = compute_eeg(diffs[:, i])
    return out.reshape(4, 4, -1)




## === cell 9
np.set_printoptions(formatter={"all": lambda x: f"{x:0.3f}"})


def _fix_length_last_axis(x: np.ndarray, target_len: int) -> np.ndarray:
    curr = x.shape[-1]
    if curr == target_len:
        return x
    if curr > target_len:
        return x[..., :target_len]
    pad = target_len - curr
    return np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, pad)], mode="edge")


def _preprocess_eeg_tensor_from_filepath(filepath: str) -> np.ndarray:
    x = compute_eeg_from_file(filepath)
    x = x.astype(np.float32, copy=False)
    x[np.isnan(x) | np.isinf(x)] = 0.0

    x = x - x.mean(axis=-1, keepdims=True)
    x = x / (np.mean(x.std(axis=-1)) + 1e-5)

    x = _fix_length_last_axis(x, 2_500)
    x = x.reshape(4, 4, 2_500)
    return x


_PREPROC_CACHE: Dict[int, np.ndarray] = {}


def _get_preprocessed_eeg_by_id(eeg_id: int, base_dir: str) -> np.ndarray:
    x = _PREPROC_CACHE.get(eeg_id)
    if x is None:
        filepath = os.path.join(base_dir, f"{eeg_id}.parquet")
        x = _preprocess_eeg_tensor_from_filepath(filepath)
        _PREPROC_CACHE[eeg_id] = x
    return x


@torch.inference_mode()
def gen_ensemble_pred_batch(models: List[nn.Module], eeg_ids: np.ndarray) -> np.ndarray:
    xs = np.stack(
        [_get_preprocessed_eeg_by_id(int(eid), EEG_DIR) for eid in eeg_ids], axis=0
    )  # (B,4,4,2500)
    xt = torch.from_numpy(xs)
    if torch.cuda.is_available():
        xt = xt.pin_memory()
    xt = xt.to(device=device, dtype=torch.float32, non_blocking=True)

    preds_accum = None
    for model in models:
        pred = model(xt).exp()  # model outputs log-probs
        if preds_accum is None:
            preds_accum = pred
        else:
            preds_accum = preds_accum + pred
    preds = (preds_accum / float(len(models))).detach().cpu().numpy()  # (B,6)

    preds = np.clip(preds, 1e-8, None)
    preds = preds / preds.sum(axis=1, keepdims=True)
    return preds


@torch.inference_mode()
def gen_ensemble_pred(models, df_row: pl.DataFrame) -> np.ndarray:
    eeg_id = int(df_row["eeg_id"].item())
    x = _get_preprocessed_eeg_by_id(eeg_id, EEG_DIR)
    xt = torch.from_numpy(x).unsqueeze(0)
    if torch.cuda.is_available():
        xt = xt.pin_memory()
    xt = xt.to(device=device, dtype=torch.float32, non_blocking=True)

    preds = []
    for model in models:
        pred = model(xt).exp()  # model outputs log-probs
        pred = pred.detach().cpu().numpy().reshape(-1)
        preds.append(pred)

    preds = np.mean(preds, axis=0)
    preds = np.clip(preds, 1e-8, None)
    preds = preds / preds.sum()
    return preds




## === cell 10
LABELS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]




## === cell 11
def _make_soft_targets_from_votes(votes: np.ndarray) -> np.ndarray:
    votes = votes.astype(np.float32, copy=False)
    votes = np.clip(votes, 0.0, None)
    s = votes.sum(axis=1, keepdims=True)
    y = votes / np.clip(s, 1e-8, None)
    y = np.clip(y, 1e-8, None)
    y = y / y.sum(axis=1, keepdims=True)
    return y


class _TrainDataset(torch.utils.data.Dataset):
    def __init__(self, eeg_ids: np.ndarray, targets: np.ndarray):
        self.eeg_ids = eeg_ids
        self.targets = targets

    def __len__(self) -> int:
        return len(self.eeg_ids)

    def __getitem__(self, idx: int):
        eeg_id = int(self.eeg_ids[idx])
        x = _get_preprocessed_eeg_by_id(eeg_id, TRAIN_EEG_DIR)
        y = self.targets[idx]
        return torch.from_numpy(x), torch.from_numpy(y)


def _train_fallback_model_if_needed(models: List[nn.Module]) -> List[nn.Module]:
    if HAS_EXTERNAL_MODELS:
        return models

    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    df = df_train.select(["eeg_id", "patient_id"] + LABELS).to_pandas()
    grp_votes = (
        df.groupby(["eeg_id", "patient_id"], sort=False)[LABELS].sum().reset_index()
    )
    eeg_ids_all = grp_votes["eeg_id"].to_numpy()
    patient_ids_all = grp_votes["patient_id"].to_numpy()
    votes_sum = grp_votes[LABELS].to_numpy(np.float32)

    alpha = 0.5
    y_all = votes_sum + alpha
    y_all = y_all / np.clip(y_all.sum(axis=1, keepdims=True), 1e-8, None)
    y_all = np.clip(y_all, 1e-8, None)
    y_all = y_all / y_all.sum(axis=1, keepdims=True)

    rs = np.random.RandomState(0)
    uniq_pat = pd.unique(patient_ids_all)
    rs.shuffle(uniq_pat)
    n_train_pat = int(0.9 * len(uniq_pat))
    train_pats = set(uniq_pat[:n_train_pat])
    train_mask = np.array([p in train_pats for p in patient_ids_all], dtype=bool)

    eeg_ids = eeg_ids_all[train_mask]
    y = y_all[train_mask]

    MAX_TRAIN = 4000
    if len(eeg_ids) > MAX_TRAIN:
        idx = rs.choice(len(eeg_ids), size=MAX_TRAIN, replace=False)
        eeg_ids = eeg_ids[idx]
        y = y[idx]

    ds = _TrainDataset(eeg_ids, y)
    nw = min(4, os.cpu_count() or 1)
    dl = DataLoader(
        ds,
        batch_size=32,
        shuffle=True,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(nw > 0),
        prefetch_factor=2,
    )

    model = models[0]
    model.train()
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    crit = nn.KLDivLoss(reduction="batchmean")

    EPOCHS = 4
    for _ in range(EPOCHS):
        for xb, yb in dl:
            xb = xb.to(device=device, dtype=torch.float32, non_blocking=True)
            yb = yb.to(device=device, dtype=torch.float32, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logp = model(xb)  # log-probs
            loss = crit(logp, yb)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            opt.step()

    return [model]


models = _train_fallback_model_if_needed(models)



## === cell 12
for m in models:
    m.eval()

eeg_ids_test = np.asarray(df_test["eeg_id"].to_list(), dtype=np.int64)

BATCH = 64
preds_final_list = []
for start in tqdm(range(0, len(eeg_ids_test), BATCH)):
    batch_ids = eeg_ids_test[start : start + BATCH]
    preds_b = gen_ensemble_pred_batch(models, batch_ids)
    preds_final_list.append(preds_b)

preds_final = np.concatenate(preds_final_list, axis=0).astype(np.float64, copy=False)
preds_final.shape



## === cell 13
df_sub = pd.DataFrame({"eeg_id": df_test["eeg_id"].to_list()})
df_sub[LABELS] = preds_final

vals = df_sub[LABELS].to_numpy(dtype=np.float64)
vals = np.clip(vals, 1e-12, None)
vals = vals / vals.sum(axis=1, keepdims=True)
df_sub[LABELS] = vals

df_sub.to_csv("submission.csv", index=False)
df_sub.head()
