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
cudf-polars-cu12==25.6.0
geopandas==0.14.4
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

# 5. Code solution

## === cell 0
from tqdm.auto import tqdm
import pandas as pd
import polars as pl
import numpy as np
import argparse
import os
import glob
import sys
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from typing import Optional, Tuple, List, Dict, Any



## === cell 1
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

DATA_DIR = "/kaggle/input/hms-harmful-brain-activity-classification/"
SPEC_DIR = os.path.join(DATA_DIR, "test_spectrograms/")
EEG_DIR = os.path.join(DATA_DIR, "test_eegs/")

df_train = pl.read_csv(os.path.join(DATA_DIR, "train.csv"))
df_test = pl.read_csv(os.path.join(DATA_DIR, "test.csv"))
sample_submission = pl.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

df_test.head()



## === cell 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True




## === cell 3
def load_model(path: str, model: nn.Module) -> nn.Module:
    ckpt = torch.load(path, map_location=torch.device("cpu"))
    state = (
        ckpt["model_state_dict"]
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt
        else ckpt
    )
    model.load_state_dict(state, strict=True)
    return model




## === cell 4
try:
    sys.path.append("/kaggle/input/hms-models/")
    from spc_cnn_att import SpectrogramCnnModel as SpcModel  # type: ignore

    HAS_EXTERNAL_MODEL = True
except Exception:
    HAS_EXTERNAL_MODEL = False


class FallbackSpcModel(nn.Module):
    def __init__(self, n_classes: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(self.net(x), dim=1)


SpcModel = SpcModel if HAS_EXTERNAL_MODEL else FallbackSpcModel
print("External model available:", HAS_EXTERNAL_MODEL)



## === cell 5
models_eeg_spc = []

model_glob = "/kaggle/input/hms-models/final_eeg_spec_kaggle/final_eeg_spec_kaggle/*"
model_paths = sorted(glob.glob(model_glob))

torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False

if HAS_EXTERNAL_MODEL and len(model_paths) > 0:
    for model_path in model_paths:
        model = SpcModel()
        print("Loading:", model_path)
        model = load_model(model_path, model)
        model = model.to(device)
        model.eval()
        models_eeg_spc.append(model)
else:
    model = SpcModel().to(device)
    model.eval()
    models_eeg_spc.append(model)
    if not HAS_EXTERNAL_MODEL:
        print(
            "WARNING: Using fallback model because spc_cnn_att / hms-models is not available."
        )
    else:
        print(
            "WARNING: Using fallback model because no model files found at:", model_glob
        )

if device.type == "cuda":
    for i in range(len(models_eeg_spc)):
        models_eeg_spc[i] = models_eeg_spc[i].to(memory_format=torch.channels_last)

size = len(models_eeg_spc)
size




## === cell 6
def MAD(signal, axis=-1):
    """Compute the robust standard deviation (MAD) of a signal."""
    median = np.median(signal, axis=axis, keepdims=True)
    absolute_deviations = np.abs(signal - median)
    median_absolute_deviation = np.median(absolute_deviations, axis=axis, keepdims=True)
    scale_factor = 1.4826
    robust_std = median_absolute_deviation * scale_factor
    return robust_std


from scipy.signal import butter, lfilter, lfilter_zi

_FS = 200
_CUTOFF = np.array([0.25, 40.0], dtype=np.float64)
_ORDER = 5
_BTYPE = "bandpass"
_B_BANDPASS, _A_BANDPASS = butter(
    N=_ORDER, Wn=_CUTOFF / (0.5 * _FS), btype=_BTYPE, analog=False
)
_ZI_BANDPASS = lfilter_zi(_B_BANDPASS, _A_BANDPASS).astype(np.float64, copy=False)


def butter_filter(eeg_data, fs=200, cutoff_freq=22, order=4, btype="lowpass"):
    b, a = butter(N=order, Wn=cutoff_freq / (0.5 * fs), btype=btype, analog=False)
    return lfilter(b, a, eeg_data)


def _filtfilt_fast_nd(
    b: np.ndarray, a: np.ndarray, x: np.ndarray, zi: np.ndarray
) -> np.ndarray:
    if x.dtype != np.float64:
        xd = x.astype(np.float64, copy=False)
    else:
        xd = x

    zi2 = (zi[None, :] * xd[:, 0:1]).astype(np.float64, copy=False)
    y, _ = lfilter(b, a, xd, axis=-1, zi=zi2)
    y = y[:, ::-1]
    zi2 = (zi[None, :] * y[:, 0:1]).astype(np.float64, copy=False)
    y, _ = lfilter(b, a, y, axis=-1, zi=zi2)
    return y[:, ::-1]


def butter_filter_bandpass_cached(eeg_data: np.ndarray) -> np.ndarray:
    if eeg_data.ndim != 1:
        raise ValueError("butter_filter_bandpass_cached expects 1D array.")
    y = _filtfilt_fast_nd(_B_BANDPASS, _A_BANDPASS, eeg_data[None, :], _ZI_BANDPASS)
    return y[0]




## === cell 7
import torchaudio
from torchaudio.transforms import Spectrogram as _Spectrogram

n_fft = 800
win_length = 256
hop_length = 44

spectrogram = _Spectrogram(
    n_fft=n_fft, win_length=win_length, hop_length=hop_length, power=None
).to(torch.device("cpu"))

_SPEC_MIN = float(math.exp(-4))
_SPEC_MAX = float(math.exp(7))


@torch.no_grad()
def compute_spec(chain):
    chain = torch.as_tensor(chain, dtype=torch.float32, device="cpu")
    chain = spectrogram(chain)
    chain = chain[:, :, 2:98]
    chain = torch.abs(chain) / 15.0
    chain = torch.log(torch.clamp(chain, _SPEC_MIN, _SPEC_MAX))
    chain = chain.mean(axis=1)
    return chain.cpu().numpy()


def compute_spec_eeg(a, b) -> np.ndarray:
    return butter_filter_bandpass_cached(a - b)


_LEADS = [
    "Fp1",
    "Fp2",
    "Fz",
    "Cz",
    "Pz",
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

_LEAD2IDX = {k: i for i, k in enumerate(_LEADS)}
_IDX = _LEAD2IDX  # alias

_PAIR_IDXS = np.array(
    [
        (_IDX["Fp1"], _IDX["F7"]),
        (_IDX["F7"], _IDX["T3"]),
        (_IDX["T3"], _IDX["T5"]),
        (_IDX["T5"], _IDX["O1"]),
        (_IDX["Fp1"], _IDX["F3"]),
        (_IDX["F3"], _IDX["C3"]),
        (_IDX["C3"], _IDX["P3"]),
        (_IDX["P3"], _IDX["O1"]),
        (_IDX["Fp2"], _IDX["F4"]),
        (_IDX["F4"], _IDX["C4"]),
        (_IDX["C4"], _IDX["P4"]),
        (_IDX["P4"], _IDX["O2"]),
        (_IDX["Fp2"], _IDX["F8"]),
        (_IDX["F8"], _IDX["T4"]),
        (_IDX["T4"], _IDX["T6"]),
        (_IDX["T6"], _IDX["O2"]),
    ],
    dtype=np.int64,
)
_CHAIN_GROUPS = ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15))

_CHAIN0 = np.fromiter(_CHAIN_GROUPS[0], dtype=np.int64)
_CHAIN1 = np.fromiter(_CHAIN_GROUPS[1], dtype=np.int64)
_CHAIN2 = np.fromiter(_CHAIN_GROUPS[2], dtype=np.int64)
_CHAIN3 = np.fromiter(_CHAIN_GROUPS[3], dtype=np.int64)


def compute_spec_chain_from_matrix(mat_19xt: np.ndarray) -> np.ndarray:
    a_idx = _PAIR_IDXS[:, 0]
    b_idx = _PAIR_IDXS[:, 1]
    diffs = mat_19xt[a_idx] - mat_19xt[b_idx]  # (16, T)
    diffs = _filtfilt_fast_nd(_B_BANDPASS, _A_BANDPASS, diffs, _ZI_BANDPASS)  # (16, T)

    chain = np.stack(
        [
            diffs[_CHAIN0],
            diffs[_CHAIN1],
            diffs[_CHAIN2],
            diffs[_CHAIN3],
        ],
        axis=0,
    )  # (4,4,T)

    mads = MAD(chain, axis=-1)
    mads = np.median(mads.reshape(-1))
    chain = chain / (mads + 1e-5)

    chain = compute_spec(chain)
    return chain


def compute_spec_chain_from_arrays(arrs: Dict[str, np.ndarray]) -> np.ndarray:
    cols = [arrs[k] for k in _LEADS]
    mat = np.stack(cols, axis=0)
    return compute_spec_chain_from_matrix(mat)


def compute_spec_chain(df_eeg: pl.DataFrame) -> np.ndarray:
    arrs = {k: df_eeg[k].to_numpy() for k in _LEADS}
    return compute_spec_chain_from_arrays(arrs)


_ARROW_COLS = _LEADS

try:
    import pyarrow.parquet as pq

    _HAS_PYARROW = True
except Exception:
    _HAS_PYARROW = False


def compute_spec_from_file(filepath: str) -> np.ndarray:
    """
    Bugfix: pyarrow.Table in this environment may not implement .to_numpy().
    Use a version-safe conversion via .to_pandas().to_numpy(), keeping same data/columns.
    """
    if _HAS_PYARROW:
        table = pq.read_table(filepath, columns=_ARROW_COLS, memory_map=True)
        mat = table.to_pandas().to_numpy(copy=False).T
    else:
        lf = pl.scan_parquet(filepath).select(_ARROW_COLS)
        df = lf.collect(streaming=True)
        mat = df.to_numpy().T  # (19, T)

    if mat.dtype != np.float32:
        mat = mat.astype(np.float32, copy=False)
    np.nan_to_num(mat, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    chain = compute_spec_chain_from_matrix(mat)
    return chain




## === cell 8
def _fix_to_4x96x224(x: np.ndarray) -> np.ndarray:
    """
    Bugfix: compute_spec_from_file() produces variable time-length.
    The model expects (4,96,224). We pad/center-crop in time to 224 and ensure freq=96.
    This keeps core logic (same spectrogram computation) and only fixes sizing.
    """
    if x.ndim != 3:
        raise ValueError(f"Unexpected eeg_spec ndim={x.ndim}, shape={x.shape}")

    F = x.shape[-1]
    if F != 96:
        if F > 96:
            x = x[..., :96]
        else:
            pad = 96 - F
            x = np.pad(x, ((0, 0), (0, 0), (0, pad)), mode="edge")

    T = x.shape[1]
    target_T = 224
    if T > target_T:
        start = (T - target_T) // 2
        x = x[:, start : start + target_T, :]
    elif T < target_T:
        pad_total = target_T - T
        pad_l = pad_total // 2
        pad_r = pad_total - pad_l
        x = np.pad(x, ((0, 0), (pad_l, pad_r), (0, 0)), mode="edge")

    x = np.transpose(x, (0, 2, 1))
    return x.astype(np.float32, copy=False)


def proc_eeg_spec(x: np.ndarray) -> np.ndarray:
    x = x[:, :, 2:-2]
    np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    x = x + 1
    x = _fix_to_4x96x224(x)
    return x




## === cell 9
LABELS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]



## === cell 10
_WORKER_SPECTROGRAM = None


def _worker_init_fn(worker_id: int):
    global spectrogram, _WORKER_SPECTROGRAM
    seed = 0 + worker_id
    np.random.seed(seed)
    torch.manual_seed(seed)

    _WORKER_SPECTROGRAM = _Spectrogram(
        n_fft=n_fft, win_length=win_length, hop_length=hop_length, power=None
    ).to(torch.device("cpu"))
    spectrogram = _WORKER_SPECTROGRAM  # compute_spec uses global 'spectrogram'


class TestEEGSpecDataset(Dataset):
    def __init__(self, df_test: pl.DataFrame, eeg_dir: str, spec_dir: str):
        self.eeg_ids = df_test["eeg_id"].to_numpy()
        self.spc_ids = df_test["spectrogram_id"].to_numpy()
        self.eeg_dir = eeg_dir
        self.spec_dir = spec_dir  # kept for structure parity (unused)

    def __len__(self):
        return int(self.eeg_ids.shape[0])

    def __getitem__(self, idx: int):
        eeg_id = int(self.eeg_ids[idx])
        spc_id = int(self.spc_ids[idx])

        eeg_filepath = os.path.join(self.eeg_dir, f"{eeg_id}.parquet")
        _ = os.path.join(self.spec_dir, f"{spc_id}.parquet")  # kept (unused)

        eeg_spec = compute_spec_from_file(eeg_filepath)
        eeg_spec = proc_eeg_spec(eeg_spec)  # (4,96,224) float32
        return eeg_id, eeg_spec


def _collate(batch):
    bsz = len(batch)
    eeg_ids = np.empty((bsz,), dtype=np.int64)
    x = np.empty((bsz, 4, 96, 224), dtype=np.float32)
    for i, (eid, spec) in enumerate(batch):
        eeg_ids[i] = eid
        x[i] = spec
    return eeg_ids, torch.from_numpy(x)


cpu_cnt = os.cpu_count() or 2

if _HAS_PYARROW:
    num_workers = 0
    bs = 64 if device.type == "cuda" else 8
else:
    if device.type == "cuda":
        num_workers = min(8, max(4, cpu_cnt // 2))
        bs = 64
    else:
        num_workers = min(4, max(0, cpu_cnt // 4))
        bs = 8

pin_memory = device.type == "cuda"

ds = TestEEGSpecDataset(df_test, EEG_DIR, SPEC_DIR)

dl = DataLoader(
    ds,
    batch_size=bs,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=(num_workers > 0),
    prefetch_factor=(4 if num_workers > 0 else None),
    collate_fn=_collate,
    worker_init_fn=_worker_init_fn if num_workers > 0 else None,
)



## === cell 11
all_preds = np.empty((len(ds), 6), dtype=np.float64)
all_eeg_ids = np.empty((len(ds),), dtype=np.int64)

offset = 0
with torch.inference_mode():
    for eeg_ids, x in tqdm(dl, total=len(dl)):
        if device.type == "cuda":
            x = x.to(device, non_blocking=True).contiguous(
                memory_format=torch.channels_last
            )
        else:
            x = x.to(device)

        probs_sum = torch.zeros((x.shape[0], 6), device=device, dtype=torch.float32)
        for model in models_eeg_spc:
            probs_sum.add_(model(x).exp())
        probs = probs_sum.mul_(1.0 / float(len(models_eeg_spc)))

        probs_np = probs.detach().cpu().numpy().astype(np.float64, copy=False)
        probs_np[~np.isfinite(probs_np)] = 0.0
        probs_np = np.clip(probs_np, 1e-12, None)
        probs_np = probs_np / probs_np.sum(axis=1, keepdims=True)

        bsz = probs_np.shape[0]
        all_preds[offset : offset + bsz] = probs_np
        all_eeg_ids[offset : offset + bsz] = eeg_ids
        offset += bsz

preds_final = all_preds
preds_final.shape



## === cell 12
test_eeg_ids = df_test["eeg_id"].to_numpy()
order = np.argsort(all_eeg_ids)
sorted_ids = all_eeg_ids[order]
sorted_preds = preds_final[order]

idx = np.searchsorted(sorted_ids, test_eeg_ids)
preds_aligned = sorted_preds[idx]

row_sums = preds_aligned.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
preds_aligned = preds_aligned / row_sums
preds_aligned = np.clip(preds_aligned, 1e-12, None)
preds_aligned = preds_aligned / preds_aligned.sum(axis=1, keepdims=True)

df_sub = pd.DataFrame({"eeg_id": test_eeg_ids})
df_sub[LABELS] = preds_aligned
df_sub = df_sub[["eeg_id"] + LABELS]
df_sub.to_csv("submission.csv", index=False)

print(df_sub.head())
print("Saved submission.csv with shape:", df_sub.shape)
print(
    "Row sums (min/max):",
    df_sub[LABELS].sum(axis=1).min(),
    df_sub[LABELS].sum(axis=1).max(),
)
