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
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from tqdm.auto import tqdm
import pandas as pd
import polars as pl
import numpy as np

import torch
import torch.nn as nn

from typing import Optional, Tuple, List, Dict, Any



## === cell 1
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

torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = True

torch.set_num_threads(1)
torch.set_num_interop_threads(1)




## === cell 3
def load_model(path: str, model: nn.Module) -> nn.Module:
    obj = torch.load(path, map_location=torch.device("cpu"))
    state = (
        obj["model_state_dict"]
        if isinstance(obj, dict) and "model_state_dict" in obj
        else obj
    )
    model.load_state_dict(state, strict=True)
    return model




## === cell 4
import sys
import glob

HMS_MODELS_DIR = "/kaggle/input/hms-models/"
HAS_EXTERNAL_MODELS = os.path.isdir(HMS_MODELS_DIR)

if HAS_EXTERNAL_MODELS:
    sys.path.append(HMS_MODELS_DIR)
    from eeg_cnn_rnn_w1 import EegModel as EegModel0
    from eeg_cnn_rnn import EegModel as EegModel1
    from spc_cnn_rnn import SpectrogramModel
else:

    class _SimpleEEGNet(nn.Module):
        def __init__(self, in_ch: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, 6),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.log_softmax(self.net(x), dim=-1)

    class _SimpleSpecNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, 6),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.log_softmax(self.net(x), dim=-1)

    class EegModel0(_SimpleEEGNet):
        def __init__(self):
            super().__init__(in_ch=4)

    class EegModel1(_SimpleEEGNet):
        def __init__(self):
            super().__init__(in_ch=16)

    class SpectrogramModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = _SimpleSpecNet()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.ndim == 5 and x.shape[2] == 1:
                x = x.squeeze(2)  # (B,4,F,T)
            return self.backbone(x)




## === cell 5
models_0, models_1, models_2 = [], [], []

if HAS_EXTERNAL_MODELS:
    model_0_files = glob.glob("/kaggle/input/hms-models/new_fold/send_kaggle/*")
    model_2_files = glob.glob(
        "/kaggle/input/hms-models/send_kaggle_spec/send_kaggle_spec/*"
    )

    for fold_path in model_0_files:
        model = EegModel0()
        model = load_model(fold_path, model)
        model = model.to(device)
        models_0.append(model)

    for fold_path in glob.glob("/kaggle/input/hms-models/baseline_eeg_diff/*"):
        model_path = os.path.join(fold_path, "model_best_val_g10.pt")
        if os.path.exists(model_path):
            model = EegModel1()
            model = load_model(model_path, model)
            model = model.to(device)
            models_1.append(model)

    for fold_path in model_2_files:
        model = SpectrogramModel()
        model = load_model(fold_path, model)
        model = model.to(device)
        models_2.append(model)

if len(models_0) == 0:
    models_0 = [EegModel0().to(device)]
if len(models_1) == 0:
    models_1 = [EegModel1().to(device)]
if len(models_2) == 0:
    models_2 = [SpectrogramModel().to(device)]

for m in models_0 + models_1 + models_2:
    m.eval()

len(models_0), len(models_1), len(models_2)



## === cell 6
import numpy as _np
from scipy.signal import butter as _butter, filtfilt as _filtfilt

_BUTTER_CACHE: Dict[
    Tuple[int, int, str, Tuple[float, ...]], Tuple[_np.ndarray, _np.ndarray]
] = {}


def MAD(signal, **kwargs):
    """Compute the robust standard deviation (MAD) of a signal."""
    scale_factor = 1.4826
    absolute_deviations = np.abs(signal - np.median(signal, **kwargs))
    median_absolute_deviation = np.median(absolute_deviations, **kwargs)
    robust_std = median_absolute_deviation * scale_factor
    return robust_std


def _as_hashable_cutoff(cutoff_freq):
    if isinstance(cutoff_freq, np.ndarray):
        cutoff_freq = cutoff_freq.tolist()
    if isinstance(cutoff_freq, (list, tuple)):
        return tuple(float(x) for x in cutoff_freq)
    return (float(cutoff_freq),)


def butter_filter(eeg_data, fs=200, cutoff_freq=22, order=4, btype="lowpass"):
    key = (int(fs), int(order), str(btype), _as_hashable_cutoff(cutoff_freq))
    ba = _BUTTER_CACHE.get(key)
    if ba is None:
        Wn = np.array(key[3], dtype=np.float64) / (0.5 * fs)
        if Wn.size == 1:
            Wn = float(Wn[0])
        b, a = _butter(N=order, Wn=Wn, btype=btype, analog=False)
        _BUTTER_CACHE[key] = (b, a)
    else:
        b, a = ba
    return _filtfilt(b, a, eeg_data, axis=-1)




## === cell 7
import scipy
import scipy.signal

STRIDE = 100
WINDOW_SIZE = 500
NOVERLAP = WINDOW_SIZE - STRIDE
PARAMS = dict(
    fs=200,
    window=("tukey", 0.25),
    nperseg=WINDOW_SIZE,
    noverlap=NOVERLAP,
    nfft=1024,
    detrend="constant",
    return_onesided=True,
    scaling="density",
    axis=-1,
    mode="psd",
)

_FS = PARAMS["fs"]
_NPERSEG = PARAMS["nperseg"]
_NOVERLAP = PARAMS["noverlap"]
_NFFT = PARAMS["nfft"]
_STEP = _NPERSEG - _NOVERLAP
_ONESIDED = True

_WINDOW = scipy.signal.get_window(PARAMS["window"], _NPERSEG, fftbins=True).astype(
    np.float64, copy=False
)

_FREQS = np.fft.rfftfreq(_NFFT, d=1.0 / _FS)
_VALID_FREQ_MASK = (_FREQS >= 0.5) & (_FREQS <= 20.0)

_TSPEC_TARGET = scipy.signal.spectrogram(np.zeros(10000, dtype=np.float64), **PARAMS)[
    2
].shape[-1]

_U = float((_WINDOW * _WINDOW).sum())
_SCALE = 1.0 / (_FS * _U)
if _ONESIDED:
    _DOUBLE_BINS = np.ones((_NFFT // 2 + 1,), dtype=np.float64)
    if _NFFT % 2 == 0:  # even nfft: Nyquist present
        _DOUBLE_BINS[1:-1] = 2.0
    else:
        _DOUBLE_BINS[1:] = 2.0


def _fix_len_last_axis(x: np.ndarray, target_len: int) -> np.ndarray:
    x = np.asarray(x)
    cur = x.shape[-1]
    if cur == target_len:
        return x
    if cur > target_len:
        return x[..., :target_len]
    pad = target_len - cur
    pad_width = [(0, 0)] * (x.ndim - 1) + [(0, pad)]
    return np.pad(x, pad_width, mode="edge")


def _spectrogram_psd_vectorized(x: np.ndarray) -> np.ndarray:
    """
    Vectorized equivalent to scipy.signal.spectrogram(..., mode='psd') for this fixed PARAMS.
    """
    x = np.asarray(x, dtype=np.float64)
    T = x.shape[-1]
    if T < _NPERSEG:
        raise ValueError("Input shorter than nperseg; not expected in this pipeline.")
    nseg = 1 + (T - _NPERSEG) // _STEP
    shape = x.shape[:-1] + (nseg, _NPERSEG)
    strides = x.strides[:-1] + (_STEP * x.strides[-1], x.strides[-1])
    frames = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

    frames = frames - frames.mean(axis=-1, keepdims=True)
    frames = frames * _WINDOW

    Xf = np.fft.rfft(frames, n=_NFFT, axis=-1)
    Pxx = (Xf.real * Xf.real + Xf.imag * Xf.imag) * _SCALE
    if _ONESIDED:
        Pxx = Pxx * _DOUBLE_BINS

    Pxx = np.swapaxes(Pxx, -2, -1)
    return Pxx


def compute_spec(eeg: np.ndarray) -> np.ndarray:
    Sxx = _spectrogram_psd_vectorized(eeg)
    return Sxx[..., _VALID_FREQ_MASK, :]


_EEG_COLS = [
    "Fp1",
    "Fp2",
    "F7",
    "F8",
    "T3",
    "T4",
    "T5",
    "T6",
    "O1",
    "O2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
]
_COL2IDX = {c: i for i, c in enumerate(_EEG_COLS)}
_DIFF_PAIRS = [
    ("Fp1", "F7"),
    ("F7", "T3"),
    ("T3", "T5"),
    ("T5", "O1"),  # ll
    ("Fp1", "F3"),
    ("F3", "C3"),
    ("C3", "P3"),
    ("P3", "O1"),  # lp
    ("Fp2", "F4"),
    ("F4", "C4"),
    ("C4", "P4"),
    ("P4", "O2"),  # rp
    ("Fp2", "F8"),
    ("F8", "T4"),
    ("T4", "T6"),
    ("T6", "O2"),  # rl
]
_A_IDX = np.array([_COL2IDX[a] for a, b in _DIFF_PAIRS], dtype=np.int64)
_B_IDX = np.array([_COL2IDX[b] for a, b in _DIFF_PAIRS], dtype=np.int64)


def compute_spec_chain(df_eeg: "pl.DataFrame|np.ndarray") -> np.ndarray:
    if isinstance(df_eeg, np.ndarray):
        X = df_eeg  # (T,16)
    else:
        X = df_eeg.select(_EEG_COLS).to_numpy()  # (T,16)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = X.T  # (16,T)

    diffs = X[_A_IDX] - X[_B_IDX]  # (16,T)
    diffs = butter_filter(
        diffs, cutoff_freq=np.array([0.25, 40.0]), order=5, btype="bandpass"
    )
    chain = diffs.reshape(4, 4, -1)  # (4,4,T)

    mads = MAD(chain, axis=-1, keepdims=True)
    mads = np.median(mads.reshape(-1))
    chain = chain / (mads + 1e-5)

    Sxx = _spectrogram_psd_vectorized(chain)  # (4,4,F,Tspec)
    chain = Sxx[..., _VALID_FREQ_MASK, :]  # (4,4,Fvalid,Tspec_var)

    chain = np.log(chain.clip(np.exp(-4), np.exp(8)))
    chain = chain.mean(axis=1, keepdims=True)  # (4,1,Fvalid,Tspec_var)

    chain = _fix_len_last_axis(chain, _TSPEC_TARGET)
    return chain


def compute_spec_from_file(filepath: str) -> np.ndarray:
    df_eeg = pl.read_parquet(filepath, columns=_EEG_COLS).fill_null(0)
    return compute_spec_chain(df_eeg)




## === cell 8
import math
from typing import Union


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
    n_bins = (curr_len + bin_size - 1) // bin_size
    new_len = n_bins * bin_size

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

    new_shape = list(array.shape)
    new_shape[axis] = n_bins
    new_shape.insert(axis + 1, bin_size)
    array = array.reshape(new_shape)

    if return_padding:
        return array, padding
    return array




## === cell 9
TARGET_T = 2500


def _fix_len_1d(x: np.ndarray, target_len: int) -> np.ndarray:
    x = np.asarray(x)
    if x.shape[-1] == target_len:
        return x
    if x.shape[-1] > target_len:
        return x[..., :target_len]
    pad = target_len - x.shape[-1]
    return np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, pad)], mode="edge")


def compute_eeg(eeg: np.ndarray) -> np.ndarray:
    eeg = butter_filter(eeg, cutoff_freq=np.array([0.25, 50]), btype="bandpass")
    eeg = bin_array(eeg, bin_size=4, mode="reflect").mean(axis=-1)
    eeg = _fix_len_1d(eeg, TARGET_T)
    return eeg


def compute_eeg_chain(df_eeg: "pl.DataFrame|np.ndarray") -> np.ndarray:
    if isinstance(df_eeg, np.ndarray):
        X = df_eeg  # (T,16)
    else:
        X = df_eeg.select(_EEG_COLS).to_numpy()  # (T,16)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = X.T  # (16,T)

    diffs = X[_A_IDX] - X[_B_IDX]  # (16,T)
    diffs = butter_filter(diffs, cutoff_freq=np.array([0.25, 50]), btype="bandpass")
    diffs = bin_array(diffs, bin_size=4, mode="reflect").mean(axis=-1)
    diffs = _fix_len_1d(diffs, TARGET_T)
    chain = diffs.reshape(4, 4, TARGET_T)
    return chain


def compute_eeg_from_file(filepath: str) -> np.ndarray:
    df_eeg = pl.read_parquet(filepath, columns=_EEG_COLS).fill_null(0)
    return compute_eeg_chain(df_eeg)




## === cell 10
LABELS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

np.set_printoptions(formatter={"all": lambda x: f"{x:0.3f}"})


def proc_0(x):
    x = x.copy()
    x = x - x.mean(axis=-1, keepdims=True)
    x = x / (np.mean(x.std(axis=-1)) + 1e-5)
    x = x.reshape(4, 4, 2_500)
    return x


def proc_1(x):
    x = x.copy()
    x = x - x.mean(axis=-1, keepdims=True)
    x = x / (x.std(axis=-1, keepdims=True) + 1e-5)
    x = x.reshape(16, 1, 2_500)
    return x


def proc_2(spec):
    spec = spec - spec.mean(axis=(-1, -2), keepdims=True)
    spec = spec / (spec.std(axis=(-1, -2), keepdims=True) + 1e-5)
    return spec


@torch.no_grad()
def gen_ensemble_pred(eeg_id: int) -> np.ndarray:
    filepath = os.path.join(EEG_DIR, f"{int(eeg_id)}.parquet")
    df_eeg = pl.read_parquet(filepath, columns=_EEG_COLS).fill_null(0)

    preds = []

    spec = compute_spec_chain(df_eeg)
    spec = proc_2(spec)
    spec_t = torch.tensor(spec, dtype=torch.float32, device=device).unsqueeze(0)

    eeg = compute_eeg_chain(df_eeg)
    eeg = np.nan_to_num(eeg, nan=0.0, posinf=0.0, neginf=0.0)

    eeg_0 = proc_0(eeg)
    eeg_1 = proc_1(eeg)

    eeg_0_t = torch.tensor(eeg_0, dtype=torch.float32, device=device).unsqueeze(0)
    eeg_1_t = torch.tensor(eeg_1, dtype=torch.float32, device=device).unsqueeze(0)

    for model in models_0:
        preds.append(model(eeg_0_t).exp().cpu().numpy().reshape(-1))

    for model in models_1:
        preds.append(model(eeg_1_t).exp().cpu().numpy().reshape(-1))

    for model in models_2:
        preds.append(model(spec_t).exp().cpu().numpy().reshape(-1))

    preds = np.mean(np.stack(preds, axis=0), axis=0)
    preds = np.clip(preds, 1e-8, None)
    preds = preds / preds.sum()
    return preds.astype(np.float32)




## === cell 11
import math
import pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor
from collections import deque

_EEG_COLS_PL = _EEG_COLS  # keep name used below


def _read_eeg_numpy(filepath: str) -> np.ndarray:
    table = pq.read_table(filepath, columns=_EEG_COLS_PL, memory_map=True)
    df = table.to_pandas(split_blocks=True, self_destruct=True)
    X = df.to_numpy(dtype=np.float64, copy=False)  # (T,16)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(X, dtype=np.float64)


def _preprocess_from_filepath(
    filepath: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = _read_eeg_numpy(filepath)

    spec = compute_spec_chain(X)
    spec = proc_2(spec).astype(np.float32, copy=False)

    eeg = compute_eeg_chain(X)
    np.nan_to_num(eeg, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    eeg0 = proc_0(eeg).astype(np.float32, copy=False)
    eeg1 = proc_1(eeg).astype(np.float32, copy=False)
    return eeg0, eeg1, spec


@torch.no_grad()
def gen_ensemble_pred_batch_from_pre(
    pre: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]
) -> np.ndarray:
    B = len(pre)
    Fvalid = int(_VALID_FREQ_MASK.sum())

    eeg0_np = np.empty((B, 4, 4, TARGET_T), dtype=np.float32)
    eeg1_np = np.empty((B, 16, 1, TARGET_T), dtype=np.float32)
    spec_np = np.empty((B, 4, 1, Fvalid, _TSPEC_TARGET), dtype=np.float32)

    for i, (e0, e1, sp) in enumerate(pre):
        eeg0_np[i, ...] = e0
        eeg1_np[i, ...] = e1
        spec_np[i, ...] = sp

    eeg0_t = torch.from_numpy(eeg0_np)
    eeg1_t = torch.from_numpy(eeg1_np)
    spec_t = torch.from_numpy(spec_np)

    if device.type == "cuda":
        eeg0_t = eeg0_t.pin_memory().to(device=device, non_blocking=True)
        eeg1_t = eeg1_t.pin_memory().to(device=device, non_blocking=True)
        spec_t = spec_t.pin_memory().to(device=device, non_blocking=True)
    else:
        eeg0_t = eeg0_t.to(device=device)
        eeg1_t = eeg1_t.to(device=device)
        spec_t = spec_t.to(device=device)

    n_models = len(models_0) + len(models_1) + len(models_2)
    preds_sum = None

    for model in models_0:
        out = model(eeg0_t).exp()
        preds_sum = out if preds_sum is None else (preds_sum + out)

    for model in models_1:
        out = model(eeg1_t).exp()
        preds_sum = out if preds_sum is None else (preds_sum + out)

    for model in models_2:
        out = model(spec_t).exp()
        preds_sum = out if preds_sum is None else (preds_sum + out)

    preds = preds_sum / float(n_models)
    preds = torch.clamp(preds, min=1e-8)
    preds = preds / preds.sum(dim=1, keepdim=True)
    return preds.detach().cpu().numpy().astype(np.float32)


preds_final = np.zeros((len(df_test), len(LABELS)), dtype=np.float32)
eeg_ids = df_test["eeg_id"].to_list()

BATCH_SIZE = 128 if device.type == "cuda" else 4

_CPU = os.cpu_count() or 4
_MAX_WORKERS = min(8, max(2, _CPU // 2))  # keep modest to avoid IO thrash
_PREFETCH_BATCHES = 2

filepaths_all = [os.path.join(EEG_DIR, f"{int(eid)}.parquet") for eid in eeg_ids]


def _submit_batch(ex, fps):
    return ex.map(_preprocess_from_filepath, fps, chunksize=1)


with ProcessPoolExecutor(max_workers=_MAX_WORKERS, mp_context=None) as ex:
    it_starts = list(range(0, len(eeg_ids), BATCH_SIZE))
    q = deque()

    for s in it_starts[:_PREFETCH_BATCHES]:
        fps = filepaths_all[s : s + BATCH_SIZE]
        q.append((s, fps, list(_submit_batch(ex, fps))))

    for idx, s in enumerate(tqdm(it_starts, total=len(it_starts))):
        if not q or q[0][0] != s:
            fps = filepaths_all[s : s + BATCH_SIZE]
            pre = list(_submit_batch(ex, fps))
        else:
            _, fps, pre = q.popleft()

        preds_final[s : s + len(fps)] = gen_ensemble_pred_batch_from_pre(pre)

        next_i = idx + _PREFETCH_BATCHES
        if next_i < len(it_starts):
            ns = it_starts[next_i]
            nfps = filepaths_all[ns : ns + BATCH_SIZE]
            q.append((ns, nfps, list(_submit_batch(ex, nfps))))

preds_final.shape, preds_final[:1]



## === cell 12
df_sub = pd.DataFrame({"eeg_id": eeg_ids})
df_sub[LABELS] = preds_final

probs = df_sub[LABELS].to_numpy(dtype=np.float64)
probs = np.clip(probs, 1e-12, None)
probs = probs / probs.sum(axis=1, keepdims=True)
df_sub[LABELS] = probs

df_sub.to_csv("submission.csv", index=False)
print(df_sub.head())
print("Wrote submission.csv with shape:", df_sub.shape)
print("Row-sum check (min/max):", probs.sum(axis=1).min(), probs.sum(axis=1).max())
