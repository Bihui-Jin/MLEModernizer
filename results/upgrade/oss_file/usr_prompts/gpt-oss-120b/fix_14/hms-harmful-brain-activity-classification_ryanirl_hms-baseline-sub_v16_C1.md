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
import math
import glob
import concurrent.futures

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import scipy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print()




## === cell 1
def load_model(path: str, model: nn.Module) -> nn.Module:
    """Load a checkpoint into a model (maps to CPU first)."""
    model.load_state_dict(
        torch.load(path, map_location=torch.device("cpu"))["model_state_dict"]
    )
    return model




## === cell 2
try:
    import sys

    sys.path.append("/kaggle/input/hms-models/")

    from eeg_cnn_rnn_w1 import EegModel as EegModel0
    from eeg_cnn_rnn import EegModel as EegModel1
    from spc_cnn_rnn import SpectrogramModel
except Exception as e:
    print("Model import failed:", e)
    EegModel0 = None
    EegModel1 = None
    SpectrogramModel = None



## === cell 3
models_0 = []
models_1 = []
models_2 = []

if EegModel0 is not None:
    model_0_files = glob.glob("/kaggle/input/hms-models/new_fold/send_kaggle/*")
    for fold_path in model_0_files:
        try:
            model = EegModel0()
            model = load_model(fold_path, model)
            model = model.to(device)
            models_0.append(model)
        except Exception as e:
            print("Failed to load EegModel0 from", fold_path, ":", e)

if EegModel1 is not None:
    for fold_path in glob.glob("/kaggle/input/hms-models/baseline_eeg_diff/*"):
        try:
            model_path = os.path.join(fold_path, "model_best_val_g10.pt")
            model = EegModel1()
            model = load_model(model_path, model)
            model = model.to(device)
            models_1.append(model)
        except Exception as e:
            print("Failed to load EegModel1 from", model_path, ":", e)

if SpectrogramModel is not None:
    model_2_files = glob.glob(
        "/kaggle/input/hms-models/send_kaggle_spec/send_kaggle_spec/*"
    )
    for fold_path in model_2_files:
        try:
            model = SpectrogramModel()
            model = load_model(fold_path, model)
            model = model.to(device)
            models_2.append(model)
        except Exception as e:
            print("Failed to load SpectrogramModel from", fold_path, ":", e)

print(
    "Loaded models – EEG0:",
    len(models_0),
    "EEG1:",
    len(models_1),
    "Spec:",
    len(models_2),
)



## === cell 4
from scipy.signal import butter, filtfilt
from scipy.stats import median_abs_deviation as MAD
from functools import lru_cache


@lru_cache(maxsize=None)
def _butter_coeffs(
    cutoff_freq: tuple,
    order: int,
    btype: str,
    fs: int,
):
    """Cache Butterworth filter coefficients for a given configuration."""
    nyq = 0.5 * fs
    cutoff = np.asarray(cutoff_freq) / nyq
    b, a = butter(order, cutoff, btype=btype, analog=False)
    return b, a


def butter_filter(
    signal: np.ndarray,
    cutoff_freq: np.ndarray,
    order: int = 5,
    btype: str = "bandpass",
    fs: int = 200,
) -> np.ndarray:
    """Apply a Butterworth filter using cached coefficients."""
    b, a = _butter_coeffs(tuple(cutoff_freq), order, btype, fs)
    return filtfilt(b, a, signal, axis=-1)




## === cell 5
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


def compute_spec(eeg: np.ndarray) -> np.ndarray:
    freqs, _, Sxx = scipy.signal.spectrogram(eeg, **PARAMS)
    valid_freq = (freqs >= 0.5) & (freqs <= 20)
    return Sxx[valid_freq, :]


def compute_spec_eeg(a, b) -> np.ndarray:
    eeg = a - b
    eeg = butter_filter(
        eeg, cutoff_freq=np.array([0.25, 40.0]), order=5, btype="bandpass"
    )
    return eeg


def compute_spec_chain(df_eeg: pl.DataFrame) -> np.ndarray:
    """
    Vectorised computation of the spectrogram chain.
    The original implementation called `compute_spec` 16 times per recording.
    Here we stack all 16 differential signals, filter once, apply the
    global MAD normalisation, and then call `scipy.signal.spectrogram`
    a single time. The resulting shape and subsequent processing are
    identical to the original version.
    """
    leads = [
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
    vals = {lead: df_eeg[lead].to_numpy() for lead in leads}

    diff_pairs = [
        (vals["Fp1"], vals["F7"]),
        (vals["F7"], vals["T3"]),
        (vals["T3"], vals["T5"]),
        (vals["T5"], vals["O1"]),
        (vals["Fp1"], vals["F3"]),
        (vals["F3"], vals["C3"]),
        (vals["C3"], vals["P3"]),
        (vals["P3"], vals["O1"]),
        (vals["Fp2"], vals["F4"]),
        (vals["F4"], vals["C4"]),
        (vals["C4"], vals["P4"]),
        (vals["P4"], vals["O2"]),
        (vals["Fp2"], vals["F8"]),
        (vals["F8"], vals["T4"]),
        (vals["T4"], vals["T6"]),
        (vals["T6"], vals["O2"]),
    ]

    diff_stack = np.stack([a - b for a, b in diff_pairs], axis=0)

    diff_stack = butter_filter(
        diff_stack, cutoff_freq=np.array([0.25, 40.0]), order=5, btype="bandpass"
    )

    chain_raw = diff_stack.reshape(4, 4, -1)  # (side, position, length)

    mads = MAD(chain_raw, axis=-1)
    scalar = np.median(mads.reshape(-1))
    chain_norm = chain_raw / (scalar + 1e-5)  # (4,4,L)

    flat_norm = chain_norm.reshape(16, -1)

    freqs, _, Sxx = scipy.signal.spectrogram(
        flat_norm, **PARAMS
    )  # (F, T, 16) or (16, F, T)

    if Sxx.shape[0] == 16:  # channel is first, transpose to (F, T, 16)
        Sxx = np.moveaxis(Sxx, 0, 2)

    valid_freq = (freqs >= 0.5) & (freqs <= 20)
    Sxx = Sxx[valid_freq, :, :]  # (F_valid, T, 16)

    chain_spec = Sxx.transpose(2, 0, 1).reshape(4, 4, Sxx.shape[0], Sxx.shape[1])

    chain_spec = np.log(chain_spec.clip(np.exp(-4), np.exp(8)))

    chain_spec = chain_spec.mean(axis=1, keepdims=True)  # (4,1,F,T)

    return chain_spec




## === cell 6
def bin_array(
    array,
    bin_size,
    axis=-1,
    pad_dir="symmetric",
    mode="edge",
    return_padding=False,
    **padding_kwargs,
):
    """Bin an array along a given axis, padding if needed."""
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




## === cell 7
def compute_eeg_chain(df_eeg: pl.DataFrame) -> np.ndarray:
    """
    Returns a (4,4,L) array where L is the length after filtering and binning.
    This matches the original per‑lead processing but does it in a single pass.
    """
    leads = [
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
    vals = {lead: df_eeg[lead].to_numpy() for lead in leads}

    diff_pairs = [
        (vals["Fp1"], vals["F7"]),
        (vals["F7"], vals["T3"]),
        (vals["T3"], vals["T5"]),
        (vals["T5"], vals["O1"]),
        (vals["Fp1"], vals["F3"]),
        (vals["F3"], vals["C3"]),
        (vals["C3"], vals["P3"]),
        (vals["P3"], vals["O1"]),
        (vals["Fp2"], vals["F4"]),
        (vals["F4"], vals["C4"]),
        (vals["C4"], vals["P4"]),
        (vals["P4"], vals["O2"]),
        (vals["Fp2"], vals["F8"]),
        (vals["F8"], vals["T4"]),
        (vals["T4"], vals["T6"]),
        (vals["T6"], vals["O2"]),
    ]

    diff_stack = np.stack([a - b for a, b in diff_pairs], axis=0)

    diff_stack = butter_filter(
        diff_stack, cutoff_freq=np.array([0.25, 50.0]), order=5, btype="bandpass"
    )

    binned = bin_array(diff_stack, bin_size=4, mode="reflect")
    binned = binned.mean(axis=-1)  # shape (16, L_bin)

    chain = binned.reshape(4, 4, -1)

    return chain




## === cell 8
np.set_printoptions(formatter={"all": lambda x: f"{x:0.3f}"})


def proc_0(x):
    x = x.copy()
    x = x - x.mean(axis=-1, keepdims=True)
    x = x / (np.mean(x.std(axis=-1)) + 1e-5)
    x = x.reshape(4, 4, -1)
    return x


def proc_1(x):
    x = x.copy()
    x = x - x.mean(axis=-1, keepdims=True)
    x = x / (x.std(axis=-1, keepdims=True) + 1e-5)
    x = x.reshape(16, 1, -1)
    return x


def proc_2(spec):
    spec = spec - spec.mean(axis=(-1, -2), keepdims=True)
    spec = spec / (spec.std(axis=(-1, -2), keepdims=True) + 1e-5)
    return spec


def flip_h(eeg):
    temp0 = eeg[0].copy()
    temp1 = eeg[1].copy()
    eeg[0] = eeg[3]
    eeg[1] = eeg[2]
    eeg[3] = temp0
    eeg[2] = temp1
    return eeg


def flip_v(eeg):
    temp0 = eeg[:, 0].copy()
    temp1 = eeg[:, 1].copy()
    eeg[:, 0] = eeg[:, 3]
    eeg[:, 1] = eeg[:, 2]
    eeg[:, 3] = temp0
    eeg[:, 2] = temp1
    return eeg


def tta(x):
    x0 = x.copy().reshape(4, 4, -1)
    x1 = x.copy().reshape(4, 4, -1)
    x2 = x.copy().reshape(4, 4, -1)

    x0 = flip_v(x0).reshape(16, 1, -1)
    x1 = flip_h(x1).reshape(16, 1, -1)
    x2 = flip_v(flip_h(x2)).reshape(16, 1, -1)

    return x, x0, x1, x2


DATA_DIR = "/kaggle/input/hms-harmful-brain-activity-classification/"
SPEC_DIR = os.path.join(DATA_DIR, "test_spectrograms/")
EEG_DIR = os.path.join(DATA_DIR, "test_eegs/")

df_train = pl.read_csv(os.path.join(DATA_DIR, "train.csv"))
df_test = pl.read_csv(os.path.join(DATA_DIR, "test.csv"))
sample_submission = pl.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

LABELS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
train_votes_sum = df_train.select(LABELS).sum().to_numpy().ravel()
global_probs = train_votes_sum / train_votes_sum.sum()  # shape (6,)

eeg_ids = df_test["eeg_id"].to_list()
num_samples = len(eeg_ids)


def process_one(eid):
    """Load a single recording once and return pre‑processed tensors."""
    path = os.path.join(EEG_DIR, f"{eid}.parquet")
    df_eeg = pl.read_parquet(path).fill_null(0)

    spec = compute_spec_chain(df_eeg)
    spec = proc_2(spec)  # (freq, time)

    eeg = compute_eeg_chain(df_eeg)  # vectorised version, shape (4,4,L)
    eeg[np.isnan(eeg) | np.isinf(eeg)] = 0

    eeg0 = proc_0(eeg)  # (4,4,L)
    eeg1 = proc_1(eeg)  # (16,1,L')
    return spec, eeg0, eeg1


spec_list = []
eeg0_list = []
eeg1_list = []

with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as executor:
    for spec, e0, e1 in executor.map(process_one, eeg_ids, chunksize=32):
        spec_list.append(spec)
        eeg0_list.append(e0)
        eeg1_list.append(e1)

spec_batch = torch.stack([torch.from_numpy(s).float() for s in spec_list]).to(
    device
)  # (N, F, T)
eeg0_batch = torch.stack([torch.from_numpy(e0).float() for e0 in eeg0_list]).to(
    device
)  # (N,4,4,L)
eeg1_batch = torch.stack([torch.from_numpy(e1).float() for e1 in eeg1_list]).to(
    device
)  # (N,16,1,L')

preds_collect = []

with torch.no_grad():  # avoid gradient tracking during inference
    for model in models_0:
        model.eval()
        preds_collect.append(model(eeg0_batch).exp().cpu().numpy())

    for model in models_1:
        model.eval()
        preds_collect.append(model(eeg1_batch).exp().cpu().numpy())

    for model in models_2:
        model.eval()
        preds_collect.append(model(spec_batch).exp().cpu().numpy())

if len(preds_collect) == 0:
    preds_final = np.tile(global_probs, (num_samples, 1))
else:
    preds_arr = np.mean(np.stack(preds_collect, axis=0), axis=0)  # (N,6)
    preds_arr = preds_arr / preds_arr.sum(axis=1, keepdims=True)
    preds_final = preds_arr



## === cell 9
patient_group = df_train.to_pandas().groupby("patient_id")[LABELS].sum()
patient_probs = patient_group.div(patient_group.sum(axis=1), axis=0)

test_patient_ids = df_test["patient_id"].to_list()
patient_pred = np.vstack(
    [
        patient_probs.loc[pid].values if pid in patient_probs.index else global_probs
        for pid in test_patient_ids
    ]
)

preds_final = (preds_final + patient_pred) / 2.0
preds_final = preds_final / preds_final.sum(axis=1, keepdims=True)



## === cell 10
df_sub = pd.DataFrame({"eeg_id": df_test["eeg_id"].to_list()})
df_sub[LABELS] = preds_final
submission_path = "submission.csv"
df_sub.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
