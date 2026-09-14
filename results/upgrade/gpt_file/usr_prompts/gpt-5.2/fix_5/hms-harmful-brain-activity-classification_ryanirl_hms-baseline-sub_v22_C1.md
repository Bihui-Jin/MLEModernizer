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
from torch.utils.data import DataLoader

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




## === cell 3
def load_model(path: str, model: nn.Module) -> nn.Module:
    state = torch.load(path, map_location=torch.device("cpu"))
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    return model




## === cell 4
LABELS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

train_votes = df_train.select(LABELS).to_numpy().astype(np.float64)
train_votes_sum = train_votes.sum(axis=1, keepdims=True)
train_probs = train_votes / np.clip(train_votes_sum, 1.0, None)
PRIOR = train_probs.mean(axis=0)
PRIOR = PRIOR / PRIOR.sum()

print("Using PRIOR (fallback, sums to 1):", PRIOR, "sum=", PRIOR.sum())


class PriorModel(nn.Module):
    def __init__(self, prior: np.ndarray):
        super().__init__()
        self.register_buffer(
            "log_prior", torch.log(torch.tensor(prior, dtype=torch.float32))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]
        return self.log_prior.unsqueeze(0).expand(bs, -1)


models_eeg = [PriorModel(PRIOR).to(device).eval()]
models_eeg_spc = [PriorModel(PRIOR).to(device).eval()]

len(models_eeg) + len(models_eeg_spc)




## === cell 5
def MAD(signal, axis=-1):
    median = np.median(signal, axis=axis, keepdims=True)
    absolute_deviations = np.abs(signal - median)
    median_absolute_deviation = np.median(absolute_deviations, axis=axis, keepdims=True)
    scale_factor = 1.4826
    robust_std = median_absolute_deviation * scale_factor
    return robust_std


from scipy.signal import welch
from scipy.stats import linregress
from scipy.signal import butter, filtfilt


def butter_filter(eeg_data, fs=200, cutoff_freq=22, order=4, btype="lowpass"):
    b, a = butter(
        N=order,
        Wn=cutoff_freq / (0.5 * fs),
        btype=btype,
        analog=False,
    )
    return filtfilt(b, a, eeg_data)




## === cell 6
import scipy
import torchaudio
from torchaudio.transforms import Spectrogram as _Spectrogram

n_fft = 800
win_length = 256
hop_length = 44

spectrogram = _Spectrogram(
    n_fft=n_fft, win_length=win_length, hop_length=hop_length, power=None
)


@torch.no_grad()
def compute_spec(chain: np.ndarray) -> np.ndarray:
    """
    Ensure consistent output shape for downstream proc_eeg_spec.
    Input: chain shape (4, T) as float.
    Output: (4, T_frames, 96) where 96 corresponds to freq bins [2:98].
    """
    x = torch.as_tensor(chain, dtype=torch.float32)  # (4, T)
    x = spectrogram(x)  # (4, F, Frames) complex
    x = x[:, 2:98, :]  # (4, 96, Frames)
    x = torch.abs(x) / 15.0
    x = torch.log(x.clamp(min=math.exp(-4), max=math.exp(7)))
    x = x.transpose(1, 2).contiguous()  # (4, Frames, 96)
    return x.cpu().numpy()


def compute_spec_eeg(a, b) -> np.ndarray:
    return butter_filter(
        a - b, cutoff_freq=np.array([0.25, 40.0]), order=5, btype="bandpass"
    )


def compute_spec_chain(df_eeg: pl.DataFrame) -> np.ndarray:
    Fp1 = df_eeg["Fp1"].to_numpy()
    Fp2 = df_eeg["Fp2"].to_numpy()
    Fz = df_eeg["Fz"].to_numpy()
    Cz = df_eeg["Cz"].to_numpy()
    Pz = df_eeg["Pz"].to_numpy()
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
            compute_spec_eeg(Fp1, F7),
            compute_spec_eeg(F7, T3),
            compute_spec_eeg(T3, T5),
            compute_spec_eeg(T5, O1),
        ]
    )
    lp = np.stack(
        [
            compute_spec_eeg(Fp1, F3),
            compute_spec_eeg(F3, C3),
            compute_spec_eeg(C3, P3),
            compute_spec_eeg(P3, O1),
        ]
    )
    rp = np.stack(
        [
            compute_spec_eeg(Fp2, F4),
            compute_spec_eeg(F4, C4),
            compute_spec_eeg(C4, P4),
            compute_spec_eeg(P4, O2),
        ]
    )
    rl = np.stack(
        [
            compute_spec_eeg(Fp2, F8),
            compute_spec_eeg(F8, T4),
            compute_spec_eeg(T4, T6),
            compute_spec_eeg(T6, O2),
        ]
    )

    chain = np.stack([ll, lp, rp, rl])[:, 0]  # (4, T)

    mads = MAD(chain, axis=-1)
    mads = np.median(mads.reshape(-1))
    chain = chain / (mads + 1e-5)

    chain = compute_spec(chain)  # (4, Frames, 96)
    return chain


def compute_spec_from_file(filepath: str) -> np.ndarray:
    df_eeg = pl.read_parquet(filepath).fill_null(0)
    chain = compute_spec_chain(df_eeg)
    return chain




## === cell 7
import albumentations as A

spec_resize = A.Resize(height=96, width=224)


def proc_eeg_spec(x: np.ndarray) -> np.ndarray:
    """
    x is (4, T_frames, 96). Convert to (96, T_frames, 4) for albumentations.
    Output: (4, 96, 224) float32.
    """
    x = x.copy()
    x[np.isnan(x) | np.isinf(x)] = 0.0

    if x.ndim != 3 or x.shape[0] != 4 or x.shape[2] != 96:
        raise ValueError(f"Unexpected eeg_spec shape {x.shape}, expected (4, T, 96)")

    x = x.transpose(2, 1, 0)  # (96, T, 4)
    x = spec_resize(image=x)["image"]  # (96, 224, 4)
    x = x.transpose(2, 0, 1)  # (4, 96, 224)

    x = x + 1.0
    return x.astype(np.float32)




## === cell 8
def bin_array(
    array,
    bin_size,
    axis=-1,
    pad_dir="symmetric",
    mode="edge",
    return_padding=False,
    **padding_kwargs,
):
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




## === cell 9
def compute_eeg(eeg: np.ndarray) -> np.ndarray:
    eeg = butter_filter(eeg, cutoff_freq=np.array([0.25, 50]), btype="bandpass")
    eeg = bin_array(eeg, bin_size=4, mode="reflect").mean(axis=-1)
    return eeg


def compute_eeg_chain(df_eeg: pl.DataFrame) -> np.ndarray:
    Fp1 = df_eeg["Fp1"].to_numpy()
    Fp2 = df_eeg["Fp2"].to_numpy()
    Fz = df_eeg["Fz"].to_numpy()
    Cz = df_eeg["Cz"].to_numpy()
    Pz = df_eeg["Pz"].to_numpy()
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

    if "EKG" in df_eeg.columns:
        ekg = df_eeg["EKG"].to_numpy()
    else:
        ekg = df_eeg["O2"].to_numpy()

    ekg = butter_filter(ekg, cutoff_freq=np.array([0.50, 20.0]), btype="bandpass")
    ekg = bin_array(ekg, bin_size=4, mode="reflect").mean(axis=-1)
    ekg = ekg.reshape(1, -1)

    ll = np.stack(
        [
            compute_eeg(Fp1 - F7),
            compute_eeg(F7 - T3),
            compute_eeg(T3 - T5),
            compute_eeg(T5 - O1),
        ]
    )
    lp = np.stack(
        [
            compute_eeg(Fp1 - F3),
            compute_eeg(F3 - C3),
            compute_eeg(C3 - P3),
            compute_eeg(P3 - O1),
        ]
    )
    rp = np.stack(
        [
            compute_eeg(Fp2 - F4),
            compute_eeg(F4 - C4),
            compute_eeg(C4 - P4),
            compute_eeg(P4 - O2),
        ]
    )
    rl = np.stack(
        [
            compute_eeg(Fp2 - F8),
            compute_eeg(F8 - T4),
            compute_eeg(T4 - T6),
            compute_eeg(T6 - O2),
        ]
    )
    mid = np.stack([compute_eeg(Fz - Cz), compute_eeg(Cz - Pz)])

    chain = np.stack([ll, lp, rp, rl])[:, 0]
    return chain, mid, ekg


def compute_eeg_from_file(filepath: str) -> np.ndarray:
    df_eeg = pl.read_parquet(filepath).fill_null(0)
    return compute_eeg_chain(df_eeg)




## === cell 10
def _center_crop_or_pad_1d(x: np.ndarray, target_len: int) -> np.ndarray:
    """Execution fix: ensure all modalities have consistent length before concat."""
    x = np.asarray(x)
    if x.shape[-1] == target_len:
        return x
    if x.shape[-1] > target_len:
        start = (x.shape[-1] - target_len) // 2
        end = start + target_len
        return x[..., start:end]
    pad_total = target_len - x.shape[-1]
    pad_l = pad_total // 2
    pad_r = pad_total - pad_l
    pad_width = [(0, 0)] * x.ndim
    pad_width[-1] = (pad_l, pad_r)
    return np.pad(x, pad_width, mode="edge")


def proc_eeg(eeg, mid, ekg):
    """
    Bugfix: eeg/mid/ekg are already binned to ~2500 samples (50s @200Hz -> /4).
    The previous reshape path accidentally reduced eeg length to 625 and broke concat.
    Keep identical features (16 eeg + 2 mid + 1 ekg) but enforce consistent *binned*
    time length before concatenation.
    """
    eeg = np.asarray(eeg, dtype=np.float32).copy()  # expected (4, 2500)
    mid = np.asarray(mid, dtype=np.float32).copy()  # expected (2, 2500)
    ekg = np.asarray(ekg, dtype=np.float32).copy()  # expected (1, 2500)

    eeg[np.isnan(eeg) | np.isinf(eeg)] = 0
    ekg[np.isnan(ekg) | np.isinf(ekg)] = 0
    mid[np.isnan(mid) | np.isinf(mid)] = 0

    eeg = eeg - eeg.mean(axis=-1, keepdims=True)
    mid = mid - mid.mean(axis=-1, keepdims=True)

    mad_std = MAD(eeg, axis=-1).reshape(-1)
    mad_std = np.median(mad_std) + 1e-5

    eeg = (eeg / mad_std).clip(-10, 10)
    mid = (mid / mad_std).clip(-10, 10)
    ekg = ekg / (MAD(ekg, axis=-1).reshape(-1) + 1e-5)

    target_len = 2500
    eeg = _center_crop_or_pad_1d(eeg, target_len)
    mid = _center_crop_or_pad_1d(mid, target_len)
    ekg = _center_crop_or_pad_1d(ekg, target_len)

    if eeg.shape[0] == 4:
        eeg = np.repeat(eeg, repeats=4, axis=0)  # (16, 2500)
    elif eeg.shape[0] != 16:
        raise ValueError(f"Unexpected eeg channels {eeg.shape[0]} (expected 4 or 16)")

    x = np.concatenate([eeg, mid, ekg], axis=0)  # (19, 2500)
    if x.shape != (19, 2500):
        raise ValueError(
            f"Unexpected proc_eeg output shape {x.shape}, expected (19,2500)"
        )
    return x.astype(np.float32)


@torch.no_grad()
def gen_ensemble_pred(df_row: dict) -> np.ndarray:
    eeg_id = int(df_row["eeg_id"])
    eeg_filepath = os.path.join(EEG_DIR, f"{eeg_id}.parquet")

    preds = []

    eeg_spec = compute_spec_from_file(eeg_filepath)  # (4, T, 96)
    eeg_spec = proc_eeg_spec(eeg_spec)  # (4, 96, 224)
    eeg_spec_t = torch.tensor(eeg_spec, dtype=torch.float32, device=device).unsqueeze(0)

    eeg, mid, ekg = compute_eeg_from_file(eeg_filepath)
    eeg_t = torch.tensor(
        proc_eeg(eeg, mid, ekg), dtype=torch.float32, device=device
    ).unsqueeze(0)

    for model in models_eeg:
        preds.append(model(eeg_t).exp().detach().cpu().numpy().reshape(-1))

    for model in models_eeg_spc:
        preds.append(model(eeg_spec_t).exp().detach().cpu().numpy().reshape(-1))

    preds = np.mean(np.stack(preds, axis=0), axis=0)
    preds = np.clip(preds, 1e-12, None)
    preds = preds / preds.sum()
    return preds.astype(np.float64)




## === cell 11
preds_final = np.zeros((df_test.height, len(LABELS)), dtype=np.float64)

for i, row in enumerate(tqdm(df_test.iter_rows(named=True), total=df_test.height)):
    preds_final[i] = gen_ensemble_pred(row)

preds_final = np.clip(preds_final, 1e-12, None)
preds_final = preds_final / preds_final.sum(axis=1, keepdims=True)

preds_final.shape, preds_final.sum(axis=1).min(), preds_final.sum(axis=1).max()



## === cell 12
df_sub = pd.DataFrame({"eeg_id": df_test["eeg_id"].to_list()})
for j, col in enumerate(LABELS):
    df_sub[col] = preds_final[:, j]

probs = df_sub[LABELS].to_numpy(dtype=np.float64)
probs = np.clip(probs, 1e-12, None)
probs = probs / probs.sum(axis=1, keepdims=True)
df_sub[LABELS] = probs

assert list(df_sub.columns) == ["eeg_id"] + LABELS

out_path = "submission.csv"
df_sub.to_csv(out_path, index=False)
print("Wrote", out_path, "shape=", df_sub.shape)
print(
    "Row sum check:", df_sub[LABELS].sum(axis=1).min(), df_sub[LABELS].sum(axis=1).max()
)
df_sub.head()
