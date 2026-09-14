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
cudf-polars-cu12==25.6.0
geopandas==0.14.4
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
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

0.3433825606701745

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

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

from typing import Optional
from typing import Tuple
from typing import List
from typing import Dict
from typing import Any

## === cell 1
DATA_DIR = "/kaggle/input/hms-harmful-brain-activity-classification/"
SPEC_DIR = os.path.join(DATA_DIR, "test_spectrograms/")
EEG_DIR = os.path.join(DATA_DIR, "test_eegs/")

df_train = pl.read_csv(os.path.join(DATA_DIR, "train.csv"))
df_test = pl.read_csv(os.path.join(DATA_DIR, "test.csv"))
sample_submission = pl.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

df_test.head()

## === cell 2
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

## === cell 3
def load_model(path: str, model: nn.Module) -> nn.Module:
    model.load_state_dict(torch.load(path, map_location = torch.device("cpu"))["model_state_dict"])
    return model

## === cell 4
import sys
sys.path.append("/kaggle/input/hms-models/")

from eeg_cnn_rnn_att import EegModel as EegModel
from spc_cnn_att import SpectrogramCnnModel as SpcModel
from comb_model import MultimodalModel

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/4115741858.py in <cell line: 0>()
      2 sys.path.append("/kaggle/input/hms-models/")
      3 
----> 4 from eeg_cnn_rnn_att import EegModel as EegModel
      5 from spc_cnn_att import SpectrogramCnnModel as SpcModel
      6 from comb_model import MultimodalModel

ModuleNotFoundError: No module named 'eeg_cnn_rnn_att'

## === cell 5
import glob
import sys

"""
models_eeg = [] 
for model_path in sorted(glob.glob("/kaggle/input/hms-models/kaggle_send/kaggle_send/*")):
    model = EegModel()
    print(model_path)
    model = load_model(model_path, model)
    model = model.to(device)
    model.eval()
    models_eeg.append(model) 

models_spc = []
for model_path in sorted(glob.glob("/kaggle/input/hms-models/final_spec_kaggle/final_spec_kaggle/*")):
    model = SpcModel()
    print(model_path)
    model = load_model(model_path, model)
    model = model.to(device)
    model.eval()
    models_spc.append(model)

models_eeg_spc = []
for model_path in sorted(glob.glob("/kaggle/input/hms-models/final_eeg_spec_kaggle/final_eeg_spec_kaggle/*"))[::-1]:
    model = SpcModel()
    print(model_path)
    model = load_model(model_path, model)
    model = model.to(device)
    model.eval()
    models_eeg_spc.append(model)
"""

models_multimodal = []
for model_path in sorted(glob.glob("/kaggle/input/hms-models/stage_6/stage_6/*")):
    try:
        model = MultimodalModel(None, None, None)
        print(model_path)
        model = load_model(model_path, model)
        model = model.to(device)
        model.eval()
        models_multimodal.append(model)
    except:
        print("Corrupted:", model_path)

size = 0


size += len(models_multimodal)

size

## === cell 6
def MAD(signal, axis = -1):
    """Compute the robust standard deviation (MAD) of a signal."""
    median = np.median(signal, axis = axis, keepdims = True)
    absolute_deviations = np.abs(signal - median)
    median_absolute_deviation = np.median(absolute_deviations, axis = axis, keepdims = True)
    scale_factor = 1.4826  # This is a constant for normal distribution
    robust_std = median_absolute_deviation * scale_factor
    return robust_std

from scipy.signal import welch
from scipy.stats import linregress
from scipy.signal import butter, filtfilt

def butter_filter(eeg_data, fs = 200, cutoff_freq = 22, order = 4, btype = "lowpass"):
    b, a = butter(
        N = order, 
        Wn = cutoff_freq / (0.5 * fs), 
        btype = btype, 
        analog = False
    )

    return filtfilt(b, a, eeg_data)

## === cell 7
import scipy
import torchaudio
from torchaudio.transforms import Spectrogram as _Spectrogram

n_fft = 800
win_length = 256
hop_length = 44

spectrogram = _Spectrogram(
    n_fft = n_fft, win_length = win_length, hop_length = hop_length, power = None
)

@torch.no_grad()
def compute_spec(chain):
    chain = torch.Tensor(chain)
    chain = spectrogram(chain)
    chain = chain[:, :, 2:98]
    chain = torch.abs(chain) / 15
    chain = torch.log(chain.clip(math.exp(-4), math.exp(7)))
    chain = chain.mean(axis = 1)
    return chain.numpy()

def compute_spec_eeg(a, b) -> np.ndarray:
    return butter_filter(a - b, cutoff_freq = np.array([0.25, 40.0]), order = 5, btype = "bandpass")

def compute_spec_chain(df_eeg: pl.DataFrame) -> np.ndarray:
    Fp1 = df_eeg["Fp1"].to_numpy() 
    Fp2 = df_eeg["Fp2"].to_numpy()
    Fz  = df_eeg["Fz"].to_numpy()
    Cz  = df_eeg["Cz"].to_numpy()
    Pz  = df_eeg["Pz"].to_numpy()
    F3  = df_eeg["F3"].to_numpy()
    F4  = df_eeg["F4"].to_numpy()
    F7  = df_eeg["F7"].to_numpy()
    F8  = df_eeg["F8"].to_numpy()
    C3  = df_eeg["C3"].to_numpy()
    C4  = df_eeg["C4"].to_numpy()
    P3  = df_eeg["P3"].to_numpy()
    P4  = df_eeg["P4"].to_numpy()
    T3  = df_eeg["T3"].to_numpy()
    T4  = df_eeg["T4"].to_numpy()
    T5  = df_eeg["T5"].to_numpy()
    T6  = df_eeg["T6"].to_numpy()
    O1  = df_eeg["O1"].to_numpy()
    O2  = df_eeg["O2"].to_numpy()
    
    ll = np.stack([(compute_spec_eeg(Fp1 , F7), compute_spec_eeg(F7 , T3), compute_spec_eeg(T3 , T5), compute_spec_eeg(T5 , O1))])
    lp = np.stack([(compute_spec_eeg(Fp1 , F3), compute_spec_eeg(F3 , C3), compute_spec_eeg(C3 , P3), compute_spec_eeg(P3 , O1))])
    rp = np.stack([(compute_spec_eeg(Fp2 , F4), compute_spec_eeg(F4 , C4), compute_spec_eeg(C4 , P4), compute_spec_eeg(P4 , O2))])
    rl = np.stack([(compute_spec_eeg(Fp2 , F8), compute_spec_eeg(F8 , T4), compute_spec_eeg(T4 , T6), compute_spec_eeg(T6 , O2))])
    chain = np.stack([ll, lp, rp, rl])[:, 0]
    
    mads = MAD(chain, axis = -1)
    mads = np.median(mads.reshape(-1))
    chain = chain / (mads + 1e-5)
    
    chain = compute_spec(chain)
    
    return chain


def compute_spec_from_file(filepath: str) -> np.ndarray:
    df_eeg = pl.read_parquet(filepath).fill_null(0)
    chain = compute_spec_chain(df_eeg)
    return chain

## === cell 8

def process_spec(spec: np.ndarray) -> np.ndarray:
    spec = spec[:, 1:]
    spec = np.stack([
        spec[:,   0:100].T,
        spec[:, 100:200].T,
        spec[:, 200:300].T,
        spec[:, 300:400].T,
    ])
    
    return spec

def compute_kaggle_spec_from_file(filepath: str) -> np.ndarray:
    spec = pl.read_parquet(filepath).to_numpy().astype(np.float32)
    spec = process_spec(spec)
    return spec

## === cell 9
import numpy as np
import math

from typing import Union
from typing import Tuple
from typing import List


def bin_array(
    array, 
    bin_size, 
    axis = -1, 
    pad_dir = "symmetric", 
    mode = "edge", 
    return_padding = False,
    **padding_kwargs
) -> Union[np.ndarray, Tuple[np.ndarray, List]]:
    """Given an array and bin size, bins the array along an arbitrary axis into
    bins of size `bin_size`. It will perform padding if the array does not split
    up into equal bin sizes. 

    Args:
        array (np.ndarray): The input array.
        bin_size (int): The size of each bin.
        axis (int): The axis to bin the array along. Default is -1.
        pad_dir (str): The padding direction. One of `left`, `right`, or
            `symmetric` (default).
        return_padding (bool): Option to return the padding width used. 
        mode (str): The padding mode. See the NumPy documentation for options. 

    Returns:
        np.ndarray: The binned array where the number of bins is first. That is,
            the shape will be `(..., n_bins, bin_size, ...)`.

    """
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
        array = np.pad(array, padding, mode = mode, **padding_kwargs)

    array = array.reshape(new_shape)

    if return_padding:
        return array, padding

    return array

## === cell 10
def compute_eeg(eeg: np.ndarray) -> np.ndarray:
    eeg = butter_filter(eeg, cutoff_freq = np.array([0.25, 50]), btype = "bandpass")
    eeg = bin_array(eeg, bin_size = 4, mode = "reflect").mean(axis = -1)
    return eeg
    
def compute_eeg_chain(df_eeg: pl.DataFrame) -> np.ndarray:
    Fp1 = df_eeg["Fp1"].to_numpy()
    Fp2 = df_eeg["Fp2"].to_numpy()
    Fz  = df_eeg["Fz"].to_numpy()
    Cz  = df_eeg["Cz"].to_numpy()
    Pz  = df_eeg["Pz"].to_numpy()
    F3  = df_eeg["F3"].to_numpy()
    F4  = df_eeg["F4"].to_numpy()
    F7  = df_eeg["F7"].to_numpy()
    F8  = df_eeg["F8"].to_numpy()
    C3  = df_eeg["C3"].to_numpy()
    C4  = df_eeg["C4"].to_numpy()
    P3  = df_eeg["P3"].to_numpy()
    P4  = df_eeg["P4"].to_numpy()
    T3  = df_eeg["T3"].to_numpy()
    T4  = df_eeg["T4"].to_numpy()
    T5  = df_eeg["T5"].to_numpy()
    T6  = df_eeg["T6"].to_numpy()
    O1  = df_eeg["O1"].to_numpy()
    O2  = df_eeg["O2"].to_numpy()
    
    ekg = df_eeg["O2"].to_numpy()
    ekg = butter_filter(ekg, cutoff_freq = np.array([0.50, 20.0]), btype = "bandpass")
    ekg = bin_array(ekg, bin_size = 4, mode = "reflect").mean(axis = -1)
    ekg = ekg.reshape(1, -1)
    
    ll = np.stack([(compute_eeg(Fp1 - F7), compute_eeg(F7 - T3), compute_eeg(T3 - T5), compute_eeg(T5 - O1))])
    lp = np.stack([(compute_eeg(Fp1 - F3), compute_eeg(F3 - C3), compute_eeg(C3 - P3), compute_eeg(P3 - O1))])
    rp = np.stack([(compute_eeg(Fp2 - F4), compute_eeg(F4 - C4), compute_eeg(C4 - P4), compute_eeg(P4 - O2))])
    rl = np.stack([(compute_eeg(Fp2 - F8), compute_eeg(F8 - T4), compute_eeg(T4 - T6), compute_eeg(T6 - O2))])
    mid = np.stack([compute_eeg(Fz - Cz), compute_eeg(Cz - Pz)])
    
    chain = np.stack([ll, lp, rp, rl])[:, 0]
    
    return chain, mid, ekg

def compute_eeg_from_file(filepath: str) -> np.ndarray:
    df_eeg = pl.read_parquet(filepath).fill_null(0)
    return compute_eeg_chain(df_eeg)

## === cell 11
LABELS = [
    "seizure_vote", 
    "lpd_vote", 
    "gpd_vote", 
    "lrda_vote", 
    "grda_vote", 
    "other_vote"
]

## === cell 12
np.set_printoptions(formatter = {"all": lambda x: f"{x:0.3f}"})

import albumentations as A
import cv2

spec_transforms = A.Compose([
    A.Resize(
        height = 96,
        width = 224,
        interpolation = cv2.INTER_CUBIC, 
        always_apply = True, 
        p = 1.0
    ),
])

def proc_kspec(x):
    x = x.copy()
        
    x = x[:, 2:98]
    x[np.isnan(x) | np.isinf(x)] = 0
    x = x.clip(np.exp(-4), np.exp(7))
    x = np.log(x)
    
    x = x - x.mean(axis = (1, 2), keepdims = True)
    x = x / (x.std(axis = (1, 2), keepdims = True) + 1e-5)
    
    x = x.transpose(1, 2, 0)
    x = spec_transforms(image = x)["image"]
    x = x.transpose(2, 0, 1)
    
    x = x.reshape(4, 96, 224)
    
    return x


def proc_eeg_spec(x):
    x = x.copy()
        
    x = x[:, :, 2:-2]
    x[np.isnan(x) | np.isinf(x)] = 0

    x = x + 1
    
    x = x.reshape(4, 96, 224)
    
    return x

def proc_eeg(eeg, mid, ekg):
    eeg = eeg.copy()
    mid = mid.copy()
    ekg = ekg.copy()

    eeg[np.isnan(eeg) | np.isinf(eeg)] = 0
    ekg[np.isnan(ekg) | np.isinf(ekg)] = 0
    mid[np.isnan(mid) | np.isinf(mid)] = 0

    eeg = eeg - eeg.mean(axis = -1, keepdims = True)
    mid = mid - mid.mean(axis = -1, keepdims = True)

    mad_std = MAD(eeg, axis = -1).reshape(-1)
    mad_std = np.median(mad_std) + 1e-5

    eeg = eeg / mad_std
    eeg = eeg.clip(-10, 10)

    mid = mid / mad_std
    mid = mid.clip(-10, 10)

    ekg = ekg / (MAD(ekg, axis = -1).reshape(-1) + 1e-5)

    eeg = eeg.reshape(16, -1)
    eeg = np.concatenate([eeg, mid, ekg], axis = 0)

    eeg = eeg.reshape(19, 2_500)

    return eeg

def flip_h(eeg):
    eeg = eeg.copy()
    eeg = eeg[::-1]
    return eeg.copy()

def flip_v(eeg):
    eeg = eeg.copy()
    eeg = eeg[:, ::-1]
    return eeg.copy()

@torch.no_grad()
def gen_ensemble_pred(df_row: pl.DataFrame) -> np.ndarray:
    eeg_id = df_row["eeg_id"].item()
    spc_id = df_row["spectrogram_id"].item()
    eeg_filepath = os.path.join(EEG_DIR, f"{eeg_id}.parquet")
    spc_filepath = os.path.join(SPEC_DIR, f"{spc_id}.parquet")

    preds = []
    
    kspec = compute_kaggle_spec_from_file(spc_filepath)
    kspec = proc_kspec(kspec)
    kspec = torch.Tensor(kspec).to(device).unsqueeze(0)
    
    eeg_spec = compute_spec_from_file(eeg_filepath)
    eeg_spec = proc_eeg_spec(eeg_spec)
    eeg_spec = torch.Tensor(eeg_spec).to(device).unsqueeze(0)
    
    eeg, mid, ekg = compute_eeg_from_file(eeg_filepath)
    eeg[np.isnan(eeg) | np.isinf(eeg)] = 0
    eeg = proc_eeg(eeg, mid, ekg)
    eeg = torch.Tensor(eeg).to(device).unsqueeze(0)

    
    
    
    for model in models_multimodal:
        model.eval()
        preds.append(model(eeg, eeg_spec, kspec).exp().cpu().detach().numpy().reshape(-1))
        
    for pred in preds:
        print(pred)
        
    preds = np.mean(preds, axis = 0)
    preds = preds / preds.sum()
    return preds

## === cell 13
from tqdm.auto import tqdm
    
preds_final = []
for i in range(len(df_test)):
    pred = gen_ensemble_pred(df_test[i])
    preds_final.append(pred)

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/652951617.py in <cell line: 0>()
      3 preds_final = []
      4 for i in range(len(df_test)):
----> 5     pred = gen_ensemble_pred(df_test[i])
      6     preds_final.append(pred)

/usr/local/lib/python3.11/dist-packages/torch/utils/_contextlib.py in decorate_context(*args, **kwargs)
    114     def decorate_context(*args, **kwargs):
    115         with ctx_factory():
--> 116             return func(*args, **kwargs)
    117 
    118     return decorate_context

/tmp/ipykernel_11/1429843133.py in gen_ensemble_pred(df_row)
    102 
    103     eeg_spec = compute_spec_from_file(eeg_filepath)
--> 104     eeg_spec = proc_eeg_spec(eeg_spec)
    105     eeg_spec = torch.Tensor(eeg_spec).to(device).unsqueeze(0)
    106 

/tmp/ipykernel_11/1429843133.py in proc_eeg_spec(x)
     43     x = x + 1
     44 
---> 45     x = x.reshape(4, 96, 224)
     46 
     47     return x

ValueError: cannot reshape array of size 141696 into shape (4,96,224)

## === cell 14
df_sub = pd.DataFrame({"eeg_id": df_test["eeg_id"].to_list()})
df_sub[LABELS] = preds_final
df_sub.to_csv("submission.csv", index = False)
df_sub

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3545080663.py in <cell line: 0>()
      1 df_sub = pd.DataFrame({"eeg_id": df_test["eeg_id"].to_list()})
----> 2 df_sub[LABELS] = preds_final
      3 df_sub.to_csv("submission.csv", index = False)
      4 df_sub

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __setitem__(self, key, value)
   4297             self._setitem_frame(key, value)
   4298         elif isinstance(key, (Series, np.ndarray, list, Index)):
-> 4299             self._setitem_array(key, value)
   4300         elif isinstance(value, DataFrame):
   4301             self._set_item_frame_value(key, value)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _setitem_array(self, key, value)
   4356 
   4357             else:
-> 4358                 self._iset_not_inplace(key, value)
   4359 
   4360     def _iset_not_inplace(self, key, value):

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _iset_not_inplace(self, key, value)
   4375         if self.columns.is_unique:
   4376             if np.shape(value)[-1] != len(key):
-> 4377                 raise ValueError("Columns must be same length as key")
   4378 
   4379             for i, col in enumerate(key):

ValueError: Columns must be same length as key
