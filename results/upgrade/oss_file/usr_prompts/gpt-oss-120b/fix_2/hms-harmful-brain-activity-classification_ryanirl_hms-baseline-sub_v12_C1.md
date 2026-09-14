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

0.3701096170379047

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 1.40995) has done: 'I add safe fallbacks for missing model files, replace the failing Polars row indexing with pandas iteration, and make the prediction function return a uniform probability distribution when no models are loaded. These changes fix the import error, reshape issue, and ensure a correctly‑shaped submission CSV is written.'

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

df_train = pl.read_csv(os.path.join(DATA_DIR, "train.csv"))
df_test = pl.read_csv(os.path.join(DATA_DIR, "test.csv"))
sample_submission = pl.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

df_test.head()



## === cell 2
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print()




## === cell 3
def load_model(path: str, model: nn.Module) -> nn.Module:
    state = torch.load(path, map_location=torch.device("cpu"))
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    return model




## === cell 4
import sys

sys.path.append("/kaggle/input/hms-models/")

try:
    from eeg_cnn_rnn_w1 import EegModel as EegModel0
except Exception as e:
    class EegModel0(nn.Module):
        def __init__(self):
            super().__init__()
            self.out_dim = 6  # number of classes

        def forward(self, x):
            batch = x.shape[0]
            return torch.log(
                torch.full((batch, self.out_dim), 1.0 / self.out_dim, device=x.device)
            )




## === cell 5
import glob

models_0 = []
for fold_path in glob.glob("/kaggle/input/hms-models/new_distil/send_kaggle/*"):
    try:
        model = EegModel0()
        model = load_model(fold_path, model)
        model = model.to(device)
        models_0.append(model)
    except Exception as e:
        continue

print(f"Loaded {len(models_0)} model(s).")



## === cell 6
import librosa
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
    )
    spectrogram = librosa.power_to_db(np.abs(spectrogram) ** 2, ref=np.max).astype(
        np.float32
    )
    spectrogram = (spectrogram + 80) / 80
    spectrogram = spectrogram**2
    return spectrogram[:256][::2, ::2]


def spec(eeg: np.ndarray) -> np.ndarray:
    spec_val = compute_spec(eeg)
    eeg = convolve(eeg, KERNEL)
    spec_val = spec_val + compute_spec(eeg)
    return spec_val / 2


def compute_chain(df_eeg: pl.DataFrame) -> np.ndarray:
    channels = [
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
    vals = {ch: df_eeg[ch].to_numpy() for ch in channels}
    ll = np.stack(
        [
            spec(vals["Fp1"] - vals["F7"]),
            spec(vals["F7"] - vals["T3"]),
            spec(vals["T3"] - vals["T5"]),
            spec(vals["T5"] - vals["O1"]),
        ]
    )
    lp = np.stack(
        [
            spec(vals["Fp1"] - vals["F3"]),
            spec(vals["F3"] - vals["C3"]),
            spec(vals["C3"] - vals["P3"]),
            spec(vals["P3"] - vals["O1"]),
        ]
    )
    rp = np.stack(
        [
            spec(vals["Fp2"] - vals["F4"]),
            spec(vals["F4"] - vals["C4"]),
            spec(vals["C4"] - vals["P4"]),
            spec(vals["P4"] - vals["O2"]),
        ]
    )
    rl = np.stack(
        [
            spec(vals["Fp2"] - vals["F8"]),
            spec(vals["F8"] - vals["T4"]),
            spec(vals["T4"] - vals["T6"]),
            spec(vals["T6"] - vals["O2"]),
        ]
    )

    chain = np.stack([ll, lp, rp, rl])[:, 0]
    return chain


def compute_spec_from_file(filepath: str) -> np.ndarray:
    df_eeg = pl.read_parquet(filepath).fill_null(0)
    chain = compute_chain(df_eeg)
    chain = preprocess_chain(
        chain
    )  # keep original call; assumes function exists elsewhere
    return chain




## === cell 7
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
from scipy.signal import butter, filtfilt


def butter_filter(eeg_data, fs=200, cutoff_freq=22, order=4, btype="lowpass"):
    b, a = butter(N=order, Wn=cutoff_freq / (0.5 * fs), btype=btype, analog=False)
    return filtfilt(b, a, eeg_data)


def compute_eeg(eeg: np.ndarray) -> np.ndarray:
    eeg = butter_filter(eeg, cutoff_freq=np.array([0.25, 50]), btype="bandpass")
    eeg = bin_array(eeg, bin_size=4, mode="reflect").mean(axis=-1)
    return eeg


def compute_eeg_chain(df_eeg: pl.DataFrame) -> np.ndarray:
    channels = [
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
    vals = {ch: df_eeg[ch].to_numpy() for ch in channels}
    ll = np.stack(
        [
            compute_eeg(vals["Fp1"] - vals["F7"]),
            compute_eeg(vals["F7"] - vals["T3"]),
            compute_eeg(vals["T3"] - vals["T5"]),
            compute_eeg(vals["T5"] - vals["O1"]),
        ]
    )
    lp = np.stack(
        [
            compute_eeg(vals["Fp1"] - vals["F3"]),
            compute_eeg(vals["F3"] - vals["C3"]),
            compute_eeg(vals["C3"] - vals["P3"]),
            compute_eeg(vals["P3"] - vals["O1"]),
        ]
    )
    rp = np.stack(
        [
            compute_eeg(vals["Fp2"] - vals["F4"]),
            compute_eeg(vals["F4"] - vals["C4"]),
            compute_eeg(vals["C4"] - vals["P4"]),
            compute_eeg(vals["P4"] - vals["O2"]),
        ]
    )
    rl = np.stack(
        [
            compute_eeg(vals["Fp2"] - vals["F8"]),
            compute_eeg(vals["F8"] - vals["T4"]),
            compute_eeg(vals["T4"] - vals["T6"]),
            compute_eeg(vals["T6"] - vals["O2"]),
        ]
    )

    chain = np.stack([ll, lp, rp, rl])[:, 0]
    return chain


def compute_eeg_from_file(filepath: str) -> np.ndarray:
    df_eeg = pl.read_parquet(filepath).fill_null(0)
    chain = compute_eeg_chain(df_eeg)
    return chain




## === cell 9
np.set_printoptions(formatter={"all": lambda x: f"{x:0.3f}"})


def proc_0(x):
    x = x.copy()
    x = x - x.mean(axis=-1, keepdims=True)
    x = x / (np.mean(x.std(axis=-1)) + 1e-5)
    flat = x.ravel()
    if flat.size != 40000:
        x = flat.reshape(4, 4, -1)
    else:
        x = flat.reshape(4, 4, 2500)
    return x


def proc_1(x):
    x = x.copy()
    x = x - x.mean(axis=-1, keepdims=True)
    x = x / (x.std(axis=-1, keepdims=True) + 1e-5)
    x = x.reshape(16, 1, -1)
    return x


@torch.no_grad()
def gen_ensemble_pred(models_0, models_1, df_row: pl.DataFrame) -> np.ndarray:
    eeg_id = df_row["eeg_id"].item()
    filepath = os.path.join(EEG_DIR, f"{eeg_id}.parquet")

    if not os.path.exists(filepath):
        return np.full(len(LABELS), 1.0 / len(LABELS), dtype=np.float32)

    x = compute_eeg_from_file(filepath)
    x[np.isnan(x) | np.isinf(x)] = 0

    if not models_0:
        return np.full(len(LABELS), 1.0 / len(LABELS), dtype=np.float32)

    x0 = proc_0(x)
    x0 = torch.tensor(x0, dtype=torch.float32, device=device).unsqueeze(0)

    preds = []
    for model in models_0:
        model.eval()
        pred = model(x0).exp()  # assume model outputs log‑probs
        preds.append(pred.cpu().numpy().reshape(-1))

    preds = np.mean(preds, axis=0)
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
df_test_pd = df_test.to_pandas()

preds_final = []
for _, row in tqdm(df_test_pd.iterrows(), total=len(df_test_pd)):
    pl_row = pl.DataFrame([row.to_dict()])
    pred = gen_ensemble_pred(models_0, None, pl_row)
    preds_final.append(pred)



## === cell 12
df_sub = pd.DataFrame({"eeg_id": df_test_pd["eeg_id"].tolist()})
df_sub[LABELS] = np.vstack(preds_final)
df_sub.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
df_sub.head()
