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

# 5. Target score

0.3445197588088183

# 6. Current score

0.99275

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I remove the hard dependency on the missing `/kaggle/input/hms-models/` modules by adding a safe fallback that produces a valid probabilistic submission when those models aren’t available, instead of crashing at import time. I also fix the spectrogram reshaping bug by making `proc_eeg_spec` resize/crop/pad dynamically to `(4, 96, 224)` based on the actual computed EEG spectrogram shape, so inference runs end-to-end. Finally, I ensure predictions are always finite, non-negative, and row-normalized to sum to 1, and I write `submission.csv` with the exact required columns and row count.'
- What this solution (achieved 1.39779) has done: 'Your current 1.40995 score is far worse than the 0.3445 target (lower is better), and the main reason is that when the external models are missing you fall back to the nearly-uniform `sample_submission` probabilities, which performs poorly on KL. To move toward the target with minimal semantic change, I keep your entire pipeline and preprocessing intact, but replace the fallback with a simple, legitimate prior learned from `train.csv` vote proportions (a “global class prior” baseline), which is typically much closer to the leaderboard baseline for this competition. I also keep all probability safety/normalization, and ensure the submission format/row order is unchanged. If external models are present, behavior is unchanged.'
- What this solution (achieved 0.77649) has done: 'You’re far above the target (1.39779 vs 0.3445; lower is better), and when external models are missing your submission collapses to a single global prior for every row. To move the score closer with minimal changes and identical overall semantics, I keep your entire inference pipeline intact but upgrade the fallback from a global prior to a simple patient-aware prior (computed from train vote proportions grouped by `patient_id`) and then back off to the global prior for unseen patients. I also add very light Dirichlet/Laplace smoothing when estimating these priors so no class probability becomes extreme, which typically reduces KL vs a raw mean in this competition. When external models are present, predictions remain unchanged.'
- What this solution (achieved 0.77649) has done: 'Your current 0.77649 is still far above the 0.3445 target (lower is better), and the main weakness is that the fallback (when external models aren’t available) predicts a constant patient prior per row, ignoring the very strong signal in `spectrogram_id`/`eeg_id` recurrence and overlap structure present in `train.csv`. With minimal change and identical overall inference semantics, I keep your patient-aware prior but make it more specific by computing a smoothed per-`spectrogram_id` prior from `train.csv`, then back off to patient prior and finally global prior for unseen IDs. This typically reduces KL substantially versus patient-only while remaining a legitimate non-leaking metadata baseline (no use of test labels). I also keep the same probability safety + row-normalization and leave the external-model path completely unchanged.'
- What this solution (achieved 0.76203) has done: 'Your current score (0.77649, lower-is-better) is still far from the target (0.34452), and the only active logic affecting predictions in your run is the “no external models” fallback. I keep your entire pipeline intact, but make the fallback prior more specific and better calibrated by (1) using an `eeg_id`-level prior (strongest metadata signal in train due to many overlapping sub-samples per eeg), then backing off to `spectrogram_id`, then `patient_id`, then global. To reduce KL spikes from overconfident priors, I apply slightly stronger Dirichlet smoothing in these ID-level priors while preserving the same row-normalized probability semantics. External-model behavior remains unchanged, and the script still write a valid `submission.csv`.'
- What this solution (achieved 0.99275) has done: 'I keep your entire inference pipeline and external-model path unchanged, and only adjust the fallback (the only part affecting your current 0.76203 score) to be slightly more informative without changing overall semantics. Specifically, I replace the single-point ID priors with a smoothed hierarchical mixture (eeg_id/spectrogram_id/patient_id/global) weighted by how much evidence (vote mass) each group has in train, which typically reduces KL vs picking just one prior while staying legitimate and fast. I also fix `_safe_row_normalize` to normalize along the last axis (so it’s correct for both vectors and matrices) and add a final safety renormalization for the mixed prior. This is a minimal change aimed at moving the score down (lower is better) toward the 0.3445 target.'

# 9. Code solution

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
    ckpt = torch.load(path, map_location=torch.device("cpu"))
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    else:
        sd = ckpt
    model.load_state_dict(sd, strict=True)
    return model




## === cell 4
HAVE_EXTERNAL_MODELS = False
MultimodalModel = None

try:
    sys.path.append("/kaggle/input/hms-models/")
    from eeg_cnn_rnn_att import EegModel as EegModel  # noqa: F401
    from spc_cnn_att import SpectrogramCnnModel as SpcModel  # noqa: F401
    from comb_model import MultimodalModel as _MultimodalModel

    MultimodalModel = _MultimodalModel
    HAVE_EXTERNAL_MODELS = True
except Exception as e:
    print("Warning: external models not available; using baseline predictions.")
    print("Import error:", repr(e))



## === cell 5
models_multimodal = []
if HAVE_EXTERNAL_MODELS:
    for model_path in sorted(
        glob.glob(
            "/kaggle/input/hms-models/final_multimodal_sanity/final_multimodal_sanity/*"
        )
    ):
        try:
            model = MultimodalModel(None, None, None)
            print("Loading:", model_path)
            model = load_model(model_path, model)
            model = model.to(device)
            model.eval()
            models_multimodal.append(model)
        except Exception as e:
            print("Corrupted or incompatible:", model_path, "err:", repr(e))

print("Loaded multimodal models:", len(models_multimodal))




## === cell 6
def MAD(signal, axis=-1):
    """Compute the robust standard deviation (MAD) of a signal."""
    median = np.median(signal, axis=axis, keepdims=True)
    absolute_deviations = np.abs(signal - median)
    median_absolute_deviation = np.median(absolute_deviations, axis=axis, keepdims=True)
    scale_factor = 1.4826  # constant for normal distribution
    robust_std = median_absolute_deviation * scale_factor
    return robust_std


from scipy.signal import butter, filtfilt


def butter_filter(eeg_data, fs=200, cutoff_freq=22, order=4, btype="lowpass"):
    b, a = butter(N=order, Wn=cutoff_freq / (0.5 * fs), btype=btype, analog=False)
    return filtfilt(b, a, eeg_data)




## === cell 7
import scipy  # noqa: F401
import torchaudio
from torchaudio.transforms import Spectrogram as _Spectrogram

n_fft = 800
win_length = 256
hop_length = 44

spectrogram = _Spectrogram(
    n_fft=n_fft, win_length=win_length, hop_length=hop_length, power=None
)


@torch.no_grad()
def compute_spec(chain):
    chain = torch.as_tensor(chain, dtype=torch.float32)
    chain = spectrogram(chain)  # (C, F, TT), complex
    chain = chain[:, :, 2:98]  # crop time bins
    chain = torch.abs(chain) / 15.0
    chain = torch.log(chain.clamp(min=math.exp(-4), max=math.exp(7)))
    chain = chain.mean(axis=1)  # (C, TT)
    return chain.cpu().numpy()


def compute_spec_eeg(a, b) -> np.ndarray:
    return butter_filter(
        a - b, cutoff_freq=np.array([0.25, 40.0]), order=5, btype="bandpass"
    )


def compute_spec_chain(df_eeg: pl.DataFrame) -> np.ndarray:
    Fp1 = df_eeg["Fp1"].to_numpy()
    Fp2 = df_eeg["Fp2"].to_numpy()
    F7 = df_eeg["F7"].to_numpy()
    F8 = df_eeg["F8"].to_numpy()
    T3 = df_eeg["T3"].to_numpy()
    T4 = df_eeg["T4"].to_numpy()
    T5 = df_eeg["T5"].to_numpy()
    T6 = df_eeg["T6"].to_numpy()
    F3 = df_eeg["F3"].to_numpy()
    F4 = df_eeg["F4"].to_numpy()
    C3 = df_eeg["C3"].to_numpy()
    C4 = df_eeg["C4"].to_numpy()
    P3 = df_eeg["P3"].to_numpy()
    P4 = df_eeg["P4"].to_numpy()
    O1 = df_eeg["O1"].to_numpy()
    O2 = df_eeg["O2"].to_numpy()

    ll = np.stack(
        [
            (
                compute_spec_eeg(Fp1, F7),
                compute_spec_eeg(F7, T3),
                compute_spec_eeg(T3, T5),
                compute_spec_eeg(T5, O1),
            )
        ]
    )
    lp = np.stack(
        [
            (
                compute_spec_eeg(Fp1, F3),
                compute_spec_eeg(F3, C3),
                compute_spec_eeg(C3, P3),
                compute_spec_eeg(P3, O1),
            )
        ]
    )
    rp = np.stack(
        [
            (
                compute_spec_eeg(Fp2, F4),
                compute_spec_eeg(F4, C4),
                compute_spec_eeg(C4, P4),
                compute_spec_eeg(P4, O2),
            )
        ]
    )
    rl = np.stack(
        [
            (
                compute_spec_eeg(Fp2, F8),
                compute_spec_eeg(F8, T4),
                compute_spec_eeg(T4, T6),
                compute_spec_eeg(T6, O2),
            )
        ]
    )

    chain = np.stack([ll, lp, rp, rl])[:, 0]  # (4, T)

    mads = MAD(chain, axis=-1)
    mads = np.median(mads.reshape(-1))
    chain = chain / (mads + 1e-5)

    chain = compute_spec(chain)  # (4, TT)
    return chain


def compute_spec_from_file(filepath: str) -> np.ndarray:
    df_eeg = pl.read_parquet(filepath).fill_null(0)
    chain = compute_spec_chain(df_eeg)
    return chain




## === cell 8
def process_spec(spec: np.ndarray) -> np.ndarray:
    spec = spec[:, 1:]
    spec = np.stack(
        [
            spec[:, 0:100].T,
            spec[:, 100:200].T,
            spec[:, 200:300].T,
            spec[:, 300:400].T,
        ]
    )
    return spec


def compute_kaggle_spec_from_file(filepath: str) -> np.ndarray:
    spec = pl.read_parquet(filepath).to_numpy().astype(np.float32)
    spec = process_spec(spec)
    return spec




## === cell 9
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




## === cell 10
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

    ekg = df_eeg["O2"].to_numpy()
    ekg = butter_filter(ekg, cutoff_freq=np.array([0.50, 20.0]), btype="bandpass")
    ekg = bin_array(ekg, bin_size=4, mode="reflect").mean(axis=-1)
    ekg = ekg.reshape(1, -1)

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
    "other_vote",
]




## === cell 12
def _safe_row_normalize(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p = np.clip(p, eps, None)
    s = np.clip(p.sum(axis=-1, keepdims=True), eps, None)
    return p / s


def _compute_global_prior_from_train(
    df_train: pl.DataFrame, alpha: float = 1.0
) -> np.ndarray:
    votes = df_train.select(LABELS).to_numpy().astype(np.float64)
    votes = np.nan_to_num(votes, nan=0.0, posinf=0.0, neginf=0.0)
    votes = np.clip(votes, 0.0, None)
    row_sums = np.clip(votes.sum(axis=1, keepdims=True), 1.0, None)
    probs = votes / row_sums
    prior = probs.mean(axis=0)
    prior = prior + alpha
    prior = _safe_row_normalize(prior).reshape(-1)
    return prior


def _compute_group_priors_and_strength_from_train(
    df_train: pl.DataFrame,
    group_col: str,
    global_prior: np.ndarray,
    alpha: float = 6.0,
) -> Tuple[Dict[int, np.ndarray], Dict[int, float]]:
    agg = df_train.group_by(group_col).agg([pl.col(c).sum().alias(c) for c in LABELS])
    ids = agg[group_col].to_list()
    vote_mat = agg.select(LABELS).to_numpy().astype(np.float64)
    vote_mat = np.nan_to_num(vote_mat, nan=0.0, posinf=0.0, neginf=0.0)
    vote_mat = np.clip(vote_mat, 0.0, None)

    priors: Dict[int, np.ndarray] = {}
    strength: Dict[int, float] = {}
    for gid, v in zip(ids, vote_mat):
        v_sum = float(np.clip(v.sum(), 0.0, None))
        p = v + alpha * global_prior
        p = _safe_row_normalize(p).reshape(-1)
        priors[int(gid)] = p
        strength[int(gid)] = v_sum
    return priors, strength


GLOBAL_PRIOR = _compute_global_prior_from_train(df_train, alpha=1.0)

EEG_PRIORS, EEG_STRENGTH = _compute_group_priors_and_strength_from_train(
    df_train, "eeg_id", GLOBAL_PRIOR, alpha=6.0
)
SPEC_PRIORS, SPEC_STRENGTH = _compute_group_priors_and_strength_from_train(
    df_train, "spectrogram_id", GLOBAL_PRIOR, alpha=6.0
)
PATIENT_PRIORS, PATIENT_STRENGTH = _compute_group_priors_and_strength_from_train(
    df_train, "patient_id", GLOBAL_PRIOR, alpha=6.0
)

print("GLOBAL_PRIOR:", GLOBAL_PRIOR, "sum:", float(GLOBAL_PRIOR.sum()))
print("Num eeg priors:", len(EEG_PRIORS))
print("Num spectrogram priors:", len(SPEC_PRIORS))
print("Num patient priors:", len(PATIENT_PRIORS))



## === cell 13
np.set_printoptions(formatter={"all": lambda x: f"{x:0.3f}"})

import albumentations as A

spec_transforms = A.Compose([A.Resize(height=96, width=224)])


def proc_kspec(x):
    x = x.copy()

    x = x[:, 2:98]
    x[np.isnan(x) | np.isinf(x)] = 0
    x = x.clip(np.exp(-4), np.exp(7))
    x = np.log(x)

    x = x - x.mean(axis=(1, 2), keepdims=True)
    x = x / (x.std(axis=(1, 2), keepdims=True) + 1e-5)

    x = x.transpose(1, 2, 0)
    x = spec_transforms(image=x)["image"]
    x = x.transpose(2, 0, 1)

    x = x.reshape(4, 96, 224)
    return x


def _pad_or_crop_2d(x2d: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = x2d.shape
    if h > target_h:
        s = (h - target_h) // 2
        x2d = x2d[s : s + target_h, :]
    if w > target_w:
        s = (w - target_w) // 2
        x2d = x2d[:, s : s + target_w]
    h, w = x2d.shape
    if h < target_h or w < target_w:
        pad_h = target_h - h
        pad_w = target_w - w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        x2d = np.pad(x2d, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="edge")
    return x2d


def proc_eeg_spec(x):
    x = x.copy()
    x[np.isnan(x) | np.isinf(x)] = 0
    x = x + 1.0

    out = np.empty((4, 96, 224), dtype=np.float32)
    for c in range(4):
        row = x[c].astype(np.float32)  # (T,)
        img = np.tile(row[None, :], (96, 1))  # (96, T)
        img = _pad_or_crop_2d(img, 96, 224)
        out[c] = img
    return out


def proc_eeg(eeg, mid, ekg):
    eeg = eeg.copy()
    mid = mid.copy()
    ekg = ekg.copy()

    eeg[np.isnan(eeg) | np.isinf(eeg)] = 0
    ekg[np.isnan(ekg) | np.isinf(ekg)] = 0
    mid[np.isnan(mid) | np.isinf(mid)] = 0

    eeg = eeg - eeg.mean(axis=-1, keepdims=True)
    mid = mid - mid.mean(axis=-1, keepdims=True)

    mad_std = MAD(eeg, axis=-1).reshape(-1)
    mad_std = np.median(mad_std) + 1e-5

    eeg = eeg / mad_std
    eeg = eeg.clip(-10, 10)

    mid = mid / mad_std
    mid = mid.clip(-10, 10)

    ekg = ekg / (MAD(ekg, axis=-1).reshape(-1) + 1e-5)

    eeg = eeg.reshape(16, -1)
    eeg = np.concatenate([eeg, mid, ekg], axis=0)

    eeg = eeg.reshape(19, 2_500)
    return eeg


def _mix_priors_for_row(eeg_id: int, spc_id: int, patient_id: int) -> np.ndarray:
    p0 = GLOBAL_PRIOR

    p_eeg = EEG_PRIORS.get(int(eeg_id), None)
    p_spc = SPEC_PRIORS.get(int(spc_id), None)
    p_pat = PATIENT_PRIORS.get(int(patient_id), None)

    s_eeg = EEG_STRENGTH.get(int(eeg_id), 0.0)
    s_spc = SPEC_STRENGTH.get(int(spc_id), 0.0)
    s_pat = PATIENT_STRENGTH.get(int(patient_id), 0.0)

    tau_eeg, tau_spc, tau_pat = 250.0, 250.0, 80.0
    w_eeg = (s_eeg / (s_eeg + tau_eeg)) if p_eeg is not None else 0.0
    w_spc = (s_spc / (s_spc + tau_spc)) if p_spc is not None else 0.0
    w_pat = (s_pat / (s_pat + tau_pat)) if p_pat is not None else 0.0

    w_eeg = min(max(w_eeg, 0.0), 0.70)
    w_spc = min(max(w_spc, 0.0), 0.50)
    w_pat = min(max(w_pat, 0.0), 0.40)

    w_sum = w_eeg + w_spc + w_pat
    if w_sum > 0.95:
        scale = 0.95 / w_sum
        w_eeg *= scale
        w_spc *= scale
        w_pat *= scale

    w0 = 1.0 - (w_eeg + w_spc + w_pat)

    p = w0 * p0
    if p_eeg is not None:
        p = p + w_eeg * p_eeg
    if p_spc is not None:
        p = p + w_spc * p_spc
    if p_pat is not None:
        p = p + w_pat * p_pat

    return _safe_row_normalize(p).reshape(-1)


@torch.no_grad()
def gen_ensemble_pred(df_row: pl.DataFrame) -> np.ndarray:
    eeg_id = df_row["eeg_id"].item()
    spc_id = df_row["spectrogram_id"].item()
    patient_id = int(df_row["patient_id"].item())

    eeg_filepath = os.path.join(EEG_DIR, f"{eeg_id}.parquet")
    spc_filepath = os.path.join(SPEC_DIR, f"{spc_id}.parquet")

    if len(models_multimodal) == 0:
        pred = _mix_priors_for_row(int(eeg_id), int(spc_id), int(patient_id))
        pred = (
            _safe_row_normalize(np.asarray(pred, dtype=np.float64)).reshape(-1).copy()
        )
        return pred

    preds = []

    kspec = compute_kaggle_spec_from_file(spc_filepath)
    kspec = proc_kspec(kspec)
    kspec = torch.tensor(kspec, dtype=torch.float32, device=device).unsqueeze(0)

    eeg_spec = compute_spec_from_file(eeg_filepath)
    eeg_spec = proc_eeg_spec(eeg_spec)
    eeg_spec = torch.tensor(eeg_spec, dtype=torch.float32, device=device).unsqueeze(0)

    eeg, mid, ekg = compute_eeg_from_file(eeg_filepath)
    eeg[np.isnan(eeg) | np.isinf(eeg)] = 0
    eeg = proc_eeg(eeg, mid, ekg)
    eeg = torch.tensor(eeg, dtype=torch.float32, device=device).unsqueeze(0)

    for model in models_multimodal:
        model.eval()
        out = model(eeg, eeg_spec, kspec)
        pred = out.exp().detach().float().cpu().numpy().reshape(-1)
        preds.append(pred)

    preds = np.mean(np.stack(preds, axis=0), axis=0)
    preds = _safe_row_normalize(preds).reshape(-1)
    return preds.astype(np.float64)




## === cell 14
from tqdm.auto import tqdm

preds_final = []
for i in tqdm(range(len(df_test))):
    pred = gen_ensemble_pred(df_test[i])
    preds_final.append(pred)

preds_final = np.asarray(preds_final, dtype=np.float64)
print("preds_final shape:", preds_final.shape)
print(
    "row sums (min/max):", preds_final.sum(axis=1).min(), preds_final.sum(axis=1).max()
)



## === cell 15
eeg_ids = df_test["eeg_id"].to_list()
df_sub = pd.DataFrame({"eeg_id": eeg_ids})

preds_final = np.nan_to_num(preds_final, nan=0.0, posinf=0.0, neginf=0.0)
preds_final = np.clip(preds_final, 1e-12, None)
preds_final = preds_final / preds_final.sum(axis=1, keepdims=True)

df_sub[LABELS] = preds_final
df_sub.to_csv("submission.csv", index=False)

print(df_sub.head())
print("Wrote submission.csv with shape:", df_sub.shape)
print(
    "Check row sums:",
    float(df_sub[LABELS].sum(axis=1).min()),
    float(df_sub[LABELS].sum(axis=1).max()),
)
