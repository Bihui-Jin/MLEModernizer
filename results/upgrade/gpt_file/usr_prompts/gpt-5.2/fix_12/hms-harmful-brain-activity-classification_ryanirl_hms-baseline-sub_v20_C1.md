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

0.5958003092721826

# 6. Current score

1.82601

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'I first fix the import failure by removing the dependency on `/kaggle/input/hms-models/` (which isn’t available) and replace it with a safe, minimal fallback that still produces valid probability predictions for all 6 classes. Next, I fix the submission construction error by ensuring `preds_final` is a proper 2D NumPy array of shape `(n_test, 6)` before assigning to the DataFrame. Finally, I guarantee each row sums to 1 (required by the metric/submission checker) and write `submission.csv` in the working directory.'
- What this solution (achieved 1.34414) has done: 'I fix the Polars row-selection bug by switching from `df_train[i]`/`df_test[i]` (which selects columns in Polars) to safe row extraction via `df.row(i, named=True)` and by passing an `eeg_id` into `gen_ensemble_pred` instead of a “row DataFrame”. This unblocks model fitting, makes `gen_ensemble_pred` actually defined, and lets the test-loop run end-to-end. Then I ensure `preds_final` is always a proper `(n_test, 6)` float array so `df_sub[LABELS] = preds_final` succeeds, and I keep the probability normalization (row sums to 1) required by the submission checker. These changes are execution/correctness fixes and keep the existing core fallback-model logic intact.'
- What this solution (achieved 1.26706) has done: 'Your current ridge-on-bandpower fallback is likely underfitting and also slightly misaligned with the competition objective (KL on vote-distribution), so we make two minimal, metric-aligned improvements without changing the overall approach. First, we compute the bandpower features only on the **central 10 seconds** of each 50s EEG (same region the labels correspond to), which usually improves KL noticeably while preserving the same feature extractor and model. Second, we make the ridge fit use a slightly larger (but still bounded) training subset and increase the prior-mix a bit for better calibration/stability, which tends to reduce extreme/confident errors that hurt KL. The rest (ridge-on-log-targets + softmax + normalization + submission writing) stays the same.'
- What this solution (achieved 1.5529) has done: 'Your current ridge-on-bandpower fallback is already metric-aware, so the most likely way to reduce KL (lower-is-better) with minimal core changes is to (1) align the train feature extraction to the same annotated 10-second region using `eeg_label_offset_seconds` instead of always center-cropping, (2) weight training examples by their total vote count (more reliable targets) in the ridge closed-form (still ridge, same loss semantics on log-probs), and (3) apply a tiny temperature smoothing plus the same prior-mix to reduce overconfident errors that inflate KL. These are small, localized changes that keep the same overall pipeline (bandpower -> ridge on log targets -> softmax -> prior mix -> normalized submission) and should move your score down toward the 0.5958 target. The script still runs end-to-end and writes `submission.csv` with valid per-row probability sums.'
- What this solution (achieved 1.48404) has done: 'Your current score (1.5529, lower-is-better) is far from the target (0.5958), so we should improve actual predictive quality but with minimal, localized edits that keep the same ridge-on-log-probs core logic. The biggest issue is that your training features are extracted from the *annotated 10s window* (via `eeg_label_offset_seconds`), while test features are extracted from the *center 10s*; we reduce this train/test mismatch by training on the same “center 10s” cropping as test (still 10s bandpower, same feature extractor and ridge). Then we (slightly) increase the training subset size to reduce underfitting/variance without changing the modeling approach. Finally, we make the softmax numerically correct (row-wise) to avoid subtle probability distortions that can worsen KL.'
- What this solution (achieved 2.18375) has done: 'We keep your ridge-on-log-prob + softmax + prior-mix pipeline intact, but fix two small issues that are likely hurting KL. First, the EKG channel is incorrectly taken from `O2` in `compute_eeg_chain`; correcting it to use the actual `EKG` column (when present) avoids contaminating features and improves signal quality without changing the modeling approach. Second, we reduce train/test distribution mismatch by fitting the ridge on a per-`eeg_id` aggregated target (mean vote distribution across all overlapping subsamples for the same EEG) and extracting features once per `eeg_id`; this matches the test’s single row per `eeg_id` and usually reduces KL materially while staying within the same feature extractor + ridge core logic. Everything else (feature type, ridge closed-form on log-targets, softmax, prior mixing, normalization, submission writing) stays the same and still runs end-to-end to produce `submission.csv`.'
- What this solution (achieved 1.71284) has done: 'Your current score (2.18375, lower-is-better) is much worse than the target (0.5958), so we should improve real predictive quality with the smallest possible changes while keeping the same bandpower→ridge-on-log-probs→softmax→prior-mix pipeline. The biggest issue is that `_make_train_features_and_targets` is currently training on a random subset of EEGs but **pairs them with vote means that are not time-aligned to the 10s label window**, and it also ignores `eeg_label_offset_seconds` entirely; we fix this by sampling actual rows from `train.csv` and cropping EEG using that row’s offset (same semantics as labels). To reduce label noise from overlapping subsamples without changing the model, we also aggregate by `label_id` (which represents one set of votes) rather than by `eeg_id` mean, then extract features at the corresponding offset. Finally, we keep your existing calibration steps but make them slightly less “flattened” (temperature closer to 1 and a smaller prior-mix) to avoid washing out signal, which should materially lower KL while staying within the same core logic.'
- What this solution (achieved 1.77446) has done: 'Your current gap to the target is large (1.71284 vs 0.5958, lower-is-better), so we should improve real predictive quality with minimal, metric-aligned edits while keeping the same bandpower → ridge-on-log-probs → softmax → prior-mix pipeline. The biggest low-risk win is to reduce feature noise by using more of the available EEG channels (your extractor currently uses only 8 channels) while keeping the same bandpower features and ridge training unchanged. We also make the ridge slightly more stable by using the *vote_sum* as a proper weight (sum of votes per label_id) instead of a mean-of-sums aggregation bug, without changing the modeling approach. Finally, we keep your existing calibration (temperature + prior-mix) but make it a touch less flattened so signal isn’t washed out, which typically reduces KL.'
- What this solution (achieved 1.93878) has done: 'Your current score (1.774, lower-is-better) is far from the target (0.596), so we should improve real predictive quality with the smallest possible edits while keeping the same bandpower→ridge-on-log-probs→softmax→prior-mix pipeline. The biggest low-risk gain is to fix the `vote_sum` bug in the `label_id` aggregation (it currently uses `sum_horizontal(...).first()` which is not the intended per-row sum), and to use `vote_sum` consistently as the sample weight. Next, we reduce train/test mismatch by aggregating labels at the `eeg_id` level (test has one row per `eeg_id`) using a vote-weighted mean target and extracting one feature vector per `eeg_id` (still the same features and ridge). Finally, we keep your calibration but make it slightly less flattened (remove the tiny temperature and reduce prior-mix) to preserve signal, which typically reduces KL.'
- What this solution (achieved 1.82601) has done: 'We keep your bandpower → ridge-on-log-probs → softmax → prior-mix pipeline exactly the same, but fix two small, score-relevant mismatches that likely inflate KL. First, we align training feature extraction to the test distribution by using the *same center 10s crop* for training too (right now train uses a vote-weighted label offset while test uses center crop), which is a minimal change that typically lowers KL. Second, we make the ridge targets consistent with the `eeg_id` aggregation by using vote-weighted class probabilities computed from summed votes (instead of ratios-of-sums), and use the same `vote_sum` weights; this reduces label-noise/calibration errors without changing the model. Everything else (feature extractor, ridge solve, softmax, prior mix, normalization, submission writing) stays intact and still produces a valid `submission.csv`.'

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
    model.load_state_dict(
        torch.load(path, map_location=torch.device("cpu"))["model_state_dict"]
    )
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

train_votes = df_train.select(LABELS).to_numpy()
train_votes = np.clip(train_votes, 0, None)
vote_sums = train_votes.sum(axis=1, keepdims=True)
vote_sums[vote_sums == 0] = 1.0
train_probs = train_votes / vote_sums

prior = train_probs.mean(axis=0)
prior = np.clip(prior, 1e-6, None)
prior = prior / prior.sum()
prior




## === cell 5
def MAD(signal, axis=-1):
    """Compute the robust standard deviation (MAD) of a signal."""
    median = np.median(signal, axis=axis, keepdims=True)
    absolute_deviations = np.abs(signal - median)
    median_absolute_deviation = np.median(absolute_deviations, axis=axis, keepdims=True)
    scale_factor = 1.4826  # constant for normal distribution
    robust_std = median_absolute_deviation * scale_factor
    return robust_std


from scipy.signal import welch
from scipy.stats import linregress
from scipy.signal import butter, filtfilt


def butter_filter(eeg_data, fs=200, cutoff_freq=22, order=4, btype="lowpass"):
    b, a = butter(N=order, Wn=cutoff_freq / (0.5 * fs), btype=btype, analog=False)
    return filtfilt(b, a, eeg_data)




## === cell 6
import scipy

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
    chain = np.stack([ll, lp, rp, rl])[:, 0]

    mads = MAD(chain, axis=-1)
    mads = np.median(mads.reshape(-1))
    chain = chain / (mads + 1e-5)

    outer = []
    for i in range(4):
        inner = []
        for j in range(4):
            inner.append(compute_spec(chain[i, j]))
        outer.append(inner)

    chain = np.array(outer)
    chain = np.log(chain.clip(np.exp(-4), np.exp(8)))
    chain = chain.mean(axis=1, keepdims=True)
    return chain


def compute_spec_from_file(filepath: str) -> np.ndarray:
    df_eeg = pl.read_parquet(filepath).fill_null(0)
    chain = compute_spec_chain(df_eeg)
    return chain




## === cell 7
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




## === cell 10
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


def proc_3(eeg):
    eeg = eeg.copy()
    eeg[np.isnan(eeg) | np.isinf(eeg)] = 0
    eeg = eeg - eeg.mean(axis=-1, keepdims=True)

    mad_std = MAD(eeg, axis=-1).reshape(-1)
    mad_std = np.median(mad_std) + 1e-5
    eeg = eeg / mad_std
    eeg = eeg.clip(-10, 10)
    eeg = eeg.reshape(4, 4, 2_500)
    return eeg


def proc_4(eeg, mid, ekg):
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


def flip_h(eeg):
    eeg = eeg.copy()
    eeg = eeg[::-1]
    return eeg.copy()


def flip_v(eeg):
    eeg = eeg.copy()
    eeg = eeg[:, ::-1]
    return eeg.copy()


EPS = 1e-12


def _safe_softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = z[None, :]
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(z)
    s = ez.sum(axis=1, keepdims=True)
    bad = (~np.isfinite(s)) | (s <= 0)
    p = ez / np.clip(s, 1e-300, None)
    if np.any(bad):
        p[bad.reshape(-1), :] = prior.copy()[None, :]
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return p.squeeze(0)


def _crop_df_eeg_by_offset(
    df_eeg: pl.DataFrame,
    offset_seconds: Optional[float],
    fs: int = 200,
    seconds: int = 10,
) -> pl.DataFrame:
    n = df_eeg.height
    need = int(fs * seconds)
    if n <= need:
        return df_eeg
    if offset_seconds is None or not np.isfinite(offset_seconds):
        start = max(0, (n - need) // 2)
        return df_eeg.slice(start, need)
    start = int(round(float(offset_seconds) * fs))
    start = max(0, min(start, n - need))
    return df_eeg.slice(start, need)


def _center_crop_df_eeg(
    df_eeg: pl.DataFrame, fs: int = 200, seconds: int = 10
) -> pl.DataFrame:
    n = df_eeg.height
    need = int(fs * seconds)
    if n <= need:
        return df_eeg
    start = max(0, (n - need) // 2)
    return df_eeg.slice(start, need)


def _bandpower_features_from_df(
    df_eeg: pl.DataFrame, offset_seconds: Optional[float] = None
) -> np.ndarray:
    cand_cols = [
        "Fp1",
        "Fp2",
        "F3",
        "F4",
        "C3",
        "C4",
        "P3",
        "P4",
        "O1",
        "O2",
        "F7",
        "F8",
        "T3",
        "T4",
        "T5",
        "T6",
        "Fz",
        "Cz",
        "Pz",
    ]
    cols = [c for c in cand_cols if c in df_eeg.columns]
    if len(cols) == 0:
        return np.zeros(11, dtype=np.float32)

    if offset_seconds is None:
        df_eeg = _center_crop_df_eeg(df_eeg, fs=200, seconds=10)
    else:
        df_eeg = _crop_df_eeg_by_offset(
            df_eeg, offset_seconds=offset_seconds, fs=200, seconds=10
        )

    X = df_eeg.select(cols).to_numpy()
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    X = X - np.median(X, axis=0, keepdims=True)

    fs = 200
    feats = []
    for ch in range(X.shape[1]):
        f, pxx = welch(X[:, ch], fs=fs, nperseg=512, noverlap=256, scaling="density")

        def bp(lo, hi):
            m = (f >= lo) & (f < hi)
            return float(np.trapz(pxx[m], f[m])) if np.any(m) else 0.0

        delta = bp(0.5, 4.0)
        theta = bp(4.0, 8.0)
        alpha = bp(8.0, 13.0)
        beta = bp(13.0, 30.0)
        total = bp(0.5, 30.0) + EPS

        feats.append(
            [
                delta / total,
                theta / total,
                alpha / total,
                beta / total,
                np.log(total),
            ]
        )

    feats = np.asarray(feats, dtype=np.float32)  # (n_ch, 5)
    f_mean = feats.mean(axis=0)
    f_std = feats.std(axis=0)
    amp = MAD(X.T, axis=1).reshape(-1)
    amp = float(np.median(amp))
    out = np.concatenate(
        [f_mean, f_std, np.array([np.log(amp + EPS)], dtype=np.float32)]
    )
    return out.astype(np.float32)


def _make_train_features_and_targets(max_eegs: int = 30000, seed: int = 0):
    rng = np.random.default_rng(seed)

    df_eeg = (
        df_train.with_columns(
            pl.sum_horizontal([pl.col(c) for c in LABELS]).alias("vote_sum")
        )
        .group_by("eeg_id")
        .agg(
            [
                pl.col("vote_sum").sum().alias("vote_sum"),
                *[pl.col(c).sum().alias(c) for c in LABELS],
            ]
        )
        .with_columns(
            [
                (pl.col(c) / pl.sum_horizontal([pl.col(x) for x in LABELS])).alias(c)
                for c in LABELS
            ]
        )
    )

    n = df_eeg.height
    if n == 0:
        return None, None, None

    idx = rng.choice(n, size=min(max_eegs, n), replace=False)

    X_list, Y_list, W_list = [], [], []
    for i in idx:
        row = df_eeg.row(int(i), named=True)
        eeg_id = int(row["eeg_id"])

        eeg_path = os.path.join(DATA_DIR, "train_eegs", f"{eeg_id}.parquet")
        if not os.path.exists(eeg_path):
            continue

        y = np.array([row[c] for c in LABELS], dtype=np.float64).reshape(-1)
        y = np.clip(y, 1e-12, None)
        y = y / y.sum()

        try:
            df_e = pl.read_parquet(eeg_path).fill_null(0)

            x = _bandpower_features_from_df(df_e, offset_seconds=None)

            if not np.all(np.isfinite(x)):
                continue
        except Exception:
            continue

        X_list.append(x)
        Y_list.append(y)
        W_list.append(max(float(row.get("vote_sum", 1.0)), 1e-6))

    if len(X_list) == 0:
        return None, None, None

    X = np.vstack(X_list).astype(np.float64)
    Y = np.vstack(Y_list).astype(np.float64)
    W = np.asarray(W_list, dtype=np.float64)
    return X, Y, W


def _fit_ridge_multitarget(
    X: np.ndarray, Y: np.ndarray, sample_w: Optional[np.ndarray] = None, l2: float = 1.0
):
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1), dtype=X.dtype)])

    if sample_w is None:
        A = Xb.T @ Xb
        B = Xb.T @ np.log(np.clip(Y, 1e-6, 1.0))
    else:
        w = np.asarray(sample_w, dtype=X.dtype).reshape(-1)
        w = w / (np.mean(w) + 1e-12)
        sw = np.sqrt(np.clip(w, 1e-12, None)).reshape(-1, 1)
        Xw = Xb * sw
        Ylogw = np.log(np.clip(Y, 1e-6, 1.0)) * sw
        A = Xw.T @ Xw
        B = Xw.T @ Ylogw

    A += l2 * np.eye(d + 1, dtype=X.dtype)
    W = np.linalg.solve(A, B)  # (d+1, 6)
    return W


Xtr, Ytr, Str = _make_train_features_and_targets(max_eegs=30000, seed=0)
if Xtr is None:
    W_ridge = None
    X_mean = None
    X_std = None
else:
    X_mean = Xtr.mean(axis=0, keepdims=True)
    X_std = Xtr.std(axis=0, keepdims=True) + 1e-6
    Xtrn = (Xtr - X_mean) / X_std
    W_ridge = _fit_ridge_multitarget(Xtrn, Ytr, sample_w=Str, l2=2.0)


@torch.no_grad()
def gen_ensemble_pred(eeg_id: int) -> np.ndarray:
    try:
        eeg_path = os.path.join(EEG_DIR, f"{int(eeg_id)}.parquet")
        if (W_ridge is None) or (not os.path.exists(eeg_path)):
            return prior.copy()

        df_eeg = pl.read_parquet(eeg_path).fill_null(0)
        x = _bandpower_features_from_df(df_eeg, offset_seconds=None).astype(
            np.float64, copy=False
        )
        x = (x.reshape(1, -1) - X_mean) / X_std
        xb = np.concatenate([x.reshape(-1), np.array([1.0])], axis=0)  # add bias
        z = xb @ W_ridge  # (6,)

        p = _safe_softmax(z.astype(np.float64))

        p = 0.97 * p + 0.03 * prior

        p = np.clip(p, 1e-12, 1.0)
        p = p / p.sum()
        return p
    except Exception:
        return prior.copy()




## === cell 11
from tqdm.auto import tqdm

test_eeg_ids = df_test["eeg_id"].to_list()

preds_final = np.zeros((len(test_eeg_ids), len(LABELS)), dtype=np.float64)
for i, eeg_id in enumerate(tqdm(test_eeg_ids, total=len(test_eeg_ids))):
    preds_final[i] = gen_ensemble_pred(int(eeg_id))

assert preds_final.shape == (len(df_test), len(LABELS)), preds_final.shape

row_sums = preds_final.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
preds_final = preds_final / row_sums

preds_final[:3], preds_final.shape



## === cell 12
df_sub = pd.DataFrame({"eeg_id": df_test["eeg_id"].to_list()})
df_sub[LABELS] = preds_final

probs = df_sub[LABELS].to_numpy(dtype=np.float64)
probs = np.clip(probs, 1e-12, 1.0)
probs = probs / probs.sum(axis=1, keepdims=True)
df_sub[LABELS] = probs

df_sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_sub.shape)
df_sub.head()
