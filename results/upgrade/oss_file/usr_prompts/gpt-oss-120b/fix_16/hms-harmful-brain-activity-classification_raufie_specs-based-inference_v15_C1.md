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

# 5. Code solution

## === cell 0
import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm


class config:
    BATCH_SIZE = 64
    MODEL = "tf_efficientnet_b2"
    NUM_WORKERS = 0
    PRINT_FREQ = 20
    SEED = 20
    VISUALIZE = False
    TEMPERATURE = 2.0
    EEG_WEIGHT = 0.1
    PATIENT_WEIGHT = 0.2
    GLOBAL_WEIGHT = 0.7  # must sum to 1.0 with the other two
    SMOOTHING = 1.0  # Laplace smoothing added to vote counts


torch.manual_seed(config.SEED)
np.random.seed(config.SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




## === cell 1
train_path = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
test_path = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
sample_submission_path = (
    "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
)

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

vote_cols = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

global_votes = train_df[vote_cols].sum(axis=0).values.astype(np.float32)
global_votes += config.SMOOTHING  # Laplace smoothing
class_prior = global_votes / global_votes.sum()  # shape (6,)

eeg_agg = train_df.groupby("eeg_id")[vote_cols].sum()
eeg_prior_dict = {}
for eid, row in eeg_agg.iterrows():
    arr = row.values.astype(np.float32) + config.SMOOTHING
    if arr.sum() > 0:
        eeg_prior_dict[int(eid)] = arr / arr.sum()
    else:
        eeg_prior_dict[int(eid)] = class_prior

patient_agg = train_df.groupby("patient_id")[vote_cols].sum()
patient_prior_dict = {}
for pid, row in patient_agg.iterrows():
    arr = row.values.astype(np.float32) + config.SMOOTHING
    if arr.sum() > 0:
        patient_prior_dict[int(pid)] = arr / arr.sum()
    else:
        patient_prior_dict[int(pid)] = class_prior

eid_to_patient = dict(zip(test_df["eeg_id"].values, test_df["patient_id"].values))


def _apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """
    Apply temperature scaling. With temperature == 1.0 this returns the input unchanged.
    """
    if temperature == 1.0:
        return probs
    eps = 1e-12
    logp = np.log(probs + eps)
    scaled = np.exp(logp / temperature)
    return scaled / scaled.sum()


def get_fallback_probs(eid, pid):
    """
    Return a calibrated probability vector for a test sample.
    - Blend per‑EEG, per‑patient and global priors according to the current config.
    - Apply temperature scaling and renormalise.
    """
    eeg_prior = eeg_prior_dict.get(int(eid))
    patient_prior = patient_prior_dict.get(int(pid))

    if eeg_prior is None and patient_prior is None:
        blended = class_prior
    else:
        blended = np.zeros_like(class_prior)
        if eeg_prior is not None:
            blended += config.EEG_WEIGHT * eeg_prior
        if patient_prior is not None:
            blended += config.PATIENT_WEIGHT * patient_prior
        blended += config.GLOBAL_WEIGHT * class_prior
        blended = blended / blended.sum()

    smoothed = _apply_temperature(blended, config.TEMPERATURE)
    return smoothed / smoothed.sum()


def _kl_divergence(p_true: np.ndarray, p_pred: np.ndarray) -> float:
    """Mean KL divergence for a single sample."""
    eps = 1e-12
    p_true = np.clip(p_true, eps, 1)
    p_pred = np.clip(p_pred, eps, 1)
    return np.sum(p_true * np.log(p_true / p_pred))


print("Running lightweight calibration to approach target score...")

eeg_weights = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
patient_weights = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
temperatures = [1.0, 1.5, 2.0, 2.5, 3.0]

best_score = float("inf")
best_cfg = None

train_eids = train_df["eeg_id"].values
train_pids = train_df["patient_id"].values
true_probs = train_df[vote_cols].values.astype(np.float32)  # raw vote counts
row_sums = true_probs.sum(axis=1, keepdims=True)
true_probs = np.where(row_sums == 0, class_prior, true_probs / row_sums)

for ew in eeg_weights:
    for pw in patient_weights:
        if ew + pw > 0.9:  # ensure global weight stays >= 0.1
            continue
        gw = 1.0 - ew - pw
        for temp in temperatures:
            config.EEG_WEIGHT = ew
            config.PATIENT_WEIGHT = pw
            config.GLOBAL_WEIGHT = gw
            config.TEMPERATURE = temp

            preds = np.vstack(
                [
                    get_fallback_probs(eid, pid)
                    for eid, pid in zip(train_eids, train_pids)
                ]
            )
            kl_vals = np.array(
                [_kl_divergence(true_probs[i], preds[i]) for i in range(len(preds))]
            )
            mean_kl = kl_vals.mean()

            if mean_kl < best_score:
                best_score = mean_kl
                best_cfg = (ew, pw, gw, temp)

if best_cfg is not None:
    (
        config.EEG_WEIGHT,
        config.PATIENT_WEIGHT,
        config.GLOBAL_WEIGHT,
        config.TEMPERATURE,
    ) = best_cfg
    print(
        f"Best calibration: EEG_WEIGHT={config.EEG_WEIGHT}, PATIENT_WEIGHT={config.PATIENT_WEIGHT}, "
        f"GLOBAL_WEIGHT={config.GLOBAL_WEIGHT}, TEMPERATURE={config.TEMPERATURE} => mean KL={best_score:.5f}"
    )
else:
    print("No calibration improvement found; keeping default config.")

test_eids = test_df["eeg_id"].values
test_pids = [eid_to_patient.get(eid, -1) for eid in test_eids]

pred_array = np.vstack(
    [get_fallback_probs(eid, pid) for eid, pid in zip(test_eids, test_pids)]
).astype(np.float32)

row_sums = pred_array.sum(axis=1, keepdims=True)
pred_array = pred_array / row_sums

submission = pd.read_csv(sample_submission_path)  # preserves column order
submission[vote_cols] = pred_array
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)

print(f"Submission file written to {submission_path} with shape {submission.shape}")
