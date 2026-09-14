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

0.4054159776631572

# 6. Current score

0.76992

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fixed the import error by removing the unavailable custom model modules and replaced the ensemble inference with a simple uniform‑probability baseline, which guarantees a valid submission and avoids shape mismatches. I also corrected the test‑set iteration (Polars → pandas) and adjusted the submission construction so the predictions align with the required columns.'
- What this solution (achieved 1.39779) has done: 'I keep the overall pipeline unchanged but replace the uniform‑baseline probabilities with class‑prior probabilities derived from the training label distribution. Computing the average vote‑based distribution once from train.csv and returning that vector for every test row gives a simple, still‑valid baseline that is expected to lower the KL divergence toward the target score.'
- What this solution (achieved 1.39779) has done: 'I compute a per‑`eeg_id` probability distribution from the training data and use it for each matching test row (falling back to the overall class prior when an `eeg_id` is unseen). This keeps the pipeline unchanged while giving more informative predictions, which should move the KL‑divergence score closer to the target.'
- What this solution (achieved 1.68479) has done: 'Implemented fixes to resolve Polars‑related errors, define prior dictionaries correctly, and ensure a valid submission is written.

Key changes:
- Switched data loading and aggregation to pandas (removing unsupported Polars groupby usage).
- Properly built `CLASS_PRIOR`, `PER_EEG_PRIOR_DICT`, and `PER_PATIENT_PRIOR_DICT`.
- Updated `gen_ensemble_pred` to work with a pandas row.
- Adjusted the prediction loop to use pandas rows and guarantee `preds_array` creation.
- Created the submission CSV with the required columns and ensured probabilities sum to 1.'
- What this solution (achieved 1.08124) has done: 'I add vote‑count based smoothing so that per‑eeg and per‑patient priors are blended with the global class prior rather than used outright. This reduces noisy predictions for IDs with few votes, moving the KL‑divergence closer to the target (lower score). The changes only extend the prior‑building step and adjust `gen_ensemble_pred` to apply the weighted blend.'
- What this solution (achieved 1.68479) has done: 'The ensemble prediction is changed to rely directly on the most specific available prior (per‑eeg, then per‑patient, then global). This removes the conservative blending that was pulling many predictions toward the overall class prior and should substantially lower the KL‑divergence, moving the score closer to the target while keeping the original pipeline intact.'
- What this solution (achieved 0.76992) has done: 'I add a simple Bayesian smoothing step that blends the per‑eeg or per‑patient prior with the global class prior based on the amount of vote data available. This reduces noisy predictions for IDs with few votes, moving the KL‑divergence closer to the target while keeping the overall logic unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch

DATA_DIR = "/kaggle/input/hms-harmful-brain-activity-classification/"
df_train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
df_test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sample_submission = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

LABELS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

total_votes = df_train[LABELS].sum()
total_sum = total_votes.sum()
CLASS_PRIOR = (total_votes / total_sum).values.astype(np.float32)

SMOOTHING = 10.0

eeg_group = df_train.groupby("eeg_id")[LABELS].sum()
eeg_sums = eeg_group.sum(axis=1)  # total votes per eeg_id
PER_EEG_PRIOR_DICT = {}
PER_EEG_COUNT_DICT = {}
for eid, votes_row in eeg_group.iterrows():
    s = eeg_sums.loc[eid]
    PER_EEG_COUNT_DICT[eid] = s
    if s > 0:
        PER_EEG_PRIOR_DICT[eid] = votes_row.values.astype(np.float32) / s
    else:
        PER_EEG_PRIOR_DICT[eid] = CLASS_PRIOR.copy()

patient_group = df_train.groupby("patient_id")[LABELS].sum()
patient_sums = patient_group.sum(axis=1)  # total votes per patient_id
PER_PATIENT_PRIOR_DICT = {}
PER_PATIENT_COUNT_DICT = {}
for pid, votes_row in patient_group.iterrows():
    s = patient_sums.loc[pid]
    PER_PATIENT_COUNT_DICT[pid] = s
    if s > 0:
        PER_PATIENT_PRIOR_DICT[pid] = votes_row.values.astype(np.float32) / s
    else:
        PER_PATIENT_PRIOR_DICT[pid] = CLASS_PRIOR.copy()




## === cell 1
@torch.no_grad()
def gen_ensemble_pred(row) -> np.ndarray:
    """
    Return a probability vector for the given test row.
    The prediction uses the most specific prior available, blended with the global class prior.
    Blending weight = count / (count + SMOOTHING), so IDs with few votes fall back toward the global prior.
    """
    eeg_id = row["eeg_id"]
    if eeg_id in PER_EEG_PRIOR_DICT:
        specific = PER_EEG_PRIOR_DICT[eeg_id]
        count = PER_EEG_COUNT_DICT.get(eeg_id, 0)
        weight = count / (count + SMOOTHING) if count > 0 else 0.0
        prob = weight * specific + (1.0 - weight) * CLASS_PRIOR
    else:
        patient_id = row.get("patient_id")
        if patient_id is not None and patient_id in PER_PATIENT_PRIOR_DICT:
            specific = PER_PATIENT_PRIOR_DICT[patient_id]
            count = PER_PATIENT_COUNT_DICT.get(patient_id, 0)
            weight = count / (count + SMOOTHING) if count > 0 else 0.0
            prob = weight * specific + (1.0 - weight) * CLASS_PRIOR
        else:
            prob = CLASS_PRIOR.copy()
    prob_sum = prob.sum()
    if prob_sum > 0:
        prob = prob / prob_sum
    return prob.astype(np.float32)




## === cell 2
preds_final = []
for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
    pred = gen_ensemble_pred(row)
    preds_final.append(pred)
preds_array = np.vstack(preds_final)  # shape (n_test, 6)




## === cell 3
submission = pd.DataFrame({"eeg_id": df_test["eeg_id"]})
for i, label in enumerate(LABELS):
    submission[label] = preds_array[:, i]

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
print(submission.head())
