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

No external packages required in the script and installed.

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

0.4094429960399035

# 6. Current score

0.75126

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'We replace the fragile external conversion and model‑testing calls with a tiny self‑contained baseline: compute the average vote distribution from the training set (after normalising each row to a probability vector) and use that same distribution as the prediction for every test record. This guarantees a correctly formatted `submission.csv` (probabilities sum to 1) and yields a reasonable KL‑divergence that moves the score toward the target without altering any core modelling logic.'
- What this solution (achieved 1.64506) has done: 'Implemented a safe workflow that fixes the indexing error during submission creation and ensures the CSV is correctly written. The key changes are:
- Resetting indices after reindexing patient‑wise probabilities so they align with the test rows.
- Filling missing patient probabilities with the overall mean.
- Normalising each probability row to sum to 1.
- Building the submission DataFrame using aligned Series/DataFrames, then saving it.'
- What this solution (achieved 1.39779) has done: 'I replace the patient‑wise averaging with a simple global‑mean baseline: every test record receives the same probability vector equal to the overall mean vote distribution from the training set. This removes the potentially noisy patient → test alignment and should lower the KL divergence, moving the score closer to the target while keeping the rest of the workflow unchanged.'
- What this solution (achieved 1.64506) has done: 'I replace the global‑mean baseline with a patient‑wise average baseline: for each patient present in the training set I compute the mean normalized vote distribution, use it for test rows of the same patient, and fall back to the overall mean when a patient is unseen. This small, deterministic change keeps the original workflow while providing more specific predictions, which should lower the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 0.92967) has done: 'I blend the patient‑wise probability estimates with the overall global mean (giving more weight to the global mean) before normalising. This keeps the same workflow but reduces over‑specific predictions, which should lower the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 1.39779) has done: 'The update reduces over‑specific predictions by removing the patient‑wise component and using only the global mean distribution (weight = 1.0). This keeps the original workflow while moving the KL‑divergence closer to the target lower score. All other steps remain unchanged, and the script still writes a correctly formatted `submission.csv`.'
- What this solution (achieved 0.81135) has done: 'I adjust the blending of patient‑specific and global probability estimates by giving the patient‑wise component a non‑zero weight (e.g., 0.6) and reducing the global‑only weight accordingly. This keeps the overall workflow unchanged while providing more tailored predictions, which should lower the KL‑divergence toward the target score. The rest of the code remains identical, ensuring a valid CSV is still written.'
- What this solution (achieved 1.10642) has done: 'I lower the patient‑specific blending weight and increase the global weight (to 0.2 / 0.8). This reduces over‑specific predictions, which typically lowers the KL‑divergence and moves the score closer to the target while keeping the original workflow intact. The only change is the two weight constants; all other logic and file handling remain the same.'
- What this solution (achieved 0.73988) has done: 'I adjust the blending weights to give more emphasis to the patient‑specific probabilities (weight = 0.8) and reduce the global‑mean contribution (weight = 0.2). This modest shift should lower the KL‑divergence, moving the score from 1.106 → closer to the target 0.409 while keeping the overall workflow unchanged.'
- What this solution (achieved 1.10642) has done: 'We lower the patient‑specific contribution and raise the global mean weight (weight_specific = 0.2, weight_global = 0.8). This reduces over‑specific predictions, which should decrease the KL‑divergence and move the score closer to the target while keeping the original workflow intact.'
- What this solution (achieved 0.72904) has done: 'I adjust the blending weights to give much more influence to the patient‑specific probability estimates (which previously produced a lower KL score). By setting `weight_specific` to 0.9 and `weight_global` to 0.1 we keep the same workflow while moving the predictions closer to the target score.'
- What this solution (achieved 0.75126) has done: 'I replace the fixed blending weights with an adaptive weighting that gives more global influence for patients with few training records and more patient‑specific influence for well‑represented patients. This keeps the original averaging logic but should reduce over‑specific predictions and move the KL‑divergence lower toward the target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np



## === cell 1
DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"



## === cell 2
train_path = os.path.join(DATA_PATH, "train.csv")
test_path = os.path.join(DATA_PATH, "test.csv")
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

row_sums = train_df[vote_cols].sum(axis=1)
row_sums.replace(0, np.nan, inplace=True)
train_probs = train_df[vote_cols].div(row_sums, axis=0).fillna(1.0 / len(vote_cols))

global_mean = train_probs.mean(axis=0)

patient_means = train_probs.groupby(train_df["patient_id"]).mean()

patient_counts = train_df.groupby("patient_id").size()

test_probs = (
    test_df[["eeg_id", "patient_id"]]
    .merge(patient_means, left_on="patient_id", right_index=True, how="left")
    .merge(
        patient_counts.rename("patient_count"),
        left_on="patient_id",
        right_index=True,
        how="left",
    )
)

test_probs[vote_cols] = test_probs[vote_cols].fillna(global_mean)

test_probs["patient_count"] = test_probs["patient_count"].fillna(0)

k = patient_counts.median() if patient_counts.median() > 0 else 1.0
test_probs["weight_specific"] = test_probs["patient_count"] / (
    test_probs["patient_count"] + k
)
test_probs["weight_specific"] = test_probs["weight_specific"].clip(0, 1)  # safety

specific_part = test_probs[vote_cols].values
global_part = global_mean.values.reshape(1, -1)
w = test_probs["weight_specific"].values.reshape(-1, 1)
test_probs[vote_cols] = w * specific_part + (1 - w) * global_part

row_sum_test = test_probs[vote_cols].sum(axis=1).replace(0, np.finfo(float).eps)
test_probs[vote_cols] = test_probs[vote_cols].div(row_sum_test, axis=0)

submission = test_probs[["eeg_id"] + vote_cols]
submission_path = os.path.join(OUT_PATH, "submission.csv")
submission.to_csv(submission_path, index=False)



## === cell 3
print("First rows of submission.csv:")
print(pd.read_csv(submission_path).head())
