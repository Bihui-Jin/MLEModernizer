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

3.13

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

0.3185157602786119

# 6. Current score

0.91274

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I fix the file‑path errors by automatically locating the train.csv and test.csv files inside the dataset folder, then compute realistic class probabilities from the training vote counts (instead of a naïve uniform distribution) to produce a valid submission where each row sums to 1. This resolves the “file not found” crashes, ensures the submission format is correct, and should improve the KL score toward the target without altering the core modelling approach.'
- What this solution (achieved 1.68479) has done: 'I keep the overall workflow the same but replace the single global prior with patient‑specific priors derived from the training vote counts. For each test record we first try the distribution of the matching `patient_id`; if the patient never appears in training we fall back to the overall class prior. This adds useful per‑row information without changing the modelling approach, and it should lower the KL divergence toward the target score.'
- What this solution (achieved 1.68479) has done: 'The fix adds missing imports, defines file paths, loads the train and test metadata, computes global, patient‑ and EEG‑ID‑specific priors, aligns them to the test set, fills missing values with the appropriate fallback prior, normalises each row to sum to 1, and finally writes a valid `submission.csv`. This resolves the NameError issues and ensures a correctly‑formatted submission, which should move the KL‑divergence toward the target score.'
- What this solution (achieved 0.91274) has done: 'I add a tiny smoothing constant and blend the EEG‑specific and patient‑specific priors (80 % EEG + 20 % patient) before the final normalisation. This keeps the original hierarchy but gives each row a bit of extra information and prevents zero probabilities, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 0.91274) has done: 'I simplify the probability blending to rely on the most specific prior available: use the EEG‑specific distribution when it exists, fall back to the patient‑specific distribution only when the EEG prior is missing, and finally use the global prior for any remaining NaNs. This removes the 80/20 mixing that dilutes the strongest signal and should bring the KL‑divergence closer to the target lower score while preserving the overall workflow.'
- What this solution (achieved 1.41934) has done: 'I replace the fallback‑only logic with a weighted combination of EEG‑specific and patient‑specific priors, using the number of votes for each as the blending weight. This leverages more information from the training counts while still falling back to the global prior when no specific data exists, which should reduce the KL‑divergence toward the target lower score. The rest of the pipeline and file handling remain unchanged.'
- What this solution (achieved 0.91274) has done: 'I replace the weighted blending of EEG‑specific and patient‑specific priors with a hierarchy that uses the most specific available distribution: use the EEG‑specific prior when that EEG appears in the training data, otherwise fall back to the patient‑specific prior, and finally to the global prior. This keeps the overall workflow unchanged but should produce more informative probabilities and lower the KL‑divergence, moving the score toward the target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

possible_paths = [
    "/kaggle/input/hms-harmful-brain-activity-classification",
    "/kaggle/input/data/hms-harmful-brain-activity-classification",
    "./data/hms-harmful-brain-activity-classification",
    "./hms-harmful-brain-activity-classification",
]
BASE_PATH = next((p for p in possible_paths if os.path.isdir(p)), ".")

TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
TEST_CSV = os.path.join(BASE_PATH, "test.csv")
SUBMISSION_CSV = "submission.csv"  # written to current working directory

TARGET_COLUMNS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

print(f"Loaded train rows: {len(train_df)}, test rows: {len(test_df)}")



## === cell 1
vote_sums = train_df[TARGET_COLUMNS].sum()
total_votes = vote_sums.sum()
global_priors = (vote_sums / total_votes).values.astype(np.float32)

print("Global class prior probabilities:")
for col, prob in zip(TARGET_COLUMNS, global_priors):
    print(f"  {col}: {prob:.5f}")

patient_vote_sums = train_df.groupby("patient_id")[TARGET_COLUMNS].sum()
patient_total_votes = patient_vote_sums.sum(axis=1).replace(0, np.nan)
patient_priors_df = patient_vote_sums.div(patient_total_votes, axis=0)

for idx, col in enumerate(TARGET_COLUMNS):
    patient_priors_df[col].fillna(global_priors[idx], inplace=True)

eeg_vote_sums = train_df.groupby("eeg_id")[TARGET_COLUMNS].sum()
eeg_total_votes = eeg_vote_sums.sum(axis=1).replace(0, np.nan)
eeg_priors_df = eeg_vote_sums.div(eeg_total_votes, axis=0)

for idx, col in enumerate(TARGET_COLUMNS):
    eeg_priors_df[col].fillna(global_priors[idx], inplace=True)



## === cell 2
eeg_aligned = eeg_priors_df.reindex(test_df["eeg_id"]).reset_index(drop=True)
patient_aligned = patient_priors_df.reindex(test_df["patient_id"]).reset_index(
    drop=True
)

eeg_votes_aligned = eeg_total_votes.reindex(test_df["eeg_id"]).reset_index(drop=True)
patient_votes_aligned = patient_total_votes.reindex(test_df["patient_id"]).reset_index(
    drop=True
)

eeg_votes_aligned = eeg_votes_aligned.fillna(0.0)
patient_votes_aligned = patient_votes_aligned.fillna(0.0)

combined = eeg_aligned.copy()
no_eeg_idx = eeg_votes_aligned[eeg_votes_aligned == 0].index
if len(no_eeg_idx) > 0:
    combined.loc[no_eeg_idx] = patient_aligned.loc[no_eeg_idx]

for idx, col in enumerate(TARGET_COLUMNS):
    combined[col].fillna(global_priors[idx], inplace=True)

epsilon = 1e-4
combined[TARGET_COLUMNS] = combined[TARGET_COLUMNS] + epsilon
row_sums = combined[TARGET_COLUMNS].sum(axis=1)
prob_df = combined.copy()
prob_df[TARGET_COLUMNS] = prob_df[TARGET_COLUMNS].div(row_sums, axis=0)

submission = pd.DataFrame({"eeg_id": test_df["eeg_id"].values})
for idx, col in enumerate(TARGET_COLUMNS):
    submission[col] = prob_df[col].values.astype(np.float32)

assert np.allclose(
    submission[TARGET_COLUMNS].sum(axis=1), 1.0, atol=1e-5
), "Row probabilities do not sum to 1"

submission.to_csv(SUBMISSION_CSV, index=False)
print(f"Submission written to {SUBMISSION_CSV} (shape: {submission.shape})")



## === cell 3
print("Preview of submission:")
print(submission.head())
