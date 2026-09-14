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

geopandas==0.14.4
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
pyarrow==19.0.1
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0

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

1.0214799308403597

# 6. Current score

0.79642

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41081) has done: 'Implemented fixes to resolve missing functions and undefined variables, added a lightweight `assemble_features` stub that creates dummy numeric features, disabled optional logistic‑regression steps to avoid shape issues, and ensured all downstream code has the necessary inputs. The script now runs end‑to‑end and outputs a valid `submission.csv` with proper probability columns summing to one.'
- What this solution (achieved 1.69407) has done: 'The fix aligns the test‑time feature columns with those used during training, removing the mismatched dummy “feat*” columns that caused the RandomForest to reject the input. By constructing the required probability columns (using the global mean probabilities) and adding a dummy `clust_id`, the model can predict correctly and the script now writes a proper `submission.csv` whose rows match the test set size.'
- What this solution (achieved 1.39532) has done: 'I replace the per‑cluster probability assignment in the final test‑set block with the global mean class probabilities, which are already calibrated and sum to 1. Using a constant, well‑balanced distribution should lower the KL‑divergence (moving the score from 1.694 toward the target ≈1.02) while keeping the core pipeline unchanged.'
- What this solution (achieved 1.41937) has done: 'The script is rewritten to remove the undefined variables and complex clustering steps. It now simply reads the training metadata, computes the overall class vote proportions, normalises them to probabilities that sum to 1, and writes these same probabilities for every test row. This guarantees a valid submission.csv with correctly‑named columns and avoids all earlier runtime errors, while providing a reasonable baseline score that moves toward the target.'
- What this solution (achieved 1.68479) has done: 'I compute per‑patient class probability distributions from the training data and use them for test rows that share a patient with the training set, falling back to the overall global probabilities otherwise. This adds modest, data‑driven variation while keeping the original simple baseline and should lower the KL‑divergence (move the score toward the target).'
- What this solution (achieved 1.68479) has done: 'I add a per‑eeg ID probability lookup derived from the training votes and use it as the first fallback before the patient‑level probabilities and the global fallback. This gives the model exact class distributions for any test rows that share an `eeg_id` with the training set, which should lower the KL‑divergence without changing the overall pipeline.'
- What this solution (achieved 0.8425) has done: 'I add a tiny Laplace‑smoothing step to the predicted probabilities before the final row‑normalisation. By adding a small constant ε to every class probability we avoid extreme zero values that can hurt KL‑divergence, while keeping the original per‑eeg, per‑patient and global fallback logic unchanged.'
- What this solution (achieved 0.78112) has done: 'The update removes the per‑eeg fallback (which was giving overly specific probabilities) and increases the Laplace smoothing constant. Using only the per‑patient probabilities (or the global fallback) makes predictions less tailored to the test rows, and the larger ε pushes the distribution toward a more uniform shape. Both changes are expected to raise the KL‑divergence value, moving the score from the current 0.8425 upward toward the target ≈ 1.02 while keeping the overall pipeline unchanged.'
- What this solution (achieved 1.41041) has done: 'The changes replace the per‑patient fallback with a pure global‑probability baseline and increase the Laplace smoothing constant from 1e‑2 to 5e‑2. This makes the predictions more uniform, which raises the KL‑divergence score, moving it closer to the target ≈ 1.02 while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.79642) has done: 'I replace the uniform‑global baseline with a patient‑specific fallback (using the per‑patient class probabilities computed earlier) and lower the additive smoothing constant. This makes predictions more informative for rows that share a patient with the training set, which should reduce the KL‑divergence and move the score closer to the target while still guaranteeing that each row sums to 1.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

BASE_DIR = "/kaggle/input/hms-harmful-brain-activity-classification/"

TRAIN_PATH = os.path.join(BASE_DIR, "train.csv")
TEST_PATH = os.path.join(BASE_DIR, "test.csv")
SUBMISSION_PATH = "submission.csv"

HBA_VOTES = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]



## === cell 1
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

missing = set(HBA_VOTES) - set(train_df.columns)
if missing:
    raise RuntimeError(f"Training data is missing expected vote columns: {missing}")

class_vote_sums = train_df[HBA_VOTES].sum(axis=0).astype(float)
total_votes = class_vote_sums.sum()
if total_votes == 0:
    raise RuntimeError(
        "Total vote count in training data is zero; cannot compute probabilities."
    )
global_probs = class_vote_sums / total_votes
global_probs /= global_probs.sum()

print("Global class probabilities (fallback):")
print(global_probs)

patient_vote_sums = train_df.groupby("patient_id")[HBA_VOTES].sum()
patient_total = patient_vote_sums.sum(axis=1).replace(0, np.nan)
patient_probs = patient_vote_sums.div(patient_total, axis=0)

eeg_vote_sums = train_df.groupby("eeg_id")[HBA_VOTES].sum()
eeg_total = eeg_vote_sums.sum(axis=1).replace(0, np.nan)
eeg_probs = eeg_vote_sums.div(eeg_total, axis=0)



## === cell 2
patient_prob_df = test_df[["patient_id"]].join(
    patient_probs, on="patient_id", how="left"
)

for vote in HBA_VOTES:
    patient_prob_df[vote].fillna(global_probs[vote], inplace=True)

submission_df = pd.DataFrame()
submission_df["eeg_id"] = test_df["eeg_id"]
submission_df[HBA_VOTES] = patient_prob_df[HBA_VOTES]

epsilon = 5e-3  # reduced smoothing to improve KL score
submission_df[HBA_VOTES] = submission_df[HBA_VOTES] + epsilon

row_sums = submission_df[HBA_VOTES].sum(axis=1)
submission_df[HBA_VOTES] = submission_df[HBA_VOTES].div(row_sums, axis=0)



## === cell 3
submission_df.to_csv(
    SUBMISSION_PATH,
    index=False,
    float_format="%.6f",
    header=True,
    na_rep="",
)
print(f"Submission written to {SUBMISSION_PATH} with {len(submission_df)} rows.")
