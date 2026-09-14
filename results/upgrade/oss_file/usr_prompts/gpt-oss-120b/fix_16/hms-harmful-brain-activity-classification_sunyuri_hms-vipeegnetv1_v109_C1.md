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

0.3522018747884437

# 6. Current score

0.77245

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'The script failed because it assumed a fixed relative path for `test.csv` when running locally, causing a FileNotFoundError. I added a small lookup that tries several common locations (the Kaggle input folder, a local `input/…` folder, and a generic fallback) and picks the first one that exists. This fixes the path issue, keeps the original logic intact, and guarantees a valid `.csv` submission is written. The rest of the code—including uniform probability generation and the row‑sum check—remains unchanged.'
- What this solution (achieved 1.41937) has done: 'I add a lightweight, data‑driven baseline: compute the overall class vote distribution from the training set and use those proportions as constant predictions for every test row. This keeps the original workflow (no model training) while providing more informed probabilities than the uniform baseline, which should lower the KL‑divergence score toward the target. The path lookup for `train.csv` mirrors the existing logic for `test.csv` and the submission format remains unchanged.'
- What this solution (achieved 1.68479) has done: 'The script failed because `DataFrame.fillna` cannot accept a NumPy array to fill each column with its own prior probability. I replace that call with a column‑wise fill that uses the corresponding element from `global_prior`. This resolves the ValueError and keeps the original patient‑wise prior logic intact, ensuring each row’s probabilities still sum to one and the submission file is written correctly.'
- What this solution (achieved 0.96626) has done: 'I blend the patient‑specific priors with the overall global prior using a modest weighting (40 % patient, 60 % global). This smooths overly extreme patient distributions, which should lower the KL‑divergence score and move it closer to the target while keeping the original workflow unchanged.'
- What this solution (achieved 1.13446) has done: 'The change reduces the influence of noisy patient‑specific priors by lowering `patient_weight` from 0.4 to 0.2, making the blended predictions rely more on the stable global class distribution. This simple adjustment is expected to decrease the KL‑divergence (lower is better) and move the score closer to the target while keeping all existing logic intact.'
- What this solution (achieved 1.41937) has done: 'I rename the cells to start at 1 as required and set the patient blending weight to 0.0, effectively using only the global class distribution for every test row. This eliminates noisy patient‑specific priors, keeping the overall probabilities normalized, and is expected to lower the KL‑divergence score toward the target.'
- What this solution (achieved 0.81597) has done: 'I increase the influence of patient‑specific priors by setting `patient_weight` to 0.7 (70 % patient, 30 % global). This uses more relevant information from the training data, which should lower the KL‑divergence score and move it closer to the target while keeping the overall workflow unchanged.'
- What this solution (achieved 0.76992) has done: 'Implemented fixes to resolve runtime errors and ensure a valid submission:

- Added robust handling for patient‑total broadcasting by converting the Series to a NumPy array before adding a new axis.
- Adjusted the patient prior computation to use NumPy arrays, avoiding deprecated pandas multi‑dimensional indexing.
- Renumbered cells to start at 1 as required, preserving the original workflow.
- Minor clean‑up to keep all variables defined and guarantee the submission CSV contains every target column with rows summing to 1.'
- What this solution (achieved 0.77245) has done: 'Implemented a modest patient‑ and EEG‑specific prior blend with stronger smoothing (ALPHA = 5) and added an EEG‑level prior fallback. The script now:
1. Loads data with robust path lookup.
2. Computes a global class prior.
3. Computes patient‑specific priors (smoothed toward global).
4. Computes EEG‑specific priors (also smoothed).
5. Builds the submission by first using EEG priors, then patient priors, and finally the global prior, ensuring each row sums to 1.'

# 9. Code solution

## === cell 0
import os, pandas as pd, numpy as np

PLATFORM = os.getenv("KAGGLE_KERNEL_RUN_TYPE", "local")  # "local" or "kaggle"
NEEDTRAIN = False  # This script only creates a submission; no model training

TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

print("Environment:", PLATFORM, "| Need training:", NEEDTRAIN)
print("Target columns:", TARGETS)



## === cell 1
possible_test_paths = [
    "/kaggle/input/hms-harmful-brain-activity-classification/test.csv",  # Kaggle default
    "./input/hms-harmful-brain-activity-classification/test.csv",  # Local repo layout
    "./working/hms-harmful-brain-activity-classification/test.csv",  # Alternate local layout
    "./test.csv",  # Fallback in current dir
]

test_path = None
for p in possible_test_paths:
    if os.path.exists(p):
        test_path = p
        break

if test_path is None:
    raise FileNotFoundError(
        "test.csv not found in any of the expected locations: "
        + ", ".join(possible_test_paths)
    )

print("Using test file at:", test_path)
test = pd.read_csv(test_path)
print("Test shape:", test.shape)
print(test.head())

possible_train_paths = [
    "/kaggle/input/hms-harmful-brain-activity-classification/train.csv",  # Kaggle default
    "./input/hms-harmful-brain-activity-classification/train.csv",  # Local repo layout
    "./working/hms-harmful-brain-activity-classification/train.csv",  # Alternate local layout
    "./train.csv",  # Fallback in current dir
]

train_path = None
for p in possible_train_paths:
    if os.path.exists(p):
        train_path = p
        break

if train_path is None:
    raise FileNotFoundError(
        "train.csv not found in any of the expected locations: "
        + ", ".join(possible_train_paths)
    )

print("Using train file at:", train_path)
train = pd.read_csv(train_path)



## === cell 2
vote_sums = train[TARGETS].sum(axis=0).astype(float)
total_votes = vote_sums.sum()
if total_votes == 0:
    global_prior = np.full(len(TARGETS), 1.0 / len(TARGETS), dtype=np.float32)
else:
    global_prior = (vote_sums / total_votes).values.astype(np.float32)
global_prior = global_prior / global_prior.sum()

print("Derived global prior probabilities from training data:")
for col, prob in zip(TARGETS, global_prior):
    print(f"{col}: {prob:.6f}")



## === cell 3
ALPHA = 5.0  # stronger smoothing (more weight to global prior)

patient_groups = train.groupby("patient_id")[TARGETS].sum()
patient_totals = patient_groups.sum(axis=1).astype(float)

smoothed_num_pat = patient_groups + ALPHA * global_prior
smoothed_den_pat = patient_totals.values[:, None] + ALPHA  # (n_patients, 1)

patient_priors = smoothed_num_pat / smoothed_den_pat
for idx, col in enumerate(TARGETS):
    patient_priors[col] = patient_priors[col].fillna(global_prior[idx])



## === cell 4
eeg_groups = train.groupby("eeg_id")[TARGETS].sum()
eeg_totals = eeg_groups.sum(axis=1).astype(float)

smoothed_num_eeg = eeg_groups + ALPHA * global_prior
smoothed_den_eeg = eeg_totals.values[:, None] + ALPHA

eeg_priors = smoothed_num_eeg / smoothed_den_eeg
for idx, col in enumerate(TARGETS):
    eeg_priors[col] = eeg_priors[col].fillna(global_prior[idx])



## === cell 5
sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})

for idx, col in enumerate(TARGETS):
    probs = test["eeg_id"].map(eeg_priors[col])
    probs = probs.fillna(test["patient_id"].map(patient_priors[col]))
    probs = probs.fillna(global_prior[idx])
    sub[col] = probs.astype(np.float32)

row_sums = sub[TARGETS].sum(axis=1)
assert np.allclose(row_sums, 1.0, atol=1e-6), "Row probability sums are not 1."



## === cell 6
submission_path = "submission.csv"
sub.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
print("Submission preview:")
print(sub.head())
