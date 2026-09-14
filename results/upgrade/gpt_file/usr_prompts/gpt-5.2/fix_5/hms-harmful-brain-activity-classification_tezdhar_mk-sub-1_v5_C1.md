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

0.4124392191517156

# 6. Current score

0.97422

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'Your notebook didn’t yield a score because it relies on external Kaggle Datasets (`/kaggle/input/hms-mk-codes`, `/kaggle/input/requirements-mk`, `/kaggle/input/hms-mk-data`) that are not present in your provided environment, so it can’t run end-to-end to generate `submission.csv`. To make it run and produce a valid submission within the given constraints, I’m replacing the missing external pipeline with a minimal, deterministic baseline that reads only the provided competition files and outputs a properly-normalized probability submission. This generate a valid `submission.csv` (so you can finally obtain a Kaggle score) and should land in a reasonable KL range by using the train-set mean class distribution (a standard safe baseline for KL). The core “training approach” in your original code cannot be preserved because the required code/checkpoint assets are unavailable; this is the smallest viable change to unblock scoring.'
- What this solution (achieved 1.01346) has done: 'I fix the merge failure by removing the incorrect `validate="one_to_one"` constraint and instead safely mapping `patient_id` from `test.csv` to the submission `eeg_id` order (test has repeated `eeg_id` in some environments). I also add a guard to ensure the mapping is deterministic by dropping duplicates on `eeg_id` (keeping the first), and I keep the rest of your probability-prior logic unchanged to preserve evaluation semantics. Finally, I ensure `out_file` is always defined by writing the submission after successful construction, so the last cell can read/print it without crashing.'
- What this solution (achieved 0.97422) has done: 'Your current solution is a patient-conditional prior with a global fallback; to move the KL score down toward the target, the smallest safe improvement is to reduce overconfidence by smoothing every prediction toward the global class distribution (a standard calibration step for KL). This preserves your core logic (same priors, same mapping) but adds one mixing hyperparameter `alpha` to shrink patient means toward the global mean, which typically improves KL on this competition. I also compute the global mean as a **vote-weighted** mean across all training rows (instead of averaging per-row normalized distributions), which better matches the label-generation process and usually reduces KL. Finally, I keep strict normalization and submission formatting unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"
os.makedirs(OUT_PATH, exist_ok=True)

TARGET_COLS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

train_csv = os.path.join(DATA_PATH, "train.csv")
test_csv = os.path.join(DATA_PATH, "test.csv")
sample_sub_csv = os.path.join(DATA_PATH, "sample_submission.csv")

assert os.path.exists(train_csv), f"Missing: {train_csv}"
assert os.path.exists(test_csv), f"Missing: {test_csv}"
assert os.path.exists(sample_sub_csv), f"Missing: {sample_sub_csv}"



## === cell 2
train = pd.read_csv(train_csv)
test = pd.read_csv(test_csv)
sub = pd.read_csv(sample_sub_csv)

assert "eeg_id" in test.columns
assert "eeg_id" in sub.columns
assert (
    "patient_id" in train.columns and "patient_id" in test.columns
), "patient_id required for conditional prior"
for c in TARGET_COLS:
    assert c in train.columns, f"train missing target col {c}"
    assert c in sub.columns, f"sample_submission missing target col {c}"

sub = sub[["eeg_id"] + TARGET_COLS].copy()



## === cell 3
votes = train[TARGET_COLS].to_numpy(dtype=np.float64)
row_sums = votes.sum(axis=1, keepdims=True)
row_sums[row_sums == 0.0] = 1.0
probs = votes / row_sums

eps = 1e-6

global_votes = votes.sum(axis=0)  # (6,)
global_mean = global_votes / max(global_votes.sum(), 1.0)
global_mean = np.clip(global_mean, eps, 1.0)
global_mean = global_mean / global_mean.sum()

train_probs_df = pd.DataFrame(probs, columns=TARGET_COLS)
train_probs_df["patient_id"] = train["patient_id"].values
patient_mean_df = train_probs_df.groupby("patient_id", sort=False)[TARGET_COLS].mean()

patient_mean_vals = patient_mean_df.to_numpy(dtype=np.float64)
patient_mean_vals = np.clip(patient_mean_vals, eps, 1.0)
patient_mean_vals = patient_mean_vals / patient_mean_vals.sum(axis=1, keepdims=True)
patient_mean_df.loc[:, TARGET_COLS] = patient_mean_vals

global_mean



## === cell 4
test_patients = test[["eeg_id", "patient_id"]].copy()
test_patients = test_patients.drop_duplicates(subset=["eeg_id"], keep="first")

merge_df = pd.DataFrame({"eeg_id": sub["eeg_id"].values}).merge(
    test_patients, on="eeg_id", how="left"
)
assert (
    merge_df["patient_id"].isna().sum() == 0
), "All submission eeg_id must exist in test.csv"

patient_pred = patient_mean_df.reindex(merge_df["patient_id"].values)[TARGET_COLS]
mask_unseen = patient_pred.isna().any(axis=1)
if mask_unseen.any():
    patient_pred.loc[mask_unseen, :] = global_mean

pred = patient_pred.to_numpy(dtype=np.float64)

alpha = (
    0.35  # patient weight; (1-alpha) goes to global. Small, safe improvement for KL.
)
pred = alpha * pred + (1.0 - alpha) * global_mean.reshape(1, -1)

pred = np.clip(pred, eps, 1.0)
pred = pred / pred.sum(axis=1, keepdims=True)

sub.loc[:, TARGET_COLS] = pred

row_sum_check = sub[TARGET_COLS].sum(axis=1).to_numpy()
max_dev = np.max(np.abs(row_sum_check - 1.0))
assert max_dev < 1e-10, f"Row probabilities do not sum to 1 (max deviation {max_dev})"

out_file = os.path.join(OUT_PATH, "submission.csv")
sub.to_csv(out_file, index=False)

out_file, sub.shape



## === cell 5
print(pd.read_csv(out_file).head())
print("Saved:", out_file)
