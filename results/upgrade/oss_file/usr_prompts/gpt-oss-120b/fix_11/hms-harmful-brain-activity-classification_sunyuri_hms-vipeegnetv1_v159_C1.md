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

0.3381727920590047

# 6. Current score

1.41423

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.64506) has done: 'Implemented a safe fallback that skips TensorFlow imports altogether, preventing protobuf‑related crashes. The script now directly computes patient‑wise and global class probability averages and builds a valid submission CSV, ensuring probability rows sum to 1.'
- What this solution (achieved 0.81597) has done: 'I replace the unweighted patient average with a weighted‐average that accounts for the number of annotator votes per row, and then blend each patient’s distribution with the overall global distribution (70 % patient, 30 % global). This keeps the same overall logic but gives more influence to rows with many votes and shrinks extreme patient predictions toward the global baseline, which should lower the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 1.0413) has done: 'I lower the patient‑specific weight (ALPHA) and add a tiny smoothing epsilon before the final normalization. Reducing the influence of the per‑patient averages pushes predictions closer to the stable global distribution, which typically lowers the KL‑divergence and moves the score toward the desired target while preserving the original baseline logic.'
- What this solution (achieved 1.41937) has done: 'I decrease the patient‑specific blending weight to zero so the predictions rely solely on the stable global class distribution, which reduces variance and moves the KL‑divergence closer to the target 0.338. A tiny smoothing epsilon remains to keep probabilities strictly positive.'
- What this solution (achieved 0.77767) has done: 'We add a small validation split to evaluate a few blending weights (ALPHA values) and automatically pick the one that gives the lowest KL‑divergence on the validation set. This keeps the original baseline logic (global vs patient‑wise averages) but restores patient information, which historically reduced the score from 1.4 to around 0.8. By selecting the best ALPHA from a short list we move the metric closer to the target 0.338 while still using the same data‑processing pipeline.'
- What this solution (achieved 1.41423) has done: 'I replace the constant blending factor with a per‑patient weight that reflects how many annotator votes each patient contributed. The weight is computed as patient_votes / (patient_votes + total_votes_global), so patients with many votes keep more of their own distribution while sparse patients rely more on the global baseline. This change preserves the original averaging logic, adds only a few lines, and is expected to reduce over‑fitting, thereby lowering the KL‑divergence toward the target score.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

PLATFORM = "kaggle"  # local kaggle


def load_csv(path_local, path_kaggle):
    return pd.read_csv(path_local if PLATFORM == "local" else path_kaggle)


train_path_local = "./input/hms-harmful-brain-activity-classification/train.csv"
train_path_kaggle = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
test_path_local = "./input/hms-harmful-brain-activity-classification/test.csv"
test_path_kaggle = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"

df_train = load_csv(train_path_local, train_path_kaggle)
df_test = load_csv(test_path_local, test_path_kaggle)

TARGETS = df_train.columns[-6:]  # ['seizure_vote', 'lpd_vote', ...]
print("Train shape:", df_train.shape, "Test shape:", df_test.shape)
print("Target columns:", list(TARGETS))

vote_counts = df_train[TARGETS].astype(float)
row_totals = vote_counts.sum(axis=1).replace(0, np.nan)  # total votes per row
row_probs = vote_counts.div(row_totals, axis=0).fillna(0)  # normalized votes per row
weights = row_totals.fillna(0)  # weight = number of votes

weighted_sum = row_probs.multiply(weights, axis=0).groupby(df_train["patient_id"]).sum()
weight_totals = weights.groupby(df_train["patient_id"]).sum()
patient_means = weighted_sum.div(weight_totals, axis=0).fillna(
    0
)  # shape: (n_patients, 6)

global_weighted_sum = (row_probs.multiply(weights, axis=0)).sum()
global_total_weight = weights.sum()
global_mean = (global_weighted_sum / global_total_weight).values
global_mean = global_mean / global_mean.sum()  # enforce sum‑to‑1

alpha_per_patient = (weight_totals / (weight_totals + global_total_weight)).fillna(0)

SMOOTH_EPS = 1e-6  # tiny smoothing to avoid exact zeros

val_frac = 0.10
val_mask = np.random.rand(len(df_train)) < val_frac
df_val = df_train[val_mask].reset_index(drop=True)
df_train_split = df_train[~val_mask].reset_index(drop=True)

vote_counts_split = df_train_split[TARGETS].astype(float)
row_totals_split = vote_counts_split.sum(axis=1).replace(0, np.nan)
row_probs_split = vote_counts_split.div(row_totals_split, axis=0).fillna(0)
weights_split = row_totals_split.fillna(0)

weighted_sum_split = (
    row_probs_split.multiply(weights_split, axis=0)
    .groupby(df_train_split["patient_id"])
    .sum()
)
weight_totals_split = weights_split.groupby(df_train_split["patient_id"]).sum()
patient_means_split = weighted_sum_split.div(weight_totals_split, axis=0).fillna(0)

global_weighted_sum_split = (row_probs_split.multiply(weights_split, axis=0)).sum()
global_total_weight_split = weights_split.sum()
global_mean_split = (global_weighted_sum_split / global_total_weight_split).values
global_mean_split = global_mean_split / global_mean_split.sum()

alpha_per_patient_split = (
    weight_totals_split / (weight_totals_split + global_total_weight_split)
).fillna(0)


def kl_divergence(true_probs, pred_probs):
    mask = true_probs > 0
    return np.sum(true_probs[mask] * np.log(true_probs[mask] / pred_probs[mask]))


val_vote_counts = df_val[TARGETS].astype(float)
val_row_totals = val_vote_counts.sum(axis=1).replace(0, np.nan)
val_true_probs = val_vote_counts.div(val_row_totals, axis=0).fillna(0).values

val_preds = []
for pid in df_val["patient_id"].values:
    if pid in patient_means_split.index:
        patient_prob = patient_means_split.loc[pid].values
        alpha = alpha_per_patient_split.loc[pid]
    else:
        patient_prob = global_mean_split
        alpha = 0.0
    prob = alpha * patient_prob + (1 - alpha) * global_mean_split
    prob = prob + SMOOTH_EPS
    prob = prob / prob.sum()
    val_preds.append(prob)

val_pred_array = np.vstack(val_preds)
val_kl = kl_divergence(val_true_probs, val_pred_array)
print(f"Validation KL with per‑patient α: {val_kl:.5f}")

sub = pd.DataFrame({"eeg_id": df_test["eeg_id"].values})
test_patients = df_test["patient_id"].values

test_preds = []
for pid in test_patients:
    if pid in patient_means.index:
        patient_prob = patient_means.loc[pid].values
        alpha = alpha_per_patient.loc[pid]
    else:
        patient_prob = global_mean
        alpha = 0.0
    prob = alpha * patient_prob + (1 - alpha) * global_mean
    prob = prob + SMOOTH_EPS
    prob = prob / prob.sum()
    test_preds.append(prob)

test_pred_array = np.vstack(test_preds)

for i, col in enumerate(TARGETS):
    sub[col] = test_pred_array[:, i]

row_sums = sub[TARGETS].sum(axis=1)
assert np.allclose(row_sums, 1.0, atol=1e-6), "Probabilities do not sum to 1."

submission_path = "submission.csv"
sub.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
print("Submission shape:", sub.shape)
print("First 5 rows of probability sums:", sub[TARGETS].sum(axis=1).head().values)
