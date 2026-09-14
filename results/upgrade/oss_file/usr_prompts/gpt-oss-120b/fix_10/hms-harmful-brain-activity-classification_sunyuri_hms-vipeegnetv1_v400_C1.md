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

0.2857099205938075

# 6. Current score

0.775

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I replace the heavy TensorFlow‑based training pipeline with a lightweight fallback that skips model loading and simply predicts the overall class distribution from the training data. This avoids the protobuf import error, guarantees a valid `.csv` submission, and keeps the predictions reasonable while staying within the original script structure.'
- What this solution (achieved 1.41937) has done: 'I remove the stray `markdown` call that caused a NameError and replace it with a harmless comment. Then I enhance the prediction logic: instead of using a single global class distribution for every test row, I compute per‑`eeg_id` class probabilities from the training data (normalizing the vote counts). For test rows whose `eeg_id` appears in the training set, the model use these specific probabilities; otherwise it fall back to the global distribution. Finally, I re‑normalize each submission row to guarantee probabilities sum to 1, ensuring a valid submission and improving the KL‑divergence score.'
- What this solution (achieved 0.8425) has done: 'I keep the overall workflow but add a small smoothing term and a patient‑level fallback for rows whose eeg_id was not seen in training. This reduces zero‑probability predictions and provides a more informed prior than the global distribution, which should lower the KL‑divergence toward the target score while preserving the original simple heuristic.'
- What this solution (achieved 0.77767) has done: 'I add a small per‑spectrogram fallback, smooth the per‑eeg, per‑patient and per‑spectrogram probabilities with the global distribution (90 % specific + 10 % global) and reduce the added epsilon to 1e‑6. These minimal changes keep the original workflow while providing more informed priors, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 0.82219) has done: 'I tighten the smoothing to reduce over‑fitting on sparse per‑eeg / patient / spectrogram statistics by adding a Laplace (+1) count before normalising and by blending the specific probabilities with the global distribution using a 0.7 / 0.3 mix (instead of the previous 0.9 / 0.1). These minimal adjustments keep the overall workflow unchanged while providing a more robust prior that should lower the KL‑divergence toward the target score.'
- What this solution (achieved 1.01122) has done: 'I increase the Laplace smoothing from +1 to +5 and give more weight to the global class distribution (0.6 global / 0.4 specific). This reduces over‑fitting on sparse per‑eeg/patient/spec groups and should lower the KL‑divergence, moving the score closer to the target while keeping the original workflow intact.'
- What this solution (achieved 0.76617) has done: 'I reduce the Laplace smoothing and give far more weight to the specific per‑eeg / patient / spectrogram statistics (which were shown to improve the KL‑divergence in earlier attempts). Changing `add_k` from 5 to 1 and the blend to 0.9 specific / 0.1 global restores a more informative prior while still handling unseen IDs, moving the score closer to the target (lower KL).'
- What this solution (achieved 0.88634) has done: 'I slightly increase Laplace smoothing and give a bit more weight to the global class distribution while also using a larger epsilon floor. This reduces overly confident per‑group predictions and avoids near‑zero probabilities that explode KL‑divergence, moving the score closer to the target without changing the overall workflow.'
- What this solution (achieved 0.775) has done: 'I reduced the amount of unnecessary blending and smoothing so the model relies more on the actual vote counts for each eeg_id, patient or spectrogram. The Laplace constant is lowered to 0.5 and the global‑weight is removed; when a specific group is missing we fall back hierarchically (eeg → patient → spectrogram → global). A tiny epsilon (1e‑6) is added only after the final normalization to keep probabilities strictly positive. These minimal changes keep the overall workflow intact while producing sharper, better‑calibrated predictions that should lower the KL‑divergence toward the target score.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

ADD_K = 0.5  # small Laplace smoothing
EPSILON = 1e-6  # tiny floor after final normalization

if os.path.isdir("/kaggle/input/hms-harmful-brain-activity-classification"):
    DATA_ROOT = "/kaggle/input/hms-harmful-brain-activity-classification"
elif os.path.isdir("hms-harmful-brain-activity-classification"):
    DATA_ROOT = "hms-harmful-brain-activity-classification"
else:
    DATA_ROOT = "."

TRAIN_PATH = os.path.join(DATA_ROOT, "train.csv")
TEST_PATH = os.path.join(DATA_ROOT, "test.csv")
SUBMISSION_PATH = "submission.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

TARGETS = train_df.columns[-6:]  # seizure_vote … other_vote

class_counts = train_df[TARGETS].sum()
global_probs = class_counts / class_counts.sum()


def dirichlet_probs(group_df, add_k=ADD_K):
    """
    Compute per‑group probabilities using a Dirichlet posterior:
        (counts + add_k) / (row_sum + add_k * n_classes)
    This provides a smoothed estimate without additional blending.
    """
    smoothed = group_df + add_k
    denom = smoothed.sum(axis=1) + add_k * len(TARGETS) - add_k * len(TARGETS)
    denom = smoothed.sum(axis=1)
    probs = smoothed.div(denom, axis=0)
    return probs


per_eeg_counts = train_df.groupby("eeg_id")[list(TARGETS)].sum()
per_eeg_probs = dirichlet_probs(per_eeg_counts)

per_patient_counts = train_df.groupby("patient_id")[list(TARGETS)].sum()
per_patient_probs = dirichlet_probs(per_patient_counts)

per_spec_counts = train_df.groupby("spectrogram_id")[list(TARGETS)].sum()
per_spec_probs = dirichlet_probs(per_spec_counts)


submission = test_df[["eeg_id", "patient_id", "spectrogram_id"]].copy()

submission = submission.merge(
    per_eeg_probs, left_on="eeg_id", right_index=True, how="left", suffixes=("", "_eeg")
)

for col in TARGETS:
    missing = submission[col].isna()
    if missing.any():
        fill_vals = submission.loc[missing, "patient_id"].map(per_patient_probs[col])
        submission.loc[missing, col] = fill_vals

for col in TARGETS:
    missing = submission[col].isna()
    if missing.any():
        fill_vals = submission.loc[missing, "spectrogram_id"].map(per_spec_probs[col])
        submission.loc[missing, col] = fill_vals

for col in TARGETS:
    submission[col].fillna(global_probs[col], inplace=True)

submission[TARGETS] = submission[TARGETS] + EPSILON
row_sums = submission[TARGETS].sum(axis=1).replace(0, 1)
submission[TARGETS] = submission[TARGETS].div(row_sums, axis=0)

submission = submission.drop(columns=["patient_id", "spectrogram_id"])
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission written to {SUBMISSION_PATH}, shape: {submission.shape}")
