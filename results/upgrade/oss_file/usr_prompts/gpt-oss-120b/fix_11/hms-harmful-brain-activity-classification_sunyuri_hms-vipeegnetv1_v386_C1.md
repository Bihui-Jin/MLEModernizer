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

0.2846556071065502

# 6. Current score

0.78743

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'We replace the TensorFlow‑heavy pipeline with a lightweight baseline that avoids the protobuf incompatibility causing the `'MessageFactory' object has no attribute 'GetPrototype'` error. The new script simply loads the training metadata, computes the overall class vote distribution, and writes those same probabilities for every test record, guaranteeing a valid `.csv` submission whose rows sum to 1.'
- What this solution (achieved 1.68479) has done: 'We keep the original global‑probability baseline but add a simple per‑patient weighting: compute each patient’s vote distribution from the training set and use it for test rows that share the same patient ID. If a patient is unseen, we fall back to the global distribution. This small, data‑driven tweak should lower the KL‑divergence (move the score from 1.419 toward the target 0.284) while preserving the overall pipeline and keeping all rows summing to 1.'
- What this solution (achieved 0.81597) has done: 'I blend each patient’s vote‑based distribution with the overall class distribution (using a modest weight) to avoid extreme zero probabilities, and then apply a tiny epsilon‑smoothing followed by renormalisation so every row sums exactly to 1. This small adjustment should lower the KL‑divergence toward the target without altering the overall pipeline.'
- What this solution (achieved 0.77742) has done: 'The changes increase the influence of patient‑specific vote distributions (raising `alpha` to 0.9) and add a tiny Laplace smoothing before normalising those distributions, which should make the predictions better aligned with the true per‑patient patterns and thus lower the KL‑divergence toward the target score. The rest of the pipeline, including the fallback to global class probabilities and the final renormalisation, stays unchanged.'
- What this solution (achieved 0.78644) has done: 'I slightly increase the patient‑specific blending weight from 0.9 to 0.95, keeping the same smoothing and fallback logic. This keeps the core pipeline unchanged while giving a bit more influence to per‑patient vote distributions, which should lower the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 0.90499) has done: 'I lower the patient‑specific blending weight from 0.95 to 0.5 (and make the tiny per‑patient count epsilon smaller) so the predictions rely more on the robust global class distribution. This modest change keeps the overall pipeline intact while moving the KL‑divergence score closer to the target (lower is better).'
- What this solution (achieved 0.77553) has done: 'Improved the baseline by increasing the smoothing added to each patient’s vote counts (‑‑ ε = 1e‑2) to avoid extreme zero probabilities, and restoring a stronger patient‑specific blending weight (α = 0.9). This keeps the original workflow while giving more reliable per‑patient distributions and a better balance with the global class distribution, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 0.78524) has done: 'I slightly reduce the patient‑specific blending weight and increase the smoothing applied to patient vote counts. A lower α (from 0.9 → 0.8) makes the predictions rely more on the robust global class distribution, while a larger εₚₐₜ (0.05) and a modest final epsilon (1e‑4) prevent extreme zero probabilities. These minimal tweaks keep the overall pipeline unchanged but should soften over‑confident per‑patient estimates, moving the KL‑divergence score closer to the target (lower is better).'
- What this solution (achieved 1.1548) has done: 'The patch reduces excessive smoothing and removes the convex blend with the global distribution, letting patient‑specific vote ratios dominate when available (which better matches the true per‑patient patterns). A tiny epsilon is kept only to avoid zero probabilities before renormalisation. This minimal change should lower the KL‑divergence toward the target while preserving the overall pipeline and output format.'
- What this solution (achieved 0.78743) has done: 'The update adds a modest Laplace smoothing for patient vote counts and blends each patient‑specific probability distribution with the overall class distribution (α = 0.8). This reduces over‑confident per‑patient predictions while keeping the original fallback to the global distribution, and it keeps rows normalized to sum = 1, moving the KL‑divergence score closer to the target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

if os.getcwd().split(os.sep)[1] == "home":
    PLATFORM = "local"
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
elif os.getcwd().split(os.sep)[1] == "kaggle":
    PLATFORM = "kaggle"
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"
else:
    PLATFORM = "local"
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"

TRAIN_CSV = os.path.join(LOAD_DATA_FROM, "train.csv")
TEST_CSV = os.path.join(LOAD_DATA_FROM, "test.csv")
SUBMISSION_PATH = "submission.csv"




## === cell 1
train_df = pd.read_csv(TRAIN_CSV)

TARGET_COLUMNS = train_df.columns[-6:]  # ['seizure_vote', 'lpd_vote', 'gpd_vote',

class_votes = train_df[TARGET_COLUMNS].sum().astype(np.float64)
class_probs = class_votes / class_votes.sum()
if not np.isfinite(class_probs).all() or abs(class_probs.sum() - 1.0) > 1e-6:
    class_probs = np.full(len(TARGET_COLUMNS), 1.0 / len(TARGET_COLUMNS))

epsilon_pat = 1e-2  # increased smoothing to avoid extreme zeroes
patient_counts = train_df.groupby("patient_id")[list(TARGET_COLUMNS)].sum()
patient_counts = patient_counts + epsilon_pat
patient_probs_df = patient_counts.div(patient_counts.sum(axis=1), axis=0)

alpha = 0.8  # give patient info most influence but retain global stability




## === cell 2
test_df = pd.read_csv(TEST_CSV)

submission = test_df[["eeg_id", "patient_id"]].copy()

submission = submission.merge(
    patient_probs_df.reset_index(), on="patient_id", how="left", suffixes=("", "_pat")
)

patient_present_mask = submission[TARGET_COLUMNS].notnull().all(axis=1)
if patient_present_mask.any():
    patient_vals = submission.loc[patient_present_mask, TARGET_COLUMNS].astype(float)
    blended_vals = alpha * patient_vals + (1 - alpha) * class_probs.values
    submission.loc[patient_present_mask, TARGET_COLUMNS] = blended_vals

missing_mask = submission[TARGET_COLUMNS].isnull().all(axis=1)
for col, prob in zip(TARGET_COLUMNS, class_probs):
    submission.loc[missing_mask, col] = prob

submission[TARGET_COLUMNS] = submission[TARGET_COLUMNS].fillna(
    pd.Series(class_probs, index=TARGET_COLUMNS)
)

epsilon = 1e-8
submission[TARGET_COLUMNS] = submission[TARGET_COLUMNS].replace(0, epsilon)

row_sums = submission[TARGET_COLUMNS].sum(axis=1)
submission[TARGET_COLUMNS] = submission[TARGET_COLUMNS].div(row_sums, axis=0)

submission = submission.drop(columns=["patient_id"])

row_sums = submission[TARGET_COLUMNS].sum(axis=1)
assert np.allclose(row_sums, 1.0, atol=1e-6), "Row probabilities do not sum to 1."

submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission written to {SUBMISSION_PATH}, shape: {submission.shape}")
