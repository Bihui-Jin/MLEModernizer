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

0.3261303143352472

# 6. Current score

0.76095

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'Here’s a lightweight fix that removes the TensorFlow‑related parts which cause the protobuf error, computes simple class‑wise average probabilities from the training data, and writes a valid submission file with rows that sum to 1.'
- What this solution (achieved 1.39779) has done: 'I replace the naïve global‑average baseline with a per‑eeg‑id probability estimate: for each eeg_id present in the training data I compute the vote‑based class distribution and use it for matching test rows, falling back to the overall baseline only when an eeg_id is unseen. This preserves the original workflow while giving predictions that reflect the actual training distribution, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 1.67825) has done: 'I add a per‑patient probability fallback (computed from the same vote counts) so that test rows with an unseen eeg_id can still receive a more informed distribution instead of the global baseline. The script now merges per‑eeg probabilities, then fills remaining NaNs with per‑patient probabilities, and finally uses the overall baseline. This small tweak keeps the original logic intact while aiming to lower the KL‑divergence toward the target score.'
- What this solution (achieved 0.77501) has done: 'I add a small Laplace smoothing constant to every vote‑based count (global, per‑eeg, per‑patient) before normalising to probabilities. This removes zero‑probability predictions that heavily penalise the KL loss, while keeping the existing hierarchical fallback logic unchanged. The changes are confined to the probability‑construction part of the script.'
- What this solution (achieved 0.82219) has done: 'I slightly increase the Laplace smoothing constant and blend the hierarchical predictions with the global baseline instead of fully replacing missing values. This keeps the original fallback logic but makes the probabilities less extreme and more robust, which should lower the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 0.9101) has done: 'I lower the Laplace smoothing constant and make the hierarchical blending give more weight to the global baseline (reduce ALPHA_EEG and ALPHA_PATIENT). This keeps the original fallback logic but yields smoother, less extreme predictions, which is expected to lower the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 1.14521) has done: 'We increase the Laplace smoothing from 0.5 to 1.0 so per‑eeg and per‑patient distributions are less extreme, and we give the global baseline more influence by lowering the blending factors (`ALPHA_EEG` and `ALPHA_PATIENT`) from 0.5 to 0.2. These small adjustments keep the original workflow intact while producing smoother probability estimates, which should reduce the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 0.9101) has done: 'I reduce the Laplace smoothing back to 0.5 (making per‑eeg and per‑patient distributions less uniform) and increase the blending factors so the hierarchical predictions rely more on the learned per‑eeg / per‑patient probabilities rather than the global baseline. These minimal constant tweaks keep the original workflow intact while moving the KL‑divergence lower toward the target score.'
- What this solution (achieved 0.77501) has done: 'I make the hierarchical fallback logic use the per‑eeg and per‑patient probabilities directly (no blending with the global baseline) because the previous blended version makes the predictions too uniform and raises the KL‑divergence. Keeping the original Laplace smoothing ( 0.5 ) and removing the ALPHA_EEG / ALPHA_PATIENT blending yields more specific probability estimates and moves the score closer to the target while preserving the overall workflow.'
- What this solution (achieved 0.76095) has done: 'I add a small blending step that mixes each row’s hierarchical prediction with the global baseline (using a 5 % baseline weight). This smooths extreme probabilities without changing the overall fallback logic, which should lower the KL‑divergence and move the score closer to the target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

PLATFORM = "kaggle"
if PLATFORM == "local":
    DATA_ROOT = "./input/hms-harmful-brain-activity-classification"
else:
    DATA_ROOT = "/kaggle/input/hms-harmful-brain-activity-classification"

train_path = os.path.join(DATA_ROOT, "train.csv")
test_path = os.path.join(DATA_ROOT, "test.csv")
submission_path = "submission.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

TARGETS = train_df.columns[-6:].tolist()
print("Target columns:", TARGETS)



## === cell 1
SMOOTH = 0.5

total_votes = train_df[TARGETS].sum()
baseline_counts = total_votes + SMOOTH
baseline_probs = baseline_counts / baseline_counts.sum()
baseline_probs = baseline_probs.values  # shape (6,)
print("Baseline probabilities (global, smoothed):", baseline_probs)

eeg_group = train_df.groupby("eeg_id")[TARGETS].sum()
eeg_counts_smoothed = eeg_group + SMOOTH
eeg_sums = eeg_counts_smoothed.sum(axis=1)
per_eeg_probs = eeg_counts_smoothed.div(eeg_sums, axis=0).fillna(0)
assert np.allclose(per_eeg_probs.sum(axis=1), 1.0, atol=1e-6)

patient_group = train_df.groupby("patient_id")[TARGETS].sum()
patient_counts_smoothed = patient_group + SMOOTH
patient_sums = patient_counts_smoothed.sum(axis=1)
per_patient_probs = patient_counts_smoothed.div(patient_sums, axis=0).fillna(0)
assert np.allclose(per_patient_probs.sum(axis=1), 1.0, atol=1e-6)



## === cell 2
BETA = 0.05  # 5 % baseline contribution

submission = test_df[["eeg_id", "patient_id"]].copy()

submission = submission.merge(
    per_eeg_probs, left_on="eeg_id", right_index=True, how="left", suffixes=("", "_eeg")
)

has_eeg = submission[TARGETS].notna().all(axis=1)

missing_eeg_mask = ~has_eeg
if missing_eeg_mask.any():
    patient_probs = per_patient_probs.reindex(
        submission.loc[missing_eeg_mask, "patient_id"]
    )
    patient_probs.index = submission.loc[missing_eeg_mask].index
    for col in TARGETS:
        submission.loc[missing_eeg_mask, col] = patient_probs[col].values

if submission[TARGETS].isna().any().any():
    baseline_df = pd.DataFrame(
        [baseline_probs] * submission.shape[0],
        columns=TARGETS,
        index=submission.index,
    )
    submission[TARGETS] = submission[TARGETS].fillna(baseline_df)

submission[TARGETS] = (1 - BETA) * submission[TARGETS] + BETA * baseline_probs

row_sums = submission[TARGETS].sum(axis=1)
submission[TARGETS] = submission[TARGETS].div(row_sums, axis=0)

row_sums = submission[TARGETS].sum(axis=1)
assert np.allclose(row_sums, 1.0, atol=1e-6), "Row probabilities do not sum to 1"

submission[["eeg_id"] + TARGETS].to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
print("Submission preview:")
print(submission.head())
