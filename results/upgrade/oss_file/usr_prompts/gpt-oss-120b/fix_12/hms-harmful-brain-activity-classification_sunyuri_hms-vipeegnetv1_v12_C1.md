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

0.47367837668064

# 6. Current score

0.99841

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.48867) has done: 'I replace the TensorFlow‑based inference with a lightweight fallback that avoids the protobuf‑related crash. The script compute the average class probabilities from the training set and use those as predictions for every test record, guaranteeing a valid CSV submission whose rows sum to 1. This fix removes the failing import, eliminates unnecessary model loading, and still produces a plausible score while keeping the core data‑handling logic.'
- What this solution (achieved 1.48867) has done: 'The script now correctly fills missing class probabilities with the per‑class mean (using a Series instead of a raw NumPy array) and keeps the TensorFlow import safely wrapped so it never crashes the run. This resolves the `fillna` error and guarantees every submission row sums to 1, producing a valid `submission.csv` without altering the core fallback logic.'
- What this solution (achieved 1.41937) has done: 'I add the missing standard imports (`os`, `pandas`, `numpy`) at the start of the script so the cells can reference them, and I guard the row‑normalisation against possible zero sums. No other logic is changed, preserving the original fallback‑prediction approach while ensuring a valid CSV submission is produced.'
- What this solution (achieved 1.41937) has done: 'I add a tiny Laplace smoothing step when computing the per‑eeg_id class probabilities (so no class has a zero probability) and also smooth the final submission rows before the final normalization. This keeps the original fallback logic but reduces extreme KL penalties, moving the score closer to the target lower‑is‑better metric.'
- What this solution (achieved 1.41937) has done: 'I add a lightweight patient‑level fallback: compute class probabilities per patient, then for each test record use the per‑eeg probabilities when available, otherwise fall back to the matching patient’s probabilities, and finally to the global mean. This small change keeps the original fallback logic but gives more informative priors for unseen eeg_id samples, which should lower the KL divergence toward the target score.'
- What this solution (achieved 1.41138) has done: 'I add a slightly stronger Laplace smoothing (increase SMOOTH_EPS) and introduce a modest blend with a uniform distribution (20 % weight) after the hierarchical fallback. This keeps the original per‑eeg → patient → global logic but smooths extreme probabilities, which reduces KL penalties and moves the score closer to the lower‑is‑better target.'
- What this solution (achieved 0.78352) has done: 'I lower the uniform blending weight and the Laplace smoothing to keep the hierarchical fallback predictions more faithful, and I correctly align the patient‑level probabilities with the test rows (the previous reindex used mismatched indexes, so most rows fell back to the global mean). These small adjustments keep the original fallback logic while providing more informative priors, which should reduce the KL‑divergence toward the target score.'
- What this solution (achieved 0.99841) has done: 'I lower the uniform blending weight to 0 so the predictions rely purely on the hierarchical fallback (eeg → patient → global) without adding a uniform component that unnecessarily increases KL‑divergence. This minor change keeps the core logic intact while moving the score closer to the lower‑is‑better target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

TF_AVAILABLE = False
print("TF available:", TF_AVAILABLE)

if os.getenv("PLATFORM") == "local" or os.path.exists(
    "./input/hms-harmful-brain-activity-classification/train.csv"
):
    train_path = "./input/hms-harmful-brain-activity-classification/train.csv"
else:
    train_path = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"

df_train = pd.read_csv(train_path)
TARGETS = df_train.columns[-6:]  # last six columns are the vote counts

SMOOTH_EPS = 1e-4
train_agg = df_train.groupby("eeg_id")[list(TARGETS)].sum()
train_agg_smooth = train_agg + SMOOTH_EPS
train_probs = train_agg_smooth.div(train_agg_smooth.sum(axis=1), axis=0)

patient_agg = df_train.groupby("patient_id")[list(TARGETS)].sum()
patient_agg_smooth = patient_agg + SMOOTH_EPS
patient_probs = patient_agg_smooth.div(patient_agg_smooth.sum(axis=1), axis=0)

total_votes = df_train[TARGETS].sum()
global_probs = (total_votes / total_votes.sum()).astype(np.float32).values
print("Global class probabilities (fallback):")
for t, p in zip(TARGETS, global_probs):
    print(f"{t}: {p:.5f}")



## === cell 1
if os.getenv("PLATFORM") == "local" or os.path.exists(
    "./input/hms-harmful-brain-activity-classification/test.csv"
):
    test_path = "./input/hms-harmful-brain-activity-classification/test.csv"
else:
    test_path = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"

df_test = pd.read_csv(test_path)
print("Test shape:", df_test.shape)

submission = pd.DataFrame()
submission["eeg_id"] = df_test["eeg_id"]

aligned_probs = train_probs.reindex(df_test["eeg_id"]).reset_index(drop=True)

aligned_patient = patient_probs.reindex(df_test["patient_id"]).reset_index(drop=True)

mean_series = pd.Series(global_probs, index=TARGETS)

aligned_probs = aligned_probs.fillna(aligned_patient).fillna(mean_series)

ROW_EPS = 1e-6
aligned_probs = aligned_probs + ROW_EPS

uniform_series = pd.Series(np.full(len(TARGETS), 1.0 / len(TARGETS)), index=TARGETS)
BLEND_WEIGHT = 0.0
aligned_probs = (1 - BLEND_WEIGHT) * aligned_probs + BLEND_WEIGHT * uniform_series

submission[TARGETS] = aligned_probs.values

row_sums = submission[TARGETS].sum(axis=1).replace(0, 1)
submission[TARGETS] = submission[TARGETS].div(row_sums, axis=0)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}, shape:", submission.shape)
print("First few rows:")
print(submission.head())

print("Row sum check (first 5 rows):")
print(submission[TARGETS].sum(axis=1).head())
