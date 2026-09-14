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

0.3438441016922353

# 6. Current score

0.71539

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'I make the TensorFlow import tolerant to protobuf issues and, when the pretrained weight files are missing, skip model loading and instead generate predictions using the average class distribution from the training data. This guarantees a valid CSV submission and avoids the FileNotFoundError.'
- What this solution (achieved 1.68479) has done: 'The fix removes all undefined‑variable references and bypasses the complex TensorFlow pipeline. It loads the training metadata, computes per‑patient average vote distributions (falling back to the overall average when a patient is unseen), applies a safe normalization so each row sums to 1, and writes a correctly‑named `submission.csv`. This resolves the NameError issues and guarantees a valid Kaggle submission, while keeping the core idea of using label‑distribution baselines that previously gave a reasonable score.'
- What this solution (achieved 0.9002) has done: 'I add a simple shrink‑age blending step that combines each patient’s mean vote distribution with the overall class average. This regularizes predictions for patients with few samples, which tends to lower the KL‑divergence score and move it closer to the target. The change is confined to the prediction‑generation cell and keeps all original logic intact.'
- What this solution (achieved 0.81392) has done: 'The update adds a per‑patient shrinkage factor based on how many training samples each patient has, replacing the fixed ALPHA blend. Patients with few records rely more on the global average, which reduces over‑confidence and typically lowers the KL‑divergence score, moving it closer to the target. The core logic and output format remain unchanged.'
- What this solution (achieved 0.91532) has done: 'I increase the shrinkage strength so that predictions rely more on the overall class distribution, which usually lowers the KL‑divergence for this baseline. The only change is setting `SHRINKAGE_K` to a larger value (50.0). This keeps the core logic unchanged while moving the score closer to the target.'
- What this solution (achieved 0.78865) has done: 'I lower the shrinkage constant so that predictions rely more on each patient’s own average vote distribution rather than being pulled heavily toward the global average. This modest change keeps the overall blending logic intact while expectedly reducing the KL‑divergence and moving the score closer to the target.'
- What this solution (achieved 0.71539) has done: 'I normalize the vote counts to probabilities before computing global and per‑patient averages. Using true class‑probability estimates makes the blend more statistically sound, which should lower the KL‑divergence and move the score toward the target while keeping the overall logic unchanged.'
- What this solution (achieved 0.77996) has done: 'I increase the shrinkage constant so predictions rely more on the overall class distribution, which typically lowers the KL‑divergence for this baseline. The change is limited to the `SHRINKAGE_K` value in cell 1, keeping all other logic unchanged.'
- What this solution (achieved 0.76992) has done: 'I replace the per‑patient shrinkage blending with a Dirichlet‑smoothed probability estimate that uses the total vote counts per patient instead of just the number of records. This provides a more statistically sound smoothing toward the global class distribution, which is expected to lower the KL‑divergence and move the score closer to the target while keeping the overall workflow unchanged. The rest of the script (loading data, writing the CSV) remains the same.'
- What this solution (achieved 0.71539) has done: 'I compute per‑row class probabilities first, then aggregate them per patient as a simple mean (giving each record equal weight). I also recompute the global average in the same way and apply Dirichlet‑style smoothing with a moderate ALPHA value. This keeps the overall workflow identical while providing more statistically sound patient estimates, which should lower the KL‑divergence and move the score toward the target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np



## === cell 1
BASE_PATH = (
    "./data/hms-harmful-brain-activity-classification"
    if os.path.isdir("./data/hms-harmful-brain-activity-classification")
    else "/kaggle/input/hms-harmful-brain-activity-classification"
)

TRAIN_PATH = os.path.join(BASE_PATH, "train.csv")
TEST_PATH = os.path.join(BASE_PATH, "test.csv")
SUBMISSION_PATH = "submission.csv"

df_train = pd.read_csv(TRAIN_PATH)
df_test = pd.read_csv(TEST_PATH)

TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

assert all(
    col in df_train.columns for col in TARGETS
), "Missing target columns in train data"

row_totals = df_train[TARGETS].sum(axis=1).replace(0, np.nan)
row_probs = df_train[TARGETS].div(row_totals, axis=0).fillna(0)

global_avg = row_probs.mean().values.astype(np.float32)

patient_probs = row_probs.groupby(df_train["patient_id"]).mean()
patient_counts = (
    df_train.groupby("patient_id").size().values
)  # number of rows per patient

ALPHA = 5.0  # smoothing strength; larger values lean more toward global average

patient_smoothed = (
    (patient_probs.values * patient_counts[:, None] + ALPHA * global_avg)
    / (patient_counts[:, None] + ALPHA)
).astype(
    np.float32
)  # shape (num_patients, 6)

patient_id_to_idx = {pid: idx for idx, pid in enumerate(patient_probs.index)}

pred_list = []
for pid in df_test["patient_id"]:
    if pid in patient_id_to_idx:
        pred_vec = patient_smoothed[patient_id_to_idx[pid]]
    else:
        pred_vec = global_avg.copy()
    row_sum = pred_vec.sum()
    if row_sum == 0:
        pred_vec = np.full_like(pred_vec, 1.0 / len(TARGETS))
    else:
        pred_vec = pred_vec / row_sum
    pred_list.append(pred_vec)

pred = np.vstack(pred_list)  # shape (num_test_rows, 6)



## === cell 2
submission = pd.DataFrame({"eeg_id": df_test["eeg_id"].values})
submission[TARGETS] = pred

row_sums = submission[TARGETS].sum(axis=1)
if not np.allclose(row_sums, 1.0):
    submission[TARGETS] = submission[TARGETS].div(row_sums, axis=0)

submission.to_csv(SUBMISSION_PATH, index=False)

print(f"Submission written to {SUBMISSION_PATH}")
print("Submission shape:", submission.shape)
print(submission.head())
