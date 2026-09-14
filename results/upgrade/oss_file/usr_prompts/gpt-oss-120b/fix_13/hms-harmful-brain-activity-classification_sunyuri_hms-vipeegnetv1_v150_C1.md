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

0.3288345313888168

# 6. Current score

0.79728

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'The fix removes the failing TensorFlow and CuPy imports, corrects the NumPy import, and replaces the complex model/generator pipeline with a simple uniform‑or‑average‑distribution predictor that directly creates valid predictions for the test set and writes them to `submission.csv`. This eliminates the runtime error while providing a reasonable baseline that moves the score toward the target.'
- What this solution (achieved 1.64506) has done: 'The fix updates the data paths to correctly locate the CSV files in the Kaggle environment, adds a small validation step to choose an optimal blending weight between patient‑specific and global averages (which nudges the KL‑divergence toward the target), and ensures the script builds and writes a proper `submission.csv` with probabilities that sum to one. This resolves the FileNotFoundError, defines `test_df` for the second cell, and keeps the core logic unchanged while improving calibration.'
- What this solution (achieved 0.92764) has done: 'The fix removes the stray “markdown” cell that caused a NameError, adds a small Laplace smoothing to the vote counts, computes patient‑specific distributions using the summed votes (instead of simple row‑averages), refines the blend‑weight search with a finer 0.01 step, and updates the prediction function accordingly. These changes keep the original blending logic but produce better‑calibrated probabilities, lowering the KL‑divergence toward the target while still writing a valid `submission.csv` with rows summing to one.'
- What this solution (achieved 1.26945) has done: 'I keep the overall averaging‑blend approach but remove the unnecessary Laplace + 1 smoothing (so the proportions reflect the true vote counts) and introduce a simple confidence scaling: patients with more total votes get a higher influence of their own distribution. This preserves the core logic while providing better‑calibrated predictions, which should lower the KL divergence toward the target. I also tighten the blend‑weight search to a finer grid for a slightly better fit.'
- What this solution (achieved 1.67825) has done: 'The update removes the confidence‑scaled reduction of the blend weight so that, when a patient’s historical votes are available, the model relies more heavily on the patient‑specific distribution (which better matches the true label distribution). This simple change is expected to lower the KL‑divergence and move the score closer to the target while keeping the original pipeline intact.'
- What this solution (achieved 1.26945) has done: 'The script’s main slowdown comes from the fine‑grid search loop, which repeatedly calls a Python‑level function for every validation sample (~19k × 1000 ≈ 19 M calls). By pre‑computing patient vectors and confidences and using fully‑vectorized NumPy operations inside the weight loop, we eliminate the per‑sample Python overhead while preserving the exact blending logic. The test‑set prediction is also vectorized, removing the `apply` loop. All other logic (patient averages, confidence scaling, KL calculation) remains unchanged.'
- What this solution (achieved 0.79728) has done: 'I adjust the confidence calculation to use a smoothing term based on the median vote count per patient, which reduces the influence of patients with very few votes and yields better‑calibrated blended predictions. This change keeps the original blending logic and validation search intact while aiming to lower the KL‑divergence toward the target score. The rest of the pipeline and submission writing remain unchanged.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def resolve_path(filename: str) -> str:
    possible_paths = [
        os.path.join("data", "hms-harmful-brain-activity-classification", filename),
        os.path.join(
            "/kaggle/input/hms-harmful-brain-activity-classification", filename
        ),
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{filename} not found in any expected location.")


TRAIN_PATH = resolve_path("train.csv")
TEST_PATH = resolve_path("test.csv")
SUBMISSION_PATH = "submission.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

TARGETS = train_df.columns[-6:]

vote_counts = train_df[TARGETS].astype(np.float32)
row_sums = vote_counts.sum(axis=1).replace(0, np.nan)
proportions = vote_counts.div(row_sums, axis=0).fillna(0)  # (n_train, n_targets)

global_avg = proportions.mean(axis=0).values
global_avg = global_avg / global_avg.sum()
global_avg_series = pd.Series(global_avg, index=TARGETS)

patient_vote_sum = train_df.groupby("patient_id")[TARGETS].sum()
patient_avg = patient_vote_sum.div(patient_vote_sum.sum(axis=1), axis=0)
patient_avg = patient_avg.apply(lambda row: row.fillna(global_avg_series), axis=1)

patient_total_votes = patient_vote_sum.sum(axis=1)  # total votes per patient

median_votes = patient_total_votes.median()
if pd.isna(median_votes) or median_votes == 0:
    median_votes = 1.0
patient_confidence = patient_total_votes / (
    patient_total_votes + median_votes
)  # values in (0,1]

train_idx, val_idx = train_test_split(proportions.index, test_size=0.2, random_state=42)
val_true = proportions.loc[val_idx].values


def kl_divergence(p, q):
    eps = 1e-15
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    return np.mean(np.sum(p * np.log(p / q), axis=1))


val_patient_ids = train_df.loc[val_idx, "patient_id"].values
val_patient_vecs = patient_avg.reindex(val_patient_ids).fillna(global_avg_series).values
val_confidences = patient_confidence.reindex(val_patient_ids).fillna(0).values

best_w = 0.0
best_score = np.inf
for w in np.arange(0.0, 1.001, 0.001):  # fine grid
    w_scaled = np.clip(w * val_confidences, 0.0, 1.0)  # shape (n_val,)
    blended = (
        w_scaled[:, None] * val_patient_vecs + (1.0 - w_scaled)[:, None] * global_avg
    )
    blended /= blended.sum(axis=1, keepdims=True)
    score = kl_divergence(val_true, blended)
    if score < best_score:
        best_score = score
        best_w = w

BLEND_WEIGHT = best_w
print(
    f"Best base blend weight on validation: {BLEND_WEIGHT:.3f} with KL={best_score:.5f}"
)




## === cell 1
test_patient_ids = test_df["patient_id"].values
test_patient_vecs = (
    patient_avg.reindex(test_patient_ids).fillna(global_avg_series).values
)
test_confidences = patient_confidence.reindex(test_patient_ids).fillna(0).values

w_scaled_test = np.clip(BLEND_WEIGHT * test_confidences, 0.0, 1.0)
pred_matrix = (
    w_scaled_test[:, None] * test_patient_vecs
    + (1.0 - w_scaled_test)[:, None] * global_avg
)
pred_matrix /= pred_matrix.sum(axis=1, keepdims=True)  # ensure rows sum to 1

submission = pd.DataFrame({"eeg_id": test_df["eeg_id"].values})
submission[TARGETS] = pred_matrix

row_sums = submission[TARGETS].sum(axis=1)
assert np.allclose(row_sums, 1.0, atol=1e-6), "Row probabilities do not sum to 1"

submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission saved to {SUBMISSION_PATH}, shape: {submission.shape}")
