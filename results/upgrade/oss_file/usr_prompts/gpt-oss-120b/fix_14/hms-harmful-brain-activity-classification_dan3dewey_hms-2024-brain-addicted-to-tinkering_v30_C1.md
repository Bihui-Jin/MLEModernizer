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

1.06367

# 6. Current score

1.25953

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.19881) has done: 'I add a deterministic seed, safeguard the “best” center calculations so they always exist, and explicitly renormalize the test‑set probabilities to guarantee each row sums to 1. These minimal tweaks keep the original modelling pipeline unchanged while fixing the missing/invalid submission issue and should move the KL score a bit closer to the target.'
- What this solution (achieved 1.46209) has done: 'Implemented robust categorical encoding without using the unsupported `axis` argument, correctly defined `y_train` after encoding, and fixed variable scope issues so the model trains and predicts properly. Added explicit handling to ensure the submission file contains the required columns and each probability row sums to 1.'
- What this solution (achieved 1.24354) has done: 'I keep the same RandomForest‑based pipeline but add a small post‑processing step that blends the model’s predicted probabilities with the overall class distribution observed in the training data. This often improves calibration (and thus KL) without changing the core model or its training loop. I also raise the number of trees slightly for a modest boost in performance. The script now writes a proper submission CSV with rows that sum to 1.'
- What this solution (achieved 1.26078) has done: 'I keep the same RandomForest‑based pipeline but add a lightweight probability calibration step (CalibratedClassifierCV with isotonic regression) to produce better‑aligned probability estimates, and I soften the blend with the global class distribution by reducing α from 0.7 to 0.5. These minimal adjustments preserve the original model logic while improving calibration, which should lower the KL‑divergence score toward the target.'
- What this solution (achieved 1.37688) has done: 'I slightly increase the model’s reliance on its calibrated predictions by raising the blend weight (α) from 0.5 to 0.8, since the RandomForest + isotonic calibration already provides a good estimate and the global‑mean blend is now less needed. I also modestly raise the number of trees to 800 to improve stability without altering the overall pipeline. These minimal adjustments keep the core logic unchanged while expectedly lowering the KL‑divergence toward the target score.'
- What this solution (achieved 1.25953) has done: 'I lower the blending weight `alpha` from 0.8 to 0.5 so the final predictions rely more on the calibrated RandomForest probabilities and less on the global class distribution. This modest change improves probability calibration, which should reduce the KL‑divergence toward the target score while keeping the overall pipeline unchanged.'
- What this solution (achieved 3.31235) has done: 'The adjustment simply increases the blending weight `alpha` to 1.0, removing the influence of the global class distribution and relying fully on the calibrated RandomForest probabilities, which should improve calibration and lower the KL‑divergence toward the target score. No other parts of the pipeline are changed.'
- What this solution (achieved 1.25687) has done: 'I lower the blending weight `alpha` so that the final predictions mix the calibrated RandomForest probabilities with the global class distribution observed in the training data. Using a value around 0.4 provides smoother predictions, which improves calibration and reduces the KL‑divergence, moving the score closer to the target while keeping the original model and pipeline unchanged.'
- What this solution (achieved 1.26771) has done: 'I lower the blending weight `alpha` from 0.4 to 0.3 so the final predictions rely more on the global class distribution, which has been shown to improve calibration for this model and should reduce the KL‑divergence toward the target. The change is limited to the post‑processing step and keeps the overall pipeline unchanged.'
- What this solution (achieved 1.25953) has done: 'I keep the overall RandomForest + isotonic calibration pipeline unchanged but adjust the post‑processing: add a tiny probability floor to avoid zeros, then blend the calibrated predictions with the global class distribution using a slightly higher weight (α = 0.5). This small tweak should improve calibration and lower the KL‑divergence, moving the score closer to the target while preserving the core model logic.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

np.random.seed(42)

possible_bases = [
    Path("/kaggle/input/hms-harmful-brain-activity-classification"),
    Path("data") / "hms-harmful-brain-activity-classification",
]
BASE = next((p for p in possible_bases if (p / "train.csv").exists()), None)
if BASE is None:
    raise FileNotFoundError("train.csv not found in expected locations.")

TRAIN_PATH = BASE / "train.csv"
TEST_PATH = BASE / "test.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

TARGET_COLS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

vote_sums = train_df[TARGET_COLS].sum(axis=1).replace(0, np.nan)
train_probs = (
    train_df[TARGET_COLS].div(vote_sums, axis=0).fillna(1.0 / len(TARGET_COLS))
)




## === cell 1
feat_cols = ["eeg_id", "spectrogram_id", "patient_id"]
combined = pd.concat([train_df[feat_cols], test_df[feat_cols]], ignore_index=True)

train_feats = pd.DataFrame(
    index=range(len(train_df)), columns=feat_cols, dtype=np.int32
)
test_feats = pd.DataFrame(index=range(len(test_df)), columns=feat_cols, dtype=np.int32)

for col in feat_cols:
    codes, _ = pd.factorize(combined[col], sort=True)
    train_feats[col] = codes[: len(train_df)]
    test_feats[col] = codes[len(train_df) :]

X_train = train_feats.values
X_test = test_feats.values
y_train = train_probs.values  # shape (n_train, 6)




## === cell 2
rf = RandomForestClassifier(
    n_estimators=800,  # increased from 600
    max_features=0.5,
    max_samples=0.7,
    oob_score=False,
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=42,
)

y_labels = np.argmax(y_train, axis=1)
rf.fit(X_train, y_labels)

calibrated_rf = CalibratedClassifierCV(rf, method="isotonic", cv="prefit")
calibrated_rf.fit(X_train, y_labels)




## === cell 3
prob_matrix = calibrated_rf.predict_proba(X_test)  # (n_test, 6)

epsilon = 1e-6
prob_matrix = np.clip(prob_matrix, epsilon, None)

row_sums = prob_matrix.sum(axis=1, keepdims=True)
prob_matrix = np.divide(
    prob_matrix, row_sums, out=np.zeros_like(prob_matrix), where=row_sums != 0
)

alpha = 0.5  # increased from 0.3
global_mean = train_probs.mean().values  # (6,)
prob_matrix = alpha * prob_matrix + (1 - alpha) * global_mean

row_sums = prob_matrix.sum(axis=1, keepdims=True)
prob_matrix = np.divide(
    prob_matrix, row_sums, out=np.zeros_like(prob_matrix), where=row_sums != 0
)

submission = pd.DataFrame(
    prob_matrix,
    columns=TARGET_COLS,
)
submission.insert(0, "eeg_id", test_df["eeg_id"].values)

submission_path = "submission.csv"
submission.to_csv(
    submission_path,
    index=False,
    float_format="%.6f",
    na_rep="",
)

print(f"Submission written to {submission_path}")
print(submission.head())
