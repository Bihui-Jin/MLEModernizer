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

0.3181784697594211

# 6. Current score

1.39957

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I bypass the failing EfficientNet import and the heavyweight model‑inference code, and instead generate a valid submission by using the overall class‑vote distribution from the training set as a constant prediction for every test record. This fixes the `MessageFactory` protobuf error, ensures the script runs end‑to‑end, and produces a properly formatted `submission.csv` whose rows sum to 1.'
- What this solution (achieved 1.40791) has done: 'The fix adds a safe fallback for the test feature columns `eeg_label_offset_seconds` and `spectrogram_label_offset_seconds`, which are absent in `test.csv`. If they are missing, a zero‑filled array is used so the model can still produce predictions without raising a KeyError. This change restores end‑to‑end execution and generates a valid `submission.csv` where each row’s probabilities sum to 1, moving the solution toward the target score.'
- What this solution (achieved 1.40096) has done: 'I add a small amount of feature scaling and ridge regularization to the linear‑regression model, and then blend its predictions with the overall class distribution. These tweaks keep the same linear‑model approach but should make the predictions less extreme and closer to the target distribution, reducing the KL‑divergence toward the desired score.'
- What this solution (achieved 1.41152) has done: 'I decrease the model’s influence and increase regularisation, which should make the predictions smoother and closer to the overall class distribution—this is expected to lower the KL‑divergence toward the target. Specifically, I raise the ridge strength (`lam`) and reduce the blending weight (`alpha`) that combines model output with the prior. No other logic is changed, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 1.41937) has done: 'I lower the model’s influence by increasing ridge regularisation and setting the blending weight `alpha` to 0 so the predictions rely almost entirely on the overall class distribution, which is closer to the target distribution and should reduce the KL‑divergence toward the desired score.'
- What this solution (achieved 1.40138) has done: 'I keep the overall workflow but add a simple interaction feature and loosen the regularisation so the linear model can capture more signal. Then I blend the model’s predictions with the overall class prior using a modest weight (α = 0.3) instead of relying solely on the prior. This adds only a few lines, preserves the ridge‑regression approach, and should move the KL‑divergence down toward the target score.'
- What this solution (achieved 1.39957) has done: 'I lower the ridge regularisation (lam = 0.5) so the linear model can fit the training data more closely and increase the blending weight (alpha = 0.7) to let the model’s predictions dominate over the overall class prior. These minimal adjustments keep the core workflow unchanged while steering predictions toward a distribution that should lower the KL‑divergence toward the target score.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

PLATFORM = "kaggle"  # or 'local' if running locally
SEED = 2024
np.random.seed(SEED)


def get_path(relative_path):
    if PLATFORM == "local":
        return os.path.join("./input", relative_path)
    else:  # kaggle
        return os.path.join("/kaggle/input", relative_path)


train_path = get_path("hms-harmful-brain-activity-classification/train.csv")
train_df = pd.read_csv(train_path)

test_path = get_path("hms-harmful-brain-activity-classification/test.csv")
test_df = pd.read_csv(test_path)

TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

vote_counts = train_df[TARGETS].astype(np.float64)
row_sums = vote_counts.sum(axis=1).replace(0, np.nan)  # avoid division by zero
y = (vote_counts.T / row_sums).T.fillna(0.0).values  # (n_samples, n_classes)

feature_cols = ["eeg_label_offset_seconds", "spectrogram_label_offset_seconds"]
X_raw_base = train_df[feature_cols].fillna(0.0).astype(np.float64).values
interaction = (X_raw_base[:, 0] * X_raw_base[:, 1]).reshape(-1, 1)
X_raw = np.hstack([X_raw_base, interaction])  # shape (n_samples, 3)

feature_means = X_raw.mean(axis=0, keepdims=True)
feature_stds = X_raw.std(axis=0, keepdims=True)
feature_stds[feature_stds == 0] = 1.0  # avoid division by zero
X_scaled = (X_raw - feature_means) / feature_stds

bias = np.ones((X_scaled.shape[0], 1), dtype=np.float64)
X = np.concatenate([bias, X_scaled], axis=1)  # (n_samples, n_features+1)

lam = 0.5
XtX = X.T @ X
ridge = lam * np.eye(XtX.shape[0])
coeffs = np.linalg.solve(XtX + ridge, X.T @ y)  # (n_features+1, n_classes)

if all(col in test_df.columns for col in feature_cols):
    X_test_base = test_df[feature_cols].fillna(0.0).astype(np.float64).values
else:
    X_test_base = np.zeros((len(test_df), len(feature_cols)), dtype=np.float64)

test_interaction = (X_test_base[:, 0] * X_test_base[:, 1]).reshape(-1, 1)
X_test_raw = np.hstack([X_test_base, test_interaction])

X_test_scaled = (X_test_raw - feature_means) / feature_stds
bias_test = np.ones((X_test_scaled.shape[0], 1), dtype=np.float64)
X_test = np.concatenate([bias_test, X_test_scaled], axis=1)

pred = X_test @ coeffs
pred = np.clip(pred, a_min=0.0, a_max=None)  # remove negatives

overall_votes = train_df[TARGETS].sum().astype(np.float64)
epsilon = 1e-12
overall_prob = (overall_votes + epsilon) / (
    overall_votes.sum() + epsilon * len(TARGETS)
)  # shape (n_classes,)

alpha = 0.7
pred = alpha * pred + (1 - alpha) * overall_prob.values.reshape(1, -1)

row_sums_pred = pred.sum(axis=1, keepdims=True)
row_sums_pred[row_sums_pred == 0] = 1.0  # safety
pred_normalised = pred / row_sums_pred

zero_rows = pred_normalised.sum(axis=1) == 0
if zero_rows.any():
    pred_normalised[zero_rows] = overall_prob.values.reshape(1, -1)

submission = pd.DataFrame({"eeg_id": test_df["eeg_id"].values})
submission[TARGETS] = pred_normalised
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)

print(f"Submission written to {submission_path}")
print("First rows of the submission:")
print(submission.head())
