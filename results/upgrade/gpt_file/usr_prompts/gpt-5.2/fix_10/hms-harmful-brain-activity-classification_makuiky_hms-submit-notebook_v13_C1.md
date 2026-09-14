# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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
pillow==11.3.0
protobuf==6.33.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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

0.9595159442047616

# 6. Current score

1.40921

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.40836) has done: 'I fix the TensorFlow/TF-DF import crash caused by an incompatible protobuf runtime by forcing the pure-Python protobuf implementation and pinning the protobuf major version inside the notebook process before importing TensorFlow. Then I fix the “same length” submission error by ensuring predictions are generated for exactly the `sample_submission` `eeg_id` order (some test metadata can have duplicates/mismatches depending on how it’s read/merged). Finally, I keep your core modeling logic intact (TF-DF per-target regressors with a prior fallback) while making the merge/indexing deterministic so the pipeline always writes a valid `submission.csv` with correct row count and row-wise probability sums of 1.'
- What this solution (achieved 1.41639) has done: 'We fix the immediate TensorFlow import crash by removing the protobuf environment overrides that are incompatible with the Kaggle runtime (this is what triggers the `MessageFactory.GetPrototype` error). Then we keep your TF-DF per-target regressor approach intact, but correct the training target to match the competition metric by predicting class probabilities derived from **Dirichlet-smoothed vote proportions** (still a regression target, just better-calibrated). Finally, we ensure the test rows align exactly to `sample_submission` order and the output probabilities are strictly positive and row-normalized so the submission is always valid and should move the KL score down toward the target.'
- What this solution (achieved 1.41639) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation before importing TensorFlow, which avoids the binary protobuf/TensorFlow mismatch in this environment. Then I keep your TF-DF per-target regression approach intact but ensure TF-DF can import reliably; if it still fails, the code fall back to the prior as before so it always produces a valid submission. Finally, I make the probability post-processing strictly KL-safe (positive, normalized, aligned to `sample_submission` order) without changing the model logic, so the pipeline runs end-to-end and writes `submission.csv`.'
- What this solution (achieved 1.40921) has done: 'I fix the TensorFlow import crash by removing the protobuf environment overrides that trigger the `MessageFactory.GetPrototype` error in this Kaggle runtime, while keeping your TF-DF training/inference logic the same. Then I make the TF-DF dependency handling more robust (clean fallback to the prior if TF-DF can’t import/train) so the notebook always runs end-to-end. Finally, I keep your Dirichlet-smoothed vote-proportion targets but add a minimal, metric-aligned post-processing step: blend a small amount of the global prior into each prediction and re-normalize, which typically reduces KL by preventing overconfident probabilities.'
- What this solution (achieved 1.40921) has done: 'We fix the immediate runtime crash on `import tensorflow as tf` by setting the protobuf implementation to pure-Python *before* importing TensorFlow, which avoids the `MessageFactory.GetPrototype` incompatibility seen in this environment. Then we keep your TF-DF per-target regression training/inference logic identical, but make the TF-DF fallback more robust so the pipeline always completes and writes a valid `submission.csv`. Finally, we keep the same Dirichlet-smoothed targets and prior-blend calibration, only adding small safety checks (types/NaNs and strictly-positive normalization) to ensure the submission is always valid and KL-safe without changing the modeling approach.'

# 9. Code solution

## === cell 0
import os
import sys

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_DIR = "/kaggle/input/hms-harmful-brain-activity-classification"
TRAIN_CSV = f"{DATA_DIR}/train.csv"
TEST_CSV = f"{DATA_DIR}/test.csv"
SAMPLE_SUB = f"{DATA_DIR}/sample_submission.csv"

TARGET_COLS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
FEATURE_COLS = ["eeg_id", "spectrogram_id", "patient_id"]



## === cell 2
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
sample = pd.read_csv(SAMPLE_SUB)

train_agg = train.groupby("eeg_id", as_index=False).agg(
    spectrogram_id=("spectrogram_id", "first"),
    patient_id=("patient_id", "first"),
    **{c: (c, "sum") for c in TARGET_COLS},
)

y_counts = train_agg[TARGET_COLS].to_numpy(dtype=np.float32)
n_classes = len(TARGET_COLS)

alpha = 1.0
den = np.clip(y_counts.sum(axis=1, keepdims=True) + alpha * n_classes, 1.0, None)
y = (y_counts + alpha) / den

X_train = train_agg[FEATURE_COLS].copy()

test_keyed = test.groupby("eeg_id", as_index=False).agg(
    spectrogram_id=("spectrogram_id", "first"),
    patient_id=("patient_id", "first"),
)
X_test_by_id = test_keyed.set_index("eeg_id")
sample_ids = sample["eeg_id"].values
X_test = X_test_by_id.reindex(sample_ids).reset_index()

X_test["spectrogram_id"] = X_test["spectrogram_id"].fillna(0).astype(np.int64)
X_test["patient_id"] = X_test["patient_id"].fillna(0).astype(np.int64)
X_test["eeg_id"] = X_test["eeg_id"].astype(np.int64)

X_train["spectrogram_id"] = X_train["spectrogram_id"].fillna(0).astype(np.int64)
X_train["patient_id"] = X_train["patient_id"].fillna(0).astype(np.int64)
X_train["eeg_id"] = X_train["eeg_id"].astype(np.int64)

prior = y.mean(axis=0).astype(np.float32)
prior = np.clip(prior, 1e-6, 1.0)
prior = prior / prior.sum()

print("TensorFlow version:", tf.__version__)
print("Train rows (aggregated by eeg_id):", len(train_agg))
print("Test unique eeg_id rows:", len(test_keyed))
print("Sample submission rows:", len(sample))
print("Prior (mean label distribution):", dict(zip(TARGET_COLS, prior.tolist())))



## === cell 3
models = None
tfdf = None
tfdf_ok = False
tfdf_error = None

try:
    import tensorflow_decision_forests as tfdf  # noqa: F401

    def fit_regressor(df_in: pd.DataFrame, label: str):
        model = tfdf.keras.RandomForestModel(
            task=tfdf.keras.Task.REGRESSION,
            random_seed=42,
        )
        ds = tfdf.keras.pd_dataframe_to_tf_dataset(
            df_in,
            label=label,
            task=tfdf.keras.Task.REGRESSION,
        )
        model.fit(ds, verbose=0)
        return model

    train_df = X_train.copy()
    for i, c in enumerate(TARGET_COLS):
        train_df[c] = y[:, i].astype(np.float32)

    models = {c: fit_regressor(train_df, c) for c in TARGET_COLS}
    tfdf_ok = True
    print("TF-DF models trained successfully.")
except Exception as e:
    tfdf_error = repr(e)
    tfdf_ok = False
    print(
        "WARNING: TF-DF unavailable or failed to train; using prior fallback. Error:",
        tfdf_error,
    )



## === cell 4
if tfdf_ok:
    test_df = X_test.copy()
    for c in TARGET_COLS:
        test_df[c] = 0.0

    preds = []
    for c in TARGET_COLS:
        ds_test_c = tfdf.keras.pd_dataframe_to_tf_dataset(
            test_df,
            label=c,
            task=tfdf.keras.Task.REGRESSION,
        )
        p = models[c].predict(ds_test_c, verbose=0).reshape(-1).astype(np.float32)
        preds.append(p)

    pred = np.stack(preds, axis=1)
else:
    pred = np.tile(prior.reshape(1, -1), (len(sample), 1)).astype(np.float32)

pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
pred = np.maximum(pred, 0.0)
row_sums = pred.sum(axis=1, keepdims=True)
pred = np.where(row_sums > 0, pred / row_sums, 1.0 / n_classes).astype(np.float32)

blend = 0.10
pred = (1.0 - blend) * pred + blend * prior.reshape(1, -1)

eps = 1e-6
pred = np.clip(pred, eps, 1.0)
pred = pred / pred.sum(axis=1, keepdims=True)

print(
    "Pred shape:",
    pred.shape,
    "Row sum min/max:",
    float(pred.sum(1).min()),
    float(pred.sum(1).max()),
)



## === cell 5
sub = sample[["eeg_id"]].copy()
for i, c in enumerate(TARGET_COLS):
    sub[c] = pred[:, i]

vals = sub[TARGET_COLS].to_numpy(dtype=np.float32)
vals = np.clip(vals, 1e-6, 1.0)
vals = vals / vals.sum(axis=1, keepdims=True)
sub[TARGET_COLS] = vals

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print(
    "Row sum min/max:",
    float(sub[TARGET_COLS].sum(axis=1).min()),
    float(sub[TARGET_COLS].sum(axis=1).max()),
)
print("Submission rows == sample rows:", len(sub) == len(sample))
