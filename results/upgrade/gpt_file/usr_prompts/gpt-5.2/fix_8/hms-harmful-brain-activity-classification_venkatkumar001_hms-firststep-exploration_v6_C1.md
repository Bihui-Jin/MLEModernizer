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
librosa==0.11.0
lightgbm==4.6.0
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
xgboost==2.0.3

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

1.138332

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa

np.random.seed(12)



## === cell 1
base_dir = pathlib.Path("/kaggle/input/hms-harmful-brain-activity-classification")
os.listdir(base_dir)



## === cell 2
path_train = base_dir / "train.csv"
train = pd.read_csv(path_train)
train.head()



## === cell 3
train.expert_consensus.unique()



## === cell 4
path_sample_sub = base_dir / "sample_submission.csv"
sample = pd.read_csv(path_sample_sub)
sample.head()



## === cell 5
path_test = base_dir / "test.csv"
test = pd.read_csv(path_test)
test.head()



## === cell 6
import seaborn as sns
import cv2

from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb

from sklearn.multioutput import MultiOutputRegressor

from sklearn.model_selection import GroupShuffleSplit



## === cell 7
train.columns



## === cell 8
RUN_EDA = False  # Change to True only for interactive exploration; keep False for scoring runs.

if RUN_EDA:
    plt.figure(figsize=(15, 10))
    sns.countplot(data=train, x="expert_consensus")



## === cell 9
binary_relation_map = {
    "Seizure": 0,
    "GPD": 1,
    "LRDA": 2,
    "Other": 3,
    "GRDA": 4,
    "LPD": 5,
}
train["expert_consensus"] = train["expert_consensus"].map(binary_relation_map)



## === cell 10
train.sample(3)



## === cell 11
train["kfold"] = -1
kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=12)
for fold, (_, valid_indices) in enumerate(kfold.split(train)):
    train.loc[valid_indices, "kfold"] = fold

print(train.kfold.value_counts())
train.to_csv("trainfold_5.csv", index=False)



## === cell 12
train.columns



## === cell 13
TARGET_COLS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

COMMON_FEATURES = [c for c in train.columns if c in test.columns]

agg_dict = {c: "first" for c in COMMON_FEATURES if c != "eeg_id"}
for c in TARGET_COLS:
    agg_dict[c] = "mean"

train_agg = train.groupby("eeg_id", as_index=False).agg(agg_dict)

X = train_agg[COMMON_FEATURES].copy()
y = train_agg[TARGET_COLS].copy()

y = y.clip(lower=0)
y_sum = y.sum(axis=1).replace(0, np.nan)
y = (y.div(y_sum, axis=0)).fillna(1.0 / len(TARGET_COLS))

print("Common features used:", COMMON_FEATURES)
print("Aggregated train rows:", len(train_agg), "original rows:", len(train))
print("X shape:", X.shape, "y shape:", y.shape)
print("Target row sum min/max:", float(y.sum(axis=1).min()), float(y.sum(axis=1).max()))



## === cell 14
for c in COMMON_FEATURES:
    X[c] = pd.to_numeric(X[c], errors="coerce")
X = X.fillna(-1)

gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=12)
train_idx, valid_idx = next(gss.split(X, y, groups=train_agg["patient_id"]))
xtrain, xvalid = X.iloc[train_idx].copy(), X.iloc[valid_idx].copy()
ytrain, yvalid = y.iloc[train_idx].copy(), y.iloc[valid_idx].copy()

print(xtrain.shape, xvalid.shape, ytrain.shape, yvalid.shape)



## === cell 15
xgb_params = {
    "learning_rate": 0.03628302216953097,
    "subsample": 0.7875490025178,
    "colsample_bytree": 0.11807135201147,
    "max_depth": 3,
    "booster": "gbtree",
    "reg_lambda": 0.0008746338866473539,
    "reg_alpha": 23.13181079976304,
    "random_state": 40,
    "n_estimators": 10000,
}

base_model = XGBRegressor(**xgb_params)
model = MultiOutputRegressor(base_model)
model.fit(xtrain, ytrain)

pred_valid = model.predict(xvalid)
pred_valid = np.asarray(pred_valid, dtype=np.float64)
print("pred_valid shape:", pred_valid.shape)
pred_valid[:2]



## === cell 16
pred_valid[0][0]




## === cell 17
def _softmax_rows(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    z = z - np.max(z, axis=1, keepdims=True)
    expz = np.exp(z)
    s = expz.sum(axis=1, keepdims=True)
    s = np.where(s == 0, 1.0, s)
    return expz / s


def _kl_divergence_rowwise(
    y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15
) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.clip(y_true, eps, 1.0)
    y_true = y_true / y_true.sum(axis=1, keepdims=True)
    y_pred = np.clip(y_pred, eps, 1.0)
    y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)
    return float(np.mean(np.sum(y_true * (np.log(y_true) - np.log(y_pred)), axis=1)))


pred_valid_prob = _softmax_rows(pred_valid)
print(
    "valid row sum min/max:",
    float(pred_valid_prob.sum(axis=1).min()),
    float(pred_valid_prob.sum(axis=1).max()),
)
print(
    "holdout KL (lower is better):",
    _kl_divergence_rowwise(yvalid.values, pred_valid_prob),
)



## === cell 18
model_full = MultiOutputRegressor(XGBRegressor(**xgb_params))
model_full.fit(X, y)



## === cell 19
test_unique = test.drop_duplicates(subset=["eeg_id"], keep="first").copy()
if test_unique["eeg_id"].duplicated().any():
    raise ValueError("test_unique still has duplicate eeg_id; cannot safely align.")

test_unique = test_unique.set_index("eeg_id")
missing = set(sample["eeg_id"]) - set(test_unique.index)
if missing:
    raise ValueError(
        f"Missing {len(missing)} eeg_id(s) from test after dedup; example: {list(missing)[:5]}"
    )

test_aligned = test_unique.loc[sample["eeg_id"]].reset_index()

X_test = test_aligned[COMMON_FEATURES].copy()
for c in COMMON_FEATURES:
    X_test[c] = pd.to_numeric(X_test[c], errors="coerce")
X_test = X_test.fillna(-1)

pred_test = model_full.predict(X_test)
pred_test = np.asarray(pred_test, dtype=np.float64)

expected_shape = (len(sample), len(TARGET_COLS))
if pred_test.ndim != 2 or pred_test.shape != expected_shape:
    raise ValueError(
        f"Unexpected pred_test shape {pred_test.shape}; expected {expected_shape}."
    )

pred_test = _softmax_rows(pred_test)

print("pred_test shape:", pred_test.shape, "expected:", expected_shape)
print(
    "row sum min/max:",
    float(pred_test.sum(axis=1).min()),
    float(pred_test.sum(axis=1).max()),
)



## === cell 20
sub = sample.copy()
sub[TARGET_COLS] = pred_test

check = sub[TARGET_COLS].sum(axis=1).values
print("row sum min/max:", float(check.min()), float(check.max()))
print("submission rows:", len(sub), "sample rows:", len(sample))

assert list(sub.columns) == ["eeg_id"] + TARGET_COLS
assert len(sub) == len(sample)

vals = sub[TARGET_COLS].to_numpy(dtype=np.float64)
vals = np.nan_to_num(
    vals,
    nan=1.0 / len(TARGET_COLS),
    posinf=1.0 / len(TARGET_COLS),
    neginf=1.0 / len(TARGET_COLS),
)
vals = np.clip(vals, 1e-15, None)
vals = vals / vals.sum(axis=1, keepdims=True)
sub[TARGET_COLS] = vals

sub.to_csv("submission.csv", index=False)
sub.head()



## === cell 21
if RUN_EDA:
    sample_rate = 200
    eeg_dir = base_dir / "train_eegs"
    eeg_ids_folder = set(x.stem for x in eeg_dir.glob("*.parquet"))
    eeg_ids_train = set(train["eeg_id"].unique())
    print("eeg_id_train == eeg_id_folder:", eeg_ids_train == eeg_ids_folder)
    print("all train eeg_id present:", eeg_ids_train.issubset(eeg_ids_folder))
    print(
        "too many in folder:",
        list(sorted(eeg_ids_folder.difference(eeg_ids_train)))[:10],
    )



## === cell 22
if RUN_EDA:
    rec = train.iloc[1]
    rec



## === cell 23
if RUN_EDA:
    path_eeg = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/1007580543.parquet"
    path_eeg



## === cell 24
if RUN_EDA:
    eeg = pd.read_parquet(path_eeg)
    eeg



## === cell 25
if RUN_EDA:
    sample_rate = 200
    i_start = int(rec.eeg_label_offset_seconds) * sample_rate
    i_stop = i_start + 50 * sample_rate
    fig, axs = plt.subplots(
        nrows=eeg.shape[1],
        figsize=(16, 10),
        tight_layout=True,
        sharex=True,
        gridspec_kw={"hspace": 0},
    )
    eeg.loc[i_start:i_stop].plot(subplots=True, ax=axs)
    for ax in np.ravel(axs):
        ax.legend(loc="upper right")
    plt.show()



## === cell 26
if RUN_EDA:
    spec_dir = base_dir / "train_spectrograms"
    spec_dir



## === cell 27
if RUN_EDA:
    path_spec = "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/1000493950.parquet"
    spec = pd.read_parquet(path_spec)
    spec = spec.set_index("time")
    spec



## === cell 28
if RUN_EDA:
    spec.columns = spec.columns.str.split("_", expand=True)
    spec.columns.names = ["region", "freq"]
    spec = spec.T
    spec = spec.reset_index()
    spec["freq"] = spec["freq"].astype("float")
    spec = spec.set_index(["region", "freq"])
    spec



## === cell 29
if RUN_EDA:
    regions = list(spec.index.get_level_values(0).unique())
    regions



## === cell 30
if RUN_EDA:
    import librosa.display

    fig, axs = plt.subplots(
        nrows=2,
        ncols=2,
        sharex="all",
        sharey="all",
        tight_layout=True,
        figsize=(16, 10),
    )
    for region, ax in zip(regions, axs.flat):
        df = spec.loc[region]
        librosa.display.specshow(df.values, cmap="rainbow", ax=ax)
        ax.set_title(region)
    plt.show()
