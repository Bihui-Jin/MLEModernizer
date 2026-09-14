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
Detect breast cancer in mammograms.

## Metric
[Probabilistic F1 score](https://aclanthology.org/2020.eval4nlp-1.9.pdf) (pF1). This extension of the traditional F score accepts probabilities instead of binary classifications. 

With pX as the probabilistic version of X:

$$
pF_1 = 2 \frac{pPrecision \cdot pRecall}{pPrecision + pRecall}
$$

where:

$$
pPrecision = \frac{pTP}{pTP + pFP}
$$

$$
pRecall = \frac{pTP}{TP + FN}
$$

## Submission Format
For each `prediction_id`, you should predict the likelihood of cancer in the corresponding `cancer` column. The submission file should have the following format:

```
prediction_id,cancer
0-L,0
0-R,0.5
0-R,0.5
1-L,1
...
# Dataset

**[train/test]_images/[patient_id]/[image_id].dcm** The mammograms, in dicom format. You can expect roughly 8,000 patients in the hidden test set. There are usually but not always 4 images per patient. Note that many of the images use the jpeg 2000 format which may you may need special libraries to load.

**sample_submission.csv** A valid sample submission.

**[train/test].csv** Metadata for each patient and image. Only the first few rows of the test set are available for download.

- `site_id` - ID code for the source hospital.
- `patient_id` - ID code for the patient.
- `image_id` - ID code for the image.
- `laterality` - Whether the image is of the left or right breast.
- `view` - The orientation of the image. The default for a screening exam is to capture two views per breast.
- `age` - The patient's age in years.
- `implant` - Whether or not the patient had breast implants. Site 1 only provides breast implant information at the patient level, not at the breast level.
- `density` - A rating for how dense the breast tissue is, with A being the least dense and D being the most dense. Extremely dense tissue can make diagnosis more difficult. Only provided for train.
- `machine_id` - An ID code for the imaging device.
- `cancer` - Whether or not the breast was positive for malignant cancer. The target value. Only provided for train.
- `biopsy` - Whether or not a follow-up biopsy was performed on the breast. Only provided for train.
- `invasive` - If the breast is positive for cancer, whether or not the cancer proved to be invasive. Only provided for train.
- `BIRADS` - 0 if the breast required follow-up, 1 if the breast was rated as negative for cancer, and 2 if the breast was rated as normal. Only provided for train.
- `prediction_id` - The ID for the matching submission row. Multiple images will share the same prediction ID. Test only.
- `difficult_negative_case` - True if the case was unusually difficult. Only provided for train.

# 2. Python version

3.11

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (190 lines)
            sample_submission.csv (2385 lines)
            sample_submission.csv.zip (6.5 kB)
            test.csv (5475 lines)
            test.csv.zip (60.6 kB)
            test.zip (160 Bytes)
            test_images.zip (29.1 GB)
            train.csv (49233 lines)
            train.csv.zip (513.7 kB)
            train.zip (162 Bytes)
            train_images.zip (260.7 GB)
            rsna-breast-cancer-detection/
                description.md (190 lines)
                sample_submission.csv (2385 lines)
                ... and 9 other files
                rsna-breast-cancer-detection/
                test_images/
                    10116/
                        1470873094.dcm (8.5 MB)
                        472095321.dcm (5.6 MB)
                        ... and 2 other files
                    10130/
                        1013166704.dcm (9.7 MB)
                        1165309236.dcm (8.7 MB)
                        ... and 5 other files
                    ... and 1191 other folders
                train_images/
                    10006/
                        1459541791.dcm (4.4 MB)
                        1864590858.dcm (4.0 MB)
                        ... and 2 other files
                    10011/
                        1031443799.dcm (2.1 MB)
                        220375232.dcm (1.7 MB)
                        ... and 2 other files
                    ... and 10720 other folders
            test_images/
                10116/
                    1470873094.dcm (8.5 MB)
                    472095321.dcm (5.6 MB)
                    ... and 2 other files
                10130/
                    1013166704.dcm (9.7 MB)
                    1165309236.dcm (8.7 MB)
                    ... and 5 other files
                ... and 1191 other folders
            train_images/
                10006/
                    1459541791.dcm (4.4 MB)
                    1864590858.dcm (4.0 MB)
                    ... and 2 other files
                10011/
                    1031443799.dcm (2.1 MB)
                    220375232.dcm (1.7 MB)
                    ... and 2 other files
                ... and 10720 other folders
        input/
            description.md (190 lines)
            sample_submission.csv (2385 lines)
            sample_submission.csv.zip (6.5 kB)
            test.csv (5475 lines)
            test.csv.zip (60.6 kB)
            test.zip (160 Bytes)
            test_images.zip (29.1 GB)
            train.csv (49233 lines)
            train.csv.zip (513.7 kB)
            train.zip (162 Bytes)
            train_images.zip (260.7 GB)
            rsna-breast-cancer-detection/
                description.md (190 lines)
                sample_submission.csv (2385 lines)
                ... and 9 other files
                rsna-breast-cancer-detection/
                test_images/
                    10116/
                        1470873094.dcm (8.5 MB)
                        472095321.dcm (5.6 MB)
                        ... and 2 other files
                    10130/
                        1013166704.dcm (9.7 MB)
                        1165309236.dcm (8.7 MB)
                        ... and 5 other files
                    ... and 1191 other folders
                train_images/
                    10006/
                        1459541791.dcm (4.4 MB)
                        1864590858.dcm (4.0 MB)
                        ... and 2 other files
                    10011/
                        1031443799.dcm (2.1 MB)
                        220375232.dcm (1.7 MB)
                        ... and 2 other files
                    ... and 10720 other folders
            test_images/
                10116/
                    1470873094.dcm (8.5 MB)
                    472095321.dcm (5.6 MB)
                    ... and 2 other files
                10130/
                    1013166704.dcm (9.7 MB)
                    1165309236.dcm (8.7 MB)
                    ... and 5 other files
                ... and 1191 other folders
            train_images/
                10006/
                    1459541791.dcm (4.4 MB)
                    1864590858.dcm (4.0 MB)
                    ... and 2 other files
                10011/
                    1031443799.dcm (2.1 MB)
                    220375232.dcm (1.7 MB)
                    ... and 2 other files
                ... and 10720 other folders
        working/
            rsna-breast-cancer-detection/
                description.md (190 lines)
                sample_submission.csv (2385 lines)
                ... and 9 other files
                rsna-breast-cancer-detection/
                test_images/
                    10116/
                        1470873094.dcm (8.5 MB)
                        472095321.dcm (5.6 MB)
                        ... and 2 other files
                    10130/
                        1013166704.dcm (9.7 MB)
                        1165309236.dcm (8.7 MB)
                        ... and 5 other files
                    ... and 1191 other folders
                train_images/
                    10006/
                        1459541791.dcm (4.4 MB)
                        1864590858.dcm (4.0 MB)
                        ... and 2 other files
                    10011/
                        1031443799.dcm (2.1 MB)
                        220375232.dcm (1.7 MB)
                        ... and 2 other files
                    ... and 10720 other folders
```

-> data/rsna-breast-cancer-detection/sample_submission.csv has 2384 rows and 2 columns.
The columns are: prediction_id, cancer

-> data/rsna-breast-cancer-detection/test.csv has 5474 rows and 9 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, implant, machine_id, prediction_id

-> data/rsna-breast-cancer-detection/train.csv has 49232 rows and 14 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, cancer, biopsy, invasive, BIRADS, implant, density, machine_id, difficult_negative_case

-> data/sample_submission.csv has 2384 rows and 2 columns.
The columns are: prediction_id, cancer

-> data/test.csv has 5474 rows and 9 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, implant, machine_id, prediction_id

-> data/train.csv has 49232 rows and 14 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, cancer, biopsy, invasive, BIRADS, implant, density, machine_id, difficult_negative_case

-> (stopped after 10 files for performance)

# 5. Target score

0.03

# 6. Current score

0.00025

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.01679) has done: 'I fix the immediate runtime error in the EDA cell by removing the unsupported `numeric_only` argument, without changing the modeling logic. Then I fix the training crash by ensuring all remaining categorical/string columns (notably `density`) are consistently one-hot encoded so LogisticRegression receives only numeric features. Finally, I keep the same upsampling + LogisticRegression + GridSearchCV approach, but make sure the fitted model is used to generate probabilities and write a valid `submission.csv` with the required columns and row order.'
- What this solution (achieved 0.00048) has done: 'Your current pipeline is likely under-scoring because the model selection is optimized for *binary F1 on a 0.5 threshold*, while the competition evaluates *probabilistic F1* on *probabilities*, so we keep the same LogisticRegression+GridSearch core but switch the GridSearch scoring to a pF1 scorer based on `predict_proba`. We also fit the selected model on the full upsampled training set after CV (rather than only the split’s train fold) to better match test-time use without changing the approach. Finally, we ensure the one-hot encoding produces perfectly aligned train/test columns by concatenating before `get_dummies` (same encoding logic, fewer silent mismatches), which typically nudges probability quality upward toward your 0.03 target.'
- What this solution (achieved 0.0) has done: 'Your current score (0.00048) is far below the target (0.03), so we should improve pF1 while keeping the same overall approach (upsampling + LogisticRegression + GridSearchCV). The biggest issue is that you both upsample to 50/50 *and* use a strong `class_weight`, which typically over-inflates probabilities and hurts probabilistic-F1; removing `class_weight` while keeping upsampling is a minimal, directly relevant change. Next, pF1 is sensitive to probability calibration, so we keep the same LogisticRegression core but wrap it in `CalibratedClassifierCV` (still logistic regression probabilities, just calibrated) and optimize calibration method via GridSearch using the same pF1 scorer. Finally, we avoid any subtle scaling leakage by fitting the scaler only on `X_train` and applying it to `X_val`/test (same StandardScaler, same features), which usually improves generalization and thus pF1.'
- What this solution (achieved 0.0) has done: 'Your score is 0.0 (far below the 0.03 target), so the smallest likely fix is to correct the pF1 implementation: your current `p_fn` term is wrong, which can select a bad model in GridSearch and yield near-zero pF1 on the leaderboard. I replace `p_fn` with the correct probabilistic FN definition `sum((1 - p) * y)` (and compute pRecall as `pTP/(pTP+pFN)`), keeping everything else (upsampling, LogisticRegression, calibration, GridSearch) the same. I also make the scaler application consistent and non-leaky by fitting it once on `X_train` and then refitting on full `X` only right before the final `best_model.fit(X, y)` (same approach you intended, just ensuring the final model and final features match). This should move you back toward non-zero pF1 without changing the core modeling logic.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is most consistent with a submission alignment/format issue rather than the model suddenly becoming useless, so the minimal path toward the 0.03 target is to make the submission generation *exactly* match the competition’s required `prediction_id` list and avoid any silent row dropping/duplication. I keep your exact modeling approach (upsampling + LogisticRegression + CalibratedClassifierCV + GridSearchCV with the same pF1 scorer), but I fix scaling to be applied consistently without mutating pandas views, and I ensure `test_features` is built from the same `test_data_orig` used for `prediction_id` so there’s no chance of mismatch. Finally, I hard-validate that the written `submission.csv` has the exact same row count and `prediction_id` order as `sample_submission.csv`, which prevents “valid CSV but invalid IDs” issues that can yield a leaderboard 0.0.'
- What this solution (achieved 0.0) has done: 'I fix the root cause of the crash: after you concatenate train+test and one-hot encode, you’re later aggregating by `prediction_id`, but `train.csv` has no `prediction_id`, so the groupby produces an empty frame and `train_test_split` sees `n_samples=0`. The minimal fix is to create a stable per-breast ID for train (e.g., `patient_id-laterality`) that matches the submission concept of one prediction per breast, and use that as the group key for train while keeping `prediction_id` for test. I also make sure `density` is not mistakenly treated as a test categorical (it’s train-only) and that the final submission is written with the exact `sample_submission` row order and `.csv` suffix.'
- What this solution (achieved 0.0003) has done: 'Your 0.0 leaderboard score strongly suggests the submission probabilities are collapsing to (almost) all zeros or becoming poorly calibrated, which pF1 punishes heavily; the safest minimal lever is to stop double-correcting class imbalance. I keep your exact pipeline (per-breast aggregation + upsampling + scaling + LogisticRegression + CalibratedClassifierCV + GridSearchCV with pF1) but remove `isotonic` calibration from the grid (it can overfit badly here) and add a small, stable probability floor/ceiling when writing the submission to avoid “all-zero” pF1 collapse. These changes don’t alter the core approach, but they usually move pF1 upward from 0.0 toward a non-trivial score like your 0.03 target band. I also add a quick diagnostic print of the prediction distribution to confirm we’re not outputting degenerate probabilities.'
- What this solution (achieved 0.00025) has done: 'Your current score (0.0003) is far below the target (0.03), so we should make the smallest changes that plausibly improve probability quality for pF1 without changing the core pipeline (per-breast aggregation + upsampling + scaling + LogisticRegression + calibration + GridSearch). The biggest low-risk issue is that `site_id` is being treated as a numeric, averaged feature; that destroys its categorical meaning and can badly hurt generalization, so we one-hot encode `site_id` and `machine_id` (as categories) using the same combined train+test `get_dummies` approach you already use. We also remove `machine_id` from scaling (scaling an ID is usually harmful) while keeping scaling for `age` only. Finally, we keep the same pF1 scorer and submission alignment, but we slightly relax the probability floor (from `1e-4` to `1e-6`) to avoid artificially inflating a large mass of tiny probabilities, which pF1 can penalize.'

# 9. Code solution

## === cell 0
import pandas as pd

train_data = pd.read_csv("/kaggle/input/rsna-breast-cancer-detection/train.csv")
test_data = pd.read_csv("/kaggle/input/rsna-breast-cancer-detection/test.csv")
sample_submission = pd.read_csv(
    "/kaggle/input/rsna-breast-cancer-detection/sample_submission.csv"
)



## === cell 1
print(train_data.info())



## === cell 2
print(test_data.info())



## === cell 3
train_data = train_data.drop(["image_id"], axis=1)
test_data = test_data.drop(["patient_id", "image_id"], axis=1)




## === cell 4
def fill_missing_mixed(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    num_cols = df.select_dtypes(include=["number"]).columns
    obj_cols = df.columns.difference(num_cols)

    if len(num_cols) > 0:
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

    for c in obj_cols:
        if df[c].isna().any():
            mode = df[c].mode(dropna=True)
            fill_val = mode.iloc[0] if len(mode) > 0 else "Unknown"
            df[c] = df[c].fillna(fill_val)
    return df


train_data = fill_missing_mixed(train_data)
test_data = fill_missing_mixed(test_data)



## === cell 5
print(test_data.info())



## === cell 6
print(train_data.info())



## === cell 7
import matplotlib.pyplot as plt
import seaborn as sns

print(train_data.describe())

if "age" in train_data.columns:
    sns.histplot(train_data["age"], kde=False)
    plt.show()



## === cell 8
from sklearn.preprocessing import StandardScaler

categorical_cols = [
    c
    for c in ["laterality", "view", "implant", "site_id", "machine_id"]
    if c in train_data.columns or c in test_data.columns
]

train_len = len(train_data)
combined = pd.concat([train_data, test_data], axis=0, ignore_index=True)

combined = pd.get_dummies(
    combined, columns=[c for c in categorical_cols if c in combined.columns]
)

train_data = combined.iloc[:train_len].copy()
test_data = combined.iloc[train_len:].copy()



## === cell 9
import matplotlib.pyplot as plt
import seaborn as sns

numeric_train = train_data.select_dtypes(include=["number"])
corr = numeric_train.corr()

fig, ax = plt.subplots(figsize=(21, 21))
sns.heatmap(corr, annot=False, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("Correlation Matrix (Numeric Columns Only)")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, horizontalalignment="right")
plt.show()



## === cell 10
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    fbeta_score,
    make_scorer,
)
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import numpy as np


def make_train_prediction_id(df: pd.DataFrame) -> pd.Series:
    if "patient_id" not in df.columns:
        raise ValueError(
            "patient_id must be present to build a train prediction_id surrogate."
        )
    if "laterality_L" in df.columns and "laterality_R" in df.columns:
        lat = np.where(df["laterality_L"].astype(float) >= 0.5, "L", "R")
    elif "laterality" in df.columns:
        lat = df["laterality"].astype(str)
    else:
        lat = "U"
    return (
        df["patient_id"].astype(str) + "-" + pd.Series(lat, index=df.index).astype(str)
    )


def aggregate_by_id(df: pd.DataFrame, group_col: str, is_train: bool) -> pd.DataFrame:
    df = df.copy()

    for c in ["age"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if is_train and "cancer" in num_cols:
        num_cols.remove("cancer")

    agg_dict = {}
    for c in num_cols:
        if c in df.columns:
            agg_dict[c] = "mean"

    if is_train:
        agg_dict["cancer"] = "max"

    other_cols = [c for c in df.columns if c not in agg_dict and c != group_col]
    for c in other_cols:
        agg_dict[c] = "first"

    out = df.groupby(group_col, as_index=False).agg(agg_dict)
    cols = [group_col] + [c for c in out.columns if c != group_col]
    return out[cols]


train_data = train_data.copy()
train_data["prediction_id"] = make_train_prediction_id(train_data)

train_data = train_data.drop(columns=["patient_id"])

train_pid = aggregate_by_id(train_data, group_col="prediction_id", is_train=True)
test_pid = aggregate_by_id(test_data, group_col="prediction_id", is_train=False)

if len(train_pid) == 0:
    raise ValueError(
        "Aggregated training data is empty; check prediction_id construction/grouping."
    )
if train_pid["cancer"].nunique() < 2:
    raise ValueError(
        "Aggregated training target has <2 classes; cannot train classifier."
    )

df_majority = train_pid[train_pid["cancer"] == 0]
df_minority = train_pid[train_pid["cancer"] == 1]

df_minority_upsampled = resample(
    df_minority, replace=True, n_samples=len(df_majority), random_state=42
)

train_data_upsampled = pd.concat(
    [df_majority, df_minority_upsampled], ignore_index=True
)
print(train_data_upsampled["cancer"].value_counts())

cols_to_scale = ["age"]
scaler = StandardScaler()

X = train_data_upsampled.drop("cancer", axis=1)
y = train_data_upsampled["cancer"]

pid = X["prediction_id"].copy()
X = X.drop(columns=["prediction_id"])

test_features = test_pid.copy()
test_pid_col = test_features["prediction_id"].copy()
test_features = test_features.drop(columns=["prediction_id"])

all_cols = sorted(set(X.columns).union(set(test_features.columns)))
X = X.reindex(columns=all_cols, fill_value=0)
test_features = test_features.reindex(columns=all_cols, fill_value=0)

X = X.fillna(0)
test_features = test_features.fillna(0)

X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
test_features = test_features.apply(pd.to_numeric, errors="coerce").fillna(0.0)




## === cell 11
def probabilistic_f1(y_true, y_prob, eps: float = 1e-15):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    p_tp = np.sum(y_prob * y_true)
    p_fp = np.sum(y_prob * (1.0 - y_true))
    p_fn = np.sum((1.0 - y_prob) * y_true)

    p_precision = p_tp / (p_tp + p_fp + eps)
    p_recall = p_tp / (p_tp + p_fn + eps)
    return 2.0 * p_precision * p_recall / (p_precision + p_recall + eps)


pf1_scorer = make_scorer(probabilistic_f1, needs_proba=True, greater_is_better=True)

X_train, X_val, y_train, y_val, pid_train, pid_val = train_test_split(
    X, y, pid, test_size=0.2, random_state=42, stratify=y
)

print(
    "Check the distribution of the target variable", train_data["cancer"].value_counts()
)

base_lr = LogisticRegression(random_state=42, solver="lbfgs", max_iter=2000)

X_train = X_train.copy()
X_val = X_val.copy()
test_features = test_features.copy()

for c in cols_to_scale:
    if c in X_train.columns:
        X_train.loc[:, c] = pd.to_numeric(X_train[c], errors="coerce").fillna(0.0)
        X_val.loc[:, c] = pd.to_numeric(X_val[c], errors="coerce").fillna(0.0)
        test_features.loc[:, c] = pd.to_numeric(
            test_features[c], errors="coerce"
        ).fillna(0.0)

if all(c in X_train.columns for c in cols_to_scale):
    X_train.loc[:, cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_val.loc[:, cols_to_scale] = scaler.transform(X_val[cols_to_scale])
    test_features.loc[:, cols_to_scale] = scaler.transform(test_features[cols_to_scale])

calibrated = CalibratedClassifierCV(estimator=base_lr, method="sigmoid", cv=3)

hyperparameters = {
    "estimator__C": [0.1, 1, 10],
    "method": ["sigmoid"],
}

clf = GridSearchCV(
    calibrated,
    hyperparameters,
    scoring=pf1_scorer,
    cv=5,
    n_jobs=-1,
    error_score="raise",
)
clf.fit(X_train, y_train)

print("Best hyperparameters:", clf.best_params_)

y_pred = clf.predict(X_val)
print("Accuracy:", accuracy_score(y_val, y_pred))
print("Precision:", precision_score(y_val, y_pred, zero_division=0))
print("Recall:", recall_score(y_val, y_pred, zero_division=0))
print("F1-score:", fbeta_score(y_val, y_pred, beta=1, average="binary", pos_label=1))

y_val_proba = clf.predict_proba(X_val)[:, 1]
print("Validation pF1 (probabilistic):", probabilistic_f1(y_val, y_val_proba))
print(
    "VAL proba stats: min/mean/max:",
    float(np.min(y_val_proba)),
    float(np.mean(y_val_proba)),
    float(np.max(y_val_proba)),
)

best_model = clf.best_estimator_

X_full = X.copy()
test_full = test_features.copy()

for c in cols_to_scale:
    if c in X_full.columns:
        X_full.loc[:, c] = pd.to_numeric(X_full[c], errors="coerce").fillna(0.0)
    if c in test_full.columns:
        test_full.loc[:, c] = pd.to_numeric(test_full[c], errors="coerce").fillna(0.0)

if all(c in X_full.columns for c in cols_to_scale):
    X_full.loc[:, cols_to_scale] = scaler.fit_transform(X_full[cols_to_scale])
    test_full.loc[:, cols_to_scale] = scaler.transform(test_full[cols_to_scale])

best_model.fit(X_full, y)



## === cell 12
test_data_orig = pd.read_csv("/kaggle/input/rsna-breast-cancer-detection/test.csv")

y_pred_proba = best_model.predict_proba(test_full)[:, 1]

prob_floor = 1e-6
y_pred_proba = np.clip(y_pred_proba, prob_floor, 1.0 - prob_floor)

pred_by_pid = pd.DataFrame(
    {"prediction_id": test_pid_col.values, "cancer": y_pred_proba}
)

submission = sample_submission[["prediction_id"]].merge(
    pred_by_pid, on="prediction_id", how="left"
)
submission["cancer"] = submission["cancer"].fillna(prob_floor).clip(0.0, 1.0)

assert submission.shape[0] == sample_submission.shape[0]
assert (
    submission["prediction_id"].tolist() == sample_submission["prediction_id"].tolist()
)
assert submission.columns.tolist() == ["prediction_id", "cancer"]

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Saved submission.csv with shape:", submission.shape)
print(
    "TEST proba stats: min/mean/max:",
    float(submission["cancer"].min()),
    float(submission["cancer"].mean()),
    float(submission["cancer"].max()),
)



## === cell 13
check = pd.read_csv("/kaggle/working/submission.csv")
print(check.head())
print(check.shape)
print(check.columns.tolist())
print("cancer min/max:", check["cancer"].min(), check["cancer"].max())
