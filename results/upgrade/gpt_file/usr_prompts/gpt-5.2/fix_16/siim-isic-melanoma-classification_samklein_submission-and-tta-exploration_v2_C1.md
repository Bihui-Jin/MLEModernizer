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
Predict whether a lesion is malignant (0 denotes **benign**, and 1 indicates **malignant**).

## Metric
Area under the ROC curve.

## Submission Format
For each `image_name` in the test set, you must predict the probability (`target`) that the sample is **malignant**. The file should contain a header and have the following format:

```
image_name,target
ISIC_0052060,0.7
ISIC_0052349,0.9
ISIC_0058510,0.8
ISIC_0073313,0.5
ISIC_0073502,0.5
etc.
```

## Dataset 
The images are provided in DICOM format.

Images are also provided in JPEG and TFRecord format (in the `jpeg` and `tfrecords` directories, respectively). Images in TFRecord format have been resized to a uniform 1024x1024.

Metadata is also provided outside of the DICOM format, in CSV files. See the `Columns` section for a description.

### Files
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

### Columns
- `image_name` - unique identifier, points to filename of related DICOM image
- `patient_id` - unique patient identifier
- `sex` - the sex of the patient (when unknown, will be blank)
- `age_approx` - approximate patient age at time of imaging
- `anatom_site_general_challenge` - location of imaged site
- `diagnosis` - detailed diagnosis information (train only)
- `benign_malignant` - indicator of malignancy of imaged lesion
- `target` - binarized version of the target variable

# 2. Python version

3.8

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
scipy==1.15.3
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (176 lines)
            jpeg.zip (24.7 GB)
            sample_submission.csv (4143 lines)
            sample_submission.csv.zip (16.4 kB)
            test.csv (4143 lines)
            test.csv.zip (42.5 kB)
            test.zip (6.4 GB)
            tfrecords.zip (9.3 GB)
            train.csv (28985 lines)
            train.csv.zip (299.7 kB)
            train.zip (46.0 GB)
            jpeg/
                test/
                    ISIC_1440063.jpg (1.1 MB)
                    ISIC_0815802.jpg (853.1 kB)
                    ... and 4140 other files
                train/
                    ISIC_1845271.jpg (1.0 MB)
                    ISIC_1970027.jpg (138.4 kB)
                    ... and 28982 other files
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
            test/
                ISIC_0052212.dcm (1.5 MB)
                ISIC_0076545.dcm (4.0 MB)
                ... and 4140 other files
                test/
            tfrecords/
                test00-2071.tfrec (579.6 MB)
                test01-2071.tfrec (583.5 MB)
                ... and 14 other files
            train/
                ISIC_0015719.dcm (2.4 MB)
                ISIC_0068279.dcm (1.3 MB)
                ... and 28982 other files
                train/
        input/
            description.md (176 lines)
            jpeg.zip (24.7 GB)
            sample_submission.csv (4143 lines)
            sample_submission.csv.zip (16.4 kB)
            test.csv (4143 lines)
            test.csv.zip (42.5 kB)
            test.zip (6.4 GB)
            tfrecords.zip (9.3 GB)
            train.csv (28985 lines)
            train.csv.zip (299.7 kB)
            train.zip (46.0 GB)
            jpeg/
                test/
                    ISIC_1440063.jpg (1.1 MB)
                    ISIC_0815802.jpg (853.1 kB)
                    ... and 4140 other files
                train/
                    ISIC_1845271.jpg (1.0 MB)
                    ISIC_1970027.jpg (138.4 kB)
                    ... and 28982 other files
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
            test/
                ISIC_0052212.dcm (1.5 MB)
                ISIC_0076545.dcm (4.0 MB)
                ... and 4140 other files
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
            tfrecords/
                test00-2071.tfrec (579.6 MB)
                test01-2071.tfrec (583.5 MB)
                ... and 14 other files
            train/
                ISIC_0015719.dcm (2.4 MB)
                ISIC_0068279.dcm (1.3 MB)
                ... and 28982 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
        working/
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
```

-> data/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> data/siim-isic-melanoma-classification/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> data/siim-isic-melanoma-classification/test.csv has 4142 rows and 5 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge

-> data/siim-isic-melanoma-classification/train.csv has 28984 rows and 8 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge, diagnosis, benign_malignant, target

-> data/test.csv has 4142 rows and 5 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge

-> data/train.csv has 28984 rows and 8 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge, diagnosis, benign_malignant, target

-> input/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> (stopped after 10 files for performance)

# 5. Target score

0.933452743708202

# 6. Current score

0.6716

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.66776) has done: 'I fix the pipeline crash caused by pandas `pd.NA` values reaching scikit-learn’s `SimpleImputer` (which triggers “boolean value of NA is ambiguous”) by converting string columns to plain object dtype and using real `np.nan` for missingness. I also make the OneHotEncoder compatible with the scikit-learn version in this environment by falling back from `sparse_output` to `sparse` when needed. Then I ensure downstream cells don’t error by only using variables that are successfully created, and I always write a valid `submission.csv` with the required `image_name,target` columns aligned to `sample_submission.csv`. These changes are execution/correctness fixes and should produce a valid submission; no core modeling approach is changed.'
- What this solution (achieved 0.66784) has done: 'Your current score (0.66776 AUC) is far below the target (0.93345), so we need a real but still “same-core-logic” improvement: keep LogisticRegression on metadata, but fix a major leakage/quality issue by doing patient-grouped out-of-fold (OOF) training and then training a final model on full data for test predictions. This preserves the same model family, same preprocessing, and same predict_proba semantics, but improves generalization because this competition has strong patient-level correlations and random row-wise fitting typically overfits/underperforms. I’m also adding `C` tuning to a conservative value and enabling a slightly stronger solver setting while keeping everything else intact; these are minimal hyperparameter adjustments, not a new approach. Finally, the submission is written once (and consistently) from the final-model probabilities to avoid the later geometric-mean cell overwriting with nearly identical values.'
- What this solution (achieved 0.76793) has done: 'Your current score (0.66784 AUC) is far below the target (0.93345), so we should make a small but meaningful improvement without changing the core model (LogisticRegression on metadata). The biggest lever available inside the same approach is to add the strongest metadata signal you currently drop: `patient_id` (as a categorical feature) and use it consistently with the existing preprocessing pipeline—this often boosts AUC a lot in this competition even for simple models. I keep the same CV/training structure (grouped OOF for sanity check + final fit on full train) and keep the submission writing logic unchanged, only updating the feature lists and ColumnTransformer accordingly. This stays within scikit-learn, preserves predict_proba semantics, and should move the public AUC substantially closer to the target.'
- What this solution (achieved 0.66784) has done: 'Your current score (0.76793 AUC) is far below the target (0.93345), so the smallest realistic move toward the target without changing the core model family is to make the cross-validation and the final fit consistent with patient-level leakage/shift: we **remove `patient_id` from the feature set** (it tends to overfit/behave inconsistently between train and test) while **keeping patient-grouped CV** for sanity checking. This preserves the same preprocessing + LogisticRegression pipeline and predict_proba semantics, but usually yields a more robust model on this competition’s hidden test distribution, moving AUC upward. I also keep the existing submission alignment logic and ensure we only write `submission.csv` from the final model predictions (the extra geometric-mean file remains optional and does not overwrite the main submission). All other code stays effectively the same.'
- What this solution (achieved 0.67626) has done: 'Your current AUC (0.66784) is far below the target (0.93345), so we should increase performance, but with minimal changes that keep the same LogisticRegression-on-metadata core. The biggest low-risk gain here is to add metadata signal you are currently discarding: the `diagnosis` text and `benign_malignant` label string (train only) can be used safely as categorical inputs by training them only on train and letting `OneHotEncoder(handle_unknown="ignore")` handle the missing columns in test. This preserves the exact same preprocessing+LogisticRegression pipeline and predict_proba semantics, and should move the score materially upward toward the target. Submission writing and alignment remain unchanged; we still always produce a valid `submission.csv`.'
- What this solution (achieved 0.66784) has done: 'We keep your exact “LogisticRegression on metadata via a sklearn Pipeline” core, but remove two train-only categorical columns (`diagnosis`, `benign_malignant`) that are not available at test-time and currently add noisy/unreliable signal (they become all-missing in test, shifting the model’s intercept/calibration and hurting ranking). We also simplify the submission mapping to a strict merge on `image_name` (same semantics, but eliminates any accidental misalignment) and keep the rest of the training/CV logic unchanged. These are minimal, low-risk changes aimed at improving test AUC toward your target without changing the model family or training approach. The script still runs end-to-end and writes `submission.csv` (and keeps your `geom_submission.csv` as a side artifact).'
- What this solution (achieved 0.6716) has done: 'I fix the crash by removing the unsupported `groups` argument passed into `LogisticRegressionCV.fit()` (your scikit-learn version doesn’t accept it), while keeping the same core pipeline (metadata preprocessing + logistic regression with AUC scoring). To preserve the intended “group-aware” validation logic, I keep `StratifiedGroupKFold` for the outer OOF sanity-check loop, but use a standard inner CV integer for `LogisticRegressionCV` so it runs. I also ensure the final fit/predict path always runs and writes a valid `submission.csv` with `image_name,target`, and make the later ensemble/geom-mean cells conditional so they don’t crash if earlier cells failed. These are execution/correctness fixes; they should also improve score versus a broken pipeline because you finally get a valid, consistent submission.'
- What this solution (achieved 0.6716) has done: 'We fix the pipeline crash by removing the unsupported `groups` argument being passed into `LogisticRegressionCV.fit()` (your installed scikit-learn doesn’t accept it). To preserve the intended patient-level leakage control without changing the core model family, we keep `StratifiedGroupKFold` only for the outer OOF sanity-check split, and use an integer `cv=5` for the inner `LogisticRegressionCV`. Then we ensure the final model fits and always writes a valid `submission.csv` with `image_name,target` aligned to `sample_submission.csv`. Finally, we guard the later “geom-mean” cells so they only run after `sub` exists, preventing cascading `NameError`s.'
- What this solution (achieved 0.77361) has done: 'The timeout is almost certainly coming from the nested CV: `cross_val_predict` runs an outer 5-fold loop, and inside each fold `LogisticRegressionCV` runs an inner 5-fold CV across 9 Cs (effectively ~25 fits per outer fold), which explodes total training time. The fastest safe fix is to keep the final training logic identical (full-data `LogisticRegressionCV` fit + test predict), but replace the expensive OOF “sanity check” with an equivalent, much cheaper OOF using a single fixed C (taken from the final model’s chosen C) so the check still measures the same pipeline’s predictive behavior without nested CV. Additionally, we avoid repeated preprocessing work by fitting the final model once first and reuse its selected hyperparameter, and we ensure the OneHotEncoder stays sparse to avoid large dense matrices. All changes preserve the core modeling logic and submission predictions.'
- What this solution (achieved 0.6716) has done: 'Your current AUC (0.77361) is far below the target (0.93345), so we should increase performance with the smallest change that keeps the same “metadata + sklearn Pipeline + LogisticRegressionCV” core. The main issue is that `patient_id` is being used as a feature: it can hurt generalization because most test patients are unseen and the model can overfit to training-patient identity rather than lesion characteristics. I minimally remove `patient_id` from the feature list (but keep it for grouped OOF sanity-check splitting), keep the rest of preprocessing and model the same, and keep the submission writing/alignment identical. This typically improves leaderboard AUC for this competition while preserving evaluation semantics and producing the same required `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_PATH = "/kaggle/data"
if not os.path.exists(BASE_PATH):
    BASE_PATH = "/kaggle/input"

TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
TEST_CSV = os.path.join(BASE_PATH, "test.csv")
SAMPLE_SUB_CSV = os.path.join(BASE_PATH, "sample_submission.csv")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.exists(TEST_CSV), f"Missing: {TEST_CSV}"
assert os.path.exists(SAMPLE_SUB_CSV), f"Missing: {SAMPLE_SUB_CSV}"

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)

print(train_df.shape, test_df.shape, sample_sub.shape)
print("Train columns:", list(train_df.columns))
print("Test columns:", list(test_df.columns))




## === cell 1
def get_means(preds):
    gmean = scipy.stats.gmean(preds, axis=1)
    average = np.array(np.mean(preds, axis=1))
    median = np.median(preds, axis=1)
    return gmean, average, median




## === cell 2
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedGroupKFold,
    cross_val_predict,
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import roc_auc_score

FEATURES = [
    "sex",
    "age_approx",
    "anatom_site_general_challenge",
]
TARGET = "target"
GROUP_COL = "patient_id"


def clean_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for c in [
        "sex",
        "anatom_site_general_challenge",
        "patient_id",
        "diagnosis",
        "benign_malignant",
    ]:
        if c in df.columns:
            s = df[c]
            s = s.astype("string").str.strip()
            s = s.replace({"": pd.NA, "unknown": pd.NA, "Unknown": pd.NA})
            df[c] = s.astype(object)
            df[c] = df[c].where(pd.notna(df[c]), np.nan)

    if "age_approx" in df.columns:
        df["age_approx"] = pd.to_numeric(df["age_approx"], errors="coerce").astype(
            float
        )

    return df


train_df = clean_metadata(train_df)
test_df = clean_metadata(test_df)

for c in FEATURES:
    if c not in test_df.columns:
        test_df[c] = np.nan

X = train_df[FEATURES]
y = train_df[TARGET].astype(int).values

groups = (
    train_df[GROUP_COL].astype(str).values if GROUP_COL in train_df.columns else None
)
X_test = test_df[FEATURES]

numeric_features = ["age_approx"]
categorical_features = [c for c in FEATURES if c not in numeric_features]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

ohe = OneHotEncoder(handle_unknown="ignore")

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", ohe),
    ]
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="drop",
)

_C_GRID = np.logspace(-3, 1, 9)
cv_inner = 5

clf = LogisticRegressionCV(
    Cs=_C_GRID,
    cv=cv_inner,
    solver="lbfgs",
    max_iter=2000,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    scoring="roc_auc",
    n_jobs=-1,
    refit=True,
)

model = Pipeline(steps=[("preprocess", preprocess), ("clf", clf)])



## === cell 3
model.fit(X, y)

best_C = float(np.ravel(model.named_steps["clf"].C_)[0])
print(f"Fitted final model. Selected C = {best_C:.6g}")

if groups is not None:
    cv_outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    print("Using StratifiedGroupKFold for OOF sanity check (patient_id groups)")
else:
    cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    print("Using StratifiedKFold for OOF sanity check (no groups available)")

fast_clf = LogisticRegression(
    C=best_C,
    solver="lbfgs",
    max_iter=2000,
    class_weight="balanced",
    random_state=RANDOM_STATE,
)
fast_model = Pipeline(steps=[("preprocess", preprocess), ("clf", fast_clf)])

oof_proba = cross_val_predict(
    estimator=fast_model,
    X=X,
    y=y,
    groups=groups,
    cv=cv_outer,
    method="predict_proba",
    n_jobs=1,  # deterministic/safe
    verbose=0,
)
oof_pred = oof_proba[:, 1].astype(np.float64)

oof_pred = np.clip(oof_pred, 1e-6, 1 - 1e-6)
oof_auc = roc_auc_score(y, oof_pred)
print(f"OOF AUC (sanity check, fixed-C to avoid nested CV): {oof_auc:.5f}")



## === cell 4
test_pred = model.predict_proba(X_test)[:, 1].astype(np.float64)
test_pred = np.clip(test_pred, 1e-6, 1 - 1e-6)

pred_df = pd.DataFrame(
    {"image_name": test_df["image_name"].values, "target": test_pred}
)
sub = sample_sub[["image_name"]].merge(pred_df, on="image_name", how="left")

if sub["target"].isna().any():
    sub["target"] = sub["target"].fillna(float(np.nanmean(test_pred)))

sub["target"] = np.clip(sub["target"].to_numpy(dtype=np.float64), 1e-6, 1 - 1e-6)
sub.to_csv("submission.csv", index=False)
sub.head()



## === cell 5
print("Submission written to submission.csv")
print(sub.describe(include="all"))
plt.figure(figsize=(8, 3))
plt.hist(sub["target"].values, bins=50)
plt.title("Predicted target distribution")
plt.show()



## === cell 6
if (
    "sub" in globals()
    and isinstance(sub, pd.DataFrame)
    and {"image_name", "target"}.issubset(sub.columns)
):
    t = sub["target"].to_numpy(dtype=np.float64)
    all_preds = pd.DataFrame(
        {
            "image_name": sub["image_name"].values,
            "1": t,
            "2": np.clip(t * 0.99 + 0.005, 1e-6, 1 - 1e-6),
            "3": np.clip(t * 1.01 - 0.005, 1e-6, 1 - 1e-6),
            "4": t,
            "5": t,
        }
    )
else:
    all_preds = None
    print("Skipping all_preds creation because `sub` is not available.")



## === cell 7
if isinstance(all_preds, pd.DataFrame):
    preds = all_preds[["1", "2", "3", "4", "5"]].to_numpy()
    means = get_means(preds)
    means_arr = np.transpose(means)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.hist(means_arr[:, 0] - means_arr[:, 1], bins=100)
    plt.title("Geometric - Average")
    plt.subplot(1, 3, 2)
    plt.hist(means_arr[:, 0] - means_arr[:, 2], bins=100)
    plt.title("Geometric - Median")
    plt.subplot(1, 3, 3)
    plt.hist(means_arr[:, 1] - means_arr[:, 2], bins=100)
    plt.title("Average - Median")
    plt.show()
else:
    print("Skipping geom-mean diagnostics because all_preds is not available.")



## === cell 8
if isinstance(all_preds, pd.DataFrame):
    preds = all_preds[["1", "2", "3", "4", "5"]].to_numpy()
    n_repeat = 10
    stds = []
    mns = []

    for _ in range(n_repeat):
        means = get_means(preds)
        stds += [np.std(means, axis=1)]
        mns += [np.mean(means, axis=1)]
        preds = np.transpose(means)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    for i in range(3):
        plt.plot(np.stack(mns, axis=0)[:, i])
    plt.title("Mean of (gmean, avg, median) across repeats")
    plt.subplot(1, 2, 2)
    for i in range(3):
        plt.plot(np.stack(stds, axis=0)[:, i])
    plt.title("Std of (gmean, avg, median) across repeats")
    plt.show()
else:
    print("Skipping repeat diagnostics because all_preds is not available.")



## === cell 9
if isinstance(all_preds, pd.DataFrame):
    preds = all_preds[["1", "2", "3", "4", "5"]].to_numpy()
    gmean, average, median = get_means(preds)

    geom_sub = pd.DataFrame(
        {"image_name": all_preds["image_name"].values, "target": gmean}
    )
    geom_sub = geom_sub.sort_values("image_name")

    geom_sub = sample_sub[["image_name"]].merge(geom_sub, on="image_name", how="left")
    geom_sub["target"] = geom_sub["target"].astype(np.float64)
    geom_sub["target"] = geom_sub["target"].fillna(float(np.nanmean(gmean)))
    geom_sub["target"] = np.clip(geom_sub["target"].values, 1e-6, 1 - 1e-6)

    geom_sub.to_csv("geom_submission.csv", index=False)

    print("Wrote geom_submission.csv (submission.csv unchanged)")
    geom_sub.head()
else:
    geom_sub = None
    print("Skipping geom_submission.csv because all_preds is not available.")



## === cell 10
if isinstance(geom_sub, pd.DataFrame):
    plt.figure(figsize=(8, 3))
    plt.hist(geom_sub["target"].values, bins=50)
    plt.title("Geometric-mean submission target distribution")
    plt.show()
else:
    print("Skipping geom_sub histogram because geom_sub is not available.")
