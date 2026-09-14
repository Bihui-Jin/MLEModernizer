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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
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

0.9366105656338396

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.6672) has done: 'Your notebook fails because it tries to read out-of-environment OOF/submission prediction files from `../input/rcsiimpreds/...`, which do not exist in the provided filesystem, so nothing downstream can be merged or written. To make it run end-to-end and still produce a meaningful ROC-AUC submission, I replace those missing external-prediction reads with a minimal, self-contained metadata-only model trained from `train.csv` and applied to `test.csv` (logistic regression with standard preprocessing). This preserves the “blend into a probability submission” evaluation semantics (probabilistic output) while ensuring correct alignment to `image_name` and writing a valid `submission.csv`. The rest of the original blend/merge cells are kept but guarded so they won’t crash when those external files aren’t present.'
- What this solution (achieved 0.70494) has done: 'Your current 0.6672 score comes from a very weak metadata-only model; to move toward the 0.9366 target without changing the overall “train a simple tabular model and output probabilities” semantics, the smallest effective upgrade is to (1) add a few strong metadata signals already present in `train.csv`/`test.csv` (patient_id + per-patient lesion count) and (2) use a slightly more expressive but still lightweight classifier (GradientBoostingClassifier) while keeping the same preprocessing pipeline. This remains fully self-contained (no external prediction files), avoids any label leakage (patient aggregates computed only from train for train and from test for test), and still outputs properly aligned `image_name,target` probabilities. I also keep the original blending/merge cells guarded exactly as before so they won’t crash, but the final submission consistently come from the improved metadata model. These changes are aimed at increasing AUC materially toward the target while staying minimal and within runtime limits.'
- What this solution (achieved 0.67757) has done: 'I keep your self-contained metadata-only approach, but make two minimal changes that are typically high-impact for this competition’s ROC-AUC: (1) replace the high-cardinality `patient_id` one-hot (which tends to overfit and generalize poorly here) with patient-level target-encoding computed out-of-fold on train and then mapped to test, and (2) add a simple log-age feature to stabilize the age signal. The model remains `GradientBoostingClassifier` with the same training flow and still outputs `image_name,target` probabilities aligned to `sample_submission.csv`. All the external blending cells remain guarded and unchanged, so the notebook still runs even without the missing `../input/rcsiimpreds/*` files. This should move your score upward toward the 0.9366 target with minimal risk and within runtime.'
- What this solution (achieved 0.69662) has done: 'I fix the crash in the metadata fallback model by ensuring groupby-count features never contain NaNs (they can when the grouping key is missing), and by using pandas’ nullable integer type before safely filling and converting to `int32`. Then I make `final` and `submit_file` always defined (even when the external `../input/rcsiimpreds/*` files are absent) so downstream blend/merge cells don’t raise `NameError`. Lastly, I ensure the script always writes a valid `submission.csv` with the exact required columns (`image_name,target`) aligned to `sample_submission.csv`, which yield a valid Kaggle submission and should improve score vs. a broken pipeline.'
- What this solution (achieved 0.71771) has done: 'Your current pipeline is already stable and produces a valid submission, but the score (0.6966) is far below the target (0.9366), so we need a modest, still “same core logic” improvement inside the metadata-only fallback. I keep the same overall approach (tabular preprocessing + GradientBoostingClassifier + target-encoding features) and only adjust two high-impact, low-risk knobs: (1) strengthen the model slightly by increasing the number of boosting stages while lowering learning rate, and (2) reduce target-encoding noise by increasing smoothing and adding a tiny Gaussian noise during training (regularization) to prevent over-confident patient/site leakage-like memorization. These changes should move AUC upward without changing the fundamental training loop or feature set, and everything still write `submission.csv` with `image_name,target` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.5) has done: 'Your current score (0.71771) is far below the target (0.93661), so the smallest safe way to move upward is to keep the same metadata-only pipeline and GradientBoostingClassifier, but fix two high-impact generalization issues: (1) remove the remaining target leakage from patient_id target-encoding by computing test encodings from out-of-fold (OOF) patient/site statistics instead of full-train statistics, and (2) add a second, very lightweight target-encoded feature for `diagnosis` (train-only) which historically carries strong signal in this competition without changing the core “tabular TE + GBDT” approach. I keep all the existing external-prediction blend cells guarded and unchanged, and ensure the final `submission.csv` is still produced with `image_name,target` aligned to `sample_submission.csv`. These are minimal changes inside the same training loop/semantics and should materially increase AUC toward your target.'

# 9. Code solution

## === cell 0
"""
submit of only B3 B4 & B5 models

NOTE (bugfix for this environment):
Original notebook depended on external prediction CSVs in ../input/rcsiimpreds/,
which are not available here, causing FileNotFoundError and preventing submission.
We provide a self-contained fallback that trains a metadata-only model and produces
submission.csv with the correct format.
"""

import os
import numpy as np
import pandas as pd

RANDOM_STATE = 42




## === cell 1
def _safe_read_csv(path):
    try:
        if os.path.exists(path):
            return pd.read_csv(path)
    except Exception:
        pass
    return None


pred_b3 = _safe_read_csv("../input/rcsiimpreds/sub_EfficientNetB3_384_9460.csv")
pred_b4 = _safe_read_csv("../input/rcsiimpreds/sub_EfficientNetB4_384_9498.csv")
pred_b5 = _safe_read_csv("../input/rcsiimpreds/sub_EfficientNetB5_384_9454.csv")
pred_b6 = _safe_read_csv("../input/rcsiimpreds/sub_EfficientNetB6_384_9481.csv")
pred_cw_b4 = _safe_read_csv("../input/rcsiimpreds/sub_EfficientNetB4_CW_384_9457.csv")
if pred_cw_b4 is not None:
    pred_cw_b4 = pred_cw_b4.rename(columns={"target": "target_cw_b4"})



## === cell 2
pred_512_B6 = _safe_read_csv("../input/rcsiimpreds/sub_B5_512_3fold_9466.csv")
if pred_512_B6 is not None:
    pred_512_B6 = pred_512_B6.rename(columns={"target": "target_B6_512"})



## === cell 3
pred_tta_b3 = _safe_read_csv("../input/rcsiimpreds/siim_tta_b3_9458.csv")
if pred_tta_b3 is not None:
    pred_tta_b3 = pred_tta_b3.rename(columns={"target": "target_tta_b3"})

pred_tta_b4 = _safe_read_csv("../input/rcsiimpreds/siim_tta_b4_9473.csv")
if pred_tta_b4 is not None:
    pred_tta_b4 = pred_tta_b4.rename(columns={"target": "target_tta_b4"})

result_tta = None
if (pred_b3 is not None) and (pred_b4 is not None):
    result_tta = pd.merge(
        pred_b3, pred_b4, on="image_name", suffixes=("_tta_b3", "_tta_b4")
    )



## === cell 4
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

DATA_DIR = "/kaggle/data"
train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_path)

y = train_df["target"].astype(int).values

train_patient_cnt = (
    train_df.groupby("patient_id")["image_name"]
    .transform("count")
    .astype("Int64")
    .fillna(0)
    .astype(np.int32)
)
test_patient_cnt = (
    test_df.groupby("patient_id")["image_name"]
    .transform("count")
    .astype("Int64")
    .fillna(0)
    .astype(np.int32)
)

train_site_cnt = (
    train_df.groupby("anatom_site_general_challenge")["image_name"]
    .transform("count")
    .astype("Int64")
    .fillna(0)
    .astype(np.int32)
)
test_site_cnt = (
    test_df.groupby("anatom_site_general_challenge")["image_name"]
    .transform("count")
    .astype("Int64")
    .fillna(0)
    .astype(np.int32)
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
global_mean = float(train_df["target"].mean())

alpha = 50.0

oof_patient_te = np.full(len(train_df), global_mean, dtype=np.float64)
oof_site_te = np.full(len(train_df), global_mean, dtype=np.float64)

oof_maps_patient = []
oof_maps_site = []

oof_diag_te = np.full(len(train_df), global_mean, dtype=np.float64)
oof_maps_diag = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, y)):
    tr = train_df.iloc[tr_idx]
    va = train_df.iloc[va_idx]

    g = tr.groupby("patient_id")["target"].agg(["mean", "count"])
    smooth = (g["mean"] * g["count"] + global_mean * alpha) / (g["count"] + alpha)
    oof_patient_te[va_idx] = va["patient_id"].map(smooth).fillna(global_mean).values
    oof_maps_patient.append(smooth)

    gs = tr.groupby("anatom_site_general_challenge")["target"].agg(["mean", "count"])
    smooth_s = (gs["mean"] * gs["count"] + global_mean * alpha) / (gs["count"] + alpha)
    oof_site_te[va_idx] = (
        va["anatom_site_general_challenge"].map(smooth_s).fillna(global_mean).values
    )
    oof_maps_site.append(smooth_s)

    gd = tr.groupby("diagnosis")["target"].agg(["mean", "count"])
    smooth_d = (gd["mean"] * gd["count"] + global_mean * alpha) / (gd["count"] + alpha)
    oof_diag_te[va_idx] = va["diagnosis"].map(smooth_d).fillna(global_mean).values
    oof_maps_diag.append(smooth_d)

test_patient_te_folds = []
test_site_te_folds = []
test_diag_te_folds = []
for smooth in oof_maps_patient:
    test_patient_te_folds.append(
        test_df["patient_id"].map(smooth).fillna(global_mean).values.astype(np.float64)
    )
for smooth_s in oof_maps_site:
    test_site_te_folds.append(
        test_df["anatom_site_general_challenge"]
        .map(smooth_s)
        .fillna(global_mean)
        .values.astype(np.float64)
    )
for smooth_d in oof_maps_diag:
    test_diag_te_folds.append(np.full(len(test_df), global_mean, dtype=np.float64))

test_patient_te = np.mean(np.vstack(test_patient_te_folds), axis=0)
test_site_te = np.mean(np.vstack(test_site_te_folds), axis=0)
test_diag_te = np.mean(np.vstack(test_diag_te_folds), axis=0)

train_age = pd.to_numeric(train_df["age_approx"], errors="coerce")
test_age = pd.to_numeric(test_df["age_approx"], errors="coerce")
train_log_age = np.log1p(train_age.clip(lower=0))
test_log_age = np.log1p(test_age.clip(lower=0))

X = train_df[["sex", "age_approx", "anatom_site_general_challenge"]].copy()
X["patient_image_count"] = train_patient_cnt.values
X["site_image_count"] = train_site_cnt.values
X["patient_target_mean"] = oof_patient_te
X["site_target_mean"] = oof_site_te
X["diagnosis_target_mean"] = oof_diag_te
X["log_age"] = train_log_age

X_test = test_df[["sex", "age_approx", "anatom_site_general_challenge"]].copy()
X_test["patient_image_count"] = test_patient_cnt.values
X_test["site_image_count"] = test_site_cnt.values
X_test["patient_target_mean"] = test_patient_te
X_test["site_target_mean"] = test_site_te
X_test["diagnosis_target_mean"] = test_diag_te
X_test["log_age"] = test_log_age

numeric_features = [
    "age_approx",
    "log_age",
    "patient_image_count",
    "site_image_count",
    "patient_target_mean",
    "site_target_mean",
    "diagnosis_target_mean",
]
categorical_features = ["sex", "anatom_site_general_challenge"]

preprocess = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
            numeric_features,
        ),
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            categorical_features,
        ),
    ],
    remainder="drop",
)

gb = GradientBoostingClassifier(
    random_state=RANDOM_STATE,
    n_estimators=600,
    learning_rate=0.02,
    max_depth=3,
    subsample=0.8,
)

model = Pipeline(steps=[("preprocess", preprocess), ("clf", gb)])

rng = np.random.RandomState(RANDOM_STATE)
X_train_fit = X.copy()
for col in ["patient_target_mean", "site_target_mean", "diagnosis_target_mean"]:
    X_train_fit[col] = (
        X_train_fit[col].values + rng.normal(0.0, 0.003, size=len(X_train_fit))
    ).clip(0.0, 1.0)

model.fit(X_train_fit, y)

oof_pred = np.zeros(len(train_df), dtype=np.float64)
for tr_idx, va_idx in skf.split(X, y):
    m = Pipeline(steps=[("preprocess", preprocess), ("clf", gb)])
    X_tr = X.iloc[tr_idx].copy()
    X_va = X.iloc[va_idx].copy()

    rng_fold = np.random.RandomState(RANDOM_STATE + int(tr_idx[0]))
    for col in ["patient_target_mean", "site_target_mean", "diagnosis_target_mean"]:
        X_tr[col] = (
            X_tr[col].values + rng_fold.normal(0.0, 0.003, size=len(X_tr))
        ).clip(0.0, 1.0)

    m.fit(X_tr, y[tr_idx])
    oof_pred[va_idx] = m.predict_proba(X_va)[:, 1]
print("Local 5-fold OOF AUC (metadata model):", roc_auc_score(y, oof_pred))

test_pred = model.predict_proba(X_test)[:, 1].astype(np.float64)

pred_df = pd.DataFrame(
    {"image_name": test_df["image_name"].values, "target": test_pred}
)
submit_file = sample_sub[["image_name"]].merge(pred_df, on="image_name", how="left")

if submit_file["target"].isna().any():
    submit_file["target"] = submit_file["target"].fillna(float(np.mean(y)))

submit_file["target"] = submit_file["target"].clip(0.0, 1.0)

final = submit_file.copy()



## === cell 5
final = submit_file.copy()



## === cell 6
result1 = None
if (pred_b3 is not None) and (pred_b4 is not None):
    result1 = pd.merge(pred_b3, pred_b4, on="image_name", suffixes=("_b3", "_b4"))



## === cell 7
if result1 is not None:
    _ = result1.head()



## === cell 8
result2 = None
if (pred_b5 is not None) and (pred_b6 is not None):
    result2 = pd.merge(pred_b5, pred_b6, on="image_name", suffixes=("_b5", "_b6"))



## === cell 9
if result2 is not None:
    _ = result2.head()



## === cell 10
semi_final = None
if (result1 is not None) and (result2 is not None):
    semi_final = pd.merge(result1, result2, on="image_name")



## === cell 11
if semi_final is not None:
    _ = semi_final.head()



## === cell 12
result3 = None
if (pred_cw_b4 is not None) and (pred_512_B6 is not None):
    result3 = pd.merge(
        pred_cw_b4, pred_512_B6, on="image_name", suffixes=("_cw_b4", "_B6_512")
    )
    _ = result3.head()



## === cell 13
if (semi_final is not None) and (result3 is not None):
    final = pd.merge(semi_final, result3, on="image_name")
    _ = final.head()



## === cell 14
if (
    (result_tta is not None)
    and ("image_name" in final.columns)
    and ("target" not in result_tta.columns)
):
    final = pd.merge(final, result_tta, on="image_name")
    _ = final.head()



## === cell 15
pred_kr_b3 = _safe_read_csv("../input/rcsiimpreds/KR_sub_EfficientNetB3_512_9520.csv")
if pred_kr_b3 is not None:
    pred_kr_b3 = pred_kr_b3.rename(columns={"target": "target_kr_b3"})

pred_kr_b4 = _safe_read_csv("../input/rcsiimpreds/KR_sub_EfficientNetB4_512_9499.csv")
if pred_kr_b4 is not None:
    pred_kr_b4 = pred_kr_b4.rename(columns={"target": "target_kr_b4"})

pred_kr_eb3 = _safe_read_csv("../input/rcsiimpreds/KR_sub_eb3_512_9554.csv")
if pred_kr_eb3 is not None:
    pred_kr_eb3 = pred_kr_eb3.rename(columns={"target": "target_kr_eb3"})



## === cell 16
kr_result = None
if (pred_kr_b3 is not None) and (pred_kr_b4 is not None):
    kr_result = pd.merge(
        pred_kr_b3, pred_kr_b4, on="image_name", suffixes=("_b3", "_b4")
    )
    if pred_kr_eb3 is not None:
        kr_result = pd.merge(kr_result, pred_kr_eb3, on="image_name")
    _ = kr_result.head()



## === cell 17
if (
    (kr_result is not None)
    and ("image_name" in final.columns)
    and ("target" not in kr_result.columns)
):
    final = pd.merge(final, kr_result, on="image_name")
    _ = final.head()



## === cell 18
pred_256_b4 = _safe_read_csv("../input/rcsiimpreds/sub_EfficientNetB4_256_9496.csv")
if pred_256_b4 is not None:
    pred_256_b4 = pred_256_b4.rename(columns={"target": "target_256_b4"})
    _ = pred_256_b4.head()



## === cell 19
if (
    (pred_256_b4 is not None)
    and ("image_name" in final.columns)
    and ("target" not in pred_256_b4.columns)
):
    final = pd.merge(final, pred_256_b4, on="image_name")
    _ = final.head()



## === cell 20
required_cols = [
    "target_b3",
    "target_b4",
    "target_b6",
    "target_B6_512",
    "target_tta_b4",
    "target_kr_b3",
    "target_kr_b4",
    "target_kr_eb3",
    "target_256_b4",
]
if all(col in final.columns for col in required_cols):
    final["target"] = (
        final["target_b3"]
        + final["target_b4"]
        + final["target_b6"]
        + final["target_B6_512"]
        + final["target_tta_b4"]
        + final["target_kr_b3"]
        + final["target_kr_b4"]
        + final["target_kr_eb3"]
        + final["target_256_b4"]
    ) / 9.0
    final["target"] = final["target"].clip(0.0, 1.0)



## === cell 21
_ = final.head()



## === cell 22
submit_file = sample_sub[["image_name"]].merge(
    final[["image_name", "target"]], on="image_name", how="left"
)

if submit_file["target"].isna().any():
    submit_file["target"] = submit_file["target"].fillna(float(np.mean(y)))

submit_file["target"] = submit_file["target"].clip(0.0, 1.0)



## === cell 23
_ = submit_file.head()



## === cell 24
submit_file.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submit_file.shape)
print(submit_file.head())
