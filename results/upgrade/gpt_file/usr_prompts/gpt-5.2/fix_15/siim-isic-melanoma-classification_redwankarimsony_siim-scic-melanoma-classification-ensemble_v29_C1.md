# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

np.random.seed(42)

DATA_ROOTS = [
    "/kaggle/input/siim-isic-melanoma-classification",
    "/kaggle/data/siim-isic-melanoma-classification",
    "/kaggle/input",
    "/kaggle/data",
]


def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


train_path = first_existing([os.path.join(r, "train.csv") for r in DATA_ROOTS])
test_path = first_existing([os.path.join(r, "test.csv") for r in DATA_ROOTS])
sample_path = first_existing(
    [os.path.join(r, "sample_submission.csv") for r in DATA_ROOTS]
)

if train_path is None or test_path is None or sample_path is None:
    raise FileNotFoundError(
        f"Missing required files. Found train={train_path}, test={test_path}, sample={sample_path}."
    )

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sub = pd.read_csv(sample_path)

for c in ["image_name", "target"]:
    if c not in sub.columns:
        raise ValueError(
            f"sample_submission.csv missing column {c}. Found {list(sub.columns)}"
        )
for c in ["target", "sex", "age_approx", "anatom_site_general_challenge", "patient_id"]:
    if c not in train_df.columns:
        raise ValueError(
            f"train.csv missing column {c}. Found {list(train_df.columns)}"
        )
for c in [
    "image_name",
    "sex",
    "age_approx",
    "anatom_site_general_challenge",
    "patient_id",
]:
    if c not in test_df.columns:
        raise ValueError(f"test.csv missing column {c}. Found {list(test_df.columns)}")

train_df["image_name"] = train_df["image_name"].astype(str)
test_df["image_name"] = test_df["image_name"].astype(str)
sub["image_name"] = sub["image_name"].astype(str)

print("Paths:")
print(" train:", train_path)
print(" test :", test_path)
print(" sample:", sample_path)
print("Shapes:", train_df.shape, test_df.shape, sub.shape)



## === cell 1
from sklearn.model_selection import GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from joblib import Parallel, delayed

TARGET = "target"
GROUP = "patient_id"


def add_minimal_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["sex"] = out["sex"].replace({"": np.nan, "unknown": np.nan, "Unknown": np.nan})
    out["anatom_site_general_challenge"] = out["anatom_site_general_challenge"].replace(
        {"": np.nan, "unknown": np.nan, "Unknown": np.nan}
    )

    out["age_approx"] = pd.to_numeric(out["age_approx"], errors="coerce")
    out["age_missing"] = out["age_approx"].isna().astype(np.int8)
    out["site_missing"] = out["anatom_site_general_challenge"].isna().astype(np.int8)

    age_clip = out["age_approx"].clip(lower=0)
    age_med = float(age_clip.median(skipna=True)) if age_clip.notna().any() else 0.0
    age_filled = age_clip.fillna(age_med)

    out["age_log1p"] = np.log1p(age_clip)
    out["age_squared"] = (age_clip**2).astype(np.float64)

    sex_is_male = (out["sex"].astype(str).to_numpy() == "male").astype(np.int8)
    out["age_x_male"] = (age_filled.to_numpy(dtype=np.float64) * sex_is_male).astype(
        np.float64
    )

    top_sites = [
        "torso",
        "lower extremity",
        "upper extremity",
        "head/neck",
        "palms/soles",
        "oral/genital",
    ]
    site_series = out["anatom_site_general_challenge"].astype(str).to_numpy()
    age_vals = age_filled.to_numpy(dtype=np.float64)
    for s in top_sites:
        out[f"age_x_site_{s}"] = (age_vals * (site_series == s).astype(np.int8)).astype(
            np.float64
        )

    return out


train_feat = add_minimal_features(train_df)
test_feat = add_minimal_features(test_df)

FEATURES_NUM = [
    "age_approx",
    "age_log1p",
    "age_squared",
    "age_missing",
    "site_missing",
    "age_x_male",
    "age_x_site_torso",
    "age_x_site_lower extremity",
    "age_x_site_upper extremity",
    "age_x_site_head/neck",
    "age_x_site_palms/soles",
    "age_x_site_oral/genital",
]
FEATURES_CAT = ["sex", "anatom_site_general_challenge"]

X_train_df = train_feat[FEATURES_NUM + FEATURES_CAT]
y_train = train_feat[TARGET].astype(np.int32).to_numpy()
groups = train_feat[GROUP].astype(str).to_numpy()
X_test_df = test_feat[FEATURES_NUM + FEATURES_CAT]

_ohe_kwargs = {"handle_unknown": "ignore"}
try:
    OneHotEncoder(sparse_output=False, **_ohe_kwargs)
    _ohe_kwargs["sparse_output"] = False
except TypeError:
    _ohe_kwargs["sparse"] = False

preprocess = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ]
            ),
            FEATURES_NUM,
        ),
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(**_ohe_kwargs)),
                ]
            ),
            FEATURES_CAT,
        ),
    ],
    remainder="drop",
)

prep_fitted = preprocess.fit(X_train_df)
X_train = prep_fitted.transform(X_train_df)
X_test = prep_fitted.transform(X_test_df)

X_train = np.asarray(X_train, dtype=np.float64)
X_test = np.asarray(X_test, dtype=np.float64)

gkf = GroupKFold(n_splits=5)
splits = [
    (tr.astype(np.intp, copy=False), va.astype(np.intp, copy=False))
    for tr, va in gkf.split(X_train, y_train, groups=groups)
]

_fold_data = []
for tr_idx, va_idx in splits:
    _fold_data.append((X_train[tr_idx], y_train[tr_idx], X_train[va_idx], va_idx))

cpu = os.cpu_count() or 2
FOLD_NJOBS = max(1, min(5, cpu))  # at most 5 folds
SOLVER_NJOBS = 1


def make_lr(C: float, class_weight, penalty: str, l1_ratio, warm_start: bool):
    return LogisticRegression(
        solver="saga",
        max_iter=2000,
        class_weight=class_weight,
        random_state=42,
        C=C,
        penalty=penalty,
        l1_ratio=l1_ratio,  # only used for elasticnet
        n_jobs=SOLVER_NJOBS,
        warm_start=warm_start,
    )


C_grid = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
class_weight_grid = [None, "balanced"]
penalty_grid = ["l2", "l1", "elasticnet"]
l1_ratio_grid = [0.1, 0.5]  # only for elasticnet; ignored otherwise

best_params = None
best_auc = -np.inf
oof = np.empty(X_train.shape[0], dtype=np.float64)


def _fit_predict_fold(clf, X_tr, y_tr, X_va, va_idx):
    clf.fit(X_tr, y_tr)
    return va_idx, clf.predict_proba(X_va)[:, 1], clf


for cw in class_weight_grid:
    for penalty in penalty_grid:
        l1_ratios = l1_ratio_grid if penalty == "elasticnet" else [None]
        for l1r in l1_ratios:
            fold_clfs = [
                make_lr(
                    C=C_grid[0],
                    class_weight=cw,
                    penalty=penalty,
                    l1_ratio=l1r,
                    warm_start=True,
                )
                for _ in range(len(_fold_data))
            ]

            for C in C_grid:
                for clf in fold_clfs:
                    clf.C = C

                results = Parallel(n_jobs=FOLD_NJOBS, prefer="processes")(
                    delayed(_fit_predict_fold)(fold_clfs[i], X_tr, y_tr, X_va, va_idx)
                    for i, (X_tr, y_tr, X_va, va_idx) in enumerate(_fold_data)
                )

                for va_idx, proba, clf_out in results:
                    oof[va_idx] = proba

                auc = roc_auc_score(y_train, oof)
                print(
                    f"class_weight={str(cw):>8}  C={C:<6}  penalty={penalty:<10}  "
                    f"l1_ratio={str(l1r):>4}  OOF AUC={auc:.6f}"
                )
                if auc > best_auc:
                    best_auc = auc
                    best_params = {
                        "C": C,
                        "class_weight": cw,
                        "penalty": penalty,
                        "l1_ratio": l1r,
                    }

print(f"Selected params: {best_params} with OOF AUC={best_auc:.6f}")

final_lr = make_lr(
    C=best_params["C"],
    class_weight=best_params["class_weight"],
    penalty=best_params["penalty"],
    l1_ratio=best_params["l1_ratio"],
    warm_start=False,  # final fit from scratch to match standard training semantics
)
final_lr.fit(X_train, y_train)

test_proba = final_lr.predict_proba(X_test)[:, 1].astype(np.float64)
test_proba = np.clip(test_proba, 0.0, 1.0)

test_pred_series = pd.Series(test_proba, index=test_feat["image_name"].to_numpy())
sub["target"] = sub["image_name"].map(test_pred_series)

fallback = float(train_feat[TARGET].mean())
sub["target"] = sub["target"].fillna(fallback).clip(0.0, 1.0)

print("Train prevalence fallback:", fallback)
print("Submission target stats:", sub["target"].describe())



## === cell 2
out_path = "submission.csv"
sub[["image_name", "target"]].to_csv(out_path, index=False)

print(sub.head())
print(
    f"Wrote: {out_path}  rows={len(sub)} cols={list(sub.columns)}  "
    f"target_range=({sub.target.min():.6f},{sub.target.max():.6f})"
)
