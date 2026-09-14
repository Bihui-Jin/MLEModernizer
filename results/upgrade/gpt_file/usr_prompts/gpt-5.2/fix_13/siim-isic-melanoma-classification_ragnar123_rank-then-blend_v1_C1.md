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
import pandas as pd
import numpy as np

os.environ.setdefault("PYTHONHASHSEED", "42")
np.random.seed(42)

DATA_DIR_CANDIDATES = [
    "/kaggle/input/siim-isic-melanoma-classification",
    "/kaggle/data",
    "/kaggle/input",
]


def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


base_dir = _first_existing(DATA_DIR_CANDIDATES)
if base_dir is None:
    raise FileNotFoundError(
        f"Could not find dataset directory in any of: {DATA_DIR_CANDIDATES}"
    )

test_path = _first_existing(
    [
        os.path.join(base_dir, "test.csv"),
        "/kaggle/data/test.csv",
        "/kaggle/input/test.csv",
    ]
)
train_path = _first_existing(
    [
        os.path.join(base_dir, "train.csv"),
        "/kaggle/data/train.csv",
        "/kaggle/input/train.csv",
    ]
)
sample_sub_path = _first_existing(
    [
        os.path.join(base_dir, "sample_submission.csv"),
        "/kaggle/data/sample_submission.csv",
        "/kaggle/input/sample_submission.csv",
    ]
)

test_df = pd.read_csv(test_path)
train_df = pd.read_csv(train_path)
sample_sub = pd.read_csv(sample_sub_path)



## === cell 1
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold


def add_meta_features(df: pd.DataFrame, train_patient_counts=None) -> pd.DataFrame:
    df = df.copy()

    age = pd.to_numeric(df["age_approx"], errors="coerce")
    age_clip0 = age.clip(lower=0)

    df["age_missing"] = age.isna().astype(np.int8)

    bins = [-np.inf, 20, 30, 40, 50, 60, 70, np.inf]
    df["age_bin"] = pd.cut(age, bins=bins, labels=False, include_lowest=True).astype(
        "float"
    )
    df["age_log1p"] = np.log1p(age_clip0).astype(np.float64)

    sex = df["sex"].fillna("unknown").astype(str)
    site = df["anatom_site_general_challenge"].fillna("unknown").astype(str)
    df["sex_site"] = (sex + "__" + site).astype(str)

    df["age_sq"] = (age_clip0**2).astype("float")
    df["sex_known"] = (
        (~df["sex"].isna()) & (df["sex"].astype(str).str.len() > 0)
    ).astype(np.int8)

    if train_patient_counts is not None:
        pid = df["patient_id"]
        pc = pid.map(train_patient_counts).fillna(0).astype("float")
        df["patient_count"] = pc
        df["patient_count_log1p"] = np.log1p(pc).astype(np.float64)
    else:
        df["patient_count"] = 0.0
        df["patient_count_log1p"] = 0.0

    return df


train_patient_counts = train_df["patient_id"].value_counts(dropna=False)

FEATURES = [
    "patient_id",
    "sex",
    "age_approx",
    "anatom_site_general_challenge",
    "age_missing",
    "age_bin",
    "age_log1p",
    "sex_site",
    "age_sq",
    "sex_known",
    "patient_count",
    "patient_count_log1p",
]

X_all = add_meta_features(train_df, train_patient_counts=train_patient_counts)[FEATURES]
y_all = train_df["target"].astype(int).to_numpy()
X_test = add_meta_features(test_df, train_patient_counts=train_patient_counts)[FEATURES]

numeric_features = [
    "age_approx",
    "age_log1p",
    "age_sq",
    "patient_count",
    "patient_count_log1p",
]
categorical_features = [
    "patient_id",
    "sex",
    "anatom_site_general_challenge",
    "age_missing",
    "age_bin",
    "sex_site",
    "sex_known",
]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore", sparse_output=True, min_frequency=5),
        ),
    ]
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)



## === cell 2
from scipy import sparse


def make_lr():
    return LogisticRegression(
        max_iter=1200,
        solver="saga",
        penalty="elasticnet",
        l1_ratio=0.15,
        C=1.2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,  # keep same training logic
    )


preprocess_fitted = preprocess.fit(X_all)
X_all_tr = preprocess_fitted.transform(X_all)
X_test_tr = preprocess_fitted.transform(X_test)

if sparse.issparse(X_all_tr):
    X_all_tr = X_all_tr.tocsr()
if sparse.issparse(X_test_tr):
    X_test_tr = X_test_tr.tocsr()

if sparse.issparse(X_all_tr) and X_all_tr.dtype != np.float32:
    X_all_tr = X_all_tr.astype(np.float32)
if sparse.issparse(X_test_tr) and X_test_tr.dtype != np.float32:
    X_test_tr = X_test_tr.astype(np.float32)

X_all_tr_csc = X_all_tr.tocsc(copy=False)

groups = train_df["patient_id"].astype(str).fillna("nan").to_numpy()
gkf = GroupKFold(n_splits=5)
fold_indices = list(gkf.split(np.empty(len(y_all)), y_all, groups=groups))

oof_pred = np.zeros(len(y_all), dtype=np.float64)

for tr_idx, va_idx in fold_indices:
    clf = make_lr()

    X_tr = X_all_tr_csc[tr_idx]
    X_va = X_all_tr[va_idx]
    y_tr = y_all[tr_idx]

    clf.fit(X_tr, y_tr)
    oof_pred[va_idx] = clf.predict_proba(X_va)[:, 1]

eps = 1e-6
oof_clip = np.clip(oof_pred, eps, 1.0 - eps)
oof_logit = np.log(oof_clip / (1.0 - oof_clip)).reshape(-1, 1)

platt = LogisticRegression(
    solver="lbfgs",
    max_iter=500,
    class_weight=None,
    random_state=42,
)
platt.fit(oof_logit, y_all)

final_clf = make_lr()
final_clf.fit(X_all_tr_csc, y_all)
meta_pred_raw = final_clf.predict_proba(X_test_tr)[:, 1]

meta_pred_clip = np.clip(meta_pred_raw, eps, 1.0 - eps)
meta_logit = np.log(meta_pred_clip / (1.0 - meta_pred_clip)).reshape(-1, 1)
meta_pred = platt.predict_proba(meta_logit)[:, 1]

from scipy.stats import rankdata

meta_rank = rankdata(meta_pred, method="average")
meta_rank = meta_rank / meta_rank.max()
meta_pred = 0.85 * meta_pred + 0.15 * meta_rank
meta_pred = np.clip(meta_pred, 0.0, 1.0)

fallback_sub = pd.DataFrame(
    {"image_name": test_df["image_name"].to_numpy(), "target": meta_pred}
)



## === cell 3
external_candidates = [
    "../input/melanoma-dif-sub/pl_0.936.csv",
    "../input/melanoma-dif-sub/pl_0.940.csv",
    "../input/melanoma-dif-sub/sub_EfficientNetB2_384.csv",
    "../input/melanoma-dif-sub/sub_EfficientNetB3_384.csv",
    "../input/melanoma-dif-sub/sub_EfficientNetB3_384_v2.csv",
]


def load_or_fallback(path, fallback_df):
    if os.path.exists(path):
        return pd.read_csv(path, usecols=["image_name", "target"])
    return fallback_df.copy()


def rank_data(sub):
    sub = sub[["image_name", "target"]].copy()
    r = rankdata(sub["target"].to_numpy(dtype=np.float64, copy=False), method="average")
    sub["target"] = r / r.max()
    return sub


sub1 = rank_data(load_or_fallback(external_candidates[0], fallback_sub))
sub2 = rank_data(load_or_fallback(external_candidates[1], fallback_sub))
sub3 = rank_data(load_or_fallback(external_candidates[2], fallback_sub))
sub4 = rank_data(load_or_fallback(external_candidates[3], fallback_sub))
sub5 = rank_data(load_or_fallback(external_candidates[4], fallback_sub))

sub1 = sub1.set_index("image_name").rename(columns={"target": "target1"})
sub2 = sub2.set_index("image_name").rename(columns={"target": "target2"})
sub3 = sub3.set_index("image_name").rename(columns={"target": "target3"})
sub4 = sub4.set_index("image_name").rename(columns={"target": "target4"})
sub5 = sub5.set_index("image_name").rename(columns={"target": "target5"})

f_sub = (
    sub1.join(sub2, how="inner")
    .join(sub3, how="inner")
    .join(sub4, how="inner")
    .join(sub5, how="inner")
)

w = np.array([0.2, 0.3, 0.1, 0.2, 0.1], dtype=np.float64)
targets = f_sub[["target1", "target2", "target3", "target4", "target5"]].to_numpy(
    dtype=np.float64, copy=False
)
f_sub["target"] = targets @ w
f_sub = f_sub[["target"]].reset_index()

out = test_df[["image_name"]].merge(
    f_sub[["image_name", "target"]], on="image_name", how="left"
)

if out["target"].isna().any():
    out = out.merge(
        fallback_sub, on="image_name", how="left", suffixes=("", "_fallback")
    )
    out["target"] = out["target"].fillna(out["target_fallback"])
    out = out[["image_name", "target"]]

out["target"] = pd.to_numeric(out["target"], errors="coerce").fillna(0.5).clip(0.0, 1.0)

out.to_csv("blend_sub.csv", index=False)
print(out.head())
print(
    f"Wrote blend_sub.csv with shape={out.shape} to {os.path.abspath('blend_sub.csv')}"
)
