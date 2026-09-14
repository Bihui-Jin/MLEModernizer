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

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_DIR_CANDIDATES = [
    "/kaggle/input/siim-isic-melanoma-classification",
    "/kaggle/data/siim-isic-melanoma-classification",
    "/kaggle/data",
    "/kaggle/input",
]


def find_file(filename: str) -> str:
    for d in DATA_DIR_CANDIDATES:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p

    common_nested = [
        "/kaggle/input/siim-isic-melanoma-classification/siim-isic-melanoma-classification",
        "/kaggle/data/siim-isic-melanoma-classification/siim-isic-melanoma-classification",
    ]
    for d in common_nested:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p

    raise FileNotFoundError(
        f"Could not find {filename} under known Kaggle dataset roots: {DATA_DIR_CANDIDATES}"
    )


train_path = find_file("train.csv")
test_path = find_file("test.csv")
sample_path = find_file("sample_submission.csv")

print("train.csv:", train_path)
print("test.csv:", test_path)
print("sample_submission.csv:", sample_path)

train_df = pd.read_csv(
    train_path,
    dtype={
        "image_name": "string",
        "patient_id": "string",
        "sex": "string",
        "anatom_site_general_challenge": "string",
        "diagnosis": "string",
        "benign_malignant": "string",
        "target": "int8",
    },
)
test_df = pd.read_csv(
    test_path,
    dtype={
        "image_name": "string",
        "patient_id": "string",
        "sex": "string",
        "anatom_site_general_challenge": "string",
    },
)
sub_df = pd.read_csv(sample_path, dtype={"image_name": "string"})

print(train_df.shape, test_df.shape, sub_df.shape)
print("train columns:", list(train_df.columns))
print("test columns:", list(test_df.columns))
print("sub columns:", list(sub_df.columns))



## === cell 1
required_train = {
    "image_name",
    "target",
    "sex",
    "age_approx",
    "anatom_site_general_challenge",
    "patient_id",
}
required_test = {
    "image_name",
    "sex",
    "age_approx",
    "anatom_site_general_challenge",
    "patient_id",
}

missing_train = required_train - set(train_df.columns)
missing_test = required_test - set(test_df.columns)
if missing_train:
    raise ValueError(f"train.csv missing columns: {missing_train}")
if missing_test:
    raise ValueError(f"test.csv missing columns: {missing_test}")

X_train = train_df[
    ["sex", "age_approx", "anatom_site_general_challenge", "patient_id"]
].copy()
y_train = train_df["target"].astype(np.int8).to_numpy()
X_test = test_df[
    ["sex", "age_approx", "anatom_site_general_challenge", "patient_id"]
].copy()

groups = train_df["patient_id"].astype("string").fillna("unknown_patient").to_numpy()

for df in (X_train, X_test):
    df["sex"] = df["sex"].astype("string")
    df["anatom_site_general_challenge"] = df["anatom_site_general_challenge"].astype(
        "string"
    )
    df["patient_id"] = df["patient_id"].astype("string")

    sex_s = df["sex"]
    site_s = df["anatom_site_general_challenge"]
    pat_s = df["patient_id"]
    age_s = df["age_approx"]

    df["sex_missing"] = sex_s.isna() | (sex_s.str.len().fillna(0) == 0)
    df["site_missing"] = site_s.isna() | (site_s.str.len().fillna(0) == 0)
    df["age_missing"] = age_s.isna() | (age_s.astype("string").str.len().fillna(0) == 0)
    df["patient_missing"] = pat_s.isna() | (pat_s.str.len().fillna(0) == 0)

    df["sex"] = sex_s.fillna("unknown").replace("", "unknown").astype(str)
    df["anatom_site_general_challenge"] = (
        site_s.fillna("unknown").replace("", "unknown").astype(str)
    )
    df["patient_id"] = (
        pat_s.fillna("unknown_patient").replace("", "unknown_patient").astype(str)
    )
    df["age_approx"] = pd.to_numeric(df["age_approx"], errors="coerce")

age_median = float(np.nanmedian(X_train["age_approx"].to_numpy()))
X_train["age_approx"] = X_train["age_approx"].fillna(age_median)
X_test["age_approx"] = X_test["age_approx"].fillna(age_median)

for df in (X_train, X_test):
    for c in ["sex_missing", "site_missing", "age_missing", "patient_missing"]:
        df[c] = df[c].astype(np.int8)



## === cell 2
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import roc_auc_score
from scipy import sparse

unique_groups = np.unique(groups)
n_splits = 5 if unique_groups.shape[0] >= 5 else max(2, unique_groups.shape[0])
gkf = GroupKFold(n_splits=n_splits)

folds = list(gkf.split(X_train, y_train, groups=groups))
folds = [
    (np.asarray(tr, dtype=np.int32), np.asarray(va, dtype=np.int32)) for tr, va in folds
]

global_rate = float(np.mean(y_train))


def _rate_to_logodds(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


global_logodds = float(_rate_to_logodds(np.asarray([global_rate]))[0])

ALPHA_PATIENT = 5.0
ALPHA_CAT_GRID = [1.0, 3.0, 5.0, 10.0]


def add_oof_target_encodings(
    Xtr: pd.DataFrame,
    Xte: pd.DataFrame,
    y: np.ndarray,
    folds_idx,
    col: str,
    prefix: str,
    alpha: float,
    global_rate: float,
    global_logodds: float,
):
    oof_rate = np.zeros(Xtr.shape[0], dtype=np.float64)
    oof_logodds = np.zeros(Xtr.shape[0], dtype=np.float64)

    tr_series = Xtr[col].astype(str)

    for tr_idx, va_idx in folds_idx:
        tr_key = tr_series.iloc[tr_idx].to_numpy()
        tr_y = y[tr_idx]

        fold_df = pd.DataFrame({"key": tr_key, "y": tr_y})
        gb = fold_df.groupby("key", sort=False)["y"]
        s = gb.sum()
        c = gb.count()

        rate_smooth = (s + alpha * global_rate) / (c + alpha)
        logodds_smooth = _rate_to_logodds(rate_smooth.to_numpy(dtype=float))

        rate_map = rate_smooth
        logodds_map = pd.Series(logodds_smooth, index=rate_smooth.index)

        va_key = tr_series.iloc[va_idx]
        oof_rate[va_idx] = (
            va_key.map(rate_map).fillna(global_rate).to_numpy(dtype=float)
        )
        oof_logodds[va_idx] = (
            va_key.map(logodds_map).fillna(global_logodds).to_numpy(dtype=float)
        )

    full_df = pd.DataFrame({"key": tr_series.to_numpy(), "y": y})
    gb_full = full_df.groupby("key", sort=False)["y"]
    s_full = gb_full.sum()
    c_full = gb_full.count()
    full_rate_smooth = (s_full + alpha * global_rate) / (c_full + alpha)
    full_logodds_smooth = _rate_to_logodds(full_rate_smooth.to_numpy(dtype=float))
    full_logodds_series = pd.Series(full_logodds_smooth, index=full_rate_smooth.index)

    te_series = Xte[col].astype(str)
    te_rate = te_series.map(full_rate_smooth).fillna(global_rate).to_numpy(dtype=float)
    te_logodds = (
        te_series.map(full_logodds_series).fillna(global_logodds).to_numpy(dtype=float)
    )

    Xtr[f"{prefix}_target_rate"] = oof_rate
    Xte[f"{prefix}_target_rate"] = te_rate
    Xtr[f"{prefix}_logodds"] = oof_logodds
    Xte[f"{prefix}_logodds"] = te_logodds


def build_design_matrices(alpha_cat: float):
    Xtr = X_train.copy()
    Xte = X_test.copy()

    oof_patient_rate = np.zeros(Xtr.shape[0], dtype=np.float64)
    oof_patient_logodds = np.zeros(Xtr.shape[0], dtype=np.float64)

    train_patient_series = Xtr["patient_id"]

    for tr_idx, va_idx in folds:
        tr_pat = train_patient_series.iloc[tr_idx].to_numpy()
        tr_y = y_train[tr_idx]

        fold_df = pd.DataFrame({"patient_id": tr_pat, "y": tr_y})
        gb = fold_df.groupby("patient_id", sort=False)["y"]
        pat_sum = gb.sum()
        pat_cnt = gb.count()

        pat_rate_smooth = (pat_sum + ALPHA_PATIENT * global_rate) / (
            pat_cnt + ALPHA_PATIENT
        )
        pat_logodds_smooth = _rate_to_logodds(pat_rate_smooth.to_numpy(dtype=float))

        pat_rate_map = pat_rate_smooth
        pat_logodds_map = pd.Series(
            pat_logodds_smooth, index=pat_rate_smooth.index, name="patient_logodds"
        )

        va_pat = train_patient_series.iloc[va_idx]
        oof_patient_rate[va_idx] = (
            va_pat.map(pat_rate_map).fillna(global_rate).to_numpy(dtype=float)
        )
        oof_patient_logodds[va_idx] = (
            va_pat.map(pat_logodds_map).fillna(global_logodds).to_numpy(dtype=float)
        )

    full_df = pd.DataFrame(
        {"patient_id": train_patient_series.to_numpy(), "y": y_train}
    )
    gb_full = full_df.groupby("patient_id", sort=False)["y"]
    full_sum = gb_full.sum()
    full_cnt = gb_full.count()
    full_rate_smooth = (full_sum + ALPHA_PATIENT * global_rate) / (
        full_cnt + ALPHA_PATIENT
    )
    full_logodds_smooth = _rate_to_logodds(full_rate_smooth.to_numpy(dtype=float))
    full_logodds_series = pd.Series(
        full_logodds_smooth, index=full_rate_smooth.index, name="patient_logodds"
    )

    Xtr["patient_target_rate"] = oof_patient_rate
    Xte["patient_target_rate"] = (
        Xte["patient_id"]
        .map(full_rate_smooth)
        .fillna(global_rate)
        .to_numpy(dtype=float)
    )

    Xtr["patient_logodds"] = oof_patient_logodds
    Xte["patient_logodds"] = (
        Xte["patient_id"]
        .map(full_logodds_series)
        .fillna(global_logodds)
        .to_numpy(dtype=float)
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    te_folds = [
        (tr.astype(np.int32), va.astype(np.int32)) for tr, va in kf.split(Xtr, y_train)
    ]

    add_oof_target_encodings(
        Xtr,
        Xte,
        y_train,
        te_folds,
        col="sex",
        prefix="sex",
        alpha=alpha_cat,
        global_rate=global_rate,
        global_logodds=global_logodds,
    )
    add_oof_target_encodings(
        Xtr,
        Xte,
        y_train,
        te_folds,
        col="anatom_site_general_challenge",
        prefix="site",
        alpha=alpha_cat,
        global_rate=global_rate,
        global_logodds=global_logodds,
    )

    return Xtr, Xte


categorical_features = ["sex", "anatom_site_general_challenge"]
numeric_features = [
    "age_approx",
    "sex_missing",
    "site_missing",
    "age_missing",
    "patient_missing",
    "patient_target_rate",
    "patient_logodds",
    "sex_target_rate",
    "sex_logodds",
    "site_target_rate",
    "site_logodds",
]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", StandardScaler(with_mean=True), numeric_features),
    ]
)

C_grid = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
cw_grid = [None, "balanced"]

best = {"auc": -np.inf, "C": None, "class_weight": None, "alpha_cat": None}

for alpha_cat in ALPHA_CAT_GRID:
    Xtr_alpha, Xte_alpha = build_design_matrices(alpha_cat=float(alpha_cat))

    fold_cache = []
    for tr_idx, va_idx in folds:
        pp_fold = ColumnTransformer(
            transformers=preprocess.transformers,
            remainder=preprocess.remainder,
            sparse_threshold=preprocess.sparse_threshold,
            n_jobs=preprocess.n_jobs,
            transformer_weights=preprocess.transformer_weights,
            verbose=preprocess.verbose,
            verbose_feature_names_out=preprocess.verbose_feature_names_out,
        )
        X_tr = Xtr_alpha.iloc[tr_idx]
        X_va = Xtr_alpha.iloc[va_idx]
        X_tr_tf = pp_fold.fit_transform(X_tr, y_train[tr_idx])
        X_va_tf = pp_fold.transform(X_va)

        if sparse.issparse(X_tr_tf):
            X_tr_tf = X_tr_tf.tocsr()
        if sparse.issparse(X_va_tf):
            X_va_tf = X_va_tf.tocsr()

        fold_cache.append((tr_idx, va_idx, X_tr_tf, X_va_tf, y_train[tr_idx]))

    for cw in cw_grid:
        fold_models = []
        for tr_idx, va_idx, X_tr_tf, X_va_tf, y_tr in fold_cache:
            clf = LogisticRegression(
                max_iter=2000,
                solver="saga",
                penalty="l2",
                C=float(C_grid[0]),
                class_weight=cw,
                random_state=RANDOM_STATE,
                n_jobs=1,
                warm_start=True,
            )
            fold_models.append(clf)

        for C in C_grid:
            oof = np.zeros(Xtr_alpha.shape[0], dtype=np.float64)
            for i, (tr_idx, va_idx, X_tr_tf, X_va_tf, y_tr) in enumerate(fold_cache):
                clf = fold_models[i]
                clf.set_params(C=float(C))
                clf.fit(X_tr_tf, y_tr)
                oof[va_idx] = clf.predict_proba(X_va_tf)[:, 1]
            auc = roc_auc_score(y_train, oof)
            print(
                f"CV AUC={auc:.6f} | alpha_cat={alpha_cat} | C={C} | class_weight={cw}"
            )
            if auc > best["auc"]:
                best = {
                    "auc": float(auc),
                    "C": float(C),
                    "class_weight": cw,
                    "alpha_cat": float(alpha_cat),
                }

print("Best params:", best)

X_train_final, X_test_final = build_design_matrices(alpha_cat=float(best["alpha_cat"]))

X_train_tf = preprocess.fit_transform(X_train_final, y_train)
X_test_tf = preprocess.transform(X_test_final)
if sparse.issparse(X_train_tf):
    X_train_tf = X_train_tf.tocsr()
if sparse.issparse(X_test_tf):
    X_test_tf = X_test_tf.tocsr()

final_clf = LogisticRegression(
    max_iter=2000,
    solver="saga",
    penalty="l2",
    C=float(best["C"]),
    class_weight=best["class_weight"],
    random_state=RANDOM_STATE,
    n_jobs=1,
)
final_clf.fit(X_train_tf, y_train)



## === cell 3
test_pred = final_clf.predict_proba(X_test_tf)[:, 1].astype(np.float64)

test_pred_map = pd.DataFrame(
    {"image_name": test_df["image_name"].to_numpy(), "target": test_pred}
)
submission = sub_df[["image_name"]].merge(test_pred_map, on="image_name", how="left")

if submission["target"].isna().any():
    submission["target"] = submission["target"].fillna(float(np.nanmean(test_pred)))

submission["target"] = submission["target"].clip(0.0, 1.0)

print(submission.head())
print(submission.shape)
print(submission.isna().sum())



## === cell 4
out_path = "submission.csv"
submission.to_csv(out_path, index=False)

assert list(submission.columns) == [
    "image_name",
    "target",
], "Submission must have columns: image_name,target"
assert (
    submission.shape[0] == sub_df.shape[0]
), "Submission row count must match sample_submission"

print(f"Wrote {out_path} with shape {submission.shape}")
print(submission.describe(include="all"))
