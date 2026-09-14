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
seaborn==0.12.2
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
import glob
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_INPUT = "../input"
COMP_DIR = os.path.join(BASE_INPUT, "siim-isic-melanoma-classification")

LABELS = ["target"]

print("Listing ../input:")
try:
    print("\n".join(sorted(os.listdir(BASE_INPUT))[:50]))
except Exception as e:
    print("Could not list ../input:", repr(e))




## === cell 1
from scipy.stats import rankdata

PSEUDO_DIR = os.path.join(BASE_INPUT, "pseudolabelmodels")
all_files = glob.glob(os.path.join(PSEUDO_DIR, "*.csv"))
print(f"Found {len(all_files)} files under {PSEUDO_DIR}")


def _load_csv_safe(path1, path2):
    if os.path.exists(path1):
        return pd.read_csv(path1)
    return pd.read_csv(path2)


train_path = os.path.join(COMP_DIR, "train.csv")
test_path = os.path.join(COMP_DIR, "test.csv")
sample_path = os.path.join(COMP_DIR, "sample_submission.csv")

train_df = _load_csv_safe(os.path.join(BASE_INPUT, "train.csv"), train_path)
test_df = _load_csv_safe(os.path.join(BASE_INPUT, "test.csv"), test_path)
sample_sub = _load_csv_safe(
    os.path.join(BASE_INPUT, "sample_submission.csv"), sample_path
)

test_df = test_df.merge(sample_sub[["image_name"]], on="image_name", how="right")
test_image_order = sample_sub["image_name"].to_numpy()
expected_n_test = len(sample_sub)

test_image_index = pd.Index(test_image_order, name="image_name")

print("Discovered CSV candidates (first 40):")
print("\n".join(all_files[:40]))
print("Total discovered candidates:", len(all_files))




## === cell 2
concat_sub = None
usable_files = []

_MIN_BYTES_PER_ROW_GUESS = 10
_MAX_BYTES_PER_ROW_GUESS = 200

_min_expected = max(1, expected_n_test) * _MIN_BYTES_PER_ROW_GUESS
_max_expected = (
    max(1, expected_n_test) * _MAX_BYTES_PER_ROW_GUESS * 50
)  # keep original leniency


def _quick_validate_pred_csv(path: str):
    try:
        st = os.stat(path)
        if st.st_size < _min_expected or st.st_size > _max_expected:
            return False
    except Exception:
        return False

    try:
        with open(path, "rb") as f:
            head = f.readline(4096)
        header = head.decode("utf-8-sig", errors="ignore").strip().lower()
    except Exception:
        return False

    cols = {c.strip() for c in header.split(",")}
    if not {"image_name", "target"}.issubset(cols):
        return False
    return True


def _read_and_validate_targets_only(path: str):
    try:
        read_kwargs = dict(
            usecols=["image_name", "target"],
            dtype={"image_name": "string", "target": "float64"},
        )
        try:
            df = pd.read_csv(path, engine="pyarrow", **read_kwargs)
        except Exception:
            df = pd.read_csv(path, **read_kwargs)
    except Exception:
        return None

    if df["image_name"].duplicated().any():
        df = df.drop_duplicates(subset=["image_name"], keep="first")

    if len(df) != expected_n_test:
        return None

    names = df["image_name"].to_numpy()
    if names.shape[0] != test_image_order.shape[0]:
        return None

    if not np.array_equal(names, test_image_order):
        try:
            df2 = df.set_index("image_name")
            if df2.index.nunique() != test_image_index.size:
                return None
            if not df2.index.isin(test_image_index).all():
                return None
            df2 = df2.reindex(test_image_index)
            tgt2 = df2["target"]
            if tgt2.isna().any():
                return None
            return tgt2.to_numpy(dtype=np.float64, copy=False)
        except Exception:
            return None

    tgt = df["target"].to_numpy(dtype=np.float64, copy=False)
    if np.isnan(tgt).any():
        return None
    return tgt


_MAX_USABLE_FILES = 50

usable_targets = []
for f in all_files:
    if len(usable_targets) >= _MAX_USABLE_FILES:
        break
    if not _quick_validate_pred_csv(f):
        continue

    tgt = _read_and_validate_targets_only(f)
    if tgt is None:
        continue

    usable_files.append(f)
    usable_targets.append(tgt)

print(
    f"Usable prediction-like CSVs with image_name+target matching test set: {len(usable_files)}"
)
print("Usable files (up to 20):")
print("\n".join(usable_files[:20]))

if len(usable_targets) >= 2:
    preds_mat = np.column_stack(usable_targets).astype(np.float64, copy=False)
    concat_sub = pd.DataFrame(
        preds_mat, columns=[f"m{i}" for i in range(preds_mat.shape[1])]
    )
    concat_sub.insert(0, "image_name", test_image_order)
    print("concat_sub shape:", concat_sub.shape)
else:
    print("Not enough external prediction files to ensemble; will use fallback model.")
    concat_sub = None




## === cell 3
m_gmean = None
if concat_sub is not None:
    corr = concat_sub.iloc[:, 1:].corr().to_numpy()
    rank = np.tril(corr, -1)
    m = int((rank > 0).sum())
    if m <= 0:
        m = rank.shape[0]
    steps = min(rank.shape[0], m)

    pairs = np.empty((steps, 2), dtype=np.int64)
    rwork = rank.copy()
    for n in range(steps):
        mx = np.unravel_index(rwork.argmin(), rwork.shape)
        pairs[n, 0] = mx[0]
        pairs[n, 1] = mx[1]
        rwork[mx] = 1.0

    n_arr = np.arange(steps, dtype=np.float64)
    denom = m + n_arr / 10.0
    w = np.where(denom != 0, (m - n_arr) / denom, 1.0)

    preds_mat = concat_sub.iloc[:, 1:].to_numpy(dtype=np.float64, copy=False)
    a = np.clip(preds_mat[:, pairs[:, 0]], 1e-12, 1.0)
    b = np.clip(preds_mat[:, pairs[:, 1]], 1e-12, 1.0)

    acc = (np.log(a) + np.log(b)) * 0.5  # shape (N, steps)
    m_gmean = np.exp((acc * w).sum(axis=1) / w.sum()).clip(0.0, 1.0)
    print("Computed m_gmean from ensemble files.")
else:
    print("Skipping m_gmean computation (no concat_sub).")




## === cell 4
predictions = None
if concat_sub is not None:
    preds_mat = concat_sub.iloc[:, 1:].to_numpy(
        dtype=np.float64, copy=False
    )  # shape (N, K)
    n, k = preds_mat.shape

    order = np.argsort(preds_mat, axis=0, kind="mergesort")
    ranked = np.empty_like(preds_mat, dtype=np.float64)
    rows = np.arange(n, dtype=np.float64)[:, None]
    cols = np.arange(k)[None, :]
    ranked[order, cols] = rows + 1.0
    ranked /= float(n)

    avg_rank = ranked.mean(axis=1).clip(0.0, 1.0)

    if m_gmean is not None and len(m_gmean) == len(avg_rank):
        avg_rank = (0.7 * avg_rank + 0.3 * np.asarray(m_gmean, dtype=np.float64)).clip(
            0.0, 1.0
        )

    predictions = avg_rank.reshape(-1, 1)
    print("Ensemble predictions computed:", predictions.shape)
else:
    print("No ensemble predictions computed in this cell.")




## === cell 5
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

if predictions is None:
    y = train_df["target"].astype(int).values

    for df in (train_df, test_df):
        if "sex" in df.columns:
            df["sex"] = df["sex"].astype("object")
        if "anatom_site_general_challenge" in df.columns:
            df["anatom_site_general_challenge"] = df[
                "anatom_site_general_challenge"
            ].astype("object")
        if "patient_id" in df.columns:
            df["patient_id"] = df["patient_id"].astype("object")

    def _add_flags(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        age = pd.to_numeric(out["age_approx"], errors="coerce")
        out["age_missing"] = age.isna().astype(np.int8)

        sex_raw = out["sex"]
        site_raw = out["anatom_site_general_challenge"]

        sex_stripped = sex_raw.fillna("").astype(str).str.strip()
        site_stripped = site_raw.fillna("").astype(str).str.strip()

        out["sex_missing"] = (sex_raw.isna() | sex_stripped.eq("")).astype(np.int8)
        out["site_missing"] = (site_raw.isna() | site_stripped.eq("")).astype(np.int8)

        sex_lower = sex_stripped.str.lower()
        site_lower = site_stripped.str.lower()
        out["sex_unknown"] = sex_lower.isin(["", "unknown"]).astype(np.int8)
        out["site_unknown"] = site_lower.isin(["", "unknown"]).astype(np.int8)

        out["sex"] = sex_stripped.mask(sex_stripped.eq(""), "__MISSING__")
        out["anatom_site_general_challenge"] = site_stripped.mask(
            site_stripped.eq(""), "__MISSING__"
        )

        pid_stripped = out["patient_id"].fillna("").astype(str).str.strip()
        out["patient_id"] = pid_stripped.mask(pid_stripped.eq(""), "__MISSING__")

        out["age_bin"] = pd.cut(
            age.astype(float),
            bins=[-np.inf, 20, 30, 40, 50, 60, 70, 80, np.inf],
            labels=["<20", "20s", "30s", "40s", "50s", "60s", "70s", "80+"],
        ).astype("object")
        out["age_bin"] = out["age_bin"].fillna("__MISSING__").astype(str)

        out["sex_x_site"] = (
            out["sex"].astype(str)
            + "__"
            + out["anatom_site_general_challenge"].astype(str)
        ).astype("object")
        out["sex_x_agebin"] = (
            out["sex"].astype(str) + "__" + out["age_bin"].astype(str)
        ).astype("object")

        return out

    train_fe = _add_flags(train_df)
    test_fe = _add_flags(test_df)

    feature_cols_num = [
        "age_approx",
        "age_missing",
        "sex_missing",
        "site_missing",
        "sex_unknown",
        "site_unknown",
    ]
    feature_cols_cat = [
        "sex",
        "anatom_site_general_challenge",
        "patient_id",
        "age_bin",
        "sex_x_site",
        "sex_x_agebin",
    ]

    X_train = train_fe[feature_cols_num + feature_cols_cat]
    X_test = test_fe[feature_cols_num + feature_cols_cat]

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, feature_cols_num),
            ("cat", categorical_transformer, feature_cols_cat),
        ],
        remainder="drop",
    )

    groups = train_df["patient_id"].astype(str).values
    try:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        split_iter = cv.split(X_train, y, groups=groups)  # avoid materializing list
        print("Using StratifiedGroupKFold with patient_id groups.")
    except Exception as e:
        print(
            "Could not use StratifiedGroupKFold; falling back to StratifiedKFold. Reason:",
            repr(e),
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        split_iter = cv.split(X_train, y)

    oof = np.zeros(len(X_train), dtype=float)
    test_pred = np.zeros(len(X_test), dtype=float)

    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    pos_w = (n_neg / n_pos) if n_pos > 0 else 1.0
    base_weights = np.where(y == 1, pos_w, 1.0).astype(float)

    n_splits = cv.get_n_splits()

    X_all = pd.concat([X_train, X_test], axis=0, ignore_index=True)
    X_all_trans = preprocessor.fit_transform(X_all)
    n_tr = len(X_train)
    X_tr_all = X_all_trans[:n_tr]
    X_te_all = X_all_trans[n_tr:]

    for fold, (tr_idx, va_idx) in enumerate(split_iter, 1):
        X_tr = X_tr_all[tr_idx]
        y_tr = y[tr_idx]
        X_va = X_tr_all[va_idx]
        y_va = y[va_idx]

        clf = LogisticRegression(
            solver="saga",
            penalty="elasticnet",
            l1_ratio=0.2,
            max_iter=1200,
            n_jobs=-1,
            C=2.0,
            random_state=42,
        )

        clf.fit(X_tr, y_tr, sample_weight=base_weights[tr_idx])

        oof[va_idx] = clf.predict_proba(X_va)[:, 1]
        test_pred += clf.predict_proba(X_te_all)[:, 1] / n_splits

    try:
        auc = roc_auc_score(y, oof)
        print(f"OOF AUC (metadata fallback, grouped if available): {auc:.6f}")
    except Exception as e:
        print("Could not compute OOF AUC:", repr(e))

    predictions = test_pred.reshape(-1, 1)
    print("Fallback metadata model predictions computed:", predictions.shape)
else:
    print("Using ensemble predictions; fallback model not used.")




## === cell 6
submission = sample_sub.copy()
submission["target"] = np.asarray(predictions).reshape(-1).astype(float)
submission["target"] = submission["target"].clip(0.0, 1.0)
submission = submission[["image_name", "target"]]

out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path, "shape:", submission.shape)
print(submission.head())
