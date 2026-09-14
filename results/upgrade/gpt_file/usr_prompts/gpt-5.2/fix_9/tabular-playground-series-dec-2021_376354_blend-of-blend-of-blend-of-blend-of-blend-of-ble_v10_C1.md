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
Predict the class of a given image from a synthetic dataset.

## MetricMulti-class classification accuracy.

## Submission FormatFor each `Id` in the test set, you must predict the `Cover_Type` class. The file should contain a header and have the following format:
```
Id,Cover_Type
4000000,2
4000001,1
4000001,3
etc.
```

## Dataset 
- train.csv - the training data with the target `Cover_Type` column
- test.csv - the test set; you will be predicting the `Cover_Type` for each row in this file (the target integer class)
- sample_submission.csv - a sample submission file in the correct format

# 2. Python version

3.10

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
            description.md (59 lines)
            sample_submission.csv (400001 lines)
            sample_submission.csv.zip (1.6 MB)
            test.csv (400001 lines)
            test.csv.zip (10.7 MB)
            train.csv (3600001 lines)
            train.csv.zip (97.9 MB)
            tabular-playground-series-dec-2021/
                description.md (59 lines)
                sample_submission.csv (400001 lines)
                ... and 5 other files
                tabular-playground-series-dec-2021/
        input/
            description.md (59 lines)
            sample_submission.csv (400001 lines)
            sample_submission.csv.zip (1.6 MB)
            test.csv (400001 lines)
            test.csv.zip (10.7 MB)
            train.csv (3600001 lines)
            train.csv.zip (97.9 MB)
            tabular-playground-series-dec-2021/
                description.md (59 lines)
                sample_submission.csv (400001 lines)
                ... and 5 other files
                tabular-playground-series-dec-2021/
        working/
            tabular-playground-series-dec-2021/
                description.md (59 lines)
                sample_submission.csv (400001 lines)
                ... and 5 other files
                tabular-playground-series-dec-2021/
```

-> data/sample_submission.csv has 400000 rows and 2 columns.
The columns are: Id, Cover_Type

-> data/tabular-playground-series-dec-2021/sample_submission.csv has 400000 rows and 2 columns.
The columns are: Id, Cover_Type

-> data/tabular-playground-series-dec-2021/test.csv has 400000 rows and 55 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 40 more columns

-> data/tabular-playground-series-dec-2021/train.csv has 3600000 rows and 56 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 41 more columns

-> data/test.csv has 400000 rows and 55 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 40 more columns

-> data/train.csv has 3600000 rows and 56 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 41 more columns

-> input/sample_submission.csv has 400000 rows and 2 columns.
The columns are: Id, Cover_Type

-> (stopped after 10 files for performance)

# 5. Target score

0.9565742857142856

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from scipy import stats  # noqa: F401

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42

DATA_DIR = "/kaggle/input/tabular-playground-series-dec-2021"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

assert os.path.exists(TRAIN_PATH), f"Missing {TRAIN_PATH}"
assert os.path.exists(TEST_PATH), f"Missing {TEST_PATH}"
assert os.path.exists(SAMPLE_SUB_PATH), f"Missing {SAMPLE_SUB_PATH}"

_nthreads = str(os.cpu_count() or 4)
os.environ.setdefault("OMP_NUM_THREADS", _nthreads)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _nthreads)
os.environ.setdefault("MKL_NUM_THREADS", _nthreads)
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", _nthreads)
os.environ.setdefault("NUMEXPR_NUM_THREADS", _nthreads)




## === cell 1
try:
    import pyarrow  # noqa: F401

    _CSV_ENGINE = "pyarrow"
except Exception:
    _CSV_ENGINE = "c"

_train_cols = pd.read_csv(TRAIN_PATH, nrows=0).columns.tolist()
_test_cols = pd.read_csv(TEST_PATH, nrows=0).columns.tolist()

target_col = "Cover_Type"
id_col = "Id"
feature_cols = [c for c in _train_cols if c not in (id_col, target_col)]

train_dtypes = {c: np.int32 for c in feature_cols}
train_dtypes[id_col] = np.int32
train_dtypes[target_col] = np.int8

test_dtypes = {c: np.int32 for c in _test_cols if c != id_col}
test_dtypes[id_col] = np.int32

train_usecols = [id_col] + feature_cols + [target_col]
test_usecols = [id_col] + feature_cols

train = pd.read_csv(
    TRAIN_PATH, dtype=train_dtypes, engine=_CSV_ENGINE, usecols=train_usecols
)
test = pd.read_csv(
    TEST_PATH, dtype=test_dtypes, engine=_CSV_ENGINE, usecols=test_usecols
)
submission = pd.read_csv(SAMPLE_SUB_PATH)

y = train[target_col].to_numpy(dtype=np.int32, copy=False)

X_tr = train[feature_cols]
y_tr = y

X_tr_np = np.ascontiguousarray(X_tr.to_numpy(dtype=np.float32, copy=False))
y_tr_np = np.asarray(y_tr, dtype=np.int32)

X_test_np = np.ascontiguousarray(
    test[feature_cols].to_numpy(dtype=np.float32, copy=False)
)
test_id_np = test[id_col].to_numpy(copy=False)




## === cell 2
import hashlib
import joblib

CACHE_DIR = "/kaggle/working/model_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def _fingerprint_array(a: np.ndarray, nbytes: int = 1_000_000) -> str:
    """Deterministic fingerprint from shape/dtype + first/last bytes (fast)."""
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(str(a.dtype).encode())
    view = a.view(np.uint8)
    if view.size <= nbytes:
        h.update(view.tobytes())
    else:
        half = nbytes // 2
        h.update(view[:half].tobytes())
        h.update(view[-half:].tobytes())
    return h.hexdigest()


def _memmap_cache(path: str, arr: np.ndarray, mode: str = "r+") -> np.memmap:
    """
    Cache a numpy array to disk as a .npy and return a memmap view.
    Correctness-preserving: data are identical; avoids extra RAM copies.
    """
    if not os.path.exists(path):
        tmp = path + ".tmp"
        np.save(tmp, np.ascontiguousarray(arr))
        os.replace(tmp, path)
    return np.load(path, mmap_mode=mode)


cfg = {
    "random_state": RANDOM_STATE,
    "lr_saga": dict(
        solver="saga", multi_class="multinomial", max_iter=200, n_jobs=-1, C=2.0
    ),
    "lr_lbfgs": dict(
        solver="lbfgs", multi_class="multinomial", max_iter=200, n_jobs=-1, C=1.0
    ),
    "gnb": dict(),
    "rf": dict(
        n_estimators=120,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
    ),
    "scaler": dict(with_mean=False),
    "X_tr_fp": _fingerprint_array(X_tr_np),
    "y_tr_fp": _fingerprint_array(y_tr_np),
    "X_test_fp": _fingerprint_array(X_test_np),
}
cfg_fp = hashlib.sha256(repr(sorted(cfg.items())).encode()).hexdigest()[:16]

Xtr_mm_path = os.path.join(CACHE_DIR, f"Xtr_{cfg_fp}.npy")
Xte_mm_path = os.path.join(CACHE_DIR, f"Xte_{cfg_fp}.npy")

X_tr_np = _memmap_cache(Xtr_mm_path, X_tr_np, mode="r")
X_test_np = _memmap_cache(Xte_mm_path, X_test_np, mode="r")

scaler_path = os.path.join(CACHE_DIR, f"scaler_{cfg_fp}.joblib")
lr_saga_path = os.path.join(CACHE_DIR, f"lr_saga_{cfg_fp}.joblib")
lr_lbfgs_path = os.path.join(CACHE_DIR, f"lr_lbfgs_{cfg_fp}.joblib")
gnb_path = os.path.join(CACHE_DIR, f"gnb_{cfg_fp}.joblib")
rf_path = os.path.join(CACHE_DIR, f"rf_{cfg_fp}.joblib")

scaler = StandardScaler(with_mean=False, copy=False)
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
    X_tr_scaled = scaler.transform(X_tr_np)
    X_test_scaled = scaler.transform(X_test_np)
else:
    X_tr_scaled = scaler.fit_transform(X_tr_np)
    X_test_scaled = scaler.transform(X_test_np)
    joblib.dump(scaler, scaler_path, compress=3)

lr_saga = LogisticRegression(
    solver="saga",
    multi_class="multinomial",
    max_iter=200,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    C=2.0,
    verbose=0,
)
lr_lbfgs = LogisticRegression(
    solver="lbfgs",
    multi_class="multinomial",
    max_iter=200,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    C=1.0,
    verbose=0,
)
gnb = GaussianNB()

rf = RandomForestClassifier(
    n_estimators=120,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

pred_matrix = []

if os.path.exists(lr_saga_path):
    lr_saga = joblib.load(lr_saga_path)
else:
    lr_saga.fit(X_tr_scaled, y_tr_np)
    joblib.dump(lr_saga, lr_saga_path, compress=3)
pred_matrix.append(lr_saga.predict(X_test_scaled).astype(np.int32, copy=False))

if os.path.exists(lr_lbfgs_path):
    lr_lbfgs = joblib.load(lr_lbfgs_path)
else:
    lr_lbfgs.fit(X_tr_scaled, y_tr_np)
    joblib.dump(lr_lbfgs, lr_lbfgs_path, compress=3)
pred_matrix.append(lr_lbfgs.predict(X_test_scaled).astype(np.int32, copy=False))

if os.path.exists(gnb_path):
    gnb = joblib.load(gnb_path)
else:
    classes_ = np.unique(y_tr_np)
    batch_size = 200_000
    n = X_tr_scaled.shape[0]
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        Xb = X_tr_scaled[start:end]
        yb = y_tr_np[start:end]
        if start == 0:
            gnb.partial_fit(Xb, yb, classes=classes_)
        else:
            gnb.partial_fit(Xb, yb)
    joblib.dump(gnb, gnb_path, compress=3)
pred_matrix.append(gnb.predict(X_test_scaled).astype(np.int32, copy=False))

if os.path.exists(rf_path):
    rf = joblib.load(rf_path)
else:
    rf.fit(X_tr_np, y_tr_np)
    joblib.dump(rf, rf_path, compress=3)
pred_matrix.append(rf.predict(X_test_np).astype(np.int32, copy=False))

pred_matrix = np.vstack(pred_matrix).T  # shape: (n_test, 4)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/598333924.py in <cell line: 0>()
     64 
     65 # Persist arrays once; then work off memmap to reduce peak memory and avoid extra copies.
---> 66 X_tr_np = _memmap_cache(Xtr_mm_path, X_tr_np, mode="r")
     67 X_test_np = _memmap_cache(Xte_mm_path, X_test_np, mode="r")
     68 

/tmp/ipykernel_11/598333924.py in _memmap_cache(path, arr, mode)
     32         tmp = path + ".tmp"
     33         np.save(tmp, np.ascontiguousarray(arr))
---> 34         os.replace(tmp, path)
     35     return np.load(path, mmap_mode=mode)
     36 

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/model_cache/Xtr_35a11478e6197c6c.npy.tmp' -> '/kaggle/working/model_cache/Xtr_35a11478e6197c6c.npy'

## === cell 3
def row_mode_int_4(arr_2d: np.ndarray) -> np.ndarray:
    a = np.asarray(arr_2d)
    if a.ndim != 2 or a.shape[1] != 4:
        raise ValueError(f"Expected shape (n, 4), got {a.shape}")

    p0, p1, p2, p3 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]

    c0 = (
        (p0 == p0).astype(np.int8)
        + (p1 == p0).astype(np.int8)
        + (p2 == p0).astype(np.int8)
        + (p3 == p0).astype(np.int8)
    )
    c1 = (
        (p0 == p1).astype(np.int8)
        + (p1 == p1).astype(np.int8)
        + (p2 == p1).astype(np.int8)
        + (p3 == p1).astype(np.int8)
    )
    c2 = (
        (p0 == p2).astype(np.int8)
        + (p1 == p2).astype(np.int8)
        + (p2 == p2).astype(np.int8)
        + (p3 == p2).astype(np.int8)
    )
    c3 = (
        (p0 == p3).astype(np.int8)
        + (p1 == p3).astype(np.int8)
        + (p2 == p3).astype(np.int8)
        + (p3 == p3).astype(np.int8)
    )

    maxc = np.maximum.reduce([c0, c1, c2, c3])
    sentinel = np.iinfo(a.dtype).max
    v0 = np.where(c0 == maxc, p0, sentinel)
    v1 = np.where(c1 == maxc, p1, sentinel)
    v2 = np.where(c2 == maxc, p2, sentinel)
    v3 = np.where(c3 == maxc, p3, sentinel)
    return np.minimum.reduce([v0, v1, v2, v3]).astype(np.int32, copy=False)


ensemble_pred = row_mode_int_4(pred_matrix).astype(int)

out = submission.copy()
out[id_col] = test_id_np  # ensure correct Id alignment (no extra pandas alignment work)
out[target_col] = ensemble_pred

assert list(out.columns) == [
    id_col,
    target_col,
], f"Unexpected submission columns: {out.columns.tolist()}"
assert len(out) == len(test), "Submission row count mismatch with test set"
assert out[target_col].notna().all(), "Found NaNs in predictions"

out.to_csv("submission.csv", index=False)
out.head()

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3540868257.py in <cell line: 0>()
     40 
     41 
---> 42 ensemble_pred = row_mode_int_4(pred_matrix).astype(int)
     43 
     44 out = submission.copy()

NameError: name 'pred_matrix' is not defined
