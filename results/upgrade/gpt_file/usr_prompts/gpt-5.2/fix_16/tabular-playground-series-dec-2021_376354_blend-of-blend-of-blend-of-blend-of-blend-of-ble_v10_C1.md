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

RANDOM_STATE = 42

_nthreads = str(os.cpu_count() or 4)
os.environ.setdefault("OMP_NUM_THREADS", _nthreads)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _nthreads)
os.environ.setdefault("MKL_NUM_THREADS", _nthreads)
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", _nthreads)
os.environ.setdefault("NUMEXPR_NUM_THREADS", _nthreads)

import numpy as np
import pandas as pd
from scipy import stats  # noqa: F401

from sklearn.model_selection import train_test_split  # noqa: F401
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

DATA_DIR = "/kaggle/input/tabular-playground-series-dec-2021"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

assert os.path.exists(TRAIN_PATH), f"Missing {TRAIN_PATH}"
assert os.path.exists(TEST_PATH), f"Missing {TEST_PATH}"
assert os.path.exists(SAMPLE_SUB_PATH), f"Missing {SAMPLE_SUB_PATH}"

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

try:
    train = pd.read_csv(
        TRAIN_PATH, dtype=train_dtypes, engine=_CSV_ENGINE, usecols=train_usecols
    )
    test = pd.read_csv(
        TEST_PATH, dtype=test_dtypes, engine=_CSV_ENGINE, usecols=test_usecols
    )
except Exception:
    train = pd.read_csv(
        TRAIN_PATH, dtype=train_dtypes, engine="c", usecols=train_usecols
    )
    test = pd.read_csv(TEST_PATH, dtype=test_dtypes, engine="c", usecols=test_usecols)

submission = pd.read_csv(SAMPLE_SUB_PATH)

y_tr_np = train[target_col].to_numpy(dtype=np.int32, copy=False)

X_tr_np = np.asarray(
    train[feature_cols].to_numpy(copy=False), dtype=np.float32, order="C"
)
X_test_np = np.asarray(
    test[feature_cols].to_numpy(copy=False), dtype=np.float32, order="C"
)
test_id_np = test[id_col].to_numpy(copy=False)



## === cell 1
import hashlib
import joblib
import time
import multiprocessing as mp

CACHE_DIR = "/kaggle/working/model_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def _fast_cache_key() -> str:
    st_tr = os.stat(TRAIN_PATH)
    st_te = os.stat(TEST_PATH)
    payload = (
        "v4",  # bump cache version because we changed caching/memmap mechanics
        tuple(feature_cols),
        int(train.shape[0]),
        int(test.shape[0]),
        int(st_tr.st_size),
        int(st_te.st_size),
        int(st_tr.st_mtime_ns),
        int(st_te.st_mtime_ns),
        int(RANDOM_STATE),
    )
    return hashlib.sha256(repr(payload).encode()).hexdigest()[:16]


cfg_fp = _fast_cache_key()

scaler_path = os.path.join(CACHE_DIR, f"scaler_{cfg_fp}.joblib")
lr_saga_path = os.path.join(CACHE_DIR, f"lr_saga_{cfg_fp}.joblib")
lr_lbfgs_path = os.path.join(CACHE_DIR, f"lr_lbfgs_{cfg_fp}.joblib")
gnb_path = os.path.join(CACHE_DIR, f"gnb_{cfg_fp}.joblib")
rf_path = os.path.join(CACHE_DIR, f"rf_{cfg_fp}.joblib")

Xtr_scaled_path = os.path.join(CACHE_DIR, f"Xtr_scaled_{cfg_fp}.npy")
Xte_scaled_path = os.path.join(CACHE_DIR, f"Xte_scaled_{cfg_fp}.npy")

lr_saga_pred_path = os.path.join(CACHE_DIR, f"lr_saga_pred_{cfg_fp}.npy")
lr_lbfgs_pred_path = os.path.join(CACHE_DIR, f"lr_lbfgs_pred_{cfg_fp}.npy")
gnb_pred_path = os.path.join(CACHE_DIR, f"gnb_pred_{cfg_fp}.npy")
rf_pred_path = os.path.join(CACHE_DIR, f"rf_pred_{cfg_fp}.npy")


def _atomic_npy_save(path: str, arr: np.ndarray) -> None:
    tmp = path + ".tmp"
    np.save(tmp, arr)
    os.replace(tmp, path)


def _get_scaled_matrices():
    need_scaled = (
        (not os.path.exists(lr_saga_pred_path))
        or (not os.path.exists(lr_lbfgs_pred_path))
        or (not os.path.exists(gnb_pred_path))
    )
    if not need_scaled:
        return None, None

    if (
        os.path.exists(scaler_path)
        and os.path.exists(Xtr_scaled_path)
        and os.path.exists(Xte_scaled_path)
    ):
        scaler = joblib.load(scaler_path)
        X_tr_scaled = np.load(Xtr_scaled_path, mmap_mode="r")
        X_test_scaled = np.load(Xte_scaled_path, mmap_mode="r")
        return X_tr_scaled, X_test_scaled

    scaler = StandardScaler(with_mean=False, copy=True)
    X_tr_scaled_tmp = scaler.fit_transform(X_tr_np)
    X_test_scaled_tmp = scaler.transform(X_test_np)

    joblib.dump(scaler, scaler_path, compress=3)
    _atomic_npy_save(
        Xtr_scaled_path, np.ascontiguousarray(X_tr_scaled_tmp, dtype=np.float32)
    )
    _atomic_npy_save(
        Xte_scaled_path, np.ascontiguousarray(X_test_scaled_tmp, dtype=np.float32)
    )

    X_tr_scaled = np.load(Xtr_scaled_path, mmap_mode="r")
    X_test_scaled = np.load(Xte_scaled_path, mmap_mode="r")
    return X_tr_scaled, X_test_scaled


def _fit_predict_lr(
    kind: str,
    model_path: str,
    pred_path: str,
    Xtr_path: str,
    Xte_path: str,
    y: np.ndarray,
):
    import joblib
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    X_tr_scaled = np.load(Xtr_path, mmap_mode="r")
    X_test_scaled = np.load(Xte_path, mmap_mode="r")

    if kind == "saga":
        model = LogisticRegression(
            solver="saga",
            multi_class="multinomial",
            max_iter=200,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            C=2.0,
            verbose=0,
        )
    elif kind == "lbfgs":
        model = LogisticRegression(
            solver="lbfgs",
            multi_class="multinomial",
            max_iter=200,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            C=1.0,
            verbose=0,
        )
    else:
        raise ValueError(kind)

    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        model.fit(X_tr_scaled, y)
        joblib.dump(model, model_path, compress=3)

    p = model.predict(X_test_scaled).astype(np.int32, copy=False)
    tmp = pred_path + ".tmp"
    np.save(tmp, p)
    os.replace(tmp, pred_path)


def _fit_predict_rf(
    model_path: str, pred_path: str, Xtr: np.ndarray, Xte: np.ndarray, y: np.ndarray
):
    import joblib
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(
        n_estimators=120,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    if os.path.exists(model_path):
        rf = joblib.load(model_path)
    else:
        rf.fit(Xtr, y)
        joblib.dump(rf, model_path, compress=3)

    p = rf.predict(Xte).astype(np.int32, copy=False)
    tmp = pred_path + ".tmp"
    np.save(tmp, p)
    os.replace(tmp, pred_path)


all_pred_cached = all(
    os.path.exists(p)
    for p in (lr_saga_pred_path, lr_lbfgs_pred_path, gnb_pred_path, rf_pred_path)
)

if all_pred_cached:
    pred_matrix = np.vstack(
        [
            np.load(lr_saga_pred_path, mmap_mode="r"),
            np.load(lr_lbfgs_pred_path, mmap_mode="r"),
            np.load(gnb_pred_path, mmap_mode="r"),
            np.load(rf_pred_path, mmap_mode="r"),
        ]
    ).T
else:
    X_tr_scaled, X_test_scaled = _get_scaled_matrices()

    if not os.path.exists(gnb_pred_path):
        gnb = GaussianNB()
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
        p = gnb.predict(X_test_scaled).astype(np.int32, copy=False)
        _atomic_npy_save(gnb_pred_path, p)

    jobs = []
    ctx = mp.get_context("spawn")

    if not os.path.exists(lr_saga_pred_path):
        jobs.append(
            ctx.Process(
                target=_fit_predict_lr,
                args=(
                    "saga",
                    lr_saga_path,
                    lr_saga_pred_path,
                    Xtr_scaled_path,
                    Xte_scaled_path,
                    y_tr_np,
                ),
            )
        )
    if not os.path.exists(lr_lbfgs_pred_path):
        jobs.append(
            ctx.Process(
                target=_fit_predict_lr,
                args=(
                    "lbfgs",
                    lr_lbfgs_path,
                    lr_lbfgs_pred_path,
                    Xtr_scaled_path,
                    Xte_scaled_path,
                    y_tr_np,
                ),
            )
        )
    if not os.path.exists(rf_pred_path):
        jobs.append(
            ctx.Process(
                target=_fit_predict_rf,
                args=(rf_path, rf_pred_path, X_tr_np, X_test_np, y_tr_np),
            )
        )

    for p in jobs:
        p.start()
    for p in jobs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Worker process failed with exit code {p.exitcode}")

    pred_matrix = np.vstack(
        [
            np.load(lr_saga_pred_path, mmap_mode="r"),
            np.load(lr_lbfgs_pred_path, mmap_mode="r"),
            np.load(gnb_pred_path, mmap_mode="r"),
            np.load(rf_pred_path, mmap_mode="r"),
        ]
    ).T  # shape: (n_test, 4)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/887033331.py in <cell line: 0>()
    184     ).T
    185 else:
--> 186     X_tr_scaled, X_test_scaled = _get_scaled_matrices()
    187 
    188     # GaussianNB stays in-process but uses the same cached scaled matrices.

/tmp/ipykernel_11/887033331.py in _get_scaled_matrices()
     76 
     77     joblib.dump(scaler, scaler_path, compress=3)
---> 78     _atomic_npy_save(
     79         Xtr_scaled_path, np.ascontiguousarray(X_tr_scaled_tmp, dtype=np.float32)
     80     )

/tmp/ipykernel_11/887033331.py in _atomic_npy_save(path, arr)
     47     tmp = path + ".tmp"
     48     np.save(tmp, arr)
---> 49     os.replace(tmp, path)
     50 
     51 

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/model_cache/Xtr_scaled_af7e6643e6ec02a3.npy.tmp' -> '/kaggle/working/model_cache/Xtr_scaled_af7e6643e6ec02a3.npy'

## === cell 2
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

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3540868257.py in <cell line: 0>()
     40 
     41 
---> 42 ensemble_pred = row_mode_int_4(pred_matrix).astype(int)
     43 
     44 out = submission.copy()

NameError: name 'pred_matrix' is not defined
