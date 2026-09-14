# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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

# 5. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU only (unused here but keep)
_n_threads = str(os.cpu_count() or 4)
os.environ.setdefault("OMP_NUM_THREADS", _n_threads)
os.environ.setdefault("MKL_NUM_THREADS", _n_threads)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _n_threads)
os.environ.setdefault("NUMEXPR_NUM_THREADS", _n_threads)

import gc
import numpy as np
import pandas as pd

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

from sklearn.ensemble import HistGradientBoostingClassifier

try:
    from threadpoolctl import threadpool_limits

    _threadpool_ctx = threadpool_limits(limits=int(_n_threads))
except Exception:
    _threadpool_ctx = None

np.random.seed(42)



## === cell 1
TRAIN_PATH = "../input/tabular-playground-series-dec-2021/train.csv"
TEST_PATH = "../input/tabular-playground-series-dec-2021/test.csv"

_cols_train = pd.read_csv(TRAIN_PATH, nrows=0).columns.tolist()
_cols_test = pd.read_csv(TEST_PATH, nrows=0).columns.tolist()

_drop_cols = [c for c in ("Soil_Type7", "Soil_Type15") if c in _cols_train]


def _build_dtype_map(cols, has_target: bool):
    dtypes = {}
    for c in cols:
        if c == "Id":
            dtypes[c] = np.int32
        elif c == "Cover_Type" and has_target:
            dtypes[c] = np.int16
        elif c.startswith("Wilderness_Area") or c.startswith("Soil_Type"):
            dtypes[c] = np.uint8
        elif c.startswith("Hillshade_"):
            dtypes[c] = np.uint8
        else:
            dtypes[c] = np.int32
    return dtypes


_usecols_train = [c for c in _cols_train if c not in _drop_cols]
_usecols_test = [c for c in _cols_test if c not in _drop_cols]

_read_kwargs_train = dict(
    usecols=_usecols_train,
    dtype=_build_dtype_map(_usecols_train, has_target=True),
    memory_map=True,
)
_read_kwargs_test = dict(
    usecols=_usecols_test,
    dtype=_build_dtype_map(_usecols_test, has_target=False),
    memory_map=True,
)

try:
    train = pd.read_csv(TRAIN_PATH, engine="pyarrow", **_read_kwargs_train)
    test = pd.read_csv(TEST_PATH, engine="pyarrow", **_read_kwargs_test)
except Exception:
    train = pd.read_csv(TRAIN_PATH, **_read_kwargs_train)
    test = pd.read_csv(TEST_PATH, **_read_kwargs_test)




## === cell 2
def _build_features_numpy(df: pd.DataFrame, feature_cols_in_order: list[str]):
    base_cols = [c for c in feature_cols_in_order if c not in ("Aspect",)]
    X_base = df.loc[:, base_cols].to_numpy(dtype=np.float32, copy=False)

    aspect = df["Aspect"].to_numpy(dtype=np.float32, copy=False)
    rad = aspect * (np.float32(np.pi) / np.float32(180.0))
    aspect_cos = np.cos(rad).astype(np.float32, copy=False)
    aspect_sin = np.sin(rad).astype(np.float32, copy=False)

    for col in ("Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"):
        if col in df.columns:
            arr = df[col].to_numpy(copy=False)
            np.clip(arr, 0, 255, out=arr)

    h = df["Horizontal_Distance_To_Hydrology"].to_numpy(dtype=np.float32, copy=False)
    v = df["Vertical_Distance_To_Hydrology"].to_numpy(dtype=np.float32, copy=False)
    ah = np.abs(h)
    av = np.abs(v)
    sum_hydro = (ah + av).astype(np.float32, copy=False)
    sub_hydro = (ah - av).astype(np.float32, copy=False)

    elev = df["Elevation"].to_numpy(dtype=np.float32, copy=False)
    road = df["Horizontal_Distance_To_Roadways"].to_numpy(dtype=np.float32, copy=False)
    fire = df["Horizontal_Distance_To_Fire_Points"].to_numpy(
        dtype=np.float32, copy=False
    )
    hs3 = df["Hillshade_3pm"].to_numpy(copy=False)

    ehi = (road * elev).astype(np.float32, copy=False)
    evi = (v * elev).astype(np.float32, copy=False)
    highwater = (v < 0).astype(np.int8, copy=False)
    evdth = (elev - v).astype(np.float32, copy=False)
    ehdth = (elev - h * np.float32(0.2)).astype(np.float32, copy=False)
    euclid = np.sqrt(h * h + v * v).astype(np.float32, copy=False)
    manhat = (h + v).astype(np.float32, copy=False)
    hydro_fire_1 = (h + fire).astype(np.float32, copy=False)
    hydro_fire_2 = np.abs(h - fire).astype(np.float32, copy=False)
    hydro_road_1 = np.abs(h + road).astype(np.float32, copy=False)
    hydro_road_2 = np.abs(h - road).astype(np.float32, copy=False)
    fire_road_1 = np.abs(fire + road).astype(np.float32, copy=False)
    fire_road_2 = np.abs(fire - road).astype(np.float32, copy=False)
    hs3_zero = (hs3 == 0).astype(np.int8, copy=False)

    extra_feats = [
        aspect_cos,
        aspect_sin,
        sum_hydro,
        sub_hydro,
        ehi,
        evi,
        highwater.astype(np.float32, copy=False),
        evdth,
        ehdth,
        euclid,
        manhat,
        hydro_fire_1,
        hydro_fire_2,
        hydro_road_1,
        hydro_road_2,
        fire_road_1,
        fire_road_2,
        hs3_zero.astype(np.float32, copy=False),
    ]
    n = X_base.shape[0]
    m = X_base.shape[1] + len(extra_feats)
    X_out = np.empty((n, m), dtype=np.float32)
    X_out[:, : X_base.shape[1]] = X_base

    j = X_base.shape[1]
    for feat in extra_feats:
        X_out[:, j] = feat
        j += 1

    out_cols = base_cols + [
        "Aspect_cos",
        "Aspect_sin",
        "Sum_Hydrology",
        "Sub_Hydrology",
        "EHiElv",
        "EViElv",
        "Highwater",
        "EVDtH",
        "EHDtH",
        "Euclidean_Distance_to_Hydrolody",
        "Manhattan_Distance_to_Hydrolody",
        "Hydro_Fire_1",
        "Hydro_Fire_2",
        "Hydro_Road_1",
        "Hydro_Road_2",
        "Fire_Road_1",
        "Fire_Road_2",
        "Hillshade_3pm_is_zero",
    ]
    return X_out, out_cols




## === cell 3
_drop_cols_runtime = [c for c in ("Soil_Type7", "Soil_Type15") if c in train.columns]
if _drop_cols_runtime:
    train.drop(columns=_drop_cols_runtime, inplace=True)
    test.drop(
        columns=[c for c in _drop_cols_runtime if c in test.columns], inplace=True
    )



## === cell 4
train["Cover_Type"] = train["Cover_Type"].astype(np.int32)

assert "Id" in test.columns and "Id" in train.columns
assert "Cover_Type" in train.columns and "Cover_Type" not in test.columns

y_full = train["Cover_Type"].to_numpy(dtype=np.int32, copy=False)

feature_cols_raw = [c for c in train.columns if c not in ("Cover_Type", "Id")]
test_feature_cols_raw = [c for c in test.columns if c != "Id"]
assert feature_cols_raw == test_feature_cols_raw, "Train/test feature columns mismatch"

if _threadpool_ctx is not None:
    with _threadpool_ctx:
        X_full, feature_cols = _build_features_numpy(train, feature_cols_raw)
        X_test, test_feature_cols = _build_features_numpy(test, test_feature_cols_raw)
else:
    X_full, feature_cols = _build_features_numpy(train, feature_cols_raw)
    X_test, test_feature_cols = _build_features_numpy(test, test_feature_cols_raw)

assert (
    feature_cols == test_feature_cols
), "Train/test engineered feature columns mismatch"

test_ids = test["Id"].to_numpy(copy=False)

mask = y_full != 5
X = np.ascontiguousarray(X_full[mask])
y = y_full[mask]
del X_full, y_full, mask, train, test
gc.collect()

classes_sorted = np.sort(np.unique(y))
y_idx = np.searchsorted(classes_sorted, y).astype(np.int32, copy=False)
del y
gc.collect()



## === cell 5
try:
    _loss = "log_loss"
    _ = HistGradientBoostingClassifier(loss=_loss)
except TypeError:
    _loss = "auto"

model = HistGradientBoostingClassifier(
    loss=_loss,
    learning_rate=0.05,
    max_depth=8,
    max_iter=1200,
    min_samples_leaf=5,
    max_leaf_nodes=2**8,  # aligns with depth constraint
    l2_regularization=0.0,
    max_bins=255,
    early_stopping=False,  # do not relax convergence criteria
    random_state=42,
    verbose=0,
)

if _threadpool_ctx is not None:
    with _threadpool_ctx:
        model.fit(X, y_idx)
        pred_idx = model.predict(X_test).astype(np.int32, copy=False)
else:
    model.fit(X, y_idx)
    pred_idx = model.predict(X_test).astype(np.int32, copy=False)

pred_labels = classes_sorted[pred_idx].astype(np.int32, copy=False)

sub = pd.DataFrame({"Id": test_ids, "Cover_Type": pred_labels})
sub.to_csv("submission.csv", index=False)

sub.head()
