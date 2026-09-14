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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
xgboost==2.0.3

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
import numpy as np
import pandas as pd

os.environ["PYTHONHASHSEED"] = "42"
np.random.seed(42)
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 4))
os.environ.setdefault("MKL_NUM_THREADS", str(os.cpu_count() or 4))

CANDIDATE_TRAIN = [
    "/kaggle/input/tabular-playground-series-dec-2021/train.csv",
    "/kaggle/input/train.csv",
    "/kaggle/data/tabular-playground-series-dec-2021/train.csv",
    "/kaggle/data/train.csv",
]
CANDIDATE_TEST = [
    "/kaggle/input/tabular-playground-series-dec-2021/test.csv",
    "/kaggle/input/test.csv",
    "/kaggle/data/tabular-playground-series-dec-2021/test.csv",
    "/kaggle/data/test.csv",
]


def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


train_path = first_existing(CANDIDATE_TRAIN)
test_path = first_existing(CANDIDATE_TEST)

if train_path is None or test_path is None:
    raise FileNotFoundError(
        f"Could not find train/test CSV. Checked:\ntrain: {CANDIDATE_TRAIN}\n"
        f"test: {CANDIDATE_TEST}"
    )

train_cols = pd.read_csv(train_path, nrows=0).columns.tolist()
test_cols = pd.read_csv(test_path, nrows=0).columns.tolist()

FEATURES = [c for c in train_cols if c not in ("Cover_Type", "Id")]

dtype_train = {c: np.float32 for c in FEATURES}
dtype_train.update({"Id": np.int64, "Cover_Type": np.int64})
dtype_test = {c: np.float32 for c in FEATURES}
dtype_test.update({"Id": np.int64})

usecols_train = ["Id"] + FEATURES + ["Cover_Type"]
usecols_test = ["Id"] + FEATURES

train_df = pd.read_csv(train_path, usecols=usecols_train, dtype=dtype_train)
test_df = pd.read_csv(test_path, usecols=usecols_test, dtype=dtype_test)

X = np.ascontiguousarray(train_df[FEATURES].to_numpy(copy=False), dtype=np.float32)
y = train_df["Cover_Type"].to_numpy(copy=False).astype(np.int64, copy=False)

y0 = y - 1  # map 1..7 -> 0..6
X_test = np.ascontiguousarray(test_df[FEATURES].to_numpy(copy=False), dtype=np.float32)

print("Using paths:", train_path, test_path)
print("Shapes:", X.shape, y0.shape, X_test.shape)



## === cell 1
vc = np.bincount(y0, minlength=7)
can_stratify = vc.min() >= 2
n = y0.shape[0]
val_size = int(np.ceil(0.01 * n))

rng = np.random.RandomState(42)

if can_stratify:
    per_class_val = np.floor(vc * 0.01).astype(int)
    per_class_val = np.maximum(per_class_val, 1)
    diff = int(val_size - per_class_val.sum())
    if diff != 0:
        frac = (vc * 0.01) - np.floor(vc * 0.01)
        order = np.argsort(-frac)  # descending remainder
        if diff > 0:
            for k in order[:diff]:
                per_class_val[k] += 1
        else:
            for k in order[::-1]:
                if diff == 0:
                    break
                if per_class_val[k] > 1:
                    per_class_val[k] -= 1
                    diff += 1

    all_idx = np.arange(n, dtype=np.int64)
    order = np.argsort(y0, kind="mergesort")  # stable, deterministic
    all_idx = all_idx[order]
    bounds = np.concatenate(([0], np.cumsum(vc))).astype(np.int64, copy=False)

    val_idx_parts = []
    for c in range(7):
        start, end = int(bounds[c]), int(bounds[c + 1])
        idx_c = all_idx[start:end].copy()
        rng.shuffle(idx_c)
        val_idx_parts.append(idx_c[: int(per_class_val[c])])
    val_idx = np.concatenate(val_idx_parts)
    rng.shuffle(val_idx)
else:
    print(
        "Warning: Cannot stratify split because min class count is",
        int(vc.min()),
        "- using non-stratified split.",
    )
    val_idx = rng.choice(n, size=val_size, replace=False)

val_mask = np.zeros(n, dtype=bool)
val_mask[val_idx] = True
train_idx = np.flatnonzero(~val_mask)
val_idx = np.flatnonzero(val_mask)

y_train = y0[train_idx]
y_val = y0[val_idx]

print(
    "Split shapes:",
    (train_idx.size, X.shape[1]),
    (val_idx.size, X.shape[1]),
    y_train.shape,
    y_val.shape,
    "n_classes_train:",
    np.unique(y_train).size,
    "n_classes_val:",
    np.unique(y_val).size,
)



## === cell 2
import xgboost as xgb

common_params = dict(
    eta=0.01,  # learning_rate
    objective="multi:softprob",
    num_class=7,
    eval_metric="mlogloss",
    seed=42,
)

num_boost_round = 20000  # preserve core training approach

dtrain = None
dval = None
bst = None
used_gpu = False

try:
    ytr = np.ascontiguousarray(y0[train_idx], dtype=np.int64)
    yva = np.ascontiguousarray(y0[val_idx], dtype=np.int64)

    params = dict(
        common_params,
        tree_method="gpu_hist",
        predictor="gpu_predictor",
        gpu_id=0,
        device="cuda",
        max_bin=256,
    )

    try:
        dtrain = xgb.QuantileDMatrix(
            X,
            label=ytr,
            max_bin=256,
            enable_categorical=False,
            indices=train_idx,
        )
        dval = xgb.QuantileDMatrix(
            X,
            label=yva,
            max_bin=256,
            enable_categorical=False,
            ref=dtrain,
            indices=val_idx,
        )
    except TypeError:
        Xtr = X[train_idx]
        Xva = X[val_idx]
        dtrain = xgb.QuantileDMatrix(
            Xtr, label=ytr, max_bin=256, enable_categorical=False
        )
        dval = xgb.QuantileDMatrix(
            Xva, label=yva, max_bin=256, enable_categorical=False, ref=dtrain
        )

    bst = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dval, "val")],
        verbose_eval=False,
    )
    used_gpu = True
except Exception as e:
    print("GPU training not available or failed; falling back to CPU. Error:", repr(e))

    try:
        from sklearnex import patch_sklearn

        patch_sklearn()
    except Exception:
        pass

    params = dict(
        common_params,
        tree_method="hist",
        predictor="cpu_predictor",
        nthread=(os.cpu_count() or 4),
        max_bin=256,
    )

    try:
        ytr = np.ascontiguousarray(y0[train_idx], dtype=np.int64)
        yva = np.ascontiguousarray(y0[val_idx], dtype=np.int64)
        dtrain = xgb.DMatrix(
            X,
            label=ytr,
            nthread=params["nthread"],
            enable_categorical=False,
            feature_names=None,
            indices=train_idx,
        )
        dval = xgb.DMatrix(
            X,
            label=yva,
            nthread=params["nthread"],
            enable_categorical=False,
            feature_names=None,
            indices=val_idx,
        )
    except TypeError:
        dtrain = xgb.DMatrix(
            X[train_idx], label=y0[train_idx], nthread=params["nthread"]
        )
        dval = xgb.DMatrix(X[val_idx], label=y0[val_idx], nthread=params["nthread"])

    bst = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dval, "val")],
        verbose_eval=False,
    )
    used_gpu = False

print("Training done. used_gpu =", used_gpu)



## === cell 3
if used_gpu:
    X_test_c = X_test  # already contiguous float32
    try:
        dtest = xgb.QuantileDMatrix(
            X_test_c, max_bin=256, enable_categorical=False, ref=dtrain
        )
        proba = bst.predict(dtest)
    except Exception:
        proba = bst.inplace_predict(X_test_c)
else:
    try:
        proba = bst.inplace_predict(X_test)
    except Exception:
        dtest = xgb.DMatrix(X_test, nthread=(os.cpu_count() or 4))
        proba = bst.predict(dtest)

y_pred0 = np.asarray(proba).reshape(-1, 7).argmax(axis=1).astype(np.int64, copy=False)
y_predict_xgbc = (y_pred0 + 1).astype(np.int64, copy=False)

print(
    "Pred summary:",
    "min=",
    int(y_predict_xgbc.min()),
    "max=",
    int(y_predict_xgbc.max()),
    "n=",
    len(y_predict_xgbc),
)



## === cell 4
result = pd.DataFrame(
    {"Id": test_df["Id"].to_numpy(copy=False), "Cover_Type": y_predict_xgbc}
)
print(result.head())

assert list(result.columns) == ["Id", "Cover_Type"]
assert result.shape[0] == test_df.shape[0]
assert result["Cover_Type"].between(1, 7).all()

result.to_csv("/kaggle/working/submission.csv", index=False)
print("Done. Wrote /kaggle/working/submission.csv with shape:", result.shape)
