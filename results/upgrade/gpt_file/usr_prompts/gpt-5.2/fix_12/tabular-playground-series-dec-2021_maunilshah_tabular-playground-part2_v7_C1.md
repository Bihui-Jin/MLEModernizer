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

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)



## === cell 1
train_path = "/kaggle/input/tabular-playground-series-dec-2021/train.csv"
test_path = "/kaggle/input/tabular-playground-series-dec-2021/test.csv"
sub_path = "/kaggle/input/tabular-playground-series-dec-2021/sample_submission.csv"

train_df_head = pd.read_csv(train_path, nrows=1)
all_cols = train_df_head.columns.tolist()

train_dtypes = {c: np.float32 for c in all_cols if c not in ("Id", "Cover_Type")}
train_dtypes.update({"Id": np.int32, "Cover_Type": np.int16})

test_df_head = pd.read_csv(test_path, nrows=1)
test_cols = test_df_head.columns.tolist()
test_dtypes = {c: np.float32 for c in test_cols if c != "Id"}
test_dtypes.update({"Id": np.int32})

train_df = pd.read_csv(train_path, dtype=train_dtypes, low_memory=False)
test_df = pd.read_csv(test_path, dtype=test_dtypes, low_memory=False)
sample_sub = pd.read_csv(
    sub_path, dtype={"Id": np.int32, "Cover_Type": np.int16}, low_memory=False
)

print(train_df.shape, test_df.shape, sample_sub.shape)
print(
    "train cols:", train_df.columns[:5].tolist(), "...", train_df.columns[-3:].tolist()
)
print("test cols:", test_df.columns[:5].tolist(), "...", test_df.columns[-3:].tolist())



## === cell 2
X_df = train_df.drop("Cover_Type", axis=1)
y_raw = train_df["Cover_Type"]

classes_sorted = np.sort(y_raw.unique())
y_cat = pd.Categorical(y_raw, categories=classes_sorted, ordered=True)
y = y_cat.codes.astype(np.int32)
if (y < 0).any():
    raise ValueError(
        "Found unmapped labels in Cover_Type mapping; check label mapping."
    )

feature_cols = [c for c in X_df.columns if c != "Id"]

X_np_all = np.ascontiguousarray(
    X_df[feature_cols].to_numpy(dtype=np.float32, copy=False)
)
X_test_np = np.ascontiguousarray(
    test_df[feature_cols].to_numpy(dtype=np.float32, copy=False)
)

print("Original classes:", classes_sorted)
print("Mapped classes:", np.unique(y))
print("X shape:", X_np_all.shape, "y shape:", y.shape, "X_test shape:", X_test_np.shape)



## === cell 3
try:
    from sklearnex import patch_sklearn

    patch_sklearn()
    print("sklearnex patch enabled")
except Exception as _e:
    print("sklearnex patch not available/enabled:", repr(_e))

from sklearn.model_selection import train_test_split

y_arr = np.asarray(y, dtype=np.int32).ravel()
num_class = int(len(classes_sorted))
counts_all = np.bincount(y_arr, minlength=num_class)
can_stratify = bool((counts_all >= 2).all())

if can_stratify:
    x_train_np, x_val_np, y_train, y_val = train_test_split(
        X_np_all, y_arr, test_size=0.15, random_state=RANDOM_STATE, stratify=y_arr
    )
else:
    print(
        "Stratified split skipped due to small class counts; using non-stratified shuffle split."
    )
    x_train_np, x_val_np, y_train, y_val = train_test_split(
        X_np_all, y_arr, test_size=0.15, random_state=RANDOM_STATE, shuffle=True
    )

print(x_train_np.shape, x_val_np.shape, y_train.shape, y_val.shape)
print(
    "Train class counts min/max:",
    np.bincount(y_train, minlength=num_class).min(),
    np.bincount(y_train, minlength=num_class).max(),
)



## === cell 4
import numpy as np

train_counts = np.bincount(y_train, minlength=num_class)
missing_in_train = np.where(train_counts == 0)[0]

sample_weight = np.ones_like(y_train, dtype=np.float32)
if missing_in_train.size > 0:
    x_train_np = np.array(x_train_np, copy=True, order="C")
    x_val_np = np.array(x_val_np, copy=True, order="C")
    y_train = y_train.copy()
    y_val = y_val.copy()

    moved_val_pos = np.empty(missing_in_train.size, dtype=np.int64)
    moved_rows = []
    for i, cls in enumerate(missing_in_train.tolist()):
        idx = np.where(y_val == cls)[0]
        if idx.size == 0:
            raise ValueError(
                f"Class {cls} is missing from training AND validation split; cannot proceed."
            )
        take_pos = int(idx[0])
        moved_val_pos[i] = take_pos
        moved_rows.append((cls, take_pos))

    x_train_np = np.concatenate([x_train_np, x_val_np[moved_val_pos]], axis=0)
    y_train = np.concatenate([y_train, y_val[moved_val_pos]], axis=0)

    keep_mask = np.ones(len(y_val), dtype=bool)
    keep_mask[moved_val_pos] = False
    x_val_np = x_val_np[keep_mask]
    y_val = y_val[keep_mask]

    sample_weight = np.ones_like(y_train, dtype=np.float32)
    print("Moved 1 sample per missing class from val->train:", moved_rows)

print("Post-fix train class bincount:", np.bincount(y_train, minlength=num_class))



## === cell 5
import xgboost as xgb

default_threads = int(os.cpu_count() or 4)
os.environ.setdefault("OMP_NUM_THREADS", str(default_threads))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(default_threads))
os.environ.setdefault("MKL_NUM_THREADS", str(default_threads))
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(default_threads))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(default_threads))
n_threads = default_threads

tree_method = "hist"
device = "cpu"

print(
    "Using tree_method:",
    tree_method,
    "| device:",
    device,
    "| num_class:",
    num_class,
    "| threads:",
    n_threads,
)

cache_prefix = "/kaggle/working/xgb_cache"


def make_dmatrix(Xn, label=None, weight=None):
    return xgb.DMatrix(Xn, label=label, weight=weight, nthread=n_threads)


def make_quantile_or_dmatrix(Xn, label=None, weight=None, max_bin=256):
    try:
        return xgb.QuantileDMatrix(
            Xn,
            label=label,
            weight=weight,
            nthread=n_threads,
            max_bin=max_bin,
            cache_prefix=cache_prefix,
        )
    except TypeError as e:
        print(
            "QuantileDMatrix unavailable/incompatible, falling back to DMatrix. Error:",
            repr(e),
        )
        return make_dmatrix(Xn, label=label, weight=weight)


params = {
    "max_depth": 6,
    "eta": 0.1,
    "objective": "multi:softprob",
    "num_class": num_class,
    "eval_metric": "mlogloss",
    "tree_method": tree_method,
    "device": device,
    "seed": RANDOM_STATE,
    "verbosity": 1,
    "nthread": n_threads,
}

subset_size = 500_000
if x_train_np.shape[0] <= subset_size:
    X_tr_sub, y_tr_sub, w_tr_sub = x_train_np, y_train, sample_weight
else:
    rng = np.random.RandomState(RANDOM_STATE)
    sub_idx = rng.choice(x_train_np.shape[0], size=subset_size, replace=False)
    X_tr_sub = x_train_np[sub_idx]
    y_tr_sub = y_train[sub_idx]
    w_tr_sub = sample_weight[sub_idx]

dtrain_sub = make_quantile_or_dmatrix(X_tr_sub, label=y_tr_sub, weight=w_tr_sub)

cv_hist = xgb.cv(
    params=params,
    dtrain=dtrain_sub,
    num_boost_round=20000,
    nfold=5,
    stratified=True,
    metrics=("mlogloss",),
    early_stopping_rounds=50,
    seed=RANDOM_STATE,
    verbose_eval=200,
)

best_ntree_limit = int(cv_hist.shape[0])
best_iter = best_ntree_limit - 1
print(
    "Selected best_iteration from subset (cv):",
    best_iter,
    "=> num_boost_round:",
    best_ntree_limit,
)

dtrain = make_quantile_or_dmatrix(x_train_np, label=y_train, weight=sample_weight)

booster = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=best_ntree_limit,
    evals=[],
    verbose_eval=False,
)

dtest = make_quantile_or_dmatrix(X_test_np, label=None, weight=None)



## === cell 6
n_rounds = booster.num_boosted_rounds()
proba = booster.predict(dtest, iteration_range=(0, n_rounds))
y_pred_idx = np.asarray(proba).argmax(axis=1).astype(np.int32)

y_pred_label = np.take(classes_sorted, y_pred_idx).astype(np.int32)

print("Pred idx unique (sample):", np.unique(y_pred_idx)[:10], "...")
print("Pred label unique (sample):", np.unique(y_pred_label)[:10], "...")
print("Pred length:", len(y_pred_label), "Expected:", len(test_df))



## === cell 7
result = sample_sub.copy()
if "Id" not in result.columns or "Cover_Type" not in result.columns:
    raise ValueError("sample_submission.csv must contain columns: Id, Cover_Type")

if np.array_equal(result["Id"].values, test_df["Id"].values):
    result["Cover_Type"] = y_pred_label
else:
    pred_df = pd.DataFrame({"Id": test_df["Id"].values, "Cover_Type": y_pred_label})
    result = result.drop(columns=["Cover_Type"]).merge(pred_df, on="Id", how="left")

assert "Cover_Type" in result.columns, "Submission must have a `Cover_Type` column"
assert result.shape[0] == sample_sub.shape[0], "Submission row count mismatch"
assert result["Cover_Type"].isna().sum() == 0, "Some Ids did not get predictions"

print(result.head())
print(result.shape)



## === cell 8
out_path = "/kaggle/working/submission.csv"
result.to_csv(out_path, index=False)
print("Wrote:", out_path)
print("Columns:", result.columns.tolist())
print("Done")
