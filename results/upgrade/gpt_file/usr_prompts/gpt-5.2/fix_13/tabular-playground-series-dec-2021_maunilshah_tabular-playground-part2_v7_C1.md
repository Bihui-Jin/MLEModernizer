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

# 5. Target score

0.9539757142857144

# 6. Current score

0.56458

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.56458) has done: 'We fix the XGBoost training crash by making the target labels contiguous (0..K-1) via a safe mapping, then map predictions back to the original Cover_Type values for submission. We also ensure the model uses the correct multi-class objective/num_class so XGBoost doesn’t mis-infer classes. To make this run on Kaggle regardless of GPU availability, we auto-select `gpu_hist` only if a CUDA device is usable, otherwise fall back to `hist` (score-neutral, prevents runtime failure). Finally, we build the submission from `sample_submission.csv` to guarantee required columns and row alignment, then write a valid `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.56458) has done: 'I fix the split crash by ensuring `y` is a plain 1D NumPy array and by guarding against any ultra-rare classes that can break stratification (fallback to non-stratified split only if needed). I also fix the GPU detection logic so it doesn’t incorrectly select `gpu_hist` on CPU-only machines, preventing training/runtime failures. To move the score up toward your target (and because the current score is far below target), I keep the same XGBoost multiclass approach but switch from `multi:softmax` to `multi:softprob` with `mlogloss` monitoring (same core model family/objective) and then take `argmax` for labels; this typically improves accuracy versus hard-class training. Finally, I ensure we train/predict on the same feature columns (dropping `Id`), and always produce a valid `/kaggle/working/submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.56458) has done: 'I fix the XGBoost “Invalid classes inferred” crash by ensuring the mapped target `y` is a contiguous 0..K-1 NumPy array **and** by explicitly passing the correct `classes_` to the estimator (this error typically happens when the estimator infers a different set of class labels from the split). I also remove unsupported/unstable arguments for your installed xgboost version (notably `early_stopping_rounds` on `XGBClassifier.fit` in some 2.0.x builds) by switching to the supported callback-based early stopping, which is score-neutral but makes training run end-to-end. Finally, I make submission generation robust by building it directly from `sample_submission.csv` and asserting alignment, so `/kaggle/working/submission.csv` is always produced with the required columns.'
- What this solution (achieved 0.56458) has done: 'The timeout is almost certainly caused by training 20,000 boosting rounds on 3.6M rows; even with early stopping, that can take too long if it doesn’t stop very early. The fastest correctness-preserving fix is to run early stopping on a small, representative validation DMatrix first to find the best_iteration, then train once on the full training data for exactly that number of rounds (same objective/params/seed), avoiding thousands of wasted full-data iterations. I also remove repeated heavy pandas copies/concats in the “missing classes” fix by doing the same move operation with index arrays, and I reduce DMatrix construction overhead by using QuantileDMatrix for hist (CPU) and enabling DMatrix caching. All changes keep the same model family, objective, metric, early-stopping semantics, and prediction logic; they only eliminate unnecessary work.'
- What this solution (achieved 0.56458) has done: 'The timeout is dominated by reading and downcasting a 3.6M-row CSV in pandas and by doing a two-stage XGBoost training where the first stage uses up to 20k rounds with early stopping. I remove the expensive per-column dtype inference/downcast pass by supplying dtypes up-front (equivalent values, far less work), and I avoid the extra pandas-to-numpy conversions/copies by converting once early and keeping arrays contiguous. I also speed up the “find best iteration” stage by using XGBoost’s built-in `cv` (same early-stopping semantics on the same subset, but significantly less Python overhead than training a full booster + predicting), then train the final model with the selected number of rounds exactly as before. Everything else (features, objective, params, subset size, early stopping criteria, final training approach) remains the same.'
- What this solution (achieved 0.56458) has done: 'The timeout is dominated by the `xgb.cv(..., num_boost_round=20000, nfold=5)` call, which trains up to 100k boosting runs (20k * 5 folds) even though early stopping usually triggers much earlier. To preserve the exact same core logic (CV to select `best_ntree_limit`, then train on full training data with that many rounds), the main speed fix is to *resume CV in chunks* using `xgb_model`, so we only pay for the rounds actually evaluated until early stopping triggers, rather than pre-allocating a huge 20k ceiling. Additionally, the `QuantileDMatrix` is made once for the CV subset and reused, and we avoid unnecessary copies/re-materialization of arrays. These changes are equivalent in semantics (same parameters, same folds/seed, same early stopping behavior) but cut worst-case runtime dramatically.'

# 9. Code solution

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


def cv_with_resume(
    params,
    dtrain,
    nfold,
    stratified,
    metrics,
    seed,
    early_stopping_rounds,
    verbose_eval,
    max_total_rounds,
    chunk_rounds,
):
    booster_cv = None
    hist_parts = []
    total = 0
    while total < max_total_rounds:
        this_chunk = min(chunk_rounds, max_total_rounds - total)
        cv_hist_part = xgb.cv(
            params=params,
            dtrain=dtrain,
            num_boost_round=this_chunk,
            nfold=nfold,
            stratified=stratified,
            metrics=metrics,
            early_stopping_rounds=early_stopping_rounds,
            seed=seed,
            verbose_eval=verbose_eval,
            xgb_model=booster_cv,  # resume training identically
        )
        hist_parts.append(cv_hist_part)
        booster_cv = cv_hist_part.attrs.get("best_cvbooster", None)
        total += cv_hist_part.shape[0]
        if cv_hist_part.shape[0] < this_chunk:
            break
    return pd.concat(hist_parts, axis=0, ignore_index=True)


cv_hist = cv_with_resume(
    params=params,
    dtrain=dtrain_sub,
    nfold=5,
    stratified=True,
    metrics=("mlogloss",),
    seed=RANDOM_STATE,
    early_stopping_rounds=50,
    verbose_eval=200,
    max_total_rounds=20000,
    chunk_rounds=500,  # small enough to stop soon after optimum; no change in semantics
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



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2867903779.py in <cell line: 0>()
    118 
    119 
--> 120 cv_hist = cv_with_resume(
    121     params=params,
    122     dtrain=dtrain_sub,

/tmp/ipykernel_11/2867903779.py in cv_with_resume(params, dtrain, nfold, stratified, metrics, seed, early_stopping_rounds, verbose_eval, max_total_rounds, chunk_rounds)
     96     while total < max_total_rounds:
     97         this_chunk = min(chunk_rounds, max_total_rounds - total)
---> 98         cv_hist_part = xgb.cv(
     99             params=params,
    100             dtrain=dtrain,

TypeError: cv() got an unexpected keyword argument 'xgb_model'

## === cell 6
n_rounds = booster.num_boosted_rounds()
proba = booster.predict(dtest, iteration_range=(0, n_rounds))
y_pred_idx = np.asarray(proba).argmax(axis=1).astype(np.int32)

y_pred_label = np.take(classes_sorted, y_pred_idx).astype(np.int32)

print("Pred idx unique (sample):", np.unique(y_pred_idx)[:10], "...")
print("Pred label unique (sample):", np.unique(y_pred_label)[:10], "...")
print("Pred length:", len(y_pred_label), "Expected:", len(test_df))



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3352093632.py in <cell line: 0>()
----> 1 n_rounds = booster.num_boosted_rounds()
      2 proba = booster.predict(dtest, iteration_range=(0, n_rounds))
      3 y_pred_idx = np.asarray(proba).argmax(axis=1).astype(np.int32)
      4 
      5 y_pred_label = np.take(classes_sorted, y_pred_idx).astype(np.int32)

NameError: name 'booster' is not defined

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



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3185124452.py in <cell line: 0>()
      4 
      5 if np.array_equal(result["Id"].values, test_df["Id"].values):
----> 6     result["Cover_Type"] = y_pred_label
      7 else:
      8     pred_df = pd.DataFrame({"Id": test_df["Id"].values, "Cover_Type": y_pred_label})

NameError: name 'y_pred_label' is not defined

## === cell 8
out_path = "/kaggle/working/submission.csv"
result.to_csv(out_path, index=False)
print("Wrote:", out_path)
print("Columns:", result.columns.tolist())
print("Done")
