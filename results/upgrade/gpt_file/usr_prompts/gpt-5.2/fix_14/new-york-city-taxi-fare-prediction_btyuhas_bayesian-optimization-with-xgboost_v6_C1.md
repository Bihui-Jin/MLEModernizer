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
Predict the fare amount for a taxi ride given the pickup and dropoff locations.

## Metric
Root mean-squared error.

## Submission Format
For each `key` in the test set, you must predict a value for the `fare_amount` variable. The file should contain a header and have the following format:

```
key,fare_amount
2015-01-27 13:08:24.0000002,11.00
2015-02-27 13:08:24.0000002,12.05
2015-03-27 13:08:24.0000002,11.23
2015-04-27 13:08:24.0000002,14.17
2015-05-27 13:08:24.0000002,15.12
etc
```

## Dataset
- **train.csv** - Input features and target `fare_amount` values for the training set (about 55M rows).
- **test.csv** - Input features for the test set (about 10K rows). Your goal is to predict `fare_amount` for each row.
- **sample_submission.csv** - a sample submission file in the correct format (columns `key` and `fare_amount`). This file 'predicts' `fare_amount` to be $`11.35` for all rows, which is the mean `fare_amount` from the training set.

### Data fields
**ID**

- **key** - Unique `string` identifying each row in both the training and test sets. Comprised of **pickup_datetime** plus a unique integer, but this doesn't matter, it should just be used as a unique ID field.Required in your submission CSV. Not necessarily needed in the training set, but could be useful to simulate a 'submission file' while doing cross-validation within the training set.

**Features**

- **pickup_datetime** - `timestamp` value indicating when the taxi ride started.
- **pickup_longitude** - `float` for longitude coordinate of where the taxi ride started.
- **pickup_latitude** - `float` for latitude coordinate of where the taxi ride started.
- **dropoff_longitude** - `float` for longitude coordinate of where the taxi ride ended.
- **dropoff_latitude** - `float` for latitude coordinate of where the taxi ride ended.
- **passenger_count** - `integer` indicating the number of passengers in the taxi ride.

**Target**

- **fare_amount** - `float` dollar amount of the cost of the taxi ride. This value is only in the training set; this is what you are predicting in the test set and it is required in your submission CSV.

# 2. Python version

3.7

# 3. Installed packages

bayesian-optimization==3.1.0
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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
xgboost==2.0.3

# 4. Data file paths

```
/
    kaggle/
        data/
            GCP-Coupons-Instructions.rtf (486 Bytes)
            description.md (100 lines)
            labels.csv (55413943 lines)
            labels.csv.zip (1.6 GB)
            sample_submission.csv (9915 lines)
            sample_submission.csv.zip (76.2 kB)
            test.csv (9915 lines)
            test.csv.zip (273.0 kB)
            train.csv (55423857 lines)
            train.csv.zip (1.6 GB)
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 8 other files
                new-york-city-taxi-fare-prediction/
        input/
            GCP-Coupons-Instructions.rtf (486 Bytes)
            description.md (100 lines)
            labels.csv (55413943 lines)
            labels.csv.zip (1.6 GB)
            sample_submission.csv (9915 lines)
            sample_submission.csv.zip (76.2 kB)
            test.csv (9915 lines)
            test.csv.zip (273.0 kB)
            train.csv (55423857 lines)
            train.csv.zip (1.6 GB)
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 8 other files
                new-york-city-taxi-fare-prediction/
        working/
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 8 other files
                new-york-city-taxi-fare-prediction/
```

-> data/labels.csv has 55413942 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/new-york-city-taxi-fare-prediction/labels.csv has 55413942 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/new-york-city-taxi-fare-prediction/sample_submission.csv has 9914 rows and 2 columns.
The columns are: key, fare_amount

-> data/new-york-city-taxi-fare-prediction/test.csv has 9914 rows and 7 columns.
The columns are: key, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/new-york-city-taxi-fare-prediction/train.csv has 55423856 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/sample_submission.csv has 9914 rows and 2 columns.
The columns are: key, fare_amount

-> data/test.csv has 9914 rows and 7 columns.
The columns are: key, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/train.csv has 55423856 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> (stopped after 10 files for performance)

# 5. Target score

3.24593

# 6. Current score

5.82078

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 8.32305) has done: 'I fix the BayesianOptimization API mismatch by removing the unsupported `acq` argument and reading the best parameters using the library’s current `max`/`res` structures, which unblocks training. I also ensure XGBoost parameters include `objective` and a consistent `seed`, and I make the cross-validation call explicitly set `metrics='rmse'` for stability with xgboost 2.x. Finally, I adjust file paths to the Kaggle-provided `/kaggle/input/...` location (your current `../input/...` is what typically breaks), and keep the rest of the feature logic/model approach unchanged while guaranteeing a valid `submission.csv` is written.'
- What this solution (achieved 6.89961) has done: 'The timeout is dominated by repeated 3-fold `xgb.cv` inside Bayesian Optimization (8 trials × up to 2000 rounds each), plus some avoidable pandas overhead during feature engineering. I keep the exact same model/training logic (BayesOpt over `xgb.cv`, then `xgb.train` with best rounds), but make `xgb.cv` much faster by enabling XGBoost’s internal fold-parallelism (`nthread` in `xgb.cv`), using histogram bin precomputation once (`QuantileDMatrix`) and reusing it across all CV calls, and avoiding repeated DataFrame-to-DMatrix conversions. I also make the datetime parsing/feature creation more vectorized and reduce pandas copies without changing any computed features. These changes are correctness-preserving (same data, same features, same objective/metric, same CV/early stopping semantics), but cut constant factors substantially.'
- What this solution (achieved 6.89961) has done: 'I fix the runtime error by removing the unsupported `nthread` argument from `xgboost.cv` (in xgboost 2.0.x you control threads via params like `nthread`). I also correct the input paths to point to the provided `/kaggle/input/new-york-city-taxi-fare-prediction/...` files so the notebook reliably finds the data in this environment. These changes are execution/correctness fixes and keep your modeling/feature logic identical; they should let Bayesian Optimization run again and typically improve score versus the fallback parameters. Finally, I keep the submission writing unchanged and ensure it produces a valid `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 6.89961) has done: 'The timeout is dominated by repeated 3-fold `xgb.cv` runs inside Bayesian Optimization (8 evaluations × up to 2000 boosting rounds each) on a ~2M-row DMatrix, plus extra overhead from DMatrix construction holding pandas objects and redundant datetime/string conversions. I keep the exact same modeling logic (same feature engineering, same BayesianOptimization loop, same CV settings/round limits, same final train/predict) but reduce overhead by (1) converting features/labels to contiguous NumPy arrays once and using `QuantileDMatrix` for faster histogram building, (2) enabling XGBoost’s built-in CV result caching to avoid redoing identical folds work, and (3) ensuring preprocessing avoids extra copies. These changes are algebraically equivalent (no approximations or relaxed training), but significantly cut constant factors in CPU time.'
- What this solution (achieved 5.82078) has done: 'I fix the immediate runtime error by removing the unsupported `enable_cv_cache` argument from `xgboost.cv` (it isn’t available in xgboost 2.0.3), while keeping the same CV/early-stopping semantics and BayesianOptimization loop intact. To move RMSE toward your target, I make one minimal metric-aligned feature correction: fix `distance_to_center` to actually use dropoff coordinates (it currently reuses pickup coords, which is a logic bug hurting signal). I also make the BO cache key stable by rounding floats to avoid redundant CV runs due to tiny float differences, improving runtime without changing training logic. The rest of the pipeline (data reading, filtering, feature set, BO→final training, and submission writing) stays the same and produce a valid `submission.csv`.'
- What this solution (achieved 5.82078) has done: 'The timeout is dominated by repeated `xgb.cv` runs inside Bayesian Optimization (8 evaluations × 3-fold × up to 2000 rounds), plus avoidable pandas overhead in feature engineering. I keep the exact same model/training logic and search space, but make runtime-equivalent changes: (1) run cross-validation on `QuantileDMatrix` to reduce CPU/memory cost while preserving training semantics for `tree_method='hist'`, (2) avoid pandas `DataFrame.copy()`/`drop()` inside `transform()` by constructing features directly into a new frame, and (3) ensure the expensive feature importance plot doesn’t render interactively (common notebook slowdown) while keeping the computation. These are provably equivalent with only negligible float differences and typically cut total runtime substantially under the 600s limit.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

BASE_INPUT = "/kaggle/input/new-york-city-taxi-fare-prediction"
TRAIN_PATH = os.path.join(BASE_INPUT, "train.csv")
TEST_PATH = os.path.join(BASE_INPUT, "test.csv")

os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 1))
os.environ.setdefault("MKL_NUM_THREADS", str(os.cpu_count() or 1))

np.random.seed(42)

print("TRAIN_PATH:", TRAIN_PATH)
print("TEST_PATH :", TEST_PATH)
print("Train exists?", os.path.exists(TRAIN_PATH))
print("Test exists? ", os.path.exists(TEST_PATH))



## === cell 1
dtype_map = {
    "fare_amount": "float64",
    "pickup_datetime": "object",
    "pickup_longitude": "float64",
    "pickup_latitude": "float64",
    "dropoff_longitude": "float64",
    "dropoff_latitude": "float64",
    "passenger_count": "int64",
}
usecols = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

df = pd.read_csv(
    TRAIN_PATH,
    nrows=2_000_000,
    usecols=usecols,
    dtype=dtype_map,
    engine="c",
)

print("Loaded train sample:", df.shape)



## === cell 2
s = df["pickup_datetime"].astype(str).str.slice(0, 16)
df["pickup_datetime"] = pd.to_datetime(s, utc=True, format="%Y-%m-%d %H:%M")



## === cell 3
df.dropna(how="any", axis="rows", inplace=True)

mask = df["pickup_longitude"].between(-75, -73)
mask &= df["dropoff_longitude"].between(-75, -73)
mask &= df["pickup_latitude"].between(40, 42)
mask &= df["dropoff_latitude"].between(40, 42)
mask &= df["passenger_count"].between(0, 8)
mask &= df["fare_amount"].between(0, 250)

df = df[mask]
print("After filtering:", df.shape)




## === cell 4
def dist(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    distance = np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)
    return distance




## === cell 5
def transform(data):
    dt = data["pickup_datetime"]
    hour = dt.dt.hour.to_numpy(copy=False)
    day = dt.dt.day.to_numpy(copy=False)
    month = dt.dt.month.to_numpy(copy=False)
    year = dt.dt.year.to_numpy(copy=False)

    plon = data["pickup_longitude"].to_numpy(copy=False)
    plat = data["pickup_latitude"].to_numpy(copy=False)
    dlon = data["dropoff_longitude"].to_numpy(copy=False)
    dlat = data["dropoff_latitude"].to_numpy(copy=False)

    nyc = (-74.0063889, 40.7141667)
    jfk = (-73.7822222222, 40.6441666667)
    ewr = (-74.175, 40.69)
    lgr = (-73.87, 40.77)

    cols = [c for c in data.columns if c != "pickup_datetime"]
    out = pd.DataFrame(index=data.index)
    for c in cols:
        out[c] = data[c].to_numpy(copy=False)

    out["hour"] = hour
    out["day"] = day
    out["month"] = month
    out["year"] = year

    out["distance_to_center"] = dist(nyc[1], nyc[0], dlat, dlon)

    out["pickup_distance_to_jfk"] = dist(jfk[1], jfk[0], plat, plon)
    out["dropoff_distance_to_jfk"] = dist(jfk[1], jfk[0], dlat, dlon)
    out["pickup_distance_to_ewr"] = dist(ewr[1], ewr[0], plat, plon)
    out["dropoff_distance_to_ewr"] = dist(ewr[1], ewr[0], dlat, dlon)
    out["pickup_distance_to_lgr"] = dist(lgr[1], lgr[0], plat, plon)
    out["dropoff_distance_to_lgr"] = dist(lgr[1], lgr[0], dlat, dlon)

    out["long_dist"] = plon - dlon
    out["lat_dist"] = plat - dlat

    out["dist"] = dist(plat, plon, dlat, dlon)
    return out


df = transform(df)
print("Transformed train:", df.shape)



## === cell 6
try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error



## === cell 7
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df.drop("fare_amount", axis=1),
    df["fare_amount"],
    test_size=0.25,
    random_state=42,
)
del df

NTHREAD = os.cpu_count() or 1

X_train_np = np.asarray(X_train.to_numpy(dtype=np.float32, copy=False), order="C")
X_test_np = np.asarray(X_test.to_numpy(dtype=np.float32, copy=False), order="C")
y_train_np = np.asarray(y_train.to_numpy(dtype=np.float32, copy=False), order="C")
y_test_np = np.asarray(y_test.to_numpy(dtype=np.float32, copy=False), order="C")
feature_names = list(X_train.columns)

del X_train, X_test, y_train, y_test

dtrain = xgb.QuantileDMatrix(
    X_train_np, label=y_train_np, feature_names=feature_names, nthread=NTHREAD
)
dtest = xgb.QuantileDMatrix(
    X_test_np, label=y_test_np, feature_names=feature_names, nthread=NTHREAD
)



## === cell 8
_cv_cache = {}

_BASE_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "subsample": 0.8,
    "eta": 0.1,
    "seed": 42,
    "verbosity": 0,
    "nthread": NTHREAD,
    "tree_method": "hist",
}


def xgb_evaluate(max_depth, gamma, colsample_bytree):
    k_max_depth = int(max_depth)
    k_gamma = float(np.round(gamma, 6))
    k_colsample = float(np.round(colsample_bytree, 6))
    key = (k_max_depth, k_gamma, k_colsample)

    if key in _cv_cache:
        return _cv_cache[key]["score"]

    params = dict(_BASE_PARAMS)
    params.update(
        {
            "max_depth": k_max_depth,
            "gamma": k_gamma,
            "colsample_bytree": k_colsample,
        }
    )

    cv_result = xgb.cv(
        params,
        dtrain,
        num_boost_round=2000,
        nfold=3,
        metrics="rmse",
        seed=42,
        verbose_eval=False,
        early_stopping_rounds=30,
        shuffle=True,
        as_pandas=True,
    )

    rmse = float(cv_result["test-rmse-mean"].iloc[-1])
    best_iteration = int(len(cv_result))
    _cv_cache[key] = {"score": -rmse, "best_iteration": best_iteration}
    return -rmse




## === cell 9
xgb_bo = BayesianOptimization(
    f=xgb_evaluate,
    pbounds={"max_depth": (3, 7), "gamma": (0, 1), "colsample_bytree": (0.3, 0.9)},
    random_state=42,
    verbose=2,
)
xgb_bo.maximize(init_points=3, n_iter=5)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
XGBoostError                              Traceback (most recent call last)
/tmp/ipykernel_11/725333033.py in <cell line: 0>()
      5     verbose=2,
      6 )
----> 7 xgb_bo.maximize(init_points=3, n_iter=5)
      8 

/usr/local/lib/python3.11/dist-packages/bayes_opt/bayesian_optimization.py in maximize(self, init_points, n_iter)
    320                 x_probe = self.suggest()
    321                 iteration += 1
--> 322             self.probe(x_probe, lazy=False)
    323 
    324             if self._bounds_transformer and iteration > 0:

/usr/local/lib/python3.11/dist-packages/bayes_opt/bayesian_optimization.py in probe(self, params, lazy)
    237             self._queue.append(params)
    238         else:
--> 239             self._space.probe(params)
    240             self.logger.log_optimization_step(
    241                 self._space.keys, self._space.res()[-1], self._space.params_config, self.max

/usr/local/lib/python3.11/dist-packages/bayes_opt/target_space.py in probe(self, params)
    553             error_msg = "No target function has been provided."
    554             raise ValueError(error_msg)
--> 555         target = self.target_func(**dict_params)
    556 
    557         if self._constraint is None:

/tmp/ipykernel_11/3071345986.py in xgb_evaluate(max_depth, gamma, colsample_bytree)
     31     )
     32 
---> 33     cv_result = xgb.cv(
     34         params,
     35         dtrain,

/usr/local/lib/python3.11/dist-packages/xgboost/training.py in cv(params, dtrain, num_boost_round, nfold, stratified, folds, metrics, obj, feval, maximize, early_stopping_rounds, fpreproc, as_pandas, verbose_eval, show_stdv, seed, callbacks, shuffle, custom_metric)
    541 
    542     results: Dict[str, List[float]] = {}
--> 543     cvfolds = mknfold(
    544         dtrain, nfold, params, seed, metrics, fpreproc, stratified, folds, shuffle
    545     )

/usr/local/lib/python3.11/dist-packages/xgboost/training.py in mknfold(dall, nfold, param, seed, evals, fpreproc, stratified, folds, shuffle)
    396     for k in range(nfold):
    397         # perform the slicing using the indexes determined by the above methods
--> 398         dtrain = dall.slice(in_idset[k])
    399         dtest = dall.slice(out_idset[k])
    400         # run preprocessing on the data set if needed

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in slice(self, rindex, allow_groups)
   1256         res.handle = ctypes.c_void_p()
   1257         rindex = _maybe_np_slice(rindex, dtype=np.int32)
-> 1258         _check_call(
   1259             _LIB.XGDMatrixSliceDMatrixEx(
   1260                 self.handle,

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in _check_call(ret)
    280     """
    281     if ret != 0:
--> 282         raise XGBoostError(py_str(_LIB.XGBGetLastError()))
    283 
    284 

XGBoostError: [03:25:20] /workspace/src/data/iterative_dmatrix.h:88: Slicing DMatrix is not supported for Quantile DMatrix.
Stack trace:
  [bt] (0) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x3effba) [0x7ffefdd02fba]
  [bt] (1) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x3ff7ab) [0x7ffefdd127ab]
  [bt] (2) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(XGDMatrixSliceDMatrixEx+0x146) [0x7ffefda73206]
  [bt] (3) /lib/x86_64-linux-gnu/libffi.so.8(+0x7e2e) [0x7ffff63ace2e]
  [bt] (4) /lib/x86_64-linux-gnu/libffi.so.8(+0x4493) [0x7ffff63a9493]
  [bt] (5) /usr/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so(+0xa4d8) [0x7ffff63bc4d8]
  [bt] (6) /usr/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so(+0x9c8e) [0x7ffff63bbc8e]
  [bt] (7) /usr/bin/python3(_PyObject_MakeTpCall+0x27c) [0x52f85c]
  [bt] (8) /usr/bin/python3(_PyEval_EvalFrameDefault+0x6bc) [0x53da0c]



## === cell 10
if (
    getattr(xgb_bo, "max", None)
    and isinstance(xgb_bo.max, dict)
    and xgb_bo.max.get("params") is not None
):
    best_params = dict(xgb_bo.max["params"])
    best_params["max_depth"] = int(best_params["max_depth"])
    best_params["gamma"] = float(np.round(float(best_params["gamma"]), 6))
    best_params["colsample_bytree"] = float(
        np.round(float(best_params["colsample_bytree"]), 6)
    )
else:
    best_params = {"max_depth": 5, "gamma": 0.0, "colsample_bytree": 0.7}

best_params.update(_BASE_PARAMS)

_cache_key = (
    int(best_params["max_depth"]),
    float(best_params["gamma"]),
    float(best_params["colsample_bytree"]),
)
best_nrounds = int(_cv_cache.get(_cache_key, {}).get("best_iteration", 500))
print("Best params:", best_params)
print("Best nrounds:", best_nrounds)



## === cell 11
model2 = xgb.train(best_params, dtrain, num_boost_round=best_nrounds)

y_pred = model2.predict(dtest)
y_train_pred = model2.predict(dtrain)

print("Holdout RMSE:", np.sqrt(mean_squared_error(y_test_np, y_pred)))
print("Train RMSE  :", np.sqrt(mean_squared_error(y_train_np, y_train_pred)))



## === cell 12
matplotlib.use("Agg", force=True)

fscores = pd.DataFrame(
    {"X": list(model2.get_fscore().keys()), "Y": list(model2.get_fscore().values())}
)
ax = fscores.sort_values(by="Y").plot.bar(x="X", figsize=(12, 4), legend=False)
_ = ax.set_title("Feature importance (fscore)")
plt.tight_layout()
plt.close()



## === cell 13
test_dtype_map = {
    "key": "object",
    "pickup_datetime": "object",
    "pickup_longitude": "float64",
    "pickup_latitude": "float64",
    "dropoff_longitude": "float64",
    "dropoff_latitude": "float64",
    "passenger_count": "int64",
}
test_usecols = list(test_dtype_map.keys())

test = pd.read_csv(
    TEST_PATH, usecols=test_usecols, dtype=test_dtype_map, engine="c"
).set_index("key")

s = test["pickup_datetime"].astype(str).str.slice(0, 16)
test["pickup_datetime"] = pd.to_datetime(s, utc=True, format="%Y-%m-%d %H:%M")

test = transform(test)

test_np = np.asarray(test.to_numpy(dtype=np.float32, copy=False), order="C")
dtest_final = xgb.QuantileDMatrix(
    test_np, feature_names=list(test.columns), nthread=NTHREAD
)

y_pred_test = model2.predict(dtest_final)
y_pred_test = np.clip(y_pred_test, 0.0, 250.0)

holdout = pd.DataFrame({"key": test.index, "fare_amount": y_pred_test})
holdout.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", holdout.shape)
print(holdout.head())
