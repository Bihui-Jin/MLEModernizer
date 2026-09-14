# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
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

    out = data.drop("pickup_datetime", axis=1).copy()

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

dtrain = xgb.DMatrix(
    X_train_np, label=y_train_np, feature_names=feature_names, nthread=NTHREAD
)
dtest = xgb.DMatrix(
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
import matplotlib.pyplot as plt

fscores = pd.DataFrame(
    {"X": list(model2.get_fscore().keys()), "Y": list(model2.get_fscore().values())}
)
ax = fscores.sort_values(by="Y").plot.bar(x="X", figsize=(12, 4), legend=False)
_ = ax.set_title("Feature importance (fscore)")
plt.tight_layout()



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
dtest_final = xgb.DMatrix(test_np, feature_names=list(test.columns), nthread=NTHREAD)

y_pred_test = model2.predict(dtest_final)
y_pred_test = np.clip(y_pred_test, 0.0, 250.0)

holdout = pd.DataFrame({"key": test.index, "fare_amount": y_pred_test})
holdout.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", holdout.shape)
print(holdout.head())
