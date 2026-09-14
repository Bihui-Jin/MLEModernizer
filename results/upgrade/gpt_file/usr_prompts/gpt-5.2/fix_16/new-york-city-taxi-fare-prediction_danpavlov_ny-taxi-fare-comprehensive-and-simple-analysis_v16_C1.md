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
seaborn==0.12.2
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

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")

import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import xgboost as xgb

np.random.seed(42)


def _resolve_input_dir():
    candidates = [
        "../input",  # classic Kaggle notebooks
        "/kaggle/input",  # Kaggle containers
        "/kaggle/data",  # some environments mount here
        "../kaggle/input",  # fallback
        "../kaggle/data",  # fallback
    ]
    for c in candidates:
        if os.path.isdir(c) and (
            os.path.exists(os.path.join(c, "train.csv"))
            or os.path.exists(
                os.path.join(c, "new-york-city-taxi-fare-prediction", "train.csv")
            )
        ):
            return c
    return "."


INPUT_DIR = _resolve_input_dir()


def _p(name):
    p1 = os.path.join(INPUT_DIR, name)
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(INPUT_DIR, "new-york-city-taxi-fare-prediction", name)
    if os.path.exists(p2):
        return p2
    return p1


print("Using INPUT_DIR:", INPUT_DIR)
print("train.csv exists:", os.path.exists(_p("train.csv")))
print("test.csv exists :", os.path.exists(_p("test.csv")))
print("sample_submission.csv exists:", os.path.exists(_p("sample_submission.csv")))



## === cell 1
pass



## === cell 2
test_types = {
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "float32",  # read as float then clean/cast
}
test_usecols = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

test = pd.read_csv(
    _p("test.csv"),
    usecols=test_usecols,
    dtype=test_types,
    engine="c",
    low_memory=False,
    parse_dates=["pickup_datetime"],
)



## === cell 3
pass



## === cell 4
types = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "float32",
}



## === cell 5
train_usecols = test_usecols + ["fare_amount"]

train = pd.read_csv(
    _p("train.csv"),
    nrows=8_000_000,
    usecols=train_usecols,
    dtype=types,
    engine="c",
    low_memory=False,
    memory_map=True,
    parse_dates=["pickup_datetime"],
)



## === cell 6
pass



## === cell 7
pass



## === cell 8
pass



## === cell 9
pass



## === cell 10
pass



## === cell 11
train.dropna(
    subset=[
        "fare_amount",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
    inplace=True,
)



## === cell 12
train["passenger_count"] = pd.to_numeric(
    train["passenger_count"], errors="coerce"
).fillna(1.0)
train["passenger_count"] = (
    train["passenger_count"].clip(lower=1, upper=9).astype("uint8")
)

plon = train["pickup_longitude"].to_numpy(copy=False)
dlon = train["dropoff_longitude"].to_numpy(copy=False)
plat = train["pickup_latitude"].to_numpy(copy=False)
dlat = train["dropoff_latitude"].to_numpy(copy=False)
fare = train["fare_amount"].to_numpy(copy=False)
pc = train["passenger_count"].to_numpy(copy=False)

m = (
    (fare > 0.0)
    & (fare < 250.0)
    & (plon < -72)
    & (dlon < -72)
    & (plat > 40)
    & (plat < 44)
    & (dlat > 40)
    & (dlat < 44)
    & (pc >= 1)
    & (pc <= 6)
)
same_loc = (plon == dlon) & (plat == dlat)
m &= ~same_loc

train = train.loc[m]
del plon, dlon, plat, dlat, fare, pc, same_loc, m
gc.collect()



## === cell 13
pass




## === cell 14
def _haversine_km_np(lat1, lon1, lat2, lon2):
    R = 6373.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


def _compute_all_dist_features(df):
    lat_pu = df["pickup_latitude"].to_numpy(copy=False).astype(np.float64, copy=False)
    lon_pu = df["pickup_longitude"].to_numpy(copy=False).astype(np.float64, copy=False)
    lat_do = df["dropoff_latitude"].to_numpy(copy=False).astype(np.float64, copy=False)
    lon_do = df["dropoff_longitude"].to_numpy(copy=False).astype(np.float64, copy=False)

    df["distance"] = _haversine_km_np(lat_pu, lon_pu, lat_do, lon_do).astype(
        "float32", copy=False
    )

    jfk_lat, jfk_lon = 40.645972, -73.785193
    lga_lat, lga_lon = 40.773335, -73.872925
    ewr_lat, ewr_lon = 40.692764, -74.184156
    mht_lat, mht_lon = 40.759006, -73.983132

    jfk_pu = _haversine_km_np(lat_pu, lon_pu, jfk_lat, jfk_lon)
    jfk_do = _haversine_km_np(lat_do, lon_do, jfk_lat, jfk_lon)
    lga_pu = _haversine_km_np(lat_pu, lon_pu, lga_lat, lga_lon)
    lga_do = _haversine_km_np(lat_do, lon_do, lga_lat, lga_lon)
    ewr_pu = _haversine_km_np(lat_pu, lon_pu, ewr_lat, ewr_lon)
    ewr_do = _haversine_km_np(lat_do, lon_do, ewr_lat, ewr_lon)
    mht_pu = _haversine_km_np(lat_pu, lon_pu, mht_lat, mht_lon)
    mht_do = _haversine_km_np(lat_do, lon_do, mht_lat, mht_lon)

    df["jfk_distance"] = np.minimum(jfk_pu, jfk_do).astype("float32", copy=False)
    df["laguardia_distance"] = np.minimum(lga_pu, lga_do).astype("float32", copy=False)
    df["newark_distance"] = np.minimum(ewr_pu, ewr_do).astype("float32", copy=False)
    df["manhattan_distance"] = np.minimum(mht_pu, mht_do).astype("float32", copy=False)

    del lat_pu, lon_pu, lat_do, lon_do
    del jfk_pu, jfk_do, lga_pu, lga_do, ewr_pu, ewr_do, mht_pu, mht_do
    gc.collect()




## === cell 15
_compute_all_dist_features(train)
_compute_all_dist_features(test)

dist = train["distance"].to_numpy(copy=False)
train = train.loc[(dist > 0.0) & (dist < 60.0)]
del dist
gc.collect()



## === cell 16
jfk_airport = (-73.785193, 40.645972)
laguardia_airport = (-73.872925, 40.773335)
newark_airport = (-74.184156, 40.692764)
manhattan = (-73.983132, 40.759006)
pass



## === cell 17
pass



## === cell 18
drop_cols = [
    "jfk_airport_pickup_dist",
    "jfk_airport_dropoff_dist",
    "laguardia_airport_pickup_dist",
    "laguardia_airport_dropoff_dist",
    "newark_airport_pickup_dist",
    "newark_airport_dropoff_dist",
    "manhattan_pickup_dist",
    "manhattan_dropoff_dist",
]
train.drop(columns=drop_cols, inplace=True, errors="ignore")
test.drop(columns=drop_cols, inplace=True, errors="ignore")



## === cell 19
train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], utc=True, errors="coerce"
).dt.tz_convert(None)
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], utc=True, errors="coerce"
).dt.tz_convert(None)
train.dropna(subset=["pickup_datetime"], inplace=True)



## === cell 20
dt = train["pickup_datetime"].dt
train["hour"] = dt.hour.astype("uint8")
train["weekday"] = dt.weekday.astype("uint8")
train["month"] = dt.month.astype("uint8")
train["year"] = dt.year.astype("uint16")

dt2 = test["pickup_datetime"].dt
test["hour"] = dt2.hour
test["weekday"] = dt2.weekday
test["month"] = dt2.month
test["year"] = dt2.year

test["passenger_count"] = pd.to_numeric(
    test["passenger_count"], errors="coerce"
).fillna(1.0)
test["passenger_count"] = test["passenger_count"].clip(lower=1, upper=9).astype("uint8")

for col in ["hour", "weekday", "month", "year"]:
    med = int(train[col].median())
    test[col] = (
        pd.to_numeric(test[col], errors="coerce").fillna(med).astype(train[col].dtype)
    )



## === cell 21
dist64 = train["distance"].to_numpy(dtype="float64", copy=False)
fare64 = train["fare_amount"].to_numpy(dtype="float64", copy=False)
fare_per_km = fare64 / np.clip(dist64, 0.1, None)

m1 = (fare64 >= 2.5) | (dist64 >= 0.2)
m2 = (fare_per_km > 0.2) & (fare_per_km < 80.0)
m3 = ~((dist64 < 0.5) & (fare64 > 50.0))
train = train.loc[m1 & m2 & m3]
del dist64, fare64, fare_per_km, m1, m2, m3
gc.collect()



## === cell 22
pass



## === cell 23
pass



## === cell 24
pass



## === cell 25
X = train.drop(["key", "fare_amount", "pickup_datetime"], axis=1)
y = train["fare_amount"]



## === cell 26
pass



## === cell 27
pass



## === cell 28
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



## === cell 29
test_pred = test.drop(["key", "pickup_datetime"], axis=1)



## === cell 30
try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

lm = LinearRegression()
lm.fit(X_train, y_train)
print(lm.score(X_train, y_train))
print(lm.score(X_test, y_test))



## === cell 31
pass



## === cell 32
pass



## === cell 33
pass



## === cell 34
pass



## === cell 35
pass



## === cell 36
Xtr_arr = np.asarray(X_train.to_numpy(copy=False), order="C", dtype=np.float32)
Xva_arr = np.asarray(X_test.to_numpy(copy=False), order="C", dtype=np.float32)
ytr_arr = np.asarray(y_train.to_numpy(copy=False), order="C", dtype=np.float32)
yva_arr = np.asarray(y_test.to_numpy(copy=False), order="C", dtype=np.float32)
Xsubmit_arr = np.asarray(test_pred.to_numpy(copy=False), order="C", dtype=np.float32)

del X_train, X_test, y_train, y_test, X, y, train
gc.collect()




## === cell 37
def XGBoost(X_train, X_test, y_train, y_test):
    dtrain = xgb.DMatrix(X_train, label=y_train, nthread=8)
    dvalid = xgb.DMatrix(X_test, label=y_test, nthread=8)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.05,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1.0,
        "gamma": 0.0,
        "lambda": 1.5,
        "alpha": 0.0,
        "seed": 42,
        "tree_method": "hist",
        "nthread": 8,
    }

    return xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=4000,
        early_stopping_rounds=100,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        verbose_eval=False,
    )




## === cell 38
xgbm = XGBoost(Xtr_arr, Xva_arr, ytr_arr, yva_arr)



## === cell 39
dtest_submit = xgb.DMatrix(Xsubmit_arr, nthread=8)

if getattr(xgbm, "best_iteration", None) is not None:
    XGBPredictions = xgbm.predict(
        dtest_submit, iteration_range=(0, xgbm.best_iteration + 1)
    )
else:
    XGBPredictions = xgbm.predict(dtest_submit)



## === cell 40
pass



## === cell 41
XGBPredictions = np.clip(XGBPredictions, 0.0, None)

fallback_fare = float(np.median(ytr_arr))

sample_sub = pd.read_csv(_p("sample_submission.csv"))
pred_df = pd.DataFrame({"key": test["key"].values, "fare_amount": XGBPredictions})

out = sample_sub.merge(pred_df, on="key", how="left")
out["fare_amount"] = out["fare_amount"].astype("float64")
out["fare_amount"].fillna(fallback_fare, inplace=True)
out["fare_amount"] = out["fare_amount"].clip(lower=0.0)

submission_path = "submission.csv"
out.to_csv(submission_path, index=False)
print("Wrote submission:", os.path.abspath(submission_path), "rows:", len(out))
print("Any NaNs left:", int(out["fare_amount"].isna().sum()))
print("Submission head:\n", out.head())
