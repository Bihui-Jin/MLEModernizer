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
geopy==2.4.1
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
import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import xgboost as xgb

import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

N_JOBS = max(1, (os.cpu_count() or 1))
os.environ.setdefault("OMP_NUM_THREADS", str(N_JOBS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(N_JOBS))
os.environ.setdefault("MKL_NUM_THREADS", str(N_JOBS))
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(N_JOBS))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(N_JOBS))



## === cell 1
print(os.listdir("../input"))



## === cell 2
types = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}

test = pd.read_csv(
    "../input/test.csv",
    dtype={k: v for k, v in types.items() if k != "fare_amount"},
    usecols=[
        "key",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
)



## === cell 3
test.dtypes



## === cell 4
train = pd.read_csv(
    "../input/train.csv",
    nrows=4000000,
    dtype=types,
    usecols=[
        "key",
        "fare_amount",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
)



## === cell 5
train.head()



## === cell 6
pass



## === cell 7
pass



## === cell 8
pass



## === cell 9
train.isnull().sum()



## === cell 10
train = train.dropna()



## === cell 11
fare = train["fare_amount"].to_numpy()
plon = train["pickup_longitude"].to_numpy()
dlon = train["dropoff_longitude"].to_numpy()
plat = train["pickup_latitude"].to_numpy()
dlat = train["dropoff_latitude"].to_numpy()
pcnt = train["passenger_count"].to_numpy()

m = (
    (fare > 0)
    & (fare < 200)
    & (plon < -72)
    & (dlon < -72)
    & (plat > 40)
    & (plat < 44)
    & (dlat > 40)
    & (dlat < 44)
    & (pcnt > 0)
    & (pcnt < 10)
    & (plon >= -180.0)
    & (plon <= 180.0)
    & (dlon >= -180.0)
    & (dlon <= 180.0)
    & (plat >= -90.0)
    & (plat <= 90.0)
    & (dlat >= -90.0)
    & (dlat <= 90.0)
)
train = train.loc[m]
del fare, plon, dlon, plat, dlat, pcnt, m



## === cell 12
pass




## === cell 13
def add_haversine_distance_km(df: pd.DataFrame) -> pd.DataFrame:
    lat1 = np.deg2rad(df["pickup_latitude"].astype("float64").values)
    lon1 = np.deg2rad(df["pickup_longitude"].astype("float64").values)
    lat2 = np.deg2rad(df["dropoff_latitude"].astype("float64").values)
    lon2 = np.deg2rad(df["dropoff_longitude"].astype("float64").values)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon / 2.0) ** 2
    )
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arcsin(np.sqrt(a))
    earth_radius_km = 6371.0088
    df["distance"] = (earth_radius_km * c).astype("float32")
    return df




## === cell 14
train = add_haversine_distance_km(train)
test = add_haversine_distance_km(test)

dist = train["distance"].to_numpy(copy=False)
train["distance"] = np.where(np.isfinite(dist), dist, 0.0).astype("float32", copy=False)
dist = test["distance"].to_numpy(copy=False)
test["distance"] = np.where(np.isfinite(dist), dist, 0.0).astype("float32", copy=False)
del dist

train = train[(train["distance"] > 0.0) & (train["distance"] < 100.0)]



## === cell 15
s = train["pickup_datetime"]
if s.dtype != "object":
    s = s.astype("string")
train["pickup_datetime"] = pd.to_datetime(
    s.str.replace(" UTC", "", regex=False),
    format="%Y-%m-%d %H:%M:%S",
    errors="coerce",
)
del s



## === cell 16
s = test["pickup_datetime"]
if s.dtype != "object":
    s = s.astype("string")
test["pickup_datetime"] = pd.to_datetime(
    s.str.replace(" UTC", "", regex=False),
    format="%Y-%m-%d %H:%M:%S",
    errors="coerce",
)
del s



## === cell 17
train = train.dropna(subset=["pickup_datetime"])

test["pickup_datetime"] = test["pickup_datetime"].fillna(
    pd.Timestamp("2010-01-01 00:00:00")
)

train_dt = train["pickup_datetime"].dt
test_dt = test["pickup_datetime"].dt

train["hour"] = train_dt.hour.astype("int16")
train["weekday"] = train_dt.weekday.astype("int16")
train["month"] = train_dt.month.astype("int16")
train["year"] = train_dt.year.astype("int16")

test["hour"] = test_dt.hour.astype("int16")
test["weekday"] = test_dt.weekday.astype("int16")
test["month"] = test_dt.month.astype("int16")
test["year"] = test_dt.year.astype("int16")

NYC_LON = -73.985428
NYC_LAT = 40.748817

for df in (train, test):
    df["pickup_longitude_c"] = (df["pickup_longitude"] - NYC_LON).astype("float32")
    df["dropoff_longitude_c"] = (df["dropoff_longitude"] - NYC_LON).astype("float32")
    df["pickup_latitude_c"] = (df["pickup_latitude"] - NYC_LAT).astype("float32")
    df["dropoff_latitude_c"] = (df["dropoff_latitude"] - NYC_LAT).astype("float32")

del train_dt, test_dt



## === cell 18
test.head()



## === cell 19
pass



## === cell 20
X = train.drop(["key", "fare_amount", "pickup_datetime"], axis=1)
y = train["fare_amount"]



## === cell 21
X.head()



## === cell 22
y.head()



## === cell 23
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)



## === cell 24
test_pred = test.drop(["key", "pickup_datetime"], axis=1)



## === cell 25
X_train_np = np.ascontiguousarray(X_train.to_numpy(dtype=np.float32, copy=False))
X_valid_np = np.ascontiguousarray(X_valid.to_numpy(dtype=np.float32, copy=False))
y_train_np = np.ascontiguousarray(y_train.to_numpy(dtype=np.float32, copy=False))
y_valid_np = np.ascontiguousarray(y_valid.to_numpy(dtype=np.float32, copy=False))
test_pred_np = np.ascontiguousarray(test_pred.to_numpy(dtype=np.float32, copy=False))

lm = LinearRegression(n_jobs=N_JOBS)
lm.fit(X_train_np, y_train_np)
print(lm.score(X_train_np, y_train_np))
print(lm.score(X_valid_np, y_valid_np))



## === cell 26
y_valid_pred = lm.predict(X_valid_np)
lrmse = np.sqrt(metrics.mean_squared_error(y_valid_pred, y_valid_np))
print("Linear RMSE (fit on train split, eval on valid split):", lrmse)



## === cell 27
LinearPredictions = lm.predict(test_pred_np)



## === cell 28
LinearPredictions.size



## === cell 29
linear_submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": LinearPredictions},
    columns=["key", "fare_amount"],
)
linear_submission.head()




## === cell 30
def XGBoost(X_train_np, X_valid_np, y_train_np, y_valid_np):
    dtrain = xgb.DMatrix(X_train_np, label=y_train_np)
    dvalid = xgb.DMatrix(X_valid_np, label=y_valid_np)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.05,
        "max_depth": 8,
        "min_child_weight": 1.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "seed": RANDOM_STATE,
        "tree_method": "hist",
        "nthread": N_JOBS,
    }

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=2000,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        verbose_eval=200,
    )
    return booster




## === cell 31
xgbm = XGBoost(X_train_np, X_valid_np, y_train_np, y_valid_np)

dvalid = xgb.DMatrix(X_valid_np)
valid_pred = xgbm.predict(dvalid)
vrmse = np.sqrt(metrics.mean_squared_error(valid_pred, y_valid_np))
print("XGB RMSE (valid):", vrmse)

X_all_np = np.ascontiguousarray(X.to_numpy(dtype=np.float32, copy=False))
y_all_np = np.ascontiguousarray(y.to_numpy(dtype=np.float32, copy=False))
dall = xgb.DMatrix(X_all_np, label=y_all_np)

params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.05,
    "max_depth": 8,
    "min_child_weight": 1.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "seed": RANDOM_STATE,
    "tree_method": "hist",
    "nthread": N_JOBS,
}

xgbm_full = xgb.train(
    params=params,
    dtrain=dall,
    num_boost_round=2000,
    evals=[(dall, "train")],
    verbose_eval=200,
)

dtest_pred = xgb.DMatrix(test_pred_np)
XGBPredictions = xgbm_full.predict(dtest_pred)

del dvalid, X_all_np, y_all_np, dall, dtest_pred



## === cell 32
XGBPredictions



## === cell 33
XGB_submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": XGBPredictions}, columns=["key", "fare_amount"]
)
XGB_submission.head()



## === cell 34
submission = XGB_submission

submission["fare_amount"] = np.where(
    np.isfinite(submission["fare_amount"].values),
    submission["fare_amount"].values,
    0.0,
).astype("float32")

submission["fare_amount"] = submission["fare_amount"].clip(lower=0, upper=200)

submission.to_csv("submission.csv", index=False)
print("Wrote submission:", "submission.csv", "rows:", len(submission))
print(submission.head())
