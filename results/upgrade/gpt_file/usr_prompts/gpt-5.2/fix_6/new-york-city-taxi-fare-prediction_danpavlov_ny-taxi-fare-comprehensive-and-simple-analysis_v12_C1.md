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
    "../input/test.csv", dtype={k: v for k, v in types.items() if k != "fare_amount"}
)



## === cell 3
test.dtypes



## === cell 4
train = pd.read_csv("../input/train.csv", nrows=4000000, dtype=types)



## === cell 5
train.head()



## === cell 6
train.describe()



## === cell 7
try:
    sns.histplot(train["fare_amount"], kde=True)
    plt.close()
except Exception:
    pass



## === cell 8
try:
    sns.histplot(train["passenger_count"], kde=False)
    plt.close()
except Exception:
    pass



## === cell 9
train.isnull().sum()



## === cell 10
train.dropna(inplace=True)



## === cell 11
train = train[train["fare_amount"] > 0]
train = train[train["pickup_longitude"] < -72]
train = train[(train["pickup_latitude"] > 40) & (train["pickup_latitude"] < 44)]
train = train[train["dropoff_longitude"] < -72]
train = train[(train["dropoff_latitude"] > 40) & (train["dropoff_latitude"] < 44)]
train = train[(train["passenger_count"] > 0) & (train["passenger_count"] < 10)]
train = train[train["fare_amount"] < 200].copy()

for c, lo, hi in [
    ("pickup_longitude", -180.0, 180.0),
    ("dropoff_longitude", -180.0, 180.0),
    ("pickup_latitude", -90.0, 90.0),
    ("dropoff_latitude", -90.0, 90.0),
]:
    train = train[(train[c] >= lo) & (train[c] <= hi)]



## === cell 12
train.describe()




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

train["distance"] = np.where(
    np.isfinite(train["distance"].values), train["distance"].values, 0.0
).astype("float32")
test["distance"] = np.where(
    np.isfinite(test["distance"].values), test["distance"].values, 0.0
).astype("float32")



## === cell 15
train["pickup_datetime"] = (
    train["pickup_datetime"].astype(str).str.replace(" UTC", "", regex=False)
)
train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
)



## === cell 16
test["pickup_datetime"] = (
    test["pickup_datetime"].astype(str).str.replace(" UTC", "", regex=False)
)
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
)



## === cell 17
train = train.dropna(subset=["pickup_datetime"]).copy()

test["pickup_datetime"] = test["pickup_datetime"].fillna(
    pd.Timestamp("2010-01-01 00:00:00")
)

train["hour"] = train.pickup_datetime.dt.hour.astype("int16")
train["weekday"] = train.pickup_datetime.dt.weekday.astype("int16")
train["month"] = train.pickup_datetime.dt.month.astype("int16")
train["year"] = train.pickup_datetime.dt.year.astype("int16")

test["hour"] = test.pickup_datetime.dt.hour.astype("int16")
test["weekday"] = test.pickup_datetime.dt.weekday.astype("int16")
test["month"] = test.pickup_datetime.dt.month.astype("int16")
test["year"] = test.pickup_datetime.dt.year.astype("int16")

NYC_LON = -73.985428
NYC_LAT = 40.748817
for df in (train, test):
    df["pickup_longitude_c"] = (
        df["pickup_longitude"].astype("float32") - NYC_LON
    ).astype("float32")
    df["dropoff_longitude_c"] = (
        df["dropoff_longitude"].astype("float32") - NYC_LON
    ).astype("float32")
    df["pickup_latitude_c"] = (
        df["pickup_latitude"].astype("float32") - NYC_LAT
    ).astype("float32")
    df["dropoff_latitude_c"] = (
        df["dropoff_latitude"].astype("float32") - NYC_LAT
    ).astype("float32")



## === cell 18
test.head()



## === cell 19
try:
    plt.figure(figsize=(15, 8))
    sns.heatmap(
        train.drop(["key", "pickup_datetime"], axis=1).corr(numeric_only=True),
        annot=True,
        fmt=".4f",
    )
    plt.close()
except Exception:
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
lm = LinearRegression()
lm.fit(X_train, y_train)
print(lm.score(X_train, y_train))
print(lm.score(X_valid, y_valid))



## === cell 26
y_valid_pred = lm.predict(X_valid)
lrmse = np.sqrt(metrics.mean_squared_error(y_valid_pred, y_valid))
print("Linear RMSE (fit on train split, eval on valid split):", lrmse)



## === cell 27
LinearPredictions = lm.predict(test_pred)



## === cell 28
LinearPredictions.size



## === cell 29
linear_submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": LinearPredictions},
    columns=["key", "fare_amount"],
)
linear_submission.head()




## === cell 30
def XGBoost(X_train, X_valid, y_train, y_valid):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

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
xgbm = XGBoost(X_train, X_valid, y_train, y_valid)

dvalid = xgb.DMatrix(X_valid)
valid_pred = xgbm.predict(dvalid)
vrmse = np.sqrt(metrics.mean_squared_error(valid_pred, y_valid))
print("XGB RMSE (valid):", vrmse)

dtest_pred = xgb.DMatrix(test_pred)
XGBPredictions = xgbm.predict(dtest_pred)



## === cell 32
XGBPredictions



## === cell 33
XGB_submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": XGBPredictions}, columns=["key", "fare_amount"]
)
XGB_submission.head()



## === cell 34
submission = XGB_submission

submission["fare_amount"] = submission["fare_amount"].clip(lower=0, upper=200)

submission.to_csv("submission.csv", index=False)
print("Wrote submission:", "submission.csv", "rows:", len(submission))
print(submission.head())
