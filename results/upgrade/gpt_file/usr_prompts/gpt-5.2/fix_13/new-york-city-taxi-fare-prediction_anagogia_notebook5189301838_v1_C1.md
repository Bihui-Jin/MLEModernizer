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

3.10

# 3. Installed packages

folium==0.20.0
geopandas==0.14.4
lightgbm==4.6.0
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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os




## === cell 1
import matplotlib.pyplot as plt
import seaborn as sns



## === cell 2
train = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
    nrows=2000000,
    parse_dates=["pickup_datetime"],
    dtype={
        "fare_amount": "float32",
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "uint8",
    },
)



## === cell 3
train.head()



## === cell 4
train.describe()



## === cell 5
required = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
train = train.dropna(subset=required)

LAT_MIN, LAT_MAX = 40.0, 42.0
LON_MIN, LON_MAX = -75.5, -72.0

mask = np.ones(len(train), dtype=bool)

mask &= ~(
    (train["pickup_longitude"] == 0)
    & (train["pickup_latitude"] == 0)
    & (train["dropoff_longitude"] == 0)
    & (train["dropoff_latitude"] == 0)
)

mask &= train["pickup_latitude"].between(LAT_MIN, LAT_MAX).to_numpy()
mask &= train["pickup_longitude"].between(LON_MIN, LON_MAX).to_numpy()
mask &= train["dropoff_latitude"].between(LAT_MIN, LAT_MAX).to_numpy()
mask &= train["dropoff_longitude"].between(LON_MIN, LON_MAX).to_numpy()

mask &= train["fare_amount"].between(2.5, 250.0).to_numpy()
mask &= train["passenger_count"].between(1, 6).to_numpy()

train = train.loc[mask].reset_index(drop=True)



## === cell 6
print(train.isnull().sum())



## === cell 9
train = train.loc[train["passenger_count"] <= 6].reset_index(drop=True)



## === cell 11
new_york = None



## === cell 12
new_york



## === cell 13
pass



## === cell 14
pass



## === cell 15
new_york



## === cell 16
train["year"] = train.pickup_datetime.dt.year.astype("int16")
train["month"] = train.pickup_datetime.dt.month.astype("int8")
train["day"] = train.pickup_datetime.dt.day.astype("int8")
train["weekday"] = train.pickup_datetime.dt.weekday.astype("int8")
train["hour"] = train.pickup_datetime.dt.hour.astype("int8")



## === cell 17
train.head()




## === cell 18
def distance(lat1, lon1, lat2, lon2):
    p = 0.0174532925199432295
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 12742 * np.arcsin(np.sqrt(a))


train["distance"] = distance(
    train["pickup_latitude"].to_numpy(),
    train["pickup_longitude"].to_numpy(),
    train["dropoff_latitude"].to_numpy(),
    train["dropoff_longitude"].to_numpy(),
).astype("float32")

train.head()



## === cell 19
pass



## === cell 20
mask = (train["distance"] > 0).to_numpy()
mask &= (train["distance"] < 200.0).to_numpy()

fare_per_km = (train["fare_amount"] / (train["distance"] + 1e-6)).to_numpy()
mask &= (fare_per_km >= 0.8) & (fare_per_km <= 50.0)

mask &= ~(((train["distance"] > 60.0) & (train["fare_amount"] < 30.0)).to_numpy())

train = train.loc[mask].reset_index(drop=True)



## === cell 21
dt = train["pickup_datetime"].copy()



## === cell 22
del train["pickup_datetime"]
del train["key"]



## === cell 23
from sklearnex import patch_sklearn

patch_sklearn()

from sklearn.metrics import mean_squared_error

y = train["fare_amount"]
X = train.drop(columns=["fare_amount"])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=50
)



## === cell 24
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print(mean_squared_error(y_test, y_pred) ** 0.5)  # RMSE



## === cell 25
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print(mean_squared_error(y_test, y_pred) ** 0.5)  # RMSE



## === cell 26
import lightgbm as lgb



## === cell 27
parameters = {
    "learning_rate": 0.05,
    "objective": "regression_l2",
    "max_depth": 6,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "verbosity": -1,
    "metric": "rmse",
    "seed": 50,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "feature_pre_filter": False,
    "num_threads": max(1, os.cpu_count() or 1),
}



## === cell 28
from sklearn.model_selection import KFold

X_all = X
y_all = y

feature_names = list(X_all.columns)
X_all_np = X_all.to_numpy(dtype=np.float32, copy=False)
y_all_np = y_all.to_numpy(dtype=np.float32, copy=False)

kf = KFold(n_splits=3, shuffle=True, random_state=50)

best_iters = []
oof_pred = np.zeros(len(X_all_np), dtype=np.float64)

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_all_np), start=1):
    X_tr, X_va = X_all_np[tr_idx], X_all_np[va_idx]
    y_tr, y_va = y_all_np[tr_idx], y_all_np[va_idx]

    train_set = lgb.Dataset(
        X_tr,
        label=y_tr,
        free_raw_data=False,
        feature_name=feature_names,
    )
    valid_set = lgb.Dataset(
        X_va,
        label=y_va,
        reference=train_set,
        free_raw_data=False,
        feature_name=feature_names,
    )

    booster = lgb.train(
        params=parameters,
        train_set=train_set,
        num_boost_round=7000,
        valid_sets=[valid_set],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)],
    )

    best_iters.append(int(booster.best_iteration))
    oof_pred[va_idx] = booster.predict(X_va, num_iteration=booster.best_iteration)

cv_rmse = mean_squared_error(y_all_np, oof_pred) ** 0.5
best_iter_final = int(np.median(best_iters))

print("CV RMSE (OOF):", cv_rmse)
print("Chosen best_iteration (median over folds):", best_iter_final)

final_train_set = lgb.Dataset(
    X_all_np,
    label=y_all_np,
    free_raw_data=False,
    feature_name=feature_names,
)
lb = lgb.train(
    params=parameters,
    train_set=final_train_set,
    num_boost_round=best_iter_final,
    valid_sets=[final_train_set],
    valid_names=["train"],
)



## === cell 29
y_pred = lb.predict(
    X_test.to_numpy(dtype=np.float32, copy=False), num_iteration=best_iter_final
)
print(mean_squared_error(y_test, y_pred) ** 0.5)  # RMSE



## === cell 30
test = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
    parse_dates=["pickup_datetime"],
    dtype={
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "uint8",
    },
)



## === cell 31
test.head()



## === cell 32
test["year"] = test.pickup_datetime.dt.year.astype("int16")
test["month"] = test.pickup_datetime.dt.month.astype("int8")
test["day"] = test.pickup_datetime.dt.day.astype("int8")
test["weekday"] = test.pickup_datetime.dt.weekday.astype("int8")
test["hour"] = test.pickup_datetime.dt.hour.astype("int8")



## === cell 33
test["distance"] = distance(
    test["pickup_latitude"].to_numpy(),
    test["pickup_longitude"].to_numpy(),
    test["dropoff_latitude"].to_numpy(),
    test["dropoff_longitude"].to_numpy(),
).astype("float32")



## === cell 34
test.head()



## === cell 35
test_mask_valid = (
    test["pickup_latitude"].between(LAT_MIN, LAT_MAX)
    & test["pickup_longitude"].between(LON_MIN, LON_MAX)
    & test["dropoff_latitude"].between(LAT_MIN, LAT_MAX)
    & test["dropoff_longitude"].between(LON_MIN, LON_MAX)
    & ~(
        (test["pickup_longitude"] == 0)
        & (test["pickup_latitude"] == 0)
        & (test["dropoff_longitude"] == 0)
        & (test["dropoff_latitude"] == 0)
    )
)

x_test_all = test.drop(["key", "pickup_datetime"], axis=1)
x_test_all = x_test_all.reindex(columns=feature_names, fill_value=0)

predictions = np.empty(len(test), dtype=np.float64)

fallback = float(np.median(y_all_np))
predictions[:] = fallback

if test_mask_valid.any():
    x_valid_np = x_test_all.loc[test_mask_valid].to_numpy(dtype=np.float32, copy=False)
    preds_valid = lb.predict(x_valid_np, num_iteration=best_iter_final)
    predictions[test_mask_valid.to_numpy()] = preds_valid

predictions = np.clip(predictions, 0.0, 250.0)



## === cell 36
test_keys = test["key"]
dataframe = pd.DataFrame({"key": test_keys.values, "fare_amount": predictions})



## === cell 37
dataframe.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", dataframe.shape)
print(dataframe.head())
