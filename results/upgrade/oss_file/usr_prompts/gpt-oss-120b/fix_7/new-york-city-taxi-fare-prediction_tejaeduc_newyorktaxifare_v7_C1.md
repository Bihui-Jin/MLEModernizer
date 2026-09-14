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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn.model_selection import train_test_split
from math import radians
import xgboost
from sklearn.preprocessing import StandardScaler
import os

print(os.listdir("../input"))




## === cell 1
dtype_dict = {
    "key": "object",
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
df = pd.read_csv("../input/train.csv", dtype=dtype_dict, low_memory=False)




## === cell 2
test = pd.read_csv(
    "../input/test.csv",
    dtype={k: v for k, v in dtype_dict.items() if k != "fare_amount"},
)




## === cell 3
testkey = test.key.copy()




## === cell 4
df = df.dropna(how="any", axis="rows")




## === cell 5
len(df)  # retained for possible logging; no effect on runtime




## === cell 6
pass  # removed df.head() display




## === cell 7
pass  # removed df.describe() display




## === cell 8
geo_mask = (
    (df.pickup_latitude > 42.5)
    | (df.pickup_latitude < 40.0)
    | (df.dropoff_latitude > 42.5)
    | (df.dropoff_latitude < 40.0)
    | (df.pickup_longitude > -73.0)
    | (df.pickup_longitude < -75.0)
    | (df.dropoff_longitude > -73.0)
    | (df.dropoff_longitude < -75.0)
)
df = df[~geo_mask]




## === cell 9
fpay_mask = (
    (df.fare_amount > 350.0)
    | (df.fare_amount < 0.0)
    | (df.passenger_count > 7)
    | (df.passenger_count < 0)
)
df = df[~fpay_mask]




## === cell 10
len(df)  # retained for possible logging




## === cell 11
pass  # placeholder to keep cell numbering (originally empty after removal)




## === cell 12
def haversine_vec(lon1, lat1, lon2, lat2):
    """Vectorized haversine distance in kilometers."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return 6373.0 * c




## === cell 13
df["dist"] = haversine_vec(
    df["pickup_longitude"].values,
    df["pickup_latitude"].values,
    df["dropoff_longitude"].values,
    df["dropoff_latitude"].values,
)




## === cell 14
test["dist"] = haversine_vec(
    test["pickup_longitude"].values,
    test["pickup_latitude"].values,
    test["dropoff_longitude"].values,
    test["dropoff_latitude"].values,
)




## === cell 15
df["abs_lat_diff"] = (df["dropoff_latitude"] - df["pickup_latitude"]).abs()
df["abs_lon_diff"] = (df["dropoff_longitude"] - df["pickup_longitude"]).abs()
test["abs_lat_diff"] = (test["dropoff_latitude"] - test["pickup_latitude"]).abs()
test["abs_lon_diff"] = (test["dropoff_longitude"] - test["pickup_longitude"]).abs()




## === cell 16
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])




## === cell 17
test["pickup_datetime"] = pd.to_datetime(test["pickup_datetime"])




## === cell 18
pass  # removed df.info() display




## === cell 19
df["latenights"] = (df["pickup_datetime"].dt.hour < 5).astype(np.int8)
test["latenights"] = (test["pickup_datetime"].dt.hour < 5).astype(np.int8)




## === cell 20
df["weekday"] = (df["pickup_datetime"].dt.weekday > 4).astype(np.int8)
test["weekday"] = (test["pickup_datetime"].dt.weekday > 4).astype(np.int8)
df["hour"] = df["pickup_datetime"].dt.hour.astype(np.int8)
test["hour"] = test["pickup_datetime"].dt.hour.astype(np.int8)
df["month"] = df["pickup_datetime"].dt.month.astype(np.int8)
test["month"] = test["pickup_datetime"].dt.month.astype(np.int8)

df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
test["hour_sin"] = np.sin(2 * np.pi * test["hour"] / 24)
test["hour_cos"] = np.cos(2 * np.pi * test["hour"] / 24)

df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
test["month_sin"] = np.sin(2 * np.pi * test["month"] / 12)
test["month_cos"] = np.cos(2 * np.pi * test["month"] / 12)




## === cell 21
pass  # removed df.head() display




## === cell 22
df["year"] = df["pickup_datetime"].dt.year.astype(np.int16)




## === cell 23
test["year"] = test["pickup_datetime"].dt.year.astype(np.int16)




## === cell 24
df["day"] = df["pickup_datetime"].dt.day.astype(np.int8)




## === cell 25
test["day"] = test["pickup_datetime"].dt.day.astype(np.int8)




## === cell 26
pass  # removed df.head() display




## === cell 27
pass  # removed test.head() display




## === cell 28
feat = df.drop(["key", "pickup_datetime"], axis=1)
test = test.drop(["key", "pickup_datetime"], axis=1)




## === cell 29
pass  # removed test.year.unique() call




## === cell 30
feat = pd.concat([feat, pd.get_dummies(feat["year"], prefix="year")], axis=1)
test = pd.concat([test, pd.get_dummies(test["year"], prefix="year")], axis=1)




## === cell 31
feat = feat.drop("year", axis=1)




## === cell 32
test = test.drop("year", axis=1)




## === cell 33
missing_cols = set(feat.columns) - set(test.columns)
for c in missing_cols:
    test[c] = 0
test = test[feat.columns]  # ensure same column order




## === cell 34
pass  # removed test.head() display




## === cell 35
label = np.log1p(feat["fare_amount"].astype(np.float32))
feat = feat.drop("fare_amount", axis=1)




## === cell 36
xtr, xts, ytr, yts = train_test_split(feat, label, random_state=42)




## === cell 37
xgbtrain = xgboost.DMatrix(xtr.values.astype(np.float32), label=ytr.values)
xgbtest = xgboost.DMatrix(xts.values.astype(np.float32), label=yts.values)
xgbfinaltest = xgboost.DMatrix(test.values.astype(np.float32))




## === cell 38
params = {
    "eval_metric": "rmse",
    "objective": "reg:squarederror",
    "max_depth": 5,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "min_child_weight": 1,
    "tree_method": "hist",  # fast histogram algorithm
    "nthread": -1,  # use all CPU cores
}




## === cell 39
xgbmodel = xgboost.train(
    params,
    dtrain=xgbtrain,
    num_boost_round=2000,
    early_stopping_rounds=50,
    evals=[(xgbtest, "test")],
)




## === cell 40
pred = xgbmodel.predict(xgbfinaltest)




## === cell 41
pred = np.expm1(pred)
pred = np.clip(pred, 0, None)
pred = np.round(pred, 2)




## === cell 42
finalset = pd.DataFrame({"key": testkey, "fare_amount": pred})




## === cell 43
finalset = finalset[["key", "fare_amount"]]




## === cell 44
finalset.head()




## === cell 45
finalset.to_csv("finaloutput.csv", index=False)
