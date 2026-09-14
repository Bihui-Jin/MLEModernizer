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

# 5. Target score

5.6891

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 936.92806) has done: 'I remove the deprecated `normalize` argument from `LinearRegression`, align the test feature columns with the training columns to avoid shape mismatches, and then use the fitted model to generate predictions and write a proper `Submission.csv` containing the required `key` and `fare_amount` columns.'
- What this solution (achieved 7.13833) has done: 'The changes reduce the data size read (from 10 M to 2 M rows) and cast numeric columns to `float32` after dropping unnecessary latitude/longitude fields, which cuts memory use and speeds up the RandomForest training while keeping the same preprocessing, feature engineering, model type, and evaluation logic. Seed settings are added for reproducibility. All modifications are limited to the original cells and preserve the core algorithm.'

# 9. Code solution

## === cell 0
import os

os.environ["SKLEARN_EX_NUM_THREADS"] = "5"  # match rf n_jobs
from sklearnex import patch_sklearn

patch_sklearn()

import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

dtype_map = {
    "key": "object",
    "fare_amount": "float32",
    "pickup_datetime": "object",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
train_df = pd.read_csv(
    "../input/train.csv",
    nrows=1_000_000,
    dtype=dtype_map,
    low_memory=False,
)
train_df.dtypes



## === cell 1
test_df = pd.read_csv(
    "../input/test.csv",
    dtype={
        "key": "object",
        "pickup_datetime": "object",
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "int8",
    },
)
test_df.dtypes



## === cell 2
print(f"Rows before dropping NaNs: {len(train_df)}")
train_df.dropna(inplace=True)
print(f"Rows after dropping NaNs: {len(train_df)}")




## === cell 3
def add_travel_vector_features(df):
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs()


add_travel_vector_features(train_df)
add_travel_vector_features(test_df)



## === cell 4
print(f"Rows before distance filters: {len(train_df)}")
train_df = train_df[
    (train_df.abs_diff_longitude < 5.0) & (train_df.abs_diff_latitude < 5.0)
]
print(f"Rows after distance filters: {len(train_df)}")




## === cell 5
def process_datetime(df):
    dt = pd.to_datetime(df["pickup_datetime"])
    df["pickuptime"] = dt.dt.hour * 100 + dt.dt.minute
    df["Weekday"] = dt.dt.weekday


process_datetime(train_df)
process_datetime(test_df)

day_names = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}
train_df["Weekday"] = train_df["Weekday"].map(day_names)
test_df["Weekday"] = test_df["Weekday"].map(day_names)

combined_weekday = pd.concat([train_df[["Weekday"]], test_df[["Weekday"]]], axis=0)
weekday_dummies = pd.get_dummies(combined_weekday["Weekday"], dtype=np.uint8)

train_dummies = weekday_dummies.iloc[: len(train_df)].reset_index(drop=True)
test_dummies = weekday_dummies.iloc[len(train_df) :].reset_index(drop=True)

train_df = pd.concat([train_df.reset_index(drop=True), train_dummies], axis=1)
test_df = pd.concat([test_df.reset_index(drop=True), test_dummies], axis=1)

train_df.drop("Weekday", axis=1, inplace=True)
test_df.drop("Weekday", axis=1, inplace=True)



## === cell 6
pass



## === cell 7
pass



## === cell 8
pass




## === cell 9
def finding_distance(df):
    R = 6373.0
    lat1 = np.radians(df["pickup_latitude"].values)
    lon1 = np.radians(df["pickup_longitude"].values)
    lat2 = np.radians(df["dropoff_latitude"].values)
    lon2 = np.radians(df["dropoff_longitude"].values)

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    df["Distance"] = distance * 0.621  # miles


finding_distance(train_df)
finding_distance(test_df)




## === cell 10
def creating_pickup_dropoff_distance(df):
    R = 6373.0
    lat1 = np.radians(df["pickup_latitude"].values)
    lon1 = np.radians(df["pickup_longitude"].values)
    lat2 = np.radians(df["dropoff_latitude"].values)
    lon2 = np.radians(df["dropoff_longitude"].values)

    lat3 = np.radians(40.6413111)
    lon3 = np.radians(-73.7781391)

    dlon_pickup = lon3 - lon1
    dlat_pickup = lat3 - lat1
    a1 = (
        np.sin(dlat_pickup / 2) ** 2
        + np.cos(lat1) * np.cos(lat3) * np.sin(dlon_pickup / 2) ** 2
    )
    c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
    df["Pickup_Distance_airport"] = R * c1 * 0.621

    dlon_dropoff = lon3 - lon2
    dlat_dropoff = lat3 - lat2
    a2 = (
        np.sin(dlat_dropoff / 2) ** 2
        + np.cos(lat2) * np.cos(lat3) * np.sin(dlon_dropoff / 2) ** 2
    )
    c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
    df["Dropoff_Distance_airport"] = R * c2 * 0.621


creating_pickup_dropoff_distance(train_df)
creating_pickup_dropoff_distance(test_df)



## === cell 11
train_numeric_cols = train_df.select_dtypes(
    include=["float64", "int64", "float32", "int8"]
).columns
train_df[train_numeric_cols] = train_df[train_numeric_cols].astype(np.float32)

test_numeric_cols = test_df.select_dtypes(
    include=["float64", "int64", "float32", "int8"]
).columns
test_df[test_numeric_cols] = test_df[test_numeric_cols].astype(np.float32)



## === cell 12
train_df["abs_diff_longitude"] = np.abs(
    train_df["abs_diff_longitude"] - np.mean(train_df["abs_diff_longitude"])
)
train_df["abs_diff_longitude"] /= np.var(train_df["abs_diff_longitude"])



## === cell 13
train_df["abs_diff_latitude"] = np.abs(
    train_df["abs_diff_latitude"] - np.mean(train_df["abs_diff_latitude"])
)
train_df["abs_diff_latitude"] /= np.var(train_df["abs_diff_latitude"])



## === cell 14
test_df["abs_diff_longitude"] = np.abs(
    test_df["abs_diff_longitude"] - np.mean(test_df["abs_diff_longitude"])
)
test_df["abs_diff_longitude"] /= np.var(test_df["abs_diff_longitude"])

test_df["abs_diff_latitude"] = np.abs(
    test_df["abs_diff_latitude"] - np.mean(test_df["abs_diff_latitude"])
)
test_df["abs_diff_latitude"] /= np.var(test_df["abs_diff_latitude"])



## === cell 15
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

X = train_df.drop(["key", "fare_amount"], axis=1)
y = train_df["fare_amount"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.01, random_state=80)

X_train_np = X_train.to_numpy(dtype=np.float32, copy=False)
X_val_np = X_val.to_numpy(dtype=np.float32, copy=False)
y_train_np = y_train.to_numpy(dtype=np.float32, copy=False)

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    random_state=42,
    n_jobs=5,
)
rf.fit(X_train_np, y_train_np)

val_pred = rf.predict(X_val_np)
val_rmse = mean_squared_error(y_val, val_pred, squared=False)
print(f"Validation RMSE: {val_rmse:.4f}")

test_features = test_df.drop(["key"], axis=1).reindex(columns=X.columns, fill_value=0)
test_features_np = test_features.to_numpy(dtype=np.float32, copy=False)
test_pred = rf.predict(test_features_np)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_10/537674626.py in <cell line: 0>()
      8 X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.01, random_state=80)
      9 
---> 10 X_train_np = X_train.to_numpy(dtype=np.float32, copy=False)
     11 X_val_np = X_val.to_numpy(dtype=np.float32, copy=False)
     12 y_train_np = y_train.to_numpy(dtype=np.float32, copy=False)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in to_numpy(self, dtype, copy, na_value)
   1991         if dtype is not None:
   1992             dtype = np.dtype(dtype)
-> 1993         result = self._mgr.as_array(dtype=dtype, copy=copy, na_value=na_value)
   1994         if result.dtype is not dtype:
   1995             result = np.asarray(result, dtype=dtype)

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in as_array(self, dtype, copy, na_value)
   1692                 arr.flags.writeable = False
   1693         else:
-> 1694             arr = self._interleave(dtype=dtype, na_value=na_value)
   1695             # The underlying data was copied within _interleave, so no need
   1696             # to further copy if copy=True or setting na_value

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in _interleave(self, dtype, na_value)
   1751             else:
   1752                 arr = blk.get_values(dtype)
-> 1753             result[rl.indexer] = arr
   1754             itemmask[rl.indexer] = 1
   1755 

ValueError: could not convert string to float: '2009-12-12 22:28:00 UTC'

## === cell 16
pred = np.round(test_pred, 2)
print(pred[:5])



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/1972775025.py in <cell line: 0>()
----> 1 pred = np.round(test_pred, 2)
      2 print(pred[:5])
      3 

NameError: name 'test_pred' is not defined

## === cell 17
Submission = pd.DataFrame({"key": test_df["key"], "fare_amount": pred})



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/3278686340.py in <cell line: 0>()
----> 1 Submission = pd.DataFrame({"key": test_df["key"], "fare_amount": pred})
      2 

NameError: name 'pred' is not defined

## === cell 18
Submission.set_index("key", inplace=True)



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/109248162.py in <cell line: 0>()
----> 1 Submission.set_index("key", inplace=True)
      2 

NameError: name 'Submission' is not defined

## === cell 19
Submission.to_csv("Submission.csv")
print("Submission file 'Submission.csv' written successfully.")

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/1303630020.py in <cell line: 0>()
----> 1 Submission.to_csv("Submission.csv")
      2 print("Submission file 'Submission.csv' written successfully.")

NameError: name 'Submission' is not defined
