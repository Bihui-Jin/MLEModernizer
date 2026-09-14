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
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

np.random.seed(42)



## === cell 1
fields = [
    "pickup_datetime",
    "fare_amount",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

train_dtypes = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16",
}

_read_csv_kwargs = dict(
    skipinitialspace=True,
    usecols=fields,
    parse_dates=["pickup_datetime"],
    dtype=train_dtypes,
)

try:
    train = pd.read_csv(
        "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
        nrows=1_000_000,
        engine="pyarrow",
        **_read_csv_kwargs,
    )
except Exception:
    train = pd.read_csv(
        "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
        nrows=1_000_000,
        **_read_csv_kwargs,
    )

print(f"{train.shape} shape")
train.head()



## === cell 2
test_usecols = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
test_dtypes = {
    "key": "string",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16",
}

try:
    test = pd.read_csv(
        "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
        usecols=test_usecols,
        parse_dates=["pickup_datetime"],
        dtype=test_dtypes,
        engine="pyarrow",
    )
except Exception:
    test = pd.read_csv(
        "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
        usecols=test_usecols,
        parse_dates=["pickup_datetime"],
        dtype=test_dtypes,
    )

print(f"{test.shape} shape")
test.head()



## === cell 3
test_key = test["key"].copy()



## === cell 4
mask_basic = (train["passenger_count"] != 208) & (train["fare_amount"] >= 0)

geo_mask = (
    (train["pickup_latitude"] > 40)
    & (train["pickup_latitude"] < 45)
    & (train["dropoff_latitude"] > 40)
    & (train["dropoff_latitude"] < 45)
    & (train["pickup_longitude"] < -71)
    & (train["pickup_longitude"] > -79)
    & (train["dropoff_longitude"] < -71)
    & (train["dropoff_longitude"] > -79)
)

mask_pass = (train["passenger_count"] >= 1) & (train["passenger_count"] <= 6)

train = train.loc[mask_basic & geo_mask & mask_pass]



## === cell 5
for df in (train, test):
    dt = df["pickup_datetime"].dt
    df["year"] = dt.year.astype("int16", copy=False)
    df["month"] = dt.month.astype("int8", copy=False)
    df["day"] = dt.day.astype("int8", copy=False)
    df["hour"] = dt.hour.astype("int8", copy=False)
    df["minute"] = dt.minute.astype("int8", copy=False)

train.drop(columns=["pickup_datetime"], inplace=True)
test.drop(columns=["pickup_datetime"], inplace=True)



## === cell 6
train.dropna(inplace=True)
test.dropna(inplace=True)

test["passenger_count"] = test["passenger_count"].clip(1, 6)

test_key = test["key"].copy()

train.shape, test.shape



## === cell 7
from sklearn.ensemble import GradientBoostingRegressor

X = train.drop(columns=["fare_amount"])
y = train["fare_amount"].astype(float)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_np = np.ascontiguousarray(X_train.to_numpy())
X_val_np = np.ascontiguousarray(X_val.to_numpy())
y_train_np = y_train.to_numpy()
y_val_np = y_val.to_numpy()

model = GradientBoostingRegressor(
    random_state=42,
    n_estimators=400,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
)
model.fit(X_train_np, y_train_np)

val_pred = model.predict(X_val_np)
rmse = mean_squared_error(y_val_np, val_pred, squared=False)
print("Validation RMSE:", rmse)



## === cell 8
feature_cols = X.columns
test_X = test[feature_cols]

test_X_np = np.ascontiguousarray(test_X.to_numpy())

test_pred = model.predict(test_X_np)
test_pred = np.clip(test_pred, 0, None)

submission = pd.DataFrame({"key": test_key.values, "fare_amount": test_pred})

print(submission.head())
print("Submission shape:", submission.shape)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
