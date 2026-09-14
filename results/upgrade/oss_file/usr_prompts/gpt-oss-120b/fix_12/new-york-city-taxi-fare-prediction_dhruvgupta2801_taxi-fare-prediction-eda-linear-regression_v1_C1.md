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

3.8

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
from pathlib import Path

import numpy as np
import pandas as pd


def locate_file(filename: str) -> str:
    """
    Return a path to *filename* searching common Kaggle data folders,
    including the competition sub‑directory if present, and the standard
    /kaggle/input path used in the competition environment.
    """
    possible_dirs = [
        Path.cwd() / "data",
        Path.cwd() / "kaggle" / "data",
        Path.cwd() / "kaggle" / "input",
        Path.cwd() / "kaggle" / "working",
        Path.cwd() / "working",
        Path.cwd().parent / "input",  # typical Kaggle sibling folder
        Path("/kaggle/input"),  # absolute Kaggle input path
    ]
    for d in possible_dirs:
        candidate = d / filename
        if candidate.is_file():
            return str(candidate)
        sub_candidate = d / "new-york-city-taxi-fare-prediction" / filename
        if sub_candidate.is_file():
            return str(sub_candidate)
    raise FileNotFoundError(f"Could not locate {filename} in typical data directories.")




## === cell 1
train_path = locate_file("train.csv")
train_data = pd.read_csv(
    train_path,
    nrows=5_000_000,  # reduced from full set for speed
    dtype={
        "key": "object",
        "fare_amount": "float32",
        "pickup_datetime": "object",
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "int8",
    },
)
train_data = train_data.sample(frac=0.30, random_state=42).reset_index(drop=True)




## === cell 2
print("Train shape after sampling:", train_data.shape)




## === cell 3
test_path = locate_file("test.csv")
test_data = pd.read_csv(
    test_path,
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




## === cell 4
print("Test shape:", test_data.shape)




## === cell 5
train_data["Difference_longitude"] = np.abs(
    train_data["pickup_longitude"] - train_data["dropoff_longitude"]
)
train_data["Difference_latitude"] = np.abs(
    train_data["pickup_latitude"] - train_data["dropoff_latitude"]
)

test_data["Difference_longitude"] = np.abs(
    test_data["pickup_longitude"] - test_data["dropoff_longitude"]
)
test_data["Difference_latitude"] = np.abs(
    test_data["pickup_latitude"] - test_data["dropoff_latitude"]
)




## === cell 6
print(f"Before dropping null values: {len(train_data)}")
train_data.dropna(inplace=True)
print(f"After dropping null values: {len(train_data)}")




## === cell 7
train_data = train_data[
    (train_data["Difference_longitude"] < 5.0)
    & (train_data["Difference_latitude"] < 5.0)
]




## === cell 8
train_dt = pd.to_datetime(train_data["pickup_datetime"])
train_data["pickuptime"] = train_dt.dt.hour * 100 + train_dt.dt.minute

test_dt = pd.to_datetime(test_data["pickup_datetime"])
test_data["pickuptime"] = test_dt.dt.hour * 100 + test_dt.dt.minute




## === cell 9
train_data["Weekday"] = train_dt.dt.weekday
test_data["Weekday"] = test_dt.dt.weekday




## === cell 10
train_data.drop("pickup_datetime", inplace=True, axis=1)
test_data.drop("pickup_datetime", inplace=True, axis=1)




## === cell 11
weekday_map = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}
train_data["Weekday"] = train_data["Weekday"].map(weekday_map)
test_data["Weekday"] = test_data["Weekday"].map(weekday_map)




## === cell 12
train_one_hot = pd.get_dummies(train_data["Weekday"], prefix="Weekday")
test_one_hot = pd.get_dummies(test_data["Weekday"], prefix="Weekday")
train_data = pd.concat([train_data, train_one_hot], axis=1)
test_data = pd.concat([test_data, test_one_hot], axis=1)

missing_in_test = set(train_one_hot.columns) - set(test_one_hot.columns)
for col in missing_in_test:
    test_data[col] = 0
test_data = test_data.reindex(columns=train_data.columns, fill_value=0)

train_data.drop("Weekday", axis=1, inplace=True)
test_data.drop("Weekday", axis=1, inplace=True)




## === cell 13
R = 6373.0  # Earth radius in km
lat1 = np.radians(train_data["pickup_latitude"].astype(np.float32))
lon1 = np.radians(train_data["pickup_longitude"].astype(np.float32))
lat2 = np.radians(train_data["dropoff_latitude"].astype(np.float32))
lon2 = np.radians(train_data["dropoff_longitude"].astype(np.float32))

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
train_data["Distance"] = (distance * 0.621).astype(np.float32)  # km → miles

lat1 = np.radians(test_data["pickup_latitude"].astype(np.float32))
lon1 = np.radians(test_data["pickup_longitude"].astype(np.float32))
lat2 = np.radians(test_data["dropoff_latitude"].astype(np.float32))
lon2 = np.radians(test_data["dropoff_longitude"].astype(np.float32))

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
test_data["Distance"] = (distance * 0.621).astype(np.float32)




## === cell 14
lat_air = np.radians(40.6413111).astype(np.float32)
lon_air = np.radians(-73.7781391).astype(np.float32)

lat1 = np.radians(train_data["pickup_latitude"].astype(np.float32))
lon1 = np.radians(train_data["pickup_longitude"].astype(np.float32))
dlon_pickup = lon_air - lon1
dlat_pickup = lat_air - lat1
a1 = (
    np.sin(dlat_pickup / 2) ** 2
    + np.cos(lat1) * np.cos(lat_air) * np.sin(dlon_pickup / 2) ** 2
)
c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
train_data["Pickup_Distance_airport"] = (R * c1 * 0.621).astype(np.float32)

lat2 = np.radians(train_data["dropoff_latitude"].astype(np.float32))
lon2 = np.radians(train_data["dropoff_longitude"].astype(np.float32))
dlon_dropoff = lon_air - lon2
dlat_dropoff = lat_air - lat2
a2 = (
    np.sin(dlat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat_air) * np.sin(dlon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
train_data["Dropoff_Distance_airport"] = (R * c2 * 0.621).astype(np.float32)

lat1 = np.radians(test_data["pickup_latitude"].astype(np.float32))
lon1 = np.radians(test_data["pickup_longitude"].astype(np.float32))
dlon_pickup = lon_air - lon1
dlat_pickup = lat_air - lat1
a1 = (
    np.sin(dlat_pickup / 2) ** 2
    + np.cos(lat1) * np.cos(lat_air) * np.sin(dlon_pickup / 2) ** 2
)
c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
test_data["Pickup_Distance_airport"] = (R * c1 * 0.621).astype(np.float32)

lat2 = np.radians(test_data["dropoff_latitude"].astype(np.float32))
lon2 = np.radians(test_data["dropoff_longitude"].astype(np.float32))
dlon_dropoff = lon_air - lon2
dlat_dropoff = lat_air - lat2
a2 = (
    np.sin(dlat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat_air) * np.sin(dlon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
test_data["Dropoff_Distance_airport"] = (R * c2 * 0.621).astype(np.float32)




## === cell 15
numeric_cols = ["Distance", "Pickup_Distance_airport", "Dropoff_Distance_airport"]
for col in numeric_cols:
    train_data[col] = np.round(train_data[col], 2).astype(np.float32)
    test_data[col] = np.round(test_data[col], 2).astype(np.float32)




## === cell 16
for df in (train_data, test_data):
    df["Difference_longitude"] = np.abs(
        df["Difference_longitude"] - np.mean(df["Difference_longitude"])
    )
    df["Difference_longitude"] = (
        df["Difference_longitude"] / np.var(df["Difference_longitude"])
    ).astype(np.float32)

    df["Difference_latitude"] = np.abs(
        df["Difference_latitude"] - np.mean(df["Difference_latitude"])
    )
    df["Difference_latitude"] = (
        df["Difference_latitude"] / np.var(df["Difference_latitude"])
    ).astype(np.float32)




## === cell 17
print("Final train shape:", train_data.shape)
print("Final test shape:", test_data.shape)




## === cell 18
try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

X = train_data.drop(["key", "fare_amount"], axis=1).astype(np.float32)
y = train_data["fare_amount"].astype(np.float32)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.01, random_state=80)

rf = RandomForestRegressor(
    n_estimators=400,
    max_depth=25,
    n_jobs=-1,
    random_state=42,
    min_samples_leaf=1,
    min_samples_split=2,
)
rf.fit(X_train.values, y_train)

val_pred = rf.predict(X_val.values)
rmse = mean_squared_error(y_val, val_pred, squared=False)
print(f"Validation RMSE: {rmse:.4f}")




## === cell 19
test_features = test_data.drop("key", axis=1).values.astype(np.float32, copy=False)
pred_raw = rf.predict(test_features)

pred = np.round(np.clip(pred_raw, a_min=0, a_max=None), 2)




## === cell 20
Submission = pd.DataFrame({"key": test_data["key"], "fare_amount": pred})
Submission = Submission[["key", "fare_amount"]]
Submission.to_csv("Submission.csv", index=False)
print("Submission saved to Submission.csv")
