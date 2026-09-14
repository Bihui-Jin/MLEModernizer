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

3.9

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

5.689

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 936.92806) has done: 'I fix the `LinearRegression` constructor by removing the deprecated `normalize` argument and align the one‑hot weekday columns between train and test so that the feature sets match. This resolves the runtime errors and guarantees a proper submission CSV is written.'
- What this solution (achieved 762.71854) has done: 'I remove the unnecessary scaling of the longitude/latitude difference features (cells 26‑28) so the model works with the original raw differences, and replace the plain LinearRegression with a ridge regression (cell 32) which adds modest regularisation and typically lowers RMSE on this data. These minimal changes keep the overall pipeline and feature engineering intact while moving the validation error closer to the target score.'
- What this solution (achieved 762.43239) has done: 'I fix the submission file format (write both columns without using the index) and add an explicit RMSE calculation on the validation split so we can see the true metric. I also wrap the ridge regression in a simple StandardScaler pipeline, which usually improves linear‑model performance without altering the core logic. These minimal changes keep the original feature engineering intact while moving the score toward the target.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import warnings
import gc  # added for explicit memory cleanup

from sklearnex import patch_all

patch_all()  # enables Intel‑optimized sklearn algorithms

from sklearnex.ensemble import GradientBoostingRegressor  # noqa: F401

warnings.filterwarnings("ignore")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/1978767209.py in <cell line: 0>()
      4 import gc  # added for explicit memory cleanup
      5 
----> 6 from sklearnex import patch_all
      7 
      8 patch_all()  # enables Intel‑optimized sklearn algorithms

ImportError: cannot import name 'patch_all' from 'sklearnex' (/usr/local/lib/python3.11/dist-packages/sklearnex/__init__.py)

## === cell 1
dtype_spec = {
    "key": "object",
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
train_data = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
    nrows=2_000_000,  # unchanged
    parse_dates=["pickup_datetime"],
    dtype=dtype_spec,
)
train_data.head()




## === cell 2
train_data.shape




## === cell 3
train_data.info()




## === cell 4
test_data = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
    parse_dates=["pickup_datetime"],
    dtype={
        "key": "object",
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "int8",
    },
)
test_data.head()




## === cell 5
test_data.info()




## === cell 6
train_data.isna().sum()




## === cell 7
train_data["Difference_longitude"] = np.abs(
    train_data["pickup_longitude"] - train_data["dropoff_longitude"]
).astype(np.float32)
train_data["Difference_latitude"] = np.abs(
    train_data["pickup_latitude"] - train_data["dropoff_latitude"]
).astype(np.float32)

test_data["Difference_longitude"] = np.abs(
    test_data["pickup_longitude"] - test_data["dropoff_longitude"]
).astype(np.float32)
test_data["Difference_latitude"] = np.abs(
    test_data["pickup_latitude"] - test_data["dropoff_latitude"]
).astype(np.float32)




## === cell 8
print(f"Before Dropping null values: {len(train_data)}")
train_data.dropna(inplace=True)
train_data = train_data[train_data["fare_amount"] > 0].copy()
print(f"After Dropping null values and non‑positive fares: {len(train_data)}")
gc.collect()  # free memory from dropped rows




## === cell 9
train_data = train_data[
    (train_data["Difference_longitude"] < 5.0)
    & (train_data["Difference_latitude"] < 5.0)
]
gc.collect()




## === cell 10
train_data["pickuptime"] = (
    train_data["pickup_datetime"].dt.hour * 100
    + train_data["pickup_datetime"].dt.minute
).astype(np.int16)
test_data["pickuptime"] = (
    test_data["pickup_datetime"].dt.hour * 100 + test_data["pickup_datetime"].dt.minute
).astype(np.int16)




## === cell 11
train_data.head()




## === cell 12
train_data["Weekday"] = train_data["pickup_datetime"].dt.weekday
test_data["Weekday"] = test_data["pickup_datetime"].dt.weekday




## === cell 13
train_data.head()




## === cell 14
test_data.head()




## === cell 15
train_data.drop("pickup_datetime", inplace=True, axis=1)
test_data.drop("pickup_datetime", inplace=True, axis=1)




## === cell 16
train_data["Weekday"].replace(
    to_replace=list(range(7)),
    value=[
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ],
    inplace=True,
)
test_data["Weekday"].replace(
    to_replace=list(range(7)),
    value=[
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ],
    inplace=True,
)




## === cell 17
train_one_hot = pd.get_dummies(train_data["Weekday"], dtype=np.float32)
test_one_hot = pd.get_dummies(test_data["Weekday"], dtype=np.float32)
train_data = pd.concat([train_data, train_one_hot], axis=1)
test_data = pd.concat([test_data, test_one_hot], axis=1)




## === cell 18
train_data.drop("Weekday", axis=1, inplace=True)
test_data.drop("Weekday", axis=1, inplace=True)

missing_in_test = (
    set(train_data.columns) - set(test_data.columns) - {"key", "fare_amount"}
)
for col in missing_in_test:
    test_data[col] = 0.0

extra_in_test = (
    set(test_data.columns) - set(train_data.columns) - {"key", "fare_amount"}
)
for col in extra_in_test:
    test_data.drop(columns=col, inplace=True)




## === cell 19
pass




## === cell 20
train_data.head()




## === cell 21
R = 6373.0
lat1 = np.radians(train_data["pickup_latitude"].values)
lon1 = np.radians(train_data["pickup_longitude"].values)
lat2 = np.radians(train_data["dropoff_latitude"].values)
lon2 = np.radians(train_data["dropoff_longitude"].values)

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
train_data["Distance"] = (distance * 0.621).astype(np.float32)

lat1 = np.radians(test_data["pickup_latitude"].values)
lon1 = np.radians(test_data["pickup_longitude"].values)
lat2 = np.radians(test_data["dropoff_latitude"].values)
lon2 = np.radians(test_data["dropoff_longitude"].values)

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
test_data["Distance"] = (distance * 0.621).astype(np.float32)




## === cell 22
R = 6373.0
lat1 = np.radians(train_data["pickup_latitude"].values)
lon1 = np.radians(train_data["pickup_longitude"].values)
lat2 = np.radians(train_data["dropoff_latitude"].values)
lon2 = np.radians(train_data["dropoff_longitude"].values)

lat3 = np.full(len(train_data), np.radians(40.6413111))
lon3 = np.full(len(train_data), np.radians(-73.7781391))
dlon_pickup = lon3 - lon1
dlat_pickup = lat3 - lat1
d_lon_dropoff = lon3 - lon2
d_lat_dropoff = lat3 - lat2

a1 = (
    np.sin(dlat_pickup / 2) ** 2
    + np.cos(lat1) * np.cos(lat3) * np.sin(dlon_pickup / 2) ** 2
)
c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
distance1 = R * c1
train_data["Pickup_Distance_airport"] = (distance1 * 0.621).astype(np.float32)

a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
train_data["Dropoff_Distance_airport"] = (distance2 * 0.621).astype(np.float32)

lat1 = np.radians(test_data["pickup_latitude"].values)
lon1 = np.radians(test_data["pickup_longitude"].values)
lat2 = np.radians(test_data["dropoff_latitude"].values)
lon2 = np.radians(test_data["dropoff_longitude"].values)

lat3 = np.full(len(test_data), np.radians(40.6413111))
lon3 = np.full(len(test_data), np.radians(-73.7781391))
dlon_pickup = lon3 - lon1
dlat_pickup = lat3 - lat1
d_lon_dropoff = lon3 - lon2
d_lat_dropoff = lat3 - lat2

a1 = (
    np.sin(dlat_pickup / 2) ** 2
    + np.cos(lat1) * np.cos(lat3) * np.sin(dlon_pickup / 2) ** 2
)
c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
distance1 = R * c1
test_data["Pickup_Distance_airport"] = (distance1 * 0.621).astype(np.float32)

a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
test_data["Dropoff_Distance_airport"] = (distance2 * 0.621).astype(np.float32)




## === cell 23
train_data["Distance"] = np.round(train_data["Distance"], 2).astype(np.float32)
train_data["Pickup_Distance_airport"] = np.round(
    train_data["Pickup_Distance_airport"], 2
).astype(np.float32)
train_data["Dropoff_Distance_airport"] = np.round(
    train_data["Dropoff_Distance_airport"], 2
).astype(np.float32)
test_data["Distance"] = np.round(test_data["Distance"], 2).astype(np.float32)
test_data["Pickup_Distance_airport"] = np.round(
    test_data["Pickup_Distance_airport"], 2
).astype(np.float32)
test_data["Dropoff_Distance_airport"] = np.round(
    test_data["Dropoff_Distance_airport"], 2
).astype(np.float32)




## === cell 24
train_data.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)
test_data.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)




## === cell 25
train_data.shape




## === cell 26
test_data.shape




## === cell 27
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X = train_data.drop(["key", "fare_amount"], axis=1)
y = train_data["fare_amount"]

X_np = X.to_numpy(dtype=np.float32, copy=False)
y_np = y.to_numpy(dtype=np.float32, copy=False)

del X, y, train_data
gc.collect()

X_train, X_val, y_train, y_val = train_test_split(
    X_np, y_np, test_size=0.2, random_state=80
)




## === cell 28
gbr = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    random_state=80,
)
gbr.fit(X_train, y_train)

val_pred = gbr.predict(X_val)
rmse = mean_squared_error(y_val, val_pred, squared=False)
print(f"Validation RMSE (raw target): {rmse:.4f}")




## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2208129053.py in <cell line: 0>()
----> 1 gbr = GradientBoostingRegressor(
      2     n_estimators=300,
      3     learning_rate=0.05,
      4     max_depth=4,
      5     subsample=0.8,

NameError: name 'GradientBoostingRegressor' is not defined

## === cell 29
test_features = test_data.drop("key", axis=1).to_numpy(dtype=np.float32, copy=False)
test_pred = gbr.predict(test_features)
test_pred = np.round(test_pred, 2)




## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1707169327.py in <cell line: 0>()
      1 test_features = test_data.drop("key", axis=1).to_numpy(dtype=np.float32, copy=False)
----> 2 test_pred = gbr.predict(test_features)
      3 test_pred = np.round(test_pred, 2)
      4 
      5 

NameError: name 'gbr' is not defined

## === cell 30
pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
).head()




## === cell 31
Submission = pd.DataFrame({"key": test_data["key"], "fare_amount": test_pred})




## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1131539982.py in <cell line: 0>()
----> 1 Submission = pd.DataFrame({"key": test_data["key"], "fare_amount": test_pred})
      2 
      3 

NameError: name 'test_pred' is not defined

## === cell 32
Submission.to_csv("submission.csv", index=False)

## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1428118525.py in <cell line: 0>()
----> 1 Submission.to_csv("submission.csv", index=False)

NameError: name 'Submission' is not defined
