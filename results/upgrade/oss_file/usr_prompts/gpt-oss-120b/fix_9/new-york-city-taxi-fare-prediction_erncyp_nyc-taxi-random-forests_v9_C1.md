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

3.41764

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 7.72263) has done: 'I add two informative features—`passenger_count` and `minute`—to the model input and increase the number of trees in the RandomForest from 20 to 100 to improve predictive power while keeping the same overall pipeline. These changes should lower the RMSE toward the target score.'
- What this solution (achieved 5.36121) has done: 'I fix the NaN‑related crashes by removing rows with non‑positive fares (which make `log1p` produce NaNs) and by filling missing “busyness” values before model training. These minimal fixes let the pipeline run end‑to‑end and generate a proper `submission.csv` without altering the core modeling logic.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


def find_data_path(filename: str) -> Path:
    """Return the first existing Path for filename in common Kaggle folders.
    Searches recursively inside the candidate directories."""
    candidates = [
        Path("./kaggle/data"),
        Path("./data"),
    ]
    for base in candidates:
        p = base / filename
        if p.exists():
            return p
        for p in base.rglob(filename):
            if p.is_file():
                return p
    raise FileNotFoundError(f"Could not find {filename} in expected locations.")




## === cell 1
usecols = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
dtypes = {
    "key": "object",
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
train_path = find_data_path("train.csv")
train_df = pd.read_csv(
    train_path,
    nrows=1_000_000,
    usecols=usecols,
    dtype=dtypes,
    low_memory=False,
)
train_df.dropna(inplace=True)
train_df = train_df[train_df["fare_amount"] > 0]




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/709557093.py in <cell line: 0>()
     18     "passenger_count": "int8",
     19 }
---> 20 train_path = find_data_path("train.csv")
     21 train_df = pd.read_csv(
     22     train_path,

/tmp/ipykernel_11/3846850250.py in find_data_path(filename)
     24             if p.is_file():
     25                 return p
---> 26     raise FileNotFoundError(f"Could not find {filename} in expected locations.")
     27 
     28 

FileNotFoundError: Could not find train.csv in expected locations.

## === cell 2
def haversine_np(lon1, lat1, lon2, lat2):
    """
    Compute great‑circle distance between two points (in km).
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km.astype(np.float32)  # keep memory light




## === cell 3
train_df["distance"] = haversine_np(
    train_df["pickup_longitude"],
    train_df["pickup_latitude"],
    train_df["dropoff_longitude"],
    train_df["dropoff_latitude"],
)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/844419457.py in <cell line: 0>()
      1 train_df["distance"] = haversine_np(
----> 2     train_df["pickup_longitude"],
      3     train_df["pickup_latitude"],
      4     train_df["dropoff_longitude"],
      5     train_df["dropoff_latitude"],

NameError: name 'train_df' is not defined

## === cell 4
train_df["pickup_datetime"] = pd.to_datetime(train_df["pickup_datetime"])




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3908905509.py in <cell line: 0>()
----> 1 train_df["pickup_datetime"] = pd.to_datetime(train_df["pickup_datetime"])
      2 
      3 

NameError: name 'train_df' is not defined

## === cell 5
train_df["year"] = train_df["pickup_datetime"].dt.year.astype(np.int16)
train_df["month"] = train_df["pickup_datetime"].dt.month.astype(np.int8)
train_df["day"] = train_df["pickup_datetime"].dt.day.astype(np.int8)
train_df["hour"] = train_df["pickup_datetime"].dt.hour.astype(np.int8)
train_df["minute"] = train_df["pickup_datetime"].dt.minute.astype(np.int8)
train_df["day_of_week"] = train_df["pickup_datetime"].dt.dayofweek.astype(np.int8)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/557015549.py in <cell line: 0>()
----> 1 train_df["year"] = train_df["pickup_datetime"].dt.year.astype(np.int16)
      2 train_df["month"] = train_df["pickup_datetime"].dt.month.astype(np.int8)
      3 train_df["day"] = train_df["pickup_datetime"].dt.day.astype(np.int8)
      4 train_df["hour"] = train_df["pickup_datetime"].dt.hour.astype(np.int8)
      5 train_df["minute"] = train_df["pickup_datetime"].dt.minute.astype(np.int8)

NameError: name 'train_df' is not defined

## === cell 6
print("Old size: %d" % len(train_df))
train_df.dropna(inplace=True)
print("New size after dropna: %d" % len(train_df))




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4088198032.py in <cell line: 0>()
----> 1 print("Old size: %d" % len(train_df))
      2 train_df.dropna(inplace=True)
      3 print("New size after dropna: %d" % len(train_df))
      4 
      5 

NameError: name 'train_df' is not defined

## === cell 7
cond = True
for col in {
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
}:
    cond &= np.abs(train_df[col] - train_df[col].mean()) < 5
cond &= train_df["fare_amount"] > 0
print("Old size (filter): %d" % len(train_df))
train_df = train_df.loc[cond]
print("New size (filter): %d" % len(train_df))




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1184758687.py in <cell line: 0>()
      6     "dropoff_longitude",
      7 }:
----> 8     cond &= np.abs(train_df[col] - train_df[col].mean()) < 5
      9 cond &= train_df["fare_amount"] > 0
     10 print("Old size (filter): %d" % len(train_df))

NameError: name 'train_df' is not defined

## === cell 8
for col in {
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
}:
    train_df["rough" + col] = train_df[col].round(2)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2397330904.py in <cell line: 0>()
      5     "dropoff_longitude",
      6 }:
----> 7     train_df["rough" + col] = train_df[col].round(2)
      8 
      9 

NameError: name 'train_df' is not defined

## === cell 9
a = train_df.groupby(["roughpickup_latitude", "roughpickup_longitude"], observed=True)[
    ["pickup_latitude", "pickup_longitude"]
].agg(["mean", "count"])




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1368156837.py in <cell line: 0>()
----> 1 a = train_df.groupby(["roughpickup_latitude", "roughpickup_longitude"], observed=True)[
      2     ["pickup_latitude", "pickup_longitude"]
      3 ].agg(["mean", "count"])
      4 
      5 

NameError: name 'train_df' is not defined

## === cell 10
a.columns = [
    "mean_pickup_latitude",
    "c1",
    "mean_pickup_longitude",
    "c2",
]
a = a[["c1"]].reset_index().rename(columns={"c1": "pickup_busyness"})




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4255069094.py in <cell line: 0>()
----> 1 a.columns = [
      2     "mean_pickup_latitude",
      3     "c1",
      4     "mean_pickup_longitude",
      5     "c2",

NameError: name 'a' is not defined

## === cell 11
b = train_df.groupby(
    ["roughdropoff_latitude", "roughdropoff_longitude"], observed=True
)[["dropoff_latitude", "dropoff_longitude"]].agg(["mean", "count"])




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/627628585.py in <cell line: 0>()
----> 1 b = train_df.groupby(
      2     ["roughdropoff_latitude", "roughdropoff_longitude"], observed=True
      3 )[["dropoff_latitude", "dropoff_longitude"]].agg(["mean", "count"])
      4 
      5 

NameError: name 'train_df' is not defined

## === cell 12
b.columns = [
    "mean_dropoff_latitude",
    "c1",
    "mean_dropoff_longitude",
    "c2",
]
b = b[["c1"]].reset_index().rename(columns={"c1": "dropoff_busyness"})




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/666988121.py in <cell line: 0>()
----> 1 b.columns = [
      2     "mean_dropoff_latitude",
      3     "c1",
      4     "mean_dropoff_longitude",
      5     "c2",

NameError: name 'b' is not defined

## === cell 13
train_df = pd.merge(
    train_df,
    a,
    how="left",
    on=["roughpickup_latitude", "roughpickup_longitude"],
)
train_df = pd.merge(
    train_df,
    b,
    how="left",
    on=["roughdropoff_latitude", "roughdropoff_longitude"],
)
train_df["pickup_busyness"].fillna(1, inplace=True)
train_df["dropoff_busyness"].fillna(1, inplace=True)




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/176051847.py in <cell line: 0>()
      1 train_df = pd.merge(
----> 2     train_df,
      3     a,
      4     how="left",
      5     on=["roughpickup_latitude", "roughpickup_longitude"],

NameError: name 'train_df' is not defined

## === cell 14
X = train_df[
    [
        "distance",
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "day_of_week",
        "passenger_count",
        "pickup_busyness",
        "dropoff_busyness",
    ]
].values.astype(np.float32)
Y = train_df["fare_amount"].values.astype(np.float32)




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1433421932.py in <cell line: 0>()
----> 1 X = train_df[
      2     [
      3         "distance",
      4         "year",
      5         "month",

NameError: name 'train_df' is not defined

## === cell 15
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

mask_train = y_train > 0
mask_val = y_val > 0
X_train = X_train[mask_train]
y_train = y_train[mask_train]
X_val = X_val[mask_val]
y_val = y_val[mask_val]

y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)

rf = RandomForestRegressor(
    n_estimators=300,
    max_features=None,
    min_samples_leaf=1,
    min_samples_split=2,
    bootstrap=True,
    max_depth=None,
    n_jobs=-1,
    random_state=42,
)

rf.fit(X_train, y_train_log)

val_pred_log = rf.predict(X_val)
val_pred = np.expm1(val_pred_log)
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f"Validation RMSE (log‑target): {val_rmse:.5f}")




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2012129716.py in <cell line: 0>()
----> 1 X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
      2 
      3 # Align X with positive targets
      4 mask_train = y_train > 0
      5 mask_val = y_val > 0

NameError: name 'X' is not defined

## === cell 16
Y_log_full = np.log1p(Y)
rf_full = RandomForestRegressor(
    n_estimators=300,
    max_features=None,
    min_samples_leaf=1,
    min_samples_split=2,
    bootstrap=True,
    max_depth=None,
    n_jobs=-1,
    random_state=42,
)
rf_full.fit(X, Y_log_full)




## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4172718246.py in <cell line: 0>()
----> 1 Y_log_full = np.log1p(Y)
      2 rf_full = RandomForestRegressor(
      3     n_estimators=300,
      4     max_features=None,
      5     min_samples_leaf=1,

NameError: name 'Y' is not defined

## === cell 17
test_path = find_data_path("test.csv")
test_df = pd.read_csv(
    test_path,
    usecols=[
        "key",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
    dtype={
        "key": "object",
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "int8",
    },
    low_memory=False,
)




## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/473153208.py in <cell line: 0>()
----> 1 test_path = find_data_path("test.csv")
      2 test_df = pd.read_csv(
      3     test_path,
      4     usecols=[
      5         "key",

/tmp/ipykernel_11/3846850250.py in find_data_path(filename)
     24             if p.is_file():
     25                 return p
---> 26     raise FileNotFoundError(f"Could not find {filename} in expected locations.")
     27 
     28 

FileNotFoundError: Could not find test.csv in expected locations.

## === cell 18
test_df["distance"] = haversine_np(
    test_df["pickup_longitude"],
    test_df["pickup_latitude"],
    test_df["dropoff_longitude"],
    test_df["dropoff_latitude"],
)




## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/905273357.py in <cell line: 0>()
      1 test_df["distance"] = haversine_np(
----> 2     test_df["pickup_longitude"],
      3     test_df["pickup_latitude"],
      4     test_df["dropoff_longitude"],
      5     test_df["dropoff_latitude"],

NameError: name 'test_df' is not defined

## === cell 19
test_df["pickup_datetime"] = pd.to_datetime(test_df["pickup_datetime"])




## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1512093491.py in <cell line: 0>()
----> 1 test_df["pickup_datetime"] = pd.to_datetime(test_df["pickup_datetime"])
      2 
      3 

NameError: name 'test_df' is not defined

## === cell 20
test_df["year"] = test_df["pickup_datetime"].dt.year.astype(np.int16)
test_df["month"] = test_df["pickup_datetime"].dt.month.astype(np.int8)
test_df["day"] = test_df["pickup_datetime"].dt.day.astype(np.int8)
test_df["hour"] = test_df["pickup_datetime"].dt.hour.astype(np.int8)
test_df["minute"] = test_df["pickup_datetime"].dt.minute.astype(np.int8)
test_df["day_of_week"] = test_df["pickup_datetime"].dt.dayofweek.astype(np.int8)




## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1791568135.py in <cell line: 0>()
----> 1 test_df["year"] = test_df["pickup_datetime"].dt.year.astype(np.int16)
      2 test_df["month"] = test_df["pickup_datetime"].dt.month.astype(np.int8)
      3 test_df["day"] = test_df["pickup_datetime"].dt.day.astype(np.int8)
      4 test_df["hour"] = test_df["pickup_datetime"].dt.hour.astype(np.int8)
      5 test_df["minute"] = test_df["pickup_datetime"].dt.minute.astype(np.int8)

NameError: name 'test_df' is not defined

## === cell 21
for col in {
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
}:
    test_df["rough" + col] = test_df[col].round(2)




## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3057192082.py in <cell line: 0>()
      5     "dropoff_longitude",
      6 }:
----> 7     test_df["rough" + col] = test_df[col].round(2)
      8 
      9 

NameError: name 'test_df' is not defined

## === cell 22
test_df = pd.merge(
    test_df,
    a,
    how="left",
    on=["roughpickup_latitude", "roughpickup_longitude"],
)
test_df = pd.merge(
    test_df,
    b,
    how="left",
    on=["roughdropoff_latitude", "roughdropoff_longitude"],
)
test_df["pickup_busyness"].fillna(1, inplace=True)
test_df["dropoff_busyness"].fillna(1, inplace=True)




## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4244035744.py in <cell line: 0>()
      1 test_df = pd.merge(
----> 2     test_df,
      3     a,
      4     how="left",
      5     on=["roughpickup_latitude", "roughpickup_longitude"],

NameError: name 'test_df' is not defined

## === cell 23
X_test = test_df[
    [
        "distance",
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "day_of_week",
        "passenger_count",
        "pickup_busyness",
        "dropoff_busyness",
    ]
].values.astype(np.float32)

test_pred_log = rf_full.predict(X_test)
test_pred = np.expm1(test_pred_log)




## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/643532688.py in <cell line: 0>()
----> 1 X_test = test_df[
      2     [
      3         "distance",
      4         "year",
      5         "month",

NameError: name 'test_df' is not defined

## === cell 24
submission = pd.DataFrame(
    {"key": test_df["key"], "fare_amount": test_pred},
    columns=["key", "fare_amount"],
)
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")

## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/574238032.py in <cell line: 0>()
      1 submission = pd.DataFrame(
----> 2     {"key": test_df["key"], "fare_amount": test_pred},
      3     columns=["key", "fare_amount"],
      4 )
      5 submission.to_csv("submission.csv", index=False)

NameError: name 'test_df' is not defined
