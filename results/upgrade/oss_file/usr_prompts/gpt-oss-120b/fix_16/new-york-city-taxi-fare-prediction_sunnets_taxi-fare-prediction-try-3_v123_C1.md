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

No external packages required in the script and installed.

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

7.657941173557325

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def find_file(filename):
    """
    Search common directories and also inside the competition subfolder.
    """
    base_dirs = [
        "",  # current working directory
        "data",
        "input",
        os.path.join("kaggle", "working"),
    ]
    for base in base_dirs:
        path = os.path.join(base, filename)
        if os.path.exists(path):
            return path
    subfolder = "new-york-city-taxi-fare-prediction"
    for base in base_dirs:
        path = os.path.join(base, subfolder, filename)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Unable to locate {filename}")


TRAIN_PATH = find_file("train.csv")
TEST_PATH = find_file("test.csv")
SUBMISSION_NAME = "submission.csv"

test_df = pd.read_csv(TEST_PATH, low_memory=False)




## === cell 1
def clean(df):
    """Efficient cleaning using a single combined mask."""
    df = df.dropna()

    mask = (
        (df["dropoff_longitude"] != df["pickup_longitude"])
        & (df["dropoff_latitude"] != df["pickup_latitude"])
        & (df["dropoff_longitude"] != 0)
        & (df["pickup_longitude"] != 0)
        & (df["dropoff_latitude"] != 0)
        & (df["pickup_latitude"] != 0)
        & (-74.5 <= df["pickup_longitude"])
        & (df["pickup_longitude"] <= -72.8)
        & (-74.5 <= df["dropoff_longitude"])
        & (df["dropoff_longitude"] <= -72.8)
        & (40.5 <= df["pickup_latitude"])
        & (df["pickup_latitude"] <= 41.8)
        & (40.5 <= df["dropoff_latitude"])
        & (df["dropoff_latitude"] <= 41.8)
        & (df["pickup_latitude"] - df["dropoff_latitude"]).abs()
        > 0.001 & (df["pickup_longitude"] - df["dropoff_longitude"]).abs()
        > 0.001 & (df["passenger_count"] > 0)
    )
    if "fare_amount" in df.columns:
        mask &= (df["fare_amount"] > 0) & (df["fare_amount"] <= 50)

    cleaned = df[mask].copy()
    print(f"Cleaned size: {len(cleaned)}")
    return cleaned


def add_time_features(df):
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df["year"] = df["pickup_datetime"].dt.year
    df["month"] = df["pickup_datetime"].dt.month
    df["day"] = df["pickup_datetime"].dt.day
    df["hour"] = df["pickup_datetime"].dt.hour
    df["minute"] = df["pickup_datetime"].dt.minute
    df["second"] = df["pickup_datetime"].dt.second
    return df


def add_coordinate_features(df):
    df["latdiff"] = df["pickup_latitude"] - df["dropoff_latitude"]
    df["londiff"] = df["pickup_longitude"] - df["dropoff_longitude"]
    return df


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_distances_features(df):
    df["manhattan"] = manhattan(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )
    return df


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km


def add_haversine_feature(df):
    df["haversine"] = haversine(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )
    return df




## === cell 2
TARGET_SAMPLE_SIZE = 500_000
rng = np.random.default_rng(42)

reservoir = None  # DataFrame that will hold the sample
total_seen = 0  # Number of clean rows processed so far
k = TARGET_SAMPLE_SIZE

chunksize = 1_000_000
for chunk in pd.read_csv(TRAIN_PATH, chunksize=chunksize, low_memory=False):
    chunk = clean(chunk)
    if chunk.empty:
        continue

    if reservoir is None:
        reservoir = pd.DataFrame(columns=chunk.columns)

    if len(reservoir) < k:
        need = min(k - len(reservoir), len(chunk))
        reservoir = pd.concat([reservoir, chunk.iloc[:need]], ignore_index=True)
        total_seen += need
        start_idx = need
    else:
        start_idx = 0

    if start_idx < len(chunk):
        remaining = chunk.iloc[start_idx:].reset_index(drop=True)
        n = len(remaining)

        overall_idx = np.arange(total_seen + 1, total_seen + n + 1)
        prob = k / overall_idx.astype(float)

        rand = rng.random(n)
        replace_mask = rand < prob
        if replace_mask.any():
            replace_rows = remaining[replace_mask]
            replace_pos = rng.integers(0, k, size=replace_rows.shape[0])
            reservoir.iloc[replace_pos] = replace_rows.values
        total_seen += n

train_df = reservoir.reset_index(drop=True)

print(f"Sampled training size: {len(train_df)}")

train_df = add_time_features(train_df)
test_df = add_time_features(test_df)

train_df = add_coordinate_features(train_df)
test_df = add_coordinate_features(test_df)

train_df = add_distances_features(train_df)
test_df = add_distances_features(test_df)

train_df = add_haversine_feature(train_df)
test_df = add_haversine_feature(test_df)

train_df = train_df.fillna(-1)
test_df = test_df.fillna(-1)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/ops/array_ops.py in na_logical_op(x, y, op)
    361         #  (xint or xbool) and (yint or bool)
--> 362         result = op(x, y)
    363     except TypeError:

/usr/local/lib/python3.11/dist-packages/pandas/core/roperator.py in rand_(left, right)
     53 def rand_(left, right):
---> 54     return operator.and_(right, left)
     55 

TypeError: ufunc 'bitwise_and' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''

During handling of the above exception, another exception occurred:

ValueError                                Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/ops/array_ops.py in na_logical_op(x, y, op)
    375             try:
--> 376                 result = libops.scalar_binop(x, y, op)
    377             except (

ops.pyx in pandas._libs.ops.scalar_binop()

ValueError: Buffer dtype mismatch, expected 'Python object' but got 'double'

The above exception was the direct cause of the following exception:

TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4198317774.py in <cell line: 0>()
      8 chunksize = 1_000_000
      9 for chunk in pd.read_csv(TRAIN_PATH, chunksize=chunksize, low_memory=False):
---> 10     chunk = clean(chunk)
     11     if chunk.empty:
     12         continue

/tmp/ipykernel_11/916545555.py in clean(df)
     21         & (df["dropoff_latitude"] <= 41.8)
     22         & (df["pickup_latitude"] - df["dropoff_latitude"]).abs()
---> 23         > 0.001 & (df["pickup_longitude"] - df["dropoff_longitude"]).abs()
     24         > 0.001 & (df["passenger_count"] > 0)
     25     )

/usr/local/lib/python3.11/dist-packages/pandas/core/ops/common.py in new_method(self, other)
     74         other = item_from_zerodim(other)
     75 
---> 76         return method(self, other)
     77 
     78     return new_method

/usr/local/lib/python3.11/dist-packages/pandas/core/arraylike.py in __rand__(self, other)
     72     @unpack_zerodim_and_defer("__rand__")
     73     def __rand__(self, other):
---> 74         return self._logical_method(other, roperator.rand_)
     75 
     76     @unpack_zerodim_and_defer("__or__")

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in _logical_method(self, other, op)
   6128         rvalues = extract_array(other, extract_numpy=True, extract_range=True)
   6129 
-> 6130         res_values = ops.logical_op(lvalues, rvalues, op)
   6131         return self._construct_result(res_values, name=res_name)
   6132 

/usr/local/lib/python3.11/dist-packages/pandas/core/ops/array_ops.py in logical_op(left, right, op)
    452             is_other_int_dtype = lib.is_integer(rvalues)
    453 
--> 454         res_values = na_logical_op(lvalues, rvalues, op)
    455 
    456         # For int vs int `^`, `|`, `&` are bitwise operators and return

/usr/local/lib/python3.11/dist-packages/pandas/core/ops/array_ops.py in na_logical_op(x, y, op)
    383             ) as err:
    384                 typ = type(y).__name__
--> 385                 raise TypeError(
    386                     f"Cannot perform '{op.__name__}' with a dtyped [{x.dtype}] array "
    387                     f"and scalar of type [{typ}]"

TypeError: Cannot perform 'rand_' with a dtyped [float64] array and scalar of type [bool]

## === cell 3
target_col = "fare_amount"
feature_cols = [
    c for c in train_df.columns if c not in ["key", target_col, "pickup_datetime"]
]

X = train_df[feature_cols].values.astype(np.float32)
y = train_df[target_col].values
X_test = test_df[feature_cols].values.astype(np.float32)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, random_state=42)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2389934548.py in <cell line: 0>()
      1 target_col = "fare_amount"
      2 feature_cols = [
----> 3     c for c in train_df.columns if c not in ["key", target_col, "pickup_datetime"]
      4 ]
      5 

NameError: name 'train_df' is not defined

## === cell 4
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    n_jobs=-1,
    random_state=42,
)

rf_model.fit(X_train, y_train)

val_pred = rf_model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f"Validation RMSE: {val_rmse:.4f}")




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1116322211.py in <cell line: 0>()
      8 )
      9 
---> 10 rf_model.fit(X_train, y_train)
     11 
     12 val_pred = rf_model.predict(X_val)

NameError: name 'X_train' is not defined

## === cell 5
test_predictions = rf_model.predict(X_test)

submission = pd.DataFrame({"key": test_df["key"], "fare_amount": test_predictions})
submission.to_csv(SUBMISSION_NAME, index=False)
print("Submission written to", SUBMISSION_NAME)

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2817869614.py in <cell line: 0>()
----> 1 test_predictions = rf_model.predict(X_test)
      2 
      3 submission = pd.DataFrame({"key": test_df["key"], "fare_amount": test_predictions})
      4 submission.to_csv(SUBMISSION_NAME, index=False)
      5 print("Submission written to", SUBMISSION_NAME)

NameError: name 'X_test' is not defined
