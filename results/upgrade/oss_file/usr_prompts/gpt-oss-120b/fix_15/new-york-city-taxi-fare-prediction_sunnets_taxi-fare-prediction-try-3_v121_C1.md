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

4.051660609378363

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 11.91892) has done: 'The script is rewritten to load the correct CSV files, clean and engineer features without external image resources, use `sklearn` models (avoiding the failing Keras imports), and finally write a proper `submissiontry_water.csv` containing the required `key` and `fare_amount` columns. All original logic is preserved where possible, and the changes only fix runtime errors and enable a valid submission.'
- What this solution (achieved 7.3606) has done: 'I modify the cleaning function so that for the test set it only drops NaNs (keeping the original row count, which ensures the submission includes every required key). I also slightly strengthen the GradientBoostingRegressor (more trees, lower learning rate) to nudge the validation RMSE closer to the target without altering the overall modeling approach.'

# 9. Code solution

## === cell 0
import os
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


def resolve_path(*parts):
    """
    Look for a path in several typical locations:
    1) /kaggle/data/...               (common Kaggle kernel layout)
    2) /kaggle/input/...              (alternative Kaggle layout)
    3) ./...                           (relative to current working directory)
    4) ...                             (as‑is)
    Returns the first existing path.
    """
    candidates = [
        os.path.join("/kaggle", "data", *parts),
        os.path.join("/kaggle", "input", *parts),
        os.path.join(".", *parts),
        os.path.join(*parts),
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f"Could not find file or directory: {'/'.join(parts)}")


BASE_INPUT = resolve_path("data")  # finds /kaggle/data
TRAIN_PATH = os.path.join(BASE_INPUT, "train.csv")
TEST_PATH = os.path.join(BASE_INPUT, "test.csv")
SUBMISSION_NAME = "submissiontry_water.csv"

DATASET_SIZE = 400_000  # subset for quick iteration; adjust if needed
RANDOM_STATE = 42
TEST_SIZE = 0.10  # validation split



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3448566841.py in <cell line: 0>()
     27 
     28 
---> 29 BASE_INPUT = resolve_path("data")  # finds /kaggle/data
     30 TRAIN_PATH = os.path.join(BASE_INPUT, "train.csv")
     31 TEST_PATH = os.path.join(BASE_INPUT, "test.csv")

/tmp/ipykernel_11/3448566841.py in resolve_path(*parts)
     24         if os.path.exists(cand):
     25             return cand
---> 26     raise FileNotFoundError(f"Could not find file or directory: {'/'.join(parts)}")
     27 
     28 

FileNotFoundError: Could not find file or directory: data

## === cell 1
dtype_train = {
    "key": "str",
    "fare_amount": "float32",
    "pickup_datetime": "str",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}
dtype_test = {
    "key": "str",
    "pickup_datetime": "str",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}

train_df = pd.read_csv(
    TRAIN_PATH, dtype=dtype_train, nrows=DATASET_SIZE, low_memory=False
)
test_df = pd.read_csv(TEST_PATH, dtype=dtype_test, low_memory=False)

print(f"train shape: {train_df.shape}, test shape: {test_df.shape}")




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3482906237.py in <cell line: 0>()
     20 
     21 train_df = pd.read_csv(
---> 22     TRAIN_PATH, dtype=dtype_train, nrows=DATASET_SIZE, low_memory=False
     23 )
     24 test_df = pd.read_csv(TEST_PATH, dtype=dtype_test, low_memory=False)

NameError: name 'TRAIN_PATH' is not defined

## === cell 2
def clean(df, is_test=False):
    """Basic cleaning."""
    print("  Old size:", len(df))
    if is_test:
        df.loc[df["pickup_longitude"] == 0, "pickup_longitude"] = np.nan
        df.loc[df["dropoff_longitude"] == 0, "dropoff_longitude"] = np.nan
        df.loc[df["pickup_latitude"] == 0, "pickup_latitude"] = np.nan
        df.loc[df["dropoff_latitude"] == 0, "dropoff_latitude"] = np.nan
        print("  After sanity fixes (test):", len(df))
        return df

    df_original = df.copy()  # keep original for possible revert
    original_len = len(df)

    min_lon, max_lon, min_lat, max_lat = -74.5, -72.8, 40.5, 41.8
    mask = (
        (
            (df["dropoff_longitude"] != df["pickup_longitude"])
            | (df["dropoff_latitude"] != df["pickup_latitude"])
        )
        & (df["pickup_longitude"] != 0)
        & (df["dropoff_longitude"] != 0)
        & (df["pickup_latitude"] != 0)
        & (df["dropoff_latitude"] != 0)
        & (df["pickup_longitude"] >= min_lon)
        & (df["pickup_longitude"] <= max_lon)
        & (df["dropoff_longitude"] >= min_lon)
        & (df["dropoff_longitude"] <= max_lon)
        & (df["pickup_latitude"] >= min_lat)
        & (df["pickup_latitude"] <= max_lat)
        & (df["dropoff_latitude"] >= min_lat)
        & (df["dropoff_latitude"] <= max_lat)
        & (df["fare_amount"] > 0)
        & (df["fare_amount"] <= 200)
        & (df["passenger_count"] > 0)
    )
    df = df[mask]
    print("  After logical filters:", len(df))

    df = df.dropna(how="any")
    print("  After dropna:", len(df))

    if df.empty:
        print("  Warning: cleaning removed all rows – reverting to original.")
        df = df_original.copy()
    else:
        print(f"  Dropped {original_len - len(df)} rows during cleaning.")
    return df


def add_time_features(df):
    """Robust temporal feature extraction – never drops rows."""
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    dt = dt.fillna(pd.Timestamp("2000-01-01"))
    df["year"] = dt.dt.year.astype(np.int16)
    df["month"] = dt.dt.month.astype(np.int8)
    df["day"] = dt.dt.day.astype(np.int8)
    df["hour"] = dt.dt.hour.astype(np.int8)
    df["weekday"] = dt.dt.weekday.astype(np.int8)
    return df


def haversine(lat1, lon1, lat2, lon2):
    """Vectorised haversine distance in kilometres."""
    p = np.pi / 180.0
    a = (
        np.sin((lat2 - lat1) * p / 2.0) ** 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2.0) ** 2
    )
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def add_distance_features(df):
    """Add haversine and Manhattan‑like distance features."""
    df["haversine"] = haversine(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )
    lat_diff = (df["pickup_latitude"] - df["dropoff_latitude"]).abs()
    lon_diff = (df["pickup_longitude"] - df["dropoff_longitude"]).abs()
    df["manhattan"] = lat_diff + lon_diff
    return df




## === cell 3
print("Cleaning training data")
train_df = clean(train_df, is_test=False)

print("Cleaning test data")
test_df = clean(test_df, is_test=True)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2376757178.py in <cell line: 0>()
      1 print("Cleaning training data")
----> 2 train_df = clean(train_df, is_test=False)
      3 
      4 print("Cleaning test data")
      5 test_df = clean(test_df, is_test=True)

NameError: name 'train_df' is not defined

## === cell 4
print("Adding temporal features")
train_df = add_time_features(train_df)
test_df = add_time_features(test_df)

print("Adding distance features")
train_df = add_distance_features(train_df)
test_df = add_distance_features(test_df)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/244457816.py in <cell line: 0>()
      1 print("Adding temporal features")
----> 2 train_df = add_time_features(train_df)
      3 test_df = add_time_features(test_df)
      4 
      5 print("Adding distance features")

NameError: name 'train_df' is not defined

## === cell 5
drop_cols = ["key", "pickup_datetime"]
X = train_df.drop(columns=drop_cols + ["fare_amount"])
y = np.log1p(train_df["fare_amount"])

X_test = test_df.drop(columns=drop_cols)

X = X.fillna(0)
X_test = X_test.fillna(0)

print(f"Feature matrix shape: {X.shape}")



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3191278263.py in <cell line: 0>()
      1 drop_cols = ["key", "pickup_datetime"]
----> 2 X = train_df.drop(columns=drop_cols + ["fare_amount"])
      3 y = np.log1p(train_df["fare_amount"])
      4 
      5 X_test = test_df.drop(columns=drop_cols)

NameError: name 'train_df' is not defined

## === cell 6
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

print(f"Train: {X_train.shape}, Validation: {X_val.shape}")

X_train = X_train.to_numpy(dtype=np.float32, copy=False)
X_val = X_val.to_numpy(dtype=np.float32, copy=False)
X_test_scaled = X_test.to_numpy(dtype=np.float32, copy=False)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2771204087.py in <cell line: 0>()
      1 X_train, X_val, y_train, y_val = train_test_split(
----> 2     X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
      3 )
      4 
      5 print(f"Train: {X_train.shape}, Validation: {X_val.shape}")

NameError: name 'X' is not defined

## === cell 7
gbr = GradientBoostingRegressor(
    n_estimators=1500,  # more trees for better fit
    learning_rate=0.01,
    max_depth=6,
    subsample=1.0,
    random_state=RANDOM_STATE,
)

gbr.fit(X_train, y_train)

val_pred_log = gbr.predict(X_val)
val_pred = np.expm1(val_pred_log)

val_rmse = np.sqrt(mean_squared_error(np.expm1(y_val), val_pred))
print(f"Validation RMSE: {val_rmse:.4f}")



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/959033485.py in <cell line: 0>()
      4     max_depth=6,
      5     subsample=1.0,
----> 6     random_state=RANDOM_STATE,
      7 )
      8 

NameError: name 'RANDOM_STATE' is not defined

## === cell 8
test_pred_log = gbr.predict(X_test_scaled)
test_pred = np.expm1(test_pred_log)

submission = pd.DataFrame({"key": test_df["key"], "fare_amount": test_pred})
submission_path = os.path.join(".", SUBMISSION_NAME)
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2390892468.py in <cell line: 0>()
----> 1 test_pred_log = gbr.predict(X_test_scaled)
      2 test_pred = np.expm1(test_pred_log)
      3 
      4 submission = pd.DataFrame({"key": test_df["key"], "fare_amount": test_pred})
      5 submission_path = os.path.join(".", SUBMISSION_NAME)

NameError: name 'gbr' is not defined
