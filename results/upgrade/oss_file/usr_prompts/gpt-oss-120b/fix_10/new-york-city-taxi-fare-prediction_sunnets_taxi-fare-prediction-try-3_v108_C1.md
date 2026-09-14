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

4.031294871457426

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 5.92572) has done: 'I fixed the script by removing the TensorFlow dependency (which was causing import errors), making the `clean` function safe for the test set, skipping cleaning on the test data to keep all rows, and replacing the neural network with a scikit‑learn RandomForest regressor. These changes let the pipeline run end‑to‑end, generate a valid CSV submission, and keep the core modelling approach while staying within the original feature‑engineering logic.'
- What this solution (achieved 5.97186) has done: 'I add a haversine distance feature (a more realistic measure of trip length) and increase the forest size to give the model a bit more capacity. Both changes keep the existing pipeline intact while targeting a lower RMSE, moving the score toward the target.'
- What this solution (achieved 6.19265) has done: 'The script now reads a smaller, still representative subset of the data, uses explicit float32 types for numeric arrays, and frees memory after heavy steps. These changes cut the training time of the GradientBoostingRegressor well below the 600 s limit while keeping the same preprocessing, feature engineering, and model‑definition logic, thus preserving result accuracy.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor  # switched from RandomForest
import gc  # for explicit memory cleanup


def clean(df):
    print(" Old size: %d" % len(df))
    df = df.dropna(how="any", axis="rows")
    print(" New size after dropna: %d" % len(df))

    mask = (
        (
            (df["dropoff_longitude"] != df["pickup_longitude"])
            | (df["dropoff_latitude"] != df["pickup_latitude"])
        )
        & (df["dropoff_longitude"] != 0)
        & (df["pickup_longitude"] != 0)
        & (df["dropoff_latitude"] != 0)
        & (df["pickup_latitude"] != 0)
    )
    MinMax = (-74.5, -72.8, 40.5, 41.8)
    mask &= (MinMax[0] <= df["pickup_longitude"]) & (
        df["pickup_longitude"] <= MinMax[1]
    )
    mask &= (MinMax[0] <= df["dropoff_longitude"]) & (
        df["dropoff_longitude"] <= MinMax[1]
    )
    mask &= (MinMax[2] <= df["pickup_latitude"]) & (df["pickup_latitude"] <= MinMax[3])
    mask &= (MinMax[2] <= df["dropoff_latitude"]) & (
        df["dropoff_latitude"] <= MinMax[3]
    )
    if "fare_amount" in df.columns:
        mask &= (0 < df["fare_amount"]) & (df["fare_amount"] <= 50)
    mask &= df["passenger_count"] > 0

    df = df[mask]
    print(" New size after all filters: %d" % len(df))
    return df


def add_time_features(df):
    if not np.issubdtype(df["pickup_datetime"].dtype, np.datetime64):
        df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df["year"] = df["pickup_datetime"].dt.year
    df["month"] = df["pickup_datetime"].dt.month
    df["day"] = df["pickup_datetime"].dt.day
    df["hour"] = df["pickup_datetime"].dt.hour
    df["minute"] = df["pickup_datetime"].dt.minute
    df["second"] = df["pickup_datetime"].dt.second
    df["weekday"] = df["pickup_datetime"].dt.weekday
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(np.int8)
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


def add_haversine_features(df):
    lat1 = np.radians(df["pickup_latitude"])
    lon1 = np.radians(df["pickup_longitude"])
    lat2 = np.radians(df["dropoff_latitude"])
    lon2 = np.radians(df["dropoff_longitude"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    earth_radius_km = 6371.0
    df["haversine"] = earth_radius_km * c
    return df


def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    df = pd.DataFrame(
        {id_column: raw_test[id_column], prediction_column: prediction.squeeze()}
    )
    df.to_csv(file_name, index=False)
    print(f"Output written to {file_name}")




## === cell 1
TRAIN_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
TEST_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"

DATASET_SIZE = 200000  # rows (reduced from 500k for speed)

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

dtype_dict = {
    "key": "object",
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}

train_df = pd.read_csv(
    TRAIN_PATH,
    nrows=DATASET_SIZE,
    usecols=usecols,
    dtype={k: v for k, v in dtype_dict.items() if k != "pickup_datetime"},
    parse_dates=["pickup_datetime"],
    low_memory=False,
)

testKaggle = pd.read_csv(
    TEST_PATH,
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
        k: v
        for k, v in dtype_dict.items()
        if k not in ["fare_amount", "pickup_datetime"]
    },
    parse_dates=["pickup_datetime"],
    low_memory=False,
)

print(f"Train rows loaded: {len(train_df)}")
print(f"Test rows loaded: {len(testKaggle)}")




## === cell 2
train_df = clean(train_df)

train_df = add_time_features(train_df)
testKaggle = add_time_features(testKaggle)

train_df = add_coordinate_features(train_df)
testKaggle = add_coordinate_features(testKaggle)

train_df = add_distances_features(train_df)
testKaggle = add_distances_features(testKaggle)

train_df = add_haversine_features(train_df)
testKaggle = add_haversine_features(testKaggle)

train_df = train_df.drop(["pickup_datetime"], axis=1)
testKaggle = testKaggle.drop(["pickup_datetime"], axis=1)

print("Feature engineering completed.")
gc.collect()  # free intermediate objects




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2254818791.py in <cell line: 0>()
      1 train_df = clean(train_df)
      2 
----> 3 train_df = add_time_features(train_df)
      4 testKaggle = add_time_features(testKaggle)
      5 

/tmp/ipykernel_11/3260340432.py in add_time_features(df)
     45 def add_time_features(df):
     46     # Convert only if still a string/object; if already datetime (from read_csv) skip conversion
---> 47     if not np.issubdtype(df["pickup_datetime"].dtype, np.datetime64):
     48         df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
     49     df["year"] = df["pickup_datetime"].dt.year

/usr/local/lib/python3.11/dist-packages/numpy/core/numerictypes.py in issubdtype(arg1, arg2)
    415     """
    416     if not issubclass_(arg1, generic):
--> 417         arg1 = dtype(arg1).type
    418     if not issubclass_(arg2, generic):
    419         arg2 = dtype(arg2).type

TypeError: Cannot interpret 'datetime64[ns, UTC]' as a data type

## === cell 3
scaler = preprocessing.MinMaxScaler()
features = [c for c in train_df.columns if c not in ["key", "fare_amount"]]

train_df[features] = scaler.fit_transform(train_df[features]).astype(np.float32)
testKaggle[features] = scaler.transform(testKaggle[features]).astype(np.float32)

train_labels = train_df["fare_amount"].values.astype(np.float32)
train_ids = train_df["key"].values
train_df = train_df.drop(["key", "fare_amount"], axis=1)

test_ids = testKaggle["key"].values
testKaggle = testKaggle.drop(["key"], axis=1)

X = train_df.values.astype(np.float32)
X_test = testKaggle.values.astype(np.float32)

X_train, X_val, y_train, y_val = train_test_split(
    X, train_labels, test_size=0.1, random_state=42
)

print(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}")
gc.collect()




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1863346684.py in <cell line: 0>()
      3 
      4 # Fit/transform on train, transform on test; cast to float32 once.
----> 5 train_df[features] = scaler.fit_transform(train_df[features]).astype(np.float32)
      6 testKaggle[features] = scaler.transform(testKaggle[features]).astype(np.float32)
      7 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py in wrapped(self, X, *args, **kwargs)
    138     @wraps(f)
    139     def wrapped(self, X, *args, **kwargs):
--> 140         data_to_wrap = f(self, X, *args, **kwargs)
    141         if isinstance(data_to_wrap, tuple):
    142             # only wrap the first output for cross decomposition

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in fit_transform(self, X, y, **fit_params)
    876         if y is None:
    877             # fit method of arity 1 (unsupervised transformation)
--> 878             return self.fit(X, **fit_params).transform(X)
    879         else:
    880             # fit method of arity 2 (supervised transformation)

/usr/local/lib/python3.11/dist-packages/sklearn/preprocessing/_data.py in fit(self, X, y)
    425         # Reset internal state before fitting
    426         self._reset()
--> 427         return self.partial_fit(X, y)
    428 
    429     def partial_fit(self, X, y=None):

/usr/local/lib/python3.11/dist-packages/sklearn/preprocessing/_data.py in partial_fit(self, X, y)
    464 
    465         first_pass = not hasattr(self, "n_samples_seen_")
--> 466         X = self._validate_data(
    467             X,
    468             reset=first_pass,

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    563             raise ValueError("Validation should be done on X, y or both.")
    564         elif not no_val_X and no_val_y:
--> 565             X = check_array(X, input_name="X", **check_params)
    566             out = X
    567         elif no_val_X and not no_val_y:

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)
    877                     array = xp.astype(array, dtype, copy=False)
    878                 else:
--> 879                     array = _asarray_with_order(array, order=order, dtype=dtype, xp=xp)
    880             except ComplexWarning as complex_warning:
    881                 raise ValueError(

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_array_api.py in _asarray_with_order(array, dtype, order, copy, xp)
    183     if xp.__name__ in {"numpy", "numpy.array_api"}:
    184         # Use NumPy API to support order
--> 185         array = numpy.asarray(array, order=order, dtype=dtype)
    186         return xp.asarray(array, copy=copy)
    187     else:

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __array__(self, dtype, copy)
   2151     ) -> np.ndarray:
   2152         values = self._values
-> 2153         arr = np.asarray(values, dtype=dtype)
   2154         if (
   2155             astype_is_view(values.dtype, arr.dtype)

TypeError: float() argument must be a string or a real number, not 'Timestamp'

## === cell 4
model = GradientBoostingRegressor(
    n_estimators=500,  # fewer trees for faster training
    learning_rate=0.04,
    max_depth=6,
    subsample=0.8,
    random_state=42,
)

model.fit(X_train, y_train)

val_pred = model.predict(X_val)
val_rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))
print(f"Validation RMSE: {val_rmse:.4f}")




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3420210623.py in <cell line: 0>()
      7 )
      8 
----> 9 model.fit(X_train, y_train)
     10 
     11 val_pred = model.predict(X_val)

NameError: name 'X_train' is not defined

## === cell 5
test_predictions = model.predict(X_test)
test_predictions = np.clip(test_predictions, 0.0, 50.0)

output_submission(
    pd.DataFrame({"key": test_ids}),
    test_predictions,
    "key",
    "fare_amount",
    SUBMISSION_NAME,
)

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1709536507.py in <cell line: 0>()
----> 1 test_predictions = model.predict(X_test)
      2 test_predictions = np.clip(test_predictions, 0.0, 50.0)
      3 
      4 output_submission(
      5     pd.DataFrame({"key": test_ids}),

NameError: name 'X_test' is not defined
