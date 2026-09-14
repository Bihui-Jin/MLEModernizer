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
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
sklearn-pandas==2.2.0
tf_keras==2.18.0

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

4.30351

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 15.27447) has done: 'The changes fix the image‑loading error in `remove_datapoints_from_water`, update the optimizer to the current Keras API, correct variable names for the predictions, and add small robustness tweaks (e.g., safe imports, increasing the sampled training size for a slightly better model). These fixes allow the notebook to run end‑to‑end and produce a valid `.csv` submission while keeping the original modelling approach.'
- What this solution (achieved 98.75948) has done: 'The update fixes the Keras import errors by switching to `tensorflow.keras`, restores the missing backend functions (e.g., `sqrt`), adds a modest EarlyStopping callback, and expands the training sample size and epochs so the model can reach a lower RMSE while keeping the original architecture unchanged. The script now runs end‑to‑end and writes a proper `.csv` submission.'
- What this solution (achieved 15.44038) has done: 'Implemented fixes to unblock execution and improve model performance:
- Switched all TensorFlow‑Keras imports to the standalone `keras` package to avoid protobuf errors.
- Corrected the `late_night` logic (now true only for hours ≤ 3).
- Improved datetime parsing in `add_time_features` and dropped rows with any resulting NaNs.
- Updated the backend import for the custom RMSE metric.
- Minor clean‑up comments kept unchanged.'
- What this solution (achieved 297.48078) has done: 'The fix switches all Keras imports to `tensorflow.keras`, which restores the missing backend functions (e.g., `K.sqrt`) and resolves the protobuf import errors. Only the import lines and the backend import are changed, preserving the original model architecture and training logic.'
- What this solution (achieved 14.15588) has done: 'Implemented missing preprocessing, feature‑engineering, and submission utilities, added NaN handling, and ensured the GradientBoostingRegressor receives clean data. The new functions (`clean`, `add_time_features`, `add_coordinate_features`, `add_distances_features`, `output_submission`) safely process timestamps, compute hour/weekday, coordinate differences, and haversine distance, then drop any rows with missing values. Predictions are now written to a proper `submissiontry_water.csv` with the required columns, allowing the script to run end‑to‑end and produce a valid submission while keeping the original modeling approach unchanged.'
- What this solution (achieved 10.12988) has done: 'The changes reduce the data sampled for training (from 500 k to 200 k rows) and lower the number of trees in the GradientBoostingRegressor (from 800 to 200). Both adjustments keep the same preprocessing, feature engineering, and model type, only trimming size‑related hyperparameters to fit comfortably within the 600 s limit while preserving prediction semantics.'
- What this solution (achieved 6.38009) has done: 'I filter out rows with non‑positive fare amounts before the log‑transform so that `np.log1p` never receives a negative value, which caused NaNs in the target array and prevented the GradientBoostingRegressor from fitting. The rest of the pipeline stays unchanged, and a valid CSV submission is produced.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt

TRAIN_PATH = os.getenv("TRAIN_PATH", "data/train.csv")
TEST_PATH = os.getenv("TEST_PATH", "data/test.csv")
SUBMISSION_NAME = os.getenv("SUBMISSION_NAME", "submission.csv")
DATASET_SIZE = int(os.getenv("DATASET_SIZE", "200000"))  # set to None to read all


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning:
    - Drop rows with NaNs
    - Keep coordinates inside typical NYC bounds
    - Keep reasonable passenger counts
    """
    lon_min, lon_max = -74.5, -73.5
    lat_min, lat_max = 40.5, 41.0
    mask = (
        (df["pickup_longitude"].between(lon_min, lon_max))
        & (df["dropoff_longitude"].between(lon_min, lon_max))
        & (df["pickup_latitude"].between(lat_min, lat_max))
        & (df["dropoff_latitude"].between(lat_min, lat_max))
        & (df["passenger_count"] > 0)
        & (df["passenger_count"] <= 6)
    )
    return df[mask].dropna().reset_index(drop=True)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Parse pickup_datetime and add hour, dayofweek, month."""
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df["pickup_hour"] = dt.dt.hour
    df["pickup_dayofweek"] = dt.dt.dayofweek
    df["pickup_month"] = dt.dt.month
    return df


def add_coordinate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add delta latitude/longitude."""
    df["delta_lon"] = df["dropoff_longitude"] - df["pickup_longitude"]
    df["delta_lat"] = df["dropoff_latitude"] - df["pickup_latitude"]
    return df


def haversine_vectorized(lon1, lat1, lon2, lat2):
    """Compute haversine distance (km) for array‑like inputs."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371.0 * c
    return km


def add_distances_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add haversine distance as a feature."""
    df["haversine_km"] = haversine_vectorized(
        df["pickup_longitude"],
        df["pickup_latitude"],
        df["dropoff_longitude"],
        df["dropoff_latitude"],
    )
    return df


def output_submission(
    test_df_original: pd.DataFrame,
    preds: np.ndarray,
    key_col: str,
    target_col: str,
    filename: str,
):
    """Write Kaggle submission CSV with required columns."""
    submission = pd.DataFrame(
        {key_col: test_df_original[key_col], target_col: preds.ravel()}
    )
    submission.to_csv(filename, index=False)
    print(f"Submission written to {filename}")


def rmse_metric(y_true, y_pred):
    return np.sqrt(np.mean((y_pred - y_true) ** 2))


trainKaggle = pd.read_csv(
    TRAIN_PATH,
    nrows=DATASET_SIZE,
    dtype={
        "fare_amount": "float32",
        "pickup_datetime": "str",
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "uint8",
    },
    usecols=[
        "fare_amount",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
)

testKaggle = pd.read_csv(
    TEST_PATH,
    dtype={
        "key": "str",
        "pickup_datetime": "str",
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "uint8",
    },
)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/4215777767.py in <cell line: 0>()
     96 # -------------------------------------------------
     97 # Load data -------------------------------------------------
---> 98 trainKaggle = pd.read_csv(
     99     TRAIN_PATH,
    100     nrows=DATASET_SIZE,

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: 'data/train.csv'

## === cell 1
train_df, val_df = train_test_split(trainKaggle, test_size=0.20, random_state=1)
val_df = val_df[:10000]



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3362650292.py in <cell line: 0>()
      1 # Train / validation split (20 % for validation)
----> 2 train_df, val_df = train_test_split(trainKaggle, test_size=0.20, random_state=1)
      3 # Limit validation to 10 k rows for speed
      4 val_df = val_df[:10000]
      5 

NameError: name 'trainKaggle' is not defined

## === cell 2
print(f"testKaggle Size {len(testKaggle)}")
print(f"train_df Size {len(train_df)}")
print(f"val_df Size {len(val_df)}")



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2733369522.py in <cell line: 0>()
----> 1 print(f"testKaggle Size {len(testKaggle)}")
      2 print(f"train_df Size {len(train_df)}")
      3 print(f"val_df Size {len(val_df)}")
      4 

NameError: name 'testKaggle' is not defined

## === cell 3
train_df = clean(train_df)
val_df = clean(val_df)
testKaggle = clean(testKaggle)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/97295341.py in <cell line: 0>()
      1 # Cleaning
----> 2 train_df = clean(train_df)
      3 val_df = clean(val_df)
      4 testKaggle = clean(testKaggle)
      5 

NameError: name 'train_df' is not defined

## === cell 4
train_df = add_time_features(train_df)
val_df = add_time_features(val_df)
testKaggle = add_time_features(testKaggle)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/286254195.py in <cell line: 0>()
      1 # Time features
----> 2 train_df = add_time_features(train_df)
      3 val_df = add_time_features(val_df)
      4 testKaggle = add_time_features(testKaggle)
      5 

NameError: name 'train_df' is not defined

## === cell 5
train_df = add_coordinate_features(train_df)
val_df = add_coordinate_features(val_df)
testKaggle = add_coordinate_features(testKaggle)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/105962696.py in <cell line: 0>()
      1 # Coordinate delta features
----> 2 train_df = add_coordinate_features(train_df)
      3 val_df = add_coordinate_features(val_df)
      4 testKaggle = add_coordinate_features(testKaggle)
      5 

NameError: name 'train_df' is not defined

## === cell 6
train_df = add_distances_features(train_df)
val_df = add_distances_features(val_df)
testKaggle = add_distances_features(testKaggle)
print("Feature engineering completed")



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2667056852.py in <cell line: 0>()
      1 # Distance features
----> 2 train_df = add_distances_features(train_df)
      3 val_df = add_distances_features(val_df)
      4 testKaggle = add_distances_features(testKaggle)
      5 print("Feature engineering completed")

NameError: name 'train_df' is not defined

## === cell 7
drop_cols = ["pickup_datetime"]
train_df = train_df.drop(columns=drop_cols)
val_df = val_df.drop(columns=drop_cols)
testKaggle_clean = testKaggle.drop(columns=drop_cols + ["key"])



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2382608348.py in <cell line: 0>()
      1 # Drop raw datetime column (no longer needed)
      2 drop_cols = ["pickup_datetime"]
----> 3 train_df = train_df.drop(columns=drop_cols)
      4 val_df = val_df.drop(columns=drop_cols)
      5 testKaggle_clean = testKaggle.drop(columns=drop_cols + ["key"])

NameError: name 'train_df' is not defined

## === cell 8
pos_mask_train = train_df["fare_amount"] > 0
pos_mask_val = val_df["fare_amount"] > 0
train_df = train_df[pos_mask_train].reset_index(drop=True)
val_df = val_df[pos_mask_val].reset_index(drop=True)

train_labels = train_df["fare_amount"].values
val_labels = val_df["fare_amount"].values

train_labels_log = np.log1p(train_labels)
val_labels_log = np.log1p(val_labels)

train_features = train_df.drop(columns=["fare_amount"])
val_features = val_df.drop(columns=["fare_amount"])

print("Labels prepared (original and log‑transformed)")



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1096866610.py in <cell line: 0>()
      1 # Prepare labels
----> 2 pos_mask_train = train_df["fare_amount"] > 0
      3 pos_mask_val = val_df["fare_amount"] > 0
      4 train_df = train_df[pos_mask_train].reset_index(drop=True)
      5 val_df = val_df[pos_mask_val].reset_index(drop=True)

NameError: name 'train_df' is not defined

## === cell 9
train_X = train_features.values
val_X = val_features.values
test_X = testKaggle_clean.values



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3728621550.py in <cell line: 0>()
      1 # Convert to numpy arrays for the model
----> 2 train_X = train_features.values
      3 val_X = val_features.values
      4 test_X = testKaggle_clean.values
      5 

NameError: name 'train_features' is not defined

## === cell 10
gbr = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    random_state=42,
)
gbr.fit(train_X, train_labels_log)

print("Model training complete")
print(f"Features used: {list(train_features.columns)}")



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3801713408.py in <cell line: 0>()
      7     random_state=42,
      8 )
----> 9 gbr.fit(train_X, train_labels_log)
     10 
     11 print("Model training complete")

NameError: name 'train_X' is not defined

## === cell 11
train_pred_log = gbr.predict(train_X)
val_pred_log = gbr.predict(val_X)

train_pred = np.expm1(train_pred_log)
val_pred = np.expm1(val_pred_log)

train_score = [
    rmse_metric(train_labels, train_pred),
    np.mean(np.abs(train_labels - train_pred)),
]
val_score = [rmse_metric(val_labels, val_pred), np.mean(np.abs(val_labels - val_pred))]
print("Train scores (RMSE, MAE):", train_score)
print("Validation scores (RMSE, MAE):", val_score)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1523367710.py in <cell line: 0>()
      1 # Predictions on train / validation (log space → original)
----> 2 train_pred_log = gbr.predict(train_X)
      3 val_pred_log = gbr.predict(val_X)
      4 
      5 train_pred = np.expm1(train_pred_log)

NameError: name 'train_X' is not defined

## === cell 12
test_pred_log = gbr.predict(test_X)
test_pred = np.expm1(test_pred_log)
print("Test predictions generated")
print("Prediction range:", test_pred.min(), "to", test_pred.max())



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2000225739.py in <cell line: 0>()
      1 # Test set predictions
----> 2 test_pred_log = gbr.predict(test_X)
      3 test_pred = np.expm1(test_pred_log)
      4 print("Test predictions generated")
      5 print("Prediction range:", test_pred.min(), "to", test_pred.max())

NameError: name 'test_X' is not defined

## === cell 13
fig, ax = plt.subplots()
ax.scatter(val_labels, val_pred, alpha=0.5, s=10)
ax.plot(
    [val_labels.min(), val_labels.max()],
    [val_labels.min(), val_labels.max()],
    "k--",
    lw=2,
)
ax.set_xlabel("Measured fare")
ax.set_ylabel("Predicted fare")
plt.show()



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/341732504.py in <cell line: 0>()
      1 # Optional scatter plot for validation diagnostics
      2 fig, ax = plt.subplots()
----> 3 ax.scatter(val_labels, val_pred, alpha=0.5, s=10)
      4 ax.plot(
      5     [val_labels.min(), val_labels.max()],

NameError: name 'val_labels' is not defined

## === cell 14
output_submission(testKaggle, test_pred, "key", "fare_amount", SUBMISSION_NAME)

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3617229766.py in <cell line: 0>()
      1 # Write submission file
----> 2 output_submission(testKaggle, test_pred, "key", "fare_amount", SUBMISSION_NAME)

NameError: name 'testKaggle' is not defined
