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
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor



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
train_path = Path("./data/train.csv")
train_df = pd.read_csv(
    train_path,
    nrows=1_000_000,
    usecols=usecols,
    dtype=dtypes,
    low_memory=False,
)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3309833061.py in <cell line: 0>()
     19 }
     20 train_path = Path("./data/train.csv")
---> 21 train_df = pd.read_csv(
     22     train_path,
     23     nrows=1_000_000,

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
/tmp/ipykernel_11/3841327952.py in <cell line: 0>()
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
/tmp/ipykernel_11/1813579294.py in <cell line: 0>()
----> 1 train_df["pickup_datetime"] = pd.to_datetime(train_df["pickup_datetime"])
      2 

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
/tmp/ipykernel_11/3185049235.py in <cell line: 0>()
----> 1 train_df["year"] = train_df["pickup_datetime"].dt.year.astype(np.int16)
      2 train_df["month"] = train_df["pickup_datetime"].dt.month.astype(np.int8)
      3 train_df["day"] = train_df["pickup_datetime"].dt.day.astype(np.int8)
      4 train_df["hour"] = train_df["pickup_datetime"].dt.hour.astype(np.int8)
      5 train_df["minute"] = train_df["pickup_datetime"].dt.minute.astype(np.int8)

NameError: name 'train_df' is not defined

## === cell 6
print("Old size: %d" % len(train_df))
train_df.dropna(inplace=True)
print("New size: %d" % len(train_df))



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3333111588.py in <cell line: 0>()
----> 1 print("Old size: %d" % len(train_df))
      2 train_df.dropna(inplace=True)
      3 print("New size: %d" % len(train_df))
      4 

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
/tmp/ipykernel_11/1919882599.py in <cell line: 0>()
      6     "dropoff_longitude",
      7 }:
----> 8     cond &= np.abs(train_df[col] - train_df[col].mean()) < 5
      9 cond &= train_df["fare_amount"] > 0
     10 

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
/tmp/ipykernel_11/952173888.py in <cell line: 0>()
      5     "dropoff_longitude",
      6 }:
----> 7     train_df["rough" + col] = train_df[col].round(2)
      8 

NameError: name 'train_df' is not defined

## === cell 9
a = train_df.groupby(["roughpickup_latitude", "roughpickup_longitude"], observed=True)[
    ["pickup_latitude", "pickup_longitude"]
].agg(["mean", "count"])



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3139378327.py in <cell line: 0>()
----> 1 a = train_df.groupby(["roughpickup_latitude", "roughpickup_longitude"], observed=True)[
      2     ["pickup_latitude", "pickup_longitude"]
      3 ].agg(["mean", "count"])
      4 

NameError: name 'train_df' is not defined

## === cell 10
a.columns = ["mean_pickup_latitude", "c1", "mean_pickup_longitude", "c2"]
a = a[["c1"]].reset_index().rename(columns={"c1": "pickup_busyness"})



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/324697715.py in <cell line: 0>()
----> 1 a.columns = ["mean_pickup_latitude", "c1", "mean_pickup_longitude", "c2"]
      2 a = a[["c1"]].reset_index().rename(columns={"c1": "pickup_busyness"})
      3 

NameError: name 'a' is not defined

## === cell 11
b = train_df.groupby(
    ["roughdropoff_latitude", "roughdropoff_longitude"], observed=True
)[["dropoff_latitude", "dropoff_longitude"]].agg(["mean", "count"])



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/745548913.py in <cell line: 0>()
----> 1 b = train_df.groupby(
      2     ["roughdropoff_latitude", "roughdropoff_longitude"], observed=True
      3 )[["dropoff_latitude", "dropoff_longitude"]].agg(["mean", "count"])
      4 

NameError: name 'train_df' is not defined

## === cell 12
b.columns = ["mean_dropoff_latitude", "c1", "mean_dropoff_longitude", "c2"]
b = b[["c1"]].reset_index().rename(columns={"c1": "dropoff_busyness"})



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3517007385.py in <cell line: 0>()
----> 1 b.columns = ["mean_dropoff_latitude", "c1", "mean_dropoff_longitude", "c2"]
      2 b = b[["c1"]].reset_index().rename(columns={"c1": "dropoff_busyness"})
      3 

NameError: name 'b' is not defined

## === cell 13
train_df = pd.merge(
    train_df, a, how="left", on=["roughpickup_latitude", "roughpickup_longitude"]
)
train_df = pd.merge(
    train_df, b, how="left", on=["roughdropoff_latitude", "roughdropoff_longitude"]
)
train_df["pickup_busyness"].fillna(1, inplace=True)
train_df["dropoff_busyness"].fillna(1, inplace=True)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3250977438.py in <cell line: 0>()
      1 train_df = pd.merge(
----> 2     train_df, a, how="left", on=["roughpickup_latitude", "roughpickup_longitude"]
      3 )
      4 train_df = pd.merge(
      5     train_df, b, how="left", on=["roughdropoff_latitude", "roughdropoff_longitude"]

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
/tmp/ipykernel_11/2748665841.py in <cell line: 0>()
----> 1 X = train_df[
      2     [
      3         "distance",
      4         "year",
      5         "month",

NameError: name 'train_df' is not defined

## === cell 15
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

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
/tmp/ipykernel_11/1029999091.py in <cell line: 0>()
----> 1 X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
      2 
      3 y_train_log = np.log1p(y_train)
      4 y_val_log = np.log1p(y_val)
      5 

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
/tmp/ipykernel_11/815704717.py in <cell line: 0>()
----> 1 Y_log_full = np.log1p(Y)
      2 rf_full = RandomForestRegressor(
      3     n_estimators=300,
      4     max_features=None,
      5     min_samples_leaf=1,

NameError: name 'Y' is not defined

## === cell 17
test_path = Path("./data/test.csv")
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
/tmp/ipykernel_11/2793737478.py in <cell line: 0>()
      1 test_path = Path("./data/test.csv")
----> 2 test_df = pd.read_csv(
      3     test_path,
      4     usecols=[
      5         "key",

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

FileNotFoundError: [Errno 2] No such file or directory: 'data/test.csv'

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
/tmp/ipykernel_11/2302251183.py in <cell line: 0>()
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
/tmp/ipykernel_11/1067261872.py in <cell line: 0>()
----> 1 test_df["pickup_datetime"] = pd.to_datetime(test_df["pickup_datetime"])
      2 

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
/tmp/ipykernel_11/341879188.py in <cell line: 0>()
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
/tmp/ipykernel_11/44095779.py in <cell line: 0>()
      5     "dropoff_longitude",
      6 }:
----> 7     test_df["rough" + col] = test_df[col].round(2)
      8 

NameError: name 'test_df' is not defined

## === cell 22
test_df = pd.merge(
    test_df, a, how="left", on=["roughpickup_latitude", "roughpickup_longitude"]
)
test_df = pd.merge(
    test_df, b, how="left", on=["roughdropoff_latitude", "roughdropoff_longitude"]
)
test_df["pickup_busyness"].fillna(1, inplace=True)
test_df["dropoff_busyness"].fillna(1, inplace=True)



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/379138138.py in <cell line: 0>()
      1 test_df = pd.merge(
----> 2     test_df, a, how="left", on=["roughpickup_latitude", "roughpickup_longitude"]
      3 )
      4 test_df = pd.merge(
      5     test_df, b, how="left", on=["roughdropoff_latitude", "roughdropoff_longitude"]

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



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1428275960.py in <cell line: 0>()
----> 1 X_test = test_df[
      2     [
      3         "distance",
      4         "year",
      5         "month",

NameError: name 'test_df' is not defined

## === cell 24
test_pred_log = rf_full.predict(X_test)
test_pred = np.expm1(test_pred_log)



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1254015325.py in <cell line: 0>()
----> 1 test_pred_log = rf_full.predict(X_test)
      2 test_pred = np.expm1(test_pred_log)
      3 

NameError: name 'rf_full' is not defined

## === cell 25
submission = pd.DataFrame(
    {"key": test_df["key"], "fare_amount": test_pred},
    columns=["key", "fare_amount"],
)
submission.to_csv("submission.csv", index=False)

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2671661369.py in <cell line: 0>()
      1 submission = pd.DataFrame(
----> 2     {"key": test_df["key"], "fare_amount": test_pred},
      3     columns=["key", "fare_amount"],
      4 )
      5 submission.to_csv("submission.csv", index=False)

NameError: name 'test_df' is not defined
