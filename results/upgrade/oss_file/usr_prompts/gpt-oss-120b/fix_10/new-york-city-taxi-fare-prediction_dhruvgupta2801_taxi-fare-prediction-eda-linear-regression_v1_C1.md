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

# 5. Target score

5.68915

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 936.92806) has done: 'The fix removes the deprecated `normalize` argument from `LinearRegression`, allowing the model to be trained correctly. With the model defined, subsequent cells can generate predictions, build the required submission DataFrame, and write a valid `Submission.csv` file containing the `key` and `fare_amount` columns.'
- What this solution (achieved 6.27662) has done: 'We speed up the pipeline by (1) loading fewer rows (5 M instead of 10 M) while keeping the same 30 % sampling, (2) forcing all engineered numeric columns to float32 to reduce memory bandwidth, and (3) converting the feature matrices to float32 once before fitting the RandomForest. These changes are purely performance‑oriented and do not alter the modelling logic or the resulting predictions.'

# 9. Code solution

## === cell 0
import os
from pathlib import Path

import numpy as np
import pandas as pd


def locate_file(filename: str) -> str:
    """
    Return a path to *filename* searching a few common data folders.
    """
    possible_dirs = [
        Path.cwd() / "data",
        Path.cwd() / "kaggle" / "data",
        Path.cwd() / "kaggle" / "input",
    ]
    for d in possible_dirs:
        candidate = d / filename
        if candidate.is_file():
            return str(candidate)
    return filename




## === cell 1
train_path = locate_file("train.csv")
train_data = pd.read_csv(
    train_path,
    nrows=5_000_000,  # reduced from full set
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



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/643631161.py in <cell line: 0>()
      1 # Load a reduced sample of the training data for speed
      2 train_path = locate_file("train.csv")
----> 3 train_data = pd.read_csv(
      4     train_path,
      5     nrows=5_000_000,  # reduced from full set

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

FileNotFoundError: [Errno 2] No such file or directory: 'train.csv'

## === cell 2
print("Train shape after sampling:", train_data.shape)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4085960100.py in <cell line: 0>()
----> 1 print("Train shape after sampling:", train_data.shape)
      2 

NameError: name 'train_data' is not defined

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



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/4053285597.py in <cell line: 0>()
      1 test_path = locate_file("test.csv")
----> 2 test_data = pd.read_csv(
      3     test_path,
      4     dtype={
      5         "key": "object",

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

FileNotFoundError: [Errno 2] No such file or directory: 'test.csv'

## === cell 4
print("Test shape:", test_data.shape)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/981603602.py in <cell line: 0>()
----> 1 print("Test shape:", test_data.shape)
      2 

NameError: name 'test_data' is not defined

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



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3892965853.py in <cell line: 0>()
      1 # Basic engineered distance features (absolute coordinate differences)
      2 train_data["Difference_longitude"] = np.abs(
----> 3     train_data["pickup_longitude"] - train_data["dropoff_longitude"]
      4 )
      5 train_data["Difference_latitude"] = np.abs(

NameError: name 'train_data' is not defined

## === cell 6
print(f"Before dropping null values: {len(train_data)}")
train_data.dropna(inplace=True)
print(f"After dropping null values: {len(train_data)}")



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2415293739.py in <cell line: 0>()
----> 1 print(f"Before dropping null values: {len(train_data)}")
      2 train_data.dropna(inplace=True)
      3 print(f"After dropping null values: {len(train_data)}")
      4 

NameError: name 'train_data' is not defined

## === cell 7
_ = train_data[:2000].plot.scatter("Difference_longitude", "Difference_latitude")



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4247987877.py in <cell line: 0>()
      1 # Simple scatter just to confirm data (optional visual, kept for parity)
      2 # plt may not be needed; we just ensure no error
----> 3 _ = train_data[:2000].plot.scatter("Difference_longitude", "Difference_latitude")
      4 

NameError: name 'train_data' is not defined

## === cell 8
train_data = train_data[
    (train_data["Difference_longitude"] < 5.0)
    & (train_data["Difference_latitude"] < 5.0)
]



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2195446494.py in <cell line: 0>()
      1 # Filter out extreme coordinate differences which are likely errors
----> 2 train_data = train_data[
      3     (train_data["Difference_longitude"] < 5.0)
      4     & (train_data["Difference_latitude"] < 5.0)
      5 ]

NameError: name 'train_data' is not defined

## === cell 9
train_dt = pd.to_datetime(train_data["pickup_datetime"])
train_data["pickuptime"] = train_dt.dt.hour * 100 + train_dt.dt.minute

test_dt = pd.to_datetime(test_data["pickup_datetime"])
test_data["pickuptime"] = test_dt.dt.hour * 100 + test_dt.dt.minute



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2374487732.py in <cell line: 0>()
      1 # Time‑based features
----> 2 train_dt = pd.to_datetime(train_data["pickup_datetime"])
      3 train_data["pickuptime"] = train_dt.dt.hour * 100 + train_dt.dt.minute
      4 
      5 test_dt = pd.to_datetime(test_data["pickup_datetime"])

NameError: name 'train_data' is not defined

## === cell 10
train_data["Weekday"] = train_dt.dt.weekday
test_data["Weekday"] = test_dt.dt.weekday



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1425385164.py in <cell line: 0>()
----> 1 train_data["Weekday"] = train_dt.dt.weekday
      2 test_data["Weekday"] = test_dt.dt.weekday
      3 

NameError: name 'train_dt' is not defined

## === cell 11
train_data.drop("pickup_datetime", inplace=True, axis=1)
test_data.drop("pickup_datetime", inplace=True, axis=1)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/218096511.py in <cell line: 0>()
      1 # Drop raw datetime string (no longer needed)
----> 2 train_data.drop("pickup_datetime", inplace=True, axis=1)
      3 test_data.drop("pickup_datetime", inplace=True, axis=1)
      4 

NameError: name 'train_data' is not defined

## === cell 12
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



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/410465164.py in <cell line: 0>()
      9     6: "Sunday",
     10 }
---> 11 train_data["Weekday"] = train_data["Weekday"].map(weekday_map)
     12 test_data["Weekday"] = test_data["Weekday"].map(weekday_map)
     13 

NameError: name 'train_data' is not defined

## === cell 13
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



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1801978145.py in <cell line: 0>()
      1 # One‑hot encode the weekday names
----> 2 train_one_hot = pd.get_dummies(train_data["Weekday"], prefix="Weekday")
      3 test_one_hot = pd.get_dummies(test_data["Weekday"], prefix="Weekday")
      4 train_data = pd.concat([train_data, train_one_hot], axis=1)
      5 test_data = pd.concat([test_data, test_one_hot], axis=1)

NameError: name 'train_data' is not defined

## === cell 14
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
train_data["Distance"] = (distance * 0.621).astype(np.float32)  # convert km → miles

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



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3673873121.py in <cell line: 0>()
      1 # Haversine distance (in miles)
      2 R = 6373.0  # Earth radius in km
----> 3 lat1 = np.radians(train_data["pickup_latitude"].astype(np.float32))
      4 lon1 = np.radians(train_data["pickup_longitude"].astype(np.float32))
      5 lat2 = np.radians(train_data["dropoff_latitude"].astype(np.float32))

NameError: name 'train_data' is not defined

## === cell 15
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



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3384314512.py in <cell line: 0>()
      4 
      5 # Pickup distance to airport
----> 6 lat1 = np.radians(train_data["pickup_latitude"].astype(np.float32))
      7 lon1 = np.radians(train_data["pickup_longitude"].astype(np.float32))
      8 dlon_pickup = lon_air - lon1

NameError: name 'train_data' is not defined

## === cell 16
numeric_cols = ["Distance", "Pickup_Distance_airport", "Dropoff_Distance_airport"]
for col in numeric_cols:
    train_data[col] = np.round(train_data[col], 2).astype(np.float32)
    test_data[col] = np.round(test_data[col], 2).astype(np.float32)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1593952821.py in <cell line: 0>()
      2 numeric_cols = ["Distance", "Pickup_Distance_airport", "Dropoff_Distance_airport"]
      3 for col in numeric_cols:
----> 4     train_data[col] = np.round(train_data[col], 2).astype(np.float32)
      5     test_data[col] = np.round(test_data[col], 2).astype(np.float32)
      6 

NameError: name 'train_data' is not defined

## === cell 17
train_data["Difference_longitude"] = np.abs(
    train_data["Difference_longitude"] - np.mean(train_data["Difference_longitude"])
)
train_data["Difference_longitude"] = (
    train_data["Difference_longitude"] / np.var(train_data["Difference_longitude"])
).astype(np.float32)

train_data["Difference_latitude"] = np.abs(
    train_data["Difference_latitude"] - np.mean(train_data["Difference_latitude"])
)
train_data["Difference_latitude"] = (
    train_data["Difference_latitude"] / np.var(train_data["Difference_latitude"])
).astype(np.float32)

test_data["Difference_longitude"] = np.abs(
    test_data["Difference_longitude"] - np.mean(test_data["Difference_longitude"])
)
test_data["Difference_longitude"] = (
    test_data["Difference_longitude"] / np.var(test_data["Difference_longitude"])
).astype(np.float32)

test_data["Difference_latitude"] = np.abs(
    test_data["Difference_latitude"] - np.mean(test_data["Difference_latitude"])
)
test_data["Difference_latitude"] = (
    test_data["Difference_latitude"] / np.var(test_data["Difference_latitude"])
).astype(np.float32)



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4142947292.py in <cell line: 0>()
      1 # Standardise the absolute coordinate differences
      2 train_data["Difference_longitude"] = np.abs(
----> 3     train_data["Difference_longitude"] - np.mean(train_data["Difference_longitude"])
      4 )
      5 train_data["Difference_longitude"] = (

NameError: name 'train_data' is not defined

## === cell 18
print("Final train shape:", train_data.shape)
print("Final test shape:", test_data.shape)



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/639026863.py in <cell line: 0>()
----> 1 print("Final train shape:", train_data.shape)
      2 print("Final test shape:", test_data.shape)
      3 

NameError: name 'train_data' is not defined

## === cell 19
from sklearnex import patch_sklearn

patch_sklearn()
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



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2906743995.py in <cell line: 0>()
      7 from sklearn.ensemble import RandomForestRegressor
      8 
----> 9 X = train_data.drop(["key", "fare_amount"], axis=1).astype(np.float32)
     10 y = train_data["fare_amount"].astype(np.float32)
     11 

NameError: name 'train_data' is not defined

## === cell 20
test_features = test_data.drop("key", axis=1).values.astype(np.float32, copy=False)
pred_raw = rf.predict(test_features)

pred = np.round(np.clip(pred_raw, a_min=0, a_max=None), 2)



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3011570033.py in <cell line: 0>()
----> 1 test_features = test_data.drop("key", axis=1).values.astype(np.float32, copy=False)
      2 pred_raw = rf.predict(test_features)
      3 
      4 # Clip negative predictions and round to two decimals (as required by competition)
      5 pred = np.round(np.clip(pred_raw, a_min=0, a_max=None), 2)

NameError: name 'test_data' is not defined

## === cell 21
Submission = pd.DataFrame({"key": test_data["key"], "fare_amount": pred})
Submission = Submission[["key", "fare_amount"]]
Submission.to_csv("Submission.csv", index=False)
print("Submission saved to Submission.csv")

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1963040441.py in <cell line: 0>()
      1 # Build the submission file
----> 2 Submission = pd.DataFrame({"key": test_data["key"], "fare_amount": pred})
      3 Submission = Submission[["key", "fare_amount"]]
      4 Submission.to_csv("Submission.csv", index=False)
      5 print("Submission saved to Submission.csv")

NameError: name 'test_data' is not defined
