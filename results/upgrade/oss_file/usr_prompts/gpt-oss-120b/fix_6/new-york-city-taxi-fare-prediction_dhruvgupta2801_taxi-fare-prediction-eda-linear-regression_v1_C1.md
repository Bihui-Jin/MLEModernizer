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

# 8. Previous improvement plan

- What this solution (achieved 936.92806) has done: 'The fix removes the deprecated `normalize` argument from `LinearRegression`, allowing the model to be trained correctly. With the model defined, subsequent cells can generate predictions, build the required submission DataFrame, and write a valid `Submission.csv` file containing the `key` and `fare_amount` columns.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
train_data = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-prediction/train.csv",
    nrows=10_000_000,
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
train_data.head()



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/229370017.py in <cell line: 0>()
----> 1 train_data = pd.read_csv(
      2     "/kaggle/input/new-york-city-taxi-prediction/train.csv",
      3     nrows=10_000_000,
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

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/new-york-city-taxi-prediction/train.csv'

## === cell 2
train_data.shape



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4031933074.py in <cell line: 0>()
----> 1 train_data.shape
      2 

NameError: name 'train_data' is not defined

## === cell 3
train_data.info()



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4151811981.py in <cell line: 0>()
----> 1 train_data.info()
      2 

NameError: name 'train_data' is not defined

## === cell 4
test_data = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-prediction/test.csv",
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
test_data.head()



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2367470456.py in <cell line: 0>()
----> 1 test_data = pd.read_csv(
      2     "/kaggle/input/new-york-city-taxi-prediction/test.csv",
      3     dtype={
      4         "key": "object",
      5         "pickup_datetime": "object",

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

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/new-york-city-taxi-prediction/test.csv'

## === cell 5
test_data.info()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1702245187.py in <cell line: 0>()
----> 1 test_data.info()
      2 

NameError: name 'test_data' is not defined

## === cell 6
train_data.isna().sum()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1085632216.py in <cell line: 0>()
----> 1 train_data.isna().sum()
      2 

NameError: name 'train_data' is not defined

## === cell 7
train_data["Difference_longitude"] = np.abs(
    np.asarray(train_data["pickup_longitude"] - train_data["dropoff_longitude"])
)
train_data["Difference_latitude"] = np.abs(
    np.asarray(train_data["pickup_latitude"] - train_data["dropoff_latitude"])
)

test_data["Difference_longitude"] = np.abs(
    np.asarray(test_data["pickup_longitude"] - test_data["dropoff_longitude"])
)
test_data["Difference_latitude"] = np.abs(
    np.asarray(test_data["pickup_latitude"] - test_data["dropoff_latitude"])
)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2029277579.py in <cell line: 0>()
      1 train_data["Difference_longitude"] = np.abs(
----> 2     np.asarray(train_data["pickup_longitude"] - train_data["dropoff_longitude"])
      3 )
      4 train_data["Difference_latitude"] = np.abs(
      5     np.asarray(train_data["pickup_latitude"] - train_data["dropoff_latitude"])

NameError: name 'train_data' is not defined

## === cell 8
print(f"Before Dropping null values: {len(train_data)}")
train_data.dropna(inplace=True)
print(f"After Dropping null values: {len(train_data)}")



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4109314509.py in <cell line: 0>()
----> 1 print(f"Before Dropping null values: {len(train_data)}")
      2 train_data.dropna(inplace=True)
      3 print(f"After Dropping null values: {len(train_data)}")
      4 

NameError: name 'train_data' is not defined

## === cell 9
plot = train_data[:2000].plot.scatter("Difference_longitude", "Difference_latitude")



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1169734720.py in <cell line: 0>()
----> 1 plot = train_data[:2000].plot.scatter("Difference_longitude", "Difference_latitude")
      2 

NameError: name 'train_data' is not defined

## === cell 10
train_data = train_data[
    (train_data["Difference_longitude"] < 5.0)
    & (train_data["Difference_latitude"] < 5.0)
]



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/297526157.py in <cell line: 0>()
----> 1 train_data = train_data[
      2     (train_data["Difference_longitude"] < 5.0)
      3     & (train_data["Difference_latitude"] < 5.0)
      4 ]
      5 

NameError: name 'train_data' is not defined

## === cell 11
train_dt = pd.to_datetime(train_data["pickup_datetime"])
train_data["pickuptime"] = train_dt.dt.hour * 100 + train_dt.dt.minute

test_dt = pd.to_datetime(test_data["pickup_datetime"])
test_data["pickuptime"] = test_dt.dt.hour * 100 + test_dt.dt.minute



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1189162366.py in <cell line: 0>()
----> 1 train_dt = pd.to_datetime(train_data["pickup_datetime"])
      2 train_data["pickuptime"] = train_dt.dt.hour * 100 + train_dt.dt.minute
      3 
      4 test_dt = pd.to_datetime(test_data["pickup_datetime"])
      5 test_data["pickuptime"] = test_dt.dt.hour * 100 + test_dt.dt.minute

NameError: name 'train_data' is not defined

## === cell 12
train_data["Weekday"] = pd.to_datetime(train_data["pickup_datetime"]).dt.weekday
test_data["Weekday"] = pd.to_datetime(test_data["pickup_datetime"]).dt.weekday



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3665707052.py in <cell line: 0>()
----> 1 train_data["Weekday"] = pd.to_datetime(train_data["pickup_datetime"]).dt.weekday
      2 test_data["Weekday"] = pd.to_datetime(test_data["pickup_datetime"]).dt.weekday
      3 

NameError: name 'train_data' is not defined

## === cell 13
train_data.head()



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3467491927.py in <cell line: 0>()
----> 1 train_data.head()
      2 

NameError: name 'train_data' is not defined

## === cell 14
test_data.head()



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/768455919.py in <cell line: 0>()
----> 1 test_data.head()
      2 

NameError: name 'test_data' is not defined

## === cell 15
train_data.drop("pickup_datetime", inplace=True, axis=1)
test_data.drop("pickup_datetime", inplace=True, axis=1)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/764264539.py in <cell line: 0>()
----> 1 train_data.drop("pickup_datetime", inplace=True, axis=1)
      2 test_data.drop("pickup_datetime", inplace=True, axis=1)
      3 

NameError: name 'train_data' is not defined

## === cell 16
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



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3040602107.py in <cell line: 0>()
      8     6: "Sunday",
      9 }
---> 10 train_data["Weekday"] = train_data["Weekday"].map(weekday_map)
     11 test_data["Weekday"] = test_data["Weekday"].map(weekday_map)
     12 

NameError: name 'train_data' is not defined

## === cell 17
train_one_hot = pd.get_dummies(train_data["Weekday"])
test_one_hot = pd.get_dummies(test_data["Weekday"])
train_data = pd.concat([train_data, train_one_hot], axis=1)
test_data = pd.concat([test_data, test_one_hot], axis=1)



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3579435750.py in <cell line: 0>()
----> 1 train_one_hot = pd.get_dummies(train_data["Weekday"])
      2 test_one_hot = pd.get_dummies(test_data["Weekday"])
      3 train_data = pd.concat([train_data, train_one_hot], axis=1)
      4 test_data = pd.concat([test_data, test_one_hot], axis=1)
      5 

NameError: name 'train_data' is not defined

## === cell 18
train_data.drop("Weekday", axis=1, inplace=True)
test_data.drop("Weekday", axis=1, inplace=True)



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4278855193.py in <cell line: 0>()
----> 1 train_data.drop("Weekday", axis=1, inplace=True)
      2 test_data.drop("Weekday", axis=1, inplace=True)
      3 

NameError: name 'train_data' is not defined

## === cell 19
pass



## === cell 20
train_data.head()



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3467491927.py in <cell line: 0>()
----> 1 train_data.head()
      2 

NameError: name 'train_data' is not defined

## === cell 21
R = 6373.0
lat1 = np.asarray(np.radians(train_data["pickup_latitude"]))
lon1 = np.asarray(np.radians(train_data["pickup_longitude"]))
lat2 = np.asarray(np.radians(train_data["dropoff_latitude"]))
lon2 = np.asarray(np.radians(train_data["dropoff_longitude"]))

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c

train_data["Distance"] = np.asarray(distance) * 0.621

lat1 = np.asarray(np.radians(test_data["pickup_latitude"]))
lon1 = np.asarray(np.radians(test_data["pickup_longitude"]))
lat2 = np.asarray(np.radians(test_data["dropoff_latitude"]))
lon2 = np.asarray(np.radians(test_data["dropoff_longitude"]))

dlon = lon2 - lon1
dlat = lat2 - lat1

a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
test_data["Distance"] = np.asarray(distance) * 0.621



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2277544786.py in <cell line: 0>()
      1 R = 6373.0
----> 2 lat1 = np.asarray(np.radians(train_data["pickup_latitude"]))
      3 lon1 = np.asarray(np.radians(train_data["pickup_longitude"]))
      4 lat2 = np.asarray(np.radians(train_data["dropoff_latitude"]))
      5 lon2 = np.asarray(np.radians(train_data["dropoff_longitude"]))

NameError: name 'train_data' is not defined

## === cell 22
R = 6373.0
lat_air = np.radians(40.6413111)
lon_air = np.radians(-73.7781391)

lat1 = np.asarray(np.radians(train_data["pickup_latitude"]))
lon1 = np.asarray(np.radians(train_data["pickup_longitude"]))
lat2 = np.asarray(np.radians(train_data["dropoff_latitude"]))
lon2 = np.asarray(np.radians(train_data["dropoff_longitude"]))

dlon_pickup = lon_air - lon1
dlat_pickup = lat_air - lat1
a1 = (
    np.sin(dlat_pickup / 2) ** 2
    + np.cos(lat1) * np.cos(lat_air) * np.sin(dlon_pickup / 2) ** 2
)
c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
distance1 = R * c1
train_data["Pickup_Distance_airport"] = np.asarray(distance1) * 0.621

dlon_dropoff = lon_air - lon2
dlat_dropoff = lat_air - lat2
a2 = (
    np.sin(dlat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat_air) * np.sin(dlon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
train_data["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621

lat1 = np.asarray(np.radians(test_data["pickup_latitude"]))
lon1 = np.asarray(np.radians(test_data["pickup_longitude"]))
lat2 = np.asarray(np.radians(test_data["dropoff_latitude"]))
lon2 = np.asarray(np.radians(test_data["dropoff_longitude"]))

dlon_pickup = lon_air - lon1
dlat_pickup = lat_air - lat1
a1 = (
    np.sin(dlat_pickup / 2) ** 2
    + np.cos(lat1) * np.cos(lat_air) * np.sin(dlon_pickup / 2) ** 2
)
c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
distance1 = R * c1
test_data["Pickup_Distance_airport"] = np.asarray(distance1) * 0.621

dlon_dropoff = lon_air - lon2
dlat_dropoff = lat_air - lat2
a2 = (
    np.sin(dlat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat_air) * np.sin(dlon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
test_data["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3180867059.py in <cell line: 0>()
      3 lon_air = np.radians(-73.7781391)
      4 
----> 5 lat1 = np.asarray(np.radians(train_data["pickup_latitude"]))
      6 lon1 = np.asarray(np.radians(train_data["pickup_longitude"]))
      7 lat2 = np.asarray(np.radians(train_data["dropoff_latitude"]))

NameError: name 'train_data' is not defined

## === cell 23
train_data["Distance"] = np.round(train_data["Distance"], 2)
train_data["Pickup_Distance_airport"] = np.round(
    train_data["Pickup_Distance_airport"], 2
)
train_data["Dropoff_Distance_airport"] = np.round(
    train_data["Dropoff_Distance_airport"], 2
)
test_data["Distance"] = np.round(test_data["Distance"], 2)
test_data["Pickup_Distance_airport"] = np.round(test_data["Pickup_Distance_airport"], 2)
test_data["Dropoff_Distance_airport"] = np.round(
    test_data["Dropoff_Distance_airport"], 2
)



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/257530458.py in <cell line: 0>()
----> 1 train_data["Distance"] = np.round(train_data["Distance"], 2)
      2 train_data["Pickup_Distance_airport"] = np.round(
      3     train_data["Pickup_Distance_airport"], 2
      4 )
      5 train_data["Dropoff_Distance_airport"] = np.round(

NameError: name 'train_data' is not defined

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



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2880163933.py in <cell line: 0>()
----> 1 train_data.drop(
      2     ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
      3     axis=1,
      4     inplace=True,
      5 )

NameError: name 'train_data' is not defined

## === cell 25
train_data["Difference_longitude"] = np.abs(
    train_data["Difference_longitude"] - np.mean(train_data["Difference_longitude"])
)
train_data["Difference_longitude"] = train_data["Difference_longitude"] / np.var(
    train_data["Difference_longitude"]
)



## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3657154957.py in <cell line: 0>()
      1 train_data["Difference_longitude"] = np.abs(
----> 2     train_data["Difference_longitude"] - np.mean(train_data["Difference_longitude"])
      3 )
      4 train_data["Difference_longitude"] = train_data["Difference_longitude"] / np.var(
      5     train_data["Difference_longitude"]

NameError: name 'train_data' is not defined

## === cell 26
train_data["Difference_latitude"] = np.abs(
    train_data["Difference_latitude"] - np.mean(train_data["Difference_latitude"])
)
train_data["Difference_latitude"] = train_data["Difference_latitude"] / np.var(
    train_data["Difference_latitude"]
)



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2012251987.py in <cell line: 0>()
      1 train_data["Difference_latitude"] = np.abs(
----> 2     train_data["Difference_latitude"] - np.mean(train_data["Difference_latitude"])
      3 )
      4 train_data["Difference_latitude"] = train_data["Difference_latitude"] / np.var(
      5     train_data["Difference_latitude"]

NameError: name 'train_data' is not defined

## === cell 27
test_data["Difference_longitude"] = np.abs(
    test_data["Difference_longitude"] - np.mean(test_data["Difference_longitude"])
)
test_data["Difference_longitude"] = test_data["Difference_longitude"] / np.var(
    test_data["Difference_longitude"]
)

test_data["Difference_latitude"] = np.abs(
    test_data["Difference_latitude"] - np.mean(test_data["Difference_latitude"])
)
test_data["Difference_latitude"] = test_data["Difference_latitude"] / np.var(
    test_data["Difference_latitude"]
)



## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1316103064.py in <cell line: 0>()
      1 test_data["Difference_longitude"] = np.abs(
----> 2     test_data["Difference_longitude"] - np.mean(test_data["Difference_longitude"])
      3 )
      4 test_data["Difference_longitude"] = test_data["Difference_longitude"] / np.var(
      5     test_data["Difference_longitude"]

NameError: name 'test_data' is not defined

## === cell 28
train_data.shape



## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4031933074.py in <cell line: 0>()
----> 1 train_data.shape
      2 

NameError: name 'train_data' is not defined

## === cell 29
test_data.shape



## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1276008450.py in <cell line: 0>()
----> 1 test_data.shape
      2 

NameError: name 'test_data' is not defined

## === cell 30
from sklearnex import patch_sklearn

patch_sklearn()

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

X = train_data.drop(["key", "fare_amount"], axis=1)
y = train_data["fare_amount"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.01, random_state=80)

X_train_np = X_train.to_numpy(dtype=np.float32, copy=False)
X_val_np = X_val.to_numpy(dtype=np.float32, copy=False)

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    n_jobs=-1,
    random_state=42,
    min_samples_leaf=1,
    min_samples_split=2,
)
rf.fit(X_train_np, y_train)

val_pred = rf.predict(X_val_np)
rmse = mean_squared_error(y_val, val_pred, squared=False)
print(f"Validation RMSE: {rmse:.4f}")



## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2910063514.py in <cell line: 0>()
      8 from sklearn.ensemble import RandomForestRegressor
      9 
---> 10 X = train_data.drop(["key", "fare_amount"], axis=1)
     11 y = train_data["fare_amount"]
     12 X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.01, random_state=80)

NameError: name 'train_data' is not defined

## === cell 31
test_features = test_data.drop("key", axis=1).to_numpy(dtype=np.float32, copy=False)
pred_raw = rf.predict(test_features)

pred = np.round(np.clip(pred_raw, a_min=0, a_max=None), 2)



## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4121281728.py in <cell line: 0>()
----> 1 test_features = test_data.drop("key", axis=1).to_numpy(dtype=np.float32, copy=False)
      2 pred_raw = rf.predict(test_features)
      3 
      4 pred = np.round(np.clip(pred_raw, a_min=0, a_max=None), 2)
      5 

NameError: name 'test_data' is not defined

## === cell 32
pd.read_csv("/kaggle/input/new-york-city-taxi-prediction/sample_submission.csv").head()



## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/291782073.py in <cell line: 0>()
----> 1 pd.read_csv("/kaggle/input/new-york-city-taxi-prediction/sample_submission.csv").head()
      2 

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

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/new-york-city-taxi-prediction/sample_submission.csv'

## === cell 33
Submission = pd.DataFrame(data=pred, columns=["fare_amount"])
Submission["key"] = test_data["key"]
Submission = Submission[["key", "fare_amount"]]



## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1618083723.py in <cell line: 0>()
----> 1 Submission = pd.DataFrame(data=pred, columns=["fare_amount"])
      2 Submission["key"] = test_data["key"]
      3 Submission = Submission[["key", "fare_amount"]]
      4 

NameError: name 'pred' is not defined

## === cell 34
Submission.set_index("key", inplace=True)



## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/109248162.py in <cell line: 0>()
----> 1 Submission.set_index("key", inplace=True)
      2 

NameError: name 'Submission' is not defined

## === cell 35
Submission.to_csv("Submission.csv")

## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3260125347.py in <cell line: 0>()
----> 1 Submission.to_csv("Submission.csv")

NameError: name 'Submission' is not defined
