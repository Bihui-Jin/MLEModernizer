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

# 8. Previous improvement plan

- What this solution (achieved 936.92806) has done: 'I remove the deprecated `normalize` argument from `LinearRegression`, ensure the model variable persists for prediction, and adjust the submission writing step to output a proper CSV with the required columns and no index column. This resolves the runtime errors and creates a valid `submission.csv` file.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)


import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
td = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-prediction/train.csv", nrows=10_000_000
)
if len(td) > 2_000_000:
    td = td.sample(frac=2_000_000 / len(td), random_state=42).reset_index(drop=True)
td.head()  # td:train data # ted:test data



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1076238115.py in <cell line: 0>()
----> 1 td = pd.read_csv(
      2     "/kaggle/input/new-york-city-taxi-prediction/train.csv", nrows=10_000_000
      3 )
      4 if len(td) > 2_000_000:
      5     td = td.sample(frac=2_000_000 / len(td), random_state=42).reset_index(drop=True)

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
td.shape



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1021359999.py in <cell line: 0>()
----> 1 td.shape
      2 

NameError: name 'td' is not defined

## === cell 3
td.info()



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3753224938.py in <cell line: 0>()
----> 1 td.info()
      2 

NameError: name 'td' is not defined

## === cell 4
ted = pd.read_csv("/kaggle/input/new-york-city-taxi-prediction/test.csv")
ted.head()



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2787480929.py in <cell line: 0>()
----> 1 ted = pd.read_csv("/kaggle/input/new-york-city-taxi-prediction/test.csv")
      2 ted.head()
      3 

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
ted.info()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2089012891.py in <cell line: 0>()
----> 1 ted.info()
      2 

NameError: name 'ted' is not defined

## === cell 6
td.isna().sum()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2873308586.py in <cell line: 0>()
----> 1 td.isna().sum()
      2 

NameError: name 'td' is not defined

## === cell 7
td["Difference_longitude"] = np.abs(
    np.asarray(td["pickup_longitude"] - td["dropoff_longitude"])
)
td["Difference_latitude"] = np.abs(
    np.asarray(td["pickup_latitude"] - td["dropoff_latitude"])
)


ted["Difference_longitude"] = np.abs(
    np.asarray(ted["pickup_longitude"] - ted["dropoff_longitude"])
)
ted["Difference_latitude"] = np.abs(
    np.asarray(ted["pickup_latitude"] - ted["dropoff_latitude"])
)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1167896466.py in <cell line: 0>()
      1 td["Difference_longitude"] = np.abs(
----> 2     np.asarray(td["pickup_longitude"] - td["dropoff_longitude"])
      3 )
      4 td["Difference_latitude"] = np.abs(
      5     np.asarray(td["pickup_latitude"] - td["dropoff_latitude"])

NameError: name 'td' is not defined

## === cell 8
print(f"Before Dropping null values: {len(td)}")
td.dropna(inplace=True)
print(f"After Dropping null values: {len(td)}")



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1837556248.py in <cell line: 0>()
----> 1 print(f"Before Dropping null values: {len(td)}")
      2 td.dropna(inplace=True)
      3 print(f"After Dropping null values: {len(td)}")
      4 

NameError: name 'td' is not defined

## === cell 9
plot = td[:2000].plot.scatter("Difference_longitude", "Difference_latitude")



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/824600276.py in <cell line: 0>()
----> 1 plot = td[:2000].plot.scatter("Difference_longitude", "Difference_latitude")
      2 

NameError: name 'td' is not defined

## === cell 10
td = td[(td["Difference_longitude"] < 5.0) & (td["Difference_latitude"] < 5.0)]



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3159731776.py in <cell line: 0>()
----> 1 td = td[(td["Difference_longitude"] < 5.0) & (td["Difference_latitude"] < 5.0)]
      2 

NameError: name 'td' is not defined

## === cell 11
td["pickup_datetime"] = pd.to_datetime(td["pickup_datetime"])
td["pickuptime"] = td["pickup_datetime"].dt.hour * 100 + td["pickup_datetime"].dt.minute
td["Weekday"] = td["pickup_datetime"].dt.weekday.map(
    {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday",
    }
)

ted["pickup_datetime"] = pd.to_datetime(ted["pickup_datetime"])
ted["pickuptime"] = (
    ted["pickup_datetime"].dt.hour * 100 + ted["pickup_datetime"].dt.minute
)
ted["Weekday"] = ted["pickup_datetime"].dt.weekday.map(
    {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday",
    }
)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/644990272.py in <cell line: 0>()
      1 # Vectorized datetime processing: extract time as HHMM integer and weekday as string
      2 # Train set
----> 3 td["pickup_datetime"] = pd.to_datetime(td["pickup_datetime"])
      4 td["pickuptime"] = td["pickup_datetime"].dt.hour * 100 + td["pickup_datetime"].dt.minute
      5 td["Weekday"] = td["pickup_datetime"].dt.weekday.map(

NameError: name 'td' is not defined

## === cell 12
td.head()



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1733393665.py in <cell line: 0>()
----> 1 td.head()
      2 

NameError: name 'td' is not defined

## === cell 13
td.head()



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2766466292.py in <cell line: 0>()
      1 # No additional loops needed; weekday strings already created in cell 11
----> 2 td.head()
      3 

NameError: name 'td' is not defined

## === cell 14
ted.head()



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2445101068.py in <cell line: 0>()
----> 1 ted.head()
      2 

NameError: name 'ted' is not defined

## === cell 15
td.drop("pickup_datetime", inplace=True, axis=1)
ted.drop("pickup_datetime", inplace=True, axis=1)

th = pd.get_dummies(td["Weekday"])
teh = pd.get_dummies(ted["Weekday"])
td = pd.concat([td, th], axis=1)
ted = pd.concat([ted, teh], axis=1)

td.drop("Weekday", axis=1, inplace=True)
ted.drop("Weekday", axis=1, inplace=True)

td.head()



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1953056211.py in <cell line: 0>()
      1 # Drop original datetime column
----> 2 td.drop("pickup_datetime", inplace=True, axis=1)
      3 ted.drop("pickup_datetime", inplace=True, axis=1)
      4 
      5 # One‑hot encode weekday strings (preserves original column names)

NameError: name 'td' is not defined

## === cell 16
R = 6373.0
lat1 = np.asarray(np.radians(td["pickup_latitude"]))
lon1 = np.asarray(np.radians(td["pickup_longitude"]))
lat2 = np.asarray(np.radians(td["dropoff_latitude"]))
lon2 = np.asarray(np.radians(td["dropoff_longitude"]))

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
td["Distance"] = np.asarray(distance) * 0.621

lat1 = np.asarray(np.radians(ted["pickup_latitude"]))
lon1 = np.asarray(np.radians(ted["pickup_longitude"]))
lat2 = np.asarray(np.radians(ted["dropoff_latitude"]))
lon2 = np.asarray(np.radians(ted["dropoff_longitude"]))

dlon = lon2 - lon1
dlat = lat2 - lat1

a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
ted["Distance"] = np.asarray(distance) * 0.621



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/213657164.py in <cell line: 0>()
      1 R = 6373.0
----> 2 lat1 = np.asarray(np.radians(td["pickup_latitude"]))
      3 lon1 = np.asarray(np.radians(td["pickup_longitude"]))
      4 lat2 = np.asarray(np.radians(td["dropoff_latitude"]))
      5 lon2 = np.asarray(np.radians(td["dropoff_longitude"]))

NameError: name 'td' is not defined

## === cell 17
R = 6373.0
lat1 = np.asarray(np.radians(td["pickup_latitude"]))
lon1 = np.asarray(np.radians(td["pickup_longitude"]))
lat2 = np.asarray(np.radians(td["dropoff_latitude"]))
lon2 = np.asarray(np.radians(td["dropoff_longitude"]))

lat3 = np.zeros(len(td)) + np.radians(40.6413111)
lon3 = np.zeros(len(td)) + np.radians(-73.7781391)
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
td["Pickup_Distance_airport"] = np.asarray(distance1) * 0.621

a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
td["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621

lat1 = np.asarray(np.radians(ted["pickup_latitude"]))
lon1 = np.asarray(np.radians(ted["pickup_longitude"]))
lat2 = np.asarray(np.radians(ted["dropoff_latitude"]))
lon2 = np.asarray(np.radians(ted["dropoff_longitude"]))

lat3 = np.zeros(len(ted)) + np.radians(40.6413111)
lon3 = np.zeros(len(ted)) + np.radians(-73.7781391)
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
ted["Pickup_Distance_airport"] = np.asarray(distance1) * 0.621

a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
ted["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621

td["Distance"] = np.round(td["Distance"], 2)
td["Pickup_Distance_airport"] = np.round(td["Pickup_Distance_airport"], 2)
td["Dropoff_Distance_airport"] = np.round(td["Dropoff_Distance_airport"], 2)
ted["Distance"] = np.round(ted["Distance"], 2)
ted["Pickup_Distance_airport"] = np.round(ted["Pickup_Distance_airport"], 2)
ted["Dropoff_Distance_airport"] = np.round(ted["Dropoff_Distance_airport"], 2)

td.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)
ted.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)

td["Difference_longitude"] = np.abs(
    td["Difference_longitude"] - np.mean(td["Difference_longitude"])
)
td["Difference_longitude"] = td["Difference_longitude"] / np.var(
    td["Difference_longitude"]
)

td["Difference_latitude"] = np.abs(
    td["Difference_latitude"] - np.mean(td["Difference_latitude"])
)
td["Difference_latitude"] = td["Difference_latitude"] / np.var(
    td["Difference_latitude"]
)

ted["Difference_longitude"] = np.abs(
    ted["Difference_longitude"] - np.mean(ted["Difference_longitude"])
)
ted["Difference_longitude"] = ted["Difference_longitude"] / np.var(
    ted["Difference_longitude"]
)

ted["Difference_latitude"] = np.abs(
    ted["Difference_latitude"] - np.mean(ted["Difference_latitude"])
)
ted["Difference_latitude"] = ted["Difference_latitude"] / np.var(
    ted["Difference_latitude"]
)

td.shape



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1505363950.py in <cell line: 0>()
      1 R = 6373.0
----> 2 lat1 = np.asarray(np.radians(td["pickup_latitude"]))
      3 lon1 = np.asarray(np.radians(td["pickup_longitude"]))
      4 lat2 = np.asarray(np.radians(td["dropoff_latitude"]))
      5 lon2 = np.asarray(np.radians(td["dropoff_longitude"]))

NameError: name 'td' is not defined

## === cell 18
ted.shape



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2087936640.py in <cell line: 0>()
----> 1 ted.shape
      2 

NameError: name 'ted' is not defined

## === cell 19
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

X = td.drop(["key", "fare_amount"], axis=1)
y = td["fare_amount"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.01, random_state=80)

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    n_jobs=5,
    random_state=42,
    min_samples_leaf=1,
)
rf.fit(X_train, y_train)

val_pred = rf.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f"Validation RMSE: {rmse:.4f}")



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/7673394.py in <cell line: 0>()
      3 from sklearn.metrics import mean_squared_error
      4 
----> 5 X = td.drop(["key", "fare_amount"], axis=1)
      6 y = td["fare_amount"]
      7 X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.01, random_state=80)

NameError: name 'td' is not defined

## === cell 20
test_features = ted.drop("key", axis=1)
pred = np.round(rf.predict(test_features), 2)

print(pred[:5])



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3695864318.py in <cell line: 0>()
----> 1 test_features = ted.drop("key", axis=1)
      2 pred = np.round(rf.predict(test_features), 2)
      3 
      4 print(pred[:5])
      5 

NameError: name 'ted' is not defined

## === cell 21
Submission = pd.DataFrame({"key": ted["key"], "fare_amount": pred})
Submission.to_csv("submission.csv", index=False)

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3739353413.py in <cell line: 0>()
----> 1 Submission = pd.DataFrame({"key": ted["key"], "fare_amount": pred})
      2 Submission.to_csv("submission.csv", index=False)

NameError: name 'ted' is not defined
