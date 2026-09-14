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

5.68932

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 936.92806) has done: 'I remove the deprecated `normalize=True` argument from `LinearRegression` so the model can be instantiated and trained, which also restores the `lr` object needed for later cells. The rest of the pipeline stays unchanged; the script now run end‑to‑end and write a proper `Submission.csv` with the required columns.'
- What this solution (achieved 936.94064) has done: 'I add a standard‑scaler to normalize the numeric features before fitting the same LinearRegression model, and then clip any negative predictions to 0 before rounding. This keeps the core LinearRegression logic unchanged while improving numerical stability, which should lower the RMSE toward the target.'
- What this solution (achieved 752.76915) has done: 'I fix the inconsistent feature scaling that caused the model to see mismatched distributions between train and test (the original code scaled the test set using its own mean/variance). The fix computes the mean and variance from the training data once and applies those exact statistics to both the training and test “Difference_” features. I also remove the unnecessary rounding of predictions before writing the submission, which slightly improves RMSE while keeping the LinearRegression core unchanged.'
- What this solution (achieved inf) has done: 'I keep the overall feature engineering and LinearRegression model but train it on the log‑transformed fare amount (log1p) and then exponentiate the predictions back to the original scale. This usually stabilises the skewed target distribution and reduces RMSE, moving the score much closer to the target. I also clip extreme fare values in the training data to a sensible range (0‑200 $) to avoid outlier distortion. The rest of the pipeline, including scaling and feature handling, remains unchanged.'
- What this solution (achieved 751.45992) has done: 'I remove the unnecessary log‑transform of the target variable, fitting the LinearRegression directly on the fare amount. This avoids the exponentiation step that can produce infinities and gives a finite validation RMSE, moving the score toward the target while keeping the core model and feature engineering unchanged.'
- What this solution (achieved inf) has done: 'I keep the overall pipeline and LinearRegression model unchanged, but apply a log‑transform to the target variable before fitting and reverse it after prediction. This stabilises the heavy‑tailed fare distribution, reduces extreme errors, and moves the RMSE much closer to the target without altering any core logic.'
- What this solution (achieved 751.45992) has done: 'I remove the log‑transform of the target variable, training the LinearRegression directly on the fare amount. This avoids the overflow that produced an infinite RMSE and yields a finite validation score. The prediction steps are also simplified to use the raw model output (clipped at 0) instead of exponentiating. All other feature‑engineering steps and the core LinearRegression model remain unchanged.'
- What this solution (achieved 784.82681) has done: 'I replace the plain LinearRegression with a Ridge regression (still a linear model) and remove the log‑transform of the target so the model directly predicts fares. I also simplify the submission write‑out by not setting the index, ensuring a proper CSV with the required columns. These minimal adjustments keep the overall feature engineering unchanged while improving calibration and should move the RMSE closer to the target.'
- What this solution (achieved 787.84792) has done: 'I remove the log‑transform of the target so the Ridge model predicts the fare amount directly, and I lower the regularisation strength (α = 0.1) which usually reduces bias for this linear model. The validation RMSE is recomputed on the raw values, and the test‑time predictions are clipped at 0 without any exponentiation. These minimal adjustments keep the original feature engineering and model type while moving the score toward the target 5.68932.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
train_data = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-prediction/train.csv", nrows=10_000_000
)
train_data.head()



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1456384226.py in <cell line: 0>()
----> 1 train_data = pd.read_csv(
      2     "/kaggle/input/new-york-city-taxi-prediction/train.csv", nrows=10_000_000
      3 )
      4 train_data.head()
      5 

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
test_data = pd.read_csv("/kaggle/input/new-york-city-taxi-prediction/test.csv")
test_data.head()



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2952326741.py in <cell line: 0>()
----> 1 test_data = pd.read_csv("/kaggle/input/new-york-city-taxi-prediction/test.csv")
      2 test_data.head()
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
test_data.info()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1702245187.py in <cell line: 0>()
----> 1 test_data.info()
      2 

NameError: name 'test_data' is not defined

## === cell 6
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



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2029277579.py in <cell line: 0>()
      1 train_data["Difference_longitude"] = np.abs(
----> 2     np.asarray(train_data["pickup_longitude"] - train_data["dropoff_longitude"])
      3 )
      4 train_data["Difference_latitude"] = np.abs(
      5     np.asarray(train_data["pickup_latitude"] - train_data["dropoff_latitude"])

NameError: name 'train_data' is not defined

## === cell 7
print(f"Before Dropping null values: {len(train_data)}")
train_data.dropna(inplace=True)
print(f"After Dropping null values: {len(train_data)}")



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4109314509.py in <cell line: 0>()
----> 1 print(f"Before Dropping null values: {len(train_data)}")
      2 train_data.dropna(inplace=True)
      3 print(f"After Dropping null values: {len(train_data)}")
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
/tmp/ipykernel_11/297526157.py in <cell line: 0>()
----> 1 train_data = train_data[
      2     (train_data["Difference_longitude"] < 5.0)
      3     & (train_data["Difference_latitude"] < 5.0)
      4 ]
      5 

NameError: name 'train_data' is not defined

## === cell 9
ls1 = list(train_data["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][11:-7:]
train_data["pickuptime"] = ls1

ls1 = list(test_data["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][11:-7:]
test_data["pickuptime"] = ls1



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3679443842.py in <cell line: 0>()
----> 1 ls1 = list(train_data["pickup_datetime"])
      2 for i in range(len(ls1)):
      3     ls1[i] = ls1[i][11:-7:]
      4 train_data["pickuptime"] = ls1
      5 

NameError: name 'train_data' is not defined

## === cell 10
train_data.head()



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3467491927.py in <cell line: 0>()
----> 1 train_data.head()
      2 

NameError: name 'train_data' is not defined

## === cell 11
ls1 = list(train_data["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][:-4:]
    ls1[i] = pd.Timestamp(ls1[i])
    ls1[i] = ls1[i].weekday()
train_data["Weekday"] = ls1

ls1 = list(test_data["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][:-4:]
    ls1[i] = pd.Timestamp(ls1[i])
    ls1[i] = ls1[i].weekday()
test_data["Weekday"] = ls1
test_data.head()



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/101897464.py in <cell line: 0>()
----> 1 ls1 = list(train_data["pickup_datetime"])
      2 for i in range(len(ls1)):
      3     ls1[i] = ls1[i][:-4:]
      4     ls1[i] = pd.Timestamp(ls1[i])
      5     ls1[i] = ls1[i].weekday()

NameError: name 'train_data' is not defined

## === cell 12
train_data.drop("pickup_datetime", inplace=True, axis=1)
test_data.drop("pickup_datetime", inplace=True, axis=1)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/764264539.py in <cell line: 0>()
----> 1 train_data.drop("pickup_datetime", inplace=True, axis=1)
      2 test_data.drop("pickup_datetime", inplace=True, axis=1)
      3 

NameError: name 'train_data' is not defined

## === cell 13
train_data["Weekday"].replace(
    to_replace=[i for i in range(0, 7)],
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
    to_replace=[i for i in range(0, 7)],
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



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2022498230.py in <cell line: 0>()
----> 1 train_data["Weekday"].replace(
      2     to_replace=[i for i in range(0, 7)],
      3     value=[
      4         "Monday",
      5         "Tuesday",

NameError: name 'train_data' is not defined

## === cell 14
train_one_hot = pd.get_dummies(train_data["Weekday"])
test_one_hot = pd.get_dummies(test_data["Weekday"])
train_data = pd.concat([train_data, train_one_hot], axis=1)
test_data = pd.concat([test_data, test_one_hot], axis=1)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3579435750.py in <cell line: 0>()
----> 1 train_one_hot = pd.get_dummies(train_data["Weekday"])
      2 test_one_hot = pd.get_dummies(test_data["Weekday"])
      3 train_data = pd.concat([train_data, train_one_hot], axis=1)
      4 test_data = pd.concat([test_data, test_one_hot], axis=1)
      5 

NameError: name 'train_data' is not defined

## === cell 15
train_data.drop("Weekday", axis=1, inplace=True)
test_data.drop("Weekday", axis=1, inplace=True)




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/927199483.py in <cell line: 0>()
----> 1 train_data.drop("Weekday", axis=1, inplace=True)
      2 test_data.drop("Weekday", axis=1, inplace=True)
      3 
      4 

NameError: name 'train_data' is not defined

## === cell 16
def add_time_features(df):
    hours = []
    minute_fracs = []
    for t in df["pickuptime"]:
        h, m = map(int, t.split(":"))
        hours.append(h)
        minute_fracs.append(m / 60.0)
    df["hour"] = hours
    df["minute_frac"] = minute_fracs
    df.drop("pickuptime", axis=1, inplace=True)


add_time_features(train_data)
add_time_features(test_data)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/417311592.py in <cell line: 0>()
     11 
     12 
---> 13 add_time_features(train_data)
     14 add_time_features(test_data)
     15 

NameError: name 'train_data' is not defined

## === cell 17
train_data.head()



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3467491927.py in <cell line: 0>()
----> 1 train_data.head()
      2 

NameError: name 'train_data' is not defined

## === cell 18
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



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2277544786.py in <cell line: 0>()
      1 R = 6373.0
----> 2 lat1 = np.asarray(np.radians(train_data["pickup_latitude"]))
      3 lon1 = np.asarray(np.radians(train_data["pickup_longitude"]))
      4 lat2 = np.asarray(np.radians(train_data["dropoff_latitude"]))
      5 lon2 = np.asarray(np.radians(train_data["dropoff_longitude"]))

NameError: name 'train_data' is not defined

## === cell 19
R = 6373.0
lat1 = np.asarray(np.radians(train_data["pickup_latitude"]))
lon1 = np.asarray(np.radians(train_data["pickup_longitude"]))
lat2 = np.asarray(np.radians(train_data["dropoff_latitude"]))
lon2 = np.asarray(np.radians(train_data["dropoff_longitude"]))

lat3 = np.zeros(len(train_data)) + np.radians(40.6413111)
lon3 = np.zeros(len(train_data)) + np.radians(-73.7781391)
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
train_data["Pickup_Distance_airport"] = np.asarray(distance1) * 0.621

a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2

train_data["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621


lat1 = np.asarray(np.radians(test_data["pickup_latitude"]))
lon1 = np.asarray(np.radians(test_data["pickup_longitude"]))
lat2 = np.asarray(np.radians(test_data["dropoff_latitude"]))
lon2 = np.asarray(np.radians(test_data["dropoff_longitude"]))

lat3 = np.zeros(len(test_data)) + np.radians(40.6413111)
lon3 = np.zeros(len(test_data)) + np.radians(-73.7781391)
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
test_data["Pickup_Distance_airport"] = np.asarray(distance1) * 0.621

a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2

test_data["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4168520821.py in <cell line: 0>()
      1 R = 6373.0
----> 2 lat1 = np.asarray(np.radians(train_data["pickup_latitude"]))
      3 lon1 = np.asarray(np.radians(train_data["pickup_longitude"]))
      4 lat2 = np.asarray(np.radians(train_data["dropoff_latitude"]))
      5 lon2 = np.asarray(np.radians(train_data["dropoff_longitude"]))

NameError: name 'train_data' is not defined

## === cell 20
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



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/257530458.py in <cell line: 0>()
----> 1 train_data["Distance"] = np.round(train_data["Distance"], 2)
      2 train_data["Pickup_Distance_airport"] = np.round(
      3     train_data["Pickup_Distance_airport"], 2
      4 )
      5 train_data["Dropoff_Distance_airport"] = np.round(

NameError: name 'train_data' is not defined

## === cell 21
train_data.shape



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4031933074.py in <cell line: 0>()
----> 1 train_data.shape
      2 

NameError: name 'train_data' is not defined

## === cell 22
test_data.shape



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1276008450.py in <cell line: 0>()
----> 1 test_data.shape
      2 

NameError: name 'test_data' is not defined

## === cell 23
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

train_data["fare_amount"] = train_data["fare_amount"].clip(lower=0, upper=200)

X = train_data.drop(["key", "fare_amount"], axis=1)
y = train_data["fare_amount"]  # <-- train on raw fare

feature_cols = X.columns

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.01, random_state=80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

lr = Ridge(alpha=0.0, fit_intercept=True)
lr.fit(X_train_scaled, y_train)

val_pred = lr.predict(X_val_scaled)
val_pred = np.clip(val_pred, 0, None)

val_rmse = mean_squared_error(y_val, val_pred, squared=False)  # RMSE on raw fare
print("Validation RMSE:", val_rmse)



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/659608791.py in <cell line: 0>()
      5 
      6 # Clip extreme fares to avoid outliers affecting the linear model
----> 7 train_data["fare_amount"] = train_data["fare_amount"].clip(lower=0, upper=200)
      8 
      9 X = train_data.drop(["key", "fare_amount"], axis=1)

NameError: name 'train_data' is not defined

## === cell 24
test_X = test_data.drop("key", axis=1)

missing_cols = set(feature_cols) - set(test_X.columns)
for col in missing_cols:
    test_X[col] = 0
test_X = test_X[feature_cols]  # enforce same order

test_features = scaler.transform(test_X)

test_pred = lr.predict(test_features)
pred = np.clip(test_pred, 0, None)



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2467803434.py in <cell line: 0>()
----> 1 test_X = test_data.drop("key", axis=1)
      2 
      3 missing_cols = set(feature_cols) - set(test_X.columns)
      4 for col in missing_cols:
      5     test_X[col] = 0

NameError: name 'test_data' is not defined

## === cell 25
pd.read_csv("/kaggle/input/new-york-city-taxi-prediction/sample_submission.csv").head()



## --- ERROR in cell 25, traceback:
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

## === cell 26
Submission = pd.DataFrame({"key": test_data["key"], "fare_amount": pred})



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1712039930.py in <cell line: 0>()
----> 1 Submission = pd.DataFrame({"key": test_data["key"], "fare_amount": pred})
      2 

NameError: name 'test_data' is not defined

## === cell 27
Submission.to_csv("Submission.csv", index=False)

## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3446740269.py in <cell line: 0>()
----> 1 Submission.to_csv("Submission.csv", index=False)

NameError: name 'Submission' is not defined
