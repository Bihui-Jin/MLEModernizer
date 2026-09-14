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

5.68914

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 936.92806) has done: 'I remove the obsolete `normalize=True` argument from the `LinearRegression` constructor (it caused a TypeError) so the model can be trained, which also restores the downstream variables (`lr`, `pred`, `Submission`) needed for creating a valid CSV submission.'
- What this solution (achieved 936.92806) has done: 'I replace the use of `LinearRegression.score` (R²) with a proper RMSE calculation, because the competition evaluates RMSE. Computing the correct metric move the reported score from the huge 936 value to a realistic value near the target (≈5‑6). This change only affects the evaluation line and adds the necessary import.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error



## === cell 1
train_path = "/kaggle/input/new-york-city-taxi-prediction/train.csv"
test_path = "/kaggle/input/new-york-city-taxi-prediction/test.csv"
train_data = pd.read_csv(train_path, nrows=10_000_000)
test_data = pd.read_csv(test_path)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/4057271714.py in <cell line: 0>()
      2 train_path = "/kaggle/input/new-york-city-taxi-prediction/train.csv"
      3 test_path = "/kaggle/input/new-york-city-taxi-prediction/test.csv"
----> 4 train_data = pd.read_csv(train_path, nrows=10_000_000)
      5 test_data = pd.read_csv(test_path)
      6 

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
train_data.dropna(inplace=True)
test_data.dropna(inplace=True)  # just in case




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2305477479.py in <cell line: 0>()
      1 # Basic cleaning
----> 2 train_data.dropna(inplace=True)
      3 test_data.dropna(inplace=True)  # just in case
      4 
      5 

NameError: name 'train_data' is not defined

## === cell 3
def add_features(df):
    df["pickup_dt"] = pd.to_datetime(df["pickup_datetime"])
    df["pickuptime"] = df["pickup_dt"].dt.hour * 100 + df["pickup_dt"].dt.minute
    df["Weekday"] = df["pickup_dt"].dt.weekday
    R = 6373.0  # km
    lat1 = np.radians(df["pickup_latitude"])
    lon1 = np.radians(df["pickup_longitude"])
    lat2 = np.radians(df["dropoff_latitude"])
    lon2 = np.radians(df["dropoff_longitude"])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance_km = R * c
    df["Distance"] = distance_km * 0.621371  # miles

    lat_air = np.radians(40.6413111)
    lon_air = np.radians(-73.7781391)

    dlon_pa = lon_air - lon1
    dlat_pa = lat_air - lat1
    a_pa = (
        np.sin(dlat_pa / 2) ** 2
        + np.cos(lat1) * np.cos(lat_air) * np.sin(dlon_pa / 2) ** 2
    )
    c_pa = 2 * np.arctan2(np.sqrt(a_pa), np.sqrt(1 - a_pa))
    df["Pickup_Distance_airport"] = (R * c_pa) * 0.621371

    dlon_da = lon_air - lon2
    dlat_da = lat_air - lat2
    a_da = (
        np.sin(dlat_da / 2) ** 2
        + np.cos(lat2) * np.cos(lat_air) * np.sin(dlon_da / 2) ** 2
    )
    c_da = 2 * np.arctan2(np.sqrt(a_da), np.sqrt(1 - a_da))
    df["Dropoff_Distance_airport"] = (R * c_da) * 0.621371

    weekday_dummies = pd.get_dummies(df["Weekday"], prefix="wd")
    df = pd.concat([df, weekday_dummies], axis=1)

    drop_cols = [
        "pickup_datetime",
        "pickup_dt",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "Weekday",
    ]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")
    return df


train_data = add_features(train_data)
test_data = add_features(test_data)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/138253714.py in <cell line: 0>()
     64 
     65 
---> 66 train_data = add_features(train_data)
     67 test_data = add_features(test_data)
     68 

NameError: name 'train_data' is not defined

## === cell 4
feature_cols = [c for c in train_data.columns if c not in ["key", "fare_amount"]]
test_data = test_data.reindex(columns=feature_cols, fill_value=0)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3529364753.py in <cell line: 0>()
      1 # Align train and test feature columns (exclude target and key)
----> 2 feature_cols = [c for c in train_data.columns if c not in ["key", "fare_amount"]]
      3 # Ensure test has exactly the same columns (missing ones get filled with 0)
      4 test_data = test_data.reindex(columns=feature_cols, fill_value=0)
      5 

NameError: name 'train_data' is not defined

## === cell 5
X = train_data[feature_cols]
y = train_data["fare_amount"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.01, random_state=80)

gbr = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    random_state=42,
)
gbr.fit(X_train, y_train)

val_pred = gbr.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f"Validation RMSE: {rmse:.5f}")



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/880593364.py in <cell line: 0>()
----> 1 X = train_data[feature_cols]
      2 y = train_data["fare_amount"]
      3 X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.01, random_state=80)
      4 
      5 gbr = GradientBoostingRegressor(

NameError: name 'train_data' is not defined

## === cell 6
test_pred = np.round(gbr.predict(test_data[feature_cols]), 2)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2028558766.py in <cell line: 0>()
      1 # Predict on the official test set
----> 2 test_pred = np.round(gbr.predict(test_data[feature_cols]), 2)
      3 

NameError: name 'gbr' is not defined

## === cell 7
submission = pd.DataFrame({"key": test_data["key"], "fare_amount": test_pred})



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3597148014.py in <cell line: 0>()
      1 # Build submission DataFrame
----> 2 submission = pd.DataFrame({"key": test_data["key"], "fare_amount": test_pred})
      3 

NameError: name 'test_data' is not defined

## === cell 8
submission.to_csv("Submission.csv", index=False)
print("Submission file 'Submission.csv' written.")

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2953070005.py in <cell line: 0>()
      1 # Write submission file
----> 2 submission.to_csv("Submission.csv", index=False)
      3 print("Submission file 'Submission.csv' written.")

NameError: name 'submission' is not defined
