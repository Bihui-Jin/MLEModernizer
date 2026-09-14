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

5.52625

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ["OMP_NUM_THREADS"] = "8"  # limit threads for Intel‑optimised ops

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearnex import patch_sklearn

patch_sklearn()

print(os.listdir("../input"))




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
    "fare_amount": np.float32,
    "pickup_datetime": "object",
    "pickup_longitude": np.float32,
    "pickup_latitude": np.float32,
    "dropoff_longitude": np.float32,
    "dropoff_latitude": np.float32,
    "passenger_count": np.int8,
}
train_df = pd.read_csv(
    "../input/train.csv",
    usecols=usecols,
    dtype=dtypes,
    nrows=10_000_000,
    low_memory=False,
)

test_df = pd.read_csv(
    "../input/test.csv",
    usecols=usecols[:-1],  # test lacks fare_amount
    dtype={k: v for k, v in dtypes.items() if k != "fare_amount"},
    low_memory=False,
)

coord_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/4283034584.py in <cell line: 0>()
     27 )
     28 
---> 29 test_df = pd.read_csv(
     30     "../input/test.csv",
     31     usecols=usecols[:-1],  # test lacks fare_amount

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
   1896 
   1897         try:
-> 1898             return mapping[engine](f, **self.options)
   1899         except Exception:
   1900             if self.handles is not None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/c_parser_wrapper.py in __init__(self, src, **kwds)
    138                 self.orig_names
    139             ):
--> 140                 self._validate_usecols_names(usecols, self.orig_names)
    141 
    142             # error: Cannot determine type of 'names'

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/base_parser.py in _validate_usecols_names(self, usecols, names)
    977         missing = [c for c in usecols if c not in names]
    978         if len(missing) > 0:
--> 979             raise ValueError(
    980                 f"Usecols do not match columns, columns expected but not found: "
    981                 f"{missing}"

ValueError: Usecols do not match columns, columns expected but not found: ['fare_amount']

## === cell 2
def add_travel_vector_features(df):
    df["abs_diff_longitude"] = (df["dropoff_longitude"] - df["pickup_longitude"]).abs()
    df["abs_diff_latitude"] = (df["dropoff_latitude"] - df["pickup_latitude"]).abs()


add_travel_vector_features(train_df)
add_travel_vector_features(test_df)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3905516419.py in <cell line: 0>()
      5 
      6 add_travel_vector_features(train_df)
----> 7 add_travel_vector_features(test_df)
      8 
      9 

NameError: name 'test_df' is not defined

## === cell 3
print(train_df.isnull().sum())
print(test_df.isnull().sum())




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3649360569.py in <cell line: 0>()
      1 print(train_df.isnull().sum())
----> 2 print(test_df.isnull().sum())
      3 
      4 

NameError: name 'test_df' is not defined

## === cell 4
train_df = train_df.dropna(how="any", axis="rows")
print(f"Rows after dropping NaNs: {len(train_df)}")




## === cell 5
mask = (train_df["abs_diff_longitude"] < 5.0) & (train_df["abs_diff_latitude"] < 5.0)
train_df = train_df[mask]
test_df = test_df[
    (test_df["abs_diff_longitude"] < 5.0) & (test_df["abs_diff_latitude"] < 5.0)
]
print(f"Rows after distance filter – train: {len(train_df)}, test: {len(test_df)}")




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2063459210.py in <cell line: 0>()
      1 mask = (train_df["abs_diff_longitude"] < 5.0) & (train_df["abs_diff_latitude"] < 5.0)
      2 train_df = train_df[mask]
----> 3 test_df = test_df[
      4     (test_df["abs_diff_longitude"] < 5.0) & (test_df["abs_diff_latitude"] < 5.0)
      5 ]

NameError: name 'test_df' is not defined

## === cell 6
train_df["pickup_datetime"] = pd.to_datetime(train_df["pickup_datetime"])
test_df["pickup_datetime"] = pd.to_datetime(test_df["pickup_datetime"])

train_df["pickup_time"] = (
    train_df["pickup_datetime"].dt.hour * 100 + train_df["pickup_datetime"].dt.minute
)
test_df["pickup_time"] = (
    test_df["pickup_datetime"].dt.hour * 100 + test_df["pickup_datetime"].dt.minute
)

train_df["Weekday"] = train_df["pickup_datetime"].dt.weekday
test_df["Weekday"] = test_df["pickup_datetime"].dt.weekday




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/337673662.py in <cell line: 0>()
      1 train_df["pickup_datetime"] = pd.to_datetime(train_df["pickup_datetime"])
----> 2 test_df["pickup_datetime"] = pd.to_datetime(test_df["pickup_datetime"])
      3 
      4 train_df["pickup_time"] = (
      5     train_df["pickup_datetime"].dt.hour * 100 + train_df["pickup_datetime"].dt.minute

NameError: name 'test_df' is not defined

## === cell 7
train_df.drop("pickup_datetime", inplace=True, axis=1)
test_df.drop("pickup_datetime", inplace=True, axis=1)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/953242425.py in <cell line: 0>()
      1 train_df.drop("pickup_datetime", inplace=True, axis=1)
----> 2 test_df.drop("pickup_datetime", inplace=True, axis=1)
      3 
      4 

NameError: name 'test_df' is not defined

## === cell 8
train_one_hot = pd.get_dummies(train_df["Weekday"], prefix="Weekday")
test_one_hot = pd.get_dummies(test_df["Weekday"], prefix="Weekday")
train_df = pd.concat([train_df, train_one_hot], axis=1)
test_df = pd.concat([test_df, test_one_hot], axis=1)
train_df.drop("Weekday", axis=1, inplace=True)
test_df.drop("Weekday", axis=1, inplace=True)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'Weekday'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1659838355.py in <cell line: 0>()
----> 1 train_one_hot = pd.get_dummies(train_df["Weekday"], prefix="Weekday")
      2 test_one_hot = pd.get_dummies(test_df["Weekday"], prefix="Weekday")
      3 train_df = pd.concat([train_df, train_one_hot], axis=1)
      4 test_df = pd.concat([test_df, test_one_hot], axis=1)
      5 train_df.drop("Weekday", axis=1, inplace=True)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'Weekday'

## === cell 9
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6373.0
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return (R * c * 0.621).astype(np.float32)


lat1 = np.radians(train_df["pickup_latitude"].values)
lon1 = np.radians(train_df["pickup_longitude"].values)
lat2 = np.radians(train_df["dropoff_latitude"].values)
lon2 = np.radians(train_df["dropoff_longitude"].values)
train_df["Distance"] = haversine_distance(lat1, lon1, lat2, lon2)

lat1 = np.radians(test_df["pickup_latitude"].values)
lon1 = np.radians(test_df["pickup_longitude"].values)
lat2 = np.radians(test_df["dropoff_latitude"].values)
lon2 = np.radians(test_df["dropoff_longitude"].values)
test_df["Distance"] = haversine_distance(lat1, lon1, lat2, lon2)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3635575680.py in <cell line: 0>()
     14 train_df["Distance"] = haversine_distance(lat1, lon1, lat2, lon2)
     15 
---> 16 lat1 = np.radians(test_df["pickup_latitude"].values)
     17 lon1 = np.radians(test_df["pickup_longitude"].values)
     18 lat2 = np.radians(test_df["dropoff_latitude"].values)

NameError: name 'test_df' is not defined

## === cell 10
lat_air = np.radians(40.6413).astype(np.float32)
lon_air = np.radians(-73.7781).astype(np.float32)


def airport_distance(lat_arr, lon_arr):
    dlon = lon_air - lon_arr
    dlat = lat_air - lat_arr
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat_arr) * np.cos(lat_air) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return (6373.0 * c * 0.621).astype(np.float32)


train_df["Pickup_Distance_airport"] = airport_distance(
    np.radians(train_df["pickup_latitude"].values),
    np.radians(train_df["pickup_longitude"].values),
)
train_df["Dropoff_Distance_airport"] = airport_distance(
    np.radians(train_df["dropoff_latitude"].values),
    np.radians(train_df["dropoff_longitude"].values),
)

test_df["Pickup_Distance_airport"] = airport_distance(
    np.radians(test_df["pickup_latitude"].values),
    np.radians(test_df["pickup_longitude"].values),
)
test_df["Dropoff_Distance_airport"] = airport_distance(
    np.radians(test_df["dropoff_latitude"].values),
    np.radians(test_df["dropoff_longitude"].values),
)




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1035322671.py in <cell line: 0>()
     24 
     25 test_df["Pickup_Distance_airport"] = airport_distance(
---> 26     np.radians(test_df["pickup_latitude"].values),
     27     np.radians(test_df["pickup_longitude"].values),
     28 )

NameError: name 'test_df' is not defined

## === cell 11
round_cols = ["Distance", "Pickup_Distance_airport", "Dropoff_Distance_airport"]
train_df[round_cols] = train_df[round_cols].round(2)
test_df[round_cols] = test_df[round_cols].round(2)




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3916015386.py in <cell line: 0>()
      1 round_cols = ["Distance", "Pickup_Distance_airport", "Dropoff_Distance_airport"]
      2 train_df[round_cols] = train_df[round_cols].round(2)
----> 3 test_df[round_cols] = test_df[round_cols].round(2)
      4 
      5 

NameError: name 'test_df' is not defined

## === cell 12
train_df.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)
test_df.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/801911681.py in <cell line: 0>()
      4     inplace=True,
      5 )
----> 6 test_df.drop(
      7     ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
      8     axis=1,

NameError: name 'test_df' is not defined

## === cell 13
train_df["abs_diff_longitude"] = (
    np.abs(train_df["abs_diff_longitude"]) - train_df["abs_diff_longitude"].mean()
) / train_df["abs_diff_longitude"].var()
train_df["abs_diff_latitude"] = (
    np.abs(train_df["abs_diff_latitude"]) - train_df["abs_diff_latitude"].mean()
) / train_df["abs_diff_latitude"].var()




## === cell 14
test_df["abs_diff_longitude"] = (
    np.abs(test_df["abs_diff_longitude"]) - test_df["abs_diff_longitude"].mean()
) / test_df["abs_diff_longitude"].var()
test_df["abs_diff_latitude"] = (
    np.abs(test_df["abs_diff_latitude"]) - test_df["abs_diff_latitude"].mean()
) / test_df["abs_diff_latitude"].var()




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2418492560.py in <cell line: 0>()
      1 test_df["abs_diff_longitude"] = (
----> 2     np.abs(test_df["abs_diff_longitude"]) - test_df["abs_diff_longitude"].mean()
      3 ) / test_df["abs_diff_longitude"].var()
      4 test_df["abs_diff_latitude"] = (
      5     np.abs(test_df["abs_diff_latitude"]) - test_df["abs_diff_latitude"].mean()

NameError: name 'test_df' is not defined

## === cell 15
X = train_df.drop(["key", "fare_amount"], axis=1)
y = train_df["fare_amount"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.01, random_state=80)

X_train_np = X_train.to_numpy(copy=False).astype(np.float32, copy=False)
X_val_np = X_val.to_numpy(copy=False).astype(np.float32, copy=False)
y_train_np = y_train.to_numpy(copy=False).astype(np.float32, copy=False)
y_val_np = y_val.to_numpy(copy=False).astype(np.float32, copy=False)




## === cell 16
hgb = HistGradientBoostingRegressor(
    max_iter=300,  # equivalent to n_estimators
    learning_rate=0.05,
    max_depth=3,
    random_state=42,
)
hgb.fit(X_train_np, y_train_np)

pred_val = hgb.predict(X_val_np)
rmse = mean_squared_error(y_val_np, pred_val, squared=False)
print(f"RMSE on validation split: {rmse:.4f}")




## === cell 17
test_features = (
    test_df[X_train.columns].to_numpy(copy=False).astype(np.float32, copy=False)
)
pred = np.clip(hgb.predict(test_features), a_min=0, a_max=None)
pred = np.round(pred, 2)




## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3010061013.py in <cell line: 0>()
      1 test_features = (
----> 2     test_df[X_train.columns].to_numpy(copy=False).astype(np.float32, copy=False)
      3 )
      4 pred = np.clip(hgb.predict(test_features), a_min=0, a_max=None)
      5 pred = np.round(pred, 2)

NameError: name 'test_df' is not defined

## === cell 18
Submission = pd.DataFrame({"key": test_df["key"], "fare_amount": pred})
Submission.set_index("key", inplace=True)
Submission.to_csv("Submission.csv")

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2459509812.py in <cell line: 0>()
----> 1 Submission = pd.DataFrame({"key": test_df["key"], "fare_amount": pred})
      2 Submission.set_index("key", inplace=True)
      3 Submission.to_csv("Submission.csv")

NameError: name 'test_df' is not defined
