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
scipy==1.15.3
seaborn==0.12.2
sklearn-pandas==2.2.0
xgboost==2.0.3

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

3.70811

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 4.575) has done: 'The fix updates the prediction step to avoid using the non‑existent `best_ntree_limit` attribute on the Booster object, ensuring a valid prediction array is generated and the submission DataFrame is created correctly.'
- What this solution (achieved 4.49979) has done: 'I keep the overall pipeline unchanged but add the most informative temporal and passenger‑count features back into the model and gently strengthen the XGBoost hyper‑parameters (lower learning rate, deeper trees, a few regularisation tweaks). These tweaks are minimal, preserve the core logic, and are expected to lower the RMSE from 4.575 toward the target 3.708 without over‑fitting.'
- What this solution (achieved 4.37935) has done: 'I load more training rows (2 million instead of 1 million) to give the model more data, slightly strengthen the XGBoost model (lower learning rate, deeper trees) while keeping early stopping, and remove the manual rounding of predictions so the submission reflects the model’s raw output—these minimal changes should lower the RMSE toward the target without altering the overall pipeline.'
- What this solution (achieved 4.59384) has done: 'I load a slightly larger training sample (3 M rows) and add a new interaction feature `dist_pass` = haversine × passenger_count. This feature is created for both train and test, scaled together with the other distance features, and kept throughout the pipeline. The added feature gives the model more information about how passenger count relates to travel distance, which should modestly reduce the RMSE and move the score closer to the target without altering the core modeling logic.'
- What this solution (achieved 4.43821) has done: 'Implemented vectorized distance calculations to eliminate per‑row Python loops and added XGBoost’s fast histogram tree method with explicit thread control. The new logic computes all required distance metrics using NumPy broadcasting, preserving the original feature set and model behavior while dramatically reducing runtime. No changes to data paths, model architecture, or training semantics were made.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import datetime as dt
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearnex import patch_sklearn

patch_sklearn()

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import Birch
import xgboost as xgb
import gc  # for explicit garbage collection



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
dtype_map = {
    "key": "object",
    "fare_amount": "float32",
    "pickup_datetime": "object",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int32",
}
train = pd.read_csv(
    "../input/train.csv", nrows=10_000_000, usecols=usecols, dtype=dtype_map
)
test = pd.read_csv("../input/test.csv", usecols=usecols, dtype=dtype_map)

coord_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]
train[coord_cols] = train[coord_cols].astype(np.float32)
test[coord_cols] = test[coord_cols].astype(np.float32)

num_int_cols = ["passenger_count"]



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1322023374.py in <cell line: 0>()
     23     "../input/train.csv", nrows=10_000_000, usecols=usecols, dtype=dtype_map
     24 )
---> 25 test = pd.read_csv("../input/test.csv", usecols=usecols, dtype=dtype_map)
     26 
     27 # Ensure coordinate columns are float32 (saves memory & speeds later ops)

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
print("Sum of NaN values for each column")
print(train.isnull().sum())

train = train.dropna()
print("Sum of NaN values for each column after dropping NaN")
print(train.isnull().sum())



## === cell 3
pickup_longitude_min = test.pickup_longitude.min()
pickup_longitude_max = test.pickup_longitude.max()
pickup_latitude_min = test.pickup_latitude.min()
pickup_latitude_max = test.pickup_latitude.max()
dropoff_longitude_min = test.dropoff_longitude.min()
dropoff_longitude_max = test.dropoff_longitude.max()
dropoff_latitude_min = test.dropoff_latitude.min()
dropoff_latitude_max = test.dropoff_latitude.max()

mask = (
    (train["fare_amount"] > 0)
    & (train["fare_amount"] < 300)
    & (train["pickup_longitude"] > pickup_longitude_min)
    & (train["pickup_longitude"] < pickup_longitude_max)
    & (train["pickup_latitude"] > pickup_latitude_min)
    & (train["pickup_latitude"] < pickup_latitude_max)
    & (train["dropoff_longitude"] > dropoff_longitude_min)
    & (train["dropoff_longitude"] < dropoff_longitude_max)
    & (train["dropoff_latitude"] > dropoff_latitude_min)
    & (train["dropoff_latitude"] < dropoff_latitude_max)
    & (train["passenger_count"] <= 8)
)
train = train.loc[mask]
train.describe()




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/799198666.py in <cell line: 0>()
      1 # Compute bounds once on the test set
----> 2 pickup_longitude_min = test.pickup_longitude.min()
      3 pickup_longitude_max = test.pickup_longitude.max()
      4 pickup_latitude_min = test.pickup_latitude.min()
      5 pickup_latitude_max = test.pickup_latitude.max()

NameError: name 'test' is not defined

## === cell 4
def haversine_vec(lat1, lon1, lat2, lon2):
    rlat1, rlon1, rlat2, rlon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = rlon2 - rlon1
    dlat = rlat2 - rlat1
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km




## === cell 5
dist_types = [
    "chebyshev",
    "euclidean",
    "canberra",
    "sqeuclidean",
    "braycurtis",
    "minkowski",
    "hamming",
    "cityblock",
]

for dataset in (train, test):
    lat1 = dataset["pickup_latitude"].values.astype(np.float32)
    lon1 = dataset["pickup_longitude"].values.astype(np.float32)
    lat2 = dataset["dropoff_latitude"].values.astype(np.float32)
    lon2 = dataset["dropoff_longitude"].values.astype(np.float32)

    dataset["haversine"] = haversine_vec(lat1, lon1, lat2, lon2).astype(np.float32)
    dataset["dist_pass"] = dataset["haversine"] * dataset["passenger_count"]

    dlat = np.abs(lat2 - lat1)
    dlon = np.abs(lon1 - lon2)

    dataset["chebyshev"] = np.maximum(dlat, dlon)
    dataset["euclidean"] = np.sqrt(dlat**2 + dlon**2)

    with np.errstate(divide="ignore", invalid="ignore"):
        ca_lat = dlat / (np.abs(lat1) + np.abs(lat2))
        ca_lon = dlon / (np.abs(lon1) + np.abs(lon2))
        ca_lat = np.nan_to_num(ca_lat)
        ca_lon = np.nan_to_num(ca_lon)
    dataset["canberra"] = ca_lat + ca_lon

    dataset["sqeuclidean"] = dlat**2 + dlon**2

    bc_num = dlat + dlon
    bc_den = np.abs(lat1) + np.abs(lat2) + np.abs(lon1) + np.abs(lon2)
    with np.errstate(divide="ignore", invalid="ignore"):
        dataset["braycurtis"] = bc_num / bc_den
        dataset["braycurtis"] = np.nan_to_num(dataset["braycurtis"])

    dataset["minkowski"] = dataset["euclidean"]
    dataset["hamming"] = 1.0
    dataset["cityblock"] = dlat + dlon

    dt_series = pd.to_datetime(
        dataset["pickup_datetime"], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce"
    )
    dataset["hour_of_day"] = dt_series.dt.hour.astype(np.int8)
    dataset["day"] = dt_series.dt.day.astype(np.int8)
    dataset["week"] = dt_series.dt.isocalendar().week.astype(np.int16)
    dataset["month"] = dt_series.dt.month.astype(np.int8)
    dataset["day_of_year"] = dt_series.dt.dayofyear.astype(np.int16)
    dataset["week_of_year"] = dt_series.dt.isocalendar().week.astype(np.int16)

gc.collect()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3287089404.py in <cell line: 0>()
     10 ]
     11 
---> 12 for dataset in (train, test):
     13     # Build arrays once
     14     lat1 = dataset["pickup_latitude"].values.astype(np.float32)

NameError: name 'test' is not defined

## === cell 6
features = [
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
    "haversine",
    "dist_pass",
] + dist_types

train[features] = train[features]
test[features] = test[features]



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2675305719.py in <cell line: 0>()
      8 ] + dist_types
      9 
---> 10 train[features] = train[features]
     11 test[features] = test[features]
     12 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['haversine', 'dist_pass', 'chebyshev', 'euclidean', 'canberra', 'sqeuclidean', 'braycurtis', 'minkowski', 'hamming', 'cityblock'] not in index"

## === cell 7
coords = [
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
]
concat = pd.concat([train[coords], test[coords]], ignore_index=True)
db = Birch(
    branching_factor=50, n_clusters=None, threshold=0.5, compute_labels=True
).fit(concat)
labels = db.labels_

train["cluster"] = labels[: train.shape[0]].astype(np.int16)
test["cluster"] = labels[train.shape[0] :].astype(np.int16)

del concat, db, labels
gc.collect()



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1496047450.py in <cell line: 0>()
      5     "dropoff_longitude",
      6 ]
----> 7 concat = pd.concat([train[coords], test[coords]], ignore_index=True)
      8 db = Birch(
      9     branching_factor=50, n_clusters=None, threshold=0.5, compute_labels=True

NameError: name 'test' is not defined

## === cell 8
pca_features = ["haversine"] + dist_types

train[pca_features] = train[pca_features].fillna(-1)
test[pca_features] = test[pca_features].fillna(-1)

pca = PCA(n_components=3, random_state=42)
p_result = pca.fit_transform(train[pca_features])
train["pca0"], train["pca1"], train["pca2"] = p_result.T

p_result = pca.transform(test[pca_features])
test["pca0"], test["pca1"], test["pca2"] = p_result.T

del p_result, pca
gc.collect()



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2946421617.py in <cell line: 0>()
      1 pca_features = ["haversine"] + dist_types
      2 
----> 3 train[pca_features] = train[pca_features].fillna(-1)
      4 test[pca_features] = test[pca_features].fillna(-1)
      5 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6247         if nmissing:
   6248             if nmissing == len(indexer):
-> 6249                 raise KeyError(f"None of [{key}] are in the [{axis_name}]")
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())

KeyError: "None of [Index(['haversine', 'chebyshev', 'euclidean', 'canberra', 'sqeuclidean',\n       'braycurtis', 'minkowski', 'hamming', 'cityblock'],\n      dtype='object')] are in the [columns]"

## === cell 9
good_dist = ["pca0", "pca1", "pca2", "cluster", "haversine", "dist_pass"] + dist_types
temporal_features = [
    "hour_of_day",
    "day",
    "week",
    "month",
    "day_of_year",
    "week_of_year",
]
train_features_to_keep = (
    ["fare_amount", "passenger_count"]
    + ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]
    + good_dist
    + temporal_features
)
train = train[train_features_to_keep]

test_features_to_keep = (
    ["key", "passenger_count"]
    + ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]
    + good_dist
    + temporal_features
)
test = test[test_features_to_keep]
x_pred = test.drop("key", axis=1)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1521395152.py in <cell line: 0>()
     14     + temporal_features
     15 )
---> 16 train = train[train_features_to_keep]
     17 
     18 test_features_to_keep = (

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['pca0', 'pca1', 'pca2', 'cluster', 'haversine', 'dist_pass', 'chebyshev', 'euclidean', 'canberra', 'sqeuclidean', 'braycurtis', 'minkowski', 'hamming', 'cityblock', 'hour_of_day', 'day', 'week', 'month', 'day_of_year', 'week_of_year'] not in index"

## === cell 10
x_train_df, x_test_df, y_train, y_test = train_test_split(
    train.drop("fare_amount", axis=1),
    train["fare_amount"],
    random_state=123,
    test_size=0.2,
)

x_train_df = x_train_df.astype(np.float32)
x_test_df = x_test_df.astype(np.float32)
x_pred = x_pred.astype(np.float32)

del train
gc.collect()




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1239846009.py in <cell line: 0>()
      6 )
      7 
----> 8 x_train_df = x_train_df.astype(np.float32)
      9 x_test_df = x_test_df.astype(np.float32)
     10 x_pred = x_pred.astype(np.float32)

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in astype(self, dtype, copy, errors)
   6641         else:
   6642             # else, only a single dtype is given
-> 6643             new_data = self._mgr.astype(dtype=dtype, copy=copy, errors=errors)
   6644             res = self._constructor_from_mgr(new_data, axes=new_data.axes)
   6645             return res.__finalize__(self, method="astype")

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in astype(self, dtype, copy, errors)
    428             copy = False
    429 
--> 430         return self.apply(
    431             "astype",
    432             dtype=dtype,

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in apply(self, f, align_keys, **kwargs)
    361                 applied = b.apply(f, **kwargs)
    362             else:
--> 363                 applied = getattr(b, f)(**kwargs)
    364             result_blocks = extend_blocks(applied, result_blocks)
    365 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/blocks.py in astype(self, dtype, copy, errors, using_cow, squeeze)
    756             values = values[0, :]  # type: ignore[call-overload]
    757 
--> 758         new_values = astype_array_safe(values, dtype, copy=copy, errors=errors)
    759 
    760         new_values = maybe_coerce_values(new_values)

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in astype_array_safe(values, dtype, copy, errors)
    235 
    236     try:
--> 237         new_values = astype_array(values, dtype, copy=copy)
    238     except (ValueError, TypeError):
    239         # e.g. _astype_nansafe can fail on object-dtype of strings

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in astype_array(values, dtype, copy)
    180 
    181     else:
--> 182         values = _astype_nansafe(values, dtype, copy=copy)
    183 
    184     # in pandas we don't store numpy str dtypes, so convert to object

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in _astype_nansafe(arr, dtype, copy, skipna)
    131     if copy or arr.dtype == object or dtype == object:
    132         # Explicit copy, or required since NumPy can't view from / to object.
--> 133         return arr.astype(dtype, copy=True)
    134 
    135     return arr.astype(dtype, copy=copy)

ValueError: could not convert string to float: '2010-08-20 12:47:00.00000070'

## === cell 11
def XGBmodel(x_train, x_test, y_train, y_test):
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)
    model = xgb.train(
        params={
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "eta": 0.03,
            "max_depth": 10,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "lambda": 1.2,
            "gamma": 0.1,
            "tree_method": "hist",
            "max_bin": 256,
            "nthread": os.cpu_count(),  # ensure full parallelism
        },
        dtrain=dtrain,
        num_boost_round=3000,
        early_stopping_rounds=50,
        evals=[(dtest, "test")],
        verbose_eval=False,
    )
    return model


model = XGBmodel(x_train_df, x_test_df, y_train, y_test)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2415296333.py in <cell line: 0>()
     26 
     27 
---> 28 model = XGBmodel(x_train_df, x_test_df, y_train, y_test)
     29 

/tmp/ipykernel_11/2415296333.py in XGBmodel(x_train, x_test, y_train, y_test)
      1 def XGBmodel(x_train, x_test, y_train, y_test):
----> 2     dtrain = xgb.DMatrix(x_train, label=y_train)
      3     dtest = xgb.DMatrix(x_test, label=y_test)
      4     model = xgb.train(
      5         params={

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in inner_f(*args, **kwargs)
    728             for k, arg in zip(sig.parameters, args):
    729                 kwargs[k] = arg
--> 730             return func(**kwargs)
    731 
    732         return inner_f

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in __init__(self, data, label, weight, base_margin, missing, silent, feature_names, feature_types, nthread, group, qid, label_lower_bound, label_upper_bound, feature_weights, enable_categorical, data_split_mode)
    855             return
    856 
--> 857         handle, feature_names, feature_types = dispatch_data_backend(
    858             data,
    859             missing=self.missing,

/usr/local/lib/python3.11/dist-packages/xgboost/data.py in dispatch_data_backend(data, missing, threads, feature_names, feature_types, enable_categorical, data_split_mode)
   1087         data = pd.DataFrame(data)
   1088     if _is_pandas_df(data):
-> 1089         return _from_pandas_df(
   1090             data, enable_categorical, missing, threads, feature_names, feature_types
   1091         )

/usr/local/lib/python3.11/dist-packages/xgboost/data.py in _from_pandas_df(data, enable_categorical, missing, nthread, feature_names, feature_types)
    520     feature_types: Optional[FeatureTypes],
    521 ) -> DispatchedDataBackendReturnType:
--> 522     data, feature_names, feature_types = _transform_pandas_df(
    523         data, enable_categorical, feature_names, feature_types
    524     )

/usr/local/lib/python3.11/dist-packages/xgboost/data.py in _transform_pandas_df(data, enable_categorical, feature_names, feature_types, meta, meta_type)
    488             or is_pa_ext_dtype(dtype)
    489         ):
--> 490             _invalid_dataframe_dtype(data)
    491         if is_pa_ext_dtype(dtype):
    492             pyarrow_extension = True

/usr/local/lib/python3.11/dist-packages/xgboost/data.py in _invalid_dataframe_dtype(data)
    306     type_err = "DataFrame.dtypes for data must be int, float, bool or category."
    307     msg = f"""{type_err} {_ENABLE_CAT_ERR} {err}"""
--> 308     raise ValueError(msg)
    309 
    310 

ValueError: DataFrame.dtypes for data must be int, float, bool or category. When categorical type is supplied, The experimental DMatrix parameter`enable_categorical` must be set to `True`.  Invalid columns:key: object, pickup_datetime: object

## === cell 12
prediction = model.predict(xgb.DMatrix(x_pred))
submission = pd.DataFrame({"key": test["key"], "fare_amount": prediction})
submission.to_csv("sub_fare.csv", index=False)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3147137417.py in <cell line: 0>()
----> 1 prediction = model.predict(xgb.DMatrix(x_pred))
      2 submission = pd.DataFrame({"key": test["key"], "fare_amount": prediction})
      3 submission.to_csv("sub_fare.csv", index=False)
      4 

NameError: name 'model' is not defined

## === cell 13
submission.head()

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3365464162.py in <cell line: 0>()
----> 1 submission.head()

NameError: name 'submission' is not defined
