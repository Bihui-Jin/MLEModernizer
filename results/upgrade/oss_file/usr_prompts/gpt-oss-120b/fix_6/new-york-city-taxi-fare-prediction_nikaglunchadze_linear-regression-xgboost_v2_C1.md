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

3.12

# 3. Installed packages

geopandas==0.14.4
joblib==1.5.2
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

3.54762

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 10.32915) has done: 'I keep the existing preprocessing and feature engineering unchanged, but improve the model by training the XGBoost regressor on the full filtered dataset (instead of only the 80 % split) and by increasing its capacity slightly (more trees and deeper depth). This uses more data, which should lower the RMSE toward the target while preserving the overall pipeline logic.'
- What this solution (achieved 9.85491) has done: 'I train the XGBoost model on the original training split (not on the whole dataframe) to avoid leakage, and I increase its capacity modestly (more trees, deeper depth, lower learning rate). This should lower the validation RMSE, moving the score toward the target while preserving the overall pipeline.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

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

dtype_spec = {
    "key": "object",
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}

ny_latitude_min, ny_latitude_max = 40.4772, 45.0153
ny_longitude_min, ny_longitude_max = -79.7624, -71.7517


def load_and_filter_train(path):
    """Load the massive CSV in larger chunks and apply all filters once."""
    chunks = []
    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        dtype=dtype_spec,
        parse_dates=False,
        chunksize=1_000_000,
    ):
        chunk = chunk.dropna()
        mask = (
            (chunk["pickup_longitude"] >= ny_longitude_min)
            & (chunk["pickup_longitude"] <= ny_longitude_max)
            & (chunk["pickup_latitude"] >= ny_latitude_min)
            & (chunk["pickup_latitude"] <= ny_latitude_max)
            & (chunk["dropoff_longitude"] >= ny_longitude_min)
            & (chunk["dropoff_longitude"] <= ny_longitude_max)
            & (chunk["dropoff_latitude"] >= ny_latitude_min)
            & (chunk["dropoff_latitude"] <= ny_latitude_max)
            & (chunk["passenger_count"] >= 1)
            & (chunk["passenger_count"] <= 6)
        )
        chunks.append(chunk[mask])
    return pd.concat(chunks, ignore_index=True)


df = load_and_filter_train("/kaggle/input/new-york-city-taxi-fare-prediction/train.csv")
test_df = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
    usecols=usecols[:-1],  # test has no fare_amount
    dtype={k: v for k, v in dtype_spec.items() if k != "fare_amount"},
    parse_dates=False,
)
df.dtypes




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3018834104.py in <cell line: 0>()
     63 
     64 df = load_and_filter_train("/kaggle/input/new-york-city-taxi-fare-prediction/train.csv")
---> 65 test_df = pd.read_csv(
     66     "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
     67     usecols=usecols[:-1],  # test has no fare_amount

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

## === cell 1
df.describe()




## === cell 2
df.head()




## === cell 3
def filter_column(df, column, range_min, range_max):
    return df[(df[column] >= range_min) & (df[column] <= range_max)]


df = filter_column(df, "fare_amount", 1, 200)




## === cell 4
df[df["fare_amount"] > 200].describe()




## === cell 5
def refactor_datetime(df):
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["year"] = df["pickup_datetime"].dt.year.astype("int16")
    df["month"] = df["pickup_datetime"].dt.month.astype("int8")
    df["day"] = df["pickup_datetime"].dt.day.astype("int8")
    df["weekday"] = df["pickup_datetime"].dt.weekday.astype("int8")
    df["hour"] = df["pickup_datetime"].dt.hour.astype("int8")
    df.drop(columns=["pickup_datetime"], inplace=True)


refactor_datetime(df)
refactor_datetime(test_df)
df.head()




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4135076365.py in <cell line: 0>()
     10 
     11 refactor_datetime(df)
---> 12 refactor_datetime(test_df)
     13 df.head()
     14 

NameError: name 'test_df' is not defined

## === cell 6
def haversine(p1, p2):
    """Fully vectorised haversine distance (kilometres)."""
    lat1, lon1 = p1
    lat2, lon2 = p2
    lat1 = np.radians(lat1.astype(np.float32))
    lon1 = np.radians(lon1.astype(np.float32))
    lat2 = np.radians(lat2.astype(np.float32))
    lon2 = np.radians(lon2.astype(np.float32))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    dist = 2 * np.arcsin(np.sqrt(a))
    km = 6367.0 * dist
    return km


ny_center = ("ny_center", (40.7128, -74.0060))
jfk_airport = ("jfk_airport", (40.6446, -73.7797))
lga_airport = ("lga_airport", (40.7733, -73.8718))
ewr_airport = ("ewr_airport", (40.6895, -74.1745))

locs = [ny_center, jfk_airport, lga_airport, ewr_airport]




## === cell 7
def insert_haversine_dists(df, locations):
    """Add distance columns for each supplied location."""
    for loc_name, (lat_ref, lon_ref) in locations:
        df[f"pickup_dist_to_{loc_name}"] = haversine(
            (df["pickup_latitude"], df["pickup_longitude"]), (lat_ref, lon_ref)
        )
        df[f"dropoff_dist_to_{loc_name}"] = haversine(
            (df["dropoff_latitude"], df["dropoff_longitude"]), (lat_ref, lon_ref)
        )
    df["ride_distance"] = haversine(
        (df["pickup_latitude"], df["pickup_longitude"]),
        (df["dropoff_latitude"], df["dropoff_longitude"]),
    )


insert_haversine_dists(df, locs)
insert_haversine_dists(test_df, locs)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3752237010.py in <cell line: 0>()
     14 
     15 
---> 16 insert_haversine_dists(df, locs)
     17 insert_haversine_dists(test_df, locs)
     18 

/tmp/ipykernel_11/3752237010.py in insert_haversine_dists(df, locations)
      2     """Add distance columns for each supplied location."""
      3     for loc_name, (lat_ref, lon_ref) in locations:
----> 4         df[f"pickup_dist_to_{loc_name}"] = haversine(
      5             (df["pickup_latitude"], df["pickup_longitude"]), (lat_ref, lon_ref)
      6         )

/tmp/ipykernel_11/2413388769.py in haversine(p1, p2)
      6     lat1 = np.radians(lat1.astype(np.float32))
      7     lon1 = np.radians(lon1.astype(np.float32))
----> 8     lat2 = np.radians(lat2.astype(np.float32))
      9     lon2 = np.radians(lon2.astype(np.float32))
     10     dlon = lon2 - lon1

AttributeError: 'float' object has no attribute 'astype'

## === cell 8
df.describe()




## === cell 9
df = df[df["ride_distance"] > 0]
df.describe()




## --- ERROR in cell 9, traceback:
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

KeyError: 'ride_distance'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2787855517.py in <cell line: 0>()
----> 1 df = df[df["ride_distance"] > 0]
      2 df.describe()
      3 
      4 

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

KeyError: 'ride_distance'

## === cell 10
from sklearn.model_selection import train_test_split

train_df, validation_df = train_test_split(df, test_size=0.2, random_state=SEED)




## === cell 11
features = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "year",
    "month",
    "day",
    "hour",
    "weekday",
    "ride_distance",
]
features += [f"pickup_dist_to_{x[0]}" for x in locs]
features += [f"dropoff_dist_to_{x[0]}" for x in locs]
fare_amount = "fare_amount"

train_features = train_df[features]
train_fare_amount = train_df[fare_amount]

validation_features = validation_df[features]
validation_fare_amount = validation_df[fare_amount]
train_features.info()




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1944825591.py in <cell line: 0>()
     16 fare_amount = "fare_amount"
     17 
---> 18 train_features = train_df[features]
     19 train_fare_amount = train_df[fare_amount]
     20 

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

KeyError: "['ride_distance', 'pickup_dist_to_ny_center', 'pickup_dist_to_jfk_airport', 'pickup_dist_to_lga_airport', 'pickup_dist_to_ewr_airport', 'dropoff_dist_to_ny_center', 'dropoff_dist_to_jfk_airport', 'dropoff_dist_to_lga_airport', 'dropoff_dist_to_ewr_airport'] not in index"

## === cell 12
from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()




## === cell 13
def estimate_model(model, df):
    pass




## === cell 14
linear_model.fit(train_features, train_fare_amount)




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1605233932.py in <cell line: 0>()
----> 1 linear_model.fit(train_features, train_fare_amount)
      2 
      3 

NameError: name 'train_features' is not defined

## === cell 15
from sklearn.metrics import mean_squared_error

linear_predictions = linear_model.predict(validation_features)
mean_squared_error(validation_fare_amount, linear_predictions, squared=False)




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3084046227.py in <cell line: 0>()
      1 from sklearn.metrics import mean_squared_error
      2 
----> 3 linear_predictions = linear_model.predict(validation_features)
      4 mean_squared_error(validation_fare_amount, linear_predictions, squared=False)
      5 

NameError: name 'validation_features' is not defined

## === cell 16
from xgboost import XGBRegressor

xgb_model = XGBRegressor(
    objective="reg:squarederror",
    learning_rate=0.05,
    n_estimators=400,  # reduced while early stopping determines optimal rounds
    max_depth=10,
    min_child_weight=2,
    subsample=0.8,
    colsample_bytree=0.8,
    max_bin=256,  # faster histogram construction
    n_jobs=-1,
    random_state=SEED,
    tree_method="hist",
)

xgb_model.fit(
    train_features,
    train_fare_amount,
    eval_set=[(validation_features, validation_fare_amount)],
    early_stopping_rounds=50,
    verbose=False,
)

xgb_predictions_val = xgb_model.predict(validation_features)
val_rmse = mean_squared_error(
    validation_fare_amount, xgb_predictions_val, squared=False
)
print("Validation RMSE (with early stopping):", val_rmse)




## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/945205200.py in <cell line: 0>()
     16 
     17 xgb_model.fit(
---> 18     train_features,
     19     train_fare_amount,
     20     eval_set=[(validation_features, validation_fare_amount)],

NameError: name 'train_features' is not defined

## === cell 17
xgb_predictions_train = xgb_model.predict(train_features)
train_rmse = mean_squared_error(train_fare_amount, xgb_predictions_train, squared=False)
print("Training RMSE:", train_rmse)




## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/240667313.py in <cell line: 0>()
----> 1 xgb_predictions_train = xgb_model.predict(train_features)
      2 train_rmse = mean_squared_error(train_fare_amount, xgb_predictions_train, squared=False)
      3 print("Training RMSE:", train_rmse)
      4 
      5 

NameError: name 'train_features' is not defined

## === cell 18
xgb_predictions_test = xgb_model.predict(test_df[features])
submission = pd.DataFrame({"key": test_df["key"], "fare_amount": xgb_predictions_test})
submission.to_csv("submission.csv", index=False)

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1712082271.py in <cell line: 0>()
----> 1 xgb_predictions_test = xgb_model.predict(test_df[features])
      2 submission = pd.DataFrame({"key": test_df["key"], "fare_amount": xgb_predictions_test})
      3 submission.to_csv("submission.csv", index=False)

NameError: name 'test_df' is not defined
