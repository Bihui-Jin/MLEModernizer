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

3.11

# 3. Installed packages

geopandas==0.14.4
lightgbm==4.6.0
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
plotly==5.24.1
plotly-express==0.4.1
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

4.377

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgbm

pd.options.mode.copy_on_write = False
pd.set_option("display.float_format", lambda x: "%.5f" % x)




## === cell 1
def haversine_np(lon1, lat1, lon2, lat2):
    """Vector‑ised haversine distance (km) – operates directly on float32 arrays."""
    R = 6371.0  # Earth radius in km
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c




## === cell 2
DATA_ROOT = os.path.join("/", "kaggle", "data")
TRAIN_PATH = os.path.join(DATA_ROOT, "train.csv")
TEST_PATH = os.path.join(DATA_ROOT, "test.csv")
OUTPUT_SUB_PATH = os.path.join(DATA_ROOT, "submission.csv")

print("Loading data...")

TRAIN_USECOLS = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

TEST_USECOLS = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

train_df = pd.read_csv(
    TRAIN_PATH,
    usecols=TRAIN_USECOLS,
    dtype={
        "fare_amount": np.float32,
        "pickup_longitude": np.float32,
        "pickup_latitude": np.float32,
        "dropoff_longitude": np.float32,
        "dropoff_latitude": np.float32,
        "passenger_count": np.int8,
    },
    parse_dates=["pickup_datetime"],
    engine="pyarrow",
    low_memory=False,
    memory_map=True,
)

test_df = pd.read_csv(
    TEST_PATH,
    usecols=TEST_USECOLS,
    dtype={
        "pickup_longitude": np.float32,
        "pickup_latitude": np.float32,
        "dropoff_longitude": np.float32,
        "dropoff_latitude": np.float32,
        "passenger_count": np.int8,
    },
    parse_dates=["pickup_datetime"],
    engine="pyarrow",
    low_memory=False,
    memory_map=True,
)

test_keys = test_df["key"].copy()

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/864602758.py in <cell line: 0>()
     27 
     28 # Use the pyarrow engine for much faster CSV parsing
---> 29 train_df = pd.read_csv(
     30     TRAIN_PATH,
     31     usecols=TRAIN_USECOLS,

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
   1605         self._currow = 0
   1606 
-> 1607         options = self._get_options_with_defaults(engine)
   1608         options["storage_options"] = kwds.get("storage_options", None)
   1609 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _get_options_with_defaults(self, engine)
   1658                         pass
   1659                     else:
-> 1660                         raise ValueError(
   1661                             f"The {repr(argname)} option is not supported with the "
   1662                             f"{repr(engine)} engine"

ValueError: The 'low_memory' option is not supported with the 'pyarrow' engine

## === cell 3
def add_features(df):
    dt = df["pickup_datetime"]
    df["pickup_hour"] = dt.dt.hour.astype(np.int8)
    df["pickup_dayofweek"] = dt.dt.dayofweek.astype(np.int8)  # Monday=0
    df["pickup_month"] = dt.dt.month.astype(np.int8)

    lon1 = df["pickup_longitude"].values
    lat1 = df["pickup_latitude"].values
    lon2 = df["dropoff_longitude"].values
    lat2 = df["dropoff_latitude"].values

    df["distance_km"] = haversine_np(lon1, lat1, lon2, lat2).astype(np.float32)

    df["manhattan_km"] = ((np.abs(lon1 - lon2) + np.abs(lat1 - lat2)) * 111.0).astype(
        np.float32
    )

    df.drop(columns=["pickup_datetime"], inplace=True)
    return df


train_df = add_features(train_df)
test_df = add_features(test_df)

FEATURES = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "pickup_hour",
    "pickup_dayofweek",
    "pickup_month",
    "distance_km",
    "manhattan_km",
]

X = train_df[FEATURES]
y = train_df["fare_amount"]
X_test = test_df[FEATURES]

del train_df, test_df
gc.collect()




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1554567277.py in <cell line: 0>()
     20 
     21 
---> 22 train_df = add_features(train_df)
     23 test_df = add_features(test_df)
     24 

NameError: name 'train_df' is not defined

## === cell 4
X_np = X.to_numpy(dtype=np.float32, copy=False)
y_np = y.to_numpy(dtype=np.float32, copy=False)

num_rows = X_np.shape[0]
all_idx = np.arange(num_rows, dtype=np.int64)
train_idx, val_idx = train_test_split(
    all_idx, test_size=0.2, random_state=42, shuffle=True
)

MAX_VAL_SIZE = 1_000_000
if val_idx.size > MAX_VAL_SIZE:
    rng = np.random.default_rng(42)
    val_idx = rng.choice(val_idx, size=MAX_VAL_SIZE, replace=False)

lgb_full = lgbm.Dataset(X_np, label=y_np, free_raw_data=False)
lgb_train = lgb_full.subset(train_idx)
lgb_valid = lgb_full.subset(val_idx)

params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.1,
    "num_leaves": 31,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "verbosity": -1,
    "seed": 42,
    "max_bin": 50,  # already reduced for speed
    "bin_construct_sample_cnt": 200000,
    "num_threads": os.cpu_count() or 1,
}

print("Training LightGBM model...")
model = lgbm.train(
    params,
    lgb_train,
    num_boost_round=150,
    valid_sets=[lgb_train, lgb_valid],
    early_stopping_rounds=30,
    verbose_eval=False,
)

val_pred = model.predict(X_np[val_idx], num_iteration=model.best_iteration)
val_rmse = np.sqrt(mean_squared_error(y_np[val_idx], val_pred))
print(f"Validation RMSE: {val_rmse:.4f}")

del X, y, X_np, y_np, lgb_full, lgb_train, lgb_valid
gc.collect()




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3508020504.py in <cell line: 0>()
----> 1 X_np = X.to_numpy(dtype=np.float32, copy=False)
      2 y_np = y.to_numpy(dtype=np.float32, copy=False)
      3 
      4 num_rows = X_np.shape[0]
      5 all_idx = np.arange(num_rows, dtype=np.int64)

NameError: name 'X' is not defined

## === cell 5
print("Predicting on test data...")
test_pred = model.predict(
    X_test.to_numpy(dtype=np.float32, copy=False), num_iteration=model.best_iteration
)

submission = pd.DataFrame({"key": test_keys, "fare_amount": test_pred})
submission.to_csv(OUTPUT_SUB_PATH, index=False)
print(f"Submission written to {OUTPUT_SUB_PATH}")

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1394404069.py in <cell line: 0>()
      1 print("Predicting on test data...")
----> 2 test_pred = model.predict(
      3     X_test.to_numpy(dtype=np.float32, copy=False), num_iteration=model.best_iteration
      4 )
      5 

NameError: name 'model' is not defined
