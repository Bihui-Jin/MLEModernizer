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
geopy==2.4.1
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

3.4594

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 13.37112) has done: 'The changes remove the invalid `%matplotlib inline` line, fix the XGBoost prediction by using the correct Booster API (no `best_ntree_limit` attribute) and update the training parameters to the current objective name. These fixes let the notebook run end‑to‑end and correctly write a `submission.csv` file with the required columns.'
- What this solution (achieved 431.76956) has done: 'I keep the overall workflow and feature set but average the LinearRegression and XGBoost predictions (both already computed) before creating the submission. Averaging usually reduces variance and brings the RMSE down, moving the score closer to the target while leaving the core modeling logic unchanged. The final submission file is still written as `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import xgboost as xgb




## === cell 1
def load_csv(fname, **kwargs):
    """Try several possible locations for a CSV file."""
    possible_paths = [
        pathlib.Path(fname),
        pathlib.Path("../input") / fname,
        pathlib.Path("/kaggle/input") / fname,
        pathlib.Path("/kaggle/working") / fname,
    ]
    for p in possible_paths:
        if p.exists():
            return pd.read_csv(p, **kwargs)
    raise FileNotFoundError(f"Could not find {fname}")




## === cell 2
types = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}

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



## === cell 3
train = load_csv("train.csv", dtype=types, usecols=usecols)
test = load_csv("test.csv", dtype=types, usecols=usecols[:-1])  # test lacks fare_amount



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/908624945.py in <cell line: 0>()
      1 train = load_csv("train.csv", dtype=types, usecols=usecols)
----> 2 test = load_csv("test.csv", dtype=types, usecols=usecols[:-1])  # test lacks fare_amount
      3 

/tmp/ipykernel_11/1374923318.py in load_csv(fname, **kwargs)
      9     for p in possible_paths:
     10         if p.exists():
---> 11             return pd.read_csv(p, **kwargs)
     12     raise FileNotFoundError(f"Could not find {fname}")
     13 

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

## === cell 4
max_rows = 2_000_000
if len(train) > max_rows:
    train = train.sample(n=max_rows, random_state=42).reset_index(drop=True)



## === cell 5
train = train.dropna()
train = train[train["fare_amount"] > 0]
train = train[train["pickup_longitude"] < -72]
train = train[(train["pickup_latitude"] > 40) & (train["pickup_latitude"] < 44)]
train = train[train["dropoff_longitude"] < -72]
train = train[(train["dropoff_latitude"] > 40) & (train["dropoff_latitude"] < 44)]
train = train[(train["passenger_count"] > 0) & (train["passenger_count"] < 10)]




## === cell 6
def quick_dist_calc(df):
    """Add Haversine distance (km) between pickup and dropoff."""
    R = 6373.0
    lat1 = np.radians(df["pickup_latitude"].astype("float64"))
    lon1 = np.radians(df["pickup_longitude"].astype("float64"))
    lat2 = np.radians(df["dropoff_latitude"].astype("float64"))
    lon2 = np.radians(df["dropoff_longitude"].astype("float64"))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    df["distance"] = (R * c).astype("float32")




## === cell 7
quick_dist_calc(train)
quick_dist_calc(test)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/12206831.py in <cell line: 0>()
      1 quick_dist_calc(train)
----> 2 quick_dist_calc(test)
      3 
      4 

NameError: name 'test' is not defined

## === cell 8
def parse_datetime(series):
    return pd.to_datetime(
        series.str.slice(0, 19), format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )


train["pickup_datetime"] = parse_datetime(train["pickup_datetime"])
test["pickup_datetime"] = parse_datetime(test["pickup_datetime"])



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/43487964.py in <cell line: 0>()
      7 
      8 train["pickup_datetime"] = parse_datetime(train["pickup_datetime"])
----> 9 test["pickup_datetime"] = parse_datetime(test["pickup_datetime"])
     10 

NameError: name 'test' is not defined

## === cell 9
for df in (train, test):
    df["hour"] = df["pickup_datetime"].dt.hour.astype("uint8")
    df["weekday"] = df["pickup_datetime"].dt.weekday.astype("uint8")
    df["month"] = df["pickup_datetime"].dt.month.astype("uint8")
    df["year"] = df["pickup_datetime"].dt.year.astype("uint16")




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/899332667.py in <cell line: 0>()
      1 # Time‑based features
----> 2 for df in (train, test):
      3     df["hour"] = df["pickup_datetime"].dt.hour.astype("uint8")
      4     df["weekday"] = df["pickup_datetime"].dt.weekday.astype("uint8")
      5     df["month"] = df["pickup_datetime"].dt.month.astype("uint8")

NameError: name 'test' is not defined

## === cell 10
def sphere_dist(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    R_earth = 6371.0
    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = map(
        np.radians, [pickup_lat, pickup_lon, dropoff_lat, dropoff_lon]
    )
    dlat = dropoff_lat - pickup_lat
    dlon = dropoff_lon - pickup_lon
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(pickup_lat) * np.cos(dropoff_lat) * np.sin(dlon / 2.0) ** 2
    )
    return 2 * R_earth * np.arcsin(np.sqrt(a))


def add_airport_dist(df):
    jfk = (40.639722, -73.778889)
    ewr = (40.6925, -74.168611)
    lga = (40.77725, -73.872611)
    df["jfk_dist"] = pd.concat(
        [
            sphere_dist(df["pickup_latitude"], df["pickup_longitude"], *jfk),
            sphere_dist(df["dropoff_latitude"], df["dropoff_longitude"], *jfk),
        ],
        axis=1,
    ).min(axis=1)
    df["ewr_dist"] = pd.concat(
        [
            sphere_dist(df["pickup_latitude"], df["pickup_longitude"], *ewr),
            sphere_dist(df["dropoff_latitude"], df["dropoff_longitude"], *ewr),
        ],
        axis=1,
    ).min(axis=1)
    df["lga_dist"] = pd.concat(
        [
            sphere_dist(df["pickup_latitude"], df["pickup_longitude"], *lga),
            sphere_dist(df["dropoff_latitude"], df["dropoff_longitude"], *lga),
        ],
        axis=1,
    ).min(axis=1)
    return df


train = add_airport_dist(train)
test = add_airport_dist(test)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1942808622.py in <cell line: 0>()
     43 
     44 train = add_airport_dist(train)
---> 45 test = add_airport_dist(test)
     46 

NameError: name 'test' is not defined

## === cell 11
X = train.drop(columns=["key", "fare_amount", "pickup_datetime"])
y = train["fare_amount"]
y_log = np.log1p(y)  # log‑scale target for better modelling



## === cell 12
np.random.seed(42)
X_train, X_valid, y_train_log, y_valid_log = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)



## === cell 13
lm = LinearRegression()
lm.fit(X_train, y_train_log)
val_pred_log = lm.predict(X_valid)
val_pred = np.expm1(val_pred_log)
val_true = np.expm1(y_valid_log)
linear_rmse = np.sqrt(metrics.mean_squared_error(val_true, val_pred))
print(f"Linear model validation RMSE (original scale): {linear_rmse:.4f}")




## === cell 14
def train_xgb(X_tr, X_va, y_tr, y_va):
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dvalid = xgb.DMatrix(X_va, label=y_va)
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "seed": 42,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "max_bin": 256,
        "nthread": -1,
    }
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=30,
        verbose_eval=False,
    )
    return bst


xgbm = train_xgb(X_train, X_valid, y_train_log, y_valid_log)

xgb_val_log = xgbm.predict(xgb.DMatrix(X_valid))
xgb_val = np.expm1(xgb_val_log)
xgb_rmse = np.sqrt(metrics.mean_squared_error(val_true, xgb_val))
print(f"XGBoost validation RMSE (original scale): {xgb_rmse:.4f}")



## === cell 15
test_features = test.drop(columns=["key", "pickup_datetime"])
linear_test_pred = np.expm1(lm.predict(test_features))
xgb_test_pred = np.expm1(xgbm.predict(xgb.DMatrix(test_features)))



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3897486955.py in <cell line: 0>()
      1 # Predictions on the test set
----> 2 test_features = test.drop(columns=["key", "pickup_datetime"])
      3 linear_test_pred = np.expm1(lm.predict(test_features))
      4 xgb_test_pred = np.expm1(xgbm.predict(xgb.DMatrix(test_features)))
      5 

NameError: name 'test' is not defined

## === cell 16
combined_pred = np.round(((linear_test_pred + xgb_test_pred) / 2), 2)
combined_pred = np.clip(combined_pred, a_min=0, a_max=None)

submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": combined_pred}, columns=["key", "fare_amount"]
)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4064823603.py in <cell line: 0>()
      1 # Simple average ensemble, rounding to 2 decimals and clipping negatives
----> 2 combined_pred = np.round(((linear_test_pred + xgb_test_pred) / 2), 2)
      3 combined_pred = np.clip(combined_pred, a_min=0, a_max=None)
      4 
      5 submission = pd.DataFrame(

NameError: name 'linear_test_pred' is not defined

## === cell 17
for var in [
    "train",
    "test",
    "X",
    "y",
    "X_train",
    "X_valid",
    "y_train_log",
    "y_valid_log",
    "linear_test_pred",
    "xgb_test_pred",
]:
    if var in globals():
        del globals()[var]
