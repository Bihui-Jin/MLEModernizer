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

4.38819

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 5.9123) has done: 'I fixed the XGBoost usage by updating the objective name, removing the deprecated `best_ntree_limit` attribute, and using the Booster’s built‑in prediction (which automatically uses the best iteration from early stopping). I also removed the Jupyter‑specific magic command and ensured the script writes a proper `submission.csv` with the required columns, so the pipeline runs end‑to‑end and produces a valid Kaggle submission file.'
- What this solution (achieved 5.83436) has done: 'I add simple time‑based features (pickup hour and weekday) to capture temporal patterns in fares, and tune the XGBoost parameters slightly (learning rate, max depth, subsample, colsample) which usually improves RMSE without changing the overall model structure. These minimal changes keep the core logic intact while helping the model move closer to the target score.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")




## === cell 1
train_usecols = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
dtype_map = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
_read_kwargs = dict(
    usecols=train_usecols,
    dtype=dtype_map,
    parse_dates=False,
    low_memory=False,
)
try:
    import pyarrow  # noqa: F401

    _read_kwargs["engine"] = "pyarrow"
    _read_kwargs.pop("low_memory", None)
except Exception:
    pass

train_path = os.path.abspath(
    os.path.join(os.getcwd(), "..", "..", "input", "train.csv")
)
train_df = pd.read_csv(train_path, **_read_kwargs)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3342130649.py in <cell line: 0>()
     36     os.path.join(os.getcwd(), "..", "..", "input", "train.csv")
     37 )
---> 38 train_df = pd.read_csv(train_path, **_read_kwargs)
     39 
     40 

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
    880         else:
    881             # Binary mode
--> 882             handle = open(handle, ioargs.mode)
    883         handles.append(handle)
    884 

FileNotFoundError: [Errno 2] No such file or directory: '/input/train.csv'

## === cell 2
print("Train shape:", train_df.shape)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/788216486.py in <cell line: 0>()
----> 1 print("Train shape:", train_df.shape)
      2 
      3 

NameError: name 'train_df' is not defined

## === cell 3
test_usecols = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
test_dtype_map = {
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
test_kwargs = dict(
    usecols=test_usecols,
    dtype=test_dtype_map,
    parse_dates=False,
    low_memory=False,
)
try:
    import pyarrow

    test_kwargs["engine"] = "pyarrow"
    test_kwargs.pop("low_memory", None)
except Exception:
    pass

test_path = os.path.abspath(os.path.join(os.getcwd(), "..", "..", "input", "test.csv"))
test_df = pd.read_csv(test_path, **test_kwargs)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3234998393.py in <cell line: 0>()
     30 
     31 test_path = os.path.abspath(os.path.join(os.getcwd(), "..", "..", "input", "test.csv"))
---> 32 test_df = pd.read_csv(test_path, **test_kwargs)
     33 
     34 

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
    880         else:
    881             # Binary mode
--> 882             handle = open(handle, ioargs.mode)
    883         handles.append(handle)
    884 

FileNotFoundError: [Errno 2] No such file or directory: '/input/test.csv'

## === cell 4
print("Test shape:", test_df.shape)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1736776108.py in <cell line: 0>()
----> 1 print("Test shape:", test_df.shape)
      2 
      3 

NameError: name 'test_df' is not defined

## === cell 5
print("Missing values per column:\n", train_df.isnull().sum())




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2668189094.py in <cell line: 0>()
----> 1 print("Missing values per column:\n", train_df.isnull().sum())
      2 
      3 

NameError: name 'train_df' is not defined

## === cell 6
train_df.dropna(inplace=True)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3501830334.py in <cell line: 0>()
----> 1 train_df.dropna(inplace=True)
      2 
      3 

NameError: name 'train_df' is not defined

## === cell 7
print(train_df.describe())




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3916936870.py in <cell line: 0>()
----> 1 print(train_df.describe())
      2 
      3 

NameError: name 'train_df' is not defined

## === cell 8
train_df = train_df[train_df["fare_amount"] > 0]




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3211135739.py in <cell line: 0>()
----> 1 train_df = train_df[train_df["fare_amount"] > 0]
      2 
      3 

NameError: name 'train_df' is not defined

## === cell 9
print("After fare filter:", train_df.shape)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/907840599.py in <cell line: 0>()
----> 1 print("After fare filter:", train_df.shape)
      2 
      3 

NameError: name 'train_df' is not defined

## === cell 10
def distance(lat1, lon1, lat2, lon2):
    lat1_r, lon1_r, lat2_r, lon2_r = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return 2 * 3958.8 * np.arcsin(np.sqrt(a))




## === cell 11
train_df["distance"] = distance(
    train_df["pickup_latitude"],
    train_df["pickup_longitude"],
    train_df["dropoff_latitude"],
    train_df["dropoff_longitude"],
)




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/863684513.py in <cell line: 0>()
      1 train_df["distance"] = distance(
----> 2     train_df["pickup_latitude"],
      3     train_df["pickup_longitude"],
      4     train_df["dropoff_latitude"],
      5     train_df["dropoff_longitude"],

NameError: name 'train_df' is not defined

## === cell 12
test_df["distance"] = distance(
    test_df["pickup_latitude"],
    test_df["pickup_longitude"],
    test_df["dropoff_latitude"],
    test_df["dropoff_longitude"],
)




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/166012443.py in <cell line: 0>()
      1 test_df["distance"] = distance(
----> 2     test_df["pickup_latitude"],
      3     test_df["pickup_longitude"],
      4     test_df["dropoff_latitude"],
      5     test_df["dropoff_longitude"],

NameError: name 'test_df' is not defined

## === cell 13
train_df = train_df[train_df["distance"] < 15]




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1567776250.py in <cell line: 0>()
----> 1 train_df = train_df[train_df["distance"] < 15]
      2 
      3 

NameError: name 'train_df' is not defined

## === cell 14
print(train_df.describe())




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3916936870.py in <cell line: 0>()
----> 1 print(train_df.describe())
      2 
      3 

NameError: name 'train_df' is not defined

## === cell 15
train_df = train_df[
    (train_df["passenger_count"] != 0) & (train_df["passenger_count"] < 10)
]




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/178041917.py in <cell line: 0>()
----> 1 train_df = train_df[
      2     (train_df["passenger_count"] != 0) & (train_df["passenger_count"] < 10)
      3 ]
      4 
      5 

NameError: name 'train_df' is not defined

## === cell 16
train_df["pickup_datetime"] = pd.to_datetime(train_df["pickup_datetime"])
test_df["pickup_datetime"] = pd.to_datetime(test_df["pickup_datetime"])

train_df["pickup_hour"] = train_df["pickup_datetime"].dt.hour.astype("int8")
train_df["pickup_weekday"] = train_df["pickup_datetime"].dt.weekday.astype("int8")
train_df["pickup_month"] = train_df["pickup_datetime"].dt.month.astype("int8")
train_df["is_weekend"] = (train_df["pickup_weekday"] >= 5).astype("int8")

test_df["pickup_hour"] = test_df["pickup_datetime"].dt.hour.astype("int8")
test_df["pickup_weekday"] = test_df["pickup_datetime"].dt.weekday.astype("int8")
test_df["pickup_month"] = test_df["pickup_datetime"].dt.month.astype("int8")
test_df["is_weekend"] = (test_df["pickup_weekday"] >= 5).astype("int8")

train_df.drop(columns=["pickup_datetime"], inplace=True)
test_df.drop(columns=["pickup_datetime"], inplace=True)




## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1773620433.py in <cell line: 0>()
      1 # Convert datetime once and create derived temporal features.
----> 2 train_df["pickup_datetime"] = pd.to_datetime(train_df["pickup_datetime"])
      3 test_df["pickup_datetime"] = pd.to_datetime(test_df["pickup_datetime"])
      4 
      5 train_df["pickup_hour"] = train_df["pickup_datetime"].dt.hour.astype("int8")

NameError: name 'train_df' is not defined

## === cell 17
feat_cols = [
    "distance",
    "passenger_count",
    "pickup_hour",
    "pickup_weekday",
    "pickup_month",
    "is_weekend",
]
X = train_df[feat_cols].astype("float32")
y = train_df["fare_amount"]




## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2952958502.py in <cell line: 0>()
      7     "is_weekend",
      8 ]
----> 9 X = train_df[feat_cols].astype("float32")
     10 y = train_df["fare_amount"]
     11 

NameError: name 'train_df' is not defined

## === cell 18
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)




## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/684273471.py in <cell line: 0>()
      1 from sklearn.model_selection import train_test_split
      2 
----> 3 X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
      4 
      5 

NameError: name 'X' is not defined

## === cell 19
import xgboost as xgb




## === cell 20
def train_xgboost(X_tr, X_va, y_tr, y_va, num_rounds=800):
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_va, label=y_va)
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "seed": 42,
        "eta": 0.1,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "max_bin": 128,
        "nthread": -1,
    }
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_rounds,
        evals=[(dval, "validation")],
        early_stopping_rounds=30,
        verbose_eval=False,
    )
    return booster




## === cell 21
xgb_model = train_xgboost(X_train, X_val, y_train, y_val)




## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/462529384.py in <cell line: 0>()
----> 1 xgb_model = train_xgboost(X_train, X_val, y_train, y_val)
      2 
      3 

NameError: name 'X_train' is not defined

## === cell 22
test_pred = xgb_model.predict(xgb.DMatrix(test_df[feat_cols].astype("float32")))




## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1007860359.py in <cell line: 0>()
----> 1 test_pred = xgb_model.predict(xgb.DMatrix(test_df[feat_cols].astype("float32")))
      2 
      3 

NameError: name 'xgb_model' is not defined

## === cell 23
test_pred = np.clip(test_pred, a_min=0, a_max=None)
submission = pd.DataFrame({"key": test_df["key"], "fare_amount": test_pred})
submission.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1402367978.py in <cell line: 0>()
----> 1 test_pred = np.clip(test_pred, a_min=0, a_max=None)
      2 submission = pd.DataFrame({"key": test_df["key"], "fare_amount": test_pred})
      3 submission.to_csv("submission.csv", index=False)
      4 print("Submission file saved as submission.csv")

NameError: name 'test_pred' is not defined
