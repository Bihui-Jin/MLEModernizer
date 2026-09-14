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

3.10

# 3. Installed packages

folium==0.20.0
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

3.41572

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 11.92436) has done: 'Your notebook doesn’t yet train a model or write a submission, so it can’t yield a Kaggle score. I keep your existing feature engineering and cleaning intact, then add the smallest necessary steps: compute a simple distance feature, train a baseline scikit-learn regressor on the sampled rows, predict on the test set (keeping `key` aligned), and write `submission.csv` with the required columns. I also avoid dropping `key` from `test` so the submission format is correct. All additions are lightweight and should run within the time limit.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

os.environ.setdefault("PYTHONHASHSEED", "0")
np.random.seed(42)



## === cell 1
fields = [
    "key",  # keep key available; does not change core training semantics (we drop it later)
    "pickup_datetime",
    "fare_amount",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

train_dtypes = {
    "fare_amount": "float64",
    "pickup_longitude": "float64",
    "pickup_latitude": "float64",
    "dropoff_longitude": "float64",
    "dropoff_latitude": "float64",
    "passenger_count": "int64",
}

read_csv_kwargs = dict(
    engine="pyarrow",
)

train = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
    nrows=1_000_000,
    usecols=fields,
    parse_dates=["pickup_datetime"],
    dtype=train_dtypes,
    **read_csv_kwargs,
)
print(f"{train.shape} shape")
train.head()



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2315896721.py in <cell line: 0>()
     24 )
     25 
---> 26 train = pd.read_csv(
     27     "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
     28     nrows=1_000_000,

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
   1641                 and value != getattr(value, "value", default)
   1642             ):
-> 1643                 raise ValueError(
   1644                     f"The {repr(argname)} option is not supported with the "
   1645                     f"'pyarrow' engine"

ValueError: The 'nrows' option is not supported with the 'pyarrow' engine

## === cell 2
_ = train.shape



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1026371117.py in <cell line: 0>()
----> 1 _ = train.shape
      2 

NameError: name 'train' is not defined

## === cell 3
test_dtypes = {
    "pickup_longitude": "float64",
    "pickup_latitude": "float64",
    "dropoff_longitude": "float64",
    "dropoff_latitude": "float64",
    "passenger_count": "int64",
}

test = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
    parse_dates=["pickup_datetime"],
    dtype=test_dtypes,
    engine="pyarrow",
)
print(f"{test.shape} shape")
test.head()



## === cell 4
_ = test.shape



## === cell 5
pass



## === cell 6
pass



## === cell 7
pass



## === cell 8
pass



## === cell 9
pass



## === cell 10
m = (train["passenger_count"] != 208) & (train["fare_amount"] >= 0)
train = train.loc[m]



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3578459456.py in <cell line: 0>()
----> 1 m = (train["passenger_count"] != 208) & (train["fare_amount"] >= 0)
      2 train = train.loc[m]
      3 

NameError: name 'train' is not defined

## === cell 11
dt = train["pickup_datetime"].dt
train["year"] = dt.year.to_numpy()
train["month"] = dt.month.to_numpy()
train["day"] = dt.day.to_numpy()
train["hour"] = dt.hour.to_numpy()
train["minute"] = dt.minute.to_numpy()

dt = test["pickup_datetime"].dt
test["year"] = dt.year.to_numpy()
test["month"] = dt.month.to_numpy()
test["day"] = dt.day.to_numpy()
test["hour"] = dt.hour.to_numpy()
test["minute"] = dt.minute.to_numpy()



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3018131046.py in <cell line: 0>()
----> 1 dt = train["pickup_datetime"].dt
      2 train["year"] = dt.year.to_numpy()
      3 train["month"] = dt.month.to_numpy()
      4 train["day"] = dt.day.to_numpy()
      5 train["hour"] = dt.hour.to_numpy()

NameError: name 'train' is not defined

## === cell 12
train = train.drop(columns=["pickup_datetime"])
test = test.drop(columns=["pickup_datetime"])



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2566746500.py in <cell line: 0>()
----> 1 train = train.drop(columns=["pickup_datetime"])
      2 test = test.drop(columns=["pickup_datetime"])
      3 

NameError: name 'train' is not defined

## === cell 13
_ = train.shape



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1026371117.py in <cell line: 0>()
----> 1 _ = train.shape
      2 

NameError: name 'train' is not defined

## === cell 14
pass



## === cell 15
train.shape



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2553811345.py in <cell line: 0>()
----> 1 train.shape
      2 

NameError: name 'train' is not defined

## === cell 16
pass



## === cell 17
pl = train["pickup_latitude"].to_numpy(dtype=np.float64, copy=False)
dl = train["dropoff_latitude"].to_numpy(dtype=np.float64, copy=False)
plo = train["pickup_longitude"].to_numpy(dtype=np.float64, copy=False)
dlo = train["dropoff_longitude"].to_numpy(dtype=np.float64, copy=False)

train_mask = (
    (pl > 40)
    & (pl < 45)
    & (dl > 40)
    & (dl < 45)
    & (plo < -71)
    & (plo > -79)
    & (dlo < -71)
    & (dlo > -79)
)

tpl = test["pickup_latitude"].to_numpy(dtype=np.float64, copy=False)
tdl = test["dropoff_latitude"].to_numpy(dtype=np.float64, copy=False)
tplo = test["pickup_longitude"].to_numpy(dtype=np.float64, copy=False)
tdlo = test["dropoff_longitude"].to_numpy(dtype=np.float64, copy=False)

test_mask = (
    (tpl > 40)
    & (tpl < 45)
    & (tdl > 40)
    & (tdl < 45)
    & (tplo < -71)
    & (tplo > -79)
    & (tdlo < -71)
    & (tdlo > -79)
)

train = train.loc[train_mask].copy()



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4281658687.py in <cell line: 0>()
----> 1 pl = train["pickup_latitude"].to_numpy(dtype=np.float64, copy=False)
      2 dl = train["dropoff_latitude"].to_numpy(dtype=np.float64, copy=False)
      3 plo = train["pickup_longitude"].to_numpy(dtype=np.float64, copy=False)
      4 dlo = train["dropoff_longitude"].to_numpy(dtype=np.float64, copy=False)
      5 

NameError: name 'train' is not defined

## === cell 18
_ = train.shape



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1026371117.py in <cell line: 0>()
----> 1 _ = train.shape
      2 

NameError: name 'train' is not defined

## === cell 19
train.shape



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2553811345.py in <cell line: 0>()
----> 1 train.shape
      2 

NameError: name 'train' is not defined

## === cell 20
pass




## === cell 21
def haversine_np(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371.0 * c
    return km


tr_plon = train["pickup_longitude"].to_numpy(dtype=np.float64, copy=False)
tr_plat = train["pickup_latitude"].to_numpy(dtype=np.float64, copy=False)
tr_dlon = train["dropoff_longitude"].to_numpy(dtype=np.float64, copy=False)
tr_dlat = train["dropoff_latitude"].to_numpy(dtype=np.float64, copy=False)

te_plon = test["pickup_longitude"].to_numpy(dtype=np.float64, copy=False)
te_plat = test["pickup_latitude"].to_numpy(dtype=np.float64, copy=False)
te_dlon = test["dropoff_longitude"].to_numpy(dtype=np.float64, copy=False)
te_dlat = test["dropoff_latitude"].to_numpy(dtype=np.float64, copy=False)

train["distance_km"] = haversine_np(tr_plon, tr_plat, tr_dlon, tr_dlat)
test["distance_km"] = haversine_np(te_plon, te_plat, te_dlon, te_dlat)

train["abs_lon_diff"] = np.abs(tr_plon - tr_dlon)
train["abs_lat_diff"] = np.abs(tr_plat - tr_dlat)
test["abs_lon_diff"] = np.abs(te_plon - te_dlon)
test["abs_lat_diff"] = np.abs(te_plat - te_dlat)

train_arr = train.select_dtypes(include=[np.number]).to_numpy(
    dtype=np.float64, copy=False
)
test_arr = test.select_dtypes(include=[np.number]).to_numpy(
    dtype=np.float64, copy=False
)
train_finite_mask = np.isfinite(train_arr).all(axis=1)
test_finite_mask = np.isfinite(test_arr).all(axis=1)

train = train.loc[train_finite_mask].copy()
test = test.loc[test_finite_mask].copy()

train.shape, test.shape



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1246495280.py in <cell line: 0>()
     12 
     13 
---> 14 tr_plon = train["pickup_longitude"].to_numpy(dtype=np.float64, copy=False)
     15 tr_plat = train["pickup_latitude"].to_numpy(dtype=np.float64, copy=False)
     16 tr_dlon = train["dropoff_longitude"].to_numpy(dtype=np.float64, copy=False)

NameError: name 'train' is not defined

## === cell 22
from sklearn.ensemble import GradientBoostingRegressor

y = train["fare_amount"].to_numpy(dtype=np.float64, copy=False)
X = train.drop(columns=["fare_amount"], errors="ignore")

X = X.drop(columns=["key"], errors="ignore")
X_test = test.drop(columns=["key"], errors="ignore")

X_test = X_test.reindex(columns=X.columns, fill_value=0)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_np = np.ascontiguousarray(X_train.to_numpy(dtype=np.float64, copy=False))
X_valid_np = np.ascontiguousarray(X_valid.to_numpy(dtype=np.float64, copy=False))
X_test_np = np.ascontiguousarray(X_test.to_numpy(dtype=np.float64, copy=False))

gbr = GradientBoostingRegressor(
    random_state=42,
    n_estimators=400,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    loss="squared_error",
)

gbr.fit(X_train_np, y_train)

valid_pred = gbr.predict(X_valid_np)
rmse = mean_squared_error(y_valid, valid_pred, squared=False)
print("Validation RMSE:", rmse)



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1139254488.py in <cell line: 0>()
      1 from sklearn.ensemble import GradientBoostingRegressor
      2 
----> 3 y = train["fare_amount"].to_numpy(dtype=np.float64, copy=False)
      4 X = train.drop(columns=["fare_amount"], errors="ignore")
      5 

NameError: name 'train' is not defined

## === cell 23
test_pred = gbr.predict(X_test_np)

test_pred = np.maximum(test_pred, 0)

submission = pd.DataFrame({"key": test["key"].values, "fare_amount": test_pred})

submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
submission.head()

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/104161518.py in <cell line: 0>()
----> 1 test_pred = gbr.predict(X_test_np)
      2 
      3 test_pred = np.maximum(test_pred, 0)
      4 
      5 submission = pd.DataFrame({"key": test["key"].values, "fare_amount": test_pred})

NameError: name 'gbr' is not defined
