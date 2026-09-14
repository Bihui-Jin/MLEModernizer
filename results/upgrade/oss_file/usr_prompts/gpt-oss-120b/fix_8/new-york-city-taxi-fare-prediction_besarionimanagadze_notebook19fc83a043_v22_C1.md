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

5.4971

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 7.62517) has done: 'I fix the boolean filter, replace the simple linear regression with a properly configured XGBoost regressor (the huge score gap justifies this change), add an RMSE computation on a validation split so you can see the improvement, and ensure the final predictions are written to a correctly‑formatted `submission.csv`. The rest of the preprocessing steps remain unchanged.'
- What this solution (achieved 7.57288) has done: 'I tighten the training filter by discarding rides with implausibly long distances (distance > 200 km) and give the XGBoost model a bit more capacity (more trees, slightly deeper, lower learning rate). After prediction I replace any negative fare estimates with 0, which removes invalid outputs without altering the core modeling approach. These modest tweaks are expected to lower the validation RMSE toward the target 5.4971 while keeping the overall pipeline unchanged.'
- What this solution (achieved 7.69548) has done: 'I loosen the downtown‑distance filter from 15 km to 30 km to keep more realistic rides, and I give the XGBoost model a modestly larger capacity with n_estimators = 800, max_depth = 10, learning_rate = 0.04 plus early‑stopping on the validation split. These small tweaks keep the original pipeline intact while aiming to lower the validation RMSE toward the target 5.4971.'
- What this solution (achieved 6.08894) has done: 'I train the model on a log‑transformed fare amount (using log1p and expm1 to keep predictions on the original scale) and slightly increase model capacity (more trees, lower learning rate). This small change aligns the loss with the skewed target distribution and is expected to lower the RMSE toward the target without altering the overall pipeline.'
- What this solution (achieved 6.30398) has done: 'I tighten the training filter by discarding implausibly high fares (fare < 200 USD) and give the XGBoost model a modestly higher capacity (more trees, slightly deeper, slightly larger subsample/colsample). These small, targeted tweaks are expected to lower the validation RMSE, moving the score from 6.09 closer to the target 5.4971, while preserving the overall pipeline and output format.'
- What this solution (achieved 6.87727) has done: 'I add a simple “distance per passenger” feature (distance divided by passenger count) to give the model more information about ride efficiency, and I give XGBoost a bit more capacity by increasing the number of trees, depth, and using the full sample of rows/columns per tree while keeping early‑stopping to avoid over‑fit. These small, targeted tweaks are expected to lower the validation RMSE and move the score closer to the target without altering the overall pipeline.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O
import matplotlib.pyplot as plt  # plotting library
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
train_data_set = pd.read_csv(
    "../input/new-york-city-taxi-prediction/train.csv",
    nrows=100_000,
    parse_dates=["pickup_datetime"],
)
train_data_set.head(5)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_56/3289615730.py in <cell line: 0>()
----> 1 train_data_set = pd.read_csv(
      2     "../input/new-york-city-taxi-prediction/train.csv",
      3     nrows=100_000,
      4     parse_dates=["pickup_datetime"],
      5 )

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

FileNotFoundError: [Errno 2] No such file or directory: '../input/new-york-city-taxi-prediction/train.csv'

## === cell 2
print(train_data_set.dtypes)
train_data_set.describe()



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/3005885289.py in <cell line: 0>()
----> 1 print(train_data_set.dtypes)
      2 train_data_set.describe()
      3 

NameError: name 'train_data_set' is not defined

## === cell 3
old_len = len(train_data_set)
train_data_set = train_data_set[train_data_set.fare_amount >= 0.1]
new_len = len(train_data_set)
print(f"Removed {(old_len - new_len)} rows with fare < 0.1")
train_data_set.describe()



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/616105454.py in <cell line: 0>()
----> 1 old_len = len(train_data_set)
      2 train_data_set = train_data_set[train_data_set.fare_amount >= 0.1]
      3 new_len = len(train_data_set)
      4 print(f"Removed {(old_len - new_len)} rows with fare < 0.1")
      5 train_data_set.describe()

NameError: name 'train_data_set' is not defined

## === cell 4
old_len = len(train_data_set)
train_data_set = train_data_set.dropna(how="any", axis="rows")
new_len = len(train_data_set)
print(f"Removed {(old_len - new_len)} rows with missing values")



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/1006643359.py in <cell line: 0>()
----> 1 old_len = len(train_data_set)
      2 train_data_set = train_data_set.dropna(how="any", axis="rows")
      3 new_len = len(train_data_set)
      4 print(f"Removed {(old_len - new_len)} rows with missing values")
      5 

NameError: name 'train_data_set' is not defined

## === cell 5
train_data_set.fare_amount.hist(bins=100, figsize=(14, 3))
plt.xlabel("fare $USD")
plt.title("Histogram of fare_amount")
plt.show()




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/1844128165.py in <cell line: 0>()
----> 1 train_data_set.fare_amount.hist(bins=100, figsize=(14, 3))
      2 plt.xlabel("fare $USD")
      3 plt.title("Histogram of fare_amount")
      4 plt.show()
      5 

NameError: name 'train_data_set' is not defined

## === cell 6
def select_within_boundingbox(df, box):
    return (
        (df.pickup_longitude >= box[0])
        & (df.pickup_longitude <= box[1])
        & (df.pickup_latitude >= box[2])
        & (df.pickup_latitude <= box[3])
        & (df.dropoff_longitude >= box[0])
        & (df.dropoff_longitude <= box[1])
        & (df.dropoff_latitude >= box[2])
        & (df.dropoff_latitude <= box[3])
    )


new_york_box = (-74.763379, -72.856164, 40.502009, 41.915509)

old_len = len(train_data_set)
train_data_set = train_data_set[select_within_boundingbox(train_data_set, new_york_box)]
new_len = len(train_data_set)
print(f"Removed {(old_len - new_len)} rows outside NYC bounding box")




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/3542506448.py in <cell line: 0>()
     14 new_york_box = (-74.763379, -72.856164, 40.502009, 41.915509)
     15 
---> 16 old_len = len(train_data_set)
     17 train_data_set = train_data_set[select_within_boundingbox(train_data_set, new_york_box)]
     18 new_len = len(train_data_set)

NameError: name 'train_data_set' is not defined

## === cell 7
def distance_on_the_sphere(lat1, lon1, lat2, lon2):
    earth_radius = 6371  # km
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return earth_radius * c


train_data_set["distance"] = distance_on_the_sphere(
    train_data_set["pickup_latitude"],
    train_data_set["pickup_longitude"],
    train_data_set["dropoff_latitude"],
    train_data_set["dropoff_longitude"],
)

train_data_set.head(5)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/1675484808.py in <cell line: 0>()
     14 
     15 train_data_set["distance"] = distance_on_the_sphere(
---> 16     train_data_set["pickup_latitude"],
     17     train_data_set["pickup_longitude"],
     18     train_data_set["dropoff_latitude"],

NameError: name 'train_data_set' is not defined

## === cell 8
train_data_set["pickup_datetime"] = pd.to_datetime(train_data_set["pickup_datetime"])
train_data_set["hour"] = train_data_set["pickup_datetime"].dt.hour
train_data_set["year"] = train_data_set["pickup_datetime"].dt.year
train_data_set["day_of_week"] = train_data_set["pickup_datetime"].dt.dayofweek
train_data_set["is_rush_hour"] = train_data_set["hour"].apply(
    lambda x: 1 if (7 <= x <= 10) or (16 <= x <= 19) else 0
)

train_data_set.head(5)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/3403519140.py in <cell line: 0>()
----> 1 train_data_set["pickup_datetime"] = pd.to_datetime(train_data_set["pickup_datetime"])
      2 train_data_set["hour"] = train_data_set["pickup_datetime"].dt.hour
      3 train_data_set["year"] = train_data_set["pickup_datetime"].dt.year
      4 train_data_set["day_of_week"] = train_data_set["pickup_datetime"].dt.dayofweek
      5 train_data_set["is_rush_hour"] = train_data_set["hour"].apply(

NameError: name 'train_data_set' is not defined

## === cell 9
nyc_downtown = (-74.0063889, 40.7141667)  # (lon, lat)

train_data_set["distance_to_downtown"] = distance_on_the_sphere(
    nyc_downtown[1],
    nyc_downtown[0],
    train_data_set["pickup_latitude"],
    train_data_set["pickup_longitude"],
)

train_data_set.head(5)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/3286409481.py in <cell line: 0>()
      4     nyc_downtown[1],
      5     nyc_downtown[0],
----> 6     train_data_set["pickup_latitude"],
      7     train_data_set["pickup_longitude"],
      8 )

NameError: name 'train_data_set' is not defined

## === cell 10
max_distance_km = 100  # reduced from 200 km
max_fare = 150  # reduced from 200 USD

idx = (
    (train_data_set["passenger_count"] != 0)
    & (train_data_set["distance_to_downtown"] < 30)  # keep existing downtown filter
    & (train_data_set["distance"] <= max_distance_km)
    & (train_data_set["fare_amount"] < max_fare)
)

train_data_set["distance_per_passenger"] = (
    train_data_set["distance"] / train_data_set["passenger_count"]
)
train_data_set["distance_per_passenger"] = np.clip(
    train_data_set["distance_per_passenger"], 0, 100
)

features = [
    "hour",
    "year",
    "distance",
    "passenger_count",
    "is_rush_hour",
    "day_of_week",
    "distance_to_downtown",
    "distance_per_passenger",
]
target = "fare_amount"

X = train_data_set.loc[idx, features].values
y = train_data_set.loc[idx, target].values

y_log = np.log1p(y)

X_train, X_val, y_train_log, y_val_log = train_test_split(
    X, y_log, test_size=0.25, random_state=42
)

xgb_model = xgb.XGBRegressor(
    n_estimators=2000,  # slightly fewer trees
    learning_rate=0.02,
    max_depth=12,  # shallower trees
    subsample=1.0,
    colsample_bytree=1.0,
    reg_lambda=1.0,  # L2 regularisation
    reg_alpha=0.0,  # L1 regularisation
    objective="reg:squarederror",
    eval_metric="rmse",
    n_jobs=5,
    random_state=35,
    tree_method="hist",
    device="cpu",
)

xgb_model.fit(
    X_train,
    y_train_log,
    eval_set=[(X_val, y_val_log)],
    early_stopping_rounds=50,
    verbose=False,
)

val_pred = np.expm1(xgb_model.predict(X_val))
val_actual = np.expm1(y_val_log)
val_rmse = mean_squared_error(val_actual, val_pred, squared=False)
print(f"Validation RMSE: {val_rmse:.4f}")



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/3176702099.py in <cell line: 0>()
      4 
      5 idx = (
----> 6     (train_data_set["passenger_count"] != 0)
      7     & (train_data_set["distance_to_downtown"] < 30)  # keep existing downtown filter
      8     & (train_data_set["distance"] <= max_distance_km)

NameError: name 'train_data_set' is not defined

## === cell 11
test_data_set = pd.read_csv("../input/new-york-city-taxi-prediction/test.csv")

test_data_set["distance"] = distance_on_the_sphere(
    test_data_set["pickup_latitude"],
    test_data_set["pickup_longitude"],
    test_data_set["dropoff_latitude"],
    test_data_set["dropoff_longitude"],
)

test_data_set["distance_to_downtown"] = distance_on_the_sphere(
    nyc_downtown[1],
    nyc_downtown[0],
    test_data_set["pickup_latitude"],
    test_data_set["pickup_longitude"],
)

test_data_set["distance_per_passenger"] = (
    test_data_set["distance"] / test_data_set["passenger_count"]
)
test_data_set["distance_per_passenger"] = np.clip(
    test_data_set["distance_per_passenger"], 0, 100
)

test_data_set["pickup_datetime"] = pd.to_datetime(test_data_set["pickup_datetime"])
test_data_set["hour"] = test_data_set["pickup_datetime"].dt.hour
test_data_set["year"] = test_data_set["pickup_datetime"].dt.year
test_data_set["day_of_week"] = test_data_set["pickup_datetime"].dt.dayofweek
test_data_set["is_rush_hour"] = test_data_set["hour"].apply(
    lambda x: 1 if (7 <= x <= 10) or (16 <= x <= 19) else 0
)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_56/1973753393.py in <cell line: 0>()
----> 1 test_data_set = pd.read_csv("../input/new-york-city-taxi-prediction/test.csv")
      2 
      3 test_data_set["distance"] = distance_on_the_sphere(
      4     test_data_set["pickup_latitude"],
      5     test_data_set["pickup_longitude"],

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

FileNotFoundError: [Errno 2] No such file or directory: '../input/new-york-city-taxi-prediction/test.csv'

## === cell 12
X_test = test_data_set[features].values
test_pred_log = xgb_model.predict(X_test)
test_pred = np.expm1(test_pred_log)

test_pred = np.where(test_pred < 0, 0, test_pred)

submission = pd.DataFrame(
    {"key": test_data_set["key"], "fare_amount": test_pred},
    columns=["key", "fare_amount"],
)

submission.to_csv("submission.csv", index=False)
print("Submission file 'submission.csv' written. Sample rows:")
print(submission.head())



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/3869278681.py in <cell line: 0>()
----> 1 X_test = test_data_set[features].values
      2 test_pred_log = xgb_model.predict(X_test)
      3 test_pred = np.expm1(test_pred_log)
      4 
      5 test_pred = np.where(test_pred < 0, 0, test_pred)

NameError: name 'test_data_set' is not defined

## === cell 13
neg_count = (submission["fare_amount"] < 0).sum()
print(f"Number of negative fare predictions: {neg_count}")

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/2350509189.py in <cell line: 0>()
----> 1 neg_count = (submission["fare_amount"] < 0).sum()
      2 print(f"Number of negative fare predictions: {neg_count}")

NameError: name 'submission' is not defined
