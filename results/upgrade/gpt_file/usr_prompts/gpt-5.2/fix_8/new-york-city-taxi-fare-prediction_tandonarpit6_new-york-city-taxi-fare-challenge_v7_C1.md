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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
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

3.98929

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 4.62664) has done: 'I fix the runtime error by calling `XGBRegressor.predict(X_test)` (the scikit-learn API doesn’t accept `data=`), which also unblock creation of `Y_pred` and prevent the downstream `NameError`. I also ensure `Y_train` is a 1D Series (not a single-column DataFrame) so XGBoost trains/predicts consistently. Finally, I write the submission to a new filename (to avoid overwriting Kaggle’s `sample_submission.csv`) and guarantee the output has exactly the required columns `key,fare_amount`, producing a valid `.csv` submission end-to-end.'
- What this solution (achieved 7.09168) has done: 'Your current RMSE (4.62664) is worse than the target (3.98929), so we should make small, score-relevant improvements without changing the overall approach (still XGBRegressor on the same engineered features). The biggest low-risk gains here come from (1) cleaning obviously bad training rows (invalid lat/lon, non-positive/unrealistic fares, passenger_count out of range), because NYC Taxi Fare is very sensitive to outliers, and (2) adding just a couple of standard, minimal XGBoost hyperparameters (more trees + smaller learning rate) to better fit the cleaned data while staying within time. I also ensure train/test columns align and clamp negative predictions to 0 (fares can’t be negative), which typically improves RMSE a bit. These changes keep your feature set and training flow intact while moving the score toward the target.'
- What this solution (achieved 6.07147) has done: 'We keep your feature engineering and XGBRegressor approach intact, but fix the biggest remaining score drag: training on a lot of “weird” trips that pass the broad bounding-box filter yet are still outliers (e.g., huge coordinate jumps) and missing an easy, core distance signal. Concretely, we (1) add one standard feature (haversine distance) derived only from existing lat/lon, (2) slightly tighten data cleaning with a max-distance sanity filter (removes extreme outliers that inflate RMSE), and (3) add a small amount of time-derived signal (month/year) without changing the overall training loop. These are minimal, score-relevant changes that usually move RMSE down toward your target without altering the modeling “core logic”. The submission writing stays the same and still produce a valid `submission.csv`.'
- What this solution (achieved 5.28389) has done: 'Your current RMSE (6.07147) is worse than the target (3.98929), so we should make the smallest, score-relevant improvements without changing the overall approach (same feature engineering + XGBRegressor). The biggest low-risk gain on this competition is stricter outlier removal: filter out “impossible” coordinates (0,0), extreme trip distances, and extreme fare-per-km rows that create huge squared errors. To keep the core logic intact, we only add these additional cleaning masks and keep the model/training flow the same; we also apply the same basic coordinate validity filter to the test set to avoid NaNs/infs and then fill any remaining missing values. This should reduce RMSE materially by removing high-error noise while preserving the same model and features.'
- What this solution (achieved 5.01223) has done: 'Your current RMSE (5.28389) is worse than the target (3.98929), so we should make small, score-relevant changes that keep the same overall approach (feature engineering + XGBRegressor) but reduce noisy training signal. The biggest low-risk improvement here is to align the feature set more with the physics of fares by keeping dropoff coordinates (they carry strong location information beyond just distance) and adding a standard “center point” (midpoint) feature; this doesn’t change your modeling approach, only the inputs. I also make the outlier filtering slightly more robust by removing extremely slow/long trips via a tighter maximum haversine distance (which typically reduces squared-error impact from mislabeled/outlier rows) while staying conservative. Submission writing stays identical and still produces a valid `submission.csv`.'
- What this solution (achieved 4.98522) has done: 'Your RMSE (5.01223) is still worse than the target (3.98929), so we should make small, score-relevant improvements without changing the overall XGBRegressor + feature-engineering approach. The biggest remaining drag is label noise/outliers that survive the broad filters; adding a standard “fare-per-km” and “minimum fare” consistency filter (using the already-computed haversine distance) typically reduces squared-error blowups and improves RMSE. To keep semantics stable, I also ensure we filter on finite distances and avoid divide-by-zero by computing the distance once and reusing it for filtering. Everything else (features, model family, training flow, submission format) stays the same.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

BASE_INPUT = "/kaggle/input"
if not os.path.exists(BASE_INPUT):
    BASE_INPUT = "../input"

print("BASE_INPUT:", BASE_INPUT)
print("Top-level input dirs:", os.listdir(BASE_INPUT)[:20])

COMP_DIR = os.path.join(BASE_INPUT, "new-york-city-taxi-fare-prediction")
if not os.path.exists(COMP_DIR):
    COMP_DIR = BASE_INPUT

train_path = os.path.join(COMP_DIR, "train.csv")
test_path = os.path.join(COMP_DIR, "test.csv")
sample_path = os.path.join(COMP_DIR, "sample_submission.csv")

print("Resolved paths:")
print("train:", train_path)
print("test:", test_path)
print("sample_submission:", sample_path)



## === cell 1
training_data = pd.read_csv(
    train_path,
    nrows=2_000_000,
    parse_dates=["pickup_datetime"],
)
test_data = pd.read_csv(
    test_path,
    parse_dates=["pickup_datetime"],
)

print("train shape:", training_data.shape)
print("test shape:", test_data.shape)



## === cell 2
training_data



## === cell 3
X_train = training_data.copy()
X_test = test_data.copy()
Y_train = training_data.copy()




## === cell 4
def haversine_km(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1.astype(float))
    lat1 = np.radians(lat1.astype(float))
    lon2 = np.radians(lon2.astype(float))
    lat2 = np.radians(lat2.astype(float))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c


def extract_dropoff_datetime_from_key(key_series):
    s = key_series.astype(str).str.slice(0, 26)
    return pd.to_datetime(s, errors="coerce", format="%Y-%m-%d %H:%M:%S.%f")


mask = (
    X_train["fare_amount"].notna()
    & X_train["pickup_longitude"].between(-75, -72)
    & X_train["dropoff_longitude"].between(-75, -72)
    & X_train["pickup_latitude"].between(40, 42)
    & X_train["dropoff_latitude"].between(40, 42)
    & X_train["passenger_count"].between(1, 6)
    & X_train["fare_amount"].between(2.5, 250.0)
)

mask = mask & ~(
    (X_train["pickup_longitude"].abs() < 1e-6)
    | (X_train["pickup_latitude"].abs() < 1e-6)
    | (X_train["dropoff_longitude"].abs() < 1e-6)
    | (X_train["dropoff_latitude"].abs() < 1e-6)
)

training_data = training_data.loc[mask].reset_index(drop=True)

dist_km = haversine_km(
    training_data["pickup_longitude"],
    training_data["pickup_latitude"],
    training_data["dropoff_longitude"],
    training_data["dropoff_latitude"],
)

finite_dist = np.isfinite(dist_km)
training_data = training_data.loc[finite_dist].reset_index(drop=True)
dist_km = dist_km[finite_dist].reset_index(drop=True)

same_point = (
    training_data["pickup_longitude"] == training_data["dropoff_longitude"]
) & (training_data["pickup_latitude"] == training_data["dropoff_latitude"])
training_data = training_data.loc[~same_point].reset_index(drop=True)
dist_km = dist_km.loc[~same_point].reset_index(drop=True)

dist_mask = dist_km.between(0.05, 45.0)
training_data = training_data.loc[dist_mask].reset_index(drop=True)
dist_km = dist_km.loc[dist_mask].reset_index(drop=True)

fare_per_km = training_data["fare_amount"] / dist_km
min_fare_ok = training_data["fare_amount"] >= (2.5 + 0.5 * dist_km)
fpkm_ok = fare_per_km.between(1.2, 25.0)

training_data = training_data.loc[min_fare_ok & fpkm_ok].reset_index(drop=True)

dropoff_dt = extract_dropoff_datetime_from_key(training_data["key"])
dur_min = (dropoff_dt - training_data["pickup_datetime"]).dt.total_seconds() / 60.0
dur_ok = dur_min.between(1.0, 240.0) & np.isfinite(dur_min)

fare_per_min = training_data["fare_amount"] / dur_min
fpm_ok = fare_per_min.between(0.2, 30.0)

training_data = training_data.loc[dur_ok & fpm_ok].reset_index(drop=True)

X_train = training_data.copy()
Y_train = training_data["fare_amount"].copy()

print("after cleaning train shape:", X_train.shape)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/datetimelike.py in _sub_datetimelike(self, other)
   1164         try:
-> 1165             self._assert_tzawareness_compat(other)
   1166         except TypeError as err:

/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/datetimes.py in _assert_tzawareness_compat(self, other)
    781             if other_tz is not None:
--> 782                 raise TypeError(
    783                     "Cannot compare tz-naive and tz-aware datetime-like objects."

TypeError: Cannot compare tz-naive and tz-aware datetime-like objects.

The above exception was the direct cause of the following exception:

TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3103994360.py in <cell line: 0>()
     71 # Rationale: these outliers create huge squared errors and are common in this dataset (bad/misaligned timestamps).
     72 dropoff_dt = extract_dropoff_datetime_from_key(training_data["key"])
---> 73 dur_min = (dropoff_dt - training_data["pickup_datetime"]).dt.total_seconds() / 60.0
     74 dur_ok = dur_min.between(1.0, 240.0) & np.isfinite(dur_min)
     75 

/usr/local/lib/python3.11/dist-packages/pandas/core/ops/common.py in new_method(self, other)
     74         other = item_from_zerodim(other)
     75 
---> 76         return method(self, other)
     77 
     78     return new_method

/usr/local/lib/python3.11/dist-packages/pandas/core/arraylike.py in __sub__(self, other)
    192     @unpack_zerodim_and_defer("__sub__")
    193     def __sub__(self, other):
--> 194         return self._arith_method(other, operator.sub)
    195 
    196     @unpack_zerodim_and_defer("__rsub__")

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in _arith_method(self, other, op)
   6133     def _arith_method(self, other, op):
   6134         self, other = self._align_for_op(other)
-> 6135         return base.IndexOpsMixin._arith_method(self, other, op)
   6136 
   6137     def _align_for_op(self, right, align_asobject: bool = False):

/usr/local/lib/python3.11/dist-packages/pandas/core/base.py in _arith_method(self, other, op)
   1380 
   1381         with np.errstate(all="ignore"):
-> 1382             result = ops.arithmetic_op(lvalues, rvalues, op)
   1383 
   1384         return self._construct_result(result, name=res_name)

/usr/local/lib/python3.11/dist-packages/pandas/core/ops/array_ops.py in arithmetic_op(left, right, op)
    271         # Timedelta/Timestamp and other custom scalars are included in the check
    272         # because numexpr will fail on it, see GH#31457
--> 273         res_values = op(left, right)
    274     else:
    275         # TODO we should handle EAs consistently and move this check before the if/else

/usr/local/lib/python3.11/dist-packages/pandas/core/ops/common.py in new_method(self, other)
     74         other = item_from_zerodim(other)
     75 
---> 76         return method(self, other)
     77 
     78     return new_method

/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/datetimelike.py in __sub__(self, other)
   1457         ):
   1458             # DatetimeIndex, ndarray[datetime64]
-> 1459             result = self._sub_datetime_arraylike(other)
   1460         elif isinstance(other_dtype, PeriodDtype):
   1461             # PeriodIndex

/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/datetimelike.py in _sub_datetime_arraylike(self, other)
   1154 
   1155         self, other = self._ensure_matching_resos(other)
-> 1156         return self._sub_datetimelike(other)
   1157 
   1158     @final

/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/datetimelike.py in _sub_datetimelike(self, other)
   1166         except TypeError as err:
   1167             new_message = str(err).replace("compare", "subtract")
-> 1168             raise type(err)(new_message) from err
   1169 
   1170         other_i8, o_mask = self._get_i8_values_and_mask(other)

TypeError: Cannot subtract tz-naive and tz-aware datetime-like objects.

## === cell 5
dropoff_dt = extract_dropoff_datetime_from_key(X_train["key"])
X_train["trip_duration_min"] = (
    (dropoff_dt - X_train["pickup_datetime"]).dt.total_seconds() / 60.0
).clip(lower=1.0, upper=240.0)

X_train["hour"] = X_train["pickup_datetime"].dt.hour
X_train["dayofweek"] = X_train["pickup_datetime"].dt.dayofweek
X_train["month"] = X_train["pickup_datetime"].dt.month
X_train["year"] = X_train["pickup_datetime"].dt.year

X_train["latitude_distance"] = (
    X_train["dropoff_latitude"] - X_train["pickup_latitude"]
).abs()
X_train["longitude_distance"] = (
    X_train["dropoff_longitude"] - X_train["pickup_longitude"]
).abs()

X_train["haversine_km"] = haversine_km(
    X_train["pickup_longitude"],
    X_train["pickup_latitude"],
    X_train["dropoff_longitude"],
    X_train["dropoff_latitude"],
)

X_train["mid_longitude"] = (
    X_train["pickup_longitude"] + X_train["dropoff_longitude"]
) / 2.0
X_train["mid_latitude"] = (
    X_train["pickup_latitude"] + X_train["dropoff_latitude"]
) / 2.0

X_train = X_train.drop(
    columns=[
        "key",
        "fare_amount",
        "pickup_datetime",
    ]
)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/datetimelike.py in _sub_datetimelike(self, other)
   1164         try:
-> 1165             self._assert_tzawareness_compat(other)
   1166         except TypeError as err:

/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/datetimes.py in _assert_tzawareness_compat(self, other)
    781             if other_tz is not None:
--> 782                 raise TypeError(
    783                     "Cannot compare tz-naive and tz-aware datetime-like objects."

TypeError: Cannot compare tz-naive and tz-aware datetime-like objects.

The above exception was the direct cause of the following exception:

TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/487789118.py in <cell line: 0>()
      2 dropoff_dt = extract_dropoff_datetime_from_key(X_train["key"])
      3 X_train["trip_duration_min"] = (
----> 4     (dropoff_dt - X_train["pickup_datetime"]).dt.total_seconds() / 60.0
      5 ).clip(lower=1.0, upper=240.0)
      6 

/usr/local/lib/python3.11/dist-packages/pandas/core/ops/common.py in new_method(self, other)
     74         other = item_from_zerodim(other)
     75 
---> 76         return method(self, other)
     77 
     78     return new_method

/usr/local/lib/python3.11/dist-packages/pandas/core/arraylike.py in __sub__(self, other)
    192     @unpack_zerodim_and_defer("__sub__")
    193     def __sub__(self, other):
--> 194         return self._arith_method(other, operator.sub)
    195 
    196     @unpack_zerodim_and_defer("__rsub__")

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in _arith_method(self, other, op)
   6133     def _arith_method(self, other, op):
   6134         self, other = self._align_for_op(other)
-> 6135         return base.IndexOpsMixin._arith_method(self, other, op)
   6136 
   6137     def _align_for_op(self, right, align_asobject: bool = False):

/usr/local/lib/python3.11/dist-packages/pandas/core/base.py in _arith_method(self, other, op)
   1380 
   1381         with np.errstate(all="ignore"):
-> 1382             result = ops.arithmetic_op(lvalues, rvalues, op)
   1383 
   1384         return self._construct_result(result, name=res_name)

/usr/local/lib/python3.11/dist-packages/pandas/core/ops/array_ops.py in arithmetic_op(left, right, op)
    271         # Timedelta/Timestamp and other custom scalars are included in the check
    272         # because numexpr will fail on it, see GH#31457
--> 273         res_values = op(left, right)
    274     else:
    275         # TODO we should handle EAs consistently and move this check before the if/else

/usr/local/lib/python3.11/dist-packages/pandas/core/ops/common.py in new_method(self, other)
     74         other = item_from_zerodim(other)
     75 
---> 76         return method(self, other)
     77 
     78     return new_method

/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/datetimelike.py in __sub__(self, other)
   1457         ):
   1458             # DatetimeIndex, ndarray[datetime64]
-> 1459             result = self._sub_datetime_arraylike(other)
   1460         elif isinstance(other_dtype, PeriodDtype):
   1461             # PeriodIndex

/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/datetimelike.py in _sub_datetime_arraylike(self, other)
   1154 
   1155         self, other = self._ensure_matching_resos(other)
-> 1156         return self._sub_datetimelike(other)
   1157 
   1158     @final

/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/datetimelike.py in _sub_datetimelike(self, other)
   1166         except TypeError as err:
   1167             new_message = str(err).replace("compare", "subtract")
-> 1168             raise type(err)(new_message) from err
   1169 
   1170         other_i8, o_mask = self._get_i8_values_and_mask(other)

TypeError: Cannot subtract tz-naive and tz-aware datetime-like objects.

## === cell 6
test_mask = (
    test_data["pickup_longitude"].between(-75, -72)
    & test_data["dropoff_longitude"].between(-75, -72)
    & test_data["pickup_latitude"].between(40, 42)
    & test_data["dropoff_latitude"].between(40, 42)
)

X_test = test_data.copy()
X_test.loc[
    ~test_mask,
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
] = np.nan

dropoff_dt_test = extract_dropoff_datetime_from_key(X_test["key"])
X_test["trip_duration_min"] = (
    (dropoff_dt_test - X_test["pickup_datetime"]).dt.total_seconds() / 60.0
).clip(lower=1.0, upper=240.0)

X_test["hour"] = X_test["pickup_datetime"].dt.hour
X_test["dayofweek"] = X_test["pickup_datetime"].dt.dayofweek
X_test["month"] = X_test["pickup_datetime"].dt.month
X_test["year"] = X_test["pickup_datetime"].dt.year

X_test["latitude_distance"] = (
    X_test["dropoff_latitude"] - X_test["pickup_latitude"]
).abs()
X_test["longitude_distance"] = (
    X_test["dropoff_longitude"] - X_test["pickup_longitude"]
).abs()

X_test["haversine_km"] = haversine_km(
    X_test["pickup_longitude"],
    X_test["pickup_latitude"],
    X_test["dropoff_longitude"],
    X_test["dropoff_latitude"],
)

X_test["mid_longitude"] = (
    X_test["pickup_longitude"] + X_test["dropoff_longitude"]
) / 2.0
X_test["mid_latitude"] = (X_test["pickup_latitude"] + X_test["dropoff_latitude"]) / 2.0

X_test = X_test.drop(columns=["key", "pickup_datetime"])

X_test = X_test[X_train.columns]

X_test = X_test.fillna(X_train.median(numeric_only=True))



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/datetimelike.py in _sub_datetimelike(self, other)
   1164         try:
-> 1165             self._assert_tzawareness_compat(other)
   1166         except TypeError as err:

/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/datetimes.py in _assert_tzawareness_compat(self, other)
    781             if other_tz is not None:
--> 782                 raise TypeError(
    783                     "Cannot compare tz-naive and tz-aware datetime-like objects."

TypeError: Cannot compare tz-naive and tz-aware datetime-like objects.

The above exception was the direct cause of the following exception:

TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3142652574.py in <cell line: 0>()
     15 dropoff_dt_test = extract_dropoff_datetime_from_key(X_test["key"])
     16 X_test["trip_duration_min"] = (
---> 17     (dropoff_dt_test - X_test["pickup_datetime"]).dt.total_seconds() / 60.0
     18 ).clip(lower=1.0, upper=240.0)
     19 

/usr/local/lib/python3.11/dist-packages/pandas/core/ops/common.py in new_method(self, other)
     74         other = item_from_zerodim(other)
     75 
---> 76         return method(self, other)
     77 
     78     return new_method

/usr/local/lib/python3.11/dist-packages/pandas/core/arraylike.py in __sub__(self, other)
    192     @unpack_zerodim_and_defer("__sub__")
    193     def __sub__(self, other):
--> 194         return self._arith_method(other, operator.sub)
    195 
    196     @unpack_zerodim_and_defer("__rsub__")

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in _arith_method(self, other, op)
   6133     def _arith_method(self, other, op):
   6134         self, other = self._align_for_op(other)
-> 6135         return base.IndexOpsMixin._arith_method(self, other, op)
   6136 
   6137     def _align_for_op(self, right, align_asobject: bool = False):

/usr/local/lib/python3.11/dist-packages/pandas/core/base.py in _arith_method(self, other, op)
   1380 
   1381         with np.errstate(all="ignore"):
-> 1382             result = ops.arithmetic_op(lvalues, rvalues, op)
   1383 
   1384         return self._construct_result(result, name=res_name)

/usr/local/lib/python3.11/dist-packages/pandas/core/ops/array_ops.py in arithmetic_op(left, right, op)
    271         # Timedelta/Timestamp and other custom scalars are included in the check
    272         # because numexpr will fail on it, see GH#31457
--> 273         res_values = op(left, right)
    274     else:
    275         # TODO we should handle EAs consistently and move this check before the if/else

/usr/local/lib/python3.11/dist-packages/pandas/core/ops/common.py in new_method(self, other)
     74         other = item_from_zerodim(other)
     75 
---> 76         return method(self, other)
     77 
     78     return new_method

/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/datetimelike.py in __sub__(self, other)
   1457         ):
   1458             # DatetimeIndex, ndarray[datetime64]
-> 1459             result = self._sub_datetime_arraylike(other)
   1460         elif isinstance(other_dtype, PeriodDtype):
   1461             # PeriodIndex

/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/datetimelike.py in _sub_datetime_arraylike(self, other)
   1154 
   1155         self, other = self._ensure_matching_resos(other)
-> 1156         return self._sub_datetimelike(other)
   1157 
   1158     @final

/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/datetimelike.py in _sub_datetimelike(self, other)
   1166         except TypeError as err:
   1167             new_message = str(err).replace("compare", "subtract")
-> 1168             raise type(err)(new_message) from err
   1169 
   1170         other_i8, o_mask = self._get_i8_values_and_mask(other)

TypeError: Cannot subtract tz-naive and tz-aware datetime-like objects.

## === cell 7
"""
from keras import models
from keras import layers
from keras import optimizers
from keras.layers import Dropout

model=models.Sequential()
model.add(layers.Dense(512,activation='relu',input_shape=(X_train.shape[1],)))
model.add(Dropout(0.2))
model.add(layers.Dense(512,activation='relu'))
model.add(Dropout(0.2))
model.add(layers.Dense(1))

rmsprop=optimizers.RMSprop(lr=0.001)

model.compile(optimizer=rmsprop,loss='mse',metrics=['mae'])

model.fit(X_train,Y_train,epochs=4,batch_size=512)

Y_pred=model.predict(X_test)
"""



## === cell 8
import xgboost as xgb

model = xgb.XGBRegressor(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
)
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

Y_pred = np.clip(Y_pred, 0, None)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3207830340.py in <cell line: 0>()
     12     tree_method="hist",
     13 )
---> 14 model.fit(X_train, Y_train)
     15 
     16 Y_pred = model.predict(X_test)

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in inner_f(*args, **kwargs)
    728             for k, arg in zip(sig.parameters, args):
    729                 kwargs[k] = arg
--> 730             return func(**kwargs)
    731 
    732         return inner_f

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in fit(self, X, y, sample_weight, base_margin, eval_set, eval_metric, early_stopping_rounds, verbose, xgb_model, sample_weight_eval_set, base_margin_eval_set, feature_weights, callbacks)
   1053         with config_context(verbosity=self.verbosity):
   1054             evals_result: TrainingCallback.EvalsLog = {}
-> 1055             train_dmatrix, evals = _wrap_evaluation_matrices(
   1056                 missing=self.missing,
   1057                 X=X,

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in _wrap_evaluation_matrices(missing, X, y, group, qid, sample_weight, base_margin, feature_weights, eval_set, sample_weight_eval_set, base_margin_eval_set, eval_group, eval_qid, create_dmatrix, enable_categorical, feature_types)
    519     """Convert array_like evaluation matrices into DMatrix.  Perform validation on the
    520     way."""
--> 521     train_dmatrix = create_dmatrix(
    522         data=X,
    523         label=y,

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in _create_dmatrix(self, ref, **kwargs)
    956         if _can_use_qdm(self.tree_method) and self.booster != "gblinear":
    957             try:
--> 958                 return QuantileDMatrix(
    959                     **kwargs, ref=ref, nthread=self.n_jobs, max_bin=self.max_bin
    960                 )

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in inner_f(*args, **kwargs)
    728             for k, arg in zip(sig.parameters, args):
    729                 kwargs[k] = arg
--> 730             return func(**kwargs)
    731 
    732         return inner_f

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in __init__(self, data, label, weight, base_margin, missing, silent, feature_names, feature_types, nthread, max_bin, ref, group, qid, label_lower_bound, label_upper_bound, feature_weights, enable_categorical, data_split_mode)
   1527                 )
   1528 
-> 1529         self._init(
   1530             data,
   1531             ref=ref,

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in _init(self, data, ref, enable_categorical, **meta)
   1586             ctypes.byref(handle),
   1587         )
-> 1588         it.reraise()
   1589         # delay check_call to throw intermediate exception first
   1590         _check_call(ret)

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in reraise(self)
    574             exc = self._exception
    575             self._exception = None
--> 576             raise exc  # pylint: disable=raising-bad-type
    577 
    578     def __del__(self) -> None:

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in _handle_exception(self, fn, dft_ret)
    555 
    556         try:
--> 557             return fn()
    558         except Exception as e:  # pylint: disable=broad-except
    559             # Defer the exception in order to return 0 and stop the iteration.

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in <lambda>()
    639 
    640         # pylint: disable=not-callable
--> 641         return self._handle_exception(lambda: self.next(input_data), 0)
    642 
    643     @abstractmethod

/usr/local/lib/python3.11/dist-packages/xgboost/data.py in next(self, input_data)
   1278             return 0
   1279         self.it += 1
-> 1280         input_data(**self.kwargs)
   1281         return 1
   1282 

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in inner_f(*args, **kwargs)
    728             for k, arg in zip(sig.parameters, args):
    729                 kwargs[k] = arg
--> 730             return func(**kwargs)
    731 
    732         return inner_f

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in input_data(data, feature_names, feature_types, **kwargs)
    622                 new, cat_codes, feature_names, feature_types = self._temporary_data
    623             else:
--> 624                 new, cat_codes, feature_names, feature_types = _proxy_transform(
    625                     data,
    626                     feature_names,

/usr/local/lib/python3.11/dist-packages/xgboost/data.py in _proxy_transform(data, feature_names, feature_types, enable_categorical)
   1313         data = pd.DataFrame(data)
   1314     if _is_pandas_df(data):
-> 1315         arr, feature_names, feature_types = _transform_pandas_df(
   1316             data, enable_categorical, feature_names, feature_types
   1317         )

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

ValueError: DataFrame.dtypes for data must be int, float, bool or category. When categorical type is supplied, The experimental DMatrix parameter`enable_categorical` must be set to `True`.  Invalid columns:key: object, pickup_datetime: datetime64[ns, UTC]

## === cell 9
submission = pd.DataFrame({"key": test_data["key"], "fare_amount": Y_pred})

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print(submission.head())
print("Columns:", submission.columns.tolist())
print("Rows:", len(submission))

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/379298291.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"key": test_data["key"], "fare_amount": Y_pred})
      2 
      3 submission_path = "submission.csv"
      4 submission.to_csv(submission_path, index=False)
      5 

NameError: name 'Y_pred' is not defined
