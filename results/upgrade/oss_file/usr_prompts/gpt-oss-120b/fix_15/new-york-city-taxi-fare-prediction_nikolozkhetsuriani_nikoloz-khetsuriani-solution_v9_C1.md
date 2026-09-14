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

No external packages required in the script and installed.

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

4.330098025036998

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 8.22378) has done: 'I remove the stray syntax error, ensure missing values in the test set are filled, clip negative predictions, and modestly improve the RandomForest (more trees) while keeping the original feature set and workflow. The cells are renumbered sequentially so the script runs end‑to‑end and outputs a valid `submission.csv`.'
- What this solution (achieved 7.2324) has done: 'I increase the training sample size (from 100 k to 500 k rows) and make the RandomForest a bit stronger (more trees and a deeper depth). These minimal tweaks keep the original workflow and features intact while giving the model more data and capacity, which should lower the validation RMSE and move the score closer to the target. No other logic is altered, and the script still writes a correct `submission.csv`.'
- What this solution (achieved 5.3879) has done: 'The changes speed up the script by parsing dates while reading the CSV (removing a costly second conversion), keeping the same feature engineering, and halving the number of trees in the RandomForest (still the same model type) to bring training time well under the 600‑second limit while preserving the overall logic and accuracy.'
- What this solution (achieved 5.84504) has done: 'The change reduces the RandomForest size by lowering the number of estimators from 300 to 150, which cuts training time dramatically while keeping the same model type, features, and depth, so the prediction logic and accuracy remain essentially unchanged. No other parts of the pipeline are altered.'
- What this solution (achieved 5.84504) has done: 'The changes reduce the amount of data read (1 M rows instead of 2 M) and let the RandomForest use all available CPU cores, which together keep the training time well under the 600 s limit. Holiday feature creation is rewritten with simple integer codes instead of a MultiIndex, removing an unnecessary overhead while giving identical results.'
- What this solution (achieved 5.91955) has done: 'The update keeps all preprocessing and feature engineering unchanged and only trims the RandomForest size to finish within the 600‑second limit. Reducing `n_estimators` from 200 to 100 cuts training time roughly in half while preserving the same model type and feature set, so the prediction logic and evaluation remain identical.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

np.random.seed(42)



## === cell 1
train_path = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
dtype_map = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
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
data = pd.read_csv(
    train_path,
    nrows=1_500_000,
    dtype=dtype_map,
    parse_dates=["pickup_datetime"],
    usecols=usecols,
    low_memory=False,
)



## === cell 2
mean_dropoff_longitude = data["dropoff_longitude"].mean()
mean_dropoff_latitude = data["dropoff_latitude"].mean()
data["dropoff_longitude"] = data["dropoff_longitude"].fillna(mean_dropoff_longitude)
data["dropoff_latitude"] = data["dropoff_latitude"].fillna(mean_dropoff_latitude)



## === cell 3
data = data[data["fare_amount"] <= 500]
data = data[(data["passenger_count"] >= 1) & (data["passenger_count"] <= 7)]

ny_lat_min, ny_lat_max = 40.4774, 40.9176
ny_lon_min, ny_lon_max = -74.2591, -73.7004
data = data[
    (data["pickup_latitude"].between(ny_lat_min, ny_lat_max))
    & (data["pickup_longitude"].between(ny_lon_min, ny_lon_max))
    & (data["dropoff_latitude"].between(ny_lat_min, ny_lat_max))
    & (data["dropoff_longitude"].between(ny_lon_min, ny_lon_max))
]



## === cell 4
test_path = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"
testData = pd.read_csv(
    test_path,
    parse_dates=["pickup_datetime"],
    dtype={
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "int8",
    },
    low_memory=False,
)



## === cell 5
testData["dropoff_longitude"] = testData["dropoff_longitude"].fillna(
    mean_dropoff_longitude
)
testData["dropoff_latitude"] = testData["dropoff_latitude"].fillna(
    mean_dropoff_latitude
)




## === cell 6
def haversine_vectorized(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c  # Earth radius in km


data["haversine_distance"] = haversine_vectorized(
    data["pickup_latitude"],
    data["pickup_longitude"],
    data["dropoff_latitude"],
    data["dropoff_longitude"],
)

testData["haversine_distance"] = haversine_vectorized(
    testData["pickup_latitude"],
    testData["pickup_longitude"],
    testData["dropoff_latitude"],
    testData["dropoff_longitude"],
)

data["log_haversine"] = np.log1p(data["haversine_distance"])
testData["log_haversine"] = np.log1p(testData["haversine_distance"])

data = data[(data["haversine_distance"] > 0.05) & (data["haversine_distance"] < 500)]



## === cell 7
data["year"] = data["pickup_datetime"].dt.year
data["month"] = data["pickup_datetime"].dt.month
data["hour_of_day"] = data["pickup_datetime"].dt.hour
data["day_of_week"] = data["pickup_datetime"].dt.dayofweek
data["is_night_time"] = (
    (data["hour_of_day"] >= 21) | (data["hour_of_day"] < 6)
).astype(int)

testData["year"] = testData["pickup_datetime"].dt.year
testData["month"] = testData["pickup_datetime"].dt.month
testData["hour_of_day"] = testData["pickup_datetime"].dt.hour
testData["day_of_week"] = testData["pickup_datetime"].dt.dayofweek
testData["is_night_time"] = (
    (testData["hour_of_day"] >= 21) | (testData["hour_of_day"] < 6)
).astype(int)



## === cell 8
public_holidays = [
    (1, 1),
    (1, 15),
    (2, 12),
    (2, 19),
    (5, 27),
    (6, 19),
    (7, 4),
    (9, 2),
    (10, 14),
    (11, 5),
    (11, 11),
    (11, 28),
    (12, 25),
]
holiday_codes = {m * 100 + d for m, d in public_holidays}

data_md = data["pickup_datetime"].dt.month * 100 + data["pickup_datetime"].dt.day
data["is_public_holiday"] = data_md.isin(holiday_codes).astype(int)

test_md = (
    testData["pickup_datetime"].dt.month * 100 + testData["pickup_datetime"].dt.day
)
testData["is_public_holiday"] = test_md.isin(holiday_codes).astype(int)



## === cell 9
airports = [(-73.7789, 40.6413), (-73.8740, 40.7769), (-74.1811, 40.6925)]

airport_lons = np.array([lon for lon, lat in airports])
airport_lats = np.array([lat for lon, lat in airports])


def near_airport_vectorized(lat_series, lon_series):
    lat1 = np.radians(lat_series.to_numpy()[:, None])  # (n,1)
    lon1 = np.radians(lon_series.to_numpy()[:, None])  # (n,1)
    lat2 = np.radians(airport_lats)  # (3,)
    lon2 = np.radians(airport_lons)  # (3,)

    dlon = lon2 - lon1  # broadcast to (n,3)
    dlat = lat2 - lat1
    a = np.sin(dlon / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlat / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    distances = 6371 * c  # (n,3) km
    min_dist = distances.min(axis=1)
    return (min_dist < 5).astype(int)


data["pickup_near_airport"] = near_airport_vectorized(
    data["pickup_latitude"], data["pickup_longitude"]
)
data["dropoff_near_airport"] = near_airport_vectorized(
    data["dropoff_latitude"], data["dropoff_longitude"]
)

testData["pickup_near_airport"] = near_airport_vectorized(
    testData["pickup_latitude"], testData["pickup_longitude"]
)
testData["dropoff_near_airport"] = near_airport_vectorized(
    testData["dropoff_latitude"], testData["dropoff_longitude"]
)



## === cell 10
data["abs_lat_diff"] = (data["dropoff_latitude"] - data["pickup_latitude"]).abs()
data["abs_lon_diff"] = (data["dropoff_longitude"] - data["pickup_longitude"]).abs()

testData["abs_lat_diff"] = (
    testData["dropoff_latitude"] - testData["pickup_latitude"]
).abs()
testData["abs_lon_diff"] = (
    testData["dropoff_longitude"] - testData["pickup_longitude"]
).abs()



## === cell 11
features = [
    "passenger_count",
    "haversine_distance",
    "log_haversine",
    "pickup_near_airport",
    "dropoff_near_airport",
    "hour_of_day",
    "day_of_week",
    "year",
    "month",
    "is_public_holiday",
    "is_night_time",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
    "abs_lat_diff",
    "abs_lon_diff",
]

X = data[features].astype("float32")
y_log = np.log1p(data["fare_amount"].astype("float32"))  # log‑transform target

X_train, X_val, y_train_log, y_val_log = train_test_split(
    X, y_log, test_size=0.25, random_state=42
)



## === cell 12
rf_model = RandomForestRegressor(
    n_estimators=200,  # modest increase for better capacity
    max_depth=30,
    random_state=42,
    n_jobs=-1,
)
rf_model.fit(X_train.values, y_train_log.values)

y_val_pred_log = rf_model.predict(X_val.values)
y_val_pred = np.expm1(y_val_pred_log)  # back‑transform
rmse = mean_squared_error(y_val.values, y_val_pred, squared=False)
print("Validation RMSE:", rmse)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2059567975.py in <cell line: 0>()
      5     n_jobs=-1,
      6 )
----> 7 rf_model.fit(X_train.values, y_train_log.values)
      8 
      9 y_val_pred_log = rf_model.predict(X_val.values)

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_forest.py in fit(self, X, y, sample_weight)
    343         if issparse(y):
    344             raise ValueError("sparse multilabel-indicator for y is not supported.")
--> 345         X, y = self._validate_data(
    346             X, y, multi_output=True, accept_sparse="csc", dtype=DTYPE
    347         )

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    582                 y = check_array(y, input_name="y", **check_y_params)
    583             else:
--> 584                 X, y = check_X_y(X, y, **check_params)
    585             out = X, y
    586 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_X_y(X, y, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, estimator)
   1120     )
   1121 
-> 1122     y = _check_y(y, multi_output=multi_output, y_numeric=y_numeric, estimator=estimator)
   1123 
   1124     check_consistent_length(X, y)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in _check_y(y, multi_output, y_numeric, estimator)
   1130     """Isolated part of check_X_y dedicated to y validation"""
   1131     if multi_output:
-> 1132         y = check_array(
   1133             y,
   1134             accept_sparse="csr",

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)
    919 
    920         if force_all_finite:
--> 921             _assert_all_finite(
    922                 array,
    923                 input_name=input_name,

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in _assert_all_finite(X, allow_nan, msg_dtype, estimator_name, input_name)
    159                 "#estimators-that-handle-nan-values"
    160             )
--> 161         raise ValueError(msg_err)
    162 
    163 

ValueError: Input y contains NaN.

## === cell 13
X_test = testData[features].astype("float32")
test_pred_log = rf_model.predict(X_test.values)
test_pred = np.expm1(test_pred_log)  # back‑transform

test_pred = np.clip(test_pred, 0, None)

submission = pd.DataFrame({"key": testData["key"], "fare_amount": test_pred})
submission.to_csv("submission.csv", index=False)
print("Submission file 'submission.csv' written.")

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/1359284449.py in <cell line: 0>()
      1 X_test = testData[features].astype("float32")
----> 2 test_pred_log = rf_model.predict(X_test.values)
      3 test_pred = np.expm1(test_pred_log)  # back‑transform
      4 
      5 test_pred = np.clip(test_pred, 0, None)

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_forest.py in predict(self, X)
    977             The predicted values.
    978         """
--> 979         check_is_fitted(self)
    980         # Check data
    981         X = self._validate_X_predict(X)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_is_fitted(estimator, attributes, msg, all_or_any)
   1388 
   1389     if not fitted:
-> 1390         raise NotFittedError(msg % {"name": type(estimator).__name__})
   1391 
   1392 

NotFittedError: This RandomForestRegressor instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.
