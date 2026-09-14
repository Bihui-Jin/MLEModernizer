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

4.07608

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 5.40188) has done: 'I fixed the file‑search logic so the script can locate the train and test CSVs wherever they reside, added a simple “hour of day” feature derived from the pickup timestamp to give the model a bit more predictive power, and updated the feature matrices accordingly. With these minimal changes the pipeline runs end‑to‑end and writes a proper `submission.csv` ready for Kaggle.'
- What this solution (achieved 5.52132) has done: 'The changes keep the exact feature formulas and model type, but streamline NumPy conversions in the distance calculation and lower the number of Gradient Boosting trees (still using the same GradientBoostingRegressor) so the training finishes well under the 600‑second limit.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


def find_file(filename, search_root="."):
    """Return first path that matches filename under search_root."""
    for root, _, files in os.walk(search_root):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(f"{filename} not found under {search_root}")


def haversine_distance(df):
    """Add a column 'distance_km' to df using haversine formula."""
    lon1 = np.radians(df["pickup_longitude"].astype(np.float64))
    lat1 = np.radians(df["pickup_latitude"].astype(np.float64))
    lon2 = np.radians(df["dropoff_longitude"].astype(np.float64))
    lat2 = np.radians(df["dropoff_latitude"].astype(np.float64))

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371.0 * c  # Earth radius in kilometers
    df["distance_km"] = km.astype(np.float32)
    return df




## === cell 1
train_path = find_file("train.csv")
test_path = find_file("test.csv")
sample_sub_path = find_file("sample_submission.csv")

dtypes = {
    "fare_amount": np.float32,
    "pickup_longitude": np.float32,
    "pickup_latitude": np.float32,
    "dropoff_longitude": np.float32,
    "dropoff_latitude": np.float32,
    "passenger_count": np.int16,
}
usecols = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
train_df = pd.read_csv(train_path, nrows=1_000_000, dtype=dtypes, usecols=usecols)
train_df = train_df.dropna(subset=["fare_amount"])

test_usecols = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
test_df = pd.read_csv(test_path, dtype=dtypes, usecols=test_usecols)




## === cell 2
train_df = haversine_distance(train_df)
train_df["hour"] = pd.to_datetime(train_df["pickup_datetime"]).dt.hour.astype(np.int16)
train_df["hour_sin"] = np.sin(2 * np.pi * train_df["hour"] / 24).astype(np.float32)
train_df["hour_cos"] = np.cos(2 * np.pi * train_df["hour"] / 24).astype(np.float32)

test_df = haversine_distance(test_df)
test_df["hour"] = pd.to_datetime(test_df["pickup_datetime"]).dt.hour.astype(np.int16)
test_df["hour_sin"] = np.sin(2 * np.pi * test_df["hour"] / 24).astype(np.float32)
test_df["hour_cos"] = np.cos(2 * np.pi * test_df["hour"] / 24).astype(np.float32)




## === cell 3
feature_cols = ["distance_km", "hour_sin", "hour_cos", "hour", "passenger_count"]
X = train_df[feature_cols].to_numpy()
y = np.log1p(train_df["fare_amount"].to_numpy())  # log‑transform target

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

imp = SimpleImputer(strategy="mean")
X_train = imp.fit_transform(X_train)
X_val = imp.transform(X_val)

regr = GradientBoostingRegressor(
    random_state=21,
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
)
regr.fit(X_train, y_train)

val_pred = np.expm1(regr.predict(X_val))  # inverse log‑transform
rmse = mean_squared_error(np.expm1(y_val), val_pred, squared=False)
print(f"Validation RMSE: {rmse:.4f}")




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2740114307.py in <cell line: 0>()
     17     subsample=0.8,
     18 )
---> 19 regr.fit(X_train, y_train)
     20 
     21 val_pred = np.expm1(regr.predict(X_val))  # inverse log‑transform

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_gb.py in fit(self, X, y, sample_weight, monitor)
    427         # trees use different types for X and y, checking them separately.
    428 
--> 429         X, y = self._validate_data(
    430             X, y, accept_sparse=["csr", "csc", "coo"], dtype=DTYPE, multi_output=True
    431         )

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

## === cell 4
X_test = test_df[feature_cols].to_numpy()
X_test = imp.transform(X_test)

test_pred = np.expm1(regr.predict(X_test))

submission = pd.DataFrame({"key": test_df["key"], "fare_amount": test_pred})
submission = submission[["key", "fare_amount"]]

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/1311472752.py in <cell line: 0>()
      2 X_test = imp.transform(X_test)
      3 
----> 4 test_pred = np.expm1(regr.predict(X_test))
      5 
      6 submission = pd.DataFrame({"key": test_df["key"], "fare_amount": test_pred})

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_gb.py in predict(self, X)
   1800         )
   1801         # In regression we can directly return the raw value from the trees.
-> 1802         return self._raw_predict(X).ravel()
   1803 
   1804     def staged_predict(self, X):

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_gb.py in _raw_predict(self, X)
    685     def _raw_predict(self, X):
    686         """Return the sum of the trees raw predictions (+ init estimator)."""
--> 687         raw_predictions = self._raw_predict_init(X)
    688         predict_stages(self.estimators_, X, self.learning_rate, raw_predictions)
    689         return raw_predictions

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_gb.py in _raw_predict_init(self, X)
    671     def _raw_predict_init(self, X):
    672         """Check input and compute raw predictions of the init estimator."""
--> 673         self._check_initialized()
    674         X = self.estimators_[0, 0]._validate_X_predict(X, check_input=True)
    675         if self.init_ == "zero":

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_gb.py in _check_initialized(self)
    380     def _check_initialized(self):
    381         """Check that the estimator is initialized, raising an error if not."""
--> 382         check_is_fitted(self)
    383 
    384     def fit(self, X, y, sample_weight=None, monitor=None):

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_is_fitted(estimator, attributes, msg, all_or_any)
   1388 
   1389     if not fitted:
-> 1390         raise NotFittedError(msg % {"name": type(estimator).__name__})
   1391 
   1392 

NotFittedError: This GradientBoostingRegressor instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.
