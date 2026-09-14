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
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
tf_keras==2.18.0

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

4.70167

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 15.21729) has done: 'Implemented fixes to resolve runtime errors and improve model performance:
- Replaced URL‑based mask loading with a safe fallback that simply returns the dataframe.
- Corrected optimizer call (`optimizers.Adam`) and its argument name.
- Added graceful handling for optional visualisation imports.
- Ensured non‑numeric columns (`key`) are removed before scaling.
- Fixed the always‑true `late_night` logic.
- Adjusted cell ordering and naming to be sequential and functional.'
- What this solution (achieved 65.52696) has done: 'The script is updated to fix the import errors (using TensorFlow Keras which provides the missing backend functions), correct the `rmse_metric`, and keep the informative **passenger_count** feature instead of dropping it. These changes resolve the runtime failures, allow the model to train and evaluate correctly, and should lower the RMSE toward the target value.'
- What this solution (achieved 15.17757) has done: 'Implemented a fix for the TensorFlow/Keras import issue by switching to the standalone Keras package, which avoids the protobuf‑related `MessageFactory` error. This change restores model building, training, and evaluation without altering the core architecture or training logic. All other cells are kept unchanged, preserving feature engineering and submission generation.'
- What this solution (achieved 471.69692) has done: 'The fix adds TensorFlow to define a proper RMSE metric (using `tf.sqrt`), updates the metric implementation, and imports TensorFlow. It also increases training epochs from 100 to 200 to give the model more learning capacity, which should lower the RMSE toward the target while keeping the original architecture unchanged.'
- What this solution (achieved 15.57248) has done: 'The fix adds the missing preprocessing and submission helper functions, switches to the standalone keras package to avoid the protobuf import error, and defines a simple `clean` routine. These changes resolve the runtime NameErrors, allow the model to train and generate predictions, and ensure a correctly‑named `.csv` submission file is written.'
- What this solution (achieved 11.53297) has done: 'I reorder the imports to load TensorFlow before Keras (preventing the protobuf MessageFactory error) and replace the custom RMSE metric with a TensorFlow‑based implementation that uses `tf.sqrt`. This fixes the runtime AttributeError, enables training to complete, and provides a proper RMSE metric so the model’s evaluation moves closer to the target score. The core model architecture and training procedure remain unchanged.'
- What this solution (achieved 12.49312) has done: 'The fix switches to TensorFlow’s bundled Keras to avoid protobuf import errors, corrects the test‑file reading by excluding the non‑existent `fare_amount` column, and slightly enlarges the training sample (to 200 k rows) so the model can learn better features. These minimal changes restore the pipeline, produce a proper `.csv` submission, and are expected to lower the RMSE toward the target.'
- What this solution (achieved 12.21387) has done: 'I increase the training sample size, remove the strong L1 regularizer that was causing under‑fitting, and keep the rest of the pipeline unchanged. These minimal tweaks let the model learn from more data and train more freely, which should lower the RMSE toward the target while still producing a correct `.csv` submission.'
- What this solution (achieved 28.21375) has done: 'Implemented a safe protobuf setting before importing TensorFlow to eliminate the `MessageFactory` error, added `os` import, and changed the first Dense layer activation from `linear` to `relu` to improve model learning. Increased training epochs to 300 to give the network more opportunity to converge, while preserving the overall architecture and workflow. These adjustments fix the runtime crash and are expected to bring the RMSE closer to the target score.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

try:
    import tensorflow as tf
except Exception as e:
    tf = None  # TensorFlow is unavailable; we will use scikit‑learn instead.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

TRAIN_PATH = "../input/train.csv"
TEST_PATH = "../input/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 256
EPOCHS = 300  # retained for compatibility; not used with scikit‑learn
LEARNING_RATE = 0.001
DATASET_SIZE = 500000  # larger sample for better learning




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning – drop rows with missing values."""
    return df.dropna().reset_index(drop=True)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract hour, day of week and month from pickup_datetime."""
    df = df.copy()
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["hour"] = df["pickup_datetime"].dt.hour.astype(np.float32)
    df["dayofweek"] = df["pickup_datetime"].dt.dayofweek.astype(np.float32)
    df["month"] = df["pickup_datetime"].dt.month.astype(np.float32)
    return df


def add_coordinate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple coordinate differences."""
    df = df.copy()
    df["diff_longitude"] = df["dropoff_longitude"] - df["pickup_longitude"]
    df["diff_latitude"] = df["dropoff_latitude"] - df["pickup_latitude"]
    return df


def add_distances_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add haversine distance (in km) between pickup and dropoff points."""
    df = df.copy()
    lon1 = np.radians(df["pickup_longitude"])
    lat1 = np.radians(df["pickup_latitude"])
    lon2 = np.radians(df["dropoff_longitude"])
    lat2 = np.radians(df["dropoff_latitude"])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    earth_radius_km = 6371.0
    df["haversine_km"] = (earth_radius_km * c).astype(np.float32)
    return df


def output_submission(
    df_input: pd.DataFrame,
    preds: np.ndarray,
    key_col: str,
    target_col: str,
    filename: str,
):
    """Create submission CSV with required columns."""
    pred_series = pd.Series(preds.ravel(), name=target_col)
    submission = pd.DataFrame({key_col: df_input[key_col], target_col: pred_series})
    submission.to_csv(filename, index=False)




## === cell 2
datatypes = {
    "key": "str",
    "fare_amount": "float32",
    "pickup_datetime": "str",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}
trainKaggle = pd.read_csv(
    TRAIN_PATH,
    nrows=DATASET_SIZE,
    dtype=datatypes,
    usecols=[
        "fare_amount",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
        "key",
    ],
)

test_dtype = {k: v for k, v in datatypes.items() if k != "fare_amount"}
testKaggle = pd.read_csv(TEST_PATH, dtype=test_dtype)




## === cell 3
train_df, test_df = train_test_split(trainKaggle, test_size=0.50, random_state=1)
test_df = test_df[:10000]  # keep a manageable size for quick testing




## === cell 4
print("trainKaggle size %d" % len(trainKaggle))
print("train_df size %d" % len(train_df))
print("test_df size %d" % len(test_df))




## === cell 5
train_df = clean(train_df)
test_df = clean(test_df)




## === cell 6
print("Adding time features")
train_df = add_time_features(train_df)
test_df = add_time_features(test_df)
testKaggle = add_time_features(testKaggle)




## === cell 7
print("Adding coordinate differences")
train_df = add_coordinate_features(train_df)
test_df = add_coordinate_features(test_df)
testKaggle = add_coordinate_features(testKaggle)




## === cell 8
print("Adding distance features")
train_df = add_distances_features(train_df)
test_df = add_distances_features(test_df)
testKaggle = add_distances_features(testKaggle)




## === cell 9
dropped_columns = ["pickup_datetime", "key"]
train_df = train_df.drop(columns=dropped_columns)
test_df = test_df.drop(columns=dropped_columns)
testKaggle_clean = testKaggle.drop(columns=dropped_columns)  # keep key for submission
print("Columns dropped; remaining features:", train_df.columns.tolist())




## === cell 10
train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=1)

train_labels = np.log1p(train_df["fare_amount"].values)
validation_labels = np.log1p(validation_df["fare_amount"].values)
test_labels = np.log1p(test_df["fare_amount"].values)

train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)
test_df = test_df.drop(["fare_amount"], axis=1)

print("Prepared log‑transformed labels and cleaned feature sets.")




## === cell 11
scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_df)
testKaggle_scaled = scaler.transform(testKaggle_clean)




## === cell 12
def rmse_numpy(y_true, y_pred):
    """Root Mean Squared Error using NumPy."""
    return np.sqrt(np.mean((y_pred - y_true) ** 2))




## === cell 13
gbr = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    random_state=42,
)
gbr.fit(train_df_scaled, train_labels)




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/3144766031.py in <cell line: 0>()
      8 )
      9 # Fit on the log‑transformed target.
---> 10 gbr.fit(train_df_scaled, train_labels)
     11 
     12 

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

## === cell 14
print("Model training completed with GradientBoostingRegressor.")




## === cell 15
train_pred_log = gbr.predict(train_df_scaled)
val_pred_log = gbr.predict(validation_df_scaled)
test_pred_log = gbr.predict(test_scaled)

train_rmse_log = rmse_numpy(train_labels, train_pred_log)
val_rmse_log = rmse_numpy(validation_labels, val_pred_log)
test_rmse_log = rmse_numpy(test_labels, test_pred_log)

print("Train RMSE (log space):", train_rmse_log)
print("Validation RMSE (log space):", val_rmse_log)
print("Test RMSE (log space):", test_rmse_log)

train_pred = np.expm1(train_pred_log)
val_pred = np.expm1(val_pred_log)
test_pred = np.expm1(test_pred_log)

train_rmse_orig = np.sqrt(np.mean((train_pred - np.expm1(train_labels)) ** 2))
val_rmse_orig = np.sqrt(np.mean((val_pred - np.expm1(validation_labels)) ** 2))
test_rmse_orig = np.sqrt(np.mean((test_pred - np.expm1(test_labels)) ** 2))

print("Train RMSE (original scale):", train_rmse_orig)
print("Validation RMSE (original scale):", val_rmse_orig)
print("Test RMSE (original scale):", test_rmse_orig)




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_55/2801139641.py in <cell line: 0>()
      1 # Evaluate RMSE on train / validation / test using the NumPy helper.
----> 2 train_pred_log = gbr.predict(train_df_scaled)
      3 val_pred_log = gbr.predict(validation_df_scaled)
      4 test_pred_log = gbr.predict(test_scaled)
      5 

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

## === cell 16
predictionKaggle_log = gbr.predict(testKaggle_scaled)
predictionKaggle = np.expm1(predictionKaggle_log)  # revert log transform
output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_55/3724006215.py in <cell line: 0>()
----> 1 predictionKaggle_log = gbr.predict(testKaggle_scaled)
      2 predictionKaggle = np.expm1(predictionKaggle_log)  # revert log transform
      3 output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)

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
