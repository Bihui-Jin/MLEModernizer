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

4.40565

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 605.61576) has done: 'I clean the imports that cause the protobuf error, drop the target column from the held‑out test split so the scaler sees matching features, add a TensorFlow import and rewrite the custom RMSE metric using `tf.math.sqrt`. The submission writer is also tweaked to flatten predictions. These minimal fixes let the notebook run end‑to‑end, produce a proper `.csv` file and move the RMSE toward the target score.'
- What this solution (achieved 15.19817) has done: 'I replace the TensorFlow import and the custom rmse implementation with a version that uses the Keras backend (which is already available). This removes the dependency on the missing `tensorflow` package that caused the protobuf error, fixes the metric definition, and lets the notebook run end‑to‑end while preserving the existing model and training workflow.'
- What this solution (achieved 15.2808) has done: 'The fix adds an environment flag to avoid the protobuf import error, and rewrites the custom RMSE metric using `K.pow(..., 0.5)` (since `K.sqrt` is unavailable). These changes let the notebook run through training, generate a valid submission CSV, and move the RMSE toward the target.'
- What this solution (achieved 15.26611) has done: 'I fixed the RMSE metric implementation, which was causing an AttributeError because `keras.backend` lacks a `pow` function. By importing TensorFlow and using `tf.sqrt` on the mean squared error, the model can compile and train successfully, allowing the script to generate a proper CSV submission.'
- What this solution (achieved 205.25819) has done: 'I fix the custom RMSE metric, which currently uses a backend function that isn’t available in the installed Keras version. By replacing it with TensorFlow‑only operations (`tf.reduce_mean` and `tf.sqrt`), the model can compile and train, allowing the script to run end‑to‑end and produce a valid submission CSV. This change is score‑neutral but removes the runtime error that prevented any training.'
- What this solution (achieved 15.23752) has done: 'I fixed the protobuf import error by removing the unnecessary TensorFlow import and rewrote the custom RMSE metric using Keras backend operations. I also toned down the L1 activity regularizer to prevent over‑regularization, which should improve the model’s RMSE and bring the score closer to the target while keeping the original workflow intact. The script now runs end‑to‑end and writes a proper `.csv` submission file.'
- What this solution (achieved 97.61049) has done: 'I added an explicit TensorFlow import (which works with the installed tf_keras package) after setting the protobuf environment variable, and rewrote the custom rmse metric to use tf.sqrt instead of the missing K.sqrt. This resolves the AttributeError, lets the model compile and train, and ensures the script runs end‑to‑end producing a valid .csv submission.'
- What this solution (achieved 5.43399) has done: 'I fix the import errors by switching to `tf.keras`, define all constants before they are used, load and preprocess the data (including a simple haversine distance feature), correctly split and scale the features, build and train the model using the existing architecture, and finally generate a proper `.csv` submission file with the required columns. These changes resolve the runtime failures, ensure a valid CSV output, and add a lightweight feature that should modestly improve the RMSE toward the target without altering the core model logic.'
- What this solution (achieved 15.26554) has done: 'I fixed the protobuf import error by switching to the standalone `keras` package instead of `tf_keras`, corrected the missing imports (EarlyStopping, Sequential, etc.), and adjusted the data paths so they resolve in the Kaggle environment. All variables are now defined in the proper order, the model compiles, trains, and a correctly‑formatted `.csv` submission is written.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from keras import optimizers, regularizers, backend as K

TRAIN_PATH = "/kaggle/input/train.csv"
TEST_PATH = "/kaggle/input/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 256
EPOCHS = 50  # early stopping will limit actual epochs
LEARNING_RATE = 0.001
DATASET_SIZE = 200_000  # subset for quick iteration
RANDOM_STATE = 42




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def rmse(y_true, y_pred):
    """Root‑mean‑squared error metric compatible with Keras."""
    return K.pow(K.mean(K.square(y_pred - y_true), axis=-1), 0.5)




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

train_df = pd.read_csv(
    TRAIN_PATH,
    nrows=DATASET_SIZE,
    dtype=datatypes,
)

test_df = pd.read_csv(
    TEST_PATH,
    dtype={
        "key": "str",
        "pickup_datetime": "str",
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "uint8",
    },
)


def haversine_distance(row):
    """Calculate haversine distance in kilometers between pickup and dropoff."""
    R = 6371.0
    lat1 = np.radians(row["pickup_latitude"])
    lon1 = np.radians(row["pickup_longitude"])
    lat2 = np.radians(row["dropoff_latitude"])
    lon2 = np.radians(row["dropoff_longitude"])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


for df in (train_df, test_df):
    df["distance"] = df.apply(haversine_distance, axis=1)


def add_time_features(df):
    dt = pd.to_datetime(df["pickup_datetime"])
    df["hour"] = dt.dt.hour.astype(np.float32)
    df["dayofweek"] = dt.dt.dayofweek.astype(np.float32)
    df["month"] = dt.dt.month.astype(np.float32)


add_time_features(train_df)
add_time_features(test_df)

train_df.dropna(
    subset=[
        "fare_amount",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
    inplace=True,
)

feature_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "distance",
    "hour",
    "dayofweek",
    "month",
]

X = train_df[feature_cols].values.astype(np.float32)
y = train_df["fare_amount"].values.astype(np.float32)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

scaler = preprocessing.StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
test_scaled = scaler.transform(test_df[feature_cols].values.astype(np.float32))



## === cell 3
from sklearn.ensemble import GradientBoostingRegressor

y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)

gbr = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    random_state=RANDOM_STATE,
)

gbr.fit(X_train_scaled, y_train_log)

val_pred_log = gbr.predict(X_val_scaled)
val_rmse = np.sqrt(np.mean((np.expm1(val_pred_log) - y_val) ** 2))
print(f"Validation RMSE (log‑model): {val_rmse:.4f}")

print(f"Dataset size: {DATASET_SIZE}")
print(f"Features used: {feature_cols}")



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/1193711298.py in <cell line: 0>()
     13 )
     14 
---> 15 gbr.fit(X_train_scaled, y_train_log)
     16 
     17 # Optional: evaluate on validation set (for debugging)

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
test_pred_log = gbr.predict(test_scaled)
test_preds = np.expm1(test_pred_log)

submission = pd.DataFrame({"key": test_df["key"], "fare_amount": test_preds})
submission.to_csv(SUBMISSION_NAME, index=False)
print(f"Submission written to {SUBMISSION_NAME}")

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_55/141228231.py in <cell line: 0>()
      1 # Predict on the test set and reverse the log transformation
----> 2 test_pred_log = gbr.predict(test_scaled)
      3 test_preds = np.expm1(test_pred_log)
      4 
      5 submission = pd.DataFrame({"key": test_df["key"], "fare_amount": test_preds})

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
