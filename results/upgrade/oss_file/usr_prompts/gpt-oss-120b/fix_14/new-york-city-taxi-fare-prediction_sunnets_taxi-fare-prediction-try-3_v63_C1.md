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

5.00327

# 6. Current score

10.02952

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 249.91784) has done: 'Implemented fixes:
- Reworked `remove_datapoints_from_water` to skip external image loading and simply return the dataframe.
- Consolidated all utility functions into the first cell.
- Switched to TensorFlow Keras imports to avoid protobuf errors.
- Corrected optimizer creation (`optimizers.Adam`) and removed the unsupported `'accuracy'` metric.
- Wrapped optional visualisation imports in a safe try/except.
- Comment‑out stray analysis cells that referenced undefined variables.
- Ensured the submission writer creates a proper CSV with the required columns.'
- What this solution (achieved 243.68857) has done: 'Implemented three key fixes: (1) set the protobuf implementation to the pure‑Python version before any TensorFlow import to resolve the `MessageFactory` import error; (2) centered the target variable by subtracting the training‑set mean, which stabilises training and greatly improves RMSE; (3) added the mean back to predictions before writing the submission file. These changes keep the original model architecture and training flow while ensuring a valid CSV output and a score much closer to the target.'
- What this solution (achieved 10.09831) has done: 'The script failed due to importing `tensorflow.keras`, which isn’t compatible with the available `tf_keras` package and caused a protobuf error. Switching to the standalone `keras` implementation resolves the import issue while keeping the original model architecture and training logic unchanged, allowing the pipeline to run and produce a valid CSV submission.'
- What this solution (achieved 266.73231) has done: 'I fix the missing square‑root operation by defining the RMSE metric with TensorFlow instead of the unavailable Keras backend function, and I replace the metric list in the model compilation with a TensorFlow RMSE metric. I also increase the training epochs slightly (to give the model a chance to improve) while keeping the overall architecture unchanged. These changes resolve the runtime errors and are expected to lower the validation RMSE toward the target score.'
- What this solution (achieved 534.02473) has done: 'Implemented two key fixes to improve model performance and reduce RMSE:  
1. Corrected the `late_night` feature logic so it no longer flags every record as late‑night, providing more meaningful temporal features.  
2. Added a geographic `haversine` distance feature, which captures true spherical distance between pickup and drop‑off points and is highly predictive of fare amount. These changes preserve the original model architecture while enhancing feature quality, helping move the validation score toward the target.'
- What this solution (achieved 115.55755) has done: 'I fix the protobuf import error (already handled), reduce the training epochs to avoid over‑fitting, add a reproducible random seed, and introduce a simple mean‑baseline fallback: after evaluating the model on the validation set we compute the RMSE of predicting the overall training mean. If this baseline is better (lower) than the model’s validation RMSE, we use the mean prediction for the test and Kaggle submissions. This keeps the original model logic untouched while providing a score much closer to the target.'
- What this solution (achieved 10.04237) has done: 'I replace the Keras imports with TensorFlow‑Keras to avoid the protobuf error and redefine the custom RMSE metric using TensorFlow operations (since the backend lacks a sqrt function). These minimal changes fix the runtime failures, allow the model to train, and ensure a valid CSV submission is written, moving the solution toward the target score.'
- What this solution (achieved 5.60283) has done: 'The fix adds a safe protobuf setting before TensorFlow is imported, loads and preprocesses the data (datetime features, haversine distance, scaling), defines the missing train/validation variables, centers the target, and writes a proper CSV submission with the required columns. The core neural‑network architecture is kept unchanged, only the input dimension is set from the new feature matrix, and the mean added back to predictions so the model’s output matches the original fare scale.'
- What this solution (achieved 10.02601) has done: 'The fix switches to the standalone Keras library with the NumPy backend (avoiding the protobuf TensorFlow issue) and adds the raw latitude/longitude coordinates to the feature set for a modest predictive boost. The custom RMSE metric is rewritten using Keras’s backend, the random seed is set via Keras, and the environment variables are configured before any imports. All other logic—including data handling, scaling, target centering, model architecture, and CSV output—remains unchanged, ensuring a valid submission while moving the RMSE closer to the target.'
- What this solution (achieved 5.57176) has done: 'The fix switches to TensorFlow’s Keras backend (removing the NumPy backend that prevented training) and updates imports accordingly. This enables `model.fit` to run, produces a proper CSV submission, and restores the original modeling logic, moving the RMSE toward the target score.'
- What this solution (achieved 10.02952) has done: 'The fix switches from the unavailable `tensorflow` package to the installed standalone `keras` library, sets the random seed via `keras.utils.set_random_seed`, and updates the imports accordingly. It also expands the training sample size and number of epochs modestly to close the gap to the target RMSE while preserving the original model architecture and preprocessing logic.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras import optimizers
from keras import backend as K
from keras.utils import set_random_seed

set_random_seed(42)

TRAIN_PATH = "../input/train.csv"
TEST_PATH = "../input/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 256
EPOCHS = 150  # increased epochs for better convergence
LEARNING_RATE = 0.001
DATASET_SIZE = 500000  # larger sample for improved training




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
df = pd.read_csv(TRAIN_PATH, nrows=DATASET_SIZE)
df = df.dropna().reset_index(drop=True)


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
df["hour"] = df["pickup_datetime"].dt.hour
df["dayofweek"] = df["pickup_datetime"].dt.dayofweek
df["month"] = df["pickup_datetime"].dt.month
df["distance"] = haversine(
    df["pickup_latitude"],
    df["pickup_longitude"],
    df["dropoff_latitude"],
    df["dropoff_longitude"],
)

feature_cols = [
    "passenger_count",
    "distance",
    "hour",
    "dayofweek",
    "month",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
]
X = df[feature_cols].values.astype(np.float32)
y = df["fare_amount"].values.astype(np.float32)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

y_mean = y_train.mean()
train_labels_centered = y_train - y_mean
validation_labels_centered = y_val - y_mean




## === cell 2
def rmse_keras(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


model = Sequential()
model.add(
    Dense(
        256,
        activation="relu",
        input_dim=X_train_scaled.shape[1],
    )
)
model.add(BatchNormalization())
model.add(Dense(128, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(64, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(32, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(8, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(1))

adam = optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(
    loss="mean_squared_error",
    optimizer=adam,
    metrics=["mae", rmse_keras, "mse"],
)

model.summary()




## === cell 3
history = model.fit(
    x=X_train_scaled,
    y=train_labels_centered,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1,
    validation_data=(X_val_scaled, validation_labels_centered),
    shuffle=True,
)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_56/1816518445.py in <cell line: 0>()
----> 1 history = model.fit(
      2     x=X_train_scaled,
      3     y=train_labels_centered,
      4     batch_size=BATCH_SIZE,
      5     epochs=EPOCHS,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_56/4058344526.py in rmse_keras(y_true, y_pred)
      1 def rmse_keras(y_true, y_pred):
----> 2     return K.sqrt(K.mean(K.square(y_pred - y_true)))
      3 
      4 
      5 model = Sequential()

AttributeError: module 'keras.api.backend' has no attribute 'sqrt'

## === cell 4
test_df = pd.read_csv(TEST_PATH)
test_df["pickup_datetime"] = pd.to_datetime(test_df["pickup_datetime"])
test_df["hour"] = test_df["pickup_datetime"].dt.hour
test_df["dayofweek"] = test_df["pickup_datetime"].dt.dayofweek
test_df["month"] = test_df["pickup_datetime"].dt.month
test_df["distance"] = haversine(
    test_df["pickup_latitude"],
    test_df["pickup_longitude"],
    test_df["dropoff_latitude"],
    test_df["dropoff_longitude"],
)

X_test = test_df[feature_cols].values.astype(np.float32)
X_test_scaled = scaler.transform(X_test)

preds_centered = model.predict(X_test_scaled, batch_size=BATCH_SIZE).flatten()
preds = preds_centered + y_mean  # add back the mean we subtracted earlier
preds = np.maximum(preds, 0)  # fare cannot be negative

submission = pd.DataFrame({"key": test_df["key"], "fare_amount": preds})
submission.to_csv(SUBMISSION_NAME, index=False)
