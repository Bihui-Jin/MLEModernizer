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
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
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

9.40951

# 6. Current score

5.77574

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 5.77574) has done: 'The crash happens before your code runs because importing `tf_keras` triggers a protobuf incompatibility in this Kaggle environment (the `MessageFactory.GetPrototype` error). The smallest safe fix is to stop using `tf_keras` and switch to the already-installed `keras` package for the same Sequential/Dense/RMSprop training loop, keeping the model and training logic identical. I also make the input path resolution robust (some Kaggle setups use `/kaggle/input/...`) and add a tiny numeric cleanup (fill NaNs/infs) to prevent training/prediction from producing invalid values, without changing the overall approach. Finally, the script still writes a valid `submission.csv` with `key,fare_amount`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import keras
from keras import models, layers, optimizers


def resolve_path(rel_path):
    candidates = [
        rel_path,
        rel_path.replace("../input", "/kaggle/input"),
        rel_path.replace(
            "../input", "/kaggle/input/new-york-city-taxi-fare-prediction"
        ),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return rel_path


print("Listing ../input (if exists):")
try:
    print(os.listdir("../input"))
except Exception as e:
    print("Could not list ../input:", repr(e))

TRAIN_PATH = resolve_path("../input/train.csv")
TEST_PATH = resolve_path("../input/test.csv")
SAMPLE_SUB_PATH = resolve_path("../input/sample_submission.csv")

print("Resolved paths:")
print("TRAIN_PATH:", TRAIN_PATH, "exists:", os.path.exists(TRAIN_PATH))
print("TEST_PATH:", TEST_PATH, "exists:", os.path.exists(TEST_PATH))
print("SAMPLE_SUB_PATH:", SAMPLE_SUB_PATH, "exists:", os.path.exists(SAMPLE_SUB_PATH))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
training_data = pd.read_csv(TRAIN_PATH, nrows=2_000_000)
test_data = pd.read_csv(TEST_PATH)

print(training_data.shape, test_data.shape)
training_data.head()



## === cell 2
X_train = training_data.copy()
X_test = test_data.copy()
Y_train = training_data.copy()



## === cell 3
X_train["pickup_datetime"] = pd.to_datetime(X_train["pickup_datetime"])
X_train["hour"] = X_train["pickup_datetime"].dt.hour
X_train["latitude_distance"] = (
    X_train["dropoff_latitude"] - X_train["pickup_latitude"]
).abs()
X_train["longitude_distance"] = (
    X_train["dropoff_longitude"] - X_train["pickup_longitude"]
).abs()
X_train = X_train.drop(
    columns=[
        "dropoff_longitude",
        "dropoff_latitude",
        "key",
        "fare_amount",
        "pickup_datetime",
    ]
)



## === cell 4
X_test["pickup_datetime"] = pd.to_datetime(X_test["pickup_datetime"])
X_test["hour"] = X_test["pickup_datetime"].dt.hour
X_test["latitude_distance"] = (
    X_test["dropoff_latitude"] - X_test["pickup_latitude"]
).abs()
X_test["longitude_distance"] = (
    X_test["dropoff_longitude"] - X_test["pickup_longitude"]
).abs()
X_test = X_test.drop(
    columns=["dropoff_longitude", "dropoff_latitude", "key", "pickup_datetime"]
)



## === cell 5
Y_train = Y_train["fare_amount"].astype("float32").values

X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0.0)
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0.0)

X_train = X_train.astype("float32").values
X_test = X_test.astype("float32").values

print("X_train:", X_train.shape, "Y_train:", Y_train.shape, "X_test:", X_test.shape)



## === cell 6
model = models.Sequential()
model.add(layers.Dense(256, activation="relu", input_shape=(X_train.shape[1],)))
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dense(1))

rmsprop = optimizers.RMSprop(learning_rate=0.001)

model.compile(optimizer=rmsprop, loss="mse", metrics=["mae"])

model.fit(X_train, Y_train, epochs=8, batch_size=512, verbose=2)

Y_pred = model.predict(X_test, verbose=0).reshape(-1).astype("float32")
print("Pred shape:", Y_pred.shape, "Pred head:", Y_pred[:5])



## === cell 7
submission = pd.DataFrame({"key": test_data["key"].values, "fare_amount": Y_pred})

assert list(submission.columns) == ["key", "fare_amount"]
assert len(submission) == len(test_data)

submission.head()



## === cell 8
OUT_PATH = "submission.csv"
submission.to_csv(OUT_PATH, index=False)
print("Wrote:", OUT_PATH, "rows:", len(submission))
print(submission.dtypes)
