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

4.30388

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from datetime import datetime
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from keras.models import Sequential, load_model
from keras.layers import Dense, BatchNormalization
from keras import optimizers, regularizers, backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping

TRAIN_PATH = "../input/train.csv"
TEST_PATH = "../input/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 256
EPOCHS = 50  # enough for the reduced dataset
LEARNING_RATE = 0.001
DATASET_SIZE = 80000  # use a manageable subset for quick training




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
dtypes = {
    "key": "str",
    "fare_amount": "float32",
    "pickup_datetime": "str",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}
train_full = pd.read_csv(TRAIN_PATH, nrows=DATASET_SIZE, dtype=dtypes)
test_full = pd.read_csv(TEST_PATH, dtype=dtypes)




## === cell 2
def clean(df, is_train=True):
    """Basic data cleaning. Skips fare_amount filter for test data."""
    print("Old size:", len(df))
    df = df.dropna()
    df = df[
        (df["dropoff_longitude"] != df["pickup_longitude"])
        | (df["dropoff_latitude"] != df["pickup_latitude"])
    ]
    df = df[
        (
            df[
                [
                    "pickup_longitude",
                    "dropoff_longitude",
                    "pickup_latitude",
                    "dropoff_latitude",
                ]
            ]
            != 0
        ).all(axis=1)
    ]
    lon_min, lon_max, lat_min, lat_max = -74.5, -72.8, 40.5, 41.8
    df = df[(lon_min <= df["pickup_longitude"]) & (df["pickup_longitude"] <= lon_max)]
    df = df[(lon_min <= df["dropoff_longitude"]) & (df["dropoff_longitude"] <= lon_max)]
    df = df[(lat_min <= df["pickup_latitude"]) & (df["pickup_latitude"] <= lat_max)]
    df = df[(lat_min <= df["dropoff_latitude"]) & (df["dropoff_latitude"] <= lat_max)]
    if is_train and "fare_amount" in df.columns:
        df = df[(df["fare_amount"] > 0) & (df["fare_amount"] <= 50)]
    df = df[df["passenger_count"] > 0]
    print("New size after cleaning:", len(df))
    return df


def add_time_features(df):
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["year"] = df["pickup_datetime"].dt.year
    df["month"] = df["pickup_datetime"].dt.month
    df["day"] = df["pickup_datetime"].dt.day
    df["hour"] = df["pickup_datetime"].dt.hour
    df["weekday"] = df["pickup_datetime"].dt.weekday
    df["night"] = ((df["hour"] >= 20) & (df["weekday"] < 5)).astype(int)
    df["late_night"] = (df["hour"] <= 3).astype(int)
    df["rush_hour"] = (
        (df["hour"] >= 16) & (df["hour"] <= 20) & (df["weekday"] < 5)
    ).astype(int)
    return df


def add_coordinate_features(df):
    df["latdiff"] = (df["pickup_latitude"] - df["dropoff_latitude"]).abs()
    df["londiff"] = (df["pickup_longitude"] - df["dropoff_longitude"]).abs()
    return df


def add_distance_features(df):
    p = np.pi / 180.0
    lat1, lon1 = df["pickup_latitude"] * p, df["pickup_longitude"] * p
    lat2, lon2 = df["dropoff_latitude"] * p, df["dropoff_longitude"] * p
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    df["haversine"] = 3958.8 * 2 * np.arcsin(np.sqrt(a))  # Earth radius in miles
    df["manhattan"] = df["latdiff"] + df["londiff"]
    return df


def prepare_features(df):
    df = add_time_features(df)
    df = add_coordinate_features(df)
    df = add_distance_features(df)
    return df




## === cell 3
train_df = clean(train_full, is_train=True)
test_df = clean(test_full, is_train=False)

train_df = prepare_features(train_df)
test_df = prepare_features(test_df)




## === cell 4
train_set, val_set = train_test_split(train_df, test_size=0.1, random_state=1)

train_labels = train_set["fare_amount"].values
val_labels = val_set["fare_amount"].values

train_set = train_set.drop(["fare_amount", "key", "pickup_datetime"], axis=1)
val_set = val_set.drop(["fare_amount", "key", "pickup_datetime"], axis=1)
test_features = test_df.drop(["key", "pickup_datetime"], axis=1)

scaler = preprocessing.MinMaxScaler()
X_train = scaler.fit_transform(train_set)
X_val = scaler.transform(val_set)
X_test = scaler.transform(test_features)




## === cell 5
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


model = Sequential(
    [
        Dense(
            256,
            activation="linear",
            input_dim=X_train.shape[1],
            activity_regularizer=regularizers.l1(0.01),
        ),
        BatchNormalization(),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dense(32, activation="relu"),
        BatchNormalization(),
        Dense(8, activation="relu"),
        BatchNormalization(),
        Dense(1),
    ]
)

adam = optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(loss="mse", optimizer=adam, metrics=["mae", rmse])

checkpoint = ModelCheckpoint("best_model.h5", save_best_only=True, verbose=0)
early_stop = EarlyStopping(patience=5, restore_best_weights=True, verbose=0)

history = model.fit(
    X_train,
    train_labels,
    validation_data=(X_val, val_labels),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint, early_stop],
    verbose=1,
)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/2772064255.py in <cell line: 0>()
     30 early_stop = EarlyStopping(patience=5, restore_best_weights=True, verbose=0)
     31 
---> 32 history = model.fit(
     33     X_train,
     34     train_labels,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_55/2772064255.py in rmse(y_true, y_pred)
      1 def rmse(y_true, y_pred):
----> 2     return K.sqrt(K.mean(K.square(y_pred - y_true)))
      3 
      4 
      5 model = Sequential(

AttributeError: module 'keras.api.backend' has no attribute 'sqrt'

## === cell 6
test_pred = model.predict(X_test, batch_size=128).flatten()
submission = pd.DataFrame({"key": test_full["key"], "fare_amount": test_pred})
submission.to_csv(SUBMISSION_NAME, index=False)
print(f"Submission written to {SUBMISSION_NAME}")

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/3717593913.py in <cell line: 0>()
      1 # Predict on the official test set and write submission
      2 test_pred = model.predict(X_test, batch_size=128).flatten()
----> 3 submission = pd.DataFrame({"key": test_full["key"], "fare_amount": test_pred})
      4 submission.to_csv(SUBMISSION_NAME, index=False)
      5 print(f"Submission written to {SUBMISSION_NAME}")

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __init__(self, data, index, columns, dtype, copy)
    776         elif isinstance(data, dict):
    777             # GH#38939 de facto copy defaults to False only in non-dict cases
--> 778             mgr = dict_to_mgr(data, index, columns, dtype=dtype, copy=copy, typ=manager)
    779         elif isinstance(data, ma.MaskedArray):
    780             from numpy.ma import mrecords

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in dict_to_mgr(data, index, columns, dtype, typ, copy)
    501             arrays = [x.copy() if hasattr(x, "dtype") else x for x in arrays]
    502 
--> 503     return arrays_to_mgr(arrays, columns, index, dtype=dtype, typ=typ, consolidate=copy)
    504 
    505 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in arrays_to_mgr(arrays, columns, index, dtype, verify_integrity, typ, consolidate)
    112         # figure out the index, if necessary
    113         if index is None:
--> 114             index = _extract_index(arrays)
    115         else:
    116             index = ensure_index(index)

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in _extract_index(data)
    688                     f"length {len(index)}"
    689                 )
--> 690                 raise ValueError(msg)
    691         else:
    692             index = default_index(lengths[0])

ValueError: array length 9541 does not match index length 9914
