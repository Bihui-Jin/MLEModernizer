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

12.78702

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from keras import regularizers




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def clean(df):
    df = df[(-76 <= df["pickup_longitude"]) & (df["pickup_longitude"] <= -72)]
    df = df[(-76 <= df["dropoff_longitude"]) & (df["dropoff_longitude"] <= -72)]
    df = df[(38 <= df["pickup_latitude"]) & (df["pickup_latitude"] <= 42)]
    df = df[(38 <= df["dropoff_latitude"]) & (df["dropoff_latitude"] <= 42)]
    df = df[(0 < df["fare_amount"]) & (df["fare_amount"] <= 250)]
    df = df[(df["dropoff_longitude"] != df["pickup_longitude"])]
    df = df[(df["dropoff_latitude"] != df["pickup_latitude"])]
    return df


def late_night(row):
    return 1 if (row["hour"] <= 6) or (row["hour"] >= 20) else 0


def night(row):
    return (
        1
        if ((row["hour"] <= 20) and (row["hour"] >= 16)) and (row["weekday"] < 5)
        else 0
    )


def process(df):
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["year"] = df["pickup_datetime"].dt.year
    df["month"] = df["pickup_datetime"].dt.month
    df["day"] = df["pickup_datetime"].dt.day
    df["hour"] = df["pickup_datetime"].dt.hour
    df["weekday"] = df["pickup_datetime"].dt.weekday
    df["night"] = df.apply(night, axis=1)
    df["late_night"] = df.apply(late_night, axis=1)

    df["pickup_longitude_binned"] = pd.qcut(df["pickup_longitude"], 16, labels=False)
    df["dropoff_longitude_binned"] = pd.qcut(df["dropoff_longitude"], 16, labels=False)
    df["pickup_latitude_binned"] = pd.qcut(df["pickup_latitude"], 16, labels=False)
    df["dropoff_latitude_binned"] = pd.qcut(df["dropoff_latitude"], 16, labels=False)

    df = df.drop("pickup_datetime", axis=1)
    return df


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_relevant_distances(df):
    ny = (-74.0063889, 40.7141667)
    jfk = (-73.7822222222, 40.6441666667)
    ewr = (-74.175, 40.69)
    lgr = (-73.87, 40.77)
    df["downtown_pickup_distance"] = manhattan(
        ny[1], ny[0], df["pickup_latitude"], df["pickup_longitude"]
    )
    df["downtown_dropoff_distance"] = manhattan(
        ny[1], ny[0], df["dropoff_latitude"], df["dropoff_longitude"]
    )
    df["jfk_pickup_distance"] = manhattan(
        jfk[1], jfk[0], df["pickup_latitude"], df["pickup_longitude"]
    )
    df["jfk_dropoff_distance"] = manhattan(
        jfk[1], jfk[0], df["dropoff_latitude"], df["dropoff_longitude"]
    )
    df["ewr_pickup_distance"] = manhattan(
        ewr[1], ewr[0], df["pickup_latitude"], df["pickup_longitude"]
    )
    df["ewr_dropoff_distance"] = manhattan(
        ewr[1], ewr[0], df["dropoff_latitude"], df["dropoff_longitude"]
    )
    df["lgr_pickup_distance"] = manhattan(
        lgr[1], lgr[0], df["pickup_latitude"], df["pickup_longitude"]
    )
    df["lgr_dropoff_distance"] = manhattan(
        lgr[1], lgr[0], df["dropoff_latitude"], df["dropoff_longitude"]
    )
    return df


def add_engineered(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]

    latdiff = lat1 - lat2
    londiff = lon1 - lon2
    euclidean = np.sqrt(latdiff**2 + londiff**2)

    df["latdiff"] = latdiff
    df["londiff"] = londiff
    df["euclidean"] = euclidean
    df["manhattan"] = manhattan(lat1, lon1, lat2, lon2)

    df = pd.get_dummies(df, columns=["weekday"])
    df = pd.get_dummies(df, columns=["month"])
    return df




## === cell 2
def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    df = pd.DataFrame(prediction, columns=[prediction_column])
    df[id_column] = raw_test[id_column].values
    df[[id_column, prediction_column]].to_csv(file_name, index=False)
    print("Output complete")


def plot_loss_accuracy(history):
    plt.figure(figsize=(20, 10))
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="validation")
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()




## === cell 3
TRAIN_PATH = "../input/train.csv"
TEST_PATH = "../input/test.csv"
SUBMISSION_NAME = "submission.csv"

BATCH_SIZE = 256
EPOCHS = 5
LEARNING_RATE = 0.0001
DATASET_SIZE = 700000




## === cell 4
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
train = pd.read_csv(
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
test = pd.read_csv(TEST_PATH, dtype=datatypes)




## === cell 5
train = clean(train)

train = process(train)
test = process(test)

train = add_relevant_distances(train)
test = add_relevant_distances(test)

train = add_engineered(train)
test = add_engineered(test)




## === cell 6
dropped_columns = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]
train_clean = train.drop(dropped_columns, axis=1)
test_clean = test.drop(dropped_columns + ["key", "passenger_count"], axis=1)

train_clean.head(5)




## === cell 7
feature_cols = [c for c in train_clean.columns if c != "fare_amount"]
missing_in_test = set(feature_cols) - set(test_clean.columns)
for col in missing_in_test:
    test_clean[col] = 0
extra_in_test = set(test_clean.columns) - set(feature_cols)
test_clean = test_clean.drop(columns=extra_in_test)

train_df, validation_df = train_test_split(train_clean, test_size=0.10, random_state=1)

train_labels = train_df["fare_amount"].values
validation_labels = validation_df["fare_amount"].values
train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)

scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_clean)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/2881378209.py in <cell line: 0>()
     15 
     16 scaler = preprocessing.MinMaxScaler()
---> 17 train_df_scaled = scaler.fit_transform(train_df)
     18 validation_df_scaled = scaler.transform(validation_df)
     19 test_scaled = scaler.transform(test_clean)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py in wrapped(self, X, *args, **kwargs)
    138     @wraps(f)
    139     def wrapped(self, X, *args, **kwargs):
--> 140         data_to_wrap = f(self, X, *args, **kwargs)
    141         if isinstance(data_to_wrap, tuple):
    142             # only wrap the first output for cross decomposition

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in fit_transform(self, X, y, **fit_params)
    876         if y is None:
    877             # fit method of arity 1 (unsupervised transformation)
--> 878             return self.fit(X, **fit_params).transform(X)
    879         else:
    880             # fit method of arity 2 (supervised transformation)

/usr/local/lib/python3.11/dist-packages/sklearn/preprocessing/_data.py in fit(self, X, y)
    425         # Reset internal state before fitting
    426         self._reset()
--> 427         return self.partial_fit(X, y)
    428 
    429     def partial_fit(self, X, y=None):

/usr/local/lib/python3.11/dist-packages/sklearn/preprocessing/_data.py in partial_fit(self, X, y)
    464 
    465         first_pass = not hasattr(self, "n_samples_seen_")
--> 466         X = self._validate_data(
    467             X,
    468             reset=first_pass,

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    563             raise ValueError("Validation should be done on X, y or both.")
    564         elif not no_val_X and no_val_y:
--> 565             X = check_array(X, input_name="X", **check_params)
    566             out = X
    567         elif no_val_X and not no_val_y:

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)
    808         # Use the original dtype for conversion if dtype is None
    809         new_dtype = dtype_orig if dtype is None else dtype
--> 810         array = array.astype(new_dtype)
    811         # Since we converted here, we do not need to convert again later
    812         dtype = None

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in astype(self, dtype, copy, errors)
   6641         else:
   6642             # else, only a single dtype is given
-> 6643             new_data = self._mgr.astype(dtype=dtype, copy=copy, errors=errors)
   6644             res = self._constructor_from_mgr(new_data, axes=new_data.axes)
   6645             return res.__finalize__(self, method="astype")

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in astype(self, dtype, copy, errors)
    428             copy = False
    429 
--> 430         return self.apply(
    431             "astype",
    432             dtype=dtype,

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in apply(self, f, align_keys, **kwargs)
    361                 applied = b.apply(f, **kwargs)
    362             else:
--> 363                 applied = getattr(b, f)(**kwargs)
    364             result_blocks = extend_blocks(applied, result_blocks)
    365 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/blocks.py in astype(self, dtype, copy, errors, using_cow, squeeze)
    756             values = values[0, :]  # type: ignore[call-overload]
    757 
--> 758         new_values = astype_array_safe(values, dtype, copy=copy, errors=errors)
    759 
    760         new_values = maybe_coerce_values(new_values)

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in astype_array_safe(values, dtype, copy, errors)
    235 
    236     try:
--> 237         new_values = astype_array(values, dtype, copy=copy)
    238     except (ValueError, TypeError):
    239         # e.g. _astype_nansafe can fail on object-dtype of strings

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in astype_array(values, dtype, copy)
    180 
    181     else:
--> 182         values = _astype_nansafe(values, dtype, copy=copy)
    183 
    184     # in pandas we don't store numpy str dtypes, so convert to object

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in _astype_nansafe(arr, dtype, copy, skipna)
    131     if copy or arr.dtype == object or dtype == object:
    132         # Explicit copy, or required since NumPy can't view from / to object.
--> 133         return arr.astype(dtype, copy=True)
    134 
    135     return arr.astype(dtype, copy=copy)

ValueError: could not convert string to float: '2010-06-03 21:25:47.0000001'

## === cell 8
model = Sequential()
model.add(
    Dense(
        256,
        activation="relu",
        input_dim=train_df_scaled.shape[1],
        activity_regularizer=regularizers.l1(0.01),
    )
)
model.add(BatchNormalization())
model.add(Dense(128, activation="relu", activity_regularizer=regularizers.l1(0.01)))
model.add(BatchNormalization())
model.add(Dense(64, activation="relu", activity_regularizer=regularizers.l1(0.01)))
model.add(BatchNormalization())
model.add(Dense(32, activation="relu", activity_regularizer=regularizers.l1(0.01)))
model.add(BatchNormalization())
model.add(Dense(16, activation="relu", activity_regularizer=regularizers.l1(0.01)))
model.add(BatchNormalization())
model.add(Dense(1))

adam = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(loss="mse", optimizer=adam, metrics=["mae"])




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/225939367.py in <cell line: 0>()
      4         256,
      5         activation="relu",
----> 6         input_dim=train_df_scaled.shape[1],
      7         activity_regularizer=regularizers.l1(0.01),
      8     )

NameError: name 'train_df_scaled' is not defined

## === cell 9
print("Dataset size:", DATASET_SIZE)
print("Epochs:", EPOCHS)
print("Learning rate:", LEARNING_RATE)
print("Batch size:", BATCH_SIZE)
print("Input dimension:", train_df_scaled.shape[1])
print("Features used:", train_df.columns.tolist())




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2409357405.py in <cell line: 0>()
      3 print("Learning rate:", LEARNING_RATE)
      4 print("Batch size:", BATCH_SIZE)
----> 5 print("Input dimension:", train_df_scaled.shape[1])
      6 print("Features used:", train_df.columns.tolist())
      7 

NameError: name 'train_df_scaled' is not defined

## === cell 10
model.summary()




## === cell 11
history = model.fit(
    x=train_df_scaled,
    y=train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1,
    validation_data=(validation_df_scaled, validation_labels),
    shuffle=True,
)




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/737627353.py in <cell line: 0>()
      1 history = model.fit(
----> 2     x=train_df_scaled,
      3     y=train_labels,
      4     batch_size=BATCH_SIZE,
      5     epochs=EPOCHS,

NameError: name 'train_df_scaled' is not defined

## === cell 12
plot_loss_accuracy(history)




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3778224606.py in <cell line: 0>()
----> 1 plot_loss_accuracy(history)
      2 
      3 

NameError: name 'history' is not defined

## === cell 13
prediction = model.predict(test_scaled, batch_size=128, verbose=1)




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1893342136.py in <cell line: 0>()
----> 1 prediction = model.predict(test_scaled, batch_size=128, verbose=1)
      2 
      3 

NameError: name 'test_scaled' is not defined

## === cell 14
output_submission(test, prediction, "key", "fare_amount", SUBMISSION_NAME)

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3068830405.py in <cell line: 0>()
----> 1 output_submission(test, prediction, "key", "fare_amount", SUBMISSION_NAME)

NameError: name 'prediction' is not defined
