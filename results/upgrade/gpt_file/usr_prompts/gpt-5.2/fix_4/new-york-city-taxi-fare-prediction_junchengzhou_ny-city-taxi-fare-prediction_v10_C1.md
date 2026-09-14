# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

print(os.listdir("../input"))



## === cell 1
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

import random

random.seed(42)
np.random.seed(42)
os.environ["PYTHONHASHSEED"] = "42"



## === cell 2
TRAIN_PATH = "../input/train.csv"
TEST_PATH = "../input/test.csv"

if not os.path.exists(TRAIN_PATH):
    TRAIN_PATH = "../input/new-york-city-taxi-fare-prediction/train.csv"
    TEST_PATH = "../input/new-york-city-taxi-fare-prediction/test.csv"

usecols_train = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
dtype_train = {
    "key": "object",
    "fare_amount": "float32",
    "pickup_datetime": "object",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16",
}

train = pd.read_csv(
    TRAIN_PATH,
    nrows=10_000_000,
    usecols=usecols_train,
    dtype=dtype_train,
    engine="c",
    memory_map=True,
)



## === cell 3
m = (
    (train["fare_amount"] > 0)
    & (train["fare_amount"] < 200)
    & (train["pickup_longitude"] > -150)
    & (train["pickup_longitude"] < 0)
    & (train["pickup_latitude"] > 0)
    & (train["pickup_latitude"] < 80)
    & (train["dropoff_longitude"] > -150)
    & (train["dropoff_longitude"] < 0)
    & (train["dropoff_latitude"] > 0)
    & (train["dropoff_latitude"] < 80)
    & (train["passenger_count"] <= 8)
)
train = train.loc[m].copy()



## === cell 4
_ = None



## === cell 5
usecols_test = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
dtype_test = {
    "key": "object",
    "pickup_datetime": "object",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16",
}

test = pd.read_csv(
    TEST_PATH,
    usecols=usecols_test,
    dtype=dtype_test,
    engine="c",
    memory_map=True,
)



## === cell 6
_ = None



## === cell 7
_ = None




## === cell 8
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points on the earth (km)
    given in decimal degrees.
    """
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371
    return c * r




## === cell 9
def add_travel_distance_vector_features(df):
    df["distance"] = haversine(
        df["dropoff_longitude"].values,
        df["dropoff_latitude"].values,
        df["pickup_longitude"].values,
        df["pickup_latitude"].values,
    ).astype(np.float32, copy=False)


add_travel_distance_vector_features(train)
add_travel_distance_vector_features(test)



## === cell 10
_ = None



## === cell 11
test_key = test["key"].copy()

train.drop(
    [
        "dropoff_longitude",
        "dropoff_latitude",
        "pickup_longitude",
        "pickup_latitude",
        "pickup_datetime",
    ],
    axis=1,
    inplace=True,
)
test.drop(
    [
        "dropoff_longitude",
        "dropoff_latitude",
        "pickup_longitude",
        "pickup_latitude",
        "pickup_datetime",
    ],
    axis=1,
    inplace=True,
)



## === cell 12
_ = None



## === cell 13
_ = None



## === cell 14
train.dropna(how="any", axis="rows", inplace=True)



## === cell 15
_ = None



## === cell 16
_ = None



## === cell 17
train = train.loc[train["fare_amount"] <= 100].copy()



## === cell 18
_ = None




## === cell 19
def key_to_int64_utc(key_series: pd.Series) -> np.ndarray:
    s = key_series.astype("string").str.slice(0, 26)
    dt64 = s.to_numpy(dtype="datetime64[ns]")
    return dt64.astype("int64", copy=False)


train["key_num"] = key_to_int64_utc(train["key"])
test["key_num"] = key_to_int64_utc(test["key"])



## === cell 20
_ = None



## === cell 21
_ = None



## === cell 22
y = train.pop("fare_amount")
X = train



## === cell 23
from sklearn.preprocessing import StandardScaler



## === cell 24
scaler = StandardScaler()



## === cell 25
X_scaled = scaler.fit_transform(X[["passenger_count", "distance", "key_num"]]).astype(
    np.float32, copy=False
)
test_scaled = scaler.transform(test[["passenger_count", "distance", "key_num"]]).astype(
    np.float32, copy=False
)



## === cell 26
X_train, X_valid, y_train, y_valid = train_test_split(
    X_scaled, y, test_size=0.01, random_state=42
)



## === cell 27
y_train_round = y_train.round(0).astype("int16")
y_valid_round = y_valid.round(0).astype("int16")



## === cell 28
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoder.fit(y_train_round)

y_train_enc = encoder.transform(y_train_round)
y_valid_enc = encoder.transform(y_valid_round)

n_classes = int(np.max(y_train_enc)) + 1

y_train_cat = np.zeros((y_train_enc.shape[0], n_classes), dtype=np.float32)
y_train_cat[np.arange(y_train_enc.shape[0]), y_train_enc] = 1.0

y_valid_cat = np.zeros((y_valid_enc.shape[0], n_classes), dtype=np.float32)
y_valid_cat[np.arange(y_valid_enc.shape[0]), y_valid_enc] = 1.0



## === cell 29
from tensorflow.keras.layers import Dense, Input, Dropout, LeakyReLU, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras import backend as Backend
import tensorflow as tf

try:
    tf.random.set_seed(42)
except Exception:
    pass




## === cell 30
def rmse(y_true, y_pred):
    return Backend.sqrt(Backend.square(y_pred - y_true))




## === cell 31
def nn(n_feature, k=1200, n_classes=101):
    model_in = Input(shape=(n_feature,))
    model = Dense(k)(model_in)
    model = LeakyReLU(0.15)(model)
    model = Dropout(0.2)(model)

    model = Dense(k)(model)
    model = LeakyReLU(0.15)(model)
    model = Dropout(0.2)(model)

    model = Dense(k)(model)
    model = LeakyReLU(0.15)(model)
    model = Dropout(0.2)(model)

    model = Dense(k)(model)
    model = LeakyReLU(0.15)(model)
    model = Dropout(0.2)(model)

    model = Dense(n_classes, activation="softmax")(model)

    model = Model(inputs=model_in, outputs=model)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=[rmse])
    return model




## === cell 32
def lstm(n):
    model_in = Input(shape=(1, n))
    model = LSTM(10)(model_in)
    model = Dense(1, activation="linear")(model)
    model = Model(model_in, model)
    model.compile(loss="mse", optimizer="adam", metrics=[rmse])
    return model




## === cell 33
model = nn(X_train.shape[1], n_classes=n_classes)



## === cell 34
history = model.fit(
    X_train,
    y_train_cat,
    batch_size=10000,
    epochs=2,
    verbose=1,
    validation_data=(X_valid, y_valid_cat),
)



## === cell 35
_ = None



## === cell 36
pres_proba = model.predict(test_scaled, batch_size=10000, verbose=1)



## === cell 37
pres_class = pres_proba.argmax(axis=-1)
pres_fare = encoder.inverse_transform(pres_class).astype("float32")



## === cell 38
submission = pd.DataFrame(
    {"key": test_key.values, "fare_amount": pres_fare}, columns=["key", "fare_amount"]
)



## === cell 39
submission.to_csv("submission.csv", index=False)



## === cell 40
print(os.listdir("."))
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
