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

4.33523

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 50.00544) has done: 'The timeout is dominated by reading 10M rows with `parse_dates` plus expensive per-row string-to-datetime parsing in `key_to_int64_utc`, and by allocating huge one-hot matrices for `y_train_cat/y_valid_cat`. I keep the same features, same model, and same training loop semantics, but speed up ingestion and feature building by (1) not parsing `pickup_datetime` at read time (it’s dropped later anyway), (2) parsing `key` to `datetime64` via fast NumPy slicing/`astype('datetime64[ns]')` instead of `pd.to_datetime`, and (3) replacing dense one-hot (`np.eye(...)[...]`) with Keras’ sparse categorical loss (exactly equivalent) so we pass integer class labels without the massive allocation. These changes preserve accuracy/logic while cutting CPU and memory pressure enough to fit the 600s budget.'

# 9. Code solution

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
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
dtype_train = {
    "key": "object",
    "fare_amount": "float32",
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
fa = train["fare_amount"].to_numpy(copy=False)
plon = train["pickup_longitude"].to_numpy(copy=False)
plat = train["pickup_latitude"].to_numpy(copy=False)
dlon = train["dropoff_longitude"].to_numpy(copy=False)
dlat = train["dropoff_latitude"].to_numpy(copy=False)
pc = train["passenger_count"].to_numpy(copy=False)

m = (
    (fa > 0)
    & (fa < 200)
    & (plon > -150)
    & (plon < 0)
    & (plat > 0)
    & (plat < 80)
    & (dlon > -150)
    & (dlon < 0)
    & (dlat > 0)
    & (dlat < 80)
    & (pc <= 8)
)
train = train.loc[m]



## === cell 4
_ = None



## === cell 5
usecols_test = [
    "key",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
dtype_test = {
    "key": "object",
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
        df["dropoff_longitude"].to_numpy(copy=False),
        df["dropoff_latitude"].to_numpy(copy=False),
        df["pickup_longitude"].to_numpy(copy=False),
        df["pickup_latitude"].to_numpy(copy=False),
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
train = train.loc[train["fare_amount"] <= 100]



## === cell 18
_ = None




## === cell 19
def key_to_int64_utc(key_series: pd.Series) -> np.ndarray:
    s = key_series.to_numpy(dtype="U32", copy=False)
    s26 = np.char.substr(s, 0, 26)
    dt64 = s26.astype("datetime64[ns]")  # invalid parses -> NaT (int64 min)
    return dt64.astype("int64", copy=False)


train["key_num"] = key_to_int64_utc(train["key"])
test["key_num"] = key_to_int64_utc(test["key"])

for df in (train, test):
    kn = df["key_num"].to_numpy(copy=False)
    bad = kn == np.iinfo(np.int64).min
    if bad.any():
        med = int(np.median(kn[~bad])) if (~bad).any() else 0
        kn2 = kn.copy()
        kn2[bad] = med
        df["key_num"] = kn2



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1926013249.py in <cell line: 0>()
     11 
     12 
---> 13 train["key_num"] = key_to_int64_utc(train["key"])
     14 test["key_num"] = key_to_int64_utc(test["key"])
     15 

/tmp/ipykernel_11/1926013249.py in key_to_int64_utc(key_series)
      6     # 'YYYY-MM-DD HH:MM:SS.ffffff' (as in the dataset key format).
      7     s = key_series.to_numpy(dtype="U32", copy=False)
----> 8     s26 = np.char.substr(s, 0, 26)
      9     dt64 = s26.astype("datetime64[ns]")  # invalid parses -> NaT (int64 min)
     10     return dt64.astype("int64", copy=False)

AttributeError: module 'numpy.core.defchararray' has no attribute 'substr'

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
feature_cols = ["passenger_count", "distance", "key_num"]
missing_train = [c for c in feature_cols if c not in X.columns]
missing_test = [c for c in feature_cols if c not in test.columns]
if missing_train or missing_test:
    raise KeyError(
        f"Missing columns. train_missing={missing_train}, test_missing={missing_test}"
    )

X_feat = X[feature_cols].to_numpy(dtype=np.float32, copy=False)
test_feat = test[feature_cols].to_numpy(dtype=np.float32, copy=False)

X_scaled = scaler.fit_transform(X_feat).astype(np.float32, copy=False)
test_scaled = scaler.transform(test_feat).astype(np.float32, copy=False)



## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2629720171.py in <cell line: 0>()
      3 missing_test = [c for c in feature_cols if c not in test.columns]
      4 if missing_train or missing_test:
----> 5     raise KeyError(
      6         f"Missing columns. train_missing={missing_train}, test_missing={missing_test}"
      7     )

KeyError: "Missing columns. train_missing=['key_num'], test_missing=['key_num']"

## === cell 26
X_train, X_valid, y_train, y_valid = train_test_split(
    X_scaled, y, test_size=0.01, random_state=42
)



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/263540527.py in <cell line: 0>()
      1 X_train, X_valid, y_train, y_valid = train_test_split(
----> 2     X_scaled, y, test_size=0.01, random_state=42
      3 )
      4 

NameError: name 'X_scaled' is not defined

## === cell 27
y_train_round = y_train.round(0).astype("int16")
y_valid_round = y_valid.round(0).astype("int16")



## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1345238122.py in <cell line: 0>()
----> 1 y_train_round = y_train.round(0).astype("int16")
      2 y_valid_round = y_valid.round(0).astype("int16")
      3 

NameError: name 'y_train' is not defined

## === cell 28
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
all_round = pd.concat(
    [y_train_round.astype("int16"), y_valid_round.astype("int16")], axis=0
)
encoder.fit(all_round)

y_train_enc = encoder.transform(y_train_round)
y_valid_enc = encoder.transform(y_valid_round)

n_classes = len(encoder.classes_)

y_train_cat = None
y_valid_cat = None



## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1518545923.py in <cell line: 0>()
      3 encoder = LabelEncoder()
      4 all_round = pd.concat(
----> 5     [y_train_round.astype("int16"), y_valid_round.astype("int16")], axis=0
      6 )
      7 encoder.fit(all_round)

NameError: name 'y_train_round' is not defined

## === cell 29
use_tf = False
tf_import_error = "Forced sklearn fallback due to known TF/protobuf incompatibility in this environment."

print("TensorFlow usable:", use_tf)
print("TensorFlow import error (will use sklearn MLPClassifier):", tf_import_error)



## === cell 30
if use_tf:
    import tensorflow as tf  # noqa: F401
    from tensorflow.keras.layers import Dense, Input, Dropout, LeakyReLU, LSTM
    from tensorflow.keras.models import Model
    from tensorflow.keras import backend as Backend

    try:
        tf.random.set_seed(42)
    except Exception:
        pass

    def rmse(y_true, y_pred):
        y_true = Backend.cast(y_true, y_pred.dtype)
        return Backend.sqrt(Backend.mean(Backend.square(y_pred - y_true)))

else:
    rmse = None



## === cell 31
if use_tf:

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

        model.compile(
            loss="sparse_categorical_crossentropy", optimizer="adam", metrics=[rmse]
        )
        return model

else:
    nn = None



## === cell 32
if use_tf:

    def lstm(n):
        model_in = Input(shape=(1, n))
        model = LSTM(10)(model_in)
        model = Dense(1, activation="linear")(model)
        model = Model(model_in, model)
        model.compile(loss="mse", optimizer="adam", metrics=[rmse])
        return model

else:
    lstm = None



## === cell 33
if use_tf:
    model = nn(X_train.shape[1], n_classes=n_classes)
else:
    from sklearn.neural_network import MLPClassifier

    model = MLPClassifier(
        hidden_layer_sizes=(1200, 1200, 1200, 1200),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size=10000,
        learning_rate_init=0.001,
        max_iter=5,
        shuffle=True,
        random_state=42,
        verbose=True,
        tol=0.0,
        n_iter_no_change=200,
    )



## === cell 34
if use_tf:
    history = model.fit(
        X_train,
        y_train_enc,  # integer labels (sparse)
        batch_size=10000,
        epochs=2,
        verbose=1,
        validation_data=(X_valid, y_valid_enc),  # integer labels (sparse)
    )
else:
    model.fit(X_train, y_train_enc)



## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2944323888.py in <cell line: 0>()
      9     )
     10 else:
---> 11     model.fit(X_train, y_train_enc)
     12 

NameError: name 'X_train' is not defined

## === cell 35
_ = None



## === cell 36
if use_tf:
    pres_proba = model.predict(test_scaled, batch_size=10000, verbose=1)
else:
    pres_proba = model.predict_proba(test_scaled)

pres_class = pres_proba.argmax(axis=-1)
pres_fare = encoder.inverse_transform(pres_class).astype("float32")

submission = pd.DataFrame(
    {"key": test_key.values, "fare_amount": pres_fare}, columns=["key", "fare_amount"]
)

submission.to_csv("submission.csv", index=False)

print(os.listdir("."))
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)

## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2175245132.py in <cell line: 0>()
      2     pres_proba = model.predict(test_scaled, batch_size=10000, verbose=1)
      3 else:
----> 4     pres_proba = model.predict_proba(test_scaled)
      5 
      6 pres_class = pres_proba.argmax(axis=-1)

NameError: name 'test_scaled' is not defined
