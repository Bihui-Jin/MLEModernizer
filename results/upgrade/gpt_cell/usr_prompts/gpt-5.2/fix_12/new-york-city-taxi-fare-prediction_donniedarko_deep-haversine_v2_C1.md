# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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
ipywidgets==8.1.5
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
protobuf==6.33.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
tf_keras==2.18.0
tqdm==4.67.1

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

3.97307

# 6. Current score

9.86951

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 14.45422) has done: 'Your notebook currently won’t yield a Kaggle score because it never reaches a stable end-to-end run in this environment: it uses a non-existent input path (`/kaggle/input/train.csv` etc.) and also tries to pip-downgrade `protobuf` then `execv`, which is brittle and can prevent `submission.csv` from being written. I make the smallest changes needed to (1) use the actual provided file paths (`/kaggle/data/...`), (2) remove the protobuf/pip restart block and rely on the already-installed `tf_keras` (compatible with TF 2.18), and (3) fix a small bug in test feature preparation (the unpack order was reversed), ensuring the predictions are aligned and a valid `submission.csv` is always produced. These changes preserve your model, features, loss, and training loop, while unblocking execution so you can get a real RMSE score to compare to the 3.97307 target.'
- What this solution (achieved 6.10433) has done: 'Diagnosis: Cell 13 crashes during `import tf_keras` because the installed `protobuf==6.33.0` is incompatible with the `tf_keras` package in this environment, triggering `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` at import time. The notebook already has `tensorflow==2.18.0`, which provides a compatible `tf.keras` implementation and avoids this protobuf mismatch. The rest of the code (cell 14) expects `keras`, `lyr`, `act`, `Model`, and `K` symbols, so we must keep those names available with the same semantics.

Patch summary: In cell 13, replace `tf_keras` imports with `tensorflow.keras` equivalents while keeping the same symbol names (`keras`, `lyr`, `act`, `Model`, `K`, `tf`). This resolves the import-time protobuf crash without changing model/training logic or downstream interfaces.

Updated cells:'
- What this solution (achieved 7.12647) has done: 'Diagnosis: The crash occurs on the first TensorFlow import in cell 13, before any model code runs. With TensorFlow 2.18.0 and protobuf 6.33.0, a known incompatibility triggers `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` during TensorFlow’s protobuf initialization. The minimal safe workaround in-notebook is to force protobuf’s pure-Python implementation before importing TensorFlow, which avoids the missing C++ API.

Patch summary: In cell 13 only, set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (and version=2 for compatibility) via `os.environ` before importing TensorFlow/Keras. Keep all existing imports and aliases (`tf`, `keras`, `lyr`, `act`, `Model`, `K`) unchanged so later cells remain compatible.

Updated cells: Cell 13 only.

Compatibility notes for cell k+1: Cell 14 uses `tf` and `K` exactly as imported in cell 13; these names and modules remain identical, so no interface/behavior changes beyond avoiding the import-time crash.

Assumptions: Environment allows setting `os.environ` in-process before the first TensorFlow import (TensorFlow not previously imported in earlier cells, which matches the provided notebook).'
- What this solution (achieved 9.86951) has done: 'Diagnosis: The crash happens during `import tensorflow as tf` in cell 13 due to an incompatibility between TensorFlow 2.18 and the installed `protobuf==6.33.0`. The error `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` is a known symptom when protobuf 5/6 is used with TensorFlow builds expecting protobuf 4.x runtime APIs. The existing environment-variable tweaks for protobuf implementation do not resolve this version-level API mismatch.

Patch summary: Modify cell 13 to force TensorFlow to use the pure-Python protobuf implementation and proactively downgrade protobuf at runtime to a TensorFlow-compatible 4.x version before importing TensorFlow. This is the minimal localized change that unblocks the import without changing any model/training logic.

Updated cells: Only cell 13 is changed.

Compatibility notes for cell k+1: The patch keeps the same imports/aliases (`tf`, `keras`, `lyr`, `act`, `Model`, `K`) so cell 14 and later cells can run unchanged.

Assumptions: The runtime allows `pip` installs during execution (common in Kaggle-like environments) and has network/package index access or cached wheels for protobuf 4.x; if not, the same fix would require the base environment protobuf to be pre-pinned to <5.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["figure.figsize"] = 16, 9
from tqdm import tqdm
from ipywidgets import widgets




## === cell 1
def read_csv_sampled(path, frac, chunksize=10**5, random_state=None):
    samples = []

    for df_chunk in tqdm(pd.read_csv(path, chunksize=chunksize)):
        samples.append(df_chunk.sample(frac=frac, random_state=random_state))

    df = pd.concat(samples, ignore_index=True)
    df["pickup_datetime"] = df["pickup_datetime"].apply(pd.Timestamp)

    return df




## === cell 2
TRAIN_PATH = "/kaggle/data/train.csv"
TEST_PATH = "/kaggle/data/test.csv"
SAMPLE_SUB_PATH = "/kaggle/data/sample_submission.csv"

df_train_sample = read_csv_sampled(TRAIN_PATH, 0.01, random_state=1989)



## === cell 3
df_train_sample.info()



## === cell 4
df_test = pd.read_csv(TEST_PATH, parse_dates=["pickup_datetime"])
df_test.info()



## === cell 5
df_train_sample.dropna(inplace=True)



## === cell 6
is_weird = df_train_sample["fare_amount"] < 0
is_weird |= ~df_train_sample["pickup_latitude"].between(40, 42)
is_weird |= ~df_train_sample["pickup_longitude"].between(-75, -72)
is_weird |= ~df_train_sample["dropoff_latitude"].between(40, 42)
is_weird |= ~df_train_sample["dropoff_longitude"].between(-75, -72)
is_weird |= df_train_sample["passenger_count"] == 0
print(is_weird.sum())

df_train_sample = df_train_sample[~is_weird]



## === cell 7
df_train_sample.describe()



## === cell 8
df_test.describe()



## === cell 9
sns.histplot(np.log1p(df_train_sample["fare_amount"].dropna()), bins=100, kde=True)



## === cell 10
plt.scatter(
    df_train_sample["pickup_longitude"],
    df_train_sample["pickup_latitude"],
    c=np.log1p(df_train_sample["fare_amount"]),
    alpha=0.7,
    s=1,
    lw=0,
)
plt.xlim(*np.percentile(df_train_sample["pickup_longitude"], [1, 99]))
plt.ylim(*np.percentile(df_train_sample["pickup_latitude"], [1, 99]))
plt.colorbar()




## === cell 11
def prep_data(df, shuffle=False):
    weekofyear = df["pickup_datetime"].dt.isocalendar().week.astype(int).to_numpy()

    X_cat = np.vstack(
        [
            df["pickup_datetime"].dt.hour,  # 0-23
            df["pickup_datetime"].dt.weekday + 24,  # 24-30
            df["pickup_datetime"].dt.dayofyear + 30,  # 31-396
            weekofyear + 396,  # 397-449
            df["pickup_datetime"].dt.year - 2009 + 450,  # 450-456
            df["passenger_count"] + 456,  # 457-463
        ]
    ).T.astype(np.int32)

    X_deg = df[
        ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]
    ].values.astype(np.float32)
    X_deg /= 180.0

    if "fare_amount" in df.columns:
        y = df["fare_amount"].values.astype(np.float32)

    if shuffle:
        rnd_ind = np.random.permutation(len(df))
        X_cat = X_cat[rnd_ind]
        X_deg = X_deg[rnd_ind]
        if "fare_amount" in df.columns:
            y = y[rnd_ind]

    if "fare_amount" in df.columns:
        return (X_cat, X_deg), y

    return (X_cat, X_deg)




## === cell 12
(X_cat, X_deg), y = prep_data(df_train_sample, shuffle=True)
print(X_cat.shape, X_deg.shape, y.shape)



## === cell 13
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

import pkgutil
import subprocess
import sys


def _ensure_protobuf_4x():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver

        major = int(pb_ver.split(".", 1)[0])
        if major >= 5:
            raise RuntimeError(f"protobuf {pb_ver} is too new for this TF build.")
    except Exception:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf>=4.21.0,<5"]
        )


_ensure_protobuf_4x()

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as lyr
import tensorflow.keras.activations as act
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K


## === cell 14
def K_haversine_bearing(x):
    R = 6371e3

    x_rad = x * np.pi

    lat1, lng1, lat2, lng2 = x_rad[:, 0], x_rad[:, 1], x_rad[:, 2], x_rad[:, 3]

    dlat = lat2 - lat1
    dlng = lng2 - lng1

    a = K.sin(dlat / 2) * K.sin(dlat / 2) + K.cos(lat1) * K.cos(lat2) * K.sin(
        dlng / 2
    ) * K.sin(dlng / 2)
    c = 2 * tf.atan2(K.sqrt(a), K.sqrt(1 - a))

    d = K.log((R * c) + 1)

    x_ = K.sin(dlng) * K.cos(lat2)
    y_ = (K.cos(lat1) * K.sin(lat2)) - (K.sin(lat1) * K.cos(lat2) * K.cos(dlng))
    b = tf.atan2(x_, y_) / np.pi

    return K.concatenate([K.reshape(d, (-1, 1)), K.reshape(b, (-1, 1))])




## === cell 15
lyr_latlng_input = lyr.Input(X_deg.shape[1:])
lyr_haversine_bearing = lyr.Lambda(K_haversine_bearing, name="haversine")(
    lyr_latlng_input
)

lyr_cat_input = lyr.Input(X_cat.shape[1:])
lyr_cat_embeddings = lyr.Embedding(int(X_cat.max()) + 1, 16)(lyr_cat_input)
lyr_cat_flatten = lyr.Flatten()(lyr_cat_embeddings)

lyr_concat = lyr.concatenate([lyr_latlng_input, lyr_haversine_bearing, lyr_cat_flatten])

lyr_dense1 = lyr.Dense(256, activation=act.selu)(lyr_concat)
lyr_dense2 = lyr.Dense(256, activation=act.selu)(lyr_dense1)
lyr_dense3 = lyr.Dense(256, activation=act.selu)(lyr_dense2)
lyr_out = lyr.Dense(1, activation=act.selu)(lyr_dense3)

model = Model([lyr_latlng_input, lyr_cat_input], lyr_out)
model.summary()



## === cell 16
model.compile(loss="mse", optimizer="nadam")



## === cell 17
model.fit([X_deg, X_cat], y, batch_size=512, epochs=50, validation_split=0.3)



## === cell 18
val_loc = int(len(X_deg) * 0.3)
y_pred = np.clip(
    model.predict([X_deg[-val_loc:], X_cat[-val_loc:]], batch_size=1024, verbose=1)[
        :, 0
    ],
    0,
    None,
)
err = pd.Series(y[-val_loc:] - y_pred)
sns.histplot(err, bins=100, kde=True)
print(f"rmse: {(err**2).mean()**0.5:.3f}")
err.describe()



## === cell 19
X_cat_test, X_deg_test = prep_data(df_test)



## === cell 20
y_test = np.clip(
    model.predict([X_deg_test, X_cat_test], batch_size=512, verbose=1), 0, None
)



## === cell 21
sns.histplot(np.log1p(y), bins=100, kde=True)
sns.histplot(np.log1p(y_test[:, 0]), bins=100, kde=True)



## === cell 22
df_sub = pd.DataFrame({"key": df_test["key"].values, "fare_amount": y_test[:, 0]})



## === cell 23
df_sub.head()



## === cell 24
df_sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_sub.shape)
print("Columns:", list(df_sub.columns))
print("Any NA in submission?:", df_sub.isna().any().to_dict())
