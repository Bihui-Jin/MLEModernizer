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

5.91088

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 8.74843) has done: 'I fix the datetime feature bug by replacing the removed `dt.weekofyear` accessor with the stable ISO calendar week (`dt.isocalendar().week`) while keeping the same categorical offset scheme. I also fix the Keras import/runtime crash by switching to `tf.keras` (compatible with the provided TensorFlow) while preserving the exact model architecture and training loop. Next, I correct a small data-prep unpacking mistake for the test set and ensure the submission file has exactly the required columns and is written as `submission.csv`. These changes are execution-stability fixes and should also improve the score versus “no submission” by enabling the model to train and predict properly.'
- What this solution (achieved 7.30552) has done: 'I fix the TensorFlow/Keras import crash caused by an incompatible protobuf version by forcing the pure-Python protobuf implementation before importing TensorFlow, which is a common Kaggle runtime workaround and should be score-neutral. I also fix a small bug in test feature unpacking (the prep function returns a tuple) so inference runs correctly end-to-end and actually writes `submission.csv`. Finally, I correct the train/validation split logic used for the quick RMSE check so it evaluates the same validation subset that `validation_split=0.3` uses (last 30% after shuffling), without changing the model or training behavior; this is score-neutral but prevents misleading diagnostics.'
- What this solution (achieved 5.63691) has done: 'I fix the TensorFlow import crash that comes from the protobuf runtime mismatch by forcing the pure‑Python protobuf implementation earlier and using a safe, consistent TensorFlow/Keras import path. I also fix the test-set unpacking bug (your `prep_data` returns a tuple of arrays) so inference uses the correct `[X_deg_test, X_cat_test]` inputs and runs end-to-end. To move RMSE toward the target without changing the model/training loop, I increase the training sample fraction modestly (still chunk-sampled) so the exact same network trains on more data, which is the smallest legitimate lever here. Finally, I keep the submission formatting strict (`key,fare_amount`) and ensure the file is written as `submission.csv`.'
- What this solution (achieved 6.32668) has done: 'I fix the TensorFlow import crash by enforcing the pure-Python protobuf implementation *before* any TensorFlow/Keras import and by also setting the `protobuf` runtime version to 3 (which avoids the `MessageFactory.GetPrototype` mismatch seen in Kaggle images). I also correct the test feature unpacking to match `prep_data`’s return structure so inference always uses `[X_deg_test, X_cat_test]` in the right order. These changes are execution-stability fixes and should be score-improving vs. the current broken run, while keeping the same model architecture, loss, and training loop. Finally, I keep the submission format strict and always write `submission.csv` with columns `key,fare_amount`.'
- What this solution (achieved 5.55942) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by ensuring the protobuf runtime is forced to the pure-Python implementation *and* by importing `google.protobuf` (and disabling C descriptors) before TensorFlow is ever imported; this is the minimal, standard Kaggle workaround and is score-neutral. I also fix the test feature unpacking bug (your `prep_data` returns `(X_cat, X_deg)`, but later code expects `X_cat_test, X_deg_test`) to ensure inference uses the correct input ordering. Finally, I update the input paths to the actual provided Kaggle paths (fallback-safe), so the script reliably reads the data and always writes a valid `submission.csv` with exactly `key,fare_amount`.'
- What this solution (achieved 5.91088) has done: 'I fix the TensorFlow import crash caused by the protobuf 6.x incompatibility by forcing the pure-Python protobuf runtime and additionally downgrading the protobuf package in-notebook to a TF-compatible 4.x version before importing TensorFlow. This is an execution-stability fix and does not change your model architecture, loss, or training loop semantics. I also fix the test feature unpacking so inference uses the correct `(X_cat, X_deg)` ordering returned by `prep_data`, which prevents silent input mixups and should improve RMSE toward your target. Finally, I keep the same submission formatting but ensure the output is always written as `submission.csv` with exactly `key,fare_amount`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["figure.figsize"] = 16, 9
from tqdm import tqdm

try:
    from ipywidgets import widgets  # noqa: F401
except Exception:
    widgets = None




## === cell 1
def read_csv_sampled(path, frac, chunksize=10**5, random_state=None):
    """
    Read a large CSV in chunks and sample a fraction from each chunk.
    Also ensures pickup_datetime is converted to pandas datetime.
    """
    samples = []
    for df_chunk in tqdm(pd.read_csv(path, chunksize=chunksize)):
        samples.append(df_chunk.sample(frac=frac, random_state=random_state))

    df = pd.concat(samples, ignore_index=True)
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    return df




## === cell 2
TRAIN_CANDIDATES = [
    "/kaggle/input/train.csv",
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
    "/kaggle/data/train.csv",
    "/kaggle/data/new-york-city-taxi-fare-prediction/train.csv",
    "../input/train.csv",
]
TEST_CANDIDATES = [
    "/kaggle/input/test.csv",
    "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
    "/kaggle/data/test.csv",
    "/kaggle/data/new-york-city-taxi-fare-prediction/test.csv",
    "../input/test.csv",
]


def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of these paths exist: {paths}")


TRAIN_PATH = _first_existing(TRAIN_CANDIDATES)
TEST_PATH = _first_existing(TEST_CANDIDATES)

print("Using TRAIN_PATH:", TRAIN_PATH)
print("Using TEST_PATH :", TEST_PATH)

df_train_sample = read_csv_sampled(TRAIN_PATH, 0.03, random_state=1989)



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

df_train_sample = df_train_sample[~is_weird].copy()



## === cell 7
df_train_sample.describe()



## === cell 8
df_test.describe()



## === cell 9
sns.histplot(
    df_train_sample["fare_amount"].apply(np.log1p).dropna(), bins=100, kde=False
)



## === cell 10
plt.scatter(
    df_train_sample["pickup_longitude"],
    df_train_sample["pickup_latitude"],
    c=df_train_sample["fare_amount"].apply(np.log1p),
    alpha=0.7,
    s=1,
    lw=0,
)
plt.xlim(*np.percentile(df_train_sample["pickup_longitude"], [1, 99]))
plt.ylim(*np.percentile(df_train_sample["pickup_latitude"], [1, 99]))
plt.colorbar()




## === cell 11
def prep_data(df, shuffle=False):
    """
    Prepare categorical time/passenger features and scaled lat/lng degrees.
    Bugfix: replace removed dt.weekofyear with ISO calendar week.
    """
    dt = df["pickup_datetime"]

    iso_week = dt.dt.isocalendar().week.astype(np.int16)

    X_cat = np.vstack(
        [
            dt.dt.hour,  # 0-23
            dt.dt.weekday + 24,  # 24-30
            dt.dt.dayofyear + 30,  # 31-396
            iso_week + 396,  # 397-449 (ISO can be 53 -> still OK)
            dt.dt.year - 2009 + 450,  # 450-...
            df["passenger_count"] + 456,  # ...
        ]
    ).T

    X_deg = df[
        ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]
    ].values.astype("float32")
    X_deg /= 180.0

    if "fare_amount" in df.columns:
        y = df["fare_amount"].values.astype("float32")

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
np.random.seed(1989)
(X_cat, X_deg), y = prep_data(df_train_sample, shuffle=True)



## === cell 13
import sys
import subprocess


def _ensure_protobuf_compat():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver

        major = int(pb_ver.split(".")[0])
        if major >= 5:
            print(
                "Detected protobuf",
                pb_ver,
                "-> installing protobuf==4.25.3 for TF compatibility",
            )
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-q",
                    "--no-deps",
                    "protobuf==4.25.3",
                ]
            )
            for m in list(sys.modules.keys()):
                if m.startswith("google.protobuf"):
                    del sys.modules[m]
    except Exception as e:
        print("protobuf compatibility check/install encountered:", repr(e))


_ensure_protobuf_compat()

try:
    from google.protobuf import descriptor as _descriptor  # noqa: F401

    if hasattr(_descriptor, "_USE_C_DESCRIPTORS"):
        _descriptor._USE_C_DESCRIPTORS = False
except Exception:
    pass

import tensorflow as tf
from tensorflow.keras import layers as lyr
from tensorflow.keras import activations as act
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K




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

    x_b = K.sin(dlng) * K.cos(lat2)
    y_b = (K.cos(lat1) * K.sin(lat2)) - (K.sin(lat1) * K.cos(lat2) * K.cos(dlng))
    b = tf.atan2(x_b, y_b) / np.pi

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
sns.histplot(err, bins=100, kde=False)
print(f"rmse: {(err**2).mean()**0.5:.3f}")
err.describe()



## === cell 19
X_cat_test, X_deg_test = prep_data(df_test)



## === cell 20
y_test = np.clip(
    model.predict([X_deg_test, X_cat_test], batch_size=512, verbose=1), 0, None
)



## === cell 21
sns.histplot(np.log1p(y), bins=100, kde=False)
sns.histplot(np.log1p(y_test[:, 0]), bins=100, kde=False)



## === cell 22
df_sub = pd.DataFrame({"key": df_test["key"].values, "fare_amount": y_test[:, 0]})
df_sub.head()



## === cell 23
df_sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_sub.shape)
print(df_sub.head())
