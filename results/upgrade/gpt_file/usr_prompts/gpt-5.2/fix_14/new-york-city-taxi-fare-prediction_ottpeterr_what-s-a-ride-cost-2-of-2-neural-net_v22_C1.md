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
protobuf==6.33.0
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

26.87094

# 6. Current score

6.26453

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 175.91148) has done: 'I fix the immediate pandas API breakages (`any(1)` and `.drop` patterns) so chunk cleaning runs and `X` is created. Then I update the TensorFlow/Keras session/config code that is incompatible with TF 2.18/Keras 3, while keeping the exact same model architecture and training loop semantics. Finally, I make the train/test preprocessing consistent (dropping `pickup_datetime` for both, keeping the same normalized columns, and safely applying min-max scaling) so inference works and a valid `submission.csv` is written.'
- What this solution (achieved 514.70482) has done: 'I fix the runtime crash happening at the TensorFlow/Keras import by aligning the protobuf runtime to a compatible implementation inside the Kaggle environment (this specific `MessageFactory.GetPrototype` error is a known protobuf/python implementation mismatch). I also make the input paths robust by preferring the competition subfolder under `/kaggle/input/` (your current paths point to non-existent files), ensuring the script can actually read train/test. Finally, to move RMSE down toward the target with minimal semantic change, I increase `num_epochs` from 1 to 3 (same architecture/loss/optimizer/training loop) so the model trains beyond the severely underfit baseline.'
- What this solution (achieved 22.7658) has done: 'I fix the TensorFlow/Keras import crash by forcing the pure-Python protobuf implementation *before* any TensorFlow/Keras import happens, and by restarting the interpreter once if needed so the env var actually takes effect (this is the root cause of the `MessageFactory.GetPrototype` error). I also make the train row counting robust by subtracting the header line so the chunk progress math is correct and avoid edge cases. Finally, to move RMSE substantially toward the target without changing the model architecture/loss/training loop, I load more than a single training chunk (still chunked, same cleaning/feature engineering) so the network trains on a representative sample rather than ~2M rows only, which is the likely reason for the very poor current score.'
- What this solution (achieved 8.16846) has done: 'I fix the TensorFlow/Keras import crash (`MessageFactory.GetPrototype`) by enforcing the pure-Python protobuf implementation and disabling the C++ protobuf backend *before* any TensorFlow import, which is the root cause in this environment. To make that enforcement reliable in Kaggle notebooks, I add a safe “restart-once” mechanism so the env vars are applied from a fresh interpreter if needed. The model architecture, training loop, preprocessing, and submission format remain unchanged; the fix is runtime-stability focused and should restore the score you previously achieved (since the current code cannot run to generate a submission). Finally, I keep file paths and the `submission.csv` writing exactly as required.'
- What this solution (achieved 202.49333) has done: 'I fix the TensorFlow/Keras import crash (`MessageFactory.GetPrototype`) by enforcing the pure-Python protobuf backend *and* disabling the C++ protobuf backend before any TF/Keras import, plus a safe one-time restart so the env vars reliably take effect in Kaggle. I also make the TF/Keras import use `tf.keras` (still the same Sequential/Dense/Dropout/BatchNorm model and training loop semantics), which avoids Keras 3 / TF integration pitfalls in this environment. These changes are runtime/stability-focused and should restore end-to-end execution and a valid `submission.csv` without intentionally changing the model/training behavior. No changes are made to feature engineering, architecture, loss, optimizer, epochs, or submission formatting.'
- What this solution (achieved 59.78829) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by ensuring the protobuf pure-Python backend is applied from a fresh interpreter before any TensorFlow-related modules are loaded; the current restart check is too narrow and can miss cases where TF/protobuf were already partially imported. This is a runtime-only change and preserves your model architecture, preprocessing, training loop, and submission formatting. Once TF imports reliably, the script run end-to-end and write a valid `submission.csv`. I also keep the existing paths and chunked loading logic unchanged.'
- What this solution (achieved 123.63599) has done: 'I fix the TensorFlow/protobuf crash by enforcing the pure-Python protobuf backend *before* any protobuf/TensorFlow import and by making the restart check robust (it currently re-execs only when some modules are already imported, which misses the common failing case). This is a runtime-only change and keeps your model, preprocessing, training loop, and submission formatting identical. After TF imports reliably, the script run end-to-end and write a valid `submission.csv` with the required `key,fare_amount` columns. No score-tuning changes are introduced beyond restoring successful training/inference execution.'
- What this solution (achieved 6.26453) has done: 'Your script already has the right end-to-end pieces (chunked cleaning, consistent scaling, model training, and submission writing), but “Not yielded” suggests it likely fails before writing `submission.csv` due to the forced `os.execvpe` restart (which can loop or break in Kaggle script execution). I replace the hard restart with a safe “set env vars before TF import and continue” approach that still prevents the protobuf crash, but won’t re-exec the process. To move RMSE down toward your target (lower is better) with minimal semantic change, I also (1) ensure the train/test feature columns are identical by dropping `key` from training early (it currently gets accidentally kept as a feature) and (2) include a very small set of simple, standard geo features (abs deltas) added consistently to both train and test; this keeps the same model architecture/training loop/loss while improving signal. The pipeline still run within the time limit and always write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import sys

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_CPP"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = os.environ.get("TF_CPP_MIN_LOG_LEVEL", "2")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import gc



## === cell 1
CANDIDATE_BASES = [
    "/kaggle/input/new-york-city-taxi-fare-prediction",
    "/kaggle/input/new-york-city-taxi-fare-prediction/new-york-city-taxi-fare-prediction",
    "/kaggle/input",
]

TRAIN_PATH = None
TEST_PATH = None
LABELS_PATH = None

for base in CANDIDATE_BASES:
    t = os.path.join(base, "train.csv")
    te = os.path.join(base, "test.csv")
    lab = os.path.join(base, "labels.csv")
    if TRAIN_PATH is None and os.path.exists(t):
        TRAIN_PATH = t
    if TEST_PATH is None and os.path.exists(te):
        TEST_PATH = te
    if LABELS_PATH is None and os.path.exists(lab):
        LABELS_PATH = lab

if TEST_PATH is None:
    raise FileNotFoundError(
        f"Could not find test.csv. Checked bases: {CANDIDATE_BASES}"
    )

TRAIN_SOURCE_PATH = LABELS_PATH if LABELS_PATH is not None else TRAIN_PATH
if TRAIN_SOURCE_PATH is None:
    raise FileNotFoundError(
        f"Could not find train source (labels.csv or train.csv). Checked bases: {CANDIDATE_BASES}"
    )

print(
    "TRAIN_SOURCE_PATH:",
    TRAIN_SOURCE_PATH,
    "exists:",
    os.path.exists(TRAIN_SOURCE_PATH),
)
print(
    "TRAIN_PATH:",
    TRAIN_PATH,
    "exists:",
    os.path.exists(TRAIN_PATH) if TRAIN_PATH else None,
)
print(
    "LABELS_PATH:",
    LABELS_PATH,
    "exists:",
    os.path.exists(LABELS_PATH) if LABELS_PATH else None,
)
print("TEST_PATH: ", TEST_PATH, "exists:", os.path.exists(TEST_PATH))



## === cell 2
p = subprocess.Popen(
    ["wc", "-l", TRAIN_SOURCE_PATH], stdout=subprocess.PIPE, stderr=subprocess.PIPE
)
result, err = p.communicate()
if p.returncode != 0:
    raise IOError(err.decode("utf-8", errors="ignore"))
n_rows = max(0, int(result.strip().split()[0]) - 1)
print("Detected train data rows (excluding header):", n_rows)




## === cell 3
def compute_haversine_distance(
    df,
    lat1="pickup_latitude",
    long1="pickup_longitude",
    lat2="dropoff_latitude",
    long2="dropoff_longitude",
):
    R = 3959  # radius of earth in miles
    phi1 = np.radians(df[lat1])
    phi2 = np.radians(df[lat2])

    delta_phi = np.radians(df[lat2] - df[lat1])
    delta_lambda = np.radians(df[long2] - df[long1])

    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = R * c
    df["distance"] = d.astype("float32")




## === cell 4
MIN_FARE = 2.50
MAX_FARE = 500

MIN_PASSENGER = 1
MAX_PASSENGER = 6

NYC_BOUNDS = {
    "pickup_longitude": (-74.3, -72.9),
    "dropoff_longitude": (-74.3, -72.9),
    "pickup_latitude": (40.5, 41.8),
    "dropoff_latitude": (40.5, 41.8),
}


def add_date_features(df, test=False):
    df["pickup_datetime_clone"] = df["pickup_datetime"].values
    df.pickup_datetime_clone = df.pickup_datetime_clone.str.slice(0, 16)
    df.pickup_datetime_clone = pd.to_datetime(
        df.pickup_datetime_clone, utc=True, format="%Y-%m-%d %H:%M"
    )
    df["year"] = df.pickup_datetime_clone.dt.year.astype("uint16")
    df["month"] = df.pickup_datetime_clone.dt.month.astype("uint8")
    df["day"] = df.pickup_datetime_clone.dt.day.astype("uint8")
    df["dayofweek"] = df.pickup_datetime_clone.dt.dayofweek.astype("uint8")
    df["hour"] = df.pickup_datetime_clone.dt.hour.astype("uint8")
    df["minute"] = df.pickup_datetime_clone.dt.minute.astype("uint8")
    df.drop(columns=["pickup_datetime_clone"], inplace=True)


def add_simple_geo_features(df):
    df["abs_lon_diff"] = np.abs(
        df["dropoff_longitude"] - df["pickup_longitude"]
    ).astype("float32")
    df["abs_lat_diff"] = np.abs(df["dropoff_latitude"] - df["pickup_latitude"]).astype(
        "float32"
    )


def clean_data(df, test=False):
    compute_haversine_distance(df)
    add_simple_geo_features(df)
    add_date_features(df, test)

    for col, (lo, hi) in NYC_BOUNDS.items():
        df.drop(df[(df[col] < lo) | (df[col] > hi)].index, axis=0, inplace=True)

    if not test:
        df.drop(df[df.isnull().any(axis=1)].index, axis=0, inplace=True)

        df.drop(
            df[(df.fare_amount > MAX_FARE) | (df.fare_amount < MIN_FARE)].index,
            axis=0,
            inplace=True,
        )
        df.drop(df[df.passenger_count > MAX_PASSENGER].index, axis=0, inplace=True)
        df.drop(df[df.passenger_count < MIN_PASSENGER].index, axis=0, inplace=True)

        df.drop(
            df[(df.pickup_latitude > 90) | (df.pickup_latitude < -90)].index,
            axis=0,
            inplace=True,
        )
        df.drop(
            df[(df.pickup_longitude > 180) | (df.pickup_longitude < -180)].index,
            axis=0,
            inplace=True,
        )
        df.drop(
            df[(df.dropoff_latitude > 90) | (df.dropoff_latitude < -90)].index,
            axis=0,
            inplace=True,
        )
        df.drop(
            df[(df.dropoff_longitude > 180) | (df.dropoff_longitude < -180)].index,
            axis=0,
            inplace=True,
        )

        df.drop(df[df.distance > 100].index, axis=0, inplace=True)
        df.drop(df[df.distance <= 0].index, axis=0, inplace=True)

        df.drop(
            df[(df.fare_amount / (df.distance + 1e-3)) > 80].index, axis=0, inplace=True
        )
        df.drop(
            df[(df.fare_amount / (df.distance + 1e-3)) < 0.5].index,
            axis=0,
            inplace=True,
        )




## === cell 5
traintypes = {
    "key": "str",
    "fare_amount": "float32",
    "pickup_datetime": "str",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}
cols = list(traintypes.keys())
chunksize = 2**21  # 2,097,152
total_chunk = n_rows // chunksize + (1 if n_rows % chunksize else 0)

MAX_CHUNKS_TO_LOAD = 8  # keep identical core training scale

df_list = []
i = 0

for df_chunk in pd.read_csv(
    TRAIN_SOURCE_PATH, usecols=cols, dtype=traintypes, chunksize=chunksize
):
    i += 1
    print(f"DataFrame Chunk {i:02d}/{max(total_chunk, 1)}")
    clean_data(df_chunk, test=False)
    df_chunk["key"] = df_chunk["key"].astype(str)
    df_list.append(df_chunk)
    del df_chunk
    if i >= MAX_CHUNKS_TO_LOAD:
        break

print("Complete: loaded chunks:", len(df_list))



## === cell 6
if len(df_list) == 0:
    raise RuntimeError("No training chunks were loaded; cannot continue.")
X = pd.concat(df_list, ignore_index=True)
del df_list
gc.collect()
print("Training rows after cleaning/concat:", len(X))



## === cell 7
X["pickup_datetime"] = pd.to_datetime(
    X["pickup_datetime"].str.slice(0, 16),
    utc=True,
    format="%Y-%m-%d %H:%M",
    errors="coerce",
)
X.drop(X[X["pickup_datetime"].isna()].index, axis=0, inplace=True)
print("Training rows after datetime parse:", len(X))



## === cell 8
validation_portion = 2.5 / 100
cutoff = X["pickup_datetime"].quantile(1.0 - validation_portion)
is_val = X["pickup_datetime"] >= cutoff
print("training:\t%d\nvalidation:\t%d" % ((~is_val).sum(), is_val.sum()))

val_df = X.loc[is_val].copy()
train_df = X.loc[~is_val].copy()

val_df.drop(columns=["pickup_datetime", "key"], inplace=True)
train_df.drop(columns=["pickup_datetime", "key"], inplace=True)

train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

del X
gc.collect()



## === cell 9
minmax = pd.DataFrame()
norm_train = pd.DataFrame()
norm_val = pd.DataFrame()

float32cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "distance",
    "abs_lon_diff",
    "abs_lat_diff",
]
float16cols = ["passenger_count", "year", "month", "day", "dayofweek", "hour", "minute"]

for col in float32cols:
    col_min = train_df[col].min()
    col_max = train_df[col].max()
    minmax[col] = (col_min, col_max)
    denom = col_max - col_min
    norm_train[col] = (
        ((train_df[col] - col_min) / denom).astype("float32")
        if denom != 0
        else np.zeros(len(train_df), dtype="float32")
    )
    norm_val[col] = (
        ((val_df[col] - col_min) / denom).astype("float32")
        if denom != 0
        else np.zeros(len(val_df), dtype="float32")
    )

for col in float16cols:
    col_min = train_df[col].min()
    col_max = train_df[col].max()
    minmax[col] = (col_min, col_max)
    denom = col_max - col_min
    norm_train[col] = (
        ((train_df[col] - col_min) / denom).astype("float16")
        if denom != 0
        else np.zeros(len(train_df), dtype="float16")
    )
    norm_val[col] = (
        ((val_df[col] - col_min) / denom).astype("float16")
        if denom != 0
        else np.zeros(len(val_df), dtype="float16")
    )

norm_train["fare_amount"] = train_df["fare_amount"].values
norm_val["fare_amount"] = val_df["fare_amount"].values

train_df = norm_train
val_df = norm_val
del norm_train, norm_val
gc.collect()

print(train_df.head())
train_df.info()



## === cell 10
y = train_df["fare_amount"].copy()
X = train_df.drop(columns="fare_amount").copy()

val_y = val_df["fare_amount"].copy()
val_X = val_df.drop(columns="fare_amount").copy()

del train_df, val_df
gc.collect()

X.info()



## === cell 11
import sys as _sys

ipython_vars = ["In", "Out", "exit", "quit", "get_ipython", "ipython_vars"]
sorted(
    [
        (x, _sys.getsizeof(globals().get(x)))
        for x in dir()
        if not x.startswith("_") and x not in _sys.modules and x not in ipython_vars
    ],
    key=lambda x: x[1],
    reverse=True,
)



## === cell 12
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras import metrics

tf.random.set_seed(42)
np.random.seed(42)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 13
model = Sequential()

model.add(Dense(64, input_dim=X.shape[1], activation="relu"))
model.add(Dropout(0.25))

for i in range(5):
    model.add(Dense(128, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

model.add(Dense(1))

model.compile(
    loss="mean_squared_error", optimizer="nadam", metrics=[metrics.MeanAbsoluteError()]
)



## === cell 14
num_epochs = 3
batch_size = 2**10
history = model.fit(
    X.values,
    y.values,
    validation_data=(val_X.values, val_y.values),
    shuffle=True,
    epochs=num_epochs,
    batch_size=batch_size,
    verbose=2,
)



## === cell 15
plt.figure()
plt.plot(history.history["loss"], color="blue")
plt.plot(history.history["val_loss"], color="red")
plt.legend(["Train", "Validation"], loc="upper left")
plt.ylabel("loss")
plt.xlabel("epoch")

plt.figure()
plt.plot(history.history["mean_absolute_error"], color="blue")
plt.plot(history.history["val_mean_absolute_error"], color="red")
plt.legend(["Train", "Validation"], loc="upper left")
plt.ylabel("Mean Abs. Error")
plt.xlabel("epoch")



## === cell 16
val_pred = model.predict(val_X.values[0:5], verbose=0).flatten()
print("actual: " + str(val_y.values[0:5]))
print("pred:   " + str(val_pred))



## === cell 17
testtypes = {
    "pickup_datetime": "str",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}
cols = list(testtypes.keys()) + ["key"]

test_raw = pd.read_csv(TEST_PATH, usecols=cols, dtype=testtypes)
test_raw["key"] = test_raw["key"].astype(str)
test_order = test_raw[["key"]].copy()

X_test = test_raw.copy()
del test_raw
gc.collect()

clean_data(X_test, test=True)
X_test.dropna(axis=0, how="any", inplace=True)

X_test_key = X_test["key"].astype(str).copy()

if "pickup_datetime" in X_test.columns:
    X_test.drop(columns=["pickup_datetime"], inplace=True)
X_test.drop(columns=["key"], inplace=True)

for col in float16cols:
    col_min, col_max = float(minmax[col].iloc[0]), float(minmax[col].iloc[1])
    denom = col_max - col_min
    X_test[col] = (
        ((X_test[col] - col_min) / denom).astype("float16")
        if denom != 0
        else np.zeros(len(X_test), dtype="float16")
    )

for col in float32cols:
    col_min, col_max = float(minmax[col].iloc[0]), float(minmax[col].iloc[1])
    denom = col_max - col_min
    X_test[col] = (
        ((X_test[col] - col_min) / denom).astype("float32")
        if denom != 0
        else np.zeros(len(X_test), dtype="float32")
    )

X_test = X_test[X.columns.tolist()]

pred = model.predict(X_test.values, verbose=0).flatten().astype("float32")
pred = np.clip(pred, MIN_FARE, MAX_FARE)

pred_df = pd.DataFrame({"key": X_test_key.values, "fare_amount": pred})

mean_pred = float(np.mean(pred)) if len(pred) else float((MIN_FARE + MAX_FARE) / 2.0)
results = test_order.merge(pred_df, on="key", how="left")
results["fare_amount"] = results["fare_amount"].fillna(mean_pred).astype("float32")

results.to_csv("submission.csv", index=False)
try:
    os.makedirs("/kaggle/working", exist_ok=True)
    results.to_csv("/kaggle/working/submission.csv", index=False)
except Exception as e:
    print("Warning: could not write /kaggle/working/submission.csv:", repr(e))

print(results.head())
print("Wrote submission.csv with shape:", results.shape)
print("Submission columns:", results.columns.tolist())
print("Any NaNs in fare_amount:", results["fare_amount"].isna().any())
