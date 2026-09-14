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

3.10

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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
xgboost==2.0.3

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

3.49822

# 6. Current score

4.88717

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 50.12407) has done: 'The crash happens before any model code runs, during `import tensorflow as tf`, and is caused by an incompatibility between the installed `protobuf==6.33.0` and TensorFlow 2.18’s expected protobuf runtime (it triggers `MessageFactory.GetPrototype` lookup failures). The minimal, localized fix is to force TensorFlow to use the pure-Python protobuf implementation, which avoids the missing `GetPrototype` in the C++ implementation for this version combo. This must be set via an environment variable *before* importing TensorFlow. The rest of the cell (keras/layers imports) remains unchanged to preserve downstream interfaces.'
- What this solution (achieved 34.66705) has done: 'Diagnosis: The crash occurs in cell 43 while importing TensorFlow/Keras due to an incompatibility between the installed `protobuf==6.33.0` and TensorFlow 2.18’s expectation of the protobuf runtime API (it tries to call `MessageFactory.GetPrototype`, which was removed in protobuf 6). Setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` is not sufficient to restore the missing API in protobuf 6, so the import still fails. The minimal fix is to force TensorFlow to use its bundled pure-Python protobuf implementation by setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` and also pinning the protobuf Python implementation version to `3`, which restores compatibility for TensorFlow’s protobuf usage without changing any modeling logic.

Patch summary: Modify only cell 43 to set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=3` before importing TensorFlow, keeping the rest of the imports unchanged. This prevents the `MessageFactory.GetPrototype` AttributeError during TensorFlow import.

Updated cells: Only cell 43 is changed.

Compatibility notes for cell k+1: Cell 44 depends on successful imports of `tensorflow`, `keras`, and `layers` but otherwise uses only scikit-learn; the patched cell 43 preserves the same imported symbols (`tf`, `keras`, `layers`) and does not alter any variables used later.

Assumptions: The environment allows setting these environment variables at runtime before importing TensorFlow, and TensorFlow 2.18 successfully import using the pure-Python protobuf implementation with version compatibility mode 3.'
- What this solution (achieved 20.05032) has done: 'Diagnosis: The crash happens while importing TensorFlow/Keras due to an incompatibility between the installed `protobuf==6.33.0` and TensorFlow 2.18’s expected protobuf runtime API; specifically, TensorFlow hits a `MessageFactory.GetPrototype` call that no longer exists in protobuf 6.x. The current workaround in cell 43 forces the pure-Python protobuf implementation, but that does not restore the removed API, so the import still fails. The minimal deterministic fix is to force TensorFlow to use the upb implementation and the C++ descriptor pool, which avoids the deprecated `MessageFactory` path that triggers `GetPrototype`.

Patch summary: Modify only cell 43 to set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` to `"upb"` (instead of `"python"`) and avoid setting the version env var, then import TensorFlow/Keras as before. This preserves the original model/training logic and unblocks execution.

Updated cells: (cell 43 only)

Compatibility notes for cell k+1: Cell 44 expects `tf`, `keras`, and `layers` imports to be available; these names are preserved exactly. No changes to dataframes or scaler inputs/outputs are introduced.

Assumptions: The environment’s protobuf build supports the `"upb"` implementation (typical with protobuf 4+ and Kaggle/TF runtimes), and TensorFlow 2.18 can import correctly under upb in this setup.'
- What this solution (achieved 300.2581) has done: 'The crash happens while importing TensorFlow/Keras because the notebook forces `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="upb"`, which is incompatible with the installed `protobuf==6.33.0` and triggers `MessageFactory.GetPrototype` errors during TensorFlow’s protobuf initialization. The minimal fix is to stop forcing the `upb` protobuf runtime and instead force the pure-Python protobuf implementation, which avoids the missing API and is compatible with TF 2.18 + protobuf 6.x. This change is localized to cell 43 and keeps all downstream TensorFlow/Keras usage unchanged. No model/training logic is modified.'
- What this solution (achieved 15.94925) has done: 'Diagnosis: The crash occurs while importing TensorFlow/Keras in cell 43 due to an incompatibility between TensorFlow 2.18 and the installed `protobuf==6.33.0`, which triggers `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. The attempted workaround of forcing `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` is not sufficient for this protobuf version. The minimal deterministic fix is to ensure TensorFlow uses a protobuf 4.x runtime, which can be done in-notebook by downgrading protobuf before importing TensorFlow.

Patch summary: In cell 43 only, add a small pre-import step that installs a protobuf version compatible with TensorFlow 2.18 (e.g., `<5`), then proceed with the existing TensorFlow/Keras imports unchanged. This keeps the training logic and downstream variables (`tf`, `keras`, `layers`) identical for later cells.

Updated cells: Only cell 43 is modified.

Compatibility notes for cell k+1: Cell 44 expects TensorFlow to have imported successfully; this patch ensures `tf`, `keras`, and `layers` are available as before, and does not change any variables used by cell 44.

Assumptions: The environment allows `pip` installs at runtime (standard in many notebook/Kaggle-like environments). If pip is blocked, the only alternative would be changing the base environment packages, which is outside the scope of notebook-only fixes.'
- What this solution (achieved 21.73149) has done: 'Diagnosis: The crash happens during model training when Keras calls the custom loss `root_mean_squared_error`. In Keras 3, `keras.backend` no longer exposes backend math ops like `sqrt/mean/square`, so `K.sqrt(...)` raises `AttributeError`. The fix is to implement the same RMSE computation using TensorFlow ops (`tf.sqrt`, `tf.reduce_mean`, `tf.square`) which are compatible with `tf.keras` training and keep identical loss semantics.

Patch summary: Modify only cell 49 to (1) remove the `%%time` IPython magic that can break in non-notebook execution contexts, and (2) re-define `root_mean_squared_error` to use TensorFlow ops before calling `dnn_model.fit(...)`. No changes to the model, optimizer, metrics, data, or training parameters.

Updated cells: Only cell 49 is changed below.

Compatibility notes for cell k+1: `history` remains defined exactly as before, and `dnn_model` is trained successfully so `dnn_model.predict(...)` in cell 51 work unchanged.

Assumptions: TensorFlow (`tf`) is already imported in cell 43 and available in scope in cell 49 (as in the provided notebook).'
- What this solution (achieved 20.76123) has done: 'Diagnosis: The crash happens during `model.fit()` because `root_mean_squared_error` is redefined in cell 49, and Keras ends up using a backend object (`keras.api.backend`) that no longer exposes `sqrt/mean/square` in this environment (Keras 3). This leads to `AttributeError: module 'keras.api.backend' has no attribute 'sqrt'` when computing the loss. The model was already compiled in cell 46 with the earlier `root_mean_squared_error`, so redefining it in cell 49 is unnecessary and creates the incompatibility. The minimal fix is to remove the redefinition and ensure the model remains compiled with the existing loss function.

Patch summary: In cell 49, delete the redefinition of `root_mean_squared_error` and keep the training call unchanged. This preserves the model, training loop, and evaluation semantics while avoiding Keras backend API incompatibilities.

Updated cells:'
- What this solution (achieved 507.48063) has done: 'Diagnosis: The crash happens during `model.fit()` because `root_mean_squared_error()` uses `keras.backend` (K) functions like `K.sqrt`, but in Keras 3 (`keras==3.8.0`) `keras.backend` is no longer the legacy TF backend and does not expose `sqrt/mean/square` the same way. Since the model is compiled with this custom loss in cell 46, training fails when the loss is called. The minimal fix is to compute the loss using TensorFlow ops (`tf.sqrt`, `tf.reduce_mean`, `tf.square`) which are available and backend-safe.

Patch summary: Update the custom loss in the failing cell (cell 49) by defining a TF-based RMSE function and recompiling the already-built model with the same optimizer and metrics, then run `.fit()` exactly as before. This preserves the same training semantics (RMSE loss + MAE metric) while avoiding the Keras 3 backend API incompatibility.

Updated cells: Only cell 49 is modified.

Compatibility notes for cell k+1: Variables `history`, `dnn_model`, `Batch`, `test_scalin` remain unchanged and compatible; cell 51 can call `dnn_model.predict(...)` as before.

Assumptions: TensorFlow is already imported as `tf` (from cell 43) and `dnn_model` is already created/compiled in cell 46; recompiling in cell 49 is acceptable as a bug fix and keeps optimizer/loss/metrics equivalent.'
- What this solution (achieved 5.10913) has done: 'You don’t yet have a valid Kaggle score, so the priority is to (1) make the pipeline reliably run end-to-end in this environment and always write a correct `key,fare_amount` CSV, and (2) remove one key evaluation-mismatch that is likely hurting RMSE: you train the neural net on `log1p(fare)` but never include a compensating feature/target treatment beyond `expm1`, which can bias predictions upward; we keep your exact model/training loop but center the log-target by subtracting the train mean and add it back at inference. I also make the custom RMSE used at initial compile backend-safe (TensorFlow ops) to avoid Keras 3 backend pitfalls, without changing loss semantics. Finally, I make the submission exactly match the sample submission row order by merging on `key` (minimal alignment safety) while keeping paths unchanged.'
- What this solution (achieved 11.98151) has done: 'I make two minimal, score-relevant adjustments while preserving your model/training loop and features: (1) fix a small but real bug in `haversine_distance` where `dLambda` is briefly computed from `phi` (this can silently degrade the distance feature), and (2) ensure we always output finite predictions by clipping/cleaning NaNs/Infs right before creating the submission (this prevents rare numeric issues from exploding RMSE). I also set deterministic seeds for stability (not to optimize), so repeated runs are consistent when you iterate toward the target. All paths and the overall pipeline remain the same, and the code still writes `taxi_fare_submission.csv`.'
- What this solution (achieved 4.88717) has done: 'Your current RMSE (11.98) is far above the target (3.50), so we should make a small, clearly score-relevant improvement without changing the NN architecture or training loop. The biggest likely issue is that the model is trained to minimize RMSE in fare space but the output layer is unconstrained; negative/low predictions in log space can map to near-zero fares and hurt RMSE, and your hard clipping at 0.01 can also add error. I keep your exact features/model/training approach, but (1) add a tiny, standard cleanup to remove extreme/outlier “Distance” values and implausible passenger counts in training (reduces label noise strongly for this competition), and (2) replace the hard min-clip with a safer lower bound based on typical NYC minimum fare (2.5) while keeping the same upper bound. These changes are minimal, do not alter the core modeling logic, and should move the score substantially toward the target band.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os

os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(42)



## === cell 1
train_df = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/train.csv", nrows=1000000
)
test_df = pd.read_csv("../input/new-york-city-taxi-fare-prediction/test.csv")



## === cell 2
train_df.isnull().sum()



## === cell 3
train_df.dropna(axis=0, subset=["dropoff_longitude", "dropoff_latitude"], inplace=True)
train_df = train_df.reset_index(drop=True)



## === cell 4
pd.set_option("display.float_format", lambda x: "%.5f" % x)
train_df.describe()



## === cell 5
print("Number of observations out of valid range in coordinate columns:", end="\n")

print("pickup_longitude", end=": ")
print(
    (train_df.pickup_longitude < -180).sum() + (train_df.pickup_longitude > 180).sum()
)

print("pickup_latitude", end=": ")
print((train_df.pickup_latitude < -90).sum() + (train_df.pickup_latitude > 90).sum())

print("dropoff_longitude", end=": ")
print(
    (train_df.dropoff_longitude < -180).sum() + (train_df.dropoff_longitude > 180).sum()
)

print("dropoff_latitude", end=": ")
print((train_df.dropoff_latitude < -90).sum() + (train_df.dropoff_latitude > 90).sum())



## === cell 6
train_df = train_df.drop(
    train_df[
        (train_df.pickup_longitude < -180) | (train_df.pickup_longitude > 180)
    ].index,
    axis=0,
)
train_df = train_df.drop(
    train_df[(train_df.pickup_latitude < -90) | (train_df.pickup_latitude > 90)].index,
    axis=0,
)
train_df = train_df.drop(
    train_df[
        (train_df.dropoff_longitude < -180) | (train_df.dropoff_longitude > 180)
    ].index,
    axis=0,
)
train_df = train_df.drop(
    train_df[
        (train_df.dropoff_latitude < -90) | (train_df.dropoff_latitude > 90)
    ].index,
    axis=0,
)



## === cell 7
train_df.describe()



## === cell 8
train_df[(train_df.pickup_longitude >= 40)]



## === cell 9
indx = train_df[(train_df.pickup_longitude >= 40)].index
train_df.loc[indx, ["dropoff_longitude", "dropoff_latitude"]] = train_df.loc[
    indx, ["dropoff_latitude", "dropoff_longitude"]
].values
train_df.loc[indx, ["pickup_longitude", "pickup_latitude"]] = train_df.loc[
    indx, ["pickup_latitude", "pickup_longitude"]
].values



## === cell 10
train_df = train_df.drop(
    train_df[
        (train_df.pickup_longitude < -75) | (train_df.pickup_longitude > -72)
    ].index,
    axis=0,
)
train_df = train_df.drop(
    train_df[
        (train_df.dropoff_longitude < -75) | (train_df.dropoff_longitude > -72)
    ].index,
    axis=0,
)
train_df = train_df.drop(
    train_df[(train_df.pickup_latitude < 40) | (train_df.pickup_latitude > 42)].index,
    axis=0,
)
train_df = train_df.drop(
    train_df[(train_df.dropoff_latitude < 40) | (train_df.dropoff_latitude > 42)].index,
    axis=0,
)



## === cell 11
train_df.describe()



## === cell 12
train_df.passenger_count.value_counts()



## === cell 13
train_df = train_df.drop(train_df[train_df.passenger_count == 0].index, axis=0)



## === cell 14
train_df.fare_amount.sort_values(ascending=False)



## === cell 15
train_df = train_df.drop(train_df[train_df.fare_amount <= 0].index, axis=0)
train_df["fare_amount"].sort_values(ascending=False)



## === cell 16
train_df = train_df.drop(train_df[train_df.fare_amount > 250].index, axis=0)
train_df = train_df.reset_index(drop=True)



## === cell 17
test_df.isna().sum()



## === cell 18
test_df.describe()



## === cell 19
train_df.dtypes



## === cell 20
train_df["pickup_datetime"] = pd.to_datetime(train_df["pickup_datetime"])
test_df["pickup_datetime"] = pd.to_datetime(test_df["pickup_datetime"])




## === cell 21
def date_splitter(df):
    df["Year"] = df["pickup_datetime"].dt.year
    df["Month"] = df["pickup_datetime"].dt.month
    df["Day"] = df["pickup_datetime"].dt.day
    df["Weekday"] = df["pickup_datetime"].dt.dayofweek
    df["Hour"] = df["pickup_datetime"].dt.hour


date_splitter(train_df)
date_splitter(test_df)

train_df.drop(["pickup_datetime"], axis=1, inplace=True)
test_df.drop(["pickup_datetime"], axis=1, inplace=True)



## === cell 22
import math


def haversine_distance(df):
    coord = [
        "pickup_latitude",
        "pickup_longitude",
        "dropoff_latitude",
        "dropoff_longitude",
    ]

    phi1, lambda1, phi2, lambda2 = [df[i] * math.pi / 180.0 for i in coord]

    R = 6371

    dPhi = phi2 - phi1
    dLambda = lambda2 - lambda1

    a = (
        np.sin(dPhi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dLambda / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = R * c

    df["Distance"] = d


haversine_distance(train_df)
haversine_distance(test_df)



## === cell 23
train_df.Distance.sort_values()



## === cell 24
train_df = train_df.drop(train_df[train_df.Distance < 0.5].index, axis=0)



## === cell 25
train_df = train_df.drop(train_df[train_df.passenger_count > 6].index, axis=0)
train_df = train_df.drop(train_df[train_df.Distance > 100].index, axis=0)
train_df = train_df.reset_index(drop=True)



## === cell 26
sns.scatterplot(x="passenger_count", y="fare_amount", data=train_df)



## === cell 27
sns.scatterplot(x="Year", y="fare_amount", data=train_df)



## === cell 28
sns.scatterplot(x="Month", y="fare_amount", data=train_df)



## === cell 29
sns.scatterplot(x="Day", y="fare_amount", data=train_df)



## === cell 30
sns.scatterplot(x="Weekday", y="fare_amount", data=train_df)



## === cell 31
sns.scatterplot(x="Hour", y="fare_amount", data=train_df)



## === cell 32
sns.scatterplot(x="Distance", y="fare_amount", data=train_df)



## === cell 33
features = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "Year",
    "Month",
    "Day",
    "Weekday",
    "Hour",
    "Distance",
]
target_col = "fare_amount"



## === cell 34
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



## === cell 35
X_train, X_test, y_train, y_test = train_test_split(
    train_df[features], train_df[target_col], test_size=0.30, random_state=42
)



## === cell 36
import sys
import subprocess

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as _pb_ver
except Exception:
    _pb_ver = None

if (
    _pb_ver is None
    or _pb_ver.split(".", 1)[0].isdigit()
    and int(_pb_ver.split(".", 1)[0]) >= 5
):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "protobuf<5"])
    import importlib

    importlib.invalidate_caches()

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(42)



## === cell 37
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

train_scalin = scaler.fit_transform(X_train)
val_scalin = scaler.transform(X_test)
test_scalin = scaler.transform(test_df[features])




## === cell 38
def root_mean_squared_error(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))




## === cell 39
def build_and_compile_model(dim):
    model = keras.Sequential(
        [
            layers.Dense(128, activation="relu", input_dim=dim),
            layers.BatchNormalization(),
            layers.Dense(64, activation="relu"),
            layers.BatchNormalization(),
            layers.Dense(32, activation="relu"),
            layers.BatchNormalization(),
            layers.Dense(8, activation="relu"),
            layers.BatchNormalization(),
            layers.Dense(1),
        ]
    )

    model.compile(
        loss=root_mean_squared_error,
        optimizer=tf.keras.optimizers.Adam(0.01),
        metrics=["mae"],
    )
    return model




## === cell 40
dnn_model = build_and_compile_model(dim=len(features))



## === cell 41
ep_no = 10
Batch = 128



## === cell 42
y_train_log = np.log1p(y_train.astype(np.float32))
y_test_log = np.log1p(y_test.astype(np.float32))

y_log_mean = float(y_train_log.mean())
y_train_log_centered = y_train_log - y_log_mean
y_test_log_centered = y_test_log - y_log_mean


def rmse_in_fare_space_from_centered_log(y_true_centered, y_pred_centered):
    y_true_centered = tf.cast(y_true_centered, tf.float32)
    y_pred_centered = tf.cast(y_pred_centered, tf.float32)

    y_true_log = y_true_centered + tf.constant(y_log_mean, dtype=tf.float32)
    y_pred_log = y_pred_centered + tf.constant(y_log_mean, dtype=tf.float32)

    y_true_fare = tf.math.expm1(y_true_log)
    y_pred_fare = tf.math.expm1(y_pred_log)

    y_true_fare = tf.clip_by_value(y_true_fare, 0.0, 1e6)
    y_pred_fare = tf.clip_by_value(y_pred_fare, 0.0, 1e6)

    return tf.sqrt(tf.reduce_mean(tf.square(y_pred_fare - y_true_fare)))


dnn_model.compile(
    loss=rmse_in_fare_space_from_centered_log,
    optimizer=tf.keras.optimizers.Adam(0.01),
    metrics=["mae"],
)

history = dnn_model.fit(
    train_scalin,
    y_train_log_centered,
    validation_data=(val_scalin, y_test_log_centered),
    validation_steps=len(val_scalin) // Batch,
    batch_size=Batch,
    epochs=ep_no,
    verbose=1,
)



## === cell 43
prediction_log_centered = dnn_model.predict(test_scalin, batch_size=Batch, verbose=1)



## === cell 44
prediction_log = prediction_log_centered.ravel() + y_log_mean
prediction = np.expm1(prediction_log)

prediction = np.nan_to_num(prediction, nan=11.35, posinf=250.0, neginf=2.5)
prediction = np.clip(prediction, 2.5, 250.0)

submission = pd.DataFrame({"key": test_df["key"], "fare_amount": prediction})

sample_sub_path = "../input/new-york-city-taxi-fare-prediction/sample_submission.csv"
sample_sub = pd.read_csv(sample_sub_path)

submission = sample_sub[["key"]].merge(submission, on="key", how="left")

submission.to_csv("taxi_fare_submission.csv", index=False)
print("Wrote taxi_fare_submission.csv with shape:", submission.shape)
print(submission.head())
