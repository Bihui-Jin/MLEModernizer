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

3.33144

# 6. Current score

4.45634

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 354.21517) has done: 'Diagnosis: The crash happens during `import tensorflow as tf` in cell 43, before any of your model/scaler code runs. With TensorFlow 2.18.0 and protobuf 6.33.0 installed, this is a known incompatibility that triggers `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` from protobuf internals during TensorFlow import. The minimal deterministic workaround is to force TensorFlow to use the pure-Python protobuf implementation (instead of the C++/upb one), which avoids the missing API path. This change is localized to cell 43 and preserves all downstream variables and logic used by cell 44.

Patch summary: In cell 43, set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (and version=2) via `os.environ` before importing TensorFlow/Keras. Keep the existing imports and semantics unchanged otherwise.

Updated cells: (cell 43 only)

Compatibility notes for cell k+1: Cell 44 still see `StandardScaler`, `tf`, `keras`, `layers`, and `callbacks` defined exactly as before; only the protobuf backend used during TensorFlow import changes.

Assumptions: The runtime allows setting environment variables at import-time (i.e., TensorFlow has not already been imported earlier in the kernel/session).'
- What this solution (achieved 286.42258) has done: 'Diagnosis: The crash occurs while importing TensorFlow in cell 43, triggered by an incompatibility between `tensorflow==2.18.0` and the installed `protobuf==6.33.0`, which raises `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. The environment-variable workaround in the cell is insufficient for protobuf 6.x. The minimal deterministic fix is to force TensorFlow to use the pure-Python protobuf implementation *before* importing TensorFlow, which avoids the failing C++/upb path that calls the missing API.

Patch summary: In cell 43, replace the `os.environ.setdefault(...)` calls with unconditional `os.environ[...] = ...` assignments (so they take effect even if something pre-set them), and ensure they are set before the TensorFlow import. Keep the rest of the imports and logic unchanged.

Updated cells:'
- What this solution (achieved 253.37889) has done: 'Diagnosis: The crash happens in cell 43 because it forces `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="cpp"`. In this environment (protobuf==6.33.0), the C++ protobuf runtime extension `google.protobuf.pyext._message` is not available, so importing TensorFlow triggers an ImportError when protobuf tries to load the C++ implementation. TensorFlow 2.18 works with the default Python protobuf implementation here, so the fix is to avoid forcing `"cpp"`.

Patch summary: Remove the environment override to `"cpp"` and instead ensure protobuf uses the Python implementation by clearing any conflicting settings and setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"` before importing TensorFlow. This is localized to cell 43 and preserves the rest of the logic and interfaces.

Updated cells: (cell 43 only)

Compatibility notes for cell k+1: Cell 44 expects `tf`, `keras`, `layers`, `callbacks`, and `StandardScaler` to be imported; this patch keeps those imports unchanged and only adjusts environment variables to allow the TensorFlow import to succeed.

Assumptions: The environment variable changes are sufficient to make TensorFlow import reliably with the installed protobuf runtime, and no later cells depend on protobuf being forced to the C++ implementation.'
- What this solution (achieved 287.08633) has done: 'Diagnosis: Cell 43 fails during TensorFlow import with `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`, which is a known incompatibility between TensorFlow 2.18 and `protobuf==6.x`. The current workaround (forcing pure-Python protobuf via env vars) is insufficient because TensorFlow still expects protobuf < 5 APIs. Since we cannot change installed packages, the minimal deterministic fix is to monkey-patch `google.protobuf.message_factory.MessageFactory.GetPrototype` to point to `GetMessageClass` when missing, before importing TensorFlow.

Patch summary: In cell 43 only, add a small compatibility shim that defines `MessageFactory.GetPrototype` if absent (protobuf 6), then proceed with the existing environment variable setup and TensorFlow/Keras imports unchanged.

Updated cells: Cell 43 only (no changes elsewhere).

Compatibility notes for cell k+1: All objects imported/defined in cell 43 (`tf`, `keras`, `layers`, `callbacks`, `StandardScaler`) remain available with the same names, so cell 44 run unchanged.

Assumptions: `google.protobuf` is importable (it is, given protobuf is installed) and protobuf 6 provides `MessageFactory.GetMessageClass`, which is the replacement API.'
- What this solution (achieved 312.6565) has done: 'Diagnosis: Cell 43 crashes because it tries to access/patch `google.protobuf.message_factory.MessageFactory.GetPrototype`, but in protobuf 6.x the `MessageFactory` class API has changed and may not expose `GetPrototype`/`GetMessageClass` the way this code expects, causing an immediate `AttributeError`. This protobuf monkey-patch is not required for importing/using TensorFlow in this environment and is the direct source of the crash.  
Patch summary: Remove the brittle protobuf `MessageFactory` patch and keep the intended protobuf runtime selection via environment variable before importing TensorFlow. This unblocks the TensorFlow/Keras imports while keeping all downstream variables (`tf`, `keras`, `layers`, `callbacks`, `StandardScaler`) available for cell 44 and later.  
Updated cells: Only cell 43 is modified.  
Compatibility notes for cell k+1: Cell 44 continues to work unchanged because `StandardScaler` is still imported in cell 43 and no variables used in cell 44 are removed/renamed.  
Assumptions: TensorFlow 2.18.0 works with the installed protobuf without needing the removed monkey-patch; selecting the pure-Python protobuf implementation is sufficient if any protobuf-runtime issue arises.'
- What this solution (achieved 203.1716) has done: 'Diagnosis: The crash happens while importing TensorFlow in cell 43 due to an incompatibility between TensorFlow 2.18.0 and the installed `protobuf==6.33.0`, which triggers `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` during TensorFlow’s protobuf initialization. The current workaround in the cell (forcing the pure-Python protobuf implementation) is not sufficient for protobuf v6. The minimal deterministic fix is to pin protobuf to a TF-compatible version (protobuf 4.x) before importing TensorFlow, and then proceed with the same imports/objects used by later cells.

Patch summary: In cell 43 only, install a compatible protobuf version (`protobuf<5`) at runtime, then import TensorFlow/Keras exactly as before. Keep the existing environment variable lines (they’re harmless) to avoid changing semantics, and ensure `tf`, `keras`, `layers`, `callbacks`, and `StandardScaler` remain defined for cell 44.

Updated cells:'
- What this solution (achieved 326.40929) has done: 'Your current RMSE is far worse than the target, so we should make small, legitimate changes that improve generalization without changing your overall pipeline (same features, same network, same loss). The biggest score hit is likely from a column mismatch bug: you scale `test_df[features]` where `features` was derived from `train_df.columns[2:]`, which (after feature engineering) can include `fare_amount` and/or be in the wrong order, causing incorrect test inputs. I minimally fix this by defining one explicit `FEATURE_COLS` list and using it consistently for train/valid/test, and I also apply the same basic NaN/coordinate/passenger cleaning to `test_df` (without touching labels) so the model doesn’t see out-of-domain garbage at inference. All other core modeling/training logic remains the same, and it still writes a valid submission CSV.'
- What this solution (achieved 5.06738) has done: 'Your RMSE is far above the target, so we should make small changes that legitimately improve generalization without changing your feature set, network architecture, or loss. The biggest likely issue is distribution shift from overly aggressive filtering (e.g., dropping all trips with Distance < 0.5 only in train), plus unbounded predictions (negative/huge fares) that get punished heavily by RMSE. I (1) replace the single hard Distance drop with a simple, symmetric clipping of extreme distances (keeps short trips the model must learn), and (2) clip final predictions to a reasonable non-negative range based on the training target quantiles to reduce catastrophic outliers. These are minimal, metric-aligned post/processing steps that preserve your overall pipeline and still produce a valid submission CSV.'
- What this solution (achieved 4.95326) has done: 'Your current RMSE (5.06738) is worse than the target (3.33144), so we should improve generalization with the smallest changes that don’t alter your model/feature logic. The two biggest low-risk wins here are (1) remove EarlyStopping so training is not prematurely cut off (you explicitly required no early stopping/relaxed convergence), and (2) train on all available cleaned training rows after using the holdout split only to choose a safe prediction clipping range, since more data usually lowers RMSE for this baseline NN. I keep the exact same features, scaling, network architecture, optimizer, loss, and batch size; only the training data used for the final fit and the removal of early stopping change. The script still runs end-to-end and writes `taxi_fare_submission.csv` with the required columns.'
- What this solution (achieved 4.94504) has done: 'Your current RMSE (4.95326) is worse than the target (3.33144), so we should make a small, legitimate change that improves generalization without changing your feature set, NN architecture, loss, or training loop semantics. The biggest issue is that you fit the `StandardScaler` on the full training data (`train_df`), while validating on a split (`X_valid`/`y_valid`), which leaks information into validation and can lead to worse training dynamics and a model that generalizes less well to the test set. I fit the scaler only on `X_train` and then transform `X_train`, `X_valid`, and `test_df` consistently—this preserves your model/training approach and typically improves RMSE for this baseline. I also make the scaling variables align with what you actually train on (so the model trains on `X_train` not the full dataset), keeping everything else unchanged, including epochs and prediction clipping.'
- What this solution (achieved 4.93442) has done: 'We make two minimal, metric-aligned fixes to move RMSE down toward your target without changing your feature set, NN architecture, loss, or training loop. First, we apply the same longitude/latitude “swap fix” (when longitude is mistakenly ~40+) to the test set as you already do for train; otherwise Distance and coordinates can be inconsistent at inference and inflate error. Second, we use `clipnorm` on the same SGD optimizer (still SGD, same loss/epochs/batch size) to reduce occasional gradient explosions/outlier-driven updates that tend to hurt RMSE for this baseline, which should improve stability and generalization modestly.'
- What this solution (achieved 4.58919) has done: 'You’re still substantially worse than the target RMSE, so we should make small, legitimate fixes that reduce generalization error without changing your features, network architecture, loss, or training loop. The biggest remaining issue is that you compute `Distance` on the raw `test_df` *before* applying the longitude/latitude “swap fix” and the coordinate-range cleaning; this makes `Distance` inconsistent with the cleaned coordinates used by the model at inference. I move the swap/clean/fill step earlier and then recompute `Distance` for `test_df` (and re-clip with the train quantiles) so train/test feature semantics match. Everything else (FEATURE_COLS, scaler usage, model, epochs, clipping predictions, submission writing) stays the same.'
- What this solution (achieved 4.45634) has done: 'Your current RMSE (4.58919) is worse than the target (3.33144), so we should make the smallest changes that improve generalization without changing your features, network architecture, loss, or training loop. The most impactful low-risk fix here is to remove the distribution mismatch created by imputing test coordinates to medians while *dropping* out-of-box rows in train: instead, apply the same “set to NaN then median-impute” policy to train as well (so the model learns on the same feature semantics it see at test time). This keeps the same core feature set and model, but avoids training only on “clean” rows while inferring on imputed rows, which typically hurts RMSE. I also ensure the post-imputation Distance is recomputed consistently after coordinate sanitation in both train and test.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



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
for col, lo, hi in [
    ("pickup_longitude", -75, -72),
    ("dropoff_longitude", -75, -72),
    ("pickup_latitude", 40, 42),
    ("dropoff_latitude", 40, 42),
]:
    bad = (train_df[col] < lo) | (train_df[col] > hi)
    train_df.loc[bad, col] = np.nan



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
test_df.isna().sum()



## === cell 17
test_df.describe()



## === cell 18
train_df.dtypes



## === cell 19
train_df["pickup_datetime"] = pd.to_datetime(train_df["pickup_datetime"])
test_df["pickup_datetime"] = pd.to_datetime(test_df["pickup_datetime"])




## === cell 20
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



## === cell 21
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




## === cell 22
train_df.Distance.sort_values() if "Distance" in train_df.columns else None



## === cell 23
FEATURE_COLS = [
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



## === cell 24
coord_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]
fill_map_train = (
    train_df[
        coord_cols + ["passenger_count", "Year", "Month", "Day", "Weekday", "Hour"]
    ]
    .median(numeric_only=True)
    .to_dict()
)
train_df[coord_cols] = train_df[coord_cols].fillna(
    {k: fill_map_train[k] for k in coord_cols}
)

haversine_distance(train_df)
haversine_distance(test_df)



## === cell 25
dist_lo, dist_hi = train_df["Distance"].quantile([0.001, 0.999]).tolist()
train_df["Distance"] = train_df["Distance"].clip(lower=dist_lo, upper=dist_hi)
test_df["Distance"] = test_df["Distance"].clip(lower=dist_lo, upper=dist_hi)



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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import os
import sys
import subprocess



## === cell 34
X_train, X_valid, y_train, y_valid = train_test_split(
    train_df[FEATURE_COLS], train_df["fare_amount"], test_size=0.30, random_state=42
)



## === cell 35
test_df = test_df.copy()

indx_t = test_df[(test_df.pickup_longitude >= 40)].index
test_df.loc[indx_t, ["dropoff_longitude", "dropoff_latitude"]] = test_df.loc[
    indx_t, ["dropoff_latitude", "dropoff_longitude"]
].values
test_df.loc[indx_t, ["pickup_longitude", "pickup_latitude"]] = test_df.loc[
    indx_t, ["pickup_latitude", "pickup_longitude"]
].values

for col, lo, hi in [
    ("pickup_longitude", -180, 180),
    ("dropoff_longitude", -180, 180),
    ("pickup_latitude", -90, 90),
    ("dropoff_latitude", -90, 90),
]:
    bad = (test_df[col] < lo) | (test_df[col] > hi)
    test_df.loc[bad, col] = np.nan

for col, lo, hi in [
    ("pickup_longitude", -75, -72),
    ("dropoff_longitude", -75, -72),
    ("pickup_latitude", 40, 42),
    ("dropoff_latitude", 40, 42),
]:
    bad = (test_df[col] < lo) | (test_df[col] > hi)
    test_df.loc[bad, col] = np.nan

test_df.loc[test_df["passenger_count"] == 0, "passenger_count"] = np.nan

fill_map = train_df[FEATURE_COLS].median(numeric_only=True).to_dict()
test_df[FEATURE_COLS] = test_df[FEATURE_COLS].fillna(fill_map)

haversine_distance(test_df)
test_df["Distance"] = test_df["Distance"].clip(lower=dist_lo, upper=dist_hi)



## === cell 36
try:
    import google.protobuf  # noqa: F401
    from importlib.metadata import version as _pkg_version

    _pb_ver = _pkg_version("protobuf")
    if int(_pb_ver.split(".", 1)[0]) >= 5:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"]
        )
        import importlib

        importlib.invalidate_caches()
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "protobuf<5"])

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



## === cell 37
scaler = StandardScaler()
scaled_train = scaler.fit_transform(X_train)
scaled_valid = scaler.transform(X_valid)
scaled_test = scaler.transform(test_df[FEATURE_COLS])



## === cell 38
model = keras.Sequential(
    [
        layers.Dense(512, activation="relu", input_dim=len(FEATURE_COLS)),
        layers.BatchNormalization(),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(32, activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(1),
    ]
)

optimizer = keras.optimizers.SGD(clipnorm=1.0)

model.compile(
    optimizer=optimizer, loss="mse", metrics=[tf.keras.metrics.RootMeanSquaredError()]
)



## === cell 39
history = model.fit(
    scaled_train,
    y_train.values,
    validation_data=(scaled_valid, y_valid.values),
    batch_size=256,
    epochs=50,
)



## === cell 40
prediction = model.predict(scaled_test, batch_size=256, verbose=1)



## === cell 41
prediction = prediction.ravel()



## === cell 42
fare_lo, fare_hi = train_df["fare_amount"].quantile([0.001, 0.999]).tolist()
prediction = np.clip(prediction, a_min=max(0.0, float(fare_lo)), a_max=float(fare_hi))



## === cell 43
submission = pd.DataFrame({"key": test_df["key"], "fare_amount": prediction})

submission.to_csv("taxi_fare_submission.csv", index=False)
print(submission.head())
print("Wrote taxi_fare_submission.csv with rows:", len(submission))
