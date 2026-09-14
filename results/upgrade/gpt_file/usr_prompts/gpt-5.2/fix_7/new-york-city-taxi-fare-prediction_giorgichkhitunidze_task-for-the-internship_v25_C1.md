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

3.41747

# 6. Current score

14.01128

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 255.03308) has done: 'I fix the import-time crash caused by an incompatibility between TensorFlow and the installed protobuf version by forcing the pure-Python protobuf implementation before importing TensorFlow. Then I fix two logic/shape issues that are severely hurting score: the `features` slice is wrong (it accidentally includes `fare_amount` and misses `Distance`), and `input_dim=features.shape[0]` is incorrect for a pandas Index (needs `len(features)`). Finally, I make the scaler/test transform use the exact same feature list used in training and ensure the submission CSV has the required columns and is written successfully.'
- What this solution (achieved 121.74875) has done: 'I fix the TensorFlow import crash by ensuring protobuf uses the pure-Python implementation early enough and by forcing a compatible protobuf major version at runtime via pip (only if needed), then restart the TF import cleanly. I also make the data paths robust to both `../input/...` and `/kaggle/input/...` layouts so the notebook runs in this environment without manual edits. Finally, I keep your feature set/model/training loop intact, but add one small stability fix (clipping negative predictions to 0) to avoid pathological RMSE inflation from impossible negative fares, and ensure the submission CSV is written with the exact required columns.'
- What this solution (achieved 12.34776) has done: 'Your RMSE is extremely high because the training data cleaning/feature engineering applies several strong filters (NYC bounding box, distance threshold, passenger_count>0) only to the training set, but not to the test set; this distribution mismatch typically causes huge errors on Kaggle for this competition. I keep your model, features, scaler, and training loop intact, but apply the same coordinate “swap” fix, NYC bounding-box filter, and passenger_count fix to `test_df` as well, and set `Distance<0.5` rides in test to a small floor value instead of dropping rows (so the submission row count stays correct). Finally, I enforce a deterministic row alignment by building the submission from `sample_submission.csv` keys merged with predictions (no accidental reordering). These are minimal, directly score-relevant changes aimed at reducing RMSE substantially toward your 3.41747 target.'
- What this solution (achieved 8.43061) has done: 'Your current score (12.35 RMSE) is far above the target (3.42), so we need a small, score-relevant improvement without changing the model/training loop. The biggest remaining issue is that your test preprocessing clips NYC coordinates **after** you already computed date features and the initial Distance once; then you recompute Distance again, but the earlier outlier/NaN handling still differs between train and test. I apply the same basic missing-value/coordinate-validity cleanup to the test set (without dropping rows), ensure the swap/clipping happens before the final Distance is computed, and add the same fare-related target cleaning on train (upper bound) to reduce extreme-label influence that typically inflates RMSE. These are minimal changes that keep your architecture, scaler, and training procedure intact while reducing train/test distribution mismatch and stabilizing the regression target.'
- What this solution (achieved 9.44552) has done: 'Your score is far worse than the target (8.43 RMSE vs 3.42), so we need a small, directly score-relevant improvement without changing the model/training loop. The biggest remaining issue is train/test preprocessing mismatch: you drop many “bad/out-of-NYC/short-trip” rows in train, but you only clip/fill in test; this makes the model see a cleaner distribution at train time than at test time. I apply the same *non-leaky* coordinate validity cleanup, swap fix, NYC bounding-box constraint, and passenger_count cleanup to the training data (before feature engineering) using the same “clip/fill instead of drop” approach you already use for test, while keeping the existing fare and Distance filtering that defines your training target distribution. This is minimal, preserves your architecture/training, and should move RMSE materially toward the target by reducing distribution shift.'
- What this solution (achieved 14.01128) has done: 'Your current RMSE (9.445) is still far above the target (3.417), so we should make one small, score-relevant change that reduces train/test mismatch without altering your model architecture or training loop. The biggest remaining issue is that you still hard-drop all “Distance < 0.5 km” rides from training, while you *keep* them in test by flooring to 0.5; this creates a distribution shift that can inflate Kaggle RMSE. I keep your existing filtering logic intact but additionally floor `train_df["Distance"]` to 0.5 (instead of dropping those rows) so the model learns the same short-trip regime it see at test time. Everything else (features, scaler, network, optimizer/loss, submission writing) remains the same and still produces a valid `taxi_fare_submission.csv`.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess
import importlib

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"


def _ensure_compatible_protobuf():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver

        major = int(pb_ver.split(".", 1)[0])
    except Exception:
        major = 999

    if major >= 5:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"]
        )
        importlib.invalidate_caches()


_ensure_compatible_protobuf()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (16, 8)
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

np.random.seed(42)
tf.random.set_seed(42)




## === cell 1
def _resolve_path(rel_path: str) -> str:
    candidates = [
        rel_path,
        rel_path.replace("../input/", "/kaggle/input/"),
        "/kaggle/input/new-york-city-taxi-fare-prediction/"
        + os.path.basename(rel_path),
        "/kaggle/input/" + os.path.basename(rel_path),
        "/kaggle/data/" + os.path.basename(rel_path),
        "/kaggle/data/new-york-city-taxi-fare-prediction/" + os.path.basename(rel_path),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return rel_path


train_path = _resolve_path("../input/new-york-city-taxi-fare-prediction/train.csv")
test_path = _resolve_path("../input/new-york-city-taxi-fare-prediction/test.csv")
sample_sub_path = _resolve_path(
    "../input/new-york-city-taxi-fare-prediction/sample_submission.csv"
)

train_df = pd.read_csv(train_path, nrows=1000000)
test_df = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_sub_path)

print("Train shape:", train_df.shape, "Test shape:", test_df.shape)
print("Train path:", train_path)
print("Test path:", test_path)
print("Sample submission shape:", sample_sub.shape)



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
train_df = train_df.drop(train_df[train_df.fare_amount > 250].index, axis=0)
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


haversine_distance(train_df)
haversine_distance(test_df)



## === cell 22
train_df.Distance.sort_values()



## === cell 23
train_df = train_df.drop(train_df[train_df.Distance < 0.5].index, axis=0)



## === cell 24
f, axes = plt.subplots(1, 2)
sns.barplot(x="passenger_count", y="fare_amount", data=train_df, ax=axes[0])
sns.scatterplot(x="passenger_count", y="fare_amount", data=train_df, ax=axes[1])



## === cell 25
f, axes = plt.subplots(1, 2)
sns.barplot(x="Year", y="fare_amount", data=train_df, ax=axes[0])
sns.scatterplot(x="Year", y="fare_amount", data=train_df, ax=axes[1])



## === cell 26
f, axes = plt.subplots(1, 2)
sns.barplot(x="Month", y="fare_amount", data=train_df, ax=axes[0])
sns.scatterplot(x="Month", y="fare_amount", data=train_df, ax=axes[1])



## === cell 27
f, axes = plt.subplots(1, 2)
sns.barplot(x="Day", y="fare_amount", data=train_df, ax=axes[0])
sns.scatterplot(x="Day", y="fare_amount", data=train_df, ax=axes[1])



## === cell 28
f, axes = plt.subplots(1, 2)
sns.barplot(x="Weekday", y="fare_amount", data=train_df, ax=axes[0])
sns.scatterplot(x="Weekday", y="fare_amount", data=train_df, ax=axes[1])



## === cell 29
f, axes = plt.subplots(1, 2)
sns.barplot(x="Hour", y="fare_amount", data=train_df, ax=axes[0])
sns.scatterplot(x="Hour", y="fare_amount", data=train_df, ax=axes[1])



## === cell 30
train_df["DistanceGroups"] = pd.qcut(train_df["Distance"], 10)



## === cell 31
f, axes = plt.subplots(1, 2)
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=70)
sns.barplot(x="DistanceGroups", y="fare_amount", data=train_df, ax=axes[0])
sns.scatterplot(x="Distance", y="fare_amount", data=train_df, ax=axes[1])



## === cell 32
train_df.columns



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
outcome = "fare_amount"



## === cell 34
for col, lo, hi in [
    ("pickup_longitude", -180, 180),
    ("dropoff_longitude", -180, 180),
    ("pickup_latitude", -90, 90),
    ("dropoff_latitude", -90, 90),
]:
    bad = (train_df[col] < lo) | (train_df[col] > hi) | (train_df[col].isna())
    if bad.any():
        train_df.loc[bad, col] = np.nan
        train_df[col] = train_df[col].fillna(train_df[col].median())

train_df["passenger_count"] = train_df["passenger_count"].clip(lower=1)

train_swap_idx = train_df[(train_df.pickup_longitude >= 40)].index
if len(train_swap_idx) > 0:
    train_df.loc[train_swap_idx, ["dropoff_longitude", "dropoff_latitude"]] = (
        train_df.loc[train_swap_idx, ["dropoff_latitude", "dropoff_longitude"]].values
    )
    train_df.loc[train_swap_idx, ["pickup_longitude", "pickup_latitude"]] = (
        train_df.loc[train_swap_idx, ["pickup_latitude", "pickup_longitude"]].values
    )

train_df["pickup_longitude"] = train_df["pickup_longitude"].clip(lower=-75, upper=-72)
train_df["dropoff_longitude"] = train_df["dropoff_longitude"].clip(lower=-75, upper=-72)
train_df["pickup_latitude"] = train_df["pickup_latitude"].clip(lower=40, upper=42)
train_df["dropoff_latitude"] = train_df["dropoff_latitude"].clip(lower=40, upper=42)

haversine_distance(train_df)

train_df["Distance"] = train_df["Distance"].clip(lower=0.5)
train_df = train_df.reset_index(drop=True)



## === cell 35
for col, lo, hi in [
    ("pickup_longitude", -180, 180),
    ("dropoff_longitude", -180, 180),
    ("pickup_latitude", -90, 90),
    ("dropoff_latitude", -90, 90),
]:
    bad = (test_df[col] < lo) | (test_df[col] > hi) | (test_df[col].isna())
    test_df.loc[bad, col] = np.nan
    test_df[col] = test_df[col].fillna(test_df[col].median())

test_swap_idx = test_df[(test_df.pickup_longitude >= 40)].index
test_df.loc[test_swap_idx, ["dropoff_longitude", "dropoff_latitude"]] = test_df.loc[
    test_swap_idx, ["dropoff_latitude", "dropoff_longitude"]
].values
test_df.loc[test_swap_idx, ["pickup_longitude", "pickup_latitude"]] = test_df.loc[
    test_swap_idx, ["pickup_latitude", "pickup_longitude"]
].values

test_df["passenger_count"] = test_df["passenger_count"].clip(lower=1)

test_df["pickup_longitude"] = test_df["pickup_longitude"].clip(lower=-75, upper=-72)
test_df["dropoff_longitude"] = test_df["dropoff_longitude"].clip(lower=-75, upper=-72)
test_df["pickup_latitude"] = test_df["pickup_latitude"].clip(lower=40, upper=42)
test_df["dropoff_latitude"] = test_df["dropoff_latitude"].clip(lower=40, upper=42)

haversine_distance(test_df)
test_df["Distance"] = test_df["Distance"].clip(lower=0.5)



## === cell 36
X_train, X_test, y_train, y_test = train_test_split(
    train_df[features], train_df[outcome], test_size=0.30, random_state=42
)



## === cell 37
scaler = StandardScaler()

scaled_train = scaler.fit_transform(X_train)
scaled_valid = scaler.transform(X_test)
scaled_test = scaler.transform(test_df[features])



## === cell 38
early_stopping = callbacks.EarlyStopping(
    min_delta=0.001,
    patience=5,
    restore_best_weights=True,
)

model = keras.Sequential(
    [
        layers.Dense(128, activation="relu", input_dim=len(features)),
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
    optimizer="adam", loss="mse", metrics=[tf.keras.metrics.RootMeanSquaredError()]
)



## === cell 39
history = model.fit(
    scaled_train,
    y_train,
    validation_data=(scaled_valid, y_test),
    batch_size=256,
    epochs=50,
    callbacks=[early_stopping],
    verbose=1,
)



## === cell 40
prediction = model.predict(scaled_test, batch_size=256, verbose=1).ravel()
prediction = np.clip(prediction, 0, None)

pred_df = pd.DataFrame({"key": test_df["key"].values, "fare_amount": prediction})
submission = sample_sub[["key"]].merge(pred_df, on="key", how="left")

if submission["fare_amount"].isna().any():
    submission["fare_amount"] = submission["fare_amount"].fillna(
        float(np.mean(prediction))
    )

submission.to_csv("taxi_fare_submission.csv", index=False)

print(submission.head())
print("Wrote taxi_fare_submission.csv with shape:", submission.shape)
print("Submission columns:", submission.columns.tolist())
print("NaNs in submission fare_amount:", submission["fare_amount"].isna().sum())
