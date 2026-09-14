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

9777.09207

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 383.02398) has done: 'I fix the runtime crash caused by an incompatibility between `tensorflow.keras` and the standalone `keras==3.x` package by using only `tf.keras` throughout. I also fix the custom RMSE loss to use TensorFlow ops instead of `keras.backend` functions that don’t exist in Keras 3, which currently prevents training. To improve RMSE toward your target (without changing the model architecture or training loop), I correct a feature-selection bug where `features = train_df.columns[2:]` mistakenly includes `fare_amount` and excludes `pickup_longitude`, causing mis-scaling/misaligned inference. Finally, I ensure the submission is written with the required columns and `.csv` suffix.'
- What this solution (achieved 373.64553) has done: 'I fix the crash in the TensorFlow import by forcing TensorFlow to use the pure-Python protobuf implementation, which avoids the `MessageFactory.GetPrototype` incompatibility seen in this Kaggle image. Then I make sure model training uses a correct validation setup (remove `validation_steps`, which is only meant for generator/`tf.data` inputs and can silently mis-handle numpy arrays), which should significantly reduce the RMSE toward your target without changing the model architecture or loss. Finally, I ensure the submission file is written as a proper `.csv` with the required `key` and `fare_amount` columns.'
- What this solution (achieved 122.43807) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by setting the protobuf environment variables before any TensorFlow/protobuf-related imports and by importing `tf.keras` only. Then I ensure the training target is a clean NumPy float32 array (to avoid dtype/object issues) and clamp negative predictions to 0 at inference time (fares can’t be negative), which is a minimal post-processing step that typically improves RMSE substantially without changing the model/training core. Finally, I keep the submission format exactly as required and verify the `.csv` is written successfully.'
- What this solution (achieved 479.01113) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by setting the protobuf environment variables before any protobuf/TensorFlow import and by force-importing `google.protobuf` first, which stabilizes TF 2.18 in this Kaggle image. I also make sure the TensorFlow import happens in the first cell so later cells don’t partially execute with a broken TF state. To move RMSE strongly toward your target without changing the model architecture or training loop, I add a minimal, standard data-cleaning filter to remove extreme outlier fares and unrealistic trips (very long distances), which otherwise dominate RMSE and yield huge errors. Finally, I keep the same submission format/filename and ensure predictions remain non-negative.'
- What this solution (achieved 503.35638) has done: 'I fix the TensorFlow import crash by using the pure-Python protobuf implementation (the current `"cpp"` setting causes the `_message` import error in this environment) and by setting those environment variables before importing TensorFlow. That unblock model creation/training so `tf` is defined and the later `NameError` issues disappear. I also keep everything else (features, cleaning, model architecture, training loop, loss) the same, and ensure the submission is always written as a valid `.csv` with `key` and `fare_amount` columns. These changes are correctness/stability focused so you can finally get a Kaggle score and then iterate toward the RMSE target.'
- What this solution (achieved 5.35491) has done: 'I fix the immediate runtime crash coming from an incompatibility between TensorFlow 2.18 and protobuf 6 by forcing protobuf to use the pure-Python implementation *and* downgrading protobuf at runtime to a TF-compatible version before importing TensorFlow. This is the minimal change that unblocks `import tensorflow as tf` in the Kaggle image you’re using, so the rest of your pipeline (feature engineering, scaling, model, training loop) can run unchanged. I also add a tiny safety guard to ensure the submission always has the correct `key,fare_amount` columns and is written with a `.csv` suffix. No model architecture, loss, or training-loop logic is altered.'
- What this solution (achieved 696.24019) has done: 'Your current RMSE (5.35491) is still above the target (3.49822), so we should make small, low-risk changes that improve generalization without changing the core model/training loop. The biggest safe gain here is to align the training objective with the competition metric: train with MSE (or Huber) while still reporting RMSE; using RMSE as the loss can distort gradients and commonly hurts regression accuracy. Second, keep the exact same features but add a minimal “sanity cleaning” step for extreme outliers (fare vs distance) which disproportionately inflate RMSE and often degrades a DNN fit. Finally, ensure the test set gets the exact same coordinate-swap correction as train (your code only fixes swapped coords in train), which reduces systematic feature mismatch and should push RMSE closer to the target.'
- What this solution (achieved 23.35104) has done: 'Your current RMSE (696) indicates the model is producing wildly mis-scaled predictions, which most commonly comes from NaNs/Infs entering the scaler/model (often caused by datetime parsing issues or invalid coordinates), or from feature mismatch between train and test after filtering. I make two minimal, score-relevant fixes: (1) ensure we drop any remaining NaNs/Infs in the exact FEATURES columns after all feature engineering (train and test), and (2) add a tiny numeric guard to replace any remaining Infs with NaNs before the drop. This keeps your model architecture, loss, optimizer, and training loop identical, but prevents the network from learning on corrupted rows and prevents the scaler/model from seeing invalid values—both of which can easily explode Kaggle RMSE. The submission writing stays the same, producing a valid `key,fare_amount` CSV.'
- What this solution (achieved 944.64019) has done: 'Your RMSE (23.35) is far above the target (3.498), and the biggest minimal, legitimate lever without changing your model/training loop is fixing a train/test distribution mismatch caused by dropping rows from `test_df` (you currently drop NaNs from test, which can remove keys and misalign predictions vs the required submission rows). I keep all feature engineering and the exact same model/optimizer/epochs, but change the “NaN/Inf handling” so that (a) train rows with NaNs are dropped (as before) while (b) test rows are never dropped—instead, test NaNs/Infs are imputed using the training feature medians. I also ensure the submission uses the original test row order and row count by keeping a separate `test_keys` series and building the submission against it. These changes are small, score-relevant, and typically reduce RMSE sharply by preventing bad/misaligned submissions and eliminating NaN-driven prediction explosions.'
- What this solution (achieved 1735.37637) has done: 'Your current RMSE (944) strongly suggests the submission rows are misaligned with the test keys, even though counts match—this can happen when the test DataFrame’s row order changes during preprocessing and you later pair predictions with the original `test_keys`. I make the smallest score-relevant fix: explicitly preserve and enforce the original test row order by setting `key` as the index for `test_df` during all feature engineering, then reindexing back to the original `test_keys` order right before scaling/prediction. This keeps the exact same model, features, loss, optimizer, and training loop, but prevents “wrong prediction for wrong key” submissions (which explode RMSE). I also add a lightweight assertion to guarantee alignment before writing the CSV.'
- What this solution (achieved 18424.24506) has done: 'Your RMSE exploded because the test-set coordinate-swap fix is currently applied using only the pickup-longitude condition, but it should mirror the train logic (swap when pickup_longitude is suspicious OR pickup_latitude is suspicious). That bug creates a systematic train/test feature mismatch (wrong coordinates and distances) and can easily produce huge errors even if key alignment is correct. I make the smallest change: adjust the `mask_t` condition in cell 9 to match the train swap condition, keeping the same feature engineering, model, loss, optimizer, epochs, and submission writing. This should move RMSE dramatically down toward your target without altering the core approach.'
- What this solution (achieved 9242.80335) has done: 'Your current RMSE is far worse than the target, which strongly suggests the test-set coordinate “swap” fix is still not mirroring the train logic and is corrupting engineered features (especially Distance) for a subset of rows. I make the smallest score-relevant change: apply the *exact same* swap condition to train and test (based on “suspicious longitude OR suspicious latitude”), and also ensure the train swap uses the same mask (instead of only `pickup_longitude >= 40`). This preserves your model, features, loss, optimizer, epochs, and submission semantics, but removes a systematic train/test mismatch that can explode RMSE. Everything else stays identical, including key alignment and the CSV output.'
- What this solution (achieved 9777.09207) has done: 'Your RMSE exploded far above target, which is consistent with a subtle but catastrophic bug in your Haversine feature: the latitude/longitude order is swapped inside `haversine_distance`, producing incorrect distances and a train/test relationship the DNN can’t learn. I make the smallest score-relevant fix by correcting the coordinate order (lon/lat) while keeping the same function, features, model, optimizer, epochs, and training loop. I also add a minimal guard to coerce the engineered `Distance` to numeric and re-run the existing NaN/Inf handling so scaling/prediction stay stable. This should move RMSE dramatically down toward your target without changing your overall approach.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as pb_ver
except Exception:
    pb_ver = "unknown"


def _major(v):
    try:
        return int(str(v).split(".", 1)[0])
    except Exception:
        return None


if _major(pb_ver) is None or _major(pb_ver) >= 5:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
    )

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers

np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow:", tf.__version__)
import google.protobuf

print("Protobuf:", google.protobuf.__version__)



## === cell 1
train_df = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/train.csv", nrows=1000000
)
test_df = pd.read_csv("../input/new-york-city-taxi-fare-prediction/test.csv")

test_keys = test_df["key"].astype(str).copy()
test_df["key"] = test_df["key"].astype(str)
test_df = test_df.set_index("key", drop=True)



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
mask_tr = (train_df["pickup_longitude"] >= 40) | (train_df["pickup_latitude"] >= 40)
train_df.loc[mask_tr, ["dropoff_longitude", "dropoff_latitude"]] = train_df.loc[
    mask_tr, ["dropoff_latitude", "dropoff_longitude"]
].values
train_df.loc[mask_tr, ["pickup_longitude", "pickup_latitude"]] = train_df.loc[
    mask_tr, ["pickup_latitude", "pickup_longitude"]
].values

mask_t = (test_df["pickup_longitude"] >= 40) | (test_df["pickup_latitude"] >= 40)
test_df.loc[mask_t, ["dropoff_longitude", "dropoff_latitude"]] = test_df.loc[
    mask_t, ["dropoff_latitude", "dropoff_longitude"]
].values
test_df.loc[mask_t, ["pickup_longitude", "pickup_latitude"]] = test_df.loc[
    mask_t, ["pickup_latitude", "pickup_longitude"]
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
train_df = train_df.drop(train_df[train_df.fare_amount > 200].index, axis=0)

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
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
    ]
    lambda1, phi1, lambda2, phi2 = [df[i] * math.pi / 180.0 for i in coord]

    R = 6371
    dPhi = phi2 - phi1
    dLambda = (
        phi2 - phi1 + (lambda2 - lambda1) - (phi2 - phi1)
    )  # no-op; preserves core logic intent
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

train_df["Distance"] = pd.to_numeric(train_df["Distance"], errors="coerce")
test_df["Distance"] = pd.to_numeric(test_df["Distance"], errors="coerce")



## === cell 22
train_df.Distance.sort_values()



## === cell 23
train_df = train_df.drop(train_df[train_df.Distance < 0.5].index, axis=0)
train_df = train_df.drop(train_df[train_df.Distance > 100].index, axis=0)



## === cell 24
fare_per_km = train_df["fare_amount"] / (train_df["Distance"] + 1e-3)
train_df = train_df.loc[(fare_per_km <= 25.0) & (fare_per_km >= 0.5)].reset_index(
    drop=True
)



## === cell 25
sns.scatterplot(x="passenger_count", y="fare_amount", data=train_df)



## === cell 26
sns.scatterplot(x="Year", y="fare_amount", data=train_df)



## === cell 27
sns.scatterplot(x="Month", y="fare_amount", data=train_df)



## === cell 28
sns.scatterplot(x="Day", y="fare_amount", data=train_df)



## === cell 29
sns.scatterplot(x="Weekday", y="fare_amount", data=train_df)



## === cell 30
sns.scatterplot(x="Hour", y="fare_amount", data=train_df)



## === cell 31
sns.scatterplot(x="Distance", y="fare_amount", data=train_df)



## === cell 32
FEATURES = [
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



## === cell 33
train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
test_df.replace([np.inf, -np.inf], np.nan, inplace=True)

before_train = len(train_df)
train_df = train_df.dropna(subset=FEATURES + ["fare_amount"]).reset_index(drop=True)
after_train = len(train_df)

train_feature_medians = train_df[FEATURES].median(numeric_only=True)

test_df[FEATURES] = test_df[FEATURES].fillna(train_feature_medians)
test_df[FEATURES] = test_df[FEATURES].fillna(0.0)

print(
    f"Dropped rows with NaN/Inf after feature engineering: train {before_train-after_train}, test 0 (imputed)"
)
print("Any NaNs left in test features:", test_df[FEATURES].isna().any().any())



## === cell 34
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



## === cell 35
X_train, X_test, y_train, y_test = train_test_split(
    train_df[FEATURES], train_df["fare_amount"], test_size=0.30, random_state=42
)



## === cell 36
scaler = StandardScaler()
train_scalin = scaler.fit_transform(X_train)
val_scalin = scaler.transform(X_test)

test_df_ordered = test_df.reindex(test_keys.values)
assert test_df_ordered.shape[0] == len(
    test_keys
), "Reindexed test must match original test key count."
assert (
    test_df_ordered.index.astype(str).values == test_keys.values
).all(), "Test row order must exactly match test_keys before prediction."

test_scalin = scaler.transform(test_df_ordered[FEATURES])

y_train = y_train.to_numpy(dtype=np.float32)
y_test = y_test.to_numpy(dtype=np.float32)




## === cell 37
def root_mean_squared_error(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))




## === cell 38
def build_and_compile_model(dim):
    model = tf.keras.Sequential(
        [
            layers.Dense(128, activation="relu", input_shape=(dim,)),
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
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(0.01),
        metrics=[root_mean_squared_error, "mae"],
    )
    return model




## === cell 39
dnn_model = build_and_compile_model(dim=len(FEATURES))



## === cell 40
ep_no = 10
Batch = 128

history = dnn_model.fit(
    train_scalin,
    y_train,
    validation_data=(val_scalin, y_test),
    batch_size=Batch,
    epochs=ep_no,
    verbose=1,
)

prediction = dnn_model.predict(test_scalin, batch_size=Batch, verbose=1).ravel()
prediction = np.clip(prediction, 0.0, None)

submission = pd.DataFrame(
    {"key": test_keys.values, "fare_amount": prediction.astype(np.float32)}
)
submission = submission[["key", "fare_amount"]]
submission.to_csv("taxi_fare_submission.csv", index=False)

print(submission.head())
print("Wrote taxi_fare_submission.csv with shape:", submission.shape)
print("fare_amount stats:", submission["fare_amount"].describe())
assert submission.shape[0] == len(
    test_keys
), "Submission row count must match test row count."
assert (
    submission["key"].astype(str).values == test_keys.values
).all(), "Submission keys must match test key order."
