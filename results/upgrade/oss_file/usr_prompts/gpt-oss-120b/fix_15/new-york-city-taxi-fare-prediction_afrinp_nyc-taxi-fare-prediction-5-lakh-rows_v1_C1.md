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

3.8

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

3.45343

# 6. Current score

None

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1252.686) has done: 'I replace the failing Keras imports with TensorFlow‑Keras, use a single StandardScaler fit on the training data (instead of separate scale calls), increase the neural‑network training length, clamp negative predictions to 0, and flatten the prediction array before writing the submission file. These fixes resolve the import error, ensure consistent feature scaling, and modestly improve the model’s RMSE, moving the score toward the target while keeping the original workflow unchanged.'
- What this solution (achieved 3267.45381) has done: 'Implemented fixes to resolve the import error, improve feature engineering, and boost model training:

- Switched to pure `keras` imports (avoids protobuf issue).
- Replaced Euclidean distance with haversine distance for a more accurate travel distance feature.
- Increased training epochs to 200 for better model convergence.
- Ensured predictions are flattened correctly before writing the submission file.'
- What this solution (achieved 163.5232) has done: 'I replace the failing direct `keras` imports with the TensorFlow‑Keras equivalents to fix the import error that stopped training. This change restores model creation and training, allowing the pipeline to generate a proper `submission.csv` and move the RMSE toward the target while keeping the original workflow unchanged.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization


def find_file(filename: str) -> str:
    """Return path to filename, first checking cwd then under /kaggle/input."""
    cwd_path = os.path.join(os.getcwd(), filename)
    if os.path.isfile(cwd_path):
        return cwd_path
    for root, _, files in os.walk("/kaggle/input"):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(f"{filename} not found in cwd or /kaggle/input")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_path = find_file("train.csv")
test_path = find_file("test.csv")
sample_path = find_file("sample_submission.csv")

df = pd.read_csv(train_path, parse_dates=["pickup_datetime"], nrows=500_000)
test = pd.read_csv(test_path, parse_dates=["pickup_datetime"])



## === cell 2
df.dropna(inplace=True)



## === cell 3
nyc_min_longitude, nyc_max_longitude = -74.3, -72.0
nyc_min_latitude, nyc_max_latitude = 40.63, 42.0

mask = (
    (df["pickup_longitude"] > nyc_min_longitude)
    & (df["pickup_longitude"] < nyc_max_longitude)
    & (df["dropoff_longitude"] > nyc_min_longitude)
    & (df["dropoff_longitude"] < nyc_max_longitude)
    & (df["pickup_latitude"] > nyc_min_latitude)
    & (df["pickup_latitude"] < nyc_max_latitude)
    & (df["dropoff_latitude"] > nyc_min_latitude)
    & (df["dropoff_latitude"] < nyc_max_latitude)
)
df = df[mask]



## === cell 4
df.loc[df["passenger_count"] == 0, "passenger_count"] = 1
df = df[(df["fare_amount"] > 0) & (df["fare_amount"] <= 100)]



## === cell 5
df["longitude_diff"] = df["dropoff_longitude"] - df["pickup_longitude"]
df["latitude_diff"] = df["dropoff_latitude"] - df["pickup_latitude"]
test["longitude_diff"] = test["dropoff_longitude"] - test["pickup_longitude"]
test["latitude_diff"] = test["dropoff_latitude"] - test["pickup_latitude"]


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


df["travel_distance"] = haversine_distance(
    df["pickup_latitude"],
    df["pickup_longitude"],
    df["dropoff_latitude"],
    df["dropoff_longitude"],
)
test["travel_distance"] = haversine_distance(
    test["pickup_latitude"],
    test["pickup_longitude"],
    test["dropoff_latitude"],
    test["dropoff_longitude"],
)

for src in [df, test]:
    src["year"] = src["pickup_datetime"].dt.year
    src["month"] = src["pickup_datetime"].dt.month
    src["day"] = src["pickup_datetime"].dt.day
    src["day_of_week"] = src["pickup_datetime"].dt.dayofweek
    src["hour"] = src["pickup_datetime"].dt.hour
    src["hour_sin"] = np.sin(2 * np.pi * src["hour"] / 24)
    src["hour_cos"] = np.cos(2 * np.pi * src["hour"] / 24)
    src.drop(columns=["pickup_datetime"], inplace=True)



## === cell 6
test_keys = test["key"].copy()
df.drop(columns=["key"], inplace=True)
test.drop(columns=["key"], inplace=True)



## === cell 7
features = df.drop(columns=["fare_amount"]).values
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

y_log = np.log1p(df["fare_amount"].values)

X = features_scaled  # already a NumPy array
del df, features, features_scaled



## === cell 8
X_train, X_val, y_train, y_val = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(10000, reshuffle_each_iteration=True)
    .batch(256)
    .prefetch(tf.data.AUTOTUNE)
)
val_ds = (
    tf.data.Dataset.from_tensor_slices((X_val, y_val))
    .batch(256)
    .prefetch(tf.data.AUTOTUNE)
)

y_train_orig = np.expm1(y_train)
y_val_orig = np.expm1(y_val)

model = Sequential()
model.add(Dense(256, activation="relu", input_dim=X_train.shape[1]))
model.add(BatchNormalization())
model.add(Dense(128, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(64, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(32, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(1))

model.compile(loss="mse", optimizer="adam", metrics=["mse"])
model.fit(train_ds, epochs=250, verbose=0)

train_pred = np.expm1(model.predict(train_ds, batch_size=256).ravel())
val_pred = np.expm1(model.predict(val_ds, batch_size=256).ravel())
train_rmse = np.sqrt(mean_squared_error(y_train_orig, train_pred))
val_rmse = np.sqrt(mean_squared_error(y_val_orig, val_pred))
print(f"Train RMSE: {train_rmse:.2f}")
print(f"Validation RMSE: {val_rmse:.2f}")



## === cell 9
test_scaled = scaler.transform(test.values)
test_ds = (
    tf.data.Dataset.from_tensor_slices(test_scaled)
    .batch(256)
    .prefetch(tf.data.AUTOTUNE)
)
test_pred = np.expm1(model.predict(test_ds, batch_size=256).ravel())
test_pred = np.clip(test_pred, a_min=0, a_max=None)



## === cell 10
submission = pd.read_csv(sample_path)
submission["fare_amount"] = test_pred
submission["key"] = test_keys.values
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
