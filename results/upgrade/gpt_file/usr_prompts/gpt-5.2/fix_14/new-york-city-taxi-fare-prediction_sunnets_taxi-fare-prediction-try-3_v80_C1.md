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

geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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

# 5. Code solution

## === cell 0
import os
import sys
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

SEED = 1
np.random.seed(SEED)
random.seed(SEED)

TRAIN_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
TEST_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 0.001

DATASET_SIZE = 200000

print("Imports OK. Using TRAIN_PATH:", TRAIN_PATH, "TEST_PATH:", TEST_PATH)
print("Will write submission:", SUBMISSION_NAME)




## === cell 1
def clean(df):
    print(" Old size: %d" % len(df))
    df = df.dropna(how="any", axis="rows")
    print(" New size after dropna: %d" % len(df))

    has_target = "fare_amount" in df.columns

    df = df[
        ~(
            (df["dropoff_longitude"] == df["pickup_longitude"])
            & (df["dropoff_latitude"] == df["pickup_latitude"])
        )
    ]
    print(" New size after removing identical pickup/dropoff: %d" % len(df))

    df = df[
        (df["dropoff_longitude"] != 0)
        & (df["pickup_longitude"] != 0)
        & (df["dropoff_latitude"] != 0)
        & (df["pickup_latitude"] != 0)
    ]
    print(" New size after removing 0 long lat: %d" % len(df))

    MinMax = (-74.5, -72.8, 40.5, 41.8)
    df = df[
        (MinMax[0] <= df["pickup_longitude"]) & (df["pickup_longitude"] <= MinMax[1])
    ]
    df = df[
        (MinMax[0] <= df["dropoff_longitude"]) & (df["dropoff_longitude"] <= MinMax[1])
    ]
    df = df[(MinMax[2] <= df["pickup_latitude"]) & (df["pickup_latitude"] <= MinMax[3])]
    df = df[
        (MinMax[2] <= df["dropoff_latitude"]) & (df["dropoff_latitude"] <= MinMax[3])
    ]
    print(" New size after NYC lang lot: %d" % len(df))

    df = df[(df["pickup_latitude"] != 0)]
    df = df[(df["pickup_longitude"] != 0)]
    df = df[(df["dropoff_longitude"] != 0)]
    df = df[(df["dropoff_latitude"] != 0)]
    print(" New size after removing 0 long lat (redundant safety): %d" % len(df))

    df = df[((df["pickup_latitude"] - df["dropoff_latitude"]).abs() > 0.001)]
    df = df[((df["pickup_longitude"] - df["dropoff_longitude"]).abs() > 0.001)]
    print(" New size after lang - lot > 0.001: %d" % len(df))

    print(" New size after only NYC: %d" % len(df))

    if has_target:
        df = df[(0 < df["fare_amount"]) & (df["fare_amount"] <= 250)]
        print(" New size after removing outliers (fare<=250): %d" % len(df))
    else:
        print(" Skipping fare_amount outlier removal for test data.")

    df = df[(df["passenger_count"] > 0)]
    print(" New size after removing 6=>passenger_count > 0 : %d" % len(df))

    return df


def remove_datapoints_from_water(df):
    raise RuntimeError(
        "remove_datapoints_from_water() requires internet to fetch a mask image; not supported in Kaggle runtime."
    )


def late_night(row):
    h = row["hour"]
    return 1 if (h <= 3) else 0


def night(row):
    h = row["hour"]
    wd = row["weekday"]
    return 1 if ((h >= 20 or h <= 6) and wd < 5) else 0


def rush_hour(row):
    h = row["hour"]
    wd = row["weekday"]
    return 1 if ((16 <= h <= 20) and wd < 5) else 0


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)
    if dt.isna().any():
        dt2 = pd.to_datetime(df["pickup_datetime"], errors="coerce")
        dt = dt.fillna(dt2)

    df["year"] = dt.dt.year.astype("int16")
    df["month"] = dt.dt.month.astype("int8")
    df["day"] = dt.dt.day.astype("int8")
    df["hour"] = dt.dt.hour.astype("int8")
    df["weekday"] = dt.dt.weekday.astype("int8")

    df["pickup_datetime"] = dt.astype(str)

    h = df["hour"].astype("int16")
    wd = df["weekday"].astype("int16")

    df["night"] = (((h >= 20) | (h <= 6)) & (wd < 5)).astype("int8")
    df["late_night"] = (h <= 3).astype("int8")
    df["rush_hour"] = ((h >= 16) & (h <= 20) & (wd < 5)).astype("int8")
    return df


def add_coordinate_features(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]

    df["latdiff"] = (lat1 - lat2).abs()
    df["londiff"] = (lon1 - lon2).abs()
    return df


def add_distances_features(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]

    df["manhattan"] = manhattan(lat1, lon1, lat2, lon2)
    df["distance"] = np.sqrt(np.abs(lon1 - lon2) ** 2 + np.abs(lat1 - lat2) ** 2)
    return df


def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    pred = np.asarray(prediction).reshape(-1)
    df = pd.DataFrame({id_column: raw_test[id_column].values, prediction_column: pred})
    df[[id_column, prediction_column]].to_csv(file_name, index=False)
    print("Output complete:", file_name, "rows:", len(df))


def plot_loss_accuracy_rmse(history):
    print("plot_loss_accuracy_rmse skipped (no Keras History object).")




## === cell 2
datatypes = {
    "key": "str",
    "fare_amount": "float32",
    "pickup_datetime": "str",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}


def read_train_random_sample_csv(path, n_rows, seed, dtype, usecols, chunksize=500_000):
    rng = np.random.RandomState(seed)

    reservoir = None
    filled = 0
    total_seen = (
        0  # number of rows seen so far (1-based for reservoir replacement math)
    )

    for chunk in pd.read_csv(path, usecols=usecols, dtype=dtype, chunksize=chunksize):
        m = len(chunk)
        if m == 0:
            continue

        if reservoir is None:
            take = min(n_rows, m)
            reservoir = chunk.iloc[:take].copy().reset_index(drop=True)
            filled = take
            total_seen += m
            if filled < n_rows:
                continue
            start_idx = take
        else:
            start_idx = 0

        if filled >= n_rows:
            for i in range(start_idx, m):
                total_seen += 1
                j = rng.randint(0, total_seen)
                if j < n_rows:
                    reservoir.iloc[j] = chunk.iloc[i].values
        else:
            need = n_rows - filled
            take = min(need, m - start_idx)
            add = chunk.iloc[start_idx : start_idx + take].copy()
            reservoir = pd.concat([reservoir, add], ignore_index=True)
            filled = len(reservoir)
            total_seen += m - start_idx

        if reservoir is not None and filled >= n_rows and start_idx == 0:
            pass

    if reservoir is None:
        return pd.DataFrame(columns=usecols)

    if len(reservoir) > n_rows:
        reservoir = reservoir.sample(n=n_rows, random_state=seed).reset_index(drop=True)
    return reservoir


trainKaggle = read_train_random_sample_csv(
    TRAIN_PATH,
    n_rows=DATASET_SIZE,
    seed=SEED,
    dtype=datatypes,
    usecols=[
        "key",
        "fare_amount",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
)

testKaggle = pd.read_csv(
    TEST_PATH,
    dtype={
        "key": "str",
        "pickup_datetime": "str",
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "uint8",
    },
)

print(
    "Loaded trainKaggle columns:",
    list(trainKaggle.columns),
    "shape:",
    trainKaggle.shape,
)
print(
    "Loaded testKaggle columns:", list(testKaggle.columns), "shape:", testKaggle.shape
)



## === cell 3
train_df, test_df = train_test_split(trainKaggle, test_size=10000, random_state=SEED)



## === cell 4
print("testKaggle Size %d" % len(testKaggle))
print("train_df Size %d" % len(train_df))
print("test_df Size %d" % len(test_df))



## === cell 5
train_df.describe()



## === cell 6
test_df.describe()



## === cell 7
print("train_df clean")
train_df = clean(train_df)
test_df = clean(test_df)

print(
    "testKaggle clean (for modeling only; will re-align to original test order for submission)"
)
testKaggle_raw = testKaggle.copy()
testKaggle_cleaned = clean(testKaggle)



## === cell 8
train_df



## === cell 9
print("train_df add_time_features")
train_df = add_time_features(train_df)
print("test_df add_time_features")
test_df = add_time_features(test_df)

print("testKaggle_cleaned add_time_features")
testKaggle_cleaned = add_time_features(testKaggle_cleaned)



## === cell 10
train_df.describe()



## === cell 11
print("train_df add_coordinate_features")
train_df = add_coordinate_features(train_df)
print("test_df add_coordinate_features")
test_df = add_coordinate_features(test_df)

print("testKaggle_cleaned add_coordinate_features")
testKaggle_cleaned = add_coordinate_features(testKaggle_cleaned)



## === cell 12
train_df.describe()



## === cell 13
print("train_df add_distances_features")
train_df = add_distances_features(train_df)
print("test_df add_distances_features")
test_df = add_distances_features(test_df)

print("testKaggle_cleaned add_distances_features")
testKaggle_cleaned = add_distances_features(testKaggle_cleaned)
print("Done with Adding features")



## === cell 14
train_df.describe()



## === cell 15
try:
    plot = train_df.iloc[:2000].plot.scatter("latdiff", "londiff")
    plot = train_df.iloc[:2000].plot.scatter("fare_amount", "passenger_count")
    plot = train_df.iloc[:2000].plot.scatter("fare_amount", "year")
    plot = train_df.iloc[:2000].plot.scatter("fare_amount", "month")
    plot = train_df.iloc[:2000].plot.scatter("fare_amount", "day")
    plot = train_df.iloc[:2000].plot.scatter("fare_amount", "hour")
    plot = train_df.iloc[:2000].plot.scatter("fare_amount", "weekday")
    plot = train_df.iloc[:2000].plot.scatter("fare_amount", "night")
    plot = train_df.iloc[:2000].plot.scatter("fare_amount", "late_night")
    plot = train_df.iloc[:2000].plot.scatter("fare_amount", "rush_hour")
    plt.show()
except Exception as e:
    print("Plotting skipped:", repr(e))



## === cell 16
dropped_columns = [
    "pickup_datetime",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "key",
]

train_df = train_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)
testKaggle_clean = testKaggle_cleaned.drop(dropped_columns, axis=1)

print("Done with dropped_columns")



## === cell 17
train_df.shape



## === cell 18
train_df.describe()



## === cell 19
test_df.describe()



## === cell 20
train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=SEED)



## === cell 21
train_df_main = train_df
validation_df_main = validation_df



## === cell 22
validation_df.describe()



## === cell 23
train_labels = train_df["fare_amount"].values
validation_labels = validation_df["fare_amount"].values
test_labels = test_df["fare_amount"].values

train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)
test_df = test_df.drop(["fare_amount"], axis=1)

print("Done with Labels")



## === cell 24
test_labels



## === cell 25
train_df.describe()



## === cell 26
validation_df.describe()



## === cell 27
test_df.describe()



## === cell 28
train_df_scaled = train_df.values
validation_df_scaled = validation_df.values
test_scaled = test_df.values
testKaggle_scaled = testKaggle_clean.values



## === cell 29
test_scaled



## === cell 30
model = RandomForestRegressor(
    n_estimators=300,
    random_state=SEED,
    n_jobs=-1,
    min_samples_leaf=1,
    max_features="sqrt",
)

train_labels_log = np.log1p(train_labels.astype("float64"))
validation_labels_f = validation_labels.astype("float64")
test_labels_f = test_labels.astype("float64")

print(
    "Training RandomForestRegressor on:",
    train_df_scaled.shape,
    "features:",
    list(train_df.columns),
    "(target = log1p(fare_amount))",
)
model.fit(train_df_scaled, train_labels_log)

val_pred_log = model.predict(validation_df_scaled)
val_pred = np.expm1(val_pred_log)
val_pred = np.maximum(val_pred, 0.0)
val_rmse = mean_squared_error(validation_labels_f, val_pred, squared=False)
print("Validation RMSE (in $ after expm1):", val_rmse)



## === cell 31
try:
    print("Model visualization skipped (sklearn model).")
except Exception as e:
    print("Model visualization skipped:", repr(e))



## === cell 32
plot_loss_accuracy_rmse(None)



## === cell 33
train_pred_log = model.predict(train_df_scaled)
train_pred = np.expm1(train_pred_log)
train_pred = np.maximum(train_pred, 0.0)
train_rmse = mean_squared_error(
    train_labels.astype("float64"), train_pred, squared=False
)
print("Train RMSE (in $ after expm1):", train_rmse)



## === cell 34
val_pred_log = model.predict(validation_df_scaled)
val_pred = np.expm1(val_pred_log)
val_pred = np.maximum(val_pred, 0.0)
val_rmse = mean_squared_error(validation_labels_f, val_pred, squared=False)
print("Validation RMSE (in $ after expm1):", val_rmse)



## === cell 35
test_pred_log = model.predict(test_scaled)
test_pred = np.expm1(test_pred_log)
test_pred = np.maximum(test_pred, 0.0)
test_rmse = mean_squared_error(test_labels_f, test_pred, squared=False)
print("Test (holdout from train) RMSE (in $ after expm1):", test_rmse)



## === cell 36
validation_predictions = val_pred.flatten()

plt.scatter(validation_labels, validation_predictions, s=5, alpha=0.3)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.axis("equal")
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot(
    [validation_predictions.min(), validation_predictions.max()],
    [validation_predictions.min(), validation_predictions.max()],
    "k--",
    lw=2,
)
plt.show()



## === cell 37
test_predictions = test_pred.flatten()

plt.scatter(test_labels, test_predictions, s=5, alpha=0.3)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.axis("equal")
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot(
    [test_predictions.min(), test_predictions.max()],
    [test_predictions.min(), test_predictions.max()],
    "k--",
    lw=2,
)
plt.show()



## === cell 38
print(np.argmax(test_predictions))
print(test_predictions[np.argmax(test_predictions)])
print(test_labels[np.argmax(test_predictions)])
test_df.iloc[np.argmax(test_predictions)]



## === cell 39
print(np.argmin(test_predictions))
print(test_predictions[np.argmin(test_predictions)])
print(test_labels[np.argmin(test_predictions)])
test_df.iloc[np.argmin(test_predictions)]



## === cell 40
fig, ax = plt.subplots()
ax.scatter(test_labels, test_predictions, s=5, alpha=0.3)
ax.plot(
    [test_labels.min(), test_labels.max()],
    [test_labels.min(), test_labels.max()],
    "k--",
    lw=2,
)
ax.set_xlabel("Measured")
ax.set_ylabel("Predicted")
plt.show()



## === cell 41
plt.figure(figsize=(20, 10))
plt.plot(validation_labels[:100])
plt.plot(validation_predictions[:100])
plt.title("Prediction vs Actual")
plt.ylabel("Fare Amount")
plt.xlabel("Transaction")
plt.legend(["Actual", "prediction"], loc="upper right")
plt.show()

validation_df_scaled[52]
validation_df.iloc[52]



## === cell 42
plt.figure(figsize=(20, 10))
plt.plot(test_labels[:100])
plt.plot(test_predictions[:100])
plt.title("Prediction vs Actual")
plt.ylabel("Fare Amount")
plt.xlabel("Transaction")
plt.legend(["Actual", "prediction"], loc="upper right")
plt.show()



## === cell 43
error = validation_predictions - validation_labels
plt.hist(error, bins=100)
plt.xlabel("Prediction Error")
_ = plt.ylabel("Count")
plt.show()



## === cell 44
error = test_predictions - test_labels
plt.hist(error, bins=50)
plt.xlabel("Prediction Error")
_ = plt.ylabel("Count")
plt.show()



## === cell 45
errorGreaterZero = error[(np.logical_or(error <= -1, error >= 1))]
print(len(error))
print(len(errorGreaterZero))

plt.hist(errorGreaterZero, bins=100)
plt.xlabel("Prediction Error (|error|>=1)")
_ = plt.ylabel("Count")
plt.show()



## === cell 46
predictionKaggle_log = model.predict(testKaggle_scaled)
predictionKaggle_cleaned = np.expm1(predictionKaggle_log)
predictionKaggle_cleaned = np.maximum(predictionKaggle_cleaned, 0.0)

fallback = (
    float(np.median(predictionKaggle_cleaned))
    if len(predictionKaggle_cleaned)
    else 11.35
)
pred_map = pd.Series(
    predictionKaggle_cleaned.astype("float32"), index=testKaggle_cleaned["key"].values
)

predictionKaggle_full = testKaggle_raw["key"].map(pred_map).astype("float32")
predictionKaggle_full = predictionKaggle_full.fillna(fallback).values
predictionKaggle_full = np.maximum(predictionKaggle_full, 0.0)

output_submission(
    testKaggle_raw, predictionKaggle_full, "key", "fare_amount", SUBMISSION_NAME
)

sub = pd.read_csv(SUBMISSION_NAME)
print(sub.head())
print("Submission shape:", sub.shape)
print("Columns:", list(sub.columns))
print("Saved submission to:", os.path.abspath(SUBMISSION_NAME))
assert (
    sub.shape[0] == testKaggle_raw.shape[0]
), "Submission rowcount must match test.csv"
assert list(sub.columns) == [
    "key",
    "fare_amount",
], "Submission columns must be exactly: key,fare_amount"
