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

# 5. Target score

4.33573

# 6. Current score

1049.24407

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 15.24834) has done: 'I fix the runtime blockers caused by Keras 3 API changes and the NYC-water-mask URL load (no internet / Matplotlib can’t read URLs directly), while keeping your model and feature logic intact. To move RMSE toward your target, I also correct the time-feature bugs (`late_night`/`night`) that currently mark almost every row as “night”, and fix the datetime parsing so it works with the dataset’s UTC timestamps. Finally, I ensure evaluation/prediction uses the same scaled feature matrices (your current `evaluate()` uses unscaled data) and that a valid `submissiontry_water.csv` is always written with `key,fare_amount`.'
- What this solution (achieved 277.57323) has done: 'I fix the runtime blocker caused by Keras 3 backend API changes by implementing RMSE using `keras.ops` instead of `keras.backend` (which no longer exposes `sqrt/mean/square` the same way). I also remove the protobuf-related import crash by switching to `tf_keras` (TensorFlow Keras) imports while keeping the exact same Sequential model architecture, optimizer, loss, and training loop semantics. To move RMSE toward your target (lower is better), I correct a critical data-loading bug where `fare_amount` was accidentally excluded (due to `usecols`), which previously caused training/evaluation to be wrong or impossible. Finally, I keep the existing scaled-feature evaluation/prediction and ensure a valid `submissiontry_water.csv` with `key,fare_amount` is always written.'
- What this solution (achieved 5.89858) has done: 'I fix the runtime crash in the very first import cell by removing the incompatible standalone `keras` import (which triggers the protobuf `MessageFactory.GetPrototype` error in this environment) while keeping your model/training code on `tf_keras` unchanged. I also fix a critical data-loading/submission alignment bug: `train.csv` was read without the `key` column, and later `clean()` can drop rows, so your features and labels can silently misalign unless we reset the index; additionally, the submission must always use the untouched `testKaggle` keys. These changes are execution/stability fixes and should also substantially improve RMSE versus the current broken/misaligned pipeline, moving it toward your target without changing the model architecture or training procedure. Finally, I add a small safety clamp for predictions (non-negative fares) and ensure the submission is written as a valid `.csv` with the required columns.'
- What this solution (achieved 131.39845) has done: 'I fix the runtime crash happening at import time by avoiding the incompatible `tf_keras`/protobuf combination and switching the model imports to `tensorflow.keras` while keeping the exact same Sequential architecture, optimizer, loss, and training loop. I also fix a key logic bug that hurts RMSE: you clean only `train_df` but not `test_df`, so evaluation happens on a different (uncleaned) distribution; I apply the same `clean()` to `test_df` (but never to `testKaggle`) and reset indices to keep labels aligned. Finally, I ensure that after cleaning, the feature matrices and label vectors stay aligned, and that the submission is still written as a valid `key,fare_amount` CSV with the original test keys.'
- What this solution (achieved 6.43844) has done: 'The import error comes from an incompatible protobuf/TensorFlow stack in this Kaggle image; the smallest safe fix is to avoid importing TensorFlow entirely and run the exact same Keras Sequential architecture/training loop using the already-installed `tf_keras` package. I also keep all your feature engineering/cleaning logic intact, but ensure the custom RMSE metric is implemented via `tf_keras.backend` so it works without TensorFlow. Finally, I keep the scaled-feature evaluation/prediction flow and guarantee the submission CSV is written with the required `key,fare_amount` columns.'
- What this solution (achieved 5.90448) has done: 'I fix the runtime import crash (`MessageFactory.GetPrototype`) by avoiding the incompatible `tf_keras` stack and using `tensorflow.keras` instead, keeping the same Sequential model, layers, optimizer, loss, and training loop. I also replace the `tf_keras.backend` RMSE metric with a TensorFlow implementation so it compiles cleanly under `tensorflow.keras`. These are execution-stability fixes and should preserve evaluation semantics while allowing training/inference to run end-to-end. The submission writing logic be kept the same and still produce `submissiontry_water.csv` with the required `key,fare_amount` columns.'
- What this solution (achieved 15.24203) has done: 'I fix the immediate runtime blocker caused by importing `tensorflow` in this environment by switching the model code to use the already-installed `tf_keras` package while preserving the exact same Sequential architecture, optimizer, and training loop. I also correct a small but impactful time-feature bug: `late_night()` currently marks almost every ride as “late night” (hour <= 3 only), so I align it with the intended window (late night: 20–3), which should legitimately reduce RMSE toward your target without changing the modeling approach. Finally, I keep the same scaling/evaluation/prediction flow and ensure the submission is always written as a valid `key,fare_amount` CSV with the original `testKaggle` keys.'
- What this solution (achieved 332.72859) has done: 'I fix the two runtime blockers preventing end-to-end execution: the protobuf-related crash caused by importing `tf_keras` at top-level, and the custom `rmse()` metric using TensorFlow ops that don’t exist on the `tf_keras` module. To keep your model and training loop intact while improving RMSE toward your target, I switch Keras imports to `tensorflow.keras` (which provides the needed TF ops for RMSE and avoids the `MessageFactory.GetPrototype` issue in this environment), and keep the same architecture/optimizer/epochs/batch size. I also make the plotting cell robust when training fails (so it won’t crash the notebook/script), and ensure the submission is always written as a valid `key,fare_amount` CSV using the original `testKaggle` keys. No changes are made to your feature engineering, cleaning logic, scaling, or training procedure beyond what’s required for correctness and stable execution.'
- What this solution (achieved 483.07041) has done: 'I fix the import-time crash (`MessageFactory.GetPrototype`) by avoiding the incompatible `tensorflow` import and switching the model code to use the already-installed `tf_keras` backend while keeping the exact same Sequential architecture, optimizer, loss, and training loop. I also make the custom RMSE metric use `tf_keras.backend` ops so it compiles/runs correctly without TensorFlow. These changes are execution blockers and should be score-neutral aside from allowing the pipeline to train properly end-to-end again (your current run can’t even start). Finally, I keep the submission writing unchanged but ensure it always produces a valid `.csv` with `key,fare_amount`.'
- What this solution (achieved 15.20088) has done: 'I fix the import-time crash (`MessageFactory.GetPrototype`) by removing the incompatible `tensorflow` import and switching the model/ops imports to the already-installed `tf_keras` package while keeping your exact same Sequential architecture, optimizer, loss, epochs, and batch size. I keep your feature engineering and cleaning logic intact, but I stop cleaning `testKaggle` (because cleaning drops rows/keys and breaks the required submission alignment) and instead only apply safe NA handling plus the same feature creation steps before scaling. Finally, I keep the same scaling/inference flow and ensure a valid `submissiontry_water.csv` is always written with exactly `key,fare_amount` and the same row count/order as `test.csv`.'
- What this solution (achieved 1049.24407) has done: 'I fix the import/runtime crash by removing the incompatible `tf_keras` dependency and switching to `tensorflow.keras`, which is available in Kaggle and matches the Sequential training logic you already have. Then I fix the custom `rmse()` metric so it uses TensorFlow ops (so `model.fit()` and `model.evaluate()` work), which is currently the direct blocker. These changes are execution/stability fixes and preserve your model architecture, feature engineering, and training loop semantics; they should also move RMSE down from the current broken/incorrect run toward your target by allowing the model to actually train and evaluate correctly end-to-end. Finally, I ensure `submissiontry_water.csv` is always written with the required `key,fare_amount` columns and aligned to the original `test.csv` keys/order.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers

TRAIN_PATH = "/kaggle/input/train.csv"
TEST_PATH = "/kaggle/input/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 1000
EPOCHS = 50
LEARNING_RATE = 0.001
DATASET_SIZE = 80000

np.random.seed(1)
tf.random.set_seed(1)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def remove_datapoints_from_water(df):
    try:
        import urllib.request
        from PIL import Image

        def lonlat_to_xy(longitude, latitude, dx, dy, BB):
            return (dx * (longitude - BB[0]) / (BB[1] - BB[0])).astype("int"), (
                dy - dy * (latitude - BB[2]) / (BB[3] - BB[2])
            ).astype("int")

        BB = (-74.5, -72.8, 40.5, 41.8)

        url = "https://aiblog.nl/download/nyc_mask-74.5_-72.8_40.5_41.8.png"
        with urllib.request.urlopen(url) as resp:
            nyc_mask = np.array(Image.open(resp))[:, :, 0] > 0.9

        pickup_x, pickup_y = lonlat_to_xy(
            df.pickup_longitude,
            df.pickup_latitude,
            nyc_mask.shape[1],
            nyc_mask.shape[0],
            BB,
        )
        dropoff_x, dropoff_y = lonlat_to_xy(
            df.dropoff_longitude,
            df.dropoff_latitude,
            nyc_mask.shape[1],
            nyc_mask.shape[0],
            BB,
        )
        idx = nyc_mask[pickup_y, pickup_x] & nyc_mask[dropoff_y, dropoff_x]
        return df[idx]
    except Exception as e:
        print(f"[WARN] Water-mask filtering skipped (reason: {type(e).__name__}: {e}).")
        return df


def clean(df):
    print(" Old size: %d" % len(df))
    df = df.dropna(how="any", axis="rows")
    print(" New size after dropna: %d" % len(df))

    df = df[
        (df["dropoff_longitude"] != df["pickup_longitude"])
        & (df["dropoff_latitude"] != df["pickup_latitude"])
    ]
    print(" New size after removing same long lat: %d" % len(df))

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
    print(" New size after only NYC: %d" % len(df))

    if "fare_amount" in df.columns:
        df = df[(0 < df["fare_amount"]) & (df["fare_amount"] <= 50)]
        print(" New size after removing outliers: %d" % len(df))

    df = df[(df["passenger_count"] > 0) & (df["passenger_count"] <= 6)]
    print(" New size after removing 6=>passenger_count > 0 : %d" % len(df))

    nyc_coord = (40.7141667, -74.0063889)
    fk_coord = (40.639722, -73.778889)
    ewr_coord = (40.6925, -74.168611)
    lga_coord = (40.77725, -73.872611)
    sol_coord = (40.6892, -74.0445)

    df = df[
        (nyc_coord[1] != df["pickup_longitude"])
        & (df["pickup_latitude"] != nyc_coord[0])
    ]
    df = df[
        (nyc_coord[1] != df["dropoff_longitude"])
        & (df["dropoff_latitude"] != nyc_coord[0])
    ]
    print(" New size after NY airport: %d" % len(df))

    df = df[
        (fk_coord[1] != df["pickup_longitude"]) & (df["pickup_latitude"] != fk_coord[0])
    ]
    df = df[
        (fk_coord[1] != df["dropoff_longitude"])
        & (df["dropoff_latitude"] != fk_coord[0])
    ]
    print(" New size after jfk airport: %d" % len(df))

    df = df[
        (ewr_coord[1] != df["pickup_longitude"])
        & (df["pickup_latitude"] != ewr_coord[0])
    ]
    df = df[
        (ewr_coord[1] != df["dropoff_longitude"])
        & (df["dropoff_latitude"] != ewr_coord[0])
    ]
    print(" New size after ewr airport: %d" % len(df))

    df = df[
        (lga_coord[1] != df["pickup_longitude"])
        & (df["pickup_latitude"] != lga_coord[0])
    ]
    df = df[
        (lga_coord[1] != df["dropoff_longitude"])
        & (df["dropoff_latitude"] != lga_coord[0])
    ]
    print(" New size after lgr airport: %d" % len(df))

    df = df[
        (sol_coord[1] != df["pickup_longitude"])
        & (df["pickup_latitude"] != sol_coord[0])
    ]
    df = df[
        (sol_coord[1] != df["dropoff_longitude"])
        & (df["dropoff_latitude"] != sol_coord[0])
    ]
    print(" New size after sol removed: %d" % len(df))

    print("Old size: %d" % len(df))
    df = remove_datapoints_from_water(df)
    print("New size: %d" % len(df))

    print(" New size: %d" % len(df))
    return df


def late_night(row):
    return 1 if ((row["hour"] >= 20) or (row["hour"] <= 3)) else 0


def night(row):
    return (
        1 if ((row["hour"] >= 20) or (row["hour"] <= 6)) and (row["weekday"] < 5) else 0
    )


def rush_hour(row):
    return (
        1
        if ((row["hour"] <= 20) and (row["hour"] >= 16)) and (row["weekday"] < 5)
        else 0
    )


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["hour"] = dt.dt.hour
    df["weekday"] = dt.dt.weekday
    df["pickup_datetime"] = dt.astype(str)

    df["night"] = df.apply(lambda x: night(x), axis=1)
    df["late_night"] = df.apply(lambda x: late_night(x), axis=1)
    df["rush_hour"] = df.apply(lambda x: rush_hour(x), axis=1)
    return df


def add_coordinate_features(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]
    df["latdiff"] = lat1 - lat2
    df["londiff"] = lon1 - lon2
    return df


def add_distances_features(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]
    df["manhattan"] = manhattan(lat1, lon1, lat2, lon2)
    df["distance"] = np.sqrt(
        np.abs(df["pickup_longitude"] - df["dropoff_longitude"]) ** 2
        + np.abs(df["pickup_latitude"] - df["dropoff_latitude"]) ** 2
    )
    return df


def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    df = pd.DataFrame(
        {
            id_column: raw_test[id_column].values,
            prediction_column: prediction.reshape(-1),
        }
    )
    df[[id_column, prediction_column]].to_csv(file_name, index=False)
    print("Output complete ->", file_name)


def plot_loss_accuracy_rmse(history):
    if history is None:
        print("[WARN] No training history to plot (history is None).")
        return

    plt.figure(figsize=(20, 10))
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper right")
    plt.show()

    if "rmse" in history.history:
        plt.figure(figsize=(20, 10))
        plt.plot(history.history["rmse"])
        plt.plot(history.history["val_rmse"])
        plt.title("Model rmse")
        plt.ylabel("rmse")
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper right")
        plt.show()




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

trainKaggle = pd.read_csv(
    TRAIN_PATH,
    nrows=DATASET_SIZE,
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



## === cell 3
train_df, test_df = train_test_split(trainKaggle, test_size=0.50, random_state=1)



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
train_df = clean(train_df).reset_index(drop=True)

print("test_df clean")
test_df = clean(test_df).reset_index(drop=True)

testKaggle = testKaggle.reset_index(drop=True)



## === cell 8
train_df.describe()



## === cell 9
print("train_df add_time_features")
train_df = add_time_features(train_df)
print("test_df add_time_features")
test_df = add_time_features(test_df)
print("testKaggle add_time_features")
testKaggle = add_time_features(testKaggle)



## === cell 10
train_df.describe()



## === cell 11
print("train_df add_coordinate_features")
train_df = add_coordinate_features(train_df)
print("test_df add_coordinate_features")
test_df = add_coordinate_features(test_df)
print("testKaggle add_coordinate_features")
testKaggle = add_coordinate_features(testKaggle)



## === cell 12
train_df.describe()



## === cell 13
print("train_df add_distances_features")
train_df = add_distances_features(train_df)
print("test_df add_distances_features")
test_df = add_distances_features(test_df)
print("testKaggle add_distances_features")
testKaggle = add_distances_features(testKaggle)
print("Done with Adding features")



## === cell 14
train_df.describe()



## === cell 15
dropped_columns = ["passenger_count", "pickup_datetime", "key"]

train_df = train_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(["passenger_count", "pickup_datetime", "key"], axis=1)
testKaggle_clean = testKaggle.drop(
    ["passenger_count", "pickup_datetime", "key"], axis=1
)

testKaggle_clean = testKaggle_clean.fillna(0)

print("Done with dropped_columns")



## === cell 16
train_df.shape



## === cell 17
train_df.describe()



## === cell 18
test_df.describe()



## === cell 19
train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=1)



## === cell 20
train_df.describe()



## === cell 21
validation_df.describe()



## === cell 22
train_labels = train_df["fare_amount"].values
validation_labels = validation_df["fare_amount"].values
test_labels = test_df["fare_amount"].values

train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)
test_df = test_df.drop(["fare_amount"], axis=1)

print("Done with Labels")



## === cell 23
scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_df)
testKaggle_scaled = scaler.transform(testKaggle_clean)




## === cell 24
def rmse(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true), axis=-1))




## === cell 25
model = Sequential()
model.add(
    Dense(
        256,
        activation="relu",
        input_dim=train_df_scaled.shape[1],
        activity_regularizer=regularizers.l1(0.01),
    )
)
model.add(BatchNormalization())
model.add(Dense(128, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(64, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(32, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(8, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(1))

adam = optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(loss="mean_squared_error", optimizer=adam, metrics=["mae", rmse, "mse"])

print("Dataset size: %s" % DATASET_SIZE)
print("Epochs: %s" % EPOCHS)
print("Learning rate: %s" % LEARNING_RATE)
print("Batch size: %s" % BATCH_SIZE)
print("Input dimension: %s" % train_df_scaled.shape[1])
print("Features used: %s" % list(train_df.columns))
model.summary()

history = model.fit(
    x=train_df_scaled,
    y=train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1,
    validation_data=(validation_df_scaled, validation_labels),
    shuffle=True,
)



## === cell 26
try:
    from IPython.display import SVG
    from tensorflow.keras.utils import model_to_dot

    SVG(model_to_dot(model, show_shapes=True).create(prog="dot", format="svg"))
except Exception as e:
    print(f"[WARN] Model visualization skipped: {type(e).__name__}: {e}")



## === cell 27
plot_loss_accuracy_rmse(history)



## === cell 28
score = model.evaluate(test_scaled, test_labels, verbose=1)
print(score)
print("Test loss:", score[0])



## === cell 29
prediction = model.predict(test_scaled, batch_size=128, verbose=1)
predictionKaggle = model.predict(testKaggle_scaled, batch_size=128, verbose=1)



## === cell 30
print("prediction shape:", prediction.shape, "test_labels shape:", test_labels.shape)



## === cell 31
predictionKaggle = np.maximum(predictionKaggle, 0.0).astype(np.float32)
output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)

print("Example prediction:", float(predictionKaggle[0]))
print("Submission head:")
print(pd.read_csv(SUBMISSION_NAME).head())
print(
    "Submission rows:", len(pd.read_csv(SUBMISSION_NAME)), "Expected:", len(testKaggle)
)
