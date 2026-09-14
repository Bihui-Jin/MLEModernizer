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

4.76029

# 6. Current score

5.91742

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 15.24434) has done: 'I fix the runtime errors caused by incompatibilities between your code and the Kaggle environment (Keras 3 API changes, URL image reading, and a couple of logic bugs that silently destroy feature quality). Specifically, I (1) replace the broken remote water-mask load with a safe, offline-compatible fallback, (2) correct the datetime parsing and the time-of-day feature logic, (3) update optimizer construction to the Keras 3 API while keeping the same model/fit loop, and (4) ensure the script always writes a valid `submissiontry_water.csv` with the required columns. These changes keep the same core pipeline (same feature set intention, same Sequential Dense model, same training loop) but should significantly reduce RMSE vs the current broken/low-quality feature engineering.'
- What this solution (achieved 384.00283) has done: 'I fix the immediate runtime errors by switching the RMSE metric implementation from deprecated `keras.backend` math ops to `keras.ops`, which is the supported Keras 3 API in this environment. I also fix the import-time crash (`MessageFactory.GetPrototype`) by forcing Keras to use the TensorFlow backend (compatible with `tf_keras` here) before importing Keras modules. These are correctness/stability fixes that keep your model, features, training loop, and loss the same, while enabling training to actually run—this should materially improve RMSE versus the current broken run. Finally, I keep the submission writing logic intact and ensure the output is a valid `.csv` with the required columns.'
- What this solution (achieved 113.36365) has done: 'I fix the two blockers that prevent an end-to-end run and thus prevent any submission: (1) the protobuf `MessageFactory.GetPrototype` crash triggered by importing `tf_keras` in this environment, by switching to `tensorflow.keras` (same Keras 2 API, same model/fit loop), and (2) the `MinMaxScaler` failure caused by a remaining string column (`key`) leaking into the features, by dropping `key` from train/val/test feature frames before scaling. These are correctness fixes that preserve your exact feature engineering and Dense network architecture/training loop, and allow the notebook to train, predict, and write a valid `submissiontry_water.csv`. I also keep the existing offline-safe water-mask fallback and ensure column alignment is consistent across train/val/test/Kaggle test to avoid silent feature mismatch.'
- What this solution (achieved 195.22508) has done: 'I fix the two remaining execution blockers while keeping your feature engineering, model architecture, and training loop unchanged. First, I prevent the protobuf `MessageFactory.GetPrototype` crash by setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` before importing TensorFlow (and avoid importing standalone `keras`). Second, I replace the removed `tf.keras.losses.mean_squared_error` call in TF/Keras 3 with an equivalent NumPy-based MSE sanity check (score-neutral, only affects the debug print). The script then run end-to-end and always write a valid `submissiontry_water.csv` with `key,fare_amount`.'
- What this solution (achieved 5.66175) has done: 'I make the pipeline produce a valid submission reliably by fixing the cell numbering and making the restart-on-protobuf setting safe in notebooks. To move RMSE toward your target (lower is better) without changing the model or training loop, I fix a feature-quality bug where `pickup_datetime` is being overwritten into a string (silently hurting downstream time features consistency) and replace the very slow `df.apply` time-flag creation with equivalent vectorized logic (same semantics, fewer coercion issues). I also ensure train/test get identical datetime parsing (timezone-aware) and that invalid test rows get a more stable fallback based on the **training** label distribution only (already intended, but made robust to shape mismatches). These are minimal changes that preserve your core architecture, loss, and training approach, while improving feature integrity and execution stability to yield a submission CSV.'
- What this solution (achieved 6.70637) has done: 'I fix the protobuf/TensorFlow import crash (`MessageFactory.GetPrototype`) by setting a safe environment variable (`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`) before TensorFlow loads, and by lazily importing TensorFlow only after data preparation so the notebook can run reliably end-to-end. To move RMSE toward your target (lower is better) without changing the model architecture or training loop, I make one minimal, score-relevant correction in cleaning: the “same long lat” filter currently removes many valid trips because it uses `&` instead of checking both coordinates simultaneously; I change it to require both lat AND lon to be identical. Everything else (feature engineering, scaler, Dense network, optimizer, epochs/batch size, and submission formatting) be kept the same, and the script always write `submissiontry_water.csv` with `key,fare_amount`.'
- What this solution (achieved 6.11116) has done: 'I fix the TensorFlow/protobuf import crash by setting the protobuf implementation environment variables before any TensorFlow-related import and by importing TensorFlow in a way that avoids triggering the problematic compiled-protobuf path in this Kaggle image. This is a pure stability fix that unblocks training/inference and should be score-neutral versus a successful run, but right now it’s preventing end-to-end execution. I also make the environment settings happen at the very top (cell 1) so they reliably take effect, and keep your model, features, scaler, training loop, and submission writing logic unchanged. The script still write `submissiontry_water.csv` with the required `key,fare_amount` columns.'
- What this solution (achieved 6.32139) has done: 'I fix the TensorFlow/protobuf import crash by moving the protobuf environment settings to the very top of the script (before any other imports) and by importing TensorFlow in a way that avoids triggering the compiled-protobuf path that causes `MessageFactory.GetPrototype`. This is a stability fix that restores end-to-end execution and submission writing without changing your model architecture, feature engineering, loss, or training loop. To nudge RMSE toward your target (lower is better) with minimal semantics change, I also make the train/val/test split `shuffle=True` (still deterministic via `random_state=1`), which improves generalization versus the current sequential split while keeping the same data size and training procedure. The submission file name, columns, and paths remain unchanged and a valid `.csv` always be written.'
- What this solution (achieved 6.09426) has done: 'I fix the TensorFlow/protobuf import crash (`MessageFactory.GetPrototype`) by moving the protobuf environment settings to the very top and ensuring TensorFlow is imported only after those settings take effect, plus adding a safe fallback to force the pure-Python protobuf implementation early. This is an execution blocker fix (your run currently cannot reach training/inference), and it is intended to be score-neutral relative to a successful run. To move RMSE toward your target (lower is better) with minimal semantic change, I also switch the training fit `shuffle` to `True` (still deterministic via seeds) to improve generalization without changing the model, features, loss, epochs, or batch size. Finally, I keep the submission writing unchanged but add a small guard to always produce a valid `submissiontry_water.csv` even if predictions contain NaNs/Infs.'
- What this solution (achieved 5.91742) has done: 'I fix the execution blocker in the TensorFlow import (`MessageFactory.GetPrototype`) by forcing pure-Python protobuf before *any* TensorFlow import and by importing TensorFlow only after those environment variables are set, which unblocks training/inference and preserves your existing model/training loop. I also make the submission-writing robust by ensuring the output file always ends with `.csv` (Kaggle requires that) while keeping the same filename base. These are minimal, score-neutral stability fixes; they should let the pipeline run end-to-end reliably, and any score change should come only from the model actually training and predicting correctly. No changes are made to your feature engineering, model architecture, loss, epochs, batch size, or evaluation semantics.'

# 9. Code solution

## === cell 0
import os
import sys

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

TRAIN_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
TEST_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 0.001
DATASET_SIZE = 80000

np.random.seed(1)



## === cell 1
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
    usecols=list(datatypes.keys()),  # key + all feature cols + target
)

testKaggle = pd.read_csv(
    TEST_PATH,
    dtype={k: v for k, v in datatypes.items() if k != "fare_amount"},
    usecols=[
        "key",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
)




## === cell 2
def remove_datapoints_from_water(df):
    def lonlat_to_xy(longitude, latitude, dx, dy, BB):
        return (dx * (longitude - BB[0]) / (BB[1] - BB[0])).astype("int"), (
            dy - dy * (latitude - BB[2]) / (BB[3] - BB[2])
        ).astype("int")

    BB = (-74.5, -72.8, 40.5, 41.8)

    try:
        local_mask_path = "nyc_mask-74.5_-72.8_40.5_41.8.png"
        if os.path.exists(local_mask_path):
            nyc_mask = plt.imread(local_mask_path)[:, :, 0] > 0.9
        else:
            raise FileNotFoundError("NYC mask image not available locally.")

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

        valid = (
            (pickup_x >= 0)
            & (pickup_x < nyc_mask.shape[1])
            & (dropoff_x >= 0)
            & (dropoff_x < nyc_mask.shape[1])
            & (pickup_y >= 0)
            & (pickup_y < nyc_mask.shape[0])
            & (dropoff_y >= 0)
            & (dropoff_y < nyc_mask.shape[0])
        )
        idx = np.zeros(len(df), dtype=bool)
        valid_arr = valid.values if hasattr(valid, "values") else valid
        idx[valid_arr] = (
            nyc_mask[pickup_y[valid_arr], pickup_x[valid_arr]]
            & nyc_mask[dropoff_y[valid_arr], dropoff_x[valid_arr]]
        )
        return df[idx]
    except Exception as e:
        print(
            f"Water-mask filter skipped (offline/unavailable mask): {type(e).__name__}: {e}"
        )
        return df


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)

    df["year"] = dt.dt.year.fillna(0).astype("int16")
    df["month"] = dt.dt.month.fillna(0).astype("int8")
    df["day"] = dt.dt.day.fillna(0).astype("int8")
    df["hour"] = dt.dt.hour.fillna(0).astype("int8")
    df["weekday"] = dt.dt.weekday.fillna(0).astype("int8")

    hour = df["hour"].astype("int16")
    weekday = df["weekday"].astype("int16")

    df["late_night"] = ((hour <= 3) | (hour >= 22)).astype("int8")
    df["night"] = (((hour >= 20) | (hour <= 6)) & (weekday < 5)).astype("int8")
    df["rush_hour"] = (((hour >= 16) & (hour <= 20)) & (weekday < 5)).astype("int8")
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
    if not str(file_name).lower().endswith(".csv"):
        file_name = f"{file_name}.csv"

    df = pd.DataFrame(
        {
            id_column: raw_test[id_column].values,
            prediction_column: prediction.reshape(-1),
        }
    )
    df[[id_column, prediction_column]].to_csv(file_name, index=False)
    print("Output complete:", file_name, "rows=", len(df))


def plot_loss_accuracy_rmse(history):
    plt.figure(figsize=(20, 10))
    plt.plot(history.history.get("loss", []))
    plt.plot(history.history.get("val_loss", []))
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper right")
    plt.show()

    plt.figure(figsize=(20, 10))
    if "rmse" in history.history and "val_rmse" in history.history:
        plt.plot(history.history["rmse"])
        plt.plot(history.history["val_rmse"])
        plt.title("Model rmse")
        plt.ylabel("rmse")
        plt.xlabel("epoch")
        plt.legend(["train", "validation"], loc="upper right")
        plt.show()




## === cell 3
def clean(df):
    print(" Old size: %d" % len(df))
    df = df.dropna(how="any", axis="rows")
    print(" New size after dropna: %d" % len(df))

    df = df[
        ~(
            (df["dropoff_longitude"] == df["pickup_longitude"])
            & (df["dropoff_latitude"] == df["pickup_latitude"])
        )
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


def clean_test_preserve_rows(df):
    df = df.copy()
    invalid = np.zeros(len(df), dtype=bool)

    miss = (
        df[
            [
                "pickup_longitude",
                "pickup_latitude",
                "dropoff_longitude",
                "dropoff_latitude",
                "passenger_count",
                "pickup_datetime",
            ]
        ]
        .isna()
        .any(axis=1)
        .values
    )
    invalid |= miss

    same = (
        (df["dropoff_longitude"] == df["pickup_longitude"])
        & (df["dropoff_latitude"] == df["pickup_latitude"])
    ).values
    invalid |= same

    zeros = (
        (df["dropoff_longitude"] == 0)
        | (df["pickup_longitude"] == 0)
        | (df["dropoff_latitude"] == 0)
        | (df["pickup_latitude"] == 0)
    ).values
    invalid |= zeros

    MinMax = (-74.5, -72.8, 40.5, 41.8)
    oob = (
        (df["pickup_longitude"] < MinMax[0])
        | (df["pickup_longitude"] > MinMax[1])
        | (df["dropoff_longitude"] < MinMax[0])
        | (df["dropoff_longitude"] > MinMax[1])
        | (df["pickup_latitude"] < MinMax[2])
        | (df["pickup_latitude"] > MinMax[3])
        | (df["dropoff_latitude"] < MinMax[2])
        | (df["dropoff_latitude"] > MinMax[3])
    ).values
    invalid |= oob

    pc_bad = ((df["passenger_count"] <= 0) | (df["passenger_count"] > 6)).values
    invalid |= pc_bad

    num_cols = [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]
    for c in num_cols:
        med = (
            float(np.nanmedian(df.loc[~invalid, c].astype("float64").values))
            if (~invalid).any()
            else float(np.nanmedian(df[c].astype("float64").values))
        )
        df.loc[invalid, c] = med

    return df, invalid




## === cell 4
print("trainKaggle clean (before split)")
trainKaggle = clean(trainKaggle)

train_df, test_df = train_test_split(
    trainKaggle, test_size=0.50, random_state=1, shuffle=True
)
train_df, validation_df = train_test_split(
    train_df, test_size=0.10, random_state=1, shuffle=True
)

print("testKaggle Size %d" % len(testKaggle))
print("train_df Size %d" % len(train_df))
print("validation_df Size %d" % len(validation_df))
print("test_df Size %d" % len(test_df))



## === cell 5
train_df.describe()



## === cell 6
validation_df.describe()



## === cell 7
test_df.describe()



## === cell 8
print("train_df add_time_features")
train_df = add_time_features(train_df)
print("validation_df add_time_features")
validation_df = add_time_features(validation_df)
print("test_df add_time_features")
test_df = add_time_features(test_df)

print("testKaggle clean_test_preserve_rows + add_time_features")
testKaggle_cleaned_for_feats, test_invalid_mask = clean_test_preserve_rows(testKaggle)
testKaggle_cleaned_for_feats = add_time_features(testKaggle_cleaned_for_feats)



## === cell 9
print("train_df add_coordinate_features")
train_df = add_coordinate_features(train_df)
print("validation_df add_coordinate_features")
validation_df = add_coordinate_features(validation_df)
print("test_df add_coordinate_features")
test_df = add_coordinate_features(test_df)

print("testKaggle add_coordinate_features")
testKaggle_cleaned_for_feats = add_coordinate_features(testKaggle_cleaned_for_feats)



## === cell 10
print("train_df add_distances_features")
train_df = add_distances_features(train_df)
print("validation_df add_distances_features")
validation_df = add_distances_features(validation_df)
print("test_df add_distances_features")
test_df = add_distances_features(test_df)
print("testKaggle add_distances_features")
testKaggle_cleaned_for_feats = add_distances_features(testKaggle_cleaned_for_feats)

print("Done with Adding features")



## === cell 11
train_df.describe()



## === cell 12
validation_df.describe()



## === cell 13
dropped_columns = ["passenger_count", "pickup_datetime"]  # keep as original

train_df = train_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)
validation_df = validation_df.drop(dropped_columns, axis=1)

testKaggle_clean = testKaggle_cleaned_for_feats.drop(dropped_columns + ["key"], axis=1)

print("Done with dropped_columns")



## === cell 14
train_labels = train_df["fare_amount"].values
validation_labels = validation_df["fare_amount"].values
test_labels = test_df["fare_amount"].values

for _df_name, _df in [
    ("train_df", train_df),
    ("validation_df", validation_df),
    ("test_df", test_df),
]:
    if "key" in _df.columns:
        _df.drop(["key"], axis=1, inplace=True)

train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)
test_df = test_df.drop(["fare_amount"], axis=1)

print("Done with Labels")



## === cell 15
train_df.shape



## === cell 16
test_df.shape



## === cell 17
validation_df.shape



## === cell 18
train_df = train_df.apply(pd.to_numeric, errors="coerce")
validation_df = validation_df.apply(pd.to_numeric, errors="coerce")
test_df = test_df.apply(pd.to_numeric, errors="coerce")
testKaggle_clean = testKaggle_clean.apply(pd.to_numeric, errors="coerce")

feature_cols = list(train_df.columns)
validation_df = validation_df[feature_cols]
test_df = test_df[feature_cols]
testKaggle_clean = testKaggle_clean[feature_cols]

train_keep = np.isfinite(train_df.values).all(axis=1)
val_keep = np.isfinite(validation_df.values).all(axis=1)
test_keep = np.isfinite(test_df.values).all(axis=1)

train_df = train_df.loc[train_keep].reset_index(drop=True)
train_labels = train_labels[train_keep]

validation_df = validation_df.loc[val_keep].reset_index(drop=True)
validation_labels = validation_labels[val_keep]

test_df = test_df.loc[test_keep].reset_index(drop=True)
test_labels = test_labels[test_keep]

test_vals = testKaggle_clean.values
finite_mask = np.isfinite(test_vals)
if not finite_mask.all():
    col_meds = np.nanmedian(np.where(finite_mask, test_vals, np.nan), axis=0)
    bad_rows, bad_cols = np.where(~finite_mask)
    test_vals[bad_rows, bad_cols] = col_meds[bad_cols]
    testKaggle_clean.iloc[:, :] = test_vals

print("Final feature columns:", feature_cols)



## === cell 19
scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_df)
testKaggle_scaled = scaler.transform(testKaggle_clean)



## === cell 20
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers

tf.random.set_seed(1)


def rmse(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true), axis=-1))




## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 21
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

model.compile(loss="mean_squared_error", optimizer=adam, metrics=["mae", rmse])

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



## === cell 22
try:
    from IPython.display import SVG
    from tensorflow.keras.utils import model_to_dot

    SVG(model_to_dot(model).create(prog="dot", format="svg"))
except Exception as e:
    print(f"Model visualization skipped: {type(e).__name__}: {e}")



## === cell 23
plot_loss_accuracy_rmse(history)



## === cell 24
prediction = model.predict(test_scaled, batch_size=128, verbose=1)
predictionKaggle = model.predict(testKaggle_scaled, batch_size=128, verbose=1)



## === cell 25
y_true = test_labels[:1000].astype(np.float32).reshape(-1)
y_pred = prediction[:1000].astype(np.float32).reshape(-1)
mse_check = float(np.mean((y_pred - y_true) ** 2))
print("Sanity MSE (first 1000):", mse_check)



## === cell 26
predictionKaggle = np.asarray(predictionKaggle, dtype=np.float64).reshape(-1)
bad_pred = ~np.isfinite(predictionKaggle)
if bad_pred.any():
    fallback_value = float(np.median(train_labels)) if len(train_labels) else 11.35
    predictionKaggle[bad_pred] = fallback_value
    print("Replaced non-finite predictions:", int(bad_pred.sum()))

predictionKaggle = np.maximum(predictionKaggle, 0.0)

fallback_value = float(np.median(train_labels)) if len(train_labels) else 11.35
if "test_invalid_mask" in globals() and test_invalid_mask is not None:
    test_invalid_mask = np.asarray(test_invalid_mask, dtype=bool).reshape(-1)
    if test_invalid_mask.shape[0] == predictionKaggle.shape[0]:
        predictionKaggle[test_invalid_mask] = fallback_value
        print("Applied fallback to invalid test rows:", int(np.sum(test_invalid_mask)))
    else:
        print(
            "Warning: invalid mask length mismatch; skipping fallback.",
            test_invalid_mask.shape[0],
            predictionKaggle.shape[0],
        )

predictionKaggle = predictionKaggle.reshape(-1, 1)



## === cell 27
output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)



## === cell 28
i = min(1000, len(prediction) - 1)
print("Example pred:", float(prediction[i]))
print("Example true:", float(test_labels[i]))
print(
    "Submission preview:",
    pd.read_csv(
        SUBMISSION_NAME
        if SUBMISSION_NAME.lower().endswith(".csv")
        else f"{SUBMISSION_NAME}.csv"
    ).head(),
)
