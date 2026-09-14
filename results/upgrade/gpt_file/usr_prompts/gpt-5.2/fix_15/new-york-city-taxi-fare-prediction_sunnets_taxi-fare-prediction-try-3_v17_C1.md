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

4.11503

# 6. Current score

9.86296

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 15.21933) has done: 'I fix the runtime issues caused by Keras 3 API changes (optimizer naming, removed vis_utils) and by attempting to read a remote NYC water mask (Kaggle has no internet for that). I keep the model and feature engineering intact, but make the “water mask” filtering a safe no-op so cleaning still runs without breaking. I also correct the datetime parsing format so it works with the dataset’s timestamps, and ensure the script always writes a valid `submissiontry_water.csv` with `key,fare_amount`. These changes are primarily correctness/stability, but they should also improve RMSE substantially versus the current broken/incorrect pipeline.'
- What this solution (achieved 505.19832) has done: 'I fix the runtime crash caused by Keras 3 backend API changes by rewriting the custom `rmse` metric to use `tf_keras.backend` (which provides `sqrt/mean/square`) while keeping the same metric semantics. I also prevent the unrelated protobuf `MessageFactory.GetPrototype` import crash by switching imports from `keras` to `tf_keras` consistently, which is compatible with this environment. Additionally, I fix a data-loading logic bug where the training CSV was read without the `fare_amount` column (due to `usecols`), which severely harms training and score; this is a minimal but necessary correction. Finally, I ensure the script always trains successfully and writes a valid `submissiontry_water.csv` with the required `key,fare_amount` columns.'
- What this solution (achieved 32.61016) has done: 'I fix the immediate runtime crash (`MessageFactory.GetPrototype`) by preventing the incompatible protobuf/keras import path from being used, while keeping your model/training/feature logic unchanged. I also correct a major data leakage/logic issue: you were training on a random half of the training data and predicting on the other half (`test_df`) instead of predicting on the real Kaggle test set, which can yield wildly mis-scaled/invalid submissions and very poor RMSE. Finally, I ensure the submission uses predictions made on `testKaggle` (the actual competition test rows) and that `clean()` is only applied to training/validation (not to test, to preserve row alignment with `key`). These are minimal changes intended to move RMSE down substantially toward your target.'
- What this solution (achieved 232.37017) has done: 'I fix the remaining `MessageFactory.GetPrototype` crash by forcing TensorFlow’s bundled protobuf implementation (pure-Python) to be used before importing `tf_keras`, which avoids the incompatible C++ protobuf path in this Kaggle image. I also fix a key logic bug that severely hurts RMSE: the training CSV is currently loaded without the `key` column, but later we need to keep `key` in test and align submission rows; we include `key` for train reads (score-neutral) and additionally drop `key` before scaling (correctness). Finally, to move RMSE down toward your target (current score is far worse than target), I fix the train/test feature mismatch caused by `clean()` not being applied to train/val consistently with feature engineering and by datatype coercions (ensuring the same feature columns and numeric dtypes are fed to the scaler/model), without changing the model architecture or training loop.'
- What this solution (achieved 778.88821) has done: 'I fix the immediate runtime crash (`MessageFactory` has no `GetPrototype`) by preventing the incompatible `protobuf` C++ implementation from being imported and by importing TensorFlow before `tf_keras` so Keras uses TF’s bundled protobuf safely. I keep your model, features, training loop, and loss unchanged, only making minimal environment/import changes needed to run end-to-end. I also add a small safety check to ensure the submission predictions are a 1D float array and that the output CSV is written with the required `key,fare_amount` columns. These changes are score-neutral logically but unblock training/inference and prevent malformed submissions that can explode RMSE.'
- What this solution (achieved 587.04088) has done: 'We fix the immediate protobuf crash (`MessageFactory` missing `GetPrototype`) by forcing the pure-Python protobuf implementation and disabling the C++ fast-path before importing TensorFlow/tf_keras, which is the root cause of the runtime error in this Kaggle image. We also make the Kaggle input paths resolve correctly by defaulting to `/kaggle/input/...` (your current `../input/...` often doesn’t exist), ensuring the script can actually load the data. Finally, we remove the inappropriate `accuracy` metric (meaningless for regression and can destabilize training/compile in some setups) while keeping the same model, loss, features, and training loop; this is score-positive but still within “core logic preserved”. The rest of the pipeline is kept intact and always write a valid `submissiontry_water.csv` with `key,fare_amount`.'
- What this solution (achieved 614.76597) has done: 'I fix the import-time protobuf crash that prevents the notebook from running by ensuring TensorFlow uses the pure-Python protobuf implementation and by removing the `tf_keras` import path that is triggering the `MessageFactory.GetPrototype` failure in this environment. I keep the exact same model architecture, feature engineering, training loop, and loss/metrics semantics, but switch the Keras API usage to `tensorflow.keras` (still TensorFlow-backed) to avoid the incompatible protobuf/keras combination. I also add a small safety fallback so the script still produces a valid `submissiontry_water.csv` even if the primary Keras import path fails. These changes are execution-unblocking and should move RMSE dramatically down from the current broken submission toward the target.'
- What this solution (achieved 49.35732) has done: 'I fix the protobuf/Keras import crash that happens before any training by setting the protobuf environment variables early and importing `tf_keras` (not `tensorflow.keras`) consistently, avoiding the incompatible protobuf path that triggers `MessageFactory.GetPrototype`. I keep your model, features, training loop, loss, and metric semantics unchanged, but ensure TensorFlow is imported in a safe order and that the custom RMSE metric uses the same backend reliably. I also make the Kaggle file path resolution correct for this dataset layout (`/kaggle/input/new-york-city-taxi-fare-prediction/...`), so the code actually reads the right files. These fixes are execution-unblocking and should reduce the RMSE drastically from the current broken submission toward your target by ensuring the model trains on real labels and predicts on the actual test set.'
- What this solution (achieved 298.63003) has done: 'We need to fix the import-time protobuf crash (`MessageFactory` missing `GetPrototype`) so the notebook can actually run and train; the minimal reliable fix in this Kaggle image is to avoid `tf_keras`/Keras-3’s protobuf path and use TensorFlow’s bundled `tensorflow.keras` consistently. I keep the same model architecture, loss, metric semantics, feature engineering, and training loop, only swapping the Keras import surface and making the custom `rmse` use `tf.keras.backend`. I also keep your existing path resolution and submission writer, ensuring `submissiontry_water.csv` is always produced with `key,fare_amount`. These changes should unblock training and also move RMSE down substantially from the current broken run toward the target.'
- What this solution (achieved 3880.52747) has done: 'I fix the import-time crash (`AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`) by forcing TensorFlow/Keras to use the pure-Python protobuf implementation *before* importing TensorFlow, and by avoiding the standalone `keras`/`tf_keras` stacks entirely. This is a minimal, execution-unblocking change that keeps your model, features, training loop, and loss/metric semantics the same. I also remove the unnecessary `import tensorflow.keras.losses` cell that can re-trigger the protobuf path and crash after training. These changes are correctness/stability focused and should move RMSE drastically down from the current broken/invalid pipeline toward your target by ensuring the model actually trains and produces a valid submission CSV.'
- What this solution (achieved 6.1927) has done: 'We need to fix the import-time protobuf crash that prevents TensorFlow/Keras from loading; the cleanest minimal change in this Kaggle image is to force the Python protobuf implementation early and import `tf_keras` (TF-bundled Keras) instead of `tensorflow.keras`. Then we keep your model, features, training loop, and loss the same, only updating the Keras import surface and the `rmse` metric backend to match. Finally, we add a small safety clamp on predictions (non-negative fares) and ensure the submission is always written with the required `key,fare_amount` columns and correct row alignment.'
- What this solution (achieved 1366.61107) has done: 'I fix the immediate runtime crash (`AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`) by avoiding the standalone `tf_keras` import path that is triggering the protobuf incompatibility in this Kaggle image, and instead using TensorFlow’s bundled `tensorflow.keras` consistently (same model/loss/training loop). I keep your feature engineering and training semantics intact, only adjusting imports and the custom `rmse` to use `tf.keras.backend` so compile/fit works reliably. To move RMSE toward the target without changing core modeling, I also make the train/test preprocessing consistent by applying the same non-label-dependent cleaning filters to the test set (without dropping rows), preventing distribution/feature mismatch that can inflate error. The pipeline still write a valid `submissiontry_water.csv` with `key,fare_amount` and correct row alignment.'
- What this solution (achieved 9.85831) has done: 'I fix the import-time crash (`MessageFactory.GetPrototype`) by avoiding TensorFlow/Keras imports in this environment and replacing them with a scikit-learn regressor, while keeping your existing data loading, cleaning, and feature engineering pipeline intact. This change is necessary because the current script cannot run end-to-end due to the protobuf/TensorFlow incompatibility, and the resulting submission is effectively broken (hence the extremely large RMSE). I also keep the scaler and prediction clamping logic, and ensure the submission file is written as `submissiontry_water.csv` with exactly `key,fare_amount`. This should move RMSE dramatically down toward your target without introducing sampling, early stopping, or other approximations.'
- What this solution (achieved 9.86296) has done: 'Your RMSE (9.858) is far worse than the target (4.115), so we should cautiously improve predictive accuracy without changing the overall pipeline. The biggest low-risk gain here is to keep your exact feature engineering/cleaning semantics but stop zero-filling invalid coordinates/datetime-derived fields (which creates unrealistic “0” trips) and instead impute missing numeric features using the training medians (consistent, non-leaky, and typically much better for tree/boosting models). I also clip predictions to the training target range (0–50) since you already trained only on that range, which reduces extreme outliers that heavily hurt RMSE. Everything else (data size, model class, iterations, learning rate, and features) remains unchanged, and it still writes `submissiontry_water.csv` with `key,fare_amount`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ.setdefault("PROTOCOL_BUFFERS_DISABLE_UPB", "1")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = os.environ.get("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn.ensemble import HistGradientBoostingRegressor

TRAIN_PATH = "/kaggle/input/train.csv"
TEST_PATH = "/kaggle/input/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 0.001
DATASET_SIZE = 80000


def resolve_path(p):
    """Resolve Kaggle dataset file locations without changing intended I/O behavior."""
    if os.path.exists(p):
        return p
    base = os.path.basename(p)
    candidates = [
        f"/kaggle/input/{base}",
        f"/kaggle/input/new-york-city-taxi-fare-prediction/{base}",
        f"/kaggle/data/{base}",
        f"/kaggle/input/new-york-city-taxi-fare-prediction/{base}.zip",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return p


TRAIN_PATH = resolve_path(TRAIN_PATH)
TEST_PATH = resolve_path(TEST_PATH)

np.random.seed(1)

print("Resolved TRAIN_PATH:", TRAIN_PATH)
print("Resolved TEST_PATH:", TEST_PATH)



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
    TEST_PATH, dtype={k: v for k, v in datatypes.items() if k != "fare_amount"}
)



## === cell 2
train_df, validation_df = train_test_split(trainKaggle, test_size=0.10, random_state=1)



## === cell 3
testKaggle.head()



## === cell 4
validation_df.head()



## === cell 5
print("testKaggle Size %d" % len(testKaggle))
print("train_df Size %d" % len(train_df))
print("validation_df Size %d" % len(validation_df))




## === cell 6
def remove_datapoints_from_water(df):
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


def clean_test_inplace(df):
    df = df.copy()

    for c in [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    same_mask = (df["dropoff_longitude"] == df["pickup_longitude"]) & (
        df["dropoff_latitude"] == df["pickup_latitude"]
    )
    df.loc[
        same_mask,
        [
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
        ],
    ] = np.nan

    zero_mask = (
        (df["dropoff_longitude"] == 0)
        | (df["pickup_longitude"] == 0)
        | (df["dropoff_latitude"] == 0)
        | (df["pickup_latitude"] == 0)
    )
    df.loc[
        zero_mask,
        [
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
        ],
    ] = np.nan

    MinMax = (-74.5, -72.8, 40.5, 41.8)
    out_nyc = (
        (df["pickup_longitude"] < MinMax[0])
        | (df["pickup_longitude"] > MinMax[1])
        | (df["dropoff_longitude"] < MinMax[0])
        | (df["dropoff_longitude"] > MinMax[1])
        | (df["pickup_latitude"] < MinMax[2])
        | (df["pickup_latitude"] > MinMax[3])
        | (df["dropoff_latitude"] < MinMax[2])
        | (df["dropoff_latitude"] > MinMax[3])
    )
    df.loc[
        out_nyc,
        [
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
        ],
    ] = np.nan

    bad_pax = (df["passenger_count"] <= 0) | (df["passenger_count"] > 6)
    df.loc[bad_pax, "passenger_count"] = np.nan

    return df


def late_night(row):
    if (row["hour"] <= 3) or (row["hour"] >= 22):
        return 1
    else:
        return 0


def night(row):
    if (row["hour"] > 20) and (row["weekday"] < 5):
        return 1
    else:
        return 0


def rush_hour(row):
    if ((row["hour"] <= 20) and (row["hour"] >= 16)) and (row["weekday"] < 5):
        return 1
    else:
        return 0


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=False)
    df["year"] = dt.dt.year.astype("int16")
    df["month"] = dt.dt.month.astype("int8")
    df["day"] = dt.dt.day.astype("int8")
    df["hour"] = dt.dt.hour.astype("int8")
    df["weekday"] = dt.dt.weekday.astype("int8")

    df["pickup_datetime"] = dt.astype(str)

    df["night"] = df.apply(lambda x: night(x), axis=1).astype("int8")
    df["late_night"] = df.apply(lambda x: late_night(x), axis=1).astype("int8")
    df["rush_hour"] = df.apply(lambda x: rush_hour(x), axis=1).astype("int8")
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
    df["distance"] = np.sqrt(np.abs(lon1 - lon2) ** 2 + np.abs(lat1 - lat2) ** 2)
    return df


def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    pred = np.asarray(prediction).reshape(-1).astype(np.float32)
    pred = np.maximum(pred, 0.0)
    pred = np.minimum(pred, 50.0)
    df = pd.DataFrame({id_column: raw_test[id_column].values, prediction_column: pred})
    df.to_csv(file_name, index=False)
    print("Output complete:", file_name, "rows=", len(df))


def plot_loss_accuracy_rmse(history):
    if history is None:
        print("No Keras history available (sklearn training used).")
        return

    plt.figure(figsize=(20, 10))
    plt.plot(history.history.get("loss", []))
    plt.plot(history.history.get("val_loss", []))
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper right")
    plt.show()

    plt.figure(figsize=(20, 10))
    if "rmse" in history.history:
        plt.plot(history.history["rmse"])
    if "val_rmse" in history.history:
        plt.plot(history.history["val_rmse"])
    plt.title("Model rmse")
    plt.ylabel("rmse")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper right")
    plt.show()




## === cell 7
print("train_df clean")
train_df = clean(train_df)
print("validation_df clean")
validation_df = clean(validation_df)

print("testKaggle clean (non-dropping)")
testKaggle = clean_test_inplace(testKaggle)

print("train_df add_time_features")
train_df = add_time_features(train_df)
print("validation_df add_time_features")
validation_df = add_time_features(validation_df)

print("testKaggle add_time_features")
testKaggle = add_time_features(testKaggle)



## === cell 8
print("train_df add_coordinate_features")
train_df = add_coordinate_features(train_df)
print("validation_df add_coordinate_features")
validation_df = add_coordinate_features(validation_df)
print("testKaggle add_coordinate_features")
testKaggle = add_coordinate_features(testKaggle)



## === cell 9
print("train_df add_distances_features")
train_df = add_distances_features(train_df)
print("validation_df add_distances_features")
validation_df = add_distances_features(validation_df)
print("testKaggle add_distances_features")
testKaggle = add_distances_features(testKaggle)

print("Done with Adding features")



## === cell 10
dropped_columns = ["passenger_count", "pickup_datetime", "key"]

train_df = train_df.drop(dropped_columns, axis=1)
validation_df = validation_df.drop(dropped_columns, axis=1)
testKaggle_clean = testKaggle.drop(
    ["passenger_count", "pickup_datetime", "key"], axis=1
)

print("Done with dropped_columns")



## === cell 11
train_labels = train_df["fare_amount"].values
validation_labels = validation_df["fare_amount"].values

train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)

print("Done with Labels")



## === cell 12
train_df = train_df.apply(pd.to_numeric, errors="coerce")
validation_df = validation_df.apply(pd.to_numeric, errors="coerce")
testKaggle_clean = testKaggle_clean.apply(pd.to_numeric, errors="coerce")

medians = train_df.median(numeric_only=True)
train_df = train_df.fillna(medians)
validation_df = validation_df.fillna(medians)
testKaggle_clean = testKaggle_clean.fillna(medians)

train_df = train_df.fillna(0.0)
validation_df = validation_df.fillna(0.0)
testKaggle_clean = testKaggle_clean.fillna(0.0)

scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
testKaggle_scaled = scaler.transform(testKaggle_clean)


def rmse_np(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))




## === cell 13
model = HistGradientBoostingRegressor(
    loss="squared_error",
    learning_rate=LEARNING_RATE,
    max_depth=None,
    max_iter=EPOCHS,  # map "epochs" to boosting iterations (keeps intent of multiple training steps)
    random_state=1,
)

print("Dataset size: %s" % DATASET_SIZE)
print("Boosting iterations (EPOCHS): %s" % EPOCHS)
print("Learning rate: %s" % LEARNING_RATE)
print("Input dimension: %s" % train_df_scaled.shape[1])
print("Features used: %s" % list(train_df.columns))

model.fit(train_df_scaled, train_labels)

val_pred = model.predict(validation_df_scaled)
print("Validation RMSE:", rmse_np(validation_labels, val_pred))
history = None



## === cell 14
print("Model visualization skipped (not applicable for sklearn).")



## === cell 15
plot_loss_accuracy_rmse(history)



## === cell 16
predictionKaggle = model.predict(testKaggle_scaled)



## === cell 17
pass



## === cell 18
output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)



## === cell 19
print(np.asarray(predictionKaggle[:10]).reshape(-1))
