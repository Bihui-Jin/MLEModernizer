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

4.69443

# 6. Current score

507.40076

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.15656) has done: 'I fix the environment/runtime crash coming from importing `tf_keras` by switching to `tensorflow.keras` (available in Kaggle) while keeping the exact same Sequential Dense/BatchNorm architecture and training loop. Then I fix the scaler failure by ensuring no string columns (specifically `key`) leak into the model matrices: we drop `key` from train/valid/test featureframes and keep it only for the final submission. Finally, I make sure the pipeline always produces a valid `submissiontry_water.csv` with columns `key,fare_amount`, and I add a small safety clip on predictions (non-negative) to avoid invalid fares without changing the modeling approach.'
- What this solution (achieved 116.34729) has done: 'I fix the current hard crash that happens before any training by forcing TensorFlow to use the pure‑Python protobuf implementation (this avoids the common `MessageFactory.GetPrototype` incompatibility) and by ensuring we import TF/Keras in a way that works with the installed Kaggle packages. Then I correct two time-feature logic bugs (`late_night` and `night`) whose current conditions are always true/false, which harms model quality; this keeps the same feature set and model/training loop but makes those features meaningful, which should move RMSE down toward the target. Finally, I keep the existing submission-writing logic but add a small guard to always output a valid CSV with the required columns.'
- What this solution (achieved 67.83809) has done: 'I fix the TensorFlow import crash caused by an incompatible protobuf runtime by switching the model code to use the already-installed `tf_keras` package (same Keras API, same layers/compile/fit semantics) instead of `tensorflow.keras`. I also correct the dataset paths to the actual provided files under `/kaggle/data/...` so the script can read data reliably in your environment. Finally, I keep your exact feature engineering and model architecture/training loop, but I make the submission writer more robust (ensuring alignment and correct columns) so a valid `.csv` is always produced end-to-end.'
- What this solution (achieved 75.19069) has done: 'I fix the immediate runtime crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation *before* any TensorFlow/tf_keras import, and by clearing any already-imported `google.protobuf` modules so the environment variable actually takes effect. I also make the data paths robust to your environment (preferring `/kaggle/data/...` but falling back to `/kaggle/input/...`) without changing any downstream logic. To move RMSE down toward the target, I fix a core modeling mistake that is currently crushing performance: the final layer uses `relu`, which forbids negative outputs and severely harms regression; switching the last activation to `linear` preserves the same architecture/training loop and matches the MSE objective. Finally, I keep the submission writer but ensure it always writes the required `key,fare_amount` CSV.'
- What this solution (achieved 23.90767) has done: 'We fix the runtime crash happening before training by ensuring TensorFlow’s protobuf is set up safely and by switching imports from `tf_keras` to the already-available `tensorflow.keras` stack, which avoids the `MessageFactory.GetPrototype` error in this environment. This is a runtime-only change: the model architecture, loss, optimizer, and training loop remain the same. We also add a small defensive fallback so the script still runs even if the protobuf env var workaround is not sufficient (by cleanly re-importing after setting env), and we keep the existing submission-writing logic to always produce `submissiontry_water.csv` with the required `key,fare_amount` columns. These changes should restore end-to-end execution and, because the model actually train properly again, move RMSE substantially down from the broken ~75 range toward your target band.'
- What this solution (achieved 6.1951) has done: 'The crash happens before training due to an incompatible TensorFlow/protobuf combination (`MessageFactory.GetPrototype`). The minimal fix is to avoid importing TensorFlow entirely and run the exact same Keras Sequential Dense/BatchNorm model using the already-installed `tf_keras` package, which matches the intended API and preserves the same training loop and loss. This change is directly runtime/stability related but should also improve RMSE substantially because the model now actually train instead of failing. I also keep your existing feature engineering and submission-writing logic unchanged, ensuring a valid `submissiontry_water.csv` with `key,fare_amount` is always produced.'
- What this solution (achieved 35.33624) has done: 'We fix the immediate crash happening in the first cell (`MessageFactory.GetPrototype`) by avoiding TensorFlow/tf_keras entirely and switching the same Sequential Dense/BatchNorm regression model to scikit-learn’s `MLPRegressor` (same MSE objective and training loop concept, just a backend that works reliably here). This is a runtime/stability fix that also tends to reduce RMSE versus a broken/non-running deep-learning stack, moving your score down toward the 4.69 target. We keep your existing cleaning and feature engineering intact, keep the same train/validation split semantics, and preserve the submission writing logic/format. Finally, we add a tiny numeric safety step (finite casting + non-negative clip already present) to ensure predictions are valid and the submission CSV is always produced.'
- What this solution (achieved 40.56388) has done: 'Your current RMSE is far worse than the target, so the smallest safe move is to fix two issues that strongly hurt generalization without changing your overall approach: (1) make time-based feature creation vectorized (same features/semantics, but avoids row-wise `apply` overhead and potential dtype quirks), and (2) align train/test distributions by fitting the model on the full cleaned training sample (train+holdout) after you’ve verified RMSE on the holdout, then predict Kaggle test. This keeps the same feature set, scaling, model family (MLPRegressor), and loss/optimizer semantics, but typically moves RMSE substantially down from the current 35 range. I also keep the submission writer unchanged and ensure the `key` alignment stays correct.'
- What this solution (achieved 507.40076) has done: 'Your current RMSE (40.56; lower is better) is far from the target (4.69), so we need a small but meaningful fix that improves generalization without changing the model family or training loop. The biggest issue is that `pickup_datetime` is converted to string inside `add_time_features`, which can invalidate the datetime parsing on later calls (you call it multiple times on different splits) and can silently produce NaT-based/garbled time features; we stop overwriting that column while keeping the exact same derived features. Second, the features include `year/day/month` which can be poorly conditioned under MinMax scaling with limited sample; switching to `StandardScaler` typically improves MLPRegressor regression stability while keeping the same model/optimizer/loss semantics. Finally, we keep the same cleaning/feature set and submission format, and only add a tiny column-alignment assert before scaling to prevent train/test feature order mismatches.'

# 9. Code solution

## === cell 0
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor


def _pick_existing_path(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return paths[0]


TRAIN_PATH = _pick_existing_path(
    [
        "/kaggle/data/train.csv",
        "/kaggle/data/new-york-city-taxi-fare-prediction/train.csv",
        "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
        "/kaggle/input/train.csv",
    ]
)
TEST_PATH = _pick_existing_path(
    [
        "/kaggle/data/test.csv",
        "/kaggle/data/new-york-city-taxi-fare-prediction/test.csv",
        "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
        "/kaggle/input/test.csv",
    ]
)

SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 0.001
DATASET_SIZE = 80000

np.random.seed(1)

print("Train path:", TRAIN_PATH, "exists:", os.path.exists(TRAIN_PATH))
print("Test path:", TEST_PATH, "exists:", os.path.exists(TEST_PATH))




## === cell 1
def remove_datapoints_from_water(df):
    """
    Bug fix: original code tries to read a remote URL via plt.imread, which fails in Kaggle.
    Minimal change: if a local mask file is present, use it; otherwise skip this filter safely.
    """

    def lonlat_to_xy(longitude, latitude, dx, dy, BB):
        return (dx * (longitude - BB[0]) / (BB[1] - BB[0])).astype("int"), (
            dy - dy * (latitude - BB[2]) / (BB[3] - BB[2])
        ).astype("int")

    BB = (-74.5, -72.8, 40.5, 41.8)

    possible_paths = [
        "/kaggle/input/nyc_mask-74.5_-72.8_40.5_41.8.png",
        "/kaggle/input/new-york-city-taxi-fare-prediction/nyc_mask-74.5_-72.8_40.5_41.8.png",
        "/kaggle/working/nyc_mask-74.5_-72.8_40.5_41.8.png",
        "/kaggle/data/nyc_mask-74.5_-72.8_40.5_41.8.png",
        "/kaggle/data/new-york-city-taxi-fare-prediction/nyc_mask-74.5_-72.8_40.5_41.8.png",
    ]
    mask_path = next((p for p in possible_paths if os.path.exists(p)), None)

    if mask_path is None:
        return df

    nyc_mask = plt.imread(mask_path)[:, :, 0] > 0.9

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

    pickup_x = np.clip(pickup_x, 0, nyc_mask.shape[1] - 1)
    dropoff_x = np.clip(dropoff_x, 0, nyc_mask.shape[1] - 1)
    pickup_y = np.clip(pickup_y, 0, nyc_mask.shape[0] - 1)
    dropoff_y = np.clip(dropoff_y, 0, nyc_mask.shape[0] - 1)

    idx = nyc_mask[pickup_y, pickup_x] & nyc_mask[dropoff_y, dropoff_x]
    return df[idx]


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
    print(" New size after NYC lang lot: %d" % len(df))

    df = df[(df["pickup_latitude"] != 0)]
    df = df[(df["pickup_latitude"] != 0)]
    df = df[(df["dropoff_longitude"] != 0)]
    df = df[(df["dropoff_latitude"] != 0)]
    print(" New size after lang lot > 0: %d" % len(df))

    df = df[((df["pickup_latitude"] - df["dropoff_latitude"]).abs() > 0.001)]
    df = df[((df["pickup_longitude"] - df["dropoff_longitude"]).abs() > 0.001)]
    print(" New size after lang - lot > 0.001: %d" % len(df))

    print(" New size after only NYC: %d" % len(df))
    df = df[(0 < df["fare_amount"]) & (df["fare_amount"] <= 50)]
    print(" New size after removing outliers: %d" % len(df))

    df = df[(df["passenger_count"] > 0) & (df["passenger_count"] <= 6)]
    print(" New size after removing 6=>passenger_count > 0 : %d" % len(df))

    nyc_coord = (40.7141667, -74.0063889)
    fk_coord = (40.639722, -73.778889)
    ewr_coord = (40.6925, -74.168611)
    lga_coord = (40.77725, -73.872611)
    sol_coord = (40.6892, -74.0445)  # Statue of Liberty

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
    h = row["hour"]
    return 1 if (h <= 3 or h >= 22) else 0


def night(row):
    h = row["hour"]
    wd = row["weekday"]
    return 1 if ((h >= 20 or h <= 6) and wd < 5) else 0


def rush_hour(row):
    if ((row["hour"] <= 20) and (row["hour"] >= 16)) and (row["weekday"] < 5):
        return 1
    else:
        return 0


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["hour"] = dt.dt.hour
    df["weekday"] = dt.dt.weekday

    h = df["hour"]
    wd = df["weekday"]

    df["night"] = (((h >= 20) | (h <= 6)) & (wd < 5)).astype(np.uint8)
    df["late_night"] = (((h <= 3) | (h >= 22))).astype(np.uint8)
    df["rush_hour"] = (((h >= 16) & (h <= 20)) & (wd < 5)).astype(np.uint8)
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
    df["distance"] = np.sqrt(
        np.abs(df["pickup_longitude"] - df["dropoff_longitude"]) ** 2
        + np.abs(df["pickup_latitude"] - df["dropoff_latitude"]) ** 2
    )
    return df


def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))


def distanceP(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))


def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    pred = np.asarray(prediction).reshape(-1)
    if len(pred) != len(raw_test):
        raise ValueError(
            f"Prediction length {len(pred)} != test length {len(raw_test)}"
        )
    df = pd.DataFrame(
        {
            id_column: raw_test[id_column].astype(str).values,
            prediction_column: pred.astype(np.float32),
        }
    )
    df.to_csv(file_name, index=False)
    print("Output complete:", file_name, "rows:", len(df))


def rmse_np(predictions, targets):
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    return np.sqrt(((predictions - targets) ** 2).mean())




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
test_df = test_df[:10000]



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
_ = train_df.iloc[:2000].plot.scatter("latdiff", "londiff")
_ = train_df.iloc[:2000].plot.scatter("fare_amount", "passenger_count")
_ = train_df.iloc[:2000].plot.scatter("fare_amount", "year")
_ = train_df.iloc[:2000].plot.scatter("fare_amount", "month")
_ = train_df.iloc[:2000].plot.scatter("fare_amount", "day")
_ = train_df.iloc[:2000].plot.scatter("fare_amount", "hour")
_ = train_df.iloc[:2000].plot.scatter("fare_amount", "weekday")
_ = train_df.iloc[:2000].plot.scatter("fare_amount", "night")
_ = train_df.iloc[:2000].plot.scatter("fare_amount", "late_night")
_ = train_df.iloc[:2000].plot.scatter("fare_amount", "rush_hour")



## === cell 16
dropped_columns = ["pickup_datetime", "key"]  # keep passenger_count
train_df = train_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)
testKaggle_clean = testKaggle.drop(["pickup_datetime", "key"], axis=1)
print("Done with dropped_columns")



## === cell 17
train_df.shape



## === cell 18
train_df.describe()



## === cell 19
test_df.describe()



## === cell 20
train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=1)



## === cell 21
train_df.describe()



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
test_labels[:10]



## === cell 25
train_df.describe()



## === cell 26
validation_df.describe()



## === cell 27
test_df.describe()



## === cell 28
scaler = preprocessing.StandardScaler()
train_df_scaled = scaler.fit_transform(train_df.astype(np.float32))
validation_df_scaled = scaler.transform(validation_df.astype(np.float32))
test_scaled = scaler.transform(test_df.astype(np.float32))

if list(testKaggle_clean.columns) != list(train_df.columns):
    testKaggle_clean = testKaggle_clean.reindex(columns=train_df.columns)
testKaggle_scaled = scaler.transform(testKaggle_clean.astype(np.float32))



## === cell 29
test_scaled[:2]



## === cell 30
mlp = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64, 32, 8),
    activation="relu",
    solver="adam",
    alpha=0.0,  # keep core logic close to original (no L2); original used activity_regularizer l1
    batch_size=BATCH_SIZE,
    learning_rate_init=LEARNING_RATE,
    max_iter=EPOCHS,
    shuffle=True,
    random_state=1,
    early_stopping=False,
    verbose=True,
)

print("Dataset size (initial fit): %s" % DATASET_SIZE)
print("Epochs(max_iter): %s" % EPOCHS)
print("Learning rate: %s" % LEARNING_RATE)
print("Batch size: %s" % BATCH_SIZE)
print("Input dimension: %s" % train_df_scaled.shape[1])
print("Features used: %s" % list(train_df.columns))

mlp.fit(train_df_scaled, train_labels)



## === cell 31
print("Training done; skipping model graph rendering (not applicable for sklearn).")



## === cell 32
if hasattr(mlp, "loss_curve_"):
    plt.figure(figsize=(20, 6))
    plt.plot(mlp.loss_curve_)
    plt.title("MLPRegressor loss curve")
    plt.ylabel("loss")
    plt.xlabel("iteration")
    plt.show()



## === cell 33
train_pred = mlp.predict(train_df_scaled)
print("Train RMSE:", rmse_np(train_pred, train_labels))



## === cell 34
val_pred = mlp.predict(validation_df_scaled)
print("Validation RMSE:", rmse_np(val_pred, validation_labels))



## === cell 35
test_pred = mlp.predict(test_scaled)
print("Holdout test RMSE:", rmse_np(test_pred, test_labels))



## === cell 36
validation_predictions = val_pred.flatten()
plt.scatter(validation_labels, validation_predictions, s=5)
plt.xlabel("True Values [10$]")
plt.ylabel("Predictions [10$]")
plt.axis("equal")
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot([-100, 100], [-100, 100])



## === cell 37
test_predictions = test_pred.flatten()
plt.scatter(test_labels, test_predictions, s=5)
plt.xlabel("True Values [10$]")
plt.ylabel("Predictions [10$]")
plt.axis("equal")
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot([-100, 100], [-100, 100])



## === cell 38
X_full = np.vstack([train_df_scaled, validation_df_scaled])
y_full = np.concatenate([train_labels, validation_labels])

mlp_full = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64, 32, 8),
    activation="relu",
    solver="adam",
    alpha=0.0,
    batch_size=BATCH_SIZE,
    learning_rate_init=LEARNING_RATE,
    max_iter=EPOCHS,
    shuffle=True,
    random_state=1,
    early_stopping=False,
    verbose=True,
)
mlp_full.fit(X_full, y_full)

predictionKaggle = mlp_full.predict(testKaggle_scaled)




## === cell 39
def rmse_np(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())




## === cell 40
print(np.argmax(test_predictions))
print(test_predictions[np.argmax(test_predictions)])
print(test_labels[np.argmax(test_predictions)])
test_df.iloc[np.argmax(test_predictions)]



## === cell 41
print(np.argmin(test_predictions))
print(test_predictions[np.argmin(test_predictions)])
print(test_labels[np.argmin(test_predictions)])
test_df.iloc[np.argmin(test_predictions)]



## === cell 42
fig, ax = plt.subplots()
ax.scatter(test_labels, test_predictions, s=5)
ax.plot(
    [test_labels.min(), test_labels.max()],
    [test_labels.min(), test_labels.max()],
    "k--",
    lw=2,
)
ax.set_xlabel("Measured")
ax.set_ylabel("Predicted")
plt.show()



## === cell 43
error = test_predictions - test_labels
plt.hist(error, bins=50)
plt.xlabel("Prediction Error [10$]")
_ = plt.ylabel("Count")



## === cell 44
predictionKaggle = np.asarray(predictionKaggle, dtype=np.float64).reshape(-1)
predictionKaggle = np.where(np.isfinite(predictionKaggle), predictionKaggle, 0.0)
predictionKaggle = np.clip(predictionKaggle, 0, None)

output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)
print(pd.read_csv(SUBMISSION_NAME).head())
print("Saved submission to:", os.path.abspath(SUBMISSION_NAME))
