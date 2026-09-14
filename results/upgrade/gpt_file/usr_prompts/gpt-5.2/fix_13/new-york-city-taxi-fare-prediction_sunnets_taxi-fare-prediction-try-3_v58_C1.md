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

4.74408

# 6. Current score

6.07356

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 25.30994) has done: 'I fix the run blockers caused by (1) fetching the NYC water mask from a URL (no internet in Kaggle) by making that step optional and safely skipped when the mask can’t be loaded, and (2) Keras 3 API changes (optimizer constructor and missing `vis_utils`) so the model compiles/trains again. I also correct a couple of time-feature logic bugs (`late_night`/`night`) and the datetime parsing to be robust to the dataset format; these are minimal semantic fixes that should improve RMSE toward your target without changing the core model/training approach. Finally, I ensure paths match the provided environment and that a valid `submissiontry_water.csv` with columns `key,fare_amount` is always written.'
- What this solution (achieved 25.56911) has done: 'I fix the runtime crash happening at import time (`AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`) by pinning protobuf to the pure‑python implementation via an environment variable set before any TensorFlow/Keras-related imports. This is a minimal, score-neutral stability fix that unblocks the notebook from running end-to-end in the Kaggle environment. I also correct the train/test CSV paths to match your provided `/kaggle/input/new-york-city-taxi-fare-prediction/` location to avoid “file not found” in Kaggle’s standard layout. Finally, I keep the model/training logic unchanged and ensure the submission CSV is always written with the required `key,fare_amount` columns.'
- What this solution (achieved 17.9455) has done: 'I fix the two run blockers that prevent any submission from being generated: the protobuf/TensorFlow import crash and the wrong input CSV paths. I make the environment-variable workaround execute before any TensorFlow/Keras-related import, and I switch `TRAIN_PATH`/`TEST_PATH` to the actual `/kaggle/input/new-york-city-taxi-fare-prediction/` files shown in your environment listing. Then I keep the core model and feature logic unchanged, but ensure the pipeline always reaches the final `to_csv` and writes a valid `submissiontry_water.csv` with columns `key,fare_amount`. These changes are primarily correctness/stability; any score change should be neutral-to-positive versus the current “no submission” state.'
- What this solution (achieved 6.14996) has done: 'I fix the import-time crash (`MessageFactory.GetPrototype`) by forcing protobuf to use the pure‑python implementation *and* disabling the C++ implementation before any TensorFlow/Keras import, plus clearing any preloaded protobuf modules to ensure the env-var takes effect. I also switch the train path to the already-provided `labels.csv` (same schema as train but without the extra unused rows file) to keep behavior identical while improving stability and reducing I/O overhead. To improve RMSE toward your target with minimal semantic change, I make the synthetic holdout split representative by not truncating `test_df` to 10k (this only affects the printed local RMSE, not the Kaggle submission), and I ensure predictions are clipped to a realistic fare range (0–50) consistent with your training outlier filter. Finally, I keep the model architecture/training loop unchanged and ensure the submission CSV is always written with `key,fare_amount`.'
- What this solution (achieved 6.21688) has done: 'I fix the import-time protobuf crash by moving the environment-variable workaround to the very top and also forcing the pure‑python protobuf implementation via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` before any TF/Keras-related imports. This is a run-blocker fix and is score-neutral (it just ensures the notebook runs consistently in the Kaggle environment). Then, to improve RMSE toward your target with minimal semantic change, I make the train/validation split representative by shuffling after cleaning/feature creation (still using the same `train_test_split` approach and random seed), so scaling and training aren’t biased by the initial row order. Finally, I keep the model architecture, optimizer, loss, and training loop the same and ensure the submission CSV is always written with correct columns and `.csv` suffix.'
- What this solution (achieved 7.87044) has done: 'I fix the run-blocking protobuf import crash by ensuring the pure‑python protobuf implementation is forced *before* any TensorFlow/Keras-related import and by avoiding the unsupported env var that can break in this environment. Then I make the training data read deterministic (shuffle via `skiprows` with a fixed seed) so the model sees a more representative sample from the huge CSV without changing your model/training loop, which should improve RMSE toward your target. Finally, I keep your architecture/loss/training procedure intact and only add a safe fallback to read from `train.csv` if `labels.csv` isn’t present, ensuring the notebook always runs end-to-end and writes a valid `.csv` submission with `key,fare_amount`.'
- What this solution (achieved 6.18923) has done: 'I fix the run-blocking protobuf/TensorFlow import crash by forcing protobuf’s pure‑python implementation *before* any TF/Keras import, and by preventing the already-loaded C++ protobuf modules from being reused. Then I make the training sample reader avoid scanning the entire 55M-row CSV just to count lines (which can time out) by switching to a deterministic random chunk sampling method that still preserves the same “random sample from the big file” intent. Finally, to nudge RMSE down toward your target without changing the model/training loop, I ensure the training sample includes a stable distribution (no header accidentally skipped; consistent dtypes), and keep the submission writing identical but guaranteed valid.'
- What this solution (achieved 7.82029) has done: 'I fix the run-blocking protobuf/TensorFlow import crash by forcing the pure‑python protobuf backend before any TF/Keras import and by avoiding the incompatible `*_VERSION` env var that triggers the `GetPrototype` failure in this Kaggle stack. I also make the protobuf-unload step more robust (only if protobuf was partially imported) so execution is deterministic. These changes are stability/compatibility fixes and should be score-neutral while ensuring the pipeline runs end-to-end and writes a valid `submissiontry_water.csv`. I not change your model architecture, training loop, feature engineering, or prediction post-processing so behavior remains the same aside from negligible runtime/environment differences.'
- What this solution (achieved 6.34823) has done: 'I fix the immediate run blocker (`MessageFactory.GetPrototype`) by forcing protobuf’s pure‑python backend *and* disabling the C++ implementation before any TensorFlow/Keras-related imports, and by setting the implementation version to `2` (the most compatible in Kaggle’s TF stacks). This is a stability fix that should not change model logic, but it allow the notebook to run end-to-end reliably and write a valid submission CSV. I also add a safe fallback to import `tf_keras` only after those environment variables are set, without altering the model architecture, features, training loop, or prediction post-processing. No score-targeted modeling changes are introduced here; the focus is to unblock execution and ensure a valid `submissiontry_water.csv` is produced.'
- What this solution (achieved 6.38011) has done: 'I fix the remaining protobuf/TensorFlow import crash by avoiding module-deletion after protobuf is already partially loaded and by adding the missing env var to disable the C++ protobuf implementation before any TF/Keras imports. This is a stability-only change intended to be score-neutral but to guarantee the notebook runs end-to-end and writes the submission. I also make the train/test split truly random (shuffle) at the first `train_test_split` to avoid order bias from chunk-sampling, which is a minimal semantic change that should improve RMSE toward your target without changing the model, features, or training loop. Finally, I keep the output format identical and ensure the submission CSV is always written with `key,fare_amount`.'
- What this solution (achieved 6.07356) has done: 'I fix the current run-blocking protobuf import crash by setting the protobuf environment variables before any TensorFlow/Keras-related imports and ensuring no protobuf modules are preloaded in this kernel session. Then I make one minimal, score-improving correction to align training and Kaggle inference: apply the same deterministic column order to all scaled matrices (train/val/test/Kaggle) to prevent accidental feature misalignment if pandas column order differs after cleaning/feature creation. Finally, I keep the model, training loop, features, and post-processing the same and ensure the submission is always written as `submissiontry_water.csv` with the required `key,fare_amount` columns.'

# 9. Code solution

## === cell 0
import os
import sys

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_CPP"] = "1"

for m in list(sys.modules.keys()):
    if m.startswith("google.protobuf"):
        del sys.modules[m]

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import tf_keras as keras
from tf_keras.models import Sequential
from tf_keras.layers import Dense, Dropout, BatchNormalization, LSTM
from tf_keras import optimizers
from tf_keras import regularizers
from tf_keras import backend

TRAIN_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/labels.csv"
TEST_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 0.001
DATASET_SIZE = 80000

np.random.seed(1)
print("Using TRAIN_PATH:", TRAIN_PATH, "exists:", os.path.exists(TRAIN_PATH))
print("Using TEST_PATH :", TEST_PATH, "exists:", os.path.exists(TEST_PATH))
print("Will write      :", SUBMISSION_NAME)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
import urllib.request
from PIL import Image


def remove_datapoints_from_water(df):
    """
    If a local NYC mask image exists in the working directory, filter rides whose pickup/dropoff are on land.
    If not available (typical Kaggle no-internet setting), return df unchanged.
    """

    def lonlat_to_xy(longitude, latitude, dx, dy, BB):
        x = (dx * (longitude - BB[0]) / (BB[1] - BB[0])).astype("int")
        y = (dy - dy * (latitude - BB[2]) / (BB[3] - BB[2])).astype("int")
        return x, y

    BB = (-74.5, -72.8, 40.5, 41.8)
    local_mask_candidates = [
        "nyc_mask-74.5_-72.8_40.5_41.8.png",
        "/kaggle/working/nyc_mask-74.5_-72.8_40.5_41.8.png",
        "/kaggle/input/nyc_mask-74.5_-72.8_40.5_41.8.png",
    ]
    mask_path = next((p for p in local_mask_candidates if os.path.exists(p)), None)

    if mask_path is None:
        return df

    nyc_mask = plt.imread(mask_path)
    if nyc_mask.ndim == 3:
        nyc_mask = nyc_mask[:, :, 0]
    nyc_mask = nyc_mask > 0.9

    pickup_x, pickup_y = lonlat_to_xy(
        df.pickup_longitude.values,
        df.pickup_latitude.values,
        nyc_mask.shape[1],
        nyc_mask.shape[0],
        BB,
    )
    dropoff_x, dropoff_y = lonlat_to_xy(
        df.dropoff_longitude.values,
        df.dropoff_latitude.values,
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

    df = df[((df["pickup_latitude"] - df["dropoff_latitude"]).abs() > 0.001)]
    df = df[((df["pickup_longitude"] - df["dropoff_longitude"]).abs() > 0.001)]
    print(" New size after lang - lot > 0.001: %d" % len(df))

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
    df2 = remove_datapoints_from_water(df)
    print("New size: %d" % len(df2))

    return df2


def late_night(row):
    return 1 if (row["hour"] <= 3 or row["hour"] >= 22) else 0


def night(row):
    return 1 if (row["hour"] >= 20 or row["hour"] <= 5) else 0


def rush_hour(row):
    return (
        1
        if ((row["hour"] <= 20) and (row["hour"] >= 16) and (row["weekday"] < 5))
        else 0
    )


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)
    if dt.isna().any():
        dt2 = pd.to_datetime(df.loc[dt.isna(), "pickup_datetime"], errors="coerce")
        dt.loc[dt.isna()] = dt2

    df["year"] = dt.dt.year.astype("int16")
    df["month"] = dt.dt.month.astype("int8")
    df["day"] = dt.dt.day.astype("int8")
    df["hour"] = dt.dt.hour.astype("int8")
    df["weekday"] = dt.dt.weekday.astype("int8")

    df["pickup_datetime"] = dt.dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")

    df["night"] = df.apply(lambda x: night(x), axis=1).astype("int8")
    df["late_night"] = df.apply(lambda x: late_night(x), axis=1).astype("int8")
    df["rush_hour"] = df.apply(lambda x: rush_hour(x), axis=1).astype("int8")
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
        (df["pickup_longitude"] - df["dropoff_longitude"]).abs() ** 2
        + (df["pickup_latitude"] - df["dropoff_latitude"]).abs() ** 2
    )
    return df


def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    df = pd.DataFrame(
        {
            id_column: raw_test[id_column].values,
            prediction_column: prediction.reshape(-1),
        }
    )
    df.to_csv(file_name, index=False)
    print("Output complete:", file_name, "rows:", len(df))


def plot_loss_accuracy_rmse(history):
    plt.figure(figsize=(20, 6))
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper right")
    plt.show()

    if "rmse_metric" in history.history:
        plt.figure(figsize=(20, 6))
        plt.plot(history.history["rmse_metric"])
        plt.plot(history.history["val_rmse_metric"])
        plt.title("Model rmse")
        plt.ylabel("rmse")
        plt.xlabel("epoch")
        plt.legend(["train", "val"], loc="upper right")
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


def _read_train_sample(path, nrows, dtype, usecols, seed=1, chunksize=200_000):
    if not os.path.exists(path):
        return None

    rng = np.random.RandomState(seed)
    collected = []
    got = 0

    reader = pd.read_csv(
        path,
        dtype=dtype,
        usecols=usecols,
        chunksize=chunksize,
    )

    for chunk in reader:
        if got >= nrows:
            break
        need = nrows - got
        take = min(need, len(chunk))
        sampled = chunk.sample(
            n=take, replace=False, random_state=int(rng.randint(0, 2**31 - 1))
        )
        collected.append(sampled)
        got += take

    if not collected:
        return pd.read_csv(path, nrows=nrows, dtype=dtype, usecols=usecols)

    out = pd.concat(collected, axis=0, ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


train_usecols = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

trainKaggle = _read_train_sample(
    TRAIN_PATH, DATASET_SIZE, datatypes, train_usecols, seed=1
)
if trainKaggle is None:
    fallback_train = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
    print("TRAIN_PATH missing; falling back to:", fallback_train)
    trainKaggle = _read_train_sample(
        fallback_train, DATASET_SIZE, datatypes, train_usecols, seed=1
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

print("Loaded trainKaggle:", trainKaggle.shape, "testKaggle:", testKaggle.shape)



## === cell 3
train_df, test_df = train_test_split(
    trainKaggle, test_size=0.50, random_state=1, shuffle=True
)



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
print("test_df clean")
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
plt.show()



## === cell 16
dropped_columns = ["passenger_count", "pickup_datetime"]

train_df = train_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)
testKaggle_clean = testKaggle.drop(dropped_columns + ["key"], axis=1)

print("Done with dropped_columns")
print("train_df columns:", train_df.columns.tolist())
print("testKaggle_clean columns:", testKaggle_clean.columns.tolist())



## === cell 17
train_df.shape



## === cell 18
train_df.describe()



## === cell 19
test_df.describe()



## === cell 20
train_df = train_df.sample(frac=1.0, random_state=1).reset_index(drop=True)
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
print(
    "Train X:", train_df.shape, "Val X:", validation_df.shape, "Test X:", test_df.shape
)



## === cell 24
feature_cols = list(train_df.columns)
validation_df = validation_df[feature_cols]
test_df = test_df[feature_cols]
testKaggle_clean = testKaggle_clean[feature_cols]

scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_df)
testKaggle_scaled = scaler.transform(testKaggle_clean)




## === cell 25
def rmse_metric(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))




## === cell 26
model = Sequential()
model.add(
    Dense(
        256,
        activation="linear",
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
model.compile(
    loss="mean_squared_error", optimizer=adam, metrics=["mae", rmse_metric, "mse"]
)

print("Dataset size: %s" % DATASET_SIZE)
print("Epochs: %s" % EPOCHS)
print("Learning rate: %s" % LEARNING_RATE)
print("Batch size: %s" % BATCH_SIZE)
print("Input dimension: %s" % train_df_scaled.shape[1])
print("Features used: %s" % train_df.columns.tolist())
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



## === cell 27
try:
    from IPython.display import SVG
    from tf_keras.utils import model_to_dot

    SVG(model_to_dot(model).create(prog="dot", format="svg"))
except Exception as e:
    print("Model visualization skipped:", repr(e))



## === cell 28
plot_loss_accuracy_rmse(history)



## === cell 29
score = model.evaluate(test_scaled, test_labels, verbose=1)
print(score)



## === cell 30
prediction = model.predict(test_scaled, batch_size=128, verbose=1)
predictionKaggle = model.predict(testKaggle_scaled, batch_size=128, verbose=1)




## === cell 31
def rmse_np(predictions, targets):
    predictions = np.asarray(predictions).reshape(-1)
    targets = np.asarray(targets).reshape(-1)
    return np.sqrt(((predictions - targets) ** 2).mean())




## === cell 32
Validation_prediction = model.predict(validation_df_scaled, batch_size=128, verbose=1)
rmse_val = rmse_np(Validation_prediction, validation_labels)
print("rmse error is: " + str(rmse_val))



## === cell 33
rmse_test = rmse_np(prediction, test_labels)
print("rmse error is: " + str(rmse_test))



## === cell 34
fig, ax = plt.subplots()
ax.scatter(test_labels, prediction, s=5)
ax.plot(
    [test_labels.min(), test_labels.max()],
    [test_labels.min(), test_labels.max()],
    "k--",
    lw=2,
)
ax.set_xlabel("Measured")
ax.set_ylabel("Predicted")
plt.show()



## === cell 35
predictionKaggle = np.nan_to_num(
    predictionKaggle.reshape(-1), nan=11.35, posinf=50.0, neginf=0.0
)
predictionKaggle = np.clip(predictionKaggle, 0.0, 50.0).astype(np.float32)

output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)
print("Submission head:")
print(pd.read_csv(SUBMISSION_NAME).head())
print("Submission shape:", pd.read_csv(SUBMISSION_NAME).shape)
