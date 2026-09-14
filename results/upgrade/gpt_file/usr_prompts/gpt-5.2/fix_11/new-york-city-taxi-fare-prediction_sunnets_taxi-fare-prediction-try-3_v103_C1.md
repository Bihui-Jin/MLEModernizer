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

4.35957

# 6. Current score

50.18379

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 15.2404) has done: 'I fix the runtime blockers while keeping your modeling approach (same Dense architecture, same loss, same training loop) intact. The biggest score issue (84 RMSE vs target 4.36) is caused by broken/incorrect preprocessing: the water-mask step tries to download an image (not allowed), datetime parsing uses an incompatible format, scaling accidentally fits a new scaler per column and even scales `manhattan` using `passenger_count`, and the Keras 3 optimizer API call is wrong. I make the water-mask step use a local mask file if present, otherwise skip it (so the pipeline runs), correct the time parsing to robust `pd.to_datetime`, and replace the per-column scaler misuse with a single MinMaxScaler fit once on all features (same semantics, but correct). Finally, I update optimizer instantiation for Keras 3 and remove the unavailable model visualization import so training completes and a valid `.csv` submission is written.'
- What this solution (achieved 29.3693) has done: 'I fix the runtime blockers preventing training/evaluation by replacing the deprecated `keras.backend` math ops (missing `sqrt` in Keras 3) with TensorFlow ops, keeping the same RMSE definition and training semantics. I also address the early import crash (`MessageFactory GetPrototype`) by switching the Keras import source to `tf_keras`, which is installed and compatible in this environment, without changing your model architecture or training loop. Finally, I make checkpoint saving compatible with Keras 3+ by using a `.keras` filename, and keep the submission-writing logic intact so a valid `.csv` is always produced.'
- What this solution (achieved 121.05726) has done: 'I fix the immediate runtime crash in the first cell (`MessageFactory GetPrototype`) by forcing TensorFlow to use the pure-Python protobuf implementation before importing TensorFlow/keras, which avoids that known incompatibility in Kaggle-style environments. I also make the data paths robust to both `/kaggle/input/...` and your current `../input/...` layout without changing any downstream logic. Finally, to move RMSE down toward the target with minimal semantic change, I keep the same Dense model/training loop but ensure we actually reload the best checkpoint (instead of using the last epoch weights) before predicting/evaluating and creating the submission—this typically improves generalization without altering the approach.'
- What this solution (achieved 20.07181) has done: 'We fix the immediate runtime crash (`MessageFactory has no attribute GetPrototype`) by pinning protobuf to the pure-Python implementation *and* disabling the C++ implementation before TensorFlow/tf_keras import, which is the common Kaggle workaround. Next, we correct a major score-killer: the model is trained **without** the `key` column, but `test.csv` is currently being read with `key` included and then scaled, causing a feature mismatch and badly calibrated predictions (or silent misalignment). Finally, we make the test reading consistent with training by loading only the same feature columns from `test.csv` (keeping `key` separately for submission), so the pipeline runs end-to-end and the RMSE should move substantially toward the target.'
- What this solution (achieved 195.339) has done: 'I fix the import-time protobuf crash by forcing the pure-Python protobuf implementation *and* disabling the C++ implementation before TensorFlow/tf_keras is imported, which directly addresses the `MessageFactory GetPrototype` error in cell 0. I keep your model, features, scaling, and training loop unchanged, only adding a safe fallback to import `tf_keras` without importing standalone `tensorflow` first (which is where the crash happens most often). I also make the data-path resolver prefer the actual `/kaggle/data/...` and `/kaggle/input/...` file locations you listed, without changing filenames or outputs. The rest of the pipeline remains identical and still write `submissiontry_water.csv` with `key,fare_amount`.'
- What this solution (achieved 25.16637) has done: 'I fix the import-time protobuf crash that prevents the notebook from running by forcing the pure-Python protobuf implementation *before* any TensorFlow/tf_keras import, and by avoiding importing standalone `tensorflow` at module import time (it triggers the crash in this environment). Then I keep your exact model/training/scaling/feature logic intact, only moving TensorFlow usage inside the `rmse()` function so the rest of the code can import safely. Finally, I ensure the pipeline always reaches the submission-writing cell and produces `submissiontry_water.csv` with the required `key,fare_amount` columns.'
- What this solution (achieved 187.83053) has done: 'I fix the import-time protobuf crash by avoiding any TensorFlow import at module import time (it’s what triggers the `MessageFactory.GetPrototype` failure in this environment) and by importing TensorFlow lazily only inside the custom `rmse()` metric. Then I make `tf_keras` load stable by not importing TensorFlow just for seeding; instead I keep NumPy seeding (score-neutral) so the notebook runs end-to-end and still trains the same Dense model with the same loop. These are minimal execution-unblocking changes that should also help your score by ensuring the model actually trains and reloads the best checkpoint consistently, rather than crashing before training.'
- What this solution (achieved 306.06663) has done: 'You’re currently crashing before any training due to a known protobuf/TensorFlow incompatibility; the environment variables alone aren’t enough if TensorFlow is imported indirectly by `tf_keras` before the workaround takes effect. I make the TensorFlow/Keras import path deterministic by importing `tensorflow` only after setting the protobuf env vars, then using `tf.keras` (same Sequential/Dense architecture, same optimizer/loss/loop) to avoid the `MessageFactory.GetPrototype` failure. I also add a small safety cast/fill for time features so `MinMaxScaler` never receives NaNs from datetime parsing (score-improving stability, not a modeling change). Finally, I keep the submission writing identical and ensure the output is a valid `.csv` with `key,fare_amount`.'
- What this solution (achieved 342.2104) has done: 'I fix the import-time crash (`MessageFactory ... GetPrototype`) by ensuring TensorFlow uses the pure-Python protobuf runtime *and* that TensorFlow/keras are imported in a safer order, with a fallback to `tf_keras` if needed. This change is execution-unblocking and score-neutral, but it allow the model to actually train and generate a valid submission. I also add a small safety step to enforce numeric dtypes and fill any remaining NaNs/Infs right before scaling so the pipeline can’t silently produce extreme predictions or fail inside `MinMaxScaler`. The model architecture, loss, training loop, and feature engineering remain unchanged, and the script still write `submissiontry_water.csv` with `key,fare_amount`.'
- What this solution (achieved 50.18379) has done: 'I fix the import-time crash (`MessageFactory has no attribute GetPrototype`) by moving the protobuf environment-variable workaround to the very top and enforcing it via `os.environ[...]` before any TensorFlow/Keras-related imports, then importing `tf_keras` directly (which is installed and avoids the problematic standalone TF import path here). This is an execution-unblocking change and should also substantially improve score because your current run is effectively failing before any meaningful training/prediction can occur. I keep your model architecture, features, scaling, and training loop unchanged, and only make the minimal import/seed adjustments needed for stability. The script still write a valid `submissiontry_water.csv` with `key,fare_amount`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_CPP"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import tf_keras as keras  # type: ignore
from tf_keras.models import Sequential  # type: ignore
from tf_keras.layers import Dense  # type: ignore
from tf_keras.callbacks import ModelCheckpoint  # type: ignore
from tf_keras import regularizers  # type: ignore
from tf_keras.optimizers import Adam  # type: ignore
import tensorflow as tf  # safe now that env vars are set


def _resolve_path(p):
    candidates = [
        p,
        p.replace("../input/", "/kaggle/input/new-york-city-taxi-fare-prediction/"),
        p.replace("../input/", "/kaggle/input/"),
        p.replace("../input/", "/kaggle/data/new-york-city-taxi-fare-prediction/"),
        p.replace("../input/", "/kaggle/data/"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return p


TRAIN_PATH = _resolve_path("../input/train.csv")
TEST_PATH = _resolve_path("../input/test.csv")
SUBMISSION_NAME = "submissiontry_water.csv"  # must end with .csv for Kaggle

BATCH_SIZE = 1000
EPOCHS = 100
LEARNING_RATE = 0.001
DATASET_SIZE = 80000

np.random.seed(1)
try:
    tf.random.set_seed(1)
except Exception:
    pass

print("Using TRAIN_PATH:", TRAIN_PATH)
print("Using TEST_PATH :", TEST_PATH)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def remove_datapoints_from_water(df):
    """
    Original code tried to read a mask image from a URL via plt.imread, which fails
    (and internet may be blocked). We try to load a local mask file if present.
    If not available, we skip this filter to keep the pipeline runnable.
    """

    def lonlat_to_xy(longitude, latitude, dx, dy, BB):
        return (dx * (longitude - BB[0]) / (BB[1] - BB[0])).astype("int"), (
            dy - dy * (latitude - BB[2]) / (BB[3] - BB[2])
        ).astype("int")

    BB = (-74.5, -72.8, 40.5, 41.8)

    local_candidates = [
        "./nyc_mask-74.5_-72.8_40.5_41.8.png",
        "../input/nyc_mask-74.5_-72.8_40.5_41.8.png",
        "/kaggle/input/nyc_mask-74.5_-72.8_40.5_41.8.png",
        "/kaggle/working/nyc_mask-74.5_-72.8_40.5_41.8.png",
    ]
    mask_path = next((p for p in local_candidates if os.path.exists(p)), None)

    if mask_path is None:
        return df

    nyc_mask = plt.imread(mask_path)
    if nyc_mask.ndim == 3:
        nyc_mask = nyc_mask[:, :, 0]
    nyc_mask = nyc_mask > 0.9

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
    df = df[(df["pickup_longitude"] != 0)]
    df = df[(df["dropoff_longitude"] != 0)]
    df = df[(df["dropoff_latitude"] != 0)]
    print(" New size after lang lot > 0: %d" % len(df))

    df = df[((df["pickup_latitude"] - df["dropoff_latitude"]).abs() > 0.001)]
    df = df[((df["pickup_longitude"] - df["dropoff_longitude"]).abs() > 0.001)]
    print(" New size after lang - lot > 0.001: %d" % len(df))

    print(" New size after only NYC: %d" % len(df))

    if "fare_amount" in df.columns:
        df = df[(0 < df["fare_amount"]) & (df["fare_amount"] <= 50)]
        print(" New size after removing outliers: %d" % len(df))

    df = df[(df["passenger_count"] > 0)]
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


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)
    df["year"] = dt.dt.year.astype("float32")
    df["month"] = dt.dt.month.astype("float32")
    df["day"] = dt.dt.day.astype("float32")
    df["hour"] = dt.dt.hour.astype("float32")
    df["minute"] = dt.dt.minute.astype("float32")
    df["second"] = dt.dt.second.astype("float32")

    for c in ["year", "month", "day", "hour", "minute", "second"]:
        df[c] = df[c].fillna(0.0).astype("float32")
    return df


def add_coordinate_features(df):
    _ = df["pickup_latitude"]
    _ = df["dropoff_latitude"]
    _ = df["pickup_longitude"]
    _ = df["dropoff_longitude"]
    return df


def add_distances_features(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]
    df["manhattan"] = manhattan(lat1, lon1, lat2, lon2).astype("float32")
    return df


def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    df_out = pd.DataFrame(
        {
            id_column: raw_test[id_column].values,
            prediction_column: np.asarray(prediction).reshape(-1),
        }
    )
    df_out.to_csv(file_name, index=False)
    print("Output complete:", file_name, "rows:", len(df_out))


def rmse(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true), axis=-1))


def plot_loss_accuracy_rmse(history):
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

train_usecols = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
test_usecols = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

trainKaggle = pd.read_csv(
    TRAIN_PATH,
    nrows=DATASET_SIZE,
    dtype=datatypes,
    usecols=train_usecols,
)

testKaggle = pd.read_csv(
    TEST_PATH,
    dtype={k: v for k, v in datatypes.items() if k != "fare_amount"},
    usecols=test_usecols,
)

print("Loaded trainKaggle:", trainKaggle.shape, "testKaggle:", testKaggle.shape)



## === cell 3
train_df, test_df = train_test_split(trainKaggle, test_size=0.50, random_state=1)
test_df = test_df[:10000]

print("testKaggle Size %d" % len(testKaggle))
print("train_df Size %d" % len(train_df))
print("test_df Size %d" % len(test_df))



## === cell 4
print("train_df clean")
train_df = clean(train_df)
test_df = clean(test_df)



## === cell 5
print("train_df add_time_features")
train_df = add_time_features(train_df)
print("test_df add_time_features")
test_df = add_time_features(test_df)
print("testKaggle add_time_features")
testKaggle = add_time_features(testKaggle)



## === cell 6
print("train_df add_coordinate_features")
train_df = add_coordinate_features(train_df)
print("test_df add_coordinate_features")
test_df = add_coordinate_features(test_df)
print("testKaggle add_coordinate_features")
testKaggle = add_coordinate_features(testKaggle)



## === cell 7
print("train_df add_distances_features")
train_df = add_distances_features(train_df)
print("test_df add_distances_features")
test_df = add_distances_features(test_df)
print("testKaggle add_distances_features")
testKaggle = add_distances_features(testKaggle)

print("Done with Adding features")



## === cell 8
dropped_columns = ["pickup_datetime"]

train_df = train_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)

testKaggle_clean = testKaggle.drop(dropped_columns + ["key"], axis=1)

print("Done with dropped_columns")
print("train_df columns:", train_df.columns.tolist())
print("testKaggle_clean columns:", testKaggle_clean.columns.tolist())



## === cell 9
train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=1)

train_labels = train_df["fare_amount"].values.astype("float32")
validation_labels = validation_df["fare_amount"].values.astype("float32")
test_labels = test_df["fare_amount"].values.astype("float32")

train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)
test_df = test_df.drop(["fare_amount"], axis=1)

print("Done with Labels")
print(
    "Shapes:",
    train_df.shape,
    validation_df.shape,
    test_df.shape,
    testKaggle_clean.shape,
)




## === cell 10
def _ensure_numeric_no_nan(df):
    df = df.copy()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0.0)
    return df.astype("float32")


train_df = _ensure_numeric_no_nan(train_df)
validation_df = _ensure_numeric_no_nan(validation_df)
test_df = _ensure_numeric_no_nan(test_df)
testKaggle_clean = _ensure_numeric_no_nan(testKaggle_clean)

scaler = preprocessing.MinMaxScaler()

train_df_scaled = pd.DataFrame(
    scaler.fit_transform(train_df.values),
    columns=train_df.columns,
    index=train_df.index,
)

validation_df_scaled = pd.DataFrame(
    scaler.transform(validation_df.values),
    columns=validation_df.columns,
    index=validation_df.index,
)

test_df_scaled = pd.DataFrame(
    scaler.transform(test_df.values),
    columns=test_df.columns,
    index=test_df.index,
)

testKaggle_scaled = pd.DataFrame(
    scaler.transform(testKaggle_clean.values),
    columns=testKaggle_clean.columns,
    index=testKaggle_clean.index,
)

print(
    "Scaled shapes:",
    train_df_scaled.shape,
    validation_df_scaled.shape,
    test_df_scaled.shape,
    testKaggle_scaled.shape,
)



## === cell 11
checkpoint_path = "my_model.keras"
checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_best_only=True,
    monitor="val_loss",
    mode="min",
)

model = Sequential()
model.add(
    Dense(
        256,
        activation="linear",
        input_dim=train_df_scaled.shape[1],
        activity_regularizer=regularizers.l1(0.01),
    )
)
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="linear"))

adam = Adam(learning_rate=LEARNING_RATE)

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
    callbacks=[checkpoint],
    validation_data=(validation_df_scaled, validation_labels),
    shuffle=True,
)

if os.path.exists(checkpoint_path):
    model = keras.models.load_model(checkpoint_path, custom_objects={"rmse": rmse})
    print("Loaded best checkpoint:", checkpoint_path)



## === cell 12
plot_loss_accuracy_rmse(history)



## === cell 13
score = model.evaluate(train_df_scaled, train_labels, verbose=1)
print("Train metrics:", dict(zip(model.metrics_names, score)))

score = model.evaluate(validation_df_scaled, validation_labels, verbose=1)
print("Validation metrics:", dict(zip(model.metrics_names, score)))

score = model.evaluate(test_df_scaled, test_labels, verbose=1)
print("Test metrics:", dict(zip(model.metrics_names, score)))



## === cell 14
validation_predictions = model.predict(
    validation_df_scaled, batch_size=1024, verbose=0
).flatten()
test_predictions = model.predict(test_df_scaled, batch_size=1024, verbose=0).flatten()

plt.figure(figsize=(6, 6))
plt.scatter(validation_labels, validation_predictions, s=2)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.axis("equal")
lims = [
    min(validation_labels.min(), validation_predictions.min()),
    max(validation_labels.max(), validation_predictions.max()),
]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims, "k--", lw=2)
plt.show()



## === cell 15
predictionKaggle = model.predict(testKaggle_scaled, batch_size=1024, verbose=1)
predictionKaggle = np.clip(predictionKaggle, 0, None)

output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)
print("Submission head:")
print(pd.read_csv(SUBMISSION_NAME).head())
print("Wrote submission file:", os.path.abspath(SUBMISSION_NAME))
