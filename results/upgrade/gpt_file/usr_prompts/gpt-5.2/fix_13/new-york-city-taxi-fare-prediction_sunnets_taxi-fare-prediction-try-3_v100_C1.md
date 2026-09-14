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

4.2871

# 6. Current score

51.7758

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 147.37347) has done: 'I fix the runtime blockers first: the NYC water-mask cleaning currently tries to read an image from a URL (no internet) and crashes, so I make that step safely skip when the mask can’t be loaded. Next, I correct the scaler bug where the `manhattan` feature in `test_df_scaled` is mistakenly transformed using `passenger_count`, and I make scaling consistent by using a single `MinMaxScaler` fit on the training set for all features (this is score-improving and still preserves the same modeling approach). Finally, I update the Keras optimizer call for Keras 3 (`Adam(learning_rate=...)`) and remove the optional model visualization import that doesn’t exist, ensuring the model trains and a valid `submissiontry_water.csv` is produced.'
- What this solution (achieved 15.20475) has done: 'I fix the runtime crash happening at the very first imports by avoiding the problematic protobuf/tensorflow initialization path and switching the Keras import to the installed `keras` (Keras 3) API, which is compatible with the environment. Then I correct the training CSV loading to actually include the `fare_amount` target column (it’s currently excluded due to `usecols`, which causes the model to train on misaligned/invalid labels and leads to the extremely bad RMSE). Finally, I keep the same model and feature logic, but ensure all numeric inputs are float32 and clamp negative predictions to 0 before writing the submission to improve RMSE without changing the core approach.'
- What this solution (achieved 67.3151) has done: 'I fix the Keras 3 backend RMSE metric that currently crashes training by switching it to a TensorFlow-based implementation that works in this environment, without changing the model or loss. I also fix the initial protobuf/TensorFlow initialization crash by explicitly using the installed `tf_keras` stack (compatible with this Kaggle image) while keeping the same Sequential/Dense architecture and training loop. Finally, I make the input file paths robust to this notebook’s filesystem layout (preferring `/kaggle/input/...` when available) so the script runs end-to-end and reliably writes `submissiontry_water.csv` with the required `key,fare_amount` columns.'
- What this solution (achieved 6.04401) has done: 'I fix the runtime crash in the first cell caused by an incompatible `tensorflow` import (protobuf `MessageFactory.GetPrototype` error) by switching fully to the already-installed `tf_keras` stack and not importing standalone `tensorflow`. Then I correct a major data bug that is causing the very poor RMSE: the training CSV load currently omits the `key` column, which later breaks the expected submission ID alignment logic and also prevents consistent end-to-end feature handling; I load `key` for train and keep it through cleaning/splitting while still dropping it from model features. Finally, I keep the same feature engineering and model architecture/training loop, but ensure we never compile with the meaningless `accuracy` metric for a regression model (it can destabilize logs/early stopping behavior) while leaving loss/optimizer unchanged; this is score-improving but does not change the training objective.'
- What this solution (achieved 15.25184) has done: 'I fix the immediate runtime crash caused by importing `tf_keras` (protobuf `MessageFactory.GetPrototype` issue) by switching to the installed `keras` (Keras 3) API, which avoids importing standalone TensorFlow while keeping the same Sequential Dense architecture and training loop. I also make `ModelCheckpoint` compatible with Keras 3 by saving to a `.keras` file instead of legacy `.h5`, preventing checkpoint write errors. To nudge RMSE toward the target with minimal semantic change, I log-transform the target (`fare_amount`) during training and invert-transform predictions for evaluation/submission (a standard regression calibration that preserves the same model/feature pipeline but typically improves RMSE). Finally, I keep the submission format/paths identical and ensure the CSV is written correctly.'
- What this solution (achieved 15.25183) has done: 'I fix the immediate runtime crash by forcing Keras to use the NumPy backend (so it won’t try to initialize TensorFlow/protobuf) while keeping the same model and training loop. Then I fix the `rmse` metric implementation to use `keras.ops` (Keras 3 backend-agnostic ops) because `keras.backend.cast` is not available in this environment, which currently prevents `fit()` and all `evaluate()` calls from running. These changes are score-neutral in intent (they restore training/evaluation), and your existing log1p target transform + inverse at inference is preserved as-is. Finally, I make plotting robust to missing `history` (in case training fails) and ensure the submission CSV is always written with the required `key,fare_amount` columns.'
- What this solution (achieved 19.07166) has done: 'I fix the runtime crash in the first cell by removing the incompatible `tf_keras`/`tensorflow` imports that trigger the protobuf `MessageFactory.GetPrototype` error, and instead use the installed Keras 3 API with the NumPy backend so the notebook can run end-to-end without TensorFlow. To keep your core model/training logic identical, I keep the same Sequential Dense architecture, optimizer settings, loss, log1p target transform, scaling, and prediction inverse-transform; I only adjust the custom `rmse` metric implementation to use backend-agnostic `keras.ops` so `fit()`/`evaluate()` work. I also ensure the submission is always written as a valid `.csv` with columns `key,fare_amount` and clamp predictions to non-negative as you already intended. These changes are primarily to unblock execution and produce a valid submission; they should also improve RMSE versus a broken run by ensuring training actually completes and inference executes correctly.'
- What this solution (achieved 53217809.6929) has done: 'I fix the runtime blocker by switching from the Keras NumPy backend (which does not implement `fit()`/`evaluate()`) to the TensorFlow backend, which is available in the Kaggle image and supports training end-to-end. To avoid the historical protobuf crash, I import `tensorflow` only after setting `TF_CPP_MIN_LOG_LEVEL` and keep the rest of your model/feature logic unchanged. I also make the `rmse` metric robust across backends by using `keras.ops` on tensors and ensure `model.evaluate()` receives NumPy arrays (not Python lists). These changes are necessary for correctness (so training actually happens) and should improve RMSE substantially toward the target because your previous run effectively could not train.'
- What this solution (achieved 597654361.88825) has done: 'The crash comes from importing TensorFlow in this Kaggle image (protobuf incompatibility), so I remove the TensorFlow backend requirement and run Keras on the JAX backend instead, which still supports `fit()/evaluate()` and keeps your model/training loop identical. To preserve evaluation semantics without changing your model, I keep your same features, scaling, log1p target transform, and inverse transform at inference; only the backend and seeding setup are adjusted for stability. I also add a small safety fallback: if JAX can’t be used for any reason, the script still write a valid submission using the sample-submission mean (so you always get a `.csv`). Finally, I keep the submission format exactly `key,fare_amount` and ensure predictions are 1D float and non-negative.'
- What this solution (achieved 2.2295218774944148e+16) has done: 'I fix the training crash (`MessageFactory.GetPrototype`) by avoiding the JAX backend and instead using the NumPy backend for Keras 3, then switching training to `sklearn`’s `MLPRegressor` (same idea: a multi-layer perceptron trained on the same scaled features and log1p target) so the pipeline trains reliably in this environment. This is a minimal semantic change focused on unblocking training and massively reducing the RMSE from the current broken score toward the target band, without changing the feature engineering, scaling, or log1p/inverse prediction logic. I also keep the existing fallback submission behavior, ensure predictions are finite and non-negative, and always write a valid `submissiontry_water.csv` with `key,fare_amount`. All file paths and the submission format remain unchanged.'
- What this solution (achieved 51.7758) has done: 'Your current RMSE is astronomically bad because the model is trained on `log1p(fare)` but you evaluate/score against raw `fare_amount`; the internal validation RMSE looks small but it’s in log-space, while Kaggle scores raw-space. I keep your exact feature pipeline and the same `MLPRegressor` approach, but switch training labels to raw fares (no log-transform) so predictions match the competition metric. To keep changes minimal and stable, I also clamp any negative predictions to 0 (already consistent with your intent) and keep the same submission writing logic and file name. This should move the score dramatically down toward your target RMSE band without changing the core modeling logic.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

os.environ["KERAS_BACKEND"] = "numpy"
os.environ.setdefault("PYTHONHASHSEED", "1")

np.random.seed(1)

import keras
from keras import ops

from sklearn.neural_network import MLPRegressor


def _pick_path(*candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


TRAIN_PATH = _pick_path(
    "/kaggle/input/train.csv",
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
    "../input/train.csv",
    "/kaggle/data/train.csv",
    "/kaggle/data/new-york-city-taxi-fare-prediction/train.csv",
)
TEST_PATH = _pick_path(
    "/kaggle/input/test.csv",
    "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
    "../input/test.csv",
    "/kaggle/data/test.csv",
    "/kaggle/data/new-york-city-taxi-fare-prediction/test.csv",
)
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 1000
EPOCHS = 100
LEARNING_RATE = 0.001
DATASET_SIZE = 80000

print("Keras package:", keras.__name__)
print("Keras backend:", keras.backend.backend())
print("Using TRAIN_PATH:", TRAIN_PATH)
print("Using TEST_PATH :", TEST_PATH)




## === cell 1
def remove_datapoints_from_water(df):
    """
    Original code attempted to fetch a NYC land/water mask from a URL, which fails in Kaggle (no internet).
    Minimal fix: try to load a local mask file if present; otherwise skip this filter safely.
    """

    def lonlat_to_xy(longitude, latitude, dx, dy, BB):
        return (dx * (longitude - BB[0]) / (BB[1] - BB[0])).astype("int"), (
            dy - dy * (latitude - BB[2]) / (BB[3] - BB[2])
        ).astype("int")

    BB = (-74.5, -72.8, 40.5, 41.8)

    local_candidates = [
        "/kaggle/input/nyc-mask/nyc_mask-74.5_-72.8_40.5_41.8.png",
        "/kaggle/input/new-york-city-taxi-fare-prediction/nyc_mask-74.5_-72.8_40.5_41.8.png",
        "/kaggle/working/nyc_mask-74.5_-72.8_40.5_41.8.png",
        "/kaggle/data/new-york-city-taxi-fare-prediction/nyc_mask-74.5_-72.8_40.5_41.8.png",
        "/kaggle/data/nyc_mask-74.5_-72.8_40.5_41.8.png",
    ]
    mask_path = next((p for p in local_candidates if os.path.exists(p)), None)

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
    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"], errors="coerce", utc=True
    )
    df["year"] = df["pickup_datetime"].dt.year
    df["month"] = df["pickup_datetime"].dt.month
    df["day"] = df["pickup_datetime"].dt.day
    df["hour"] = df["pickup_datetime"].dt.hour
    df["minute"] = df["pickup_datetime"].dt.minute
    df["second"] = df["pickup_datetime"].dt.second
    return df


def add_coordinate_features(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]
    return df


def add_distances_features(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]
    df["manhattan"] = manhattan(lat1, lon1, lat2, lon2)
    return df


def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    pred = np.asarray(prediction).reshape(-1)
    df = pd.DataFrame({id_column: raw_test[id_column].values, prediction_column: pred})
    df.to_csv(file_name, index=False)
    print("Output complete:", file_name)


def plot_loss_accuracy_rmse(history):
    if history is None or not hasattr(history, "history"):
        print("No Keras training history available to plot.")
        return

    plt.figure(figsize=(20, 10))
    plt.plot(history.history.get("loss", []))
    plt.plot(history.history.get("val_loss", []))
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper right")
    plt.show()

    plt.figure(figsize=(20, 10))
    plt.plot(history.history.get("rmse", []))
    plt.plot(history.history.get("val_rmse", []))
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
print("train_df add_time_features")
train_df = add_time_features(train_df)
print("test_df add_time_features")
test_df = add_time_features(test_df)
print("testKaggle add_time_features")
testKaggle = add_time_features(testKaggle)



## === cell 9
train_df.describe()



## === cell 10
print("train_df add_coordinate_features")
train_df = add_coordinate_features(train_df)
print("test_df add_coordinate_features Disabled!")
test_df = add_coordinate_features(test_df)
print("testKaggle add_coordinate_features")
testKaggle = add_coordinate_features(testKaggle)



## === cell 11
train_df.describe()



## === cell 12
print("train_df add_distances_features")
train_df = add_distances_features(train_df)
print("test_df add_distances_features")
test_df = add_distances_features(test_df)
print("testKaggle add_distances_features")
testKaggle = add_distances_features(testKaggle)

print("Done with Adding features")



## === cell 13
train_df.describe()



## === cell 14
plot = train_df.iloc[:2000].plot.scatter("fare_amount", "passenger_count")
plot = train_df.iloc[:2000].plot.scatter("fare_amount", "year")
plot = train_df.iloc[:2000].plot.scatter("fare_amount", "month")
plot = train_df.iloc[:2000].plot.scatter("fare_amount", "day")
plot = train_df.iloc[:2000].plot.scatter("fare_amount", "hour")



## === cell 15
dropped_columns = ["pickup_datetime", "key"]
train_df = train_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)

testKaggle_clean = testKaggle.drop(["pickup_datetime", "key"], axis=1)

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
train_df_main = train_df
validation_df_main = validation_df



## === cell 21
validation_df.describe()



## === cell 22
train_labels = train_df["fare_amount"].values.astype("float32")
validation_labels = validation_df["fare_amount"].values.astype("float32")
test_labels = test_df["fare_amount"].values.astype("float32")

train_labels_raw = train_labels
validation_labels_raw = validation_labels
test_labels_raw = test_labels

train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)
test_df = test_df.drop(["fare_amount"], axis=1)

print("Done with Labels (raw fare; aligned to Kaggle RMSE)")



## === cell 23
test_labels



## === cell 24
train_df.describe()



## === cell 25
validation_df.describe()



## === cell 26
test_df.describe()



## === cell 27
feature_cols = train_df.columns.tolist()

scaler = preprocessing.MinMaxScaler()
train_df_scaled = pd.DataFrame(
    scaler.fit_transform(train_df[feature_cols]),
    columns=feature_cols,
    index=train_df.index,
)
validation_df_scaled = pd.DataFrame(
    scaler.transform(validation_df[feature_cols]),
    columns=feature_cols,
    index=validation_df.index,
)
test_df_scaled = pd.DataFrame(
    scaler.transform(test_df[feature_cols]), columns=feature_cols, index=test_df.index
)
testKaggle_scaled = pd.DataFrame(
    scaler.transform(testKaggle_clean[feature_cols]),
    columns=feature_cols,
    index=testKaggle_clean.index,
)

train_df_scaled = train_df_scaled.astype("float32")
validation_df_scaled = validation_df_scaled.astype("float32")
test_df_scaled = test_df_scaled.astype("float32")
testKaggle_scaled = testKaggle_scaled.astype("float32")

X_train = train_df_scaled.to_numpy(dtype=np.float32, copy=False)
X_val = validation_df_scaled.to_numpy(dtype=np.float32, copy=False)
X_test = test_df_scaled.to_numpy(dtype=np.float32, copy=False)
X_kaggle = testKaggle_scaled.to_numpy(dtype=np.float32, copy=False)

y_train = np.asarray(train_labels, dtype=np.float32)
y_val = np.asarray(validation_labels, dtype=np.float32)
y_test = np.asarray(test_labels, dtype=np.float32)




## === cell 28
def rmse(y_true, y_pred):
    y_true = ops.cast(y_true, "float32")
    y_pred = ops.cast(y_pred, "float32")
    return ops.sqrt(ops.mean(ops.square(y_pred - y_true), axis=-1))




## === cell 29
_training_ok = True
history = None

try:
    mlp = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64, 32, 8),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size=BATCH_SIZE,
        learning_rate_init=LEARNING_RATE,
        max_iter=EPOCHS,
        shuffle=True,
        random_state=1,
        early_stopping=False,
        validation_fraction=0.0,
        n_iter_no_change=EPOCHS + 1,
        verbose=True,
    )
    mlp.fit(X_train, y_train)
except Exception as e:
    _training_ok = False
    mlp = None
    print(
        "Training failed; will fall back to mean-prediction submission. Error:", repr(e)
    )



## === cell 30
print(
    "Model training complete; skipping model_to_dot visualization (not available in this environment)."
)



## === cell 31
plot_loss_accuracy_rmse(history)



## === cell 32
if _training_ok and mlp is not None:
    train_pred = mlp.predict(X_train).astype("float32")
    train_rmse = float(np.sqrt(np.mean((train_pred - y_train) ** 2)))
    train_mae = float(np.mean(np.abs(train_pred - y_train)))
    train_mse = float(np.mean((train_pred - y_train) ** 2))
    print([train_mse, train_mae, train_rmse, train_mse])
    print("train mean_squared_error:", train_mse)
    print("train mae:", train_mae)
    print("train rmse:", train_rmse)
    print("train mse:", train_mse)



## === cell 33
if _training_ok and mlp is not None:
    val_pred = mlp.predict(X_val).astype("float32")
    val_rmse = float(np.sqrt(np.mean((val_pred - y_val) ** 2)))
    val_mae = float(np.mean(np.abs(val_pred - y_val)))
    val_mse = float(np.mean((val_pred - y_val) ** 2))
    print([val_mse, val_mae, val_rmse, val_mse])
    print("Validation mean_squared_error:", val_mse)
    print("Validation mae:", val_mae)
    print("Validation rmse:", val_rmse)
    print("Validation mse:", val_mse)



## === cell 34
if _training_ok and mlp is not None:
    test_pred = mlp.predict(X_test).astype("float32")
    test_rmse = float(np.sqrt(np.mean((test_pred - y_test) ** 2)))
    test_mae = float(np.mean(np.abs(test_pred - y_test)))
    test_mse = float(np.mean((test_pred - y_test) ** 2))
    print([test_mse, test_mae, test_rmse, test_mse])
    print("Test mean_squared_error:", test_mse)
    print("Test mae:", test_mae)
    print("Test rmse:", test_rmse)
    print("Test mse:", test_mse)



## === cell 35
if _training_ok and mlp is not None:
    validation_predictions = mlp.predict(X_val).astype("float32").flatten()
    validation_predictions = np.clip(validation_predictions, 0.0, None)

    plt.scatter(validation_labels_raw, validation_predictions)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.axis("equal")
    plt.xlim(plt.xlim())
    plt.ylim(plt.ylim())
    _ = plt.plot(
        [validation_predictions.min(), validation_predictions.max()],
        [validation_predictions.min(), validation_predictions.max()],
        "k--",
        lw=4,
    )



## === cell 36
if _training_ok and mlp is not None:
    test_predictions = mlp.predict(X_test).astype("float32").flatten()
    test_predictions = np.clip(test_predictions, 0.0, None)

    plt.scatter(test_labels_raw, test_predictions)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.axis("equal")
    plt.xlim(plt.xlim())
    plt.ylim(plt.ylim())
    _ = plt.plot(
        [test_predictions.min(), test_predictions.max()],
        [test_predictions.min(), test_predictions.max()],
        "k--",
        lw=4,
    )



## === cell 37
if _training_ok and mlp is not None:
    print(np.argmax(test_predictions))
    print(test_predictions[np.argmax(test_predictions)])
    print(test_labels_raw[np.argmax(test_predictions)])
    test_df.iloc[np.argmax(test_predictions)]



## === cell 38
if _training_ok and mlp is not None:
    print(np.argmin(test_predictions))
    print(test_predictions[np.argmin(test_predictions)])
    print(test_labels_raw[np.argmin(test_predictions)])
    test_df.iloc[np.argmin(test_predictions)]



## === cell 39
if _training_ok and mlp is not None:
    fig, ax = plt.subplots()
    ax.scatter(test_labels_raw, test_predictions)
    ax.plot(
        [test_labels_raw.min(), test_labels_raw.max()],
        [test_labels_raw.min(), test_labels_raw.max()],
        "k--",
        lw=4,
    )
    ax.set_xlabel("Measured")
    ax.set_ylabel("Predicted")
    plt.show()



## === cell 40
if _training_ok and mlp is not None:
    plt.figure(figsize=(20, 10))
    plt.plot(validation_labels_raw[:100])
    plt.plot(validation_predictions[:100])
    plt.title("Prediction vs Actual")
    plt.ylabel("Fare Amount")
    plt.xlabel("Transaction")
    plt.legend(["Actual", "prediction"], loc="upper right")
    plt.show()



## === cell 41
if _training_ok and mlp is not None:
    plt.figure(figsize=(20, 10))
    plt.plot(test_labels_raw[:100])
    plt.plot(test_predictions[:100])
    plt.title("Prediction vs Actual")
    plt.ylabel("Fare Amount")
    plt.xlabel("Transaction")
    plt.legend(["Actual", "prediction"], loc="upper right")
    plt.show()



## === cell 42
if _training_ok and mlp is not None:
    error = validation_predictions - validation_labels_raw
    plt.hist(error, bins=100)
    plt.xlabel("Prediction Error")
    _ = plt.ylabel("Count")



## === cell 43
if _training_ok and mlp is not None:
    error = test_predictions - test_labels_raw
    plt.hist(error, bins=50)
    plt.xlabel("Prediction Error")
    _ = plt.ylabel("Count")



## === cell 44
if _training_ok and mlp is not None:
    print(len(error))
    errorGreaterZero = error[(np.logical_or(error <= -1, error >= 1))]
    print(len(errorGreaterZero))
    plt.hist(errorGreaterZero, bins=100)
    plt.xlabel("Prediction Error")
    _ = plt.ylabel("Count")



## === cell 45
testKaggle_scaled



## === cell 46
if _training_ok and mlp is not None:
    predictionKaggle = mlp.predict(X_kaggle).astype("float32").reshape(-1)
    predictionKaggle = np.clip(predictionKaggle, 0.0, None)
    predictionKaggle = np.where(
        np.isfinite(predictionKaggle), predictionKaggle, 0.0
    ).astype("float32")
else:
    predictionKaggle = np.full(
        shape=(len(testKaggle),),
        fill_value=float(trainKaggle["fare_amount"].mean()),
        dtype="float32",
    )

output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)
print("Wrote submission:", SUBMISSION_NAME)
print(pd.read_csv(SUBMISSION_NAME).head())
