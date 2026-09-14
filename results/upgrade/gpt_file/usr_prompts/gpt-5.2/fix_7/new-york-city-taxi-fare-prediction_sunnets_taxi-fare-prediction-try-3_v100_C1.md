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

4.2871

# 6. Current score

15.25183

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 147.37347) has done: 'I fix the runtime blockers first: the NYC water-mask cleaning currently tries to read an image from a URL (no internet) and crashes, so I make that step safely skip when the mask can’t be loaded. Next, I correct the scaler bug where the `manhattan` feature in `test_df_scaled` is mistakenly transformed using `passenger_count`, and I make scaling consistent by using a single `MinMaxScaler` fit on the training set for all features (this is score-improving and still preserves the same modeling approach). Finally, I update the Keras optimizer call for Keras 3 (`Adam(learning_rate=...)`) and remove the optional model visualization import that doesn’t exist, ensuring the model trains and a valid `submissiontry_water.csv` is produced.'
- What this solution (achieved 15.20475) has done: 'I fix the runtime crash happening at the very first imports by avoiding the problematic protobuf/tensorflow initialization path and switching the Keras import to the installed `keras` (Keras 3) API, which is compatible with the environment. Then I correct the training CSV loading to actually include the `fare_amount` target column (it’s currently excluded due to `usecols`, which causes the model to train on misaligned/invalid labels and leads to the extremely bad RMSE). Finally, I keep the same model and feature logic, but ensure all numeric inputs are float32 and clamp negative predictions to 0 before writing the submission to improve RMSE without changing the core approach.'
- What this solution (achieved 67.3151) has done: 'I fix the Keras 3 backend RMSE metric that currently crashes training by switching it to a TensorFlow-based implementation that works in this environment, without changing the model or loss. I also fix the initial protobuf/TensorFlow initialization crash by explicitly using the installed `tf_keras` stack (compatible with this Kaggle image) while keeping the same Sequential/Dense architecture and training loop. Finally, I make the input file paths robust to this notebook’s filesystem layout (preferring `/kaggle/input/...` when available) so the script runs end-to-end and reliably writes `submissiontry_water.csv` with the required `key,fare_amount` columns.'
- What this solution (achieved 6.04401) has done: 'I fix the runtime crash in the first cell caused by an incompatible `tensorflow` import (protobuf `MessageFactory.GetPrototype` error) by switching fully to the already-installed `tf_keras` stack and not importing standalone `tensorflow`. Then I correct a major data bug that is causing the very poor RMSE: the training CSV load currently omits the `key` column, which later breaks the expected submission ID alignment logic and also prevents consistent end-to-end feature handling; I load `key` for train and keep it through cleaning/splitting while still dropping it from model features. Finally, I keep the same feature engineering and model architecture/training loop, but ensure we never compile with the meaningless `accuracy` metric for a regression model (it can destabilize logs/early stopping behavior) while leaving loss/optimizer unchanged; this is score-improving but does not change the training objective.'
- What this solution (achieved 15.25184) has done: 'I fix the immediate runtime crash caused by importing `tf_keras` (protobuf `MessageFactory.GetPrototype` issue) by switching to the installed `keras` (Keras 3) API, which avoids importing standalone TensorFlow while keeping the same Sequential Dense architecture and training loop. I also make `ModelCheckpoint` compatible with Keras 3 by saving to a `.keras` file instead of legacy `.h5`, preventing checkpoint write errors. To nudge RMSE toward the target with minimal semantic change, I log-transform the target (`fare_amount`) during training and invert-transform predictions for evaluation/submission (a standard regression calibration that preserves the same model/feature pipeline but typically improves RMSE). Finally, I keep the submission format/paths identical and ensure the CSV is written correctly.'
- What this solution (achieved 15.25183) has done: 'I fix the immediate runtime crash by forcing Keras to use the NumPy backend (so it won’t try to initialize TensorFlow/protobuf) while keeping the same model and training loop. Then I fix the `rmse` metric implementation to use `keras.ops` (Keras 3 backend-agnostic ops) because `keras.backend.cast` is not available in this environment, which currently prevents `fit()` and all `evaluate()` calls from running. These changes are score-neutral in intent (they restore training/evaluation), and your existing log1p target transform + inverse at inference is preserved as-is. Finally, I make plotting robust to missing `history` (in case training fails) and ensure the submission CSV is always written with the required `key,fare_amount` columns.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

os.environ.setdefault("KERAS_BACKEND", "numpy")

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
from keras import regularizers
from keras import ops


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

np.random.seed(1)

print("Keras backend:", getattr(keras, "backend", lambda: "unknown")())
print("Using TRAIN_PATH:", TRAIN_PATH)
print("Using TEST_PATH :", TEST_PATH)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1266814093.py in <cell line: 0>()
     50 np.random.seed(1)
     51 
---> 52 print("Keras backend:", getattr(keras, "backend", lambda: "unknown")())
     53 print("Using TRAIN_PATH:", TRAIN_PATH)
     54 print("Using TEST_PATH :", TEST_PATH)

TypeError: 'module' object is not callable

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
    df = pd.DataFrame(prediction, columns=[prediction_column])
    df[id_column] = raw_test[id_column].values
    df[[id_column, prediction_column]].to_csv(file_name, index=False)
    print("Output complete:", file_name)


def plot_loss_accuracy_rmse(history):
    if history is None:
        print("No training history available to plot.")
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
train_labels_raw = train_df["fare_amount"].values.astype("float32")
validation_labels_raw = validation_df["fare_amount"].values.astype("float32")
test_labels_raw = test_df["fare_amount"].values.astype("float32")

train_labels = np.log1p(train_labels_raw).astype("float32")
validation_labels = np.log1p(validation_labels_raw).astype("float32")
test_labels = np.log1p(test_labels_raw).astype("float32")

train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)
test_df = test_df.drop(["fare_amount"], axis=1)

print("Done with Labels (log1p transform)")



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




## === cell 28
def rmse(y_true, y_pred):
    y_true = ops.cast(y_true, "float32")
    y_pred = ops.cast(y_pred, "float32")
    return ops.sqrt(ops.mean(ops.square(y_pred - y_true), axis=-1))




## === cell 29
checkpoint = ModelCheckpoint(filepath="my_model.keras", verbose=1, save_best_only=True)

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

adam = optimizers.Adam(learning_rate=LEARNING_RATE)

model.compile(loss="mean_squared_error", optimizer=adam, metrics=["mae", rmse, "mse"])

print("Dataset size: %s" % DATASET_SIZE)
print("Epochs: %s" % EPOCHS)
print("Learning rate: %s" % LEARNING_RATE)
print("Batch size: %s" % BATCH_SIZE)
print("Input dimension: %s" % train_df_scaled.shape[1])
print("Features used: %s" % train_df.columns)
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



## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NotImplementedError                       Traceback (most recent call last)
/tmp/ipykernel_11/4190704846.py in <cell line: 0>()
     28 model.summary()
     29 
---> 30 history = model.fit(
     31     x=train_df_scaled,
     32     y=train_labels,

/usr/local/lib/python3.11/dist-packages/keras/src/backend/numpy/trainer.py in fit(self, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_batch_size, validation_freq)
    167         validation_freq=1,
    168     ):
--> 169         raise NotImplementedError("fit not implemented for NumPy backend.")
    170 
    171     @traceback_utils.filter_traceback

NotImplementedError: fit not implemented for NumPy backend.

## === cell 30
print(
    "Model training complete; skipping model_to_dot visualization (not available in this environment)."
)



## === cell 31
plot_loss_accuracy_rmse(history)



## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2579130885.py in <cell line: 0>()
----> 1 plot_loss_accuracy_rmse(history)
      2 

NameError: name 'history' is not defined

## === cell 32
score = model.evaluate(train_df_scaled, train_labels, verbose=1)
print(score)
print("train mean_squared_error:", score[0])
print("train mae:", score[1])
print("train rmse:", score[2])
print("train mse:", score[3])



## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/4103044261.py in <cell line: 0>()
----> 1 score = model.evaluate(train_df_scaled, train_labels, verbose=1)
      2 print(score)
      3 print("train mean_squared_error:", score[0])
      4 print("train mae:", score[1])
      5 print("train rmse:", score[2])

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/backend/numpy/numpy.py in sum(x, axis, keepdims)
   1152 def sum(x, axis=None, keepdims=False):
   1153     axis = standardize_axis_for_numpy(axis)
-> 1154     dtype = standardize_dtype(x.dtype)
   1155     # follow jax's rule
   1156     if dtype in ("bool", "int8", "int16"):

AttributeError: 'list' object has no attribute 'dtype'

## === cell 33
score = model.evaluate(validation_df_scaled, validation_labels, verbose=1)
print(score)
print("Validation mean_squared_error:", score[0])
print("Validation mae:", score[1])
print("Validation rmse:", score[2])
print("Validation mse:", score[3])



## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2701956346.py in <cell line: 0>()
----> 1 score = model.evaluate(validation_df_scaled, validation_labels, verbose=1)
      2 print(score)
      3 print("Validation mean_squared_error:", score[0])
      4 print("Validation mae:", score[1])
      5 print("Validation rmse:", score[2])

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/backend/numpy/numpy.py in sum(x, axis, keepdims)
   1152 def sum(x, axis=None, keepdims=False):
   1153     axis = standardize_axis_for_numpy(axis)
-> 1154     dtype = standardize_dtype(x.dtype)
   1155     # follow jax's rule
   1156     if dtype in ("bool", "int8", "int16"):

AttributeError: 'list' object has no attribute 'dtype'

## === cell 34
score = model.evaluate(test_df_scaled, test_labels, verbose=1)
print(score)
print("Test mean_squared_error:", score[0])
print("Test mae:", score[1])
print("Test rmse:", score[2])
print("Test mse:", score[3])



## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/369954523.py in <cell line: 0>()
----> 1 score = model.evaluate(test_df_scaled, test_labels, verbose=1)
      2 print(score)
      3 print("Test mean_squared_error:", score[0])
      4 print("Test mae:", score[1])
      5 print("Test rmse:", score[2])

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/backend/numpy/numpy.py in sum(x, axis, keepdims)
   1152 def sum(x, axis=None, keepdims=False):
   1153     axis = standardize_axis_for_numpy(axis)
-> 1154     dtype = standardize_dtype(x.dtype)
   1155     # follow jax's rule
   1156     if dtype in ("bool", "int8", "int16"):

AttributeError: 'list' object has no attribute 'dtype'

## === cell 35
validation_predictions_log = model.predict(validation_df_scaled).flatten()
validation_predictions = np.expm1(validation_predictions_log).astype("float32")
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
test_predictions_log = model.predict(test_df_scaled).flatten()
test_predictions = np.expm1(test_predictions_log).astype("float32")
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
print(np.argmax(test_predictions))
print(test_predictions[np.argmax(test_predictions)])
print(test_labels_raw[np.argmax(test_predictions)])
test_df.iloc[np.argmax(test_predictions)]



## === cell 38
print(np.argmin(test_predictions))
print(test_predictions[np.argmin(test_predictions)])
print(test_labels_raw[np.argmin(test_predictions)])
test_df.iloc[np.argmin(test_predictions)]



## === cell 39
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
plt.figure(figsize=(20, 10))
plt.plot(validation_labels_raw[:100])
plt.plot(validation_predictions[:100])
plt.title("Prediction vs Actual")
plt.ylabel("Fare Amount")
plt.xlabel("Transaction")
plt.legend(["Actual", "prediction"], loc="upper right")
plt.show()



## === cell 41
plt.figure(figsize=(20, 10))
plt.plot(test_labels_raw[:100])
plt.plot(test_predictions[:100])
plt.title("Prediction vs Actual")
plt.ylabel("Fare Amount")
plt.xlabel("Transaction")
plt.legend(["Actual", "prediction"], loc="upper right")
plt.show()



## === cell 42
error = validation_predictions - validation_labels_raw
plt.hist(error, bins=100)
plt.xlabel("Prediction Error")
_ = plt.ylabel("Count")



## === cell 43
error = test_predictions - test_labels_raw
plt.hist(error, bins=50)
plt.xlabel("Prediction Error")
_ = plt.ylabel("Count")



## === cell 44
print(len(error))
errorGreaterZero = error[(np.logical_or(error <= -1, error >= 1))]
print(len(errorGreaterZero))
plt.hist(errorGreaterZero, bins=100)
plt.xlabel("Prediction Error")
_ = plt.ylabel("Count")



## === cell 45
testKaggle_scaled



## === cell 46
predictionKaggle_log = model.predict(
    testKaggle_scaled, batch_size=128, verbose=1
).astype("float32")
predictionKaggle = np.expm1(predictionKaggle_log).astype("float32")
predictionKaggle = np.clip(predictionKaggle, 0.0, None)



## === cell 47
output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)
print("Wrote submission:", SUBMISSION_NAME)
print(pd.read_csv(SUBMISSION_NAME).head())
