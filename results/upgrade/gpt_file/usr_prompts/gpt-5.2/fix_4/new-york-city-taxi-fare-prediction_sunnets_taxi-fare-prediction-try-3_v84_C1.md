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

5.16721

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 124.47956) has done: 'I fix the early import/runtime crash by removing the protobuf-dependent `tf_keras` usage and switching to the already-installed `tensorflow.keras`, which avoids the `MessageFactory.GetPrototype` error and keeps the same Keras API and model structure. I also fix the scaler crash by ensuring no string/object columns (specifically `key`) leak into the feature matrix before `MinMaxScaler`, and by aligning train/validation/test/testKaggle feature columns after feature engineering. Finally, I make submission writing robust (correct columns, correct row order, `.csv` suffix) so the notebook always produces a valid `submissiontry_water.csv` end-to-end without changing the core model/training loop.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("KERAS_BACKEND", "tf_keras")

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers

np.random.seed(1)
tf.random.set_seed(1)

TRAIN_PATH = "/kaggle/input/train.csv"
TEST_PATH = "/kaggle/input/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 0.001
DATASET_SIZE = 80000




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def remove_datapoints_from_water(df):
    """
    Bugfix: original code attempted to read a NYC land/water mask from an external URL,
    which fails on Kaggle (no internet and matplotlib doesn't read URL directly).
    Minimal, score-preserving fallback: if mask cannot be loaded locally, skip this filter.
    """

    def lonlat_to_xy(longitude, latitude, dx, dy, BB):
        return (dx * (longitude - BB[0]) / (BB[1] - BB[0])).astype("int"), (
            dy - dy * (latitude - BB[2]) / (BB[3] - BB[2])
        ).astype("int")

    BB = (-74.5, -72.8, 40.5, 41.8)

    local_mask_path = "/kaggle/input/nyc_mask-74.5_-72.8_40.5_41.8.png"
    if not os.path.exists(local_mask_path):
        return df

    try:
        nyc_mask = plt.imread(local_mask_path)[:, :, 0] > 0.9
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
    except Exception:
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
    else:
        print(" (No fare_amount in df; skipping outlier removal)")

    df = df[(df["passenger_count"] > 0)]
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
    return 1 if (row["hour"] <= 3) else 0


def night(row):
    return (
        1
        if (((row["hour"] >= 20) or (row["hour"] <= 5)) and (row["weekday"] < 5))
        else 0
    )


def rush_hour(row):
    if ((row["hour"] <= 20) and (row["hour"] >= 16)) and (row["weekday"] < 5):
        return 1
    else:
        return 0


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
    df["weekday"] = df["pickup_datetime"].dt.weekday
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
    df = pd.DataFrame(prediction, columns=[prediction_column])
    df[id_column] = raw_test[id_column].values
    df = df[[id_column, prediction_column]]
    df.to_csv(file_name, index=False)
    print("Output complete:", file_name, "rows:", len(df))


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



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/997133801.py in <cell line: 0>()
     11 
     12 trainKaggle = pd.read_csv(
---> 13     TRAIN_PATH,
     14     nrows=DATASET_SIZE,
     15     dtype=datatypes,

NameError: name 'TRAIN_PATH' is not defined

## === cell 3
train_df, test_df = train_test_split(trainKaggle, test_size=0.50, random_state=1)
test_df = test_df[:10000]



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/19325734.py in <cell line: 0>()
----> 1 train_df, test_df = train_test_split(trainKaggle, test_size=0.50, random_state=1)
      2 test_df = test_df[:10000]
      3 

NameError: name 'trainKaggle' is not defined

## === cell 4
print("testKaggle Size %d" % len(testKaggle))
print("train_df Size %d" % len(train_df))
print("test_df Size %d" % len(test_df))



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/395627971.py in <cell line: 0>()
----> 1 print("testKaggle Size %d" % len(testKaggle))
      2 print("train_df Size %d" % len(train_df))
      3 print("test_df Size %d" % len(test_df))
      4 

NameError: name 'testKaggle' is not defined

## === cell 5
train_df.describe()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1577644986.py in <cell line: 0>()
----> 1 train_df.describe()
      2 

NameError: name 'train_df' is not defined

## === cell 6
test_df.describe()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4294142452.py in <cell line: 0>()
----> 1 test_df.describe()
      2 

NameError: name 'test_df' is not defined

## === cell 7
print("train_df clean")
train_df = clean(train_df)
test_df = clean(test_df)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1475834959.py in <cell line: 0>()
      1 print("train_df clean")
----> 2 train_df = clean(train_df)
      3 test_df = clean(test_df)
      4 

NameError: name 'train_df' is not defined

## === cell 8
print("train_df add_time_features")
train_df = add_time_features(train_df)
print("test_df add_time_features")
test_df = add_time_features(test_df)
print("testKaggle add_time_features")
testKaggle = add_time_features(testKaggle)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1145226127.py in <cell line: 0>()
      1 print("train_df add_time_features")
----> 2 train_df = add_time_features(train_df)
      3 print("test_df add_time_features")
      4 test_df = add_time_features(test_df)
      5 print("testKaggle add_time_features")

NameError: name 'train_df' is not defined

## === cell 9
train_df.describe()



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1577644986.py in <cell line: 0>()
----> 1 train_df.describe()
      2 

NameError: name 'train_df' is not defined

## === cell 10
print("train_df add_coordinate_features")
train_df = add_coordinate_features(train_df)
print("test_df add_coordinate_features Disabled!")
test_df = add_coordinate_features(test_df)
print("testKaggle add_coordinate_features")
testKaggle = add_coordinate_features(testKaggle)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3943412115.py in <cell line: 0>()
      1 print("train_df add_coordinate_features")
----> 2 train_df = add_coordinate_features(train_df)
      3 print("test_df add_coordinate_features Disabled!")
      4 test_df = add_coordinate_features(test_df)
      5 print("testKaggle add_coordinate_features")

NameError: name 'train_df' is not defined

## === cell 11
train_df.describe()



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1577644986.py in <cell line: 0>()
----> 1 train_df.describe()
      2 

NameError: name 'train_df' is not defined

## === cell 12
print("train_df add_distances_features")
train_df = add_distances_features(train_df)
print("test_df add_distances_features")
test_df = add_distances_features(test_df)
print("testKaggle add_distances_features")
testKaggle = add_distances_features(testKaggle)

print("Done with Adding features")



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/470363350.py in <cell line: 0>()
      1 print("train_df add_distances_features")
----> 2 train_df = add_distances_features(train_df)
      3 print("test_df add_distances_features")
      4 test_df = add_distances_features(test_df)
      5 print("testKaggle add_distances_features")

NameError: name 'train_df' is not defined

## === cell 13
train_df.describe()



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1577644986.py in <cell line: 0>()
----> 1 train_df.describe()
      2 

NameError: name 'train_df' is not defined

## === cell 14
plot = train_df.iloc[:2000].plot.scatter("latdiff", "londiff")
plot = train_df.iloc[:2000].plot.scatter("fare_amount", "passenger_count")
plot = train_df.iloc[:2000].plot.scatter("fare_amount", "year")
plot = train_df.iloc[:2000].plot.scatter("fare_amount", "month")
plot = train_df.iloc[:2000].plot.scatter("fare_amount", "day")
plot = train_df.iloc[:2000].plot.scatter("fare_amount", "hour")
plot = train_df.iloc[:2000].plot.scatter("fare_amount", "weekday")



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1321718946.py in <cell line: 0>()
----> 1 plot = train_df.iloc[:2000].plot.scatter("latdiff", "londiff")
      2 plot = train_df.iloc[:2000].plot.scatter("fare_amount", "passenger_count")
      3 plot = train_df.iloc[:2000].plot.scatter("fare_amount", "year")
      4 plot = train_df.iloc[:2000].plot.scatter("fare_amount", "month")
      5 plot = train_df.iloc[:2000].plot.scatter("fare_amount", "day")

NameError: name 'train_df' is not defined

## === cell 15
dropped_columns = ["pickup_datetime", "key"]

train_df = train_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)

testKaggle_clean = testKaggle.drop(["pickup_datetime", "key"], axis=1)

print("Done with dropped_columns")



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/165149784.py in <cell line: 0>()
      1 dropped_columns = ["pickup_datetime", "key"]
      2 
----> 3 train_df = train_df.drop(dropped_columns, axis=1)
      4 test_df = test_df.drop(dropped_columns, axis=1)
      5 

NameError: name 'train_df' is not defined

## === cell 16
train_df.shape



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/186828624.py in <cell line: 0>()
----> 1 train_df.shape
      2 

NameError: name 'train_df' is not defined

## === cell 17
train_df.describe()



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1577644986.py in <cell line: 0>()
----> 1 train_df.describe()
      2 

NameError: name 'train_df' is not defined

## === cell 18
test_df.describe()



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4294142452.py in <cell line: 0>()
----> 1 test_df.describe()
      2 

NameError: name 'test_df' is not defined

## === cell 19
train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=1)



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1154705564.py in <cell line: 0>()
----> 1 train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=1)
      2 

NameError: name 'train_df' is not defined

## === cell 20
train_df_main = train_df
validation_df_main = validation_df



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2901810251.py in <cell line: 0>()
----> 1 train_df_main = train_df
      2 validation_df_main = validation_df
      3 

NameError: name 'train_df' is not defined

## === cell 21
validation_df.describe()



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3277078349.py in <cell line: 0>()
----> 1 validation_df.describe()
      2 

NameError: name 'validation_df' is not defined

## === cell 22
train_labels = train_df["fare_amount"].values
validation_labels = validation_df["fare_amount"].values
test_labels = test_df["fare_amount"].values

train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)
test_df = test_df.drop(["fare_amount"], axis=1)

print("Done with Labels")



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/786339670.py in <cell line: 0>()
----> 1 train_labels = train_df["fare_amount"].values
      2 validation_labels = validation_df["fare_amount"].values
      3 test_labels = test_df["fare_amount"].values
      4 
      5 train_df = train_df.drop(["fare_amount"], axis=1)

NameError: name 'train_df' is not defined

## === cell 23
test_labels



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/805156543.py in <cell line: 0>()
----> 1 test_labels
      2 

NameError: name 'test_labels' is not defined

## === cell 24
train_df.describe()



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1577644986.py in <cell line: 0>()
----> 1 train_df.describe()
      2 

NameError: name 'train_df' is not defined

## === cell 25
validation_df.describe()



## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3277078349.py in <cell line: 0>()
----> 1 validation_df.describe()
      2 

NameError: name 'validation_df' is not defined

## === cell 26
test_df.describe()



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4294142452.py in <cell line: 0>()
----> 1 test_df.describe()
      2 

NameError: name 'test_df' is not defined

## === cell 27
feature_cols = list(train_df.columns)
validation_df = validation_df[feature_cols]
test_df = test_df[feature_cols]
testKaggle_clean = testKaggle_clean[feature_cols]

for _df in (train_df, validation_df, test_df, testKaggle_clean):
    for c in _df.columns:
        _df[c] = pd.to_numeric(_df[c], errors="coerce")
    _df.fillna(_df.median(numeric_only=True), inplace=True)

scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_df)
testKaggle_scaled = scaler.transform(testKaggle_clean)



## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2977032498.py in <cell line: 0>()
----> 1 feature_cols = list(train_df.columns)
      2 validation_df = validation_df[feature_cols]
      3 test_df = test_df[feature_cols]
      4 testKaggle_clean = testKaggle_clean[feature_cols]
      5 

NameError: name 'train_df' is not defined

## === cell 28
test_scaled



## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1145478310.py in <cell line: 0>()
----> 1 test_scaled
      2 

NameError: name 'test_scaled' is not defined

## === cell 29
from tensorflow.keras import backend


def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))




## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1183545402.py in <cell line: 0>()
----> 1 from tensorflow.keras import backend
      2 
      3 
      4 def rmse(y_true, y_pred):
      5     return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

/usr/local/lib/python3.11/dist-packages/tensorflow/__init__.py in <module>
    465     importlib.import_module("tf_keras.src.optimizers")
    466   else:
--> 467     importlib.import_module("keras.src.optimizers")
    468 except (ImportError, AttributeError):
    469   pass

/usr/lib/python3.11/importlib/__init__.py in import_module(name, package)
    124                 break
    125             level += 1
--> 126     return _bootstrap._gcd_import(name[level:], package, level)
    127 
    128 

/usr/local/lib/python3.11/dist-packages/keras/__init__.py in <module>
      1 # DO NOT EDIT. Generated by api_gen.sh
----> 2 from keras.api import DTypePolicy
      3 from keras.api import FloatDTypePolicy
      4 from keras.api import Function
      5 from keras.api import Initializer

/usr/local/lib/python3.11/dist-packages/keras/api/__init__.py in <module>
      6 
      7 
----> 8 from keras.api import activations
      9 from keras.api import applications
     10 from keras.api import backend

/usr/local/lib/python3.11/dist-packages/keras/api/activations/__init__.py in <module>
      5 """
      6 
----> 7 from keras.src.activations import deserialize
      8 from keras.src.activations import get
      9 from keras.src.activations import serialize

/usr/local/lib/python3.11/dist-packages/keras/src/__init__.py in <module>
----> 1 from keras.src import activations
      2 from keras.src import applications
      3 from keras.src import backend
      4 from keras.src import constraints
      5 from keras.src import datasets

/usr/local/lib/python3.11/dist-packages/keras/src/activations/__init__.py in <module>
      1 import types
      2 
----> 3 from keras.src.activations.activations import celu
      4 from keras.src.activations.activations import elu
      5 from keras.src.activations.activations import exponential

/usr/local/lib/python3.11/dist-packages/keras/src/activations/activations.py in <module>
----> 1 from keras.src import backend
      2 from keras.src import ops
      3 from keras.src.api_export import keras_export
      4 
      5 

/usr/local/lib/python3.11/dist-packages/keras/src/backend/__init__.py in <module>
     57     distribution_lib = None
     58 else:
---> 59     raise ValueError(f"Unable to import backend : {backend()}")
     60 
     61 

ValueError: Unable to import backend : tf_keras

## === cell 30
checkpoint = ModelCheckpoint(filepath="my_model.h5", verbose=1, save_best_only=True)

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

adam = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
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



## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3063984128.py in <cell line: 0>()
----> 1 checkpoint = ModelCheckpoint(filepath="my_model.h5", verbose=1, save_best_only=True)
      2 
      3 model = Sequential()
      4 model.add(
      5     Dense(

NameError: name 'ModelCheckpoint' is not defined

## === cell 31
pass



## === cell 32
plot_loss_accuracy_rmse(history)



## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2579130885.py in <cell line: 0>()
----> 1 plot_loss_accuracy_rmse(history)
      2 

NameError: name 'history' is not defined

## === cell 33
score = model.evaluate(train_df_scaled, train_labels, verbose=1)
print(score)
print("train mean_squared_error:", score[0])
print("train mae:", score[1])
print("train rmse:", score[2])
print("train mse:", score[3])



## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4103044261.py in <cell line: 0>()
----> 1 score = model.evaluate(train_df_scaled, train_labels, verbose=1)
      2 print(score)
      3 print("train mean_squared_error:", score[0])
      4 print("train mae:", score[1])
      5 print("train rmse:", score[2])

NameError: name 'model' is not defined

## === cell 34
score = model.evaluate(validation_df_scaled, validation_labels, verbose=1)
print(score)
print("Validation mean_squared_error:", score[0])
print("Validation mae:", score[1])
print("Validation rmse:", score[2])
print("Validation mse:", score[3])



## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2701956346.py in <cell line: 0>()
----> 1 score = model.evaluate(validation_df_scaled, validation_labels, verbose=1)
      2 print(score)
      3 print("Validation mean_squared_error:", score[0])
      4 print("Validation mae:", score[1])
      5 print("Validation rmse:", score[2])

NameError: name 'model' is not defined

## === cell 35
score = model.evaluate(test_scaled, test_labels, verbose=1)
print(score)
print("Test mean_squared_error:", score[0])
print("Test mae:", score[1])
print("Test rmse:", score[2])
print("Test mse:", score[3])



## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1107548853.py in <cell line: 0>()
----> 1 score = model.evaluate(test_scaled, test_labels, verbose=1)
      2 print(score)
      3 print("Test mean_squared_error:", score[0])
      4 print("Test mae:", score[1])
      5 print("Test rmse:", score[2])

NameError: name 'model' is not defined

## === cell 36
validation_predictions = model.predict(validation_df_scaled).flatten()

plt.scatter(validation_labels, validation_predictions)
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



## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2392837.py in <cell line: 0>()
----> 1 validation_predictions = model.predict(validation_df_scaled).flatten()
      2 
      3 plt.scatter(validation_labels, validation_predictions)
      4 plt.xlabel("True Values")
      5 plt.ylabel("Predictions")

NameError: name 'model' is not defined

## === cell 37
test_predictions = model.predict(test_scaled).flatten()

plt.scatter(test_labels, test_predictions)
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



## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/481671663.py in <cell line: 0>()
----> 1 test_predictions = model.predict(test_scaled).flatten()
      2 
      3 plt.scatter(test_labels, test_predictions)
      4 plt.xlabel("True Values")
      5 plt.ylabel("Predictions")

NameError: name 'model' is not defined

## === cell 38
print(np.argmax(test_predictions))
print(test_predictions[np.argmax(test_predictions)])
print(test_labels[np.argmax(test_predictions)])
test_df.iloc[np.argmax(test_predictions)]



## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3782083718.py in <cell line: 0>()
----> 1 print(np.argmax(test_predictions))
      2 print(test_predictions[np.argmax(test_predictions)])
      3 print(test_labels[np.argmax(test_predictions)])
      4 test_df.iloc[np.argmax(test_predictions)]
      5 

NameError: name 'test_predictions' is not defined

## === cell 39
print(np.argmin(test_predictions))
print(test_predictions[np.argmin(test_predictions)])
print(test_labels[np.argmin(test_predictions)])
test_df.iloc[np.argmin(test_predictions)]



## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1775569351.py in <cell line: 0>()
----> 1 print(np.argmin(test_predictions))
      2 print(test_predictions[np.argmin(test_predictions)])
      3 print(test_labels[np.argmin(test_predictions)])
      4 test_df.iloc[np.argmin(test_predictions)]
      5 

NameError: name 'test_predictions' is not defined

## === cell 40
fig, ax = plt.subplots()
ax.scatter(test_labels, test_predictions)
ax.plot(
    [test_labels.min(), test_labels.max()],
    [test_labels.min(), test_labels.max()],
    "k--",
    lw=4,
)
ax.set_xlabel("Measured")
ax.set_ylabel("Predicted")
plt.show()



## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3918075604.py in <cell line: 0>()
      1 fig, ax = plt.subplots()
----> 2 ax.scatter(test_labels, test_predictions)
      3 ax.plot(
      4     [test_labels.min(), test_labels.max()],
      5     [test_labels.min(), test_labels.max()],

NameError: name 'test_labels' is not defined

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



## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3059730636.py in <cell line: 0>()
      1 plt.figure(figsize=(20, 10))
----> 2 plt.plot(validation_labels[:100])
      3 plt.plot(validation_predictions[:100])
      4 plt.title("Prediction vs Actual")
      5 plt.ylabel("Fare Amount")

NameError: name 'validation_labels' is not defined

## === cell 42
plt.figure(figsize=(20, 10))
plt.plot(test_labels[:100])
plt.plot(test_predictions[:100])
plt.title("Prediction vs Actual")
plt.ylabel("Fare Amount")
plt.xlabel("Transaction")
plt.legend(["Actual", "prediction"], loc="upper right")
plt.show()



## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1260486629.py in <cell line: 0>()
      1 plt.figure(figsize=(20, 10))
----> 2 plt.plot(test_labels[:100])
      3 plt.plot(test_predictions[:100])
      4 plt.title("Prediction vs Actual")
      5 plt.ylabel("Fare Amount")

NameError: name 'test_labels' is not defined

## === cell 43
error = validation_predictions - validation_labels
plt.hist(error, bins=100)
plt.xlabel("Prediction Error")
_ = plt.ylabel("Count")



## --- ERROR in cell 43, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1386752517.py in <cell line: 0>()
----> 1 error = validation_predictions - validation_labels
      2 plt.hist(error, bins=100)
      3 plt.xlabel("Prediction Error")
      4 _ = plt.ylabel("Count")
      5 

NameError: name 'validation_predictions' is not defined

## === cell 44
error = test_predictions - test_labels
plt.hist(error, bins=50)
plt.xlabel("Prediction Error")
_ = plt.ylabel("Count")



## --- ERROR in cell 44, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/281852328.py in <cell line: 0>()
----> 1 error = test_predictions - test_labels
      2 plt.hist(error, bins=50)
      3 plt.xlabel("Prediction Error")
      4 _ = plt.ylabel("Count")
      5 

NameError: name 'test_predictions' is not defined

## === cell 45
print(len(error))

errorGreaterZero = error[(np.logical_or(error <= -1, error >= 1))]
print(len(errorGreaterZero))

plt.hist(errorGreaterZero, bins=100)
plt.xlabel("Prediction Error")
_ = plt.ylabel("Count")



## --- ERROR in cell 45, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2451681468.py in <cell line: 0>()
----> 1 print(len(error))
      2 
      3 errorGreaterZero = error[(np.logical_or(error <= -1, error >= 1))]
      4 print(len(errorGreaterZero))
      5 

NameError: name 'error' is not defined

## === cell 46
predictionKaggle = model.predict(testKaggle_scaled, batch_size=128, verbose=1)



## --- ERROR in cell 46, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/522070667.py in <cell line: 0>()
----> 1 predictionKaggle = model.predict(testKaggle_scaled, batch_size=128, verbose=1)
      2 

NameError: name 'model' is not defined

## === cell 47
predictionKaggle = predictionKaggle.reshape(-1)
predictionKaggle = np.maximum(predictionKaggle, 0.0)

output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)

sub_df = pd.read_csv(SUBMISSION_NAME)
assert list(sub_df.columns) == [
    "key",
    "fare_amount",
], f"Bad columns: {sub_df.columns.tolist()}"
assert len(sub_df) == len(
    testKaggle
), f"Row mismatch: sub={len(sub_df)} test={len(testKaggle)}"

print("Saved submission to:", SUBMISSION_NAME)
print(sub_df.head())
print("Submission rows:", len(sub_df))

## --- ERROR in cell 47, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2155544584.py in <cell line: 0>()
----> 1 predictionKaggle = predictionKaggle.reshape(-1)
      2 predictionKaggle = np.maximum(predictionKaggle, 0.0)
      3 
      4 output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)
      5 

NameError: name 'predictionKaggle' is not defined
