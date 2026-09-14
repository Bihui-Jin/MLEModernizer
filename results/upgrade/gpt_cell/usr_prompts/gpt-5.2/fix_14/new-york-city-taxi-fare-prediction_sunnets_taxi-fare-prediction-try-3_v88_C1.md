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

4.31009

# 6. Current score

None

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 15.36672) has done: 'Diagnosis: The crash happens during the `keras` import in cell 1 and is triggered by an incompatible `protobuf` runtime used indirectly by `google-api-python-client` (via `google.protobuf`), causing `MessageFactory.GetPrototype` to be missing. This is a known protobuf API change; the quickest deterministic fix is to force the pure-Python protobuf implementation, which retains the legacy API surface expected by some dependencies. This change must occur before any imports that transitively load protobuf, so it belongs at the very top of cell 1.

Patch summary: In cell 1, set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (and its version flag) before importing anything else. Keep all other imports and constants unchanged to preserve downstream behavior.

Updated cells

Compatibility notes for cell k+1: No variables, names, or interfaces are changed; cell 2 see the same `TRAIN_PATH`, `TEST_PATH`, and other constants and run as-is once cell 1 imports succeed.

Assumptions: The environment allows setting `os.environ` at runtime before imports, and using the pure-Python protobuf implementation is acceptable for this notebook’s functionality/performance.'
- What this solution (achieved 15.34931) has done: 'Diagnosis: The crash happens immediately when importing `keras` in cell 1. With Python 3.7 and the installed mix of `keras==3.8.0` plus `tf_keras==2.18.0`, Keras 3 pulls in protobuf-generated code that is incompatible with the runtime protobuf API (error: `MessageFactory` missing `GetPrototype`). This is an environment/version mismatch triggered at import time, not by your model code. The minimal deterministic fix is to use the already-installed `tf_keras` package (Keras 2.x API) instead of `keras`, which avoids the incompatible protobuf path while keeping the same Sequential/Dense/etc. core logic.

Patch summary: Modify only cell 1 imports so they come from `tf_keras` (and its `callbacks/layers/models/optimizers/regularizers`) rather than `keras`. Keep all variables, constants, and the rest of the notebook interface unchanged.

Updated cells: (cell 1 only)

Compatibility notes for cell k+1: Cell 2 uses `TRAIN_PATH`, `TEST_PATH`, and other constants from cell 1; these remain unchanged. The model-building symbols (`Sequential`, `Dense`, etc.) keep the same names and behavior under `tf_keras`, so later cells that reference them still work.

Assumptions: `tf_keras==2.18.0` is functional in this environment and does not trigger the protobuf `MessageFactory.GetPrototype` import error.'
- What this solution (achieved 15.28709) has done: 'Diagnosis: The crash happens during imports in cell 1, before any model/data logic runs. With Python 3.7 and `tf_keras==2.18.0`, importing TensorFlow/Keras can trigger a protobuf incompatibility where the runtime `google.protobuf.message_factory.MessageFactory` no longer exposes `GetPrototype`, raising `AttributeError`. The two environment variables currently set (`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION*`) do not reliably avoid this in all protobuf versions. For this environment, the minimal deterministic fix is to pin protobuf to the legacy 3.20.x line at runtime (via pip) before importing `tf_keras`, which restores the expected API.

Patch summary: In cell 1 only, add a small pre-import block that checks the installed `protobuf` version and, if it’s >=4, installs `protobuf==3.20.3`, then clears any already-imported `google.protobuf` modules and proceeds with the original imports unchanged. This keeps the model/training logic identical while unblocking execution.

Updated cells:'
- What this solution (achieved 15.23376) has done: 'Diagnosis: The crash occurs inside `remove_datapoints_from_water()` when calling `plt.imread()` with an HTTPS URL. Newer Matplotlib/Pillow stacks do not allow `imread()` to directly open URLs and raise a `ValueError` instructing to open the URL via `urllib` and `PIL.Image` first. Since `clean()` depends on this function, the minimal fix is to change only the mask-loading line to fetch the image bytes via `urllib.request.urlopen` and decode with `PIL.Image.open`, then convert to a NumPy array. This preserves the existing water-filtering logic and the returned dataframe interface.

Patch summary: In cell 7, replace `plt.imread(url)` with `np.array(PIL.Image.open(urllib.request.urlopen(url)))` and keep the rest of the logic unchanged.

Updated cells: Cell 7 only (function definition updated in-place).

Compatibility notes for cell k+1: The returned `train_df`/`test_df` remain pandas DataFrames with the same columns and filtering semantics, so cell 8 (`add_time_features`) continues to work unchanged.

Assumptions: Network access to `https://aiblog.nl/download/nyc_mask-74.5_-72.8_40.5_41.8.png` is available as intended by the original notebook; PIL is available via Matplotlib dependencies (and is imported explicitly to be safe).'
- What this solution (achieved 15.16323) has done: 'Diagnosis: The crash occurs inside `remove_datapoints_from_water()` when calling `plt.imread()` with a URL. Newer Matplotlib versions no longer support reading remote URLs directly and raise a `ValueError`, requiring the URL to be opened via `urllib` and then decoded via Pillow. This is triggered by `clean()` in cell 7.  
Patch summary: Update `remove_datapoints_from_water()` in cell 0 to fetch the PNG via `urllib.request.urlopen`, open it with `PIL.Image.open`, convert to a NumPy array, and then apply the same boolean mask logic as before. This preserves the exact cleaning semantics while making it compatible with the current Matplotlib/Pillow behavior.  
Updated cells: Only cell 0 is changed (bug source), keeping all existing function interfaces unchanged.  
Compatibility notes for cell k+1: Cell 8 expects `clean()` and `remove_datapoints_from_water()` to return filtered DataFrames; this remains identical, so downstream feature engineering remains compatible.  
Assumptions: Network access to `https://aiblog.nl/download/nyc_mask-74.5_-72.8_40.5_41.8.png` is available as originally intended by the notebook; the environment has Pillow installed (it is imported in cell 7).'
- What this solution (achieved 15.07179) has done: 'Diagnosis: Cell 32 crashes because `tf_keras.optimizers` (TF-Keras 2.18) does not expose a lowercase constructor `optimizers.adam`; the correct API is the class `optimizers.Adam`. Additionally, the keyword `lr` is deprecated in modern Keras/TF-Keras in favor of `learning_rate`, so even after switching to the class, keeping `lr` can fail depending on version.  
Patch summary: In cell 32 only, replace `optimizers.adam(lr=...)` with `optimizers.Adam(learning_rate=...)`, preserving the same optimizer type and learning rate, and leaving the rest of the model/training logic unchanged.  
Updated cells: Only cell 32 is changed.  
Compatibility notes for cell k+1: `model` and `history` are still created the same way; `my_model.h5` is still written by `ModelCheckpoint`, so `load_model` in cell 33 remains compatible.  
Assumptions: TF-Keras 2.18.0 follows the standard `tf.keras.optimizers.Adam`-style API and supports `learning_rate` as the constructor argument.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

try:
    import google.protobuf  # noqa: F401

    try:
        from google.protobuf import __version__ as _pb_ver
    except Exception:
        _pb_ver = "0"
    try:
        _major = int(str(_pb_ver).split(".", 1)[0])
    except Exception:
        _major = 0

    if _major >= 4:
        import sys
        import subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf==3.20.3"]
        )

        for _m in list(sys.modules.keys()):
            if _m.startswith("google.protobuf"):
                del sys.modules[_m]
except Exception:
    pass

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import urllib.request
from PIL import Image

from tf_keras.models import Sequential, load_model
from tf_keras.layers import Dense, Dropout, BatchNormalization, LSTM
from tf_keras.callbacks import EarlyStopping, ModelCheckpoint
from tf_keras import optimizers, regularizers
from tf_keras import backend as K

TRAIN_PATH = "../input/train.csv"
TEST_PATH = "../input/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 1000
EPOCHS = 100
LEARNING_RATE = 0.001
DATASET_SIZE = 800000



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
testKaggle = pd.read_csv(TEST_PATH, dtype=datatypes)



## === cell 2
train_df, test_df = train_test_split(trainKaggle, test_size=0.50, random_state=1)
test_df = test_df[:10000]



## === cell 3
print("testKaggle Size %d" % len(testKaggle))
print("train_df Size %d" % len(train_df))
print("test_df Size %d" % len(test_df))



## === cell 4
train_df.describe()



## === cell 5
test_df.describe()




## === cell 6
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


def remove_datapoints_from_water(df):
    def lonlat_to_xy(longitude, latitude, dx, dy, BB):
        return (dx * (longitude - BB[0]) / (BB[1] - BB[0])).astype("int"), (
            dy - dy * (latitude - BB[2]) / (BB[3] - BB[2])
        ).astype("int")

    BB = (-74.5, -72.8, 40.5, 41.8)

    url = "https://aiblog.nl/download/nyc_mask-74.5_-72.8_40.5_41.8.png"
    with urllib.request.urlopen(url) as _resp:
        _img = Image.open(_resp)
        nyc_mask = np.array(_img)[:, :, 0] > 0.9

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


def late_night(row):
    if (row["hour"] <= 3) or (row["hour"] >= 0):
        return 1
    else:
        return 0


def night(row):
    if ((row["hour"] > 20) and (row["hour"] > 0)) and (row["weekday"] < 5):
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
    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"], format="%Y-%m-%d %H:%M:%S %Z"
    )
    df["year"] = df["pickup_datetime"].apply(lambda x: x.year)
    df["month"] = df["pickup_datetime"].apply(lambda x: x.month)
    df["day"] = df["pickup_datetime"].apply(lambda x: x.day)
    df["hour"] = df["pickup_datetime"].apply(lambda x: x.hour)
    df["weekday"] = df["pickup_datetime"].apply(lambda x: x.weekday())
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


def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295  # Pi/180
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))  # 2*R*asin...


def distanceP(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295  # Pi/180
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))  # 2*R*asin...


def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    df = pd.DataFrame(prediction, columns=[prediction_column])
    df[id_column] = raw_test[id_column]
    df[[id_column, prediction_column]].to_csv((file_name), index=False)
    print("Output complete")


def plot_loss_accuracy_rmse(history):

    plt.figure(figsize=(20, 10))
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper right")
    plt.show()

    plt.figure(figsize=(20, 10))
    plt.plot(history.history["rmse"])
    plt.plot(history.history["val_rmse"])
    plt.title("Model rmse")
    plt.ylabel("rmse")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper right")
    plt.show()




## === cell 7
print("train_df add_time_features")
train_df = add_time_features(train_df)
print("test_df add_time_features")
test_df = add_time_features(test_df)
print("testKaggle add_time_features")
testKaggle = add_time_features(testKaggle)



## === cell 8
train_df.describe()



## === cell 9
print("train_df add_coordinate_features")
add_coordinate_features(train_df)
print("test_df add_coordinate_features Disabled!")
add_coordinate_features(test_df)
print("testKaggle add_coordinate_features")
add_coordinate_features(testKaggle)



## === cell 10
train_df.describe()



## === cell 11
print("train_df add_distances_features")
train_df = add_distances_features(train_df)
print("test_df add_distances_features")
test_df = add_distances_features(test_df)
print("testKaggle add_distances_features")
testKaggle = add_distances_features(testKaggle)

print("Done with Adding features")



## === cell 12
train_df.describe()



## === cell 13
plot = train_df.iloc[:2000].plot.scatter("fare_amount", "passenger_count")
plot = train_df.iloc[:2000].plot.scatter("fare_amount", "year")
plot = train_df.iloc[:2000].plot.scatter("fare_amount", "month")
plot = train_df.iloc[:2000].plot.scatter("fare_amount", "day")
plot = train_df.iloc[:2000].plot.scatter("fare_amount", "hour")
plot = train_df.iloc[:2000].plot.scatter("fare_amount", "weekday")



## === cell 14
dropped_columns = ["pickup_datetime"]  # keep core logic: only drop pickup_datetime

train_df = train_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)

testKaggle_clean = testKaggle.drop(dropped_columns + ["key"], axis=1)

print("Done with dropped_columns")



## === cell 15
train_df.shape



## === cell 16
train_df.describe()



## === cell 17
test_df.describe()



## === cell 18
train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=1)



## === cell 19
train_df_main = train_df
validation_df_main = validation_df



## === cell 20
validation_df.describe()



## === cell 21
train_labels = train_df["fare_amount"].values
validation_labels = validation_df["fare_amount"].values
test_labels = test_df["fare_amount"].values

train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)
test_df = test_df.drop(["fare_amount"], axis=1)

print("Done with Labels")



## === cell 22
test_labels



## === cell 23
train_df.describe()



## === cell 24
validation_df.describe()



## === cell 25
test_df.describe()



## === cell 26
for _df_name in ["train_df", "validation_df", "test_df"]:
    _df = locals()[_df_name]
    if "key" in _df.columns:
        locals()[_df_name] = _df.drop(["key"], axis=1)

scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_df)
testKaggle_scaled = scaler.transform(testKaggle_clean)


## === cell 27
test_scaled




## === cell 28
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))




## === cell 29
import tensorflow as tf


def rmse(y_true, y_pred):
    return tf.math.sqrt(tf.reduce_mean(tf.square(y_pred - y_true), axis=-1))


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
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1))

adam = optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(
    loss="mean_squared_error", optimizer=adam, metrics=["mae", "accuracy", rmse, "mse"]
)

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



## === cell 30
_loaded_model = load_model



## === cell 31
try:
    from IPython.display import SVG
    from tf_keras.utils import model_to_dot

    SVG(model_to_dot(model).create(prog="dot", format="svg"))
except Exception as e:
    print("Model visualization skipped:", repr(e))



## === cell 32
plot_loss_accuracy_rmse(history)



## === cell 33
score = model.evaluate(train_df_scaled, train_labels, verbose=1)
print(score)
print("train mean_squared_error:", score[0])
print("train mae:", score[1])
print("train accuracy:", score[2])
print("train rmse:", score[3])
print("train mse:", score[4])



## === cell 34
score = model.evaluate(validation_df_scaled, validation_labels, verbose=1)
print(score)
print("Validation mean_squared_error:", score[0])
print("Validation mae:", score[1])
print("Validation accuracy:", score[2])
print("Validation rmse:", score[3])
print("Validation mse:", score[4])



## === cell 35
score = model.evaluate(test_scaled, test_labels, verbose=1)
print(score)
print("Test mean_squared_error:", score[0])
print("Test mae:", score[1])
print("Test accuracy:", score[2])
print("Test rmse:", score[3])
print("Test mse:", score[4])



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



## === cell 41
plt.figure(figsize=(20, 10))
plt.plot(validation_labels[:1000])
plt.plot(validation_predictions[:1000])
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
error = np.asarray(error).ravel()
error = error[np.isfinite(error)]

if error.size == 0:
    print(
        "Skipping validation error histogram: residuals contain no finite values (all NaN/Inf)."
    )
else:
    plt.hist(error, bins=100)
    plt.xlabel("Prediction Error")
    _ = plt.ylabel("Count")



## === cell 44
error = test_predictions - test_labels
error = np.asarray(error).ravel()
error = error[np.isfinite(error)]

if error.size == 0:
    print(
        "Skipping test error histogram: residuals contain no finite values (all NaN/Inf)."
    )
else:
    plt.hist(error, bins=50)
    plt.xlabel("Prediction Error")
    _ = plt.ylabel("Count")



## === cell 45
print(len(error))

errorGreaterZero = error[(np.logical_or(error <= -1, error >= 1))]
print(len(errorGreaterZero))

plt.hist(errorGreaterZero, bins=100)
plt.xlabel("Prediction Error")
_ = plt.ylabel("Count")



## === cell 46
predictionKaggle = model.predict(testKaggle_scaled, batch_size=128, verbose=1)



## === cell 47
output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)
