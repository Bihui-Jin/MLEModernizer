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

4.30684

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 15.25184) has done: 'I fix the early crash caused by an incompatible TensorFlow/protobuf stack by switching the imports to the installed `tf_keras` package while keeping the exact same Keras model architecture, loss, and training loop. I also correct the broken boolean logic in the time-based feature functions (they were always returning 1 due to `hour >= 0`) to improve the quality of engineered features and move RMSE toward the target. Finally, I prevent the plotting/histogram cells from crashing when predictions contain NaNs/Infs by filtering to finite values (this is score-neutral but makes the notebook run end-to-end) and ensure a valid `.csv` submission is always written.'
- What this solution (achieved 15.25184) has done: 'I fix the early crash caused by the protobuf/TensorFlow import incompatibility by removing the `tensorflow` import entirely and using `tf_keras.utils.set_random_seed` for determinism instead. To move RMSE down toward the target with minimal semantic change, I align the distance feature with the intended (and already-implemented) Haversine distance by adding it as an additional feature while keeping the same model, loss, optimizer, and training loop. I also correct the input file paths to the actual Kaggle directory (`/kaggle/input/new-york-city-taxi-fare-prediction/...`) so the notebook runs reliably in the provided environment. Finally, I keep the submission writing logic but ensure predictions are flattened to 1D and saved to a `.csv` with the required columns.'
- What this solution (achieved 15.25184) has done: 'I fix the early crash (`MessageFactory.GetPrototype`) by preventing the protobuf C++ implementation from being used, which is the common root cause in Kaggle images when importing Keras/TensorFlow stacks. I keep your exact model, loss, optimizer, and training loop intact, but replace the slow/bug-prone `df.apply(..., axis=1)` time feature engineering with equivalent vectorized column logic (same semantics) so the features are actually computed correctly and efficiently, which should materially improve RMSE toward the target. I also ensure the checkpoint path uses a Keras-3-compatible extension and add a small safety clamp for predictions (finite + non-negative) without changing training behavior. The script still write a valid `submissiontry_water.csv` with the required `key,fare_amount` columns.'
- What this solution (achieved 15.25184) has done: 'I fix the early crash by enforcing the pure-Python protobuf implementation before any protobuf/keras-related imports, and by importing `tf_keras` in a safer way that avoids triggering the incompatible C++ protobuf path. I also correct a major logic bug where the test split accidentally includes the `fare_amount` column (it was taken from the training set), which can badly distort evaluation and training flow; this is a correctness fix and should move RMSE toward the target. Additionally, I ensure `accuracy` is not used as a metric for regression (score-neutral but avoids misleading/unstable metric behavior) while keeping the same model, loss, optimizer, and training loop. Finally, I keep the submission writing identical but guarantee the CSV is produced with correct columns and finite, non-negative predictions.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import tf_keras as keras
from tf_keras.models import Sequential
from tf_keras.layers import Dense
from tf_keras.callbacks import ModelCheckpoint
from tf_keras import optimizers, regularizers, backend

TRAIN_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/new-york-city-taxi-fare-prediction/train.csv"
TEST_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/new-york-city-taxi-fare-prediction/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 1000
EPOCHS = 100
LEARNING_RATE = 0.001
DATASET_SIZE = 80000

np.random.seed(1)
keras.utils.set_random_seed(1)


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
        print(" Skipping fare_amount outlier removal (no fare_amount column).")

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
        1 if ((row["hour"] > 20) or (row["hour"] < 6)) and (row["weekday"] < 5) else 0
    )


def rush_hour(row):
    return 1 if (16 <= row["hour"] <= 20) and (row["weekday"] < 5) else 0


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df):
    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"], errors="coerce", utc=False
    )
    df["year"] = df["pickup_datetime"].dt.year
    df["month"] = df["pickup_datetime"].dt.month
    df["day"] = df["pickup_datetime"].dt.day
    df["hour"] = df["pickup_datetime"].dt.hour
    df["minute"] = df["pickup_datetime"].dt.minute
    df["second"] = df["pickup_datetime"].dt.second
    df["weekday"] = df["pickup_datetime"].dt.weekday

    hour = df["hour"]
    weekday = df["weekday"]

    df["late_night"] = (hour <= 3).astype(np.uint8)
    df["night"] = (((hour > 20) | (hour < 6)) & (weekday < 5)).astype(np.uint8)
    df["rush_hour"] = (((hour >= 16) & (hour <= 20)) & (weekday < 5)).astype(np.uint8)
    return df


def add_coordinate_features(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]
    return df


def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295  # Pi/180
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))  # miles


def distanceP(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295  # Pi/180
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * np.cos((lon2 - lon1) * p) / 2
    )
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))  # miles


def add_distances_features(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]
    df["manhattan"] = manhattan(lat1, lon1, lat2, lon2)
    df["haversine"] = distance(lat1, lon1, lat2, lon2)
    return df


def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    pred = np.asarray(prediction).reshape(-1)
    df = pd.DataFrame({id_column: raw_test[id_column].values, prediction_column: pred})
    if not file_name.lower().endswith(".csv"):
        file_name = file_name + ".csv"
    df[[id_column, prediction_column]].to_csv(file_name, index=False)
    print("Output complete:", file_name)


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




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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

trainKaggle = pd.read_csv(TRAIN_PATH, nrows=DATASET_SIZE, dtype=datatypes)
testKaggle = pd.read_csv(
    TEST_PATH, dtype={k: v for k, v in datatypes.items() if k != "fare_amount"}
)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/700599355.py in <cell line: 0>()
     10 }
     11 
---> 12 trainKaggle = pd.read_csv(TRAIN_PATH, nrows=DATASET_SIZE, dtype=datatypes)
     13 testKaggle = pd.read_csv(
     14     TEST_PATH, dtype={k: v for k, v in datatypes.items() if k != "fare_amount"}

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/new-york-city-taxi-fare-prediction/new-york-city-taxi-fare-prediction/train.csv'

## === cell 2
train_df, test_df = train_test_split(trainKaggle, test_size=0.50, random_state=1)
test_df = test_df[:10000]



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/19325734.py in <cell line: 0>()
----> 1 train_df, test_df = train_test_split(trainKaggle, test_size=0.50, random_state=1)
      2 test_df = test_df[:10000]
      3 

NameError: name 'trainKaggle' is not defined

## === cell 3
print("testKaggle Size %d" % len(testKaggle))
print("train_df Size %d" % len(train_df))
print("test_df Size %d" % len(test_df))



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/395627971.py in <cell line: 0>()
----> 1 print("testKaggle Size %d" % len(testKaggle))
      2 print("train_df Size %d" % len(train_df))
      3 print("test_df Size %d" % len(test_df))
      4 

NameError: name 'testKaggle' is not defined

## === cell 4
train_df.describe()



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1577644986.py in <cell line: 0>()
----> 1 train_df.describe()
      2 

NameError: name 'train_df' is not defined

## === cell 5
test_df.describe()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4294142452.py in <cell line: 0>()
----> 1 test_df.describe()
      2 

NameError: name 'test_df' is not defined

## === cell 6
print("train_df clean")
train_df = clean(train_df)
test_df = clean(test_df)

print("testKaggle clean")
testKaggle = clean(testKaggle)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1533432998.py in <cell line: 0>()
      1 print("train_df clean")
----> 2 train_df = clean(train_df)
      3 test_df = clean(test_df)
      4 
      5 print("testKaggle clean")

NameError: name 'train_df' is not defined

## === cell 7
print("train_df add_time_features")
train_df = add_time_features(train_df)
print("test_df add_time_features")
test_df = add_time_features(test_df)
print("testKaggle add_time_features")
testKaggle = add_time_features(testKaggle)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1145226127.py in <cell line: 0>()
      1 print("train_df add_time_features")
----> 2 train_df = add_time_features(train_df)
      3 print("test_df add_time_features")
      4 test_df = add_time_features(test_df)
      5 print("testKaggle add_time_features")

NameError: name 'train_df' is not defined

## === cell 8
train_df.describe()



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1577644986.py in <cell line: 0>()
----> 1 train_df.describe()
      2 

NameError: name 'train_df' is not defined

## === cell 9
print("train_df add_coordinate_features")
add_coordinate_features(train_df)
print("test_df add_coordinate_features Disabled!")
add_coordinate_features(test_df)
print("testKaggle add_coordinate_features")
add_coordinate_features(testKaggle)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2476012946.py in <cell line: 0>()
      1 print("train_df add_coordinate_features")
----> 2 add_coordinate_features(train_df)
      3 print("test_df add_coordinate_features Disabled!")
      4 add_coordinate_features(test_df)
      5 print("testKaggle add_coordinate_features")

NameError: name 'train_df' is not defined

## === cell 10
print("train_df add_distances_features")
train_df = add_distances_features(train_df)
print("test_df add_distances_features")
test_df = add_distances_features(test_df)
print("testKaggle add_distances_features")
testKaggle = add_distances_features(testKaggle)

print("Done with Adding features")



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/470363350.py in <cell line: 0>()
      1 print("train_df add_distances_features")
----> 2 train_df = add_distances_features(train_df)
      3 print("test_df add_distances_features")
      4 test_df = add_distances_features(test_df)
      5 print("testKaggle add_distances_features")

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
plot = train_df.iloc[:2000].plot.scatter("fare_amount", "passenger_count")
plot = train_df.iloc[:2000].plot.scatter("fare_amount", "year")
plot = train_df.iloc[:2000].plot.scatter("fare_amount", "month")
plot = train_df.iloc[:2000].plot.scatter("fare_amount", "day")
plot = train_df.iloc[:2000].plot.scatter("fare_amount", "hour")



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4253417587.py in <cell line: 0>()
----> 1 plot = train_df.iloc[:2000].plot.scatter("fare_amount", "passenger_count")
      2 plot = train_df.iloc[:2000].plot.scatter("fare_amount", "year")
      3 plot = train_df.iloc[:2000].plot.scatter("fare_amount", "month")
      4 plot = train_df.iloc[:2000].plot.scatter("fare_amount", "day")
      5 plot = train_df.iloc[:2000].plot.scatter("fare_amount", "hour")

NameError: name 'train_df' is not defined

## === cell 13
dropped_columns = ["pickup_datetime"]

train_df = train_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)

testKaggle_clean = testKaggle.drop(dropped_columns + ["key"], axis=1)

print("Done with dropped_columns")



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4151285212.py in <cell line: 0>()
      1 dropped_columns = ["pickup_datetime"]
      2 
----> 3 train_df = train_df.drop(dropped_columns, axis=1)
      4 test_df = test_df.drop(dropped_columns, axis=1)
      5 

NameError: name 'train_df' is not defined

## === cell 14
train_df.shape



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/186828624.py in <cell line: 0>()
----> 1 train_df.shape
      2 

NameError: name 'train_df' is not defined

## === cell 15
train_df.describe()



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1577644986.py in <cell line: 0>()
----> 1 train_df.describe()
      2 

NameError: name 'train_df' is not defined

## === cell 16
test_df.describe()



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4294142452.py in <cell line: 0>()
----> 1 test_df.describe()
      2 

NameError: name 'test_df' is not defined

## === cell 17
train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=1)



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1154705564.py in <cell line: 0>()
----> 1 train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=1)
      2 

NameError: name 'train_df' is not defined

## === cell 18
train_df_main = train_df
validation_df_main = validation_df



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2901810251.py in <cell line: 0>()
----> 1 train_df_main = train_df
      2 validation_df_main = validation_df
      3 

NameError: name 'train_df' is not defined

## === cell 19
validation_df.describe()



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3277078349.py in <cell line: 0>()
----> 1 validation_df.describe()
      2 

NameError: name 'validation_df' is not defined

## === cell 20
train_labels = train_df["fare_amount"].values.astype("float32")
validation_labels = validation_df["fare_amount"].values.astype("float32")
test_labels = test_df["fare_amount"].values.astype("float32")

train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)
test_df = test_df.drop(["fare_amount"], axis=1)

print("Done with Labels")



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2024178754.py in <cell line: 0>()
----> 1 train_labels = train_df["fare_amount"].values.astype("float32")
      2 validation_labels = validation_df["fare_amount"].values.astype("float32")
      3 test_labels = test_df["fare_amount"].values.astype("float32")
      4 
      5 train_df = train_df.drop(["fare_amount"], axis=1)

NameError: name 'train_df' is not defined

## === cell 21
test_labels



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/805156543.py in <cell line: 0>()
----> 1 test_labels
      2 

NameError: name 'test_labels' is not defined

## === cell 22
train_df.describe()



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1577644986.py in <cell line: 0>()
----> 1 train_df.describe()
      2 

NameError: name 'train_df' is not defined

## === cell 23
validation_df.describe()



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3277078349.py in <cell line: 0>()
----> 1 validation_df.describe()
      2 

NameError: name 'validation_df' is not defined

## === cell 24
test_df.describe()



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4294142452.py in <cell line: 0>()
----> 1 test_df.describe()
      2 

NameError: name 'test_df' is not defined

## === cell 25
feature_cols = list(train_df.columns)
validation_df = validation_df.reindex(columns=feature_cols)
test_df = test_df.reindex(columns=feature_cols)
testKaggle_clean = testKaggle_clean.reindex(columns=feature_cols)

for c in feature_cols:
    train_df[c] = pd.to_numeric(train_df[c], errors="coerce")
    validation_df[c] = pd.to_numeric(validation_df[c], errors="coerce")
    test_df[c] = pd.to_numeric(test_df[c], errors="coerce")
    testKaggle_clean[c] = pd.to_numeric(testKaggle_clean[c], errors="coerce")

train_medians = train_df[feature_cols].median(numeric_only=True)
train_df[feature_cols] = train_df[feature_cols].fillna(train_medians)
validation_df[feature_cols] = validation_df[feature_cols].fillna(train_medians)
test_df[feature_cols] = test_df[feature_cols].fillna(train_medians)
testKaggle_clean[feature_cols] = testKaggle_clean[feature_cols].fillna(train_medians)

train_df_scaled = train_df.copy()
validation_df_scaled = validation_df.copy()
test_df_scaled = test_df.copy()
testKaggle_scaled = testKaggle_clean.copy()

scaler = preprocessing.MinMaxScaler()
train_df_scaled[feature_cols] = scaler.fit_transform(train_df[feature_cols])
validation_df_scaled[feature_cols] = scaler.transform(validation_df[feature_cols])
test_df_scaled[feature_cols] = scaler.transform(test_df[feature_cols])
testKaggle_scaled[feature_cols] = scaler.transform(testKaggle_clean[feature_cols])

X_train = train_df_scaled[feature_cols].to_numpy(dtype=np.float32)
X_val = validation_df_scaled[feature_cols].to_numpy(dtype=np.float32)
X_test = test_df_scaled[feature_cols].to_numpy(dtype=np.float32)
X_kaggle = testKaggle_scaled[feature_cols].to_numpy(dtype=np.float32)

print("Feature columns:", feature_cols)
print("Shapes:", X_train.shape, X_val.shape, X_test.shape, X_kaggle.shape)




## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/247174635.py in <cell line: 0>()
----> 1 feature_cols = list(train_df.columns)
      2 validation_df = validation_df.reindex(columns=feature_cols)
      3 test_df = test_df.reindex(columns=feature_cols)
      4 testKaggle_clean = testKaggle_clean.reindex(columns=feature_cols)
      5 

NameError: name 'train_df' is not defined

## === cell 26
def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))




## === cell 27
checkpoint = ModelCheckpoint(filepath="my_model.keras", verbose=1, save_best_only=True)

model = Sequential()
model.add(
    Dense(
        256,
        activation="linear",
        input_dim=X_train.shape[1],
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
print("Input dimension: %s" % X_train.shape[1])
print("Features used: %s" % feature_cols)
model.summary()

history = model.fit(
    x=X_train,
    y=train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1,
    callbacks=[checkpoint],
    validation_data=(X_val, validation_labels),
    shuffle=True,
)



## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/746203514.py in <cell line: 0>()
      6         256,
      7         activation="linear",
----> 8         input_dim=X_train.shape[1],
      9         activity_regularizer=regularizers.l1(0.01),
     10     )

NameError: name 'X_train' is not defined

## === cell 28
plot_loss_accuracy_rmse(history)



## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2579130885.py in <cell line: 0>()
----> 1 plot_loss_accuracy_rmse(history)
      2 

NameError: name 'history' is not defined

## === cell 29
score = model.evaluate(X_train, train_labels, verbose=1)
print(score)
print("train mean_squared_error:", score[0])
print("train mae:", score[1])
print("train rmse:", score[2])
print("train mse:", score[3])



## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/226472602.py in <cell line: 0>()
----> 1 score = model.evaluate(X_train, train_labels, verbose=1)
      2 print(score)
      3 print("train mean_squared_error:", score[0])
      4 print("train mae:", score[1])
      5 print("train rmse:", score[2])

NameError: name 'X_train' is not defined

## === cell 30
score = model.evaluate(X_val, validation_labels, verbose=1)
print(score)
print("Validation mean_squared_error:", score[0])
print("Validation mae:", score[1])
print("Validation rmse:", score[2])
print("Validation mse:", score[3])



## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1362660799.py in <cell line: 0>()
----> 1 score = model.evaluate(X_val, validation_labels, verbose=1)
      2 print(score)
      3 print("Validation mean_squared_error:", score[0])
      4 print("Validation mae:", score[1])
      5 print("Validation rmse:", score[2])

NameError: name 'X_val' is not defined

## === cell 31
score = model.evaluate(X_test, test_labels, verbose=1)
print(score)
print("Test mean_squared_error:", score[0])
print("Test mae:", score[1])
print("Test rmse:", score[2])
print("Test mse:", score[3])



## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/325469344.py in <cell line: 0>()
----> 1 score = model.evaluate(X_test, test_labels, verbose=1)
      2 print(score)
      3 print("Test mean_squared_error:", score[0])
      4 print("Test mae:", score[1])
      5 print("Test rmse:", score[2])

NameError: name 'X_test' is not defined

## === cell 32
validation_predictions = model.predict(X_val, verbose=0).flatten()

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



## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3150211706.py in <cell line: 0>()
----> 1 validation_predictions = model.predict(X_val, verbose=0).flatten()
      2 
      3 plt.scatter(validation_labels, validation_predictions)
      4 plt.xlabel("True Values")
      5 plt.ylabel("Predictions")

NameError: name 'X_val' is not defined

## === cell 33
test_predictions = model.predict(X_test, verbose=0).flatten()

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



## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1157946699.py in <cell line: 0>()
----> 1 test_predictions = model.predict(X_test, verbose=0).flatten()
      2 
      3 plt.scatter(test_labels, test_predictions)
      4 plt.xlabel("True Values")
      5 plt.ylabel("Predictions")

NameError: name 'X_test' is not defined

## === cell 34
print(np.argmax(test_predictions))
print(test_predictions[np.argmax(test_predictions)])
print(test_labels[np.argmax(test_predictions)])
test_df.iloc[np.argmax(test_predictions)]



## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3782083718.py in <cell line: 0>()
----> 1 print(np.argmax(test_predictions))
      2 print(test_predictions[np.argmax(test_predictions)])
      3 print(test_labels[np.argmax(test_predictions)])
      4 test_df.iloc[np.argmax(test_predictions)]
      5 

NameError: name 'test_predictions' is not defined

## === cell 35
print(np.argmin(test_predictions))
print(test_predictions[np.argmin(test_predictions)])
print(test_labels[np.argmin(test_predictions)])
test_df.iloc[np.argmin(test_predictions)]



## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1775569351.py in <cell line: 0>()
----> 1 print(np.argmin(test_predictions))
      2 print(test_predictions[np.argmin(test_predictions)])
      3 print(test_labels[np.argmin(test_predictions)])
      4 test_df.iloc[np.argmin(test_predictions)]
      5 

NameError: name 'test_predictions' is not defined

## === cell 36
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



## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3918075604.py in <cell line: 0>()
      1 fig, ax = plt.subplots()
----> 2 ax.scatter(test_labels, test_predictions)
      3 ax.plot(
      4     [test_labels.min(), test_labels.max()],
      5     [test_labels.min(), test_labels.max()],

NameError: name 'test_labels' is not defined

## === cell 37
plt.figure(figsize=(20, 10))
plt.plot(validation_labels[:100])
plt.plot(validation_predictions[:100])
plt.title("Prediction vs Actual")
plt.ylabel("Fare Amount")
plt.xlabel("Transaction")
plt.legend(["Actual", "prediction"], loc="upper right")
plt.show()



## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3331021377.py in <cell line: 0>()
      1 plt.figure(figsize=(20, 10))
----> 2 plt.plot(validation_labels[:100])
      3 plt.plot(validation_predictions[:100])
      4 plt.title("Prediction vs Actual")
      5 plt.ylabel("Fare Amount")

NameError: name 'validation_labels' is not defined

## === cell 38
plt.figure(figsize=(20, 10))
plt.plot(test_labels[:100])
plt.plot(test_predictions[:100])
plt.title("Prediction vs Actual")
plt.ylabel("Fare Amount")
plt.xlabel("Transaction")
plt.legend(["Actual", "prediction"], loc="upper right")
plt.show()



## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1260486629.py in <cell line: 0>()
      1 plt.figure(figsize=(20, 10))
----> 2 plt.plot(test_labels[:100])
      3 plt.plot(test_predictions[:100])
      4 plt.title("Prediction vs Actual")
      5 plt.ylabel("Fare Amount")

NameError: name 'test_labels' is not defined

## === cell 39
error = (validation_predictions - validation_labels).astype(np.float64)
error = error[np.isfinite(error)]
if len(error) > 0:
    plt.hist(error, bins=100)
    plt.xlabel("Prediction Error")
    _ = plt.ylabel("Count")
else:
    print("No finite validation errors available to plot.")



## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1376241056.py in <cell line: 0>()
----> 1 error = (validation_predictions - validation_labels).astype(np.float64)
      2 error = error[np.isfinite(error)]
      3 if len(error) > 0:
      4     plt.hist(error, bins=100)
      5     plt.xlabel("Prediction Error")

NameError: name 'validation_predictions' is not defined

## === cell 40
error = (test_predictions - test_labels).astype(np.float64)
error = error[np.isfinite(error)]
if len(error) > 0:
    plt.hist(error, bins=50)
    plt.xlabel("Prediction Error")
    _ = plt.ylabel("Count")
else:
    print("No finite test errors available to plot.")



## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3632372387.py in <cell line: 0>()
----> 1 error = (test_predictions - test_labels).astype(np.float64)
      2 error = error[np.isfinite(error)]
      3 if len(error) > 0:
      4     plt.hist(error, bins=50)
      5     plt.xlabel("Prediction Error")

NameError: name 'test_predictions' is not defined

## === cell 41
print(len(error))

errorGreaterZero = error[(np.logical_or(error <= -1, error >= 1))]
print(len(errorGreaterZero))

if len(errorGreaterZero) > 0:
    plt.hist(errorGreaterZero, bins=100)
    plt.xlabel("Prediction Error")
    _ = plt.ylabel("Count")
else:
    print("No finite test errors with |err|>=1 available to plot.")



## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2777915193.py in <cell line: 0>()
----> 1 print(len(error))
      2 
      3 errorGreaterZero = error[(np.logical_or(error <= -1, error >= 1))]
      4 print(len(errorGreaterZero))
      5 

NameError: name 'error' is not defined

## === cell 42
testKaggle_scaled.head()



## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3918719489.py in <cell line: 0>()
----> 1 testKaggle_scaled.head()
      2 

NameError: name 'testKaggle_scaled' is not defined

## === cell 43
predictionKaggle = model.predict(X_kaggle, batch_size=128, verbose=1)

predictionKaggle = np.asarray(predictionKaggle, dtype=np.float64).reshape(-1)
predictionKaggle[~np.isfinite(predictionKaggle)] = 0.0
predictionKaggle = np.maximum(predictionKaggle, 0.0)



## --- ERROR in cell 43, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1895626346.py in <cell line: 0>()
----> 1 predictionKaggle = model.predict(X_kaggle, batch_size=128, verbose=1)
      2 
      3 predictionKaggle = np.asarray(predictionKaggle, dtype=np.float64).reshape(-1)
      4 predictionKaggle[~np.isfinite(predictionKaggle)] = 0.0
      5 predictionKaggle = np.maximum(predictionKaggle, 0.0)

NameError: name 'X_kaggle' is not defined

## === cell 44
output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)

out_name = (
    SUBMISSION_NAME
    if SUBMISSION_NAME.lower().endswith(".csv")
    else SUBMISSION_NAME + ".csv"
)
print("Wrote submission to:", os.path.abspath(out_name))
print(pd.read_csv(out_name).head())

## --- ERROR in cell 44, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2125937524.py in <cell line: 0>()
----> 1 output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)
      2 
      3 out_name = (
      4     SUBMISSION_NAME
      5     if SUBMISSION_NAME.lower().endswith(".csv")

NameError: name 'testKaggle' is not defined
