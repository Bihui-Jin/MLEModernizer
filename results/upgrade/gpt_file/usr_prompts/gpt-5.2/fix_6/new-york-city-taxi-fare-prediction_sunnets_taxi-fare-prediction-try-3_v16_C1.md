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

4.82559

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 5.83243) has done: 'I fix the runtime blockers so the notebook runs end-to-end in the Kaggle environment: (1) remove the Keras/protobuf import crash by switching to `tf_keras` (installed) consistently, (2) replace the URL-based water-mask loading (no internet + `plt.imread` URL error) with a safe no-op fallback that preserves the cleaning pipeline, and (3) update deprecated optimizer API (`optimizers.adam`) to the correct `Adam` constructor. I also fix the broken time parsing and the incorrect night/late_night logic (it currently always returns 1) because that hurts RMSE substantially while keeping the same feature set and model architecture/training loop. Finally, I ensure the submission is written as a valid `.csv` with the required columns.'
- What this solution (achieved 507.54868) has done: 'I fix the immediate runtime crash caused by an incompatible protobuf version that `tf_keras` pulls in, by switching the Keras imports to `tensorflow.keras` (available in Kaggle’s TensorFlow runtime) while keeping the exact same model/loss/training loop. I also correct the training data read so it includes the required `fare_amount` column (it is currently excluded by `usecols`, which would break labels and/or silently train on wrong data depending on execution). Finally, to nudge RMSE toward the target with minimal semantic change, I compile with the already-defined RMSE metric instead of the irrelevant `"accuracy"` metric (no training objective change) and keep submission writing unchanged to ensure a valid `key,fare_amount` CSV is produced.'
- What this solution (achieved 6.06708) has done: 'I fix the TensorFlow/protobuf import crash by forcing the notebook to use the installed `tf_keras` backend (instead of `tensorflow.keras`), which avoids the `MessageFactory.GetPrototype` error in this environment. I also correct the input paths to point at the actual Kaggle-mounted CSVs under `/kaggle/input/`, ensuring the data loads reliably. To move RMSE strongly toward your target (and away from the current catastrophic score), I fix the major logic bug causing train/test feature mismatch: you currently drop `passenger_count` from training but keep it in the Kaggle test features, which breaks scaling/prediction semantics and yields unusable predictions. These changes keep the same feature set, model architecture, loss, and training loop, but make the pipeline consistent end-to-end and produce a valid `key,fare_amount` submission CSV.'
- What this solution (achieved 55.30183) has done: 'I fix the immediate runtime blocker (`MessageFactory.GetPrototype`) by avoiding importing TensorFlow/keras at module import time and forcing a safe protobuf implementation before any TF-related import, while keeping the same `tf_keras` model, layers, loss, and training loop. I also correct the Kaggle input paths to point at the actual mounted competition folder so data loading is reliable. To move RMSE toward your target with minimal semantic change, I clip both train and test predictions to be non-negative (taxi fares can’t be negative) so evaluation RMSE doesn’t get penalized by negative outputs during validation. Finally, I ensure the submission is written as a valid `.csv` with exactly `key,fare_amount`.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras import optimizers, regularizers
import tensorflow.keras.backend as K

TRAIN_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/new-york-city-taxi-fare-prediction/train.csv"
TEST_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/new-york-city-taxi-fare-prediction/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 0.001
DATASET_SIZE = 80000

np.random.seed(1)
tf.random.set_seed(1)



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

trainKaggle = pd.read_csv(
    TRAIN_PATH,
    nrows=DATASET_SIZE,
    dtype=datatypes,
    usecols=[
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



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2435707798.py in <cell line: 0>()
     10 }
     11 
---> 12 trainKaggle = pd.read_csv(
     13     TRAIN_PATH,
     14     nrows=DATASET_SIZE,

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
train_df = trainKaggle.copy()
test_df = trainKaggle.iloc[
    :0
].copy()  # kept only to preserve original cell structure/prints



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1774548634.py in <cell line: 0>()
      1 # Bugfix + score improvement: do not discard 50% of the already-subsampled training data.
      2 # Keep all loaded rows for training; only create a validation split.
----> 3 train_df = trainKaggle.copy()
      4 test_df = trainKaggle.iloc[
      5     :0

NameError: name 'trainKaggle' is not defined

## === cell 3
testKaggle.head()



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2294344613.py in <cell line: 0>()
----> 1 testKaggle.head()
      2 

NameError: name 'testKaggle' is not defined

## === cell 4
train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=1)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1154705564.py in <cell line: 0>()
----> 1 train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=1)
      2 

NameError: name 'train_df' is not defined

## === cell 5
print("testKaggle Size %d" % len(testKaggle))
print("train_df Size %d" % len(train_df))
print("validation_df Size %d" % len(validation_df))
print("test_df Size %d" % len(test_df))



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1009358473.py in <cell line: 0>()
----> 1 print("testKaggle Size %d" % len(testKaggle))
      2 print("train_df Size %d" % len(train_df))
      3 print("validation_df Size %d" % len(validation_df))
      4 print("test_df Size %d" % len(test_df))
      5 

NameError: name 'testKaggle' is not defined

## === cell 6
test_df.head()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3233580699.py in <cell line: 0>()
----> 1 test_df.head()
      2 

NameError: name 'test_df' is not defined

## === cell 7
validation_df.head()




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1063343436.py in <cell line: 0>()
----> 1 validation_df.head()
      2 
      3 

NameError: name 'validation_df' is not defined

## === cell 8
def remove_datapoints_from_water(df):
    """
    Original intent: remove pickups/dropoffs in water using an NYC land mask.
    Kaggle kernels typically have no internet; fall back to returning df unchanged.
    """
    try:
        local_mask_path = "nyc_mask-74.5_-72.8_40.5_41.8.png"
        if not os.path.exists(local_mask_path):
            return df

        nyc_mask = plt.imread(local_mask_path)[:, :, 0] > 0.9

        def lonlat_to_xy(longitude, latitude, dx, dy, BB):
            return (dx * (longitude - BB[0]) / (BB[1] - BB[0])).astype("int"), (
                dy - dy * (latitude - BB[2]) / (BB[3] - BB[2])
            ).astype("int")

        BB = (-74.5, -72.8, 40.5, 41.8)

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

    print(" New size after only NYC: %d" % len(df))

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
    df = remove_datapoints_from_water(df)
    print("New size: %d" % len(df))

    print(" New size: %d" % len(df))
    return df


def late_night(row):
    h = row["hour"]
    return 1 if (h <= 3) else 0


def night(row):
    h = row["hour"]
    wd = row["weekday"]
    return 1 if ((h >= 20 or h <= 6) and (wd < 5)) else 0


def rush_hour(row):
    h = row["hour"]
    wd = row["weekday"]
    return 1 if ((16 <= h <= 20) and (wd < 5)) else 0


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=False)
    df["year"] = dt.dt.year.astype("float32")
    df["month"] = dt.dt.month.astype("float32")
    df["day"] = dt.dt.day.astype("float32")
    df["hour"] = dt.dt.hour.astype("float32")
    df["weekday"] = dt.dt.weekday.astype("float32")
    df["pickup_datetime"] = df["pickup_datetime"].astype(str)

    df["night"] = df.apply(lambda x: night(x), axis=1).astype("uint8")
    df["late_night"] = df.apply(lambda x: late_night(x), axis=1).astype("uint8")
    df["rush_hour"] = df.apply(lambda x: rush_hour(x), axis=1).astype("uint8")
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


def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))


def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    df = pd.DataFrame(
        {
            id_column: raw_test[id_column].values,
            prediction_column: prediction.reshape(-1),
        }
    )
    df.to_csv(file_name, index=False)
    print("Output complete:", file_name, "shape=", df.shape)


def plot_loss_accuracy(history):
    plt.figure(figsize=(20, 10))
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper right")
    plt.show()




## === cell 9
validation_df.head()



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/726333114.py in <cell line: 0>()
----> 1 validation_df.head()
      2 

NameError: name 'validation_df' is not defined

## === cell 10
print("train_df clean")
train_df = clean(train_df)
print("validation_df clean")
validation_df = clean(validation_df)

print("test_df clean")
test_df = clean(test_df) if len(test_df) else test_df

train_df.describe()

print("train_df add_time_features")
train_df = add_time_features(train_df)
print("validation_df add_time_features")
validation_df = add_time_features(validation_df)

print("test_df add_time_features")
test_df = add_time_features(test_df) if len(test_df) else test_df

print("testKaggle add_time_features")
testKaggle = add_time_features(testKaggle)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/40261165.py in <cell line: 0>()
      1 print("train_df clean")
----> 2 train_df = clean(train_df)
      3 print("validation_df clean")
      4 validation_df = clean(validation_df)
      5 

NameError: name 'train_df' is not defined

## === cell 11
print("train_df add_coordinate_features")
train_df = add_coordinate_features(train_df)
print("validation_df add_coordinate_features")
validation_df = add_coordinate_features(validation_df)

print("test_df add_coordinate_features")
test_df = add_coordinate_features(test_df) if len(test_df) else test_df

print("testKaggle add_coordinate_features")
testKaggle = add_coordinate_features(testKaggle)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/769957512.py in <cell line: 0>()
      1 print("train_df add_coordinate_features")
----> 2 train_df = add_coordinate_features(train_df)
      3 print("validation_df add_coordinate_features")
      4 validation_df = add_coordinate_features(validation_df)
      5 

NameError: name 'train_df' is not defined

## === cell 12
print("train_df add_distances_features")
train_df = add_distances_features(train_df)
print("validation_df add_distances_features")
validation_df = add_distances_features(validation_df)

print("test_df add_distances_features")
test_df = add_distances_features(test_df) if len(test_df) else test_df

print("testKaggle add_distances_features")
testKaggle = add_distances_features(testKaggle)

print("Done with Adding features")



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2531607731.py in <cell line: 0>()
      1 print("train_df add_distances_features")
----> 2 train_df = add_distances_features(train_df)
      3 print("validation_df add_distances_features")
      4 validation_df = add_distances_features(validation_df)
      5 

NameError: name 'train_df' is not defined

## === cell 13
dropped_columns = ["pickup_datetime"]

train_df = train_df.drop(dropped_columns, axis=1)
validation_df = validation_df.drop(dropped_columns, axis=1)
testKaggle_clean = testKaggle.drop(dropped_columns + ["key"], axis=1)

test_df = test_df.drop(dropped_columns, axis=1) if len(test_df) else test_df

print("Done with dropped_columns")



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3016496761.py in <cell line: 0>()
      1 dropped_columns = ["pickup_datetime"]
      2 
----> 3 train_df = train_df.drop(dropped_columns, axis=1)
      4 validation_df = validation_df.drop(dropped_columns, axis=1)
      5 testKaggle_clean = testKaggle.drop(dropped_columns + ["key"], axis=1)

NameError: name 'train_df' is not defined

## === cell 14
train_labels = train_df["fare_amount"].values
validation_labels = validation_df["fare_amount"].values

train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)

test_labels = (
    test_df["fare_amount"].values
    if ("fare_amount" in test_df.columns and len(test_df))
    else None
)
test_df = (
    test_df.drop(["fare_amount"], axis=1)
    if ("fare_amount" in test_df.columns and len(test_df))
    else test_df
)

print("Done with Labels")



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2405676117.py in <cell line: 0>()
----> 1 train_labels = train_df["fare_amount"].values
      2 validation_labels = validation_df["fare_amount"].values
      3 
      4 train_df = train_df.drop(["fare_amount"], axis=1)
      5 validation_df = validation_df.drop(["fare_amount"], axis=1)

NameError: name 'train_df' is not defined

## === cell 15
train_df.shape



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/186828624.py in <cell line: 0>()
----> 1 train_df.shape
      2 

NameError: name 'train_df' is not defined

## === cell 16
test_df.shape



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2772256981.py in <cell line: 0>()
----> 1 test_df.shape
      2 

NameError: name 'test_df' is not defined

## === cell 17
validation_df.shape



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/306016390.py in <cell line: 0>()
----> 1 validation_df.shape
      2 

NameError: name 'validation_df' is not defined

## === cell 18
scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
testKaggle_scaled = scaler.transform(testKaggle_clean)


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))




## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3667625854.py in <cell line: 0>()
      1 scaler = preprocessing.MinMaxScaler()
----> 2 train_df_scaled = scaler.fit_transform(train_df)
      3 validation_df_scaled = scaler.transform(validation_df)
      4 testKaggle_scaled = scaler.transform(testKaggle_clean)
      5 

NameError: name 'train_df' is not defined

## === cell 19
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
model.add(Dense(16, activation="relu"))
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



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3736382739.py in <cell line: 0>()
      4         256,
      5         activation="relu",
----> 6         input_dim=train_df_scaled.shape[1],
      7         activity_regularizer=regularizers.l1(0.01),
      8     )

NameError: name 'train_df_scaled' is not defined

## === cell 20
print("Model trained; skipping model plot (graphviz/vis_utils not available).")



## === cell 21
plot_loss_accuracy(history)



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1049299230.py in <cell line: 0>()
----> 1 plot_loss_accuracy(history)
      2 

NameError: name 'history' is not defined

## === cell 22
predictionKaggle = model.predict(testKaggle_scaled, batch_size=128, verbose=1)



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/522070667.py in <cell line: 0>()
----> 1 predictionKaggle = model.predict(testKaggle_scaled, batch_size=128, verbose=1)
      2 

NameError: name 'testKaggle_scaled' is not defined

## === cell 23
predictionKaggle = np.clip(predictionKaggle, 0, None)



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4241482987.py in <cell line: 0>()
----> 1 predictionKaggle = np.clip(predictionKaggle, 0, None)
      2 

NameError: name 'predictionKaggle' is not defined

## === cell 24
output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3107110803.py in <cell line: 0>()
----> 1 output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)
      2 

NameError: name 'testKaggle' is not defined

## === cell 25
print(predictionKaggle[:10].reshape(-1))
print("Wrote:", SUBMISSION_NAME)

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1519799801.py in <cell line: 0>()
----> 1 print(predictionKaggle[:10].reshape(-1))
      2 print("Wrote:", SUBMISSION_NAME)

NameError: name 'predictionKaggle' is not defined
