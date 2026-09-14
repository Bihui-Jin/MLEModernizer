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
seaborn==0.12.2
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

4.12199

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 15.27775) has done: 'I fixed the feature‑leakage issue by dropping the target column before scaling, corrected the scaler mismatch, and replaced the failing TensorFlow import with a plain Keras import that works in the given environment. These minimal changes let the pipeline run end‑to‑end and generate a proper `submission.csv` while keeping the original neural‑network architecture unchanged.'
- What this solution (achieved 6.23608) has done: 'I fixed the import errors by switching to TensorFlow’s Keras API, replaced the custom RMSE metric with TensorFlow’s built‑in `RootMeanSquaredError`, and updated the model compilation to use this metric. These changes unblock training so the model actually learns from the data, which lower the RMSE toward the target score while keeping the original architecture untouched.'
- What this solution (achieved 6.27974) has done: 'I replace the failing TensorFlow import with the standalone Keras import that works in this environment and adjust the small custom RMSE helper to use Keras backend. This fixes the import error, lets the model compile and train, and keeps the original architecture unchanged, moving the RMSE toward the target score.'

# 9. Code solution

## === cell 0
import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)


print(os.listdir("./input"))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/689155718.py in <cell line: 0>()
      7 # tf.config.threading.set_inter_op_parallelism_threads(8)
      8 
----> 9 print(os.listdir("./input"))
     10 
     11 

FileNotFoundError: [Errno 2] No such file or directory: './input'

## === cell 1
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from math import radians, cos, sin, asin, sqrt
import warnings

warnings.filterwarnings("ignore")




## === cell 2
dtype_spec = {
    "key": "object",
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
use_cols = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

train = pd.read_csv(
    "./input/train.csv",
    nrows=10_000_000,
    usecols=use_cols,
    dtype=dtype_spec,
    parse_dates=["pickup_datetime"],
)

train["hour"] = train["pickup_datetime"].dt.hour.astype(np.float32)
train["weekday"] = train["pickup_datetime"].dt.weekday.astype(np.float32)
train["month"] = train["pickup_datetime"].dt.month.astype(np.float32)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/197460417.py in <cell line: 0>()
     21 ]
     22 
---> 23 train = pd.read_csv(
     24     "./input/train.csv",
     25     nrows=10_000_000,

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

FileNotFoundError: [Errno 2] No such file or directory: './input/train.csv'

## === cell 3
test = pd.read_csv(
    "./input/test.csv",
    usecols=[c for c in use_cols if c != "fare_amount"],
    dtype={k: v for k, v in dtype_spec.items() if k != "fare_amount"},
    parse_dates=["pickup_datetime"],
)

test["hour"] = test["pickup_datetime"].dt.hour.astype(np.float32)
test["weekday"] = test["pickup_datetime"].dt.weekday.astype(np.float32)
test["month"] = test["pickup_datetime"].dt.month.astype(np.float32)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1354068728.py in <cell line: 0>()
----> 1 test = pd.read_csv(
      2     "./input/test.csv",
      3     usecols=[c for c in use_cols if c != "fare_amount"],
      4     dtype={k: v for k, v in dtype_spec.items() if k != "fare_amount"},
      5     parse_dates=["pickup_datetime"],

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

FileNotFoundError: [Errno 2] No such file or directory: './input/test.csv'

## === cell 4
train.head()




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1975634829.py in <cell line: 0>()
----> 1 train.head()
      2 
      3 

NameError: name 'train' is not defined

## === cell 5
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great‑circle distance between two points on the earth (specified in decimal degrees)
    """
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Earth radius in kilometers
    return c * r




## === cell 6
def add_travel_distance_vector_features(df):
    df["distance"] = haversine(
        df["dropoff_longitude"],
        df["dropoff_latitude"],
        df["pickup_longitude"],
        df["pickup_latitude"],
    )
    df["log_distance"] = np.log1p(df["distance"])


add_travel_distance_vector_features(train)
add_travel_distance_vector_features(test)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/800059266.py in <cell line: 0>()
      9 
     10 
---> 11 add_travel_distance_vector_features(train)
     12 add_travel_distance_vector_features(test)
     13 

NameError: name 'train' is not defined

## === cell 7
train.dtypes




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/810068874.py in <cell line: 0>()
----> 1 train.dtypes
      2 
      3 

NameError: name 'train' is not defined

## === cell 8
train.drop(
    [
        "dropoff_longitude",
        "dropoff_latitude",
        "pickup_longitude",
        "pickup_latitude",
        "pickup_datetime",
    ],
    axis=1,
    inplace=True,
)
test.drop(
    [
        "dropoff_longitude",
        "dropoff_latitude",
        "pickup_longitude",
        "pickup_latitude",
        "pickup_datetime",
    ],
    axis=1,
    inplace=True,
)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2371125042.py in <cell line: 0>()
----> 1 train.drop(
      2     [
      3         "dropoff_longitude",
      4         "dropoff_latitude",
      5         "pickup_longitude",

NameError: name 'train' is not defined

## === cell 9
train_features = train.drop(["key", "fare_amount"], axis=1)
test_features = test.drop("key", axis=1)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/681094438.py in <cell line: 0>()
----> 1 train_features = train.drop(["key", "fare_amount"], axis=1)
      2 test_features = test.drop("key", axis=1)
      3 
      4 

NameError: name 'train' is not defined

## === cell 10
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(train_features.astype(np.float32))
test_scaled = scaler.transform(test_features.astype(np.float32))




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/861964632.py in <cell line: 0>()
      2 
      3 scaler = StandardScaler()
----> 4 X = scaler.fit_transform(train_features.astype(np.float32))
      5 test_scaled = scaler.transform(test_features.astype(np.float32))
      6 

NameError: name 'train_features' is not defined

## === cell 11
y = train["fare_amount"].astype(np.float32)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2044814287.py in <cell line: 0>()
----> 1 y = train["fare_amount"].astype(np.float32)
      2 X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
      3 
      4 

NameError: name 'train' is not defined

## === cell 12
from keras import layers, models, backend as K, metrics




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 13
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))




## === cell 14
def nn(n_feature, k=32):
    model_in = layers.Input(shape=(n_feature,))
    x = layers.Dense(k)(model_in)
    x = layers.LeakyReLU(alpha=0.15)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(k * 4)(x)
    x = layers.LeakyReLU(alpha=0.15)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(k * 16)(x)
    x = layers.LeakyReLU(alpha=0.15)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(k * 16)(x)
    x = layers.LeakyReLU(alpha=0.15)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(k * 4)(x)
    x = layers.LeakyReLU(alpha=0.15)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(k)(x)
    x = layers.LeakyReLU(alpha=0.15)(x)
    x = layers.Dropout(0.2)(x)

    output = layers.Dense(1, activation="linear")(x)

    model = models.Model(inputs=model_in, outputs=output)
    model.compile(
        loss="mse",
        optimizer="adam",
        metrics=[metrics.RootMeanSquaredError(name="rmse")],
    )
    return model




## === cell 15
model = nn(X.shape[1])




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1706600067.py in <cell line: 0>()
----> 1 model = nn(X.shape[1])
      2 
      3 

NameError: name 'X' is not defined

## === cell 16
history = model.fit(
    X_train,
    y_train,
    batch_size=8192,
    epochs=50,
    verbose=1,
    validation_data=(X_val, y_val),
)




## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1212479997.py in <cell line: 0>()
----> 1 history = model.fit(
      2     X_train,
      3     y_train,
      4     batch_size=8192,
      5     epochs=50,

NameError: name 'model' is not defined

## === cell 17
plt.plot(history.history["rmse"], label="train")
plt.plot(history.history["val_rmse"], label="val")
plt.title("Model RMSE")
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.legend()
plt.show()




## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3260495312.py in <cell line: 0>()
----> 1 plt.plot(history.history["rmse"], label="train")
      2 plt.plot(history.history["val_rmse"], label="val")
      3 plt.title("Model RMSE")
      4 plt.ylabel("RMSE")
      5 plt.xlabel("Epoch")

NameError: name 'history' is not defined

## === cell 18
pres = model.predict(test_scaled)




## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3545082749.py in <cell line: 0>()
----> 1 pres = model.predict(test_scaled)
      2 
      3 

NameError: name 'model' is not defined

## === cell 19
test_original = pd.read_csv("./input/test.csv")




## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/3429560329.py in <cell line: 0>()
----> 1 test_original = pd.read_csv("./input/test.csv")
      2 
      3 

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

FileNotFoundError: [Errno 2] No such file or directory: './input/test.csv'

## === cell 20
preds = pres.reshape(-1)
preds = np.where(preds < 0, 0, preds)

submission = pd.DataFrame(
    {"key": test_original["key"], "fare_amount": preds},
    columns=["key", "fare_amount"],
)




## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/533586015.py in <cell line: 0>()
----> 1 preds = pres.reshape(-1)
      2 preds = np.where(preds < 0, 0, preds)
      3 
      4 submission = pd.DataFrame(
      5     {"key": test_original["key"], "fare_amount": preds},

NameError: name 'pres' is not defined

## === cell 21
submission.to_csv("submission.csv", index=False)




## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/376638180.py in <cell line: 0>()
----> 1 submission.to_csv("submission.csv", index=False)
      2 
      3 

NameError: name 'submission' is not defined

## === cell 22
print(os.listdir("."))
