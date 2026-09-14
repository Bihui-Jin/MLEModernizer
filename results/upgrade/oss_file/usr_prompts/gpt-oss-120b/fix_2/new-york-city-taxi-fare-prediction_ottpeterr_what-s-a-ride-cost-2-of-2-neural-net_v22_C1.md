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
protobuf==6.33.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
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

26.87094

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import gc




## === cell 1
TRAIN_PATH = "../input/train.csv"
TEST_PATH = "../input/test.csv"




## === cell 2
p = subprocess.Popen(
    ["wc", "-l", TRAIN_PATH], stdout=subprocess.PIPE, stderr=subprocess.PIPE
)
result, err = p.communicate()
if p.returncode != 0:
    raise IOError(err)
n_rows = int(result.strip().split()[0]) + 1




## === cell 3
def compute_haversine_distance(
    df,
    lat1="pickup_latitude",
    long1="pickup_longitude",
    lat2="dropoff_latitude",
    long2="dropoff_longitude",
):
    R = 3959  # radius of earth in miles
    phi1 = np.radians(df[lat1])
    phi2 = np.radians(df[lat2])

    delta_phi = np.radians(df[lat2] - df[lat1])
    delta_lambda = np.radians(df[long2] - df[long1])

    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = R * c
    df["distance"] = d.astype("float32")




## === cell 4
MIN_FARE = 2.50
MAX_FARE = 500

MIN_PASSENGER = 1
MAX_PASSENGER = 6


def clean_data(df, test=False):
    compute_haversine_distance(df)
    add_date_features(df, test)

    if not test:
        df.drop(df[df.isnull().any(axis=1)].index, axis=0, inplace=True)

        df.drop(
            ((df[df.fare_amount > MAX_FARE]) | (df[df.fare_amount < MIN_FARE])).index,
            axis=0,
            inplace=True,
        )
        df.drop(df[df.passenger_count > MAX_PASSENGER].index, axis=0, inplace=True)
        df.drop(df[df.passenger_count < MIN_PASSENGER].index, axis=0, inplace=True)
        df.drop(
            ((df[df.pickup_latitude > 90]) | (df[df.pickup_latitude < -90])).index,
            axis=0,
            inplace=True,
        )
        df.drop(
            ((df[df.pickup_longitude > 180]) | (df[df.pickup_longitude < -180])).index,
            axis=0,
            inplace=True,
        )
        df.drop(
            ((df[df.dropoff_latitude > 90]) | (df[df.dropoff_latitude < -90])).index,
            axis=0,
            inplace=True,
        )
        df.drop(
            (
                (df[df.dropoff_longitude > 180]) | (df[df.dropoff_longitude < -180])
            ).index,
            axis=0,
            inplace=True,
        )

        df.drop(df[df.distance > 100].index, axis=0, inplace=True)
        df.drop(df[df.distance <= 0].index, axis=0, inplace=True)

    if not test:
        df.drop(columns=["pickup_datetime"], inplace=True)


def add_date_features(df, test=False):
    df["pickup_datetime_clone"] = df["pickup_datetime"].values
    df.pickup_datetime_clone = df.pickup_datetime_clone.str.slice(0, 16)
    df.pickup_datetime_clone = pd.to_datetime(
        df.pickup_datetime_clone, utc=True, format="%Y-%m-%d %H:%M"
    )
    df["year"] = df.pickup_datetime_clone.dt.year.astype("uint8")
    df["month"] = df.pickup_datetime_clone.dt.month.astype("uint8")
    df["day"] = df.pickup_datetime_clone.dt.day.astype("uint8")
    df["dayofweek"] = df.pickup_datetime_clone.dt.dayofweek.astype("uint8")
    df["hour"] = df.pickup_datetime_clone.dt.hour.astype("uint8")
    df["minute"] = df.pickup_datetime_clone.dt.minute.astype("uint8")
    df.drop(columns=["pickup_datetime_clone"], inplace=True)




## === cell 5
traintypes = {
    "fare_amount": "float32",
    "pickup_datetime": "str",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}
cols = list(traintypes.keys())
chunksize = 2**21  # 2,097,152 rows per chunk
total_chunk = n_rows // chunksize + 1
df_list = []
i = 0

for df_chunk in pd.read_csv(
    TRAIN_PATH, usecols=cols, dtype=traintypes, chunksize=chunksize
):
    i += 1
    print(f"DataFrame Chunk {i:02d}/{total_chunk}")
    clean_data(df_chunk)
    df_list.append(df_chunk)
    del df_chunk
    break  # keep a single chunk for speed in this environment
print("Complete")




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/ops/array_ops.py in na_logical_op(x, y, op)
    361         #  (xint or xbool) and (yint or bool)
--> 362         result = op(x, y)
    363     except TypeError:

TypeError: unsupported operand type(s) for |: 'float' and 'float'

During handling of the above exception, another exception occurred:

TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/611171117.py in <cell line: 0>()
     19     i += 1
     20     print(f"DataFrame Chunk {i:02d}/{total_chunk}")
---> 21     clean_data(df_chunk)
     22     df_list.append(df_chunk)
     23     del df_chunk

/tmp/ipykernel_55/4163151638.py in clean_data(df, test)
     15 
     16         df.drop(
---> 17             ((df[df.fare_amount > MAX_FARE]) | (df[df.fare_amount < MIN_FARE])).index,
     18             axis=0,
     19             inplace=True,

/usr/local/lib/python3.11/dist-packages/pandas/core/ops/common.py in new_method(self, other)
     74         other = item_from_zerodim(other)
     75 
---> 76         return method(self, other)
     77 
     78     return new_method

/usr/local/lib/python3.11/dist-packages/pandas/core/arraylike.py in __or__(self, other)
     76     @unpack_zerodim_and_defer("__or__")
     77     def __or__(self, other):
---> 78         return self._logical_method(other, operator.or_)
     79 
     80     @unpack_zerodim_and_defer("__ror__")

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _arith_method(self, other, op)
   7911 
   7912         with np.errstate(all="ignore"):
-> 7913             new_data = self._dispatch_frame_op(other, op, axis=axis)
   7914         return self._construct_result(new_data)
   7915 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _dispatch_frame_op(self, right, func, axis)
   7954 
   7955             # TODO operate_blockwise expects a manager of the same type
-> 7956             bm = self._mgr.operate_blockwise(
   7957                 # error: Argument 1 to "operate_blockwise" of "ArrayManager" has
   7958                 # incompatible type "Union[ArrayManager, BlockManager]"; expected

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in operate_blockwise(self, other, array_op)
   1509         Apply array_op blockwise with another (aligned) BlockManager.
   1510         """
-> 1511         return operate_blockwise(self, other, array_op)
   1512 
   1513     def _equal_values(self: BlockManager, other: BlockManager) -> bool:

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/ops.py in operate_blockwise(left, right, array_op)
     63     res_blks: list[Block] = []
     64     for lvals, rvals, locs, left_ea, right_ea, rblk in _iter_block_pairs(left, right):
---> 65         res_values = array_op(lvals, rvals)
     66         if (
     67             left_ea

/usr/local/lib/python3.11/dist-packages/pandas/core/ops/array_ops.py in logical_op(left, right, op)
    452             is_other_int_dtype = lib.is_integer(rvalues)
    453 
--> 454         res_values = na_logical_op(lvalues, rvalues, op)
    455 
    456         # For int vs int `^`, `|`, `&` are bitwise operators and return

/usr/local/lib/python3.11/dist-packages/pandas/core/ops/array_ops.py in na_logical_op(x, y, op)
    367             x = ensure_object(x)
    368             y = ensure_object(y)
--> 369             result = libops.vec_binop(x.ravel(), y.ravel(), op)
    370         else:
    371             # let null fall thru

ops.pyx in pandas._libs.ops.vec_binop()

ops.pyx in pandas._libs.ops.vec_binop()

TypeError: unsupported operand type(s) for |: 'float' and 'bool'

## === cell 6
X = pd.concat(df_list, ignore_index=True)
del df_list
gc.collect()




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/3741886720.py in <cell line: 0>()
----> 1 X = pd.concat(df_list, ignore_index=True)
      2 del df_list
      3 gc.collect()
      4 
      5 

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/concat.py in concat(objs, axis, join, ignore_index, keys, levels, names, verify_integrity, sort, copy)
    380         copy = False
    381 
--> 382     op = _Concatenator(
    383         objs,
    384         axis=axis,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/concat.py in __init__(self, objs, axis, join, keys, levels, names, ignore_index, verify_integrity, copy, sort)
    443         self.copy = copy
    444 
--> 445         objs, keys = self._clean_keys_and_objs(objs, keys)
    446 
    447         # figure out what our result ndim is going to be

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/concat.py in _clean_keys_and_objs(self, objs, keys)
    505 
    506         if len(objs_list) == 0:
--> 507             raise ValueError("No objects to concatenate")
    508 
    509         if keys is None:

ValueError: No objects to concatenate

## === cell 7
minmax = {}  # store (min, max) for each column
norm = pd.DataFrame()

float32cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "distance",
]
float16cols = ["passenger_count", "year", "month", "day", "dayofweek", "hour", "minute"]

for col in float32cols:
    col_min = X[col].min()
    col_max = X[col].max()
    minmax[col] = (col_min, col_max)
    norm[col] = ((X[col] - col_min) / (col_max - col_min)).astype("float32")

for col in float16cols:
    col_min = X[col].min()
    col_max = X[col].max()
    minmax[col] = (col_min, col_max)
    norm[col] = ((X[col] - col_min) / (col_max - col_min)).astype("float16")

norm["fare_amount"] = X["fare_amount"]
X = norm
del norm
gc.collect()
print(X.head())
X.info()




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/460276268.py in <cell line: 0>()
     12 
     13 for col in float32cols:
---> 14     col_min = X[col].min()
     15     col_max = X[col].max()
     16     minmax[col] = (col_min, col_max)

NameError: name 'X' is not defined

## === cell 8
X = X.sample(frac=1, random_state=42).reset_index(drop=True)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2925413729.py in <cell line: 0>()
      1 # shuffle the data
----> 2 X = X.sample(frac=1, random_state=42).reset_index(drop=True)
      3 
      4 

NameError: name 'X' is not defined

## === cell 9
y = X["fare_amount"]
X.drop(columns="fare_amount", inplace=True)
X.info()




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3711948083.py in <cell line: 0>()
----> 1 y = X["fare_amount"]
      2 X.drop(columns="fare_amount", inplace=True)
      3 X.info()
      4 
      5 

NameError: name 'X' is not defined

## === cell 10
validation_portion = 2.5 / 100
index = int(X.shape[0] * validation_portion)
print("training:\t%d\nvalidation:\t%d" % (X.shape[0] - index, index))




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2931733817.py in <cell line: 0>()
      1 validation_portion = 2.5 / 100
----> 2 index = int(X.shape[0] * validation_portion)
      3 print("training:\t%d\nvalidation:\t%d" % (X.shape[0] - index, index))
      4 
      5 

NameError: name 'X' is not defined

## === cell 11
val_X = X.iloc[:index].copy()
X.drop(X.index[:index], inplace=True)




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2672669485.py in <cell line: 0>()
----> 1 val_X = X.iloc[:index].copy()
      2 X.drop(X.index[:index], inplace=True)
      3 
      4 

NameError: name 'X' is not defined

## === cell 12
val_y = y.iloc[:index].copy()
y.drop(y.index[:index], inplace=True)




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2093607062.py in <cell line: 0>()
----> 1 val_y = y.iloc[:index].copy()
      2 y.drop(y.index[:index], inplace=True)
      3 
      4 

NameError: name 'y' is not defined

## === cell 13
import sys

ipython_vars = ["In", "Out", "exit", "quit", "get_ipython", "ipython_vars"]
sorted(
    [
        (x, sys.getsizeof(globals().get(x)))
        for x in dir()
        if not x.startswith("_") and x not in sys.modules and x not in ipython_vars
    ],
    key=lambda x: x[1],
    reverse=True,
)[:10]




## === cell 15
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras import metrics as keras_metrics

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(e)




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 16
model = Sequential()
model.add(Dense(64, input_dim=X.shape[1], activation="relu"))
model.add(Dropout(0.25))

for _ in range(5):
    model.add(Dense(128, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

model.add(Dense(1))

model.compile(
    loss="mean_squared_error",
    optimizer="nadam",
    metrics=[keras_metrics.MeanAbsoluteError()],
)




## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3023504250.py in <cell line: 0>()
      1 model = Sequential()
----> 2 model.add(Dense(64, input_dim=X.shape[1], activation="relu"))
      3 model.add(Dropout(0.25))
      4 
      5 for _ in range(5):

NameError: name 'X' is not defined

## === cell 17
num_epochs = 2  # a few epochs are enough for this demo
batch_size = 2**10
history = model.fit(
    X.values,
    y.values,
    validation_data=(val_X.values, val_y.values),
    shuffle=True,
    epochs=num_epochs,
    batch_size=batch_size,
    verbose=2,
)




## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2216314625.py in <cell line: 0>()
      2 batch_size = 2**10
      3 history = model.fit(
----> 4     X.values,
      5     y.values,
      6     validation_data=(val_X.values, val_y.values),

NameError: name 'X' is not defined

## === cell 18
plt.figure()
plt.plot(history.history["loss"], color="blue", label="Train")
plt.plot(history.history["val_loss"], color="red", label="Validation")
plt.legend(loc="upper left")
plt.ylabel("Loss")
plt.xlabel("Epoch")

plt.figure()
plt.plot(history.history["mean_absolute_error"], color="blue", label="Train")
plt.plot(history.history["val_mean_absolute_error"], color="red", label="Validation")
plt.legend(loc="upper left")
plt.ylabel("Mean Absolute Error")
plt.xlabel("Epoch")
plt.show()




## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2087450784.py in <cell line: 0>()
      1 plt.figure()
----> 2 plt.plot(history.history["loss"], color="blue", label="Train")
      3 plt.plot(history.history["val_loss"], color="red", label="Validation")
      4 plt.legend(loc="upper left")
      5 plt.ylabel("Loss")

NameError: name 'history' is not defined

## === cell 19
val_pred = model.predict(val_X.values[:5]).flatten()
print("actual:", val_y.values[:5])
print("pred:  ", val_pred)




## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1252441242.py in <cell line: 0>()
----> 1 val_pred = model.predict(val_X.values[:5]).flatten()
      2 print("actual:", val_y.values[:5])
      3 print("pred:  ", val_pred)
      4 
      5 

NameError: name 'val_X' is not defined

## === cell 20
traintypes_test = {
    "pickup_datetime": "str",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}
cols_test = list(traintypes_test.keys())
cols_test.append("key")

X_test = pd.read_csv(TEST_PATH, usecols=cols_test, dtype=traintypes_test)
clean_data(X_test, test=True)
X_test_key = X_test["key"].copy()
X_test.drop(columns=["pickup_datetime", "key"], inplace=True)

for col in float16cols:
    col_min, col_max = minmax[col]
    X_test[col] = ((X_test[col] - col_min) / (col_max - col_min)).astype("float16")

for col in float32cols:
    col_min, col_max = minmax[col]
    X_test[col] = ((X_test[col] - col_min) / (col_max - col_min)).astype("float32")




## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/2224619357.py in <cell line: 0>()
     16 
     17 for col in float16cols:
---> 18     col_min, col_max = minmax[col]
     19     X_test[col] = ((X_test[col] - col_min) / (col_max - col_min)).astype("float16")
     20 

KeyError: 'passenger_count'

## === cell 21
pred = model.predict(X_test.values).flatten()
pred = np.round(pred, 2)




## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/1218071476.py in <cell line: 0>()
----> 1 pred = model.predict(X_test.values).flatten()
      2 pred = np.round(pred, 2)
      3 
      4 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/models/sequential.py in build(self, input_shape)
    162             return
    163         if not self._layers:
--> 164             raise ValueError(
    165                 f"Sequential model {self.name} cannot be built because it has "
    166                 "no layers. Call `model.add(layer)`."

ValueError: Sequential model sequential cannot be built because it has no layers. Call `model.add(layer)`.

## === cell 22
results = pd.DataFrame({"key": X_test_key.astype(str), "fare_amount": pred})
results.info()
results.to_csv("submission.csv", index=False)

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/632875686.py in <cell line: 0>()
----> 1 results = pd.DataFrame({"key": X_test_key.astype(str), "fare_amount": pred})
      2 results.info()
      3 results.to_csv("submission.csv", index=False)

NameError: name 'pred' is not defined
