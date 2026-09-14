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

60.85243

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 15.25316) has done: 'Diagnosis: The crash happens in cell 16 because this notebook uses TensorFlow 1.x session/config APIs (`tf.ConfigProto`, `tf.Session`, `keras.backend.set_session`) and an old Keras import path (`keras.layers.normalization.BatchNormalization`). In this environment you have TensorFlow 2.18 + Keras 3, where those TF1 session APIs are removed and the legacy Keras backend/session wiring is unsupported; the resulting import/initialization chain triggers a protobuf incompatibility error (`MessageFactory...GetPrototype`). The fix is to remove TF1 session configuration and use TF2-compatible configuration (`tf.config.*`) while keeping the same model/training logic. We also update the BatchNormalization import to the supported Keras 3 path so that cell 17 can use `BatchNormalization` unchanged.'
- What this solution (achieved 15.25885) has done: 'Diagnosis: The crash happens during `import tensorflow/keras` in cell 16 due to an incompatibility between TensorFlow 2.18 and the installed `protobuf==6.33.0`, which triggers `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` while TensorFlow initializes protocol buffers. This is a known breakage with protobuf 5/6 where TensorFlow expects the legacy `GetPrototype` API. Since we cannot change installed packages, the minimal runtime fix is to force TensorFlow to use the pure-Python protobuf implementation, which preserves the expected behavior.

Patch summary: In cell 16 only, set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (and version `2`) **before** importing TensorFlow/Keras. This avoids the C++ protobuf backend that exposes the incompatible `MessageFactory` behavior, unblocking the imports and leaving the model/training logic unchanged.

Updated cells: (cell 16 only)

Compatibility notes for cell k+1: All symbols imported in cell 16 (`tf`, `keras`, `Sequential`, `Dense`, `Dropout`, `BatchNormalization`, `RMSprop`, `metrics`, `K`) remain defined exactly as before, so cell 17 run unchanged.

Assumptions: The environment allows setting environment variables at runtime before importing TensorFlow, and using the Python protobuf backend is acceptable for this notebook (slower but correct and deterministic).'
- What this solution (achieved 15.24387) has done: 'Diagnosis: The crash occurs in cell 16 during `import tensorflow as tf` because the cell forces `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="cpp"`, which makes TensorFlow try to use the protobuf C++ extension (`google.protobuf.pyext._message`) that is not available/compatible in this environment (protobuf==6.33.0). This results in `ImportError: cannot import name '_message'`. TensorFlow works reliably here with the pure-Python protobuf backend instead of the forced C++ backend.

Patch summary: Modify only cell 16 to stop forcing the C++ protobuf implementation and instead force the pure-Python protobuf backend before importing TensorFlow/Keras. This avoids the missing `_message` extension and allows the rest of the notebook (model definition in cell 17 and onward) to run unchanged.

Updated cells: Cell 16 only.

Compatibility notes for cell k+1: All imports and symbols used in cell 17 (`Sequential`, `Dense`, `Dropout`, `BatchNormalization`, `metrics`, etc.) are still imported with the same names, so cell 17 remains compatible without any changes.

Assumptions: The environment supports TensorFlow 2.18 with the Python protobuf implementation (it does), and no other code relies on protobuf C++ acceleration.'
- What this solution (achieved 15.25554) has done: 'Diagnosis: The crash happens while importing TensorFlow/Keras in cell 16 due to an incompatibility between TensorFlow 2.18 and `protobuf==6.33.0`. TensorFlow 2.18 expects protobuf 4.x/5.x APIs, and with protobuf 6 the internal call path hits `MessageFactory.GetPrototype`, which no longer exists, causing `AttributeError`. The attempted workaround via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` is not sufficient to restore the removed API. The minimal deterministic fix is to downgrade protobuf to a TF-compatible version (e.g., 5.28.3) at runtime before importing TensorFlow/Keras.

Patch summary: In cell 16 only, install a TensorFlow-compatible protobuf version with pip (quietly), then import TensorFlow/Keras as before. Keep the existing environment-variable settings and the rest of the logic unchanged.

Updated cells: Cell 16 only (below).

Compatibility notes for cell k+1: All symbols used in cell 17 (`Sequential`, `Dense`, `Dropout`, `BatchNormalization`, etc.) remain imported exactly as before; only the protobuf version is corrected so the imports succeed.

Assumptions: The environment allows `pip` installs during runtime (common in notebook runtimes), and downgrading protobuf to 5.28.3 does not conflict with the current TensorFlow 2.18 installation.'
- What this solution (achieved 78.31645) has done: 'Diagnosis: Cell 17 crashes because in Keras 3 the `keras.metrics` module no longer exposes the alias `mae` as an attribute (`metrics.mae`), so accessing it raises `AttributeError`. The model compile call expects a valid metric object or a supported string identifier. Using the canonical string name `"mae"` preserves the exact evaluation semantics (mean absolute error) without changing the model architecture or training behavior.

Patch summary: In cell 17, replace `metrics=[metrics.mae]` with `metrics=["mae"]` so Keras can resolve the metric correctly under Keras 3. No other logic is changed.

Updated cells: (cell 17 only)

Compatibility notes for cell k+1: `model` remains compiled with MAE tracked, so `model.fit(...)` in cell 18 works unchanged; `history` still contain MAE keys (typically `mae` / `val_mae`).

Assumptions: The intended metric is mean absolute error; using the string identifier `"mae"` is equivalent to the previous alias and is supported by the installed Keras version.'
- What this solution (achieved 60.85243) has done: 'Diagnosis: Cell 19 crashes because Keras 3 records metric history under the compiled metric name (here `"mae"`), not the legacy key `"mean_absolute_error"`. Therefore `history.history['mean_absolute_error']` and `history.history['val_mean_absolute_error']` do not exist, raising a KeyError.  
Patch summary: In cell 19 only, change the history dictionary keys to use `"mae"` and `"val_mae"` while keeping plotting logic identical.  
Updated cells: Only cell 19 is modified.  
Compatibility notes for cell k+1: No variables are renamed or removed; `history` remains unchanged and subsequent cells (e.g., model prediction in cell 20) are unaffected.  
Assumptions: The model was compiled with `metrics=["mae"]` as shown, so `history.history` contains `"mae"` and `"val_mae"`.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import gc


## === cell 1
TRAIN_PATH = '../input/train.csv'
TEST_PATH = '../input/test.csv'


## === cell 2
p = subprocess.Popen(['wc', '-l', TRAIN_PATH], stdout=subprocess.PIPE, 
                                               stderr=subprocess.PIPE)
result, err = p.communicate()
if p.returncode != 0:
    raise IOError(err)
n_rows = int(result.strip().split()[0])+1


## === cell 3
def compute_haversine_distance(df, lat1='pickup_latitude', long1='pickup_longitude', lat2='dropoff_latitude', long2='dropoff_longitude'):
    R = 3959 # radius of earth in miles
    phi1 = np.radians(df[lat1])
    phi2 = np.radians(df[lat2])

    delta_phi = np.radians(df[lat2]-df[lat1])
    delta_lambda = np.radians(df[long2]-df[long1])

    a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    d = (R * c)
    df["distance"] = d.astype('float32')


## === cell 4
MIN_FARE = 2.50
MAX_FARE = 500

MIN_PASSENGER = 1
MAX_PASSENGER = 6

def clean_data(df, test=False):
    compute_haversine_distance(df)
    add_date_features(df, test)
    
    if not test:
        df.drop(df[df.isnull().any(1)].index, axis = 0, inplace=True)

        df.drop(((df[df.fare_amount>MAX_FARE]) | (df[df.fare_amount<MIN_FARE])).index, axis=0, inplace=True)
        df.drop(df[df.passenger_count > MAX_PASSENGER].index, axis = 0, inplace=True)
        df.drop(df[df.passenger_count < MIN_PASSENGER].index, axis = 0, inplace=True)
        df.drop(((df[df.pickup_latitude>90])    | (df[df.pickup_latitude<-90])    ).index, axis=0, inplace=True)
        df.drop(((df[df.pickup_longitude>180])  | (df[df.pickup_longitude<-180])  ).index, axis=0, inplace=True)
        df.drop(((df[df.dropoff_latitude>90])   | (df[df.dropoff_latitude<-90])   ).index, axis=0, inplace=True)
        df.drop(((df[df.dropoff_longitude>180]) | (df[df.dropoff_longitude<-180]) ).index, axis=0, inplace=True)
    
        df.drop(df[df.distance > 100].index, axis = 0, inplace=True)
        df.drop(df[df.distance <= 0].index, axis = 0, inplace=True)

    if not test:
        df.drop(columns=['pickup_datetime'], inplace=True) 

def add_date_features(df, test=False):
    df["pickup_datetime_clone"] = df["pickup_datetime"].values
    df.pickup_datetime_clone = df.pickup_datetime_clone.str.slice(0, 16)
    df.pickup_datetime_clone = pd.to_datetime(df.pickup_datetime_clone, utc=True, format='%Y-%m-%d %H:%M')
    df['year'] = df.pickup_datetime_clone.dt.year.astype('uint8')
    df['month'] = df.pickup_datetime_clone.dt.month.astype('uint8')
    df['day'] = df.pickup_datetime_clone.dt.day.astype('uint8')
    df['dayofweek'] = df.pickup_datetime_clone.dt.dayofweek.astype('uint8')
    df['hour'] = df.pickup_datetime_clone.dt.hour.astype('uint8')
    df['minute'] = df.pickup_datetime_clone.dt.minute.astype('uint8')
    df.drop(columns=['pickup_datetime_clone'], inplace=True) 


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
chunksize = 2**21  # 2,097,152
total_chunk = n_rows // chunksize + 1
df_list = []  # list to hold the batch dataframe
i = 0

_original_clean_data = clean_data


def clean_data(df, test=False):
    compute_haversine_distance(df)
    add_date_features(df, test)

    if not test:
        df.drop(df[df.isnull().any(axis=1)].index, axis=0, inplace=True)

        df.drop(
            df[(df.fare_amount > MAX_FARE) | (df.fare_amount < MIN_FARE)].index,
            axis=0,
            inplace=True,
        )
        df.drop(df[df.passenger_count > MAX_PASSENGER].index, axis=0, inplace=True)
        df.drop(df[df.passenger_count < MIN_PASSENGER].index, axis=0, inplace=True)
        df.drop(
            df[(df.pickup_latitude > 90) | (df.pickup_latitude < -90)].index,
            axis=0,
            inplace=True,
        )
        df.drop(
            df[(df.pickup_longitude > 180) | (df.pickup_longitude < -180)].index,
            axis=0,
            inplace=True,
        )
        df.drop(
            df[(df.dropoff_latitude > 90) | (df.dropoff_latitude < -90)].index,
            axis=0,
            inplace=True,
        )
        df.drop(
            df[(df.dropoff_longitude > 180) | (df.dropoff_longitude < -180)].index,
            axis=0,
            inplace=True,
        )

        df.drop(df[df.distance > 100].index, axis=0, inplace=True)
        df.drop(df[df.distance <= 0].index, axis=0, inplace=True)

    if not test:
        df.drop(columns=["pickup_datetime"], inplace=True)


for df_chunk in pd.read_csv(
    TRAIN_PATH, usecols=cols, dtype=traintypes, chunksize=chunksize
):
    i = i + 1
    print(f"DataFrame Chunk {i:02d}/{total_chunk}")
    clean_data(df_chunk)
    df_list.append(df_chunk)
    del df_chunk
    break
print("Complete")


## === cell 6
X = pd.concat(df_list)
del df_list


## === cell 7
minmax = pd.DataFrame()
norm = pd.DataFrame()
float32cols = ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "distance"]
float16cols = ["passenger_count", "year", "month", "day", "dayofweek", "hour", "minute"]
for col in float32cols:
    col_min = X[col].min()
    col_max = X[col].max()
    minmax[col] = (col_min, col_max)
    norm[col] = ((X[col] - col_min) / (col_max-col_min)).astype('float32')
    
for col in float16cols:
    col_min = X[col].min()
    col_max = X[col].max()
    minmax[col] = (col_min, col_max)
    norm[col] = ((X[col] - col_min) / (col_max-col_min)).astype('float16')

norm["fare_amount"] = X.fare_amount
X = norm
del norm
print(X[0:5])
X.info()


## === cell 8
X = X.sample(frac=1).reset_index(drop=True)


## === cell 9
gc.collect()


## === cell 10
y = X.fare_amount
X.drop(columns="fare_amount", inplace=True)
X.info()


## === cell 11
validation_portion = 2.5/100
index = int(X.shape[0]*validation_portion)
print("training:\t%d\nvalidation:\t%d" % (n_rows-index, index))


## === cell 12
val_X = X[0:index]
X.drop(X.index[0:index], inplace=True)


## === cell 13
val_y = y[0:index]
y.drop(y.index[0:index], inplace=True)


## === cell 14
import sys
ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']
sorted([(x, sys.getsizeof(globals().get(x))) for x in dir() if not x.startswith('_') and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True)


## === cell 15
X.drop(columns="fare_amount", inplace=True, errors="ignore")


## === cell 16
import os
import sys
import subprocess

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "protobuf==5.28.3"]
)

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import RMSprop
from keras import metrics
from keras import backend as K

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass


## === cell 17
model = Sequential()

model.add(Dense(64, input_dim=X.shape[1], activation="relu"))
model.add(Dropout(0.25))

for i in range(5):
    model.add(Dense(128, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

model.add(Dense(1))

model.compile(loss="mean_squared_error", optimizer="nadam", metrics=["mae"])


## === cell 18
num_epochs = 1
batch_size = 2**10
history = model.fit(X.values, y.values, 
                    validation_data = (val_X, val_y),
                    shuffle=True, 
                    epochs=num_epochs, 
                    batch_size=batch_size)


## === cell 19
plt.figure()
plt.plot(history.history["loss"], color="blue")
plt.plot(history.history["val_loss"], color="red")
plt.legend(["Train", "Validation"], loc="upper left")
plt.ylabel("loss")
plt.xlabel("epoch")

plt.figure()
plt.plot(history.history["mae"], color="blue")
plt.plot(history.history["val_mae"], color="red")
plt.legend(["Train", "Validation"], loc="upper left")
plt.ylabel("Mean Abs. Error")
plt.xlabel("epoch")


## === cell 20
val_pred = model.predict(val_X[0:5]).flatten()
print("actual: "+str(val_y[0:5].values))
print("pred:   "+str(val_pred))


## === cell 21
traintypes = {'pickup_datetime': 'str', 
              'pickup_longitude': 'float32',
              'pickup_latitude': 'float32',
              'dropoff_longitude': 'float32',
              'dropoff_latitude': 'float32',
              'passenger_count': 'uint8'}
cols = list(traintypes.keys())
cols.append('key')

X_test = pd.read_csv(TEST_PATH, usecols=cols, dtype=traintypes)
clean_data(X_test, test=True)
X_test_key = X_test['key']
X_test.drop(columns=['pickup_datetime'], inplace=True)
X_test.drop(columns=['key'], inplace=True)

float32cols = ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "distance"]
float16cols = ["passenger_count", "year", "month", "day", "dayofweek", "hour", "minute"]

for col in float16cols:
    col_min, col_max = minmax[col]
    X_test[col] = ((X_test[col] - col_min) / (col_max-col_min)).astype('float16')

for col in float32cols:
    col_min, col_max = minmax[col]
    X_test[col] = ((X_test[col] - col_min) / (col_max-col_min)).astype('float32')


## === cell 22
pred = model.predict(X_test).flatten()
pred = np.round(pred,2)


## === cell 23
results = pd.DataFrame({'key': X_test_key, 'fare_amount': pred})
results.key = results.key.astype(str)
results.info()
results.to_csv('submission.csv', index=False)
