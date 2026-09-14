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

4.28899

# 6. Current score

202.39076

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 15.25766) has done: 'Diagnosis: The crash happens immediately when importing `keras` in cell 0. In this environment, `keras==3.8.0` pulls in a protobuf-related dependency chain that is incompatible with the installed `google-api-python-client`/protobuf stack, triggering `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` during import time. Since the notebook’s code uses the classic `keras.*` API and `tf_keras==2.18.0` is available, the minimal, safe fix is to switch the imports in cell 0 from `keras` to `tf_keras`, which avoids the problematic dependency path while keeping the same API surface for the used objects.

Patch summary: Modify only the Keras-related imports in cell 0 to import from `tf_keras` instead of `keras`. Keep all constants, paths, and the rest of the logic unchanged.

Updated cells: Cell 0 only.

Compatibility notes for cell k+1: Cell 1 does not depend on Keras symbols; it only uses `pd.read_csv` and the path constants defined in cell 0. Those constants remain unchanged, so cell 1 remains fully compatible.

Assumptions: `tf_keras==2.18.0` is importable in this environment and provides the same modules/classes used here (`Sequential`, `Dense`, `Dropout`, `BatchNormalization`, `LSTM`, `EarlyStopping`, `optimizers`, `regularizers`).'
- What this solution (achieved 15.21396) has done: 'Diagnosis: The crash happens immediately on importing `tf_keras` in cell 0, before any data/model code runs. This is a known incompatibility between the installed `protobuf` runtime and some TensorFlow/Keras builds, where `google.protobuf.message_factory.MessageFactory.GetPrototype` is missing in newer protobuf versions. The minimal, deterministic fix is to force protobuf to use the pure-Python implementation before importing `tf_keras`, which avoids that attribute call path. No model/training logic is changed.

Patch summary: In cell 0, set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (and version) via `os.environ` before importing anything from `tf_keras`. Keep all other imports and constants unchanged.

Updated cells: only cell 0.

Compatibility notes for cell k+1: All symbols (`np`, `pd`, `TRAIN_PATH`, etc.) remain defined exactly as before; cell 1 run unchanged.

Assumptions: The environment allows setting `os.environ` at runtime and the protobuf Python implementation is available (standard with protobuf installs).'
- What this solution (achieved 15.25064) has done: 'The crash happens immediately on importing `tf_keras` in cell 0, before any data loading, due to an incompatibility between the installed protobuf runtime and TensorFlow/Keras that triggers `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. The minimal fix is to force the pure-Python protobuf implementation earlier and also disable C++ descriptors, which avoids the missing `GetPrototype` path in this environment. This change is localized to cell 0 and keeps the model/training logic untouched. All variable names, constants, and imports used by later cells remain the same.'
- What this solution (achieved 15.23554) has done: 'Diagnosis: The crash happens in cell 0 before any model/data code runs, due to an incompatibility between the installed `protobuf` version and libraries imported in this cell (triggered during TensorFlow/Keras/protobuf initialization). The specific error (`'MessageFactory' object has no attribute 'GetPrototype'`) is a known symptom when newer `protobuf` APIs are used while some dependency expects the older `GetPrototype` method. The existing environment-variable workaround is insufficient in this environment because it does not guarantee that the missing method exists.

Patch summary: In cell 0 only, add a small, deterministic compatibility shim that monkey-patches `google.protobuf.message_factory.MessageFactory` to provide a `GetPrototype` method when it is missing, delegating to `GetMessageClass` (or falling back safely). This avoids changing any model/training logic and unblocks the imports that currently crash.

Updated cells: cell 0 only.

Compatibility notes for cell k+1: All variables and imports defined in cell 0 remain available with the same names and semantics (`np`, `pd`, tf_keras imports, constants like `TRAIN_PATH`, etc.), so cell 1 continues to run unchanged.

Assumptions: `google.protobuf` is available (it is, since the error originates from it), and providing `GetPrototype` via `GetMessageClass` is sufficient to satisfy the older API expectation without altering downstream behavior.'
- What this solution (achieved 15.23109) has done: 'The crash happens because `matplotlib.pyplot.imread()` no longer accepts a URL string directly, so reading the NYC land/water mask fails inside `remove_datapoints_from_water()`. The minimal fix is to open the URL as a binary stream and load it via Pillow (PIL), then convert to a NumPy array—exactly as the error message suggests. This keeps the same mask logic and downstream filtering semantics unchanged while restoring compatibility with the installed Matplotlib version. The change is fully contained within the failing cell and does not alter any interfaces used later.'
- What this solution (achieved 15.26652) has done: 'Diagnosis: Cell 9 fails inside `remove_datapoints_from_water()` because newer Matplotlib no longer allows `plt.imread()` to read directly from a URL string, raising a `ValueError`. The function already imports `urllib.request` and `PIL.Image` in cell 9, but that doesn’t affect the function’s internal URL read. We should change only cell 9 to monkey‑patch `plt.imread` so that it can accept URLs by fetching bytes via `urllib.request.urlopen` and decoding with Pillow, while preserving the same return type/shape expected by the existing mask logic.

Patch summary: In cell 9, before calling `clean()`, wrap/override `plt.imread` with a small adapter that detects `http(s)://` strings and loads them via `PIL.Image.open(urllib.request.urlopen(url))`, returning a NumPy array just like `plt.imread`. Non-URL paths are delegated to the original `plt.imread`, keeping all other behavior unchanged.

Updated cells: Only cell 9 is modified.

Compatibility notes for cell k+1: `train_df` and `validation_df` are still produced by `clean()` exactly as before, so `train_df.describe()` in cell 10 remains compatible.

Assumptions: Network access to download the NYC mask image is available; if it is blocked, the original code would still not be able to proceed without further changes (not made here).'
- What this solution (achieved 15.1959) has done: 'The crash is caused by `remove_datapoints_from_water()` trying to download an NYC land/water mask image from an external URL that now returns HTTP 404, so `plt.imread()` fails. To keep the existing cleaning pipeline intact and deterministic, I modify cell 9 to intercept `plt.imread()` calls for that specific URL and instead load a local copy of the mask if available. If no local copy exists, I fall back to a safe all-land mask (so the function won’t filter anything out due to the missing image) and emit a short warning. This change is localized to cell 9 and does not alter the model/training logic or downstream interfaces.'
- What this solution (achieved 15.46665) has done: 'Diagnosis: Cell 24 crashes because `tf_keras.optimizers` (TF-Keras 2.18) does not expose a lowercase `adam` constructor; the correct class is `optimizers.Adam`. Also, the legacy argument name `lr` is not accepted in this API; it should be `learning_rate`.  
Patch summary: In cell 24 only, replace `optimizers.adam(lr=LEARNING_RATE)` with `optimizers.Adam(learning_rate=LEARNING_RATE)` to create the same optimizer without changing training semantics.  
Updated cells: Only cell 24 is modified below.  
Compatibility notes for cell k+1: The variable `model` remains a compiled `Sequential` model, so cell 25’s visualization code continues to work unchanged.  
Assumptions: `tf_keras==2.18.0` follows standard TF-Keras optimizer naming (`Adam`) and accepts `learning_rate`.'
- What this solution (achieved 344.44037) has done: 'Diagnosis: The crash happens during `model.fit()` when computing the custom `rmse` metric. In this environment, `from keras import backend` resolves to Keras 3’s `keras.api.backend`, which no longer exposes legacy backend math ops like `sqrt/mean/square`, so calling `backend.sqrt(...)` raises `AttributeError`. The model itself is built with `tf_keras`, so the metric should use TensorFlow math ops compatible with `tf_keras` execution.

Patch summary: Update cell 24 to rebind the `rmse` metric locally to a TensorFlow-based implementation (same formula/semantics), overriding the incompatible version imported earlier. This is the smallest change that fixes training without altering model architecture, data, loss, or training loop behavior.

Updated cells: cell 24 only.

Compatibility notes for cell k+1: `model` and `history` remain defined exactly as before; `model_to_dot(model)` in cell 25 still work. The only change is that `rmse` now executes correctly during training.

Assumptions: TensorFlow is available through `tf_keras` (standard in this environment), so `tf.math.*` ops are usable.'
- What this solution (achieved 66.67489) has done: 'Diagnosis: Cell 25 crashes because `keras==3.8.0` no longer provides `keras.utils.vis_utils.model_to_dot` (it was part of legacy `tf.keras` utilities), so the import fails with `ModuleNotFoundError`. The rest of the notebook does not depend on this visualization, so we should make the visualization step compatible with Keras 3 without altering training/evaluation logic. The minimal safe fix is to try the old import first, then fall back to the Keras 3-supported `keras.utils.model_to_dot`, and if neither is available (or Graphviz is missing), skip rendering instead of crashing.

Patch summary: Update only cell 25 to use a compatibility import path for `model_to_dot` and guard the SVG rendering so missing optional dependencies do not stop execution.

Updated cells: Only cell 25.

Compatibility notes for cell k+1: `history` and `model` remain unchanged; cell 26 (`plot_loss_accuracy_rmse(history)`) run as before.

Assumptions: Model visualization is optional and should not be required for subsequent cells; Graphviz/pydot may be absent in the environment, so failing gracefully is acceptable.'
- What this solution (achieved 202.39076) has done: 'Diagnosis: Cell 30 crashes because `keras.losses.mean_squared_error(y_true, y_pred)` requires `y_true` and `y_pred` to have broadcastable shapes, but the code slices them to different lengths (`test_labels[1000:]` vs `prediction[:1000]`). This produces incompatible tensors (roughly 39k vs 1k rows), triggering the `InvalidArgumentError` during subtraction. The intent appears to be computing MSE on a comparable subset; we align both arrays to the same slice length without changing any modeling logic.

Patch summary: In cell 30, change the slicing so `y_true` and `y_pred` refer to the same first 1000 samples (or any equal-length slice), and ensure `y_true` is shaped `(N, 1)` to match the model’s `(N, 1)` predictions.

Updated cells: Only cell 30 is modified.

Compatibility notes for cell k+1: No variables used by cell 32 are changed; `predictionKaggle` and `testKaggle` remain untouched and `output_submission(...)` work as before.

Assumptions: The goal of cell 30 is a quick sanity-check loss value on a small subset rather than any specific offset slice.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")
os.environ.setdefault(
    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_C_DESCRIPTORS", "1"
)

try:
    from google.protobuf import message_factory as _message_factory

    if hasattr(_message_factory, "MessageFactory") and not hasattr(
        _message_factory.MessageFactory, "GetPrototype"
    ):

        def _GetPrototype(self, descriptor):
            if hasattr(self, "GetMessageClass"):
                return self.GetMessageClass(descriptor)
            try:
                return _message_factory.GetMessageClass(descriptor)
            except Exception:
                return None

        _message_factory.MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from tf_keras.models import Sequential
from tf_keras.layers import Dense, Dropout, BatchNormalization, LSTM
from tf_keras.callbacks import EarlyStopping
from tf_keras import optimizers
from tf_keras import regularizers

TRAIN_PATH = "../input/train.csv"
TEST_PATH = "../input/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 0.001
DATASET_SIZE = 80000


## === cell 1
datatypes = {'key': 'str', 
              'fare_amount': 'float32',
              'pickup_datetime': 'str', 
              'pickup_longitude': 'float32',
              'pickup_latitude': 'float32',
              'dropoff_longitude': 'float32',
              'dropoff_latitude': 'float32',
              'passenger_count': 'uint8'}

trainKaggle = pd.read_csv(TRAIN_PATH, nrows=DATASET_SIZE, dtype=datatypes, usecols=[1,2,3,4,5,6,7])
testKaggle = pd.read_csv(TEST_PATH)    


## === cell 2
train_df, test_df = train_test_split(trainKaggle, test_size=0.50, random_state=1)


## === cell 3
train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=1)


## === cell 4
print('testKaggle Size %d' % len(testKaggle))
print('train_df Size %d' % len(train_df))
print('validation_df Size %d' % len(validation_df))
print('test_df Size %d' % len(test_df))


## === cell 5
train_df.describe()


## === cell 6
validation_df.describe()


## === cell 7
test_df.describe()


## === cell 8
def clean(df):
    
    print(' Old size: %d' % len(df))
    df = df.dropna(how = 'any', axis = 'rows')
    print(' New size after dropna: %d' % len(df))  
    
    df = df[(df['dropoff_longitude'] != df['pickup_longitude']) & (df['dropoff_latitude'] != df['pickup_latitude'])]
    print(' New size after removing same long lat: %d' % len(df))                 
    
    df = df[(df['dropoff_longitude'] != 0) & (df['pickup_longitude'] != 0) & (df['dropoff_latitude'] != 0) & (df['pickup_latitude'] != 0)] 
    print(' New size after removing 0 long lat: %d' % len(df))                          
             
    MinMax = (-74.5, -72.8, 40.5, 41.8)
    df = df[(MinMax[0] <= df['pickup_longitude']) & (df['pickup_longitude'] <= MinMax[1])]
    df = df[(MinMax[0] <= df['dropoff_longitude']) & (df['dropoff_longitude'] <= MinMax[1])]
    df = df[(MinMax[2] <= df['pickup_latitude']) & (df['pickup_latitude'] <= MinMax[3])]
    df = df[(MinMax[2] <= df['dropoff_latitude']) & (df['dropoff_latitude'] <= MinMax[3])]
    
    print(' New size after only NYC: %d' % len(df)) 
    df = df[(0 < df['fare_amount']) & (df['fare_amount'] <= 50)]
    
    print(' New size after removing outliers: %d' % len(df)) 
    
    df = df[(df['passenger_count'] > 0) & (df['passenger_count'] <= 6)]
    print(' New size after removing 6=>passenger_count > 0 : %d' % len(df)) 
    
    
    
    
    nyc_coord = (40.7141667,-74.0063889) 
    fk_coord = (40.639722, -73.778889)
    ewr_coord = (40.6925, -74.168611)
    lga_coord = (40.77725, -73.872611)
    sol_coord = (40.6892,-74.0445) # Statue of Liberty
    
             

    df = df[(nyc_coord[1] != df['pickup_longitude']) & (df['pickup_latitude'] != nyc_coord[0])]
    df = df[(nyc_coord[1] != df['dropoff_longitude']) & (df['dropoff_latitude'] != nyc_coord[0])]
    
    print(' New size after NY airport: %d' % len(df))
    
    df = df[(fk_coord[1] != df['pickup_longitude']) & (df['pickup_latitude'] != fk_coord[0])]
    df = df[(fk_coord[1] != df['dropoff_longitude']) & (df['dropoff_latitude'] != fk_coord[0])]
    
    print(' New size after jfk airport: %d' % len(df))
    
    df = df[(ewr_coord[1] != df['pickup_longitude']) & (df['pickup_latitude'] != ewr_coord[0])]
    df = df[(ewr_coord[1] != df['dropoff_longitude']) & (df['dropoff_latitude'] != ewr_coord[0])]
    
    print(' New size after ewr airport: %d' % len(df))
    df = df[(lga_coord[1] != df['pickup_longitude']) & (df['pickup_latitude'] != lga_coord[0])]
    df = df[(lga_coord[1] != df['dropoff_longitude']) & (df['dropoff_latitude'] != lga_coord[0])]
    

    print(' New size after lgr airport: %d' % len(df))
             
    df = df[(sol_coord[1] != df['pickup_longitude']) & (df['pickup_latitude'] != sol_coord[0])]
    df = df[(sol_coord[1] != df['dropoff_longitude']) & (df['dropoff_latitude'] != sol_coord[0])]
    

    print(' New size after sol removed: %d' % len(df))             
    
            
    print('Old size: %d' % len(df))
    df = remove_datapoints_from_water(df)
    print('New size: %d' % len(df))
    
        
    print(' New size: %d' % len(df))
    
    return df

def remove_datapoints_from_water(df):
    def lonlat_to_xy(longitude, latitude, dx, dy, BB):
        return (dx*(longitude - BB[0])/(BB[1]-BB[0])).astype('int'), \
               (dy - dy*(latitude - BB[2])/(BB[3]-BB[2])).astype('int')

    BB = (-74.5, -72.8, 40.5, 41.8)
    
    nyc_mask = plt.imread('https://aiblog.nl/download/nyc_mask-74.5_-72.8_40.5_41.8.png')[:,:,0] > 0.9
    
    pickup_x, pickup_y = lonlat_to_xy(df.pickup_longitude, df.pickup_latitude, 
                                      nyc_mask.shape[1], nyc_mask.shape[0], BB)
    dropoff_x, dropoff_y = lonlat_to_xy(df.dropoff_longitude, df.dropoff_latitude, 
                                      nyc_mask.shape[1], nyc_mask.shape[0], BB)    
    idx = nyc_mask[pickup_y, pickup_x] & nyc_mask[dropoff_y, dropoff_x]
    
    return df[idx]
    
def late_night (row):
     if (row['hour'] <= 3) or (row['hour'] >= 0):
        return 1
     else:
        return 0


def night (row):
    if ((row['hour'] > 20) and (row['hour'] > 0)) and (row['weekday'] < 5):
        return 1
    else:
        return 0
    
def rush_hour (row):
    if ((row['hour'] <= 20) and (row['hour'] >= 16)) and (row['weekday'] < 5):
        return 1
    else:
        return 0   
    

       
    
    
def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df):
    df['pickup_datetime'] =  pd.to_datetime(df['pickup_datetime'], format='%Y-%m-%d %H:%M:%S %Z')
    df['year'] = df['pickup_datetime'].apply(lambda x: x.year)
    df['month'] = df['pickup_datetime'].apply(lambda x: x.month)
    df['day'] = df['pickup_datetime'].apply(lambda x: x.day)
    df['hour'] = df['pickup_datetime'].apply(lambda x: x.hour)
    df['weekday'] = df['pickup_datetime'].apply(lambda x: x.weekday())
    df['pickup_datetime'] =  df['pickup_datetime'].apply(lambda x: str(x))
    df['night'] = df.apply (lambda x: night(x), axis=1)
    df['late_night'] = df.apply (lambda x: late_night(x), axis=1)
    df['rush_hour'] = df.apply (lambda x: rush_hour(x), axis=1)
    
    return df


def add_coordinate_features(df):
    lat1 = df['pickup_latitude']
    lat2 = df['dropoff_latitude']
    lon1 = df['pickup_longitude']
    lon2 = df['dropoff_longitude']
    
    df['latdiff'] = (lat1 - lat2)
    df['londiff'] = (lon1 - lon2)
    
    
    return df


def add_distances_features(df):
    
    lat1 = df['pickup_latitude']
    lat2 = df['dropoff_latitude']
    lon1 = df['pickup_longitude']
    lon2 = df['dropoff_longitude']
    
    df['manhattan'] = manhattan(lat1, lon1, lat2, lon2)
    df['distance'] = distance(lat1, lon1, lat2, lon2)
    
    
  
    return df

def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a)) # 2*R*asin...
def distanceP(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a)) # 2*R*asin...
    
def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    df = pd.DataFrame(prediction, columns=[prediction_column])
    df[id_column] = raw_test[id_column]
    df[[id_column, prediction_column]].to_csv((file_name), index=False)
    print('Output complete')
    
    
def plot_loss_accuracy_rmse(history):
    
    plt.figure(figsize=(20,10))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()
    
    
    plt.figure(figsize=(20,10))
    plt.plot(history.history['rmse'])
    plt.plot(history.history['val_rmse'])
    plt.title('Model rmse')
    plt.ylabel('rmse')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()
    
    


## === cell 9

import urllib.request
from PIL import Image
import io

_original_imread = plt.imread

_NYC_MASK_URL = "https://aiblog.nl/download/nyc_mask-74.5_-72.8_40.5_41.8.png"
_NYC_MASK_LOCAL_CANDIDATES = [
    "nyc_mask-74.5_-72.8_40.5_41.8.png",
    "../input/nyc_mask-74.5_-72.8_40.5_41.8.png",
    "../input/new-york-city-taxi-fare-prediction/nyc_mask-74.5_-72.8_40.5_41.8.png",
]

_cached_nyc_mask_img = None


def _imread_with_url_support(fname, format=None):
    global _cached_nyc_mask_img

    if isinstance(fname, str) and fname == _NYC_MASK_URL:
        if _cached_nyc_mask_img is not None:
            return _cached_nyc_mask_img

        for p in _NYC_MASK_LOCAL_CANDIDATES:
            if os.path.exists(p):
                _cached_nyc_mask_img = _original_imread(p)
                return _cached_nyc_mask_img

        try:
            with urllib.request.urlopen(fname) as resp:
                img = Image.open(resp)
                _cached_nyc_mask_img = np.array(img)
                return _cached_nyc_mask_img
        except Exception as e:
            print(
                f"Warning: failed to load NYC mask from URL ({e}); using all-land fallback mask."
            )
            _cached_nyc_mask_img = np.ones((1024, 1024, 3), dtype=np.uint8) * 255
            return _cached_nyc_mask_img

    if isinstance(fname, str) and fname.startswith(("http://", "https://")):
        with urllib.request.urlopen(fname) as resp:
            img = Image.open(resp)
            return np.array(img)

    return _original_imread(fname, format=format)


plt.imread = _imread_with_url_support

print("train_df clean")
train_df = clean(train_df)
print("validation_df clean")
validation_df = clean(validation_df)


## === cell 10
train_df.describe()


## === cell 11
validation_df.describe()


## === cell 12
print('train_df add_time_features')
train_df = add_time_features(train_df)
print('validation_df add_time_features')
validation_df = add_time_features(validation_df)
print('test_df add_time_features')
test_df = add_time_features(test_df)
print('testKaggle add_time_features')
testKaggle = add_time_features(testKaggle)


   


## === cell 13
print('train_df add_coordinate_features')
add_coordinate_features(train_df)
print('validation_df add_coordinate_features')
add_coordinate_features(validation_df)
print('test_df add_coordinate_features Disabled!')
add_coordinate_features(test_df)
print('testKaggle add_coordinate_features')
add_coordinate_features(testKaggle)


## === cell 14
print('train_df add_distances_features')
train_df = add_distances_features(train_df)
print('validation_df add_distances_features')
validation_df = add_distances_features(validation_df)
print('test_df add_distances_features')
test_df = add_distances_features(test_df)
print('testKaggle add_distances_features')
testKaggle = add_distances_features(testKaggle)


print('Done with Adding features')


## === cell 15
train_df.describe()


## === cell 16
validation_df.describe()


## === cell 17
dropped_columns = ['passenger_count', 'pickup_datetime'] #, 'pickup_latitude','pickup_longitude', 'dropoff_longitude', 'dropoff_latitude']

train_df = train_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)
validation_df = validation_df.drop(dropped_columns, axis=1)
testKaggle_clean = testKaggle.drop(dropped_columns + ['key'], axis=1)

print('Done with dropped_columns')


## === cell 18
train_labels = train_df['fare_amount'].values
validation_labels = validation_df['fare_amount'].values
test_labels = test_df['fare_amount'].values

train_df = train_df.drop(['fare_amount'], axis=1)
validation_df = validation_df.drop(['fare_amount'], axis=1)
test_df = test_df.drop(['fare_amount'], axis=1)

print('Done with Labels')


## === cell 19
train_df.shape


## === cell 20
test_df.shape


## === cell 21
validation_df.shape


## === cell 22
scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_df)
testKaggle_scaled = scaler.transform(testKaggle_clean)


## === cell 23
from keras import backend
def rmse(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


## === cell 24
import tensorflow as tf


def rmse(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true), axis=-1))


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
model.add(Dense(8, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(1))

adam = optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(
    loss="mean_squared_error", optimizer=adam, metrics=["mae", "accuracy", rmse]
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
    validation_data=(validation_df_scaled, validation_labels),
    shuffle=True,
)


## === cell 25
from IPython.display import SVG

try:
    from keras.utils.vis_utils import model_to_dot  # legacy (older tf.keras/keras)
except ModuleNotFoundError:
    from keras.utils import model_to_dot  # keras>=3

try:
    SVG(model_to_dot(model).create(prog="dot", format="svg"))
except Exception as e:
    print(f"Warning: could not render model graph ({e}). Skipping visualization.")


## === cell 26
plot_loss_accuracy_rmse(history)


## === cell 29
prediction = model.predict(test_scaled, batch_size=128, verbose=1)
predictionKaggle = model.predict(testKaggle_scaled, batch_size=128, verbose=1)


## === cell 30
import keras.losses  # import mean_squared_error

n = 1000
keras.losses.mean_squared_error(test_labels[:n].reshape(-1, 1), prediction[:n])


## === cell 32

output_submission(testKaggle, predictionKaggle, 'key', 'fare_amount', SUBMISSION_NAME)


## === cell 33
print(prediction[10000])
print(test_labels[10000])
