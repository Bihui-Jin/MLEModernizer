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

18.93437

# 6. Current score

828.54431

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 15.2513) has done: 'The crash happens during `from keras...` imports in cell 1, before any of your code executes, and the traceback (`MessageFactory` has no `GetPrototype`) is a known incompatibility between the standalone `keras==3.x` package and the protobuf version in some environments. Since your notebook otherwise uses the TensorFlow-backed Keras API, the minimal fix is to switch these imports to `tf_keras` (which is installed as `tf_keras==2.18.0`) to avoid the protobuf issue while keeping the same high-level Keras classes (Sequential, Dense, etc.). No model/training logic is changed—only the import sources. Paths and constants remain untouched so later cells keep working.'
- What this solution (achieved 15.26778) has done: 'Diagnosis: The crash happens during the imports in cell 1, before any data is loaded. With `google-api-python-client==2.177.0` installed, importing TensorFlow/Keras can trigger a protobuf incompatibility where older protobuf APIs (used indirectly by installed google/protobuf stack) expect `MessageFactory.GetPrototype`, which is missing in newer protobuf versions. This manifests as `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` at import time. The minimal, deterministic workaround is to force protobuf to use the pure-Python implementation via environment variables before importing `tf_keras`.

Patch summary: In cell 1 only, set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (and version to `2`) via `os.environ` before importing any `tf_keras` modules. This avoids the incompatible C++/upb protobuf path that triggers the missing method error, while leaving the model/training logic unchanged.

Updated cells:  

Compatibility notes for cell k+1: All variable names/constants (`TRAIN_PATH`, `TEST_PATH`, `SUBMISSION_NAME`, `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE`, `DATASET_SIZE`) and imported symbols remain available exactly as before; cell 2 can continue to use `TRAIN_PATH/TEST_PATH` unchanged.

Assumptions: The environment allows setting `os.environ` within the notebook process before the first protobuf/tf_keras import; no earlier cell imported TensorFlow/protobuf in a way that would lock in the incompatible implementation.'
- What this solution (achieved 15.22277) has done: 'Diagnosis: The crash happens in cell 1 while importing `tf_keras`, which triggers an import of `tensorflow` and then fails inside `google.protobuf` with `cannot import name '_message'`. This indicates a protobuf C-extension incompatibility in the runtime, and the simplest way to unblock execution is to force protobuf to use the pure-Python implementation rather than the C-extension. The existing environment variable setting uses `"cpp"` which explicitly requests the C++/C-extension path and causes the failure.

Patch summary: In cell 1, change `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` from `"cpp"` to `"python"` (and keep the version unset) so protobuf avoids importing `google.protobuf.pyext._message`. This is a minimal, localized change and does not alter model/training logic.

Updated cells: Provided below (cell 1 only).

Compatibility notes for cell k+1: All imports/variables defined in cell 1 (`np`, `pd`, `plt`, sklearn utilities, model/layers/callbacks/optimizers/regularizers, constants like `TRAIN_PATH`, etc.) remain defined with the same names and intended usage, so cell 2 continues to run unchanged.

Assumptions: TensorFlow is available in the environment via `tf_keras` and import correctly once protobuf is forced to pure-Python.'
- What this solution (achieved 15.24914) has done: 'Diagnosis: The crash happens during `import tf_keras` in cell 1 and is a known incompatibility between TensorFlow/Keras protobuf bindings and the currently installed `protobuf` runtime, where older TF/keras code expects `google.protobuf.message_factory.MessageFactory.GetPrototype` but newer protobuf versions removed/changed it. Setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` alone is not sufficient here because the missing method is still referenced at import time. The minimal unblock is to monkey-patch `MessageFactory.GetPrototype` to delegate to `GetMessageClass` (or fall back to `GetPrototype` if present) before importing `tf_keras`. This keeps the rest of the notebook unchanged and preserves the `tf_keras` API expected by later cells.

Patch summary: In cell 1, add a small compatibility patch for protobuf (`MessageFactory.GetPrototype`) immediately after setting environment variables and before importing `tf_keras`. No other logic, paths, or model code is modified.

Updated cells: cell 1 only.

Compatibility notes for cell k+1: Cell 2 reads `TRAIN_PATH`, `TEST_PATH`, and uses `pd`/`np`; these remain defined exactly as before. The `tf_keras` imports now succeed so later training code can run unchanged.

Assumptions: The environment has a protobuf version where `MessageFactory` lacks `GetPrototype` but provides `GetMessageClass`; if not, the patch safely no-ops.'
- What this solution (achieved 15.20356) has done: 'Diagnosis: The crash happens inside `remove_datapoints_from_water()` because recent Matplotlib/Pillow no longer allows `plt.imread()` to read directly from a URL string, raising a `ValueError`. This is an I/O API change, not a logic bug in your filtering.  
Patch summary: In cell 9, keep the same mask-based water filtering logic, but load the PNG via `urllib.request.urlopen` and `PIL.Image.open`, then convert to a NumPy array before applying the existing threshold and indexing. This preserves identical semantics while restoring compatibility.  
Updated cells: Only cell 9 is modified.  
Compatibility notes for cell k+1: `clean()` still returns a filtered DataFrame with the same columns and types, so cell 10 (`train_df.describe()`) behaves unchanged.  
Assumptions: Network access is allowed to download the NYC mask image from the provided URL; Pillow is available via Matplotlib dependencies.'
- What this solution (achieved 15.40036) has done: 'Diagnosis: Cell 9 crashes inside `remove_datapoints_from_water()` because it tries to download the NYC land/water mask image from `https://aiblog.nl/...`, which returns HTTP 404 in this environment. Since `clean()` calls `remove_datapoints_from_water()` unconditionally, the entire preprocessing fails.  
Patch summary: Keep the same filtering logic when the mask can be loaded, but add a safe fallback: if downloading/loading the mask fails for any reason, return the input dataframe unchanged (skipping only the water-mask filter) so the pipeline can proceed deterministically.  
Updated cells: Only cell 9 is changed; no other cells are touched.  
Compatibility notes for cell k+1: `train_df` remains a pandas DataFrame and `train_df = clean(train_df)` still runs; cell 10 (`train_df.describe()`) works unchanged.  
Assumptions: External internet resources may be unavailable or the URL may be stale; skipping the water-mask filter is acceptable as a bug-fix fallback to unblock execution.'
- What this solution (achieved 15.16348) has done: 'Diagnosis: Cell 34 crashes because `tf_keras.optimizers` exposes the optimizer class as `Adam` (capital A) rather than a lowercase factory function `adam`, so `optimizers.adam(...)` does not exist in this environment. This is an API mismatch across Keras/TensorFlow Keras variants. The fix is to instantiate `optimizers.Adam(...)` and pass the learning rate using the supported argument name.  

Patch summary: In cell 34 only, replace `optimizers.adam(lr=...)` with `optimizers.Adam(learning_rate=...)` to match `tf_keras` API and keep the same optimizer/learning-rate semantics.  

Updated cells:'
- What this solution (achieved 43.48237) has done: 'Diagnosis: The crash happens during training because the custom `rmse` metric uses `keras.backend`, but in your environment `keras==3.8.0` maps `backend` to `keras.api.backend`, which no longer exposes legacy ops like `sqrt/mean/square`. As a result, `backend.sqrt(...)` raises `AttributeError`. The model itself is built with `tf_keras`, so the metric should also use `tf_keras.backend` (or `tf` ops) to stay compatible with `tf_keras` training.

Patch summary: In cell 34 only, replace the imported backend used by `rmse` with `tf_keras.backend` and define `rmse` locally in the same cell (so it overrides the incompatible one defined earlier). No changes to model architecture, optimizer, loss, or training loop are made.

Updated cells: cell 34 only.

Compatibility notes for cell k+1: `model` and `history` are still created exactly as before, so cell 35 (`model_to_dot(model)`) remains compatible. The metric name and behavior remain RMSE; only the backend implementation changes to prevent the crash.

Assumptions: `tf_keras` is available (it is installed and already used elsewhere in the notebook), and `tf_keras.backend` provides `sqrt/mean/square` in this environment.'
- What this solution (achieved 315.86304) has done: 'Diagnosis: Cell 35 crashes because `keras==3.8.0` no longer provides `keras.utils.vis_utils.model_to_dot` (it was part of legacy `tf.keras`). This visualization utility moved/changed across Keras/TensorFlow versions, so the import path is invalid in the current environment. The simplest fix is to use `tf_keras.utils.model_to_dot`, which matches the rest of this notebook (it already uses `tf_keras` for the model and layers).  
Patch summary: Replace the failing import with `from tf_keras.utils import model_to_dot` and keep the rest of the visualization call unchanged.  
Updated cells: Only cell 35 is modified.  
Compatibility notes for cell k+1: Cell 36 only uses `history` and `plot_loss_accuracy_rmse(history)`; both remain unchanged and available.  
Assumptions: `tf_keras==2.18.0` includes `tf_keras.utils.model_to_dot` (standard in TF-Keras), and Graphviz rendering behavior remains consistent with the prior call.'
- What this solution (achieved 828.54431) has done: 'Diagnosis: Cell 39 crashes because `keras.losses.mean_squared_error` receives tensors with incompatible shapes: `test_labels[1000:]` has length `len(test_labels)-1000` while `prediction[:1000]` has length `1000`, so subtraction cannot broadcast. The intent appears to be comparing the same slice range of ground truth and predictions.  
Patch summary: In cell 39, make both arrays use the same slice (`[:1000]`) and ensure `prediction` is 1D to match `test_labels` (since `model.predict` returns shape `(n, 1)`). This fixes the broadcasting error without changing model/training logic.  
Updated cells: Only cell 39 is changed.  
Compatibility notes for cell k+1: No variables used by cell 41 are modified; `predictionKaggle` remains unchanged and `output_submission(...)` work as before.  
Assumptions: The goal of cell 39 is to compute MSE on a small comparable subset, not specifically an offset slice mismatch.'

# 9. Code solution

## === cell 0
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
    df['distance'] = np.sqrt(np.abs(df['pickup_longitude']-df['dropoff_longitude'])**2 + np.abs(df['pickup_latitude']-df['dropoff_latitude'])**2)
    
  
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
    
    


## === cell 1
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

try:
    from google.protobuf import message_factory as _message_factory

    if hasattr(_message_factory, "MessageFactory") and not hasattr(
        _message_factory.MessageFactory, "GetPrototype"
    ):

        def _GetPrototype(self, descriptor):
            if hasattr(self, "GetMessageClass"):
                return self.GetMessageClass(descriptor)
            raise AttributeError(
                "MessageFactory has neither GetPrototype nor GetMessageClass"
            )

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


## === cell 2
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


## === cell 3
train_df, test_df = train_test_split(trainKaggle, test_size=0.50, random_state=1)


## === cell 5
print('testKaggle Size %d' % len(testKaggle))
print('train_df Size %d' % len(train_df))
print('test_df Size %d' % len(test_df))


## === cell 6
train_df.describe()


## === cell 7
test_df.describe()


## === cell 9


def remove_datapoints_from_water(df):
    def lonlat_to_xy(longitude, latitude, dx, dy, BB):
        return (dx * (longitude - BB[0]) / (BB[1] - BB[0])).astype("int"), (
            dy - dy * (latitude - BB[2]) / (BB[3] - BB[2])
        ).astype("int")

    BB = (-74.5, -72.8, 40.5, 41.8)

    try:
        from urllib.request import urlopen
        from PIL import Image

        url = "https://aiblog.nl/download/nyc_mask-74.5_-72.8_40.5_41.8.png"
        nyc_mask_img = np.array(Image.open(urlopen(url)))
        nyc_mask = nyc_mask_img[:, :, 0] > 0.9

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
    except Exception as e:
        print(
            f"Warning: could not load NYC water mask ({type(e).__name__}: {e}). Skipping water filtering."
        )
        return df


print("train_df clean")
train_df = clean(train_df)


## === cell 10
train_df.describe()


## === cell 11
print('train_df add_time_features')
train_df = add_time_features(train_df)
print('test_df add_time_features')
test_df = add_time_features(test_df)
print('testKaggle add_time_features')
testKaggle = add_time_features(testKaggle)


   


## === cell 12
train_df.describe()


## === cell 13
print('train_df add_coordinate_features')
add_coordinate_features(train_df)
print('test_df add_coordinate_features Disabled!')
add_coordinate_features(test_df)
print('testKaggle add_coordinate_features')
add_coordinate_features(testKaggle)


## === cell 14
train_df.describe()


## === cell 15
print('train_df add_distances_features')
train_df = add_distances_features(train_df)
print('test_df add_distances_features')
test_df = add_distances_features(test_df)
print('testKaggle add_distances_features')
testKaggle = add_distances_features(testKaggle)


print('Done with Adding features')


## === cell 16
train_df.describe()


## === cell 17
dropped_columns = ['passenger_count', 'pickup_datetime'] #'pickup_latitude','pickup_longitude', 'dropoff_longitude', 'dropoff_latitude']

train_df = train_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)
testKaggle_clean = testKaggle.drop(dropped_columns + ['key'], axis=1)

print('Done with dropped_columns')


## === cell 18
train_df.shape


## === cell 19
train_df.describe()


## === cell 20
test_df.describe()


## === cell 21
train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=1)


## === cell 22
train_df.describe()


## === cell 23
validation_df.describe()


## === cell 24
train_labels = train_df['fare_amount'].values
validation_labels = validation_df['fare_amount'].values
test_labels = test_df['fare_amount'].values

train_df = train_df.drop(['fare_amount'], axis=1)
validation_df = validation_df.drop(['fare_amount'], axis=1)
test_df = test_df.drop(['fare_amount'], axis=1)

print('Done with Labels')


## === cell 25
test_labels


## === cell 28
train_df.describe()


## === cell 29
validation_df.describe()


## === cell 30
test_df.describe()


## === cell 31
scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.fit_transform(validation_df)
test_scaled = scaler.transform(test_df)
testKaggle_scaled = scaler.transform(testKaggle_clean)


## === cell 32
test_scaled


## === cell 33
from keras import backend
def rmse(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


## === cell 34
from tf_keras import backend as backend


def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


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
    validation_data=(validation_df_scaled, validation_labels),
    shuffle=True,
)


## === cell 35
from IPython.display import SVG

from tf_keras.utils import model_to_dot

SVG(model_to_dot(model).create(prog="dot", format="svg"))


## === cell 36
plot_loss_accuracy_rmse(history)


## === cell 37
score = model.evaluate(test_df, test_labels, verbose=1)
print(score)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


## === cell 38
prediction = model.predict(test_scaled, batch_size=128, verbose=1)
predictionKaggle = model.predict(testKaggle_scaled, batch_size=128, verbose=1)


## === cell 39
import keras.losses  # import mean_squared_error

keras.losses.mean_squared_error(test_labels[:1000], prediction[:1000].reshape(-1))


## === cell 41

output_submission(testKaggle, predictionKaggle, 'key', 'fare_amount', SUBMISSION_NAME)


## === cell 42
print(prediction[10000])
print(test_labels[10000])
