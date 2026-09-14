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

4.1558

# 6. Current score

95.01954

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 15.21357) has done: 'Diagnosis: The crash happens immediately when importing `keras` (cell 1). With Python 3.7 and `keras==3.8.0`, the standalone Keras package pulls in dependencies that expect a newer protobuf API, leading to `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. This environment also has `tf_keras==2.18.0`, which is the TensorFlow-compatible Keras implementation and avoids this protobuf incompatibility for typical Sequential/Dense usage.  
Patch summary: In cell 1 only, switch all Keras imports from `keras.*` to `tf_keras.*` while keeping the same classes/functions and training semantics. No model/training logic is changed; only the import source is adjusted to prevent the protobuf crash.  
Updated cells: Only cell 1 is modified as below.  
Compatibility notes for cell k+1: Cell 2 only uses `pd.read_csv` and the constants defined in cell 1; those remain unchanged. The imported symbols (`Sequential`, `Dense`, `Dropout`, etc.) keep the same names, so downstream model code continue to work.  
Assumptions: `tf_keras==2.18.0` is functional in this runtime and provides the same public APIs used later in the notebook (Sequential, layers, callbacks, optimizers, regularizers).'
- What this solution (achieved 15.21925) has done: 'Diagnosis: The crash happens during the imports in cell 1, before any dataset/model code runs. The `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` is a known incompatibility between `google/protobuf` runtime versions and libraries that indirectly rely on protobuf-generated APIs (often triggered during `tf_keras` / TensorFlow-related imports). Since the failure occurs at import time, the minimal fix is to force protobuf to use the pure-Python implementation before importing `tf_keras`, which avoids the C++ MessageFactory path that triggers this missing method.

Patch summary: In cell 1 only, set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (and version) via `os.environ` *before* importing `tf_keras`. This is a localized import-time compatibility workaround and does not change model architecture, training logic, or data processing.

Updated cells: (cell 1 only)

Compatibility notes for cell k+1: All variables defined in cell 1 (`TRAIN_PATH`, `TEST_PATH`, `SUBMISSION_NAME`, `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE`, `DATASET_SIZE`) are preserved unchanged, so cell 2 can read the CSVs exactly as before.

Assumptions: The environment’s protobuf/TensorFlow stack is incompatible in the default (C++/upb) protobuf mode, and switching to the Python protobuf implementation is sufficient to unblock `tf_keras` imports without affecting downstream semantics beyond negligible performance differences during import/serialization.'
- What this solution (achieved 15.32269) has done: 'Diagnosis: The crash happens during imports in cell 1: `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. This is a known incompatibility between some TensorFlow/Keras wheels and newer `protobuf` versions, and it can occur before any model code runs. Setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` alone is not sufficient here because `protobuf>=5` removed `MessageFactory.GetPrototype`, which some TF components still reference. The minimal fix is to monkey‑patch `google.protobuf.message_factory.MessageFactory.GetPrototype` to call `GetMessageClass`, and to do it before importing `tf_keras`.

Patch summary: In cell 1 only, add a small protobuf compatibility patch (no changes to the modeling logic) right after the environment-variable setup and before importing `tf_keras`. This restores the missing method and unblocks the imports deterministically.

Updated cells: cell 1 only (below).

Compatibility notes for cell k+1: All variables and imports defined in cell 1 remain unchanged (`np`, `pd`, `TRAIN_PATH`, etc.), so cell 2 run as-is and continue reading the same CSV paths.

Assumptions: `google.protobuf` is installed in the environment (it is, via `google-api-python-client` dependency chain), and the failure is triggered by a protobuf API change rather than missing files or other import issues.'
- What this solution (achieved 15.20938) has done: 'Diagnosis: The crash happens inside `remove_datapoints_from_water()` when calling `plt.imread()` with an `https://...` URL. With the installed Matplotlib/Pillow versions, `imread` no longer accepts URLs directly and raises a `ValueError`, so `clean()` fails before returning filtered dataframes.  
Patch summary: In cell 7 only, replace the URL-based `plt.imread(url)` with the supported approach: open the URL via `urllib.request.urlopen`, load it with `PIL.Image.open`, then convert to a NumPy array (keeping the exact same mask logic and thresholds). Add minimal imports locally inside the function to avoid changing other cells.  
Updated cells: Only cell 7 is modified.  
Compatibility notes for cell k+1: `clean()` still returns a filtered dataframe with the same columns; downstream calls to `add_time_features()` in cell 8 remain unchanged and compatible.  
Assumptions: Outbound HTTP access to `https://aiblog.nl/...png` is available in this environment (same dependency as the original code), and Pillow is available via Matplotlib’s dependencies.'
- What this solution (achieved 15.41971) has done: 'Diagnosis: Cell 7 crashes inside `remove_datapoints_from_water()` because it downloads the NYC land/water mask from `https://aiblog.nl/...png`, and that URL now returns HTTP 404. Since `clean()` always calls `remove_datapoints_from_water()`, this prevents any further preprocessing.  
Patch summary: In cell 7 only, make `remove_datapoints_from_water()` robust to network/404 failures by catching the exception and returning the input dataframe unchanged when the mask cannot be fetched/parsed. This preserves the existing cleaning pipeline and avoids introducing new logic beyond a safe fallback.  
Updated cells: Cell 7 updated below.  
Compatibility notes for cell k+1: `train_df` and `test_df` remain pandas DataFrames with the same columns, so cell 8 (`add_time_features`) continues to work unchanged.  
Assumptions: Internet access may be blocked or the external URL may be unavailable; in that case, skipping the water-mask filtering is acceptable to allow the notebook to run.'
- What this solution (achieved 15.20251) has done: 'Diagnosis: The crash occurs because `tf_keras.optimizers` in TF/Keras 2.18 does not expose a lowercase factory `optimizers.adam`; the correct API is the `Adam` class (or `legacy.Adam`). This leads to `AttributeError` when creating the optimizer. The rest of the model definition and training code is fine, so we only need to change optimizer construction to the supported call signature while keeping the same learning rate semantics.

Patch summary: In cell 33, replace `optimizers.adam(lr=...)` with `optimizers.Adam(learning_rate=...)` (and keep a fallback for older arg names), leaving all model architecture, compile settings, and training loop unchanged.

Updated cells: (cell 33 only)

Compatibility notes for cell k+1: `model` and `history` are still created the same way, and checkpointing still writes `my_model.h5`, so `load_model` in cell 34 remains compatible.

Assumptions: `tf_keras.optimizers.Adam` is available in this environment (TF-Keras 2.18.0), and using `learning_rate` is the correct replacement for deprecated `lr` with negligible numeric differences.'
- What this solution (achieved 94.82469) has done: 'Diagnosis: The crash happens during `model.fit()` when Keras calls the custom `rmse` metric. In this environment, `from keras import backend` resolves to Keras 3’s `keras.api.backend`, which no longer exposes legacy math ops like `sqrt/mean/square`, causing `AttributeError: ... backend ... has no attribute 'sqrt'`. Since the model itself is built with `tf_keras`, the safest minimal fix is to define `rmse` using TensorFlow ops (available through `tf_keras`) rather than `keras.backend`.  

Patch summary: In the failing cell, replace the `keras.backend` dependency with TensorFlow math ops by importing `tensorflow as tf` and rewriting `rmse()` to use `tf.sqrt`, `tf.reduce_mean`, and `tf.square`. This preserves the same RMSE definition and keeps the metric callable compatible with `model.compile(...)`.  

Updated cells:  

Compatibility notes for cell k+1: Cell 34 (`from keras.models import load_model`) is untouched; the fix only affects the custom metric used during training/compilation in cell 33. The variable name `rmse` remains defined with the same signature `(y_true, y_pred)`, so `model.compile(..., metrics=[..., rmse, ...])` continues to work unchanged.  

Assumptions: TensorFlow is available in the runtime (it is, as `tf_keras` is installed and functioning), and using TensorFlow ops inside a `tf_keras` training graph is supported (standard behavior).'
- What this solution (achieved 95.01954) has done: 'Diagnosis: Cell 35 crashes because `keras==3.8.0` no longer provides `keras.utils.vis_utils` (it was removed/moved in Keras 3), so `from keras.utils.vis_utils import model_to_dot` raises `ModuleNotFoundError`. The intent of the cell is purely visualization; it should not block the rest of the notebook. We switch to the supported Keras 3 import path (`keras.utils.model_to_dot`) and add a small fallback that skips visualization cleanly if optional Graphviz/pydot dependencies are missing.

Patch summary: Update the import in cell 35 to use `from keras.utils import model_to_dot` (Keras 3 compatible) and wrap rendering in a `try/except` to avoid crashing when Graphviz/pydot are unavailable.

Updated cells: Only cell 35 is modified.

Compatibility notes for cell k+1: Cell 36 uses `history` and plotting only; this patch does not change `model` or `history`, and cell 35 no longer raise, so execution proceeds normally.

Assumptions: Visualization is optional; if Graphviz/pydot is not installed in the environment, it is acceptable to skip the SVG rendering rather than error out.'

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
       

    print(' New size after NYC lang lot: %d' % len(df))         
    
    df = df[(df['pickup_latitude'] != 0)]
    df = df[(df['pickup_latitude'] != 0)]
    df = df[(df['dropoff_longitude']!= 0)]
    df = df[(df['dropoff_latitude'] != 0)]
    
    print(' New size after lang lot > 0: %d' % len(df))         

    df = df[((df['pickup_latitude'] - df['dropoff_latitude']).abs() > 0.001)]
    df = df[((df['pickup_longitude'] - df['dropoff_longitude']).abs() > 0.001)]
    
    print(' New size after lang - lot > 0.001: %d' % len(df))         
    
    print(' New size after only NYC: %d' % len(df)) 
    df = df[(0 < df['fare_amount']) & (df['fare_amount'] <= 50)]
    
    print(' New size after removing outliers: %d' % len(df)) 
    
    df = df[(df['passenger_count'] > 0)]
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
    df['minute'] = df['pickup_datetime'].apply(lambda x: x.minute)
    df['second'] = df['pickup_datetime'].apply(lambda x: x.second)
    
    return df


def add_coordinate_features(df):
    lat1 = df['pickup_latitude']
    lat2 = df['dropoff_latitude']
    lon1 = df['pickup_longitude']
    lon2 = df['dropoff_longitude']
    
    
    
    return df


def add_distances_features(df):
    
    lat1 = df['pickup_latitude']
    lat2 = df['dropoff_latitude']
    lon1 = df['pickup_longitude']
    lon2 = df['dropoff_longitude']
    
    df['manhattan'] = manhattan(lat1, lon1, lat2, lon2)
    
  
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

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

try:
    from google.protobuf import message_factory as _message_factory

    if hasattr(_message_factory, "MessageFactory") and not hasattr(
        _message_factory.MessageFactory, "GetPrototype"
    ):

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

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
from tf_keras.callbacks import ModelCheckpoint


TRAIN_PATH = "../input/train.csv"
TEST_PATH = "../input/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"


BATCH_SIZE = 1000
EPOCHS = 100
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
test_df = test_df[:10000]


## === cell 4
print('testKaggle Size %d' % len(testKaggle))
print('train_df Size %d' % len(train_df))
print('test_df Size %d' % len(test_df))


## === cell 5
train_df.describe()


## === cell 6
test_df.describe()


## === cell 7


def remove_datapoints_from_water(df):
    from urllib.request import urlopen
    from PIL import Image

    def lonlat_to_xy(longitude, latitude, dx, dy, BB):
        return (dx * (longitude - BB[0]) / (BB[1] - BB[0])).astype("int"), (
            dy - dy * (latitude - BB[2]) / (BB[3] - BB[2])
        ).astype("int")

    BB = (-74.5, -72.8, 40.5, 41.8)

    url = "https://aiblog.nl/download/nyc_mask-74.5_-72.8_40.5_41.8.png"
    try:
        with urlopen(url) as resp:
            nyc_mask_img = np.array(Image.open(resp))
        nyc_mask = nyc_mask_img[:, :, 0] > 0.9
    except Exception as e:
        print(f"Warning: could not load NYC water mask ({e}). Skipping water filter.")
        return df

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


print("train_df clean")
train_df = clean(train_df)
test_df = clean(test_df)


## === cell 8
print('train_df add_time_features')
train_df = add_time_features(train_df)
print('test_df add_time_features')
test_df = add_time_features(test_df)
print('testKaggle add_time_features')
testKaggle = add_time_features(testKaggle)


   


## === cell 9
train_df.describe()


## === cell 10
print('train_df add_coordinate_features')
add_coordinate_features(train_df)
print('test_df add_coordinate_features Disabled!')
add_coordinate_features(test_df)
print('testKaggle add_coordinate_features')
add_coordinate_features(testKaggle)


## === cell 11
train_df.describe()


## === cell 12
print('train_df add_distances_features')
train_df = add_distances_features(train_df)
print('test_df add_distances_features')
test_df = add_distances_features(test_df)
print('testKaggle add_distances_features')
testKaggle = add_distances_features(testKaggle)


print('Done with Adding features')


## === cell 13
train_df.describe()


## === cell 14
plot = train_df.iloc[:2000].plot.scatter('fare_amount', 'passenger_count')
plot = train_df.iloc[:2000].plot.scatter('fare_amount', 'year')
plot = train_df.iloc[:2000].plot.scatter('fare_amount', 'month')
plot = train_df.iloc[:2000].plot.scatter('fare_amount', 'day')
plot = train_df.iloc[:2000].plot.scatter('fare_amount', 'hour')


## === cell 15
dropped_columns = ['pickup_datetime']#, 'pickup_latitude','pickup_longitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count' ]

train_df = train_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)

testKaggle_clean = testKaggle.drop(dropped_columns + ['key'], axis=1)

print('Done with dropped_columns')


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
train_labels = train_df['fare_amount'].values
validation_labels = validation_df['fare_amount'].values
test_labels = test_df['fare_amount'].values

train_df = train_df.drop(['fare_amount'], axis=1)
validation_df = validation_df.drop(['fare_amount'], axis=1)
test_df = test_df.drop(['fare_amount'], axis=1)

print('Done with Labels')


## === cell 23
test_labels


## === cell 26
train_df.describe()


## === cell 27
validation_df.describe()


## === cell 28
test_df.describe()


## === cell 30
scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_df)
testKaggle_scaled = scaler.transform(testKaggle_clean)


## === cell 31
test_scaled


## === cell 32
from keras import backend
def rmse(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


## === cell 33
import tensorflow as tf


def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true), axis=-1))


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
model.add(Dense(1, activation="linear"))

try:
    adam = optimizers.Adam(learning_rate=LEARNING_RATE)
except TypeError:
    adam = optimizers.Adam(lr=LEARNING_RATE)

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


## === cell 34
from keras.models import load_model


## === cell 35
from IPython.display import SVG

try:
    from keras.utils import model_to_dot

    SVG(model_to_dot(model).create(prog="dot", format="svg"))
except Exception as e:
    print(f"Warning: could not render model graph ({e}). Skipping visualization.")


## === cell 36
plot_loss_accuracy_rmse(history)


## === cell 37
score = model.evaluate(train_df_scaled, train_labels, verbose=1)
print(score)
print('train mean_squared_error:', score[0])
print('train mae:', score[1])
print('train accuracy:', score[2])
print('train rmse:', score[3])
print('train mse:', score[4])


## === cell 38
score = model.evaluate(validation_df_scaled, validation_labels, verbose=1)
print(score)
print('Validation mean_squared_error:', score[0])
print('Validation mae:', score[1])
print('Validation accuracy:', score[2])
print('Validation rmse:', score[3])
print('Validation mse:', score[4])


## === cell 39
score = model.evaluate(test_scaled, test_labels, verbose=1)
print(score)
print('Test mean_squared_error:', score[0])
print('Test mae:', score[1])
print('Test accuracy:', score[2])
print('Test rmse:', score[3])
print('Test mse:', score[4])


## === cell 41
validation_predictions = model.predict(validation_df_scaled).flatten()

plt.scatter(validation_labels, validation_predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot([validation_predictions.min(), validation_predictions.max()], [validation_predictions.min(), validation_predictions.max()], 'k--', lw=4)


## === cell 42
test_predictions = model.predict(test_scaled).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot([test_predictions.min(), test_predictions.max()], [test_predictions.min(), test_predictions.max()], 'k--', lw=4)


## === cell 44
print(np.argmax(test_predictions))
print(test_predictions[np.argmax(test_predictions)])
print(test_labels[np.argmax(test_predictions)])
test_df.iloc[np.argmax(test_predictions)]


## === cell 45
print(np.argmin(test_predictions))
print(test_predictions[np.argmin(test_predictions)])
print(test_labels[np.argmin(test_predictions)])
test_df.iloc[np.argmin(test_predictions)]


## === cell 46
fig, ax = plt.subplots()
ax.scatter(test_labels, test_predictions)
ax.plot([test_labels.min(), test_labels.max()], [test_labels.min(), test_labels.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()


## === cell 47
plt.figure(figsize=(20,10))
plt.plot(validation_labels[:100])
plt.plot(validation_predictions[:100])
plt.title('Prediction vs Actual')
plt.ylabel('Fare Amount')
plt.xlabel('Transaction')
plt.legend(['Actual', 'prediction'], loc='upper right')
plt.show()



## === cell 48
plt.figure(figsize=(20,10))
plt.plot(test_labels[:100])
plt.plot(test_predictions[:100])
plt.title('Prediction vs Actual')
plt.ylabel('Fare Amount')
plt.xlabel('Transaction')
plt.legend(['Actual', 'prediction'], loc='upper right')
plt.show()



## === cell 49
error = validation_predictions - validation_labels
plt.hist(error, bins = 100)
plt.xlabel("Prediction Error")
_ = plt.ylabel("Count")


## === cell 50
error = test_predictions - test_labels
plt.hist(error, bins = 50)
plt.xlabel("Prediction Error")
_ = plt.ylabel("Count")


## === cell 51
print(len(error))

errorGreaterZero = error[ (np.logical_or(error<=-1, error>=1))]
print(len(errorGreaterZero))    

plt.hist(errorGreaterZero, bins = 100)
plt.xlabel("Prediction Error")
_ = plt.ylabel("Count")


## === cell 53
predictionKaggle = model.predict(testKaggle_scaled, batch_size=128, verbose=1)


## === cell 54

output_submission(testKaggle, predictionKaggle, 'key', 'fare_amount', SUBMISSION_NAME)
