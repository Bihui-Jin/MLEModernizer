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

4.44991

# 6. Current score

None

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 15.25422) has done: 'The crash happens before any notebook logic runs: importing `keras` triggers an `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`, which is a known incompatibility between certain `protobuf` versions and the legacy `keras` package in some environments. The minimal fix is to force Keras 3 to use the installed TensorFlow-Keras backend (`tf_keras==2.18.0`) by setting `KERAS_BACKEND=tensorflow` *before* importing anything from `keras`. This keeps the same `keras.*` API usage and model code intact while avoiding the protobuf-related crash. No other cells need changes, and the variables/constants defined in cell 0 remain identical for downstream cells.'
- What this solution (achieved 15.2493) has done: 'The crash happens immediately on importing `keras` because this environment (Python 3.7) has `keras==3.8.0`, which is not compatible here and triggers a protobuf `MessageFactory.GetPrototype` attribute error during import. The minimal deterministic fix is to use the bundled TensorFlow-compatible Keras package `tf_keras==2.18.0` that is installed, while keeping the rest of the notebook’s API and model logic unchanged. I only change the Keras import lines in cell 0 to import from `tf_keras` instead of `keras`. No other logic, parameters, paths, or training semantics are altered.'
- What this solution (achieved 15.2379) has done: 'The crash happens immediately during imports in cell 0, before any notebook logic runs, due to an incompatibility between TensorFlow/Keras-related imports and the installed `protobuf` version (`MessageFactory.GetPrototype` removal in newer protobuf). The smallest safe fix is to force the pure-Python protobuf implementation before importing anything that transitively imports protobuf, which restores the expected API surface for these older TF/Keras stacks. This change is localized to cell 0 and does not alter the model, data processing, training loop, or outputs. No other cells need to change.'
- What this solution (achieved 15.24864) has done: 'The crash happens during the imports in cell 0: `tf_keras` triggers a protobuf compatibility issue (`MessageFactory.GetPrototype` missing) with the current installed `protobuf` version. The most localized fix is to avoid importing `tf_keras` entirely and instead import the same Keras API from `tensorflow.keras`, which is compatible with the runtime and keeps the same model-building/training semantics. No model architecture, layers, optimizer choices, or training loop logic are changed—only the import source. This unblocks execution so the subsequent cells can read the CSVs and proceed as intended.'
- What this solution (achieved 15.22634) has done: 'The crash happens immediately when importing `tensorflow.keras` in cell 0, before any of your notebook logic runs. With `keras==3.8.0` and `tf_keras==2.18.0` installed, importing TensorFlow/Keras can trigger a protobuf incompatibility (`MessageFactory.GetPrototype` missing) depending on the protobuf runtime version. The minimal fix is to pin protobuf’s Python implementation and version *before* importing anything from TensorFlow, then perform the same imports as before so downstream cells remain unchanged. This keeps your model/training logic identical; it only prevents the environment-level import crash.'
- What this solution (achieved 15.22457) has done: 'Diagnosis: Cell 9 crashes inside `remove_datapoints_from_water()` because `plt.imread()` (via Pillow) no longer accepts a URL string directly; it requires a file-like object or an already-opened stream. The code attempts to read a remote PNG mask using a URL, triggering `ValueError: Please open the URL for reading...`. This is an API/IO compatibility issue, not a modeling/feature bug.

Patch summary: Modify only cell 9 by monkey-patching `plt.imread` to transparently handle URL inputs using `urllib.request.urlopen` + `PIL.Image.open`, returning a NumPy array exactly like `imread` would. This keeps the existing cleaning logic and mask semantics unchanged while restoring compatibility with the current matplotlib/Pillow behavior.

Updated cells: Only cell 9 is changed (added a small compatibility patch before calling `clean()`).

Compatibility notes for cell k+1: `train_df` and `validation_df` remain DataFrames produced by the same `clean()` function, so `cell 10` (`train_df.describe()`) continues to work unchanged.

Assumptions: Outbound HTTP access to `https://aiblog.nl/download/nyc_mask-74.5_-72.8_40.5_41.8.png` is available in the environment; if it is blocked, the original approach would still fail (separately from this API fix).'
- What this solution (achieved 15.27908) has done: 'The crash in cell 9 is caused by `remove_datapoints_from_water()` calling `plt.imread()` on an external URL that returns HTTP 404, which raises an uncaught `HTTPError` and stops execution. To keep the notebook runnable and deterministic in environments without that remote asset, we wrap the `clean()` calls in cell 9 with a targeted try/except around the water-mask cleaning step only. If the mask cannot be loaded, we fall back to skipping that single filtering step (returning the already-cleaned dataframe from earlier filters), preserving the rest of the pipeline and variables expected downstream. This patch is localized to cell 9 and does not change model/training logic.'
- What this solution (achieved 15.33593) has done: 'The crash happens because in newer Keras/TensorFlow the optimizer factory function `optimizers.adam(...)` no longer exists (it used to in older APIs), so calling it raises `AttributeError`. The minimal fix is to instantiate the Adam optimizer via the supported class `optimizers.Adam(...)` while preserving the same learning-rate value. This keeps the model architecture, compile configuration, and training loop unchanged. The rest of the cell can remain identical.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

try:
    import google.protobuf  # noqa: F401
    import subprocess, sys

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "protobuf==3.20.3"]
    )
except Exception:
    pass

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers

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
    df = df[(0.99 < df['fare_amount']) & (df['fare_amount'] <= 50)]
    
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
def distance(lat1, lon1, lat2, lon2):
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
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
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

_original_imread = plt.imread


def _imread_compat(fname, format=None):
    if isinstance(fname, str) and fname.startswith(("http://", "https://")):
        with urllib.request.urlopen(fname) as resp:
            img = Image.open(resp)
            return np.array(img)
    return _original_imread(fname, format=format)


plt.imread = _imread_compat

print("train_df clean")
try:
    train_df = clean(train_df)
except Exception as e:
    if isinstance(e, urllib.error.HTTPError):
        print(
            f"Warning: water-mask download failed ({e}). Skipping remove_datapoints_from_water for train_df."
        )
        train_df = train_df.dropna(how="any", axis="rows")
        train_df = train_df[
            (train_df["dropoff_longitude"] != train_df["pickup_longitude"])
            & (train_df["dropoff_latitude"] != train_df["pickup_latitude"])
        ]
        train_df = train_df[
            (train_df["dropoff_longitude"] != 0)
            & (train_df["pickup_longitude"] != 0)
            & (train_df["dropoff_latitude"] != 0)
            & (train_df["pickup_latitude"] != 0)
        ]
        MinMax = (-74.5, -72.8, 40.5, 41.8)
        train_df = train_df[
            (MinMax[0] <= train_df["pickup_longitude"])
            & (train_df["pickup_longitude"] <= MinMax[1])
        ]
        train_df = train_df[
            (MinMax[0] <= train_df["dropoff_longitude"])
            & (train_df["dropoff_longitude"] <= MinMax[1])
        ]
        train_df = train_df[
            (MinMax[2] <= train_df["pickup_latitude"])
            & (train_df["pickup_latitude"] <= MinMax[3])
        ]
        train_df = train_df[
            (MinMax[2] <= train_df["dropoff_latitude"])
            & (train_df["dropoff_latitude"] <= MinMax[3])
        ]
        train_df = train_df[
            (0.99 < train_df["fare_amount"]) & (train_df["fare_amount"] <= 50)
        ]
        train_df = train_df[
            (train_df["passenger_count"] > 0) & (train_df["passenger_count"] <= 6)
        ]

        nyc_coord = (40.7141667, -74.0063889)
        fk_coord = (40.639722, -73.778889)
        ewr_coord = (40.6925, -74.168611)
        lga_coord = (40.77725, -73.872611)
        sol_coord = (40.6892, -74.0445)

        train_df = train_df[
            (nyc_coord[1] != train_df["pickup_longitude"])
            & (train_df["pickup_latitude"] != nyc_coord[0])
        ]
        train_df = train_df[
            (nyc_coord[1] != train_df["dropoff_longitude"])
            & (train_df["dropoff_latitude"] != nyc_coord[0])
        ]
        train_df = train_df[
            (fk_coord[1] != train_df["pickup_longitude"])
            & (train_df["pickup_latitude"] != fk_coord[0])
        ]
        train_df = train_df[
            (fk_coord[1] != train_df["dropoff_longitude"])
            & (train_df["dropoff_latitude"] != fk_coord[0])
        ]
        train_df = train_df[
            (ewr_coord[1] != train_df["pickup_longitude"])
            & (train_df["pickup_latitude"] != ewr_coord[0])
        ]
        train_df = train_df[
            (ewr_coord[1] != train_df["dropoff_longitude"])
            & (train_df["dropoff_latitude"] != ewr_coord[0])
        ]
        train_df = train_df[
            (lga_coord[1] != train_df["pickup_longitude"])
            & (train_df["pickup_latitude"] != lga_coord[0])
        ]
        train_df = train_df[
            (lga_coord[1] != train_df["dropoff_longitude"])
            & (train_df["dropoff_latitude"] != lga_coord[0])
        ]
        train_df = train_df[
            (sol_coord[1] != train_df["pickup_longitude"])
            & (train_df["pickup_latitude"] != sol_coord[0])
        ]
        train_df = train_df[
            (sol_coord[1] != train_df["dropoff_longitude"])
            & (train_df["dropoff_latitude"] != sol_coord[0])
        ]
    else:
        raise

print("validation_df clean")
try:
    validation_df = clean(validation_df)
except Exception as e:
    if isinstance(e, urllib.error.HTTPError):
        print(
            f"Warning: water-mask download failed ({e}). Skipping remove_datapoints_from_water for validation_df."
        )
        validation_df = validation_df.dropna(how="any", axis="rows")
        validation_df = validation_df[
            (validation_df["dropoff_longitude"] != validation_df["pickup_longitude"])
            & (validation_df["dropoff_latitude"] != validation_df["pickup_latitude"])
        ]
        validation_df = validation_df[
            (validation_df["dropoff_longitude"] != 0)
            & (validation_df["pickup_longitude"] != 0)
            & (validation_df["dropoff_latitude"] != 0)
            & (validation_df["pickup_latitude"] != 0)
        ]
        MinMax = (-74.5, -72.8, 40.5, 41.8)
        validation_df = validation_df[
            (MinMax[0] <= validation_df["pickup_longitude"])
            & (validation_df["pickup_longitude"] <= MinMax[1])
        ]
        validation_df = validation_df[
            (MinMax[0] <= validation_df["dropoff_longitude"])
            & (validation_df["dropoff_longitude"] <= MinMax[1])
        ]
        validation_df = validation_df[
            (MinMax[2] <= validation_df["pickup_latitude"])
            & (validation_df["pickup_latitude"] <= MinMax[3])
        ]
        validation_df = validation_df[
            (MinMax[2] <= validation_df["dropoff_latitude"])
            & (validation_df["dropoff_latitude"] <= MinMax[3])
        ]
        validation_df = validation_df[
            (0.99 < validation_df["fare_amount"]) & (validation_df["fare_amount"] <= 50)
        ]
        validation_df = validation_df[
            (validation_df["passenger_count"] > 0)
            & (validation_df["passenger_count"] <= 6)
        ]

        nyc_coord = (40.7141667, -74.0063889)
        fk_coord = (40.639722, -73.778889)
        ewr_coord = (40.6925, -74.168611)
        lga_coord = (40.77725, -73.872611)
        sol_coord = (40.6892, -74.0445)

        validation_df = validation_df[
            (nyc_coord[1] != validation_df["pickup_longitude"])
            & (validation_df["pickup_latitude"] != nyc_coord[0])
        ]
        validation_df = validation_df[
            (nyc_coord[1] != validation_df["dropoff_longitude"])
            & (validation_df["dropoff_latitude"] != nyc_coord[0])
        ]
        validation_df = validation_df[
            (fk_coord[1] != validation_df["pickup_longitude"])
            & (validation_df["pickup_latitude"] != fk_coord[0])
        ]
        validation_df = validation_df[
            (fk_coord[1] != validation_df["dropoff_longitude"])
            & (validation_df["dropoff_latitude"] != fk_coord[0])
        ]
        validation_df = validation_df[
            (ewr_coord[1] != validation_df["pickup_longitude"])
            & (validation_df["pickup_latitude"] != ewr_coord[0])
        ]
        validation_df = validation_df[
            (ewr_coord[1] != validation_df["dropoff_longitude"])
            & (validation_df["dropoff_latitude"] != ewr_coord[0])
        ]
        validation_df = validation_df[
            (lga_coord[1] != validation_df["pickup_longitude"])
            & (validation_df["pickup_latitude"] != lga_coord[0])
        ]
        validation_df = validation_df[
            (lga_coord[1] != validation_df["dropoff_longitude"])
            & (validation_df["dropoff_latitude"] != lga_coord[0])
        ]
        validation_df = validation_df[
            (sol_coord[1] != validation_df["pickup_longitude"])
            & (validation_df["pickup_latitude"] != sol_coord[0])
        ]
        validation_df = validation_df[
            (sol_coord[1] != validation_df["dropoff_longitude"])
            & (validation_df["dropoff_latitude"] != sol_coord[0])
        ]
    else:
        raise


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
dropped_columns = ['passenger_count', 'pickup_datetime', 'pickup_latitude','pickup_longitude', 'dropoff_longitude', 'dropoff_latitude']

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
from tensorflow.keras import backend as K


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


## === cell 25
from IPython.display import SVG

try:
    from tensorflow.keras.utils import model_to_dot  # TF-Keras compatible import

    SVG(model_to_dot(model).create(prog="dot", format="svg"))
except Exception as e:
    print(f"Model visualization skipped (model_to_dot/graphviz unavailable): {e}")


## === cell 26
if "history" in globals() and hasattr(history, "history"):
    plot_loss_accuracy_rmse(history)
else:
    print(
        "Skipping plot_loss_accuracy_rmse: 'history' is not defined (model may not have been trained yet)."
    )


## === cell 29
if "model" not in globals():
    print(
        "Skipping predictions: `model` is not defined. "
        "Ensure the model is built and trained in an earlier cell before calling predict()."
    )
    prediction = None
    predictionKaggle = None
else:
    prediction = model.predict(test_scaled, batch_size=128, verbose=1)
    predictionKaggle = model.predict(testKaggle_scaled, batch_size=128, verbose=1)


## === cell 30
import keras.losses  # import mean_squared_error

if prediction is None:
    print(
        "Skipping mean_squared_error: `prediction` is None (model not defined/trained)."
    )
else:
    n = min(1000, len(test_labels), len(prediction))
    keras.losses.mean_squared_error(test_labels[:n], prediction[:n])


## === cell 32

output_submission(testKaggle, predictionKaggle, 'key', 'fare_amount', SUBMISSION_NAME)


## === cell 33
if prediction is None:
    print(
        "Skipping debug print: `prediction` is None (model not defined/trained earlier)."
    )
else:
    print(prediction[10000:] if len(prediction) > 10000 else prediction)

if test_labels is None or len(test_labels) == 0:
    print("Skipping label print: `test_labels` is missing/empty.")
else:
    print(test_labels[10000] if len(test_labels) > 10000 else test_labels[-1])
