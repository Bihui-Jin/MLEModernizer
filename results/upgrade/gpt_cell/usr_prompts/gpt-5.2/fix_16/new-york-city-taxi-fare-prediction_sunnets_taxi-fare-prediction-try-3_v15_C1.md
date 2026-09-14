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

6.82552

# 6. Current score

58.31916

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 15.19647) has done: 'Diagnosis: The notebook crashes immediately in cell 0 during `keras` imports due to an incompatibility between the standalone `keras==3.8.0` package and the environment’s protobuf runtime (`MessageFactory.GetPrototype` missing). This is a known failure mode when importing Keras 3 in some protobuf versions. Since later cells only need `Sequential`, layers, callbacks, optimizers, and regularizers, the safest minimal fix is to switch imports to the installed `tf_keras==2.18.0` backend, which avoids the protobuf/Keras 3 conflict while preserving the exact same high-level APIs used in the notebook. No model/training logic is changed—only the import source.

Patch summary: Replace `from keras...` imports with equivalent `from tf_keras...` imports inside cell 0 so the environment can import Keras successfully. Keep all constants and non-keras imports unchanged.

Updated cells: (cell 0 only)

Compatibility notes for cell k+1: Cell 1 does not depend on Keras symbols, so it is unaffected. Any later references to `Sequential`, `Dense`, `Dropout`, `BatchNormalization`, `EarlyStopping`, `optimizers`, and `regularizers` remain available with the same names and compatible behavior via `tf_keras`.

Assumptions: `tf_keras==2.18.0` is properly installed and importable in this environment (it is listed in installed packages), and using it is acceptable as a drop-in replacement for the Keras APIs used here.'
- What this solution (achieved 15.24467) has done: 'Diagnosis: The crash happens during imports in cell 0, before any of your notebook logic runs. With `keras==3.8.0` installed alongside `tf_keras==2.18.0`, importing `tf_keras` can trigger an internal protobuf compatibility issue that raises `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. This is an environment mismatch problem rather than a bug in your model code.

Patch summary: Avoid importing `tf_keras` entirely and instead use `keras` (Keras 3) for `Sequential`, layers, callbacks, optimizers, and regularizers. This keeps the same model/training semantics while removing the dependency that triggers the protobuf crash. No other logic, constants, or paths are changed.

Updated cells: Only cell 0 is modified as required.

Compatibility notes for cell k+1: Cell 1 does not reference any of the changed imports; all variables defined in cell 0 (paths and hyperparameters) remain unchanged and be available exactly as before.

Assumptions: `keras==3.8.0` is functional in this environment and provides the same APIs used here (`Sequential`, `Dense`, `Dropout`, `BatchNormalization`, `EarlyStopping`, `optimizers`, `regularizers`).'
- What this solution (achieved 15.22013) has done: 'Diagnosis: The crash happens during the Keras import in cell 0, before any model code runs. With `keras==3.8.0` on Python 3.7, the standalone `keras` package pulls in dependencies (via protobuf) that are incompatible in this environment, triggering `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. The simplest deterministic fix is to use the already-installed `tf_keras==2.18.0` (TensorFlow Keras) API instead of standalone Keras 3, keeping the same model-building interfaces (Sequential, Dense, Dropout, BatchNormalization, EarlyStopping, optimizers, regularizers). No changes are needed to data paths or downstream variable names.

Patch summary: Replace `keras.*` imports with `tf_keras.*` imports in cell 0 only, preserving the same imported symbols so later cells continue to work unchanged.

Updated cells: cell 0 only (imports adjusted).

Compatibility notes for cell k+1: Cell 1 is unaffected; it relies only on pandas/numpy and the path constants defined in cell 0, which remain unchanged. All Keras symbols used later (Sequential, layers, callbacks, optimizers, regularizers) remain available under the same names.

Assumptions: `tf_keras` is functional in this environment (it is installed as `tf_keras==2.18.0`), and later code uses only the imported Keras symbols rather than directly depending on Keras 3-specific features.'
- What this solution (achieved 15.26717) has done: 'Diagnosis: The crash happens immediately on importing `tf_keras` in cell 0, before any model code runs. This `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` is a known incompatibility between certain TensorFlow/Keras protobuf expectations and the environment’s installed protobuf runtime; it’s triggered during import. Since you already have standalone `keras==3.8.0` installed, switching imports from `tf_keras` to `keras` avoids the incompatible dependency chain while keeping the same Sequential/Dense-style API and training semantics.

Patch summary: Modify only cell 0 to import `Sequential`, layers, callbacks, optimizers, and regularizers from `keras` instead of `tf_keras`. No model logic, hyperparameters, paths, or downstream variable names are changed.

Updated cells: (cell 0 only)

Compatibility notes for cell k+1: Cell 1 only relies on constants like `TRAIN_PATH`, `TEST_PATH`, `DATASET_SIZE` defined in cell 0; these remain unchanged. No variables used by later cells are renamed.

Assumptions: The notebook uses standard Keras APIs (`Sequential`, `Dense`, etc.) compatible with `keras==3.8.0`, and no later cell depends specifically on `tf_keras`-only behavior.'
- What this solution (achieved 15.29582) has done: 'Diagnosis: The crash happens during `keras` imports in cell 0, before any data is read. With the installed stack (keras==3.8.0 + tf_keras==2.18.0 on Python 3.7), importing the standalone `keras` package can trigger an incompatibility in the protobuf stack, raising `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. The notebook’s code uses classic `tf.keras` APIs (Sequential, Dense, callbacks, optimizers), so switching imports to `tf_keras` (already installed and compatible with this environment) preserves the same model/training semantics while avoiding the failing standalone-keras import path.  

Patch summary: Modify only cell 0 by replacing `keras.*` imports with `tf_keras.*` equivalents; keep all constants, paths, and downstream interfaces unchanged so cell 1 and later cells can run identically.  

Updated cells / Compatibility notes for cell k+1 / Assumptions:
- Updated only cell 0.  
- Cell 1 is unaffected because it only relies on variables defined in cell 0 (paths and constants), which remain unchanged.  
- Assumption: `tf_keras==2.18.0` is usable in this environment and provides the imported symbols (`Sequential`, `Dense`, `Dropout`, `BatchNormalization`, `EarlyStopping`, `optimizers`, `regularizers`).'
- What this solution (achieved 15.2698) has done: 'Diagnosis: The crash happens during the imports in cell 0, before any of your notebook logic runs. With `tf_keras==2.18.0` on Python 3.7, TensorFlow/protobuf initialization can fail with `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`, which is a known protobuf API incompatibility. The core model code doesn’t need protobuf directly; it only needs `tf_keras` to import cleanly. The minimal fix is to force protobuf to use the pure-Python implementation before importing `tf_keras`, avoiding the incompatible compiled path.

Patch summary: In cell 0, set the environment variable `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (and `..._VERSION=2` for stability) **before** importing `tf_keras`. No changes to model architecture, training logic, paths, or later variables.

Updated cells: only cell 0 is modified.

Compatibility notes for cell k+1: All names and constants (`TRAIN_PATH`, `TEST_PATH`, `SUBMISSION_NAME`, `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE`, `DATASET_SIZE`) remain identical, so cell 1 continues to run unchanged.

Assumptions: The environment already has `protobuf` installed and honors `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` at import time (standard behavior); using the Python protobuf implementation is acceptable for this notebook since protobuf is only needed for framework initialization.'
- What this solution (achieved 15.24209) has done: 'Diagnosis: The crash happens during imports in cell 0, before any notebook logic runs. Setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` forces the pure-Python protobuf backend, which is incompatible with the installed protobuf runtime and leads to `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` during TensorFlow/Keras initialization. This is an environment/API compatibility issue, not a modeling/data issue. Fix is to stop forcing the Python protobuf implementation so TensorFlow can use the compatible (C++) backend.

Patch summary: Remove the two `os.environ.setdefault(...)` lines that pin protobuf to the pure-Python implementation. Keep all other imports, constants, and training logic unchanged.

Updated cells: Only cell 0 is modified.

Compatibility notes for cell k+1: All variables/constants (`TRAIN_PATH`, `TEST_PATH`, `SUBMISSION_NAME`, etc.) and imports remain defined exactly as before, so cell 1 run unchanged.

Assumptions: TensorFlow (via `tf_keras`) can load successfully with the default protobuf backend available in the environment.'
- What this solution (achieved 15.12621) has done: 'Diagnosis: The crash happens in cell 0 during the `tf_keras` import chain, before any of your notebook logic runs. With Python 3.7 and `tf_keras==2.18.0`, TensorFlow’s protobuf stubs can trigger `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` when an incompatible protobuf runtime is picked up. Since your code only needs Keras-style APIs (`Sequential`, `Dense`, `Dropout`, etc.), the smallest safe fix is to switch those imports from `tf_keras` to the already-installed standalone `keras` package (Keras 3.8.0), which avoids the TensorFlow/protobuf path that is failing. This keeps the model architecture/training semantics intact and unblocks execution.

Patch summary: Replace `tf_keras` imports with equivalent `keras` imports inside cell 0 only; leave all constants and downstream variables unchanged.

Updated cells: cell 0 only.

Compatibility notes for cell k+1: No variables used by later cells are renamed or removed; `Sequential`, layer classes, callbacks, optimizers, and regularizers remain available under the same imported names, so model-building code in later cells continue to work.

Assumptions: The environment’s `keras==3.8.0` is functional and backend-configured (as provided), and later cells do not rely on TensorFlow-specific objects beyond the Keras API imported here.'
- What this solution (achieved 15.25423) has done: 'Diagnosis: The crash happens immediately during `keras` imports in cell 0, before any data is read. With `keras==3.8.0` on Python 3.7, importing the standalone `keras` package can trigger a protobuf-related incompatibility that surfaces as `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. This environment also has `tf_keras==2.18.0`, which provides the same Keras API but avoids the failing standalone-keras/protobuf path.  

Patch summary: In cell 0 only, switch all Keras imports from `keras` to `tf_keras` while keeping the same classes/functions (Sequential, Dense/Dropout/BatchNormalization, EarlyStopping, optimizers, regularizers). No model logic, hyperparameters, paths, or downstream variable names are changed—only the import source to prevent the import-time crash.  

Updated cells: cell 0 only.  

Compatibility notes for cell k+1: Cell 1 does not depend on Keras symbols; it only reads CSVs. All variables defined in cell 0 (paths and constants) remain unchanged and available for cell 1.  

Assumptions: `tf_keras` is correctly installed and importable in the runtime (it is listed in installed packages).'
- What this solution (achieved 15.22182) has done: 'Diagnosis: The notebook crashes immediately on importing `tf_keras` in cell 0 due to an incompatibility between `tf_keras==2.18.0` and the installed protobuf runtime, which triggers `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` during import-time initialization. This prevents any subsequent cells from running. The smallest safe fix is to force protobuf to use the pure-Python implementation before importing `tf_keras`, which avoids the C++-backed API surface that causes this error in such mismatched environments. This change is localized to cell 0 and preserves the model/training logic unchanged.

Patch summary: In cell 0, set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (and version) before importing `tf_keras`, keeping all other imports/variables intact.

Updated cells: Only cell 0 is modified.

Compatibility notes for cell k+1: All names imported from `tf_keras` (`Sequential`, `Dense`, `Dropout`, `BatchNormalization`, `EarlyStopping`, `optimizers`, `regularizers`) remain available with the same interfaces, so later cells work unchanged.

Assumptions: The environment allows setting `os.environ` before importing `tf_keras` and that using the pure-Python protobuf implementation is acceptable for this notebook’s runtime needs.'
- What this solution (achieved 15.22848) has done: 'Diagnosis: The crash occurs during imports in cell 0 and matches a known protobuf runtime incompatibility where some library expects `google.protobuf.message_factory.MessageFactory.GetPrototype`, but newer protobuf versions removed/changed it. Your current environment variables forcing the pure-Python protobuf implementation don’t prevent the mismatch. The minimal fix is to patch `MessageFactory` by providing a `GetPrototype` alias to the newer `GetMessageClass` when missing, before importing TensorFlow/Keras (`tf_keras`) which triggers protobuf usage.

Patch summary: In cell 0 only, add a small compatibility shim that imports `google.protobuf.message_factory` and defines `MessageFactory.GetPrototype` if absent. Keep all existing imports, constants, and core logic unchanged so downstream cells and training behavior remain the same.

Updated cells (cell 0 only):  
Compatibility notes for cell k+1: No variable names, paths, or interfaces are changed; `TRAIN_PATH`, `TEST_PATH`, and all imports used by cell 1 remain available.  
Assumptions: `google.protobuf` is installed (it must be, given the traceback) and provides `MessageFactory.GetMessageClass` in this environment.'
- What this solution (achieved 15.23095) has done: 'Diagnosis: The crash happens inside `remove_datapoints_from_water()` (called by `clean()` in cell 10) because `matplotlib.pyplot.imread()` no longer accepts a URL string directly in this environment; it requires a file-like object/bytes. This raises a `ValueError` before cleaning finishes, so none of the later feature-engineering steps run.  
Patch summary: In cell 10 only, monkey-patch `plt.imread` locally to support URL inputs by fetching the bytes via `urllib.request` and decoding them with `PIL.Image`, returning a NumPy array as expected. This preserves the existing cleaning logic (same URL, same mask usage) while unblocking execution deterministically.  
Updated cells: Only cell 10 is changed.  
Compatibility notes for cell k+1: All variables (`train_df`, `validation_df`, `test_df`, `testKaggle`) keep the same names and types; `clean()` and later feature functions behave the same aside from fixing URL image loading.  
Assumptions: Pillow (PIL) is available as an indirect Matplotlib dependency in this environment; network access to the mask URL is allowed (same as the original intent).'
- What this solution (achieved 15.16478) has done: 'Diagnosis: Cell 10 crashes inside `clean()` because `remove_datapoints_from_water()` calls `plt.imread()` on an external URL (`https://aiblog.nl/...png`) that returns HTTP 404 in this environment. The URL-dependent mask download is not required for the rest of the pipeline to run, but the unhandled exception stops execution.  
Patch summary: In cell 10 only, wrap the URL image fetch inside `_imread_with_url_support()` with a try/except so that on any download/read failure it returns a locally-generated “all land” boolean mask image instead of raising. This preserves the existing cleaning logic and deterministically keeps all points when the mask cannot be fetched.  
Updated cells: Only cell 10 is modified.  
Compatibility notes for cell k+1: All variables (`train_df`, `validation_df`, `test_df`, `testKaggle`) are still produced with the same columns/types expected by cell 11; `plt.imread` remains overridden and compatible with later calls.  
Assumptions: Network access and/or the external mask file is unavailable or moved; falling back to “no water filtering” is acceptable to unblock execution while preserving core semantics as closely as possible.'
- What this solution (achieved 291.41262) has done: 'Diagnosis: Cell 19 crashes because `tf_keras.optimizers` in this environment (tf_keras==2.18.0) does not expose a lowercase factory `optimizers.adam`; instead it provides the `Adam` optimizer class. The rest of the model/training code is fine, but optimizer construction must use the correct API.  
Patch summary: In cell 19 only, replace `optimizers.adam(lr=LEARNING_RATE)` with `optimizers.Adam(learning_rate=LEARNING_RATE)` to match tf_keras’ optimizer API and argument naming. This preserves identical training semantics (Adam with the same learning rate) and keeps the `adam` variable used in `model.compile`.  
Updated cells: Only cell 19 is changed below.  
Compatibility notes for cell k+1: `history` remains created by `model.fit(...)`, so cell 20 (`plot_loss_accuracy(history)`) continues to work unchanged.  
Assumptions: `tf_keras.optimizers.Adam` is available (standard in tf_keras 2.x) and `LEARNING_RATE` is a float.'
- What this solution (achieved 58.31916) has done: 'Diagnosis: The crash happens because this environment uses Keras 3, where `keras.backend` no longer exposes legacy tensor math ops like `sqrt/mean/square`. Your `rmse()` function in cell 18 uses those backend functions, but in cell 25 you call `rmse(test_labels, prediction)` with NumPy arrays, so using NumPy is both sufficient and compatible.  
Patch summary: Update only cell 25 to compute RMSE via NumPy (same metric semantics) and return a plain float, avoiding `keras.backend` entirely and keeping `metric` defined for cell 26.  
Updated cells: Only cell 25 is changed.  
Compatibility notes for cell k+1: Cell 26 expects `metric` to exist and be printable; it now be a Python `float`, so `print(metric)` works.  
Assumptions: `test_labels` and `prediction` are numeric arrays with compatible shapes (e.g., `(n,)` and `(n,1)`), and RMSE should be computed over all samples.'

# 9. Code solution

## === cell 0
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
from tf_keras.layers import Dense, Dropout, BatchNormalization
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
train_df, test_df = train_test_split(trainKaggle, test_size=0.20, random_state=1)


## === cell 3
testKaggle.head()


## === cell 4
train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=1)


## === cell 5
print('testKaggle Size %d' % len(testKaggle))
print('train_df Size %d' % len(train_df))
print('validation_df Size %d' % len(validation_df))
print('test_df Size %d' % len(test_df))


## === cell 6
test_df.head()


## === cell 7
validation_df.head()


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
    
    
def plot_loss_accuracy(history):
    
    plt.figure(figsize=(20,10))
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()
    
    plt.figure(figsize=(20,10))
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('val_loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()
    
    


## === cell 9
validation_df.head()


## === cell 10

import urllib.request
from PIL import Image

_original_imread = plt.imread


def _imread_with_url_support(fname, format=None):
    if isinstance(fname, str) and fname.startswith(("http://", "https://")):
        try:
            with urllib.request.urlopen(fname) as resp:
                img = Image.open(resp)
                return np.array(img)
        except Exception:
            return np.ones((1024, 1024, 3), dtype=np.float32) * 255.0
    return _original_imread(fname, format=format)


plt.imread = _imread_with_url_support


print("train_df clean")
train_df = clean(train_df)
print("validation_df clean")
validation_df = clean(validation_df)

train_df.describe()

print("train_df add_time_features")
train_df = add_time_features(train_df)
print("validation_df add_time_features")
validation_df = add_time_features(validation_df)
print("test_df add_time_features")
test_df = add_time_features(test_df)
print("testKaggle add_time_features")
testKaggle = add_time_features(testKaggle)


## === cell 11
print('train_df add_coordinate_features')
add_coordinate_features(train_df)
print('validation_df add_coordinate_features')
add_coordinate_features(validation_df)
print('test_df add_coordinate_features Disabled!')
add_coordinate_features(test_df)
print('testKaggle add_coordinate_features')
add_coordinate_features(testKaggle)


## === cell 12
print('train_df add_distances_features')
train_df = add_distances_features(train_df)
print('validation_df add_distances_features')
validation_df = add_distances_features(validation_df)
print('test_df add_distances_features')
test_df = add_distances_features(test_df)
print('testKaggle add_distances_features')
testKaggle = add_distances_features(testKaggle)


print('Done with Adding features')


## === cell 13
dropped_columns = ['passenger_count', 'pickup_datetime']

train_df = train_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)
validation_df = validation_df.drop(dropped_columns, axis=1)
testKaggle_clean = testKaggle.drop(dropped_columns + ['key'], axis=1)

print('Done with dropped_columns')


## === cell 14
train_labels = train_df['fare_amount'].values
validation_labels = validation_df['fare_amount'].values
test_labels = test_df['fare_amount'].values

train_df = train_df.drop(['fare_amount'], axis=1)
validation_df = validation_df.drop(['fare_amount'], axis=1)
test_df = test_df.drop(['fare_amount'], axis=1)

print('Done with Labels')


## === cell 15
train_df.shape


## === cell 16
test_df.shape


## === cell 17
validation_df.shape


## === cell 18
scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_df)
testKaggle_scaled = scaler.transform(testKaggle_clean)



import keras.backend as K
def rmse (y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


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
model.add(Dense(8, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(1))

adam = optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(loss="mean_squared_error", optimizer=adam, metrics=["mae"])

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


## === cell 20
plot_loss_accuracy(history)


## === cell 21
prediction = model.predict(test_scaled, batch_size=128, verbose=1)


## === cell 22
test_labels.shape


## === cell 24
prediction = model.predict(test_scaled, batch_size=128, verbose=1)
predictionKaggle = model.predict(testKaggle_scaled, batch_size=128, verbose=1)


## === cell 25
y_true = np.asarray(test_labels).reshape(-1)
y_pred = np.asarray(prediction).reshape(-1)
metric = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


## === cell 26
print(metric)


## === cell 27

output_submission(testKaggle, predictionKaggle, 'key', 'fare_amount', SUBMISSION_NAME)


## === cell 28
print(predictionKaggle)
