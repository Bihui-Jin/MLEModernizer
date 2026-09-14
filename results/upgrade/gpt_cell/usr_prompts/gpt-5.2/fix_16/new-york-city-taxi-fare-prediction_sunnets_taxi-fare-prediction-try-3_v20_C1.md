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

7.29085

# 6. Current score

None

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 15.2163) has done: 'Diagnosis: The crash happens in cell 0 during `from keras...` imports, before any dataset code runs. With the installed stack (keras==3.8.0 and tf_keras==2.18.0 on Python 3.7), importing standalone `keras` can trigger an internal protobuf incompatibility that surfaces as `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. The notebook’s code is written in the TensorFlow-Keras style and does not require Keras 3-specific APIs, so switching imports to `tf_keras` avoids the protobuf/keras import path that is crashing. This is a minimal change limited to cell 0 and preserves the model/training logic used in later cells.

Patch summary: Replace `keras` imports with `tf_keras` equivalents (models/layers/callbacks/optimizers/regularizers), keeping all names (`Sequential`, `Dense`, etc.) unchanged so downstream cells remain compatible. No other logic, constants, or paths are modified.'
- What this solution (achieved 15.21995) has done: 'The crash happens immediately in cell 0 during imports, before any model/data logic runs, so the issue is an environment/library compatibility problem rather than your code. With `tf_keras==2.18.0` on Python 3.7, the import can trigger a protobuf API incompatibility (`MessageFactory.GetPrototype` missing) depending on the protobuf version present. To unblock execution while keeping the same Keras API and model semantics, switch the imports from `tf_keras` to the standalone `keras` package already installed (`keras==3.8.0`), and map the same classes (Sequential/Dense/etc.) from `keras.*`. No other logic is changed; constants and paths remain identical.'
- What this solution (achieved 15.4395) has done: 'The crash happens immediately on importing `keras` in cell 0, before any of your notebook logic runs. With your environment (`keras==3.8.0` plus `tf_keras==2.18.0`), standalone Keras 3 may trigger a protobuf compatibility issue (`MessageFactory.GetPrototype`) during import in some setups. The smallest safe fix is to switch these imports to use `tf_keras` (TensorFlow’s bundled Keras API) which matches the installed `tf_keras==2.18.0` and avoids the failing import path, while keeping the same model-building semantics (Sequential/Dense/Dropout/BatchNorm/EarlyStopping/etc.). No other cells need changes because downstream code only relies on these imported symbols.'
- What this solution (achieved 15.22165) has done: 'Diagnosis: The crash happens during the imports in cell 0, before any of your notebook logic runs. The error `'MessageFactory' object has no attribute 'GetPrototype'` is a known incompatibility between TensorFlow/Keras-related imports and newer `protobuf` versions where `MessageFactory.GetPrototype` was removed/changed. Since your code imports `tf_keras` (which pulls in TensorFlow/protobuf), we can fix the environment-side protobuf implementation selection by forcing the pure-Python protobuf runtime before importing `tf_keras`, which restores compatibility without changing your model/training logic.

Patch summary: In cell 0 only, set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (and version flag) via `os.environ` *before* importing `tf_keras`. This is a minimal, deterministic workaround that prevents the protobuf API mismatch during import and does not alter your downstream code, model architecture, or training semantics.

Updated cells: (cell 0 only)

Compatibility notes for cell k+1: All variables and imports used in cell 1 (`np`, `pd`, constants like `TRAIN_PATH`, `DATASET_SIZE`, etc.) remain defined exactly as before. No interfaces, names, or paths are changed; only an environment variable is set prior to importing `tf_keras`.

Assumptions: The failure is triggered by protobuf runtime incompatibility during `tf_keras`/TensorFlow import, and the pure-Python protobuf implementation is available in this environment (standard in protobuf installations).'
- What this solution (achieved 15.29977) has done: 'The crash happens before any model/data code runs: importing `tf_keras` triggers a protobuf incompatibility where `MessageFactory.GetPrototype` is missing. This is a known issue with newer `protobuf` versions and some TensorFlow/Keras builds. The minimal, deterministic fix is to force protobuf to use the Python implementation **and** pin it to the compatible API version **before** importing anything that may load protobuf (like `tf_keras`). I only change cell 0 to set the correct protobuf environment variable so the rest of the notebook can run unchanged.'
- What this solution (achieved 15.19175) has done: 'The crash happens immediately on importing `tf_keras` because the environment’s `google.protobuf` version is incompatible with TensorFlow/Keras 2.18’s expected protobuf API (`MessageFactory.GetPrototype`). Setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` does not avoid this mismatch. The minimal deterministic fix is to pin protobuf to the compatible implementation by forcing the pure-Python protobuf and also disabling C++ fast parsing via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION`, then import `google.protobuf` once before importing `tf_keras` so the correct backend is chosen. No model/training logic is changed; only import-time environment setup is adjusted so the notebook can run.'
- What this solution (achieved 15.2029) has done: 'The crash happens before any model code runs: importing `google.protobuf` triggers an incompatibility between the bundled `protobuf` runtime and code expecting `MessageFactory.GetPrototype`, which no longer exists in newer protobuf versions. This notebook doesn’t actually need `google.protobuf`; it was only imported to “activate” the environment variables, but that import is what crashes. The minimal fix is to remove the `google.protobuf` import and keep the environment-variable lines so downstream `tf_keras` imports still see them. No model/training logic is changed, and all variables defined in cell 0 remain defined with the same names and values.'
- What this solution (achieved 15.20873) has done: 'The crash is happening during imports in cell 0 due to a protobuf runtime incompatibility: some Google/protobuf code being imported ends up calling `MessageFactory.GetPrototype`, which no longer exists in newer protobuf versions. The simplest deterministic fix is to force the pure-Python protobuf implementation before any protobuf-dependent imports, and (for safety) provide a tiny compatibility shim that restores `GetPrototype` by delegating to the modern `GetMessageClass` when needed. This keeps the rest of your notebook unchanged and avoids touching model/training logic. The patch is localized to cell 0 only and preserves all existing variable names used by later cells.'
- What this solution (achieved 15.22515) has done: 'Diagnosis: Cell 9 crashes inside `remove_datapoints_from_water()` because `plt.imread()` in current Matplotlib/Pillow no longer accepts a URL string directly, raising a `ValueError`. The function expects an image array, but it is trying to load the NYC land/water mask from an HTTPS URL. This is an I/O compatibility issue, not a modeling/feature logic issue. Fix is to open the URL as a file-like object via `urllib.request.urlopen`, then decode with `PIL.Image.open`, convert to a NumPy array, and proceed exactly as before.

Patch summary: Modify only cell 9 to monkey-patch `remove_datapoints_from_water()` (used by `clean()`) to load the mask image via `urllib` + `PIL`, keeping the mask computation and filtering logic unchanged. No changes to data paths, cleaning thresholds, or downstream interfaces.

Updated cells: Only cell 9.

Compatibility notes for cell k+1: `train_df` and `validation_df` remain pandas DataFrames with the same columns as produced by `clean()`, so `train_df.describe()` in cell 10 works unchanged.

Assumptions: Network access to `https://aiblog.nl/download/nyc_mask-74.5_-72.8_40.5_41.8.png` is available in the runtime; if not, this step would still fail due to inability to download the mask.'
- What this solution (achieved 15.28128) has done: 'Diagnosis: Cell 9 crashes because it attempts to download a NYC land/water mask from `https://aiblog.nl/...png`, but the URL now returns HTTP 404, so `urllib.request.urlopen()` raises `HTTPError`. This is an external dependency and not deterministic/available in the execution environment. The simplest fix is to make `remove_datapoints_from_water()` robust: try to load the mask, but if it fails (404/no internet/etc.), fall back to returning the dataframe unchanged so `clean()` can proceed. This preserves the rest of the cleaning logic and keeps downstream variables (`train_df`, `validation_df`) available.

Patch summary: In cell 9 only, wrap the mask download in a try/except and return `df` unchanged on failure. No changes to the model/training logic or to other cleaning steps.

Updated cells: cell 9 only.

Compatibility notes for cell k+1: `train_df` and `validation_df` remain pandas DataFrames as before, so `train_df.describe()` in cell 10 continues to work unchanged.

Assumptions: Network access and/or the external URL may be unavailable; skipping the water-mask filtering is acceptable to unblock execution while keeping the rest of the pipeline intact.'
- What this solution (achieved 15.26877) has done: 'Diagnosis: The crash happens in cell 24 when constructing the optimizer via `optimizers.adam(...)`. In `tf_keras==2.18.0`, optimizers are exposed as classes like `optimizers.Adam` (capital A), and the `lr` argument has been standardized to `learning_rate`. Therefore `tf_keras.optimizers` has no lowercase attribute `adam`, causing the `AttributeError`.

Patch summary: Update only the optimizer construction line in cell 24 to use `optimizers.Adam(learning_rate=LEARNING_RATE)`. This preserves the same optimizer type, learning rate value, and training semantics while matching the installed `tf_keras` API.

Updated cells: (cell 24 only)

Compatibility notes for cell k+1: `model` remains a compiled `Sequential` model and `history` remains the output of `model.fit(...)`, so visualization in cell 25 (`model_to_dot(model)`) stays compatible.

Assumptions: `tf_keras.optimizers.Adam` is available in the provided environment (it is in tf_keras 2.18), and the rest of the model/metrics compile unchanged.'
- What this solution (achieved 270.50497) has done: 'Diagnosis: The crash happens during training because `rmse()` uses `from keras import backend`, which resolves to Keras 3’s backend (`keras.api.backend`). In Keras 3, backend no longer exposes math ops like `sqrt/mean/square`, so `backend.sqrt(...)` raises `AttributeError`. Since the model itself is built with `tf_keras`, the safest minimal fix is to implement `rmse` using TensorFlow math ops (or `tf_keras.backend`) so it works inside the training graph.

Patch summary: Modify only cell 24 to (1) import TensorFlow and (2) redefine `rmse` locally using `tf.math` ops, overriding the earlier incompatible `rmse` definition. Keep the model architecture, compile settings, and training call unchanged.

Updated cells: (cell 24 only)

Compatibility notes for cell k+1: `model` and `history` remain the same objects; the next cell’s visualization code still works unchanged.

Assumptions: TensorFlow is available in the environment as a dependency of `tf_keras` (standard in this setup).'
- What this solution (achieved 5.58735) has done: 'Diagnosis: Cell 25 crashes because in Keras 3 the legacy module path `keras.utils.vis_utils` was removed/relocated, so importing `model_to_dot` from there raises `ModuleNotFoundError`. This is an API-compatibility issue, not a modeling/training problem.  
Patch summary: Update cell 25 to import `model_to_dot` from the supported Keras 3 location (`keras.utils`) and fall back to `tf_keras.utils` if needed, keeping the same visualization call and outputs.  
Updated cells: Only cell 25 is changed.  
Compatibility notes for cell k+1: No variables used by cell 26 are modified; `history` remains unchanged and plotting work the same.  
Assumptions: `pydot`/Graphviz may or may not be installed; this patch only fixes the import crash (any later Graphviz runtime errors are outside the reported traceback).'
- What this solution (achieved 153.62872) has done: 'The crash happens because `plot_loss_accuracy_rmse()` tries to read `history.history['acc']` and `history.history['val_acc']`, but in your current Keras/tf_keras versions the metric is logged under `'accuracy'` and `'val_accuracy'` instead. Since the failing cell is cell 26 (the call site), the minimal fix is to patch cell 26 to temporarily alias the missing keys in `history.history` before calling the plotting function. This keeps the plotting logic and training semantics unchanged while preventing the KeyError. No other variables used by later cells are modified.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")

try:
    import google.protobuf.message_factory as _mf

    if hasattr(_mf, "MessageFactory") and not hasattr(
        _mf.MessageFactory, "GetPrototype"
    ):

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        _mf.MessageFactory.GetPrototype = _GetPrototype
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
    usecols=[1, 2, 3, 4, 5, 6, 7],
)
testKaggle = pd.read_csv(TEST_PATH)



## === cell 2
train_df, test_df = train_test_split(trainKaggle, test_size=0.50, random_state=1)



## === cell 3
train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=1)



## === cell 4
print("testKaggle Size %d" % len(testKaggle))
print("train_df Size %d" % len(train_df))
print("validation_df Size %d" % len(validation_df))
print("test_df Size %d" % len(test_df))



## === cell 5
train_df.describe()



## === cell 6
validation_df.describe()



## === cell 7
test_df.describe()




## === cell 8
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
        df = df[(0.99 < df["fare_amount"]) & (df["fare_amount"] <= 50)]
        print(" New size after removing outliers: %d" % len(df))
    else:
        print(" Skipping fare_amount outlier removal (fare_amount not present)")

    df = df[(df["passenger_count"] > 0) & (df["passenger_count"] <= 6)]
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
    return 1 if ((row["hour"] >= 20) and (row["weekday"] < 5)) else 0


def rush_hour(row):
    return (
        1
        if ((row["hour"] <= 20) and (row["hour"] >= 16) and (row["weekday"] < 5))
        else 0
    )


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df):
    dt = pd.to_datetime(
        df["pickup_datetime"], errors="coerce", infer_datetime_format=True, utc=False
    )
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["hour"] = dt.dt.hour
    df["weekday"] = dt.dt.weekday
    df["pickup_datetime"] = df["pickup_datetime"].astype(str)
    df["night"] = df.apply(lambda x: night(x), axis=1)
    df["late_night"] = df.apply(lambda x: late_night(x), axis=1)
    df["rush_hour"] = df.apply(lambda x: rush_hour(x), axis=1)
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
    df["distance"] = np.sqrt(
        np.abs(df["pickup_longitude"] - df["dropoff_longitude"]) ** 2
        + np.abs(df["pickup_latitude"] - df["dropoff_latitude"]) ** 2
    )
    return df


def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295  # Pi/180
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))  # 2*R*asin...


def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295  # Pi/180
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))  # 2*R*asin...


def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    df = pd.DataFrame(prediction, columns=[prediction_column])
    df[id_column] = raw_test[id_column]
    df[[id_column, prediction_column]].to_csv((file_name), index=False)
    print("Output complete")


def plot_loss_accuracy_rmse(history):

    plt.figure(figsize=(20, 10))
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper right")
    plt.show()

    plt.figure(figsize=(20, 10))
    plt.plot(history.history["acc"])
    plt.plot(history.history["val_acc"])
    plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper right")
    plt.show()

    plt.figure(figsize=(20, 10))
    plt.plot(history.history["rmse"])
    plt.plot(history.history["val_rmse"])
    plt.title("Model rmse")
    plt.ylabel("rmse")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper right")
    plt.show()




## === cell 9
import urllib.request
from PIL import Image


def remove_datapoints_from_water(df):
    def lonlat_to_xy(longitude, latitude, dx, dy, BB):
        return (dx * (longitude - BB[0]) / (BB[1] - BB[0])).astype("int"), (
            dy - dy * (latitude - BB[2]) / (BB[3] - BB[2])
        ).astype("int")

    BB = (-74.5, -72.8, 40.5, 41.8)

    url = "https://aiblog.nl/download/nyc_mask-74.5_-72.8_40.5_41.8.png"
    try:
        with urllib.request.urlopen(url) as resp:
            nyc_mask_img = np.array(Image.open(resp))
        nyc_mask = nyc_mask_img[:, :, 0] > 0.9
    except Exception as e:
        print(
            f"Warning: could not load NYC water mask from {url} ({type(e).__name__}: {e}). "
            f"Skipping water-mask filtering."
        )
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
print("validation_df clean")
validation_df = clean(validation_df)
print("test_df clean")
test_df = clean(test_df)
print("testKaggle clean")
testKaggle = clean(testKaggle)



## === cell 10
train_df.describe()



## === cell 11
validation_df.describe()



## === cell 12
print("train_df add_time_features")
train_df = add_time_features(train_df)
print("validation_df add_time_features")
validation_df = add_time_features(validation_df)
print("test_df add_time_features")
test_df = add_time_features(test_df)
print("testKaggle add_time_features")
testKaggle = add_time_features(testKaggle)



## === cell 13
print("train_df add_coordinate_features")
add_coordinate_features(train_df)
print("validation_df add_coordinate_features")
add_coordinate_features(validation_df)
print("test_df add_coordinate_features")
add_coordinate_features(test_df)
print("testKaggle add_coordinate_features")
add_coordinate_features(testKaggle)



## === cell 14
print("train_df add_distances_features")
train_df = add_distances_features(train_df)
print("validation_df add_distances_features")
validation_df = add_distances_features(validation_df)
print("test_df add_distances_features")
test_df = add_distances_features(test_df)
print("testKaggle add_distances_features")
testKaggle = add_distances_features(testKaggle)

print("Done with Adding features")



## === cell 15
train_df.describe()



## === cell 16
validation_df.describe()



## === cell 17
dropped_columns = ["pickup_datetime"]

train_df = train_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)
validation_df = validation_df.drop(dropped_columns, axis=1)
testKaggle_clean = testKaggle.drop(dropped_columns + ["key"], axis=1)

print("Done with dropped_columns")



## === cell 18
train_labels = train_df["fare_amount"].values
validation_labels = validation_df["fare_amount"].values
test_labels = test_df["fare_amount"].values

train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)
test_df = test_df.drop(["fare_amount"], axis=1)

print("Done with Labels")



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
    from keras.utils import model_to_dot
except Exception:
    from tf_keras.utils import model_to_dot

SVG(model_to_dot(model).create(prog="dot", format="svg"))



## === cell 26
if "acc" not in history.history and "accuracy" in history.history:
    history.history["acc"] = history.history["accuracy"]
if "val_acc" not in history.history and "val_accuracy" in history.history:
    history.history["val_acc"] = history.history["val_accuracy"]

plot_loss_accuracy_rmse(history)



## === cell 27
prediction = model.predict(test_scaled, batch_size=128, verbose=1)
predictionKaggle = model.predict(testKaggle_scaled, batch_size=128, verbose=1)



## === cell 28
import keras.losses  # import mean_squared_error



## === cell 29
output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)



## === cell 30
print(predictionKaggle)
