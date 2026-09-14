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

4.02509

# 6. Current score

None

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 15.2408) has done: 'Diagnosis: The crash occurs during `import` in cell 1, before any of your notebook logic runs. With `google-api-python-client==2.177.0` installed, its dependency chain imports `google.protobuf`, and newer `protobuf` versions removed/changed `MessageFactory.GetPrototype`, causing `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` at import time. This is an environment-level protobuf incompatibility that is triggered by importing packages; your notebook doesn’t directly use the Google client libraries, but the import-time side effect still breaks execution. The smallest deterministic fix is to pin protobuf to a compatible version *before* importing anything that might touch it, by installing a known-good protobuf version at runtime.

Patch summary: In cell 1 only, install/force a protobuf version that still provides the expected API (3.20.x is widely compatible with older code paths), then proceed with the existing imports unchanged. This keeps your model/training code identical and unblocks the rest of the notebook.

Updated cells:'
- What this solution (achieved 15.28592) has done: 'Diagnosis: The crash happens inside `remove_datapoints_from_water()` when calling `plt.imread()` with an HTTPS URL. Recent Matplotlib versions no longer accept URLs directly in `imread`, raising the shown `ValueError`. The fix is to explicitly open the URL as a binary stream (via `urllib.request.urlopen`) and decode it with Pillow (`PIL.Image.open`), then convert to a NumPy array—exactly as the error message suggests. This keeps the same mask-based filtering logic and returns the same dataframe columns/types expected by the rest of the notebook.

Patch summary: In cell 7, replace the direct `plt.imread(url)` call with `np.array(PIL.Image.open(urllib.request.urlopen(url)))` and keep the rest of the function unchanged. Add minimal imports inside the function to avoid touching earlier cells and to ensure dependencies are available at runtime.

Updated cells: (only the buggy cell 7)

Compatibility notes for cell k+1: `clean()` still returns a filtered `DataFrame` with the same columns; `add_time_features()` in cell 8 work unchanged.

Assumptions: Outbound HTTP(S) access is allowed to download the NYC mask image; Pillow is available via Matplotlib’s dependency stack (and can be imported as `PIL`).'
- What this solution (achieved 15.24836) has done: 'Diagnosis: The crash happens inside `remove_datapoints_from_water()` because recent Matplotlib/Pillow no longer allow `plt.imread()` to read directly from an `https://` URL, raising a `ValueError`. This function is called by `clean()` in cell 7, so execution stops before any later feature engineering. The fix is to explicitly open the URL via `urllib.request.urlopen`, load it with `PIL.Image.open`, convert to a NumPy array, and then apply the existing mask logic unchanged.

Patch summary: Update only cell 7 by monkey-patching `remove_datapoints_from_water` (same name/signature) to fetch and decode the PNG mask via `urllib` + `PIL` and keep the rest of the computations identical. This preserves the `clean()` behavior and avoids changing earlier cells.

Updated cells: (cell 7 only)

Compatibility notes for cell k+1: `train_df` and `test_df` remain pandas DataFrames with the same columns as before, so cell 8 can call `add_time_features()` unchanged.

Assumptions: Outbound HTTP access to `https://aiblog.nl/download/nyc_mask-74.5_-72.8_40.5_41.8.png` is permitted in this environment; `Pillow` is available as a Matplotlib dependency (importable as `PIL`).'
- What this solution (achieved 15.45132) has done: 'Diagnosis: Cell 7 crashes because `remove_datapoints_from_water()` downloads an NYC land/water mask from `https://aiblog.nl/...png`, but that URL now returns HTTP 404, so `urllib.request.urlopen()` raises `HTTPError`. This external network dependency makes the notebook non-deterministic and brittle in this environment. The rest of `clean()` expects `remove_datapoints_from_water()` to return a filtered DataFrame; to preserve the pipeline and unblock execution, we should fall back to returning the unmodified input when the mask cannot be retrieved.  

Patch summary: Modify only cell 7’s `remove_datapoints_from_water()` to catch download/parse failures (HTTPError/URLError/etc.) and return `df` unchanged in that case. This keeps `clean()` semantics intact aside from skipping the water-mask filter when unavailable, and prevents the crash.  

Updated cells: Cell 7 only (as requested).  

Compatibility notes for cell k+1: No interface changes—`train_df`, `test_df`, and `testKaggle` remain DataFrames with the same columns, so cell 8 can still call `add_time_features()` on them.  

Assumptions: Internet access or the remote file is unreliable/unavailable; skipping the water-mask filter is acceptable to proceed and matches the “minimal change to fix crash” requirement.'
- What this solution (achieved 15.38586) has done: 'Diagnosis: The crash occurs in cell 32 because Keras 3 no longer exposes the optimizer constructor as `optimizers.adam`; instead it uses the class `optimizers.Adam` (and it expects `learning_rate` rather than `lr`). This leads to `AttributeError: module 'keras.api.optimizers' has no attribute 'adam'`. The rest of the model definition, compile, and fit logic can remain unchanged.

Patch summary: Update the optimizer instantiation in cell 32 to use `optimizers.Adam(learning_rate=LEARNING_RATE)` (with a small compatibility fallback to `lr` if needed). No other model/training logic is modified.

Updated cells:'
- What this solution (achieved 83.12194) has done: 'Diagnosis: The crash happens during `model.fit()` when Keras evaluates the custom `rmse` metric. In Keras 3, `keras.backend` no longer exposes legacy tensor math ops like `sqrt/mean/square`, so calling `backend.sqrt(...)` raises `AttributeError`. The metric itself is fine; it just needs to use supported ops (e.g., `keras.ops`) to remain compatible with the current Keras API.

Patch summary: In cell 32, define a local `rmse` function that uses `keras.ops` (available in Keras 3) and compile the model with this metric instead of the broken backend-based version. This keeps the model architecture/training loop unchanged and only swaps the implementation of the metric to an equivalent computation.

Updated cells: Cell 32 only (buggy cell).

Compatibility notes for cell k+1: `load_model` in cell 33 still work; the model can be loaded, and the `rmse` metric remains available in the current runtime (it’s defined in cell 32 before training). If later code reloads the model with `custom_objects`, it should reference `rmse` exactly as before.

Assumptions: Keras 3.x is being used (as per installed packages), so `keras.ops` is available and supports `sqrt`, `mean`, and `square` on tensors.'
- What this solution (achieved 177.36166) has done: 'Diagnosis: Cell 34 crashes because `keras==3.x` no longer exposes `keras.utils.vis_utils` (that path existed in older `tf.keras`). In Keras 3, `model_to_dot` moved under `keras.utils` and may also require optional `pydot`/Graphviz; we should import from the new location and avoid failing if those optional deps aren’t available. The goal of the cell is visualization only, so making it a best-effort display preserves training/inference semantics.

Patch summary: Update cell 34 to try importing `model_to_dot` from the Keras 3 location first, fall back to older locations for compatibility, and guard the SVG rendering so the notebook continues even if visualization dependencies are missing.

Updated cells: Cell 34 only.

Compatibility notes for cell k+1: `history` and `model` remain unchanged; cell 35 continues to run normally.

Assumptions: Visualization dependencies (`pydot`, Graphviz) may be absent; in that case we skip rendering rather than crashing.'
- What this solution (achieved 96.69148) has done: 'Your current RMSE (177) is far worse than the target (4.025), so we should make small changes that reliably reduce error without changing the model/training core. The biggest issues are (1) `late_night()` and `night()` logic is incorrect and effectively always true/false, and (2) you never actually create the intended time-based binary features, so the network is missing useful signals. I minimally fix the time-flag functions and add those three derived columns inside `add_time_features()` (no change to the model, loss, or training loop), and I also enforce non-negative fare predictions at submission time (fares can’t be negative), which typically improves RMSE.'
- What this solution (achieved 28.38211) has done: 'Your current public score (96.69 RMSE; lower is better) is far above the target (4.03), so we should make a couple of small, low-risk fixes that typically reduce error a lot without changing the model/training core. The biggest issue is that your coordinate features function currently does nothing; adding standard, lightweight spatial features (abs deltas and a haversine distance) keeps the same pipeline and model architecture but gives the network the key signal it’s missing. I also make datetime parsing robust (the current format string often yields NaT on this dataset, which breaks/zeros time features) while keeping the same derived columns. Finally, I clip submission predictions to a realistic fare range (0–500) to reduce RMSE impact from extreme outliers without “gaming” the metric.'

# 9. Code solution

## === cell 0
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

    print(" New size after NYC lang lot: %d" % len(df))

    df = df[(df["pickup_latitude"] != 0)]
    df = df[(df["pickup_latitude"] != 0)]
    df = df[(df["dropoff_longitude"] != 0)]
    df = df[(df["dropoff_latitude"] != 0)]

    print(" New size after lang lot > 0: %d" % len(df))

    df = df[((df["pickup_latitude"] - df["dropoff_latitude"]).abs() > 0.001)]
    df = df[((df["pickup_longitude"] - df["dropoff_longitude"]).abs() > 0.001)]

    print(" New size after lang - lot > 0.001: %d" % len(df))

    print(" New size after only NYC: %d" % len(df))
    df = df[(0 < df["fare_amount"]) & (df["fare_amount"] <= 50)]

    print(" New size after removing outliers: %d" % len(df))

    df = df[(df["passenger_count"] > 0)]
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

    return df


def remove_datapoints_from_water(df):
    def lonlat_to_xy(longitude, latitude, dx, dy, BB):
        return (dx * (longitude - BB[0]) / (BB[1] - BB[0])).astype("int"), (
            dy - dy * (latitude - BB[2]) / (BB[3] - BB[2])
        ).astype("int")

    BB = (-74.5, -72.8, 40.5, 41.8)

    nyc_mask = (
        plt.imread("https://aiblog.nl/download/nyc_mask-74.5_-72.8_40.5_41.8.png")[
            :, :, 0
        ]
        > 0.9
    )

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


def late_night(row):
    if (row["hour"] <= 3) or (row["hour"] >= 22):
        return 1
    else:
        return 0


def night(row):
    if ((row["hour"] > 20) or (row["hour"] < 6)) and (row["weekday"] < 5):
        return 1
    else:
        return 0


def rush_hour(row):
    if ((row["hour"] <= 20) and (row["hour"] >= 16)) and (row["weekday"] < 5):
        return 1
    else:
        return 0


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df):
    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"], errors="coerce", utc=True
    )
    df["pickup_datetime"] = df["pickup_datetime"].fillna(
        pd.Timestamp("2010-01-01", tz="UTC")
    )

    df["year"] = df["pickup_datetime"].dt.year
    df["month"] = df["pickup_datetime"].dt.month
    df["day"] = df["pickup_datetime"].dt.day
    df["hour"] = df["pickup_datetime"].dt.hour
    df["weekday"] = df["pickup_datetime"].dt.weekday

    df["late_night"] = df.apply(late_night, axis=1).astype("uint8")
    df["night"] = df.apply(night, axis=1).astype("uint8")
    df["rush_hour"] = df.apply(rush_hour, axis=1).astype("uint8")

    return df


def add_coordinate_features(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]

    df["abs_lat_diff"] = (lat1 - lat2).abs()
    df["abs_lon_diff"] = (lon1 - lon2).abs()
    df["haversine_miles"] = distance(lat1, lon1, lat2, lon2)

    return df


def add_distances_features(df):

    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]

    df["manhattan"] = manhattan(lat1, lon1, lat2, lon2)

    return df


def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295  # Pi/180
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))  # 2*R*asin...


def distanceP(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295  # Pi/180
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))  # 2*R*asin...


def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    prediction = np.asarray(prediction).reshape(-1)
    prediction = np.clip(prediction, 0.0, 500.0)

    df = pd.DataFrame(
        {id_column: raw_test[id_column].values, prediction_column: prediction}
    )
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
    plt.plot(history.history["rmse"])
    plt.plot(history.history["val_rmse"])
    plt.title("Model rmse")
    plt.ylabel("rmse")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper right")
    plt.show()




## === cell 1
import sys, subprocess

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "protobuf==3.20.3"]
)

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, LSTM
from keras.callbacks import EarlyStopping
from keras import optimizers
from keras import regularizers
from keras.callbacks import ModelCheckpoint


TRAIN_PATH = "/kaggle/input/train.csv"
TEST_PATH = "/kaggle/input/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"


BATCH_SIZE = 1000
EPOCHS = 100
LEARNING_RATE = 0.001
DATASET_SIZE = 800000



## === cell 2
import os

if not os.path.exists(TRAIN_PATH):
    alt_train = "/kaggle/data/train.csv"
    if os.path.exists(alt_train):
        TRAIN_PATH = alt_train
if not os.path.exists(TEST_PATH):
    alt_test = "/kaggle/data/test.csv"
    if os.path.exists(alt_test):
        TEST_PATH = alt_test

print("Using TRAIN_PATH:", TRAIN_PATH)
print("Using TEST_PATH:", TEST_PATH)



## === cell 3
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
        "key",
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
    TEST_PATH, dtype={k: v for k, v in datatypes.items() if k != "fare_amount"}
)



## === cell 4
train_df, test_df = train_test_split(trainKaggle, test_size=0.50, random_state=1)
test_df = test_df[:10000]



## === cell 5
print("testKaggle Size %d" % len(testKaggle))
print("train_df Size %d" % len(train_df))
print("test_df Size %d" % len(test_df))



## === cell 6
train_df.describe()



## === cell 7
test_df.describe()




## === cell 8
def remove_datapoints_from_water(df):
    import urllib.request
    from urllib.error import HTTPError, URLError
    from PIL import Image

    def lonlat_to_xy(longitude, latitude, dx, dy, BB):
        return (dx * (longitude - BB[0]) / (BB[1] - BB[0])).astype("int"), (
            dy - dy * (latitude - BB[2]) / (BB[3] - BB[2])
        ).astype("int")

    BB = (-74.5, -72.8, 40.5, 41.8)

    url = "https://aiblog.nl/download/nyc_mask-74.5_-72.8_40.5_41.8.png"
    try:
        with urllib.request.urlopen(url) as resp:
            nyc_img = np.array(Image.open(resp))
        nyc_mask = nyc_img[:, :, 0] > 0.9
    except (HTTPError, URLError, OSError, ValueError, IndexError):
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

    pickup_x = np.clip(pickup_x, 0, nyc_mask.shape[1] - 1)
    dropoff_x = np.clip(dropoff_x, 0, nyc_mask.shape[1] - 1)
    pickup_y = np.clip(pickup_y, 0, nyc_mask.shape[0] - 1)
    dropoff_y = np.clip(dropoff_y, 0, nyc_mask.shape[0] - 1)

    idx = nyc_mask[pickup_y, pickup_x] & nyc_mask[dropoff_y, dropoff_x]

    return df[idx]


def clean_test(df):
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
    print(" New size after NYC lang lot: %d" % len(df))

    df = df[(df["pickup_latitude"] != 0)]
    df = df[(df["pickup_latitude"] != 0)]
    df = df[(df["dropoff_longitude"] != 0)]
    df = df[(df["dropoff_latitude"] != 0)]
    print(" New size after lang lot > 0: %d" % len(df))

    df = df[((df["pickup_latitude"] - df["dropoff_latitude"]).abs() > 0.001)]
    df = df[((df["pickup_longitude"] - df["dropoff_longitude"]).abs() > 0.001)]
    print(" New size after lang - lot > 0.001: %d" % len(df))

    df = df[(df["passenger_count"] > 0)]
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


print("train_df clean")
train_df = clean(train_df)
test_df = clean(test_df)

print("testKaggle clean (no fare filter)")
testKaggle = clean_test(testKaggle)



## === cell 9
print("train_df add_time_features")
train_df = add_time_features(train_df)
print("test_df add_time_features")
test_df = add_time_features(test_df)
print("testKaggle add_time_features")
testKaggle = add_time_features(testKaggle)



## === cell 10
train_df.describe()



## === cell 11
print("train_df add_coordinate_features")
train_df = add_coordinate_features(train_df)
print("test_df add_coordinate_features Disabled!")
test_df = add_coordinate_features(test_df)
print("testKaggle add_coordinate_features")
testKaggle = add_coordinate_features(testKaggle)



## === cell 12
train_df.describe()



## === cell 13
print("train_df add_distances_features")
train_df = add_distances_features(train_df)
print("test_df add_distances_features")
test_df = add_distances_features(test_df)
print("testKaggle add_distances_features")
testKaggle = add_distances_features(testKaggle)

print("Done with Adding features")



## === cell 14
train_df.describe()



## === cell 15
plot = train_df.iloc[:2000].plot.scatter("fare_amount", "passenger_count")
plot = train_df.iloc[:2000].plot.scatter("fare_amount", "year")
plot = train_df.iloc[:2000].plot.scatter("fare_amount", "month")
plot = train_df.iloc[:2000].plot.scatter("fare_amount", "day")
plot = train_df.iloc[:2000].plot.scatter("fare_amount", "hour")
plot = train_df.iloc[:2000].plot.scatter("fare_amount", "weekday")



## === cell 16
dropped_columns = [
    "pickup_datetime"
]  # , 'pickup_latitude','pickup_longitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count' ]

train_df = train_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)

testKaggle_clean = testKaggle.drop(dropped_columns + ["key"], axis=1)

print("Done with dropped_columns")



## === cell 17
train_df.shape



## === cell 18
train_df.describe()



## === cell 19
test_df.describe()



## === cell 20
train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=1)



## === cell 21
train_df_main = train_df
validation_df_main = validation_df



## === cell 22
validation_df.describe()



## === cell 23
train_labels = train_df["fare_amount"].values
validation_labels = validation_df["fare_amount"].values
test_labels = test_df["fare_amount"].values

train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)
test_df = test_df.drop(["fare_amount"], axis=1)

print("Done with Labels")



## === cell 24
test_labels



## === cell 25
train_df.describe()



## === cell 26
validation_df.describe()



## === cell 27
test_df.describe()



## === cell 28
train_df_num = train_df.select_dtypes(include=[np.number])
validation_df_num = validation_df.select_dtypes(include=[np.number])
test_df_num = test_df.select_dtypes(include=[np.number])
testKaggle_clean_num = testKaggle_clean.select_dtypes(include=[np.number])

scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df_num)
validation_df_scaled = scaler.transform(validation_df_num)
test_scaled = scaler.transform(test_df_num)
testKaggle_scaled = scaler.transform(testKaggle_clean_num)


## === cell 29
test_scaled



## === cell 30
from keras import backend


def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))




## === cell 31
import keras
from keras import ops


def rmse(y_true, y_pred):
    return ops.sqrt(ops.mean(ops.square(y_pred - y_true), axis=-1))


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
model.add(Dense(1))

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



## === cell 32
from keras.models import load_model



## === cell 33
from IPython.display import SVG, display

try:
    from keras.utils import model_to_dot  # Keras 3+
except Exception:
    try:
        from tensorflow.keras.utils import model_to_dot  # older tf.keras fallback
    except Exception:
        model_to_dot = None

if model_to_dot is not None:
    try:
        display(SVG(model_to_dot(model).create(prog="dot", format="svg")))
    except Exception as e:
        print(
            "Model visualization skipped (missing optional dependencies or graphviz error):",
            str(e),
        )
else:
    print(
        "Model visualization skipped (model_to_dot not available in this Keras installation)."
    )



## === cell 34
plot_loss_accuracy_rmse(history)



## === cell 35
score = model.evaluate(train_df_scaled, train_labels, verbose=1)
print(score)
print("train mean_squared_error:", score[0])
print("train mae:", score[1])
print("train accuracy:", score[2])
print("train rmse:", score[3])
print("train mse:", score[4])



## === cell 36
score = model.evaluate(validation_df_scaled, validation_labels, verbose=1)
print(score)
print("Validation mean_squared_error:", score[0])
print("Validation mae:", score[1])
print("Validation accuracy:", score[2])
print("Validation rmse:", score[3])
print("Validation mse:", score[4])



## === cell 37
score = model.evaluate(test_scaled, test_labels, verbose=1)
print(score)
print("Test mean_squared_error:", score[0])
print("Test mae:", score[1])
print("Test accuracy:", score[2])
print("Test rmse:", score[3])
print("Test mse:", score[4])



## === cell 38
validation_predictions = model.predict(validation_df_scaled).flatten()

plt.scatter(validation_labels, validation_predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.axis("equal")
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot(
    [validation_predictions.min(), validation_predictions.max()],
    [validation_predictions.min(), validation_predictions.max()],
    "k--",
    lw=4,
)



## === cell 39
test_predictions = model.predict(test_scaled).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.axis("equal")
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
_ = plt.plot(
    [test_predictions.min(), test_predictions.max()],
    [test_predictions.min(), test_predictions.max()],
    "k--",
    lw=4,
)



## === cell 40
print(np.argmax(test_predictions))
print(test_predictions[np.argmax(test_predictions)])
print(test_labels[np.argmax(test_predictions)])
test_df.iloc[np.argmax(test_predictions)]



## === cell 41
print(np.argmin(test_predictions))
print(test_predictions[np.argmin(test_predictions)])
print(test_labels[np.argmin(test_predictions)])
test_df.iloc[np.argmin(test_predictions)]



## === cell 42
fig, ax = plt.subplots()
ax.scatter(test_labels, test_predictions)
ax.plot(
    [test_labels.min(), test_labels.max()],
    [test_labels.min(), test_labels.max()],
    "k--",
    lw=4,
)
ax.set_xlabel("Measured")
ax.set_ylabel("Predicted")
plt.show()



## === cell 43
plt.figure(figsize=(20, 10))
plt.plot(validation_labels[:1000])
plt.plot(validation_predictions[:1000])
plt.title("Prediction vs Actual")
plt.ylabel("Fare Amount")
plt.xlabel("Transaction")
plt.legend(["Actual", "prediction"], loc="upper right")
plt.show()

validation_df_scaled[52]
validation_df.iloc[52]



## === cell 44
plt.figure(figsize=(20, 10))
plt.plot(test_labels[:100])
plt.plot(test_predictions[:100])
plt.title("Prediction vs Actual")
plt.ylabel("Fare Amount")
plt.xlabel("Transaction")
plt.legend(["Actual", "prediction"], loc="upper right")
plt.show()



## === cell 45
error = validation_predictions - validation_labels
plt.hist(error, bins=100)
plt.xlabel("Prediction Error")
_ = plt.ylabel("Count")



## === cell 46
error = test_predictions - test_labels
plt.hist(error, bins=50)
plt.xlabel("Prediction Error")
_ = plt.ylabel("Count")



## === cell 47
print(len(error))

errorGreaterZero = error[(np.logical_or(error <= -1, error >= 1))]
print(len(errorGreaterZero))

plt.hist(errorGreaterZero, bins=100)
plt.xlabel("Prediction Error")
_ = plt.ylabel("Count")



## === cell 48
predictionKaggle = model.predict(testKaggle_scaled, batch_size=128, verbose=1)



## === cell 49
output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)
