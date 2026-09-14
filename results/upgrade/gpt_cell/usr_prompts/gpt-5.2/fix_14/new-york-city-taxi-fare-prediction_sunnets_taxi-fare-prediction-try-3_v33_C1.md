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

4.64984

# 6. Current score

None

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 15.24452) has done: 'Diagnosis: The crash happens immediately in cell 0 during `from keras...` imports, before any of your notebook logic runs. In this environment, `keras==3.8.0` pulls in dependencies that trigger a `protobuf` incompatibility (`MessageFactory.GetPrototype` missing), which is a known failure mode when standalone Keras and the protobuf stack are mismatched. Since you also have `tf_keras==2.18.0` installed (TensorFlow’s bundled Keras), the minimal, deterministic fix is to import from `tf_keras` instead of `keras` to avoid the problematic standalone Keras/protobuf path. This keeps the same API surface (Sequential, layers, callbacks, optimizers, regularizers) and preserves the model/training semantics.

Patch summary: In cell 0 only, switch Keras imports from `keras.*` to `tf_keras.*`. No other logic, constants, paths, or downstream variable names are changed.

Updated cells: cell 0 only (buggy cell).

Compatibility notes for cell k+1: Cell 1 uses only `pd.read_csv` and variables defined in cell 0 (TRAIN_PATH/TEST_PATH/DATASET_SIZE). Those names remain unchanged; only the Keras import source changes, so cell 1 remains fully compatible.

Assumptions: `tf_keras==2.18.0` is functional in this runtime and provides the same symbols used in the notebook (`Sequential`, `Dense`, `Dropout`, `BatchNormalization`, `LSTM`, `EarlyStopping`, `optimizers`, `regularizers`).'
- What this solution (achieved 15.26889) has done: 'The crash happens during `tf_keras` import in cell 0: `google.protobuf` recent versions removed `MessageFactory.GetPrototype`, which older TensorFlow/Keras internals still try to call. To unblock execution without changing your modeling/training logic, we can monkey-patch `google.protobuf.message_factory.MessageFactory.GetPrototype` to point to `GetMessageClass` (or a safe fallback) before importing `tf_keras`. This keeps the same APIs and lets the existing `tf_keras` imports succeed. No changes are needed to later cells; variables like `TRAIN_PATH`, `BATCH_SIZE`, etc. remain identical.'
- What this solution (achieved 15.26587) has done: 'Diagnosis: The crash happens inside `remove_datapoints_from_water()` when calling `plt.imread()` with an HTTPS URL. In current Matplotlib versions, `imread()` no longer accepts a URL string directly and raises the shown `ValueError`, so the cleaning step fails before returning `train_df`/`validation_df`.

Patch summary: Update `remove_datapoints_from_water()` to fetch the PNG via `urllib.request.urlopen`, decode it with `PIL.Image.open`, convert to a NumPy array, and then apply the same boolean-mask logic as before. This preserves the exact filtering semantics while making the image load compatible with the installed Matplotlib.

Updated cells: cell 9 only.

Compatibility notes for cell k+1: `train_df` and `validation_df` are still reassigned to the cleaned DataFrames with the same columns and types, so cell 10 (`train_df.describe()`) work unchanged.

Assumptions: Network access to `https://aiblog.nl/download/nyc_mask-74.5_-72.8_40.5_41.8.png` is available (as the original code intended); Pillow (PIL) is available in the environment via Matplotlib dependencies.'
- What this solution (achieved 15.11119) has done: 'The crash happens because `remove_datapoints_from_water()` in cell 9 tries to download an external PNG mask from `https://aiblog.nl/...`, which returns HTTP 404 in this environment. To keep the pipeline running without changing downstream interfaces, the minimal fix is to make this function robust to download failures: if fetching/decoding the mask fails for any reason, it simply return the input dataframe unchanged. This preserves the same function signature and keeps `clean()` working, while avoiding the hard dependency on external network resources. No other logic, variables, or call sites are modified.'
- What this solution (achieved 15.30328) has done: 'Diagnosis: The crash occurs in cell 24 because `tf_keras.optimizers` (TF-Keras 2.18) does not expose a lowercase factory `optimizers.adam`; instead it provides the class `optimizers.Adam`. Additionally, recent TF-Keras uses the argument name `learning_rate` rather than the deprecated `lr`, so the current call fails before training starts.  
Patch summary: In cell 24 only, replace `optimizers.adam(lr=...)` with `optimizers.Adam(learning_rate=...)` while keeping the same optimizer type and learning rate, preserving all model/training semantics.  
Updated cells: Only cell 24 is modified below.  
Compatibility notes for cell k+1: Variables `model` and `history` are still created exactly as before, so cell 25 (model visualization) remains compatible.  
Assumptions: `tf_keras.optimizers.Adam` is available in the installed `tf_keras==2.18.0` and accepts `learning_rate`.'
- What this solution (achieved 149.5789) has done: 'Diagnosis: The crash happens during training when computing the custom `rmse` metric: `from keras import backend` resolves to Keras 3’s `keras.api.backend`, which no longer exposes math ops like `sqrt/mean/square`. Since the model itself is built with `tf_keras`, the metric should use the `tf_keras` backend (or TensorFlow ops) to remain compatible. The rest of the training call and model compilation are fine; only the backend import for the metric is incompatible.

Patch summary: In cell 24, add a local override that imports the backend from `tf_keras` and (re)defines `rmse` to use that backend, ensuring `sqrt/mean/square` exist. This keeps the same RMSE computation and preserves compile/fit semantics without changing model architecture or training loop.

Updated cells: Only cell 24 is modified.

Compatibility notes for cell k+1: The variable `model` and `history` are still created the same way; cell 25 can still visualize `model`. No interface/shape changes.

Assumptions: `tf_keras` is available (it is listed as installed) and its backend exposes `sqrt/mean/square` as in tf.keras.'
- What this solution (achieved 110.14101) has done: 'Diagnosis: Cell 25 crashes because `keras==3.8.0` no longer provides `keras.utils.vis_utils.model_to_dot`, so importing it raises `ModuleNotFoundError`. The intent of the cell is only to visualize the model; this should not block the rest of the notebook. The minimal safe fix is to use the currently supported `keras.utils.model_to_dot` import path, and gracefully skip visualization if optional Graphviz/pydot dependencies are unavailable.

Patch summary: Update the import in cell 25 to prefer `keras.utils.model_to_dot` with a backward-compatible fallback, and wrap the rendering in a try/except so training results in later cells remain usable even if visualization dependencies are missing.

Updated cells: Only cell 25 is modified.

Compatibility notes for cell k+1: Cell 26 only uses `history` and `plot_loss_accuracy_rmse(history)`; this patch does not change `history` or `model`, and it ensures cell 25 not raise, so cell 26 execute normally.

Assumptions: Visualization is optional for the pipeline; if Graphviz/pydot is not installed, skipping the SVG output is acceptable to keep execution unblocked.'
- What this solution (achieved 32.59455) has done: 'Diagnosis: Cell 30 crashes because `keras.losses.mean_squared_error(y_true, y_pred)` requires `y_true` and `y_pred` to have broadcastable shapes, but the code slices them inconsistently: `test_labels[1000:]` (length N-1000) vs `prediction[:1000]` (length 1000). This produces a TensorFlow `InvalidArgumentError` during subtraction inside the MSE. The fix is to slice the same range from both arrays (and ensure they are compatible ranks/dtypes).  

Patch summary: Update cell 30 to compute MSE on the same subset of examples from `test_labels` and `prediction` (first 1000), converting them to tensors with matching shapes (flattening to 1-D) to avoid rank/broadcast issues in Keras 3. No other cells are changed.  

Updated cells:'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

try:
    from google.protobuf import message_factory as _message_factory

    if hasattr(_message_factory, "MessageFactory") and not hasattr(
        _message_factory.MessageFactory, "GetPrototype"
    ):

        def _GetPrototype(self, descriptor):
            if hasattr(self, "GetMessageClass"):
                return self.GetMessageClass(descriptor)
            if hasattr(_message_factory, "GetMessageClass"):
                return _message_factory.GetMessageClass(descriptor)
            raise AttributeError(
                "No compatible protobuf MessageFactory method found for GetPrototype."
            )

        _message_factory.MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass

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

train_usecols = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
trainKaggle = pd.read_csv(
    TRAIN_PATH, nrows=DATASET_SIZE, dtype=datatypes, usecols=train_usecols
)

testKaggle = pd.read_csv(
    TEST_PATH,
    dtype=datatypes,
    usecols=[
        "key",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
)



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
        df = df[(0 < df["fare_amount"]) & (df["fare_amount"] <= 50)]
        print(" New size after removing outliers: %d" % len(df))

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
    if (row["hour"] <= 3) or (row["hour"] >= 23):
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
        df["pickup_datetime"], errors="coerce", utc=False
    )
    df["year"] = df["pickup_datetime"].dt.year
    df["month"] = df["pickup_datetime"].dt.month
    df["day"] = df["pickup_datetime"].dt.day
    df["hour"] = df["pickup_datetime"].dt.hour
    df["weekday"] = df["pickup_datetime"].dt.weekday
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


def distanceP(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295  # Pi/180
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))  # 2*R*asin...


def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    preds = np.asarray(prediction).reshape(-1)
    out = pd.DataFrame(
        {id_column: raw_test[id_column].values, prediction_column: preds}
    )
    out.to_csv(file_name, index=False)
    print("Output complete:", file_name, "rows=", len(out))


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
        with urlopen(url) as f:
            nyc_mask_img = np.array(Image.open(f))
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
    except Exception:
        return df


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
print("test_df add_coordinate_features Disabled!")
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
dropped_columns = [
    "passenger_count",
    "pickup_datetime",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_longitude",
    "dropoff_latitude",
]

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

train_X = train_df.select_dtypes(include=[np.number])
validation_X = validation_df.select_dtypes(include=[np.number])
test_X = test_df.select_dtypes(include=[np.number])
testKaggle_X = testKaggle_clean.select_dtypes(include=[np.number])

train_df_scaled = scaler.fit_transform(train_X)
validation_df_scaled = scaler.transform(validation_X)
test_scaled = scaler.transform(test_X)
testKaggle_scaled = scaler.transform(testKaggle_X)



## === cell 23
from tf_keras import backend as backend


def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))




## === cell 24
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
    shuffle=False,
)



## === cell 25
from IPython.display import SVG

try:
    try:
        from keras.utils import model_to_dot  # Keras 3+
    except Exception:
        from keras.utils.vis_utils import model_to_dot  # Older Keras fallback

    SVG(model_to_dot(model).create(prog="dot", format="svg"))
except Exception as e:
    print(f"Skipping model visualization (model_to_dot/graphviz not available): {e}")



## === cell 26
plot_loss_accuracy_rmse(history)



## === cell 27
prediction = model.predict(test_scaled, batch_size=128, verbose=1)
predictionKaggle = model.predict(testKaggle_scaled, batch_size=128, verbose=1)



## === cell 28
import keras.losses
import tensorflow as tf

y_true = tf.convert_to_tensor(test_labels[:1000], dtype=tf.float32)
y_pred = tf.convert_to_tensor(prediction[:1000], dtype=tf.float32)

y_true = tf.reshape(y_true, (-1,))
y_pred = tf.reshape(y_pred, (-1,))

keras.losses.mean_squared_error(y_true, y_pred)



## === cell 29
output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)



## === cell 30
idx = min(10000, len(prediction) - 1, len(test_labels) - 1)
print(prediction[idx])
print(test_labels[idx])
