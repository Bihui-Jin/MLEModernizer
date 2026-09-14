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

12.78702

# 6. Current score

48.97437

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 2021.11148) has done: 'I fix the early import/runtime crash by switching from `tf_keras` to the standard `tensorflow.keras` API that is stable on Kaggle, while keeping the same model architecture and training loop. I also fix the feature-mismatch bug by preserving `key` in `test` during processing (so it doesn’t get dropped) and by ensuring `fare_amount` is not accidentally included in test features passed to the scaler. Finally, I make train/test feature columns align deterministically (same columns, same order) before scaling and prediction, so `test_scaled` is created successfully and a valid `submission.csv` is written.'
- What this solution (achieved 120.11285) has done: 'I fix the import/runtime crash in the first cell that prevents the notebook from running by avoiding TensorFlow (which is triggering the protobuf `MessageFactory.GetPrototype` error in this environment) and switching to the already-installed `tf_keras` backend. Then I correct a major feature-engineering logic bug that makes test-time binned features inconsistent with training (using `qcut` separately on train and test), which is a key reason the score is extremely bad; I fit bin edges on train and apply the same bins to both train and test while keeping the same engineered feature idea. Finally, I keep the model architecture/training loop intact, ensure train/test columns align deterministically, and write a valid `submission.csv` with columns `key,fare_amount`.'
- What this solution (achieved 2021.1194) has done: 'I fix the immediate runtime crash caused by the protobuf/TensorFlow stack by switching from `tf_keras` to the stable `keras` (Keras 3) API already installed in your environment, keeping the exact same Sequential architecture and training loop. I also make binning robust by ensuring `_apply_bins` never returns NaNs (values outside train bin edges be clipped into the edge bins), which should substantially improve RMSE without changing the overall feature idea. Finally, I keep train/test feature alignment deterministic and ensure the script always writes a valid `submission.csv` with the required `key,fare_amount` columns.'
- What this solution (achieved 102.75444) has done: 'I fix the immediate runtime crash caused by Keras importing a TensorFlow/protobuf stack that’s broken in this environment by switching the model code to `tf_keras` (which is installed and avoids that protobuf `GetPrototype` error). I also fix the major scoring logic issue where the `key` string column is accidentally included in the training feature matrix (making the scaler/model behave incorrectly), by explicitly excluding `key` from `feature_cols` and keeping it only for submission IDs. Finally, I keep the same feature engineering and the same neural network architecture/training loop, ensuring train/test columns align and a valid `submission.csv` with `key,fare_amount` is always written.'
- What this solution (achieved 25.20895) has done: 'I fix the immediate runtime crash caused by `tf_keras` triggering a protobuf `MessageFactory.GetPrototype` error in this environment by switching only the imports to the stable standalone `keras` package (Keras 3) while keeping the same Sequential model, layers, and training loop. Then I fix a major feature logic issue that is hurting RMSE: bin edges are currently fit on the already-cleaned/processed dataframe (after filtering), and binning can still produce NaNs/edge issues; I ensure bin edges are fit on the cleaned training data only and applied consistently, with robust clipping already present. Finally, I add a small but legitimate post-processing calibration step: use the validation split to fit a 1D linear correction (scale+bias) to predictions, then apply it to test predictions; this often reduces RMSE substantially without changing the core model/training semantics and should move the score toward the target band.'
- What this solution (achieved 109.08354) has done: 'I fix the immediate runtime crash by switching the model imports from standalone `keras` (which is pulling in a broken protobuf/TensorFlow stack here) to the already-installed `tf_keras` package, while keeping the exact same Sequential architecture, layers, optimizer, and training loop. I also set seeds through `tf_keras.utils.set_random_seed` to preserve determinism. No feature engineering, data cleaning, scaling, calibration, or submission-writing logic be changed, so behavior and evaluation semantics remain the same aside from removing the crash. This allow the notebook to run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 25.20895) has done: 'I fix the protobuf-related crash coming from importing `tf_keras` by switching the model imports to the standalone `keras` (Keras 3) package that’s installed in your environment; this keeps the exact same Sequential architecture and training loop. I also correct the train path to the actual Kaggle input location (`/kaggle/input/new-york-city-taxi-fare-prediction/train.csv`) so you train on real labeled data instead of crashing or reading a missing file. Finally, I keep the existing feature engineering, scaling, and submission-writing logic intact so the notebook runs end-to-end and writes a valid `submission.csv` with `key,fare_amount`, which should move RMSE substantially toward the target.'
- What this solution (achieved 101.28398) has done: 'I fix the immediate runtime crash (`MessageFactory.GetPrototype`) by switching from standalone `keras` imports to the already-installed `tf_keras` package, while keeping the exact same Sequential model architecture and training loop. I also make the bin-edge fitting robust by ensuring bin edges are computed on non-null values and that `_apply_bins` never produces NaNs even if a column has missing values, which stabilizes feature generation and should improve RMSE toward your target. Finally, I keep train/test feature alignment deterministic and ensure a valid `submission.csv` with `key,fare_amount` is always written.'
- What this solution (achieved 376.94295) has done: 'I fix the current runtime crash caused by importing `tf_keras` (protobuf `MessageFactory.GetPrototype` error) by switching the model API imports to the standalone `keras` package that is installed in your environment, keeping the exact same Sequential architecture, training loop, and loss/optimizer. I also make sure the Keras backend is set to TensorFlow explicitly (via `KERAS_BACKEND=tensorflow`) before importing `keras`, which avoids backend ambiguity and stabilizes execution on Kaggle. The rest of the pipeline (data loading, cleaning, binning, scaling, training, calibration, prediction, and submission writing) is left unchanged to preserve evaluation semantics while enabling an end-to-end run. This should also move RMSE substantially toward the target simply because the notebook train and infer correctly instead of crashing.'
- What this solution (achieved 105.03295) has done: 'I fix the import/runtime crash coming from `keras` pulling in a broken TensorFlow/protobuf stack (`MessageFactory.GetPrototype`) by switching only the model API imports to the already-installed `tf_keras` package, keeping the exact same Sequential architecture, optimizer, loss, and training loop. I also keep deterministic seeding via `tf_keras.utils.set_random_seed` to avoid run-to-run variance. No feature engineering, binning, scaling, calibration, or submission formatting logic be changed, so evaluation semantics stay the same while the notebook runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 48.97437) has done: 'I fix the immediate runtime crash (`MessageFactory.GetPrototype`) by avoiding importing `tf_keras`/TensorFlow entirely and switching only the model imports to scikit-learn’s `MLPRegressor`, which preserves the same “multi-layer dense neural network trained with MSE via gradient descent” core approach while running reliably in this environment. I keep all your existing feature engineering, binning, cleaning, scaling, and the linear calibration step intact, because those are central to your current logic and are already aligned with RMSE. I also ensure train/test feature columns remain perfectly aligned and numeric (float32) before fitting to prevent silent dtype issues that can inflate RMSE. Finally, the script still write a valid `submission.csv` with the required `key,fare_amount` columns.'
- What this solution (achieved 48.97437) has done: 'Your RMSE is far above target, so we need a small, legitimate improvement that preserves your overall pipeline. The biggest issue is the distance feature: you currently compute “manhattan” using a swapped latitude/longitude order, which makes the central engineered signal incorrect and can blow up RMSE; we fix that without changing the model/training approach. We also vectorize the `night`/`late_night` feature creation (same semantics, fewer opportunities for row-wise dtype issues) while keeping the same feature set. Everything else (binning, scaling, MLPRegressor, calibration, submission writing) stays intact to keep changes minimal and stable.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

SEED = 1
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)




## === cell 1
def clean(df):
    df = df[(-76 <= df["pickup_longitude"]) & (df["pickup_longitude"] <= -72)]
    df = df[(-76 <= df["dropoff_longitude"]) & (df["dropoff_longitude"] <= -72)]
    df = df[(38 <= df["pickup_latitude"]) & (df["pickup_latitude"] <= 42)]
    df = df[(38 <= df["dropoff_latitude"]) & (df["dropoff_latitude"] <= 42)]
    df = df[(0 < df["fare_amount"]) & (df["fare_amount"] <= 250)]
    df = df[(df["dropoff_longitude"] != df["pickup_longitude"])]
    df = df[(df["dropoff_latitude"] != df["pickup_latitude"])]
    return df


def late_night(row):
    if (row["hour"] <= 6) or (row["hour"] >= 20):
        return 1
    else:
        return 0


def night(row):
    if ((row["hour"] <= 20) and (row["hour"] >= 16)) and (row["weekday"] < 5):
        return 1
    else:
        return 0


_BIN_EDGES = {}


def _fit_bin_edges(train_df, col, q=16):
    s = (
        pd.to_numeric(train_df[col], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if len(s) == 0:
        return np.array([0.0, 1.0], dtype=np.float32)

    edges = pd.qcut(s, q, retbins=True, duplicates="drop")[1]
    edges = np.unique(edges.astype(np.float64))
    if len(edges) < 2:
        edges = np.array([float(s.min()), float(s.max())], dtype=np.float64)
    if len(edges) == 2 and edges[0] == edges[1]:
        edges = np.array([edges[0] - 1.0, edges[1] + 1.0], dtype=np.float64)
    return edges.astype(np.float32)


def _apply_bins(df, col, edges):
    x = pd.to_numeric(df[col], errors="coerce").astype("float32")
    lo = float(edges[0])
    hi = float(edges[-1])
    x = x.clip(lower=lo, upper=hi).fillna(lo)
    b = pd.cut(x, bins=edges, labels=False, include_lowest=True)
    return b.astype("float32").fillna(0).astype("int32")


def process(df, fit_bins=False):
    key = df["key"] if "key" in df.columns else None

    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"], errors="coerce", utc=True
    )
    df["year"] = df["pickup_datetime"].dt.year
    df["month"] = df["pickup_datetime"].dt.month
    df["day"] = df["pickup_datetime"].dt.day
    df["hour"] = df["pickup_datetime"].dt.hour
    df["weekday"] = df["pickup_datetime"].dt.weekday

    hr = df["hour"].astype("float32")
    wd = df["weekday"].astype("float32")
    df["night"] = (((hr <= 20) & (hr >= 16) & (wd < 5))).astype("int8")
    df["late_night"] = (((hr <= 6) | (hr >= 20))).astype("int8")

    cols_to_bin = [
        "pickup_longitude",
        "dropoff_longitude",
        "pickup_latitude",
        "dropoff_latitude",
    ]

    global _BIN_EDGES
    if fit_bins:
        for c in cols_to_bin:
            _BIN_EDGES[c] = _fit_bin_edges(df, c, q=16)

    df["pickup_longitude_binned"] = _apply_bins(
        df, "pickup_longitude", _BIN_EDGES["pickup_longitude"]
    )
    df["dropoff_longitude_binned"] = _apply_bins(
        df, "dropoff_longitude", _BIN_EDGES["dropoff_longitude"]
    )
    df["pickup_latitude_binned"] = _apply_bins(
        df, "pickup_latitude", _BIN_EDGES["pickup_latitude"]
    )
    df["dropoff_latitude_binned"] = _apply_bins(
        df, "dropoff_latitude", _BIN_EDGES["dropoff_latitude"]
    )

    df = df.drop("pickup_datetime", axis=1)

    if key is not None:
        df["key"] = key.values
    return df


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_relevant_distances(df):
    ny_lat, ny_lon = (40.7141667, -74.0063889)
    jfk_lat, jfk_lon = (40.6441666667, -73.7822222222)
    ewr_lat, ewr_lon = (40.69, -74.175)
    lgr_lat, lgr_lon = (40.77, -73.87)

    df["downtown_pickup_distance"] = manhattan(
        ny_lat, ny_lon, df["pickup_latitude"], df["pickup_longitude"]
    )
    df["downtown_dropoff_distance"] = manhattan(
        ny_lat, ny_lon, df["dropoff_latitude"], df["dropoff_longitude"]
    )
    df["jfk_pickup_distance"] = manhattan(
        jfk_lat, jfk_lon, df["pickup_latitude"], df["pickup_longitude"]
    )
    df["jfk_dropoff_distance"] = manhattan(
        jfk_lat, jfk_lon, df["dropoff_latitude"], df["dropoff_longitude"]
    )
    df["ewr_pickup_distance"] = manhattan(
        ewr_lat, ewr_lon, df["pickup_latitude"], df["pickup_longitude"]
    )
    df["ewr_dropoff_distance"] = manhattan(
        ewr_lat, ewr_lon, df["dropoff_latitude"], df["dropoff_longitude"]
    )
    df["lgr_pickup_distance"] = manhattan(
        lgr_lat, lgr_lon, df["pickup_latitude"], df["pickup_longitude"]
    )
    df["lgr_dropoff_distance"] = manhattan(
        lgr_lat, lgr_lon, df["dropoff_latitude"], df["dropoff_longitude"]
    )
    return df


def add_engineered(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]

    latdiff = lat1 - lat2
    londiff = lon1 - lon2
    euclidean = (latdiff**2 + londiff**2) ** 0.5

    df["latdiff"] = latdiff
    df["londiff"] = londiff
    df["euclidean"] = euclidean

    df["manhattan"] = manhattan(lat1, lon1, lat2, lon2)

    df = pd.get_dummies(df, columns=["weekday"])
    df = pd.get_dummies(df, columns=["month"])
    return df




## === cell 2
def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    df = pd.DataFrame(prediction, columns=[prediction_column])
    df[id_column] = raw_test[id_column].values
    df[[id_column, prediction_column]].to_csv(file_name, index=False)
    print(
        "Output complete:",
        file_name,
        "shape=",
        df[[id_column, prediction_column]].shape,
    )


def plot_loss_accuracy(history):
    if history is None:
        return
    plt.figure(figsize=(20, 10))
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper right")
    plt.show()




## === cell 3
TRAIN_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
TEST_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"
SUBMISSION_NAME = "submission.csv"

BATCH_SIZE = 256
EPOCHS = 5
LEARNING_RATE = 0.0001
DATASET_SIZE = 700000



## === cell 4
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

train = pd.read_csv(
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

test = pd.read_csv(
    TEST_PATH,
    dtype={k: v for k, v in datatypes.items() if k != "fare_amount"},
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



## === cell 5
train = clean(train)

train = process(train, fit_bins=True)
test = process(test, fit_bins=False)

train = add_relevant_distances(train)
test = add_relevant_distances(test)

train = add_engineered(train)
test = add_engineered(test)

feature_cols = [c for c in train.columns if c not in ["fare_amount", "key"]]

all_feature_cols = sorted(
    set(feature_cols) | set([c for c in test.columns if c != "key"])
)
all_feature_cols = [c for c in all_feature_cols if c != "key"]

train = train.reindex(columns=(["fare_amount", "key"] + all_feature_cols), fill_value=0)
test = test.reindex(columns=(["key"] + all_feature_cols), fill_value=0)

train = train.fillna(0)
test = test.fillna(0)



## === cell 6
dropped_columns = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]

train_clean = train.drop(dropped_columns, axis=1)
test_clean = test.drop(dropped_columns, axis=1)

train_clean.head(5)



## === cell 7
train_df, validation_df = train_test_split(train_clean, test_size=0.10, random_state=1)

train_labels = train_df["fare_amount"].values.astype(np.float32)
validation_labels = validation_df["fare_amount"].values.astype(np.float32)

train_df = train_df.drop(["fare_amount", "key"], axis=1, errors="ignore")
validation_df = validation_df.drop(["fare_amount", "key"], axis=1, errors="ignore")



## === cell 8
test_key = test_clean["key"].values if "key" in test_clean.columns else None
test_features = test_clean.drop(["key"], axis=1, errors="ignore")

test_features = test_features.reindex(columns=train_df.columns, fill_value=0)

train_df = train_df.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)
validation_df = (
    validation_df.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)
)
test_features = (
    test_features.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)
)

scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df).astype(np.float32)
validation_df_scaled = scaler.transform(validation_df).astype(np.float32)
test_scaled = scaler.transform(test_features).astype(np.float32)



## === cell 9
model = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64, 32, 16),
    activation="relu",
    solver="adam",
    alpha=0.0,  # keep regularization minimal; original used activity_regularizer, not weight decay
    batch_size=BATCH_SIZE,
    learning_rate_init=LEARNING_RATE,
    max_iter=EPOCHS,  # corresponds to epochs
    shuffle=True,
    random_state=SEED,
    early_stopping=False,
    validation_fraction=0.0,
    verbose=True,
)

print("Dataset size: %s" % DATASET_SIZE)
print("Epochs: %s" % EPOCHS)
print("Learning rate: %s" % LEARNING_RATE)
print("Batch size: %s" % BATCH_SIZE)
print("Input dimension: %s" % train_df_scaled.shape[1])
print("Features used: %s" % list(train_df.columns))



## === cell 10
model.fit(train_df_scaled, train_labels)



## === cell 11
history = None



## === cell 12
plot_loss_accuracy(history)



## === cell 13
val_pred = model.predict(validation_df_scaled).reshape(-1)
y_val = validation_labels.reshape(-1)

X = np.vstack([val_pred, np.ones_like(val_pred)]).T.astype(np.float64)
coef, _, _, _ = np.linalg.lstsq(X, y_val.astype(np.float64), rcond=None)
a, b = float(coef[0]), float(coef[1])

print("Calibration: y ~= a*pred + b with a=%.6f b=%.6f" % (a, b))



## === cell 14
prediction = model.predict(test_scaled).reshape(-1)
prediction = a * prediction + b
prediction = prediction.reshape(-1, 1)
prediction = np.clip(prediction, 0, None)



## === cell 15
raw_test = pd.read_csv(TEST_PATH, usecols=["key"])
output_submission(raw_test, prediction, "key", "fare_amount", SUBMISSION_NAME)
