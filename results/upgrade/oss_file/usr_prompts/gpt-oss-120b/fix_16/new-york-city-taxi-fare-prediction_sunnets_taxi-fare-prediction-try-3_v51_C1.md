# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

4.16348

# 6. Current score

603.25239

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 15.26194) has done: 'The fixes address the runtime errors that stopped the pipeline (invalid image URL handling, outdated optimizer API, missing visualization module) and ensure the model can train and produce a proper submission file. Minor tweaks (flattening predictions, safe water‑mask removal) keep the original architecture while improving the RMSE toward the target.'
- What this solution (achieved 15.14519) has done: 'The changes drop the problematic TensorFlow import, replace it with the standalone Keras API, ensure all non‑numeric columns are removed before scaling, and rewrite the custom RMSE metric using Keras backend. These fixes resolve the import error, the scaler conversion error, and the undefined‑variable issues so the script runs end‑to‑end and writes a proper *.csv* submission.'
- What this solution (achieved 3542.03957) has done: 'Implemented missing preprocessing, feature‑engineering, plotting, and submission utilities; replaced TensorFlow imports with pure Keras to avoid protobuf errors. Added data cleaning, datetime extraction, Haversine distance calculation, and safe CSV writing. The model now uses Keras’ built‑in RMSE metric, preserving the original architecture while ensuring the script runs end‑to‑end and produces a valid `submissiontry_water.csv` file.'
- What this solution (achieved 2136.51389) has done: 'Implemented fixes to resolve the import error, ensure correct test data loading, and simplify the model regularization which together allow the script to run end‑to‑end and produce a proper Kaggle submission CSV. The changes replace the problematic `keras` imports with the compatible `tf_keras` module, limit the columns read for the test set, and remove the aggressive L1 activity regularizer from the first dense layer.'
- What this solution (achieved 4485.61949) has done: 'I replaced the problematic `tf_keras` imports with the compatible standalone `keras` imports, which eliminates the protobuf‑related `MessageFactory` error and lets the notebook run end‑to‑end while keeping the original model architecture unchanged.'
- What this solution (achieved 2447.3316) has done: 'I replaced the problematic `keras` imports with the compatible `tf_keras` imports to eliminate the protobuf `MessageFactory` error, and I kept the `passenger_count` feature (it was previously dropped) to give the model more predictive power and move the RMSE toward the target. The rest of the logic and architecture remain unchanged.'
- What this solution (achieved 5.92043) has done: 'Implemented three focused fixes:  
1. Switched all `tf_keras` imports to the standalone `keras` package to remove the protobuf `MessageFactory` error.  
2. Strengthened data cleaning by filtering unrealistic `fare_amount` values (0‑200 USD) alongside coordinate bounds.  
3. Corrected the RMSE output index after model evaluation (now uses `score[2]`).  
These minimal adjustments restore end‑to‑end execution, produce a valid submission CSV, and improve the RMSE toward the target.'
- What this solution (achieved 2282.76864) has done: 'Implemented minimal, targeted fixes to resolve the protobuf import error and nudge model performance toward the target RMSE.  
- Switched all Keras imports to the compatible `tf_keras` package, eliminating the `MessageFactory` crash.  
- Increased training epochs from 30 to 60 (keeping the same architecture) to allow better convergence on the 80 k sample.  
- Adjusted the learning‑rate constant comment for clarity.  
All other logic, preprocessing, and submission steps remain unchanged, ensuring the script runs end‑to‑end and produces a valid CSV submission.'
- What this solution (achieved 1876.29434) has done: 'Implemented two key fixes:  
1. Switched all Keras imports from the problematic `tf_keras` package to the standalone `keras` library to eliminate the protobuf `MessageFactory` error.  
2. Applied the same data‑cleaning routine to the held‑out test split (`test_df`) so that model evaluation uses data with realistic coordinates and fare ranges, preventing massive RMSE inflation.

These minimal changes restore end‑to‑end execution and bring the model’s performance much closer to the target metric.'
- What this solution (achieved 2202.41515) has done: 'I replace the TensorFlow‑dependent Keras imports with the pure‑numpy `keras_core` equivalents, which eliminates the protobuf `MessageFactory` error that stops the script. The rest of the pipeline—including data cleaning, feature engineering, scaling, model definition, training, and CSV output—remains unchanged, so the core logic and evaluation semantics are preserved while allowing the code to run end‑to‑end and produce a valid submission file.'
- What this solution (achieved 1457.90567) has done: 'I replace the problematic `keras_core` imports with the standard `keras` package to eliminate the protobuf `MessageFactory` error, and I slightly lower the learning‑rate and number of epochs to keep training stable while preserving the original architecture. All other logic stays the same, ensuring a valid CSV submission is produced and the RMSE moves toward the target.'
- What this solution (achieved 4660.252) has done: 'I replaced the TensorFlow‑dependent Keras imports with the pure‑NumPy `keras_core` equivalents to eliminate the protobuf `MessageFactory` error, and I increased the training epochs from 30 to 100 so the model can converge better toward the target RMSE while keeping the original architecture unchanged. All other logic, preprocessing, and submission steps remain the same, ensuring the script runs end‑to‑end and produces a valid CSV submission.'
- What this solution (achieved 603.25239) has done: 'Implemented two key fixes:  
1. Replaced the `keras_core` imports with the standard `keras` package to eliminate the protobuf `MessageFactory` error that halted execution.  
2. Slightly lowered the learning rate and increased training epochs to give the model more opportunity to converge, nudging the RMSE closer to the target while keeping the original architecture unchanged.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras import optimizers, metrics, regularizers
from keras.layers import Dense, BatchNormalization

TRAIN_PATH = "../input/train.csv"
TEST_PATH = "../input/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 256
EPOCHS = 200  # increased epochs for better convergence
LEARNING_RATE = 0.0003  # slightly reduced learning rate
DATASET_SIZE = 80000


def clean(df):
    """Drop NaNs and filter out unrealistic coordinates and fare amounts."""
    df = df.dropna()
    lon_min, lon_max = -74.05, -73.75
    lat_min, lat_max = 40.63, 40.85
    mask = (
        (df["pickup_longitude"].between(lon_min, lon_max))
        & (df["dropoff_longitude"].between(lon_min, lon_max))
        & (df["pickup_latitude"].between(lat_min, lat_max))
        & (df["dropoff_latitude"].between(lat_min, lat_max))
        & (df["fare_amount"].between(0, 200))
    )
    return df[mask]


def add_time_features(df):
    """Extract hour, day of week and month from pickup_datetime."""
    df = df.copy()
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["hour"] = df["pickup_datetime"].dt.hour.astype(np.int8)
    df["dayofweek"] = df["pickup_datetime"].dt.dayofweek.astype(np.int8)
    df["month"] = df["pickup_datetime"].dt.month.astype(np.int8)
    return df


def add_coordinate_features(df):
    """Placeholder – coordinates already present; return unchanged."""
    return df


def haversine_np(lon1, lat1, lon2, lat2):
    """Vectorised haversine distance in kilometers."""
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def add_distances_features(df):
    """Add Haversine distance between pickup and dropoff points."""
    df = df.copy()
    df["distance"] = haversine_np(
        df["pickup_longitude"],
        df["pickup_latitude"],
        df["dropoff_longitude"],
        df["dropoff_latitude"],
    )
    return df


def plot_loss_accuracy_rmse(history):
    """Simple loss/RMSE plot."""
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    if "rmse" in history.history:
        plt.subplot(1, 2, 2)
        plt.plot(history.history["rmse"], label="train rmse")
        plt.plot(history.history["val_rmse"], label="val rmse")
        plt.title("RMSE")
        plt.xlabel("Epoch")
        plt.legend()
    plt.tight_layout()
    plt.show()


def output_submission(test_df_original, preds, key_col, target_col, filename):
    """Write Kaggle submission file with proper column order."""
    preds_flat = preds.ravel()
    preds_flat = np.where(preds_flat < 0, 0, preds_flat)
    submission = pd.DataFrame(
        {key_col: test_df_original[key_col].values, target_col: preds_flat}
    )
    submission.to_csv(filename, index=False)
    print(f"Submission written to {filename}")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
    usecols=[
        "fare_amount",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
        "key",
    ],
)

testKaggle = pd.read_csv(
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



## === cell 2
train_df, test_df = train_test_split(trainKaggle, test_size=0.50, random_state=1)



## === cell 3
print("testKaggle Size %d" % len(testKaggle))
print("train_df Size %d" % len(train_df))
print("test_df Size %d" % len(test_df))



## === cell 4
print("train_df clean")
train_df = clean(train_df)
test_df = clean(test_df)



## === cell 5
print("train_df add_time_features")
train_df = add_time_features(train_df)
print("test_df add_time_features")
test_df = add_time_features(test_df)
print("testKaggle add_time_features")
testKaggle = add_time_features(testKaggle)



## === cell 6
print("train_df add_coordinate_features")
train_df = add_coordinate_features(train_df)
print("test_df add_coordinate_features")
test_df = add_coordinate_features(test_df)
print("testKaggle add_coordinate_features")
testKaggle = add_coordinate_features(testKaggle)



## === cell 7
print("train_df add_distances_features")
train_df = add_distances_features(train_df)
print("test_df add_distances_features")
test_df = add_distances_features(test_df)
print("testKaggle add_distances_features")
testKaggle = add_distances_features(testKaggle)
print("Done with Adding features")



## === cell 8
dropped_columns = ["pickup_datetime"]
train_df = train_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)
testKaggle_clean = testKaggle.drop(dropped_columns + ["key"], axis=1)

numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [col for col in numeric_cols if col != "fare_amount"]

train_df = train_df[feature_cols + ["fare_amount"]]
test_df = test_df[feature_cols + ["fare_amount"]]
testKaggle_clean = testKaggle_clean[feature_cols]

print("Done with dropped_columns and numeric filtering")



## === cell 9
train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=1)



## === cell 10
train_labels = train_df["fare_amount"].values
validation_labels = validation_df["fare_amount"].values
test_labels = test_df["fare_amount"].values

train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)
test_df = test_df.drop(["fare_amount"], axis=1)

print("Done with Labels")



## === cell 11
scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_df)
testKaggle_scaled = scaler.transform(testKaggle_clean)



## === cell 12
model = Sequential()
model.add(
    Dense(
        256,
        activation="relu",
        input_dim=train_df_scaled.shape[1],
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
    loss="mean_squared_error",
    optimizer=adam,
    metrics=["mae", metrics.RootMeanSquaredError(name="rmse"), "mse"],
)

print("Dataset size: %s" % DATASET_SIZE)
print("Epochs: %s" % EPOCHS)
print("Learning rate: %s" % LEARNING_RATE)
print("Batch size: %s" % BATCH_SIZE)
print("Input dimension: %s" % train_df_scaled.shape[1])
print("Features used: %s" % train_df.columns)
model.summary()



## === cell 13
history = model.fit(
    x=train_df_scaled,
    y=train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1,
    validation_data=(validation_df_scaled, validation_labels),
    shuffle=True,
)



## === cell 14
plot_loss_accuracy_rmse(history)



## === cell 15
score = model.evaluate(test_scaled, test_labels, verbose=1)
print(score)
print("Test loss (MSE):", score[0])
print("Test RMSE:", score[2])  # RMSE is the third element in the returned list



## === cell 16
prediction = model.predict(test_scaled, batch_size=128, verbose=1)
predictionKaggle = model.predict(testKaggle_scaled, batch_size=128, verbose=1)



## === cell 17
output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)



## === cell 18
print(prediction[10000])
print(test_labels[10000])
