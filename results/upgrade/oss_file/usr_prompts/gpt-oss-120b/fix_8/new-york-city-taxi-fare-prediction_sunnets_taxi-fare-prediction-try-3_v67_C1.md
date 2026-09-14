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

4.70167

# 6. Current score

11.53297

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 15.21729) has done: 'Implemented fixes to resolve runtime errors and improve model performance:
- Replaced URL‑based mask loading with a safe fallback that simply returns the dataframe.
- Corrected optimizer call (`optimizers.Adam`) and its argument name.
- Added graceful handling for optional visualisation imports.
- Ensured non‑numeric columns (`key`) are removed before scaling.
- Fixed the always‑true `late_night` logic.
- Adjusted cell ordering and naming to be sequential and functional.'
- What this solution (achieved 65.52696) has done: 'The script is updated to fix the import errors (using TensorFlow Keras which provides the missing backend functions), correct the `rmse_metric`, and keep the informative **passenger_count** feature instead of dropping it. These changes resolve the runtime failures, allow the model to train and evaluate correctly, and should lower the RMSE toward the target value.'
- What this solution (achieved 15.17757) has done: 'Implemented a fix for the TensorFlow/Keras import issue by switching to the standalone Keras package, which avoids the protobuf‑related `MessageFactory` error. This change restores model building, training, and evaluation without altering the core architecture or training logic. All other cells are kept unchanged, preserving feature engineering and submission generation.'
- What this solution (achieved 471.69692) has done: 'The fix adds TensorFlow to define a proper RMSE metric (using `tf.sqrt`), updates the metric implementation, and imports TensorFlow. It also increases training epochs from 100 to 200 to give the model more learning capacity, which should lower the RMSE toward the target while keeping the original architecture unchanged.'
- What this solution (achieved 15.57248) has done: 'The fix adds the missing preprocessing and submission helper functions, switches to the standalone keras package to avoid the protobuf import error, and defines a simple `clean` routine. These changes resolve the runtime NameErrors, allow the model to train and generate predictions, and ensure a correctly‑named `.csv` submission file is written.'
- What this solution (achieved 11.53297) has done: 'I reorder the imports to load TensorFlow before Keras (preventing the protobuf MessageFactory error) and replace the custom RMSE metric with a TensorFlow‑based implementation that uses `tf.sqrt`. This fixes the runtime AttributeError, enables training to complete, and provides a proper RMSE metric so the model’s evaluation moves closer to the target score. The core model architecture and training procedure remain unchanged.'

# 9. Code solution

## === cell 0
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras import optimizers, regularizers, backend as K

TRAIN_PATH = "../input/train.csv"
TEST_PATH = "../input/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 256
EPOCHS = 200  # retained for full training
LEARNING_RATE = 0.001
DATASET_SIZE = 80000




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning – drop rows with missing values."""
    return df.dropna().reset_index(drop=True)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract hour, day of week and month from pickup_datetime."""
    df = df.copy()
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["hour"] = df["pickup_datetime"].dt.hour.astype(np.float32)
    df["dayofweek"] = df["pickup_datetime"].dt.dayofweek.astype(np.float32)
    df["month"] = df["pickup_datetime"].dt.month.astype(np.float32)
    return df


def add_coordinate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple coordinate differences."""
    df = df.copy()
    df["diff_longitude"] = df["dropoff_longitude"] - df["pickup_longitude"]
    df["diff_latitude"] = df["dropoff_latitude"] - df["pickup_latitude"]
    return df


def add_distances_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add haversine distance (in km) between pickup and dropoff points."""
    df = df.copy()
    lon1 = np.radians(df["pickup_longitude"])
    lat1 = np.radians(df["pickup_latitude"])
    lon2 = np.radians(df["dropoff_longitude"])
    lat2 = np.radians(df["dropoff_latitude"])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    earth_radius_km = 6371.0
    df["haversine_km"] = (earth_radius_km * c).astype(np.float32)
    return df


def output_submission(
    df_input: pd.DataFrame,
    preds: np.ndarray,
    key_col: str,
    target_col: str,
    filename: str,
):
    """Create submission CSV with required columns."""
    pred_series = pd.Series(preds.ravel(), name=target_col)
    submission = pd.DataFrame({key_col: df_input[key_col], target_col: pred_series})
    submission.to_csv(filename, index=False)




## === cell 2
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
testKaggle = pd.read_csv(TEST_PATH, dtype=datatypes)




## === cell 3
train_df, test_df = train_test_split(trainKaggle, test_size=0.50, random_state=1)
test_df = test_df[:10000]  # keep a manageable size for quick testing




## === cell 4
print("trainKaggle size %d" % len(trainKaggle))
print("train_df size %d" % len(train_df))
print("test_df size %d" % len(test_df))




## === cell 5
train_df = clean(train_df)
test_df = clean(test_df)




## === cell 6
print("Adding time features")
train_df = add_time_features(train_df)
test_df = add_time_features(test_df)
testKaggle = add_time_features(testKaggle)




## === cell 7
print("Adding coordinate differences")
train_df = add_coordinate_features(train_df)
test_df = add_coordinate_features(test_df)
testKaggle = add_coordinate_features(testKaggle)




## === cell 8
print("Adding distance features")
train_df = add_distances_features(train_df)
test_df = add_distances_features(test_df)
testKaggle = add_distances_features(testKaggle)




## === cell 9
dropped_columns = ["pickup_datetime", "key"]
train_df = train_df.drop(columns=dropped_columns)
test_df = test_df.drop(columns=dropped_columns)
testKaggle_clean = testKaggle.drop(
    columns=dropped_columns
)  # keep key separately for submission
print("Columns dropped; remaining features:", train_df.columns.tolist())




## === cell 10
train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=1)




## === cell 11
train_labels = train_df["fare_amount"].values
validation_labels = validation_df["fare_amount"].values
test_labels = test_df["fare_amount"].values

train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)
test_df = test_df.drop(["fare_amount"], axis=1)

print("Prepared labels and cleaned feature sets.")




## === cell 12
scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_df)
testKaggle_scaled = scaler.transform(testKaggle_clean)




## === cell 13
def rmse_metric(y_true, y_pred):
    """Root Mean Squared Error using TensorFlow operations."""
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))




## === cell 14
model = Sequential()
model.add(
    Dense(
        256,
        activation="linear",
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
    loss="mean_squared_error",
    optimizer=adam,
    metrics=["mae", rmse_metric, "mse"],
)
model.summary()




## === cell 15
history = model.fit(
    x=train_df_scaled,
    y=train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1,
    validation_data=(validation_df_scaled, validation_labels),
    shuffle=True,
)




## === cell 16
def plot_loss_accuracy_rmse(hist):
    plt.figure(figsize=(12, 5))
    plt.plot(hist.history["loss"], label="train loss")
    plt.plot(hist.history["val_loss"], label="val loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.show()

    if "rmse_metric" in hist.history:
        plt.figure(figsize=(12, 5))
        plt.plot(hist.history["rmse_metric"], label="train rmse")
        plt.plot(hist.history["val_rmse_metric"], label="val rmse")
        plt.title("RMSE Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("RMSE")
        plt.legend()
        plt.show()


plot_loss_accuracy_rmse(history)




## === cell 17
train_score = model.evaluate(train_df_scaled, train_labels, verbose=0)
val_score = model.evaluate(validation_df_scaled, validation_labels, verbose=0)
test_score = model.evaluate(test_scaled, test_labels, verbose=0)

print("Train RMSE:", train_score[2])  # rmse_metric is third metric after mae
print("Validation RMSE:", val_score[2])
print("Test RMSE:", test_score[2])




## === cell 18
predictionKaggle = model.predict(testKaggle_scaled, batch_size=128, verbose=1)
output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)
