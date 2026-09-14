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

4.74408

# 6. Current score

15.35031

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 15.45712) has done: 'I replace the TensorFlow imports with the pure Keras package to avoid the protobuf error, and I ensure only numeric columns are fed to MinMaxScaler so the scaler no longer encounters string timestamps. This lets the notebook run end‑to‑end, produce a trained model, and write a proper submission.csv file while keeping the original architecture and logic unchanged.'
- What this solution (achieved 15.35031) has done: 'I added all missing imports, defined the helper functions (`clean`, `add_time_features`, `add_coordinate_features`, `add_distances_features`, `plot_loss_accuracy_rmse`, `output_submission`), fixed the test‑file reading dtype mismatch, and ensured the Keras objects are imported from the pure Keras package. These changes unblock the notebook, let the model train and predict, and finally write a correctly formatted `submissiontry_water.csv` file, moving the solution toward the target RMSE.'

# 9. Code solution

## === cell 0
TRAIN_PATH = "../input/train.csv"
TEST_PATH = "../input/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 256
EPOCHS = 200
LEARNING_RATE = 0.001
DATASET_SIZE = 200000  # increased from 80000 for better training



## === cell 1
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt

from keras import layers, regularizers, optimizers, backend
import keras


def clean(df):
    """Basic cleaning: drop rows with any NaN and ensure sensible passenger count."""
    df = df.dropna()
    df = df[df["passenger_count"] > 0]
    return df.reset_index(drop=True)


def add_time_features(df):
    """Extract hour, weekday and month from pickup_datetime."""
    dt = pd.to_datetime(df["pickup_datetime"])
    df["pickup_hour"] = dt.dt.hour
    df["pickup_weekday"] = dt.dt.weekday
    df["pickup_month"] = dt.dt.month
    return df


def haversine_distance(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in kilometers."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def add_coordinate_features(df):
    """Placeholder – coordinates are already present; keep as‑is."""
    return df


def add_distances_features(df):
    """Add haversine distance between pickup and dropoff."""
    df["haversine_km"] = haversine_distance(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )
    return df


def plot_loss_accuracy_rmse(history):
    """Simple loss / MAE / RMSE plot."""
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    ax[0].plot(history.history["loss"], label="train loss")
    ax[0].plot(history.history["val_loss"], label="val loss")
    ax[0].set_title("Loss")
    ax[0].legend()
    ax[1].plot(history.history["mae"], label="train MAE")
    ax[1].plot(history.history["val_mae"], label="val MAE")
    ax[1].set_title("MAE")
    ax[1].legend()
    if "rmse_metric" in history.history:
        ax[2].plot(history.history["rmse_metric"], label="train RMSE")
        ax[2].plot(history.history["val_rmse_metric"], label="val RMSE")
    else:
        ax[2].plot(np.sqrt(history.history["loss"]), label="train RMSE")
        ax[2].plot(np.sqrt(history.history["val_loss"]), label="val RMSE")
    ax[2].set_title("RMSE")
    ax[2].legend()
    plt.show()


def output_submission(test_df, predictions, key_col, target_col, file_name):
    """Write predictions together with the key column to a CSV file."""
    pred = predictions.squeeze()
    out_df = pd.DataFrame({key_col: test_df[key_col], target_col: pred})
    out_df.to_csv(file_name, index=False)
    print(f"Submission written to {file_name}")


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

test_dtypes = {k: v for k, v in datatypes.items() if k != "fare_amount"}
testKaggle = pd.read_csv(TEST_PATH, dtype=test_dtypes)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
train_df, test_df = train_test_split(trainKaggle, test_size=0.50, random_state=1)
test_df = test_df[:10000]



## === cell 3
print("testKaggle Size %d" % len(testKaggle))
print("train_df Size %d" % len(train_df))
print("test_df Size %d" % len(test_df))



## === cell 4
print("Cleaning train_df")
train_df = clean(train_df)
print("Cleaning test_df")
test_df = clean(test_df)



## === cell 5
print("Adding time features")
train_df = add_time_features(train_df)
test_df = add_time_features(test_df)
testKaggle = add_time_features(testKaggle)



## === cell 6
print("Adding coordinate features")
train_df = add_coordinate_features(train_df)
test_df = add_coordinate_features(test_df)
testKaggle = add_coordinate_features(testKaggle)



## === cell 7
print("Adding distance features")
train_df = add_distances_features(train_df)
test_df = add_distances_features(test_df)
testKaggle = add_distances_features(testKaggle)



## === cell 8
dropped_columns = ["pickup_datetime"]
train_df = train_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)
testKaggle_clean = testKaggle.drop(dropped_columns + ["key"], axis=1)

print("Columns dropped; passenger_count retained.")



## === cell 9
print("train_df shape:", train_df.shape)



## === cell 10
train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=1)



## === cell 11
train_labels = train_df["fare_amount"].values
validation_labels = validation_df["fare_amount"].values
test_labels = test_df["fare_amount"].values

train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)
test_df = test_df.drop(["fare_amount"], axis=1)

print("Labels separated.")



## === cell 12
numeric_cols = train_df.select_dtypes(include=[np.number]).columns

train_df_num = train_df[numeric_cols]
validation_df_num = validation_df[numeric_cols]
test_df_num = test_df[numeric_cols]
testKaggle_num = testKaggle_clean[numeric_cols]

scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df_num)
validation_df_scaled = scaler.transform(validation_df_num)
test_scaled = scaler.transform(test_df_num)
testKaggle_scaled = scaler.transform(testKaggle_num)




## === cell 13
def rmse_metric(y_true, y_pred):
    return backend.mean(backend.square(y_pred - y_true), axis=-1) ** 0.5




## === cell 14
model = keras.Sequential(
    [
        layers.Dense(
            256,
            activation="linear",
            input_dim=train_df_scaled.shape[1],
            activity_regularizer=regularizers.l1(0.01),
        ),
        layers.BatchNormalization(),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(32, activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(8, activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(1),
    ]
)

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



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/4072242463.py in <cell line: 0>()
----> 1 history = model.fit(
      2     x=train_df_scaled,
      3     y=train_labels,
      4     batch_size=BATCH_SIZE,
      5     epochs=EPOCHS,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_55/837193386.py in rmse_metric(y_true, y_pred)
      1 def rmse_metric(y_true, y_pred):
----> 2     return backend.mean(backend.square(y_pred - y_true), axis=-1) ** 0.5
      3 
      4 

AttributeError: module 'keras.api.backend' has no attribute 'mean'

## === cell 16
plot_loss_accuracy_rmse(history)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2579130885.py in <cell line: 0>()
----> 1 plot_loss_accuracy_rmse(history)
      2 

NameError: name 'history' is not defined

## === cell 17
score = model.evaluate(test_scaled, test_labels, verbose=1)
print("Test loss, MAE, RMSE, MSE:", score)



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/3437567575.py in <cell line: 0>()
----> 1 score = model.evaluate(test_scaled, test_labels, verbose=1)
      2 print("Test loss, MAE, RMSE, MSE:", score)
      3 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_55/837193386.py in rmse_metric(y_true, y_pred)
      1 def rmse_metric(y_true, y_pred):
----> 2     return backend.mean(backend.square(y_pred - y_true), axis=-1) ** 0.5
      3 
      4 

AttributeError: module 'keras.api.backend' has no attribute 'mean'

## === cell 18
prediction = model.predict(test_scaled, batch_size=128, verbose=1)
predictionKaggle = model.predict(testKaggle_scaled, batch_size=128, verbose=1)



## === cell 19
validation_pred = model.predict(validation_df_scaled, batch_size=128, verbose=0)
rmse_val = np.sqrt(((validation_pred.squeeze() - validation_labels) ** 2).mean())
print(f"Validation RMSE: {rmse_val:.4f}")



## === cell 20
output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)
