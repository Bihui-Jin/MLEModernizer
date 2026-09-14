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

4.15617

# 6. Current score

6.14634

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 449.30416) has done: 'I fixed the import to use the standalone Keras package (avoiding the protobuf error) and ensured that only numeric columns are kept before scaling, which removes the stray datetime strings that caused the MinMaxScaler to fail. These minimal fixes let the notebook run end‑to‑end and produce a valid “submissiontry_water.csv” file.'
- What this solution (achieved 92.67363) has done: 'I fix the import error by using TensorFlow’s Keras API, and I remove the strong L1 activity regularizer that was causing severe under‑fitting, which should dramatically lower the validation RMSE while keeping the overall pipeline unchanged.'
- What this solution (achieved 5.86648) has done: 'The fixes add a protobuf compatibility flag, replace the TensorFlow model with a scikit‑learn RandomForest (avoiding the TF import error), and adjust the downstream cells so training, validation, and submission work without errors. This keeps the original preprocessing and feature engineering while dramatically improving the RMSE toward the target.'
- What this solution (achieved 6.35793) has done: 'I fix the import error, keep the passenger count feature (it is useful), apply a log‑transform to the target to reduce skew, increase the number of trees for a stronger RandomForest, and revert the log‑transform when evaluating and creating the submission. These minimal changes resolve the runtime issue and should lower the validation RMSE toward the target score.'
- What this solution (achieved 5.6952) has done: 'Implemented fixes to resolve the protobuf import error, added a more accurate Haversine distance feature, and tuned the RandomForest hyper‑parameters for better generalisation. These changes keep the overall pipeline unchanged while enabling the notebook to run end‑to‑end and should lower the validation RMSE toward the target.'
- What this solution (achieved 6.26363) has done: 'The updates add cyclic time‑of‑day and weekday features (which capture periodic patterns) and make the RandomForest a bit stronger by using more trees, removing the depth limit and allowing leaves of size 1. These modest changes keep the overall pipeline unchanged while aiming to lower the validation RMSE toward the target.'
- What this solution (achieved 5.56823) has done: 'The change replaces the very deep random forest with a HistGradientBoostingRegressor, which typically yields lower RMSE on tabular data while keeping the same preprocessing and feature set. The model keeps the log‑target transformation and scaling, so the core pipeline is unchanged, but the stronger gradient‑boosting learner should move the validation RMSE closer to the target value.'
- What this solution (achieved 6.19535) has done: 'I increase the training sample size to give the model more data and tweak the HistGradientBoostingRegressor hyper‑parameters (more iterations, slightly lower learning rate and a smaller leaf size). These modest adjustments keep the overall pipeline unchanged while encouraging a lower validation RMSE, moving the score closer to the target.'
- What this solution (achieved 6.14634) has done: 'I keep the overall pipeline unchanged and only adjust the HistGradientBoostingRegressor hyper‑parameters so the model has higher capacity and less regularisation, which should lower the validation RMSE and move it closer to the target (while still being a minimal change). The rest of the code, feature engineering and preprocessing remain identical.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor

from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor

TRAIN_PATH = "../input/train.csv"
TEST_PATH = "../input/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 0.001
DATASET_SIZE = 200000




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
trainKaggle = pd.read_csv(TRAIN_PATH, nrows=DATASET_SIZE, dtype=datatypes)
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
train_df, test_df = train_test_split(trainKaggle, test_size=0.20, random_state=1)




## === cell 3
train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=1)




## === cell 4
def clean(df):
    print(" Old size: %d" % len(df))
    df = df.dropna(how="any", axis="rows")
    print(" New size after dropna: %d" % len(df))

    df = df[
        (df["dropoff_longitude"] != df["pickup_longitude"])
        | (df["dropoff_latitude"] != df["pickup_latitude"])
    ]
    print(" New size after removing same long/lat: %d" % len(df))

    df = df[
        (df["dropoff_longitude"] != 0)
        & (df["pickup_longitude"] != 0)
        & (df["dropoff_latitude"] != 0)
        & (df["pickup_latitude"] != 0)
    ]
    print(" New size after removing 0 long/lat: %d" % len(df))

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
    print(" New size after NYC bounds: %d" % len(df))

    df = df[(0 < df["fare_amount"]) & (df["fare_amount"] <= 50)]
    print(" New size after fare outlier filter: %d" % len(df))

    df = df[(df["passenger_count"] > 0) & (df["passenger_count"] <= 6)]
    print(" New size after passenger count filter: %d" % len(df))

    return df


def remove_datapoints_from_water(df):
    return df


def late_night(row):
    return 1 if row["hour"] <= 3 else 0


def night(row):
    return 1 if (row["hour"] > 20) and (row["weekday"] < 5) else 0


def rush_hour(row):
    return 1 if (16 <= row["hour"] <= 20) and (row["weekday"] < 5) else 0


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def haversine_distance(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    R = 6371.0
    lat1 = np.radians(pickup_lat)
    lat2 = np.radians(dropoff_lat)
    dlat = np.radians(dropoff_lat - pickup_lat)
    dlon = np.radians(dropoff_long - pickup_long)

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def add_time_features(df):
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df["year"] = df["pickup_datetime"].dt.year
    df["month"] = df["pickup_datetime"].dt.month
    df["day"] = df["pickup_datetime"].dt.day
    df["hour"] = df["pickup_datetime"].dt.hour
    df["weekday"] = df["pickup_datetime"].dt.weekday
    df["night"] = df.apply(night, axis=1)
    df["late_night"] = df.apply(late_night, axis=1)
    df["rush_hour"] = df.apply(rush_hour, axis=1)
    df["pickup_datetime"] = df["pickup_datetime"].astype(str)
    return df


def add_coordinate_features(df):
    df["latdiff"] = df["pickup_latitude"] - df["dropoff_latitude"]
    df["londiff"] = df["pickup_longitude"] - df["dropoff_longitude"]
    return df


def add_distances_features(df):
    df["manhattan"] = manhattan(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )
    df["distance"] = np.sqrt(
        (df["pickup_longitude"] - df["dropoff_longitude"]) ** 2
        + (df["pickup_latitude"] - df["dropoff_latitude"]) ** 2
    )
    df["haversine"] = haversine_distance(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )
    return df


def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    df = pd.DataFrame(
        {id_column: raw_test[id_column], prediction_column: prediction.ravel()}
    )
    df.to_csv(file_name, index=False)
    print("Output complete")


def plot_loss_accuracy(history):
    if history is None:
        print("No training history to plot.")
        return
    plt.figure(figsize=(12, 5))
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title("Model loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()




## === cell 5
print("Cleaning train set")
train_df = clean(train_df)
print("Cleaning validation set")
validation_df = clean(validation_df)




## === cell 6
print("Adding time features")
train_df = add_time_features(train_df)
validation_df = add_time_features(validation_df)
test_df = add_time_features(test_df)
testKaggle = add_time_features(testKaggle)


def add_cyclic_features(df):
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    return df


train_df = add_cyclic_features(train_df)
validation_df = add_cyclic_features(validation_df)
test_df = add_cyclic_features(test_df)
testKaggle = add_cyclic_features(testKaggle)




## === cell 7
print("Adding coordinate features")
train_df = add_coordinate_features(train_df)
validation_df = add_coordinate_features(validation_df)
test_df = add_coordinate_features(test_df)
testKaggle = add_coordinate_features(testKaggle)




## === cell 8
print("Adding distance features")
train_df = add_distances_features(train_df)
validation_df = add_distances_features(validation_df)
test_df = add_distances_features(test_df)
testKaggle = add_distances_features(testKaggle)




## === cell 9
dropped_columns = ["pickup_datetime"]
train_df = train_df.drop(columns=dropped_columns)
validation_df = validation_df.drop(columns=dropped_columns)
test_df = test_df.drop(columns=dropped_columns)
testKaggle_clean = testKaggle.drop(columns=dropped_columns + ["key"])
numeric_cols = train_df.select_dtypes(include=[np.number]).columns
train_df = train_df[numeric_cols]
validation_df = validation_df[numeric_cols]
test_df = test_df[numeric_cols]
testKaggle_clean = testKaggle_clean.select_dtypes(include=[np.number])




## === cell 10
validation_labels_original = validation_df["fare_amount"].values
train_labels = np.log1p(train_df["fare_amount"].values)
validation_labels = np.log1p(validation_labels_original)
test_labels = test_df["fare_amount"].values  # not used further

train_df = train_df.drop(columns=["fare_amount"])
validation_df = validation_df.drop(columns=["fare_amount"])
test_df = test_df.drop(columns=["fare_amount"])




## === cell 11
scaler = preprocessing.MinMaxScaler()
train_scaled = scaler.fit_transform(train_df)
validation_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_df)
testKaggle_scaled = scaler.transform(testKaggle_clean)




## === cell 12
hgb_model = HistGradientBoostingRegressor(
    max_iter=1000,  # more trees for better fit
    learning_rate=0.02,  # smaller step to stabilise learning
    max_depth=None,
    min_samples_leaf=1,  # reduce regularisation
    l2_regularization=0.0,
    random_state=42,
)
hgb_model.fit(train_scaled, train_labels)
print("HistGradientBoosting training complete")
rf_model = hgb_model
history = None  # placeholder for compatibility with plot function




## === cell 13
plot_loss_accuracy(history)




## === cell 14
pred_test = rf_model.predict(test_scaled)
pred_kaggle = rf_model.predict(testKaggle_scaled)

pred_test = np.expm1(pred_test)
pred_kaggle = np.expm1(pred_kaggle)




## === cell 15
val_pred_log = rf_model.predict(validation_scaled)
val_pred = np.expm1(val_pred_log)
rmse_val = np.sqrt(mean_squared_error(validation_labels_original, val_pred.ravel()))
print(f"Validation RMSE: {rmse_val:.4f}")




## === cell 16
output_submission(testKaggle, pred_kaggle, "key", "fare_amount", SUBMISSION_NAME)
