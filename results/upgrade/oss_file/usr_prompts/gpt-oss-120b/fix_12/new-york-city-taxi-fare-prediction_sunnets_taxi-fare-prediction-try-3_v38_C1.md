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

4.44194

# 6. Current score

6.98998

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 15.65867) has done: 'The fix replaces the TensorFlow import with Keras‑Core (which works without the protobuf issue), ensures only numeric columns are passed to MinMaxScaler, and adjusts the small loss‑check cell to use Keras‑Core backend operations. These changes let the notebook run end‑to‑end and produce a valid `submissiontry_water.csv` file while keeping the original model architecture and feature engineering unchanged.'
- What this solution (achieved 1005.96608) has done: 'The changes move the heavy Keras imports into the training cell to avoid the protobuf import error, drop only the datetime column (keeping useful features like passenger count), remove the broken custom RMSE metric and compute RMSE manually after training, and adjust the evaluation logic accordingly. This fixes the runtime crashes, ensures a proper submission file is written, and improves model performance toward the target score.'
- What this solution (achieved 6.98998) has done: 'Implemented robust data directory discovery and fallback logic so the script correctly locates train.csv and test.csv in typical Kaggle environments (e.g., /kaggle/input or a local new‑york‑city‑taxi‑fare‑prediction folder). Added a safe default to the current working directory and clarified error handling. No other logic was altered, preserving the original model pipeline and ensuring a valid submission.csv is written.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

default_base = Path(os.getenv("KAGGLE_INPUT_DIR", "/kaggle/input"))
if not default_base.exists():
    default_base = Path("data")
base_dir = default_base

candidate_paths = [
    base_dir / "new-york-city-taxi-fare-prediction",
    base_dir
    / "new-york-city-taxi-fare-prediction"
    / "new-york-city-taxi-fare-prediction",
    base_dir,
    Path("new-york-city-taxi-fare-prediction"),  # local fallback
]


def find_data_dir(paths):
    for p in paths:
        if (p / "train.csv").exists() and (p / "test.csv").exists():
            return p
    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents):
        if (parent / "train.csv").exists() and (parent / "test.csv").exists():
            return parent
    return None


DATA_DIR = find_data_dir(candidate_paths)

if DATA_DIR is None:
    raise FileNotFoundError(
        "train.csv and test.csv not found in any expected location. "
        "Checked candidate paths and current/parent directories."
    )

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SUBMISSION_NAME = "submission.csv"

DATASET_SIZE = 200_000
print(f"Data directory resolved to: {DATA_DIR}")
print(f"Training file: {TRAIN_PATH}")
print(f"Test file: {TEST_PATH}")



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
testKaggle = pd.read_csv(TEST_PATH, dtype=datatypes)



## === cell 2
train_df, test_df = train_test_split(trainKaggle, test_size=0.50, random_state=1)
train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=1)




## === cell 3
def remove_datapoints_from_water(df):
    return df


def clean(df):
    print(f" Cleaning: original size {len(df)}")
    df = df.dropna(how="any", axis="rows")
    df = df[
        (df["dropoff_longitude"] != df["pickup_longitude"])
        & (df["dropoff_latitude"] != df["pickup_latitude"])
    ]
    df = df[
        (df["dropoff_longitude"] != 0)
        & (df["pickup_longitude"] != 0)
        & (df["dropoff_latitude"] != 0)
        & (df["pickup_latitude"] != 0)
    ]
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
    df = df[(0 < df["fare_amount"]) & (df["fare_amount"] <= 50)]
    df = df[(df["passenger_count"] > 0) & (df["passenger_count"] <= 6)]
    df = remove_datapoints_from_water(df)
    print(f" Cleaning: resulting size {len(df)}")
    return df


def late_night(row):
    return int(row["hour"] <= 3)


def night(row):
    return int((row["hour"] > 20) and (row["weekday"] < 5))


def rush_hour(row):
    return int((16 <= row["hour"] <= 20) and (row["weekday"] < 5))


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


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
    return df


def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    df = pd.DataFrame({prediction_column: prediction.ravel()})
    df[id_column] = raw_test[id_column].values
    df = df[[id_column, prediction_column]]
    df.to_csv(file_name, index=False)
    print(f"Submission written to {file_name}")




## === cell 4
print("Cleaning train_df")
train_df = clean(train_df)
print("Cleaning validation_df")
validation_df = clean(validation_df)
print("Cleaning test_df")
test_df = clean(test_df)

train_df = add_time_features(train_df)
validation_df = add_time_features(validation_df)
test_df = add_time_features(test_df)
testKaggle = add_time_features(testKaggle)

train_df = add_coordinate_features(train_df)
validation_df = add_coordinate_features(validation_df)
test_df = add_coordinate_features(test_df)
testKaggle = add_coordinate_features(testKaggle)

train_df = add_distances_features(train_df)
validation_df = add_distances_features(validation_df)
test_df = add_distances_features(test_df)
testKaggle = add_distances_features(testKaggle)



## === cell 5
drop_cols = ["pickup_datetime"]
train_df = train_df.drop(columns=drop_cols)
validation_df = validation_df.drop(columns=drop_cols)
test_df = test_df.drop(columns=drop_cols)
testKaggle_clean = testKaggle.drop(columns=drop_cols + ["key"])



## === cell 6
train_labels = train_df["fare_amount"].values
validation_labels = validation_df["fare_amount"].values
test_labels = test_df["fare_amount"].values

train_df = train_df.drop(columns=["fare_amount"])
validation_df = validation_df.drop(columns=["fare_amount"])
test_df = test_df.drop(columns=["fare_amount"])

print("Label extraction done")



## === cell 7
print("train shape:", train_df.shape)
print("validation shape:", validation_df.shape)
print("test shape:", test_df.shape)



## === cell 8
numeric_cols = train_df.select_dtypes(include=[np.number]).columns
scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df[numeric_cols])
validation_df_scaled = scaler.transform(validation_df[numeric_cols])
test_scaled = scaler.transform(test_df[numeric_cols])
testKaggle_scaled = scaler.transform(testKaggle_clean[numeric_cols])



## === cell 9
model = GradientBoostingRegressor(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    random_state=42,
)
print("Training GradientBoostingRegressor...")
model.fit(train_df_scaled, train_labels)
print("Training complete.")



## === cell 10
val_pred = model.predict(validation_df_scaled)
val_rmse = np.sqrt(mean_squared_error(validation_labels, val_pred))
print(f"Validation RMSE: {val_rmse:.4f}")



## === cell 11
test_pred = model.predict(test_scaled)
test_rmse = np.sqrt(mean_squared_error(test_labels, test_pred))
print(f"Internal test RMSE: {test_rmse:.4f}")



## === cell 12
predictionKaggle = model.predict(testKaggle_scaled).reshape(-1, 1)
output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)



## === cell 13
print("Example prediction vs label (internal test set)")
print("Prediction[0]:", test_pred[0])
print("Label[0]:", test_labels[0])
