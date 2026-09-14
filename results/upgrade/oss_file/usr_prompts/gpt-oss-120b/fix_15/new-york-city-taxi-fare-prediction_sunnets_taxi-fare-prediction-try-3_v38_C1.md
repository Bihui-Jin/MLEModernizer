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

4.44194

# 6. Current score

5.61473

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 15.65867) has done: 'The fix replaces the TensorFlow import with Keras‑Core (which works without the protobuf issue), ensures only numeric columns are passed to MinMaxScaler, and adjusts the small loss‑check cell to use Keras‑Core backend operations. These changes let the notebook run end‑to‑end and produce a valid `submissiontry_water.csv` file while keeping the original model architecture and feature engineering unchanged.'
- What this solution (achieved 1005.96608) has done: 'The changes move the heavy Keras imports into the training cell to avoid the protobuf import error, drop only the datetime column (keeping useful features like passenger count), remove the broken custom RMSE metric and compute RMSE manually after training, and adjust the evaluation logic accordingly. This fixes the runtime crashes, ensures a proper submission file is written, and improves model performance toward the target score.'
- What this solution (achieved 6.98998) has done: 'Implemented robust data directory discovery and fallback logic so the script correctly locates train.csv and test.csv in typical Kaggle environments (e.g., /kaggle/input or a local new‑york‑city‑taxi‑fare‑prediction folder). Added a safe default to the current working directory and clarified error handling. No other logic was altered, preserving the original model pipeline and ensuring a valid submission.csv is written.'
- What this solution (achieved 5.61473) has done: 'Optimized the pipeline by cleaning and engineering features once on the full training and test sets before splitting, and combined multiple filtering steps into a single boolean mask to avoid repeated DataFrame copies. This reduces redundant work and memory overhead while keeping all feature calculations and model logic unchanged, ensuring identical predictions and evaluation semantics.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from pathlib import Path

from sklearnex import patch_sklearn

patch_sklearn()
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

DATASET_SIZE = 300_000
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
def clean(df):
    print(f" Cleaning: original size {len(df)}")
    MinMax = (-74.5, -72.8, 40.5, 41.8)
    mask = (
        df.notnull().all(axis=1)
        & (df["dropoff_longitude"] != df["pickup_longitude"])
        & (df["dropoff_latitude"] != df["pickup_latitude"])
        & (df["dropoff_longitude"] != 0)
        & (df["pickup_longitude"] != 0)
        & (df["dropoff_latitude"] != 0)
        & (df["pickup_latitude"] != 0)
        & (df["pickup_longitude"] >= MinMax[0])
        & (df["pickup_longitude"] <= MinMax[1])
        & (df["dropoff_longitude"] >= MinMax[0])
        & (df["dropoff_longitude"] <= MinMax[1])
        & (df["pickup_latitude"] >= MinMax[2])
        & (df["pickup_latitude"] <= MinMax[3])
        & (df["dropoff_latitude"] >= MinMax[2])
        & (df["dropoff_latitude"] <= MinMax[3])
        & (df["fare_amount"] > 0)
        & (df["fare_amount"] <= 50)
        & (df["passenger_count"] > 0)
        & (df["passenger_count"] <= 6)
    )
    cleaned = df.loc[mask].copy()
    print(f" Cleaning: resulting size {len(cleaned)}")
    return cleaned


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df):
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df["year"] = df["pickup_datetime"].dt.year
    df["month"] = df["pickup_datetime"].dt.month
    df["day"] = df["pickup_datetime"].dt.day
    df["hour"] = df["pickup_datetime"].dt.hour
    df["weekday"] = df["pickup_datetime"].dt.weekday
    df["night"] = ((df["hour"] > 20) & (df["weekday"] < 5)).astype(int)
    df["late_night"] = (df["hour"] <= 3).astype(int)
    df["rush_hour"] = (
        (df["hour"] >= 16) & (df["hour"] <= 20) & (df["weekday"] < 5)
    ).astype(int)
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


trainKaggle = clean(trainKaggle)
testKaggle = clean(testKaggle)

trainKaggle = add_time_features(trainKaggle)
testKaggle = add_time_features(testKaggle)

trainKaggle = add_coordinate_features(trainKaggle)
testKaggle = add_coordinate_features(testKaggle)

trainKaggle = add_distances_features(trainKaggle)
testKaggle = add_distances_features(testKaggle)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'fare_amount'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_54/2048656387.py in <cell line: 0>()
     78 # Apply cleaning and feature engineering once on the full datasets
     79 trainKaggle = clean(trainKaggle)
---> 80 testKaggle = clean(testKaggle)
     81 
     82 trainKaggle = add_time_features(trainKaggle)

/tmp/ipykernel_54/2048656387.py in clean(df)
     19         & (df["dropoff_latitude"] >= MinMax[2])
     20         & (df["dropoff_latitude"] <= MinMax[3])
---> 21         & (df["fare_amount"] > 0)
     22         & (df["fare_amount"] <= 50)
     23         & (df["passenger_count"] > 0)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'fare_amount'

## === cell 3
train_df, temp_df = train_test_split(trainKaggle, test_size=0.50, random_state=1)
validation_df, test_df = train_test_split(train_df, test_size=0.10, random_state=1)
test_df = temp_df



## === cell 4
drop_cols = ["pickup_datetime"]
train_df = train_df.drop(columns=drop_cols)
validation_df = validation_df.drop(columns=drop_cols)
test_df = test_df.drop(columns=drop_cols)
testKaggle_clean = testKaggle.drop(columns=drop_cols + ["key"])



## === cell 5
train_labels = train_df["fare_amount"].values
validation_labels = validation_df["fare_amount"].values
test_labels = test_df["fare_amount"].values

train_labels_log = np.log1p(train_labels)
validation_labels_log = np.log1p(validation_labels)
test_labels_log = np.log1p(test_labels)

train_df = train_df.drop(columns=["fare_amount"])
validation_df = validation_df.drop(columns=["fare_amount"])
test_df = test_df.drop(columns=["fare_amount"])

print("Label extraction and log‑transform done")



## === cell 6
print("train shape:", train_df.shape)
print("validation shape:", validation_df.shape)
print("test shape:", test_df.shape)



## === cell 7
numeric_cols = train_df.select_dtypes(include=[np.number]).columns
scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df[numeric_cols])
validation_df_scaled = scaler.transform(validation_df[numeric_cols])
test_scaled = scaler.transform(test_df[numeric_cols])
testKaggle_scaled = scaler.transform(testKaggle_clean[numeric_cols])



## === cell 8
model = GradientBoostingRegressor(
    n_estimators=1200,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.9,
    random_state=42,
)
print("Training GradientBoostingRegressor on log‑target...")
model.fit(train_df_scaled, train_labels_log)
print("Training complete.")



## === cell 9
val_pred_log = model.predict(validation_df_scaled)
val_pred = np.expm1(val_pred_log)
val_rmse = np.sqrt(mean_squared_error(validation_labels, val_pred))
print(f"Validation RMSE (original scale): {val_rmse:.4f}")



## === cell 10
test_pred_log = model.predict(test_scaled)
test_pred = np.expm1(test_pred_log)
test_rmse = np.sqrt(mean_squared_error(test_labels, test_pred))
print(f"Internal test RMSE (original scale): {test_rmse:.4f}")



## === cell 11
predictionKaggle_log = model.predict(testKaggle_scaled).reshape(-1, 1)
predictionKaggle = np.expm1(predictionKaggle_log)
output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)



## === cell 12
print("Example prediction vs label (internal test set)")
print("Prediction[0]:", test_pred[0])
print("Label[0]:", test_labels[0])
