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
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 3 other files
        input/
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 3 other files
        working/
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 3 other files
```

-> data/new-york-city-taxi-fare-prediction/labels.csv has 55413942 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/new-york-city-taxi-fare-prediction/sample_submission.csv has 9914 rows and 2 columns.
The columns are: key, fare_amount

-> data/new-york-city-taxi-fare-prediction/test.csv has 9914 rows and 7 columns.
The columns are: key, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> input/new-york-city-taxi-fare-prediction/labels.csv has 55413942 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> input/new-york-city-taxi-fare-prediction/sample_submission.csv has 9914 rows and 2 columns.
The columns are: key, fare_amount

-> input/new-york-city-taxi-fare-prediction/test.csv has 9914 rows and 7 columns.
The columns are: key, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> working/new-york-city-taxi-fare-prediction/labels.csv has 55413942 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> (stopped after 10 files for performance)

# 5. Target score

4.11503

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 423.69379) has done: 'I replace the problematic imports with TensorFlow‑Keras equivalents, simplify the water‑mask cleaning to avoid external image loading, fix the Adam optimizer call, and guard the optional visualization import. These minimal changes resolve the runtime errors and let the model train, producing a valid *.csv* submission while keeping the original modelling logic unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor


def locate_file(*candidates):
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"None of the candidate paths exist: {candidates}")


BASE_DIRS = [
    "input/new-york-city-taxi-fare-prediction",
    "data/new-york-city-taxi-fare-prediction",
    "working/new-york-city-taxi-fare-prediction",
]

TRAIN_PATH = locate_file(
    os.path.join(BASE_DIRS[0], "labels.csv"),
    os.path.join(BASE_DIRS[1], "labels.csv"),
    os.path.join(BASE_DIRS[2], "labels.csv"),
)

TEST_PATH = locate_file(
    os.path.join(BASE_DIRS[0], "test.csv"),
    os.path.join(BASE_DIRS[1], "test.csv"),
    os.path.join(BASE_DIRS[2], "test.csv"),
)

SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 0.001
DATASET_SIZE = 80000  # manageable subset for quick runs



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1618893495.py in <cell line: 0>()
     23 ]
     24 
---> 25 TRAIN_PATH = locate_file(
     26     os.path.join(BASE_DIRS[0], "labels.csv"),
     27     os.path.join(BASE_DIRS[1], "labels.csv"),

/tmp/ipykernel_11/1618893495.py in locate_file(*candidates)
     14         if os.path.exists(path):
     15             return path
---> 16     raise FileNotFoundError(f"None of the candidate paths exist: {candidates}")
     17 
     18 

FileNotFoundError: None of the candidate paths exist: ('input/new-york-city-taxi-fare-prediction/labels.csv', 'data/new-york-city-taxi-fare-prediction/labels.csv', 'working/new-york-city-taxi-fare-prediction/labels.csv')

## === cell 1
trainKaggle = pd.read_csv(
    TRAIN_PATH,
    nrows=DATASET_SIZE,
    usecols=[
        "fare_amount",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
    dtype={
        "fare_amount": "float32",
        "pickup_datetime": "str",
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "int8",
    },
)

testKaggle = pd.read_csv(TEST_PATH, dtype={"key": "str"})



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/18234675.py in <cell line: 0>()
      1 # Load a subset of the training data (the full set is too large for this environment)
      2 trainKaggle = pd.read_csv(
----> 3     TRAIN_PATH,
      4     nrows=DATASET_SIZE,
      5     usecols=[

NameError: name 'TRAIN_PATH' is not defined

## === cell 2
train_df, test_df = train_test_split(trainKaggle, test_size=0.50, random_state=1)
train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=1)

print(f"Internal train size: {len(train_df)}")
print(f"Internal validation size: {len(validation_df)}")
print(f"Internal test size: {len(test_df)}")
print(f"Kaggle test size: {len(testKaggle)}")


def clean(df):
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
    return df


def late_night(row):
    return 1 if row["hour"] <= 3 else 0


def night(row):
    return 1 if (row["hour"] > 20) and (row["weekday"] < 5) else 0


def rush_hour(row):
    return 1 if (16 <= row["hour"] <= 20) and (row["weekday"] < 5) else 0


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


for name, dataset in [
    ("train", train_df),
    ("validation", validation_df),
    ("test_internal", test_df),
    ("test_kaggle", testKaggle),
]:
    if name != "test_kaggle":
        dataset = clean(dataset)
    dataset = add_time_features(dataset)
    dataset = add_coordinate_features(dataset)
    dataset = add_distances_features(dataset)
    if name == "train":
        train_df = dataset
    elif name == "validation":
        validation_df = dataset
    elif name == "test_internal":
        test_df = dataset
    else:
        testKaggle = dataset



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3810415099.py in <cell line: 0>()
      1 # Split training data into train / validation / internal test
----> 2 train_df, test_df = train_test_split(trainKaggle, test_size=0.50, random_state=1)
      3 train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=1)
      4 
      5 print(f"Internal train size: {len(train_df)}")

NameError: name 'trainKaggle' is not defined

## === cell 3
dropped_columns = ["passenger_count", "pickup_datetime"]
train_df = train_df.drop(columns=dropped_columns)
validation_df = validation_df.drop(columns=dropped_columns)
test_df = test_df.drop(columns=dropped_columns)
testKaggle_clean = testKaggle.drop(columns=dropped_columns + ["key"])

train_labels = train_df["fare_amount"].values
validation_labels = validation_df["fare_amount"].values

train_features = train_df.drop(columns=["fare_amount"])
validation_features = validation_df.drop(columns=["fare_amount"])
test_features = test_df.drop(columns=["fare_amount"])

scaler = preprocessing.MinMaxScaler()
train_scaled = scaler.fit_transform(train_features)
validation_scaled = scaler.transform(validation_features)
test_scaled = scaler.transform(test_features)
testKaggle_scaled = scaler.transform(testKaggle_clean)


def rmse(y_true, y_pred):
    return np.sqrt(metrics.mean_squared_error(y_true, y_pred))




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2372813445.py in <cell line: 0>()
      1 # Drop columns that are not used for modeling
      2 dropped_columns = ["passenger_count", "pickup_datetime"]
----> 3 train_df = train_df.drop(columns=dropped_columns)
      4 validation_df = validation_df.drop(columns=dropped_columns)
      5 test_df = test_df.drop(columns=dropped_columns)

NameError: name 'train_df' is not defined

## === cell 4
gbr = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
)
gbr.fit(train_scaled, train_labels)

val_pred = gbr.predict(validation_scaled)
val_rmse = rmse(validation_labels, val_pred)
print(f"Validation RMSE: {val_rmse:.4f}")




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3030110834.py in <cell line: 0>()
      6     random_state=42,
      7 )
----> 8 gbr.fit(train_scaled, train_labels)
      9 
     10 # Validation performance

NameError: name 'train_scaled' is not defined

## === cell 5
def output_submission(raw_test, predictions, id_column, prediction_column, file_name):
    df = pd.DataFrame(
        {id_column: raw_test[id_column].values, prediction_column: predictions.ravel()}
    )
    df.to_csv(file_name, index=False)
    print(f"Submission written to {file_name}")


internal_pred = gbr.predict(test_scaled)
print("Sample internal predictions (first 5):", internal_pred[:5])

kaggle_pred = gbr.predict(testKaggle_scaled)
output_submission(testKaggle, kaggle_pred, "key", "fare_amount", SUBMISSION_NAME)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/559450263.py in <cell line: 0>()
      8 
      9 # Predict on internal test set (optional inspection)
---> 10 internal_pred = gbr.predict(test_scaled)
     11 print("Sample internal predictions (first 5):", internal_pred[:5])
     12 

NameError: name 'test_scaled' is not defined

## === cell 6
plt.figure(figsize=(6, 4))
plt.title("Feature Importances")
plt.barh(range(len(gbr.feature_importances_)), gbr.feature_importances_)
plt.yticks(range(len(gbr.feature_importances_)), train_features.columns)
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/3610760489.py in <cell line: 0>()
      2 plt.figure(figsize=(6, 4))
      3 plt.title("Feature Importances")
----> 4 plt.barh(range(len(gbr.feature_importances_)), gbr.feature_importances_)
      5 plt.yticks(range(len(gbr.feature_importances_)), train_features.columns)
      6 plt.xlabel("Importance")

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_gb.py in feature_importances_(self)
    742             array of zeros.
    743         """
--> 744         self._check_initialized()
    745 
    746         relevant_trees = [

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_gb.py in _check_initialized(self)
    380     def _check_initialized(self):
    381         """Check that the estimator is initialized, raising an error if not."""
--> 382         check_is_fitted(self)
    383 
    384     def fit(self, X, y, sample_weight=None, monitor=None):

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_is_fitted(estimator, attributes, msg, all_or_any)
   1388 
   1389     if not fitted:
-> 1390         raise NotFittedError(msg % {"name": type(estimator).__name__})
   1391 
   1392 

NotFittedError: This GradientBoostingRegressor instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.
