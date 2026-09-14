# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
import matplotlib.pyplot as plt


def locate_file(name):
    """Search for *name* under the current working directory and return its full path."""
    for root, _, files in os.walk(os.path.abspath(".")):
        if name in files:
            return os.path.join(root, name)
    raise FileNotFoundError(f"Unable to locate {name}")


TRAIN_PATH = locate_file("train.csv")
TEST_PATH = locate_file("test.csv")
SUBMISSION_NAME = "submission.csv"

MAX_ROWS = 250_000  # adjust as needed for the environment
train_file_size = os.path.getsize(TRAIN_PATH)
read_nrows = MAX_ROWS if train_file_size > 150_000_000 else None  # ~150 MB threshold

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
trainKaggle = pd.read_csv(TRAIN_PATH, dtype=datatypes, nrows=read_nrows)
testKaggle = pd.read_csv(TEST_PATH, dtype=datatypes)



## === cell 1
train_df, validation_df = train_test_split(trainKaggle, test_size=0.10, random_state=1)

print("train_df Size %d" % len(train_df))
print("validation_df Size %d" % len(validation_df))




## === cell 2
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

    df = df[(0.99 < df["fare_amount"]) & (df["fare_amount"] <= 50)]
    print(" New size after removing outliers: %d" % len(df))

    df = df[(df["passenger_count"] > 0) & (df["passenger_count"] <= 6)]
    print(" New size after passenger filter: %d" % len(df))

    print("Skipping water‑mask filter")
    return df


def late_night(row):
    return 1 if (0 <= row["hour"] <= 3) else 0


def night(row):
    return 1 if (row["hour"] > 20 and row["hour"] > 0 and row["weekday"] < 5) else 0


def rush_hour(row):
    return 1 if (16 <= row["hour"] <= 20 and row["weekday"] < 5) else 0


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
    return df


def add_haversine_features(df):
    R = 6371.0
    lat1 = np.radians(df["pickup_latitude"])
    lon1 = np.radians(df["pickup_longitude"])
    lat2 = np.radians(df["dropoff_latitude"])
    lon2 = np.radians(df["dropoff_longitude"])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    df["haversine"] = R * c
    return df


def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    df = pd.DataFrame(
        {id_column: raw_test[id_column], prediction_column: prediction.ravel()}
    )
    df.to_csv(file_name, index=False)
    print("Output complete")




## === cell 3
print("train_df clean")
train_df = clean(train_df)
print("validation_df clean")
validation_df = clean(validation_df)



## === cell 4
print("train_df add_time_features")
train_df = add_time_features(train_df)
print("validation_df add_time_features")
validation_df = add_time_features(validation_df)
print("testKaggle add_time_features")
testKaggle = add_time_features(testKaggle)



## === cell 5
print("train_df add_coordinate_features")
train_df = add_coordinate_features(train_df)
print("validation_df add_coordinate_features")
validation_df = add_coordinate_features(validation_df)
print("testKaggle add_coordinate_features")
testKaggle = add_coordinate_features(testKaggle)



## === cell 6
print("train_df add_distances_features")
train_df = add_distances_features(train_df)
print("validation_df add_distances_features")
validation_df = add_distances_features(validation_df)
print("testKaggle add_distances_features")
testKaggle = add_distances_features(testKaggle)

print("train_df add_haversine_features")
train_df = add_haversine_features(train_df)
print("validation_df add_haversine_features")
validation_df = add_haversine_features(validation_df)
print("testKaggle add_haversine_features")
testKaggle = add_haversine_features(testKaggle)

print("Done with Adding features")



## === cell 7
dropped_columns = [
    "key",
    "pickup_datetime",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_longitude",
    "dropoff_latitude",
]

train_df = train_df.drop(dropped_columns, axis=1)
validation_df = validation_df.drop(dropped_columns, axis=1)
testKaggle_clean = testKaggle.drop(dropped_columns, axis=1)

print("Done with dropped_columns")



## === cell 8
train_labels_original = train_df["fare_amount"].values
validation_labels_original = validation_df["fare_amount"].values

train_labels = np.log1p(train_labels_original)
validation_labels = np.log1p(validation_labels_original)

train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)

testKaggle_features = testKaggle_clean.values

train_features = train_df.values
validation_features = validation_df.values
test_features = testKaggle_features  # reuse for optional printing

rf = RandomForestRegressor(
    n_estimators=800,  # reduced a bit for speed on the sampled data
    max_depth=None,
    max_features=1.0,
    random_state=42,
    n_jobs=-1,
    min_samples_split=2,
    min_samples_leaf=1,
)
rf.fit(train_features, train_labels)
rf_val_pred_log = rf.predict(validation_features)
rf_val_pred = np.expm1(rf_val_pred_log)
rf_val_rmse = np.sqrt(np.mean((rf_val_pred - validation_labels_original) ** 2))
print(f"Validation RMSE (RandomForest): {rf_val_rmse:.4f}")

gbr = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    random_state=42,
)
gbr.fit(train_features, train_labels)
gbr_val_pred = np.expm1(gbr.predict(validation_features))
gbr_val_rmse = np.sqrt(np.mean((gbr_val_pred - validation_labels_original) ** 2))
print(f"Validation RMSE (GradientBoosting): {gbr_val_rmse:.4f}")

etr = ExtraTreesRegressor(
    n_estimators=800,
    max_depth=None,
    max_features=0.8,
    random_state=42,
    n_jobs=-1,
    min_samples_split=2,
    min_samples_leaf=1,
)
etr.fit(train_features, train_labels)
etr_val_pred = np.expm1(etr.predict(validation_features))
etr_val_rmse = np.sqrt(np.mean((etr_val_pred - validation_labels_original) ** 2))
print(f"Validation RMSE (ExtraTrees): {etr_val_rmse:.4f}")

hgb = HistGradientBoostingRegressor(
    max_iter=400,
    learning_rate=0.05,
    max_depth=None,
    random_state=42,
)
hgb.fit(train_features, train_labels)
hgb_val_pred = np.expm1(hgb.predict(validation_features))
hgb_val_rmse = np.sqrt(np.mean((hgb_val_pred - validation_labels_original) ** 2))
print(f"Validation RMSE (HistGradientBoosting): {hgb_val_rmse:.4f}")

best_model = rf
best_name = "RandomForest"
best_rmse = rf_val_rmse

if gbr_val_rmse < best_rmse:
    best_model = gbr
    best_name = "GradientBoosting"
    best_rmse = gbr_val_rmse

if etr_val_rmse < best_rmse:
    best_model = etr
    best_name = "ExtraTrees"
    best_rmse = etr_val_rmse

if hgb_val_rmse < best_rmse:
    best_model = hgb
    best_name = "HistGradientBoosting"
    best_rmse = hgb_val_rmse

print(f"Selected model: {best_name} with Validation RMSE: {best_rmse:.4f}")



## === cell 9
try:
    plt.figure(figsize=(20, 10))
    plt.title(f"{best_name} - no loss curve")
    plt.show()
except Exception as e:
    print("Plotting error:", e)



## === cell 10
prediction_log = best_model.predict(test_features)
prediction = np.expm1(prediction_log).reshape(-1, 1)



## === cell 11
output_submission(testKaggle, prediction, "key", "fare_amount", SUBMISSION_NAME)



## === cell 12
print("Sample predictions (first 5):")
print(prediction[:5].ravel())
