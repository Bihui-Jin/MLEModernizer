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

4.25412

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 15.22047) has done: 'Implemented fixes to resolve runtime errors and improve model performance:
- Updated `DATASET_SIZE` to use more training rows.
- Corrected optimizer creation (`optimizers.Adam` with `learning_rate` argument).
- Simplified `remove_datapoints_from_water` to bypass external image loading.
- Adjusted evaluation to use scaled test data.
- Fixed submission output to handle prediction shape.
- Guarded optional visualization import.
- Minor cleanup for consistency.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor


def resolve_path(*parts):
    base_paths = [
        os.path.join("input", "new-york-city-taxi-fare-prediction"),
        os.path.join("working", "new-york-city-taxi-fare-prediction"),
        os.path.join("data", "new-york-city-taxi-fare-prediction"),
    ]
    for base in base_paths:
        p = os.path.join(base, *parts)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Unable to locate {'/'.join(parts)} in any known directory."
    )


TRAIN_PATH = resolve_path("labels.csv")
TEST_PATH = resolve_path("test.csv")
SUBMISSION_NAME = "submission.csv"

DATASET_SIZE = 200_000  # sample size for quick runs; set to None for full data
EPOCHS = 5  # kept for compatibility, not used with RandomForest
BATCH_SIZE = 256  # kept for compatibility, not used with RandomForest
LEARNING_RATE = 1e-3  # kept for compatibility, not used with RandomForest



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/670130491.py in <cell line: 0>()
     23 
     24 
---> 25 TRAIN_PATH = resolve_path("labels.csv")
     26 TEST_PATH = resolve_path("test.csv")
     27 SUBMISSION_NAME = "submission.csv"

/tmp/ipykernel_11/670130491.py in resolve_path(*parts)
     18         if os.path.exists(p):
     19             return p
---> 20     raise FileNotFoundError(
     21         f"Unable to locate {'/'.join(parts)} in any known directory."
     22     )

FileNotFoundError: Unable to locate labels.csv in any known directory.

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
train_raw = pd.read_csv(
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
test_raw = pd.read_csv(
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



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4008476577.py in <cell line: 0>()
     10 }
     11 train_raw = pd.read_csv(
---> 12     TRAIN_PATH,
     13     nrows=DATASET_SIZE,
     14     dtype=datatypes,

NameError: name 'TRAIN_PATH' is not defined

## === cell 2
train_df, validation_df = train_test_split(train_raw, test_size=0.10, random_state=1)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3473191352.py in <cell line: 0>()
----> 1 train_df, validation_df = train_test_split(train_raw, test_size=0.10, random_state=1)
      2 
      3 

NameError: name 'train_raw' is not defined

## === cell 3
def remove_datapoints_from_water(df):
    return df


def clean(df):
    print(" Old size: %d" % len(df))
    df = df.dropna(how="any", axis="rows")
    print(" New size after dropna: %d" % len(df))

    df = df[
        (df["dropoff_longitude"] != df["pickup_longitude"])
        & (df["dropoff_latitude"] != df["pickup_latitude"])
    ]
    print(" New size after removing same long/lat: %d" % len(df))

    df = df[
        (df["dropoff_longitude"] != 0)
        & (df["pickup_longitude"] != 0)
        & (df["dropoff_latitude"] != 0)
        & (df["pickup_latitude"] != 0)
    ]
    print(" New size after removing zero coords: %d" % len(df))

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
    print(" New size after NYC bbox filter: %d" % len(df))

    df = df[(0 < df["fare_amount"]) & (df["fare_amount"] <= 50)]
    df = df[(df["passenger_count"] > 0) & (df["passenger_count"] <= 6)]
    print(" New size after fare/passenger filters: %d" % len(df))

    nyc_coord = (40.7141667, -74.0063889)
    fk_coord = (40.639722, -73.778889)
    ewr_coord = (40.6925, -74.168611)
    lga_coord = (40.77725, -73.872611)
    sol_coord = (40.6892, -74.0445)

    for coord in [nyc_coord, fk_coord, ewr_coord, lga_coord, sol_coord]:
        df = df[
            (coord[1] != df["pickup_longitude"]) | (coord[0] != df["pickup_latitude"])
        ]
        df = df[
            (coord[1] != df["dropoff_longitude"]) | (coord[0] != df["dropoff_latitude"])
        ]

    print(" New size after airport/landmark filters: %d" % len(df))
    print("Old size before water removal: %d" % len(df))
    df = remove_datapoints_from_water(df)
    print("New size after water removal (no‑op): %d" % len(df))
    return df




## === cell 4
print("Cleaning training set")
train_df = clean(train_df)
print("Cleaning validation set")
validation_df = clean(validation_df)
print("Cleaning test set")
test_df = clean(test_raw.copy())




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2647568076.py in <cell line: 0>()
      1 print("Cleaning training set")
----> 2 train_df = clean(train_df)
      3 print("Cleaning validation set")
      4 validation_df = clean(validation_df)
      5 print("Cleaning test set")

NameError: name 'train_df' is not defined

## === cell 5
def late_night(row):
    return 1 if (row["hour"] <= 3) else 0


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


for name, frame in [
    ("train", train_df),
    ("validation", validation_df),
    ("test", test_df),
]:
    print(f"Adding time features to {name}")
    frame = add_time_features(frame)
    print(f"Adding coordinate features to {name}")
    frame = add_coordinate_features(frame)
    print(f"Adding distance features to {name}")
    frame = add_distances_features(frame)
    if name == "train":
        train_df = frame
    elif name == "validation":
        validation_df = frame
    else:
        test_df = frame



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2234390937.py in <cell line: 0>()
     50 
     51 for name, frame in [
---> 52     ("train", train_df),
     53     ("validation", validation_df),
     54     ("test", test_df),

NameError: name 'train_df' is not defined

## === cell 6
drop_cols = ["key", "pickup_datetime"]
train_df = train_df.drop(columns=drop_cols)
validation_df = validation_df.drop(columns=drop_cols)
test_df = test_df.drop(columns=drop_cols)

test_keys = test_raw["key"].values



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1212040982.py in <cell line: 0>()
      1 drop_cols = ["key", "pickup_datetime"]
----> 2 train_df = train_df.drop(columns=drop_cols)
      3 validation_df = validation_df.drop(columns=drop_cols)
      4 test_df = test_df.drop(columns=drop_cols)
      5 

NameError: name 'train_df' is not defined

## === cell 7
train_labels = np.log1p(train_df["fare_amount"].values)
validation_labels = np.log1p(validation_df["fare_amount"].values)

train_df = train_df.drop(columns=["fare_amount"])
validation_df = validation_df.drop(columns=["fare_amount"])



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3276514090.py in <cell line: 0>()
----> 1 train_labels = np.log1p(train_df["fare_amount"].values)
      2 validation_labels = np.log1p(validation_df["fare_amount"].values)
      3 
      4 train_df = train_df.drop(columns=["fare_amount"])
      5 validation_df = validation_df.drop(columns=["fare_amount"])

NameError: name 'train_df' is not defined

## === cell 8
scaler = preprocessing.MinMaxScaler()
train_scaled = scaler.fit_transform(train_df)
validation_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_df)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3021952662.py in <cell line: 0>()
      1 scaler = preprocessing.MinMaxScaler()
----> 2 train_scaled = scaler.fit_transform(train_df)
      3 validation_scaled = scaler.transform(validation_df)
      4 test_scaled = scaler.transform(test_df)
      5 

NameError: name 'train_df' is not defined

## === cell 9
rf_model = RandomForestRegressor(
    n_estimators=120,
    max_depth=None,
    n_jobs=-1,
    random_state=42,
    verbose=0,
)
rf_model.fit(train_scaled, train_labels)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/953410006.py in <cell line: 0>()
      7     verbose=0,
      8 )
----> 9 rf_model.fit(train_scaled, train_labels)
     10 

NameError: name 'train_scaled' is not defined

## === cell 10
importances = rf_model.feature_importances_
print("Feature importances (top 5):")
for idx in np.argsort(importances)[-5:][::-1]:
    print(f"{train_df.columns[idx]:30s}: {importances[idx]:.4f}")



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/3745941769.py in <cell line: 0>()
      1 # Optional: display feature importance summary
----> 2 importances = rf_model.feature_importances_
      3 print("Feature importances (top 5):")
      4 for idx in np.argsort(importances)[-5:][::-1]:
      5     print(f"{train_df.columns[idx]:30s}: {importances[idx]:.4f}")

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_forest.py in feature_importances_(self)
    626             array of zeros.
    627         """
--> 628         check_is_fitted(self)
    629 
    630         all_importances = Parallel(n_jobs=self.n_jobs, prefer="threads")(

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_is_fitted(estimator, attributes, msg, all_or_any)
   1388 
   1389     if not fitted:
-> 1390         raise NotFittedError(msg % {"name": type(estimator).__name__})
   1391 
   1392 

NotFittedError: This RandomForestRegressor instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.

## === cell 11
pred_log = rf_model.predict(test_scaled)
pred = np.expm1(pred_log).ravel()
submission = pd.DataFrame({"key": test_keys, "fare_amount": pred})
submission.to_csv(SUBMISSION_NAME, index=False)
print(f"Submission written to {SUBMISSION_NAME}")

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/897679647.py in <cell line: 0>()
----> 1 pred_log = rf_model.predict(test_scaled)
      2 pred = np.expm1(pred_log).ravel()
      3 submission = pd.DataFrame({"key": test_keys, "fare_amount": pred})
      4 submission.to_csv(SUBMISSION_NAME, index=False)
      5 print(f"Submission written to {SUBMISSION_NAME}")

NameError: name 'test_scaled' is not defined
