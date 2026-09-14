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

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 144578.81059) has done: 'The changes fix the import conflict that caused the protobuf error, remove the non‑numeric `key` column from the training features (so the scaler can operate), and keep the numeric `passenger_count` column in both train and test sets. These fixes allow the data preprocessing, model building, training, and submission generation to run end‑to‑end, producing a valid `submission.csv` while preserving the original modeling logic.'
- What this solution (achieved 979.72693) has done: 'Implemented fixes to resolve the protobuf import error by switching to the standalone `keras` library, increased training epochs for better model convergence, and added early stopping to prevent over‑training. Minor adjustments to optimizer imports ensure compatibility, while retaining the original feature engineering and model architecture. The script now runs end‑to‑end and writes a proper `submission.csv` ready for Kaggle.'
- What this solution (achieved 2203.58063) has done: 'I align the test feature columns to exactly match the training feature order before scaling, preventing mismatched inputs that cause huge prediction errors. This small change keeps all core logic unchanged while ensuring the model receives correct data, which should lower the RMSE toward the target.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearnex import patch

patch()  # patches scikit‑learn estimators in place




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_55/32252109.py in <cell line: 0>()
      8 # Enable Intel® Extension for Scikit‑Learn to accelerate estimators
      9 # (pure speed‑up, no change to algorithmic logic or results)
---> 10 from sklearnex import patch
     11 
     12 patch()  # patches scikit‑learn estimators in place

ImportError: cannot import name 'patch' from 'sklearnex' (/usr/local/lib/python3.11/dist-packages/sklearnex/__init__.py)

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
    return 1 if (row["hour"] <= 6) or (row["hour"] >= 20) else 0


def night(row):
    return (
        1
        if ((row["hour"] <= 20) and (row["hour"] >= 16)) and (row["weekday"] < 5)
        else 0
    )


def process(df):
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["year"] = df["pickup_datetime"].dt.year
    df["month"] = df["pickup_datetime"].dt.month
    df["day"] = df["pickup_datetime"].dt.day
    df["hour"] = df["pickup_datetime"].dt.hour
    df["weekday"] = df["pickup_datetime"].dt.weekday

    df["night"] = (
        (df["hour"] >= 16) & (df["hour"] <= 20) & (df["weekday"] < 5)
    ).astype(int)
    df["late_night"] = ((df["hour"] <= 6) | (df["hour"] >= 20)).astype(int)

    df["pickup_longitude_binned"] = pd.qcut(df["pickup_longitude"], 16, labels=False)
    df["dropoff_longitude_binned"] = pd.qcut(df["dropoff_longitude"], 16, labels=False)
    df["pickup_latitude_binned"] = pd.qcut(df["pickup_latitude"], 16, labels=False)
    df["dropoff_latitude_binned"] = pd.qcut(df["dropoff_latitude"], 16, labels=False)

    df = df.drop("pickup_datetime", axis=1)
    return df


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_relevant_distances(df):
    ny = (-74.0063889, 40.7141667)
    jfk = (-73.7822222222, 40.6441666667)
    ewr = (-74.175, 40.69)
    lgr = (-73.87, 40.77)
    df["downtown_pickup_distance"] = manhattan(
        ny[1], ny[0], df["pickup_latitude"], df["pickup_longitude"]
    )
    df["downtown_dropoff_distance"] = manhattan(
        ny[1], ny[0], df["dropoff_latitude"], df["dropoff_longitude"]
    )
    df["jfk_pickup_distance"] = manhattan(
        jfk[1], jfk[0], df["pickup_latitude"], df["pickup_longitude"]
    )
    df["jfk_dropoff_distance"] = manhattan(
        jfk[1], jfk[0], df["dropoff_latitude"], df["dropoff_longitude"]
    )
    df["ewr_pickup_distance"] = manhattan(
        ewr[1], ewr[0], df["pickup_latitude"], df["pickup_longitude"]
    )
    df["ewr_dropoff_distance"] = manhattan(
        ewr[1], ewr[0], df["dropoff_latitude"], df["dropoff_longitude"]
    )
    df["lgr_pickup_distance"] = manhattan(
        lgr[1], lgr[0], df["pickup_latitude"], df["pickup_longitude"]
    )
    df["lgr_dropoff_distance"] = manhattan(
        lgr[1], lgr[0], df["dropoff_latitude"], df["dropoff_longitude"]
    )
    return df


def add_engineered(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]

    latdiff = lat1 - lat2
    londiff = lon1 - lon2
    euclidean = np.sqrt(latdiff**2 + londiff**2)

    df["latdiff"] = latdiff
    df["londiff"] = londiff
    df["euclidean"] = euclidean
    df["manhattan"] = manhattan(lat1, lon1, lat2, lon2)

    df = pd.get_dummies(df, columns=["weekday"])
    df = pd.get_dummies(df, columns=["month"])
    return df




## === cell 2
BASE_INPUT = "/kaggle/input/new-york-city-taxi-fare-prediction"
TRAIN_PATH = os.path.join(BASE_INPUT, "train.csv")
TEST_PATH = os.path.join(BASE_INPUT, "test.csv")
SUBMISSION_NAME = "submission.csv"

BATCH_SIZE = 256
EPOCHS = 60  # retained for reference; not used by sklearn model
LEARNING_RATE = 0.0005  # retained for reference
DATASET_SIZE = 700000




## === cell 3
train_dtypes = {
    "key": "str",
    "fare_amount": "float32",
    "pickup_datetime": "str",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}
test_dtypes = {
    "key": "str",
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
    dtype=train_dtypes,
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
test = pd.read_csv(TEST_PATH, dtype=test_dtypes)




## === cell 4
train = clean(train)

train = process(train)
test = process(test)

train = add_relevant_distances(train)
test = add_relevant_distances(test)

train = add_engineered(train)
test = add_engineered(test)




## === cell 5
dropped_columns = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]
train_clean = train.drop(dropped_columns + ["key"], axis=1)
test_clean = test.drop(dropped_columns + ["key"], axis=1)




## === cell 6
feature_cols = [c for c in train_clean.columns if c != "fare_amount"]
missing_in_test = set(feature_cols) - set(test_clean.columns)
for col in missing_in_test:
    test_clean[col] = 0
extra_in_test = set(test_clean.columns) - set(feature_cols)
test_clean = test_clean.drop(columns=extra_in_test)

test_clean = test_clean[train_clean.columns.drop("fare_amount")]

train_df, validation_df = train_test_split(train_clean, test_size=0.10, random_state=1)

train_labels = train_df["fare_amount"].values
validation_labels = validation_df["fare_amount"].values
train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)

scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df).astype(np.float32)
validation_df_scaled = scaler.transform(validation_df).astype(np.float32)
test_scaled = scaler.transform(test_clean).astype(np.float32)




## === cell 7
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42,
    presort=False,
)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1300215164.py in <cell line: 0>()
      3 # presort=False avoids expensive sorting of the data at each iteration,
      4 # yielding a speed‑up without affecting the model’s predictions.
----> 5 model = GradientBoostingRegressor(
      6     n_estimators=300,
      7     learning_rate=0.05,

TypeError: GradientBoostingRegressor.__init__() got an unexpected keyword argument 'presort'

## === cell 8
print("Dataset size:", DATASET_SIZE)
print("Training rows:", train_df.shape[0])
print("Validation rows:", validation_df.shape[0])
print("Input dimension:", train_df_scaled.shape[1])
print("Features used:", train_df.columns.tolist())




## === cell 9
print("Model configuration:")
print(model)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/40655017.py in <cell line: 0>()
      1 print("Model configuration:")
----> 2 print(model)
      3 
      4 

NameError: name 'model' is not defined

## === cell 10
model.fit(train_df_scaled, train_labels)




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1636543319.py in <cell line: 0>()
----> 1 model.fit(train_df_scaled, train_labels)
      2 
      3 

NameError: name 'model' is not defined

## === cell 11
from sklearn.metrics import mean_squared_error

val_pred = model.predict(validation_df_scaled)
val_rmse = np.sqrt(mean_squared_error(validation_labels, val_pred))
print(f"Validation RMSE: {val_rmse:.4f}")




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1986565892.py in <cell line: 0>()
      1 from sklearn.metrics import mean_squared_error
      2 
----> 3 val_pred = model.predict(validation_df_scaled)
      4 val_rmse = np.sqrt(mean_squared_error(validation_labels, val_pred))
      5 print(f"Validation RMSE: {val_rmse:.4f}")

NameError: name 'model' is not defined

## === cell 12
prediction = model.predict(test_scaled).reshape(-1, 1)




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1682282045.py in <cell line: 0>()
----> 1 prediction = model.predict(test_scaled).reshape(-1, 1)
      2 
      3 

NameError: name 'model' is not defined

## === cell 13
def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    pred = prediction.reshape(-1)
    df = pd.DataFrame({id_column: raw_test[id_column].values, prediction_column: pred})
    df.to_csv(file_name, index=False)
    print(f"Submission written to {file_name}")


output_submission(test, prediction, "key", "fare_amount", SUBMISSION_NAME)

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4089490570.py in <cell line: 0>()
      6 
      7 
----> 8 output_submission(test, prediction, "key", "fare_amount", SUBMISSION_NAME)

NameError: name 'prediction' is not defined
