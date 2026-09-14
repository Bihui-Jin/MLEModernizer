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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0
xgboost==2.0.3

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

3.98929

# 6. Current score

7.09168

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.62664) has done: 'I fix the runtime error by calling `XGBRegressor.predict(X_test)` (the scikit-learn API doesn’t accept `data=`), which also unblock creation of `Y_pred` and prevent the downstream `NameError`. I also ensure `Y_train` is a 1D Series (not a single-column DataFrame) so XGBoost trains/predicts consistently. Finally, I write the submission to a new filename (to avoid overwriting Kaggle’s `sample_submission.csv`) and guarantee the output has exactly the required columns `key,fare_amount`, producing a valid `.csv` submission end-to-end.'
- What this solution (achieved 7.09168) has done: 'Your current RMSE (4.62664) is worse than the target (3.98929), so we should make small, score-relevant improvements without changing the overall approach (still XGBRegressor on the same engineered features). The biggest low-risk gains here come from (1) cleaning obviously bad training rows (invalid lat/lon, non-positive/unrealistic fares, passenger_count out of range), because NYC Taxi Fare is very sensitive to outliers, and (2) adding just a couple of standard, minimal XGBoost hyperparameters (more trees + smaller learning rate) to better fit the cleaned data while staying within time. I also ensure train/test columns align and clamp negative predictions to 0 (fares can’t be negative), which typically improves RMSE a bit. These changes keep your feature set and training flow intact while moving the score toward the target.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

BASE_INPUT = "/kaggle/input"
if not os.path.exists(BASE_INPUT):
    BASE_INPUT = "../input"

print("BASE_INPUT:", BASE_INPUT)
print("Top-level input dirs:", os.listdir(BASE_INPUT)[:20])

COMP_DIR = os.path.join(BASE_INPUT, "new-york-city-taxi-fare-prediction")
if not os.path.exists(COMP_DIR):
    COMP_DIR = BASE_INPUT

train_path = os.path.join(COMP_DIR, "train.csv")
test_path = os.path.join(COMP_DIR, "test.csv")
sample_path = os.path.join(COMP_DIR, "sample_submission.csv")

print("Resolved paths:")
print("train:", train_path)
print("test:", test_path)
print("sample_submission:", sample_path)



## === cell 1
training_data = pd.read_csv(
    train_path,
    nrows=2_000_000,
    parse_dates=["pickup_datetime"],
)
test_data = pd.read_csv(
    test_path,
    parse_dates=["pickup_datetime"],
)

print("train shape:", training_data.shape)
print("test shape:", test_data.shape)



## === cell 2
training_data



## === cell 3
X_train = training_data.copy()
X_test = test_data.copy()
Y_train = training_data.copy()



## === cell 4
mask = (
    X_train["fare_amount"].notna()
    & X_train["pickup_longitude"].between(-75, -72)
    & X_train["dropoff_longitude"].between(-75, -72)
    & X_train["pickup_latitude"].between(40, 42)
    & X_train["dropoff_latitude"].between(40, 42)
    & X_train["passenger_count"].between(1, 6)
    & X_train["fare_amount"].between(2.5, 250.0)
)
training_data = training_data.loc[mask].reset_index(drop=True)

X_train = training_data.copy()
Y_train = training_data["fare_amount"].copy()

print("after cleaning train shape:", X_train.shape)



## === cell 5
X_train["hour"] = X_train["pickup_datetime"].dt.hour
X_train["dayofweek"] = X_train["pickup_datetime"].dt.dayofweek
X_train["latitude_distance"] = (
    X_train["dropoff_latitude"] - X_train["pickup_latitude"]
).abs()
X_train["longitude_distance"] = (
    X_train["dropoff_longitude"] - X_train["pickup_longitude"]
).abs()

X_train = X_train.drop(
    columns=[
        "dropoff_longitude",
        "dropoff_latitude",
        "key",
        "fare_amount",
        "pickup_datetime",
    ]
)



## === cell 6
X_test["hour"] = X_test["pickup_datetime"].dt.hour
X_test["dayofweek"] = X_test["pickup_datetime"].dt.dayofweek
X_test["latitude_distance"] = (
    X_test["dropoff_latitude"] - X_test["pickup_latitude"]
).abs()
X_test["longitude_distance"] = (
    X_test["dropoff_longitude"] - X_test["pickup_longitude"]
).abs()

X_test = X_test.drop(
    columns=["dropoff_longitude", "dropoff_latitude", "key", "pickup_datetime"]
)

X_test = X_test[X_train.columns]



## === cell 7
"""
from keras import models
from keras import layers
from keras import optimizers
from keras.layers import Dropout

model=models.Sequential()
model.add(layers.Dense(512,activation='relu',input_shape=(X_train.shape[1],)))
model.add(Dropout(0.2))
model.add(layers.Dense(512,activation='relu'))
model.add(Dropout(0.2))
model.add(layers.Dense(1))

rmsprop=optimizers.RMSprop(lr=0.001)

model.compile(optimizer=rmsprop,loss='mse',metrics=['mae'])

model.fit(X_train,Y_train,epochs=4,batch_size=512)

Y_pred=model.predict(X_test)
"""



## === cell 8
import xgboost as xgb

model = xgb.XGBRegressor(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
)
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

Y_pred = np.clip(Y_pred, 0, None)



## === cell 9
submission = pd.DataFrame({"key": test_data["key"], "fare_amount": Y_pred})

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print(submission.head())
print("Columns:", submission.columns.tolist())
print("Rows:", len(submission))
