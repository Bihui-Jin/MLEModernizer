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

3.11

# 3. Installed packages

geopandas==0.14.4
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
seaborn==0.12.2
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

3.7787

# 6. Current score

5.09489

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 5.09489) has done: 'You’re dropping rows from `test` during cleaning, which makes `new_pred` shorter than the original `key` list and causes the length-mismatch error when building the submission. I keep the same XGBoost regressor and the same feature set, but change test-time cleaning to *not drop rows*; instead we flag invalid rows and still produce predictions for all 9914 keys (filling invalid rows with a simple fallback), so the submission is always aligned. I also fix the incorrect `StandardScaler` usage (you were fitting but never transforming your train/test matrices properly) in a minimal way that doesn’t change the core model. Finally, I write a valid `submission1.csv` with exactly `key` and `fare_amount` columns.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as MSE



## === cell 2
train = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/train.csv", nrows=1000000
)
test = pd.read_csv("../input/new-york-city-taxi-fare-prediction/test.csv")
train.shape, test.shape



## === cell 3
train.head()



## === cell 4
train.isnull().sum()



## === cell 5
train = train.dropna(axis="rows")
test.isnull().sum()



## === cell 6
train.head()



## === cell 7
train["fare_amount"].describe()



## === cell 8
train.drop(train[train["pickup_longitude"] == 0].index, axis=0, inplace=True)
train.drop(train[train["pickup_latitude"] == 0].index, axis=0, inplace=True)
train.drop(train[train["dropoff_longitude"] == 0].index, axis=0, inplace=True)
train.drop(train[train["dropoff_latitude"] == 0].index, axis=0, inplace=True)
train.drop(train[train["passenger_count"] > 5].index, axis=0, inplace=True)
train.drop(train[train["passenger_count"] == 0].index, axis=0, inplace=True)



## === cell 9
train.drop(["key"], axis=1, inplace=True)



## === cell 10
train.drop(["pickup_datetime"], axis=1, inplace=True)



## === cell 11
train.dropna(inplace=True)

train.drop(
    train.index[
        (train.pickup_longitude < -75)
        | (train.pickup_longitude > -72)
        | (train.pickup_latitude < 40)
        | (train.pickup_latitude > 42)
    ],
    inplace=True,
)
train.drop(
    train.index[
        (train.dropoff_longitude < -75)
        | (train.dropoff_longitude > -72)
        | (train.dropoff_latitude < 40)
        | (train.dropoff_latitude > 42)
    ],
    inplace=True,
)



## === cell 12
X, y = train.drop("fare_amount", axis=1), train["fare_amount"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=12
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



## === cell 13
xgb_r = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=400,
    seed=123,
)
xgb_r.fit(X_train_scaled, y_train)



## === cell 14
y_pred = xgb_r.predict(X_test_scaled)
rmse = np.sqrt(MSE(y_test, y_pred))
print("RMSE : % f" % (rmse))



## === cell 15
test_raw = pd.read_csv("../input/new-york-city-taxi-fare-prediction/test.csv")
test_feat = test_raw.copy()

test_feat.drop(["pickup_datetime"], axis=1, inplace=True)

keys = test_feat["key"].copy()
test_feat.drop(["key"], axis=1, inplace=True)



## === cell 16
required_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
for c in required_cols:
    if c not in test_feat.columns:
        raise ValueError(f"Missing expected column in test: {c}")

invalid = (
    test_feat[required_cols].isna().any(axis=1)
    | (test_feat["pickup_longitude"] == 0)
    | (test_feat["pickup_latitude"] == 0)
    | (test_feat["dropoff_longitude"] == 0)
    | (test_feat["dropoff_latitude"] == 0)
    | (test_feat["passenger_count"] > 5)
    | (test_feat["passenger_count"] == 0)
    | (test_feat["pickup_longitude"] < -75)
    | (test_feat["pickup_longitude"] > -72)
    | (test_feat["pickup_latitude"] < 40)
    | (test_feat["pickup_latitude"] > 42)
    | (test_feat["dropoff_longitude"] < -75)
    | (test_feat["dropoff_longitude"] > -72)
    | (test_feat["dropoff_latitude"] < 40)
    | (test_feat["dropoff_latitude"] > 42)
)

train_feature_medians = X_train.median(numeric_only=True)
test_feat_filled = test_feat.copy()
test_feat_filled = test_feat_filled.fillna(train_feature_medians)

for col in test_feat_filled.columns:
    if col in train_feature_medians.index:
        test_feat_filled.loc[invalid, col] = train_feature_medians[col]

test_scaled = scaler.transform(test_feat_filled)



## === cell 17
new_pred = xgb_r.predict(test_scaled)

fallback_fare = float(np.median(y_train.values))
new_pred = np.asarray(new_pred, dtype=float)
new_pred[invalid.values] = fallback_fare



## === cell 18
submission = pd.DataFrame({"key": keys.values, "fare_amount": new_pred})
submission.head()



## === cell 19
submission.to_csv("submission1.csv", index=False)
print("Wrote submission1.csv with shape:", submission.shape)
print("Columns:", submission.columns.tolist())
