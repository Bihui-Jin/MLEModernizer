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

5.81871

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.32437) has done: 'I fixed the preprocessing bugs that caused a length mismatch between the test data and the predictions, corrected the scaler usage, and ensured the submission file contains the required `key` and `fare_amount` columns for every test row.'
- What this solution (achieved 7.4498) has done: 'I add useful geographic and temporal features (Haversine distance, hour of day, weekday) to both train and test data, then slightly adjust the XGBoost hyper‑parameters (more trees, smaller learning rate, depth and subsampling) so the model can exploit the richer features and lower the RMSE toward the target while keeping the original pipeline unchanged.'
- What this solution (achieved 5.64462) has done: 'I add modest feature improvements (Manhattan distance), clip extreme fare amounts, train the model on a log‑transformed target and undo the transformation for evaluation, and slightly tune the XGBoost hyper‑parameters (more trees, lower learning rate, deeper trees). These changes keep the overall pipeline intact while aiming to lower the RMSE toward the target score.'
- What this solution (achieved 6.0447) has done: 'I add early‑stopping to the XGBoost training and slightly adjust its hyper‑parameters (more trees, lower learning‑rate, a bit deeper) so the model can keep learning without over‑fitting. This change keeps the overall pipeline identical, only modifies the fit call, and is expected to lower the validation RMSE, moving the score closer to the target.'
- What this solution (achieved 5.81871) has done: 'I add the missing imports for pandas and numpy, which cause all subsequent `NameError`s. By importing `pandas as pd` and `numpy as np` in the first cell, the data loading, preprocessing, model training, and submission creation steps can run correctly, producing a valid `submission1.csv` file. No other logic is altered, preserving the original pipeline and its intended performance.'

# 9. Code solution

## === cell 0
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as MSE
import xgboost as xgb




## === cell 1
train = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/train.csv", nrows=2_000_000
)
test_original = pd.read_csv("../input/new-york-city-taxi-fare-prediction/test.csv")
print(train.shape, test_original.shape)




## === cell 2
def haversine(lon1, lat1, lon2, lat2):
    """
    Compute the great‑circle distance between two points on the Earth (km).
    Vectors are expected in decimal degrees.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371.0 * c
    return km


train = train.dropna(axis="rows")
train = train[
    (train["pickup_longitude"] != 0)
    & (train["pickup_latitude"] != 0)
    & (train["dropoff_longitude"] != 0)
    & (train["dropoff_latitude"] != 0)
    & (train["passenger_count"] > 0)
    & (train["passenger_count"] <= 5)
    & (train["pickup_longitude"] >= -75)
    & (train["pickup_longitude"] <= -72)
    & (train["pickup_latitude"] >= 40)
    & (train["pickup_latitude"] <= 42)
    & (train["dropoff_longitude"] >= -75)
    & (train["dropoff_longitude"] <= -72)
    & (train["dropoff_latitude"] >= 40)
    & (train["dropoff_latitude"] <= 42)
]

train["distance"] = haversine(
    train["pickup_longitude"],
    train["pickup_latitude"],
    train["dropoff_longitude"],
    train["dropoff_latitude"],
)
train["distance"] = train["distance"].clip(upper=100)

train["manhattan"] = np.abs(
    train["pickup_longitude"] - train["dropoff_longitude"]
) + np.abs(train["pickup_latitude"] - train["dropoff_latitude"])

train["pickup_datetime"] = pd.to_datetime(train["pickup_datetime"])
train["hour"] = train["pickup_datetime"].dt.hour
train["weekday"] = train["pickup_datetime"].dt.weekday

train["sin_hour"] = np.sin(2 * np.pi * train["hour"] / 24)
train["cos_hour"] = np.cos(2 * np.pi * train["hour"] / 24)
train["sin_weekday"] = np.sin(2 * np.pi * train["weekday"] / 7)
train["cos_weekday"] = np.cos(2 * np.pi * train["weekday"] / 7)

train = train.drop(["key", "pickup_datetime"], axis=1)




## === cell 3
fare = train["fare_amount"].clip(lower=1, upper=200)

y = np.log1p(fare)

X = train.drop("fare_amount", axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=12
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)




## === cell 4
xgb_r = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=3000,
    learning_rate=0.02,
    max_depth=10,
    subsample=0.9,
    colsample_bytree=0.9,
    seed=123,
    n_jobs=4,
    verbosity=0,
)

xgb_r.fit(
    X_train_scaled,
    y_train,
    eval_set=[(X_valid_scaled, y_valid)],
    eval_metric="rmse",
    early_stopping_rounds=100,
    verbose=False,
)




## === cell 5
y_pred_log = xgb_r.predict(X_valid_scaled)
y_pred_valid = np.expm1(y_pred_log)

y_pred_valid = np.clip(y_pred_valid, 1, 200)

y_valid_original = np.expm1(y_valid)

rmse = np.sqrt(MSE(y_valid_original, y_pred_valid))
print(f"RMSE : {rmse:.4f}")




## === cell 6
test_features = test_original.drop(["key"], axis=1)

test_features["distance"] = haversine(
    test_features["pickup_longitude"],
    test_features["pickup_latitude"],
    test_features["dropoff_longitude"],
    test_features["dropoff_latitude"],
)
test_features["distance"] = test_features["distance"].clip(upper=100)

test_features["manhattan"] = np.abs(
    test_features["pickup_longitude"] - test_features["dropoff_longitude"]
) + np.abs(test_features["pickup_latitude"] - test_features["dropoff_latitude"])

test_features["pickup_datetime"] = pd.to_datetime(test_features["pickup_datetime"])
test_features["hour"] = test_features["pickup_datetime"].dt.hour
test_features["weekday"] = test_features["pickup_datetime"].dt.weekday

test_features["sin_hour"] = np.sin(2 * np.pi * test_features["hour"] / 24)
test_features["cos_hour"] = np.cos(2 * np.pi * test_features["hour"] / 24)
test_features["sin_weekday"] = np.sin(2 * np.pi * test_features["weekday"] / 7)
test_features["cos_weekday"] = np.cos(2 * np.pi * test_features["weekday"] / 7)

test_features = test_features.drop(["pickup_datetime"], axis=1)
test_features = test_features.fillna(0)

test_scaled = scaler.transform(test_features)




## === cell 7
test_pred_log = xgb_r.predict(test_scaled)
test_pred = np.expm1(test_pred_log)

test_pred = np.clip(test_pred, 1, 200)




## === cell 8
submission = pd.DataFrame({"key": test_original["key"], "fare_amount": test_pred})
submission_path = "submission1.csv"
submission.to_csv(submission_path, index=False)
print(f"Saved submission to {submission_path}")
