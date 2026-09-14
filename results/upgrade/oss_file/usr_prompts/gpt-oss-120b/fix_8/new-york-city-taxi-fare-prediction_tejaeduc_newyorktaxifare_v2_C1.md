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

3.22951

# 6. Current score

5.28668

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.37091) has done: 'I added richer time‑based features (hour, weekday, month), aligned one‑hot‑encoded year columns between train and test, and tweaked the XGBoost parameters (deeper trees and a lower learning rate). These changes keep the overall modeling pipeline intact while giving the model more useful signals, which should lower the RMSE toward the target value.'
- What this solution (achieved 4.62187) has done: 'Implemented missing imports, defined the haversine helper, added essential time‑based features, performed a train/validation split, built XGBoost `DMatrix` objects, trained the model, generated predictions for the test set, and finally wrote a correctly‑formatted `submission.csv` containing the required `key` and `fare_amount` columns.'
- What this solution (achieved 4.57653) has done: 'I added a clean‑up step that removes rows with missing or non‑positive fare amounts before the train/validation split, which eliminates the NaN/inf labels that caused XGBoost to raise an error. With a valid label vector the model trains correctly, and the subsequent prediction cell now finds the `model` variable. No other logic is changed, preserving the original feature set and training configuration while fixing the runtime failure and enabling a proper submission file.'
- What this solution (achieved 5.33768) has done: 'I trim extreme fare outliers (top 1 % of values) before training to reduce noise, and clip any negative predictions to zero after exponentiation. Removing these extreme cases and preventing impossible negative fares should lower the RMSE, moving the score closer to the target while keeping the original modeling pipeline intact.'
- What this solution (achieved 5.28668) has done: 'I add a simple bias‑correction step: after training, compute the average fare on the validation set and the average of the model’s exponentiated predictions, then scale all predictions (including the test set) by their ratio. This small adjustment aligns the model’s output distribution with the true target distribution and typically reduces RMSE without changing the core modeling pipeline. I also cap extreme predictions to a reasonable maximum to avoid a few outliers inflating the error.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from datetime import datetime


def haversine(lon1, lat1, lon2, lat2):
    """
    Vectorized haversine distance (km) between two points.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371.0 * c
    return km




## === cell 1
df = pd.read_csv("../input/train.csv", nrows=2_000_000)
test = pd.read_csv("../input/test.csv")
testkey = test["key"].copy()

df = df[df["fare_amount"] > 0].copy()
fare_cap = df["fare_amount"].quantile(0.99)
df = df[df["fare_amount"] <= fare_cap].copy()
df = df.reset_index(drop=True)


def add_time_features(df):
    dt = pd.to_datetime(df["pickup_datetime"])
    df["pickup_hour"] = dt.dt.hour
    df["pickup_weekday"] = dt.dt.weekday
    df["pickup_month"] = dt.dt.month
    return df


df = add_time_features(df)
test = add_time_features(test)




## === cell 2
df["dist"] = haversine(
    df["pickup_longitude"],
    df["pickup_latitude"],
    df["dropoff_longitude"],
    df["dropoff_latitude"],
)
test["dist"] = haversine(
    test["pickup_longitude"],
    test["pickup_latitude"],
    test["dropoff_longitude"],
    test["dropoff_latitude"],
)

df["delta_long"] = df["dropoff_longitude"] - df["pickup_longitude"]
df["delta_lat"] = df["dropoff_latitude"] - df["pickup_latitude"]
df["dist_sq"] = df["dist"] ** 2

test["delta_long"] = test["dropoff_longitude"] - test["pickup_longitude"]
test["delta_lat"] = test["dropoff_latitude"] - test["pickup_latitude"]
test["dist_sq"] = test["dist"] ** 2

df["log_dist"] = np.log1p(df["dist"])
test["log_dist"] = np.log1p(test["dist"])

df = df.fillna(df.median(numeric_only=True))
test = test.fillna(df.median(numeric_only=True))




## === cell 3
feature_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "dist",
    "log_dist",
    "delta_long",
    "delta_lat",
    "dist_sq",
    "pickup_hour",
    "pickup_weekday",
    "pickup_month",
]

X = df[feature_cols]
y = df["fare_amount"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)

dtrain = xgb.DMatrix(X_train, label=y_train_log)
dval = xgb.DMatrix(X_val, label=y_val_log)

params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "eta": 0.05,
    "max_depth": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "lambda": 1.0,
    "seed": 42,
}

model = xgb.train(
    params,
    dtrain,
    num_boost_round=1200,
    evals=[(dval, "val")],
    early_stopping_rounds=30,
    verbose_eval=False,
)

val_pred_log = model.predict(dval)
val_pred = np.expm1(val_pred_log)
scale_factor = y_val.mean() / val_pred.mean()
bias_scale = scale_factor




## === cell 4
dtest = xgb.DMatrix(test[feature_cols])
test_pred_log = model.predict(dtest)
test_pred = np.maximum(np.expm1(test_pred_log) * bias_scale, 0)
test_pred = np.clip(test_pred, 0, 200)

submission = pd.DataFrame({"key": testkey, "fare_amount": test_pred})

submission = submission.set_index("key").loc[test["key"]].reset_index()

submission.to_csv("submission.csv", index=False)
