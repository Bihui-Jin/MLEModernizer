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

5.13539

# 6. Current score

5.7374

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 22.73196) has done: 'Diagnosis: The crash happens because XGBoost 2.x no longer exposes `Booster.best_ntree_limit` on the object returned by `xgb.train()`. With early stopping enabled, the correct compatible way is to use `iteration_range` based on `model.best_iteration` (or fall back to full prediction if it’s not present). This keeps the same evaluation semantics (use the best iteration found by early stopping) while avoiding the removed attribute.  

Patch summary: Modify only the prediction line in cell 4 to use `iteration_range=(0, best_iteration + 1)` when available, otherwise predict without limiting iterations. No changes to model training, features, or file I/O.  

Updated cells:  

Compatibility notes for cell k+1: `prediction` remains a 1D NumPy array of predicted fares with the same length/order as `test`, so cell 5 can build the submission identically.  

Assumptions: Early stopping runs as before and `model.best_iteration` is available in this XGBoost version; if not, the fallback path predicts using all boosting rounds (still deterministic).'
- What this solution (achieved 5.7374) has done: 'Your current score is far from the target (22.73 vs 5.14; lower is better), so the main issue is model/data quality rather than minor XGBoost API compatibility. With minimal disruption to your core approach (same feature engineering + same `xgb.train` training loop), I (1) fix two bugs in the distance features (your “distance” formula and haversine `a` term are wrong), (2) add lightweight, standard NYC taxi data cleaning (bounding boxes, positive fares, passenger bounds) to remove extreme outliers that dominate RMSE, and (3) keep your same two-feature model but make predictions non-negative and write a correctly named submission CSV. These changes typically move RMSE substantially toward single digits without changing the modeling approach.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import datetime as dt

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import xgboost as xgb

TRAIN_PATHS = [
    "../input/train.csv",
    "/kaggle/input/train.csv",
    "/kaggle/data/train.csv",
]
TEST_PATHS = ["../input/test.csv", "/kaggle/input/test.csv", "/kaggle/data/test.csv"]


def _read_first_existing(paths, **kwargs):
    for p in paths:
        try:
            return pd.read_csv(p, **kwargs)
        except Exception:
            pass
    raise FileNotFoundError(f"None of these paths could be read: {paths}")


train = _read_first_existing(TRAIN_PATHS, nrows=10_000_000)
test = _read_first_existing(TEST_PATHS)

combine = [train, test]
test.dtypes



## === cell 1
for dataset in combine:
    dataset["longitude_distance"] = (
        dataset["pickup_longitude"] - dataset["dropoff_longitude"]
    ).abs()
    dataset["latitude_distance"] = (
        dataset["pickup_latitude"] - dataset["dropoff_latitude"]
    ).abs()

    dist_euclid = np.sqrt(
        dataset["longitude_distance"] ** 2 + dataset["latitude_distance"] ** 2
    )
    dataset["distance_travelled"] = dist_euclid
    dataset["distance_travelled_sin"] = np.sin(dist_euclid)
    dataset["distance_travelled_cos"] = np.cos(dist_euclid)
    dataset["distance_travelled_sin_sqrd"] = np.sin(dist_euclid) ** 2
    dataset["distance_travelled_cos_sqrd"] = np.cos(dist_euclid) ** 2

    R = 6371e3
    phi1 = np.radians(dataset["pickup_latitude"])
    phi2 = np.radians(dataset["dropoff_latitude"])
    dphi = np.radians(dataset["dropoff_latitude"] - dataset["pickup_latitude"])
    dlambda = np.radians(dataset["dropoff_longitude"] - dataset["pickup_longitude"])
    a = (np.sin(dphi / 2) ** 2) + np.cos(phi1) * np.cos(phi2) * (
        np.sin(dlambda / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    dataset["haversine"] = R * c

    y = np.sin(dlambda * np.cos(phi2))
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlambda)
    dataset["bearing"] = np.degrees(np.arctan2(y, x))

    psi_chg = np.log(np.tan(np.pi / 4 + phi2 / 2) / np.tan(np.pi / 4 + phi1 / 2))
    q = dphi / psi_chg.replace(0, np.nan)
    q = q.fillna(np.cos(phi1))
    dataset["rhumb_lines"] = np.sqrt(dphi**2 + (q**2) * (dlambda**2)) * R

    dataset["pickup_datetime"] = pd.to_datetime(
        dataset["pickup_datetime"], errors="coerce", utc=False
    )
    dataset["hour_of_day"] = dataset.pickup_datetime.dt.hour
    dataset["day"] = dataset.pickup_datetime.dt.day
    dataset["week"] = (
        dataset.pickup_datetime.dt.isocalendar().week.astype("Int64").astype(float)
    )
    dataset["month"] = dataset.pickup_datetime.dt.month
    dataset["day_of_year"] = dataset.pickup_datetime.dt.dayofyear
    dataset["week_of_year"] = (
        dataset.pickup_datetime.dt.isocalendar().week.astype("Int64").astype(float)
    )

train.head(3)



## === cell 2
colormap = plt.cm.RdBu
plt.figure(figsize=(20, 20))
plt.title("Pearson Correlation of Features", y=1.05, size=15)

corr_mat = train.corr(numeric_only=True)

sns.heatmap(
    corr_mat,
    linewidths=0.1,
    vmax=1.0,
    square=True,
    cmap=colormap,
    linecolor="white",
    annot=True,
)



## === cell 3
train = train.dropna(
    subset=[
        "fare_amount",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
        "distance_travelled_sin",
        "haversine",
    ]
)

train = train[
    (train["fare_amount"] > 0)
    & (train["fare_amount"] <= 300)
    & (train["passenger_count"] >= 1)
    & (train["passenger_count"] <= 6)
    & (train["pickup_longitude"].between(-74.3, -73.7))
    & (train["dropoff_longitude"].between(-74.3, -73.7))
    & (train["pickup_latitude"].between(40.5, 41.0))
    & (train["dropoff_latitude"].between(40.5, 41.0))
    & (train["haversine"] > 0)
    & (train["haversine"] < 200_000)
].copy()

train_features_to_keep = ["distance_travelled_sin", "haversine", "fare_amount"]
train.drop(columns=train.columns.difference(train_features_to_keep), inplace=True)

test_features_to_keep = ["key", "distance_travelled_sin", "haversine"]
test.drop(columns=test.columns.difference(test_features_to_keep), inplace=True)



## === cell 4
x_pred = test.drop("key", axis=1)

x_train, x_test, y_train, y_test = train_test_split(
    train.drop("fare_amount", axis=1),
    train["fare_amount"],
    random_state=123,
    test_size=0.2,
)


def XGBmodel(x_train, x_test, y_train, y_test):
    matrix_train = xgb.DMatrix(x_train, label=y_train)
    matrix_test = xgb.DMatrix(x_test, label=y_test)
    model = xgb.train(
        params={"objective": "reg:linear", "eval_metric": "rmse"},
        dtrain=matrix_train,
        num_boost_round=300,
        early_stopping_rounds=30,
        evals=[(matrix_test, "test")],
    )
    return model


model = XGBmodel(x_train, x_test, y_train, y_test)

dmat_pred = xgb.DMatrix(x_pred)
if hasattr(model, "best_iteration") and model.best_iteration is not None:
    prediction = model.predict(dmat_pred, iteration_range=(0, model.best_iteration + 1))
else:
    prediction = model.predict(dmat_pred)

prediction = np.clip(prediction, 0, None)



## === cell 5
submission = pd.DataFrame({"key": test["key"], "fare_amount": prediction.round(2)})

submission.to_csv("sub_fare.csv", index=False)

submission.head()



## === cell 6
submission
