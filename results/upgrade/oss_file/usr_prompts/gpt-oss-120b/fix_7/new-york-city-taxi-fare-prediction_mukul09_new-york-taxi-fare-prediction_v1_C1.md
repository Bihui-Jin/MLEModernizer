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

3.27133

# 6. Current score

6.23318

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.73153) has done: 'I fixed the KFold initialization (enabled shuffling so the random_state works), corrected the latitude/longitude range filters, removed references to the undefined `tune_model` variable, and kept the core modeling logic unchanged. The script now runs end‑to‑end, creates the required features, performs cross‑validation, runs a grid‑search, predicts on the test set, and writes a valid `taxi_fare_submission.csv` file.'
- What this solution (achieved 4.51655) has done: 'I increase the training sample size to give the model more data and add a sensible number of trees (`n_estimators`) plus a slightly larger learning rate in the grid‑search. These modest changes keep the original modeling pipeline intact while expected to lower the RMSE toward the target.'
- What this solution (achieved 4.20698) has done: 'The changes speed up the pipeline by (1) loading only the needed 1 million rows instead of first reading 6 million and then sampling, and (2) collapsing the hyper‑parameter grid to a single, sensible combination so GridSearchCV runs just one fit (still using the same XGBRegressor and cross‑validation framework). Both adjustments keep the feature engineering, model type, and evaluation unchanged, preserving result accuracy while fitting well within the 600‑second limit.'
- What this solution (achieved 6.23318) has done: 'I increase the training sample size, filter out unrealistic rows, and tune the XGBoost hyper‑parameters slightly (deeper trees, more estimators, smaller learning rate). These low‑risk changes keep the original modeling pipeline intact while aiming to lower the RMSE toward the target.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from xgboost import XGBRegressor

train_path = "../input/train.csv"
test_path = "../input/test.csv"

train_data = pd.read_csv(train_path, nrows=2_000_000)  # ↑ increase sample size
test_data = pd.read_csv(test_path)




## === cell 1
def distance(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    """
    Great‑circle distance (km) between pickup and drop‑off coordinates.
    """
    R_earth = 6371.0
    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = map(
        np.radians, [pickup_lat, pickup_lon, dropoff_lat, dropoff_lon]
    )
    dlat = dropoff_lat - pickup_lat
    dlon = dropoff_lon - pickup_lon
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(pickup_lat) * np.cos(dropoff_lat) * np.sin(dlon / 2.0) ** 2
    )
    return 2 * R_earth * np.arcsin(np.sqrt(a))


def date_time_info(df):
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df["hour"] = df["pickup_datetime"].dt.hour
    df["day"] = df["pickup_datetime"].dt.day
    df["month"] = df["pickup_datetime"].dt.month
    df["weekday"] = df["pickup_datetime"].dt.weekday
    df["year"] = df["pickup_datetime"].dt.year
    return df


train_data = date_time_info(train_data)
train_data["distance"] = distance(
    train_data["pickup_latitude"],
    train_data["pickup_longitude"],
    train_data["dropoff_latitude"],
    train_data["dropoff_longitude"],
)
train_data["lat_diff"] = train_data["dropoff_latitude"] - train_data["pickup_latitude"]
train_data["lon_diff"] = (
    train_data["dropoff_longitude"] - train_data["pickup_longitude"]
)

train_data = train_data.dropna(subset=["pickup_datetime"])
train_data = train_data[
    (train_data["fare_amount"] > 0)
    & (train_data["fare_amount"] < 200)
    & (train_data["passenger_count"] >= 1)
    & (train_data["passenger_count"] <= 6)
    & (train_data["pickup_longitude"].between(-74.5, -73))
    & (train_data["dropoff_longitude"].between(-74.5, -73))
    & (train_data["pickup_latitude"].between(40, 41.5))
    & (train_data["dropoff_latitude"].between(40, 41.5))
]

test_data = date_time_info(test_data)
test_data["distance"] = distance(
    test_data["pickup_latitude"],
    test_data["pickup_longitude"],
    test_data["dropoff_latitude"],
    test_data["dropoff_longitude"],
)
test_data["lat_diff"] = test_data["dropoff_latitude"] - test_data["pickup_latitude"]
test_data["lon_diff"] = test_data["dropoff_longitude"] - test_data["pickup_longitude"]
test_key = test_data["key"]



## === cell 2
X = train_data.drop(columns=["key", "fare_amount", "pickup_datetime"])
y = train_data["fare_amount"]

cv_split = KFold(n_splits=5, shuffle=True, random_state=42)

params = {
    "max_depth": [8],  # deeper trees
    "learning_rate": [0.05],  # smaller step size
    "subsample": [1.0],
    "colsample_bytree": [0.8],
    "n_estimators": [600],  # more trees
    "objective": ["reg:squarederror"],
    "eval_metric": ["rmse"],
    "n_jobs": [-1],
}

grid = GridSearchCV(
    XGBRegressor(random_state=0),
    param_grid=params,
    scoring="neg_mean_squared_error",
    cv=cv_split,
    n_jobs=-1,
    verbose=1,
)
grid.fit(X, y)



## === cell 3
X_test = test_data.drop(columns=["key", "pickup_datetime"])
test_pred = grid.best_estimator_.predict(X_test)

submission = pd.DataFrame({"key": test_key, "fare_amount": test_pred})
submission_path = "taxi_fare_submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
