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

4.15448

# 6. Current score

5.72314

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 81.02165) has done: 'I fix the runtime errors caused by deprecated pandas datetime accessors (`.dt.week` and `.dt.weekofyear`) by switching to ISO calendar week extraction that works in pandas 2.2. I also correct the XGBoost parameters to match xgboost 2.x (remove deprecated `silent`, use `reg:squarederror`, and keep the same training call), which should materially improve RMSE toward your target without changing the overall modeling approach. Finally, I ensure the Kaggle file paths match your environment (`/kaggle/input/...`) and that a valid submission CSV with the required `key,fare_amount` columns is always written.'
- What this solution (achieved 98.88482) has done: 'Your current RMSE is far worse than the target, so we should improve accuracy with the smallest changes that keep your overall approach (same models + simple feature engineering + averaging ensemble). The biggest issue is your feature set: using only absolute lat/long differences misses true travel distance and key NYC location structure, so adding a haversine distance feature and a couple of standard geospatial/time-derived features typically yields a large RMSE drop without changing the modeling “core.” To keep behavior stable and within Kaggle constraints, we also align train/test preprocessing more safely (drop NAs consistently, coerce invalid passenger counts), and slightly strengthen XGBoost via a few conservative parameters and a modest increase in boosting rounds (still the same xgb.train call). The output submission format and paths remain the same, and it still write a valid `.csv`.'
- What this solution (achieved 98.88482) has done: 'Your current RMSE (98.88) is far worse than the target (4.15), so we should make the smallest changes that address the most likely root cause: train/test row misalignment created by dropping NAs in `test` after reading it (this changes the row count/order vs. `sample_submission`, and can massively degrade the score). I preserve your exact model set and ensemble logic, but ensure we never drop rows from `test`; instead we only engineer features, and impute any missing engineered feature values in `test` (and `train`) with training medians so prediction order stays identical to `sample_submission`. I also keep the same basic cleaning on `train`, and add a minimal safeguard to clip negative predictions to 0 (fares can’t be negative), which typically reduces RMSE without changing the modeling approach. The script still run end-to-end and write a valid `key,fare_amount` submission CSV with exactly the test set keys in the correct order.'
- What this solution (achieved 1947.25542) has done: 'Your RMSE is massively worse than the target, so the smallest likely “root cause” fix is to ensure your feature engineering doesn’t inject NaNs/Inf or mis-scaled coordinates (which can make all models behave badly). I add a minimal, standard NYC bounding-box filter on *training only* using correct longitude/latitude ranges, plus a couple of lightweight “core” features (log distance and distance*passengers) that are consistent with your existing distance-based approach and typically drops RMSE a lot without changing the modeling family. I also harden the haversine computation against invalid values and then impute any remaining NaNs using train medians (keeping test row order intact), and keep your ensemble/training loop exactly as-is. The submission writing stays identical and still produce a valid `key,fare_amount` CSV.'
- What this solution (achieved 1947.25542) has done: 'Your RMSE is far above the target (lower is better), so we need the smallest changes that correct a likely “catastrophic” issue rather than tuning models. The biggest problem in your current code is that `submission["key"]` is taken from `sample_submission.csv` order, while `predictions` are produced in `test.csv` order; those key orders are not guaranteed to match, and a mismatch can explode RMSE into the thousands. I keep your exact feature engineering and models/weights, but build the submission from `test[["key"]]` (preserving prediction alignment) and add a defensive check that keys match if you still want to use `sample_submission` for schema. This should move RMSE dramatically down toward the target without changing modeling semantics.'
- What this solution (achieved 5.72314) has done: 'Your current RMSE is catastrophically worse than the target (lower is better), so we should focus on fixing one likely “fatal” issue rather than tuning models: bad/invalid coordinate rows in the **test** set can produce huge/NaN distance features, and those propagate into wildly wrong predictions. I keep your exact model set, ensemble weights, and training approach, but add a minimal test-time coordinate sanitization (clip to valid lat/lon ranges and coerce passenger_count) plus a hard guard to replace any remaining NaN/Inf engineered features with train medians. This preserves row order (no dropping test rows) and typically brings RMSE back down dramatically. Finally, I ensure the submission is always built from `test[['key']]` (already correct) and written as a valid `.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

INPUT_DIR = "/kaggle/input/new-york-city-taxi-fare-prediction"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

train = pd.read_csv(TRAIN_PATH, nrows=300_000)
test = pd.read_csv(TEST_PATH)



## === cell 1
train.shape



## === cell 2
train.head()



## === cell 3
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt



## === cell 4
train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], errors="coerce", utc=True
)
train["hour"] = train["pickup_datetime"].dt.hour
train["day"] = train["pickup_datetime"].dt.day
train["month"] = train["pickup_datetime"].dt.month
train["day_of_year"] = train["pickup_datetime"].dt.dayofyear
train["week"] = train["pickup_datetime"].dt.isocalendar().week.astype("int16")
train["week_of_year"] = train["week"]



## === cell 5
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], errors="coerce", utc=True
)
test["hour"] = test["pickup_datetime"].dt.hour
test["day"] = test["pickup_datetime"].dt.day
test["month"] = test["pickup_datetime"].dt.month
test["day_of_year"] = test["pickup_datetime"].dt.dayofyear
test["week"] = test["pickup_datetime"].dt.isocalendar().week.astype("int16")
test["week_of_year"] = test["week"]



## === cell 6
train.head()
train = train.dropna(how="any", axis="rows")



## === cell 7
train = train.loc[(train["fare_amount"] > 0) & (train["fare_amount"] < 200)]
train = train.loc[
    (train["pickup_longitude"] > -74.5) & (train["pickup_longitude"] < -72.8)
]
train = train.loc[(train["pickup_latitude"] > 40.5) & (train["pickup_latitude"] < 41.8)]
train = train.loc[
    (train["dropoff_longitude"] > -74.5) & (train["dropoff_longitude"] < -72.8)
]
train = train.loc[
    (train["dropoff_latitude"] > 40.5) & (train["dropoff_latitude"] < 41.8)
]
train = train.loc[(train["passenger_count"] >= 1) & (train["passenger_count"] <= 8)]




## === cell 8
def _haversine_km(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1.astype(float))
    lat1 = np.radians(lat1.astype(float))
    lon2 = np.radians(lon2.astype(float))
    lat2 = np.radians(lat2.astype(float))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0088 * c  # Earth radius in km


train["abs_diff_longitude"] = (
    train["pickup_longitude"] - train["dropoff_longitude"]
).abs()
train["abs_diff_latitude"] = (
    train["pickup_latitude"] - train["dropoff_latitude"]
).abs()

train["haversine_km"] = _haversine_km(
    train["pickup_longitude"],
    train["pickup_latitude"],
    train["dropoff_longitude"],
    train["dropoff_latitude"],
)
train["manhattan_approx_km"] = _haversine_km(
    train["pickup_longitude"],
    train["pickup_latitude"],
    train["dropoff_longitude"],
    train["pickup_latitude"],
) + _haversine_km(
    train["pickup_longitude"],
    train["pickup_latitude"],
    train["pickup_longitude"],
    train["dropoff_latitude"],
)

train["log_haversine_km"] = np.log1p(train["haversine_km"].clip(lower=0))
train["haversine_x_passengers"] = train["haversine_km"] * train["passenger_count"]



## === cell 9
for col, lo, hi in [
    ("pickup_longitude", -74.5, -72.8),
    ("dropoff_longitude", -74.5, -72.8),
    ("pickup_latitude", 40.5, 41.8),
    ("dropoff_latitude", 40.5, 41.8),
]:
    test[col] = pd.to_numeric(test[col], errors="coerce").clip(lo, hi)

test["passenger_count"] = pd.to_numeric(test["passenger_count"], errors="coerce")
test["passenger_count"] = test["passenger_count"].clip(1, 8)



## === cell 10
test["abs_diff_longitude"] = (
    test["pickup_longitude"] - test["dropoff_longitude"]
).abs()
test["abs_diff_latitude"] = (test["pickup_latitude"] - test["dropoff_latitude"]).abs()

test["haversine_km"] = _haversine_km(
    test["pickup_longitude"],
    test["pickup_latitude"],
    test["dropoff_longitude"],
    test["dropoff_latitude"],
)
test["manhattan_approx_km"] = _haversine_km(
    test["pickup_longitude"],
    test["pickup_latitude"],
    test["dropoff_longitude"],
    test["pickup_latitude"],
) + _haversine_km(
    test["pickup_longitude"],
    test["pickup_latitude"],
    test["pickup_longitude"],
    test["dropoff_latitude"],
)

test["log_haversine_km"] = np.log1p(test["haversine_km"].clip(lower=0))
test["haversine_x_passengers"] = test["haversine_km"] * test["passenger_count"]



## === cell 11
train.head()



## === cell 12
train.head()



## === cell 13
_ = sns.barplot(data=train, x="passenger_count", y="fare_amount")



## === cell 14
feature_names = [
    "hour",
    "passenger_count",
    "abs_diff_longitude",
    "abs_diff_latitude",
    "haversine_km",
    "manhattan_approx_km",
    "log_haversine_km",
    "haversine_x_passengers",
]
feature_names



## === cell 15
label_name = "fare_amount"
label_name



## === cell 16
X_train = train[feature_names].copy()
y_train = train[label_name].copy()
X_test = test[feature_names].copy()

X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

train_medians = X_train.median(numeric_only=True)
X_train = X_train.fillna(train_medians)
X_test = X_test.fillna(train_medians)



## === cell 17
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb



## === cell 18
regr = LinearRegression()
regr.fit(X_train, y_train)
regr_prediction = regr.predict(X_test)



## === cell 19
knr = KNeighborsRegressor(n_neighbors=20, weights="distance", n_jobs=-1)
knr.fit(X_train, y_train)
knr_prediction = knr.predict(X_test)



## === cell 20
rfr = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rfr.fit(X_train, y_train)
rfr_prediction = rfr.predict(X_test)



## === cell 21
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)



## === cell 22
params = {
    "max_depth": 7,
    "eta": 0.05,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1.0,
    "lambda": 1.0,
    "alpha": 0.0,
    "verbosity": 0,
    "seed": 42,
}
num_rounds = 300



## === cell 23
xb = xgb.train(params, dtrain, num_rounds)



## === cell 24
y_pred_xgb = xb.predict(dtest)
print(y_pred_xgb)



## === cell 25
predictions = (
    regr_prediction + 2 * rfr_prediction + knr_prediction + 5 * y_pred_xgb
) / 9



## === cell 26
predictions = np.clip(predictions, 0, None)



## === cell 27
predictions



## === cell 28
submission = test[["key"]].copy()
submission["fare_amount"] = predictions

sample_sub = pd.read_csv(SAMPLE_SUB_PATH, usecols=["key"])
if len(sample_sub) == len(submission):
    same_order = sample_sub["key"].astype(str).equals(submission["key"].astype(str))
    print("sample_submission key order matches test:", bool(same_order))

assert len(submission) == len(
    test
), f"Row mismatch: submission={len(submission)} test={len(test)}"



## === cell 29
submission.head()



## === cell 30
out_path = "./simplenewyorktaxi.csv"
submission.to_csv(out_path, index=False)
print("Wrote submission to:", out_path, "shape:", submission.shape)
