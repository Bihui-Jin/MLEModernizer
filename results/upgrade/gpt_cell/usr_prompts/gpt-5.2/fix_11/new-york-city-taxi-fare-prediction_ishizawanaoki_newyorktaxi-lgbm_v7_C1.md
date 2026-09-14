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

3.10

# 3. Installed packages

geopandas==0.14.4
lightgbm==4.6.0
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

3.42426

# 6. Current score

5.23159

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.57292) has done: 'Diagnosis: The crash happens in cell 14 because LightGBM 4.6.0 removed the `verbose_eval` and `early_stopping_rounds` keyword arguments from `lgb.train()`. In newer versions these behaviors must be provided via callbacks instead, otherwise `TypeError: train() got an unexpected keyword argument 'verbose_eval'` is raised.  
Patch summary: Update the `lgb.train(...)` call in cell 14 to use `callbacks=[lgb.log_evaluation(...), lgb.early_stopping(...)]` while keeping the same evaluation frequency (10) and early stopping patience (10). No other training logic, parameters, or outputs are changed.  
Updated cells: Only cell 14 is modified.  
Compatibility notes for cell k+1: `models`, `oof_train`, and each model’s `best_score` / `best_iteration` remain available and behave consistently, so cell 15 work unchanged.  
Assumptions: LightGBM callbacks API is available in 4.6.0 (it is), and the intent is identical verbosity and early stopping semantics as in the original code.'
- What this solution (achieved 5.60271) has done: 'Your score (RMSE 4.57292) is worse than the target (3.42426), so we should improve accuracy with minimal, safe changes that keep the same LightGBM/KFold training flow. The biggest issue is that you currently train on many clearly-invalid/outlier rides (bad coordinates, extreme fares, unrealistic passenger counts), which strongly hurts RMSE; we add standard NYC Taxi competition cleaning filters before training (same model, same CV loop). We also fix a subtle indexing bug: `KFold.split` yields positional indices, so we must use `.iloc` (not `.loc`) to avoid mis-selection after dropping rows. Finally, we keep the same feature set but make the `key` parsing stable (extract digits) and ensure the submission is aligned to `test` keys.'
- What this solution (achieved 5.91929) has done: 'We keep your LightGBM + 5-fold training exactly as-is, but fix the main reason you currently can’t yield a meaningful Kaggle score: the submission is being written with `sample_submission` keys (which can silently mismatch ordering/rows) instead of using the actual `test.csv` keys you predicted for. We preserve keys before feature engineering, then build the submission from `test_keys` to guarantee 1:1 alignment between each prediction and the correct `key`. Additionally, we add a very small, competition-standard post-processing step: clip negative fares to 0 (valid fare domain), which typically reduces RMSE a bit without changing your model/training logic. These changes are minimal, run end-to-end, and produce a valid `submission_lightgbm.csv`.'
- What this solution (achieved 5.2462) has done: 'Your current RMSE (5.919) is much worse than the target (3.424), so we should improve accuracy with the smallest possible change that doesn’t alter your LightGBM + KFold training flow. The biggest remaining issue is that the model still sees many “bad” training rows (e.g., zero-distance trips with non-trivial fare, and very long trips that are rare/outlier), which inflates RMSE; we add two standard NYC Taxi filters: remove near-zero-distance rides and cap extreme haversine distances. This keeps your feature set and model the same, just improves the training data quality feeding that same pipeline. We also ensure the new filters are applied after feature engineering (so they’re based on the same `haversine_km` feature you already compute), and keep submission alignment unchanged.'
- What this solution (achieved 5.23159) has done: 'We make two minimal, score-relevant fixes that keep your LightGBM + 5-fold CV training flow and feature set intact. First, we prevent `pickup_hour/dayofweek/month` from becoming missing (and thus causing NaNs/Int64 issues) by filling invalid datetimes with a sentinel and casting to small integers; this typically improves stability and RMSE a bit. Second, we apply the same coordinate sanity filters to the test set as you already do for train (without dropping rows), by clipping test coordinates into the same bounds; this reduces extreme out-of-distribution distances that can blow up predictions and hurt RMSE. Submission writing stays aligned to `test.csv` keys and produces `submission_lightgbm.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
train = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/train.csv", nrows=1_000_000
)
test = pd.read_csv("../input/new-york-city-taxi-fare-prediction/test.csv")
sample_submission = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/sample_submission.csv"
)



## === cell 2
train.isnull().sum()



## === cell 3
train.dropna(inplace=True)



## === cell 4
train.describe()



## === cell 5
train.quantile(0.99, numeric_only=True)



## === cell 6
train.quantile(0.01, numeric_only=True)



## === cell 7
train.describe()



## === cell 8
train = train[
    (train["fare_amount"] > 0)
    & (train["fare_amount"] <= 250)
    & (train["passenger_count"] >= 1)
    & (train["passenger_count"] <= 6)
    & (train["pickup_longitude"].between(-74.5, -72.8))
    & (train["dropoff_longitude"].between(-74.5, -72.8))
    & (train["pickup_latitude"].between(40.5, 41.8))
    & (train["dropoff_latitude"].between(40.5, 41.8))
].copy()

train.reset_index(drop=True, inplace=True)
train



## === cell 9
test_keys = test["key"].astype(str).copy()

y_train_raw = train["fare_amount"].astype(np.float64).copy()

train_features_only = train.drop(columns=["fare_amount"])
data = pd.concat([train_features_only, test], sort=False, ignore_index=True)



## === cell 10
data.head()



## === cell 11
keys_all = data["key"].astype(str).copy()




## === cell 12
def haversine_np(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1.astype(np.float64))
    lat1 = np.radians(lat1.astype(np.float64))
    lon2 = np.radians(lon2.astype(np.float64))
    lat2 = np.radians(lat2.astype(np.float64))

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371.0 * c
    return km


pickup_dt = pd.to_datetime(data["pickup_datetime"], errors="coerce", utc=True)

data["pickup_hour"] = pickup_dt.dt.hour.fillna(-1).astype(np.int16)
data["pickup_dayofweek"] = pickup_dt.dt.dayofweek.fillna(-1).astype(np.int16)
data["pickup_month"] = pickup_dt.dt.month.fillna(-1).astype(np.int16)

n_train_tmp = len(train_features_only)
for col, lo, hi in [
    ("pickup_longitude", -74.5, -72.8),
    ("dropoff_longitude", -74.5, -72.8),
    ("pickup_latitude", 40.5, 41.8),
    ("dropoff_latitude", 40.5, 41.8),
]:
    data.loc[n_train_tmp:, col] = data.loc[n_train_tmp:, col].clip(lo, hi)

data["haversine_km"] = haversine_np(
    data["pickup_longitude"].values,
    data["pickup_latitude"].values,
    data["dropoff_longitude"].values,
    data["dropoff_latitude"].values,
)

data = data.drop(["key", "pickup_datetime"], axis=1)

data.head()



## === cell 13
n_train = len(train)  # original (cleaned) train length
train_fe = data.iloc[:n_train].copy()
test_fe = data.iloc[n_train:].copy()

distance_mask = (train_fe["haversine_km"] >= 0.01) & (train_fe["haversine_km"] <= 80.0)
train_fe = train_fe.loc[distance_mask].copy()
y_train_raw = y_train_raw.loc[distance_mask].copy()

train_fe.reset_index(drop=True, inplace=True)
y_train_raw.reset_index(drop=True, inplace=True)

train_fe.shape, test_fe.shape



## === cell 14
train = train_fe.copy()
train["fare_amount"] = y_train_raw.values
test = test_fe.copy()
test["fare_amount"] = np.nan

y_train = train["fare_amount"].astype(np.float64)
X_train = train.drop("fare_amount", axis=1)
X_test = test.drop("fare_amount", axis=1)

X_train.head()



## === cell 15
from sklearn.model_selection import KFold

y_preds = []
models = []
oof_train = np.zeros((len(X_train),), dtype=np.float64)
cv = KFold(n_splits=5, shuffle=True, random_state=0)

categorical_features = []



## === cell 16
import lightgbm as lgb

params = {
    "objective": "regression",
    "max_bin": 300,
    "learning_rate": 0.05,
    "num_leaves": 40,
    "metric": "rmse",
    "verbosity": -1,
}

for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train, y_train)):
    X_tr = X_train.iloc[train_index, :]
    X_val = X_train.iloc[valid_index, :]
    y_tr = y_train.iloc[train_index]
    y_val = y_train.iloc[valid_index]

    lgb_train = lgb.Dataset(X_tr, y_tr, categorical_feature=categorical_features)
    lgb_eval = lgb.Dataset(
        X_val, y_val, reference=lgb_train, categorical_feature=categorical_features
    )

    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_eval],
        num_boost_round=1000,
        callbacks=[
            lgb.log_evaluation(period=10),
            lgb.early_stopping(stopping_rounds=10),
        ],
    )

    oof_train[valid_index] = model.predict(X_val, num_iteration=model.best_iteration)
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)

    y_preds.append(y_pred)
    models.append(model)



## === cell 17
pd.DataFrame(oof_train).to_csv("oof_train_kfold.csv", index=False)

scores = [m.best_score["valid_1"]["rmse"] for m in models]
score = sum(scores) / len(scores)
print("===CV rmse scores===")
print(scores)
print(score)



## === cell 18
from sklearn.metrics import mean_squared_error

y_pred_oof = oof_train
np.sqrt(mean_squared_error(y_train, y_pred_oof))



## === cell 19
len(y_preds)



## === cell 20
y_preds[0][:10]



## === cell 21
y_sub = sum(y_preds) / len(y_preds)

y_sub = np.clip(y_sub.astype(np.float64), 0.0, None)

y_sub[:10]



## === cell 22
sub_lgb = pd.DataFrame({"key": test_keys.values, "fare_amount": y_sub})
sub_lgb.to_csv("submission_lightgbm.csv", index=False)

sub_lgb.head()
