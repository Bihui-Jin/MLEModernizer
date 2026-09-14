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

4.14546

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.91454) has done: 'I fix the pandas `quantile()` failures by restricting quantiles to numeric columns (the `key` and `pickup_datetime` strings were causing the `str - str` error), which is score-neutral. I also fix the feature engineering bug by not attempting to cast the string `key` into float; instead, we drop `key` from modeling features but keep it intact for submission IDs (this is required for a valid submission and avoids the conversion error). Finally, I update the LightGBM training call to be compatible with LightGBM 4.6.0 by replacing deprecated arguments (`verbose_eval`, `early_stopping_rounds`) with callbacks, so training runs and produces predictions. The script always write a valid `submission_lightgbm.csv` with columns `key,fare_amount`.'
- What this solution (achieved 5.68325) has done: 'Your current gap is 4.91454 − 3.42426 = 1.49028 (lower is better), so we need a modest but real improvement without changing the overall approach (LightGBM + KFold). The biggest score win with minimal semantic impact is to fix the feature set: dropping `pickup_datetime` removes an important signal, so we extract a few standard time parts (year/month/day/hour/weekday) and then drop the raw timestamp column. We also add a single, very standard distance feature (haversine) plus absolute deltas, which keeps the same model/training loop but usually reduces RMSE materially for this competition. Finally, we add a lightweight, competition-standard coordinate/fare filter on the sampled training rows to reduce extreme outliers that hurt RMSE, while keeping everything else (CV, params, early stopping, submission format) intact.'
- What this solution (achieved 4.1158) has done: 'Your current RMSE (5.68325) is worse than the target (3.42426), so we should make small, high-impact fixes that keep the same LightGBM+KFold training loop but improve feature signal and reduce noise. The largest gain with minimal disruption is to (1) fix the distance feature to use miles (this competition’s fares are closer to miles than km), and (2) add two standard, non-invasive geospatial features (Manhattan distance and bearing) that typically reduce RMSE without changing the modeling approach. We also apply the same coordinate sanity filter to the concatenated `data` so test-time feature distributions match train-time filtering behavior (without dropping any test rows), and we clip negative predictions to 0 since fares can’t be negative (a safe post-processing step that usually helps RMSE slightly). Everything else (data loading, filtering, CV, LightGBM training, submission schema/path) stays intact.'
- What this solution (achieved 4.12061) has done: 'We need to reduce RMSE from 4.1158 toward 3.42426 (lower is better), so we make small, high-impact fixes without changing the LightGBM+KFold core training loop. The biggest likely gain at this stage is correcting the “manhattan_miles” feature: it currently adds degree differences and then multiplies by a miles conversion factor, which is dimensionally wrong; we compute Manhattan distance in miles using latitude/longitude scaling at NYC latitudes. We also add one very standard, low-risk feature (`euclidean_miles`) derived from the same corrected deltas, which preserves the overall approach but typically helps RMSE. Everything else (data loading, filters, CV, LightGBM training, callbacks, submission format/path) stays the same.'
- What this solution (achieved 4.14187) has done: 'We need to move RMSE down from 4.12061 toward 3.42426 (lower is better), so we should make small, high-impact changes without altering the LightGBM+KFold core loop. The most “minimal but meaningful” gain for NYC Taxi Fare is adding a couple of standard geospatial/time features that preserve the same semantics: airport-distance features (JFK/LGA/EWR) and a simple “near NYC center” distance, which often reduces error on airport trips. We also slightly tighten the training filters to remove clearly invalid zero-distance rides and extreme coordinate deltas that add label noise, while keeping the same sampling (1M rows), model type, and training procedure. Finally, we keep submission generation identical (same columns, same filename) and maintain deterministic behavior.'
- What this solution (achieved 4.14187) has done: 'We need to move RMSE down from 4.14187 toward 3.42426 (lower is better), so the smallest likely win is to fix two correctness bugs in feature engineering that currently degrade model signal. First, the `bearing_np()` computation has a broken term (a no-op `* np.sin(lat1) * 0`) and an unnecessarily convoluted expression; correcting it to the standard bearing formula improves a key directional feature without changing the modeling approach. Second, we currently impute all numeric columns (including `fare_amount`) inside the concatenated train+test frame, which can leak/warp the target and hurts learning; we exclude `fare_amount` from imputation while keeping the same median-impute strategy for features. Everything else (data size, filters, feature set intent, LightGBM + 5-fold KFold + early stopping, submission file/columns) stays the same.'
- What this solution (achieved 4.14546) has done: 'Your RMSE (4.14187) is still above the target (3.42426), so we should make a small, legitimate improvement that keeps the same LightGBM+KFold loop and loss/metric. The highest-impact minimal change here is to use LightGBM’s default boosting more effectively by adding a few standard regularization/robustness parameters (feature_fraction/bagging, min_data_in_leaf, and a slightly higher num_leaves), which typically improves generalization on this dataset without changing the approach. We also add one very standard interaction feature (`passenger_count * distance`) that preserves your existing feature engineering style and often reduces RMSE modestly. Everything else (sampling size, filters, feature creation intent, training loop, and submission format/path) stays the same.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
train = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", nrows=1_000_000
)
test = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")
sample_submission = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
)



## === cell 2
train.isnull().sum()



## === cell 3
train.dropna(inplace=True)



## === cell 4
train = train[(train["fare_amount"] > 0) & (train["fare_amount"] <= 250)].copy()

train = train[
    (train["pickup_longitude"].between(-75, -72))
    & (train["dropoff_longitude"].between(-75, -72))
    & (train["pickup_latitude"].between(40, 42))
    & (train["dropoff_latitude"].between(40, 42))
].copy()

train = train[train["passenger_count"].between(1, 6)].copy()

raw_abs_lon = (train["pickup_longitude"] - train["dropoff_longitude"]).abs()
raw_abs_lat = (train["pickup_latitude"] - train["dropoff_latitude"]).abs()
train = train[(raw_abs_lon + raw_abs_lat) > 1e-6].copy()

train = train[(raw_abs_lon < 1.0) & (raw_abs_lat < 1.0)].copy()

train.reset_index(drop=True, inplace=True)



## === cell 5
train.describe()



## === cell 6
train.quantile(0.99, numeric_only=True)



## === cell 7
train.quantile(0.01, numeric_only=True)



## === cell 8
train.describe()



## === cell 9
train.reset_index(drop=True, inplace=True)
train



## === cell 10
data = pd.concat([train, test], sort=False, ignore_index=True)



## === cell 11
data.head()



## === cell 12
coord_mask = (
    (data["pickup_longitude"].between(-75, -72))
    & (data["dropoff_longitude"].between(-75, -72))
    & (data["pickup_latitude"].between(40, 42))
    & (data["dropoff_latitude"].between(40, 42))
)
for c in [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]:
    data.loc[~coord_mask, c] = np.nan

data["pickup_datetime"] = pd.to_datetime(
    data["pickup_datetime"], utc=True, errors="coerce"
)
data["pickup_year"] = data["pickup_datetime"].dt.year.astype("float32")
data["pickup_month"] = data["pickup_datetime"].dt.month.astype("float32")
data["pickup_day"] = data["pickup_datetime"].dt.day.astype("float32")
data["pickup_hour"] = data["pickup_datetime"].dt.hour.astype("float32")
data["pickup_weekday"] = data["pickup_datetime"].dt.weekday.astype("float32")


def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c  # km


def bearing_np(lon1, lat1, lon2, lat2):
    """Initial bearing from (lon1,lat1) to (lon2,lat2), in radians."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.arctan2(y, x)


data["abs_lon_diff"] = (
    (data["pickup_longitude"] - data["dropoff_longitude"]).abs().astype("float32")
)
data["abs_lat_diff"] = (
    (data["pickup_latitude"] - data["dropoff_latitude"]).abs().astype("float32")
)

data["haversine_miles"] = (
    haversine_np(
        data["pickup_longitude"].astype("float64"),
        data["pickup_latitude"].astype("float64"),
        data["dropoff_longitude"].astype("float64"),
        data["dropoff_latitude"].astype("float64"),
    )
    * 0.621371
).astype("float32")

mean_lat_rad = np.radians(
    (
        (
            data["pickup_latitude"].astype("float64")
            + data["dropoff_latitude"].astype("float64")
        )
        / 2.0
    )
)
miles_per_deg_lat = 69.172  # ~miles per degree latitude
miles_per_deg_lon = 69.172 * np.cos(mean_lat_rad)

dx_miles = (data["abs_lon_diff"].astype("float64") * miles_per_deg_lon).astype(
    "float32"
)
dy_miles = (data["abs_lat_diff"].astype("float64") * miles_per_deg_lat).astype(
    "float32"
)

data["manhattan_miles"] = (dx_miles + dy_miles).astype("float32")
data["euclidean_miles"] = (
    np.sqrt(dx_miles.astype("float64") ** 2 + dy_miles.astype("float64") ** 2)
).astype("float32")

data["bearing"] = bearing_np(
    data["pickup_longitude"].astype("float64"),
    data["pickup_latitude"].astype("float64"),
    data["dropoff_longitude"].astype("float64"),
    data["dropoff_latitude"].astype("float64"),
).astype("float32")

JFK = (-73.7781, 40.6413)
LGA = (-73.8740, 40.7769)
EWR = (-74.1745, 40.6895)
NYC_CENTER = (-73.985428, 40.748817)  # Midtown (Empire State), robust proxy


def haversine_to_point_miles(lon, lat, point_lon, point_lat):
    return (
        haversine_np(
            lon.astype("float64"),
            lat.astype("float64"),
            np.float64(point_lon),
            np.float64(point_lat),
        )
        * 0.621371
    ).astype("float32")


for name, (alon, alat) in [("jfk", JFK), ("lga", LGA), ("ewr", EWR)]:
    data[f"pickup_to_{name}_miles"] = haversine_to_point_miles(
        data["pickup_longitude"], data["pickup_latitude"], alon, alat
    )
    data[f"dropoff_to_{name}_miles"] = haversine_to_point_miles(
        data["dropoff_longitude"], data["dropoff_latitude"], alon, alat
    )

data["pickup_to_center_miles"] = haversine_to_point_miles(
    data["pickup_longitude"], data["pickup_latitude"], NYC_CENTER[0], NYC_CENTER[1]
)
data["dropoff_to_center_miles"] = haversine_to_point_miles(
    data["dropoff_longitude"], data["dropoff_latitude"], NYC_CENTER[0], NYC_CENTER[1]
)

data["pc_x_haversine"] = (
    data["passenger_count"].astype("float32")
    * data["haversine_miles"].astype("float32")
).astype("float32")

data = data.drop("pickup_datetime", axis=1)

num_cols = data.select_dtypes(include=[np.number]).columns
feature_num_cols = [c for c in num_cols if c != "fare_amount"]
medians = data.loc[: len(train) - 1, feature_num_cols].median(numeric_only=True)
data[feature_num_cols] = data[feature_num_cols].fillna(medians)

data.head()



## === cell 13
train2 = data.iloc[: len(train)].copy()
test2 = data.iloc[len(train) :].copy()

y_train = train2["fare_amount"].astype(float)
X_train = train2.drop(["fare_amount", "key"], axis=1)
X_test = test2.drop(["fare_amount", "key"], axis=1)

X_train.head()



## === cell 14
from sklearn.model_selection import KFold

y_preds = []
models = []
oof_train = np.zeros((len(X_train),), dtype=np.float64)
cv = KFold(n_splits=5, shuffle=True, random_state=0)

categorical_features = []



## === cell 15
import lightgbm as lgb

params = {
    "objective": "regression",
    "metric": "l2",
    "learning_rate": 0.05,
    "max_bin": 300,
    "num_leaves": 64,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 0.1,
    "verbosity": -1,
}

for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train, y_train)):
    X_tr = X_train.iloc[train_index, :]
    X_val = X_train.iloc[valid_index, :]
    y_tr = y_train.iloc[train_index]
    y_val = y_train.iloc[valid_index]

    lgb_train = lgb.Dataset(
        X_tr, y_tr, categorical_feature=categorical_features, free_raw_data=False
    )
    lgb_eval = lgb.Dataset(
        X_val,
        y_val,
        reference=lgb_train,
        categorical_feature=categorical_features,
        free_raw_data=False,
    )

    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_eval],
        valid_names=["train", "valid"],
        num_boost_round=2000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=20),
            lgb.log_evaluation(period=50),
        ],
    )

    oof_train[valid_index] = model.predict(X_val, num_iteration=model.best_iteration)
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)

    y_preds.append(y_pred)
    models.append(model)



## === cell 16
pd.DataFrame(oof_train).to_csv("oof_train_kfold.csv", index=False)

scores = [m.best_score["valid"]["l2"] for m in models]
score = sum(scores) / len(scores)
print("===CV scores===")
print(scores)
print(score)



## === cell 17
from sklearn.metrics import mean_squared_error

y_pred_oof = oof_train
np.sqrt(mean_squared_error(y_train, y_pred_oof))



## === cell 18
len(y_preds)



## === cell 19
y_preds[0][:10]



## === cell 20
y_sub = sum(y_preds) / len(y_preds)

y_sub = np.clip(y_sub, 0.0, None)

y_sub[:10]



## === cell 21
sub_lgb = pd.DataFrame({"key": test["key"].values, "fare_amount": y_sub})
sub_lgb.to_csv("submission_lightgbm.csv", index=False)

sub_lgb.head()
