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

4.00706

# 6. Current score

5.80347

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.77089) has done: 'I fix the XGBoost prediction call (the sklearn API uses `predict(X)` rather than `predict(data=...)`), which currently stops the notebook before `Y_pred` is created. I also make `Y_train` a 1D Series (instead of a single-column DataFrame) to match `XGBRegressor` expectations and avoid downstream shape quirks. Finally, I ensure the submission is created with exactly the required columns (`key`, `fare_amount`) and saved to a clearly named `.csv` file so Kaggle recognizes it. These changes are minimal, keep the same modeling approach, and should yield a valid submission and a reasonable RMSE.'
- What this solution (achieved 6.95265) has done: 'Your current score (4.77089 RMSE) is worse than the target (4.00706), so we should make small, low-risk improvements that keep the same overall XGBoost approach and feature set. The biggest gain with minimal logic change is to clean obviously bad training rows (invalid lat/lon ranges, passenger_count outliers, non-positive/huge fares), which reduces noise and typically improves RMSE without changing the model type. I also keep the exact same feature engineering but ensure train/test dtypes are consistent and apply a simple non-negative clip to predictions (fares can’t be negative), which usually helps RMSE slightly. Everything still trains an `XGBRegressor()` with default params and writes a valid `submission.csv`.'
- What this solution (achieved 5.83406) has done: 'We need to move RMSE down from 6.95 toward 4.01 (lower is better), so we should keep the same XGBoost regressor + same basic time/distance features, but remove training noise that hurts generalization. The smallest high-impact change is to add one more cleaning rule: drop rows with unrealistically long trips using a simple haversine distance computed from the same lat/lon columns you already use (no new modeling logic). I also ensure train/test feature columns are aligned in the same order and keep the non-negative clipping for valid fares. These changes typically reduce RMSE substantially without changing the overall approach or adding complexity.'
- What this solution (achieved 7.38338) has done: 'We need to move RMSE down from 5.83406 toward the target 4.00706 (lower is better), so the smallest likely win is to keep the same XGBoost regressor and the same feature set, but remove remaining high-noise rows that slip through. I add one extra, standard NYC Taxi cleanup: drop trips where pickup and dropoff coordinates are identical (near-zero distance), and constrain fares by a loose but realistic “fare per km” band computed from the same haversine distance you already use (no new model/feature logic). I also set a deterministic `random_state` on the existing `XGBRegressor()` to stabilize scoring while keeping the same default training approach and prediction semantics. The submission writing stays identical and still produces a valid `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 7.64891) has done: 'Your RMSE (7.38) is far worse than the target (4.01), so the most direct minimal fix is to correct a subtle but severe feature bug: you currently *drop* the dropoff coordinates and never add a real distance feature, so the model is missing the key signal for fare. I keep your exact XGBoost approach and your existing time + abs-diff features, but I additionally compute a haversine trip distance and add it as a feature for both train and test (same function you already use for cleaning). I also keep your cleaning and submission logic intact, only ensuring the new feature is created before dropping columns so train/test stay aligned. This should substantially reduce RMSE while preserving core logic and staying within runtime constraints.'
- What this solution (achieved 6.50395) has done: 'We need to move RMSE down from 7.64891 toward 4.00706 (lower is better), so we keep the same XGBRegressor approach and the same existing feature set, but fix a key inconsistency: the model currently drops dropoff coordinates (and keeps pickup coords), which removes essential information and also makes “latitude/longitude distance” partially redundant. The minimal, high-impact change is to keep both pickup and dropoff coordinates as numeric features (do not drop them), while preserving your existing engineered features (hour/dayofweek, abs diffs, haversine). I also ensure train/test columns are perfectly aligned and keep the same clipping and submission formatting, so it still runs end-to-end and writes a valid `submission.csv`. This is a small change to feature retention (not a new model or training loop) and should significantly reduce RMSE toward the target.'
- What this solution (achieved 6.02078) has done: 'To move RMSE down toward the 4.007 target without changing the core XGBoost approach, I make two minimal, high-impact fixes: (1) remove a likely index misalignment bug in the submission key selection by preserving `test_data["key"]` before any drops, and (2) add a single, standard feature extraction for `pickup_datetime` (year/month/day) to capture seasonal effects while keeping the same feature-engineering style you already use. I also set conservative XGBoost parameters (`n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`) that typically improve RMSE materially versus defaults without changing the model family or training loop. Finally, I keep your existing cleaning rules and prediction clipping, and ensure train/test columns remain perfectly aligned so the submission is valid.'
- What this solution (achieved 5.82558) has done: 'Your current RMSE (6.02078) is still meaningfully worse than the target (4.00706), so we should make a small, high-impact improvement without changing the overall XGBoost approach. The biggest issue is that the model is learning on the raw heavy-tailed `fare_amount`; applying a simple log1p transform during training and inverting with expm1 at prediction time often reduces RMSE substantially for this competition while keeping the same regressor/training loop. I also add a minimal, standard coordinate “centering” cleanup (remove rows with (0,0) or extreme non-NYC coordinates that slip through) and ensure the test keys stay aligned after datetime NaT drops by reindexing predictions back to the original test order. The submission format and file writing stay identical (`submission.csv` with `key,fare_amount`).'
- What this solution (achieved 4.92065) has done: 'Your current RMSE (5.82558) is worse than the target (4.00706), so we should make small, low-risk improvements that keep the same XGBoost regressor and feature set. The biggest likely win without changing the modeling approach is to bring the training labels in sync with the cleaned training rows: right now `Y_train` is taken **before** you drop invalid `pickup_datetime` rows, which silently misaligns X and y and can severely hurt RMSE. I also apply the same basic coordinate sanity filtering to the test set (without touching labels, of course) and then reindex predictions back to the original test order—this keeps submission alignment correct and avoids the model seeing extreme out-of-distribution coordinates at inference. Everything else (features, log1p/expm1, XGBRegressor params, submission format) remains the same.'
- What this solution (achieved 4.79096) has done: 'Your RMSE is worse than the target (lower is better), so the smallest likely improvement is to reduce training noise and bring train/test feature distributions closer without changing the XGBoost approach. I add two standard NYC-taxi features that use only existing columns (straight-line bearing and pickup/dropoff proximity to NYC center) and apply the *same* basic trip-distance outlier filtering to the test set to avoid out-of-distribution inputs at inference. I keep your existing cleaning rules, log1p target transform, model type, and training loop intact, and I preserve the exact submission format and key alignment. These are minimal, low-risk changes that typically move RMSE down toward ~4.0 for this competition.'
- What this solution (achieved 5.80347) has done: 'Your current RMSE (4.79096) is worse than the target (4.00706), so we should make small, low-risk changes that reduce noise and fix a likely distribution mismatch without changing the overall XGBoost + engineered-features approach. The biggest minimal win here is to stop *dropping* test rows (and then median-filling), and instead keep all test rows while only clipping extreme/out-of-range coordinates—dropping rows creates out-of-distribution “median” predictions that can hurt RMSE. I also make the train/test cleaning symmetric by not requiring `pickup_datetime` inside `_basic_clean` (datetime is handled later), and I align the distance-based outlier filtering using one consistent mask to avoid subtle array/dataframe misalignment. These changes preserve your core model, features, and log1p/expm1 target transform, but should move RMSE down toward the target band.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

BASE_INPUT = "/kaggle/input" if os.path.exists("/kaggle/input") else "../input"
print("BASE_INPUT:", BASE_INPUT)
print(os.listdir(BASE_INPUT))



## === cell 1
training_data = pd.read_csv(f"{BASE_INPUT}/train.csv", nrows=2_500_000)
test_data = pd.read_csv(f"{BASE_INPUT}/test.csv")

test_keys = test_data["key"].values



## === cell 2
training_data



## === cell 3
X_train = training_data.copy()
X_test = test_data.copy()
Y_train = training_data.copy()




## === cell 4
def _haversine_km(lon1, lat1, lon2, lat2):
    """
    Minimal, deterministic feature/cleaning helper.
    Using haversine only to filter extreme outliers improves RMSE (reduces label noise)
    while preserving the same overall modeling approach.
    """
    lon1 = np.radians(lon1.astype(np.float64))
    lat1 = np.radians(lat1.astype(np.float64))
    lon2 = np.radians(lon2.astype(np.float64))
    lat2 = np.radians(lat2.astype(np.float64))

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return 6371.0 * c  # Earth radius in km


def _bearing_rad(lon1, lat1, lon2, lat2):
    """
    Small, standard geo feature for this competition.
    Keeps core logic (tree model on engineered numeric features) but adds directional signal.
    """
    lon1 = np.radians(lon1.astype(np.float64))
    lat1 = np.radians(lat1.astype(np.float64))
    lon2 = np.radians(lon2.astype(np.float64))
    lat2 = np.radians(lat2.astype(np.float64))
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.arctan2(y, x).astype(np.float32)


def _basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(
        subset=[
            "fare_amount",
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
            "passenger_count",
        ]
    ).copy()

    df = df[(df["fare_amount"] > 0) & (df["fare_amount"] <= 500)]
    df = df[(df["passenger_count"] >= 1) & (df["passenger_count"] <= 6)]

    for col in ["pickup_longitude", "dropoff_longitude"]:
        df = df[df[col].between(-74.5, -72.8)]
    for col in ["pickup_latitude", "dropoff_latitude"]:
        df = df[df[col].between(40.0, 41.8)]

    df = df[~((df["pickup_longitude"] == 0) & (df["pickup_latitude"] == 0))].copy()
    df = df[~((df["dropoff_longitude"] == 0) & (df["dropoff_latitude"] == 0))].copy()

    dist_km = _haversine_km(
        df["pickup_longitude"].values,
        df["pickup_latitude"].values,
        df["dropoff_longitude"].values,
        df["dropoff_latitude"].values,
    )

    same_loc = (df["pickup_longitude"].values == df["dropoff_longitude"].values) & (
        df["pickup_latitude"].values == df["dropoff_latitude"].values
    )
    keep = (~same_loc) & (dist_km >= 0.05) & (dist_km <= 100.0)
    df = df.loc[df.index[keep]].copy()
    dist_km = dist_km[keep]

    fare_per_km = df["fare_amount"].values / np.maximum(dist_km, 0.1)
    df = df[(fare_per_km >= 1.0) & (fare_per_km <= 60.0)].copy()

    return df


training_data = _basic_clean(training_data)

X_train = training_data.copy()
Y_train = training_data["fare_amount"].copy()



## === cell 5
NYC_LON, NYC_LAT = -73.985428, 40.748817  # Midtown Manhattan (stable reference point)

X_train["pickup_datetime"] = pd.to_datetime(
    X_train["pickup_datetime"], errors="coerce", utc=True
)
mask_dt = X_train["pickup_datetime"].notna()
X_train = X_train.loc[mask_dt].copy()
Y_train = Y_train.loc[X_train.index].copy()

X_train["hour"] = X_train["pickup_datetime"].dt.hour.astype(np.int16)
X_train["dayofweek"] = X_train["pickup_datetime"].dt.dayofweek.astype(np.int16)
X_train["month"] = X_train["pickup_datetime"].dt.month.astype(np.int16)
X_train["year"] = X_train["pickup_datetime"].dt.year.astype(np.int16)
X_train["day"] = X_train["pickup_datetime"].dt.day.astype(np.int16)

X_train["latitude_distance"] = (
    (X_train["dropoff_latitude"] - X_train["pickup_latitude"]).abs().astype(np.float32)
)
X_train["longitude_distance"] = (
    (X_train["dropoff_longitude"] - X_train["pickup_longitude"])
    .abs()
    .astype(np.float32)
)

X_train["trip_distance_km"] = _haversine_km(
    X_train["pickup_longitude"].values,
    X_train["pickup_latitude"].values,
    X_train["dropoff_longitude"].values,
    X_train["dropoff_latitude"].values,
).astype(np.float32)

X_train["bearing"] = _bearing_rad(
    X_train["pickup_longitude"].values,
    X_train["pickup_latitude"].values,
    X_train["dropoff_longitude"].values,
    X_train["dropoff_latitude"].values,
)

X_train["pickup_to_center_km"] = _haversine_km(
    X_train["pickup_longitude"].values,
    X_train["pickup_latitude"].values,
    np.full(len(X_train), NYC_LON, dtype=np.float64),
    np.full(len(X_train), NYC_LAT, dtype=np.float64),
).astype(np.float32)

X_train["dropoff_to_center_km"] = _haversine_km(
    X_train["dropoff_longitude"].values,
    X_train["dropoff_latitude"].values,
    np.full(len(X_train), NYC_LON, dtype=np.float64),
    np.full(len(X_train), NYC_LAT, dtype=np.float64),
).astype(np.float32)

X_train = X_train.drop(
    columns=[
        "key",
        "fare_amount",
        "pickup_datetime",
    ]
)

X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0.0)



## === cell 6
test_index_original = X_test.index

for col in ["pickup_longitude", "dropoff_longitude"]:
    X_test[col] = X_test[col].clip(-74.5, -72.8)
for col in ["pickup_latitude", "dropoff_latitude"]:
    X_test[col] = X_test[col].clip(40.0, 41.8)

X_test["passenger_count"] = X_test["passenger_count"].clip(1, 6)

zero_pick = (X_test["pickup_longitude"] == 0) & (X_test["pickup_latitude"] == 0)
zero_drop = (X_test["dropoff_longitude"] == 0) & (X_test["dropoff_latitude"] == 0)
X_test.loc[zero_pick, "pickup_longitude"] = NYC_LON
X_test.loc[zero_pick, "pickup_latitude"] = NYC_LAT
X_test.loc[zero_drop, "dropoff_longitude"] = NYC_LON
X_test.loc[zero_drop, "dropoff_latitude"] = NYC_LAT

X_test["pickup_datetime"] = pd.to_datetime(
    X_test["pickup_datetime"], errors="coerce", utc=True
)
fill_dt = pd.Timestamp("2014-06-15 12:00:00", tz="UTC")
X_test["pickup_datetime"] = X_test["pickup_datetime"].fillna(fill_dt)

X_test["hour"] = X_test["pickup_datetime"].dt.hour.astype(np.int16)
X_test["dayofweek"] = X_test["pickup_datetime"].dt.dayofweek.astype(np.int16)
X_test["month"] = X_test["pickup_datetime"].dt.month.astype(np.int16)
X_test["year"] = X_test["pickup_datetime"].dt.year.astype(np.int16)
X_test["day"] = X_test["pickup_datetime"].dt.day.astype(np.int16)

X_test["latitude_distance"] = (
    (X_test["dropoff_latitude"] - X_test["pickup_latitude"]).abs().astype(np.float32)
)
X_test["longitude_distance"] = (
    (X_test["dropoff_longitude"] - X_test["pickup_longitude"]).abs().astype(np.float32)
)

X_test["trip_distance_km"] = _haversine_km(
    X_test["pickup_longitude"].values,
    X_test["pickup_latitude"].values,
    X_test["dropoff_longitude"].values,
    X_test["dropoff_latitude"].values,
).astype(np.float32)

X_test["bearing"] = _bearing_rad(
    X_test["pickup_longitude"].values,
    X_test["pickup_latitude"].values,
    X_test["dropoff_longitude"].values,
    X_test["dropoff_latitude"].values,
)

X_test["pickup_to_center_km"] = _haversine_km(
    X_test["pickup_longitude"].values,
    X_test["pickup_latitude"].values,
    np.full(len(X_test), NYC_LON, dtype=np.float64),
    np.full(len(X_test), NYC_LAT, dtype=np.float64),
).astype(np.float32)

X_test["dropoff_to_center_km"] = _haversine_km(
    X_test["dropoff_longitude"].values,
    X_test["dropoff_latitude"].values,
    np.full(len(X_test), NYC_LON, dtype=np.float64),
    np.full(len(X_test), NYC_LAT, dtype=np.float64),
).astype(np.float32)

X_test = X_test.drop(columns=["key", "pickup_datetime"])

X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0.0)

X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)



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
    random_state=42,
    n_estimators=600,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    n_jobs=4,
)

y_train_log = np.log1p(Y_train.astype(np.float64).values)

model.fit(X_train, y_train_log)

y_pred_log = model.predict(X_test)
Y_pred = np.expm1(y_pred_log)

Y_pred = np.clip(Y_pred, 0, None)

if len(Y_pred) != len(test_keys):
    pred_series = pd.Series(Y_pred, index=X_test.index)
    pred_series = pred_series.reindex(test_index_original)
    fill_value = float(np.nanmedian(pred_series.values))
    pred_series = pred_series.fillna(fill_value)
    Y_pred = pred_series.values



## === cell 9
submission = pd.DataFrame({"key": test_keys, "fare_amount": Y_pred})

submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with columns:", submission.columns.tolist())
print("Submission rows:", len(submission))
