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
haversine==2.9.0
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

4.01592

# 6. Current score

5.95398

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.42618) has done: 'Your score is far above the target (RMSE 14.44 vs 4.02, lower is better), so we should improve materially while keeping the same overall approach (basic cleaning + engineered distance/time features + RandomForestRegressor). The biggest issue is a bug in the latitude filter (`dropoff_latitude < 72`), which lets many invalid rows through and hurts fit; fixing it to `< 42` typically gives a large RMSE drop with minimal logic change. Next, your `add_distance_feature` uses slow row-iteration and can be inconsistent/fragile; switching to a vectorized haversine formula preserves the same feature semantics but improves correctness and stability. Finally, set RandomForest parameters for stronger generalization (more trees + sensible depth/min leaf) without changing the model type, and keep the submission format unchanged.'
- What this solution (achieved 5.35054) has done: 'Your current RMSE (5.42618) is worse than the target (4.01592), so we should improve it with the smallest safe tweaks that don’t change the overall approach (same filters/feature set + RandomForest). The biggest low-risk gain on this competition is to avoid training on huge-fare outliers that are mostly label noise; adding a standard fare cap filter (e.g., `< 250`) typically drops RMSE materially without changing the modeling logic. To keep train/test feature handling consistent and prevent any NaT-derived missing values at inference, we also ensure datetime parsing fills missing `day_of_week/hour_of_day` with a sentinel and keep the distance feature robust. Finally, we increase `n_estimators` slightly for a steadier fit (still the same model), staying within the time budget for 100k rows.'
- What this solution (achieved 5.25082) has done: 'Your current RMSE (5.35054) is worse than the target (4.01592), so we should improve it with minimal, low-risk tweaks while keeping the same core approach (same cleaning + same engineered features + RandomForestRegressor). The biggest likely gain is removing remaining mislabeled/noisy examples by tightening the fare filter to the commonly used range for this competition, which usually reduces RMSE without changing semantics. Next, we clip extreme coordinate values in a slightly narrower NYC bounding box to reduce obvious GPS noise while staying consistent with your existing geo-filter logic. Finally, we train the same RandomForest but on all cleaned rows (not just the split’s training fold) for the final model before predicting test, which usually improves the leaderboard score with no architectural change.'
- What this solution (achieved 5.49913) has done: 'Your current RMSE (5.25082) is worse than the target (4.01592), so we should make small, low-risk data-cleaning tweaks that usually yield a meaningful drop on this competition while keeping the same feature set and the same RandomForest approach. The biggest likely gain with minimal semantic change is (1) filtering out rides with near-zero distance (many are data errors with non-trivial fares) by increasing the distance floor slightly, and (2) removing very long-distance rides (often noisy/outlier-heavy) by adding a conservative upper distance cap. These are simple additional row filters after your existing haversine feature and generally improve RMSE without changing model architecture/training loop. Everything else (paths, features, model type, submission format) is kept the same, and the script still writes `submission.csv`.'
- What this solution (achieved 5.46532) has done: 'Your current RMSE (5.49913) is worse than the target (4.01592), so we should improve it with the smallest safe data-quality tweaks while keeping the same features and the same RandomForestRegressor. The most likely issue is that your distance filter removes a lot of legitimate short NYC rides (many are <0.5 km), which tends to hurt RMSE; lowering the minimum distance threshold is a minimal semantic change and usually yields an immediate gain. Next, your fare cap of 100 is unnecessarily tight and can bias the model; expanding it to a standard cap (250) keeps outliers controlled while improving fit. Everything else (NYC bbox filters, engineered features, model type/params, and submission format) is kept the same.'
- What this solution (achieved 4.77378) has done: 'Your current RMSE (5.46532) is still well above the target (4.01592), so we should improve materially but with minimal, safe changes while keeping the same overall approach (same cleaning + engineered distance/time features + RandomForestRegressor). The biggest low-risk gain is to stop forcing coordinates into a tight NYC bounding box during training, because the competition’s test set includes some out-of-box-but-valid rides and your current filter can bias the model; we keep only a looser “valid coordinate” sanity filter instead. We also remove the hard upper distance cap (30 km) and instead only drop near-zero-distance anomalies, since long rides exist and are predictable from distance—capping them can hurt leaderboard RMSE. Everything else (feature set, model type, training loop, submission format/path) stays the same and still writes `submission.csv`.'
- What this solution (achieved 5.24729) has done: 'Your current RMSE (4.77378) is still above the target (4.01592), so we should make a small, safe improvement without changing the overall approach (same cleaning + same distance/time features + RandomForestRegressor). The biggest low-risk gain on this competition is tightening obviously-invalid GPS points to a *broad* NYC metro bounding box (rather than worldwide [-180,180]/[-90,90]), because global-valid coordinates still include many junk rows that hurt fit. We also apply the same distance floor filter to the test set (just to keep feature distribution consistent) and clip negative predictions to 0 since fares can’t be negative (usually a minor RMSE improvement). Everything else (data size, features, model type/params, training flow, and writing `submission.csv`) stays the same.'
- What this solution (achieved 5.30855) has done: 'Your current RMSE (5.24729) is still above the target (4.01592), so we should make a small, low-risk improvement that keeps the same data pipeline (100k rows, same engineered features, same RandomForestRegressor) but reduces noise in training. The biggest likely gain with minimal semantic change is to remove obvious “mismatched” trips where the haversine distance is large but the fare is implausibly small (and vice versa), because these are typically data errors that inflate RMSE. We add two conservative, distance-aware fare filters after computing `distance`, keeping the existing fare/passenger/bbox/distance-floor logic intact. Everything else (model type/params, training flow, and writing `submission.csv`) remains unchanged.'
- What this solution (achieved 5.95248) has done: 'Your current RMSE (5.30855) is still well above the target (4.01592), so the safest way to move toward the target while keeping the same RandomForest + distance/time features is to reduce label/GPS noise in the *training* sample. I add two standard, minimal NYC Taxi Fare cleaning steps: remove rows with identical pickup/dropoff coordinates (common bad records) and filter out implausibly high-speed trips using the already-available datetime + haversine distance (this keeps core features/model unchanged but removes strong outliers that inflate RMSE). I also apply the exact same basic coordinate sanity filter to the test set (without dropping rows) by clipping to the training bounds to avoid extrapolation spikes, which typically improves leaderboard RMSE without changing the model. Everything else (paths, feature set, model type/params, training flow, submission format) stays the same and still writes `submission.csv`.'
- What this solution (achieved 5.95398) has done: 'We need to move RMSE down from 5.95248 toward 4.01592 (lower is better), so we should improve performance with minimal, low-risk changes while keeping the same RandomForest + distance/time features. The biggest issue in your current pipeline is the *test-time feature mismatch*: you compute `distance` before clipping coordinates, so many test rows get a distance derived from raw (possibly out-of-bounds) coordinates while the model sees clipped coordinates—this distribution shift can hurt leaderboard RMSE. I fix this by clipping coordinates first and then recomputing `distance` on the clipped coordinates (training already uses filtered NYC bounds, so this aligns train/test semantics). I also apply the same simple passenger_count cleaning to the test set (clip to [1,8]) to avoid out-of-range values causing weird predictions, without changing the model or feature set.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

print(os.listdir("../input"))



## === cell 1
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor



## === cell 2
train_path = "../input/train.csv"
test_path = "../input/test.csv"

train_df = pd.read_csv(train_path, nrows=100000)
train_df.head(2)



## === cell 3
train_df.dtypes



## === cell 4
train_df.shape



## === cell 5
print(train_df.isnull().sum())



## === cell 6
print("Old size: %d" % len(train_df))
train_df = train_df.dropna(how="any", axis="rows")
print("New size: %d" % len(train_df))



## === cell 7
train_df.describe()



## === cell 8
train_df = train_df[(train_df.fare_amount > 0) & (train_df.fare_amount <= 250)]
train_df = train_df[(train_df.passenger_count > 0) & (train_df.passenger_count < 9)]
train_df.shape



## === cell 9
train_df = train_df[
    (train_df.pickup_longitude >= -75.5)
    & (train_df.pickup_longitude <= -72.8)
    & (train_df.dropoff_longitude >= -75.5)
    & (train_df.dropoff_longitude <= -72.8)
    & (train_df.pickup_latitude >= 40.0)
    & (train_df.pickup_latitude <= 41.8)
    & (train_df.dropoff_latitude >= 40.0)
    & (train_df.dropoff_latitude <= 41.8)
]
train_df.shape




## === cell 10
def add_distance_feature(df):
    lat1 = np.radians(df["pickup_latitude"].astype(float).values)
    lon1 = np.radians(df["pickup_longitude"].astype(float).values)
    lat2 = np.radians(df["dropoff_latitude"].astype(float).values)
    lon2 = np.radians(df["dropoff_longitude"].astype(float).values)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    earth_radius_km = 6371.0
    df["distance"] = earth_radius_km * c




## === cell 11
add_distance_feature(train_df)
train_df.head(2)




## === cell 12
def add_time_and_day_feature(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df["day_of_week"] = dt.dt.dayofweek.fillna(-1).astype(np.int16)
    df["hour_of_day"] = dt.dt.hour.fillna(-1).astype(np.int16)


add_time_and_day_feature(train_df)
train_df.head(2)



## === cell 13
train_df.describe()



## === cell 14
train_df = train_df[(train_df.distance >= 0.05)]
train_df = train_df[~((train_df.distance < 0.2) & (train_df.fare_amount < 2.5))]
train_df = train_df[~((train_df.distance > 5.0) & (train_df.fare_amount < 3.0))]
train_df.describe()



## === cell 15
same_point = (train_df["pickup_longitude"] == train_df["dropoff_longitude"]) & (
    train_df["pickup_latitude"] == train_df["dropoff_latitude"]
)
train_df = train_df[~same_point]

dt = pd.to_datetime(train_df["pickup_datetime"], errors="coerce")
train_df = train_df[dt.notna()].copy()
train_df["pickup_datetime_parsed"] = dt[dt.notna()]

fare_per_km = train_df["fare_amount"].values / np.maximum(
    train_df["distance"].values, 1e-6
)
train_df = train_df[(fare_per_km >= 0.5) & (fare_per_km <= 50.0)]

train_df.drop(columns=["pickup_datetime_parsed"], inplace=True)
train_df.describe()



## === cell 16
X = train_df[
    [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
        "distance",
        "day_of_week",
        "hour_of_day",
    ]
]
y = train_df["fare_amount"].values

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.2, random_state=1
)
print(X_train.shape, y_train.shape)

reg = RandomForestRegressor(
    n_estimators=500,
    random_state=1,
    n_jobs=-1,
    min_samples_leaf=2,
    max_features="sqrt",
)

reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print("Validation RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

reg.fit(X, y)

test_df = pd.read_csv(test_path)

test_df["pickup_longitude"] = test_df["pickup_longitude"].clip(-75.5, -72.8)
test_df["dropoff_longitude"] = test_df["dropoff_longitude"].clip(-75.5, -72.8)
test_df["pickup_latitude"] = test_df["pickup_latitude"].clip(40.0, 41.8)
test_df["dropoff_latitude"] = test_df["dropoff_latitude"].clip(40.0, 41.8)

add_distance_feature(test_df)
add_time_and_day_feature(test_df)

test_df["passenger_count"] = test_df["passenger_count"].clip(1, 8)

test_df["distance"] = np.maximum(test_df["distance"].astype(float).values, 0.05)

test_X = test_df[
    [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
        "distance",
        "day_of_week",
        "hour_of_day",
    ]
]

test_y_predictions = reg.predict(test_X)
test_y_predictions = np.clip(test_y_predictions, 0.0, None)

submission = pd.DataFrame(
    {"key": test_df.key, "fare_amount": test_y_predictions},
    columns=["key", "fare_amount"],
)

submission.to_csv("submission.csv", index=False)

print(os.listdir("."))
print(submission.head())
