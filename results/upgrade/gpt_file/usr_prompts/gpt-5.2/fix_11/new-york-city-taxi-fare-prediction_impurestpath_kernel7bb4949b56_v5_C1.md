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

3.8

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

4.43977

# 6. Current score

5.73733

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.92064) has done: 'I fix the XGBoost inference bug caused by deprecated `ntree_limit`/`best_ntree_limit` usage in xgboost 2.x by switching to `iteration_range` and using `model.best_iteration` from early stopping. I also make the training objective compatible with newer XGBoost (`reg:squarederror` instead of deprecated `reg:linear`) without changing the overall modeling approach. To ensure the notebook runs end-to-end in a script-like Kaggle environment, I remove the IPython-only `%matplotlib inline` and keep plotting code non-blocking. Finally, I guarantee a valid `submission.csv` is written with the required `key,fare_amount` columns.'
- What this solution (achieved 5.35507) has done: 'To move RMSE down toward the 4.44 target without changing the model type or overall approach, I (1) add a minimal set of well-known taxi-fare baseline features (passenger_count, weekday, and simple NYC location/delta features) while keeping the same XGBoost training API and loss/metric, and (2) make the datetime feature extraction vectorized (same semantics, less overhead). I also set a fixed seed in XGBoost for stability and increase boosting rounds while keeping early stopping (same training approach, just allowing the model to reach a better early-stopped iteration). Finally, I keep the submission format identical (`key,fare_amount`) and still write `submission.csv`.'
- What this solution (achieved 5.33039) has done: 'We’re currently worse than the target (RMSE 5.355 vs 4.439, lower is better), so the smallest safe path is to improve generalization without changing the model type/training loop. I keep the same XGBoost training approach but (1) add two classic, minimal taxi-fare features (haversine distance to NYC center for pickup/dropoff) that are highly predictive and cheap, and (2) make train/test cleaning consistent by applying the same simple geographic/passenger filters to test inputs (without touching labels), preventing out-of-distribution test rows from causing extreme predictions. Finally, I remove rounding of predictions (it only hurts RMSE and doesn’t change semantics) while still outputting a valid `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 5.30881) has done: 'You’re currently worse than the target (RMSE 5.33 vs 4.44; lower is better), so the smallest safe move is to improve feature signal without changing the model type, loss, or training loop. I (1) add two classic, very cheap baseline features that strongly help this competition—log-distance and a simple Manhattan-distance proxy—while keeping all existing features and XGBoost training unchanged. I also (2) apply the same basic cleaning logic to the training data for passenger_count bounds (1–6), and avoid inconsistent train/test distributions; this typically reduces RMSE without altering the core approach. Finally, I keep the submission schema identical and still write `submission.csv`.'
- What this solution (achieved 5.33276) has done: 'To reduce RMSE toward the 4.44 target without changing the XGBoost approach, I’m keeping the same training loop and objective but adding two very standard, minimal-scope taxi-fare features: (1) the trip bearing (direction) and (2) the haversine “circle” distance between pickup and dropoff relative to NYC center (a simple interaction-like signal). I’m also applying the same core geographic sanity filtering to training as you already clip for test (NYC-ish lon/lat bounds), which typically removes extreme outliers that hurt RMSE while preserving the overall logic. Finally, I train on `np.log1p(fare_amount)` and invert with `expm1` at prediction time; this keeps the same squared-error objective but usually improves RMSE for heavy-tailed fares, and it’s a minimal semantic change that directly targets the metric.'
- What this solution (achieved 5.35276) has done: 'You’re currently worse than the target (RMSE 5.33276 vs 4.43977; lower is better), so the smallest safe move is to improve the model’s fit/generalization without changing the model family, loss, or overall training loop. I keep XGBoost + early stopping intact, but (1) add a few standard, very low-risk datetime/location interaction features (month, dayofmonth, and a coarse grid “area” for pickup/dropoff) that are cheap and often materially reduce RMSE in this competition. I also (2) use a slightly more expressive but still conservative XGBoost parameter set (bounded depth, subsampling/colsample, and mild regularization) while keeping the same objective/metric and early stopping, which typically nudges RMSE down without changing core semantics. Finally, I make train/test feature engineering exactly symmetric for these added features and keep the same `submission.csv` format.'
- What this solution (achieved 5.22187) has done: 'You currently don’t get a Kaggle score because the code silently drops test rows with out-of-bounds coordinates, producing a submission with fewer rows than `sample_submission.csv` (Kaggle rejects it). I keep your exact XGBoost approach and feature set, but change test handling to never drop rows: instead, compute features for all test rows and clip out-of-range values to the same bounds used in training. I also align the submission keys/predictions to `sample_submission.csv` so row count and ordering are always correct, filling any remaining invalid-feature rows with a safe fallback prediction (the median train fare) to ensure a valid submission. These are minimal, evaluation-relevant fixes (format/alignment + consistent preprocessing) that should move RMSE down from “no score” toward your target.'
- What this solution (achieved 6.06626) has done: 'Your current RMSE (5.22187) is still above the target (4.43977), so we should make a small, safe improvement without changing the XGBoost approach or loss. The biggest score drag here is typically label noise/outliers remaining after basic filters, so I add one minimal, competition-standard cleaning step: remove rows with unrealistic high fares-per-km (using your already-computed haversine distance), which usually improves generalization while keeping the same features/model. I also apply the same filtering logic consistently (only on train, never dropping test rows) and keep the exact submission alignment logic you already fixed.'
- What this solution (achieved 5.73733) has done: 'Your current RMSE (6.06626) is worse than the 4.43977 target (lower is better), so the smallest safe move is to reduce training-target noise/outliers without changing the XGBoost approach. I keep the exact same model/training loop/features, but tighten cleaning in a metric-aligned way by removing (a) obviously invalid passenger_count=0 rows and (b) extreme long-distance outliers that survive the current filters yet dominate squared error. I also make the time-based split robust by sorting and then building `X,y` from the same sorted frame to avoid any accidental misalignment between features and labels. Submission writing and key alignment remain unchanged.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-whitegrid")



## === cell 1
df_train = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/train.csv",
    nrows=1_000_000,
    parse_dates=["pickup_datetime"],
)
df_train.head()



## === cell 2
df_train.describe()



## === cell 3
df_test = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/test.csv",
    parse_dates=["pickup_datetime"],
)
df_test.head()



## === cell 4
df_test.describe()



## === cell 5
print("Old size: %d" % len(df_train))
df_train = df_train[df_train.fare_amount >= 0]
print("New size: %d" % len(df_train))



## === cell 6
print("Old size: %d" % len(df_train))
df_train = df_train.dropna(how="any", axis="rows")
print("New size: %d" % len(df_train))



## === cell 7
df_train[df_train.fare_amount < 80].fare_amount.hist(bins=100)
plt.xlabel("fare $USD")
plt.close()



## === cell 8
df_train["diff_long"] = (df_train.dropoff_longitude - df_train.pickup_longitude).abs()
df_train["diff_long"].describe()



## === cell 9
df_train["diff_lat"] = (df_train.dropoff_latitude - df_train.pickup_latitude).abs()
df_train["diff_lat"].describe()



## === cell 10
print("Old size: %d" % len(df_train))
df_train = df_train[(df_train.diff_long < 5.0) & (df_train.diff_lat < 5.0)]
print("New size: %d" % len(df_train))



## === cell 11
df_train["year"] = df_train["pickup_datetime"].dt.year
df_train["weekday"] = df_train["pickup_datetime"].dt.weekday
df_train["hour"] = df_train["pickup_datetime"].dt.hour

df_train["month"] = df_train["pickup_datetime"].dt.month
df_train["dayofmonth"] = df_train["pickup_datetime"].dt.day



## === cell 12
df_train.describe()



## === cell 13
df_train[["fare_amount", "hour"]].groupby(["hour"], as_index=False).mean().sort_values(
    by="fare_amount", ascending=False
)



## === cell 14
df_train[["fare_amount", "weekday"]].groupby(
    ["weekday"], as_index=False
).mean().sort_values(by="fare_amount", ascending=False)



## === cell 15
df_train[["fare_amount", "year"]].groupby(["year"], as_index=False).mean().sort_values(
    by="fare_amount", ascending=False
)




## === cell 16
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295  # Pi/180
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 12742 * np.arcsin(np.sqrt(a))  # 2*R*asin...




## === cell 17
df_train["distance"] = distance(
    df_train.pickup_latitude,
    df_train.pickup_longitude,
    df_train.dropoff_latitude,
    df_train.dropoff_longitude,
)



## === cell 18
plot = df_train.plot.scatter("distance", "fare_amount")
plt.close()



## === cell 19
plot = df_train[(df_train.distance < 50) & (df_train.fare_amount < 100)].plot.scatter(
    "distance", "fare_amount", alpha=0.1
)
plt.close()



## === cell 20
print("Old size: %d" % len(df_train))
df_train = df_train[(df_train.distance >= 0.1)]
print("New size: %d" % len(df_train))



## === cell 21
plot = df_train[(df_train.distance < 50) & (df_train.fare_amount < 100)].plot.scatter(
    "distance", "fare_amount", alpha=0.1
)
plt.close()



## === cell 22
print("Old size: %d" % len(df_train))
df_train = df_train[(df_train.distance <= 50)]
print("New size: %d" % len(df_train))



## === cell 23
plot = df_train.plot.scatter("distance", "fare_amount", alpha=0.1)
plt.close()



## === cell 24
print("Old size: %d" % len(df_train))
df_train = df_train[(df_train.fare_amount <= 200)]
print("New size: %d" % len(df_train))



## === cell 25
plot = df_train.plot.scatter("distance", "fare_amount", alpha=0.1)
plt.close()



## === cell 26
print("Old size: %d" % len(df_train))
df_train = df_train[
    (df_train.pickup_longitude.between(-75.0, -72.0))
    & (df_train.dropoff_longitude.between(-75.0, -72.0))
    & (df_train.pickup_latitude.between(40.0, 42.0))
    & (df_train.dropoff_latitude.between(40.0, 42.0))
]
print("New size: %d" % len(df_train))

for col in ["passenger_count"]:
    df_train[col] = pd.to_numeric(df_train[col], errors="coerce")
df_train = df_train.dropna(subset=["passenger_count"])
df_train = df_train[df_train["passenger_count"].between(1, 6)]
df_train["passenger_count"] = df_train["passenger_count"].astype(np.int16)

fare_per_km = df_train["fare_amount"] / (df_train["distance"] + 1e-6)
print("Old size (before fare/km filter): %d" % len(df_train))
df_train = df_train[fare_per_km.between(0.5, 60.0)]
print("New size (after fare/km filter): %d" % len(df_train))

print("Old size (before long-trip filter): %d" % len(df_train))
df_train = df_train[df_train["distance"] <= 35.0]
print("New size (after long-trip filter): %d" % len(df_train))

df_train["abs_lat_diff"] = (
    df_train["dropoff_latitude"] - df_train["pickup_latitude"]
).abs()
df_train["abs_lon_diff"] = (
    df_train["dropoff_longitude"] - df_train["pickup_longitude"]
).abs()

NYC_LAT, NYC_LON = 40.7141667, -74.0063889
df_train["pickup_dist_nyc"] = distance(
    df_train["pickup_latitude"], df_train["pickup_longitude"], NYC_LAT, NYC_LON
)
df_train["dropoff_dist_nyc"] = distance(
    df_train["dropoff_latitude"], df_train["dropoff_longitude"], NYC_LAT, NYC_LON
)

df_train["manhattan_dist"] = df_train["abs_lat_diff"] + df_train["abs_lon_diff"]
df_train["log_distance"] = np.log1p(df_train["distance"])

df_train["bearing"] = np.arctan2(
    (df_train["dropoff_longitude"] - df_train["pickup_longitude"]),
    (df_train["dropoff_latitude"] - df_train["pickup_latitude"]),
).astype(np.float32)

df_train["center_dist_change"] = (
    df_train["dropoff_dist_nyc"] - df_train["pickup_dist_nyc"]
).astype(np.float32)

df_train["pickup_grid_lat"] = np.floor(df_train["pickup_latitude"] * 100.0).astype(
    np.int32
)
df_train["pickup_grid_lon"] = np.floor(df_train["pickup_longitude"] * 100.0).astype(
    np.int32
)
df_train["dropoff_grid_lat"] = np.floor(df_train["dropoff_latitude"] * 100.0).astype(
    np.int32
)
df_train["dropoff_grid_lon"] = np.floor(df_train["dropoff_longitude"] * 100.0).astype(
    np.int32
)

df_train["pickup_latitude"] = df_train["pickup_latitude"].astype(np.float32)
df_train["pickup_longitude"] = df_train["pickup_longitude"].astype(np.float32)
df_train["dropoff_latitude"] = df_train["dropoff_latitude"].astype(np.float32)
df_train["dropoff_longitude"] = df_train["dropoff_longitude"].astype(np.float32)

features = [
    "year",
    "month",
    "dayofmonth",
    "weekday",
    "hour",
    "distance",
    "log_distance",
    "passenger_count",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
    "abs_lat_diff",
    "abs_lon_diff",
    "manhattan_dist",
    "pickup_dist_nyc",
    "dropoff_dist_nyc",
    "bearing",
    "center_dist_change",
    "pickup_grid_lat",
    "pickup_grid_lon",
    "dropoff_grid_lat",
    "dropoff_grid_lon",
]

X = df_train[features].values
y = np.log1p(df_train["fare_amount"].values)



## === cell 27
df_test["year"] = df_test["pickup_datetime"].dt.year
df_test["weekday"] = df_test["pickup_datetime"].dt.weekday
df_test["hour"] = df_test["pickup_datetime"].dt.hour

df_test["month"] = df_test["pickup_datetime"].dt.month
df_test["dayofmonth"] = df_test["pickup_datetime"].dt.day

for col in [
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
]:
    df_test[col] = pd.to_numeric(df_test[col], errors="coerce")

df_test["passenger_count"] = pd.to_numeric(df_test["passenger_count"], errors="coerce")

df_test["pickup_longitude"] = df_test["pickup_longitude"].clip(-75.0, -72.0)
df_test["dropoff_longitude"] = df_test["dropoff_longitude"].clip(-75.0, -72.0)
df_test["pickup_latitude"] = df_test["pickup_latitude"].clip(40.0, 42.0)
df_test["dropoff_latitude"] = df_test["dropoff_latitude"].clip(40.0, 42.0)

df_test["passenger_count"] = (
    df_test["passenger_count"].fillna(1.0).clip(lower=1, upper=6)
)

df_test["distance"] = distance(
    df_test["pickup_latitude"],
    df_test["pickup_longitude"],
    df_test["dropoff_latitude"],
    df_test["dropoff_longitude"],
)

df_test["abs_lat_diff"] = (
    df_test["dropoff_latitude"] - df_test["pickup_latitude"]
).abs()
df_test["abs_lon_diff"] = (
    df_test["dropoff_longitude"] - df_test["pickup_longitude"]
).abs()

NYC_LAT, NYC_LON = 40.7141667, -74.0063889
df_test["pickup_dist_nyc"] = distance(
    df_test["pickup_latitude"], df_test["pickup_longitude"], NYC_LAT, NYC_LON
)
df_test["dropoff_dist_nyc"] = distance(
    df_test["dropoff_latitude"], df_test["dropoff_longitude"], NYC_LAT, NYC_LON
)

df_test["diff_long"] = (df_test.dropoff_longitude - df_test.pickup_longitude).abs()
df_test["diff_lat"] = (df_test.dropoff_latitude - df_test.pickup_latitude).abs()

df_test["distance"] = df_test["distance"].clip(lower=0.0, upper=35.0)
df_test["diff_long"] = df_test["diff_long"].clip(lower=0.0, upper=5.0)
df_test["diff_lat"] = df_test["diff_lat"].clip(lower=0.0, upper=5.0)

df_test["manhattan_dist"] = df_test["abs_lat_diff"] + df_test["abs_lon_diff"]
df_test["log_distance"] = np.log1p(df_test["distance"])

df_test["pickup_latitude"] = df_test["pickup_latitude"].astype(np.float32)
df_test["pickup_longitude"] = df_test["pickup_longitude"].astype(np.float32)
df_test["dropoff_latitude"] = df_test["dropoff_latitude"].astype(np.float32)
df_test["dropoff_longitude"] = df_test["dropoff_longitude"].astype(np.float32)

df_test["bearing"] = np.arctan2(
    (df_test["dropoff_longitude"] - df_test["pickup_longitude"]),
    (df_test["dropoff_latitude"] - df_test["pickup_latitude"]),
).astype(np.float32)
df_test["center_dist_change"] = (
    df_test["dropoff_dist_nyc"] - df_test["pickup_dist_nyc"]
).astype(np.float32)

df_test["pickup_grid_lat"] = np.floor(df_test["pickup_latitude"] * 100.0).astype(
    np.int32
)
df_test["pickup_grid_lon"] = np.floor(df_test["pickup_longitude"] * 100.0).astype(
    np.int32
)
df_test["dropoff_grid_lat"] = np.floor(df_test["dropoff_latitude"] * 100.0).astype(
    np.int32
)
df_test["dropoff_grid_lon"] = np.floor(df_test["dropoff_longitude"] * 100.0).astype(
    np.int32
)

X_kaggle_test = df_test[features].values



## === cell 28
from sklearn.model_selection import train_test_split
import xgboost as xgb

df_train_sorted = df_train.sort_values("pickup_datetime").reset_index(drop=True)
X_sorted = df_train_sorted[features].values
y_sorted = np.log1p(df_train_sorted["fare_amount"].values)

split_idx = int(0.7 * len(df_train_sorted))
X_train, X_valid = X_sorted[:split_idx], X_sorted[split_idx:]
y_train, y_valid = y_sorted[:split_idx], y_sorted[split_idx:]


def XGBmodel(x_train, x_valid, y_train, y_valid):
    matrix_train = xgb.DMatrix(x_train, label=y_train)
    matrix_valid = xgb.DMatrix(x_valid, label=y_valid)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "seed": 0,
        "eta": 0.05,
        "max_depth": 8,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "gamma": 0.0,
        "tree_method": "hist",
    }

    model = xgb.train(
        params=params,
        dtrain=matrix_train,
        num_boost_round=4000,
        early_stopping_rounds=50,
        evals=[(matrix_valid, "valid")],
        verbose_eval=False,
    )
    return model


model = XGBmodel(X_train, X_valid, y_train, y_valid)

dtest = xgb.DMatrix(X_kaggle_test)
best_iter = getattr(model, "best_iteration", None)
if best_iter is None:
    pred_log = model.predict(dtest)
else:
    pred_log = model.predict(dtest, iteration_range=(0, best_iter + 1))

prediction = np.expm1(pred_log)
prediction = np.clip(prediction, 0.0, None)



## === cell 29
sample_sub = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/sample_submission.csv"
)
fallback_fare = float(df_train["fare_amount"].median())

pred_series = pd.Series(prediction, index=df_test["key"])
submission = sample_sub.copy()
submission["fare_amount"] = submission["key"].map(pred_series).astype(float)
submission["fare_amount"] = (
    submission["fare_amount"].fillna(fallback_fare).clip(lower=0.0)
)

submission.to_csv("submission.csv", index=False)
submission.head()
