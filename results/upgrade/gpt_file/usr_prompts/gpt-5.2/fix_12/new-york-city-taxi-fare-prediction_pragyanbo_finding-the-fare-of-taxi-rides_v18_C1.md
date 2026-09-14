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

4.05769

# 6. Current score

7.31708

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.6816) has done: 'I fix the XGBoost prediction call to be compatible with xgboost==2.0.3 by using `iteration_range` (or omitting deprecated `best_ntree_limit`) so inference runs and `xgbm_pred` is created. I also remove the notebook-only `%matplotlib inline` magic that crashes in a .py/script environment, while keeping plotting optional. Finally, I import the missing `operator` module so the feature-importance cell runs, and I ensure the submission is written as a valid `.csv` with the required `key,fare_amount` columns.'
- What this solution (achieved 5.68385) has done: 'To move RMSE down toward your 4.05769 target (from 5.6816; lower is better), the smallest effective change is to keep the same XGBoost approach but fix avoidable data/split issues that are currently costing accuracy. I (1) make the train/validation split deterministic (so performance is stable and comparable), (2) add the standard NYC coordinate bounding-box filter to remove obvious GPS outliers (same core features, just cleaner data), and (3) include a light `min_child_weight/subsample/colsample_bytree` regularization setup while keeping the same model type, objective, and training flow. I also clip negative predictions to 0 to avoid nonsensical fares that can inflate RMSE. The script still write a valid `key,fare_amount` submission CSV.'
- What this solution (achieved 5.92496) has done: 'You’re currently worse than the target RMSE (5.68385 vs 4.05769; lower is better), so we should make small, legitimate improvements that usually reduce error without changing the modeling approach. The biggest low-risk gain here is to make your “distance” feature correct: your current function mixes kilometers (12742) with miles (0.6213712), shrinking distances and hurting the model. I switch to a consistent Haversine-in-km distance and also filter out zero-distance rides in training (common noisy records) while keeping the same features/model/training flow. Everything else (XGBoost reg:squarederror, same feature set, same split/training loop) is preserved, and the script still writes a valid `key,fare_amount` submission CSV.'
- What this solution (achieved 6.65275) has done: 'We’re currently worse than the target RMSE (5.92496 vs 4.05769; lower is better), so the smallest likely improvement is to reduce label noise and improve generalization without changing your overall XGBoost approach. I (1) add a standard fare upper-bound filter to remove extreme outliers that inflate RMSE, (2) add two lightweight, competition-standard geospatial features (absolute lat/lon deltas) while keeping the same distance + time feature logic, and (3) remove early stopping (since it can underfit on this small 1M sample) and instead train the fixed number of rounds, using the same objective/metric/training flow. The submission path/format remains identical and still write a valid `key,fare_amount` CSV.'
- What this solution (achieved 6.3088) has done: 'Your current issue is that `current_score` is None, so the priority is to reliably generate a valid submission CSV end-to-end in this environment. The smallest score-relevant improvement (without changing the model type, loss, training loop, or feature set) is to ensure `test_df` always retains all rows/keys: filtering the test set by NYC bounds can silently drop rows and invalidate/misalign the submission. I keep the same NYC-bounds filtering for training (to reduce noise) but stop filtering the test rows; instead I clip test coordinates to the bounds just for feature computation stability. I also make the datetime parsing robust and explicitly check the submission row count matches the original test row count before writing the CSV.'
- What this solution (achieved 6.31818) has done: 'To move your RMSE down toward the 4.05769 target (from 6.3088; lower is better) with minimal disruption, I keep the same XGBoost reg:squarederror training flow and feature set but fix two high-impact, low-risk issues: (1) compute `distance` only after clipping test coordinates (your current order computes distance on unclipped/out-of-bounds test points), and (2) add the standard `fare_amount >= 2.5` floor filter to remove noisy/invalid training labels that inflate RMSE. I also make the time-based split robust by recomputing the datetime series after filtering so it stays perfectly aligned with `X`. Everything still runs end-to-end and writes a valid `key,fare_amount` submission CSV.'
- What this solution (achieved 6.53187) has done: 'Your current RMSE (6.31818) is worse than the target (4.05769; lower is better), so we should make the smallest legitimate improvements that typically reduce error without changing the overall XGBoost approach. The biggest low-risk win here is fixing the train/validation split: right now it’s based on a quantile of a 1M-row sample that may not be time-representative, which can push the model toward a suboptimal fit; instead we do a deterministic random split so the training set is more representative while keeping the same training loop and parameters. We also add two standard, minimal datetime features (day-of-week and month) which usually help materially for NYC fares without changing the model type or loss. Finally, we keep your existing cleaning/filters and submission writing, and we ensure the feature list is used consistently for both train and test.'
- What this solution (achieved 7.56324) has done: 'To move RMSE down toward your 4.05769 target (from 6.53187; lower is better) with minimal disruption, I’m keeping the same XGBoost reg:squarederror training flow and feature set, but fixing two high-impact data issues that commonly dominate this competition’s error. First, I remove extreme/invalid GPS points using the standard competition-wide coordinate/fare filters (including longitude/latitude sanity, passenger_count bounds, and a more realistic max-distance cap) while keeping your same engineered features. Second, I add the canonical “haversine + manhattan distance” pair (still just distance features—no model/loop change) and clip negative predictions as you already do. These are small, legitimate adjustments that typically reduce RMSE substantially without changing the core modeling approach or submission semantics.'
- What this solution (achieved 6.64499) has done: 'Your current RMSE (7.56324; lower is better) is far worse than the target (4.05769), so we should make small but high-impact fixes without changing the XGBoost approach or feature set. The biggest issue is that you are training with a random split and evaluating on a random holdout, but the competition test distribution is time-dependent; switching to a deterministic time-based split typically improves generalization for this dataset while keeping the same training loop/model. I also remove duplicated/contradictory filters (passenger_count and fare floor are applied twice) to avoid unintended sample loss, and I ensure the evaluation set used for monitoring is consistent and aligned with the filtered dataframe. Everything else (features, objective, num_rounds, prediction/clipping, and submission writing) stays the same.'
- What this solution (achieved 7.31708) has done: 'We’re currently worse than the target (6.64499 vs 4.05769; lower is better), so the smallest likely improvement is to keep the same XGBoost reg:squarederror setup and features, but fix the split so training better matches the competition’s time distribution. Right now you train only on the oldest ~70% of sampled rows; instead, we do a deterministic random split for validation while training the model on all available rows for the final fit (same algorithm/params/rounds), which typically lowers public RMSE. I also ensure `pickup_datetime` stays perfectly aligned with `train_df` after all filtering (currently it can be out-of-sync due to earlier filtering before recomputation), which can silently corrupt the time-based split logic. Submission writing, feature engineering, model type, loss, and boosting rounds remain unchanged.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import operator

sns.set_style("whitegrid")



## === cell 1
INPUT_DIR = "../input"
if not os.path.exists(INPUT_DIR):
    if os.path.exists("/kaggle/input"):
        INPUT_DIR = "/kaggle/input"
    elif os.path.exists("/kaggle/data"):
        INPUT_DIR = "/kaggle/data"

train_path = os.path.join(INPUT_DIR, "train.csv")
test_path = os.path.join(INPUT_DIR, "test.csv")

RANDOM_STATE = 42
train_df = pd.read_csv(train_path, nrows=1000000)



## === cell 2
train_df.shape



## === cell 3
test_df = pd.read_csv(test_path)



## === cell 4
test_df.shape



## === cell 5
train_df.head(5)



## === cell 6
train_df.isnull().sum()



## === cell 7
train_df.dropna(inplace=True)



## === cell 8
train_df.describe()



## === cell 9
train_df = train_df[train_df["fare_amount"] >= 2.5]



## === cell 10
train_df.shape




## === cell 11
def distance(lat1, lon1, lat2, lon2):
    lat1 = np.asarray(lat1, dtype=np.float64)
    lon1 = np.asarray(lon1, dtype=np.float64)
    lat2 = np.asarray(lat2, dtype=np.float64)
    lon2 = np.asarray(lon2, dtype=np.float64)

    r = 6371.0088  # Earth mean radius in kilometers
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    dphi = np.deg2rad(lat2 - lat1)
    dlambda = np.deg2rad(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * (
        np.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * r * np.arcsin(np.sqrt(a))




## === cell 12
nyc_bounds = {
    "lon_min": -74.5,
    "lon_max": -72.8,
    "lat_min": 40.0,
    "lat_max": 41.8,
}

train_df = train_df[
    (train_df["passenger_count"] >= 1) & (train_df["passenger_count"] <= 6)
]

train_df = train_df[
    (train_df["pickup_longitude"].between(nyc_bounds["lon_min"], nyc_bounds["lon_max"]))
    & (
        train_df["dropoff_longitude"].between(
            nyc_bounds["lon_min"], nyc_bounds["lon_max"]
        )
    )
    & (
        train_df["pickup_latitude"].between(
            nyc_bounds["lat_min"], nyc_bounds["lat_max"]
        )
    )
    & (
        train_df["dropoff_latitude"].between(
            nyc_bounds["lat_min"], nyc_bounds["lat_max"]
        )
    )
]

test_df = test_df.copy()
for col, lo, hi in [
    ("pickup_longitude", nyc_bounds["lon_min"], nyc_bounds["lon_max"]),
    ("dropoff_longitude", nyc_bounds["lon_min"], nyc_bounds["lon_max"]),
    ("pickup_latitude", nyc_bounds["lat_min"], nyc_bounds["lat_max"]),
    ("dropoff_latitude", nyc_bounds["lat_min"], nyc_bounds["lat_max"]),
]:
    test_df[col] = test_df[col].clip(lo, hi)



## === cell 13
train_df["distance"] = distance(
    train_df.pickup_latitude,
    train_df.pickup_longitude,
    train_df.dropoff_latitude,
    train_df.dropoff_longitude,
)

test_df["distance"] = distance(
    test_df.pickup_latitude,
    test_df.pickup_longitude,
    test_df.dropoff_latitude,
    test_df.dropoff_longitude,
)

train_df["distance_manhattan"] = distance(
    train_df.pickup_latitude,
    train_df.pickup_longitude,
    train_df.pickup_latitude,
    train_df.dropoff_longitude,
) + distance(
    train_df.pickup_latitude,
    train_df.pickup_longitude,
    train_df.dropoff_latitude,
    train_df.pickup_longitude,
)

test_df["distance_manhattan"] = distance(
    test_df.pickup_latitude,
    test_df.pickup_longitude,
    test_df.pickup_latitude,
    test_df.dropoff_longitude,
) + distance(
    test_df.pickup_latitude,
    test_df.pickup_longitude,
    test_df.dropoff_latitude,
    test_df.pickup_longitude,
)



## === cell 14
train_df = train_df[train_df["distance"] > 0]
train_df = train_df[
    train_df["distance"] < 60
]  # keeps airport trips while removing absurd GPS outliers



## === cell 15
train_df.describe()



## === cell 16
train_df = train_df.copy()



## === cell 17
train_df = train_df[(train_df["fare_amount"] >= 2.5) & (train_df["fare_amount"] <= 200)]



## === cell 18
train_pickup_dt = pd.to_datetime(
    train_df["pickup_datetime"], errors="coerce", utc=False
)
test_pickup_dt = pd.to_datetime(test_df["pickup_datetime"], errors="coerce", utc=False)

train_df = train_df.loc[train_pickup_dt.notna()].copy()
train_pickup_dt = pd.to_datetime(
    train_df["pickup_datetime"], errors="coerce", utc=False
)

test_pickup_dt = test_pickup_dt.fillna(pd.Timestamp("2010-01-01 00:00:00"))

train_df["hour"] = train_pickup_dt.dt.hour
train_df["year"] = train_pickup_dt.dt.year
train_df["weekday"] = train_pickup_dt.dt.weekday
train_df["month"] = train_pickup_dt.dt.month

test_df["hour"] = test_pickup_dt.dt.hour
test_df["year"] = test_pickup_dt.dt.year
test_df["weekday"] = test_pickup_dt.dt.weekday
test_df["month"] = test_pickup_dt.dt.month



## === cell 19
train_df["abs_lon_diff"] = (
    train_df["pickup_longitude"] - train_df["dropoff_longitude"]
).abs()
train_df["abs_lat_diff"] = (
    train_df["pickup_latitude"] - train_df["dropoff_latitude"]
).abs()
test_df["abs_lon_diff"] = (
    test_df["pickup_longitude"] - test_df["dropoff_longitude"]
).abs()
test_df["abs_lat_diff"] = (
    test_df["pickup_latitude"] - test_df["dropoff_latitude"]
).abs()



## === cell 20
train_df["mid_lon"] = (
    train_df["pickup_longitude"] + train_df["dropoff_longitude"]
) / 2.0
train_df["mid_lat"] = (train_df["pickup_latitude"] + train_df["dropoff_latitude"]) / 2.0
test_df["mid_lon"] = (test_df["pickup_longitude"] + test_df["dropoff_longitude"]) / 2.0
test_df["mid_lat"] = (test_df["pickup_latitude"] + test_df["dropoff_latitude"]) / 2.0



## === cell 21
feat_cols_s = [
    "distance",
    "distance_manhattan",
    "abs_lon_diff",
    "abs_lat_diff",
    "mid_lon",
    "mid_lat",
    "passenger_count",
    "hour",
    "weekday",
    "month",
    "year",
]

X = train_df[feat_cols_s]
y = train_df["fare_amount"]



## === cell 22
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE
)



## === cell 23
import xgboost as xgb




## === cell 24
def XGBoost(X_train, X_test, y_train, y_test, num_rounds=800):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.1,
        "max_depth": 8,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": RANDOM_STATE,
    }

    return xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_rounds,
        evals=[(dtest, "test")],
        verbose_eval=False,
    )




## === cell 25
xgbm = XGBoost(X_train, X_test, y_train, y_test)

dtrain_full = xgb.DMatrix(X, label=y)
params_full = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "eta": 0.1,
    "max_depth": 8,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": RANDOM_STATE,
}
xgbm_full = xgb.train(
    params=params_full,
    dtrain=dtrain_full,
    num_boost_round=800,
    evals=[(dtrain_full, "train")],
    verbose_eval=False,
)

dtest_submit = xgb.DMatrix(test_df[feat_cols_s])

best_iter = getattr(xgbm_full, "best_iteration", None)
if best_iter is None:
    xgbm_pred = xgbm_full.predict(dtest_submit)
else:
    xgbm_pred = xgbm_full.predict(dtest_submit, iteration_range=(0, best_iter + 1))

xgbm_pred = np.clip(xgbm_pred, 0, None)



## === cell 26
if len(xgbm_pred) != len(test_df):
    raise ValueError(
        f"Prediction length {len(xgbm_pred)} does not match test rows {len(test_df)}"
    )

submission = pd.DataFrame(
    {"key": test_df["key"].astype(str), "fare_amount": xgbm_pred},
    columns=["key", "fare_amount"],
)

submission.to_csv("XGboost_regression.csv", index=False)

submission.head()



## === cell 27
importance = xgbm_full.get_score()
importance = sorted(importance.items(), key=operator.itemgetter(1))
df = pd.DataFrame(importance, columns=["feature", "score"])

ax = df.plot(kind="barh", x="feature", y="score", legend=False, figsize=(8, 5))
plt.title("Feature Importance")
plt.tight_layout()
plt.show()
