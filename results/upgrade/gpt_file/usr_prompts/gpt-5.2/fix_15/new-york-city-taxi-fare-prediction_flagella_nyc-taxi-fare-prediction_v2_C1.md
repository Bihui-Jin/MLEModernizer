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

3.11

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

3.7787

# 6. Current score

6.24117

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.09489) has done: 'You’re dropping rows from `test` during cleaning, which makes `new_pred` shorter than the original `key` list and causes the length-mismatch error when building the submission. I keep the same XGBoost regressor and the same feature set, but change test-time cleaning to *not drop rows*; instead we flag invalid rows and still produce predictions for all 9914 keys (filling invalid rows with a simple fallback), so the submission is always aligned. I also fix the incorrect `StandardScaler` usage (you were fitting but never transforming your train/test matrices properly) in a minimal way that doesn’t change the core model. Finally, I write a valid `submission1.csv` with exactly `key` and `fare_amount` columns.'
- What this solution (achieved 5.06994) has done: 'Your current RMSE (5.09489) is worse than the target (3.7787), so we should cautiously improve generalization without changing the core model/feature set. The largest low-risk gain here is making train-time filtering consistent with your test-time “invalid” logic: right now you drop invalid geos/passenger_count in train, but you *don’t* drop or flag out-of-range fares (and you also never remove negative/huge fares), which injects noisy labels and hurts RMSE. I add minimal fare_amount sanity filtering (standard for this competition) and use a deterministic median fallback computed on the cleaned y_train; everything else (XGBRegressor, scaling, features, train/test split, submission schema) stays the same. This should move the score down toward the target band without altering the modeling approach.'
- What this solution (achieved 5.06994) has done: 'To move RMSE down toward your target with minimal risk, I keep the same XGBRegressor and the same raw feature set, but make train/test preprocessing consistent by applying the exact same invalid-row rules to the training sample (instead of only dropping some zero/out-of-range rows). This reduces label noise from clearly-bad coordinates and passenger counts that still slip into training and typically improves RMSE without changing the model. I also compute the fallback fare from the cleaned training target (same as you already do) and keep the submission aligned to all test keys. Everything still runs end-to-end within the same paths and writes `submission1.csv` with the required columns.'
- What this solution (achieved 5.63047) has done: 'I fix the runtime errors caused by using the deprecated/invalid `clip(min=...)` argument in pandas 2.2 by switching to the correct `clip(lower=...)`, which unblocks feature engineering for both train and test. I also remove a heavy, unnecessary re-read of `train.csv` inside the test preprocessing (it wastes time/memory and isn’t used for the model), while keeping the same model, features, and training loop. Finally, I ensure the test feature pipeline produces `test_scaled` deterministically so predictions are created for all test keys and a valid `submission1.csv` is written with the required columns.'
- What this solution (achieved 5.62686) has done: 'Your current RMSE (5.63047) is worse than the target (3.7787), so we should make small, safe improvements that reduce label noise and better align training with how you score on Kaggle, without changing the model or feature set. The biggest low-risk win is to train on a larger sample (still within runtime) while keeping the same cleaning rules; 1M rows is often too small/biased for this competition. I also refit the final model on all cleaned training data (not just the train split) after validating RMSE, which usually improves leaderboard RMSE with no semantic change. Finally, I keep the exact same test-time “don’t drop rows” logic and submission alignment so the CSV remains valid.'
- What this solution (achieved 5.4456) has done: 'To reduce RMSE toward your target without changing the core model/feature set, I keep the same XGBRegressor and the same two features (`passenger_count`, `log1p_distance_km`), but remove redundant/contradictory train cleaning and make the remaining cleaning consistent and vectorized. The main score-improving change is to cap extreme `passenger_count` values instead of dropping them (reduces distribution shift between train/test) while still filtering obviously invalid geo coordinates and implausible fares. I also keep the same scaling approach but ensure the fallback fare is computed from the cleaned training target (not the pre-cleaned one) and that test invalid rows are handled identically without dropping any test rows, preserving submission alignment.'
- What this solution (achieved 5.55668) has done: 'You’re currently worse than the target (RMSE 5.4456 vs 3.7787, lower is better), so we should make a small, low-risk improvement that keeps the exact same model type and feature set but reduces a known source of noise: very short trips with “normal” fares and other mismatched label/feature pairs. I add one minimal, standard cleaning rule on the training data only: drop rows where computed trip distance is ~zero but fare is non-trivially positive (a common artifact in this dataset that hurts RMSE when you only use distance + passenger_count). I keep the same XGBRegressor, same scaler usage, same test “don’t drop rows” alignment and fallback logic, and still write `submission1.csv` with the required columns. This should move RMSE down toward your target without changing the core approach.'
- What this solution (achieved 5.63392) has done: 'Your current RMSE (5.55668) is worse than the target (3.7787), so we should make a small, competition-standard improvement that reduces label noise while keeping the same XGBRegressor and the same two-feature setup (`passenger_count`, `log1p_distance_km`). The biggest low-risk issue left is that the model is trained on raw fare targets, which are heavy-tailed; switching to a log-target during training and then inverting at prediction time usually improves RMSE materially for this specific competition without changing features or model type. I keep the same cleaning rules, scaling, and submission alignment, but train/predict in log-space and compute the fallback in the same space for consistency. The output remains a valid `submission1.csv` with `key,fare_amount` for all 9914 rows.'
- What this solution (achieved 5.64724) has done: 'You’re currently worse than the target (RMSE 5.63392 vs 3.7787; lower is better), so we should make a small, legitimate improvement without changing the model type, features, or training loop. The biggest remaining issue is that the log-target model can still produce negative fares after `expm1` (when predicted log is < 0), and your current post-processing clips them to 0, which is an unrealistic value that tends to hurt RMSE; we instead floor fares to a small positive minimum (standard for this competition) while keeping the same semantics. We also make the fallback fare computation consistent with your train cleaning by using the median of the cleaned **raw** fare (then log1p), and we cap train-time `log1p_distance_km` to a high quantile to reduce the influence of a few extreme-distance rows (still same single distance feature, just a robust cap). These are minimal post-processing/robustness changes that typically reduce RMSE without altering the core approach, and the script still write a valid `submission1.csv` for all 9914 keys.'
- What this solution (achieved 5.6756) has done: 'Your current RMSE (5.64724) is worse than the target (3.7787), so we should make the smallest legitimate improvements that reduce label noise and better match typical NYC taxi pricing without changing your model, features, or training loop. The main issue is that with only `passenger_count` and `log1p_distance_km`, the model struggles because the target includes a strong “fixed base fare + per-distance” structure; we can capture that without adding new model complexity by transforming the target to “fare above base” during training and then adding the base back at prediction time. This keeps the exact same XGBRegressor setup and feature set, but makes the learning problem simpler and usually lowers RMSE. I also make the fallback fare consistent with this base-fare transform and keep your existing invalid-row handling and submission alignment unchanged.'
- What this solution (achieved 5.67579) has done: 'We’re currently worse than the target (RMSE 5.6756 vs 3.7787; lower is better), so we need a small, legitimate improvement without changing your core model or feature set. The biggest low-risk gain is to make the base-fare transform actually help by replacing the hard `clip(lower=0)` residual with a symmetric residual (allows the model to correct below/near-base fares instead of forcing everything to be ≥ base), while still flooring final predictions to a reasonable minimum fare for submission. I also fix a pandas-2.2 incompatibility bug in your fallback computation (`clip(min=...)`), and I apply the exact same distance cap computed on train to the test feature to keep distributions aligned. Everything else (XGBRegressor, n_estimators, two features, scaling, train/test invalid handling, output schema) stays the same and it still writes `submission1.csv`.'
- What this solution (achieved 5.67806) has done: 'We’re currently worse than the target (RMSE 5.67579 vs 3.7787; lower is better), so we should make a small, legitimate improvement that reduces label noise without changing your model type, feature set, or training loop. The highest-impact minimal fix is to remove remaining outlier label/feature pairs by adding a standard “speed sanity” filter on the training data using the already-available `pickup_datetime` (no new features are used by the model; it’s only for cleaning). This typically lowers RMSE for this competition because extreme fares for tiny distances or impossible implied speeds distort learning when you only use distance and passenger_count. Everything else stays the same: same two features, same scaling, same XGBRegressor config, same base-fare residual log transform, same test-time invalid handling, and we still write a valid `submission1.csv`.'
- What this solution (achieved 6.24117) has done: 'To move RMSE down toward your target without changing the model type or feature set, I keep the exact same XGBRegressor + two features, but fix one inconsistency: you train the final model on `X_full_scaled` from `scaler_full`, yet the target `y_log` was created before dropping rows later during datetime/distance cleaning; we recompute `y_log` from the cleaned `train` right before fitting so `X` and `y_log` are perfectly aligned. I also compute the fallback residual from the same cleaned target used for training (instead of the pre-cleaned `y.values`) so invalid-test handling matches the learned distribution. Finally, I add a minimal, competition-standard speed sanity filter that uses `pickup_datetime` only for cleaning (not as a feature) and should reduce label noise, which typically improves RMSE while preserving your core approach.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as MSE



## === cell 2
train = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/train.csv", nrows=3000000
)
test = pd.read_csv("../input/new-york-city-taxi-fare-prediction/test.csv")
train.shape, test.shape



## === cell 3
train.head()



## === cell 4
train.isnull().sum()



## === cell 5
train = train.dropna(axis="rows")
test.isnull().sum()



## === cell 6
train.head()



## === cell 7
train["fare_amount"].describe()



## === cell 8
pass



## === cell 9
train.drop(["key"], axis=1, inplace=True)



## === cell 10
train.dropna(inplace=True)

required_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

train["passenger_count"] = train["passenger_count"].clip(lower=1, upper=6)

invalid_train = (
    train[required_cols].isna().any(axis=1)
    | (train["pickup_longitude"] == 0)
    | (train["pickup_latitude"] == 0)
    | (train["dropoff_longitude"] == 0)
    | (train["dropoff_latitude"] == 0)
    | ((train["passenger_count"] == 0))
    | (train["pickup_longitude"] < -75)
    | (train["pickup_longitude"] > -72)
    | (train["pickup_latitude"] < 40)
    | (train["pickup_latitude"] > 42)
    | (train["dropoff_longitude"] < -75)
    | (train["dropoff_longitude"] > -72)
    | (train["dropoff_latitude"] < 40)
    | (train["dropoff_latitude"] > 42)
)
train = train.loc[~invalid_train].copy()

train = train[(train["fare_amount"] > 0) & (train["fare_amount"] <= 250)].copy()




## === cell 11
def haversine_km(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1.astype(float))
    lat1 = np.radians(lat1.astype(float))
    lon2 = np.radians(lon2.astype(float))
    lat2 = np.radians(lat2.astype(float))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return 6371.0 * c


dist_km = haversine_km(
    train["pickup_longitude"].values,
    train["pickup_latitude"].values,
    train["dropoff_longitude"].values,
    train["dropoff_latitude"].values,
)
train["distance_km"] = dist_km

train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], errors="coerce", utc=True
)
train = train.dropna(subset=["pickup_datetime"]).copy()

tiny_km = 0.2
train = train.loc[
    ~((train["distance_km"] < tiny_km) & (train["fare_amount"] > 50.0))
].copy()

eps_km = 0.05  # ~50 meters: treat as effectively zero-distance
train = train.loc[
    ~((train["distance_km"] < eps_km) & (train["fare_amount"] > 2.5))
].copy()

train = train.sort_values("pickup_datetime").copy()
dt_sec = train["pickup_datetime"].diff().dt.total_seconds().abs()
speed_kmh = train["distance_km"] / (dt_sec / 3600.0)
train = train.loc[speed_kmh.isna() | (speed_kmh < 200.0)].copy()

train["log1p_distance_km"] = np.log1p(train["distance_km"].clip(lower=0.0))
logdist_cap = float(train["log1p_distance_km"].quantile(0.999))
train["log1p_distance_km"] = train["log1p_distance_km"].clip(upper=logdist_cap)

train.drop(
    [
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "distance_km",
    ],
    axis=1,
    inplace=True,
)



## === cell 12
X = train.drop("fare_amount", axis=1)
y = train["fare_amount"].astype(float)

BASE_FARE = 2.5
y_resid = y - BASE_FARE
y_log = np.sign(y_resid) * np.log1p(np.abs(y_resid))

X_train, X_test, y_train_log, y_test_log = train_test_split(
    X, y_log, test_size=0.2, random_state=12
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



## === cell 13
xgb_r = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=400,
    seed=123,
)
xgb_r.fit(X_train_scaled, y_train_log)



## === cell 14
y_pred_log = xgb_r.predict(X_test_scaled)

y_pred_resid = np.sign(y_pred_log) * (np.expm1(np.abs(y_pred_log)))
y_true_resid = np.sign(y_test_log) * (np.expm1(np.abs(y_test_log)))

y_pred = BASE_FARE + y_pred_resid
y_true = BASE_FARE + y_true_resid

y_pred = np.clip(y_pred, 0.0, 250.0)
y_true = np.clip(y_true, 0.0, 250.0)

rmse = np.sqrt(MSE(y_true, y_pred))
print("RMSE : % f" % (rmse))



## === cell 15
scaler_full = StandardScaler()
X_full_scaled = scaler_full.fit_transform(X)

xgb_r_full = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=400,
    seed=123,
)
xgb_r_full.fit(X_full_scaled, y_log)



## === cell 16
test_raw = pd.read_csv("../input/new-york-city-taxi-fare-prediction/test.csv")
test_feat = test_raw.copy()

test_feat.drop(["pickup_datetime"], axis=1, inplace=True)

keys = test_feat["key"].copy()
test_feat.drop(["key"], axis=1, inplace=True)



## === cell 17
required_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
for c in required_cols:
    if c not in test_feat.columns:
        raise ValueError(f"Missing expected column in test: {c}")

test_feat = test_feat.copy()
test_feat["passenger_count"] = test_feat["passenger_count"].clip(lower=1, upper=6)

invalid = (
    test_feat[required_cols].isna().any(axis=1)
    | (test_feat["pickup_longitude"] == 0)
    | (test_feat["pickup_latitude"] == 0)
    | (test_feat["dropoff_longitude"] == 0)
    | (test_feat["dropoff_latitude"] == 0)
    | ((test_feat["passenger_count"] == 0))
    | (test_feat["pickup_longitude"] < -75)
    | (test_feat["pickup_longitude"] > -72)
    | (test_feat["pickup_latitude"] < 40)
    | (test_feat["pickup_latitude"] > 42)
    | (test_feat["dropoff_longitude"] < -75)
    | (test_feat["dropoff_longitude"] > -72)
    | (test_feat["dropoff_latitude"] < 40)
    | (test_feat["dropoff_latitude"] > 42)
)

train_feature_medians = X.median(numeric_only=True)

test_feat_filled = test_feat.copy()

nyc_defaults = {
    "pickup_longitude": -73.985428,
    "pickup_latitude": 40.748817,
    "dropoff_longitude": -73.985428,
    "dropoff_latitude": 40.748817,
}

for col in [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]:
    if col in nyc_defaults:
        test_feat_filled[col] = test_feat_filled[col].fillna(nyc_defaults[col])
    else:
        test_feat_filled[col] = test_feat_filled[col].fillna(
            test_feat_filled[col].median()
        )

for col, val in nyc_defaults.items():
    test_feat_filled.loc[invalid, col] = val

test_feat_filled.loc[invalid, "passenger_count"] = float(
    train_feature_medians.get("passenger_count", 1.0)
)

test_feat_filled["distance_km"] = haversine_km(
    test_feat_filled["pickup_longitude"].values,
    test_feat_filled["pickup_latitude"].values,
    test_feat_filled["dropoff_longitude"].values,
    test_feat_filled["dropoff_latitude"].values,
)

test_feat_filled["log1p_distance_km"] = np.log1p(
    test_feat_filled["distance_km"].clip(lower=0.0)
)

test_feat_filled["log1p_distance_km"] = test_feat_filled["log1p_distance_km"].clip(
    upper=logdist_cap
)

test_model = test_feat_filled[["passenger_count", "log1p_distance_km"]].copy()
test_model = test_model.fillna(train_feature_medians)

test_scaled = scaler_full.transform(test_model)



## === cell 18
new_pred_log = xgb_r_full.predict(test_scaled)

fallback_resid_raw = float(np.median((y.astype(float).values - BASE_FARE)))
fallback_resid_log = float(
    np.sign(fallback_resid_raw) * np.log1p(np.abs(fallback_resid_raw))
)

new_pred_log = np.asarray(new_pred_log, dtype=float)
new_pred_log[invalid.values] = fallback_resid_log

new_pred_resid = np.sign(new_pred_log) * (np.expm1(np.abs(new_pred_log)))
new_pred = BASE_FARE + new_pred_resid

new_pred = np.clip(new_pred, BASE_FARE, 250.0)



## === cell 19
submission = pd.DataFrame({"key": keys.values, "fare_amount": new_pred})
submission.head()



## === cell 20
submission.to_csv("submission1.csv", index=False)
print("Wrote submission1.csv with shape:", submission.shape)
print("Columns:", submission.columns.tolist())
print(submission.head())
