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

3.33042

# 6. Current score

272.21596

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.63306) has done: 'I keep your XGBoost approach and existing features, but fix two issues that are strongly hurting RMSE: (1) your train/test feature columns are not guaranteed to align after `get_dummies` (different year/day values in train vs test), and (2) you are rounding predictions to cents, which adds extra error under RMSE. I also remove early stopping (it violates your “no early stopping” requirement) while keeping the same number of boosting rounds so the training approach stays the same. Finally, I update the model objective to the non-deprecated equivalent and set a fixed random seed for stable splits and training.'
- What this solution (achieved 5.52924) has done: 'You’re already using a reasonable XGBoost regressor, so the biggest remaining RMSE loss comes from (1) the extremely slow per-row distance/feature engineering (risking timeouts and limiting practical training size) and (2) mild target noise/outliers that survive current filters. I keep the exact same feature set and training approach, but vectorize the distance and datetime-derived features (same formulas, just faster) so you can safely raise `nrows` (more signal) within the 600s budget. I also add one minimal, standard NYC-taxi cleanup: remove zero-distance rides with non-trivial fare (and vice versa) which are frequent data errors and disproportionately hurt RMSE. The submission file path/format stays identical and still writes `finaloutput.csv`.'
- What this solution (achieved 4.89876) has done: 'The crash comes from filtering rows out of `test` (zeros/out-of-bounds), which makes `testkey` still length 9914 while `pred` is only for the filtered test rows. I keep your modeling and features identical, but stop filtering rows out of the test set; instead, I only filter training data and then impute/clip invalid test coordinates so every original test `key` gets a prediction. I also ensure the submission is written with exactly 9914 rows in the original test order, producing a valid `finaloutput.csv` for Kaggle.'
- What this solution (achieved 5.0312) has done: 'You’re currently well above the target RMSE (4.90 vs 3.33), so we should make small, reliable improvements without changing the overall XGBoost+engineered-features approach. The biggest minimal win is to add two standard, cheap features that don’t alter the training loop: absolute lat/lon deltas and a simple “manhattan” distance proxy; these usually reduce error for this competition while keeping your model/feature engineering style intact. I also add one more common, low-risk cleanup that mainly removes egregious label noise: drop rows where `dist` is near-zero but fare is non-trivially high (and keep your existing filter), which tends to help RMSE. Everything else (data paths, sampling size, dummies alignment via concatenation, training procedure, submission writing) stays the same.'
- What this solution (achieved 4.92614) has done: 'To move RMSE down toward your 3.33042 target (lower is better) while keeping your XGBoost + current feature set/training loop intact, I make two minimal changes that typically reduce NYC Taxi RMSE: add a couple of very standard datetime-derived features (hour and month) using the same extraction style you already use, and one airport proximity feature (distance to JFK/LGA/EWR) computed with the same haversine function (no new modeling approach). I also apply a small, common additional training cleanup that removes unrealistically low fares (e.g., fare < 2.5) which are mostly label noise and disproportionately hurt RMSE, without touching the test set. Everything else (data paths, sampling size, dummies alignment via concatenation, XGBoost training call, submission writing) stays the same.'
- What this solution (achieved 4.9637) has done: 'I fix the crash in the airport-distance feature generation by making `haversine_vectorized` accept scalar lon/lat inputs (like JFK[0]) as well as arrays, without changing the feature formulas. This is a correctness/runtime fix only; it restores the intended features and allows training/inference to run end-to-end. I also make datetime parsing robust to any malformed strings by using `errors="coerce"` and then dropping resulting NA rows in train only, preventing rare downstream failures while keeping test row count intact. The rest of the pipeline (data paths, filters, features, XGBoost training, and submission writing) stays the same and produce `finaloutput.csv`.'
- What this solution (achieved 5.40059) has done: 'We need to move RMSE down from 4.9637 toward 3.33042 (lower is better), so we make small, reliable improvements without changing your XGBoost training loop or feature “style”. The biggest low-risk gain here is to align train/test preprocessing more strictly: apply the same coordinate cleanup/imputation to train (not just test) instead of dropping many borderline rows, which can bias the model and waste data. We also add one standard, very cheap signal feature that fits your existing logic—`log1p(dist)` (and `log1p(manhattan_km)`)—which helps the model handle the strong nonlinearity between distance and fare without changing the model architecture. Finally, we add a minimal, common NYC-taxi cleanup to remove obvious outliers in the *target* distribution (e.g., absurdly high fare-per-km), which typically reduces RMSE with minimal disruption.'
- What this solution (achieved 5.25692) has done: 'Your current RMSE (5.40059) is still far above the target (3.33042, lower is better), so we should make small, reliable improvements without changing the XGBoost training loop or overall feature-engineering style. The biggest likely issue is that `passenger_count` is being treated as a numeric ordinal, while in this competition it behaves much more like a categorical feature; one-hot encoding it (using the same `get_dummies` pipeline you already have) is a minimal change that often yields a noticeable RMSE drop. I also add two extremely standard, cheap, and non-disruptive features (`bearing` and `center_lat/center_lon`) computed from existing coordinates, which typically helps the model explain direction-dependent fare patterns without changing the model class. Everything else (data loading, cleaning approach, haversine/manhattan/log features, train/test dummy alignment by concatenation, XGBoost params/rounds, and writing `finaloutput.csv`) stays intact.'
- What this solution (achieved 5.26384) has done: 'To reduce RMSE from 5.25692 toward your 3.33042 target (lower is better) while keeping the same XGBoost training approach and overall feature style, I make three minimal, high-signal adjustments: (1) tighten training-data cleaning with a very standard NYC bounding box and a stricter fare-per-km filter to remove label noise that disproportionately hurts RMSE, (2) add one lightweight but strong geographic feature (pickup/dropoff distance to Manhattan center) without changing the model class or loop, and (3) keep the test set unfiltered but apply the exact same coordinate imputation/clipping so submission rows stay aligned. These changes keep the core logic intact (same feature engineering paradigm + xgboost.train with fixed rounds) but typically bring RMSE down meaningfully for this competition.'
- What this solution (achieved 5.10373) has done: 'Your RMSE (5.26384) is still far above the 3.33042 target (lower is better), so we should make small, reliable improvements without changing the overall XGBoost+engineered-features approach. The biggest issue here is likely that the model is overfitting noisy/outlier-heavy training data and also missing a key nonlinearity: fare scales roughly with (distance × time-of-day/traffic), so adding the standard `sqrt_dist` and `dist*passenger_count` interaction features is a minimal extension of your current feature style. Additionally, your current target cleaning uses a global fare-per-km rule that can inadvertently remove many valid long trips (e.g., airport/highway) and keep some short-trip label noise; we replace it with a slightly more robust, commonly used set of *minimal* outlier rules (still simple thresholds, no new modeling). Everything else (data loading, coordinate handling, datetime features, dummy alignment, XGBoost params/training call, and submission writing) stays the same.'
- What this solution (achieved 5.10677) has done: 'To move RMSE down from 5.10373 toward your 3.33042 target (lower is better) without changing the XGBoost training approach, I make two minimal, competition-standard data/feature adjustments that usually give a reliable RMSE drop. First, I add a simple `min_airport_km` feature derived from the airport-distance features you already compute (no new feature “family”, just an aggregation) so the model can better handle airport trips. Second, I apply one more very common, low-risk training-only cleanup: remove rows with extreme `fare_per_km` spikes for very short trips (where label noise dominates RMSE), while keeping the test set intact and aligned. Everything else (paths, rows read, feature engineering style, dummy alignment, model params/rounds, and submission writing) stays the same.'
- What this solution (achieved 5.10695) has done: 'We need to move RMSE down from 5.10677 toward 3.33042 (lower is better), so the smallest reliable gain without changing your XGBoost approach is to (1) add the standard NYC baseline feature `abs(bearing)` (same bearing already computed, just a simple transform), and (2) add a very common training-only cleanup removing unrealistic “too-fast” trips using pickup_datetime and your existing haversine distance (these are mostly data errors that inflate RMSE). This keeps the same model class, same training API/call, same feature-engineering style, and doesn’t filter the test set (submission alignment stays correct). Everything else—including file paths, dummy alignment strategy, boosting rounds, and submission writing—remains unchanged.'
- What this solution (achieved 5.07907) has done: 'Your current RMSE (5.10695, lower is better) is still far above the target (3.33042), so we should make a small, reliable improvement without changing the overall XGBoost training loop or feature family. The biggest minimal win left is to make the bearing feature physically correct: bearing should depend on longitude *scaled by cos(latitude)*, and using raw degree deltas distorts direction information; we keep the same “bearing/abs_bearing” features but compute them with proper scaling. We also add a standard “manhattan bearing” computed from your existing `manhattan_km` components (same feature style, very cheap) to better capture grid-like travel patterns. Everything else (rows read, cleaning rules, get_dummies alignment, XGBoost params/rounds, and submission writing to `finaloutput.csv`) stays intact.'
- What this solution (achieved 272.21596) has done: 'You’re still far above the target RMSE (5.079 vs 3.330, lower is better), so we should make a small, reliable improvement without changing the overall XGBoost training approach. The biggest minimal win here is to fix a core target issue: `fare_amount` has a very long tail and heteroscedastic noise; switching to the standard `reg:pseudohubererror` objective (still an XGBoost built-in regression objective with the same training loop) typically reduces RMSE by being less sensitive to outliers while remaining stable. To avoid accidentally removing too much useful signal, I also slightly relax the most aggressive short-trip outlier filter (`fare_per_km < 80`) to a more permissive threshold while keeping the rest of your cleaning and all features unchanged. Everything else (data paths, feature engineering, dummy alignment, training call/rounds, and submission writing) stays intact and still outputs `finaloutput.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import xgboost
import os

print(os.listdir("../input"))



## === cell 1
TRAIN_NROWS = 6_000_000

train_path = "../input/train.csv"
test_path = "../input/test.csv"

usecols_train = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
usecols_test = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

df = pd.read_csv(train_path, nrows=TRAIN_NROWS, usecols=usecols_train)
test = pd.read_csv(test_path, usecols=usecols_test)



## === cell 2
testkey = test["key"].copy()



## === cell 3
df = df.dropna(how="any", axis="rows")



## === cell 4
coord_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]

for col in coord_cols:
    df[col] = df[col].replace(0.0, np.nan)

df = df.dropna(subset=coord_cols).copy()



## === cell 5
l = df[
    (df.pickup_latitude > 41.0)
    | (df.pickup_latitude < 40.4)
    | (df.dropoff_latitude > 41.0)
    | (df.dropoff_latitude < 40.4)
    | (df.pickup_longitude > -73.6)
    | (df.pickup_longitude < -74.3)
    | (df.dropoff_longitude > -73.6)
    | (df.dropoff_longitude < -74.3)
    | (df.dropoff_longitude > -73.6)
    | (df.dropoff_longitude < -74.3)
].index
df = df.drop(l, axis=0)



## === cell 6
z = df[
    (df.fare_amount > 300.0)
    | (df.fare_amount < 2.5)
    | (df.passenger_count > 7.0)
    | (df.passenger_count < 1.0)
].index
df = df.drop(z, axis=0)



## === cell 7
for c in [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]:
    df[c] = df[c].astype("float32")
    test[c] = test[c].astype("float32")
df["passenger_count"] = df["passenger_count"].astype("int16")
test["passenger_count"] = test["passenger_count"].astype("int16")
df["fare_amount"] = df["fare_amount"].astype("float32")

len(df)




## === cell 8
def haversine_vectorized(lon1, lat1, lon2, lat2):
    lon1 = np.asarray(lon1, dtype="float64")
    lat1 = np.asarray(lat1, dtype="float64")
    lon2 = np.asarray(lon2, dtype="float64")
    lat2 = np.asarray(lat2, dtype="float64")

    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (np.sin(dlat / 2.0) ** 2) + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return 6373.0 * c




## === cell 9
for col in coord_cols:
    test[col] = test[col].replace(0.0, np.nan)

train_medians = df[coord_cols].median(numeric_only=True)

df[coord_cols] = df[coord_cols].fillna(train_medians)
test[coord_cols] = test[coord_cols].fillna(train_medians)

for _d in (df, test):
    _d["pickup_latitude"] = _d["pickup_latitude"].clip(40.4, 41.0)
    _d["dropoff_latitude"] = _d["dropoff_latitude"].clip(40.4, 41.0)
    _d["pickup_longitude"] = _d["pickup_longitude"].clip(-74.3, -73.6)
    _d["dropoff_longitude"] = _d["dropoff_longitude"].clip(-74.3, -73.6)



## === cell 10
df["dist"] = haversine_vectorized(
    df["pickup_longitude"].values,
    df["pickup_latitude"].values,
    df["dropoff_longitude"].values,
    df["dropoff_latitude"].values,
).astype("float32")
test["dist"] = haversine_vectorized(
    test["pickup_longitude"].values,
    test["pickup_latitude"].values,
    test["dropoff_longitude"].values,
    test["dropoff_latitude"].values,
).astype("float32")



## === cell 11
df["abs_lon_diff"] = np.abs(df["dropoff_longitude"] - df["pickup_longitude"]).astype(
    "float32"
)
df["abs_lat_diff"] = np.abs(df["dropoff_latitude"] - df["pickup_latitude"]).astype(
    "float32"
)
test["abs_lon_diff"] = np.abs(
    test["dropoff_longitude"] - test["pickup_longitude"]
).astype("float32")
test["abs_lat_diff"] = np.abs(
    test["dropoff_latitude"] - test["pickup_latitude"]
).astype("float32")

lat_mean_train = ((df["pickup_latitude"] + df["dropoff_latitude"]) / 2.0).astype(
    "float32"
)
lat_mean_test = ((test["pickup_latitude"] + test["dropoff_latitude"]) / 2.0).astype(
    "float32"
)
df["manhattan_km"] = (
    (df["abs_lat_diff"] * 111.0)
    + (
        df["abs_lon_diff"]
        * (111.0 * np.cos(np.radians(lat_mean_train.astype("float64"))))
    )
).astype("float32")
test["manhattan_km"] = (
    (test["abs_lat_diff"] * 111.0)
    + (
        test["abs_lon_diff"]
        * (111.0 * np.cos(np.radians(lat_mean_test.astype("float64"))))
    )
).astype("float32")

df["log_dist"] = np.log1p(df["dist"]).astype("float32")
test["log_dist"] = np.log1p(test["dist"]).astype("float32")
df["log_manhattan_km"] = np.log1p(df["manhattan_km"]).astype("float32")
test["log_manhattan_km"] = np.log1p(test["manhattan_km"]).astype("float32")

lat_mean_rad_train = np.radians(lat_mean_train.astype("float64"))
lat_mean_rad_test = np.radians(lat_mean_test.astype("float64"))

dy_train = (df["dropoff_latitude"] - df["pickup_latitude"]).astype("float64")
dx_train = (
    (df["dropoff_longitude"] - df["pickup_longitude"]).astype("float64")
) * np.cos(lat_mean_rad_train)
df["bearing"] = np.arctan2(dy_train, dx_train).astype("float32")

dy_test = (test["dropoff_latitude"] - test["pickup_latitude"]).astype("float64")
dx_test = (
    (test["dropoff_longitude"] - test["pickup_longitude"]).astype("float64")
) * np.cos(lat_mean_rad_test)
test["bearing"] = np.arctan2(dy_test, dx_test).astype("float32")

df["abs_bearing"] = np.abs(df["bearing"]).astype("float32")
test["abs_bearing"] = np.abs(test["bearing"]).astype("float32")

man_dx_train = (df["abs_lon_diff"].astype("float64")) * (
    111.0 * np.cos(lat_mean_rad_train)
)
man_dy_train = (df["abs_lat_diff"].astype("float64")) * 111.0
df["manhattan_bearing"] = np.arctan2(man_dy_train, man_dx_train).astype("float32")

man_dx_test = (test["abs_lon_diff"].astype("float64")) * (
    111.0 * np.cos(lat_mean_rad_test)
)
man_dy_test = (test["abs_lat_diff"].astype("float64")) * 111.0
test["manhattan_bearing"] = np.arctan2(man_dy_test, man_dx_test).astype("float32")

df["center_lat"] = ((df["pickup_latitude"] + df["dropoff_latitude"]) / 2.0).astype(
    "float32"
)
df["center_lon"] = ((df["pickup_longitude"] + df["dropoff_longitude"]) / 2.0).astype(
    "float32"
)
test["center_lat"] = (
    (test["pickup_latitude"] + test["dropoff_latitude"]) / 2.0
).astype("float32")
test["center_lon"] = (
    (test["pickup_longitude"] + test["dropoff_longitude"]) / 2.0
).astype("float32")

df["sqrt_dist"] = np.sqrt(df["dist"].clip(0.0)).astype("float32")
test["sqrt_dist"] = np.sqrt(test["dist"].clip(0.0)).astype("float32")

df["dist_x_passenger"] = (df["dist"] * df["passenger_count"].astype("float32")).astype(
    "float32"
)
test["dist_x_passenger"] = (
    test["dist"] * test["passenger_count"].astype("float32")
).astype("float32")



## === cell 12
MANHATTAN = (-73.9855, 40.7580)  # Times Square-ish

df["pickup_manhattan_km"] = haversine_vectorized(
    df["pickup_longitude"].values,
    df["pickup_latitude"].values,
    MANHATTAN[0],
    MANHATTAN[1],
).astype("float32")
df["dropoff_manhattan_km"] = haversine_vectorized(
    df["dropoff_longitude"].values,
    df["dropoff_latitude"].values,
    MANHATTAN[0],
    MANHATTAN[1],
).astype("float32")

test["pickup_manhattan_km"] = haversine_vectorized(
    test["pickup_longitude"].values,
    test["pickup_latitude"].values,
    MANHATTAN[0],
    MANHATTAN[1],
).astype("float32")
test["dropoff_manhattan_km"] = haversine_vectorized(
    test["dropoff_longitude"].values,
    test["dropoff_latitude"].values,
    MANHATTAN[0],
    MANHATTAN[1],
).astype("float32")



## === cell 13
bad_dist_fare = df[
    ((df["dist"] < 0.01) & (df["fare_amount"] > 5.0))
    | ((df["dist"] > 0.5) & (df["fare_amount"] < 2.5))
].index
df = df.drop(bad_dist_fare, axis=0)

df = df[~((df["dist"] < 0.2) & (df["fare_amount"] > 50.0))].copy()
df = df[df["dist"] < 100.0].copy()

eps = 1e-3
fare_per_km = df["fare_amount"] / (df["dist"] + eps)
df = df[fare_per_km < 120.0].copy()

df = df[~((df["dist"] < 0.5) & (fare_per_km > 40.0))].copy()



## === cell 14
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
test["pickup_datetime"] = pd.to_datetime(test["pickup_datetime"], errors="coerce")
df = df.dropna(subset=["pickup_datetime"]).copy()



## === cell 15
trip_hours = (
    df["pickup_datetime"].dt.hour.astype("float32")
    + df["pickup_datetime"].dt.minute.astype("float32") / 60.0
)
df = df[~((df["dist"] > 30.0) & (df["fare_amount"] < 20.0))].copy()
fare_per_km = df["fare_amount"] / (df["dist"] + 1e-3)
df = df[~((df["dist"] < 1.0) & (fare_per_km > 60.0))].copy()



## === cell 16
df["latenights"] = (df["pickup_datetime"].dt.hour < 5).astype(np.int8)
test["latenights"] = (test["pickup_datetime"].dt.hour < 5).astype(np.int8)

df["weekday"] = (df["pickup_datetime"].dt.weekday > 4).astype(np.int8)
test["weekday"] = (test["pickup_datetime"].dt.weekday > 4).astype(np.int8)

df["year"] = df["pickup_datetime"].dt.year.astype(np.int16)
test["year"] = test["pickup_datetime"].dt.year.astype(np.int16)

df["day"] = df["pickup_datetime"].dt.day.astype(np.int8)
test["day"] = test["pickup_datetime"].dt.day.astype(np.int8)

df["hour"] = df["pickup_datetime"].dt.hour.astype(np.int8)
test["hour"] = test["pickup_datetime"].dt.hour.astype(np.int8)

df["month"] = df["pickup_datetime"].dt.month.astype(np.int8)
test["month"] = test["pickup_datetime"].dt.month.astype(np.int8)



## === cell 17
JFK = (-73.7781, 40.6413)
LGA = (-73.8740, 40.7769)
EWR = (-74.1745, 40.6895)

df["pickup_jfk_km"] = haversine_vectorized(
    df["pickup_longitude"].values, df["pickup_latitude"].values, JFK[0], JFK[1]
).astype("float32")
df["dropoff_jfk_km"] = haversine_vectorized(
    df["dropoff_longitude"].values, df["dropoff_latitude"].values, JFK[0], JFK[1]
).astype("float32")
df["pickup_lga_km"] = haversine_vectorized(
    df["pickup_longitude"].values, df["pickup_latitude"].values, LGA[0], LGA[1]
).astype("float32")
df["dropoff_lga_km"] = haversine_vectorized(
    df["dropoff_longitude"].values, df["dropoff_latitude"].values, LGA[0], LGA[1]
).astype("float32")
df["pickup_ewr_km"] = haversine_vectorized(
    df["pickup_longitude"].values, df["pickup_latitude"].values, EWR[0], EWR[1]
).astype("float32")
df["dropoff_ewr_km"] = haversine_vectorized(
    df["dropoff_longitude"].values, df["dropoff_latitude"].values, EWR[0], EWR[1]
).astype("float32")

test["pickup_jfk_km"] = haversine_vectorized(
    test["pickup_longitude"].values, test["pickup_latitude"].values, JFK[0], JFK[1]
).astype("float32")
test["dropoff_jfk_km"] = haversine_vectorized(
    test["dropoff_longitude"].values, test["dropoff_latitude"].values, JFK[0], JFK[1]
).astype("float32")
test["pickup_lga_km"] = haversine_vectorized(
    test["pickup_longitude"].values, test["pickup_latitude"].values, LGA[0], LGA[1]
).astype("float32")
test["dropoff_lga_km"] = haversine_vectorized(
    test["dropoff_longitude"].values, test["dropoff_latitude"].values, LGA[0], LGA[1]
).astype("float32")
test["pickup_ewr_km"] = haversine_vectorized(
    test["pickup_longitude"].values, test["pickup_latitude"].values, EWR[0], EWR[1]
).astype("float32")
test["dropoff_ewr_km"] = haversine_vectorized(
    test["dropoff_longitude"].values, test["dropoff_latitude"].values, EWR[0], EWR[1]
).astype("float32")



## === cell 18
df["min_airport_km"] = np.minimum.reduce(
    [
        df["pickup_jfk_km"].values,
        df["dropoff_jfk_km"].values,
        df["pickup_lga_km"].values,
        df["dropoff_lga_km"].values,
        df["pickup_ewr_km"].values,
        df["dropoff_ewr_km"].values,
    ]
).astype("float32")
test["min_airport_km"] = np.minimum.reduce(
    [
        test["pickup_jfk_km"].values,
        test["dropoff_jfk_km"].values,
        test["pickup_lga_km"].values,
        test["dropoff_lga_km"].values,
        test["pickup_ewr_km"].values,
        test["dropoff_ewr_km"].values,
    ]
).astype("float32")



## === cell 19
feat = df.drop(["key", "pickup_datetime"], axis=1)
test_feat = test.drop(["key", "pickup_datetime"], axis=1)

label = feat["fare_amount"].copy()



## === cell 20
combined = pd.concat(
    [feat.drop(["fare_amount"], axis=1), test_feat], axis=0, ignore_index=True
)

combined = pd.get_dummies(
    combined,
    columns=["year", "day", "hour", "month", "passenger_count"],
    prefix=["year", "day", "hour", "month", "passenger_count"],
)

X = combined.iloc[: len(feat), :].copy()
X_test = combined.iloc[len(feat) :, :].copy()

for col in X.columns:
    if X[col].dtype == "float64":
        X[col] = X[col].astype("float32")
        X_test[col] = X_test[col].astype("float32")



## === cell 21
xtr, xts, ytr, yts = train_test_split(X, label, test_size=0.25, random_state=42)

xgbtrain = xgboost.DMatrix(xtr, label=ytr)
xgbtest = xgboost.DMatrix(xts, label=yts)
xgbfinaltest = xgboost.DMatrix(X_test)



## === cell 22
params = {
    "eval_metric": "rmse",
    "objective": "reg:pseudohubererror",
    "seed": 42,
    "tree_method": "hist",
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eta": 0.1,
}

xgbmodel = xgboost.train(
    params,
    dtrain=xgbtrain,
    num_boost_round=150,
    evals=[(xgbtest, "test")],
    verbose_eval=False,
)



## === cell 23
pred = xgbmodel.predict(xgbfinaltest)
pred = np.clip(pred, 0.0, 300.0)

if len(pred) != len(testkey):
    raise RuntimeError(
        f"Prediction length {len(pred)} does not match test keys {len(testkey)}. "
        "Do not drop rows from test."
    )

finalset = pd.DataFrame({"key": testkey.values, "fare_amount": pred})
finalset = finalset[["key", "fare_amount"]]
finalset.to_csv("finaloutput.csv", index=False)
print("Wrote submission:", "finaloutput.csv", "rows:", len(finalset))
print(finalset.head())
