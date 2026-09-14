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

3.12

# 3. Installed packages

geopandas==0.14.4
joblib==1.5.2
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

3.19317

# 6. Current score

4.97122

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 8.36979) has done: 'I fix the runtime error in `haversine_vec` by allowing scalar reference coordinates (the location lat/lon floats) to be broadcast against the Series inputs, which unblock feature creation and eliminate the downstream KeyErrors. Then I keep the same feature engineering and model choices, but make the pipeline robust by ensuring required feature columns exist in both train and test after transformations, and by importing `mean_squared_error` before it’s used in the XGBoost CV cell. Finally, I ensure the script always writes a valid `submission.csv` with columns `key,fare_amount` in the working directory so Kaggle can score it.'
- What this solution (achieved 6.79123) has done: 'Your current gap to target is large (8.37 vs 3.19, lower is better), and the main issue is that the model is being trained on raw coordinates without the standard NYC Taxi fixes that dramatically reduce RMSE: removing obvious outliers (including zero-distance rides and unrealistically large distances) and adding a small but crucial feature (the Haversine distance to NYC center as a “trip direction/airport” proxy is already present, but we also need to clamp passenger_count/test-time missing datetime rows consistently and ensure identical filtering-derived feature distributions). I keep the same feature set and XGBRegressor training approach, but tighten data cleaning with a few additional, competition-standard filters that don’t change the core logic (still numeric tabular regression on engineered distances). I also train the final XGB model on the full cleaned training data (instead of only the 80% split) before predicting test, which keeps the same model/loop but improves generalization. Finally, I make sure the submission aligns keys 1:1 with the test file and always writes `submission.csv`.'
- What this solution (achieved 5.57904) has done: 'You’re currently far from the target (6.79 vs 3.19 RMSE; lower is better), so the smallest safe way to move toward the target is to fix systematic label noise/outliers that inflate RMSE without changing the modeling approach. I keep your exact feature set and XGBRegressor usage, but add a few standard NYC Taxi cleaning filters (invalid coordinates, extreme fares, and long-distance rides) that are consistent with the task and reduce error. I also apply the same “basic validity” filtering to the test set (without dropping rows) by imputing/cleaning values so feature distributions match training more closely. Finally, I keep the same submission format but ensure test features are finite and aligned.'
- What this solution (achieved 5.96297) has done: 'Your current RMSE (5.579) is well above the target (3.193), so the smallest safe move is to reduce label noise and obvious data errors that XGBoost can’t “learn around.” I keep your exact feature set and XGBRegressor training approach, but tighten the training-only cleaning with a few competition-standard filters that remove (a) obviously wrong coordinates (0,0 and extreme NYC-bbox edges), (b) extreme/unrealistic fares relative to distance, and (c) very short trips with non-minimum fares. I also ensure the same numeric coercions/finite-value handling are applied to *all* model features in both train and test right before fitting/predicting (no new features, just robustness). This should materially lower RMSE while preserving your core logic and producing the same `submission.csv` format.'
- What this solution (achieved 4.93119) has done: 'We keep your exact feature set and XGBRegressor approach, but fix a key mismatch between train-time and test-time preprocessing: you currently clean/filter training rows heavily while leaving the test coordinates/passenger_count largely unbounded, which creates distribution shift and hurts RMSE. The smallest safe improvement is to apply the same coordinate bounding (NYC bbox + valid lat/lon ranges) and passenger_count clamping/imputation to `test_df` *without dropping rows*, so all engineered distance features are computed from realistic inputs. We also enforce the same numeric/finite handling right after feature creation for both train and test, and keep the final “fit on all cleaned train then predict test” logic unchanged. These changes should move RMSE down toward the 3.19 target without altering the model architecture or adding new features.'
- What this solution (achieved 4.96598) has done: 'Your current RMSE (4.93) is well above the target (3.19, lower is better), so the smallest safe move is to reduce systematic label noise/outliers that inflate error without changing your feature set or XGBRegressor approach. I keep the same features and training flow, but add a few competition-standard, strictly training-only filters: (1) remove negative/near-zero fares, (2) remove extreme fares relative to distance (a tighter, more realistic fare-per-km band), and (3) remove very long rides and extreme coordinate deltas that usually indicate bad GPS. I also make test-time preprocessing match train-time coordinate cleaning more closely (without dropping test rows) and clip final predictions to the legal minimum fare (2.5) rather than 0.0, which typically improves RMSE on this competition. These are minimal, targeted changes that preserve your core logic while moving the score toward the target band.'
- What this solution (achieved 4.99531) has done: 'Your RMSE (4.96598) is still far above the target (3.19317), so we should improve it (lower is better) with the smallest changes that don’t alter your model/feature logic. The biggest remaining systematic issue is that the training set still contains many mislabeled/outlier rides that pass your current filters; tightening a couple of *training-only* outlier rules using distance–fare consistency typically drops RMSE materially on this competition. I keep your exact features and XGBRegressor setup, but (1) add a minimal “distance-aware fare band” filter (different bounds for short vs long trips), and (2) remove implausible “very long distance with too-low fare” cases. Submission generation stays identical and still writes `submission.csv`.'
- What this solution (achieved 4.95479) has done: 'Your current RMSE (4.995) is still far above the target (3.193, lower is better), so we should cautiously improve generalization without changing the model type, feature set, or training flow. The biggest remaining issue is that we never remove “impossible” label noise that survives your existing filters (e.g., fares inconsistent with passenger_count and the NYC meter minimum rules), which inflates RMSE. I add two very small, training-only consistency filters (minimum fare by passenger_count, and a modest upper bound on fare for very short trips) and I also clip *test-time* engineered distance features to the same plausible ranges used in training (without dropping any rows) to reduce distribution shift. Everything else (features, XGBRegressor, objective, submission writing) stays the same.'
- What this solution (achieved 4.98512) has done: 'We keep your exact feature set and XGBRegressor training flow, but tighten one remaining source of label noise that commonly keeps RMSE around ~5: airport/long-distance outliers where the route is far from NYC (yet still inside your wide bbox). The minimal change is to add a *training-only* “both endpoints must be reasonably close to NYC center” filter using your already-computed `pickup_dist_to_ny_center` / `dropoff_dist_to_ny_center`, and to apply the same constraint to test by clipping (imputing) extreme endpoint distances rather than dropping rows. This reduces outlier-driven error without altering model architecture, loss, or adding new features. Submission writing remains identical and still produces `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 4.95936) has done: 'Your current RMSE (4.985) is still far above the target (3.193, lower is better), so we should improve it with the smallest changes that preserve your model and feature logic. The biggest remaining gap is that XGBRegressor is being run with almost all-default tree/regularization settings, which underfits/overfits this feature set; we can keep the same XGBRegressor approach but set a few standard, lightweight parameters (max_depth/min_child_weight/subsample/colsample_bytree/reg_lambda) and slightly increase n_estimators while lowering learning_rate to move RMSE down without changing the training loop or features. We also enforce deterministic, metric-aligned behavior by using `tree_method="hist"` and `eval_metric="rmse"` (semantics unchanged) and keep the final “fit on all cleaned train then predict test” submission flow identical. No new features, no early stopping, no sampling changes, and we still write `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 4.97122) has done: 'We need to move your RMSE down (lower is better) toward 3.193 from 4.959, so the smallest high-impact change without altering the model/feature core is to prevent the model from learning on remaining mislabeled/outlier fares. I keep the same features and XGBRegressor training flow, but add one training-only “fare vs distance” sanity filter using your already-computed `ride_distance` (a capped, meter-like linear envelope) which removes extreme label noise that survives fare_per_km rules. I also align datetime handling by avoiding median-imputation for categorical time parts in test (use mode instead) to reduce distribution shift without adding new features. Submission writing stays identical (`submission.csv` with `key,fare_amount`).'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

TRAIN_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
TEST_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"

NROWS = 6_000_000

df = pd.read_csv(TRAIN_PATH, nrows=NROWS)
test_df = pd.read_csv(TEST_PATH)

df.dtypes



## === cell 1
df.describe()



## === cell 2
df.head()




## === cell 3
def filter_column(df_, column, range_min, range_max):
    return df_[(df_[column] >= range_min) & (df_[column] <= range_max)]


ny_latitude_min, ny_latitude_max = 40.4772, 45.0153
ny_longitude_min, ny_longitude_max = -79.7624, -71.7517

df = df.dropna()

for c in ["pickup_latitude", "dropoff_latitude"]:
    df = filter_column(df, c, -90.0, 90.0)
for c in ["pickup_longitude", "dropoff_longitude"]:
    df = filter_column(df, c, -180.0, 180.0)

df = filter_column(df, "pickup_longitude", ny_longitude_min, ny_longitude_max)
df = filter_column(df, "pickup_latitude", ny_latitude_min, ny_latitude_max)
df = filter_column(df, "dropoff_longitude", ny_longitude_min, ny_longitude_max)
df = filter_column(df, "dropoff_latitude", ny_latitude_min, ny_latitude_max)

df = filter_column(df, "passenger_count", 1, 6)

df = filter_column(df, "fare_amount", 2.5, 200.0)
df = df[df["fare_amount"] < 150.0].copy()



## === cell 4
df[df["fare_amount"] > 200].describe()



## === cell 5
df = df




## === cell 6
def refactor_datetime(df_):
    dt = pd.to_datetime(df_["pickup_datetime"], errors="coerce", utc=True)
    df_["year"] = dt.dt.year
    df_["month"] = dt.dt.month
    df_["day"] = dt.dt.day
    df_["weekday"] = dt.dt.weekday
    df_["hour"] = dt.dt.hour
    df_.drop(columns=["pickup_datetime"], inplace=True)


refactor_datetime(df)
refactor_datetime(test_df)

df = df.dropna(subset=["year", "month", "day", "weekday", "hour"])

for c in ["year", "month", "day", "weekday", "hour"]:
    mode_val = test_df[c].mode(dropna=True)
    if len(mode_val) == 0 or not np.isfinite(mode_val.iloc[0]):
        fallback = (
            int(pd.to_datetime("2015-01-01", utc=True).year) if c == "year" else 1
        )
        test_df[c] = test_df[c].fillna(fallback).astype(int)
    else:
        test_df[c] = test_df[c].fillna(int(mode_val.iloc[0])).astype(int)

df.head()




## === cell 7
def haversine_vec(lat1, lon1, lat2, lon2):
    lat1 = np.radians(np.asarray(lat1, dtype="float64"))
    lon1 = np.radians(np.asarray(lon1, dtype="float64"))
    lat2 = np.radians(np.asarray(lat2, dtype="float64"))
    lon2 = np.radians(np.asarray(lon2, dtype="float64"))

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(a))
    return 6367.0 * c  # km


ny_center = ("ny_center", (40.7128, -74.0060))
jfk_airport = ("jfk_airport", (40.6446, -73.7797))
lga_airport = ("lga_airport", (40.7733, -73.8718))
ewr_airport = ("ewr_airport", (40.6895, -74.1745))

locs = [ny_center, jfk_airport, lga_airport, ewr_airport]




## === cell 8
def insert_haversine_dists(df_, locations):
    for name, (lat0, lon0) in locations:
        df_["pickup_dist_to_" + name] = haversine_vec(
            df_["pickup_latitude"], df_["pickup_longitude"], lat0, lon0
        )
        df_["dropoff_dist_to_" + name] = haversine_vec(
            df_["dropoff_latitude"], df_["dropoff_longitude"], lat0, lon0
        )

    df_["ride_distance"] = haversine_vec(
        df_["pickup_latitude"],
        df_["pickup_longitude"],
        df_["dropoff_latitude"],
        df_["dropoff_longitude"],
    )

    df_["abs_lat_diff"] = (df_["pickup_latitude"] - df_["dropoff_latitude"]).abs()
    df_["abs_lon_diff"] = (df_["pickup_longitude"] - df_["dropoff_longitude"]).abs()
    df_["manhattan_approx"] = df_["abs_lat_diff"] + df_["abs_lon_diff"]


def clean_test_inplace(test_):
    for c in [
        "pickup_latitude",
        "dropoff_latitude",
        "pickup_longitude",
        "dropoff_longitude",
        "passenger_count",
    ]:
        test_[c] = pd.to_numeric(test_[c], errors="coerce")

    pc_med = test_["passenger_count"].median()
    if not np.isfinite(pc_med):
        pc_med = 1.0
    test_["passenger_count"] = test_["passenger_count"].fillna(pc_med)
    test_["passenger_count"] = test_["passenger_count"].clip(1, 6)

    lat_med_pick = test_["pickup_latitude"].median()
    lon_med_pick = test_["pickup_longitude"].median()
    lat_med_drop = test_["dropoff_latitude"].median()
    lon_med_drop = test_["dropoff_longitude"].median()
    if not np.isfinite(lat_med_pick):
        lat_med_pick = 40.7128
    if not np.isfinite(lon_med_pick):
        lon_med_pick = -74.0060
    if not np.isfinite(lat_med_drop):
        lat_med_drop = 40.7128
    if not np.isfinite(lon_med_drop):
        lon_med_drop = -74.0060

    for c, lo, hi, med in [
        ("pickup_latitude", -90.0, 90.0, lat_med_pick),
        ("dropoff_latitude", -90.0, 90.0, lat_med_drop),
        ("pickup_longitude", -180.0, 180.0, lon_med_pick),
        ("dropoff_longitude", -180.0, 180.0, lon_med_drop),
    ]:
        bad = (~test_[c].between(lo, hi)) | (~np.isfinite(test_[c]))
        if bad.any():
            test_.loc[bad, c] = med

    for c, lo, hi, med in [
        ("pickup_longitude", ny_longitude_min, ny_longitude_max, lon_med_pick),
        ("pickup_latitude", ny_latitude_min, ny_latitude_max, lat_med_pick),
        ("dropoff_longitude", ny_longitude_min, ny_longitude_max, lon_med_drop),
        ("dropoff_latitude", ny_latitude_min, ny_latitude_max, lat_med_drop),
    ]:
        bad = (~test_[c].between(lo, hi)) | (~np.isfinite(test_[c]))
        if bad.any():
            test_.loc[bad, c] = med

    for c, med in [
        ("pickup_latitude", lat_med_pick),
        ("dropoff_latitude", lat_med_drop),
        ("pickup_longitude", lon_med_pick),
        ("dropoff_longitude", lon_med_drop),
    ]:
        bad = test_[c].abs() <= 0.001
        if bad.any():
            test_.loc[bad, c] = med

    return test_


clean_test_inplace(test_df)

insert_haversine_dists(df, locs)
insert_haversine_dists(test_df, locs)

for c in ["pickup_latitude", "dropoff_latitude"]:
    df = df[df[c].abs() > 0.001]
for c in ["pickup_longitude", "dropoff_longitude"]:
    df = df[df[c].abs() > 0.001]

df = df[df["ride_distance"] > 0.2].copy()
df = df[df["ride_distance"] < 45.0].copy()  # was 60.0

df = df[df["manhattan_approx"] < 2.5].copy()  # was 3.0

df["_fare_per_km"] = df["fare_amount"] / (df["ride_distance"] + 1e-3)
df = df[(df["_fare_per_km"] > 1.0) & (df["_fare_per_km"] < 60.0)].copy()
df.drop(columns=["_fare_per_km"], inplace=True)

df = df[~((df["ride_distance"] < 1.0) & (df["fare_amount"] > 50.0))].copy()

d = df["ride_distance"].astype("float64")
fare = df["fare_amount"].astype("float64")
fare_per_km = fare / (d + 1e-3)

short = d < 2.0
medium = (d >= 2.0) & (d < 10.0)
long = d >= 10.0

df = df[
    (short & (fare_per_km.between(2.0, 120.0)))
    | (medium & (fare_per_km.between(1.2, 70.0)))
    | (long & (fare_per_km.between(1.0, 40.0)))
].copy()

df = df[~((df["ride_distance"] > 20.0) & (df["fare_amount"] < 20.0))].copy()

min_fare_by_pc = {1: 2.5, 2: 3.0, 3: 3.5, 4: 4.0, 5: 4.5, 6: 5.0}
pc_int = df["passenger_count"].round().astype(int).clip(1, 6)
df = df[df["fare_amount"] >= pc_int.map(min_fare_by_pc).astype("float64")].copy()

df = df[~((df["ride_distance"] < 0.5) & (df["fare_amount"] > 30.0))].copy()

CENTER_MAX_KM = 60.0
df = df[
    (df["pickup_dist_to_ny_center"].between(0.0, CENTER_MAX_KM))
    & (df["dropoff_dist_to_ny_center"].between(0.0, CENTER_MAX_KM))
].copy()

for col in ["pickup_dist_to_ny_center", "dropoff_dist_to_ny_center"]:
    test_df[col] = pd.to_numeric(test_df[col], errors="coerce")
    med = test_df[col].median()
    if not np.isfinite(med):
        med = 0.0
    test_df[col] = test_df[col].fillna(med).clip(0.0, CENTER_MAX_KM)

test_df["ride_distance"] = pd.to_numeric(
    test_df["ride_distance"], errors="coerce"
).fillna(test_df["ride_distance"].median())
test_df["ride_distance"] = test_df["ride_distance"].clip(0.2, 45.0)
test_df["manhattan_approx"] = pd.to_numeric(
    test_df["manhattan_approx"], errors="coerce"
).fillna(test_df["manhattan_approx"].median())
test_df["manhattan_approx"] = test_df["manhattan_approx"].clip(0.0, 2.5)

for col in (
    ["ride_distance", "abs_lat_diff", "abs_lon_diff", "manhattan_approx"]
    + ["pickup_dist_to_" + x[0] for x in locs]
    + ["dropoff_dist_to_" + x[0] for x in locs]
):
    test_df[col] = pd.to_numeric(test_df[col], errors="coerce")
    if test_df[col].isna().any():
        test_df[col] = test_df[col].fillna(test_df[col].median())
    test_df[col] = test_df[col].replace([np.inf, -np.inf], test_df[col].median())

d = df["ride_distance"].astype("float64")
fare = df["fare_amount"].astype("float64")

lower = 2.5 + 0.8 * d
upper = 7.0 + 18.0 * d + 10.0
df = df[(fare >= lower) & (fare <= upper)].copy()

df.describe()



## === cell 9
df.describe()



## === cell 10
from sklearn.model_selection import train_test_split

train_df, validation_df = train_test_split(df, test_size=0.2, random_state=42)



## === cell 11
features = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "year",
    "month",
    "day",
    "hour",
    "weekday",
    "ride_distance",
    "abs_lat_diff",
    "abs_lon_diff",
    "manhattan_approx",
]
features += ["pickup_dist_to_" + x[0] for x in locs]
features += ["dropoff_dist_to_" + x[0] for x in locs]

fare_amount = "fare_amount"

missing_train = [c for c in features if c not in train_df.columns]
missing_valid = [c for c in features if c not in validation_df.columns]
missing_test = [c for c in features if c not in test_df.columns]
if missing_train or missing_valid or missing_test:
    raise KeyError(
        f"Missing columns - train: {missing_train}, valid: {missing_valid}, test: {missing_test}"
    )


def ensure_numeric_finite(df_, cols):
    for c in cols:
        df_[c] = pd.to_numeric(df_[c], errors="coerce")
        if df_[c].isna().any():
            df_[c] = df_[c].fillna(df_[c].median())
        df_[c] = df_[c].replace([np.inf, -np.inf], df_[c].median())
    return df_


train_df = ensure_numeric_finite(train_df, features + [fare_amount])
validation_df = ensure_numeric_finite(validation_df, features + [fare_amount])
test_df = ensure_numeric_finite(test_df, features)

train_features = train_df[features]
train_fare_amount = train_df[fare_amount]

validation_features = validation_df[features]
validation_fare_amount = validation_df[fare_amount]

train_features.info()



## === cell 12
from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()



## === cell 13
from sklearn.model_selection import cross_val_score


def estimate_model(model, df_):
    X = df_[features]
    y = df_[fare_amount]
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")
    rmse_scores = np.sqrt(-cv_scores)
    print("RMSE scores for each fold:", rmse_scores)
    print("Mean RMSE:", rmse_scores.mean())
    print("Standard Deviation of RMSE:", rmse_scores.std())




## === cell 14
estimate_model(linear_model, train_df)



## === cell 15
from sklearn.metrics import mean_squared_error

linear_model.fit(train_features, train_fare_amount)

linear_predictions = linear_model.predict(validation_features)
mean_squared_error(validation_fare_amount, linear_predictions, squared=False)



## === cell 16
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import seaborn as sns

learning_rates = [0.1, 0.15, 0.2]
n_estimators = [80, 100, 150]

sample_fraction = 0.08
train_sample = df.sample(frac=sample_fraction, random_state=42)
X = train_sample[features]
y = train_sample[fare_amount]


def cross_val_rmse(lr, ne, X_, y_):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_rmse = []

    for train_index, val_index in kf.split(X_):
        X_train, X_val = X_.iloc[train_index], X_.iloc[val_index]
        y_train, y_val = y_.iloc[train_index], y_.iloc[val_index]

        model = XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
            learning_rate=lr,
            n_estimators=ne,
            max_depth=8,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=42,
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_val)
        rmse = mean_squared_error(y_val, predictions, squared=False)
        fold_rmse.append(rmse)

    avg_rmse = float(np.mean(fold_rmse))
    return lr, ne, avg_rmse


results = Parallel(n_jobs=-1)(
    delayed(cross_val_rmse)(lr, ne, X, y)
    for lr in learning_rates
    for ne in n_estimators
)

results_df = pd.DataFrame(results, columns=["learning_rate", "n_estimators", "rmse"])
results_df.replace([np.inf, -np.inf], np.nan, inplace=True)

plt.figure(figsize=(12, 8))
sns.lineplot(
    data=results_df, x="n_estimators", y="rmse", hue="learning_rate", marker="o"
)
plt.title("RMSE for Different Learning Rates and n_estimators")
plt.xlabel("Number of Estimators")
plt.ylabel("RMSE")
plt.legend(title="Learning Rate")
plt.show()



## === cell 17
from xgboost import XGBRegressor

xgb_model = XGBRegressor(
    objective="reg:squarederror",
    eval_metric="rmse",
    tree_method="hist",
    learning_rate=0.08,
    n_estimators=400,
    max_depth=8,
    min_child_weight=2,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    n_jobs=-1,
    random_state=42,
)

xgb_model.fit(train_features, train_fare_amount)

xgb_predictions = xgb_model.predict(validation_features)
mean_squared_error(validation_fare_amount, xgb_predictions, squared=False)



## === cell 18
xgb_predictions = xgb_model.predict(train_features)
mean_squared_error(train_fare_amount, xgb_predictions, squared=False)



## === cell 19
final_model = XGBRegressor(
    objective="reg:squarederror",
    eval_metric="rmse",
    tree_method="hist",
    learning_rate=0.08,
    n_estimators=400,
    max_depth=8,
    min_child_weight=2,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    n_jobs=-1,
    random_state=42,
)
final_model.fit(df[features], df[fare_amount])

test_predictions = final_model.predict(test_df[features])

test_predictions = np.clip(test_predictions, 2.5, None)

submission = pd.DataFrame(
    {"key": test_df["key"].values, "fare_amount": test_predictions}
)
submission.to_csv("submission.csv", index=False)

submission.head()
