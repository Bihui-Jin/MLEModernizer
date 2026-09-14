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

3.44252

# 6. Current score

5.25093

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 10.34255) has done: 'I fix the runtime error in the Haversine feature engineering by making the distance function accept scalar reference coordinates (the airport/center lat/lon), which currently crashes because it calls `.astype()` on Python floats. Once those features are created, the downstream KeyErrors for `ride_distance` and the engineered distance columns disappear, allowing the train/validation split, model training, and test prediction to run end-to-end. I also ensure the test set is not filtered in a way that drops rows (to preserve exact submission keys/row count), while keeping the same feature logic by computing distances for all test rows and only filtering train rows. Finally, I ensure a valid `submission.csv` with `key,fare_amount` is always written.'
- What this solution (achieved 10.34255) has done: 'You’re far from the target (10.34 vs 3.44 RMSE; lower is better), so we should legitimately improve generalization without changing your overall pipeline. The biggest score drag in your current setup is inconsistent preprocessing: you filter obvious outliers only in train, but you never apply equivalent coordinate/passenger sanity handling to test (and you only drop ride_distance==0 in train), which can produce wildly wrong predictions for a small number of problematic test rows and balloon RMSE. I keep the same features and XGBoost model, but (1) add a minimal, standard NYC bounding-box + passenger_count clipping preprocessing for test (without dropping rows) and (2) clip ride_distance to a small epsilon in both train/test so the model never sees zeros/negatives and never predicts pathological values for those cases. This is a minimal change that typically reduces tail errors and should move RMSE substantially toward your target without altering the core modeling approach.'
- What this solution (achieved 7.88188) has done: 'Your RMSE is far worse than target, so the smallest legitimate improvements are to (1) remove clearly wrong geographic points using a tighter NYC bounding box (still only filtering train, clipping test to preserve row count) and (2) add one minimal, standard feature that preserves your core logic (bearing) to help the same XGBoost regressor fit route directionality. I also make sure train/test preprocessing is consistent for passenger_count and ride_distance (no zeros), because a few pathological rows can dominate RMSE. Finally, I keep your same model class and training flow, but fit the final XGBoost on the full filtered training data (instead of only train split) before predicting test, which typically improves leaderboard RMSE without changing architecture or loss.'
- What this solution (achieved 9.5798) has done: 'Your current RMSE (7.88) is far above the target (3.44, lower is better), so the smallest legitimate improvement is to make preprocessing more consistent and reduce systematic error without changing your model/training loop. I keep the same XGBRegressor setup and feature set, but (1) add two standard, minimal time/geo features that usually give a large RMSE drop for this competition (`abs_lon_diff`, `abs_lat_diff` and `log1p_ride_distance`) while preserving the same overall feature-engineering approach, and (2) apply the same ride-distance sanity handling to both train and test (clip to epsilon in both, instead of dropping all zero-distance rows only in train). Finally, I clip coordinates and passenger_count in both train/test in a consistent way (filtering train, clipping test to preserve row count) to avoid a few pathological rows blowing up error.'
- What this solution (achieved 5.32469) has done: 'Your RMSE (9.58, lower is better) is still far from the target (3.44), so we need a small but meaningful generalization improvement without changing your overall approach. The biggest remaining drag is that the model is trained on raw fares; this competition benefits a lot from the standard trick of training in log-space (predicting `log1p(fare_amount)`), then converting back with `expm1`—same model class and loss type, just a stable target transform that reduces the impact of heavy-tailed outliers. I keep your exact feature set and XGBRegressor setup, but switch only the target used in `.fit()`/`.predict()` and ensure the same transform is used consistently for validation RMSE and test predictions. This should substantially lower RMSE toward your target while remaining within minimal, legitimate changes and still producing a valid `submission.csv`.'
- What this solution (achieved 6.0688) has done: 'Your current RMSE (5.32469; lower is better) is still far above the target (3.44252), so we should make a small, legitimate improvement that usually yields a sizable RMSE drop in this competition without changing your overall model/training flow. The biggest low-risk gain is to add a standard geodesic feature set that complements your existing haversine distances: `manhattan_distance` (lat/long “city-block” distance), plus `pickup/dropoff` coordinates scaled by `cos(latitude)` and the corresponding diffs; these are simple deterministic transforms of existing inputs and keep the same XGBRegressor and training procedure. I also add minimal time features (`minute` and a binary `is_weekend`) derived from the same `pickup_datetime`, which typically improves fare modeling and is consistent with your current datetime refactor. All changes are applied identically to train and test, and the script still writes a valid `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 6.0688) has done: 'You’re still far above the target RMSE (6.07 vs 3.44, lower is better), so the smallest likely gain is to fix a key inconsistency: you clip *test* coordinates to the NYC box but you never apply the same clipping to *train*, which makes the model learn on raw coords while predicting on clipped coords (distribution shift). I keep your exact feature set, target log-transform, XGBRegressor, and training flow, and only add symmetric coordinate clipping for the already-filtered training data (no extra row drops). I also ensure any infinities/NaNs created during feature engineering are removed from train (not test) to avoid training on invalid rows. These changes are minimal, deterministic, and typically reduce RMSE by removing train/test preprocessing mismatch.'
- What this solution (achieved 5.28322) has done: 'Your RMSE (6.0688, lower is better) is still far above the target (3.44252), so we should make a small, high-impact improvement without changing the core modeling approach (same XGBRegressor, same training flow, same loss/metric). The biggest remaining miss in your feature set is that fare depends strongly on absolute pickup location (borough/airport trips) and not just relative geometry—adding the standard minimal “distance to key NYC landmarks” (JFK/LGA/EWR/TimesSq) as numeric features is a low-risk way to reduce systematic error. I also add a tiny, deterministic cleanup to drop obviously invalid engineered values in train only (not test) so training isn’t polluted, while preserving the same submission row order/keys. Everything else (sampling, model class, log1p target transform, clipping, file paths, and submission format) stays the same.'
- What this solution (achieved 5.19867) has done: 'Your current RMSE (5.28322) is still well above the target (3.44252, lower is better), so we need a small, legitimate improvement without changing your overall pipeline. The biggest low-risk gain here is to align the *training objective with your log1p target*: keep predicting `log1p(fare)` but switch XGBoost’s objective to `reg:squaredlogerror`, which is designed for log-space errors and typically reduces RMSE for this competition with minimal code change. I also add a minimal prediction post-processing step to avoid pathological expm1 outputs (replace non-finite, clip to a reasonable range) which reduces tail error without altering the model logic. Everything else (same features, same train/test preprocessing, same training flow, same submission schema/path) stays the same.'
- What this solution (achieved 5.28322) has done: 'Your current RMSE (5.19867) is still far above the target (3.44252, lower is better), so we should make a small improvement that reduces systematic error without changing your overall pipeline. The biggest issue left is an objective/target mismatch: you train on `log1p(fare)` but use XGBoost’s `reg:squaredlogerror`, which expects raw positive labels and internally applies a log—this double-logs the target and hurts performance. I switch the objective back to plain squared error (`reg:squarederror`) while keeping the log1p target transform and the exact same features/training flow. I also make the CV block consistent with this target transform so the tuning plot reflects the real training setup (this doesn’t change the final model logic, just aligns evaluation).'
- What this solution (achieved 5.13753) has done: 'Your current RMSE (5.28322) is still far above the target (3.44252), so we should make a small, legitimate improvement that typically lowers RMSE without changing your overall pipeline. The biggest low-risk gain is to add the standard NYC Taxi Fare “cyclical time” encoding (sin/cos for hour-of-day and day-of-week) using your existing datetime-derived columns; this helps the same XGBRegressor model capture periodic demand/traffic effects with minimal code change. I also add one minimal, deterministic interaction feature (`distance_per_passenger`) which often improves fit without changing the model/training logic. Everything else (data loading, filtering, log1p target transform, XGBRegressor, train/valid split, and submission writing) stays the same.'
- What this solution (achieved 5.25093) has done: 'Your current RMSE (5.13753; lower is better) is still far above the target (3.44252), so we should make a small, safe improvement that reduces systematic error without changing your overall approach (same features + XGBRegressor + log1p target). The main low-risk gain here is to prevent a train/test distribution mismatch: you currently clip coordinates in both sets, but you *only* filter train rows by the NYC box; instead, we keep your filtering but also clip passenger_count in train (not just test) and remove obviously impossible rides by filtering extreme engineered distances in train only. Additionally, we add a minimal “airport_trip” indicator derived from your existing landmark distances (no new data, no new model type), which typically helps separate airport fares and should move RMSE closer to your target. Everything else (paths, training flow, objective, post-processing, submission format) stays the same and the script still writes `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

TRAIN_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
TEST_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"

df = pd.read_csv(TRAIN_PATH, nrows=4_000_000)
test_df = pd.read_csv(TEST_PATH)

df.dtypes



## === cell 1
df.describe()



## === cell 2
df.head()




## === cell 3
def filter_column(df_, column, range_min, range_max):
    return df_[(df_[column] >= range_min) & (df_[column] <= range_max)]


ny_latitude_min, ny_latitude_max = 40.5, 41.0
ny_longitude_min, ny_longitude_max = -74.3, -73.7

df = df.dropna()

df = filter_column(df, "pickup_longitude", ny_longitude_min, ny_longitude_max)
df = filter_column(df, "pickup_latitude", ny_latitude_min, ny_latitude_max)
df = filter_column(df, "dropoff_longitude", ny_longitude_min, ny_longitude_max)
df = filter_column(df, "dropoff_latitude", ny_latitude_min, ny_latitude_max)
df = filter_column(df, "passenger_count", 1, 6)



## === cell 4
df[df["fare_amount"] > 200].describe()



## === cell 5
df = filter_column(df, "fare_amount", 1, 200)




## === cell 6
def refactor_datetime(df_):
    dt = pd.to_datetime(df_["pickup_datetime"], errors="coerce", utc=True)
    df_["year"] = dt.dt.year.astype("Int16")
    df_["month"] = dt.dt.month.astype("Int8")
    df_["day"] = dt.dt.day.astype("Int8")
    df_["weekday"] = dt.dt.weekday.astype("Int8")
    df_["hour"] = dt.dt.hour.astype("Int8")
    df_["minute"] = dt.dt.minute.astype("Int8")
    df_["is_weekend"] = (dt.dt.weekday >= 5).astype("Int8")
    df_.drop(columns=["pickup_datetime"], inplace=True)


refactor_datetime(df)
refactor_datetime(test_df)

df = df.dropna(
    subset=["year", "month", "day", "weekday", "hour", "minute", "is_weekend"]
)

for col in ["year", "month", "day", "weekday", "hour", "minute", "is_weekend"]:
    if test_df[col].isna().any():
        fill_val = (
            int(df[col].mode(dropna=True).iloc[0]) if df[col].notna().any() else 0
        )
        test_df[col] = test_df[col].fillna(fill_val).astype(df[col].dtype)

df.head()




## === cell 7
def haversine_np(lat1, lon1, lat2, lon2):
    """
    Bug fix: allow lat2/lon2 to be scalars (floats) or arrays/Series.
    The original version failed because it did `.astype()` on Python floats.
    """
    lat1 = np.radians(np.asarray(lat1, dtype=np.float64))
    lon1 = np.radians(np.asarray(lon1, dtype=np.float64))
    lat2 = np.radians(np.asarray(lat2, dtype=np.float64))
    lon2 = np.radians(np.asarray(lon2, dtype=np.float64))

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6367.0 * c


ny_center = ("ny_center", (40.7128, -74.0060))
times_sq = ("times_sq", (40.7580, -73.9855))
jfk_airport = ("jfk_airport", (40.6446, -73.7797))
lga_airport = ("lga_airport", (40.7733, -73.8718))
ewr_airport = ("ewr_airport", (40.6895, -74.1745))
locs = [ny_center, times_sq, jfk_airport, lga_airport, ewr_airport]




## === cell 8
def bearing_np(lat1, lon1, lat2, lon2):
    """
    Change (score improvement): add bearing (direction) as a minimal extra feature.
    This preserves the same overall feature-engineering style and often reduces RMSE for taxi fares.
    Returns bearing in degrees in [0, 360).
    """
    lat1 = np.radians(np.asarray(lat1, dtype=np.float64))
    lon1 = np.radians(np.asarray(lon1, dtype=np.float64))
    lat2 = np.radians(np.asarray(lat2, dtype=np.float64))
    lon2 = np.radians(np.asarray(lon2, dtype=np.float64))

    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    brng = np.degrees(np.arctan2(y, x))
    return (brng + 360.0) % 360.0


def insert_haversine_dists(df_, locations):
    for name, (lat0, lon0) in locations:
        df_["pickup_dist_to_" + name] = haversine_np(
            df_["pickup_latitude"], df_["pickup_longitude"], lat0, lon0
        )
        df_["dropoff_dist_to_" + name] = haversine_np(
            df_["dropoff_latitude"], df_["dropoff_longitude"], lat0, lon0
        )
    df_["ride_distance"] = haversine_np(
        df_["pickup_latitude"],
        df_["pickup_longitude"],
        df_["dropoff_latitude"],
        df_["dropoff_longitude"],
    )
    df_["bearing"] = bearing_np(
        df_["pickup_latitude"],
        df_["pickup_longitude"],
        df_["dropoff_latitude"],
        df_["dropoff_longitude"],
    )

    df_["abs_lon_diff"] = (df_["dropoff_longitude"] - df_["pickup_longitude"]).abs()
    df_["abs_lat_diff"] = (df_["dropoff_latitude"] - df_["pickup_latitude"]).abs()

    lat1 = df_["pickup_latitude"]
    lon1 = df_["pickup_longitude"]
    lat2 = df_["dropoff_latitude"]
    lon2 = df_["dropoff_longitude"]
    df_["manhattan_distance"] = haversine_np(lat1, lon1, lat1, lon2) + haversine_np(
        lat1, lon1, lat2, lon1
    )

    lat_mean = (lat1.to_numpy(dtype=np.float64) + lat2.to_numpy(dtype=np.float64)) * 0.5
    cos_lat_mean = np.cos(np.radians(lat_mean))
    df_["pickup_longitude_coslat"] = (
        df_["pickup_longitude"].to_numpy(dtype=np.float64) * cos_lat_mean
    )
    df_["dropoff_longitude_coslat"] = (
        df_["dropoff_longitude"].to_numpy(dtype=np.float64) * cos_lat_mean
    )
    df_["lon_diff_coslat"] = (
        df_["dropoff_longitude_coslat"] - df_["pickup_longitude_coslat"]
    ).astype(np.float64)
    df_["lat_diff"] = (df_["dropoff_latitude"] - df_["pickup_latitude"]).astype(
        np.float64
    )


insert_haversine_dists(df, locs)
insert_haversine_dists(test_df, locs)



## === cell 9
df.describe()



## === cell 10
for col, lo, hi in [
    ("pickup_longitude", ny_longitude_min, ny_longitude_max),
    ("dropoff_longitude", ny_longitude_min, ny_longitude_max),
    ("pickup_latitude", ny_latitude_min, ny_latitude_max),
    ("dropoff_latitude", ny_latitude_min, ny_latitude_max),
]:
    df[col] = df[col].clip(lo, hi)
    test_df[col] = test_df[col].clip(lo, hi)

df["passenger_count"] = df["passenger_count"].clip(1, 6)
test_df["passenger_count"] = test_df["passenger_count"].clip(1, 6)

eps_km = 1e-6
df["ride_distance"] = df["ride_distance"].clip(lower=eps_km)
test_df["ride_distance"] = test_df["ride_distance"].clip(lower=eps_km)

df["log1p_ride_distance"] = np.log1p(df["ride_distance"])
test_df["log1p_ride_distance"] = np.log1p(test_df["ride_distance"])

df["manhattan_distance"] = df["manhattan_distance"].clip(lower=eps_km)
test_df["manhattan_distance"] = test_df["manhattan_distance"].clip(lower=eps_km)
df["log1p_manhattan_distance"] = np.log1p(df["manhattan_distance"])
test_df["log1p_manhattan_distance"] = np.log1p(test_df["manhattan_distance"])

df["hour_sin"] = np.sin(2.0 * np.pi * df["hour"].to_numpy(dtype=np.float64) / 24.0)
df["hour_cos"] = np.cos(2.0 * np.pi * df["hour"].to_numpy(dtype=np.float64) / 24.0)
test_df["hour_sin"] = np.sin(
    2.0 * np.pi * test_df["hour"].to_numpy(dtype=np.float64) / 24.0
)
test_df["hour_cos"] = np.cos(
    2.0 * np.pi * test_df["hour"].to_numpy(dtype=np.float64) / 24.0
)

df["weekday_sin"] = np.sin(2.0 * np.pi * df["weekday"].to_numpy(dtype=np.float64) / 7.0)
df["weekday_cos"] = np.cos(2.0 * np.pi * df["weekday"].to_numpy(dtype=np.float64) / 7.0)
test_df["weekday_sin"] = np.sin(
    2.0 * np.pi * test_df["weekday"].to_numpy(dtype=np.float64) / 7.0
)
test_df["weekday_cos"] = np.cos(
    2.0 * np.pi * test_df["weekday"].to_numpy(dtype=np.float64) / 7.0
)

df["distance_per_passenger"] = df["ride_distance"] / df["passenger_count"].clip(
    lower=1
).to_numpy(dtype=np.float64)
test_df["distance_per_passenger"] = test_df["ride_distance"] / test_df[
    "passenger_count"
].clip(lower=1).to_numpy(dtype=np.float64)

airport_cols_pickup = [
    "pickup_dist_to_jfk_airport",
    "pickup_dist_to_lga_airport",
    "pickup_dist_to_ewr_airport",
]
airport_cols_dropoff = [
    "dropoff_dist_to_jfk_airport",
    "dropoff_dist_to_lga_airport",
    "dropoff_dist_to_ewr_airport",
]
df["airport_trip"] = (
    (df[airport_cols_pickup].min(axis=1) < 1.5)
    | (df[airport_cols_dropoff].min(axis=1) < 1.5)
).astype("Int8")
test_df["airport_trip"] = (
    (test_df[airport_cols_pickup].min(axis=1) < 1.5)
    | (test_df[airport_cols_dropoff].min(axis=1) < 1.5)
).astype("Int8")

numeric_cols = df.select_dtypes(include=[np.number]).columns
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=list(numeric_cols))

df = df[(df["ride_distance"] > 0.0) & (df["ride_distance"] < 100.0)]
df = df[(df["manhattan_distance"] > 0.0) & (df["manhattan_distance"] < 150.0)]

df.describe()



## === cell 11
from sklearn.model_selection import train_test_split

train_df, validation_df = train_test_split(df, test_size=0.2, random_state=42)



## === cell 12
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
    "minute",
    "is_weekend",
    "ride_distance",
    "log1p_ride_distance",
    "manhattan_distance",
    "log1p_manhattan_distance",
    "bearing",
    "abs_lon_diff",
    "abs_lat_diff",
    "pickup_longitude_coslat",
    "dropoff_longitude_coslat",
    "lon_diff_coslat",
    "lat_diff",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "distance_per_passenger",
    "airport_trip",
]
features += ["pickup_dist_to_" + x[0] for x in locs]
features += ["dropoff_dist_to_" + x[0] for x in locs]
fare_amount = "fare_amount"

missing_train = [c for c in features + [fare_amount] if c not in train_df.columns]
missing_test = [c for c in features if c not in test_df.columns]
if missing_train:
    raise KeyError(
        f"Missing columns in train_df after feature engineering: {missing_train}"
    )
if missing_test:
    raise KeyError(
        f"Missing columns in test_df after feature engineering: {missing_test}"
    )

train_features = train_df[features]
train_fare_amount = train_df[fare_amount]

validation_features = validation_df[features]
validation_fare_amount = validation_df[fare_amount]

train_features.info()



## === cell 13
from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()



## === cell 14
from sklearn.model_selection import cross_val_score


def estimate_model(model, df_):
    X = df_[features]
    y = df_[fare_amount]
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")
    rmse_scores = np.sqrt(-cv_scores)
    print("RMSE scores for each fold:", rmse_scores)
    print("Mean RMSE:", rmse_scores.mean())
    print("Standard Deviation of RMSE:", rmse_scores.std())




## === cell 15
estimate_model(linear_model, train_df)



## === cell 16
linear_model.fit(train_features, train_fare_amount)



## === cell 17
from sklearn.metrics import mean_squared_error

linear_predictions = linear_model.predict(validation_features)
mean_squared_error(validation_fare_amount, linear_predictions, squared=False)



## === cell 18
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import seaborn as sns
from sklearn.metrics import mean_squared_error

learning_rates = [0.1, 0.15, 0.2]
n_estimators = [80, 100, 150]

sample_fraction = 0.1
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
            learning_rate=lr,
            n_estimators=ne,
            n_jobs=-1,
            random_state=42,
        )
        model.fit(X_train, np.log1p(y_train.to_numpy(dtype=np.float64)))
        pred_log = model.predict(X_val)
        pred_log = np.where(np.isfinite(pred_log), pred_log, 0.0)
        pred = np.expm1(pred_log)
        pred = np.where(np.isfinite(pred), pred, 0.0)
        pred = np.clip(pred, 0.0, 500.0)
        rmse = mean_squared_error(y_val, pred, squared=False)
        fold_rmse.append(rmse)
    return lr, ne, float(np.mean(fold_rmse))


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



## === cell 19
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

xgb_model = XGBRegressor(
    objective="reg:squarederror",
    learning_rate=0.15,
    n_estimators=150,
    max_depth=6,
    min_child_weight=2,
    n_jobs=-1,
    random_state=42,
)

y_train_log = np.log1p(train_fare_amount.to_numpy(dtype=np.float64))
xgb_model.fit(train_features, y_train_log)

val_pred_log = xgb_model.predict(validation_features)

val_pred_log = np.where(np.isfinite(val_pred_log), val_pred_log, 0.0)

val_pred = np.expm1(val_pred_log)
val_pred = np.where(np.isfinite(val_pred), val_pred, 0.0)
val_pred = np.clip(val_pred, 0.0, 500.0)

mean_squared_error(validation_fare_amount, val_pred, squared=False)



## === cell 20
xgb_predictions_train_log = xgb_model.predict(train_features)
xgb_predictions_train_log = np.where(
    np.isfinite(xgb_predictions_train_log), xgb_predictions_train_log, 0.0
)
xgb_predictions_train = np.expm1(xgb_predictions_train_log)
xgb_predictions_train = np.where(
    np.isfinite(xgb_predictions_train), xgb_predictions_train, 0.0
)
xgb_predictions_train = np.clip(xgb_predictions_train, 0.0, 500.0)

mean_squared_error(train_fare_amount, xgb_predictions_train, squared=False)



## === cell 21
from xgboost import XGBRegressor

final_xgb_model = XGBRegressor(
    objective="reg:squarederror",
    learning_rate=0.15,
    n_estimators=150,
    max_depth=6,
    min_child_weight=2,
    n_jobs=-1,
    random_state=42,
)

final_xgb_model.fit(df[features], np.log1p(df[fare_amount].to_numpy(dtype=np.float64)))

test_pred_log = final_xgb_model.predict(test_df[features])

test_pred_log = np.where(np.isfinite(test_pred_log), test_pred_log, 0.0)
test_pred = np.expm1(test_pred_log)
test_pred = np.where(np.isfinite(test_pred), test_pred, 0.0)
test_pred = np.clip(test_pred, 0.0, 500.0)

submission = pd.DataFrame({"key": test_df["key"], "fare_amount": test_pred})
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
