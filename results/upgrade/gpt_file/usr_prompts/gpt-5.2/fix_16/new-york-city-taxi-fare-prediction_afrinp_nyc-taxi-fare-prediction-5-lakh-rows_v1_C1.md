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
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
sklearn-pandas==2.2.0
tf_keras==2.18.0

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

3.45343

# 6. Current score

## error

Could not parse JSON from mlebench output.
Please be reminded that the grader relies on 'mle-bench' being installed and navigated.
Raw output (tail):
Traceback (most recent call last):
  File "/home/b27jin/miniconda3/envs/mle_env/bin/mlebench", line 5, in <module>
    from mlebench.cli import main
  File "/home/b27jin/mle-bench/mlebench/cli.py", line 5, in <module>
    from mlebench.data import download_and_prepare_dataset, ensure_leaderboard_exists
  File "/home/b27jin/mle-bench/mlebench/data.py", line 29, in <module>
    cache = dc.Cache("cache", size_limit=2**26)  # 64 MB
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/b27jin/miniconda3/envs/mle_env/lib/python3.11/site-packages/diskcache/core.py", line 499, in __init__
    sql(query, (key, value))
  File "/home/b27jin/miniconda3/envs/mle_env/lib/python3.11/site-packages/diskcache/core.py", line 666, in _execute_with_retry
    return sql(statement, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
sqlite3.OperationalError: disk I/O error

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 567.82654) has done: 'I fix the runtime crash caused by importing `keras` in this Kaggle environment by switching to the installed `tf_keras` backend while keeping the exact same Sequential/Dense architecture and training loop. I also fix a subtle preprocessing bug: `scale(test)` was being fit on the test set instead of using the training scaling parameters, which harms RMSE; I replace it with a `StandardScaler` fit on training features and applied to both train and test (same semantics as scaling, but correct). Finally, I ensure the submission file has exactly the required columns (`key,fare_amount`) and that predictions are written as a 1D float array.'
- What this solution (achieved 679.75833) has done: 'I fix the crash in the `tf_keras` import that’s caused by an environment protobuf incompatibility by switching to the already-available `tensorflow` Keras backend (same Sequential/Dense model, same training loop). I also keep the corrected scaling (fit scaler on train, apply to both train and test) and ensure the test columns match the train feature columns exactly to avoid silent misalignment. Finally, I keep the submission format strictly as `key,fare_amount` and write `submission.csv` in the working directory so Kaggle accepts it.'
- What this solution (achieved 1363.71157) has done: 'I fix the runtime crash caused by importing `tensorflow` (protobuf incompatibility) by switching the model backend to the already-installed `tf_keras`, keeping the exact same Sequential/Dense architecture, compile settings, and training loop. I also add deterministic seeding and force CPU usage to avoid GPU/TF initialization paths that can trigger the same protobuf issue, without changing the modeling semantics. Finally, I keep the corrected scaling (fit on train, apply to test) and ensure the submission is written as `submission.csv` with exactly `key,fare_amount` and aligned row order.'
- What this solution (achieved 74.36088) has done: 'I fix the crash in the import cell by removing the dependency on `tf_keras` (which is triggering a protobuf `MessageFactory` error in this Kaggle image) and switching to scikit-learn’s `MLPRegressor`, keeping the same “dense MLP on scaled engineered numeric features trained with MSE” core approach. I keep all existing feature engineering and the train-fit scaler / test-transform scaler logic intact, and only change the modeling backend so the notebook runs end-to-end reliably. I also ensure predictions are finite and written to `submission.csv` with exactly `key,fare_amount` in the test row order. This should dramatically reduce RMSE compared to the currently-broken/degraded run and move the score toward the target.'
- What this solution (achieved 14.03156) has done: 'Your current RMSE (74) is far above the target (3.45), so we should improve performance with the smallest changes that don’t alter the overall “scaled engineered numeric features → dense MLP regressor trained with MSE” approach. The biggest issue is that your model is learning on raw latitude/longitude differences and an unscaled Euclidean distance in degrees, which is poorly aligned with true trip distance; adding a proper haversine distance feature (in km) is a minimal feature-engineering fix that usually yields a large RMSE drop on this competition. I also apply the exact same passenger_count cleanup to the test set (you currently only fix it in train), and I clip predictions to a more reasonable upper bound (100, consistent with your training label filter) to reduce extreme outliers that hurt RMSE. Everything else (data size, scaler-fit-on-train, MLPRegressor, training loop) stays the same.'
- What this solution (achieved 14.20676) has done: 'Your current RMSE (14.03) is still far above the 3.45 target, so the smallest change that should materially improve score is to remove extreme/noisy training rows that the model can’t fit well with your current MLP features. I add two standard NYC taxi-fare cleaning steps: (1) filter out near-zero-distance trips with non-trivial fares and (2) filter unrealistic fare-per-km outliers using the haversine distance you already computed. This keeps the exact same feature set, scaler, MLPRegressor architecture, and training loop, but gives the model a cleaner signal that typically reduces RMSE significantly on this competition. I also keep your prediction clipping and submission formatting unchanged.'
- What this solution (achieved 14.20604) has done: 'Your RMSE (14.21) is still far above the target (3.45), so we should improve it with minimal, competition-standard data cleaning and feature alignment while keeping your exact core approach (same engineered numeric features → StandardScaler → MLPRegressor with the same hidden layers and training loop). The biggest remaining issue is that the model is trained on many label/geometry outliers (bad coordinates, airport/outside-NYC points, impossible fares) that dominate RMSE; we add two small, widely-used filters (distance cap and fare-per-km cap) and drop obviously-wrong passenger counts, without changing the model itself. We also add one minimal, high-signal feature (`abs_longitude_diff/abs_latitude_diff`) derived from your existing diffs (not a new modeling approach), which typically helps MLPs fit better without altering semantics. Finally, we keep the same submission writing logic and clipping, ensuring a valid `submission.csv`.'
- What this solution (achieved 14.20604) has done: 'Your current RMSE (14.206) is much worse than the target (3.453), so we should improve score with minimal, standard NYC-taxi cleaning while keeping your exact core pipeline (same engineered numeric features → StandardScaler → same MLPRegressor settings and training loop). The biggest remaining issue is that the model is still trained on mislabeled/outlier rows that dominate squared error; we add two very common filters: drop rows where pickup==dropoff but fare is non-trivial (GPS glitches) and remove extreme coordinate-delta rows (still within your broad NYC box) that create unrealistic “teleport” trips. We also add a very small fare floor (>=2.5) consistent with NYC minimum fare to reduce noise from invalid labels, without changing the model. Finally, we keep submission formatting identical, but ensure the test set isn’t accidentally reduced by passenger_count filtering (submission must include all keys) by only cleaning passenger_count values in test rather than dropping rows.'
- What this solution (achieved 7.04785) has done: 'Your current RMSE (14.206) is far above the target (3.453), so we need a real signal boost while keeping your exact core pipeline (engineered numeric features → StandardScaler → same `MLPRegressor` setup and fit call). The biggest minimal win here is to convert your distance feature into a more linear “fare driver” by adding a standard airport/city-center distance pair (pickup/dropoff distance to midtown Manhattan), which is simple feature engineering on existing coordinates and often drops RMSE substantially on this competition. I also apply the same NYC coordinate box filtering to the test set (without dropping rows) by clipping coordinates into the train box to avoid extreme extrapolation that produces huge errors. Finally, I keep your existing cleaning and prediction clipping, but ensure no NaNs propagate into engineered features after clipping.'
- What this solution (achieved 7.0205) has done: 'I fix the crash in the haversine feature engineering by making `haversine_km` accept either pandas Series/NumPy arrays or Python floats (for the Manhattan center constants), which currently causes the pipeline to stop before training and submission writing. This is a pure bug fix that preserves your feature set and modeling approach, while allowing the existing Manhattan-distance features to be computed correctly. I also add a small safety cast to ensure the engineered Manhattan-distance columns are numeric and aligned, without changing semantics. The rest of the pipeline (cleaning, scaling fit-on-train, MLPRegressor settings, clipping, and submission format) remain unchanged.'
- What this solution (achieved 7.06428) has done: 'Your current RMSE (7.0205) is well above the target (3.45343), so the smallest meaningful move toward the target is to fix a common modeling mismatch for this competition: the target `fare_amount` is extremely right-skewed and an MLP trained on raw dollars tends to over-penalize large fares, which hurts RMSE. Without changing your feature set, model type, or training loop, we can train on `log1p(fare_amount)` and then invert predictions with `expm1` at inference; this usually improves RMSE substantially and should move you closer to the target band. I also (minimally) tighten cleaning by removing the remaining “valid-but-damaging” high-error outliers (very large fares near the 100 cap and very long trips near 60km) via a simple upper trim based on existing filters, keeping semantics intact. Submission writing stays identical (`key,fare_amount`) and we still clip predictions to `[0, 100]` to avoid extreme errors.'
- What this solution (achieved 7.06508) has done: 'Your current RMSE (7.064) is still far above the target (3.453), so we should make a small, high-signal correction that reduces error without changing your core pipeline (same features → StandardScaler → same MLPRegressor fit). The biggest remaining mismatch is that you train on `log1p(fare_amount)` but compute RMSE in dollars; with a log-target, the optimal point prediction under squared error in log-space is the *conditional geometric mean*, which is biased low in dollar space. We can correct this with a single, legitimate post-processing step: apply a multiplicative bias-correction factor estimated from the validation residual variance in log-space (equivalent to assuming log-normal errors), then apply it to test predictions. This preserves the model, training loop, and feature engineering, but typically moves RMSE down noticeably toward your target; we keep your existing clipping and submission format unchanged.'
- What this solution (achieved 7.86018) has done: 'We need to move RMSE down from ~7.07 toward 3.45, so we should improve generalization with the smallest changes that don’t alter your core pipeline (same engineered numeric features → StandardScaler → same `MLPRegressor` and fit call). The biggest low-risk gain is to make the train/validation split representative of the Kaggle test distribution by splitting on time (pickup_datetime) instead of random; this avoids leakage-like distribution mixing and typically reduces public RMSE on this competition without changing the model. To keep semantics identical, we keep your log1p target and bias correction, but estimate the bias correction on a time-based validation window and then train the same model on all data for final predictions. Finally, we add one minimal, metric-aligned post-processing step: enforce the NYC minimum fare floor (2.5) on predictions (you already filtered training to >=2.5), which reduces squared-error impact from underestimates.'
- What this solution (achieved 7.1419) has done: 'Your current RMSE (7.86) is still far above the target (3.45), so we should make the smallest changes that typically reduce error on this competition without changing your core pipeline (same engineered numeric features → StandardScaler → same MLPRegressor fit calls). The biggest remaining issue is that `MLPRegressor` with `max_iter=20` is under-training; increasing iterations while keeping the exact same architecture/optimizer/loss is a direct way to reduce RMSE. We also add one minimal, standard distance-like feature (`manhattan_km` = lat_km + lon_km scaled by cos(lat)) derived from your existing coordinates, which usually helps a lot and doesn’t change the modeling approach. Finally, we apply the same bias-correction computation but keep prediction clipping and submission formatting unchanged so you still get a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # keep CPU-only for stability
random.seed(42)
np.random.seed(42)

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
df = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
    parse_dates=["pickup_datetime"],
    nrows=500000,
)
df.dropna(inplace=True)



## === cell 2
df



## === cell 3
test = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
    parse_dates=["pickup_datetime"],
)
test



## === cell 4
df.describe()



## === cell 5
test.describe()



## === cell 6
nyc_min_longitude = -74.3
nyc_max_longitude = -72
nyc_min_latitude = 40.63
nyc_max_latitude = 42



## === cell 7
for long in ["pickup_longitude", "dropoff_longitude"]:
    df = df[(df[long] > nyc_min_longitude) & (df[long] < nyc_max_longitude)]
for lat in ["pickup_latitude", "dropoff_latitude"]:
    df = df[(df[lat] > nyc_min_latitude) & (df[lat] < nyc_max_latitude)]

for long in ["pickup_longitude", "dropoff_longitude"]:
    test[long] = test[long].clip(nyc_min_longitude, nyc_max_longitude)
for lat in ["pickup_latitude", "dropoff_latitude"]:
    test[lat] = test[lat].clip(nyc_min_latitude, nyc_max_latitude)



## === cell 8
df.loc[df["passenger_count"] == 0, "passenger_count"] = 1
df = df[(df["passenger_count"] >= 1) & (df["passenger_count"] <= 6)].copy()

test.loc[test["passenger_count"] == 0, "passenger_count"] = 1
test.loc[test["passenger_count"] < 1, "passenger_count"] = 1
test.loc[test["passenger_count"] > 6, "passenger_count"] = 6



## === cell 9
df = df[(df["fare_amount"] >= 2.5) & (df["fare_amount"] <= 100)].copy()



## === cell 10
df.describe()



## === cell 11
df["longitude_diff"] = df["dropoff_longitude"] - df["pickup_longitude"]
df["latitude_diff"] = df["dropoff_latitude"] - df["pickup_latitude"]

df["abs_longitude_diff"] = df["longitude_diff"].abs()
df["abs_latitude_diff"] = df["latitude_diff"].abs()
df.describe()



## === cell 12
test["longitude_diff"] = test["dropoff_longitude"] - test["pickup_longitude"]
test["latitude_diff"] = test["dropoff_latitude"] - test["pickup_latitude"]

test["abs_longitude_diff"] = test["longitude_diff"].abs()
test["abs_latitude_diff"] = test["latitude_diff"].abs()
test.describe()



## === cell 13
df["year"] = df["pickup_datetime"].dt.year
df["month"] = df["pickup_datetime"].dt.month
df["day"] = df["pickup_datetime"].dt.day
df["day_of_week"] = df["pickup_datetime"].dt.dayofweek
df["hour"] = df["pickup_datetime"].dt.hour



## === cell 14
test["year"] = test["pickup_datetime"].dt.year
test["month"] = test["pickup_datetime"].dt.month
test["day"] = test["pickup_datetime"].dt.day
test["day_of_week"] = test["pickup_datetime"].dt.dayofweek
test["hour"] = test["pickup_datetime"].dt.hour



## === cell 15
test_keys = test["key"].copy()




## === cell 16
def euc_distance(lat1, long1, lat2, long2):
    return ((lat1 - lat2) ** 2 + (long1 - long2) ** 2) ** 0.5


df["travel_distance"] = euc_distance(
    df["pickup_latitude"],
    df["pickup_longitude"],
    df["dropoff_latitude"],
    df["dropoff_longitude"],
)
test["travel_distance"] = euc_distance(
    test["pickup_latitude"],
    test["pickup_longitude"],
    test["dropoff_latitude"],
    test["dropoff_longitude"],
)


def haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.asarray(lat1, dtype=np.float64)
    lon1 = np.asarray(lon1, dtype=np.float64)
    lat2 = np.asarray(lat2, dtype=np.float64)
    lon2 = np.asarray(lon2, dtype=np.float64)

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(a))
    return 6371.0 * c  # Earth radius in km


df["haversine_km"] = haversine_km(
    df["pickup_latitude"],
    df["pickup_longitude"],
    df["dropoff_latitude"],
    df["dropoff_longitude"],
)
test["haversine_km"] = haversine_km(
    test["pickup_latitude"],
    test["pickup_longitude"],
    test["dropoff_latitude"],
    test["dropoff_longitude"],
)

MANH_LAT, MANH_LON = 40.7580, -73.9855  # Times Square / Midtown approximation
df["pickup_to_manh_km"] = haversine_km(
    df["pickup_latitude"], df["pickup_longitude"], MANH_LAT, MANH_LON
).astype(np.float64)
df["dropoff_to_manh_km"] = haversine_km(
    df["dropoff_latitude"], df["dropoff_longitude"], MANH_LAT, MANH_LON
).astype(np.float64)

test["pickup_to_manh_km"] = haversine_km(
    test["pickup_latitude"], test["pickup_longitude"], MANH_LAT, MANH_LON
).astype(np.float64)
test["dropoff_to_manh_km"] = haversine_km(
    test["dropoff_latitude"], test["dropoff_longitude"], MANH_LAT, MANH_LON
).astype(np.float64)


def manhattan_km(lat1, lon1, lat2, lon2):
    lat1 = np.asarray(lat1, dtype=np.float64)
    lon1 = np.asarray(lon1, dtype=np.float64)
    lat2 = np.asarray(lat2, dtype=np.float64)
    lon2 = np.asarray(lon2, dtype=np.float64)

    mean_lat = np.radians((lat1 + lat2) / 2.0)
    dlat_km = 111.32 * np.abs(lat2 - lat1)
    dlon_km = 111.32 * np.cos(mean_lat) * np.abs(lon2 - lon1)
    return dlat_km + dlon_km


df["manhattan_km"] = manhattan_km(
    df["pickup_latitude"],
    df["pickup_longitude"],
    df["dropoff_latitude"],
    df["dropoff_longitude"],
).astype(np.float64)
test["manhattan_km"] = manhattan_km(
    test["pickup_latitude"],
    test["pickup_longitude"],
    test["dropoff_latitude"],
    test["dropoff_longitude"],
).astype(np.float64)

JFK_LAT, JFK_LON = 40.6413, -73.7781
LGA_LAT, LGA_LON = 40.7769, -73.8740
EWR_LAT, EWR_LON = 40.6895, -74.1745

df["pickup_to_jfk_km"] = haversine_km(
    df["pickup_latitude"], df["pickup_longitude"], JFK_LAT, JFK_LON
).astype(np.float64)
df["dropoff_to_jfk_km"] = haversine_km(
    df["dropoff_latitude"], df["dropoff_longitude"], JFK_LAT, JFK_LON
).astype(np.float64)
df["pickup_to_lga_km"] = haversine_km(
    df["pickup_latitude"], df["pickup_longitude"], LGA_LAT, LGA_LON
).astype(np.float64)
df["dropoff_to_lga_km"] = haversine_km(
    df["dropoff_latitude"], df["dropoff_longitude"], LGA_LAT, LGA_LON
).astype(np.float64)
df["pickup_to_ewr_km"] = haversine_km(
    df["pickup_latitude"], df["pickup_longitude"], EWR_LAT, EWR_LON
).astype(np.float64)
df["dropoff_to_ewr_km"] = haversine_km(
    df["dropoff_latitude"], df["dropoff_longitude"], EWR_LAT, EWR_LON
).astype(np.float64)

test["pickup_to_jfk_km"] = haversine_km(
    test["pickup_latitude"], test["pickup_longitude"], JFK_LAT, JFK_LON
).astype(np.float64)
test["dropoff_to_jfk_km"] = haversine_km(
    test["dropoff_latitude"], test["dropoff_longitude"], JFK_LAT, JFK_LON
).astype(np.float64)
test["pickup_to_lga_km"] = haversine_km(
    test["pickup_latitude"], test["pickup_longitude"], LGA_LAT, LGA_LON
).astype(np.float64)
test["dropoff_to_lga_km"] = haversine_km(
    test["dropoff_latitude"], test["dropoff_longitude"], LGA_LAT, LGA_LON
).astype(np.float64)
test["pickup_to_ewr_km"] = haversine_km(
    test["pickup_latitude"], test["pickup_longitude"], EWR_LAT, EWR_LON
).astype(np.float64)
test["dropoff_to_ewr_km"] = haversine_km(
    test["dropoff_latitude"], test["dropoff_longitude"], EWR_LAT, EWR_LON
).astype(np.float64)

thr_km = 2.0
df["near_jfk"] = (
    (df["pickup_to_jfk_km"] < thr_km) | (df["dropoff_to_jfk_km"] < thr_km)
).astype(np.int8)
df["near_lga"] = (
    (df["pickup_to_lga_km"] < thr_km) | (df["dropoff_to_lga_km"] < thr_km)
).astype(np.int8)
df["near_ewr"] = (
    (df["pickup_to_ewr_km"] < thr_km) | (df["dropoff_to_ewr_km"] < thr_km)
).astype(np.int8)

test["near_jfk"] = (
    (test["pickup_to_jfk_km"] < thr_km) | (test["dropoff_to_jfk_km"] < thr_km)
).astype(np.int8)
test["near_lga"] = (
    (test["pickup_to_lga_km"] < thr_km) | (test["dropoff_to_lga_km"] < thr_km)
).astype(np.int8)
test["near_ewr"] = (
    (test["pickup_to_ewr_km"] < thr_km) | (test["dropoff_to_ewr_km"] < thr_km)
).astype(np.int8)



## === cell 17
df.describe()



## === cell 18
test.describe()



## === cell 19
print(df.isnull().sum())
print(test.isnull().sum())



## === cell 20
same_point = (df["abs_longitude_diff"] < 1e-6) & (df["abs_latitude_diff"] < 1e-6)
df = df[~(same_point & (df["fare_amount"] > 2.5))].copy()

df = df[(df["abs_longitude_diff"] <= 0.7) & (df["abs_latitude_diff"] <= 0.7)].copy()

eps_km = 0.01  # 10 meters
df = df[~((df["haversine_km"] < eps_km) & (df["fare_amount"] > 2.5))].copy()

df = df[df["haversine_km"] <= 60.0].copy()

dist_for_rate = 0.3  # km
rate = df["fare_amount"] / np.maximum(df["haversine_km"], dist_for_rate)
df = df[(rate > 1.0) & (rate < 30.0)].copy()

df = df[(df["fare_amount"] <= 80.0) & (df["haversine_km"] <= 45.0)].copy()

print("After cleaning, df shape:", df.shape)



## === cell 21
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor



## === cell 22
df = df.sort_values("pickup_datetime").reset_index(drop=True)

val_fraction = 0.2
cut = int(len(df) * (1.0 - val_fraction))
df_train = df.iloc[:cut].copy()
df_val = df.iloc[cut:].copy()

y_train_raw = df_train["fare_amount"].astype(np.float32)
y_val_raw = df_val["fare_amount"].astype(np.float32)

y_train = np.log1p(y_train_raw).astype(np.float32)
y_val = np.log1p(y_val_raw).astype(np.float32)

X_train_df = df_train.drop(["fare_amount", "key", "pickup_datetime"], axis=1)
X_val_df = df_val.drop(["fare_amount", "key", "pickup_datetime"], axis=1)

X_test_full = test.drop(["key", "pickup_datetime"], axis=1)
X_test_full = X_test_full.reindex(columns=X_train_df.columns)

X_train_df = X_train_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
X_val_df = X_val_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
X_test_full = X_test_full.replace([np.inf, -np.inf], np.nan).fillna(0.0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_df)
X_val_scaled = scaler.transform(X_val_df)
X_test_scaled = scaler.transform(X_test_full)



## === cell 23
model = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32, 8),
    activation="relu",
    solver="adam",
    alpha=0.0,  # keep identical core regularization choice
    batch_size=256,
    learning_rate_init=0.001,
    max_iter=60,  # keep as-is (already increased)
    shuffle=True,
    random_state=42,
    early_stopping=False,  # do not introduce early stopping
    n_iter_no_change=1000,  # irrelevant when early_stopping=False
    verbose=True,
)

model.fit(X_train_scaled, y_train)

train_pred_log = model.predict(X_train_scaled).reshape(-1)
val_pred_log = model.predict(X_val_scaled).reshape(-1)

val_resid = (y_val - val_pred_log).astype(np.float64)
val_resid = np.clip(val_resid, -3.0, 3.0)  # very conservative clip in log-space
sigma2 = float(np.mean(val_resid**2))
bias_correction = float(np.exp(0.5 * sigma2))
print(
    f"Log-space residual variance sigma^2={sigma2:.6f}, bias_correction={bias_correction:.6f}"
)

train_pred = np.expm1(train_pred_log) * bias_correction
val_pred = np.expm1(val_pred_log) * bias_correction

train_rmse = np.sqrt(mean_squared_error(np.expm1(y_train), train_pred))
val_rmse = np.sqrt(mean_squared_error(np.expm1(y_val), val_pred))

print("Train RMSE (dollars): {:0.2f}".format(train_rmse))
print("Val RMSE (dollars): {:0.2f}".format(val_rmse))
print("------------------------")



## === cell 24
y_raw_all = df["fare_amount"].astype(np.float32)
y_all = np.log1p(y_raw_all).astype(np.float32)

X_all = df.drop(["fare_amount", "key", "pickup_datetime"], axis=1)
X_all = X_all.replace([np.inf, -np.inf], np.nan).fillna(0.0)

X_test_full2 = test.drop(["key", "pickup_datetime"], axis=1)
X_test_full2 = X_test_full2.reindex(columns=X_all.columns)
X_test_full2 = X_test_full2.replace([np.inf, -np.inf], np.nan).fillna(0.0)

scaler_all = StandardScaler()
X_all_scaled = scaler_all.fit_transform(X_all)
X_test_scaled2 = scaler_all.transform(X_test_full2)

model_all = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32, 8),
    activation="relu",
    solver="adam",
    alpha=0.0,
    batch_size=256,
    learning_rate_init=0.001,
    max_iter=60,
    shuffle=True,
    random_state=42,
    early_stopping=False,
    n_iter_no_change=1000,
    verbose=True,
)
model_all.fit(X_all_scaled, y_all)

pred_log = model_all.predict(X_test_scaled2).reshape(-1)
pred_log.shape



## === cell 25
pred_log = np.nan_to_num(pred_log, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

pred = (np.expm1(pred_log).astype(np.float32) * np.float32(bias_correction)).astype(
    np.float32
)
pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

pred = np.clip(pred, 2.5, 100.0)

submission = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
)
submission["key"] = test_keys.values
submission["fare_amount"] = pred

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
