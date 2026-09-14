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

3.95459

# 6. Current score

6.59575

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.53501) has done: 'I fix the haversine helper to accept scalar airport coordinates (the current error comes from calling `.astype` on Python floats), which allow airport distance features to be created and downstream feature logic to work. Then I make the train/test feature columns consistent by dropping the same columns from both with `errors="ignore"` and by ensuring `pickup_datetime` is not passed into XGBoost (it currently triggers a dtype error). Finally, I update the XGBoost parameters to use the current objective name (`reg:squarederror`) while keeping the same training loop and write a valid `taxi_fare_submission.csv` with the required `key,fare_amount` columns.'
- What this solution (achieved 10.69171) has done: 'Your current gap is 5.53501 − 3.95459 = 1.58042 (worse than target; lower RMSE is better), so we need a modest but real improvement without changing the overall approach. The biggest low-risk win here is to stop dropping the airport distance features you carefully created (they are informative), while keeping `pickup_datetime` excluded to avoid dtype issues and keeping the same XGBoost training loop/objective. I also add a minimal, standard label-cleaning filter for extreme/unrealistic fares (very large outliers disproportionately hurt RMSE) and ensure the holdout/test feature frame gets the same datetime-derived columns filled safely. These are small, direct changes that typically move this baseline meaningfully toward ~4 RMSE without altering the core model/training semantics.'
- What this solution (achieved 6.79771) has done: 'We need to move RMSE down from 10.69 toward 3.95, so we should make small, high-impact fixes without changing the overall XGBoost approach. The biggest likely issue is that your train sample is the *first* 100k rows (very old dates) which badly mismatches the test distribution; switching to a deterministic random sample from the full file keeps the same pipeline but improves generalization. Next, we apply the exact same geographic bounding-box filter to the test set (clipping rather than dropping) to prevent extreme/invalid coordinates from producing crazy distances and predictions. Finally, we train using the best_iteration found on the tiny validation split but then refit once on all training data for that fixed number of boosting rounds (same model/params, just uses more data), which typically reduces RMSE substantially while preserving the core logic.'
- What this solution (achieved 11.49554) has done: 'We need to move RMSE down from 6.80 toward 3.95 (lower is better), so we make the smallest changes that typically improve generalization without changing your model/training loop or feature set. The biggest low-risk win is to read a larger, still-random sample from train (your current 100k is quite small for this noisy regression), keeping the same random skiprows sampling logic. Next, we apply one more standard NYC Taxi cleanup that directly reduces RMSE: drop rows with invalid/zero coordinates and cap extreme trip distances more conservatively after computing haversine (these outliers hurt squared error disproportionately). Finally, we ensure prediction post-processing doesn’t hurt RMSE by removing the rounding to cents (rounding adds error on Kaggle’s continuous RMSE metric) while keeping non-negativity clipping.'
- What this solution (achieved 5.66592) has done: 'To move RMSE down from 11.50 toward the 3.95 target without changing your model/training loop or feature set, I make three small, high-impact data-quality fixes that directly reduce squared error: (1) remove obvious geographic/fare outliers more robustly (NYC bounding box + plausible fare by distance), (2) add a simple “manhattan distance” feature alongside haversine (still same feature-engineering approach; just one extra deterministic distance feature), and (3) ensure the train sampling is truly uniform and reproducible by sampling row indices instead of probabilistic `skiprows` (same “random sample from full file” logic but less variance and less bias). These changes typically cut RMSE substantially while keeping the same XGBoost objective, training procedure, and submission semantics. The pipeline still writes `taxi_fare_submission.csv` with `key,fare_amount`.'
- What this solution (achieved 6.02752) has done: 'Your current RMSE (5.66592) is still above the target (3.95459), so we should make a small change that usually yields a meaningful RMSE drop without changing the model/training loop or feature set. The biggest low-risk issue is the `trip_distance < 25km` filter, which removes many legitimate long trips (especially airport rides) and can bias the model, hurting squared error; we relax this cap modestly while keeping the rest of your cleaning logic intact. To prevent the relaxed cap from introducing extreme outliers, we add a very conservative upper bound on `trip_distance` (still a standard cleanup) and keep your existing fare/trip_rate filters. Everything else (features, DBSCAN logic, XGBoost training/early stopping/refit, submission format/path) stays the same.'
- What this solution (achieved 5.99012) has done: 'To move RMSE down from 6.02752 toward the 3.95459 target (lower is better) with minimal disruption, I keep your exact feature set, DBSCAN step, and XGBoost training loop, but fix one high-impact data issue: your sampling can include many rows from very different distributions/outliers because it’s purely uniform and only lightly cleaned. I add a single standard NYC Taxi cleanup that directly reduces squared error without changing modeling: remove rows where `trip_distance` is inconsistent with coordinates (very tiny distance but non-trivial fare, and extremely large distance with tiny fare), using conservative bounds so we don’t throw away valid airport trips. I also ensure the `trip_distance` filter is not redundant/conflicting by consolidating it into one consistent condition (still the same thresholds you intended), which reduces accidental retention of pathological rows. Everything else stays the same, and the script still writes a valid `taxi_fare_submission.csv` with `key,fare_amount`.'
- What this solution (achieved 6.75045) has done: 'We need to move RMSE down from 5.99012 toward the 3.95459 target (lower is better), so the smallest safe improvements are to (1) add one standard, very low-risk geographic feature (`abs_dlon`, `abs_dlat`) that helps XGBoost model directionality/scale without changing the overall approach, and (2) make the train/test distribution closer by lightly tightening the “bad short/long trip” filters using fare-per-km consistency (this reduces squared-error blowups from mislabeled/noisy rows while keeping airport trips). I keep your sampling strategy, DBSCAN step, datetime features, airport features, and XGBoost training/refit loop unchanged. The submission format and output path remain identical.'
- What this solution (achieved 6.75045) has done: 'Your current RMSE (6.75045) is worse than the target (3.95459), so we should make small, high-impact generalization improvements without changing the XGBoost approach or feature logic. The biggest likely issue is that the `trip_distance` cap is still effectively 60km (because you filter twice), which removes many legitimate airport/longer rides and can hurt RMSE; I remove the redundant tighter cap so the intended 200km cap applies. Next, I add `tree_method="hist"` to keep training stable and fast within the timeout while keeping the same model class and objective. Finally, I ensure the holdout/test has no missing numeric values before DMatrix creation (rare but can cause unpredictable behavior), while keeping the same submission format.'
- What this solution (achieved 6.59317) has done: 'Your current RMSE (6.75045) is still far above the target (3.95459), so we need a meaningful improvement while keeping your XGBoost pipeline and feature logic intact. The biggest high-impact, low-risk fix is to stop letting noisy/outlier labels dominate RMSE by adding one standard NYC Taxi cleaning step: drop rows with obviously wrong fare-vs-distance consistency (using conservative bounds and keeping airport trips). Next, we align train/test feature distributions a bit better by adding the same basic “coordinate validity” handling to the test set (clip already exists, but we also guard passenger_count and NaNs consistently). Finally, we keep the same training loop but make it deterministic and slightly more stable by setting `seed`/`random_state` in XGBoost params (no change in approach, just reduces variance and usually improves generalization slightly).'
- What this solution (achieved 6.59575) has done: 'Your current RMSE (6.59317) is far above the target (3.95459), so we should make a small, legitimate improvement that typically reduces squared-error blowups without changing your XGBoost pipeline or feature engineering approach. The biggest low-risk issue here is that the model can still see rare but extreme outlier labels (and some remaining coordinate/pathology cases) that disproportionately hurt RMSE; we add one conservative, standard NYC Taxi cleanup step based on a generous “fare per km” band and remove a few remaining invalid datetime rows. Then we ensure the exact same cleanup-derived feature columns exist and are finite in both train and test before building DMatrix (avoids silent NaN-driven splits/prediction quirks). Everything else (sampling strategy, feature set incl. DBSCAN/airport features, train/early-stopping/refit loop, objective/metric, submission schema/path) stays the same.'

# 9. Code solution

## === cell 0
KMS_PER_RADIAN = 6371.0088

JFK_GEO_LOCATION = (40.6413, -73.7781)
LGR_GEO_LOCATION = (40.7769, -73.8740)
EWR_GEO_LOCATION = (40.6895, -74.1745)



## === cell 1
MAX_TRAINING_SIZE = 300_000

EPS_IN_KM = 0.5  ## NOTE that lat/long are available till 5th decimal value & 0.1km = 1.xe-5, hence avoid using smaller DBSCAN's eps, i.e., radius threshold for clustering
MIN_SAMPLES_CLUSTER = 500

RADIUS_VICINITY_AIRPORTS = 1.0

THERSHOLD_TRIP_FARE_RATE = 50.0

THRESHOLD_TRIP_DISTANCE = 60.0



## === cell 2
import os
import timeit

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN

import xgboost as xgb

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-whitegrid")

np.random.seed(0)



## === cell 3
BASE_INPUT_CANDIDATES = [
    "/kaggle/input/new-york-city-taxi-fare-prediction",
    "/kaggle/input",
    "../input/new-york-city-taxi-fare-prediction",
    "../input",
]


def _find_file(filename: str) -> str:
    for base in BASE_INPUT_CANDIDATES:
        path = os.path.join(base, filename)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not find {filename} under {BASE_INPUT_CANDIDATES}")


TRAIN_PATH = _find_file("train.csv")
TEST_PATH = _find_file("test.csv")
SAMPLE_SUB_PATH = _find_file("sample_submission.csv")

TRAIN_PATH, TEST_PATH, SAMPLE_SUB_PATH



## === cell 4
start_time = timeit.default_timer()

TRAIN_N_ROWS = (
    55_423_856  # provided in prompt; used only to sample indices reproducibly
)
rng = np.random.RandomState(0)
skip_idx = rng.choice(
    np.arange(1, TRAIN_N_ROWS + 1), size=TRAIN_N_ROWS - MAX_TRAINING_SIZE, replace=False
)
skip_idx = np.sort(skip_idx)

df_train = pd.read_csv(
    TRAIN_PATH,
    parse_dates=["pickup_datetime"],
    skiprows=skip_idx,
)

if len(df_train) > MAX_TRAINING_SIZE:
    df_train = df_train.sample(n=MAX_TRAINING_SIZE, random_state=0)

df_holdout = pd.read_csv(TEST_PATH, parse_dates=["pickup_datetime"])

test_key = df_holdout["key"].copy()

df_train.drop(columns=["key"], inplace=True)
df_holdout.drop(columns=["key"], inplace=True)

elapsed = timeit.default_timer() - start_time
len(df_train), elapsed



## === cell 5
print("Old size: %d" % len(df_train))

df_train = df_train[df_train.fare_amount >= 0]
df_train = df_train.dropna(how="any", axis="rows")

df_train = df_train.drop(
    index=df_train[df_train.passenger_count >= 7].index, axis="rows"
)
df_train = df_train.drop(
    index=df_train[df_train.passenger_count == 0].index, axis="rows"
)

df_train = df_train[(df_train.fare_amount > 0) & (df_train.fare_amount <= 250)]

coord_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]
for c in coord_cols:
    df_train = df_train[np.isfinite(df_train[c])]
df_train = df_train[
    (df_train.pickup_longitude != 0)
    & (df_train.pickup_latitude != 0)
    & (df_train.dropoff_longitude != 0)
    & (df_train.dropoff_latitude != 0)
]

print("New size: %d" % len(df_train))




## === cell 6
def select_within_boundingbox(df, BB):
    return (
        (df.pickup_longitude >= BB[0])
        & (df.pickup_longitude <= BB[1])
        & (df.pickup_latitude >= BB[2])
        & (df.pickup_latitude <= BB[3])
        & (df.dropoff_longitude >= BB[0])
        & (df.dropoff_longitude <= BB[1])
        & (df.dropoff_latitude >= BB[2])
        & (df.dropoff_latitude <= BB[3])
    )


BB = (-74.5, -72.8, 40.5, 41.8)

print("Old size: %d" % len(df_train))
df_train = df_train[select_within_boundingbox(df_train, BB)]
print("New size: %d" % len(df_train))

df_holdout["pickup_longitude"] = df_holdout["pickup_longitude"].clip(BB[0], BB[1])
df_holdout["dropoff_longitude"] = df_holdout["dropoff_longitude"].clip(BB[0], BB[1])
df_holdout["pickup_latitude"] = df_holdout["pickup_latitude"].clip(BB[2], BB[3])
df_holdout["dropoff_latitude"] = df_holdout["dropoff_latitude"].clip(BB[2], BB[3])

df_holdout["passenger_count"] = pd.to_numeric(
    df_holdout["passenger_count"], errors="coerce"
)
df_holdout["passenger_count"] = (
    df_holdout["passenger_count"].fillna(1).clip(1, 6).astype(np.int16)
)

for c in coord_cols:
    df_holdout[c] = pd.to_numeric(df_holdout[c], errors="coerce").fillna(0.0)




## === cell 7
def _haversine_km_vec(lat1, lon1, lat2, lon2):
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return KMS_PER_RADIAN * c


def addPickDropDistanceFeature(df):
    df["trip_distance"] = _haversine_km_vec(
        df["pickup_latitude"].values,
        df["pickup_longitude"].values,
        df["dropoff_latitude"].values,
        df["dropoff_longitude"].values,
    )
    return df


def addAirportDistanceFeatures(df):
    df["pickup_distance_to_jfk"] = _haversine_km_vec(
        df["pickup_latitude"].values,
        df["pickup_longitude"].values,
        JFK_GEO_LOCATION[0],
        JFK_GEO_LOCATION[1],
    )
    df["drop_distance_to_jfk"] = _haversine_km_vec(
        df["dropoff_latitude"].values,
        df["dropoff_longitude"].values,
        JFK_GEO_LOCATION[0],
        JFK_GEO_LOCATION[1],
    )

    df["pickup_distance_to_lgr"] = _haversine_km_vec(
        df["pickup_latitude"].values,
        df["pickup_longitude"].values,
        LGR_GEO_LOCATION[0],
        LGR_GEO_LOCATION[1],
    )
    df["drop_distance_to_lgr"] = _haversine_km_vec(
        df["dropoff_latitude"].values,
        df["dropoff_longitude"].values,
        LGR_GEO_LOCATION[0],
        LGR_GEO_LOCATION[1],
    )

    df["pickup_distance_to_ewr"] = _haversine_km_vec(
        df["pickup_latitude"].values,
        df["pickup_longitude"].values,
        EWR_GEO_LOCATION[0],
        EWR_GEO_LOCATION[1],
    )
    df["drop_distance_to_ewr"] = _haversine_km_vec(
        df["dropoff_latitude"].values,
        df["dropoff_longitude"].values,
        EWR_GEO_LOCATION[0],
        EWR_GEO_LOCATION[1],
    )
    return df


def getAirportTrips(df, airportVicinity):
    ids = (
        (df.pickup_distance_to_jfk < airportVicinity)
        | (df.drop_distance_to_jfk < airportVicinity)
        | (df.pickup_distance_to_lgr < airportVicinity)
        | (df.drop_distance_to_lgr < airportVicinity)
        | (df.pickup_distance_to_ewr < airportVicinity)
        | (df.drop_distance_to_ewr < airportVicinity)
    )
    return ids




## === cell 8
start_time = timeit.default_timer()

df_train = addPickDropDistanceFeature(df_train)
df_holdout = addPickDropDistanceFeature(df_holdout)

km_per_degree_lat = 111.32
mean_lat_rad_train = np.radians(df_train["pickup_latitude"].astype(float).values)
cos_mean_lat_train = np.cos(mean_lat_rad_train)

df_train["trip_manhattan_km"] = (
    np.abs(df_train["dropoff_latitude"].values - df_train["pickup_latitude"].values)
    * km_per_degree_lat
) + (
    np.abs(df_train["dropoff_longitude"].values - df_train["pickup_longitude"].values)
    * km_per_degree_lat
    * cos_mean_lat_train
)

mean_lat_rad_test = np.radians(df_holdout["pickup_latitude"].astype(float).values)
cos_mean_lat_test = np.cos(mean_lat_rad_test)
df_holdout["trip_manhattan_km"] = (
    np.abs(df_holdout["dropoff_latitude"].values - df_holdout["pickup_latitude"].values)
    * km_per_degree_lat
) + (
    np.abs(
        df_holdout["dropoff_longitude"].values - df_holdout["pickup_longitude"].values
    )
    * km_per_degree_lat
    * cos_mean_lat_test
)

df_train["abs_dlon"] = np.abs(
    df_train["dropoff_longitude"].values - df_train["pickup_longitude"].values
).astype(float)
df_train["abs_dlat"] = np.abs(
    df_train["dropoff_latitude"].values - df_train["pickup_latitude"].values
).astype(float)

df_holdout["abs_dlon"] = np.abs(
    df_holdout["dropoff_longitude"].values - df_holdout["pickup_longitude"].values
).astype(float)
df_holdout["abs_dlat"] = np.abs(
    df_holdout["dropoff_latitude"].values - df_holdout["pickup_latitude"].values
).astype(float)

elapsed = timeit.default_timer() - start_time
elapsed



## === cell 9
try:
    bucketsCount = 100
    feat = "trip_distance"
    df_train[feat].hist(bins=bucketsCount, figsize=(15, 8))
    df_holdout[feat].hist(bins=bucketsCount, figsize=(15, 8))
    plt.yscale("log")
    plt.xlabel(feat)
    plt.ylabel("Frequency Log")
    plt.close()
except Exception:
    pass



## === cell 10
print("Old size: %d" % len(df_train))

df_train = df_train[(df_train.trip_distance > 0) & (df_train.trip_distance < 200.0)]

short_trip_bad = (df_train.trip_distance < 0.8) & (df_train.fare_amount > 60)
long_trip_bad = (df_train.trip_distance > 35.0) & (df_train.fare_amount < 8.0)

fare_per_km = df_train.fare_amount / np.maximum(df_train.trip_distance, 0.2)
inconsistent_low = (df_train.trip_distance > 10.0) & (fare_per_km < 1.0)
inconsistent_high = (df_train.trip_distance > 2.0) & (fare_per_km > 80.0)

min_expected = 2.5 + 1.0 * df_train.trip_distance  # low base + low per-km
max_expected = (
    7.0 + 25.0 * df_train.trip_distance
)  # generous per-km for traffic/surcharges
fare_distance_inconsistent = (df_train.fare_amount < min_expected) | (
    df_train.fare_amount > max_expected
)

df_train = df_train[
    ~(
        short_trip_bad
        | long_trip_bad
        | inconsistent_low
        | inconsistent_high
        | fare_distance_inconsistent
    )
]

fare_per_km2 = df_train.fare_amount / np.maximum(df_train.trip_distance, 0.5)
df_train = df_train[(fare_per_km2 >= 1.2) & (fare_per_km2 <= 40.0)]

print("New size: %d" % len(df_train))




## === cell 11
def add_datetime_features(df):
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df["hour"] = df.pickup_datetime.dt.hour
    df["day"] = df.pickup_datetime.dt.day
    df["month"] = df.pickup_datetime.dt.month
    df["weekday"] = df.pickup_datetime.dt.weekday
    df["year"] = df.pickup_datetime.dt.year
    return df


start_time = timeit.default_timer()

df_train = add_datetime_features(df_train)
df_holdout = add_datetime_features(df_holdout)

df_train = df_train[df_train["pickup_datetime"].notna()]

for c in ["hour", "day", "month", "weekday", "year"]:
    if c in df_train.columns:
        df_train[c] = df_train[c].fillna(-1).astype(np.int16)
    if c in df_holdout.columns:
        df_holdout[c] = df_holdout[c].fillna(-1).astype(np.int16)

elapsed = timeit.default_timer() - start_time
elapsed



## === cell 12
train_len = len(df_train)
df_nyc_taxi = pd.concat([df_train, df_holdout], axis=0, ignore_index=False, sort=False)

train_len, df_nyc_taxi.shape



## === cell 13
EPS_IN_RADIAN = EPS_IN_KM / KMS_PER_RADIAN



## === cell 14
start_time = timeit.default_timer()

pickup_coords = np.radians(df_nyc_taxi[["pickup_latitude", "pickup_longitude"]].values)
dbscan_pick = DBSCAN(
    eps=EPS_IN_RADIAN,
    min_samples=MIN_SAMPLES_CLUSTER,
    algorithm="ball_tree",
    metric="haversine",
).fit(pickup_coords)
labels_pick = dbscan_pick.labels_

elapsed = timeit.default_timer() - start_time
elapsed



## === cell 15
start_time = timeit.default_timer()

drop_coords = np.radians(df_nyc_taxi[["dropoff_latitude", "dropoff_longitude"]].values)
dbscan_drop = DBSCAN(
    eps=EPS_IN_RADIAN,
    min_samples=MIN_SAMPLES_CLUSTER,
    algorithm="ball_tree",
    metric="haversine",
).fit(drop_coords)
labels_drop = dbscan_drop.labels_

elapsed = timeit.default_timer() - start_time
elapsed



## === cell 16
df_nyc_taxi["dense_DBSCAN_trips"] = (labels_pick != -1) & (labels_drop != -1)



## === cell 17
try:
    df_tmp = df_nyc_taxi.loc[df_nyc_taxi.dense_DBSCAN_trips == 1]
    plt.figure(figsize=(8, 6))
    plt.plot(df_tmp.pickup_longitude, df_tmp.pickup_latitude, "o", markersize=1)
    plt.close()
except Exception:
    pass



## === cell 18
df_train = df_nyc_taxi.iloc[:train_len, :].copy()
df_holdout = df_nyc_taxi.iloc[train_len:, :].copy()
if "fare_amount" in df_holdout.columns:
    df_holdout = df_holdout.drop(columns=["fare_amount"])

(len(df_train), len(df_holdout))



## === cell 19
df_train.loc[df_train.trip_distance < 0.2, "trip_distance"] = 0.2

try:
    (df_train.fare_amount / df_train.trip_distance).hist(bins=100, figsize=(15, 8))
    plt.yscale("log")
    plt.xlabel("trip_rate")
    plt.ylabel("Log Frequency")
    plt.close()
except Exception:
    pass



## === cell 20
df_train["trip_rate"] = (df_train["fare_amount"] / df_train["trip_distance"]).astype(
    float
)



## === cell 21
ids = (df_train.trip_rate > 0) & (df_train.trip_rate < THERSHOLD_TRIP_FARE_RATE)

print("Old size: %d" % len(df_train))
df_train = df_train[ids]
print("New size: %d" % len(df_train))



## === cell 22
start_time = timeit.default_timer()

df_train = addAirportDistanceFeatures(df_train)
df_holdout = addAirportDistanceFeatures(df_holdout)

elapsed = timeit.default_timer() - start_time
elapsed



## === cell 23
airportTripsIds = getAirportTrips(df_holdout, RADIUS_VICINITY_AIRPORTS)
df_holdout["airport_bound"] = airportTripsIds

airportTripsIds_train = getAirportTrips(df_train, RADIUS_VICINITY_AIRPORTS)
df_train["airport_bound"] = airportTripsIds_train

df_airport_trips = df_train.loc[airportTripsIds_train]
df_city_trips = df_train.loc[~airportTripsIds_train]

(len(df_airport_trips), len(df_city_trips))



## === cell 24
try:
    pd.DataFrame(
        data={
            "Airport Trips": df_airport_trips.trip_rate,
            "City Trips": df_city_trips.trip_rate,
        }
    ).describe()
except Exception:
    pass



## === cell 25
try:
    pd.DataFrame(
        data={
            "Good Density Trips": df_train.loc[
                df_train.dense_DBSCAN_trips == 1
            ].trip_rate,
            "LOW Density Pickups": df_train.loc[
                df_train.dense_DBSCAN_trips == 0
            ].trip_rate,
        }
    ).describe()
except Exception:
    pass



## === cell 26
DROP_COLS_FOR_MODEL = [
    "pickup_datetime",
    "trip_rate",
]

df_train = df_train.drop(columns=DROP_COLS_FOR_MODEL, errors="ignore")
df_train.info()



## === cell 27
y = df_train["fare_amount"]
train = df_train.drop(columns=["fare_amount"])

x_train, x_test, y_train, y_test = train_test_split(
    train, y, random_state=0, test_size=0.01
)

x_train.shape, x_test.shape



## === cell 28
params = {
    "max_depth": 8,  # Result of tuning with CV
    "eta": 0.03,  # Result of tuning with CV
    "subsample": 1,  # Result of tuning with CV
    "colsample_bytree": 0.8,  # Result of tuning with CV
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "seed": 0,
    "random_state": 0,
    "silent": 1,
}

CV = False
if CV:
    dtrain = xgb.DMatrix(train, label=y)
    gridsearch_params = [(eta) for eta in np.arange(0.04, 0.12, 0.02)]

    min_rmse = float("Inf")
    best_params = None
    for eta in gridsearch_params:
        print("CV with eta={} ".format(eta))

        params["eta"] = eta

        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=1000,
            nfold=3,
            metrics={"rmse"},
            early_stopping_rounds=10,
        )

        mean_rmse = cv_results["test-rmse-mean"].min()
        boost_rounds = cv_results["test-rmse-mean"].argmin()
        print("\tRMSE {} for {} rounds".format(mean_rmse, boost_rounds))
        if mean_rmse < min_rmse:
            min_rmse = mean_rmse
            best_params = eta

    print("Best params: {}, RMSE: {}".format(best_params, min_rmse))
else:
    params["silent"] = 0  # Turn on output
    print(params)




## === cell 29
def XGBmodel(x_train, x_test, y_train, y_test, params):
    matrix_train = xgb.DMatrix(x_train, label=y_train)
    matrix_test = xgb.DMatrix(x_test, label=y_test)
    model = xgb.train(
        params=params,
        dtrain=matrix_train,
        num_boost_round=5000,
        early_stopping_rounds=10,
        evals=[(matrix_test, "test")],
    )
    return model


model = XGBmodel(x_train, x_test, y_train, y_test, params)



## === cell 30
best_rounds = int(model.best_iteration + 1)
dtrain_full = xgb.DMatrix(train, label=y)
model_full = xgb.train(params=params, dtrain=dtrain_full, num_boost_round=best_rounds)

best_rounds



## === cell 31
x_pred = df_holdout.drop(columns=DROP_COLS_FOR_MODEL, errors="ignore")

for c in x_pred.columns:
    if x_pred[c].dtype.kind in "iufb":
        x_pred[c] = np.where(np.isfinite(x_pred[c].values), x_pred[c].values, 0.0)
    else:
        x_pred[c] = pd.to_numeric(x_pred[c], errors="coerce").fillna(0.0)

x_pred = x_pred.reindex(columns=train.columns, fill_value=0)
x_pred = x_pred.fillna(0)

prediction = model_full.predict(xgb.DMatrix(x_pred))

prediction[:5], len(prediction)



## === cell 32
len(test_key)



## === cell 33
prediction = np.maximum(prediction, 0)

submission = pd.DataFrame({"key": test_key, "fare_amount": prediction})

out_path = "taxi_fare_submission.csv"
submission.to_csv(out_path, index=False)

submission.head(), out_path
