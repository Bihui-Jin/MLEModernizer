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

3.87285

# 6. Current score

6.61272

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 7.4143) has done: 'I fix the plotting cells so they no longer crash the notebook by ensuring the x/y arrays passed to seaborn have matching lengths and by using the correct trained model variable name. I also fix the input paths to match this environment’s `/kaggle/input/` layout so the script runs end-to-end on Kaggle without manual path edits. To move the RMSE toward your target (lower is better), I make two minimal, standard changes that don’t alter the overall approach: remove obviously invalid/outlier training rows (nonpositive/huge fares, invalid passenger_count, and coordinates outside a NYC bounding box) and prevent negative predictions by clipping at 0. Finally, I ensure a valid `submission.csv` with the required `key,fare_amount` columns is written.'
- What this solution (achieved 6.34355) has done: 'To move RMSE down toward your target (lower is better) without changing the model or feature set, I make three minimal, metric-aligned adjustments: (1) train on `log1p(fare_amount)` and invert with `expm1` at prediction time (keeps the same RandomForest/core pipeline but reduces the impact of heavy-tail fares on RMSE), (2) add a single simple feature `abs_hour_delta` (difference between pickup hour and a fixed reference) is NOT allowed since it changes feature extraction—so instead I only use the already-computed `minute` feature in both train and test to slightly improve time resolution while preserving the same datetime extraction logic, and (3) clip predictions to a reasonable upper bound (same semantics as your existing non-negative clip, but prevents rare huge errors). These are small, standard changes that typically reduce NYC Taxi Fare RMSE from the ~7 range toward ~4–5 on a 1M sample without altering the training approach. The script still run end-to-end and write a valid `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 6.35654) has done: 'To move your RMSE down toward 3.87285 (lower is better) with minimal changes and without altering the model/feature set, I (1) train with a proper holdout split and report a true validation RMSE (your current “rmse” is in-sample and misleading), (2) increase `n_estimators` moderately to reduce RandomForest variance (same model/approach, typically a clear RMSE gain on this task), and (3) slightly tighten the training cleaning by removing very short “non-trips” and extreme distances/fares that disproportionately hurt RMSE. I keep the exact same features (`distance, year, month, day, hour, minute`) and the same log1p target transform, and still clip predictions to the same bounds for submission stability. The script still run end-to-end within the 600s budget and write a valid `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 6.32221) has done: 'You’re currently worse than the target (RMSE 6.35654 vs 3.87285), so we should make small, metric-aligned improvements without changing the core RandomForest + haversine + datetime-feature approach. The biggest minimal win here is to fix a train/test mismatch: you clean train coordinates to an NYC bounding box but don’t apply the same treatment to test, so out-of-bounds test rows get nonsensical distances and hurt RMSE; we clip test coordinates into the same bbox and recompute distance. Next, we retrain the same model on all cleaned training data after the holdout RMSE check (so the final submission uses maximum data), which usually improves leaderboard RMSE a bit with no semantic changes. Finally, we ensure the distance used for prediction is consistent (recomputed after clipping) and keep your existing log1p transform and prediction clipping unchanged.'
- What this solution (achieved 6.45281) has done: 'Your current RMSE (6.32221) is still well above the target (3.87285), so we should make a small, metric-aligned improvement without changing the overall RandomForest + haversine + datetime-features approach. The biggest minimal win here is to add two standard, “same-feature-family” distance features derived from the same coordinates you already use: absolute lon/lat deltas and a simple Manhattan distance proxy; these usually reduce NYC Taxi Fare RMSE substantially while keeping the exact same modeling approach and training loop. I keep your existing cleaning, log1p target, clipping, and submission writing, and only expand the feature matrix consistently for train and test. This should move the score meaningfully toward the target while staying within runtime constraints.'
- What this solution (achieved 6.48096) has done: 'Your current RMSE (6.45281) is still far above the target (3.87285), so we should make small, metric-aligned improvements without changing the overall RandomForest + distance + datetime-features approach. The biggest minimal win is to fix an inconsistent/weak “Manhattan distance” feature: the current `(abs_lon_diff + abs_lat_diff)*111` is geometrically wrong because longitude degrees scale by `cos(latitude)`, which can inject noise and hurt RMSE. We replace it with a properly scaled Manhattan-km computed from the same existing coordinate deltas (no new raw inputs), and use the same computation for both train and test. We keep your cleaning, log1p target, clipping, train/valid split, final refit on full data, and submission writing unchanged.'
- What this solution (achieved 6.55449) has done: 'We’re still far above the target RMSE (6.48096 vs 3.87285, lower is better), so the smallest legitimate way to move toward the target without changing the model/loop is to fix a metric-misalignment bug: we currently clip predictions at 250, but we also *remove* training fares >200, which biases the model low and can create large squared errors on higher-fare test rides. I align the training cleaning and prediction clipping by allowing fares up to the same `PRED_MAX` and slightly relaxing the distance upper bound accordingly, keeping all features, the log1p transform, and the RandomForest training approach unchanged. This typically reduces RMSE materially on this competition because it reduces systematic underprediction on longer trips while preserving your existing outlier handling. The script still run end-to-end and write a valid `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 6.55449) has done: 'Your current RMSE (6.55449) is still much worse than the target (3.87285, lower is better), so we should make small, metric-aligned improvements without changing the overall RandomForest + distance/time-features approach. The biggest minimal win here is fixing a feature-consistency bug in your plotting cell that reflects a real modeling issue: in training you use `manhattan_km` as a *different* quantity than `distance`, but in the illustrative line you incorrectly set the second feature equal to `distance` (and also set deltas to zero), which indicates we’re not being consistent about how the model expects these correlated distance features to relate. I centralize feature engineering into one shared function used for both train and test (same exact features, just computed consistently), and I also switch the model’s split criterion to `squared_error` (correct for regression in sklearn>=1.0) to avoid any fallback/compat quirks that can degrade accuracy. These are minimal changes that keep the core logic identical (same model family, same features, same log1p transform, same cleaning philosophy) but usually reduce RMSE by making the feature pipeline fully consistent and deterministic end-to-end.'
- What this solution (achieved 6.72069) has done: 'We make two minimal, score-relevant fixes while keeping your RandomForest + (distance/time) feature approach unchanged. First, we stop clipping test coordinates into the NYC bbox (that can severely distort true long trips in the test set and inflate RMSE) and instead only *filter* clearly invalid coordinates in training, leaving test coordinates as-is. Second, we add two very standard “same inputs, same logic” features derived from your existing coordinates (`bearing` and `haversine_sq`) which typically help RandomForest model nonlinearity without changing the learning setup. Everything else (log1p target, model family/params, training loop, submission writing) stays the same, and the script still write a valid `submission.csv`.'
- What this solution (achieved 5.033) has done: 'Your current RMSE (6.72069) is still far above the target (3.87285), so we should make small, metric-aligned improvements without changing the overall RandomForest + coordinate/time feature approach. The biggest low-risk win is to stop *discarding* all training rows outside the tight NYC bbox (which removes many legitimate airport/outer-borough trips and biases the model), and instead keep them while filtering only truly invalid coordinates; this usually reduces systematic underprediction and big squared errors. Second, we bring test-time robustness back in a minimal way by imputing any invalid/out-of-range test coordinates with training medians (rather than leaving them to produce nonsensical distances), which reduces rare huge errors without changing the feature set. Everything else (features, log1p target, RandomForest training loop/params, prediction clipping, and submission writing) stays the same.'
- What this solution (achieved 6.61272) has done: 'Your current RMSE (5.033) is still above the target (3.87285, lower is better), so the smallest likely win is to tighten *label/feature quality* without changing your model or feature set. I keep the same RandomForest + log1p target + existing engineered features, but make the training cleaning less “mean-based” (which can accidentally keep bad GPS points if the sample mean is skewed) by replacing it with standard, domain-based NYC/nearby coordinate sanity bounds and a more realistic distance cap. I also add one minimal, metric-aligned postprocess: clip *training* labels to the same [PRED_MIN, PRED_MAX] used at inference so the model doesn’t learn to chase impossible targets that be clipped anyway. Everything else (paths, features, model params, train/valid split, submission writing) stays the same and still produces `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

TRAIN_PATH = "/kaggle/input/train.csv"
TEST_PATH = "/kaggle/input/test.csv"
SAMPLE_SUB_PATH = "/kaggle/input/sample_submission.csv"

PRED_MIN = 0.0
PRED_MAX = 250.0

RANDOM_STATE = 42

nyc_bbox = {
    "lon_min": -74.3,
    "lon_max": -73.7,
    "lat_min": 40.5,
    "lat_max": 41.0,
}



## === cell 1
train_df = pd.read_csv(TRAIN_PATH, nrows=1_000_000)




## === cell 2
def haversine_np(lon1, lat1, lon2, lat2):
    """
    Fast haversine distance in km. All args must be of equal length.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


def add_features(df, nyc_bbox=None, clip_to_bbox=False, train_medians=None):
    df = df.copy()

    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df["year"] = df["pickup_datetime"].dt.year
    df["month"] = df["pickup_datetime"].dt.month
    df["day"] = df["pickup_datetime"].dt.day
    df["hour"] = df["pickup_datetime"].dt.hour
    df["minute"] = df["pickup_datetime"].dt.minute

    if clip_to_bbox and (nyc_bbox is not None):
        for col, lo, hi in [
            ("pickup_longitude", nyc_bbox["lon_min"], nyc_bbox["lon_max"]),
            ("dropoff_longitude", nyc_bbox["lon_min"], nyc_bbox["lon_max"]),
            ("pickup_latitude", nyc_bbox["lat_min"], nyc_bbox["lat_max"]),
            ("dropoff_latitude", nyc_bbox["lat_min"], nyc_bbox["lat_max"]),
        ]:
            df[col] = df[col].clip(lo, hi)

    df["distance"] = haversine_np(
        df["pickup_longitude"],
        df["pickup_latitude"],
        df["dropoff_longitude"],
        df["dropoff_latitude"],
    )
    df["abs_lon_diff"] = (df["pickup_longitude"] - df["dropoff_longitude"]).abs()
    df["abs_lat_diff"] = (df["pickup_latitude"] - df["dropoff_latitude"]).abs()

    mid_lat_rad = np.radians(
        (df["pickup_latitude"].values + df["dropoff_latitude"].values) / 2.0
    )
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(mid_lat_rad)
    df["manhattan_km"] = (
        df["abs_lat_diff"].values * km_per_deg_lat
        + df["abs_lon_diff"].values * km_per_deg_lon
    )

    lon1 = np.radians(df["pickup_longitude"].astype(float).values)
    lat1 = np.radians(df["pickup_latitude"].astype(float).values)
    lon2 = np.radians(df["dropoff_longitude"].astype(float).values)
    lat2 = np.radians(df["dropoff_latitude"].astype(float).values)

    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    df["bearing"] = np.arctan2(y, x)  # radians in [-pi, pi]
    df["haversine_sq"] = df["distance"].astype(float).values ** 2

    if train_medians is not None:
        df = df.fillna(
            {
                "pickup_longitude": train_medians.get("pickup_longitude", 0.0),
                "pickup_latitude": train_medians.get("pickup_latitude", 0.0),
                "dropoff_longitude": train_medians.get("dropoff_longitude", 0.0),
                "dropoff_latitude": train_medians.get("dropoff_latitude", 0.0),
                "year": train_medians["year"],
                "month": train_medians["month"],
                "day": train_medians["day"],
                "hour": train_medians["hour"],
                "minute": train_medians["minute"],
                "distance": 0.0,
                "abs_lon_diff": 0.0,
                "abs_lat_diff": 0.0,
                "manhattan_km": 0.0,
                "bearing": 0.0,
                "haversine_sq": 0.0,
            }
        )

    return df




## === cell 3
train_df = add_features(train_df, nyc_bbox=nyc_bbox, clip_to_bbox=False)



## === cell 4
print("Old size: %d" % len(train_df))
train_df = train_df.dropna(how="any", axis="rows")
print("New size: %d" % len(train_df))



## === cell 5
new_york_lat = 40
new_york_long = -74
train_df.describe()



## === cell 6
cond = np.ones(len(train_df), dtype=bool)

cond &= (train_df["pickup_longitude"].values >= -180.0) & (
    train_df["pickup_longitude"].values <= 180.0
)
cond &= (train_df["dropoff_longitude"].values >= -180.0) & (
    train_df["dropoff_longitude"].values <= 180.0
)
cond &= (train_df["pickup_latitude"].values >= -90.0) & (
    train_df["pickup_latitude"].values <= 90.0
)
cond &= (train_df["dropoff_latitude"].values >= -90.0) & (
    train_df["dropoff_latitude"].values <= 90.0
)

NYC_WIDE = {"lon_min": -75.5, "lon_max": -72.5, "lat_min": 39.0, "lat_max": 42.5}
cond &= (train_df["pickup_longitude"].values >= NYC_WIDE["lon_min"]) & (
    train_df["pickup_longitude"].values <= NYC_WIDE["lon_max"]
)
cond &= (train_df["dropoff_longitude"].values >= NYC_WIDE["lon_min"]) & (
    train_df["dropoff_longitude"].values <= NYC_WIDE["lon_max"]
)
cond &= (train_df["pickup_latitude"].values >= NYC_WIDE["lat_min"]) & (
    train_df["pickup_latitude"].values <= NYC_WIDE["lat_max"]
)
cond &= (train_df["dropoff_latitude"].values >= NYC_WIDE["lat_min"]) & (
    train_df["dropoff_latitude"].values <= NYC_WIDE["lat_max"]
)

cond &= (train_df["fare_amount"].values > 0.0) & (
    train_df["fare_amount"].values < PRED_MAX
)
cond &= (train_df["passenger_count"].values >= 1) & (
    train_df["passenger_count"].values <= 6
)

cond &= (train_df["distance"].values > 0.05) & (train_df["distance"].values < 80.0)

print("Old size: %d" % len(train_df))
train_df = train_df[cond].copy()
print("New size: %d" % len(train_df))



## === cell 7
train_df.describe()



## === cell 8
train_df["fare_amount"] = train_df["fare_amount"].clip(PRED_MIN, PRED_MAX)

X = train_df[
    [
        "distance",
        "manhattan_km",
        "abs_lon_diff",
        "abs_lat_diff",
        "bearing",
        "haversine_sq",
        "year",
        "month",
        "day",
        "hour",
        "minute",
    ]
].values
Y = np.log1p(train_df["fare_amount"].values)



## === cell 9
from sklearn.ensemble import RandomForestRegressor



## === cell 10
kwargs = {
    "bootstrap": True,
    "max_depth": None,
    "max_features": 3,
    "min_samples_leaf": 9,
    "min_samples_split": 2,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "criterion": "squared_error",
}
rand_regr = RandomForestRegressor(n_estimators=80, **kwargs)



## === cell 11
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(
    X, Y, test_size=0.2, random_state=RANDOM_STATE
)

rand_regr.fit(X_train, y_train)

y_valid_pred_log = rand_regr.predict(X_valid)
y_valid_pred = np.expm1(y_valid_pred_log)
y_valid_pred = np.clip(y_valid_pred, PRED_MIN, PRED_MAX)
y_valid_true = np.expm1(y_valid)

valid_rmse = np.sqrt(np.mean((y_valid_true - y_valid_pred) ** 2.0))
print("Validation RMSE (holdout):", valid_rmse)



## === cell 12
rand_regr.score(X_valid, y_valid)



## === cell 13
rand_regr.fit(X, Y)



## === cell 14
test_df = pd.read_csv(TEST_PATH)



## === cell 15
train_medians = {
    "year": train_df["year"].median(),
    "month": train_df["month"].median(),
    "day": train_df["day"].median(),
    "hour": train_df["hour"].median(),
    "minute": train_df["minute"].median(),
    "pickup_longitude": train_df["pickup_longitude"].median(),
    "pickup_latitude": train_df["pickup_latitude"].median(),
    "dropoff_longitude": train_df["dropoff_longitude"].median(),
    "dropoff_latitude": train_df["dropoff_latitude"].median(),
}

for c, lo, hi in [
    ("pickup_longitude", -180.0, 180.0),
    ("dropoff_longitude", -180.0, 180.0),
    ("pickup_latitude", -90.0, 90.0),
    ("dropoff_latitude", -90.0, 90.0),
]:
    v = test_df[c].astype(float)
    test_df.loc[(v < lo) | (v > hi), c] = np.nan

test_df = add_features(
    test_df, nyc_bbox=nyc_bbox, clip_to_bbox=False, train_medians=train_medians
)

X_to_pred = test_df[
    [
        "distance",
        "manhattan_km",
        "abs_lon_diff",
        "abs_lat_diff",
        "bearing",
        "haversine_sq",
        "year",
        "month",
        "day",
        "hour",
        "minute",
    ]
].values

y_pred_test_log = rand_regr.predict(X_to_pred)
y_pred_test = np.expm1(y_pred_test_log)
y_pred_test = np.clip(y_pred_test, PRED_MIN, PRED_MAX)



## === cell 16
submission = pd.DataFrame(
    {"key": test_df["key"], "fare_amount": y_pred_test}, columns=["key", "fare_amount"]
)
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)



## === cell 17
X_dist = train_df["distance"].values
Y_fare = np.expm1(Y)  # plot in original fare units for interpretability
mask = np.isfinite(X_dist) & np.isfinite(Y_fare)

with sns.axes_style("white"):
    sns.jointplot(x=X_dist[mask], y=Y_fare[mask], kind="hex", color="k", bins="log")



## === cell 18
X_dist = train_df["distance"].values
Y_fare = np.expm1(Y)
mask = (X_dist < 50) & (Y_fare < 100) & np.isfinite(X_dist) & np.isfinite(Y_fare)

with sns.axes_style("white"):
    p = sns.jointplot(x=X_dist[mask], y=Y_fare[mask], kind="hex", color="k", bins="log")

x_line = np.linspace(0, 50, 200)
year_med = int(train_df["year"].median())
month_med = int(train_df["month"].median())
day_med = int(train_df["day"].median())
hour_med = int(train_df["hour"].median())
minute_med = int(train_df["minute"].median())

abs_lat = np.zeros_like(x_line)
abs_lon = np.zeros_like(x_line)
manhattan = x_line.copy()
bearing_line = np.zeros_like(x_line)
haversine_sq_line = x_line**2

X_line = np.column_stack(
    [
        x_line,  # distance
        manhattan,  # manhattan_km (illustrative)
        abs_lon,  # abs_lon_diff
        abs_lat,  # abs_lat_diff
        bearing_line,  # bearing
        haversine_sq_line,  # haversine_sq
        np.full_like(x_line, year_med),
        np.full_like(x_line, month_med),
        np.full_like(x_line, day_med),
        np.full_like(x_line, hour_med),
        np.full_like(x_line, minute_med),
    ]
)
y_line_log = rand_regr.predict(X_line)
y_line = np.expm1(y_line_log)
y_line = np.clip(y_line, PRED_MIN, PRED_MAX)

p.ax_joint.plot(x_line, y_line, color="red", linewidth=2)
plt.show()
