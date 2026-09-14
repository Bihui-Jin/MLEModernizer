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
pyproj==3.7.1
pyproject_hooks==1.2.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
seaborn==0.12.2
sklearn-pandas==2.2.0
ydata-profiling==4.17.0

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

3.53576

# 6. Current score

9.25457

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.53705) has done: 'I make the notebook produce a valid `submission.csv` end-to-end (your current code stops after profiling and never trains/predicts). To move RMSE down toward your target while keeping the same core modeling approach (KNN regression), I (1) train KNN on a manageable but stronger training subset, (2) add a minimal, competition-standard geodesic distance feature (plus keep passenger_count) so KNN has a meaningful signal, and (3) apply basic, safe row filtering on impossible coordinates/outliers to reduce noise without changing the overall approach. I also ensure the submission uses the exact `key,fare_amount` schema and matches test row order.'
- What this solution (achieved 5.33099) has done: 'I keep your KNN approach and overall pipeline intact, but adjust it in two minimal, competition-standard ways that usually reduce RMSE materially: (1) add a few more time/location-derived features (still purely from given inputs) so KNN has more signal than just distance + hour/dow, and (2) make scaling more robust by using a `RobustScaler` (less sensitive to long-tail distance/fare patterns that can distort neighbor geometry). I also tune `n_neighbors` slightly (still KNN with distance weights) because your current 25 neighbors often over-smooths on this dataset and inflates RMSE. Everything still runs end-to-end, trains once, predicts once, and writes `submission.csv` with the required `key,fare_amount` schema.'
- What this solution (achieved 5.06877) has done: 'Your current RMSE (5.33099) is worse than the target (3.53576), so we should improve the model slightly while keeping the same KNN core. The biggest safe gain with minimal semantic change is to add two competition-standard geospatial features (pickup/dropoff “center” lat/lon) and use latitude-aware scaling of lon/lat deltas (cosine latitude) so KNN distances better reflect real geometry. I also keep your existing RobustScaler + KNN settings, but add a tiny, safe extra cleaning step to drop clearly bad “zero-distance but non-trivial fare” artifacts that otherwise confuse neighbor geometry. Everything still trains once, predicts once, and writes a valid `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 5.11484) has done: 'To move your RMSE down toward the 3.53576 target while keeping the same KNN + RobustScaler core, I make two minimal, competition-standard upgrades: (1) augment your existing geospatial feature set with a simple Manhattan-distance proxy and a couple of airport-distance features (JFK/LGA/EWR) that are highly predictive for NYC fares, and (2) fit the scaler on the combined train+test feature matrix so KNN distances are computed in a consistent space between train/test (this doesn’t change the model type or training loop). I also add a very small, safe cleanup rule to remove unrealistic extreme fares-per-km artifacts that tend to confuse nearest-neighbor geometry. The rest of your pipeline (reading subset, KNN settings, single fit/predict, submission merge) remains unchanged.'
- What this solution (achieved 8.19089) has done: 'Your current RMSE (5.11484) is still well above the target (3.53576), so we should improve it with the smallest safe changes that keep your KNN+RobustScaler core intact. The biggest likely gain with minimal semantic impact is to correct feature scaling: fitting the scaler on train+test is non-standard and can distort neighbor geometry; we fit the scaler on train only so distances reflect the training distribution. Then, because airport features help but can dominate, we slightly rebalance feature contributions by scaling distances vs. time/passenger features using a simple per-feature multiplier (still the same KNN on engineered features). Finally, we make the cleaning a touch more consistent with common NYC Taxi kernels by removing a few extreme geographic outliers (still “obvious bad rows”), which typically reduces RMSE without changing the modeling approach.'
- What this solution (achieved 6.64988) has done: 'Your current RMSE (8.19089) is far worse than the target (3.53576), and the jump from ~5.11 to 8.19 strongly suggests a regression caused by the last “feature multiplier rebalancing” step rather than the underlying KNN approach. To move back toward the target with minimal change and identical core logic, I remove the per-feature multipliers so all engineered features contribute in the scaled space as intended by the RobustScaler. I also slightly tighten the obviously-bad-row cleanup (still only removing extreme geographic outliers) to reduce noise without changing the model or training loop. Everything else (same features, RobustScaler-on-train, KNN settings, submission schema) stays the same.'
- What this solution (achieved 6.53028) has done: 'Your current RMSE (6.64988) is much worse than the target (3.53576), and given the prior history (you were ~5.1 with essentially the same pipeline), the most likely cause is that the KNN neighbor geometry is being polluted by a small set of still-bad training rows and a slightly-too-local neighbor setting for this noisy regression. I keep the exact same core approach (feature engineering → RobustScaler → KNeighborsRegressor) and make only two minimal, competition-standard adjustments: (1) strengthen the cleaning with a tight but safe NYC-taxi rule that removes unrealistic “too-fast” trips (a known major noise source), and (2) slightly increase `n_neighbors` to reduce variance/sensitivity to remaining outliers while keeping `weights="distance"`. These changes are directly aimed at reducing RMSE without changing the model class, training loop, or feature set, and the script still write a valid `submission.csv` with `key,fare_amount` aligned to the sample submission order.'
- What this solution (achieved 6.65471) has done: 'Your current RMSE (6.53028) is much worse than the target (3.53576), so we should improve it with minimal, model-preserving changes. The biggest likely regression is the `RobustScaler(quantile_range=(10,90))`, which can shrink important tails and distort KNN neighbor geometry on this dataset; switching back to the standard IQR `(25,75)` is a minimal change that often helps RMSE without changing the approach. I also remove one overly-aggressive cleaning rule (`per_km < 80`) that can discard legitimate high-fare airport trips and harm generalization, while keeping the rest of the “obvious bad rows” filters intact. Finally, I keep the same KNN model/fit/predict flow and submission schema, but tune `n_neighbors` slightly (25→15) to reduce oversmoothing that tends to inflate RMSE in this competition.'
- What this solution (achieved 6.52229) has done: 'We need to move RMSE down from 6.65471 toward 3.53576 (lower is better), so we should improve signal without changing the core KNN+RobustScaler approach. The biggest minimal fix is to add a single, highly predictive feature that your pipeline already almost computes but never uses: `pickup_seconds` (time as a continuous variable), and to compute the same feature for test so train/test feature spaces match. Then we include `pickup_seconds` in `FEATURES` so KNN can distinguish rush hour/seasonality patterns better than hour/dow/month alone, which typically reduces RMSE materially on this competition while preserving your exact modeling loop and loss semantics. Everything else (feature engineering, scaler, KNN settings, cleaning, submission writing) stays intact.'
- What this solution (achieved 6.52229) has done: 'Your current RMSE (6.52229) is far above the target (3.53576), so we should improve generalization without changing the core KNN+RobustScaler approach. The biggest minimal win for NYC taxi fares is removing remaining label/feature noise that breaks nearest-neighbor geometry: specifically, filter out implausible “too-fast” trips using pickup time and distance (a standard cleanup for this competition). To keep semantics identical otherwise, I leave your feature set, scaler, and KNN configuration intact, and only add a derived `speed_kmh` used solely for cleaning (not as a model feature). This should move RMSE down materially while preserving the same training loop and submission format.'
- What this solution (achieved 6.60468) has done: 'Your current RMSE (6.52229) is far above the target (3.53576), so we should improve it with the smallest changes that keep your KNN+RobustScaler pipeline intact. The biggest low-risk issue is that `pickup_seconds` is an enormous “epoch time” feature that tends to dominate neighbor geometry even after robust scaling; replacing it with a cyclic within-day time representation (sin/cos of seconds-of-day) preserves the same idea (time signal) but makes KNN distances meaningful. I also add two tiny, competition-standard location features (`pickup_lon/lat` and `dropoff_lon/lat`) because KNN needs absolute position to learn borough/airport/toll patterns beyond just deltas/distances. Everything else (data reading, cleaning approach, scaler, KNN config, submission schema) stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved 7.16895) has done: 'Your current RMSE (6.60468) is much worse than the target (3.53576), so we should improve performance with the smallest changes that preserve your KNN+RobustScaler pipeline. The biggest likely source of error is that pure KNN distance in the engineered feature space still doesn’t match fare behavior well for long/airport trips; adding a single, standard “fare baseline” feature (estimated meter-like fare from distance + passenger) gives KNN a much more meaningful geometry without changing the model type or training loop. I also add one minimal, standard cleanup that removes negative/zero fares and extreme per-km anomalies more safely (focused on obvious noise), which reduces label noise that KNN is very sensitive to. Everything else (data loading, feature engineering flow, RobustScaler-on-train, KNN fit/predict, and submission schema) remains intact and still writes `submission.csv`.'
- What this solution (achieved 7.16895) has done: 'Your current RMSE (7.16895) is far above the target (3.53576), so we should improve with the smallest changes that keep your KNN+RobustScaler pipeline intact. The biggest likely regression is the “trip duration from key vs pickup_datetime” cleaning: `key` is not a trip-end timestamp, so this filter removes valid rows in a biased way and harms KNN neighbor geometry. I remove that invalid duration/speed cleaning block entirely (keeping the rest of your existing “obvious bad rows” filters unchanged) and, to stabilize neighbor geometry, I also ensure train/test use consistent numeric feature matrices by filling any remaining NaNs in engineered features before scaling. Everything else (features, scaler, KNN settings, single fit/predict, submission schema/path) remains the same.'
- What this solution (achieved 7.64525) has done: 'Your current RMSE (7.16895) is much worse than the target (3.53576), so we should improve with minimal, KNN-preserving changes. The most likely issue is neighbor geometry being dominated by many correlated/large-scale features (airport distances, absolute lon/lat, etc.), which can hurt KNN badly even with robust scaling; a small, competition-standard fix is to use a more appropriate distance metric (Manhattan/L1) for taxi-like movement while keeping the exact same KNN regressor and feature set. I also make the cleaning slightly less aggressive by widening the valid NYC bounding box a bit (still removing obvious bad rows) so we don’t throw away too much legitimate training data, which can destabilize KNN. Finally, I keep the same submission schema and ensure we always write a valid `submission.csv`.'
- What this solution (achieved 9.25457) has done: 'Your current RMSE (7.64525) is much worse than the target (3.53576), so we should make a small, KNN-preserving change that improves neighbor geometry without changing the overall pipeline. I revert the distance metric from Manhattan/L1 back to the standard Euclidean/L2 in the scaled feature space, because with RobustScaler and many correlated engineered features, L1 often degrades KNN regression on this competition. I also tighten one obviously-noisy filter that KNN is very sensitive to (removing extreme per-km outliers a bit more aggressively) while keeping the same cleaning approach and thresholds style. Everything else (features, RobustScaler fit on train, single KNN fit/predict, and `submission.csv` schema) stays the same.'

# 9. Code solution

## === cell 0
import os
import warnings

import numpy as np
import pandas as pd
from pyproj import Geod

from sklearn.neighbors import KNeighborsRegressor as KN_R
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")
pd.set_option("display.float_format", lambda x: "%.4f" % x)

TRAIN_PATH = "../input/new-york-city-taxi-fare-prediction/train.csv"
TEST_PATH = "../input/new-york-city-taxi-fare-prediction/test.csv"
SAMPLE_SUB_PATH = "../input/new-york-city-taxi-fare-prediction/sample_submission.csv"

SUBMISSION_PATH = "submission.csv"

RANDOM_STATE = 42

N_TRAIN_ROWS = 2_000_000



## === cell 1
train = pd.read_csv(
    TRAIN_PATH,
    nrows=N_TRAIN_ROWS,
    usecols=[
        "key",
        "fare_amount",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
)

test = pd.read_csv(
    TEST_PATH,
    usecols=[
        "key",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
)

train.shape, test.shape



## === cell 2
geod = Geod(ellps="WGS84")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    dt = pd.to_datetime(d["pickup_datetime"], errors="coerce", utc=True)
    d["pickup_hour"] = dt.dt.hour.astype("float32")
    d["pickup_dow"] = dt.dt.dayofweek.astype("float32")
    d["pickup_month"] = dt.dt.month.astype("float32")

    sec_of_day = (
        (dt.dt.hour.fillna(0).astype("int32") * 3600)
        + (dt.dt.minute.fillna(0).astype("int32") * 60)
        + (dt.dt.second.fillna(0).astype("int32"))
    ).astype("float32")
    d["pickup_sin_time"] = np.sin(2.0 * np.pi * (sec_of_day / 86400.0)).astype(
        "float32"
    )
    d["pickup_cos_time"] = np.cos(2.0 * np.pi * (sec_of_day / 86400.0)).astype(
        "float32"
    )

    d["pickup_longitude"] = pd.to_numeric(
        d["pickup_longitude"], errors="coerce"
    ).astype("float32")
    d["pickup_latitude"] = pd.to_numeric(d["pickup_latitude"], errors="coerce").astype(
        "float32"
    )
    d["dropoff_longitude"] = pd.to_numeric(
        d["dropoff_longitude"], errors="coerce"
    ).astype("float32")
    d["dropoff_latitude"] = pd.to_numeric(
        d["dropoff_latitude"], errors="coerce"
    ).astype("float32")

    _, _, dist_m = geod.inv(
        d["pickup_longitude"].astype(float).values,
        d["pickup_latitude"].astype(float).values,
        d["dropoff_longitude"].astype(float).values,
        d["dropoff_latitude"].astype(float).values,
    )
    d["distance_km"] = (dist_m / 1000.0).astype("float32")

    mean_lat_rad = np.deg2rad(
        (
            (
                d["pickup_latitude"].astype("float64")
                + d["dropoff_latitude"].astype("float64")
            )
            / 2.0
        ).values
    )
    cos_mean_lat = np.cos(mean_lat_rad).astype("float32")
    d["abs_lon_diff_scaled"] = (
        d["dropoff_longitude"] - d["pickup_longitude"]
    ).abs().astype("float32") * cos_mean_lat
    d["abs_lat_diff"] = (
        (d["dropoff_latitude"] - d["pickup_latitude"]).abs().astype("float32")
    )

    d["center_lon"] = (d["pickup_longitude"] + d["dropoff_longitude"]) / 2.0
    d["center_lat"] = (d["pickup_latitude"] + d["dropoff_latitude"]) / 2.0

    d["manhattan_km"] = (
        (d["abs_lat_diff"].astype("float32") * 111.0)
        + (d["abs_lon_diff_scaled"].astype("float32") * 111.0)
    ).astype("float32")

    airports = {
        "jfk": (-73.7781, 40.6413),
        "lga": (-73.8740, 40.7769),
        "ewr": (-74.1745, 40.6895),
    }

    for code, (alon, alat) in airports.items():
        _, _, dm_pu = geod.inv(
            d["pickup_longitude"].astype(float).values,
            d["pickup_latitude"].astype(float).values,
            np.full(len(d), alon, dtype="float64"),
            np.full(len(d), alat, dtype="float64"),
        )
        _, _, dm_do = geod.inv(
            d["dropoff_longitude"].astype(float).values,
            d["dropoff_latitude"].astype(float).values,
            np.full(len(d), alon, dtype="float64"),
            np.full(len(d), alat, dtype="float64"),
        )
        d[f"pickup_to_{code}_km"] = (dm_pu / 1000.0).astype("float32")
        d[f"dropoff_to_{code}_km"] = (dm_do / 1000.0).astype("float32")

    d["passenger_count"] = pd.to_numeric(d["passenger_count"], errors="coerce").astype(
        "float32"
    )

    base_fare = 2.5
    per_km = 1.55  # ~ $/km proxy
    per_pass = 0.5
    d["fare_est"] = (
        base_fare
        + per_km * d["manhattan_km"].astype("float32")
        + per_pass * np.clip(d["passenger_count"].astype("float32") - 1.0, 0.0, 5.0)
    ).astype("float32")

    return d


train_f = add_features(train)
test_f = add_features(test)

train_f[
    [
        "distance_km",
        "manhattan_km",
        "fare_est",
        "abs_lon_diff_scaled",
        "abs_lat_diff",
        "center_lon",
        "center_lat",
        "pickup_hour",
        "pickup_dow",
        "pickup_month",
        "pickup_sin_time",
        "pickup_cos_time",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
        "pickup_to_jfk_km",
        "dropoff_to_jfk_km",
        "pickup_to_lga_km",
        "dropoff_to_lga_km",
        "pickup_to_ewr_km",
        "dropoff_to_ewr_km",
    ]
].describe()




## === cell 3
def clean_train(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    d = d.dropna(
        subset=[
            "fare_amount",
            "pickup_datetime",
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
            "passenger_count",
            "distance_km",
            "manhattan_km",
            "fare_est",
            "abs_lon_diff_scaled",
            "abs_lat_diff",
            "center_lon",
            "center_lat",
            "pickup_hour",
            "pickup_dow",
            "pickup_month",
            "pickup_sin_time",
            "pickup_cos_time",
            "pickup_to_jfk_km",
            "dropoff_to_jfk_km",
            "pickup_to_lga_km",
            "dropoff_to_lga_km",
            "pickup_to_ewr_km",
            "dropoff_to_ewr_km",
        ]
    )

    d = d[
        (d["pickup_longitude"].between(-74.5, -73.5))
        & (d["dropoff_longitude"].between(-74.5, -73.5))
        & (d["pickup_latitude"].between(40.3, 41.05))
        & (d["dropoff_latitude"].between(40.3, 41.05))
    ]

    d = d[
        (d["center_lon"].between(-74.5, -73.5)) & (d["center_lat"].between(40.3, 41.05))
    ]

    d = d[(d["passenger_count"] >= 1) & (d["passenger_count"] <= 6)]
    d = d[(d["fare_amount"] > 0) & (d["fare_amount"] < 250)]
    d = d[(d["distance_km"] >= 0) & (d["distance_km"] < 200)]
    d = d[(d["manhattan_km"] >= 0) & (d["manhattan_km"] < 250)]

    d = d[~((d["distance_km"] < 0.05) & (d["fare_amount"] > 25.0))]
    d = d[~((d["distance_km"] > 50) & (d["fare_amount"] < 30))]

    per_km = d["fare_amount"].astype("float64") / np.maximum(
        d["manhattan_km"].astype("float64"), 0.2
    )
    d = d[(per_km > 0.5) & (per_km < 60.0)]

    return d


train_c = clean_train(train_f)

train_c.shape, train_f.shape



## === cell 4
FEATURES = [
    "distance_km",
    "manhattan_km",
    "fare_est",
    "abs_lon_diff_scaled",
    "abs_lat_diff",
    "center_lon",
    "center_lat",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "pickup_hour",
    "pickup_dow",
    "pickup_month",
    "pickup_sin_time",
    "pickup_cos_time",
    "pickup_to_jfk_km",
    "dropoff_to_jfk_km",
    "pickup_to_lga_km",
    "dropoff_to_lga_km",
    "pickup_to_ewr_km",
    "dropoff_to_ewr_km",
]

med = train_c[FEATURES].median(numeric_only=True)
train_X_df = train_c[FEATURES].fillna(med)
test_X_df = test_f[FEATURES].fillna(med)

X = train_X_df.astype("float32").values
y = train_c["fare_amount"].astype("float32").values
X_test = test_X_df.astype("float32").values

scaler = RobustScaler(
    with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0)
)
scaler.fit(X)
Xz = scaler.transform(X)
X_testz = scaler.transform(X_test)

knn = KN_R(
    n_neighbors=15,
    weights="distance",
    metric="minkowski",
    p=2,
    n_jobs=-1,
)

knn.fit(Xz, y)

pred = knn.predict(X_testz).astype("float32")
pred = np.clip(pred, 0.0, None)

pred[:5], float(pred.mean()), float(pred.min()), float(pred.max())



## === cell 5
sub = pd.DataFrame({"key": test["key"].values, "fare_amount": pred})

if os.path.exists(SAMPLE_SUB_PATH):
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH, usecols=["key"])
    sub = sample_sub.merge(sub, on="key", how="left")
    fallback = float(np.mean(y)) if len(y) else 11.35
    sub["fare_amount"] = sub["fare_amount"].fillna(fallback)

sub.to_csv(SUBMISSION_PATH, index=False)

sub.head(), sub.shape, SUBMISSION_PATH
