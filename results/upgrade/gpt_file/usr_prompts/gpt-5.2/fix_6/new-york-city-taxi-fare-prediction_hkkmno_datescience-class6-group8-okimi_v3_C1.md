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

8.19089

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.53705) has done: 'I make the notebook produce a valid `submission.csv` end-to-end (your current code stops after profiling and never trains/predicts). To move RMSE down toward your target while keeping the same core modeling approach (KNN regression), I (1) train KNN on a manageable but stronger training subset, (2) add a minimal, competition-standard geodesic distance feature (plus keep passenger_count) so KNN has a meaningful signal, and (3) apply basic, safe row filtering on impossible coordinates/outliers to reduce noise without changing the overall approach. I also ensure the submission uses the exact `key,fare_amount` schema and matches test row order.'
- What this solution (achieved 5.33099) has done: 'I keep your KNN approach and overall pipeline intact, but adjust it in two minimal, competition-standard ways that usually reduce RMSE materially: (1) add a few more time/location-derived features (still purely from given inputs) so KNN has more signal than just distance + hour/dow, and (2) make scaling more robust by using a `RobustScaler` (less sensitive to long-tail distance/fare patterns that can distort neighbor geometry). I also tune `n_neighbors` slightly (still KNN with distance weights) because your current 25 neighbors often over-smooths on this dataset and inflates RMSE. Everything still runs end-to-end, trains once, predicts once, and writes `submission.csv` with the required `key,fare_amount` schema.'
- What this solution (achieved 5.06877) has done: 'Your current RMSE (5.33099) is worse than the target (3.53576), so we should improve the model slightly while keeping the same KNN core. The biggest safe gain with minimal semantic change is to add two competition-standard geospatial features (pickup/dropoff “center” lat/lon) and use latitude-aware scaling of lon/lat deltas (cosine latitude) so KNN distances better reflect real geometry. I also keep your existing RobustScaler + KNN settings, but add a tiny, safe extra cleaning step to drop clearly bad “zero-distance but non-trivial fare” artifacts that otherwise confuse neighbor geometry. Everything still trains once, predicts once, and writes a valid `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 5.11484) has done: 'To move your RMSE down toward the 3.53576 target while keeping the same KNN + RobustScaler core, I make two minimal, competition-standard upgrades: (1) augment your existing geospatial feature set with a simple Manhattan-distance proxy and a couple of airport-distance features (JFK/LGA/EWR) that are highly predictive for NYC fares, and (2) fit the scaler on the combined train+test feature matrix so KNN distances are computed in a consistent space between train/test (this doesn’t change the model type or training loop). I also add a very small, safe cleanup rule to remove unrealistic extreme fares-per-km artifacts that tend to confuse nearest-neighbor geometry. The rest of your pipeline (reading subset, KNN settings, single fit/predict, submission merge) remains unchanged.'
- What this solution (achieved 8.19089) has done: 'Your current RMSE (5.11484) is still well above the target (3.53576), so we should improve it with the smallest safe changes that keep your KNN+RobustScaler core intact. The biggest likely gain with minimal semantic impact is to correct feature scaling: fitting the scaler on train+test is non-standard and can distort neighbor geometry; we fit the scaler on train only so distances reflect the training distribution. Then, because airport features help but can dominate, we slightly rebalance feature contributions by scaling distances vs. time/passenger features using a simple per-feature multiplier (still the same KNN on engineered features). Finally, we make the cleaning a touch more consistent with common NYC Taxi kernels by removing a few extreme geographic outliers (still “obvious bad rows”), which typically reduces RMSE without changing the modeling approach.'

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

    d["center_lon"] = (
        d["pickup_longitude"].astype("float32")
        + d["dropoff_longitude"].astype("float32")
    ) / 2.0
    d["center_lat"] = (
        d["pickup_latitude"].astype("float32") + d["dropoff_latitude"].astype("float32")
    ) / 2.0

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

    return d


train_f = add_features(train)
test_f = add_features(test)

train_f[
    [
        "distance_km",
        "manhattan_km",
        "abs_lon_diff_scaled",
        "abs_lat_diff",
        "center_lon",
        "center_lat",
        "pickup_hour",
        "pickup_dow",
        "pickup_month",
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
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
            "passenger_count",
            "distance_km",
            "manhattan_km",
            "abs_lon_diff_scaled",
            "abs_lat_diff",
            "center_lon",
            "center_lat",
            "pickup_hour",
            "pickup_dow",
            "pickup_month",
            "pickup_to_jfk_km",
            "dropoff_to_jfk_km",
            "pickup_to_lga_km",
            "dropoff_to_lga_km",
            "pickup_to_ewr_km",
            "dropoff_to_ewr_km",
        ]
    )

    d = d[
        (d["pickup_longitude"].between(-74.5, -72.8))
        & (d["dropoff_longitude"].between(-74.5, -72.8))
        & (d["pickup_latitude"].between(40.0, 41.8))
        & (d["dropoff_latitude"].between(40.0, 41.8))
    ]

    d = d[
        (d["center_lon"].between(-74.3, -73.6))
        & (d["center_lat"].between(40.45, 40.95))
    ]

    d = d[(d["passenger_count"] >= 1) & (d["passenger_count"] <= 6)]
    d = d[(d["fare_amount"] > 0) & (d["fare_amount"] < 250)]
    d = d[(d["distance_km"] >= 0) & (d["distance_km"] < 200)]
    d = d[(d["manhattan_km"] >= 0) & (d["manhattan_km"] < 250)]

    d = d[~((d["distance_km"] < 0.05) & (d["fare_amount"] > 25.0))]

    per_km = d["fare_amount"] / (d["distance_km"] + 0.1)
    d = d[per_km < 80.0]

    return d


train_c = clean_train(train_f)

train_c.shape, train_f.shape



## === cell 4
FEATURES = [
    "distance_km",
    "manhattan_km",
    "abs_lon_diff_scaled",
    "abs_lat_diff",
    "center_lon",
    "center_lat",
    "passenger_count",
    "pickup_hour",
    "pickup_dow",
    "pickup_month",
    "pickup_to_jfk_km",
    "dropoff_to_jfk_km",
    "pickup_to_lga_km",
    "dropoff_to_lga_km",
    "pickup_to_ewr_km",
    "dropoff_to_ewr_km",
]

X = train_c[FEATURES].astype("float32").values
y = train_c["fare_amount"].astype("float32").values
X_test = test_f[FEATURES].astype("float32").values

scaler = RobustScaler(
    with_centering=True, with_scaling=True, quantile_range=(10.0, 90.0)
)
scaler.fit(X)
Xz = scaler.transform(X)
X_testz = scaler.transform(X_test)

feat_names = FEATURES
mult = np.ones(len(feat_names), dtype="float32")
distance_like = {
    "distance_km",
    "manhattan_km",
    "abs_lon_diff_scaled",
    "abs_lat_diff",
    "pickup_to_jfk_km",
    "dropoff_to_jfk_km",
    "pickup_to_lga_km",
    "dropoff_to_lga_km",
    "pickup_to_ewr_km",
    "dropoff_to_ewr_km",
}
time_like = {"pickup_hour", "pickup_dow", "pickup_month"}
coord_like = {"center_lon", "center_lat"}
pass_like = {"passenger_count"}

for i, f in enumerate(feat_names):
    if f in distance_like:
        mult[i] = 1.25
    elif f in coord_like:
        mult[i] = 0.85
    elif f in time_like:
        mult[i] = 0.75
    elif f in pass_like:
        mult[i] = 0.90

Xz = Xz * mult
X_testz = X_testz * mult

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
