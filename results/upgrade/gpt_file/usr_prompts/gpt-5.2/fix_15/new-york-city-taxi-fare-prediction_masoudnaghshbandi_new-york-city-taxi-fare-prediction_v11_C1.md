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

3.10

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

5.10583

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.87043) has done: 'You’re getting a length mismatch because you filter/drop rows in `test` (so predictions are for fewer rows), but then you assign those predictions back to an unfiltered `sample_new` with all rows. I fix this by keeping `key` intact, building predictions only for rows that survive cleaning, and then merging back onto the full test keys so every row gets a `fare_amount` (using the model’s training-mean as a safe fallback for dropped/invalid rows). I also fix the incorrect scaler usage (`fit_transform(X_train, X_test)` doesn’t do what you intend) without changing the core modeling logic (you weren’t actually using scaled features anyway). Finally, the script always write a valid `submission1.csv` with exactly the required columns `key,fare_amount`.'
- What this solution (achieved 4.83917) has done: 'You’re currently underperforming the target (RMSE 4.87 vs 3.78; lower is better), so the smallest safe move is to improve feature signal without changing the model/training loop. I keep the same XGBoost regressor and training approach, but add a minimal, standard NYC-taxi feature: Haversine distance (computed from the same lat/lon columns) for both train and test. I also fix one obvious no-op/typo in the test cleaning block that can accidentally drop/keep rows inconsistently, which can slightly hurt generalization. Everything else (rows loaded, filters, model params, submission writing/merge fallback) stays intact and still writes `submission1.csv`.'
- What this solution (achieved 4.53568) has done: 'You’re below the target (4.839 > 3.7787, lower is better), so we make the smallest legitimate signal improvements without changing the core model/training loop. We keep the same XGBRegressor and same training approach, but (1) add a few standard time-based features extracted from `pickup_datetime` (hour, dayofweek, month) and (2) add a simple Manhattan-distance feature in lat/lon space alongside your existing Haversine distance. We also apply the exact same basic zero/passenger filters to `test_features` as you already do for `train` (previously, train/test cleaning was slightly inconsistent), while keeping the same submission merge+fallback logic to guarantee a valid `submission1.csv`.'
- What this solution (achieved 4.73308) has done: 'You’re currently worse than the target (RMSE 4.53568 vs 3.7787; lower is better), so we make the smallest legitimate signal/robustness improvements without changing your model type or training loop. The main score drag here is that `fare_amount` outliers remain in the training sample; adding the standard NYC Taxi fare filter (keep fares in a reasonable range) typically improves generalization while preserving your approach. We also make the XGBoost objective modern/consistent (`reg:squarederror`) to avoid legacy behavior differences, and we actually apply the already-fitted scaler to train/test features (your current code fits it but never uses the scaled arrays), which is a minimal fix that often yields a modest RMSE gain. Submission writing/merge+fallback logic stays the same to guarantee a valid `submission1.csv`.'
- What this solution (achieved 4.57519) has done: 'Your current gap to the target is large (4.73308 vs 3.7787; lower is better), so we need a modest but still “same-core-logic” boost in signal without changing the model type or training loop. The biggest safe lift here is to add a couple of standard geospatial features that use only the existing columns: (1) Haversine distance in **miles** (fares are in USD and NYC heuristics align better in miles), and (2) simple coordinate interaction features (`abs_dlon`, `abs_dlat`). We keep your exact XGBRegressor setup (same n_estimators/training flow/loss), keep your existing cleaning, and only expand the feature matrix consistently for train and test. We also keep the same submission merge+fallback behavior to guarantee a valid `submission1.csv`.'
- What this solution (achieved 4.53844) has done: 'To move RMSE down toward your 3.7787 target (current 4.57519; lower is better) without changing the core XGBRegressor/training loop, I add a couple of standard “same-input-columns” geospatial features that usually give a modest lift: bearing (direction of travel) and simple NYC-center distances (pickup/dropoff distance to Manhattan). I also make train/test cleaning fully consistent by applying the same passenger_count filter (`!=208`) to test and clamping negative predictions to 0 (a safe, metric-aligned postprocess that often reduces RMSE because fares can’t be negative). Everything else (data loading size, model type, n_estimators, scaling, submission merge+fallback) stays the same and still writes a valid `submission1.csv`.'
- What this solution (achieved 4.64493) has done: 'I fix the runtime error by making `haversine_km` and `bearing_rad` accept scalar constants (NYC center lon/lat) as well as pandas Series/arrays, which currently breaks on `.astype` for floats. I keep the model/training loop and all existing features intact, only adjusting the helper functions to be robust and ensuring the feature engineering cells run for both train and test. This is score-neutral in intent (same features, same model) but unblocks the pipeline so you can actually generate a submission and preserve the improvements already included. The script still write a valid `submission1.csv` with exactly `key,fare_amount` and aligned row count.'
- What this solution (achieved 4.57677) has done: 'Your current RMSE (4.64493) is worse than the target (3.7787), so we should make a small, legitimate improvement without changing the core model/training loop. The biggest likely score drag here is that latitude/longitude are being fed in raw degrees with a StandardScaler; instead, we minimally add a physically meaningful `manhattan_km` feature (km-scaled Manhattan approximation) and a simple nonlinearity `haversine_km_sq`, which often improves XGBoost splits while keeping the same model and training procedure. We also ensure train/test datetime parsing doesn’t silently create NaNs after earlier `dropna` by dropping rows with invalid `pickup_datetime` immediately after parsing (same semantics, just consistent). Everything else (XGBRegressor, n_estimators, scaling, cleaning logic, submission merge+fallback) remains intact and still writes a valid `submission1.csv`.'
- What this solution (achieved 4.58363) has done: 'Your current RMSE (4.57677) is worse than the target (3.7787), so we should make a small, legitimate improvement without changing the core XGBRegressor/training loop. The biggest likely drag left is noisy training labels and remaining spatial outliers; we add two standard, lightweight filters: drop rides with implausibly large distances and remove extreme high fares (keeps the same target/metric semantics but improves generalization). We also add a single, classic NYC feature (airport proximity flags to JFK/LGA/EWR) derived only from existing lat/lon columns, which often gives a modest RMSE gain without changing the model. Everything else (same model type, same objective, same scaling usage, same submission merge+fallback) remains intact and still writes a valid `submission1.csv`.'
- What this solution (achieved 5.17863) has done: 'We’re still materially worse than the target (4.58363 vs 3.7787 RMSE; lower is better), so the smallest safe lift without changing your model/training loop is to fix a key train/test inconsistency: you currently apply the `haversine_km <= 100` distance filter only to the **test** features, but not to the **training** data until after features are created (and without using the same definition earlier), which can distort the learned mapping for long trips and worsen generalization. I add the same `haversine_km`-based distance filter to the training data using the *same* engineered `haversine_km` (already present) and also tighten an obvious NYC-taxi sanity filter by removing trips with implausibly small movement but nontrivial fare (a classic label-noise reducer). These are minimal, metric-aligned data-cleaning changes that preserve your XGBRegressor, feature set, scaling, and submission merge+fallback logic, and should move RMSE down toward the target band.'
- What this solution (achieved 5.26872) has done: 'Your current RMSE (5.17863) is worse than the target (3.7787), so we should make the smallest legitimate improvement without changing the core XGBRegressor/training loop or feature set. The main issue is a train/test distribution mismatch introduced by filtering test rows with `haversine_km <= 100` but keeping all those long-trip rows in training (and then also filtering training later in a way that can become inconsistent when NaNs/invalid coords slip through). I make the distance-based sanity filtering consistent by applying the same `haversine_km <= 100` and coordinate bounds after feature creation for both train and test, and I add one standard, minimal label-noise filter (drop extreme fares per km) that typically reduces RMSE while preserving your semantics. I also keep the submission merge+fallback logic unchanged to guarantee a valid `submission1.csv`.'
- What this solution (achieved 5.09154) has done: 'Your RMSE is far above the target (5.26872 vs 3.7787; lower is better), so the smallest safe move is to remove the strongest remaining label-noise source without changing your model or feature set: training rows with impossible/near-zero trip distance. Right now those rows stay in training because you clamp distance to 0.1 in `fare_per_km`, so “0-distance but nontrivial fare” samples can survive and distort the mapping; we drop near-zero-distance rides before computing `fare_per_km`. We also apply the same near-zero-distance sanity filter to `test_features` (so train/test filtering is consistent) while keeping your merge+fallback submission logic unchanged. Everything else (XGBRegressor, n_estimators, scaling, feature engineering) stays the same.'
- What this solution (achieved 5.12337) has done: 'Your current RMSE (5.09154) is still far above the target (3.7787), so we need a small but meaningful generalization gain without changing your model type or training loop. The biggest low-risk issue is that the model is trained on random row splits, which leaks very similar trips across train/valid and doesn’t reflect the public/private test distribution; we keep the same XGBRegressor and features, but switch to a time-based split using `pickup_datetime` while still training on the full cleaned sample. Also, the airport distance features are in km but the “< 2.0” threshold is very tight; we keep the same feature but add a slightly wider (still reasonable) airport flag feature at 5km (without removing the existing one), which typically reduces RMSE modestly. Finally, we ensure train/test feature columns are strictly aligned before scaling/predicting to avoid silent column-order mismatches.'
- What this solution (achieved 5.10583) has done: 'Your current RMSE (5.12337) is worse than the 3.7787 target (lower is better), so we make the smallest legitimate improvements that don’t change your model type or training loop: (1) fix a subtle distribution shift by applying the same time-based split logic only for validation, but still fitting the final model on the full cleaned training data before predicting test, and (2) add one very standard, minimal feature (`passenger_count` as float + a `night` flag) that uses only existing columns and tends to reduce error slightly without altering architecture. We also ensure the exact same feature columns (same order) are used for train/valid/test, and keep your existing submission merge+fallback so a valid `submission1.csv` is always written. These changes are aimed to move RMSE down toward the target band without “over-optimizing” or restructuring the approach.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as MSE



## === cell 2
train = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/train.csv", nrows=1000000
)
test = pd.read_csv("../input/new-york-city-taxi-fare-prediction/test.csv")



## === cell 3
train.shape, test.shape



## === cell 4
train.head()



## === cell 5
train.isnull().sum()



## === cell 6
train = train.dropna(how="any", axis="rows")



## === cell 7
test.isnull().sum()



## === cell 8
train.head()



## === cell 9
train["fare_amount"].describe()



## === cell 10
train.drop(train[train["pickup_longitude"] == 0].index, axis=0, inplace=True)
train.drop(train[train["pickup_latitude"] == 0].index, axis=0, inplace=True)
train.drop(train[train["dropoff_longitude"] == 0].index, axis=0, inplace=True)
train.drop(train[train["dropoff_latitude"] == 0].index, axis=0, inplace=True)
train.drop(train[train["passenger_count"] == 208].index, axis=0, inplace=True)
train.drop(train[train["passenger_count"] > 5].index, axis=0, inplace=True)
train.drop(train[train["passenger_count"] == 0].index, axis=0, inplace=True)



## === cell 11
train.head()



## === cell 12
train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], errors="coerce", utc=True
)
train = train.dropna(subset=["pickup_datetime"]).copy()

train["pickup_hour"] = train["pickup_datetime"].dt.hour.astype("float32")
train["pickup_dayofweek"] = train["pickup_datetime"].dt.dayofweek.astype("float32")
train["pickup_month"] = train["pickup_datetime"].dt.month.astype("float32")

train["is_night"] = ((train["pickup_hour"] <= 6) | (train["pickup_hour"] >= 20)).astype(
    "int8"
)



## === cell 13
pass



## === cell 14
train.dropna(inplace=True)

train.drop(
    train.index[
        (train.pickup_longitude < -75)
        | (train.pickup_longitude > -72)
        | (train.pickup_latitude < 40)
        | (train.pickup_latitude > 42)
    ],
    inplace=True,
)
train.drop(
    train.index[
        (train.dropoff_longitude < -75)
        | (train.dropoff_longitude > -72)
        | (train.dropoff_latitude < 40)
        | (train.dropoff_latitude > 42)
    ],
    inplace=True,
)

train = train[(train["fare_amount"] >= 2.5) & (train["fare_amount"] <= 150.0)].copy()



## === cell 15
train.head()




## === cell 16
def _to_numpy_float(x):
    if isinstance(x, (pd.Series, pd.Index)):
        return x.to_numpy(dtype="float64")
    if isinstance(x, np.ndarray):
        return x.astype("float64", copy=False)
    return np.asarray(x, dtype="float64")


def haversine_km(lon1, lat1, lon2, lat2):
    lon1 = np.radians(_to_numpy_float(lon1))
    lat1 = np.radians(_to_numpy_float(lat1))
    lon2 = np.radians(_to_numpy_float(lon2))
    lat2 = np.radians(_to_numpy_float(lat2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c  # Earth radius in km




## === cell 17
def bearing_rad(lon1, lat1, lon2, lat2):
    lon1 = np.radians(_to_numpy_float(lon1))
    lat1 = np.radians(_to_numpy_float(lat1))
    lon2 = np.radians(_to_numpy_float(lon2))
    lat2 = np.radians(_to_numpy_float(lat2))
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.arctan2(y, x)  # [-pi, pi]


NYC_LON = -73.985428  # approx Times Square / Midtown
NYC_LAT = 40.748817

JFK_LON, JFK_LAT = -73.7781, 40.6413
LGA_LON, LGA_LAT = -73.8740, 40.7769
EWR_LON, EWR_LAT = -74.1745, 40.6895



## === cell 18
train["haversine_km"] = haversine_km(
    train["pickup_longitude"],
    train["pickup_latitude"],
    train["dropoff_longitude"],
    train["dropoff_latitude"],
)
train["haversine_miles"] = (train["haversine_km"] * 0.621371).astype("float32")

train["manhattan"] = (
    (train["pickup_longitude"] - train["dropoff_longitude"]).abs()
    + (train["pickup_latitude"] - train["dropoff_latitude"]).abs()
).astype("float32")
train["abs_dlon"] = (
    (train["pickup_longitude"] - train["dropoff_longitude"]).abs().astype("float32")
)
train["abs_dlat"] = (
    (train["pickup_latitude"] - train["dropoff_latitude"]).abs().astype("float32")
)

train["bearing"] = bearing_rad(
    train["pickup_longitude"],
    train["pickup_latitude"],
    train["dropoff_longitude"],
    train["dropoff_latitude"],
).astype("float32")
train["pickup_dist_to_nyc_km"] = haversine_km(
    train["pickup_longitude"], train["pickup_latitude"], NYC_LON, NYC_LAT
).astype("float32")
train["dropoff_dist_to_nyc_km"] = haversine_km(
    train["dropoff_longitude"], train["dropoff_latitude"], NYC_LON, NYC_LAT
).astype("float32")

KM_PER_DEG_LAT = 111.32
train["manhattan_km"] = (
    (train["abs_dlat"] * KM_PER_DEG_LAT)
    + (train["abs_dlon"] * KM_PER_DEG_LAT * np.cos(np.radians(NYC_LAT)))
).astype("float32")
train["haversine_km_sq"] = (train["haversine_km"] ** 2).astype("float32")

train["pickup_to_jfk_km"] = haversine_km(
    train["pickup_longitude"], train["pickup_latitude"], JFK_LON, JFK_LAT
).astype("float32")
train["dropoff_to_jfk_km"] = haversine_km(
    train["dropoff_longitude"], train["dropoff_latitude"], JFK_LON, JFK_LAT
).astype("float32")
train["pickup_to_lga_km"] = haversine_km(
    train["pickup_longitude"], train["pickup_latitude"], LGA_LON, LGA_LAT
).astype("float32")
train["dropoff_to_lga_km"] = haversine_km(
    train["dropoff_longitude"], train["dropoff_latitude"], LGA_LON, LGA_LAT
).astype("float32")
train["pickup_to_ewr_km"] = haversine_km(
    train["pickup_longitude"], train["pickup_latitude"], EWR_LON, EWR_LAT
).astype("float32")
train["dropoff_to_ewr_km"] = haversine_km(
    train["dropoff_longitude"], train["dropoff_latitude"], EWR_LON, EWR_LAT
).astype("float32")

train["is_airport_trip"] = (
    (train["pickup_to_jfk_km"] < 2.0)
    | (train["dropoff_to_jfk_km"] < 2.0)
    | (train["pickup_to_lga_km"] < 2.0)
    | (train["dropoff_to_lga_km"] < 2.0)
    | (train["pickup_to_ewr_km"] < 2.0)
    | (train["dropoff_to_ewr_km"] < 2.0)
).astype("int8")

train["is_airport_trip_5km"] = (
    (train["pickup_to_jfk_km"] < 5.0)
    | (train["dropoff_to_jfk_km"] < 5.0)
    | (train["pickup_to_lga_km"] < 5.0)
    | (train["dropoff_to_lga_km"] < 5.0)
    | (train["pickup_to_ewr_km"] < 5.0)
    | (train["dropoff_to_ewr_km"] < 5.0)
).astype("int8")

train = train[train["haversine_km"] <= 100.0].copy()
train = train[train["haversine_km"] >= 0.05].copy()

fare_per_km = train["fare_amount"] / np.maximum(
    train["haversine_km"].astype("float64"), 0.1
)
train = train[fare_per_km <= 80.0].copy()

train = train[~((train["haversine_km"] < 0.05) & (train["fare_amount"] > 10.0))].copy()

train["passenger_count"] = train["passenger_count"].astype("float32")



## === cell 19
train_sorted = train.sort_values("pickup_datetime").reset_index(drop=True)
split_idx = int(len(train_sorted) * 0.8)
train_part = train_sorted.iloc[:split_idx].copy()
valid_part = train_sorted.iloc[split_idx:].copy()

feature_cols = [
    c for c in train.columns if c not in ["fare_amount", "key", "pickup_datetime"]
]

X_train = train_part[feature_cols]
y_train = train_part["fare_amount"].astype("float32")

X_test = valid_part[feature_cols]
y_test = valid_part["fare_amount"].astype("float32")



## === cell 20
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



## === cell 21
xgb_r = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=400, seed=123)



## === cell 22
xgb_r.fit(X_train_scaled, y_train)



## === cell 23
y_pred = xgb_r.predict(X_test_scaled)



## === cell 24
rmse = np.sqrt(MSE(y_test, y_pred))
print("RMSE : % f" % (rmse))



## === cell 25
test.head()



## === cell 26
test_features = test.copy()



## === cell 27
test_features = test_features.dropna()

test_features.drop(
    test_features[test_features["pickup_longitude"] == 0].index, axis=0, inplace=True
)
test_features.drop(
    test_features[test_features["pickup_latitude"] == 0].index, axis=0, inplace=True
)
test_features.drop(
    test_features[test_features["dropoff_longitude"] == 0].index, axis=0, inplace=True
)
test_features.drop(
    test_features[test_features["dropoff_latitude"] == 0].index, axis=0, inplace=True
)

test_features.drop(
    test_features[test_features["passenger_count"] == 208].index, axis=0, inplace=True
)
test_features.drop(
    test_features[test_features["passenger_count"] > 5].index, axis=0, inplace=True
)
test_features.drop(
    test_features[test_features["passenger_count"] == 0].index, axis=0, inplace=True
)

test_features.drop(
    test_features.index[
        (test_features.pickup_longitude < -75)
        | (test_features.pickup_longitude > -72)
        | (test_features.pickup_latitude < 40)
        | (test_features.pickup_latitude > 42)
    ],
    inplace=True,
)

test_features.drop(
    test_features.index[
        (test_features.dropoff_longitude < -75)
        | (test_features.dropoff_longitude > -72)
        | (test_features.dropoff_latitude < 40)
        | (test_features.dropoff_latitude > 42)
    ],
    inplace=True,
)



## === cell 28
test_features.head()



## === cell 29
test_features["pickup_datetime"] = pd.to_datetime(
    test_features["pickup_datetime"], errors="coerce", utc=True
)
test_features = test_features.dropna(subset=["pickup_datetime"]).copy()

test_features["pickup_hour"] = test_features["pickup_datetime"].dt.hour.astype(
    "float32"
)
test_features["pickup_dayofweek"] = test_features[
    "pickup_datetime"
].dt.dayofweek.astype("float32")
test_features["pickup_month"] = test_features["pickup_datetime"].dt.month.astype(
    "float32"
)

test_features["is_night"] = (
    (test_features["pickup_hour"] <= 6) | (test_features["pickup_hour"] >= 20)
).astype("int8")



## === cell 30
test_features = test_features.drop(["pickup_datetime"], axis=1)



## === cell 31
test_features["haversine_km"] = haversine_km(
    test_features["pickup_longitude"],
    test_features["pickup_latitude"],
    test_features["dropoff_longitude"],
    test_features["dropoff_latitude"],
)
test_features["haversine_miles"] = (test_features["haversine_km"] * 0.621371).astype(
    "float32"
)

test_features["manhattan"] = (
    (test_features["pickup_longitude"] - test_features["dropoff_longitude"]).abs()
    + (test_features["pickup_latitude"] - test_features["dropoff_latitude"]).abs()
).astype("float32")
test_features["abs_dlon"] = (
    (test_features["pickup_longitude"] - test_features["dropoff_longitude"])
    .abs()
    .astype("float32")
)
test_features["abs_dlat"] = (
    (test_features["pickup_latitude"] - test_features["dropoff_latitude"])
    .abs()
    .astype("float32")
)

test_features["bearing"] = bearing_rad(
    test_features["pickup_longitude"],
    test_features["pickup_latitude"],
    test_features["dropoff_longitude"],
    test_features["dropoff_latitude"],
).astype("float32")
test_features["pickup_dist_to_nyc_km"] = haversine_km(
    test_features["pickup_longitude"],
    test_features["pickup_latitude"],
    NYC_LON,
    NYC_LAT,
).astype("float32")
test_features["dropoff_dist_to_nyc_km"] = haversine_km(
    test_features["dropoff_longitude"],
    test_features["dropoff_latitude"],
    NYC_LON,
    NYC_LAT,
).astype("float32")

KM_PER_DEG_LAT = 111.32
test_features["manhattan_km"] = (
    (test_features["abs_dlat"] * KM_PER_DEG_LAT)
    + (test_features["abs_dlon"] * KM_PER_DEG_LAT * np.cos(np.radians(NYC_LAT)))
).astype("float32")
test_features["haversine_km_sq"] = (test_features["haversine_km"] ** 2).astype(
    "float32"
)

test_features["pickup_to_jfk_km"] = haversine_km(
    test_features["pickup_longitude"],
    test_features["pickup_latitude"],
    JFK_LON,
    JFK_LAT,
).astype("float32")
test_features["dropoff_to_jfk_km"] = haversine_km(
    test_features["dropoff_longitude"],
    test_features["dropoff_latitude"],
    JFK_LON,
    JFK_LAT,
).astype("float32")
test_features["pickup_to_lga_km"] = haversine_km(
    test_features["pickup_longitude"],
    test_features["pickup_latitude"],
    LGA_LON,
    LGA_LAT,
).astype("float32")
test_features["dropoff_to_lga_km"] = haversine_km(
    test_features["dropoff_longitude"],
    test_features["dropoff_latitude"],
    LGA_LON,
    LGA_LAT,
).astype("float32")
test_features["pickup_to_ewr_km"] = haversine_km(
    test_features["pickup_longitude"],
    test_features["pickup_latitude"],
    EWR_LON,
    EWR_LAT,
).astype("float32")
test_features["dropoff_to_ewr_km"] = haversine_km(
    test_features["dropoff_longitude"],
    test_features["dropoff_latitude"],
    EWR_LON,
    EWR_LAT,
).astype("float32")
test_features["is_airport_trip"] = (
    (test_features["pickup_to_jfk_km"] < 2.0)
    | (test_features["dropoff_to_jfk_km"] < 2.0)
    | (test_features["pickup_to_lga_km"] < 2.0)
    | (test_features["dropoff_to_lga_km"] < 2.0)
    | (test_features["pickup_to_ewr_km"] < 2.0)
    | (test_features["dropoff_to_ewr_km"] < 2.0)
).astype("int8")

test_features["is_airport_trip_5km"] = (
    (test_features["pickup_to_jfk_km"] < 5.0)
    | (test_features["dropoff_to_jfk_km"] < 5.0)
    | (test_features["pickup_to_lga_km"] < 5.0)
    | (test_features["dropoff_to_lga_km"] < 5.0)
    | (test_features["pickup_to_ewr_km"] < 5.0)
    | (test_features["dropoff_to_ewr_km"] < 5.0)
).astype("int8")

test_features = test_features[test_features["haversine_km"] <= 100.0].copy()
test_features = test_features[test_features["haversine_km"] >= 0.05].copy()

test_features["passenger_count"] = test_features["passenger_count"].astype("float32")



## === cell 32
test_keys_valid = test_features["key"].copy()

X_test_pred = test_features.reindex(columns=feature_cols).copy()
X_test_pred_scaled = scaler.transform(X_test_pred)



## === cell 33
new_pred = xgb_r.predict(X_test_pred_scaled)
new_pred = np.maximum(new_pred, 0.0)



## === cell 34
fallback = float(y_train.mean())

submission = test[["key"]].copy()
pred_df = pd.DataFrame({"key": test_keys_valid.values, "fare_amount": new_pred})
submission = submission.merge(pred_df, on="key", how="left")
submission["fare_amount"] = submission["fare_amount"].fillna(fallback)



## === cell 35
submission.head()



## === cell 36
submission = submission[["key", "fare_amount"]]
submission.to_csv("submission1.csv", index=False)
print("Wrote submission1.csv with shape:", submission.shape)
print(submission.columns.tolist())



## === cell 37
assert submission.shape[0] == test.shape[0]
assert list(submission.columns) == ["key", "fare_amount"]
assert submission["fare_amount"].notna().all()
submission.describe(include="all")
