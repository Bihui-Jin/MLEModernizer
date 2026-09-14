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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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

4.87021

# 6. Current score

6.03464

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.63573) has done: 'I fix the pandas `DataFrame.drop()` calls that now error under pandas 2.x by switching to keyword arguments (`axis=1` / `columns=`), which unblock preprocessing. Then I address the TensorFlow crash (`MessageFactory` / protobuf incompatibility) by using a scikit-learn regressor as a drop-in replacement so the pipeline trains and predicts reliably in this environment while keeping the same features/metric semantics (regression with MSE/RMSE). Finally, I ensure the test matrix is purely numeric (no object dtype) and write a valid `submission.csv` with exactly `key,fare_amount` columns.'
- What this solution (achieved 5.60991) has done: 'You’re currently worse than the target (RMSE 5.63573 vs 4.87021, lower is better), so we should make small, low-risk improvements that usually reduce RMSE without changing the overall approach (same features, same model family, same train/fit/predict flow). The biggest likely issue is that the model can’t use missingness properly because test NaNs are filled with 0 (a strong, misleading signal after scaling), and the model is also trained without any robustness for NaNs. We keep the same HistGradientBoostingRegressor but (1) stop forcing missing values to 0 and let the model handle NaNs natively, (2) remove MinMaxScaler (tree models don’t need scaling; scaling plus NaN handling can be counterproductive), and (3) add very light, standard NYC taxi coordinate sanity filtering + passenger_count filtering on train only to reduce label noise (this typically improves RMSE while preserving feature logic). These changes are minimal, keep the same features and model, and should move the score toward the target.'
- What this solution (achieved 5.95509) has done: 'Your current RMSE (5.60991) is worse than the target (4.87021), so we make small, low-risk changes that typically reduce error without changing the overall modeling approach (same features and same HistGradientBoostingRegressor fit/predict flow). The biggest likely gain is adding a couple of standard taxi-fare features (straight-line distance in kilometers, Manhattan distance, and a simple “airport-ish” indicator) while keeping your existing distance/time features intact. We also add a mild outlier filter on fare-per-mile to reduce noisy training labels and clip negative predictions to 0 for metric stability. These are minimal additions that usually move this specific competition score downward.'
- What this solution (achieved 5.90313) has done: 'Your current RMSE (5.95509) is worse than the target (4.87021), so we should make a small, low-risk improvement that typically reduces taxi-fare error without changing the core model or training loop. The biggest missing signal in your current feature set is the “direction” of the trip; adding simple bearing features (sin/cos of the heading) often improves RMSE while keeping the same overall approach (same HGBRegressor, same fit/predict flow, same existing features). I also clamp `passenger_count` in the test set to the valid range to reduce distribution mismatch (train is filtered to 1–6 but test is not), which is a minimal, metric-relevant fix. Everything else (data reading, filters, model class/hyperparams, submission schema) remains unchanged.'
- What this solution (achieved 5.78247) has done: 'To move your RMSE down toward the 4.87 target without changing the overall modeling approach, I make a small, metric-aligned change: train the model on `log1p(fare_amount)` and then invert with `expm1` at prediction time. This preserves the same feature engineering, the same `HistGradientBoostingRegressor` fit/predict flow, and the same squared-error objective, but typically reduces the impact of heavy-tailed fares and improves RMSE on this competition. I keep your existing filters and post-processing, only adjusting the target transform and clipping after inverse-transform to keep predictions valid. The script still run end-to-end and write a valid `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 5.88667) has done: 'Your current RMSE (5.78247) is worse than the target (4.87021), so we should make the smallest changes that typically reduce error without changing your overall approach (same features, same HistGradientBoostingRegressor training/predict flow, same log1p target transform). The biggest low-risk gain here is fixing feature consistency and missingness: right now train rows with any NaN are dropped, but test NaNs are kept and passed through, which can degrade predictions; we align by filling remaining test NaNs with train-feature medians. Next, your model is trained on a random split, but taxi fares are time-dependent; switching the split to be time-based (train early dates, validate later dates) typically produces a better-generalizing fit (even if the printed validation RMSE changes), and we still train on all training data before predicting test. Finally, we keep everything else intact and only add a very mild clip on extreme distances in test to match your train filter (`distance_miles < 17`) so the model doesn’t extrapolate as much.'
- What this solution (achieved 6.03464) has done: 'Your current RMSE (5.88667) is worse than the target (4.87021), so we should make small, low-risk changes that usually reduce error without changing the overall pipeline (same features, same model class, same fit/predict flow, same log1p target transform). The biggest likely regression here is that the model is trained on raw (unscaled) features with very different magnitudes (e.g., `min_airport_km` vs `dayofyear` vs distances), which can hurt HistGradientBoosting’s binning; adding a simple StandardScaler (fit on train only, apply to valid/test) often improves RMSE while preserving semantics. Additionally, your `manhattan_dist` is in degrees while other distance features are in miles/km; converting it to an approximate kilometer scale makes it consistent and usually improves generalization. Finally, we clip the training distances the same way as test (both miles and km), preventing a small train/test mismatch that can increase error on edge cases.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:50]:
        print(os.path.join(dirname, filename))



## === cell 1
import matplotlib.pyplot as plt
import seaborn as sns

sample_sub = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
)
train_df = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", nrows=500000
)
test_df = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")

train_df["pickup_datetime"] = pd.to_datetime(
    train_df["pickup_datetime"], errors="coerce", utc=False
)
test_df["pickup_datetime"] = pd.to_datetime(
    test_df["pickup_datetime"], errors="coerce", utc=False
)

key = test_df["key"].copy()

test_df = test_df.drop(columns=["key"])[
    [
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]
]

train_df = train_df.drop(columns=["key"])[
    [
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
        "fare_amount",
    ]
]



## === cell 2
train_df = train_df[train_df["fare_amount"] > 0]
train_df = train_df[train_df["fare_amount"] < 100]

train_df = train_df.dropna(subset=["pickup_datetime"])
test_df = test_df.dropna(subset=["pickup_datetime"])

test_df["passenger_count"] = test_df["passenger_count"].clip(lower=1, upper=6)

train_df = train_df[
    (train_df["passenger_count"] >= 1) & (train_df["passenger_count"] <= 6)
]

nyc_lon_min, nyc_lon_max = -74.5, -72.5
nyc_lat_min, nyc_lat_max = 40.0, 41.8
train_df = train_df[
    (train_df["pickup_longitude"].between(nyc_lon_min, nyc_lon_max))
    & (train_df["dropoff_longitude"].between(nyc_lon_min, nyc_lon_max))
    & (train_df["pickup_latitude"].between(nyc_lat_min, nyc_lat_max))
    & (train_df["dropoff_latitude"].between(nyc_lat_min, nyc_lat_max))
]




## === cell 3
def haversine_km(lat1, lon1, lat2, lon2):
    p = np.pi / 180.0
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2.0
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1.0 - np.cos((lon2 - lon1) * p)) / 2.0
    )
    return 12742.0 * np.arcsin(np.sqrt(a))  # 2*R where R≈6371km


def distance_miles(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295  # Pi/180
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))


train_df["distance_miles"] = distance_miles(
    train_df.pickup_latitude,
    train_df.pickup_longitude,
    train_df.dropoff_latitude,
    train_df.dropoff_longitude,
)
test_df["distance_miles"] = distance_miles(
    test_df.pickup_latitude,
    test_df.pickup_longitude,
    test_df.dropoff_latitude,
    test_df.dropoff_longitude,
)

train_df["distance_km"] = haversine_km(
    train_df.pickup_latitude,
    train_df.pickup_longitude,
    train_df.dropoff_latitude,
    train_df.dropoff_longitude,
).astype(np.float32)
test_df["distance_km"] = haversine_km(
    test_df.pickup_latitude,
    test_df.pickup_longitude,
    test_df.dropoff_latitude,
    test_df.dropoff_longitude,
).astype(np.float32)

LAT_KM_PER_DEG = 111.0
LON_KM_PER_DEG_NYC = 85.0

train_df["manhattan_dist"] = (
    (train_df["pickup_latitude"] - train_df["dropoff_latitude"]).abs() * LAT_KM_PER_DEG
    + (train_df["pickup_longitude"] - train_df["dropoff_longitude"]).abs()
    * LON_KM_PER_DEG_NYC
).astype(np.float32)
test_df["manhattan_dist"] = (
    (test_df["pickup_latitude"] - test_df["dropoff_latitude"]).abs() * LAT_KM_PER_DEG
    + (test_df["pickup_longitude"] - test_df["dropoff_longitude"]).abs()
    * LON_KM_PER_DEG_NYC
).astype(np.float32)


def bearing_rad(lat1, lon1, lat2, lon2):
    lat1r = np.deg2rad(lat1)
    lat2r = np.deg2rad(lat2)
    dlon = np.deg2rad(lon2 - lon1)
    y = np.sin(dlon) * np.cos(lat2r)
    x = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
    return np.arctan2(y, x)


train_bearing = bearing_rad(
    train_df["pickup_latitude"],
    train_df["pickup_longitude"],
    train_df["dropoff_latitude"],
    train_df["dropoff_longitude"],
).astype(np.float32)
test_bearing = bearing_rad(
    test_df["pickup_latitude"],
    test_df["pickup_longitude"],
    test_df["dropoff_latitude"],
    test_df["dropoff_longitude"],
).astype(np.float32)

train_df["bearing_sin"] = np.sin(train_bearing).astype(np.float32)
train_df["bearing_cos"] = np.cos(train_bearing).astype(np.float32)
test_df["bearing_sin"] = np.sin(test_bearing).astype(np.float32)
test_df["bearing_cos"] = np.cos(test_bearing).astype(np.float32)



## === cell 4
JFK_LAT, JFK_LON = 40.6413, -73.7781
LGA_LAT, LGA_LON = 40.7769, -73.8740
EWR_LAT, EWR_LON = 40.6895, -74.1745


def min_airport_dist_km(df):
    d_jfk_p = haversine_km(
        df["pickup_latitude"], df["pickup_longitude"], JFK_LAT, JFK_LON
    )
    d_lga_p = haversine_km(
        df["pickup_latitude"], df["pickup_longitude"], LGA_LAT, LGA_LON
    )
    d_ewr_p = haversine_km(
        df["pickup_latitude"], df["pickup_longitude"], EWR_LAT, EWR_LON
    )
    d_jfk_d = haversine_km(
        df["dropoff_latitude"], df["dropoff_longitude"], JFK_LAT, JFK_LON
    )
    d_lga_d = haversine_km(
        df["dropoff_latitude"], df["dropoff_longitude"], LGA_LAT, LGA_LON
    )
    d_ewr_d = haversine_km(
        df["dropoff_latitude"], df["dropoff_longitude"], EWR_LAT, EWR_LON
    )
    return np.minimum.reduce(
        [d_jfk_p, d_lga_p, d_ewr_p, d_jfk_d, d_lga_d, d_ewr_d]
    ).astype(np.float32)


train_df["min_airport_km"] = min_airport_dist_km(train_df)
test_df["min_airport_km"] = min_airport_dist_km(test_df)

train_df["near_airport"] = (train_df["min_airport_km"] < 2.0).astype(np.int8)
test_df["near_airport"] = (test_df["min_airport_km"] < 2.0).astype(np.int8)



## === cell 5
train_df = train_df.drop(
    columns=[
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
    ]
)
train_df["hour_of_day"] = train_df["pickup_datetime"].dt.hour
train_df["dayofyear"] = train_df["pickup_datetime"].dt.dayofyear
train_df["month"] = train_df["pickup_datetime"].dt.month
train_df["year"] = train_df["pickup_datetime"].dt.year
train_df["year"] = train_df["year"].apply(
    lambda x: int(str(int(x))[-2:]) if pd.notnull(x) else np.nan
)
train_df = train_df.drop(columns=["pickup_datetime"])
train_df = train_df[train_df["distance_miles"] < 17]

test_df = test_df.drop(
    columns=[
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
    ]
)
test_df["hour_of_day"] = test_df["pickup_datetime"].dt.hour
test_df["dayofyear"] = test_df["pickup_datetime"].dt.dayofyear
test_df["month"] = test_df["pickup_datetime"].dt.month
test_df["year"] = test_df["pickup_datetime"].dt.year
test_df["year"] = test_df["year"].apply(
    lambda x: int(str(int(x))[-2:]) if pd.notnull(x) else np.nan
)
test_df = test_df.drop(columns=["pickup_datetime"])

train_df = train_df.replace([np.inf, -np.inf], np.nan).dropna()
test_df = test_df.replace([np.inf, -np.inf], np.nan)

ppm = train_df["fare_amount"] / (train_df["distance_miles"] + 1e-3)
train_df = train_df[(ppm > 0.5) & (ppm < 50)].copy()

train_df["distance_miles"] = train_df["distance_miles"].clip(lower=0.0, upper=17.0)
train_df["distance_km"] = train_df["distance_km"].clip(lower=0.0, upper=27.4)
train_df["manhattan_dist"] = train_df["manhattan_dist"].clip(lower=0.0)

test_df["distance_miles"] = test_df["distance_miles"].clip(lower=0.0, upper=17.0)
test_df["distance_km"] = test_df["distance_km"].clip(lower=0.0, upper=27.4)  # ~17 miles
test_df["manhattan_dist"] = test_df["manhattan_dist"].clip(lower=0.0)

feature_cols = [c for c in train_df.columns if c != "fare_amount"]
train_feature_medians = train_df[feature_cols].median(numeric_only=True)
test_df[feature_cols] = test_df[feature_cols].fillna(train_feature_medians)



## === cell 6
from sklearn.preprocessing import StandardScaler

X = train_df.drop(columns=["fare_amount"]).to_numpy(dtype=np.float32)
y = np.log1p(train_df["fare_amount"].to_numpy(dtype=np.float32))

dt_proxy = (
    train_df["year"].to_numpy(dtype=np.float32) * 1000.0
    + train_df["dayofyear"].to_numpy(dtype=np.float32)
    + train_df["hour_of_day"].to_numpy(dtype=np.float32) / 24.0
)
order = np.argsort(dt_proxy)
cut = int(0.70 * len(order))
train_idx = order[:cut]
valid_idx = order[cut:]

X_train, X_valid = X[train_idx], X[valid_idx]
y_train, y_valid = y[train_idx], y[valid_idx]

X_test_final = test_df.to_numpy(dtype=np.float32)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test_final = scaler.transform(X_test_final)

X_scaled_full = scaler.fit_transform(X)



## === cell 7
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

model = HistGradientBoostingRegressor(
    loss="squared_error", random_state=101, max_depth=6, learning_rate=0.1, max_iter=300
)

model.fit(X_train, y_train)

valid_pred_log = model.predict(X_valid)
valid_pred = np.expm1(valid_pred_log)
y_valid_dollars = np.expm1(y_valid)

valid_pred = np.clip(valid_pred, 0.0, None)
rmse = mean_squared_error(y_valid_dollars, valid_pred, squared=False)
print("Validation RMSE:", rmse)

model.fit(X_scaled_full, y)



## === cell 8
sub_pred_log = model.predict(X_test_final).astype(np.float32)
sub_pred = np.expm1(sub_pred_log).astype(np.float32)

sub_pred = np.clip(sub_pred, 0.0, None)

submission = pd.DataFrame({"key": key.values, "fare_amount": sub_pred})
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
