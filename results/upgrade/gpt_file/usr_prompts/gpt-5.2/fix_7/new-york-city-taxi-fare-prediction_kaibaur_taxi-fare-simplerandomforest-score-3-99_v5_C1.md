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

3.11

# 3. Installed packages

geopandas==0.14.4
lightgbm==4.6.0
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
plotly==5.24.1
plotly-express==0.4.1
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

3.99076

# 6. Current score

4.56016

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.43343) has done: 'I fix the LightGBM fit-time crash by replacing the unsupported `verbose` argument with the LightGBM 4.x-compatible callback-based logging, so training completes. Then the downstream `NotFittedError` disappears because the model actually be trained before inference. I also make the `skiprows` sampler deterministic by using the row index rather than consuming a mutable RNG state (this avoids subtle sampling variability across runs and Pandas internals while preserving the same core approach). Finally, I ensure the submission is written as `submission.csv` with exactly the required columns aligned to `test_keys`.'
- What this solution (achieved 5.15882) has done: 'Your current score (5.43343 RMSE) is worse than the target (3.99076), so we should make small, legitimate improvements that typically reduce RMSE without changing the overall LightGBM approach. The biggest win with minimal logic change is to train on log1p(fare) and invert with expm1 at prediction time, which reduces the impact of large-fare outliers while still optimizing RMSE on the original scale after inversion. I also add a couple of standard NYC-taxi features (Manhattan distance and distance-to-JFK/LGA) that are simple deterministic transformations of existing columns and commonly improve this competition. Finally, I retrain on the full cleaned sample after confirming validation (same model family/training loop) so the submission benefits from all available sampled data.'
- What this solution (achieved 4.66997) has done: 'Your current RMSE (5.15882) is worse than the target (3.99076), so we should make small, safe changes that usually improve this competition without changing the overall LightGBM approach. The biggest likely gain with minimal logic change is to train with LightGBM’s built-in missing-value handling by converting impossible coordinates (0,0 and zeros) to NaN instead of keeping them as real locations, and to add a single standard interaction feature (bearing) that complements your existing distance features. I also align the split to be deterministic and slightly more representative by using a fixed-size validation sampled from the already-cleaned data (still `train_test_split`), and keep all core model/training settings intact. The submission writing stays identical (same columns, order, filename), just with improved features/cleaning feeding the same model family.'
- What this solution (achieved 4.49353) has done: 'We need to move your RMSE down toward the target, so the smallest legitimate gain is to fix a likely distribution mismatch: your training cleaning keeps only NYC-bounding-box rides, but you don’t apply the same coordinate sanity rules to the test set (and you turn zeros into NaN), which can create very bad predictions for out-of-box/NaN rows. I keep the exact LightGBM setup and training loop, but add a minimal “test guardrail” that (a) flags invalid/out-of-bounds coordinates and (b) replaces their predictions with a simple baseline (the median fare on the original scale from the cleaned training sample). This typically reduces tail errors and improves RMSE without changing the model architecture or optimization. I also add `min_child_samples` (a standard regularization knob in LightGBM) very lightly to reduce overfitting to the sampled subset; everything else remains the same and the submission format stays identical.'
- What this solution (achieved 4.56016) has done: 'Your RMSE (4.49353) is above the target (3.99076), so we should make a small, legitimate improvement that typically reduces error without changing the LightGBM approach. The biggest low-risk gain here is to add the standard NYC feature `abs_bearing` (you already compute bearing, but the sign is often arbitrary; using its absolute value is a tiny deterministic transform that often improves RMSE). Second, the current “invalid/OOB test guardrail” uses a single global median fare; we can keep the same guardrail idea but make it slightly smarter by using a baseline derived from training medians by `passenger_count` (falling back to global median), which usually reduces large errors for those flagged rows. Everything else (sampling, cleaning, log1p target, model type/hyperparams, training loop, submission format) stays the same.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

import os
import gc
import datetime as dt

import lightgbm as lgbm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

RANDOM_STATE = 42

DATA_DIR = "/kaggle/input" if os.path.exists("/kaggle/input") else "/kaggle/data"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

if not os.path.exists(TRAIN_PATH):
    TRAIN_PATH = os.path.join(
        DATA_DIR, "new-york-city-taxi-fare-prediction", "train.csv"
    )
if not os.path.exists(TEST_PATH):
    TEST_PATH = os.path.join(DATA_DIR, "new-york-city-taxi-fare-prediction", "test.csv")

assert os.path.exists(TRAIN_PATH), f"train.csv not found at {TRAIN_PATH}"
assert os.path.exists(TEST_PATH), f"test.csv not found at {TEST_PATH}"



## === cell 1
N_TRAIN = 2_000_000  # bounded subset for speed + better score than tiny samples
DTYPE_MAP = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16",
}

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

n_total_guess = 55_423_856  # provided in prompt
keep_prob = min(1.0, N_TRAIN / n_total_guess)


def _skiprow(i: int) -> bool:
    if i == 0:
        return False
    x = (i * 1103515245 + RANDOM_STATE * 12345) & 0xFFFFFFFF
    u = x / 2**32
    return u > keep_prob


train = pd.read_csv(
    TRAIN_PATH,
    usecols=usecols_train,
    dtype=DTYPE_MAP,
    parse_dates=["pickup_datetime"],
    skiprows=_skiprow,
)

test = pd.read_csv(
    TEST_PATH,
    usecols=usecols_test,
    dtype={k: v for k, v in DTYPE_MAP.items() if k != "fare_amount"},
    parse_dates=["pickup_datetime"],
)

print("Train shape (sampled):", train.shape)
print("Test shape:", test.shape)




## === cell 2
def haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.deg2rad(lat1)
    lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2)
    lon2 = np.deg2rad(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon / 2.0) ** 2
    )
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(a))


def bearing_rad(lat1, lon1, lat2, lon2):
    lat1r = np.deg2rad(lat1)
    lat2r = np.deg2rad(lat2)
    dlon = np.deg2rad(lon2 - lon1)
    y = np.sin(dlon) * np.cos(lat2r)
    x = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
    return np.arctan2(y, x)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["pickup_hour"] = out["pickup_datetime"].dt.hour.astype("int16")
    out["pickup_dayofweek"] = out["pickup_datetime"].dt.dayofweek.astype("int16")
    out["pickup_month"] = out["pickup_datetime"].dt.month.astype("int16")
    out["pickup_year"] = out["pickup_datetime"].dt.year.astype("int16")

    out["haversine_km"] = haversine_km(
        out["pickup_latitude"].values,
        out["pickup_longitude"].values,
        out["dropoff_latitude"].values,
        out["dropoff_longitude"].values,
    ).astype("float32")

    out["manhattan_dist"] = (
        np.abs(out["pickup_longitude"] - out["dropoff_longitude"])
        + np.abs(out["pickup_latitude"] - out["dropoff_latitude"])
    ).astype("float32")

    out["abs_lon_diff"] = np.abs(
        out["pickup_longitude"] - out["dropoff_longitude"]
    ).astype("float32")
    out["abs_lat_diff"] = np.abs(
        out["pickup_latitude"] - out["dropoff_latitude"]
    ).astype("float32")

    out["bearing"] = bearing_rad(
        out["pickup_latitude"].values,
        out["pickup_longitude"].values,
        out["dropoff_latitude"].values,
        out["dropoff_longitude"].values,
    ).astype("float32")

    out["abs_bearing"] = np.abs(out["bearing"]).astype("float32")

    JFK_LAT, JFK_LON = 40.6413, -73.7781
    LGA_LAT, LGA_LON = 40.7769, -73.8740
    out["pickup_to_jfk_km"] = haversine_km(
        out["pickup_latitude"].values,
        out["pickup_longitude"].values,
        JFK_LAT,
        JFK_LON,
    ).astype("float32")
    out["dropoff_to_jfk_km"] = haversine_km(
        out["dropoff_latitude"].values,
        out["dropoff_longitude"].values,
        JFK_LAT,
        JFK_LON,
    ).astype("float32")
    out["pickup_to_lga_km"] = haversine_km(
        out["pickup_latitude"].values,
        out["pickup_longitude"].values,
        LGA_LAT,
        LGA_LON,
    ).astype("float32")
    out["dropoff_to_lga_km"] = haversine_km(
        out["dropoff_latitude"].values,
        out["dropoff_longitude"].values,
        LGA_LAT,
        LGA_LON,
    ).astype("float32")

    out["passenger_count"] = (
        out["passenger_count"].clip(lower=0, upper=8).astype("int16")
    )
    return out


def clean_train(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out[out["fare_amount"].notna()]
    out = out[out["fare_amount"] > 0]
    out = out[out["fare_amount"] < 500]  # remove extreme outliers

    out = out[
        (out["pickup_longitude"].between(-74.5, -72.8))
        & (out["dropoff_longitude"].between(-74.5, -72.8))
        & (out["pickup_latitude"].between(40.5, 41.8))
        & (out["dropoff_latitude"].between(40.5, 41.8))
    ]

    out = out[
        ~(
            (out["pickup_longitude"] == out["dropoff_longitude"])
            & (out["pickup_latitude"] == out["dropoff_latitude"])
            & (out["fare_amount"] > 10)
        )
    ]
    return out


def replace_zero_coords_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    coord_cols = [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
    ]
    for c in coord_cols:
        out.loc[out[c] == 0, c] = np.nan
    return out


def invalid_or_oob_mask(df: pd.DataFrame) -> np.ndarray:
    coord_cols = [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
    ]
    any_nan = df[coord_cols].isna().any(axis=1).values
    oob = ~(
        df["pickup_longitude"].between(-74.5, -72.8)
        & df["dropoff_longitude"].between(-74.5, -72.8)
        & df["pickup_latitude"].between(40.5, 41.8)
        & df["dropoff_latitude"].between(40.5, 41.8)
    ).values
    return any_nan | oob


train = replace_zero_coords_with_nan(train)
test = replace_zero_coords_with_nan(test)

test_bad_mask = invalid_or_oob_mask(test)

train = clean_train(train)

baseline_fare_global = float(train["fare_amount"].median())
baseline_by_pax = (
    train.groupby(train["passenger_count"].clip(lower=0, upper=8))["fare_amount"]
    .median()
    .to_dict()
)
print("Baseline global median fare:", baseline_fare_global)
print("Invalid/OOB test rows:", int(test_bad_mask.sum()), "out of", len(test_bad_mask))

train_feat = add_features(train)
test_feat = add_features(test)

target = np.log1p(train_feat["fare_amount"].astype("float32"))
train_keys = train_feat["key"]
test_keys = test_feat["key"]

drop_cols = ["key", "pickup_datetime", "fare_amount"]
X = train_feat.drop(columns=[c for c in drop_cols if c in train_feat.columns])
X_test = test_feat.drop(
    columns=[c for c in ["key", "pickup_datetime"] if c in test_feat.columns]
)

print("Features:", list(X.columns))
print("Cleaned train shape:", X.shape)

del train, train_feat
gc.collect()



## === cell 3
X_train, X_valid, y_train, y_valid = train_test_split(
    X, target, test_size=0.1, random_state=RANDOM_STATE
)

model = lgbm.LGBMRegressor(
    objective="regression",
    n_estimators=1200,
    learning_rate=0.05,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    min_child_samples=30,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="rmse",
    callbacks=[lgbm.log_evaluation(period=200)],
)

valid_pred_log = model.predict(X_valid)
valid_pred = np.expm1(valid_pred_log)
valid_pred = np.clip(valid_pred, 0, None)

rmse = mean_squared_error(np.expm1(y_valid), valid_pred, squared=False)
print("Local validation RMSE (original scale):", rmse)

del X_train, X_valid, y_train, y_valid, valid_pred, valid_pred_log
gc.collect()



## === cell 4
model_full = lgbm.LGBMRegressor(
    objective="regression",
    n_estimators=1200,
    learning_rate=0.05,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    min_child_samples=30,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

model_full.fit(
    X,
    target,
    eval_metric="rmse",
    callbacks=[lgbm.log_evaluation(period=200)],
)

test_pred_log = model_full.predict(X_test)
test_pred = np.expm1(test_pred_log)
test_pred = np.clip(test_pred, 0, None)

if test_bad_mask.any():
    test_pred = test_pred.astype("float32", copy=False)
    pax = test_feat["passenger_count"].values
    fallback = np.array(
        [baseline_by_pax.get(int(p), baseline_fare_global) for p in pax],
        dtype="float32",
    )
    test_pred[test_bad_mask] = fallback[test_bad_mask]

submission = pd.DataFrame(
    {"key": test_keys.values, "fare_amount": test_pred.astype("float32")}
)
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print(submission.head())
print("Submission shape:", submission.shape)
