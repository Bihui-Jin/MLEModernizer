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

3.43884

# 6. Current score

5.1486

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.86445) has done: 'I fix the Pandas datetime feature extraction to work with Pandas 2.2 (replacing deprecated `.dt.week`/`.dt.weekofyear`), which currently stops feature generation and causes downstream KeyErrors. I also update the LightGBM training call to use the v4 callback-based early stopping API (since `early_stopping_rounds` is no longer a valid keyword), so the model actually trains. Finally, I ensure the submission file matches the required format (`key,fare_amount`) and is written as a `.csv` in the working directory, while keeping the model and feature logic otherwise unchanged.'
- What this solution (achieved 4.27813) has done: 'Your current gap is 4.86445 − 3.43884 = 1.42561 (about 41% worse than target), so we need a legitimate quality improvement while keeping the same LightGBM training approach. The biggest score issue is a bug in the haversine computation (missing squares in the formula), which makes the key distance feature incorrect and hurts RMSE; fixing it preserves the same feature idea but corrects the math. I also make the datetime parsing robust to the dataset’s actual format (it includes microseconds) to avoid silent `NaT` values, and I apply the same geographic cleaning bounds to the test set (without fare filter) to reduce distribution mismatch while keeping row order and restoring predictions for any filtered-out rows. These are minimal, directly score-relevant changes and still write a valid `submission.csv`.'
- What this solution (achieved 4.33371) has done: 'To move RMSE down toward the 3.43884 target with minimal disruption, I keep your exact feature set and LightGBM training loop, but improve data correctness and alignment where it directly affects RMSE. Specifically: (1) ensure datetime parsing is consistent and doesn’t silently produce `NaT` (which can create missing/garbage time features), (2) apply the same passenger_count validity filter to training that you already intended (your `clean_df()` currently ignores passenger_count), and (3) make the bearing calculation correct (it currently uses `sin(delta * cos(phi2))` instead of `sin(delta) * cos(phi2)`), which improves a core geometric feature without changing the approach. These are small, legitimate fixes that typically reduce error on this competition while preserving your model/feature logic and producing the same submission format.'
- What this solution (achieved 4.33371) has done: 'You’re currently above the target RMSE (4.33371 vs 3.43884, lower is better), so we need a modest real quality gain while keeping the same LightGBM approach and the same overall feature set idea. The biggest score drag left in this script is that you train with two leakage features (`fare_to_dist_ratio` and `fare_npassenger_to_dist_ratio`) that use `fare_amount` (the target) but are not available for test, and then you silently drop them by aligning to `test_df_valid.columns`; this mismatch harms training usefulness and generalization. I remove those target-derived features entirely (rather than partially using/dropping) so train/test feature generation is consistent, and I also ensure `clean_df()` doesn’t inadvertently depend on `fare_amount` when used for test-like frames. These are minimal, directly score-relevant fixes that preserve the same model/training loop and should move RMSE down toward the target.'
- What this solution (achieved 4.36969) has done: 'We need to reduce RMSE from 4.33371 toward 3.43884 (lower is better), so we make small, legitimate quality improvements without changing the overall LightGBM approach. The biggest gain with minimal disruption is to add a couple of standard NYC Taxi engineered features that are consistent with your existing geometry/time feature logic: absolute delta in lon/lat, Manhattan distance, and a simple “airport-ish” flag to help with long trips. We also add a light, RMSE-relevant data cleaning step to drop implausible coordinate pairs (0,0) and extremely tiny/huge trip distances in training only, which reduces label noise while keeping your core pipeline intact. Finally, we keep train/test feature alignment explicit and still produce the required `submission.csv`.'
- What this solution (achieved 4.63104) has done: 'Your pipeline doesn’t currently yield a Kaggle score because the environment errors out before grading; the safest way to move toward the target RMSE is to (a) make the run stable within the 600s constraint and (b) keep your existing feature/model logic intact while training on a cleaner, representative subset. I keep the same LightGBM training approach and the same engineered feature set, but switch to reading a deterministic random sample from the full `train.csv` (instead of the first N rows, which are distribution-shifted) using `skiprows`, which typically lowers RMSE without changing semantics. I also make the input paths robust to either `../input/*.csv` or `/kaggle/input/*.csv` so it runs in your provided filesystem, and I keep the submission writing exactly `submission.csv` with `key,fare_amount` aligned to the original test order.'
- What this solution (achieved 4.8101) has done: 'To move RMSE down toward the 3.43884 target (lower is better) with minimal disruption, I keep your exact LightGBM training loop and existing feature set, but fix one data/feature alignment issue that hurts generalization: you currently train on a subset sampled from `train.csv`, which is noisier than the already-provided cleaned `labels.csv`; switching the sampled source to `labels.csv` (same schema) typically reduces label noise and improves RMSE without changing the model approach. I also make datetime parsing explicitly `utc=True` to avoid occasional `NaT`/timezone inconsistencies that can degrade time features, while keeping the same extracted fields. Finally, I add LightGBM’s `extra_trees` and `lambda_l2` regularization knobs (same GBDT, same objective/metric) to modestly improve stability/fit without changing your architecture or training semantics, and still write a valid `submission.csv`.'
- What this solution (achieved 4.8101) has done: 'To reduce RMSE from 4.8101 toward the 3.43884 target (lower is better) with minimal disruption, I keep your exact LightGBM approach and feature set, but fix a key train/test mismatch: you currently clean/filter `test_df_valid` (dropping some rows) but keep `y` from the pre-cleaned train, which misaligns labels with features and harms model quality. I move the `y = train.fare_amount` extraction to *after* all training-side cleaning/feature engineering so `X` and `y` stay perfectly aligned. I also drop `fare_amount` from the training features explicitly (rather than relying on column alignment), which avoids accidental target leakage/column confusion and typically improves generalization RMSE without changing the core pipeline. Submission writing and the valid-row fill-back logic remain the same.'
- What this solution (achieved 4.90087) has done: 'To move RMSE down toward the 3.43884 target with minimal disruption, I keep your exact LightGBM training approach and feature set, but fix two correctness issues that directly hurt model quality. First, your distance-based trig features mistakenly use `sqrt(lon^2 * lat^2)` instead of the intended `sqrt(lon^2 + lat^2)` (consistent with `distance_travelled`), which injects noisy/incorrect signals; correcting this keeps the same feature idea but makes it mathematically consistent. Second, you clean/filter the training frame but only apply a haversine-range filter to train (not to the test-valid subset), creating a train/test distribution mismatch; applying the same haversine filter to `test_df_valid` (while keeping your “fill back” logic to preserve all test rows) typically reduces RMSE. The rest of the pipeline (sampling, cleaning, time features, model params, training loop, and submission writing) stays the same.'
- What this solution (achieved 4.89132) has done: 'To move RMSE down toward your 3.43884 target (lower is better) with minimal disruption, I’m keeping the exact LightGBM approach and your existing features, but I fix a train/test mismatch that’s currently injecting noise: you filter `test_df_valid` by haversine but still predict on it using all columns (including `pickup_datetime`), while training drops `pickup_datetime`—this inconsistency can hurt generalization. I also add two standard, very small, geometry-derived features (pickup/dropoff distance to NYC center) that fit your current “distance/time feature engineering” logic without changing the modeling approach. Finally, I make sure we align train/test columns *after* all feature creation and drop `pickup_datetime` from both train and test in the same place, preserving your “fill back” submission logic and output format.'
- What this solution (achieved 5.10927) has done: 'To move your RMSE down toward the 3.43884 target (lower is better) with minimal disruption, I’m keeping your LightGBM setup and feature pipeline but fixing a key train/test mismatch: you currently *drop rows from the test set* via `clean_df()` and the haversine filter, then fill missing predictions with a mean—this hurts RMSE because many test rows are valid and should be predicted directly. I keep all test rows and instead “clip” out-of-bound test coordinates to the same geographic window (so feature generation remains sane) while only filtering rows on the training side (to reduce noise without changing the learning approach). I also stop filtering `test_df_valid` by haversine (train-only cleaning is fine) and keep your column alignment and submission writing identical. These are small, directly score-relevant corrections that typically reduce RMSE without changing your model architecture or training loop.'
- What this solution (achieved 4.87297) has done: 'To move RMSE down toward the 3.43884 target from 5.10927 (lower is better) with minimal disruption, I keep your LightGBM training loop and the same feature ideas, but fix a key train/test mismatch: you currently filter training rows by haversine after feature creation but do not apply the same filter logic to test (you only clip), which leaves many test rows with extreme/near-zero distances that the model never saw during training. I compute a validity mask for test using the same haversine bounds, predict only for valid rows, and fill invalid rows with a stable fallback (the training mean fare) while preserving the original test order and submission format. I also clamp predictions to the same [0, 200] target range used in training cleaning, which typically reduces RMSE by avoiding absurd negative/huge outputs without changing the modeling approach. All changes are localized to the test-time post-processing and do not alter the model architecture, objective, or training semantics.'
- What this solution (achieved 4.85259) has done: 'To reduce RMSE toward your 3.43884 target (from 4.87297, lower is better) with minimal disruption, I keep the same LightGBM training loop and feature set, but make two score-relevant corrections. First, I stop treating “small haversine” test rows as invalid (those often include short NYC trips that are common in test), and only fallback-fill truly broken geo rows (zeros / out-of-bounds), so the model predicts for essentially all realistic test trips. Second, I add a simple but standard coordinate normalization by converting degree deltas into approximate kilometers (lat/lon scaling by cos(lat)), which preserves your distance-feature idea but makes it physically consistent and typically lowers RMSE without changing the modeling approach. Submission writing, paths, and the overall pipeline remain the same.'
- What this solution (achieved 5.1486) has done: 'To move RMSE down from 4.85259 toward the 3.43884 target (lower is better) with minimal disruption, I keep your exact LightGBM training approach and feature set, but fix two score-relevant train/test mismatches. First, your `test_valid_mask` is currently based on the *raw* `test_df` (pre-clipping), which incorrectly marks some rows invalid and forces mean-fare fallback; I compute the mask consistently from the same clipped frame you actually predict on, and only treat truly broken geo as invalid. Second, a few time features can become missing (NaT) and propagate NaNs into the model; I fill those time-derived columns with safe defaults after parsing, which typically reduces noise without changing the feature definitions. The submission writing and column alignment remain identical and it still produces `submission.csv`.'

# 9. Code solution

## === cell 0
import time

notebookstart = time.time()

import numpy as np
import pandas as pd
import gc
import os

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import metrics

import seaborn as sns
import matplotlib.pyplot as plt

Debug = False

NROWS = 2_000_000
if Debug is True:
    NROWS = 50_000


def _resolve_path(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of these paths exist: {candidates}")


train_path = _resolve_path(
    [
        "/kaggle/data/labels.csv",
        "/kaggle/input/labels.csv",
        "../input/labels.csv",
        "/kaggle/data/new-york-city-taxi-fare-prediction/labels.csv",
        "/kaggle/input/new-york-city-taxi-fare-prediction/labels.csv",
        "../input/new-york-city-taxi-fare-prediction/labels.csv",
        "/kaggle/data/train.csv",
        "/kaggle/input/train.csv",
        "../input/train.csv",
        "/kaggle/data/new-york-city-taxi-fare-prediction/train.csv",
        "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
        "../input/new-york-city-taxi-fare-prediction/train.csv",
    ]
)
test_path = _resolve_path(
    [
        "../input/test.csv",
        "/kaggle/input/test.csv",
        "/kaggle/data/test.csv",
        "../input/new-york-city-taxi-fare-prediction/test.csv",
        "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
        "/kaggle/data/new-york-city-taxi-fare-prediction/test.csv",
    ]
)


def _sample_train_csv(path, nrows, seed=23):
    try:
        with open(path, "rb") as f:
            n_lines = sum(1 for _ in f)
        n_data = max(0, n_lines - 1)
        if n_data <= nrows:
            return pd.read_csv(path, nrows=nrows, index_col="key")
        rng = np.random.RandomState(seed)
        keep = set(rng.choice(np.arange(1, n_data + 1), size=nrows, replace=False))
        skip = lambda i: (i != 0) and (i not in keep)
        return pd.read_csv(path, skiprows=skip, index_col="key")
    except Exception as e:
        print(
            "Sampling via skiprows failed; falling back to first N rows. Error:",
            repr(e),
        )
        return pd.read_csv(path, nrows=nrows, index_col="key")


train = _sample_train_csv(train_path, NROWS, seed=23)
train = train.dropna()

test_df = pd.read_csv(test_path, index_col="key")
testdex = test_df.index

gc.collect()



## === cell 1
print(
    "Percent of Training Set with Zero and Below Fair: ",
    round(
        (
            (
                train.loc[train["fare_amount"] <= 0, "fare_amount"].shape[0]
                / train.shape[0]
            )
            * 100
        ),
        5,
    ),
)
print(
    "Percent of Training Set 200 and Above Fair: ",
    round(
        (
            train.loc[train["fare_amount"] >= 200, "fare_amount"].shape[0]
            / train.shape[0]
        )
        * 100,
        5,
    ),
)
train = train.loc[(train["fare_amount"] > 0) & (train["fare_amount"] <= 200), :]

print(
    "\nPercent of Training Set with Zero and Below Passenger Count: ",
    round(
        (
            train.loc[train["passenger_count"] <= 0, "passenger_count"].shape[0]
            / train.shape[0]
        )
        * 100,
        5,
    ),
)
print(
    "Percent of Training Set with Nine and Above Passenger Count: ",
    round(
        (
            train.loc[train["passenger_count"] >= 9, "passenger_count"].shape[0]
            / train.shape[0]
        )
        * 100,
        5,
    ),
)
train = train.loc[(train["passenger_count"] > 0) & (train["passenger_count"] <= 9), :]




## === cell 2
def clean_df(df):
    cond = (
        (df.passenger_count > 0)
        & (df.passenger_count <= 9)
        & (df.pickup_longitude > -80)
        & (df.pickup_longitude < -70)
        & (df.pickup_latitude > 35)
        & (df.pickup_latitude < 45)
        & (df.dropoff_longitude > -80)
        & (df.dropoff_longitude < -70)
        & (df.dropoff_latitude > 35)
        & (df.dropoff_latitude < 45)
    )

    cond = cond & ~(
        ((df.pickup_longitude == 0) & (df.pickup_latitude == 0))
        | ((df.dropoff_longitude == 0) & (df.dropoff_latitude == 0))
    )

    if "fare_amount" in df.columns:
        cond = cond & (df.fare_amount > 0) & (df.fare_amount <= 200)
    return df[cond]


def clip_test_geo(df):
    df = df.copy()
    df["passenger_count"] = df["passenger_count"].clip(lower=1, upper=9)

    df["pickup_longitude"] = df["pickup_longitude"].clip(lower=-80, upper=-70)
    df["dropoff_longitude"] = df["dropoff_longitude"].clip(lower=-80, upper=-70)
    df["pickup_latitude"] = df["pickup_latitude"].clip(lower=35, upper=45)
    df["dropoff_latitude"] = df["dropoff_latitude"].clip(lower=35, upper=45)

    nyc_center = (-73.985428, 40.748817)
    pu_zero = (df["pickup_longitude"] == 0) & (df["pickup_latitude"] == 0)
    do_zero = (df["dropoff_longitude"] == 0) & (df["dropoff_latitude"] == 0)
    if pu_zero.any():
        df.loc[pu_zero, "pickup_longitude"] = nyc_center[0]
        df.loc[pu_zero, "pickup_latitude"] = nyc_center[1]
    if do_zero.any():
        df.loc[do_zero, "dropoff_longitude"] = nyc_center[0]
        df.loc[do_zero, "dropoff_latitude"] = nyc_center[1]
    return df




## === cell 3
def prepare_distance_features(df):
    df["longitude_distance"] = abs(df["pickup_longitude"] - df["dropoff_longitude"])
    df["latitude_distance"] = abs(df["pickup_latitude"] - df["dropoff_latitude"])

    lat_km_per_deg = 111.32
    mean_lat_rad = np.radians((df["pickup_latitude"] + df["dropoff_latitude"]) / 2.0)
    lon_km_per_deg = 111.32 * np.cos(mean_lat_rad)

    df["longitude_km"] = df["longitude_distance"] * lon_km_per_deg
    df["latitude_km"] = df["latitude_distance"] * lat_km_per_deg

    df["distance_travelled"] = (
        df["longitude_distance"] ** 2 + df["latitude_distance"] ** 2
    ) ** 0.5
    df["distance_travelled_km"] = (
        df["longitude_km"] ** 2 + df["latitude_km"] ** 2
    ) ** 0.5

    _euclid = (df["longitude_distance"] ** 2 + df["latitude_distance"] ** 2) ** 0.5
    df["distance_travelled_sin"] = np.sin(_euclid)
    df["distance_travelled_cos"] = np.cos(_euclid)
    df["distance_travelled_sin_sqrd"] = np.sin(_euclid) ** 2
    df["distance_travelled_cos_sqrd"] = np.cos(_euclid) ** 2

    R_km = 6371.0  # Kilometers
    phi1 = np.radians(df["pickup_latitude"])
    phi2 = np.radians(df["dropoff_latitude"])
    phi_chg = np.radians(df["pickup_latitude"] - df["dropoff_latitude"])
    delta_chg = np.radians(df["pickup_longitude"] - df["dropoff_longitude"])

    a = (np.sin(phi_chg / 2) ** 2) + np.cos(phi1) * np.cos(phi2) * (
        np.sin(delta_chg / 2) ** 2
    )
    c = 2 * np.arctan2(a**0.5, (1 - a) ** 0.5)
    d_km = R_km * c
    df["haversine"] = d_km

    y = np.sin(delta_chg) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_chg)
    df["bearing"] = np.arctan2(y, x)

    df["manhattan_distance"] = df["longitude_distance"] + df["latitude_distance"]
    df["lon_lat_sum"] = df["longitude_distance"] + df["latitude_distance"]
    df["lon_lat_diff"] = df["longitude_distance"] - df["latitude_distance"]

    df["manhattan_km"] = df["longitude_km"] + df["latitude_km"]

    return df


def prepare_time_features(df):
    if df["pickup_datetime"].dtype != "datetime64[ns]":
        s = df["pickup_datetime"].astype(str).str.replace(" UTC", "", regex=False)
        df["pickup_datetime"] = pd.to_datetime(
            s, errors="coerce", utc=True
        ).dt.tz_convert(None)

    df["hour_of_day"] = df.pickup_datetime.dt.hour
    df["month"] = df.pickup_datetime.dt.month
    df["day_of_year"] = df.pickup_datetime.dt.dayofyear

    iso = df.pickup_datetime.dt.isocalendar()
    df["week"] = iso.week.astype("int16")
    df["week_of_year"] = iso.week.astype("int16")

    df["Weekday"] = df.pickup_datetime.dt.weekday
    df["Quarter"] = df.pickup_datetime.dt.quarter
    df["Day of Month"] = df.pickup_datetime.dt.day

    df["is_weekend"] = (df["Weekday"] >= 5).astype("int8")

    time_cols = [
        "hour_of_day",
        "month",
        "day_of_year",
        "week",
        "week_of_year",
        "Weekday",
        "Quarter",
        "Day of Month",
        "is_weekend",
    ]
    for c in time_cols:
        if c in df.columns:
            if df[c].dtype.kind in ("f", "i", "u"):
                df[c] = df[c].fillna(0)
            else:
                df[c] = df[c].fillna(0)

    return df




## === cell 4
train = clean_df(train)

test_df_valid = clip_test_geo(test_df)

train = prepare_distance_features(train)
test_df_valid = prepare_distance_features(test_df_valid)

train = prepare_time_features(train)
test_df_valid = prepare_time_features(test_df_valid)

train_haversine_low = 0.05
train_haversine_high = 200.0
if "haversine" in train.columns:
    train = train.loc[
        (train["haversine"] >= train_haversine_low)
        & (train["haversine"] <= train_haversine_high)
    ].copy()

broken_geo = (
    (test_df_valid["pickup_longitude"] == 0) & (test_df_valid["pickup_latitude"] == 0)
) | (
    (test_df_valid["dropoff_longitude"] == 0) & (test_df_valid["dropoff_latitude"] == 0)
)
test_valid_mask = ~broken_geo


def add_airport_flags(df):
    jfk = (-73.7781, 40.6413)
    lga = (-73.8740, 40.7769)
    ewr = (-74.1745, 40.6895)

    def approx_deg_dist(lon, lat, center):
        return np.sqrt((lon - center[0]) ** 2 + (lat - center[1]) ** 2)

    pu_jfk = approx_deg_dist(df["pickup_longitude"], df["pickup_latitude"], jfk)
    do_jfk = approx_deg_dist(df["dropoff_longitude"], df["dropoff_latitude"], jfk)
    pu_lga = approx_deg_dist(df["pickup_longitude"], df["pickup_latitude"], lga)
    do_lga = approx_deg_dist(df["dropoff_longitude"], df["dropoff_latitude"], lga)
    pu_ewr = approx_deg_dist(df["pickup_longitude"], df["pickup_latitude"], ewr)
    do_ewr = approx_deg_dist(df["dropoff_longitude"], df["dropoff_latitude"], ewr)

    df["near_airport"] = (
        (pu_jfk < 0.05)
        | (do_jfk < 0.05)
        | (pu_lga < 0.05)
        | (do_lga < 0.05)
        | (pu_ewr < 0.05)
        | (do_ewr < 0.05)
    ).astype("int8")
    return df


train = add_airport_flags(train)
test_df_valid = add_airport_flags(test_df_valid)


def add_center_dist(df):
    nyc_center = (-73.985428, 40.748817)  # approx Midtown Manhattan
    df["pickup_to_center"] = np.sqrt(
        (df["pickup_longitude"] - nyc_center[0]) ** 2
        + (df["pickup_latitude"] - nyc_center[1]) ** 2
    )
    df["dropoff_to_center"] = np.sqrt(
        (df["dropoff_longitude"] - nyc_center[0]) ** 2
        + (df["dropoff_latitude"] - nyc_center[1]) ** 2
    )
    return df


train = add_center_dist(train)
test_df_valid = add_center_dist(test_df_valid)

gc.collect()



## === cell 5
try:
    f, ax = plt.subplots(1, 2, figsize=[10, 5])
    sns.countplot(x=train["passenger_count"], ax=ax[0])
    sns.countplot(x=test_df["passenger_count"], ax=ax[1])
    ax[0].set_title("Train Set - Passenger Count")
    ax[1].set_title("Test Set - Passenger Count")
    plt.tight_layout()
    plt.close()
except Exception as e:
    print("Plotting skipped:", repr(e))



## === cell 6
try:
    f, ax = plt.subplots(figsize=[6, 5])
    sns.kdeplot(train["fare_amount"], ax=ax)
    ax.set_title("Fare Distribution")
    plt.tight_layout()
    plt.close()
except Exception as e:
    print("Plotting skipped:", repr(e))




## === cell 7
def time_slicer(df, timeframes, value, color="purple"):
    """
    Function to count observation occurrence through different lenses of time.
    """
    f, ax = plt.subplots(len(timeframes), figsize=[12, 10])
    if len(timeframes) == 1:
        ax = [ax]
    for i, x in enumerate(timeframes):
        df.loc[:, [x, value]].groupby([x]).mean().plot(ax=ax[i], color=color)
        ax[i].set_ylabel(value.replace("_", " ").title())
        ax[i].set_title(
            "{} by {}".format(
                value.replace("_", " ").title(), x.replace("_", " ").title()
            )
        )
        ax[i].set_xlabel("")
    ax[len(timeframes) - 1].set_xlabel("Time Frame")
    plt.tight_layout(pad=0)
    plt.close()




## === cell 8
try:
    time_slicer(
        df=train,
        timeframes=["day_of_year", "month", "Day of Month", "week", "hour_of_day"],
        value="fare_amount",
        color="blue",
    )
except Exception as e:
    print("time_slicer skipped:", repr(e))



## === cell 9
try:
    time_slicer(
        df=train,
        timeframes=["day_of_year", "month", "Day of Month", "week", "hour_of_day"],
        value="distance_travelled",
        color="green",
    )
except Exception as e:
    print("time_slicer skipped:", repr(e))



## === cell 10
try:
    pass
except Exception as e:
    print("time_slicer skipped:", repr(e))



## === cell 11
try:
    pass
except Exception as e:
    print("time_slicer skipped:", repr(e))



## === cell 12
y = train["fare_amount"].copy()
train = train.drop(columns=["fare_amount"])

if "pickup_datetime" in test_df_valid.columns:
    test_df_valid = test_df_valid.drop(columns=["pickup_datetime"])
if "pickup_datetime" in train.columns:
    train = train.drop(columns=["pickup_datetime"])

train = train[test_df_valid.columns]
print(
    "Does Train feature equal test feature?: ",
    all(train.columns == test_df_valid.columns),
)

X_train, X_val, y_train, y_val = train_test_split(
    train, y, test_size=0.1, random_state=23
)

dtrain = lgb.Dataset(X_train, label=y_train)
dvalid = lgb.Dataset(X_val, label=y_val)



## === cell 13
print("Light Gradient Boosting Regressor: ")
lgbm_params = {
    "task": "train",
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": 23,
    "feature_fraction_seed": 23,
    "bagging_seed": 23,
    "extra_trees": True,
    "lambda_l2": 0.2,
}



## === cell 14
modelstart = time.time()
lgb_reg = lgb.train(
    lgbm_params,
    dtrain,
    num_boost_round=500,
    valid_sets=[dtrain, dvalid],
    valid_names=["train", "valid"],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100),
    ],
)

print("Model Evaluation Stage")
val_pred = lgb_reg.predict(X_val, num_iteration=lgb_reg.best_iteration)
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_val, val_pred)))



## === cell 15
try:
    f, ax = plt.subplots(figsize=[7, 10])
    lgb.plot_importance(lgb_reg, max_num_features=50, ax=ax)
    plt.title("Light GBM Feature Importance")
    plt.tight_layout()
    plt.savefig("feature_import.png")
    plt.close()
except Exception as e:
    print("Feature importance plot skipped:", repr(e))



## === cell 16
train_mean_fare = float(y.mean())

lgpred_full = np.full(
    shape=(test_df_valid.shape[0],), fill_value=train_mean_fare, dtype=np.float64
)

if test_valid_mask is None:
    test_valid_mask = pd.Series(True, index=test_df_valid.index)

if test_valid_mask.any():
    lgpred_full[test_valid_mask.values] = lgb_reg.predict(
        test_df_valid.loc[test_valid_mask, :],
        num_iteration=lgb_reg.best_iteration,
    )

lgpred_full = np.clip(lgpred_full, 0.0, 200.0)

submission = pd.DataFrame({"key": testdex, "fare_amount": lgpred_full})
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print("Model Runtime: %0.2f Minutes" % ((time.time() - modelstart) / 60))
print("Notebook Runtime: %0.2f Minutes" % ((time.time() - notebookstart) / 60))

submission.head()
