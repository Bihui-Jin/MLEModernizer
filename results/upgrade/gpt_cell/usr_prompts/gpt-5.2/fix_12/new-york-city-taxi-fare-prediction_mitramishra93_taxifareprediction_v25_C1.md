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

3.40466

# 6. Current score

5.09343

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 10.50219) has done: 'Most of the timeout comes from doing expensive full-DataFrame sorts, repeated DataFrame filtering/dropping, and especially `DataFrame.apply(axis=1)` (row-wise Python loops) on up to 1M rows. I keep the exact same feature engineering, cleaning rules, and RandomForest training/prediction logic, but replace slow row-wise `apply` with vectorized NumPy math, combine multiple row drops into boolean masks (same semantics), and remove/avoid operations that compute huge intermediate tables but are not used for the model (plots/sorts). I also speed up CSV parsing with explicit dtypes and fast datetime parsing while preserving the same values, and ensure the RandomForest uses all CPU cores (`n_jobs=-1`) without changing the model type or training approach.'
- What this solution (achieved 17.70984) has done: 'Your current score (10.50 RMSE) is far from the target (3.40), so we should improve predictive signal with minimal, metric-aligned fixes while keeping the same overall pipeline (feature engineering + RandomForestRegressor). The biggest issue is that the model is using raw coordinates without strong geographic priors, and it also trains on many noisy/outlier trips (outside NYC area) which inflates RMSE. I add two lightweight, standard taxi-fare features (absolute lat/lon deltas and Manhattan distance) and apply a conservative NYC bounding-box filter on the training set only; this preserves the same model type and training loop but meaningfully reduces label noise. I also clip negative predictions to 0 to avoid obvious RMSE penalties from impossible fares, while keeping the same submission format and paths.'
- What this solution (achieved 8.30971) has done: 'Your current RMSE (17.71) is far worse than the target (3.40), so we should improve predictive signal with minimal, metric-aligned fixes while keeping the same RandomForest approach and existing feature set. The biggest remaining issue is that `S_Distance` is being overwritten for “high distance” rows using a fare-derived formula, which injects label information into a feature during training and then cannot be reproduced at test time—this train/test feature mismatch typically hurts generalization badly. I keep your `S_Distance` computation and all downstream features, but remove the label-derived overwrite and instead only filter extreme/unrealistic distances/fare outliers in training (a standard, conservative cleanup for this competition). This preserves your core logic (same model type, same training call, same feature engineering) while making train/test feature distributions consistent, which should move RMSE substantially toward the target.'
- What this solution (achieved 4.82632) has done: 'The timeout is dominated by fitting a 200-tree RandomForest on 3,000,000 rows, which is far beyond what finish in 600s on typical Kaggle CPU. The fastest equivalent fix is to keep the same model/parameters but train on a much smaller, deterministic subset (this does not change the algorithm; it only makes the run feasible). In addition, I reduce overhead by (a) filtering/feature-engineering in a more vectorized way, (b) avoiding repeated pandas-to-numpy conversions, and (c) ensuring all features are contiguous `float32` arrays before fitting to minimize conversion/copy costs inside scikit-learn. These changes preserve evaluation semantics and stability (fixed seed) while removing the main runtime bottleneck.'
- What this solution (achieved 4.79964) has done: 'Your current RMSE (4.82632) is worse than the target (3.40466), so we should improve signal with minimal, safe changes while keeping the same RandomForest and feature pipeline. The biggest low-risk gain is to train on a larger (but still runnable) deterministic sample and to clean a couple of obvious label/coordinate outliers more consistently (both reduce noise and typically improve RMSE). I also ensure train/test feature columns are aligned identically before converting to NumPy (prevents silent column-order mismatch issues) and keep the same prediction post-processing and submission format. These changes preserve your core logic (same model, same engineered features, same training call) and should move RMSE toward the target band.'
- What this solution (achieved 4.7949) has done: 'You’re currently worse than the target (4.80 vs 3.40 RMSE, lower is better), so we should improve generalization with minimal, safe changes while keeping your RandomForest + current features intact. The biggest low-risk gain is to (1) add two standard, computation-light geospatial features (bearing and euclidean degree distance) that complement your existing Haversine and manhattan deltas, and (2) slightly increase training sample size while staying within the 600s budget thanks to sklearnex + vectorized feature engineering. I also add a conservative passenger_count >= 1 filter (common noise source) and keep your NYC bbox + outlier filters and the same submission filling logic for non-NYC test rows. These changes preserve the core model/training approach and evaluation semantics while typically reducing RMSE for this competition.'
- What this solution (achieved 5.09185) has done: 'You’re still outside the target band (4.7949 vs 3.40466 RMSE; lower is better), so we should improve generalization with minimal, metric-aligned tweaks while keeping the same RandomForest training approach and feature pipeline. The biggest low-risk gain is adding one standard taxi-fare feature (straight-line distance in kilometers) derived from your existing Haversine computation, and slightly tightening a couple of clearly noisy training cases (very small distance but non-minimum fare, and unrealistically high $/km) without changing how test rows are handled. These changes preserve your core logic (same model type, same fit/predict flow, same existing features) and typically reduce RMSE by reducing label noise and giving the model a more linear distance signal. Submission format, paths, and the “fill non-NYC test rows with a default fare” behavior are kept identical.'
- What this solution (achieved 5.09343) has done: 'Your current RMSE (5.09185) is worse than the target (3.40466), so we should improve generalization with the smallest safe changes while keeping the same RandomForest + existing feature engineering. The lowest-risk gain here is to remove a remaining major noise source: erroneous `passenger_count==0` rows in the test set currently get extreme/garbage predictions, and they are common in this competition; we set them to 1 for both train/test to match typical data cleaning without changing the model logic. Next, we add one more conservative, standard training-only filter that usually reduces RMSE: remove trips with very small distances but fares far above plausible minimum fare, using a slightly tighter threshold than before (still consistent with your existing near-zero/rate filters). Everything else (features, model, fit/predict flow, submission schema/paths) stays the same and it still writes `submission_1.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

try:
    from sklearnex import patch_sklearn  # type: ignore

    patch_sklearn()
except Exception:
    pass

from sklearn.ensemble import RandomForestRegressor

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print(os.listdir("../input"))



## === cell 1
train_dtypes = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}
test_dtypes = {
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}

NROWS_TRAIN = 1_500_000

train_df = pd.read_csv(
    "../input/train.csv",
    nrows=NROWS_TRAIN,
    dtype=train_dtypes,
    parse_dates=["pickup_datetime"],
)
test_df = pd.read_csv(
    "../input/test.csv",
    dtype=test_dtypes,
    parse_dates=["pickup_datetime"],
)



## === cell 2
_ = (train_df.shape, test_df.shape)



## === cell 3
train_df.dropna(axis=0, how="any", inplace=True)



## === cell 4
for df in (train_df, test_df):
    if "passenger_count" in df.columns:
        pc = df["passenger_count"].to_numpy(copy=False)
        if (pc == 0).any():
            df.loc[pc == 0, "passenger_count"] = np.uint8(1)

fare = train_df["fare_amount"].to_numpy()
pc = train_df["passenger_count"].to_numpy()
pl = train_df["pickup_latitude"].to_numpy()
p_lon = train_df["pickup_longitude"].to_numpy()

mask = (
    (fare >= 0)
    & (pc >= 1)
    & (pc <= 6)
    & (pl >= -90)
    & (pl <= 90)
    & (p_lon >= -180)
    & (p_lon <= 180)
)
train_df = train_df.loc[mask]



## === cell 5
for df in (train_df, test_df):
    dt = df["pickup_datetime"].dt
    df["date"] = dt.day.astype("uint8")
    df["month"] = dt.month.astype("uint8")
    df["day_of_week"] = dt.dayofweek.astype("uint8")
    df["hour"] = dt.hour.astype("uint8")
    df["year"] = dt.year.astype("uint16")




## === cell 6
def add_sphere_distance(df):
    R = np.float32(6367.0)
    lat1 = np.radians(df["pickup_latitude"].to_numpy(dtype=np.float64, copy=False))
    lat2 = np.radians(df["dropoff_latitude"].to_numpy(dtype=np.float64, copy=False))
    dlat = lat2 - lat1
    dlon = np.radians(
        df["dropoff_longitude"].to_numpy(dtype=np.float64, copy=False)
        - df["pickup_longitude"].to_numpy(dtype=np.float64, copy=False)
    )
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    df["S_Distance"] = (R * c).astype("float32", copy=False)


add_sphere_distance(train_df)
add_sphere_distance(test_df)



## === cell 7
mask_bad = (
    (train_df["pickup_latitude"].to_numpy() == 0)
    & (train_df["pickup_longitude"].to_numpy() == 0)
    & (train_df["dropoff_latitude"].to_numpy() != 0)
    & (train_df["dropoff_longitude"].to_numpy() != 0)
    & (train_df["fare_amount"].to_numpy() == 0)
)
if mask_bad.any():
    train_df = train_df.loc[~mask_bad]

mask_zero = (train_df["S_Distance"].to_numpy() == 0) & (
    train_df["fare_amount"].to_numpy() == 0
)
if mask_zero.any():
    train_df = train_df.loc[~mask_zero]

rush_hour_mask = (
    (train_df["hour"].to_numpy() >= 6)
    & (train_df["hour"].to_numpy() <= 20)
    & (train_df["day_of_week"].to_numpy() >= 1)
    & (train_df["day_of_week"].to_numpy() <= 5)
    & (train_df["S_Distance"].to_numpy() == 0)
    & (train_df["fare_amount"].to_numpy() < 2.5)
)
if rush_hour_mask.any():
    train_df = train_df.loc[~rush_hour_mask]



## === cell 8
nyc_mask_train = (
    train_df["pickup_longitude"].between(-74.3, -73.6)
    & train_df["dropoff_longitude"].between(-74.3, -73.6)
    & train_df["pickup_latitude"].between(40.4, 41.0)
    & train_df["dropoff_latitude"].between(40.4, 41.0)
)
train_df = train_df.loc[nyc_mask_train]

nyc_mask_test = (
    test_df["pickup_longitude"].between(-74.3, -73.6)
    & test_df["dropoff_longitude"].between(-74.3, -73.6)
    & test_df["pickup_latitude"].between(40.4, 41.0)
    & test_df["dropoff_latitude"].between(40.4, 41.0)
)



## === cell 9
train_df = train_df.loc[
    (train_df["fare_amount"] >= 2.5)
    & (train_df["fare_amount"] <= 250.0)
    & (train_df["S_Distance"] >= 0.0)
    & (train_df["S_Distance"] <= 100.0)
]

sd = train_df["S_Distance"].to_numpy(dtype=np.float32, copy=False)
fa = train_df["fare_amount"].to_numpy(dtype=np.float32, copy=False)

near_zero_bad = (sd < np.float32(0.3)) & (fa > np.float32(12.0))

rate = fa / np.maximum(sd, np.float32(1e-3))
rate_bad = rate > np.float32(50.0)

if (near_zero_bad | rate_bad).any():
    train_df = train_df.loc[~(near_zero_bad | rate_bad)]



## === cell 10
for df in (train_df, test_df):
    p_lon = df["pickup_longitude"].to_numpy(dtype=np.float32, copy=False)
    d_lon = df["dropoff_longitude"].to_numpy(dtype=np.float32, copy=False)
    p_lat = df["pickup_latitude"].to_numpy(dtype=np.float32, copy=False)
    d_lat = df["dropoff_latitude"].to_numpy(dtype=np.float32, copy=False)

    dlon = (p_lon - d_lon).astype("float32", copy=False)
    dlat = (p_lat - d_lat).astype("float32", copy=False)

    abs_dlon = np.abs(dlon, dtype=np.float32)
    abs_dlat = np.abs(dlat, dtype=np.float32)

    df["abs_dlon"] = abs_dlon
    df["abs_dlat"] = abs_dlat
    df["manhattan_dist"] = (abs_dlon + abs_dlat).astype("float32", copy=False)

    df["euclidean_deg_dist"] = np.sqrt(
        abs_dlon * abs_dlon + abs_dlat * abs_dlat
    ).astype("float32", copy=False)

    lat1 = np.radians(p_lat.astype(np.float64, copy=False))
    lat2 = np.radians(d_lat.astype(np.float64, copy=False))
    dlon_rad = np.radians((d_lon - p_lon).astype(np.float64, copy=False))
    y = np.sin(dlon_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon_rad)
    df["bearing"] = np.arctan2(y, x).astype("float32", copy=False)

    df["distance_km"] = df["S_Distance"].astype("float32", copy=False)



## === cell 11
test_keys = test_df["key"].copy()

train_df = train_df.drop(["key", "pickup_datetime"], axis=1)
test_features = test_df.drop(["key", "pickup_datetime"], axis=1)

x_train_df = train_df.loc[:, train_df.columns != "fare_amount"]
y_train = train_df["fare_amount"].to_numpy()

x_train_df = x_train_df.sort_index(axis=1)
test_features = test_features.reindex(columns=x_train_df.columns)

x_train = np.ascontiguousarray(x_train_df.to_numpy(dtype=np.float32, copy=False))
x_test = np.ascontiguousarray(test_features.to_numpy(dtype=np.float32, copy=False))



## === cell 12
rg = RandomForestRegressor(
    n_estimators=200,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=RANDOM_STATE,
)
rg.fit(x_train, y_train)
y_predict = rg.predict(x_test)
y_predict = np.clip(y_predict, 0.0, None).astype("float32")



## === cell 13
submission = pd.read_csv("../input/sample_submission.csv")

default_fare = float(np.median(y_train))
pred_full = pd.Series(y_predict, index=test_df.index, dtype="float32")
pred_full.loc[~nyc_mask_test] = np.float32(default_fare)

submission["fare_amount"] = pred_full.values.astype("float32")
submission.to_csv("submission_1.csv", index=False)
submission.head(10)
