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

3.44303

# 6. Current score

8.69498

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.29803) has done: 'I fix the XGBoost inference bug caused by using the deprecated `ntree_limit/model.best_ntree_limit` API in xgboost 2.x by switching to `iteration_range` (and falling back safely if early stopping isn’t set). I also update the objective from deprecated `reg:linear` to `reg:squarederror` (same regression semantics) to avoid warnings and ensure stable training. Finally, I make sure the script always writes a valid `submission.csv` with the required `key,fare_amount` columns in the working directory using the existing pipeline and features.'
- What this solution (achieved 7.88196) has done: 'You don’t yet have a valid Kaggle score, so the most likely blocker is that `clean_df(test_df)` can drop test rows, producing a submission with fewer rows than `sample_submission.csv` (Kaggle reject it). I keep your exact feature engineering and XGBoost training logic, but change test-time handling to never filter out rows; instead, I clip out-of-range coordinates/passenger counts to safe bounds (so every `key` remains) and fill any remaining missing values. I also align the submission to `sample_submission.csv` keys to guarantee correct order/rowcount, and I make a small, safe training tweak by using more boosting rounds with early stopping (same approach, typically improves RMSE toward your target without changing core logic). These changes should produce a valid `submission.csv` and move RMSE downward from your prior ~6.3 toward the 3.44 target.'
- What this solution (achieved 6.33061) has done: 'Your current RMSE (7.88) is far worse than the target (3.44), so we should make a small, legitimate improvement without changing the overall XGBoost approach. The biggest issue is that the model is likely being trained on untransformed raw coordinates/time without enough structure for NYC taxi fares; we can keep the same feature engineering but add two standard, minimal extensions that usually cut RMSE a lot: (1) add a few simple geospatial/time features (absolute deltas, haversine distance in km already exists, plus a rough Manhattan-distance proxy), and (2) use a slightly more appropriate XGBoost configuration for this problem (still `xgb.train` regression, same loss/metric) by adding conservative tree/regularization defaults and training on a larger sample (still bounded to run under the time limit). We also keep your “never drop test rows” logic and ensure the submission is aligned to `sample_submission.csv` keys and has exactly the required columns.'
- What this solution (achieved 6.06716) has done: 'We make the smallest changes likely to reduce RMSE toward 3.44 without changing your overall XGBoost regression approach: (1) use a validation split that is less noisy by making it deterministic and (2) slightly tighten the tree complexity/regularization so the model generalizes better on this classic dataset. We also ensure train/test feature columns are perfectly aligned (same order and same set) to avoid subtle DMatrix column mismatches that can hurt score. Finally, we add a light, standard label cleaning step (dropping obviously wrong coordinates already exists; we also remove extreme-distance outliers) which typically improves RMSE materially while keeping core logic intact.'
- What this solution (achieved 6.35244) has done: 'Your RMSE (6.067) is still far worse than the 3.443 target (lower is better), so we should make a small, legitimate generalization improvement without changing the XGBoost approach or feature set. The biggest low-risk win here is making the train/validation split more representative: instead of a random split that mixes many “easy” short trips across both sets (and can bias early stopping), we split by time (using `pickup_datetime`) so validation better matches the future-like distribution Kaggle tests on. To keep core logic intact, we keep the same cleaning, same features, same `xgb.train` call and parameters, but we only change how we create the validation set and we keep column alignment and submission generation exactly as before. This typically reduces public RMSE materially for this competition while staying within constraints and runtime.'
- What this solution (achieved 5.34605) has done: 'We keep your exact XGBoost regression setup and existing feature engineering, but make one small, metric-aligned correction: train the model on `log1p(fare_amount)` and then `expm1` the predictions back to dollars. This keeps the same objective/loss (RMSE on the trained target) while typically reducing RMSE on raw fares because it down-weights large-fare outliers that otherwise dominate squared error. We also clip negative/implausibly large predictions to the same [0, 250] range you already enforce in training labels, which is a minimal post-processing step consistent with your label cleaning. Everything else (data reading, cleaning, time split, distance/time features, XGBoost training loop, submission alignment) stays the same and still produces `submission.csv`.'
- What this solution (achieved 4.9166) has done: 'Your current RMSE (5.346) is still far above the target (3.443), so we should make the smallest changes that reliably improve generalization without changing the overall XGBoost regression pipeline. The biggest low-risk gain here is to tighten training-data cleaning to remove known-bad records that inflate RMSE (invalid lat/lon zeros, extreme/implausible fares, and extreme geodesic distances) while keeping the same feature set and model. I also add one minimal but impactful NYC-specific feature (pickup/dropoff distance to JFK/LGA) which is standard for this competition and doesn’t change the core approach (still tabular features + XGBoost). Finally, I keep your existing time-based split, log1p target, inference iteration_range fix, and submission alignment exactly as before.'
- What this solution (achieved 5.34373) has done: 'We keep your exact XGBoost/log1p pipeline and feature set, but make two minimal, high-impact generalization fixes to move RMSE down toward the 3.443 target: (1) add standard NYC bounding-box cleaning (tighter than the current broad [-80,-70]/[35,45]) plus passenger_count and distance sanity checks to remove obvious junk rows that inflate RMSE, and (2) add one very common, low-risk feature for this competition: the haversine distance from pickup/dropoff to the NYC center (captures “within-Manhattan vs outer boroughs” effects). We also make sure any datetime-derived columns are fully numeric and have no NaNs in both train and test, and keep your submission alignment against `sample_submission.csv` unchanged. These changes preserve the core logic (same model, same training loop/early stopping, same loss, same features style) while typically reducing RMSE meaningfully from ~4.92 toward your target band.'
- What this solution (achieved 5.1835) has done: 'To move RMSE down toward your 3.443 target with minimal disruption, I keep your exact XGBoost + log1p pipeline and feature set, but fix two issues that typically hurt this competition: (1) `pickup_datetime` is currently parsed twice and your time-based split uses a broken `_pickup_dt` (it’s computed after you overwrite the column with engineered features and becomes all-NaT, making the split effectively arbitrary), and (2) the model benefits from a couple of standard, low-risk time features (year + dayofyear) derived from the same timestamp without changing the overall approach. I also align train/test datetime parsing to UTC-naive consistently and keep submission creation unchanged. These are small, metric-aligned corrections that should improve generalization and reduce RMSE from ~5.34 toward the target band.'
- What this solution (achieved 5.1835) has done: 'We’re still far above the target RMSE (5.18 vs 3.44, lower is better), so the smallest likely win is to fix a subtle train/test feature mismatch: your time features are derived from `pickup_datetime`, but in test you parse `pickup_datetime` once in `sanitize_test_df` and then parse again inside `add_datetime_info`, which can introduce extra NaTs and noisier splits/features. I make datetime parsing happen exactly once (in `add_datetime_info`) and remove the duplicate parse from `sanitize_test_df`, keeping the same feature set and model. I also ensure train and test have consistent datetime-derived integer features by filling any NaT-derived values *before* extracting `.dt` fields, which reduces random zeroing and typically improves generalization without changing the core approach. Everything else (cleaning logic, feature engineering, log1p target, XGBoost training/early stopping, submission alignment) remains the same and still writes a valid `submission.csv`.'
- What this solution (achieved 5.50259) has done: 'Your RMSE (5.1835) is still well above the target (3.44303, lower is better), so we should make a small, legitimate generalization improvement while keeping the same XGBoost+log1p pipeline and feature style. The biggest low-risk win here is correcting the temporal validation setup: we keep the time-based split, but add a small “gap” between train and validation periods to reduce leakage/near-duplicate effects that can mislead early stopping and hurt test RMSE. Second, we add one standard, minimal geospatial feature that commonly improves NYC taxi fares without changing the approach: a coarse “bearing” (direction) between pickup and dropoff. Everything else (cleaning bounds, existing engineered features, xgb.train with early stopping, iteration_range inference fix, submission alignment) stays the same and still writes `submission.csv`.'
- What this solution (achieved 5.48334) has done: 'We’re still well above the target RMSE (5.50 vs 3.44, lower is better), so we should make a small, legitimate accuracy improvement without changing your core XGBoost+log1p pipeline or feature style. The biggest low-risk win here is to add a standard NYC taxi baseline feature: a calibrated “straight-line fare” estimate from distance (meter drop + per-km + per-minute using simple average speed), which preserves your existing model and simply provides a strong extra signal. To avoid hurting generalization, we keep all your existing cleaning, split-by-time-with-gap, early stopping, and submission alignment exactly the same, and we only add this derived numeric feature to both train and test consistently. This should move RMSE down toward the target band while staying within runtime and constraints.'
- What this solution (achieved 8.69498) has done: 'You’re still far above the target RMSE (5.48 vs 3.44, lower is better), so we should make one small, model-consistent improvement that typically reduces error without changing the overall XGBoost+log1p approach. The biggest low-risk issue is that `fare_est` can become overly influential (it’s built from fixed heuristics that may be miscalibrated), so we keep it but make it “learnable” by also providing its log form and a residual-to-estimate feature; this preserves core logic while giving the model a more stable signal. We also add two standard, minimal interaction features derived from your existing distances (`distance^2` and `distance*passenger_count`) which often improve RMSE for this competition without changing architecture/training semantics. Everything else (cleaning, time split with gap, early stopping, inference iteration_range, submission alignment) remains the same and still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import datetime as dt
from sklearn.model_selection import train_test_split
import xgboost as xgb
import os

print(os.listdir("../input"))



## === cell 1
train_df = pd.read_csv("../input/train.csv", nrows=3_000_000)
train_df.dtypes



## === cell 2
print(train_df.isnull().sum())



## === cell 3
train_df = train_df.dropna(how="any", axis="rows")



## === cell 4
train_df.head()



## === cell 5
try:
    ax1 = train_df.iloc[:1000].plot.scatter("pickup_longitude", "pickup_latitude")
    ax2 = train_df.iloc[:1000].plot.scatter("dropoff_longitude", "dropoff_latitude")
except Exception as e:
    print(f"Plotting skipped due to: {e}")

train_df.describe()




## === cell 6
def clean_df(df):
    base = (
        (df.pickup_longitude >= -74.3)
        & (df.pickup_longitude <= -72.9)
        & (df.pickup_latitude >= 40.5)
        & (df.pickup_latitude <= 41.8)
        & (df.dropoff_longitude >= -74.3)
        & (df.dropoff_longitude <= -72.9)
        & (df.dropoff_latitude >= 40.5)
        & (df.dropoff_latitude <= 41.8)
        & (df.passenger_count >= 1)
        & (df.passenger_count <= 6)
    )
    return df[base]


train_df = clean_df(train_df)

train_df = train_df[
    (train_df["pickup_longitude"].abs() > 1e-6)
    & (train_df["pickup_latitude"].abs() > 1e-6)
    & (train_df["dropoff_longitude"].abs() > 1e-6)
    & (train_df["dropoff_latitude"].abs() > 1e-6)
]

train_df = train_df[(train_df.fare_amount >= 2.5) & (train_df.fare_amount <= 250)]
print(len(train_df))




## === cell 7
def sphere_dist(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    R_earth = 6371
    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = map(
        np.radians, [pickup_lat, pickup_lon, dropoff_lat, dropoff_lon]
    )
    dlat = dropoff_lat - pickup_lat
    dlon = dropoff_lon - pickup_lon

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(pickup_lat) * np.cos(dropoff_lat) * np.sin(dlon / 2.0) ** 2
    )

    return 2 * R_earth * np.arcsin(np.sqrt(a))


def add_datetime_info(dataset):
    dataset = dataset.copy()
    dataset["pickup_datetime"] = pd.to_datetime(
        dataset["pickup_datetime"], errors="coerce", utc=True
    ).dt.tz_convert(None)

    med_dt = dataset["pickup_datetime"].dropna().median()
    dataset["pickup_datetime"] = dataset["pickup_datetime"].fillna(med_dt)

    dataset["hour"] = dataset.pickup_datetime.dt.hour
    dataset["day"] = dataset.pickup_datetime.dt.day
    dataset["month"] = dataset.pickup_datetime.dt.month
    dataset["weekday"] = dataset.pickup_datetime.dt.weekday

    dataset["year"] = dataset.pickup_datetime.dt.year
    dataset["dayofyear"] = dataset.pickup_datetime.dt.dayofyear

    return dataset


def add_geo_features(df):
    df = df.copy()
    df["abs_lon_diff"] = (df["pickup_longitude"] - df["dropoff_longitude"]).abs()
    df["abs_lat_diff"] = (df["pickup_latitude"] - df["dropoff_latitude"]).abs()

    mean_lat = np.radians((df["pickup_latitude"] + df["dropoff_latitude"]) / 2.0)
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * np.cos(mean_lat)

    df["manhattan_km"] = (
        df["abs_lat_diff"] * km_per_deg_lat + df["abs_lon_diff"] * km_per_deg_lon
    )
    return df


def add_airport_features(df):
    df = df.copy()
    jfk_lat, jfk_lon = 40.6413, -73.7781
    lga_lat, lga_lon = 40.7769, -73.8740

    df["pickup_to_jfk_km"] = sphere_dist(
        df["pickup_latitude"], df["pickup_longitude"], jfk_lat, jfk_lon
    )
    df["dropoff_to_jfk_km"] = sphere_dist(
        df["dropoff_latitude"], df["dropoff_longitude"], jfk_lat, jfk_lon
    )
    df["pickup_to_lga_km"] = sphere_dist(
        df["pickup_latitude"], df["pickup_longitude"], lga_lat, lga_lon
    )
    df["dropoff_to_lga_km"] = sphere_dist(
        df["dropoff_latitude"], df["dropoff_longitude"], lga_lat, lga_lon
    )
    return df


def add_center_features(df):
    df = df.copy()
    nyc_lat, nyc_lon = 40.7580, -73.9855  # Times Square-ish
    df["pickup_to_center_km"] = sphere_dist(
        df["pickup_latitude"], df["pickup_longitude"], nyc_lat, nyc_lon
    )
    df["dropoff_to_center_km"] = sphere_dist(
        df["dropoff_latitude"], df["dropoff_longitude"], nyc_lat, nyc_lon
    )
    return df


def add_bearing_feature(df):
    df = df.copy()
    lat1 = np.radians(df["pickup_latitude"].astype(float))
    lat2 = np.radians(df["dropoff_latitude"].astype(float))
    dlon = np.radians((df["dropoff_longitude"] - df["pickup_longitude"]).astype(float))
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.arctan2(y, x)  # radians, [-pi, pi]
    df["bearing"] = bearing.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def add_fare_estimate_feature(df):
    df = df.copy()
    if "distance" not in df.columns:
        df["distance"] = sphere_dist(
            df["pickup_latitude"],
            df["pickup_longitude"],
            df["dropoff_latitude"],
            df["dropoff_longitude"],
        )
    if "hour" not in df.columns:
        df = add_datetime_info(df)

    base_fare = 2.50
    per_km = 1.60  # ~$2.57 per mile
    per_min = 0.50  # rough waiting/time component
    avg_speed_kmh = 18.0

    minutes = (df["distance"].astype(float) / avg_speed_kmh) * 60.0
    rush = df["hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(float)
    mult = 1.0 + 0.05 * rush

    est = (base_fare + per_km * df["distance"].astype(float) + per_min * minutes) * mult
    df["fare_est"] = est.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


train_df["distance"] = sphere_dist(
    train_df["pickup_latitude"],
    train_df["pickup_longitude"],
    train_df["dropoff_latitude"],
    train_df["dropoff_longitude"],
)

train_df = add_datetime_info(train_df)
train_df["_pickup_dt"] = train_df["pickup_datetime"]

train_df = add_geo_features(train_df)
train_df = add_airport_features(train_df)
train_df = add_center_features(train_df)
train_df = add_bearing_feature(train_df)
train_df = add_fare_estimate_feature(train_df)

train_df = train_df[(train_df["distance"] >= 0) & (train_df["distance"] <= 45)]

train_df["distance_sq"] = (train_df["distance"].astype(float) ** 2).replace(
    [np.inf, -np.inf], np.nan
)
train_df["dist_x_pax"] = (
    train_df["distance"].astype(float) * train_df["passenger_count"].astype(float)
).replace([np.inf, -np.inf], np.nan)

train_df["fare_est_log1p"] = np.log1p(
    train_df["fare_est"].astype(float).clip(lower=0.0)
)
train_df["fare_minus_est"] = (
    train_df["fare_amount"].astype(float) - train_df["fare_est"].astype(float)
).replace([np.inf, -np.inf], np.nan)

for c in ["hour", "day", "month", "weekday", "year", "dayofyear"]:
    train_df[c] = train_df[c].fillna(0).astype(int)

feat_cols_train = [
    c for c in train_df.columns if c not in ["key", "pickup_datetime", "fare_amount"]
]
train_df[feat_cols_train] = (
    train_df[feat_cols_train].replace([np.inf, -np.inf], np.nan).fillna(0)
)

train_df.head()



## === cell 8
train_df.head()



## === cell 9
med_dt = train_df["_pickup_dt"].dropna().median()
train_df["_pickup_dt"] = train_df["_pickup_dt"].fillna(med_dt)

train_df = train_df.sort_values("_pickup_dt").reset_index(drop=True)

y = np.log1p(train_df["fare_amount"].astype(float))

X_full = train_df.drop(columns=["fare_amount", "key", "pickup_datetime"])

split_idx = int(len(train_df) * 0.8)
gap = min(50_000, max(0, len(train_df) // 50))

train_end = max(0, split_idx - gap)
valid_start = split_idx

x_train = X_full.iloc[:train_end].copy()
x_test = X_full.iloc[valid_start:].copy()
y_train = y.iloc[:train_end].copy()
y_test = y.iloc[valid_start:].copy()

for df_ in (x_train, x_test):
    if "_pickup_dt" in df_.columns:
        df_.drop(columns=["_pickup_dt"], inplace=True)




## === cell 10
def XGBmodel(x_train, x_test, y_train, y_test):
    matrix_train = xgb.DMatrix(x_train, label=y_train)
    matrix_test = xgb.DMatrix(x_test, label=y_test)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "seed": 0,
        "eta": 0.05,
        "max_depth": 7,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 1.0,
        "alpha": 0.0,
        "gamma": 0.0,
        "tree_method": "hist",
    }

    model = xgb.train(
        params=params,
        dtrain=matrix_train,
        num_boost_round=4000,
        early_stopping_rounds=50,
        evals=[(matrix_test, "test")],
        verbose_eval=False,
    )
    return model


model = XGBmodel(x_train, x_test, y_train, y_test)




## === cell 11
def sanitize_test_df(df):
    df = df.copy()
    df["pickup_longitude"] = df["pickup_longitude"].clip(-74.3, -72.9)
    df["dropoff_longitude"] = df["dropoff_longitude"].clip(-74.3, -72.9)
    df["pickup_latitude"] = df["pickup_latitude"].clip(40.5, 41.8)
    df["dropoff_latitude"] = df["dropoff_latitude"].clip(40.5, 41.8)

    df["passenger_count"] = df["passenger_count"].fillna(1)
    df["passenger_count"] = df["passenger_count"].clip(1, 6)

    return df


test_df = pd.read_csv("../input/test.csv")
sample_sub = pd.read_csv("../input/sample_submission.csv")

test_df = sanitize_test_df(test_df)

test_df["distance"] = sphere_dist(
    test_df["pickup_latitude"],
    test_df["pickup_longitude"],
    test_df["dropoff_latitude"],
    test_df["dropoff_longitude"],
)
test_df = add_datetime_info(test_df)
test_df = add_geo_features(test_df)
test_df = add_airport_features(test_df)
test_df = add_center_features(test_df)
test_df = add_bearing_feature(test_df)
test_df = add_fare_estimate_feature(test_df)

test_df["distance_sq"] = (test_df["distance"].astype(float) ** 2).replace(
    [np.inf, -np.inf], np.nan
)
test_df["dist_x_pax"] = (
    test_df["distance"].astype(float) * test_df["passenger_count"].astype(float)
).replace([np.inf, -np.inf], np.nan)
test_df["fare_est_log1p"] = np.log1p(test_df["fare_est"].astype(float).clip(lower=0.0))

test_df["fare_minus_est"] = 0.0

for c in ["hour", "day", "month", "weekday", "year", "dayofyear"]:
    if c in test_df.columns:
        test_df[c] = test_df[c].fillna(0).astype(int)

feat_cols = [c for c in test_df.columns if c not in ["key", "pickup_datetime"]]
test_df[feat_cols] = test_df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

test_key = test_df["key"]

x_pred = test_df.drop(columns=["key", "pickup_datetime"])
x_pred = x_pred.reindex(columns=x_train.columns, fill_value=0)

dm_pred = xgb.DMatrix(x_pred)

best_iter = getattr(model, "best_iteration", None)
if best_iter is not None:
    pred_log = model.predict(dm_pred, iteration_range=(0, best_iter + 1))
else:
    pred_log = model.predict(dm_pred)

prediction = np.expm1(pred_log)
prediction = np.clip(prediction, 0.0, 250.0)

pred_map = pd.DataFrame({"key": test_key, "fare_amount": prediction})
submission = sample_sub[["key"]].merge(pred_map, on="key", how="left")

submission["fare_amount"] = submission["fare_amount"].fillna(
    float(np.nanmean(prediction))
)

submission.to_csv("submission.csv", index=False)
submission
