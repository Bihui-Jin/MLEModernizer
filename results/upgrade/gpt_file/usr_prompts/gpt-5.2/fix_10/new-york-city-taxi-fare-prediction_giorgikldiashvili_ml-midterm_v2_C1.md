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
geopy==2.4.1
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
tf_keras==2.18.0
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

3.61245

# 6. Current score

5.90522

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.53493) has done: 'I fix the TensorFlow/Keras crash caused by an incompatible `protobuf==6.x` (it triggers the `MessageFactory.GetPrototype` error in TF 2.18) by removing the neural-net section from execution so the notebook runs end-to-end. Your extremely bad Kaggle score is consistent with a `key`/prediction misalignment or invalid predictions; I ensure predictions are finite, clipped to reasonable positive fares, and written using the `sample_submission.csv` key order to guarantee correct alignment. I also replace the very slow `geopy.great_circle` loop with a vectorized haversine distance (same feature intent: trip distance) to improve both runtime and RMSE without changing the overall approach (still distance/time + XGBoost). Finally, the script always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 4.45916) has done: 'Your current score (5.53493, lower-is-better) is worse than the target (3.61245), so we should make small, low-risk improvements that typically reduce RMSE without changing the overall approach (still: basic cleaning + distance/time features + XGBoost). The biggest likely win with minimal disruption is to (1) apply the same NYC bounding-box filter to the test set (to avoid extreme out-of-distribution coordinates producing wild predictions), and (2) add a couple of standard, lightweight geospatial features (manhattan distance and coordinate deltas) that preserve the same “distance-based” feature intent. Finally, we keep the submission alignment-by-key logic, but add a safe fallback for any test rows filtered out so the submission always has all keys filled. These are minimal changes that usually move this classic Taxi Fare baseline down toward ~3–4 RMSE.'
- What this solution (achieved 4.58739) has done: 'Your current score (4.45916 RMSE; lower is better) is worse than the target (3.61245), so we should make small, low-risk improvements that usually reduce RMSE without changing the overall “distance/time features + XGBoost” approach. The biggest likely gain is fixing the train/validation split to be more representative: instead of a random split, use a time-based split so the model is validated on later rides (closer to the test distribution), which tends to improve generalization for this competition. Second, we keep your exact feature intent but add one standard, lightweight geospatial feature (`bearing`) that complements distance without changing the modeling approach. Finally, we keep your safe submission alignment-by-key logic unchanged.'
- What this solution (achieved 4.13981) has done: 'We’re currently worse than the target RMSE (4.58739 vs 3.61245; lower is better), so we make a small, low-risk improvement that usually generalizes better without changing the overall “distance/time features + XGBoost” approach. The most impactful minimal tweak is to make the time features cyclical (sin/cos for hour and weekday) so the model can learn periodicity smoothly, while keeping the original raw time fields to preserve semantics. We also remove the hard “distance > 0.1” cutoff (keep only an upper cap) so short trips are retained—those are common in NYC and removing them typically hurts RMSE. Everything else (data paths, model family, training flow, submission alignment) stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved 4.23488) has done: 'We’re currently worse than the target RMSE (4.13981 vs 3.61245; lower is better), so the safest way to move toward the target without changing the modeling approach is to improve training data quality and add one standard “distance to city center” feature that stays within your existing geospatial feature intent. Specifically, we (1) remove known-bad training rows (fare outliers and the common “0,0” coordinate issues) using simple, competition-standard filters, and (2) add `pickup_dist_center` and `dropoff_dist_center` using the same haversine function (no new dependencies, same feature family). We keep the same time-based split, XGBoost regressor family, submission key alignment, and prediction clipping so the pipeline remains stable and still writes a valid `submission.csv`. These minimal data/feature tweaks typically reduce RMSE for this competition and should move you closer to 3.61.'
- What this solution (achieved 4.62488) has done: 'Your current RMSE (4.23488) is worse than the target (3.61245), so we make two very small, competition-standard data-quality tweaks that typically reduce RMSE without changing your overall approach (same features + XGBoost training flow). First, we remove a handful of well-known fare outliers that survive your current filters (e.g., very low fares and unusually high fares), which reduces label noise. Second, we drop rides with extremely small “distance” but non-trivial fares (and vice versa) using a light consistency filter; this helps the model learn a cleaner distance→fare relationship without changing the model or feature set. Everything else (time-based split, feature engineering, XGBRegressor, submission alignment and fallback fill) stays the same and still writes `submission.csv`.'
- What this solution (achieved 6.19718) has done: 'We’re currently worse than the target (4.62488 vs 3.61245 RMSE; lower is better), so we should make a small, low-risk improvement that tends to generalize better without changing your overall “distance/time features + XGBoost” approach. The biggest minimal win here is to stop dropping all out-of-bounding-box test rows (and then median-filling them), because that fallback hurts RMSE if any real test points fall outside your box; instead, we compute features and predict for all test rows, while still keeping the stricter bbox filter for training quality. Second, we remove the overly harsh “distance < 0.01 & fare > 30” / “distance > 15 & fare < 5” consistency filter that likely threw away legitimate (noisy) but informative training examples and made the model underfit rare cases. Everything else (features, time split, XGBRegressor, clipping, submission key alignment) stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved 5.79017) has done: 'Your current RMSE (6.19718, lower-is-better) is much worse than the target (3.61245), and the code already uses a sensible XGBoost + distance/time feature baseline, so the most likely issue is not “model weakness” but training data quality and distribution mismatch. I make two minimal, competition-standard cleaning improvements that usually give a large RMSE drop without changing the core approach: (1) remove obvious label/geo outliers using a simple “fare per mile” sanity band (keeps typical NYC rides, drops broken rows), and (2) cap extreme passenger counts (already done) while also removing rows with extremely tiny nonzero distances that often correspond to bad coordinates. Everything else (features, time-split, XGBRegressor, submission alignment by `sample_submission` key order, and prediction clipping) stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved 5.90522) has done: 'Your current score (5.79017 RMSE; lower is better) is still far from the target (3.61245), so we should make a small, low-risk improvement that typically reduces RMSE without changing your core “distance/time features + XGBoost” approach. The biggest likely issue is label noise and residual outliers: your current `fare_per_mile` band is too permissive and still leaves many bad rows that degrade learning. I tighten the fare-per-mile sanity band to a more NYC-realistic range and add a minimal additional cleanup for unrealistically large coordinate deltas (often data errors) while keeping the same features, split logic, model type, training flow, and submission alignment. This should improve generalization and move RMSE downward toward your target while keeping runtime within limits.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

INPUT_DIR = "/kaggle/input/new-york-city-taxi-fare-prediction"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

print("Files in /kaggle/input/new-york-city-taxi-fare-prediction:")
for f in sorted(os.listdir(INPUT_DIR))[:20]:
    print(" -", f)



## === cell 1
train_df = pd.read_csv(TRAIN_PATH, nrows=1_000_000)
test_df = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

print(train_df.shape, test_df.shape, sample_sub.shape)
print(train_df.head(2))
print(test_df.head(2))



## === cell 2
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor



## === cell 3
train_df = train_df.dropna()

train_df = train_df[train_df["passenger_count"] < 8]
train_df = train_df[train_df["passenger_count"] >= 1]

train_df = train_df[train_df["fare_amount"] >= 2.5]
train_df = train_df[train_df["fare_amount"] <= 200.0]

print("After basic cleaning (+outlier filters):", train_df.shape)
print(train_df.isnull().sum())



## === cell 4
RANGE = (-74.26, -72.99, 40.56, 41.71)


def select_within_boundingbox(df, rng):
    return (
        (df.pickup_longitude >= rng[0])
        & (df.pickup_longitude <= rng[1])
        & (df.pickup_latitude >= rng[2])
        & (df.pickup_latitude <= rng[3])
        & (df.dropoff_longitude >= rng[0])
        & (df.dropoff_longitude <= rng[1])
        & (df.dropoff_latitude >= rng[2])
        & (df.dropoff_latitude <= rng[3])
    )


old_size = len(train_df)
train_df = train_df[select_within_boundingbox(train_df, RANGE)]
print(f"Bounding box filter (train): {old_size} -> {len(train_df)}")

test_in_bbox_mask = select_within_boundingbox(test_df, RANGE)
print(
    f"Bounding box stats (test): total {len(test_df)}; in-bbox {int(test_in_bbox_mask.sum())}; we will still predict ALL rows"
)




## === cell 5
def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.7613  # Earth radius in miles
    lat1 = np.radians(lat1.astype(float))
    lon1 = np.radians(lon1.astype(float))
    lat2 = np.radians(lat2.astype(float))
    lon2 = np.radians(lon2.astype(float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def bearing_degrees(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1.astype(float))
    lon1 = np.radians(lon1.astype(float))
    lat2 = np.radians(lat2.astype(float))
    lon2 = np.radians(lon2.astype(float))
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    brng = np.degrees(np.arctan2(y, x))
    brng = (brng + 360.0) % 360.0
    return brng


NYC_CENTER_LAT = 40.7141667
NYC_CENTER_LON = -74.0063889


def add_features(df):
    df = df.copy()

    df["distance"] = haversine_miles(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )

    df["abs_lat_diff"] = (
        (df["dropoff_latitude"] - df["pickup_latitude"]).abs().astype("float32")
    )
    df["abs_lon_diff"] = (
        (df["dropoff_longitude"] - df["pickup_longitude"]).abs().astype("float32")
    )
    df["manhattan_approx"] = (df["abs_lat_diff"] + df["abs_lon_diff"]).astype("float32")

    df["bearing"] = bearing_degrees(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    ).astype("float32")

    df["pickup_dist_center"] = haversine_miles(
        df["pickup_latitude"],
        df["pickup_longitude"],
        np.full(len(df), NYC_CENTER_LAT, dtype="float64"),
        np.full(len(df), NYC_CENTER_LON, dtype="float64"),
    ).astype("float32")

    df["dropoff_dist_center"] = haversine_miles(
        df["dropoff_latitude"],
        df["dropoff_longitude"],
        np.full(len(df), NYC_CENTER_LAT, dtype="float64"),
        np.full(len(df), NYC_CENTER_LON, dtype="float64"),
    ).astype("float32")

    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)
    df["hour"] = dt.dt.hour.astype("float32")
    df["day"] = dt.dt.day.astype("float32")
    df["month"] = dt.dt.month.astype("float32")
    df["year"] = dt.dt.year.astype("float32")
    df["weekday"] = dt.dt.weekday.astype("float32")
    df["pickup_ts"] = (dt.view("int64") // 10**9).astype("int64")

    hour_rad = 2.0 * np.pi * (df["hour"].astype("float32") / 24.0)
    wday_rad = 2.0 * np.pi * (df["weekday"].astype("float32") / 7.0)
    df["hour_sin"] = np.sin(hour_rad).astype("float32")
    df["hour_cos"] = np.cos(hour_rad).astype("float32")
    df["weekday_sin"] = np.sin(wday_rad).astype("float32")
    df["weekday_cos"] = np.cos(wday_rad).astype("float32")

    return df


train_df = add_features(train_df)
test_df_all = add_features(test_df.copy())

train_df = train_df.dropna(
    subset=[
        "distance",
        "hour",
        "day",
        "month",
        "year",
        "weekday",
        "pickup_ts",
        "bearing",
        "hour_sin",
        "hour_cos",
        "weekday_sin",
        "weekday_cos",
        "pickup_dist_center",
        "dropoff_dist_center",
        "abs_lat_diff",
        "abs_lon_diff",
        "manhattan_approx",
    ]
)
test_df_all = test_df_all.dropna(
    subset=[
        "distance",
        "hour",
        "day",
        "month",
        "year",
        "weekday",
        "pickup_ts",
        "bearing",
        "hour_sin",
        "hour_cos",
        "weekday_sin",
        "weekday_cos",
        "pickup_dist_center",
        "dropoff_dist_center",
        "abs_lat_diff",
        "abs_lon_diff",
        "manhattan_approx",
    ]
)

print("Feature-added shapes:", train_df.shape, test_df_all.shape)



## === cell 6
coord_bad = (
    (train_df["pickup_latitude"].abs() < 1e-6)
    | (train_df["pickup_longitude"].abs() < 1e-6)
    | (train_df["dropoff_latitude"].abs() < 1e-6)
    | (train_df["dropoff_longitude"].abs() < 1e-6)
)
old_n = len(train_df)
train_df = train_df.loc[~coord_bad].copy()
print(f"Removed near-zero coords (train): {old_n} -> {len(train_df)}")

train_df = train_df[(train_df["distance"] < 25) & (train_df["distance"] >= 0.0)]
print("After distance filter:", train_df.shape)

eps = 1e-3
fare_per_mile = train_df["fare_amount"] / (train_df["distance"] + eps)
old_n = len(train_df)
train_df = train_df[(fare_per_mile >= 1.5) & (fare_per_mile <= 30.0)].copy()
print(f"Fare-per-mile sanity filter (tighter): {old_n} -> {len(train_df)}")

old_n = len(train_df)
train_df = train_df[
    (train_df["abs_lat_diff"] <= 0.5) & (train_df["abs_lon_diff"] <= 0.5)
].copy()
print(f"Extreme coord-delta filter: {old_n} -> {len(train_df)}")

old_n = len(train_df)
train_df = train_df[
    (train_df["distance"] >= 0.02) | (train_df["fare_amount"] <= 6.0)
].copy()
print(f"Tiny-distance noise filter: {old_n} -> {len(train_df)}")

print(
    "Skipped implausible distance/fare combo filter to reduce underfitting and improve generalization."
)



## === cell 7
features = [
    "passenger_count",
    "distance",
    "abs_lat_diff",
    "abs_lon_diff",
    "manhattan_approx",
    "bearing",
    "pickup_dist_center",
    "dropoff_dist_center",
    "hour",
    "day",
    "month",
    "year",
    "weekday",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
]

X = train_df[features].astype("float32")
y = train_df["fare_amount"].astype("float32")

cutoff = train_df["pickup_ts"].quantile(0.8)
train_mask = train_df["pickup_ts"] <= cutoff
valid_mask = ~train_mask

X_train, y_train = X.loc[train_mask], y.loc[train_mask]
X_valid, y_valid = X.loc[valid_mask], y.loc[valid_mask]

print("Time-split sizes:", X_train.shape, X_valid.shape)



## === cell 8
model = XGBRegressor(
    learning_rate=0.1,
    n_estimators=300,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="reg:squarederror",
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

y_val_pred = model.predict(X_valid)
rmse = float(np.sqrt(mean_squared_error(y_valid, y_val_pred)))
print("Validation RMSE:", rmse)



## === cell 9
X_test = test_df_all[features].astype("float32")
y_test_pred = model.predict(X_test)

y_test_pred = np.asarray(y_test_pred, dtype="float64")
y_test_pred = np.where(np.isfinite(y_test_pred), y_test_pred, np.nan)
median_fare = float(np.nanmedian(y)) if np.isfinite(np.nanmedian(y)) else 11.35
y_test_pred = np.nan_to_num(
    y_test_pred, nan=median_fare, posinf=median_fare, neginf=median_fare
)

y_test_pred = np.clip(y_test_pred, 0.0, 250.0)

pred_df = pd.DataFrame({"key": test_df_all["key"].values, "fare_amount": y_test_pred})

submission_df = sample_sub[["key"]].merge(pred_df, on="key", how="left")
submission_df["fare_amount"] = (
    submission_df["fare_amount"].fillna(median_fare).astype("float64")
)

submission_df.to_csv("submission.csv", index=False)
print(submission_df.head())
print("Wrote submission.csv with shape:", submission_df.shape)
print(
    "Missing predictions filled with median_fare:",
    int(submission_df["fare_amount"].isna().sum()),
)



## === cell 10
print(
    "NN section skipped due to TF/protobuf incompatibility; submission.csv generated via XGBoost."
)
