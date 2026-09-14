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

4.79008

# 6. Current score

6.02947

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.23058) has done: 'I fix the datetime feature extraction to be compatible with modern pandas (replacing deprecated `.dt.week`/`.dt.weekofyear` with ISO calendar week) so your preprocessing runs. Then I ensure no datetime-typed columns leak into the model matrix by coercing all feature columns to numeric, which resolves the RandomForest `DTypePromotionError`. Finally, I keep your same model and training approach but make the train/test feature alignment explicit and write a valid `submission.csv` with the required columns.'
- What this solution (achieved 6.27152) has done: 'Your current RMSE (6.23058) is worse than the target (4.79008), so we should make a small, legitimate improvement without changing the core model/training approach. The biggest low-risk gain here is adding standard NYC Taxi data sanity filters (latitude/longitude bounds and outlier-distance removal) which reduces label noise and typically improves RMSE for this exact competition while keeping your same RandomForest and features. I also ensure we train on a numeric, NaN-free matrix consistently and clip negative predictions to 0 to avoid obviously invalid fares that can hurt RMSE. These changes are minimal, metric-aligned, and keep the same overall pipeline structure.'
- What this solution (achieved 6.26375) has done: 'You’re worse than the target (6.27152 vs 4.79008; lower is better), so we should make a small, safe improvement without changing your core approach (same RF model, same features, same training flow). The biggest issue is your current distance filter (`distance_travelled < 1.0`) is far too strict for NYC trips and discards many valid rides, hurting generalization; widening this to a more realistic bound typically reduces RMSE. I also compute a proper geodesic (haversine) distance feature alongside your existing Euclidean-in-degrees distance (keeping your original feature intact) and add a simple `abs_lon/abs_lat` sum as a tiny extra signal—these are minimal feature additions that keep the same model/training semantics but usually help this competition. Finally, I keep your existing sanity filters but slightly tighten only the most harmful outliers (e.g., absurd fares) while preserving valid longer trips.'
- What this solution (achieved 6.27378) has done: 'Your current RMSE (6.26375) is still worse than the target (4.79008), so we make a small, legitimate improvement without changing the core approach (same RandomForestRegressor, same training flow, same basic feature engineering). The biggest low-risk gain is to correct the overly-strict `distance_travelled < 0.30` filter: that filter is in “degrees” and is effectively removing many valid trips and distorting the training distribution; switching that particular bound to use your already-computed `haversine_km` is a minimal fix that typically improves RMSE. We keep your existing NYC bounding-box and fare/passenger filters, but make the distance filtering consistent and avoid double-filtering that conflicts. The submission writing stays the same and still produce a valid `submission.csv`.'
- What this solution (achieved 6.58884) has done: 'We’re currently worse than the target (6.27378 vs 4.79008; lower is better), so the smallest legitimate push toward the target is to reduce noise in the training labels and align the model to the heavy-tailed fare distribution without changing your core model/training loop. I add two standard NYC Taxi cleanup steps: remove rows where pickup/dropoff coordinates are identical (zero-distance but nonzero fare) and drop extreme outliers in `haversine_km` vs `fare_amount` using a very conservative speed-like cap. Then I train the same RandomForest on `log1p(fare_amount)` and invert with `expm1` at prediction time (same model, same features, same training flow), which typically improves RMSE for this competition by stabilizing large-fare errors. Submission writing and file name stay identical.'
- What this solution (achieved 6.57285) has done: 'Your current RMSE (6.58884; lower is better) is still far from the target (4.79008), so we should make a small, legitimate improvement without changing the core approach (same RandomForestRegressor, same feature family, same single fit/predict flow). The biggest low-risk issue is that the log-target transform is currently trained on the full (noisy) label distribution; adding a very standard, conservative “fare per km” filter (lower and upper bounds) typically removes mislabeled/erroneous trips and improves generalization for this competition. I keep all your existing filters and features, but add one additional cleanup step based on `fare_amount / haversine_km` (with safeguards for small distances) and keep submission writing identical. This should move the score downward (better) toward the target without altering the model architecture or training semantics.'
- What this solution (achieved 6.02947) has done: 'Main bottlenecks are (1) parsing 5M datetimes with multiple `.dt` accesses and `isocalendar()` overhead, (2) repeated Pandas Series operations during cleaning, and (3) fitting a large RandomForest on a large dense matrix. The refactor keeps the exact same features, filters, model, and training semantics, but speeds things up by vectorizing date feature extraction from the raw timestamp string (no `to_datetime`), computing distance features fully in NumPy without extra float64 conversions, and moving the cleanup mask computation to NumPy arrays to avoid Pandas alignment overhead. It also enables Intel-optimized scikit-learn where available and avoids unnecessary copies when building `X_train/X_test`. These changes are provably equivalent for this dataset format (fixed `YYYY-MM-DD HH:MM:SS...`), preserve determinism, and reduce total runtime substantially.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

np.random.seed(0)

INPUT_DIR_CANDIDATES = ["../input", "/kaggle/input", "/kaggle/data"]
INPUT_DIR = None
for p in INPUT_DIR_CANDIDATES:
    if os.path.isdir(p):
        INPUT_DIR = p
        break
if INPUT_DIR is None:
    INPUT_DIR = "../input"

print("Using INPUT_DIR:", INPUT_DIR)
print(os.listdir(INPUT_DIR))



## === cell 1
train_path = os.path.join(INPUT_DIR, "train.csv")
test_path = os.path.join(INPUT_DIR, "test.csv")

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

dtype_common = {
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
dtype_train = dict(dtype_common)
dtype_train["fare_amount"] = "float32"

train = pd.read_csv(
    train_path,
    nrows=5000000,
    usecols=usecols_train,
    dtype=dtype_train,
    low_memory=False,
)
test = pd.read_csv(
    test_path,
    usecols=usecols_test,
    dtype=dtype_common,
    low_memory=False,
)



## === cell 2
pass




## === cell 3
def handle_date_inplace(df):
    s = df["pickup_datetime"].astype(str)

    s = s.str.replace(" UTC", "", regex=False)

    year = s.str.slice(0, 4).astype("int16")
    month = s.str.slice(5, 7).astype("int16")
    day = s.str.slice(8, 10).astype("int16")
    hour = s.str.slice(11, 13).astype("int16")

    dt = pd.to_datetime(
        year.astype(str)
        + "-"
        + month.astype(str).str.zfill(2)
        + "-"
        + day.astype(str).str.zfill(2),
        errors="coerce",
        cache=True,
        format="%Y-%m-%d",
    )

    dt_ = dt.dt
    week = dt.dt.strftime("%V").astype("int16")

    df["hour_of_day"] = hour
    df["week"] = week
    df["month"] = month
    df["year"] = year
    df["day_of_year"] = dt_.dayofyear.astype("int16")
    df["week_of_year"] = week
    df["weekday"] = dt_.weekday.astype("int16")
    df["quarter"] = (((month - 1) // 3) + 1).astype("int16")
    df["day_of_month"] = day

    df.drop(columns=["pickup_datetime"], inplace=True)
    return df


train = handle_date_inplace(train)
test = handle_date_inplace(test)




## === cell 4
def handle_distance_inplace(df):
    plon = df["pickup_longitude"].to_numpy(dtype=np.float32, copy=False)
    plat = df["pickup_latitude"].to_numpy(dtype=np.float32, copy=False)
    dlon = df["dropoff_longitude"].to_numpy(dtype=np.float32, copy=False)
    dlat = df["dropoff_latitude"].to_numpy(dtype=np.float32, copy=False)

    lon_dist = np.abs(plon - dlon, dtype=np.float32)
    lat_dist = np.abs(plat - dlat, dtype=np.float32)

    df["distance_travelled"] = np.sqrt(
        lon_dist * lon_dist + lat_dist * lat_dist, dtype=np.float32
    ).astype(np.float32, copy=False)

    R = np.float32(6371.0)
    lat1 = np.deg2rad(plat.astype(np.float32, copy=False))
    lat2 = np.deg2rad(dlat.astype(np.float32, copy=False))
    dlat_r = lat2 - lat1
    dlon_r = np.deg2rad((dlon - plon).astype(np.float32, copy=False))
    sin_dlat = np.sin(dlat_r * np.float32(0.5))
    sin_dlon = np.sin(dlon_r * np.float32(0.5))
    a = sin_dlat * sin_dlat + np.cos(lat1) * np.cos(lat2) * (sin_dlon * sin_dlon)
    df["haversine_km"] = (np.float32(2.0) * R * np.arcsin(np.sqrt(a))).astype(
        np.float32, copy=False
    )

    df["manhattan_approx"] = (lon_dist + lat_dist).astype(np.float32, copy=False)
    return df


train = handle_distance_inplace(train)
test = handle_distance_inplace(test)




## === cell 5
def clean_up_train(train):
    train = train.dropna()

    fare = train["fare_amount"].to_numpy(dtype=np.float32, copy=False)
    pc = train["passenger_count"].to_numpy(dtype=np.int16, copy=False)
    plon = train["pickup_longitude"].to_numpy(dtype=np.float32, copy=False)
    dlon = train["dropoff_longitude"].to_numpy(dtype=np.float32, copy=False)
    plat = train["pickup_latitude"].to_numpy(dtype=np.float32, copy=False)
    dlat = train["dropoff_latitude"].to_numpy(dtype=np.float32, copy=False)
    hv = train["haversine_km"].to_numpy(dtype=np.float32, copy=False)

    mask = (
        (fare > 0)
        & (pc > 0)
        & (pc < 7)
        & (plon >= -74.3)
        & (plon <= -73.7)
        & (dlon >= -74.3)
        & (dlon <= -73.7)
        & (plat >= 40.5)
        & (plat <= 41.0)
        & (dlat >= 40.5)
        & (dlat <= 41.0)
        & (hv > 0.05)
        & (hv < 80.0)
        & (fare < 200)
    )

    same_loc = (plon == dlon) & (plat == dlat)
    mask &= ~same_loc

    mask &= fare <= (2.5 + 10.0 * hv)

    denom_km = np.maximum(hv.astype(np.float32, copy=False), np.float32(0.2))
    fare_per_km = fare / denom_km
    mask &= (fare_per_km >= 0.5) & (fare_per_km <= 60.0)

    return train.loc[mask]


train = clean_up_train(train)



## === cell 6
feature_cols = test.columns.drop("key")

X_train = train.loc[:, feature_cols].to_numpy(dtype=np.float32, copy=False)
y = train["fare_amount"].to_numpy(dtype=np.float64, copy=False)

X_test = test.loc[:, feature_cols].to_numpy(dtype=np.float32, copy=False)

np.nan_to_num(X_train, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
np.nan_to_num(X_test, copy=False, nan=0.0, posinf=0.0, neginf=0.0)



## === cell 7
try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=120,
    max_depth=14,
    min_samples_leaf=1,
    random_state=0,
    n_jobs=-1,
)

y_train_log = np.log1p(y)
model.fit(X_train, y_train_log)

preds_log = model.predict(X_test)
preds = np.expm1(preds_log)
preds = np.clip(preds, 0.0, None)

submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": preds}, columns=["key", "fare_amount"]
)
submission.to_csv("submission.csv", index=False)

print(submission.head(20))
