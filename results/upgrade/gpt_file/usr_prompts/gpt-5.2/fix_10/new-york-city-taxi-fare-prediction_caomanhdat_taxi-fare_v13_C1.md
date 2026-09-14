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

3.74883

# 6. Current score

5.99009

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.86501) has done: 'I fix the datetime feature engineering to work with modern pandas by replacing deprecated `.dt.week`/`.dt.weekofyear` with ISO week extraction, and I ensure `pickup_datetime` is fully dropped so XGBoost only sees numeric columns. I also correct the `get_samples_output()` call so it doesn’t incorrectly pass a DataFrame with `key` already removed (which can cause column misalignment). Finally, I update the XGBoost parameter `silent` (deprecated) to `verbosity` while keeping the same model type and training approach, and I write a valid `submission.csv` with the required columns.'
- What this solution (achieved 5.62813) has done: 'You’re currently far above the target RMSE (4.865 vs 3.749; lower is better), so we should make a small, safe improvement without changing the overall approach (still the same simple XGBoost regressor on engineered numeric features). The biggest likely gain with minimal disruption is to remove obvious coordinate outliers (NYC bounds) and to use a more realistic geodesic distance feature (haversine) instead of raw Euclidean degrees, while keeping your existing distance feature as well. I also fix a small bug in `get_samples_output()` (it ignores its argument and relies on global `test`) and ensure we drop rows where datetime parsing failed. These changes typically reduce RMSE materially on this competition while keeping runtime under the limit and still producing a valid `submission.csv`.'
- What this solution (achieved 5.62813) has done: 'You’re currently worse than the target RMSE (5.63 vs 3.75; lower is better), so the smallest safe move is to improve feature quality without changing the model type or training loop. I keep the same XGBRegressor setup but (1) clean the test set using the same coordinate/date sanity rules as train, (2) add a single, classic minimal feature for this competition (straight-line distance in kilometers derived from haversine, i.e., `haversine_km` already exists, plus `haversine_miles` as a simple rescale), and (3) ensure strict train/test feature alignment and NaN handling so XGBoost doesn’t get degraded by coerced missing values. These changes typically reduce RMSE materially while staying within the existing approach and producing the same required `submission.csv` format.'
- What this solution (achieved 5.63694) has done: 'Your current RMSE (5.63) is worse than the target (3.75), so we should make small, safe improvements that keep the same XGBRegressor and training flow but improve signal quality. The biggest low-risk gain on this competition is adding a couple of classic, lightweight geographic features (Manhattan distance in km and bearing) and clipping extreme predictions to a plausible fare range, which usually reduces RMSE without changing the modeling approach. I also add one more standard cleanup: remove extreme fare outliers in training (very high fares) that otherwise distort the squared-error objective. All changes preserve the core logic (same model type, same fit/predict loop, same loss) and still write a valid `submission.csv`.'
- What this solution (achieved 5.54502) has done: 'We’re currently worse than the target RMSE (5.63694 vs 3.74883; lower is better), so the smallest reliable move toward the target is to keep the same XGBRegressor and training flow but strengthen the signal in the existing features. I add one classic, lightweight geographic feature that doesn’t change the modeling approach: absolute deltas in longitude/latitude (often very helpful alongside haversine/manhattan) and a simple “airport proximity” feature (distance to JFK/LGA/EWR) computed from the same coordinates. I also keep train/test feature alignment but switch to using the full training feature set (rather than deriving feature list only from test) to avoid silently dropping informative columns if test cleaning removes rows/columns. These are minimal additions that typically reduce RMSE materially on this competition without altering the core training loop or loss.'
- What this solution (achieved 5.54502) has done: 'Your current RMSE (5.545) is still well above the target (3.749), so we should make a small, legitimate improvement without changing the overall approach (same XGBRegressor + same fit/predict flow). The biggest low-risk gain here is to stop accidentally using a mis-scaled “manhattan_km” (it currently multiplies radians by Earth radius without converting lon/lat deltas to km correctly), because that noisy feature can hurt performance. I fix `manhattan_km` to a proper Manhattan distance in kilometers (using per-degree km scaling at the mean latitude) while keeping all other features and the model/training loop intact. I also ensure `bearing` is converted to degrees (still same feature, just a stable scale) and keep submission generation unchanged.'
- What this solution (achieved 7.64599) has done: 'We’re still far above the target RMSE (5.545 vs 3.749; lower is better), so we need a small but meaningful quality lift without changing the overall approach (same engineered numeric features + single XGBRegressor fit/predict). The biggest low-risk gain is to stop underfitting by increasing XGBoost capacity (more trees + slightly deeper trees) while keeping the exact same objective and training flow. I also add a standard `subsample`/`colsample_bytree` to reduce noise/variance from 5M sampled rows and set `n_jobs` for faster training under the time limit; these are parameter-only changes and preserve core semantics. Submission format/paths remain unchanged and we still clip predictions to the valid fare range.'
- What this solution (achieved 6.08895) has done: 'We’re far worse than the target RMSE (7.65 vs 3.75; lower is better), so we should make the smallest changes that reliably improve generalization without changing the overall approach (single XGBRegressor on engineered numeric features with the same fit/predict flow). The biggest likely issue is that you’re training on the first 5M rows, which are strongly time-ordered and not representative; switching to a deterministic random sample of 5M from the full file usually yields a large RMSE drop on this competition while preserving the same modeling logic. I also add a light, standard cleanup to remove near-zero-distance rides with non-trivial fares (and vice versa) that disproportionately hurt squared error. Everything else (features, model type, training loop, submission format) stays the same and it still writes `submission.csv`.'
- What this solution (achieved 5.99009) has done: 'We’re currently above the target RMSE (6.09 vs 3.75, lower is better), so the smallest reliable move toward the target is to reduce train/test distribution mismatch and add one classic low-risk signal without changing the model type or training loop. I keep your same feature pipeline and XGBRegressor fit/predict flow, but (1) add standard NYC “center/airport” distance features (to Manhattan and airport centroids) which are lightweight and usually improve RMSE a lot, and (2) fix train/test alignment issues by applying the same median-imputation to both sets and dropping any remaining NaNs deterministically. I also make the random row sampling robust by computing the train row count from the file (so the skiprows sampling truly spans the full dataset), which typically improves generalization versus sampling from an assumed total. Submission format/path stays identical and still writes `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
import xgboost as xgb

print(os.listdir("../input"))



## === cell 1
train_path = "../input/train.csv"
test_path = "../input/test.csv"

N_TRAIN = 5_000_000
rng = np.random.RandomState(42)

with open(train_path, "rb") as f:
    n_data_rows = sum(1 for _ in f) - 1  # exclude header

keep = set(rng.choice(np.arange(1, n_data_rows + 1), size=N_TRAIN, replace=False))
skip = lambda i: (i != 0) and (i not in keep)

train = pd.read_csv(train_path, skiprows=skip)
test = pd.read_csv(test_path)



## === cell 2
train.head()



## === cell 3
test.head()



## === cell 4
train.describe()



## === cell 5
test.describe()



## === cell 6
train.isnull().sum()



## === cell 7
test.isnull().sum()




## === cell 8
def handle_date(df):
    df = df.copy()
    df["pickup_datetime"] = (
        df["pickup_datetime"].astype(str).str.replace(" UTC", "", regex=False)
    )
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")

    dt = df["pickup_datetime"].dt
    df["hour_of_day"] = dt.hour
    df["week"] = dt.isocalendar().week.astype("int16")
    df["month"] = dt.month.astype("int16")
    df["year"] = dt.year.astype("int16")
    df["day_of_year"] = dt.dayofyear.astype("int16")
    df["week_of_year"] = df["week"]
    df["weekday"] = dt.weekday.astype("int16")
    df["quarter"] = dt.quarter.astype("int16")
    df["day_of_month"] = dt.day.astype("int16")

    df = df.drop("pickup_datetime", axis=1)
    return df


train = handle_date(train)
test = handle_date(test)




## === cell 9
def handle_distance(df):
    df = df.copy()

    R = 6371.0  # km
    pickup_lat = np.radians(df["pickup_latitude"].astype(float))
    pickup_lon = np.radians(df["pickup_longitude"].astype(float))
    drop_lat = np.radians(df["dropoff_latitude"].astype(float))
    drop_lon = np.radians(df["dropoff_longitude"].astype(float))

    dlat = drop_lat - pickup_lat
    dlon = drop_lon - pickup_lon

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(pickup_lat) * np.cos(drop_lat) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(a))
    df["haversine_km"] = R * c

    df["longitude_distance"] = (df["pickup_longitude"] - df["dropoff_longitude"]).abs()
    df["latitude_distance"] = (df["pickup_latitude"] - df["dropoff_latitude"]).abs()
    df["distance_travelled"] = (
        df["longitude_distance"] ** 2 + df["latitude_distance"] ** 2
    ) ** 0.5
    df = df.drop(["longitude_distance", "latitude_distance"], axis=1)

    df["haversine_miles"] = df["haversine_km"] * 0.621371
    return df


train = handle_distance(train)
test = handle_distance(test)




## === cell 10
def add_geo_features(df):
    df = df.copy()

    lat1 = df["pickup_latitude"].astype(float)
    lon1 = df["pickup_longitude"].astype(float)
    lat2 = df["dropoff_latitude"].astype(float)
    lon2 = df["dropoff_longitude"].astype(float)

    mean_lat_rad = np.radians((lat1 + lat2) / 2.0)
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * np.cos(mean_lat_rad)

    dlat_deg = (lat2 - lat1).abs()
    dlon_deg = (lon2 - lon1).abs()

    df["manhattan_km"] = dlat_deg * km_per_deg_lat + dlon_deg * km_per_deg_lon

    lat1r = np.radians(lat1)
    lon1r = np.radians(lon1)
    lat2r = np.radians(lat2)
    lon2r = np.radians(lon2)

    y = np.sin(lon2r - lon1r) * np.cos(lat2r)
    x = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(
        lon2r - lon1r
    )
    df["bearing"] = np.degrees(np.arctan2(y, x))

    return df


train = add_geo_features(train)
test = add_geo_features(test)




## === cell 11
def add_delta_and_airport_features(df):
    df = df.copy()

    df["abs_lon_diff"] = (df["pickup_longitude"] - df["dropoff_longitude"]).abs()
    df["abs_lat_diff"] = (df["pickup_latitude"] - df["dropoff_latitude"]).abs()

    def haversine_to_point_km(lat, lon, lat0, lon0):
        R = 6371.0
        lat = np.radians(lat.astype(float))
        lon = np.radians(lon.astype(float))
        lat0 = np.radians(float(lat0))
        lon0 = np.radians(float(lon0))
        dlat = lat - lat0
        dlon = lon - lon0
        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat) * np.cos(lat0) * np.sin(dlon / 2.0) ** 2
        )
        return 2.0 * R * np.arcsin(np.sqrt(a))

    JFK = (40.6413, -73.7781)
    LGA = (40.7769, -73.8740)
    EWR = (40.6895, -74.1745)

    df["pickup_jfk_km"] = haversine_to_point_km(
        df["pickup_latitude"], df["pickup_longitude"], JFK[0], JFK[1]
    )
    df["pickup_lga_km"] = haversine_to_point_km(
        df["pickup_latitude"], df["pickup_longitude"], LGA[0], LGA[1]
    )
    df["pickup_ewr_km"] = haversine_to_point_km(
        df["pickup_latitude"], df["pickup_longitude"], EWR[0], EWR[1]
    )

    df["dropoff_jfk_km"] = haversine_to_point_km(
        df["dropoff_latitude"], df["dropoff_longitude"], JFK[0], JFK[1]
    )
    df["dropoff_lga_km"] = haversine_to_point_km(
        df["dropoff_latitude"], df["dropoff_longitude"], LGA[0], LGA[1]
    )
    df["dropoff_ewr_km"] = haversine_to_point_km(
        df["dropoff_latitude"], df["dropoff_longitude"], EWR[0], EWR[1]
    )

    df["pickup_airport_km"] = np.minimum.reduce(
        [df["pickup_jfk_km"], df["pickup_lga_km"], df["pickup_ewr_km"]]
    )
    df["dropoff_airport_km"] = np.minimum.reduce(
        [df["dropoff_jfk_km"], df["dropoff_lga_km"], df["dropoff_ewr_km"]]
    )

    return df


train = add_delta_and_airport_features(train)
test = add_delta_and_airport_features(test)




## === cell 12
def add_city_center_features(df):
    df = df.copy()

    def haversine_to_point_km(lat, lon, lat0, lon0):
        R = 6371.0
        lat = np.radians(pd.to_numeric(lat, errors="coerce"))
        lon = np.radians(pd.to_numeric(lon, errors="coerce"))
        lat0 = np.radians(float(lat0))
        lon0 = np.radians(float(lon0))
        dlat = lat - lat0
        dlon = lon - lon0
        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat) * np.cos(lat0) * np.sin(dlon / 2.0) ** 2
        )
        return 2.0 * R * np.arcsin(np.sqrt(a))

    MANH = (40.7580, -73.9855)
    MIDTOWN = (40.7549, -73.9840)
    DOWNTOWN = (40.7060, -74.0086)
    BROOKLYN = (40.6782, -73.9442)

    df["pickup_manh_km"] = haversine_to_point_km(
        df["pickup_latitude"], df["pickup_longitude"], MANH[0], MANH[1]
    )
    df["dropoff_manh_km"] = haversine_to_point_km(
        df["dropoff_latitude"], df["dropoff_longitude"], MANH[0], MANH[1]
    )

    df["pickup_midtown_km"] = haversine_to_point_km(
        df["pickup_latitude"], df["pickup_longitude"], MIDTOWN[0], MIDTOWN[1]
    )
    df["dropoff_midtown_km"] = haversine_to_point_km(
        df["dropoff_latitude"], df["dropoff_longitude"], MIDTOWN[0], MIDTOWN[1]
    )

    df["pickup_downtown_km"] = haversine_to_point_km(
        df["pickup_latitude"], df["pickup_longitude"], DOWNTOWN[0], DOWNTOWN[1]
    )
    df["dropoff_downtown_km"] = haversine_to_point_km(
        df["dropoff_latitude"], df["dropoff_longitude"], DOWNTOWN[0], DOWNTOWN[1]
    )

    df["pickup_brooklyn_km"] = haversine_to_point_km(
        df["pickup_latitude"], df["pickup_longitude"], BROOKLYN[0], BROOKLYN[1]
    )
    df["dropoff_brooklyn_km"] = haversine_to_point_km(
        df["dropoff_latitude"], df["dropoff_longitude"], BROOKLYN[0], BROOKLYN[1]
    )

    return df


train = add_city_center_features(train)
test = add_city_center_features(test)



## === cell 13
train.describe()




## === cell 14
def clean_up_train(train_df):
    train_df = train_df.dropna()

    date_cols = [
        "hour_of_day",
        "week",
        "month",
        "year",
        "day_of_year",
        "weekday",
        "quarter",
        "day_of_month",
    ]
    train_df = train_df.dropna(subset=date_cols)

    train_df = train_df[train_df["fare_amount"] > 0]
    train_df = train_df[train_df["passenger_count"] > 0]
    train_df = train_df[train_df["passenger_count"] < 7]

    train_df = train_df[
        (train_df["pickup_longitude"].between(-74.5, -72.8))
        & (train_df["dropoff_longitude"].between(-74.5, -72.8))
        & (train_df["pickup_latitude"].between(40.5, 41.8))
        & (train_df["dropoff_latitude"].between(40.5, 41.8))
    ]

    train_df = train_df[train_df["haversine_km"].between(0.0, 200.0)]

    train_df = train_df[train_df["fare_amount"].between(2.5, 250.0)]

    near_zero = train_df["haversine_km"] < 0.01
    train_df = train_df[~(near_zero & (train_df["fare_amount"] > 10.0))]

    return train_df


train = clean_up_train(train)
train.describe()




## === cell 15
def clean_up_test(test_df):
    test_df = test_df.copy()

    date_cols = [
        "hour_of_day",
        "week",
        "month",
        "year",
        "day_of_year",
        "weekday",
        "quarter",
        "day_of_month",
    ]
    test_df = test_df.dropna(subset=date_cols)

    test_df["passenger_count"] = pd.to_numeric(
        test_df["passenger_count"], errors="coerce"
    )
    test_df.loc[~test_df["passenger_count"].between(1, 6), "passenger_count"] = 1

    for col, lo, hi in [
        ("pickup_longitude", -74.5, -72.8),
        ("dropoff_longitude", -74.5, -72.8),
        ("pickup_latitude", 40.5, 41.8),
        ("dropoff_latitude", 40.5, 41.8),
    ]:
        test_df[col] = pd.to_numeric(test_df[col], errors="coerce")
        test_df[col] = test_df[col].clip(lo, hi)

    test_df["haversine_km"] = pd.to_numeric(
        test_df["haversine_km"], errors="coerce"
    ).clip(0.0, 200.0)
    test_df["haversine_miles"] = pd.to_numeric(
        test_df["haversine_miles"], errors="coerce"
    ).clip(0.0, 200.0 * 0.621371)
    test_df["distance_travelled"] = pd.to_numeric(
        test_df["distance_travelled"], errors="coerce"
    )

    test_df["manhattan_km"] = pd.to_numeric(test_df["manhattan_km"], errors="coerce")
    test_df["bearing"] = pd.to_numeric(test_df["bearing"], errors="coerce")

    for c in [
        "abs_lon_diff",
        "abs_lat_diff",
        "pickup_jfk_km",
        "pickup_lga_km",
        "pickup_ewr_km",
        "dropoff_jfk_km",
        "dropoff_lga_km",
        "dropoff_ewr_km",
        "pickup_airport_km",
        "dropoff_airport_km",
        "pickup_manh_km",
        "dropoff_manh_km",
        "pickup_midtown_km",
        "dropoff_midtown_km",
        "pickup_downtown_km",
        "dropoff_downtown_km",
        "pickup_brooklyn_km",
        "dropoff_brooklyn_km",
    ]:
        if c in test_df.columns:
            test_df[c] = pd.to_numeric(test_df[c], errors="coerce")

    num_cols = test_df.drop(columns=["key"], errors="ignore").columns
    med = test_df[num_cols].median(numeric_only=True)
    test_df[num_cols] = test_df[num_cols].fillna(med)

    return test_df


test = clean_up_test(test)




## === cell 16
def get_samples_output(train_df):
    feature_cols = [c for c in train_df.columns if c not in ["key", "fare_amount"]]
    return (train_df[feature_cols], train_df["fare_amount"], feature_cols)


samples_train, samples_label, feature_cols = get_samples_output(train)



## === cell 17
samples_train = samples_train.apply(pd.to_numeric, errors="coerce")
test_features = (
    test.reindex(columns=["key"] + feature_cols)
    .drop("key", axis=1)
    .apply(pd.to_numeric, errors="coerce")
)

train_medians = samples_train.median(numeric_only=True)
samples_train = samples_train.fillna(train_medians)
test_features = test_features.fillna(train_medians)

samples_train = samples_train.fillna(0.0)
test_features = test_features.fillna(0.0)

test_features = test_features.reindex(columns=samples_train.columns)

model = xgb.XGBRegressor(
    max_depth=6,
    n_estimators=400,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    verbosity=0,
    objective="reg:squarederror",
    random_state=0,
    n_jobs=4,
)

model.fit(samples_train, samples_label)

preds = model.predict(test_features)

preds = np.clip(preds, 2.5, 250.0)

submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": preds},
    columns=["key", "fare_amount"],
)
submission.to_csv("submission.csv", index=False)
submission.head(20)
