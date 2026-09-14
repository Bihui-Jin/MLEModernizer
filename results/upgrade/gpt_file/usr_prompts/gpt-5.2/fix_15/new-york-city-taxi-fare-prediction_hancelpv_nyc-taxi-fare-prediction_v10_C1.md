# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

4.22488

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 5.34176) has done: 'I fix the runtime errors caused by pandas API changes by replacing the deprecated `weekday_name` accessor with the supported `day_name()` and by ensuring `pickup_datetime` does not remain as a Timestamp in the model matrix. I also make the train/test concatenation consistent by reading only the needed test columns (including `key`) and dropping `key` from the feature set so it can’t leak into `X`. These changes are minimal and preserve the original feature engineering + RandomForest training flow, while making the pipeline run end-to-end and write a valid `submission.csv` with `key,fare_amount`. This should also improve score versus the broken run because the intended time features actually be created and used.'
- What this solution (achieved 5.72582) has done: 'Your current gap is 5.34176 − 4.22488 = 1.11688 (lower is better), so we should improve the RMSE but with minimal, low-risk changes that preserve your RandomForest + same engineered features pipeline. The biggest controllable issues here are (1) the model is using default hyperparameters and (2) a few strong, standard NYC Taxi data cleaning filters (NYC bounding box + fare upper cap) are missing, which typically reduces label noise a lot without changing the core approach. I add those minimal cleaning filters and set conservative RandomForest hyperparameters (more trees, fixed seed, sensible depth/min leaf) to move the score toward the target while keeping the same model family and feature set. I also ensure the train/test one-hot alignment stays identical (still via concatenation) and keep the submission schema unchanged.'
- What this solution (achieved 5.12493) has done: 'Your score gap is 5.72582 − 4.22488 = 1.50094 RMSE (lower is better), so we should improve accuracy with minimal, low-risk changes that keep the same RandomForest + feature engineering pipeline. The biggest issue is you only train on `nrows=10000`, which is far too small for this problem; increasing this to a still-manageable sample size usually yields a large RMSE drop without changing core logic. I also apply the same NYC bounding-box/valid-lat-lon filters to the test set (for safer feature distribution) and handle those filtered rows by falling back to the training mean fare so the submission stays complete and valid. Finally, I ensure any remaining missing engineered features (from invalid datetimes) are filled deterministically before `get_dummies`, preventing silent row drops or NaNs affecting the model.'
- What this solution (achieved 5.23349) has done: 'We should move RMSE down (lower is better) from 5.12493 toward 4.22488, so the smallest safe lever is better data quality and slightly stronger RandomForest generalization without changing the model family or feature set. I (1) add one standard, low-risk cleaning rule that removes “zero distance” trips (pickup==dropoff), which are mostly label noise, and (2) add the missing numeric geo feature (`squared_long`/`squared_lat`) that your code already computes but never feeds to the model. I also use `max_features="sqrt"` (a common RF default for better generalization) while keeping the same training loop and core pipeline, and keep the submission schema identical.'
- What this solution (achieved 5.10887) has done: 'To move RMSE down toward the 4.22488 target (lower is better) with minimal risk, I keep your exact RandomForest + engineered time/geo features pipeline but reduce label noise and add one strong, standard geo feature. Specifically: (1) add a conservative minimum-trip-distance filter using your existing lat/lon diffs to remove near-zero trips that are often mislabels, (2) add haversine distance as an additional numeric feature (no change to model family/training loop), and (3) slightly increase the number of trees for more stable generalization while keeping all other hyperparameters intact. These changes are small, legitimate, and typically reduce RMSE without altering the core approach or submission semantics.'
- What this solution (achieved 5.23889) has done: 'Your current RMSE (5.10887) is above the target (4.22488), so we should improve accuracy with the smallest low-risk changes that keep the same RandomForest + engineered time/geo feature pipeline. The biggest lever without changing core logic is to reduce label noise further by filtering out implausible long trips and fares-per-distance outliers using your already-computed haversine distance. I add two conservative training-only filters (max haversine and reasonable $/km bounds) and keep the test path unchanged except for using the existing fallback for invalid rows. This typically moves RMSE down while preserving the same model, features, training approach, and submission semantics.'
- What this solution (achieved 5.10887) has done: 'We should move RMSE down (lower is better) from 5.23889 toward the 4.22488 target, so I keep your exact RandomForest + current engineered features pipeline but make two minimal, high-impact data-quality fixes that typically reduce noise: (1) ensure passenger_count is also filtered in the test-valid mask (so the model isn’t asked to extrapolate for invalid counts), and (2) add a conservative “reasonable speed” filter on the training set using the already-computed haversine distance and pickup_datetime to remove mislabeled/outlier trips. These do not change model architecture, loss, or feature engineering; they only remove clearly implausible training rows and prevent invalid test rows from being treated as in-distribution. The submission format and fallback behavior stay identical.'
- What this solution (achieved 5.36625) has done: 'I fix the runtime error in the additional cleaning cell by ensuring we extract hours from `pickup_datetime` in a way that works whether `pd.to_datetime(...)` returns a Series or a DatetimeIndex. This is a correctness/stability change only and preserves your existing “reasonable speed” filter logic and overall RandomForest + feature engineering pipeline. I also add a small guard so the mask aligns with `y` length to avoid accidental misalignment if parsing creates NaTs. The rest of the training, feature set, and submission writing stays the same.'
- What this solution (achieved 5.25002) has done: 'Your RMSE (5.36625) is still above the target (4.22488), so we should make the smallest reliable improvement without changing the core RandomForest + feature-engineering pipeline. The biggest current accuracy drag is that the “speed” filter uses a synthetic duration proxy (based only on hour), which can incorrectly remove many good training rows and bias the model; I replace it with a standard, conservative outlier filter that uses only your already-computed haversine distance and fare (fare-per-km bounds + max distance), keeping everything else identical. I also add one minimal, metric-aligned post-processing step: clip predictions to a reasonable upper bound (the same 250 cap you already apply to training fares) to reduce RMSE impact from occasional extreme RF outputs. These changes are directly tied to reducing noise/outlier impact while preserving model family, features, training flow, and submission semantics.'
- What this solution (achieved 8.8052) has done: 'The crash comes from building `all_data`/`x_test_valid` using an earlier version of `model_test_mask`, then later shrinking `model_test_mask` again with the airport filter—so the number of predicted rows no longer matches the final mask. I keep your exact model and feature engineering, but recompute the “valid rows to model” mask once (including the airport exclusion) *before* concatenating train+test for encoding, and I only predict on those rows. I also add a small alignment guard (reindex) to ensure the boolean mask always aligns to the test index and avoid silent misalignment with pandas. This is a correctness/stability fix and produce a valid `submission.csv`.'
- What this solution (achieved 8.8052) has done: 'Your current RMSE (8.8052) is far above the target (4.22488), so we should fix the most likely correctness bug that can silently wreck score: after you further filter the *training* rows (fare/km + airport removal), you rebuild `all_data` with `ignore_index=True`, which breaks alignment between the original `model_test_mask` (indexed like `test`) and the rows in `x_test_valid`; this can cause predictions to be written onto the wrong test keys even though lengths match. I preserve your exact model, features, and cleaning logic, but carry the test `key` through the concatenation and reconstruct the “model rows” boolean mask from that key after all filtering/concats, so predictions map back to the correct test rows. I also ensure we never one-hot encode `key` (to avoid leakage/huge sparsity) while still using it only for alignment. These are minimal stability/correctness fixes that should substantially reduce RMSE toward the target without changing the core approach.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import os

for p in [
    "../input",
    "/kaggle/input",
    "/kaggle/data",
    "/kaggle/data/new-york-city-taxi-fare-prediction",
]:
    if os.path.exists(p):
        print(p, "->", os.listdir(p)[:10])



## === cell 1
types = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}

cols_train = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

cols_test = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]




## === cell 2
def resolve_path(fname):
    candidates = [
        f"../input/{fname}",
        f"/kaggle/input/{fname}",
        f"/kaggle/data/{fname}",
        f"/kaggle/data/new-york-city-taxi-fare-prediction/{fname}",
        f"/kaggle/input/new-york-city-taxi-fare-prediction/{fname}",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return f"../input/{fname}"


train_path = resolve_path("train.csv")
test_path = resolve_path("test.csv")
samp_path = resolve_path("sample_submission.csv")

NROWS_TRAIN = 300000  # chosen to stay within typical Kaggle notebook time/memory

train = pd.read_csv(train_path, nrows=NROWS_TRAIN, usecols=cols_train, dtype=types)
test = pd.read_csv(test_path, usecols=cols_test)
samp = pd.read_csv(samp_path)



## === cell 3
train.dropna(how="any", axis="rows", inplace=True)
train = train[train.fare_amount > 0]
train = train[train["passenger_count"] <= 6]
train = train[train["fare_amount"] <= 250]



## === cell 4
latitude_mask_pickup = (train.pickup_latitude > -90) & (train.pickup_latitude < 90)
train = train[latitude_mask_pickup]

latitude_mask_dropoff = (train.dropoff_latitude > -90) & (train.dropoff_latitude < 90)
train = train[latitude_mask_dropoff]



## === cell 5
longitude_mask_pickup = (train.pickup_longitude > -180) & (train.pickup_longitude < 180)
train = train[longitude_mask_pickup]

longitude_mask_dropoff = (train.dropoff_longitude > -180) & (
    train.dropoff_longitude < 180
)
train = train[longitude_mask_dropoff]

nyc_lat_min, nyc_lat_max = 40.5, 41.0
nyc_lon_min, nyc_lon_max = -74.3, -73.6

train = train[
    (train.pickup_latitude.between(nyc_lat_min, nyc_lat_max))
    & (train.dropoff_latitude.between(nyc_lat_min, nyc_lat_max))
    & (train.pickup_longitude.between(nyc_lon_min, nyc_lon_max))
    & (train.dropoff_longitude.between(nyc_lon_min, nyc_lon_max))
]

train = train[
    ~(
        (train["pickup_latitude"] == train["dropoff_latitude"])
        & (train["pickup_longitude"] == train["dropoff_longitude"])
    )
]

abs_dlon = (train["dropoff_longitude"] - train["pickup_longitude"]).abs()
abs_dlat = (train["dropoff_latitude"] - train["pickup_latitude"]).abs()
train = train[(abs_dlon + abs_dlat) >= 0.001]

test_id = test.key
test_features_only = test.drop(columns=["key"]).copy()

test_valid = (
    test_features_only.pickup_latitude.between(-90, 90)
    & test_features_only.dropoff_latitude.between(-90, 90)
    & test_features_only.pickup_longitude.between(-180, 180)
    & test_features_only.dropoff_longitude.between(-180, 180)
    & test_features_only.pickup_latitude.between(nyc_lat_min, nyc_lat_max)
    & test_features_only.dropoff_latitude.between(nyc_lat_min, nyc_lat_max)
    & test_features_only.pickup_longitude.between(nyc_lon_min, nyc_lon_max)
    & test_features_only.dropoff_longitude.between(nyc_lon_min, nyc_lon_max)
    & (test_features_only["passenger_count"] <= 6)
)




## === cell 6
def near_point(lat, lon, lat0, lon0, r_deg=0.015):
    return (lat.between(lat0 - r_deg, lat0 + r_deg)) & (
        lon.between(lon0 - r_deg, lon0 + r_deg)
    )


JFK_LAT, JFK_LON = 40.6413, -73.7781
LGA_LAT, LGA_LON = 40.7769, -73.8740
EWR_LAT, EWR_LON = 40.6895, -74.1745

pickup_airport_test = (
    near_point(
        test_features_only["pickup_latitude"],
        test_features_only["pickup_longitude"],
        JFK_LAT,
        JFK_LON,
    )
    | near_point(
        test_features_only["pickup_latitude"],
        test_features_only["pickup_longitude"],
        LGA_LAT,
        LGA_LON,
    )
    | near_point(
        test_features_only["pickup_latitude"],
        test_features_only["pickup_longitude"],
        EWR_LAT,
        EWR_LON,
    )
)
dropoff_airport_test = (
    near_point(
        test_features_only["dropoff_latitude"],
        test_features_only["dropoff_longitude"],
        JFK_LAT,
        JFK_LON,
    )
    | near_point(
        test_features_only["dropoff_latitude"],
        test_features_only["dropoff_longitude"],
        LGA_LAT,
        LGA_LON,
    )
    | near_point(
        test_features_only["dropoff_latitude"],
        test_features_only["dropoff_longitude"],
        EWR_LAT,
        EWR_LON,
    )
)

model_test_mask = (test_valid & ~(pickup_airport_test | dropoff_airport_test)).copy()
model_test_mask = model_test_mask.reindex(test.index, fill_value=False)



## === cell 7
test_model_part = test.loc[model_test_mask, cols_test].copy()

train_part = train.drop(columns=["fare_amount"]).copy()
train_part["__key__"] = pd.NA
train_part["__is_test_row__"] = False
train_part["__test_index__"] = -1

test_part = test_model_part.copy()
test_part = test_part.rename(columns={"key": "__key__"})
test_part["__is_test_row__"] = True
test_part["__test_index__"] = test_part.index.astype(np.int64)

all_data = pd.concat((train_part, test_part), axis=0)  # keep original indices

y = train.fare_amount.values
n_train = len(train)
n_test_model = int(model_test_mask.sum())

fallback_fare = float(np.mean(y))




## === cell 8
def week_num(day):
    """
    given the day of the month, return the week number of the month
    """
    if day <= 7:
        return "first"
    if (day > 7) and (day <= 14):
        return "second"
    if (day > 14) and (day <= 21):
        return "third"
    if (day > 21) and (day <= 28):
        return "fourth"
    return "fifth"




## === cell 9
def add_time_features(data):
    data = data.copy()
    data["pickup_datetime"] = pd.to_datetime(data["pickup_datetime"], errors="coerce")

    data["hour"] = data["pickup_datetime"].dt.hour
    data["day_of_week"] = data["pickup_datetime"].dt.day_name()
    data["day_of_month"] = data["pickup_datetime"].dt.day
    data["week_of_month"] = data["day_of_month"].map(week_num)
    data["month"] = data["pickup_datetime"].dt.month
    data["year"] = data["pickup_datetime"].dt.year

    data["hour"] = data["hour"].astype("Int64").astype(str)
    data["month"] = data["month"].astype("Int64").astype(str)
    data["year"] = data["year"].astype("Int64").astype(str)

    data["day_of_week"] = data["day_of_week"].fillna("Unknown")
    data["week_of_month"] = data["week_of_month"].fillna("Unknown")
    data["hour"] = data["hour"].replace({"<NA>": "Unknown"})
    data["month"] = data["month"].replace({"<NA>": "Unknown"})
    data["year"] = data["year"].replace({"<NA>": "Unknown"})

    data.drop(["pickup_datetime", "day_of_month"], axis=1, inplace=True)
    return data




## === cell 10
def add_geo_features(data):
    data = data.copy()
    data["abs_diff_longitude"] = (data.dropoff_longitude - data.pickup_longitude).abs()
    data["abs_diff_latitude"] = (data.dropoff_latitude - data.pickup_latitude).abs()

    data["manhattan_distance"] = data["abs_diff_longitude"] + data["abs_diff_latitude"]

    data["squared_long"] = np.power(data["abs_diff_longitude"], 2)
    data["squared_lat"] = np.power(data["abs_diff_latitude"], 2)

    lat1 = np.deg2rad(data["pickup_latitude"].astype("float64"))
    lon1 = np.deg2rad(data["pickup_longitude"].astype("float64"))
    lat2 = np.deg2rad(data["dropoff_latitude"].astype("float64"))
    lon2 = np.deg2rad(data["dropoff_longitude"].astype("float64"))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(a))
    earth_radius_km = 6371.0
    data["haversine_km"] = earth_radius_km * c

    return data




## === cell 11
all_data = add_time_features(all_data)
all_data = add_geo_features(all_data)



## === cell 12
haversine_all = all_data["haversine_km"].values
is_test_row = all_data["__is_test_row__"].values.astype(bool)

h_train = haversine_all[~is_test_row].astype(np.float64, copy=False)

max_km = 60.0
mask_km = h_train <= max_km

km_safe = np.maximum(h_train, 0.1)
fare_per_km = y.astype(np.float64, copy=False) / km_safe
mask_fpk = (fare_per_km >= 1.0) & (fare_per_km <= 35.0)

mask_keep = mask_km & mask_fpk

train_idx = all_data.index[~is_test_row]
test_idx = all_data.index[is_test_row]

train_keep_idx = train_idx[mask_keep]
all_data_train_filtered = all_data.loc[train_keep_idx].copy()
y = y[mask_keep]

all_data_test_part = all_data.loc[test_idx].copy()

all_data = pd.concat([all_data_train_filtered, all_data_test_part], axis=0)

fallback_fare = float(np.mean(y))



## === cell 13
train_geo = all_data.loc[
    all_data["__is_test_row__"] == False,
    ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"],
].copy()

pickup_near_airport = (
    near_point(
        train_geo["pickup_latitude"], train_geo["pickup_longitude"], JFK_LAT, JFK_LON
    )
    | near_point(
        train_geo["pickup_latitude"], train_geo["pickup_longitude"], LGA_LAT, LGA_LON
    )
    | near_point(
        train_geo["pickup_latitude"], train_geo["pickup_longitude"], EWR_LAT, EWR_LON
    )
)
dropoff_near_airport = (
    near_point(
        train_geo["dropoff_latitude"], train_geo["dropoff_longitude"], JFK_LAT, JFK_LON
    )
    | near_point(
        train_geo["dropoff_latitude"], train_geo["dropoff_longitude"], LGA_LAT, LGA_LON
    )
    | near_point(
        train_geo["dropoff_latitude"], train_geo["dropoff_longitude"], EWR_LAT, EWR_LON
    )
)

mask_airport_train = ~(pickup_near_airport | dropoff_near_airport)
train_rows_current = all_data.loc[all_data["__is_test_row__"] == False].copy()

all_data_train = train_rows_current.loc[mask_airport_train].copy()
y = y[mask_airport_train.values]

all_data_test = all_data.loc[all_data["__is_test_row__"] == True].copy()

all_data = pd.concat([all_data_train, all_data_test], axis=0)
fallback_fare = float(np.mean(y))



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/1501649180.py in <cell line: 0>()
     32 
     33 all_data_train = train_rows_current.loc[mask_airport_train].copy()
---> 34 y = y[mask_airport_train.values]
     35 
     36 all_data_test = all_data.loc[all_data["__is_test_row__"] == True].copy()

IndexError: boolean index did not match indexed array along dimension 0; dimension is 287557 but corresponding boolean dimension is 296361

## === cell 14
is_test_row_final = all_data["__is_test_row__"].values.astype(bool)
test_index_in_matrix = (
    all_data.loc[is_test_row_final, "__test_index__"].astype(np.int64).values
)

features = [
    "passenger_count",
    "hour",
    "day_of_week",
    "week_of_month",
    "month",
    "year",
    "abs_diff_longitude",
    "abs_diff_latitude",
    "manhattan_distance",
    "squared_long",
    "squared_lat",
    "haversine_km",
]

all_data = all_data[["__is_test_row__", "__test_index__"] + features]

num_cols = [
    "abs_diff_longitude",
    "abs_diff_latitude",
    "manhattan_distance",
    "squared_long",
    "squared_lat",
    "haversine_km",
]
all_data[num_cols] = all_data[num_cols].fillna(0.0)

all_data_dum = pd.get_dummies(
    all_data.drop(columns=["__is_test_row__", "__test_index__"])
)



## === cell 15
n_train = int((~is_test_row_final).sum())
x = all_data_dum.iloc[:n_train, :]
x_test_valid = all_data_dum.iloc[n_train:, :]



## === cell 16
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=450,
    random_state=42,
    n_jobs=-1,
    max_depth=18,
    min_samples_leaf=2,
    max_features="sqrt",
)



## === cell 17
model.fit(x, y)

test_pred_valid = model.predict(x_test_valid)
test_pred_valid = np.clip(test_pred_valid, 0, 250.0)

test_pred_full = np.full(shape=(len(test),), fill_value=fallback_fare, dtype=np.float64)

if len(test_index_in_matrix) != len(test_pred_valid):
    raise RuntimeError(
        f"Index/prediction length mismatch: len(test_index_in_matrix)={len(test_index_in_matrix)} "
        f"but len(test_pred_valid)={len(test_pred_valid)}"
    )

test_pred_full[test_index_in_matrix] = test_pred_valid.astype(np.float64)

sub = pd.DataFrame({"key": test_id, "fare_amount": test_pred_full})
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Base-valid test rows:", int(test_valid.sum()), "out of", len(test))
print("Model-predicted test rows:", int(model_test_mask.sum()), "out of", len(test))
print("Training rows used after cleaning:", n_train)
print("Wrote submission.csv with shape:", sub.shape)

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/994211765.py in <cell line: 0>()
----> 1 model.fit(x, y)
      2 
      3 test_pred_valid = model.predict(x_test_valid)
      4 test_pred_valid = np.clip(test_pred_valid, 0, 250.0)
      5 

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_forest.py in fit(self, X, y, sample_weight)
    343         if issparse(y):
    344             raise ValueError("sparse multilabel-indicator for y is not supported.")
--> 345         X, y = self._validate_data(
    346             X, y, multi_output=True, accept_sparse="csc", dtype=DTYPE
    347         )

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    582                 y = check_array(y, input_name="y", **check_y_params)
    583             else:
--> 584                 X, y = check_X_y(X, y, **check_params)
    585             out = X, y
    586 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_X_y(X, y, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, estimator)
   1122     y = _check_y(y, multi_output=multi_output, y_numeric=y_numeric, estimator=estimator)
   1123 
-> 1124     check_consistent_length(X, y)
   1125 
   1126     return X, y

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_consistent_length(*arrays)
    395     uniques = np.unique(lengths)
    396     if len(uniques) > 1:
--> 397         raise ValueError(
    398             "Found input variables with inconsistent numbers of samples: %r"
    399             % [int(l) for l in lengths]

ValueError: Found input variables with inconsistent numbers of samples: [296361, 287557]
