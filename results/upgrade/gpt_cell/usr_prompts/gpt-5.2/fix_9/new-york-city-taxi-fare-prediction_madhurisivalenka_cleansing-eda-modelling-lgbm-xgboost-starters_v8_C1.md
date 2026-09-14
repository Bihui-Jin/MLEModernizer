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

3.39809

# 6. Current score

4.80418

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 16.9774) has done: 'The timeout is driven by expensive Pandas operations repeated many times on a 1M-row frame (especially `drop(...index...)`, multiple boolean-filter materializations, and several `.apply(axis=1)` row-wise calls), plus heavy plotting/sorting that is irrelevant to producing the submission. I keep the exact same cleaning rules, feature engineering, and RandomForest training/prediction, but make them faster by (1) consolidating row drops into boolean masks (single pass), (2) replacing `.apply(axis=1)` with vectorized arithmetic on the same subsets, (3) avoiding `train.update(...)` in favor of direct `.loc[...]` assignment, and (4) disabling all plotting/expensive exploratory outputs so runtime is spent only on the pipeline needed for the final predictions.'
- What this solution (achieved 16.77318) has done: 'You’re currently crippling model performance by coercing the `key` (a string id) into `datetime`, which produces many `NaT` values and can silently misalign/contaminate later steps; keeping `key` as the original string also ensures the submission rows match the test set exactly. The other major issue is that you never filter invalid longitude ranges for `dropoff_longitude` (cell 32 mistakenly checks `dropoff_latitude` against [-180, 180]); fixing that restores a large amount of correct training signal without changing the model/feature set. Finally, to move RMSE closer to your target without changing the model, we clip negative predictions to 0 (fares can’t be negative), which typically improves RMSE on this competition. All changes are minimal and keep your RandomForest training and feature engineering intact.'
- What this solution (achieved 10.54572) has done: 'Your RMSE is far from the target, so we make a minimal, high-impact fix that doesn’t change the model: remove extreme/unrealistic fares and rides that corrupt training (a very common cause of ~15–20 RMSE in this competition). Specifically, we (1) filter to NYC-like coordinate bounds (instead of the full world range) and (2) drop extreme outlier fares (e.g., > 250) and extreme distances, which typically stabilizes RandomForest regression a lot. We keep your existing feature engineering (Haversine + datetime parts) and the same RandomForestRegressor training/prediction, and we still write `submission_1.csv` with the required columns. These filters are standard “data cleaning” and are the smallest change likely to move RMSE substantially closer to ~3.4.'
- What this solution (achieved 7.13382) has done: 'Your current RMSE (10.55) is still far above the target (3.40), so the smallest likely improvement is to remove remaining “impossible” training rows that the model can’t learn (e.g., passenger_count of 0, and obviously invalid coordinates like 0/0 that slipped through earlier filters) without changing your model or features. We also make the train/test feature columns explicitly aligned (same order) to avoid any silent column-order mismatch bugs that can severely hurt RandomForest performance. Finally, we keep your exact RandomForestRegressor setup and prediction pipeline, but add only a very small, standard distance-based cleaning rule: drop tiny-distance rides with large fares (data errors) which often disproportionately inflate RMSE in this competition.'
- What this solution (achieved 6.12613) has done: 'Your current RMSE (7.13382) is still far above the target (3.39809), so we need a small but meaningful improvement without changing the core RandomForest approach. The biggest remaining bottleneck is that the default `RandomForestRegressor` (no depth/leaf constraints) tends to overfit noisy taxi data and generalize poorly; adding very light regularization (`min_samples_leaf` and `max_features`) keeps the same model class/training loop but usually reduces RMSE materially on this competition. I also fix `y_train` to be a 1D Series (same semantics, avoids any multioutput edge-cases) and ensure the train/test feature alignment stays explicit. The submission format and path remain unchanged (`submission_1.csv` with `key,fare_amount`).'
- What this solution (achieved 4.6876) has done: 'Your current RMSE (6.126) is still well above the target (3.398), so we should make a small, high-impact improvement without changing the core RandomForest approach. The most direct lever here is to align the model’s features with the real NYC fare structure by adding a few simple, standard geospatial features (Manhattan distance, bearing, and “airport trip” flags), which typically improves generalization on this competition while keeping the same training loop and model class. We also apply the exact same safe coordinate cleaning to the test set (NYC bounds + nonzero coords) to prevent out-of-distribution inputs that can inflate error, while keeping submission row order intact by only masking predictions (not dropping rows). Finally, we keep your existing cleaning rules, haversine feature, and RandomForest hyperparameters, and still write `submission_1.csv` with `key,fare_amount`.'
- What this solution (achieved 4.80418) has done: 'Your RMSE (4.6876) is still above the target (3.39809), so we should make a small, standard improvement that keeps your exact RandomForest approach and existing features. The biggest remaining issue is that your engineered distance features (`H_Distance`, `manhattan_dist`) are in degrees/km mismatched scales and the model is trained on only 1M rows; adding very lightweight, well-known NYC taxi features (straight-line distance in miles, and simple interaction terms with passenger count and time) typically closes a meaningful part of the gap without changing the training loop or model class. We also apply the same “distance sanity” masking to test predictions (like you already do for geo bounds) so out-of-distribution distances don’t get RF extrapolation noise. All changes are additive feature columns and safe masking/post-processing; submission format/path stays `submission_1.csv` with `key,fare_amount`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

print(os.listdir("../input"))

np.random.seed(42)



## === cell 1
train = pd.read_csv(
    "../input/train.csv",
    nrows=1000000,
    parse_dates=["pickup_datetime"],
    dtype={
        "fare_amount": "float64",
        "pickup_longitude": "float64",
        "pickup_latitude": "float64",
        "dropoff_longitude": "float64",
        "dropoff_latitude": "float64",
        "passenger_count": "int64",
    },
)
test = pd.read_csv(
    "../input/test.csv",
    parse_dates=["pickup_datetime"],
    dtype={
        "pickup_longitude": "float64",
        "pickup_latitude": "float64",
        "dropoff_longitude": "float64",
        "dropoff_latitude": "float64",
        "passenger_count": "int64",
    },
)



## === cell 2
train.shape



## === cell 3
test.shape



## === cell 4
train.head(10)



## === cell 5
_ = train.isnull().sum()



## === cell 6
_ = test.isnull().sum()



## === cell 7
train = train.dropna(axis=0, how="any")



## === cell 8
train.shape



## === cell 9
train["fare_amount"].describe()



## === cell 10
from collections import Counter

Counter(train["fare_amount"] < 0)



## === cell 11
train = train.loc[train["fare_amount"] >= 0].copy()
train.shape



## === cell 12
train["fare_amount"].describe()



## === cell 13
train["fare_amount"].nlargest(10)



## === cell 14
train["passenger_count"].describe()



## === cell 15
train[train["passenger_count"] > 6].head()



## === cell 16
train = train.loc[train["passenger_count"] != 208].copy()



## === cell 17
train = train.loc[train["passenger_count"].between(1, 6)].copy()



## === cell 18
train["passenger_count"].describe()



## === cell 19
train["pickup_latitude"].describe()



## === cell 20
train[train["pickup_latitude"] < -90].head()



## === cell 21
train[train["pickup_latitude"] > 90].head()



## === cell 22
train = train.loc[
    (train["pickup_latitude"] >= -90) & (train["pickup_latitude"] <= 90)
].copy()



## === cell 23
train.shape



## === cell 24
train["pickup_longitude"].describe()



## === cell 25
train[train["pickup_longitude"] < -180].head()



## === cell 26
train[train["pickup_longitude"] > 180].head()



## === cell 27
train = train.loc[
    (train["pickup_longitude"] >= -180) & (train["pickup_longitude"] <= 180)
].copy()



## === cell 28
train.shape



## === cell 29
train[train["dropoff_latitude"] < -90].head()



## === cell 30
train[train["dropoff_latitude"] > 90].head()



## === cell 31
train = train.loc[
    (train["dropoff_latitude"] >= -90) & (train["dropoff_latitude"] <= 90)
].copy()



## === cell 32
train.shape



## === cell 33
train[(train["dropoff_longitude"] < -180) | (train["dropoff_longitude"] > 180)].head()



## === cell 34
train = train.loc[
    (train["dropoff_longitude"] >= -180) & (train["dropoff_longitude"] <= 180)
].copy()



## === cell 35
zero_coord = (train["pickup_longitude"].eq(0) & train["pickup_latitude"].eq(0)) | (
    train["dropoff_longitude"].eq(0) & train["dropoff_latitude"].eq(0)
)
train = train.loc[~zero_coord].copy()



## === cell 36
nyc_lon_min, nyc_lon_max = -74.5, -72.8
nyc_lat_min, nyc_lat_max = 40.3, 41.2

geo_mask = (
    train["pickup_longitude"].between(nyc_lon_min, nyc_lon_max)
    & train["dropoff_longitude"].between(nyc_lon_min, nyc_lon_max)
    & train["pickup_latitude"].between(nyc_lat_min, nyc_lat_max)
    & train["dropoff_latitude"].between(nyc_lat_min, nyc_lat_max)
)
train = train.loc[geo_mask].copy()



## === cell 37
train = train.loc[train["fare_amount"].between(2.5, 250.0)].copy()



## === cell 38
train.dtypes



## === cell 39
train.dtypes



## === cell 40
test.dtypes



## === cell 41
train.dtypes



## === cell 42
test.dtypes



## === cell 43
train.head()



## === cell 44
test.head()




## === cell 45
def _add_haversine(df, lat1, lon1, lat2, lon2, out_col="H_Distance"):
    R = 6371.0
    phi1 = np.radians(df[lat1].to_numpy())
    phi2 = np.radians(df[lat2].to_numpy())
    dphi = np.radians((df[lat2] - df[lat1]).to_numpy())
    dlmb = np.radians((df[lon2] - df[lon1]).to_numpy())
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlmb / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    df[out_col] = R * c


def haversine_distance(lat1, long1, lat2, long2):
    _add_haversine(train, lat1, long1, lat2, long2, out_col="H_Distance")
    _add_haversine(test, lat1, long1, lat2, long2, out_col="H_Distance")
    return train["H_Distance"]




## === cell 46
haversine_distance(
    "pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"
)



## === cell 47
train["H_Distance"].head(10)



## === cell 48
test["H_Distance"].head(10)



## === cell 49
train.head(10)



## === cell 50
test.head(10)



## === cell 51
for df in (train, test):
    dt = df["pickup_datetime"].dt
    df["Year"] = dt.year
    df["Month"] = dt.month
    df["Date"] = dt.day
    df["Day of Week"] = dt.dayofweek
    df["Hour"] = dt.hour



## === cell 52
train.head()



## === cell 53
test.head()



## === cell 54
pass



## === cell 55
pass



## === cell 56
pass



## === cell 57
pass



## === cell 58
pass



## === cell 59
pass



## === cell 60
pass



## === cell 61
train[["H_Distance", "fare_amount"]].nlargest(5, "H_Distance")



## === cell 62
len(train)



## === cell 63
pass



## === cell 64
pass



## === cell 65
pass



## === cell 66
train.loc[
    ((train["pickup_latitude"] == 0) & (train["pickup_longitude"] == 0))
    & ((train["dropoff_latitude"] != 0) & (train["dropoff_longitude"] != 0))
    & (train["fare_amount"] == 0)
].head()



## === cell 67
mask_bad1 = (
    (train["pickup_latitude"].eq(0) & train["pickup_longitude"].eq(0))
    & (~train["dropoff_latitude"].eq(0) & ~train["dropoff_longitude"].eq(0))
    & train["fare_amount"].eq(0)
)
train = train.loc[~mask_bad1].copy()



## === cell 68
train.shape



## === cell 69
test.loc[
    ((test["pickup_latitude"] == 0) & (test["pickup_longitude"] == 0))
    & ((test["dropoff_latitude"] != 0) & (test["dropoff_longitude"] != 0))
].head()



## === cell 70
train.loc[
    ((train["pickup_latitude"] != 0) & (train["pickup_longitude"] != 0))
    & ((train["dropoff_latitude"] == 0) & (train["dropoff_longitude"] == 0))
    & (train["fare_amount"] == 0)
].head()



## === cell 71
mask_bad2 = (
    (~train["pickup_latitude"].eq(0) & ~train["pickup_longitude"].eq(0))
    & (train["dropoff_latitude"].eq(0) & train["dropoff_longitude"].eq(0))
    & train["fare_amount"].eq(0)
)
train = train.loc[~mask_bad2].copy()



## === cell 72
train.shape



## === cell 73
test.loc[
    ((test["pickup_latitude"] != 0) & (test["pickup_longitude"] != 0))
    & ((test["dropoff_latitude"] == 0) & (test["dropoff_longitude"] == 0))
].head()



## === cell 74
high_distance = train.loc[
    (train["H_Distance"] > 200) & (train["fare_amount"] != 0)
].copy()



## === cell 75
high_distance.head()



## === cell 76
high_distance.shape



## === cell 77
high_distance.loc[:, "H_Distance"] = (high_distance["fare_amount"] - 2.50) / 1.56



## === cell 78
high_distance.head()



## === cell 79
train.loc[high_distance.index, "H_Distance"] = high_distance["H_Distance"]



## === cell 80
train.shape



## === cell 81
train[train["H_Distance"] == 0].head()



## === cell 82
train[(train["H_Distance"] == 0) & (train["fare_amount"] == 0)].head()



## === cell 83
train = train.loc[~((train["H_Distance"] == 0) & (train["fare_amount"] == 0))].copy()



## === cell 84
train[(train["H_Distance"] == 0)].shape



## === cell 85
rush_hour = train.loc[
    ((train["Hour"] >= 6) & (train["Hour"] <= 20))
    & ((train["Day of Week"] >= 1) & (train["Day of Week"] <= 5))
    & (train["H_Distance"] == 0)
    & (train["fare_amount"] < 2.5)
]
rush_hour.head()



## === cell 86
train = train.drop(rush_hour.index, axis=0)



## === cell 87
train.shape



## === cell 88
non_rush_hour = train.loc[
    (((train["Hour"] < 6) | (train["Hour"] > 20)))
    & ((train["Day of Week"] >= 1) & (train["Day of Week"] <= 5))
    & (train["H_Distance"] == 0)
    & (train["fare_amount"] < 3.0)
]
non_rush_hour.head()



## === cell 89
weekends = train.loc[
    ((train["Day of Week"] == 0) | (train["Day of Week"] == 6))
    & (train["H_Distance"] == 0)
    & (train["fare_amount"] < 3.0)
]
weekends.head()



## === cell 90
train.loc[(train["H_Distance"] != 0) & (train["fare_amount"] == 0)].head()



## === cell 91
scenario_3 = train.loc[(train["H_Distance"] != 0) & (train["fare_amount"] == 0)].copy()



## === cell 92
len(scenario_3)



## === cell 93
scenario_3.nlargest(5, "H_Distance")



## === cell 94
scenario_3.loc[:, "fare_amount"] = (scenario_3["H_Distance"] * 1.56) + 2.50



## === cell 95
scenario_3["fare_amount"].head()



## === cell 96
train.loc[scenario_3.index, "fare_amount"] = scenario_3["fare_amount"]



## === cell 97
train.shape



## === cell 98
train.loc[(train["H_Distance"] == 0) & (train["fare_amount"] != 0)].head()



## === cell 99
scenario_4 = train.loc[(train["H_Distance"] == 0) & (train["fare_amount"] != 0)].copy()



## === cell 100
len(scenario_4)



## === cell 101
scenario_4.loc[
    (scenario_4["fare_amount"] <= 3.0) & (scenario_4["H_Distance"] == 0)
].head()



## === cell 102
scenario_4.loc[
    (scenario_4["fare_amount"] > 3.0) & (scenario_4["H_Distance"] == 0)
].head()



## === cell 103
scenario_4_sub = scenario_4.loc[
    (scenario_4["fare_amount"] > 3.0) & (scenario_4["H_Distance"] == 0)
].copy()



## === cell 104
len(scenario_4_sub)



## === cell 105
scenario_4_sub.loc[:, "H_Distance"] = (scenario_4_sub["fare_amount"] - 2.50) / 1.56



## === cell 106
train.loc[scenario_4_sub.index, "H_Distance"] = scenario_4_sub["H_Distance"]



## === cell 107
train.shape



## === cell 108
train = train.loc[train["H_Distance"].between(0.0, 100.0)].copy()



## === cell 109
train = train.loc[
    ~((train["H_Distance"] < 0.05) & (train["fare_amount"] > 30.0))
].copy()



## === cell 110
train.columns



## === cell 111
test.columns




## === cell 112
def _add_geo_features(df):
    df["manhattan_dist"] = (df["pickup_longitude"] - df["dropoff_longitude"]).abs() + (
        df["pickup_latitude"] - df["dropoff_latitude"]
    ).abs()

    lat1 = np.radians(df["pickup_latitude"].to_numpy())
    lat2 = np.radians(df["dropoff_latitude"].to_numpy())
    dlon = np.radians((df["dropoff_longitude"] - df["pickup_longitude"]).to_numpy())
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    df["bearing"] = np.arctan2(y, x)

    jfk = (-73.8352, -73.7401, 40.6195, 40.6659)
    lga = (-73.8895, -73.8510, 40.7660, 40.7765)
    ewr = (-74.1970, -74.1480, 40.6700, 40.7080)

    def in_box(lon, lat, box):
        lon_min, lon_max, lat_min, lat_max = box
        return lon.between(lon_min, lon_max) & lat.between(lat_min, lat_max)

    pu = df
    pickup_jfk = in_box(pu["pickup_longitude"], pu["pickup_latitude"], jfk)
    drop_jfk = in_box(pu["dropoff_longitude"], pu["dropoff_latitude"], jfk)
    pickup_lga = in_box(pu["pickup_longitude"], pu["pickup_latitude"], lga)
    drop_lga = in_box(pu["dropoff_longitude"], pu["dropoff_latitude"], lga)
    pickup_ewr = in_box(pu["pickup_longitude"], pu["pickup_latitude"], ewr)
    drop_ewr = in_box(pu["dropoff_longitude"], pu["dropoff_latitude"], ewr)

    df["is_airport_trip"] = (
        pickup_jfk | drop_jfk | pickup_lga | drop_lga | pickup_ewr | drop_ewr
    ).astype("int8")
    df["is_jfk_trip"] = (pickup_jfk | drop_jfk).astype("int8")
    df["is_lga_trip"] = (pickup_lga | drop_lga).astype("int8")
    df["is_ewr_trip"] = (pickup_ewr | drop_ewr).astype("int8")


_add_geo_features(train)
_add_geo_features(test)




## === cell 113
def _add_distance_scale_features(df):
    df["H_Distance_miles"] = df["H_Distance"] * 0.621371

    df["dist_x_passengers"] = df["H_Distance_miles"] * df["passenger_count"].astype(
        "float64"
    )
    df["dist_x_hour"] = df["H_Distance_miles"] * df["Hour"].astype("float64")


_add_distance_scale_features(train)
_add_distance_scale_features(test)



## === cell 114
test_key = test["key"].copy()

test_zero_coord = (test["pickup_longitude"].eq(0) & test["pickup_latitude"].eq(0)) | (
    test["dropoff_longitude"].eq(0) & test["dropoff_latitude"].eq(0)
)
test_geo_mask = (
    test["pickup_longitude"].between(nyc_lon_min, nyc_lon_max)
    & test["dropoff_longitude"].between(nyc_lon_min, nyc_lon_max)
    & test["pickup_latitude"].between(nyc_lat_min, nyc_lat_max)
    & test["dropoff_latitude"].between(nyc_lat_min, nyc_lat_max)
)

test_dist_mask = test["H_Distance"].between(0.0, 100.0)

test_valid_mask = (~test_zero_coord) & test_geo_mask & test_dist_mask



## === cell 115
train = train.drop(["key", "pickup_datetime"], axis=1)
test_features = test.drop(["key", "pickup_datetime"], axis=1)



## === cell 116
x_train = train.drop(columns=["fare_amount"])
y_train = train["fare_amount"].astype("float64")
x_test = test_features



## === cell 117
x_test = x_test.reindex(columns=x_train.columns)



## === cell 118
x_train.shape



## === cell 119
x_train.columns



## === cell 120
y_train.shape



## === cell 121
x_test.shape



## === cell 122
x_test.columns



## === cell 123
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=200,
    n_jobs=-1,
    random_state=42,
    min_samples_leaf=3,
    max_features="sqrt",
)
rf.fit(x_train, y_train)

rf_predict = np.full(
    shape=(len(x_test),), fill_value=float(y_train.median()), dtype="float64"
)
valid_idx = np.where(test_valid_mask.to_numpy())[0]
if len(valid_idx) > 0:
    rf_predict[valid_idx] = rf.predict(x_test.iloc[valid_idx])



## === cell 124
rf_predict = np.clip(rf_predict, 0.0, None)



## === cell 125
submission = pd.DataFrame({"key": test_key.values, "fare_amount": rf_predict})
submission.to_csv("submission_1.csv", index=False)
submission.head(20)
