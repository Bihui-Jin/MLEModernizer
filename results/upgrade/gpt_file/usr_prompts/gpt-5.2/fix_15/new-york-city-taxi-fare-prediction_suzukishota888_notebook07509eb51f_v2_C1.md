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

4.20479

# 6. Current score

5.30816

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.05507) has done: 'You’re currently losing a lot of score because the distance feature is computed with `geopy.geodesic` inside a Python loop, which is slow and also less consistent than a standard fast haversine computation; replacing it with a vectorized haversine distance keeps the same core feature idea (“distance between pickup/dropoff”) but yields a better-behaved signal for the RandomForest and typically improves RMSE toward your target. I also stop concatenating train/test just to compute distance (we can compute it separately with identical semantics), and I ensure we never accidentally include NaNs/infs in the model inputs. Finally, I keep your model/training approach intact (RandomForestRegressor with the same split logic) and still write `submission.csv` with the required columns.'
- What this solution (achieved 5.49296) has done: 'Your current gap is large (6.05507 vs target 4.20479; lower is better), so we should make a small but meaningful improvement without changing the overall approach (RandomForest on engineered distance + raw coords/passenger_count). The biggest low-risk gain is to add a couple of standard NYC Taxi baseline features derived from `pickup_datetime` (hour/day-of-week) and to use a log1p transform on the target during training with an expm1 inverse at prediction time; this preserves the same model family and training loop while typically reducing RMSE substantially on this competition. I also keep your existing cleaning rules, keep the haversine distance, and ensure the same submission schema (`key,fare_amount`) with row alignment to `sample_submission.csv`. These changes are directly aimed at improving RMSE toward your target while staying within Kaggle constraints and finishing fast on 10k rows.'
- What this solution (achieved 5.52201) has done: 'You’re still quite a bit above the target RMSE (5.49 vs 4.20; lower is better), so we should make a small, legitimate feature improvement without changing the overall approach (RandomForest on engineered features + log1p target). The most “baseline-safe” gain on this competition is adding a couple of standard geospatial features derived from the same pickup/dropoff coords you already use: straight deltas, manhattan distance, and distances to an NYC center point; these preserve your core feature extraction idea and model family. I also make the `RandomForestRegressor` slightly stronger via `n_estimators`/`min_samples_leaf` (still the same model/loop) to reduce error without changing semantics. Submission writing and key alignment stay exactly as you already do.'
- What this solution (achieved 5.21033) has done: 'You’re currently above the target RMSE (5.52 vs 4.20, lower is better), so we make a small, safe improvement that keeps the same RandomForest + log1p target core logic. The biggest low-risk gain on NYC Taxi is adding a couple of standard time and geo interaction features without changing the model family: year and a simple “radians + bearing” style directional signal. We also make the train/validation split deterministic but slightly more representative by stratifying on coarse fare bins (still the same train_test_split usage, just less variance), which typically nudges RMSE down. Finally, we keep submission alignment by `key` exactly as you already do and ensure the output CSV is valid.'
- What this solution (achieved 5.06973) has done: 'We’re still above the target RMSE (5.21033 vs 4.20479; lower is better), so the smallest likely gain without changing your core approach is to (1) add one strong, standard baseline feature for this competition: haversine distance to/from the nearest NYC airport (JFK/LGA/EWR), and (2) slightly adjust the RandomForest capacity/regularization to reduce bias (more trees, allow slightly deeper trees via `min_samples_leaf=1`). This keeps the same model family (RandomForest), same training procedure (train/valid split + log1p target), and same existing features, just adding a small amount of high-signal geography. It should move RMSE downward toward the target band while staying within runtime and without changing evaluation semantics. Submission writing and key alignment remain unchanged.'
- What this solution (achieved 6.14187) has done: 'You’re currently above the target RMSE (5.06973 vs 4.20479; lower is better), so we make the smallest “same-core-logic” change that usually yields a real RMSE drop in this competition: add a simple, high-signal geospatial feature (trip distance clipped to a reasonable max) and a mild target-cleaning step that removes extreme fares per-km that RandomForest struggles with on small samples. This preserves your exact model family (RandomForestRegressor), log1p training, split/training loop, and overall feature concept, while reducing noise/outlier impact to move RMSE downward toward the target band. We also keep submission alignment by `key` exactly as you do and still write `submission.csv` with the required columns. Runtime stays within limits (still only 10k train rows).'
- What this solution (achieved 6.14187) has done: 'We’re still well above the target RMSE (6.14187 vs 4.20479; lower is better), so the smallest meaningful improvement is to (1) fix the log-target training bug by fitting on log1p of the full training labels (you currently fit on log labels but stratify/split on raw labels, and you validate against raw without consistent handling), and (2) make the train/valid split stratify on the same transformed scale to stabilize error. I also keep your exact RandomForest approach and feature set, but add a single, standard post-processing step for this competition: clip predictions to the cleaned training fare range to reduce outlier RMSE. These are minimal changes that preserve your core logic while typically nudging RMSE downward toward your target, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 5.27916) has done: 'Your current RMSE (6.14187) is still well above the target (4.20479, lower is better), so the smallest likely improvement without changing your core approach (RandomForest + engineered geo/time features + log1p target) is to fix a distribution mismatch you currently have: you aggressively filter/clip outliers only in train, but leave test completely unfiltered, which pushes the model into extrapolation and typically worsens RMSE. I add the exact same coordinate/passenger/distance sanity filtering to `test_df` (dropping only obviously invalid rows) and then re-align predictions back to the original test keys so the submission stays valid and complete. I also clip predictions using the *post-cleaning* train fare range (still 2.5–100 here) but make it derive from data to stay consistent with your cleaning. Everything else (features, model, split, log1p training) stays the same.'
- What this solution (achieved 9.82142) has done: 'Your current RMSE (5.27916) is still well above the target (4.20479; lower is better), so the smallest legitimate push toward the target—without changing your core RandomForest+log1p approach—is to make the training sample (10k rows) less noisy and more representative. I keep all your existing features and model, but add two standard NYC Taxi cleanup filters that reduce label noise a lot: removing obviously bad coordinates (zeros/out-of-range) and filtering extreme speeds using pickup_datetime (eliminates corrupted trips). I also apply the same coordinate sanity checks consistently before feature engineering to avoid NaNs/inf propagation and reduce extrapolation. Everything else (feature set, log1p target, train/valid split, RandomForestRegressor, submission alignment/format) remains intact.'
- What this solution (achieved 10.02927) has done: 'Your current RMSE (9.82142) is far above the target (4.20479, lower is better), so the smallest safe move toward the target is to reduce training noise and distribution mismatch without changing your core RandomForest+log1p setup. The biggest issue in your latest code is that the “speed filter” is effectively not filtering anything (it uses a fixed 1-minute duration for all trips), which leaves many bad/noisy rows that hurt RMSE; I replace it with a true speed estimate using `pickup_datetime` to derive trip duration from `dropoff_datetime` parsed from the `key` (available in both train/test), and filter only extreme speeds. I keep your existing features/model/training loop intact and only adjust the cleaning to be correct and consistent, then ensure the submission remains aligned to `sample_submission.csv` keys and is fully populated.'
- What this solution (achieved 5.29502) has done: 'The crash happens because your cleaning pipeline is over-filtering the 10k training sample down to zero rows (mainly due to the incorrect “dropoff datetime from key” speed filter), so the train/valid split fails and the model never fits. I keep your core RandomForest + log1p target + engineered geo/time features intact, but remove the dropoff-time speed filter and replace it with a simple, standard distance-only sanity filter that can’t zero out the data. I also add a small safeguard so stratified splitting falls back to a normal split if qcut can’t form bins, and I ensure the submission is fully aligned to `sample_submission.csv` keys and always writes `submission.csv`.'
- What this solution (achieved 5.3095) has done: 'Your current RMSE (5.295) is still well above the target (4.204, lower is better), so the smallest likely improvement is to reduce variance and overfitting without changing your model family or feature set. I keep the exact same engineered features and log1p training/inference, but I switch the RandomForest to use bootstrap OOB scoring plus a small amount of regularization (`min_samples_leaf=2`, `max_features="sqrt"`) and modestly increase tree count—this typically improves generalization on this task while staying within the same core approach. I also ensure all datetime-derived features are created from UTC but made timezone-naive to avoid any subtle dtype issues, and I keep your submission alignment by `key` exactly as-is. The script still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 5.3095) has done: 'We’re still above the target RMSE (5.3095 vs 4.20479; lower is better), so the smallest likely move toward the target without changing your core RandomForest+log1p approach is to fix a feature bug: `manhattan_km` currently uses `dropoff_latitude` where it should use `pickup_latitude` for the “longitude leg”, which injects incorrect geometry noise. I also make prediction clipping consistent with your cleaned training distribution by computing `fare_min/fare_max` from the *post-cleaning* training labels (not the pre-filter array), which typically reduces RMSE by avoiding unnecessary clipping bias. Everything else (data size, model family, training loop, log1p transform, submission alignment) stays the same, and it still writes a valid `submission.csv`.'
- What this solution (achieved 5.30816) has done: 'Your current gap to the target is still sizable (5.3095 vs 4.20479; lower is better), so the smallest likely improvement while preserving your RandomForest+log1p core is to reduce label noise and distribution shift by adding one standard cleanup filter: remove trips with near-zero distance but non-trivial fares and remove very long-distance outliers using your already-computed `distance` feature. I also add two very lightweight, high-signal time features (minute-of-hour and a weekend flag) derived from the same `pickup_datetime` you already parse; this doesn’t change the modeling approach but often improves RMSE materially. Finally, I keep your submission alignment logic intact and only adjust the fallback fill value to be on the same clipped/log-trained scale (median of cleaned train fares), which avoids injecting an out-of-distribution constant for filtered test rows.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import matplotlib.pyplot as plt
import seaborn as sns



## === cell 2
train_df = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", nrows=10000
)
test_df = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")



## === cell 3
train_df



## === cell 4
test_df



## === cell 5
train_df.columns



## === cell 6
test_df.columns



## === cell 7
train_df.isnull().sum().sort_values(ascending=False)



## === cell 8
test_df.isnull().sum().sort_values(ascending=False)



## === cell 9
a = train_df[
    train_df["dropoff_longitude"].isnull() | train_df["dropoff_latitude"].isnull()
]
print(a)



## === cell 10
train_df.drop(a.index, axis=0, inplace=True)



## === cell 11
train_df.isnull().sum().sort_values(ascending=False)



## === cell 12
train_df.describe()



## === cell 13
test_df.describe()



## === cell 14
sns.histplot(data=train_df, x="fare_amount", kde=True, bins=100)



## === cell 15
sns.boxplot(data=train_df, y="fare_amount")



## === cell 16
train_df["passenger_count"].describe()



## === cell 17
sns.histplot(data=train_df, x="passenger_count")
plt.ylim(0, 1000)



## === cell 18
train_df = train_df[(train_df["fare_amount"] >= 2.5) & (train_df["fare_amount"] < 100)]



## === cell 19
train_df["fare_amount"].describe()



## === cell 20
sns.histplot(data=train_df, x="fare_amount", kde=True, bins=100)
plt.xlim(0, 80)
plt.ylim(0, 150000)



## === cell 21
sns.boxplot(data=train_df, y="fare_amount")



## === cell 22
train_df = train_df[
    (train_df["passenger_count"] <= 6) & (train_df["passenger_count"] >= 1)
]
sns.histplot(data=train_df, x="passenger_count")
plt.ylim(0, 1000000)



## === cell 23
train_df = train_df[
    (train_df["pickup_longitude"] <= -73.0) & (train_df["pickup_longitude"] >= -74.5)
]
train_df = train_df[
    (train_df["pickup_latitude"] >= 40.5) & (train_df["pickup_latitude"] <= 42)
]
train_df = train_df[
    (train_df["dropoff_longitude"] <= -73.0) & (train_df["dropoff_longitude"] >= -74.5)
]
train_df = train_df[
    (train_df["dropoff_latitude"] >= 40.5) & (train_df["dropoff_latitude"] <= 42)
]



## === cell 24
train_df.describe()



## === cell 25
train_df




## === cell 26
def haversine_km(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    R = 6371.0  # Earth radius in km
    lat1 = np.radians(pickup_lat.astype(float))
    lon1 = np.radians(pickup_lon.astype(float))
    lat2 = np.radians(dropoff_lat.astype(float))
    lon2 = np.radians(dropoff_lon.astype(float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(a))
    return R * c


def coord_sanity_mask(df: pd.DataFrame) -> pd.Series:
    m = pd.Series(True, index=df.index)
    for c in [
        "pickup_longitude",
        "dropoff_longitude",
    ]:
        m &= df[c].between(-74.5, -73.0)
    for c in [
        "pickup_latitude",
        "dropoff_latitude",
    ]:
        m &= df[c].between(40.5, 42.0)

    m &= ~(
        (df["pickup_longitude"].abs() < 1e-6)
        | (df["pickup_latitude"].abs() < 1e-6)
        | (df["dropoff_longitude"].abs() < 1e-6)
        | (df["dropoff_latitude"].abs() < 1e-6)
    )
    return m


train_df = train_df.loc[coord_sanity_mask(train_df)].copy()

train_dt = pd.to_datetime(train_df["pickup_datetime"], errors="coerce", utc=True)
train_df = train_df.loc[~train_dt.isna()].copy()

dist_km_tmp = haversine_km(
    train_df["pickup_latitude"].values,
    train_df["pickup_longitude"].values,
    train_df["dropoff_latitude"].values,
    train_df["dropoff_longitude"].values,
).astype("float64")

train_df = train_df.loc[(dist_km_tmp >= 0.0) & (dist_km_tmp <= 100.0)].copy()



## === cell 27
test_df_original = test_df.copy()

required_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "pickup_datetime",
]
test_valid_mask = ~test_df[required_cols].isnull().any(axis=1)

test_valid_mask &= (test_df["passenger_count"] <= 6) & (test_df["passenger_count"] >= 1)
test_valid_mask &= (test_df["pickup_longitude"] <= -73.0) & (
    test_df["pickup_longitude"] >= -74.5
)
test_valid_mask &= (test_df["pickup_latitude"] >= 40.5) & (
    test_df["pickup_latitude"] <= 42
)
test_valid_mask &= (test_df["dropoff_longitude"] <= -73.0) & (
    test_df["dropoff_longitude"] >= -74.5
)
test_valid_mask &= (test_df["dropoff_latitude"] >= 40.5) & (
    test_df["dropoff_latitude"] <= 42
)

test_valid_mask &= coord_sanity_mask(test_df)

test_df = test_df.loc[test_valid_mask].copy()

for df in (train_df, test_df):
    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"], errors="coerce", utc=True
    ).dt.tz_convert(None)
    df["pickup_hour"] = df["pickup_datetime"].dt.hour.astype("float32")
    df["pickup_minute"] = df["pickup_datetime"].dt.minute.astype("float32")
    df["pickup_dow"] = df["pickup_datetime"].dt.dayofweek.astype("float32")
    df["pickup_is_weekend"] = (df["pickup_dow"] >= 5).astype("float32")
    df["pickup_month"] = df["pickup_datetime"].dt.month.astype("float32")
    df["pickup_year"] = df["pickup_datetime"].dt.year.astype("float32")

test_key_full = test_df_original["key"].copy()
test_key_filtered = test_df["key"].copy()

train_df.drop(["key", "pickup_datetime"], axis=1, inplace=True)
test_df.drop(["key", "pickup_datetime"], axis=1, inplace=True)



## === cell 28
train_df.columns



## === cell 29
test_df.columns



## === cell 30
train_len = len(train_df)
print(train_len)




## === cell 31
def add_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    dlat = (df["dropoff_latitude"] - df["pickup_latitude"]).astype("float32")
    dlon = (df["dropoff_longitude"] - df["pickup_longitude"]).astype("float32")
    df["abs_dlat"] = np.abs(dlat)
    df["abs_dlon"] = np.abs(dlon)

    df["manhattan_km"] = (
        haversine_km(
            df["pickup_latitude"].values,
            df["pickup_longitude"].values,
            df["dropoff_latitude"].values,
            df["pickup_longitude"].values,
        )
        + haversine_km(
            df["pickup_latitude"].values,
            df["pickup_longitude"].values,
            df["pickup_latitude"].values,
            df["dropoff_longitude"].values,
        )
    ).astype("float32")

    nyc_lat, nyc_lon = 40.7141667, -74.0063889
    df["pickup_to_nyc_km"] = haversine_km(
        df["pickup_latitude"].values,
        df["pickup_longitude"].values,
        np.full(len(df), nyc_lat),
        np.full(len(df), nyc_lon),
    ).astype("float32")
    df["dropoff_to_nyc_km"] = haversine_km(
        df["dropoff_latitude"].values,
        df["dropoff_longitude"].values,
        np.full(len(df), nyc_lat),
        np.full(len(df), nyc_lon),
    ).astype("float32")

    lat1 = np.radians(df["pickup_latitude"].astype(float).values)
    lon1 = np.radians(df["pickup_longitude"].astype(float).values)
    lat2 = np.radians(df["dropoff_latitude"].astype(float).values)
    lon2 = np.radians(df["dropoff_longitude"].astype(float).values)
    dlon_r = lon2 - lon1
    y = np.sin(dlon_r) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon_r)
    df["bearing"] = np.arctan2(y, x).astype("float32")

    airports = np.array(
        [
            [40.6413, -73.7781],  # JFK
            [40.7769, -73.8740],  # LGA
            [40.6895, -74.1745],  # EWR
        ],
        dtype="float64",
    )
    p_lat = df["pickup_latitude"].values.astype("float64")
    p_lon = df["pickup_longitude"].values.astype("float64")
    d_lat = df["dropoff_latitude"].values.astype("float64")
    d_lon = df["dropoff_longitude"].values.astype("float64")

    dist_p = np.vstack(
        [
            haversine_km(p_lat, p_lon, np.full(len(df), ap[0]), np.full(len(df), ap[1]))
            for ap in airports
        ]
    )
    dist_d = np.vstack(
        [
            haversine_km(d_lat, d_lon, np.full(len(df), ap[0]), np.full(len(df), ap[1]))
            for ap in airports
        ]
    )
    df["pickup_airport_km"] = dist_p.min(axis=0).astype("float32")
    df["dropoff_airport_km"] = dist_d.min(axis=0).astype("float32")

    return df


train_df["distance"] = haversine_km(
    train_df["pickup_latitude"].values,
    train_df["pickup_longitude"].values,
    train_df["dropoff_latitude"].values,
    train_df["dropoff_longitude"].values,
).astype("float32")
test_df["distance"] = haversine_km(
    test_df["pickup_latitude"].values,
    test_df["pickup_longitude"].values,
    test_df["dropoff_latitude"].values,
    test_df["dropoff_longitude"].values,
).astype("float32")

train_df = add_geo_features(train_df)
test_df = add_geo_features(test_df)

for df in (train_df, test_df):
    df["distance"] = df["distance"].clip(lower=0.0, upper=100.0).astype("float32")
    df["manhattan_km"] = (
        df["manhattan_km"].clip(lower=0.0, upper=150.0).astype("float32")
    )

train_df = train_df.loc[
    ~((train_df["distance"] < 0.05) & (train_df["fare_amount"] > 10.0))
].copy()
train_df = train_df.loc[train_df["distance"] <= 60.0].copy()

eps = 1e-3
fare_per_km = train_df["fare_amount"] / (train_df["distance"] + eps)
train_df = train_df[(fare_per_km <= 50.0) & (fare_per_km >= 0.5)].copy()



## === cell 32
train_df.head()



## === cell 33
test_df.head()



## === cell 34
train_df = train_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
test_df = test_df.replace([np.inf, -np.inf], np.nan).fillna(
    test_df.median(numeric_only=True)
)



## === cell 35
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split



## === cell 36
train_X = train_df.drop("fare_amount", axis=1).values
train_y = train_df["fare_amount"].values.astype("float64")
test_X = test_df.values

train_y_log = np.log1p(train_y)

do_stratify = True
try:
    fare_bins = pd.qcut(train_y_log, q=10, duplicates="drop")
    bin_counts = fare_bins.value_counts()
    if (bin_counts.min() < 2) or (fare_bins.nunique() < 2):
        do_stratify = False
except Exception:
    do_stratify = False
    fare_bins = None

train_x, valid_x, train_y_log_split, valid_y_log_split = train_test_split(
    train_X,
    train_y_log,
    test_size=0.3,
    random_state=0,
    stratify=fare_bins if do_stratify else None,
)



## === cell 37
model = RandomForestRegressor(
    random_state=0,
    n_jobs=-1,
    n_estimators=1200,
    min_samples_leaf=2,
    max_features="sqrt",
    bootstrap=True,
    oob_score=True,
)

model.fit(train_x, train_y_log_split)

valid_pred_log = model.predict(valid_x)
valid_pred = np.expm1(valid_pred_log)
valid_y = np.expm1(valid_y_log_split)

fare_min = float(train_df["fare_amount"].min())
fare_max = float(train_df["fare_amount"].max())
valid_pred = np.clip(valid_pred, fare_min, fare_max)

print("score:" + str(np.sqrt(mean_squared_error(valid_y, valid_pred))))
print("oob_score (R^2 on log1p scale):", getattr(model, "oob_score_", None))



## === cell 38
sub = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
)

test_pred_log = model.predict(test_X)
test_pred = np.expm1(test_pred_log)

test_pred = np.clip(test_pred, fare_min, fare_max)

pred_by_key = pd.DataFrame({"key": test_key_filtered.values, "fare_amount": test_pred})

sub = sub.drop(columns=["fare_amount"]).merge(pred_by_key, on="key", how="left")

fallback = float(np.median(train_df["fare_amount"].values))
sub["fare_amount"] = sub["fare_amount"].fillna(fallback)



## === cell 39
sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with rows:", len(sub))
