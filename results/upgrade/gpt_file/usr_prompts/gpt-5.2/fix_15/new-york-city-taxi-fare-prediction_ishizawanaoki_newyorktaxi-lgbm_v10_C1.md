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

3.10

# 3. Installed packages

geopandas==0.14.4
geopy==2.4.1
lightgbm==4.6.0
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

3.27817

# 6. Current score

5.74027

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.38139) has done: 'I fix the pipeline break in feature engineering by keeping `key` as a string ID (it must not be coerced to float) and only cleaning it in a non-destructive way. I update the LightGBM training call to be compatible with lightgbm==4.6.0 by replacing deprecated `verbose_eval`/`early_stopping_rounds` with the proper callback API, so training completes and produces predictions. I also make the distance feature computation vectorized (same feature, much faster) so it finishes within Kaggle’s time limits. Finally, I ensure the submission is written as a valid `.csv` with exactly `key` and `fare_amount` aligned to the test set.'
- What this solution (achieved 4.24594) has done: 'You’re currently far worse than the target (RMSE 4.38 vs 3.28, lower is better), so we should make the smallest changes that are likely to improve generalization without changing the core LightGBM approach. The biggest issue is that `key_clean` injects a high-cardinality, mostly-random identifier feature that encourages memorization in CV but hurts the true test score; dropping it is a minimal, high-impact fix. Next, dropping `pickup_datetime` loses strong signal; we add a few simple datetime-derived features (hour, dayofweek, month, year) while still using the same model/training loop. Finally, we use a slightly more appropriate LightGBM metric (`rmse`) for early stopping (objective stays regression), which aligns training selection with the competition RMSE without changing semantics.'
- What this solution (achieved 4.35106) has done: 'Your pipeline already trains and writes a valid submission; the main reason you likely didn’t get a Kaggle score is that you never actually evaluated on Kaggle yet, and performance is still far from the 3.278 target. To move RMSE down with minimal logic changes, I (1) align train/test cleaning by applying the same geographic/passenger sanity filters to the test set (without touching `fare_amount` there), (2) add two tiny but high-signal geospatial features (`manhattan_distance` and `bearing`) alongside your existing haversine/abs deltas, and (3) make CV more stable for this dataset by using a time-aware split (sort by pickup time, then KFold without shuffling) to reduce leakage between train/valid. These are small, standard NYC-taxi fixes that typically improve public RMSE without changing the model family or training loop, and they keep runtime within limits.'
- What this solution (achieved 6.27108) has done: 'Your current RMSE (4.351) is worse than the target (3.278, lower is better), so we should make small, safe changes that usually improve generalization without changing the LightGBM approach. The biggest win with minimal disruption is to stop filtering out “weird” test rows (you currently drop them and then fill them with a constant mean), because that fallback creates large errors on those rows; instead, we keep the full test set and only apply train-only target-based cleaning. Next, we add two tiny geospatial features (`log1p(distance)` and `distance_squared`) that often help a linear-in-distance fare relationship while keeping the same model and training loop. Finally, we use the OOF RMSE (not the average of fold best_scores) as the CV readout, but prediction/training semantics stay the same.'
- What this solution (achieved 10.28351) has done: 'Your current score (6.27108 RMSE) is far worse than the target (3.27817), so we should make small, high-signal improvements without changing the LightGBM/KFold training core. The biggest likely issue is that `pickup_time` is a huge raw timestamp feature that can encourage brittle splits/memorization; replacing it with simple cyclical time-of-day and day-of-week encodings typically improves generalization while keeping the same “datetime features” approach. Next, NYC Taxi fares depend strongly on airport trips; adding two boolean features for proximity to JFK/LGA plus a “to/from airport” flag is a minimal feature addition that often reduces RMSE materially. Finally, we keep your submission pipeline unchanged (full test, correct `key` alignment) and only add these features, so it still runs end-to-end and writes `submission_lightgbm.csv`.'
- What this solution (achieved 6.19369) has done: 'Your current RMSE (10.28351) is far worse than the target (3.27817, lower is better), so we make the smallest fixes that typically recover “normal” NYC Taxi performance without changing the LightGBM/KFold core. The biggest likely bug is time leakage/instability from concatenating train+test before parsing time, and from using `.view("int64")` on timezone-aware datetimes; we compute time features separately for train and test with safe conversions and drop the raw epoch seconds as before. Next, we remove the airport-distance features that can dominate and overfit when trained on only 1M rows (keeping your distance/manhattan/bearing/log/squared core intact), which should move RMSE down substantially toward the target. Finally, we keep the exact submission schema/merge logic unchanged so it still writes a valid `submission_lightgbm.csv`.'
- What this solution (achieved 5.97881) has done: 'You’re substantially worse than the target (6.19 vs 3.28 RMSE, lower is better), so we should make small, standard NYC-taxi improvements without changing the core LightGBM+KFold approach. The biggest likely performance drag is a distribution shift from training on heavily filtered “NYC-only” coordinates while leaving the test unfiltered; we apply the same coordinate/passenger sanity filters to test (but never drop rows—just clip to bounds) so feature ranges match. Next, we add two tiny, high-signal geo features that don’t alter the model family: (1) distance-to-NYC-center for pickup/dropoff and (2) a simple “night” indicator; these typically reduce RMSE with minimal risk. Finally, we make `passenger_count` an actual categorical feature for LightGBM (same training loop) which often improves splits on this dataset.'
- What this solution (achieved 10.93126) has done: 'Your current RMSE (5.98) is much worse than the target (3.28, lower is better), so we make small, high-signal fixes without changing the LightGBM+KFold core. The biggest issue is that the model is missing classic NYC-taxi fare signals: trip distance in “NYC miles” along streets and airport trips; we add minimal, standard airport-distance and “to/from airport” features while keeping your existing distance/bearing features intact. We also reintroduce a numeric `pickup_time` feature safely (as days since epoch) since you already sort by time but then drop it, and add a simple “cross-borough-ish” feature (`abs_delta_lon * abs_delta_lat`) that often helps. Finally, we keep the same training loop and submission alignment, and set a fixed seed for stability (not for max score), so results are more consistent run-to-run.'
- What this solution (achieved 5.51669) has done: 'Your current notebook likely didn’t yield a Kaggle score because it never actually produced/validated a submission in the exact `sample_submission.csv` row order (Kaggle is strict about alignment), and it may also be silently generating NaN/inf geo features (e.g., from invalid coords) that LightGBM handles poorly. I make minimal, score-relevant fixes: (1) enforce numeric sanity (finite values) in both train and test after feature engineering (without dropping test rows), (2) clip and cast key features consistently to avoid LightGBM surprises, and (3) write the submission by starting from `sample_submission.csv` and filling predictions by `key` to guarantee perfect row order and full coverage. Core logic (same LightGBM, same features, same KFold approach) remains unchanged; these are stability/correctness adjustments that typically reduce RMSE versus a broken/misaligned submission and prevent training/prediction degradation from NaNs/infs.'
- What this solution (achieved 5.04437) has done: 'You don’t currently yield a Kaggle score because your script is in a `cell 0..29` format, while the required runner expects cells starting at `## === cell 1` with no gaps; I renumber cells to ensure it runs end-to-end and actually writes `submission_lightgbm.csv`. To move RMSE down toward the 3.278 target without changing the core LightGBM/KFold approach, I make one minimal, high-signal correction: compute your “manhattan_distance” in miles (not raw degrees) using the standard latitude-dependent longitude scaling; this keeps the same feature intent but fixes a unit bug that typically hurts generalization. I also add a tiny safety fix in `add_time_features` to avoid `NaT -> int64` issues by filling `dt` before casting, preventing silent overflows/zeros that degrade the time signal. Everything else (data size, training loop, model family/params, submission schema and alignment via `sample_submission`) stays the same.'
- What this solution (achieved 5.74027) has done: 'Your current RMSE (5.044) is far from the target (3.278, lower is better), so we make two minimal, high-signal fixes that keep the same LightGBM+KFold training loop and the same overall feature set intent. First, we replace the current `KFold(shuffle=False)` (which still mixes future/past in validation) with a simple time-based fold splitter using your already-sorted `pickup_time_days`, reducing leakage and typically improving public RMSE. Second, we remove the most error-prone rows from training by applying standard NYC bounding-box + passenger sanity filtering (train-only) rather than the very loose global lat/lon bounds; this reduces noise/outliers without touching the test set or changing the model. Everything else (feature engineering functions, objective/metric, early stopping, submission alignment via `sample_submission`) stays the same.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

INPUT_DIR = "/kaggle/input/new-york-city-taxi-fare-prediction"

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:50]:
        print(os.path.join(dirname, filename))



## === cell 1
train = pd.read_csv(f"{INPUT_DIR}/train.csv", nrows=1_000_000)
test = pd.read_csv(f"{INPUT_DIR}/test.csv")
sample_submission = pd.read_csv(f"{INPUT_DIR}/sample_submission.csv")

print(train.shape, test.shape, sample_submission.shape)
train.head()



## === cell 2
train.isnull().sum()



## === cell 3
train.dropna(inplace=True)



## === cell 4
train.describe()



## === cell 5
train.query("passenger_count > 6").head()



## === cell 6
train.query("passenger_count < 1").head()



## === cell 7
train.query("fare_amount < 0").head()



## === cell 8
train.query("pickup_longitude < -180 or pickup_longitude > 180").head()



## === cell 9
train.query("dropoff_longitude < -180 or dropoff_longitude > 180").head()



## === cell 10
train.query("pickup_latitude < -90 or pickup_latitude > 90").head()



## === cell 11
train.query("dropoff_latitude < -90 or dropoff_latitude > 90").head()



## === cell 12
train = train.query(
    "1 <= passenger_count <= 6 and "
    "0 <= fare_amount <= 500 and "
    "-74.3 <= pickup_longitude <= -72.9 and "
    "-74.3 <= dropoff_longitude <= -72.9 and "
    "40.5 <= pickup_latitude <= 41.0 and "
    "40.5 <= dropoff_latitude <= 41.0"
).copy()

train = train.loc[
    ~(
        (train["pickup_longitude"] == train["dropoff_longitude"])
        & (train["pickup_latitude"] == train["dropoff_latitude"])
    )
].copy()

train.describe()



## === cell 13
train.reset_index(drop=True, inplace=True)
train.head()



## === cell 14
test_full = pd.read_csv(f"{INPUT_DIR}/test.csv")
test = test_full.copy()
test["key"] = test["key"].astype(str)
test_full["key"] = test_full["key"].astype(str)

print("Using full test shape (no filtering):", test.shape)



## === cell 15
test["passenger_count"] = test["passenger_count"].fillna(1)
test["passenger_count"] = np.clip(
    test["passenger_count"].astype(np.int16), 1, 6
).astype(np.int16)
print("Test passenger_count describe (clipped 1..6):")
print(test[["passenger_count"]].describe())




## === cell 16
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["key"] = df["key"].astype(str)

    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)
    dt_filled = dt.fillna(pd.Timestamp("1970-01-01", tz="UTC"))

    df["pickup_year"] = dt.dt.year.fillna(0).astype(np.int16)
    df["pickup_month"] = dt.dt.month.fillna(0).astype(np.int8)
    df["pickup_dayofweek"] = dt.dt.dayofweek.fillna(0).astype(np.int8)
    df["pickup_hour"] = dt.dt.hour.fillna(0).astype(np.int8)

    hour = df["pickup_hour"].to_numpy(dtype=np.float32)
    dow = df["pickup_dayofweek"].to_numpy(dtype=np.float32)
    df["pickup_hour_sin"] = np.sin(2 * np.pi * (hour / 24.0))
    df["pickup_hour_cos"] = np.cos(2 * np.pi * (hour / 24.0))
    df["pickup_dow_sin"] = np.sin(2 * np.pi * (dow / 7.0))
    df["pickup_dow_cos"] = np.cos(2 * np.pi * (dow / 7.0))

    dt_ns = dt_filled.astype("int64")
    df["pickup_time_days"] = (dt_ns // (10**9 * 86400)).astype(np.int32)

    df["is_night"] = ((df["pickup_hour"] <= 6) | (df["pickup_hour"] >= 20)).astype(
        np.int8
    )

    df = df.drop(columns=["pickup_datetime"])
    return df


train_tf = add_time_features(train)
test_tf = add_time_features(test)

train_tf.head()




## === cell 17
def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.7613  # Earth radius in miles
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def bearing_rad(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.arctan2(y, x)


def add_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["distance"] = haversine_miles(
        df["pickup_latitude"].to_numpy(),
        df["pickup_longitude"].to_numpy(),
        df["dropoff_latitude"].to_numpy(),
        df["dropoff_longitude"].to_numpy(),
    )

    df["abs_delta_lat"] = (df["dropoff_latitude"] - df["pickup_latitude"]).abs()
    df["abs_delta_lon"] = (df["dropoff_longitude"] - df["pickup_longitude"]).abs()

    mean_lat_rad = np.radians(
        ((df["pickup_latitude"] + df["dropoff_latitude"]) / 2.0).to_numpy(
            dtype=np.float64
        )
    )
    lat_miles = 69.0 * df["abs_delta_lat"].to_numpy(dtype=np.float64)
    lon_miles = (
        69.0 * np.cos(mean_lat_rad) * df["abs_delta_lon"].to_numpy(dtype=np.float64)
    )
    df["manhattan_distance"] = (lat_miles + lon_miles).astype(np.float32)

    df["bearing"] = bearing_rad(
        df["pickup_latitude"].to_numpy(),
        df["pickup_longitude"].to_numpy(),
        df["dropoff_latitude"].to_numpy(),
        df["dropoff_longitude"].to_numpy(),
    )

    dist = df["distance"].to_numpy()
    df["log1p_distance"] = np.log1p(dist)
    df["distance_squared"] = np.square(dist)

    df["delta_lat_lon_product"] = (df["abs_delta_lat"] * df["abs_delta_lon"]).astype(
        np.float32
    )

    nyc_lat, nyc_lon = 40.7580, -73.9855  # Times Square-ish
    df["pickup_to_center"] = haversine_miles(
        df["pickup_latitude"].to_numpy(),
        df["pickup_longitude"].to_numpy(),
        np.full(len(df), nyc_lat, dtype=np.float64),
        np.full(len(df), nyc_lon, dtype=np.float64),
    )
    df["dropoff_to_center"] = haversine_miles(
        df["dropoff_latitude"].to_numpy(),
        df["dropoff_longitude"].to_numpy(),
        np.full(len(df), nyc_lat, dtype=np.float64),
        np.full(len(df), nyc_lon, dtype=np.float64),
    )

    JFK = (40.6413, -73.7781)
    LGA = (40.7769, -73.8740)
    EWR = (40.6895, -74.1745)

    df["pickup_to_jfk"] = haversine_miles(
        df["pickup_latitude"].to_numpy(),
        df["pickup_longitude"].to_numpy(),
        np.full(len(df), JFK[0], dtype=np.float64),
        np.full(len(df), JFK[1], dtype=np.float64),
    )
    df["dropoff_to_jfk"] = haversine_miles(
        df["dropoff_latitude"].to_numpy(),
        df["dropoff_longitude"].to_numpy(),
        np.full(len(df), JFK[0], dtype=np.float64),
        np.full(len(df), JFK[1], dtype=np.float64),
    )

    df["pickup_to_lga"] = haversine_miles(
        df["pickup_latitude"].to_numpy(),
        df["pickup_longitude"].to_numpy(),
        np.full(len(df), LGA[0], dtype=np.float64),
        np.full(len(df), LGA[1], dtype=np.float64),
    )
    df["dropoff_to_lga"] = haversine_miles(
        df["dropoff_latitude"].to_numpy(),
        df["dropoff_longitude"].to_numpy(),
        np.full(len(df), LGA[0], dtype=np.float64),
        np.full(len(df), LGA[1], dtype=np.float64),
    )

    df["pickup_to_ewr"] = haversine_miles(
        df["pickup_latitude"].to_numpy(),
        df["pickup_longitude"].to_numpy(),
        np.full(len(df), EWR[0], dtype=np.float64),
        np.full(len(df), EWR[1], dtype=np.float64),
    )
    df["dropoff_to_ewr"] = haversine_miles(
        df["dropoff_latitude"].to_numpy(),
        df["dropoff_longitude"].to_numpy(),
        np.full(len(df), EWR[0], dtype=np.float64),
        np.full(len(df), EWR[1], dtype=np.float64),
    )

    df["is_airport_trip"] = (
        (df["pickup_to_jfk"] < 1.0)
        | (df["dropoff_to_jfk"] < 1.0)
        | (df["pickup_to_lga"] < 1.0)
        | (df["dropoff_to_lga"] < 1.0)
        | (df["pickup_to_ewr"] < 1.0)
        | (df["dropoff_to_ewr"] < 1.0)
    ).astype(np.int8)

    return df


train_fe = add_geo_features(train_tf)
test_fe = add_geo_features(test_tf)

train_fe.head()



## === cell 18
train_fe = train_fe.loc[
    ~((train_fe["distance"] < 0.01) & (train_fe["fare_amount"] > 3.0))
].copy()
train_fe.reset_index(drop=True, inplace=True)




## === cell 19
def sanitize_numeric_features(df: pd.DataFrame, exclude_cols=None) -> pd.DataFrame:
    if exclude_cols is None:
        exclude_cols = []
    df = df.copy()
    num_cols = [
        c
        for c in df.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
    ]
    if num_cols:
        arr = df[num_cols].to_numpy()
        arr = np.where(np.isfinite(arr), arr, np.nan)
        df[num_cols] = arr
        df[num_cols] = df[num_cols].fillna(0)
    return df


train_fe = sanitize_numeric_features(train_fe, exclude_cols=["key"])
test_fe = sanitize_numeric_features(test_fe, exclude_cols=["key"])




## === cell 20
def clip_features_by_train_quantiles(
    train_df: pd.DataFrame, test_df: pd.DataFrame, cols, q_low=0.001, q_high=0.999
):
    train_df = train_df.copy()
    test_df = test_df.copy()
    for c in cols:
        if c in train_df.columns and c in test_df.columns:
            lo = float(train_df[c].quantile(q_low))
            hi = float(train_df[c].quantile(q_high))
            if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                train_df[c] = train_df[c].clip(lo, hi)
                test_df[c] = test_df[c].clip(lo, hi)
    return train_df, test_df


geo_clip_cols = [
    "distance",
    "manhattan_distance",
    "log1p_distance",
    "distance_squared",
    "pickup_to_center",
    "dropoff_to_center",
    "pickup_to_jfk",
    "dropoff_to_jfk",
    "pickup_to_lga",
    "dropoff_to_lga",
    "pickup_to_ewr",
    "dropoff_to_ewr",
]

train_fe, test_fe = clip_features_by_train_quantiles(
    train_fe, test_fe, geo_clip_cols, q_low=0.001, q_high=0.999
)



## === cell 21
y_train = train_fe["fare_amount"].astype(float)
X_train = train_fe.drop("fare_amount", axis=1)
X_test = test_fe.drop("fare_amount", axis=1, errors="ignore")

test_keys = test_fe["key"].astype(str).values

X_train = X_train.drop(columns=["key"])
X_test = X_test.drop(columns=["key"])

for df in (X_train, X_test):
    if "passenger_count" in df.columns:
        df["passenger_count"] = df["passenger_count"].astype(np.int8)
    if "is_airport_trip" in df.columns:
        df["is_airport_trip"] = df["is_airport_trip"].astype(np.int8)

X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

order = np.argsort(X_train["pickup_time_days"].to_numpy(), kind="mergesort")
X_train = X_train.iloc[order].reset_index(drop=True)
y_train = y_train.iloc[order].reset_index(drop=True)

X_train.head()



## === cell 22
from sklearn.model_selection import KFold

y_preds = []
models = []
oof_train = np.zeros((len(X_train),), dtype=float)


def time_folds(n_samples: int, n_splits: int = 5):
    idx = np.arange(n_samples)
    folds = np.array_split(idx, n_splits)
    for i in range(n_splits):
        valid_idx = folds[i]
        train_idx = np.hstack([folds[j] for j in range(n_splits) if j != i])
        yield train_idx, valid_idx


cv_splits = list(time_folds(len(X_train), n_splits=5))

categorical_features = ["passenger_count", "is_airport_trip"]



## === cell 23
import lightgbm as lgb

params = {
    "objective": "regression",
    "metric": "rmse",
    "max_bin": 300,
    "learning_rate": 0.05,
    "num_leaves": 40,
    "verbosity": -1,
    "seed": 42,
    "feature_fraction_seed": 42,
    "bagging_seed": 42,
}

for fold_id, (train_index, valid_index) in enumerate(cv_splits):
    X_tr = X_train.iloc[train_index, :]
    X_val = X_train.iloc[valid_index, :]
    y_tr = y_train.iloc[train_index]
    y_val = y_train.iloc[valid_index]

    lgb_train = lgb.Dataset(
        X_tr, y_tr, categorical_feature=categorical_features, free_raw_data=False
    )
    lgb_eval = lgb.Dataset(
        X_val,
        y_val,
        reference=lgb_train,
        categorical_feature=categorical_features,
        free_raw_data=False,
    )

    callbacks = [
        lgb.early_stopping(stopping_rounds=10, verbose=False),
        lgb.log_evaluation(period=50),
    ]

    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_eval],
        valid_names=["train", "valid"],
        num_boost_round=1000,
        callbacks=callbacks,
    )

    oof_train[valid_index] = model.predict(X_val, num_iteration=model.best_iteration)
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)

    y_preds.append(y_pred)
    models.append(model)

len(models), len(y_preds)



## === cell 24
pd.DataFrame({"oof_pred": oof_train}).to_csv("oof_train_kfold.csv", index=False)

scores = [m.best_score["valid"]["rmse"] for m in models]
score = float(sum(scores) / len(scores))
print("===CV fold best scores (rmse)===")
print(scores)
print("Mean of fold best rmse:", score)



## === cell 25
from sklearn.metrics import mean_squared_error

rmse = float(np.sqrt(mean_squared_error(y_train, oof_train)))
print("OOF RMSE:", rmse)



## === cell 26
len(y_preds)



## === cell 27
y_preds[0][:10]



## === cell 28
y_sub = np.mean(np.vstack(y_preds), axis=0)
y_sub = np.clip(y_sub, 0.0, None)
y_sub[:10]



## === cell 29
pred_map = pd.DataFrame({"key": test_keys, "fare_amount": y_sub})
pred_map["key"] = pred_map["key"].astype(str)

sub_full = sample_submission[["key"]].copy()
sub_full["key"] = sub_full["key"].astype(str)
sub_full = sub_full.merge(pred_map, on="key", how="left")

fallback = float(train["fare_amount"].mean())
sub_full["fare_amount"] = sub_full["fare_amount"].fillna(fallback).astype(float)
sub_full["fare_amount"] = np.clip(sub_full["fare_amount"].to_numpy(), 0.0, None)

sub_full.to_csv("submission_lightgbm.csv", index=False)

print(sub_full.head())
print("Saved submission_lightgbm.csv with shape:", sub_full.shape)
print("Columns:", sub_full.columns.tolist())
print("Any NaNs in fare_amount:", sub_full["fare_amount"].isna().any())
print("Matches sample_submission row count:", len(sub_full) == len(sample_submission))
print("Submission file:", os.path.abspath("submission_lightgbm.csv"))
