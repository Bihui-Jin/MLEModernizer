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
scipy==1.15.3
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

4.20958

# 6. Current score

5.97509

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.4395) has done: 'The timeout is dominated by repeatedly parsing huge CSV chunks with `parse_dates` and by fitting `GradientBoostingRegressor` 56 times on ~1e6-row chunks (warm_start doesn’t make `fit()` incremental for this model, so it effectively refits each loop). To keep identical core logic and semantics, the biggest safe wins are: (1) drastically reduce per-chunk I/O and parsing overhead via explicit `usecols`, fixed `dtype`s, and parsing datetimes as strings (the datetime column isn’t used), and (2) eliminate repeated, redundant feature computation inside `data_clean()` (it calls `distance_travel()` again). Additionally, compute `distance_travel` with NumPy arrays to avoid slow pandas Series arithmetic and suppress `SettingWithCopy` overhead, while keeping the exact same formulas and outputs.'
- What this solution (achieved 5.6559) has done: 'Your current score (6.4395 RMSE) is far from the target (4.20958), so we should improve performance while keeping the same overall approach: GradientBoostingRegressor on the same three features (distance_travel, passenger_count, bias). The biggest issue is that the loop is unintentionally refitting the model 56 times on different chunks without actually “incrementally learning”, which tends to hurt generalization; the smallest core-logic-preserving fix is to train once on a single, reasonably large cleaned sample instead of repeatedly overwriting the model. To move RMSE down toward the target, we also keep the exact same distance feature formula but add a minimal, standard lat/long sanity filter (NYC bounding box) that removes pathological coordinates that inflate error. Finally, we keep the same submission schema and write `submission.csv`.'
- What this solution (achieved 5.86495) has done: 'To move RMSE down toward the 4.20958 target (lower is better) without changing your core approach (GBR on distance + passenger_count + bias), I make two minimal, high-impact fixes: (1) correct a numerical instability in `distance_travel` where division by zero can produce inf/NaN (hurting both training and test predictions), and (2) apply the exact same NYC bounding-box filter to the test set as you already do for training, then merge predictions back to the full test keys so the submission format stays valid. These changes preserve your features/model/training loop while improving data consistency and preventing pathological feature values that inflate RMSE. The submission still contain all test rows in the original order with columns `key,fare_amount`.'
- What this solution (achieved 6.21) has done: 'Your current RMSE (5.86495) is still far above the target (4.20958), so we should improve generalization without changing the core approach (GBR on distance + passenger_count + bias). The highest-impact minimal fix is to stop using `warm_start=True` (it doesn’t help with a single `fit()` and can slightly worsen stability) and slightly increase the training signal by using a larger cleaned training sample while keeping the same single-shot training. I also ensure the training/test filtering logic is consistent by applying the same “positive distance” and passenger-count sanity checks to test before feature building, but still predicting for all rows (so submission stays complete). These changes are small, keep the same features and model family, and are expected to move RMSE downward toward the target band.'
- What this solution (achieved 5.93854) has done: 'Your current RMSE (6.21; lower is better) is still far above the target (4.20958), so we should improve generalization with the smallest changes that keep the same core model (GradientBoostingRegressor) and the same three-feature design (distance_travel, passenger_count, bias). The biggest low-risk gain is to make the training sample higher quality by applying the same passenger_count and “distance>0” checks, plus the NYC bounding box, consistently and also filtering out the known-invalid (0,0) GPS rows, which otherwise create misleading near-zero distances and inflate error. I also switch the training target from raw fare to log1p(fare) and invert with expm1 at prediction time; this keeps the same model/training loop but aligns the loss more robustly for heavy-tailed fares and typically reduces RMSE substantially on this competition. Finally, I clamp any negative predictions to 0 (a valid domain constraint) to avoid penalized impossible fares.'
- What this solution (achieved 5.93893) has done: 'Your current RMSE (5.93854, lower is better) is still far above the target (4.20958), so we should make a small change that typically improves this competition without changing your model family or training loop. The biggest low-risk issue is that your training filters keep many “bad but not excluded” rows (especially unrealistically low fares for non-trivial trips), which strongly degrades RMSE for simple distance-based models. I add a minimal, standard “fare vs distance” sanity filter after your existing cleaning/outlier steps (and apply the same validity logic consistently on test only for feature construction, while still predicting for every test row). This preserves your core features (distance_travel, passenger_count, bias), the same GradientBoostingRegressor training approach, and the same log1p/expm1 target transform, but should move RMSE downward toward the target band.'
- What this solution (achieved 6.19244) has done: 'Your current RMSE (5.93893; lower is better) is still far above the target (4.20958), so we should make a small, legitimate improvement that keeps the same model family and the same three-feature design. The highest-impact minimal change here is to use the exact same cleaning/sanity filters for training **and** apply the corresponding “validity” logic consistently during training (including removing extreme high fares and extremely short/long trips already) while also adding one standard, low-risk filter that removes “fare too high for very short distance” label noise (a common issue in this dataset that hurts simple distance-based models). This does not change your features, model, training loop, or loss; it only improves training label quality. Finally, we keep the same prediction flow and submission format, still predicting for every test row.'
- What this solution (achieved 6.01225) has done: 'The timeout is dominated by reading/holding ~4M cleaned rows from a 55M-row CSV and then concatenating them, plus repeated pandas filtering/copying per chunk. I keep the exact same features, filters, target transform, and model, but make the pipeline streaming and memory-stable by preallocating numpy arrays and filling them chunk-by-chunk (no giant `parts` list / `concat`). I also reduce pandas overhead by applying filters via a single combined boolean mask per chunk and by building the training matrix directly from numpy views. Test-time feature building is kept identical, but avoids unnecessary copies.'
- What this solution (achieved 6.01225) has done: 'Your current RMSE (6.01225; lower is better) is still far above the target (4.20958), so we should make a small, metric-aligned improvement without changing your model family, features, or training loop. The biggest safe gain here is to make the time features actually usable: right now `parse_dates` yields timezone-aware datetimes in many Kaggle environments, and `pd.to_datetime(..., utc=False)` on an already-aware series can coerce to `NaT`, silently destroying `pickup_hour/pickup_dow`. I adjust `add_time_features()` to robustly handle both tz-aware and string datetimes and keep the same two time features, which should improve fit quality and reduce RMSE. I also remove `parse_dates` from CSV reading (we still compute the same features) to avoid mixed datetime types and keep train/test consistent.'
- What this solution (achieved 5.77764) has done: 'To move your RMSE down toward the 4.20958 target (lower is better) without changing the core model or feature set, I’m making two minimal, metric-aligned fixes. First, I remove obviously invalid timestamp rows (where `pickup_datetime` can’t be parsed) from training only, because those currently become NaN time-features and add noise to a distance-only model. Second, I add one standard, low-risk geographic sanity filter for the most common label noise: trips involving airports (JFK/LGA/EWR) often have fixed surcharges/tolls that your simple feature set can’t model well, so removing those rows from training typically lowers RMSE while keeping the same semantics and prediction pipeline. Submission generation stays identical and still predicts for every test row.'
- What this solution (achieved 5.97509) has done: 'Your current RMSE (5.77764) is worse than the target (4.20958), so we should make a small, legitimate improvement that keeps your exact model/feature set and training semantics intact. The biggest low-risk gain here is to stop throwing away airport-region trips entirely (which creates a train–test distribution mismatch) and instead keep them while reducing their impact via deterministic sample-weighting during `fit()`. This preserves the same GradientBoostingRegressor, the same features (distance, passenger_count, hour, dow, bias), and the same log1p/expm1 target transform, but typically improves generalization on this competition. I also apply the same airport weighting logic consistently to the test set (no rows removed; only affects how the model learned).'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer

INPUT_DIR = "/kaggle/input/new-york-city-taxi-fare-prediction"
print(os.listdir(INPUT_DIR))

TRAIN_USECOLS = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
TEST_USECOLS = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

TRAIN_DTYPES = {
    "fare_amount": "float64",
    "pickup_datetime": "object",  # keep as string; we'll parse robustly later
    "pickup_longitude": "float64",
    "pickup_latitude": "float64",
    "dropoff_longitude": "float64",
    "dropoff_latitude": "float64",
    "passenger_count": "int16",
}
TEST_DTYPES = {
    "key": "object",
    "pickup_datetime": "object",  # keep as string; we'll parse robustly later
    "pickup_longitude": "float64",
    "pickup_latitude": "float64",
    "dropoff_longitude": "float64",
    "dropoff_latitude": "float64",
    "passenger_count": "int16",
}


def chunck_generator(filename, chunk_size=10**6):
    for chunk in pd.read_csv(
        filename,
        delimiter=",",
        iterator=True,
        chunksize=chunk_size,
        usecols=TRAIN_USECOLS,
        dtype=TRAIN_DTYPES,
        engine="c",
        low_memory=False,
    ):
        yield chunk




## === cell 1
alpha_ang = 0.506


def distance_travel(df):
    plon = df["pickup_longitude"].to_numpy(dtype=np.float64, copy=False)
    plat = df["pickup_latitude"].to_numpy(dtype=np.float64, copy=False)
    dlon = df["dropoff_longitude"].to_numpy(dtype=np.float64, copy=False)
    dlat = df["dropoff_latitude"].to_numpy(dtype=np.float64, copy=False)

    abs_diff_longitude = np.abs(dlon - plon) * 50.0
    abs_diff_latitude = np.abs(dlat - plat) * 69.0

    displacement_vector = np.sqrt(
        abs_diff_latitude * abs_diff_latitude + abs_diff_longitude * abs_diff_longitude
    )

    eps = 1e-12
    angle = np.arctan(abs_diff_longitude / (abs_diff_latitude + eps)) - alpha_ang
    actual_long = np.abs(displacement_vector * np.sin(angle))
    actual_lat = np.abs(displacement_vector * np.cos(angle))

    df["abs_diff_longitude"] = abs_diff_longitude
    df["abs_diff_latitude"] = abs_diff_latitude
    df["displacement_vector"] = displacement_vector
    df["actual_long"] = actual_long
    df["actual_lat"] = actual_lat
    df["distance_travel"] = actual_long + actual_lat
    return df




## === cell 2
def add_time_features(df):
    s = df["pickup_datetime"]
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    dt = dt.dt.tz_convert(None)  # tz-naive for stable .dt access
    df["pickup_hour"] = dt.dt.hour.astype("float64")
    df["pickup_dow"] = dt.dt.dayofweek.astype("float64")
    return df


def data_clean(df):
    df = df[df.passenger_count > 0]
    df.fare_amount = df.fare_amount.astype(np.float64)
    df = df[df.fare_amount > 0]
    df = df[df.distance_travel > 0]
    return df


def remove_outliers(df):
    df = df[df.distance_travel < 30]
    df = df[df.fare_amount < 100]
    return df


def filter_nyc_bbox(df):
    return df[
        (df["pickup_longitude"].between(-74.3, -73.6))
        & (df["dropoff_longitude"].between(-74.3, -73.6))
        & (df["pickup_latitude"].between(40.5, 41.0))
        & (df["dropoff_latitude"].between(40.5, 41.0))
    ]


def filter_zero_gps(df):
    return df[
        ~(
            (df["pickup_longitude"] == 0.0)
            & (df["pickup_latitude"] == 0.0)
            & (df["dropoff_longitude"] == 0.0)
            & (df["dropoff_latitude"] == 0.0)
        )
    ]


def filter_fare_distance_sanity(df):
    return df[~((df["distance_travel"] > 0.2) & (df["fare_amount"] < 2.5))]


def filter_high_fare_short_trip(df):
    return df[~((df["distance_travel"] < 0.05) & (df["fare_amount"] > 30.0))]


def airport_mask(df):
    plon = df["pickup_longitude"].to_numpy(dtype=np.float64, copy=False)
    plat = df["pickup_latitude"].to_numpy(dtype=np.float64, copy=False)
    dlon = df["dropoff_longitude"].to_numpy(dtype=np.float64, copy=False)
    dlat = df["dropoff_latitude"].to_numpy(dtype=np.float64, copy=False)

    def in_box(lon, lat, lon_min, lon_max, lat_min, lat_max):
        return (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)

    jfk_p = in_box(plon, plat, -73.90, -73.75, 40.60, 40.68)
    jfk_d = in_box(dlon, dlat, -73.90, -73.75, 40.60, 40.68)
    lga_p = in_box(plon, plat, -73.90, -73.83, 40.75, 40.79)
    lga_d = in_box(dlon, dlat, -73.90, -73.83, 40.75, 40.79)
    ewr_p = in_box(plon, plat, -74.22, -74.14, 40.67, 40.71)
    ewr_d = in_box(dlon, dlat, -74.22, -74.14, 40.67, 40.71)

    return jfk_p | jfk_d | lga_p | lga_d | ewr_p | ewr_d




## === cell 3
def incremental_training(train_X, train_y, regr, sample_weight=None):
    regr.fit(train_X, train_y, sample_weight=sample_weight)
    return regr




## === cell 4
filename = os.path.join(INPUT_DIR, "train.csv")
gen = chunck_generator(filename=filename, chunk_size=10**6)

TARGET_ROWS = 4_000_000

X_buf = np.empty((TARGET_ROWS, 5), dtype=np.float64)
y_buf = np.empty((TARGET_ROWS,), dtype=np.float64)
w_buf = np.empty((TARGET_ROWS,), dtype=np.float64)

rows = 0
chunk_idx = 0

AIRPORT_WEIGHT = 0.35

while rows < TARGET_ROWS:
    df = next(gen)
    chunk_idx += 1

    df = distance_travel(df)
    df = add_time_features(df)  # keep feature building consistent between train/test

    pc = df["passenger_count"].to_numpy(copy=False)
    fare = df["fare_amount"].to_numpy(dtype=np.float64, copy=False)
    dist = df["distance_travel"].to_numpy(dtype=np.float64, copy=False)

    plon = df["pickup_longitude"].to_numpy(dtype=np.float64, copy=False)
    plat = df["pickup_latitude"].to_numpy(dtype=np.float64, copy=False)
    dlon = df["dropoff_longitude"].to_numpy(dtype=np.float64, copy=False)
    dlat = df["dropoff_latitude"].to_numpy(dtype=np.float64, copy=False)

    pickup_hour = df["pickup_hour"].to_numpy(dtype=np.float64, copy=False)
    pickup_dow = df["pickup_dow"].to_numpy(dtype=np.float64, copy=False)

    dt_ok = np.isfinite(pickup_hour) & np.isfinite(pickup_dow)

    mask = (
        dt_ok
        & (pc > 0)
        & (fare > 0)
        & (dist > 0)
        & (dist < 30)
        & (fare < 100)
        & (plon >= -74.3)
        & (plon <= -73.6)
        & (dlon >= -74.3)
        & (dlon <= -73.6)
        & (plat >= 40.5)
        & (plat <= 41.0)
        & (dlat >= 40.5)
        & (dlat <= 41.0)
        & ~((plon == 0.0) & (plat == 0.0) & (dlon == 0.0) & (dlat == 0.0))
        & ~((dist > 0.2) & (fare < 2.5))
        & ~((dist < 0.05) & (fare > 30.0))
    )

    if not np.any(mask):
        continue

    df_m = df.loc[mask]
    if len(df_m) == 0:
        continue

    a_mask = airport_mask(df_m)

    dist_m = df_m["distance_travel"].to_numpy(dtype=np.float64, copy=False)
    pc_m = df_m["passenger_count"].to_numpy(dtype=np.float64, copy=False)
    hr_m = df_m["pickup_hour"].to_numpy(dtype=np.float64, copy=False)
    dow_m = df_m["pickup_dow"].to_numpy(dtype=np.float64, copy=False)
    fare_m = df_m["fare_amount"].to_numpy(dtype=np.float64, copy=False)

    n = len(dist_m)
    take = min(n, TARGET_ROWS - rows)
    if take <= 0:
        break

    X_buf[rows : rows + take, 0] = dist_m[:take]
    X_buf[rows : rows + take, 1] = pc_m[:take]
    X_buf[rows : rows + take, 2] = hr_m[:take]
    X_buf[rows : rows + take, 3] = dow_m[:take]
    X_buf[rows : rows + take, 4] = 1.0

    y_buf[rows : rows + take] = np.log1p(fare_m[:take])

    w = np.ones((take,), dtype=np.float64)
    w[a_mask[:take]] = AIRPORT_WEIGHT
    w_buf[rows : rows + take] = w

    rows += take
    print(f"Collected rows: {rows} (after chunk {chunk_idx})")

train_X = X_buf[:rows]
train_y = y_buf[:rows]
train_w = w_buf[:rows]

imp = SimpleImputer(missing_values=np.nan, strategy="mean")
train_X = imp.fit_transform(train_X)

regr = GradientBoostingRegressor(n_estimators=100, warm_start=False, random_state=42)
regr = incremental_training(train_X, train_y, regr, sample_weight=train_w)
print("Training done on rows:", rows)



## === cell 5
tdf = pd.read_csv(
    os.path.join(INPUT_DIR, "test.csv"),
    usecols=TEST_USECOLS,
    dtype=TEST_DTYPES,
    engine="c",
    low_memory=False,
)

tdf_all = tdf  # same object; we only need 'key' and original ordering.

tdf = distance_travel(tdf)
tdf = add_time_features(tdf)  # keep feature building consistent

tdf_valid = tdf[(tdf["passenger_count"] > 0) & (tdf["distance_travel"] > 0)]
tdf_valid = filter_zero_gps(tdf_valid)

tdf_in = filter_nyc_bbox(tdf_valid)

ttrain_X_in = np.column_stack(
    (
        np.asarray(tdf_in.distance_travel, dtype=np.float64),
        np.asarray(tdf_in.passenger_count, dtype=np.float64),
        np.asarray(tdf_in.pickup_hour, dtype=np.float64),
        np.asarray(tdf_in.pickup_dow, dtype=np.float64),
        np.ones(len(tdf_in), dtype=np.float64),
    )
)
ttrain_X_in = imp.transform(ttrain_X_in)
pred_in = regr.predict(ttrain_X_in)

ttrain_X_all = np.column_stack(
    (
        np.asarray(tdf.distance_travel, dtype=np.float64),
        np.asarray(tdf.passenger_count, dtype=np.float64),
        np.asarray(tdf.pickup_hour, dtype=np.float64),
        np.asarray(tdf.pickup_dow, dtype=np.float64),
        np.ones(len(tdf), dtype=np.float64),
    )
)
ttrain_X_all = imp.transform(ttrain_X_all)
pred_all = regr.predict(ttrain_X_all)

pred_series = pd.Series(pred_all, index=tdf.index)
pred_series.loc[tdf_in.index] = pred_in

output = np.expm1(pred_series.reindex(tdf_all.index).to_numpy())
output = np.maximum(output, 0.0)

print(output[:10])



## === cell 6
my_submission = pd.DataFrame({"key": tdf_all["key"], "fare_amount": output})
my_submission.to_csv("submission.csv", index=False)
print(my_submission.head())
print("Wrote submission.csv with shape:", my_submission.shape)
