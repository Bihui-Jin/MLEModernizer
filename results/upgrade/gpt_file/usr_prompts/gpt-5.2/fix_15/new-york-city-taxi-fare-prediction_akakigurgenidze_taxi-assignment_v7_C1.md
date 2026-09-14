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

3.12

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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

3.39438

# 6. Current score

7.77098

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 4.82837) has done: 'I fix the crash caused by an incompatible protobuf version loaded indirectly by TensorFlow by removing the unused TensorFlow/Keras imports (your model is scikit-learn + XGBoost only). Then I correct a logic bug in the XGBoost validation DMatrix (it was built from `X_val` but without `label=y_val`, which can misalign evaluation/debugging) and add a small, metric-aligned, score-improving change: log1p-transform the target for XGBoost training and invert with expm1 at inference (keeps the same model family/training loop while typically reducing RMSE for fare data). Finally, I ensure the submission is written as `submission.csv` with the required `key,fare_amount` columns and non-negative predictions.'
- What this solution (achieved 1068.73817) has done: 'You’re currently far above the target RMSE (4.83 vs 3.39, lower is better), so we should cautiously improve generalization without changing the core approach (same features, same XGBoost training API). The biggest likely issue is the outlier handling: replacing bad coordinates with the mean injects many unrealistic “average-location” trips that add noise; instead, we minimally change it to *drop* those invalid rows from the training set only (test stays untouched), which usually improves RMSE. We also add a small, standard NYC-taxi feature (`haversine_distance`) while keeping the same overall feature-engineering style and model family. Finally, we train XGBoost on the full cleaned training data (still 10M max rows) with slightly more boosting rounds (no early stopping) to move the score toward the target band.'
- What this solution (achieved 5.12924) has done: 'Your current public score (RMSE ~1068) indicates a submission alignment/format bug rather than a modeling issue, because the same approach typically scores single-digit RMSE. The smallest high-impact fix is to ensure the `key` column is read and written as an exact string (no float/scientific notation) and that predictions are aligned to the exact row order of `test.csv` (no implicit reindexing). I also harden datetime parsing and ensure feature columns have consistent numeric dtypes so XGBoost doesn’t silently ingest object/NaN values that can explode predictions. These changes preserve your model, features, training loop, and log1p target transform—just make the pipeline deterministic and submission-valid.'
- What this solution (achieved 1141.00656) has done: 'Your current RMSE (5.129) is still above the target (3.394, lower is better), so we should make small, safe improvements that don’t change your overall XGBoost approach. The biggest likely miss is that the model is trained on only the 80% training split; switching to train on all cleaned data after validating typically reduce public RMSE without changing architecture or features. I also add the standard NYC taxi “abs_lat_diff/abs_lon_diff” features (very small feature-engineering extension consistent with your current style) and clamp negative prediction *before* applying log1p to avoid rare invalid values. Finally, I keep key handling and row order untouched to preserve submission alignment.'
- What this solution (achieved 4.79534) has done: 'Your RMSE (~1141) is so far from the target (3.39, lower is better) that it almost certainly comes from invalid/NaN/inf features in the **test** set causing extreme predictions, even though training rows were cleaned. I make the smallest change that fixes this: apply the same validity mask to the test set as a *diagnostic*, but instead of dropping rows (which would break submission), we **set predictions for invalid test rows to a safe fallback** (the training median fare in original scale). I also harden feature generation by replacing any inf/-inf in engineered features with NaN and then imputing with training medians (so XGBoost never sees inf). This preserves your model, features, and training loop; it only fixes test-time data integrity and prediction fallback to bring the score back into a normal range toward the target.'
- What this solution (achieved 831.75287) has done: 'Your current RMSE (4.795) is still worse than the target (3.394, lower is better), so we make the smallest changes likely to improve generalization without changing your overall XGBoost approach. The main issue is that `pickup_month` is engineered but never used as a feature, which throws away strong seasonality signal; we add it to `features`. We also add one very standard, minimal geospatial feature (`euclidean_distance` in lat/lon space) consistent with your existing distance features. Finally, we slightly increase boosting rounds (from 300 to 500) to reduce underfitting while keeping the same params/training API and no early stopping.'
- What this solution (achieved 30.37376) has done: 'Your current score (RMSE 831) is far worse than the target, which strongly suggests a submission integrity issue rather than the model itself. The smallest high-impact fix is to prevent catastrophic predictions on a few test rows by clipping extreme distance features to the training distribution (winsorization) and ensuring all remaining NaNs/infs are imputed consistently from training stats. This preserves your exact model family, training API, features, and log1p target transform; it only stabilizes test-time feature values so XGBoost can’t output huge fares. I also make `key` handling stricter (string + no NaNs) and keep test row order unchanged to avoid any alignment problems.'
- What this solution (achieved 30.37376) has done: 'Your current RMSE (30.37, lower is better) is far above the target (3.39), and this code path most likely has a test-time integrity issue: the test-set NaNs created by datetime parsing are never imputed (only train rows are dropped), so XGBoost can receive NaNs/invalid time features and produce unstable predictions for some rows. I make the smallest fix that preserves your model and features: impute missing time-derived features in `test_df` with training medians (just like other features), and also impute missing `near_airport/is_weekend/is_holiday` as 0 for safety. Additionally, I clip the final predictions to a reasonable upper bound learned from the training distribution (e.g., 99.9th percentile) to prevent a few extreme predictions from dominating RMSE, while keeping your fallback for invalid coordinate rows unchanged. These changes should move your score down toward the target band without changing the core XGBoost approach.'
- What this solution (achieved 4.75912) has done: 'Your current RMSE (30.37) is far above the target (3.39), so the most likely issue is still test-time feature integrity: even though you impute `test_df[features]` before clipping, the subsequent clipping can reintroduce NaNs if any clip bounds are NaN, and the engineered time/boolean columns can remain NaN on some rows. I make a minimal, score-directed fix by (1) ensuring clip bounds are always finite (fallback to global finite min/max), (2) re-imputing `test_df[features]` *after* clipping, and (3) adding a very small safeguard to replace any remaining non-finite values in `X_test` with training medians before building the DMatrix. This keeps your model, features, log1p target transform, and training loop the same, but prevents a small number of pathological test rows from causing large prediction errors that dominate RMSE. The submission format and row order remain unchanged, and we still keep your invalid-coordinate fallback and fare cap.'
- What this solution (achieved 30.37376) has done: 'We’re still above the target RMSE (4.759 vs 3.394, lower is better), so we make two minimal, high-signal changes that preserve your exact XGBoost approach and features. First, we align the log1p training with the RMSE metric by adding a custom evaluation function that computes RMSE in the original fare scale (expm1), and we use the validation set to choose a better number of boosting rounds via early stopping (this doesn’t change the model family, loss, or features, and typically moves RMSE down materially). Second, we apply the same “distance feature clipping” to `train_df` before recomputing `train_medians` and building `X/y`, so the imputation statistics match the post-clip distribution (reduces train/test feature distribution mismatch). Submission format, key alignment/order, invalid-test fallback, and fare caps remain unchanged.'
- What this solution (achieved 7.75612) has done: 'The timeout is dominated by reading 10M rows with expensive dtype inference, repeated full-column conversions (`pd.to_numeric` in Python loops), repeated quantile/median passes over a wide DataFrame, and (most of all) training XGBoost twice (once for early stopping, then again on the full dataset). I make the data load and feature engineering provably equivalent but faster by using explicit dtypes, vectorized NumPy feature computation, avoiding redundant passes (single replace/clip/fill), and computing quantiles/medians on NumPy arrays. For XGBoost, I preserve the exact training/evaluation semantics while removing the second full retrain by using the early-stopped model (best_iteration) to predict on test; this avoids duplicating the most expensive step without changing the model selection logic. I also enable Intel scikit-learn patching (if available) for the linear regression baseline and remove notebook-only display calls that waste time.'
- What this solution (achieved 7.77081) has done: 'The timeout is dominated by heavy pandas work on 10M rows (datetime parsing, repeated DataFrame copies, multiple quantile computations, and repeated NaN/inf replacements) plus very large intermediate DataFrames kept longer than needed. I keep the same features, model choices, and training semantics, but make the preprocessing and clipping steps operate on NumPy arrays with fewer passes, avoid unnecessary `.copy()` and repeated `.replace()`, and compute all clip quantiles in a single vectorized call. I also free large objects as soon as they’re no longer needed and ensure XGBoost uses the fastest equivalent settings (same algorithm: histogram tree method, same rounds/early stopping/metric) while avoiding redundant DMatrix construction work. These changes are correctness-preserving (same rows kept/dropped, same feature definitions, same clipping definition, same training/validation split and models) and reduce constant-factor overhead enough to fit within 600 seconds.'
- What this solution (achieved 7.77098) has done: 'The timeout is dominated by reading 10M rows with `parse_dates` plus slow feature engineering and quantile/median computations on a huge DataFrame, and then training an XGBoost model with potentially thousands of boosting rounds. I keep the exact same features, models, loss/metrics, and training semantics, but speed up data loading by parsing datetimes after reading (vectorized, same UTC logic), reduce unnecessary copies/temporaries during preprocessing, and avoid repeated expensive conversions. I also make XGBoost use an external-memory cache for the DMatrix to reduce RAM pressure and overhead, while keeping identical parameters/early stopping behavior. All outputs/paths remain unchanged.'
- What this solution (achieved 7.77098) has done: 'The timeout is dominated by (1) reading and preprocessing 10M training rows with heavy pandas overhead and (2) very long XGBoost training with 5000 boosting rounds, plus an extra full-pass prediction on the entire training set. I keep the exact same features, cleaning rules, model types, and evaluation semantics, but make the pipeline faster by switching the CSV ingest to a chunked + vectorized NumPy approach (same rows, same dtypes), doing feature engineering in-place without pandas `.dt` accessors, and eliminating unnecessary full-data copies. I also avoid the expensive “predict on all X for train RMSE” (not used for submission) and let XGBoost use its native external-memory DMatrix (cache) to reduce memory pressure and speed training on large data while keeping the same early stopping logic and best-iteration usage. All randomness/seed behavior is preserved.'

# 9. Code solution

## === cell 0
import os
import gc
import numpy as np
import pandas as pd

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

import xgboost as xgb

TRAIN_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
TEST_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"

DTYPES_TRAIN = {
    "key": "string",
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}
DTYPES_TEST = {
    "key": "string",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}

TRAIN_USECOLS = list(DTYPES_TRAIN.keys()) + ["pickup_datetime"]
TEST_USECOLS = list(DTYPES_TEST.keys()) + ["pickup_datetime"]

NROWS_TRAIN = 10_000_000
CHUNK_SIZE = 1_000_000

train_chunks = []
read_rows = 0
for chunk in pd.read_csv(
    TRAIN_PATH,
    dtype=DTYPES_TRAIN,
    usecols=TRAIN_USECOLS,
    chunksize=CHUNK_SIZE,
):
    remaining = NROWS_TRAIN - read_rows
    if remaining <= 0:
        break
    if len(chunk) > remaining:
        chunk = chunk.iloc[:remaining].copy()
    train_chunks.append(chunk)
    read_rows += len(chunk)
    if read_rows >= NROWS_TRAIN:
        break
train_df = pd.concat(train_chunks, ignore_index=True)
del train_chunks
gc.collect()

test_df = pd.read_csv(
    TEST_PATH,
    dtype=DTYPES_TEST,
    usecols=TEST_USECOLS,
)

train_df["pickup_datetime"] = pd.to_datetime(
    train_df["pickup_datetime"], errors="coerce", utc=True
)
test_df["pickup_datetime"] = pd.to_datetime(
    test_df["pickup_datetime"], errors="coerce", utc=True
)



## === cell 1
print("Loaded train/test:", train_df.shape, test_df.shape)



## === cell 2
print("Train dtypes:", train_df.dtypes.to_dict())



## === cell 3
pass



## === cell 4
num_rows = len(train_df)
mask_pos = train_df["fare_amount"].to_numpy(copy=False) > 0
train_df = train_df.loc[mask_pos]
print(f"Drop {num_rows - len(train_df)} rows")
del mask_pos
gc.collect()




## === cell 5
def get_valid_mask_np(df: pd.DataFrame) -> np.ndarray:
    plat = df["pickup_latitude"].to_numpy(copy=False)
    dlat = df["dropoff_latitude"].to_numpy(copy=False)
    plon = df["pickup_longitude"].to_numpy(copy=False)
    dlon = df["dropoff_longitude"].to_numpy(copy=False)
    pc = df["passenger_count"].to_numpy(copy=False)
    return (
        (plat >= 40.5)
        & (plat <= 41.0)
        & (dlat >= 40.5)
        & (dlat <= 41.0)
        & (plon >= -74.3)
        & (plon <= -73.60)
        & (dlon >= -74.3)
        & (dlon <= -73.60)
        & (pc >= 1)
        & (pc <= 10)
    )


before_len = len(train_df)
train_mask = get_valid_mask_np(train_df)
train_df = train_df.loc[train_mask]
print("Dropped invalid/outlier rows from training:", before_len - len(train_df))

test_valid_mask = get_valid_mask_np(test_df)
print(
    "Test rows (unchanged):",
    len(test_df),
    "| invalid rows:",
    int((~test_valid_mask).sum()),
)
del train_mask
gc.collect()



## === cell 6
pass



## === cell 7
pass




## === cell 8
def preprocess_data(df: pd.DataFrame) -> None:
    airport_lat, airport_lon = 40.644600, -73.779700
    lga_lat, lga_lon = 40.7733, -73.8718

    plat = df["pickup_latitude"].to_numpy(dtype=np.float32, copy=False)
    plon = df["pickup_longitude"].to_numpy(dtype=np.float32, copy=False)
    dlat = df["dropoff_latitude"].to_numpy(dtype=np.float32, copy=False)
    dlon = df["dropoff_longitude"].to_numpy(dtype=np.float32, copy=False)

    near_jfk = (
        (plat <= airport_lat + 0.005)
        & (plat >= airport_lat - 0.005)
        & (plon <= airport_lon + 0.005)
        & (plon >= airport_lon - 0.005)
    )
    near_lga = (
        (plat <= lga_lat + 0.002)
        & (plat >= lga_lat - 0.003)
        & (plon <= lga_lon + 0.005)
        & (plon >= lga_lon - 0.005)
    )
    df["near_airport"] = (near_jfk | near_lga).astype(np.int8)

    lon_diff = plon - dlon
    lat_diff = plat - dlat

    df["manhattan_distance"] = (np.abs(lon_diff) + np.abs(lat_diff)).astype(np.float32)
    df["abs_lon_diff"] = np.abs(lon_diff).astype(np.float32)
    df["abs_lat_diff"] = np.abs(lat_diff).astype(np.float32)
    df["euclidean_distance"] = np.sqrt(
        lon_diff * lon_diff + lat_diff * lat_diff
    ).astype(np.float32)

    R = np.float32(6371.0)  # km
    lat1 = np.radians(plat.astype(np.float64, copy=False))
    lon1 = np.radians(plon.astype(np.float64, copy=False))
    lat2 = np.radians(dlat.astype(np.float64, copy=False))
    lon2 = np.radians(dlon.astype(np.float64, copy=False))
    dlat_r = lat2 - lat1
    dlon_r = lon2 - lon1
    a = (
        np.sin(dlat_r / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon_r / 2.0) ** 2
    )
    df["haversine_distance"] = (2.0 * float(R) * np.arcsin(np.sqrt(a))).astype(
        np.float32
    )

    dt = df["pickup_datetime"]
    if not pd.api.types.is_datetime64_any_dtype(dt):
        dt = pd.to_datetime(dt, errors="coerce", utc=True)
        df["pickup_datetime"] = dt

    dt64 = dt.to_numpy(copy=False)  # datetime64[ns, UTC] as ndarray (may contain NaT)
    dt_ns = dt64.astype("datetime64[ns]")

    years = (dt_ns.astype("datetime64[Y]").astype(np.int32) + 1970).astype(np.float32)

    months = (dt_ns.astype("datetime64[M]").astype(np.int32) % 12 + 1).astype(
        np.float32
    )

    hours = (
        (
            (dt_ns.astype("datetime64[h]") - dt_ns.astype("datetime64[D]"))
            .astype("timedelta64[h]")
            .astype(np.int32)
        )
    ).astype(np.float32)

    days_since_epoch = dt_ns.astype("datetime64[D]").astype(np.int32)
    dow = ((days_since_epoch + 3) % 7).astype(np.float32)

    df["pickup_year"] = years
    df["pickup_month"] = months
    df["pickup_hour"] = hours
    df["pickup_day"] = dow

    is_weekend = ((dow >= 5) & (dow <= 6)).astype(np.int8)
    df["is_weekend"] = is_weekend

    day_of_month = (
        (dt_ns.astype("datetime64[D]") - dt_ns.astype("datetime64[M]") + 1)
        .astype("timedelta64[D]")
        .astype(np.int32)
    )

    month_i = months.astype(np.int32)
    df["is_holiday"] = (
        ((month_i == 12) & (day_of_month == 25))
        | ((month_i == 12) & (day_of_month == 26))
        | ((month_i == 12) & (day_of_month == 31))
        | ((month_i == 1) & (day_of_month == 1))
        | ((month_i == 7) & (day_of_month == 4))
    ).astype(np.int8)


preprocess_data(train_df)
preprocess_data(test_df)

time_cols = ["pickup_year", "pickup_month", "pickup_hour", "pickup_day"]
before_len = len(train_df)
train_df = train_df.dropna(subset=time_cols)
print("Dropped train rows with invalid pickup_datetime:", before_len - len(train_df))
gc.collect()



## === cell 9
print("Train preview columns:", train_df.columns.tolist()[:10], "...")



## === cell 10
print("near_airport sum:", int(train_df["near_airport"].sum()))



## === cell 11
print("is_holiday sum:", int(train_df["is_holiday"].sum()))



## === cell 12
print("is_weekend sum:", int(train_df["is_weekend"].sum()))



## === cell 13
features = [
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
    "near_airport",
    "manhattan_distance",
    "abs_lon_diff",
    "abs_lat_diff",
    "euclidean_distance",
    "haversine_distance",
    "passenger_count",
    "pickup_year",
    "pickup_month",
    "pickup_hour",
    "is_weekend",
    "is_holiday",
]

distance_like = [
    "manhattan_distance",
    "abs_lon_diff",
    "abs_lat_diff",
    "euclidean_distance",
    "haversine_distance",
]

dist_train = train_df[distance_like].to_numpy(dtype=np.float32, copy=False)
dist_train = np.where(np.isfinite(dist_train), dist_train, np.nan)

q_lo = np.nanquantile(dist_train, 0.001, axis=0)
q_hi = np.nanquantile(dist_train, 0.999, axis=0)

finite_min = np.nanmin(dist_train, axis=0)
finite_max = np.nanmax(dist_train, axis=0)

q_lo = np.where(np.isfinite(q_lo), q_lo, finite_min)
q_hi = np.where(np.isfinite(q_hi), q_hi, finite_max)

swap = q_hi < q_lo
q_lo2 = np.where(swap, q_hi, q_lo)
q_hi2 = np.where(swap, q_lo, q_hi)
q_lo, q_hi = q_lo2, q_hi2
del swap, q_lo2, q_hi2, finite_min, finite_max, dist_train
gc.collect()

train_dist = train_df[distance_like].to_numpy(dtype=np.float32, copy=False)
test_dist = test_df[distance_like].to_numpy(dtype=np.float32, copy=False)
train_dist = np.clip(train_dist, q_lo.astype(np.float32), q_hi.astype(np.float32))
test_dist = np.clip(test_dist, q_lo.astype(np.float32), q_hi.astype(np.float32))
train_df.loc[:, distance_like] = train_dist
test_df.loc[:, distance_like] = test_dist
del train_dist, test_dist
gc.collect()

train_medians = train_df[features].median(numeric_only=True)

for c in ["near_airport", "is_weekend", "is_holiday"]:
    test_df[c] = test_df[c].fillna(0)

test_df[features] = test_df[features].fillna(train_medians)

before_len = len(train_df)
train_df = train_df.dropna(subset=features + ["fare_amount"])
print("Dropped train rows with NaNs in features/target:", before_len - len(train_df))

X = train_df[features].to_numpy(dtype=np.float32, copy=False)
y = train_df["fare_amount"].to_numpy(dtype=np.float32, copy=False)

finite_rows = np.isfinite(X).all(axis=1) & np.isfinite(y)
if not finite_rows.all():
    X = X[finite_rows]
    y = y[finite_rows]
print("Final training rows:", X.shape[0])

fallback_fare = float(np.median(y))
print("Fallback fare (train median, original scale):", fallback_fare)

finite_y = y[np.isfinite(y)]
fare_cap_hi = float(np.quantile(finite_y, 0.999)) if finite_y.size else float(np.max(y))
if not np.isfinite(fare_cap_hi) or fare_cap_hi <= 0:
    fare_cap_hi = float(np.max(y))
print("Fare cap high (train 99.9th pct):", fare_cap_hi)

del train_df
gc.collect()



## === cell 14
X_test = test_df[features].to_numpy(dtype=np.float32, copy=False)
if not np.isfinite(X_test).all():
    med = train_medians.to_numpy(dtype=np.float32, copy=False)
    X_test = np.where(np.isfinite(X_test), X_test, med)



## === cell 15
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=69)



## === cell 16
print(X.shape, y.shape, X_train.shape, y_train.shape, X_val.shape, y_val.shape)



## === cell 17
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)



## === cell 18
validation_predictions_lr = lr_model.predict(X_val)
validation_rmse_lr = np.sqrt(mean_squared_error(y_val, validation_predictions_lr))
print("Validation RMSE (Linear Regression):", float(validation_rmse_lr))

train_predictions_lr = lr_model.predict(X)
train_rmse_lr = np.sqrt(mean_squared_error(y, train_predictions_lr))
print("Training RMSE (Linear Regression):", float(train_rmse_lr))



## === cell 19
test_predictions_lr = lr_model.predict(X_test)

submission_df_lr = pd.DataFrame(
    {
        "key": test_df["key"].astype("string").fillna(""),
        "fare_amount": test_predictions_lr,
    },
    columns=["key", "fare_amount"],
)
submission_df_lr.to_csv("lr_submission.csv", index=False)



## === cell 20
xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "eta": 0.05,
    "min_child_weight": 3,
    "gamma": 0.1,
    "seed": 42,
    "tree_method": "hist",
    "nthread": -1,
}

y_train_log = np.log1p(np.maximum(y_train, 0.0))
y_val_log = np.log1p(np.maximum(y_val, 0.0))

cache_prefix = "/kaggle/working/xgb_cache"
os.makedirs(cache_prefix, exist_ok=True)

train_np_path = os.path.join(cache_prefix, "train.npy")
val_np_path = os.path.join(cache_prefix, "val.npy")
test_np_path = os.path.join(cache_prefix, "test.npy")
np.save(train_np_path, X_train)
np.save(val_np_path, X_val)
np.save(test_np_path, X_test)

dtrain = xgb.DMatrix(
    f"{train_np_path}?format=npy&cache_prefix={cache_prefix}/dtrain",
    label=y_train_log,
    missing=np.nan,
    nthread=-1,
)
dval = xgb.DMatrix(
    f"{val_np_path}?format=npy&cache_prefix={cache_prefix}/dval",
    label=y_val_log,
    missing=np.nan,
    nthread=-1,
)
dtest = xgb.DMatrix(
    f"{test_np_path}?format=npy&cache_prefix={cache_prefix}/dtest",
    missing=np.nan,
    nthread=-1,
)


def rmse_original_scale(preds_log, dmatrix):
    y_true_log = dmatrix.get_label()
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(preds_log)
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    return "rmse_orig", rmse


watchlist = [(dtrain, "train"), (dval, "valid")]
xgb_model = xgb.train(
    xgb_params,
    dtrain,
    num_boost_round=5000,
    evals=watchlist,
    feval=rmse_original_scale,
    maximize=False,
    early_stopping_rounds=100,
    verbose_eval=200,
)



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
XGBoostError                              Traceback (most recent call last)
/tmp/ipykernel_11/2573386015.py in <cell line: 0>()
     29 np.save(test_np_path, X_test)
     30 
---> 31 dtrain = xgb.DMatrix(
     32     f"{train_np_path}?format=npy&cache_prefix={cache_prefix}/dtrain",
     33     label=y_train_log,

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in inner_f(*args, **kwargs)
    728             for k, arg in zip(sig.parameters, args):
    729                 kwargs[k] = arg
--> 730             return func(**kwargs)
    731 
    732         return inner_f

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in __init__(self, data, label, weight, base_margin, missing, silent, feature_names, feature_types, nthread, group, qid, label_lower_bound, label_upper_bound, feature_weights, enable_categorical, data_split_mode)
    855             return
    856 
--> 857         handle, feature_names, feature_types = dispatch_data_backend(
    858             data,
    859             missing=self.missing,

/usr/local/lib/python3.11/dist-packages/xgboost/data.py in dispatch_data_backend(data, missing, threads, feature_names, feature_types, enable_categorical, data_split_mode)
   1077         )
   1078     if _is_uri(data):
-> 1079         return _from_uri(data, missing, feature_names, feature_types, data_split_mode)
   1080     if _is_list(data):
   1081         return _from_list(data, missing, threads, feature_names, feature_types)

/usr/local/lib/python3.11/dist-packages/xgboost/data.py in _from_uri(data, missing, feature_names, feature_types, data_split_mode)
    992     }
    993     config = bytes(json.dumps(args), "utf-8")
--> 994     _check_call(_LIB.XGDMatrixCreateFromURI(config, ctypes.byref(handle)))
    995     return handle, feature_names, feature_types
    996 

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in _check_call(ret)
    280     """
    281     if ret != 0:
--> 282         raise XGBoostError(py_str(_LIB.XGBGetLastError()))
    283 
    284 

XGBoostError: [03:22:48] /workspace/dmlc-core/src/data.cc:97: Unknown data type npy
Stack trace:
  [bt] (0) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xc5e62a) [0x7f96d630f62a]
  [bt] (1) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xc6806b) [0x7f96d631906b]
  [bt] (2) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x3621a1) [0x7f96d5a131a1]
  [bt] (3) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(XGDMatrixCreateFromURI+0xe8) [0x7f96d5816258]
  [bt] (4) /usr/lib/x86_64-linux-gnu/libffi.so.8(+0x7e2e) [0x7f9752066e2e]
  [bt] (5) /usr/lib/x86_64-linux-gnu/libffi.so.8(+0x4493) [0x7f9752063493]
  [bt] (6) /usr/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so(+0xa4d8) [0x7f9750ee44d8]
  [bt] (7) /usr/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so(+0x9c8e) [0x7f9750ee3c8e]
  [bt] (8) /usr/bin/python3(_PyObject_MakeTpCall+0x27c) [0x52f85c]



## === cell 21
validation_predictions_xgb_log = xgb_model.predict(
    dval, iteration_range=(0, xgb_model.best_iteration + 1)
)
validation_predictions_xgb = np.expm1(validation_predictions_xgb_log)
validation_rmse_xgb = np.sqrt(mean_squared_error(y_val, validation_predictions_xgb))
print("Validation RMSE (XGBoost):", float(validation_rmse_xgb))
print("Best iteration chosen:", int(xgb_model.best_iteration))

print("Train RMSE (XGBoost): skipped to meet 600s runtime budget.")



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2266191587.py in <cell line: 0>()
----> 1 validation_predictions_xgb_log = xgb_model.predict(
      2     dval, iteration_range=(0, xgb_model.best_iteration + 1)
      3 )
      4 validation_predictions_xgb = np.expm1(validation_predictions_xgb_log)
      5 validation_rmse_xgb = np.sqrt(mean_squared_error(y_val, validation_predictions_xgb))

NameError: name 'xgb_model' is not defined

## === cell 22
test_predictions_xgb_log = xgb_model.predict(
    dtest, iteration_range=(0, xgb_model.best_iteration + 1)
)
test_predictions_xgb = np.expm1(test_predictions_xgb_log)

test_predictions_xgb = np.where(test_valid_mask, test_predictions_xgb, fallback_fare)
test_predictions_xgb = np.maximum(test_predictions_xgb, 0.0)
test_predictions_xgb = np.minimum(test_predictions_xgb, fare_cap_hi)

submission_df_xgb = pd.DataFrame(
    {
        "key": test_df["key"].astype("string").fillna(""),
        "fare_amount": test_predictions_xgb,
    },
    columns=["key", "fare_amount"],
)
submission_df_xgb.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission_df_xgb.shape)
print("Submission dtypes:", submission_df_xgb.dtypes.to_dict())
print("Invalid test rows handled with fallback:", int((~test_valid_mask).sum()))
print("First 3 rows:\n", submission_df_xgb.head(3))
print(
    "Any NaN fare in submission?:",
    bool(pd.isna(submission_df_xgb["fare_amount"]).any()),
)
print("Fare stats:", submission_df_xgb["fare_amount"].describe().to_dict())
print("Rounds used (best_iteration+1):", int(xgb_model.best_iteration) + 1)

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/433093770.py in <cell line: 0>()
----> 1 test_predictions_xgb_log = xgb_model.predict(
      2     dtest, iteration_range=(0, xgb_model.best_iteration + 1)
      3 )
      4 test_predictions_xgb = np.expm1(test_predictions_xgb_log)
      5 

NameError: name 'xgb_model' is not defined
