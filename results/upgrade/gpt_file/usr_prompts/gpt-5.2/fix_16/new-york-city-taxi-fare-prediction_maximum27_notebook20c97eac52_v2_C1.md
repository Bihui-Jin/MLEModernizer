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

cudf-polars-cu12==25.6.0
geopandas==0.14.4
lightgbm==4.6.0
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
polars==1.25.0
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

3.31631

# 6. Current score

5.99252

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.59576) has done: 'I fix the Polars concatenation error by ensuring train/test have identical columns before `pl.concat`, and by explicitly separating the target (`fare_amount`) before feature engineering so it doesn’t leak into the concat widths. Then I prevent LightGBM’s dtype error by guaranteeing `key` and `pickup_datetime` are not present in the training features (they be kept only for submission IDs). Finally, I make GPU training robust by falling back to CPU automatically if the Kaggle runtime doesn’t expose a compatible LightGBM GPU device, ensuring the pipeline always trains and writes a valid `submission_*.csv`.'
- What this solution (achieved 9.83271) has done: 'Your score gap is positive (4.59576 vs target 3.31631; lower is better), so we should improve accuracy toward the target with minimal risk. The largest, safest gain here is to remove obvious outliers/invalid rows from the sampled training data (bad coordinates, extreme fares, zero-distance edge cases) before training—this preserves your model and feature logic but reduces noise that LightGBM otherwise tries to fit. I also set `num_boost_round` to use `best_iteration` via a standard LightGBM callback (still the same training approach/loops; just prevents under/over-training) and keep GPU fallback unchanged. Finally, I clamp negative predictions to 0.0 (fares cannot be negative), which typically reduces RMSE a bit without changing the model architecture.'
- What this solution (achieved 5.20371) has done: 'Your current RMSE (9.83) is far worse than the target (3.316), so we should improve accuracy with the smallest, safest changes that don’t alter your model/feature logic. The biggest issue is that your script reads the full 55M-row CSV into memory and then does a huge random index sample; this tends to be unstable/slow and can indirectly degrade model quality due to runtime pressure and inconsistent sampling behavior. I switch to Polars’ built-in sampling (same sample size, same seed) after cleaning, which is both faster and more reliable, and I add one more standard NYC taxi cleaning filter to remove implausible trips based on your already-computed “distance” (very large distances with small fares are label-noisy). Finally, I ensure `passenger_count` is treated consistently as numeric and keep everything else (features, LightGBM setup, post-processing) unchanged.'
- What this solution (achieved 5.25374) has done: 'Your current RMSE (5.20371) is worse than the target (3.31631), so we should improve accuracy with minimal, low-risk changes that keep your overall pipeline intact. The biggest quality issue is likely that your validation split is random across all years, while the test set is from 2015; switching to a time-based split (train on earlier rides, validate on later rides) improves generalization without changing the model or feature set. I also remove early stopping (it can underfit and adds nondeterminism in iteration count) and instead use a fixed number of boosting rounds so training is stable and typically closer to the public LB for this competition. Finally, I add one standard NYC taxi cleaning constraint (reasonably bounding the engineered `distance` before sampling) to reduce label noise while preserving your feature logic.'
- What this solution (achieved 5.99244) has done: 'Your current RMSE (5.25374, lower-is-better) is still well above the target (3.31631), so we should improve generalization with the smallest changes that don’t alter your model/feature pipeline. The biggest likely issue is the validation split: using only `pickup_year >= 2014` doesn’t match the 2015 test distribution well; switching to a time-based split using `pickup_datetime` (already parsed during preprocessing) better simulates the test period while keeping the same LightGBM training loop. I also add one very standard, low-risk cleaning constraint used in this competition (bounding the engineered `distance` and `fare/distance`) to reduce label noise without changing features or the model. Finally, I keep submission formatting identical but ensure `test_key` alignment is preserved.'
- What this solution (achieved 5.16999) has done: 'I fix the root cause of the empty training set by ensuring `pickup_datetime` is parsed correctly for this competition’s actual timestamp format (it includes microseconds and a trailing “ UTC” in many rows), which currently turns most datetimes into nulls and drops everything. I also prevent the `y_pl` misalignment bug by postponing sampling until after feature engineering (so any rows dropped by datetime parsing/distance filters drop the target in-sync). Finally, I make inference robust by aligning test columns to the final training feature columns (`X.columns`) so submission creation always works, and keep the LightGBM training core logic unchanged.'
- What this solution (achieved 6.06953) has done: 'Your current RMSE (5.16999, lower-is-better) is still well above the target (3.31631), so we should make small, low-risk changes that typically improve generalization without changing your model/feature core. The biggest “safe” gain in this competition is better label/feature denoising: add a standard fare-per-distance sanity filter (using your already-engineered `distance`) and a mild upper bound on `distance` before training to reduce noisy/implausible trips. I also fix a subtle but important alignment risk: after dropping rows via the distance-based filters, `y_pl` must be filtered with the exact same boolean mask expression (not via Series extraction), to guarantee perfect row alignment. Everything else (feature engineering steps, LightGBM objective, boosting type, training loop, prediction post-processing, and submission format) stays the same.'
- What this solution (achieved 6.04208) has done: 'Your current blocker is not model quality but an environment/runtime error (`sqlite3.OperationalError: disk I/O error`) coming from `mlebench` caching; since we can’t safely control that from inside the notebook, the most reliable way to “move toward the target score” is to ensure the notebook always finishes and writes a valid Kaggle submission CSV deterministically. I keep your model and feature engineering identical, but make file paths robust to both `/kaggle/input/...` and the provided `/kaggle/data/...` layout, and I always also write `submission.csv` in addition to `submission_lgbm.csv` so graders that look for a fixed filename can find it. I also add a strict assertion that the submission row count matches the test file and that keys are aligned, which prevents silent format/alignment issues that can destroy RMSE. No changes are made to the LightGBM params, training loop, feature set, cleaning logic, or prediction post-processing other than these submission/IO robustness checks.'
- What this solution (achieved 6.1502) has done: 'I remove the stray block that tries to construct a Polars DataFrame from an unevaluated `pl.Expr`, which is the direct cause of the first crash and prevents `train/test` from being created for downstream cells. I also make the distance and fare-per-km filtering apply via a single Polars boolean expression on the same DataFrame that contains both features and target, so row alignment is guaranteed (this is bug-fix/quality, not a modeling change). Finally, I ensure `test_key` is a plain Python list in the same order as `test_raw` and keep the submission writing unchanged so a valid `submission.csv` is always produced end-to-end.'
- What this solution (achieved 5.99252) has done: 'Your current RMSE (6.1502, lower-is-better) is still far above the target (3.31631), so we should improve accuracy with very small, low-risk changes that keep your model, features, and training loop intact. The biggest likely source of avoidable error here is inconsistent scaling and a suboptimal distance approximation; replacing the linear “km per degree” distance with a standard haversine distance (still just a single engineered `distance` feature) usually gives a large RMSE improvement in this competition without changing the model. I also add one more canonical NYC taxi cleaning filter (bounding lat/lon to realistic NYC ranges and removing extreme fare-per-km noise using your already-present `distance`) before training to reduce label noise. Everything else (LightGBM params, boosting rounds, split logic, submission writing) stays the same.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import polars as pl
import os


def _resolve_path(*candidates: str) -> str:
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "None of the candidate paths exist:\n" + "\n".join(candidates)
    )


TRAIN_PATH = _resolve_path(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
    "/kaggle/data/new-york-city-taxi-fare-prediction/train.csv",
    "/kaggle/data/train.csv",
)
TEST_PATH = _resolve_path(
    "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
    "/kaggle/data/new-york-city-taxi-fare-prediction/test.csv",
    "/kaggle/data/test.csv",
)

SAMPLE_SIZE_RAW = 5_000_000
SEED = 0

train_raw = pl.read_csv(
    TRAIN_PATH,
    infer_schema_length=100,
    ignore_errors=True,
).drop_nulls()

test_raw = pl.read_csv(
    TEST_PATH,
    infer_schema_length=100,
    ignore_errors=True,
).drop_nulls()

train_raw = train_raw.filter(
    (pl.col("fare_amount") > 0)
    & (pl.col("fare_amount") < 200)
    & (pl.col("passenger_count") >= 1)
    & (pl.col("passenger_count") <= 6)
    & (pl.col("pickup_longitude") >= -74.3)
    & (pl.col("pickup_longitude") <= -72.9)
    & (pl.col("dropoff_longitude") >= -74.3)
    & (pl.col("dropoff_longitude") <= -72.9)
    & (pl.col("pickup_latitude") >= 40.5)
    & (pl.col("pickup_latitude") <= 41.0)
    & (pl.col("dropoff_latitude") >= 40.5)
    & (pl.col("dropoff_latitude") <= 41.0)
    & ~(
        (pl.col("pickup_longitude") == pl.col("dropoff_longitude"))
        & (pl.col("pickup_latitude") == pl.col("dropoff_latitude"))
    )
)

if SAMPLE_SIZE_RAW < train_raw.height:
    train_raw = train_raw.sample(n=SAMPLE_SIZE_RAW, with_replacement=False, seed=SEED)

print("train_raw shape (after cleaning + sampling):", train_raw.shape)
print("test_raw shape:", test_raw.shape)



## === cell 1
import numpy as np
import polars as pl


def _parse_pickup_datetime(expr: pl.Expr) -> pl.Expr:
    cleaned = expr.cast(pl.Utf8).str.replace(r"\s+UTC$", "", literal=False)

    dt_us = cleaned.str.strptime(
        pl.Datetime, format="%Y-%m-%d %H:%M:%S%.f", strict=False
    )
    dt_s = cleaned.str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
    return pl.coalesce([dt_us, dt_s])


def preprocess(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        _parse_pickup_datetime(pl.col("pickup_datetime")).alias("pickup_datetime")
    ).drop_nulls(["pickup_datetime"])

    df = df.with_columns(
        [
            pl.col("pickup_datetime").dt.year().alias("pickup_year"),
            pl.col("pickup_datetime").dt.month().alias("pickup_month"),
            pl.col("pickup_datetime").dt.day().alias("pickup_day"),
            pl.col("pickup_datetime").dt.hour().alias("pickup_hour"),
            pl.col("pickup_datetime").dt.minute().alias("pickup_minute"),
            pl.col("pickup_datetime").dt.second().alias("pickup_second"),
            pl.col("pickup_datetime").dt.weekday().alias("pickup_weekday"),
        ]
    )

    df = df.with_columns(
        [
            (pl.col("pickup_longitude") - pl.col("dropoff_longitude"))
            .abs()
            .alias("abs_longitude"),
            (pl.col("pickup_latitude") - pl.col("dropoff_latitude"))
            .abs()
            .alias("abs_latitude"),
        ]
    )
    return df


def distance(df: pl.DataFrame) -> pl.DataFrame:
    r_km = 6371.0088
    to_rad = np.pi / 180.0

    lat1 = pl.col("pickup_latitude") * to_rad
    lon1 = pl.col("pickup_longitude") * to_rad
    lat2 = pl.col("dropoff_latitude") * to_rad
    lon2 = pl.col("dropoff_longitude") * to_rad

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (dlat / 2).sin().pow(2) + (lat1.cos() * lat2.cos() * (dlon / 2).sin().pow(2))
    c = (a.sqrt()).arcsin() * 2.0

    return df.with_columns((pl.lit(r_km) * c).alias("distance"))


def drop_encoding(df: pl.DataFrame) -> pl.DataFrame:
    drop_columns = ["key", "pickup_datetime", "pickup_minute", "pickup_second"]
    existing = [c for c in drop_columns if c in df.columns]
    return df.drop(existing)


def cycling_encoding(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        [
            (pl.col("pickup_month") * 2 * np.pi / 12).sin().alias("pickup_month_sin"),
            (pl.col("pickup_month") * 2 * np.pi / 12).cos().alias("pickup_month_cos"),
            (pl.col("pickup_hour") * 2 * np.pi / 24).sin().alias("pickup_hour_sin"),
            (pl.col("pickup_hour") * 2 * np.pi / 24).cos().alias("pickup_hour_cos"),
        ]
    )
    return df


def one_hot_encoding(df: pl.DataFrame) -> pl.DataFrame:
    one_hot_cols = ["pickup_year", "pickup_day"]
    one_hot_cols = [c for c in one_hot_cols if c in df.columns]
    if not one_hot_cols:
        return df
    other_cols = [c for c in df.columns if c not in one_hot_cols]
    dummies = df.select(one_hot_cols).to_dummies(one_hot_cols)
    return pl.concat([df.select(other_cols), dummies], how="horizontal")


def is_central(df: pl.DataFrame) -> pl.DataFrame:
    left_latitude = 40.737676
    right_latitude = 40.791716
    left_longitude = -74.015781
    right_longitude = -73.945521

    df = df.with_columns(
        (
            (pl.col("pickup_latitude") >= left_latitude)
            & (pl.col("pickup_latitude") <= right_latitude)
            & (pl.col("pickup_longitude") >= left_longitude)
            & (pl.col("pickup_longitude") <= right_longitude)
        )
        .cast(pl.Int8)
        .alias("is_pickup_central")
    )
    df = df.with_columns(
        (
            (pl.col("dropoff_latitude") >= left_latitude)
            & (pl.col("dropoff_latitude") <= right_latitude)
            & (pl.col("dropoff_longitude") >= left_longitude)
            & (pl.col("dropoff_longitude") <= right_longitude)
        )
        .cast(pl.Int8)
        .alias("is_dropoff_central")
    )
    return df


def is_short_distance(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        (pl.col("distance") < 0.32).cast(pl.Int8).alias("is_short_distance")
    )


test_key = test_raw["key"].to_list()

y_pl = train_raw["fare_amount"]
train_X_raw = train_raw.drop("fare_amount")
test_X_raw = test_raw  # no fare_amount in test

train_feat = preprocess(train_X_raw)
train_feat = distance(train_feat)
train_feat = drop_encoding(train_feat)
train_feat = cycling_encoding(train_feat)

test_feat = preprocess(test_X_raw)
test_feat = distance(test_feat)
test_feat = drop_encoding(test_feat)
test_feat = cycling_encoding(test_feat)

if "distance" in train_feat.columns and train_feat.height > 0:
    train_tmp = train_feat.with_columns(y_pl.alias("fare_amount"))

    keep_expr = (
        (pl.col("distance") >= 0.05)
        & (pl.col("distance") <= 80.0)
        & ((pl.col("fare_amount") / pl.col("distance")) >= 0.8)
        & ((pl.col("fare_amount") / pl.col("distance")) <= 30.0)
    )

    kept = int(train_tmp.select(keep_expr.sum().alias("kept"))["kept"][0])
    if kept >= 1000:
        train_tmp = train_tmp.filter(keep_expr)
        y_pl = train_tmp["fare_amount"]
        train_feat = train_tmp.drop("fare_amount")
    else:
        print(
            f"Warning: distance/$per_km filter would keep only {kept} rows; skipping."
        )

train_feat = train_feat.with_columns(pl.lit(1).alias("__is_train__"))
test_feat = test_feat.with_columns(pl.lit(0).alias("__is_train__"))

train_cols = set(train_feat.columns)
test_cols = set(test_feat.columns)
all_cols = sorted(train_cols | test_cols)


def align_cols(df: pl.DataFrame, all_cols_list: list[str]) -> pl.DataFrame:
    missing = [c for c in all_cols_list if c not in df.columns]
    if missing:
        df = df.with_columns([pl.lit(0).alias(c) for c in missing])
    return df.select(all_cols_list)


train_feat = align_cols(train_feat, all_cols)
test_feat = align_cols(test_feat, all_cols)

all_feat = pl.concat([train_feat, test_feat], how="vertical")

all_feat = one_hot_encoding(all_feat)
all_feat = is_central(all_feat)
all_feat = is_short_distance(all_feat)

train_feat_final = all_feat.filter(pl.col("__is_train__") == 1).drop("__is_train__")
test_feat_final = all_feat.filter(pl.col("__is_train__") == 0).drop("__is_train__")

train = train_feat_final.with_columns(y_pl.alias("fare_amount"))
test = test_feat_final

print("train engineered shape:", train.shape)
print("test engineered shape:", test.shape)



## === cell 2
import polars as pl

dtypes = train.dtypes
float64_columns = [
    col for col, dtype in zip(train.columns, dtypes) if dtype == pl.Float64
]
if float64_columns:
    train = train.with_columns(
        [pl.col(col).cast(pl.Float32) for col in float64_columns]
    )

dtypes = test.dtypes
float64_columns = [
    col for col, dtype in zip(test.columns, dtypes) if dtype == pl.Float64
]
if float64_columns:
    test = test.with_columns([pl.col(col).cast(pl.Float32) for col in float64_columns])

if "passenger_count" in train.columns:
    train = train.with_columns(pl.col("passenger_count").cast(pl.Int16))
if "passenger_count" in test.columns:
    test = test.with_columns(pl.col("passenger_count").cast(pl.Int16))

numeric_dtypes = (
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Float32,
    pl.Float64,
)

non_numeric_train = [
    c for c, dt in zip(train.columns, train.dtypes) if dt not in numeric_dtypes
]
if non_numeric_train:
    print("Dropping non-numeric train columns:", non_numeric_train)
    train = train.drop(non_numeric_train)

non_numeric_test = [
    c for c, dt in zip(test.columns, test.dtypes) if dt not in numeric_dtypes
]
if non_numeric_test:
    print("Dropping non-numeric test columns:", non_numeric_test)
    test = test.drop(non_numeric_test)



## === cell 3
import warnings

warnings.simplefilter("ignore")

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

if train.height == 0:
    raise RuntimeError(
        "Training set is empty after preprocessing/cleaning; cannot train model."
    )

X = train.drop(["fare_amount"]).to_pandas()
y = train["fare_amount"].to_pandas()

for c in X.columns:
    if X[c].dtype == "object":
        X[c] = pd.to_numeric(X[c], errors="coerce")
X = X.fillna(0)

pickup_year_col = "pickup_year" if "pickup_year" in X.columns else None
pickup_month_col = "pickup_month" if "pickup_month" in X.columns else None

if pickup_year_col is None or pickup_month_col is None:
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
else:
    years = X[pickup_year_col].astype(np.int16)
    months = X[pickup_month_col].astype(np.int16)

    max_year = int(years.max())
    max_month = (
        int(months[years == max_year].max())
        if (years == max_year).any()
        else int(months.max())
    )

    val_mask = (years == max_year) & (months >= max(1, max_month - 2))

    if val_mask.mean() < 0.05 or val_mask.mean() > 0.40:
        val_mask = (years > 2014) | ((years == 2014) & (months >= 10))

    if val_mask.mean() < 0.05 or val_mask.mean() > 0.95:
        from sklearn.model_selection import train_test_split

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=0
        )
    else:
        X_train, X_val = X.loc[~val_mask], X.loc[val_mask]
        y_train, y_val = y.loc[~val_mask], y.loc[val_mask]

if len(X_train) == 0 or len(X_val) == 0:
    raise RuntimeError(f"Invalid split sizes: train={len(X_train)}, val={len(X_val)}")

train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

base_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.1,
    "num_leaves": 31,
    "max_depth": -1,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 0,
    "feature_fraction_seed": 0,
    "bagging_seed": 0,
    "force_col_wise": True,
}

params = dict(base_params)
params["device"] = "gpu"

num_boost_round = 2000

try:
    bst = lgb.train(
        params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[val_data],
    )
except Exception as e:
    print("GPU training failed, falling back to CPU. Error was:", repr(e))
    params = dict(base_params)
    params["device"] = "cpu"
    bst = lgb.train(
        params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[val_data],
    )

y_pred = bst.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"rmse:{rmse}")

model_name = "lgbm"



## === cell 4
test_pd = test.to_pandas()
test_pd = test_pd.reindex(columns=X.columns, fill_value=0)

for c in test_pd.columns:
    if test_pd[c].dtype == "object":
        test_pd[c] = pd.to_numeric(test_pd[c], errors="coerce")
test_pd = test_pd.fillna(0)

sub_pred = bst.predict(test_pd)
sub_pred = np.clip(sub_pred, 0.0, None)

submission = pd.DataFrame({"key": test_key, "fare_amount": sub_pred})

assert (
    submission.shape[0] == test_raw.height
), f"Submission rows ({submission.shape[0]}) != test rows ({test_raw.height})."
assert submission["key"].isnull().sum() == 0, "Null keys in submission."

out1 = f"submission_{model_name}.csv"
out2 = "submission.csv"
submission.to_csv(out1, index=False)
submission.to_csv(out2, index=False)

print("Wrote:", out1, "and", out2, "shape:", submission.shape)
print(submission.head())



## === cell 5
import matplotlib.pyplot as plt
import pandas as pd

feature_importance_df = pd.DataFrame()
feature_importance_df["feature"] = X.columns
feature_importance_df["importance"] = bst.feature_importance()
feature_importance_df = feature_importance_df.sort_values(
    by="importance", ascending=False
).head(15)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df["feature"], feature_importance_df["importance"])
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Feature Importance")
plt.gca().invert_yaxis()
plt.show()
