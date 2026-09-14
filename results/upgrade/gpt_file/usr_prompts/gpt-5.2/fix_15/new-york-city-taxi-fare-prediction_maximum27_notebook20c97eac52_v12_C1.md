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

3.73427

# 6. Current score

4.61459

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.93258) has done: 'I fix two correctness issues that are likely inflating your RMSE: (1) your datetime parsing format doesn’t match the dataset (no `" UTC"` suffix), which can silently create nulls and degrade features, and (2) your train/test one-hot encoding is done independently, causing column mismatches (missing/extra dummy columns) between train and test at prediction time. I keep your feature engineering and LightGBM setup the same, but I align dummy columns using train’s schema, fill missing columns with 0, and ensure `bst.predict()` receives a pandas DataFrame with identical columns to training. These are minimal changes that usually produce a meaningful RMSE improvement without changing the modeling approach or training loop. The script still run end-to-end and write a valid submission CSV.'
- What this solution (achieved 5.70416) has done: 'I fix the crash where training becomes empty by correcting the datetime parsing format to match the dataset (it includes microseconds) and by making the outlier filters robust to boundary cases (inclusive ranges and proper fare upper bound), while keeping your feature engineering and LightGBM training logic intact. I also fix the `NameError` for `test_key_all` by defining it outside the preprocessing cell so it always exists at submission time. Finally, I ensure train/test one-hot columns stay aligned exactly as you already intended, so the model receives identical feature columns at inference and a valid `submission.csv` is always written.'
- What this solution (achieved 5.63331) has done: 'I fix the Polars haversine distance computation by replacing the unsupported `Expr.arctan2` call with `pl.arctan2(y, x)`, which unblocks preprocessing and keeps the same feature logic. I also guarantee `test_key_all` is always defined (even if preprocessing fails earlier) by capturing it immediately after reading `test.csv`, eliminating the `NameError` at submission time. Finally, I keep your train/test dummy alignment logic intact but make it deterministic by using the trained feature column order at inference, ensuring a valid `submission.csv` is written end-to-end.'
- What this solution (achieved 6.82106) has done: 'Your current RMSE (5.633) is far above the target (3.734), so we should improve generalization with the smallest change that doesn’t alter your feature engineering or LightGBM training loop semantics. The biggest remaining score drag is that you apply strict NYC bounding-box filters only to train but only “clip” test, creating a train/test distribution mismatch; I apply the same cleaning logic to test in a non-destructive way (filter only truly invalid rows and then reinsert predictions for dropped rows using a safe fallback). I also align datetime null-handling consistently (drop null datetimes in train and set deterministic defaults in test) and add a simple non-negative post-processing for predictions (fares can’t be negative), which usually reduces RMSE. All changes keep your architecture, features, and training procedure intact and still write a valid `submission.csv`.'
- What this solution (achieved 4.29911) has done: 'Your current RMSE (6.821) is far worse than the target (3.734), so we should make small, low-risk changes that reduce error without changing your overall LightGBM approach or feature set. The biggest remaining drag is that you’re throwing away far-away trips by filtering to a tight NYC box; that can badly hurt because the test set contains many rides outside that box, so I keep your filter logic but widen it to the standard competition bounds (NYC metro area) for both train and test. I also add one minimal, metric-aligned feature (`distance_sq`) derived from your existing haversine `distance` to help LightGBM fit the nonlinearity of fare vs distance without changing the model or training loop. Everything else (sampling, preprocessing steps, LightGBM training) stays the same, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 4.28783) has done: 'Your current RMSE (4.29911) is worse than the target (3.73427), so we should make the smallest, low-risk change that improves generalization without altering your LightGBM setup or feature logic. The biggest likely drag now is that you sample uniformly from the full training set, which over-represents noisy/outlier trips and under-represents the core NYC distribution; we instead sample *after applying your existing outlier filters*, keeping the same preprocessing/features/model. This preserves your core pipeline but gives the model a cleaner, more on-distribution 10M sample, which typically reduces RMSE. Everything else (feature engineering, one-hot alignment, training loop, submission construction) stays the same and still writes `submission.csv`.'
- What this solution (achieved 4.61459) has done: 'Your current RMSE (4.28783) is worse than the target (3.73427), so we should make a small change that improves accuracy without changing your feature set or LightGBM training loop. The most leverage here is aligning the training objective with the competition: instead of fitting to raw fares (which are heavy‑tailed), train LightGBM on `log1p(fare_amount)` and then invert with `expm1` for predictions; this typically reduces RMSE materially on this competition while keeping the same model and preprocessing pipeline. I keep all your preprocessing, sampling, one-hot alignment, and submission construction unchanged, and only change the target transform plus consistent inverse transform and fallback definition. This should move your score closer to the target band with minimal risk and still produces a valid `submission.csv`.'
- What this solution (achieved 4.61459) has done: 'Your RMSE (4.61459) is worse than the target (3.73427), so we should make a small, low-risk change that improves accuracy without changing your feature engineering or LightGBM training loop. The biggest remaining issue is a metric mismatch introduced by training on `log1p(fare)` while LightGBM still optimizes RMSE in log-space; we keep your log target but switch the LightGBM objective/metric to an L2 loss in log-space (same semantics) and use early-stopping-style “best_iteration” selection without altering the loop length by enabling `first_metric_only` and adding the training set to valid_sets to stabilize best-iteration selection. Additionally, we make the test fallback consistent with your log-target training by using `expm1(median(log1p(y)))` rather than `median(raw_fare)`, which reduces error for any filtered-out test rows without changing the model. These are minimal changes that typically reduce public RMSE for this competition while keeping the rest of your pipeline intact and still producing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import polars as pl

train_df = pl.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/train.csv")
test_df = pl.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")

test_key_all = test_df["key"]

num_rows = train_df.height
sample_size = min(10_000_000, num_rows)
rng = np.random.default_rng(0)
random_indices = rng.choice(num_rows, size=sample_size, replace=False)
train_df = train_df[random_indices]
train_df = train_df.drop_nulls()



## === cell 2
train = train_df
test = test_df



## === cell 3
import numpy as np
import polars as pl


def preprocess(df: pl.DataFrame) -> pl.DataFrame:
    dt_clean = pl.col("pickup_datetime").str.replace(r"\s+UTC$", "", literal=False)
    df = df.with_columns(
        dt_clean.str.strptime(
            pl.Datetime, format="%Y-%m-%d %H:%M:%S%.f", strict=False
        ).alias("pickup_datetime")
    )

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
    p_lat = pl.col("pickup_latitude") * np.pi / 180.0
    p_lon = pl.col("pickup_longitude") * np.pi / 180.0
    d_lat = pl.col("dropoff_latitude") * np.pi / 180.0
    d_lon = pl.col("dropoff_longitude") * np.pi / 180.0

    dphi = d_lat - p_lat
    dlambda = d_lon - p_lon

    a = (dphi / 2.0).sin() ** 2 + p_lat.cos() * d_lat.cos() * (dlambda / 2.0).sin() ** 2
    c = 2.0 * pl.arctan2(a.sqrt(), (1.0 - a).sqrt())
    R = 6371.0  # km

    return df.with_columns((R * c).alias("distance"))


def drop_encoding(df: pl.DataFrame) -> pl.DataFrame:
    drop_columns = ["key", "pickup_datetime", "pickup_minute", "pickup_second"]
    existing = [c for c in drop_columns if c in df.columns]
    return df.drop(existing)


def cycling_encoding(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        [
            (pl.col("pickup_month") * 2 * np.pi / 12).sin().alias("pickup_month_sin"),
            (pl.col("pickup_month") * 2 * np.pi / 12).cos().alias("pickup_month_cos"),
            (pl.col("pickup_hour") * 2 * np.pi / 24).sin().alias("pickup_hour_sin"),
            (pl.col("pickup_hour") * 2 * np.pi / 24).cos().alias("pickup_hour_cos"),
        ]
    )


def one_hot_encoding(df: pl.DataFrame) -> pl.DataFrame:
    one_hot_cols = ["pickup_year", "pickup_day"]
    existing = [c for c in one_hot_cols if c in df.columns]
    return df.to_dummies(existing)


def max_min_scaling(df: pl.DataFrame) -> pl.DataFrame:
    df = df.filter(
        (pl.col("pickup_latitude") >= 40) & (pl.col("pickup_latitude") <= 42)
    )
    df = df.filter(
        (pl.col("pickup_longitude") >= -75) & (pl.col("pickup_longitude") <= -72)
    )
    df = df.filter(
        (pl.col("dropoff_latitude") >= 40) & (pl.col("dropoff_latitude") <= 42)
    )
    df = df.filter(
        (pl.col("dropoff_longitude") >= -75) & (pl.col("dropoff_longitude") <= -72)
    )
    return df


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


def add_center_distance(df: pl.DataFrame) -> pl.DataFrame:
    center_lat = 40.7580
    center_lon = -73.9855

    p_lat = pl.col("pickup_latitude") * np.pi / 180.0
    p_lon = pl.col("pickup_longitude") * np.pi / 180.0
    d_lat = pl.col("dropoff_latitude") * np.pi / 180.0
    d_lon = pl.col("dropoff_longitude") * np.pi / 180.0

    c_lat = pl.lit(center_lat) * np.pi / 180.0
    c_lon = pl.lit(center_lon) * np.pi / 180.0

    def hav_expr(lat1, lon1, lat2, lon2):
        dphi = lat2 - lat1
        dlambda = lon2 - lon1
        a = (dphi / 2.0).sin() ** 2 + lat1.cos() * lat2.cos() * (
            dlambda / 2.0
        ).sin() ** 2
        c = 2.0 * pl.arctan2(a.sqrt(), (1.0 - a).sqrt())
        return 6371.0 * c

    return df.with_columns(
        [
            hav_expr(p_lat, p_lon, c_lat, c_lon).alias("pickup_to_center_km"),
            hav_expr(d_lat, d_lon, c_lat, c_lon).alias("dropoff_to_center_km"),
        ]
    )


def add_distance_poly(df: pl.DataFrame) -> pl.DataFrame:
    if "distance" in df.columns:
        return df.with_columns((pl.col("distance") ** 2).alias("distance_sq"))
    return df


def drop_outliner_train_only(df: pl.DataFrame) -> pl.DataFrame:
    filtered_train_df = df.filter(
        (pl.col("pickup_latitude") >= 40) & (pl.col("pickup_latitude") <= 42)
    )
    filtered_train_df = filtered_train_df.filter(
        (pl.col("pickup_longitude") >= -75) & (pl.col("pickup_longitude") <= -72)
    )
    filtered_train_df = filtered_train_df.filter(
        (pl.col("dropoff_longitude") >= -75) & (pl.col("dropoff_longitude") <= -72)
    )
    filtered_train_df = filtered_train_df.filter(
        (pl.col("dropoff_latitude") >= 40) & (pl.col("dropoff_latitude") <= 42)
    )
    filtered_train_df = filtered_train_df.filter(
        (pl.col("passenger_count") >= 1) & (pl.col("passenger_count") <= 6)
    )
    filtered_train_df = filtered_train_df.filter(
        (pl.col("fare_amount") > 0) & (pl.col("fare_amount") < 500)
    )
    return filtered_train_df


def filter_like_train_for_test(df: pl.DataFrame) -> pl.DataFrame:
    df2 = df.filter(
        (pl.col("pickup_latitude") >= 40) & (pl.col("pickup_latitude") <= 42)
    )
    df2 = df2.filter(
        (pl.col("pickup_longitude") >= -75) & (pl.col("pickup_longitude") <= -72)
    )
    df2 = df2.filter(
        (pl.col("dropoff_longitude") >= -75) & (pl.col("dropoff_longitude") <= -72)
    )
    df2 = df2.filter(
        (pl.col("dropoff_latitude") >= 40) & (pl.col("dropoff_latitude") <= 42)
    )
    df2 = df2.filter(
        (pl.col("passenger_count") >= 1) & (pl.col("passenger_count") <= 6)
    )
    return df2


def build_train_with_retries(
    full_train_path: str,
    seed: int = 0,
    initial_sample: int = 10_000_000,
    min_rows_required: int = 200_000,
    max_retries: int = 3,
) -> pl.DataFrame:
    base = pl.read_csv(full_train_path).drop_nulls()
    base = preprocess(base).drop_nulls(["pickup_datetime"])
    base = drop_outliner_train_only(base)

    n = base.height
    if n == 0:
        raise RuntimeError(
            "No rows left in training data after basic cleaning filters."
        )

    rng_local = np.random.default_rng(seed)

    last_df = None
    for attempt in range(max_retries + 1):
        factor = 1 if attempt == 0 else (2**attempt)
        sample_n = min(n, initial_sample * factor)
        idx = rng_local.choice(n, size=sample_n, replace=False)
        df = base[idx]

        df = max_min_scaling(df)

        df = distance(df)
        df = add_center_distance(df)
        df = add_distance_poly(df)

        df = is_central(df)
        df = drop_encoding(df)
        df = cycling_encoding(df)
        df = one_hot_encoding(df)

        last_df = df
        if df.height >= min_rows_required:
            return df

    if last_df is None or last_df.height == 0:
        raise RuntimeError(
            "Training dataframe is empty after preprocessing/filtering even after retries."
        )
    return last_df


train = build_train_with_retries(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
    seed=0,
    initial_sample=min(
        10_000_000, train.height if isinstance(train, pl.DataFrame) else 10_000_000
    ),
    min_rows_required=200_000,
    max_retries=3,
)

test_raw = test  # original test dataframe (all rows)

test_proc = preprocess(test_raw)

datetime_derived = [
    "pickup_year",
    "pickup_month",
    "pickup_day",
    "pickup_hour",
    "pickup_minute",
    "pickup_second",
    "pickup_weekday",
]
test_proc = test_proc.with_columns([pl.col(c).fill_null(0) for c in datetime_derived])

test_proc = test_proc.with_row_index("row_id")

test_proc_in = filter_like_train_for_test(test_proc)

kept_row_ids = test_proc_in["row_id"]

test_proc_in = distance(test_proc_in)
test_proc_in = add_center_distance(test_proc_in)
test_proc_in = add_distance_poly(test_proc_in)

test_proc_in = is_central(test_proc_in)
test_proc_in = drop_encoding(test_proc_in)
test_proc_in = cycling_encoding(test_proc_in)
test_proc_in = one_hot_encoding(test_proc_in)

test = test_proc_in

print("Train rows after preprocessing:", train.height)
print("Test rows kept after train-like filtering:", test.height, "/", test_raw.height)



## === cell 4
import polars as pld

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



## === cell 5
import warnings

warnings.simplefilter("ignore")

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if train.height == 0:
    raise RuntimeError(
        "Training dataframe is empty after preprocessing/filtering. Cannot train model."
    )

X_pl = train.drop(["fare_amount"])
for bad in ["key", "pickup_datetime"]:
    if bad in X_pl.columns:
        X_pl = X_pl.drop(bad)

X = X_pl.to_pandas()

y_raw = train["fare_amount"].to_pandas().astype(np.float32)

y = np.log1p(y_raw)

for c in X.columns:
    if X[c].dtype == "bool":
        X[c] = X[c].astype("int8")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

params = {
    "objective": "regression_l2",
    "metric": "rmse",
    "device": "gpu",
    "boosting_type": "gbdt",
    "learning_rate": 0.1,
    "num_leaves": 31,
    "max_depth": -1,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "first_metric_only": True,
}

bst = lgb.train(
    params,
    train_data,
    num_boost_round=100,
    valid_sets=[train_data, val_data],
    valid_names=["train", "valid"],
)

y_pred_log = bst.predict(X_val, num_iteration=bst.best_iteration)
y_pred = np.expm1(y_pred_log)
y_val_fare = np.expm1(y_val)

rmse = np.sqrt(mean_squared_error(y_val_fare, y_pred))
print(f"rmse:{rmse}")
model_name = "lgbm"

test_pl = test
for bad in ["key", "pickup_datetime"]:
    if bad in test_pl.columns:
        test_pl = test_pl.drop(bad)

test_pd = test_pl.to_pandas()
for c in test_pd.columns:
    if test_pd[c].dtype == "bool":
        test_pd[c] = test_pd[c].astype("int8")

missing_cols = [c for c in X.columns if c not in test_pd.columns]
for c in missing_cols:
    test_pd[c] = 0
extra_cols = [c for c in test_pd.columns if c not in X.columns]
if extra_cols:
    test_pd = test_pd.drop(columns=extra_cols)

test_pd = test_pd[X.columns]



## === cell 6
sub_pred_kept_log = bst.predict(test_pd, num_iteration=bst.best_iteration)
sub_pred_kept = np.expm1(sub_pred_kept_log)

sub_pred_kept = np.clip(sub_pred_kept, 0.0, None)

exp_num = "is_central"

fallback = float(np.expm1(np.median(np.log1p(y_raw))))
full_pred = np.full(shape=(test_raw.height,), fill_value=fallback, dtype=np.float32)

kept_idx = kept_row_ids.to_numpy()
full_pred[kept_idx] = sub_pred_kept.astype(np.float32)

submission = pd.DataFrame({"key": test_key_all.to_pandas(), "fare_amount": full_pred})

submission = submission[["key", "fare_amount"]]
submission.to_csv(f"submission_{model_name}_{exp_num}.csv", index=False)
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Rows in submission:", len(submission))
print("Wrote:", f"submission_{model_name}_{exp_num}.csv", "and submission.csv")



## === cell 7
import matplotlib.pyplot as plt

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
