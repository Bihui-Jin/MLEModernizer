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

3.82404

# 6. Current score

5.15585

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.97435) has done: 'I fix the Polars concat width mismatch by ensuring train/test have identical columns before concatenation (the mismatch comes from `fare_amount` only existing in train), and I keep `key`/`pickup_datetime` out of the LightGBM feature matrix to resolve the “bad pandas dtypes” error. I also make the LightGBM GPU setting robust by falling back to CPU automatically if GPU training fails in this environment, so the pipeline always completes. Finally, I ensure the submission is written as a proper `.csv` with exactly the required columns (`key`, `fare_amount`) and aligned row order.'
- What this solution (achieved 5.95627) has done: 'Your current RMSE (5.97) is much worse than the target (3.82), so we should legitimately improve generalization with minimal, low-risk changes that don’t alter the model family or training loop. The biggest win here is fixing train/test feature misalignment caused by doing one-hot encoding on the concatenated data: it leaks test categories into training and also creates a huge sparse feature space; instead we one-hot encode **train and test separately** and then align columns. We also remove the artificial `fare_amount=None` column from test before concatenation (it can create unnecessary missingness/feature noise), and we add a standard NYC Taxi cleanup on `fare_amount` (positive and reasonable upper bound) to reduce label noise; this stays within your existing outlier filtering intent. Everything else (LightGBM regressor, objective/metric, 100 boosting rounds, split strategy) remains the same.'
- What this solution (achieved 5.81901) has done: 'Your RMSE (5.956) is worse than the target (3.824), so we should make a small, legitimate generalization improvement without changing the model family or training loop. The biggest low-risk fix is to replace the current “linearized distance” with a proper Haversine distance feature (still a single scalar distance, same role in the pipeline), which is known to materially improve NYC Taxi Fare performance. We keep all your other feature engineering, one-hot/column alignment, and LightGBM settings intact, and we also clip negative predictions to 0 (fares can’t be negative) to reduce error on edge cases. These changes are minimal, metric-aligned, and should move RMSE down toward your target.'
- What this solution (achieved 5.65314) has done: 'We need to move RMSE down from 5.819 toward 3.824 (lower is better), so we make small, legitimate generalization improvements without changing the model family (LightGBM GBDT), training loop style, or overall feature approach. The main issue is that your random row sampling happens before outlier filtering, so after filtering you may end up with a much smaller/easier-to-distort effective training set; we instead apply the same preprocessing+outlier filter first and only then subsample to 10M to keep data quality high. We also add two very standard, low-risk location features (absolute delta and manhattan distance) built from your already-present coordinate deltas to help the same model fit better without changing architecture. Finally, we keep submission formatting identical but also ensure test rows with bad datetimes don’t introduce null-derived features by filling the derived datetime parts with 0.'
- What this solution (achieved 5.15585) has done: 'Your RMSE (5.653) is still far from the target (3.824), so we should make a small, legitimate generalization improvement without changing the model family or training loop. The most impactful minimal fix is to add a standard “airport distance” feature (distance to JFK/LGA/EWR) computed from your existing lat/lon columns; this keeps the same feature-engineering style (just extra columns) and often reduces RMSE meaningfully on this dataset. We also remove the forced Float32 downcast (keeping Float64) to avoid losing small but important geospatial precision; this doesn’t change the approach, just numeric fidelity. Everything else (filters, sampling-after-filtering, LightGBM training with 100 rounds, submission format) stays the same.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import polars as pl

train_df = pl.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/train.csv")
test_df = pl.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")



## === cell 2
train = train_df
test = test_df



## === cell 3
import numpy as np
import polars as pl


def preprocess(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.col("pickup_datetime")
        .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
        .alias("pickup_datetime")
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

    dt_cols = [
        "pickup_year",
        "pickup_month",
        "pickup_day",
        "pickup_hour",
        "pickup_minute",
        "pickup_second",
        "pickup_weekday",
    ]
    df = df.with_columns([pl.col(c).fill_null(0).cast(pl.Int32) for c in dt_cols])

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

    df = df.with_columns(
        (pl.col("abs_longitude") + pl.col("abs_latitude")).alias("manhattan_dist")
    )

    return df


def distance(df: pl.DataFrame) -> pl.DataFrame:
    r_earth_km = 6371.0
    df = df.with_columns(
        [
            (pl.col("pickup_latitude") * (np.pi / 180.0)).alias("pickup_lat_rad"),
            (pl.col("dropoff_latitude") * (np.pi / 180.0)).alias("dropoff_lat_rad"),
            (pl.col("pickup_longitude") * (np.pi / 180.0)).alias("pickup_lon_rad"),
            (pl.col("dropoff_longitude") * (np.pi / 180.0)).alias("dropoff_lon_rad"),
        ]
    )

    df = df.with_columns(
        [
            (pl.col("dropoff_lat_rad") - pl.col("pickup_lat_rad")).alias("dlat"),
            (pl.col("dropoff_lon_rad") - pl.col("pickup_lon_rad")).alias("dlon"),
        ]
    )

    df = df.with_columns(
        [
            (
                (pl.col("dlat") / 2.0).sin().pow(2)
                + pl.col("pickup_lat_rad").cos()
                * pl.col("dropoff_lat_rad").cos()
                * (pl.col("dlon") / 2.0).sin().pow(2)
            ).alias("a")
        ]
    )

    df = df.with_columns(
        [(2.0 * pl.col("a").sqrt().arcsin() * r_earth_km).alias("distance")]
    )

    df = df.drop(
        [
            "pickup_lat_rad",
            "dropoff_lat_rad",
            "pickup_lon_rad",
            "dropoff_lon_rad",
            "dlat",
            "dlon",
            "a",
        ]
    )
    return df


def airport_features(df: pl.DataFrame) -> pl.DataFrame:
    r_earth_km = 6371.0

    def haversine_expr(lat_col: str, lon_col: str, lat0: float, lon0: float) -> pl.Expr:
        lat1 = pl.col(lat_col) * (np.pi / 180.0)
        lon1 = pl.col(lon_col) * (np.pi / 180.0)
        lat2 = pl.lit(lat0) * (np.pi / 180.0)
        lon2 = pl.lit(lon0) * (np.pi / 180.0)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (dlat / 2.0).sin().pow(2) + lat1.cos() * lat2.cos() * (
            dlon / 2.0
        ).sin().pow(2)
        return 2.0 * a.sqrt().arcsin() * r_earth_km

    airports = {
        "jfk": (40.6413, -73.7781),
        "lga": (40.7769, -73.8740),
        "ewr": (40.6895, -74.1745),
    }

    exprs = []
    for code, (alat, alon) in airports.items():
        exprs.append(
            haversine_expr("pickup_latitude", "pickup_longitude", alat, alon).alias(
                f"pickup_dist_{code}"
            )
        )
        exprs.append(
            haversine_expr("dropoff_latitude", "dropoff_longitude", alat, alon).alias(
                f"dropoff_dist_{code}"
            )
        )

    df = df.with_columns(exprs)

    df = df.with_columns(
        [
            pl.min_horizontal(
                [
                    pl.col("pickup_dist_jfk"),
                    pl.col("pickup_dist_lga"),
                    pl.col("pickup_dist_ewr"),
                ]
            ).alias("pickup_dist_nearest_airport"),
            pl.min_horizontal(
                [
                    pl.col("dropoff_dist_jfk"),
                    pl.col("dropoff_dist_lga"),
                    pl.col("dropoff_dist_ewr"),
                ]
            ).alias("dropoff_dist_nearest_airport"),
        ]
    )
    return df


def drop_encoding(df: pl.DataFrame) -> pl.DataFrame:
    drop_columns = ["key", "pickup_datetime", "pickup_minute", "pickup_second"]
    existing = [c for c in drop_columns if c in df.columns]
    df = df.drop(existing)
    return df


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
    df = df.to_dummies(one_hot_cols)
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


def drop_outliner(df: pl.DataFrame) -> pl.DataFrame:
    filtered_train_df = df.filter(
        (pl.col("pickup_latitude") >= 40) & (pl.col("pickup_latitude") < 41)
    )
    filtered_train_df = filtered_train_df.filter(
        (pl.col("pickup_longitude") >= -74) & (pl.col("pickup_longitude") < -73)
    )
    filtered_train_df = filtered_train_df.filter(
        (pl.col("dropoff_longitude") >= -74) & (pl.col("dropoff_longitude") < -73)
    )
    filtered_train_df = filtered_train_df.filter(
        (pl.col("dropoff_latitude") >= 40) & (pl.col("dropoff_latitude") < 41)
    )
    filtered_train_df = filtered_train_df.filter(
        (pl.col("passenger_count") >= 1) & (pl.col("passenger_count") < 6)
    )
    if "fare_amount" in filtered_train_df.columns:
        filtered_train_df = filtered_train_df.filter(
            pl.col("fare_amount").is_not_null()
        )
        filtered_train_df = filtered_train_df.filter(pl.col("fare_amount") > 0)
        filtered_train_df = filtered_train_df.filter(pl.col("fare_amount") < 500)
    return filtered_train_df


train_feat = preprocess(train)
train_feat = distance(train_feat)
train_feat = airport_features(train_feat)
train_feat = drop_encoding(train_feat)
train_feat = cycling_encoding(train_feat)
train_feat = is_central(train_feat)
train_feat = drop_outliner(train_feat)
train_feat = train_feat.drop_nulls()

np.random.seed(0)
num_rows = train_feat.height
sample_n = min(10_000_000, num_rows)
if sample_n < num_rows:
    random_indices = np.random.choice(num_rows, size=sample_n, replace=False)
    train_feat = train_feat[random_indices]

test_feat = preprocess(test)
test_feat = distance(test_feat)
test_feat = airport_features(test_feat)
test_key = test_feat["key"]
test_feat = drop_encoding(test_feat)
test_feat = cycling_encoding(test_feat)
test_feat = is_central(test_feat)

train_feat = one_hot_encoding(train_feat)
test_feat = one_hot_encoding(test_feat)

target_col = "fare_amount"
train_cols = set(train_feat.columns)
test_cols = set(test_feat.columns)

for c in sorted(train_cols - test_cols):
    if c != target_col:
        test_feat = test_feat.with_columns(pl.lit(0).cast(pl.Int8).alias(c))

for c in sorted(test_cols - train_cols):
    if c != target_col:
        train_feat = train_feat.with_columns(pl.lit(0).cast(pl.Int8).alias(c))

feature_cols = sorted([c for c in train_feat.columns if c != target_col])
train = train_feat.select(feature_cols + [target_col])
test = test_feat.select(feature_cols)



## === cell 4
import polars as pld




## === cell 5
import warnings

warnings.simplefilter("ignore")

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

target_col = "fare_amount"
drop_cols = [c for c in [target_col, "key", "pickup_datetime"] if c in train.columns]

X = train.drop(drop_cols).to_pandas()
y = train[target_col].to_pandas()

for col in X.columns:
    if X[col].dtype == "object":
        X[col] = pd.to_numeric(X[col], errors="coerce")
X = X.fillna(0)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

params = {
    "objective": "regression",
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
}

try:
    bst = lgb.train(params, train_data, num_boost_round=100, valid_sets=[val_data])
except Exception as e:
    print("LightGBM GPU training failed, falling back to CPU. Error was:", repr(e))
    params_cpu = dict(params)
    params_cpu["device"] = "cpu"
    bst = lgb.train(params_cpu, train_data, num_boost_round=100, valid_sets=[val_data])

y_pred = bst.predict(X_val, num_iteration=bst.best_iteration)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"rmse:{rmse}")
model_name = "lgbm"



## === cell 6
X_test = test.to_pandas()
for col in X_test.columns:
    if X_test[col].dtype == "object":
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")
X_test = X_test.fillna(0)

X_test = X_test.reindex(columns=X.columns, fill_value=0)

sub_pred = bst.predict(X_test, num_iteration=bst.best_iteration)

sub_pred = np.clip(sub_pred, 0, None)

exp_num = "distance_airports"
submission = pd.DataFrame({"key": test_key.to_pandas(), "fare_amount": sub_pred})

submission.to_csv(f"submission_{model_name}_{exp_num}.csv", index=False)
print(submission.head())
print("Wrote:", f"submission_{model_name}_{exp_num}.csv", "rows:", len(submission))



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
