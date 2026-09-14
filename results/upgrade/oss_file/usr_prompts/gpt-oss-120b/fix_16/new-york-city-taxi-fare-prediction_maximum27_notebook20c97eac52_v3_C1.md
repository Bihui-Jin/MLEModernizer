# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import polars as pl
import numpy as np
import pandas as pd
from pathlib import Path


def resolve_path(relative_path: str) -> Path:
    """Return the first existing path among several common locations, including the absolute Kaggle data folder."""
    candidates = [
        Path("data") / relative_path,
        Path("data") / "new-york-city-taxi-fare-prediction" / relative_path,
        Path(relative_path),
        Path("/kaggle/data") / relative_path,  # absolute location on the Kaggle VM
        Path("/kaggle/input") / "new-york-city-taxi-fare-prediction" / relative_path,
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(f"Could not find {relative_path} in any expected location.")


train_path = resolve_path("train.csv")
test_path = resolve_path("test.csv")

train = pl.read_csv(train_path, infer_schema_length=1000)
test = pl.read_csv(test_path, infer_schema_length=1000)




## === cell 1
def build_features(df: pl.LazyFrame) -> pl.DataFrame:
    """Create all features in a single pass to minimise data copies."""
    R = 6371.0  # km
    deg_to_rad = np.pi / 180.0

    left_latitude = 40.737676
    right_latitude = 40.791716
    left_longitude = -74.015781
    right_longitude = -73.945521

    df = df.with_columns(
        [
            pl.col("pickup_datetime")
            .str.strptime(pl.Datetime, fmt="%Y-%m-%d %H:%M:%S", strict=False)
            .alias("pickup_dt"),
            pl.col("pickup_dt").dt.year().alias("pickup_year"),
            pl.col("pickup_dt").dt.month().alias("pickup_month"),
            pl.col("pickup_dt").dt.day().alias("pickup_day"),
            pl.col("pickup_dt").dt.hour().alias("pickup_hour"),
            pl.col("pickup_dt").dt.minute().alias("pickup_minute"),
            pl.col("pickup_dt").dt.second().alias("pickup_second"),
            pl.col("pickup_dt").dt.weekday().alias("pickup_weekday"),
            (
                (pl.col("pickup_longitude") - pl.col("dropoff_longitude")) ** 2
                + (pl.col("pickup_latitude") - pl.col("dropoff_latitude")) ** 2
            )
            .sqrt()
            .alias("distance"),
            (
                2
                * pl.arcsin(
                    pl.sqrt(
                        (
                            pl.sin(
                                (
                                    (
                                        pl.col("dropoff_latitude")
                                        - pl.col("pickup_latitude")
                                    )
                                    * deg_to_rad
                                    / 2
                                )
                            )
                            ** 2
                        )
                        + pl.cos(pl.col("pickup_latitude") * deg_to_rad)
                        * pl.cos(pl.col("dropoff_latitude") * deg_to_rad)
                        * (
                            pl.sin(
                                (
                                    (
                                        pl.col("dropoff_longitude")
                                        - pl.col("pickup_longitude")
                                    )
                                    * deg_to_rad
                                    / 2
                                )
                            )
                            ** 2
                        )
                    )
                )
                * R
            ).alias("haversine_distance"),
            (pl.col("pickup_month") * 2 * np.pi / 12).sin().alias("pickup_month_sin"),
            (pl.col("pickup_month") * 2 * np.pi / 12).cos().alias("pickup_month_cos"),
            (pl.col("pickup_hour") * 2 * np.pi / 24).sin().alias("pickup_hour_sin"),
            (pl.col("pickup_hour") * 2 * np.pi / 24).cos().alias("pickup_hour_cos"),
            (
                (pl.col("pickup_latitude") >= left_latitude)
                & (pl.col("pickup_latitude") <= right_latitude)
                & (pl.col("pickup_longitude") >= left_longitude)
                & (pl.col("pickup_longitude") <= right_longitude)
            )
            .cast(pl.Int8)
            .alias("is_pickup_central"),
            (
                (pl.col("dropoff_latitude") >= left_latitude)
                & (pl.col("dropoff_latitude") <= right_latitude)
                & (pl.col("dropoff_longitude") >= left_longitude)
                & (pl.col("dropoff_longitude") <= right_longitude)
            )
            .cast(pl.Int8)
            .alias("is_dropoff_central"),
        ]
    )

    filters = [
        (pl.col("pickup_latitude") >= 40) & (pl.col("pickup_latitude") < 41),
        (pl.col("pickup_longitude") >= -74) & (pl.col("pickup_longitude") < -73),
        (pl.col("dropoff_longitude") >= -74) & (pl.col("dropoff_longitude") < -73),
        (pl.col("dropoff_latitude") >= 40) & (pl.col("dropoff_latitude") < 41),
        (pl.col("passenger_count") >= 1) & (pl.col("passenger_count") < 6),
    ]
    if "fare_amount" in df.columns:
        filters.append(pl.col("fare_amount") < 10000)
    combined = filters[0]
    for f in filters[1:]:
        combined &= f
    df = df.filter(combined)

    df = df.with_columns(
        (pl.col("distance") < 0.32).cast(pl.Int8).alias("is_short_distance")
    )

    df = df.to_dummies(["pickup_year", "pickup_day"])

    df = df.drop(
        ["key", "pickup_datetime", "pickup_minute", "pickup_second", "pickup_dt"]
    )

    return df.collect()




## === cell 2
test_key = test["key"].to_pandas()

train = build_features(train.lazy())
test = build_features(test.lazy())



## === cell 3
float64_cols_train = [
    c for c, dt in zip(train.columns, train.dtypes) if dt == pl.Float64
]
train = train.with_columns([pl.col(c).cast(pl.Float32) for c in float64_cols_train])

float64_cols_test = [c for c, dt in zip(test.columns, test.dtypes) if dt == pl.Float64]
test = test.with_columns([pl.col(c).cast(pl.Float32) for c in float64_cols_test])



## === cell 4
import warnings

warnings.simplefilter("ignore")
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

numeric_cols = train.select(
    pl.col(pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64, pl.Boolean)
).columns
numeric_cols = [c for c in numeric_cols if c != "fare_amount"]

X_np = train.select(numeric_cols).to_numpy()
y_original = train["fare_amount"].to_numpy()
y = np.log1p(y_original)  # log‑transform target for better stability

X_train, X_val, y_train, y_val, y_orig_train, y_orig_val = train_test_split(
    X_np, y, y_original, test_size=0.2, random_state=0
)

train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

params = {
    "objective": "regression",
    "metric": "rmse",
    "device": "cpu",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": -1,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}

bst = lgb.train(
    params,
    train_data,
    num_boost_round=1500,
    valid_sets=[valid_data],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
)

y_pred_log = bst.predict(X_val, num_iteration=bst.best_iteration)
y_pred = np.expm1(y_pred_log)

rmse = np.sqrt(mean_squared_error(y_orig_val, y_pred))
print(f"rmse:{rmse}")

model_name = "lgbm_log_hav"



## === cell 5
test_features_np = test.select(numeric_cols).to_numpy()
sub_pred_log = bst.predict(test_features_np)
sub_pred = np.expm1(sub_pred_log)  # revert log transform

submission = pd.DataFrame({"key": test_key, "fare_amount": sub_pred})
submission_path = f"submission_{model_name}.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")



## === cell 6
import matplotlib.pyplot as plt

feature_importance_df = pd.DataFrame(
    {"feature": numeric_cols, "importance": bst.feature_importance()}
)
feature_importance_df = feature_importance_df.sort_values(
    by="importance", ascending=False
).head(15)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df["feature"], feature_importance_df["importance"])
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Top 15 Feature Importance")
plt.gca().invert_yaxis()
plt.show()
