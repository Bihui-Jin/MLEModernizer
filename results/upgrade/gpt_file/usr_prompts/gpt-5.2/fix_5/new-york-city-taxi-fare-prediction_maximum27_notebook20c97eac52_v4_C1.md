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

3.83762

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 5.90559) has done: 'I fix the LightGBM prediction crash by ensuring the test feature matrix has exactly the same columns (names and order) as the training feature matrix used to build the Dataset. This is caused by train/test getting slightly different dummy/nullable columns after preprocessing, so converting to pandas separately can yield a shape mismatch (20 vs 21). I add a minimal column-alignment step that reindexes test to train’s columns and fills any missing columns with zeros, which is score-neutral but makes the pipeline run end-to-end and write a valid `.csv` submission. I also set a random seed for reproducibility without changing the core modeling approach.'
- What this solution (achieved 5.64689) has done: 'Your current score (5.90559 RMSE) is worse than the target (3.83762), so we should make small, legitimate changes that typically improve generalization without changing the overall approach (still LightGBM regression on the same engineered features). The biggest issue is that you train on a random 10M sample but don’t do any label/coordinate sanity filtering beyond broad bounds, so noisy/outlier fares and invalid coords can dominate RMSE; we add a few standard NYC-taxi filters that remove clear data errors while preserving the same feature set and model. We also add LightGBM parameters that stabilize training (seed, subsampling seeds) and allow more boosting rounds with RMSE-based early stopping to find a better iteration count (same objective/metric, same model family). These are minimal changes expected to reduce RMSE toward the target while keeping runtime within limits.'

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

np.random.seed(0)

train = pl.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/train.csv")
test = pl.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")

num_rows = train.height
random_indices = np.random.choice(num_rows, size=10000000, replace=False)
train = train[random_indices]
train = train.drop_nulls()



## === cell 2
import numpy as np
import polars as pl


def preprocess(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.col("pickup_datetime")
        .str.strptime(pl.Datetime, strict=False)
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
    longitude = 85.393
    latitude = 111.034
    df = df.with_columns(
        [
            (
                pl.col("abs_longitude") * longitude + pl.col("abs_latitude") * latitude
            ).alias("distance")
        ]
    )
    return df


def drop_encoding(df: pl.DataFrame) -> pl.DataFrame:
    drop_columns = ["key", "pickup_datetime", "pickup_minute", "pickup_second"]
    return df.drop(drop_columns)


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
    return df.to_dummies(one_hot_cols)


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


def drop_outliner(df: pl.DataFrame) -> pl.DataFrame:
    filtered_train_df = df.filter(pl.col("fare_amount").is_not_null())
    filtered_train_df = filtered_train_df.filter(
        (pl.col("fare_amount") > 0) & (pl.col("fare_amount") < 200)
    )

    filtered_train_df = filtered_train_df.filter(
        (pl.col("pickup_latitude") >= 40.5) & (pl.col("pickup_latitude") <= 41.0)
    )
    filtered_train_df = filtered_train_df.filter(
        (pl.col("pickup_longitude") >= -74.5) & (pl.col("pickup_longitude") <= -72.8)
    )
    filtered_train_df = filtered_train_df.filter(
        (pl.col("dropoff_latitude") >= 40.5) & (pl.col("dropoff_latitude") <= 41.0)
    )
    filtered_train_df = filtered_train_df.filter(
        (pl.col("dropoff_longitude") >= -74.5) & (pl.col("dropoff_longitude") <= -72.8)
    )

    filtered_train_df = filtered_train_df.filter(
        (pl.col("passenger_count") >= 1) & (pl.col("passenger_count") <= 6)
    )

    filtered_train_df = filtered_train_df.filter(pl.col("distance") > 0.01)

    filtered_train_df = filtered_train_df.with_columns(
        (pl.col("fare_amount") / (pl.col("distance") + 1e-6)).alias("fare_per_dist")
    )
    filtered_train_df = filtered_train_df.filter(
        (pl.col("fare_per_dist") > 0.5) & (pl.col("fare_per_dist") < 80.0)
    ).drop("fare_per_dist")

    filtered_train_df = filtered_train_df.filter(pl.col("distance") < 50.0)

    return filtered_train_df


train = preprocess(train)
train = distance(train)
train = cycling_encoding(train)
train = is_central(train)
train = drop_outliner(train)
train = is_short_distance(train)

test = preprocess(test)
test = distance(test)
test_key = test["key"]
test = cycling_encoding(test)
test = is_central(test)
test = is_short_distance(test)

train_rows = train.height
combined = pl.concat([train, test], how="diagonal_relaxed")
combined = one_hot_encoding(combined)
train = combined.head(train_rows)
test = combined.tail(combined.height - train_rows)

train = drop_encoding(train)
test = drop_encoding(test)



## === cell 3
import polars as pld

dtypes = train.dtypes
float64_columns = [
    col for col, dtype in zip(train.columns, dtypes) if dtype == pl.Float64
]
train = train.with_columns([pl.col(col).cast(pl.Float32) for col in float64_columns])

dtypes = test.dtypes
float64_columns = [
    col for col, dtype in zip(test.columns, dtypes) if dtype == pl.Float64
]
test = test.with_columns([pl.col(col).cast(pl.Float32) for col in float64_columns])



## === cell 4
import warnings

warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

X = train.drop(["fare_amount"]).to_pandas()
y = train["fare_amount"].to_pandas()

time_cols = ["pickup_year", "pickup_month", "pickup_day", "pickup_hour"]
for c in time_cols:
    if c not in X.columns:
        raise ValueError(f"Missing required time column for time-based split: {c}")

t = (
    X["pickup_year"].astype("int32") * 10_000_00
    + X["pickup_month"].astype("int32") * 10_000
    + X["pickup_day"].astype("int32") * 100
    + X["pickup_hour"].astype("int32")
)
cut = np.quantile(t.values, 0.8)
val_mask = t.values >= cut

X_train, y_train = X.loc[~val_mask], y.loc[~val_mask]
X_val, y_val = X.loc[val_mask], y.loc[val_mask]

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
    "seed": 0,
    "feature_fraction_seed": 0,
    "bagging_seed": 0,
    "data_random_seed": 0,
}

callbacks = [
    lgb.early_stopping(stopping_rounds=50, verbose=False),
    lgb.log_evaluation(period=50),
]

bst = lgb.train(
    params,
    train_data,
    num_boost_round=2000,
    valid_sets=[val_data],
    callbacks=callbacks,
)

y_pred = bst.predict(X_val, num_iteration=bst.best_iteration)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
mse = mean_squared_error(y_val, y_pred)
print(f"rmse:{rmse}")
model_name = "lgbm"



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3872977900.py in <cell line: 0>()
     16 for c in time_cols:
     17     if c not in X.columns:
---> 18         raise ValueError(f"Missing required time column for time-based split: {c}")
     19 
     20 t = (

ValueError: Missing required time column for time-based split: pickup_year

## === cell 5
test_X = test.to_pandas()

test_X = test_X.reindex(columns=X.columns, fill_value=0)

sub_pred = bst.predict(test_X, num_iteration=bst.best_iteration)

exp_num = "03"
submission = pd.DataFrame({"key": test_key.to_list(), "fare_amount": sub_pred})
submission.to_csv(f"submission_{model_name}_{exp_num}.csv", index=False)

print(submission.head())
print("Wrote:", f"submission_{model_name}_{exp_num}.csv", "rows:", len(submission))



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3598073445.py in <cell line: 0>()
      3 test_X = test_X.reindex(columns=X.columns, fill_value=0)
      4 
----> 5 sub_pred = bst.predict(test_X, num_iteration=bst.best_iteration)
      6 
      7 exp_num = "03"

NameError: name 'bst' is not defined

## === cell 6
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

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2783425903.py in <cell line: 0>()
      3 feature_importance_df = pd.DataFrame()
      4 feature_importance_df["feature"] = X.columns
----> 5 feature_importance_df["importance"] = bst.feature_importance()
      6 feature_importance_df = feature_importance_df.sort_values(
      7     by="importance", ascending=False

NameError: name 'bst' is not defined
