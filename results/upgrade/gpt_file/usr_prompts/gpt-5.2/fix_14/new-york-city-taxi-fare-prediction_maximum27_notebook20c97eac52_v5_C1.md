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

3.76365

# 6. Current score

5.71529

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.56585) has done: 'Your current gap to target is large (RMSE 5.82 vs 3.76, lower is better), and the biggest likely causes are (1) a datetime parsing format mismatch that silently creates nulls/bad time features, (2) train/test feature misalignment from `to_dummies` (train has pickup_year/day levels that test may not, and vice versa), and (3) using `test` as a Polars DataFrame directly in `bst.predict`, which can mis-handle column order/types. I make minimal fixes: robust datetime parsing, build dummies using a shared schema (concat train+test then split back), and ensure prediction uses the exact same pandas column order as training. These are small, core-logic-preserving changes that typically improve RMSE materially for this competition.'
- What this solution (achieved 4.56585) has done: 'The crash happens because `train` contains `fare_amount` while `test` does not, so vertical concatenation fails due to different widths. I fix this by temporarily adding a placeholder `fare_amount` column to `test` before concatenation, then removing it after one-hot encoding so train/test end up with an identical feature schema. I also ensure datetime parsing handles the fractional seconds present in the dataset by not forcing a mismatched fixed format, which can otherwise create nulls and degrade features and score. These changes preserve your core feature logic and LightGBM training loop, but should both unblock execution and improve RMSE toward the target by fixing feature alignment and time-feature quality.'
- What this solution (achieved 5.6103) has done: 'I fix the Polars vertical concat error by enforcing identical column sets and identical column order between train and test before concatenation (the current failure happens because `fare_amount` exists only in train and gets appended at the end in test, causing a name mismatch by position). I keep your feature logic and LightGBM training the same, but make datetime parsing explicitly handle the dataset’s fractional seconds consistently to avoid silent null timestamps that can harm the time-derived features and RMSE. I also ensure the final test matrix uses the exact same column order as the training matrix before prediction (already mostly done) and always write a valid `.csv` submission with `key,fare_amount`.'
- What this solution (achieved 5.82926) has done: 'I fix the root cause of the empty training set by correcting the datetime parsing: your current `str.strptime` formats don’t match the dataset’s `pickup_datetime` strings (which include `UTC`), so almost all datetimes become null and are dropped. I keep your feature engineering and LightGBM training logic intact, but make parsing robust by stripping the trailing timezone token and parsing without forcing a mismatched format. I also make `drop_outliner` safe to apply to both train and test by only filtering on `fare_amount` when that column exists, preventing accidental row drops if it ever gets used elsewhere. After that, the pipeline train, predict, and write a valid `submission_*.csv` with `key,fare_amount`.'
- What this solution (achieved 5.82927) has done: 'Your RMSE (5.83) is still far from the target (3.76, lower is better), and the most likely “minimal but meaningful” miss is that the datetime parsing is still silently failing for many rows (the dataset uses a space-separated `YYYY-mm-dd HH:MM:SS UTC` style, which `str.strptime` may not reliably infer when fractional seconds appear). I make datetime parsing deterministic by explicitly stripping `UTC` and using a concrete format with optional fractional seconds; this keeps your exact feature set but improves their correctness. I also keep your existing train/test dummy-schema alignment, but ensure that any remaining null numeric features are filled to 0 right before LightGBM so the model isn’t learning from missing-value patterns created by preprocessing. These are small, core-logic-preserving changes that typically reduce RMSE for this competition without changing the model/training approach.'
- What this solution (achieved 5.72909) has done: 'Your RMSE is still far from the target (5.83 vs 3.76, lower is better), and the most likely “minimal but meaningful” fix is that your `distance` feature is currently a linear degrees-to-km approximation that is noticeably wrong at NYC latitudes. I keep your exact feature set and LightGBM training loop intact, but compute `distance` using a proper haversine formula (same single feature name, just more accurate values), which typically drops RMSE materially in this competition. I also clamp negative predictions to 0.0 (fares can’t be negative) to reduce RMSE on outliers without changing the model architecture/training approach. Everything else (sampling, filtering, one-hot alignment, training params, submission format) stays the same.'
- What this solution (achieved 5.72909) has done: 'Your current RMSE (5.729) is worse than the target (3.764), so we should make a small, legitimate improvement without changing the model or feature set. The biggest minimal win here is to fix train/test feature schema alignment around one-hot encoding: `to_dummies` can create different dummy columns depending on which split sees which categories, and doing it on the combined data currently also creates a dummy for the artificial `__is_test__` flag (leak/noise). I keep your exact feature logic and LightGBM setup, but (1) exclude `__is_test__` from dummy expansion and (2) explicitly ensure the post-dummy train/test columns are identical and in the same order before converting to pandas. This typically reduces RMSE meaningfully while staying within your core approach.'
- What this solution (achieved 5.71529) has done: 'Your current RMSE (5.729) is still well above the target (3.764, lower is better), so we should make a small, legitimate improvement without changing the model or feature set. The most impactful minimal fix here is to stop one-hot encoding `pickup_year`: in this competition, `pickup_year` is essentially constant in train/test (mostly 2009–2015) and turning it into sparse dummies adds noise and schema fragility; keeping it as a numeric feature typically improves generalization and lowers RMSE. I keep your exact LightGBM setup and all engineered features, but change `one_hot_cols` to only dummy `pickup_day`, and I also explicitly drop the artificial `__is_test__` flag before dummy expansion so it can’t ever become a feature. These are minimal, core-logic-preserving adjustments aimed at moving RMSE down toward the target band.'
- What this solution (achieved 5.71529) has done: 'I make two minimal, score-relevant fixes that also ensure a valid submission is always produced: (1) preserve all `key` values from the original test set by saving `test_key` before any filtering, because filtering test rows breaks submission row count/alignment and can severely hurt RMSE; and (2) stop applying `drop_outliner` to the test set (keep it only for training) so we don’t drop valid test rides and we keep the prediction distribution consistent with Kaggle evaluation. These changes keep your model, features, and training loop identical, but fix the most likely cause of “no score / invalid submission” and should move RMSE down toward the target by preventing misaligned/partial submissions. Everything else (feature engineering, LightGBM params, one-hot alignment, prediction/clamp) remains the same.'
- What this solution (achieved 5.71529) has done: 'Your RMSE (5.715) is still worse than the target (3.764, lower is better), so we should make a small, legitimate improvement without changing your model or feature set. The biggest minimal gain here is to stop dropping training rows with missing *any* column (`train.drop_nulls()`), which currently discards a large fraction of data (often due to missing `fare_amount` or rare nulls) and hurts generalization; instead we only drop rows with null `fare_amount` and let feature nulls be handled by your existing `.fillna(0.0)`. I also keep `test_key` aligned with the post-preprocessing test rows (even though we don’t filter test now) to prevent any accidental mismatch if datetime nulls appear. Everything else (features, one-hot strategy, LightGBM training loop/params, prediction/clamp, output format) remains the same.'

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

train = pl.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/train.csv")
test = pl.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")

num_rows = train.height
rng = np.random.default_rng(0)
random_indices = rng.choice(num_rows, size=10_000_000, replace=False)
train = train[random_indices]

train = train.drop_nulls(["fare_amount"])



## === cell 2
import numpy as np
import polars as pl


def preprocess(df: pl.DataFrame) -> pl.DataFrame:
    dt_str = pl.col("pickup_datetime").cast(pl.Utf8).str.strip_chars()
    dt_str = dt_str.str.replace(r"\s+UTC$", "", literal=False)

    dt = dt_str.str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S%.f", strict=False)
    dt2 = dt_str.str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
    dt = pl.coalesce([dt, dt2])

    df = df.with_columns(dt.alias("pickup_datetime"))
    df = df.drop_nulls(["pickup_datetime"])

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
    R = 6371.0  # Earth radius in km

    lat1 = pl.col("pickup_latitude") * (np.pi / 180.0)
    lat2 = pl.col("dropoff_latitude") * (np.pi / 180.0)
    dlat = (pl.col("dropoff_latitude") - pl.col("pickup_latitude")) * (np.pi / 180.0)
    dlon = (pl.col("dropoff_longitude") - pl.col("pickup_longitude")) * (np.pi / 180.0)

    a = (dlat / 2).sin() ** 2 + (lat1.cos() * lat2.cos() * (dlon / 2).sin() ** 2)
    c = (a.sqrt()).arcsin() * 2.0

    return df.with_columns((R * c).alias("distance"))


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
    one_hot_cols = ["pickup_day"]
    df = df.with_columns(
        [
            pl.col(c).fill_null(-1).cast(pl.Int32).alias(c)
            for c in one_hot_cols
            if c in df.columns
        ]
    )
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
    filtered_df = df.filter(
        (pl.col("pickup_latitude") >= 40) & (pl.col("pickup_latitude") < 41)
    )
    filtered_df = filtered_df.filter(
        (pl.col("pickup_longitude") >= -74) & (pl.col("pickup_longitude") < -73)
    )
    filtered_df = filtered_df.filter(
        (pl.col("dropoff_longitude") >= -74) & (pl.col("dropoff_longitude") < -73)
    )
    filtered_df = filtered_df.filter(
        (pl.col("dropoff_latitude") >= 40) & (pl.col("dropoff_latitude") < 41)
    )
    filtered_df = filtered_df.filter(
        (pl.col("passenger_count") >= 1) & (pl.col("passenger_count") < 6)
    )
    if "fare_amount" in filtered_df.columns:
        filtered_df = filtered_df.filter(
            (pl.col("fare_amount") > 0) & (pl.col("fare_amount") < 500)
        )
    return filtered_df


def rule_base(df: pl.DataFrame) -> pl.DataFrame:
    is_weekday = pl.col("pickup_weekday").is_in([0, 1, 2, 3, 4])
    is_weekend = pl.col("pickup_weekday").is_in([5, 6])

    return df.with_columns(
        pl.when(is_weekday & pl.col("pickup_hour").is_in([16, 17, 18, 19, 20]))
        .then(9 * pl.col("distance") / 0.32)
        .when(
            is_weekday
            & pl.col("pickup_hour").is_in([20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6])
        )
        .then(pl.col("distance") / 0.32)
        .when(is_weekday)
        .then(0.5 * pl.col("distance") / 0.32)
        .when(
            is_weekend
            & pl.col("pickup_hour").is_in([20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6])
        )
        .then(60 * pl.col("distance") / 19.2)
        .otherwise(30 * pl.col("distance") / 19.2)
        .alias("calculated_value")
    )


def drop_encoding(df: pl.DataFrame) -> pl.DataFrame:
    drop_columns = ["key", "pickup_datetime", "pickup_minute", "pickup_second"]
    existing = [c for c in drop_columns if c in df.columns]
    return df.drop(existing)


train = preprocess(train)
train = distance(train)
train = cycling_encoding(train)
train = is_central(train)
train = is_short_distance(train)
train = rule_base(train)
train = drop_outliner(train)

test = preprocess(test)
test = distance(test)
test = cycling_encoding(test)
test = is_central(test)
test = is_short_distance(test)
test = rule_base(test)

test_key = test["key"]

if "fare_amount" not in test.columns:
    test = test.with_columns(pl.lit(None).cast(pl.Float64).alias("fare_amount"))

all_cols = sorted(set(train.columns) | set(test.columns))
train = train.select(
    [pl.col(c) if c in train.columns else pl.lit(None).alias(c) for c in all_cols]
)
test = test.select(
    [pl.col(c) if c in test.columns else pl.lit(None).alias(c) for c in all_cols]
)

train = train.with_columns(pl.lit(0).cast(pl.Int8).alias("__is_test__"))
test = test.with_columns(pl.lit(1).cast(pl.Int8).alias("__is_test__"))
combined = pl.concat([train, test], how="vertical", rechunk=True)

combined_is_test = combined.select(["__is_test__"])
combined = combined.drop("__is_test__")
combined = one_hot_encoding(combined)
combined = pl.concat([combined, combined_is_test], how="horizontal")

train = combined.filter(pl.col("__is_test__") == 0).drop("__is_test__")
test = combined.filter(pl.col("__is_test__") == 1).drop("__is_test__")

train = drop_encoding(train)
test = drop_encoding(test)

if "fare_amount" in test.columns:
    test = test.drop("fare_amount")

train_cols = set(train.columns)
test_cols = set(test.columns)
missing_in_test = sorted(list(train_cols - test_cols))
missing_in_train = sorted(list(test_cols - train_cols))

if missing_in_test:
    test = test.with_columns(
        [pl.lit(0).cast(pl.Int8).alias(c) for c in missing_in_test]
    )
if missing_in_train:
    train = train.with_columns(
        [pl.lit(0).cast(pl.Int8).alias(c) for c in missing_in_train]
    )

feature_cols = [c for c in sorted(train.columns) if c != "fare_amount"]
train = train.select(feature_cols + ["fare_amount"])
test = test.select(feature_cols)



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

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if train.height == 0:
    raise ValueError(
        "Training dataframe is empty after preprocessing/filtering. "
        "Check feature engineering and filtering steps."
    )

X = train.drop(["fare_amount"]).to_pandas().fillna(0.0)
y = train["fare_amount"].to_pandas()

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
    "seed": 0,
    "feature_fraction_seed": 0,
    "bagging_seed": 0,
}

bst = lgb.train(params, train_data, num_boost_round=100, valid_sets=[val_data])

y_pred = bst.predict(X_val, num_iteration=bst.best_iteration)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"rmse:{rmse}")
model_name = "lgbm"



## === cell 5
X_test = test.to_pandas().fillna(0.0)
X_test = X_test.reindex(columns=X.columns, fill_value=0.0)

sub_pred = bst.predict(X_test, num_iteration=bst.best_iteration)

sub_pred = np.maximum(sub_pred, 0.0)

exp_num = "01"
submission = pd.DataFrame({"key": test_key.to_list(), "fare_amount": sub_pred})
submission.to_csv(f"submission_{model_name}_{exp_num}.csv", index=False)
print(f"Wrote submission_{model_name}_{exp_num}.csv with shape {submission.shape}")



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
