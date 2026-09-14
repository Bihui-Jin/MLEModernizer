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

3.73427

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 5.93258) has done: 'I fix two correctness issues that are likely inflating your RMSE: (1) your datetime parsing format doesn’t match the dataset (no `" UTC"` suffix), which can silently create nulls and degrade features, and (2) your train/test one-hot encoding is done independently, causing column mismatches (missing/extra dummy columns) between train and test at prediction time. I keep your feature engineering and LightGBM setup the same, but I align dummy columns using train’s schema, fill missing columns with 0, and ensure `bst.predict()` receives a pandas DataFrame with identical columns to training. These are minimal changes that usually produce a meaningful RMSE improvement without changing the modeling approach or training loop. The script still run end-to-end and write a valid submission CSV.'
- What this solution (achieved 5.70416) has done: 'I fix the crash where training becomes empty by correcting the datetime parsing format to match the dataset (it includes microseconds) and by making the outlier filters robust to boundary cases (inclusive ranges and proper fare upper bound), while keeping your feature engineering and LightGBM training logic intact. I also fix the `NameError` for `test_key_all` by defining it outside the preprocessing cell so it always exists at submission time. Finally, I ensure train/test one-hot columns stay aligned exactly as you already intended, so the model receives identical feature columns at inference and a valid `submission.csv` is always written.'

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
    c = 2.0 * (a.sqrt()).arctan2((1.0 - a).sqrt())
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
        c = 2.0 * (a.sqrt()).arctan2((1.0 - a).sqrt())
        return 6371.0 * c

    return df.with_columns(
        [
            hav_expr(p_lat, p_lon, c_lat, c_lon).alias("pickup_to_center_km"),
            hav_expr(d_lat, d_lon, c_lat, c_lon).alias("dropoff_to_center_km"),
        ]
    )


def drop_outliner_train_only(df: pl.DataFrame) -> pl.DataFrame:
    filtered_train_df = df.filter(
        (pl.col("pickup_latitude") >= 40) & (pl.col("pickup_latitude") <= 41)
    )
    filtered_train_df = filtered_train_df.filter(
        (pl.col("pickup_longitude") >= -74) & (pl.col("pickup_longitude") <= -73)
    )
    filtered_train_df = filtered_train_df.filter(
        (pl.col("dropoff_longitude") >= -74) & (pl.col("dropoff_longitude") <= -73)
    )
    filtered_train_df = filtered_train_df.filter(
        (pl.col("dropoff_latitude") >= 40) & (pl.col("dropoff_latitude") <= 41)
    )
    filtered_train_df = filtered_train_df.filter(
        (pl.col("passenger_count") >= 1) & (pl.col("passenger_count") <= 6)
    )
    filtered_train_df = filtered_train_df.filter(
        (pl.col("fare_amount") > 0) & (pl.col("fare_amount") < 500)
    )
    return filtered_train_df


def build_train_with_retries(
    full_train_path: str,
    seed: int = 0,
    initial_sample: int = 10_000_000,
    min_rows_required: int = 200_000,
    max_retries: int = 3,
) -> pl.DataFrame:
    base = pl.read_csv(full_train_path).drop_nulls()
    n = base.height
    rng_local = np.random.default_rng(seed)

    last_df = None
    for attempt in range(max_retries + 1):
        factor = 1 if attempt == 0 else (2**attempt)
        sample_n = min(n, initial_sample * factor)
        idx = rng_local.choice(n, size=sample_n, replace=False)
        df = base[idx]

        df = preprocess(df)
        df = df.drop_nulls(["pickup_datetime"])
        df = max_min_scaling(df)

        df = distance(df)
        df = add_center_distance(df)

        df = is_central(df)
        df = drop_outliner_train_only(df)
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
test_key_all = test_raw["key"]

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

test_proc = test_proc.with_columns(
    [
        pl.col("pickup_latitude").clip(40, 42).alias("pickup_latitude"),
        pl.col("dropoff_latitude").clip(40, 42).alias("dropoff_latitude"),
        pl.col("pickup_longitude").clip(-75, -72).alias("pickup_longitude"),
        pl.col("dropoff_longitude").clip(-75, -72).alias("dropoff_longitude"),
    ]
)

test_proc = distance(test_proc)
test_proc = add_center_distance(test_proc)

test_proc = is_central(test_proc)
test_proc = drop_encoding(test_proc)
test_proc = cycling_encoding(test_proc)
test_proc = one_hot_encoding(test_proc)

test = test_proc

print("Train rows after preprocessing:", train.height)
print(
    "Test rows after preprocessing (should equal original test rows):",
    test.height,
    "/",
    test_raw.height,
)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2811478142.py in <cell line: 0>()
    216 
    217 
--> 218 train = build_train_with_retries(
    219     "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
    220     seed=0,

/tmp/ipykernel_11/2811478142.py in build_train_with_retries(full_train_path, seed, initial_sample, min_rows_required, max_retries)
    196         df = max_min_scaling(df)
    197 
--> 198         df = distance(df)
    199         df = add_center_distance(df)
    200 

/tmp/ipykernel_11/2811478142.py in distance(df)
     48 
     49     a = (dphi / 2.0).sin() ** 2 + p_lat.cos() * d_lat.cos() * (dlambda / 2.0).sin() ** 2
---> 50     c = 2.0 * (a.sqrt()).arctan2((1.0 - a).sqrt())
     51     R = 6371.0  # km
     52 

AttributeError: 'Expr' object has no attribute 'arctan2'

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
y = train["fare_amount"].to_pandas()

for c in X.columns:
    if X[c].dtype == "bool":
        X[c] = X[c].astype("int8")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

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

bst = lgb.train(params, train_data, num_boost_round=100, valid_sets=[test_data])

y_pred = bst.predict(X_val, num_iteration=bst.best_iteration)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
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
sub_pred = bst.predict(test_pd, num_iteration=bst.best_iteration)

exp_num = "is_central"
submission = pd.DataFrame({"key": test_key_all.to_pandas(), "fare_amount": sub_pred})

submission = submission[["key", "fare_amount"]]
submission.to_csv(f"submission_{model_name}_{exp_num}.csv", index=False)
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Rows in submission:", len(submission))
print("Wrote:", f"submission_{model_name}_{exp_num}.csv", "and submission.csv")



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1908913650.py in <cell line: 0>()
      2 
      3 exp_num = "is_central"
----> 4 submission = pd.DataFrame({"key": test_key_all.to_pandas(), "fare_amount": sub_pred})
      5 
      6 submission = submission[["key", "fare_amount"]]

NameError: name 'test_key_all' is not defined

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
