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

eli5==0.13.0
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
seaborn==0.12.2
sklearn-pandas==2.2.0
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

4.57128

# 6. Current score

6.95408

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 8.21817) has done: 'I fix the datetime parsing bug by handling already tz-aware timestamps (using `utc=True` and avoiding `tz_localize`), and replace the platform-dependent `strftime('%-H')`/`'%-d'`/`'%-y'` with robust `.dt.hour/.dt.day/.dt.year`. I also prevent the exploratory cells that currently error (the `df/grouped` analysis and `eli5` permutation importance) from stopping the pipeline by guarding them, since they are not needed to train or create `submission.csv`. Finally, I keep the model/training approach intact but set reasonable default XGBoost hyperparameters (and a fixed random seed) to improve RMSE toward your target without changing the overall logic.'
- What this solution (achieved 8.2181) has done: 'I fix the pathing so the notebook reliably reads the competition CSVs from `/kaggle/input/new-york-city-taxi-fare-prediction/` (your current `../input/...` can fail depending on where it’s run). I also remove an unnecessary `.round(2)` on predictions (this is score-harming for RMSE and is not required for submission), which should improve your score toward the 4.57 target without changing the model/training approach. Finally, I keep the `eli5` block guarded (it errors due to protobuf incompatibility) and ensure the submission is written as `submission.csv` with the exact required columns and row alignment.'
- What this solution (achieved 8.08359) has done: 'I fix the one blocking runtime error by guarding the `eli5` block more defensively so protobuf incompatibilities can’t stop execution, while keeping it purely optional. I also correct a key logic bug that harms score: you currently compute the “cwd_factor” features on the test set using test frequencies (distribution leakage from test into features); instead, those factors must be computed from train and then mapped onto both train/val/test. Finally, I keep the model/training loop and feature set intact, but I train the final XGBoost model on the full filtered training data (train+val) before predicting test, so the submission uses all available labeled data and should improve RMSE toward the 4.57 target without changing the approach.'
- What this solution (achieved 6.84965) has done: 'I make the pipeline robust end-to-end by turning the optional `eli5` block into a guaranteed no-op (it can currently throw a protobuf-related `AttributeError` and halt some Kaggle runs). I also fix the biggest score issue without changing the modeling approach: the current `Normalizer()` step collapses meaningful distance/coordinate magnitudes and typically hurts RMSE; replacing it with `StandardScaler()` keeps the same “scale numeric features then fit XGBoost” semantics but calibrates features in a way that matches tree-splitting much better. Finally, I add a small, score-helping but still “same core logic” improvement by using a log1p target transform during training and inverting at prediction time (common for this competition, preserves the same regressor and loss while improving calibration of large fares). The script still train/validate, then refit on full train+val, and always write `submission.csv` with the required `key,fare_amount` columns.'
- What this solution (achieved 6.84965) has done: 'I remove the failing `eli5` block entirely (it’s purely exploratory and currently crashes due to a protobuf incompatibility), so the pipeline always completes and writes `submission.csv`. I also fix a major logic issue: the code currently overwrites `key` with an integer extracted from the string, then later groups by `key` and trains using key-derived frequencies; this unintentionally uses a near-unique identifier and destroys the intended frequency features. Instead, I preserve the original string `key` for submission and create a separate numeric `key_id` for any internal use, and I update the `calc_cwd_factor` logic to count rows robustly without relying on `key` being numeric. These changes keep the same overall feature engineering + scaler + XGBoost approach, but should substantially improve RMSE toward your 4.57128 target.'
- What this solution (achieved 7.06987) has done: 'Your current gap is +2.27837 RMSE above the target (6.84965 vs 4.57128, lower is better), so we should make small, safe improvements that reduce RMSE without changing the overall modeling approach. The biggest score limiter is that you only train on 1,000,000 rows while the competition has far more; increasing the training sample size (still via `nrows`, same logic) typically improves this baseline substantially and stays within the same pipeline. I also replace the slow Python-loop Haversine computation with a vectorized version (identical feature semantics) so the larger sample still finishes under the 600s constraint. Finally, I clamp tiny/zero distances to a small epsilon to prevent inf/huge `fare_per_km` artifacts from dominating filtering and training, which usually reduces noisy outliers and improves RMSE.'
- What this solution (achieved 7.03775) has done: 'Your current RMSE (7.06987) is well above the target (4.57128), so we should make a small, safe change that meaningfully improves generalization without changing the model/feature logic. The biggest remaining issue is that you compute `fare_per_km` and filter outliers using raw `fare_amount`, but you never remove extreme `fare_amount` outliers themselves; those dominate RMSE and are known to hurt this competition badly. I add a single, standard fare cap filter (keeping all existing feature engineering and XGBoost setup intact) and ensure the test-set row order/keys remain unchanged for submission alignment. Everything else (core features, log1p target transform, train/val split, model type/params, and submission format) stays the same.'
- What this solution (achieved 6.95408) has done: 'Your current RMSE (7.03775) is still well above the target (4.57128), so we should make a minimal change that typically reduces RMSE for this specific competition without altering your model type, training loop, loss/objective, or core feature set. The biggest remaining issue is that `pickup_datetime` is engineered (hour/weekday/day/year) but the raw datetime column is never actually used as a feature, so the model misses key time-of-day/seasonality signals. I add a single robust numeric time feature (`pickup_datetime` as Unix seconds) to both train and test, scale it alongside your existing numeric columns, and allow it into the existing feature selection/pipeline. This preserves your overall workflow (feature engineering → scaling → SelectKBest on categorical-like ints → XGBoost on selected+numeric features → log1p target) while improving generalization toward the target RMSE.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

import os

print(os.listdir("/kaggle/input"))

import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
import math

DATA_DIR = "/kaggle/input/new-york-city-taxi-fare-prediction"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")




## === cell 1
train = pd.read_csv(TRAIN_PATH, nrows=3000000)
test = pd.read_csv(TEST_PATH)
train.head()




## === cell 2
train.isnull().sum()




## === cell 3
train = train.dropna(how="any", axis=0)




## === cell 4
train["abs_diff_longitude"] = np.abs(
    train["dropoff_longitude"] - train["pickup_longitude"]
)
train["abs_diff_latitude"] = np.abs(
    train["dropoff_latitude"] - train["pickup_latitude"]
)
test["abs_diff_longitude"] = np.abs(
    test["dropoff_longitude"] - test["pickup_longitude"]
)
test["abs_diff_latitude"] = np.abs(test["dropoff_latitude"] - test["pickup_latitude"])




## === cell 5
train = train.loc[train["fare_amount"] > 0, :]
train = train.loc[
    train["fare_amount"] < 250, :
]  # standard NYC taxi-fare cleanup threshold

train = train.loc[(train["passenger_count"] <= 6) & (train["passenger_count"] > 0), :]
train = train.loc[
    (train["abs_diff_latitude"] < 2) & (train["abs_diff_longitude"] < 2), :
]
train = train.loc[
    (train["abs_diff_latitude"] > 0) & (train["abs_diff_longitude"] > 0), :
]




## === cell 6
train.loc[:, "timestamp_with_key"] = train.loc[:, "key"].astype(str)
test.loc[:, "timestamp_with_key"] = test.loc[:, "key"].astype(str)


def _extract_key_id(s: pd.Series) -> pd.Series:
    suffix = s.astype(str).str.split(".").str[-1]
    return pd.to_numeric(suffix, errors="coerce").astype("Int64")


train.loc[:, "key_id"] = _extract_key_id(train["key"])
test.loc[:, "key_id"] = _extract_key_id(test["key"])




## === cell 7
train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], utc=True, errors="coerce"
)
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], utc=True, errors="coerce"
)

train = train.dropna(subset=["pickup_datetime"]).reset_index(drop=True)
test = test.dropna(subset=["pickup_datetime"]).reset_index(drop=True)

train.loc[:, "hour_no"] = train["pickup_datetime"].dt.hour.astype("int")
test.loc[:, "hour_no"] = test["pickup_datetime"].dt.hour.astype("int")

train.loc[:, "weekday_no"] = train["pickup_datetime"].dt.weekday.astype(
    "int"
)  # Monday=0
test.loc[:, "weekday_no"] = test["pickup_datetime"].dt.weekday.astype("int")

train.loc[:, "day_no"] = train["pickup_datetime"].dt.day.astype("int")
test.loc[:, "day_no"] = test["pickup_datetime"].dt.day.astype("int")

train.loc[:, "year_no"] = (train["pickup_datetime"].dt.year % 100).astype("int")
test.loc[:, "year_no"] = (test["pickup_datetime"].dt.year % 100).astype("int")

train.loc[:, "pickup_unix"] = (
    train["pickup_datetime"].view("int64") // 10**9
).astype("int64")
test.loc[:, "pickup_unix"] = (test["pickup_datetime"].view("int64") // 10**9).astype(
    "int64"
)




## === cell 8
def haversine_km_vec(pick_lon, pick_lat, drop_lon, drop_lat):
    R = 6371.0
    pick_lat = np.radians(pick_lat.astype(float))
    drop_lat = np.radians(drop_lat.astype(float))
    pick_lon = np.radians(pick_lon.astype(float))
    drop_lon = np.radians(drop_lon.astype(float))

    dlat = drop_lat - pick_lat
    dlon = drop_lon - pick_lon
    a = np.sin(dlat / 2.0) ** 2 + np.cos(pick_lat) * np.cos(drop_lat) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


train["dist_haversine_km"] = haversine_km_vec(
    train["pickup_longitude"].values,
    train["pickup_latitude"].values,
    train["dropoff_longitude"].values,
    train["dropoff_latitude"].values,
)
test["dist_haversine_km"] = haversine_km_vec(
    test["pickup_longitude"].values,
    test["pickup_latitude"].values,
    test["dropoff_longitude"].values,
    test["dropoff_latitude"].values,
)

eps = 1e-6
train["dist_haversine_km"] = np.maximum(train["dist_haversine_km"].astype(float), eps)
test["dist_haversine_km"] = np.maximum(test["dist_haversine_km"].astype(float), eps)




## === cell 9
train["fare_per_km"] = train["fare_amount"] / (train["dist_haversine_km"])
train["fare_per_km_passenger"] = train["fare_amount"] / (
    train["dist_haversine_km"] * train["passenger_count"]
)




## === cell 10
train.groupby("key").agg(
    {
        "fare_per_km_passenger": "mean",
        "key": "count",
        "passenger_count": "mean",
        "fare_amount": "mean",
    }
)




## === cell 11
train.loc[
    train["fare_per_km_passenger"] > 20,
    ["fare_per_km_passenger", "fare_amount", "dist_haversine_km"],
]




## === cell 12
try:
    grouped_df = train.groupby("key")
    count = 0
    df = None
    for key, item in grouped_df:
        count += 1
        if count == 2:
            filtered = grouped_df.get_group(key)["dist_haversine_km"] > 1
            df = pd.DataFrame(
                grouped_df.get_group(key)
                .loc[filtered, :]
                .sort_values(by="pickup_datetime")
            )
            break

    if (
        df is not None
        and not df.empty
        and pd.api.types.is_datetime64_any_dtype(df["pickup_datetime"])
    ):
        indexes = ["key", df["pickup_datetime"].dt.strftime("%a"), "hour_no"]
        grouped = df.groupby(indexes).agg(
            {"fare_per_km_passenger": "mean", "hour_no": "count"}
        )
        grouped.rename(columns={"hour_no": "count"}, inplace=True)
        grouped
except Exception as e:
    print("Skipping exploratory grouping block due to:", repr(e))




## === cell 13
try:
    if "grouped" in globals():
        reindexed = grouped.reset_index().drop("key", axis=1)
        get_max_count = reindexed.groupby(["pickup_datetime"]).agg({"count": "max"})
        get_max_count = get_max_count.reindex(
            reindexed["pickup_datetime"], method="ffill"
        )
        reindexed = reindexed.set_index("pickup_datetime")
        reindexed.loc[get_max_count["count"] == reindexed["count"], :]
except Exception as e:
    print("Skipping exploratory reindexing block due to:", repr(e))




## === cell 14
train = train.loc[~((train["fare_per_km"] < 0.2) & (train["dist_haversine_km"] > 1))]
train = train.loc[~((train["dist_haversine_km"] < 0.01) & (train["fare_per_km"] > 50))]




## === cell 15
print(
    train.shape[0]
    - train.loc[
        train["pickup_latitude"].between(40.5, 41)
        | train["dropoff_latitude"].between(40.5, 41)
        | train["pickup_longitude"].between(-74, -73.9)
        | train["dropoff_longitude"].between(-74, -73.9)
    ].shape[0]
)
old_train = train.copy()
train = train.loc[
    train["pickup_latitude"].between(40.5, 41)
    & train["dropoff_latitude"].between(40.5, 41)
    & train["pickup_longitude"].between(-74, -73.9)
    & train["dropoff_longitude"].between(-74, -73.9)
]




## === cell 16
train.loc[:, "pickuplat_no"], pick_lat_bin = pd.cut(
    train["pickup_latitude"], 100, labels=False, retbins=True
)
train.loc[:, "pickuplong_no"], pick_long_bin = pd.cut(
    train["pickup_longitude"], 100, labels=False, retbins=True
)
train.loc[:, "dropofflat_no"], drop_lat_bin = pd.cut(
    train["dropoff_latitude"], 100, labels=False, retbins=True
)
train.loc[:, "dropofflong_no"], drop_long_bin = pd.cut(
    train["dropoff_longitude"], 100, labels=False, retbins=True
)
test["pickuplat_no"] = pd.cut(
    test["pickup_latitude"], pick_lat_bin, labels=False
).fillna(int(train.loc[:, "pickuplat_no"].mean()))
test["pickuplong_no"] = pd.cut(
    test["pickup_longitude"], pick_long_bin, labels=False
).fillna(int(train.loc[:, "pickuplong_no"].mean()))




## === cell 17
test["dropofflat_no"] = pd.cut(
    test["dropoff_latitude"], drop_lat_bin, labels=False
).fillna(int(train.loc[:, "dropofflat_no"].mean()))
test["dropofflong_no"] = pd.cut(
    test["dropoff_longitude"], drop_long_bin, labels=False
).fillna(int(train.loc[:, "dropofflong_no"].mean()))




## === cell 18
train.loc[:, "pickdrop_lat_diff"] = abs(
    train["pickuplat_no"].astype(int) - train["dropofflat_no"].astype(int)
)
train.loc[:, "pickdrop_long_diff"] = abs(
    train["pickuplong_no"].astype(int) - train["dropofflong_no"].astype(int)
)
train.loc[:, "final_dist_factor"] = train["pickdrop_lat_diff"].astype(int) + train[
    "pickdrop_long_diff"
].astype(int)
test.loc[:, "pickdrop_lat_diff"] = abs(
    test["pickuplat_no"].astype(int) - test["dropofflat_no"].astype(int)
)
test.loc[:, "pickdrop_long_diff"] = abs(
    test["pickuplong_no"].astype(int) - test["dropofflong_no"].astype(int)
)
test.loc[:, "final_dist_factor"] = test["pickdrop_lat_diff"].astype(int) + test[
    "pickdrop_long_diff"
].astype(int)




## === cell 19
print(train.shape, test.shape)




## === cell 20
train = train.drop(["fare_per_km_passenger", "fare_per_km"], axis=1)




## === cell 21
def calc_cwd_factor(df, col):
    new_df = (
        df.groupby(col)
        .size()
        .sort_values(ascending=False)
        .reset_index(name="row_count")
    )
    new_df["cwd_factor"] = 1
    count = 1
    for i in range(1, new_df.shape[0]):
        count += 1
        if new_df.loc[i - 1, "row_count"] == new_df.loc[i, "row_count"]:
            count -= 1
        new_df.loc[i, "cwd_factor"] = count
    new_df.index = new_df[col]
    return new_df


def add_cwd_factor_from_train(train_df, other_df, col, out_col):
    fact_df = calc_cwd_factor(train_df, col)
    fallback = int(fact_df["cwd_factor"].max())  # unseen bin -> least frequent rank
    train_df.loc[:, out_col] = (
        train_df[col].map(fact_df["cwd_factor"]).fillna(fallback).astype(int)
    )
    other_df.loc[:, out_col] = (
        other_df[col].map(fact_df["cwd_factor"]).fillna(fallback).astype(int)
    )
    return train_df, other_df


for c in ["pickuplat_no", "pickuplong_no", "dropofflat_no", "dropofflong_no"]:
    train, test = add_cwd_factor_from_train(
        train, test, c, f"{c.replace('_no','')}_cwd_factor"
    )




## === cell 22
print(train.shape, test.shape)




## === cell 23
import sklearn
from sklearn import *
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle




## === cell 24
orig_train = train.copy()
orig_test = test.copy()




## === cell 25
train = shuffle(train.iloc[:, :], random_state=42).reset_index(drop=True)
val = train.iloc[int(0.9 * train.shape[0]) :, :].copy()
train = train.iloc[: int(0.9 * train.shape[0]), :].copy()

cols_to_scale = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "abs_diff_longitude",
    "abs_diff_latitude",
    "dist_haversine_km",
    "pickup_unix",
]

scaler = StandardScaler().fit(train.loc[:, cols_to_scale])
train.loc[:, cols_to_scale] = scaler.transform(train.loc[:, cols_to_scale])
val.loc[:, cols_to_scale] = scaler.transform(val.loc[:, cols_to_scale])
test.loc[:, cols_to_scale] = scaler.transform(test.loc[:, cols_to_scale])




## === cell 26
import seaborn as sns

categorical_cols = [
    i
    for i in train.columns
    if i
    not in [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "abs_diff_longitude",
        "abs_diff_latitude",
        "dist_haversine_km",
        "pickup_unix",
        "key",
        "pickup_datetime",
        "timestamp_with_key",
        "fare_amount",
        "key_id",
    ]
]
numerical_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "abs_diff_longitude",
    "abs_diff_latitude",
    "dist_haversine_km",
    "pickup_unix",
    "fare_amount",
]
object_cols = ["key", "pickup_datetime", "timestamp_with_key"]
cor = train.loc[:, numerical_cols]
f, ax = plt.subplots(1, 1, figsize=(18, 7))
sns.heatmap(cor.corr(), annot=True)




## === cell 27
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

obj = SelectKBest(f_regression, k=10)
obj.fit(train[categorical_cols], train["fare_amount"])




## === cell 28
categories_selected = []
for i in range(len(categorical_cols)):
    if obj.get_support()[i]:
        categories_selected.append(categorical_cols[i])
categories_selected




## === cell 29
train_y = np.log1p(train["fare_amount"].astype(float).values)
val_y = np.log1p(val["fare_amount"].astype(float).values)

cols = [
    i
    for i in categories_selected + numerical_cols
    if i not in ["fare_amount"] + object_cols
]
train_x = train.loc[:, cols]
val_x = val.loc[:, cols]
test_x = test.loc[:, cols]




## === cell 30
import xgboost as xgb
from xgboost import XGBRegressor




## === cell 31
xgbr = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
)
xgb_train = xgbr.fit(train_x, train_y)




## === cell 32
pred_train = np.expm1(xgbr.predict(train_x))
pred_val = np.expm1(xgbr.predict(val_x))
pred_test = np.expm1(xgbr.predict(test_x))




## === cell 33
from sklearn.metrics import mean_squared_error

rmse_train = np.sqrt(mean_squared_error(np.expm1(train_y), pred_train))
rmse_val = np.sqrt(mean_squared_error(np.expm1(val_y), pred_val))
print(rmse_train, rmse_val)




## === cell 34
full_train = pd.concat([train, val], axis=0, ignore_index=True)
full_y = np.log1p(full_train["fare_amount"].astype(float).values)
full_x = full_train.loc[:, cols].copy()

xgbr_final = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
)
xgbr_final.fit(full_x, full_y)
pred_test_final = np.expm1(xgbr_final.predict(test_x))

pred_test_final = np.maximum(pred_test_final, 0.0)

final = pd.DataFrame(
    {"key": test["timestamp_with_key"].values, "fare_amount": pred_test_final},
    columns=["key", "fare_amount"],
)

final.to_csv("submission.csv", index=False)

print(final.shape)
final.head()
