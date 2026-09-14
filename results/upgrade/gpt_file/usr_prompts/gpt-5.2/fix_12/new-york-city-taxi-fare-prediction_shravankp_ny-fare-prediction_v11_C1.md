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

4.41786

# 6. Current score

4.8734

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 11.87486) has done: 'I fix the datetime handling so it works whether the parsed timestamps are tz-naive or already tz-aware, which currently stops execution early and leaves `pickup_datetime` as strings in later cells. I also prevent binning-induced NaNs in the test set from crashing integer casts and from breaking the “cwd_factor” lookup, by filling those NaNs with safe default bin codes/factors. Finally, I ensure the train/test feature columns are aligned when building `test_x`, so the model can score and a valid `submission.csv` with `key,fare_amount` is always written. These changes are bug-fixes and robustness fixes; they preserve your model choice (XGBRegressor default) and the overall feature engineering flow.'
- What this solution (achieved 9.24055) has done: 'I fix the failing weekday feature cell by avoiding `pd.DataFrame(..., dtype=int)` on object data and instead using pandas’ datetime accessors to produce a clean integer `weekday_no` for both train and test. I also keep datetime parsing robust (UTC-aware) and ensure any remaining NaTs are filled so downstream feature engineering doesn’t break. These are execution/robustness fixes that preserve your existing feature set and XGBRegressor training logic, and they should also improve RMSE by correctly adding the intended weekday signal instead of crashing. The script still write a valid `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 8.72474) has done: 'Your current RMSE gap is large (9.24 vs target 4.42; lower is better), so we need a small set of high-impact fixes that keep your XGBRegressor + engineered features intact but correct two major score killers: (1) you are destroying the real `key` (ID) by converting it to a numeric suffix, which breaks the submission alignment; and (2) you are rounding predictions to 2 decimals before scoring/submitting, which unnecessarily increases RMSE. I preserve your feature engineering and training flow, but keep the original `key` string for submission (and for any grouping you still do) by moving the numeric conversion into a separate helper column, and I remove rounding on predictions. These two minimal changes are legitimate, keep semantics, and should move the score strongly toward your target without changing the model architecture or training approach.'
- What this solution (achieved 12.13652) has done: 'Two score-killers remain while keeping your core XGBRegressor + feature pipeline intact: you’re computing the “cwd_factor” frequency ranks separately on the test set (so the same bin code means different things in train vs test), and you’re leaving a few label-derived intermediate columns (like `key_num`) in the model that don’t generalize and can harm RMSE. I (1) compute the cwd-factor lookup tables only from the training data and apply them to both train and test for consistent semantics, and (2) drop `key_num` from the model features (still keeping the original string `key` for submission), which usually reduces overfitting/noise. These are minimal, execution-safe changes that preserve your model choice, training loop, and engineered features, but should move RMSE down toward your 4.42 target. The script still run end-to-end and write a valid `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 11.44879) has done: 'Your current RMSE (12.13652; lower is better) is still far from the target (4.41786), so we need a small but high-impact improvement without changing your overall pipeline (feature engineering + XGBRegressor fit/predict). The biggest remaining score-killer is that you fit the XGBRegressor with default hyperparameters, which are too weak for this problem and typically underfit badly; adjusting a few standard, stable parameters (while keeping the same model class and training loop) should move RMSE substantially downward toward the target band. I also add a minimal, deterministic `n_jobs`/`tree_method` selection for speed/stability within the 600s limit, without changing semantics. Everything else (your cleaning, engineered features, split, Normalizer usage, and submission writing) is kept intact.'
- What this solution (achieved 10.92248) has done: 'Your RMSE (11.45) is still far from the target (4.42; lower is better), so we should make a small, legitimate change that improves predictive power without changing your overall pipeline (same engineered features + XGBRegressor fit/predict). The most impactful minimal step is to add early-stopping on your existing 90/10 split so the model stops at the best validation iteration rather than over/under-shooting with a fixed 800 trees, which typically reduces RMSE substantially for this competition. I also set `eval_metric="rmse"` and use the best iteration automatically for `predict`, keeping everything else (data loading, cleaning, feature engineering, Normalizer, and submission format) the same. This should move the score closer to the target band while preserving your core logic and producing the same `submission.csv` schema.'
- What this solution (achieved 10.92248) has done: 'Your current RMSE (10.92248; lower is better) is still far above the target (4.41786), so we should make a small but high-impact improvement without changing your feature engineering or model class. The biggest remaining score-killer is that your coordinate filter in cell 18 is logically wrong (it uses `between(...) | between(...)`), which unintentionally removes almost all valid NYC trips; fixing this to the correct bounding-box logic preserves your intent and should substantially improve generalization. I also make the `pd.cut` binning consistent by using `include_lowest=True` on the training cuts (matching what you already do for test), which reduces train/test bin-edge mismatch at the boundary. Everything else (XGBRegressor setup, early stopping, Normalizer usage, columns used, and submission writing) stays the same and it still writes a valid `submission.csv`.'
- What this solution (achieved 6.3347) has done: 'Your current RMSE (10.92248) is far above the target (4.41786; lower is better), so we need a minimal but high-impact change that preserves your feature engineering and XGBRegressor training loop. The biggest remaining score-killer is that the model never gets the single most important signal in this competition: a scaled distance feature (your `Normalizer()` makes all coordinate-based magnitudes unit-length and destroys distance scale). I keep the same preprocessing structure (a single transformer fit on train and applied to val/test) but switch from `Normalizer` to `StandardScaler`, which preserves relative scale information and typically drops RMSE substantially for taxi-fare. I also clip negative predictions to 0 (a legitimate post-processing for fare amounts) to reduce RMSE outliers without changing the model or loss.'
- What this solution (achieved 6.3347) has done: 'Your current RMSE (6.3347) is still well above the target (4.41786; lower is better), so we make one minimal, high-impact improvement while keeping your XGBRegressor training loop and engineered features intact. The biggest remaining issue is that you compute `key_num` from the `key` string and then use it in multiple feature constructions (`calc_cwd_factor` groups on it), but on this dataset `key` does not reliably contain a “.suffix”, so `key_num` becomes mostly NaN and those downstream group-based features become noisy/degenerate. I rebuild `key_num` deterministically from the existing `timestamp_with_key` by extracting the datetime portion and using its integer nanosecond value (stable, non-NaN, and derived from the same ID), preserving the intended “time-derived grouping” behavior without changing your model or training approach. This should improve feature signal quality and move RMSE down toward the target band while still producing the same valid `submission.csv`.'
- What this solution (achieved 5.05873) has done: 'Your current RMSE (6.3347) is still far above the target (4.41786), so we should make one minimal, legitimate improvement that preserves your feature engineering + XGBRegressor training approach. The biggest remaining score-killer is that you’re training on raw `fare_amount` with a squared-error objective, which over-weights large-fare outliers; switching to a log1p target transform (and inverse-transforming predictions) is a standard, stable way to improve RMSE on this competition without changing the model class, features, or training loop. I keep the same split, scaler, columns, XGB parameters, and early stopping, but train on `log1p(fare_amount)` and then apply `expm1` to predictions. I also keep the existing non-negative clipping after inverse-transform to avoid impossible negative fares.'
- What this solution (achieved 4.8734) has done: 'Your current RMSE (5.05873) is worse than the target (4.41786), so we should make a small, legitimate improvement without changing the overall approach (same XGBRegressor, same split, same training loop, same metric). The lowest-risk, high-signal fix is to add a couple of standard time-derived features from `pickup_datetime` (hour, month, year) that are already available but currently unused, which typically reduces RMSE materially in this competition. I also make the final training feature list explicitly include these new columns (and keep the existing exclusions), so train/val/test stay perfectly aligned. Everything else—including the log1p target, StandardScaler usage, early stopping, and submission writing—stays intact.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

print(os.listdir("/kaggle/input"))

import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
import math



## === cell 1
train_path_candidates = [
    "../input/train.csv",
    "/kaggle/input/train.csv",
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
]
test_path_candidates = [
    "../input/test.csv",
    "/kaggle/input/test.csv",
    "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
]


def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of these paths exist: {paths}")


train_path = _first_existing(train_path_candidates)
test_path = _first_existing(test_path_candidates)

train = pd.read_csv(train_path, nrows=1000000)
test = pd.read_csv(test_path)
train.head()



## === cell 2
len(train["fare_amount"].unique())



## === cell 3
train.isnull().sum()



## === cell 4
train = train.dropna(how="any", axis=0)



## === cell 5
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



## === cell 6
train = train.loc[train["fare_amount"] > 0, :]
train = train.loc[(train["passenger_count"] <= 6) & (train["passenger_count"] > 0), :]
train = train.loc[
    (train["abs_diff_latitude"] < 2) & (train["abs_diff_longitude"] < 2), :
]
train = train.loc[
    (train["abs_diff_latitude"] > 0) & (train["abs_diff_longitude"] > 0), :
]



## === cell 7
train.loc[:, "timestamp_with_key"] = train.loc[:, "key"]
test.loc[:, "timestamp_with_key"] = test.loc[:, "key"]


def _key_to_keynum(s: pd.Series) -> pd.Series:
    dt_part = s.astype(str).str.split(".", n=1).str[0]
    ts = pd.to_datetime(dt_part, errors="coerce", utc=True)
    return ts.view("int64").fillna(0).astype("int64")


train["key_num"] = _key_to_keynum(train["timestamp_with_key"])
test["key_num"] = _key_to_keynum(test["timestamp_with_key"])



## === cell 8
from math import floor


def chooseSlot(x):
    hr = x.hour
    return int(hr / 3 + 1)


train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], errors="coerce", utc=True
)
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], errors="coerce", utc=True
)

default_ts = pd.Timestamp("2010-01-01", tz="UTC")
train["pickup_datetime"] = train["pickup_datetime"].fillna(default_ts)
test["pickup_datetime"] = test["pickup_datetime"].fillna(default_ts)

train["time_slot"] = pd.DataFrame(
    list(map(lambda x: chooseSlot(x), train["pickup_datetime"][:])), index=train.index
)
test["time_slot"] = pd.DataFrame(
    list(map(lambda x: chooseSlot(x), test["pickup_datetime"][:])), index=test.index
)



## === cell 9
train["weekday_no"] = train["pickup_datetime"].dt.weekday.astype(np.int16)
test["weekday_no"] = test["pickup_datetime"].dt.weekday.astype(np.int16)



## === cell 10
train["pickup_hour"] = train["pickup_datetime"].dt.hour.astype(np.int16)
test["pickup_hour"] = test["pickup_datetime"].dt.hour.astype(np.int16)

train["pickup_month"] = train["pickup_datetime"].dt.month.astype(np.int16)
test["pickup_month"] = test["pickup_datetime"].dt.month.astype(np.int16)

train["pickup_year"] = train["pickup_datetime"].dt.year.astype(np.int16)
test["pickup_year"] = test["pickup_datetime"].dt.year.astype(np.int16)




## === cell 11
def dist_haversine(x):
    R = 6371  # km
    picklat = math.radians(x[1])
    droplat = math.radians(x[3])
    latdiff = abs(droplat - picklat)
    picklon = math.radians(x[0])
    droplon = math.radians(x[2])
    londiff = abs(droplon - picklon)

    a = math.sin(latdiff / 2) * math.sin(latdiff / 2) + math.cos(picklat) * math.cos(
        droplat
    ) * math.sin(londiff / 2) * math.sin(londiff / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


train["dist_haversine_km"] = pd.DataFrame(
    list(
        map(
            lambda x: dist_haversine(x),
            train[
                [
                    "pickup_longitude",
                    "pickup_latitude",
                    "dropoff_longitude",
                    "dropoff_latitude",
                ]
            ].values,
        )
    ),
    index=train.index,
)
test["dist_haversine_km"] = pd.DataFrame(
    list(
        map(
            lambda x: dist_haversine(x),
            test[
                [
                    "pickup_longitude",
                    "pickup_latitude",
                    "dropoff_longitude",
                    "dropoff_latitude",
                ]
            ].values,
        )
    ),
    index=test.index,
)



## === cell 12
train["fare_per_km"] = train["fare_amount"] / (train["dist_haversine_km"])
train["fare_per_km_passenger"] = train["fare_amount"] / (
    train["dist_haversine_km"] * train["passenger_count"]
)



## === cell 13
train.groupby("key_num").agg(
    {
        "fare_per_km_passenger": "mean",
        "key_num": "count",
        "passenger_count": "mean",
        "fare_amount": "mean",
    }
)



## === cell 14
train.loc[
    train["fare_per_km_passenger"] > 20,
    ["fare_per_km_passenger", "fare_amount", "dist_haversine_km"],
]



## === cell 15
grouped_df = train.groupby("key_num")
count = 0
df = None
for k, item in grouped_df:
    count += 1
    if count == 2:  ## to view key = 2
        filtered = (
            grouped_df.get_group(k)["dist_haversine_km"] > 1
        )  # ignoring drives within 1km
        df = pd.DataFrame(
            grouped_df.get_group(k).loc[filtered, :].sort_values(by="pickup_datetime")
        )
        break



## === cell 16
if (
    df is not None
    and "pickup_datetime" in df.columns
    and pd.api.types.is_datetime64_any_dtype(df["pickup_datetime"])
):
    indexes = ["key_num", df["pickup_datetime"].dt.strftime("%a"), "time_slot"]
    grouped = (
        df[:][:]
        .groupby(indexes)
        .agg({"fare_per_km_passenger": "mean", "time_slot": "count"})
    )
    grouped.rename(columns={"time_slot": "count"}, inplace=True)
    grouped
else:
    grouped = None



## === cell 17
if grouped is not None:
    reindexed = grouped.reset_index().drop("key_num", axis=1)
    get_max_count = reindexed.groupby(["pickup_datetime"]).agg({"count": "max"})
    get_max_count = get_max_count.reindex(reindexed["pickup_datetime"], method="ffill")
    reindexed = reindexed.set_index("pickup_datetime")
    reindexed.loc[get_max_count["count"] == reindexed["count"], :]
else:
    reindexed = None



## === cell 18
train = train.loc[~((train["fare_per_km"] < 0.2) & (train["dist_haversine_km"] > 1))]
train = train.loc[~((train["dist_haversine_km"] < 0.01) & (train["fare_per_km"] > 50))]



## === cell 19
outside_mask = ~(
    train["pickup_latitude"].between(39, 42)
    & train["dropoff_latitude"].between(39, 42)
    & train["pickup_longitude"].between(-74.4, -72.8)
    & train["dropoff_longitude"].between(-74.4, -72.8)
)
print(outside_mask.sum())

train = train.loc[
    train["pickup_latitude"].between(39, 42)
    & train["dropoff_latitude"].between(39, 42)
    & train["pickup_longitude"].between(-74.4, -72.8)
    & train["dropoff_longitude"].between(-74.4, -72.8)
].copy()



## === cell 20
train.loc[:, "pickuplat_no"], pick_lat_bin = pd.cut(
    train["pickup_latitude"], 100, labels=False, retbins=True, include_lowest=True
)
train.loc[:, "pickuplong_no"], pick_long_bin = pd.cut(
    train["pickup_longitude"], 100, labels=False, retbins=True, include_lowest=True
)
train.loc[:, "dropofflat_no"], drop_lat_bin = pd.cut(
    train["dropoff_latitude"], 100, labels=False, retbins=True, include_lowest=True
)
train.loc[:, "dropofflong_no"], drop_long_bin = pd.cut(
    train["dropoff_longitude"], 100, labels=False, retbins=True, include_lowest=True
)
test.loc[:, "pickuplat_no"] = pd.cut(
    test["pickup_latitude"], pick_lat_bin, labels=False, include_lowest=True
)
test.loc[:, "pickuplong_no"] = pd.cut(
    test["pickup_longitude"], pick_long_bin, labels=False, include_lowest=True
)
test.loc[:, "dropofflat_no"] = pd.cut(
    test["dropoff_latitude"], drop_lat_bin, labels=False, include_lowest=True
)
test.loc[:, "dropofflong_no"] = pd.cut(
    test["dropoff_longitude"], drop_long_bin, labels=False, include_lowest=True
)



## === cell 21
for c in ["pickuplat_no", "pickuplong_no", "dropofflat_no", "dropofflong_no"]:
    train[c] = train[c].fillna(0)
    test[c] = test[c].fillna(0)

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



## === cell 22
print(train.shape, test.shape)



## === cell 23
train = train.drop(["fare_per_km_passenger", "fare_per_km"], axis=1)




## === cell 24
def calc_cwd_factor(df, col):
    new_df = (
        df.groupby(col)["key_num"].count().sort_values(ascending=False).reset_index()
    )
    new_df["cwd_factor"] = 1
    count = 1
    for i in range(1, new_df.shape[0]):
        count += 1
        if new_df.loc[i - 1, "key_num"] == new_df.loc[i, "key_num"]:
            count -= 1
        new_df.loc[i, "cwd_factor"] = count
    new_df.index = new_df[col]
    return new_df


fact_picklat = calc_cwd_factor(train, "pickuplat_no")
fact_picklong = calc_cwd_factor(train, "pickuplong_no")
fact_droplat = calc_cwd_factor(train, "dropofflat_no")
fact_droplong = calc_cwd_factor(train, "dropofflong_no")

train.loc[:, "pickuplat_cwd_factor"] = (
    pd.Series(train["pickuplat_no"]).map(fact_picklat["cwd_factor"]).fillna(1).values
)
train.loc[:, "pickuplong_cwd_factor"] = (
    pd.Series(train["pickuplong_no"]).map(fact_picklong["cwd_factor"]).fillna(1).values
)
train.loc[:, "dropofflat_cwd_factor"] = (
    pd.Series(train["dropofflat_no"]).map(fact_droplat["cwd_factor"]).fillna(1).values
)
train.loc[:, "dropofflong_cwd_factor"] = (
    pd.Series(train["dropofflong_no"]).map(fact_droplong["cwd_factor"]).fillna(1).values
)

test.loc[:, "pickuplat_cwd_factor"] = (
    pd.Series(test["pickuplat_no"]).map(fact_picklat["cwd_factor"]).fillna(1).values
)
test.loc[:, "pickuplong_cwd_factor"] = (
    pd.Series(test["pickuplong_no"]).map(fact_picklong["cwd_factor"]).fillna(1).values
)
test.loc[:, "dropofflat_cwd_factor"] = (
    pd.Series(test["dropofflat_no"]).map(fact_droplat["cwd_factor"]).fillna(1).values
)
test.loc[:, "dropofflong_cwd_factor"] = (
    pd.Series(test["dropofflong_no"]).map(fact_droplong["cwd_factor"]).fillna(1).values
)



## === cell 25
print(train.shape, test.shape)



## === cell 26
import sklearn
from sklearn import *
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle



## === cell 27
orig_train = train.copy()
orig_test = test.copy()



## === cell 28
train = orig_train.copy()
test = orig_test.copy()



## === cell 29
train = shuffle(train.iloc[:, :], random_state=42).reset_index(drop=True)
val = train.iloc[int(0.9 * train.shape[0]) :, :].copy()
train = train.iloc[: int(0.9 * train.shape[0]), :].copy()

train_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "abs_diff_longitude",
    "abs_diff_latitude",
    "dist_haversine_km",
]
test_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "abs_diff_longitude",
    "abs_diff_latitude",
    "dist_haversine_km",
]

transformer = StandardScaler().fit(train.loc[:, train_cols])
train.loc[:, train_cols] = transformer.transform(train.loc[:, train_cols])
val.loc[:, train_cols] = transformer.transform(val.loc[:, train_cols])
test.loc[:, test_cols] = transformer.transform(test.loc[:, test_cols])



## === cell 30
train_y = np.log1p(train["fare_amount"].astype(np.float64).values)
val_y = np.log1p(val["fare_amount"].astype(np.float64).values)

cols = [
    i
    for i in train.columns
    if i
    not in [
        "fare_amount",
        "key",
        "key_num",
        "pickup_datetime",
        "timestamp_with_key",
    ]
]

train_x = train.loc[:, cols]
val_x = val.loc[:, cols]

missing_in_test = [c for c in cols if c not in test.columns]
for c in missing_in_test:
    test[c] = 0
test_x = test.loc[:, cols]



## === cell 31
import xgboost as xgb
from xgboost import XGBRegressor



## === cell 32
xgbr = XGBRegressor(
    random_state=42,
    n_estimators=800,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="reg:squarederror",
    eval_metric="rmse",
    n_jobs=max(1, (os.cpu_count() or 2) - 1),
    tree_method="hist",
)

xgbr.fit(
    train_x,
    train_y,
    eval_set=[(val_x, val_y)],
    verbose=False,
    early_stopping_rounds=50,
)



## === cell 33
pred_train_log = xgbr.predict(train_x, iteration_range=(0, xgbr.best_iteration + 1))
pred_val_log = xgbr.predict(val_x, iteration_range=(0, xgbr.best_iteration + 1))
pred_test_log = xgbr.predict(test_x, iteration_range=(0, xgbr.best_iteration + 1))

pred_train = np.expm1(pred_train_log)
pred_val = np.expm1(pred_val_log)
pred_test = np.expm1(pred_test_log)



## === cell 34
from sklearn.metrics import mean_squared_error

rmse_train = np.sqrt(mean_squared_error(np.expm1(train_y), pred_train))
rmse_val = np.sqrt(mean_squared_error(np.expm1(val_y), pred_val))
print(rmse_train, rmse_val, "best_iteration:", xgbr.best_iteration)



## === cell 35
pred_test = np.maximum(pred_test, 0.0)

final = pd.DataFrame(
    {"key": test["timestamp_with_key"], "fare_amount": pred_test},
    columns=["key", "fare_amount"],
)
final.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", final.shape)



## === cell 36
final.head()
