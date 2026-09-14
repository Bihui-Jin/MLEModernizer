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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
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

3.84109

# 6. Current score

5.3345

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.15099) has done: 'I fix the runtime error caused by deprecated `np.object` by using the built-in `object` dtype check, which is score-neutral but unblocks execution. I also fix the `date_extraction` function bug where `drop(..., inplace=True)` returns `None` and inadvertently overwrites the dataframe, which otherwise breaks downstream feature engineering and training. Finally, I correct the Haversine feature computation (wrong longitude variable and swapped diffs, plus missing return), which is a minimal logic bugfix directly improving the distance feature and should move RMSE down toward your target without changing the model family or training approach. I keep the RandomForest setup intact and ensure a valid `key,fare_amount` submission CSV is written.'
- What this solution (achieved 5.35679) has done: 'Your current RMSE (5.15099) is worse than the target (3.84109), so we should make small, safe fixes that improve generalization without changing the model family or overall approach. The biggest score drag in this script is that it trains on many physically impossible/outlier trips (bad lat/long ranges, absurd passenger_count, extreme fares/distances), which RandomForest overfit and which hurts leaderboard RMSE; adding standard NYC Taxi Fare cleaning filters is a minimal, competition-relevant adjustment. I also fix a minor date feature bug (your “weekday” is actually day-of-month) and ensure the simple distance feature is actually written back to the DataFrames (right now you’re not assigning the returned DataFrame in the loop). These are narrow changes to preprocessing/feature correctness that typically move RMSE down toward ~3–4 without altering the core RandomForest training loop.'
- What this solution (achieved 5.55187) has done: 'Your current gap is 5.35679 − 3.84109 (lower is better), so we need a modest, safe RMSE reduction without changing the RandomForest approach. The biggest remaining score drag is that you’re still training on noisy/outlier examples that survive your filters, and your “distance_travelled/10e3” feature is a radians-based proxy that can mis-scale long/lat differences; we keep it but add a standard “manhattan distance” feature (sum of absolute deltas) and a simple “night” flag from the existing datetime features, which are minimal feature additions that typically reduce RMSE for this competition. We also make one correctness fix: you currently drop all rows where any column equals 0 (including `year`, `month`, etc. after feature engineering if the code order changes); we replace that with targeted coordinate/ passenger_count/fare cleaning only, which is directly relevant to the metric and avoids throwing away good rows. Finally, we ensure `key` handling stays consistent and the submission aligns exactly with the test order.'
- What this solution (achieved 5.46282) has done: 'Your current RMSE (5.55187) is worse than the target (3.84109), so we should make small, competition-standard preprocessing fixes that improve signal quality without changing the RandomForest approach. The biggest remaining score drag is that the engineered distance features are computed on raw coordinates without handling obvious bad/placeholder coordinates, and the model can also output negative fares; we add a minimal, targeted coordinate cleaning step and clip predictions to a valid non-negative range. We also add one very small, standard feature (`abs_lon_diff`, `abs_lat_diff`) derived from existing columns (no change to model family/training loop) and avoid unnecessary rounding of key distance features which can remove useful precision for trees. These changes typically reduce RMSE toward the ~3–4 range on this competition while keeping core logic intact and still producing the same `key,fare_amount` submission CSV.'
- What this solution (achieved 5.08022) has done: 'Your RMSE (5.46282) is worse than the target (3.84109), so we should make a small, competition-standard improvement that doesn’t change the model family or training loop. The biggest remaining issue is that the train/test feature distributions aren’t aligned: you clean out impossible/outlier trips only in `train`, but you leave `test` with potentially invalid coordinates/passenger_count, which causes the RandomForest to extrapolate badly and inflates leaderboard RMSE. I apply the same *coordinate/passenger_count validity filters* to `test` (without dropping rows; we impute invalid rows’ features to safe medians so every `key` is still predicted), and I also fill any NaNs created by datetime coercion using training medians to keep feature columns consistent. These are minimal, metric-relevant preprocessing fixes that typically move RMSE down toward the ~3–4 range while preserving your core logic.'
- What this solution (achieved 5.145) has done: 'We need to reduce RMSE (lower is better) from 5.08022 toward 3.84109, so we make minimal, competition-standard fixes that improve signal without changing the RandomForest approach. The biggest remaining drag is that the model is trained on raw dollars and gets disproportionately hurt by rare high-fare outliers; switching the target to a log1p scale (and inverting with expm1) is a small, standard regression transformation that typically reduces RMSE on this dataset while keeping the same model/loop/loss semantics. We also enforce a consistent, safe numeric dtype (float32) for feature matrices to stabilize splits/thresholds and avoid pandas object edge cases, without changing features. Finally, we keep your existing cleaning/imputation and ensure the submission stays `key,fare_amount` aligned to the original test order.'
- What this solution (achieved 5.36699) has done: 'To move RMSE down from 5.145 toward 3.84109 (lower is better) with minimal disruption, I keep your RandomForest + existing features and instead fix two metric-relevant issues: (1) you currently train on distance outliers and some remaining noisy examples, so I add a very standard “fare-per-km” sanity filter and a couple of mild distance bounds that typically improve generalization for this competition; (2) your model is trained on `log1p(fare)` but you still allow very large predicted fares, so I clip predictions to the same fare range you trained on (0–250) to reduce RMSE impact from rare extreme predictions. I also ensure the train/test matrices share identical feature columns (same order and missing-column handling) to prevent silent train/test mismatch. These are small, competition-standard preprocessing/postprocessing adjustments that preserve your core model and training loop while aiming to reduce the leaderboard error.'
- What this solution (achieved 5.35347) has done: 'We need to reduce RMSE from 5.36699 toward the 3.84109 target (lower is better), so the smallest likely win is to stop training the RandomForest on a mis-scaled distance proxy. Right now `distance_travelled/10e3` is radians-based and then multiplied by 1000, which makes it not comparable to km-based `harvesine/km` and can confuse tree splits; I change it to a simple, correctly-scaled equirectangular distance in **kilometers** while keeping the same feature name so the rest of your pipeline stays intact. I also ensure the same feature computation applies identically to train/test and keep your existing cleaning, log1p target transform, model, and submission format unchanged. These are minimal, metric-relevant fixes that typically reduce error without altering your overall approach.'
- What this solution (achieved 5.35347) has done: 'Your RMSE (5.35347, lower-is-better) is still well above the 3.84109 target, so we should make a small, safe improvement that preserves your RandomForest approach and features. The biggest current issue is that `year/month/weekday/hour/is_night` are being fed to the forest as continuous floats, which makes split thresholds arbitrary and often hurts generalization; we keep the same features but cast them to integer categories consistently in both train and test. Additionally, your `distance_travelled/10e3` feature name no longer matches its scale (it’s already km), and you’re duplicating essentially the same distance signal twice; we keep the feature but stabilize it by clipping extreme values and ensuring no NaNs/inf propagate after cleaning. These changes are minimal, metric-relevant, and should move RMSE downward toward the target without changing the model family, training loop, or loss.'
- What this solution (achieved 5.3345) has done: 'The timeout is dominated by reading and sorting the full 55M-row `train.csv`, then doing repeated full-data scans (describe/null checks/zero counts) and expensive pandas operations on huge frames before downsampling. The key speed fix is to downsample *during CSV read* (deterministically) and avoid sorting the entire dataset, while keeping the same “evenly spaced over file order” sampling semantics and all downstream feature engineering/modeling unchanged. I also replace repeated diagnostic cells’ full scans with cheap shape/dtype-only outputs (they don’t affect training) and make feature functions operate via fast NumPy arrays to reduce pandas overhead without changing computed values. RandomForest training/prediction logic and feature set remain identical.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing
import os

INPUT_DIR = "../input"
if not os.path.exists(INPUT_DIR):
    INPUT_DIR = "/kaggle/input"

print("Listing input dir:", INPUT_DIR)
print(os.listdir(INPUT_DIR))



## === cell 1
TRAIN_PATH = f"{INPUT_DIR}/train.csv"
TEST_PATH = f"{INPUT_DIR}/test.csv"

n_target = 1_000_000

import csv

with open(TRAIN_PATH, "r", newline="") as f:
    n_rows_total = sum(1 for _ in f) - 1  # excluding header

if n_rows_total > n_target:
    idx = np.linspace(0, n_rows_total - 1, n_target).round().astype(np.int64)
    idx_set = set(idx.tolist())

    def _skip_fn(i):
        if i == 0:
            return False
        return (i - 1) not in idx_set

    train = pd.read_csv(
        TRAIN_PATH,
        parse_dates=["pickup_datetime"],
        skiprows=_skip_fn,
    )
else:
    train = pd.read_csv(TRAIN_PATH, parse_dates=["pickup_datetime"])

train = train.dropna(subset=["pickup_datetime"]).sort_values("pickup_datetime")

test = pd.read_csv(TEST_PATH)
train.head()



## === cell 2
test.head()



## === cell 3
train.shape



## === cell 4
test.shape



## === cell 5
train.dtypes



## === cell 6
test.dtypes



## === cell 7
train = train.dropna()
train.isnull().sum()



## === cell 8
_ = None



## === cell 9
_ = None



## === cell 10
train.shape



## === cell 11
train.head(3)



## === cell 12
train.head(3)



## === cell 13
train.dtypes.value_counts()



## === cell 14
object_data = train.dtypes == object
categoricals = train.columns[object_data]
categoricals



## === cell 15
train.head()



## === cell 16
import datetime as dt


def date_extraction(data):
    data["pickup_datetime"] = pd.to_datetime(data["pickup_datetime"], errors="coerce")
    dt_series = data["pickup_datetime"].dt
    data["year"] = dt_series.year
    data["month"] = dt_series.month
    data["weekday"] = dt_series.weekday
    data["hour"] = dt_series.hour
    data["is_night"] = ((data["hour"] <= 6) | (data["hour"] >= 20)).astype(np.int8)
    data.drop("pickup_datetime", axis=1, inplace=True)
    return data


train = date_extraction(train)



## === cell 17
train.head()



## === cell 18
test = date_extraction(test)
test.head()




## === cell 19
def long_lat_distance(x):
    plon = x["pickup_longitude"].to_numpy()
    dlonv = x["dropoff_longitude"].to_numpy()
    plat = x["pickup_latitude"].to_numpy()
    dlatv = x["dropoff_latitude"].to_numpy()

    dlon = np.radians(plon - dlonv)
    dlat = np.radians(plat - dlatv)

    x["Longitude_distance"] = dlon
    x["Latitude_distance"] = dlat

    lat_avg = np.radians((plat + dlatv) / 2.0)
    r_km = 6371.0
    x["distance_travelled/10e3"] = r_km * np.sqrt(
        (dlon * np.cos(lat_avg)) ** 2 + dlat**2
    )
    return x




## === cell 20
train = long_lat_distance(train)
test = long_lat_distance(test)

train.head()




## === cell 21
def harvesine(x):
    r = 6371000  # meters

    plat = x["pickup_latitude"].to_numpy()
    dlat = x["dropoff_latitude"].to_numpy()
    plon = x["pickup_longitude"].to_numpy()
    dlon = x["dropoff_longitude"].to_numpy()

    theta_1 = np.radians(plat)
    theta_2 = np.radians(dlat)
    lambda_1 = np.radians(plon)
    lambda_2 = np.radians(dlon)

    theta_diff = theta_2 - theta_1
    lambda_diff = lambda_2 - lambda_1

    a = (np.sin(theta_diff / 2) ** 2) + (
        np.cos(theta_1) * np.cos(theta_2) * (np.sin(lambda_diff / 2) ** 2)
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    x["harvesine/km"] = (r * c) / 1000.0
    return x




## === cell 22
train = harvesine(train)
test = harvesine(test)

train.head()



## === cell 23
train["manhattan_dist"] = (
    train["pickup_longitude"] - train["dropoff_longitude"]
).abs() + (train["pickup_latitude"] - train["dropoff_latitude"]).abs()
test["manhattan_dist"] = (
    test["pickup_longitude"] - test["dropoff_longitude"]
).abs() + (test["pickup_latitude"] - test["dropoff_latitude"]).abs()

train["abs_lon_diff"] = (train["pickup_longitude"] - train["dropoff_longitude"]).abs()
train["abs_lat_diff"] = (train["pickup_latitude"] - train["dropoff_latitude"]).abs()
test["abs_lon_diff"] = (test["pickup_longitude"] - test["dropoff_longitude"]).abs()
test["abs_lat_diff"] = (test["pickup_latitude"] - test["dropoff_latitude"]).abs()

train["abs_hour_diff"] = (train["hour"] - 12).abs()
test["abs_hour_diff"] = (test["hour"] - 12).abs()

train.dtypes.value_counts()



## === cell 24
train.head()



## === cell 25
test.head()



## === cell 26
train.head(3)



## === cell 27
print("Are there any nulls\nan in the train data: ")
print(train.isnull().sum())

print("\nAre there any nulls\nans in the test data: ")
print(test.isnull().sum())



## === cell 28
datetime_int_cols = ["year", "month", "weekday", "hour", "is_night"]
for c in datetime_int_cols:
    if c in train.columns:
        train[c] = train[c].astype("Int64")
    if c in test.columns:
        test[c] = test[c].astype("Int64")

for df in (train, test):
    for c in [
        "harvesine/km",
        "distance_travelled/10e3",
        "manhattan_dist",
        "abs_lon_diff",
        "abs_lat_diff",
        "abs_hour_diff",
    ]:
        if c in df.columns:
            df[c] = df[c].replace([np.inf, -np.inf], np.nan)

median_fill_cols = [
    "harvesine/km",
    "manhattan_dist",
    "abs_lon_diff",
    "abs_lat_diff",
    "distance_travelled/10e3",
    "abs_hour_diff",
    "year",
    "month",
    "weekday",
    "hour",
]
for col in median_fill_cols:
    if col in train.columns:
        med = train[col].median()
        train[col] = train[col].fillna(med)
        if col in test.columns:
            test[col] = test[col].fillna(med)

if "is_night" in train.columns:
    night_mode = (
        int(train["is_night"].mode().iloc[0]) if train["is_night"].notna().any() else 0
    )
    train["is_night"] = train["is_night"].fillna(night_mode)
    if "is_night" in test.columns:
        test["is_night"] = test["is_night"].fillna(night_mode)

for c in datetime_int_cols:
    if c in train.columns:
        train[c] = train[c].astype(np.int16)
    if c in test.columns:
        test[c] = test[c].astype(np.int16)



## === cell 29
train = train[(train["fare_amount"] > 0) & (train["fare_amount"] <= 250)].copy()
train = train[(train["passenger_count"] >= 1) & (train["passenger_count"] <= 6)].copy()

train = train[
    (train["pickup_longitude"].between(-74.5, -72.8))
    & (train["dropoff_longitude"].between(-74.5, -72.8))
    & (train["pickup_latitude"].between(40.5, 41.8))
    & (train["dropoff_latitude"].between(40.5, 41.8))
].copy()

train = train[
    (train["pickup_longitude"] != 0)
    & (train["dropoff_longitude"] != 0)
    & (train["pickup_latitude"] != 0)
    & (train["dropoff_latitude"] != 0)
].copy()

train = train[(train["harvesine/km"] >= 0) & (train["harvesine/km"] <= 200)].copy()
train = train[~((train["harvesine/km"] < 0.05) & (train["fare_amount"] > 50))].copy()
train = train[~((train["harvesine/km"] > 50) & (train["fare_amount"] < 2.5))].copy()

eps = 1e-3
fare_per_km = train["fare_amount"] / (train["harvesine/km"] + eps)
train = train[(fare_per_km <= 200) & (fare_per_km >= 0)].copy()
train = train[train["harvesine/km"] <= 80].copy()

if "distance_travelled/10e3" in train.columns:
    train["distance_travelled/10e3"] = train["distance_travelled/10e3"].clip(0, 80)
if "distance_travelled/10e3" in test.columns:
    test["distance_travelled/10e3"] = test["distance_travelled/10e3"].clip(0, 80)

train.shape



## === cell 30
valid_test = (
    (test["passenger_count"].between(1, 6))
    & (test["pickup_longitude"].between(-74.5, -72.8))
    & (test["dropoff_longitude"].between(-74.5, -72.8))
    & (test["pickup_latitude"].between(40.5, 41.8))
    & (test["dropoff_latitude"].between(40.5, 41.8))
    & (test["pickup_longitude"] != 0)
    & (test["dropoff_longitude"] != 0)
    & (test["pickup_latitude"] != 0)
    & (test["dropoff_latitude"] != 0)
    & (test["harvesine/km"].between(0, 200))
)

feature_impute_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "Longitude_distance",
    "Latitude_distance",
    "distance_travelled/10e3",
    "harvesine/km",
    "manhattan_dist",
    "abs_lon_diff",
    "abs_lat_diff",
    "abs_hour_diff",
    "year",
    "month",
    "weekday",
    "hour",
    "is_night",
]
for col in feature_impute_cols:
    if col in test.columns and col in train.columns:
        fill_val = (
            train[col].median() if col != "is_night" else int(train[col].mode().iloc[0])
        )
        test.loc[~valid_test, col] = fill_val



## === cell 31
from sklearn.ensemble import RandomForestRegressor

feature_cols = [x for x in train.columns if x not in ["fare_amount", "key"]]
X = train[feature_cols]
y = train["fare_amount"]



## === cell 32
X_corr = X.astype(np.float32, copy=False)
correlations = X_corr.corrwith(y)
correlations = abs(correlations * 100)
correlations.sort_values(ascending=False, inplace=True)

correlations



## === cell 33
ax = correlations.plot(kind="bar")
ax.set(ylim=[-1, 1], ylabel="pearson correlation")



## === cell 34
train.head()



## === cell 35
train_1 = train.drop(
    [
        "pickup_longitude",
        "dropoff_longitude",
        "pickup_latitude",
        "dropoff_latitude",
        "Longitude_distance",
        "Latitude_distance",
    ],
    axis=1,
)

train_1.head()



## === cell 36
train_1.head()



## === cell 37
train_1.head(3)



## === cell 38
test_1 = test.drop(
    [
        "pickup_longitude",
        "dropoff_longitude",
        "pickup_latitude",
        "dropoff_latitude",
        "Longitude_distance",
        "Latitude_distance",
    ],
    axis=1,
)

test_1.head()



## === cell 39
from sklearn.model_selection import train_test_split

feat_cols = [x for x in train_1.columns if x not in ["fare_amount", "key"]]
X_1 = train_1[feat_cols]
y_1 = train_1["fare_amount"]

y_1_log = np.log1p(y_1)

X_1 = X_1.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X_1, y_1_log, test_size=0.25, random_state=42
)



## === cell 40
rf = RandomForestRegressor(n_estimators=100, max_features=5, random_state=42, n_jobs=-1)
rf = rf.fit(X_train, y_train)



## === cell 41
test.head()



## === cell 42
final_prediction = rf.predict(X_test)



## === cell 43
test_features = test_1.reindex(columns=feat_cols).copy()
for c in feat_cols:
    if c not in test_features.columns:
        test_features[c] = train_1[c].median()
test_features = test_features[feat_cols].astype(np.float32)

test_features.head()



## === cell 44
final_prediction = rf.predict(test_features)
final_prediction = np.expm1(final_prediction)

final_prediction = np.clip(final_prediction, 0.0, 250.0)

NYCtaxiFare_submission = pd.DataFrame(
    {"key": test["key"].values, "fare_amount": final_prediction}
)
NYCtaxiFare_submission.to_csv("NYCtaxiFare_prediction.csv", index=False)



## === cell 45
NYCtaxiFare_submission.head()
