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
geopy==2.4.1
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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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

4.03634

# 6. Current score

7.73147

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.40111) has done: 'I fix the runtime error caused by `geopy.distance.VincentyDistance` (removed in modern geopy) by switching to a vectorized Haversine distance computation, which is both compatible and much faster. Then I ensure LightGBM is trained only on numeric feature columns (your intended `column_list`) to resolve the “bad pandas dtypes” error and keep the same modeling approach. Finally, I make sure the test set gets the same engineered features and that `submission.csv` is always written with the required `key,fare_amount` columns.'
- What this solution (achieved 5.77439) has done: 'To move RMSE down toward your 4.03634 target (from 5.40111), I keep the same LightGBM regressor and the same basic feature set, but fix two high-impact data issues that currently hurt accuracy: (1) `key2` is being parsed from `key` (which includes a trailing integer), producing many NaT values and thus broken time features; we should parse `pickup_datetime` instead. (2) We should remove obviously bad training rows (negative/too-large fares, zero/invalid passenger_count, and NYC-bounding-box filtering) so the model doesn’t learn from noise/outliers, while keeping the same training loop and submission format. These are minimal, metric-aligned preprocessing fixes and should reduce RMSE without changing the model architecture/training approach.'
- What this solution (achieved 5.89729) has done: 'Your gap to target is large (RMSE 5.77439 vs 4.03634; lower is better), so we should improve model signal without changing the core model/training approach. The biggest accuracy issue left is that `key2` can be `NaT` (or time features become NaN) and LightGBM then learns from missing/noisy time features; we make time parsing robust and fill derived time features deterministically. Next, we remove a small set of remaining extreme outliers that your current filters miss (especially near-zero/zero-distance rides and very large distances), because they disproportionately hurt RMSE while keeping the same feature set and LightGBM. Finally, we ensure train/test use identical preprocessing (same NaN handling after feature engineering) so the submission is consistent and stable.'
- What this solution (achieved 6.09643) has done: 'We need to move RMSE down (lower is better) from 5.89729 toward 4.03634, so we make small, metric-aligned data fixes without changing your LightGBM model or feature set. The biggest gain with minimal disruption is to remove remaining high-noise rows that inflate RMSE: (1) enforce a realistic fare-per-km band (keeps the same features/model, just improves label quality), and (2) tighten passenger_count handling by dropping missing/zero rather than imputing to 1 in train (imputation can inject wrong labels). We also make the datetime parsing deterministic (no timezone ambiguity) and ensure both train/test derived time features are integer-filled consistently so LightGBM sees stable inputs. Everything else (Haversine distance, the same `column_list`, LightGBM hyperparameters, training loop, and submission format) stays the same.'
- What this solution (achieved 6.11608) has done: 'We need to reduce RMSE from 6.09643 toward 4.03634 (lower is better), so we make only minimal, metric-aligned data fixes without changing your LightGBM model, its hyperparameters, or the feature set. The main issue is that the current training sample includes many mislabeled/noisy rows due to missing `dropoff_latitude/longitude` (common in NYC Taxi Fare data); those rows pass your current validity checks because comparisons with NaN evaluate False, and they end up with NaN distances that get filled to 0, creating inconsistent training signal. We explicitly drop any rows with missing geo coordinates and missing `pickup_datetime` before engineering distance/time features, and we avoid imputing `distance` to 0 (we instead keep only rows with valid distance and fill test missing distances with the median). We also make training deterministic (`random_state`) to stabilize results; this doesn’t change the core logic but reduces score variance.'
- What this solution (achieved 6.11608) has done: 'We need to move RMSE down (lower is better) from 6.11608 toward 4.03634, so the smallest impactful change is to stop unintentionally leaking non-feature columns into training and to align the model’s final fit with the full filtered training data. Concretely, your second `fit()` (cell 36) currently trains only on the reduced `X_train` split again, which weakens the final model used for the test prediction; we instead train on the full cleaned training set before predicting test. Additionally, we ensure `X_train` never carries non-numeric columns like `key`, `pickup_datetime`, and `key2` (even if later subset), to avoid subtle alignment/dtype issues and keep preprocessing consistent. These are minimal, metric-aligned fixes that keep the same LightGBM model, hyperparameters, features, and training approach, but should reduce RMSE.'
- What this solution (achieved 7.73147) has done: 'Your code currently drops test rows outside the NYC bounding box, which create a submission with fewer than 9914 rows and typically leads to an invalid Kaggle submission (or a very poor score if somehow accepted). I keep the same LightGBM model and the same feature engineering, but ensure the test set keeps all rows by only filtering train (and instead clipping/repairing test coordinates so distance features are still computable). I also make datetime parsing for test robust by dropping rows with invalid `pickup_datetime` (rare) only if needed and aligning features so `submission.csv` always matches `sample_submission.csv` keys/rowcount. These changes are minimal, metric-aligned, and should move RMSE down toward your 4.03634 target from “no score”.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use("fivethirtyeight")

import geopy.distance  # kept to preserve original imports, but not used for distance anymore

import os
import gc

print(os.listdir("../input"))




## === cell 1
def load_Data():
    train = pd.read_csv("../input/train.csv", nrows=1_000_000, low_memory=True)
    test = pd.read_csv("../input/test.csv", low_memory=True)
    return train, test




## === cell 2
train, test = load_Data()



## === cell 3
train.head(5)



## === cell 4
train.describe()



## === cell 5
train.info()



## === cell 6
train.isnull().sum()



## === cell 7
core_cols = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
train = train.dropna(subset=core_cols).copy()

test = test.dropna(
    subset=[
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]
).copy()

train["key2"] = pd.to_datetime(
    train["pickup_datetime"], errors="coerce", utc=True
).dt.tz_convert(None)
train = train[train["key2"].notna()].copy()
train["key2"].head()
train.info()



## === cell 8
test["key2"] = pd.to_datetime(
    test["pickup_datetime"], errors="coerce", utc=True
).dt.tz_convert(None)
test = test[test["key2"].notna()].copy()



## === cell 9
train["fare_amount"].plot(kind="box")



## === cell 10
gc.collect()
train.describe()



## === cell 11
print(
    "% of fares above 25$ - {:0.2f}".format(
        train[train["fare_amount"] > 25]["key"].count() * 100 / train["key"].count()
    )
)
print(
    "% of fares above 50$ - {:0.2f}".format(
        train[train["fare_amount"] > 50]["key"].count() * 100 / train["key"].count()
    )
)
print(
    "% of fares above 100$ - {:0.2f}".format(
        train[train["fare_amount"] > 100]["key"].count() * 100 / train["key"].count()
    )
)
print(
    "% of fares below 0$ - {:0.2f}".format(
        train[train["fare_amount"] < 0]["key"].count() * 100 / train["key"].count()
    )
)



## === cell 12
fig, axarr = plt.subplots(2, 2, figsize=(20, 10))

train[~(train["fare_amount"] > 25)]["fare_amount"].plot(kind="box", ax=axarr[0][0])
train[~(train["fare_amount"] > 50)]["fare_amount"].plot(kind="box", ax=axarr[0][1])
train[~(train["fare_amount"] > 100)]["fare_amount"].plot(kind="box", ax=axarr[1][0])
train[~(train["fare_amount"] < 0)]["fare_amount"].plot(kind="box", ax=axarr[1][1])



## === cell 13
train["passenger_count"].plot(kind="box")



## === cell 14
print(
    "Count of invalid pickup latitude",
    train[(train["pickup_latitude"] > 90) | (train["pickup_latitude"] < -90)][
        "pickup_latitude"
    ].count(),
)
print(
    "Count of invalid dropoff latitude",
    train[(train["dropoff_latitude"] > 90) | (train["dropoff_latitude"] < -90)][
        "dropoff_latitude"
    ].count(),
)
print(
    "Count of invalid pickup longitude",
    train[(train["pickup_longitude"] > 180) | (train["pickup_longitude"] < -180)][
        "pickup_longitude"
    ].count(),
)
print(
    "Count of invalid dropoff longitude",
    train[(train["dropoff_longitude"] > 180) | (train["dropoff_longitude"] < -180)][
        "dropoff_longitude"
    ].count(),
)



## === cell 15
print(
    "Count of invalid pickup latitude",
    test[(test["pickup_latitude"] > 90) | (test["pickup_latitude"] < -90)][
        "pickup_latitude"
    ].count(),
)
print(
    "Count of invalid dropoff latitude",
    test[(test["dropoff_latitude"] > 90) | (test["dropoff_latitude"] < -90)][
        "dropoff_latitude"
    ].count(),
)
print(
    "Count of invalid pickup longitude",
    test[(test["pickup_longitude"] > 180) | (test["pickup_longitude"] < -180)][
        "pickup_longitude"
    ].count(),
)
print(
    "Count of invalid dropoff longitude",
    test[(test["dropoff_longitude"] > 180) | (test["dropoff_longitude"] < -180)][
        "dropoff_longitude"
    ].count(),
)



## === cell 16
train = train[(train["fare_amount"] > 0) & (train["fare_amount"] <= 250)]
train = train[train["passenger_count"].notna()].copy()
train = train[(train["passenger_count"] >= 1) & (train["passenger_count"] <= 6)]

train = train[~((train["pickup_latitude"] > 90) | (train["pickup_latitude"] < -90))]
train = train[~((train["dropoff_latitude"] > 90) | (train["dropoff_latitude"] < -90))]
train = train[~((train["pickup_longitude"] > 180) | (train["pickup_longitude"] < -180))]
train = train[
    ~((train["dropoff_longitude"] > 180) | (train["dropoff_longitude"] < -180))
]

nyc = {
    "lon_min": -74.5,
    "lon_max": -72.8,
    "lat_min": 40.5,
    "lat_max": 41.8,
}

train = train[
    (train["pickup_longitude"].between(nyc["lon_min"], nyc["lon_max"]))
    & (train["dropoff_longitude"].between(nyc["lon_min"], nyc["lon_max"]))
    & (train["pickup_latitude"].between(nyc["lat_min"], nyc["lat_max"]))
    & (train["dropoff_latitude"].between(nyc["lat_min"], nyc["lat_max"]))
].copy()

for c, lo, hi in [
    ("pickup_longitude", nyc["lon_min"], nyc["lon_max"]),
    ("dropoff_longitude", nyc["lon_min"], nyc["lon_max"]),
    ("pickup_latitude", nyc["lat_min"], nyc["lat_max"]),
    ("dropoff_latitude", nyc["lat_min"], nyc["lat_max"]),
]:
    test[c] = test[c].clip(lower=lo, upper=hi)




## === cell 17
def haversine_km(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1.astype(float))
    lat1 = np.radians(lat1.astype(float))
    lon2 = np.radians(lon2.astype(float))
    lat2 = np.radians(lat2.astype(float))

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    R = 6371.0  # Earth radius in km
    return R * c


for df in (train, test):
    df["abs_lon_diff"] = (df["dropoff_longitude"] - df["pickup_longitude"]).abs()
    df["abs_lat_diff"] = (df["dropoff_latitude"] - df["pickup_latitude"]).abs()

train["distance"] = haversine_km(
    train["pickup_longitude"],
    train["pickup_latitude"],
    train["dropoff_longitude"],
    train["dropoff_latitude"],
)

test["distance"] = haversine_km(
    test["pickup_longitude"],
    test["pickup_latitude"],
    test["dropoff_longitude"],
    test["dropoff_latitude"],
)

KM_PER_DEG_LAT = 111.32
for df in (train, test):
    mean_lat = 0.5 * (
        df["pickup_latitude"].astype(float) + df["dropoff_latitude"].astype(float)
    )
    km_per_deg_lon = KM_PER_DEG_LAT * np.cos(np.radians(mean_lat))
    df["manhattan_km"] = (df["abs_lat_diff"].astype(float) * KM_PER_DEG_LAT) + (
        df["abs_lon_diff"].astype(float) * km_per_deg_lon
    )



## === cell 18
train.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)



## === cell 19
test.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)



## === cell 20
print(
    "% of trips above 25 KM - {:0.2f}".format(
        train[train["distance"] > 25]["key"].count() * 100 / train.count()["key"]
    )
)



## === cell 21
train = train.replace([np.inf, -np.inf], np.nan).copy()
train = train[train["distance"].notna()].copy()

train = train[(train["distance"] > 0.05) & (train["distance"] < 60)].copy()

fare_per_km = train["fare_amount"] / train["distance"]
train = train[(fare_per_km >= 0.5) & (fare_per_km <= 50.0)].copy()
del fare_per_km



## === cell 22
train["year"] = train["key2"].dt.year
train["month"] = train["key2"].dt.month
train["day"] = train["key2"].dt.day
train["day of week"] = train["key2"].dt.weekday
train["hour"] = train["key2"].dt.hour



## === cell 23
test["year"] = test["key2"].dt.year
test["month"] = test["key2"].dt.month
test["day"] = test["key2"].dt.day
test["day of week"] = test["key2"].dt.weekday
test["hour"] = test["key2"].dt.hour



## === cell 24
train_dist_median = float(train["distance"].median())
train_manh_median = float(
    train["manhattan_km"].replace([np.inf, -np.inf], np.nan).median()
)
train_abs_lon_median = float(
    train["abs_lon_diff"].replace([np.inf, -np.inf], np.nan).median()
)
train_abs_lat_median = float(
    train["abs_lat_diff"].replace([np.inf, -np.inf], np.nan).median()
)

for df in (train, test):
    df["passenger_count"] = df["passenger_count"].fillna(1).astype(np.int16)
    for c in ["year", "month", "day", "day of week", "hour"]:
        df[c] = df[c].fillna(-1).astype(np.int16)

train["distance"] = train["distance"].astype(np.float32)
test["distance"] = test["distance"].fillna(train_dist_median).astype(np.float32)

train["manhattan_km"] = (
    train["manhattan_km"].replace([np.inf, -np.inf], np.nan).astype(np.float32)
)
test["manhattan_km"] = (
    test["manhattan_km"]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(train_manh_median)
    .astype(np.float32)
)

train["abs_lon_diff"] = (
    train["abs_lon_diff"].replace([np.inf, -np.inf], np.nan).astype(np.float32)
)
test["abs_lon_diff"] = (
    test["abs_lon_diff"]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(train_abs_lon_median)
    .astype(np.float32)
)

train["abs_lat_diff"] = (
    train["abs_lat_diff"].replace([np.inf, -np.inf], np.nan).astype(np.float32)
)
test["abs_lat_diff"] = (
    test["abs_lat_diff"]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(train_abs_lat_median)
    .astype(np.float32)
)



## === cell 25
column_list = [
    "passenger_count",
    "distance",
    "manhattan_km",
    "abs_lon_diff",
    "abs_lat_diff",
    "year",
    "month",
    "day",
    "day of week",
    "hour",
]
y_train = train["fare_amount"]
X_train = train[column_list].copy()



## === cell 26
X_test = test[column_list].copy()



## === cell 27
X_train.shape, y_train.shape



## === cell 28
from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42
)



## === cell 29
X_tr.shape, X_val.shape, y_tr.shape, y_val.shape



## === cell 30
from lightgbm import LGBMRegressor



## === cell 31
lgb = LGBMRegressor(
    boosting_type="gbdt",
    class_weight=None,
    colsample_bytree=0.9,
    learning_rate=0.1,
    max_depth=8,
    min_child_samples=55,
    min_child_weight=0.001,
    min_split_gain=0.1,
    n_estimators=500,
    n_jobs=-1,
    num_leaves=45,
    random_state=42,
    reg_alpha=5.0,
    reg_lambda=3.0,
    silent=True,
    subsample=1.0,
    subsample_for_bin=200000,
    subsample_freq=1,
)



## === cell 32
lgb.fit(X_tr, y_tr)



## === cell 33
pred = lgb.predict(X_val)



## === cell 34
from sklearn.metrics import r2_score

r2_score(y_val, pred)



## === cell 35
lgb.fit(X_train, y_train)




## === cell 36
def display_importances(feature_importance_df_, doWorst=False, n_feat=50):
    if not doWorst:
        cols = (
            feature_importance_df_[["feature", "importance"]]
            .groupby("feature")
            .mean()
            .sort_values(by="importance", ascending=False)[:n_feat]
            .index
        )
    else:
        cols = (
            feature_importance_df_[["feature", "importance"]]
            .groupby("feature")
            .mean()
            .sort_values(by="importance", ascending=False)[-n_feat:]
            .index
        )

    mean_imp = (
        feature_importance_df_[["feature", "importance"]].groupby("feature").mean()
    )
    df_2_neglect = mean_imp[mean_imp["importance"] < 1e-3]
    print("The list of features with 0 importance: ")
    print(df_2_neglect.index.values.tolist())
    del mean_imp, df_2_neglect

    best_features = feature_importance_df_.loc[
        feature_importance_df_.feature.isin(cols)
    ]

    plt.figure(figsize=(8, 10))
    sns.barplot(
        x="importance",
        y="feature",
        data=best_features.sort_values(by="importance", ascending=False),
    )
    plt.title("LightGBM Features")
    plt.tight_layout()
    plt.savefig("lgbm_importances.png")


importance_df = pd.DataFrame()
importance_df["feature"] = column_list
importance_df["importance"] = lgb.feature_importances_
display_importances(feature_importance_df_=importance_df, n_feat=20)



## === cell 37
y_pred = lgb.predict(X_test)



## === cell 38
y_pred = np.maximum(y_pred, 0.0)

sample_sub = pd.read_csv("../input/sample_submission.csv")
pred_df = pd.DataFrame({"key": test["key"].values, "fare_amount": y_pred})
submission = sample_sub[["key"]].merge(pred_df, on="key", how="left")

submission["fare_amount"] = submission["fare_amount"].fillna(float(np.mean(y_pred)))

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
