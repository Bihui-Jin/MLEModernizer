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

5.43674

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.40359) has done: 'Diagnosis: Cell 20 crashes because `geopy.distance.VincentyDistance` was removed from modern `geopy` (it no longer exists in geopy 2.x). The training set in cell 18 already uses `geopy.distance.geodesic(...).km`, so test-time distance must be computed with the same available API to keep feature semantics consistent.  

Patch summary: In cell 20 only, replace the call to the removed `VincentyDistance` with `geodesic` using the same coordinate ordering and `.km` as used for train. This is the minimal change that restores execution and preserves the intended “distance in kilometers” feature.  

Updated cells: Cell 20 only (buggy cell).  

Compatibility notes for cell k+1: `test['distance']` remains a numeric column and the original latitude/longitude columns remain present so cell 21 can drop them unchanged.  

Assumptions: Using `geodesic` as a drop-in replacement for Vincenty is acceptable for this notebook’s feature definition; it keeps distance units and overall logic consistent with the train feature in cell 18.'
- What this solution (achieved 5.40344) has done: 'Your current gap to the target is large (RMSE 5.40359 vs 4.03634; lower is better), so we need a small, legitimate accuracy improvement without changing the model or feature set. The biggest issue hurting RMSE here is that `key2` is parsed from `key` (which contains a trailing unique integer), producing many `NaT` values and therefore broken time features; we instead parse `pickup_datetime` (the true timestamp) for both train and test. We also ensure the train/validation split is deterministic (seeded) and remove the redundant second fit so the final model is trained on all available training rows before predicting test. These are minimal changes that preserve the core LightGBM approach, the same features, and the same evaluation semantics while improving feature correctness and stability.'
- What this solution (achieved 5.90289) has done: 'Your current RMSE (5.40344) is worse than the target (4.03634), so we should improve accuracy with minimal, low-risk changes that keep the same LightGBM model and the same feature set. The biggest remaining issue is label/feature noise: the code fills missing values with 0 before datetime parsing and never removes obviously invalid rows (negative/huge fares, passenger_count out of range, zero coordinates, or unrealistic distances), which severely harms RMSE. I add a small, standard NYC-taxi cleaning step (only filtering training rows) and keep all existing feature engineering and the same model hyperparameters. This should reduce noise and move RMSE toward the target without changing the core approach.'
- What this solution (achieved 5.93882) has done: 'I make two minimal, score-relevant fixes: (1) ensure the model is trained on the intended feature matrix (right now `X_train` still contains non-feature columns like `key` and `key2`, which can degrade validation training and also make behavior inconsistent), and (2) clip negative/very small predictions to a small positive value because fares are non-negative and this often reduces RMSE on this competition without changing the model/feature logic. I also make the LightGBM training deterministic (set `random_state`) to stabilize the score while keeping the same algorithm and hyperparameters. All paths and the overall approach (data load → clean → distance → datetime features → LightGBM → submission.csv) remain the same.'
- What this solution (achieved 5.91055) has done: 'Your current RMSE (5.93882) is worse than the target (4.03634), so we should improve accuracy with minimal, low-risk changes while keeping the same LightGBM model and the same feature set. The biggest remaining error source is that distance is computed with a very slow row-wise `geodesic` apply, which forces you to train on only 1M rows and can also be unstable; replacing it with a vectorized haversine distance preserves the “distance in km” feature semantics but runs much faster, letting us safely increase `nrows` to use more training data within the same time budget. I keep the exact same downstream columns, split, LightGBM hyperparameters, and submission format; only distance computation and the training sample size change. This should move RMSE down toward the target without altering the overall approach.'
- What this solution (achieved 5.96662) has done: 'Your RMSE (5.91055) is still far from the target (4.03634; lower is better), so we should make a small, legitimate accuracy improvement without changing the LightGBM model or the feature set. The biggest remaining issue is that `distance` can be wildly wrong for out-of-NYC coordinates and the model is trained without any scaling/log transform, so those outliers inflate RMSE; we add a standard geographic bounding-box filter for NYC area and remove extreme `distance` outliers only in the training data (test remains untouched). This keeps your core pipeline identical (same features, same model, same training approach) but reduces label/feature noise that typically hurts this competition. We also apply the same missing/zero coordinate filtering to `test` (not dropping rows, just leaving them as-is after distance calc) to avoid NaNs/infs and keep submission row count aligned.'
- What this solution (achieved 5.96437) has done: 'To move RMSE down toward your target with minimal risk, I keep the same LightGBM model and the same feature set, but fix the biggest remaining source of avoidable noise: training rows with implausible NYC coordinates that create misleading distances. Your code defines a NYC bounding box but never applies it; I apply it to the training set *before* computing distance (so we don’t waste compute on junk rows) and add a simple “reasonable distance vs fare” filter that removes extreme outliers known to harm RMSE in this competition. I also compute the datetime-derived features using vectorized `.dt` accessors (same semantics, faster and less error-prone than per-row `apply`), helping you keep more clean rows within the time budget. Submission format/paths remain unchanged and `submission.csv` is still written.'
- What this solution (achieved 5.96966) has done: 'Your RMSE (5.96437) is still much worse than the target (4.03634; lower is better), so we should make a small, legitimate accuracy improvement without changing your model or feature set. The biggest remaining avoidable error is that you train on raw `fare_amount`, which is heavy-tailed; switching the target to `log1p(fare_amount)` and inverting with `expm1` at prediction time typically reduces RMSE on this competition while keeping the same LightGBM regressor, same features, and same training loop. I keep your cleaning, haversine distance, datetime features, and all LightGBM hyperparameters unchanged, and only adjust the target/prediction transform plus the validation metric computation to reflect the transform. Submission formatting and filename remain identical.'
- What this solution (achieved 5.43674) has done: 'To move RMSE down toward your target with minimal risk and without changing the model/features, I’m fixing two score-hurting data issues: (1) the test set is currently unfiltered for obviously invalid/zero coordinates and passenger_count (unlike train), which can create extreme `distance` values and bad predictions; (2) the `log1p`/`expm1` approach can produce very large fares for outlier distances, so I add a conservative upper clip using the training target cap you already apply (250). These changes preserve your exact feature set, LightGBM setup, and training loop while reducing outlier-driven error. The script still runs end-to-end and writes `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use("fivethirtyeight")

import geopy.distance

import os

print(os.listdir("../input"))
import gc



## === cell 1
NROWS_TRAIN = 2_000_000
NROWS_TEST = 200_000  # test is small anyway; keep a safe cap


def load_Data():
    train = pd.read_csv("../input/train.csv", nrows=NROWS_TRAIN, low_memory=True)
    test = pd.read_csv("../input/test.csv", nrows=NROWS_TEST, low_memory=True)
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
pass



## === cell 8
train["key2"] = pd.to_datetime(train["pickup_datetime"], errors="coerce", utc=False)
train["key2"].head()
train.info()



## === cell 9
test["key2"] = pd.to_datetime(test["pickup_datetime"], errors="coerce", utc=False)



## === cell 10
train["fare_amount"].plot(kind="box")



## === cell 11
gc.collect()
train.describe()



## === cell 12
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



## === cell 13
fig, axarr = plt.subplots(2, 2, figsize=(20, 10))

train[~(train["fare_amount"] > 25)]["fare_amount"].plot(kind="box", ax=axarr[0][0])
train[~(train["fare_amount"] > 50)]["fare_amount"].plot(kind="box", ax=axarr[0][1])
train[~(train["fare_amount"] > 100)]["fare_amount"].plot(kind="box", ax=axarr[1][0])
train[~(train["fare_amount"] < 0)]["fare_amount"].plot(kind="box", ax=axarr[1][1])



## === cell 14
train["passenger_count"].plot(kind="box")



## === cell 15
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



## === cell 16
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



## === cell 17
train = train[train["fare_amount"].notnull()]
train = train[train["fare_amount"] > 0]
train = train[train["fare_amount"] <= 250]

train = train[train["passenger_count"].notnull()]
train = train[(train["passenger_count"] >= 1) & (train["passenger_count"] <= 6)]

train = train[train["key2"].notnull()]
test = test[test["key2"].notnull()].copy()

for col in ["pickup_longitude", "dropoff_longitude"]:
    train = train[train[col].notnull()]
    train = train[(train[col] != 0.0)]
for col in ["pickup_latitude", "dropoff_latitude"]:
    train = train[train[col].notnull()]
    train = train[(train[col] != 0.0)]

test.loc[
    (test["passenger_count"].isna())
    | (test["passenger_count"] < 1)
    | (test["passenger_count"] > 6),
    "passenger_count",
] = np.nan
for col in [
    "pickup_longitude",
    "dropoff_longitude",
    "pickup_latitude",
    "dropoff_latitude",
]:
    test.loc[test[col].isna(), col] = np.nan
test.loc[
    (test["pickup_longitude"] == 0.0) | (test["dropoff_longitude"] == 0.0),
    ["pickup_longitude", "dropoff_longitude"],
] = np.nan
test.loc[
    (test["pickup_latitude"] == 0.0) | (test["dropoff_latitude"] == 0.0),
    ["pickup_latitude", "dropoff_latitude"],
] = np.nan



## === cell 18
train = train[~((train["pickup_latitude"] > 90) | (train["pickup_latitude"] < -90))]
train = train[~((train["dropoff_latitude"] > 90) | (train["dropoff_latitude"] < -90))]
train = train[~((train["pickup_longitude"] > 180) | (train["pickup_longitude"] < -180))]
train = train[
    ~((train["dropoff_longitude"] > 180) | (train["dropoff_longitude"] < -180))
]

test.loc[
    (test["pickup_latitude"] > 90) | (test["pickup_latitude"] < -90), "pickup_latitude"
] = np.nan
test.loc[
    (test["dropoff_latitude"] > 90) | (test["dropoff_latitude"] < -90),
    "dropoff_latitude",
] = np.nan
test.loc[
    (test["pickup_longitude"] > 180) | (test["pickup_longitude"] < -180),
    "pickup_longitude",
] = np.nan
test.loc[
    (test["dropoff_longitude"] > 180) | (test["dropoff_longitude"] < -180),
    "dropoff_longitude",
] = np.nan



## === cell 19
nyc_lon_min, nyc_lon_max = -74.3, -72.7
nyc_lat_min, nyc_lat_max = 40.5, 41.8

train = train[
    (train["pickup_longitude"].between(nyc_lon_min, nyc_lon_max))
    & (train["dropoff_longitude"].between(nyc_lon_min, nyc_lon_max))
    & (train["pickup_latitude"].between(nyc_lat_min, nyc_lat_max))
    & (train["dropoff_latitude"].between(nyc_lat_min, nyc_lat_max))
].copy()

out_bbox = ~(
    test["pickup_longitude"].between(nyc_lon_min, nyc_lon_max)
    & test["dropoff_longitude"].between(nyc_lon_min, nyc_lon_max)
    & test["pickup_latitude"].between(nyc_lat_min, nyc_lat_max)
    & test["dropoff_latitude"].between(nyc_lat_min, nyc_lat_max)
)
test.loc[
    out_bbox,
    ["pickup_longitude", "dropoff_longitude", "pickup_latitude", "dropoff_latitude"],
] = np.nan




## === cell 20
def haversine_km(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1.astype("float64"))
    lat1 = np.radians(lat1.astype("float64"))
    lon2 = np.radians(lon2.astype("float64"))
    lat2 = np.radians(lat2.astype("float64"))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c  # Earth radius in km


train["distance"] = haversine_km(
    train["pickup_longitude"].values,
    train["pickup_latitude"].values,
    train["dropoff_longitude"].values,
    train["dropoff_latitude"].values,
)



## === cell 21
train.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)



## === cell 22
test["distance"] = haversine_km(
    test["pickup_longitude"].values,
    test["pickup_latitude"].values,
    test["dropoff_longitude"].values,
    test["dropoff_latitude"].values,
)



## === cell 23
test.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)



## === cell 24
print(
    "% of trips above 25 KM - {:0.2f}".format(
        train[train["distance"] > 25]["key"].count() * 100 / train.count()["key"]
    )
)



## === cell 25
train = train[
    (train["distance"].notnull()) & (train["distance"] > 0) & (train["distance"] <= 80)
].copy()

train = train[~((train["distance"] < 0.1) & (train["fare_amount"] > 30))].copy()
train = train[~((train["distance"] > 30) & (train["fare_amount"] < 10))].copy()



## === cell 26
train["year"] = train["key2"].dt.year
train["month"] = train["key2"].dt.month
train["day"] = train["key2"].dt.day
train["day of week"] = train["key2"].dt.weekday
train["hour"] = train["key2"].dt.hour



## === cell 27
test["year"] = test["key2"].dt.year
test["month"] = test["key2"].dt.month
test["day"] = test["key2"].dt.day
test["day of week"] = test["key2"].dt.weekday
test["hour"] = test["key2"].dt.hour



## === cell 28
train = train.fillna(0)
test = test.fillna(0)



## === cell 29
column_list = [
    "passenger_count",
    "distance",
    "year",
    "month",
    "day",
    "day of week",
    "hour",
]

y_train = np.log1p(train["fare_amount"].astype("float64"))
X_train = train[column_list].copy()



## === cell 30
X_test = test[column_list].copy()



## === cell 31
X_train.shape, y_train.shape



## === cell 32
from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.1,
    random_state=42,
)



## === cell 33
X_tr.shape, X_val.shape, y_tr.shape, y_val.shape



## === cell 34
from lightgbm import LGBMRegressor



## === cell 35
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



## === cell 36
lgb.fit(X_tr, y_tr)



## === cell 37
pred = lgb.predict(X_val)



## === cell 38
from sklearn.metrics import mean_squared_error

rmse = mean_squared_error(np.expm1(y_val), np.expm1(pred), squared=False)
rmse



## === cell 39
pass




## === cell 40
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



## === cell 41
lgb.fit(X_train, y_train)



## === cell 42
y_pred = np.expm1(lgb.predict(X_test))



## === cell 43
y_pred = np.clip(y_pred, 0.01, 250.0)



## === cell 44
submission = pd.DataFrame({"key": test["key"], "fare_amount": y_pred})
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
