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

4.22957

# 6. Current score

6.28795

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.78341) has done: 'Diagnosis: Cell 20 crashes because `geopy.distance.VincentyDistance` was removed from modern `geopy` (it no longer exists in geopy==2.4.1). The train set distance in cell 18 already uses `geopy.distance.geodesic(...).km`, which is the supported replacement and preserves the intended “distance in km” feature semantics.  
Patch summary: Replace the unavailable `VincentyDistance` call with `geodesic`, keeping the same `.apply(..., axis=1)` structure and returning kilometers so downstream code (cell 21) continues to work unchanged.  
Updated cells: Only cell 20 is modified.  
Compatibility notes for cell k+1: `test['distance']` is still created as a numeric column before `test.drop(...)` in cell 21, so cell 21 remains compatible.  
Assumptions: Using `geodesic` is acceptable as a direct replacement for removed Vincenty in geopy, with negligible floating-point differences.'
- What this solution (achieved 5.77234) has done: 'You’re substantially worse than the target RMSE (5.78341 vs 4.22957), so we should make a small, legitimate improvement without changing the model type or the overall pipeline. The biggest issue is that the current training features accidentally include non-numeric/leaky columns (`key`, `pickup_datetime`, `key2`, plus extra engineered columns) because `X_train = train.drop(['fare_amount'], ...)` is used before splitting; LightGBM either ignore/object-cast these inconsistently and hurt generalization. I restrict both train and validation to the exact intended numeric `column_list` (same as test), and I make the split deterministic (fixed `random_state`) for stability; this keeps the core logic (same features, same LGBMRegressor, same training flow) but removes the accidental feature mismatch. I also fit the final model on the full cleaned training data using the same features before predicting test, which is consistent with your intent and should move RMSE toward the target.'
- What this solution (achieved 5.82938) has done: 'Your pipeline already produces a valid `submission.csv`, but it likely underperforms because `filter_nyc_bbox(test)` drops many test rows; Kaggle expects predictions for every test `key`, so missing rows effectively become wrong and inflate RMSE. I keep the exact same feature engineering and LightGBM model, but stop filtering the test set and instead only filter the training set (where it’s legitimate). To keep the submission aligned and complete, I always build the submission from the original full test keys, merging predictions back and filling any unscored rows with a conservative fallback (the training mean fare) so every test row is present. This is a minimal, semantics-preserving fix aimed at moving RMSE down toward your 4.22957 target without changing the model architecture or training approach.'
- What this solution (achieved 5.82938) has done: 'Your current RMSE is far worse than the target, so the smallest legitimate improvement is to fix a key feature-engineering bug and align train/test preprocessing without changing the model or overall approach. In `prepare_distance_features`, the haversine formula is implemented incorrectly (missing squares), which makes the `haversine` feature essentially wrong; correcting it is a minimal change that should materially reduce error while preserving your pipeline. In addition, you fill NA only for train, not for test; we fill NA for test too so the model sees consistent inputs and avoids silent NaN handling. Finally, we keep the exact same LightGBM model/training flow and submission-writing logic, just ensuring the engineered features are computed correctly and consistently.'
- What this solution (achieved 6.02548) has done: 'Your current RMSE (5.82938) is worse than the target (4.22957), so we should make a small, legitimate improvement without changing the model type or training approach. The biggest mismatch is that you engineer rich time features (`hour_of_day`, `weekday`, etc.) and a haversine distance, but then you later overwrite time features from `key2` and only train on a small `column_list`, effectively discarding useful information. I keep the same LightGBM regressor and training flow, but expand `column_list` to include the already-created time/distance features (and avoid the duplicated/overwritten time columns from `key2`). This is a minimal feature-selection fix that should move RMSE down toward the target while keeping everything else intact.'
- What this solution (achieved 6.0107) has done: 'Your current RMSE (6.02548) is still far worse than the target (4.22957), so we should make a small but meaningful improvement without changing the model type or training loop. The biggest quality issue left is that `distance` is computed with `geopy.geodesic` row-wise, which is extremely slow and also mixes units with `haversine` (meters) and other features (degrees); we replace `distance` with a fast, vectorized haversine-in-km computed from the same coordinates before dropping them, keeping the overall feature set and LightGBM training unchanged. We also convert `week_of_year` (and `week`) to a plain integer dtype to avoid nullable `Int64` quirks and ensure train/test dtypes match exactly. These are minimal preprocessing/feature-consistency fixes intended to reduce RMSE toward the target while keeping the core modeling logic intact and still writing a valid `submission.csv`.'
- What this solution (achieved 6.0107) has done: 'Your RMSE (6.0107) is still far worse than the target (4.22957), so we should make a small, legitimate improvement without changing the model/training loop. The biggest remaining quality issue is inconsistent/missing time features: you engineer rich time features from `pickup_datetime`, but then you overwrite year/month/day/hour from `key2` (which is often `NaT` due to the extra integer suffix) and never add these derived columns to `column_list`, so you end up training with partially broken time information. I instead parse `key2` from `pickup_datetime` (stable, always present) and compute the same `year/month/day/day of week/hour` for both train and test using vectorized `.dt` access, keeping all existing features and the same LightGBM configuration. This keeps core logic intact, improves feature correctness/consistency, and should move RMSE down toward the target while still writing a valid `submission.csv`.'
- What this solution (achieved 6.0107) has done: 'Your RMSE (6.0107) is still far above the target (4.22957), so we should make a small, legitimate improvement that preserves your core LightGBM model and training flow. The biggest remaining correctness issue is that `haversine` is computed in **meters** while `distance` is in **kilometers**, which makes distance-related features inconsistent in scale and can hurt LightGBM splits; we convert `haversine` to kilometers to match `distance`. We also ensure all feature columns used by the model are numeric (coercing any unexpected object dtypes to floats) for consistent train/test handling. These changes keep the same features, model, and pipeline structure, but improve feature consistency to move RMSE down toward the target.'
- What this solution (achieved 6.0107) has done: 'You’re still well above the target RMSE, so the smallest safe improvement is to fix the one clear modeling mistake that hurts generalization: you fit the model twice but the second fit (cell 36) accidentally retrains on the same X_train only and never uses the validation split, while also leaving your “final” model dependent on earlier state. I change cell 36 to train on the full training fold (X_train+y_train) plus validation monitoring (eval_set) without altering the model type, features, or loss, which typically improves the resulting fitted parameters and stability. I also make predictions non-negative (fares can’t be negative) right before writing the submission; this is metric-aligned post-processing that usually reduces RMSE on this competition without changing core logic. All paths and submission schema remain identical, and it still write a valid `submission.csv`.'
- What this solution (achieved 6.28795) has done: 'Your current RMSE (6.0107) is still much worse than the target (4.22957), so we need a small but meaningful quality boost without changing the core LightGBM approach. The largest remaining issue is that `distance_travelled_sin*`/`cos*` features are mathematically malformed (they use `sqrt(lon^2 * lat^2)` instead of a consistent distance-like quantity), which injects noise; I minimally correct them to use `distance_travelled` directly while keeping the same feature names and downstream pipeline intact. I also stop overwriting time columns from `key2` (those columns aren’t used anyway) to avoid introducing NaNs/inconsistencies and ensure dtypes match across train/test. Finally, I add the standard `min_child_samples` alias for your `min_data_in_leaf` to ensure LightGBM 4.6.0 applies the intended constraint consistently (no architecture/training-loop change), and keep the same submission-writing logic.'
- What this solution (achieved 6.28795) has done: 'We need to move your RMSE down toward the target (4.22957) from 6.28795, so we should make small, metric-aligned improvements without changing the core LightGBM approach. The biggest safe gain here is to reduce label noise by training on a more representative sample of the 55M rows (still the same model, same features, same training loop): instead of the first 1,000,000 rows, we take a deterministic uniform sample across the full file using chunked reading. In addition, we remove a small known source of massive error by dropping “zero-distance” rides (pickup==dropoff), which are usually data issues and create very hard outliers under RMSE. Everything else (feature engineering, column_list, LGBMRegressor parameters, prediction + non-negativity clip, submission alignment) remains unchanged.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use("fivethirtyeight")

import geopy.distance

import os
import gc

print(os.listdir("../input"))




## === cell 1
def load_Data():
    train_path = "../input/train.csv"
    test_path = "../input/test.csv"

    target_train_rows = 1_000_000  # keep runtime/memory similar to original
    chunk_size = 250_000
    rng = np.random.RandomState(42)

    chunks = []
    seen = 0
    kept = 0

    for chunk in pd.read_csv(train_path, chunksize=chunk_size, low_memory=True):
        n = len(chunk)
        seen += n
        remaining = target_train_rows - kept
        if remaining <= 0:
            break

        frac = min(1.0, remaining / n)
        mask = rng.rand(n) < frac
        sub = chunk.loc[mask]
        if len(sub) > remaining:
            sub = sub.sample(n=remaining, random_state=42)
        chunks.append(sub)
        kept += len(sub)

    train = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    test = pd.read_csv(test_path, nrows=10_00_000, low_memory=True)
    return train, test


def prepare_distance_features(df):
    df["longitude_distance"] = abs(df["pickup_longitude"] - df["dropoff_longitude"])
    df["latitude_distance"] = abs(df["pickup_latitude"] - df["dropoff_latitude"])

    df["distance_travelled"] = (
        df["longitude_distance"] ** 2 + df["latitude_distance"] ** 2
    ) ** 0.5

    base = df["distance_travelled"].astype(float)
    df["distance_travelled_sin"] = np.sin(base)
    df["distance_travelled_cos"] = np.cos(base)
    df["distance_travelled_sin_sqrd"] = np.sin(base) ** 2
    df["distance_travelled_cos_sqrd"] = np.cos(base) ** 2

    R_km = 6371.0  # kilometers
    phi1 = np.radians(df["pickup_latitude"])
    phi2 = np.radians(df["dropoff_latitude"])
    dphi = np.radians(df["dropoff_latitude"] - df["pickup_latitude"])
    dlambda = np.radians(df["dropoff_longitude"] - df["pickup_longitude"])
    a = (np.sin(dphi / 2) ** 2) + np.cos(phi1) * np.cos(phi2) * (
        np.sin(dlambda / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d_km = R_km * c
    df["haversine"] = d_km

    y = np.sin(dlambda * np.cos(phi2))
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlambda)
    df["bearing"] = np.arctan2(y, x)

    return df




## === cell 2
train, test = load_Data()




## === cell 3
def prepare_time_features(df):
    df["pickup_datetime"] = df["pickup_datetime"].str.replace(" UTC", "", regex=False)
    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )
    df["hour_of_day"] = df.pickup_datetime.dt.hour
    df["week"] = df.pickup_datetime.dt.isocalendar().week.astype("int16")
    df["month"] = df.pickup_datetime.dt.month
    df["year"] = df.pickup_datetime.dt.year
    df["day_of_year"] = df.pickup_datetime.dt.dayofyear
    df["week_of_year"] = df.pickup_datetime.dt.isocalendar().week.astype("int16")
    df["weekday"] = df.pickup_datetime.dt.weekday
    df["quarter"] = df.pickup_datetime.dt.quarter
    df["day_of_month"] = df.pickup_datetime.dt.day
    return df


train = prepare_time_features(train)
train = prepare_distance_features(train)

test = prepare_time_features(test)
test = prepare_distance_features(test)



## === cell 4
train.describe()



## === cell 5
train.info()



## === cell 6
train.isnull().sum()



## === cell 7
train = train.fillna(0)
test = test.fillna(0)



## === cell 8
train["key2"] = train["pickup_datetime"]
train["key2"].head()
train.info()



## === cell 9
test["key2"] = test["pickup_datetime"]



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
def filter_nyc_bbox(df):
    return df[
        (df["pickup_longitude"].between(-74.3, -73.6))
        & (df["dropoff_longitude"].between(-74.3, -73.6))
        & (df["pickup_latitude"].between(40.5, 41.0))
        & (df["dropoff_latitude"].between(40.5, 41.0))
    ]


train = train[~((train["pickup_latitude"] > 90) | (train["pickup_latitude"] < -90))]
train = train[~((train["dropoff_latitude"] > 90) | (train["dropoff_latitude"] < -90))]
train = train[~((train["pickup_longitude"] > 180) | (train["pickup_longitude"] < -180))]
train = train[
    ~((train["dropoff_longitude"] > 180) | (train["dropoff_longitude"] < -180))
]

train = filter_nyc_bbox(train)

train = train[(train["fare_amount"] > 0) & (train["fare_amount"] <= 250)]
train = train[train["passenger_count"].between(1, 6)]

train = train[train["key2"].notna()]




## === cell 18
def add_distance_km_haversine(df):
    lat1 = np.radians(df["pickup_latitude"].astype(float))
    lon1 = np.radians(df["pickup_longitude"].astype(float))
    lat2 = np.radians(df["dropoff_latitude"].astype(float))
    lon2 = np.radians(df["dropoff_longitude"].astype(float))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    km = 6371.0 * c
    df["distance"] = km
    return df


train = add_distance_km_haversine(train)

train = train[train["distance"] > 0]



## === cell 19
train.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)



## === cell 20
test = add_distance_km_haversine(test)



## === cell 21
test.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)



## === cell 22
print(
    "% of trips above 25 KM - {:0.2f}".format(
        train[train["distance"] > 25]["key"].count() * 100 / train.count()["key"]
    )
)



## === cell 23
pass



## === cell 24
pass



## === cell 25
column_list = [
    "passenger_count",
    "distance",
    "haversine",
    "bearing",
    "longitude_distance",
    "latitude_distance",
    "distance_travelled",
    "distance_travelled_sin",
    "distance_travelled_cos",
    "distance_travelled_sin_sqrd",
    "distance_travelled_cos_sqrd",
    "hour_of_day",
    "weekday",
    "quarter",
    "day_of_month",
    "day_of_year",
    "week_of_year",
]
y = train["fare_amount"]
X = train[column_list]



## === cell 26
X_test = test[column_list]



## === cell 27
for c in column_list:
    X[c] = pd.to_numeric(X[c], errors="coerce")
    X_test[c] = pd.to_numeric(X_test[c], errors="coerce")
X = X.fillna(0)
X_test = X_test.fillna(0)



## === cell 28
X.shape, y.shape



## === cell 29
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)



## === cell 30
X_train.shape, X_val.shape, y_train.shape, y_val.shape



## === cell 31
from lightgbm import LGBMRegressor



## === cell 32
lgb = LGBMRegressor(
    objective="regression",
    num_leaves=5,
    learning_rate=0.05,
    n_estimators=720,
    max_bin=55,
    bagging_fraction=0.8,
    bagging_freq=5,
    feature_fraction=0.2319,
    feature_fraction_seed=9,
    bagging_seed=9,
    min_data_in_leaf=6,
    min_child_samples=6,
    min_sum_hessian_in_leaf=11,
)



## === cell 33
lgb.fit(X_train, y_train)



## === cell 34
pred = lgb.predict(X_val)



## === cell 35
from sklearn.metrics import r2_score

r2_score(y_val, pred)



## === cell 36
lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="rmse")




## === cell 37
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



## === cell 38
lgb.fit(X, y)



## === cell 39
y_pred = lgb.predict(X_test)



## === cell 40
y_pred = np.maximum(y_pred, 0.0)

test_keys_full = pd.read_csv("../input/test.csv", usecols=["key"], low_memory=True)
pred_df = pd.DataFrame({"key": test["key"].values, "fare_amount": y_pred})

submission = test_keys_full.merge(pred_df, on="key", how="left")
submission["fare_amount"] = submission["fare_amount"].fillna(y.mean())

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Missing predictions filled:", submission["fare_amount"].isna().sum())
