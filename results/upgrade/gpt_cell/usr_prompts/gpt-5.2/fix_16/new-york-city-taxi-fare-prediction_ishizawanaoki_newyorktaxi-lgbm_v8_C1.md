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

3.10

# 3. Installed packages

geopandas==0.14.4
lightgbm==4.6.0
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

3.40781

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.44226) has done: 'Your current pipeline likely didn’t yield a Kaggle score because it either didn’t run to completion within limits (training 5-fold with early stopping on 1M rows can be slow) or produced a misaligned submission (using `sample_submission` rather than `test` keys can silently mismatch if row order differs). To move RMSE toward your target with minimal logic change, I (1) ensure the submission is built from `test[['key']]` to guarantee correct key alignment, (2) replace early-stopping callbacks with a fixed `num_boost_round` (same training approach, but avoids callback overhead/instability and ensures completion), and (3) add a small, standard NYC coordinate filter on the training sample to reduce obvious outliers that hurt RMSE without changing the model type or features. These changes should produce a valid `submission.csv` deterministically and typically improve RMSE versus the current setup.'
- What this solution (achieved 5.65613) has done: 'Your current RMSE (5.44) is worse than the target (3.41), so we should modestly improve model signal without changing the overall approach (LightGBM regressor on the same raw coordinate/passenger features with KFold). The biggest low-risk gain here is to add a single, standard distance feature (haversine) computed from the existing lat/longs; this keeps the same model/training loop/loss but gives the model the key nonlinearity it’s currently missing. I also switch the LightGBM objective/metric to RMSE (equivalent to L2 minimization but aligned with the competition metric) and keep num_boost_round fixed to preserve determinism and runtime. Submission generation remain keyed off `test[['key']]` so row alignment stays correct.'
- What this solution (achieved 5.65182) has done: 'Your current notebook should already be able to write a valid `submission.csv`, so the “Not yielded” is most likely from a runtime/memory interruption or a silent NaN/inf issue after feature engineering. To move RMSE closer to the 3.41 target with minimal core-logic change, I (1) make the datetime conversion consistent and avoid dropping test rows due to datetime parse by only dropping NA rows on the training part, (2) add a very small, standard outlier filter on the engineered distance (removes impossible trips that otherwise hurt RMSE), and (3) ensure predictions are always aligned to the original `test.csv` order/keys even after concatenation/processing. These are small, legitimate fixes that typically improve RMSE without changing the model family, loss, or training loop. The script still trains LightGBM with the same KFold approach and writes `submission.csv` deterministically.'
- What this solution (achieved 6.03532) has done: 'Your RMSE (5.65) is worse than the target (3.41), so we should make a small, legitimate improvement without changing the core LightGBM/KFold approach. The biggest issue is that fare is dominated by trip length and typical NYC airport trips; adding just two lightweight features (log-distance and a simple “airport-ish” flag from existing coordinates) usually improves RMSE a lot while keeping the same model/training loop. I also add a minimal fare-per-km sanity filter on the training sample to remove egregious label/noise outliers that otherwise inflate RMSE, without touching the test set. Submission generation remains keyed to `test.csv` order/keys to avoid any alignment mistakes.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
train = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/train.csv",
    nrows=1_000_000,
    parse_dates=["pickup_datetime"],
)
test = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/test.csv",
    parse_dates=["pickup_datetime"],
)
sample_submission = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/sample_submission.csv"
)



## === cell 2
train.isnull().sum()



## === cell 3
train.dropna(inplace=True)



## === cell 4
train.describe()



## === cell 5
train.query("passenger_count > 6")



## === cell 6
train.query("passenger_count < 1")



## === cell 7
train.query("fare_amount < 0")



## === cell 8
train = train.query("1 <= passenger_count <= 6 and 0 <= fare_amount").copy()
train = train.query("fare_amount <= 250").copy()

train = train[
    (train["pickup_longitude"].between(-75, -72))
    & (train["dropoff_longitude"].between(-75, -72))
    & (train["pickup_latitude"].between(40, 42))
    & (train["dropoff_latitude"].between(40, 42))
].copy()

train.describe()



## === cell 9
train.reset_index(drop=True, inplace=True)
train



## === cell 10
test_keys = test["key"].astype(str).copy()

train["_is_train"] = 1
test["_is_train"] = 0
test["fare_amount"] = np.nan

data = pd.concat([train, test], sort=False, ignore_index=True)



## === cell 11
data.head()




## === cell 12
def haversine_np(lon1, lat1, lon2, lat2):
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371.0 * c
    return km


def bearing_np(lon1, lat1, lon2, lat2):
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    brng = np.arctan2(y, x)  # [-pi, pi]
    return brng


data["pickup_datetime"] = pd.to_datetime(data["pickup_datetime"], errors="coerce")

data["pickup_hour"] = data["pickup_datetime"].dt.hour.astype("float32")
data["pickup_weekday"] = data["pickup_datetime"].dt.weekday.astype("float32")
data["pickup_month"] = data["pickup_datetime"].dt.month.astype("float32")

data["distance_km"] = haversine_np(
    data["pickup_longitude"],
    data["pickup_latitude"],
    data["dropoff_longitude"],
    data["dropoff_latitude"],
).astype("float32")

data["delta_lon"] = (
    data["dropoff_longitude"].astype(float) - data["pickup_longitude"].astype(float)
).astype("float32")
data["delta_lat"] = (
    data["dropoff_latitude"].astype(float) - data["pickup_latitude"].astype(float)
).astype("float32")

lat_rad = np.radians(data["pickup_latitude"].astype(float))
data["manhattan_km"] = (
    111.0 * np.abs(data["delta_lat"].astype(float))
    + 111.0 * np.cos(lat_rad) * np.abs(data["delta_lon"].astype(float))
).astype("float32")

data["log_distance_km"] = np.log1p(data["distance_km"].astype(float)).astype("float32")

data["distance_km2"] = (data["distance_km"].astype(float) ** 2).astype("float32")
data["center_lon"] = (
    (data["pickup_longitude"].astype(float) + data["dropoff_longitude"].astype(float))
    / 2.0
).astype("float32")
data["center_lat"] = (
    (data["pickup_latitude"].astype(float) + data["dropoff_latitude"].astype(float))
    / 2.0
).astype("float32")
data["bearing"] = bearing_np(
    data["pickup_longitude"],
    data["pickup_latitude"],
    data["dropoff_longitude"],
    data["dropoff_latitude"],
).astype("float32")

nyc_lon, nyc_lat = -73.985428, 40.748817  # Midtown Manhattan-ish (ESB)
data["pickup_to_nyc_km"] = haversine_np(
    data["pickup_longitude"], data["pickup_latitude"], nyc_lon, nyc_lat
).astype("float32")
data["dropoff_to_nyc_km"] = haversine_np(
    data["dropoff_longitude"], data["dropoff_latitude"], nyc_lon, nyc_lat
).astype("float32")


def _in_box(lon, lat, lon_min, lon_max, lat_min, lat_max):
    return (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)


p_lon = data["pickup_longitude"].astype(float)
p_lat = data["pickup_latitude"].astype(float)
d_lon = data["dropoff_longitude"].astype(float)
d_lat = data["dropoff_latitude"].astype(float)

jfk_p = _in_box(p_lon, p_lat, -73.90, -73.75, 40.62, 40.68)
jfk_d = _in_box(d_lon, d_lat, -73.90, -73.75, 40.62, 40.68)
lga_p = _in_box(p_lon, p_lat, -73.90, -73.84, 40.75, 40.79)
lga_d = _in_box(d_lon, d_lat, -73.90, -73.84, 40.75, 40.79)
ewr_p = _in_box(p_lon, p_lat, -74.20, -74.13, 40.67, 40.71)
ewr_d = _in_box(d_lon, d_lat, -74.20, -74.13, 40.67, 40.71)

data["airport_trip"] = (
    (jfk_p | lga_p | ewr_p | jfk_d | lga_d | ewr_d).astype("int8")
).astype("float32")

data = data.drop("pickup_datetime", axis=1)
data["key"] = data["key"].astype(str)

data = data.reset_index(drop=True)

is_train_mask = data["_is_train"] == 1
data = data.loc[(~is_train_mask) | (data["distance_km"].between(0.0, 200.0))].copy()

train_mask = data["_is_train"] == 1
fare = data.loc[train_mask, "fare_amount"].astype(float)
dist = data.loc[train_mask, "distance_km"].astype(float)
fare_per_km = fare / np.maximum(dist, 0.1)
keep_train = fare_per_km.between(0.5, 50.0)  # broad bounds to only drop egregious cases
data = pd.concat(
    [data.loc[~train_mask], data.loc[train_mask].loc[keep_train]],
    axis=0,
    ignore_index=True,
)

train_na_mask = (data["_is_train"] == 1) & data.isna().any(axis=1)
if train_na_mask.any():
    data = data.loc[~train_na_mask].copy()

data = data.reset_index(drop=True)
data.head()



## === cell 13
train = data.loc[data["_is_train"] == 1].copy()
test = data.loc[data["_is_train"] == 0].copy()

y_train = train["fare_amount"].astype(float).clip(lower=0.0)

y_cap = float(np.nanquantile(y_train.values, 0.999))
y_train = y_train.clip(upper=y_cap)

X_train = train.drop(["fare_amount", "_is_train", "key"], axis=1)
X_test = test.drop(["fare_amount", "_is_train", "key"], axis=1)

X_train.head()



## === cell 14
from sklearn.model_selection import KFold

y_preds = []
models = []
oof_train = np.zeros((len(X_train),), dtype=np.float64)

cv = KFold(n_splits=3, shuffle=True, random_state=0)

categorical_features = []



## === cell 15
import lightgbm as lgb

params = {
    "objective": "regression_l2",
    "metric": "rmse",
    "max_bin": 300,
    "learning_rate": 0.05,
    "num_leaves": 40,
    "min_data_in_leaf": 50,
    "lambda_l2": 1.0,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "verbosity": -1,
    "seed": 0,
    "feature_fraction_seed": 0,
    "bagging_seed": 0,
}

num_boost_round = 600

for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train, y_train)):
    X_tr = X_train.iloc[train_index, :]
    X_val = X_train.iloc[valid_index, :]
    y_tr = y_train.iloc[train_index]
    y_val = y_train.iloc[valid_index]

    lgb_train = lgb.Dataset(
        X_tr, y_tr, categorical_feature=categorical_features, free_raw_data=False
    )
    lgb_eval = lgb.Dataset(
        X_val,
        y_val,
        reference=lgb_train,
        categorical_feature=categorical_features,
        free_raw_data=False,
    )

    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_eval],
        valid_names=["train", "valid"],
        num_boost_round=num_boost_round,
        callbacks=[
            lgb.log_evaluation(period=50),
        ],
    )

    oof_train[valid_index] = model.predict(X_val, num_iteration=num_boost_round)

    if X_test.shape[0] > 0:
        y_pred = model.predict(X_test, num_iteration=num_boost_round)
    else:
        y_pred = np.array([], dtype=np.float64)

    y_preds.append(y_pred)
    models.append(model)



## === cell 16
pd.DataFrame(oof_train).to_csv("oof_train_kfold.csv", index=False)

scores = [m.best_score["valid"]["rmse"] for m in models]
score = sum(scores) / len(scores)
print("===CV scores (RMSE on valid, fare space)===")
print(scores)
print(score)



## === cell 17
from sklearn.metrics import mean_squared_error

y_pred_oof_fare = np.clip(oof_train, 0.0, None)
y_true_fare = train["fare_amount"].astype(float).values
np.sqrt(mean_squared_error(y_true_fare, y_pred_oof_fare))



## === cell 18
len(y_preds)



## === cell 19
y_preds[0][:10]



## === cell 20
y_sub = sum(y_preds) / len(y_preds)

y_sub = np.clip(y_sub, 0.0, None)

y_sub[:10]



## === cell 21
sub_lgb = pd.DataFrame({"key": test_keys})
sub_lgb["fare_amount"] = y_sub.astype(float)

fill_value = float(np.nanmean(y_sub)) if len(y_sub) else 11.35
sub_lgb["fare_amount"] = (
    sub_lgb["fare_amount"].replace([np.inf, -np.inf], np.nan).fillna(fill_value)
)

sub_lgb.to_csv("submission.csv", index=False)
sub_lgb.to_csv("submission_lightgbm.csv", index=False)

sub_lgb.head()
