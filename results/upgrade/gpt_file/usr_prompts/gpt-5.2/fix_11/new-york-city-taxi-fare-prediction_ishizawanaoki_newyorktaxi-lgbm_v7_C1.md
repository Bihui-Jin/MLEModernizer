# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os




## === cell 1
train_dtypes = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
test_dtypes = {
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}

train = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
    nrows=1_000_000,
    dtype=train_dtypes,
)
test = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
    dtype=test_dtypes,
)
sample_submission = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
)



## === cell 2
train.dropna(inplace=True)



## === cell 3
m = (train["fare_amount"] > 0) & (train["fare_amount"] <= 250)
m &= train["pickup_longitude"].between(-75, -72)
m &= train["dropoff_longitude"].between(-75, -72)
m &= train["pickup_latitude"].between(40, 42)
m &= train["dropoff_latitude"].between(40, 42)
m &= train["passenger_count"].between(1, 6)

raw_abs_lon = (train["pickup_longitude"] - train["dropoff_longitude"]).abs()
raw_abs_lat = (train["pickup_latitude"] - train["dropoff_latitude"]).abs()
m &= (raw_abs_lon + raw_abs_lat) > 1e-6
m &= (raw_abs_lon < 1.0) & (raw_abs_lat < 1.0)

train = train.loc[m].reset_index(drop=True)



## === cell 9
data = pd.concat([train, test], sort=False, ignore_index=True)



## === cell 11
coord_mask = (
    (data["pickup_longitude"].between(-75, -72))
    & (data["dropoff_longitude"].between(-75, -72))
    & (data["pickup_latitude"].between(40, 42))
    & (data["dropoff_latitude"].between(40, 42))
)
for c in [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]:
    data.loc[~coord_mask, c] = np.nan

data["pickup_datetime"] = pd.to_datetime(
    data["pickup_datetime"], utc=True, errors="coerce"
)
dt = data["pickup_datetime"].dt
data["pickup_year"] = dt.year.astype("float32")
data["pickup_month"] = dt.month.astype("float32")
data["pickup_day"] = dt.day.astype("float32")
data["pickup_hour"] = dt.hour.astype("float32")
data["pickup_weekday"] = dt.weekday.astype("float32")


def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c  # km


def bearing_np(lon1, lat1, lon2, lat2):
    """Initial bearing from (lon1,lat1) to (lon2,lat2), in radians."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.arctan2(y, x)


plon = data["pickup_longitude"].to_numpy(dtype=np.float64, copy=False)
plat = data["pickup_latitude"].to_numpy(dtype=np.float64, copy=False)
dlon = data["dropoff_longitude"].to_numpy(dtype=np.float64, copy=False)
dlat = data["dropoff_latitude"].to_numpy(dtype=np.float64, copy=False)
pcnt = data["passenger_count"].to_numpy(dtype=np.float32, copy=False)

abs_lon_diff = np.abs(plon - dlon).astype(np.float32)
abs_lat_diff = np.abs(plat - dlat).astype(np.float32)
data["abs_lon_diff"] = abs_lon_diff
data["abs_lat_diff"] = abs_lat_diff

haversine_miles = (haversine_np(plon, plat, dlon, dlat) * 0.621371).astype(np.float32)
data["haversine_miles"] = haversine_miles

mean_lat_rad = np.radians(((plat + dlat) / 2.0))
miles_per_deg_lat = 69.172
miles_per_deg_lon = 69.172 * np.cos(mean_lat_rad)

dx_miles = (abs_lon_diff.astype(np.float64) * miles_per_deg_lon).astype(np.float32)
dy_miles = (abs_lat_diff.astype(np.float64) * miles_per_deg_lat).astype(np.float32)

data["manhattan_miles"] = (dx_miles + dy_miles).astype(np.float32)
data["euclidean_miles"] = np.sqrt(
    dx_miles.astype(np.float64) ** 2 + dy_miles.astype(np.float64) ** 2
).astype(np.float32)

data["bearing"] = bearing_np(plon, plat, dlon, dlat).astype(np.float32)

JFK = (-73.7781, 40.6413)
LGA = (-73.8740, 40.7769)
EWR = (-74.1745, 40.6895)
NYC_CENTER = (-73.985428, 40.748817)


def haversine_to_point_miles(lon, lat, point_lon, point_lat):
    return (
        haversine_np(lon, lat, np.float64(point_lon), np.float64(point_lat)) * 0.621371
    ).astype(np.float32)


for name, (alon, alat) in [("jfk", JFK), ("lga", LGA), ("ewr", EWR)]:
    data[f"pickup_to_{name}_miles"] = haversine_to_point_miles(plon, plat, alon, alat)
    data[f"dropoff_to_{name}_miles"] = haversine_to_point_miles(dlon, dlat, alon, alat)

data["pickup_to_center_miles"] = haversine_to_point_miles(
    plon, plat, NYC_CENTER[0], NYC_CENTER[1]
)
data["dropoff_to_center_miles"] = haversine_to_point_miles(
    dlon, dlat, NYC_CENTER[0], NYC_CENTER[1]
)

data["pc_x_haversine"] = (pcnt * haversine_miles).astype(np.float32)

data = data.drop("pickup_datetime", axis=1)

num_cols = data.select_dtypes(include=[np.number]).columns
feature_num_cols = [c for c in num_cols if c != "fare_amount"]
medians = data.loc[: len(train) - 1, feature_num_cols].median(numeric_only=True)
data[feature_num_cols] = data[feature_num_cols].fillna(medians)



## === cell 12
train2 = data.iloc[: len(train)].copy()
test2 = data.iloc[len(train) :].copy()

y_train = train2["fare_amount"].astype(float)
X_train = train2.drop(["fare_amount", "key"], axis=1)
X_test = test2.drop(["fare_amount", "key"], axis=1)

X_train_np = np.ascontiguousarray(X_train.to_numpy(dtype=np.float32, copy=False))
X_test_np = np.ascontiguousarray(X_test.to_numpy(dtype=np.float32, copy=False))
y_train_np = y_train.to_numpy(dtype=np.float64, copy=False)

feature_names = list(X_train.columns)



## === cell 13
from sklearn.model_selection import KFold

y_preds = []
models = []
oof_train = np.zeros((len(X_train_np),), dtype=np.float64)
cv = KFold(n_splits=5, shuffle=True, random_state=0)

categorical_features = []



## === cell 14
import lightgbm as lgb

params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "dart",
    "learning_rate": 0.05,
    "max_bin": 300,
    "num_leaves": 64,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 0.1,
    "verbosity": -1,
    "num_threads": max(1, (os.cpu_count() or 1) - 1),
    "seed": 0,
    "feature_fraction_seed": 0,
    "bagging_seed": 0,
    "data_random_seed": 0,
    "deterministic": True,
    "force_row_wise": True,
    "feature_pre_filter": False,
}

for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train_np, y_train_np)):
    X_tr = X_train_np[train_index]
    X_val = X_train_np[valid_index]
    y_tr = y_train_np[train_index]
    y_val = y_train_np[valid_index]

    lgb_train = lgb.Dataset(
        X_tr,
        y_tr,
        feature_name=feature_names,
        categorical_feature=categorical_features,
        free_raw_data=False,
    )
    lgb_eval = lgb.Dataset(
        X_val,
        y_val,
        reference=lgb_train,
        feature_name=feature_names,
        categorical_feature=categorical_features,
        free_raw_data=False,
    )

    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_eval],
        valid_names=["train", "valid"],
        num_boost_round=2000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=20),
            lgb.log_evaluation(period=50),
        ],
    )

    oof_train[valid_index] = model.predict(X_val, num_iteration=model.best_iteration)
    y_pred = model.predict(X_test_np, num_iteration=model.best_iteration)

    y_preds.append(y_pred)
    models.append(model)



## === cell 15
pd.DataFrame(oof_train).to_csv("oof_train_kfold.csv", index=False)

scores = [m.best_score["valid"]["rmse"] for m in models]
score = sum(scores) / len(scores)
print("===CV scores (rmse)===")
print(scores)
print(score)



## === cell 16
from sklearn.metrics import mean_squared_error

y_pred_oof = oof_train
np.sqrt(mean_squared_error(y_train_np, y_pred_oof))



## === cell 17
len(y_preds)



## === cell 18
y_preds[0][:10]



## === cell 19
y_sub = sum(y_preds) / len(y_preds)
y_sub = np.clip(y_sub, 0.0, None)
y_sub[:10]



## === cell 20
sub_lgb = pd.DataFrame({"key": test["key"].values, "fare_amount": y_sub})
sub_lgb.to_csv("submission_lightgbm.csv", index=False)

sub_lgb.head()
