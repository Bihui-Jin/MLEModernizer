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

None

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np, pandas as pd, os, math
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

print(os.listdir("/kaggle/input"))

BASE_PATH = "/kaggle/input"



## === cell 1
train = pd.read_csv(os.path.join(BASE_PATH, "train.csv"), nrows=2_000_000)
test = pd.read_csv(os.path.join(BASE_PATH, "test.csv"))
print(train.shape, test.shape)



## === cell 2
train = train.dropna()
train = train.loc[train["fare_amount"] > 0]
train = train.loc[(train["passenger_count"] > 0) & (train["passenger_count"] <= 6)]

lat_bounds = (39, 42)
lon_bounds = (-74.4, -72.8)
train = train.loc[
    train["pickup_latitude"].between(*lat_bounds)
    & train["dropoff_latitude"].between(*lat_bounds)
    & train["pickup_longitude"].between(*lon_bounds)
    & train["dropoff_longitude"].between(*lon_bounds)
]
test = test.loc[
    test["pickup_latitude"].between(*lat_bounds)
    & test["dropoff_latitude"].between(*lat_bounds)
    & test["pickup_longitude"].between(*lon_bounds)
    & test["dropoff_longitude"].between(*lon_bounds)
]



## === cell 3
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

train["abs_diff_sum"] = train["abs_diff_longitude"] + train["abs_diff_latitude"]
test["abs_diff_sum"] = test["abs_diff_longitude"] + test["abs_diff_latitude"]


def dist_haversine(row):
    R = 6371.0  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(
        math.radians,
        [
            row["pickup_latitude"],
            row["pickup_longitude"],
            row["dropoff_latitude"],
            row["dropoff_longitude"],
        ],
    )
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


train["dist_haversine_km"] = train.apply(dist_haversine, axis=1)
test["dist_haversine_km"] = test.apply(dist_haversine, axis=1)

train["log_dist_haversine_km"] = np.log1p(train["dist_haversine_km"])
test["log_dist_haversine_km"] = np.log1p(test["dist_haversine_km"])

train["pickup_lat_lon"] = train["pickup_latitude"] * train["pickup_longitude"]
test["pickup_lat_lon"] = test["pickup_latitude"] * test["pickup_longitude"]

train["dropoff_lat_lon"] = train["dropoff_latitude"] * train["dropoff_longitude"]
test["dropoff_lat_lon"] = test["dropoff_latitude"] * test["dropoff_longitude"]

for df in (train, test):
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["hour"] = df["pickup_datetime"].dt.hour
    df["weekday"] = df["pickup_datetime"].dt.weekday
    df["month"] = df["pickup_datetime"].dt.month

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

train["dist_haversine_km_sq"] = train["dist_haversine_km"] ** 2
test["dist_haversine_km_sq"] = test["dist_haversine_km"] ** 2



## === cell 4
feature_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "abs_diff_longitude",
    "abs_diff_latitude",
    "abs_diff_sum",
    "dist_haversine_km",
    "dist_haversine_km_sq",
    "log_dist_haversine_km",
    "pickup_lat_lon",
    "dropoff_lat_lon",
    "hour",
    "hour_sin",
    "hour_cos",
    "weekday",
    "month",
    "passenger_count",
]

train_shuffled = shuffle(train, random_state=42).reset_index(drop=True)
val = train_shuffled.iloc[int(0.9 * len(train_shuffled)) :]  # last 10 %
train_final = train_shuffled.iloc[: int(0.9 * len(train_shuffled))]

X_train = train_final[feature_cols]
y_train = np.log1p(train_final["fare_amount"])  # log‑transform target
X_val = val[feature_cols]
y_val = np.log1p(val["fare_amount"])  # log‑transform validation target
X_test = test[feature_cols]



## === cell 5
model = XGBRegressor(
    n_estimators=2000,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    eval_metric="rmse",
    n_jobs=5,
    random_state=42,
    reg_lambda=1.0,
)
model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=False,
)



## === cell 6
val_pred_log = model.predict(X_val)
val_pred = np.expm1(val_pred_log)  # revert log transform
val_pred = np.maximum(0, val_pred)  # ensure non‑negative fares
rmse_val = np.sqrt(mean_squared_error(np.expm1(y_val), val_pred))
print(f"Validation RMSE: {rmse_val:.5f}")



## === cell 7
test_pred_log = model.predict(X_test)
test_pred = np.expm1(test_pred_log)
test_pred = np.maximum(0, test_pred)



## === cell 8
submission = pd.DataFrame({"key": test["key"], "fare_amount": test_pred})
submission.to_csv("submission.csv", index=False)
print("submission.csv written – shape:", submission.shape)
