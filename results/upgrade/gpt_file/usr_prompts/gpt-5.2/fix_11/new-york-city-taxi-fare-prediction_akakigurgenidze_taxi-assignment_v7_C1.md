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

3.12

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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
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

# 5. Code solution

## === cell 0
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

import xgboost as xgb

train_df = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
    nrows=10_000_000,
    dtype={"key": "string"},
)
test_df = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
    dtype={"key": "string"},
)



## === cell 1
train_df.head()



## === cell 2
test_df.head()



## === cell 3
train_df.dtypes



## === cell 4
train_df.describe()



## === cell 5
num_rows = len(train_df)
train_df = train_df[(train_df["fare_amount"] > 0)]
print(f"Drop {num_rows - len(train_df)} rows")




## === cell 6
def get_valid_mask(df):
    mask = (
        df["pickup_latitude"].between(40.5, 41.0)
        & df["dropoff_latitude"].between(40.5, 41.0)
        & df["pickup_longitude"].between(-74.3, -73.60)
        & df["dropoff_longitude"].between(-74.3, -73.60)
        & df["passenger_count"].between(1, 10)
    )
    return mask


before_len = len(train_df)
train_df = train_df[get_valid_mask(train_df)].copy()
print("Dropped invalid/outlier rows from training:", before_len - len(train_df))

test_valid_mask = get_valid_mask(test_df).to_numpy()
print(
    "Test rows (unchanged):",
    len(test_df),
    "| invalid rows:",
    int((~test_valid_mask).sum()),
)



## === cell 7
train_df.describe()



## === cell 8
train_df.isnull().sum()




## === cell 9
def preprocess_data(df):
    airport_lat_long = (40.644600, -73.779700)
    la_guardia_airport_lat_long = (40.7733, -73.8718)

    near_airport = (
        (
            (df["pickup_latitude"] <= airport_lat_long[0] + 0.005)
            & (df["pickup_latitude"] >= airport_lat_long[0] - 0.005)
            & (df["pickup_longitude"] <= airport_lat_long[1] + 0.005)
            & (df["pickup_longitude"] >= airport_lat_long[1] - 0.005)
        )
        | (
            (df["pickup_latitude"] <= la_guardia_airport_lat_long[0] + 0.002)
            & (df["pickup_latitude"] >= la_guardia_airport_lat_long[0] - 0.003)
            & (df["pickup_longitude"] <= la_guardia_airport_lat_long[1] + 0.005)
            & (df["pickup_longitude"] >= la_guardia_airport_lat_long[1] - 0.005)
        )
    ).astype(int)
    df["near_airport"] = near_airport

    df["manhattan_distance"] = abs(
        df["pickup_longitude"] - df["dropoff_longitude"]
    ) + abs(df["pickup_latitude"] - df["dropoff_latitude"])

    df["abs_lon_diff"] = (df["pickup_longitude"] - df["dropoff_longitude"]).abs()
    df["abs_lat_diff"] = (df["pickup_latitude"] - df["dropoff_latitude"]).abs()

    df["euclidean_distance"] = np.sqrt(
        (df["pickup_longitude"] - df["dropoff_longitude"]) ** 2
        + (df["pickup_latitude"] - df["dropoff_latitude"]) ** 2
    )

    R = 6371.0  # km
    lat1 = np.radians(df["pickup_latitude"].astype(float))
    lon1 = np.radians(df["pickup_longitude"].astype(float))
    lat2 = np.radians(df["dropoff_latitude"].astype(float))
    lon2 = np.radians(df["dropoff_longitude"].astype(float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    df["haversine_distance"] = 2.0 * R * np.arcsin(np.sqrt(a))

    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"], errors="coerce", utc=True
    )
    df["pickup_year"] = df["pickup_datetime"].dt.year
    df["pickup_month"] = df["pickup_datetime"].dt.month
    df["pickup_hour"] = df["pickup_datetime"].dt.hour
    df["pickup_day"] = df["pickup_datetime"].dt.dayofweek

    is_weekend = ((df["pickup_day"] >= 5) & (df["pickup_day"] <= 6)).astype(int)
    df["is_weekend"] = is_weekend

    is_holiday = (
        ((df["pickup_month"] == 12) & (df["pickup_datetime"].dt.day == 25))
        | ((df["pickup_month"] == 12) & (df["pickup_datetime"].dt.day == 26))
        | ((df["pickup_month"] == 12) & (df["pickup_datetime"].dt.day == 31))
        | ((df["pickup_month"] == 1) & (df["pickup_datetime"].dt.day == 1))
        | ((df["pickup_month"] == 7) & (df["pickup_datetime"].dt.day == 4))
    ).astype(int)
    df["is_holiday"] = is_holiday


preprocess_data(train_df)
preprocess_data(test_df)

time_cols = ["pickup_year", "pickup_month", "pickup_hour", "pickup_day"]
before_len = len(train_df)
train_df = train_df.dropna(subset=time_cols).copy()
print("Dropped train rows with invalid pickup_datetime:", before_len - len(train_df))



## === cell 10
train_df.head()



## === cell 11
train_df["near_airport"].sum()



## === cell 12
train_df["is_holiday"].sum()



## === cell 13
train_df["is_weekend"].sum()



## === cell 14
features = [
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
    "near_airport",
    "manhattan_distance",
    "abs_lon_diff",
    "abs_lat_diff",
    "euclidean_distance",
    "haversine_distance",
    "passenger_count",
    "pickup_year",
    "pickup_month",
    "pickup_hour",
    "is_weekend",
    "is_holiday",
]

for c in features:
    train_df[c] = pd.to_numeric(train_df[c], errors="coerce")
    test_df[c] = pd.to_numeric(test_df[c], errors="coerce")

train_df[features] = train_df[features].replace([np.inf, -np.inf], np.nan)
test_df[features] = test_df[features].replace([np.inf, -np.inf], np.nan)

distance_like = [
    "manhattan_distance",
    "abs_lon_diff",
    "abs_lat_diff",
    "euclidean_distance",
    "haversine_distance",
]
clip_bounds = {}
for c in distance_like:
    lo = float(train_df[c].quantile(0.001))
    hi = float(train_df[c].quantile(0.999))

    finite_vals = train_df[c].to_numpy()
    finite_vals = finite_vals[np.isfinite(finite_vals)]
    if finite_vals.size == 0:
        lo_fallback, hi_fallback = 0.0, 0.0
    else:
        lo_fallback, hi_fallback = float(np.min(finite_vals)), float(
            np.max(finite_vals)
        )

    if not np.isfinite(lo):
        lo = lo_fallback
    if not np.isfinite(hi):
        hi = hi_fallback
    if hi < lo:
        lo, hi = hi, lo

    clip_bounds[c] = (lo, hi)

for c, (lo, hi) in clip_bounds.items():
    train_df[c] = train_df[c].clip(lo, hi)
    test_df[c] = test_df[c].clip(lo, hi)

train_df[features] = train_df[features].replace([np.inf, -np.inf], np.nan)
test_df[features] = test_df[features].replace([np.inf, -np.inf], np.nan)

train_medians = train_df[features].median(numeric_only=True)

for c in ["near_airport", "is_weekend", "is_holiday"]:
    test_df[c] = test_df[c].fillna(0)

test_df[features] = test_df[features].fillna(train_medians)

before_len = len(train_df)
train_df = train_df.dropna(subset=features + ["fare_amount"]).copy()
print("Dropped train rows with NaNs in features/target:", before_len - len(train_df))

X = train_df[features].to_numpy(dtype=np.float32)
y = train_df["fare_amount"].to_numpy(dtype=np.float32)

fallback_fare = float(np.median(y))
print("Fallback fare (train median, original scale):", fallback_fare)

fare_cap_hi = float(np.quantile(y, 0.999))
if not np.isfinite(fare_cap_hi) or fare_cap_hi <= 0:
    fare_cap_hi = float(np.max(y))
print("Fare cap high (train 99.9th pct):", fare_cap_hi)



## === cell 15
X_test = test_df[features].to_numpy(dtype=np.float32)

if not np.isfinite(X_test).all():
    X_test = np.where(
        np.isfinite(X_test), X_test, train_medians.to_numpy(dtype=np.float32)
    )



## === cell 16
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=69)



## === cell 17
print(X.shape)
print(y.shape)

print(X_train.shape)
print(y_train.shape)

print(X_val.shape)
print(y_val.shape)



## === cell 18
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)



## === cell 19
validation_predictions_lr = lr_model.predict(X_val)
validation_rmse_lr = np.sqrt(mean_squared_error(y_val, validation_predictions_lr))
print("Validation RMSE (Linear Regression):", validation_rmse_lr)

train_predictions_lr = lr_model.predict(X)
train_rmse_lr = np.sqrt(mean_squared_error(y, train_predictions_lr))
print("Training RMSE (Linear Regression):", train_rmse_lr)



## === cell 20
test_predictions_lr = lr_model.predict(X_test)

submission_df_lr = pd.DataFrame(
    {
        "key": test_df["key"].astype("string").fillna(""),
        "fare_amount": test_predictions_lr,
    },
    columns=["key", "fare_amount"],
)
submission_df_lr.to_csv("lr_submission.csv", index=False)



## === cell 21
xgb_params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "eta": 0.05,
    "min_child_weight": 3,
    "gamma": 0.1,
    "seed": 42,
    "tree_method": "hist",
    "nthread": -1,
}

y_train_log = np.log1p(np.maximum(y_train, 0.0))
y_val_log = np.log1p(np.maximum(y_val, 0.0))

dtrain = xgb.DMatrix(X_train, label=y_train_log)
dval = xgb.DMatrix(X_val, label=y_val_log)
dtest = xgb.DMatrix(X_test)

dallTrain = xgb.DMatrix(X, label=np.log1p(np.maximum(y, 0.0)))


def rmse_original_scale(preds_log, dmatrix):
    y_true_log = dmatrix.get_label()
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(preds_log)
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    return "rmse_orig", rmse


watchlist = [(dtrain, "train"), (dval, "valid")]
xgb_model = xgb.train(
    xgb_params,
    dtrain,
    num_boost_round=5000,  # allow enough rounds; best_iteration chosen by early stopping
    evals=watchlist,
    feval=rmse_original_scale,
    maximize=False,
    early_stopping_rounds=100,  # no approximation/relaxed convergence; selects best iteration on valid
    verbose_eval=200,
)



## === cell 22
validation_predictions_xgb_log = xgb_model.predict(
    dval, iteration_range=(0, xgb_model.best_iteration + 1)
)
validation_predictions_xgb = np.expm1(validation_predictions_xgb_log)
validation_rmse_xgb = np.sqrt(mean_squared_error(y_val, validation_predictions_xgb))
print("Validation RMSE (XGBoost):", validation_rmse_xgb)
print("Best iteration chosen:", int(xgb_model.best_iteration))

train_predictions_xgb_log = xgb_model.predict(
    dallTrain, iteration_range=(0, xgb_model.best_iteration + 1)
)
train_predictions_xgb = np.expm1(train_predictions_xgb_log)
train_rmse_xgb = np.sqrt(mean_squared_error(y, train_predictions_xgb))
print("Train RMSE (XGBoost):", train_rmse_xgb)



## === cell 23
best_rounds = int(xgb_model.best_iteration) + 1
xgb_model_full = xgb.train(xgb_params, dallTrain, num_boost_round=best_rounds)

test_predictions_xgb_log = xgb_model_full.predict(dtest)
test_predictions_xgb = np.expm1(test_predictions_xgb_log)

test_predictions_xgb = np.where(test_valid_mask, test_predictions_xgb, fallback_fare)
test_predictions_xgb = np.maximum(test_predictions_xgb, 0.0)
test_predictions_xgb = np.minimum(test_predictions_xgb, fare_cap_hi)

submission_df_xgb = pd.DataFrame(
    {
        "key": test_df["key"].astype("string").fillna(""),
        "fare_amount": test_predictions_xgb,
    },
    columns=["key", "fare_amount"],
)

submission_df_xgb.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission_df_xgb.shape)
print("Submission dtypes:", submission_df_xgb.dtypes.to_dict())
print("Invalid test rows handled with fallback:", int((~test_valid_mask).sum()))
print("First 3 rows:\n", submission_df_xgb.head(3))
print(
    "Any NaN fare in submission?:",
    bool(pd.isna(submission_df_xgb["fare_amount"]).any()),
)
print("Fare stats:", submission_df_xgb["fare_amount"].describe().to_dict())
print("Full-train rounds used:", best_rounds)
