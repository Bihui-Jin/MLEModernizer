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
import os, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 1
train_path = "../input/new-york-city-taxi-fare-prediction/train.csv"
test_path = "../input/new-york-city-taxi-fare-prediction/test.csv"
sample_sub_path = "../input/new-york-city-taxi-fare-prediction/sample_submission.csv"

train = pd.read_csv(train_path, nrows=2_000_000)
test = pd.read_csv(test_path)
sample_submission = pd.read_csv(sample_sub_path)




## === cell 2
train.dropna(inplace=True)

train = train[(train["fare_amount"] > 0) & (train["fare_amount"] < 200)]




## === cell 3
def haversine_np(lon1, lat1, lon2, lat2):
    """Vectorized haversine distance (km)."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    km = 6371.0 * 2 * np.arcsin(np.sqrt(a))
    return km


def add_time_features(df):
    dt = pd.to_datetime(df["pickup_datetime"])
    df["pickup_hour"] = dt.dt.hour
    df["pickup_dayofweek"] = dt.dt.dayofweek
    df["pickup_month"] = dt.dt.month
    df["is_weekend"] = (df["pickup_dayofweek"] >= 5).astype(int)
    return df


def add_cyclic_features(df):
    df["hour_sin"] = np.sin(2 * np.pi * df["pickup_hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["pickup_hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["pickup_dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["pickup_dayofweek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["pickup_month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["pickup_month"] / 12)
    return df


train = add_time_features(train)
test = add_time_features(test)

train["distance"] = haversine_np(
    train["pickup_longitude"],
    train["pickup_latitude"],
    train["dropoff_longitude"],
    train["dropoff_latitude"],
)
test["distance"] = haversine_np(
    test["pickup_longitude"],
    test["pickup_latitude"],
    test["dropoff_longitude"],
    test["dropoff_latitude"],
)

train["distance_per_passenger"] = train["distance"] / train["passenger_count"]
test["distance_per_passenger"] = test["distance"] / test["passenger_count"]

train["delta_lat"] = train["dropoff_latitude"] - train["pickup_latitude"]
test["delta_lat"] = test["dropoff_latitude"] - test["pickup_latitude"]
train["delta_lon"] = train["dropoff_longitude"] - train["pickup_longitude"]
test["delta_lon"] = test["dropoff_longitude"] - test["pickup_longitude"]
train["manhattan"] = np.abs(train["delta_lat"]) + np.abs(train["delta_lon"])
test["manhattan"] = np.abs(test["delta_lat"]) + np.abs(test["delta_lon"])
train["log_manhattan"] = np.log1p(train["manhattan"])
test["log_manhattan"] = np.log1p(test["manhattan"])

train = add_cyclic_features(train)
test = add_cyclic_features(test)

train["log_distance"] = np.log1p(train["distance"])
test["log_distance"] = np.log1p(test["distance"])

train["log_distance_per_passenger"] = np.log1p(train["distance_per_passenger"])
test["log_distance_per_passenger"] = np.log1p(test["distance_per_passenger"])

train["log_passenger"] = np.log1p(train["passenger_count"])
test["log_passenger"] = np.log1p(test["passenger_count"])

train["is_night"] = ((train["pickup_hour"] >= 22) | (train["pickup_hour"] <= 5)).astype(
    int
)
test["is_night"] = ((test["pickup_hour"] >= 22) | (test["pickup_hour"] <= 5)).astype(
    int
)

cols_to_drop = ["key", "pickup_datetime"]
train = train.drop(columns=cols_to_drop)
test = test.drop(columns=cols_to_drop)




## === cell 4
y_train = train["fare_amount"]
y_train_log = np.log1p(y_train).astype(np.float32).values

X_train = train.drop("fare_amount", axis=1).astype(np.float32)  # float32 conversion
X_test = test.copy().astype(np.float32)




## === cell 5
X_train_np = X_train.values  # shape (n_samples, n_features)
X_test_np = X_test.values
y_train_log_np = y_train_log  # already a NumPy array

cv = KFold(n_splits=5, shuffle=True, random_state=0)

oof_preds = np.zeros(len(X_train_np))
test_preds = []

params = {
    "objective": "regression",
    "max_bin": 400,
    "learning_rate": 0.02,
    "num_leaves": 1024,
    "feature_fraction": 0.90,
    "bagging_fraction": 0.80,
    "bagging_freq": 5,
    "metric": "rmse",
    "verbosity": -1,
    "min_data_in_leaf": 5,
    "n_jobs": -1,
}

for fold, (train_idx, valid_idx) in enumerate(cv.split(X_train_np, y_train_log_np)):
    X_tr, X_val = X_train_np[train_idx], X_train_np[valid_idx]
    y_tr_log, y_val_log = y_train_log_np[train_idx], y_train_log_np[valid_idx]

    lgb_train = lgb.Dataset(X_tr, y_tr_log, free_raw_data=False)
    lgb_valid = lgb.Dataset(X_val, y_val_log, reference=lgb_train, free_raw_data=False)

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=5000,
        valid_sets=[lgb_train, lgb_valid],
        callbacks=[
            lgb.log_evaluation(period=10),
            lgb.early_stopping(stopping_rounds=200, verbose=False),
        ],
    )

    oof_preds[valid_idx] = np.expm1(
        model.predict(X_val, num_iteration=model.best_iteration)
    )
    test_preds.append(
        np.expm1(model.predict(X_test_np, num_iteration=model.best_iteration))
    )




## === cell 6
rmse = np.sqrt(mean_squared_error(y_train, oof_preds))
print(f"OOF RMSE: {rmse:.5f}")

pd.DataFrame(oof_preds, columns=["oof_pred"]).to_csv("oof_train_lgb.csv", index=False)




## === cell 7
if len(test_preds) > 0:
    final_test_pred = np.mean(test_preds, axis=0)
else:
    final_test_pred = np.full(len(X_test), y_train.mean())

submission = sample_submission.copy()
submission["fare_amount"] = final_test_pred
submission_path = "submission_lightgbm.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
