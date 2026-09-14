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

3.7

# 3. Installed packages

geopandas==0.14.4
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
scipy==1.15.3
sklearn-pandas==2.2.0
tqdm==4.67.1

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
import os, random, pickle
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import lightgbm as lgb
import matplotlib.pyplot as plt

print("working dir:", os.getcwd())
print("input dirs:", os.listdir("../input"))



## === cell 1
train_path = "../input/new-york-city-taxi-fare-prediction/train.csv"
test_path = "../input/new-york-city-taxi-fare-prediction/test.csv"

sample_size = 5_000_000
dtype_train = {
    "key": "object",
    "fare_amount": "float32",
    "pickup_datetime": "object",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}
train = pd.read_csv(
    train_path,
    nrows=sample_size,
    dtype=dtype_train,
    usecols=list(dtype_train.keys()),
    engine="c",
)

test = pd.read_csv(
    test_path,
    dtype={
        "key": "object",
        "pickup_datetime": "object",
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "uint8",
    },
    engine="c",
)

test_id = test["key"].values.copy()  # keep for submission




## === cell 2
def preprocess(df):
    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"], utc=True, errors="coerce"
    )
    df["hour"] = df["pickup_datetime"].dt.hour.astype("uint8")
    df["minute"] = df["pickup_datetime"].dt.minute.astype("uint8")
    df["month"] = df["pickup_datetime"].dt.month.astype("uint8")
    df["day_of_week"] = df["pickup_datetime"].dt.dayofweek.astype("uint8")
    df["year"] = df["pickup_datetime"].dt.year.astype("uint16")
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24).astype("float32")
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24).astype("float32")
    df.drop(columns=["pickup_datetime", "key"], inplace=True, errors="ignore")
    return df


train = preprocess(train)
test = preprocess(test)




## === cell 3
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371.01 * c
    return km


train["distance"] = haversine(
    train["pickup_latitude"],
    train["pickup_longitude"],
    train["dropoff_latitude"],
    train["dropoff_longitude"],
).astype("float32")
test["distance"] = haversine(
    test["pickup_latitude"],
    test["pickup_longitude"],
    test["dropoff_latitude"],
    test["dropoff_longitude"],
).astype("float32")

train["log_distance"] = np.log1p(train["distance"]).astype("float32")
test["log_distance"] = np.log1p(test["distance"]).astype("float32")

for df in (train, test):
    df["delta_lat"] = (df["dropoff_latitude"] - df["pickup_latitude"]).astype("float32")
    df["delta_lon"] = (df["dropoff_longitude"] - df["pickup_longitude"]).astype(
        "float32"
    )
    df["abs_delta_lat"] = df["delta_lat"].abs()
    df["abs_delta_lon"] = df["delta_lon"].abs()
    df["hour_distance"] = (df["hour"].astype("float32") * df["distance"]).astype(
        "float32"
    )
    df["speed"] = (df["distance"] / (df["minute"] + 1e-3)).astype("float32")



## === cell 4
train = train[
    (train["fare_amount"] > 0)
    & (train["fare_amount"] < 200)
    & (train["distance"] < 100)
].copy()
train.dropna(inplace=True)
test.dropna(inplace=True)

y_fare = train["fare_amount"].astype("float32")
X = train.drop(columns=["fare_amount"])

assert all(
    pd.api.types.is_numeric_dtype(dt) for dt in X.dtypes
), "Non‑numeric columns remain!"

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y_fare, test_size=0.1, random_state=42
)

params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting": "gbdt",
    "learning_rate": 0.01,
    "num_leaves": 512,  # a bit larger capacity
    "max_depth": -1,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_data_in_leaf": 80,  # slight regularisation tweak
    "lambda_l2": 0.1,
    "verbosity": -1,
    "seed": 17,
}

d_train = lgb.Dataset(X_train, label=y_train)
d_valid = lgb.Dataset(X_valid, label=y_valid, reference=d_train)

model = lgb.train(
    params,
    d_train,
    num_boost_round=8000,
    valid_sets=[d_valid],
    callbacks=[
        lgb.early_stopping(stopping_rounds=400, verbose=False),
        lgb.log_evaluation(period=200),
    ],
)

pred_valid = model.predict(X_valid, num_iteration=model.best_iteration)
rmse_original = sqrt(mean_squared_error(y_valid, pred_valid))
print(f"Validation RMSE (original scale): {rmse_original:.5f}")



## === cell 5
test_pred = model.predict(test, num_iteration=model.best_iteration)
test_pred = np.where(test_pred > 0, test_pred, 0)  # enforce non‑negative fares

submission = pd.DataFrame({"key": test_id, "fare_amount": test_pred})
submission.to_csv("submission_lgb.csv", index=False)
print("Submission saved to submission_lgb.csv – rows:", len(submission))

plt.figure(figsize=(10, 6))
plt.hist(test_pred, bins=100, color="steelblue", edgecolor="black")
plt.title("Distribution of predicted fare amounts")
plt.xlabel("Fare amount")
plt.ylabel("Count")
plt.show()
