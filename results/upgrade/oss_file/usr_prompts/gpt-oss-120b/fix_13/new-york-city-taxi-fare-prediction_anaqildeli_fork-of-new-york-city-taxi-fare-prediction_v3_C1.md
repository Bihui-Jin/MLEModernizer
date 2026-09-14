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

folium==0.20.0
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
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

train_fields = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
test_fields = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

train_path = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
test_path = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"

train = pd.read_csv(
    train_path,
    nrows=3_000_000,  # sample for faster iteration
    usecols=train_fields,
    parse_dates=["pickup_datetime"],
    skipinitialspace=True,
)
test = pd.read_csv(
    test_path,
    usecols=test_fields,
    parse_dates=["pickup_datetime"],
    skipinitialspace=True,
)

print(f"Train shape: {train.shape}, Test shape: {test.shape}")



## === cell 1
for df in (train, test):
    df["year"] = df["pickup_datetime"].dt.year.astype(np.int16)
    df["month"] = df["pickup_datetime"].dt.month.astype(np.int8)
    df["day"] = df["pickup_datetime"].dt.day.astype(np.int8)
    df["hour"] = df["pickup_datetime"].dt.hour.astype(np.int8)
    df["minute"] = df["pickup_datetime"].dt.minute.astype(np.int8)
    df["weekday"] = df["pickup_datetime"].dt.weekday.astype(np.int8)
    df["is_weekend"] = (df["weekday"] >= 5).astype(np.int8)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24).astype(np.float32)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24).astype(np.float32)

train.drop(columns=["pickup_datetime"], inplace=True)
test.drop(columns=["pickup_datetime"], inplace=True)




## === cell 2
def haversine_vec(lon1, lat1, lon2, lat2):
    R = 6371.0  # Earth radius in km
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


train["distance"] = haversine_vec(
    train["pickup_longitude"],
    train["pickup_latitude"],
    train["dropoff_longitude"],
    train["dropoff_latitude"],
).astype(np.float32)

test["distance"] = haversine_vec(
    test["pickup_longitude"],
    test["pickup_latitude"],
    test["dropoff_longitude"],
    test["dropoff_latitude"],
).astype(np.float32)

for df in (train, test):
    df["delta_lat"] = (
        (df["dropoff_latitude"] - df["pickup_latitude"]).abs().astype(np.float32)
    )
    df["delta_lon"] = (
        (df["dropoff_longitude"] - df["pickup_longitude"]).abs().astype(np.float32)
    )
    df["manhattan_km"] = (df["delta_lat"] + df["delta_lon"]) * 111.0
    df["euclidean_km"] = np.sqrt(df["delta_lat"] ** 2 + df["delta_lon"] ** 2) * 111.0
    df["abs_delta_diff"] = (df["delta_lat"] - df["delta_lon"]).abs() * 111.0



## === cell 3
numeric_cols = [
    "fare_amount",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "weekday",
    "is_weekend",
    "distance",
    "delta_lat",
    "delta_lon",
    "manhattan_km",
    "euclidean_km",
    "abs_delta_diff",
    "hour_sin",
    "hour_cos",
]

train[numeric_cols] = train[numeric_cols].astype(np.float32)
test[numeric_cols[1:]] = test[numeric_cols[1:]].astype(np.float32)

test = test.fillna(0)

feature_cols = [c for c in train.columns if c not in ("fare_amount", "key")]

X = train[feature_cols].values
y = np.log1p(train["fare_amount"].values)  # log‑transform target

valid_mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
X = X[valid_mask]
y = y[valid_mask]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

gbr = HistGradientBoostingRegressor(
    max_iter=2500,
    learning_rate=0.01,
    max_depth=12,
    random_state=42,
    loss="squared_error",
    early_stopping=True,
    validation_fraction=0.1,
)

gbr.fit(X_train, y_train)

val_pred_log = gbr.predict(X_val)
val_pred = np.expm1(val_pred_log)

train_pred_log = gbr.predict(X_train)
train_pred = np.expm1(train_pred_log)
scale_factor = np.expm1(y_train).mean() / train_pred.mean()
val_pred_scaled = val_pred * scale_factor

rmse = mean_squared_error(np.expm1(y_val), val_pred_scaled, squared=False)
print(f"Validation RMSE (scaled): {rmse}")



## === cell 4
gbr.fit(X, y)

test_keys = test["key"].values
test_features = test[feature_cols].values
test_pred_log = gbr.predict(test_features)
test_pred = np.expm1(test_pred_log) * scale_factor
test_pred = test_pred.astype(np.float32)

submission = pd.DataFrame({"key": test_keys, "fare_amount": test_pred})
submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}, shape {submission.shape}")
