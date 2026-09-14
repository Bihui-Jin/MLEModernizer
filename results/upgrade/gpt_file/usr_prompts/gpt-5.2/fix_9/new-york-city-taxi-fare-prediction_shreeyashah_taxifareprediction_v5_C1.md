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

3.14

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
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


np.random.seed(42)



## === cell 1
TRAIN_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
TEST_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"



## === cell 2
train_usecols = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
train_dtypes = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16",
}
df_train = pd.read_csv(
    TRAIN_PATH,
    nrows=2_000_000,
    usecols=train_usecols,
    dtype=train_dtypes,
    parse_dates=["pickup_datetime"],
)



## === cell 3
pass



## === cell 4
pass



## === cell 5
pass



## === cell 6
pass



## === cell 7
pass



## === cell 8
df_train = df_train.dropna()



## === cell 9
df_train = df_train[(df_train["fare_amount"] > 1) & (df_train["fare_amount"] < 100)]



## === cell 10
df_train = df_train[
    (df_train["passenger_count"] >= 1) & (df_train["passenger_count"] <= 6)
]



## === cell 11
df_train = df_train[
    (df_train["pickup_longitude"] > -75)
    & (df_train["pickup_longitude"] < -72)
    & (df_train["dropoff_longitude"] > -75)
    & (df_train["dropoff_longitude"] < -72)
    & (df_train["pickup_latitude"] > 40)
    & (df_train["pickup_latitude"] < 42)
    & (df_train["dropoff_latitude"] > 40)
    & (df_train["dropoff_latitude"] < 42)
]




## === cell 12
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c  # kilometers




## === cell 13
df_train["distance_km"] = haversine(
    df_train["pickup_latitude"],
    df_train["pickup_longitude"],
    df_train["dropoff_latitude"],
    df_train["dropoff_longitude"],
)
df_train = df_train[(df_train["distance_km"] > 0.1) & (df_train["distance_km"] < 30)]




## === cell 14
def add_time_features(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df = df.copy()
    df["pickup_year"] = dt.dt.year.astype(np.int16)
    df["pickup_month"] = dt.dt.month.astype(np.int8)
    df["pickup_day"] = dt.dt.day.astype(np.int8)
    df["pickup_hour"] = dt.dt.hour.astype(np.int8)
    df["pickup_weekday"] = dt.dt.weekday.astype(np.int8)
    return df


df_train = add_time_features(df_train)
df_train = df_train.dropna(
    subset=[
        "pickup_year",
        "pickup_month",
        "pickup_day",
        "pickup_hour",
        "pickup_weekday",
    ]
)



## === cell 15
feature_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "distance_km",
    "pickup_year",
    "pickup_month",
    "pickup_day",
    "pickup_hour",
    "pickup_weekday",
]

X = df_train[feature_cols]
y = df_train["fare_amount"]



## === cell 16
X = X.astype(np.float32, copy=False)
y = y.astype(np.float32, copy=False)

X_np = X.to_numpy(dtype=np.float32, copy=False)
y_np = y.to_numpy(dtype=np.float32, copy=False)

mask_finite = np.isfinite(X_np).all(axis=1) & np.isfinite(y_np)
if not mask_finite.all():
    X = X.loc[mask_finite]
    y = y.loc[mask_finite]
    X_np = X.to_numpy(dtype=np.float32, copy=False)
    y_np = y.to_numpy(dtype=np.float32, copy=False)

train_feature_medians = X.median(axis=0)
train_target_median = float(np.median(y_np))



## === cell 17
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=2,
    bootstrap=True,
    max_samples=5000,
    n_jobs=1,
)



## === cell 18
from sklearn.linear_model import LinearRegression

lr = LinearRegression()



## === cell 19
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

svr_scaled = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("svr", SVR(kernel="rbf")),
    ]
)

br = BaggingRegressor(
    n_estimators=10,
    base_estimator=svr_scaled,
    max_samples=5000,
    bootstrap=True,
    random_state=2,
)



## === cell 20
from sklearn.ensemble import VotingRegressor

vr = VotingRegressor([("lr", lr), ("rf", rf), ("svr", br)], verbose=False, n_jobs=-1)



## === cell 21
vr.fit(X, y)

pred_fit = vr.predict(X).astype(np.float32, copy=False)
y_true_fit = y.to_numpy(dtype=np.float32, copy=False)

m = np.isfinite(pred_fit) & np.isfinite(y_true_fit)
pred_fit = pred_fit[m]
y_true_fit = y_true_fit[m]

if pred_fit.size >= 10 and float(np.var(pred_fit)) > 1e-8:
    a, b = np.polyfit(pred_fit, y_true_fit, deg=1)
    a = float(a)
    b = float(b)
else:
    a, b = 1.0, 0.0

print(f"Calibration: y ≈ {a:.6f} * pred + {b:.6f}")



## === cell 22
test_usecols = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
test_dtypes = {
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16",
}
df_test = pd.read_csv(
    TEST_PATH,
    usecols=test_usecols,
    dtype=test_dtypes,
    parse_dates=["pickup_datetime"],
)



## === cell 23
key = df_test["key"].copy()



## === cell 24
df_test["distance_km"] = haversine(
    df_test["pickup_latitude"],
    df_test["pickup_longitude"],
    df_test["dropoff_latitude"],
    df_test["dropoff_longitude"],
)

df_test = add_time_features(df_test)

for c in ["pickup_year", "pickup_month", "pickup_day", "pickup_hour", "pickup_weekday"]:
    if df_test[c].isna().any():
        df_test[c] = (
            df_test[c]
            .fillna(df_test[c].mode(dropna=True).iloc[0])
            .astype(df_test[c].dtype)
        )

valid_geo = (
    (df_test["pickup_longitude"] > -75)
    & (df_test["pickup_longitude"] < -72)
    & (df_test["dropoff_longitude"] > -75)
    & (df_test["dropoff_longitude"] < -72)
    & (df_test["pickup_latitude"] > 40)
    & (df_test["pickup_latitude"] < 42)
    & (df_test["dropoff_latitude"] > 40)
    & (df_test["dropoff_latitude"] < 42)
)
valid_dist = (df_test["distance_km"] > 0.1) & (df_test["distance_km"] < 30)
test_valid_mask = (valid_geo & valid_dist).to_numpy()



## === cell 25
X_test = df_test[feature_cols].astype(np.float32, copy=False)

X_test = X_test.replace([np.inf, -np.inf], np.nan)
if X_test.isna().to_numpy().any():
    X_test = X_test.fillna(train_feature_medians)



## === cell 26
y_pred = vr.predict(X_test)
y_pred = np.asarray(y_pred, dtype=np.float32)

if np.isfinite(a) and np.isfinite(b):
    y_pred[test_valid_mask] = (a * y_pred[test_valid_mask] + b).astype(np.float32)

y_pred[~test_valid_mask] = np.float32(train_target_median)



## === cell 27
y_pred = np.clip(y_pred, 0.0, 500.0)



## === cell 28
results = pd.DataFrame({"key": key, "fare_amount": y_pred})
print(results.head())
print("Submission rows:", len(results))



## === cell 29
results.to_csv("submission.csv", index=False)
print("Wrote submission.csv")
