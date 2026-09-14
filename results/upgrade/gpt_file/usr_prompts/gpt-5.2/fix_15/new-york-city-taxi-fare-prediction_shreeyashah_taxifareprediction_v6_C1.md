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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import warnings
import time

warnings.filterwarnings("ignore")

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:50]:
        print(os.path.join(dirname, filename))
    break

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass



## === cell 1
import matplotlib.pyplot as plt
import seaborn as sns




## === cell 2
def add_datetime_features(df, col="pickup_datetime"):
    dt = df[col]
    if not pd.api.types.is_datetime64_any_dtype(dt):
        dt = pd.to_datetime(dt, errors="coerce", utc=True, cache=True)
    df["pickup_year"] = dt.dt.year.astype("float32")
    df["pickup_month"] = dt.dt.month.astype("float32")
    df["pickup_day"] = dt.dt.day.astype("float32")
    df["pickup_hour"] = dt.dt.hour.astype("float32")
    df["pickup_weekday"] = dt.dt.weekday.astype("float32")
    df["pickup_minute"] = dt.dt.minute.astype("float32")
    df["is_weekend"] = (dt.dt.weekday >= 5).astype("float32")
    return df




## === cell 3
train_path = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
usecols_train = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
dtype_train = {
    "fare_amount": "float32",
    "pickup_datetime": "string",  # faster read; converted once in add_datetime_features
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "float32",
}
t0 = time.time()
df_train = pd.read_csv(
    train_path,
    nrows=3_000_000,
    usecols=usecols_train,
    dtype=dtype_train,
    engine="c",
    low_memory=False,
)
print("Loaded train in %.2fs, shape=%s" % (time.time() - t0, df_train.shape))
df_train.head()



## === cell 4
pass



## === cell 5
df_train = add_datetime_features(df_train, "pickup_datetime")
df_train = df_train.drop(columns=["pickup_datetime"])



## === cell 6
df_train.info()



## === cell 7
df_train.shape



## === cell 8
df_train.describe()



## === cell 9
df_train.isna().sum()



## === cell 10
df_train = df_train.dropna()



## === cell 11
df_train.isna().sum()



## === cell 12
df_train = df_train[(df_train["fare_amount"] > 1) & (df_train["fare_amount"] < 100)]



## === cell 13
df_train = df_train[
    (df_train["passenger_count"] >= 1) & (df_train["passenger_count"] <= 6)
]



## === cell 14
df_train = df_train[
    (df_train["pickup_longitude"] > -74.5)
    & (df_train["pickup_longitude"] < -72.8)
    & (df_train["dropoff_longitude"] > -74.5)
    & (df_train["dropoff_longitude"] < -72.8)
    & (df_train["pickup_latitude"] > 40.5)
    & (df_train["pickup_latitude"] < 41.9)
    & (df_train["dropoff_latitude"] > 40.5)
    & (df_train["dropoff_latitude"] < 41.9)
]

df_train = df_train[
    (df_train["pickup_longitude"] != 0)
    & (df_train["pickup_latitude"] != 0)
    & (df_train["dropoff_longitude"] != 0)
    & (df_train["dropoff_latitude"] != 0)
]




## === cell 15
def haversine(lat1, lon1, lat2, lon2):
    lat1 = np.radians(np.asarray(lat1, dtype=np.float32))
    lon1 = np.radians(np.asarray(lon1, dtype=np.float32))
    lat2 = np.radians(np.asarray(lat2, dtype=np.float32))
    lon2 = np.radians(np.asarray(lon2, dtype=np.float32))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = np.float32(2.0) * np.arcsin(np.sqrt(a))
    return np.float32(6371.0) * c  # kilometers




## === cell 16
df_train["distance_km"] = haversine(
    df_train["pickup_latitude"].to_numpy(copy=False),
    df_train["pickup_longitude"].to_numpy(copy=False),
    df_train["dropoff_latitude"].to_numpy(copy=False),
    df_train["dropoff_longitude"].to_numpy(copy=False),
).astype("float32")

df_train = df_train[(df_train["distance_km"] > 0.1) & (df_train["distance_km"] < 30)]



## === cell 17
target_col = "fare_amount"
feature_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "pickup_year",
    "pickup_month",
    "pickup_day",
    "pickup_hour",
    "pickup_weekday",
    "pickup_minute",
    "is_weekend",
    "distance_km",
]

df_train["passenger_count"] = df_train["passenger_count"].round().clip(1, 6)

df_train = df_train.replace([np.inf, -np.inf], np.nan).dropna(
    subset=feature_cols + [target_col]
)

X = df_train[feature_cols]
y = df_train[target_col]

fallback_fare = float(y.median())



## === cell 18
X.head(), y.head()



## === cell 19
from sklearn.ensemble import ExtraTreesRegressor

rf = ExtraTreesRegressor(
    n_estimators=400,
    max_depth=18,
    random_state=2,
    bootstrap=True,
    max_samples=None,
    n_jobs=-1,
)



## === cell 20
from sklearn.linear_model import LinearRegression

lr = LinearRegression()



## === cell 21
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor

sr = SVR(kernel="rbf")

br = BaggingRegressor(
    n_estimators=10,
    estimator=sr,
    max_samples=5000,
    bootstrap=True,
    random_state=2,
)



## === cell 22
from sklearn.ensemble import VotingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

lr_poly = Pipeline(
    steps=[
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("lr", lr),
    ]
)

br_scaled = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("br", br),
    ]
)

vr = VotingRegressor(
    [("lr", lr_poly), ("rf", rf), ("svr", br_scaled)], verbose=True, n_jobs=-1
)

X_fit = np.ascontiguousarray(X.to_numpy(dtype=np.float32, copy=False))
y_fit = np.ascontiguousarray(y.to_numpy(dtype=np.float32, copy=False))

t0 = time.time()
vr.fit(X_fit, y_fit)
print("Model fit in %.2fs" % (time.time() - t0))



## === cell 23
test_path = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"
usecols_test = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
dtype_test = {
    "key": "string",
    "pickup_datetime": "string",  # faster read; converted once in add_datetime_features
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "float32",
}
df_test = pd.read_csv(
    test_path,
    usecols=usecols_test,
    dtype=dtype_test,
    engine="c",
    low_memory=False,
)
df_test.head()



## === cell 24
key = df_test["key"].copy()

df_test = add_datetime_features(df_test, "pickup_datetime")
df_test = df_test.drop(columns=["pickup_datetime", "key"])



## === cell 25
df_test["distance_km"] = haversine(
    df_test["pickup_latitude"].to_numpy(copy=False),
    df_test["pickup_longitude"].to_numpy(copy=False),
    df_test["dropoff_latitude"].to_numpy(copy=False),
    df_test["dropoff_longitude"].to_numpy(copy=False),
).astype("float32")



## === cell 26
df_test["passenger_count"] = df_test["passenger_count"].round().clip(1, 6)
df_test = df_test.replace([np.inf, -np.inf], np.nan)

valid_mask = (
    df_test["pickup_longitude"].between(-74.5, -72.8)
    & df_test["dropoff_longitude"].between(-74.5, -72.8)
    & df_test["pickup_latitude"].between(40.5, 41.9)
    & df_test["dropoff_latitude"].between(40.5, 41.9)
    & df_test["passenger_count"].between(1, 6)
    & df_test["distance_km"].between(0.1, 30)
    & (df_test["pickup_longitude"] != 0)
    & (df_test["pickup_latitude"] != 0)
    & (df_test["dropoff_longitude"] != 0)
    & (df_test["dropoff_latitude"] != 0)
)

df_test_valid = df_test.loc[valid_mask, feature_cols].copy()

train_medians = X.median(numeric_only=True)
df_test_valid[feature_cols] = df_test_valid[feature_cols].fillna(train_medians)

X_test_valid = np.ascontiguousarray(
    df_test_valid.to_numpy(dtype=np.float32, copy=False)
)



## === cell 27
y_pred_full = np.full(shape=(len(df_test),), fill_value=fallback_fare, dtype=np.float64)

t0 = time.time()
y_pred_valid = vr.predict(X_test_valid)
print("Predicted valid rows in %.2fs" % (time.time() - t0))

y_pred_valid = np.clip(y_pred_valid, 0.0, None)

y_pred_full[valid_mask.to_numpy()] = y_pred_valid



## === cell 28
results = pd.DataFrame({"key": key, "fare_amount": y_pred_full})
print(results.head())
print("Submission rows:", len(results))
print(
    "Valid predicted rows:",
    int(valid_mask.sum()),
    "Invalid/fallback rows:",
    int((~valid_mask).sum()),
)



## === cell 29
results.to_csv("submission.csv", index=False)
print("Wrote submission.csv")
