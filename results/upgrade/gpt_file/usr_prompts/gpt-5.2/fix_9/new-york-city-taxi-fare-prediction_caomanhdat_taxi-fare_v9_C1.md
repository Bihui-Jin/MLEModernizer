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
import os
import numpy as np
import pandas as pd

np.random.seed(0)

INPUT_DIR_CANDIDATES = ["../input", "/kaggle/input", "/kaggle/data"]
INPUT_DIR = None
for p in INPUT_DIR_CANDIDATES:
    if os.path.isdir(p):
        INPUT_DIR = p
        break
if INPUT_DIR is None:
    INPUT_DIR = "../input"

print("Using INPUT_DIR:", INPUT_DIR)
print(os.listdir(INPUT_DIR))



## === cell 1
train_path = os.path.join(INPUT_DIR, "train.csv")
test_path = os.path.join(INPUT_DIR, "test.csv")

usecols_train = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
usecols_test = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

dtype_common = {
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
dtype_train = dict(dtype_common)
dtype_train["fare_amount"] = "float32"

train = pd.read_csv(train_path, nrows=5000000, usecols=usecols_train, dtype=dtype_train)
test = pd.read_csv(test_path, usecols=usecols_test, dtype=dtype_common)



## === cell 2
train.head()



## === cell 3
test.head()



## === cell 4
train.describe()



## === cell 5
test.describe()



## === cell 6
train.isnull().sum()



## === cell 7
test.isnull().sum()




## === cell 8
def handle_date_inplace(df):
    s = df["pickup_datetime"]
    if s.dtype == object:
        s = s.str.replace(" UTC", "", regex=False)
    dt = pd.to_datetime(s, errors="coerce", cache=True)

    df["hour_of_day"] = dt.dt.hour.astype("int16")
    iso = dt.dt.isocalendar()
    wk = iso.week.astype("int16")
    df["week"] = wk
    df["month"] = dt.dt.month.astype("int16")
    df["year"] = dt.dt.year.astype("int16")
    df["day_of_year"] = dt.dt.dayofyear.astype("int16")
    df["week_of_year"] = wk
    df["weekday"] = dt.dt.weekday.astype("int16")
    df["quarter"] = dt.dt.quarter.astype("int16")
    df["day_of_month"] = dt.dt.day.astype("int16")

    df.drop(columns=["pickup_datetime"], inplace=True)
    return df


train = handle_date_inplace(train)
test = handle_date_inplace(test)




## === cell 9
def handle_distance_inplace(df):
    plon = df["pickup_longitude"].to_numpy(dtype=np.float64, copy=False)
    plat = df["pickup_latitude"].to_numpy(dtype=np.float64, copy=False)
    dlon = df["dropoff_longitude"].to_numpy(dtype=np.float64, copy=False)
    dlat = df["dropoff_latitude"].to_numpy(dtype=np.float64, copy=False)

    lon_dist = np.abs(plon - dlon)
    lat_dist = np.abs(plat - dlat)
    df["distance_travelled"] = np.sqrt(
        lon_dist * lon_dist + lat_dist * lat_dist
    ).astype(np.float32)

    R = 6371.0
    lat1 = np.deg2rad(plat)
    lat2 = np.deg2rad(dlat)
    dlat_r = lat2 - lat1
    dlon_r = np.deg2rad(dlon - plon)
    a = (
        np.sin(dlat_r / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon_r / 2.0) ** 2
    )
    df["haversine_km"] = (2.0 * R * np.arcsin(np.sqrt(a))).astype(np.float32)

    df["manhattan_approx"] = (lon_dist + lat_dist).astype(np.float32)
    return df


train = handle_distance_inplace(train)
test = handle_distance_inplace(test)



## === cell 10
train.describe()




## === cell 11
def clean_up_train(train):
    train = train.dropna()

    mask = (
        (train["fare_amount"] > 0)
        & (train["passenger_count"] > 0)
        & (train["passenger_count"] < 7)
        & (train["pickup_longitude"].between(-74.3, -73.7))
        & (train["dropoff_longitude"].between(-74.3, -73.7))
        & (train["pickup_latitude"].between(40.5, 41.0))
        & (train["dropoff_latitude"].between(40.5, 41.0))
        & (train["haversine_km"] > 0.05)
        & (train["haversine_km"] < 80.0)
        & (train["fare_amount"] < 200)
    )
    train = train[mask]

    same_loc = (train["pickup_longitude"] == train["dropoff_longitude"]) & (
        train["pickup_latitude"] == train["dropoff_latitude"]
    )
    train = train[~same_loc]

    max_reasonable = 2.5 + 10.0 * train["haversine_km"]
    train = train[train["fare_amount"] <= max_reasonable]

    denom_km = np.maximum(train["haversine_km"].astype(float), 0.2)
    fare_per_km = train["fare_amount"].astype(float) / denom_km
    train = train[(fare_per_km >= 0.5) & (fare_per_km <= 60.0)]

    return train


train = clean_up_train(train)
train.describe()




## === cell 12
def get_samples_output(train_df, test_df):
    feature_cols = test_df.drop("key", axis=1).columns
    X = train_df[feature_cols].copy()
    y = train_df["fare_amount"].copy()
    return X, y


samples_train, samples_label = get_samples_output(
    train.drop("key", axis=1, errors="ignore"), test
)

X_train = samples_train.to_numpy(dtype=np.float32, copy=False)
X_test = test.drop("key", axis=1).to_numpy(dtype=np.float32, copy=False)

X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

y = samples_label.to_numpy(dtype=np.float64, copy=False)



## === cell 13
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=120,
    max_depth=14,
    min_samples_leaf=1,
    random_state=0,
    n_jobs=-1,
)

y_train_log = np.log1p(y)
model.fit(X_train, y_train_log)

preds_log = model.predict(X_test)
preds = np.expm1(preds_log)
preds = np.clip(preds, 0.0, None)

submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": preds}, columns=["key", "fare_amount"]
)
submission.to_csv("submission.csv", index=False)

submission.head(20)
