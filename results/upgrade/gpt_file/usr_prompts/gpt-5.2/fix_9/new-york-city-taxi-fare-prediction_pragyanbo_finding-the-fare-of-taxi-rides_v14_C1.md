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
seaborn==0.12.2
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

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

np.random.seed(42)



## === cell 1
train_path = "../input/train.csv"
test_path = "../input/test.csv"

train_df = pd.read_csv(train_path, nrows=800000)
train_df.shape



## === cell 2
test_df = pd.read_csv(test_path)
test_df.shape



## === cell 3
train_df.head(5)



## === cell 4
train_df.isnull().sum()



## === cell 5
train_df.dropna(inplace=True)



## === cell 6
train_df.describe()



## === cell 7
train_df = train_df[train_df["fare_amount"] > 0].copy()
train_df.shape




## === cell 8
def distance(lat1, lon1, lat2, lon2):
    lat1 = np.asarray(lat1, dtype=np.float64)
    lon1 = np.asarray(lon1, dtype=np.float64)
    lat2 = np.asarray(lat2, dtype=np.float64)
    lon2 = np.asarray(lon2, dtype=np.float64)

    r = 6371.0088  # Earth radius in km
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    lat1r = np.radians(lat1)
    lat2r = np.radians(lat2)

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    )
    km = 2.0 * r * np.arcsin(np.sqrt(a))
    miles = km * 0.6213712
    return miles




## === cell 9
train_df["distance"] = distance(
    train_df.pickup_latitude,
    train_df.pickup_longitude,
    train_df.dropoff_latitude,
    train_df.dropoff_longitude,
)

test_df["distance"] = distance(
    test_df.pickup_latitude,
    test_df.pickup_longitude,
    test_df.dropoff_latitude,
    test_df.dropoff_longitude,
)



## === cell 10
nyc_lat_min, nyc_lat_max = 40.5, 41.0
nyc_lon_min, nyc_lon_max = -74.5, -73.0

train_df = train_df[
    (train_df["pickup_latitude"].between(nyc_lat_min, nyc_lat_max))
    & (train_df["dropoff_latitude"].between(nyc_lat_min, nyc_lat_max))
    & (train_df["pickup_longitude"].between(nyc_lon_min, nyc_lon_max))
    & (train_df["dropoff_longitude"].between(nyc_lon_min, nyc_lon_max))
].copy()

train_df = train_df[train_df["distance"] < 15].copy()

train_df = train_df[
    (train_df["passenger_count"] >= 1) & (train_df["passenger_count"] < 10)
].copy()

train_df = train_df[
    (train_df["fare_amount"] >= 2.5) & (train_df["fare_amount"] <= 250)
].copy()

train_df["abs_lon_diff"] = (
    train_df["pickup_longitude"] - train_df["dropoff_longitude"]
).abs()
train_df["abs_lat_diff"] = (
    train_df["pickup_latitude"] - train_df["dropoff_latitude"]
).abs()
train_df = train_df[
    (train_df["abs_lon_diff"] < 2.0) & (train_df["abs_lat_diff"] < 2.0)
].copy()

train_df = train_df[
    ~((train_df["distance"] < 0.05) & (train_df["fare_amount"] > 8.0))
].copy()

train_df.describe()



## === cell 11
train_df["pickup_datetime"] = pd.to_datetime(
    train_df["pickup_datetime"], errors="coerce", utc=True
)
test_df["pickup_datetime"] = pd.to_datetime(
    test_df["pickup_datetime"], errors="coerce", utc=True
)

train_df = train_df.dropna(subset=["pickup_datetime"]).copy()

for df in (train_df, test_df):
    df["hour"] = df["pickup_datetime"].dt.hour.astype("float32")
    df["year"] = df["pickup_datetime"].dt.year.astype("float32")
    df["month"] = df["pickup_datetime"].dt.month.astype("float32")
    df["weekday"] = df["pickup_datetime"].dt.weekday.astype("float32")
    df["is_weekend"] = (df["weekday"] >= 5).astype("float32")

time_cols = ["hour", "year", "month", "weekday", "is_weekend"]
test_df[time_cols] = test_df[time_cols].fillna(0.0)




## === cell 12
def add_geo_features(df):
    df = df.copy()
    df["abs_lon_diff"] = (
        (df["pickup_longitude"] - df["dropoff_longitude"]).abs().astype("float32")
    )
    df["abs_lat_diff"] = (
        (df["pickup_latitude"] - df["dropoff_latitude"]).abs().astype("float32")
    )
    df["manhattan_dist"] = (df["abs_lon_diff"] + df["abs_lat_diff"]).astype("float32")
    return df


train_df = add_geo_features(train_df)
test_df = add_geo_features(test_df)



## === cell 13
for df in (train_df, test_df):
    df["distance_sq"] = (df["distance"].astype("float32") ** 2).astype("float32")
    df["dist_x_pax"] = (
        df["distance"].astype("float32") * df["passenger_count"].astype("float32")
    ).astype("float32")




## === cell 14
def bearing(lat1, lon1, lat2, lon2):
    """
    Add trip bearing (direction). Returns bearing in radians in [-pi, pi].
    """
    lat1 = np.radians(np.asarray(lat1, dtype=np.float64))
    lon1 = np.radians(np.asarray(lon1, dtype=np.float64))
    lat2 = np.radians(np.asarray(lat2, dtype=np.float64))
    lon2 = np.radians(np.asarray(lon2, dtype=np.float64))

    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    brng = np.arctan2(y, x)
    return brng


NYC_C_LAT, NYC_C_LON = 40.7580, -73.9855

for df in (train_df, test_df):
    df["bearing"] = bearing(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    ).astype("float32")
    df["center_dist"] = distance(
        df["pickup_latitude"],
        df["pickup_longitude"],
        NYC_C_LAT,
        NYC_C_LON,
    ).astype("float32")



## === cell 15
nyc_lat_min, nyc_lat_max = 40.5, 41.0
nyc_lon_min, nyc_lon_max = -74.5, -73.0

test_valid_mask = (
    test_df["pickup_latitude"].between(nyc_lat_min, nyc_lat_max)
    & test_df["dropoff_latitude"].between(nyc_lat_min, nyc_lat_max)
    & test_df["pickup_longitude"].between(nyc_lon_min, nyc_lon_max)
    & test_df["dropoff_longitude"].between(nyc_lon_min, nyc_lon_max)
    & (test_df["distance"] < 15)
    & (test_df["passenger_count"] >= 1)
    & (test_df["passenger_count"] < 10)
)



## === cell 16
feat_cols_s = [
    "distance",
    "distance_sq",
    "dist_x_pax",
    "manhattan_dist",
    "abs_lon_diff",
    "abs_lat_diff",
    "bearing",
    "center_dist",
    "passenger_count",
    "hour",
    "year",
    "month",
    "weekday",
    "is_weekend",
]
X = train_df[feat_cols_s]
y = train_df["fare_amount"]



## === cell 17
split_ts = train_df["pickup_datetime"].quantile(0.7)
train_mask = train_df["pickup_datetime"] <= split_ts
valid_mask = ~train_mask

X_train, y_train = X.loc[train_mask], y.loc[train_mask]
X_valid, y_valid = X.loc[valid_mask], y.loc[valid_mask]

print(
    f"Time split timestamp: {split_ts} | train rows: {len(X_train)} | valid rows: {len(X_valid)}"
)



## === cell 18
from sklearn.ensemble import RandomForestRegressor

r_reg = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
r_reg.fit(X_train, y_train)



## === cell 19
fare_low = float(np.percentile(y, 0.5))
fare_high = float(np.percentile(y, 99.5))
fare_baseline = float(np.median(y))

test_feat_finite_mask = np.isfinite(
    test_df[feat_cols_s].to_numpy(dtype=np.float64)
).all(axis=1)
test_valid_mask = test_valid_mask & pd.Series(
    test_feat_finite_mask, index=test_df.index
)

y_pred_final = np.full(
    shape=(len(test_df),), fill_value=fare_baseline, dtype=np.float64
)
y_pred_valid = r_reg.predict(test_df.loc[test_valid_mask, feat_cols_s])
y_pred_valid = np.clip(y_pred_valid, fare_low, fare_high)
y_pred_final[test_valid_mask.values] = y_pred_valid

submission = pd.DataFrame({"key": test_df["key"], "fare_amount": y_pred_final})
submission.to_csv("submission.csv", index=False)
submission.head()



## === cell 20
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

model_l = Pipeline(
    steps=[("standard_scaler", StandardScaler()), ("lin_reg", LinearRegression())]
)
model_l.fit(X_train, y_train)

y_pred_lr = np.full(shape=(len(test_df),), fill_value=fare_baseline, dtype=np.float64)
y_pred_lr_valid = model_l.predict(test_df.loc[test_valid_mask, feat_cols_s])
y_pred_lr_valid = np.clip(y_pred_lr_valid, fare_low, fare_high)
y_pred_lr[test_valid_mask.values] = y_pred_lr_valid

submission_lr = pd.DataFrame({"key": test_df["key"], "fare_amount": y_pred_lr})
submission_lr.to_csv("linear_reg.csv", index=False)



## === cell 21
print("Wrote submission.csv (RandomForest) and linear_reg.csv (LinearRegression).")
print(
    f"Clipped predictions to [{fare_low:.2f}, {fare_high:.2f}] based on cleaned train fares."
)
print(
    f"Invalid/filtered test rows: {(~test_valid_mask).sum()} / {len(test_valid_mask)}; baseline used: {fare_baseline:.2f}"
)
print(f"Features used: {feat_cols_s}")
