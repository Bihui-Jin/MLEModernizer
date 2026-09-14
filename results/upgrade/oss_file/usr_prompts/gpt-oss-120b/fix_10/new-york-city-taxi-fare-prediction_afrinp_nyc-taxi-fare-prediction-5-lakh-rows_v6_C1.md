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

3.8

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
tf_keras==2.18.0

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
import numpy as np
import random
import os

np.random.seed(42)
random.seed(42)
os.environ["PYTHONHASHSEED"] = "42"




## === cell 1
df = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
    parse_dates=["pickup_datetime"],
    nrows=500_000,  # keep the original sample size
)
df.dropna(inplace=True)




## === cell 2
test = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
    parse_dates=["pickup_datetime"],
)




## === cell 3
nyc_min_longitude = -74.3
nyc_max_longitude = -72
nyc_min_latitude = 40.63
nyc_max_latitude = 42




## === cell 4
lon_cond = df["pickup_longitude"].between(nyc_min_longitude, nyc_max_longitude) & df[
    "dropoff_longitude"
].between(nyc_min_longitude, nyc_max_longitude)
lat_cond = df["pickup_latitude"].between(nyc_min_latitude, nyc_max_latitude) & df[
    "dropoff_latitude"
].between(nyc_min_latitude, nyc_max_latitude)
df = df.loc[lon_cond & lat_cond].copy()  # copy to avoid SettingWithCopy warnings




## === cell 5
df.loc[df["passenger_count"] == 0, "passenger_count"] = 1




## === cell 6
df = df[(df["fare_amount"] > 0) & (df["fare_amount"] <= 100)]




## === cell 7
def haversine_distance_np(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


df_lat1 = df["pickup_latitude"].to_numpy(dtype=np.float32, copy=False)
df_lon1 = df["pickup_longitude"].to_numpy(dtype=np.float32, copy=False)
df_lat2 = df["dropoff_latitude"].to_numpy(dtype=np.float32, copy=False)
df_lon2 = df["dropoff_longitude"].to_numpy(dtype=np.float32, copy=False)

df["haversine_distance"] = haversine_distance_np(df_lat1, df_lon1, df_lat2, df_lon2)

test_lat1 = test["pickup_latitude"].to_numpy(dtype=np.float32, copy=False)
test_lon1 = test["pickup_longitude"].to_numpy(dtype=np.float32, copy=False)
test_lat2 = test["dropoff_latitude"].to_numpy(dtype=np.float32, copy=False)
test_lon2 = test["dropoff_longitude"].to_numpy(dtype=np.float32, copy=False)

test["haversine_distance"] = haversine_distance_np(
    test_lat1, test_lon1, test_lat2, test_lon2
)




## === cell 8
df["year"] = df["pickup_datetime"].dt.year
df["month"] = df["pickup_datetime"].dt.month
df["day"] = df["pickup_datetime"].dt.day
df["day_of_week"] = df["pickup_datetime"].dt.dayofweek
df["hour"] = df["pickup_datetime"].dt.hour
df.drop(columns=["pickup_datetime"], inplace=True)




## === cell 9
test["year"] = test["pickup_datetime"].dt.year
test["month"] = test["pickup_datetime"].dt.month
test["day"] = test["pickup_datetime"].dt.day
test["day_of_week"] = test["pickup_datetime"].dt.dayofweek
test["hour"] = test["pickup_datetime"].dt.hour
test.drop(columns=["pickup_datetime"], inplace=True)




## === cell 10
test.drop(columns=["key"], inplace=True)
df.drop(columns=["key"], inplace=True)




## === cell 11
print(df.isnull().sum())
print(test.isnull().sum())




## === cell 12
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor




## === cell 13
X = df.drop(columns=["fare_amount"])
y = df["fare_amount"]




## === cell 14
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)




## === cell 15
model = GradientBoostingRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=8,
    subsample=0.8,
    max_features=0.8,
    random_state=42,
)

X_train_np = X_train.to_numpy(dtype=np.float32, copy=False)
y_train_np = y_train.to_numpy(dtype=np.float32, copy=False)
model.fit(X_train_np, y_train_np)

X_val_np = X_val.to_numpy(dtype=np.float32, copy=False)
val_pred = model.predict(X_val_np)
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f"Validation RMSE: {val_rmse:0.2f}")




## === cell 16
test_np = test.to_numpy(dtype=np.float32, copy=False)




## === cell 17
pred = model.predict(test_np)
pred = np.clip(pred, 0, 100)  # enforce reasonable fare bounds
print(pred.shape)




## === cell 18
submission = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
)
submission["fare_amount"] = pred
submission.to_csv("submission.csv", index=False)
