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
haversine==2.9.0
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
pass



## === cell 1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d
import seaborn as sns

import math
from math import sqrt

from numpy import absolute, mean, std

from sklearn import metrics
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import (
    PolynomialFeatures,
    scale,
    MinMaxScaler,
    StandardScaler,
)
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    KFold,
    RepeatedKFold,
)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn import neighbors
from sklearn.ensemble import GradientBoostingRegressor  # modelling



## === cell 2
train = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
    nrows=500000,
    parse_dates=["pickup_datetime"],
)
test = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")



## === cell 3
print(train.shape)
print(test.shape)



## === cell 4
train.head()



## === cell 5
train.dtypes



## === cell 6
train.describe()



## === cell 7
print(train.isnull().sum())



## === cell 8
train = train.dropna()



## === cell 9
print("Old size: %d" % len(train))



## === cell 10
fare_mask = (train["fare_amount"] >= 2.5) & (train["fare_amount"] <= 300)
train = train[fare_mask]



## === cell 11
passenger_mask = (train["passenger_count"] >= 0) & (train["passenger_count"] <= 6)
train = train[passenger_mask]



## === cell 12
lat_pick_mask = (train["pickup_latitude"] >= -90) & (train["pickup_latitude"] <= 90)
train = train[lat_pick_mask]



## === cell 13
lon_pick_mask = (train["pickup_longitude"] >= -180) & (train["pickup_longitude"] <= 180)
train = train[lon_pick_mask]



## === cell 14
lat_drop_mask = (train["dropoff_latitude"] >= -90) & (train["dropoff_latitude"] <= 90)
lon_drop_mask = (train["dropoff_longitude"] >= -180) & (
    train["dropoff_longitude"] <= 180
)
train = train[lat_drop_mask & lon_drop_mask]




## === cell 15
def select_outside_boundingbox(df, BB):
    filter_df = df.loc[
        (df["pickup_longitude"] < BB[0])
        | (df["pickup_longitude"] > BB[1])
        | (df["pickup_latitude"] < BB[2])
        | (df["pickup_latitude"] > BB[3])
        | (df["dropoff_longitude"] < BB[0])
        | (df["dropoff_longitude"] > BB[1])
        | (df["dropoff_latitude"] < BB[2])
        | (df["dropoff_latitude"] > BB[3])
    ]
    return filter_df


NYC_BB = (-74.5, -72.8, 40.5, 41.8)



## === cell 16
outliers = select_outside_boundingbox(train, NYC_BB)



## === cell 17
train = train.drop(outliers.index, axis=0)



## === cell 18
print("New size: %d" % len(train))



## === cell 19
test.dtypes



## === cell 20
pass




## === cell 21
def haversine_np(lat1, lon1, lat2, lon2):
    """Return distance in kilometers between two points given as arrays."""
    R = 6371.0
    lat1_rad, lat2_rad = np.radians(lat1), np.radians(lat2)
    dlat = lat2_rad - lat1_rad
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


train["H_Distance"] = haversine_np(
    train["pickup_latitude"],
    train["pickup_longitude"],
    train["dropoff_latitude"],
    train["dropoff_longitude"],
)




## === cell 22
def chebyshev(pickup_long, dropoff_long, pickup_lat, dropoff_lat):
    return np.maximum(
        np.abs(pickup_long - dropoff_long), np.abs(pickup_lat - dropoff_lat)
    )


train["Chebyshev"] = chebyshev(
    train["pickup_longitude"],
    train["dropoff_longitude"],
    train["pickup_latitude"],
    train["dropoff_latitude"],
)



## === cell 23
train.head()



## === cell 24
train["hour"] = train.pickup_datetime.dt.hour
train["day_of_week"] = train.pickup_datetime.dt.weekday
train["day_of_month"] = train.pickup_datetime.dt.day
train["week"] = train.pickup_datetime.dt.isocalendar().week
train["month"] = train.pickup_datetime.dt.month
train["year"] = train.pickup_datetime.dt.year - 2000

train["minute"] = train["pickup_datetime"].dt.minute
train["second"] = train["pickup_datetime"].dt.second
train["dayofyear"] = train["pickup_datetime"].dt.dayofyear



## === cell 25
train.head()



## === cell 26
train["fare_to_dist_ratio"] = train["fare_amount"] / (train["H_Distance"] + 0.0001)



## === cell 27
train = train.drop(train[train["H_Distance"] == 0.0].index, axis=0)




## === cell 28
def add_distances_from_airport(dataset):
    jfk = (40.639722, -73.778889)
    ewr = (40.6925, -74.168611)
    lga = (40.77725, -73.872611)

    dataset["pickup_jfk_distance"] = haversine_np(
        dataset["pickup_latitude"], dataset["pickup_longitude"], jfk[0], jfk[1]
    )
    dataset["pickup_ewr_distance"] = haversine_np(
        dataset["pickup_latitude"], dataset["pickup_longitude"], ewr[0], ewr[1]
    )
    dataset["pickup_lga_distance"] = haversine_np(
        dataset["pickup_latitude"], dataset["pickup_longitude"], lga[0], lga[1]
    )
    dataset["dropof_jfk_distance"] = haversine_np(
        dataset["dropoff_latitude"], dataset["dropoff_longitude"], jfk[0], jfk[1]
    )
    dataset["dropof_ewr_distance"] = haversine_np(
        dataset["dropoff_latitude"], dataset["dropoff_longitude"], ewr[0], ewr[1]
    )
    dataset["dropof_lga_distance"] = haversine_np(
        dataset["dropoff_latitude"], dataset["dropoff_longitude"], lga[0], lga[1]
    )
    return dataset


train = add_distances_from_airport(train)



## === cell 29
test["pickup_datetime"] = pd.to_datetime(test["pickup_datetime"])

test["H_Distance"] = haversine_np(
    test["pickup_latitude"],
    test["pickup_longitude"],
    test["dropoff_latitude"],
    test["dropoff_longitude"],
)

test["Chebyshev"] = chebyshev(
    test["pickup_longitude"],
    test["dropoff_longitude"],
    test["pickup_latitude"],
    test["dropoff_latitude"],
)

test["hour"] = test.pickup_datetime.dt.hour
test["day_of_week"] = test.pickup_datetime.dt.weekday
test["day_of_month"] = test.pickup_datetime.dt.day
test["week"] = test.pickup_datetime.dt.isocalendar().week
test["month"] = test.pickup_datetime.dt.month
test["year"] = test.pickup_datetime.dt.year - 2000

test["minute"] = test["pickup_datetime"].dt.minute
test["second"] = test["pickup_datetime"].dt.second
test["dayofyear"] = test["pickup_datetime"].dt.dayofyear

test = add_distances_from_airport(test)




## === cell 30
def downcast(df):
    for col in df.select_dtypes(include=["int64", "int32"]).columns:
        df[col] = df[col].astype("uint32")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    return df


train = downcast(train)
test = downcast(test)



## === cell 31
pass



## === cell 32
pass



## === cell 33
pass



## === cell 34
exclude_cols = [
    "fare_amount",
    "key",
    "pickup_datetime",
    "loc1",
    "loc2",
    "fare_to_dist_ratio",
]
feature_cols = [c for c in train.columns if c not in exclude_cols]

MAX_TRAIN_ROWS = 500000
if len(train) > MAX_TRAIN_ROWS:
    train = train.sample(n=MAX_TRAIN_ROWS, random_state=42)

X = train[feature_cols]
y = train["fare_amount"]

y_log = np.log1p(y)

X_train, X_val, y_train_log, y_val_log = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

X_train_np = X_train.to_numpy(dtype=np.float32, copy=False)
X_val_np = X_val.to_numpy(dtype=np.float32, copy=False)
y_train_log_np = y_train_log.to_numpy(dtype=np.float32, copy=False)
y_val_log_np = y_val_log.to_numpy(dtype=np.float32, copy=False)

model = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    random_state=42,
)
model.fit(X_train_np, y_train_log_np)

val_pred_log = model.predict(X_val_np)
val_pred = np.expm1(val_pred_log)
val_rmse = mean_squared_error(np.expm1(y_val_log_np), val_pred, squared=False)
print(f"Validation RMSE: {val_rmse:.5f}")



## === cell 35
test_features_np = test[feature_cols].to_numpy(dtype=np.float32, copy=False)
test_pred_log = model.predict(test_features_np)
test_pred = np.expm1(test_pred_log)

submission = pd.DataFrame({"key": test["key"], "fare_amount": test_pred})
submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
