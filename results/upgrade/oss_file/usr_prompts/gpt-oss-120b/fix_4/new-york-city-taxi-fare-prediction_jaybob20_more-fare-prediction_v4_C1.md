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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
seaborn==0.12.2
sklearn-pandas==2.2.0
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
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import Birch
import xgboost as xgb



## === cell 1
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")



## === cell 2
train = train.dropna()
train = train.loc[(train["fare_amount"] > 0) & (train["fare_amount"] < 300)]

lon_min = train["pickup_longitude"].min()
lon_max = train["pickup_longitude"].max()
lat_min = train["pickup_latitude"].min()
lat_max = train["pickup_latitude"].max()
drop_lon_min = train["dropoff_longitude"].min()
drop_lon_max = train["dropoff_longitude"].max()
drop_lat_min = train["dropoff_latitude"].min()
drop_lat_max = train["dropoff_latitude"].max()

train = train.loc[
    (train["pickup_longitude"] > lon_min)
    & (train["pickup_longitude"] < lon_max)
    & (train["pickup_latitude"] > lat_min)
    & (train["pickup_latitude"] < lat_max)
    & (train["dropoff_longitude"] > drop_lon_min)
    & (train["dropoff_longitude"] < drop_lon_max)
    & (train["dropoff_latitude"] > drop_lat_min)
    & (train["dropoff_latitude"] < drop_lat_max)
    & (train["passenger_count"] <= 8)
]

test = test.loc[
    (test["pickup_longitude"] > lon_min)
    & (test["pickup_longitude"] < lon_max)
    & (test["pickup_latitude"] > lat_min)
    & (test["pickup_latitude"] < lat_max)
    & (test["dropoff_longitude"] > drop_lon_min)
    & (test["dropoff_longitude"] < drop_lon_max)
    & (test["dropoff_latitude"] > drop_lat_min)
    & (test["dropoff_latitude"] < drop_lat_max)
    & (test["passenger_count"] <= 8)
]




## === cell 3
def haversine_vec(df):
    lon1 = np.radians(df["pickup_longitude"].values)
    lat1 = np.radians(df["pickup_latitude"].values)
    lon2 = np.radians(df["dropoff_longitude"].values)
    lat2 = np.radians(df["dropoff_latitude"].values)

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6367 * c  # Earth radius in km


train["haversine"] = haversine_vec(train)
test["haversine"] = haversine_vec(test)


train["pickup_datetime"] = pd.to_datetime(train["pickup_datetime"])
test["pickup_datetime"] = pd.to_datetime(test["pickup_datetime"])

for df in [train, test]:
    df["hour_of_day"] = df["pickup_datetime"].dt.hour
    df["day"] = df["pickup_datetime"].dt.day
    df["week"] = df["pickup_datetime"].dt.isocalendar().week.astype(int)
    df["month"] = df["pickup_datetime"].dt.month
    df["day_of_year"] = df["pickup_datetime"].dt.dayofyear
    df["week_of_year"] = df["pickup_datetime"].dt.isocalendar().week.astype(int)



## === cell 4
feature_cols = [
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
    "haversine",
]  # no zero‑distance columns
scaler = StandardScaler()
train[feature_cols] = scaler.fit_transform(train[feature_cols])
test[feature_cols] = scaler.transform(test[feature_cols])



## === cell 5
coords = [
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
]
concat_coords = pd.concat([train[coords], test[coords]], ignore_index=True)

birch = Birch(branching_factor=50, threshold=0.5, compute_labels=True).fit(
    concat_coords
)
labels = birch.labels_
train["cluster"] = labels[: len(train)]
test["cluster"] = labels[len(train) :]



## === cell 6
pca_feature = "haversine"

train[pca_feature] = train[pca_feature].fillna(0)
test[pca_feature] = test[pca_feature].fillna(0)

train["pca00"] = train[pca_feature]
test["pca00"] = test[pca_feature]

train["pca01"] = 0.0
train["pca02"] = 0.0
test["pca01"] = 0.0
test["pca02"] = 0.0



## === cell 7
good_dist = ["pca00", "pca01", "pca02", "cluster"]
train_features_to_keep = [
    "fare_amount",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
] + good_dist

train = train[train_features_to_keep]

test_features_to_keep = [
    "key",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
] + good_dist
test = test[test_features_to_keep]

x_pred = test.drop("key", axis=1)



## === cell 8
x_train, x_valid, y_train, y_valid = train_test_split(
    train.drop("fare_amount", axis=1),
    train["fare_amount"],
    test_size=0.2,
    random_state=123,
)




## === cell 9
def XGBmodel(x_tr, x_va, y_tr, y_va):
    dtrain = xgb.DMatrix(x_tr, label=y_tr)
    dvalid = xgb.DMatrix(x_va, label=y_va)
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.3,
        "max_depth": 4,
        "min_child_weight": 3,
        "seed": 42,
        "tree_method": "hist",
    }
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=300,
        evals=[(dvalid, "validation")],
        early_stopping_rounds=10,
        verbose_eval=False,
    )
    return model


model = XGBmodel(x_train, x_valid, y_train, y_valid)



## === cell 10
prediction = model.predict(xgb.DMatrix(x_pred), ntree_limit=model.best_ntree_limit)
submission = pd.DataFrame({"key": test["key"], "fare_amount": np.round(prediction, 2)})
submission.to_csv("sub_fare.csv", index=False)
