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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn.model_selection import train_test_split
from math import sin, cos, sqrt, atan2, radians
import xgboost
from sklearn.preprocessing import StandardScaler
import os

print(os.listdir("../input"))




## === cell 1
df = pd.read_csv(
    "../input/train.csv", nrows=3000000
)  # raised from 1 500 000 to 3 000 000 rows




## === cell 2
test = pd.read_csv("../input/test.csv")




## === cell 3
testkey = test.key




## === cell 4
df = df.dropna(how="any", axis="rows")




## === cell 5
len(df)




## === cell 6
df.head()




## === cell 7
df.describe()




## === cell 8
l = df[
    (df.pickup_latitude > 42.5)
    | (df.pickup_latitude < 40.0)
    | (df.dropoff_latitude > 42.5)
    | (df.dropoff_latitude < 40.0)
    | (df.pickup_longitude > -73.0)
    | (df.pickup_longitude < -75.0)
    | (df.dropoff_longitude > -73.0)
    | (df.dropoff_longitude < -75.0)
].index




## === cell 9
df = df.drop(l, axis=0)




## === cell 10
z = df[
    (df.fare_amount > 350.0)
    | (df.fare_amount < 0.0)
    | (df.passenger_count > 7.0)
    | (df.passenger_count < 0.0)
].index




## === cell 11
df = df.drop(z, axis=0)




## === cell 12
len(df)




## === cell 13
def distlatlong(lon1, lat1, lon2, lat2):
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    lon1 = radians(lon1)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (sin(dlat / 2)) ** 2 + cos(lat1) * cos(lat2) * (sin(dlon / 2)) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = 6373.0 * c
    return distance




## === cell 14
r = np.radians
lon1 = r(df["pickup_longitude"].values)
lat1 = r(df["pickup_latitude"].values)
lon2 = r(df["dropoff_longitude"].values)
lat2 = r(df["dropoff_latitude"].values)
dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
df["dist"] = 6373.0 * c
df["log_dist"] = np.log1p(df["dist"])
df["dist_per_passenger"] = df["dist"] / (df["passenger_count"] + 1)




## === cell 15
lon1_t = r(test["pickup_longitude"].values)
lat1_t = r(test["pickup_latitude"].values)
lon2_t = r(test["dropoff_longitude"].values)
lat2_t = r(test["dropoff_latitude"].values)
dlon_t = lon2_t - lon1_t
dlat_t = lat2_t - lat1_t
a_t = (
    np.sin(dlat_t / 2) ** 2 + np.cos(lat1_t) * np.cos(lat2_t) * np.sin(dlon_t / 2) ** 2
)
c_t = 2 * np.arctan2(np.sqrt(a_t), np.sqrt(1 - a_t))
test["dist"] = 6373.0 * c_t
test["log_dist"] = np.log1p(test["dist"])
test["dist_per_passenger"] = test["dist"] / (test["passenger_count"] + 1)




## === cell 16
test.head()




## === cell 17
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])




## === cell 18
test["pickup_datetime"] = pd.to_datetime(test["pickup_datetime"])




## === cell 19
df.info()




## === cell 20
df["latenights"] = (df["pickup_datetime"].dt.hour < 5).astype(int)
test["latenights"] = (test["pickup_datetime"].dt.hour < 5).astype(int)

df["hour"] = df["pickup_datetime"].dt.hour
test["hour"] = test["pickup_datetime"].dt.hour

df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
test["hour_sin"] = np.sin(2 * np.pi * test["hour"] / 24)
test["hour_cos"] = np.cos(2 * np.pi * test["hour"] / 24)




## === cell 21
df.pickup_datetime[0].weekday()




## === cell 22
df["weekday"] = (df["pickup_datetime"].dt.weekday > 4).astype(int)
test["weekday"] = (test["pickup_datetime"].dt.weekday > 4).astype(int)




## === cell 23
df.head()




## === cell 24
df["year"] = df["pickup_datetime"].dt.year
df["month"] = df["pickup_datetime"].dt.month




## === cell 25
test["year"] = test["pickup_datetime"].dt.year
test["month"] = test["pickup_datetime"].dt.month




## === cell 26
df["day"] = df["pickup_datetime"].dt.day




## === cell 27
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
df["day_sin"] = np.sin(2 * np.pi * df["day"] / 31)
df["day_cos"] = np.cos(2 * np.pi * df["day"] / 31)

df.head()




## === cell 28
test["day"] = test["pickup_datetime"].dt.day




## === cell 29
test["month_sin"] = np.sin(2 * np.pi * test["month"] / 12)
test["month_cos"] = np.cos(2 * np.pi * test["month"] / 12)
test["day_sin"] = np.sin(2 * np.pi * test["day"] / 31)
test["day_cos"] = np.cos(2 * np.pi * test["day"] / 31)

test.head()




## === cell 30
feat = df.drop(["key", "pickup_datetime"], axis=1)




## === cell 31
test = test.drop(["key", "pickup_datetime"], axis=1)




## === cell 32
test.year.unique()




## === cell 33
feat.head()




## === cell 34
test.head()




## === cell 35
label = np.log1p(feat["fare_amount"])




## === cell 36
feat = feat.drop("fare_amount", axis=1)




## === cell 37
xtr, xts, ytr, yts = train_test_split(feat, label, test_size=0.2, random_state=42)




## === cell 38
xgbtrain = xgboost.DMatrix(xtr, ytr)
xgbtest = xgboost.DMatrix(xts, yts)
xgbfinaltest = xgboost.DMatrix(test)




## === cell 39
params = {
    "eval_metric": "rmse",
    "objective": "reg:squarederror",
    "learning_rate": 0.01,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "max_depth": 6,
    "lambda": 1.5,
    "alpha": 0.0,
    "gamma": 0.2,
    "min_child_weight": 5,
    "seed": 42,
    "tree_method": "hist",  # faster histogram algorithm
}




## === cell 40
xgbmodel = xgboost.train(
    params,
    dtrain=xgbtrain,
    num_boost_round=10000,
    early_stopping_rounds=100,
    evals=[(xgbtest, "test")],
)




## === cell 41
pred_log = xgbmodel.predict(xgbfinaltest)




## === cell 42
pred = np.expm1(pred_log)




## === cell 43
finalset = pd.DataFrame({"key": testkey, "fare_amount": pred})




## === cell 44
finalset = finalset[["key", "fare_amount"]]




## === cell 45
finalset.head()




## === cell 46
finalset.to_csv("finaloutput.csv", index=False)
