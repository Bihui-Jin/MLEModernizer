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
import pandas as pd

TRAIN_PATH = "/kaggle/input/train.csv"
TEST_PATH = "/kaggle/input/test.csv"
SAMPLE_SUB_PATH = "/kaggle/input/sample_submission.csv"

train = pd.read_csv(TRAIN_PATH, nrows=1_000_000)
test = pd.read_csv(TEST_PATH)



## === cell 1
train.shape



## === cell 2
train.head()



## === cell 3
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt



## === cell 4
train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], errors="coerce", utc=False
)
train["hour"] = train["pickup_datetime"].dt.hour
train["day"] = train["pickup_datetime"].dt.day
train["week"] = train["pickup_datetime"].dt.isocalendar().week.astype("int16")
train["month"] = train["pickup_datetime"].dt.month
train["day_of_year"] = train["pickup_datetime"].dt.dayofyear
train["week_of_year"] = train["pickup_datetime"].dt.isocalendar().week.astype("int16")



## === cell 5
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], errors="coerce", utc=False
)
test["hour"] = test["pickup_datetime"].dt.hour
test["day"] = test["pickup_datetime"].dt.day
test["week"] = test["pickup_datetime"].dt.isocalendar().week.astype("int16")
test["month"] = test["pickup_datetime"].dt.month
test["day_of_year"] = test["pickup_datetime"].dt.dayofyear
test["week_of_year"] = test["pickup_datetime"].dt.isocalendar().week.astype("int16")



## === cell 6
train.head()
train = train.dropna(how="any", axis="rows")



## === cell 7
train = train.loc[(train["fare_amount"] > 0) & (train["fare_amount"] < 200)]
train = train.loc[(train["pickup_longitude"] > -75) & (train["pickup_longitude"] < -72)]
train = train.loc[(train["pickup_latitude"] > 40) & (train["pickup_latitude"] < 45)]
train = train.loc[
    (train["dropoff_longitude"] > -75) & (train["dropoff_longitude"] < -72)
]
train = train.loc[(train["dropoff_latitude"] > 40) & (train["dropoff_latitude"] < 45)]
train = train.loc[train["passenger_count"].between(1, 8)]

test["passenger_count"] = pd.to_numeric(test["passenger_count"], errors="coerce")
test["passenger_count"] = (
    test["passenger_count"].clip(lower=1, upper=8).fillna(1).astype(int)
)



## === cell 8
train["abs_diff_longitude"] = (
    train["pickup_longitude"] - train["dropoff_longitude"]
).abs()
train["abs_diff_latitude"] = (
    train["pickup_latitude"] - train["dropoff_latitude"]
).abs()



## === cell 9
test["abs_diff_longitude"] = (
    test["pickup_longitude"] - test["dropoff_longitude"]
).abs()
test["abs_diff_latitude"] = (test["pickup_latitude"] - test["dropoff_latitude"]).abs()



## === cell 10
import numpy as np


def haversine_km(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1.astype(float))
    lat1 = np.radians(lat1.astype(float))
    lon2 = np.radians(lon2.astype(float))
    lat2 = np.radians(lat2.astype(float))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    R = 6371.0
    return R * c


train["haversine_km"] = haversine_km(
    train["pickup_longitude"],
    train["pickup_latitude"],
    train["dropoff_longitude"],
    train["dropoff_latitude"],
)
test["haversine_km"] = haversine_km(
    test["pickup_longitude"],
    test["pickup_latitude"],
    test["dropoff_longitude"],
    test["dropoff_latitude"],
)

train["haversine_km"] = train["haversine_km"].clip(lower=0.0, upper=200.0)
test["haversine_km"] = test["haversine_km"].clip(lower=0.0, upper=200.0)

train = train.replace([np.inf, -np.inf], np.nan).dropna(axis=0)



## === cell 11
train = train.loc[~((train["haversine_km"] < 0.05) & (train["fare_amount"] > 7.5))]

train = train.loc[train["haversine_km"].between(0.05, 80.0)]
eps = 1e-6
fare_per_km = train["fare_amount"] / (train["haversine_km"] + eps)
train = train.loc[fare_per_km.between(0.5, 50.0)]



## === cell 12
train.head()



## === cell 13
train.head()



## === cell 14
_ = sns.barplot(data=train, x="passenger_count", y="fare_amount")



## === cell 15
feature_names = [
    "abs_diff_longitude",
    "abs_diff_latitude",
    "haversine_km",
    "passenger_count",
    "hour",
    "day",
    "week",
    "month",
    "day_of_year",
    "week_of_year",
]
feature_names



## === cell 16
label_name = "fare_amount"
label_name



## === cell 17
X_train = train[feature_names]
y_train = train[label_name]
X_test = test[feature_names]

X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)
medians = X_train.median(numeric_only=True)
X_train = X_train.fillna(medians)
X_test = X_test.fillna(medians)



## === cell 18
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb



## === cell 19
regr = LinearRegression()
regr.fit(X_train, y_train)
regr_prediction = regr.predict(X_test)



## === cell 20
knr = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "knn",
            KNeighborsRegressor(
                n_neighbors=30, weights="distance", metric="minkowski", p=2
            ),
        ),
    ]
)
knr.fit(X_train, y_train)
knr_prediction = knr.predict(X_test)



## === cell 21
rfr = RandomForestRegressor(
    random_state=42,
    n_jobs=-1,
    n_estimators=350,
    min_samples_leaf=1,
)
rfr.fit(X_train, y_train)
rfr_prediction = rfr.predict(X_test)



## === cell 22
predictions = (regr_prediction * 2 + rfr_prediction * 2 + knr_prediction) / 5



## === cell 23
predictions



## === cell 24
submission = pd.read_csv(SAMPLE_SUB_PATH)
pred_df = pd.DataFrame({"key": test["key"].values, "fare_amount": predictions})

submission = submission[["key"]].merge(pred_df, on="key", how="left")
submission["fare_amount"] = submission["fare_amount"].fillna(float(y_train.mean()))

lo = float(y_train.quantile(0.001))
hi = float(y_train.quantile(0.999))
submission["fare_amount"] = submission["fare_amount"].clip(lower=lo, upper=hi)



## === cell 25
submission.head()



## === cell 26
submission.to_csv("./simplenewyorktaxi.csv", index=False)
print("Wrote submission:", "./simplenewyorktaxi.csv", "rows:", len(submission))
