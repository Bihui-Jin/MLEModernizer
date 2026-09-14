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
import pandas as pd  # data processing, CSV file I/O
import os

print(os.listdir("../input"))



## === cell 1
train = pd.read_csv("../input/train.csv", nrows=1000000)
test = pd.read_csv("../input/test.csv")
train.head()



## === cell 2
test.head()



## === cell 3
train.shape



## === cell 4
test.shape



## === cell 5
train.dtypes.value_counts()



## === cell 6
test.dtypes.value_counts()



## === cell 7
train.isnull().sum()



## === cell 8
train = train.dropna()
train.isnull().sum()



## === cell 9
(train == 0).astype(int).sum()



## === cell 10
train = train.loc[~(train == 0).any(axis=1)]



## === cell 11
(train == 0).astype(int).sum()



## === cell 12
train.shape



## === cell 13
train.describe()



## === cell 14
object_data = train.dtypes == object
categoricals = train.columns[object_data]
categoricals



## === cell 15
train.drop("key", axis=1, inplace=True)
train.head()



## === cell 16
import datetime as dt


def date_extraction(df):
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["year"] = df["pickup_datetime"].dt.year
    df["month"] = df["pickup_datetime"].dt.month
    df["weekday"] = df["pickup_datetime"].dt.weekday
    df["hour"] = df["pickup_datetime"].dt.hour
    df.drop("pickup_datetime", axis=1, inplace=True)
    return df


train = date_extraction(train)
test = date_extraction(test)
train.head()




## === cell 17
def long_lat_distance(df):
    df["Longitude_distance"] = np.radians(
        df["pickup_longitude"] - df["dropoff_longitude"]
    )
    df["Latitude_distance"] = np.radians(df["pickup_latitude"] - df["dropoff_latitude"])
    df["distance_travelled/10e3"] = (
        (df["Longitude_distance"] ** 2 + df["Latitude_distance"] ** 2) ** 0.5
    ) * 1000
    return df


for df in [train, test]:
    long_lat_distance(df)




## === cell 18
def haversine(df):
    r = 6371000  # Earth radius in metres
    lat1 = np.radians(df["pickup_latitude"])
    lat2 = np.radians(df["dropoff_latitude"])
    dlat = np.radians(df["dropoff_latitude"] - df["pickup_latitude"])
    dlon = np.radians(df["dropoff_longitude"] - df["pickup_longitude"])
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    df["haversine_km"] = (r * c) / 1000
    return df


for df in [train, test]:
    haversine(df)



## === cell 19
train.dtypes.value_counts()



## === cell 20
train.head()



## === cell 21
test.head()



## === cell 22
print("Any nulls in train:", train.isnull().sum().sum())
print("Any nulls in test :", test.isnull().sum().sum())



## === cell 23
train["haversine_km"] = train["haversine_km"].fillna(train["haversine_km"].median())



## === cell 24
from sklearn.ensemble import RandomForestRegressor

feature_cols = [c for c in train.columns if c != "fare_amount"]
X = train[feature_cols]
y = train["fare_amount"]



## === cell 25
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, ElasticNetCV, LassoCV, RidgeCV
import matplotlib.pyplot as plt
import seaborn as sns




## === cell 26
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))




## === cell 27
feat_cols = (
    [c for c in train_1.columns if c != "fare_amount"]
    if "train_1" in globals()
    else feature_cols
)
X_1 = train[feat_cols]
y_1 = train["fare_amount"]
X_train, X_valid, y_train, y_valid = train_test_split(
    X_1, y_1, test_size=0.25, random_state=42
)

X_train_np = X_train.values
X_valid_np = X_valid.values
y_train_np = y_train.values
y_valid_np = y_valid.values



## === cell 28
lr = LinearRegression().fit(X_train_np, y_train_np)
lr_rmse = rmse(y_valid_np, lr.predict(X_valid_np))
print("LinearRegression RMSE:", lr_rmse)



## === cell 29
alphas = [0.05, 0.03, 0.01, 0.5, 0.3, 0.1, 1, 3, 5]
rr = RidgeCV(alphas=alphas, cv=4).fit(X_train_np, y_train_np)
rr_rmse = rmse(y_valid_np, rr.predict(X_valid_np))
print("Ridge alpha:", rr.alpha_, "RMSE:", rr_rmse)



## === cell 30
la = LassoCV(alphas=alphas, max_iter=int(5e4), cv=4, n_jobs=-1).fit(
    X_train_np, y_train_np
)
la_rmse = rmse(y_valid_np, la.predict(X_valid_np))
print("Lasso alpha:", la.alpha_, "RMSE:", la_rmse)



## === cell 31
l1_ratios = np.linspace(0.1, 0.5, 5)
en = ElasticNetCV(
    alphas=alphas, l1_ratio=l1_ratios, max_iter=int(1e4), cv=4, n_jobs=-1
).fit(X_train_np, y_train_np)
en_rmse = rmse(y_valid_np, en.predict(X_valid_np))
print("ElasticNet alpha:", en.alpha_, "l1_ratio:", en.l1_ratio_, "RMSE:", en_rmse)



## === cell 32
rf = RandomForestRegressor(
    n_estimators=800,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
    max_depth=None,
    min_samples_leaf=1,
).fit(X_train_np, y_train_np)
rf_rmse = rmse(y_valid_np, rf.predict(X_valid_np))
print("RandomForest RMSE:", rf_rmse)



## === cell 33
labels = ["Linear", "Ridge", "Lasso", "Elastic-Net", "RandomForest"]
models_rmse = [lr_rmse, rr_rmse, la_rmse, en_rmse, rf_rmse]
rmse_df = pd.Series(models_rmse, index=labels).to_frame(name="RMSE")
rmse_df



## === cell 34
test_1 = test.drop("key", axis=1)
test_1 = test_1[X_1.columns]

pred_lr = lr.predict(test_1.values)
pred_rr = rr.predict(test_1.values)
pred_la = la.predict(test_1.values)
pred_en = en.predict(test_1.values)
pred_rf = rf.predict(test_1.values)

final_prediction = (pred_lr + pred_rr + pred_la + pred_en + pred_rf) / 5

submission = pd.DataFrame({"key": test["key"], "fare_amount": final_prediction})
submission.to_csv("NYCtaxiFare_prediction.csv", index=False)
submission.head()
