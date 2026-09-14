# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

# 5. Target score

9.57996

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))
import matplotlib.pyplot as plt

import seaborn as sns
import datetime as dt

from sklearn.model_selection import train_test_split
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")



## === cell 1
train = pd.read_csv("../input/train.csv", nrows=2000000)
test = pd.read_csv("../input/test.csv", nrows=2000000)



## === cell 2
train.dtypes



## === cell 3
train.isnull().sum()



## === cell 4
test.isnull().sum()



## === cell 5
train.describe()



## === cell 6
train = train.loc[(train["fare_amount"] > 0) & (train["fare_amount"] < 200)]
train = train.loc[
    (train["pickup_longitude"] > -300) & (train["pickup_longitude"] < 300)
]
train = train.loc[(train["pickup_latitude"] > -300) & (train["pickup_latitude"] < 300)]
train = train.loc[
    (train["dropoff_longitude"] > -300) & (train["dropoff_longitude"] < 300)
]
train = train.loc[
    (train["dropoff_latitude"] > -300) & (train["dropoff_latitude"] < 300)
]
train = train.loc[train["passenger_count"] <= 8]
train.describe()



## === cell 7
combine = [test, train]
for dataset in combine:
    dataset["longitude_distance"] = (
        dataset["pickup_longitude"] - dataset["dropoff_longitude"]
    )
    dataset["latitude_distance"] = (
        dataset["pickup_latitude"] - dataset["dropoff_latitude"]
    )

    dataset["distance_travelled"] = (
        dataset["longitude_distance"] ** 2 + dataset["latitude_distance"] ** 2
    ) ** 0.5
    dataset["distance_travelled_sin"] = np.sin(
        (dataset["longitude_distance"] ** 2 * dataset["latitude_distance"] ** 2) ** 0.5
    )
    dataset["distance_travelled_cos"] = np.cos(
        (dataset["longitude_distance"] ** 2 * dataset["latitude_distance"] ** 2) ** 0.5
    )
    dataset["distance_travelled_sin_sqrd"] = (
        np.sin(
            (dataset["longitude_distance"] ** 2 * dataset["latitude_distance"] ** 2)
            ** 0.5
        )
        ** 2
    )
    dataset["distance_travelled_cos_sqrd"] = (
        np.cos(
            (dataset["longitude_distance"] ** 2 * dataset["latitude_distance"] ** 2)
            ** 0.5
        )
        ** 2
    )

    R = 6371e3  # metres
    phi1 = np.radians(dataset["pickup_latitude"])
    phi2 = np.radians(dataset["dropoff_latitude"])
    phi_chg = np.radians(dataset["pickup_latitude"] - dataset["dropoff_latitude"])
    delta_chg = np.radians(dataset["pickup_longitude"] - dataset["dropoff_longitude"])
    a = (
        np.sin(phi_chg / 2) ** 0.5
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_chg / 2) ** 0.5
    )
    c = 2 * np.arctan2(a**0.5, (1 - a) ** 0.5)
    d = R * c
    dataset["haversine"] = d

    y = np.sin(delta_chg * np.cos(phi2))
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_chg)
    dataset["bearing"] = np.arctan2(y, x)

    dataset["pickup_datetime"] = pd.to_datetime(dataset["pickup_datetime"])
    dataset["hour_of_day"] = dataset["pickup_datetime"].dt.hour
    dataset["day"] = dataset["pickup_datetime"].dt.day
    dataset["week"] = dataset["pickup_datetime"].dt.isocalendar().week
    dataset["month"] = dataset["pickup_datetime"].dt.month
    dataset["day_of_year"] = dataset["pickup_datetime"].dt.dayofyear
    dataset["week_of_year"] = dataset["pickup_datetime"].dt.isocalendar().week



## === cell 8
train = train.loc[train["haversine"] != 0]
train = train.dropna()



## === cell 9
train.head()



## === cell 10
train.isnull().sum()



## === cell 11
test.isnull().sum()



## === cell 12
median = test["haversine"].median()
test["haversine"] = test["haversine"].fillna(median)



## === cell 13
numeric_corr = train.select_dtypes(include=[np.number]).corr()
colormap = plt.cm.RdBu
plt.figure(figsize=(12, 10))
plt.title("Pearson Correlation of Numeric Features", y=1.05, size=15)
sns.heatmap(
    numeric_corr,
    linewidths=0.1,
    vmax=1.0,
    square=True,
    cmap=colormap,
    linecolor="white",
    annot=True,
)



## === cell 14
train_features_to_keep = [
    "haversine",
    "hour_of_day",
    "day",
    "week",
    "month",
    "day_of_year",
    "passenger_count",
    "fare_amount",
]
train.drop(columns=train.columns.difference(train_features_to_keep), inplace=True)

test_features_to_keep = [
    "haversine",
    "hour_of_day",
    "day",
    "week",
    "month",
    "day_of_year",
    "passenger_count",
    "key",
]
test.drop(columns=test.columns.difference(test_features_to_keep), inplace=True)



## === cell 15
x_train_lr = train.drop("fare_amount", axis=1)
y_train_lr = train["fare_amount"]
x_test_lr = test.drop("key", axis=1)

from sklearn.linear_model import LinearRegression

regr = LinearRegression()
regr.fit(x_train_lr, y_train_lr)
regr_pred = regr.predict(x_test_lr)

from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=5)
rfr.fit(x_train_lr, y_train_lr)
rfr_pred = rfr.predict(x_test_lr)



## === cell 16
x_tr, x_val, y_tr, y_val = train_test_split(
    x_train_lr, y_train_lr, test_size=0.2, random_state=123
)


def XGBmodel(x_train, x_val, y_train, y_val):
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dval = xgb.DMatrix(x_val, label=y_val)
    params = {"objective": "reg:squarederror", "eval_metric": "rmse", "seed": 42}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=[(dval, "validation")],
        early_stopping_rounds=20,
        verbose_eval=False,
    )
    return model


model = XGBmodel(x_tr, x_val, y_tr, y_val)
best_ntree_limit = model.best_iteration + 1
xgb_pred = model.predict(xgb.DMatrix(x_test_lr), ntree_limit=best_ntree_limit)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3695843044.py in <cell line: 0>()
     22 # Use the best iteration found during early stopping
     23 best_ntree_limit = model.best_iteration + 1
---> 24 xgb_pred = model.predict(xgb.DMatrix(x_test_lr), ntree_limit=best_ntree_limit)
     25 

TypeError: Booster.predict() got an unexpected keyword argument 'ntree_limit'

## === cell 17
regr_weight = 1
rfr_weight = 1
xgb_weight = 3
prediction = (
    regr_pred * regr_weight + rfr_pred * rfr_weight + xgb_pred * xgb_weight
) / (regr_weight + rfr_weight + xgb_weight)



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1383582363.py in <cell line: 0>()
      3 xgb_weight = 3
      4 prediction = (
----> 5     regr_pred * regr_weight + rfr_pred * rfr_weight + xgb_pred * xgb_weight
      6 ) / (regr_weight + rfr_weight + xgb_weight)
      7 

NameError: name 'xgb_pred' is not defined

## === cell 18
submission = pd.DataFrame({"key": test["key"], "fare_amount": np.round(prediction, 2)})
submission.to_csv("sub_fare.csv", index=False)



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/976820138.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"key": test["key"], "fare_amount": np.round(prediction, 2)})
      2 submission.to_csv("sub_fare.csv", index=False)
      3 

NameError: name 'prediction' is not defined

## === cell 19
submission.head()

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3365464162.py in <cell line: 0>()
----> 1 submission.head()

NameError: name 'submission' is not defined
