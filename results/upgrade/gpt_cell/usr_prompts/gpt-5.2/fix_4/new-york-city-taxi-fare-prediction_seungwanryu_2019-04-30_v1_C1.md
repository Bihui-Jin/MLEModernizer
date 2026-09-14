# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.7

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import datetime as dt


from sklearn.model_selection import train_test_split
import xgboost as xgb



## === cell 1
train = pd.read_csv("../input/train.csv", nrows = 100000)
test = pd.read_csv("../input/test.csv",nrows = 100000)


## === cell 2
train.dtypes


## === cell 3
train.isnull().sum()


## === cell 4
test.isnull().sum()


## === cell 5
train.describe()


## === cell 6
train = train.loc[(train['fare_amount'] > 0) & (train['fare_amount'] < 200)]
train = train.loc[(train['pickup_longitude'] > -300) & (train['pickup_longitude'] < 300)]
train = train.loc[(train['pickup_latitude'] > -300) & (train['pickup_latitude'] < 300)]
train = train.loc[(train['dropoff_longitude'] > -300) & (train['dropoff_longitude'] < 300)]
train = train.loc[(train['dropoff_latitude'] > -300) & (train['dropoff_latitude'] < 300)]
train = train.loc[train['passenger_count'] <= 8]
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

    R = 6371e3  # Metres
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
    dataset["hour_of_day"] = dataset.pickup_datetime.dt.hour
    dataset["day"] = dataset.pickup_datetime.dt.day

    _iso_week = dataset.pickup_datetime.dt.isocalendar().week.astype(int)
    dataset["week"] = _iso_week
    dataset["month"] = dataset.pickup_datetime.dt.month
    dataset["day_of_year"] = dataset.pickup_datetime.dt.dayofyear
    dataset["week_of_year"] = _iso_week


## === cell 8
train = train.loc[train['haversine'] != 0]
train = train.dropna()


## === cell 9
train.head()


## === cell 10
train.isnull().sum()


## === cell 11
test.isnull().sum()


## === cell 12
median = test['haversine'].median()
test['haversine'] = test['haversine'].fillna(median)


## === cell 13
colormap = plt.cm.RdBu
plt.figure(figsize=(20, 20))
plt.title("Pearson Correlation of Features", y=1.05, size=15)

sns.heatmap(
    train.corr(numeric_only=True),
    linewidths=0.1,
    vmax=1.0,
    square=True,
    cmap=colormap,
    linecolor="white",
    annot=True,
)


## === cell 14
train_features_to_keep = ["haversine", "fare_amount"]
train.drop(train.columns.difference(train_features_to_keep), axis=1, inplace=True)

test_features_to_keep = ["haversine", "key"]
test.drop(test.columns.difference(test_features_to_keep), axis=1, inplace=True)


## === cell 15
x_train = train.drop('fare_amount', axis=1)
y_train = train['fare_amount']
x_test = test.drop('key', axis=1)

from sklearn.linear_model import LinearRegression
regr = LinearRegression()
regr.fit(x_train, y_train)
regr_pred = regr.predict(x_test)

from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
rfr.fit(x_train, y_train)
rfr_pred = rfr.predict(x_test)


## === cell 16
x_pred = test.drop('key', axis=1)

x_train,x_test,y_train,y_test = train_test_split(train.drop('fare_amount',axis=1),train.pop('fare_amount'),random_state=123,test_size=0.2)

def XGBmodel(x_train,x_test,y_train,y_test):
    matrix_train = xgb.DMatrix(x_train,label=y_train)
    matrix_test = xgb.DMatrix(x_test,label=y_test)
    model=xgb.train(params={'objective':'reg:linear','eval_metric':'rmse'}
                    ,dtrain=matrix_train,num_boost_round=200, 
                    early_stopping_rounds=20,evals=[(matrix_test,'test')],)
    return model

model=XGBmodel(x_train,x_test,y_train,y_test)
xgb_pred = model.predict(xgb.DMatrix(x_pred), ntree_limit = model.best_ntree_limit)


## --- ERROR in cell 16, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2660636660.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     13[0m [0;34m[0m[0m
[1;32m     14[0m [0mmodel[0m[0;34m=[0m[0mXGBmodel[0m[0;34m([0m[0mx_train[0m[0;34m,[0m[0mx_test[0m[0;34m,[0m[0my_train[0m[0;34m,[0m[0my_test[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 15[0;31m [0mxgb_pred[0m [0;34m=[0m [0mmodel[0m[0;34m.[0m[0mpredict[0m[0;34m([0m[0mxgb[0m[0;34m.[0m[0mDMatrix[0m[0;34m([0m[0mx_pred[0m[0;34m)[0m[0;34m,[0m [0mntree_limit[0m [0;34m=[0m [0mmodel[0m[0;34m.[0m[0mbest_ntree_limit[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;31mAttributeError[0m: 'Booster' object has no attribute 'best_ntree_limit'

## === cell 17
regr_pred, rfr_pred, xgb_pred
