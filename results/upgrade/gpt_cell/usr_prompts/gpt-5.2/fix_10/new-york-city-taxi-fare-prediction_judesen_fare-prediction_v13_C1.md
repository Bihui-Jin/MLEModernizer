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
import numpy as np
import pandas as pd
import datetime as dt

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import xgboost as xgb

train = pd.read_csv("../input/train.csv", nrows=10_000_000)
test = pd.read_csv("../input/test.csv")

train.dtypes



## === cell 1
print("Sum of NaN values for each column")
print(train.isnull().sum())

train = train.dropna()
print("Sum of NaN values for each column after dropping NaN")
print(train.isnull().sum())



## === cell 2
train.describe()



## === cell 3
num_cols = [
    "fare_amount",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
for c in num_cols:
    if c in train.columns:
        train[c] = pd.to_numeric(train[c], errors="coerce")
for c in num_cols:
    if c in test.columns:
        test[c] = pd.to_numeric(test[c], errors="coerce")

train = train.dropna(subset=[c for c in num_cols if c in train.columns])

train = train.loc[(train["fare_amount"] > 0) & (train["fare_amount"] < 200)]
train = train.loc[(train["pickup_longitude"] > -150) & (train["pickup_longitude"] < 0)]
train = train.loc[(train["pickup_latitude"] > 0) & (train["pickup_latitude"] < 80)]
train = train.loc[
    (train["dropoff_longitude"] > -150) & (train["dropoff_longitude"] < 0)
]
train = train.loc[
    (train["dropoff_latitude"] > 0) & (train["dropoff_latitude"] < 80)
]  # was mistakenly using dropoff_longitude

train = train.loc[(train["passenger_count"] >= 1) & (train["passenger_count"] <= 8)]

nyc_lon_min, nyc_lon_max = -74.5, -72.8
nyc_lat_min, nyc_lat_max = 40.0, 41.8
train = train.loc[
    train["pickup_longitude"].between(nyc_lon_min, nyc_lon_max)
    & train["dropoff_longitude"].between(nyc_lon_min, nyc_lon_max)
    & train["pickup_latitude"].between(nyc_lat_min, nyc_lat_max)
    & train["dropoff_latitude"].between(nyc_lat_min, nyc_lat_max)
]

train.describe()



## === cell 4
test_valid_mask = (
    test["pickup_longitude"].between(nyc_lon_min, nyc_lon_max)
    & test["dropoff_longitude"].between(nyc_lon_min, nyc_lon_max)
    & test["pickup_latitude"].between(nyc_lat_min, nyc_lat_max)
    & test["dropoff_latitude"].between(nyc_lat_min, nyc_lat_max)
    & test["passenger_count"].between(1, 8)
)
test_valid = test.loc[test_valid_mask].copy()

combine = [train, test_valid]
for dataset in combine:
    dataset["longitude_distance"] = (
        dataset["pickup_longitude"] - dataset["dropoff_longitude"]
    ).abs()
    dataset["latitude_distance"] = (
        dataset["pickup_latitude"] - dataset["dropoff_latitude"]
    ).abs()

    dataset["distance_travelled"] = np.sqrt(
        dataset["longitude_distance"] ** 2 + dataset["latitude_distance"] ** 2
    )

    R = 6371e3  # Metres
    phi1 = np.radians(dataset["pickup_latitude"])
    phi2 = np.radians(dataset["dropoff_latitude"])
    dphi = np.radians(dataset["dropoff_latitude"] - dataset["pickup_latitude"])
    dlambda = np.radians(dataset["dropoff_longitude"] - dataset["pickup_longitude"])

    a = (np.sin(dphi / 2) ** 2) + np.cos(phi1) * np.cos(phi2) * (
        np.sin(dlambda / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    dataset["haversine"] = R * c

    y = np.sin(dlambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlambda)
    dataset["bearing"] = np.arctan2(y, x)

    dataset["pickup_datetime"] = pd.to_datetime(dataset["pickup_datetime"])
    dataset["hour_of_day"] = dataset.pickup_datetime.dt.hour
    dataset["day"] = dataset.pickup_datetime.dt.day
    dataset["week"] = dataset.pickup_datetime.dt.isocalendar().week.astype(int)
    dataset["month"] = dataset.pickup_datetime.dt.month
    dataset["day_of_year"] = dataset.pickup_datetime.dt.dayofyear
    dataset["week_of_year"] = dataset.pickup_datetime.dt.isocalendar().week.astype(int)

    dataset["distance_travelled_sin"] = np.sin(dataset["haversine"] / 1000.0)

    dataset["distance_travelled_cos"] = np.cos(dataset["distance_travelled"])
    dataset["distance_travelled_sin_sqrd"] = np.sin(dataset["distance_travelled"]) ** 2
    dataset["distance_travelled_cos_sqrd"] = np.cos(dataset["distance_travelled"]) ** 2

train = train.loc[~((train["haversine"] < 50.0) & (train["fare_amount"] > 10.0))]
train = train.loc[
    train["haversine"] < 100_000.0
]  # >100km trips are usually GPS glitches in this dataset

test_valid = test_valid.loc[test_valid["haversine"] < 100_000.0]

train.head(3)



## === cell 5
colormap = plt.cm.RdBu
plt.figure(figsize=(20, 20))
plt.title("Pearson Correlation of Features", y=1.05, size=15)

corr = train.corr(numeric_only=True)

sns.heatmap(
    corr,
    linewidths=0.1,
    vmax=1.0,
    square=True,
    cmap=colormap,
    linecolor="white",
    annot=True,
)



## === cell 6
train_features_to_keep = ["haversine", "distance_travelled_sin", "fare_amount"]
train.drop(train.columns.difference(train_features_to_keep), axis=1, inplace=True)

test_features_to_keep = ["haversine", "distance_travelled_sin", "key"]
test_valid.drop(
    test_valid.columns.difference(test_features_to_keep), axis=1, inplace=True
)



## === cell 7
x_pred = test_valid.drop("key", axis=1)

x_train, x_test, y_train, y_test = train_test_split(
    train.drop("fare_amount", axis=1),
    train.pop("fare_amount"),
    random_state=123,
    test_size=0.2,
)


def XGBmodel(x_train, x_test, y_train, y_test):
    matrix_train = xgb.DMatrix(x_train, label=y_train)
    matrix_test = xgb.DMatrix(x_test, label=y_test)
    model = xgb.train(
        params={"objective": "reg:squarederror", "eval_metric": "rmse", "seed": 123},
        dtrain=matrix_train,
        num_boost_round=100,
        early_stopping_rounds=10,
        evals=[(matrix_test, "test")],
    )
    return model


model = XGBmodel(x_train, x_test, y_train, y_test)

dm_pred = xgb.DMatrix(x_pred)
if hasattr(model, "best_iteration") and model.best_iteration is not None:
    prediction_valid = model.predict(
        dm_pred, iteration_range=(0, model.best_iteration + 1)
    )
else:
    prediction_valid = model.predict(dm_pred)

prediction_valid = np.clip(prediction_valid, 0, None)

fallback_fare = float(pd.Series(y_train).median())
prediction_all = pd.Series(fallback_fare, index=test.index, dtype="float64")
prediction_all.loc[
    (
        test_valid_mask.loc[test_valid.index]
        if hasattr(test_valid_mask, "loc")
        else test_valid.index
    )
] = prediction_valid
prediction_all = prediction_all.values



## --- ERROR in cell 7, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mIndexingError[0m                             Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3268328497.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     38[0m [0mfallback_fare[0m [0;34m=[0m [0mfloat[0m[0;34m([0m[0mpd[0m[0;34m.[0m[0mSeries[0m[0;34m([0m[0my_train[0m[0;34m)[0m[0;34m.[0m[0mmedian[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     39[0m [0mprediction_all[0m [0;34m=[0m [0mpd[0m[0;34m.[0m[0mSeries[0m[0;34m([0m[0mfallback_fare[0m[0;34m,[0m [0mindex[0m[0;34m=[0m[0mtest[0m[0;34m.[0m[0mindex[0m[0;34m,[0m [0mdtype[0m[0;34m=[0m[0;34m"float64"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 40[0;31m prediction_all.loc[
[0m[1;32m     41[0m     (
[1;32m     42[0m         [0mtest_valid_mask[0m[0;34m.[0m[0mloc[0m[0;34m[[0m[0mtest_valid[0m[0;34m.[0m[0mindex[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py[0m in [0;36m__setitem__[0;34m(self, key, value)[0m
[1;32m    905[0m             [0mmaybe_callable[0m [0;34m=[0m [0mcom[0m[0;34m.[0m[0mapply_if_callable[0m[0;34m([0m[0mkey[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0mobj[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    906[0m             [0mkey[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_check_deprecated_callable_usage[0m[0;34m([0m[0mkey[0m[0;34m,[0m [0mmaybe_callable[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 907[0;31m         [0mindexer[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_get_setitem_indexer[0m[0;34m([0m[0mkey[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    908[0m         [0mself[0m[0;34m.[0m[0m_has_valid_setitem_indexer[0m[0;34m([0m[0mkey[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    909[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py[0m in [0;36m_get_setitem_indexer[0;34m(self, key)[0m
[1;32m    778[0m             [0mkey[0m [0;34m=[0m [0mlist[0m[0;34m([0m[0mkey[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    779[0m [0;34m[0m[0m
[0;32m--> 780[0;31m         [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_convert_to_indexer[0m[0;34m([0m[0mkey[0m[0;34m,[0m [0maxis[0m[0;34m=[0m[0;36m0[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    781[0m [0;34m[0m[0m
[1;32m    782[0m     [0;34m@[0m[0mfinal[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py[0m in [0;36m_convert_to_indexer[0;34m(self, key, axis)[0m
[1;32m   1517[0m [0;34m[0m[0m
[1;32m   1518[0m             [0;32mif[0m [0mcom[0m[0;34m.[0m[0mis_bool_indexer[0m[0;34m([0m[0mkey[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1519[0;31m                 [0mkey[0m [0;34m=[0m [0mcheck_bool_indexer[0m[0;34m([0m[0mlabels[0m[0;34m,[0m [0mkey[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1520[0m                 [0;32mreturn[0m [0mkey[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1521[0m             [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py[0m in [0;36mcheck_bool_indexer[0;34m(index, key)[0m
[1;32m   2660[0m         [0mindexer[0m [0;34m=[0m [0mresult[0m[0;34m.[0m[0mindex[0m[0;34m.[0m[0mget_indexer_for[0m[0;34m([0m[0mindex[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   2661[0m         [0;32mif[0m [0;34m-[0m[0;36m1[0m [0;32min[0m [0mindexer[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 2662[0;31m             raise IndexingError(
[0m[1;32m   2663[0m                 [0;34m"Unalignable boolean Series provided as "[0m[0;34m[0m[0;34m[0m[0m
[1;32m   2664[0m                 [0;34m"indexer (index of the boolean Series and of "[0m[0;34m[0m[0;34m[0m[0m

[0;31mIndexingError[0m: Unalignable boolean Series provided as indexer (index of the boolean Series and of the indexed object do not match).

## === cell 8
submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": np.round(prediction_all, 2)}
)

submission.to_csv("sub_fare.csv", index=False)
