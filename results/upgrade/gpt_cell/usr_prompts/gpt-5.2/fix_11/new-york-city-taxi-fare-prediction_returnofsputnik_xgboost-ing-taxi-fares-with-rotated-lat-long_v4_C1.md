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
from sklearn.model_selection import train_test_split
import xgboost as xgb
import os

print(os.listdir("../input"))



## === cell 1
RANDOM_SEED = 42
NROWS = 1_000_000

train_path = "../input/train.csv"
n_total = sum(1 for _ in open(train_path)) - 1  # exclude header
rng = np.random.default_rng(RANDOM_SEED)

if NROWS >= n_total:
    skip = None
else:
    n_skip = n_total - NROWS
    skip_idx = rng.choice(
        np.arange(1, n_total + 1), size=n_skip, replace=False
    )  # 1..n_total
    skip = set(skip_idx.tolist())

train_df = pd.read_csv(train_path, skiprows=skip)
train_df.dtypes



## === cell 2
print(train_df.isnull().sum())



## === cell 3
train_df = train_df.dropna(how="any", axis="rows")



## === cell 4
train_df.head()



## === cell 5
train_df.iloc[:1000].plot.scatter("pickup_longitude", "pickup_latitude")
train_df.iloc[:1000].plot.scatter("dropoff_longitude", "dropoff_latitude")

train_df.describe()




## === cell 6
def clean_df(df):
    return df[
        (df.fare_amount > 0)
        & (df.pickup_longitude > -80)
        & (df.pickup_longitude < -70)
        & (df.pickup_latitude > 35)
        & (df.pickup_latitude < 45)
        & (df.dropoff_longitude > -80)
        & (df.dropoff_longitude < -70)
        & (df.dropoff_latitude > 35)
        & (df.dropoff_latitude < 45)
        & (df.passenger_count > 0)
        & (df.passenger_count < 10)
    ]


train_df = clean_df(train_df)
print(len(train_df))




## === cell 7
def sphere_dist(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    R_earth = 6371
    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = map(
        np.radians, [pickup_lat, pickup_lon, dropoff_lat, dropoff_lon]
    )
    dlat = dropoff_lat - pickup_lat
    dlon = dropoff_lon - pickup_lon

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(pickup_lat) * np.cos(dropoff_lat) * np.sin(dlon / 2.0) ** 2
    )

    return 2 * R_earth * np.arcsin(np.sqrt(a))


def add_datetime_info(dataset):
    dataset["pickup_datetime"] = pd.to_datetime(dataset["pickup_datetime"])

    dataset["hour"] = dataset.pickup_datetime.dt.hour
    dataset["day"] = dataset.pickup_datetime.dt.day
    dataset["month"] = dataset.pickup_datetime.dt.month
    dataset["weekday"] = dataset.pickup_datetime.dt.weekday

    return dataset


train_df["distance"] = sphere_dist(
    train_df["pickup_latitude"],
    train_df["pickup_longitude"],
    train_df["dropoff_latitude"],
    train_df["dropoff_longitude"],
)

train_df = add_datetime_info(train_df)

train_df.head()



## === cell 8
train_df.drop(columns=["key", "pickup_datetime"], inplace=True)
train_df.head()



## === cell 9
train_df["pickup_long_15"] = train_df["pickup_longitude"] * np.cos(
    15 * np.pi / 180
) - train_df["pickup_latitude"] * np.sin(15 * np.pi / 180)
train_df["pickup_long_30"] = train_df["pickup_longitude"] * np.cos(
    30 * np.pi / 180
) - train_df["pickup_latitude"] * np.sin(30 * np.pi / 180)
train_df["pickup_long_45"] = train_df["pickup_longitude"] * np.cos(
    45 * np.pi / 180
) - train_df["pickup_latitude"] * np.sin(45 * np.pi / 180)
train_df["pickup_long_60"] = train_df["pickup_longitude"] * np.cos(
    60 * np.pi / 180
) - train_df["pickup_latitude"] * np.sin(60 * np.pi / 180)
train_df["pickup_long_75"] = train_df["pickup_longitude"] * np.cos(
    75 * np.pi / 180
) - train_df["pickup_latitude"] * np.sin(75 * np.pi / 180)

train_df["pickup_lat_15"] = train_df["pickup_longitude"] * np.sin(
    15 * np.pi / 180
) + train_df["pickup_latitude"] * np.cos(15 * np.pi / 180)
train_df["pickup_lat_30"] = train_df["pickup_longitude"] * np.sin(
    30 * np.pi / 180
) + train_df["pickup_latitude"] * np.cos(30 * np.pi / 180)
train_df["pickup_lat_45"] = train_df["pickup_longitude"] * np.sin(
    45 * np.pi / 180
) + train_df["pickup_latitude"] * np.cos(45 * np.pi / 180)
train_df["pickup_lat_60"] = train_df["pickup_longitude"] * np.sin(
    60 * np.pi / 180
) + train_df["pickup_latitude"] * np.cos(60 * np.pi / 180)
train_df["pickup_lat_75"] = train_df["pickup_longitude"] * np.sin(
    75 * np.pi / 180
) + train_df["pickup_latitude"] * np.cos(75 * np.pi / 180)

train_df["dropoff_long_15"] = train_df["dropoff_longitude"] * np.cos(
    15 * np.pi / 180
) - train_df["dropoff_latitude"] * np.sin(15 * np.pi / 180)
train_df["dropoff_long_30"] = train_df["dropoff_longitude"] * np.cos(
    30 * np.pi / 180
) - train_df["dropoff_latitude"] * np.sin(30 * np.pi / 180)
train_df["dropoff_long_45"] = train_df["dropoff_longitude"] * np.cos(
    45 * np.pi / 180
) - train_df["dropoff_latitude"] * np.sin(45 * np.pi / 180)
train_df["dropoff_long_60"] = train_df["dropoff_longitude"] * np.cos(
    60 * np.pi / 180
) - train_df["dropoff_latitude"] * np.sin(60 * np.pi / 180)
train_df["dropoff_long_75"] = train_df["dropoff_longitude"] * np.cos(
    75 * np.pi / 180
) - train_df["dropoff_latitude"] * np.sin(75 * np.pi / 180)

train_df["dropoff_lat_15"] = train_df["dropoff_longitude"] * np.sin(
    15 * np.pi / 180
) + train_df["dropoff_latitude"] * np.cos(15 * np.pi / 180)
train_df["dropoff_lat_30"] = train_df["dropoff_longitude"] * np.sin(
    30 * np.pi / 180
) + train_df["dropoff_latitude"] * np.cos(30 * np.pi / 180)
train_df["dropoff_lat_45"] = train_df["dropoff_longitude"] * np.sin(
    45 * np.pi / 180
) + train_df["dropoff_latitude"] * np.cos(45 * np.pi / 180)
train_df["dropoff_lat_60"] = train_df["dropoff_longitude"] * np.sin(
    60 * np.pi / 180
) + train_df["dropoff_latitude"] * np.cos(60 * np.pi / 180)
train_df["dropoff_lat_75"] = train_df["dropoff_longitude"] * np.sin(
    75 * np.pi / 180
) + train_df["dropoff_latitude"] * np.cos(75 * np.pi / 180)



## === cell 10
dt_df = pd.read_csv(train_path, usecols=["pickup_datetime"], skiprows=skip)
dt_df = dt_df.dropna(how="any", axis="rows")
dt_df = clean_df(dt_df)
pickup_dt = pd.to_datetime(dt_df["pickup_datetime"], errors="coerce").reset_index(
    drop=True
)

y = train_df["fare_amount"]
train = train_df.drop(columns=["fare_amount"])

train = train.apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(y, errors="coerce")
mask = train.notnull().all(axis=1) & y.notnull() & pickup_dt.notnull()

train = train.loc[mask].reset_index(drop=True)
y = y.loc[mask].reset_index(drop=True)
pickup_dt = pickup_dt.loc[mask].reset_index(drop=True)

tmp = train.copy()
tmp["__y__"] = y.values
tmp["__pickup_datetime__"] = pickup_dt.values
tmp = tmp.sort_values("__pickup_datetime__").drop(columns=["__pickup_datetime__"])

y = tmp["__y__"]
train = tmp.drop(columns=["__y__"])

n = len(train)
split_idx = int(n * 0.8)

x_train = train.iloc[:split_idx].copy()
y_train = y.iloc[:split_idx].copy()
x_test = train.iloc[split_idx:].copy()
y_test = y.iloc[split_idx:].copy()


## --- ERROR in cell 10, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3145018230.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      4[0m [0mdt_df[0m [0;34m=[0m [0mpd[0m[0;34m.[0m[0mread_csv[0m[0;34m([0m[0mtrain_path[0m[0;34m,[0m [0musecols[0m[0;34m=[0m[0;34m[[0m[0;34m"pickup_datetime"[0m[0;34m][0m[0;34m,[0m [0mskiprows[0m[0;34m=[0m[0mskip[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0mdt_df[0m [0;34m=[0m [0mdt_df[0m[0;34m.[0m[0mdropna[0m[0;34m([0m[0mhow[0m[0;34m=[0m[0;34m"any"[0m[0;34m,[0m [0maxis[0m[0;34m=[0m[0;34m"rows"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 6[0;31m [0mdt_df[0m [0;34m=[0m [0mclean_df[0m[0;34m([0m[0mdt_df[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      7[0m pickup_dt = pd.to_datetime(dt_df["pickup_datetime"], errors="coerce").reset_index(
[1;32m      8[0m     [0mdrop[0m[0;34m=[0m[0;32mTrue[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/1453588314.py[0m in [0;36mclean_df[0;34m(df)[0m
[1;32m      1[0m [0;32mdef[0m [0mclean_df[0m[0;34m([0m[0mdf[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m      2[0m     return df[
[0;32m----> 3[0;31m         [0;34m([0m[0mdf[0m[0;34m.[0m[0mfare_amount[0m [0;34m>[0m [0;36m0[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      4[0m         [0;34m&[0m [0;34m([0m[0mdf[0m[0;34m.[0m[0mpickup_longitude[0m [0;34m>[0m [0;34m-[0m[0;36m80[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m         [0;34m&[0m [0;34m([0m[0mdf[0m[0;34m.[0m[0mpickup_longitude[0m [0;34m<[0m [0;34m-[0m[0;36m70[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py[0m in [0;36m__getattr__[0;34m(self, name)[0m
[1;32m   6297[0m         ):
[1;32m   6298[0m             [0;32mreturn[0m [0mself[0m[0;34m[[0m[0mname[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 6299[0;31m         [0;32mreturn[0m [0mobject[0m[0;34m.[0m[0m__getattribute__[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mname[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   6300[0m [0;34m[0m[0m
[1;32m   6301[0m     [0;34m@[0m[0mfinal[0m[0;34m[0m[0;34m[0m[0m

[0;31mAttributeError[0m: 'DataFrame' object has no attribute 'fare_amount'

## === cell 11
def XGBmodel(x_train, x_test, y_train, y_test):
    matrix_train = xgb.DMatrix(x_train, label=y_train)
    matrix_test = xgb.DMatrix(x_test, label=y_test)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.05,
        "max_depth": 8,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 1.0,
        "alpha": 0.0,
        "seed": RANDOM_SEED,
        "nthread": max(1, os.cpu_count() or 1),
    }

    model = xgb.train(
        params=params,
        dtrain=matrix_train,
        num_boost_round=5000,
        early_stopping_rounds=100,
        evals=[(matrix_test, "test")],
        verbose_eval=False,
    )
    return model, params


model, params = XGBmodel(x_train, x_test, y_train, y_test)

best_iter = getattr(model, "best_iteration", None)
if best_iter is None:
    final_num_boost_round = 5000
else:
    final_num_boost_round = int(best_iter) + 1  # best_iteration is 0-based

dall = xgb.DMatrix(train, label=y)
final_model = xgb.train(
    params=params,
    dtrain=dall,
    num_boost_round=final_num_boost_round,
    verbose_eval=False,
)
