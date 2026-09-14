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
lightgbm==4.6.0
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
sklearn-pandas==2.2.0

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
from matplotlib import pyplot as plt
import os

print(os.listdir("../input"))



## === cell 1
TRAIN_ROWS = 10**6
df = pd.read_csv("../input/train.csv")
df = df.sample(n=TRAIN_ROWS, random_state=42).reset_index(drop=True)

test_set = pd.read_csv("../input/test.csv")




## === cell 2
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295  # Pi/180
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 12742 * np.arcsin(np.sqrt(a))  # 2*R*asin...


df["distance_km"] = distance(
    df.pickup_latitude, df.pickup_longitude, df.dropoff_latitude, df.dropoff_longitude
)

test_set["distance_km"] = distance(
    test_set.pickup_latitude,
    test_set.pickup_longitude,
    test_set.dropoff_latitude,
    test_set.dropoff_longitude,
)

KM_PER_DEG_LAT = 111.32
df["manhattan_km"] = (
    df["pickup_latitude"] - df["dropoff_latitude"]
).abs() * KM_PER_DEG_LAT + (df["pickup_longitude"] - df["dropoff_longitude"]).abs() * (
    KM_PER_DEG_LAT * np.cos(np.deg2rad(df["pickup_latitude"].clip(-90, 90)))
)
test_set["manhattan_km"] = (
    test_set["pickup_latitude"] - test_set["dropoff_latitude"]
).abs() * KM_PER_DEG_LAT + (
    test_set["pickup_longitude"] - test_set["dropoff_longitude"]
).abs() * (
    KM_PER_DEG_LAT * np.cos(np.deg2rad(test_set["pickup_latitude"].clip(-90, 90)))
)

NYC_LAT, NYC_LON = 40.7128, -74.0060
df["pickup_center_km"] = distance(
    df["pickup_latitude"], df["pickup_longitude"], NYC_LAT, NYC_LON
)
df["dropoff_center_km"] = distance(
    df["dropoff_latitude"], df["dropoff_longitude"], NYC_LAT, NYC_LON
)
test_set["pickup_center_km"] = distance(
    test_set["pickup_latitude"], test_set["pickup_longitude"], NYC_LAT, NYC_LON
)
test_set["dropoff_center_km"] = distance(
    test_set["dropoff_latitude"], test_set["dropoff_longitude"], NYC_LAT, NYC_LON
)



## === cell 3
BB = (-75, -73, 40, 41.5)


def select_within_boundingbox(df, BB):
    return (
        (df.pickup_longitude >= BB[0])
        & (df.pickup_longitude <= BB[1])
        & (df.pickup_latitude >= BB[2])
        & (df.pickup_latitude <= BB[3])
        & (df.dropoff_longitude >= BB[0])
        & (df.dropoff_longitude <= BB[1])
        & (df.dropoff_latitude >= BB[2])
        & (df.dropoff_latitude <= BB[3])
    )


print("Old size: %d" % len(df))
df = df[select_within_boundingbox(df, BB)]

df = df[(df.passenger_count >= 1) & (df.passenger_count <= 6)]
print("New size: %d" % len(df))

df["passenger_count"] = df["passenger_count"].clip(lower=1, upper=6)
test_set["passenger_count"] = test_set["passenger_count"].clip(lower=1, upper=6)




## === cell 4
def add_datetime_info(dataset):
    dataset["pickup_datetime"] = pd.to_datetime(dataset["pickup_datetime"])
    dataset["hour"] = dataset.pickup_datetime.dt.hour
    dataset["day"] = dataset.pickup_datetime.dt.day
    dataset["month"] = dataset.pickup_datetime.dt.month
    dataset["weekday"] = dataset.pickup_datetime.dt.weekday
    return dataset


df = add_datetime_info(df)
test_set = add_datetime_info(test_set)




## === cell 5
def _in_airport_box(lon, lat, airport="jfk"):
    if airport == "jfk":
        return (lon >= -73.90) & (lon <= -73.75) & (lat >= 40.62) & (lat <= 40.78)
    if airport == "lga":
        return (lon >= -73.90) & (lon <= -73.84) & (lat >= 40.75) & (lat <= 40.78)
    if airport == "ewr":
        return (lon >= -74.22) & (lon <= -74.15) & (lat >= 40.67) & (lat <= 40.71)
    raise ValueError("Unknown airport")


def add_flags(dataset):
    dataset["is_night"] = np.where(
        (
            ((dataset["hour"] >= 20) & (dataset["hour"] <= 23))
            | ((dataset["hour"] >= 0) & (dataset["hour"] < 6))
        ),
        1,
        0,
    )

    drop_jfk = _in_airport_box(
        dataset["dropoff_longitude"], dataset["dropoff_latitude"], "jfk"
    )
    drop_lga = _in_airport_box(
        dataset["dropoff_longitude"], dataset["dropoff_latitude"], "lga"
    )
    drop_ewr = _in_airport_box(
        dataset["dropoff_longitude"], dataset["dropoff_latitude"], "ewr"
    )
    pick_jfk = _in_airport_box(
        dataset["pickup_longitude"], dataset["pickup_latitude"], "jfk"
    )
    pick_lga = _in_airport_box(
        dataset["pickup_longitude"], dataset["pickup_latitude"], "lga"
    )
    pick_ewr = _in_airport_box(
        dataset["pickup_longitude"], dataset["pickup_latitude"], "ewr"
    )
    dataset["is_airport"] = np.where(
        (drop_jfk | drop_lga | drop_ewr | pick_jfk | pick_lga | pick_ewr), 1, 0
    )

    dataset["is_surge"] = np.where(
        (
            (dataset["hour"] >= 16)
            & (dataset["hour"] < 20)
            & (dataset["weekday"] != 5)
            & (dataset["weekday"] != 6)
        ),
        1,
        0,
    )
    return dataset


df = add_flags(df)
test_set = add_flags(test_set)



## === cell 6
df = df.drop(df[df["fare_amount"] < 0].index, axis=0)
df = df[df["fare_amount"] > 0]
df = df[df["fare_amount"] <= 250]

coord_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]
for c in coord_cols:
    df = df[df[c] != 0]

df = df[
    (df["pickup_latitude"].between(40.0, 41.5))
    & (df["dropoff_latitude"].between(40.0, 41.5))
]
df = df[
    (df["pickup_longitude"].between(-75.0, -73.0))
    & (df["dropoff_longitude"].between(-75.0, -73.0))
]

df = df[(df["distance_km"] > 0) & (df["distance_km"] < 100)]

df = df[df["fare_amount"] >= 2.5]  # base fare in NYC; removes corrupt very small labels

df = df[df["fare_amount"] <= 3.0 + 15.0 * df["distance_km"] + 10.0]

fare_per_km = df["fare_amount"] / df["distance_km"].clip(lower=0.2)
df = df[(fare_per_km >= 0.5) & (fare_per_km <= 80.0)]

same_coord = (df["pickup_latitude"] == df["dropoff_latitude"]) & (
    df["pickup_longitude"] == df["dropoff_longitude"]
)
df = df[~same_coord]

drop_dt = pd.to_datetime(
    df["pickup_datetime"], errors="coerce", utc=True
).dt.tz_convert(None)
key_dt = pd.to_datetime(
    df["key"].astype(str).str.slice(0, 19), errors="coerce", utc=True
).dt.tz_convert(None)

trip_seconds = (key_dt - drop_dt).dt.total_seconds()
mask_time = trip_seconds.notna() & (trip_seconds > 60) & (trip_seconds < 4 * 3600)
speed_kmh = df["distance_km"] / (trip_seconds.clip(lower=60) / 3600.0)
mask_speed = mask_time & (speed_kmh >= 1.0) & (speed_kmh <= 150.0)
df = df[mask_speed].copy()

df["pickup_datetime_epoch"] = (df["pickup_datetime"].astype("int64") // 10**9).astype(
    "int64"
)

df = df.drop(columns=["pickup_datetime_epoch"])


## === cell 7
plt.scatter(df["is_night"], df["fare_amount"], c="r")
plt.show()



## === cell 8
drop_cols = ["key", "fare_amount", "pickup_datetime"]
feature_cols = [c for c in df.columns if c not in drop_cols]

df = df.sort_values("pickup_datetime").reset_index(drop=True)

X_all = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
y_all = df["fare_amount"]

split_idx = int(len(df) * 0.9)
X_train, X_test = X_all.iloc[:split_idx], X_all.iloc[split_idx:]
y_train, y_test = y_all.iloc[:split_idx], y_all.iloc[split_idx:]



## === cell 9
import lightgbm as lgbm

params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "nthread": -1,
    "verbose": 0,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "max_depth": -1,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.6,
    "reg_alpha": 1,
    "reg_lambda": 0.001,
    "metric": "rmse",
    "min_split_gain": 0.5,
    "min_child_weight": 1,
    "min_child_samples": 10,
    "scale_pos_weight": 1,
}

train_set = lgbm.Dataset(X_train, y_train)
valid_set = lgbm.Dataset(X_test, y_test, reference=train_set)
model = lgbm.train(
    params,
    train_set=train_set,
    num_boost_round=300,
    valid_sets=[valid_set],
    valid_names=["valid"],
)



## --- ERROR in cell 9, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3088657045.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     23[0m [0mtrain_set[0m [0;34m=[0m [0mlgbm[0m[0;34m.[0m[0mDataset[0m[0;34m([0m[0mX_train[0m[0;34m,[0m [0my_train[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     24[0m [0mvalid_set[0m [0;34m=[0m [0mlgbm[0m[0;34m.[0m[0mDataset[0m[0;34m([0m[0mX_test[0m[0;34m,[0m [0my_test[0m[0;34m,[0m [0mreference[0m[0;34m=[0m[0mtrain_set[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 25[0;31m model = lgbm.train(
[0m[1;32m     26[0m     [0mparams[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     27[0m     [0mtrain_set[0m[0;34m=[0m[0mtrain_set[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/lightgbm/engine.py[0m in [0;36mtrain[0;34m(params, train_set, num_boost_round, valid_sets, valid_names, feval, init_model, keep_training_booster, callbacks)[0m
[1;32m    295[0m     [0;31m# construct booster[0m[0;34m[0m[0;34m[0m[0m
[1;32m    296[0m     [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 297[0;31m         [0mbooster[0m [0;34m=[0m [0mBooster[0m[0;34m([0m[0mparams[0m[0;34m=[0m[0mparams[0m[0;34m,[0m [0mtrain_set[0m[0;34m=[0m[0mtrain_set[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    298[0m         [0;32mif[0m [0mis_valid_contain_train[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    299[0m             [0mbooster[0m[0;34m.[0m[0mset_train_data_name[0m[0;34m([0m[0mtrain_data_name[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py[0m in [0;36m__init__[0;34m(self, params, train_set, model_file, model_str)[0m
[1;32m   3654[0m                 )
[1;32m   3655[0m             [0;31m# construct booster object[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 3656[0;31m             [0mtrain_set[0m[0;34m.[0m[0mconstruct[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   3657[0m             [0;31m# copy the parameters from train_set[0m[0;34m[0m[0;34m[0m[0m
[1;32m   3658[0m             [0mparams[0m[0;34m.[0m[0mupdate[0m[0;34m([0m[0mtrain_set[0m[0;34m.[0m[0mget_params[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py[0m in [0;36mconstruct[0;34m(self)[0m
[1;32m   2588[0m             [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   2589[0m                 [0;31m# create train[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 2590[0;31m                 self._lazy_init(
[0m[1;32m   2591[0m                     [0mdata[0m[0;34m=[0m[0mself[0m[0;34m.[0m[0mdata[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m   2592[0m                     [0mlabel[0m[0;34m=[0m[0mself[0m[0;34m.[0m[0mlabel[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py[0m in [0;36m_lazy_init[0;34m(self, data, label, reference, weight, group, init_score, predictor, feature_name, categorical_feature, params, position)[0m
[1;32m   2121[0m             [0mcategorical_feature[0m [0;34m=[0m [0mreference[0m[0;34m.[0m[0mcategorical_feature[0m[0;34m[0m[0;34m[0m[0m
[1;32m   2122[0m         [0;32mif[0m [0misinstance[0m[0;34m([0m[0mdata[0m[0;34m,[0m [0mpd_DataFrame[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 2123[0;31m             data, feature_name, categorical_feature, self.pandas_categorical = _data_from_pandas(
[0m[1;32m   2124[0m                 [0mdata[0m[0;34m=[0m[0mdata[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m   2125[0m                 [0mfeature_name[0m[0;34m=[0m[0mfeature_name[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py[0m in [0;36m_data_from_pandas[0;34m(data, feature_name, categorical_feature, pandas_categorical)[0m
[1;32m    832[0m ) -> Tuple[np.ndarray, List[str], Union[List[str], List[int]], List[List]]:
[1;32m    833[0m     [0;32mif[0m [0mlen[0m[0;34m([0m[0mdata[0m[0;34m.[0m[0mshape[0m[0;34m)[0m [0;34m!=[0m [0;36m2[0m [0;32mor[0m [0mdata[0m[0;34m.[0m[0mshape[0m[0;34m[[0m[0;36m0[0m[0;34m][0m [0;34m<[0m [0;36m1[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 834[0;31m         [0;32mraise[0m [0mValueError[0m[0;34m([0m[0;34m"Input data must be 2 dimensional and non empty."[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    835[0m [0;34m[0m[0m
[1;32m    836[0m     [0;31m# take shallow copy in case we modify categorical columns[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: Input data must be 2 dimensional and non empty.

## === cell 10
predictions = model.predict(X_test, num_iteration=model.best_iteration)
