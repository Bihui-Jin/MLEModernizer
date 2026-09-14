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
import time

print(os.listdir("../input"))



## === cell 1
TRAIN_PATH = "../input/train.csv"
N_SAMPLED = 1_000_000
RANDOM_STATE = 42


def read_reservoir_sample_csv(path, n_sample, seed=42, chunksize=250_000):
    try:
        import pyarrow as pa
        import pyarrow.csv as pv
    except Exception:
        pa = None
        pv = None

    rng = np.random.default_rng(seed)

    usecols = [
        "key",
        "fare_amount",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]

    dtypes = {
        "key": "string",
        "fare_amount": "float32",
        "pickup_datetime": "string",
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "int16",
    }

    if pv is None:
        reservoir_df = None
        reservoir_arr = None
        filled = 0
        seen = 0

        for chunk in pd.read_csv(
            path,
            usecols=usecols,
            dtype={k: v for k, v in dtypes.items() if k != "pickup_datetime"},
            parse_dates=["pickup_datetime"],
            infer_datetime_format=True,
            chunksize=chunksize,
        ):
            m = len(chunk)
            if m == 0:
                continue

            if reservoir_df is None:
                reservoir_df = (
                    chunk.iloc[: min(n_sample, m)].copy().reindex(range(n_sample))
                )
                reservoir_arr = reservoir_df.to_numpy(copy=False)

            if filled < n_sample:
                take = min(n_sample - filled, m)
                if take > 0:
                    reservoir_arr[filled : filled + take, :] = chunk.iloc[
                        :take
                    ].to_numpy()
                    filled += take
                    seen += take
                if take == m:
                    continue
                chunk = chunk.iloc[take:]
                m = len(chunk)

            idx_stream = np.arange(seen + 1, seen + m + 1, dtype=np.int64)
            j = rng.integers(0, idx_stream, endpoint=False)
            mask = j < n_sample
            if np.any(mask):
                replace_pos = j[mask].astype(np.int64, copy=False)
                replace_rows = chunk.iloc[np.nonzero(mask)[0]].to_numpy()
                reservoir_arr[replace_pos, :] = replace_rows

            seen += m

        if reservoir_df is None:
            return pd.DataFrame(columns=usecols)
        if filled < n_sample:
            reservoir_df = reservoir_df.iloc[:filled].copy()
        else:
            reservoir_df = reservoir_df.copy()
        return reservoir_df.reset_index(drop=True)

    convert_opts = pv.ConvertOptions(
        include_columns=usecols,
        column_types={
            "key": pa.string(),
            "fare_amount": pa.float32(),
            "pickup_datetime": pa.string(),
            "pickup_longitude": pa.float32(),
            "pickup_latitude": pa.float32(),
            "dropoff_longitude": pa.float32(),
            "dropoff_latitude": pa.float32(),
            "passenger_count": pa.int16(),
        },
        strings_can_be_null=True,
    )
    read_opts = pv.ReadOptions(block_size=1 << 24, use_threads=True)  # ~16MB blocks
    parse_opts = pv.ParseOptions(
        delimiter=",", quote_char='"', double_quote=True, escape_char=False
    )

    reader = pv.open_csv(
        path,
        read_options=read_opts,
        parse_options=parse_opts,
        convert_options=convert_opts,
    )

    reservoir = None
    filled = 0
    seen = 0

    for batch in reader:
        df = batch.to_pandas(split_blocks=True, self_destruct=True)
        if df is None or df.empty:
            continue

        m = len(df)
        if reservoir is None:
            reservoir = np.empty((n_sample, len(usecols)), dtype=object)

        if filled < n_sample:
            take = min(n_sample - filled, m)
            if take > 0:
                reservoir[filled : filled + take, :] = df.iloc[:take][usecols].to_numpy(
                    dtype=object, copy=False
                )
                filled += take
                seen += take
            if take == m:
                continue
            df = df.iloc[take:]
            m = len(df)

        idx_stream = np.arange(seen + 1, seen + m + 1, dtype=np.int64)
        j = rng.integers(0, idx_stream, endpoint=False)
        mask = j < n_sample
        if np.any(mask):
            replace_pos = j[mask].astype(np.int64, copy=False)
            replace_rows = df.iloc[np.nonzero(mask)[0]][usecols].to_numpy(
                dtype=object, copy=False
            )
            reservoir[replace_pos, :] = replace_rows

        seen += m

    if reservoir is None:
        return pd.DataFrame(columns=usecols)

    out = reservoir[:filled] if filled < n_sample else reservoir
    reservoir_df = pd.DataFrame(out, columns=usecols)

    reservoir_df["key"] = reservoir_df["key"].astype("string")
    reservoir_df["fare_amount"] = reservoir_df["fare_amount"].astype("float32")
    reservoir_df["pickup_longitude"] = reservoir_df["pickup_longitude"].astype(
        "float32"
    )
    reservoir_df["pickup_latitude"] = reservoir_df["pickup_latitude"].astype("float32")
    reservoir_df["dropoff_longitude"] = reservoir_df["dropoff_longitude"].astype(
        "float32"
    )
    reservoir_df["dropoff_latitude"] = reservoir_df["dropoff_latitude"].astype(
        "float32"
    )
    reservoir_df["passenger_count"] = reservoir_df["passenger_count"].astype("int16")

    return reservoir_df.reset_index(drop=True)


t0 = time.time()
train_df = read_reservoir_sample_csv(
    TRAIN_PATH, N_SAMPLED, seed=RANDOM_STATE, chunksize=250_000
)
print("Sampled rows:", len(train_df), "time_s:", round(time.time() - t0, 2))
train_df.dtypes



## === cell 2
print(train_df.isnull().sum())



## === cell 3
train_df = train_df.dropna(how="any", axis="rows")



## === cell 4
train_df.head()



## === cell 5
pass




## === cell 6
def clean_df(df):
    return df[
        (df.fare_amount > 0)
        & (df.fare_amount <= 250)
        & (df.pickup_longitude > -75.5)
        & (df.pickup_longitude < -72.5)
        & (df.pickup_latitude > 40.0)
        & (df.pickup_latitude < 41.8)
        & (df.dropoff_longitude > -75.5)
        & (df.dropoff_longitude < -72.5)
        & (df.dropoff_latitude > 40.0)
        & (df.dropoff_latitude < 41.8)
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
    if not np.issubdtype(dataset["pickup_datetime"].dtype, np.datetime64):
        dataset["pickup_datetime"] = pd.to_datetime(
            dataset["pickup_datetime"], errors="coerce"
        )

    dataset["hour"] = dataset.pickup_datetime.dt.hour
    dataset["day"] = dataset.pickup_datetime.dt.day
    dataset["month"] = dataset.pickup_datetime.dt.month
    dataset["weekday"] = dataset.pickup_datetime.dt.weekday

    return dataset


def bearing(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    lat1 = np.radians(pickup_lat)
    lat2 = np.radians(dropoff_lat)
    dlon = np.radians(dropoff_lon - pickup_lon)
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.arctan2(y, x)


def add_geo_features(df):
    df["distance"] = sphere_dist(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )

    df["abs_lon_diff"] = (df["pickup_longitude"] - df["dropoff_longitude"]).abs()
    df["abs_lat_diff"] = (df["pickup_latitude"] - df["dropoff_latitude"]).abs()
    df["distance_sq"] = df["distance"] ** 2

    df["distance_manhattan"] = df["abs_lon_diff"] + df["abs_lat_diff"]
    df["distance_manhattan_sq"] = df["distance_manhattan"] ** 2

    df["bearing"] = bearing(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )

    JFK_LAT, JFK_LON = 40.6413, -73.7781
    LGA_LAT, LGA_LON = 40.7769, -73.8740

    df["pickup_dist_jfk"] = sphere_dist(
        df["pickup_latitude"], df["pickup_longitude"], JFK_LAT, JFK_LON
    )
    df["dropoff_dist_jfk"] = sphere_dist(
        df["dropoff_latitude"], df["dropoff_longitude"], JFK_LAT, JFK_LON
    )
    df["pickup_dist_lga"] = sphere_dist(
        df["pickup_latitude"], df["pickup_longitude"], LGA_LAT, LGA_LON
    )
    df["dropoff_dist_lga"] = sphere_dist(
        df["dropoff_latitude"], df["dropoff_longitude"], LGA_LAT, LGA_LON
    )
    df["airport_min_dist"] = np.minimum.reduce(
        [
            df["pickup_dist_jfk"],
            df["dropoff_dist_jfk"],
            df["pickup_dist_lga"],
            df["dropoff_dist_lga"],
        ]
    )

    return df


train_df = add_geo_features(train_df)
train_df = add_datetime_info(train_df)

train_df = train_df[(train_df["distance"] >= 0) & (train_df["distance"] <= 100)]
train_df = train_df.dropna(subset=["pickup_datetime"])

train_df.head()



## === cell 8
train_df.drop(columns=["key"], inplace=True)
train_df.head()



## === cell 9
y = train_df["fare_amount"]
X = train_df.drop(columns=["fare_amount"])

X = X.sort_values("pickup_datetime")
y = y.loc[X.index]

split_idx = int(len(X) * 0.8)
x_train = X.iloc[:split_idx].drop(columns=["pickup_datetime"])
y_train = y.iloc[:split_idx]
x_test = X.iloc[split_idx:].drop(columns=["pickup_datetime"])
y_test = y.iloc[split_idx:]




## === cell 10
def XGBmodel(x_train, x_test, y_train, y_test):
    y_train_log = np.log1p(y_train.values)
    y_test_log = np.log1p(y_test.values)

    Xtr = np.ascontiguousarray(x_train.to_numpy())
    Xva = np.ascontiguousarray(x_test.to_numpy())

    matrix_train = xgb.DMatrix(
        Xtr, label=y_train_log, feature_names=list(x_train.columns)
    )
    matrix_test = xgb.DMatrix(Xva, label=y_test_log, feature_names=list(x_test.columns))

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.05,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1.0,
        "lambda": 1.0,
        "alpha": 0.0,
        "seed": RANDOM_STATE,
        "nthread": max(1, os.cpu_count() or 1),
    }

    model = xgb.train(
        params=params,
        dtrain=matrix_train,
        num_boost_round=600,
        early_stopping_rounds=30,
        evals=[(matrix_test, "valid")],
        verbose_eval=False,
    )
    return model


model = XGBmodel(x_train, x_test, y_train, y_test)



## === cell 11
test_df = pd.read_csv(
    "../input/test.csv",
    parse_dates=["pickup_datetime"],
    infer_datetime_format=True,
    dtype={
        "key": "string",
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "int16",
    },
)

test_df = add_geo_features(test_df)
test_df = add_datetime_info(test_df)

test_key = test_df["key"]
x_pred = test_df.drop(columns=["key", "pickup_datetime"])

dtest = xgb.DMatrix(
    np.ascontiguousarray(x_pred.to_numpy()), feature_names=list(x_pred.columns)
)

if hasattr(model, "best_iteration") and model.best_iteration is not None:
    pred_log = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))
else:
    pred_log = model.predict(dtest)

prediction = np.expm1(pred_log)
prediction = np.clip(prediction, 0.0, None)

submission = pd.DataFrame({"key": test_key, "fare_amount": prediction})
submission.to_csv("taxi_fare_submission.csv", index=False)
submission

## --- ERROR in cell 11, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mTypeError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1654554419.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     14[0m [0;34m[0m[0m
[1;32m     15[0m [0mtest_df[0m [0;34m=[0m [0madd_geo_features[0m[0;34m([0m[0mtest_df[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 16[0;31m [0mtest_df[0m [0;34m=[0m [0madd_datetime_info[0m[0;34m([0m[0mtest_df[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     17[0m [0;34m[0m[0m
[1;32m     18[0m [0mtest_key[0m [0;34m=[0m [0mtest_df[0m[0;34m[[0m[0;34m"key"[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/2416943992.py[0m in [0;36madd_datetime_info[0;34m(dataset)[0m
[1;32m     18[0m     [0;31m# --- Speed fix: avoid redundant parsing when already datetime64.[0m[0;34m[0m[0;34m[0m[0m
[1;32m     19[0m     [0;31m# Preserves semantics because pd.to_datetime(datetime64) is identity.[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 20[0;31m     [0;32mif[0m [0;32mnot[0m [0mnp[0m[0;34m.[0m[0missubdtype[0m[0;34m([0m[0mdataset[0m[0;34m[[0m[0;34m"pickup_datetime"[0m[0;34m][0m[0;34m.[0m[0mdtype[0m[0;34m,[0m [0mnp[0m[0;34m.[0m[0mdatetime64[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     21[0m         dataset["pickup_datetime"] = pd.to_datetime(
[1;32m     22[0m             [0mdataset[0m[0;34m[[0m[0;34m"pickup_datetime"[0m[0;34m][0m[0;34m,[0m [0merrors[0m[0;34m=[0m[0;34m"coerce"[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/numpy/core/numerictypes.py[0m in [0;36missubdtype[0;34m(arg1, arg2)[0m
[1;32m    415[0m     """
[1;32m    416[0m     [0;32mif[0m [0;32mnot[0m [0missubclass_[0m[0;34m([0m[0marg1[0m[0;34m,[0m [0mgeneric[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 417[0;31m         [0marg1[0m [0;34m=[0m [0mdtype[0m[0;34m([0m[0marg1[0m[0;34m)[0m[0;34m.[0m[0mtype[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    418[0m     [0;32mif[0m [0;32mnot[0m [0missubclass_[0m[0;34m([0m[0marg2[0m[0;34m,[0m [0mgeneric[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    419[0m         [0marg2[0m [0;34m=[0m [0mdtype[0m[0;34m([0m[0marg2[0m[0;34m)[0m[0;34m.[0m[0mtype[0m[0;34m[0m[0;34m[0m[0m

[0;31mTypeError[0m: Cannot interpret 'datetime64[ns, UTC]' as a data type
