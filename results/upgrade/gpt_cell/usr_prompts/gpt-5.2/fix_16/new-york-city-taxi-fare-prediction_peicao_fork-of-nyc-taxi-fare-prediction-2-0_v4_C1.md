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

bayesian-optimization==3.1.0
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
import os
import multiprocessing as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

print(os.listdir("../input"))
os.chdir("/kaggle/working/")



## === cell 1
dtypes_train = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16",
}
df_train = pd.read_csv(
    "../input/train.csv",
    nrows=800000,
    parse_dates=["pickup_datetime"],
    dtype=dtypes_train,
)
df_train.head()



## === cell 2
df_train.describe()
df_train.dtypes



## === cell 3
dtypes_test = {
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16",
}
df_train = df_train[(df_train["fare_amount"] > 0.05) & (df_train.passenger_count > 0)]
df_train.dropna(how="any", axis="rows", inplace=True)
print("New Size: {}".format(len(df_train)))
df_test = pd.read_csv(
    "../input/test.csv", parse_dates=["pickup_datetime"], dtype=dtypes_test
)



## === cell 4
mask = df_train["pickup_longitude"].between(-75, -73)
mask &= df_train["dropoff_longitude"].between(-75, -73)
mask &= df_train["pickup_latitude"].between(40, 42)
mask &= df_train["dropoff_latitude"].between(40, 42)
mask &= df_train["passenger_count"].between(1, 6)  # small, standard denoise
mask &= df_train["fare_amount"].between(0, 250)

mask &= ~((df_train["pickup_longitude"] == 0) & (df_train["pickup_latitude"] == 0))
mask &= ~((df_train["dropoff_longitude"] == 0) & (df_train["dropoff_latitude"] == 0))

df_train = df_train[mask]




## === cell 5
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295  # Pi/180
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))  # miles


def haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1.astype(np.float64))
    lon1 = np.radians(lon1.astype(np.float64))
    lat2 = np.radians(lat2.astype(np.float64))
    lon2 = np.radians(lon2.astype(np.float64))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return (6371.0088 * 2.0 * np.arcsin(np.sqrt(a))).astype(np.float32)


def add_features(df):
    df = df.copy(deep=False)

    df["distance_miles"] = distance(
        df.pickup_latitude,
        df.pickup_longitude,
        df.dropoff_latitude,
        df.dropoff_longitude,
    ).astype(np.float32)

    df["distance_km"] = (df["distance_miles"].astype(np.float64) * 1.609344).astype(
        np.float32
    )

    df["haversine_km"] = haversine_km(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )

    df["haversine_km2"] = (df["haversine_km"] ** 2).astype(np.float32)

    dt = df.pickup_datetime.dt
    df["year"] = dt.year.astype(np.int16)
    df["month"] = dt.month.astype(np.int8)
    df["day"] = dt.day.astype(np.int8)
    df["dayofweek"] = dt.dayofweek.astype(np.int8)
    df["hour"] = dt.hour.astype(np.int8)
    df["rush_hour"] = (
        ((df["hour"].between(7, 9)) | (df["hour"].between(16, 19)))
        & (df["dayofweek"].between(0, 4))
    ).astype(np.int8)

    df["minute"] = dt.minute.astype(np.int16)
    df["dayofyear"] = dt.dayofyear.astype(np.int16)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(np.int8)

    df["is_night"] = ((df["hour"] <= 5) | (df["hour"] >= 20)).astype(np.int8)

    df["abs_lon_diff"] = (
        (df["pickup_longitude"] - df["dropoff_longitude"]).abs().astype(np.float32)
    )
    df["abs_lat_diff"] = (
        (df["pickup_latitude"] - df["dropoff_latitude"]).abs().astype(np.float32)
    )
    df["manhattan_dist"] = (df["abs_lon_diff"] + df["abs_lat_diff"]).astype(np.float32)

    df["delta_lon"] = (df["dropoff_longitude"] - df["pickup_longitude"]).astype(
        np.float32
    )
    df["delta_lat"] = (df["dropoff_latitude"] - df["pickup_latitude"]).astype(
        np.float32
    )

    jfk_lon, jfk_lat = -73.7781, 40.6413
    lga_lon, lga_lat = -73.8740, 40.7769
    ewr_lon, ewr_lat = -74.1745, 40.6895

    def near(lat, lon, center_lat, center_lon, thr=0.05):
        return ((lat - center_lat).abs() < thr) & ((lon - center_lon).abs() < thr)

    pickup_jfk = near(df["pickup_latitude"], df["pickup_longitude"], jfk_lat, jfk_lon)
    dropoff_jfk = near(
        df["dropoff_latitude"], df["dropoff_longitude"], jfk_lat, jfk_lon
    )
    pickup_lga = near(df["pickup_latitude"], df["pickup_longitude"], lga_lat, lga_lon)
    dropoff_lga = near(
        df["dropoff_latitude"], df["dropoff_longitude"], lga_lat, lga_lon
    )
    pickup_ewr = near(df["pickup_latitude"], df["pickup_longitude"], ewr_lat, ewr_lon)
    dropoff_ewr = near(
        df["dropoff_latitude"], df["dropoff_longitude"], ewr_lat, ewr_lon
    )

    df["airport_trip"] = (
        pickup_jfk | dropoff_jfk | pickup_lga | dropoff_lga | pickup_ewr | dropoff_ewr
    ).astype(np.int8)

    nyc_lon, nyc_lat = -73.985428, 40.748817  # Midtown Manhattan (approx)

    df["pickup_to_center_miles"] = distance(
        df["pickup_latitude"], df["pickup_longitude"], nyc_lat, nyc_lon
    ).astype(np.float32)
    df["dropoff_to_center_miles"] = distance(
        df["dropoff_latitude"], df["dropoff_longitude"], nyc_lat, nyc_lon
    ).astype(np.float32)

    dlon = (df["dropoff_longitude"] - df["pickup_longitude"]).to_numpy(
        dtype=np.float64, copy=False
    )
    dlat = (df["dropoff_latitude"] - df["pickup_latitude"]).to_numpy(
        dtype=np.float64, copy=False
    )
    bearing = np.arctan2(dlon, dlat)  # radians
    df["bearing_sin"] = np.sin(bearing).astype(np.float32)
    df["bearing_cos"] = np.cos(bearing).astype(np.float32)

    df["pickup_longitude_f"] = df["pickup_longitude"].astype(np.float32)
    df["pickup_latitude_f"] = df["pickup_latitude"].astype(np.float32)
    df["dropoff_longitude_f"] = df["dropoff_longitude"].astype(np.float32)
    df["dropoff_latitude_f"] = df["dropoff_latitude"].astype(np.float32)

    df["pickup_lat_x_pickup_lon"] = (
        df["pickup_latitude_f"] * df["pickup_longitude_f"]
    ).astype(np.float32)
    df["dropoff_lat_x_dropoff_lon"] = (
        df["dropoff_latitude_f"] * df["dropoff_longitude_f"]
    ).astype(np.float32)
    df["pickup_lon_x_dropoff_lon"] = (
        df["pickup_longitude_f"] * df["dropoff_longitude_f"]
    ).astype(np.float32)
    df["pickup_lat_x_dropoff_lat"] = (
        df["pickup_latitude_f"] * df["dropoff_latitude_f"]
    ).astype(np.float32)

    return df


df_train = add_features(df_train)
df_test = add_features(df_test)

train_mask = df_train["pickup_longitude"].between(-74.3, -72.9)
train_mask &= df_train["dropoff_longitude"].between(-74.3, -72.9)
train_mask &= df_train["pickup_latitude"].between(40.5, 41.8)
train_mask &= df_train["dropoff_latitude"].between(40.5, 41.8)

train_mask &= df_train["haversine_km"].between(0.05, 200.0)  # >=50m, <=200km

train_mask &= ~(
    (df_train["abs_lon_diff"] < 1e-6)
    & (df_train["abs_lat_diff"] < 1e-6)
    & (df_train["fare_amount"] > 3.0)
)

train_mask &= df_train["distance_miles"].between(0.01, 100.0)
train_mask &= df_train["fare_amount"] >= (2.5 + 0.5 * df_train["distance_miles"])
train_mask &= df_train["fare_amount"] <= (
    2.5 + 10.0 * df_train["distance_miles"] + 50.0
)

ratio = (df_train["haversine_km"] / (df_train["distance_km"] + 1e-3)).astype(np.float32)
train_mask &= ratio.between(0.2, 5.0)

train_mask &= df_train["haversine_km"].between(0.05, 120.0)

df_train = df_train[train_mask].copy()
print("Size after additional cleaning:", len(df_train))



## === cell 6
features = [
    "year",
    "month",
    "day",
    "dayofweek",
    "hour",
    "minute",
    "dayofyear",
    "is_weekend",
    "rush_hour",
    "is_night",
    "distance_miles",
    "distance_km",
    "haversine_km",
    "haversine_km2",  # added
    "passenger_count",
    "abs_lon_diff",
    "abs_lat_diff",
    "manhattan_dist",
    "delta_lon",  # added
    "delta_lat",  # added
    "airport_trip",
    "pickup_to_center_miles",
    "dropoff_to_center_miles",
    "bearing_sin",
    "bearing_cos",
    "pickup_longitude_f",
    "pickup_latitude_f",
    "dropoff_longitude_f",
    "dropoff_latitude_f",
    "pickup_lat_x_pickup_lon",
    "dropoff_lat_x_dropoff_lon",
    "pickup_lon_x_dropoff_lon",
    "pickup_lat_x_dropoff_lat",
]

X = np.ascontiguousarray(df_train[features].to_numpy(dtype=np.float32))
y = df_train["fare_amount"].to_numpy(dtype=np.float32)
X_test = np.ascontiguousarray(df_test[features].to_numpy(dtype=np.float32))
df_test.head(5)



## === cell 7
import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold



## === cell 8
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

NTHREAD = min(max(1, mp.cpu_count()), 8)

dtrain = xgb.QuantileDMatrix(X_train, label=y_train, max_bin=256)

rng = np.random.RandomState(42)
sub_idx = rng.choice(X_train.shape[0], size=min(60000, X_train.shape[0]), replace=False)
dtrain_sub = xgb.QuantileDMatrix(X_train[sub_idx], label=y_train[sub_idx], max_bin=256)


def xgb_eva(
    max_depth,
    gamma,
    colsample_bytree,
    min_child_weight,
    reg_lambda,
    reg_alpha,
    subsample,
    eta,
):
    params = {
        "eval_metric": "rmse",
        "max_depth": int(max_depth),
        "subsample": float(subsample),
        "eta": float(eta),
        "gamma": float(gamma),
        "colsample_bytree": float(colsample_bytree),
        "min_child_weight": float(min_child_weight),
        "lambda": float(reg_lambda),
        "alpha": float(reg_alpha),
        "max_delta_step": 1.0,
        "objective": "reg:squarederror",
        "seed": 42,
        "tree_method": "hist",
        "nthread": NTHREAD,
    }
    cv_result = xgb.cv(
        params,
        dtrain_sub,
        num_boost_round=2000,
        nfold=3,
        shuffle=True,
        seed=42,
        verbose_eval=False,
        early_stopping_rounds=30,
        prediction_cache=True,  # speed: reuse CV preds cache internally
    )
    return -1.0 * cv_result["test-rmse-mean"].iloc[-1]




## === cell 9
xgb_bo = BayesianOptimization(
    xgb_eva,
    {
        "max_depth": (3, 8),
        "gamma": (0.0, 2.0),
        "colsample_bytree": (0.3, 1.0),
        "min_child_weight": (1.0, 20.0),
        "reg_lambda": (0.0, 10.0),
        "reg_alpha": (0.0, 5.0),
        "subsample": (0.5, 1.0),
        "eta": (0.02, 0.2),
    },
    random_state=42,
)
xgb_bo.maximize(init_points=3, n_iter=7)



## --- ERROR in cell 9, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mTypeError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2908328181.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     13[0m     [0mrandom_state[0m[0;34m=[0m[0;36m42[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     14[0m )
[0;32m---> 15[0;31m [0mxgb_bo[0m[0;34m.[0m[0mmaximize[0m[0;34m([0m[0minit_points[0m[0;34m=[0m[0;36m3[0m[0;34m,[0m [0mn_iter[0m[0;34m=[0m[0;36m7[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     16[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/bayes_opt/bayesian_optimization.py[0m in [0;36mmaximize[0;34m(self, init_points, n_iter)[0m
[1;32m    320[0m                 [0mx_probe[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0msuggest[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    321[0m                 [0miteration[0m [0;34m+=[0m [0;36m1[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 322[0;31m             [0mself[0m[0;34m.[0m[0mprobe[0m[0;34m([0m[0mx_probe[0m[0;34m,[0m [0mlazy[0m[0;34m=[0m[0;32mFalse[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    323[0m [0;34m[0m[0m
[1;32m    324[0m             [0;32mif[0m [0mself[0m[0;34m.[0m[0m_bounds_transformer[0m [0;32mand[0m [0miteration[0m [0;34m>[0m [0;36m0[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/bayes_opt/bayesian_optimization.py[0m in [0;36mprobe[0;34m(self, params, lazy)[0m
[1;32m    237[0m             [0mself[0m[0;34m.[0m[0m_queue[0m[0;34m.[0m[0mappend[0m[0;34m([0m[0mparams[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    238[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 239[0;31m             [0mself[0m[0;34m.[0m[0m_space[0m[0;34m.[0m[0mprobe[0m[0;34m([0m[0mparams[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    240[0m             self.logger.log_optimization_step(
[1;32m    241[0m                 [0mself[0m[0;34m.[0m[0m_space[0m[0;34m.[0m[0mkeys[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0m_space[0m[0;34m.[0m[0mres[0m[0;34m([0m[0;34m)[0m[0;34m[[0m[0;34m-[0m[0;36m1[0m[0;34m][0m[0;34m,[0m [0mself[0m[0;34m.[0m[0m_space[0m[0;34m.[0m[0mparams_config[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0mmax[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/bayes_opt/target_space.py[0m in [0;36mprobe[0;34m(self, params)[0m
[1;32m    553[0m             [0merror_msg[0m [0;34m=[0m [0;34m"No target function has been provided."[0m[0;34m[0m[0;34m[0m[0m
[1;32m    554[0m             [0;32mraise[0m [0mValueError[0m[0;34m([0m[0merror_msg[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 555[0;31m         [0mtarget[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mtarget_func[0m[0;34m([0m[0;34m**[0m[0mdict_params[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    556[0m [0;34m[0m[0m
[1;32m    557[0m         [0;32mif[0m [0mself[0m[0;34m.[0m[0m_constraint[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/2004206594.py[0m in [0;36mxgb_eva[0;34m(max_depth, gamma, colsample_bytree, min_child_weight, reg_lambda, reg_alpha, subsample, eta)[0m
[1;32m     42[0m         [0;34m"nthread"[0m[0;34m:[0m [0mNTHREAD[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     43[0m     }
[0;32m---> 44[0;31m     cv_result = xgb.cv(
[0m[1;32m     45[0m         [0mparams[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     46[0m         [0mdtrain_sub[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;31mTypeError[0m: cv() got an unexpected keyword argument 'prediction_cache'

## === cell 10
best = dict(xgb_bo.max["params"])
best["max_depth"] = int(best["max_depth"])

params = {
    "eval_metric": "rmse",
    "objective": "reg:squarederror",
    "seed": 42,
    "tree_method": "hist",
    "nthread": NTHREAD,
    "max_depth": best["max_depth"],
    "gamma": float(best["gamma"]),
    "colsample_bytree": float(best["colsample_bytree"]),
    "min_child_weight": float(best["min_child_weight"]),
    "lambda": float(best["reg_lambda"]),
    "alpha": float(best["reg_alpha"]),
    "subsample": float(best["subsample"]),
    "eta": float(best["eta"]),
    "max_delta_step": 1.0,
}

dall = xgb.QuantileDMatrix(X, label=y, max_bin=256)

cv_final = xgb.cv(
    params,
    dall,
    num_boost_round=4000,
    nfold=3,
    shuffle=True,
    seed=42,
    verbose_eval=False,
    early_stopping_rounds=30,
    prediction_cache=True,
)
best_num_boost_round = int(len(cv_final))
