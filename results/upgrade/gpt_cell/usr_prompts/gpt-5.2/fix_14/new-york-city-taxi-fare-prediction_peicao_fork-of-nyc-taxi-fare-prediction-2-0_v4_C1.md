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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os

print(os.listdir("../input"))
os.chdir("/kaggle/working/")



## === cell 1
df_train = pd.read_csv(
    "../input/train.csv", nrows=800000, parse_dates=["pickup_datetime"]
)
df_train.head()



## === cell 2
df_train.describe()
df_train.dtypes



## === cell 3
df_train = df_train[(df_train["fare_amount"] > 0.05) & (df_train.passenger_count > 0)]
df_train.dropna(how="any", axis="rows", inplace=True)
print("New Size: {}".format(len(df_train)))
df_test = pd.read_csv("../input/test.csv", parse_dates=["pickup_datetime"])



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
    df = df.copy()

    df["distance_miles"] = distance(
        df.pickup_latitude,
        df.pickup_longitude,
        df.dropoff_latitude,
        df.dropoff_longitude,
    )

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

    df["year"] = df.pickup_datetime.dt.year
    df["month"] = df.pickup_datetime.dt.month
    df["day"] = df.pickup_datetime.dt.day
    df["dayofweek"] = df.pickup_datetime.dt.dayofweek
    df["hour"] = df.pickup_datetime.dt.hour
    df["rush_hour"] = (
        ((df["hour"].between(7, 9)) | (df["hour"].between(16, 19)))
        & (df["dayofweek"].between(0, 4))
    ).astype(np.int8)

    df["minute"] = df.pickup_datetime.dt.minute.astype(np.int16)
    df["dayofyear"] = df.pickup_datetime.dt.dayofyear.astype(np.int16)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(np.int8)

    df["is_night"] = ((df["hour"] <= 5) | (df["hour"] >= 20)).astype(np.int8)

    df["abs_lon_diff"] = (df["pickup_longitude"] - df["dropoff_longitude"]).abs()
    df["abs_lat_diff"] = (df["pickup_latitude"] - df["dropoff_latitude"]).abs()
    df["manhattan_dist"] = df["abs_lon_diff"] + df["abs_lat_diff"]

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
    )
    df["dropoff_to_center_miles"] = distance(
        df["dropoff_latitude"], df["dropoff_longitude"], nyc_lat, nyc_lon
    )

    dlon = (df["dropoff_longitude"] - df["pickup_longitude"]).astype(np.float64)
    dlat = (df["dropoff_latitude"] - df["pickup_latitude"]).astype(np.float64)
    bearing = np.arctan2(dlon.values, dlat.values)  # radians
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

X = df_train[features].astype(np.float32).values
y = df_train["fare_amount"].values.astype(np.float32)
X_test = df_test[features].astype(np.float32)
df_test.head(5)



## === cell 7
import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split



## === cell 8
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train)

rng = np.random.RandomState(42)
sub_idx = rng.choice(X_train.shape[0], size=min(60000, X_train.shape[0]), replace=False)
dtrain_sub = xgb.DMatrix(X_train[sub_idx], label=y_train[sub_idx])


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
        "nthread": 4,
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



## === cell 10
best = dict(xgb_bo.max["params"])
best["max_depth"] = int(best["max_depth"])

params = {
    "eval_metric": "rmse",
    "objective": "reg:squarederror",
    "seed": 42,
    "nthread": 4,
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

cv_final = xgb.cv(
    params,
    xgb.DMatrix(X, label=y),
    num_boost_round=4000,
    nfold=3,
    shuffle=True,
    seed=42,
    verbose_eval=False,
    early_stopping_rounds=30,
)
best_num_boost_round = int(len(cv_final))



## === cell 11
model2 = xgb.train(
    params, xgb.DMatrix(X, label=y), num_boost_round=best_num_boost_round
)

X_testm = xgb.DMatrix(X_test.values)
y_test = model2.predict(X_testm)

y_test = np.clip(y_test, 0.0, 250.0)



## === cell 12
X_test.head(5)



## === cell 13
sub = pd.DataFrame()
sub["key"] = df_test.key
sub["fare_amount"] = y_test
sub.to_csv("submission.csv", index=False)
sub
