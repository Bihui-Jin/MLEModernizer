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

os.environ.setdefault("PYTHONHASHSEED", "0")

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
os.environ.setdefault("BLIS_NUM_THREADS", "4")

np.random.seed(0)

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass




## === cell 1
import os


def _resolve_input(path_candidates):
    for p in path_candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of these paths exist: {path_candidates}")


TRAIN_PATH = _resolve_input(
    ["../input/train.csv", "/kaggle/input/train.csv", "/kaggle/data/train.csv"]
)
TEST_PATH = _resolve_input(
    ["../input/test.csv", "/kaggle/input/test.csv", "/kaggle/data/test.csv"]
)
SAMPLE_SUB_PATH = _resolve_input(
    [
        "../input/sample_submission.csv",
        "/kaggle/input/sample_submission.csv",
        "/kaggle/data/sample_submission.csv",
    ]
)

train_dtypes = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}
test_dtypes = {
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}

TRAIN_NROWS = 1_000_000

train_data = pd.read_csv(
    TRAIN_PATH,
    nrows=TRAIN_NROWS,
    dtype=train_dtypes,
    parse_dates=["pickup_datetime"],
)
test_data = pd.read_csv(
    TEST_PATH,
    dtype=test_dtypes,
    parse_dates=["pickup_datetime"],
)


def _ensure_utc(s):
    if pd.api.types.is_datetime64tz_dtype(s):
        return s.dt.tz_convert("UTC")
    return s.dt.tz_localize("UTC")


train_data["pickup_datetime"] = _ensure_utc(train_data["pickup_datetime"])
test_data["pickup_datetime"] = _ensure_utc(test_data["pickup_datetime"])

train_data.head()
test_data.head()




## === cell 2
train_data.head(1)




## === cell 3
train_data.shape




## === cell 4
def changeDataType(dataset):
    dataset["passenger_count"] = dataset.passenger_count.astype("uint8", copy=False)
    dataset["pickup_longitude"] = dataset.pickup_longitude.astype("float32", copy=False)
    dataset["pickup_latitude"] = dataset.pickup_latitude.astype("float32", copy=False)
    dataset["dropoff_longitude"] = dataset.dropoff_longitude.astype(
        "float32", copy=False
    )
    dataset["dropoff_latitude"] = dataset.dropoff_latitude.astype("float32", copy=False)
    if not pd.api.types.is_datetime64_any_dtype(dataset["pickup_datetime"]):
        dataset["pickup_datetime"] = pd.to_datetime(
            dataset["pickup_datetime"], errors="coerce", utc=True
        )


changeDataType(train_data)
print("--" * 40)
changeDataType(test_data)

train_data["fare_amount"] = train_data.fare_amount.astype("float32", copy=False)




## === cell 5
train_data["pickup_datetime"].head()




## === cell 6
train_data.head(1)




## === cell 7
required_cols = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
fare = train_data["fare_amount"].to_numpy(copy=False)
pc = train_data["passenger_count"].to_numpy(copy=False)

m = train_data[required_cols].notna().all(axis=1).to_numpy(copy=False)
m &= fare > 0
m &= fare < 400
m &= pc <= 6

pu_lat = train_data["pickup_latitude"]
do_lat = train_data["dropoff_latitude"]
pu_lon = train_data["pickup_longitude"]
do_lon = train_data["dropoff_longitude"]

lat_ok = pu_lat.between(-90, 90) & do_lat.between(-90, 90)
lon_ok = pu_lon.between(-180, 180) & do_lon.between(-180, 180)
m &= (lat_ok & lon_ok).to_numpy(copy=False)

NYC_LAT_MIN, NYC_LAT_MAX = 40.0, 41.5
NYC_LON_MIN, NYC_LON_MAX = -75.0, -72.0
m &= pu_lat.between(NYC_LAT_MIN, NYC_LAT_MAX).to_numpy(copy=False)
m &= do_lat.between(NYC_LAT_MIN, NYC_LAT_MAX).to_numpy(copy=False)
m &= pu_lon.between(NYC_LON_MIN, NYC_LON_MAX).to_numpy(copy=False)
m &= do_lon.between(NYC_LON_MIN, NYC_LON_MAX).to_numpy(copy=False)

train_data = train_data.loc[m]




## === cell 8
pd.set_option("float_format", "{:f}".format)
train_data.head(1)




## === cell 9
pass




## === cell 10
pass




## === cell 11
pass




## === cell 12
pass




## === cell 13
pass




## === cell 14
pass




## === cell 15
pass




## === cell 16
pass




## === cell 17
pass




## === cell 18
pass




## === cell 19
pass




## === cell 20
def degree_to_radion(degree):
    return degree * (np.pi / 180)


def calculate_distance(
    pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude
):
    from_lat = degree_to_radion(pickup_latitude)
    from_long = degree_to_radion(pickup_longitude)
    to_lat = degree_to_radion(dropoff_latitude)
    to_long = degree_to_radion(dropoff_longitude)

    radius = 6371.01
    lat_diff = to_lat - from_lat
    long_diff = to_long - from_long

    a = (
        np.sin(lat_diff / 2) ** 2
        + np.cos(from_lat) * np.cos(to_lat) * np.sin(long_diff / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return radius * c


def add_geo_features(df):
    pu_lat = df["pickup_latitude"].to_numpy(dtype=np.float32, copy=False)
    pu_lon = df["pickup_longitude"].to_numpy(dtype=np.float32, copy=False)
    do_lat = df["dropoff_latitude"].to_numpy(dtype=np.float32, copy=False)
    do_lon = df["dropoff_longitude"].to_numpy(dtype=np.float32, copy=False)

    abs_lat_diff = np.abs(pu_lat - do_lat).astype(np.float32, copy=False)
    abs_lon_diff = np.abs(pu_lon - do_lon).astype(np.float32, copy=False)

    mean_lat = (pu_lat + do_lat) * np.float32(0.5)
    mean_lat_rad = degree_to_radion(mean_lat.astype(np.float32, copy=False))

    km_per_deg_lat = np.float32(111.32)
    km_per_deg_lon = (np.float32(111.32) * np.cos(mean_lat_rad)).astype(
        np.float32, copy=False
    )

    manhattan_km = (
        abs_lat_diff * km_per_deg_lat + abs_lon_diff * km_per_deg_lon
    ).astype(np.float32, copy=False)

    df["abs_lat_diff"] = abs_lat_diff
    df["abs_lon_diff"] = abs_lon_diff
    df["manhattan_km"] = manhattan_km
    return df




## === cell 21
train_data["distance"] = calculate_distance(
    train_data["pickup_latitude"].to_numpy(dtype=np.float32, copy=False),
    train_data["pickup_longitude"].to_numpy(dtype=np.float32, copy=False),
    train_data["dropoff_latitude"].to_numpy(dtype=np.float32, copy=False),
    train_data["dropoff_longitude"].to_numpy(dtype=np.float32, copy=False),
).astype(np.float32, copy=False)




## === cell 22
train_data.head(1)




## === cell 23
test_data["distance"] = calculate_distance(
    test_data["pickup_latitude"].to_numpy(dtype=np.float32, copy=False),
    test_data["pickup_longitude"].to_numpy(dtype=np.float32, copy=False),
    test_data["dropoff_latitude"].to_numpy(dtype=np.float32, copy=False),
    test_data["dropoff_longitude"].to_numpy(dtype=np.float32, copy=False),
).astype(np.float32, copy=False)




## === cell 24
test_data.head(1)




## === cell 25
pass




## === cell 26
train_data = train_data.loc[train_data.distance < 200]  # 150




## === cell 27
train_data.head(1)




## === cell 28
pass




## === cell 29
train_data = train_data.drop(columns="key")




## === cell 30
train_data.head(1)




## === cell 31
test_data_key = test_data["key"].copy()
test_data = test_data.drop(columns="key")




## === cell 32
test_data.head()




## === cell 33
for df in (train_data, test_data):
    dt = df["pickup_datetime"]
    df["Year"] = dt.dt.year.astype(np.int16, copy=False)
    df["Month"] = dt.dt.month.astype(np.int8, copy=False)
    df["Date"] = dt.dt.day.astype(np.int8, copy=False)
    df["Day of Week"] = dt.dt.dayofweek.astype(np.int8, copy=False)
    df["Hour"] = dt.dt.hour.astype(np.int8, copy=False)




## === cell 34
train_data.head()




## === cell 35
test_data.head()




## === cell 36
pass




## === cell 37
pass




## === cell 38
pass




## === cell 39
train_data.head(1)




## === cell 40
pass




## === cell 41
pass




## === cell 42
pass




## === cell 43
pass




## === cell 44
pass




## === cell 45
pass




## === cell 46
pass




## === cell 47
pass




## === cell 48
pass




## === cell 49
m_nonzero = (
    (train_data.pickup_latitude != 0)
    & (train_data.pickup_longitude != 0)
    & (train_data.dropoff_latitude != 0)
    & (train_data.dropoff_longitude != 0)
)
train_data = train_data.loc[m_nonzero]




## === cell 50
coord_cols = [
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
]
train_medians = train_data[coord_cols].median()

test_coords = test_data[coord_cols]
m0_any = (test_coords == 0).any(axis=1)
if bool(m0_any.any()):
    for c in coord_cols:
        z = test_data[c].to_numpy(copy=False) == 0
        if z.any():
            test_data.loc[z, c] = train_medians[c]

    test_data["distance"] = calculate_distance(
        test_data["pickup_latitude"].to_numpy(dtype=np.float32, copy=False),
        test_data["pickup_longitude"].to_numpy(dtype=np.float32, copy=False),
        test_data["dropoff_latitude"].to_numpy(dtype=np.float32, copy=False),
        test_data["dropoff_longitude"].to_numpy(dtype=np.float32, copy=False),
    ).astype(np.float32, copy=False)

train_data = add_geo_features(train_data)
test_data = add_geo_features(test_data)




## === cell 51
train_data = train_data.drop(columns="pickup_datetime", axis=1)
test_data = test_data.drop(columns="pickup_datetime", axis=1)




## === cell 52
X = train_data.loc[:, train_data.columns != "fare_amount"]
y = train_data["fare_amount"]




## === cell 53
from sklearn import preprocessing  # kept to preserve import side-effects/compat
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, f1_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)




## === cell 54
X_train.head(1)




## === cell 55
pass




## === cell 56
pass




## === cell 57
pass




## === cell 58
from sklearn.ensemble import GradientBoostingRegressor

gradient_reg = GradientBoostingRegressor(random_state=0)
pass




## === cell 59
from xgboost import XGBRegressor

xgreg = XGBRegressor(random_state=0, n_estimators=300, n_jobs=4)
pass




## === cell 60
import lightgbm as lgb

model_lgb = lgb.LGBMRegressor(random_state=0, n_jobs=4)
pass




## === cell 61
from sklearn.model_selection import KFold




## === cell 62
n_folds = 5


def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42)
    rmse = np.sqrt(
        -cross_val_score(
            model, X_train, y_train, scoring="neg_mean_squared_error", cv=kf
        )
    )
    return rmse




## === cell 63
from sklearn.ensemble import GradientBoostingRegressor

GBoost = GradientBoostingRegressor(
    n_estimators=3000,
    learning_rate=0.05,
    max_depth=4,
    max_features="sqrt",
    min_samples_leaf=15,
    min_samples_split=10,
    loss="huber",
    random_state=5,
)




## === cell 64
from xgboost import XGBRegressor

model_xgb = XGBRegressor(
    colsample_bytree=0.4603,
    gamma=0.0468,
    learning_rate=0.05,
    max_depth=3,
    min_child_weight=1.7817,
    n_estimators=2200,
    reg_alpha=0.4640,
    reg_lambda=0.8571,
    subsample=0.5213,
    random_state=7,
    n_jobs=4,
)




## === cell 65
import lightgbm as lgb

model_lgb1 = lgb.LGBMRegressor(
    objective="regression",
    num_leaves=5,
    learning_rate=0.05,
    n_estimators=720,
    max_bin=55,
    bagging_fraction=0.8,
    bagging_freq=5,
    feature_fraction=0.2319,
    feature_fraction_seed=9,
    bagging_seed=9,
    min_data_in_leaf=6,
    min_sum_hessian_in_leaf=11,
    n_jobs=4,
)




## === cell 66
pass




## === cell 67
pass




## === cell 68
pass




## === cell 69
from sklearn.base import BaseEstimator


class AverageModel(BaseEstimator):
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
        return self

    def predict(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.models])
        return np.mean(predictions, axis=1)




## === cell 70
pass




## === cell 71
pass




## === cell 72
from sklearn.base import clone
from sklearn.model_selection import KFold
import xgboost as xgb
import lightgbm as lgb
from xgboost import XGBRegressor

base_model = [GBoost, model_xgb, model_lgb1]

kfold = KFold(n_splits=5, shuffle=True, random_state=0)

X_np = np.ascontiguousarray(X.to_numpy(copy=False), dtype=np.float32)
y_np = np.ascontiguousarray(y.to_numpy(copy=False), dtype=np.float32)

folds = list(kfold.split(X_np, y_np))

fold_data = []
for tr_idx, va_idx in folds:
    X_tr = X_np[tr_idx]
    y_tr = y_np[tr_idx]
    X_va = X_np[va_idx]

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dvalid = xgb.DMatrix(X_va)

    lgb_train = lgb.Dataset(X_tr, label=y_tr, free_raw_data=True)

    fold_data.append((tr_idx, va_idx, X_tr, y_tr, X_va, dtrain, dvalid, lgb_train))

out_of_fold_predictions_full = np.empty(
    (X_np.shape[0], len(base_model)), dtype=np.float32
)

trained_fold_models = [[] for _ in range(len(base_model))]

for i, model in enumerate(base_model):
    oof = np.empty(X_np.shape[0], dtype=np.float32)

    for tr_idx, va_idx, X_tr, y_tr, X_va, dtrain, dvalid, lgb_train in fold_data:
        m = clone(model)

        if isinstance(m, XGBRegressor):
            params = m.get_xgb_params()
            params["seed"] = int(params.get("random_state", 0) or 0)
            params.setdefault("objective", "reg:squarederror")

            booster = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=int(m.get_params()["n_estimators"]),
                verbose_eval=False,
            )
            pred = booster.predict(dvalid)
            trained_fold_models[i].append(booster)

        elif isinstance(m, lgb.LGBMRegressor):
            params = m.get_params()
            params_lgb = {
                "objective": params.get("objective", "regression"),
                "learning_rate": params.get("learning_rate", 0.1),
                "num_leaves": params.get("num_leaves", 31),
                "max_bin": params.get("max_bin", 255),
                "bagging_fraction": params.get("bagging_fraction", 1.0),
                "bagging_freq": params.get("bagging_freq", 0),
                "feature_fraction": params.get("feature_fraction", 1.0),
                "feature_fraction_seed": params.get("feature_fraction_seed", 0),
                "bagging_seed": params.get("bagging_seed", 0),
                "min_data_in_leaf": params.get("min_data_in_leaf", 20),
                "min_sum_hessian_in_leaf": params.get("min_sum_hessian_in_leaf", 1e-3),
                "seed": int(params.get("random_state", 0) or 0),
                "num_threads": int(params.get("n_jobs", 4) or 4),
                "verbosity": -1,
            }
            booster = lgb.train(
                params_lgb,
                lgb_train,
                num_boost_round=int(params.get("n_estimators", 100)),
            )
            pred = booster.predict(X_va, num_iteration=booster.best_iteration)
            trained_fold_models[i].append(booster)

        else:
            m.fit(X_tr, y_tr)
            pred = m.predict(X_va)
            trained_fold_models[i].append(m)

        oof[va_idx] = np.asarray(pred, dtype=np.float32)

    out_of_fold_predictions_full[:, i] = oof

out_of_fold_predictions_full




## === cell 73
meta_model = lgb.LGBMRegressor(random_state=0, n_jobs=4)
meta_model.fit(out_of_fold_predictions_full, y_np)




## === cell 74
test_np = np.ascontiguousarray(test_data.to_numpy(copy=False), dtype=np.float32)

fitted_base_preds = []

dtest = xgb.DMatrix(test_np)  # cache once
for i, m in enumerate(base_model):
    fold_models = trained_fold_models[i]

    if isinstance(m, XGBRegressor):
        preds = np.zeros(test_np.shape[0], dtype=np.float32)
        for booster in fold_models:
            preds += booster.predict(dtest).astype(np.float32, copy=False)
        preds /= np.float32(len(fold_models))
        fitted_base_preds.append(preds)

    elif isinstance(m, lgb.LGBMRegressor):
        preds = np.zeros(test_np.shape[0], dtype=np.float32)
        for booster in fold_models:
            preds += booster.predict(
                test_np, num_iteration=booster.best_iteration
            ).astype(np.float32, copy=False)
        preds /= np.float32(len(fold_models))
        fitted_base_preds.append(preds)

    else:
        preds = np.zeros(test_np.shape[0], dtype=np.float32)
        for est in fold_models:
            preds += est.predict(test_np).astype(np.float32, copy=False)
        preds /= np.float32(len(fold_models))
        fitted_base_preds.append(preds)

feature_data = np.column_stack(fitted_base_preds).astype(np.float32, copy=False)

meta_y = meta_model.predict(feature_data)

train_fare_lo = float(np.percentile(y_np, 0.5))
train_fare_hi = float(np.percentile(y_np, 99.5))
meta_y = np.clip(meta_y, train_fare_lo, train_fare_hi).astype("float32", copy=False)

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
submission = sample_sub[["key"]].copy()
key_to_pred = pd.Series(meta_y, index=test_data_key.values)
submission["fare_amount"] = submission["key"].map(key_to_pred).astype("float32")

submission["fare_amount"] = submission["fare_amount"].fillna(float(np.mean(meta_y)))

submission.to_csv("submission.csv", index=False)
submission.head()
