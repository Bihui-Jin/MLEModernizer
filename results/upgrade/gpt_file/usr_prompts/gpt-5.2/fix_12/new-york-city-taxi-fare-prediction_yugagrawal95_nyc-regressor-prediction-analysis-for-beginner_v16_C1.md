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
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

np.random.seed(0)

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass



## === cell 1
TRAIN_PATH = "/kaggle/input/train.csv"
TEST_PATH = "/kaggle/input/test.csv"
SAMPLE_SUB_PATH = "/kaggle/input/sample_submission.csv"

train_cols = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
test_cols = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
dtype_train = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}
dtype_test = {
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}

train_data = pd.read_csv(
    TRAIN_PATH,
    nrows=500000,
    usecols=train_cols,
    dtype=dtype_train,
    parse_dates=["pickup_datetime"],
    engine="c",
    low_memory=False,
)
test_data = pd.read_csv(
    TEST_PATH,
    usecols=test_cols,
    dtype=dtype_test,
    parse_dates=["pickup_datetime"],
    engine="c",
    low_memory=False,
)



## === cell 2
pass



## === cell 3
pass



## === cell 4
from pandas.api.types import is_datetime64_any_dtype


def changeDataType(dataset):
    if dataset["passenger_count"].dtype != np.uint8:
        dataset["passenger_count"] = dataset["passenger_count"].astype(
            "uint8", copy=False
        )
    for c in [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
    ]:
        if dataset[c].dtype != np.float32:
            dataset[c] = dataset[c].astype("float32", copy=False)

    dt = dataset["pickup_datetime"]
    if not is_datetime64_any_dtype(dt):
        dataset["pickup_datetime"] = pd.to_datetime(dt, utc=True, errors="coerce")
    else:
        try:
            if getattr(dt.dt, "tz", None) is None:
                dataset["pickup_datetime"] = dt.dt.tz_localize(
                    "UTC", nonexistent="NaT", ambiguous="NaT"
                )
            else:
                dataset["pickup_datetime"] = dt.dt.tz_convert("UTC")
        except Exception:
            dataset["pickup_datetime"] = pd.to_datetime(dt, utc=True, errors="coerce")


changeDataType(train_data)
changeDataType(test_data)

train_data["fare_amount"] = train_data["fare_amount"].astype("float32", copy=False)



## === cell 5
pass



## === cell 6
pass



## === cell 7
mask_ok = train_data.notna().all(axis=1).to_numpy()
mask_ok &= np.isfinite(train_data["fare_amount"].to_numpy(copy=False))
train_data = train_data.loc[mask_ok]



## === cell 8
pd.set_option("display.float_format", "{:f}".format)



## === cell 9
fa = train_data["fare_amount"].to_numpy(copy=False)
train_data = train_data.loc[(fa > 0) & (fa < 400)]



## === cell 10
pass



## === cell 11
pass



## === cell 12
pass



## === cell 13
pass



## === cell 14
train_data = train_data.loc[train_data["passenger_count"] <= 6]



## === cell 15
pass



## === cell 16
pass



## === cell 17
pl = train_data["pickup_latitude"].to_numpy(copy=False)
dl = train_data["dropoff_latitude"].to_numpy(copy=False)
plo = train_data["pickup_longitude"].to_numpy(copy=False)
dlo = train_data["dropoff_longitude"].to_numpy(copy=False)

mask_geo = (
    (pl >= -90)
    & (pl <= 90)
    & (dl >= -90)
    & (dl <= 90)
    & (plo >= -180)
    & (plo <= 180)
    & (dlo >= -180)
    & (dlo <= 180)
)
train_data = train_data.loc[mask_geo]



## === cell 18
NYC_BOUNDS = {
    "pickup_latitude": (40.0, 42.0),
    "dropoff_latitude": (40.0, 42.0),
    "pickup_longitude": (-75.0, -72.0),
    "dropoff_longitude": (-75.0, -72.0),
}
pl = train_data["pickup_latitude"].to_numpy(copy=False)
dl = train_data["dropoff_latitude"].to_numpy(copy=False)
plo = train_data["pickup_longitude"].to_numpy(copy=False)
dlo = train_data["dropoff_longitude"].to_numpy(copy=False)

mask_nyc = (
    (pl >= NYC_BOUNDS["pickup_latitude"][0])
    & (pl <= NYC_BOUNDS["pickup_latitude"][1])
    & (dl >= NYC_BOUNDS["dropoff_latitude"][0])
    & (dl <= NYC_BOUNDS["dropoff_latitude"][1])
    & (plo >= NYC_BOUNDS["pickup_longitude"][0])
    & (plo <= NYC_BOUNDS["pickup_longitude"][1])
    & (dlo >= NYC_BOUNDS["dropoff_longitude"][0])
    & (dlo <= NYC_BOUNDS["dropoff_longitude"][1])
)
train_data = train_data.loc[mask_nyc]



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




## === cell 21
train_data["distance"] = calculate_distance(
    train_data["pickup_latitude"].to_numpy(copy=False),
    train_data["pickup_longitude"].to_numpy(copy=False),
    train_data["dropoff_latitude"].to_numpy(copy=False),
    train_data["dropoff_longitude"].to_numpy(copy=False),
).astype("float32", copy=False)



## === cell 22
pass



## === cell 23
test_data["distance"] = calculate_distance(
    test_data["pickup_latitude"].to_numpy(copy=False),
    test_data["pickup_longitude"].to_numpy(copy=False),
    test_data["dropoff_latitude"].to_numpy(copy=False),
    test_data["dropoff_longitude"].to_numpy(copy=False),
).astype("float32", copy=False)



## === cell 24
pass



## === cell 25
pass



## === cell 26
train_data = train_data.loc[train_data.distance < 200]  # keep core logic



## === cell 27
pass



## === cell 28
pass



## === cell 29
train_data = train_data.drop(columns="key")



## === cell 30
pass



## === cell 31
test_data_key = test_data["key"].copy()
test_data = test_data.drop(columns="key")



## === cell 32
pass



## === cell 33
data = [train_data, test_data]
for i in data:
    dt = i["pickup_datetime"]
    if not is_datetime64_any_dtype(dt):
        dt = pd.to_datetime(dt, utc=True, errors="coerce")
    else:
        try:
            if getattr(dt.dt, "tz", None) is None:
                dt = dt.dt.tz_localize("UTC", nonexistent="NaT", ambiguous="NaT")
            else:
                dt = dt.dt.tz_convert("UTC")
        except Exception:
            dt = pd.to_datetime(dt, utc=True, errors="coerce")

    dt = dt.dt.tz_localize(None)

    dtdt = dt.dt
    i["Year"] = dtdt.year.astype("int16", copy=False)
    i["Month"] = dtdt.month.astype("int8", copy=False)
    i["Date"] = dtdt.day.astype("int8", copy=False)
    i["Day of Week"] = dtdt.dayofweek.astype("int8", copy=False)
    i["Hour"] = dtdt.hour.astype("int8", copy=False)



## === cell 34
pass



## === cell 35
pass



## === cell 36
pass



## === cell 37
pass



## === cell 38
pass



## === cell 39
pass



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
zmask = (
    (train_data["pickup_latitude"].to_numpy(copy=False) != 0)
    & (train_data["pickup_longitude"].to_numpy(copy=False) != 0)
    & (train_data["dropoff_latitude"].to_numpy(copy=False) != 0)
    & (train_data["dropoff_longitude"].to_numpy(copy=False) != 0)
)
train_data = train_data.loc[zmask]



## === cell 50
cols = ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]
for col in cols:
    med = float(test_data.loc[test_data[col] != 0, col].median())
    arr = test_data[col].to_numpy(copy=False)
    zeros = arr == 0
    if zeros.any():
        arr[zeros] = med



## === cell 51
train_data = train_data.drop(columns="pickup_datetime", axis=1)
test_data = test_data.drop(columns="pickup_datetime", axis=1)



## === cell 52
X = train_data.loc[:, train_data.columns != "fare_amount"]
y = train_data["fare_amount"]



## === cell 53
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



## === cell 54
pass



## === cell 55
SKIP_UNUSED_BASELINE_EVALS = True

if not SKIP_UNUSED_BASELINE_EVALS:
    from sklearn.ensemble import RandomForestRegressor

    ran_for_reg = RandomForestRegressor(max_depth=400, random_state=0, n_jobs=-1)
    ran_for_reg.fit(X_train, y_train)
    y_ranfor_pred = ran_for_reg.predict(X_test)
    error = np.sqrt(mean_squared_error(y_test, y_ranfor_pred))
    error



## === cell 56
if not SKIP_UNUSED_BASELINE_EVALS:
    from sklearn.ensemble import BaggingRegressor
    from sklearn.tree import DecisionTreeRegressor

    bagreg = BaggingRegressor(
        estimator=DecisionTreeRegressor(),
        n_estimators=10,
        bootstrap=True,
        random_state=0,
        n_jobs=-1,
    )
    bagreg.fit(X_train, y_train)
    y_bagg_pred = bagreg.predict(X_test)
    error = np.sqrt(mean_squared_error(y_test, y_bagg_pred))
    error



## === cell 57
if not SKIP_UNUSED_BASELINE_EVALS:
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.tree import DecisionTreeRegressor

    adareg = AdaBoostRegressor(DecisionTreeRegressor(random_state=0), random_state=0)
    adareg.fit(X_train, y_train)
    y_adareg_pred = adareg.predict(X_test)
    error = np.sqrt(mean_squared_error(y_test, y_adareg_pred))
    error



## === cell 58
from sklearn.ensemble import GradientBoostingRegressor

gradient_reg = GradientBoostingRegressor(random_state=0)
gradient_reg.fit(X_train, y_train)
y_gradient_pred = gradient_reg.predict(X_test)
error = np.sqrt(mean_squared_error(y_test, y_gradient_pred))
error



## === cell 59
from xgboost import XGBRegressor

xgreg = XGBRegressor(random_state=0, n_estimators=100, n_jobs=-1, verbosity=0)
xgreg.fit(X_train, y_train)
y_xgreg_pred = xgreg.predict(X_test)
error = np.sqrt(mean_squared_error(y_test, y_xgreg_pred))
error



## === cell 60
import lightgbm as lgb

model_lgb = lgb.LGBMRegressor(random_state=0, n_jobs=-1)
model_lgb.fit(X_train, y_train)
y_lgb_pred = model_lgb.predict(X_test)
error = np.sqrt(mean_squared_error(y_test, y_lgb_pred))
error



## === cell 61
from sklearn.model_selection import cross_val_score, KFold



## === cell 62
n_folds = 5


def rmsle_cv(model):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    rmse = np.sqrt(
        -cross_val_score(
            model, X_train, y_train, scoring="neg_mean_squared_error", cv=kf
        )
    )
    return rmse




## === cell 63
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
    verbosity=0,
    random_state=7,
    nthread=-1,
)



## === cell 65
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
    n_jobs=-1,
)



## === cell 66
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




## === cell 67
RUN_EXPENSIVE_CV = False
if RUN_EXPENSIVE_CV:
    average_model = AverageModel(models=(GBoost, model_xgb, model_lgb1))
    score = rmsle_cv(average_model)
else:
    score = np.array([], dtype=np.float32)
score



## === cell 68
score.mean() if score.size else None



## === cell 69
from sklearn.base import clone
import xgboost as xgb

base_model = [GBoost, model_xgb, model_lgb1]

X_np_full = np.ascontiguousarray(X.to_numpy(copy=False), dtype=np.float32)
y_np_full = np.ascontiguousarray(y.to_numpy(copy=False), dtype=np.float32)

kfold = KFold(n_splits=5, shuffle=True, random_state=0)
splits = [
    (tr, ho) for tr, ho in kfold.split(X_np_full, y_np_full)
]  # cache indices once


def _fit_predict_sklearn_estimator(estimator, Xtr, ytr, Xho):
    estimator.fit(Xtr, ytr)
    return estimator.predict(Xho)


def _fit_predict_xgb_regressor(model_template, Xtr, ytr, Xho):
    params = model_template.get_xgb_params()
    num_boost_round = int(model_template.get_params().get("n_estimators", 100))
    dtr = xgb.DMatrix(Xtr, label=ytr)
    dho = xgb.DMatrix(Xho)
    booster = xgb.train(params, dtr, num_boost_round=num_boost_round)
    return booster.predict(dho), booster


def _fit_predict_lgb_regressor(model_template, Xtr, ytr, Xho):
    params = model_template.get_params()
    num_boost_round = int(params.pop("n_estimators"))
    params.setdefault("verbosity", -1)
    dtr = lgb.Dataset(Xtr, label=ytr, free_raw_data=True)
    booster = lgb.train(params, dtr, num_boost_round=num_boost_round)
    return booster.predict(Xho), booster


def test1(X_np, y_np):
    out_of_fold_predictions = np.zeros(
        (X_np.shape[0], len(base_model)), dtype=np.float32
    )

    fitted_full_models = []

    for i, model in enumerate(base_model):
        for train_index, holdout_index in splits:
            Xtr = X_np[train_index]
            ytr = y_np[train_index]
            Xho = X_np[holdout_index]

            if isinstance(model, XGBRegressor):
                y_pred, _ = _fit_predict_xgb_regressor(model, Xtr, ytr, Xho)
            elif isinstance(model, lgb.LGBMRegressor):
                y_pred, _ = _fit_predict_lgb_regressor(model, Xtr, ytr, Xho)
            else:
                m = clone(model)
                y_pred = _fit_predict_sklearn_estimator(m, Xtr, ytr, Xho)

            out_of_fold_predictions[holdout_index, i] = np.asarray(
                y_pred, dtype=np.float32
            )

        if isinstance(model, XGBRegressor):
            _, booster = _fit_predict_xgb_regressor(model, X_np, y_np, X_np[:1])
            fitted_full_models.append(booster)
        elif isinstance(model, lgb.LGBMRegressor):
            _, booster = _fit_predict_lgb_regressor(model, X_np, y_np, X_np[:1])
            fitted_full_models.append(booster)
        else:
            model.fit(X_np, y_np)
            fitted_full_models.append(model)

    return out_of_fold_predictions, fitted_full_models


out_of_fold_predictions, fitted_base_models = test1(X_np_full, y_np_full)
out_of_fold_predictions



## === cell 70
meta_model = lgb.LGBMRegressor(random_state=0, n_jobs=-1)
meta_model.fit(out_of_fold_predictions, y_np_full)



## === cell 71
test_np = np.ascontiguousarray(test_data.to_numpy(copy=False), dtype=np.float32)

pred_cols = []
for m in fitted_base_models:
    if isinstance(m, xgb.Booster):
        pred_cols.append(m.predict(xgb.DMatrix(test_np)))
    elif isinstance(m, lgb.Booster):
        pred_cols.append(m.predict(test_np))
    else:
        pred_cols.append(m.predict(test_np))

feature_data = np.column_stack(pred_cols).astype(np.float32, copy=False)
feature_data.shape



## === cell 72
meta_y = meta_model.predict(feature_data).astype(np.float32, copy=False)
meta_y = np.clip(meta_y, 0.0, None)

assert len(test_data_key) == len(meta_y), (len(test_data_key), len(meta_y))

submission = pd.DataFrame(
    {"key": test_data_key, "fare_amount": meta_y}, columns=["key", "fare_amount"]
)
submission.to_csv("submission.csv", index=False)
submission.head()
