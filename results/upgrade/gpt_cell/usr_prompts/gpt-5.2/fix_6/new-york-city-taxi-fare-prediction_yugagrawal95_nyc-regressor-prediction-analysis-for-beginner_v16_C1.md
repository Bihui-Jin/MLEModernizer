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
import warnings

warnings.filterwarnings("ignore")

RANDOM_STATE = 0
np.random.seed(RANDOM_STATE)

DO_PLOTS = False



## === cell 1
TRAIN_NROWS = 500000

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

train_data = pd.read_csv("../input/train.csv", nrows=TRAIN_NROWS, dtype=train_dtypes)
test_data = pd.read_csv("../input/test.csv", dtype=test_dtypes)

train_data["pickup_datetime"] = pd.to_datetime(
    train_data["pickup_datetime"], errors="coerce", utc=False
)
test_data["pickup_datetime"] = pd.to_datetime(
    test_data["pickup_datetime"], errors="coerce", utc=False
)



## === cell 2
pass



## === cell 3
pass




## === cell 4
def changeDataType(dataset):
    dataset["passenger_count"] = dataset["passenger_count"].astype("uint8", copy=False)
    dataset["pickup_longitude"] = dataset["pickup_longitude"].astype(
        "float32", copy=False
    )
    dataset["pickup_latitude"] = dataset["pickup_latitude"].astype(
        "float32", copy=False
    )
    dataset["dropoff_longitude"] = dataset["dropoff_longitude"].astype(
        "float32", copy=False
    )
    dataset["dropoff_latitude"] = dataset["dropoff_latitude"].astype(
        "float32", copy=False
    )
    dataset["pickup_datetime"] = pd.to_datetime(
        dataset["pickup_datetime"], errors="coerce", utc=False
    )


changeDataType(train_data)
changeDataType(test_data)

train_data["fare_amount"] = train_data["fare_amount"].astype("float32", copy=False)



## === cell 5
pass



## === cell 6
pass



## === cell 7
train_data = train_data.dropna(axis=0)



## === cell 8
pass



## === cell 9
train_data = train_data.loc[train_data["fare_amount"] > 0]



## === cell 10
pass



## === cell 11
train_data = train_data[train_data.fare_amount < 400]



## === cell 12
pass



## === cell 13
pass



## === cell 14
train_data = train_data[train_data.passenger_count <= 6]



## === cell 15
pass



## === cell 16
pass



## === cell 17
mask = (
    (train_data["pickup_latitude"].between(-90, 90))
    & (train_data["dropoff_latitude"].between(-90, 90))
    & (train_data["pickup_longitude"].between(-180, 180))
    & (train_data["dropoff_longitude"].between(-180, 180))
)
train_data = train_data.loc[mask]



## === cell 18
pl_min, pl_max = test_data.pickup_latitude.min(), test_data.pickup_latitude.max()
plo_min, plo_max = test_data.pickup_longitude.min(), test_data.pickup_longitude.max()
dl_min, dl_max = test_data.dropoff_latitude.min(), test_data.dropoff_latitude.max()
dlo_min, dlo_max = test_data.dropoff_longitude.min(), test_data.dropoff_longitude.max()

mask = (
    train_data.pickup_latitude.between(pl_min, pl_max)
    & train_data.pickup_longitude.between(plo_min, plo_max)
    & train_data.dropoff_latitude.between(dl_min, dl_max)
    & train_data.dropoff_longitude.between(dlo_min, dlo_max)
)
train_data = train_data.loc[mask]



## === cell 19
pass




## === cell 20
def degree_to_radion(degree):
    return degree * (np.pi / 180.0)


def calculate_distance(
    pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude
):
    from_lat = degree_to_radion(pickup_latitude)
    from_long = degree_to_radion(pickup_longitude)
    to_lat = degree_to_radion(dropoff_latitude)
    to_long = degree_to_radion(dropoff_longitude)

    radius = 6371.01  # km
    lat_diff = to_lat - from_lat
    long_diff = to_long - from_long

    a = (
        np.sin(lat_diff / 2.0) ** 2
        + np.cos(from_lat) * np.cos(to_lat) * np.sin(long_diff / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return radius * c




## === cell 21
train_data["distance"] = calculate_distance(
    train_data.pickup_latitude,
    train_data.pickup_longitude,
    train_data.dropoff_latitude,
    train_data.dropoff_longitude,
).astype("float32")



## === cell 22
pass



## === cell 23
test_data["distance"] = calculate_distance(
    test_data.pickup_latitude,
    test_data.pickup_longitude,
    test_data.dropoff_latitude,
    test_data.dropoff_longitude,
).astype("float32")



## === cell 24
pass



## === cell 25
pass



## === cell 26
train_data = train_data.loc[train_data.distance < 200]  # keep as original intent



## === cell 27
pass



## === cell 28
pass



## === cell 29
train_data = train_data.drop(columns="key")



## === cell 30
pass



## === cell 31
test_data_key = test_data["key"]
test_data = test_data.drop(columns="key")



## === cell 32
pass



## === cell 33
for df in (train_data, test_data):
    dt = df["pickup_datetime"].dt
    df["Year"] = dt.year.astype("int16")
    df["Month"] = dt.month.astype("uint8")
    df["Date"] = dt.day.astype("uint8")
    df["Day of Week"] = dt.dayofweek.astype("uint8")
    df["Hour"] = dt.hour.astype("uint8")



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
train_data = train_data.loc[
    (train_data.pickup_latitude != 0)
    & (train_data.pickup_longitude != 0)
    & (train_data.dropoff_latitude != 0)
    & (train_data.dropoff_longitude != 0)
]



## === cell 50
train_data = train_data.drop(columns="pickup_datetime", axis=1)
test_data = test_data.drop(columns="pickup_datetime", axis=1)



## === cell 51
X = train_data.loc[:, train_data.columns != "fare_amount"]
y = train_data["fare_amount"]



## === cell 52
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE
)
y_train = np.log(y_train.to_numpy(dtype=np.float32))
y_test = np.log(y_test.to_numpy(dtype=np.float32))

X_train_np = np.ascontiguousarray(X_train.to_numpy(dtype=np.float32))
X_test_np = np.ascontiguousarray(X_test.to_numpy(dtype=np.float32))
test_np = np.ascontiguousarray(test_data.to_numpy(dtype=np.float32))



## === cell 53
pass



## === cell 54
from sklearn.ensemble import RandomForestRegressor

ran_for_reg = RandomForestRegressor(max_depth=400, random_state=RANDOM_STATE, n_jobs=-1)
ran_for_reg.fit(X_train_np, y_train)
y_ranfor_pred = ran_for_reg.predict(X_test_np)
error = np.sqrt(mean_squared_error(y_test, y_ranfor_pred))
error



## === cell 55
pass



## === cell 56
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

bagreg = BaggingRegressor(
    base_estimator=DecisionTreeRegressor(random_state=RANDOM_STATE),
    n_estimators=10,
    bootstrap=True,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
bagreg.fit(X_train_np, y_train)
y_bagg_pred = bagreg.predict(X_test_np)
error = np.sqrt(mean_squared_error(y_test, y_bagg_pred))
error



## === cell 57
from sklearn.ensemble import AdaBoostRegressor

adareg = AdaBoostRegressor(
    DecisionTreeRegressor(random_state=RANDOM_STATE), random_state=RANDOM_STATE
)
adareg.fit(X_train_np, y_train)
y_adareg_pred = adareg.predict(X_test_np)
error = np.sqrt(mean_squared_error(y_test, y_adareg_pred))
error



## === cell 58
from sklearn.ensemble import GradientBoostingRegressor

gradient_reg = GradientBoostingRegressor(random_state=RANDOM_STATE)
gradient_reg.fit(X_train_np, y_train)
y_gradient_pred = gradient_reg.predict(X_test_np)
error = np.sqrt(mean_squared_error(y_test, y_gradient_pred))
error



## === cell 59
from xgboost import XGBRegressor

xgreg = XGBRegressor(
    random_state=RANDOM_STATE,
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=1.0,
    colsample_bytree=1.0,
    n_jobs=-1,
)
xgreg.fit(X_train_np, y_train)
y_xgreg_pred = xgreg.predict(X_test_np)
error = np.sqrt(mean_squared_error(y_test, y_xgreg_pred))
error



## === cell 60
import lightgbm as lgb

model_lgb = lgb.LGBMRegressor(random_state=RANDOM_STATE)
model_lgb.fit(X_train_np, y_train)
y_lgb_pred = model_lgb.predict(X_test_np)
error = np.sqrt(mean_squared_error(y_test, y_lgb_pred))
error



## === cell 61
from sklearn.model_selection import cross_val_score, KFold



## === cell 62
n_folds = 5


def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42)
    rmse = np.sqrt(
        -cross_val_score(
            model,
            X_train_np,
            y_train,
            scoring="neg_mean_squared_error",
            cv=kf,
            n_jobs=1,
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
    random_state=RANDOM_STATE,
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
average_model = AverageModel(models=(GBoost, model_xgb, model_lgb1))
score = rmsle_cv(average_model)
score



## === cell 68
score.mean()



## === cell 69
base_model = [xgreg, gradient_reg, model_lgb]

from sklearn.base import clone


def test1(X_np, y_np):
    kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    folds = list(kfold.split(X_np, y_np))

    out_of_fold_predictions = np.zeros(
        (X_np.shape[0], len(base_model)), dtype=np.float32
    )

    for i, model in enumerate(base_model):
        for train_index, holdout_index in folds:
            m = clone(model)
            m.fit(X_np[train_index], y_np[train_index])
            y_pred = m.predict(X_np[holdout_index])
            out_of_fold_predictions[holdout_index, i] = y_pred.astype(
                np.float32, copy=False
            )
    return out_of_fold_predictions


out_of_fold_predictions = test1(X_train_np, y_train)
out_of_fold_predictions



## === cell 70
meta_model = lgb.LGBMRegressor(random_state=RANDOM_STATE)
meta_model.fit(out_of_fold_predictions, y_train)



## === cell 71
base_model = [xgreg, gradient_reg, model_lgb]

for m in base_model:
    m.fit(X_train_np, y_train)

feature_data = np.column_stack([m.predict(test_np) for m in base_model]).astype(
    np.float32
)



## === cell 72
meta_y = meta_model.predict(feature_data)
meta_y = np.exp(meta_y)

meta_y = np.clip(meta_y, 2.5, 500.0)



## === cell 73
aligned_test_key = test_data_key.reset_index(drop=True)

submission = pd.DataFrame(
    {"key": aligned_test_key, "fare_amount": meta_y}, columns=["key", "fare_amount"]
)

if submission["fare_amount"].isna().any():
    fallback = float(np.exp(np.nanmean(y_train)))
    submission["fare_amount"] = submission["fare_amount"].fillna(fallback)

submission.to_csv("submission.csv", index=False)
submission.head()
