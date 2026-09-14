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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



## === cell 1
TRAIN_PATH = "../input/train.csv"
TEST_PATH = "../input/test.csv"
SAMPLE_PATH = "../input/sample_submission.csv"

if not os.path.exists(TRAIN_PATH):
    TRAIN_PATH = "/kaggle/input/train.csv"
if not os.path.exists(TEST_PATH):
    TEST_PATH = "/kaggle/input/test.csv"
if not os.path.exists(SAMPLE_PATH):
    SAMPLE_PATH = "/kaggle/input/sample_submission.csv"

NROWS = 1_000_000

df = pd.read_csv(TRAIN_PATH, nrows=NROWS, usecols=[1, 2, 3, 4, 5, 6, 7])



## === cell 2
df["pickup_datetime"] = df["pickup_datetime"].str.slice(0, 16)
df["pickup_datetime"] = pd.to_datetime(
    df["pickup_datetime"], utc=True, format="%Y-%m-%d %H:%M"
)



## === cell 3
df.head()



## === cell 4
df.dtypes



## === cell 5
test = pd.read_csv(TEST_PATH).set_index("key")
test["pickup_datetime"] = test["pickup_datetime"].str.slice(0, 16)
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], utc=True, format="%Y-%m-%d %H:%M"
)
test.head()



## === cell 6
test.describe(include="all")



## === cell 7
df.dropna(how="any", axis="rows", inplace=True)

mask = df["pickup_longitude"].between(-75, -72.8)
mask &= df["dropoff_longitude"].between(-75, -72.8)
mask &= df["pickup_latitude"].between(40, 42)
mask &= df["dropoff_latitude"].between(40, 42)
mask &= df["passenger_count"].between(0, 7)
mask &= df["fare_amount"].between(0, 250)

mask &= ~(
    (df["pickup_longitude"] == df["dropoff_longitude"])
    & (df["pickup_latitude"] == df["dropoff_latitude"])
)

df = df[mask]




## === cell 8
def dist(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    R = 6371.0  # km

    lat1 = np.radians(np.asarray(pickup_lat, dtype=float))
    lon1 = np.radians(np.asarray(pickup_long, dtype=float))
    lat2 = np.radians(np.asarray(dropoff_lat, dtype=float))
    lon2 = np.radians(np.asarray(dropoff_long, dtype=float))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    return R * c




## === cell 9
def transform(data):
    data = data.copy()

    if "key" in data.columns:
        data = data.drop("key", axis=1)

    data["hour"] = data["pickup_datetime"].dt.hour
    data["day"] = data["pickup_datetime"].dt.day
    data["month"] = data["pickup_datetime"].dt.month
    data["year"] = data["pickup_datetime"].dt.year
    data = data.drop("pickup_datetime", axis=1)

    nyc = (-74.0063889, 40.7141667)
    jfk = (-73.7822222222, 40.6441666667)
    ewr = (-74.175, 40.69)
    lgr = (-73.87, 40.77)

    data["distance_to_center"] = dist(
        nyc[1], nyc[0], data["pickup_latitude"], data["pickup_longitude"]
    )
    data["pickup_distance_to_jfk"] = dist(
        jfk[1], jfk[0], data["pickup_latitude"], data["pickup_longitude"]
    )
    data["dropoff_distance_to_jfk"] = dist(
        jfk[1], jfk[0], data["dropoff_latitude"], data["dropoff_longitude"]
    )
    data["pickup_distance_to_ewr"] = dist(
        ewr[1], ewr[0], data["pickup_latitude"], data["pickup_longitude"]
    )
    data["dropoff_distance_to_ewr"] = dist(
        ewr[1], ewr[0], data["dropoff_latitude"], data["dropoff_longitude"]
    )
    data["pickup_distance_to_lgr"] = dist(
        lgr[1], lgr[0], data["pickup_latitude"], data["pickup_longitude"]
    )
    data["dropoff_distance_to_lgr"] = dist(
        lgr[1], lgr[0], data["dropoff_latitude"], data["dropoff_longitude"]
    )

    data["long_dist"] = data["pickup_longitude"] - data["dropoff_longitude"]
    data["lat_dist"] = data["pickup_latitude"] - data["dropoff_latitude"]

    data["dist"] = dist(
        data["pickup_latitude"],
        data["pickup_longitude"],
        data["dropoff_latitude"],
        data["dropoff_longitude"],
    )

    return data


df = transform(df)



## === cell 10
import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error



## === cell 11
df = df.sort_values(["year", "month", "day", "hour"]).reset_index(drop=True)

split_idx = int(len(df) * 0.75)
X_train = df.drop("fare_amount", axis=1).iloc[:split_idx].copy()
y_train = df["fare_amount"].iloc[:split_idx].copy()
X_test = df.drop("fare_amount", axis=1).iloc[split_idx:].copy()
y_test = df["fare_amount"].iloc[split_idx:].copy()
del df

X_train = X_train.apply(pd.to_numeric, errors="coerce")
X_test = X_test.apply(pd.to_numeric, errors="coerce")
X_train = X_train.fillna(0.0)
X_test = X_test.fillna(0.0)

dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_test, label=y_test)



## === cell 12
from sklearn.model_selection import TimeSeriesSplit


def xgb_evaluate(max_depth, gamma, colsample_bytree):
    params = {
        "eval_metric": "rmse",
        "max_depth": int(max_depth),
        "subsample": 0.8,
        "eta": 0.1,
        "gamma": float(gamma),
        "colsample_bytree": float(colsample_bytree),
        "objective": "reg:squarederror",
        "verbosity": 0,
        "seed": 42,
    }

    tss = TimeSeriesSplit(n_splits=3)
    rmses = []
    for tr_idx, va_idx in tss.split(X_train):
        dtr = xgb.DMatrix(X_train.iloc[tr_idx], label=y_train.iloc[tr_idx])
        dva = xgb.DMatrix(X_train.iloc[va_idx], label=y_train.iloc[va_idx])
        bst = xgb.train(
            params,
            dtr,
            num_boost_round=1200,
            evals=[(dva, "valid")],
            verbose_eval=False,
        )
        pred = bst.predict(dva)
        rmse = float(np.sqrt(mean_squared_error(y_train.iloc[va_idx], pred)))
        rmses.append(rmse)

    return -1.0 * float(np.mean(rmses))




## === cell 13
xgb_bo = BayesianOptimization(
    f=xgb_evaluate,
    pbounds={"max_depth": (3, 7), "gamma": (0, 1), "colsample_bytree": (0.3, 0.9)},
    random_state=42,
    verbose=0,
)

bo_ok = True
try:
    xgb_bo.maximize(init_points=3, n_iter=5)
except Exception as e:
    print("Bayesian optimization failed; using fallback params. Error:", repr(e))
    bo_ok = False



## === cell 14
if (
    bo_ok
    and getattr(xgb_bo, "max", None)
    and isinstance(xgb_bo.max, dict)
    and ("params" in xgb_bo.max)
):
    params = dict(xgb_bo.max["params"])
    params["max_depth"] = int(params["max_depth"])
else:
    params = {"max_depth": 6, "gamma": 0.0, "colsample_bytree": 0.7}

params["subsample"] = 0.8
params["eta"] = 0.1
params["eval_metric"] = "rmse"
params["objective"] = "reg:squarederror"
params["verbosity"] = 0
params["seed"] = 42

tss = TimeSeriesSplit(n_splits=3)
folds = [(tr, va) for tr, va in tss.split(X_train)]

cv_res = xgb.cv(
    params,
    xgb.DMatrix(X_train, label=y_train),
    num_boost_round=2000,
    folds=folds,
    seed=42,
    verbose_eval=False,
)

best_num_boost_round = int(np.argmin(cv_res["test-rmse-mean"].values) + 1)



## === cell 15
model2 = xgb.train(
    params,
    dtrain,
    num_boost_round=best_num_boost_round,
)

y_pred = model2.predict(dvalid)
y_train_pred = model2.predict(dtrain)

print("Holdout RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("Train RMSE:", np.sqrt(mean_squared_error(y_train, y_train_pred)))
print("best_num_boost_round:", best_num_boost_round)



## === cell 16
import matplotlib.pyplot as plt

fscores = pd.DataFrame(
    {"X": list(model2.get_fscore().keys()), "Y": list(model2.get_fscore().values())}
)
fscores.sort_values(by="Y").plot.bar(x="X")



## === cell 17
test_feat = transform(test)
test_feat = test_feat.apply(pd.to_numeric, errors="coerce").fillna(0.0)

dtest = xgb.DMatrix(test_feat)
y_pred_test = model2.predict(dtest)

y_pred_test = np.clip(y_pred_test, 0, None)

submission = pd.DataFrame({"key": test.index, "fare_amount": y_pred_test})
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
