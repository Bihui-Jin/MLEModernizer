# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

3.12

# 3. Installed packages



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

# 5. Target score

5.67556

# 6. Current score

7.38298

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 7.403) has done: 'I fix the two seaborn API errors (newer seaborn requires passing `x=`/`y=` and `data=` when using `order=` and regplot no longer accepts two positional vectors). I also remove the `pip install` step (internet is unavailable on Kaggle), and make the pipeline robust to missing optional packages (holoviews/catboost/lightgbm) so it still runs end-to-end. To nudge RMSE down toward your target with minimal semantic change, I make the final submission come from the better-performing “fare = fare_amount - base_fare” modeling path (cells 88–90) and ensure it is written as a valid `.csv` with the exact required columns. Finally, I add a simple non-negative clipping on predictions (fare can’t be negative) to avoid extreme errors and improve score slightly.'
- What this solution (achieved 7.37804) has done: 'You currently don’t have a Kaggle score because the notebook produces a submission but it can be invalid/misaligned: you filter `df_test` rows (cell 43), which changes the required row count and keys, and that can break submission validity or silently hurt RMSE. I keep your exact modeling approach, but stop filtering the test set (only clean/filter train), and instead handle invalid/out-of-bound test rows by producing predictions for all original test keys (with safe clipping). I also ensure `StandardScaler` is fit only on train features (as you already do) and that the final submission uses the original test ordering/keys so it’s always accepted. These are minimal changes that should move RMSE down versus a broken/misaligned submission and get you closer to the target.'
- What this solution (achieved 7.39092) has done: 'Your current RMSE (7.378) is worse than the target (5.676), so we should make a small, legitimate improvement without changing the overall approach. The biggest low-risk gain here is to fix a subtle evaluation mismatch: in the “fare = fare_amount - base_fare” path you train on a target that includes zero-distance rides, but you filtered those out earlier; also your `haversineDistanceInKM()` helper truncates distances to `int` (not used later, but it’s a signal that quantization may have crept into thinking). I keep your exact model choices and pipeline, but make the train cleaning consistent with the engineered “fare” target by removing impossible/near-zero distance rides before training that final model, and I add a conservative upper clip on predictions (NYC taxi fares rarely exceed a few hundred in this dataset) to reduce outlier RMSE without changing the core logic. The submission writing and key alignment remain unchanged and still produce a valid `submission.csv`.'
- What this solution (achieved 7.38298) has done: 'Your current score (7.39092 RMSE) is worse than the target (5.67556), so we should make a small, legitimate improvement without changing the overall modeling approach. The lowest-risk gain is to remove obvious outliers in the training target (`fare_amount`) and ensure `fare = fare_amount - base_fare` remains non-negative during the final-model training, which reduces the impact of label noise and extreme rides on RMSE. I keep your exact feature engineering and final “model fare then add base_fare” logic, but add conservative NYC-taxi-specific cleaning for training only (never filtering test) and apply consistent clipping to the intermediate `fare` and final predictions. This should nudge RMSE down while preserving your pipeline structure and producing the same valid `submission.csv` format.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import warnings

warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor



## === cell 1
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ.setdefault("PYTHONHASHSEED", str(SEED))
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")



## === cell 2
TRAIN_PATH = "../input/new-york-city-taxi-fare-prediction/train.csv"
TEST_PATH = "../input/new-york-city-taxi-fare-prediction/test.csv"

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

train_dtypes = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
    "key": "string",
}
test_dtypes = {
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
    "key": "string",
}

df = pd.read_csv(TRAIN_PATH, usecols=train_cols, nrows=200000, dtype=train_dtypes)
df_test = pd.read_csv(TEST_PATH, usecols=test_cols, dtype=test_dtypes)

df_test_full = df_test.copy()

df.head()



## === cell 3
df.shape



## === cell 4
_ = df.dtypes



## === cell 5
pass



## === cell 6
pass



## === cell 7
pass



## === cell 8
pass



## === cell 9
pass



## === cell 10
df.dropna(axis=0, inplace=True)
np.sum(pd.isnull(df))



## === cell 11
df = df[(df["fare_amount"] > 0) & (df["fare_amount"] < 250)].copy()



## === cell 12
df.loc[df["fare_amount"] < 0, "fare_amount"] = 0.1
df[df["fare_amount"] < 0]



## === cell 13
df["pickup_datetime"] = pd.to_datetime(df.pickup_datetime, utc=False, errors="coerce")
df_test["pickup_datetime"] = pd.to_datetime(
    df_test.pickup_datetime, utc=False, errors="coerce"
)
df_test_full["pickup_datetime"] = pd.to_datetime(
    df_test_full.pickup_datetime, utc=False, errors="coerce"
)



## === cell 14
df.loc[:, "pickup_hour"] = df["pickup_datetime"].dt.hour.astype("int8")
df.loc[:, "pickup_date"] = df["pickup_datetime"].dt.day.astype("int8")
df.loc[:, "pickup_month"] = df["pickup_datetime"].dt.month.astype("int8")
df.loc[:, "pickup_day"] = df["pickup_datetime"].dt.dayofweek.astype("int8")

df_test.loc[:, "pickup_hour"] = df_test["pickup_datetime"].dt.hour.astype("int8")
df_test.loc[:, "pickup_date"] = df_test["pickup_datetime"].dt.day.astype("int8")
df_test.loc[:, "pickup_month"] = df_test["pickup_datetime"].dt.month.astype("int8")
df_test.loc[:, "pickup_day"] = df_test["pickup_datetime"].dt.dayofweek.astype("int8")

df_test_full.loc[:, "pickup_hour"] = df_test_full["pickup_datetime"].dt.hour.astype(
    "int8"
)
df_test_full.loc[:, "pickup_date"] = df_test_full["pickup_datetime"].dt.day.astype(
    "int8"
)
df_test_full.loc[:, "pickup_month"] = df_test_full["pickup_datetime"].dt.month.astype(
    "int8"
)
df_test_full.loc[:, "pickup_day"] = df_test_full["pickup_datetime"].dt.dayofweek.astype(
    "int8"
)

df.loc[:, "pickup_weekday"] = "NA"
df_test.loc[:, "pickup_weekday"] = "NA"
df_test_full.loc[:, "pickup_weekday"] = "NA"



## === cell 15
h = df["pickup_hour"].to_numpy()
base = np.full(h.shape, 2.50, dtype=np.float32)
base[(h >= 16) & (h <= 19)] = 3.50
base[(h >= 20) & (h <= 23)] = 3.00
df["base_fare"] = base

h2 = df_test["pickup_hour"].to_numpy()
base2 = np.full(h2.shape, 2.50, dtype=np.float32)
base2[(h2 >= 16) & (h2 <= 19)] = 3.50
base2[(h2 >= 20) & (h2 <= 23)] = 3.00
df_test["base_fare"] = base2

h2f = df_test_full["pickup_hour"].to_numpy()
base2f = np.full(h2f.shape, 2.50, dtype=np.float32)
base2f[(h2f >= 16) & (h2f <= 19)] = 3.50
base2f[(h2f >= 20) & (h2f <= 23)] = 3.00
df_test_full["base_fare"] = base2f

df["base_fare"], df["pickup_hour"]



## === cell 16
df["fare"] = (df["fare_amount"] - df["base_fare"]).clip(lower=0.0)



## === cell 17
pass



## === cell 18
from math import radians, cos, sin, asin, sqrt


def haversineDistanceInKM(latA, lonA, latB, lonB):
    lonA, latA, lonB, latB = map(radians, [lonA, latA, lonB, latB])
    return int(
        12734
        * asin(
            sqrt(
                sin((latB - latA) / 2) ** 2
                + cos(latA) * cos(latB) * sin((lonB - lonA) / 2) ** 2
            )
        )
    )


latA = df["pickup_latitude"].iloc[0]
lonA = df["pickup_longitude"].iloc[0]
latB = df["dropoff_latitude"].iloc[0]
lonB = df["dropoff_longitude"].iloc[0]
print(haversineDistanceInKM(latA, lonA, latB, lonB))




## === cell 19
def haversine_distance(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h


df["haversine_distance"] = haversine_distance(
    df["pickup_latitude"].values,
    df["pickup_longitude"].values,
    df["dropoff_latitude"].values,
    df["dropoff_longitude"].values,
).astype("float32")
df_test["haversine_distance"] = haversine_distance(
    df_test["pickup_latitude"].values,
    df_test["pickup_longitude"].values,
    df_test["dropoff_latitude"].values,
    df_test["dropoff_longitude"].values,
).astype("float32")
df_test_full["haversine_distance"] = haversine_distance(
    df_test_full["pickup_latitude"].values,
    df_test_full["pickup_longitude"].values,
    df_test_full["dropoff_latitude"].values,
    df_test_full["dropoff_longitude"].values,
).astype("float32")



## === cell 20
df["haversine_distance"].median(), df["haversine_distance"].mean(),



## === cell 21
df.head()



## === cell 22
pass



## === cell 23
pass



## === cell 24
pass



## === cell 25
pass



## === cell 26
pass



## === cell 27
pass



## === cell 28
pass



## === cell 29
pass



## === cell 30
pass



## === cell 31
pass



## === cell 32
pass



## === cell 33
pass



## === cell 34
IQR = df["haversine_distance"].quantile(0.75) - df["haversine_distance"].quantile(0.25)
IQR



## === cell 35
Q1 = df["haversine_distance"].quantile(0.25)
Q3 = df["haversine_distance"].quantile(0.75)
whisker_1 = Q1 - (1.5 * IQR)
whisker_2 = Q3 + (1.5 * IQR)

whisker_1, whisker_2



## === cell 36
df = df.loc[(df["haversine_distance"] > 0.001) & (df["haversine_distance"] < 8)]
df.shape



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
df = df.loc[(df.pickup_latitude > 40.6) & (df.pickup_latitude < 40.9)]
df = df.loc[(df.dropoff_latitude > 40.6) & (df.dropoff_latitude < 40.9)]
df = df.loc[(df.dropoff_longitude > -74.05) & (df.dropoff_longitude < -73.7)]
df = df.loc[(df.pickup_longitude > -74.05) & (df.pickup_longitude < -73.7)]



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
X = df.drop(
    [
        "key",
        "pickup_datetime",
        "pickup_weekday",
        "fare_amount",
        "fare",
        "base_fare",
        "dropoff_latitude",
        "dropoff_longitude",
    ],
    axis=1,
)
y = df["fare_amount"]



## === cell 49
from sklearn import preprocessing

X = preprocessing.StandardScaler().fit_transform(X)
X[0:5]



## === cell 50
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=SEED
)
print(X_train.ndim)
print(y_train.ndim)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)



## === cell 51
from sklearn.metrics import mean_squared_error
from math import sqrt

mean_pred = np.repeat(y_train.mean(), len(y_test))
sqrt(mean_squared_error(y_test, mean_pred))



## === cell 52
print("Skipping pip install (no internet).")



## === cell 53
pass



## === cell 54
pass



## === cell 55
pass



## === cell 56
pass



## === cell 57
pass



## === cell 58
pass



## === cell 59
pass



## === cell 60
pass



## === cell 61
pass



## === cell 62
pass



## === cell 63
pass



## === cell 64
pass



## === cell 65
pass



## === cell 66
pass



## === cell 67
df["haversine_distance_log"] = np.log(df["haversine_distance"].values + 1).astype(
    "float32"
)
df["haversine_distance_sqrt"] = np.sqrt(df["haversine_distance"].values).astype(
    "float32"
)
df["haversine_distance_sq"] = (df["haversine_distance"].values ** 2).astype("float32")

df_test["haversine_distance_log"] = np.log(
    df_test["haversine_distance"].values + 1
).astype("float32")
df_test["haversine_distance_sqrt"] = np.sqrt(
    df_test["haversine_distance"].values
).astype("float32")
df_test["haversine_distance_sq"] = (df_test["haversine_distance"].values ** 2).astype(
    "float32"
)

df_test_full["haversine_distance_log"] = np.log(
    df_test_full["haversine_distance"].values + 1
).astype("float32")
df_test_full["haversine_distance_sqrt"] = np.sqrt(
    df_test_full["haversine_distance"].values
).astype("float32")
df_test_full["haversine_distance_sq"] = (
    df_test_full["haversine_distance"].values ** 2
).astype("float32")



## === cell 68
df_train = df.drop(
    [
        "key",
        "pickup_datetime",
        "pickup_weekday",
        "fare",
        "fare_amount",
        "base_fare",
        "haversine_distance",
        "haversine_distance_sq",
        "haversine_distance_sqrt",
    ],
    axis=1,
)

df_test_copy = df_test_full.drop(
    [
        "key",
        "pickup_datetime",
        "pickup_weekday",
        "base_fare",
        "haversine_distance",
        "haversine_distance_sq",
        "haversine_distance_sqrt",
    ],
    axis=1,
)

df_test_copy = df_test_copy.reindex(columns=df_train.columns)

X = df_train.copy()
y = df["fare_amount"]
df_train.columns, df_test_copy.columns



## === cell 69
from sklearn import preprocessing

scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)
test_X = scaler.transform(df_test_copy)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=SEED
)



## === cell 70
from sklearn.linear_model import RidgeCV

ridge = RidgeCV(cv=5).fit(X_train, y_train)
ridge.score(X_train, y_train)



## === cell 71
df_test_full.head()




## === cell 72
def model_train_evaluation(y, ypred, model_name):
    from sklearn.metrics import (
        mean_squared_error,
        mean_absolute_error,
        explained_variance_score,
        r2_score,
        mean_absolute_percentage_error,
    )

    print("\n \n Model Evaluation Report: ")
    print("Mean Absolute Error(MAE) of", model_name, ":", mean_absolute_error(y, ypred))
    print("Mean Squared Error(MSE) of", model_name, ":", mean_squared_error(y, ypred))
    print(
        "Root Mean Squared Error (RMSE) of",
        model_name,
        ":",
        mean_squared_error(y, ypred, squared=False),
    )
    print(
        "Mean absolute percentage error (MAPE) of",
        model_name,
        ":",
        mean_absolute_percentage_error(y, ypred),
    )
    print(
        "Explained Variance Score (EVS) of",
        model_name,
        ":",
        explained_variance_score(y, ypred),
    )
    print("R2 of", model_name, ":", (r2_score(y, ypred)).round(2))
    print("\n \n")

    return




## === cell 73
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
Yhat_lr = lr.predict(X_test)
model_train_evaluation(y_test, Yhat_lr, "Linear regression Model")



## === cell 74
test_pred = lr.predict(test_X)
Submission_lr = pd.DataFrame(test_pred, columns=["fare_amount"])
Submission_lr["key"] = df_test_full["key"].values
Submission_lr = Submission_lr[["key", "fare_amount"]]
Submission_lr.head()



## === cell 75
from sklearn.linear_model import Ridge

ridge = Ridge(random_state=SEED)
ridge.fit(X_train, y_train)
Yhat_ridge = ridge.predict(X_test)
model_train_evaluation(y_test, Yhat_ridge, "Ridge regression Model")



## === cell 76
test_pred = ridge.predict(test_X)
Submission_ridge = pd.DataFrame(test_pred, columns=["fare_amount"])
Submission_ridge["key"] = df_test_full["key"].values
Submission_ridge = Submission_ridge[["key", "fare_amount"]]
Submission_ridge.head()



## === cell 77
pass



## === cell 78
pass



## === cell 79
pass



## === cell 80
pass



## === cell 81
pass



## === cell 82
pass



## === cell 83
pass



## === cell 84
pass



## === cell 85
pass



## === cell 86
df_test_full["haversine_distance_log"] = np.log(
    df_test_full["haversine_distance"].values + 1
).astype("float32")
df_test_full["haversine_distance_sqrt"] = np.sqrt(
    df_test_full["haversine_distance"].values
).astype("float32")

df_train = df.drop(
    [
        "key",
        "pickup_datetime",
        "pickup_weekday",
        "fare",
        "fare_amount",
        "base_fare",
        "haversine_distance_sq",
        "pickup_hour",
    ],
    axis=1,
)

df_test_copy = df_test_full.drop(
    ["key", "base_fare", "pickup_datetime", "pickup_weekday", "pickup_hour"], axis=1
)

df_test_copy = df_test_copy.reindex(columns=df_train.columns)

X = df_train.copy()
y = df["fare"]

from sklearn import preprocessing

scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)
test_X = scaler.transform(df_test_copy)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=SEED
)



## === cell 87
final_model_name = None
try:
    from lightgbm import LGBMRegressor

    LGBM = LGBMRegressor(
        boosting_type="gbdt",
        num_leaves=31,
        learning_rate=0.1,
        max_depth=5,
        n_estimators=100,
        random_state=SEED,
        n_jobs=4,
    )
    LGBM.fit(X_train, y_train)
    Yhat_LGBM = LGBM.predict(X_test)
    model_train_evaluation(y_test, Yhat_LGBM, "LGBM Regression Model (fare)")
    final_model = LGBM
    final_model_name = "LGBMRegressor"
except Exception as e:
    print("LightGBM unavailable, using GradientBoostingRegressor instead:", repr(e))
    from sklearn.ensemble import GradientBoostingRegressor

    final_model = GradientBoostingRegressor(random_state=SEED)
    final_model.fit(X_train, y_train)
    Yhat_GB2 = final_model.predict(X_test)
    model_train_evaluation(y_test, Yhat_GB2, "GradientBoostingRegressor (fare)")
    final_model_name = "GradientBoostingRegressor"



## === cell 88
test_pred_fare = final_model.predict(test_X)
test_pred_fare = np.clip(test_pred_fare, 0.0, 300.0)

Submission = pd.DataFrame(test_pred_fare, columns=["fare"])
Submission["fare_amount"] = df_test_full["base_fare"].values + Submission["fare"].values
Submission["fare_amount"] = Submission["fare_amount"].clip(lower=0.0, upper=300.0)

Submission["key"] = df_test_full["key"].values
Submission = Submission[["key", "fare_amount"]]
Submission.head()



## === cell 89
Submission.to_csv("submission.csv", index=False)
print(
    f"Wrote final submission to submission.csv using {final_model_name}. Shape={Submission.shape}"
)
print(Submission.head())
print("Unique keys in submission:", Submission["key"].nunique())
