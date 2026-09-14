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

No external packages required in the script and installed.

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

7.40075

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.92646) has done: 'The crash happens because newer Seaborn versions interpret the first positional argument to `countplot` as `data` (not `x`), so `sb.countplot(df['pickup_weekday'], order=[...])` is no longer treated as a pandas Series for reordering and triggers “Input data must be a pandas object to reorder”. The minimal fix is to pass the Series explicitly via the `x=` keyword for all `countplot` calls in this cell. This preserves the exact plotting intent and keeps all variables unchanged for the next cell.'
- What this solution (achieved 8.07061) has done: 'The crash happens because recent seaborn versions made `regplot` accept only keyword arguments for `x` and `y`, so passing them positionally (`sb.regplot(x, y, ...)`) raises a `TypeError`. The fix is to call `regplot` with explicit `x=` and `y=` keyword arguments while keeping the same data, regression line settings, and plot semantics. No other logic (stats calculation, labels, styling) is changed. This preserves the `ax` variable used immediately after for the legend.'
- What this solution (achieved 6.88386) has done: 'Your current score (8.07061 RMSE) is worse than the target (5.67556), so we should improve generalization with the smallest changes that don’t alter the modeling approach. The biggest leakage/quality issue is that `StandardScaler` is fit on the full dataset before `train_test_split`, which contaminates validation and typically hurts real test performance; we fit the scaler only on `X_train` and transform `X_test`/`test` with it. We also (1) make the train/test feature columns match exactly and in the same order, and (2) clip negative fare predictions to a small positive value (fares can’t be negative), which usually reduces RMSE without changing core model logic. Finally, we ensure the script reliably writes a valid `Submission.csv` using the final model’s predictions.'
- What this solution (achieved 7.40075) has done: 'The timeout is dominated by a few superlinear computations and repeated expensive model evaluations: the full pairwise haversine distance matrices (O(n²) on ~200k rows), many heavyweight plotting/EDA calls, VIF calculation, and repeated cross-validation for multiple models. To preserve the core modeling logic and final submission semantics, the optimized script removes only the non-essential EDA/diagnostic computations (plots, pairwise distance matrices, VIF/holoviews, repeated CV/model comparisons) while keeping the same feature engineering, scaling, train/valid split, and final model used to generate `Submission.csv`. It also replaces slow row-wise `.apply(baseFare)` with an exactly equivalent vectorized computation and avoids duplicate `.fit()` calls. These changes are provably equivalent for the final predictions (same data, same engineered features used in the final model, same training procedure for the selected final model), while cutting runtime to well under 600 seconds.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore")

import datetime as dt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso

from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

np.random.seed(42)



## === cell 1
train_path = "../input/new-york-city-taxi-fare-prediction/train.csv"
test_path = "../input/new-york-city-taxi-fare-prediction/test.csv"

dtypes_train = {
    "fare_amount": "float64",
    "pickup_longitude": "float64",
    "pickup_latitude": "float64",
    "dropoff_longitude": "float64",
    "dropoff_latitude": "float64",
    "passenger_count": "int64",
}
dtypes_test = {
    "pickup_longitude": "float64",
    "pickup_latitude": "float64",
    "dropoff_longitude": "float64",
    "dropoff_latitude": "float64",
    "passenger_count": "int64",
}

df = pd.read_csv(train_path, nrows=200000, dtype=dtypes_train)
df_test = pd.read_csv(test_path, dtype=dtypes_test)

df.head()



## === cell 2
df.shape



## === cell 3
df.info()



## === cell 4
df.describe()



## === cell 5
df.isnull().sum()



## === cell 6
df_test.isnull().sum()



## === cell 7
df.nunique()



## === cell 8
df.duplicated().sum()



## === cell 9
df.dropna(axis=0, inplace=True)
np.sum(pd.isnull(df))



## === cell 10
df.loc[df["fare_amount"] < 0, "fare_amount"] = 0.1
df[df["fare_amount"] < 0]



## === cell 11
df["pickup_datetime"] = pd.to_datetime(df.pickup_datetime)
df_test["pickup_datetime"] = pd.to_datetime(df_test.pickup_datetime)



## === cell 12
df.loc[:, "pickup_hour"] = df["pickup_datetime"].dt.hour
df.loc[:, "pickup_weekday"] = df["pickup_datetime"].dt.day_name()
df.loc[:, "pickup_date"] = df["pickup_datetime"].dt.day
df.loc[:, "pickup_month"] = df["pickup_datetime"].dt.month
df.loc[:, "pickup_day"] = df["pickup_datetime"].dt.dayofweek

df_test.loc[:, "pickup_hour"] = df_test["pickup_datetime"].dt.hour
df_test.loc[:, "pickup_weekday"] = df_test["pickup_datetime"].dt.day_name()
df_test.loc[:, "pickup_date"] = df_test["pickup_datetime"].dt.day
df_test.loc[:, "pickup_month"] = df_test["pickup_datetime"].dt.month
df_test.loc[:, "pickup_day"] = df_test["pickup_datetime"].dt.dayofweek



## === cell 13
h = df["pickup_hour"].to_numpy()
df["base_fare"] = np.where(
    (h >= 16) & (h < 20), 3.5, np.where((h >= 20) & (h < 24), 3.0, 2.5)
)

h_t = df_test["pickup_hour"].to_numpy()
df_test["base_fare"] = np.where(
    (h_t >= 16) & (h_t < 20), 3.5, np.where((h_t >= 20) & (h_t < 24), 3.0, 2.5)
)

df["base_fare"], df["pickup_hour"]



## === cell 14
df["fare"] = df["fare_amount"] - df["base_fare"]



## === cell 15
try:
    from geopy.distance import great_circle

    coordA = (df["pickup_latitude"].iloc[0], df["pickup_longitude"].iloc[0])
    coordB = (df["dropoff_latitude"].iloc[0], df["dropoff_longitude"].iloc[0])
    print(int(great_circle(coordA, coordB).kilometers))
except Exception as e:
    print("Skipping geopy distance demo:", repr(e))



## === cell 16
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




## === cell 17
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
)
df_test["haversine_distance"] = haversine_distance(
    df_test["pickup_latitude"].values,
    df_test["pickup_longitude"].values,
    df_test["dropoff_latitude"].values,
    df_test["dropoff_longitude"].values,
)



## === cell 18
df["haversine_distance"].median(), df["haversine_distance"].mean(),



## === cell 19
df.head()



## === cell 20
dist_km = None
df_dist_km = None
"skipped"



## === cell 21
result = None
"skipped"



## === cell 22
distance = None
"skipped"



## === cell 23
"skipped plotting"



## === cell 24
"skipped plotting"



## === cell 25
"skipped plotting"



## === cell 26
"skipped plotting"



## === cell 27
"skipped plotting"



## === cell 28
"skipped plotting"



## === cell 29
"skipped plotting"



## === cell 30
df["haversine_distance"].describe(), print(
    "Median       ", df["haversine_distance"].median()
)



## === cell 31
df["haversine_distance"].quantile(0.25), df["haversine_distance"].quantile(0.75)



## === cell 32
IQR = df["haversine_distance"].quantile(0.75) - df["haversine_distance"].quantile(0.25)
IQR



## === cell 33
Q1 = df["haversine_distance"].quantile(0.25)
Q3 = df["haversine_distance"].quantile(0.75)
whisker_1 = Q1 - (1.5 * IQR)
whisker_2 = Q3 + (1.5 * IQR)

whisker_1, whisker_2



## === cell 34
df = df.loc[(df["haversine_distance"] != 0) & (df["haversine_distance"] < 8)]
df.shape



## === cell 35
"skipped plotting"



## === cell 36
"skipped plotting"



## === cell 37
"skipped plotting"



## === cell 38
"skipped plotting"



## === cell 39
"skipped plotting"



## === cell 40
"skipped plotting"



## === cell 41
df = df.loc[(df.pickup_latitude > 40.6) & (df.pickup_latitude < 40.9)]
df = df.loc[(df.dropoff_latitude > 40.6) & (df.dropoff_latitude < 40.9)]
df = df.loc[(df.dropoff_longitude > -74.05) & (df.dropoff_longitude < -73.7)]
df = df.loc[(df.pickup_longitude > -74.05) & (df.pickup_longitude < -73.7)]
df = df.loc[(df["passenger_count"] >= 1) & (df["passenger_count"] <= 6)]
df = df.loc[(df["fare_amount"] >= 0.1) & (df["fare_amount"] <= 100.0)]

df_data_new = df.copy()
"filtered and skipped plotting"



## === cell 42
"skipped holoviews"



## === cell 43
"skipped VIF"



## === cell 44
"skipped plotting"



## === cell 45
df[
    ["pickup_latitude", "pickup_longitude", "dropoff_longitude", "dropoff_latitude"]
].corr()



## === cell 46
df.info()



## === cell 47
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



## === cell 48
from sklearn import preprocessing

X = preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]



## === cell 49
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
print(X_train.ndim)
print(y_train.ndim)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)



## === cell 50
from sklearn.metrics import mean_squared_error
from math import sqrt

mean_pred = np.repeat(y_train.mean(), len(y_test))
sqrt(mean_squared_error(y_test, mean_pred))



## === cell 51
"skipped pip install"



## === cell 52
"skipped CV"



## === cell 53
"skipped"



## === cell 54
"skipped"



## === cell 55
"skipped"



## === cell 56
"skipped"



## === cell 57
"skipped"



## === cell 58
"skipped"



## === cell 59
"skipped"



## === cell 60
"skipped"



## === cell 61
"skipped"



## === cell 62
"skipped"



## === cell 63
"skipped"



## === cell 64
"skipped"



## === cell 65
"skipped"



## === cell 66
df["haversine_distance_log"] = np.log(df["haversine_distance"].values + 1)
df["haversine_distance_sqrt"] = np.sqrt(df["haversine_distance"].values)
df["haversine_distance_sq"] = df["haversine_distance"].values ** 2

df_test["haversine_distance_log"] = np.log(df_test["haversine_distance"].values + 1)
df_test["haversine_distance_sqrt"] = np.sqrt(df_test["haversine_distance"].values)

"engineered distance transforms (skipped plotting)"



## === cell 67
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
df_test_copy = df_test.drop(
    [
        "key",
        "base_fare",
        "pickup_datetime",
        "pickup_weekday",
        "base_fare",
        "haversine_distance",
        "haversine_distance_sqrt",
    ],
    axis=1,
)
X = df_train.copy()
y = df["fare_amount"]
df_train.columns, df_test_copy.columns



## === cell 68
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

df_test_copy = df_test_copy.reindex(columns=df_train.columns)

X_train_df, X_valid_df, y_train, y_valid = train_test_split(
    df_train, y, test_size=0.3, random_state=42
)

scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train_df)
X_test = scaler.transform(X_valid_df)
test_X = scaler.transform(df_test_copy)

"skipped ridge CV coefficients"



## === cell 69
from sklearn.linear_model import RidgeCV

ridge = RidgeCV(cv=5).fit(X_train, y_train)
ridge.score(X_train, y_train)



## === cell 70
df_test.head()




## === cell 71
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




## === cell 72
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
Yhat_lr = lr.predict(X_test)
model_train_evaluation(y_valid, Yhat_lr, "Linear regression Model")



## === cell 73
test_pred = lr.predict(test_X)
test_pred = np.maximum(test_pred, 0.1)
Submission = pd.DataFrame(test_pred, columns=["fare_amount"])
Submission["key"] = df_test["key"]
Submission = Submission[["key", "fare_amount"]]
Submission.head()



## === cell 74
from sklearn.linear_model import Ridge

ridge = Ridge()
ridge.fit(X_train, y_train)
Yhat_ridge = ridge.predict(X_test)
model_train_evaluation(y_valid, Yhat_ridge, "Ridge regression Model")



## === cell 75
test_pred = ridge.predict(test_X)
test_pred = np.maximum(test_pred, 0.1)
Submission = pd.DataFrame(test_pred, columns=["fare_amount"])
Submission["key"] = df_test["key"]
Submission = Submission[["key", "fare_amount"]]
Submission.head()



## === cell 76
"skipped RandomForest training"



## === cell 77
"skipped RandomForest prediction"



## === cell 78
"skipped CatBoost"



## === cell 79
"skipped CatBoost prediction"



## === cell 80
"skipped SGD"



## === cell 81
"skipped SGD prediction"



## === cell 82
"skipped LightGBM"



## === cell 83
"skipped LightGBM prediction"



## === cell 84
from sklearn.ensemble import GradientBoostingRegressor

GB = GradientBoostingRegressor()
GB.fit(X_train, y_train)
Yhat_GB = GB.predict(X_test)
model_train_evaluation(y_valid, Yhat_GB, "Gradient Boosting Regression Model")



## === cell 85
test_pred = GB.predict(test_X)
test_pred = np.maximum(test_pred, 0.1)
Submission = pd.DataFrame(test_pred, columns=["fare_amount"])
Submission["key"] = df_test["key"]
Submission = Submission[["key", "fare_amount"]]
Submission.to_csv("Submission.csv", index=False)
Submission.head()



## === cell 86
df_test["haversine_distance_log"] = np.log(df_test["haversine_distance"].values + 1)
df_test["haversine_distance_sqrt"] = np.sqrt(df_test["haversine_distance"].values)

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
df_test_copy = df_test.drop(
    ["key", "base_fare", "pickup_datetime", "pickup_weekday", "pickup_hour"], axis=1
)
X = df_train.copy()
y = df["fare"]

df_test_copy = df_test_copy.reindex(columns=df_train.columns)

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

X_train_df, X_test_df, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train_df)
X_test = scaler.transform(X_test_df)
test_X = scaler.transform(df_test_copy)



## === cell 87
try:
    from lightgbm import LGBMRegressor

    LGBM = LGBMRegressor(
        boosting_type="gbdt",
        num_leaves=31,
        learning_rate=0.1,
        max_depth=5,
        n_estimators=100,
        silent=True,
    )
    LGBM.fit(X_train, y_train)
    Yhat_LGBM = LGBM.predict(X_test)
    model_train_evaluation(y_test, Yhat_LGBM, "LGBM Regression Model")
except Exception as e:
    print("Skipping LightGBM final model (not installed):", repr(e))



## === cell 88
try:
    test_pred = LGBM.predict(test_X)
except Exception:
    test_pred = GB.predict(test_X)

test_pred = np.maximum(test_pred, 0.0)
Submission = pd.DataFrame(test_pred, columns=["fare"])
Submission["fare_amount"] = df_test["base_fare"].values + Submission["fare"].values
Submission["fare_amount"] = np.maximum(Submission["fare_amount"].values, 0.1)
Submission["key"] = df_test["key"]
Submission = Submission[["key", "fare_amount"]]
Submission.to_csv("Submission.csv", index=False)
Submission.head()
