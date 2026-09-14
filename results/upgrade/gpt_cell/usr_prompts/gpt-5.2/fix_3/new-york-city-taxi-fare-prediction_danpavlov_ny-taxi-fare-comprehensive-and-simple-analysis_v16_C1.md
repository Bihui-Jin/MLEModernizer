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

3.7

# 3. Installed packages

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

# 5. Target score

3.623

# 6. Current score

7.46225

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 8.49873) has done: 'Diagnosis: Cell 37 crashes because in xgboost==2.0.3 the returned `Booster` no longer exposes `best_ntree_limit` (it was available in older versions and depended on early stopping). The model is still trained with early stopping, but the correct way to select the best iteration is via `best_iteration`, and for prediction you should pass `iteration_range` instead of `ntree_limit`.  
Patch summary: Update the prediction call in cell 37 to use `iteration_range=(0, xgbm.best_iteration + 1)` when `best_iteration` exists; otherwise fall back to default `predict` (no limit). This preserves the same semantics (use best iteration from early stopping) while being compatible with newer XGBoost.  
Updated cells: Only cell 37 is changed.  
Compatibility notes for cell k+1: `XGBPredictions` remains a NumPy array of predictions with the same shape as before, so `cell 38` work unchanged.  
Assumptions: `xgb.train(...)` in this environment sets `xgbm.best_iteration` when early stopping is used; if not, the fallback still produce predictions without crashing.'
- What this solution (achieved 7.46225) has done: 'Your current score (8.49873 RMSE) is far worse than the target (3.623), so we should improve model quality with very small, low-risk changes that keep the same overall approach (feature engineering + XGBoost regression with early stopping). The biggest issue hurting score here is that XGBoost is being trained with the deprecated `'reg:linear'` objective (now treated differently) and with extremely weak/default tree parameters, so it underfits badly. I change the objective to `'reg:squarederror'` (same squared-error regression semantics) and set a small, standard set of regression tree parameters (depth/eta/subsample/colsample/min_child_weight) while keeping the same training loop and early stopping behavior. I also make the train/test split deterministic to stabilize the achieved score while keeping the same split strategy.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn import metrics  # evaluating models
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
)  # set splitting and validation
from sklearn.linear_model import LinearRegression
import xgboost as xgb  # XGBoost classifier
import matplotlib.pyplot as plt  # plotting
import seaborn as sns  # plotting
from math import sin, cos, sqrt, atan2, radians

try:
    get_ipython().run_line_magic("matplotlib", "inline")
except Exception:
    pass



## === cell 1
print(os.listdir("../input"))



## === cell 2
test = pd.read_csv("../input/test.csv")



## === cell 3
test.dtypes



## === cell 4
types = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}



## === cell 5
train = pd.read_csv("../input/train.csv", nrows=100000, dtype=types)



## === cell 6
train.head()



## === cell 7
train.describe()



## === cell 8
sns.distplot(train["fare_amount"])



## === cell 9
sns.distplot(train["passenger_count"])



## === cell 10
train.isnull().sum()



## === cell 11
train.dropna(inplace=True)



## === cell 12
train = train[train["fare_amount"] > 0]
train = train[train["pickup_longitude"] < -72]
train = train[(train["pickup_latitude"] > 40) & (train["pickup_latitude"] < 44)]
train = train[train["dropoff_longitude"] < -72]
train = train[(train["dropoff_latitude"] > 40) & (train["dropoff_latitude"] < 44)]
train = train[(train["passenger_count"] > 0) & (train["passenger_count"] < 10)]



## === cell 13
train.describe()




## === cell 14
def quick_dist_calc(df):
    R = 6373.0
    for i, row in df.iterrows():

        lat1 = radians(row["pickup_latitude"])
        lon1 = radians(row["pickup_longitude"])
        lat2 = radians(row["dropoff_latitude"])
        lon2 = radians(row["dropoff_longitude"])

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        distance = R * c
        df.at[i, "distance"] = distance




## === cell 15
def quick_dist_calc_loc(df, c1, c2, cname):
    R = 6373.0
    for i, row in df.iterrows():

        lat1 = radians(row["pickup_latitude"])
        lon1 = radians(row["pickup_longitude"])
        lat2 = radians(row["dropoff_latitude"])
        lon2 = radians(row["dropoff_longitude"])
        lat3 = radians(c1)
        lon3 = radians(c2)

        dlon1 = lon3 - lon1
        dlon2 = lon3 - lon2
        dlat1 = lat3 - lat1
        dlat2 = lat3 - lat2

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        ap = sin(dlat1 / 2) ** 2 + cos(lat3) * cos(lat1) * sin(dlon1 / 2) ** 2
        cp = 2 * atan2(sqrt(ap), sqrt(1 - ap))

        ad = sin(dlat2 / 2) ** 2 + cos(lat3) * cos(lat2) * sin(dlon2 / 2) ** 2
        cd = 2 * atan2(sqrt(ad), sqrt(1 - ad))

        distance_p = R * cp
        distance_d = R * cd

        df.at[i, cname + "_pickup_dist"] = distance_p
        df.at[i, cname + "_dropoff_dist"] = distance_d




## === cell 16
quick_dist_calc(train)
quick_dist_calc(test)



## === cell 17
jfk_airport = (-73.785193, 40.645972)
laguardia_airport = (-73.872925, 40.773335)
newark_airport = (-74.184156, 40.692764)
manhattan = (-73.983132, 40.759006)


quick_dist_calc_loc(train, jfk_airport[1], jfk_airport[0], "jfk_airport")
quick_dist_calc_loc(
    train, laguardia_airport[1], laguardia_airport[0], "laguardia_airport"
)
quick_dist_calc_loc(train, newark_airport[1], newark_airport[0], "newark_airport")
quick_dist_calc_loc(train, manhattan[1], manhattan[0], "manhattan")

quick_dist_calc_loc(test, jfk_airport[1], jfk_airport[0], "jfk_airport")
quick_dist_calc_loc(
    test, laguardia_airport[1], laguardia_airport[0], "laguardia_airport"
)
quick_dist_calc_loc(test, newark_airport[1], newark_airport[0], "newark_airport")
quick_dist_calc_loc(test, manhattan[1], manhattan[0], "manhattan")



## === cell 18
train["jfk_distance"] = pd.concat(
    [train["jfk_airport_pickup_dist"], train["jfk_airport_dropoff_dist"]], axis=1
).min(axis=1)
train["laguardia_distance"] = pd.concat(
    [train["laguardia_airport_pickup_dist"], train["laguardia_airport_dropoff_dist"]],
    axis=1,
).min(axis=1)
train["newark_distance"] = pd.concat(
    [train["newark_airport_pickup_dist"], train["newark_airport_dropoff_dist"]], axis=1
).min(axis=1)
train["manhattan_distance"] = pd.concat(
    [train["manhattan_pickup_dist"], train["manhattan_dropoff_dist"]], axis=1
).min(axis=1)

test["jfk_distance"] = pd.concat(
    [test["jfk_airport_pickup_dist"], test["jfk_airport_dropoff_dist"]], axis=1
).min(axis=1)
test["laguardia_distance"] = pd.concat(
    [test["laguardia_airport_pickup_dist"], test["laguardia_airport_dropoff_dist"]],
    axis=1,
).min(axis=1)
test["newark_distance"] = pd.concat(
    [test["newark_airport_pickup_dist"], test["newark_airport_dropoff_dist"]], axis=1
).min(axis=1)
test["manhattan_distance"] = pd.concat(
    [test["manhattan_pickup_dist"], test["manhattan_dropoff_dist"]], axis=1
).min(axis=1)



## === cell 19
train.drop("jfk_airport_pickup_dist", inplace=True, axis=1)
train.drop("jfk_airport_dropoff_dist", inplace=True, axis=1)
train.drop("laguardia_airport_pickup_dist", inplace=True, axis=1)
train.drop("laguardia_airport_dropoff_dist", inplace=True, axis=1)
train.drop("newark_airport_pickup_dist", inplace=True, axis=1)
train.drop("newark_airport_dropoff_dist", inplace=True, axis=1)
train.drop("manhattan_pickup_dist", inplace=True, axis=1)
train.drop("manhattan_dropoff_dist", inplace=True, axis=1)

test.drop("jfk_airport_pickup_dist", inplace=True, axis=1)
test.drop("jfk_airport_dropoff_dist", inplace=True, axis=1)
test.drop("laguardia_airport_pickup_dist", inplace=True, axis=1)
test.drop("laguardia_airport_dropoff_dist", inplace=True, axis=1)
test.drop("newark_airport_pickup_dist", inplace=True, axis=1)
test.drop("newark_airport_dropoff_dist", inplace=True, axis=1)
test.drop("manhattan_pickup_dist", inplace=True, axis=1)
test.drop("manhattan_dropoff_dist", inplace=True, axis=1)



## === cell 20
train["pickup_datetime"] = train["pickup_datetime"].str.replace(" UTC", "")
train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], format="%Y-%m-%d %H:%M:%S"
)

test["pickup_datetime"] = test["pickup_datetime"].str.replace(" UTC", "")
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], format="%Y-%m-%d %H:%M:%S"
)



## === cell 21
train["hour"] = train.pickup_datetime.dt.hour
train["weekday"] = train.pickup_datetime.dt.weekday
train["month"] = train.pickup_datetime.dt.month
train["year"] = train.pickup_datetime.dt.year

test["hour"] = test.pickup_datetime.dt.hour
test["weekday"] = test.pickup_datetime.dt.weekday
test["month"] = test.pickup_datetime.dt.month
test["year"] = test.pickup_datetime.dt.year



## === cell 22
train.head()



## === cell 23
test.head()



## === cell 24
plt.figure(figsize=(20, 12))
sns.heatmap(
    train.drop(["key", "pickup_datetime"], axis=1).corr(), annot=True, fmt=".4f"
)



## === cell 25
X = train.drop(["key", "fare_amount", "pickup_datetime"], axis=1)
y = train["fare_amount"]



## === cell 26
X.head()



## === cell 27
y.head()



## === cell 28
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



## === cell 29
test_pred = test.drop(["key", "pickup_datetime"], axis=1)



## === cell 30
lm = LinearRegression()
lm.fit(X_train, y_train)
print(lm.score(X_train, y_train))
print(lm.score(X_test, y_test))



## === cell 31
y_pred = lm.predict(X)
lrmse = np.sqrt(metrics.mean_squared_error(y_pred, y))
lrmse



## === cell 32
LinearPredictions = lm.predict(test_pred)
LinearPredictions = np.round(LinearPredictions, decimals=2)
LinearPredictions



## === cell 33
LinearPredictions.size



## === cell 34
linear_submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": LinearPredictions},
    columns=["key", "fare_amount"],
)



## === cell 35
linear_submission.head()




## === cell 36
def XGBoost(X_train, X_test, y_train, y_test):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.1,
        "max_depth": 8,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
    }

    return xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=400,
        early_stopping_rounds=30,
        evals=[(dtest, "test")],
    )




## === cell 37
xgbm = XGBoost(X_train, X_test, y_train, y_test)

dtest = xgb.DMatrix(test_pred)
if hasattr(xgbm, "best_iteration") and xgbm.best_iteration is not None:
    XGBPredictions = xgbm.predict(dtest, iteration_range=(0, xgbm.best_iteration + 1))
else:
    XGBPredictions = xgbm.predict(dtest)



## === cell 38
XGBPredictions



## === cell 39
XGBPredictions = np.round(XGBPredictions, decimals=2)
XGBPredictions



## === cell 40
XGB_submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": XGBPredictions}, columns=["key", "fare_amount"]
)
XGB_submission.head()



## === cell 41
submission = XGB_submission



## === cell 42
submission.to_csv("XGBSubmission23082018_2M.csv", index=False)
