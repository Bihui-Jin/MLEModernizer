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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn import metrics  # evaluating models
from sklearn.model_selection import train_test_split  # set splitting and validation
from sklearn.linear_model import LinearRegression
import xgboost as xgb  # XGBoost
import matplotlib.pyplot as plt  # plotting
import seaborn as sns  # plotting



## === cell 1
print(os.listdir("../input"))



## === cell 2
test_types = {
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "float32",  # read as float then clean/cast
}
test = pd.read_csv("../input/test.csv", dtype=test_types)



## === cell 3
test.dtypes



## === cell 4
types = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "float32",  # CHANGE: read as float then clean like test to reduce noise from bad types/values
}



## === cell 5
train = pd.read_csv("../input/train.csv", nrows=8_000_000, dtype=types)



## === cell 6
train.head()



## === cell 7
train.describe()



## === cell 8
sns.histplot(train["fare_amount"], kde=True)



## === cell 9
sns.histplot(train["passenger_count"], kde=False)



## === cell 10
train.isnull().sum()



## === cell 11
train.dropna(inplace=True)



## === cell 12
train["passenger_count"] = pd.to_numeric(
    train["passenger_count"], errors="coerce"
).fillna(1.0)
train["passenger_count"] = (
    train["passenger_count"].clip(lower=1, upper=9).astype("uint8")
)

train = train[train["fare_amount"] > 0]
train = train[train["pickup_longitude"] < -72]
train = train[(train["pickup_latitude"] > 40) & (train["pickup_latitude"] < 44)]
train = train[train["dropoff_longitude"] < -72]
train = train[(train["dropoff_latitude"] > 40) & (train["dropoff_latitude"] < 44)]
train = train[(train["passenger_count"] > 0) & (train["passenger_count"] < 10)]

train = train[train["fare_amount"] < 250.0]



## === cell 13
train.describe()




## === cell 14
def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6373.0

    lat1 = np.radians(np.asarray(lat1, dtype="float64"))
    lon1 = np.radians(np.asarray(lon1, dtype="float64"))
    lat2 = np.radians(np.asarray(lat2, dtype="float64"))
    lon2 = np.radians(np.asarray(lon2, dtype="float64"))

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


def quick_dist_calc(df):
    df["distance"] = _haversine_km(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    ).astype("float32")




## === cell 15
def quick_dist_calc_loc(df, c1, c2, cname):
    df[cname + "_pickup_dist"] = _haversine_km(
        df["pickup_latitude"], df["pickup_longitude"], c1, c2
    ).astype("float32")
    df[cname + "_dropoff_dist"] = _haversine_km(
        df["dropoff_latitude"], df["dropoff_longitude"], c1, c2
    ).astype("float32")




## === cell 16
quick_dist_calc(train)
quick_dist_calc(test)

train = train[(train["distance"] > 0.0) & (train["distance"] < 80.0)]



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
train.drop("jfk_airport_pickup_dist", inplace=True, axis=1, errors="ignore")
train.drop("jfk_airport_dropoff_dist", inplace=True, axis=1, errors="ignore")
train.drop("laguardia_airport_pickup_dist", inplace=True, axis=1, errors="ignore")
train.drop("laguardia_airport_dropoff_dist", inplace=True, axis=1, errors="ignore")
train.drop("newark_airport_pickup_dist", inplace=True, axis=1, errors="ignore")
train.drop("newark_airport_dropoff_dist", inplace=True, axis=1, errors="ignore")
train.drop("manhattan_pickup_dist", inplace=True, axis=1, errors="ignore")
train.drop("manhattan_dropoff_dist", inplace=True, axis=1, errors="ignore")

test.drop("jfk_airport_pickup_dist", inplace=True, axis=1, errors="ignore")
test.drop("jfk_airport_dropoff_dist", inplace=True, axis=1, errors="ignore")
test.drop("laguardia_airport_pickup_dist", inplace=True, axis=1, errors="ignore")
test.drop("laguardia_airport_dropoff_dist", inplace=True, axis=1, errors="ignore")
test.drop("newark_airport_pickup_dist", inplace=True, axis=1, errors="ignore")
test.drop("newark_airport_dropoff_dist", inplace=True, axis=1, errors="ignore")
test.drop("manhattan_pickup_dist", inplace=True, axis=1, errors="ignore")
test.drop("manhattan_dropoff_dist", inplace=True, axis=1, errors="ignore")



## === cell 20
train["pickup_datetime"] = train["pickup_datetime"].str.replace(" UTC", "", regex=False)
train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
)

test["pickup_datetime"] = test["pickup_datetime"].str.replace(" UTC", "", regex=False)
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
)

train.dropna(subset=["pickup_datetime"], inplace=True)
test.dropna(subset=["pickup_datetime"], inplace=True)



## === cell 21
train["hour"] = train.pickup_datetime.dt.hour
train["weekday"] = train.pickup_datetime.dt.weekday
train["month"] = train.pickup_datetime.dt.month
train["year"] = train.pickup_datetime.dt.year

test["hour"] = test.pickup_datetime.dt.hour
test["weekday"] = test.pickup_datetime.dt.weekday
test["month"] = test.pickup_datetime.dt.month
test["year"] = test.pickup_datetime.dt.year

test["passenger_count"] = pd.to_numeric(
    test["passenger_count"], errors="coerce"
).fillna(1.0)
test["passenger_count"] = test["passenger_count"].clip(lower=1, upper=9).astype("uint8")



## === cell 22
fare_per_km = train["fare_amount"] / np.clip(
    train["distance"].astype("float64"), 0.1, None
)
train = train[(train["fare_amount"] >= 2.5) | (train["distance"] >= 0.2)]
train = train[(fare_per_km > 0.2) & (fare_per_km < 80.0)]



## === cell 23
train.head()



## === cell 24
test.head()



## === cell 25
plt.figure(figsize=(20, 12))
sns.heatmap(
    train.drop(["key", "pickup_datetime"], axis=1).corr(), annot=True, fmt=".4f"
)



## === cell 26
X = train.drop(["key", "fare_amount", "pickup_datetime"], axis=1)
y = train["fare_amount"]



## === cell 27
X.head()



## === cell 28
y.head()



## === cell 29
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



## === cell 30
test_pred = test.drop(["key", "pickup_datetime"], axis=1)



## === cell 31
lm = LinearRegression()
lm.fit(X_train, y_train)
print(lm.score(X_train, y_train))
print(lm.score(X_test, y_test))



## === cell 32
y_pred = lm.predict(X)
lrmse = np.sqrt(metrics.mean_squared_error(y_pred, y))
lrmse



## === cell 33
LinearPredictions = lm.predict(test_pred)



## === cell 34
LinearPredictions.size



## === cell 35
linear_submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": LinearPredictions},
    columns=["key", "fare_amount"],
)



## === cell 36
linear_submission.head()




## === cell 37
def XGBoost(X_train, X_test, y_train, y_test):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_test, label=y_test)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.1,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1.0,
        "gamma": 0.0,
        "lambda": 1.0,
        "alpha": 0.0,
        "seed": 42,
        "tree_method": "hist",
    }

    return xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=2000,
        early_stopping_rounds=50,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        verbose_eval=False,
    )




## === cell 38
xgbm = XGBoost(X_train, X_test, y_train, y_test)



## === cell 39
dtest_submit = xgb.DMatrix(test_pred)

if getattr(xgbm, "best_iteration", None) is not None:
    XGBPredictions = xgbm.predict(
        dtest_submit, iteration_range=(0, xgbm.best_iteration + 1)
    )
else:
    XGBPredictions = xgbm.predict(dtest_submit)



## === cell 40
XGBPredictions



## === cell 41
XGBPredictions = np.clip(XGBPredictions, 0.0, None)

fallback_fare = float(np.median(y.values))

sample_sub = pd.read_csv("../input/sample_submission.csv")
submission = sample_sub.copy()

pred_map = pd.Series(XGBPredictions, index=test["key"])

submission["fare_amount"] = submission["key"].map(pred_map).astype("float64")
submission["fare_amount"].fillna(fallback_fare, inplace=True)
submission["fare_amount"] = submission["fare_amount"].clip(lower=0.0)

submission_path = "XGBSubmission23082018_2M.csv"
submission.to_csv(submission_path, index=False)
print("Wrote submission:", os.path.abspath(submission_path), "rows:", len(submission))
print("Any NaNs left:", int(submission["fare_amount"].isna().sum()))
