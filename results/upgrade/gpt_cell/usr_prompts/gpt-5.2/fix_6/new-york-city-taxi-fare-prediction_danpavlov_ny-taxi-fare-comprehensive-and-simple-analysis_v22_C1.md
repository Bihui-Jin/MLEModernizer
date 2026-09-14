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
geopy==2.4.1
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

3.4594

# 6. Current score

8.46216

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 9.21109) has done: 'Diagnosis: The crash happens in cell 38 because `xgb.train()` in XGBoost 2.0.3 returns a `Booster` that no longer exposes the legacy attribute `best_ntree_limit`. With early stopping, the correct way to limit trees is to use the best iteration via `iteration_range` (or omit tree limiting entirely), so accessing `xgbm.best_ntree_limit` raises `AttributeError`.  
Patch summary: Update cell 38 to compute predictions using the booster’s best iteration if available, falling back safely to a normal predict call when early stopping metadata isn’t present. This preserves the same training procedure and uses the best model selected by early stopping.  
Updated cells: Only cell 38 is modified.  
Compatibility notes for cell k+1: `XGBPredictions` remains defined as a NumPy array of predictions, so `cell 39` continues to work unchanged.  
Assumptions: XGBoost 2.0.3 supports `Booster.best_iteration` and `predict(iteration_range=...)`; if not present, the fallback `predict()` is acceptable and keeps execution unblocked.'
- What this solution (achieved 12.46206) has done: 'Your current score (9.21109 RMSE) is much worse than the target (3.4594), so we should improve generalization with the smallest changes that keep your core model/feature logic intact. The biggest issue hurting performance is that you compute `distance` using slow row-iteration and a slightly inconsistent radius, and you also round predictions to cents before scoring, which typically worsens RMSE. I replace the row-wise distance loop with a vectorized haversine computation (same “distance feature” concept, just computed correctly/consistently and much faster), keep the same train/test feature set, and remove rounding on predictions. I also set a fixed random seed in `train_test_split` for stability and update the XGBoost objective to the correct modern name (`reg:squarederror`) while keeping the same training procedure and early stopping behavior.'
- What this solution (achieved 5.7638) has done: 'Your RMSE (12.46206) is far worse than the target (3.4594), so we should make the smallest changes that improve generalization without changing the model/feature “shape.” The biggest single limiter is that you only train on 100k rows out of 55M, which strongly hurts accuracy; increasing the training sample (still using `nrows`, same pipeline) is a minimal, high-impact change that should move RMSE substantially toward the target. I also align train/test parsing by loading `test.csv` with the same numeric dtypes (avoids subtle type issues) and add a small, standard fare upper-bound filter to remove extreme outliers that distort RMSE, while keeping your existing filters/features/model intact. The script still runs end-to-end and writes a valid `XGBSubmission.csv` with the required columns.'
- What this solution (achieved 5.83229) has done: 'Your current RMSE (5.7638) is still materially worse than the target (3.4594), so we should improve accuracy with the smallest changes that keep your feature set and training flow intact. The biggest score limiter remaining is that the XGBoost model is training with essentially-default tree settings (depth/eta/subsampling), which underfits this problem even with 2M rows. I keep the same XGBoost training API, early stopping, and features, but add a small set of standard regression tree parameters (depth/eta/subsample/colsample/min_child_weight) and increase boosting rounds modestly so early stopping can still pick the best iteration. I also ensure XGBoost gets float32 feature matrices consistently (reduces subtle dtype issues and improves stability) while preserving identical semantics.'
- What this solution (achieved 8.46216) has done: 'Your current RMSE (5.83229) is worse than the target (3.4594), so we should improve generalization with minimal, score-relevant changes while keeping the same features and XGBoost training flow. The biggest remaining limiter is outlier/noise in the 2M-row sample (bad coordinates and “zero distance” rides), which can disproportionately hurt RMSE; we add a couple of standard NYC Taxi sanity filters that remove clearly-invalid trips but don’t change the model/feature design. We also add a tiny L2 regularization (`lambda`) and a modest `gamma` to reduce overfitting to noise without changing the core model architecture, and we evaluate on both train/test during early stopping to stabilize the chosen iteration. The script still run end-to-end and write `XGBSubmission.csv` with `key,fare_amount`.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from geopy.distance import great_circle  # calculate distances
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
types = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}
test = pd.read_csv(
    "../input/test.csv", dtype={k: v for k, v in types.items() if k != "fare_amount"}
)



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
train = pd.read_csv("../input/train.csv", nrows=2_000_000, dtype=types)



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
train = train[train["fare_amount"] < 250]

train = train[train["pickup_longitude"] < -72]
train = train[(train["pickup_latitude"] > 40) & (train["pickup_latitude"] < 44)]
train = train[train["dropoff_longitude"] < -72]
train = train[(train["dropoff_latitude"] > 40) & (train["dropoff_latitude"] < 44)]
train = train[(train["passenger_count"] > 0) & (train["passenger_count"] < 10)]



## === cell 13
train.describe()




## === cell 14
def dist_calc(df):
    for i, row in df.iterrows():
        df.at[i, "distance"] = great_circle(
            (row["pickup_latitude"], row["pickup_longitude"]),
            (row["dropoff_latitude"], row["dropoff_longitude"]),
        ).km




## === cell 15
def quick_dist_calc(df):
    """
    Change (score-relevant, minimal semantics change): compute the same "distance" feature
    but vectorized with the standard earth radius (6371 km). The previous row-loop is slow
    and used 6373, and per-row `df.at` can introduce dtype/object issues; this improves
    feature quality and consistency, helping RMSE while preserving core logic (distance feature).
    """
    R = 6371.0
    lat1 = np.radians(df["pickup_latitude"].astype("float64").to_numpy())
    lon1 = np.radians(df["pickup_longitude"].astype("float64").to_numpy())
    lat2 = np.radians(df["dropoff_latitude"].astype("float64").to_numpy())
    lon2 = np.radians(df["dropoff_longitude"].astype("float64").to_numpy())

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    df["distance"] = (R * c).astype("float32")




## === cell 16
quick_dist_calc(train)
quick_dist_calc(test)



## === cell 17
train = train[(train["distance"] > 0.0) & (train["distance"] < 200.0)]



## === cell 18
train["pickup_datetime"] = train["pickup_datetime"].str.replace(" UTC", "")
train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], format="%Y-%m-%d %H:%M:%S"
)

test["pickup_datetime"] = test["pickup_datetime"].str.replace(" UTC", "")
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], format="%Y-%m-%d %H:%M:%S"
)



## === cell 19
train["hour"] = train.pickup_datetime.dt.hour
train["weekday"] = train.pickup_datetime.dt.weekday
train["month"] = train.pickup_datetime.dt.month
train["year"] = train.pickup_datetime.dt.year

test["hour"] = test.pickup_datetime.dt.hour
test["weekday"] = test.pickup_datetime.dt.weekday
test["month"] = test.pickup_datetime.dt.month
test["year"] = test.pickup_datetime.dt.year




## === cell 20
def sphere_dist(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    R_earth = 6371
    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = map(
        np.radians, [pickup_lat, pickup_lon, dropoff_lat, dropoff_lon]
    )
    dlat = dropoff_lat - pickup_lat
    dlon = dropoff_lon - pickup_lon

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(pickup_lat) * np.cos(dropoff_lat) * np.sin(dlon / 2.0) ** 2
    )

    return 2 * R_earth * np.arcsin(np.sqrt(a))




## === cell 21
def add_airport_dist(dataset):
    jfk_coord = (40.639722, -73.778889)
    ewr_coord = (40.6925, -74.168611)
    lga_coord = (40.77725, -73.872611)

    pickup_lat = dataset["pickup_latitude"]
    dropoff_lat = dataset["dropoff_latitude"]
    pickup_lon = dataset["pickup_longitude"]
    dropoff_lon = dataset["dropoff_longitude"]

    pickup_jfk = sphere_dist(pickup_lat, pickup_lon, jfk_coord[0], jfk_coord[1])
    dropoff_jfk = sphere_dist(jfk_coord[0], jfk_coord[1], dropoff_lat, dropoff_lon)
    pickup_ewr = sphere_dist(pickup_lat, pickup_lon, ewr_coord[0], ewr_coord[1])
    dropoff_ewr = sphere_dist(ewr_coord[0], ewr_coord[1], dropoff_lat, dropoff_lon)
    pickup_lga = sphere_dist(pickup_lat, pickup_lon, lga_coord[0], lga_coord[1])
    dropoff_lga = sphere_dist(lga_coord[0], lga_coord[1], dropoff_lat, dropoff_lon)

    dataset["jfk_dist"] = pd.concat([pickup_jfk, dropoff_jfk], axis=1).min(axis=1)
    dataset["ewr_dist"] = pd.concat([pickup_ewr, dropoff_ewr], axis=1).min(axis=1)
    dataset["lga_dist"] = pd.concat([pickup_lga, dropoff_lga], axis=1).min(axis=1)

    return dataset




## === cell 22
train = add_airport_dist(train)
test = add_airport_dist(test)



## === cell 23
train.head()



## === cell 24
test.head()



## === cell 25
plt.figure(figsize=(15, 8))
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
LinearPredictions



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
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.05,
        "max_depth": 8,
        "min_child_weight": 1.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 1.0,
        "gamma": 0.1,
        "seed": 42,
    }

    return xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=2000,
        early_stopping_rounds=50,
        evals=[(dtrain, "train"), (dtest, "test")],
        verbose_eval=100,
    )




## === cell 38
xgbm = XGBoost(X_train, X_test, y_train, y_test)

test_pred_xgb = test_pred.astype("float32")
dtest_full = xgb.DMatrix(test_pred_xgb)
best_iter = getattr(xgbm, "best_iteration", None)

if best_iter is not None:
    XGBPredictions = xgbm.predict(dtest_full, iteration_range=(0, best_iter + 1))
else:
    XGBPredictions = xgbm.predict(dtest_full)



## === cell 39
XGBPredictions



## === cell 40
XGBPredictions



## === cell 41
XGB_submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": XGBPredictions}, columns=["key", "fare_amount"]
)
XGB_submission.head()



## === cell 42
submission = XGB_submission



## === cell 43
submission.to_csv("XGBSubmission.csv", index=False)
print(
    "Wrote submission:",
    os.path.abspath("XGBSubmission.csv"),
    "rows=",
    len(submission),
    "cols=",
    submission.columns.tolist(),
)
