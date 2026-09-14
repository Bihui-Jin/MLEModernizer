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

# 5. Target score

4.04343

# 6. Current score

5.51766

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.82845) has done: 'Diagnosis: The crash happens because `bayesian-optimization==3.1.0` exposes `xgb_bo.res` as a list of result dicts, not a dict keyed by `'max'`. Indexing it with a string therefore raises `TypeError: list indices must be integers or slices, not str`. The correct way to retrieve the best-found parameters in this version is via `xgb_bo.max['params']` (or by taking `max(xgb_bo.res, key=lambda r: r['target'])['params']`).  

Patch summary: Update cell 10 to fetch best parameters using the current BayesianOptimization API, while keeping the same downstream `params` dict semantics expected by `xgb.train`. Also keep the explicit `int()` cast for `max_depth` to preserve prior behavior.  

Updated cells: Only cell 10 is modified.  

Compatibility notes for cell k+1: Cell 12 expects a `params` dict suitable for `xgb.train`; the patch still produces that dict and preserves `params['max_depth']` as an integer. No variable names or shapes used by cell 12 are changed.  

Assumptions: `xgb_bo.max` is populated after `xgb_bo.maximize(...)` in cell 9 (as is standard for bayesian-optimization 3.x).'
- What this solution (achieved 5.51766) has done: 'You’re currently far worse than the target RMSE (5.83 vs 4.04), so the smallest safe way to move toward the target is to reduce variance and improve feature quality without changing the overall model/training approach. I (1) make the split deterministic, (2) ensure `xgb.cv` actually uses labels (your `X_val` DMatrix currently drops labels and the CV call doesn’t shuffle), (3) add standard NYC-taxi baseline features derived from the same raw columns (longitude/latitude deltas, Manhattan distance, and a simple “to/from airport” flag), and (4) constrain predictions to be positive and reasonable to avoid RMSE blow-ups from rare negatives/outliers. These are minimal additions around your existing feature engineering + XGBoost + Bayesian optimization + `xgb.train` pipeline and should move RMSE downward toward the target band.'

# 9. Code solution

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
    "../input/train.csv", nrows=50000, parse_dates=["pickup_datetime"]
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
mask &= df_train["passenger_count"].between(0, 8)
mask &= df_train["fare_amount"].between(0, 250)

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


def add_features(df):
    df = df.copy()
    df["distance_miles"] = distance(
        df.pickup_latitude,
        df.pickup_longitude,
        df.dropoff_latitude,
        df.dropoff_longitude,
    )
    df["year"] = df.pickup_datetime.dt.year
    df["hour"] = df.pickup_datetime.dt.hour

    df["abs_lon_diff"] = (df["pickup_longitude"] - df["dropoff_longitude"]).abs()
    df["abs_lat_diff"] = (df["pickup_latitude"] - df["dropoff_latitude"]).abs()
    df["manhattan_dist"] = df["abs_lon_diff"] + df["abs_lat_diff"]

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
    return df


df_train = add_features(df_train)
df_test = add_features(df_test)



## === cell 6
features = [
    "year",
    "hour",
    "distance_miles",
    "passenger_count",
    "abs_lon_diff",
    "abs_lat_diff",
    "manhattan_dist",
    "airport_trip",
]
X = df_train[features].values
y = df_train["fare_amount"].values
X_test = df_test[features]
df_test.head(5)



## === cell 7
import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split



## === cell 8
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train)


def xgb_eva(max_depth, gamma, colsample_bytree):
    params = {
        "eval_metric": "rmse",
        "max_depth": int(max_depth),
        "subsample": 0.8,
        "eta": 0.1,
        "gamma": gamma,
        "colsample_bytree": colsample_bytree,
        "objective": "reg:squarederror",
        "seed": 42,
    }
    cv_result = xgb.cv(
        params, dtrain, num_boost_round=100, nfold=3, shuffle=True, seed=42
    )
    return -1.0 * cv_result["test-rmse-mean"].iloc[-1]




## === cell 9
xgb_bo = BayesianOptimization(
    xgb_eva,
    {"max_depth": (3, 7), "gamma": (0, 1), "colsample_bytree": (0.3, 0.9)},
    random_state=42,
)
xgb_bo.maximize(init_points=3, n_iter=5)



## === cell 10
params = dict(xgb_bo.max["params"])
params["max_depth"] = int(params["max_depth"])

params.update(
    {
        "eval_metric": "rmse",
        "subsample": 0.8,
        "eta": 0.1,
        "objective": "reg:squarederror",
        "seed": 42,
    }
)



## === cell 11
model2 = xgb.train(params, xgb.DMatrix(X, label=y), num_boost_round=250)
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
