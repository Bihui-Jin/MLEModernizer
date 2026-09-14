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

11.15751

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.8036) has done: 'I fix the Bayesian Optimization API mismatch with the installed `bayesian-optimization==3.1.0` by removing the unsupported `acq` argument and correctly extracting the best parameters from `xgb_bo.max`. I also fix the XGBoost validation setup (the current `DMatrix` for validation is missing labels) and make the train/validation split deterministic to ensure stable execution. Finally, I ensure the test matrix uses the same feature columns and write a valid `submission.csv` with exactly `key` and `fare_amount`.'
- What this solution (achieved 5.66275) has done: 'You’re currently far above the target RMSE (5.8036 vs 4.04343; lower is better), so we should make a small, legitimate boost in model signal without changing the overall approach. The biggest gain with minimal disruption is to enrich the existing datetime features (still the same feature-extraction style) by adding standard components (month, day-of-week, day-of-month), and to use a slightly better-calibrated distance by also including absolute lat/lon deltas (still simple geometric features). To keep behavior stable and avoid accidental train/test mismatch, I compute datetime features vectorized for both train and test, ensure identical feature columns/order, and keep the same XGBoost training flow and Bayesian optimization loop. This should move RMSE down toward the target without altering the core model family or training semantics.'
- What this solution (achieved 6.46302) has done: 'We’re well above the target RMSE (5.66275 vs 4.04343; lower is better), so the smallest legitimate way to move toward the target is to (1) train on more signal (increase the training sample size without changing the approach), (2) add a couple of standard NYC-taxi geometric features (same “simple feature engineering” core logic), and (3) ensure the model uses the same boosting length found best during CV instead of a fixed 250 rounds. I keep the same XGBoost regressor + Bayesian optimization flow, same loss/metric semantics, and the same submission format. These changes typically reduce RMSE materially in this competition while staying within the same modeling approach and within runtime limits.'
- What this solution (achieved 11.15751) has done: 'You’re far above the target RMSE (6.463 vs 4.043; lower is better), so we need a legitimate accuracy boost while keeping the same XGBoost + simple feature engineering approach. The biggest minimal win here is to fix a feature bug: you currently include `year` (mostly constant) but not more informative datetime parts like `minute`, and you don’t use the common “airport/NYC center” location-distance features that fit the same feature-extraction style and often reduce RMSE substantially. I add a few lightweight, standard geographic features (pickup/dropoff to midtown + JFK/LGA/EWR distances) and `minute`, keep the same training/CV/BO flow, and ensure train/test feature alignment remains identical. I also increase the training sample moderately (still within the same approach) to reduce variance and improve generalization without changing the core model/training semantics.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import os

print(os.listdir("../input"))
os.chdir("/kaggle/working/")



## === cell 1
df_train = pd.read_csv(
    "../input/train.csv", nrows=800000, parse_dates=["pickup_datetime"]
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


NYC_CENTER = (40.7580, -73.9855)  # Times Square / Midtown
JFK = (40.6413, -73.7781)
LGA = (40.7769, -73.8740)
EWR = (40.6895, -74.1745)

for df in (df_train, df_test):
    df["distance_miles"] = distance(
        df.pickup_latitude,
        df.pickup_longitude,
        df.dropoff_latitude,
        df.dropoff_longitude,
    )
    df["abs_lon_diff"] = (df["pickup_longitude"] - df["dropoff_longitude"]).abs()
    df["abs_lat_diff"] = (df["pickup_latitude"] - df["dropoff_latitude"]).abs()

    df["manhattan_dist"] = df["abs_lon_diff"] + df["abs_lat_diff"]

    df["euclidean_deg"] = np.sqrt(df["abs_lon_diff"] ** 2 + df["abs_lat_diff"] ** 2)
    df["euclidean_miles_approx"] = np.sqrt(
        (df["abs_lat_diff"] * 69.0) ** 2 + (df["abs_lon_diff"] * 52.0) ** 2
    )

    df["pickup_to_center_miles"] = distance(
        df["pickup_latitude"], df["pickup_longitude"], NYC_CENTER[0], NYC_CENTER[1]
    )
    df["dropoff_to_center_miles"] = distance(
        df["dropoff_latitude"], df["dropoff_longitude"], NYC_CENTER[0], NYC_CENTER[1]
    )

    df["pickup_to_jfk_miles"] = distance(
        df["pickup_latitude"], df["pickup_longitude"], JFK[0], JFK[1]
    )
    df["dropoff_to_jfk_miles"] = distance(
        df["dropoff_latitude"], df["dropoff_longitude"], JFK[0], JFK[1]
    )

    df["pickup_to_lga_miles"] = distance(
        df["pickup_latitude"], df["pickup_longitude"], LGA[0], LGA[1]
    )
    df["dropoff_to_lga_miles"] = distance(
        df["dropoff_latitude"], df["dropoff_longitude"], LGA[0], LGA[1]
    )

    df["pickup_to_ewr_miles"] = distance(
        df["pickup_latitude"], df["pickup_longitude"], EWR[0], EWR[1]
    )
    df["dropoff_to_ewr_miles"] = distance(
        df["dropoff_latitude"], df["dropoff_longitude"], EWR[0], EWR[1]
    )

    dt = df["pickup_datetime"]
    df["year"] = dt.dt.year
    df["hour"] = dt.dt.hour
    df["minute"] = (
        dt.dt.minute
    )  # Change: add minute (informative, minimal, same semantics)
    df["month"] = dt.dt.month
    df["dayofweek"] = dt.dt.dayofweek
    df["day"] = dt.dt.day



## === cell 6
features = [
    "year",
    "month",
    "day",
    "dayofweek",
    "hour",
    "minute",
    "distance_miles",
    "abs_lat_diff",
    "abs_lon_diff",
    "manhattan_dist",
    "euclidean_deg",
    "euclidean_miles_approx",
    "pickup_to_center_miles",
    "dropoff_to_center_miles",
    "pickup_to_jfk_miles",
    "dropoff_to_jfk_miles",
    "pickup_to_lga_miles",
    "dropoff_to_lga_miles",
    "pickup_to_ewr_miles",
    "dropoff_to_ewr_miles",
    "passenger_count",
]
X = df_train[features].values
y = df_train["fare_amount"].values
X_test = df_test[features]
df_test.head(5)



## === cell 7
import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split



## === cell 8
X_train_np, X_val_np, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

dtrain = xgb.DMatrix(X_train_np, label=y_train)
dval = xgb.DMatrix(X_val_np, label=y_val)


def xgb_eva(max_depth, gamma, colsample_bytree, num_boost_round):
    params = {
        "eval_metric": "rmse",
        "max_depth": int(max_depth),
        "subsample": 0.8,
        "eta": 0.1,
        "gamma": float(gamma),
        "colsample_bytree": float(colsample_bytree),
        "objective": "reg:squarederror",
        "seed": 42,
    }
    nrounds = int(num_boost_round)
    nrounds = max(50, min(400, nrounds))
    cv_result = xgb.cv(
        params,
        dtrain,
        num_boost_round=nrounds,
        nfold=3,
        seed=42,
        shuffle=True,
        verbose_eval=False,
    )
    return -1.0 * cv_result["test-rmse-mean"].iloc[-1]




## === cell 9
xgb_bo = BayesianOptimization(
    f=xgb_eva,
    pbounds={
        "max_depth": (3, 7),
        "gamma": (0, 1),
        "colsample_bytree": (0.3, 0.9),
        "num_boost_round": (120, 380),
    },
    random_state=42,
    verbose=2,
)
xgb_bo.maximize(init_points=3, n_iter=5)



## === cell 10
best_params = dict(xgb_bo.max["params"])
best_params["max_depth"] = int(best_params["max_depth"])
best_num_boost_round = int(best_params.pop("num_boost_round"))
best_num_boost_round = max(50, min(400, best_num_boost_round))

params = {
    "eval_metric": "rmse",
    "max_depth": best_params["max_depth"],
    "subsample": 0.8,
    "eta": 0.1,
    "gamma": float(best_params["gamma"]),
    "colsample_bytree": float(best_params["colsample_bytree"]),
    "objective": "reg:squarederror",
    "seed": 42,
}

params



## === cell 11
model2 = xgb.train(
    params, xgb.DMatrix(X, label=y), num_boost_round=best_num_boost_round
)

X_testm = xgb.DMatrix(X_test.values)
y_test = model2.predict(X_testm)



## === cell 12
X_test.head(5)



## === cell 13
sub = pd.DataFrame()
sub["key"] = df_test["key"].values
sub["fare_amount"] = y_test
sub.to_csv("submission.csv", index=False)
sub
