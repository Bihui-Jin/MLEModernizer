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

3.86972

# 6. Current score

5.8328

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.83403) has done: 'I fix the notebook-to-script issues so it runs in Kaggle’s Python environment (remove the Jupyter magic and correct the cell numbering). I also fix the XGBoost API breakage causing `best_ntree_limit` to be missing in xgboost 2.0.3 by using `iteration_range` (or falling back safely) so prediction works. To keep runtime under control and avoid huge slowdowns, I replace the row-wise `great_circle` loop with a vectorized haversine distance computation (same feature intent: trip distance), which is necessary for the code to finish end-to-end. Finally, I ensure a valid submission CSV with the required filename suffix and columns (`key,fare_amount`) is always written.'
- What this solution (achieved 5.76239) has done: 'Your current score (5.83403 RMSE) is worse than the target (3.86972), so we should cautiously improve performance while keeping your approach intact (same features, same XGBoost training loop, same loss/metric). The biggest “minimal-change” gain here is to make XGBoost actually learn the temporal + distance signal better by setting a few standard tree/learning parameters (still the same model class and training API) and using a larger, but still manageable, training sample to reduce variance. I also remove rounding during prediction (rounding increases RMSE) while keeping the exact same submission schema and clipping. Finally, I ensure the output file is a valid `.csv` with the required columns.'
- What this solution (achieved 5.8328) has done: 'To move RMSE down toward your target while keeping the same overall approach (same basic features + XGBoost regressor), I make two minimal, high-impact fixes: (1) remove the `early_stopping_rounds` usage (it’s effectively “relaxed convergence criteria” and can stop too early), and (2) correct the train/validation evaluation so RMSE is computed on a proper holdout split (and train on train split only). I also add one standard, competition-safe cleanup that stays within your feature intent: drop extreme `fare_amount` outliers (very large fares dominate RMSE and hurt generalization) without changing the model/feature set. Finally, I ensure the test set keeps all rows (don’t drop test rows with datetime parse issues) so the submission row count always matches `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import xgboost as xgb

import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)



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
train = pd.read_csv("../input/train.csv", nrows=2000000, dtype=types)



## === cell 6
train.head()



## === cell 7
train.describe()



## === cell 8
try:
    sns.histplot(train["fare_amount"], kde=True)
    plt.close()
except Exception:
    pass



## === cell 9
try:
    sns.histplot(train["passenger_count"], kde=False)
    plt.close()
except Exception:
    pass



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

train = train[train["fare_amount"] < 200].copy()



## === cell 13
train.describe()




## === cell 14
def add_haversine_distance_km(df: pd.DataFrame) -> pd.DataFrame:
    lat1 = np.deg2rad(df["pickup_latitude"].astype("float64").values)
    lon1 = np.deg2rad(df["pickup_longitude"].astype("float64").values)
    lat2 = np.deg2rad(df["dropoff_latitude"].astype("float64").values)
    lon2 = np.deg2rad(df["dropoff_longitude"].astype("float64").values)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(a))
    earth_radius_km = 6371.0088
    df["distance"] = (earth_radius_km * c).astype("float32")
    return df




## === cell 15
train = add_haversine_distance_km(train)
test = add_haversine_distance_km(test)



## === cell 16
train["pickup_datetime"] = (
    train["pickup_datetime"].astype(str).str.replace(" UTC", "", regex=False)
)
train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
)



## === cell 17
test["pickup_datetime"] = (
    test["pickup_datetime"].astype(str).str.replace(" UTC", "", regex=False)
)
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
)



## === cell 18
train = train.dropna(subset=["pickup_datetime"]).copy()

test["pickup_datetime"] = test["pickup_datetime"].fillna(
    pd.Timestamp("2010-01-01 00:00:00")
)

train["hour"] = train.pickup_datetime.dt.hour.astype("int16")
train["weekday"] = train.pickup_datetime.dt.weekday.astype("int16")
train["month"] = train.pickup_datetime.dt.month.astype("int16")
train["year"] = train.pickup_datetime.dt.year.astype("int16")

test["hour"] = test.pickup_datetime.dt.hour.astype("int16")
test["weekday"] = test.pickup_datetime.dt.weekday.astype("int16")
test["month"] = test.pickup_datetime.dt.month.astype("int16")
test["year"] = test.pickup_datetime.dt.year.astype("int16")



## === cell 19
test.head()



## === cell 20
try:
    plt.figure(figsize=(15, 8))
    sns.heatmap(
        train.drop(["key", "pickup_datetime"], axis=1).corr(numeric_only=True),
        annot=True,
        fmt=".4f",
    )
    plt.close()
except Exception:
    pass



## === cell 21
X = train.drop(
    [
        "key",
        "fare_amount",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
    ],
    axis=1,
)
y = train["fare_amount"]



## === cell 22
X.head()



## === cell 23
y.head()



## === cell 24
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)



## === cell 25
test_pred = test.drop(
    [
        "key",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
    ],
    axis=1,
)



## === cell 26
lm = LinearRegression()
lm.fit(X_train, y_train)
print(lm.score(X_train, y_train))
print(lm.score(X_valid, y_valid))



## === cell 27
y_valid_pred = lm.predict(X_valid)
lrmse = np.sqrt(metrics.mean_squared_error(y_valid_pred, y_valid))
print("Linear RMSE (fit on train split, eval on valid split):", lrmse)



## === cell 28
LinearPredictions = lm.predict(test_pred)



## === cell 29
LinearPredictions.size



## === cell 30
linear_submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": LinearPredictions},
    columns=["key", "fare_amount"],
)
linear_submission.head()




## === cell 31
def XGBoost(X_train, X_valid, y_train, y_valid):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.05,
        "max_depth": 8,
        "min_child_weight": 1.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "seed": RANDOM_STATE,
        "tree_method": "hist",
    }

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=2000,
        evals=[(dvalid, "valid")],
        verbose_eval=False,
    )
    return booster




## === cell 32
xgbm = XGBoost(X_train, X_valid, y_train, y_valid)

dvalid = xgb.DMatrix(X_valid)
valid_pred = xgbm.predict(dvalid)
vrmse = np.sqrt(metrics.mean_squared_error(valid_pred, y_valid))
print("XGB RMSE (valid):", vrmse)

dtest_pred = xgb.DMatrix(test_pred)
XGBPredictions = xgbm.predict(dtest_pred)



## === cell 33
XGBPredictions



## === cell 34
XGB_submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": XGBPredictions}, columns=["key", "fare_amount"]
)
XGB_submission.head()



## === cell 35
submission = XGB_submission

submission["fare_amount"] = submission["fare_amount"].clip(lower=0)

submission.to_csv("submission.csv", index=False)
print("Wrote submission:", "submission.csv", "rows:", len(submission))
print(submission.head())
