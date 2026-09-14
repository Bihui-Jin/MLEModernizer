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

# 5. Target score

4.04952

# 6. Current score

4.95629

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.34259) has done: 'The script is updated to fix the broken distance calculation (using `geodesic` instead of the removed `VincentyDistance`), remove the IPython magic line, correct the datetime week extraction, ensure only numeric columns are fed to LightGBM, and properly create the submission CSV after a successful train‑validation split and model fitting.'
- What this solution (achieved 5.35103) has done: 'I fix the LightGBM API misuse by replacing the removed `early_stopping_rounds` argument with the proper callback, import the needed callbacks, and adjust the re‑training step to use the discovered best iteration. These minimal changes resolve the fitting errors, allow validation metrics to be computed, and enable generation of a valid `submission.csv` while keeping the original modeling logic intact.'
- What this solution (achieved 5.36139) has done: 'I add a simple “weekend” binary feature to give the model more temporal information, include it in the training columns, and slightly increase the model capacity (more leaves and a deeper tree) so the LightGBM regressor can capture more patterns. These minimal adjustments keep the original workflow unchanged while aiming to lower the RMSE toward the target.'
- What this solution (achieved 5.71663) has done: 'I filter extreme trips (distance > 100 km or fare > 200 $) before training to reduce outlier noise, and slightly increase LightGBM capacity (more leaves and deeper trees) which should lower the RMSE toward the target while keeping the original workflow unchanged.'
- What this solution (achieved 5.01552) has done: 'I keep the overall LightGBM workflow but add a few lightweight features and train on the log‑transformed fare amount, which usually reduces RMSE for this dataset. The latitude/longitude columns are no longer dropped and a “direction” angle feature is added, giving the model more spatial information. The model is fitted on `log1p(fare_amount)` and predictions are back‑transformed with `expm1` before computing the final RMSE and creating the submission file. These minimal changes are expected to move the score closer to the target 4.04952 while preserving the core logic.'
- What this solution (achieved 5.0446) has done: 'I added lightweight cyclical encodings for hour, day‑of‑week and month, plus a squared‑distance feature, which give the model a bit more expressive power without changing its core logic. These new columns are created for both train and test, and the feature list is updated accordingly so LightGBM sees them during training and prediction. This should modestly lower the validation RMSE, moving the score closer to the target while keeping the original workflow intact.'
- What this solution (achieved 5.00727) has done: 'I slightly adjust the LightGBM hyper‑parameters to give the model more capacity while keeping the same training pipeline and log‑target transformation. A lower learning rate with more allowed trees (and a modest bagging fraction) usually improves validation RMSE without changing the core logic, moving the score closer to the target.'
- What this solution (achieved 4.95629) has done: 'I add four simple geographic delta features (lat/lon differences and their absolute values) to give the model more spatial information, and I slightly increase the LightGBM capacity (more trees, deeper trees, more leaves, lower learning rate) with a longer early‑stopping patience. These changes keep the original workflow intact while providing the model extra signals that should lower the validation RMSE and move the score closer to the target.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("fivethirtyeight")
import geopy.distance
import os

print(os.listdir("../input"))
import gc
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from xgboost import XGBRegressor
from sklearn import svm
from sklearn.linear_model import SGDRegressor
from sklearn import tree
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA




## === cell 1
def load_Data():
    train = pd.read_csv("../input/train.csv", nrows=10_00_000, low_memory=True)
    test = pd.read_csv("../input/test.csv", nrows=10_00_000, low_memory=True)
    return train, test




## === cell 2
train, test = load_Data()



## === cell 3
train.head(5)



## === cell 4
train.describe()



## === cell 5
train.info()



## === cell 6
train.isnull().sum()



## === cell 7
train = train.dropna(subset=["fare_amount"])
train = train[train["fare_amount"] > 0].reset_index(drop=True)
train = train.fillna(0)



## === cell 8
train["key2"] = pd.to_datetime(train["key"], errors="coerce")
train["key2"].head()
train.info()



## === cell 9
test["key2"] = pd.to_datetime(test["key"], errors="coerce")



## === cell 10
train["fare_amount"].plot(kind="box")



## === cell 11
gc.collect()
train.describe()



## === cell 12
print(
    "% of fares above 25$ - {:0.2f}".format(
        train[train["fare_amount"] > 25]["key"].count() * 100 / train["key"].count()
    )
)
print(
    "% of fares above 50$ - {:0.2f}".format(
        train[train["fare_amount"] > 50]["key"].count() * 100 / train["key"].count()
    )
)
print(
    "% of fares above 100$ - {:0.2f}".format(
        train[train["fare_amount"] > 100]["key"].count() * 100 / train["key"].count()
    )
)
print(
    "% of fares below 0$ - {:0.2f}".format(
        train[train["fare_amount"] < 0]["key"].count() * 100 / train["key"].count()
    )
)



## === cell 13
fig, axarr = plt.subplots(2, 2, figsize=(20, 10))
train[~(train["fare_amount"] > 25)]["fare_amount"].plot(kind="box", ax=axarr[0][0])
train[~(train["fare_amount"] > 50)]["fare_amount"].plot(kind="box", ax=axarr[0][1])
train[~(train["fare_amount"] > 100)]["fare_amount"].plot(kind="box", ax=axarr[1][0])
train[~(train["fare_amount"] < 0)]["fare_amount"].plot(kind="box", ax=axarr[1][1])



## === cell 14
train["passenger_count"].plot(kind="box")



## === cell 15
print(
    "Count of invalid pickup latitude",
    train[(train["pickup_latitude"] > 90) | (train["pickup_latitude"] < -90)][
        "pickup_latitude"
    ].count(),
)
print(
    "Count of invalid dropoff latitude",
    train[(train["dropoff_latitude"] > 90) | (train["dropoff_latitude"] < -90)][
        "dropoff_latitude"
    ].count(),
)
print(
    "Count of invalid pickup longitude",
    train[(train["pickup_longitude"] > 180) | (train["pickup_longitude"] < -180)][
        "pickup_longitude"
    ].count(),
)
print(
    "Count of invalid dropoff longitude",
    train[(train["dropoff_longitude"] > 180) | (train["dropoff_longitude"] < -180)][
        "dropoff_longitude"
    ].count(),
)



## === cell 16
print(
    "Count of invalid pickup latitude (test)",
    test[(test["pickup_latitude"] > 90) | (test["pickup_latitude"] < -90)][
        "pickup_latitude"
    ].count(),
)
print(
    "Count of invalid dropoff latitude (test)",
    test[(test["dropoff_latitude"] > 90) | (test["dropoff_latitude"] < -90)][
        "dropoff_latitude"
    ].count(),
)
print(
    "Count of invalid pickup longitude (test)",
    test[(test["pickup_longitude"] > 180) | (test["pickup_longitude"] < -180)][
        "pickup_longitude"
    ].count(),
)
print(
    "Count of invalid dropoff longitude (test)",
    test[(test["dropoff_longitude"] > 180) | (test["dropoff_longitude"] < -180)][
        "dropoff_longitude"
    ].count(),
)



## === cell 17
train = train[~((train["pickup_latitude"] > 90) | (train["pickup_latitude"] < -90))]
train = train[~((train["dropoff_latitude"] > 90) | (train["dropoff_latitude"] < -90))]
train = train[~((train["pickup_longitude"] > 180) | (train["pickup_longitude"] < -180))]
train = train[
    ~((train["dropoff_longitude"] > 180) | (train["dropoff_longitude"] < -180))
]



## === cell 18
train["distance"] = train[
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]
].apply(
    lambda x: geopy.distance.geodesic(
        (x["pickup_latitude"], x["pickup_longitude"]),
        (x["dropoff_latitude"], x["dropoff_longitude"]),
    ).km,
    axis=1,
)
train["direction"] = np.arctan2(
    train["dropoff_latitude"] - train["pickup_latitude"],
    train["dropoff_longitude"] - train["pickup_longitude"],
)
train["distance_squared"] = train["distance"] ** 2



## === cell 19
test["distance"] = test[
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]
].apply(
    lambda x: geopy.distance.geodesic(
        (x["pickup_latitude"], x["pickup_longitude"]),
        (x["dropoff_latitude"], x["dropoff_longitude"]),
    ).km,
    axis=1,
)
test["direction"] = np.arctan2(
    test["dropoff_latitude"] - test["pickup_latitude"],
    test["dropoff_longitude"] - test["pickup_longitude"],
)
test["distance_squared"] = test["distance"] ** 2



## === cell 20
print(
    "% of trips above 25 KM - {:0.2f}".format(
        train[train["distance"] > 25]["key"].count() * 100 / train.shape[0]
    )
)



## === cell 21
train["year"] = train["key2"].dt.year
train["month"] = train["key2"].dt.month
train["day"] = train["key2"].dt.day
train["day_of_week"] = train["key2"].dt.weekday
train["hour"] = train["key2"].dt.hour
train["week"] = train["key2"].dt.isocalendar().week
train["day_of_year"] = train["key2"].dt.dayofyear
train["week_of_year"] = train["key2"].dt.isocalendar().week
train["quarter"] = train["key2"].dt.quarter
train["is_weekend"] = train["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

train["hour_sin"] = np.sin(2 * np.pi * train["hour"] / 24)
train["hour_cos"] = np.cos(2 * np.pi * train["hour"] / 24)
train["dow_sin"] = np.sin(2 * np.pi * train["day_of_week"] / 7)
train["dow_cos"] = np.cos(2 * np.pi * train["day_of_week"] / 7)
train["month_sin"] = np.sin(2 * np.pi * train["month"] / 12)
train["month_cos"] = np.cos(2 * np.pi * train["month"] / 12)



## === cell 22
train.columns



## === cell 23
test["year"] = test["key2"].dt.year
test["month"] = test["key2"].dt.month
test["day"] = test["key2"].dt.day
test["day_of_week"] = test["key2"].dt.weekday
test["hour"] = test["key2"].dt.hour
test["week"] = test["key2"].dt.isocalendar().week
test["day_of_year"] = test["key2"].dt.dayofyear
test["week_of_year"] = test["key2"].dt.isocalendar().week
test["quarter"] = test["key2"].dt.quarter
test["is_weekend"] = test["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

test["hour_sin"] = np.sin(2 * np.pi * test["hour"] / 24)
test["hour_cos"] = np.cos(2 * np.pi * test["hour"] / 24)
test["dow_sin"] = np.sin(2 * np.pi * test["day_of_week"] / 7)
test["dow_cos"] = np.cos(2 * np.pi * test["day_of_week"] / 7)
test["month_sin"] = np.sin(2 * np.pi * test["month"] / 12)
test["month_cos"] = np.cos(2 * np.pi * test["month"] / 12)



## === cell 24
train["delta_lat"] = train["dropoff_latitude"] - train["pickup_latitude"]
train["delta_lon"] = train["dropoff_longitude"] - train["pickup_longitude"]
train["abs_delta_lat"] = train["delta_lat"].abs()
train["abs_delta_lon"] = train["delta_lon"].abs()

test["delta_lat"] = test["dropoff_latitude"] - test["pickup_latitude"]
test["delta_lon"] = test["dropoff_longitude"] - test["pickup_longitude"]
test["abs_delta_lat"] = test["delta_lat"].abs()
test["abs_delta_lon"] = test["delta_lon"].abs()

column_list = [
    "passenger_count",
    "distance",
    "distance_squared",
    "direction",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "year",
    "month",
    "day",
    "day_of_week",
    "hour",
    "week",
    "day_of_year",
    "week_of_year",
    "quarter",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "delta_lat",
    "delta_lon",
    "abs_delta_lat",
    "abs_delta_lon",
]
y_train_ = train["fare_amount"]
X_train_ = train[column_list]
X_test = test[column_list]



## === cell 25
mask = (train["distance"] <= 100) & (train["fare_amount"] <= 200)
X_train_ = X_train_[mask]
y_train_ = y_train_[mask]
print("After outlier filtering:", X_train_.shape, y_train_.shape)



## === cell 26
X_train, X_val, y_train, y_val = train_test_split(
    X_train_, y_train_, test_size=0.1, random_state=42
)



## === cell 27
X_train.shape, X_val.shape, y_train.shape, y_val.shape



## === cell 28
from sklearn.metrics import mean_squared_error



## === cell 29
y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)

lgb = LGBMRegressor(
    boosting_type="gbdt",
    colsample_bytree=0.9,
    learning_rate=0.015,  # smaller step for finer convergence
    max_depth=14,  # deeper trees
    min_child_samples=55,
    min_child_weight=0.001,
    min_split_gain=0.1,
    n_estimators=6000,  # more trees, early stopping will cut back
    n_jobs=-1,
    num_leaves=150,  # more leaves for higher expressivity
    reg_alpha=5.0,
    reg_lambda=3.0,
    subsample=0.8,
    subsample_for_bin=200000,
    subsample_freq=1,
    random_state=42,
    verbose=-1,
)

lgb.fit(
    X_train,
    y_train_log,
    eval_set=[(X_val, y_val_log)],
    callbacks=[
        early_stopping(stopping_rounds=50, verbose=False),
        log_evaluation(period=0),
    ],
)



## === cell 30
pred_val_log = lgb.predict(X_val)
pred_val = np.expm1(pred_val_log)

print("RMSE on validation:", np.sqrt(mean_squared_error(y_val, pred_val)))



## === cell 31
best_iter = lgb.best_iteration_ if lgb.best_iteration_ else lgb.n_estimators
lgb.set_params(n_estimators=best_iter)

y_train_log_full = np.log1p(y_train_)
lgb.fit(X_train_, y_train_log_full)



## === cell 32
pred_test_log = lgb.predict(X_test)
y_pred = np.expm1(pred_test_log)



## === cell 33
submission = pd.DataFrame({"key": test["key"], "fare_amount": y_pred})
submission.to_csv("submission.csv", index=False)
