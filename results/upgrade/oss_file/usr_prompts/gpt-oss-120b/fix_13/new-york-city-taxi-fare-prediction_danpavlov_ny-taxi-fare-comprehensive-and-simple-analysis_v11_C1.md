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

3.91427

# 6. Current score

inf

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.87797) has done: 'We fix the XGBoost prediction step which used the deprecated `ntree_limit` argument, causing a TypeError, and simplify the workflow so the model’s predictions are generated correctly. By removing the unsupported argument and directly calling `predict`, the script complete without errors and produce a valid CSV submission file. No other logic is changed, preserving the original model and feature engineering.'
- What this solution (achieved 5.83638) has done: 'I add a log‑transform of the target variable for the XGBoost model and convert the predictions back with exp‑1. This small change often reduces RMSE on skewed fare data without altering the overall workflow or model architecture. I also pass the transformed targets to the training function and apply the inverse transform after prediction, keeping everything else unchanged.'
- What this solution (achieved 5.83753) has done: 'I keep the same overall workflow but adjust the XGBoost parameters (increase tree depth) and retrain the model on the full training set using the best number of boosting rounds found during early‑stopping. I also remove the unnecessary rounding of predictions and clip any negative values, which helps lower the RMSE toward the target without altering the core logic of the solution.'
- What this solution (achieved 5.83663) has done: 'I increase the amount of training data used (from 500 k to 2 M rows) and slightly strengthen the XGBoost model by using a deeper tree and a smaller learning rate, which should reduce the RMSE and move the score closer to the target while keeping the overall workflow unchanged.'
- What this solution (achieved 180.69316) has done: 'I add a day‑of‑week feature, keep the latitude/longitude coordinates as model inputs, and make the XGBoost and linear regression predictions blend together (60 % XGB, 40 % linear). These small, targeted changes keep the original workflow intact while providing richer features and a modest ensemble that should pull the RMSE nearer the target value.'
- What this solution (achieved inf) has done: 'I adjust the linear regression to predict the log‑transformed fare (consistent with the XGBoost model) and then inverse‑transform its predictions, removing the unnecessary rounding. I also make the blending rely more on the stronger XGBoost predictions (80 % XGB + 20 % linear). These minimal tweaks keep the overall pipeline unchanged while better aligning the models with the RMSE metric, moving the score toward the target.'
- What this solution (achieved inf) has done: 'I make the prediction step robust by ensuring a valid number of boosting rounds is always used (fallback to 500 if early‑stopping does not provide one) and by sanitising any NaN/inf values that might appear after XGBoost prediction. This prevents the “inf” RMSE and yields a finite submission that moves the score toward the target while keeping the original modelling pipeline unchanged.'
- What this solution (achieved inf) has done: 'I fill any missing values in the test features before predicting, preventing NaNs from propagating into the linear model and causing infinite scores. I also add a quick validation step that blends the XGBoost and linear predictions on the held‑out split and prints the resulting RMSE, so we can see the score moving from “inf” toward the target. No core model logic is changed.'
- What this solution (achieved inf) has done: 'I fix the validation step that produces infinite RMSE by ensuring there are no NaN values in the features fed to the XGBoost model. The linear‑regression part already fills missing values, but the XGBoost prediction on `X_test` can still contain NaNs, leading to NaN predictions and an infinite error. I fill missing values in `X_test` before creating the DMatrix, then compute the blended validation predictions as before. This small change keeps all core logic intact and turns the “inf” score into a finite RMSE, moving the result toward the target.'
- What this solution (achieved inf) has done: 'I increase the XGBoost tree depth slightly (from 10 to 12) to give the model more capacity, and I give the XGBoost predictions a higher weight in the blended final prediction (90 % XGB + 10 % linear). These minimal adjustments keep the original pipeline intact while expectedly lowering the RMSE toward the target.'
- What this solution (achieved inf) has done: 'The update adds robust handling for possible NaN/inf values in the XGBoost validation predictions, ensures the early‑stopping iteration count is never zero, and introduces a configurable blending weight (set to 0.8) for combining XGBoost and linear predictions. These minimal changes keep the original model and feature engineering intact while preventing the “inf” RMSE and moving the score toward the target.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from geopy.distance import great_circle
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns



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
def dist_calc(df):
    coords_start = list(zip(df["pickup_latitude"], df["pickup_longitude"]))
    coords_end = list(zip(df["dropoff_latitude"], df["dropoff_longitude"]))
    distances = [great_circle(s, e).km for s, e in zip(coords_start, coords_end)]
    df["distance"] = distances  # modify in‑place




## === cell 14
dist_calc(train)
dist_calc(test)



## === cell 15
train["pickup_datetime"] = train["pickup_datetime"].str.replace(" UTC", "")
train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], format="%Y-%m-%d %H:%M:%S"
)



## === cell 16
test["pickup_datetime"] = test["pickup_datetime"].str.replace(" UTC", "")
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], format="%Y-%m-%d %H:%M:%S"
)



## === cell 17
train["year"] = train.pickup_datetime.dt.year
train["hour"] = train.pickup_datetime.dt.hour
train["dow"] = train.pickup_datetime.dt.dayofweek  # 0=Mon ... 6=Sun
test["year"] = test.pickup_datetime.dt.year
test["hour"] = test.pickup_datetime.dt.hour
test["dow"] = test.pickup_datetime.dt.dayofweek



## === cell 18
test.head()



## === cell 19
plt.figure(figsize=(15, 8))
sns.heatmap(
    train.drop(["key", "pickup_datetime"], axis=1).corr(), annot=True, fmt=".4f"
)



## === cell 20
X = train.drop(
    [
        "key",
        "fare_amount",
        "pickup_datetime",
    ],
    axis=1,
)
y = train["fare_amount"]



## === cell 21
X.head()



## === cell 22
y.head()



## === cell 23
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)



## === cell 24
test_pred = test.drop(
    [
        "key",
        "pickup_datetime",
    ],
    axis=1,
)
test_pred = test_pred.fillna(0)



## === cell 25
lm = LinearRegression()
lm.fit(X_train, y_train_log)
print("Linear model R^2 (log target):", lm.score(X_train, y_train_log))



## === cell 26
lin_log_pred_full = lm.predict(X)
LinearPredictions_full = np.expm1(lin_log_pred_full)
lrmse_full = np.sqrt(metrics.mean_squared_error(y, LinearPredictions_full))
print("Linear model RMSE on training data (after exp):", lrmse_full)



## === cell 27
LinearPredictions = np.expm1(lm.predict(test_pred))



## === cell 28
linear_submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": LinearPredictions},
    columns=["key", "fare_amount"],
)
linear_submission.head()




## === cell 29
def XGBoost(X_train, X_test, y_train_log, y_test_log):
    dtrain = xgb.DMatrix(X_train, label=y_train_log)
    dtest = xgb.DMatrix(X_test, label=y_test_log)
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.03,
        "max_depth": 12,  # increased depth for slightly better capacity
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=1000,
        early_stopping_rounds=30,
        evals=[(dtest, "test")],
        verbose_eval=False,
    )
    return model, params




## === cell 30
xgbm, params = XGBoost(X_train, X_test, y_train_log, y_test_log)

best_iter = getattr(xgbm, "best_iteration", None)
if best_iter is None or best_iter <= 0:
    best_iter = 500

BLEND_WEIGHT = 0.8



## === cell 31
dtrain_full = xgb.DMatrix(X, label=np.log1p(y))
final_model = xgb.train(
    params=params,
    dtrain=dtrain_full,
    num_boost_round=best_iter,
    verbose_eval=False,
)
XGBPredictions_log = final_model.predict(xgb.DMatrix(test_pred))

XGBPredictions = np.expm1(XGBPredictions_log)
XGBPredictions = np.nan_to_num(XGBPredictions, nan=0.0, posinf=0.0, neginf=0.0)
XGBPredictions = np.clip(XGBPredictions, a_min=0, a_max=None)



## === cell 32
blended_pred = BLEND_WEIGHT * XGBPredictions + (1 - BLEND_WEIGHT) * LinearPredictions
blended_pred = np.clip(blended_pred, a_min=0, a_max=None)
XGB_submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": blended_pred},
    columns=["key", "fare_amount"],
)
XGB_submission.head()



## === cell 33
X_test_filled = X_test.fillna(0)

X_test_pred = X_test_filled.copy()
lin_val_log = lm.predict(X_test_pred)
lin_val = np.expm1(lin_val_log)
lin_val = np.nan_to_num(lin_val, nan=0.0, posinf=0.0, neginf=0.0)

xgb_val_log = xgbm.predict(xgb.DMatrix(X_test_filled))
xgb_val = np.expm1(xgb_val_log)
xgb_val = np.nan_to_num(xgb_val, nan=0.0, posinf=0.0, neginf=0.0)

blended_val = BLEND_WEIGHT * xgb_val + (1 - BLEND_WEIGHT) * lin_val
blended_val = np.clip(blended_val, a_min=0, a_max=None)
val_rmse = np.sqrt(metrics.mean_squared_error(y_test, blended_val))
print("Validation RMSE (blended model):", val_rmse)



## === cell 34
submission = XGB_submission



## === cell 35
submission.to_csv("submission.csv", index=False)
