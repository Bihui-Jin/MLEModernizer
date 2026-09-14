# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

3.60104

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O
import os
from geopy.distance import great_circle
from sklearn import metrics, ensemble, linear_model
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

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
train = pd.read_csv("../input/train.csv", nrows=500000, dtype=types)




## === cell 6
train.head()




## === cell 7
train.describe()




## === cell 8
sns.histplot(train["fare_amount"], kde=True)




## === cell 9
sns.histplot(train["passenger_count"], kde=True)




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
def dist_calc(df):
    """Add a 'distance' column (km) using great‑circle distance."""
    for i, row in df.iterrows():
        df.at[i, "distance"] = great_circle(
            (row["pickup_latitude"], row["pickup_longitude"]),
            (row["dropoff_latitude"], row["dropoff_longitude"]),
        ).km




## === cell 15
dist_calc(train)
dist_calc(test)




## === cell 16
train["pickup_datetime"] = train["pickup_datetime"].str.replace(" UTC", "")
train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], format="%Y-%m-%d %H:%M:%S"
)




## === cell 17
test["pickup_datetime"] = test["pickup_datetime"].str.replace(" UTC", "")
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], format="%Y-%m-%d %H:%M:%S"
)




## === cell 18
train["hour"] = train.pickup_datetime.dt.hour
train["weekday"] = train.pickup_datetime.dt.weekday
train["month"] = train.pickup_datetime.dt.month
train["year"] = train.pickup_datetime.dt.year

test["hour"] = test.pickup_datetime.dt.hour
test["weekday"] = test.pickup_datetime.dt.weekday
test["month"] = test.pickup_datetime.dt.month
test["year"] = test.pickup_datetime.dt.year




## === cell 19
test.head()




## === cell 20
plt.figure(figsize=(15, 8))
sns.heatmap(
    train.drop(["key", "pickup_datetime"], axis=1).corr(), annot=True, fmt=".4f"
)




## === cell 21
X = train.drop(
    ["key", "fare_amount", "pickup_datetime", "pickup_longitude", "dropoff_longitude"],
    axis=1,
)
y = train["fare_amount"]




## === cell 22
X.head()




## === cell 23
y.head()




## === cell 24
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)




## === cell 25
test_pred = test.drop(
    ["key", "pickup_datetime", "pickup_longitude", "dropoff_longitude"], axis=1
)




## === cell 27
lm = LinearRegression()
lm.fit(X_train, y_train)
print("train R^2:", lm.score(X_train, y_train))
print("valid R^2:", lm.score(X_test, y_test))




## === cell 28
y_pred = lm.predict(X)
lrmse = np.sqrt(metrics.mean_squared_error(y_pred, y))
print("Linear model RMSE on training data:", lrmse)




## === cell 29
LinearPredictions = lm.predict(test_pred)
LinearPredictions = np.round(LinearPredictions, 2)




## === cell 30
print("Number of Linear predictions:", LinearPredictions.size)




## === cell 31
linear_submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": LinearPredictions},
    columns=["key", "fare_amount"],
)




## === cell 32
linear_submission.head()




## === cell 33
def XGBoost(X_train, X_test, y_train, y_test):
    """Train XGBoost with early stopping and return the Booster."""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_test, label=y_test)

    params = {"objective": "reg:squarederror", "eval_metric": "rmse", "seed": 42}

    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=300,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=20,
        verbose_eval=False,
    )
    return bst




## === cell 34
xgbm = XGBoost(X_train, X_test, y_train, y_test)

XGBPredictions = xgbm.predict(xgb.DMatrix(test_pred), ntree_limit=xgbm.best_iteration)
XGBPredictions = np.round(XGBPredictions, 2)




## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1250861719.py in <cell line: 0>()
      2 
      3 # Predict on the test set – using the best iteration found during training
----> 4 XGBPredictions = xgbm.predict(xgb.DMatrix(test_pred), ntree_limit=xgbm.best_iteration)
      5 XGBPredictions = np.round(XGBPredictions, 2)
      6 

TypeError: Booster.predict() got an unexpected keyword argument 'ntree_limit'

## === cell 35
print("Number of XGBoost predictions:", XGBPredictions.size)




## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1968355828.py in <cell line: 0>()
----> 1 print("Number of XGBoost predictions:", XGBPredictions.size)
      2 
      3 

NameError: name 'XGBPredictions' is not defined

## === cell 36
XGB_submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": XGBPredictions}, columns=["key", "fare_amount"]
)




## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/153447892.py in <cell line: 0>()
      1 XGB_submission = pd.DataFrame(
----> 2     {"key": test["key"], "fare_amount": XGBPredictions}, columns=["key", "fare_amount"]
      3 )
      4 
      5 

NameError: name 'XGBPredictions' is not defined

## === cell 37
submission = XGB_submission




## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2025662917.py in <cell line: 0>()
----> 1 submission = XGB_submission
      2 
      3 

NameError: name 'XGB_submission' is not defined

## === cell 38
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")

## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1674613990.py in <cell line: 0>()
----> 1 submission.to_csv("submission.csv", index=False)
      2 print("Submission saved to submission.csv")

NameError: name 'submission' is not defined
