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

3.16533

# 6. Current score

5.06488

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.83281) has done: 'The script lacked a valid XGBoost prediction call (the old `best_ntree_limit` attribute no longer exists) and contained a Jupyter‑specific magic command that breaks in a plain Python run. I removed the `%matplotlib inline` line, adjusted the XGBoost helper to return the trained booster, and changed the prediction step to use the booster’s default predict method (which automatically uses the best iteration after early stopping). Finally, the submission dataframe is built and saved as a CSV file, guaranteeing a proper `.csv` output.'
- What this solution (achieved 6.30672) has done: 'I keep the overall workflow and model but improve the XGBoost part, which is the biggest driver of the score. By adding a modest set of well‑known hyper‑parameters (smaller learning rate, limited depth, subsampling) and allowing more boosting rounds with early stopping, the trained model should generalize better and reduce the validation RMSE, moving the Kaggle score toward the target 3.16533. I also add a quick validation RMSE print (doesn’t affect the submission) to confirm the improvement, while preserving every other step and ensuring the final CSV is written correctly.'
- What this solution (achieved 272.15392) has done: 'We keep the overall workflow unchanged but add a simple blend of the linear regression and XGBoost models, which usually reduces error without altering the core modeling logic. After training both models we compute their predictions on the validation split, average them, and report the blended RMSE. The same averaging is applied to the test‑set predictions, and the blended result is written out as the final submission file.'
- What this solution (achieved 269.57222) has done: 'Implemented fixes to eliminate NameErrors, ensure the XGBoost model is trained and used for predictions, and correctly generate a blended submission CSV. Added the missing model training call, defined `xgbm`, computed validation and test predictions, and kept the original linear regression blend. Updated comments for clarity and retained all original logic.'
- What this solution (achieved 4.96147) has done: 'Implemented a modest yet effective upgrade:
- Trains XGBoost on the log‑transformed fare (log1p) to better capture the skewed distribution and then back‑transforms predictions.
- Removes the linear‑model blend (which added unnecessary error) and uses the refined XGBoost predictions alone.
- Keeps the distance feature and other engineered time features, but drops the final rounding before evaluation (rounding is only applied when writing the CSV, preserving submission format).
- Updated validation and test prediction steps accordingly, ensuring a clean CSV output.'
- What this solution (achieved 5.06488) has done: 'I tighten the distance feature by clipping extreme values (which otherwise hurt the model) and slightly adjust the XGBoost hyper‑parameters to give the learner a bit more capacity while keeping the overall workflow unchanged. These minimal tweaks are expected to lower the validation RMSE, moving the score closer to the target 3.16533.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from geopy.distance import great_circle  # calculate distances
from sklearn import metrics  # evaluating models
from sklearn.model_selection import train_test_split  # set splitting and validation
from sklearn.linear_model import LinearRegression
import xgboost as xgb  # XGBoost
import matplotlib.pyplot as plt  # plotting
import seaborn as sns  # plotting



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
train = pd.read_csv("../input/train.csv", nrows=1000000, dtype=types)



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
def dist_calc(df):
    """
    Compute great‑circle distance (km) for each row.
    Afterwards we will clip extreme distances to reduce outlier impact.
    """
    for i, row in df.iterrows():
        df.at[i, "distance"] = great_circle(
            (row["pickup_latitude"], row["pickup_longitude"]),
            (row["dropoff_latitude"], row["dropoff_longitude"]),
        ).km




## === cell 15
dist_calc(train)
dist_calc(test)

train["distance"] = train["distance"].clip(upper=100)
test["distance"] = test["distance"].clip(upper=100)



## === cell 16
train["pickup_datetime"] = train["pickup_datetime"].str.replace(" UTC", "")
train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], format="%Y-%m-%d %H:%M:%S"
)
test["pickup_datetime"] = test["pickup_datetime"].str.replace(" UTC", "")
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], format="%Y-%m-%d %H:%M:%S"
)



## === cell 17
train["hour"] = train.pickup_datetime.dt.hour
train["weekday"] = train.pickup_datetime.dt.weekday
train["month"] = train.pickup_datetime.dt.month
train["year"] = train.pickup_datetime.dt.year

test["hour"] = test.pickup_datetime.dt.hour
test["weekday"] = test.pickup_datetime.dt.weekday
test["month"] = test.pickup_datetime.dt.month
test["year"] = test.pickup_datetime.dt.year



## === cell 18
test.head()



## === cell 19
plt.figure(figsize=(15, 8))
sns.heatmap(
    train.drop(["key", "pickup_datetime"], axis=1).corr(), annot=True, fmt=".4f"
)



## === cell 20
X = train.drop(["key", "fare_amount", "pickup_datetime"], axis=1)
y = train["fare_amount"]



## === cell 21
X.head()



## === cell 22
y.head()



## === cell 23
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



## === cell 24
test_pred = test.drop(["key", "pickup_datetime"], axis=1)



## === cell 25
lm = LinearRegression()
lm.fit(X_train, y_train)
print(lm.score(X_train, y_train))
print(lm.score(X_test, y_test))



## === cell 26
y_pred = lm.predict(X)
lrmse = np.sqrt(metrics.mean_squared_error(y_pred, y))
lrmse



## === cell 27
LinearPredictions = lm.predict(test_pred)
LinearPredictions = np.round(LinearPredictions, decimals=2)



## === cell 28
LinearPredictions.size



## === cell 29
linear_submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": LinearPredictions},
    columns=["key", "fare_amount"],
)



## === cell 30
linear_submission.head()




## === cell 31
def XGBoost_log_target(X_train, X_test, y_train, y_test):
    """
    Train XGBoost on log‑transformed fare_amount.
    Slightly tweaked hyper‑parameters (lower eta, deeper trees) to improve performance
    while preserving the original training logic.
    """
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    dtrain = xgb.DMatrix(X_train, label=y_train_log)
    dtest = xgb.DMatrix(X_test, label=y_test_log)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.03,  # smaller learning rate for smoother convergence
        "max_depth": 8,  # a bit deeper to capture interactions
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "lambda": 1,
        "seed": 42,
    }

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=2000,  # allow more rounds; early stopping will cap it
        early_stopping_rounds=30,
        evals=[(dtest, "eval")],
        verbose_eval=False,
    )
    return model




## === cell 32
xgbm = XGBoost_log_target(X_train, X_test, y_train, y_test)

val_pred_log = xgbm.predict(xgb.DMatrix(X_test))
val_pred = np.expm1(val_pred_log)

val_rmse = np.sqrt(metrics.mean_squared_error(y_test, val_pred))
print(f"Validation RMSE (XGBoost, log target): {val_rmse:.4f}")

XGBPredictions_log = xgbm.predict(xgb.DMatrix(test_pred))
XGBPredictions = np.expm1(XGBPredictions_log)



## === cell 33
XGBPredictions = np.round(XGBPredictions, decimals=2)



## === cell 34
submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": XGBPredictions},
    columns=["key", "fare_amount"],
)



## === cell 35
submission.to_csv("submission.csv", index=False)
