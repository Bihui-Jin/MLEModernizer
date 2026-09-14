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

3.60104

# 6. Current score

1.2720133588756004e+97

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.37707) has done: 'I fix the XGBoost prediction call that uses the removed `ntree_limit` argument, replacing it with the supported `iteration_range` parameter. This eliminates the TypeError, allowing XGBoost predictions to be generated and the submission DataFrame to be created and saved correctly.'
- What this solution (achieved 5.4899) has done: 'I add two modest improvements that keep the original modeling pipeline intact while nudging the RMSE toward the target. First, after computing the great‑circle distance I create two simple coordinate‑difference features (`delta_lat` and `delta_long`) for both train and test; these often help tree models capture directionality of trips. Second, I slightly tune the XGBoost parameters (lower learning rate, deeper trees, subsampling) which usually yields a steadier fit without changing the overall model structure. These changes are minimal, respect the existing code flow, and are expected to lower the validation RMSE, moving the score closer to the target.'
- What this solution (achieved 5.05405) has done: 'I keep the overall pipeline unchanged but train the XGBoost model on the log‑transformed fare amount (using np.log1p) and then exponentiate the predictions back to the original scale. This often reduces skew‑related error and moves the RMSE closer to the target. I also slightly tighten the tree depth and learning rate for a more stable fit while preserving the original feature set.'
- What this solution (achieved 4.78511) has done: 'The update adds a log‑scaled distance feature and removes extreme trips (distance > 100 km) to reduce outlier noise, then tightens the XGBoost hyper‑parameters (shallower trees, higher learning rate, modest L2 regularisation) so the model fits more robustly on the log‑transformed target. These focused tweaks keep the original pipeline intact while aiming to lower the RMSE toward the target score.'
- What this solution (achieved 441.75825) has done: 'I add a lightweight validation check and blend the XGBoost and linear‑regression predictions (simple averaging).  This keeps the original model pipeline intact, adds only a few lines, and typically lowers RMSE by reducing individual model errors, moving the score closer to the target.  The blended predictions are then used for the final submission file.'
- What this solution (achieved 1.2720133588756004e+97) has done: 'The changes train the linear model on the log‑scaled fare (matching the XGBoost target), exponentiate its predictions back to the original scale, and clip any negative values. This alignment reduces skew‑related error and makes the blended ensemble more consistent, moving the validation RMSE closer to the target while keeping the original pipeline untouched.'
- What this solution (achieved 1.2720133588756004e+97) has done: 'I add a safeguard that caps any extreme values after converting the log‑predictions back to the original scale. By replacing infinities (or overly large numbers) with a reasonable maximum fare (e.g., $500) before clipping and rounding, the model’s predictions become finite, which dramatically reduces the RMSE from the astronomically large value and moves the score toward the target while keeping the original pipeline intact.'

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
train["log_distance"] = np.log1p(train["distance"])
test["log_distance"] = np.log1p(test["distance"])
train = train[train["distance"] <= 100]




## === cell 17
train["pickup_datetime"] = train["pickup_datetime"].str.replace(" UTC", "")
train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], format="%Y-%m-%d %H:%M:%S"
)




## === cell 18
test["pickup_datetime"] = test["pickup_datetime"].str.replace(" UTC", "")
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], format="%Y-%m-%d %H:%M:%S"
)




## === cell 19
train["delta_lat"] = train["dropoff_latitude"] - train["pickup_latitude"]
train["delta_long"] = train["dropoff_longitude"] - train["pickup_longitude"]
test["delta_lat"] = test["dropoff_latitude"] - test["pickup_latitude"]
test["delta_long"] = test["dropoff_longitude"] - test["pickup_longitude"]




## === cell 20
train["hour"] = train.pickup_datetime.dt.hour
train["weekday"] = train.pickup_datetime.dt.weekday
train["month"] = train.pickup_datetime.dt.month
train["year"] = train.pickup_datetime.dt.year

test["hour"] = test.pickup_datetime.dt.hour
test["weekday"] = test.pickup_datetime.dt.weekday
test["month"] = test.pickup_datetime.dt.month
test["year"] = test.pickup_datetime.dt.year




## === cell 21
test.head()




## === cell 22
plt.figure(figsize=(15, 8))
sns.heatmap(
    train.drop(["key", "pickup_datetime"], axis=1).corr(), annot=True, fmt=".4f"
)




## === cell 23
X = train.drop(
    ["key", "fare_amount", "pickup_datetime", "pickup_longitude", "dropoff_longitude"],
    axis=1,
)
y = train["fare_amount"]




## === cell 24
X.head()




## === cell 25
y.head()




## === cell 26
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)




## === cell 27
test_pred = test.drop(
    ["key", "pickup_datetime", "pickup_longitude", "dropoff_longitude"], axis=1
)




## === cell 28
lm_log = LinearRegression()
lm_log.fit(X_train, y_train_log)
print("Linear (log) train R^2:", lm_log.score(X_train, y_train_log))
print("Linear (log) valid R^2:", lm_log.score(X_test, y_test_log))




## === cell 29
lin_log_pred_train = np.expm1(lm_log.predict(X_train))
lin_log_rmse_train = np.sqrt(mean_squared_error(y_train, lin_log_pred_train))
print("Linear (log) RMSE on training data:", lin_log_rmse_train)




## === cell 30
LinearPredictions = np.expm1(lm_log.predict(test_pred))
LinearPredictions = np.clip(LinearPredictions, 0, None)  # fares cannot be negative
LinearPredictions = np.round(LinearPredictions, 2)




## === cell 31
print("Number of Linear predictions:", LinearPredictions.size)




## === cell 32
linear_submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": LinearPredictions},
    columns=["key", "fare_amount"],
)




## === cell 33
linear_submission.head()




## === cell 34
def XGBoostLog(X_train, X_test, y_train_log, y_test_log):
    """Train XGBoost on log‑transformed target and return the Booster."""
    dtrain = xgb.DMatrix(X_train, label=y_train_log)
    dvalid = xgb.DMatrix(X_test, label=y_test_log)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "seed": 42,
        "learning_rate": 0.05,
        "max_depth": 7,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
    }

    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=1200,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    return bst




## === cell 35
xgbm = XGBoostLog(X_train, X_test, y_train_log, y_test_log)

best_iter = getattr(xgbm, "best_iteration", None)

if best_iter is not None:
    preds_log_test = xgbm.predict(
        xgb.DMatrix(test_pred), iteration_range=(0, best_iter + 1)
    )
else:
    preds_log_test = xgbm.predict(xgb.DMatrix(test_pred))

XGBPred_test = np.expm1(preds_log_test)
XGBPred_test = np.nan_to_num(XGBPred_test, nan=0.0, posinf=500.0, neginf=0.0)
XGBPred_test = np.clip(XGBPred_test, 0, 500)
XGBPred_test = np.round(XGBPred_test, 2)

if best_iter is not None:
    preds_log_valid = xgbm.predict(
        xgb.DMatrix(X_test), iteration_range=(0, best_iter + 1)
    )
else:
    preds_log_valid = xgbm.predict(xgb.DMatrix(X_test))

XGBPred_valid = np.expm1(preds_log_valid)
XGBPred_valid = np.nan_to_num(XGBPred_valid, nan=0.0, posinf=500.0, neginf=0.0)
XGBPred_valid = np.clip(XGBPred_valid, 0, 500)

LinearPred_valid = np.expm1(lm_log.predict(X_test))
LinearPred_valid = np.clip(LinearPred_valid, 0, 500)

Ensemble_valid = (XGBPred_valid + LinearPred_valid) / 2.0
valid_rmse = np.sqrt(mean_squared_error(y_test, Ensemble_valid))
print("Ensemble validation RMSE (train‑split):", valid_rmse)

LinearPred_test = LinearPredictions  # already rounded & clipped
Ensemble_test = (XGBPred_test + LinearPred_test) / 2.0
Ensemble_test = np.round(Ensemble_test, 2)

XGBPredictions = Ensemble_test




## === cell 36
print("Number of blended XGBoost/Linear predictions:", XGBPredictions.size)




## === cell 37
XGB_submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": XGBPredictions},
    columns=["key", "fare_amount"],
)




## === cell 38
submission = XGB_submission




## === cell 39
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
