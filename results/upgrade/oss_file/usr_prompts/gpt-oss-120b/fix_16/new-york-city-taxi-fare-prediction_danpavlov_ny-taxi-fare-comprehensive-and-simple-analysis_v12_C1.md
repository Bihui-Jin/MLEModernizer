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

5.80312

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.86517) has done: 'The changes remove the notebook‑only magic command, fix the XGBoost training call (use the current objective name and obtain the best iteration correctly), and adjust the prediction step to use the trained Booster without accessing a non‑existent attribute. These fixes allow the pipeline to run end‑to‑end and produce a properly formatted `submission.csv` file, moving the solution toward the target RMSE.'
- What this solution (achieved 5.8033) has done: 'I keep the overall pipeline unchanged but give XGBoost a more suitable set of hyper‑parameters (learning rate, depth, subsampling) and allow it to run longer so it can find a better model. I also remove the premature rounding of predictions (and clip any negative values) which can inflate the RMSE. Finally I add a quick validation RMSE printout so we can see the improvement before writing the submission.'
- What this solution (achieved 5.78902) has done: 'Implemented a log‑transformation of the target variable for the XGBoost (and Linear Regression) model, then exponentiated predictions back to the original scale before evaluating RMSE and creating the submission. This simple calibration often reduces error on skewed fare data, moving the validation RMSE closer to the target value.'
- What this solution (achieved 5.78883) has done: 'We fix the validation‑RMSE calculation by selecting the true fares with label‑based indexing (`.loc`) instead of positional `.iloc`, which caused the out‑of‑bounds error. Additionally, we slightly strengthen the XGBoost hyper‑parameters (deeper trees, higher subsample rates, and a larger early‑stopping window) to improve predictive performance while keeping the overall pipeline unchanged.'
- What this solution (achieved 5.78883) has done: 'I add a lightweight blending step that combines the XGBoost and linear regression predictions. By evaluating both models on the validation split, I compute an optimal weight `w` that minimizes validation RMSE, then use that same weight to blend the test‑set predictions. This small change keeps the core modelling untouched while typically lowering the overall RMSE, moving the score closer to the target.'
- What this solution (achieved 5.79383) has done: 'I add a simple distance‑based feature (log‑distance) and filter out unrealistically large trips, then make the XGBoost model slightly more expressive (deeper trees, full column/row sampling and a smaller learning rate). These minimal, targeted changes keep the overall pipeline intact while providing stronger predictive power and should reduce the RMSE toward the target.'
- What this solution (achieved 5.760588451154757e+91) has done: 'I tighten unrealistic trips by discarding rows with a taxi‑distance > 100 km right after the distance is computed, and I give XGBoost a small L2 regularisation term (“lambda”) to reduce over‑fitting. Both tweaks keep the original pipeline intact while plausibly lowering the validation RMSE, moving the score closer to the target.'
- What this solution (achieved 6.43937) has done: 'I add a modest clipping step to the model predictions in the validation and test stages so that any extreme log‑fare values (which explode when exponentiated) are limited to a realistic range. This prevents overflow‑induced huge RMSE values and moves the score toward the target without changing the core modeling logic.'
- What this solution (achieved 236.83424) has done: 'I keep the overall pipeline intact but remove the log‑transform of the target so the models are trained directly on the fare amount, which generally yields a lower RMSE for this competition. I also slightly adjust the XGBoost hyper‑parameters (a bit deeper trees and a higher learning rate) to give the model more capacity. All prediction steps are updated to work with raw fares and are clipped only to enforce non‑negative values. These focused changes preserve the core logic while moving the validation RMSE toward the target.'
- What this solution (achieved 8.640882676732136e+91) has done: 'I re‑introduce a log‑transform of the fare target (using log1p) and exponentiate the predictions back to the original scale for both the linear regression and XGBoost models. This keeps the same modeling pipeline while dramatically reducing the RMSE, moving the score from the huge 236 down toward the target 3.87. The changes affect only the target handling, prediction conversion, and associated RMSE calculations.'
- What this solution (achieved 8.640882676732136e+91) has done: 'I add safe clipping of the XGBoost log‑predictions before exponentiating them back to fare amounts. This prevents overflow‑induced huge values that cause the RMSE to explode, moving the score much closer to the target while keeping the modeling pipeline unchanged.'
- What this solution (achieved 5.76907) has done: 'I tighten the XGBoost model to reduce over‑fitting (shallower trees and stronger L2 regularisation) and make the validation target alignment explicit, which prevents any index mismatches that can inflate the RMSE. These minimal tweaks keep the overall pipeline unchanged while expected to lower the validation RMSE and therefore move the score nearer to the target.'
- What this solution (achieved 5.78063) has done: 'The fixes import the missing matplotlib backend, increase the training sample size, and standard‑scale the features for the linear model – which corrects the NameError, gives the XGBoost model more data, and improves the linear regression component, leading to a lower validation RMSE and a valid submission.csv output.'
- What this solution (achieved 112.76183) has done: 'I switch the models to train directly on the original fare amount instead of a log‑transformed target, remove the unnecessary log‑exponentiation steps, and slightly strengthen the XGBoost hyper‑parameters (higher learning rate, deeper trees, fewer boost rounds). These focused changes keep the overall pipeline and feature set intact while expected to lower the RMSE toward the target.'
- What this solution (achieved 5.80312) has done: 'I apply a log‑transform to the fare target so the models train on a less skewed distribution, then exponentiate predictions back to the original scale for evaluation and submission. This small change keeps the overall pipeline untouched while dramatically lowering the RMSE, moving the score toward the target.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
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
    for i, row in df.iterrows():
        df.at[i, "distance"] = great_circle(
            (row["pickup_latitude"], row["pickup_longitude"]),
            (row["dropoff_latitude"], row["dropoff_longitude"]),
        ).km




## === cell 15
dist_calc(train)
dist_calc(test)
train["log_distance"] = np.log1p(train["distance"])
test["log_distance"] = np.log1p(test["distance"])
train = train[train["distance"] <= 100]



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
y = np.log1p(train["fare_amount"])  # log‑target for modeling
y_original = train["fare_amount"]  # keep original for evaluation



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
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_scaled_full = scaler.transform(X)

lm = LinearRegression()
lm.fit(X_train_scaled, y_train)
print("Linear train R²:", lm.score(X_train_scaled, y_train))
print("Linear val   R²:", lm.score(X_test_scaled, y_test))



## === cell 27
y_pred_log = lm.predict(X_scaled_full)
y_pred = np.expm1(y_pred_log)  # back to fare amount
lrmse = np.sqrt(metrics.mean_squared_error(y_original, y_pred))
print("Linear RMSE on full training data (original scale):", lrmse)



## === cell 28
test_pred_scaled = scaler.transform(test_pred)
LinearPredictions_log = lm.predict(test_pred_scaled)
LinearPredictions = np.expm1(LinearPredictions_log)  # convert back
LinearPredictions = np.round(LinearPredictions, 2)



## === cell 29
linear_submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": LinearPredictions},
    columns=["key", "fare_amount"],
)




## === cell 30
def XGBoost(X_tr, X_va, y_tr, y_va):
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dvalid = xgb.DMatrix(X_va, label=y_va)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.1,
        "max_depth": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 2.0,
        "seed": 42,
        "verbosity": 0,
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=100,
        verbose_eval=False,
    )
    return model




## === cell 31
xgbm = XGBoost(X_train, X_test, y_train, y_test)



## === cell 32
val_true = y_original.loc[y_test.index]

val_pred_log = xgbm.predict(xgb.DMatrix(X_test))
val_pred = np.expm1(val_pred_log)  # back to fare amount
val_rmse = np.sqrt(mean_squared_error(val_true, val_pred))
print("XGBoost validation RMSE (original scale):", val_rmse)

lin_val_pred_log = lm.predict(X_test_scaled)
lin_val_pred = np.expm1(lin_val_pred_log)
lin_val_rmse = np.sqrt(mean_squared_error(val_true, lin_val_pred))
print("Linear Regression validation RMSE (original scale):", lin_val_rmse)

weights = np.arange(0.0, 1.01, 0.01)
best_w = 0.0
best_rmse = np.inf
for w in weights:
    blended = w * val_pred + (1 - w) * lin_val_pred
    rmse = np.sqrt(mean_squared_error(val_true, blended))
    if rmse < best_rmse:
        best_rmse = rmse
        best_w = w
print(
    f"Optimal blend weight for XGBoost: {best_w:.2f} (blended validation RMSE: {best_rmse:.4f})"
)



## === cell 33
xgb_test_pred_log = xgbm.predict(xgb.DMatrix(test_pred))
xgb_test_pred = np.expm1(xgb_test_pred_log)

lin_test_pred_log = lm.predict(test_pred_scaled)
lin_test_pred = np.expm1(lin_test_pred_log)

blended_test_pred = best_w * xgb_test_pred + (1 - best_w) * lin_test_pred
blended_test_pred = np.clip(blended_test_pred, 0, None)  # ensure non‑negative fares
XGBPredictions = blended_test_pred



## === cell 34
XGB_submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": XGBPredictions}, columns=["key", "fare_amount"]
)



## === cell 35
XGB_submission.head()



## === cell 36
submission = XGB_submission



## === cell 37
submission.to_csv("submission.csv", index=False)



## === cell 38
print("Submission file written to submission.csv with shape:", submission.shape)
