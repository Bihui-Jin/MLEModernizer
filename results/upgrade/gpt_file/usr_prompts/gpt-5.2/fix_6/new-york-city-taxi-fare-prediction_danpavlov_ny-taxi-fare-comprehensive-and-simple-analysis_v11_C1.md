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

5.77263

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.85277) has done: 'I fix the XGBoost prediction crash caused by using deprecated/removed attributes (`best_ntree_limit`) and objective (`reg:linear`) under xgboost 2.0.3, while keeping the same training loop and model approach. I also make the distance feature computation fast enough to complete by vectorizing it (the current row-wise `great_circle` loop likely time out on 500k rows). Finally, I remove notebook-only syntax (`%matplotlib inline`) so the script runs as a .py and ensure a valid `submission.csv` with the required `key,fare_amount` columns is written.'
- What this solution (achieved 6.00386) has done: 'You’re currently far worse than the target (RMSE 5.85 vs 3.91, lower is better), so we make small, metric-aligned improvements without changing the overall approach (same features + LinearRegression + XGBoost training). The biggest easy win is to stop rounding predictions to cents before scoring—rounding increases RMSE and hurts leaderboard score. We also remove early stopping (it changes training convergence/quality) and instead train a fixed number of rounds while using a small set of standard XGBoost regression hyperparameters that typically improve RMSE on this competition without altering the core model family. Finally, we compute and print a local holdout RMSE for the XGBoost model to sanity-check that the change improves generalization before writing the same `submission.csv` format.'
- What this solution (achieved 6.24117) has done: 'You’re currently worse than the target (RMSE 6.00 vs 3.91, lower is better), so we make small, metric-aligned improvements without changing the overall modeling approach (same LinearRegression + XGBoost on engineered features). The biggest legitimate gains here typically come from (1) adding standard time-based features from `pickup_datetime` (day-of-week/month) and (2) slightly improving the XGBoost setup to reduce overfitting and better match RMSE, while keeping the same `xgb.train` loop and objective. We also ensure train/test receive identical feature engineering and fill any missing values consistently so the submission is valid and stable. These changes should move your score down toward the target without altering the core logic family.'
- What this solution (achieved 5.85246) has done: 'We’re currently worse than the target (6.24117 vs 3.91427 RMSE; lower is better), so the smallest likely win is to fix an underfitting/overfitting mismatch in XGBoost without changing the overall approach (same engineered features + `xgb.train`). I keep your feature set and training loop, but tune a few XGBoost regularization knobs that usually reduce RMSE on this competition: use a shallower tree, add `min_child_weight`, `gamma`, and a bit more `lambda`, and use a slightly smaller `eta` with a proportionally larger number of boosting rounds. I also add `verbosity: 0` for stability and keep the same deterministic seed and histogram tree method. This should move the score down toward your target while preserving your core logic and still writing a valid `submission.csv`.'
- What this solution (achieved 5.77263) has done: 'You’re well below the target performance (RMSE 5.85 vs 3.91, lower is better), so we make the smallest changes that typically reduce RMSE without changing the overall approach (same engineered features + `xgb.train`). The main issue is underpowered/less stable training from a random split and a “one-size” boosting schedule; we (1) add `early_stopping_rounds` back in but keep the same training loop (`xgb.train`) and (2) use the best iteration automatically at prediction time (xgboost 2.x supports `iteration_range`) to avoid over/under-training. We also make the split deterministic but stratified-ish by time by using a simple chronological split on `pickup_datetime` (still the same holdout concept, just less leakage and more realistic), which usually improves generalization on this dataset. These are minimal, metric-aligned adjustments and still write a valid `submission.csv` with `key,fare_amount`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
from geopy.distance import great_circle  # kept to preserve original intent
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)



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
try:
    sns.distplot(train["fare_amount"])
    plt.show()
except Exception:
    pass



## === cell 9
try:
    sns.distplot(train["passenger_count"])
    plt.show()
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



## === cell 13
train.describe()




## === cell 14
def dist_calc(df: pd.DataFrame) -> None:
    lat1 = np.deg2rad(df["pickup_latitude"].astype("float64").to_numpy())
    lon1 = np.deg2rad(df["pickup_longitude"].astype("float64").to_numpy())
    lat2 = np.deg2rad(df["dropoff_latitude"].astype("float64").to_numpy())
    lon2 = np.deg2rad(df["dropoff_longitude"].astype("float64").to_numpy())

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    R_km = 6371.0
    df["distance"] = (R_km * c).astype("float32")




## === cell 15
dist_calc(train)
dist_calc(test)



## === cell 16
train["pickup_datetime"] = train["pickup_datetime"].str.replace(" UTC", "", regex=False)
train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
)



## === cell 17
test["pickup_datetime"] = test["pickup_datetime"].str.replace(" UTC", "", regex=False)
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
)



## === cell 18
train.dropna(subset=["pickup_datetime"], inplace=True)
test["pickup_datetime"] = test["pickup_datetime"].fillna(
    pd.Timestamp("2015-01-01 00:00:00")
)

train["year"] = train.pickup_datetime.dt.year.astype("int16")
train["hour"] = train.pickup_datetime.dt.hour.astype("int8")
train["month"] = train.pickup_datetime.dt.month.astype("int8")
train["dayofweek"] = train.pickup_datetime.dt.dayofweek.astype("int8")

test["year"] = test.pickup_datetime.dt.year.astype("int16")
test["hour"] = test.pickup_datetime.dt.hour.astype("int8")
test["month"] = test.pickup_datetime.dt.month.astype("int8")
test["dayofweek"] = test.pickup_datetime.dt.dayofweek.astype("int8")



## === cell 19
test.head()



## === cell 20
try:
    plt.figure(figsize=(15, 8))
    sns.heatmap(
        train.drop(["key", "pickup_datetime"], axis=1).corr(), annot=True, fmt=".4f"
    )
    plt.show()
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
split_idx = int(len(train) * 0.8)
order = np.argsort(train["pickup_datetime"].values)
train_idx = order[:split_idx]
valid_idx = order[split_idx:]

X_train, X_test = X.iloc[train_idx].copy(), X.iloc[valid_idx].copy()
y_train, y_test = y.iloc[train_idx].copy(), y.iloc[valid_idx].copy()



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
fill_values = X_train.median(numeric_only=True)
X_train = X_train.fillna(fill_values)
X_test = X_test.fillna(fill_values)
test_pred = test_pred.fillna(fill_values)



## === cell 27
lm = LinearRegression()
lm.fit(X_train, y_train)
print(lm.score(X_train, y_train))
print(lm.score(X_test, y_test))



## === cell 28
y_pred = lm.predict(X)
lrmse = np.sqrt(metrics.mean_squared_error(y_pred, y))
lrmse



## === cell 29
LinearPredictions = lm.predict(test_pred).astype("float32")
LinearPredictions



## === cell 30
LinearPredictions.size



## === cell 31
linear_submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": LinearPredictions},
    columns=["key", "fare_amount"],
)
linear_submission.head()




## === cell 32
def XGBoost(X_train, X_test, y_train, y_test):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_test, label=y_test)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.03,
        "max_depth": 6,
        "min_child_weight": 5.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 2.0,
        "alpha": 0.0,
        "gamma": 0.1,
        "seed": 42,
        "tree_method": "hist",
        "verbosity": 0,
    }

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=2000,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    return booster




## === cell 33
xgbm = XGBoost(X_train, X_test, y_train, y_test)

dvalid = xgb.DMatrix(X_test)
best_iter = int(getattr(xgbm, "best_iteration", 0) or 0)
valid_pred = xgbm.predict(dvalid, iteration_range=(0, best_iter + 1))
valid_rmse = float(np.sqrt(metrics.mean_squared_error(y_test, valid_pred)))
print("Holdout RMSE (XGB):", valid_rmse, "| best_iteration:", best_iter)

dtest_full = xgb.DMatrix(test_pred)
XGBPredictions = xgbm.predict(dtest_full, iteration_range=(0, best_iter + 1))



## === cell 34
XGBPredictions



## === cell 35
XGBPredictions = XGBPredictions.astype("float32")
XGBPredictions



## === cell 36
XGB_submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": XGBPredictions}, columns=["key", "fare_amount"]
)
XGB_submission.head()



## === cell 37
submission = XGB_submission



## === cell 38
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
