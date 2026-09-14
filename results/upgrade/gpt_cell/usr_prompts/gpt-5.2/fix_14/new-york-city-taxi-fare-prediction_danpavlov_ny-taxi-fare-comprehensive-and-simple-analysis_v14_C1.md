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

5.34633

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.92937) has done: 'Diagnosis: The crash happens in cell 33 because newer XGBoost versions (here xgboost==2.0.3) no longer expose `Booster.best_ntree_limit` in the same way older notebooks used to, so accessing `xgbm.best_ntree_limit` raises an `AttributeError`. The model training itself completes; only the prediction call fails due to this deprecated attribute usage. The most compatible fix is to stop relying on `best_ntree_limit` and instead use the booster’s `best_iteration` (when early stopping is used) to limit trees via `iteration_range`; otherwise fall back to a normal predict.

Patch summary: Update cell 33 to compute predictions using `iteration_range=(0, best_iteration + 1)` when available, and otherwise call `predict` without any tree-limit argument. This keeps early-stopping semantics (use the best iteration) while avoiding deprecated attributes in XGBoost 2.x.

Updated cells: Only cell 33 is changed.

Compatibility notes for cell k+1: `XGBPredictions` is still created as a NumPy array of predictions with the same shape/order as before, so `cell 34` (`XGBPredictions`) work unchanged.

Assumptions: `xgb.train(..., early_stopping_rounds=20, ...)` sets `best_iteration` on the returned booster in xgboost==2.0.3; if for any reason it doesn’t, the code safely falls back to a standard `predict`.'
- What this solution (achieved 5.46784) has done: 'Your current gap to the target is large (5.92937 vs 3.16533; lower is better), so the smallest legitimate improvement is to keep the same XGBoost approach but fix two issues that commonly inflate RMSE: (1) the distance feature is computed via an extremely slow row-iteration and may be incomplete/incorrect under timeouts, and (2) the train/test split is not deterministic and the model uses a deprecated objective name. I replace the distance calculation with a fast, vectorized haversine distance (same feature intent: great-circle distance), keep all existing features, and make the XGBoost parameters equivalent but compatible (`reg:squarederror`) while adding a fixed random seed for stability. I also ensure the submission file name ends with `.csv` and keep the early-stopping prediction logic you already fixed for XGBoost 2.x.'
- What this solution (achieved 7.11686) has done: 'Your RMSE is far above target, so we should make small, legitimate tweaks that usually improve generalization without changing the overall “XGBoost on engineered time + haversine features” approach. The biggest likely issue is that the model is being early-stopped on a holdout set but trained on raw-dollar targets with heavy-tailed outliers, which can inflate RMSE; clipping extreme fares and applying a standard NYC baseline “fare = base + per-km * distance” feature typically gives a sizable gain with minimal code change. I (1) add a light outlier cap on `fare_amount` after your existing geographic/passenger filters, (2) add a single derived feature `distance_fare = 2.5 + 1.56*distance` to both train/test, and (3) keep your same XGBoost training loop but slightly extend boosting rounds to let early stopping find a better best_iteration (no approximation/early-stop change). The submission writing stays the same and still produces a valid `.csv`.'
- What this solution (achieved 7.11649) has done: 'Your current RMSE (7.11686) is far above the target (3.16533), so the most likely issue is train/test row misalignment caused by dropping `pickup_datetime` NaNs in the test set, which silently reduces rows and breaks the required 1:1 mapping to `key` (this typically destroys leaderboard RMSE). I keep your exact feature set and XGBoost training logic, but ensure the test set keeps all rows by parsing datetimes without dropping and filling missing-derived time features with safe defaults. I also harden submission creation to always use the original test `key` order and to enforce a non-negative fare post-processing (common minimal RMSE improvement for this competition). These are minimal, evaluation-semantic-preserving fixes aimed at moving RMSE substantially toward your target without changing the model approach.'
- What this solution (achieved 7.11649) has done: 'Your current RMSE is far above target (needs to go down), and the smallest likely culprit is that the train/test feature distributions don’t match because only the training set is filtered for valid NYC coordinates/outliers while the test set is left unfiltered, which can produce extreme distances and unstable predictions. I keep your exact feature set and XGBoost training loop, but apply the same coordinate/passenger sanity filtering *to the test features only* by clipping to the training-valid bounds (so we keep all test rows/keys and avoid row drops). I also cap the engineered `distance` and `distance_fare` in test to the training range to prevent out-of-range extrapolation. These changes are minimal, preserve core logic, and typically reduce RMSE substantially for this competition.'
- What this solution (achieved 7.11649) has done: 'Your RMSE is far above the target (needs to go down), and the most likely cause given the current code is that the model is trained on a filtered “NYC-only, reasonable fare” subset while the test set still contains out-of-distribution coordinates/distances that lead to unstable extrapolation. I keep your exact feature set and XGBoost training logic, but make test features match the training domain by clipping pickup/dropoff coordinates to the training-valid ranges (without dropping any test rows/keys). I also clip `distance` and `distance_fare` in test to the observed training range, which is a minimal, legitimate post-processing step for covariate shift and usually reduces RMSE substantially. Everything else (data size, model, early-stopping semantics, submission format) remains unchanged.'
- What this solution (achieved 4.91426) has done: 'Your RMSE is much worse than the target (needs to go down), so the smallest likely improvement is to make the engineered `distance` feature consistent with the coordinate clipping you already do for the test set. Right now, `distance` is computed on raw test coordinates and only later coordinates are clipped, so `distance` (and `distance_fare`) can remain out-of-distribution and hurt predictions; we recompute `distance` and `distance_fare` after clipping and then re-apply numeric NaN/inf filling. This preserves the same feature set and XGBoost training logic, but fixes a train/test feature mismatch that commonly inflates RMSE. I also ensure the submission filename ends with `.csv` (unchanged) and keep your non-negative post-processing.'
- What this solution (achieved 4.91426) has done: 'The crash happens because `lm` was fitted on `X_train` (which inherits the exact column order from `X`), but `test_pred` has a different feature column order, and scikit-learn 1.2+ enforces matching feature names *and order* at predict time. The minimal safe fix is to reorder `test_pred` columns to match `X.columns` right before calling `lm.predict`, without changing any features or model logic. This preserves the same evaluation semantics and only addresses the mismatch causing the `ValueError`. The rest of the prediction post-processing (clipping at 0 and rounding) stays unchanged.'
- What this solution (achieved 4.72586) has done: 'We need to move RMSE down from 4.91426 toward 3.16533 (lower is better), so we should make small, legitimate fixes that improve generalization without changing your overall “engineered time+distance features + XGBoost” approach. The biggest low-risk gain here is to correct evaluation leakage: you currently early-stop using the test split and then evaluate/predict with a model selected on that same split, which can pick a suboptimal number of trees for true generalization; switching to a proper train/valid split for early stopping (still within the same training approach) typically improves leaderboard RMSE. I keep all your feature engineering and parameters, but split the training data into (train, valid) for early stopping, then retrain once on full training data using the chosen best_iteration before predicting test. I also add a minimal “fill numeric NaNs/infs” step to X itself (not just test) to avoid rare training-time NaNs affecting the booster, while keeping submission format and file path unchanged.'
- What this solution (achieved 4.80328) has done: 'Your RMSE (4.72586) is still substantially worse than the target (3.16533; lower is better), so we should make a small, legitimate improvement that keeps your exact “time + haversine + distance_fare + XGBoost with early-stopping then refit” approach intact. The highest-impact minimal issue is that `X` (train features) currently isn’t constrained the same way as `test_pred`: you clip coordinates/distances for test, but not for train, which can let rare bad training coordinates inflate noise and hurt generalization. I apply the same coordinate clipping (to the already-filtered NYC bounds you used) to the training features **after** cleaning, then recompute `distance` and `distance_fare` consistently—this preserves the same feature set but improves train/test feature alignment. I also ensure the XGBoost DMatrix uses `missing=np.nan` explicitly (no semantic change, just robustness), and keep the submission format/filename unchanged.'
- What this solution (achieved 4.82511) has done: 'Your RMSE (4.80328) is still well above the target (3.16533; lower is better), so the most “minimal but high-impact” step is to improve generalization without changing your feature set or training approach. The smallest lever within your existing XGBoost-with-early-stopping-then-refit flow is to add standard, conservative XGBoost regularization parameters (tree depth/eta/subsample/colsample/min_child_weight) while keeping the same objective/metric, early stopping, and refit semantics. This typically reduces overfitting on a 1M-row sample and moves leaderboard RMSE down without altering core logic. I keep the CSV output format and file path the same.'
- What this solution (achieved 5.34633) has done: 'Your current RMSE (4.82511) is still far above the target (3.16533; lower is better), so we should make a small, legitimate improvement without changing the overall “engineered time+distance features + XGBoost with early-stopping then refit” approach. The biggest low-risk gain is to make the training sample less noisy by filtering out impossible/rare trips: drop rows with near-zero trip distance but non-trivial fare, and remove the longest-distance tail where coordinates can still be “valid” but fares become extremely variable. This keeps the same features, model, and training flow, but usually reduces RMSE materially on this competition by reducing label noise/outlier influence. The rest of the pipeline (test row preservation, feature alignment, prediction post-processing, and submission writing) remains unchanged.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn import metrics  # evaluating models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import xgboost as xgb  # XGBoost
import matplotlib.pyplot as plt  # plotting
import seaborn as sns  # plotting



## === cell 1
print(os.listdir("../input"))



## === cell 2
test = pd.read_csv("../input/test.csv")
test_keys = test["key"].copy()



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
try:
    sns.histplot(train["fare_amount"], kde=True)
    plt.show()
except Exception:
    pass



## === cell 9
try:
    sns.histplot(train["passenger_count"], kde=False)
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

train = train[train["fare_amount"] <= 250].copy()



## === cell 13
train.describe()




## === cell 14
def add_haversine_km(df):
    lat1 = np.radians(df["pickup_latitude"].astype("float64"))
    lon1 = np.radians(df["pickup_longitude"].astype("float64"))
    lat2 = np.radians(df["dropoff_latitude"].astype("float64"))
    lon2 = np.radians(df["dropoff_longitude"].astype("float64"))

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    R_km = 6371.0
    df["distance"] = (R_km * c).astype("float32")
    return df




## === cell 15
train = add_haversine_km(train)
test = add_haversine_km(test)

train["distance_fare"] = (2.50 + 1.56 * train["distance"]).astype("float32")
test["distance_fare"] = (2.50 + 1.56 * test["distance"]).astype("float32")



## === cell 16
train = train[(train["distance"] >= 0.01) | (train["fare_amount"] <= 5.0)].copy()
train = train[train["distance"] <= 60.0].copy()



## === cell 17
train["pickup_datetime"] = train["pickup_datetime"].str.replace(" UTC", "", regex=False)
train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
)

test["pickup_datetime"] = test["pickup_datetime"].str.replace(" UTC", "", regex=False)
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
)



## === cell 18
train = train.dropna(subset=["pickup_datetime"]).copy()



## === cell 19
train["hour"] = train.pickup_datetime.dt.hour
train["weekday"] = train.pickup_datetime.dt.weekday
train["month"] = train.pickup_datetime.dt.month
train["year"] = train.pickup_datetime.dt.year

test["hour"] = test.pickup_datetime.dt.hour
test["weekday"] = test.pickup_datetime.dt.weekday
test["month"] = test.pickup_datetime.dt.month
test["year"] = test.pickup_datetime.dt.year

for col, default in [("hour", 0), ("weekday", 0), ("month", 1), ("year", 2010)]:
    test[col] = test[col].fillna(default).astype("int16")



## === cell 20
test.head()



## === cell 21
try:
    plt.figure(figsize=(15, 8))
    sns.heatmap(
        train.drop(["key", "pickup_datetime"], axis=1).corr(), annot=True, fmt=".4f"
    )
    plt.show()
except Exception:
    pass



## === cell 22
X = train.drop(["key", "fare_amount", "pickup_datetime"], axis=1)
y = train["fare_amount"]



## === cell 23
X.head()



## === cell 24
y.head()



## === cell 25
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



## === cell 26
coord_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]

train_bounds = {
    "pickup_longitude": (-180.0, -72.0),
    "dropoff_longitude": (-180.0, -72.0),
    "pickup_latitude": (40.0, 44.0),
    "dropoff_latitude": (40.0, 44.0),
}

X = X.copy()
for c in coord_cols:
    if c in X.columns and c in train_bounds:
        lo, hi = train_bounds[c]
        X[c] = X[c].astype("float64").clip(lo, hi).astype("float32")

X = add_haversine_km(X)
X["distance_fare"] = (2.50 + 1.56 * X["distance"]).astype("float32")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



## === cell 27
test_pred = test.drop(
    ["key", "pickup_datetime", "distance", "distance_fare"], axis=1, errors="ignore"
).copy()

coord_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]
for col in coord_cols:
    if col in test_pred.columns and col in X.columns:
        lo = float(np.nanmin(X[col].values))
        hi = float(np.nanmax(X[col].values))
        test_pred[col] = test_pred[col].astype("float64").clip(lo, hi).astype("float32")

if "passenger_count" in test_pred.columns:
    test_pred["passenger_count"] = (
        test_pred["passenger_count"]
        .astype("float64")
        .round()
        .clip(1, 9)
        .astype("uint8")
    )

for col in ["hour", "weekday", "month", "year"]:
    if col in test_pred.columns and col in X.columns:
        lo = int(np.nanmin(X[col].values))
        hi = int(np.nanmax(X[col].values))
        test_pred[col] = (
            test_pred[col]
            .astype("float64")
            .fillna(np.nanmedian(X[col].values))
            .clip(lo, hi)
            .astype("int16")
        )

test_pred = add_haversine_km(test_pred)
test_pred["distance_fare"] = (2.50 + 1.56 * test_pred["distance"]).astype("float32")

for col in ["distance", "distance_fare"]:
    if col in test_pred.columns and col in X.columns:
        lo = float(np.nanmin(X[col].values))
        hi = float(np.nanmax(X[col].values))
        test_pred[col] = test_pred[col].astype("float64").clip(lo, hi).astype("float32")

for col in test_pred.columns:
    if col in X.columns and pd.api.types.is_numeric_dtype(test_pred[col]):
        test_pred[col] = test_pred[col].replace([np.inf, -np.inf], np.nan)
        if test_pred[col].isna().any():
            test_pred[col] = test_pred[col].fillna(float(np.nanmedian(X[col].values)))



## === cell 28
lm = LinearRegression()
lm.fit(X_train, y_train)
print(lm.score(X_train, y_train))
print(lm.score(X_test, y_test))



## === cell 29
y_pred = lm.predict(X)
lrmse = np.sqrt(metrics.mean_squared_error(y_pred, y))
lrmse



## === cell 30
test_pred = test_pred.reindex(columns=X.columns)

LinearPredictions = lm.predict(test_pred)
LinearPredictions = np.maximum(0.0, LinearPredictions)
LinearPredictions = np.round(LinearPredictions, decimals=2)
LinearPredictions



## === cell 31
LinearPredictions.size



## === cell 32
linear_submission = pd.DataFrame(
    {"key": test_keys, "fare_amount": LinearPredictions},
    columns=["key", "fare_amount"],
)



## === cell 33
linear_submission.head()




## === cell 34
def XGBoost_with_valid_then_refit(X_all, y_all, seed=42):
    X_all = X_all.replace([np.inf, -np.inf], np.nan)
    for c in X_all.columns:
        if X_all[c].isna().any():
            X_all[c] = X_all[c].fillna(float(np.nanmedian(X_all[c].values)))

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_all, y_all, test_size=0.2, random_state=seed
    )

    dtrain = xgb.DMatrix(X_tr, label=y_tr, missing=np.nan)
    dvalid = xgb.DMatrix(X_val, label=y_val, missing=np.nan)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "seed": seed,
        "eta": 0.05,
        "max_depth": 8,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
    }

    booster_es = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=2000,
        early_stopping_rounds=20,
        evals=[(dvalid, "valid")],
        verbose_eval=False,
    )

    best_iter = booster_es.best_iteration
    if best_iter is None:
        best_iter = booster_es.num_boosted_rounds() - 1

    dfull = xgb.DMatrix(X_all, label=y_all, missing=np.nan)
    booster_full = xgb.train(
        params=params,
        dtrain=dfull,
        num_boost_round=int(best_iter) + 1,
        evals=[],
        verbose_eval=False,
    )
    return booster_full




## === cell 35
xgbm = XGBoost_with_valid_then_refit(X, y)

test_pred = test_pred.reindex(columns=X.columns)
dtest_pred = xgb.DMatrix(test_pred, missing=np.nan)
XGBPredictions = xgbm.predict(dtest_pred)



## === cell 36
XGBPredictions



## === cell 37
XGBPredictions = np.maximum(0.0, XGBPredictions)
XGBPredictions = np.round(XGBPredictions, decimals=2)
XGBPredictions



## === cell 38
XGB_submission = pd.DataFrame(
    {"key": test_keys, "fare_amount": XGBPredictions}, columns=["key", "fare_amount"]
)
XGB_submission.head()



## === cell 39
submission = XGB_submission

assert len(submission) == len(test_keys)
assert submission["key"].iloc[0] == test_keys.iloc[0]



## === cell 40
submission.to_csv("XGBSubmission17082018_2.csv", index=False)
print("Wrote submission:", "XGBSubmission17082018_2.csv", "rows=", len(submission))
