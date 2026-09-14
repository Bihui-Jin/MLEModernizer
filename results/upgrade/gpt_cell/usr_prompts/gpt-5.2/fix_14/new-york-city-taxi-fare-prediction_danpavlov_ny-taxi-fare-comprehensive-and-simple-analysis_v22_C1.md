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

3.4594

# 6. Current score

6.07242

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 9.21109) has done: 'Diagnosis: The crash happens in cell 38 because `xgb.train()` in XGBoost 2.0.3 returns a `Booster` that no longer exposes the legacy attribute `best_ntree_limit`. With early stopping, the correct way to limit trees is to use the best iteration via `iteration_range` (or omit tree limiting entirely), so accessing `xgbm.best_ntree_limit` raises `AttributeError`.  
Patch summary: Update cell 38 to compute predictions using the booster’s best iteration if available, falling back safely to a normal predict call when early stopping metadata isn’t present. This preserves the same training procedure and uses the best model selected by early stopping.  
Updated cells: Only cell 38 is modified.  
Compatibility notes for cell k+1: `XGBPredictions` remains defined as a NumPy array of predictions, so `cell 39` continues to work unchanged.  
Assumptions: XGBoost 2.0.3 supports `Booster.best_iteration` and `predict(iteration_range=...)`; if not present, the fallback `predict()` is acceptable and keeps execution unblocked.'
- What this solution (achieved 12.46206) has done: 'Your current score (9.21109 RMSE) is much worse than the target (3.4594), so we should improve generalization with the smallest changes that keep your core model/feature logic intact. The biggest issue hurting performance is that you compute `distance` using slow row-iteration and a slightly inconsistent radius, and you also round predictions to cents before scoring, which typically worsens RMSE. I replace the row-wise distance loop with a vectorized haversine computation (same “distance feature” concept, just computed correctly/consistently and much faster), keep the same train/test feature set, and remove rounding on predictions. I also set a fixed random seed in `train_test_split` for stability and update the XGBoost objective to the correct modern name (`reg:squarederror`) while keeping the same training procedure and early stopping behavior.'
- What this solution (achieved 5.7638) has done: 'Your RMSE (12.46206) is far worse than the target (3.4594), so we should make the smallest changes that improve generalization without changing the model/feature “shape.” The biggest single limiter is that you only train on 100k rows out of 55M, which strongly hurts accuracy; increasing the training sample (still using `nrows`, same pipeline) is a minimal, high-impact change that should move RMSE substantially toward the target. I also align train/test parsing by loading `test.csv` with the same numeric dtypes (avoids subtle type issues) and add a small, standard fare upper-bound filter to remove extreme outliers that distort RMSE, while keeping your existing filters/features/model intact. The script still runs end-to-end and writes a valid `XGBSubmission.csv` with the required columns.'
- What this solution (achieved 5.83229) has done: 'Your current RMSE (5.7638) is still materially worse than the target (3.4594), so we should improve accuracy with the smallest changes that keep your feature set and training flow intact. The biggest score limiter remaining is that the XGBoost model is training with essentially-default tree settings (depth/eta/subsampling), which underfits this problem even with 2M rows. I keep the same XGBoost training API, early stopping, and features, but add a small set of standard regression tree parameters (depth/eta/subsample/colsample/min_child_weight) and increase boosting rounds modestly so early stopping can still pick the best iteration. I also ensure XGBoost gets float32 feature matrices consistently (reduces subtle dtype issues and improves stability) while preserving identical semantics.'
- What this solution (achieved 8.46216) has done: 'Your current RMSE (5.83229) is worse than the target (3.4594), so we should improve generalization with minimal, score-relevant changes while keeping the same features and XGBoost training flow. The biggest remaining limiter is outlier/noise in the 2M-row sample (bad coordinates and “zero distance” rides), which can disproportionately hurt RMSE; we add a couple of standard NYC Taxi sanity filters that remove clearly-invalid trips but don’t change the model/feature design. We also add a tiny L2 regularization (`lambda`) and a modest `gamma` to reduce overfitting to noise without changing the core model architecture, and we evaluate on both train/test during early stopping to stabilize the chosen iteration. The script still run end-to-end and write `XGBSubmission.csv` with `key,fare_amount`.'
- What this solution (achieved 6.82158) has done: 'Your current RMSE (8.46216) is far worse than the target (3.4594), so we should make a small change that improves generalization without altering your feature set or training flow. The biggest remaining issue is that the model is learning from many “non-physical” examples (distance > 0 but fare extremely low, and unrealistically high $/km), which heavily hurts RMSE. I add two standard, lightweight sanity filters based on fare-per-km and a minimum fare threshold, applied only to the training data after distance is computed, keeping the same XGBoost setup and submission format. This should move the score substantially toward the target while preserving your core logic.'
- What this solution (achieved 6.31762) has done: 'Your current RMSE (6.82158) is still far above the target (3.4594), so we should make the smallest change that improves generalization without changing your feature set or training flow. The largest remaining source of error is that the training sample still contains many mislabeled/noisy rows (especially with slightly-off NYC bounding boxes and rare-but-harmful “bad coordinate” trips) that inflate RMSE; tightening the coordinate sanity filters to a standard NYC bounding box is a minimal, high-impact cleanup. This keeps the same distance feature, same datetime features, same airport-distance features, and the same XGBoost training approach; it only removes clearly invalid geography so the model learns a cleaner mapping. The script still runs end-to-end and writes the same `XGBSubmission.csv` with `key,fare_amount`.'
- What this solution (achieved 6.07242) has done: 'Your current RMSE (6.31762) is still far above the target (3.4594), so we need a small, reliable accuracy gain without changing the overall modeling pipeline. The biggest likely limiter is that fare noise/outliers are still present after basic geo filters; tightening these with a standard NYC-specific “cleaning” step (including lat/lon sanity, passenger_count sanity, and removing extreme fare-per-distance cases more robustly) typically improves RMSE materially while keeping the same features and XGBoost training flow. I also add a simple clipping of negative/too-small predictions at submission time (legitimate post-processing for fares) to reduce RMSE impact from occasional negative predictions. All changes keep the same model, same feature set (plus the already-existing engineered ones), and still write `XGBSubmission.csv` in the required format.'
- What this solution (achieved 5.69855) has done: 'Your current RMSE (6.07242) is worse than the target (3.4594), so we should make a small, score-relevant improvement without changing your feature set or XGBoost training flow. The biggest remaining issue is that the model can still be dominated by noisy/outlier training examples and the raw fare scale, which tree boosting often models better in log-space. I add a minimal log1p target transform inside the existing XGBoost training function (same model/params/early stopping), invert it at prediction time, and keep the same submission format; this typically reduces RMSE substantially for this competition. I also apply the same non-negativity clipping after inversion to keep predictions physically valid.'
- What this solution (achieved 5.67956) has done: 'Your current RMSE (5.69855) is still well above the target (3.4594), so we should make the smallest reliable improvement without changing your feature set or XGBoost training flow. The biggest remaining issue is that the log1p target transform requires training/evaluating RMSE in log-space, which is misaligned with the competition metric (RMSE in dollars); we keep the log1p transform but switch XGBoost’s `eval_metric` to `rmse` computed in original fare space via a custom feval, so early stopping selects the iteration that minimizes Kaggle RMSE. We keep the same parameters, early stopping, and features, but ensure the “best iteration” is chosen against the correct metric. Submission writing stays identical and still outputs `XGBSubmission.csv` with `key,fare_amount`.'
- What this solution (achieved 5.67956) has done: 'We’re still far above the target RMSE, so we make two minimal, score-relevant fixes that keep your feature set and XGBoost training flow intact. First, early stopping is currently guided by XGBoost’s built-in `rmse` on the *log1p* target (because `eval_metric` is set), not by your custom fare-space RMSE; we remove `eval_metric` so early stopping uses the custom `feval` that matches Kaggle’s metric. Second, we switch the log transform to a standard “shifted log” using a constant offset (e.g., +1.0) rather than `log1p`, to reduce compression bias on small/medium fares while preserving the same model/approach (still “log-space training + exp inverse”). These are small semantic adjustments expected to improve generalization toward the target without changing architecture, features, or training procedure, and the script still write a valid `XGBSubmission.csv`.'
- What this solution (achieved 5.55151) has done: 'Your current RMSE (5.67956) is still well above the target (3.4594), so we should make a small change that improves accuracy without changing your features or overall XGBoost training flow. The biggest issue is that `objective='reg:squarederror'` plus a custom `feval` does **not** actually train on your fare-space RMSE; it trains on squared error in **log space**, and early stopping just *selects* iterations by fare-space RMSE—this mismatch can leave performance on the table. The minimal fix is to keep the same model/params/early stopping, but switch to XGBoost’s native RMSLE objective (`reg:squaredlogerror`) and train on `y` directly (no manual log/exp), while still early-stopping on standard RMSE in dollars. This preserves the core approach (XGBoost with the same engineered features and early stopping) and typically moves RMSE materially toward your target for this competition.'
- What this solution (achieved 6.07242) has done: 'Your RMSE (5.55151) is still worse than the target (3.4594), so we should make a small change that improves accuracy without changing your feature set or training flow. The biggest issue is an objective/target mismatch: `reg:squaredlogerror` expects the label to be `log1p(fare)` (and requires non-negative labels), but you’re feeding raw dollars, which degrades learning. I keep the same XGBoost training API, parameters, early stopping, and features, but switch the objective back to standard squared error in dollars (`reg:squarederror`) so training aligns with Kaggle’s RMSE metric. Everything else stays the same, including writing `XGBSubmission.csv` with the correct columns.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from geopy.distance import great_circle  # calculate distances
from sklearn import metrics  # evaluating models
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
)  # set splitting and validation
from sklearn.linear_model import LinearRegression
import xgboost as xgb  # XGBoost classifier
import matplotlib.pyplot as plt  # plotting
import seaborn as sns  # plotting
from math import sin, cos, sqrt, atan2, radians

try:
    get_ipython().run_line_magic("matplotlib", "inline")
except Exception:
    pass



## === cell 1
print(os.listdir("../input"))



## === cell 2
types = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}
test = pd.read_csv(
    "../input/test.csv", dtype={k: v for k, v in types.items() if k != "fare_amount"}
)



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
train = pd.read_csv("../input/train.csv", nrows=2_000_000, dtype=types)



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
train = train[train["fare_amount"] < 250]

train = train[train["pickup_longitude"].between(-74.25, -73.70)]
train = train[train["dropoff_longitude"].between(-74.25, -73.70)]
train = train[train["pickup_latitude"].between(40.50, 41.00)]
train = train[train["dropoff_latitude"].between(40.50, 41.00)]

train = train[(train["passenger_count"] > 0) & (train["passenger_count"] <= 6)]



## === cell 13
train.describe()




## === cell 14
def dist_calc(df):
    for i, row in df.iterrows():
        df.at[i, "distance"] = great_circle(
            (row["pickup_latitude"], row["pickup_longitude"]),
            (row["dropoff_latitude"], (row["dropoff_longitude"])),
        ).km




## === cell 15
def quick_dist_calc(df):
    """
    Compute the same "distance" feature vectorized with standard earth radius (6371 km).
    Keeps identical feature concept but improves numerical consistency and avoids dtype issues.
    """
    R = 6371.0
    lat1 = np.radians(df["pickup_latitude"].astype("float64").to_numpy())
    lon1 = np.radians(df["pickup_longitude"].astype("float64").to_numpy())
    lat2 = np.radians(df["dropoff_latitude"].astype("float64").to_numpy())
    lon2 = np.radians(df["dropoff_longitude"].astype("float64").to_numpy())

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    df["distance"] = (R * c).astype("float32")




## === cell 16
quick_dist_calc(train)
quick_dist_calc(test)



## === cell 17
train = train[(train["distance"] > 0.0) & (train["distance"] < 100.0)]



## === cell 18
fare_per_km = train["fare_amount"] / train["distance"].clip(lower=0.2)
train = train[train["fare_amount"] >= 2.5]
train = train[(fare_per_km >= 1.0) & (fare_per_km <= 35.0)]



## === cell 19
train["pickup_datetime"] = train["pickup_datetime"].str.replace(" UTC", "")
train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], format="%Y-%m-%d %H:%M:%S"
)

test["pickup_datetime"] = test["pickup_datetime"].str.replace(" UTC", "")
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], format="%Y-%m-%d %H:%M:%S"
)



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
def sphere_dist(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    R_earth = 6371
    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = map(
        np.radians, [pickup_lat, pickup_lon, dropoff_lat, dropoff_lon]
    )
    dlat = dropoff_lat - pickup_lat
    dlon = dropoff_lon - pickup_lon

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(pickup_lat) * np.cos(dropoff_lat) * np.sin(dlon / 2.0) ** 2
    )

    return 2 * R_earth * np.arcsin(np.sqrt(a))




## === cell 22
def add_airport_dist(dataset):
    jfk_coord = (40.639722, -73.778889)
    ewr_coord = (40.6925, -74.168611)
    lga_coord = (40.77725, -73.872611)

    pickup_lat = dataset["pickup_latitude"]
    dropoff_lat = dataset["dropoff_latitude"]
    pickup_lon = dataset["pickup_longitude"]
    dropoff_lon = dataset["dropoff_longitude"]

    pickup_jfk = sphere_dist(pickup_lat, pickup_lon, jfk_coord[0], jfk_coord[1])
    dropoff_jfk = sphere_dist(jfk_coord[0], jfk_coord[1], dropoff_lat, dropoff_lon)
    pickup_ewr = sphere_dist(pickup_lat, pickup_lon, ewr_coord[0], ewr_coord[1])
    dropoff_ewr = sphere_dist(ewr_coord[0], ewr_coord[1], dropoff_lat, dropoff_lon)
    pickup_lga = sphere_dist(pickup_lat, pickup_lon, lga_coord[0], lga_coord[1])
    dropoff_lga = sphere_dist(lga_coord[0], lga_coord[1], dropoff_lat, dropoff_lon)

    dataset["jfk_dist"] = pd.concat([pickup_jfk, dropoff_jfk], axis=1).min(axis=1)
    dataset["ewr_dist"] = pd.concat([pickup_ewr, dropoff_ewr], axis=1).min(axis=1)
    dataset["lga_dist"] = pd.concat([pickup_lga, dropoff_lga], axis=1).min(axis=1)

    return dataset




## === cell 23
train = add_airport_dist(train)
test = add_airport_dist(test)



## === cell 24
train.head()



## === cell 25
test.head()



## === cell 26
plt.figure(figsize=(15, 8))
sns.heatmap(
    train.drop(["key", "pickup_datetime"], axis=1).corr(), annot=True, fmt=".4f"
)



## === cell 27
X = train.drop(["key", "fare_amount", "pickup_datetime"], axis=1)
y = train["fare_amount"]



## === cell 28
X.head()



## === cell 29
y.head()



## === cell 30
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



## === cell 31
test_pred = test.drop(["key", "pickup_datetime"], axis=1)



## === cell 32
lm = LinearRegression()
lm.fit(X_train, y_train)
print(lm.score(X_train, y_train))
print(lm.score(X_test, y_test))



## === cell 33
y_pred = lm.predict(X)
lrmse = np.sqrt(metrics.mean_squared_error(y_pred, y))
lrmse



## === cell 34
LinearPredictions = lm.predict(test_pred)
LinearPredictions



## === cell 35
LinearPredictions.size



## === cell 36
linear_submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": LinearPredictions},
    columns=["key", "fare_amount"],
)



## === cell 37
linear_submission.head()




## === cell 38
def XGBoost(X_train, X_test, y_train, y_test):
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        "objective": "reg:squarederror",
        "eta": 0.05,
        "max_depth": 8,
        "min_child_weight": 1.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 1.0,
        "gamma": 0.1,
        "seed": 42,
        "eval_metric": "rmse",
    }

    return xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=2000,
        early_stopping_rounds=50,
        evals=[(dtrain, "train"), (dtest, "test")],
        maximize=False,
        verbose_eval=100,
    )




## === cell 39
xgbm = XGBoost(X_train, X_test, y_train, y_test)

test_pred_xgb = test_pred.astype("float32")
dtest_full = xgb.DMatrix(test_pred_xgb)
best_iter = getattr(xgbm, "best_iteration", None)

if best_iter is not None:
    XGBPredictions = xgbm.predict(
        dtest_full, iteration_range=(0, best_iter + 1)
    ).astype("float32")
else:
    XGBPredictions = xgbm.predict(dtest_full).astype("float32")



## === cell 40
XGBPredictions



## === cell 41
XGBPredictions



## === cell 42
XGBPredictions = np.clip(XGBPredictions, 0.0, None)

XGB_submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": XGBPredictions}, columns=["key", "fare_amount"]
)
XGB_submission.head()



## === cell 43
submission = XGB_submission



## === cell 44
submission.to_csv("XGBSubmission.csv", index=False)
print(
    "Wrote submission:",
    os.path.abspath("XGBSubmission.csv"),
    "rows=",
    len(submission),
    "cols=",
    submission.columns.tolist(),
)
