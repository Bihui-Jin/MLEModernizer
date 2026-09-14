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

5.74559

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.83403) has done: 'I fix the notebook-to-script issues so it runs in Kaggle’s Python environment (remove the Jupyter magic and correct the cell numbering). I also fix the XGBoost API breakage causing `best_ntree_limit` to be missing in xgboost 2.0.3 by using `iteration_range` (or falling back safely) so prediction works. To keep runtime under control and avoid huge slowdowns, I replace the row-wise `great_circle` loop with a vectorized haversine distance computation (same feature intent: trip distance), which is necessary for the code to finish end-to-end. Finally, I ensure a valid submission CSV with the required filename suffix and columns (`key,fare_amount`) is always written.'
- What this solution (achieved 5.76239) has done: 'Your current score (5.83403 RMSE) is worse than the target (3.86972), so we should cautiously improve performance while keeping your approach intact (same features, same XGBoost training loop, same loss/metric). The biggest “minimal-change” gain here is to make XGBoost actually learn the temporal + distance signal better by setting a few standard tree/learning parameters (still the same model class and training API) and using a larger, but still manageable, training sample to reduce variance. I also remove rounding during prediction (rounding increases RMSE) while keeping the exact same submission schema and clipping. Finally, I ensure the output file is a valid `.csv` with the required columns.'
- What this solution (achieved 5.8328) has done: 'To move RMSE down toward your target while keeping the same overall approach (same basic features + XGBoost regressor), I make two minimal, high-impact fixes: (1) remove the `early_stopping_rounds` usage (it’s effectively “relaxed convergence criteria” and can stop too early), and (2) correct the train/validation evaluation so RMSE is computed on a proper holdout split (and train on train split only). I also add one standard, competition-safe cleanup that stays within your feature intent: drop extreme `fare_amount` outliers (very large fares dominate RMSE and hurt generalization) without changing the model/feature set. Finally, I ensure the test set keeps all rows (don’t drop test rows with datetime parse issues) so the submission row count always matches `sample_submission.csv`.'
- What this solution (achieved 7.49789) has done: 'Your current RMSE (5.8328) is worse than the target (3.86972), so we should legitimately improve generalization with the smallest changes that keep your exact core approach (same features: distance + time parts + passenger_count; same XGBoost training API/loop; same squared-error objective). The biggest minimal fix is that you currently drop the raw lat/long columns from modeling; adding them back (while keeping the existing engineered distance) usually reduces RMSE substantially for this competition without changing the modeling paradigm. I also add a tiny, metric-aligned post-processing step that is standard for RMSE on fares: clip unrealistically large predictions to the same cap you trained on (<200), which reduces the impact of occasional extreme predictions. Everything else (data reading size, cleaning, model family, boosting rounds) remains the same, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 5.74559) has done: 'The timeout is dominated by (1) repeatedly filtering the 4M-row training DataFrame with many chained boolean index operations that create multiple copies, (2) expensive string-based datetime cleaning/parsing, and (3) single-thread/inefficient defaults in XGBoost training. I keep the same model logic (LinearRegression + XGBoost with 2000 rounds, same features and params), but make the pipeline faster by building one combined boolean mask for filtering, parsing datetimes with a vectorized fast path (avoiding `.astype(str)`), and enabling multi-threading consistently (`n_jobs`/`nthread`) while ensuring dtypes/contiguity are friendly for `DMatrix`. I also remove/collapse heavy EDA computations (describe/corr/plots) that don’t affect training/predictions, preserving identical outputs (submission) aside from negligible float rounding.'

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

N_JOBS = max(1, (os.cpu_count() or 1))
os.environ.setdefault("OMP_NUM_THREADS", str(N_JOBS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(N_JOBS))
os.environ.setdefault("MKL_NUM_THREADS", str(N_JOBS))
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(N_JOBS))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(N_JOBS))




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
    "../input/test.csv",
    dtype={k: v for k, v in types.items() if k != "fare_amount"},
    usecols=[
        "key",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
)




## === cell 3
test.dtypes




## === cell 4
train = pd.read_csv(
    "../input/train.csv",
    nrows=4000000,
    dtype=types,
    usecols=[
        "key",
        "fare_amount",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
)




## === cell 5
train.head()




## === cell 6
pass




## === cell 7
pass




## === cell 8
pass




## === cell 9
train.isnull().sum()




## === cell 10
train.dropna(inplace=True)




## === cell 11
m = (
    (train["fare_amount"] > 0)
    & (train["fare_amount"] < 200)
    & (train["pickup_longitude"] < -72)
    & (train["dropoff_longitude"] < -72)
    & (train["pickup_latitude"] > 40)
    & (train["pickup_latitude"] < 44)
    & (train["dropoff_latitude"] > 40)
    & (train["dropoff_latitude"] < 44)
    & (train["passenger_count"] > 0)
    & (train["passenger_count"] < 10)
    & (train["pickup_longitude"] >= -180.0)
    & (train["pickup_longitude"] <= 180.0)
    & (train["dropoff_longitude"] >= -180.0)
    & (train["dropoff_longitude"] <= 180.0)
    & (train["pickup_latitude"] >= -90.0)
    & (train["pickup_latitude"] <= 90.0)
    & (train["dropoff_latitude"] >= -90.0)
    & (train["dropoff_latitude"] <= 90.0)
)
train = train.loc[m].copy()
del m




## === cell 12
pass




## === cell 13
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
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arcsin(np.sqrt(a))
    earth_radius_km = 6371.0088
    df["distance"] = (earth_radius_km * c).astype("float32")
    return df




## === cell 14
train = add_haversine_distance_km(train)
test = add_haversine_distance_km(test)

train_dist = train["distance"].values
test_dist = test["distance"].values
train["distance"] = np.where(np.isfinite(train_dist), train_dist, 0.0).astype("float32")
test["distance"] = np.where(np.isfinite(test_dist), test_dist, 0.0).astype("float32")
del train_dist, test_dist




## === cell 15
s = train["pickup_datetime"]
if s.dtype != "object":
    s = s.astype("string")
train["pickup_datetime"] = pd.to_datetime(
    s.str.replace(" UTC", "", regex=False),
    format="%Y-%m-%d %H:%M:%S",
    errors="coerce",
)
del s




## === cell 16
s = test["pickup_datetime"]
if s.dtype != "object":
    s = s.astype("string")
test["pickup_datetime"] = pd.to_datetime(
    s.str.replace(" UTC", "", regex=False),
    format="%Y-%m-%d %H:%M:%S",
    errors="coerce",
)
del s




## === cell 17
train = train.dropna(subset=["pickup_datetime"]).copy()

test["pickup_datetime"] = test["pickup_datetime"].fillna(
    pd.Timestamp("2010-01-01 00:00:00")
)

train_dt = train["pickup_datetime"].dt
test_dt = test["pickup_datetime"].dt

train["hour"] = train_dt.hour.astype("int16")
train["weekday"] = train_dt.weekday.astype("int16")
train["month"] = train_dt.month.astype("int16")
train["year"] = train_dt.year.astype("int16")

test["hour"] = test_dt.hour.astype("int16")
test["weekday"] = test_dt.weekday.astype("int16")
test["month"] = test_dt.month.astype("int16")
test["year"] = test_dt.year.astype("int16")

NYC_LON = -73.985428
NYC_LAT = 40.748817

for df in (train, test):
    df["pickup_longitude_c"] = (df["pickup_longitude"] - NYC_LON).astype("float32")
    df["dropoff_longitude_c"] = (df["dropoff_longitude"] - NYC_LON).astype("float32")
    df["pickup_latitude_c"] = (df["pickup_latitude"] - NYC_LAT).astype("float32")
    df["dropoff_latitude_c"] = (df["dropoff_latitude"] - NYC_LAT).astype("float32")

del train_dt, test_dt




## === cell 18
test.head()




## === cell 19
pass




## === cell 20
X = train.drop(["key", "fare_amount", "pickup_datetime"], axis=1)
y = train["fare_amount"]




## === cell 21
X.head()




## === cell 22
y.head()




## === cell 23
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)




## === cell 24
test_pred = test.drop(["key", "pickup_datetime"], axis=1)




## === cell 25
lm = LinearRegression(n_jobs=N_JOBS)
lm.fit(X_train, y_train)
print(lm.score(X_train, y_train))
print(lm.score(X_valid, y_valid))




## === cell 26
y_valid_pred = lm.predict(X_valid)
lrmse = np.sqrt(metrics.mean_squared_error(y_valid_pred, y_valid))
print("Linear RMSE (fit on train split, eval on valid split):", lrmse)




## === cell 27
LinearPredictions = lm.predict(test_pred)




## === cell 28
LinearPredictions.size




## === cell 29
linear_submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": LinearPredictions},
    columns=["key", "fare_amount"],
)
linear_submission.head()




## === cell 30
def XGBoost(X_train, X_valid, y_train, y_valid):
    X_train_c = np.ascontiguousarray(X_train.values)
    X_valid_c = np.ascontiguousarray(X_valid.values)
    y_train_c = np.ascontiguousarray(y_train.values)
    y_valid_c = np.ascontiguousarray(y_valid.values)

    dtrain = xgb.DMatrix(X_train_c, label=y_train_c)
    dvalid = xgb.DMatrix(X_valid_c, label=y_valid_c)

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
        "nthread": N_JOBS,
    }

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=2000,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        verbose_eval=200,
    )
    return booster




## === cell 31
xgbm = XGBoost(X_train, X_valid, y_train, y_valid)

X_valid_c = np.ascontiguousarray(X_valid.values)
dvalid = xgb.DMatrix(X_valid_c)
valid_pred = xgbm.predict(dvalid)
vrmse = np.sqrt(metrics.mean_squared_error(valid_pred, y_valid))
print("XGB RMSE (valid):", vrmse)

test_pred_c = np.ascontiguousarray(test_pred.values)
dtest_pred = xgb.DMatrix(test_pred_c)
XGBPredictions = xgbm.predict(dtest_pred)

del X_valid_c, test_pred_c




## === cell 32
XGBPredictions




## === cell 33
XGB_submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": XGBPredictions}, columns=["key", "fare_amount"]
)
XGB_submission.head()




## === cell 34
submission = XGB_submission

submission["fare_amount"] = submission["fare_amount"].clip(lower=0, upper=200)

submission.to_csv("submission.csv", index=False)
print("Wrote submission:", "submission.csv", "rows:", len(submission))
print(submission.head())
