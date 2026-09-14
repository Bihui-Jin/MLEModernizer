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

bayesian-optimization==3.1.0
geopandas==0.14.4
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

3.60475

# 6. Current score

13.01495

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 8.14908) has done: 'I fix the Bayesian Optimization API mismatch causing the first crash by removing the unsupported `acq` argument and updating how the best parameters are retrieved for bayesian-optimization v3.x. Then I ensure the train/test CSV paths resolve in this environment by falling back to `/kaggle/input/...` if `../input/...` is not found, without changing the modeling approach. Finally, I make sure the test set keeps the `key` index through feature engineering and that a valid `submission.csv` with `key,fare_amount` is always written.'
- What this solution (achieved 7.61072) has done: 'I fix the `dist()` function so it works with both scalars (constants like NYC/JFK coords) and pandas Series, which currently crashes feature engineering. Then I ensure `pickup_datetime` is dropped during `transform()` for both train and test so XGBoost only sees numeric features, resolving the DMatrix dtype error. Finally, I make Bayesian Optimization robust by ensuring `dtrain` exists before optimization and by safely falling back to reasonable default params if BO fails, so the script always trains and writes a valid `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 8.14725) has done: 'You’re currently far from the target (7.61 vs 3.60 RMSE; lower is better), so we need a small but meaningful modeling-quality improvement without changing the core approach (same features, same XGBoost training flow). The biggest issue is that you’re evaluating/tuning with a random row split, which leaks near-duplicate rides across folds via shared pickup_datetime patterns and yields weaker generalization; switching to a time-based split and time-based CV should materially reduce test RMSE while keeping the same model/feature logic. I keep Bayesian Optimization and XGBoost exactly as-is, but make BO’s objective use a chronological fold split (TimeSeriesSplit) and also train/validate chronologically. I also ensure we never accidentally train on a non-numeric `key` column by explicitly dropping it inside `transform()` (it’s currently present in the test features path).'
- What this solution (achieved 13.01495) has done: 'The timeout is dominated by repeated XGBoost training inside Bayesian optimization (3 folds × 1200 rounds × 8 BO trials) plus another expensive xgb.cv up to 2000 rounds, and by slow pandas operations (apply/to_numeric, repeated DMatrix construction). I keep the exact model, objective, and evaluation semantics, but remove redundant work by converting features once to contiguous NumPy arrays, reusing DMatrix objects for folds, and letting XGBoost use multiple CPU threads deterministically. I also speed up datetime parsing/feature engineering using vectorized operations and avoid unnecessary copies. These changes are provably equivalent to the existing logic (same data, same features, same training procedure/loops), but with much lower Python overhead and less repeated preprocessing.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)



## === cell 1
TRAIN_PATH = "../input/train.csv"
TEST_PATH = "../input/test.csv"
SAMPLE_PATH = "../input/sample_submission.csv"

if not os.path.exists(TRAIN_PATH):
    TRAIN_PATH = "/kaggle/input/train.csv"
if not os.path.exists(TEST_PATH):
    TEST_PATH = "/kaggle/input/test.csv"
if not os.path.exists(SAMPLE_PATH):
    SAMPLE_PATH = "/kaggle/input/sample_submission.csv"

NROWS = 1_000_000

train_dtypes = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
df = pd.read_csv(
    TRAIN_PATH,
    nrows=NROWS,
    usecols=[1, 2, 3, 4, 5, 6, 7],
    dtype=train_dtypes,
)



## === cell 2
dt_str = df["pickup_datetime"].astype("string").str.slice(0, 16)
df["pickup_datetime"] = pd.to_datetime(
    dt_str, utc=True, format="%Y-%m-%d %H:%M", errors="coerce"
)



## === cell 3
df.head()



## === cell 4
df.dtypes



## === cell 5
test_dtypes = {
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
test = pd.read_csv(TEST_PATH, dtype=test_dtypes).set_index("key")
dt_str_t = test["pickup_datetime"].astype("string").str.slice(0, 16)
test["pickup_datetime"] = pd.to_datetime(
    dt_str_t, utc=True, format="%Y-%m-%d %H:%M", errors="coerce"
)
test.head()



## === cell 6
test.describe(include="all")



## === cell 7
df.dropna(how="any", axis="rows", inplace=True)

mask = df["pickup_longitude"].between(-75, -72.8)
mask &= df["dropoff_longitude"].between(-75, -72.8)
mask &= df["pickup_latitude"].between(40, 42)
mask &= df["dropoff_latitude"].between(40, 42)
mask &= df["passenger_count"].between(0, 7)
mask &= df["fare_amount"].between(0, 250)

mask &= ~(
    (df["pickup_longitude"] == df["dropoff_longitude"])
    & (df["pickup_latitude"] == df["dropoff_latitude"])
)

df = df[mask]




## === cell 8
def dist(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    R = 6371.0  # km

    lat1 = np.radians(np.asarray(pickup_lat, dtype=float))
    lon1 = np.radians(np.asarray(pickup_long, dtype=float))
    lat2 = np.radians(np.asarray(dropoff_lat, dtype=float))
    lon2 = np.radians(np.asarray(dropoff_long, dtype=float))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    return R * c




## === cell 9
def transform(data):
    data = data.copy()

    if "key" in data.columns:
        data = data.drop("key", axis=1)

    dt = data["pickup_datetime"]
    data["hour"] = dt.dt.hour
    data["day"] = dt.dt.day
    data["month"] = dt.dt.month
    data["year"] = dt.dt.year
    data = data.drop("pickup_datetime", axis=1)

    nyc = (-74.0063889, 40.7141667)
    jfk = (-73.7822222222, 40.6441666667)
    ewr = (-74.175, 40.69)
    lgr = (-73.87, 40.77)

    data["distance_to_center"] = dist(
        nyc[1], nyc[0], data["pickup_latitude"], data["pickup_longitude"]
    )
    data["pickup_distance_to_jfk"] = dist(
        jfk[1], jfk[0], data["pickup_latitude"], data["pickup_longitude"]
    )
    data["dropoff_distance_to_jfk"] = dist(
        jfk[1], jfk[0], data["dropoff_latitude"], data["dropoff_longitude"]
    )
    data["pickup_distance_to_ewr"] = dist(
        ewr[1], ewr[0], data["pickup_latitude"], data["pickup_longitude"]
    )
    data["dropoff_distance_to_ewr"] = dist(
        ewr[1], ewr[0], data["dropoff_latitude"], data["dropoff_longitude"]
    )
    data["pickup_distance_to_lgr"] = dist(
        lgr[1], lgr[0], data["pickup_latitude"], data["pickup_longitude"]
    )
    data["dropoff_distance_to_lgr"] = dist(
        lgr[1], lgr[0], data["dropoff_latitude"], data["dropoff_longitude"]
    )

    data["long_dist"] = data["pickup_longitude"] - data["dropoff_longitude"]
    data["lat_dist"] = data["pickup_latitude"] - data["dropoff_latitude"]

    data["dist"] = dist(
        data["pickup_latitude"],
        data["pickup_longitude"],
        data["dropoff_latitude"],
        data["dropoff_longitude"],
    )

    return data


df = transform(df)



## === cell 10
import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error



## === cell 11
df = df.sort_values(["year", "month", "day", "hour"]).reset_index(drop=True)

split_idx = int(len(df) * 0.75)
X_all = df.drop("fare_amount", axis=1)
y_all = df["fare_amount"].to_numpy()

X_train_df = X_all.iloc[:split_idx]
y_train = y_all[:split_idx]
X_test_df = X_all.iloc[split_idx:]
y_test = y_all[split_idx:]
del df, X_all, y_all

X_train_np = X_train_df.to_numpy(dtype=np.float32, copy=False)
X_test_np = X_test_df.to_numpy(dtype=np.float32, copy=False)
X_train_np = np.nan_to_num(X_train_np, nan=0.0, posinf=0.0, neginf=0.0)
X_test_np = np.nan_to_num(X_test_np, nan=0.0, posinf=0.0, neginf=0.0)

dtrain = xgb.DMatrix(X_train_np, label=y_train)
dvalid = xgb.DMatrix(X_test_np, label=y_test)

n_train = X_train_np.shape[0]



## === cell 12
from sklearn.model_selection import TimeSeriesSplit

tss = TimeSeriesSplit(n_splits=3)
fold_indices = [(tr_idx, va_idx) for tr_idx, va_idx in tss.split(np.arange(n_train))]

fold_dmatrices = []
for tr_idx, va_idx in fold_indices:
    dtr = xgb.DMatrix(X_train_np[tr_idx], label=y_train[tr_idx])
    dva = xgb.DMatrix(X_train_np[va_idx], label=y_train[va_idx])
    fold_dmatrices.append((dtr, dva, va_idx))


def xgb_evaluate(max_depth, gamma, colsample_bytree):
    params = {
        "eval_metric": "rmse",
        "max_depth": int(max_depth),
        "subsample": 0.8,
        "eta": 0.1,
        "gamma": float(gamma),
        "colsample_bytree": float(colsample_bytree),
        "objective": "reg:squarederror",
        "verbosity": 0,
        "seed": 42,
        "nthread": max(1, (os.cpu_count() or 2) - 1),
    }

    rmses = []
    for dtr, dva, va_idx in fold_dmatrices:
        bst = xgb.train(
            params,
            dtr,
            num_boost_round=1200,
            evals=[(dva, "valid")],
            verbose_eval=False,
        )
        pred = bst.predict(dva)
        rmse = float(np.sqrt(mean_squared_error(y_train[va_idx], pred)))
        rmses.append(rmse)

    return -1.0 * float(np.mean(rmses))




## === cell 13
xgb_bo = BayesianOptimization(
    f=xgb_evaluate,
    pbounds={"max_depth": (3, 7), "gamma": (0, 1), "colsample_bytree": (0.3, 0.9)},
    random_state=42,
    verbose=0,
)

bo_ok = True
try:
    xgb_bo.maximize(init_points=3, n_iter=5)
except Exception as e:
    print("Bayesian optimization failed; using fallback params. Error:", repr(e))
    bo_ok = False



## === cell 14
if (
    bo_ok
    and getattr(xgb_bo, "max", None)
    and isinstance(xgb_bo.max, dict)
    and ("params" in xgb_bo.max)
):
    params = dict(xgb_bo.max["params"])
    params["max_depth"] = int(params["max_depth"])
else:
    params = {"max_depth": 6, "gamma": 0.0, "colsample_bytree": 0.7}

params["subsample"] = 0.8
params["eta"] = 0.1
params["eval_metric"] = "rmse"
params["objective"] = "reg:squarederror"
params["verbosity"] = 0
params["seed"] = 42
params["nthread"] = max(1, (os.cpu_count() or 2) - 1)

folds = [(tr, va) for tr, va in fold_indices]
cv_res = xgb.cv(
    params,
    xgb.DMatrix(X_train_np, label=y_train),
    num_boost_round=2000,
    folds=folds,
    seed=42,
    verbose_eval=False,
)

best_num_boost_round = int(np.argmin(cv_res["test-rmse-mean"].values) + 1)



## === cell 15
model2 = xgb.train(
    params,
    dtrain,
    num_boost_round=best_num_boost_round,
)

y_pred = model2.predict(dvalid)
y_train_pred = model2.predict(dtrain)

print("Holdout RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("Train RMSE:", np.sqrt(mean_squared_error(y_train, y_train_pred)))
print("best_num_boost_round:", best_num_boost_round)



## === cell 16
import matplotlib.pyplot as plt

fscores = pd.DataFrame(
    {"X": list(model2.get_fscore().keys()), "Y": list(model2.get_fscore().values())}
)
fscores.sort_values(by="Y").plot.bar(x="X")



## === cell 17
test_feat = transform(test)
test_np = test_feat.to_numpy(dtype=np.float32, copy=False)
test_np = np.nan_to_num(test_np, nan=0.0, posinf=0.0, neginf=0.0)

dtest = xgb.DMatrix(test_np)
y_pred_test = model2.predict(dtest)

y_pred_test = np.clip(y_pred_test, 0, None)

submission = pd.DataFrame({"key": test.index, "fare_amount": y_pred_test})
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
