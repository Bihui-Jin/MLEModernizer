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

12.07035

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 8.14908) has done: 'I fix the Bayesian Optimization API mismatch causing the first crash by removing the unsupported `acq` argument and updating how the best parameters are retrieved for bayesian-optimization v3.x. Then I ensure the train/test CSV paths resolve in this environment by falling back to `/kaggle/input/...` if `../input/...` is not found, without changing the modeling approach. Finally, I make sure the test set keeps the `key` index through feature engineering and that a valid `submission.csv` with `key,fare_amount` is always written.'
- What this solution (achieved 7.61072) has done: 'I fix the `dist()` function so it works with both scalars (constants like NYC/JFK coords) and pandas Series, which currently crashes feature engineering. Then I ensure `pickup_datetime` is dropped during `transform()` for both train and test so XGBoost only sees numeric features, resolving the DMatrix dtype error. Finally, I make Bayesian Optimization robust by ensuring `dtrain` exists before optimization and by safely falling back to reasonable default params if BO fails, so the script always trains and writes a valid `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 8.14725) has done: 'You’re currently far from the target (7.61 vs 3.60 RMSE; lower is better), so we need a small but meaningful modeling-quality improvement without changing the core approach (same features, same XGBoost training flow). The biggest issue is that you’re evaluating/tuning with a random row split, which leaks near-duplicate rides across folds via shared pickup_datetime patterns and yields weaker generalization; switching to a time-based split and time-based CV should materially reduce test RMSE while keeping the same model/feature logic. I keep Bayesian Optimization and XGBoost exactly as-is, but make BO’s objective use a chronological fold split (TimeSeriesSplit) and also train/validate chronologically. I also ensure we never accidentally train on a non-numeric `key` column by explicitly dropping it inside `transform()` (it’s currently present in the test features path).'
- What this solution (achieved 13.01495) has done: 'The timeout is dominated by repeated XGBoost training inside Bayesian optimization (3 folds × 1200 rounds × 8 BO trials) plus another expensive xgb.cv up to 2000 rounds, and by slow pandas operations (apply/to_numeric, repeated DMatrix construction). I keep the exact model, objective, and evaluation semantics, but remove redundant work by converting features once to contiguous NumPy arrays, reusing DMatrix objects for folds, and letting XGBoost use multiple CPU threads deterministically. I also speed up datetime parsing/feature engineering using vectorized operations and avoid unnecessary copies. These changes are provably equivalent to the existing logic (same data, same features, same training procedure/loops), but with much lower Python overhead and less repeated preprocessing.'
- What this solution (achieved 11.60786) has done: 'The timeout is dominated by repeatedly training large XGBoost models: BayesianOptimization triggers 3-fold training per evaluation (8 evaluations → 24 trainings at 1200 rounds each), then `xgb.cv` runs up to 2000 rounds again. To keep identical logic/semantics but cut runtime, I (1) cache expensive feature computations (especially repeated `dist()` trig terms) via a vectorized haversine core that reuses radians, (2) enable XGBoost histogram + external-memory DMatrix (on-disk cache) and reuse one shared base DMatrix to build fold DMatrices cheaply, and (3) make CV/BO stop at the best iteration using XGBoost’s built-in early stopping (no change to objective/metric; it simply avoids wasted extra rounds beyond the optimum). These changes preserve the model family, features, loss, and evaluation approach, but remove redundant work that was causing the 10+ minute overrun.'
- What this solution (achieved 11.60786) has done: 'Your current RMSE (11.61) is far worse than the target (3.60), so the smallest score-improving change is to fix a key schema bug: you accidentally *did not read the `key` and `pickup_datetime` columns from train* (wrong `usecols`), yet later you rely on `pickup_datetime` for feature engineering; this mismatch forces broken/degenerate features and hurts generalization badly. I minimally change the train `read_csv` to include `key` and `pickup_datetime` (and keep `fare_amount`) with appropriate dtypes, and keep everything else (feature engineering, XGBoost, BO/CV flow, submission writing) the same. I also make the post-train diagnostic `dvalid` consistent with the holdout split (train-on-full vs validate-on-holdout was mixing domains), without changing the final model used for test predictions. These changes should materially reduce RMSE while preserving your core approach and runtime constraints.'
- What this solution (achieved 11.43011) has done: 'Your current RMSE (11.61) is much worse than the 3.60 target (lower is better), and the biggest driver is that the model is being trained on “raw” degrees plus a few distances, which tends to generalize poorly on this competition unless you add the standard coordinate interactions. I keep the same overall pipeline (same transform→XGBoost→BO/CV→train full→predict) but make a minimal, metric-aligned feature tweak inside `transform()` by adding the commonly used `abs` deltas and a simple `manhattan` distance (built from your existing haversine helper), which typically yields a large RMSE drop without changing the modeling approach. I also ensure feature column order is identical between train and test (a subtle but important correctness issue when using `.to_numpy()` on two separately-constructed DataFrames). These changes are small, legitimate, and directly aimed at reducing RMSE toward your target while keeping runtime within the same envelope.'
- What this solution (achieved 11.43011) has done: 'You’re far worse than the target (11.43 vs 3.60 RMSE; lower is better), so we need a small but meaningful quality improvement while keeping the same XGBoost + feature-engineering pipeline. The biggest low-risk gain here is to stop converting missing/invalid datetimes into “all-zero time features” (via `nan_to_num`), because that creates a large, non-physical cluster that degrades generalization; instead we impute missing `pickup_datetime` using the median timestamp from the training chunk (for both train and test) so hour/day/month/year remain realistic. We also ensure the time features are explicitly cast to numeric (float32) before NumPy conversion to avoid any dtype edge-cases, and we keep everything else (BO, CV, training, submission format) unchanged. These changes are minimal, metric-aligned, and should move RMSE down toward your target without altering the core modeling approach.'
- What this solution (achieved 12.07035) has done: 'Your current RMSE (11.43) is far above the target (3.60, lower is better), so the smallest likely win is to stop training on a year-stratified sampled subset that breaks the true time-order distribution while you simultaneously use a time-based split/CV. I keep the same XGBoost+BayesianOpt+TimeSeriesSplit pipeline and the same feature set, but change the subsampling step to take one contiguous, chronologically-sorted tail chunk (most recent rows) before the time split, which aligns training/validation with the competition’s temporal patterns and usually reduces RMSE materially. I also make the final “diagnostic holdout RMSE” consistent by evaluating with `best_iteration` (like you already do inside BO), avoiding a misleading metric during development without changing the test predictions logic. Everything else (paths, features, model objective/metric, submission schema) stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved 12.07035) has done: 'Your RMSE (12.07, lower-is-better) is far above the 3.60 target, so we need a small, legitimate improvement that doesn’t change the core XGBoost/feature-engineering/training flow. The biggest correctness issue hurting generalization is that time features are being generated from a UTC-converted timestamp; this competition’s rides are in NYC local time, and using UTC shifts hour/day patterns and degrades RMSE. I minimally adjust datetime parsing to be timezone-naive NYC local time (no UTC conversion) for both train and test, while keeping the same time-based split/CV, features, XGBoost objective/metric, and submission writing. This should reduce RMSE toward your target without altering the model family or training semantics.'

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

NROWS_READ = 5_000_000  # read a larger contiguous chunk
NROWS_TRAIN = 1_000_000  # keep training size similar for runtime stability

train_dtypes = {
    "key": "string",
    "pickup_datetime": "string",
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
df = pd.read_csv(
    TRAIN_PATH,
    nrows=NROWS_READ,
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
    dtype=train_dtypes,
)



## === cell 2
dt_str = df["pickup_datetime"].astype(str).str.slice(0, 16)
df["pickup_datetime"] = pd.to_datetime(
    dt_str, utc=False, format="%Y-%m-%d %H:%M", errors="coerce"
)
train_dt_median = df["pickup_datetime"].dropna().median()
if pd.isna(train_dt_median):
    train_dt_median = pd.Timestamp("2010-01-01")
df["pickup_datetime"] = df["pickup_datetime"].fillna(train_dt_median)



## === cell 3
df.head()



## === cell 4
df.dtypes



## === cell 5
test_dtypes = {
    "key": "string",
    "pickup_datetime": "string",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
test = pd.read_csv(TEST_PATH, dtype=test_dtypes).set_index("key")

dt_str_t = test["pickup_datetime"].astype(str).str.slice(0, 16)
test["pickup_datetime"] = pd.to_datetime(
    dt_str_t, utc=False, format="%Y-%m-%d %H:%M", errors="coerce"
)
test["pickup_datetime"] = test["pickup_datetime"].fillna(train_dt_median)
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

df = df[mask].copy()

df = df.sort_values("pickup_datetime").reset_index(drop=True)
if len(df) > NROWS_TRAIN:
    df = df.iloc[-NROWS_TRAIN:].reset_index(drop=True)




## === cell 8
def _haversine_from_radians(
    lat1_r, lon1_r, lat2_r, lon2_r, cos_lat1=None, cos_lat2=None
):
    R = 6371.0  # km
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    if cos_lat1 is None:
        cos_lat1 = np.cos(lat1_r)
    if cos_lat2 is None:
        cos_lat2 = np.cos(lat2_r)
    a = np.sin(dlat / 2.0) ** 2 + cos_lat1 * cos_lat2 * (np.sin(dlon / 2.0) ** 2)
    c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    return R * c


def dist(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    lat1 = np.radians(np.asarray(pickup_lat, dtype=float))
    lon1 = np.radians(np.asarray(pickup_long, dtype=float))
    lat2 = np.radians(np.asarray(dropoff_lat, dtype=float))
    lon2 = np.radians(np.asarray(dropoff_long, dtype=float))
    return _haversine_from_radians(lat1, lon1, lat2, lon2)




## === cell 9
def transform(data):
    data = data.copy()

    if "key" in data.columns:
        data = data.drop("key", axis=1)

    dt = data["pickup_datetime"]

    data["hour"] = dt.dt.hour.astype("float32")
    data["day"] = dt.dt.day.astype("float32")
    data["month"] = dt.dt.month.astype("float32")
    data["year"] = dt.dt.year.astype("float32")
    data = data.drop("pickup_datetime", axis=1)

    p_lat = data["pickup_latitude"].to_numpy(dtype=np.float64, copy=False)
    p_lon = data["pickup_longitude"].to_numpy(dtype=np.float64, copy=False)
    d_lat = data["dropoff_latitude"].to_numpy(dtype=np.float64, copy=False)
    d_lon = data["dropoff_longitude"].to_numpy(dtype=np.float64, copy=False)

    p_lat_r = np.radians(p_lat)
    p_lon_r = np.radians(p_lon)
    d_lat_r = np.radians(d_lat)
    d_lon_r = np.radians(d_lon)

    cos_p_lat = np.cos(p_lat_r)
    cos_d_lat = np.cos(d_lat_r)

    nyc = (-74.0063889, 40.7141667)
    jfk = (-73.7822222222, 40.6441666667)
    ewr = (-74.175, 40.69)
    lgr = (-73.87, 40.77)

    nyc_lat_r, nyc_lon_r = np.radians(nyc[1]), np.radians(nyc[0])
    jfk_lat_r, jfk_lon_r = np.radians(jfk[1]), np.radians(jfk[0])
    ewr_lat_r, ewr_lon_r = np.radians(ewr[1]), np.radians(ewr[0])
    lgr_lat_r, lgr_lon_r = np.radians(lgr[1]), np.radians(lgr[0])

    cos_nyc = np.cos(nyc_lat_r)
    cos_jfk = np.cos(jfk_lat_r)
    cos_ewr = np.cos(ewr_lat_r)
    cos_lgr = np.cos(lgr_lat_r)

    data["distance_to_center"] = _haversine_from_radians(
        nyc_lat_r, nyc_lon_r, p_lat_r, p_lon_r, cos_lat1=cos_nyc, cos_lat2=cos_p_lat
    )
    data["pickup_distance_to_jfk"] = _haversine_from_radians(
        jfk_lat_r, jfk_lon_r, p_lat_r, p_lon_r, cos_lat1=cos_jfk, cos_lat2=cos_p_lat
    )
    data["dropoff_distance_to_jfk"] = _haversine_from_radians(
        jfk_lat_r, jfk_lon_r, d_lat_r, d_lon_r, cos_lat1=cos_jfk, cos_lat2=cos_d_lat
    )
    data["pickup_distance_to_ewr"] = _haversine_from_radians(
        ewr_lat_r, ewr_lon_r, p_lat_r, p_lon_r, cos_lat1=cos_ewr, cos_lat2=cos_p_lat
    )
    data["dropoff_distance_to_ewr"] = _haversine_from_radians(
        ewr_lat_r, ewr_lon_r, d_lat_r, d_lon_r, cos_lat1=cos_ewr, cos_lat2=cos_d_lat
    )
    data["pickup_distance_to_lgr"] = _haversine_from_radians(
        lgr_lat_r, lgr_lon_r, p_lat_r, p_lon_r, cos_lat1=cos_lgr, cos_lat2=cos_p_lat
    )
    data["dropoff_distance_to_lgr"] = _haversine_from_radians(
        lgr_lat_r, lgr_lon_r, d_lat_r, d_lon_r, cos_lat1=cos_lgr, cos_lat2=cos_d_lat
    )

    data["long_dist"] = data["pickup_longitude"] - data["dropoff_longitude"]
    data["lat_dist"] = data["pickup_latitude"] - data["dropoff_latitude"]

    data["abs_long_dist"] = np.abs(data["long_dist"])
    data["abs_lat_dist"] = np.abs(data["lat_dist"])

    data["dist"] = _haversine_from_radians(
        p_lat_r, p_lon_r, d_lat_r, d_lon_r, cos_lat1=cos_p_lat, cos_lat2=cos_d_lat
    )

    data["manhattan_dist"] = _haversine_from_radians(
        p_lat_r, p_lon_r, p_lat_r, d_lon_r, cos_lat1=cos_p_lat, cos_lat2=cos_p_lat
    ) + _haversine_from_radians(
        p_lat_r, d_lon_r, d_lat_r, d_lon_r, cos_lat1=cos_p_lat, cos_lat2=cos_d_lat
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
del X_all, y_all

X_train_np = X_train_df.to_numpy(dtype=np.float32, copy=False)
X_test_np = X_test_df.to_numpy(dtype=np.float32, copy=False)
X_train_np = np.nan_to_num(X_train_np, nan=0.0, posinf=0.0, neginf=0.0)
X_test_np = np.nan_to_num(X_test_np, nan=0.0, posinf=0.0, neginf=0.0)

NTHREAD = max(1, (os.cpu_count() or 2) - 1)
dtrain = xgb.DMatrix(X_train_np, label=y_train, nthread=NTHREAD)
dholdout = xgb.DMatrix(X_test_np, label=y_test, nthread=NTHREAD)

n_train = X_train_np.shape[0]



## === cell 12
from sklearn.model_selection import TimeSeriesSplit

tss = TimeSeriesSplit(n_splits=3)
fold_indices = [(tr_idx, va_idx) for tr_idx, va_idx in tss.split(np.arange(n_train))]

fold_dmatrices = []
for tr_idx, va_idx in fold_indices:
    dtr = dtrain.slice(tr_idx)
    dva = dtrain.slice(va_idx)
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
        "nthread": NTHREAD,
        "tree_method": "hist",
    }

    rmses = []
    for dtr, dva, va_idx in fold_dmatrices:
        bst = xgb.train(
            params,
            dtr,
            num_boost_round=1200,
            evals=[(dva, "valid")],
            verbose_eval=False,
            early_stopping_rounds=50,
        )
        pred = bst.predict(dva, iteration_range=(0, bst.best_iteration + 1))
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
params["nthread"] = NTHREAD
params["tree_method"] = "hist"

folds = [(tr, va) for tr, va in fold_indices]

cv_res = xgb.cv(
    params,
    dtrain,
    num_boost_round=2000,
    folds=folds,
    seed=42,
    verbose_eval=False,
    early_stopping_rounds=50,
)

best_num_boost_round = int(len(cv_res))



## === cell 15
X_full = df.drop("fare_amount", axis=1).to_numpy(dtype=np.float32, copy=False)
y_full = df["fare_amount"].to_numpy()
X_full = np.nan_to_num(X_full, nan=0.0, posinf=0.0, neginf=0.0)

dfull = xgb.DMatrix(X_full, label=y_full, nthread=NTHREAD)

model2 = xgb.train(
    params,
    dfull,
    num_boost_round=best_num_boost_round,
)

y_pred = model2.predict(dholdout, iteration_range=(0, best_num_boost_round))
y_train_pred = model2.predict(dtrain, iteration_range=(0, best_num_boost_round))

print("Holdout RMSE (diagnostic):", np.sqrt(mean_squared_error(y_test, y_pred)))
print("Train RMSE (diagnostic):", np.sqrt(mean_squared_error(y_train, y_train_pred)))
print("best_num_boost_round:", best_num_boost_round)



## === cell 16
import matplotlib.pyplot as plt

fscores = pd.DataFrame(
    {"X": list(model2.get_fscore().keys()), "Y": list(model2.get_fscore().values())}
)
fscores.sort_values(by="Y").plot.bar(x="X")



## === cell 17
test_feat = transform(test)

train_feature_cols = df.drop(columns=["fare_amount"]).columns
test_feat = test_feat.reindex(columns=train_feature_cols)

test_np = test_feat.to_numpy(dtype=np.float32, copy=False)
test_np = np.nan_to_num(test_np, nan=0.0, posinf=0.0, neginf=0.0)

dtest = xgb.DMatrix(test_np, nthread=NTHREAD)
y_pred_test = model2.predict(dtest, iteration_range=(0, best_num_boost_round))

y_pred_test = np.clip(y_pred_test, 0, None)

submission = pd.DataFrame({"key": test.index, "fare_amount": y_pred_test})
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
