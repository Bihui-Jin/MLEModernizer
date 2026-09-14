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

4.04343

# 6. Current score

6.03907

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.82845) has done: 'Diagnosis: The crash happens because `bayesian-optimization==3.1.0` exposes `xgb_bo.res` as a list of result dicts, not a dict keyed by `'max'`. Indexing it with a string therefore raises `TypeError: list indices must be integers or slices, not str`. The correct way to retrieve the best-found parameters in this version is via `xgb_bo.max['params']` (or by taking `max(xgb_bo.res, key=lambda r: r['target'])['params']`).  

Patch summary: Update cell 10 to fetch best parameters using the current BayesianOptimization API, while keeping the same downstream `params` dict semantics expected by `xgb.train`. Also keep the explicit `int()` cast for `max_depth` to preserve prior behavior.  

Updated cells: Only cell 10 is modified.  

Compatibility notes for cell k+1: Cell 12 expects a `params` dict suitable for `xgb.train`; the patch still produces that dict and preserves `params['max_depth']` as an integer. No variable names or shapes used by cell 12 are changed.  

Assumptions: `xgb_bo.max` is populated after `xgb_bo.maximize(...)` in cell 9 (as is standard for bayesian-optimization 3.x).'
- What this solution (achieved 5.51766) has done: 'You’re currently far worse than the target RMSE (5.83 vs 4.04), so the smallest safe way to move toward the target is to reduce variance and improve feature quality without changing the overall model/training approach. I (1) make the split deterministic, (2) ensure `xgb.cv` actually uses labels (your `X_val` DMatrix currently drops labels and the CV call doesn’t shuffle), (3) add standard NYC-taxi baseline features derived from the same raw columns (longitude/latitude deltas, Manhattan distance, and a simple “to/from airport” flag), and (4) constrain predictions to be positive and reasonable to avoid RMSE blow-ups from rare negatives/outliers. These are minimal additions around your existing feature engineering + XGBoost + Bayesian optimization + `xgb.train` pipeline and should move RMSE downward toward the target band.'
- What this solution (achieved 5.4574) has done: 'Your RMSE (5.52) is still substantially worse than the target (4.04), so we should make small, reliable improvements that reduce error without changing the overall XGBoost + BayesianOptimization pipeline. The biggest safe gain for this competition is to add a few standard time-based features (month, day-of-week, day, and a simple rush-hour flag) using the same `pickup_datetime` you already parse, and include them in both train/test consistently. We also make the XGB training consistent with the CV objective by adding `gamma` and `colsample_bytree` from the optimized params into the final training dict (currently only `max_depth` is ensured and the rest rely on defaults), which is a minimal correctness fix that typically improves generalization. Finally, we keep your clipping (stability) and all core logic intact.'
- What this solution (achieved 6.39624) has done: 'Your current RMSE (5.4574) is still far above the target (4.04343), so we should make small, legitimate improvements that typically reduce error while keeping your XGBoost + BayesianOptimization pipeline intact. The biggest low-risk gain here is to add a couple of standard NYC-taxi location features (pickup/dropoff distance to NYC center + a simple “same borough-ish” proxy via bearing) and to use a slightly larger, but still lightweight, training sample (more rows reduces variance without changing the approach). I also ensure the exact same feature columns are used consistently (as float32) for train/test DMatrix to improve numerical stability and avoid subtle dtype issues. The submission writing stays identical (`submission.csv` with `key,fare_amount`).'
- What this solution (achieved 6.63176) has done: 'Your current RMSE (6.39624) is still much worse than the target (4.04343), so we make the smallest changes that typically reduce error without changing your overall XGBoost+BayesianOptimization approach. The main issue is that your CV optimizes a model with only a subset of the final training hyperparameters (e.g., missing `min_child_weight`, `reg_lambda`, `reg_alpha`, `max_delta_step`), which often yields poorly calibrated params for this noisy regression; we add a few standard, safe regularization/search dimensions and then faithfully carry **all** tuned params into final training. We also make the Bayesian optimization evaluate on a fixed subsample of the training DMatrix to reduce noise/variance in the objective (same logic, just less jitter), and we keep everything deterministic with `nthread`/seeds. These are minimal, metric-aligned tweaks that usually move RMSE down meaningfully toward ~4 without changing feature engineering or the training paradigm.'
- What this solution (achieved 6.36311) has done: 'To move your RMSE down toward the 4.04 target while keeping the same XGBoost + BayesianOptimization core, I make two minimal, high-impact adjustments: (1) include `subsample` and `eta` in the Bayesian search (they’re currently fixed, but tuning them is often the fastest way to reduce error on this dataset), and (2) use early stopping inside `xgb.cv` to select an appropriate number of boosting rounds, then train the final model with that selected `best_iteration` (instead of a fixed 600), which typically reduces overfitting and lowers RMSE. These changes preserve your model type, objective, feature set, and overall training approach, and they keep the same submission format/path. Everything remains deterministic via the existing seed setup.'
- What this solution (achieved 6.72401) has done: 'Your current RMSE (6.36) is far above the target (4.04), so the smallest safe move toward the target is to reduce underfitting from missing key baseline signal while keeping your exact XGBoost+BayesianOptimization training flow intact. I add two standard, lightweight NYC-taxi features (Haversine distance in km and a simple “night” flag) inside your existing `add_features` function and include them in `features`, which typically lowers RMSE materially without changing the model type or loop. I also fix a subtle bug in `pickup_to_center_miles/dropoff_to_center_miles` where lat/lon were passed in the wrong order, which can inject noise into those features and hurt RMSE. Everything else (BayesOpt search, CV with early stopping, final training, clipping, and submission writing) stays the same.'
- What this solution (achieved 6.03907) has done: 'Your RMSE is still far above the 4.04 target, so the smallest reliable move is to reduce obvious noise/outliers in training without changing your XGBoost/BayesOpt/CV/training flow. I add standard NYC Taxi data cleaning filters (valid lat/lon ranges, drop identical pickup/dropoff, and fare-vs-distance consistency bounds) applied after feature creation so your engineered features remain consistent. This usually improves generalization materially on this competition while preserving the same model architecture, loss, Bayesian search, and submission semantics. I also ensure the same DMatrix construction style is used for test (pass `X_test.values`) to avoid any subtle dtype/index issues.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os

print(os.listdir("../input"))
os.chdir("/kaggle/working/")



## === cell 1
df_train = pd.read_csv(
    "../input/train.csv", nrows=200000, parse_dates=["pickup_datetime"]
)
df_train.head()



## === cell 2
df_train.describe()
df_train.dtypes



## === cell 3
df_train = df_train[(df_train["fare_amount"] > 0.05) & (df_train.passenger_count > 0)]
df_train.dropna(how="any", axis="rows", inplace=True)
print("New Size: {}".format(len(df_train)))
df_test = pd.read_csv("../input/test.csv", parse_dates=["pickup_datetime"])



## === cell 4
mask = df_train["pickup_longitude"].between(-75, -73)
mask &= df_train["dropoff_longitude"].between(-75, -73)
mask &= df_train["pickup_latitude"].between(40, 42)
mask &= df_train["dropoff_latitude"].between(40, 42)
mask &= df_train["passenger_count"].between(0, 8)
mask &= df_train["fare_amount"].between(0, 250)

df_train = df_train[mask]




## === cell 5
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295  # Pi/180
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a))  # miles


def add_features(df):
    df = df.copy()

    df["distance_miles"] = distance(
        df.pickup_latitude,
        df.pickup_longitude,
        df.dropoff_latitude,
        df.dropoff_longitude,
    )

    df["distance_km"] = (df["distance_miles"].astype(np.float64) * 1.609344).astype(
        np.float32
    )

    df["year"] = df.pickup_datetime.dt.year
    df["month"] = df.pickup_datetime.dt.month
    df["day"] = df.pickup_datetime.dt.day
    df["dayofweek"] = df.pickup_datetime.dt.dayofweek
    df["hour"] = df.pickup_datetime.dt.hour
    df["rush_hour"] = (
        ((df["hour"].between(7, 9)) | (df["hour"].between(16, 19)))
        & (df["dayofweek"].between(0, 4))
    ).astype(np.int8)

    df["is_night"] = ((df["hour"] <= 5) | (df["hour"] >= 20)).astype(np.int8)

    df["abs_lon_diff"] = (df["pickup_longitude"] - df["dropoff_longitude"]).abs()
    df["abs_lat_diff"] = (df["pickup_latitude"] - df["dropoff_latitude"]).abs()
    df["manhattan_dist"] = df["abs_lon_diff"] + df["abs_lat_diff"]

    jfk_lon, jfk_lat = -73.7781, 40.6413
    lga_lon, lga_lat = -73.8740, 40.7769
    ewr_lon, ewr_lat = -74.1745, 40.6895

    def near(lat, lon, center_lat, center_lon, thr=0.05):
        return ((lat - center_lat).abs() < thr) & ((lon - center_lon).abs() < thr)

    pickup_jfk = near(df["pickup_latitude"], df["pickup_longitude"], jfk_lat, jfk_lon)
    dropoff_jfk = near(
        df["dropoff_latitude"], df["dropoff_longitude"], jfk_lat, jfk_lon
    )
    pickup_lga = near(df["pickup_latitude"], df["pickup_longitude"], lga_lat, lga_lon)
    dropoff_lga = near(
        df["dropoff_latitude"], df["dropoff_longitude"], lga_lat, lga_lon
    )
    pickup_ewr = near(df["pickup_latitude"], df["pickup_longitude"], ewr_lat, ewr_lon)
    dropoff_ewr = near(
        df["dropoff_latitude"], df["dropoff_longitude"], ewr_lat, ewr_lon
    )

    df["airport_trip"] = (
        pickup_jfk | dropoff_jfk | pickup_lga | dropoff_lga | pickup_ewr | dropoff_ewr
    ).astype(np.int8)

    nyc_lon, nyc_lat = -73.985428, 40.748817  # Midtown Manhattan (approx)

    df["pickup_to_center_miles"] = distance(
        df["pickup_latitude"], df["pickup_longitude"], nyc_lat, nyc_lon
    )
    df["dropoff_to_center_miles"] = distance(
        df["dropoff_latitude"], df["dropoff_longitude"], nyc_lat, nyc_lon
    )

    dlon = (df["dropoff_longitude"] - df["pickup_longitude"]).astype(np.float64)
    dlat = (df["dropoff_latitude"] - df["pickup_latitude"]).astype(np.float64)
    bearing = np.arctan2(dlon.values, dlat.values)  # radians
    df["bearing_sin"] = np.sin(bearing).astype(np.float32)
    df["bearing_cos"] = np.cos(bearing).astype(np.float32)

    return df


df_train = add_features(df_train)
df_test = add_features(df_test)

train_mask = df_train["pickup_longitude"].between(-74.3, -72.9)
train_mask &= df_train["dropoff_longitude"].between(-74.3, -72.9)
train_mask &= df_train["pickup_latitude"].between(40.5, 41.8)
train_mask &= df_train["dropoff_latitude"].between(40.5, 41.8)

train_mask &= ~(
    (df_train["abs_lon_diff"] < 1e-6)
    & (df_train["abs_lat_diff"] < 1e-6)
    & (df_train["fare_amount"] > 3.0)
)

train_mask &= df_train["distance_miles"].between(0.01, 100.0)
train_mask &= df_train["fare_amount"] >= (2.5 + 0.5 * df_train["distance_miles"])
train_mask &= df_train["fare_amount"] <= (
    2.5 + 10.0 * df_train["distance_miles"] + 50.0
)

df_train = df_train[train_mask].copy()
print("Size after additional cleaning:", len(df_train))



## === cell 6
features = [
    "year",
    "month",
    "day",
    "dayofweek",
    "hour",
    "rush_hour",
    "is_night",  # added (score)
    "distance_miles",
    "distance_km",  # added (score)
    "passenger_count",
    "abs_lon_diff",
    "abs_lat_diff",
    "manhattan_dist",
    "airport_trip",
    "pickup_to_center_miles",
    "dropoff_to_center_miles",
    "bearing_sin",
    "bearing_cos",
]

X = df_train[features].astype(np.float32).values
y = df_train["fare_amount"].values.astype(np.float32)
X_test = df_test[features].astype(np.float32)
df_test.head(5)



## === cell 7
import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split



## === cell 8
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train)

rng = np.random.RandomState(42)
sub_idx = rng.choice(X_train.shape[0], size=min(60000, X_train.shape[0]), replace=False)
dtrain_sub = xgb.DMatrix(X_train[sub_idx], label=y_train[sub_idx])


def xgb_eva(
    max_depth,
    gamma,
    colsample_bytree,
    min_child_weight,
    reg_lambda,
    reg_alpha,
    subsample,
    eta,
):
    params = {
        "eval_metric": "rmse",
        "max_depth": int(max_depth),
        "subsample": float(subsample),
        "eta": float(eta),
        "gamma": float(gamma),
        "colsample_bytree": float(colsample_bytree),
        "min_child_weight": float(min_child_weight),
        "lambda": float(reg_lambda),
        "alpha": float(reg_alpha),
        "objective": "reg:squarederror",
        "seed": 42,
        "nthread": 4,
    }
    cv_result = xgb.cv(
        params,
        dtrain_sub,
        num_boost_round=2000,
        nfold=3,
        shuffle=True,
        seed=42,
        verbose_eval=False,
        early_stopping_rounds=30,
    )
    return -1.0 * cv_result["test-rmse-mean"].iloc[-1]




## === cell 9
xgb_bo = BayesianOptimization(
    xgb_eva,
    {
        "max_depth": (3, 8),
        "gamma": (0.0, 2.0),
        "colsample_bytree": (0.3, 1.0),
        "min_child_weight": (1.0, 20.0),
        "reg_lambda": (0.0, 10.0),
        "reg_alpha": (0.0, 5.0),
        "subsample": (0.5, 1.0),
        "eta": (0.02, 0.2),
    },
    random_state=42,
)
xgb_bo.maximize(init_points=3, n_iter=7)



## === cell 10
best = dict(xgb_bo.max["params"])
best["max_depth"] = int(best["max_depth"])

params = {
    "eval_metric": "rmse",
    "objective": "reg:squarederror",
    "seed": 42,
    "nthread": 4,
    "max_depth": best["max_depth"],
    "gamma": float(best["gamma"]),
    "colsample_bytree": float(best["colsample_bytree"]),
    "min_child_weight": float(best["min_child_weight"]),
    "lambda": float(best["reg_lambda"]),
    "alpha": float(best["reg_alpha"]),
    "subsample": float(best["subsample"]),
    "eta": float(best["eta"]),
}

cv_final = xgb.cv(
    params,
    xgb.DMatrix(X, label=y),
    num_boost_round=4000,
    nfold=3,
    shuffle=True,
    seed=42,
    verbose_eval=False,
    early_stopping_rounds=30,
)
best_num_boost_round = int(len(cv_final))



## === cell 11
model2 = xgb.train(
    params, xgb.DMatrix(X, label=y), num_boost_round=best_num_boost_round
)

X_testm = xgb.DMatrix(X_test.values)
y_test = model2.predict(X_testm)

y_test = np.clip(y_test, 0.0, 250.0)



## === cell 12
X_test.head(5)



## === cell 13
sub = pd.DataFrame()
sub["key"] = df_test.key
sub["fare_amount"] = y_test
sub.to_csv("submission.csv", index=False)
sub
