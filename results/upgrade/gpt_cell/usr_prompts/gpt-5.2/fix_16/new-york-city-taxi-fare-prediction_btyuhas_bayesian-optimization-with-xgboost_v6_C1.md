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

3.24593

# 6. Current score

4.84824

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 9.55674) has done: 'Diagnosis: The crash happens because `bayesian-optimization==3.1.0` stores optimization results in `xgb_bo.res` as a list of dicts (one per evaluation), not as a dict keyed by `'max'`. Indexing that list with the string `'max'` raises `TypeError: list indices must be integers or slices, not str`. We should retrieve the best parameters using the library’s public API (`xgb_bo.max`) and then read its `'params'` field, which matches the expected structure for downstream training. We must also keep `max_depth` cast to `int` as before.

Patch summary: In cell 10, replace the incorrect access `xgb_bo.res['max']['max_params']` with `xgb_bo.max['params']` to be compatible with the installed bayesian-optimization version, and keep the existing `max_depth` integer conversion.

Updated cells: Only cell 10 is changed.

Compatibility notes for cell k+1: `params` remains a dict of XGBoost hyperparameters and is still passed unchanged into `xgb.train(params, dtrain, ...)` in cell 11; only the retrieval method is fixed.

Assumptions: `xgb_bo.max` exists after `xgb_bo.maximize(...)` and contains a dict with key `'params'` (standard in bayesian-optimization 3.x).'
- What this solution (achieved 9.35868) has done: 'Your score gap is large (9.55674 vs target 3.24593; lower is better), so we need a small set of high-impact fixes that keep your overall XGBoost+BayesianOptimization approach intact. The biggest issue is that your final training uses only the tuned params but silently drops key fixed params used during CV (eval_metric/eta/subsample), and it also omits `objective`, so training can be inconsistent with CV and underperform. I (1) make the train/valid split deterministic for stability, (2) ensure the final `params` merges the tuned values with the same base params as CV plus a proper regression objective, and (3) add `random_state/seed` in XGBoost to reduce variance; everything else (features, model family, training flow, num_boost_round) stays the same and it still writes `submission.csv`.'
- What this solution (achieved 9.12088) has done: 'Your score (9.35868 RMSE; lower is better) is far from the target (3.24593), so the most direct way to move toward the target without changing the overall XGBoost+BayesianOptimization approach is to fix a key training mismatch: you’re tuning with CV but then training only on a partial split (and predicting on a holdout), which wastes data and inflates error. I keep the same feature engineering, model family, BO loop, and boosting rounds, but (1) create a proper `dvalid` with labels and use it in CV (so `xgb.cv` is evaluating the intended labels), and (2) after tuning, train the final model on **all** filtered training data (train+valid) before predicting test. This preserves your core logic and evaluation semantics while typically giving a large RMSE improvement by using the full dataset for final fit and ensuring CV is computed correctly.'
- What this solution (achieved 8.11071) has done: 'Your RMSE is far above target (9.12 vs 3.25; lower is better), so we need a high-impact fix without changing the overall XGBoost+BayesianOptimization + 50 boosting rounds core flow. The biggest issue is that your “distance” features use a Manhattan delta in degrees, which is a very weak proxy for real travel distance; swapping this to a proper haversine distance keeps the same feature slots/logic (still “a distance function”) but materially improves signal. I also clip negative predictions to 0 (fares can’t be negative), which usually reduces RMSE on this competition without altering the model. Everything else (data size, filtering, BO loop, train/cv approach, training rounds, submission format/path) stays the same.'
- What this solution (achieved 7.95116) has done: 'Your current RMSE (8.11) is still far above the target (3.24593; lower is better), so the smallest high-impact improvement that preserves your XGBoost+BO+50-round core logic is to add one standard, cheap feature: haversine “trip distance” was already added, but the model still lacks an interaction capturing directionality and airport/center effects cleanly. I keep the same training flow and hyperparameter tuning, but (1) add `abs_long_dist`/`abs_lat_dist` (same raw deltas, just magnitude) and a `bearing` feature derived from pickup/dropoff coordinates, and (2) include `min_child_weight` in BayesianOptimization bounds (still XGBoost, same objective/rounds/CV), which typically reduces RMSE materially on this dataset with minimal risk. I also ensure the new features are applied identically to train and test and keep the submission format unchanged.'
- What this solution (achieved 5.13525) has done: 'Your current RMSE (7.951) is far above the target (3.246; lower is better), so we need a small but high-impact improvement without changing your overall XGBoost+BayesianOptimization+50-round setup. The biggest missing signal is that taxi fares scale strongly with trip distance and time, and XGBoost benefits from a more linearized target; we can keep the same model/training loop but train on `log1p(fare_amount)` and invert with `expm1` at prediction time (still optimizing RMSE on the original scale, just better conditioned). Additionally, the single `distance_to_center` currently uses only pickup-to-center; adding a matching `dropoff_distance_to_center` and a simple `euclidean_degree` distance is minimal feature expansion that tends to materially reduce error on this dataset. All changes are applied identically to train/test and the script still writes a valid `submission.csv`.'
- What this solution (achieved 5.35765) has done: 'You’re still well above the target RMSE (5.135 vs 3.246; lower is better), so we should make one high-impact but minimal change that keeps the exact XGBoost+BO+50-round flow intact. The biggest remaining mismatch is that you’re training on the *log1p* target but tuning/evaluating (via `xgb.cv`) on the log scale, which doesn’t align with Kaggle’s RMSE on the original dollar scale; we instead evaluate BO on a small holdout using original-scale RMSE after `expm1`, while keeping the same parameters and boosting rounds. This only changes the *evaluation function* used for choosing hyperparameters; the model architecture/training loop remains XGBoost with 50 rounds and the same features. I also set `nthread` for stability/speed and keep the submission writing unchanged.'
- What this solution (achieved 4.93573) has done: 'We keep your XGBoost + BayesianOptimization + 50-round training exactly as-is, but make two small, high-impact adjustments that usually reduce RMSE on this competition without changing the core approach. First, we add a couple of standard “time” features (weekday and a weekend flag) derived from `pickup_datetime`, which are cheap and often materially improve fare prediction. Second, we ensure `passenger_count` is filtered to the valid Kaggle range (1–6) instead of allowing 0–8, which removes noisy/out-of-distribution rows and typically improves generalization. Everything else (log1p target, BO loop, params, boosting rounds, submission schema/path) stays the same.'
- What this solution (achieved 4.7379) has done: 'Your current RMSE (4.93573; lower is better) is still far above the target (3.24593), so we should make one high-impact but minimal change that preserves your XGBoost + BayesianOptimization + 50-round setup. The largest remaining gap typically comes from missing location-specific structure: adding a small set of standard “grid/bin” features for pickup/dropoff lat/lon gives XGBoost simple non-linear locality cues without changing the model/training approach. I also apply the same coordinate/`passenger_count`/fare filters to the training slice only (as you already do) while keeping test untouched, and keep the same log1p target + expm1 inversion and submission schema. Everything else (BO loop, bounds, objective/loss, rounds, file paths, output `submission.csv`) stays the same.'
- What this solution (achieved 4.80869) has done: 'We’re still far above the target RMSE (4.7379 vs 3.24593; lower is better), so the smallest high-impact change that preserves your exact XGBoost+BO+50-round core flow is to improve the feature signal using standard NYC taxi “geodesic” components while keeping your existing distance/bearing features intact. Specifically, I add `haversine_lat_km` and `haversine_lon_km` (the north/south and east/west km components of the trip) plus `manhattan_km` (their L1 sum), which typically helps XGBoost model fare structure (base fare + per-km components) without changing model/training logic. I also add a very small numerical stability fix by casting all feature columns to `float32` right before creating each DMatrix (train/test), which can reduce inconsistencies between train/test dtypes and marginally improve generalization. Everything else (data sampling size, filtering, log1p target + expm1 inversion, BO loop, params, 50 boosting rounds, submission schema) remains the same and it still writes `submission.csv`.'
- What this solution (achieved 15.25184) has done: 'We’re far above the target RMSE (4.80869 vs 3.24593; lower is better), so we need a small, high-impact change that preserves your XGBoost+BayesianOptimization+50-round setup and existing feature set. The biggest remaining issue is that the model is currently learning with plain squared error on the log1p target, which tends to underweight large-fare errors; switching the XGBoost objective to `reg:pseudohubererror` (still a regression objective and still evaluated by Kaggle RMSE after expm1) often improves robustness and reduces overall RMSE on this competition without changing the training loop or feature engineering. I keep everything else the same (same data sample size, filters, BO loop, bounds, rounds, and submission format), and I apply the objective consistently in both BO evaluation and final training. This should move the score downward toward the target while keeping changes minimal and stable.'
- What this solution (achieved 4.96171) has done: 'Your current score (15.25 RMSE) is much worse than both the target (3.24593) and your previously achieved ~4.8–5.1, and the only major “recent” change that plausibly caused a regression is switching the XGBoost objective to `reg:pseudohubererror`. To move back toward the target with minimal disruption, I revert the objective to standard squared error (`reg:squarederror`) while keeping your log1p target strategy, the same feature engineering, the same BO loop, and the same 50 boosting rounds. I also add `tree_method="hist"` to improve training stability/speed on Kaggle CPU without changing semantics. Everything else (filters, splits, BO bounds, prediction inversion, clipping, submission writing) remains the same.'
- What this solution (achieved 4.84772) has done: 'To move your RMSE down toward the 3.24593 target without changing the overall XGBoost+BayesianOptimization+50-round/log1p approach, I’m making two small, high-impact adjustments that preserve the same model family and training loop. First, I include `reg_lambda` (L2) and `reg_alpha` (L1) in Bayesian Optimization bounds and use them in training; light regularization is often the missing piece once you already have strong distance/time/grid features, and it typically improves generalization (lower Kaggle RMSE). Second, I increase `nthread` to use available CPU cores (stability/speed) while keeping everything else identical, and I keep the same submission writing logic and schema.'
- What this solution (achieved 4.84824) has done: 'Your current RMSE (4.84772; lower is better) is still well above the 3.24593 target, so we should make one minimal, high-impact improvement without changing your overall XGBoost+BO+50-round/log1p approach. The biggest remaining generalization gain typically comes from reducing bias in the BO evaluation split: switching from a single fixed holdout to a small K-fold CV objective (still evaluating on original-dollar RMSE after expm1) makes Bayesian Optimization pick more robust hyperparameters. This keeps the same feature set, same model family, same boosting rounds, and same loss/objective; it only changes how we *score* candidate params during tuning. Everything else (final training on all data, prediction inversion/clipping, and writing `submission.csv`) stays the same.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



## === cell 1
df = pd.read_csv("../input/train.csv", nrows=500_000, usecols=[1, 2, 3, 4, 5, 6, 7])



## === cell 2
df["pickup_datetime"] = df["pickup_datetime"].str.slice(0, 16)
df["pickup_datetime"] = pd.to_datetime(
    df["pickup_datetime"], utc=True, format="%Y-%m-%d %H:%M"
)



## === cell 3
df.dropna(how="any", axis="rows", inplace=True)

mask = df["pickup_longitude"].between(-75, -73)
mask &= df["dropoff_longitude"].between(-75, -73)
mask &= df["pickup_latitude"].between(40, 42)
mask &= df["dropoff_latitude"].between(40, 42)

mask &= df["passenger_count"].between(1, 6)

mask &= df["fare_amount"].between(0, 250)

df = df[mask]




## === cell 4
def dist(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    R = 6371.0  # km
    lat1 = np.radians(pickup_lat)
    lon1 = np.radians(pickup_long)
    lat2 = np.radians(dropoff_lat)
    lon2 = np.radians(dropoff_long)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(a))
    return R * c




## === cell 5
def bearing(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    lat1 = np.radians(pickup_lat)
    lon1 = np.radians(pickup_long)
    lat2 = np.radians(dropoff_lat)
    lon2 = np.radians(dropoff_long)

    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    brng = np.arctan2(y, x)  # radians, [-pi, pi]
    return brng




## === cell 6
def transform(data):
    data["hour"] = data["pickup_datetime"].dt.hour
    data["day"] = data["pickup_datetime"].dt.day
    data["month"] = data["pickup_datetime"].dt.month
    data["year"] = data["pickup_datetime"].dt.year

    data["weekday"] = data["pickup_datetime"].dt.weekday
    data["is_weekend"] = (data["weekday"] >= 5).astype(np.int8)

    bin_size = 0.01  # ~1.1km latitude; minimal additional features, strong signal
    data["pickup_lat_bin"] = np.floor(data["pickup_latitude"] / bin_size).astype(
        np.int32
    )
    data["pickup_lon_bin"] = np.floor(data["pickup_longitude"] / bin_size).astype(
        np.int32
    )
    data["dropoff_lat_bin"] = np.floor(data["dropoff_latitude"] / bin_size).astype(
        np.int32
    )
    data["dropoff_lon_bin"] = np.floor(data["dropoff_longitude"] / bin_size).astype(
        np.int32
    )

    data = data.drop("pickup_datetime", axis=1)

    nyc = (-74.0063889, 40.7141667)
    jfk = (-73.7822222222, 40.6441666667)
    ewr = (-74.175, 40.69)
    lgr = (-73.87, 40.77)

    data["distance_to_center"] = dist(
        nyc[1], nyc[0], data["pickup_latitude"], data["pickup_longitude"]
    )
    data["dropoff_distance_to_center"] = dist(
        nyc[1], nyc[0], data["dropoff_latitude"], data["dropoff_longitude"]
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

    data["abs_long_dist"] = np.abs(data["long_dist"])
    data["abs_lat_dist"] = np.abs(data["lat_dist"])

    data["dist"] = dist(
        data["pickup_latitude"],
        data["pickup_longitude"],
        data["dropoff_latitude"],
        data["dropoff_longitude"],
    )

    data["euclidean_degree"] = np.sqrt(
        data["abs_long_dist"] ** 2 + data["abs_lat_dist"] ** 2
    )

    data["bearing"] = bearing(
        data["pickup_latitude"],
        data["pickup_longitude"],
        data["dropoff_latitude"],
        data["dropoff_longitude"],
    )

    R = 6371.0
    pickup_lat_rad = np.radians(data["pickup_latitude"])
    dropoff_lat_rad = np.radians(data["dropoff_latitude"])
    pickup_lon_rad = np.radians(data["pickup_longitude"])
    dropoff_lon_rad = np.radians(data["dropoff_longitude"])

    dlat = dropoff_lat_rad - pickup_lat_rad
    dlon = dropoff_lon_rad - pickup_lon_rad

    mean_lat = 0.5 * (pickup_lat_rad + dropoff_lat_rad)
    data["haversine_lat_km"] = (R * np.abs(dlat)).astype(np.float32)
    data["haversine_lon_km"] = (R * np.cos(mean_lat) * np.abs(dlon)).astype(np.float32)
    data["manhattan_km"] = (data["haversine_lat_km"] + data["haversine_lon_km"]).astype(
        np.float32
    )

    return data


df = transform(df)



## === cell 7
import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error



## === cell 8
from sklearn.model_selection import train_test_split

df["fare_amount_log1p"] = np.log1p(df["fare_amount"])

X = df.drop(["fare_amount", "fare_amount_log1p"], axis=1)
y = df["fare_amount_log1p"]

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
)

X_train = X_train.astype(np.float32)
X_valid = X_valid.astype(np.float32)
X = X.astype(np.float32)

dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)

dfull = xgb.DMatrix(X, label=y)

del X_train, X_valid, y_train, y_valid, X, y, df



## === cell 9
from sklearn.model_selection import KFold


def xgb_evaluate(
    max_depth, gamma, colsample_bytree, min_child_weight, reg_alpha, reg_lambda
):
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": int(max_depth),
        "subsample": 0.8,
        "eta": 0.1,
        "gamma": gamma,
        "colsample_bytree": colsample_bytree,
        "min_child_weight": min_child_weight,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
        "seed": 42,
        "nthread": -1,
        "tree_method": "hist",
    }

    X_all = dtrain.get_data()
    y_all = dtrain.get_label()

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    rmses = []
    for tr_idx, va_idx in kf.split(X_all):
        dtr = xgb.DMatrix(X_all[tr_idx], label=y_all[tr_idx])
        dva = xgb.DMatrix(X_all[va_idx], label=y_all[va_idx])

        model = xgb.train(params, dtr, num_boost_round=50)
        pred_va_log = model.predict(dva)

        pred_va = np.expm1(pred_va_log)
        true_va = np.expm1(dva.get_label())
        pred_va = np.maximum(pred_va, 0.0)

        rmses.append(np.sqrt(mean_squared_error(true_va, pred_va)))

    rmse = float(np.mean(rmses))
    return -rmse




## === cell 10
xgb_bo = BayesianOptimization(
    xgb_evaluate,
    {
        "max_depth": (3, 7),
        "gamma": (0, 1),
        "colsample_bytree": (0.3, 0.9),
        "min_child_weight": (1, 10),
        "reg_alpha": (0.0, 1.0),
        "reg_lambda": (0.5, 5.0),
    },
)
xgb_bo.maximize(init_points=3, n_iter=5)



## === cell 11
best_params = xgb_bo.max["params"]
best_params["max_depth"] = int(best_params["max_depth"])

params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "subsample": 0.8,
    "eta": 0.1,
    "seed": 42,
    "nthread": -1,
    "tree_method": "hist",
}
params.update(best_params)

model2 = xgb.train(params, dfull, num_boost_round=50)

y_valid_pred_log = model2.predict(dvalid)
y_valid_pred = np.expm1(y_valid_pred_log)
y_valid_true = np.expm1(dvalid.get_label())
y_valid_pred = np.maximum(y_valid_pred, 0.0)
print(
    "Holdout RMSE (original scale):",
    np.sqrt(mean_squared_error(y_valid_true, y_valid_pred)),
)



## === cell 12
import matplotlib.pyplot as plt

fscores = pd.DataFrame(
    {"X": list(model2.get_fscore().keys()), "Y": list(model2.get_fscore().values())}
)
fscores.sort_values(by="Y").plot.bar(x="X")



## === cell 13
test = pd.read_csv("../input/test.csv").set_index("key")
test["pickup_datetime"] = test["pickup_datetime"].str.slice(0, 16)
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], utc=True, format="%Y-%m-%d %H:%M"
)

test = transform(test)

test = test.astype(np.float32)

dtest = xgb.DMatrix(test)

y_pred_test_log = model2.predict(dtest)
y_pred_test = np.expm1(y_pred_test_log)
y_pred_test = np.maximum(y_pred_test, 0.0)



## === cell 14
holdout = pd.DataFrame({"key": test.index, "fare_amount": y_pred_test})
holdout.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", holdout.shape)
