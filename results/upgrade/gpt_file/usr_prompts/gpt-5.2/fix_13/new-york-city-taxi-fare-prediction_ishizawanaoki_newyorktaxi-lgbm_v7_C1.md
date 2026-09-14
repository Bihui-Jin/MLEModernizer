# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

3.10

# 3. Installed packages

geopandas==0.14.4
lightgbm==4.6.0
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

3.42426

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 4.91454) has done: 'I fix the pandas `quantile()` failures by restricting quantiles to numeric columns (the `key` and `pickup_datetime` strings were causing the `str - str` error), which is score-neutral. I also fix the feature engineering bug by not attempting to cast the string `key` into float; instead, we drop `key` from modeling features but keep it intact for submission IDs (this is required for a valid submission and avoids the conversion error). Finally, I update the LightGBM training call to be compatible with LightGBM 4.6.0 by replacing deprecated arguments (`verbose_eval`, `early_stopping_rounds`) with callbacks, so training runs and produces predictions. The script always write a valid `submission_lightgbm.csv` with columns `key,fare_amount`.'
- What this solution (achieved 5.68325) has done: 'Your current gap is 4.91454 − 3.42426 = 1.49028 (lower is better), so we need a modest but real improvement without changing the overall approach (LightGBM + KFold). The biggest score win with minimal semantic impact is to fix the feature set: dropping `pickup_datetime` removes an important signal, so we extract a few standard time parts (year/month/day/hour/weekday) and then drop the raw timestamp column. We also add a single, very standard distance feature (haversine) plus absolute deltas, which keeps the same model/training loop but usually reduces RMSE materially for this competition. Finally, we add a lightweight, competition-standard coordinate/fare filter on the sampled training rows to reduce extreme outliers that hurt RMSE, while keeping everything else (CV, params, early stopping, submission format) intact.'
- What this solution (achieved 4.1158) has done: 'Your current RMSE (5.68325) is worse than the target (3.42426), so we should make small, high-impact fixes that keep the same LightGBM+KFold training loop but improve feature signal and reduce noise. The largest gain with minimal disruption is to (1) fix the distance feature to use miles (this competition’s fares are closer to miles than km), and (2) add two standard, non-invasive geospatial features (Manhattan distance and bearing) that typically reduce RMSE without changing the modeling approach. We also apply the same coordinate sanity filter to the concatenated `data` so test-time feature distributions match train-time filtering behavior (without dropping any test rows), and we clip negative predictions to 0 since fares can’t be negative (a safe post-processing step that usually helps RMSE slightly). Everything else (data loading, filtering, CV, LightGBM training, submission schema/path) stays intact.'
- What this solution (achieved 4.12061) has done: 'We need to reduce RMSE from 4.1158 toward 3.42426 (lower is better), so we make small, high-impact fixes without changing the LightGBM+KFold core training loop. The biggest likely gain at this stage is correcting the “manhattan_miles” feature: it currently adds degree differences and then multiplies by a miles conversion factor, which is dimensionally wrong; we compute Manhattan distance in miles using latitude/longitude scaling at NYC latitudes. We also add one very standard, low-risk feature (`euclidean_miles`) derived from the same corrected deltas, which preserves the overall approach but typically helps RMSE. Everything else (data loading, filters, CV, LightGBM training, callbacks, submission format/path) stays the same.'
- What this solution (achieved 4.14187) has done: 'We need to move RMSE down from 4.12061 toward 3.42426 (lower is better), so we should make small, high-impact changes without altering the LightGBM+KFold core loop. The most “minimal but meaningful” gain for NYC Taxi Fare is adding a couple of standard geospatial/time features that preserve the same semantics: airport-distance features (JFK/LGA/EWR) and a simple “near NYC center” distance, which often reduces error on airport trips. We also slightly tighten the training filters to remove clearly invalid zero-distance rides and extreme coordinate deltas that add label noise, while keeping the same sampling (1M rows), model type, and training procedure. Finally, we keep submission generation identical (same columns, same filename) and maintain deterministic behavior.'
- What this solution (achieved 4.14187) has done: 'We need to move RMSE down from 4.14187 toward 3.42426 (lower is better), so the smallest likely win is to fix two correctness bugs in feature engineering that currently degrade model signal. First, the `bearing_np()` computation has a broken term (a no-op `* np.sin(lat1) * 0`) and an unnecessarily convoluted expression; correcting it to the standard bearing formula improves a key directional feature without changing the modeling approach. Second, we currently impute all numeric columns (including `fare_amount`) inside the concatenated train+test frame, which can leak/warp the target and hurts learning; we exclude `fare_amount` from imputation while keeping the same median-impute strategy for features. Everything else (data size, filters, feature set intent, LightGBM + 5-fold KFold + early stopping, submission file/columns) stays the same.'
- What this solution (achieved 4.14546) has done: 'Your RMSE (4.14187) is still above the target (3.42426), so we should make a small, legitimate improvement that keeps the same LightGBM+KFold loop and loss/metric. The highest-impact minimal change here is to use LightGBM’s default boosting more effectively by adding a few standard regularization/robustness parameters (feature_fraction/bagging, min_data_in_leaf, and a slightly higher num_leaves), which typically improves generalization on this dataset without changing the approach. We also add one very standard interaction feature (`passenger_count * distance`) that preserves your existing feature engineering style and often reduces RMSE modestly. Everything else (sampling size, filters, feature creation intent, training loop, and submission format/path) stays the same.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

os.environ.setdefault("PYTHONHASHSEED", "0")
np.random.seed(0)



## === cell 1
train_dtypes = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
test_dtypes = {
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}

train = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
    nrows=1_000_000,
    dtype=train_dtypes,
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
    parse_dates=["pickup_datetime"],
)
test = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
    dtype=test_dtypes,
    usecols=[
        "key",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
    parse_dates=["pickup_datetime"],
)
sample_submission = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
)



## === cell 2
train.dropna(inplace=True)



## === cell 3
m = (train["fare_amount"] > 0) & (train["fare_amount"] <= 250)
m &= train["pickup_longitude"].between(-75, -72)
m &= train["dropoff_longitude"].between(-75, -72)
m &= train["pickup_latitude"].between(40, 42)
m &= train["dropoff_latitude"].between(40, 42)
m &= train["passenger_count"].between(1, 6)

raw_abs_lon = (train["pickup_longitude"] - train["dropoff_longitude"]).abs()
raw_abs_lat = (train["pickup_latitude"] - train["dropoff_latitude"]).abs()
m &= (raw_abs_lon + raw_abs_lat) > 1e-6
m &= (raw_abs_lon < 1.0) & (raw_abs_lat < 1.0)

train = train.loc[m].reset_index(drop=True)



## === cell 4
JFK = (-73.7781, 40.6413)
LGA = (-73.8740, 40.7769)
EWR = (-74.1745, 40.6895)
NYC_CENTER = (-73.985428, 40.748817)


def _haversine_km_np(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c  # km


def _bearing_rad_np(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.arctan2(y, x)


def _add_features_inplace(df: pd.DataFrame) -> pd.DataFrame:
    coord_mask = (
        (df["pickup_longitude"].between(-75, -72))
        & (df["dropoff_longitude"].between(-75, -72))
        & (df["pickup_latitude"].between(40, 42))
        & (df["dropoff_latitude"].between(40, 42))
    )
    for c in [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
    ]:
        df.loc[~coord_mask, c] = np.nan

    dt = pd.to_datetime(df["pickup_datetime"], utc=True, errors="coerce").dt
    df["pickup_year"] = dt.year.astype("float32")
    df["pickup_month"] = dt.month.astype("float32")
    df["pickup_day"] = dt.day.astype("float32")
    df["pickup_hour"] = dt.hour.astype("float32")
    df["pickup_weekday"] = dt.weekday.astype("float32")

    plon = df["pickup_longitude"].to_numpy(dtype=np.float64, copy=False)
    plat = df["pickup_latitude"].to_numpy(dtype=np.float64, copy=False)
    dlon = df["dropoff_longitude"].to_numpy(dtype=np.float64, copy=False)
    dlat = df["dropoff_latitude"].to_numpy(dtype=np.float64, copy=False)
    pcnt = df["passenger_count"].to_numpy(dtype=np.float32, copy=False)

    abs_lon_diff = np.abs(plon - dlon).astype(np.float32, copy=False)
    abs_lat_diff = np.abs(plat - dlat).astype(np.float32, copy=False)
    df["abs_lon_diff"] = abs_lon_diff
    df["abs_lat_diff"] = abs_lat_diff

    haversine_miles = (_haversine_km_np(plon, plat, dlon, dlat) * 0.621371).astype(
        np.float32
    )
    df["haversine_miles"] = haversine_miles

    mean_lat_rad = np.radians((plat + dlat) * 0.5)
    miles_per_deg_lat = 69.172
    miles_per_deg_lon = 69.172 * np.cos(mean_lat_rad)

    dx_miles = (abs_lon_diff.astype(np.float64) * miles_per_deg_lon).astype(np.float32)
    dy_miles = (abs_lat_diff.astype(np.float64) * miles_per_deg_lat).astype(np.float32)

    df["manhattan_miles"] = (dx_miles + dy_miles).astype(np.float32, copy=False)
    df["euclidean_miles"] = np.sqrt(
        dx_miles.astype(np.float64) ** 2 + dy_miles.astype(np.float64) ** 2
    ).astype(np.float32)

    df["bearing"] = _bearing_rad_np(plon, plat, dlon, dlat).astype(np.float32)

    def _to_point_miles(lon, lat, point_lon, point_lat):
        return (
            _haversine_km_np(lon, lat, np.float64(point_lon), np.float64(point_lat))
            * 0.621371
        ).astype(np.float32)

    for name, (alon, alat) in [("jfk", JFK), ("lga", LGA), ("ewr", EWR)]:
        df[f"pickup_to_{name}_miles"] = _to_point_miles(plon, plat, alon, alat)
        df[f"dropoff_to_{name}_miles"] = _to_point_miles(dlon, dlat, alon, alat)

    df["pickup_to_center_miles"] = _to_point_miles(
        plon, plat, NYC_CENTER[0], NYC_CENTER[1]
    )
    df["dropoff_to_center_miles"] = _to_point_miles(
        dlon, dlat, NYC_CENTER[0], NYC_CENTER[1]
    )

    df["pc_x_haversine"] = (pcnt * haversine_miles).astype(np.float32)

    df.drop(columns=["pickup_datetime"], inplace=True)
    return df


train_fe = train.copy()
test_fe = test.copy()
_add_features_inplace(train_fe)
_add_features_inplace(test_fe)



## === cell 5
num_cols_train = train_fe.select_dtypes(include=[np.number]).columns
feature_num_cols = [c for c in num_cols_train if c != "fare_amount"]
medians = train_fe[feature_num_cols].median(numeric_only=True)

train_fe[feature_num_cols] = train_fe[feature_num_cols].fillna(medians)
test_fe[feature_num_cols] = test_fe[feature_num_cols].fillna(medians)



## === cell 6
y_train = train_fe["fare_amount"].astype(float)
X_train = train_fe.drop(["fare_amount", "key"], axis=1)
X_test = test_fe.drop(["key"], axis=1)

X_train_np = np.ascontiguousarray(X_train.to_numpy(dtype=np.float32, copy=False))
X_test_np = np.ascontiguousarray(X_test.to_numpy(dtype=np.float32, copy=False))
y_train_np = y_train.to_numpy(dtype=np.float64, copy=False)

feature_names = list(X_train.columns)



## === cell 7
try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

from sklearn.model_selection import KFold

y_preds = []
models = []
oof_train = np.zeros((len(X_train_np),), dtype=np.float64)
cv = KFold(n_splits=5, shuffle=True, random_state=0)
categorical_features = []



## === cell 8
import lightgbm as lgb

params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "dart",
    "learning_rate": 0.05,
    "max_bin": 300,
    "num_leaves": 64,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 0.1,
    "verbosity": -1,
    "num_threads": max(1, (os.cpu_count() or 1) - 1),
    "seed": 0,
    "feature_fraction_seed": 0,
    "bagging_seed": 0,
    "data_random_seed": 0,
    "deterministic": True,
    "force_row_wise": True,
    "feature_pre_filter": False,
}

cache_dir = "/kaggle/working/lgb_cache"
os.makedirs(cache_dir, exist_ok=True)

for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train_np, y_train_np)):
    X_tr = X_train_np[train_index]
    X_val = X_train_np[valid_index]
    y_tr = y_train_np[train_index]
    y_val = y_train_np[valid_index]

    lgb_train = lgb.Dataset(
        X_tr,
        y_tr,
        feature_name=feature_names,
        categorical_feature=categorical_features,
        free_raw_data=True,  # allow LightGBM to free raw matrix after constructing bins
    )
    lgb_train.construct()
    train_bin_path = os.path.join(cache_dir, f"train_fold{fold_id}.bin")
    lgb_train.save_binary(train_bin_path)

    lgb_train = lgb.Dataset(
        train_bin_path,
        feature_name=feature_names,
        categorical_feature=categorical_features,
        free_raw_data=True,
    )

    lgb_eval = lgb.Dataset(
        X_val,
        y_val,
        reference=lgb_train,
        feature_name=feature_names,
        categorical_feature=categorical_features,
        free_raw_data=True,
    )
    lgb_eval.construct()
    eval_bin_path = os.path.join(cache_dir, f"valid_fold{fold_id}.bin")
    lgb_eval.save_binary(eval_bin_path)
    lgb_eval = lgb.Dataset(
        eval_bin_path,
        reference=lgb_train,
        feature_name=feature_names,
        categorical_feature=categorical_features,
        free_raw_data=True,
    )

    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_eval],
        valid_names=["train", "valid"],
        num_boost_round=2000,
        callbacks=[
            lgb.log_evaluation(period=50),
        ],
    )

    num_iter = model.current_iteration()
    oof_train[valid_index] = model.predict(X_val, num_iteration=num_iter)
    y_pred = model.predict(X_test_np, num_iteration=num_iter)

    y_preds.append(y_pred)
    models.append(model)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
LightGBMError                             Traceback (most recent call last)
/tmp/ipykernel_11/1329666052.py in <cell line: 0>()
     76     )
     77 
---> 78     model = lgb.train(
     79         params,
     80         lgb_train,

/usr/local/lib/python3.11/dist-packages/lightgbm/engine.py in train(params, train_set, num_boost_round, valid_sets, valid_names, feval, init_model, keep_training_booster, callbacks)
    242         init_iteration = 0
    243 
--> 244     train_set._update_params(params)._set_predictor(predictor)
    245 
    246     is_valid_contain_train = False

/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py in _update_params(self, params)
   2745                     self._free_handle()
   2746                 else:
-> 2747                     raise LightGBMError(_LIB.LGBM_GetLastError().decode("utf-8"))
   2748         return self
   2749 

LightGBMError: Cannot change data_random_seed after constructed Dataset handle.

## === cell 9
pd.DataFrame(oof_train).to_csv("oof_train_kfold.csv", index=False)

scores = [m.best_score["valid"]["rmse"] for m in models]
score = sum(scores) / len(scores)
print("===CV scores (rmse)===")
print(scores)
print(score)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ZeroDivisionError                         Traceback (most recent call last)
/tmp/ipykernel_11/1929325761.py in <cell line: 0>()
      2 
      3 scores = [m.best_score["valid"]["rmse"] for m in models]
----> 4 score = sum(scores) / len(scores)
      5 print("===CV scores (rmse)===")
      6 print(scores)

ZeroDivisionError: division by zero

## === cell 10
from sklearn.metrics import mean_squared_error

y_pred_oof = oof_train
np.sqrt(mean_squared_error(y_train_np, y_pred_oof))



## === cell 11
len(y_preds)



## === cell 12
y_preds[0][:10]



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/3456157511.py in <cell line: 0>()
----> 1 y_preds[0][:10]
      2 

IndexError: list index out of range

## === cell 13
y_sub = sum(y_preds) / len(y_preds)
y_sub = np.clip(y_sub, 0.0, None)
y_sub[:10]



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
ZeroDivisionError                         Traceback (most recent call last)
/tmp/ipykernel_11/2659218224.py in <cell line: 0>()
----> 1 y_sub = sum(y_preds) / len(y_preds)
      2 y_sub = np.clip(y_sub, 0.0, None)
      3 y_sub[:10]
      4 

ZeroDivisionError: division by zero

## === cell 14
sub_lgb = pd.DataFrame({"key": test["key"].values, "fare_amount": y_sub})
sub_lgb.to_csv("submission_lightgbm.csv", index=False)

sub_lgb.head()

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2686002444.py in <cell line: 0>()
----> 1 sub_lgb = pd.DataFrame({"key": test["key"].values, "fare_amount": y_sub})
      2 sub_lgb.to_csv("submission_lightgbm.csv", index=False)
      3 
      4 sub_lgb.head()

NameError: name 'y_sub' is not defined

## --- ERROR in outputing the csv:
Invalid submission: Submission should have a fare_amount column
