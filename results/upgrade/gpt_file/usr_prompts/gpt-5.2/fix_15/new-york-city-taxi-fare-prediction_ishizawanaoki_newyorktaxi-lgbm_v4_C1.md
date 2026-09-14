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

3.43515

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.38121) has done: 'Your code already trains and writes a submission CSV, but Kaggle didn’t “yield” a score likely because the file name/placement didn’t match what you uploaded, or because the pipeline can crash due to invalid target values (negative/zero fares, NaNs, out-of-range lat/lon) that LightGBM still fit but generalize poorly. I make minimal, score-relevant fixes: ensure we always write `submission.csv` (the common Kaggle default), add lightweight NYC taxi sanity filtering on the 1M training rows (removing impossible coordinates/fare/passenger_count) to improve RMSE without changing the model or training loop, and clip predictions to a valid non-negative range. These changes keep the same LightGBM setup and CV approach, just improve data correctness and submission reliability.'
- What this solution (achieved 5.05637) has done: 'Your current RMSE (5.38) is worse than the target (3.44), so we should improve generalization with minimal, score-relevant changes while keeping the same LightGBM/KFold training loop. The biggest gap is that the model is only using raw lon/lat/passenger_count after dropping datetime, which is too weak; we keep LightGBM as-is but add a few standard, cheap geospatial features (Haversine distance + coordinate deltas) derived from the same columns. We also apply the same NYC bounding-box sanitation to the test set (without dropping rows) and fill any invalid/out-of-box test coordinates with the train medians so the engineered features remain reasonable. Finally, we keep the same submission schema/path and still clip predictions to a valid range.'
- What this solution (achieved 4.99491) has done: 'We need to move RMSE down (lower is better) from 5.056 toward 3.435, so the smallest score-relevant improvements are to add a few more standard taxi features without changing the LightGBM training loop or objective. I keep your exact CV/training approach but (1) extract a handful of datetime features you currently drop, and (2) add two simple geospatial features (Manhattan distance proxy and bearing) that commonly reduce RMSE on this dataset. I also switch LightGBM’s metric from `l2` to `rmse` (same objective; just better-aligned eval/early-stopping) and increase `stopping_rounds` slightly to reduce premature stopping—this doesn’t change core logic, only training termination. Submission writing stays identical and still outputs `submission.csv`.'
- What this solution (achieved 5.30126) has done: 'Your current RMSE (4.99) is still far above the target (3.44), so we should improve generalization with minimal, score-relevant changes while keeping the same LightGBM/KFold loop and objective. The biggest remaining gap is data quality: NYC Taxi Fare has many noisy/outlier records even within a broad NYC box, and tightening the “sane trip” filters (distance, coordinate sanity, and a simple fare-vs-distance plausibility bound) typically yields a large RMSE drop without changing the model itself. I also make one small, metric-aligned tweak: train on `log1p(fare_amount)` and invert with `expm1` at prediction time (still regression + RMSE, same training loop), which usually reduces the impact of heavy-tailed fares and improves RMSE. Submission writing remains identical and still produces `submission.csv`.'
- What this solution (achieved 5.30359) has done: 'The crash comes from `X_test` being empty: after you filtered `train` heavily, you rebuilt `train/test` by slicing the concatenated `data` using a fixed `len_original_train=1_000_000`, which no longer matches the filtered train length and can consume all test rows into the “train slice”. I fix this by preserving a `__is_test__` flag before concatenation and then splitting back by that flag (core model/training loop stays the same). I also keep the same feature engineering and filtering logic, but apply the distance/fare plausibility filters only to the true train rows, ensuring `X_test` remains non-empty. Finally, I add a couple of small guards to always produce a valid `submission.csv` with the required columns even if something unexpected happens.'
- What this solution (achieved 5.33069) has done: 'Your current RMSE (5.30359) is worse than the target (3.43515), so we should improve generalization with the smallest changes that don’t alter the LightGBM/KFold training loop or objective. The biggest likely issue is that the model is being trained on raw dollars but with a log transform, yet the filtering/plausibility rules are still in raw space and may be keeping heavy-noise points; we make the filters slightly more standard and add a single, very common geospatial feature (`euclidean_km` approximation) that’s derived from the same coordinates (no new data). We also make the datetime parsing more robust (avoid coerced NaT leakage into numeric medians) and ensure passenger_count is treated consistently as numeric. These are minimal, score-relevant tweaks that usually reduce RMSE noticeably on this competition while preserving your core architecture and training semantics.'
- What this solution (achieved 5.69603) has done: 'Your current run likely didn’t “yield” a Kaggle score because the notebook can time out (1M rows + 5-fold LGBM with up to 1000 rounds) or occasionally fail from accidental train/test misalignment after filtering. I make minimal, score-relevant changes to (1) guarantee a submission CSV is always produced even if training fails, (2) cut training time while preserving the same LightGBM/KFold training approach by reducing folds and boosting rounds (still early-stopped), and (3) add one small, standard sanitation: clip negative/too-large engineered distances and ensure datetime-derived features are filled deterministically. These changes keep your core feature set, model type, loss/objective, and training loop semantics the same, but make it far more likely you actually get a valid scored submission within Kaggle’s runtime.'
- What this solution (achieved 5.3505) has done: 'Your current RMSE (5.696) is far worse than the target (3.435), so we should improve generalization with the smallest changes that don’t alter the LightGBM/KFold training loop or objective. The biggest score drag in your script is that you’re training on `log1p(fare)` but LightGBM is early-stopping on RMSE in *log space*, which selects iterations that aren’t optimal for Kaggle’s RMSE in *dollar space*. I keep the exact model/training approach, but add a custom LightGBM evaluation metric that computes RMSE after `expm1` (i.e., in original fare units) so early stopping is aligned with Kaggle’s metric. I also make one minimal data-correctness fix: apply the same NYC bounding-box imputation to test rows that are out-of-box (not just NaNs), since out-of-box coordinates can explode distance features and hurt predictions.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
train = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", nrows=1_000_000
)
test = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")
sample_submission = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
)



## === cell 2
train = train.dropna(
    subset=[
        "fare_amount",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
        "pickup_datetime",
    ]
)

train["fare_amount"] = pd.to_numeric(train["fare_amount"], errors="coerce")
train["passenger_count"] = pd.to_numeric(train["passenger_count"], errors="coerce")
train = train.dropna(subset=["fare_amount", "passenger_count"])

train = train[(train["fare_amount"] > 0) & (train["fare_amount"] < 250)]
train = train[(train["passenger_count"] >= 1) & (train["passenger_count"] <= 6)]


def _in_nyc_box(df):
    return (
        (df["pickup_longitude"].between(-74.3, -73.7))
        & (df["dropoff_longitude"].between(-74.3, -73.7))
        & (df["pickup_latitude"].between(40.5, 41.0))
        & (df["dropoff_latitude"].between(40.5, 41.0))
    )


train = train[_in_nyc_box(train)].copy()



## === cell 3
fallback = sample_submission.copy()
fallback["fare_amount"] = float(train["fare_amount"].median())
fallback.to_csv("submission.csv", index=False)

train["__is_test__"] = 0
test["__is_test__"] = 1
test["fare_amount"] = np.nan  # keep column so concat has consistent schema

data = pd.concat([train, test], sort=False, ignore_index=True)



## === cell 4
data.head()




## === cell 5
def _haversine_km(lon1, lat1, lon2, lat2):
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    dlon = lon2 - lon1
    dlon = (dlon + np.pi) % (2.0 * np.pi) - np.pi  # dateline-safe
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 6371.0 * (2.0 * np.arcsin(np.sqrt(a)))


def _bearing(lon1, lat1, lon2, lat2):
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    dlon = lon2 - lon1
    dlon = (dlon + np.pi) % (2.0 * np.pi) - np.pi  # dateline-safe
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    brng = np.arctan2(y, x)  # radians in [-pi, pi]
    return np.asarray(brng, dtype=float)


coord_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]
train_coord_medians = train[coord_cols].median()

test_mask = data["__is_test__"].astype(int).eq(1)  # rows belonging to test

test_oob = test_mask & (~_in_nyc_box(data))
data.loc[test_oob, coord_cols] = train_coord_medians.values
data.loc[test_mask, coord_cols] = data.loc[test_mask, coord_cols].fillna(
    train_coord_medians
)

data["abs_lon_diff"] = (data["pickup_longitude"] - data["dropoff_longitude"]).abs()
data["abs_lat_diff"] = (data["pickup_latitude"] - data["dropoff_latitude"]).abs()

data["haversine_km"] = _haversine_km(
    data["pickup_longitude"],
    data["pickup_latitude"],
    data["dropoff_longitude"],
    data["dropoff_latitude"],
).astype(float)

lat_mean = (
    (data["pickup_latitude"].astype(float) + data["dropoff_latitude"].astype(float))
    / 2.0
).astype(float)
km_per_deg_lat = 111.32
km_per_deg_lon = 111.32 * np.cos(np.radians(lat_mean))
data["euclidean_km"] = np.sqrt(
    (data["abs_lat_diff"] * km_per_deg_lat) ** 2
    + (data["abs_lon_diff"] * km_per_deg_lon) ** 2
).astype(float)

data["manhattan_km"] = (
    _haversine_km(
        data["pickup_longitude"],
        data["pickup_latitude"],
        data["dropoff_longitude"],
        data["pickup_latitude"],
    )
    + _haversine_km(
        data["pickup_longitude"],
        data["pickup_latitude"],
        data["pickup_longitude"],
        data["dropoff_latitude"],
    )
).astype(float)

data["bearing_rad"] = _bearing(
    data["pickup_longitude"],
    data["pickup_latitude"],
    data["dropoff_longitude"],
    data["dropoff_latitude"],
).astype(float)

NYC_LON, NYC_LAT = -73.985428, 40.748817  # Midtown Manhattan (Empire State Building)
data["pickup_to_center_km"] = _haversine_km(
    data["pickup_longitude"], data["pickup_latitude"], NYC_LON, NYC_LAT
).astype(float)
data["dropoff_to_center_km"] = _haversine_km(
    data["dropoff_longitude"], data["dropoff_latitude"], NYC_LON, NYC_LAT
).astype(float)

dt = pd.to_datetime(data["pickup_datetime"], errors="coerce", utc=True).dt.tz_convert(
    None
)

data["pickup_hour"] = dt.dt.hour.astype("float32")
data["pickup_dayofweek"] = dt.dt.dayofweek.astype("float32")
data["pickup_month"] = dt.dt.month.astype("float32")
data["pickup_day"] = dt.dt.day.astype("float32")

if "pickup_datetime" in data.columns:
    data = data.drop("pickup_datetime", axis=1)

data["key"] = data["key"].astype(str)
data["passenger_count"] = pd.to_numeric(data["passenger_count"], errors="coerce")

for c, hi in [
    ("haversine_km", 120.0),
    ("euclidean_km", 120.0),
    ("manhattan_km", 200.0),
    ("pickup_to_center_km", 120.0),
    ("dropoff_to_center_km", 120.0),
]:
    if c in data.columns:
        data[c] = pd.to_numeric(data[c], errors="coerce").clip(lower=0.0, upper=hi)

data.head()



## === cell 6
train = data.loc[data["__is_test__"].astype(int).eq(0)].copy()
test = data.loc[data["__is_test__"].astype(int).eq(1)].copy()

train = train.dropna(subset=["fare_amount"]).copy()

train = train[(train["haversine_km"] > 0.10) & (train["haversine_km"] < 60.0)].copy()

min_fare = 2.5 + 0.6 * train["haversine_km"]
max_fare = 20.0 + 35.0 * train["haversine_km"]
train = train[
    (train["fare_amount"] >= min_fare) & (train["fare_amount"] <= max_fare)
].copy()

y_train = train["fare_amount"].astype(float)
X_train = train.drop(["fare_amount", "key", "__is_test__"], axis=1)
X_test = test.drop(["fare_amount", "key", "__is_test__"], axis=1)

X_train["passenger_count"] = (
    X_train["passenger_count"].round().clip(1, 6).astype("int8")
)
X_test["passenger_count"] = X_test["passenger_count"].round().clip(1, 6).astype("int8")

X_train = X_train.apply(pd.to_numeric, errors="coerce")
X_test = X_test.apply(pd.to_numeric, errors="coerce")

fill_vals = X_train.median(numeric_only=True)
X_train = X_train.fillna(fill_vals)
X_test = X_test.fillna(fill_vals)

assert X_train.shape[0] > 0 and X_train.shape[1] > 0, (
    X_train.shape,
    X_train.columns[:5],
)
assert X_test.shape[0] > 0 and X_test.shape[1] > 0, (X_test.shape, X_test.columns[:5])
assert list(X_train.columns) == list(X_test.columns)

X_train.head()



## === cell 7
from sklearn.model_selection import KFold

y_preds = []
models = []
oof_train = np.zeros((len(X_train),), dtype=float)

cv = KFold(n_splits=3, shuffle=True, random_state=0)

categorical_features = ["passenger_count"]



## === cell 8
import lightgbm as lgb

params = {
    "objective": "regression",
    "metric": "rmse",
    "max_bin": 300,
    "learning_rate": 0.05,
    "num_leaves": 40,
    "verbosity": -1,
}

y_train_log = np.log1p(y_train.values.astype(float))


def feval_rmse_expm1(y_pred, dataset):
    y_true = dataset.get_label()
    pred = np.expm1(y_pred)
    true = np.expm1(y_true)
    pred = np.clip(pred, 0.0, 500.0)
    true = np.clip(true, 0.0, 500.0)
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    return ("rmse_expm1", rmse, False)


for fold_id, (train_index, valid_index) in enumerate(
    cv.split(X_train, y_train_log), start=1
):
    X_tr = X_train.iloc[train_index, :]
    X_val = X_train.iloc[valid_index, :]
    y_tr = y_train_log[train_index]
    y_val = y_train_log[valid_index]

    lgb_train = lgb.Dataset(
        X_tr, label=y_tr, categorical_feature=categorical_features, free_raw_data=False
    )
    lgb_eval = lgb.Dataset(
        X_val,
        label=y_val,
        reference=lgb_train,
        categorical_feature=categorical_features,
        free_raw_data=False,
    )

    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_eval],
        valid_names=["train", "valid"],
        num_boost_round=600,
        feval=feval_rmse_expm1,
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=True),
            lgb.log_evaluation(period=20),
        ],
    )

    oof_log = model.predict(X_val, num_iteration=model.best_iteration)
    oof_train[valid_index] = np.expm1(oof_log)

    test_log = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = np.expm1(test_log)

    y_preds.append(y_pred)
    models.append(model)



## === cell 9
pd.DataFrame(oof_train, columns=["oof_pred"]).to_csv("oof_train_kfold.csv", index=False)

scores = [m.best_score["valid"].get("rmse_expm1", np.nan) for m in models]
score = float(sum(scores) / len(scores)) if len(scores) else float("nan")
print("===CV scores (rmse in $ via expm1) ===")
print(scores)
print(score)



## === cell 10
from sklearn.metrics import mean_squared_error

y_pred_oof = oof_train
rmse = float(np.sqrt(mean_squared_error(y_train, y_pred_oof)))
rmse



## === cell 11
len(y_preds)



## === cell 12
y_preds[0][:10] if len(y_preds) else None



## === cell 13
if len(y_preds) == 0:
    y_sub = np.full((len(X_test),), float(y_train.median()), dtype=float)
else:
    y_sub = sum(y_preds) / len(y_preds)

y_sub = np.clip(np.asarray(y_sub, dtype=float), 0.0, 500.0)
y_sub[:10]



## === cell 14
sub_lgb = pd.DataFrame(
    {
        "key": test["key"].astype(str).values,
        "fare_amount": y_sub.astype(float),
    }
)

assert list(sub_lgb.columns) == ["key", "fare_amount"]
assert len(sub_lgb) == len(test)

sub_lgb.to_csv("submission.csv", index=False)
sub_lgb.to_csv("submission_lightgbm.csv", index=False)

sub_lgb.head()
