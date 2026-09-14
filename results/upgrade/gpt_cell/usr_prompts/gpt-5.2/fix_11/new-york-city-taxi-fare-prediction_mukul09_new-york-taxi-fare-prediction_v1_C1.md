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

3.27133

# 6. Current score

5.221

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.55737) has done: 'The crash happens because `KFold` defaults to `shuffle=False` in scikit-learn 1.2+, and providing `random_state` while not shuffling now raises a `ValueError`. The minimal fix is to explicitly enable shuffling so the existing intent (a deterministic split controlled by `random_state=0`) remains valid. This preserves the same cross-validation approach and keeps `cv_split` compatible with the next cell’s `cross_validate` call. No other logic is changed.'
- What this solution (achieved 4.44594) has done: 'Diagnosis: `cross_validate` only returns train scores when `return_train_score=True`; otherwise the `'train_score'` key is absent, causing the `KeyError`. Additionally, this notebook prints `tune_model.*` in cell 16, but `tune_model` is not defined anywhere in cells 0–15 (it appears tuning is done later as `submit_xgb` in cell 17), which would immediately cause a `NameError` after fixing the first issue.  
Patch summary: Update cell 16 to (1) compute and print only the always-present test score from `base_results`, and (2) guard the “After XGB …” prints so they run only if a `tune_model` object exists, preserving intended behavior when/if it’s defined elsewhere. This is the minimal change to stop the crash without altering model training or evaluation semantics.  
Updated cells: Only cell 16 is changed.  
Compatibility notes for cell k+1: No variables used by cell 17 are modified; `xgb`, `base_results`, `X`, `y`, and `cv_split` remain intact.  
Assumptions: The intent of cell 16 is reporting; it is acceptable to skip “After XGB …” reporting when `tune_model` is not defined yet.'
- What this solution (achieved 4.63234) has done: 'You’re currently far from the target RMSE (4.44594 vs 3.27133, lower is better), so we should make a small, legitimate improvement without changing the overall modeling approach (still XGBoost + same features + same CV/tuning structure). The biggest score drag is that your data “range filters” use `|` (OR), which effectively keeps almost all invalid coordinates; switching these to `&` (AND) is a minimal bug fix that preserves intent and typically improves RMSE materially. Also, training uses only 100k sampled rows; modestly increasing this sample size (still from the same 6M already loaded) improves generalization while keeping runtime reasonable. Finally, align the tuned model’s `random_state` with the baseline for stability, without changing the architecture or training semantics, and keep the submission format unchanged.'
- What this solution (achieved 4.65185) has done: 'Your RMSE (4.63234) is worse than the target (3.27133), so we need a small, legitimate improvement without changing the overall approach (still: same engineered features + XGBRegressor + CV/tuning + same submission semantics). The biggest low-risk gain here is to remove a training/inference mismatch: you filter invalid/outlier coordinates only in train, but not in test, which can cause extreme distances and unstable predictions; we add the same basic coordinate/passenger sanity checks to test and then clip predictions to a reasonable positive range. Additionally, your `pickup_datetime` parsing format is too strict for this dataset (often no `" UTC"` suffix), which can silently create NaTs and weaken time features; we switch to `pd.to_datetime(..., errors="coerce", utc=True)` to robustly parse while keeping the same time-derived features. These changes preserve the core model/training loop and feature set, but reduce noise/outliers to move RMSE downward toward the target.'
- What this solution (achieved 5.49606) has done: 'Your current RMSE (4.65185) is still far above the target (3.27133), so we need a small, legitimate improvement without changing the overall approach (same features + XGBRegressor + same CV/tuning structure). The biggest low-risk score drag left is that the training set still contains many extreme-but-“valid” NYC-outlier coordinates (e.g., points in oceans/deserts) that pass the global lat/lon bounds; adding a simple NYC bounding-box filter (common for this competition) reduces noise while keeping the same feature engineering and model. We also remove a strong, unnecessary prediction degradation: rounding predictions to 2 decimals increases RMSE slightly; keeping full precision in the submission is valid and typically improves RMSE a bit. Finally, we make the XGBoost parameter names compatible with xgboost==2.0.3 (use `learning_rate` and `verbosity`) to ensure the intended tuned parameters are actually applied.'
- What this solution (achieved 4.96512) has done: 'Your current RMSE (5.49606, lower is better) is far above the target (3.27133), so we make small, legitimate changes that reduce noise without changing the modeling approach (still: same feature set + XGBRegressor + same CV/tuning flow). The biggest low-risk gain is fixing a train/test distribution mismatch: you apply a NYC bounding-box filter to train but not to test; we apply the same NYC mask to test, and for rows outside NYC we impute features with medians from NYC-valid test rows (same strategy you already use for invalid rows). Next, we drop rows where datetime parsing failed (NaT) in train so time features aren’t polluted by missing values. Finally, we cap extreme trip distances in train (and mirror the same cap in test) to reduce the impact of remaining outliers while preserving the same distance feature and model.'
- What this solution (achieved 4.98724) has done: 'You’re still substantially above the target RMSE (4.965 vs 3.271, lower is better), so the smallest likely win without changing your overall approach is to fix a train/test mismatch you introduced: you clip `distance` to 100 km in train, but NYC-only rides can still legitimately exceed this, and the hard clip can bias the model upward/downward on longer airport/suburb trips. I replace this with a softer, data-driven cap computed from the training distances (high quantile), and apply the exact same cap to test so the feature distribution stays aligned. I also clip target fares to a reasonable upper bound (common for this dataset) to reduce the influence of rare labeling/outlier noise while keeping the same model, features, and training/tuning flow. These changes are minimal, deterministic, and should move RMSE downward toward your target without altering the core logic.'
- What this solution (achieved 5.19369) has done: 'To move RMSE down toward the 3.27 target without changing your overall approach (still the same XGBRegressor + same engineered features + same CV/tuning flow), I make two minimal, high-impact data-quality fixes: (1) remove obvious outliers using the standard “fare vs distance” plausibility rule (keeps your distance feature but drops mislabeled/noisy rows that hurt RMSE), and (2) add two lightweight geographic features (absolute lat/lon deltas) derived from the same existing columns to help the model separate short vs long rides without changing the model architecture. I keep the train/test feature pipeline perfectly aligned (same new features computed for test, same distance cap), and keep deterministic behavior. Submission format/path stay identical.'
- What this solution (achieved 5.06581) has done: 'We need to move your RMSE down (5.19 → 3.27), so the smallest legitimate gains come from fixing a train/test mismatch and reducing label noise without changing your XGBoost approach. I (1) replace the overly-tight NYC bounding box with a standard wider NYC metro box (still the same “bbox filtering” idea, just less underfitting and better coverage), applying it consistently to both train and test. I also (2) add the classic “zero-distance fare” cleanup (drop rows where distance is ~0 but fare is non-trivial), which removes obvious meter/label errors that hurt RMSE. Finally, I keep your existing feature set/model/tuning flow and ensure the submission is written exactly as required.'
- What this solution (achieved 5.221) has done: 'Your current RMSE (5.06581) is far above the target (3.27133), so we should make a small, legitimate improvement without changing the overall approach (still the same engineered features + XGBRegressor + CV/tuning). The biggest low-risk gain is to fix a metric mismatch: you tune with `neg_mean_squared_error` even though the competition is RMSE; switching to `neg_root_mean_squared_error` aligns CV selection with the leaderboard metric while keeping the same GridSearch structure. Next, the model is currently trained on raw `fare_amount`, which is heavy‑tailed; adding a minimal target transform via `TransformedTargetRegressor` (log1p/expm1) keeps the same base XGB model but typically reduces RMSE by stabilizing large-fare residuals. Finally, we keep test/train feature handling identical and add a small clamp to avoid negative fares without altering core feature engineering or the model class.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.compose import TransformedTargetRegressor

import warnings

warnings.filterwarnings("ignore")



## === cell 1
original_train_data = pd.read_csv("../input/train.csv", nrows=6000000)
train_data = original_train_data.sample(n=300000, random_state=0)
train_data.info()



## === cell 2
test_data = pd.read_csv("../input/test.csv")
test_data.info()



## === cell 3
train_data.isnull().sum()



## === cell 4
train_data.dropna(axis=0, inplace=True)



## === cell 5
train_data.describe()



## === cell 6
train_data = train_data[train_data["fare_amount"] > 0]
train_data = train_data[train_data["fare_amount"] <= 250]

train_data = train_data[
    (train_data["passenger_count"] <= 6) & (train_data["passenger_count"] > 0)
]

train_data = train_data[
    (train_data["pickup_latitude"] > -90) & (train_data["pickup_latitude"] <= 90)
]
train_data = train_data[
    (train_data["dropoff_latitude"] > -90) & (train_data["dropoff_latitude"] <= 90)
]

train_data = train_data[
    (train_data["pickup_longitude"] >= -180) & (train_data["pickup_longitude"] <= 180)
]
train_data = train_data[
    (train_data["dropoff_longitude"] >= -180) & (train_data["dropoff_longitude"] <= 180)
]

NYC_LAT_MIN, NYC_LAT_MAX = 40.0, 42.0
NYC_LON_MIN, NYC_LON_MAX = -75.0, -72.0

nyc_mask = (
    train_data["pickup_latitude"].between(NYC_LAT_MIN, NYC_LAT_MAX, inclusive="both")
    & train_data["dropoff_latitude"].between(NYC_LAT_MIN, NYC_LAT_MAX, inclusive="both")
    & train_data["pickup_longitude"].between(NYC_LON_MIN, NYC_LON_MAX, inclusive="both")
    & train_data["dropoff_longitude"].between(
        NYC_LON_MIN, NYC_LON_MAX, inclusive="both"
    )
)
train_data = train_data[nyc_mask]



## === cell 7
train_data.shape



## === cell 8
train_data.info()



## === cell 9
train_data.head(5)




## === cell 10
def distance(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    """
    Return distance along great radius between pickup and dropoff coordinates.
    """
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


def date_time_info(data):
    data["pickup_datetime"] = pd.to_datetime(
        data["pickup_datetime"], errors="coerce", utc=True
    )

    data["hour"] = data["pickup_datetime"].dt.hour
    data["day"] = data["pickup_datetime"].dt.day
    data["month"] = data["pickup_datetime"].dt.month
    data["weekday"] = data["pickup_datetime"].dt.weekday
    data["year"] = data["pickup_datetime"].dt.year

    return data


train_data = date_time_info(train_data)

train_data = train_data[train_data["pickup_datetime"].notna()]

train_data["distance"] = distance(
    train_data["pickup_latitude"],
    train_data["pickup_longitude"],
    train_data["dropoff_latitude"],
    train_data["dropoff_longitude"],
)

train_data["abs_lat_diff"] = (
    train_data["pickup_latitude"] - train_data["dropoff_latitude"]
).abs()
train_data["abs_lon_diff"] = (
    train_data["pickup_longitude"] - train_data["dropoff_longitude"]
).abs()

DIST_CAP = float(train_data["distance"].quantile(0.999))
DIST_CAP = max(1.0, min(DIST_CAP, 200.0))
train_data["distance"] = train_data["distance"].clip(lower=0.0, upper=DIST_CAP)

fare_per_km = train_data["fare_amount"] / (train_data["distance"] + 1e-3)
train_data = train_data[(fare_per_km >= 0.5) & (fare_per_km <= 50.0)]

train_data = train_data[
    ~((train_data["distance"] < 0.01) & (train_data["fare_amount"] > 2.5))
]

train_data.head()



## === cell 11
train_data.drop(["key", "pickup_datetime"], axis=1, inplace=True)
train_data.head()



## === cell 12
test_data.head()



## === cell 13
test_data = date_time_info(test_data)

valid_mask = (
    (test_data["passenger_count"].between(1, 6, inclusive="both"))
    & (test_data["pickup_latitude"].between(-90, 90, inclusive="right"))
    & (test_data["dropoff_latitude"].between(-90, 90, inclusive="right"))
    & (test_data["pickup_longitude"].between(-180, 180, inclusive="both"))
    & (test_data["dropoff_longitude"].between(-180, 180, inclusive="both"))
)

nyc_mask_test = (
    test_data["pickup_latitude"].between(NYC_LAT_MIN, NYC_LAT_MAX, inclusive="both")
    & test_data["dropoff_latitude"].between(NYC_LAT_MIN, NYC_LAT_MAX, inclusive="both")
    & test_data["pickup_longitude"].between(NYC_LON_MIN, NYC_LON_MAX, inclusive="both")
    & test_data["dropoff_longitude"].between(NYC_LON_MIN, NYC_LON_MAX, inclusive="both")
)

test_data["distance"] = distance(
    test_data["pickup_latitude"],
    test_data["pickup_longitude"],
    test_data["dropoff_latitude"],
    test_data["dropoff_longitude"],
)

test_data["abs_lat_diff"] = (
    test_data["pickup_latitude"] - test_data["dropoff_latitude"]
).abs()
test_data["abs_lon_diff"] = (
    test_data["pickup_longitude"] - test_data["dropoff_longitude"]
).abs()

test_data["distance"] = test_data["distance"].clip(lower=0.0, upper=DIST_CAP)

test_key = test_data["key"]
x_pred = test_data.drop(columns=["key", "pickup_datetime"])

impute_mask = valid_mask & nyc_mask_test
feature_medians = x_pred[impute_mask].median(numeric_only=True)
x_pred.loc[~impute_mask, feature_medians.index] = feature_medians.values



## === cell 14
y = train_data["fare_amount"]
X = train_data.drop(["fare_amount"], axis=1)

cv_split = KFold(n_splits=10, shuffle=True, random_state=0)



## === cell 15
xgb = TransformedTargetRegressor(
    regressor=XGBRegressor(random_state=0),
    func=np.log1p,
    inverse_func=np.expm1,
)

base_results = cross_validate(xgb, X, y, cv=cv_split)
xgb.fit(X, y)



## === cell 16
print("Best XGB parameters (base regressor): ", xgb.regressor_.get_params())

print(
    "Before XGB Testing score mean: {:.2f}".format(
        base_results["test_score"].mean() * 100
    )
)
print("#" * 20)

if "tune_model" in globals():
    print("After XGB Parameters: ", tune_model.best_params_)
    print(
        "After XGB Training score mean: {:.2f}".format(
            tune_model.cv_results_["mean_train_score"][tune_model.best_index_] * 100
        )
    )
    print(
        "After XGB Testing score mean: {:.2f}".format(
            tune_model.cv_results_["mean_test_score"][tune_model.best_index_] * 100
        )
    )
    print("#" * 20)



## === cell 17
params = {
    "regressor__max_depth": [8],  # Result of tuning with CV
    "regressor__learning_rate": [0.03],  # (was eta)
    "regressor__subsample": [1],  # Result of tuning with CV
    "regressor__colsample_bytree": [0.8],  # Result of tuning with CV
    "regressor__objective": ["reg:squarederror"],  # (was reg:linear, deprecated/alias)
    "regressor__eval_metric": ["rmse"],
    "regressor__verbosity": [0],  # (was silent)
}

submit_xgb = GridSearchCV(
    TransformedTargetRegressor(
        regressor=XGBRegressor(random_state=0),
        func=np.log1p,
        inverse_func=np.expm1,
    ),
    param_grid=params,
    scoring="neg_root_mean_squared_error",
    cv=cv_split,
)
submit_xgb.fit(X, y)



## === cell 18
prediction = submit_xgb.predict(x_pred)

prediction = np.clip(prediction, 0.0, 500.0)



## === cell 19
submission = pd.DataFrame({"key": test_key, "fare_amount": prediction})

submission.to_csv("taxi_fare_submission.csv", index=False)
submission.head()
