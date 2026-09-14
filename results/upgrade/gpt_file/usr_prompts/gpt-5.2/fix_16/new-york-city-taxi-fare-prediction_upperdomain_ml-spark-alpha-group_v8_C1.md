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

3.7

# 3. Installed packages

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
scipy==1.15.3
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

4.08766

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 5.71754) has done: 'I fix the immediate runtime blocker by replacing the deprecated `sklearn.preprocessing.Imputer` with `sklearn.impute.SimpleImputer`, and ensure all required imports execute so later cells have `pd/np/df` defined. I also adjust file paths from `../input/...` to the provided Kaggle-style path `/kaggle/input/...` (with a safe fallback) so the CSVs can actually be read in your environment. To preserve the core modeling logic, I keep the same feature construction and `GradientBoostingRegressor` training, but make the missing-value imputation actually applied (it was fit but never used for train/test features). Finally, I guarantee a correctly formatted `submission.csv` with columns `key` and `fare_amount` is written.'
- What this solution (achieved 6.17147) has done: 'You’re currently far from the target (RMSE 5.72 vs 4.09; lower is better), so we should make small, metric-aligned improvements without changing the model family or overall approach. The biggest gap is likely from (1) training/serving skew caused by not applying the same outlier filtering logic to test features, and (2) a non-random split (first 70% of rows) that hurts the model fit and hyperparameter choice even though it doesn’t leak labels. I (a) apply the same basic validity filters to the test set (passenger_count and plausible coordinate ranges) and then clip test `distance_travel` to the training-supported range instead of leaving extreme values unhandled, and (b) shuffle before the 70/30 split to get a more representative training set while keeping the same training loop and model. These are minimal changes that typically reduce RMSE materially for this competition and should move you closer to the 4.09 target without altering your core feature design or estimator.'
- What this solution (achieved 5.25413) has done: 'Your current RMSE (6.17) is worse than the target (4.09), so we should make the smallest changes that legitimately improve generalization without changing the model family or feature set. The biggest issue is training/serving skew: you filter outliers on train (`distance_travel<30`, `fare_amount<100`) but only clip `distance_travel` on test, and you also drop many test rows then “fill back” with a different pipeline. I (1) apply the same coordinate/passenger validity filtering to the *training* sample before feature creation (to reduce noise/outliers consistently), (2) avoid dropping test rows at all (keep row alignment) and instead set invalid rows’ features to NaN then impute (same as train), and (3) cap the model prediction at 100 to match the training target support you enforced. These are minimal, metric-aligned adjustments that typically reduce RMSE for this competition while preserving your core logic (same features + GradientBoostingRegressor + same training loop).'
- What this solution (achieved 5.32933) has done: 'We keep your exact feature set and GradientBoostingRegressor training loop, but make two minimal, metric-aligned fixes that typically reduce RMSE for this competition: (1) add a small set of time-derived features from `pickup_datetime` (hour, weekday, month) to both train and test using the same pipeline, and (2) switch the model to train directly on RMSE-like loss by setting `loss="huber"` (more robust to remaining outliers) while keeping the same estimator family and number of trees. We also ensure the engineered `distance_travel` doesn’t create train/test skew by computing datetime features before splitting and applying identical transforms to test. These changes are small, don’t change your overall approach, and are expected to move your 5.254 RMSE downward toward the 4.088 target without heavy tuning.'
- What this solution (achieved 5.31715) has done: 'Your current RMSE (5.32933) is worse than the target (4.08766), so we make the smallest legitimate changes that usually reduce RMSE for this competition without changing your model family, feature set, or training loop. The main issue is remaining label noise/outliers and a train/test skew around unrealistic fares; we add a standard NYC-taxi cleanup filter on the training sample (`fare_amount <= 250`) while keeping your existing `fare_amount < 100` cap logic consistent by removing the too-aggressive cap and instead clipping predictions to the cleaned target range. We also add two tiny, metric-aligned location features (haversine distance and Manhattan distance) computed from the same already-present coordinates, keeping the rest of your pipeline identical. These changes typically improve generalization materially while staying within your “core logic preserved” constraint and still produce a valid `submission.csv`.'
- What this solution (achieved 5.24959) has done: 'We’re still worse than the target (RMSE 5.317 vs 4.088; lower is better), so the smallest likely win without changing your model family or training loop is to remove remaining feature/target outliers more consistently and add one minimal, standard NYC feature that’s highly predictive: the pickup/dropoff bearing (direction). This keeps your existing engineered distances/time features and the same `GradientBoostingRegressor(loss="huber")`, but gives it a bit more signal and reduces noise that hurts RMSE. I also make the train/test distance cap consistent by clipping `distance_travel` on train before filtering (prevents a few extreme computed values from slipping through) while keeping your existing thresholds and submission format unchanged. These changes are small, deterministic, and should move RMSE downward toward the target without altering core semantics.'
- What this solution (achieved 6.58653) has done: 'We’re still well above the target RMSE (5.24959 vs 4.08766; lower is better), so we make the smallest metric-aligned improvements without changing your core model family or feature set. The main fix is to switch the GradientBoostingRegressor to optimize squared-error (RMSE-aligned) instead of huber, and modestly increase the number of estimators while keeping the same approach and deterministic training. We also clip/clean the training target to the same support you already assume at inference (0–250) to reduce the impact of rare noisy labels without changing features or the training loop. Finally, we keep the submission schema identical and ensure row alignment is preserved.'
- What this solution (achieved 7.04205) has done: 'The timeout is almost certainly from fitting `GradientBoostingRegressor` with 300 estimators on ~1,000,000 rows, which is far too slow within 600s. To preserve identical core logic while making it run fast, I keep the same model/params/features but (a) reduce training rows to a deterministic cap that still matches the same pipeline, and (b) eliminate avoidable pandas overhead by vectorizing feature creation with NumPy arrays, avoiding repeated dtype casts, avoiding expensive `DataFrame.sample(frac=1.0)` shuffles, and preallocating the bias column once. These changes are equivalent in semantics (same feature formulas, same train/holdout split proportion, same estimator) but drastically cut wall time. I also disable plotting (it can be unexpectedly slow) without affecting predictions.'
- What this solution (achieved 7.23583) has done: 'You’re far above the target (RMSE 7.04 vs 4.09; lower is better), so we should make a small, metric-aligned improvement without changing your overall approach (same features + GradientBoostingRegressor). The biggest likely issue is underfitting from training on only 200k rows; increasing the training sample size (while keeping the same preprocessing and model) is the minimal, legitimate change that typically reduces RMSE noticeably. To keep runtime within 600s, we only moderately raise `NROWS_TRAIN` and also set `subsample` on the existing GradientBoostingRegressor to speed fitting while preserving the same estimator family and loss. Everything else (feature engineering, imputation, clipping, submission schema) stays the same.'
- What this solution (achieved 7.28171) has done: 'Your current RMSE (7.23583) is still far above the target (4.08766; lower is better), and the most likely cause is underfitting from too-small effective training data plus a bit of training/test preprocessing mismatch. To move toward the target with minimal core-logic changes, I only (1) increase the training sample size moderately (while keeping it bounded for the 600s limit), and (2) add a lightweight, standard cleanup filter (coordinate/passenger + reasonable fare/distance) that reduces label noise without changing features or the model family. I keep your exact feature engineering and `GradientBoostingRegressor` approach, and preserve the same submission format. These changes typically reduce RMSE materially on this competition and should move you closer to the target band.'
- What this solution (achieved 6.21779) has done: 'We’re far above the target (RMSE 7.28 vs 4.09; lower is better), so we need a small, legitimate boost without changing your core approach (same feature set, same GradientBoostingRegressor training loop, same loss). The most likely reason for the regression in score is underfitting from training on a noisy/too-large sample under a tight runtime budget; boosting trees on 1.2M rows can force compromises (or partial fits/time pressure) and hurt generalization. I make one minimal, metric-aligned change: train on a deterministic, cleaned subset size that is known to work well for this competition within 600s, and I also slightly increase model capacity in a way that typically improves RMSE without changing the model family (increase `max_depth` to 3 via `max_depth`/`max_leaf_nodes` equivalent is not available here, so we use `max_depth` through `max_depth`?—GradientBoostingRegressor uses `max_depth` inside `max_depth` of base learners as `max_depth` parameter). Finally, I ensure the same clipping/filters remain identical to avoid train/test skew.'

# 9. Code solution

## === cell 0
import os
import math
import numpy as np
import pandas as pd

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor

plt = None

np.random.seed(42)

INPUT_DIR_CANDIDATES = [
    "/kaggle/input/new-york-city-taxi-fare-prediction",
    "/kaggle/input",
    "../input",
]
INPUT_DIR = None
for d in INPUT_DIR_CANDIDATES:
    if os.path.isdir(d):
        INPUT_DIR = d
        break
if INPUT_DIR is None:
    raise FileNotFoundError(
        "Could not locate Kaggle input directory among known candidates."
    )

print("Using INPUT_DIR:", INPUT_DIR)
print("Top-level INPUT_DIR listing:", os.listdir(INPUT_DIR)[:20])


def _resolve_file(*names):
    """Try resolving a file either directly under INPUT_DIR or at INPUT_DIR/<competition-subdir>/."""
    for name in names:
        p1 = os.path.join(INPUT_DIR, name)
        if os.path.exists(p1):
            return p1
        p2 = os.path.join(INPUT_DIR, "new-york-city-taxi-fare-prediction", name)
        if os.path.exists(p2):
            return p2
    raise FileNotFoundError(f"Could not resolve any of: {names} under {INPUT_DIR}")


TRAIN_PATH = _resolve_file("train.csv")
TEST_PATH = _resolve_file("test.csv")
SAMPLE_SUB_PATH = _resolve_file("sample_submission.csv")



## === cell 1
NROWS_TRAIN = int(os.environ.get("NROWS_TRAIN", "600000"))
NROWS_TRAIN = int(min(max(NROWS_TRAIN, 300000), 900000))

usecols = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
dtype = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "float32",
}
df = pd.read_csv(
    TRAIN_PATH,
    nrows=NROWS_TRAIN,
    usecols=usecols,
    dtype=dtype,
    parse_dates=["pickup_datetime"],
    infer_datetime_format=True,
    utc=True,
)
df.head()



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2045328920.py in <cell line: 0>()
     22     "passenger_count": "float32",
     23 }
---> 24 df = pd.read_csv(
     25     TRAIN_PATH,
     26     nrows=NROWS_TRAIN,

TypeError: read_csv() got an unexpected keyword argument 'utc'

## === cell 2
mask = (df["passenger_count"] > 0) & (df["fare_amount"] > 0)
mask &= df["pickup_longitude"].between(-75, -72) & df["dropoff_longitude"].between(
    -75, -72
)
mask &= df["pickup_latitude"].between(40, 42) & df["dropoff_latitude"].between(40, 42)
df = df.loc[mask].reset_index(drop=True)
df.head()



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3419904463.py in <cell line: 0>()
      1 # Speed: use a single boolean mask to filter once (avoids repeated DataFrame copies).
      2 # Correctness: same filtering conditions as before.
----> 3 mask = (df["passenger_count"] > 0) & (df["fare_amount"] > 0)
      4 mask &= df["pickup_longitude"].between(-75, -72) & df["dropoff_longitude"].between(
      5     -75, -72

NameError: name 'df' is not defined

## === cell 3
alpha_ang = 0.506


def distance_travel(df_):
    plon = df_["pickup_longitude"].to_numpy(dtype=np.float32, copy=False)
    plat = df_["pickup_latitude"].to_numpy(dtype=np.float32, copy=False)
    dlon = df_["dropoff_longitude"].to_numpy(dtype=np.float32, copy=False)
    dlat = df_["dropoff_latitude"].to_numpy(dtype=np.float32, copy=False)

    abs_diff_longitude = np.abs(dlon - plon) * np.float32(50.0)
    abs_diff_latitude = np.abs(dlat - plat) * np.float32(69.0)
    displacement_vector = np.sqrt(
        abs_diff_latitude * abs_diff_latitude + abs_diff_longitude * abs_diff_longitude
    )

    denom = abs_diff_latitude.astype(np.float32, copy=False)
    denom = denom.copy()
    denom[denom == 0.0] = np.nan
    angle = np.arctan(abs_diff_longitude / denom)

    actual_long = np.abs(displacement_vector * np.sin(angle - alpha_ang))
    actual_lat = np.abs(displacement_vector * np.cos(angle - alpha_ang))
    distance = actual_long + actual_lat

    df_["abs_diff_longitude"] = abs_diff_longitude.astype("float32", copy=False)
    df_["abs_diff_latitude"] = abs_diff_latitude.astype("float32", copy=False)
    df_["displacement_vector"] = displacement_vector.astype("float32", copy=False)
    df_["actual_long"] = actual_long.astype("float32", copy=False)
    df_["actual_lat"] = actual_lat.astype("float32", copy=False)
    df_["distance_travel"] = distance.astype("float32", copy=False)


def add_time_features(df_):
    dt = df_.get("pickup_datetime", None)
    if not (
        isinstance(dt, pd.Series) and pd.api.types.is_datetime64_any_dtype(dt.dtype)
    ):
        dt = pd.to_datetime(df_["pickup_datetime"], errors="coerce", utc=True)
        df_["pickup_datetime"] = dt
    df_["pickup_hour"] = dt.dt.hour.astype("float32")
    df_["pickup_weekday"] = dt.dt.weekday.astype("float32")
    df_["pickup_month"] = dt.dt.month.astype("float32")


def add_geo_features(df_):
    plat = df_["pickup_latitude"].to_numpy(dtype=np.float64, copy=False)
    plon = df_["pickup_longitude"].to_numpy(dtype=np.float64, copy=False)
    dlat = df_["dropoff_latitude"].to_numpy(dtype=np.float64, copy=False)
    dlon = df_["dropoff_longitude"].to_numpy(dtype=np.float64, copy=False)

    lat1 = np.radians(plat)
    lon1 = np.radians(plon)
    lat2 = np.radians(dlat)
    lon2 = np.radians(dlon)

    dlat_r = lat2 - lat1
    dlon_r = lon2 - lon1
    a = (
        np.sin(dlat_r / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon_r / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(a))
    earth_radius_km = 6371.0
    haversine_km = earth_radius_km * c

    dlon_miles = np.abs(dlon - plon) * 50.0
    dlat_miles = np.abs(dlat - plat) * 69.0
    manhattan_miles = dlon_miles + dlat_miles

    df_["haversine_km"] = haversine_km.astype("float32", copy=False)
    df_["manhattan_miles"] = manhattan_miles.astype("float32", copy=False)


def add_bearing_feature(df_):
    plat = df_["pickup_latitude"].to_numpy(dtype=np.float64, copy=False)
    plon = df_["pickup_longitude"].to_numpy(dtype=np.float64, copy=False)
    dlat = df_["dropoff_latitude"].to_numpy(dtype=np.float64, copy=False)
    dlon = df_["dropoff_longitude"].to_numpy(dtype=np.float64, copy=False)

    lat1 = np.radians(plat)
    lon1 = np.radians(plon)
    lat2 = np.radians(dlat)
    lon2 = np.radians(dlon)

    dlon_r = lon2 - lon1
    y = np.sin(dlon_r) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon_r)
    bearing = np.arctan2(y, x)
    df_["bearing"] = bearing.astype("float32", copy=False)


def add_interaction_features(df_):
    dt = df_["distance_travel"].to_numpy(dtype=np.float32, copy=False)
    pc = df_["passenger_count"].to_numpy(dtype=np.float32, copy=False)
    df_["distance_sq"] = (dt * dt).astype("float32", copy=False)
    df_["dist_x_passengers"] = (dt * pc).astype("float32", copy=False)


distance_travel(df)
add_time_features(df)
add_geo_features(df)
add_bearing_feature(df)

df = df[df.distance_travel > 0]

add_interaction_features(df)

required_cols = [
    "distance_travel",
    "passenger_count",
    "pickup_hour",
    "pickup_weekday",
    "pickup_month",
    "haversine_km",
    "manhattan_miles",
    "bearing",
    "distance_sq",
    "dist_x_passengers",
    "fare_amount",
]
df = df.dropna(subset=required_cols).reset_index(drop=True)

df.head()



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1360291130.py in <cell line: 0>()
     99 
    100 
--> 101 distance_travel(df)
    102 add_time_features(df)
    103 add_geo_features(df)

NameError: name 'df' is not defined

## === cell 4
if plt is not None:
    test = df[df.passenger_count == 1]
    _ = test.iloc[: len(test)].plot.scatter("distance_travel", "fare_amount")



## === cell 5
df["distance_travel"] = df["distance_travel"].clip(lower=0, upper=30)

df = df[df.distance_travel < 30]

df = df[(df.fare_amount <= 250) & (df.fare_amount >= 2.5)]

df["fare_amount"] = df["fare_amount"].clip(lower=0, upper=250)

add_interaction_features(df)

if plt is not None:
    _ = df.iloc[:100000].plot.scatter("distance_travel", "fare_amount")



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3826101319.py in <cell line: 0>()
----> 1 df["distance_travel"] = df["distance_travel"].clip(lower=0, upper=30)
      2 
      3 df = df[df.distance_travel < 30]
      4 
      5 df = df[(df.fare_amount <= 250) & (df.fare_amount >= 2.5)]

NameError: name 'df' is not defined

## === cell 6
l = len(df)
print(l)

perm = np.random.RandomState(42).permutation(l)
df = df.iloc[perm].reset_index(drop=True)

df_train = df[: int(0.7 * l)]
df_test = df[int(0.7 * l) :]

feature_cols = [
    "distance_travel",
    "passenger_count",
    "pickup_hour",
    "pickup_weekday",
    "pickup_month",
    "haversine_km",
    "manhattan_miles",
    "bearing",
    "distance_sq",
    "dist_x_passengers",
]
train_mat = df_train[feature_cols].to_numpy(dtype=np.float32, copy=False)
test_mat = df_test[feature_cols].to_numpy(dtype=np.float32, copy=False)

train_X = np.empty((train_mat.shape[0], train_mat.shape[1] + 1), dtype=np.float32)
train_X[:, :-1] = train_mat
train_X[:, -1] = 1.0

test_X = np.empty((test_mat.shape[0], test_mat.shape[1] + 1), dtype=np.float32)
test_X[:, :-1] = test_mat
test_X[:, -1] = 1.0

train_y = df_train["fare_amount"].to_numpy(dtype=np.float32, copy=False)
test_y = df_test["fare_amount"].to_numpy(dtype=np.float32, copy=False)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/81682480.py in <cell line: 0>()
----> 1 l = len(df)
      2 print(l)
      3 
      4 perm = np.random.RandomState(42).permutation(l)
      5 df = df.iloc[perm].reset_index(drop=True)

NameError: name 'df' is not defined

## === cell 7
imp = SimpleImputer(missing_values=np.nan, strategy="mean")
train_X_imp = imp.fit_transform(train_X)
test_X_imp = imp.transform(test_X)

regr = GradientBoostingRegressor(
    n_estimators=300,
    random_state=42,
    loss="squared_error",
    learning_rate=0.1,
    subsample=0.7,
    max_depth=3,
)
regr.fit(train_X_imp, train_y)

print("Holdout R^2:", regr.score(test_X_imp, test_y))

full_mat = df[feature_cols].to_numpy(dtype=np.float32, copy=False)

full_X = np.empty((full_mat.shape[0], full_mat.shape[1] + 1), dtype=np.float32)
full_X[:, :-1] = full_mat
full_X[:, -1] = 1.0
full_y = df["fare_amount"].to_numpy(dtype=np.float32, copy=False)

imp_full = SimpleImputer(missing_values=np.nan, strategy="mean")
full_X_imp = imp_full.fit_transform(full_X)

regr_full = GradientBoostingRegressor(
    n_estimators=300,
    random_state=42,
    loss="squared_error",
    learning_rate=0.1,
    subsample=0.7,
    max_depth=3,
)
regr_full.fit(full_X_imp, full_y)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/164561715.py in <cell line: 0>()
      1 imp = SimpleImputer(missing_values=np.nan, strategy="mean")
----> 2 train_X_imp = imp.fit_transform(train_X)
      3 test_X_imp = imp.transform(test_X)
      4 
      5 regr = GradientBoostingRegressor(

NameError: name 'train_X' is not defined

## === cell 8
t_usecols = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
t_dtype = {
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "float32",
}
tdf = pd.read_csv(
    TEST_PATH,
    usecols=t_usecols,
    dtype=t_dtype,
    parse_dates=["pickup_datetime"],
    infer_datetime_format=True,
    utc=True,
)

valid = tdf["passenger_count"] > 0
for c in ["pickup_longitude", "dropoff_longitude"]:
    valid &= tdf[c].between(-75, -72)
for c in ["pickup_latitude", "dropoff_latitude"]:
    valid &= tdf[c].between(40, 42)

distance_travel(tdf)
add_time_features(tdf)
add_geo_features(tdf)
add_bearing_feature(tdf)

tdf["distance_travel"] = tdf["distance_travel"].clip(lower=0, upper=30)

add_interaction_features(tdf)

for col in feature_cols:
    tdf.loc[~valid, col] = np.nan

tdf.head()



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3085382826.py in <cell line: 0>()
     16     "passenger_count": "float32",
     17 }
---> 18 tdf = pd.read_csv(
     19     TEST_PATH,
     20     usecols=t_usecols,

TypeError: read_csv() got an unexpected keyword argument 'utc'

## === cell 9
t_mat = tdf[feature_cols].to_numpy(dtype=np.float32, copy=False)

ttrain_X = np.empty((t_mat.shape[0], t_mat.shape[1] + 1), dtype=np.float32)
ttrain_X[:, :-1] = t_mat
ttrain_X[:, -1] = 1.0

ttrain_X = imp_full.transform(ttrain_X)
output = regr_full.predict(ttrain_X)

output = np.clip(output, 0, 250)

print(output[:10], " ... ", "n_preds=", len(output))



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/564780254.py in <cell line: 0>()
----> 1 t_mat = tdf[feature_cols].to_numpy(dtype=np.float32, copy=False)
      2 
      3 # Speed: avoid hstack copy by preallocating.
      4 ttrain_X = np.empty((t_mat.shape[0], t_mat.shape[1] + 1), dtype=np.float32)
      5 ttrain_X[:, :-1] = t_mat

NameError: name 'tdf' is not defined

## === cell 10
my_submission = pd.DataFrame({"key": tdf.key, "fare_amount": output})
my_submission.to_csv("submission.csv", index=False)
my_submission.head()

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2839364836.py in <cell line: 0>()
----> 1 my_submission = pd.DataFrame({"key": tdf.key, "fare_amount": output})
      2 my_submission.to_csv("submission.csv", index=False)
      3 my_submission.head()

NameError: name 'tdf' is not defined
