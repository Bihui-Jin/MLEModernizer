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
geopy==2.4.1
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

3.27817

# 6. Current score

5.53169

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.39232) has done: 'Diagnosis: The crash happens because LightGBM 4.6.0 removed the `verbose_eval` and `early_stopping_rounds` keyword arguments from `lgb.train()`. In this version, verbosity and early stopping must be provided via callbacks (e.g., `lgb.log_evaluation()` and `lgb.early_stopping()`), otherwise `TypeError` is raised. The rest of the training loop and data preparation are fine, so we only need to adjust the `lgb.train()` call in cell 20 to use callbacks while keeping identical training semantics.

Patch summary: In cell 20, replace `verbose_eval=10` and `early_stopping_rounds=10` with `callbacks=[lgb.early_stopping(10), lgb.log_evaluation(10)]`. This preserves the same early stopping behavior and logging frequency and keeps `model.best_iteration` and `model.best_score` available for cell 21.

Updated cells / Compatibility notes for cell k+1 / Assumptions:
- Updated only cell 20.
- `models`, `oof_train`, and each `model.best_score['valid_1']['l2']` remain available and consistent for cell 21.
- Assumption: Default `first_metric_only=False` is acceptable (same effective behavior as prior usage since only `l2` metric is used by default for regression).'
- What this solution (achieved 5.92558) has done: 'You’re currently underperforming the target (RMSE 4.39232 vs 3.27817; lower is better), so we should improve score with minimal, semantics-preserving changes. The biggest win without changing the model/training loop is fixing the `key` handling: stripping to digits and converting to float destroys a high-cardinality identifier and can harm generalization; we keep `key` as a string purely for joining/submission and exclude it from training features. Next, we replace the extremely slow `geopy.distance(...).miles` row-wise apply with a vectorized haversine distance in kilometers, which is the same core feature idea (“distance”) but computed correctly/consistently and fast enough to allow using the full 1M rows. Finally, we add a very standard NYC bounding-box + distance sanity filter (still the same data-cleaning stage) to remove outliers that inflate RMSE, while keeping the LightGBM setup intact.'
- What this solution (achieved 7.20094) has done: 'Your pipeline likely didn’t yield a score because the run either timed out or didn’t complete reliably due to using a too-large training sample with Python/pandas overhead. To move RMSE toward the 3.278 target with minimal semantic change, I (1) keep the same LightGBM+KFold training loop and the same feature set, (2) switch to fast, stable column-only reading and explicitly parse `pickup_datetime` with `utc=True` while keeping the same derived time features, and (3) slightly expand `nrows` (still bounded for runtime) and add two standard, low-risk geographic features (center lat/lon and bearing) that are consistent with the existing “geo feature” approach and typically improve RMSE without changing the model family. The script always write `/kaggle/working/submission.csv` with exactly `key,fare_amount` aligned to `test.csv`.'
- What this solution (achieved 6.44709) has done: 'Diagnosis: Cell 14 crashes because `np.issubdtype(dt.dtype, np.datetime64)` cannot handle pandas timezone-aware dtypes like `datetime64[ns, UTC]` (it raises `TypeError: Cannot interpret 'datetime64[ns, UTC]' as a data type`). In this notebook, `pickup_datetime` is parsed with `utc=True` during `read_csv`, so it is timezone-aware for both train and test. We should use pandas’ datetime type check (`pd.api.types.is_datetime64_any_dtype`) which supports both tz-naive and tz-aware datetimes, and only coerce with `pd.to_datetime(..., utc=True)` when needed. This keeps the same feature extraction logic while making it robust to timezone-aware dtypes.

Patch summary: Replace the NumPy dtype check with a pandas-safe datetime dtype check in `add_time_features`, leaving all downstream columns and semantics unchanged.

Updated cells: Only cell 14 is modified.

Compatibility notes for cell k+1: `train_fe` and `test_fe` keep the same columns (`key`, original numeric/location columns, and the four new time feature columns) and `pickup_datetime` is still dropped, so cell 15 continues to work unchanged.

Assumptions: `pickup_datetime` exists in both input DataFrames and is either already datetime-like (possibly tz-aware) or convertible via `pd.to_datetime` with `utc=True`.'
- What this solution (achieved 5.53169) has done: 'Your current RMSE (6.447) is worse than the target (3.278), so we should improve generalization with minimal, core-logic-preserving changes. The biggest likely issue is that the model is trained on unnormalized coordinates and raw distances without the standard log transform of the target; using `log1p(fare_amount)` for training and `expm1` at inference is a very common, low-risk tweak for this competition that reduces the effect of long-tail fares and typically improves RMSE without changing the model family or training loop. I also add a tiny, safe cleanup: drop any remaining rows with invalid time parsing (NaT) after feature extraction to avoid polluted time features. Everything else (LightGBM training approach, folds, features) stays the same, and the script still writes `/kaggle/working/submission.csv` with `key,fare_amount`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

os.environ["PYTHONHASHSEED"] = "0"

INPUT_DIR = "/kaggle/input/new-york-city-taxi-fare-prediction"
WORKING_DIR = "/kaggle/working"

print("Listing /kaggle/input (truncated):")
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))
    break



## === cell 1
NROWS = 2_000_000
CHUNKSIZE = 250_000

TRAIN_COLS = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
TEST_COLS = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

DTYPES_TRAIN = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
    "key": "string",
}
DTYPES_TEST = {
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
    "key": "string",
}

train_path = os.path.join(INPUT_DIR, "train.csv")

train_chunks = []
read_rows = 0
for chunk in pd.read_csv(
    train_path,
    usecols=TRAIN_COLS,
    dtype=DTYPES_TRAIN,
    parse_dates=["pickup_datetime"],
    chunksize=CHUNKSIZE,
):
    need = NROWS - read_rows
    if need <= 0:
        break
    if len(chunk) > need:
        chunk = chunk.iloc[:need].copy()
    train_chunks.append(chunk)
    read_rows += len(chunk)
    if read_rows >= NROWS:
        break

train = pd.concat(train_chunks, ignore_index=True)

test = pd.read_csv(
    os.path.join(INPUT_DIR, "test.csv"),
    usecols=TEST_COLS,
    dtype=DTYPES_TEST,
    parse_dates=["pickup_datetime"],
)
sample_submission = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))

print(train.shape, test.shape, sample_submission.shape)



## === cell 2
train.isnull().sum()



## === cell 3
train.dropna(inplace=True)



## === cell 4
train.describe()



## === cell 5
train.query("passenger_count > 6")



## === cell 6
train.query("passenger_count < 1")



## === cell 7
train.query("fare_amount < 0")



## === cell 8
train.query("pickup_longitude < -180 or pickup_longitude > 180")



## === cell 9
train.query("dropoff_longitude < -180 or dropoff_longitude > 180")



## === cell 10
train.query("pickup_latitude < -90 or pickup_latitude > 90")



## === cell 11
train.query("dropoff_latitude < -90 or dropoff_latitude > 90")



## === cell 12
plon = train["pickup_longitude"]
plat = train["pickup_latitude"]
dlon = train["dropoff_longitude"]
dlat = train["dropoff_latitude"]
pc = train["passenger_count"]
fare = train["fare_amount"]

mask = (
    pc.between(1, 6)
    & fare.between(0, 250)
    & plon.between(-180, 180)
    & dlon.between(-180, 180)
    & plat.between(-90, 90)
    & dlat.between(-90, 90)
    & plon.between(-74.5, -72.8)
    & dlon.between(-74.5, -72.8)
    & plat.between(40.5, 41.8)
    & dlat.between(40.5, 41.8)
)

mask &= ~(
    (plon.abs() < 1e-6)
    & (plat.abs() < 1e-6)
    & (dlon.abs() < 1e-6)
    & (dlat.abs() < 1e-6)
)

mask &= ~((plon == dlon) & (plat == dlat) & (fare > 20.0))

train = train.loc[mask].copy()
train.describe()



## === cell 13
train.reset_index(drop=True, inplace=True)
train.head()




## === cell 14
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(deep=False)

    out["key"] = out["key"].astype(str)

    dt = out["pickup_datetime"]
    if not pd.api.types.is_datetime64_any_dtype(dt):
        dt = pd.to_datetime(dt, errors="coerce", utc=True)

    out["pickup_hour"] = dt.dt.hour.astype("float32").fillna(-1.0)
    out["pickup_dayofweek"] = dt.dt.dayofweek.astype("float32").fillna(-1.0)
    out["pickup_month"] = dt.dt.month.astype("float32").fillna(-1.0)
    out["pickup_year"] = dt.dt.year.astype("float32").fillna(-1.0)

    out = out.drop(columns=["pickup_datetime"])
    return out


train_fe = add_time_features(train)
test_fe = add_time_features(test)

train_fe.head()



## === cell 15
EARTH_RADIUS_KM = 6371.0088


def _to_rad_f32(arr: pd.Series) -> np.ndarray:
    return np.radians(arr.to_numpy(dtype=np.float64, copy=False))


def haversine_km_from_rad(lat1r, lon1r, lat2r, lon2r):
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c


def bearing_rad_from_rad(lat1r, lon1r, lat2r, lon2r):
    dlon = lon2r - lon1r
    y = np.sin(dlon) * np.cos(lat2r)
    x = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
    return np.arctan2(y, x)


def add_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(deep=False)

    plat_r = _to_rad_f32(out["pickup_latitude"])
    plon_r = _to_rad_f32(out["pickup_longitude"])
    dlat_r = _to_rad_f32(out["dropoff_latitude"])
    dlon_r = _to_rad_f32(out["dropoff_longitude"])

    dist = haversine_km_from_rad(plat_r, plon_r, dlat_r, dlon_r)
    out["distance"] = dist

    out["abs_lon_diff"] = (out["pickup_longitude"] - out["dropoff_longitude"]).abs()
    out["abs_lat_diff"] = (out["pickup_latitude"] - out["dropoff_latitude"]).abs()

    man1 = haversine_km_from_rad(plat_r, plon_r, plat_r, dlon_r)
    man2 = haversine_km_from_rad(plat_r, plon_r, dlat_r, plon_r)
    out["manhattan_km"] = man1 + man2

    out["center_lat"] = (
        (out["pickup_latitude"] + out["dropoff_latitude"]) / 2.0
    ).astype("float32")
    out["center_lon"] = (
        (out["pickup_longitude"] + out["dropoff_longitude"]) / 2.0
    ).astype("float32")

    b = bearing_rad_from_rad(plat_r, plon_r, dlat_r, dlon_r).astype("float32")
    out["bearing"] = b
    out["bearing_abs"] = np.abs(b).astype("float32")

    return out


train_fe = add_geo_features(train_fe)
test_fe = add_geo_features(test_fe)

train_fe.head()



## === cell 16
test_fe["pickup_longitude"] = test_fe["pickup_longitude"].clip(-74.5, -72.8)
test_fe["dropoff_longitude"] = test_fe["dropoff_longitude"].clip(-74.5, -72.8)
test_fe["pickup_latitude"] = test_fe["pickup_latitude"].clip(40.5, 41.8)
test_fe["dropoff_latitude"] = test_fe["dropoff_latitude"].clip(40.5, 41.8)

train_fe = train_fe[train_fe["distance"].between(0.0, 200.0)].copy()
train_fe.reset_index(drop=True, inplace=True)

test_fe["distance"] = test_fe["distance"].clip(0.0, 200.0)

train_fe.shape, test_fe.shape



## === cell 17
train_fe = train_fe[
    ~((train_fe["distance"] < 0.05) & (train_fe["fare_amount"] > 50.0))
].copy()
train_fe.reset_index(drop=True, inplace=True)
train_fe.shape



## === cell 18
time_ok = (
    (train_fe["pickup_hour"] >= 0)
    & (train_fe["pickup_dayofweek"] >= 0)
    & (train_fe["pickup_month"] >= 0)
    & (train_fe["pickup_year"] >= 0)
)
train_fe = train_fe.loc[time_ok].copy()
train_fe.reset_index(drop=True, inplace=True)

train_fe.shape



## === cell 19
y_train = np.log1p(train_fe["fare_amount"].astype(np.float64))

X_train = train_fe.drop(["fare_amount", "key"], axis=1)
X_test = test_fe.drop(["key"], axis=1)

X_train_np = X_train.to_numpy(dtype=np.float32, copy=False)
X_test_np = X_test.to_numpy(dtype=np.float32, copy=False)
y_train_np = y_train.to_numpy(copy=False)

X_train.head()



## === cell 20
from sklearn.model_selection import KFold

y_preds = []
models = []
oof_train = np.zeros((len(X_train_np),), dtype=np.float64)
cv = KFold(n_splits=5, shuffle=True, random_state=0)

categorical_features = []



## === cell 21
import lightgbm as lgb

params = {
    "objective": "regression",
    "metric": "rmse",
    "max_bin": 300,
    "learning_rate": 0.05,
    "num_leaves": 40,
    "feature_fraction": 0.9,
    "min_child_samples": 25,
    "seed": 0,
    "num_threads": max(1, (os.cpu_count() or 2) - 1),
    "deterministic": True,
    "feature_pre_filter": False,
}

for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train_np, y_train_np)):
    X_tr = X_train_np[train_index]
    X_val = X_train_np[valid_index]
    y_tr = y_train_np[train_index]
    y_val = y_train_np[valid_index]

    lgb_train = lgb.Dataset(
        X_tr, y_tr, categorical_feature=categorical_features, free_raw_data=False
    )
    lgb_eval = lgb.Dataset(
        X_val,
        y_val,
        reference=lgb_train,
        categorical_feature=categorical_features,
        free_raw_data=False,
    )

    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_eval],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=10),
            lgb.log_evaluation(period=10),
        ],
    )

    oof_train[valid_index] = model.predict(X_val, num_iteration=model.best_iteration)
    y_pred = model.predict(X_test_np, num_iteration=model.best_iteration)

    y_preds.append(y_pred)
    models.append(model)



## === cell 22
pd.DataFrame(oof_train).to_csv(
    os.path.join(WORKING_DIR, "oof_train_kfold.csv"), index=False
)

scores = [m.best_score["valid_1"]["rmse"] for m in models]
score = sum(scores) / len(scores)
print("===CV scores (RMSE on log1p target)===")
print(scores)
print("mean:", score)



## === cell 23
from sklearn.metrics import mean_squared_error

y_pred_oof_fare = np.expm1(oof_train)
y_true_fare = train_fe["fare_amount"].to_numpy(dtype=np.float64, copy=False)
print(
    "OOF RMSE (fare space):", np.sqrt(mean_squared_error(y_true_fare, y_pred_oof_fare))
)



## === cell 24
len(y_preds)



## === cell 25
y_preds[0][:10]



## === cell 26
y_sub_log = sum(y_preds) / len(y_preds)
y_sub = np.expm1(y_sub_log)

y_sub = np.clip(y_sub, 0.0, 500.0)
y_sub[:10]



## === cell 27
sub_lgb = pd.DataFrame({"key": test["key"].astype(str).values, "fare_amount": y_sub})

assert list(sub_lgb.columns) == ["key", "fare_amount"]
assert len(sub_lgb) == len(test), (len(sub_lgb), len(test))
assert sub_lgb["key"].isnull().sum() == 0

submission_path = os.path.join(WORKING_DIR, "submission.csv")
sub_lgb.to_csv(submission_path, index=False)

print("Wrote:", submission_path, "shape:", sub_lgb.shape)
sub_lgb.head()
