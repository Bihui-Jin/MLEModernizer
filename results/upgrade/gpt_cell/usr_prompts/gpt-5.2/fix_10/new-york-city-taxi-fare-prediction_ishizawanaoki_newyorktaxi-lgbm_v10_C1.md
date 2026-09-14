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

Not yielded

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
NROWS = 2_000_000  # keep bounded for <600s on Kaggle CPU

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

train = pd.read_csv(
    os.path.join(INPUT_DIR, "train.csv"),
    usecols=TRAIN_COLS,
    nrows=NROWS,
    dtype=DTYPES_TRAIN,
)
test = pd.read_csv(
    os.path.join(INPUT_DIR, "test.csv"),
    usecols=TEST_COLS,
    dtype=DTYPES_TEST,
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
train = train.query(
    "1 <= passenger_count <= 6 and "
    "0 <= fare_amount <= 250 and "
    "-180 <= pickup_longitude <= 180 and "
    "-180 <= dropoff_longitude <= 180 and "
    "-90 <= pickup_latitude <= 90 and "
    "-90 <= dropoff_latitude <= 90"
)

train = train[
    (train["pickup_longitude"].between(-74.5, -72.8))
    & (train["dropoff_longitude"].between(-74.5, -72.8))
    & (train["pickup_latitude"].between(40.5, 41.8))
    & (train["dropoff_latitude"].between(40.5, 41.8))
].copy()

train.describe()



## === cell 13
train.reset_index(drop=True, inplace=True)
train.head()




## === cell 14
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["key"] = df["key"].astype(str)

    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)
    df["pickup_hour"] = dt.dt.hour.astype("float32").fillna(-1.0)
    df["pickup_dayofweek"] = dt.dt.dayofweek.astype("float32").fillna(-1.0)
    df["pickup_month"] = dt.dt.month.astype("float32").fillna(-1.0)
    df["pickup_year"] = dt.dt.year.astype("float32").fillna(-1.0)

    df = df.drop("pickup_datetime", axis=1)
    return df


train_fe = add_time_features(train)
test_fe = add_time_features(test)

train_fe.head()




## === cell 15
def haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1.astype(float))
    lon1 = np.radians(lon1.astype(float))
    lat2 = np.radians(lat2.astype(float))
    lon2 = np.radians(lon2.astype(float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0088 * c  # Earth radius in km


def bearing_rad(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1.astype(float))
    lon1 = np.radians(lon1.astype(float))
    lat2 = np.radians(lat2.astype(float))
    lon2 = np.radians(lon2.astype(float))
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.arctan2(y, x)


def add_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["distance"] = haversine_km(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )

    df["abs_lon_diff"] = (df["pickup_longitude"] - df["dropoff_longitude"]).abs()
    df["abs_lat_diff"] = (df["pickup_latitude"] - df["dropoff_latitude"]).abs()

    df["manhattan_km"] = haversine_km(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["pickup_latitude"],
        df["dropoff_longitude"],
    ) + haversine_km(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["pickup_longitude"],
    )

    df["center_lat"] = ((df["pickup_latitude"] + df["dropoff_latitude"]) / 2.0).astype(
        "float32"
    )
    df["center_lon"] = (
        (df["pickup_longitude"] + df["dropoff_longitude"]) / 2.0
    ).astype("float32")
    df["bearing"] = bearing_rad(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    ).astype("float32")
    df["bearing_abs"] = np.abs(df["bearing"]).astype("float32")

    return df


train_fe = add_geo_features(train_fe)
test_fe = add_geo_features(test_fe)

train_fe.head()



## === cell 16
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
y_train = train_fe["fare_amount"].astype(float)

X_train = train_fe.drop(["fare_amount", "key"], axis=1)
X_test = test_fe.drop(["key"], axis=1)

X_train.head()



## === cell 19
from sklearn.model_selection import KFold

y_preds = []
models = []
oof_train = np.zeros((len(X_train),), dtype=float)
cv = KFold(n_splits=5, shuffle=True, random_state=0)

categorical_features = []



## === cell 20
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
}

for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train, y_train)):
    X_tr = X_train.iloc[train_index, :]
    X_val = X_train.iloc[valid_index, :]
    y_tr = y_train.iloc[train_index]
    y_val = y_train.iloc[valid_index]

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
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)

    y_preds.append(y_pred)
    models.append(model)



## === cell 21
pd.DataFrame(oof_train).to_csv(
    os.path.join(WORKING_DIR, "oof_train_kfold.csv"), index=False
)

scores = [m.best_score["valid_1"]["rmse"] for m in models]
score = sum(scores) / len(scores)
print("===CV scores===")
print(scores)
print("mean:", score)



## === cell 22
from sklearn.metrics import mean_squared_error

y_pred_oof = oof_train
print("OOF RMSE:", np.sqrt(mean_squared_error(y_train, y_pred_oof)))



## === cell 23
len(y_preds)



## === cell 24
y_preds[0][:10]



## === cell 25
y_sub = sum(y_preds) / len(y_preds)
y_sub = np.clip(y_sub, 0.0, 500.0)
y_sub[:10]



## === cell 26
sub_lgb = pd.DataFrame({"key": test_fe["key"].astype(str).values, "fare_amount": y_sub})

submission_path = os.path.join(WORKING_DIR, "submission.csv")
sub_lgb.to_csv(submission_path, index=False)

print("Wrote:", submission_path, "shape:", sub_lgb.shape)
sub_lgb.head()
