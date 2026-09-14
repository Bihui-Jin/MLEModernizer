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

3.40091

# 6. Current score

5.50287

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.39598) has done: 'Your code likely didn’t yield a Kaggle score because the file paths point to `/kaggle/input/...`, but in your environment the data actually lives under `/kaggle/data/...`, causing the notebook/script to fail before writing `submission.csv`. I make the smallest possible change to robustly find the CSVs in both `/kaggle/input` and `/kaggle/data` so it always runs end-to-end and produces a valid submission. To gently improve RMSE toward your target (without changing the model/training approach), I add standard NYC coordinate and fare sanity filters that remove obvious outliers while keeping your LightGBM + KFold pipeline intact. Everything else (feature set, CV loop, LightGBM training call) stays the same.'
- What this solution (achieved 5.50287) has done: 'Your run currently doesn’t yield a score because the notebook never reads/writes a submission in Kaggle’s expected place/name in some runtimes; I make the output path explicit and also assert the submission schema/row alignment before writing. To move RMSE down toward your 3.40091 target without changing the model/training loop, I make a minimal but impactful correction: compute the CV metric as RMSE (your current printed CV “l2” is MSE) and set LightGBM’s evaluation metric to RMSE so iteration selection/monitoring matches the competition metric. Finally, I add a very small, standard “data concatenation” fix to avoid train/test slicing issues after filtering (use an explicit train length variable), preserving your exact feature engineering and KFold setup.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
TRAIN_PATH_CANDIDATES = [
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
    "/kaggle/input/new-york-city-taxi-fare-prediction/new-york-city-taxi-fare-prediction/train.csv",
    "/kaggle/input/train.csv",
    "/kaggle/data/new-york-city-taxi-fare-prediction/train.csv",
    "/kaggle/data/new-york-city-taxi-fare-prediction/new-york-city-taxi-fare-prediction/train.csv",
    "/kaggle/data/train.csv",
]
TEST_PATH_CANDIDATES = [
    "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
    "/kaggle/input/new-york-city-taxi-fare-prediction/new-york-city-taxi-fare-prediction/test.csv",
    "/kaggle/input/test.csv",
    "/kaggle/data/new-york-city-taxi-fare-prediction/test.csv",
    "/kaggle/data/new-york-city-taxi-fare-prediction/new-york-city-taxi-fare-prediction/test.csv",
    "/kaggle/data/test.csv",
]
SAMPLE_SUB_PATH_CANDIDATES = [
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv",
    "/kaggle/input/new-york-city-taxi-fare-prediction/new-york-city-taxi-fare-prediction/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
    "/kaggle/data/new-york-city-taxi-fare-prediction/sample_submission.csv",
    "/kaggle/data/new-york-city-taxi-fare-prediction/new-york-city-taxi-fare-prediction/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
]


def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of these paths exist: {paths}")


TRAIN_PATH = _first_existing(TRAIN_PATH_CANDIDATES)
TEST_PATH = _first_existing(TEST_PATH_CANDIDATES)
SAMPLE_SUB_PATH = _first_existing(SAMPLE_SUB_PATH_CANDIDATES)

print("Using:")
print("TRAIN_PATH:", TRAIN_PATH)
print("TEST_PATH:", TEST_PATH)
print("SAMPLE_SUB_PATH:", SAMPLE_SUB_PATH)



## === cell 2
train = pd.read_csv(TRAIN_PATH, nrows=1_000_000)
test = pd.read_csv(TEST_PATH)
sample_submission = pd.read_csv(SAMPLE_SUB_PATH)



## === cell 3
train.isnull().sum()



## === cell 4
train.dropna(inplace=True)



## === cell 5
train.describe()



## === cell 6
train.query("passenger_count > 6")



## === cell 7
train.query("passenger_count < 1")



## === cell 8
train.query("fare_amount < 0")



## === cell 9
train.query("pickup_longitude < -180 or pickup_longitude > 180")



## === cell 10
train.query("dropoff_longitude < -180 or dropoff_longitude > 180")



## === cell 11
train.query("pickup_latitude < -90 or pickup_latitude > 90")



## === cell 12
train.query("dropoff_latitude < -90 or dropoff_latitude > 90")



## === cell 13
train = train.query(
    "1 <= passenger_count <= 6 and "
    "0 <= fare_amount <= 250 and "
    "-180 <= pickup_longitude <= 180 and "
    "-180 <= dropoff_longitude <= 180 and "
    "-90 <= pickup_latitude <= 90 and "
    "-90 <= dropoff_latitude <= 90 and "
    "-74.3 <= pickup_longitude <= -73.7 and "
    "-74.3 <= dropoff_longitude <= -73.7 and "
    "40.5 <= pickup_latitude <= 41.0 and "
    "40.5 <= dropoff_latitude <= 41.0"
)

zero_dist = (train["pickup_longitude"] == train["dropoff_longitude"]) & (
    train["pickup_latitude"] == train["dropoff_latitude"]
)
train = train.loc[~(zero_dist & (train["fare_amount"] > 5.0))].copy()

train.describe()



## === cell 14
train.reset_index(drop=True, inplace=True)
train



## === cell 15
n_train = len(train)
data = pd.concat([train, test], sort=False, ignore_index=True)



## === cell 16
data.head()




## === cell 17
def _haversine_km(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1.astype(np.float64))
    lat1 = np.radians(lat1.astype(np.float64))
    lon2 = np.radians(lon2.astype(np.float64))
    lat2 = np.radians(lat2.astype(np.float64))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return 6371.0 * c


def _bearing(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1.astype(np.float64))
    lat1 = np.radians(lat1.astype(np.float64))
    lon2 = np.radians(lon2.astype(np.float64))
    lat2 = np.radians(lat2.astype(np.float64))
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.arctan2(y, x)


data["pickup_datetime"] = pd.to_datetime(
    data["pickup_datetime"], errors="coerce", utc=True
)
data["pickup_hour"] = data["pickup_datetime"].dt.hour.astype("Int16")
data["pickup_dayofweek"] = data["pickup_datetime"].dt.dayofweek.astype("Int16")
data["pickup_month"] = data["pickup_datetime"].dt.month.astype("Int16")

data["abs_lon_diff"] = (data["dropoff_longitude"] - data["pickup_longitude"]).abs()
data["abs_lat_diff"] = (data["dropoff_latitude"] - data["pickup_latitude"]).abs()

data["haversine_km"] = _haversine_km(
    data["pickup_longitude"].values,
    data["pickup_latitude"].values,
    data["dropoff_longitude"].values,
    data["dropoff_latitude"].values,
)
data["manhattan_km"] = _haversine_km(
    data["pickup_longitude"].values,
    data["pickup_latitude"].values,
    data["dropoff_longitude"].values,
    data["pickup_latitude"].values,
) + _haversine_km(
    data["pickup_longitude"].values,
    data["pickup_latitude"].values,
    data["pickup_longitude"].values,
    data["dropoff_latitude"].values,
)

data["bearing"] = _bearing(
    data["pickup_longitude"].values,
    data["pickup_latitude"].values,
    data["dropoff_longitude"].values,
    data["dropoff_latitude"].values,
)

data["key"] = data["key"].astype(str)

data = data.drop("pickup_datetime", axis=1)

data.head()



## === cell 18
train = data.iloc[:n_train].copy()
test = data.iloc[n_train:].copy()

y_train = train["fare_amount"]
X_train = train.drop("fare_amount", axis=1)
X_test = test.drop("fare_amount", axis=1)

if "key" in X_train.columns:
    X_train = X_train.drop(columns=["key"])
if "key" in X_test.columns:
    X_test = X_test.drop(columns=["key"])

X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

X_train.head()



## === cell 19
from sklearn.model_selection import KFold

y_preds = []
models = []
oof_train = np.zeros((len(X_train),), dtype=np.float64)
cv = KFold(n_splits=5, shuffle=True, random_state=0)

categorical_features = []



## === cell 20
import lightgbm as lgb

X_train = X_train.copy()
X_test = X_test.copy()

for col in X_train.columns:
    if X_train[col].dtype == "object":
        X_train[col] = pd.to_numeric(X_train[col], errors="raise")
for col in X_test.columns:
    if X_test[col].dtype == "object":
        X_test[col] = pd.to_numeric(X_test[col], errors="raise")

params = {
    "objective": "regression",
    "metric": "rmse",
    "max_bin": 300,
    "learning_rate": 0.05,
    "num_leaves": 40,
}

for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train, y_train)):
    X_tr = X_train.iloc[train_index, :]
    X_val = X_train.iloc[valid_index, :]
    y_tr = y_train.iloc[train_index]
    y_val = y_train.iloc[valid_index]

    lgb_train = lgb.Dataset(X_tr, y_tr, categorical_feature=categorical_features)
    lgb_eval = lgb.Dataset(
        X_val, y_val, reference=lgb_train, categorical_feature=categorical_features
    )

    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_eval],
        valid_names=["train", "valid"],
        num_boost_round=1000,
        callbacks=[
            lgb.log_evaluation(period=10),
        ],
    )

    best_iter = model.best_iteration if model.best_iteration is not None else 1000

    oof_train[valid_index] = model.predict(X_val, num_iteration=best_iter)
    y_pred = model.predict(X_test, num_iteration=best_iter)

    y_preds.append(y_pred)
    models.append(model)



## === cell 21
pd.DataFrame(oof_train).to_csv("oof_train_kfold.csv", index=False)

scores_rmse = [m.best_score["valid"]["rmse"] for m in models]
score_rmse = sum(scores_rmse) / len(scores_rmse)
print("===CV scores (RMSE)===")
print(scores_rmse)
print(score_rmse)



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
y_sub[:10]



## === cell 26
y_sub = np.clip(y_sub, 0.0, None)

sub_lgb = pd.DataFrame({"key": test["key"].values.astype(str), "fare_amount": y_sub})
sub_lgb = sub_lgb[["key", "fare_amount"]]

assert (
    sub_lgb.shape[0] == sample_submission.shape[0]
), "Row count mismatch vs sample_submission"
assert list(sub_lgb.columns) == ["key", "fare_amount"], "Submission columns mismatch"

out_path = "submission.csv"
sub_lgb.to_csv(out_path, index=False)
print("Wrote:", out_path)
sub_lgb.head()
