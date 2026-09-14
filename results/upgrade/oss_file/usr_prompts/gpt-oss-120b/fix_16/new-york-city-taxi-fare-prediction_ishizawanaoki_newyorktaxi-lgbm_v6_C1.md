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

3.7695

# 6. Current score

10.11589

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.97309) has done: 'I keep the overall workflow unchanged but add a few light feature‑engineering steps that are known to improve taxi‑fare predictions (hour, weekday, month) and some simple geometric differences, and I also restore the original `key` values (removing the earlier cleaning that altered IDs). These modest changes should lower the RMSE toward the target without altering the core model or training loop.'
- What this solution (achieved 5.34912) has done: 'I tighten the data cleaning to keep only realistic NYC coordinates, add a simple Manhattan distance and an hour‑passenger interaction feature (both cheap to compute), and slightly enlarge the LightGBM model (more leaves and modest feature/bagging fractions). These tweaks keep the original workflow intact while giving the model a bit more useful signal, which should lower the RMSE toward the target.'
- What this solution (achieved 5.28329) has done: 'I add a log‑transform of the target variable, train LightGBM on the transformed values and exponentiate the predictions back to the original scale. This small change often reduces RMSE without altering the core model or feature set, moving the score toward the target while keeping the workflow otherwise unchanged.'
- What this solution (achieved 5.23703) has done: 'I add a few cheap engineered features that give the model more signal (log‑distance, log‑passenger, distance‑per‑passenger) and set a modest regularisation parameter (`min_data_in_leaf`) in the LightGBM config. These tweaks preserve the overall workflow while likely lowering the RMSE toward the target.'
- What this solution (achieved 10.11589) has done: 'I add the missing imports, limit the training data size to keep memory usage reasonable, and ensure all variables are defined before use. This fixes the NameError errors, allows the LightGBM workflow to run, and creates a proper `submission_lightgbm.csv` file with the required columns.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import gc



## === cell 1
dtypes = {
    "key": "object",
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
usecols = list(dtypes.keys()) + ["pickup_datetime"]

train_path = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
test_path = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"
sample_sub_path = (
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
)

MAX_TRAIN_ROWS = 2_000_000  # adjust if more memory is available
train = pd.read_csv(
    train_path,
    usecols=usecols,
    dtype=dtypes,
    parse_dates=False,
    low_memory=False,
    nrows=MAX_TRAIN_ROWS,
)
test = pd.read_csv(
    test_path,
    usecols=[c for c in usecols if c != "fare_amount"],
    dtype={k: v for k, v in dtypes.items() if k != "fare_amount"},
    parse_dates=False,
    low_memory=False,
)
sample_submission = pd.read_csv(sample_sub_path)

train.dropna(inplace=True)
mask = (
    (train["passenger_count"].between(1, 6))
    & (train["fare_amount"].between(3.3, 52.33))
    & (train["pickup_longitude"].between(-74.5, -73.0))
    & (train["pickup_latitude"].between(40.0, 41.5))
    & (train["dropoff_longitude"].between(-74.5, -73.0))
    & (train["dropoff_latitude"].between(40.0, 41.5))
)
train = train.loc[mask].reset_index(drop=True)



## === cell 2
data = pd.concat([train, test], sort=False)

data["pickup_datetime"] = pd.to_datetime(data["pickup_datetime"], errors="coerce")
data["pickup_hour"] = data["pickup_datetime"].dt.hour.astype(np.int8)
data["pickup_weekday"] = data["pickup_datetime"].dt.weekday.astype(np.int8)
data["pickup_month"] = data["pickup_datetime"].dt.month.astype(np.int8)
data = data.drop(columns=["pickup_datetime"])



## === cell 3
train = data.iloc[: len(train)].reset_index(drop=True)
test = data.iloc[len(train) :].reset_index(drop=True)

y_train = train["fare_amount"]
y_train_log = np.log1p(y_train)

X_train_raw = train.drop(columns=["fare_amount", "key"])
X_test_raw = test.drop(columns=["fare_amount", "key"])

del train, test, data
gc.collect()



## === cell 4
lon1 = np.radians(X_train_raw["pickup_longitude"].values.astype(np.float32))
lat1 = np.radians(X_train_raw["pickup_latitude"].values.astype(np.float32))
lon2 = np.radians(X_train_raw["dropoff_longitude"].values.astype(np.float32))
lat2 = np.radians(X_train_raw["dropoff_latitude"].values.astype(np.float32))

lon1_t = np.radians(X_test_raw["pickup_longitude"].values.astype(np.float32))
lat1_t = np.radians(X_test_raw["pickup_latitude"].values.astype(np.float32))
lon2_t = np.radians(X_test_raw["dropoff_longitude"].values.astype(np.float32))
lat2_t = np.radians(X_test_raw["dropoff_latitude"].values.astype(np.float32))


def haversine_vec(lon1, lat1, lon2, lat2):
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


distance_train = haversine_vec(lon1, lat1, lon2, lat2)
distance_test = haversine_vec(lon1_t, lat1_t, lon2_t, lat2_t)


def manhattan_vec(lon1, lat1, lon2, lat2):
    leg1 = haversine_vec(lon1, lat1, lon2, lat1)
    leg2 = haversine_vec(lon2, lat1, lon2, lat2)
    return leg1 + leg2


manhattan_train = manhattan_vec(lon1, lat1, lon2, lat2)
manhattan_test = manhattan_vec(lon1_t, lat1_t, lon2_t, lat2_t)

hour_train = X_train_raw["pickup_hour"].values.astype(np.float32)
hour_test = X_test_raw["pickup_hour"].values.astype(np.float32)

passenger_train = X_train_raw["passenger_count"].values.astype(np.float32)
passenger_test = X_test_raw["passenger_count"].values.astype(np.float32)

lat_diff_train = np.abs(
    X_train_raw["pickup_latitude"].values - X_train_raw["dropoff_latitude"].values
).astype(np.float32)
lon_diff_train = np.abs(
    X_train_raw["pickup_longitude"].values - X_train_raw["dropoff_longitude"].values
).astype(np.float32)

lat_diff_test = np.abs(
    X_test_raw["pickup_latitude"].values - X_test_raw["dropoff_latitude"].values
).astype(np.float32)
lon_diff_test = np.abs(
    X_test_raw["pickup_longitude"].values - X_test_raw["dropoff_longitude"].values
).astype(np.float32)

hour_passenger_train = (hour_train * passenger_train).astype(np.float32)
hour_passenger_test = (hour_test * passenger_test).astype(np.float32)

hour_sin_train = np.sin(2 * np.pi * hour_train / 24).astype(np.float32)
hour_sin_test = np.sin(2 * np.pi * hour_test / 24).astype(np.float32)

hour_cos_train = np.cos(2 * np.pi * hour_train / 24).astype(np.float32)
hour_cos_test = np.cos(2 * np.pi * hour_test / 24).astype(np.float32)

distance_passenger_train = (distance_train * passenger_train).astype(np.float32)
distance_passenger_test = (distance_test * passenger_test).astype(np.float32)

distance_log_train = np.log1p(distance_train).astype(np.float32)
distance_log_test = np.log1p(distance_test).astype(np.float32)

passenger_log_train = np.log1p(passenger_train).astype(np.float32)
passenger_log_test = np.log1p(passenger_test).astype(np.float32)

distance_per_passenger_train = np.where(
    passenger_train == 0,
    0.0,
    (distance_train / passenger_train).astype(np.float32),
)
distance_per_passenger_test = np.where(
    passenger_test == 0,
    0.0,
    (distance_test / passenger_test).astype(np.float32),
)

X_train_np = np.column_stack(
    [
        X_train_raw["pickup_longitude"].values.astype(np.float32),
        X_train_raw["pickup_latitude"].values.astype(np.float32),
        X_train_raw["dropoff_longitude"].values.astype(np.float32),
        X_train_raw["dropoff_latitude"].values.astype(np.float32),
        X_train_raw["passenger_count"].values.astype(np.float32),
        X_train_raw["pickup_hour"].values.astype(np.float32),
        X_train_raw["pickup_weekday"].values.astype(np.float32),
        X_train_raw["pickup_month"].values.astype(np.float32),
        distance_train.astype(np.float32),
        manhattan_train.astype(np.float32),
        lat_diff_train,
        lon_diff_train,
        hour_passenger_train,
        hour_sin_train,
        hour_cos_train,
        distance_passenger_train,
        distance_log_train,
        passenger_log_train,
        distance_per_passenger_train,
    ]
)

X_test_np = np.column_stack(
    [
        X_test_raw["pickup_longitude"].values.astype(np.float32),
        X_test_raw["pickup_latitude"].values.astype(np.float32),
        X_test_raw["dropoff_longitude"].values.astype(np.float32),
        X_test_raw["dropoff_latitude"].values.astype(np.float32),
        X_test_raw["passenger_count"].values.astype(np.float32),
        X_test_raw["pickup_hour"].values.astype(np.float32),
        X_test_raw["pickup_weekday"].values.astype(np.float32),
        X_test_raw["pickup_month"].values.astype(np.float32),
        distance_test.astype(np.float32),
        manhattan_test.astype(np.float32),
        lat_diff_test,
        lon_diff_test,
        hour_passenger_test,
        hour_sin_test,
        hour_cos_test,
        distance_passenger_test,
        distance_log_test,
        passenger_log_test,
        distance_per_passenger_test,
    ]
)

y_train_log_np = y_train_log.values.astype(np.float32)

del X_train_raw, X_test_raw, y_train, y_train_log
gc.collect()



## === cell 5
cv = KFold(n_splits=5, shuffle=True, random_state=0)
categorical_features = []  # no categorical features

params = {
    "objective": "regression",
    "max_bin": 255,
    "learning_rate": 0.05,
    "num_leaves": 256,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "metric": "rmse",
    "verbosity": -1,
    "min_data_in_leaf": 20,
    "lambda_l2": 0.1,
    "seed": 0,
}

from concurrent.futures import ThreadPoolExecutor, as_completed


def train_one_fold(fold_id, train_idx, val_idx):
    fold_params = params.copy()
    fold_params["num_threads"] = 1

    X_tr, X_val = X_train_np[train_idx], X_train_np[val_idx]
    y_tr, y_val = y_train_log_np[train_idx], y_train_log_np[val_idx]

    lgb_train = lgb.Dataset(X_tr, y_tr, categorical_feature=categorical_features)
    lgb_valid = lgb.Dataset(
        X_val, y_val, reference=lgb_train, categorical_feature=categorical_features
    )

    model = lgb.train(
        fold_params,
        lgb_train,
        num_boost_round=2000,
        valid_sets=[lgb_train, lgb_valid],
        callbacks=[
            lgb.early_stopping(stopping_rounds=20, verbose=False),
            lgb.log_evaluation(period=10, show_stdv=False),
        ],
    )

    oof_pred = np.expm1(model.predict(X_val, num_iteration=model.best_iteration))
    test_pred = np.expm1(model.predict(X_test_np, num_iteration=model.best_iteration))

    return {
        "fold_id": fold_id,
        "val_idx": val_idx,
        "oof_pred": oof_pred,
        "test_pred": test_pred,
        "model": model,
    }


oof_train = np.zeros(len(X_train_np), dtype=np.float32)
y_preds = []
models = []

with ThreadPoolExecutor(max_workers=5) as executor:
    futures = {
        executor.submit(train_one_fold, fid, train_idx, val_idx): fid
        for fid, (train_idx, val_idx) in enumerate(cv.split(X_train_np, y_train_log_np))
    }
    for fut in as_completed(futures):
        res = fut.result()
        fold_id = res["fold_id"]
        oof_train[res["val_idx"]] = res["oof_pred"]
        y_preds.append(res["test_pred"])
        models.append(res["model"])



## === cell 6
bias = (y_train_log_np - oof_train).mean()
oof_train += bias
y_preds = [p + bias for p in y_preds]

scores = [m.best_score["valid_1"]["rmse"] for m in models]
mean_cv = sum(scores) / len(scores) if scores else float("nan")
print("===CV scores (log‑target)===")
print(scores)
print("Mean CV RMSE (log‑target):", mean_cv)



## === cell 7
print(
    "OOF RMSE (original scale):",
    np.sqrt(mean_squared_error(np.expm1(y_train_log_np), oof_train)),
)



## === cell 8
print("Number of folds predictions collected:", len(y_preds))



## === cell 9
y_sub = np.mean(y_preds, axis=0) if y_preds else np.zeros(len(X_test_np))
print("First 10 averaged predictions:", y_sub[:10])



## === cell 10
sub_lgb = sample_submission.copy()
sub_lgb["fare_amount"] = y_sub
sub_lgb.to_csv("submission_lightgbm.csv", index=False)
print("Submission file written: submission_lightgbm.csv")
print(sub_lgb.head())
