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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
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

4.46345

# 6. Current score

8.9702

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.05635) has done: 'I remove the TensorFlow install/downgrade cell that crashes due to protobuf incompatibilities in this environment, since TensorFlow isn’t used anywhere in the solution. I also fix the datetime feature engineering so it doesn’t break on non-UTC strings and avoids slow row-wise `.apply`, while keeping the same engineered features and dropped columns. Finally, I ensure the train/test filtering doesn’t accidentally drop test rows (which would mismatch the required submission keys) and that the submission is written as a valid `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 5.99995) has done: 'We make the model selection consistent with the competition metric (RMSE) by using squared error for training and RMSE on the validation split, instead of optimizing MAE while the leaderboard uses RMSE. We also fix determinism and split stability by setting `random_state`, and we make the final training use the best parameters found by the (same) small grid search rather than hardcoding `(700, 0.25)`, which is likely suboptimal for RMSE. These are minimal changes that keep the same features, model family (XGBRegressor), and overall training approach, but should move your score downward toward the 4.46345 target. The script still write a valid `submission.csv` with `key,fare_amount` and the full test row alignment.'
- What this solution (achieved 6.24343) has done: 'Your current score (5.99995 RMSE) is worse than the target (4.46345), so we should make a small, legitimate improvement without changing the overall approach (same features + XGBRegressor + simple split + small hyperparameter search). The biggest score drag here is that you only train on 10,000 rows and only remove one obvious bad condition (dropoff_longitude==0), leaving many outliers/noise that badly hurt RMSE; we modestly increase the training sample size (still safe under 600s) and apply standard NYC Taxi Fare sanity filters on coordinates, passenger_count, and fare_amount. We also train the final model on all filtered training data using the best hyperparameters found (instead of only X_train), which improves generalization for the submission while keeping the same model/logic. These changes should move RMSE downward toward your target while preserving core semantics and producing a valid `submission.csv`.'
- What this solution (achieved 15.31394) has done: 'We keep your exact feature engineering and XGBRegressor approach, but make two minimal changes that typically reduce RMSE for this competition: (1) compute distance in a more realistic way (Haversine distance in km) while still retaining `x_dis` and `y_dis`, and (2) add two simple location features (`pickup_distance_to_jfk`, `pickup_distance_to_manhattan`) derived from the same coordinates you already use, which helps the model learn airport/Manhattan fare structure without changing the model or training loop. We not change your grid search/training procedure, but we increase `n_jobs` to use all available cores (faster, same semantics) and ensure train/test are featurized identically. These are small, legitimate feature tweaks aimed at moving RMSE down from 6.243 toward your 4.463 target while still producing a valid `submission.csv`.'
- What this solution (achieved 15.31394) has done: 'Your current RMSE (15.31) is far worse than the target (4.46), and the biggest issue is that the model is being trained on a dataset indexed by `key`, but `key` is **not unique** in this competition, so `train_test_split` is effectively splitting with duplicated indices and can misalign/overlap examples, producing a very degraded model. I make the minimal fix of reading the CSVs **without** using `key` as the index, then set `key` aside only for the submission output (keeping identical features/model/grid-search logic). I also add a tiny safety step to ensure all feature columns are numeric and consistent between train/test after feature engineering (no semantic change, but prevents subtle dtype issues). These changes should legitimately move RMSE down substantially toward your target while keeping the same core approach and still writing a valid `submission.csv`.'
- What this solution (achieved 16.87699) has done: 'Your current RMSE (15.31) is far worse than the target (4.46), so we should make a small, legitimate improvement without changing the core approach (same feature engineering style + XGBRegressor + simple holdout + small hyperparameter loop). The biggest likely score killer is that the model has no “trip distance” signal that correlates well with fare because `dis` is computed, but the *dropoff_latitude* is dropped, and *pickup_latitude/longitudes* are dropped too—so the model effectively can’t infer a sensible trip distance for many cases. I make the minimal fix of keeping both dropoff latitude and pickup coordinates (instead of dropping them), while still keeping the same engineered time and distance features and the same XGB training loop; this typically drops RMSE a lot on this competition. I also ensure `passenger_count` stays in the features (it currently does) and keep submission alignment unchanged.'
- What this solution (achieved 7.18885) has done: 'Your current score (16.87699 RMSE) is far worse than the target (4.46345), and the most likely cause is that you’re training on a non-representative slice of the data: the first 200k rows are strongly time-ordered and can bias the model, hurting generalization on the (time-mixed) test set. To move RMSE down toward the target with minimal changes and identical modeling logic, we keep the same features and XGBRegressor grid-search, but change the training data read to a deterministic random sample (via `skiprows`) from the full file while still only loading ~200k rows. This preserves the core approach while making the training distribution closer to the test distribution, which typically improves RMSE substantially on this competition. We also keep test loading unchanged and still write a valid `submission.csv` with `key,fare_amount` aligned to all test rows.'
- What this solution (achieved 14.8603) has done: 'Your current RMSE (7.18885, lower is better) is still far above the target (4.46345), so we should make a small, legitimate improvement without changing your overall approach (same feature set, same XGBRegressor, same simple holdout + small grid search). The biggest issue in your current code is the way you “randomly sample” the huge training CSV: building a `set` of ~55M skiprows is infeasible/slow and can silently break sampling/reading behavior; instead we use a deterministic, chunk-based reservoir-style sample that reads the file once and keeps only `N_TRAIN_ROWS`. Then we keep your exact outlier filters but also apply the same `dropna` + datetime coercion guard as you already do in `preparedataset2`, ensuring no hidden NaT rows degrade training. This should reduce noise and make the training sample truly representative, moving RMSE down toward the target while keeping the core logic intact and still writing a valid `submission.csv`.'
- What this solution (achieved 8.9702) has done: 'Your current RMSE (14.8603) is far worse than the target (4.46345), so we should make a small, safe improvement without changing your core approach (same XGBRegressor, same engineered time+distance features, same holdout/grid search). The biggest likely issue is your chunk-sampling: it “fills” the sample by taking an equal number from each early chunk, which biases heavily toward the start of the file (time-ordered), hurting generalization. I replace it with a true uniform random sample over the whole CSV using chunk-level reservoir sampling (single pass, fixed memory, deterministic), while keeping the same training size and downstream logic. This should legitimately reduce RMSE toward the target band while preserving semantics and still writing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_iop_path = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
test_iop_path = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"

N_TRAIN_ROWS = 200_000
N_TEST_ROWS = 10_000  # test has ~9914 rows; keep a safe upper bound


def read_train_sample_chunked(path, n_rows, seed=42, chunksize=250_000):
    rng = np.random.RandomState(seed)

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

    reservoir = None
    seen = 0

    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        chunk = chunk.reset_index(drop=True)
        m = len(chunk)

        if reservoir is None:
            take = min(n_rows, m)
            reservoir = chunk.iloc[:take].copy()
            seen = take

            if take < m:
                rest = chunk.iloc[take:].reset_index(drop=True)
                m2 = len(rest)
                j = rng.randint(0, seen + np.arange(1, m2 + 1))
                replace_mask = j < n_rows
                if replace_mask.any():
                    idx_res = j[replace_mask]
                    reservoir.iloc[idx_res] = rest.loc[replace_mask].to_numpy()
                seen += m2
            continue

        j = rng.randint(0, seen + np.arange(1, m + 1))
        replace_mask = j < n_rows
        if replace_mask.any():
            idx_res = j[replace_mask]
            reservoir.iloc[idx_res] = chunk.loc[replace_mask].to_numpy()
        seen += m

    if reservoir is None:
        return pd.DataFrame(columns=usecols)

    reservoir = reservoir.reset_index(drop=True)
    return reservoir


dataset_train = read_train_sample_chunked(
    train_iop_path, N_TRAIN_ROWS, seed=42, chunksize=250_000
)
dataset_test = pd.read_csv(test_iop_path, nrows=N_TEST_ROWS)



## === cell 1
print("dataset_train old size", len(dataset_train))

dataset_train = dataset_train.dropna()

dataset_train = dataset_train[dataset_train["fare_amount"].between(2.5, 500)]
dataset_train = dataset_train[dataset_train["passenger_count"].between(1, 6)]

dataset_train = dataset_train[dataset_train["pickup_longitude"].between(-74.5, -72.8)]
dataset_train = dataset_train[dataset_train["dropoff_longitude"].between(-74.5, -72.8)]
dataset_train = dataset_train[dataset_train["pickup_latitude"].between(40.5, 41.8)]
dataset_train = dataset_train[dataset_train["dropoff_latitude"].between(40.5, 41.8)]

dataset_train = dataset_train[
    (dataset_train["pickup_longitude"] != 0)
    & (dataset_train["pickup_latitude"] != 0)
    & (dataset_train["dropoff_longitude"] != 0)
    & (dataset_train["dropoff_latitude"] != 0)
]

dataset_train = dataset_train[
    ~(
        (dataset_train["pickup_longitude"] == dataset_train["dropoff_longitude"])
        & (dataset_train["pickup_latitude"] == dataset_train["dropoff_latitude"])
    )
]

print("new size", len(dataset_train))
dataset_train.head(5)



## === cell 2
print("dataset_test old size", len(dataset_test))

dataset_test = dataset_test.dropna()

print("new size", len(dataset_test))
dataset_test.head(5)




## === cell 3
def get_year(pickup_date):
    return pickup_date.year


def get_month(pickup_date):
    return pickup_date.month


def get_day(pickup_date):
    return pickup_date.day


def get_hour(pickup_date):
    return pickup_date.hour




## === cell 4
import warnings


def _haversine_km(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1.astype(float))
    lat1 = np.radians(lat1.astype(float))
    lon2 = np.radians(lon2.astype(float))
    lat2 = np.radians(lat2.astype(float))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c  # Earth radius (km)


def preparedataset2(datasetname: pd.DataFrame) -> pd.DataFrame:
    warnings.filterwarnings("ignore")

    datasetname = datasetname.copy()
    datasetname["pickup_datetime"] = pd.to_datetime(
        datasetname["pickup_datetime"], errors="coerce", utc=True
    )
    datasetname["pickup_datetime"] = datasetname["pickup_datetime"].fillna(
        pd.Timestamp("2009-01-01", tz="UTC")
    )

    datasetname["pickup_year"] = datasetname["pickup_datetime"].dt.year.astype(np.int16)
    datasetname["pickup_month"] = datasetname["pickup_datetime"].dt.month.astype(
        np.int8
    )
    datasetname["pickup_day"] = datasetname["pickup_datetime"].dt.day.astype(np.int8)
    datasetname["pickup_hour"] = datasetname["pickup_datetime"].dt.hour.astype(np.int8)

    datasetname["x_dis"] = (
        datasetname["dropoff_longitude"] - datasetname["pickup_longitude"]
    )
    datasetname["y_dis"] = (
        datasetname["dropoff_latitude"] - datasetname["pickup_latitude"]
    )

    datasetname["dis"] = _haversine_km(
        datasetname["pickup_longitude"],
        datasetname["pickup_latitude"],
        datasetname["dropoff_longitude"],
        datasetname["dropoff_latitude"],
    )

    datasetname["pickup_distance_to_jfk"] = _haversine_km(
        datasetname["pickup_longitude"],
        datasetname["pickup_latitude"],
        np.full(len(datasetname), -73.7781),
        np.full(len(datasetname), 40.6413),
    )
    datasetname["pickup_distance_to_manhattan"] = _haversine_km(
        datasetname["pickup_longitude"],
        datasetname["pickup_latitude"],
        np.full(len(datasetname), -73.9855),
        np.full(len(datasetname), 40.7580),
    )

    datasetname = datasetname.drop(["pickup_datetime"], axis=1)

    return datasetname




## === cell 5
from datetime import datetime as dt

warnings.filterwarnings("ignore")


def preparedataset(datasetname):
    datasetname["pickup_year"] = 0
    datasetname["pickup_month"] = 0
    datasetname["pickup_day"] = 0
    datasetname["pickup_hour"] = 0
    datasetname["dis"] = 0
    datasetname["x_dis"] = 0
    datasetname["y_dis"] = 0

    datasetname.head()

    for k in range(len(datasetname.index)):
        datetime = dt.strptime(
            datasetname["pickup_datetime"][k].replace("UTC", ""), "%Y-%m-%d %H:%M:%S "
        )
        datasetname["pickup_year"][k] = datetime.year
        datasetname["pickup_month"][k] = datetime.month
        datasetname["pickup_day"][k] = datetime.day
        datasetname["pickup_hour"][k] = datetime.hour

    datasetname["x_dis"] = (
        datasetname["dropoff_longitude"] - datasetname["pickup_longitude"]
    )
    datasetname["y_dis"] = (
        datasetname["dropoff_latitude"] - datasetname["pickup_latitude"]
    )
    datasetname["dis"] = (
        (datasetname["dropoff_longitude"] - datasetname["pickup_longitude"]) ** 2
        + (datasetname["dropoff_latitude"] - datasetname["pickup_latitude"]) ** 2
    ) ** 0.5
    datasetname = datasetname.drop(["pickup_datetime"], axis=1)
    datasetname = datasetname.drop(["pickup_longitude"], axis=1)
    datasetname = datasetname.drop(["dropoff_latitude"], axis=1)
    datasetname = datasetname.drop(["dropoff_longitude"], axis=1)
    datasetname = datasetname.drop(["pickup_latitude"], axis=1)

    return datasetname




## === cell 6
train_keys = dataset_train["key"].copy()
test_keys = dataset_test["key"].copy()

df = preparedataset2(dataset_train)

df = df.dropna()

df.head(5)



## === cell 7
test_df = preparedataset2(dataset_test)

test_df = test_df.fillna(0.0)

test_df.head(5)



## === cell 8
from sklearn.model_selection import train_test_split

y = df.fare_amount.astype(float)
X = df.drop("fare_amount", axis=1)

X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
test_df = test_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

test_df = test_df.reindex(columns=X.columns, fill_value=0.0)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=42)



## === cell 9
import warnings

warnings.filterwarnings("ignore")

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

N_JOBS = -1

result = {}
best_istemator = 0
best_learing_rate = 0
best_rmse = float("inf")

for lr in [0.05, 0.1, 0.15, 0.2, 0.25]:
    for ns in [300, 400, 500, 600, 700, 800]:
        my_model = XGBRegressor(
            n_estimators=ns,
            learning_rate=lr,
            n_jobs=N_JOBS,
            objective="reg:squarederror",
            random_state=42,
        )
        my_model.fit(X_train, y_train)

        predictions = my_model.predict(X_valid)
        rmse = mean_squared_error(y_valid, predictions, squared=False)

        if rmse < best_rmse:
            best_rmse = rmse
            best_istemator = ns
            best_learing_rate = lr
            print("better found")
            print(ns, lr, rmse)
        result[(ns, lr)] = rmse

my_model_2 = XGBRegressor(
    n_estimators=best_istemator,
    learning_rate=best_learing_rate,
    n_jobs=N_JOBS,
    objective="reg:squarederror",
    random_state=42,
)
my_model_2.fit(X_train, y_train)
predictions_2 = my_model_2.predict(X_valid)
rmse_2 = mean_squared_error(y_valid, predictions_2, squared=False)

print("best_istemator:", best_istemator)
print("best_learing_rate:", best_learing_rate)
print("Validation RMSE:", rmse_2)



## === cell 10
from xgboost import XGBRegressor

final_model = XGBRegressor(
    n_estimators=best_istemator,
    learning_rate=best_learing_rate,
    n_jobs=-1,
    objective="reg:squarederror",
    random_state=42,
)
final_model.fit(X, y)

test_preds = final_model.predict(test_df)

test_preds = np.clip(test_preds, 0, None)

output = pd.DataFrame({"key": test_keys.values, "fare_amount": test_preds})
output.to_csv("submission.csv", index=False)

print(output.head())
print("Wrote submission.csv with shape:", output.shape)
print("Submission columns:", list(output.columns))
print("Unique keys in submission:", output["key"].nunique(), "out of", len(output))
