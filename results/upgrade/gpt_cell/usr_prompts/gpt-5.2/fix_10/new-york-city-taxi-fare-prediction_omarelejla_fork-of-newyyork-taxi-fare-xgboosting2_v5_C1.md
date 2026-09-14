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

4.73415

# 6. Current score

6.56805

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.36849) has done: 'Your code currently can’t reasonably hit the target RMSE because it (a) trains on only 1,000 rows, (b) drops test rows (so the submission row count won’t match), and (c) tunes using MAE while the competition is scored with RMSE. I make minimal changes that keep the same XGBoost regressor + same engineered features, but (1) increase training rows within the 600s budget, (2) stop filtering the test set and instead keep all keys (and handle bad rows via safe feature creation), and (3) switch the in-notebook validation metric to RMSE and set the model objective to squared error to better align with the Kaggle metric. This should produce a valid `submission.csv` with exactly the same keys as `test.csv` and move the score downward toward your target.'
- What this solution (achieved 5.4398) has done: 'The timeout is dominated by (1) the extremely slow `skiprows=lambda ...` sampling (it executes Python for ~55M rows) and (2) the 72-fit nested grid search in XGBoost. To keep the same core model/feature logic and identical evaluation semantics, this refactor replaces the slow CSV sampling with a single fast deterministic sample built from a small set of contiguous row blocks (plus an exact top-up), and it keeps the exact same hyperparameter grid but evaluates it in parallel using `joblib` so the wall-clock time drops sharply. It also avoids redundant copies and ensures consistent dtypes/column order so results remain stable aside from negligible floating-point differences. File paths and the feature engineering/model architecture/loss remain unchanged.'
- What this solution (achieved 6.56805) has done: 'The timeout is dominated by two hotspots: (1) the reservoir sampling loop does per-row `iloc` assignment for the full CSV stream (tens of millions of Python-level iterations), and (2) the XGBoost grid search trains 64 models with 650 trees each and then re-predicts 8 times per model, which is far too slow. I keep the exact same sampling semantics (uniform reservoir sampling) but implement it in a vectorized, chunk-wise way that avoids per-row DataFrame operations while remaining deterministic. I also keep the same XGBoost hyperparameter search logic and evaluation semantics, but reduce work by using XGBoost’s built-in early stopping on the validation set to find the best `n_estimators` for each hyperparameter combo (equivalent to scanning iteration counts, but without redundant predictions and without training trees that won’t be used). Finally, I cache/cast NumPy arrays once and avoid repeated conversions and repeated per-iteration prediction loops.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import (
    RandomForestRegressor,
)  # kept (not used) to preserve core structure

import os
import time

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_iop_path = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
test_iop_path = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"

TRAIN_NROWS = 200_000
RANDOM_SEED = 42

USECOLS = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
DTYPES = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}


def reservoir_sample_csv(
    path,
    sample_size,
    usecols,
    dtypes,
    seed,
    chunksize=200_000,
):
    rng = np.random.RandomState(seed)

    reservoir = None
    seen = 0

    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        dtype=dtypes,
        chunksize=chunksize,
    ):
        m = len(chunk)
        if reservoir is None:
            if m >= sample_size:
                reservoir = chunk.iloc[:sample_size].copy()
                seen = sample_size
                start_idx = sample_size
            else:
                reservoir = chunk.copy()
                seen = m
                start_idx = m
        else:
            start_idx = 0

        if start_idx >= m:
            continue

        remaining = m - start_idx
        t = np.arange(seen + 1, seen + remaining + 1, dtype=np.int64)
        j = rng.randint(0, t, size=remaining, dtype=np.int64)
        take_mask = j < sample_size

        if np.any(take_mask):
            rep_idx = j[take_mask].astype(np.int64, copy=False)
            src_idx = np.nonzero(take_mask)[0] + start_idx  # row indices in chunk
            reservoir.iloc[rep_idx] = chunk.iloc[src_idx].to_numpy()

        seen += remaining

    return reservoir


dataset_train = reservoir_sample_csv(
    train_iop_path,
    sample_size=TRAIN_NROWS,
    usecols=USECOLS,
    dtypes=DTYPES,
    seed=RANDOM_SEED,
    chunksize=250_000,
)

dataset_train = dataset_train.drop_duplicates(subset=["key"], keep="first").set_index(
    "key"
)
if len(dataset_train) < TRAIN_NROWS:
    need = TRAIN_NROWS - len(dataset_train)
    topup = pd.read_csv(
        train_iop_path,
        usecols=USECOLS,
        dtype=DTYPES,
        nrows=need,
    ).set_index("key")
    dataset_train = pd.concat([dataset_train, topup], axis=0)
    dataset_train = dataset_train[~dataset_train.index.duplicated(keep="first")]
    if len(dataset_train) > TRAIN_NROWS:
        dataset_train = dataset_train.sample(n=TRAIN_NROWS, random_state=RANDOM_SEED)

dataset_test = pd.read_csv(
    test_iop_path,
    nrows=10000,
    index_col="key",
    dtype={
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "int8",
    },
)

print("train shape:", dataset_train.shape)
print("test shape:", dataset_test.shape)



## === cell 1
print("dataset_train old size", len(dataset_train))

dataset_train = dataset_train.dropna()

dataset_train = dataset_train[
    (dataset_train["fare_amount"] > 0) & (dataset_train["fare_amount"] < 500)
]
dataset_train = dataset_train[
    (dataset_train["passenger_count"] > 0) & (dataset_train["passenger_count"] <= 6)
]

dataset_train = dataset_train[
    (dataset_train["pickup_longitude"].between(-74.5, -72.8))
    & (dataset_train["dropoff_longitude"].between(-74.5, -72.8))
    & (dataset_train["pickup_latitude"].between(40.5, 41.8))
    & (dataset_train["dropoff_latitude"].between(40.5, 41.8))
]

dataset_train = dataset_train[dataset_train.dropoff_longitude != 0]

print("dataset_train new size", len(dataset_train))
dataset_train.head(5)



## === cell 2
print("dataset_test size (kept intact for submission alignment)", len(dataset_test))
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
from datetime import datetime as dt
import warnings


def preparedataset2(datasetname):
    warnings.filterwarnings("ignore")

    df = datasetname.copy()

    dt_series = pd.to_datetime(
        df["pickup_datetime"].astype(str).str.replace("UTC", "", regex=False),
        errors="coerce",
    )

    df["pickup_year"] = dt_series.dt.year.fillna(0).astype(np.int16)
    df["pickup_month"] = dt_series.dt.month.fillna(0).astype(np.int8)
    df["pickup_day"] = dt_series.dt.day.fillna(0).astype(np.int8)
    df["pickup_hour"] = dt_series.dt.hour.fillna(0).astype(np.int8)

    df["x_dis"] = df["dropoff_longitude"] - df["pickup_longitude"]
    df["y_dis"] = df["dropoff_latitude"] - df["pickup_latitude"]
    df["dis"] = np.sqrt(
        (df["dropoff_longitude"] - df["pickup_longitude"]) ** 2
        + (df["dropoff_latitude"] - df["pickup_latitude"]) ** 2
    )

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for c in [
        "pickup_year",
        "pickup_month",
        "pickup_day",
        "pickup_hour",
        "x_dis",
        "y_dis",
        "dis",
    ]:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    df = df.drop(["pickup_datetime"], axis=1)
    df = df.drop(["pickup_longitude"], axis=1)
    df = df.drop(["dropoff_latitude"], axis=1)
    df = df.drop(["dropoff_longitude"], axis=1)
    df = df.drop(["pickup_latitude"], axis=1)

    return df




## === cell 5
from datetime import datetime as dt
import warnings

warnings.filterwarnings("ignore")


def preparedataset(datasetname):
    datasetname = datasetname.copy()
    datasetname["pickup_year"] = 0
    datasetname["pickup_month"] = 0
    datasetname["pickup_day"] = 0
    datasetname["pickup_hour"] = 0
    datasetname["dis"] = 0
    datasetname["x_dis"] = 0
    datasetname["y_dis"] = 0

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
df = preparedataset2(dataset_train)
df.head(5)



## === cell 7
test_df = preparedataset2(dataset_test)
test_df.head(5)



## === cell 8
from sklearn.model_selection import train_test_split

y = df.fare_amount
X = df.drop("fare_amount", axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, random_state=42, test_size=0.2
)



## === cell 9
import warnings

warnings.filterwarnings("ignore")

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

result = {}
best_istemator = 0
best_learing_rate = 0
best_rmse = 1e18

best_max_depth = 0
best_subsample = 0
best_colsample = 0
best_min_child_weight = 0

N_JOBS = max(1, (os.cpu_count() or 4) - 1)

lrs = [x / 100 for x in range(15, 41, 5)]  # 0.15..0.40
nss = list(range(300, 651, 50))  # 300..650
max_depths = [6, 8]
subsamples = [0.8, 1.0]
colsamples = [0.8, 1.0]
min_child_weights = [1, 5]

max_ns = max(nss)

Xtr_np = np.ascontiguousarray(X_train.to_numpy(dtype=np.float32))
Xva_np = np.ascontiguousarray(X_valid.to_numpy(dtype=np.float32))
ytr_np = np.ascontiguousarray(y_train.to_numpy(dtype=np.float32))
yva_np = np.ascontiguousarray(y_valid.to_numpy(dtype=np.float32))


for lr in lrs:
    for md in max_depths:
        for ss in subsamples:
            for cs in colsamples:
                for mcw in min_child_weights:
                    model = XGBRegressor(
                        n_estimators=max_ns,
                        learning_rate=lr,
                        max_depth=md,
                        subsample=ss,
                        colsample_bytree=cs,
                        min_child_weight=mcw,
                        n_jobs=N_JOBS,
                        objective="reg:squarederror",
                        random_state=42,
                        tree_method="hist",
                        eval_metric="rmse",
                    )

                    model.fit(
                        Xtr_np,
                        ytr_np,
                        eval_set=[(Xva_np, yva_np)],
                        verbose=False,
                        early_stopping_rounds=50,
                    )

                    best_ns_this = int(getattr(model, "best_iteration", max_ns - 1)) + 1

                    preds = model.predict(Xva_np, iteration_range=(0, best_ns_this))
                    rmse = mean_squared_error(yva_np, preds, squared=False)

                    result[(best_ns_this, lr, md, ss, cs, mcw)] = rmse
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_istemator = best_ns_this
                        best_learing_rate = lr
                        best_max_depth = md
                        best_subsample = ss
                        best_colsample = cs
                        best_min_child_weight = mcw
                        print("better found")
                        print(best_ns_this, lr, md, ss, cs, mcw, rmse)

my_model_2 = XGBRegressor(
    n_estimators=best_istemator,
    learning_rate=best_learing_rate,
    max_depth=best_max_depth,
    subsample=best_subsample,
    colsample_bytree=best_colsample,
    min_child_weight=best_min_child_weight,
    n_jobs=N_JOBS,
    objective="reg:squarederror",
    random_state=42,
    tree_method="hist",
    eval_metric="rmse",
)
my_model_2.fit(Xtr_np, ytr_np)

predictions_2 = my_model_2.predict(Xva_np)
rmse_2 = mean_squared_error(yva_np, predictions_2, squared=False)

print("best_istemator:", best_istemator)
print("best_learing_rate:", best_learing_rate)
print("best_max_depth:", best_max_depth)
print("best_subsample:", best_subsample)
print("best_colsample:", best_colsample)
print("best_min_child_weight:", best_min_child_weight)
print("Validation RMSE:", rmse_2)



## === cell 10
from xgboost import XGBRegressor

X_np = np.ascontiguousarray(X.to_numpy(dtype=np.float32))
y_np = np.ascontiguousarray(y.to_numpy(dtype=np.float32))
test_np = np.ascontiguousarray(test_df.to_numpy(dtype=np.float32))

my_model_2 = XGBRegressor(
    n_estimators=best_istemator if best_istemator else 650,
    learning_rate=best_learing_rate if best_learing_rate else 0.2,
    max_depth=best_max_depth if best_max_depth else 8,
    subsample=best_subsample if best_subsample else 0.8,
    colsample_bytree=best_colsample if best_colsample else 0.8,
    min_child_weight=best_min_child_weight if best_min_child_weight else 1,
    n_jobs=N_JOBS,
    objective="reg:squarederror",
    random_state=42,
    tree_method="hist",
    eval_metric="rmse",
)
my_model_2.fit(X_np, y_np)

test_preds = my_model_2.predict(test_np)

output = pd.DataFrame({"key": test_df.index, "fare_amount": test_preds})
output.to_csv("submission.csv", index=False)

print("Wrote submission.csv with rows:", len(output))
print(output.head())
