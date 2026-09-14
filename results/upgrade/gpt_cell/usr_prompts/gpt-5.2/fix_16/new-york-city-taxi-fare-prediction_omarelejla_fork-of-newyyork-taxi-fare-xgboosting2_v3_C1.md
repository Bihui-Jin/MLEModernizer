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

5.62595

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.98406) has done: 'I first make the notebook reliably produce a valid `submission.csv` by preserving `key` alignment and preventing the test-set row filtering from dropping rows (which can invalidate the submission row count). Then I keep your exact feature engineering and XGBoost approach, but make a minimal metric-consistent tweak: select hyperparameters using RMSE (the competition metric) instead of MAE, so the chosen model moves the public RMSE closer to your target. Finally, I train the final model using the best-found parameters (instead of hard-coding `700, 0.25`) and add a small safety clip to keep fares non-negative.'
- What this solution (achieved 6.03673) has done: 'You’re currently worse than the target (5.98406 vs 4.46345; lower is better), so the smallest safe way to move RMSE downward is to reduce feature/label noise and make the XGBoost training objective match the competition metric more directly. I keep your exact feature engineering and overall XGBoost approach, but (1) clean only the training rows with clearly invalid coordinates/fare values (test is left untouched to preserve row count), (2) use XGBoost’s `objective='reg:squarederror'` and `eval_metric='rmse'` for metric-consistent fitting, and (3) retrain the final model on all cleaned training data using the best hyperparameters found. These are minimal changes that typically reduce RMSE materially without altering your core logic or adding early stopping/sampling.'
- What this solution (achieved 5.91892) has done: 'You’re currently worse than the target (6.03673 vs 4.46345; lower is better), so we should make a small, legitimate improvement that reduces label noise without changing your core model or feature logic. The biggest issue is you only train on 10,000 rows (which is very weak for this competition); we safely increase `nrows` to use more training data while keeping runtime under 600s and keeping your same XGBoost + same engineered features. To stabilize training and reduce outlier impact (still consistent with RMSE objective), we also add two very standard filters: remove rows where pickup and dropoff are identical (distance ~0 but non-min fare) and remove extreme distances that are almost surely bad data. Everything else (feature engineering, train/valid split, hyperparameter search, objective/metric, and submission format) stays the same.'
- What this solution (achieved 6.67873) has done: 'You’re currently worse than the target (5.91892 vs 4.46345; lower is better), so we should make a small, legitimate improvement that reduces RMSE without changing your core XGBoost + engineered-features approach. The biggest RMSE drag in this competition is remaining noisy/outlier training rows, so I add two very standard, minimal filters on the training set only: remove obviously swapped lat/lon (lat outside NYC range or lon outside NYC range) and remove extreme “fare per distance” outliers (implausibly low/high $/degree). I also make the datetime feature extraction use vectorized `.dt` accessors (same semantics, less overhead) so we can afford a bit more training data within the 600s budget, which usually improves RMSE. Everything else (features, model type, objective/metric, split, hyperparameter search, submission format) is preserved.'
- What this solution (achieved 6.28218) has done: 'You’re currently above the target RMSE (6.67873 vs 4.46345; lower is better), so the smallest safe move is to reduce training label noise without changing your XGBoost model class or your existing engineered features. I keep your exact feature engineering and your train/valid hyperparameter selection loop, but tighten the training-only cleaning with two standard NYC Taxi filters that usually cut RMSE a lot: (1) bounding box around NYC core area (removes out-of-city coordinates) and (2) fare-per-km sanity bounds using a proper Haversine distance (removes implausible fare/distance outliers). Test rows not be filtered to preserve the required submission row count and key alignment. I also set `n_jobs=-1` for XGBoost to use available CPU within the same training procedure and time budget.'
- What this solution (achieved 5.97436) has done: 'Diagnosis: Cell 9 crashes because it references `df`, which is never defined in earlier cells; the prepared/cleaned training data exists as `dataset_train`. Patch summary: In cell 9, create `df` by applying the existing `preparedataset2()` feature engineering to `dataset_train`, then proceed with the same train/valid split logic using `df.fare_amount` and `df.drop(...)`. Updated cells: Only cell 9 is changed to define `df` deterministically and ensure any datetime-derived columns have NaNs filled similarly to the test preprocessing. Compatibility notes for cell k+1: The patch preserves `X_train, X_valid, y_train, y_valid` exactly as expected by cell 10, so the XGBoost loop remains unchanged. Assumptions: `preparedataset2` is the intended preprocessing function for both train and test (it already exists and is used for test in cell 8), and `dataset_train` still contains `fare_amount` prior to preprocessing.'
- What this solution (achieved 5.97436) has done: 'You’re currently worse than the target (RMSE 5.97436 vs 4.46345; lower is better), so we should make the smallest legitimate change that usually reduces RMSE without changing your XGBoost core approach. The biggest gain with minimal semantic change is to stop using a random train/valid split and instead split by time, because this competition’s test data is later in time; this reduces distribution shift and typically improves public RMSE. I keep your feature engineering and training loop intact, but change the split in cell 9 to a deterministic chronological split based on `pickup_datetime` from the original `dataset_train`. I also remove the duplicated/late re-definition of `haversine_km` by ensuring it’s defined once before `preparedataset2` is used, so your engineered distance features are consistent.'
- What this solution (achieved 5.97436) has done: 'Diagnosis: `haversine_km(...)` returns a NumPy `ndarray`, so calling `trip_km.between(...)` fails because `.between` is a pandas `Series` method. The code later tries to realign `trip_km` with `dataset_train.index`, but it converts to `Series` only after using `.between`, so the crash happens first.

Patch summary: In cell 2, convert `trip_km` to a pandas `Series` immediately after computing it (using the current `dataset_train.index`) so `.between(...)` works and the boolean mask aligns with the dataframe. Keep the rest of the filtering logic identical.

Updated cells: cell 2 only.

Compatibility notes for cell k+1: Cell 3 uses `dataset_test` only; this change does not alter `dataset_test`. `dataset_train` remains a filtered DataFrame as originally intended, and `trip_km` remains available as a Series for subsequent computations in cell 2.

Assumptions: `dataset_train.index` is stable at the time `trip_km` is computed (true here), and the intent was to filter using trip distance while preserving index alignment.'
- What this solution (achieved 5.62595) has done: 'Your current RMSE (5.97436) is worse than the target (4.46345), so we should make a small, legitimate improvement that typically reduces error without changing your core XGBoost + feature-engineering approach. The biggest low-risk gain here is to remove a mild train/test mismatch: you train with `passenger_count` possibly being 0/7+ in test (not filtered) but you filtered them out in train; instead, keep train filtering as-is but clamp test `passenger_count` into the same [1, 6] range so the model sees consistent inputs. Second, we add a minimal log-transform on the target (train) with inverse-transform on predictions (test), while keeping the same model type/objective; this often reduces RMSE for this competition by stabilizing the heavy-tailed fare distribution. Everything else (data size, cleaning, engineered features, hyperparameter loop, chronological split, submission writing) remains the same.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "protobuf>=3.20.3,<5"]
)

import tensorflow as tf

print(tf.__version__)



## === cell 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import (
    RandomForestRegressor,
)  # unused but kept to preserve your original imports

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_iop_path = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
test_iop_path = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"

NROWS_TRAIN = 600_000
NROWS_TEST = 10_000  # test has ~9914 rows; keep as-is

dataset_train = pd.read_csv(train_iop_path, nrows=NROWS_TRAIN, index_col="key")
dataset_test = pd.read_csv(test_iop_path, nrows=NROWS_TEST, index_col="key")



## === cell 2
print("dataset_train old size", len(dataset_train))

dataset_train = dataset_train[dataset_train.dropoff_longitude != 0]

coord_ok = (
    dataset_train["pickup_longitude"].between(-75, -72)
    & dataset_train["dropoff_longitude"].between(-75, -72)
    & dataset_train["pickup_latitude"].between(40, 42)
    & dataset_train["dropoff_latitude"].between(40, 42)
)
nyc_ok = (
    dataset_train["pickup_longitude"].between(-74.3, -73.7)
    & dataset_train["dropoff_longitude"].between(-74.3, -73.7)
    & dataset_train["pickup_latitude"].between(40.5, 40.95)
    & dataset_train["dropoff_latitude"].between(40.5, 40.95)
)

fare_ok = dataset_train["fare_amount"].between(2.5, 250.0)
pax_ok = dataset_train["passenger_count"].between(1, 6)

same_loc = (dataset_train["pickup_longitude"] == dataset_train["dropoff_longitude"]) & (
    dataset_train["pickup_latitude"] == dataset_train["dropoff_latitude"]
)

dataset_train = dataset_train[coord_ok & nyc_ok & fare_ok & pax_ok & (~same_loc)].copy()

dx = dataset_train["dropoff_longitude"] - dataset_train["pickup_longitude"]
dy = dataset_train["dropoff_latitude"] - dataset_train["pickup_latitude"]
deg_dist = np.sqrt(dx * dx + dy * dy)
dataset_train = dataset_train[deg_dist.between(0.0005, 1.0)].copy()

swap_like = (
    dataset_train["pickup_latitude"].between(-75, -72)  # lat looks like a longitude
    | dataset_train["dropoff_latitude"].between(-75, -72)
    | dataset_train["pickup_longitude"].between(40, 42)  # lon looks like a latitude
    | dataset_train["dropoff_longitude"].between(40, 42)
)
dataset_train = dataset_train[~swap_like].copy()

dx = dataset_train["dropoff_longitude"] - dataset_train["pickup_longitude"]
dy = dataset_train["dropoff_latitude"] - dataset_train["pickup_latitude"]
deg_dist = np.sqrt(dx * dx + dy * dy)
fare_per_deg = dataset_train["fare_amount"] / np.maximum(deg_dist, 1e-6)
dataset_train = dataset_train[fare_per_deg.between(5.0, 500.0)].copy()


def haversine_km(lon1, lat1, lon2, lat2):
    r = 6371.0088
    lon1 = np.radians(np.asarray(lon1, dtype="float64"))
    lat1 = np.radians(np.asarray(lat1, dtype="float64"))
    lon2 = np.radians(np.asarray(lon2, dtype="float64"))
    lat2 = np.radians(np.asarray(lat2, dtype="float64"))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(a))


trip_km = haversine_km(
    dataset_train["pickup_longitude"],
    dataset_train["pickup_latitude"],
    dataset_train["dropoff_longitude"],
    dataset_train["dropoff_latitude"],
)

trip_km = pd.Series(trip_km, index=dataset_train.index)

dataset_train = dataset_train[trip_km.between(0.05, 80.0)].copy()
trip_km = trip_km.loc[dataset_train.index]

fare_per_km = dataset_train["fare_amount"] / np.maximum(trip_km, 0.05)
dataset_train = dataset_train[fare_per_km.between(0.5, 50.0)].copy()

print("new size", len(dataset_train))
dataset_train.head(5)



## === cell 3
print("dataset_test old size", len(dataset_test))
print("new size (unchanged to preserve required row count)", len(dataset_test))
dataset_test.head(5)




## === cell 4
def get_year(pickup_date):
    return pickup_date.year


def get_month(pickup_date):
    return pickup_date.month


def get_day(pickup_date):
    return pickup_date.day


def get_hour(pickup_date):
    return pickup_date.hour




## === cell 5
from datetime import datetime as dt
import warnings


def preparedataset2(datasetname):
    warnings.filterwarnings("ignore")

    datasetname = datasetname.copy()

    datasetname["pickup_datetime"] = pd.to_datetime(
        datasetname["pickup_datetime"], errors="coerce", utc=False
    )

    datasetname["pickup_year"] = datasetname["pickup_datetime"].dt.year
    datasetname["pickup_month"] = datasetname["pickup_datetime"].dt.month
    datasetname["pickup_day"] = datasetname["pickup_datetime"].dt.day
    datasetname["pickup_hour"] = datasetname["pickup_datetime"].dt.hour

    datasetname["x_dis"] = (
        datasetname["dropoff_longitude"] - datasetname["pickup_longitude"]
    )
    datasetname["y_dis"] = (
        datasetname["dropoff_latitude"] - datasetname["pickup_latitude"]
    )
    datasetname["dis"] = (datasetname["x_dis"] ** 2 + datasetname["y_dis"] ** 2) ** 0.5

    plon = datasetname["pickup_longitude"].astype("float64")
    plat = datasetname["pickup_latitude"].astype("float64")
    dlon = datasetname["dropoff_longitude"].astype("float64")
    dlat = datasetname["dropoff_latitude"].astype("float64")

    datasetname["haversine_km"] = haversine_km(plon, plat, dlon, dlat)

    JFK = (-73.7781, 40.6413)
    LGA = (-73.8740, 40.7769)
    EWR = (-74.1745, 40.6895)
    MANHATTAN = (-73.9855, 40.7580)

    datasetname["pickup_to_jfk_km"] = haversine_km(plon, plat, JFK[0], JFK[1])
    datasetname["dropoff_to_jfk_km"] = haversine_km(dlon, dlat, JFK[0], JFK[1])

    datasetname["pickup_to_lga_km"] = haversine_km(plon, plat, LGA[0], LGA[1])
    datasetname["dropoff_to_lga_km"] = haversine_km(dlon, dlat, LGA[0], LGA[1])

    datasetname["pickup_to_ewr_km"] = haversine_km(plon, plat, EWR[0], EWR[1])
    datasetname["dropoff_to_ewr_km"] = haversine_km(dlon, dlat, EWR[0], EWR[1])

    datasetname["pickup_to_manhattan_km"] = haversine_km(
        plon, plat, MANHATTAN[0], MANHATTAN[1]
    )
    datasetname["dropoff_to_manhattan_km"] = haversine_km(
        dlon, dlat, MANHATTAN[0], MANHATTAN[1]
    )

    datasetname = datasetname.drop(["pickup_datetime"], axis=1)
    datasetname = datasetname.drop(["pickup_longitude"], axis=1)
    datasetname = datasetname.drop(["dropoff_latitude"], axis=1)
    datasetname = datasetname.drop(["dropoff_longitude"], axis=1)
    datasetname = datasetname.drop(["pickup_latitude"], axis=1)

    return datasetname




## === cell 6
from datetime import datetime as dt
import warnings

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




## === cell 7
print("dataset_train old size", len(dataset_train))

dataset_train = dataset_train[dataset_train.dropoff_longitude != 0]

coord_ok = (
    dataset_train["pickup_longitude"].between(-75, -72)
    & dataset_train["dropoff_longitude"].between(-75, -72)
    & dataset_train["pickup_latitude"].between(40, 42)
    & dataset_train["dropoff_latitude"].between(40, 42)
)
nyc_ok = (
    dataset_train["pickup_longitude"].between(-74.3, -73.7)
    & dataset_train["dropoff_longitude"].between(-74.3, -73.7)
    & dataset_train["pickup_latitude"].between(40.5, 40.95)
    & dataset_train["dropoff_latitude"].between(40.5, 40.95)
)

fare_ok = dataset_train["fare_amount"].between(2.5, 250.0)
pax_ok = dataset_train["passenger_count"].between(1, 6)

same_loc = (dataset_train["pickup_longitude"] == dataset_train["dropoff_longitude"]) & (
    dataset_train["pickup_latitude"] == dataset_train["dropoff_latitude"]
)

dataset_train = dataset_train[coord_ok & nyc_ok & fare_ok & pax_ok & (~same_loc)].copy()

dx = dataset_train["dropoff_longitude"] - dataset_train["pickup_longitude"]
dy = dataset_train["dropoff_latitude"] - dataset_train["pickup_latitude"]
deg_dist = np.sqrt(dx * dx + dy * dy)
dataset_train = dataset_train[deg_dist.between(0.0005, 1.0)].copy()

swap_like = (
    dataset_train["pickup_latitude"].between(-75, -72)  # lat looks like a longitude
    | dataset_train["dropoff_latitude"].between(-75, -72)
    | dataset_train["pickup_longitude"].between(40, 42)  # lon looks like a latitude
    | dataset_train["dropoff_longitude"].between(40, 42)
)
dataset_train = dataset_train[~swap_like].copy()

dx = dataset_train["dropoff_longitude"] - dataset_train["pickup_longitude"]
dy = dataset_train["dropoff_latitude"] - dataset_train["pickup_latitude"]
deg_dist = np.sqrt(dx * dx + dy * dy)
fare_per_deg = dataset_train["fare_amount"] / np.maximum(deg_dist, 1e-6)
dataset_train = dataset_train[fare_per_deg.between(5.0, 500.0)].copy()

trip_km = haversine_km(
    dataset_train["pickup_longitude"],
    dataset_train["pickup_latitude"],
    dataset_train["dropoff_longitude"],
    dataset_train["dropoff_latitude"],
)
trip_km = pd.Series(trip_km, index=dataset_train.index)

dataset_train = dataset_train[trip_km.between(0.05, 80.0)].copy()
trip_km = trip_km.loc[dataset_train.index]

fare_per_km = dataset_train["fare_amount"] / np.maximum(trip_km, 0.05)
dataset_train = dataset_train[fare_per_km.between(0.5, 50.0)].copy()

print("new size", len(dataset_train))
dataset_train.head(5)



## === cell 8
dataset_test = dataset_test.copy()
dataset_test["passenger_count"] = dataset_test["passenger_count"].clip(lower=1, upper=6)

test_df = preparedataset2(dataset_test)

for c in ["pickup_year", "pickup_month", "pickup_day", "pickup_hour"]:
    if c in test_df.columns:
        test_df[c] = test_df[c].fillna(0)

test_df.head(5)



## === cell 9
from sklearn.model_selection import train_test_split

train_times = pd.to_datetime(
    dataset_train["pickup_datetime"], errors="coerce", utc=False
)

df = preparedataset2(dataset_train)

for c in ["pickup_year", "pickup_month", "pickup_day", "pickup_hour"]:
    if c in df.columns:
        df[c] = df[c].fillna(0)

valid_time_mask = train_times.notna()
df = df.loc[valid_time_mask]
train_times = train_times.loc[valid_time_mask]

order = np.argsort(train_times.values)
n = len(df)
cut = int(n * 0.8)

train_idx = df.index[order[:cut]]
valid_idx = df.index[order[cut:]]

y = df.fare_amount
X = df.drop("fare_amount", axis=1)

X_train, y_train = X.loc[train_idx], y.loc[train_idx]
X_valid, y_valid = X.loc[valid_idx], y.loc[valid_idx]



## === cell 10
import warnings

warnings.filterwarnings("ignore")

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

y_train_t = np.log1p(y_train)
y_valid_t = np.log1p(y_valid)

best_istemator = 0
best_learing_rate = 0
best_rmse = float("inf")

for lr in [X / 100 for X in range(10, 50, 5)]:
    for ns in range(200, 701, 50):
        my_model = XGBRegressor(
            n_estimators=ns,
            learning_rate=lr,
            n_jobs=-1,  # use all CPU
            random_state=42,
            objective="reg:squarederror",
            eval_metric="rmse",
        )
        my_model.fit(X_train, y_train_t)

        pred_log = my_model.predict(X_valid)
        pred = np.expm1(pred_log)
        pred = np.clip(pred, 0, None)

        rmse = mean_squared_error(y_valid, pred, squared=False)

        if rmse < best_rmse:
            best_rmse = rmse
            best_istemator = ns
            best_learing_rate = lr
            print("better found")
            print(ns, lr, rmse)

print("best_istemator:", best_istemator)
print("best_learing_rate:", best_learing_rate)
print("Validation RMSE:", best_rmse)



## === cell 11
from xgboost import XGBRegressor

X_full = X
y_full = y

y_full_t = np.log1p(y_full)

my_model_2 = XGBRegressor(
    n_estimators=best_istemator if best_istemator else 700,
    learning_rate=best_learing_rate if best_learing_rate else 0.25,
    n_jobs=-1,
    random_state=42,
    objective="reg:squarederror",
    eval_metric="rmse",
)
my_model_2.fit(X_full, y_full_t)

test_pred_log = my_model_2.predict(test_df)
test_preds = np.expm1(test_pred_log)

test_preds = np.clip(test_preds, 0, None)

output = pd.DataFrame({"key": test_df.index, "fare_amount": test_preds})
output.to_csv("submission.csv", index=False)

print("Wrote submission.csv with rows:", len(output))
print(output.head())
