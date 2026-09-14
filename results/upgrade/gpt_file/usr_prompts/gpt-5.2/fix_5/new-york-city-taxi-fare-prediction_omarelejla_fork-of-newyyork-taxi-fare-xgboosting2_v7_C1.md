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

4.19822

# 6. Current score

5.72773

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.09397) has done: 'I remove the TensorFlow installation/import cell that is crashing due to an incompatible protobuf/TensorFlow combination and is not used by the solution. I fix the date parsing/feature engineering so it works with the actual `pickup_datetime` format in this dataset and doesn’t rely on slow/incorrect row-wise `apply`. I also ensure train/test filtering doesn’t drop test rows (which would break submission alignment) and that the submission uses the exact `key` values from the original test file. Finally, I keep the XGBoost model logic intact while switching the validation metric to RMSE (the competition metric) so the chosen hyperparameters move score in the correct direction.'
- What this solution (achieved 6.12084) has done: 'You’re currently training on a random sample of 10k rows without cleaning obvious outliers/invalid coordinates, and you also ignore the hyperparameter search result by hard-coding (700, 0.2) later; both issues typically inflate RMSE for this competition. I keep your exact core approach (same feature engineering + XGBRegressor + simple grid search) but (1) add minimal, standard NYC Taxi cleaning on the *training* subset only (fare/lat/lon/passenger bounds) to reduce noise, and (2) use the best (n_estimators, learning_rate) found in your grid search for the final model. I also ensure feature columns between train and test are aligned identically before fitting/predicting to avoid any silent column-order issues. These small changes should move your score down toward the 4.198 target without changing the overall solution design.'
- What this solution (achieved 6.20459) has done: 'We keep your same feature engineering and XGBRegressor approach, but make two minimal changes that typically move RMSE down for this competition: (1) switch the distance feature from plain Euclidean degrees to a vectorized Haversine distance in kilometers (still “simple distance”, just in the correct geometry), and (2) add a tiny, safe post-processing clip to keep predictions within the same plausible fare range you trained on (prevents a few extreme negatives/huge values from inflating RMSE). Everything else (sampling 10k rows, grid search over `n_estimators`/`learning_rate`, train/valid split, and submission format) stays the same. These changes should improve your score toward the 4.19822 target without altering the overall pipeline design. The script still runs end-to-end and writes `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 5.72773) has done: 'You’re currently training on only 10k rows, which is the main reason your RMSE is far above the 4.198 target; the smallest score-relevant change is to increase the training sample size while keeping the same model/feature logic. To stay within the 600s runtime, I keep your exact XGBRegressor approach and grid-search structure but (a) read a larger random sample from `train.csv` using `skiprows` (so we don’t load 55M rows), and (b) cut the grid slightly (still searching `n_estimators` and `learning_rate`) so total training time stays reasonable. I also set `tree_method='hist'` (same model semantics, much faster) and add XGBoost’s `eval_metric='rmse'` to align training with the competition metric (no architecture change). Everything else—cleaning rules, haversine distance feature, split, clipping, and submission format—remains the same.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
from pathlib import Path

RANDOM_STATE = 42

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))

train_iop_path = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
test_iop_path = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"

N_TRAIN_SAMPLE = (
    200_000  # kept moderate to fit within time; much better than 10k for RMSE
)
TRAIN_TOTAL_ROWS = 55_423_857  # approx total rows in train.csv including header (from dataset description)

rng = np.random.RandomState(RANDOM_STATE)
skip_n = max(0, (TRAIN_TOTAL_ROWS - 1) - N_TRAIN_SAMPLE)
skip_idx = (
    set(rng.choice(np.arange(1, TRAIN_TOTAL_ROWS), size=skip_n, replace=False))
    if skip_n > 0
    else None
)

dataset_train = pd.read_csv(train_iop_path, skiprows=skip_idx, index_col="key")
dataset_test = pd.read_csv(test_iop_path, index_col="key")

print("Loaded train:", dataset_train.shape, "test:", dataset_test.shape)



## === cell 1
print("dataset_train old size", len(dataset_train))

dataset_train = dataset_train[dataset_train.dropoff_longitude != 0]

dataset_train = dataset_train[
    (dataset_train["fare_amount"] > 0) & (dataset_train["fare_amount"] <= 250)
]
dataset_train = dataset_train[
    (dataset_train["passenger_count"] >= 1) & (dataset_train["passenger_count"] <= 6)
]

dataset_train = dataset_train[
    (dataset_train["pickup_longitude"].between(-75, -72))
    & (dataset_train["dropoff_longitude"].between(-75, -72))
    & (dataset_train["pickup_latitude"].between(40, 42))
    & (dataset_train["dropoff_latitude"].between(40, 42))
]

print("new size", len(dataset_train))
dataset_train.head(5)



## === cell 2
print("dataset_test size", len(dataset_test))
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

warnings.filterwarnings("ignore")


def _haversine_km(lon1, lat1, lon2, lat2):
    """
    Change is directly score-relevant: RMSE improves when distance is computed on a sphere
    rather than Euclidean degrees, while preserving the same "distance feature" intent.
    Vectorized and fast.
    """
    lon1 = np.radians(lon1.astype(float))
    lat1 = np.radians(lat1.astype(float))
    lon2 = np.radians(lon2.astype(float))
    lat2 = np.radians(lat2.astype(float))

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    R = 6371.0  # Earth radius in km
    return R * c


def preparedataset2(datasetname: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized, robust feature engineering for NYC taxi fare data.
    Keeps core features (time parts + simple distances) identical in meaning.
    """
    df = datasetname.copy()

    dt_series = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)
    dt_series = dt_series.dt.tz_convert(None)

    df["pickup_year"] = dt_series.dt.year.astype("Int64")
    df["pickup_month"] = dt_series.dt.month.astype("Int64")
    df["pickup_day"] = dt_series.dt.day.astype("Int64")
    df["pickup_hour"] = dt_series.dt.hour.astype("Int64")

    df["x_dis"] = df["dropoff_longitude"] - df["pickup_longitude"]
    df["y_dis"] = df["dropoff_latitude"] - df["pickup_latitude"]

    df["dis"] = _haversine_km(
        df["pickup_longitude"],
        df["pickup_latitude"],
        df["dropoff_longitude"],
        df["dropoff_latitude"],
    )

    model_drop_cols = [
        "pickup_datetime",
        "pickup_longitude",
        "dropoff_latitude",
        "dropoff_longitude",
        "pickup_latitude",
    ]
    df = df.drop(columns=model_drop_cols, errors="ignore")

    for c in ["pickup_year", "pickup_month", "pickup_day", "pickup_hour"]:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].mode(dropna=True).iloc[0]).astype(int)

    return df




## === cell 5
import warnings

warnings.filterwarnings("ignore")


def preparedataset(datasetname):
    return preparedataset2(datasetname)




## === cell 6
df = preparedataset2(dataset_train)
df.head(5)



## === cell 7
test_df = preparedataset2(dataset_test)
test_df.head(5)



## === cell 8
from sklearn.model_selection import train_test_split

y = df["fare_amount"]
X = df.drop("fare_amount", axis=1)

test_df = test_df.reindex(columns=X.columns)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=RANDOM_STATE)



## === cell 9
import warnings

warnings.filterwarnings("ignore")

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

result = {}
best_istemator = 0
best_learing_rate = 0
best_rmse = float("inf")

for lr in [0.10, 0.15, 0.20, 0.25, 0.30]:
    for ns in [300, 400, 500, 600, 700]:
        my_model = XGBRegressor(
            n_estimators=ns,
            learning_rate=lr,
            n_jobs=4,
            random_state=RANDOM_STATE,
            tree_method="hist",
            eval_metric="rmse",
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
    n_jobs=4,
    random_state=RANDOM_STATE,
    tree_method="hist",
    eval_metric="rmse",
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
    n_jobs=4,
    random_state=RANDOM_STATE,
    tree_method="hist",
    eval_metric="rmse",
)
final_model.fit(X, y)

test_df = test_df.reindex(columns=X.columns)
test_preds = final_model.predict(test_df)

test_preds = np.clip(test_preds, 0.0, 250.0)

output = pd.DataFrame({"key": test_df.index, "fare_amount": test_preds})
output.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", output.shape)
print(output.head())
