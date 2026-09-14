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

6.34253

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.09397) has done: 'I remove the TensorFlow installation/import cell that is crashing due to an incompatible protobuf/TensorFlow combination and is not used by the solution. I fix the date parsing/feature engineering so it works with the actual `pickup_datetime` format in this dataset and doesn’t rely on slow/incorrect row-wise `apply`. I also ensure train/test filtering doesn’t drop test rows (which would break submission alignment) and that the submission uses the exact `key` values from the original test file. Finally, I keep the XGBoost model logic intact while switching the validation metric to RMSE (the competition metric) so the chosen hyperparameters move score in the correct direction.'
- What this solution (achieved 6.12084) has done: 'You’re currently training on a random sample of 10k rows without cleaning obvious outliers/invalid coordinates, and you also ignore the hyperparameter search result by hard-coding (700, 0.2) later; both issues typically inflate RMSE for this competition. I keep your exact core approach (same feature engineering + XGBRegressor + simple grid search) but (1) add minimal, standard NYC Taxi cleaning on the *training* subset only (fare/lat/lon/passenger bounds) to reduce noise, and (2) use the best (n_estimators, learning_rate) found in your grid search for the final model. I also ensure feature columns between train and test are aligned identically before fitting/predicting to avoid any silent column-order issues. These small changes should move your score down toward the 4.198 target without changing the overall solution design.'
- What this solution (achieved 6.20459) has done: 'We keep your same feature engineering and XGBRegressor approach, but make two minimal changes that typically move RMSE down for this competition: (1) switch the distance feature from plain Euclidean degrees to a vectorized Haversine distance in kilometers (still “simple distance”, just in the correct geometry), and (2) add a tiny, safe post-processing clip to keep predictions within the same plausible fare range you trained on (prevents a few extreme negatives/huge values from inflating RMSE). Everything else (sampling 10k rows, grid search over `n_estimators`/`learning_rate`, train/valid split, and submission format) stays the same. These changes should improve your score toward the 4.19822 target without altering the overall pipeline design. The script still runs end-to-end and writes `submission.csv` with `key,fare_amount`.'
- What this solution (achieved 5.72773) has done: 'You’re currently training on only 10k rows, which is the main reason your RMSE is far above the 4.198 target; the smallest score-relevant change is to increase the training sample size while keeping the same model/feature logic. To stay within the 600s runtime, I keep your exact XGBRegressor approach and grid-search structure but (a) read a larger random sample from `train.csv` using `skiprows` (so we don’t load 55M rows), and (b) cut the grid slightly (still searching `n_estimators` and `learning_rate`) so total training time stays reasonable. I also set `tree_method='hist'` (same model semantics, much faster) and add XGBoost’s `eval_metric='rmse'` to align training with the competition metric (no architecture change). Everything else—cleaning rules, haversine distance feature, split, clipping, and submission format—remains the same.'
- What this solution (achieved 5.54503) has done: 'I fix the runtime error in `_haversine_km`/`_bearing` by making them robust to scalar lon/lat inputs (NYC center coordinates), which currently crash because scalars don’t have `.astype`. Then the downstream `df/X_train` NameErrors disappear because feature prep complete successfully. I also ensure the train/test feature columns stay aligned and that we always write a valid `submission.csv` with `key,fare_amount` using the original test keys. These are execution-blocking fixes and should be score-neutral aside from enabling the intended feature set to train properly.'
- What this solution (achieved 6.34253) has done: 'I fix the execution-blocking XGBoost API misuse: `XGBRegressor.fit()` in xgboost 2.x requires `(X, y)` and cannot be called with a `QuantileDMatrix` as the only argument. To keep your core logic identical (same features, same XGBRegressor, same grid over `n_estimators`/`learning_rate`, same RMSE selection, same clipping), I pass the NumPy arrays directly to `fit()` and keep the `QuantileDMatrix` only if needed (it isn’t here). I also ensure test feature alignment stays intact and that `submission.csv` is always written with the original test `key` index and required columns.'

# 9. Code solution

## === cell 0
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RANDOM_STATE = 42

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))

train_iop_path = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
test_iop_path = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"

N_TRAIN_SAMPLE = 1_000_000  # keep identical sample size target


def _reservoir_sample_csv(
    path,
    n_rows,
    seed,
    usecols,
    index_col=None,
    chunksize=200_000,
    dtype=None,
):
    """
    Reservoir sampling over CSV chunks.

    Bug fix: the previous implementation assumed the reservoir had size n_rows once created.
    If the first chunk is smaller than n_rows, the reservoir is smaller, but j was still
    compared to n_rows, which can produce indices > len(reservoir)-1 and crash.

    Fix: track current reservoir size (k = len(reservoir)) at all times and use k
    consistently for replacement decisions and index bounds.
    """
    r = np.random.RandomState(seed)
    reservoir = None
    columns = None
    seen_total = 0

    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        chunksize=chunksize,
        dtype=dtype,
    ):
        if columns is None:
            columns = list(chunk.columns)

        arr = chunk.to_numpy(copy=False)
        m = arr.shape[0]
        if m == 0:
            continue

        if reservoir is None:
            take = min(n_rows, m)
            reservoir = arr[:take].copy()
            seen_total = take
            arr = arr[take:]
            m = arr.shape[0]
            if m == 0:
                continue

        k = reservoir.shape[0]
        if k == 0:
            continue

        if k < n_rows:
            need = min(n_rows - k, m)
            if need > 0:
                reservoir = np.vstack([reservoir, arr[:need].copy()])
                seen_total += need
                arr = arr[need:]
                m = arr.shape[0]
                if m == 0:
                    continue
                k = reservoir.shape[0]

        t = seen_total + np.arange(1, m + 1, dtype=np.int64)  # 1-based stream index
        j = (r.random_sample(size=m) * t.astype(np.float64)).astype(
            np.int64
        ) + 1  # 1..t

        replace_mask = j <= k
        if np.any(replace_mask):
            idx = j[replace_mask] - 1  # 0..k-1 (guaranteed in-bounds)
            reservoir[idx] = arr[replace_mask]

        seen_total += m

    if reservoir is None or columns is None:
        raise RuntimeError("Failed to read any data from train.csv")

    reservoir_df = pd.DataFrame(reservoir, columns=columns)

    if dtype is not None:
        for col, dt in dtype.items():
            if col in reservoir_df.columns:
                try:
                    reservoir_df[col] = reservoir_df[col].astype(dt)
                except Exception:
                    pass

    if index_col is not None and index_col in reservoir_df.columns:
        reservoir_df = reservoir_df.set_index(index_col, drop=True)
    return reservoir_df


train_usecols = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
test_usecols = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
train_dtypes = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "float32",
    "pickup_datetime": "string",
    "key": "string",
}
test_dtypes = {
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "float32",
    "pickup_datetime": "string",
    "key": "string",
}

dataset_train = _reservoir_sample_csv(
    train_iop_path,
    n_rows=N_TRAIN_SAMPLE,
    seed=RANDOM_STATE,
    usecols=train_usecols,
    index_col="key",
    chunksize=250_000,
    dtype=train_dtypes,
)

dataset_test = pd.read_csv(
    test_iop_path,
    usecols=test_usecols,
    dtype=test_dtypes,
    index_col="key",
)

print("Loaded train:", dataset_train.shape, "test:", dataset_test.shape)



## === cell 1
print("dataset_train old size", len(dataset_train))

m = dataset_train["dropoff_longitude"].ne(0)
m &= dataset_train["fare_amount"].gt(0) & dataset_train["fare_amount"].le(250)
m &= dataset_train["passenger_count"].ge(1) & dataset_train["passenger_count"].le(6)
m &= (
    dataset_train["pickup_longitude"].between(-75, -72)
    & dataset_train["dropoff_longitude"].between(-75, -72)
    & dataset_train["pickup_latitude"].between(40, 42)
    & dataset_train["dropoff_latitude"].between(40, 42)
)

dataset_train = dataset_train.loc[m]

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


def _to_radians(x):
    arr = np.asarray(x, dtype=np.float64)
    return np.radians(arr)


def _haversine_km(lon1, lat1, lon2, lat2):
    lon1 = _to_radians(lon1)
    lat1 = _to_radians(lat1)
    lon2 = _to_radians(lon2)
    lat2 = _to_radians(lat2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    R = 6371.0
    return R * c


def _bearing(lon1, lat1, lon2, lat2):
    lon1 = _to_radians(lon1)
    lat1 = _to_radians(lat1)
    lon2 = _to_radians(lon2)
    lat2 = _to_radians(lat2)

    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.arctan2(y, x)


def preparedataset2(datasetname: pd.DataFrame) -> pd.DataFrame:
    df = datasetname.copy()

    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True).dt.tz_convert(
        None
    )
    year = dt.dt.year
    month = dt.dt.month
    day = dt.dt.day
    hour = dt.dt.hour

    if year.isna().any():
        year = year.fillna(int(year.mode(dropna=True).iloc[0]))
    if month.isna().any():
        month = month.fillna(int(month.mode(dropna=True).iloc[0]))
    if day.isna().any():
        day = day.fillna(int(day.mode(dropna=True).iloc[0]))
    if hour.isna().any():
        hour = hour.fillna(int(hour.mode(dropna=True).iloc[0]))

    df["pickup_year"] = year.astype(np.int16, copy=False)
    df["pickup_month"] = month.astype(np.int8, copy=False)
    df["pickup_day"] = day.astype(np.int8, copy=False)
    df["pickup_hour"] = hour.astype(np.int8, copy=False)

    df["x_dis"] = (df["dropoff_longitude"] - df["pickup_longitude"]).astype(
        np.float32, copy=False
    )
    df["y_dis"] = (df["dropoff_latitude"] - df["pickup_latitude"]).astype(
        np.float32, copy=False
    )

    df["dis"] = _haversine_km(
        df["pickup_longitude"].to_numpy(),
        df["pickup_latitude"].to_numpy(),
        df["dropoff_longitude"].to_numpy(),
        df["dropoff_latitude"].to_numpy(),
    ).astype(np.float32, copy=False)

    df["bearing"] = _bearing(
        df["pickup_longitude"].to_numpy(),
        df["pickup_latitude"].to_numpy(),
        df["dropoff_longitude"].to_numpy(),
        df["dropoff_latitude"].to_numpy(),
    ).astype(np.float32, copy=False)

    nyc_lon, nyc_lat = -73.985428, 40.748817
    df["pickup_to_center_km"] = _haversine_km(
        df["pickup_longitude"].to_numpy(),
        df["pickup_latitude"].to_numpy(),
        nyc_lon,
        nyc_lat,
    ).astype(np.float32, copy=False)
    df["dropoff_to_center_km"] = _haversine_km(
        df["dropoff_longitude"].to_numpy(),
        df["dropoff_latitude"].to_numpy(),
        nyc_lon,
        nyc_lat,
    ).astype(np.float32, copy=False)

    model_drop_cols = [
        "pickup_datetime",
        "pickup_longitude",
        "dropoff_latitude",
        "dropoff_longitude",
        "pickup_latitude",
    ]
    df = df.drop(columns=model_drop_cols, errors="ignore")

    if "passenger_count" in df.columns:
        pc = pd.to_numeric(df["passenger_count"], errors="coerce")
        df["passenger_count"] = pc.fillna(1).clip(0, 6).astype(np.float32, copy=False)

    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
        df[num_cols] = df[num_cols].fillna(df[num_cols].median(numeric_only=True))

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

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)



## === cell 9
import warnings

warnings.filterwarnings("ignore")

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

result = {}
best_istemator = 0
best_learing_rate = 0
best_rmse = float("inf")

fixed_params = dict(
    n_jobs=4,
    random_state=RANDOM_STATE,
    tree_method="hist",
    eval_metric="rmse",
    subsample=0.8,
    colsample_bytree=0.8,
)

X_train_np = X_train.to_numpy(dtype=np.float32, copy=False)
X_valid_np = X_valid.to_numpy(dtype=np.float32, copy=False)
y_train_np = y_train.to_numpy(dtype=np.float32, copy=False)
y_valid_np = y_valid.to_numpy(dtype=np.float32, copy=False)

for lr in [0.10, 0.15, 0.20, 0.25, 0.30]:
    for ns in [300, 400, 500, 600, 700]:
        my_model = XGBRegressor(n_estimators=ns, learning_rate=lr, **fixed_params)
        my_model.fit(X_train_np, y_train_np)

        predictions = my_model.predict(X_valid_np)
        rmse = mean_squared_error(y_valid_np, predictions, squared=False)

        if rmse < best_rmse:
            best_rmse = rmse
            best_istemator = ns
            best_learing_rate = lr
            print("better found")
            print(ns, lr, rmse)

        result[(ns, lr)] = rmse

my_model_2 = XGBRegressor(
    n_estimators=best_istemator, learning_rate=best_learing_rate, **fixed_params
)
my_model_2.fit(X_train_np, y_train_np)
predictions_2 = my_model_2.predict(X_valid_np)
rmse_2 = mean_squared_error(y_valid_np, predictions_2, squared=False)

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
    subsample=0.8,
    colsample_bytree=0.8,
)

X_np = X.to_numpy(dtype=np.float32, copy=False)
y_np = y.to_numpy(dtype=np.float32, copy=False)

final_model.fit(X_np, y_np)

test_df = test_df.reindex(columns=X.columns)
test_np = test_df.to_numpy(dtype=np.float32, copy=False)

test_preds = final_model.predict(test_np)
test_preds = np.clip(test_preds, 0.0, 250.0)

output = pd.DataFrame({"key": test_df.index, "fare_amount": test_preds})
output.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", output.shape)
print(output.head())
