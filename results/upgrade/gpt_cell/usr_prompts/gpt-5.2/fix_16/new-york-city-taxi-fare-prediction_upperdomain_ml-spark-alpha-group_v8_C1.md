# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
from sklearn.impute import SimpleImputer as Imputer
import numpy as np
import pandas as pd
import os
import math

os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 1))
os.environ.setdefault("MKL_NUM_THREADS", str(os.cpu_count() or 1))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(os.cpu_count() or 1))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(os.cpu_count() or 1))

print(os.listdir("../input"))

np.random.seed(42)



## === cell 1
TRAIN_PATH = "../input/train.csv"
TEST_PATH = "../input/test.csv"

NROWS = 5_000_000

train_usecols = [
    "fare_amount",
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
    "passenger_count": "uint8",
}



## === cell 2
alpha_ang = 0.506


def distance_travel(df):
    lat1 = df["pickup_latitude"].to_numpy(dtype=np.float64, copy=False)
    lon1 = df["pickup_longitude"].to_numpy(dtype=np.float64, copy=False)
    lat2 = df["dropoff_latitude"].to_numpy(dtype=np.float64, copy=False)
    lon2 = df["dropoff_longitude"].to_numpy(dtype=np.float64, copy=False)

    dlat = lat2 - lat1
    dlon = (lon2 - lon1) * np.cos(alpha_ang * (lat2 + lat1))
    df["distance_travel"] = np.sqrt(dlat * dlat + dlon * dlon).astype(np.float32)


def filter_nyc_coords(frame):
    frame = frame[
        (frame.pickup_longitude.between(-74.3, -73.6))
        & (frame.dropoff_longitude.between(-74.3, -73.6))
        & (frame.pickup_latitude.between(40.4, 41.0))
        & (frame.dropoff_latitude.between(40.4, 41.0))
    ]
    frame = frame[
        (frame.pickup_longitude != 0)
        & (frame.dropoff_longitude != 0)
        & (frame.pickup_latitude != 0)
        & (frame.dropoff_latitude != 0)
    ]
    return frame


def valid_nyc_mask(frame):
    m = (
        frame.pickup_longitude.between(-74.3, -73.6)
        & frame.dropoff_longitude.between(-74.3, -73.6)
        & frame.pickup_latitude.between(40.4, 41.0)
        & frame.dropoff_latitude.between(40.4, 41.0)
        & (frame.pickup_longitude != 0)
        & (frame.dropoff_longitude != 0)
        & (frame.pickup_latitude != 0)
        & (frame.dropoff_latitude != 0)
    )
    return m


def add_time_features(frame):
    dt = frame["pickup_datetime"]
    if not pd.api.types.is_datetime64_any_dtype(dt):
        dt = pd.to_datetime(dt, errors="coerce")
    frame["pickup_hour"] = dt.dt.hour.astype(np.float32)
    frame["pickup_weekday"] = dt.dt.weekday.astype(np.float32)
    frame["pickup_month"] = dt.dt.month.astype(np.float32)
    return frame


def add_manhattan(frame):
    pl = frame["pickup_latitude"].to_numpy(dtype=np.float64, copy=False)
    dl = frame["dropoff_latitude"].to_numpy(dtype=np.float64, copy=False)
    plo = frame["pickup_longitude"].to_numpy(dtype=np.float64, copy=False)
    dlo = frame["dropoff_longitude"].to_numpy(dtype=np.float64, copy=False)

    frame["manhattan"] = (np.abs(pl - dl) + np.abs(plo - dlo)).astype(np.float32)
    return frame


def _chunk_valid_mask_np(chunk):
    pc = chunk["passenger_count"].to_numpy(copy=False)
    fare = chunk["fare_amount"].to_numpy(copy=False)
    plon = chunk["pickup_longitude"].to_numpy(copy=False)
    dlon = chunk["dropoff_longitude"].to_numpy(copy=False)
    plat = chunk["pickup_latitude"].to_numpy(copy=False)
    dlat = chunk["dropoff_latitude"].to_numpy(copy=False)

    m = (pc > 0) & (fare > 0)
    m &= (plon >= -74.3) & (plon <= -73.6)
    m &= (dlon >= -74.3) & (dlon <= -73.6)
    m &= (plat >= 40.4) & (plat <= 41.0)
    m &= (dlat >= 40.4) & (dlat <= 41.0)
    m &= (plon != 0) & (dlon != 0) & (plat != 0) & (dlat != 0)
    return m


def load_and_prepare_train(train_path, nrows_valid, chunksize=1_000_000):
    kept = []
    kept_rows = 0

    reader = pd.read_csv(
        train_path,
        usecols=train_usecols,
        dtype=train_dtypes,
        parse_dates=["pickup_datetime"],
        chunksize=chunksize,
    )

    for chunk in reader:
        m = _chunk_valid_mask_np(chunk)
        if not np.any(m):
            continue

        chunk = chunk.loc[m]

        chunk = add_time_features(chunk)
        chunk = add_manhattan(chunk)
        distance_travel(chunk)

        dtv = chunk["distance_travel"].to_numpy(copy=False)
        fare = chunk["fare_amount"].to_numpy(copy=False)
        m2 = (dtv > 0) & (dtv < 30) & (fare < 100)

        if not np.any(m2):
            continue

        chunk = chunk.loc[m2]

        need = nrows_valid - kept_rows
        if len(chunk) > need:
            chunk = chunk.iloc[:need].copy(deep=False)

        kept.append(chunk)
        kept_rows += len(chunk)

        if kept_rows >= nrows_valid:
            break

    if not kept:
        return pd.DataFrame(
            columns=train_usecols
            + [
                "pickup_hour",
                "pickup_weekday",
                "pickup_month",
                "manhattan",
                "distance_travel",
            ]
        )

    df_out = pd.concat(kept, axis=0, ignore_index=True, copy=False)
    return df_out


df = load_and_prepare_train(TRAIN_PATH, NROWS, chunksize=1_000_000)
df.head()



## === cell 3
pass



## === cell 4
pass



## === cell 5
l = len(df)
print(l)

rng = np.random.RandomState(42)
perm = rng.permutation(l)
cut = int(0.7 * l)

train_mask = np.zeros(l, dtype=bool)
train_mask[perm[:cut]] = True
test_mask = ~train_mask

df_train = df.loc[train_mask]
df_test = df.loc[test_mask]


def _design_matrix(frame):
    cols = (
        "distance_travel",
        "manhattan",
        "passenger_count",
        "pickup_hour",
        "pickup_weekday",
        "pickup_month",
    )
    X0 = frame.loc[:, cols].to_numpy(dtype=np.float32, copy=False)
    X = np.empty((X0.shape[0], X0.shape[1] + 1), dtype=np.float32)
    X[:, :-1] = X0
    X[:, -1] = 1.0
    return X


train_X = _design_matrix(df_train)
test_X = _design_matrix(df_test)

train_y = df_train.fare_amount.to_numpy(dtype=np.float32, copy=False)
test_y = df_test.fare_amount.to_numpy(dtype=np.float32, copy=False)
train_y_log = np.log1p(train_y)
test_y_log = np.log1p(test_y)



## === cell 6
try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

imp = Imputer(missing_values=np.nan, strategy="mean")
imp = imp.fit(train_X)
train_X_imp = imp.transform(train_X)
test_X_imp = imp.transform(test_X)

regr = GradientBoostingRegressor(
    n_estimators=600,
    learning_rate=0.03,
    max_depth=3,
    random_state=42,
)
regr.fit(train_X_imp, train_y_log)

val_pred = np.expm1(regr.predict(test_X_imp))
rmse = float(np.sqrt(mean_squared_error(test_y, val_pred)))
print("Validation RMSE (approx):", rmse)



## === cell 7
test_usecols = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
test_dtypes = {
    "key": "object",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}

tdf = pd.read_csv(
    TEST_PATH,
    usecols=test_usecols,
    dtype=test_dtypes,
    parse_dates=["pickup_datetime"],
)

valid_mask = valid_nyc_mask(tdf)

tdf_work = tdf.copy(deep=False)
coord_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]
tdf_work.loc[~valid_mask, coord_cols] = np.nan

tdf_work = add_time_features(tdf_work)
tdf_work = add_manhattan(tdf_work)
distance_travel(tdf_work)

tdf_work.loc[~valid_mask, ["distance_travel", "manhattan"]] = np.nan

tdf_work.head()



## === cell 8
ttrain_X = _design_matrix(tdf_work)
ttrain_X = imp.transform(ttrain_X)

output_log = regr.predict(ttrain_X)
output = np.expm1(output_log)

lo = float(np.percentile(train_y, 0.5))
hi = float(np.percentile(train_y, 99.5))
output = np.clip(output, lo, hi)

print(output)



## === cell 9
my_submission = pd.DataFrame({"key": tdf.key, "fare_amount": output})

if len(my_submission) != 9914:
    all_test = pd.read_csv(TEST_PATH, usecols=["key"])
    my_submission = all_test.merge(my_submission, on="key", how="left")
    my_submission["fare_amount"] = my_submission["fare_amount"].fillna(
        np.median(train_y)
    )

my_submission.to_csv("submission.csv", index=False)
my_submission.head()
