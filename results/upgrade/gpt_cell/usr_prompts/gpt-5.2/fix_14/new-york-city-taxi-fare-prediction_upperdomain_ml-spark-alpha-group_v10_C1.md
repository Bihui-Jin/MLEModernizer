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
from sklearnex import patch_sklearn

patch_sklearn()

from sklearn.impute import SimpleImputer as Imputer
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import os

print(os.listdir("../input"))

_READ_DTYPES = {
    "key": "object",
    "fare_amount": "float64",
    "pickup_longitude": "float64",
    "pickup_latitude": "float64",
    "dropoff_longitude": "float64",
    "dropoff_latitude": "float64",
    "passenger_count": "int16",
}


def chunck_generator(filename, header=False, chunk_size=10**5):
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
    for chunk in pd.read_csv(
        filename,
        delimiter=",",
        iterator=True,
        chunksize=chunk_size,
        parse_dates=["pickup_datetime"],
        usecols=usecols,
        dtype=_READ_DTYPES,
        low_memory=False,
    ):
        yield chunk




## === cell 1
alpha_ang = 0.506


def distance_travel(df):
    plon = df["pickup_longitude"].to_numpy(dtype=np.float64, copy=False)
    plat = df["pickup_latitude"].to_numpy(dtype=np.float64, copy=False)
    dlon = df["dropoff_longitude"].to_numpy(dtype=np.float64, copy=False)
    dlat = df["dropoff_latitude"].to_numpy(dtype=np.float64, copy=False)

    abs_diff_longitude = np.abs(dlon - plon) * 50.0
    abs_diff_latitude = np.abs(dlat - plat) * 69.0
    disp = np.sqrt(
        abs_diff_latitude * abs_diff_latitude + abs_diff_longitude * abs_diff_longitude
    )

    ang = np.arctan2(abs_diff_longitude, abs_diff_latitude)
    actual_long = np.abs(disp * np.sin(ang - alpha_ang))
    actual_lat = np.abs(disp * np.cos(ang - alpha_ang))

    df["abs_diff_longitude"] = abs_diff_longitude
    df["abs_diff_latitude"] = abs_diff_latitude
    df["displacement_vector"] = disp
    df["actual_long"] = actual_long
    df["actual_lat"] = actual_lat
    df["distance_travel"] = actual_long + actual_lat
    return df




## === cell 2
def add_time_features(df):
    dt = df["pickup_datetime"].dt
    df["pickup_hour"] = dt.hour.to_numpy(dtype=np.float64, copy=False)
    df["pickup_weekday"] = dt.weekday.to_numpy(dtype=np.float64, copy=False)
    df["pickup_month"] = dt.month.to_numpy(dtype=np.float64, copy=False)
    return df


def make_features(df):
    distance_travel(df)
    add_time_features(df)
    return df




## === cell 3
def data_clean(df):
    df = df.loc[df["passenger_count"] > 0]
    df = df.loc[df["fare_amount"] > 0]
    df = df.loc[df["distance_travel"] > 0]
    df.loc[:, "fare_amount"] = df["fare_amount"].astype(np.float64, copy=False)
    return df




## === cell 4
def remove_outliers(df):
    df = df[df.distance_travel < 30]
    df = df[df.fare_amount < 100]
    return df




## === cell 5
def graph_presesnt(df):
    test = df[df.passenger_count == 1]
    plot = test.iloc[: len(test)].plot.scatter("distance_travel", "fare_amount")
    plot = df.iloc[:100000].plot.scatter("distance_travel", "fare_amount")




## === cell 6
def incremental_training(train_X, train_y, regr, is_first_fit=False):
    regr.fit(train_X, train_y)
    return regr




## === cell 7
filename = r"../input/train.csv"
gen = chunck_generator(filename=filename)

regr = GradientBoostingRegressor(n_estimators=20, warm_start=True, random_state=42)
imp = Imputer(missing_values=np.nan, strategy="mean")
rng = np.random.RandomState(42)


def to_X(df):
    n = len(df)
    X = np.empty((n, 6), dtype=np.float64)
    X[:, 0] = df.distance_travel.to_numpy(dtype=np.float64, copy=False)
    X[:, 1] = df.passenger_count.to_numpy(dtype=np.float64, copy=False)
    X[:, 2] = df.pickup_hour.to_numpy(dtype=np.float64, copy=False)
    X[:, 3] = df.pickup_weekday.to_numpy(dtype=np.float64, copy=False)
    X[:, 4] = df.pickup_month.to_numpy(dtype=np.float64, copy=False)
    X[:, 5] = 1.0
    return X


imp_fit_rows_target = 300_000
imp_fit_X = np.empty((imp_fit_rows_target, 6), dtype=np.float64)
imp_fit_rows = 0

while imp_fit_rows < imp_fit_rows_target:
    df0 = next(gen)
    make_features(df0)
    df0 = data_clean(df0)
    df0 = remove_outliers(df0)
    if len(df0) == 0:
        continue
    take = min(len(df0), imp_fit_rows_target - imp_fit_rows)
    X0 = to_X(df0.iloc[:take])
    imp_fit_X[imp_fit_rows : imp_fit_rows + take, :] = X0
    imp_fit_rows += take

imp.fit(imp_fit_X)

buffer_rows_max = 600_000

X_ring = None
y_ring = None
ring_size = 0
ring_ptr = 0

_X_train_scratch = None
_y_train_scratch = None


def ring_append_imputed(X_new_imp, y_new):
    global X_ring, y_ring, ring_size, ring_ptr
    n_new = len(y_new)
    if n_new == 0:
        return

    if X_ring is None:
        X_ring = np.empty((buffer_rows_max, X_new_imp.shape[1]), dtype=np.float64)
        y_ring = np.empty((buffer_rows_max,), dtype=np.float64)

    if n_new >= buffer_rows_max:
        X_new_imp = X_new_imp[-buffer_rows_max:]
        y_new = y_new[-buffer_rows_max:]
        n_new = buffer_rows_max

    end = ring_ptr + n_new
    if end <= buffer_rows_max:
        X_ring[ring_ptr:end, :] = X_new_imp
        y_ring[ring_ptr:end] = y_new
    else:
        first = buffer_rows_max - ring_ptr
        X_ring[ring_ptr:, :] = X_new_imp[:first]
        y_ring[ring_ptr:] = y_new[:first]
        rem = n_new - first
        X_ring[:rem, :] = X_new_imp[first:]
        y_ring[:rem] = y_new[first:]

    ring_ptr = (ring_ptr + n_new) % buffer_rows_max
    ring_size = min(buffer_rows_max, ring_size + n_new)


def ring_view_slices():
    if ring_size == 0:
        return None, None
    if ring_size < buffer_rows_max:
        return "contig", (X_ring[:ring_size], y_ring[:ring_size])
    if ring_ptr == 0:
        return "contig", (X_ring, y_ring)
    return "split", (
        (X_ring[ring_ptr:], X_ring[:ring_ptr]),
        (y_ring[ring_ptr:], y_ring[:ring_ptr]),
    )


def ring_to_train_arrays(kind, data):
    global _X_train_scratch, _y_train_scratch
    if kind == "contig":
        Xb, yb = data
        return Xb, yb

    (X1, X2), (y1, y2) = data
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    n = n1 + n2
    d = X1.shape[1]

    if (
        (_X_train_scratch is None)
        or (_X_train_scratch.shape[0] != n)
        or (_X_train_scratch.shape[1] != d)
    ):
        _X_train_scratch = np.empty((n, d), dtype=np.float64)
    if (_y_train_scratch is None) or (_y_train_scratch.shape[0] != n):
        _y_train_scratch = np.empty((n,), dtype=np.float64)

    _X_train_scratch[:n1, :] = X1
    _X_train_scratch[n1:, :] = X2
    _y_train_scratch[:n1] = y1
    _y_train_scratch[n1:] = y2
    return _X_train_scratch, _y_train_scratch


n_chunks = 80
trees_per_chunk = 10

is_first_fit = True
t = n_chunks
while t > 0:
    df = next(gen)
    make_features(df)
    df = data_clean(df)
    df = remove_outliers(df)

    if len(df) == 0:
        t -= 1
        continue

    X_chunk = to_X(df)
    y_chunk = df.fare_amount.to_numpy(dtype=np.float64, copy=False)

    X_chunk_imp = imp.transform(X_chunk)
    ring_append_imputed(X_chunk_imp, y_chunk)

    kind, data = ring_view_slices()
    X_train, y_train = ring_to_train_arrays(kind, data)

    if is_first_fit:
        regr = incremental_training(X_train, y_train, regr, is_first_fit=True)
        is_first_fit = False
    else:
        regr.set_params(n_estimators=regr.n_estimators + trees_per_chunk)
        regr = incremental_training(X_train, y_train, regr, is_first_fit=False)

    t -= 1



## === cell 8
tdf = pd.read_csv(
    "../input/test.csv",
    parse_dates=["pickup_datetime"],
    dtype={
        "key": "object",
        "pickup_longitude": "float64",
        "pickup_latitude": "float64",
        "dropoff_longitude": "float64",
        "dropoff_latitude": "float64",
        "passenger_count": "int16",
    },
    low_memory=False,
)
make_features(tdf)
tdf.head()



## === cell 9
ttrain_X = to_X(tdf)
ttrain_X = imp.transform(ttrain_X)
output = regr.predict(ttrain_X)

output = np.maximum(output, 0.0)

print(output)



## === cell 10
my_submission = pd.DataFrame({"key": tdf.key, "fare_amount": output})
my_submission.to_csv("submission.csv", index=False)
my_submission.head()
