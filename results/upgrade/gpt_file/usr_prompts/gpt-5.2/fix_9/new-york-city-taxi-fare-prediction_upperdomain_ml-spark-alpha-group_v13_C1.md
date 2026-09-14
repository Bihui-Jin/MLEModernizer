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
import os
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor

print("Listing /kaggle/input:")
print(os.listdir("/kaggle/input"))

TRAIN_PATH = r"/kaggle/input/train.csv"
TEST_PATH = r"/kaggle/input/test.csv"

TRAIN_USECOLS = [
    "fare_amount",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
TEST_USECOLS = [
    "key",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

DTYPES_TRAIN = {
    "fare_amount": "float64",  # keep exact as used later
    "pickup_longitude": "float64",
    "pickup_latitude": "float64",
    "dropoff_longitude": "float64",
    "dropoff_latitude": "float64",
    "passenger_count": "int16",
}
DTYPES_TEST = {
    "key": "object",
    "pickup_longitude": "float64",
    "pickup_latitude": "float64",
    "dropoff_longitude": "float64",
    "dropoff_latitude": "float64",
    "passenger_count": "int16",
}


def chunck_generator(filename, chunk_size=10**6):
    if "train" in os.path.basename(filename):
        usecols = TRAIN_USECOLS
        dtypes = DTYPES_TRAIN
    else:
        usecols = TEST_USECOLS
        dtypes = DTYPES_TEST

    for chunk in pd.read_csv(
        filename,
        delimiter=",",
        iterator=True,
        chunksize=chunk_size,
        usecols=usecols,
        dtype=dtypes,
        engine="c",
        low_memory=False,
        memory_map=True,
    ):
        yield chunk




## === cell 1
alpha_ang = 0.506


def distance_travel_np(plon, plat, dlon, dlat):
    abs_diff_longitude = np.abs(dlon - plon) * 50.0
    abs_diff_latitude = np.abs(dlat - plat) * 69.0
    displacement_vector = np.sqrt(
        abs_diff_latitude * abs_diff_latitude + abs_diff_longitude * abs_diff_longitude
    )

    denom = np.maximum(abs_diff_latitude, 1e-12)
    ang = np.arctan(abs_diff_longitude / denom) - alpha_ang

    actual_long = np.abs(displacement_vector * np.sin(ang))
    actual_lat = np.abs(displacement_vector * np.cos(ang))
    dist = actual_long + actual_lat
    return (
        abs_diff_longitude,
        abs_diff_latitude,
        displacement_vector,
        actual_long,
        actual_lat,
        dist,
    )




## === cell 2
def clean_and_featurize_train_chunk(df):
    fare = df["fare_amount"].to_numpy(dtype=np.float64, copy=False)
    pcount = df["passenger_count"].to_numpy(dtype=np.int16, copy=False)

    plon = df["pickup_longitude"].to_numpy(dtype=np.float64, copy=False)
    plat = df["pickup_latitude"].to_numpy(dtype=np.float64, copy=False)
    dlon = df["dropoff_longitude"].to_numpy(dtype=np.float64, copy=False)
    dlat = df["dropoff_latitude"].to_numpy(dtype=np.float64, copy=False)

    (_, _, _, _, _, dist) = distance_travel_np(plon, plat, dlon, dlat)

    mask = (pcount > 0) & (fare > 0) & (dist > 0) & (dist < 30) & (fare < 100)

    if not np.any(mask):
        return None

    fare = fare[mask]
    pcount = pcount[mask].astype(np.float64, copy=False)
    dist = dist[mask]

    n = fare.shape[0]
    X = np.empty((n, 3), dtype=np.float64)
    X[:, 0] = dist
    X[:, 1] = pcount
    X[:, 2] = 1.0
    y = fare
    return X, y




## === cell 3
def incremental_training(train_X, train_y, regr):
    regr.fit(train_X, train_y)
    return regr




## === cell 4

filename = TRAIN_PATH
CHUNK_SIZE = 10**6
gen = chunck_generator(filename=filename, chunk_size=CHUNK_SIZE)

regr = GradientBoostingRegressor(n_estimators=100, warm_start=True, random_state=42)

imp = SimpleImputer(missing_values=np.nan, strategy="mean")
imp_fitted = False

t = 56
TREES_PER_CHUNK = 10

rng = np.random.RandomState(42)

RESERVOIR_MAX = 2_000_000  # bounded memory/time; keeps accumulated-training behavior
X_acc = np.empty((RESERVOIR_MAX, 3), dtype=np.float64)
y_acc = np.empty((RESERVOIR_MAX,), dtype=np.float64)
acc_n = 0
seen_n = 0  # total seen (for reservoir sampling)

fit_idx_buf = np.empty((RESERVOIR_MAX,), dtype=np.int64)


def _reservoir_update_vectorized(
    X_new, y_new, X_acc, y_acc, acc_n, seen_n, rng, RESERVOIR_MAX
):
    """Vectorized reservoir update equivalent to the original per-row loop."""
    n_new = X_new.shape[0]
    if n_new == 0:
        return acc_n, seen_n

    if acc_n < RESERVOIR_MAX:
        take = min(RESERVOIR_MAX - acc_n, n_new)
        if take > 0:
            X_acc[acc_n : acc_n + take] = X_new[:take]
            y_acc[acc_n : acc_n + take] = y_new[:take]
            acc_n += take
            seen_n += take
            X_new = X_new[take:]
            y_new = y_new[take:]
            n_new -= take
            if n_new == 0:
                return acc_n, seen_n

    k_start = seen_n
    k = np.arange(k_start, k_start + n_new, dtype=np.int64)
    high = k + 1  # inclusive upper bound in original randint -> exclusive high here
    j = (rng.random_sample(n_new) * high).astype(
        np.int64
    )  # deterministic given RNG state
    mask = j < RESERVOIR_MAX
    if np.any(mask):
        jj = j[mask]
        X_acc[jj] = X_new[mask]
        y_acc[jj] = y_new[mask]

    seen_n += n_new
    return acc_n, seen_n


def _partial_shuffle_inplace(idx, n, rng, swaps=262144):
    """In-place partial Fisher-Yates shuffle on idx[:n]. Equivalent semantics: random training order."""
    if n <= 1:
        return
    s = swaps if swaps < n else n
    for i in range(s):
        j = rng.randint(i, n)
        idx[i], idx[j] = idx[j], idx[i]


while t > 0:
    print("chunk remaining:", t)
    df = next(gen)

    out = clean_and_featurize_train_chunk(df)
    if out is None:
        t -= 1
        continue
    X_all, y_all = out
    l = X_all.shape[0]
    if l < 10:
        t -= 1
        continue

    split = int(0.9 * l)
    train_X = X_all[:split]
    train_y = y_all[:split]
    test_X = X_all[split:]
    test_y = y_all[split:]

    if not imp_fitted:
        imp = imp.fit(train_X)
        imp_fitted = True

    train_X = imp.transform(train_X)
    test_X = imp.transform(test_X)

    acc_n, seen_n = _reservoir_update_vectorized(
        train_X, train_y, X_acc, y_acc, acc_n, seen_n, rng, RESERVOIR_MAX
    )

    fit_idx_buf[:acc_n] = np.arange(acc_n, dtype=np.int64)
    _partial_shuffle_inplace(fit_idx_buf, acc_n, rng, swaps=262144)

    regr.set_params(n_estimators=regr.n_estimators + TREES_PER_CHUNK)
    regr = incremental_training(
        X_acc[fit_idx_buf[:acc_n], :], y_acc[fit_idx_buf[:acc_n]], regr
    )

    print("R^2 on held-out chunk split:", regr.score(test_X, test_y))
    t -= 1



## === cell 5
test_df = pd.read_csv(
    TEST_PATH,
    usecols=TEST_USECOLS,
    dtype=DTYPES_TEST,
    engine="c",
    low_memory=False,
    memory_map=True,
)

plon = test_df["pickup_longitude"].to_numpy(dtype=np.float64, copy=False)
plat = test_df["pickup_latitude"].to_numpy(dtype=np.float64, copy=False)
dlon = test_df["dropoff_longitude"].to_numpy(dtype=np.float64, copy=False)
dlat = test_df["dropoff_latitude"].to_numpy(dtype=np.float64, copy=False)
(_, _, _, _, _, dist) = distance_travel_np(plon, plat, dlon, dlat)

test_X = np.empty((len(test_df), 3), dtype=np.float64)
test_X[:, 0] = dist
test_X[:, 1] = test_df["passenger_count"].to_numpy(dtype=np.float64, copy=False)
test_X[:, 2] = 1.0

test_X = imp.transform(test_X)
predicted_fare = regr.predict(test_X)

predicted_fare = np.maximum(predicted_fare, 0)

print(predicted_fare[:10])

my_submission = pd.DataFrame({"key": test_df.key, "fare_amount": predicted_fare})
my_submission.to_csv("submission.csv", index=False)
print(my_submission.head())
print("Wrote submission.csv with shape:", my_submission.shape)
