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

# 5. Target score

4.22828

# 6. Current score

5.38062

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.38888) has done: 'The timeout is dominated by repeatedly fitting `GradientBoostingRegressor` on million-row chunks (and re-fitting the imputer each time), plus extra work from computing `distance_travel` twice per chunk and creating many temporary pandas objects. I keep the exact same model, training loop, features, and cleaning rules, but (1) stream the CSV with an explicit `usecols`/`dtype` plan to reduce parsing overhead, (2) compute `distance_travel` once per chunk using NumPy arrays (equivalent math, fewer pandas temporaries), (3) avoid redundant work in `data_clean`, and (4) fit the imputer once (it never changes because the third column is all ones and the other two have no NaNs after parsing) while still applying the same transformation semantics. These changes reduce constant factors substantially without changing the algorithmic logic or training semantics.'
- What this solution (achieved 6.24004) has done: 'Your current score (RMSE 6.38888) is worse than the target (4.22828), so we should make small, legitimate changes that improve generalization without changing the core approach (same model class, same features, same training loop structure). The biggest issue is that `GradientBoostingRegressor` does not support true incremental learning, and your loop is effectively re-fitting from scratch on each chunk while keeping `warm_start=True`, which leads to unstable/biased training and poor performance. A minimal fix that preserves the same model and loop semantics is to actually grow the ensemble across chunks by increasing `n_estimators` each iteration (warm-start behavior), fitting on each new chunk; this is the closest you can get to “incremental” training with this estimator. Additionally, we keep the exact same features/cleaning, but we add a tiny numeric safeguard in `distance_travel` to avoid division-by-zero in the angle computation, which can otherwise introduce NaNs/inf and harm the fit.'
- What this solution (achieved 6.69867) has done: 'The timeout is driven mainly by repeatedly fitting `GradientBoostingRegressor` on a 2,000,000-row reservoir every chunk (plus doing an `rng.permutation(acc_n)` and a `regr.score` each loop), which makes training cost grow quickly and does a lot of redundant work. The fastest correctness-preserving fix is to keep the exact same model and warm-start training loop, but change the reservoir from a “growing full refit” to a fixed-size rolling window (still accumulated-training behavior, but bounded and constant-time per chunk), and eliminate the costly full permutation by using deterministic contiguous slices. Additionally, we reduce CSV parsing overhead with tighter `read_csv` options (still identical rows/columns) and avoid repeated dtype conversions/copies where not needed. These changes keep the same features, same filtering, same imputation, same GBDT setup and warm-start training semantics, but remove the superlinear work that causes the 10+ minute timeout.'
- What this solution (achieved 6.69867) has done: 'Your current RMSE (6.69867) is worse than the target (4.22828), so we should make small changes that improve accuracy without changing the core model/loop/feature set. The biggest accuracy bug is the rolling “ring buffer” update: as written, `X_acc[:acc_n]` is not guaranteed to contain the most recent `acc_n` samples (it contains stale/overwritten rows), so the model is trained on corrupted data, harming RMSE. I fix this by keeping the same fixed-size reservoir concept but using a contiguous “sliding window” buffer (shift-left + append) so that `X_acc[:acc_n]` always holds valid training rows. Everything else (distance feature, filtering rules, imputer usage, GradientBoostingRegressor warm-start + growing `n_estimators`) stays the same, and the script still writes `submission.csv`.'
- What this solution (achieved 6.19632) has done: 'Your RMSE (6.69867) is worse than the target (4.22828), so we should improve accuracy with minimal, semantics-preserving changes. The biggest accuracy issue is that the model is being trained on a rolling window that discards most historical data and also uses a weak/unstable internal holdout (the last 10% of each chunk), both of which hurt generalization; a tiny change that keeps the same model, features, and warm-start loop is to switch the bounded “sliding window” to a bounded *reservoir sampler* so the training set remains representative of the whole stream. Additionally, we should make the train/holdout split randomized within each chunk (same proportion, same purpose) so the score print is less biased by file ordering, without affecting the model’s training objective or features. These changes keep the exact same core approach (GBR warm_start growing trees on chunked data, same 3 features, same cleaning rules, same imputer) while typically moving RMSE down toward your target.'
- What this solution (achieved 5.38062) has done: 'We keep the exact same core model (GradientBoostingRegressor with warm_start, growing n_estimators) and the same 3-feature setup, but fix two accuracy-critical issues that are currently holding RMSE back. First, we stop feeding raw geographic outliers into the test predictions by applying the same basic validity mask used in training, and we fall back to a safe baseline prediction (the training mean fare) for any filtered-out test rows to preserve submission row count/order. Second, we make the distance feature more stable and closer to what the competition’s strong baselines use by switching to a true haversine distance (still “distance from lat/lon” as the single main feature, same 3 columns, same training loop), which typically reduces RMSE substantially on this dataset. These changes are minimal, end-to-end safe, and should move your score down toward the target.'

# 9. Code solution

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
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16",
}
DTYPES_TEST = {
    "key": "object",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16",
}


def chunck_generator(filename, chunk_size=10**6):
    if "train" in os.path.basename(filename):
        usecols = TRAIN_USECOLS
        dtypes = DTYPES_TRAIN
    else:
        usecols = TEST_USECOLS
        dtypes = DTYPES_TEST

    reader = pd.read_csv(
        filename,
        sep=",",
        iterator=True,
        chunksize=chunk_size,
        usecols=usecols,
        dtype=dtypes,
        engine="c",
        low_memory=False,
        memory_map=True,
        na_values=["", " "],
        keep_default_na=True,
        cache_dates=False,
    )
    for chunk in reader:
        yield chunk




## === cell 1
def haversine_distance_km(plon, plat, dlon, dlat):
    plon = np.asarray(plon, dtype=np.float64)
    plat = np.asarray(plat, dtype=np.float64)
    dlon = np.asarray(dlon, dtype=np.float64)
    dlat = np.asarray(dlat, dtype=np.float64)

    r = 6371.0
    phi1 = np.deg2rad(plat)
    phi2 = np.deg2rad(dlat)
    dphi = np.deg2rad(dlat - plat)
    dlmb = np.deg2rad(dlon - plon)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * (
        np.sin(dlmb / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return r * c




## === cell 2
def clean_and_featurize_train_chunk(df):
    fare = df["fare_amount"].to_numpy(dtype=np.float64, copy=False)
    pcount = df["passenger_count"].to_numpy(dtype=np.int16, copy=False)

    plon = df["pickup_longitude"].to_numpy(dtype=np.float64, copy=False)
    plat = df["pickup_latitude"].to_numpy(dtype=np.float64, copy=False)
    dlon = df["dropoff_longitude"].to_numpy(dtype=np.float64, copy=False)
    dlat = df["dropoff_latitude"].to_numpy(dtype=np.float64, copy=False)

    dist_km = haversine_distance_km(plon, plat, dlon, dlat)

    mask = (pcount > 0) & (fare > 0) & (dist_km > 0) & (dist_km < 60) & (fare < 100)

    if not np.any(mask):
        return None

    fare = fare[mask]
    pcount = pcount[mask].astype(np.float64, copy=False)
    dist_km = dist_km[mask]

    n = fare.shape[0]
    X = np.empty((n, 3), dtype=np.float64)
    X[:, 0] = dist_km
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

RESERVOIR_MAX = 600_000
X_acc = np.empty((RESERVOIR_MAX, 3), dtype=np.float64)
y_acc = np.empty((RESERVOIR_MAX,), dtype=np.float64)
acc_n = 0
seen_n = 0

sum_fare = 0.0
cnt_fare = 0


def _reservoir_update(X_new, y_new, X_acc, y_acc, acc_n, seen_n, RESERVOIR_MAX, rng):
    n_new = X_new.shape[0]
    if n_new == 0:
        return acc_n, seen_n

    for i in range(n_new):
        seen_n += 1
        if acc_n < RESERVOIR_MAX:
            X_acc[acc_n] = X_new[i]
            y_acc[acc_n] = y_new[i]
            acc_n += 1
        else:
            j = rng.randint(0, seen_n)
            if j < RESERVOIR_MAX:
                X_acc[j] = X_new[i]
                y_acc[j] = y_new[i]
    return acc_n, seen_n


while t > 0:
    print("chunk remaining:", t)
    try:
        df = next(gen)
    except StopIteration:
        print("Reached end of training file.")
        break

    out = clean_and_featurize_train_chunk(df)
    if out is None:
        t -= 1
        continue
    X_all, y_all = out
    l = X_all.shape[0]
    if l < 10:
        t -= 1
        continue

    sum_fare += float(np.sum(y_all))
    cnt_fare += int(y_all.shape[0])

    idx = rng.permutation(l)
    split = int(0.9 * l)
    train_idx = idx[:split]
    test_idx = idx[split:]

    train_X = X_all[train_idx]
    train_y = y_all[train_idx]
    test_X = X_all[test_idx]
    test_y = y_all[test_idx]

    if not imp_fitted:
        imp = imp.fit(train_X)
        imp_fitted = True

    train_X = imp.transform(train_X)
    test_X = imp.transform(test_X)

    acc_n, seen_n = _reservoir_update(
        train_X, train_y, X_acc, y_acc, acc_n, seen_n, RESERVOIR_MAX, rng
    )

    regr.set_params(n_estimators=regr.n_estimators + TREES_PER_CHUNK)
    regr = incremental_training(X_acc[:acc_n, :], y_acc[:acc_n], regr)

    print("R^2 on held-out chunk split:", regr.score(test_X, test_y))
    t -= 1

if not imp_fitted:
    imp = imp.fit(np.array([[0.0, 1.0, 1.0]], dtype=np.float64))
    imp_fitted = True

train_mean_fare = (sum_fare / max(cnt_fare, 1)) if cnt_fare > 0 else 11.35
print("Training mean fare (cleaned):", train_mean_fare)



## === cell 5
test_df = pd.read_csv(
    TEST_PATH,
    usecols=TEST_USECOLS,
    dtype=DTYPES_TEST,
    engine="c",
    low_memory=False,
    memory_map=True,
    na_values=["", " "],
    keep_default_na=True,
    cache_dates=False,
)

plon = test_df["pickup_longitude"].to_numpy(dtype=np.float64, copy=False)
plat = test_df["pickup_latitude"].to_numpy(dtype=np.float64, copy=False)
dlon = test_df["dropoff_longitude"].to_numpy(dtype=np.float64, copy=False)
dlat = test_df["dropoff_latitude"].to_numpy(dtype=np.float64, copy=False)
pcount = test_df["passenger_count"].to_numpy(dtype=np.float64, copy=False)

dist_km = haversine_distance_km(plon, plat, dlon, dlat)

valid_mask = (pcount > 0) & np.isfinite(dist_km) & (dist_km > 0) & (dist_km < 60)

test_X = np.empty((len(test_df), 3), dtype=np.float64)
test_X[:, 0] = dist_km
test_X[:, 1] = pcount
test_X[:, 2] = 1.0

test_X = imp.transform(test_X)

predicted_fare = np.full((len(test_df),), train_mean_fare, dtype=np.float64)
if np.any(valid_mask):
    predicted_fare[valid_mask] = regr.predict(test_X[valid_mask])

predicted_fare = np.maximum(predicted_fare, 0)

print(predicted_fare[:10])

my_submission = pd.DataFrame({"key": test_df["key"], "fare_amount": predicted_fare})
my_submission.to_csv("submission.csv", index=False)
print(my_submission.head())
print("Wrote submission.csv with shape:", my_submission.shape)
