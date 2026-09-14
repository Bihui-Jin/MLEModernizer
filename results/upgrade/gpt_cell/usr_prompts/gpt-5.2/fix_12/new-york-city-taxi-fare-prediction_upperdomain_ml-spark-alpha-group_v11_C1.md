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

4.21737

# 6. Current score

6.13261

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.39478) has done: 'Diagnosis: The crash happens in cell 6 when fitting `SimpleImputer` because it was configured with `missing_values="NaN"` (a string). In scikit-learn 1.2, `missing_values` must be a numeric value (e.g., `np.nan`) when the input `X` is numeric; otherwise it raises the shown `ValueError`. The core training logic does not actually use the imputed output afterward, so we should not alter the model/training flow—only fix the incorrect `missing_values` type to unblock execution.

Patch summary: In cell 6, change the imputer initialization from `missing_values="NaN"` to `missing_values=np.nan` so `imp.fit(train_X)` works with `float64` arrays. No other logic, data paths, or model settings are changed.

Updated cells: (cell 6 only)

Compatibility notes for cell k+1: This change only affects `imp` behavior inside the training loop and does not modify any variables used in cell 8 (`tdf`, `distance_travel`, file paths). Cell 8 remains compatible and unchanged.

Assumptions: The intended missing-value marker in numeric arrays is actual NaN (`np.nan`), not the literal string `"NaN"`.'
- What this solution (achieved 6.18976) has done: 'The timeout is dominated by repeatedly parsing 55 chunks of a 55M-row CSV with `parse_dates`, plus heavy Pandas column creation inside `distance_travel` and repeated imputer fitting. I keep the exact same training loop/logic and model, but reduce overhead by (1) disabling date parsing (the datetime column is unused), (2) computing the same `distance_travel` feature with vectorized NumPy arrays to avoid multiple Pandas intermediate columns, and (3) avoiding redundant `SimpleImputer.fit()` each chunk (mean-imputation is a no-op here because the constructed features have no NaNs; keeping the transformer preserves semantics). These changes are mathematically equivalent for the used features and should drastically cut per-chunk CPU time and CSV parsing cost while preserving predictions up to negligible floating-point differences. I also use faster CSV parsing knobs that don’t change results (C engine, `memory_map=True`) and keep all file paths unchanged.'
- What this solution (achieved 6.13261) has done: 'The timeout is dominated by repeatedly fitting `GradientBoostingRegressor` on very large per-chunk datasets (up to ~1e6 rows) 55 times, plus unnecessary per-iteration allocations and DataFrame overhead. To preserve the exact training scheme (same number of chunks, same estimator, same warm-start growth, same feature set and split semantics), the main speed win is to keep the logic identical but reduce CPU and pandas overhead: convert chunks to NumPy arrays once, avoid creating intermediate DataFrames/columns, preallocate and reuse feature matrices, and avoid unnecessary copies/casts. We also reduce per-iteration Python overhead by factoring repeated operations and using in-place fills, while keeping determinism (same RNG usage) and identical evaluation steps. File paths, model, imputer, and overall loop structure are preserved.'

# 9. Code solution

## === cell 0
from sklearn.impute import SimpleImputer as Imputer
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import GradientBoostingRegressor
import os

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

print(os.listdir("../input"))

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
    "fare_amount": "float64",
    "pickup_longitude": "float64",
    "pickup_latitude": "float64",
    "dropoff_longitude": "float64",
    "dropoff_latitude": "float64",
    "passenger_count": "int16",
}
DTYPES_TEST = {
    "pickup_longitude": "float64",
    "pickup_latitude": "float64",
    "dropoff_longitude": "float64",
    "dropoff_latitude": "float64",
    "passenger_count": "int16",
}


def chunck_generator(filename, chunk_size=10**6):
    reader = pd.read_csv(
        filename,
        delimiter=",",
        iterator=True,
        chunksize=chunk_size,
        usecols=TRAIN_USECOLS,
        dtype=DTYPES_TRAIN,
        engine="c",
        memory_map=True,
        low_memory=False,
    )
    for chunk in reader:
        yield chunk




## === cell 1
alpha_ang = 0.506


def distance_travel_arrays(plon, plat, dlon0, dlat0):
    dlon = np.abs(dlon0 - plon) * 50.0
    dlat = np.abs(dlat0 - plat) * 69.0

    disp = np.sqrt(dlat * dlat + dlon * dlon)

    theta = np.arctan2(dlon, dlat) - alpha_ang  # equivalent angle, avoids division
    actual_long = np.abs(disp * np.sin(theta))
    actual_lat = np.abs(disp * np.cos(theta))
    return actual_long + actual_lat


def distance_travel(df):
    plon = df["pickup_longitude"].to_numpy(copy=False)
    plat = df["pickup_latitude"].to_numpy(copy=False)
    dlon0 = df["dropoff_longitude"].to_numpy(copy=False)
    dlat0 = df["dropoff_latitude"].to_numpy(copy=False)
    df["distance_travel"] = distance_travel_arrays(plon, plat, dlon0, dlat0)
    return df




## === cell 2
def data_clean(df):
    fa = df["fare_amount"].to_numpy(copy=False).astype(np.float64, copy=False)
    pc = df["passenger_count"].to_numpy(copy=False)
    dist = df["distance_travel"].to_numpy(copy=False)

    mask = (pc > 0) & (fa > 0) & (dist > 0)
    if not np.all(mask):
        df = df.loc[mask].copy()
        df["fare_amount"] = df["fare_amount"].astype(np.float64, copy=False)
    else:
        df["fare_amount"] = df["fare_amount"].astype(np.float64, copy=False)
    return df




## === cell 3
def remove_outliers(df):
    dist = df["distance_travel"].to_numpy(copy=False)
    fa = df["fare_amount"].to_numpy(copy=False)
    mask = (dist < 30) & (fa < 100)
    if not np.all(mask):
        df = df.loc[mask]
    return df




## === cell 4
def graph_presesnt(df):
    test = df[df.passenger_count == 1]
    plot = test.iloc[: len(test)].plot.scatter("distance_travel", "fare_amount")
    plot = df.iloc[:100000].plot.scatter("distance_travel", "fare_amount")




## === cell 5
def incremental_training(train_X, train_y, regr):
    regr.fit(train_X, train_y)
    return regr




## === cell 6
filename = r"../input/train.csv"
gen = chunck_generator(filename=filename)

regr = GradientBoostingRegressor(n_estimators=100, warm_start=True)
imp = Imputer(missing_values=np.nan, strategy="mean")

t = 55
trees_per_chunk = 10
chunk_i = 0

imputer_fitted = False

rng = np.random.RandomState(42)

train_X_buf = None
test_X_buf = None

while t > 0:
    df = next(gen)

    plon = df["pickup_longitude"].to_numpy(copy=False)
    plat = df["pickup_latitude"].to_numpy(copy=False)
    dlon0 = df["dropoff_longitude"].to_numpy(copy=False)
    dlat0 = df["dropoff_latitude"].to_numpy(copy=False)
    pc_all = df["passenger_count"].to_numpy(copy=False)
    fa_all = df["fare_amount"].to_numpy(copy=False).astype(np.float64, copy=False)

    dist_all = distance_travel_arrays(plon, plat, dlon0, dlat0).astype(
        np.float64, copy=False
    )

    mask = (
        (pc_all > 0) & (fa_all > 0) & (dist_all > 0) & (dist_all < 30) & (fa_all < 100)
    )
    if not np.any(mask):
        t -= 1
        chunk_i += 1
        continue

    dist = dist_all[mask]
    pc = pc_all[mask].astype(np.float64, copy=False)
    y = fa_all[mask]

    l = y.shape[0]
    idx = np.arange(l)
    rng.shuffle(idx)
    split = int(0.9 * l)
    train_idx = idx[:split]
    test_idx = idx[split:]

    train_dist = dist[train_idx]
    train_pc = pc[train_idx]
    test_dist = dist[test_idx]
    test_pc = pc[test_idx]

    ntr = train_dist.shape[0]
    nte = test_dist.shape[0]

    if (train_X_buf is None) or (train_X_buf.shape[0] < ntr):
        train_X_buf = np.empty((ntr, 3), dtype=np.float64)
    if (test_X_buf is None) or (test_X_buf.shape[0] < nte):
        test_X_buf = np.empty((nte, 3), dtype=np.float64)

    train_X = train_X_buf[:ntr]
    test_X = test_X_buf[:nte]

    train_X[:, 0] = train_dist
    train_X[:, 1] = train_pc
    train_X[:, 2] = 1.0

    test_X[:, 0] = test_dist
    test_X[:, 1] = test_pc
    test_X[:, 2] = 1.0

    train_y = y[train_idx]
    test_y = y[test_idx]

    if not imputer_fitted:
        imp = imp.fit(train_X)
        imputer_fitted = True

    train_X = imp.transform(train_X)
    test_X = imp.transform(test_X)

    if chunk_i > 0:
        regr.set_params(n_estimators=regr.n_estimators + trees_per_chunk)

    regr = incremental_training(train_X, train_y, regr)
    print(regr.score(test_X, test_y))

    chunk_i += 1
    t -= 1



## === cell 7
tdf = pd.read_csv(
    "../input/test.csv",
    usecols=TEST_USECOLS,
    dtype=DTYPES_TEST,
    engine="c",
    memory_map=True,
    low_memory=False,
)
distance_travel(tdf)
tdf.head()



## === cell 8
test_dist = tdf["distance_travel"].to_numpy(dtype=np.float64, copy=False)
test_pc = tdf["passenger_count"].to_numpy(dtype=np.float64, copy=False)

ttrain_X = np.empty((test_dist.shape[0], 3), dtype=np.float64)
ttrain_X[:, 0] = test_dist
ttrain_X[:, 1] = test_pc
ttrain_X[:, 2] = 1.0

ttrain_X = imp.transform(ttrain_X)
output = regr.predict(ttrain_X)
print(output)



## === cell 9
my_submission = pd.DataFrame({"key": tdf.key, "fare_amount": output})
my_submission.to_csv("submission.csv", index=False)
my_submission.head()
