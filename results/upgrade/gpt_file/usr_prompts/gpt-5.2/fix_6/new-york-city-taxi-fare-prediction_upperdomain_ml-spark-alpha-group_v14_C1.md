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

4.20022

# 6. Current score

5.70761

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.70026) has done: 'The timeout comes from repeatedly re-reading and fully training a new `GradientBoostingRegressor(n_estimators=100)` on each of 56 huge 1M-row chunks; this is computationally infeasible in 600s and also discards all prior training each loop. To preserve the same model architecture and evaluation semantics while removing wasted work, the optimized version trains the same regressor exactly once on a single preprocessed chunk (still using the same feature extraction, cleaning, imputation, and model hyperparameters) and then proceeds to test-time prediction identically. Runtime is further reduced by avoiding unnecessary plots, using `usecols` + explicit dtypes when reading CSVs, and making `distance_travel` compute via NumPy arrays to cut pandas overhead while producing identical columns/values. All file paths and the core feature logic and model remain unchanged.'
- What this solution (achieved 5.48656) has done: 'Your current score (5.70026 RMSE) is worse than the target (4.20022), so we should improve accuracy with the smallest changes that keep the same model and feature logic. The biggest gain available without changing the model is to train on more representative data than a single first chunk: we fit the same `GradientBoostingRegressor(n_estimators=100)` once, but on a small stratified sample accumulated from a few chunks (still within time), which reduces chunk-order bias. We also fix a subtle preprocessing bug where `distance_travel` was computed twice and then recomputed inside `data_clean` on an already-mutated frame; we compute it exactly once per frame to keep filtering consistent. Finally, we keep everything else (features, imputer, model hyperparameters, submission format/paths) identical and ensure a valid `submission.csv` is written.'
- What this solution (achieved 5.43907) has done: 'We move your RMSE down toward 4.20022 by making the training sample more representative while keeping the exact same model (`GradientBoostingRegressor(n_estimators=100)`) and the same two core features (distance + passenger_count). The biggest low-risk gain is to sample uniformly across the whole training file (not just the first few chunks), so we use a small set of evenly spaced chunks via `skiprows` and keep the same per-chunk sampling and preprocessing. We also ensure the “holdout split” is randomized (still 90/10) so it’s not biased by concatenation order, without affecting Kaggle submission semantics. Everything else (distance computation, cleaning rules, imputer strategy, submission schema/path) stays the same and still produces `submission.csv`.'
- What this solution (achieved 5.70761) has done: 'To reduce RMSE toward the 4.20022 target without changing the model or feature set, I keep the exact same `GradientBoostingRegressor(n_estimators=100)` and the same two core inputs (`distance_travel`, `passenger_count`), but make the training pool more representative of the whole dataset. The smallest high-impact change is to draw multiple random chunks using `skiprows` with a deterministic random row-start (instead of evenly spaced starts), which reduces bias from periodicity/order effects in the huge CSV while staying within the same time budget. I also clip negative predictions to 0.0 (fares can’t be negative), which typically improves RMSE slightly without altering the learning procedure. Everything else (cleaning rules, imputer, submission schema/path) remains the same and still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D

from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor

INPUT_DIR = "/kaggle/input/new-york-city-taxi-fare-prediction"
print("Input exists:", os.path.exists(INPUT_DIR))
print("Files:", os.listdir(INPUT_DIR)[:20])

TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

TRAIN_USECOLS = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
TEST_USECOLS = [
    "key",
    "pickup_datetime",
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
    "passenger_count": "int8",
}
DTYPES_TEST = {
    "key": "object",
    "pickup_longitude": "float64",
    "pickup_latitude": "float64",
    "dropoff_longitude": "float64",
    "dropoff_latitude": "float64",
    "passenger_count": "int8",
}

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)




## === cell 1
def chunck_generator(filename, header=False, chunk_size=10**6):
    usecols = TRAIN_USECOLS if os.path.basename(filename) == "train.csv" else None
    dtypes = DTYPES_TRAIN if os.path.basename(filename) == "train.csv" else None
    for chunk in pd.read_csv(
        filename,
        delimiter=",",
        iterator=True,
        chunksize=chunk_size,
        usecols=usecols,
        dtype=dtypes,
        parse_dates=["pickup_datetime"],
    ):
        yield chunk




## === cell 2
alpha_ang = 0.506


def distance_travel(df):
    p_long = df["pickup_longitude"].to_numpy(dtype=np.float64, copy=False)
    d_long = df["dropoff_longitude"].to_numpy(dtype=np.float64, copy=False)
    p_lat = df["pickup_latitude"].to_numpy(dtype=np.float64, copy=False)
    d_lat = df["dropoff_latitude"].to_numpy(dtype=np.float64, copy=False)

    abs_diff_longitude = np.abs(d_long - p_long) * 50.0
    abs_diff_latitude = np.abs(d_lat - p_lat) * 69.0
    displacement_vector = np.sqrt(
        abs_diff_latitude * abs_diff_latitude + abs_diff_longitude * abs_diff_longitude
    )

    denom = abs_diff_latitude.copy()
    denom[denom == 0.0] = np.nan
    angle = np.arctan(abs_diff_longitude / denom)

    actual_long = np.abs(displacement_vector * np.sin(angle - alpha_ang))
    actual_lat = np.abs(displacement_vector * np.cos(angle - alpha_ang))
    dist = actual_long + actual_lat
    dist = np.nan_to_num(dist, nan=0.0)

    df["abs_diff_longitude"] = abs_diff_longitude
    df["abs_diff_latitude"] = abs_diff_latitude
    df["displacement_vector"] = displacement_vector
    df["actual_long"] = actual_long
    df["actual_lat"] = actual_lat
    df["distance_travel"] = dist
    return df




## === cell 3
def data_clean(df):
    df = df[df.passenger_count > 0]
    df.fare_amount = df.fare_amount.astype(np.float64)
    df = df[df.fare_amount > 0]
    df = df[df.distance_travel > 0]
    return df




## === cell 4
def remove_outliers(df):
    df = df[df.distance_travel < 30]
    df = df[df.fare_amount < 60]
    return df




## === cell 5
def graph_present(df):
    test = df[df.passenger_count == 1]
    plot = test.iloc[: len(test)].plot.scatter("distance_travel", "fare_amount")
    plot = df.iloc[:100000].plot.scatter("distance_travel", "fare_amount")




## === cell 6
def data_preprocessing(df):
    df = distance_travel(df)
    df = data_clean(df)
    df = remove_outliers(df)
    return df




## === cell 7
if False:
    df = pd.read_csv(
        TRAIN_PATH,
        nrows=100_000,
        usecols=TRAIN_USECOLS,
        dtype=DTYPES_TRAIN,
        parse_dates=["pickup_datetime"],
    )
    df = distance_travel(df)
    df = df[df.passenger_count == 1]
    df = df[df.distance_travel < 30]
    df.distance_travel.hist(bins=50, figsize=(12, 4))
    plt.xlabel("distance miles")
    plt.title("Histogram ride distances in miles")
    print(df.distance_travel.describe())



## === cell 8
if False:
    df = pd.read_csv(
        TRAIN_PATH,
        nrows=100_000,
        usecols=TRAIN_USECOLS,
        dtype=DTYPES_TRAIN,
        parse_dates=["pickup_datetime"],
    )
    df = distance_travel(df)
    df = df[df.passenger_count <= 6]
    df = df[df.distance_travel < 30]
    df = df[df.fare_amount > 0]
    df = df[df.fare_amount < 60]
    df.fare_amount.hist(bins=50, figsize=(12, 4))
    plt.xlabel("fare_amount")
    print(df.fare_amount.describe())



## === cell 9
regr = None
imp = SimpleImputer(missing_values=np.nan, strategy="mean")

CHUNK_SIZE = 1_000_000
MAX_CHUNKS = 6
SAMPLE_PER_CHUNK = 80_000

APPROX_TOTAL_ROWS = 55_000_000

rng = np.random.RandomState(RANDOM_STATE)
max_start = max(1, APPROX_TOTAL_ROWS - CHUNK_SIZE - 1)
start_rows = rng.randint(1, max_start + 1, size=MAX_CHUNKS)

pooled = []
for i, start_row in enumerate(start_rows):
    skip = range(1, int(start_row))
    chunk = pd.read_csv(
        TRAIN_PATH,
        usecols=TRAIN_USECOLS,
        dtype=DTYPES_TRAIN,
        parse_dates=["pickup_datetime"],
        skiprows=skip,
        nrows=CHUNK_SIZE,
    )
    chunk = data_preprocessing(chunk)
    if len(chunk) == 0:
        continue
    if len(chunk) > SAMPLE_PER_CHUNK:
        chunk = chunk.sample(n=SAMPLE_PER_CHUNK, random_state=RANDOM_STATE + i)
    pooled.append(chunk)

if len(pooled) == 0:
    raise RuntimeError("No training data left after cleaning; adjust filters/chunking.")

df = pd.concat(pooled, axis=0, ignore_index=True)

l = len(df)
if l < 10_000:
    raise RuntimeError(
        f"Pooled training data too small after cleaning ({l}); increase MAX_CHUNKS/SAMPLE_PER_CHUNK."
    )

df = df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

df_train = df[: int(0.9 * l)]
df_test = df[int(0.9 * l) :]

train_X = np.column_stack(
    (
        df_train.distance_travel.to_numpy(copy=False),
        df_train.passenger_count.to_numpy(copy=False),
        np.ones(len(df_train)),
    )
)
test_X = np.column_stack(
    (
        df_test.distance_travel.to_numpy(copy=False),
        df_test.passenger_count.to_numpy(copy=False),
        np.ones(len(df_test)),
    )
)
train_y = df_train.fare_amount.to_numpy(copy=False)
test_y = df_test.fare_amount.to_numpy(copy=False)

imp = imp.fit(train_X)
train_X = imp.transform(train_X)
test_X = imp.transform(test_X)

regr = GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE)
regr = regr.fit(train_X, train_y)

print("R^2 on pooled holdout:", regr.score(test_X, test_y))
print("Pooled rows used:", len(df), " Train:", len(df_train), " Holdout:", len(df_test))

if regr is None:
    raise RuntimeError(
        "Training did not produce a fitted model; check data cleaning filters."
    )



## === cell 10
test_df = pd.read_csv(
    TEST_PATH,
    usecols=TEST_USECOLS,
    dtype=DTYPES_TEST,
    parse_dates=["pickup_datetime"],
)
distance_travel(test_df)
print(test_df.head())



## === cell 11
test_X = np.column_stack(
    (
        test_df.distance_travel.to_numpy(copy=False),
        test_df.passenger_count.to_numpy(copy=False),
        np.ones(len(test_df)),
    )
)
test_X = imp.transform(test_X)
predicted_fare = regr.predict(test_X)

predicted_fare = np.maximum(predicted_fare, 0.0)

print(predicted_fare[:10])



## === cell 12
my_submission = pd.DataFrame({"key": test_df.key, "fare_amount": predicted_fare})
my_submission.to_csv("submission.csv", index=False)
print(my_submission.head())
print("Wrote submission.csv with shape:", my_submission.shape)
