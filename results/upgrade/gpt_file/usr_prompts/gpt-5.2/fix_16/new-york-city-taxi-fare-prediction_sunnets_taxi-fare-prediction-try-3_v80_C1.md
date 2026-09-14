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
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
sklearn-pandas==2.2.0
tf_keras==2.18.0

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

4.3735

# 6. Current score

5.11681

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.68165) has done: 'I fix the runtime errors caused by the Keras 3 API changes (optimizer name, removed `vis_utils`, and protobuf-related import crash) while keeping the same model architecture/training loop and feature logic. I also correct a couple of logic bugs in the time-feature flags (they currently always return 1 due to incorrect conditions), which should legitimately improve RMSE toward your target without changing the overall approach. I ensure file paths match your environment (`/kaggle/input/...`) and that a valid `submissiontry_water.csv` is always written with the required `key,fare_amount` columns. Finally, I make the plotting/model-visualization cells non-fatal so the notebook runs end-to-end.'
- What this solution (achieved 5.65597) has done: 'I fix the immediate runtime crash in the first cell (`AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`) by pinning protobuf to a compatible pure-Python implementation *before* importing `tf_keras`, which is a common Kaggle/TensorFlow-protobuf mismatch. Then I correct a key data-loading bug where `fare_amount` was accidentally excluded from the training read (`usecols` missing it), which severely hurts model learning and Kaggle RMSE; this change preserves your model/training loop but restores the intended target signal. Finally, I keep all paths and submission writing unchanged and add a small safety clamp on predictions (non-negative) to avoid impossible negative fares that can worsen RMSE, without altering the core approach.'
- What this solution (achieved 5.73667) has done: 'I fix the initial import crash by avoiding the TensorFlow/tf_keras + protobuf incompatibility and switching to a scikit-learn regressor while keeping the same feature engineering and train/validation split semantics. I also fix the broken Kaggle file paths (`/kaggle/input/...` should point into the competition folder) so the data loads correctly and all downstream `NameError`s disappear. To keep the output valid for Kaggle, I ensure we always write `submissiontry_water.csv` with exactly `key,fare_amount` and the correct row order. Finally, I keep prediction post-processing minimal (non-negative clamp) to avoid invalid negative fares that can worsen RMSE.'
- What this solution (achieved 5.5863) has done: 'To move RMSE down toward your 4.3735 target (current 5.73667; lower is better), I make the smallest changes that improve generalization without changing your feature set or training loop structure. The biggest issue is that `testKaggle` is not cleaned the same way as training (so it contains out-of-distribution rows your model never sees), which typically hurts Kaggle RMSE; I apply the same `clean()` to `testKaggle` and then align the submission keys back to the original row order. I also keep your RandomForest but make it slightly less biased (very small hyperparameter tweak) while staying in the same model family and training approach. Finally, I ensure the submission has exactly the test set’s row count and correct ordering.'
- What this solution (achieved 5.75239) has done: 'We make two minimal, score-relevant changes that keep your feature engineering, model family, and training flow intact while improving generalization toward the 4.3735 RMSE target (current 5.5863; lower is better). First, we stop throwing away most of the original training data: instead of splitting 50/50 and then truncating the holdout to 10k, we use a small fixed validation split (10k) and train on the rest of the 80k sample, which typically reduces Kaggle RMSE without changing the approach. Second, we remove the MinMaxScaler (tree models don’t benefit and it can add noise) while keeping the same RandomForestRegressor and features; this is a safe, semantics-preserving adjustment for RF. Submission writing, test cleaning + key re-alignment, and prediction clamping stay the same.'
- What this solution (achieved 6.0416) has done: 'Your current RMSE (5.75239) is worse than the target (4.3735), so we should make small, legitimate improvements that don’t change your overall modeling approach. The biggest score-relevant issue is that you fit the RandomForest on raw fares without handling the very heavy-tailed target; for RMSE, predicting in log-space and converting back often reduces error while keeping the same model family and training loop. We keep all feature engineering identical, but train the same RandomForest on `log1p(fare_amount)` and then `expm1` predictions back to dollars (plus the existing non-negative clamp). Additionally, we apply the same log transform consistently for validation/test RMSE reporting (so you still see meaningful diagnostics), and leave submission format and row alignment unchanged.'
- What this solution (achieved 5.20077) has done: 'Your current RMSE (6.0416; lower is better) is meaningfully worse than the target (4.3735), so we make a small, legitimate improvement that keeps your core approach (same features, same RandomForest training loop, same log1p target) but improves generalization. The highest-impact minimal change is to stop over-filtering the training data by removing the `fare_amount <= 50` cap, which throws away many legitimate higher fares and biases predictions low on Kaggle. We keep cleaning identical for coordinates/passengers, but widen the target outlier filter to a standard NYC Taxi range (0 < fare <= 250) that preserves more signal while still removing extreme label noise. Submission writing, test re-alignment, and the non-negative clamp remain unchanged so the pipeline still produces a valid `submissiontry_water.csv`.'
- What this solution (achieved 5.16701) has done: 'To move RMSE down toward your 4.3735 target (current 5.20077; lower is better) without changing the core approach (same cleaning, same engineered features, same RandomForest + log1p target), the smallest high-impact change is to use the full 55M-row training file more effectively under the 600s limit by sampling more rows instead of only reading the first 80k. Reading only the first chunk is time-ordered and not representative; switching to `skiprows`-based random sampling reduces bias and usually improves public RMSE while keeping the model and features identical. I also increase the sample size moderately (to 200k) to improve generalization while keeping the RandomForest configuration and training loop the same. Submission writing and test re-alignment remain unchanged.'
- What this solution (achieved 5.14585) has done: 'We make two minimal, score-relevant fixes while keeping your exact feature set and RandomForest+log1p approach unchanged. First, the current `skiprows` sampling builds a massive `skip` array (~55M ints) which is slow/memory-heavy and can bias what actually gets read; we replace it with deterministic chunk-based random sampling that reliably yields an i.i.d. sample of the requested size within the 600s limit. Second, we vectorize the time-feature flags (`night`, `late_night`, `rush_hour`) to remove the expensive `df.apply(...)` calls; this doesn’t change the semantics but lets you safely increase effective training signal without timing out. These changes should reduce your RMSE toward the 4.3735 target (current 5.16701; lower is better) without altering model family, loss/metric, or engineered features.'
- What this solution (achieved 5.18028) has done: 'To reduce your RMSE toward the 4.3735 target (current 5.14585; lower is better) while keeping your RandomForest + log1p target and the same engineered features, I fix the main data issue: your “random sample” reader is currently taking the first 200k rows (time-ordered) rather than an i.i.d. sample, which hurts generalization on Kaggle. I replace it with a deterministic chunk-wise reservoir-style sampler that produces a uniform-ish sample across the whole 55M rows without building a 55M skiprows list and still fits the 600s limit. Everything else (cleaning, feature engineering, model hyperparameters, training loop, prediction post-processing, and submission writing/re-alignment) stays the same to keep changes minimal and semantics consistent. This should legitimately improve public RMSE by removing the time/ordering bias in the training sample.'
- What this solution (achieved 5.15663) has done: 'Your RMSE (5.18028) is still worse than the target (4.3735), so we should make small, legitimate improvements without changing the core RandomForest+log1p approach or feature set. The largest likely issue is the coordinate-cleaning logic: it currently filters with `pickup_latitude != 0` after already restricting to NYC bounds, which unintentionally removes almost all valid NYC rows (since NYC latitudes are never 0 and are all > 0), shrinking and biasing the training sample; fixing this keeps the same cleaning intent but stops throwing away good data. Additionally, we should avoid “training sample” early-stopping at 5M seen rows (which biases the sampling toward early file segments); letting the sampler run until it actually fills the requested reservoir improves representativeness while keeping the same sampling method. These two minimal fixes should move RMSE downward toward your target while preserving model/training semantics and still producing the same valid submission file.'
- What this solution (achieved 5.2347) has done: 'The timeout is dominated by two hotspots: (1) doing a full reservoir sample by scanning the entire 55M-row CSV, and (2) fitting a 300-tree RandomForest on ~180k+ rows, plus repeated extra predictions and many plotting cells. To preserve the same core model/feature logic while finishing under 600s, the main fix is to replace the full-file reservoir sampling with an equivalent “random sample” obtained by reading only a small deterministic fraction of the CSV (skiprows-based sampling) and then sub-sampling to exactly `DATASET_SIZE`. Additionally, we avoid expensive DataFrame `.iloc` row-by-row assignments, drop redundant duplicate predictions, and hard-skip all plotting/EDA cells (they don’t affect training or submission). All paths, features, model hyperparameters, and evaluation semantics are kept the same; only wasted work and full-file scans are removed.'
- What this solution (achieved 5.11681) has done: 'To reduce RMSE from 5.2347 toward your 4.3735 target (lower is better) with minimal disruption, I keep your exact feature set, log1p target, and RandomForest training flow, but fix two score-relevant issues: (1) the training sample is still biased because it only draws from a deterministic subset of line indices; switching to chunk-wise random sampling yields a more i.i.d. sample without scanning all 55M rows, and (2) your current cleaning drops many legitimate short trips via the `> 0.001` lat/lon deltas; relaxing this to a much smaller epsilon preserves intended “remove identical pickup/dropoff” logic while keeping real short rides (important for Kaggle RMSE). I also ensure the same cleaning is applied consistently and keep submission re-alignment identical so the output stays valid. All paths, columns, model hyperparameters, and post-processing remain unchanged.'

# 9. Code solution

## === cell 0
import os
import sys
import random
import warnings
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

SEED = 1
np.random.seed(SEED)
random.seed(SEED)

TRAIN_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
TEST_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"
SUBMISSION_NAME = "submissiontry_water.csv"

BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 0.001

DATASET_SIZE = 200000

FAST_MODE = True

print("Imports OK. Using TRAIN_PATH:", TRAIN_PATH, "TEST_PATH:", TEST_PATH)
print("Will write submission:", SUBMISSION_NAME)




## === cell 1
def clean(df):
    print(" Old size: %d" % len(df))
    df = df.dropna(how="any", axis="rows")
    print(" New size after dropna: %d" % len(df))

    has_target = "fare_amount" in df.columns

    df = df[
        ~(
            (df["dropoff_longitude"] == df["pickup_longitude"])
            & (df["dropoff_latitude"] == df["pickup_latitude"])
        )
    ]
    print(" New size after removing identical pickup/dropoff: %d" % len(df))

    df = df[
        (df["dropoff_longitude"] != 0)
        & (df["pickup_longitude"] != 0)
        & (df["dropoff_latitude"] != 0)
        & (df["pickup_latitude"] != 0)
    ]
    print(" New size after removing 0 long lat: %d" % len(df))

    MinMax = (-74.5, -72.8, 40.5, 41.8)
    df = df[
        (MinMax[0] <= df["pickup_longitude"]) & (df["pickup_longitude"] <= MinMax[1])
    ]
    df = df[
        (MinMax[0] <= df["dropoff_longitude"]) & (df["dropoff_longitude"] <= MinMax[1])
    ]
    df = df[(MinMax[2] <= df["pickup_latitude"]) & (df["pickup_latitude"] <= MinMax[3])]
    df = df[
        (MinMax[2] <= df["dropoff_latitude"]) & (df["dropoff_latitude"] <= MinMax[3])
    ]
    print(" New size after NYC lang lot: %d" % len(df))

    df = df[(df["pickup_latitude"] != 0)]
    df = df[(df["pickup_longitude"] != 0)]
    df = df[(df["dropoff_longitude"] != 0)]
    df = df[(df["dropoff_latitude"] != 0)]
    print(" New size after removing 0 long lat (redundant safety): %d" % len(df))

    eps = 1e-5
    df = df[((df["pickup_latitude"] - df["dropoff_latitude"]).abs() > eps)]
    df = df[((df["pickup_longitude"] - df["dropoff_longitude"]).abs() > eps)]
    print(f" New size after lang - lot > {eps}: %d" % len(df))

    print(" New size after only NYC: %d" % len(df))

    if has_target:
        df = df[(0 < df["fare_amount"]) & (df["fare_amount"] <= 250)]
        print(" New size after removing outliers (fare<=250): %d" % len(df))
    else:
        print(" Skipping fare_amount outlier removal for test data.")

    df = df[(df["passenger_count"] > 0)]
    print(" New size after removing 6=>passenger_count > 0 : %d" % len(df))

    return df


def remove_datapoints_from_water(df):
    raise RuntimeError(
        "remove_datapoints_from_water() requires internet to fetch a mask image; not supported in Kaggle runtime."
    )


def late_night(row):
    h = row["hour"]
    return 1 if (h <= 3) else 0


def night(row):
    h = row["hour"]
    wd = row["weekday"]
    return 1 if ((h >= 20 or h <= 6) and wd < 5) else 0


def rush_hour(row):
    h = row["hour"]
    wd = row["weekday"]
    return 1 if ((16 <= h <= 20) and wd < 5) else 0


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)
    if dt.isna().any():
        dt2 = pd.to_datetime(df["pickup_datetime"], errors="coerce")
        dt = dt.fillna(dt2)

    df["year"] = dt.dt.year.astype("int16")
    df["month"] = dt.dt.month.astype("int8")
    df["day"] = dt.dt.day.astype("int8")
    df["hour"] = dt.dt.hour.astype("int8")
    df["weekday"] = dt.dt.weekday.astype("int8")

    df["pickup_datetime"] = dt.astype(str)

    h = df["hour"].astype("int16")
    wd = df["weekday"].astype("int16")

    df["night"] = (((h >= 20) | (h <= 6)) & (wd < 5)).astype("int8")
    df["late_night"] = (h <= 3).astype("int8")
    df["rush_hour"] = ((h >= 16) & (h <= 20) & (wd < 5)).astype("int8")
    return df


def add_coordinate_features(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]

    df["latdiff"] = (lat1 - lat2).abs()
    df["londiff"] = (lon1 - lon2).abs()
    return df


def add_distances_features(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]

    df["manhattan"] = manhattan(lat1, lon1, lat2, lon2)
    df["distance"] = np.sqrt(np.abs(lon1 - lon2) ** 2 + np.abs(lat1 - lat2) ** 2)
    return df


def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    pred = np.asarray(prediction).reshape(-1)
    df = pd.DataFrame({id_column: raw_test[id_column].values, prediction_column: pred})
    df[[id_column, prediction_column]].to_csv(file_name, index=False)
    print("Output complete:", file_name, "rows:", len(df))


def plot_loss_accuracy_rmse(history):
    print("plot_loss_accuracy_rmse skipped (no Keras History object).")




## === cell 2
datatypes = {
    "key": "str",
    "fare_amount": "float32",
    "pickup_datetime": "str",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}


def read_train_random_sample_csv_fast(
    path, n_rows, seed, dtype, usecols, oversample_factor=3, chunksize=250_000
):
    rng = np.random.RandomState(seed)
    target = int(n_rows * oversample_factor)

    pieces = []
    total = 0
    t0 = time.time()

    for chunk in pd.read_csv(path, usecols=usecols, dtype=dtype, chunksize=chunksize):
        if len(chunk) == 0:
            continue
        frac = min(1.0, 1.0 / float(oversample_factor))
        take = chunk.sample(frac=frac, random_state=int(rng.randint(0, 2**31 - 1)))
        pieces.append(take)
        total += len(take)

        if total >= target:
            break

        if time.time() - t0 > 120:
            break

    if not pieces:
        df = pd.read_csv(path, usecols=usecols, dtype=dtype, nrows=n_rows)
        return df.sample(n=min(n_rows, len(df)), random_state=seed).reset_index(
            drop=True
        )

    df = pd.concat(pieces, axis=0, ignore_index=True)
    if len(df) > n_rows:
        df = df.sample(n=n_rows, random_state=seed).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    return df


trainKaggle = read_train_random_sample_csv_fast(
    TRAIN_PATH,
    n_rows=DATASET_SIZE,
    seed=SEED,
    dtype=datatypes,
    usecols=[
        "key",
        "fare_amount",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
)

testKaggle = pd.read_csv(
    TEST_PATH,
    dtype={
        "key": "str",
        "pickup_datetime": "str",
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "uint8",
    },
)

print(
    "Loaded trainKaggle columns:",
    list(trainKaggle.columns),
    "shape:",
    trainKaggle.shape,
)
print(
    "Loaded testKaggle columns:", list(testKaggle.columns), "shape:", testKaggle.shape
)



## === cell 3
train_df, test_df = train_test_split(trainKaggle, test_size=10000, random_state=SEED)



## === cell 4
print("testKaggle Size %d" % len(testKaggle))
print("train_df Size %d" % len(train_df))
print("test_df Size %d" % len(test_df))



## === cell 5
if not FAST_MODE:
    train_df.describe()



## === cell 6
if not FAST_MODE:
    test_df.describe()



## === cell 7
print("train_df clean")
train_df = clean(train_df)
test_df = clean(test_df)

print(
    "testKaggle clean (for modeling only; will re-align to original test order for submission)"
)
testKaggle_raw = testKaggle.copy()
testKaggle_cleaned = clean(testKaggle)



## === cell 8
if not FAST_MODE:
    train_df



## === cell 9
print("train_df add_time_features")
train_df = add_time_features(train_df)
print("test_df add_time_features")
test_df = add_time_features(test_df)

print("testKaggle_cleaned add_time_features")
testKaggle_cleaned = add_time_features(testKaggle_cleaned)



## === cell 10
if not FAST_MODE:
    train_df.describe()



## === cell 11
print("train_df add_coordinate_features")
train_df = add_coordinate_features(train_df)
print("test_df add_coordinate_features")
test_df = add_coordinate_features(test_df)

print("testKaggle_cleaned add_coordinate_features")
testKaggle_cleaned = add_coordinate_features(testKaggle_cleaned)



## === cell 12
if not FAST_MODE:
    train_df.describe()



## === cell 13
print("train_df add_distances_features")
train_df = add_distances_features(train_df)
print("test_df add_distances_features")
test_df = add_distances_features(test_df)

print("testKaggle_cleaned add_distances_features")
testKaggle_cleaned = add_distances_features(testKaggle_cleaned)
print("Done with Adding features")



## === cell 14
if not FAST_MODE:
    train_df.describe()



## === cell 15
if not FAST_MODE:
    try:
        plot = train_df.iloc[:2000].plot.scatter("latdiff", "londiff")
        plot = train_df.iloc[:2000].plot.scatter("fare_amount", "passenger_count")
        plot = train_df.iloc[:2000].plot.scatter("fare_amount", "year")
        plot = train_df.iloc[:2000].plot.scatter("fare_amount", "month")
        plot = train_df.iloc[:2000].plot.scatter("fare_amount", "day")
        plot = train_df.iloc[:2000].plot.scatter("fare_amount", "hour")
        plot = train_df.iloc[:2000].plot.scatter("fare_amount", "weekday")
        plot = train_df.iloc[:2000].plot.scatter("fare_amount", "night")
        plot = train_df.iloc[:2000].plot.scatter("fare_amount", "late_night")
        plot = train_df.iloc[:2000].plot.scatter("fare_amount", "rush_hour")
        plt.show()
    except Exception as e:
        print("Plotting skipped:", repr(e))



## === cell 16
dropped_columns = [
    "pickup_datetime",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "key",
]

train_df = train_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)
testKaggle_clean = testKaggle_cleaned.drop(dropped_columns, axis=1)

print("Done with dropped_columns")



## === cell 17
train_df.shape



## === cell 18
if not FAST_MODE:
    train_df.describe()



## === cell 19
if not FAST_MODE:
    test_df.describe()



## === cell 20
train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=SEED)



## === cell 21
train_df_main = train_df
validation_df_main = validation_df



## === cell 22
if not FAST_MODE:
    validation_df.describe()



## === cell 23
train_labels = train_df["fare_amount"].values
validation_labels = validation_df["fare_amount"].values
test_labels = test_df["fare_amount"].values

train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)
test_df = test_df.drop(["fare_amount"], axis=1)

print("Done with Labels")



## === cell 24
if not FAST_MODE:
    test_labels



## === cell 25
if not FAST_MODE:
    train_df.describe()



## === cell 26
if not FAST_MODE:
    validation_df.describe()



## === cell 27
if not FAST_MODE:
    test_df.describe()



## === cell 28
train_df_scaled = train_df.values
validation_df_scaled = validation_df.values
test_scaled = test_df.values
testKaggle_scaled = testKaggle_clean.values



## === cell 29
if not FAST_MODE:
    test_scaled



## === cell 30
model = RandomForestRegressor(
    n_estimators=300,
    random_state=SEED,
    n_jobs=-1,
    min_samples_leaf=1,
    max_features="sqrt",
)

train_labels_log = np.log1p(train_labels.astype("float64"))
validation_labels_f = validation_labels.astype("float64")
test_labels_f = test_labels.astype("float64")

print(
    "Training RandomForestRegressor on:",
    train_df_scaled.shape,
    "features:",
    list(train_df.columns),
    "(target = log1p(fare_amount))",
)
model.fit(train_df_scaled, train_labels_log)

val_pred_log = model.predict(validation_df_scaled)
val_pred = np.expm1(val_pred_log)
val_pred = np.maximum(val_pred, 0.0)
val_rmse = mean_squared_error(validation_labels_f, val_pred, squared=False)
print("Validation RMSE (in $ after expm1):", val_rmse)



## === cell 31
try:
    print("Model visualization skipped (sklearn model).")
except Exception as e:
    print("Model visualization skipped:", repr(e))



## === cell 32
plot_loss_accuracy_rmse(None)



## === cell 33
if not FAST_MODE:
    train_pred_log = model.predict(train_df_scaled)
    train_pred = np.expm1(train_pred_log)
    train_pred = np.maximum(train_pred, 0.0)
    train_rmse = mean_squared_error(
        train_labels.astype("float64"), train_pred, squared=False
    )
    print("Train RMSE (in $ after expm1):", train_rmse)
else:
    print("Train RMSE skipped in FAST_MODE (no impact on model or submission).")



## === cell 34
if not FAST_MODE:
    val_pred_log2 = model.predict(validation_df_scaled)
    val_pred2 = np.expm1(val_pred_log2)
    val_pred2 = np.maximum(val_pred2, 0.0)
    val_rmse2 = mean_squared_error(validation_labels_f, val_pred2, squared=False)
    print("Validation RMSE (in $ after expm1):", val_rmse2)
else:
    print("Repeated validation RMSE skipped in FAST_MODE (already computed).")



## === cell 35
test_pred_log = model.predict(test_scaled)
test_pred = np.expm1(test_pred_log)
test_pred = np.maximum(test_pred, 0.0)
test_rmse = mean_squared_error(test_labels_f, test_pred, squared=False)
print("Test (holdout from train) RMSE (in $ after expm1):", test_rmse)



## === cell 36
if not FAST_MODE:
    validation_predictions = val_pred.flatten()

    plt.scatter(validation_labels, validation_predictions, s=5, alpha=0.3)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.axis("equal")
    plt.xlim(plt.xlim())
    plt.ylim(plt.ylim())
    _ = plt.plot(
        [validation_predictions.min(), validation_predictions.max()],
        [validation_predictions.min(), validation_predictions.max()],
        "k--",
        lw=2,
    )
    plt.show()



## === cell 37
if not FAST_MODE:
    test_predictions = test_pred.flatten()

    plt.scatter(test_labels, test_predictions, s=5, alpha=0.3)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.axis("equal")
    plt.xlim(plt.xlim())
    plt.ylim(plt.ylim())
    _ = plt.plot(
        [test_predictions.min(), test_predictions.max()],
        [test_predictions.min(), test_predictions.max()],
        "k--",
        lw=2,
    )
    plt.show()



## === cell 38
if not FAST_MODE:
    test_predictions = test_pred.flatten()
    print(np.argmax(test_predictions))
    print(test_predictions[np.argmax(test_predictions)])
    print(test_labels[np.argmax(test_predictions)])
    test_df.iloc[np.argmax(test_predictions)]



## === cell 39
if not FAST_MODE:
    test_predictions = test_pred.flatten()
    print(np.argmin(test_predictions))
    print(test_predictions[np.argmin(test_predictions)])
    print(test_labels[np.argmin(test_predictions)])
    test_df.iloc[np.argmin(test_predictions)]



## === cell 40
if not FAST_MODE:
    test_predictions = test_pred.flatten()
    fig, ax = plt.subplots()
    ax.scatter(test_labels, test_predictions, s=5, alpha=0.3)
    ax.plot(
        [test_labels.min(), test_labels.max()],
        [test_labels.min(), test_labels.max()],
        "k--",
        lw=2,
    )
    ax.set_xlabel("Measured")
    ax.set_ylabel("Predicted")
    plt.show()



## === cell 41
if not FAST_MODE:
    validation_predictions = val_pred.flatten()
    plt.figure(figsize=(20, 10))
    plt.plot(validation_labels[:100])
    plt.plot(validation_predictions[:100])
    plt.title("Prediction vs Actual")
    plt.ylabel("Fare Amount")
    plt.xlabel("Transaction")
    plt.legend(["Actual", "prediction"], loc="upper right")
    plt.show()

    validation_df_scaled[52]
    validation_df.iloc[52]



## === cell 42
if not FAST_MODE:
    test_predictions = test_pred.flatten()
    plt.figure(figsize=(20, 10))
    plt.plot(test_labels[:100])
    plt.plot(test_predictions[:100])
    plt.title("Prediction vs Actual")
    plt.ylabel("Fare Amount")
    plt.xlabel("Transaction")
    plt.legend(["Actual", "prediction"], loc="upper right")
    plt.show()



## === cell 43
if not FAST_MODE:
    validation_predictions = val_pred.flatten()
    error = validation_predictions - validation_labels
    plt.hist(error, bins=100)
    plt.xlabel("Prediction Error")
    _ = plt.ylabel("Count")
    plt.show()



## === cell 44
if not FAST_MODE:
    test_predictions = test_pred.flatten()
    error = test_predictions - test_labels
    plt.hist(error, bins=50)
    plt.xlabel("Prediction Error")
    _ = plt.ylabel("Count")
    plt.show()



## === cell 45
if not FAST_MODE:
    errorGreaterZero = error[(np.logical_or(error <= -1, error >= 1))]
    print(len(error))
    print(len(errorGreaterZero))

    plt.hist(errorGreaterZero, bins=100)
    plt.xlabel("Prediction Error (|error|>=1)")
    _ = plt.ylabel("Count")
    plt.show()



## === cell 46
predictionKaggle_log = model.predict(testKaggle_scaled)
predictionKaggle_cleaned = np.expm1(predictionKaggle_log)
predictionKaggle_cleaned = np.maximum(predictionKaggle_cleaned, 0.0)

fallback = (
    float(np.median(predictionKaggle_cleaned))
    if len(predictionKaggle_cleaned)
    else 11.35
)
pred_map = pd.Series(
    predictionKaggle_cleaned.astype("float32"), index=testKaggle_cleaned["key"].values
)

predictionKaggle_full = testKaggle_raw["key"].map(pred_map).astype("float32")
predictionKaggle_full = predictionKaggle_full.fillna(fallback).values
predictionKaggle_full = np.maximum(predictionKaggle_full, 0.0)

output_submission(
    testKaggle_raw, predictionKaggle_full, "key", "fare_amount", SUBMISSION_NAME
)

sub = pd.read_csv(SUBMISSION_NAME)
print(sub.head())
print("Submission shape:", sub.shape)
print("Columns:", list(sub.columns))
print("Saved submission to:", os.path.abspath(SUBMISSION_NAME))
assert (
    sub.shape[0] == testKaggle_raw.shape[0]
), "Submission rowcount must match test.csv"
assert list(sub.columns) == [
    "key",
    "fare_amount",
], "Submission columns must be exactly: key,fare_amount"
