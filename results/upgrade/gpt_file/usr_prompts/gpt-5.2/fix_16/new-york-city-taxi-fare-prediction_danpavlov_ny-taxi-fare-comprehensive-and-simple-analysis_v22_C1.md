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
geopy==2.4.1
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
seaborn==0.12.2
sklearn-pandas==2.2.0
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

3.4594

# 6. Current score

5.03925

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 12.46205) has done: 'I fix the XGBoost inference crash by replacing the deprecated `ntree_limit`/missing `best_ntree_limit` usage with the modern `iteration_range` approach, while keeping the same training setup (early stopping still used). I also make the notebook-compatible bits run as a plain Kaggle script by removing the IPython magic and by ensuring the correct `/kaggle/input` paths are used (your current `../input` paths won’t exist here). Finally, I ensure a valid submission CSV is always written (falling back to the linear model submission only if XGBoost fails), with the required columns and `.csv` suffix.'
- What this solution (achieved 8.78615) has done: 'Your current RMSE (12.46) is far above the target (3.46), so we should make small, legitimate improvements that preserve your core pipeline. The biggest issue is that you train on only 100k rows and compute distances with a slow row loop; we can safely increase the training sample (still within time) and vectorize distance so we can afford it, which usually yields a large RMSE drop for this competition without changing model type or loss. We also make the train/valid split slightly more robust by using XGBoost’s built-in `rmse` eval on a validation set (as you already do), and ensure predictions are clipped to a reasonable minimum (fares can’t be negative), which reduces RMSE from occasional negatives. Finally, we write the submission as a `.csv` with the required name and columns.'
- What this solution (achieved 16.73467) has done: 'Your current score (8.786) is much worse than the target (3.459), so we should legitimately improve generalization with the smallest changes that don’t alter your core model/training logic. The biggest gap driver here is label noise/outliers and weak feature scaling for a distance-based fare problem, so we add a standard NYC Taxi cleanup filter (remove extreme fares/distances and zero-distance rides) while keeping the same features and XGBoost setup. We also add a simple, standard “abs deltas” feature (|Δlat|, |Δlon|) and a `log1p(distance)` transform feature; this is still the same feature-extraction approach (tabular engineered features) and typically yields a large RMSE drop without changing the model. Finally, we make train/test feature columns explicitly aligned to avoid silent column-order issues and keep the same submission format/writing.'
- What this solution (achieved 15.88542) has done: 'We need to move RMSE down from 16.73 toward 3.46 (lower is better), so we make the smallest changes that fix likely score-killers without changing your overall “engineered tabular features + XGBoost + early stopping” approach. The main issue is that you’re training on an arbitrary first 1M rows (not representative) and your XGBoost params are extremely underpowered for this competition; we instead sample rows uniformly using `skiprows` (still 1M) and add standard, minimal XGBoost hyperparameters (depth/eta/subsample/colsample/min_child_weight) to improve generalization while keeping the same model family and training loop. We also add a very standard cleanup for obviously bad coordinates (0/0 and out-of-bounds) and ensure test rows with missing datetime are kept (filled with default time features) so submission row count always matches. Finally, we keep your exact feature set but remove forced rounding to 2 decimals (unnecessary quantization that can worsen RMSE) while still clipping negatives.'
- What this solution (achieved 4.33188) has done: 'Your RMSE is far above the target (lower is better), so we should make a small, legitimate improvement that keeps your exact “engineered tabular features + XGBoost + early stopping” core logic. The biggest score-killer is the way the 1M-row sample is drawn: `skiprows=np.random.choice(...)` is extremely memory-heavy and can behave unpredictably; instead we stream the train file in chunks and take a uniform random sample (same size) deterministically, which usually improves representativeness and stability. We also add a standard NYC bounding-box cleanup for the test set (without dropping rows) by setting out-of-range coordinates to NaN and then imputing from medians, preventing extreme distance/airport features from blowing up predictions. Finally, we train XGBoost on a fixed validation split (same split call) but with `evals=[(dtrain,'train'),(dvalid,'valid')]` so early stopping has a consistent signal; prediction and submission format stay unchanged.'
- What this solution (achieved 5.39809) has done: 'Your current RMSE (4.33188) is worse than the target (3.4594), so we should make a small, legitimate improvement without changing the overall “engineered tabular features + XGBoost + early stopping” approach. The biggest likely remaining score-killer is train/test distribution mismatch: the competition’s test set is from 2015, but your training sample is spread across all years; we keep the same features and model but filter the sampled training data to years close to test (2014–2015), which typically reduces RMSE materially here. To keep training size stable after filtering, we slightly oversample during streaming (still bounded/time-safe) and then downsample back to 1,000,000 rows deterministically. Everything else (feature engineering, split, XGBoost params, submission format) stays the same.'
- What this solution (achieved 4.22571) has done: 'The timeout is dominated by two hotspots: the per-row Python loop in the reservoir sampling (millions of `iloc` assignments) and extra overhead from repeated parsing/plotting/correlation on large frames. I keep the same algorithm (uniform random sample of 1,000,000 rows, same feature engineering, same LinearRegression and XGBoost training) but make the sampling truly vectorized per chunk and avoid row-wise updates. I also enable Intel’s scikit-learn acceleration (already installed) and remove heavy plotting/correlation work that doesn’t affect the model outputs. These changes preserve identical training semantics and should bring runtime under 600 seconds.'
- What this solution (achieved 4.30536) has done: 'You’re currently worse than the target (RMSE 4.2257 vs 3.4594, lower is better), so the smallest safe push is to make the validation split match the true test distribution: the test set is entirely 2015, but your split is random across many years, which misguides early stopping and weakens generalization to 2015. I keep the exact same feature engineering and XGBoost training loop, but switch to a time-based split: train on earlier years and validate on 2015 within your sampled/filtered training data. To avoid any subtle train/test distribution mismatch, I also impute missing test coordinates using medians computed from the filtered training set (instead of pre-filter medians). These minimal changes typically reduce RMSE materially for this competition while preserving your core approach and runtime.'
- What this solution (achieved 4.31199) has done: 'Your current RMSE (4.30536) is above the target (3.4594), so we need a modest, legitimate improvement without changing the core “engineered tabular features + XGBoost with early stopping” pipeline. The smallest high-impact issue is that the time-based split uses `year==2015` for validation but trains on *all non-2015 years*, which can include very early years that are distribution-mismatched; we keep the same split idea but restrict training years to a tighter window (2010–2014) while still validating on 2015. We also make airport-distance feature creation fully NumPy (avoid `pd.concat(...).min(axis=1)`), which preserves the same feature semantics but reduces overhead and helps keep runtime stable under the 600s limit. Finally, we add a standard nonnegative distance safety filter after feature creation (drop any residual NaN/inf distances in train only) to reduce noise that hurts RMSE, while keeping the same model and evaluation semantics.'
- What this solution (achieved 4.39282) has done: 'To move RMSE down toward the 3.4594 target (lower is better) without changing your core “engineered tabular features + XGBoost with early stopping” approach, I make two minimal, high-impact adjustments. First, I align the training data distribution to the 2015 test set by filtering the sampled training data to years 2011–2015 (keeping 2015 as validation, 2011–2014 as train) instead of training on earlier, mismatched years that can degrade generalization. Second, I add a single standard feature used in this competition—Haversine distance in kilometers (with R=6371)—while keeping your existing `distance` feature intact; this often yields a noticeable RMSE reduction with negligible logic change. Everything else (sampling, cleaning, split scheme, XGBoost training loop, prediction clipping, and submission writing) stays the same.'
- What this solution (achieved 4.49589) has done: 'We need to reduce RMSE from 4.39282 toward 3.4594 (lower is better), so the smallest high-impact change is to fix a train/test feature mismatch in your distance feature: you compute `distance` with Earth radius 6373 while `hav_distance` uses 6371, which injects slight inconsistency and noise into the model. I unify both to the same radius (6371 km) while keeping the same features, training loop, model type, and early stopping unchanged. I also add one very standard NYC bounding-box filter for *Manhattan-ish* coordinates (tighten to [-74.3..-73.7], [40.5..41.0]) but only on the already-sampled training data to remove remaining noisy rides that tend to hurt RMSE; this is a minimal extension of your existing cleaning (same semantics). Everything else (sampling, split logic, XGBoost params, prediction clipping, and submission writing) stays the same.'
- What this solution (achieved 5.11909) has done: 'Your RMSE (4.49589) is still above the target (3.4594, lower is better), so we make the smallest high-impact changes that keep your same “engineered tabular features + XGBoost with early stopping” pipeline. The biggest remaining score-killer is the overly tight “Manhattan-ish” bounding-box filter on training only, which discards many legitimate NYC-area rides (including airports) and biases the model; we remove that tight filter while keeping your broader NYC sanity bounds. Next, we align the time distribution more closely to the test set by narrowing training years to 2013–2015 (train on 2013–2014, validate on 2015), without changing the split approach itself. Finally, we use `tree_method='hist'` to keep runtime stable while training on the (now less-truncated) sample; predictions/submission formatting stay identical.'
- What this solution (achieved 5.05739) has done: 'To move RMSE down from 5.119 toward the 3.459 target (lower is better) with minimal disruption, I keep your exact feature set and XGBoost training loop but fix one key distribution issue: your test is entirely 2015 while you train on 2013–2014 only, so the model is forced to extrapolate across time. The smallest high-impact change is to train on 2013–2015 and validate on a held-out slice of 2015 (time-based), so early stopping tunes for the real test distribution without leaking. I also align the coordinate imputation medians to the same filtered (2013–2015) training distribution to avoid subtle train/test mismatch. Everything else (sampling, cleaning thresholds, engineered features, XGBoost params, prediction clipping, and submission writing) stays the same.'
- What this solution (achieved 5.03925) has done: 'Your current RMSE (5.057) is still worse than the target (3.459, lower is better), so the smallest likely high-impact fix is to remove a train/test mismatch in how missing datetimes are handled: right now you drop bad datetimes in train but keep them (filled to year=2010) in test, which can create unrealistic time features and hurt predictions. I keep your exact feature set and XGBoost training loop, but (1) impute missing `pickup_datetime` in test using a neutral 2015 timestamp (matching the test distribution) and (2) impute missing time parts from that parsed datetime rather than hardcoding year=2010. Additionally, I add one minimal, standard NYC cleanup filter for training only (remove rides with near-zero coordinate deltas) that reduces label noise without changing model type/architecture; submission format and paths remain unchanged.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from geopy.distance import great_circle  # calculate distances
from sklearn import metrics  # evaluating models
from sklearn.model_selection import train_test_split  # set splitting and validation
from sklearn.linear_model import LinearRegression
import xgboost as xgb  # XGBoost regressor
import matplotlib.pyplot as plt  # plotting
import seaborn as sns  # plotting
from math import sin, cos, sqrt, atan2, radians

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)



## === cell 1
INPUT_DIR = "/kaggle/input"
if not os.path.exists(INPUT_DIR):
    INPUT_DIR = "../input"

print("Input dir:", INPUT_DIR)
print(os.listdir(INPUT_DIR)[:50])



## === cell 2
test_path = os.path.join(INPUT_DIR, "test.csv")
train_path = os.path.join(INPUT_DIR, "train.csv")
sample_sub_path = os.path.join(INPUT_DIR, "sample_submission.csv")

test = pd.read_csv(
    test_path,
    usecols=[
        "key",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
)



## === cell 3
test.dtypes



## === cell 4
types = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}



## === cell 5
TRAIN_NROWS = 1_000_000
CHUNK_SIZE = 250_000  # keeps runtime/memory within Kaggle limits

rng = np.random.RandomState(RANDOM_STATE)

reservoir = None
seen = 0

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
    train_path,
    dtype=types,
    usecols=usecols,
    chunksize=CHUNK_SIZE,
):
    chunk = chunk.dropna()
    n = len(chunk)
    if n == 0:
        continue

    if reservoir is None:
        if n >= TRAIN_NROWS:
            reservoir = chunk.sample(n=TRAIN_NROWS, random_state=rng).reset_index(
                drop=True
            )
            seen = n
        else:
            reservoir = chunk.copy().reset_index(drop=True)
            seen = n
        continue

    if len(reservoir) < TRAIN_NROWS:
        need = TRAIN_NROWS - len(reservoir)
        take = min(need, n)
        if take > 0:
            reservoir = pd.concat(
                [reservoir, chunk.sample(n=take, random_state=rng)],
                ignore_index=True,
            )
        seen += n
        continue

    k = seen + np.arange(1, n + 1, dtype=np.int64)
    u = rng.randint(1, k + 1)  # size=n; high is array => broadcasts elementwise
    mask = u <= TRAIN_NROWS
    if np.any(mask):
        slots = (u[mask] - 1).astype(np.int64)
        rows = np.flatnonzero(mask).astype(np.int64)

        order = np.lexsort((rows, slots))  # sort by slot, then time
        slots_s = slots[order]
        rows_s = rows[order]

        last = np.ones_like(slots_s, dtype=bool)
        last[:-1] = slots_s[:-1] != slots_s[1:]

        slots_final = slots_s[last]
        rows_final = rows_s[last]

        reservoir.iloc[slots_final] = chunk.iloc[rows_final].to_numpy()

    seen += n

train = reservoir.reset_index(drop=True)
print("Loaded uniform train sample:", train.shape)



## === cell 6
train.head()



## === cell 7
train.describe()



## === cell 8
pass



## === cell 9
pass



## === cell 10
train.isnull().sum()



## === cell 11
train.dropna(inplace=True)



## === cell 12
train["pickup_datetime"] = (
    train["pickup_datetime"].astype(str).str.replace(" UTC", "", regex=False)
)
train["pickup_datetime"] = pd.to_datetime(train["pickup_datetime"], errors="coerce")
train = train.dropna(subset=["pickup_datetime"]).copy()

if len(train) > TRAIN_NROWS:
    train = train.sample(n=TRAIN_NROWS, random_state=rng).reset_index(drop=True)

print("After datetime parse & downsample:", train.shape)



## === cell 13
train = train[train["fare_amount"] > 0]
train = train[train["fare_amount"] < 250]  # common filter in this competition
train = train[(train["pickup_longitude"] != 0) & (train["pickup_latitude"] != 0)]
train = train[(train["dropoff_longitude"] != 0) & (train["dropoff_latitude"] != 0)]
train = train[(train["pickup_longitude"] > -80) & (train["pickup_longitude"] < -70)]
train = train[(train["dropoff_longitude"] > -80) & (train["dropoff_longitude"] < -70)]
train = train[(train["pickup_latitude"] > 35) & (train["pickup_latitude"] < 45)]
train = train[(train["dropoff_latitude"] > 35) & (train["dropoff_latitude"] < 45)]
train = train[(train["passenger_count"] > 0) & (train["passenger_count"] < 10)]



## === cell 14
for col, lo, hi in [
    ("pickup_longitude", -80.0, -70.0),
    ("dropoff_longitude", -80.0, -70.0),
    ("pickup_latitude", 35.0, 45.0),
    ("dropoff_latitude", 35.0, 45.0),
]:
    test.loc[(test[col] < lo) | (test[col] > hi) | (test[col] == 0), col] = np.nan

test["passenger_count"] = test["passenger_count"].clip(lower=1, upper=9)



## === cell 15
train.describe()




## === cell 16
def dist_calc(df):
    for i, row in df.iterrows():
        df.at[i, "distance"] = great_circle(
            (row["pickup_latitude"], row["pickup_longitude"]),
            (row["dropoff_latitude"], row["dropoff_longitude"]),
        ).km




## === cell 17
def quick_dist_calc(df):
    R = 6373.0
    if "distance" not in df.columns:
        df["distance"] = np.nan

    for i, row in df.iterrows():
        lat1 = radians(row["pickup_latitude"])
        lon1 = radians(row["pickup_longitude"])
        lat2 = radians(row["dropoff_latitude"])
        lon2 = radians(row["dropoff_longitude"])

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        distance = R * c
        df.at[i, "distance"] = distance




## === cell 18
def quick_dist_calc_vectorized(df):
    R = 6371.0
    lat1 = np.radians(df["pickup_latitude"].astype("float64").values)
    lon1 = np.radians(df["pickup_longitude"].astype("float64").values)
    lat2 = np.radians(df["dropoff_latitude"].astype("float64").values)
    lon2 = np.radians(df["dropoff_longitude"].astype("float64").values)

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    df["distance"] = (R * c).astype("float32")
    return df


coord_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]
train_meds = train.loc[
    train["pickup_datetime"].dt.year.between(2013, 2015), coord_cols
].median(numeric_only=True)
if train_meds.isna().any():
    train_meds = train[coord_cols].median(numeric_only=True)

test[coord_cols] = test[coord_cols].fillna(train_meds)

quick_dist_calc_vectorized(train)
quick_dist_calc_vectorized(test)



## === cell 19
test["pickup_datetime"] = (
    test["pickup_datetime"].astype(str).str.replace(" UTC", "", regex=False)
)
test["pickup_datetime"] = pd.to_datetime(test["pickup_datetime"], errors="coerce")
test["pickup_datetime"] = test["pickup_datetime"].fillna(
    pd.Timestamp("2015-01-01 00:00:00")
)



## === cell 20
train["hour"] = train.pickup_datetime.dt.hour
train["weekday"] = train.pickup_datetime.dt.weekday
train["month"] = train.pickup_datetime.dt.month
train["year"] = train.pickup_datetime.dt.year

test["hour"] = test.pickup_datetime.dt.hour
test["weekday"] = test.pickup_datetime.dt.weekday
test["month"] = test.pickup_datetime.dt.month
test["year"] = test.pickup_datetime.dt.year

test[["hour", "weekday", "month", "year"]] = (
    test[["hour", "weekday", "month", "year"]]
    .fillna({"hour": 0, "weekday": 0, "month": 1, "year": 2015})
    .astype("int16")
)




## === cell 21
def sphere_dist(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    R_earth = 6371
    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = map(
        np.radians, [pickup_lat, pickup_lon, dropoff_lat, dropoff_lon]
    )
    dlat = dropoff_lat - pickup_lat
    dlon = dropoff_lon - pickup_lon

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(pickup_lat) * np.cos(dropoff_lat) * np.sin(dlon / 2.0) ** 2
    )

    return 2 * R_earth * np.arcsin(np.sqrt(a))




## === cell 22
def add_airport_dist(dataset):
    jfk_coord = (40.639722, -73.778889)
    ewr_coord = (40.6925, -74.168611)
    lga_coord = (40.77725, -73.872611)

    pickup_lat = dataset["pickup_latitude"].astype("float64").values
    dropoff_lat = dataset["dropoff_latitude"].astype("float64").values
    pickup_lon = dataset["pickup_longitude"].astype("float64").values
    dropoff_lon = dataset["dropoff_longitude"].astype("float64").values

    pickup_jfk = sphere_dist(pickup_lat, pickup_lon, jfk_coord[0], jfk_coord[1])
    dropoff_jfk = sphere_dist(jfk_coord[0], jfk_coord[1], dropoff_lat, dropoff_lon)
    pickup_ewr = sphere_dist(pickup_lat, pickup_lon, ewr_coord[0], ewr_coord[1])
    dropoff_ewr = sphere_dist(ewr_coord[0], ewr_coord[1], dropoff_lat, dropoff_lon)
    pickup_lga = sphere_dist(pickup_lat, pickup_lon, lga_coord[0], lga_coord[1])
    dropoff_lga = sphere_dist(lga_coord[0], lga_coord[1], dropoff_lat, dropoff_lon)

    dataset["jfk_dist"] = np.minimum(pickup_jfk, dropoff_jfk).astype("float32")
    dataset["ewr_dist"] = np.minimum(pickup_ewr, dropoff_ewr).astype("float32")
    dataset["lga_dist"] = np.minimum(pickup_lga, dropoff_lga).astype("float32")

    return dataset




## === cell 23
train = add_airport_dist(train)
test = add_airport_dist(test)



## === cell 24
train.head()



## === cell 25
test.head()



## === cell 26
pass



## === cell 27
for df in (train, test):
    df["abs_lon_diff"] = (
        (df["pickup_longitude"] - df["dropoff_longitude"]).abs().astype("float32")
    )
    df["abs_lat_diff"] = (
        (df["pickup_latitude"] - df["dropoff_latitude"]).abs().astype("float32")
    )
    df["log1p_distance"] = np.log1p(df["distance"].astype("float64")).astype("float32")

train["hav_distance"] = sphere_dist(
    train["pickup_latitude"].astype("float64").values,
    train["pickup_longitude"].astype("float64").values,
    train["dropoff_latitude"].astype("float64").values,
    train["dropoff_longitude"].astype("float64").values,
).astype("float32")
test["hav_distance"] = sphere_dist(
    test["pickup_latitude"].astype("float64").values,
    test["pickup_longitude"].astype("float64").values,
    test["dropoff_latitude"].astype("float64").values,
    test["dropoff_longitude"].astype("float64").values,
).astype("float32")

train = train.replace([np.inf, -np.inf], np.nan)
train = train.dropna(
    subset=["distance", "jfk_dist", "ewr_dist", "lga_dist", "hav_distance"]
)

train = train[train["distance"] > 0.0]
train = train[train["distance"] < 200.0]

eps = 1e-4
train = train[(train["abs_lon_diff"] > eps) | (train["abs_lat_diff"] > eps)]



## === cell 28
if "year" in train.columns:
    before = len(train)
    train = train[(train["year"] >= 2013) & (train["year"] <= 2015)].copy()
    after = len(train)
    print("Filtered train years to [2013..2015]:", before, "->", after)



## === cell 29
X = train.drop(["key", "fare_amount", "pickup_datetime"], axis=1)
y = train["fare_amount"]



## === cell 30
X.head()



## === cell 31
y.head()



## === cell 32
if "year" in train.columns and (train["year"] == 2015).sum() >= 5000:
    valid_mask_2015 = train["year"] == 2015
    idx_2015 = (
        train.loc[valid_mask_2015].sort_values("pickup_datetime").index.to_numpy()
    )

    split_at = int(len(idx_2015) * 0.8)
    train_2015_idx = idx_2015[:split_at]
    valid_2015_idx = idx_2015[split_at:]

    train_idx = train.index[
        (train["year"] >= 2013) & (train["year"] <= 2015)
    ].to_numpy()
    train_idx = np.setdiff1d(train_idx, valid_2015_idx, assume_unique=False)

    X_train = X.loc[train_idx].reset_index(drop=True)
    y_train = y.loc[train_idx].reset_index(drop=True)
    X_test = X.loc[valid_2015_idx].reset_index(drop=True)
    y_test = y.loc[valid_2015_idx].reset_index(drop=True)

    print(
        "Using time-based split: train=2013–2015 minus last-20%-of-2015, valid=last-20%-of-2015",
        "train:",
        X_train.shape,
        "valid:",
        X_test.shape,
    )
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print("Fallback to random split", "train:", X_train.shape, "valid:", X_test.shape)



## === cell 33
test_pred = test.drop(["key", "pickup_datetime"], axis=1)
test_pred = test_pred.reindex(columns=X.columns)



## === cell 34
lm = LinearRegression()
lm.fit(X_train, y_train)
print(lm.score(X_train, y_train))
print(lm.score(X_test, y_test))



## === cell 35
y_pred = lm.predict(X)
lrmse = np.sqrt(metrics.mean_squared_error(y_pred, y))
lrmse



## === cell 36
LinearPredictions = lm.predict(test_pred)
LinearPredictions = np.clip(LinearPredictions, 0.0, None)
LinearPredictions



## === cell 37
LinearPredictions.size



## === cell 38
linear_submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": LinearPredictions},
    columns=["key", "fare_amount"],
)



## === cell 39
linear_submission.head()




## === cell 40
def XGBoost(X_train, X_test, y_train, y_test):
    X_train_np = np.ascontiguousarray(X_train.to_numpy(dtype=np.float32))
    X_test_np = np.ascontiguousarray(X_test.to_numpy(dtype=np.float32))
    y_train_np = np.ascontiguousarray(y_train.to_numpy(dtype=np.float32))
    y_test_np = np.ascontiguousarray(y_test.to_numpy(dtype=np.float32))

    dtrain = xgb.DMatrix(X_train_np, label=y_train_np)
    dvalid = xgb.DMatrix(X_test_np, label=y_test_np)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "seed": RANDOM_STATE,
        "eta": 0.08,
        "max_depth": 8,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "nthread": int(os.environ.get("OMP_NUM_THREADS", "0")) or os.cpu_count(),
    }
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=2000,
        early_stopping_rounds=50,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        verbose_eval=False,
    )
    return booster




## === cell 41
xgbm = XGBoost(X_train, X_test, y_train, y_test)

test_pred_np = np.ascontiguousarray(test_pred.to_numpy(dtype=np.float32))
dtest_full = xgb.DMatrix(test_pred_np)

best_iter = getattr(xgbm, "best_iteration", None)
if best_iter is None:
    XGBPredictions = xgbm.predict(dtest_full)
else:
    XGBPredictions = xgbm.predict(dtest_full, iteration_range=(0, best_iter + 1))



## === cell 42
XGBPredictions



## === cell 43
XGBPredictions = np.clip(XGBPredictions, 0.0, None)
XGBPredictions



## === cell 44
XGB_submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": XGBPredictions}, columns=["key", "fare_amount"]
)
XGB_submission.head()



## === cell 45
submission = (
    XGB_submission
    if "XGB_submission" in globals() and XGB_submission is not None
    else linear_submission
)

out_path = "submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote submission:", out_path)
print(submission.head())
print("Submission shape:", submission.shape)
assert submission.shape[1] == 2 and list(submission.columns) == ["key", "fare_amount"]
assert submission.shape[0] == test.shape[0]
