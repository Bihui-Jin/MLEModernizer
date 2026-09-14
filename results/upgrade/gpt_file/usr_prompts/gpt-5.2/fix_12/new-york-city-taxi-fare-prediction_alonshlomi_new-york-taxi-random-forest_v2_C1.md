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

3.12

# 3. Installed packages

geopandas==0.14.4
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

4.069922676663

# 6. Current score

5.88818

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 7.1173) has done: 'Your pipeline already runs and writes `submission.csv`, but it unintentionally *drops rows from the test set* during cleaning, which usually create an invalid submission (row count mismatch) and prevent getting a Kaggle score. I change the cleaning so that we only filter out bad rows in training (where we can), while keeping all test rows and instead clipping out-of-bounds coordinates to the NYC bounding box so every test key gets a prediction. I also align the training-time coordinate filtering and remove the fragile `for df in (train, test)` reassignment pattern to avoid silent mistakes. These are minimal changes that preserve your model/features and should yield a valid submission and a score that can move toward the target.'
- What this solution (achieved 5.88818) has done: 'The timeout is dominated by (1) heavy CSV parsing for 5M rows and (2) training a 300-tree RandomForest on millions of samples. To keep the same model and features while cutting wall time, the main changes are: use faster CSV I/O (pyarrow backend with explicit datetime parsing disabled and minimal columns), avoid expensive pandas operations by filtering/featurizing mostly in NumPy, and enable Intel scikit-learn acceleration (scikit-learn-intelex) for much faster RandomForest fitting/prediction without changing the algorithm. I also reduce temporary allocations/copies and avoid repeated dtype conversions, while keeping the exact same feature set, cleaning rules, and RF hyperparameters. Output paths and submission semantics remain unchanged.'
- What this solution (achieved 5.88818) has done: 'Your current gap is positive (5.88818 vs target 4.0699, lower is better), so we should improve RMSE with the smallest changes that keep your RandomForest + feature set intact. The most leveraged minimal fix here is to add the standard NYC Taxi cleaning that removes “0-coordinate” rows and extreme outliers in longitude/latitude and fare-per-km, because those heavily degrade a tree model when training on a large random slice. I keep your same 5M-row cap, same features (including haversine + time), and same RF hyperparameters/training loop, but tighten the training-only filtering and add a conservative upper bound on trip distance to reduce mislabeled/outlier influence. Submission writing and row alignment remain unchanged.'
- What this solution (achieved 5.88818) has done: 'I fix the crash in `add_time_features_fast` caused by calling `.to_numpy()` on an array returned by `dates.isna()`, ensuring the time features are actually created. That also resolve the downstream `KeyError` for missing `day_of_week/month/hour` columns and let training/inference run end-to-end. I keep your model, features, cleaning, and file paths the same, and only make the minimal changes needed for correctness. The script then write a valid `submission.csv` with all test `key`s aligned to `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

try:
    from sklearnex import patch_sklearn  # scikit-learn-intelex

    patch_sklearn()
except Exception:
    pass

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

BASE_PATH = "/kaggle/input"
TRAIN_PATH = f"{BASE_PATH}/train.csv"
TEST_PATH = f"{BASE_PATH}/test.csv"

print("Listing /kaggle/input:")
print(os.listdir(BASE_PATH))
print("Train exists:", os.path.exists(TRAIN_PATH))
print("Test exists:", os.path.exists(TEST_PATH))

np.random.seed(42)



## === cell 1
NROWS = 5_000_000

usecols_train = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
usecols_test = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

dtypes_train = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16",
    "key": "object",
    "pickup_datetime": "object",
}
dtypes_test = {k: v for k, v in dtypes_train.items() if k != "fare_amount"}

try:
    train = pd.read_csv(
        TRAIN_PATH,
        nrows=NROWS,
        usecols=usecols_train,
        dtype=dtypes_train,
        engine="pyarrow",
    )
    test = pd.read_csv(
        TEST_PATH,
        usecols=usecols_test,
        dtype=dtypes_test,
        engine="pyarrow",
    )
except Exception:
    train = pd.read_csv(
        TRAIN_PATH,
        nrows=NROWS,
        usecols=usecols_train,
        dtype=dtypes_train,
        engine="c",
    )
    test = pd.read_csv(
        TEST_PATH,
        usecols=usecols_test,
        dtype=dtypes_test,
        engine="c",
    )

print(train.shape, test.shape)
train.head()



## === cell 2
train = train.dropna()

fare = train["fare_amount"].to_numpy(copy=False)
pc = train["passenger_count"].to_numpy(copy=False)

LON_MIN, LON_MAX = -75.0, -72.0
LAT_MIN, LAT_MAX = 40.0, 42.0

plon = train["pickup_longitude"].to_numpy(copy=False)
plat = train["pickup_latitude"].to_numpy(copy=False)
dlon = train["dropoff_longitude"].to_numpy(copy=False)
dlat = train["dropoff_latitude"].to_numpy(copy=False)

mask = (
    (fare > 0)
    & (fare < 300)
    & (pc >= 1)
    & (pc <= 6)
    & (plon >= LON_MIN)
    & (plon <= LON_MAX)
    & (dlon >= LON_MIN)
    & (dlon <= LON_MAX)
    & (plat >= LAT_MIN)
    & (plat <= LAT_MAX)
    & (dlat >= LAT_MIN)
    & (dlat <= LAT_MAX)
    & (plon != 0.0)
    & (plat != 0.0)
    & (dlon != 0.0)
    & (dlat != 0.0)
)
train = train.loc[mask]

for col, lo, hi in [
    ("pickup_longitude", LON_MIN, LON_MAX),
    ("dropoff_longitude", LON_MIN, LON_MAX),
    ("pickup_latitude", LAT_MIN, LAT_MAX),
    ("dropoff_latitude", LAT_MIN, LAT_MAX),
]:
    arr = test[col].to_numpy(copy=False)
    np.clip(arr, lo, hi, out=arr)

pc_t = test["passenger_count"].to_numpy(copy=False)
np.clip(pc_t, 1, 6, out=pc_t)

print("After cleaning:", train.shape, test.shape)




## === cell 3
def haversine_distance_np(lat1, lon1, lat2, lon2):
    """
    Great-circle distance (km) between two points given in degrees.
    Accepts array-like; returns float64 array.
    """
    R = 6371.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


def add_time_features_fast(df, dt_col="pickup_datetime"):
    dt = df[dt_col].to_numpy(copy=False)  # object array
    n = len(dt)

    def _safe_int_slice(s, a, b, default):
        try:
            return int(str(s)[a:b])
        except Exception:
            return default

    month = np.fromiter(
        (_safe_int_slice(s, 5, 7, 1) for s in dt), dtype=np.int16, count=n
    ).astype(np.int8, copy=False)
    hour = np.fromiter(
        (_safe_int_slice(s, 11, 13, 0) for s in dt), dtype=np.int16, count=n
    ).astype(np.int8, copy=False)

    date_str = np.fromiter((str(s)[0:10] for s in dt), dtype="U10", count=n)
    dates = pd.to_datetime(date_str, format="%Y-%m-%d", errors="coerce")
    dates64 = dates.to_numpy(dtype="datetime64[ns]", copy=False)

    dow = (dates64.astype("datetime64[D]").view("int64") + 3) % 7  # Mon=0,...Sun=6

    nat_mask = dates.isna()
    if np.any(nat_mask):
        dow = dow.astype(np.int16, copy=True)
        dow[nat_mask] = 0
    dow = dow.astype(np.int8, copy=False)

    df["month"] = month
    df["hour"] = hour
    df["day_of_week"] = dow
    return df




## === cell 4
lat1 = train["pickup_latitude"].to_numpy(dtype=np.float64, copy=False)
lon1 = train["pickup_longitude"].to_numpy(dtype=np.float64, copy=False)
lat2 = train["dropoff_latitude"].to_numpy(dtype=np.float64, copy=False)
lon2 = train["dropoff_longitude"].to_numpy(dtype=np.float64, copy=False)

dist_train = haversine_distance_np(lat1, lon1, lat2, lon2).astype("float32", copy=False)
train["distance"] = dist_train

dist = train["distance"].to_numpy(copy=False)
mask = (dist > 0.05) & (dist < 200.0)
train = train.loc[mask]

dist = train["distance"].to_numpy(copy=False)
fare = train["fare_amount"].to_numpy(copy=False)
fare_per_km = (fare / dist).astype("float32", copy=False)

mask = (fare_per_km > 0.5) & (fare_per_km < 100.0)
train = train.loc[mask]

train = add_time_features_fast(train, "pickup_datetime")

lat1t = test["pickup_latitude"].to_numpy(dtype=np.float64, copy=False)
lon1t = test["pickup_longitude"].to_numpy(dtype=np.float64, copy=False)
lat2t = test["dropoff_latitude"].to_numpy(dtype=np.float64, copy=False)
lon2t = test["dropoff_longitude"].to_numpy(dtype=np.float64, copy=False)

test["distance"] = haversine_distance_np(lat1t, lon1t, lat2t, lon2t).astype(
    "float32", copy=False
)
test = add_time_features_fast(test, "pickup_datetime")

train.head()



## === cell 5
FEATURES = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "distance",
    "day_of_week",
    "month",
    "hour",
]

X = train[FEATURES].to_numpy(copy=False)
if X.dtype != np.float32:
    X = X.astype(np.float32, copy=False)
y = train["fare_amount"].to_numpy(copy=False)
if y.dtype != np.float32:
    y = y.astype(np.float32, copy=False)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

random_forest = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
    warm_start=False,
)

random_forest.fit(X_train, y_train)

y_pred = random_forest.predict(X_val)
rmse = mean_squared_error(y_val, y_pred, squared=False)
print(f"Validation RMSE: {rmse:.6f}")

X_test = test[FEATURES].to_numpy(copy=False)
if X_test.dtype != np.float32:
    X_test = X_test.astype(np.float32, copy=False)

y_pred_test = random_forest.predict(X_test).astype("float32", copy=False)

np.clip(y_pred_test, 0.0, 300.0, out=y_pred_test)

submission = pd.DataFrame({"key": test["key"].astype(str), "fare_amount": y_pred_test})

sample_sub_path = f"{BASE_PATH}/sample_submission.csv"
if os.path.exists(sample_sub_path):
    sample_sub = pd.read_csv(sample_sub_path, usecols=["key"], dtype={"key": "object"})
    sub_map = submission.set_index("key")["fare_amount"]
    aligned = sample_sub.copy()
    aligned["fare_amount"] = aligned["key"].map(sub_map)
    fill_value = float(np.median(y))
    aligned["fare_amount"] = aligned["fare_amount"].fillna(fill_value).astype("float32")
    submission = aligned

submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("File exists:", os.path.exists("submission.csv"))
print("Any NaNs in fare_amount:", submission["fare_amount"].isna().any())
print(
    "Fare min/max:",
    float(submission["fare_amount"].min()),
    float(submission["fare_amount"].max()),
)
