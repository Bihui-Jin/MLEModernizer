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
seaborn==0.12.2
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

4.24009

# 6. Current score

6.05115

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.54255) has done: 'I fix the two failing visualization cells by ensuring `X` and `Y` refer to the same training design matrix at the time of plotting (they currently mismatch because `X` is reassigned later for the multi-feature model). I also make the RandomForest reproducible and use a slightly more sensible set of hyperparameters (still the same model/approach) to move RMSE down toward your target without changing the overall pipeline. Additionally, I harden data loading for the Kaggle path layout you showed (`/kaggle/input/...`) so the notebook runs end-to-end and always writes a valid `submission.csv`. All other core feature engineering and modeling steps remain the same.'
- What this solution (achieved 7.19761) has done: 'You’re currently worse than the target RMSE (6.54 vs 4.24), so we should make small, legitimate improvements that keep the same overall approach (feature engineering + scikit-learn regression, no training-loop changes). The biggest gain with minimal risk is to (1) clean obvious label outliers/non-physical values (negative/huge fares, zero-distance rides) and (2) tighten the NYC geographic bounding instead of the very loose “within 5 degrees of mean” filter, which leaves many bad points that hurt RMSE. I keep the same features and RandomForest model family, but adjust a couple of RF hyperparameters slightly (still a RandomForestRegressor) to generalize better. Finally, I apply the same feature columns consistently and clip negative predictions to 0 to avoid invalid fares that worsen RMSE.'
- What this solution (achieved 7.35794) has done: 'You’re currently worse than the target (RMSE 7.20 vs 4.24, lower is better), so we make the smallest legitimate improvements that typically reduce RMSE without changing the overall approach (same feature set + RandomForestRegressor). The main issue is remaining noisy/out-of-distribution training rows; we add a couple of standard NYC Taxi Fare cleaning filters (drop impossible coordinates/zeros, drop extreme fare-per-km outliers, and drop a tiny set of point-to-point “teleport” rides) while keeping the same engineered features and model family. We also fix one small bug in the “quadratic” diagnostic print (it was predicting with the wrong design matrix), and we apply the exact same NYC bounding filter to test (only for safe feature sanity; we still predict for all rows). Submission format and path stay the same and still write `submission.csv`.'
- What this solution (achieved 5.98385) has done: 'The timeout is dominated by fitting a very large `RandomForestRegressor` (500 trees, depth 24) on a multi‑million-row dataset, plus unnecessary extra model fits/predictions that don’t affect the final submission. To stay within 600 seconds without changing the model/training logic, the main speedups are: (1) cache the expensive fitted model to disk and reuse it on subsequent runs, (2) reduce redundant work by removing training-set prediction (only used for a printed metric), and (3) keep the data pipeline fast and memory-efficient with contiguous float32 arrays and early column dropping. These changes preserve the exact model, hyperparameters, features, and predictions used for the submission.'
- What this solution (achieved 6.05115) has done: 'Your current RMSE (5.98385) is worse than the target (4.24009), so we should make small, legitimate changes that typically reduce error without changing the overall pipeline (same features + RandomForestRegressor, same training flow). The biggest likely gain with minimal risk is to make the RandomForest fit a bit more precise by switching the training labels/features from float32 to float64 (RF splitting can be sensitive to quantization at large scale) while keeping all features and hyperparameters the same. To make the cleaned training subset more representative (and reduce noise that inflates RMSE), we also add one standard filter that removes “fare vs distance” inconsistencies using a broad, non-leaky upper bound derived from distance (still the same data cleaning stage). Finally, we keep submission generation identical but ensure dtype consistency between train/test feature matrices.'

# 9. Code solution

## === cell 0
import os
import numpy as np  # linear algebra
import pandas as pd  # CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42


def _resolve_path(*candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


TRAIN_PATH = _resolve_path(
    "/kaggle/input/train.csv",
    "/kaggle/data/train.csv",
    "../input/train.csv",
)
TEST_PATH = _resolve_path(
    "/kaggle/input/test.csv",
    "/kaggle/data/test.csv",
    "../input/test.csv",
)
SAMPLE_SUB_PATH = _resolve_path(
    "/kaggle/input/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
    "../input/sample_submission.csv",
)

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

np.random.seed(RANDOM_STATE)



## === cell 1
read_rows = (
    2_000_000  # identical effective training subset as original after truncation
)
usecols = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
dtypes = {
    "fare_amount": "float32",
    "pickup_datetime": "object",  # parsed later with pd.to_datetime(errors="coerce")
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16",
}

train_df = pd.read_csv(
    TRAIN_PATH,
    nrows=read_rows,
    usecols=usecols,
    dtype=dtypes,
    engine="c",
)




## === cell 2
def haversine_np(lon1, lat1, lon2, lat2):
    """
    https://stackoverflow.com/questions/29545704/fast-haversine-approximation-python-pandas
    Calculate the great circle distance between two points on the earth (specified in decimal degrees)

    All args must be of equal length.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km




## === cell 3
train_df["distance"] = haversine_np(
    train_df["pickup_longitude"].to_numpy(),
    train_df["pickup_latitude"].to_numpy(),
    train_df["dropoff_longitude"].to_numpy(),
    train_df["dropoff_latitude"].to_numpy(),
).astype(np.float32, copy=False)



## === cell 4
train_df["pickup_datetime"] = pd.to_datetime(
    train_df["pickup_datetime"], errors="coerce"
)



## === cell 5
dt = train_df["pickup_datetime"].dt
train_df["year"] = dt.year.astype("int16", copy=False)
train_df["month"] = dt.month.astype("int8", copy=False)
train_df["day"] = dt.day.astype("int8", copy=False)
train_df["hour"] = dt.hour.astype("int8", copy=False)
train_df["minute"] = dt.minute.astype("int8", copy=False)



## === cell 6
print("Old size: %d" % len(train_df))
train_df = train_df.dropna(how="any", axis="rows")
print("New size: %d" % len(train_df))



## === cell 7
new_york_lat = 40
new_york_long = -74




## === cell 8
def _nyc_bbox_filter(df):
    lon_min, lon_max = -74.3, -73.7
    lat_min, lat_max = 40.5, 41.0
    return (
        df["pickup_longitude"].between(lon_min, lon_max)
        & df["dropoff_longitude"].between(lon_min, lon_max)
        & df["pickup_latitude"].between(lat_min, lat_max)
        & df["dropoff_latitude"].between(lat_min, lat_max)
    )


cond = _nyc_bbox_filter(train_df)



## === cell 9
print("Old size: %d" % len(train_df))
train_df = train_df.loc[cond].reset_index(drop=True)
print("New size: %d" % len(train_df))



## === cell 10
print("Old size: %d" % len(train_df))

plon = train_df["pickup_longitude"].to_numpy()
plat = train_df["pickup_latitude"].to_numpy()
dlon = train_df["dropoff_longitude"].to_numpy()
dlat = train_df["dropoff_latitude"].to_numpy()
dist = train_df["distance"].to_numpy()
fare = train_df["fare_amount"].to_numpy()
pc = train_df["passenger_count"].to_numpy()

same_point = (plon == dlon) & (plat == dlat)
has_zero_coord = (plon == 0.0) | (plat == 0.0) | (dlon == 0.0) | (dlat == 0.0)
fare_per_km = fare / (dist + 1e-6)

mask = (
    (fare >= 2.5)
    & (fare <= 250.0)
    & (dist > 0.01)
    & (dist <= 80.0)
    & (pc >= 1)
    & (pc <= 6)
    & (~same_point)
    & (~has_zero_coord)
    & (fare_per_km >= 0.5)
    & (fare_per_km <= 50.0)
)

train_df = train_df.loc[mask].reset_index(drop=True)
print("New size: %d" % len(train_df))



## === cell 11
fare = train_df["fare_amount"].to_numpy()
dist = train_df["distance"].to_numpy()
max_reasonable = (3.0 + 12.0 * dist + 3.0 * np.sqrt(np.maximum(dist, 0.0))).astype(
    np.float32
)
min_reasonable = (2.5 + 0.2 * dist).astype(np.float32)
consistency_mask = (fare >= min_reasonable) & (
    fare <= np.minimum(250.0, max_reasonable)
)
train_df = train_df.loc[consistency_mask].reset_index(drop=True)
print("After fare-distance consistency filter size: %d" % len(train_df))



## === cell 12
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



## === cell 13
regr = LinearRegression()
regr_quad = LinearRegression()



## === cell 14
X = train_df[["distance"]].to_numpy()
Y = train_df["fare_amount"].to_numpy()



## === cell 15
regr.fit(X, Y)



## === cell 16
X2 = X * X
regr_quad.fit(X2, Y)



## === cell 17
y_pred = regr.predict(X)
print("chi squared linear %s" % (np.sum((Y - y_pred) ** 2.0) / len(Y)) ** 0.5)

y_pred_quad = regr_quad.predict(X2)
print("chi squared quadratic %s" % (np.sum((Y - y_pred_quad) ** 2.0) / len(Y)) ** 0.5)



## === cell 18
regr_more = LinearRegression()
X = train_df[["distance", "year", "month", "day", "hour"]].to_numpy()
Y = train_df["fare_amount"].to_numpy()



## === cell 19
regr_more.fit(X, Y)
y_pred = regr_more.predict(X)
print("chi squared linear with date %s" % (np.sum((Y - y_pred) ** 2.0) / len(Y)) ** 0.5)



## === cell 20
from sklearn.ensemble import RandomForestRegressor



## === cell 21
rand_regr = RandomForestRegressor(
    n_estimators=500,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    min_samples_leaf=3,
    min_samples_split=6,
    max_features="sqrt",
    bootstrap=True,
    max_depth=24,
)



## === cell 22
import joblib

MODEL_CACHE_PATH = "rf_model.joblib"

X = np.ascontiguousarray(X, dtype=np.float64)
Y = np.ascontiguousarray(Y, dtype=np.float64)

if os.path.exists(MODEL_CACHE_PATH):
    rand_regr = joblib.load(MODEL_CACHE_PATH)
else:
    rand_regr.fit(X, Y)
    joblib.dump(rand_regr, MODEL_CACHE_PATH, compress=3)



## === cell 23
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
    "pickup_datetime": "object",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16",
}
test_df = pd.read_csv(
    TEST_PATH,
    usecols=test_usecols,
    dtype=test_dtypes,
    engine="c",
)



## === cell 24
test_df["distance"] = haversine_np(
    test_df["pickup_longitude"].to_numpy(),
    test_df["pickup_latitude"].to_numpy(),
    test_df["dropoff_longitude"].to_numpy(),
    test_df["dropoff_latitude"].to_numpy(),
).astype(np.float32, copy=False)



## === cell 25
test_df["pickup_datetime"] = pd.to_datetime(test_df["pickup_datetime"], errors="coerce")



## === cell 26
dt = test_df["pickup_datetime"].dt
test_df["year"] = dt.year.astype("int16", copy=False)
test_df["month"] = dt.month.astype("int8", copy=False)
test_df["day"] = dt.day.astype("int8", copy=False)
test_df["hour"] = dt.hour.astype("int8", copy=False)
test_df["minute"] = dt.minute.astype("int8", copy=False)



## === cell 27
X_to_pred = test_df[["distance", "year", "month", "day", "hour"]].to_numpy()
X_to_pred = np.ascontiguousarray(X_to_pred, dtype=np.float64)
y_pred = rand_regr.predict(X_to_pred)

if np.any(~np.isfinite(y_pred)):
    med = np.nanmedian(y_pred[np.isfinite(y_pred)])
    y_pred[~np.isfinite(y_pred)] = med

y_pred = np.clip(y_pred, 0.0, None)



## === cell 28
submission = pd.DataFrame(
    {"key": test_df.key, "fare_amount": y_pred}, columns=["key", "fare_amount"]
)
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)



## === cell 29
if False:
    X_plot = train_df[["distance"]].values.flatten()
    Y_plot = train_df["fare_amount"].values

    with sns.axes_style("white"):
        sns.jointplot(x=X_plot, y=Y_plot, kind="hex", color="k", bins="log")



## === cell 30
if False:
    X_plot = train_df[["distance"]].values.flatten()
    Y_plot = train_df["fare_amount"].values
    mask = (X_plot < 50) & (Y_plot < 100)

    with sns.axes_style("white"):
        p = sns.jointplot(
            x=X_plot[mask], y=Y_plot[mask], kind="hex", color="k", bins="log"
        )

    x = np.arange(0, 50)
    y = regr.predict(x.reshape(-1, 1))
    p.ax_joint.plot(x, y)
