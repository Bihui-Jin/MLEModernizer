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

# 5. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

TRAIN_PATH = "/kaggle/input/train.csv"
TEST_PATH = "/kaggle/input/test.csv"
SAMPLE_SUB_PATH = "/kaggle/input/sample_submission.csv"

PRED_MIN = 0.0
PRED_MAX = 500.0

RANDOM_STATE = 42

nyc_bbox = {
    "lon_min": -74.3,
    "lon_max": -73.7,
    "lat_min": 40.5,
    "lat_max": 41.0,
}



## === cell 1
train_df = pd.read_csv(TRAIN_PATH, nrows=2_000_000)




## === cell 2
def haversine_np(lon1, lat1, lon2, lat2):
    """
    Fast haversine distance in km. All args must be of equal length.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


def add_features(df, nyc_bbox=None, clip_to_bbox=False, train_medians=None):
    df = df.copy()

    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df["year"] = df["pickup_datetime"].dt.year
    df["month"] = df["pickup_datetime"].dt.month
    df["day"] = df["pickup_datetime"].dt.day
    df["hour"] = df["pickup_datetime"].dt.hour
    df["minute"] = df["pickup_datetime"].dt.minute

    if clip_to_bbox and (nyc_bbox is not None):
        for col, lo, hi in [
            ("pickup_longitude", nyc_bbox["lon_min"], nyc_bbox["lon_max"]),
            ("dropoff_longitude", nyc_bbox["lon_min"], nyc_bbox["lon_max"]),
            ("pickup_latitude", nyc_bbox["lat_min"], nyc_bbox["lat_max"]),
            ("dropoff_latitude", nyc_bbox["lat_min"], nyc_bbox["lat_max"]),
        ]:
            df[col] = df[col].clip(lo, hi)

    df["distance"] = haversine_np(
        df["pickup_longitude"],
        df["pickup_latitude"],
        df["dropoff_longitude"],
        df["dropoff_latitude"],
    )
    df["abs_lon_diff"] = (df["pickup_longitude"] - df["dropoff_longitude"]).abs()
    df["abs_lat_diff"] = (df["pickup_latitude"] - df["dropoff_latitude"]).abs()

    mid_lat_rad = np.radians(
        (df["pickup_latitude"].values + df["dropoff_latitude"].values) / 2.0
    )
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(mid_lat_rad)
    df["manhattan_km"] = (
        df["abs_lat_diff"].values * km_per_deg_lat
        + df["abs_lon_diff"].values * km_per_deg_lon
    )

    lon1 = np.radians(df["pickup_longitude"].astype(float).values)
    lat1 = np.radians(df["pickup_latitude"].astype(float).values)
    lon2 = np.radians(df["dropoff_longitude"].astype(float).values)
    lat2 = np.radians(df["dropoff_latitude"].astype(float).values)

    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    df["bearing"] = np.arctan2(y, x)  # radians in [-pi, pi]
    df["haversine_sq"] = df["distance"].astype(float).values ** 2

    if train_medians is not None:
        df = df.fillna(
            {
                "pickup_longitude": train_medians.get("pickup_longitude", 0.0),
                "pickup_latitude": train_medians.get("pickup_latitude", 0.0),
                "dropoff_longitude": train_medians.get("dropoff_longitude", 0.0),
                "dropoff_latitude": train_medians.get("dropoff_latitude", 0.0),
                "year": train_medians["year"],
                "month": train_medians["month"],
                "day": train_medians["day"],
                "hour": train_medians["hour"],
                "minute": train_medians["minute"],
                "distance": 0.0,
                "abs_lon_diff": 0.0,
                "abs_lat_diff": 0.0,
                "manhattan_km": 0.0,
                "bearing": 0.0,
                "haversine_sq": 0.0,
            }
        )

    return df




## === cell 3
train_df = add_features(train_df, nyc_bbox=nyc_bbox, clip_to_bbox=False)



## === cell 4
print("Old size: %d" % len(train_df))
train_df = train_df.dropna(how="any", axis="rows")
print("New size: %d" % len(train_df))



## === cell 5
new_york_lat = 40
new_york_long = -74
train_df.describe()



## === cell 6
cond = np.ones(len(train_df), dtype=bool)

cond &= (train_df["pickup_longitude"].values >= -180.0) & (
    train_df["pickup_longitude"].values <= 180.0
)
cond &= (train_df["dropoff_longitude"].values >= -180.0) & (
    train_df["dropoff_longitude"].values <= 180.0
)
cond &= (train_df["pickup_latitude"].values >= -90.0) & (
    train_df["pickup_latitude"].values <= 90.0
)
cond &= (train_df["dropoff_latitude"].values >= -90.0) & (
    train_df["dropoff_latitude"].values <= 90.0
)

NYC_WIDE = {"lon_min": -75.5, "lon_max": -72.5, "lat_min": 39.0, "lat_max": 42.5}
cond &= (train_df["pickup_longitude"].values >= NYC_WIDE["lon_min"]) & (
    train_df["pickup_longitude"].values <= NYC_WIDE["lon_max"]
)
cond &= (train_df["dropoff_longitude"].values >= NYC_WIDE["lon_min"]) & (
    train_df["dropoff_longitude"].values <= NYC_WIDE["lon_max"]
)
cond &= (train_df["pickup_latitude"].values >= NYC_WIDE["lat_min"]) & (
    train_df["pickup_latitude"].values <= NYC_WIDE["lat_max"]
)
cond &= (train_df["dropoff_latitude"].values >= NYC_WIDE["lat_min"]) & (
    train_df["dropoff_latitude"].values <= NYC_WIDE["lat_max"]
)

cond &= (train_df["fare_amount"].values > 0.0) & (
    train_df["fare_amount"].values < PRED_MAX
)
cond &= (train_df["passenger_count"].values >= 1) & (
    train_df["passenger_count"].values <= 6
)

cond &= (train_df["distance"].values >= 0.0) & (train_df["distance"].values < 150.0)

print("Old size: %d" % len(train_df))
train_df = train_df[cond].copy()
print("New size: %d" % len(train_df))



## === cell 7
train_df.describe()



## === cell 8
train_df["fare_amount"] = train_df["fare_amount"].clip(PRED_MIN, PRED_MAX)

X = train_df[
    [
        "distance",
        "manhattan_km",
        "abs_lon_diff",
        "abs_lat_diff",
        "bearing",
        "haversine_sq",
        "year",
        "month",
        "day",
        "hour",
        "minute",
    ]
].values
Y = np.log1p(train_df["fare_amount"].values)



## === cell 9
from sklearn.ensemble import RandomForestRegressor



## === cell 10
kwargs = {
    "bootstrap": True,
    "oob_score": True,
    "max_depth": None,
    "max_features": 3,
    "min_samples_leaf": 9,
    "min_samples_split": 2,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "criterion": "squared_error",
}
rand_regr = RandomForestRegressor(n_estimators=140, **kwargs)



## === cell 11
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(
    X, Y, test_size=0.2, random_state=RANDOM_STATE
)

rand_regr.fit(X_train, y_train)

y_valid_pred_log = rand_regr.predict(X_valid)
y_valid_pred = np.expm1(y_valid_pred_log)
y_valid_pred = np.clip(y_valid_pred, PRED_MIN, PRED_MAX)
y_valid_true = np.expm1(y_valid)

valid_rmse = np.sqrt(np.mean((y_valid_true - y_valid_pred) ** 2.0))
print("Validation RMSE (holdout):", valid_rmse)
if hasattr(rand_regr, "oob_score_"):
    print("OOB R^2 (train split):", rand_regr.oob_score_)



## === cell 12
rand_regr.score(X_valid, y_valid)



## === cell 13
rand_regr.fit(X, Y)
if hasattr(rand_regr, "oob_score_"):
    print("OOB R^2 (full fit):", rand_regr.oob_score_)



## === cell 14
test_df = pd.read_csv(TEST_PATH)



## === cell 15
train_medians = {
    "year": train_df["year"].median(),
    "month": train_df["month"].median(),
    "day": train_df["day"].median(),
    "hour": train_df["hour"].median(),
    "minute": train_df["minute"].median(),
    "pickup_longitude": train_df["pickup_longitude"].median(),
    "pickup_latitude": train_df["pickup_latitude"].median(),
    "dropoff_longitude": train_df["dropoff_longitude"].median(),
    "dropoff_latitude": train_df["dropoff_latitude"].median(),
}

for c, lo, hi in [
    ("pickup_longitude", -180.0, 180.0),
    ("dropoff_longitude", -180.0, 180.0),
    ("pickup_latitude", -90.0, 90.0),
    ("dropoff_latitude", -90.0, 90.0),
]:
    v = test_df[c].astype(float)
    test_df.loc[(v < lo) | (v > hi), c] = np.nan

test_df = add_features(
    test_df, nyc_bbox=nyc_bbox, clip_to_bbox=False, train_medians=train_medians
)

X_to_pred = test_df[
    [
        "distance",
        "manhattan_km",
        "abs_lon_diff",
        "abs_lat_diff",
        "bearing",
        "haversine_sq",
        "year",
        "month",
        "day",
        "hour",
        "minute",
    ]
].values

y_pred_test_log = rand_regr.predict(X_to_pred)
y_pred_test = np.expm1(y_pred_test_log)
y_pred_test = np.clip(y_pred_test, PRED_MIN, PRED_MAX)



## === cell 16
submission = pd.DataFrame(
    {"key": test_df["key"], "fare_amount": y_pred_test}, columns=["key", "fare_amount"]
)
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)



## === cell 17
X_dist = train_df["distance"].values
Y_fare = np.expm1(Y)  # plot in original fare units for interpretability
mask = np.isfinite(X_dist) & np.isfinite(Y_fare)

with sns.axes_style("white"):
    sns.jointplot(x=X_dist[mask], y=Y_fare[mask], kind="hex", color="k", bins="log")



## === cell 18
X_dist = train_df["distance"].values
Y_fare = np.expm1(Y)
mask = (X_dist < 50) & (Y_fare < 100) & np.isfinite(X_dist) & np.isfinite(Y_fare)

with sns.axes_style("white"):
    p = sns.jointplot(x=X_dist[mask], y=Y_fare[mask], kind="hex", color="k", bins="log")

x_line = np.linspace(0, 50, 200)
year_med = int(train_df["year"].median())
month_med = int(train_df["month"].median())
day_med = int(train_df["day"].median())
hour_med = int(train_df["hour"].median())
minute_med = int(train_df["minute"].median())

abs_lat = np.zeros_like(x_line)
abs_lon = np.zeros_like(x_line)
manhattan = x_line.copy()
bearing_line = np.zeros_like(x_line)
haversine_sq_line = x_line**2

X_line = np.column_stack(
    [
        x_line,  # distance
        manhattan,  # manhattan_km (illustrative)
        abs_lon,  # abs_lon_diff
        abs_lat,  # abs_lat_diff
        bearing_line,  # bearing
        haversine_sq_line,  # haversine_sq
        np.full_like(x_line, year_med),
        np.full_like(x_line, month_med),
        np.full_like(x_line, day_med),
        np.full_like(x_line, hour_med),
        np.full_like(x_line, minute_med),
    ]
)
y_line_log = rand_regr.predict(X_line)
y_line = np.expm1(y_line_log)
y_line = np.clip(y_line, PRED_MIN, PRED_MAX)

p.ax_joint.plot(x_line, y_line, color="red", linewidth=2)
plt.show()
