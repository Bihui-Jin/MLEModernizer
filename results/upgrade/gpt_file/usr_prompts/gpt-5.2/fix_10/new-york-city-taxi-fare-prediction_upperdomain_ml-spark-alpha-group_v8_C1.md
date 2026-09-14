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
import math
import numpy as np
import pandas as pd

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

np.random.seed(42)

INPUT_DIR_CANDIDATES = [
    "/kaggle/input/new-york-city-taxi-fare-prediction",
    "/kaggle/input",
    "../input",
]
INPUT_DIR = None
for d in INPUT_DIR_CANDIDATES:
    if os.path.isdir(d):
        INPUT_DIR = d
        break
if INPUT_DIR is None:
    raise FileNotFoundError(
        "Could not locate Kaggle input directory among known candidates."
    )

print("Using INPUT_DIR:", INPUT_DIR)
print("Top-level INPUT_DIR listing:", os.listdir(INPUT_DIR)[:20])


def _resolve_file(*names):
    """Try resolving a file either directly under INPUT_DIR or at INPUT_DIR/<competition-subdir>/."""
    for name in names:
        p1 = os.path.join(INPUT_DIR, name)
        if os.path.exists(p1):
            return p1
        p2 = os.path.join(INPUT_DIR, "new-york-city-taxi-fare-prediction", name)
        if os.path.exists(p2):
            return p2
    raise FileNotFoundError(f"Could not resolve any of: {names} under {INPUT_DIR}")


TRAIN_PATH = _resolve_file("train.csv")
TEST_PATH = _resolve_file("test.csv")
SAMPLE_SUB_PATH = _resolve_file("sample_submission.csv")



## === cell 1
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
dtype = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "float32",  # will be filtered as numeric; float is fine
}
df = pd.read_csv(TRAIN_PATH, nrows=1_000_000, usecols=usecols, dtype=dtype)
df.head()



## === cell 2
num_cols = [
    "fare_amount",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df[df.passenger_count > 0]
df = df[df.fare_amount > 0]

for c in ["pickup_longitude", "dropoff_longitude"]:
    df = df[df[c].between(-75, -72)]
for c in ["pickup_latitude", "dropoff_latitude"]:
    df = df[df[c].between(40, 42)]

df.head()



## === cell 3
alpha_ang = 0.506


def distance_travel(df_):
    df_["abs_diff_longitude"] = (
        df_.dropoff_longitude - df_.pickup_longitude
    ).abs() * 50
    df_["abs_diff_latitude"] = (df_.dropoff_latitude - df_.pickup_latitude).abs() * 69
    df_["displacement_vector"] = (
        df_.abs_diff_latitude**2 + df_.abs_diff_longitude**2
    ) ** 0.5

    denom = df_.abs_diff_latitude.replace(0, np.nan)
    angle = np.arctan(df_.abs_diff_longitude / denom)

    df_["actual_long"] = (df_.displacement_vector * np.sin(angle - alpha_ang)).abs()
    df_["actual_lat"] = (df_.displacement_vector * np.cos(angle - alpha_ang)).abs()
    df_["distance_travel"] = df_.actual_long + df_.actual_lat


def add_time_features(df_):
    dt = pd.to_datetime(df_["pickup_datetime"], errors="coerce", utc=True)
    df_["pickup_hour"] = dt.dt.hour.astype("float32")
    df_["pickup_weekday"] = dt.dt.weekday.astype("float32")
    df_["pickup_month"] = dt.dt.month.astype("float32")


def add_geo_features(df_):
    lat1 = np.radians(df_["pickup_latitude"].astype("float64"))
    lon1 = np.radians(df_["pickup_longitude"].astype("float64"))
    lat2 = np.radians(df_["dropoff_latitude"].astype("float64"))
    lon2 = np.radians(df_["dropoff_longitude"].astype("float64"))

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    earth_radius_km = 6371.0
    df_["haversine_km"] = (earth_radius_km * c).astype("float32")

    dlon_miles = (df_["dropoff_longitude"] - df_["pickup_longitude"]).abs() * 50.0
    dlat_miles = (df_["dropoff_latitude"] - df_["pickup_latitude"]).abs() * 69.0
    df_["manhattan_miles"] = (dlon_miles + dlat_miles).astype("float32")


def add_bearing_feature(df_):
    lat1 = np.radians(df_["pickup_latitude"].astype("float64"))
    lon1 = np.radians(df_["pickup_longitude"].astype("float64"))
    lat2 = np.radians(df_["dropoff_latitude"].astype("float64"))
    lon2 = np.radians(df_["dropoff_longitude"].astype("float64"))

    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.arctan2(y, x)
    df_["bearing"] = bearing.astype("float32")


def add_interaction_features(df_):
    dt = df_["distance_travel"].astype("float64")
    df_["distance_sq"] = (dt**2).astype("float32")
    df_["dist_x_passengers"] = (dt * df_["passenger_count"].astype("float64")).astype(
        "float32"
    )


distance_travel(df)
add_time_features(df)
add_geo_features(df)
add_bearing_feature(df)

df = df[df.distance_travel > 0]

add_interaction_features(df)

required_cols = [
    "distance_travel",
    "passenger_count",
    "pickup_hour",
    "pickup_weekday",
    "pickup_month",
    "haversine_km",
    "manhattan_miles",
    "bearing",
    "distance_sq",
    "dist_x_passengers",
    "fare_amount",
]
df = df.dropna(subset=required_cols).reset_index(drop=True)

df.head()



## === cell 4
if plt is not None:
    test = df[df.passenger_count == 1]
    _ = test.iloc[: len(test)].plot.scatter("distance_travel", "fare_amount")



## === cell 5
df["distance_travel"] = df["distance_travel"].clip(lower=0, upper=30)

df = df[df.distance_travel < 30]
df = df[df.fare_amount <= 250]

df["fare_amount"] = df["fare_amount"].clip(lower=0, upper=250)

add_interaction_features(df)

if plt is not None:
    _ = df.iloc[:100000].plot.scatter("distance_travel", "fare_amount")



## === cell 6
l = len(df)
print(l)

df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

df_train = df[: int(0.7 * l)]
df_test = df[int(0.7 * l) :]

feature_cols = [
    "distance_travel",
    "passenger_count",
    "pickup_hour",
    "pickup_weekday",
    "pickup_month",
    "haversine_km",
    "manhattan_miles",
    "bearing",
    "distance_sq",
    "dist_x_passengers",
]
train_mat = df_train[feature_cols].to_numpy(dtype=np.float32, copy=False)
test_mat = df_test[feature_cols].to_numpy(dtype=np.float32, copy=False)

train_X = np.concatenate(
    [train_mat, np.ones((train_mat.shape[0], 1), dtype=np.float32)], axis=1
)
test_X = np.concatenate(
    [test_mat, np.ones((test_mat.shape[0], 1), dtype=np.float32)], axis=1
)

train_y = df_train["fare_amount"].to_numpy(dtype=np.float32, copy=False)
test_y = df_test["fare_amount"].to_numpy(dtype=np.float32, copy=False)



## === cell 7
imp = SimpleImputer(missing_values=np.nan, strategy="mean")
train_X_imp = imp.fit_transform(train_X)
test_X_imp = imp.transform(test_X)

regr = GradientBoostingRegressor(
    n_estimators=300,
    random_state=42,
    loss="squared_error",
    learning_rate=0.1,
)
regr.fit(train_X_imp, train_y)

print("Holdout R^2:", regr.score(test_X_imp, test_y))



## === cell 8
t_usecols = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
t_dtype = {
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "float32",
}
tdf = pd.read_csv(TEST_PATH, usecols=t_usecols, dtype=t_dtype)

for c in [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]:
    tdf[c] = pd.to_numeric(tdf[c], errors="coerce")

valid = tdf["passenger_count"] > 0
for c in ["pickup_longitude", "dropoff_longitude"]:
    valid &= tdf[c].between(-75, -72)
for c in ["pickup_latitude", "dropoff_latitude"]:
    valid &= tdf[c].between(40, 42)

distance_travel(tdf)
add_time_features(tdf)
add_geo_features(tdf)
add_bearing_feature(tdf)

tdf["distance_travel"] = tdf["distance_travel"].clip(lower=0, upper=30)

add_interaction_features(tdf)

for col in feature_cols:
    tdf.loc[~valid, col] = np.nan

tdf.head()



## === cell 9
t_mat = tdf[feature_cols].to_numpy(dtype=np.float32, copy=False)
ttrain_X = np.concatenate(
    [t_mat, np.ones((t_mat.shape[0], 1), dtype=np.float32)], axis=1
)

ttrain_X = imp.transform(ttrain_X)
output = regr.predict(ttrain_X)

output = np.clip(output, 0, 250)

print(output[:10], " ... ", "n_preds=", len(output))



## === cell 10
my_submission = pd.DataFrame({"key": tdf.key, "fare_amount": output})
my_submission.to_csv("submission.csv", index=False)
my_submission.head()
