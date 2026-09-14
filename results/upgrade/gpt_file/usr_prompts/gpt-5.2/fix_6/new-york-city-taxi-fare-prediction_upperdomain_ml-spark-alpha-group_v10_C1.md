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

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer

DATA_DIR_CANDIDATES = [
    "/kaggle/data/new-york-city-taxi-fare-prediction",
    "/kaggle/data",
    "/kaggle/input/new-york-city-taxi-fare-prediction",
    "/kaggle/input",
    "../input",
]
DATA_DIR = None
for _d in DATA_DIR_CANDIDATES:
    if os.path.exists(_d):
        if os.path.exists(os.path.join(_d, "train.csv")) and os.path.exists(
            os.path.join(_d, "test.csv")
        ):
            DATA_DIR = _d
            break
        comp_dir = os.path.join(_d, "new-york-city-taxi-fare-prediction")
        if os.path.exists(os.path.join(comp_dir, "train.csv")) and os.path.exists(
            os.path.join(comp_dir, "test.csv")
        ):
            DATA_DIR = comp_dir
            break

if DATA_DIR is None:
    raise FileNotFoundError(
        "Could not find train.csv/test.csv in any known DATA_DIR candidates: "
        + str(DATA_DIR_CANDIDATES)
    )

print("Using DATA_DIR:", DATA_DIR)


def chunck_generator(filename, chunk_size=10**5):
    for chunk in pd.read_csv(
        filename,
        delimiter=",",
        iterator=True,
        chunksize=chunk_size,
        parse_dates=["pickup_datetime"],
    ):
        yield chunk




## === cell 1
alpha_ang = 0.506


def distance_travel(df):
    df = df.copy()

    for c in [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    mean_lat = ((df["pickup_latitude"] + df["dropoff_latitude"]) / 2.0).astype(float)
    miles_per_deg_lat = 69.0
    miles_per_deg_lon = miles_per_deg_lat * np.cos(np.deg2rad(mean_lat))

    df["abs_diff_longitude"] = (
        df.dropoff_longitude - df.pickup_longitude
    ).abs() * miles_per_deg_lon
    df["abs_diff_latitude"] = (
        df.dropoff_latitude - df.pickup_latitude
    ).abs() * miles_per_deg_lat

    df["displacement_vector"] = (
        df.abs_diff_latitude**2 + df.abs_diff_longitude**2
    ) ** 0.5
    ratio = df.abs_diff_longitude / df.abs_diff_latitude.replace(0, np.nan)
    ang = np.arctan(ratio)
    df["actual_long"] = (df.displacement_vector * np.sin(ang - alpha_ang)).abs()
    df["actual_lat"] = (df.displacement_vector * np.cos(ang - alpha_ang)).abs()
    df["distance_travel"] = df.actual_long + df.actual_lat

    return df


def add_haversine(df):
    df = df.copy()
    R_km = 6371.0

    lat1 = np.deg2rad(df["pickup_latitude"].astype(float))
    lat2 = np.deg2rad(df["dropoff_latitude"].astype(float))
    dlat = lat2 - lat1
    lon1 = np.deg2rad(df["pickup_longitude"].astype(float))
    lon2 = np.deg2rad(df["dropoff_longitude"].astype(float))
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(a))
    df["haversine_km"] = R_km * c
    return df


def add_time_features(df):
    df = df.copy()
    if "pickup_datetime" in df.columns:
        dt = pd.to_datetime(df["pickup_datetime"], errors="coerce")
        df["pickup_hour"] = dt.dt.hour.astype("float64")
        df["pickup_dayofweek"] = dt.dt.dayofweek.astype("float64")
    else:
        df["pickup_hour"] = np.nan
        df["pickup_dayofweek"] = np.nan
    return df




## === cell 2
def data_clean(df):
    df = df.copy()

    if "passenger_count" in df.columns:
        df = df[df.passenger_count > 0]

    if "fare_amount" in df.columns:
        df["fare_amount"] = pd.to_numeric(df["fare_amount"], errors="coerce").astype(
            np.float64
        )
        df = df[df.fare_amount > 0]

    if "distance_travel" in df.columns:
        df = df[df.distance_travel > 0]
        df = df.dropna(subset=["distance_travel"])

    if "haversine_km" in df.columns:
        df = df[df.haversine_km >= 0]
        df = df.dropna(subset=["haversine_km"])

    return df


def data_clean_no_filter(df):
    df = df.copy()

    if "passenger_count" in df.columns:
        df["passenger_count"] = pd.to_numeric(df["passenger_count"], errors="coerce")

    for c in [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "distance_travel" in df.columns:
        df["distance_travel"] = pd.to_numeric(df["distance_travel"], errors="coerce")
        df.loc[~np.isfinite(df["distance_travel"].values), "distance_travel"] = np.nan

    if "haversine_km" in df.columns:
        df["haversine_km"] = pd.to_numeric(df["haversine_km"], errors="coerce")
        df.loc[~np.isfinite(df["haversine_km"].values), "haversine_km"] = np.nan

    for c in ["pickup_hour", "pickup_dayofweek"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df




## === cell 3
def remove_outliers(df):
    df = df.copy()
    if "distance_travel" in df.columns:
        df = df[df.distance_travel < 30]
    if "haversine_km" in df.columns:
        df = df[df.haversine_km < 100]
    if "fare_amount" in df.columns:
        df = df[df.fare_amount < 100]

    for c in ["pickup_longitude", "dropoff_longitude"]:
        if c in df.columns:
            df = df[(df[c] >= -75) & (df[c] <= -72)]
    for c in ["pickup_latitude", "dropoff_latitude"]:
        if c in df.columns:
            df = df[(df[c] >= 40) & (df[c] <= 42)]

    return df


def remove_outliers_no_filter(df):
    df = df.copy()
    if "distance_travel" in df.columns:
        df.loc[df["distance_travel"] >= 30, "distance_travel"] = np.nan
    if "haversine_km" in df.columns:
        df.loc[df["haversine_km"] >= 100, "haversine_km"] = np.nan
    if "passenger_count" in df.columns:
        df.loc[df["passenger_count"] <= 0, "passenger_count"] = np.nan

    for c in ["pickup_longitude", "dropoff_longitude"]:
        if c in df.columns:
            bad = ~df[c].between(-75, -72)
            df.loc[bad, c] = np.nan
    for c in ["pickup_latitude", "dropoff_latitude"]:
        if c in df.columns:
            bad = ~df[c].between(40, 42)
            df.loc[bad, c] = np.nan

    return df




## === cell 4
def incremental_training(train_X, train_y, regr):
    regr.fit(train_X, train_y)
    return regr




## === cell 5
def build_features(df):
    return np.column_stack(
        (
            df["distance_travel"].values,
            df["haversine_km"].values,
            df["passenger_count"].values,
            df["pickup_hour"].values,
            df["pickup_dayofweek"].values,
            np.ones(len(df)),
        )
    ).astype(np.float64)




## === cell 6
train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")

assert os.path.exists(train_path), f"train.csv not found at {train_path}"
assert os.path.exists(test_path), f"test.csv not found at {test_path}"
assert os.path.exists(sample_path), f"sample_submission.csv not found at {sample_path}"

gen = chunck_generator(filename=train_path, chunk_size=10**5)

regr = GradientBoostingRegressor(n_estimators=100, warm_start=True, random_state=42)
imp = SimpleImputer(strategy="mean")

t = 100  # number of chunks to train on

imputer_fit_X = []

for i in range(t):
    df = next(gen)

    df = df.sample(frac=1.0, random_state=42 + i).reset_index(drop=True)

    df = distance_travel(df)
    df = add_haversine(df)
    df = add_time_features(df)
    df = data_clean(df)
    df = remove_outliers(df)

    if len(df) < 10:
        continue

    l = len(df)
    split = int(0.7 * l)

    df_train = df.iloc[:split]
    df_test = df.iloc[split:]  # unused but kept to preserve original semantics

    train_X = build_features(df_train)
    train_y = df_train.fare_amount.values.astype(np.float64)

    imputer_fit_X.append(train_X[:2000])

if len(imputer_fit_X) == 0:
    raise RuntimeError("No usable training data after cleaning/outlier removal.")
imp = imp.fit(np.vstack(imputer_fit_X))

gen = chunck_generator(filename=train_path, chunk_size=10**5)

effective_chunk = 0
for i in range(t):
    df = next(gen)

    df = df.sample(frac=1.0, random_state=42 + i).reset_index(drop=True)

    df = distance_travel(df)
    df = add_haversine(df)
    df = add_time_features(df)
    df = data_clean(df)
    df = remove_outliers(df)

    if len(df) < 10:
        continue

    l = len(df)
    split = int(0.7 * l)

    df_train = df.iloc[:split]
    df_test = df.iloc[split:]  # unused but kept to preserve original semantics

    train_X = build_features(df_train)
    train_y = df_train.fare_amount.values.astype(np.float64)

    train_X = imp.transform(train_X)

    effective_chunk += 1
    regr.set_params(n_estimators=100 * effective_chunk)
    regr = incremental_training(train_X, train_y, regr)

print("Training complete. Final n_estimators:", regr.n_estimators)



## === cell 7
tdf_raw = pd.read_csv(test_path, parse_dates=["pickup_datetime"])
test_keys = tdf_raw["key"].values

tdf = distance_travel(tdf_raw)
tdf = add_haversine(tdf)
tdf = add_time_features(tdf)
tdf = data_clean_no_filter(tdf)
tdf = remove_outliers_no_filter(tdf)

ttrain_X = build_features(tdf)
ttrain_X = imp.transform(ttrain_X)
output = regr.predict(ttrain_X)

output = np.asarray(output, dtype=np.float64)
output = np.clip(output, 0, None)

print(
    "Pred stats:", float(np.min(output)), float(np.mean(output)), float(np.max(output))
)



## === cell 8
my_submission = pd.DataFrame({"key": test_keys, "fare_amount": output})
my_submission = my_submission[["key", "fare_amount"]]
my_submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", my_submission.shape)
print(my_submission.head())
