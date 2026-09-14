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
pyproj==3.7.1
pyproject_hooks==1.2.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
seaborn==0.12.2
sklearn-pandas==2.2.0
ydata-profiling==4.17.0

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

3.53576

# 6. Current score

5.53705

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 5.53705) has done: 'I make the notebook produce a valid `submission.csv` end-to-end (your current code stops after profiling and never trains/predicts). To move RMSE down toward your target while keeping the same core modeling approach (KNN regression), I (1) train KNN on a manageable but stronger training subset, (2) add a minimal, competition-standard geodesic distance feature (plus keep passenger_count) so KNN has a meaningful signal, and (3) apply basic, safe row filtering on impossible coordinates/outliers to reduce noise without changing the overall approach. I also ensure the submission uses the exact `key,fare_amount` schema and matches test row order.'

# 9. Code solution

## === cell 0
import os
import warnings

import numpy as np
import pandas as pd
from pyproj import Geod

from sklearn.neighbors import KNeighborsRegressor as KN_R

warnings.filterwarnings("ignore")
pd.set_option("display.float_format", lambda x: "%.4f" % x)

TRAIN_PATH = "../input/new-york-city-taxi-fare-prediction/train.csv"
TEST_PATH = "../input/new-york-city-taxi-fare-prediction/test.csv"
SAMPLE_SUB_PATH = "../input/new-york-city-taxi-fare-prediction/sample_submission.csv"

SUBMISSION_PATH = "submission.csv"

RANDOM_STATE = 42

N_TRAIN_ROWS = (
    2_000_000  # strong baseline while still feasible in most Kaggle CPU environments
)




## === cell 1

train = pd.read_csv(
    TRAIN_PATH,
    nrows=N_TRAIN_ROWS,
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

test = pd.read_csv(
    TEST_PATH,
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

train.shape, test.shape




## === cell 2
geod = Geod(ellps="WGS84")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    dt = pd.to_datetime(d["pickup_datetime"], errors="coerce", utc=True)
    d["pickup_hour"] = dt.dt.hour.astype("float32")
    d["pickup_dow"] = dt.dt.dayofweek.astype("float32")

    az12, az21, dist_m = geod.inv(
        d["pickup_longitude"].astype(float).values,
        d["pickup_latitude"].astype(float).values,
        d["dropoff_longitude"].astype(float).values,
        d["dropoff_latitude"].astype(float).values,
    )
    d["distance_km"] = (dist_m / 1000.0).astype("float32")

    d["passenger_count"] = pd.to_numeric(d["passenger_count"], errors="coerce").astype(
        "float32"
    )

    return d


train_f = add_features(train)
test_f = add_features(test)

train_f[["distance_km", "pickup_hour", "pickup_dow", "passenger_count"]].describe()




## === cell 3


def clean_train(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    d = d.dropna(
        subset=[
            "fare_amount",
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
            "passenger_count",
            "distance_km",
        ]
    )

    d = d[
        (d["pickup_longitude"].between(-74.5, -72.8))
        & (d["dropoff_longitude"].between(-74.5, -72.8))
        & (d["pickup_latitude"].between(40.0, 41.8))
        & (d["dropoff_latitude"].between(40.0, 41.8))
    ]

    d = d[(d["passenger_count"] >= 1) & (d["passenger_count"] <= 6)]

    d = d[(d["fare_amount"] > 0) & (d["fare_amount"] < 250)]

    d = d[(d["distance_km"] >= 0) & (d["distance_km"] < 200)]

    return d


train_c = clean_train(train_f)

train_c.shape, train_f.shape




## === cell 4

FEATURES = ["distance_km", "passenger_count", "pickup_hour", "pickup_dow"]

X = train_c[FEATURES].astype("float32").values
y = train_c["fare_amount"].astype("float32").values

X_test = test_f[FEATURES].astype("float32").values

mu = np.nanmean(X, axis=0)
sigma = np.nanstd(X, axis=0)
sigma[sigma == 0] = 1.0

Xz = (X - mu) / sigma
X_testz = (X_test - mu) / sigma

knn = KN_R(
    n_neighbors=25,
    weights="distance",
    metric="minkowski",
    p=2,
    n_jobs=-1,
)

knn.fit(Xz, y)

pred = knn.predict(X_testz).astype("float32")

pred = np.clip(pred, 0.0, None)

pred[:5], pred.mean(), pred.min(), pred.max()




## === cell 5
sub = pd.DataFrame({"key": test["key"].values, "fare_amount": pred})

if os.path.exists(SAMPLE_SUB_PATH):
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH, usecols=["key"])
    sub = sample_sub.merge(sub, on="key", how="left")
    fallback = float(np.mean(y)) if len(y) else 11.35
    sub["fare_amount"] = sub["fare_amount"].fillna(fallback)

sub.to_csv(SUBMISSION_PATH, index=False)

sub.head(), sub.shape, SUBMISSION_PATH
