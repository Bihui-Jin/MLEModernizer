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

# 5. Code solution

## === cell 0
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import numpy as np
import pandas as pd

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

np.random.seed(42)

TRAIN_PATH = "../input/train.csv"
TEST_PATH = "../input/test.csv"
SAMP_PATH = "../input/sample_submission.csv"

USECOLS_TRAIN = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
USECOLS_TEST = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

train = pd.read_csv(
    TRAIN_PATH,
    nrows=1000000,
    usecols=USECOLS_TRAIN,
    dtype={
        "fare_amount": "float32",
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "int16",
        "key": "object",
        "pickup_datetime": "object",
    },
)
test = pd.read_csv(
    TEST_PATH,
    usecols=USECOLS_TEST,
    dtype={
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "int16",
        "key": "object",
        "pickup_datetime": "object",
    },
)
samp = pd.read_csv(SAMP_PATH)



## === cell 1
fare = train["fare_amount"].to_numpy(copy=False)
plon = train["pickup_longitude"].to_numpy(copy=False)
plat = train["pickup_latitude"].to_numpy(copy=False)
dlon = train["dropoff_longitude"].to_numpy(copy=False)
dlat = train["dropoff_latitude"].to_numpy(copy=False)
pc = train["passenger_count"].to_numpy(copy=False)

dt_notna = train["pickup_datetime"].notna().to_numpy()

mask = (
    dt_notna
    & np.isfinite(fare)
    & np.isfinite(plon)
    & np.isfinite(plat)
    & np.isfinite(dlon)
    & np.isfinite(dlat)
    & np.isfinite(pc.astype(np.float32, copy=False))
    & (fare > 0.0)
    & (fare <= 250.0)
    & (pc >= 1)
    & (pc <= 6)
    & (plon >= -75.0)
    & (plon <= -72.0)
    & (dlon >= -75.0)
    & (dlon <= -72.0)
    & (plat >= 40.0)
    & (plat <= 42.0)
    & (dlat >= 40.0)
    & (dlat <= 42.0)
)
train = train.loc[mask].reset_index(drop=True)



## === cell 2
_ = test.shape



## === cell 3
y = train.fare_amount.values
test_id = test.key




## === cell 4
def week_num(day):
    """
    given the day of the month, return the week number of the month
    """
    if day <= 7:
        return "first"
    if (day > 7) and (day <= 14):
        return "second"
    if (day > 14) and (day <= 21):
        return "third"
    if (day > 21) and (day <= 28):
        return "fourth"
    return "fifth"




## === cell 5
_DOW_NAMES = np.array(
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    dtype=object,
)


def add_time_features(data):
    dt = pd.to_datetime(data["pickup_datetime"], cache=True)

    hour = dt.dt.hour.astype("int16")
    data["hour"] = pd.Categorical(hour.astype(str), ordered=False)

    dow = _DOW_NAMES[dt.dt.dayofweek.to_numpy(dtype=np.int8, copy=False)]
    data["day_of_week"] = pd.Categorical(dow, ordered=False)

    dom = dt.dt.day.to_numpy(dtype=np.int16, copy=False)
    wom = np.empty(dom.shape[0], dtype=object)
    wom[dom <= 7] = "first"
    m = (dom > 7) & (dom <= 14)
    wom[m] = "second"
    m = (dom > 14) & (dom <= 21)
    wom[m] = "third"
    m = (dom > 21) & (dom <= 28)
    wom[m] = "fourth"
    wom[dom > 28] = "fifth"
    data["week_of_month"] = pd.Categorical(
        wom, categories=["first", "second", "third", "fourth", "fifth"], ordered=False
    )

    data["month"] = pd.Categorical(
        dt.dt.month.astype("int16").astype(str), ordered=False
    )
    data["year"] = pd.Categorical(dt.dt.year.astype("int16").astype(str), ordered=False)
    return data




## === cell 6
def add_geo_features(data):
    drop_long = data["dropoff_longitude"].to_numpy(copy=False, dtype=np.float32)
    pick_long = data["pickup_longitude"].to_numpy(copy=False, dtype=np.float32)
    drop_lat = data["dropoff_latitude"].to_numpy(copy=False, dtype=np.float32)
    pick_lat = data["pickup_latitude"].to_numpy(copy=False, dtype=np.float32)

    abs_diff_longitude = np.empty_like(drop_long, dtype=np.float32)
    np.subtract(drop_long, pick_long, out=abs_diff_longitude)
    np.abs(abs_diff_longitude, out=abs_diff_longitude)

    abs_diff_latitude = np.empty_like(drop_lat, dtype=np.float32)
    np.subtract(drop_lat, pick_lat, out=abs_diff_latitude)
    np.abs(abs_diff_latitude, out=abs_diff_latitude)

    data["abs_diff_longitude"] = abs_diff_longitude
    data["abs_diff_latitude"] = abs_diff_latitude

    manhattan_distance = abs_diff_longitude + abs_diff_latitude
    data["manhattan_distance"] = manhattan_distance

    eu = abs_diff_longitude * abs_diff_longitude + abs_diff_latitude * abs_diff_latitude
    np.sqrt(eu, out=eu)
    data["euclid_disance"] = eu
    return data




## === cell 7
train_fe = add_geo_features(add_time_features(train))
test_fe = add_geo_features(add_time_features(test))



## === cell 8
pass



## === cell 9
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

features = [
    "passenger_count",
    "hour",
    "day_of_week",
    "week_of_month",
    "month",
    "year",
    "abs_diff_longitude",
    "abs_diff_latitude",
    "euclid_disance",
    "manhattan_distance",
]

train_x = train_fe[features]
test_x = test_fe[features]

cat_cols = ["hour", "day_of_week", "week_of_month", "month", "year"]
num_cols = [
    "passenger_count",
    "abs_diff_longitude",
    "abs_diff_latitude",
    "euclid_disance",
    "manhattan_distance",
]

ohe = OneHotEncoder(
    handle_unknown="ignore",
    sparse=True,
    min_frequency=50,
    dtype=np.float32,
)

preprocess = ColumnTransformer(
    transformers=[
        ("cat", ohe, cat_cols),
        ("num", "passthrough", num_cols),
    ],
    sparse_threshold=1.0,
)

X = preprocess.fit_transform(train_x)
X_test = preprocess.transform(test_x)



## === cell 10
from scipy import sparse as sp

if sp.issparse(X):
    X = X.tocsr(copy=False)
    if X.dtype != np.float32:
        X = X.astype(np.float32, copy=False)
    X.sort_indices()
else:
    X = np.asarray(X, dtype=np.float32, order="C")

if sp.issparse(X_test):
    X_test = X_test.tocsr(copy=False)
    if X_test.dtype != np.float32:
        X_test = X_test.astype(np.float32, copy=False)
    X_test.sort_indices()
else:
    X_test = np.asarray(X_test, dtype=np.float32, order="C")



## === cell 11
x = X
x_test = X_test



## === cell 12
from sklearn.ensemble import ExtraTreesRegressor

model = ExtraTreesRegressor(
    n_estimators=200,
    n_jobs=-1,
    random_state=42,
    bootstrap=True,
)



## === cell 14
pass



## === cell 15
pass



## === cell 16
y_fit = np.ascontiguousarray(y, dtype=np.float32)

model.fit(x, y_fit)
test_pred = model.predict(x_test).astype(np.float32, copy=False)

np.nan_to_num(test_pred, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
np.maximum(test_pred, 0.0, out=test_pred)



## === cell 17
pass



## === cell 18
sub = pd.DataFrame()
sub["key"] = test_id
sub["fare_amount"] = test_pred
sub.to_csv("submission.csv", index=False)
