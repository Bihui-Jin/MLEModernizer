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
import os
import numpy as np
import pandas as pd

N_THREADS = os.cpu_count() or 1
os.environ.setdefault("OMP_NUM_THREADS", str(N_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(N_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(N_THREADS))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(N_THREADS))

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

np.random.seed(42)

print(os.listdir("../input"))



## === cell 1
read_csv_kwargs_train = dict(
    filepath_or_buffer="../input/train.csv",
    nrows=5000000,
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
    dtype={
        "key": "object",
        "fare_amount": "float64",
        "pickup_longitude": "float64",
        "pickup_latitude": "float64",
        "dropoff_longitude": "float64",
        "dropoff_latitude": "float64",
        "passenger_count": "int64",
    },
    parse_dates=["pickup_datetime"],
)

read_csv_kwargs_test = dict(
    filepath_or_buffer="../input/test.csv",
    usecols=[
        "key",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
    dtype={
        "key": "object",
        "pickup_longitude": "float64",
        "pickup_latitude": "float64",
        "dropoff_longitude": "float64",
        "dropoff_latitude": "float64",
        "passenger_count": "int64",
    },
    parse_dates=["pickup_datetime"],
)

try:
    train = pd.read_csv(engine="pyarrow", **read_csv_kwargs_train)
    test = pd.read_csv(engine="pyarrow", **read_csv_kwargs_test)
except Exception:
    train = pd.read_csv(**read_csv_kwargs_train)
    test = pd.read_csv(**read_csv_kwargs_test)



## === cell 2
train.shape



## === cell 3
test.shape



## === cell 4
train.head(3)



## === cell 5
_ = None



## === cell 6
_ = None



## === cell 7
pl = train["pickup_latitude"].to_numpy()
dl = train["dropoff_latitude"].to_numpy()
plo = train["pickup_longitude"].to_numpy()
dlo = train["dropoff_longitude"].to_numpy()
fa = train["fare_amount"].to_numpy()
pc_tr = train["passenger_count"].to_numpy()

mask = (
    np.isfinite(pl)
    & np.isfinite(dl)
    & np.isfinite(plo)
    & np.isfinite(dlo)
    & np.isfinite(fa)
    & np.isfinite(pc_tr)
)
mask &= fa >= 0
mask &= pc_tr != 208

mask &= (pl >= -90) & (pl <= 90)
mask &= (dl >= -90) & (dl <= 90)
mask &= (plo >= -180) & (plo <= 180)
mask &= (dlo >= -180) & (dlo <= 180)

nyc_bounds = {"min_lat": 40.5, "max_lat": 41.0, "min_lon": -74.3, "max_lon": -73.6}
mask &= (pl >= nyc_bounds["min_lat"]) & (pl <= nyc_bounds["max_lat"])
mask &= (dl >= nyc_bounds["min_lat"]) & (dl <= nyc_bounds["max_lat"])
mask &= (plo >= nyc_bounds["min_lon"]) & (plo <= nyc_bounds["max_lon"])
mask &= (dlo >= nyc_bounds["min_lon"]) & (dlo <= nyc_bounds["max_lon"])

train = train.loc[mask]

tpl = test["pickup_latitude"].to_numpy()
tdl = test["dropoff_latitude"].to_numpy()
tplo = test["pickup_longitude"].to_numpy()
tdlo = test["dropoff_longitude"].to_numpy()
pc = test["passenger_count"].to_numpy()

tmask = (
    np.isfinite(tpl)
    & np.isfinite(tdl)
    & np.isfinite(tplo)
    & np.isfinite(tdlo)
    & np.isfinite(pc)
)
tmask &= (pc > 0) & (pc <= 8)

tmask &= (tpl >= -90) & (tpl <= 90)
tmask &= (tdl >= -90) & (tdl <= 90)
tmask &= (tplo >= -180) & (tplo <= 180)
tmask &= (tdlo >= -180) & (tdlo <= 180)

tmask &= (tpl >= nyc_bounds["min_lat"]) & (tpl <= nyc_bounds["max_lat"])
tmask &= (tdl >= nyc_bounds["min_lat"]) & (tdl <= nyc_bounds["max_lat"])
tmask &= (tplo >= nyc_bounds["min_lon"]) & (tplo <= nyc_bounds["max_lon"])
tmask &= (tdlo >= nyc_bounds["min_lon"]) & (tdlo <= nyc_bounds["max_lon"])

test = test.loc[tmask]



## === cell 8
train.shape



## === cell 9
_ = None



## === cell 10
_ = None



## === cell 11
train.shape



## === cell 12
_ = None



## === cell 13
_ = None



## === cell 14
_ = None



## === cell 15
_ = None



## === cell 16
_ = None



## === cell 17
_ = None



## === cell 18
_ = None



## === cell 19
_ = None



## === cell 20
_ = None



## === cell 21
train.shape



## === cell 22
_ = None



## === cell 23
_ = None



## === cell 24
_ = None



## === cell 25
_ = None



## === cell 26
_ = None



## === cell 27
_ = None



## === cell 28
_ = None



## === cell 29
_ = None



## === cell 30
_ = None



## === cell 31
train.dtypes



## === cell 32
train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], cache=True, utc=False
)



## === cell 33
test["pickup_datetime"] = pd.to_datetime(test["pickup_datetime"], cache=True, utc=False)



## === cell 34
train.dtypes




## === cell 35
def _add_haversine(df, lat1, long1, lat2, long2, out_col="H_Distance"):
    r = 6371.0
    lat1v = df[lat1].to_numpy()
    lat2v = df[lat2].to_numpy()
    lon1v = df[long1].to_numpy()
    lon2v = df[long2].to_numpy()

    phi1 = np.radians(lat1v)
    phi2 = np.radians(lat2v)
    delta_phi = np.radians(lat2v - lat1v)
    delta_lambda = np.radians(lon2v - lon1v)

    a = (np.sin(delta_phi / 2.0) ** 2) + (
        np.cos(phi1) * np.cos(phi2) * (np.sin(delta_lambda / 2.0) ** 2)
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    out = r * c
    df[out_col] = out
    return out




## === cell 36
_add_haversine(
    train,
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
)
_add_haversine(
    test, "pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"
)



## === cell 37
train["H_Distance"].head(3)



## === cell 38
dt = train["pickup_datetime"].dt
train["Year"] = dt.year
train["Month"] = dt.month
train["Date"] = dt.day
train["Day of Week"] = dt.dayofweek
train["Hour"] = dt.hour

dt = test["pickup_datetime"].dt
test["Year"] = dt.year
test["Month"] = dt.month
test["Date"] = dt.day
test["Day of Week"] = dt.dayofweek
test["Hour"] = dt.hour



## === cell 39
_ = None



## === cell 40
cond = (
    (train["pickup_latitude"].to_numpy() == 0)
    & (train["pickup_longitude"].to_numpy() == 0)
    & ~(train["dropoff_latitude"].to_numpy() == 0)
    & ~(train["dropoff_longitude"].to_numpy() == 0)
    & (train["fare_amount"].to_numpy() == 0)
)
if cond.any():
    train = train.loc[~cond]



## === cell 41
train.shape



## === cell 42
_ = None



## === cell 43
cond = (
    ~(train["pickup_latitude"].to_numpy() == 0)
    & ~(train["pickup_longitude"].to_numpy() == 0)
    & (train["dropoff_latitude"].to_numpy() == 0)
    & (train["dropoff_longitude"].to_numpy() == 0)
    & (train["fare_amount"].to_numpy() == 0)
)
if cond.any():
    train = train.loc[~cond]



## === cell 44
hd = train["H_Distance"].to_numpy()
fa = train["fare_amount"].to_numpy()
high_mask = (hd > 200) & (fa != 0)



## === cell 45
_ = None



## === cell 46
if high_mask.any():
    train.loc[high_mask, "H_Distance"] = (
        train.loc[high_mask, "fare_amount"] - 2.50
    ) / 1.56



## === cell 47
_ = None



## === cell 48
_ = None



## === cell 49
_ = None



## === cell 50
hd = train["H_Distance"].to_numpy()
fa = train["fare_amount"].to_numpy()
zero_zero_mask = (hd == 0) & (fa == 0)
if zero_zero_mask.any():
    train = train.loc[~zero_zero_mask]



## === cell 51
train.loc[train["H_Distance"] == 0].shape



## === cell 52
hour = train["Hour"].to_numpy()
dow = train["Day of Week"].to_numpy()
hd = train["H_Distance"].to_numpy()
fa = train["fare_amount"].to_numpy()

rush_hour_mask = (
    (hour >= 6) & (hour <= 20) & (dow >= 1) & (dow <= 5) & (hd == 0) & (fa < 2.5)
)



## === cell 53
if rush_hour_mask.any():
    train = train.loc[~rush_hour_mask]



## === cell 54
_ = None



## === cell 55
_ = None



## === cell 56
_ = None



## === cell 57
hd = train["H_Distance"].to_numpy()
fa = train["fare_amount"].to_numpy()
scenario3_mask = (hd != 0) & (fa == 0)



## === cell 58
int(scenario3_mask.sum())



## === cell 59
_ = None



## === cell 60
if scenario3_mask.any():
    train.loc[scenario3_mask, "fare_amount"] = (
        train.loc[scenario3_mask, "H_Distance"] * 1.56
    ) + 2.50



## === cell 61
_ = None



## === cell 62
_ = None



## === cell 63
_ = None



## === cell 64
hd = train["H_Distance"].to_numpy()
fa = train["fare_amount"].to_numpy()
scenario4_mask = (hd == 0) & (fa != 0)



## === cell 65
int(scenario4_mask.sum())



## === cell 66
_ = None



## === cell 67
_ = None



## === cell 68
scenario4_sub_mask = scenario4_mask & (fa > 3.0)



## === cell 69
scenario4_sub_mask = scenario4_mask & (fa > 3.0)



## === cell 70
int(scenario4_sub_mask.sum())



## === cell 71
if scenario4_sub_mask.any():
    train.loc[scenario4_sub_mask, "H_Distance"] = (
        train.loc[scenario4_sub_mask, "fare_amount"] - 2.50
    ) / 1.56



## === cell 72
_ = None



## === cell 73
train.columns



## === cell 74
test.columns



## === cell 75
test_key = test["key"].copy()



## === cell 76
train = train.drop(["key", "pickup_datetime"], axis=1)
test = test.drop(["key", "pickup_datetime"], axis=1)



## === cell 77
x_train = train.iloc[:, train.columns != "fare_amount"]
y_train = train["fare_amount"].values
x_test = test



## === cell 78
x_test = x_test.reindex(columns=x_train.columns)



## === cell 79
from sklearn.ensemble import RandomForestRegressor

X_tr = np.ascontiguousarray(x_train.to_numpy(dtype=np.float64, copy=False))
y_tr = np.ascontiguousarray(y_train.astype(np.float64, copy=False))
X_te = np.ascontiguousarray(x_test.to_numpy(dtype=np.float64, copy=False))

rf = RandomForestRegressor(
    n_estimators=300,
    criterion="squared_error",
    n_jobs=-1,
    random_state=42,
)
rf.fit(X_tr, y_tr)
rf_predict = rf.predict(X_te)

rf_predict = np.clip(rf_predict, 0.0, None)

submission = pd.DataFrame({"key": test_key, "fare_amount": rf_predict})
submission.to_csv("submission_1.csv", index=False)
submission.head(5)
