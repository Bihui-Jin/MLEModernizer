# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.7

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

## === cell 0
from sklearn.impute import SimpleImputer as Imputer
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import os

print(os.listdir("../input"))

TRAIN_PATH = r"../input/train.csv"
TEST_PATH = r"../input/test.csv"

TRAIN_USECOLS = [
    "fare_amount",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
TEST_USECOLS = [
    "key",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

DTYPES_TRAIN = {
    "fare_amount": "float64",  # keep float64
    "pickup_longitude": "float64",
    "pickup_latitude": "float64",
    "dropoff_longitude": "float64",
    "dropoff_latitude": "float64",
    "passenger_count": "int8",
}
DTYPES_TEST = {
    "key": "object",
    "pickup_longitude": "float64",
    "pickup_latitude": "float64",
    "dropoff_longitude": "float64",
    "dropoff_latitude": "float64",
    "passenger_count": "int8",
}


def chunck_generator(filename, chunk_size=10**6, usecols=None, dtypes=None):
    for chunk in pd.read_csv(
        filename,
        delimiter=",",
        iterator=True,
        chunksize=chunk_size,
        usecols=usecols,
        dtype=dtypes,
        engine="c",
        skipinitialspace=True,
    ):
        yield chunk




## === cell 1
alpha_ang = 0.506


def distance_travel(df):
    plon = df["pickup_longitude"].to_numpy(dtype=np.float64, copy=False)
    plat = df["pickup_latitude"].to_numpy(dtype=np.float64, copy=False)
    dlon = df["dropoff_longitude"].to_numpy(dtype=np.float64, copy=False)
    dlat = df["dropoff_latitude"].to_numpy(dtype=np.float64, copy=False)

    abs_diff_longitude = np.abs(dlon - plon) * 50.0
    abs_diff_latitude = np.abs(dlat - plat) * 69.0

    displacement_vector = np.sqrt(
        abs_diff_latitude * abs_diff_latitude + abs_diff_longitude * abs_diff_longitude
    )

    ratio = np.empty_like(abs_diff_longitude)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(abs_diff_longitude, abs_diff_latitude, out=ratio)

    angle = np.arctan(ratio) - alpha_ang

    actual_long = np.abs(displacement_vector * np.sin(angle))
    actual_lat = np.abs(displacement_vector * np.cos(angle))

    df["abs_diff_longitude"] = abs_diff_longitude
    df["abs_diff_latitude"] = abs_diff_latitude
    df["displacement_vector"] = displacement_vector
    df["actual_long"] = actual_long
    df["actual_lat"] = actual_lat
    df["distance_travel"] = actual_long + actual_lat
    return df




## === cell 2
def data_clean(df):
    df["fare_amount"] = df["fare_amount"].astype(np.float64)
    m = (
        (df["passenger_count"] > 0)
        & (df["fare_amount"] > 0)
        & (df["distance_travel"] > 0)
    )
    return df[m]


def remove_outliers(df):
    m = (df["distance_travel"] < 30) & (df["fare_amount"] < 100)
    return df[m]




## === cell 3
def incremental_training(train_X, train_y, regr):
    regr.fit(train_X, train_y)
    return regr




## === cell 4
CHUNK_SIZE = 1_000_000
N_CHUNKS = 56  # same as original t=56
RANDOM_STATE = 42

FINAL_N_ESTIMATORS = 20 + 20 * N_CHUNKS
regr = GradientBoostingRegressor(
    n_estimators=FINAL_N_ESTIMATORS, warm_start=True, random_state=RANDOM_STATE
)

sum_ = np.zeros(3, dtype=np.float64)
count_ = np.zeros(3, dtype=np.int64)

gen = chunck_generator(
    filename=TRAIN_PATH,
    chunk_size=CHUNK_SIZE,
    usecols=TRAIN_USECOLS,
    dtypes=DTYPES_TRAIN,
)

t = N_CHUNKS
while t > 0:
    print(f"imputer pass, chunks left: {t}")
    try:
        df = next(gen)
    except StopIteration:
        break

    df = distance_travel(df)
    df = data_clean(df)
    df = remove_outliers(df)

    if len(df) == 0:
        t -= 1
        continue

    X0 = df["distance_travel"].to_numpy(dtype=np.float64, copy=False)
    X1 = df["passenger_count"].to_numpy(copy=False).astype(np.float64, copy=False)

    m0 = ~np.isnan(X0)
    m1 = ~np.isnan(X1)

    if m0.any():
        sum_[0] += X0[m0].sum(dtype=np.float64)
        count_[0] += int(m0.sum())
    if m1.any():
        sum_[1] += X1[m1].sum(dtype=np.float64)
        count_[1] += int(m1.sum())

    sum_[2] += float(len(df)) * 1.0
    count_[2] += int(len(df))

    t -= 1

means = sum_ / count_
imp = Imputer(missing_values=np.nan, strategy="mean")
imp.fit(np.zeros((1, 3), dtype=np.float64))
imp.statistics_ = means

gen2 = chunck_generator(
    filename=TRAIN_PATH,
    chunk_size=CHUNK_SIZE,
    usecols=TRAIN_USECOLS,
    dtypes=DTYPES_TRAIN,
)

t = N_CHUNKS
trained_chunks = 0
prev_stage_pred = None  # raw prediction from previous stage for current chunk rows

while t > 0:
    print(f"train pass, chunks left: {t}")
    try:
        df = next(gen2)
    except StopIteration:
        break

    df = distance_travel(df)
    df = data_clean(df)
    df = remove_outliers(df)

    n = len(df)
    if n == 0:
        t -= 1
        prev_stage_pred = None
        continue

    X0 = df["distance_travel"].to_numpy(dtype=np.float64, copy=False)
    X1 = df["passenger_count"].to_numpy(copy=False).astype(np.float64, copy=False)
    y = df["fare_amount"].to_numpy(dtype=np.float64, copy=False)

    X = np.empty((n, 3), dtype=np.float64)
    X[:, 0] = X0
    X[:, 1] = X1
    X[:, 2] = 1.0
    X = imp.transform(X)

    trained_chunks += 1

    regr.set_params(n_estimators=20, warm_start=False)
    if prev_stage_pred is None:
        regr.set_params(init=None)
    else:
        regr.set_params(init=prev_stage_pred)

    regr = incremental_training(X, y, regr)

    prev_stage_pred = next(regr.staged_predict(X))

    t -= 1



## --- ERROR in cell 4, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mInvalidParameterError[0m                     Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1235441827.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m    115[0m         [0mregr[0m[0;34m.[0m[0mset_params[0m[0;34m([0m[0minit[0m[0;34m=[0m[0mprev_stage_pred[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    116[0m [0;34m[0m[0m
[0;32m--> 117[0;31m     [0mregr[0m [0;34m=[0m [0mincremental_training[0m[0;34m([0m[0mX[0m[0;34m,[0m [0my[0m[0;34m,[0m [0mregr[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    118[0m [0;34m[0m[0m
[1;32m    119[0m     [0;31m# Update prev_stage_pred for next chunk: prediction of model up to this stage.[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/3896938299.py[0m in [0;36mincremental_training[0;34m(train_X, train_y, regr)[0m
[1;32m      1[0m [0;32mdef[0m [0mincremental_training[0m[0;34m([0m[0mtrain_X[0m[0;34m,[0m [0mtrain_y[0m[0;34m,[0m [0mregr[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 2[0;31m     [0mregr[0m[0;34m.[0m[0mfit[0m[0;34m([0m[0mtrain_X[0m[0;34m,[0m [0mtrain_y[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      3[0m     [0;32mreturn[0m [0mregr[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0;34m[0m[0m
[1;32m      5[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_gb.py[0m in [0;36mfit[0;34m(self, X, y, sample_weight, monitor)[0m
[1;32m    418[0m             [0mFitted[0m [0mestimator[0m[0;34m.[0m[0;34m[0m[0;34m[0m[0m
[1;32m    419[0m         """
[0;32m--> 420[0;31m         [0mself[0m[0;34m.[0m[0m_validate_params[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    421[0m [0;34m[0m[0m
[1;32m    422[0m         [0;32mif[0m [0;32mnot[0m [0mself[0m[0;34m.[0m[0mwarm_start[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/base.py[0m in [0;36m_validate_params[0;34m(self)[0m
[1;32m    598[0m         [0maccepted[0m [0mconstraints[0m[0;34m.[0m[0;34m[0m[0;34m[0m[0m
[1;32m    599[0m         """
[0;32m--> 600[0;31m         validate_parameter_constraints(
[0m[1;32m    601[0m             [0mself[0m[0;34m.[0m[0m_parameter_constraints[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    602[0m             [0mself[0m[0;34m.[0m[0mget_params[0m[0;34m([0m[0mdeep[0m[0;34m=[0m[0;32mFalse[0m[0;34m)[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/utils/_param_validation.py[0m in [0;36mvalidate_parameter_constraints[0;34m(parameter_constraints, params, caller_name)[0m
[1;32m     95[0m                 )
[1;32m     96[0m [0;34m[0m[0m
[0;32m---> 97[0;31m             raise InvalidParameterError(
[0m[1;32m     98[0m                 [0;34mf"The {param_name!r} parameter of {caller_name} must be"[0m[0;34m[0m[0;34m[0m[0m
[1;32m     99[0m                 [0;34mf" {constraints_str}. Got {param_val!r} instead."[0m[0;34m[0m[0;34m[0m[0m

[0;31mInvalidParameterError[0m: The 'init' parameter of GradientBoostingRegressor must be a str among {'zero'}, None or an object implementing 'fit' and 'predict'. Got array([10.7972427 , 11.98201468, 10.7972427 , ..., 10.7972427 ,
       11.09372417, 10.7972427 ]) instead.

## === cell 5
tdf = pd.read_csv(
    TEST_PATH,
    usecols=TEST_USECOLS,
    dtype=DTYPES_TEST,
    engine="c",
    skipinitialspace=True,
)
distance_travel(tdf)
tdf.head()
