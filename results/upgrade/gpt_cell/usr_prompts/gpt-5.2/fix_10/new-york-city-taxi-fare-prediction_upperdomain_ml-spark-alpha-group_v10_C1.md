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

_READ_DTYPES = {
    "key": "object",
    "fare_amount": "float64",
    "pickup_longitude": "float64",
    "pickup_latitude": "float64",
    "dropoff_longitude": "float64",
    "dropoff_latitude": "float64",
    "passenger_count": "int16",
}


def chunck_generator(filename, header=False, chunk_size=10**5):
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
        filename,
        delimiter=",",
        iterator=True,
        chunksize=chunk_size,
        parse_dates=["pickup_datetime"],
        usecols=usecols,
        dtype=_READ_DTYPES,
        sort=False,
        low_memory=False,
    ):
        yield chunk




## === cell 1
alpha_ang = 0.506


def distance_travel(df):
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs() * 50
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs() * 69
    df["displacement_vector"] = (
        df.abs_diff_latitude**2 + df.abs_diff_longitude**2
    ) ** 0.5

    ang = np.arctan2(df["abs_diff_longitude"].values, df["abs_diff_latitude"].values)
    df["actual_long"] = (
        df["displacement_vector"].values * np.sin(ang - alpha_ang)
    ).astype(np.float64)
    df["actual_lat"] = (
        df["displacement_vector"].values * np.cos(ang - alpha_ang)
    ).astype(np.float64)

    df["actual_long"] = np.abs(df["actual_long"])
    df["actual_lat"] = np.abs(df["actual_lat"])
    df["distance_travel"] = df["actual_long"] + df["actual_lat"]
    return df




## === cell 2
def add_time_features(df):
    dt = df["pickup_datetime"]
    df["pickup_hour"] = dt.dt.hour.astype(np.float64)
    df["pickup_weekday"] = dt.dt.weekday.astype(np.float64)
    df["pickup_month"] = dt.dt.month.astype(np.float64)
    return df


def make_features(df):
    distance_travel(df)
    add_time_features(df)
    return df




## === cell 3
def data_clean(df):
    df = df.loc[df["passenger_count"] > 0]
    df = df.loc[df["fare_amount"] > 0]
    df = df.loc[df["distance_travel"] > 0]
    df.loc[:, "fare_amount"] = df["fare_amount"].astype(np.float64, copy=False)
    return df




## === cell 4
def remove_outliers(df):
    df = df[df.distance_travel < 30]
    df = df[df.fare_amount < 100]
    return df




## === cell 5
def graph_presesnt(df):
    test = df[df.passenger_count == 1]
    plot = test.iloc[: len(test)].plot.scatter("distance_travel", "fare_amount")
    plot = df.iloc[:100000].plot.scatter("distance_travel", "fare_amount")




## === cell 6
def incremental_training(train_X, train_y, regr, is_first_fit=False):
    regr.fit(train_X, train_y)
    return regr




## === cell 7
filename = r"../input/train.csv"
gen = chunck_generator(filename=filename)

regr = GradientBoostingRegressor(n_estimators=20, warm_start=True, random_state=42)

imp = Imputer(missing_values=np.nan, strategy="mean")

rng = np.random.RandomState(42)


def to_X(df):
    return np.column_stack(
        (
            df.distance_travel.values,
            df.passenger_count.values,
            df.pickup_hour.values,
            df.pickup_weekday.values,
            df.pickup_month.values,
            np.ones(len(df)),
        )
    )


imp_fit_rows_target = 300_000
imp_fit_rows = 0
imp_fit_X_parts = []

while imp_fit_rows < imp_fit_rows_target:
    df0 = next(gen)
    make_features(df0)
    df0 = data_clean(df0)
    df0 = remove_outliers(df0)
    if len(df0) == 0:
        continue
    take = min(len(df0), imp_fit_rows_target - imp_fit_rows)
    df0 = df0.iloc[:take]
    X0 = to_X(df0)
    imp_fit_X_parts.append(X0)
    imp_fit_rows += take

imp_fit_X = np.vstack(imp_fit_X_parts)
imp.fit(imp_fit_X)

buffer_rows_max = 600_000

X_ring = None
y_ring = None
ring_size = 0
ring_ptr = 0


def ring_append(X_new, y_new):
    global X_ring, y_ring, ring_size, ring_ptr
    n_new = len(y_new)
    if n_new == 0:
        return

    if X_ring is None:
        X_ring = np.empty((buffer_rows_max, X_new.shape[1]), dtype=np.float64)
        y_ring = np.empty((buffer_rows_max,), dtype=np.float64)

    if n_new >= buffer_rows_max:
        X_new = X_new[-buffer_rows_max:]
        y_new = y_new[-buffer_rows_max:]
        n_new = buffer_rows_max

    end = ring_ptr + n_new
    if end <= buffer_rows_max:
        X_ring[ring_ptr:end, :] = X_new
        y_ring[ring_ptr:end] = y_new
    else:
        first = buffer_rows_max - ring_ptr
        X_ring[ring_ptr:, :] = X_new[:first]
        y_ring[ring_ptr:] = y_new[:first]
        rem = n_new - first
        X_ring[:rem, :] = X_new[first:]
        y_ring[:rem] = y_new[first:]

    ring_ptr = (ring_ptr + n_new) % buffer_rows_max
    ring_size = min(buffer_rows_max, ring_size + n_new)


def ring_view():
    if ring_size == 0:
        return None, None
    if ring_size < buffer_rows_max:
        Xv = X_ring[:ring_size]
        yv = y_ring[:ring_size]
        return Xv, yv
    if ring_ptr == 0:
        return X_ring, y_ring
    Xv = np.vstack((X_ring[ring_ptr:], X_ring[:ring_ptr]))
    yv = np.concatenate((y_ring[ring_ptr:], y_ring[:ring_ptr]))
    return Xv, yv


n_chunks = 80
trees_per_chunk = 10

is_first_fit = True
t = n_chunks
while t > 0:
    df = next(gen)
    make_features(df)
    df = data_clean(df)
    df = remove_outliers(df)

    if len(df) == 0:
        t -= 1
        continue

    df = df.sample(frac=1.0, random_state=rng).reset_index(drop=True)

    X_chunk = to_X(df).astype(np.float64, copy=False)
    y_chunk = df.fare_amount.values.astype(np.float64, copy=False)

    ring_append(X_chunk, y_chunk)

    X_buf, y_buf = ring_view()
    X_buf_imp = imp.transform(X_buf)

    if is_first_fit:
        regr = incremental_training(X_buf_imp, y_buf, regr, is_first_fit=True)
        is_first_fit = False
    else:
        regr.set_params(n_estimators=regr.n_estimators + trees_per_chunk)
        regr = incremental_training(X_buf_imp, y_buf, regr, is_first_fit=False)

    t -= 1



## --- ERROR in cell 7, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mTypeError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3403800519.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     28[0m [0;31m# Speed: compute features once per chunk and avoid redundant make_features() inside data_clean.[0m[0;34m[0m[0;34m[0m[0m
[1;32m     29[0m [0;32mwhile[0m [0mimp_fit_rows[0m [0;34m<[0m [0mimp_fit_rows_target[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 30[0;31m     [0mdf0[0m [0;34m=[0m [0mnext[0m[0;34m([0m[0mgen[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     31[0m     [0mmake_features[0m[0;34m([0m[0mdf0[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     32[0m     [0mdf0[0m [0;34m=[0m [0mdata_clean[0m[0;34m([0m[0mdf0[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/1287585439.py[0m in [0;36mchunck_generator[0;34m(filename, header, chunk_size)[0m
[1;32m     31[0m         [0;34m"passenger_count"[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     32[0m     ]
[0;32m---> 33[0;31m     for chunk in pd.read_csv(
[0m[1;32m     34[0m         [0mfilename[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     35[0m         [0mdelimiter[0m[0;34m=[0m[0;34m","[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;31mTypeError[0m: read_csv() got an unexpected keyword argument 'sort'

## === cell 8
tdf = pd.read_csv(
    "../input/test.csv",
    parse_dates=["pickup_datetime"],
    dtype={
        "key": "object",
        "pickup_longitude": "float64",
        "pickup_latitude": "float64",
        "dropoff_longitude": "float64",
        "dropoff_latitude": "float64",
        "passenger_count": "int16",
    },
    sort=False,
    low_memory=False,
)
make_features(tdf)
tdf.head()
