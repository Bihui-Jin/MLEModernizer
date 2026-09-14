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
import os
import numpy as np
import pandas as pd

try:
    from sklearnex import patch_sklearn

    patch_sklearn()  # Speeds up sklearn algorithms via oneDAL where available; semantics preserved.
except Exception:
    pass

from sklearn.impute import SimpleImputer as Imputer
from sklearn.ensemble import GradientBoostingRegressor

print(os.listdir("../input"))

TRAIN_COLS = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
TEST_COLS = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

DTYPES_TRAIN_FAST = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
DTYPES_TEST_FAST = {
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
    "key": "string",
}

np.random.seed(42)




## === cell 1
def chunck_generator(filename, header=False, chunk_size=10**6):
    usecols = [
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
        usecols=usecols,
        dtype={k: v for k, v in DTYPES_TRAIN_FAST.items() if k != "pickup_datetime"},
        engine="c",
        memory_map=True,
    ):
        yield chunk




## === cell 2
alpha_ang = 0.506


def _distance_travel_from_arrays(pu_lon, pu_lat, do_lon, do_lat):
    abs_diff_long = np.abs(do_lon - pu_lon) * 50.0
    abs_diff_lat = np.abs(do_lat - pu_lat) * 69.0
    disp = np.sqrt(abs_diff_lat * abs_diff_lat + abs_diff_long * abs_diff_long)

    denom = abs_diff_lat.copy()
    denom[denom == 0.0] = np.nan
    angle = np.arctan(abs_diff_long / denom)
    angle = np.nan_to_num(angle, nan=0.0, posinf=0.0, neginf=0.0)

    actual_long = np.abs(disp * np.sin(angle - alpha_ang))
    actual_lat = np.abs(disp * np.cos(angle - alpha_ang))
    return actual_long + actual_lat


def distance_travel(df):
    dist = _distance_travel_from_arrays(
        df["pickup_longitude"].to_numpy(dtype=np.float64, copy=False),
        df["pickup_latitude"].to_numpy(dtype=np.float64, copy=False),
        df["dropoff_longitude"].to_numpy(dtype=np.float64, copy=False),
        df["dropoff_latitude"].to_numpy(dtype=np.float64, copy=False),
    )
    df["distance_travel"] = dist
    return df




## === cell 3
def data_clean(df):
    pc = df["passenger_count"].to_numpy(copy=False)
    mask = pc > 0

    if "fare_amount" in df.columns:
        fa = df["fare_amount"].to_numpy(dtype=np.float64, copy=False)
        mask &= fa > 0.0

    if not mask.all():
        df = df.loc[mask]

    distance_travel(df)

    dist = df["distance_travel"].to_numpy(dtype=np.float64, copy=False)
    df = df.loc[dist > 0.0]
    return df




## === cell 4
def remove_outliers(df):
    dist = df["distance_travel"].to_numpy(dtype=np.float64, copy=False)
    mask = dist < 30.0
    if "fare_amount" in df.columns:
        fa = df["fare_amount"].to_numpy(dtype=np.float64, copy=False)
        mask &= fa < 60.0
    return df.loc[mask]




## === cell 5
def graph_present(df):
    test = df[df.passenger_count == 1]
    plot = test.iloc[: len(test)].plot.scatter("distance_travel", "fare_amount")
    plot = df.iloc[:100000].plot.scatter("distance_travel", "fare_amount")




## === cell 6
def data_preprocessing(df):
    df = data_clean(df)
    df = remove_outliers(df)
    return df




## === cell 7
if False:
    df = pd.read_csv("../input/train.csv", nrows=10_00_000)
    df = df = distance_travel(df)
    df = df[df.passenger_count == 1]
    df = df[df.distance_travel < 30]
    df.distance_travel.hist(bins=50, figsize=(12, 4))
    import matplotlib.pyplot as plt

    plt.xlabel("distance miles")
    plt.title("Histogram ride distances in miles")
    df.distance_travel.describe()



## === cell 8
if False:
    df = pd.read_csv("../input/train.csv", nrows=1_00_000)
    df = df = distance_travel(df)
    df = df[df.passenger_count <= 6]
    df = df[df.distance_travel < 30]
    df = df[df.fare_amount > 0]
    df = df[df.fare_amount < 60]
    df.fare_amount.hist(bins=50, figsize=(12, 4))
    import matplotlib.pyplot as plt

    plt.xlabel("fare_amount")
    df.fare_amount.describe()



## === cell 9
filename = r"../input/train.csv"
gen = chunck_generator(filename=filename)

regr = GradientBoostingRegressor(n_estimators=50, warm_start=True, random_state=42)
imp = Imputer(missing_values=np.nan, strategy="mean")

t = 56
trees_per_chunk = 5

imputer_fitted = False
min_rows_per_chunk_after_clean = 500


def _fast_hour_from_datetime_str(values):
    arr = np.asarray(values)
    s = arr.astype("U32", copy=False)
    hh = np.char.substr(s, 11, 2)
    out = np.empty(hh.shape[0], dtype=np.float64)
    out[:] = np.fromiter(
        (float(x) if x.isdigit() else np.nan for x in hh),
        dtype=np.float64,
        count=hh.shape[0],
    )
    return out


def _build_time_features(pickup_dt_series):
    hour = _fast_hour_from_datetime_str(pickup_dt_series.to_numpy(copy=False))

    dt = pd.to_datetime(pickup_dt_series, errors="coerce", utc=False, cache=True)
    dow = dt.dt.dayofweek.astype("float64").to_numpy(copy=False)
    return hour, dow


def _build_X(dist, pcount, hour, dow):
    n = dist.shape[0]
    X = np.empty((n, 5), dtype=np.float64)
    X[:, 0] = dist
    X[:, 1] = pcount
    X[:, 2] = hour
    X[:, 3] = dow
    X[:, 4] = 1.0
    return X


while t > 0:
    print(t)
    df = next(gen)
    df = data_preprocessing(df)

    if len(df) < min_rows_per_chunk_after_clean:
        print(f"skip_chunk_too_small_after_clean: {len(df)}")
        continue

    dist = df["distance_travel"].to_numpy(dtype=np.float64, copy=False)
    pcount = df["passenger_count"].to_numpy(dtype=np.float64, copy=False)

    hour, dow = _build_time_features(df["pickup_datetime"])

    train_X = _build_X(dist, pcount, hour, dow)
    train_y = df["fare_amount"].to_numpy(dtype=np.float64, copy=False)

    if not imputer_fitted:
        imp = imp.fit(train_X)
        imputer_fitted = True

    train_X = imp.transform(train_X)

    regr.set_params(n_estimators=regr.n_estimators + trees_per_chunk)
    regr = regr.fit(train_X, train_y)

    try:
        print(regr.score(train_X, train_y))
    except Exception as e:
        print("score_failed:", e)

    t -= 1


## --- ERROR in cell 9, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3036229136.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     58[0m     [0mpcount[0m [0;34m=[0m [0mdf[0m[0;34m[[0m[0;34m"passenger_count"[0m[0;34m][0m[0;34m.[0m[0mto_numpy[0m[0;34m([0m[0mdtype[0m[0;34m=[0m[0mnp[0m[0;34m.[0m[0mfloat64[0m[0;34m,[0m [0mcopy[0m[0;34m=[0m[0;32mFalse[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     59[0m [0;34m[0m[0m
[0;32m---> 60[0;31m     [0mhour[0m[0;34m,[0m [0mdow[0m [0;34m=[0m [0m_build_time_features[0m[0;34m([0m[0mdf[0m[0;34m[[0m[0;34m"pickup_datetime"[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     61[0m [0;34m[0m[0m
[1;32m     62[0m     [0mtrain_X[0m [0;34m=[0m [0m_build_X[0m[0;34m([0m[0mdist[0m[0;34m,[0m [0mpcount[0m[0;34m,[0m [0mhour[0m[0;34m,[0m [0mdow[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/3036229136.py[0m in [0;36m_build_time_features[0;34m(pickup_dt_series)[0m
[1;32m     28[0m [0;34m[0m[0m
[1;32m     29[0m [0;32mdef[0m [0m_build_time_features[0m[0;34m([0m[0mpickup_dt_series[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 30[0;31m     [0mhour[0m [0;34m=[0m [0m_fast_hour_from_datetime_str[0m[0;34m([0m[0mpickup_dt_series[0m[0;34m.[0m[0mto_numpy[0m[0;34m([0m[0mcopy[0m[0;34m=[0m[0;32mFalse[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     31[0m [0;34m[0m[0m
[1;32m     32[0m     [0mdt[0m [0;34m=[0m [0mpd[0m[0;34m.[0m[0mto_datetime[0m[0;34m([0m[0mpickup_dt_series[0m[0;34m,[0m [0merrors[0m[0;34m=[0m[0;34m"coerce"[0m[0;34m,[0m [0mutc[0m[0;34m=[0m[0;32mFalse[0m[0;34m,[0m [0mcache[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/3036229136.py[0m in [0;36m_fast_hour_from_datetime_str[0;34m(values)[0m
[1;32m     17[0m     [0;31m# Fix: NumPy 1.26 doesn't provide np.char.str_slice; use np.char.substr(start, length)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     18[0m     [0;31m# to extract hour substring at positions [11:13] of "YYYY-MM-DD HH:MM:SS".[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 19[0;31m     [0mhh[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0mchar[0m[0;34m.[0m[0msubstr[0m[0;34m([0m[0ms[0m[0;34m,[0m [0;36m11[0m[0;34m,[0m [0;36m2[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     20[0m     [0mout[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0mempty[0m[0;34m([0m[0mhh[0m[0;34m.[0m[0mshape[0m[0;34m[[0m[0;36m0[0m[0;34m][0m[0;34m,[0m [0mdtype[0m[0;34m=[0m[0mnp[0m[0;34m.[0m[0mfloat64[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     21[0m     out[:] = np.fromiter(

[0;31mAttributeError[0m: module 'numpy.core.defchararray' has no attribute 'substr'

## === cell 10
test_df = pd.read_csv(
    "../input/test.csv",
    nrows=10_00_000,
    usecols=[
        "key",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
    dtype={k: v for k, v in DTYPES_TEST_FAST.items() if k != "pickup_datetime"},
    engine="c",
    memory_map=True,
)

distance_travel(test_df)
test_df.head()
