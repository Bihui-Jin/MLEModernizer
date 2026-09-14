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

3.10

# 2. Installed packages

geopandas==0.14.4
geopy==2.4.1
lightgbm==4.6.0
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

os.environ["PYTHONHASHSEED"] = "0"

INPUT_DIR = "/kaggle/input/new-york-city-taxi-fare-prediction"
WORKING_DIR = "/kaggle/working"

print("Listing /kaggle/input (truncated):")
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))
    break



## === cell 1
NROWS = 2_000_000
CHUNKSIZE = 250_000

TRAIN_COLS = [
    "key",
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

DTYPES_TRAIN = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
    "key": "string",
}
DTYPES_TEST = {
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
    "key": "string",
}

train_path = os.path.join(INPUT_DIR, "train.csv")

train_chunks = []
read_rows = 0
for chunk in pd.read_csv(
    train_path,
    usecols=TRAIN_COLS,
    dtype=DTYPES_TRAIN,
    parse_dates=["pickup_datetime"],
    chunksize=CHUNKSIZE,
):
    need = NROWS - read_rows
    if need <= 0:
        break
    if len(chunk) > need:
        chunk = chunk.iloc[:need].copy()
    train_chunks.append(chunk)
    read_rows += len(chunk)
    if read_rows >= NROWS:
        break

train = pd.concat(train_chunks, ignore_index=True)

test = pd.read_csv(
    os.path.join(INPUT_DIR, "test.csv"),
    usecols=TEST_COLS,
    dtype=DTYPES_TEST,
    parse_dates=["pickup_datetime"],
)
sample_submission = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))

print(train.shape, test.shape, sample_submission.shape)



## === cell 2
train.isnull().sum()



## === cell 3
train.dropna(inplace=True)



## === cell 4
train.describe()



## === cell 5
train.query("passenger_count > 6")



## === cell 6
train.query("passenger_count < 1")



## === cell 7
train.query("fare_amount < 0")



## === cell 8
train.query("pickup_longitude < -180 or pickup_longitude > 180")



## === cell 9
train.query("dropoff_longitude < -180 or dropoff_longitude > 180")



## === cell 10
train.query("pickup_latitude < -90 or pickup_latitude > 90")



## === cell 11
train.query("dropoff_latitude < -90 or dropoff_latitude > 90")



## === cell 12
plon = train["pickup_longitude"]
plat = train["pickup_latitude"]
dlon = train["dropoff_longitude"]
dlat = train["dropoff_latitude"]
pc = train["passenger_count"]
fare = train["fare_amount"]

mask = (
    pc.between(1, 6)
    & fare.between(0, 250)
    & plon.between(-180, 180)
    & dlon.between(-180, 180)
    & plat.between(-90, 90)
    & dlat.between(-90, 90)
    & plon.between(-74.5, -72.8)
    & dlon.between(-74.5, -72.8)
    & plat.between(40.5, 41.8)
    & dlat.between(40.5, 41.8)
)

mask &= ~(
    (plon.abs() < 1e-6)
    & (plat.abs() < 1e-6)
    & (dlon.abs() < 1e-6)
    & (dlat.abs() < 1e-6)
)

mask &= ~((plon == dlon) & (plat == dlat) & (fare > 20.0))

train = train.loc[mask].copy()
train.describe()



## === cell 13
train.reset_index(drop=True, inplace=True)
train.head()




## === cell 14
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(deep=False)

    out["key"] = out["key"].astype(str)

    dt = out["pickup_datetime"]
    if not np.issubdtype(dt.dtype, np.datetime64):
        dt = pd.to_datetime(dt, errors="coerce", utc=True)

    out["pickup_hour"] = dt.dt.hour.astype("float32").fillna(-1.0)
    out["pickup_dayofweek"] = dt.dt.dayofweek.astype("float32").fillna(-1.0)
    out["pickup_month"] = dt.dt.month.astype("float32").fillna(-1.0)
    out["pickup_year"] = dt.dt.year.astype("float32").fillna(-1.0)

    out = out.drop(columns=["pickup_datetime"])
    return out


train_fe = add_time_features(train)
test_fe = add_time_features(test)

train_fe.head()



## --- ERROR in cell 14, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mTypeError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/192116539.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     20[0m [0;34m[0m[0m
[1;32m     21[0m [0;34m[0m[0m
[0;32m---> 22[0;31m [0mtrain_fe[0m [0;34m=[0m [0madd_time_features[0m[0;34m([0m[0mtrain[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     23[0m [0mtest_fe[0m [0;34m=[0m [0madd_time_features[0m[0;34m([0m[0mtest[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     24[0m [0;34m[0m[0m

[0;32m/tmp/ipykernel_11/192116539.py[0m in [0;36madd_time_features[0;34m(df)[0m
[1;32m      8[0m     [0mdt[0m [0;34m=[0m [0mout[0m[0;34m[[0m[0;34m"pickup_datetime"[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m      9[0m     [0;31m# If any unexpected object dtype sneaks in, coerce once (still equivalent to original).[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 10[0;31m     [0;32mif[0m [0;32mnot[0m [0mnp[0m[0;34m.[0m[0missubdtype[0m[0;34m([0m[0mdt[0m[0;34m.[0m[0mdtype[0m[0;34m,[0m [0mnp[0m[0;34m.[0m[0mdatetime64[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     11[0m         [0mdt[0m [0;34m=[0m [0mpd[0m[0;34m.[0m[0mto_datetime[0m[0;34m([0m[0mdt[0m[0;34m,[0m [0merrors[0m[0;34m=[0m[0;34m"coerce"[0m[0;34m,[0m [0mutc[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     12[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/numpy/core/numerictypes.py[0m in [0;36missubdtype[0;34m(arg1, arg2)[0m
[1;32m    415[0m     """
[1;32m    416[0m     [0;32mif[0m [0;32mnot[0m [0missubclass_[0m[0;34m([0m[0marg1[0m[0;34m,[0m [0mgeneric[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 417[0;31m         [0marg1[0m [0;34m=[0m [0mdtype[0m[0;34m([0m[0marg1[0m[0;34m)[0m[0;34m.[0m[0mtype[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    418[0m     [0;32mif[0m [0;32mnot[0m [0missubclass_[0m[0;34m([0m[0marg2[0m[0;34m,[0m [0mgeneric[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    419[0m         [0marg2[0m [0;34m=[0m [0mdtype[0m[0;34m([0m[0marg2[0m[0;34m)[0m[0;34m.[0m[0mtype[0m[0;34m[0m[0;34m[0m[0m

[0;31mTypeError[0m: Cannot interpret 'datetime64[ns, UTC]' as a data type

## === cell 15
EARTH_RADIUS_KM = 6371.0088


def _to_rad_f32(arr: pd.Series) -> np.ndarray:
    return np.radians(arr.to_numpy(dtype=np.float64, copy=False))


def haversine_km_from_rad(lat1r, lon1r, lat2r, lon2r):
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c


def bearing_rad_from_rad(lat1r, lon1r, lat2r, lon2r):
    dlon = lon2r - lon1r
    y = np.sin(dlon) * np.cos(lat2r)
    x = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
    return np.arctan2(y, x)


def add_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(deep=False)

    plat_r = _to_rad_f32(out["pickup_latitude"])
    plon_r = _to_rad_f32(out["pickup_longitude"])
    dlat_r = _to_rad_f32(out["dropoff_latitude"])
    dlon_r = _to_rad_f32(out["dropoff_longitude"])

    dist = haversine_km_from_rad(plat_r, plon_r, dlat_r, dlon_r)
    out["distance"] = dist

    out["abs_lon_diff"] = (out["pickup_longitude"] - out["dropoff_longitude"]).abs()
    out["abs_lat_diff"] = (out["pickup_latitude"] - out["dropoff_latitude"]).abs()

    man1 = haversine_km_from_rad(plat_r, plon_r, plat_r, dlon_r)
    man2 = haversine_km_from_rad(plat_r, plon_r, dlat_r, plon_r)
    out["manhattan_km"] = man1 + man2

    out["center_lat"] = (
        (out["pickup_latitude"] + out["dropoff_latitude"]) / 2.0
    ).astype("float32")
    out["center_lon"] = (
        (out["pickup_longitude"] + out["dropoff_longitude"]) / 2.0
    ).astype("float32")

    b = bearing_rad_from_rad(plat_r, plon_r, dlat_r, dlon_r).astype("float32")
    out["bearing"] = b
    out["bearing_abs"] = np.abs(b).astype("float32")

    return out


train_fe = add_geo_features(train_fe)
test_fe = add_geo_features(test_fe)

train_fe.head()
