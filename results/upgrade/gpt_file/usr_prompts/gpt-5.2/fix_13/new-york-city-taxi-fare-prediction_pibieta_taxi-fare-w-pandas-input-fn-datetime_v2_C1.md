# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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
protobuf==6.33.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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

4.3003

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 102.11584) has done: 'I fix the TensorFlow build error by adding explicit `output_shape` (and dtype) to the `Lambda` layers used for string joining, because Keras can’t infer shapes for those ops in TF/Keras 2.18. I also ensure the model input arrays have shape `(n, 1)` to match the model `Input(shape=(1,))` specs, avoiding silent shape mismatches. After that, the rest of the pipeline (training, predicting, clipping, and writing `submission_file.csv`) run end-to-end and produce a valid submission with the required columns. These changes are execution/stability fixes and keep the core wide-and-deep logic and training semantics intact.'
- What this solution (achieved 5.91327) has done: 'Your score is far worse than the target (lower-is-better), which strongly suggests the model is training on a misaligned target/feature set rather than just being “underfit.” The key issue is that `traindf`/`evaldf` were created by dropping `key,pickup_datetime` but keeping the original row order, while your random mask is fine; however the biggest practical problem for this competition is that using only 100k rows without also removing extreme/outlier fares and invalid coordinate patterns often yields wildly unstable predictions on the public test distribution, producing huge RMSE. To move the score sharply toward ~4.3 with minimal core-logic change, we (1) strengthen the standard NYC Taxi Fare cleaning with a few additional safe filters (fare upper bound and coordinate zero checks) while keeping the same features/model, and (2) ensure passenger_count is treated as integer-like input consistently (still float32 tensor) and clip predictions to a realistic upper bound to prevent catastrophic errors. These changes preserve the same wide-and-deep architecture/training loop but prevent the “blow-up” cases that typically cause RMSE ~100.'
- What this solution (achieved 7.76793) has done: 'Your current RMSE (5.91) is worse than the target (4.30), so we should improve generalization with the smallest changes that don’t alter the wide-and-deep architecture or training loop. The biggest low-risk gain for this competition is stronger but standard NYC taxi cleaning (tighter fare cap, remove airport-far trips via distance cap, and remove extreme coordinate outliers) and applying the same geographic filter to the test set so features stay in-distribution. We also add a very mild label transform at prediction time only via clipping to a more realistic upper bound to prevent rare huge errors from dominating RMSE (still legitimate post-processing). Everything else (features, model, optimizer, epochs, batching) is left intact, and it still writes a valid `submission_file.csv`.'
- What this solution (achieved 10.10954) has done: 'Your current RMSE (7.76793) is worse than the target (4.3003), so we should improve generalization with the smallest changes that don’t alter the wide-and-deep model or training loop. The biggest low-risk lever here is making the training sample more representative: increase the number of training rows modestly (still feasible in time) and add a couple of standard NYC taxi cleaning rules that remove clearly-invalid rows that otherwise distort training. To avoid hurting RMSE with overly-aggressive test filtering, we keep predicting for all test rows but add an explicit “invalid test row fallback” that uses the train median fare only for obviously impossible coordinates/passenger_count. Finally, we clip predictions to a realistic upper bound aligned with typical public leaderboard distributions (without changing the loss/model), which reduces the impact of rare extreme predictions on RMSE.'
- What this solution (achieved 8.65863) has done: 'You’re far above the target RMSE (10.11 vs 4.30, lower is better), so we should make the smallest changes that improve generalization without changing the wide-and-deep model or training loop. The main low-risk gain is to strengthen “standard NYC taxi” training-data cleaning that removes mislabeled/invalid rows (especially `fare_amount==0`, extreme fares, and implausible per-km rates) that otherwise distort regression and blow up RMSE. We also make the train/test geographic/time features consistent by adding the same derived features and keeping all test rows, but we replace only clearly-impossible test rows with a robust fallback. Finally, we slightly relax the prediction upper clip from 150 to 250 to avoid underpredicting legitimate high-fare trips (which can hurt RMSE), while still preventing rare extreme outputs.'
- What this solution (achieved 5.46637) has done: 'We should move RMSE down toward 4.3003 (lower-is-better) with minimal risk and without changing the wide-and-deep model/training loop. The biggest low-change gain here is improving the training sample quality: your current cleaning removes some bad rows, but it still leaves common NYC taxi label issues (fare vs distance inconsistencies, extremely long trips, and too-small fares) that distort regression. I add two standard, lightweight filters that are directly tied to RMSE blow-ups: (1) remove “too-low fare for any trip” and “implausible distance for the fare” using conservative bounds, and (2) slightly increase the training rows (still feasible) to stabilize the learned embedding/hash weights. I also apply a matching *non-dropping* “out-of-distribution test fallback” for extreme distances (keep all rows, just replace clearly impossible ones with the robust median), which typically reduces catastrophic errors on the leaderboard without changing model semantics.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess
import numpy as np
import pandas as pd

try:
    import google.protobuf  # noqa: F401
    from importlib.metadata import version as _pkg_version

    pb_ver = _pkg_version("protobuf")
except Exception:
    pb_ver = None


def _major(v):
    try:
        return int(str(v).split(".")[0])
    except Exception:
        return None


if pb_ver is not None and _major(pb_ver) is not None and _major(pb_ver) >= 5:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
    )
    import importlib
    import google.protobuf as _gp

    importlib.reload(_gp)

import tensorflow as tf

print("/kaggle:", os.listdir("/kaggle")[:10])
print("/kaggle/input:", os.listdir("/kaggle/input")[:10])
print("tf version:", tf.__version__)



## === cell 1
TRAIN_PATH = "/kaggle/input/train.csv"
TEST_PATH = "/kaggle/input/test.csv"
SAMPLE_SUB_PATH = "/kaggle/input/sample_submission.csv"

NROWS_TRAIN = 5_000_000

df = pd.read_csv(TRAIN_PATH, nrows=NROWS_TRAIN, parse_dates=["pickup_datetime"])
test = pd.read_csv(TEST_PATH, parse_dates=["pickup_datetime"])
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

print(
    "train shape:",
    df.shape,
    "test shape:",
    test.shape,
    "sample_submission shape:",
    sample_sub.shape,
)
print("train cols:", df.columns.tolist())
print("test cols:", test.columns.tolist())



## === cell 2
_ = df.pickup_datetime.dt.day_name()
print("Datetime parsing OK. Example weekday:", df.pickup_datetime.dt.day_name().iloc[0])



## === cell 3
from math import cos, asin, sqrt


def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295  # Pi/180
    a = (
        0.5
        - cos((lat2 - lat1) * p) / 2
        + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
    )
    return 12742 * asin(sqrt(a))  # 2*R*asin...




## === cell 4
def add_feats(dfin: pd.DataFrame) -> pd.DataFrame:
    df = dfin.copy()

    p = 0.017453292519943295  # Pi/180
    lat1 = df["pickup_latitude"].astype("float64").to_numpy()
    lon1 = df["pickup_longitude"].astype("float64").to_numpy()
    lat2 = df["dropoff_latitude"].astype("float64").to_numpy()
    lon2 = df["dropoff_longitude"].astype("float64").to_numpy()

    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    df["distance"] = (12742 * np.arcsin(np.sqrt(a))).astype("float32")

    df["hour"] = df.pickup_datetime.dt.hour.astype("int16")
    df["weekday"] = df.pickup_datetime.dt.weekday.astype("int16")
    return df




## === cell 5
df = add_feats(df)
print(df.dtypes)



## === cell 6
print("Null pickup_datetime:", int(df.pickup_datetime.isnull().sum()))



## === cell 7
FARE_MIN = 2.5
FARE_MAX_TRAIN = 250.0  # keep legitimate long/airport trips
DIST_MIN = 0.2
DIST_MAX = 60.0
PC_MIN, PC_MAX = 1, 6

LON_MIN, LON_MAX = -74.5, -72.8
LAT_MIN, LAT_MAX = 40.0, 41.8

ABS_LON_MAX = 180.0
ABS_LAT_MAX = 90.0

RATE_MIN = 0.5
RATE_MAX = 50.0

BASE_FARE = 2.5
PER_KM_MIN = 0.5  # conservative
PER_KM_MAX = 20.0  # conservative for NYC (including surcharges/tolls, still wide)

base_mask = (
    ((df.pickup_longitude >= LON_MIN) & (df.pickup_longitude <= LON_MAX))
    & ((df.pickup_latitude >= LAT_MIN) & (df.pickup_latitude <= LAT_MAX))
    & ((df.dropoff_longitude >= LON_MIN) & (df.dropoff_longitude <= LON_MAX))
    & ((df.dropoff_latitude >= LAT_MIN) & (df.dropoff_latitude <= LAT_MAX))
    & (df.pickup_longitude.abs() <= ABS_LON_MAX)
    & (df.dropoff_longitude.abs() <= ABS_LON_MAX)
    & (df.pickup_latitude.abs() <= ABS_LAT_MAX)
    & (df.dropoff_latitude.abs() <= ABS_LAT_MAX)
    & (df.fare_amount > FARE_MIN)
    & (df.fare_amount < FARE_MAX_TRAIN)
    & (df.passenger_count >= PC_MIN)
    & (df.passenger_count <= PC_MAX)
    & (df.distance > DIST_MIN)
    & (df.distance < DIST_MAX)
    & ~(
        (df.pickup_longitude == 0)
        | (df.pickup_latitude == 0)
        | (df.dropoff_longitude == 0)
        | (df.dropoff_latitude == 0)
    )
    & ~((df.distance < 0.01) & (df.fare_amount > 5.0))
)

dist = df["distance"].astype("float64")
fare = df["fare_amount"].astype("float64")
rate = fare / np.maximum(dist, 1e-3)
rate_mask = (rate >= RATE_MIN) & (rate <= RATE_MAX)

per_km = (fare - BASE_FARE) / np.maximum(dist, 1e-3)
per_km_mask = (per_km >= PER_KM_MIN) & (per_km <= PER_KM_MAX)

low_fare_for_dist_mask = ~((dist > 1.0) & (fare < 3.0))

same_loc = (df["pickup_latitude"] == df["dropoff_latitude"]) & (
    df["pickup_longitude"] == df["dropoff_longitude"]
)
same_loc_bad_fare_mask = ~(same_loc & (df["fare_amount"] > 3.5))

key_dt = pd.to_datetime(
    df["key"].astype(str).str.split(".").str[0], errors="coerce", utc=False
)
dur_s = (key_dt - df["pickup_datetime"]).dt.total_seconds()
dur_ok = dur_s.notna() & (dur_s >= 30.0) & (dur_s <= 4 * 3600.0)
speed_kmh = dist / (dur_s / 3600.0)
speed_ok = (~dur_ok) | ((speed_kmh >= 1.0) & (speed_kmh <= 120.0))

dfc = df[
    base_mask
    & rate_mask
    & per_km_mask
    & low_fare_for_dist_mask
    & same_loc_bad_fare_mask
    & speed_ok
].copy()
print("Filtered train shape:", dfc.shape)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/datetimelike.py in _sub_datetimelike(self, other)
   1164         try:
-> 1165             self._assert_tzawareness_compat(other)
   1166         except TypeError as err:

/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/datetimes.py in _assert_tzawareness_compat(self, other)
    781             if other_tz is not None:
--> 782                 raise TypeError(
    783                     "Cannot compare tz-naive and tz-aware datetime-like objects."

TypeError: Cannot compare tz-naive and tz-aware datetime-like objects.

The above exception was the direct cause of the following exception:

TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3607302896.py in <cell line: 0>()
     65     df["key"].astype(str).str.split(".").str[0], errors="coerce", utc=False
     66 )
---> 67 dur_s = (key_dt - df["pickup_datetime"]).dt.total_seconds()
     68 # keep only positive, reasonable durations (30s..4h)
     69 dur_ok = dur_s.notna() & (dur_s >= 30.0) & (dur_s <= 4 * 3600.0)

/usr/local/lib/python3.11/dist-packages/pandas/core/ops/common.py in new_method(self, other)
     74         other = item_from_zerodim(other)
     75 
---> 76         return method(self, other)
     77 
     78     return new_method

/usr/local/lib/python3.11/dist-packages/pandas/core/arraylike.py in __sub__(self, other)
    192     @unpack_zerodim_and_defer("__sub__")
    193     def __sub__(self, other):
--> 194         return self._arith_method(other, operator.sub)
    195 
    196     @unpack_zerodim_and_defer("__rsub__")

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in _arith_method(self, other, op)
   6133     def _arith_method(self, other, op):
   6134         self, other = self._align_for_op(other)
-> 6135         return base.IndexOpsMixin._arith_method(self, other, op)
   6136 
   6137     def _align_for_op(self, right, align_asobject: bool = False):

/usr/local/lib/python3.11/dist-packages/pandas/core/base.py in _arith_method(self, other, op)
   1380 
   1381         with np.errstate(all="ignore"):
-> 1382             result = ops.arithmetic_op(lvalues, rvalues, op)
   1383 
   1384         return self._construct_result(result, name=res_name)

/usr/local/lib/python3.11/dist-packages/pandas/core/ops/array_ops.py in arithmetic_op(left, right, op)
    271         # Timedelta/Timestamp and other custom scalars are included in the check
    272         # because numexpr will fail on it, see GH#31457
--> 273         res_values = op(left, right)
    274     else:
    275         # TODO we should handle EAs consistently and move this check before the if/else

/usr/local/lib/python3.11/dist-packages/pandas/core/ops/common.py in new_method(self, other)
     74         other = item_from_zerodim(other)
     75 
---> 76         return method(self, other)
     77 
     78     return new_method

/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/datetimelike.py in __sub__(self, other)
   1457         ):
   1458             # DatetimeIndex, ndarray[datetime64]
-> 1459             result = self._sub_datetime_arraylike(other)
   1460         elif isinstance(other_dtype, PeriodDtype):
   1461             # PeriodIndex

/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/datetimelike.py in _sub_datetime_arraylike(self, other)
   1154 
   1155         self, other = self._ensure_matching_resos(other)
-> 1156         return self._sub_datetimelike(other)
   1157 
   1158     @final

/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/datetimelike.py in _sub_datetimelike(self, other)
   1166         except TypeError as err:
   1167             new_message = str(err).replace("compare", "subtract")
-> 1168             raise type(err)(new_message) from err
   1169 
   1170         other_i8, o_mask = self._get_i8_values_and_mask(other)

TypeError: Cannot subtract tz-naive and tz-aware datetime-like objects.

## === cell 8
np.random.seed(seed=1)
msk = np.random.rand(len(dfc)) < 0.8
traindf = dfc[msk].drop(["key", "pickup_datetime"], axis=1).copy()
evaldf = dfc[~msk].drop(["key", "pickup_datetime"], axis=1).copy()

print("traindf:", traindf.shape, "evaldf:", evaldf.shape)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1996008555.py in <cell line: 0>()
      1 np.random.seed(seed=1)
----> 2 msk = np.random.rand(len(dfc)) < 0.8
      3 traindf = dfc[msk].drop(["key", "pickup_datetime"], axis=1).copy()
      4 evaldf = dfc[~msk].drop(["key", "pickup_datetime"], axis=1).copy()
      5 

NameError: name 'dfc' is not defined

## === cell 9
testdf = add_feats(test)

TEST_DIST_MAX_FOR_MODEL = 80.0  # only for fallback, not for dropping
TEST_DIST_MIN_FOR_MODEL = 0.0

invalid_test_mask = (
    (testdf["pickup_longitude"].isna())
    | (testdf["pickup_latitude"].isna())
    | (testdf["dropoff_longitude"].isna())
    | (testdf["dropoff_latitude"].isna())
    | (testdf["pickup_longitude"].abs() > ABS_LON_MAX)
    | (testdf["dropoff_longitude"].abs() > ABS_LON_MAX)
    | (testdf["pickup_latitude"].abs() > ABS_LAT_MAX)
    | (testdf["dropoff_latitude"].abs() > ABS_LAT_MAX)
    | (testdf["passenger_count"].isna())
    | (testdf["passenger_count"] < PC_MIN)
    | (testdf["passenger_count"] > PC_MAX)
    | (testdf["pickup_longitude"] == 0)
    | (testdf["pickup_latitude"] == 0)
    | (testdf["dropoff_longitude"] == 0)
    | (testdf["dropoff_latitude"] == 0)
    | (testdf["distance"].isna())
    | (testdf["distance"] < TEST_DIST_MIN_FOR_MODEL)
    | (testdf["distance"] > TEST_DIST_MAX_FOR_MODEL)
)

testdf_full = testdf.drop(["key", "pickup_datetime"], axis=1).copy()
print(
    "testdf_full:",
    testdf_full.shape,
    "invalid_test_rows:",
    int(invalid_test_mask.sum()),
)



## === cell 10
print("weekday head:", traindf.weekday.head().tolist())




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2575052169.py in <cell line: 0>()
----> 1 print("weekday head:", traindf.weekday.head().tolist())
      2 
      3 

NameError: name 'traindf' is not defined

## === cell 11
def build_wide_and_deep_model(nbuckets=10):
    lat_cuts = np.linspace(38.0, 42.0, nbuckets + 1)[1:-1].tolist()
    lon_cuts = np.linspace(-75.0, -72.0, nbuckets + 1)[1:-1].tolist()

    inputs = {
        "pickup_longitude": tf.keras.Input(
            shape=(1,), name="pickup_longitude", dtype=tf.float32
        ),
        "pickup_latitude": tf.keras.Input(
            shape=(1,), name="pickup_latitude", dtype=tf.float32
        ),
        "dropoff_longitude": tf.keras.Input(
            shape=(1,), name="dropoff_longitude", dtype=tf.float32
        ),
        "dropoff_latitude": tf.keras.Input(
            shape=(1,), name="dropoff_latitude", dtype=tf.float32
        ),
        "passenger_count": tf.keras.Input(
            shape=(1,), name="passenger_count", dtype=tf.float32
        ),
        "distance": tf.keras.Input(shape=(1,), name="distance", dtype=tf.float32),
        "weekday": tf.keras.Input(shape=(1,), name="weekday", dtype=tf.int32),
        "hour": tf.keras.Input(shape=(1,), name="hour", dtype=tf.int32),
    }

    def bucketize(x, boundaries, name):
        return tf.keras.layers.Discretization(bin_boundaries=boundaries, name=name)(x)

    b_plat = bucketize(inputs["pickup_latitude"], lat_cuts, "b_plat")
    b_plon = bucketize(inputs["pickup_longitude"], lon_cuts, "b_plon")
    b_dlat = bucketize(inputs["dropoff_latitude"], lat_cuts, "b_dlat")
    b_dlon = bucketize(inputs["dropoff_longitude"], lon_cuts, "b_dlon")

    as_string = tf.keras.layers.Lambda(
        lambda t: tf.strings.as_string(t),
        name="as_string",
        output_shape=(1,),
        dtype=tf.string,
    )

    def join2(sep, nm):
        return tf.keras.layers.Lambda(
            lambda xs: tf.strings.join(xs, separator=sep),
            name=nm,
            output_shape=(1,),
            dtype=tf.string,
        )

    ploc_tok = join2("_", "ploc_tok")([as_string(b_plat), as_string(b_plon)])
    dloc_tok = join2("_", "dloc_tok")([as_string(b_dlat), as_string(b_dlon)])
    pd_pair_tok = join2("__", "pd_pair_tok")([ploc_tok, dloc_tok])
    day_hr_tok = join2("-", "day_hr_tok")(
        [as_string(inputs["weekday"]), as_string(inputs["hour"])]
    )

    wide_features = []

    for name, tok, bins in [
        ("ploc", ploc_tok, nbuckets * nbuckets),
        ("dloc", dloc_tok, nbuckets * nbuckets),
        ("pd_pair", pd_pair_tok, nbuckets**4),
        ("day_hr", day_hr_tok, 24 * 7),
    ]:
        hashed = tf.keras.layers.Hashing(num_bins=bins, name=f"{name}_hash")(tok)
        onehot = tf.keras.layers.CategoryEncoding(
            num_tokens=bins, output_mode="one_hot", name=f"{name}_onehot"
        )(hashed)
        wide_features.append(onehot)

    wide_features.append(
        tf.keras.layers.Lambda(
            lambda t: tf.cast(t, tf.float32),
            name="weekday_f32",
            output_shape=(1,),
            dtype=tf.float32,
        )(inputs["weekday"])
    )
    wide_features.append(
        tf.keras.layers.Lambda(
            lambda t: tf.cast(t, tf.float32),
            name="hour_f32",
            output_shape=(1,),
            dtype=tf.float32,
        )(inputs["hour"])
    )
    wide_features.append(inputs["passenger_count"])

    wide = tf.keras.layers.Concatenate(name="wide_concat")(wide_features)

    deep_features = []

    for name, tok, bins, emb_dim in [
        ("pd_pair", pd_pair_tok, nbuckets**4, 10),
        ("day_hr", day_hr_tok, 24 * 7, 10),
    ]:
        hashed = tf.keras.layers.Hashing(num_bins=bins, name=f"{name}_hash_deep")(tok)
        emb = tf.keras.layers.Embedding(
            input_dim=bins, output_dim=emb_dim, name=f"{name}_emb"
        )(hashed)
        emb = tf.keras.layers.Reshape((emb_dim,), name=f"{name}_emb_flat")(emb)
        deep_features.append(emb)

    deep_numeric = tf.keras.layers.Concatenate(name="deep_numeric")(
        [
            inputs["pickup_latitude"],
            inputs["pickup_longitude"],
            inputs["dropoff_latitude"],
            inputs["dropoff_longitude"],
            inputs["distance"],
        ]
    )
    deep = tf.keras.layers.Concatenate(name="deep_concat")(
        deep_features + [deep_numeric]
    )

    x = tf.keras.layers.Dense(128, activation="relu")(deep)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dense(4, activation="relu")(x)

    all_features = tf.keras.layers.Concatenate(name="wide_deep_concat")([wide, x])
    out = tf.keras.layers.Dense(1, name="fare_amount")(all_features)

    model = tf.keras.Model(inputs=inputs, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model




## === cell 12
OUTDIR = "./taxi_trained"
os.makedirs(OUTDIR, exist_ok=True)




## === cell 13
def df_to_model_inputs(dfin: pd.DataFrame):
    pc = pd.to_numeric(dfin["passenger_count"], errors="coerce").fillna(1).clip(1, 6)

    x = {
        "pickup_longitude": dfin["pickup_longitude"]
        .astype("float32")
        .to_numpy()
        .reshape(-1, 1),
        "pickup_latitude": dfin["pickup_latitude"]
        .astype("float32")
        .to_numpy()
        .reshape(-1, 1),
        "dropoff_longitude": dfin["dropoff_longitude"]
        .astype("float32")
        .to_numpy()
        .reshape(-1, 1),
        "dropoff_latitude": dfin["dropoff_latitude"]
        .astype("float32")
        .to_numpy()
        .reshape(-1, 1),
        "passenger_count": pc.astype("float32").to_numpy().reshape(-1, 1),
        "distance": dfin["distance"].astype("float32").to_numpy().reshape(-1, 1),
        "weekday": dfin["weekday"].astype("int32").to_numpy().reshape(-1, 1),
        "hour": dfin["hour"].astype("int32").to_numpy().reshape(-1, 1),
    }
    return x


BATCH_SIZE = 512
EPOCHS = 5

x_train = df_to_model_inputs(traindf.drop(["fare_amount"], axis=1))
y_train = traindf["fare_amount"].astype("float32").to_numpy()

x_eval = df_to_model_inputs(evaldf.drop(["fare_amount"], axis=1))
y_eval = evaldf["fare_amount"].astype("float32").to_numpy()

x_test_full = df_to_model_inputs(testdf_full)

print("Prepared model inputs. Train y mean:", float(np.mean(y_train)))



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/585883168.py in <cell line: 0>()
     30 EPOCHS = 5
     31 
---> 32 x_train = df_to_model_inputs(traindf.drop(["fare_amount"], axis=1))
     33 y_train = traindf["fare_amount"].astype("float32").to_numpy()
     34 

NameError: name 'traindf' is not defined

## === cell 14
tf.random.set_seed(1)
np.random.seed(1)

model = build_wide_and_deep_model(nbuckets=10)
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_eval, y_eval),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=2,
)

eval_metrics = model.evaluate(x_eval, y_eval, batch_size=4096, verbose=0)
print("Eval metrics:", dict(zip(model.metrics_names, eval_metrics)))



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3224164562.py in <cell line: 0>()
      4 model = build_wide_and_deep_model(nbuckets=10)
      5 history = model.fit(
----> 6     x_train,
      7     y_train,
      8     validation_data=(x_eval, y_eval),

NameError: name 'x_train' is not defined

## === cell 15
pred = (
    model.predict(x_test_full, batch_size=4096, verbose=0)
    .reshape(-1)
    .astype(np.float64)
)
assert len(pred) == len(test), f"Pred length {len(pred)} != test length {len(test)}"
print(
    "Pred stats (raw):", float(np.min(pred)), float(np.mean(pred)), float(np.max(pred))
)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2841230691.py in <cell line: 0>()
      1 pred = (
----> 2     model.predict(x_test_full, batch_size=4096, verbose=0)
      3     .reshape(-1)
      4     .astype(np.float64)
      5 )

NameError: name 'x_test_full' is not defined

## === cell 16
PRED_MAX = 200.0
pred = np.clip(pred, 0.0, PRED_MAX)

train_fare_median = float(np.median(y_train))
pred = np.where(np.isfinite(pred), pred, train_fare_median)
pred = pred.copy()
pred[invalid_test_mask.to_numpy()] = train_fare_median

print(
    "Pred stats (post):",
    float(np.min(pred)),
    float(np.mean(pred)),
    float(np.max(pred)),
    "fallback_rows:",
    int(invalid_test_mask.sum()),
)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1730437163.py in <cell line: 0>()
      2 # while keeping legitimate fares (still far above typical median/mean).
      3 PRED_MAX = 200.0
----> 4 pred = np.clip(pred, 0.0, PRED_MAX)
      5 
      6 train_fare_median = float(np.median(y_train))

NameError: name 'pred' is not defined

## === cell 17
output = pd.DataFrame({"key": test["key"].values, "fare_amount": pred})

assert list(output.columns) == ["key", "fare_amount"]
assert output.shape[0] == sample_sub.shape[0] == test.shape[0]
print(output.head())



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2876687259.py in <cell line: 0>()
----> 1 output = pd.DataFrame({"key": test["key"].values, "fare_amount": pred})
      2 
      3 assert list(output.columns) == ["key", "fare_amount"]
      4 assert output.shape[0] == sample_sub.shape[0] == test.shape[0]
      5 print(output.head())

NameError: name 'pred' is not defined

## === cell 18
SUB_PATH = "submission_file.csv"
output.to_csv(SUB_PATH, index=False)
print(f"Wrote {SUB_PATH} with shape:", output.shape)



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2771474330.py in <cell line: 0>()
      1 SUB_PATH = "submission_file.csv"
----> 2 output.to_csv(SUB_PATH, index=False)
      3 print(f"Wrote {SUB_PATH} with shape:", output.shape)
      4 

NameError: name 'output' is not defined

## === cell 19
chk = pd.read_csv(SUB_PATH, nrows=5)
print("Submission preview:")
print(chk)



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/4269619853.py in <cell line: 0>()
----> 1 chk = pd.read_csv(SUB_PATH, nrows=5)
      2 print("Submission preview:")
      3 print(chk)
      4 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: 'submission_file.csv'

## === cell 20
print("Done. Submission ready at:", os.path.abspath(SUB_PATH))
