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
geopy==2.4.1
lightgbm==4.6.0
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

# 5. Target score

4.22957

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 5.77234) has done: 'I fix the runtime errors caused by deprecated pandas datetime accessors (`.dt.week`, `.dt.weekofyear`) by switching to the supported `isocalendar()` equivalents. I replace the removed `geopy.distance.VincentyDistance` with `geopy.distance.geodesic` while keeping the same “distance in km” feature and ensure it’s computed efficiently enough for 1M rows. I also make sure LightGBM only receives numeric feature columns (not `key`/datetimes), so `fit()` succeeds and the train/validation split works. Finally, I ensure a valid `submission.csv` (with `key,fare_amount`) is always written to the working directory.'
- What this solution (achieved 5.63018) has done: 'Your current RMSE (5.77) is worse than the target (4.23), so we should make small, legitimate improvements that usually reduce error without changing the model/training loop. The biggest win here is fixing the haversine calculation (it currently misses the required squares, which makes the feature wrong) and removing redundant/low-signal time features derived from `key` (which is not a real timestamp feature) while keeping the same overall feature-engineering + LightGBM approach. I also add the standard NYC Taxi sanity filters (fare bounds, passenger_count bounds, and NYC-ish coordinate bounds) to reduce training noise; this is a common minimal improvement for this competition and typically improves RMSE. Finally, I keep the submission schema identical (`key,fare_amount`) and still clip negative predictions to 0.'
- What this solution (achieved 5.43711) has done: 'Your RMSE (5.63) is worse than the target (4.23), so we should make small, legitimate improvements without changing the overall LightGBM + feature-engineering pipeline. The biggest issue is that you currently drop the raw lat/long columns after creating distance features; keeping the raw coordinates is a standard small improvement for this competition and usually reduces RMSE while preserving the same model/training approach. I also remove the unused `key2` parsing (it adds work and can introduce NaTs/side-effects but is not used for modeling), and I add a minimal, metric-aligned post-process to clip extreme predictions to the same fare range used in training filters (0–250), which typically slightly improves RMSE stability. All paths and the submission schema (`key,fare_amount`) remain unchanged and a valid `submission.csv` is always written.'
- What this solution (achieved 5.44603) has done: 'I make two minimal, score-relevant fixes while keeping your LightGBM + feature-engineering pipeline intact: (1) stop dropping rows from the test set (your current `_apply_common_filters_test` removes keys and produces an invalid submission row count), and instead only coerce/clip invalid test values so every test `key` is predicted; (2) add one standard, lightweight geospatial feature (`abs_lonlat_sum`) that typically reduces RMSE without changing the model/training loop. These changes should both ensure a valid submission CSV and nudge RMSE down toward the 4.23 target without altering the core approach.'
- What this solution (achieved 5.58137) has done: 'Your current RMSE (5.446) is still far from the target (4.230), so we should make small, legitimate improvements that usually reduce error while keeping the same LightGBM regressor + feature-engineering structure. The biggest score drag here is that several “distance_travelled_*” trig features are mathematically inconsistent (they multiply squared deltas instead of using the actual traveled distance), and LightGBM can overfit noise from extreme/invalid coordinates even after basic bounding. I (1) correct those trig features to be functions of the already-defined `distance_travelled`, (2) add two very standard, lightweight geo features (`manhattan` and `euclidean_km`) without changing the modeling approach, and (3) apply one additional common training-only filter to remove zero-distance rides with non-trivial fares (noise) while leaving test untouched to preserve row count and submission validity. These changes are minimal, keep the same training loop and model type, and should move RMSE down toward the target.'
- What this solution (achieved 5.39655) has done: 'We need to move RMSE down from 5.58 toward 4.23 (lower is better), so we keep your exact LightGBM approach and existing features but make two minimal, competition-standard fixes that usually reduce error without changing the core pipeline. First, we compute a proper geodesic “center distance to NYC” feature and a simple “JFK/LaGuardia distance” feature (both derived from existing lat/lon) to help the model distinguish airport trips—this is a small, legitimate feature add. Second, we tighten the training-only data cleaning by dropping obviously wrong coordinates where pickup/dropoff are identical but fare is high, and by removing a small set of extreme outliers based on your already-present `distance` feature; test is still never dropped to preserve submission row count. These changes are minimal, keep the same training loop/model, and should reduce RMSE toward your target band.'
- What this solution (achieved 5.46746) has done: 'We need to move RMSE down from 5.39655 toward 4.22957 (lower is better), so I keep your same LightGBM regressor/training flow and existing feature set, but make two minimal, score-relevant fixes that typically reduce error in this competition. First, I correct the bearing computation bug (your `y = sin(delta_chg * cos(phi2))` is not the standard formula and injects noise) and leave everything else in the distance block unchanged. Second, I add two lightweight, standard geospatial interaction features (`delta_lon_coslat` and `direction`) derived from existing columns to help the model capture trip orientation without altering the modeling approach. The rest (filters, split, model params, submission format/paths) stays the same and still produces a valid `submission.csv`.'
- What this solution (achieved 5.42764) has done: 'We need to reduce RMSE from 5.46746 toward 4.22957 (lower is better), so we keep your exact LightGBM regressor/training flow and the same feature set, but fix one high-impact feature bug and a train-only data quality issue. First, your haversine distance uses `pickup_latitude - dropoff_latitude` / `pickup_longitude - dropoff_longitude` for deltas; while distance magnitude is symmetric, this couples with earlier feature blocks and can introduce subtle inconsistencies—so we compute deltas in the standard direction and ensure `a` is clipped into [0,1] for numeric stability. Second, we add a minimal, competition-standard train-only filter to drop zero/near-zero coordinate-change rides (pickup==dropoff) which are mostly data errors and add noise; we not drop any test rows to keep submission alignment. These changes are small, preserve the same overall approach, and typically nudge RMSE downward without changing the model architecture or training loop.'
- What this solution (achieved 5.45145) has done: 'We need to reduce RMSE from 5.42764 toward the 4.22957 target (lower is better), so we keep your LightGBM regressor and the same overall feature-engineering pipeline, but make two minimal, competition-standard improvements that typically reduce error without changing the modeling approach. First, we fix the “distance” scale mismatch by computing `distance_km` from the haversine (km) and keeping your existing `haversine` (meters) intact—this aligns the distance feature’s units with typical fare relationships. Second, we add a very lightweight, standard time-derived signal (`is_weekend`) and a simple coordinate interaction (`latlon_manhattan_km`) built from already-present columns, which usually improves generalization. Finally, we clip predictions to the same bounds used in train cleaning (0–250) as before and still write a valid `submission.csv` with the required schema.'
- What this solution (achieved 5.45145) has done: 'We need to move RMSE down from 5.451 toward 4.230 (lower is better), so we keep your exact LightGBM training flow and feature pipeline, but make two minimal, high-impact, competition-standard corrections that reduce noise without changing the modeling approach. First, we fix the time parsing to correctly handle the dataset’s microseconds in `pickup_datetime` (your current fixed format coerces many rows to NaT → zeros, losing signal). Second, we switch the train/validation split to be deterministic and representative by using a simple random subsample before splitting (still the same `train_test_split` approach), which typically improves generalization when training on 1M rows. All paths remain unchanged, no test rows are dropped, and we still write a valid `submission.csv` with `key,fare_amount`.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use("fivethirtyeight")

import geopy.distance

import os
import gc

INPUT_DIR_CANDIDATES = [
    "../input",
    "/kaggle/input",
    "/kaggle/data",
    "../kaggle/input",
    "../kaggle/data",
]
INPUT_DIR = None
for d in INPUT_DIR_CANDIDATES:
    if os.path.isdir(d):
        INPUT_DIR = d
        break

print("Detected INPUT_DIR:", INPUT_DIR)
if INPUT_DIR is not None:
    try:
        print("Top-level listing:", os.listdir(INPUT_DIR)[:20])
    except Exception as e:
        print("Could not list input dir:", e)




## === cell 1
def _resolve_path(filename: str) -> str:
    """
    Keep original relative paths if they exist; otherwise try common Kaggle locations.
    """
    p1 = os.path.join("../input", filename)
    if os.path.exists(p1):
        return p1

    if INPUT_DIR is not None:
        p2 = os.path.join(INPUT_DIR, filename)
        if os.path.exists(p2):
            return p2

        p3 = os.path.join(INPUT_DIR, "new-york-city-taxi-fare-prediction", filename)
        if os.path.exists(p3):
            return p3

    return filename


def load_Data():
    train_path = _resolve_path("train.csv")
    test_path = _resolve_path("test.csv")

    train = pd.read_csv(
        train_path,
        nrows=1_000_000,
        low_memory=True,
        random_state=42,
    )
    test = pd.read_csv(test_path, nrows=10_000_000, low_memory=True)
    return train, test


def prepare_distance_features(df):
    df["longitude_distance"] = abs(df["pickup_longitude"] - df["dropoff_longitude"])
    df["latitude_distance"] = abs(df["pickup_latitude"] - df["dropoff_latitude"])

    df["distance_travelled"] = (
        df["longitude_distance"] ** 2 + df["latitude_distance"] ** 2
    ) ** 0.5

    dt = df["distance_travelled"].astype(float)
    df["distance_travelled_sin"] = np.sin(dt)
    df["distance_travelled_cos"] = np.cos(dt)
    df["distance_travelled_sin_sqrd"] = np.sin(dt) ** 2
    df["distance_travelled_cos_sqrd"] = np.cos(dt) ** 2

    R = 6371e3  # Metres
    phi1 = np.radians(df["pickup_latitude"])
    phi2 = np.radians(df["dropoff_latitude"])
    dphi = np.radians(df["dropoff_latitude"] - df["pickup_latitude"])
    dlambda = np.radians(df["dropoff_longitude"] - df["pickup_longitude"])
    a = (np.sin(dphi / 2) ** 2) + np.cos(phi1) * np.cos(phi2) * (
        np.sin(dlambda / 2) ** 2
    )
    a = np.clip(a.astype(float), 0.0, 1.0)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    d = R * c
    df["haversine"] = d

    dlon = np.radians(df["dropoff_longitude"] - df["pickup_longitude"])
    y = np.sin(dlon) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlon)
    df["bearing"] = np.arctan2(y, x)

    return df


def prepare_time_features(df):
    s = df["pickup_datetime"].astype(str).str.replace(" UTC", "", regex=False)
    df["pickup_datetime"] = pd.to_datetime(
        s, errors="coerce"
    )  # handles microseconds too

    df["hour_of_day"] = df.pickup_datetime.dt.hour

    iso = df.pickup_datetime.dt.isocalendar()
    df["week"] = iso.week.astype("int16")
    df["week_of_year"] = iso.week.astype("int16")

    df["month"] = df.pickup_datetime.dt.month
    df["year"] = df.pickup_datetime.dt.year
    df["day_of_year"] = df.pickup_datetime.dt.dayofyear
    df["weekday"] = df.pickup_datetime.dt.weekday
    df["quarter"] = df.pickup_datetime.dt.quarter
    df["day_of_month"] = df.pickup_datetime.dt.day

    df["is_weekend"] = (df["weekday"] >= 5).astype("int8")

    return df


def _apply_common_filters_train(df):
    df = df[df["fare_amount"].between(0, 250)]
    df = df[df["passenger_count"].between(1, 6)]
    df = df[df["pickup_longitude"].between(-75, -72)]
    df = df[df["dropoff_longitude"].between(-75, -72)]
    df = df[df["pickup_latitude"].between(40, 42)]
    df = df[df["dropoff_latitude"].between(40, 42)]
    return df


def _apply_common_filters_test(df):
    return df




## === cell 2
train, test = load_Data()



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3731616490.py in <cell line: 0>()
----> 1 train, test = load_Data()
      2 

/tmp/ipykernel_11/439694028.py in load_Data()
     25     # Change (score): deterministic 1M-row sampling (same size as before) to stabilize
     26     # training distribution and cleaned set; reduces run-to-run noise and often improves RMSE.
---> 27     train = pd.read_csv(
     28         train_path,
     29         nrows=1_000_000,

TypeError: read_csv() got an unexpected keyword argument 'random_state'

## === cell 3
train = prepare_time_features(train)
train = prepare_distance_features(train)

test = prepare_time_features(test)
test = prepare_distance_features(test)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/102392425.py in <cell line: 0>()
----> 1 train = prepare_time_features(train)
      2 train = prepare_distance_features(train)
      3 
      4 test = prepare_time_features(test)
      5 test = prepare_distance_features(test)

NameError: name 'train' is not defined

## === cell 4
train.describe()



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/199662553.py in <cell line: 0>()
----> 1 train.describe()
      2 

NameError: name 'train' is not defined

## === cell 5
train.info()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2342793378.py in <cell line: 0>()
----> 1 train.info()
      2 

NameError: name 'train' is not defined

## === cell 6
train.isnull().sum()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2969740008.py in <cell line: 0>()
----> 1 train.isnull().sum()
      2 

NameError: name 'train' is not defined

## === cell 7
train = train.fillna(0)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2930721484.py in <cell line: 0>()
----> 1 train = train.fillna(0)
      2 

NameError: name 'train' is not defined

## === cell 8
train.info()



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2342793378.py in <cell line: 0>()
----> 1 train.info()
      2 

NameError: name 'train' is not defined

## === cell 9
pass



## === cell 10
train["fare_amount"].plot(kind="box")



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2748529303.py in <cell line: 0>()
----> 1 train["fare_amount"].plot(kind="box")
      2 

NameError: name 'train' is not defined

## === cell 11
gc.collect()
train.describe()



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1362745405.py in <cell line: 0>()
      1 gc.collect()
----> 2 train.describe()
      3 

NameError: name 'train' is not defined

## === cell 12
print(
    "% of fares above 25$ - {:0.2f}".format(
        train[train["fare_amount"] > 25]["key"].count() * 100 / train["key"].count()
    )
)
print(
    "% of fares above 50$ - {:0.2f}".format(
        train[train["fare_amount"] > 50]["key"].count() * 100 / train["key"].count()
    )
)
print(
    "% of fares above 100$ - {:0.2f}".format(
        train[train["fare_amount"] > 100]["key"].count() * 100 / train["key"].count()
    )
)
print(
    "% of fares below 0$ - {:0.2f}".format(
        train[train["fare_amount"] < 0]["key"].count() * 100 / train["key"].count()
    )
)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/573446209.py in <cell line: 0>()
      1 print(
      2     "% of fares above 25$ - {:0.2f}".format(
----> 3         train[train["fare_amount"] > 25]["key"].count() * 100 / train["key"].count()
      4     )
      5 )

NameError: name 'train' is not defined

## === cell 13
fig, axarr = plt.subplots(2, 2, figsize=(20, 10))

train[~(train["fare_amount"] > 25)]["fare_amount"].plot(kind="box", ax=axarr[0][0])
train[~(train["fare_amount"] > 50)]["fare_amount"].plot(kind="box", ax=axarr[0][1])
train[~(train["fare_amount"] > 100)]["fare_amount"].plot(kind="box", ax=axarr[1][0])
train[~(train["fare_amount"] < 0)]["fare_amount"].plot(kind="box", ax=axarr[1][1])



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4150874805.py in <cell line: 0>()
      1 fig, axarr = plt.subplots(2, 2, figsize=(20, 10))
      2 
----> 3 train[~(train["fare_amount"] > 25)]["fare_amount"].plot(kind="box", ax=axarr[0][0])
      4 train[~(train["fare_amount"] > 50)]["fare_amount"].plot(kind="box", ax=axarr[0][1])
      5 train[~(train["fare_amount"] > 100)]["fare_amount"].plot(kind="box", ax=axarr[1][0])

NameError: name 'train' is not defined

## === cell 14
train["passenger_count"].plot(kind="box")



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2668212536.py in <cell line: 0>()
----> 1 train["passenger_count"].plot(kind="box")
      2 

NameError: name 'train' is not defined

## === cell 15
print(
    "Count of invalid pickup latitude",
    train[(train["pickup_latitude"] > 90) | (train["pickup_latitude"] < -90)][
        "pickup_latitude"
    ].count(),
)
print(
    "Count of invalid dropoff latitude",
    train[(train["dropoff_latitude"] > 90) | (train["dropoff_latitude"] < -90)][
        "dropoff_latitude"
    ].count(),
)
print(
    "Count of invalid pickup longitude",
    train[(train["pickup_longitude"] > 180) | (train["pickup_longitude"] < -180)][
        "pickup_longitude"
    ].count(),
)
print(
    "Count of invalid dropoff longitude",
    train[(train["dropoff_longitude"] > 180) | (train["dropoff_longitude"] < -180)][
        "dropoff_longitude"
    ].count(),
)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/351628942.py in <cell line: 0>()
      1 print(
      2     "Count of invalid pickup latitude",
----> 3     train[(train["pickup_latitude"] > 90) | (train["pickup_latitude"] < -90)][
      4         "pickup_latitude"
      5     ].count(),

NameError: name 'train' is not defined

## === cell 16
print(
    "Count of invalid pickup latitude",
    test[(test["pickup_latitude"] > 90) | (test["pickup_latitude"] < -90)][
        "pickup_latitude"
    ].count(),
)
print(
    "Count of invalid dropoff latitude",
    test[(test["dropoff_latitude"] > 90) | (test["dropoff_latitude"] < -90)][
        "dropoff_latitude"
    ].count(),
)
print(
    "Count of invalid pickup longitude",
    test[(test["pickup_longitude"] > 180) | (test["pickup_longitude"] < -180)][
        "pickup_longitude"
    ].count(),
)
print(
    "Count of invalid dropoff longitude",
    test[(test["dropoff_longitude"] > 180) | (test["dropoff_longitude"] < -180)][
        "dropoff_longitude"
    ].count(),
)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/356559153.py in <cell line: 0>()
      1 print(
      2     "Count of invalid pickup latitude",
----> 3     test[(test["pickup_latitude"] > 90) | (test["pickup_latitude"] < -90)][
      4         "pickup_latitude"
      5     ].count(),

NameError: name 'test' is not defined

## === cell 17
train = _apply_common_filters_train(train)



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2040160537.py in <cell line: 0>()
----> 1 train = _apply_common_filters_train(train)
      2 

NameError: name 'train' is not defined

## === cell 18
train = train[~((train["pickup_latitude"] > 90) | (train["pickup_latitude"] < -90))]
train = train[~((train["dropoff_latitude"] > 90) | (train["dropoff_latitude"] < -90))]
train = train[~((train["pickup_longitude"] > 180) | (train["pickup_longitude"] < -180))]
train = train[
    ~((train["dropoff_longitude"] > 180) | (train["dropoff_longitude"] < -180))
]



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1162860583.py in <cell line: 0>()
----> 1 train = train[~((train["pickup_latitude"] > 90) | (train["pickup_latitude"] < -90))]
      2 train = train[~((train["dropoff_latitude"] > 90) | (train["dropoff_latitude"] < -90))]
      3 train = train[~((train["pickup_longitude"] > 180) | (train["pickup_longitude"] < -180))]
      4 train = train[
      5     ~((train["dropoff_longitude"] > 180) | (train["dropoff_longitude"] < -180))

NameError: name 'train' is not defined

## === cell 19
test = _apply_common_filters_test(test)




## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3280212987.py in <cell line: 0>()
----> 1 test = _apply_common_filters_test(test)
      2 
      3 

NameError: name 'test' is not defined

## === cell 20
def _geodesic_km(lat1, lon1, lat2, lon2):
    try:
        return geopy.distance.geodesic((lat1, lon1), (lat2, lon2)).km
    except Exception:
        return np.nan


HAVERSINE_CAP_METERS = 100_000.0  # 100km
train["haversine"] = np.clip(
    train["haversine"].astype(float), 0.0, HAVERSINE_CAP_METERS
)
test["haversine"] = np.clip(test["haversine"].astype(float), 0.0, HAVERSINE_CAP_METERS)

train["distance_km"] = train["haversine"] / 1000.0
test["distance_km"] = test["haversine"] / 1000.0

train["abs_lonlat_sum"] = train["longitude_distance"] + train["latitude_distance"]
test["abs_lonlat_sum"] = test["longitude_distance"] + test["latitude_distance"]

train["manhattan"] = train["longitude_distance"] + train["latitude_distance"]
test["manhattan"] = test["longitude_distance"] + test["latitude_distance"]

KM_PER_DEG_LAT = 111.32
train_lat_mean = train["pickup_latitude"].astype(float).fillna(40.75).mean()
cos_lat = float(np.cos(np.radians(train_lat_mean)))
train["euclidean_km"] = np.sqrt(
    (train["longitude_distance"] * KM_PER_DEG_LAT * cos_lat) ** 2
    + (train["latitude_distance"] * KM_PER_DEG_LAT) ** 2
)
test["euclidean_km"] = np.sqrt(
    (test["longitude_distance"] * KM_PER_DEG_LAT * cos_lat) ** 2
    + (test["latitude_distance"] * KM_PER_DEG_LAT) ** 2
)

NYC_CENTER = (40.7580, -73.9855)  # Times Sq approx
JFK = (40.6413, -73.7781)
LGA = (40.7769, -73.8740)


def _haversine_km_vec(lat1, lon1, lat2, lon2):
    R_km = 6371.0
    lat1 = np.radians(lat1.astype(float))
    lon1 = np.radians(lon1.astype(float))
    lat2 = np.radians(lat2 if isinstance(lat2, (float, int)) else lat2.astype(float))
    lon2 = np.radians(lon2 if isinstance(lon2, (float, int)) else lon2.astype(float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    a = np.clip(a.astype(float), 0.0, 1.0)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R_km * c


train["pickup_to_center_km"] = _haversine_km_vec(
    train["pickup_latitude"], train["pickup_longitude"], NYC_CENTER[0], NYC_CENTER[1]
)
train["dropoff_to_center_km"] = _haversine_km_vec(
    train["dropoff_latitude"], train["dropoff_longitude"], NYC_CENTER[0], NYC_CENTER[1]
)
test["pickup_to_center_km"] = _haversine_km_vec(
    test["pickup_latitude"], test["pickup_longitude"], NYC_CENTER[0], NYC_CENTER[1]
)
test["dropoff_to_center_km"] = _haversine_km_vec(
    test["dropoff_latitude"], test["dropoff_longitude"], NYC_CENTER[0], NYC_CENTER[1]
)

train["pickup_to_jfk_km"] = _haversine_km_vec(
    train["pickup_latitude"], train["pickup_longitude"], JFK[0], JFK[1]
)
train["dropoff_to_jfk_km"] = _haversine_km_vec(
    train["dropoff_latitude"], train["dropoff_longitude"], JFK[0], JFK[1]
)
test["pickup_to_jfk_km"] = _haversine_km_vec(
    test["pickup_latitude"], test["pickup_longitude"], JFK[0], JFK[1]
)
test["dropoff_to_jfk_km"] = _haversine_km_vec(
    test["dropoff_latitude"], test["dropoff_longitude"], JFK[0], JFK[1]
)

train["pickup_to_lga_km"] = _haversine_km_vec(
    train["pickup_latitude"], train["pickup_longitude"], LGA[0], LGA[1]
)
train["dropoff_to_lga_km"] = _haversine_km_vec(
    train["dropoff_latitude"], train["dropoff_longitude"], LGA[0], LGA[1]
)
test["pickup_to_lga_km"] = _haversine_km_vec(
    test["pickup_latitude"], test["pickup_longitude"], LGA[0], LGA[1]
)
test["dropoff_to_lga_km"] = _haversine_km_vec(
    test["dropoff_latitude"], test["dropoff_longitude"], LGA[0], LGA[1]
)

train["min_airport_km"] = np.minimum(
    np.minimum(train["pickup_to_jfk_km"], train["dropoff_to_jfk_km"]),
    np.minimum(train["pickup_to_lga_km"], train["dropoff_to_lga_km"]),
)
test["min_airport_km"] = np.minimum(
    np.minimum(test["pickup_to_jfk_km"], test["dropoff_to_jfk_km"]),
    np.minimum(test["pickup_to_lga_km"], test["dropoff_to_lga_km"]),
)

train["delta_lon_coslat"] = (
    train["dropoff_longitude"] - train["pickup_longitude"]
) * np.cos(np.radians((train["pickup_latitude"] + train["dropoff_latitude"]) / 2.0))
test["delta_lon_coslat"] = (
    test["dropoff_longitude"] - test["pickup_longitude"]
) * np.cos(np.radians((test["pickup_latitude"] + test["dropoff_latitude"]) / 2.0))
train["direction"] = np.cos(train["bearing"].astype(float))
test["direction"] = np.cos(test["bearing"].astype(float))

train["latlon_manhattan_km"] = (
    train["longitude_distance"] * KM_PER_DEG_LAT * cos_lat
) + (train["latitude_distance"] * KM_PER_DEG_LAT)
test["latlon_manhattan_km"] = (
    test["longitude_distance"] * KM_PER_DEG_LAT * cos_lat
) + (test["latitude_distance"] * KM_PER_DEG_LAT)

train = train[
    ~((train["longitude_distance"] < 1e-6) & (train["latitude_distance"] < 1e-6))
]
train = train[~((train["distance_km"] < 0.01) & (train["fare_amount"] > 3.0))]
train = train[~((train["distance_km"] > 60.0) & (train["fare_amount"] < 10.0))]

train = train[train["fare_amount"] <= 200.0]

train = train[~((train["distance_km"] > 30.0) & (train["fare_amount"] < 15.0))]
train = train[~((train["distance_km"] < 1.0) & (train["fare_amount"] > 80.0))]

gc.collect()



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3158868570.py in <cell line: 0>()
      8 HAVERSINE_CAP_METERS = 100_000.0  # 100km
      9 train["haversine"] = np.clip(
---> 10     train["haversine"].astype(float), 0.0, HAVERSINE_CAP_METERS
     11 )
     12 test["haversine"] = np.clip(test["haversine"].astype(float), 0.0, HAVERSINE_CAP_METERS)

NameError: name 'train' is not defined

## === cell 21
pass



## === cell 22
pass



## === cell 23
print(
    "% of trips above 25 KM - {:0.2f}".format(
        train[train["distance_km"] > 25]["key"].count() * 100 / train.count()["key"]
    )
)



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1393379310.py in <cell line: 0>()
      1 print(
      2     "% of trips above 25 KM - {:0.2f}".format(
----> 3         train[train["distance_km"] > 25]["key"].count() * 100 / train.count()["key"]
      4     )
      5 )

NameError: name 'train' is not defined

## === cell 24
column_list = [
    "passenger_count",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "distance_km",
    "haversine",
    "bearing",
    "abs_lonlat_sum",
    "manhattan",
    "euclidean_km",
    "latlon_manhattan_km",
    "pickup_to_center_km",
    "dropoff_to_center_km",
    "pickup_to_jfk_km",
    "dropoff_to_jfk_km",
    "pickup_to_lga_km",
    "dropoff_to_lga_km",
    "min_airport_km",
    "delta_lon_coslat",
    "direction",
    "hour_of_day",
    "week",
    "month",
    "year",
    "day_of_year",
    "weekday",
    "is_weekend",
    "quarter",
    "day_of_month",
]

train[column_list] = train[column_list].apply(pd.to_numeric, errors="coerce").fillna(0)
test[column_list] = test[column_list].apply(pd.to_numeric, errors="coerce").fillna(0)

test["passenger_count"] = test["passenger_count"].clip(1, 6)
test["pickup_longitude"] = test["pickup_longitude"].clip(-75, -72)
test["dropoff_longitude"] = test["dropoff_longitude"].clip(-75, -72)
test["pickup_latitude"] = test["pickup_latitude"].clip(40, 42)
test["dropoff_latitude"] = test["dropoff_latitude"].clip(40, 42)

y_train_full = train["fare_amount"].astype(float)
X_train_full = train[column_list]
X_test = test[column_list]



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2195423904.py in <cell line: 0>()
     32 ]
     33 
---> 34 train[column_list] = train[column_list].apply(pd.to_numeric, errors="coerce").fillna(0)
     35 test[column_list] = test[column_list].apply(pd.to_numeric, errors="coerce").fillna(0)
     36 

NameError: name 'train' is not defined

## === cell 25
X_train_full.shape, y_train_full.shape



## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2295244986.py in <cell line: 0>()
----> 1 X_train_full.shape, y_train_full.shape
      2 

NameError: name 'X_train_full' is not defined

## === cell 26
from sklearn.model_selection import train_test_split

N_SUBSAMPLE_FOR_SPLIT = 300_000
if len(X_train_full) > N_SUBSAMPLE_FOR_SPLIT:
    idx = X_train_full.sample(n=N_SUBSAMPLE_FOR_SPLIT, random_state=42).index
    X_for_split = X_train_full.loc[idx]
    y_for_split = y_train_full.loc[idx]
else:
    X_for_split = X_train_full
    y_for_split = y_train_full

X_train, X_val, y_train, y_val = train_test_split(
    X_for_split, y_for_split, test_size=0.1, random_state=42
)



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3263236682.py in <cell line: 0>()
      2 
      3 N_SUBSAMPLE_FOR_SPLIT = 300_000
----> 4 if len(X_train_full) > N_SUBSAMPLE_FOR_SPLIT:
      5     idx = X_train_full.sample(n=N_SUBSAMPLE_FOR_SPLIT, random_state=42).index
      6     X_for_split = X_train_full.loc[idx]

NameError: name 'X_train_full' is not defined

## === cell 27
X_train.shape, X_val.shape, y_train.shape, y_val.shape



## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1307963576.py in <cell line: 0>()
----> 1 X_train.shape, X_val.shape, y_train.shape, y_val.shape
      2 

NameError: name 'X_train' is not defined

## === cell 28
from lightgbm import LGBMRegressor



## === cell 29
lgb = LGBMRegressor(
    objective="regression",
    num_leaves=5,
    learning_rate=0.05,
    n_estimators=720,
    max_bin=55,
    bagging_fraction=0.8,
    bagging_freq=5,
    feature_fraction=0.2319,
    feature_fraction_seed=9,
    bagging_seed=9,
    min_data_in_leaf=6,
    min_sum_hessian_in_leaf=11,
)



## === cell 30
lgb.fit(X_train, y_train)



## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1351089673.py in <cell line: 0>()
----> 1 lgb.fit(X_train, y_train)
      2 

NameError: name 'X_train' is not defined

## === cell 31
pred = lgb.predict(X_val)



## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1990325146.py in <cell line: 0>()
----> 1 pred = lgb.predict(X_val)
      2 

NameError: name 'X_val' is not defined

## === cell 32
from sklearn.metrics import mean_squared_error, r2_score

rmse = mean_squared_error(y_val, pred, squared=False)
r2 = r2_score(y_val, pred)
rmse, r2



## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/39818729.py in <cell line: 0>()
      1 from sklearn.metrics import mean_squared_error, r2_score
      2 
----> 3 rmse = mean_squared_error(y_val, pred, squared=False)
      4 r2 = r2_score(y_val, pred)
      5 rmse, r2

NameError: name 'y_val' is not defined

## === cell 33
lgb.fit(X_train_full, y_train_full)




## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4218421634.py in <cell line: 0>()
----> 1 lgb.fit(X_train_full, y_train_full)
      2 
      3 

NameError: name 'X_train_full' is not defined

## === cell 34
def display_importances(feature_importance_df_, doWorst=False, n_feat=50):
    if not doWorst:
        cols = (
            feature_importance_df_[["feature", "importance"]]
            .groupby("feature")
            .mean()
            .sort_values(by="importance", ascending=False)[:n_feat]
            .index
        )
    else:
        cols = (
            feature_importance_df_[["feature", "importance"]]
            .groupby("feature")
            .mean()
            .sort_values(by="importance", ascending=False)[-n_feat:]
            .index
        )

    mean_imp = (
        feature_importance_df_[["feature", "importance"]].groupby("feature").mean()
    )
    df_2_neglect = mean_imp[mean_imp["importance"] < 1e-3]
    print("The list of features with 0 importance: ")
    print(df_2_neglect.index.values.tolist())
    del mean_imp, df_2_neglect

    best_features = feature_importance_df_.loc[
        feature_importance_df_.feature.isin(cols)
    ]

    plt.figure(figsize=(8, 10))
    sns.barplot(
        x="importance",
        y="feature",
        data=best_features.sort_values(by="importance", ascending=False),
    )
    plt.title("LightGBM Features")
    plt.tight_layout()
    plt.savefig("lgbm_importances.png")


importance_df = pd.DataFrame()
importance_df["feature"] = column_list
importance_df["importance"] = lgb.feature_importances_
display_importances(feature_importance_df_=importance_df, n_feat=20)



## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/394966183.py in <cell line: 0>()
     42 importance_df = pd.DataFrame()
     43 importance_df["feature"] = column_list
---> 44 importance_df["importance"] = lgb.feature_importances_
     45 display_importances(feature_importance_df_=importance_df, n_feat=20)
     46 

/usr/local/lib/python3.11/dist-packages/lightgbm/sklearn.py in feature_importances_(self)
   1264         """
   1265         if not self.__sklearn_is_fitted__():
-> 1266             raise LGBMNotFittedError("No feature_importances found. Need to call fit beforehand.")
   1267         return self._Booster.feature_importance(importance_type=self.importance_type)  # type: ignore[union-attr]
   1268 

NotFittedError: No feature_importances found. Need to call fit beforehand.

## === cell 35
y_pred = lgb.predict(X_test)



## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3681145001.py in <cell line: 0>()
----> 1 y_pred = lgb.predict(X_test)
      2 

NameError: name 'X_test' is not defined

## === cell 36
y_pred = np.clip(y_pred, 0, 250)

submission = pd.DataFrame({"key": test["key"], "fare_amount": y_pred})
submission.to_csv("submission.csv", index=False)
submission.head()



## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1018034027.py in <cell line: 0>()
----> 1 y_pred = np.clip(y_pred, 0, 250)
      2 
      3 submission = pd.DataFrame({"key": test["key"], "fare_amount": y_pred})
      4 submission.to_csv("submission.csv", index=False)
      5 submission.head()

NameError: name 'y_pred' is not defined

## === cell 37
print("Wrote submission to:", os.path.abspath("submission.csv"))
print("Submission shape:", submission.shape)
print("Submission columns:", submission.columns.tolist())
print(submission.describe())
print("Test rows:", len(test))
print("Submission rows:", len(submission))
assert len(submission) == len(
    test
), "Submission must have exactly one prediction per test row."
assert submission.columns.tolist() == ["key", "fare_amount"]
assert submission["key"].isna().sum() == 0
assert submission["fare_amount"].isna().sum() == 0

## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/670772531.py in <cell line: 0>()
      1 print("Wrote submission to:", os.path.abspath("submission.csv"))
----> 2 print("Submission shape:", submission.shape)
      3 print("Submission columns:", submission.columns.tolist())
      4 print(submission.describe())
      5 print("Test rows:", len(test))

NameError: name 'submission' is not defined
