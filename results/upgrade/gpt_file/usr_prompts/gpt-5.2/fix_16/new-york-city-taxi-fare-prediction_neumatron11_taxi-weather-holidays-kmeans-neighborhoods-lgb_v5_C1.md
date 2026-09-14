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

3.7

# 3. Installed packages

geopandas==0.14.4
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
scipy==1.15.3
sklearn-pandas==2.2.0
tqdm==4.67.1

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

3.382252995833693

# 6. Current score

6.50395

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.29042) has done: 'I remove the hard dependency on missing external datasets (NYC weather, bank holidays, and a pre-trained kmeans pickle) by providing safe fallbacks that keep the rest of the pipeline and feature columns intact. I ensure `nyc_weather`, `holidays`, and `kmeans` are always defined so later feature engineering and one-hot encoding don’t crash. I also fix a couple of logic bugs in the existing code (wrong `drop(..., axis=0)` usage, incorrect `SNOW/TMIN` assignments, inconsistent `day_hour` string construction, and deprecated `np.asscalar`) so training and submission generation run end-to-end. Finally, I keep the LightGBM training approach the same and make sure a valid `submission_lgb.csv` with columns `key,fare_amount` is written.'
- What this solution (achieved 4.62162) has done: 'The timeout is dominated by (1) the extremely slow Python-loop reservoir sampling over ~55M rows and (2) building one huge concatenated sparse matrix for train+test before splitting. I replace reservoir sampling with a deterministic, vectorized two-pass approach that picks exact row indices and reads only those rows (same sampling logic/size, but without per-row Python loops), and I build one-hot features for train and test separately after fitting the encoder (avoids the expensive full concat + vstack + hstack on tens of millions of rows). I also remove expensive plotting during training and make KMeans/predict steps fully vectorized and memory-friendly without changing features or model semantics. All file paths, features, model params, and training/validation logic remain the same.'
- What this solution (achieved 4.80252) has done: 'Your public RMSE is worse than the target, so the smallest safe improvement is to fix a train/test feature mismatch that’s currently injecting noise: `hot_day`/`cold_day` thresholds differ between train and test, which shifts the distribution of those binary features and hurts generalization. I align the thresholds so train and test use the exact same rules (this preserves the core feature set and model, but makes the feature semantics consistent). I also make the train/test split deterministic and more faithful by disabling shuffling (time leakage isn’t the goal here—this just reduces variance and can slightly stabilize RMSE without changing the model). Everything else (sampling size, feature engineering, OHE+LGBM training, and submission format/path) stays the same and still writes `submission_lgb.csv`.'
- What this solution (achieved 6.50395) has done: 'The timeout is dominated by reading/processing a 12M-row reservoir sample from the 55M-row CSV (full pass with heavy per-row RNG) and then training LightGBM on millions of rows. To preserve the exact core model/training semantics while making it fit in 600s, I keep the same feature engineering + LightGBM setup but (a) replace the reservoir sampler with a provably equivalent fast fixed-row subsample using `skiprows` (still an unbiased random sample of rows without replacement), (b) reduce pandas overhead by using pyarrow dtype backend + vectorized filtering, and (c) avoid extra copies/merges by using indexed joins for weather/holiday tables. These changes only remove unnecessary work (full-file pass + Python-level sampling) and keep the algorithm and evaluation unchanged aside from negligible floating-point effects.'
- What this solution (achieved 6.50395) has done: 'Your current RMSE (6.50) is far worse than the target (3.38), and the biggest issue is that weather/holiday features are almost always empty due to missing external datasets, which makes those columns effectively random/constant and harms generalization. The smallest legitimate improvement (without changing your model, features, or training semantics) is to load the weather and holidays from the competition-provided `nyc_weather.csv` and `holidays.csv` that exist under `new-york-city-taxi-fare-prediction/`, instead of falling back to empty tables. I also ensure the datetime parsing/merge keys match the file formats used in that dataset (and keep the same downstream columns `AWND,PRCP,SNOW,TMAX,TMIN` and `Holiday`). This should move your score substantially downward toward the target while keeping the rest of the pipeline intact and still producing `submission_lgb.csv`.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from tqdm import tqdm
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import MiniBatchKMeans
import random
import gc
import time

BASE_INPUT_CANDIDATES = [
    "../input",
    "/kaggle/input",
    "/kaggle/data",
    "/kaggle/data/new-york-city-taxi-fare-prediction",
]
print("CWD:", os.getcwd())
for p in BASE_INPUT_CANDIDATES:
    if os.path.exists(p):
        try:
            print("Found input path:", p, "contains", len(os.listdir(p)), "items")
        except Exception:
            print("Found input path:", p)

np.random.seed(17)
random.seed(17)
os.environ["PYTHONHASHSEED"] = "17"




## === cell 1
def _load_nyc_weather_or_fallback():
    """
    Change (score-improving, minimal): Prefer the competition-provided nyc_weather.csv
    (present under new-york-city-taxi-fare-prediction/) instead of falling back to empty.
    This preserves the same downstream feature columns but makes them informative.
    """
    candidates = [
        "../input/new-york-city-taxi-fare-prediction/nyc_weather.csv",
        "/kaggle/input/new-york-city-taxi-fare-prediction/nyc_weather.csv",
        "/kaggle/data/new-york-city-taxi-fare-prediction/nyc_weather.csv",
        "/kaggle/data/nyc_weather.csv",
        "../input/nyc-weather/nyc_weather.csv",
        "/kaggle/input/nyc-weather/nyc_weather.csv",
    ]
    for fp in candidates:
        if os.path.exists(fp):
            nyc_weather = pd.read_csv(fp)

            if "DATE" not in nyc_weather.columns:
                for alt in ["date", "Date"]:
                    if alt in nyc_weather.columns:
                        nyc_weather = nyc_weather.rename(columns={alt: "DATE"})
                        break

            weather_cols = ["DATE", "AWND", "PRCP", "SNOW", "TMAX", "TMIN"]
            existing = [c for c in weather_cols if c in nyc_weather.columns]
            nyc_weather = nyc_weather[existing].copy()

            nyc_weather["DATE"] = pd.to_datetime(
                nyc_weather["DATE"], utc=True, errors="coerce"
            )
            nyc_weather = nyc_weather.dropna(subset=["DATE"]).copy()

            for c in ["AWND", "PRCP", "SNOW", "TMAX", "TMIN"]:
                if c not in nyc_weather.columns:
                    nyc_weather[c] = np.nan

            nyc_weather = nyc_weather[weather_cols].copy()
            return nyc_weather

    return pd.DataFrame(columns=["DATE", "AWND", "PRCP", "SNOW", "TMAX", "TMIN"])


def _load_holidays_or_fallback():
    """
    Change (score-improving, minimal): Prefer the competition-provided holidays.csv
    (present under new-york-city-taxi-fare-prediction/) instead of falling back to empty.
    Keeps the same downstream 'Holiday' feature semantics.
    """
    candidates = [
        "../input/new-york-city-taxi-fare-prediction/holidays.csv",
        "/kaggle/input/new-york-city-taxi-fare-prediction/holidays.csv",
        "/kaggle/data/new-york-city-taxi-fare-prediction/holidays.csv",
        "/kaggle/data/holidays.csv",
        "../input/us-bank-holidays-20092018/US Bank Holidays 2012-2018.csv",
        "/kaggle/input/us-bank-holidays-20092018/US Bank Holidays 2012-2018.csv",
    ]
    for fp in candidates:
        if os.path.exists(fp):
            holidays = pd.read_csv(fp)

            if "Date" not in holidays.columns:
                for alt in ["date", "DATE"]:
                    if alt in holidays.columns:
                        holidays = holidays.rename(columns={alt: "Date"})
                        break
            if "Holiday" not in holidays.columns:
                for alt in ["holiday", "Name", "name"]:
                    if alt in holidays.columns:
                        holidays = holidays.rename(columns={alt: "Holiday"})
                        break

            if "Date" in holidays.columns:
                holidays["Date"] = pd.to_datetime(
                    holidays["Date"], utc=True, errors="coerce"
                )
                holidays = holidays.dropna(subset=["Date"]).copy()
            else:
                holidays["Date"] = pd.to_datetime([], utc=True)

            if "Holiday" not in holidays.columns:
                holidays["Holiday"] = "None"

            holidays["Holiday"] = holidays["Holiday"].astype(str)
            return holidays[["Date", "Holiday"]].copy()

    return pd.DataFrame(
        {"Date": pd.to_datetime([], utc=True), "Holiday": pd.Series([], dtype="object")}
    )


nyc_weather = _load_nyc_weather_or_fallback()
holidays = _load_holidays_or_fallback()

print("nyc_weather shape:", nyc_weather.shape, "cols:", list(nyc_weather.columns))
print("holidays shape:", holidays.shape, "cols:", list(holidays.columns))



## === cell 2
if (
    len(nyc_weather) > 0
    and "TMAX" in nyc_weather.columns
    and "TMIN" in nyc_weather.columns
):
    nyc_weather.head()
else:
    print("nyc_weather not available; using fallback empty table.")



## === cell 3
print("Skipping weather plots to save runtime.")



## === cell 4
if len(nyc_weather) > 0:
    nyc_weather.describe()
else:
    print("No weather data to describe (fallback).")



## === cell 5
if len(holidays) > 0:
    holidays.head(12)
else:
    print("No holidays data loaded (fallback).")




## === cell 6
def find_file(filename):
    candidates = [
        f"../input/new-york-city-taxi-fare-prediction/{filename}",
        f"/kaggle/input/new-york-city-taxi-fare-prediction/{filename}",
        f"/kaggle/data/{filename}",
        f"/kaggle/data/new-york-city-taxi-fare-prediction/{filename}",
        f"../input/{filename}",
    ]
    for fp in candidates:
        if os.path.exists(fp):
            return fp
    raise FileNotFoundError(f"Could not find {filename} in known input locations.")


train_path = find_file("train.csv")
test_path = find_file("test.csv")
sample_path = find_file("sample_submission.csv")

print("Using train:", train_path)
print("Using test:", test_path)
print("Using sample:", sample_path)




## === cell 7
def fast_random_row_sample_csv(
    path,
    sample_size,
    seed=17,
    usecols=None,
    dtype=None,
    total_rows=None,
):
    """
    Uniform sample without replacement from a CSV by randomly skipping rows.
    Requires total_rows (excluding header) to avoid a full scan for counting.
    """
    k = int(sample_size)
    if k <= 0:
        return pd.DataFrame(columns=usecols if usecols is not None else None)
    if total_rows is None:
        raise ValueError("total_rows must be provided to avoid scanning the full file.")
    if k >= total_rows:
        return pd.read_csv(path, usecols=usecols, dtype=dtype, engine="c")

    rng = np.random.RandomState(seed)
    chosen = rng.choice(
        np.arange(1, total_rows + 1, dtype=np.int64), size=k, replace=False
    )
    chosen.sort()

    chosen_set = set(chosen.tolist())
    skip = lambda i: (i != 0) and (i not in chosen_set)

    try:
        df = pd.read_csv(
            path,
            usecols=usecols,
            dtype=dtype,
            engine="c",
            skiprows=skip,
            dtype_backend="pyarrow",
        )
        return df
    except TypeError:
        return pd.read_csv(
            path,
            usecols=usecols,
            dtype=dtype,
            engine="c",
            skiprows=skip,
        )


t0 = time.time()

s = 12_000_000  # desired sample size (as in original)

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
read_dtype = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}

TRAIN_TOTAL_ROWS = 55_423_856

train = fast_random_row_sample_csv(
    train_path,
    sample_size=s,
    seed=17,
    usecols=usecols,
    dtype=read_dtype,
    total_rows=TRAIN_TOTAL_ROWS,
)

test = pd.read_csv(
    test_path,
    dtype={
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "uint8",
    },
    engine="c",
)
test_id = test.key.values  # set this value for final submission

print("Train shape:", train.shape, "Test shape:", test.shape)
print("Sampling+test read seconds:", round(time.time() - t0, 2))
gc.collect()



## === cell 8
train.head()



## === cell 9
t0 = time.time()

train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], utc=True, errors="coerce", cache=True
).dt.floor("min")
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], utc=True, errors="coerce", cache=True
).dt.floor("min")

train_sample = train.dropna().copy()
del train
gc.collect()

train_sample.drop(labels="key", axis=1, inplace=True, errors="ignore")
test.drop(labels="key", axis=1, inplace=True, errors="ignore")

train_sample.loc[:, "passenger_count"] = train_sample.passenger_count.astype(
    dtype="uint8"
)
train_sample["pickup_longitude"] = train_sample.pickup_longitude.astype(dtype="float32")
train_sample["pickup_latitude"] = train_sample.pickup_latitude.astype(dtype="float32")
train_sample["dropoff_longitude"] = train_sample.dropoff_longitude.astype(
    dtype="float32"
)
train_sample["dropoff_latitude"] = train_sample.dropoff_latitude.astype(dtype="float32")
train_sample["fare_amount"] = train_sample.fare_amount.astype(dtype="float32")

test["pickup_longitude"] = test.pickup_longitude.astype(dtype="float32")
test["pickup_latitude"] = test.pickup_latitude.astype(dtype="float32")
test["dropoff_longitude"] = test.dropoff_longitude.astype(dtype="float32")
test["dropoff_latitude"] = test.dropoff_latitude.astype(dtype="float32")
test.loc[:, "passenger_count"] = test.passenger_count.astype(dtype="uint8")

plon_min, plon_max = float(test.pickup_longitude.min()), float(
    test.pickup_longitude.max()
)
plat_min, plat_max = float(test.pickup_latitude.min()), float(
    test.pickup_latitude.max()
)
dlon_min, dlon_max = float(test.dropoff_longitude.min()), float(
    test.dropoff_longitude.max()
)
dlat_min, dlat_max = float(test.dropoff_latitude.min()), float(
    test.dropoff_latitude.max()
)

mask = (
    train_sample.pickup_longitude.between(plon_min, plon_max)
    & train_sample.pickup_latitude.between(plat_min, plat_max)
    & train_sample.dropoff_longitude.between(dlon_min, dlon_max)
    & train_sample.dropoff_latitude.between(dlat_min, dlat_max)
    & (train_sample["fare_amount"] > 0)
    & (train_sample["fare_amount"] <= 250.0)
    & train_sample["passenger_count"].between(1, 6)
    & train_sample["pickup_latitude"].between(40.0, 42.0)
    & train_sample["dropoff_latitude"].between(40.0, 42.0)
    & train_sample["pickup_longitude"].between(-75.0, -72.0)
    & train_sample["dropoff_longitude"].between(-75.0, -72.0)
)
train_sample = train_sample.loc[mask].copy()

train_sample["hour"] = train_sample["pickup_datetime"].dt.hour
train_sample["month"] = train_sample["pickup_datetime"].dt.month
train_sample["day_of_week"] = train_sample["pickup_datetime"].dt.dayofweek
train_sample["year"] = train_sample["pickup_datetime"].dt.year

test["hour"] = test["pickup_datetime"].dt.hour
test["month"] = test["pickup_datetime"].dt.month
test["day_of_week"] = test["pickup_datetime"].dt.dayofweek
test["year"] = test["pickup_datetime"].dt.year

train_sample["hour"] = train_sample.hour.astype(dtype="uint8")
train_sample["month"] = train_sample.month.astype(dtype="uint8")
train_sample["day_of_week"] = train_sample.day_of_week.astype(dtype="uint8")
train_sample["year"] = train_sample.year.astype(dtype="uint16")

test["hour"] = test.hour.astype(dtype="uint8")
test["month"] = test.month.astype(dtype="uint8")
test["day_of_week"] = test.day_of_week.astype(dtype="uint8")
test["year"] = test.year.astype(dtype="uint16")

train_sample["pickup_day"] = train_sample.pickup_datetime.dt.floor("d")
test["pickup_day"] = test["pickup_datetime"].dt.floor("d")

if len(nyc_weather) > 0 and "DATE" in nyc_weather.columns:
    nyc_w = nyc_weather.set_index("DATE")
    train_sample = train_sample.join(nyc_w, on="pickup_day", how="left")
    test = test.join(nyc_w, on="pickup_day", how="left")
else:
    for col in ["AWND", "PRCP", "SNOW", "TMAX", "TMIN"]:
        train_sample[col] = np.nan
        test[col] = np.nan

train_sample.drop(columns=["pickup_day"], inplace=True, errors="ignore")
test.drop(columns=["pickup_day"], inplace=True, errors="ignore")

for col in ["AWND", "PRCP", "SNOW", "TMAX", "TMIN"]:
    if col not in train_sample.columns:
        train_sample[col] = np.nan
    if col not in test.columns:
        test[col] = np.nan

train_sample["AWND"] = train_sample.AWND.astype(dtype="float16")
train_sample["PRCP"] = train_sample.PRCP.astype(dtype="float16")
train_sample["SNOW"] = train_sample.SNOW.astype(dtype="float16")
train_sample["TMAX"] = train_sample.TMAX.astype(dtype="float16")
train_sample["TMIN"] = train_sample.TMIN.astype(dtype="float16")

test["AWND"] = test.AWND.astype(dtype="float16")
test["PRCP"] = test.PRCP.astype(dtype="float16")
test["SNOW"] = test.SNOW.astype(dtype="float16")
test["TMAX"] = test.TMAX.astype(dtype="float16")
test["TMIN"] = test.TMIN.astype(dtype="float16")

HOT_TMAX_THRESHOLD = 30.0
COLD_TMIN_THRESHOLD = 0.0

train_sample["hot_day"] = np.where(train_sample.TMAX >= HOT_TMAX_THRESHOLD, 1, 0)
train_sample["cold_day"] = np.where(train_sample.TMIN <= COLD_TMIN_THRESHOLD, 1, 0)

test["hot_day"] = np.where(test.TMAX >= HOT_TMAX_THRESHOLD, 1, 0)
test["cold_day"] = np.where(test.TMIN <= COLD_TMIN_THRESHOLD, 1, 0)

train_sample["hot_day"] = train_sample.hot_day.astype(dtype="uint8")
train_sample["cold_day"] = train_sample.cold_day.astype(dtype="uint8")
test["hot_day"] = test.hot_day.astype(dtype="uint8")
test["cold_day"] = test.cold_day.astype(dtype="uint8")


def degree_to_radion(degree):
    return degree * (np.pi / 180)


def calculate_distance(
    pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude
):
    from_lat = degree_to_radion(pickup_latitude)
    from_long = degree_to_radion(pickup_longitude)
    to_lat = degree_to_radion(dropoff_latitude)
    to_long = degree_to_radion(dropoff_longitude)

    radius = 6371.01
    lat_diff = to_lat - from_lat
    long_diff = to_long - from_long

    a = (
        np.sin(lat_diff / 2) ** 2
        + np.cos(from_lat) * np.cos(to_lat) * np.sin(long_diff / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return radius * c


train_sample["distance"] = calculate_distance(
    train_sample.pickup_latitude,
    train_sample.pickup_longitude,
    train_sample.dropoff_latitude,
    train_sample.dropoff_longitude,
)
test["distance"] = calculate_distance(
    test.pickup_latitude,
    test.pickup_longitude,
    test.dropoff_latitude,
    test.dropoff_longitude,
)

train_sample["distance"] = train_sample.distance.astype(dtype="float32")
test["distance"] = test.distance.astype(dtype="float32")

train_sample = train_sample.loc[
    (train_sample["distance"] >= 0.0) & (train_sample["distance"] <= 200.0)
].copy()

train_sample["day_hour"] = (
    train_sample.day_of_week.astype(str) + "_" + train_sample.hour.astype(str)
)
train_sample["day_hour"] = train_sample["day_hour"].astype("category")

test["day_hour"] = test.day_of_week.astype(str) + "_" + test.hour.astype(str)
test["day_hour"] = test["day_hour"].astype("category")

train_sample["pickup_day"] = train_sample.pickup_datetime.dt.floor("d")
test["pickup_day"] = test.pickup_datetime.dt.floor("d")

if len(holidays) > 0 and "Date" in holidays.columns:
    hol = holidays.set_index("Date")
    train_sample = train_sample.join(hol[["Holiday"]], on="pickup_day", how="left")
    test = test.join(hol[["Holiday"]], on="pickup_day", how="left")
else:
    train_sample["Holiday"] = "None"
    test["Holiday"] = "None"

train_sample["Holiday"] = train_sample["Holiday"].fillna("None")
test["Holiday"] = test["Holiday"].fillna("None")

le = LabelEncoder()
le.fit(pd.concat([train_sample["Holiday"], test["Holiday"]], axis=0).astype(str).values)

train_sample["holiday"] = le.transform(train_sample.Holiday.astype(str).values)
test["holiday"] = le.transform(test.Holiday.astype(str).values)

train_sample.drop(["Holiday", "pickup_day"], axis=1, inplace=True, errors="ignore")
test.drop(["Holiday", "pickup_day"], axis=1, inplace=True, errors="ignore")

train_sample["holiday"] = train_sample.holiday.astype(dtype="uint8")
test["holiday"] = test.holiday.astype(dtype="uint8")

print("Prepared train_sample:", train_sample.shape, "test:", test.shape)
print("Feature prep seconds:", round(time.time() - t0, 2))
gc.collect()



## === cell 10
train_sample.head()



## === cell 11
print("Columns in train_sample:", sorted(train_sample.columns.tolist()))
print("Columns in test:", sorted(test.columns.tolist()))



## === cell 12
num_clusters = 200



## === cell 13
round_decimals = 4
max_kmeans_points = 2_000_000  # cap only for KMeans fitting; does not change downstream feature computation
rng = np.random.RandomState(17)

pickup_xy = np.column_stack(
    [train_sample.pickup_longitude.values, train_sample.pickup_latitude.values]
).astype(np.float32)
dropoff_xy = np.column_stack(
    [train_sample.dropoff_longitude.values, train_sample.dropoff_latitude.values]
).astype(np.float32)
test_pickup_xy = np.column_stack(
    [test.pickup_longitude.values, test.pickup_latitude.values]
).astype(np.float32)
test_dropoff_xy = np.column_stack(
    [test.dropoff_longitude.values, test.dropoff_latitude.values]
).astype(np.float32)

pickup_xy = np.round(pickup_xy, round_decimals)
dropoff_xy = np.round(dropoff_xy, round_decimals)
test_pickup_xy = np.round(test_pickup_xy, round_decimals)
test_dropoff_xy = np.round(test_dropoff_xy, round_decimals)

all_xy = np.vstack([pickup_xy, dropoff_xy, test_pickup_xy, test_dropoff_xy])

if all_xy.shape[0] > max_kmeans_points:
    idx = rng.choice(all_xy.shape[0], size=max_kmeans_points, replace=False)
    X_kmeans = all_xy[idx]
else:
    X_kmeans = all_xy

kmeans = MiniBatchKMeans(
    n_clusters=num_clusters,
    random_state=17,
    batch_size=100_000,
    n_init=10,
    reassignment_ratio=0.01,
)
kmeans.fit(X_kmeans)

print("Fitted MiniBatchKMeans on points:", X_kmeans.shape[0], "clusters:", num_clusters)

del pickup_xy, dropoff_xy, test_pickup_xy, test_dropoff_xy, all_xy, X_kmeans
gc.collect()



## === cell 14
train_pickup_mat = np.column_stack(
    [train_sample.pickup_longitude.values, train_sample.pickup_latitude.values]
).astype(np.float32, copy=False)
train_dropoff_mat = np.column_stack(
    [train_sample.dropoff_longitude.values, train_sample.dropoff_latitude.values]
).astype(np.float32, copy=False)
test_pickup_mat = np.column_stack(
    [test.pickup_longitude.values, test.pickup_latitude.values]
).astype(np.float32, copy=False)
test_dropoff_mat = np.column_stack(
    [test.dropoff_longitude.values, test.dropoff_latitude.values]
).astype(np.float32, copy=False)

train_sample["pickup_neighborhood"] = kmeans.predict(train_pickup_mat)
train_sample["dropoff_neighborhood"] = kmeans.predict(train_dropoff_mat)
test["pickup_neighborhood"] = kmeans.predict(test_pickup_mat)
test["dropoff_neighborhood"] = kmeans.predict(test_dropoff_mat)

del train_pickup_mat, train_dropoff_mat, test_pickup_mat, test_dropoff_mat
gc.collect()

train_sample["pickup_neighborhood"] = train_sample.pickup_neighborhood.astype(
    dtype="uint8"
)
train_sample["dropoff_neighborhood"] = train_sample.dropoff_neighborhood.astype(
    dtype="uint8"
)
test["pickup_neighborhood"] = test.pickup_neighborhood.astype(dtype="uint8")
test["dropoff_neighborhood"] = test.dropoff_neighborhood.astype(dtype="uint8")



## === cell 15
train_sample.head()



## === cell 16
with open("kmeans_200_round4_v2.pkl", "wb") as fid:
    pickle.dump(kmeans, fid)

print("Saved kmeans_200_round4_v2.pkl")



## === cell 17
categorical_cols = [
    "day_hour",
    "month",
    "year",
    "pickup_neighborhood",
    "dropoff_neighborhood",
    "passenger_count",
    "hot_day",
    "cold_day",
    "holiday",
]
numerical_cols = ["distance", "AWND", "PRCP", "SNOW"]

missing_train = [
    c
    for c in categorical_cols + numerical_cols + ["fare_amount"]
    if c not in train_sample.columns
]
missing_test = [c for c in categorical_cols + numerical_cols if c not in test.columns]
print("Missing train cols:", missing_train)
print("Missing test cols:", missing_test)



## === cell 18
for c in categorical_cols:
    if c in train_sample.columns:
        train_sample[c] = train_sample[c].astype("category")
    if c in test.columns:
        test[c] = test[c].astype("category")

for c in categorical_cols:
    train_cats = train_sample[c].cat.categories
    test_cats = test[c].cat.categories
    union_cats = train_cats.union(test_cats)
    train_sample[c] = train_sample[c].cat.set_categories(union_cats)
    test[c] = test[c].cat.set_categories(union_cats)

X_nums_train = train_sample[numerical_cols].to_numpy(dtype=np.float32, copy=False)
X_nums_test = test[numerical_cols].to_numpy(dtype=np.float32, copy=False)
X_nums_train = np.nan_to_num(X_nums_train, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
X_nums_test = np.nan_to_num(X_nums_test, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

X_cats_train = np.column_stack(
    [
        train_sample[c].cat.codes.to_numpy(dtype=np.int32, copy=False)
        for c in categorical_cols
    ]
)
X_cats_test = np.column_stack(
    [test[c].cat.codes.to_numpy(dtype=np.int32, copy=False) for c in categorical_cols]
)

X = np.hstack([X_nums_train, X_cats_train])
X_public = np.hstack([X_nums_test, X_cats_test])

y = train_sample.fare_amount.values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=17, shuffle=False
)

cat_feature_indices = list(
    range(len(numerical_cols), len(numerical_cols) + len(categorical_cols))
)

print("X_train:", X_train.shape, "X_test:", X_test.shape, "X_public:", X_public.shape)
gc.collect()



## === cell 19
print("y stats:", float(np.nanmin(y)), float(np.nanmean(y)), float(np.nanmax(y)))



## === cell 20
params = {
    "objective": "regression",
    "boosting": "gbdt",
    "metric": "rmse",
    "num_leaves": 50,
    "max_depth": 8,
    "learning_rate": 0.1,
    "bagging_fraction": 0.8,
    "feature_fraction": 0.8,
    "min_split_gain": 0.02,
    "min_child_samples": 10,
    "min_child_weight": 0.02,
    "lambda_l2": 0.0475,
    "min_data_in_leaf": 50,
    "min_sum_hessian_in_leaf": 1e-3,
    "verbosity": -1,
    "seed": 17,
    "data_random_seed": 17,
    "feature_fraction_seed": 17,
    "bagging_seed": 17,
    "num_threads": max(1, os.cpu_count() or 1),
    "deterministic": True,
    "force_col_wise": True,
    "bin_construct_sample_cnt": 200000,  # as in original
}

d_train = lgb.Dataset(
    X_train,
    label=y_train,
    free_raw_data=True,
    categorical_feature=cat_feature_indices,
)
d_valid = lgb.Dataset(
    X_test,
    label=y_test,
    reference=d_train,
    free_raw_data=True,
    categorical_feature=cat_feature_indices,
)

num_rounds = 2000
verbose_eval = 200
early_stop = 200

model_lgb = lgb.train(
    params,
    train_set=d_train,
    num_boost_round=num_rounds,
    valid_sets=[d_train, d_valid],
    callbacks=[
        lgb.early_stopping(early_stop, verbose=False),
        lgb.log_evaluation(period=verbose_eval),
    ],
)

pred_test_y_lgb = model_lgb.predict(X_test, num_iteration=model_lgb.best_iteration)
print("LGB Loss = " + str(sqrt(mean_squared_error(y_test, pred_test_y_lgb))))



## === cell 21
del X_train, X_test, y_train, y_test, d_train, d_valid
gc.collect()



## === cell 22
lgb_public = model_lgb.predict(X_public, num_iteration=model_lgb.best_iteration)
final_pred_public = lgb_public.flatten()

final_pred_public = np.where(final_pred_public > 0, final_pred_public, 0.0).astype(
    float
)
sample = pd.DataFrame({"key": test_id, "fare_amount": final_pred_public})
sample = sample.reindex(["key", "fare_amount"], axis=1)
sample.to_csv("submission_lgb.csv", index=False)

print(sample.head())
print("Wrote submission_lgb.csv with shape:", sample.shape)



## === cell 23
print("Skipping prediction histogram to save runtime.")
