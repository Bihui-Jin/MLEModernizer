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

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 4.29042) has done: 'I remove the hard dependency on missing external datasets (NYC weather, bank holidays, and a pre-trained kmeans pickle) by providing safe fallbacks that keep the rest of the pipeline and feature columns intact. I ensure `nyc_weather`, `holidays`, and `kmeans` are always defined so later feature engineering and one-hot encoding don’t crash. I also fix a couple of logic bugs in the existing code (wrong `drop(..., axis=0)` usage, incorrect `SNOW/TMIN` assignments, inconsistent `day_hour` string construction, and deprecated `np.asscalar`) so training and submission generation run end-to-end. Finally, I keep the LightGBM training approach the same and make sure a valid `submission_lgb.csv` with columns `key,fare_amount` is written.'

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
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans, MiniBatchKMeans
import random

BASE_INPUT_CANDIDATES = [
    "../input",
    "/kaggle/input",
    "/kaggle/data",
    "/kaggle/data/new-york-city-taxi-fare-prediction",
]
print("CWD:", os.getcwd())
for p in BASE_INPUT_CANDIDATES:
    if os.path.exists(p):
        print("Found input path:", p, "contains", len(os.listdir(p)), "items")

np.random.seed(17)
random.seed(17)
os.environ["PYTHONHASHSEED"] = "17"




## === cell 1
def _load_nyc_weather_or_fallback():
    candidates = [
        "../input/nyc-weather/nyc_weather.csv",
        "/kaggle/input/nyc-weather/nyc_weather.csv",
    ]
    for fp in candidates:
        if os.path.exists(fp):
            nyc_weather = pd.read_csv(fp)
            weather_cols = ["DATE", "AWND", "PRCP", "SNOW", "TMAX", "TMIN"]
            nyc_weather = nyc_weather[weather_cols].copy()
            nyc_weather["DATE"] = pd.to_datetime(
                nyc_weather["DATE"], utc=True, format="%m/%d/%Y"
            )
            return nyc_weather

    return pd.DataFrame(columns=["DATE", "AWND", "PRCP", "SNOW", "TMAX", "TMIN"])


def _load_holidays_or_fallback():
    candidates = [
        "../input/us-bank-holidays-20092018/US Bank Holidays 2012-2018.csv",
        "/kaggle/input/us-bank-holidays-20092018/US Bank Holidays 2012-2018.csv",
    ]
    for fp in candidates:
        if os.path.exists(fp):
            holidays = pd.read_csv(fp)
            holidays["Date"] = pd.to_datetime(
                holidays["Date"], utc=True, format="%m/%d/%y"
            )
            return holidays
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
if len(nyc_weather) > 0 and nyc_weather[["TMAX", "TMIN"]].notna().any().any():
    plt.rc("figure", figsize=(15, 8))
    plt.subplot(1, 2, 1)
    plt.hist(nyc_weather.TMAX.dropna(), bins=30)
    plt.xlabel("Temperature (C)")
    plt.ylabel("Frequency Count")
    plt.title("Max Daily Temperature")
    plt.subplot(1, 2, 2)
    plt.hist(nyc_weather.TMIN.dropna(), bins=30)
    plt.xlabel("Temperature (C)")
    plt.ylabel("Frequency Count")
    plt.title("Min Daily Temperature")
    plt.show()
else:
    print("Skipping weather plots (no weather data loaded).")




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
def reservoir_sample_csv(path, sample_size, chunksize=1_000_000, seed=17):
    rng = np.random.RandomState(seed)
    reservoir = None
    seen = 0

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
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        arr = chunk.to_numpy(copy=False)
        if reservoir is None:
            take = min(sample_size, len(chunk))
            reservoir = arr[:take].copy()
            seen = take
            if take < sample_size and len(chunk) > take:
                remaining = arr[take:]
                for row in remaining:
                    j = rng.randint(0, seen + 1)
                    if seen < sample_size:
                        reservoir = np.vstack([reservoir, row])
                    else:
                        if j < sample_size:
                            reservoir[j] = row
                    seen += 1
            else:
                for row in arr[take:]:
                    j = rng.randint(0, seen + 1)
                    if j < sample_size:
                        reservoir[j] = row
                    seen += 1
            continue

        for row in arr:
            j = rng.randint(0, seen + 1)
            if j < sample_size:
                reservoir[j] = row
            seen += 1

    cols = [
        "key",
        "fare_amount",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]
    return pd.DataFrame(reservoir, columns=cols)


s = 12_000_000  # desired sample size (as in original)
train = reservoir_sample_csv(train_path, sample_size=s, chunksize=750_000, seed=17)

test = pd.read_csv(test_path)
test_id = test.key.values  # set this value for final submission

print("Train shape:", train.shape, "Test shape:", test.shape)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/3477686910.py in <cell line: 0>()
     67 
     68 s = 12_000_000  # desired sample size (as in original)
---> 69 train = reservoir_sample_csv(train_path, sample_size=s, chunksize=750_000, seed=17)
     70 
     71 test = pd.read_csv(test_path)

/tmp/ipykernel_11/3477686910.py in reservoir_sample_csv(path, sample_size, chunksize, seed)
     48             j = rng.randint(0, seen + 1)
     49             if j < sample_size:
---> 50                 reservoir[j] = row
     51             seen += 1
     52 

IndexError: index 750234 is out of bounds for axis 0 with size 750000

## === cell 8
train.head()




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1975634829.py in <cell line: 0>()
----> 1 train.head()
      2 
      3 

NameError: name 'train' is not defined

## === cell 9
train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"].astype(str).str.slice(0, 16),
    utc=True,
    format="%Y-%m-%d %H:%M",
    errors="coerce",
)
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"].astype(str).str.slice(0, 16),
    utc=True,
    format="%Y-%m-%d %H:%M",
    errors="coerce",
)

train_sample = train.dropna()
del train

train_sample.drop(labels="key", axis=1, inplace=True)
test.drop(labels="key", axis=1, inplace=True)

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

train_sample = train_sample.loc[
    train_sample.pickup_longitude.between(
        test.pickup_longitude.min(), test.pickup_longitude.max()
    )
]
train_sample = train_sample.loc[
    train_sample.pickup_latitude.between(
        test.pickup_latitude.min(), test.pickup_latitude.max()
    )
]
train_sample = train_sample.loc[
    train_sample.dropoff_longitude.between(
        test.dropoff_longitude.min(), test.dropoff_longitude.max()
    )
]
train_sample = train_sample.loc[
    train_sample.dropoff_latitude.between(
        test.dropoff_latitude.min(), test.dropoff_latitude.max()
    )
]

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
train_sample = train_sample.merge(
    nyc_weather, how="left", left_on="pickup_day", right_on="DATE"
)
train_sample.drop(columns=["pickup_day", "DATE"], inplace=True, errors="ignore")

test["pickup_day"] = test["pickup_datetime"].dt.floor("d")
test = test.merge(nyc_weather, how="left", left_on="pickup_day", right_on="DATE")
test.drop(columns=["pickup_day", "DATE"], inplace=True, errors="ignore")

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

train_sample["hot_day"] = np.where(train_sample.TMAX >= 30, 1, 0)
train_sample["cold_day"] = np.where(train_sample.TMIN <= 0, 1, 0)

test["hot_day"] = np.where(test.TMAX >= 35, 1, 0)
test["cold_day"] = np.where(test.TMIN <= -5, 1, 0)

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

train_sample["day_hour"] = (
    train_sample.day_of_week.astype(str) + "_" + train_sample.hour.astype(str)
)
train_sample["day_hour"] = train_sample["day_hour"].astype("category")

test["day_hour"] = test.day_of_week.astype(str) + "_" + test.hour.astype(str)
test["day_hour"] = test["day_hour"].astype("category")

train_sample = train_sample[train_sample.fare_amount > 0]

train_sample["pickup_day"] = train_sample.pickup_datetime.dt.floor("d")
train_sample = train_sample.merge(
    holidays, left_on="pickup_day", right_on="Date", how="left"
)
if "Holiday" not in train_sample.columns:
    train_sample["Holiday"] = "None"
train_sample["Holiday"] = train_sample.Holiday.fillna("None")

test["pickup_day"] = test.pickup_datetime.dt.floor("d")
test = test.merge(holidays, left_on="pickup_day", right_on="Date", how="left")
if "Holiday" not in test.columns:
    test["Holiday"] = "None"
test["Holiday"] = test.Holiday.fillna("None")

le = LabelEncoder()
le.fit(pd.concat([train_sample["Holiday"], test["Holiday"]], axis=0).astype(str).values)

train_sample["holiday"] = le.transform(train_sample.Holiday.astype(str).values)
test["holiday"] = le.transform(test.Holiday.astype(str).values)

train_sample.drop(
    ["Holiday", "Date", "pickup_day"], axis=1, inplace=True, errors="ignore"
)
test.drop(["Holiday", "Date", "pickup_day"], axis=1, inplace=True, errors="ignore")

train_sample["holiday"] = train_sample.holiday.astype(dtype="uint8")
test["holiday"] = test.holiday.astype(dtype="uint8")

print("Prepared train_sample:", train_sample.shape, "test:", test.shape)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3571437049.py in <cell line: 0>()
      2 # Correctness preserved: still truncates to minutes exactly as original.
      3 train["pickup_datetime"] = pd.to_datetime(
----> 4     train["pickup_datetime"].astype(str).str.slice(0, 16),
      5     utc=True,
      6     format="%Y-%m-%d %H:%M",

NameError: name 'train' is not defined

## === cell 10
train_sample.head()




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/54140371.py in <cell line: 0>()
----> 1 train_sample.head()
      2 
      3 

NameError: name 'train_sample' is not defined

## === cell 11
print("Columns in train_sample:", sorted(train_sample.columns.tolist()))
print("Columns in test:", sorted(test.columns.tolist()))




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3785779852.py in <cell line: 0>()
----> 1 print("Columns in train_sample:", sorted(train_sample.columns.tolist()))
      2 print("Columns in test:", sorted(test.columns.tolist()))
      3 
      4 

NameError: name 'train_sample' is not defined

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

all_xy = np.vstack([pickup_xy, dropoff_xy, test_pickup_xy, test_dropoff_xy])
all_xy = np.round(all_xy, round_decimals)

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

centers = kmeans.cluster_centers_
x_centers = [pair[0] for pair in centers]
y_centers = [pair[1] for pair in centers]
z_centers = np.arange(num_clusters)

plt.subplot(1, 2, 1)
plot_idx = rng.choice(
    X_kmeans.shape[0], size=min(50_000, X_kmeans.shape[0]), replace=False
)
plt.scatter(
    X_kmeans[plot_idx, 0],
    X_kmeans[plot_idx, 1],
    c=kmeans.predict(X_kmeans[plot_idx]),
    s=2,
)
plt.gray()
plt.xlabel("Pickup/Dropoff Longitude")
plt.ylabel("Pickup/Dropoff Latitude")
plt.title("Clusters of NYC locations (subsample)")
plt.subplot(1, 2, 2)
plt.scatter(x_centers, y_centers, c=z_centers, s=10)
plt.gray()
plt.xlabel("Pickup/Dropoff Longitude")
plt.ylabel("Pickup/Dropoff Latitude")
plt.title("Cluster Centers of NYC locations")
plt.show()

del pickup_xy, dropoff_xy, test_pickup_xy, test_dropoff_xy, all_xy, X_kmeans




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/849382413.py in <cell line: 0>()
      6 
      7 pickup_xy = np.column_stack(
----> 8     [train_sample.pickup_longitude.values, train_sample.pickup_latitude.values]
      9 ).astype(np.float32)
     10 dropoff_xy = np.column_stack(

NameError: name 'train_sample' is not defined

## === cell 14
train_sample["pickup_neighborhood"] = kmeans.predict(
    np.column_stack(
        [train_sample.pickup_longitude.values, train_sample.pickup_latitude.values]
    ).astype(np.float32)
)
train_sample["dropoff_neighborhood"] = kmeans.predict(
    np.column_stack(
        [train_sample.dropoff_longitude.values, train_sample.dropoff_latitude.values]
    ).astype(np.float32)
)

test["pickup_neighborhood"] = kmeans.predict(
    np.column_stack([test.pickup_longitude.values, test.pickup_latitude.values]).astype(
        np.float32
    )
)
test["dropoff_neighborhood"] = kmeans.predict(
    np.column_stack(
        [test.dropoff_longitude.values, test.dropoff_latitude.values]
    ).astype(np.float32)
)

train_sample["pickup_neighborhood"] = train_sample.pickup_neighborhood.astype(
    dtype="uint8"
)
train_sample["dropoff_neighborhood"] = train_sample.dropoff_neighborhood.astype(
    dtype="uint8"
)
test["pickup_neighborhood"] = test.pickup_neighborhood.astype(dtype="uint8")
test["dropoff_neighborhood"] = test.dropoff_neighborhood.astype(dtype="uint8")




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3032901992.py in <cell line: 0>()
----> 1 train_sample["pickup_neighborhood"] = kmeans.predict(
      2     np.column_stack(
      3         [train_sample.pickup_longitude.values, train_sample.pickup_latitude.values]
      4     ).astype(np.float32)
      5 )

NameError: name 'kmeans' is not defined

## === cell 15
train_sample.head()




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/54140371.py in <cell line: 0>()
----> 1 train_sample.head()
      2 
      3 

NameError: name 'train_sample' is not defined

## === cell 16
with open("kmeans_200_round4_v2.pkl", "wb") as fid:
    pickle.dump(kmeans, fid)

print("Saved kmeans_200_round4_v2.pkl")




## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3363583878.py in <cell line: 0>()
      1 with open("kmeans_200_round4_v2.pkl", "wb") as fid:
----> 2     pickle.dump(kmeans, fid)
      3 
      4 print("Saved kmeans_200_round4_v2.pkl")
      5 

NameError: name 'kmeans' is not defined

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




## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3133474635.py in <cell line: 0>()
     12 numerical_cols = ["distance", "AWND", "PRCP", "SNOW"]
     13 
---> 14 missing_train = [
     15     c
     16     for c in categorical_cols + numerical_cols + ["fare_amount"]

/tmp/ipykernel_11/3133474635.py in <listcomp>(.0)
     15     c
     16     for c in categorical_cols + numerical_cols + ["fare_amount"]
---> 17     if c not in train_sample.columns
     18 ]
     19 missing_test = [c for c in categorical_cols + numerical_cols if c not in test.columns]

NameError: name 'train_sample' is not defined

## === cell 18
X_cats = train_sample[categorical_cols]
X_cats_test = test[categorical_cols]
X_cats_full = pd.concat([X_cats, X_cats_test], axis=0)

ohe = OneHotEncoder(categories="auto", handle_unknown="ignore")
X_onehot = ohe.fit_transform(X_cats_full)
del X_cats, X_cats_test, X_cats_full

X_nums = train_sample[numerical_cols].to_numpy()
X_nums_test = test[numerical_cols].to_numpy()
X_nums_full = np.vstack([X_nums, X_nums_test])
del X_nums, X_nums_test

X_nums_full = np.nan_to_num(X_nums_full, nan=0.0, posinf=0.0, neginf=0.0)
X_nums_sparse = csr_matrix(X_nums_full)
del X_nums_full

X_full = hstack([X_onehot, X_nums_sparse], format="csr")
del X_onehot, X_nums_sparse

X = X_full[: train_sample.shape[0], :]
X_public = X_full[train_sample.shape[0] :, :]

y = train_sample.fare_amount.values
del X_full

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=17
)

print("X_train:", X_train.shape, "X_test:", X_test.shape, "X_public:", X_public.shape)




## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1623545934.py in <cell line: 0>()
      1 # Speed: avoid np.append (makes dense copies); use sparse vstack/hstack directly.
----> 2 X_cats = train_sample[categorical_cols]
      3 X_cats_test = test[categorical_cols]
      4 X_cats_full = pd.concat([X_cats, X_cats_test], axis=0)
      5 

NameError: name 'train_sample' is not defined

## === cell 19
print("y stats:", float(np.nanmin(y)), float(np.nanmean(y)), float(np.nanmax(y)))




## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2741740160.py in <cell line: 0>()
----> 1 print("y stats:", float(np.nanmin(y)), float(np.nanmean(y)), float(np.nanmax(y)))
      2 
      3 

NameError: name 'y' is not defined

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
}

d_train = lgb.Dataset(X_train, label=y_train, free_raw_data=True)
d_test = lgb.Dataset(X_test, label=y_test, free_raw_data=True)
watchlist = [d_train, d_test]

num_rounds = 2000
verbose_eval = 200
early_stop = 200

model_lgb = lgb.train(
    params,
    train_set=d_train,
    num_boost_round=num_rounds,
    valid_sets=watchlist,
    callbacks=[
        lgb.early_stopping(early_stop, verbose=False),
        lgb.log_evaluation(period=verbose_eval),
    ],
)

pred_test_y_lgb = model_lgb.predict(X_test, num_iteration=model_lgb.best_iteration)
print("LGB Loss = " + str(sqrt(mean_squared_error(y_test, pred_test_y_lgb))))




## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3511555399.py in <cell line: 0>()
     23 }
     24 
---> 25 d_train = lgb.Dataset(X_train, label=y_train, free_raw_data=True)
     26 d_test = lgb.Dataset(X_test, label=y_test, free_raw_data=True)
     27 watchlist = [d_train, d_test]

NameError: name 'X_train' is not defined

## === cell 21
del X_train, X_test, y_train, y_test, d_train, d_test, watchlist




## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2680121494.py in <cell line: 0>()
----> 1 del X_train, X_test, y_train, y_test, d_train, d_test, watchlist
      2 
      3 

NameError: name 'X_train' is not defined

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




## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2813235900.py in <cell line: 0>()
----> 1 lgb_public = model_lgb.predict(X_public, num_iteration=model_lgb.best_iteration)
      2 final_pred_public = lgb_public.flatten()
      3 
      4 # Speed: vectorize post-processing, then convert once.
      5 final_pred_public = np.where(final_pred_public > 0, final_pred_public, 0.0).astype(

NameError: name 'model_lgb' is not defined

## === cell 23
plt.rc("figure", figsize=(10, 10))
plt.hist(final_pred_public, bins=100)
plt.xlabel("Predicticted Price")
plt.ylabel("Frequency")
plt.title("Predictions from LGB")
plt.show()

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3726066384.py in <cell line: 0>()
      1 # Speed: plotting is not required for submission; keep minimal to avoid timeouts.
      2 plt.rc("figure", figsize=(10, 10))
----> 3 plt.hist(final_pred_public, bins=100)
      4 plt.xlabel("Predicticted Price")
      5 plt.ylabel("Frequency")

NameError: name 'final_pred_public' is not defined
