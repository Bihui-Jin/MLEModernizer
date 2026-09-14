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

3.9

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

# 5. Target score

5.68923

# 6. Current score

865.04477

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 936.9245) has done: 'I fix the scikit-learn runtime error by removing the deprecated `normalize=True` argument from `LinearRegression` (it was removed in newer sklearn versions), which currently prevents the model from training and blocks submission generation. I keep the same model and feature logic, but add a minimal, equivalent preprocessing step using a `Pipeline` with `StandardScaler` so the intent of normalization is preserved without changing the learning algorithm. I also ensure train/test one-hot columns are aligned (to prevent silent column mismatch issues) and that the final submission is written as a valid CSV with the required header `key,fare_amount`. These changes are correctness/stability focused and should yield a reasonable RMSE closer to the target band while preserving the core approach.'
- What this solution (achieved 752.76715) has done: 'Your score is catastrophically worse than the target (936.9 vs 5.69, lower is better), which strongly suggests a correctness bug rather than a “model quality” issue. The biggest issue is that you normalize `Difference_longitude/latitude` using the *test-set* mean/variance (and separately from train), which makes train/test features live on different scales and breaks the learned mapping. I make a minimal fix: compute these normalization statistics on the training set once, apply them to both train and test (same feature semantics), and keep everything else (feature engineering, LinearRegression+StandardScaler pipeline, train/test split, rounding, CSV format) unchanged. This should move RMSE dramatically toward the target band without changing the core approach.'
- What this solution (achieved 941.293) has done: 'Your RMSE is far above the target, which indicates remaining data-quality issues rather than model capacity. With minimal changes, I (1) filter clearly invalid training labels (non-positive / extreme fares) and (2) filter clearly invalid coordinates/passenger_count using standard NYC bounds so the linear model isn’t dominated by garbage rows. I also apply the same coordinate validity filtering to the test set (without dropping rows) by clipping passenger_count and leaving coordinates as-is, to avoid misalignment while keeping features in a sane range. Core feature engineering + LinearRegression+StandardScaler pipeline, split, and submission semantics remain unchanged.'
- What this solution (achieved 865.04477) has done: 'Your RMSE is still far above target, so this is almost certainly a data/feature correctness issue rather than model capacity. With minimal changes that preserve your exact model and features, I (1) fix the `Difference_*` “normalization” bug (you divide by variance instead of standard deviation, which badly distorts scale), (2) switch the time parsing to robust datetime extraction (same semantic feature: pickup HHMM and weekday, but without fragile string slicing), and (3) clip negative predictions to 0 (fares can’t be negative; this reduces extreme-error outliers under RMSE). Everything else—feature set, LinearRegression in a StandardScaler Pipeline, train/test alignment, and submission format—remains the same.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # CSV file I/O (e.g. pd.read_csv)
import os  # reading the input files we have access to

INPUT_DIR_CANDIDATES = [
    "../input",
    "/kaggle/input",
    "/kaggle/data",
    "/kaggle/input/new-york-city-taxi-fare-prediction",
]
INPUT_DIR = None
for p in INPUT_DIR_CANDIDATES:
    if os.path.exists(p):
        INPUT_DIR = p
        break

print("Detected INPUT_DIR:", INPUT_DIR)
if INPUT_DIR is not None:
    print("Top-level listing:", os.listdir(INPUT_DIR)[:20])




## === cell 1
def resolve_path(filename: str) -> str:
    direct = os.path.join(INPUT_DIR, filename)
    nested = os.path.join(INPUT_DIR, "new-york-city-taxi-fare-prediction", filename)
    if INPUT_DIR is None:
        return filename
    if os.path.exists(direct):
        return direct
    if os.path.exists(nested):
        return nested
    return direct


TRAIN_PATH = resolve_path("train.csv")
TEST_PATH = resolve_path("test.csv")
SAMPLE_PATH = resolve_path("sample_submission.csv")

train_data = pd.read_csv(TRAIN_PATH, nrows=10_000_000)
train_data.dtypes



## === cell 2
test_data = pd.read_csv(TEST_PATH)
test_data.head()



## === cell 3
train_data["Difference_longitude"] = np.abs(
    np.asarray(train_data["pickup_longitude"] - train_data["dropoff_longitude"])
)
train_data["Difference_latitude"] = np.abs(
    np.asarray(train_data["pickup_latitude"] - train_data["dropoff_latitude"])
)

test_data["Difference_longitude"] = np.abs(
    np.asarray(test_data["pickup_longitude"] - test_data["dropoff_longitude"])
)
test_data["Difference_latitude"] = np.abs(
    np.asarray(test_data["pickup_latitude"] - test_data["dropoff_latitude"])
)



## === cell 4
print(train_data.isnull().sum())



## === cell 5
print("Old size: %d" % len(train_data))
train_data = train_data.dropna(how="any", axis="rows")
print("New size: %d" % len(train_data))



## === cell 6
try:
    _ = train_data[:2000].plot.scatter("Difference_longitude", "Difference_latitude")
except Exception as e:
    print("Plot skipped:", repr(e))



## === cell 7
print("Old size: %d" % len(train_data))
train_data = train_data[
    (train_data.Difference_longitude < 5.0) & (train_data.Difference_latitude < 5.0)
]
print("New size: %d" % len(train_data))



## === cell 8
fare_mask = (train_data["fare_amount"] > 0) & (train_data["fare_amount"] <= 250)

coord_mask = (
    train_data["pickup_longitude"].between(-75, -72)
    & train_data["dropoff_longitude"].between(-75, -72)
    & train_data["pickup_latitude"].between(40, 42)
    & train_data["dropoff_latitude"].between(40, 42)
)

pax_mask = train_data["passenger_count"].between(1, 6)

print("Old size (pre-clean): %d" % len(train_data))
train_data = train_data[fare_mask & coord_mask & pax_mask].copy()
print("New size (post-clean): %d" % len(train_data))

test_data["passenger_count"] = pd.to_numeric(
    test_data["passenger_count"], errors="coerce"
).fillna(1)
test_data["passenger_count"] = (
    test_data["passenger_count"].clip(lower=1, upper=6).astype(int)
)



## === cell 9
train_dt = pd.to_datetime(train_data["pickup_datetime"], errors="coerce", utc=True)
test_dt = pd.to_datetime(test_data["pickup_datetime"], errors="coerce", utc=True)

train_data["pickuptime"] = (
    train_dt.dt.hour.fillna(0).astype(int) * 100
    + train_dt.dt.minute.fillna(0).astype(int)
).astype(int)
test_data["pickuptime"] = (
    test_dt.dt.hour.fillna(0).astype(int) * 100
    + test_dt.dt.minute.fillna(0).astype(int)
).astype(int)

train_data["Weekday"] = train_dt.dt.weekday.fillna(0).astype(int)
test_data["Weekday"] = test_dt.dt.weekday.fillna(0).astype(int)



## === cell 10
train_data.head()



## === cell 11
train_data.drop("pickup_datetime", inplace=True, axis=1)
test_data.drop("pickup_datetime", inplace=True, axis=1)



## === cell 12
train_data["Weekday"].replace(
    to_replace=[i for i in range(0, 7)],
    value=[
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ],
    inplace=True,
)
test_data["Weekday"].replace(
    to_replace=[i for i in range(0, 7)],
    value=[
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ],
    inplace=True,
)



## === cell 13
train_data.head()



## === cell 14
train_one_hot = pd.get_dummies(train_data["Weekday"])
test_one_hot = pd.get_dummies(test_data["Weekday"])

train_data = pd.concat([train_data, train_one_hot], axis=1)
test_data = pd.concat([test_data, test_one_hot], axis=1)

missing_in_test = [c for c in train_one_hot.columns if c not in test_data.columns]
for c in missing_in_test:
    test_data[c] = 0
missing_in_train = [c for c in test_one_hot.columns if c not in train_data.columns]
for c in missing_in_train:
    train_data[c] = 0

dummy_cols = sorted(set(train_one_hot.columns).union(set(test_one_hot.columns)))



## === cell 15
train_data.drop("Weekday", axis=1, inplace=True)
test_data.drop("Weekday", axis=1, inplace=True)



## === cell 16
R = 6373.0  # radius of earth
lat1 = np.asarray(np.radians(train_data["pickup_latitude"]))
lon1 = np.asarray(np.radians(train_data["pickup_longitude"]))
lat2 = np.asarray(np.radians(train_data["dropoff_latitude"]))
lon2 = np.asarray(np.radians(train_data["dropoff_longitude"]))

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
train_data["Distance"] = np.asarray(distance) * 0.621

lat1 = np.asarray(np.radians(test_data["pickup_latitude"]))
lon1 = np.asarray(np.radians(test_data["pickup_longitude"]))
lat2 = np.asarray(np.radians(test_data["dropoff_latitude"]))
lon2 = np.asarray(np.radians(test_data["dropoff_longitude"]))

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
test_data["Distance"] = np.asarray(distance) * 0.621



## === cell 17
R = 6373.0
lat1 = np.asarray(np.radians(train_data["pickup_latitude"]))
lon1 = np.asarray(np.radians(train_data["pickup_longitude"]))
lat2 = np.asarray(np.radians(train_data["dropoff_latitude"]))
lon2 = np.asarray(np.radians(train_data["dropoff_longitude"]))

lat3 = np.zeros(len(train_data)) + np.radians(40.6413111)  # latitude of jfk airport
lon3 = np.zeros(len(train_data)) + np.radians(-73.7781391)  # longitude of jfk airport
dlon_pickup = lon3 - lon1
dlat_pickup = lat3 - lat1
d_lon_dropoff = lon3 - lon2
d_lat_dropoff = lat3 - lat2
a1 = (
    np.sin(dlat_pickup / 2) ** 2
    + np.cos(lat1) * np.cos(lat3) * np.sin(dlon_pickup / 2) ** 2
)
c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
distance1 = R * c1
train_data["Pickup_Distance_airport"] = np.asarray(distance1) * 0.621

a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
train_data["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621

lat1 = np.asarray(np.radians(test_data["pickup_latitude"]))
lon1 = np.asarray(np.radians(test_data["pickup_longitude"]))
lat2 = np.asarray(np.radians(test_data["dropoff_latitude"]))
lon2 = np.asarray(np.radians(test_data["dropoff_longitude"]))

lat3 = np.zeros(len(test_data)) + np.radians(40.6413111)
lon3 = np.zeros(len(test_data)) + np.radians(-73.7781391)
dlon_pickup = lon3 - lon1
dlat_pickup = lat3 - lat1
d_lon_dropoff = lon3 - lon2
d_lat_dropoff = lat3 - lat2
a1 = (
    np.sin(dlat_pickup / 2) ** 2
    + np.cos(lat1) * np.cos(lat3) * np.sin(dlon_pickup / 2) ** 2
)
c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
distance1 = R * c1
test_data["Pickup_Distance_airport"] = np.asarray(distance1) * 0.621

a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
test_data["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621



## === cell 18
train_data["Distance"] = np.round(train_data["Distance"], 2)
train_data["Pickup_Distance_airport"] = np.round(
    train_data["Pickup_Distance_airport"], 2
)
train_data["Dropoff_Distance_airport"] = np.round(
    train_data["Dropoff_Distance_airport"], 2
)
test_data["Distance"] = np.round(test_data["Distance"], 2)
test_data["Pickup_Distance_airport"] = np.round(test_data["Pickup_Distance_airport"], 2)
test_data["Dropoff_Distance_airport"] = np.round(
    test_data["Dropoff_Distance_airport"], 2
)



## === cell 19
train_data.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)
test_data.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)



## === cell 20
eps = 1e-12

dl_mean = float(train_data["Difference_longitude"].mean())
dl_std = float(train_data["Difference_longitude"].std(ddof=1))
dlat_mean = float(train_data["Difference_latitude"].mean())
dlat_std = float(train_data["Difference_latitude"].std(ddof=1))

dl_std = max(dl_std, eps)
dlat_std = max(dlat_std, eps)

train_data["Difference_longitude"] = (
    np.abs(train_data["Difference_longitude"] - dl_mean) / dl_std
)
train_data["Difference_latitude"] = (
    np.abs(train_data["Difference_latitude"] - dlat_mean) / dlat_std
)

test_data["Difference_longitude"] = (
    np.abs(test_data["Difference_longitude"] - dl_mean) / dl_std
)
test_data["Difference_latitude"] = (
    np.abs(test_data["Difference_latitude"] - dlat_mean) / dlat_std
)



## === cell 21
from sklearn.model_selection import train_test_split

X = train_data.drop(["key", "fare_amount"], axis=1)
y = train_data["fare_amount"]

test_X = test_data.drop("key", axis=1)
test_X = test_X.reindex(columns=X.columns, fill_value=0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.01, random_state=80
)



## === cell 22
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

lr = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", LinearRegression()),
    ]
)

lr.fit(X_train, y_train)
print(lr.score(X_test, y_test))



## === cell 23
pred = lr.predict(test_X)
pred = np.clip(pred, 0, None)
pred = np.round(pred, 2)



## === cell 24
Submission = pd.DataFrame(data=pred, columns=["fare_amount"])
Submission["key"] = test_data["key"].values
Submission = Submission[["key", "fare_amount"]]

Submission.to_csv("Submission.csv", index=False)
print("Wrote Submission.csv with shape:", Submission.shape)
print(Submission.head())
