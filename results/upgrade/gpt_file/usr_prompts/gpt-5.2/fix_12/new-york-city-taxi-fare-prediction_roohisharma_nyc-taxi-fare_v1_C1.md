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

218.67476

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 936.9245) has done: 'I fix the scikit-learn runtime error by removing the deprecated `normalize=True` argument from `LinearRegression` (it was removed in newer sklearn versions), which currently prevents the model from training and blocks submission generation. I keep the same model and feature logic, but add a minimal, equivalent preprocessing step using a `Pipeline` with `StandardScaler` so the intent of normalization is preserved without changing the learning algorithm. I also ensure train/test one-hot columns are aligned (to prevent silent column mismatch issues) and that the final submission is written as a valid CSV with the required header `key,fare_amount`. These changes are correctness/stability focused and should yield a reasonable RMSE closer to the target band while preserving the core approach.'
- What this solution (achieved 752.76715) has done: 'Your score is catastrophically worse than the target (936.9 vs 5.69, lower is better), which strongly suggests a correctness bug rather than a “model quality” issue. The biggest issue is that you normalize `Difference_longitude/latitude` using the *test-set* mean/variance (and separately from train), which makes train/test features live on different scales and breaks the learned mapping. I make a minimal fix: compute these normalization statistics on the training set once, apply them to both train and test (same feature semantics), and keep everything else (feature engineering, LinearRegression+StandardScaler pipeline, train/test split, rounding, CSV format) unchanged. This should move RMSE dramatically toward the target band without changing the core approach.'
- What this solution (achieved 941.293) has done: 'Your RMSE is far above the target, which indicates remaining data-quality issues rather than model capacity. With minimal changes, I (1) filter clearly invalid training labels (non-positive / extreme fares) and (2) filter clearly invalid coordinates/passenger_count using standard NYC bounds so the linear model isn’t dominated by garbage rows. I also apply the same coordinate validity filtering to the test set (without dropping rows) by clipping passenger_count and leaving coordinates as-is, to avoid misalignment while keeping features in a sane range. Core feature engineering + LinearRegression+StandardScaler pipeline, split, and submission semantics remain unchanged.'
- What this solution (achieved 865.04477) has done: 'Your RMSE is still far above target, so this is almost certainly a data/feature correctness issue rather than model capacity. With minimal changes that preserve your exact model and features, I (1) fix the `Difference_*` “normalization” bug (you divide by variance instead of standard deviation, which badly distorts scale), (2) switch the time parsing to robust datetime extraction (same semantic feature: pickup HHMM and weekday, but without fragile string slicing), and (3) clip negative predictions to 0 (fares can’t be negative; this reduces extreme-error outliers under RMSE). Everything else—feature set, LinearRegression in a StandardScaler Pipeline, train/test alignment, and submission format—remains the same.'
- What this solution (achieved 865.04477) has done: 'Your RMSE is still orders of magnitude above the target, which strongly suggests the model is being trained on the wrong target values rather than “just needing better features.” The most likely cause in this competition is that `fare_amount` is being read as an `object` (strings like `"4.5"`), so LinearRegression ends up fitting on incorrectly coerced/parsed targets; we force numeric parsing of `fare_amount` early and drop any rows that fail to parse. In the same minimal spirit, we also coerce all feature columns used by the model to numeric (handling any hidden non-numeric parsing issues), and ensure the model never sees NaNs after feature engineering. These changes preserve your exact feature set and model/pipeline, but fix target/feature dtypes so the learned mapping becomes meaningful and RMSE should move dramatically toward the target band.'
- What this solution (achieved 825.40177) has done: 'Your RMSE is still massively worse than the target, so we should focus on a minimal “correctness” fix that can dramatically reduce error without changing the model or feature set. The biggest remaining issue is that `fare_amount` contains extreme outliers (and occasional non-sensical values) that a plain linear regression try to fit, producing wildly wrong predictions on the test set under RMSE; we apply the standard NYC Taxi Fare competition cleaning rule to cap/remove unusually large fares (keep core logic unchanged). We also enforce `passenger_count` numeric/valid on the training set (it’s only cleaned in test currently), and ensure datetime parsing failures don’t silently create degenerate time features. Everything else—feature engineering, StandardScaler+LinearRegression pipeline, split, and submission format—remains the same.'
- What this solution (achieved 956.95941) has done: 'Your RMSE is still orders of magnitude worse than the target, which indicates a remaining *feature correctness* bug rather than model capacity. The smallest high-impact fix is to stop “normalizing” `Difference_longitude/latitude` with an absolute value around the mean (which destroys linear relationships) and instead apply a standard z-score using the *training* mean/std for both train and test. I also remove the premature `(Difference_* < 5)` filter that’s based on raw degree differences (it unintentionally removes a lot of valid NYC trips and skews the distribution), while keeping your existing core cleaning rules (fare/coords/passenger_count), features, and the same `StandardScaler + LinearRegression` pipeline. These changes keep the same model/training approach but should move RMSE dramatically toward the target band by fixing feature semantics.'
- What this solution (achieved 1086.16988) has done: 'Your RMSE is still catastrophically worse than the target, so we should focus on a minimal “correctness” fix rather than model tuning. The biggest remaining issue is that `Difference_longitude/latitude` are built with `np.abs(...)`, which discards direction and breaks the linear relationship with fare (e.g., east-west vs west-east trips become identical); keeping the signed deltas is a small feature-semantics fix that typically yields a huge RMSE improvement while preserving your exact model/pipeline/training loop. I also add a very small guardrail to drop obviously broken rows where distance computations become non-finite, and I keep all other logic (cleaning bounds, time features, one-hot weekdays, StandardScaler+LinearRegression, clipping/rounding, submission format) unchanged. This should move the score dramatically toward the 5.689 target band without changing the core approach.'
- What this solution (achieved 1086.16988) has done: 'Your RMSE is still catastrophically worse than the target (1086 vs 5.69, lower is better), which points to a remaining correctness/alignment issue rather than model capacity. The smallest high-impact fix is to ensure `key` and the feature rows stay perfectly aligned when we fill NaNs in `test_data`: right now `test_data[feature_cols_for_fill] = ...` can inadvertently change row alignment if columns differ, and we also risk leaving any NaNs in `test_X` after reindexing. I (1) build the test feature matrix from the *intersection* of feature columns present in both train/test, (2) fill NaNs directly on `test_X` after reindexing (so the matrix fed to the model is fully numeric and aligned), and (3) ensure the submission uses the original test `key` order without any chance of mismatch. This preserves your exact feature engineering and the same `StandardScaler + LinearRegression` pipeline, but fixes the most likely silent train/test matrix mismatch that can explode RMSE.'
- What this solution (achieved 1086.14817) has done: 'Your RMSE (1086) is so far from the target (5.69) that this is still almost certainly a feature/label correctness issue, not a model-capacity issue. The smallest high-impact fix that preserves your exact model/pipeline is to fix the unit mistake in your haversine features: you use `R=6373.0` (kilometers) but multiply by `0.621` (km→miles should be `0.621371`), which shrinks all distance-derived features and can explode predictions under a linear model. I change only the distance computations to use consistent Earth radius and conversion (or directly use miles), keep all other cleaning/feature logic the same, and ensure we still write a valid `Submission.csv` with `key,fare_amount`. This should move the score dramatically downward toward the target band without changing the training approach or architecture.'
- What this solution (achieved 218.67476) has done: 'Your RMSE is still catastrophically far from the target (1086 vs 5.69, lower is better), which strongly indicates a remaining correctness issue rather than a need for “better modeling.” The smallest high-impact fix while preserving your exact model/pipeline is to stop rounding the distance-based features to 2 decimals before training/predicting: for linear regression, that quantization injects avoidable noise and can blow up errors, especially when combined with scaling. I also keep the same feature set but add a minimal safeguard to clip extreme distance outliers in both train/test (without dropping test rows) so a handful of bad coordinates don’t dominate RMSE. Everything else (data loading, cleaning rules, feature engineering, StandardScaler+LinearRegression, split, prediction clipping, and submission format) stays the same.'

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
train_data["fare_amount"] = pd.to_numeric(train_data["fare_amount"], errors="coerce")

train_data["Difference_longitude"] = np.asarray(
    train_data["pickup_longitude"] - train_data["dropoff_longitude"]
)
train_data["Difference_latitude"] = np.asarray(
    train_data["pickup_latitude"] - train_data["dropoff_latitude"]
)

test_data["Difference_longitude"] = np.asarray(
    test_data["pickup_longitude"] - test_data["dropoff_longitude"]
)
test_data["Difference_latitude"] = np.asarray(
    test_data["pickup_latitude"] - test_data["dropoff_latitude"]
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
print("Skipping raw Difference_* < 5.0 filter to preserve correct data distribution.")



## === cell 8
train_data["passenger_count"] = pd.to_numeric(
    train_data["passenger_count"], errors="coerce"
)

fare_mask = (train_data["fare_amount"] > 0) & (train_data["fare_amount"] <= 100)

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

train_hour = train_dt.dt.hour
train_min = train_dt.dt.minute
test_hour = test_dt.dt.hour
test_min = test_dt.dt.minute

hour_fill = int(train_hour.median(skipna=True)) if train_hour.notna().any() else 0
min_fill = int(train_min.median(skipna=True)) if train_min.notna().any() else 0

train_data["pickuptime"] = (
    train_hour.fillna(hour_fill).astype(int) * 100
    + train_min.fillna(min_fill).astype(int)
).astype(int)
test_data["pickuptime"] = (
    test_hour.fillna(hour_fill).astype(int) * 100
    + test_min.fillna(min_fill).astype(int)
).astype(int)

train_wd = train_dt.dt.weekday
wd_fill = int(train_wd.median(skipna=True)) if train_wd.notna().any() else 0
train_data["Weekday"] = train_wd.fillna(wd_fill).astype(int)
test_data["Weekday"] = test_dt.dt.weekday.fillna(wd_fill).astype(int)



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
R_miles = 3958.7613  # Earth radius in miles (consistent units)

lat1 = np.asarray(np.radians(train_data["pickup_latitude"]))
lon1 = np.asarray(np.radians(train_data["pickup_longitude"]))
lat2 = np.asarray(np.radians(train_data["dropoff_latitude"]))
lon2 = np.asarray(np.radians(train_data["dropoff_longitude"]))

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R_miles * c
train_data["Distance"] = np.asarray(distance)

lat1 = np.asarray(np.radians(test_data["pickup_latitude"]))
lon1 = np.asarray(np.radians(test_data["pickup_longitude"]))
lat2 = np.asarray(np.radians(test_data["dropoff_latitude"]))
lon2 = np.asarray(np.radians(test_data["dropoff_longitude"]))

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R_miles * c
test_data["Distance"] = np.asarray(distance)



## === cell 17
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
distance1 = R_miles * c1
train_data["Pickup_Distance_airport"] = np.asarray(distance1)

a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R_miles * c2
train_data["Dropoff_Distance_airport"] = np.asarray(distance2)

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
distance1 = R_miles * c1
test_data["Pickup_Distance_airport"] = np.asarray(distance1)

a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R_miles * c2
test_data["Dropoff_Distance_airport"] = np.asarray(distance2)



## === cell 18

dist_cols = ["Distance", "Pickup_Distance_airport", "Dropoff_Distance_airport"]
for col in dist_cols:
    train_data[col] = pd.to_numeric(train_data[col], errors="coerce")
    test_data[col] = pd.to_numeric(test_data[col], errors="coerce")

cap = float(train_data["Distance"].quantile(0.999)) if len(train_data) else 100.0
cap = max(cap, 50.0)  # ensure not too small
for col in dist_cols:
    train_data[col] = train_data[col].clip(lower=0, upper=cap)
    test_data[col] = test_data[col].clip(lower=0, upper=cap)



## === cell 19
nonfinite_mask = (
    np.isfinite(train_data["Distance"].to_numpy())
    & np.isfinite(train_data["Pickup_Distance_airport"].to_numpy())
    & np.isfinite(train_data["Dropoff_Distance_airport"].to_numpy())
)
before_nf = len(train_data)
train_data = train_data.loc[nonfinite_mask].copy()
print(f"Dropped non-finite distance rows: {before_nf - len(train_data)}")



## === cell 20
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



## === cell 21
eps = 1e-12
dl_mean = float(train_data["Difference_longitude"].mean())
dl_std = float(train_data["Difference_longitude"].std(ddof=1))
dlat_mean = float(train_data["Difference_latitude"].mean())
dlat_std = float(train_data["Difference_latitude"].std(ddof=1))

dl_std = max(dl_std, eps)
dlat_std = max(dlat_std, eps)

train_data["Difference_longitude"] = (
    train_data["Difference_longitude"] - dl_mean
) / dl_std
train_data["Difference_latitude"] = (
    train_data["Difference_latitude"] - dlat_mean
) / dlat_std

test_data["Difference_longitude"] = (
    test_data["Difference_longitude"] - dl_mean
) / dl_std
test_data["Difference_latitude"] = (
    test_data["Difference_latitude"] - dlat_mean
) / dlat_std



## === cell 22
for c in train_data.columns:
    if c not in ["key"]:
        train_data[c] = pd.to_numeric(train_data[c], errors="coerce")
for c in test_data.columns:
    if c not in ["key"]:
        test_data[c] = pd.to_numeric(test_data[c], errors="coerce")

train_data = train_data.dropna(how="any", axis="rows").copy()



## === cell 23
from sklearn.model_selection import train_test_split

X = train_data.drop(["key", "fare_amount"], axis=1)
y = train_data["fare_amount"]

test_X = test_data.drop("key", axis=1).reindex(columns=X.columns, fill_value=np.nan)

fill_values = X.median(numeric_only=True)
test_X = test_X.fillna(fill_values)
test_X = test_X.fillna(0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.01, random_state=80
)



## === cell 24
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



## === cell 25
pred = lr.predict(test_X)
pred = np.clip(pred, 0, None)
pred = np.round(pred, 2)

Submission = pd.DataFrame({"key": test_data["key"].values, "fare_amount": pred})
Submission.to_csv("Submission.csv", index=False)
print("Wrote Submission.csv with shape:", Submission.shape)
print(Submission.head())
