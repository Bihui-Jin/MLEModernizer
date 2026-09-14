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

3.8

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

5.68915

# 6. Current score

15.25184

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 936.92806) has done: 'Diagnosis: The crash happens because `LinearRegression(normalize=True)` is no longer a valid argument in scikit-learn 1.2.2; the `normalize` parameter was deprecated and then removed, so passing it raises a `TypeError`. The rest of the code expects `lr` to be a fitted `LinearRegression` instance that supports `.score()` and (in cell 32) `.predict()`.

Patch summary: Remove the unsupported `normalize=True` argument and instantiate `LinearRegression()` with default parameters, keeping the same training/prediction semantics and ensuring `lr` exists for cell 32.

Updated cells: Only cell 31 is changed.

Compatibility notes for cell k+1: `lr` remains a `LinearRegression` model fit on `X_train, y_train`, so `lr.predict(...)` in cell 32 continues to work unchanged.

Assumptions: No other scikit-learn API incompatibilities are present in later cells, and feature columns in `test_data.drop('key', axis=1)` match those used during fitting.'
- What this solution (achieved 865.37662) has done: 'Your current score (936) suggests the submission is severely miscalibrated (often caused by extreme/negative predictions from unfiltered noisy training rows) rather than a small modeling deficiency. I keep your exact model (LinearRegression) and feature logic, but add minimal, standard NYC Taxi Fare data cleaning (valid fare range, passenger_count bounds, and NYC lat/lon bounding box) to remove obvious corrupt rows that dominate the fit and explode RMSE. I also make the train/test one-hot columns consistent (so train and test have identical feature columns) and apply your normalization parameters from train to test (instead of normalizing test using its own mean/variance), both of which preserve semantics but prevent distribution/feature mismatches. Finally, I clip negative predictions to 0 (fares can’t be negative) to avoid catastrophic RMSE outliers while still keeping the same regression model.'
- What this solution (achieved 865.37662) has done: 'Your RMSE is still catastrophically high, which usually means the submission `key` order is misaligned with predictions rather than the model being slightly weak. I keep your exact feature engineering and `LinearRegression` core logic, but ensure train/test one-hot columns are perfectly aligned and, critically, write the submission in the exact required format without setting `key` as the index (Kaggle expects `key` as a column in the same row order as `test.csv`). I also add a small sanity check that the submission keys match the test keys row-by-row to prevent silent misalignment. These changes should dramatically reduce RMSE toward the target band without changing the modeling approach.'
- What this solution (achieved 865.37662) has done: 'Your current RMSE is still far from the target, so the most likely remaining issue is feature-column/order mismatch between what the model is trained on (`X_train`) and what you predict on (`test_data.drop("key")`). I make a minimal fix to enforce identical feature columns *and identical column order* for train vs test by building `feature_cols` from `X.columns`, then reindexing `test_data` to those columns (and dropping any extras), so `lr.predict()` sees exactly what it expects. I also add a small guard to ensure `test_data` still contains `key` after alignment and to keep the submission row order identical to the original `test.csv`. No model/feature logic is changed; this is purely alignment correctness to reduce catastrophic RMSE.'
- What this solution (achieved 865.37662) has done: 'Your RMSE is still so large that it strongly suggests a remaining train/test schema mismatch: `get_dummies()` can create different weekday columns in train vs test, and right now you only add *missing* features to test but don’t drop *extra* features or ensure deterministic ordering at the point of model fitting. I make a minimal fix to explicitly align the one-hot weekday columns **immediately after** dummy creation (so both frames have identical columns from that point onward), which prevents silent feature drift later. I also enforce numeric dtypes and fill any remaining NaNs/inf in `X`/`test_X` just before fitting/predicting (LinearRegression can behave badly otherwise), while keeping the exact same model and feature logic. These are small correctness/stability changes aimed at drastically reducing RMSE toward your target without changing the core approach.'
- What this solution (achieved 811.55538) has done: 'Your RMSE is still far from the target, and with the model/feature set you’re using that typically indicates a remaining data-quality problem rather than small modeling weakness. I keep your exact LinearRegression training and the same feature engineering, but add one minimal, standard cleaning step: remove rows where pickup and dropoff coordinates are identical (distance ~ 0), which are common corrupted/invalid records that heavily distort the fit. I also clamp `passenger_count` in the test set to the same valid range you trained on (1–6) to avoid out-of-distribution inputs at prediction time. These changes should move RMSE substantially downward while preserving your core approach and producing the same `submission.csv` format.'
- What this solution (achieved 17.04693) has done: 'Your RMSE is still catastrophically high versus the target, which strongly suggests there are still many extreme/invalid labels leaking into training (even after your current filters) and/or a remaining distribution mismatch on `fare_amount`. Without changing your model (LinearRegression) or feature set, I add a minimal but standard NYC Taxi clean-up: filter out obvious outlier fares relative to trip distance (e.g., very high $/mile and very low $/mile) and remove extreme distances, which are common corrupt records that make linear regression explode. I also add a very small “reasonable cap” to predictions (e.g., 0–250) consistent with your training label filter to avoid a few huge predictions dominating RMSE. These are simple row filters/post-processing that preserve your core approach but should move RMSE substantially toward the target band.'
- What this solution (achieved 29.02081) has done: 'Your current RMSE (17.05) is still far above the target (5.69), so we should improve legitimately without changing the model or feature set. The biggest remaining lever consistent with your constraints is fixing the scaling bug in your “normalization”: you divide by variance instead of standard deviation, which badly distorts feature magnitudes and hurts linear regression. I change the normalization to use `std` (and keep applying train-derived stats to test), with a tiny epsilon guard against divide-by-zero. Everything else (data loading, cleaning, feature engineering, LinearRegression fit/predict, submission format) remains the same.'
- What this solution (achieved 29.02081) has done: 'Your RMSE is still far above the target, and with your current pipeline the most likely remaining cause is a brittle datetime parsing that creates wrong `Weekday` / `pickuptime` values for many rows (string-slicing `pickup_datetime` is easy to get subtly wrong), which then poisons the linear regression fit. I keep the exact same core model (LinearRegression) and the same features you already use (weekday one-hot + pickuptime + coordinate diffs + distances), but switch the datetime feature extraction to a vectorized `pd.to_datetime(...)` path that reliably handles the provided format. This should legitimately reduce error without changing the architecture/training approach, and it remains within the 600s budget because it replaces Python loops with fast pandas ops. Everything else (cleaning, scaling with train stats, column alignment, clipping, and submission formatting) is kept the same.'
- What this solution (achieved 27.25865) has done: 'Your RMSE is still far above the target, so we should make a small, legitimate improvement that doesn’t change your model or feature set: the current train split is random, which can leak “future” patterns and destabilize a linear model on this dataset; switching to a time-based split (by pickup_datetime) typically improves generalization and reduces RMSE without changing architecture or training semantics. To keep features identical, we use the parsed `train_dt` already computed to build a deterministic chronological split and then fit/predict exactly as before. Additionally, we apply the same (already-used) standardization approach to the three distance-based features as well (Distance + airport distances), using train-derived mean/std and applying to test; this is still the same core feature set, just consistent scaling that helps LinearRegression coefficients behave sensibly. Submission writing and key alignment remain unchanged.'
- What this solution (achieved 36.72115) has done: 'Your RMSE (27.26) is far above the target (5.69), so the most likely issue is still training noise/outliers that a plain LinearRegression can’t handle well. Keeping your exact model and feature set, I add one minimal, standard NYC Taxi cleanup step that’s directly tied to the metric: remove extreme residual-outlier trips by filtering on very large implied speed (Distance vs. trip time) using the already-available `pickup_datetime` and parsed datetimes. This preserves the same architecture and features (we won’t add new columns to X), but it should reduce label noise and bring RMSE down materially. I also ensure the time-based split uses a stable sort and that the speed filter is applied only where we can compute a positive duration, so it doesn’t accidentally drop good rows.'
- What this solution (achieved 15.25184) has done: 'Diagnosis: Cell 38 crashes because after the earlier filtering/cleaning steps, the training set can become empty (0 rows). The current guard only checks `X_train.shape[0] == 0`, but still tries to fit on `X_sorted` which is also empty, triggering scikit-learn’s “minimum of 1 sample required” error.  
Patch summary: Add a minimal emptiness check before fitting: if `X_sorted` has 0 rows, fit a constant “dummy” LinearRegression model on a single zero-feature row with target 0.0 so that `lr.predict(test_X)` in cell 39 still works deterministically. Keep the existing training logic unchanged for the normal (non-empty) case.  
Updated cells: Only cell 38 is modified.  
Compatibility notes for cell k+1: `lr` remains a fitted `LinearRegression` instance, so `pred = lr.predict(test_X)` continues to work with the same interface and output shape.  
Assumptions: If the cleaned dataset is empty, predicting a constant 0 fare is an acceptable fallback to avoid crashing (and preserves the pipeline’s semantics of producing predictions without altering other feature logic).'
- What this solution (achieved 15.25184) has done: 'Your RMSE is still far above the target (lower is better), so we should make a small, legitimate improvement that doesn’t change the core model or feature set: add robust weighting so the linear regression fit is less dominated by the remaining noisy/outlier trips that slip through filters. I keep the same `LinearRegression` and the same features, but switch the fit call to use `sample_weight` based on trip distance and passenger_count (downweight very long trips and unusually high passenger counts), which is a minimal training-semantics tweak and tends to materially reduce RMSE for this dataset. I also make the time-based split slightly less extreme (99%/1% can be unstable) to 95%/5% to get a more reliable fit/validation behavior without changing the approach. Submission writing, key alignment, scaling, and clipping remain unchanged.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
train_data = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", nrows=10_000_000
)
train_data.head()



## === cell 2
train_data.shape



## === cell 3
train_data.info()



## === cell 4
test_data = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")
test_data.head()



## === cell 5
test_data.info()



## === cell 6
train_data.isna().sum()



## === cell 7
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



## === cell 8
print(f"Before Dropping null values: {len(train_data)}")
train_data.dropna(inplace=True)
print(f"After Dropping null values: {len(train_data)}")



## === cell 9
plot = train_data[:2000].plot.scatter("Difference_longitude", "Difference_latitude")



## === cell 10
train_data = train_data[
    (train_data["Difference_longitude"] < 5.0)
    & (train_data["Difference_latitude"] < 5.0)
]



## === cell 11
train_data = train_data[
    (train_data["fare_amount"] > 0) & (train_data["fare_amount"] <= 250)
]
train_data = train_data[
    (train_data["passenger_count"] >= 1) & (train_data["passenger_count"] <= 6)
]

train_data = train_data[
    (train_data["pickup_longitude"].between(-74.3, -72.9))
    & (train_data["dropoff_longitude"].between(-74.3, -72.9))
    & (train_data["pickup_latitude"].between(40.5, 41.8))
    & (train_data["dropoff_latitude"].between(40.5, 41.8))
].copy()



## === cell 12
train_data = train_data[
    ~(
        (train_data["pickup_longitude"] == train_data["dropoff_longitude"])
        & (train_data["pickup_latitude"] == train_data["dropoff_latitude"])
    )
].copy()



## === cell 13
test_data["passenger_count"] = pd.to_numeric(
    test_data["passenger_count"], errors="coerce"
)
test_data["passenger_count"] = (
    test_data["passenger_count"].clip(lower=1, upper=6).fillna(1)
)



## === cell 14
train_dt = pd.to_datetime(train_data["pickup_datetime"], errors="coerce", utc=True)
test_dt = pd.to_datetime(test_data["pickup_datetime"], errors="coerce", utc=True)

train_dt = train_dt.fillna(pd.Timestamp("2009-01-01", tz="UTC"))
test_dt = test_dt.fillna(pd.Timestamp("2009-01-01", tz="UTC"))

train_data["Weekday"] = train_dt.dt.weekday.astype(np.int8)
test_data["Weekday"] = test_dt.dt.weekday.astype(np.int8)

train_data["pickuptime"] = (train_dt.dt.hour * 100 + train_dt.dt.minute).astype(
    np.int16
)
test_data["pickuptime"] = (test_dt.dt.hour * 100 + test_dt.dt.minute).astype(np.int16)



## === cell 15
train_data.head()



## === cell 16
test_data.head()



## === cell 17
train_data.drop("pickup_datetime", inplace=True, axis=1)
test_data.drop("pickup_datetime", inplace=True, axis=1)



## === cell 18
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



## === cell 19
all_days = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
train_data["Weekday"] = pd.Categorical(train_data["Weekday"], categories=all_days)
test_data["Weekday"] = pd.Categorical(test_data["Weekday"], categories=all_days)

train_one_hot = pd.get_dummies(train_data["Weekday"])
test_one_hot = pd.get_dummies(test_data["Weekday"])

train_one_hot, test_one_hot = train_one_hot.align(
    test_one_hot, join="outer", axis=1, fill_value=0
)

train_data = pd.concat([train_data, train_one_hot], axis=1)
test_data = pd.concat([test_data, test_one_hot], axis=1)



## === cell 20
train_data.drop("Weekday", axis=1, inplace=True)
test_data.drop("Weekday", axis=1, inplace=True)



## === cell 21
train_data["pickuptime"] = (
    pd.to_numeric(train_data["pickuptime"], errors="coerce").fillna(0).astype(int)
)
test_data["pickuptime"] = (
    pd.to_numeric(test_data["pickuptime"], errors="coerce").fillna(0).astype(int)
)



## === cell 22
train_data.head()



## === cell 23
R = 6373.0
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



## === cell 24
R = 6373.0
lat1 = np.asarray(np.radians(train_data["pickup_latitude"]))
lon1 = np.asarray(np.radians(train_data["pickup_longitude"]))
lat2 = np.asarray(np.radians(train_data["dropoff_latitude"]))
lon2 = np.asarray(np.radians(train_data["dropoff_longitude"]))

lat3 = np.zeros(len(train_data)) + np.radians(40.6413111)
lon3 = np.zeros(len(train_data)) + np.radians(-73.7781391)
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



## === cell 25
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



## === cell 26
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



## === cell 27
train_data = train_data[
    (train_data["Distance"] > 0.05) & (train_data["Distance"] < 60)
].copy()
fare_per_mile = train_data["fare_amount"] / train_data["Distance"]
train_data = train_data[fare_per_mile.between(0.5, 50)].copy()



## === cell 28
try:
    pt_bin = (train_data["pickuptime"].astype(np.int32) // 50).clip(lower=0, upper=47)
    med_dist_by_bin = train_data.groupby(pt_bin)["Distance"].median()
    typical = pt_bin.map(med_dist_by_bin).astype(float)

    ratio = train_data["Distance"] / (typical.replace(0, np.nan))
    mask = ratio.isna() | ratio.between(0.05, 25.0)
    train_data = train_data.loc[mask].copy()
except Exception:
    pass



## === cell 29
try:
    drop_dt = pd.to_datetime(
        train_data["key"].astype(str).str.slice(0, 26),
        errors="coerce",
        utc=True,
    )
    pick_dt = train_dt.loc[train_data.index]

    dur_sec = (drop_dt - pick_dt).dt.total_seconds()
    dur_hr = dur_sec / 3600.0

    mph = train_data["Distance"] / dur_hr

    mask = (
        dur_sec.notna()
        & (dur_sec > 0)
        & (dur_sec <= 4 * 3600)
        & mph.between(0.1, 100.0)
    )
    train_data = train_data.loc[mask].copy()

    train_dt = pick_dt.loc[train_data.index]
except Exception:
    pass



## === cell 30
_eps = 1e-12
dl_mean = np.mean(train_data["Difference_longitude"])
dl_std = np.std(train_data["Difference_longitude"])
dl_std = dl_std if dl_std > _eps else 1.0

train_data["Difference_longitude"] = train_data["Difference_longitude"] - dl_mean
train_data["Difference_longitude"] = train_data["Difference_longitude"] / dl_std



## === cell 31
dlat_mean = np.mean(train_data["Difference_latitude"])
dlat_std = np.std(train_data["Difference_latitude"])
dlat_std = dlat_std if dlat_std > _eps else 1.0

train_data["Difference_latitude"] = train_data["Difference_latitude"] - dlat_mean
train_data["Difference_latitude"] = train_data["Difference_latitude"] / dlat_std



## === cell 32
test_data["Difference_longitude"] = test_data["Difference_longitude"] - dl_mean
test_data["Difference_longitude"] = test_data["Difference_longitude"] / dl_std

test_data["Difference_latitude"] = test_data["Difference_latitude"] - dlat_mean
test_data["Difference_latitude"] = test_data["Difference_latitude"] / dlat_std



## === cell 33
scale_cols = ["Distance", "Pickup_Distance_airport", "Dropoff_Distance_airport"]
scale_stats = {}
for col in scale_cols:
    m = float(np.mean(train_data[col]))
    s = float(np.std(train_data[col]))
    s = s if s > _eps else 1.0
    scale_stats[col] = (m, s)
    train_data[col] = (train_data[col] - m) / s
    test_data[col] = (test_data[col] - m) / s



## === cell 34
train_data.shape



## === cell 35
test_data.shape



## === cell 36
from sklearn.model_selection import train_test_split

X = train_data.drop(["key", "fare_amount"], axis=1)
y = train_data["fare_amount"]

X = (
    X.apply(pd.to_numeric, errors="coerce")
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
)

train_dt_series = train_dt.loc[train_data.index]
order = np.argsort(train_dt_series.values.astype("datetime64[ns]"), kind="mergesort")
X_sorted = X.iloc[order]
y_sorted = y.iloc[order]

split_idx = int(len(X_sorted) * 0.95)
X_train = X_sorted.iloc[:split_idx]
y_train = y_sorted.iloc[:split_idx]
X_test = X_sorted.iloc[split_idx:]
y_test = y_sorted.iloc[split_idx:]



## === cell 37
test_keys = test_data["key"].copy()

test_data_features = test_data.drop(columns=["key"], errors="ignore")
test_data_features = (
    test_data_features.apply(pd.to_numeric, errors="coerce")
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
)

test_X = test_data_features.reindex(columns=list(X.columns), fill_value=0.0)

assert len(test_X.columns) == X.shape[1]
assert test_keys.shape[0] == test_X.shape[0]



## === cell 38
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

if X_sorted.shape[0] == 0:
    lr.fit(np.zeros((1, X.shape[1]), dtype=float), np.array([0.0], dtype=float))
    if X_test.shape[0] > 0:
        print(np.nan)
    else:
        print(np.nan)
elif X_train.shape[0] == 0:
    lr.fit(X_sorted, y_sorted)
    if X_test.shape[0] > 0:
        print(lr.score(X_test, y_test))
    else:
        print(np.nan)
else:
    if "Distance" in X_train.columns:
        dist_raw = X_train["Distance"].to_numpy(dtype=float)
    else:
        dist_raw = np.zeros(X_train.shape[0], dtype=float)

    if "passenger_count" in X_train.columns:
        pc_raw = X_train["passenger_count"].to_numpy(dtype=float)
    else:
        pc_raw = np.ones(X_train.shape[0], dtype=float)

    w_dist = 1.0 / (1.0 + np.abs(dist_raw))
    w_pc = 1.0 / (1.0 + 0.5 * np.maximum(0.0, pc_raw - 1.0))
    sample_weight = (w_dist * w_pc).astype(float)

    lr.fit(X_train, y_train, sample_weight=sample_weight)
    print(lr.score(X_test, y_test))



## === cell 39
pred = lr.predict(test_X)

pred = np.clip(pred, 0, 250)
pred = np.round(pred, 2)



## === cell 40
pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
).head()



## === cell 41
Submission = pd.DataFrame({"key": test_keys.values, "fare_amount": pred})

assert Submission["key"].equals(
    test_keys
), "Key order mismatch: submission misaligned with test rows."

Submission.to_csv("submission.csv", index=False)
print(Submission.head())
print("Wrote submission.csv with shape:", Submission.shape)
