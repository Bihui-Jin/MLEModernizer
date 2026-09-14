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

9.10831

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 752.78354) has done: 'I fix the runtime error caused by the deprecated `normalize` argument in `LinearRegression` (removed in scikit-learn 1.2+), by replacing it with an equivalent `Pipeline(StandardScaler, LinearRegression)` so the intended normalization behavior is preserved. I also ensure train/test one-hot encoded weekday columns are aligned to avoid feature mismatch at inference, and I standardize the train/test scaling of the “Difference_*” features using train statistics (a correctness fix that should improve RMSE vs scaling test by its own stats). Finally, I make sure the submission is written as a valid CSV with columns exactly `key,fare_amount` and without setting `key` as the index (to match Kaggle’s required format).'
- What this solution (achieved 941.41) has done: 'Your current RMSE (752) indicates the model is being trained on many invalid/outlier rows (e.g., negative/huge fares, impossible coordinates, passenger_count=0), which overwhelms a linear model and destroys generalization. Keeping your exact model and feature logic, the smallest high-impact change is to add standard NYC Taxi Fare competition “sanity filters” on the training data (fare bounds, lat/lon bounds, passenger_count bounds) after dropping NaNs, and to avoid dividing by (near-)zero variance in the Difference_* scaling. These are correctness/data-cleaning fixes that typically move RMSE down by orders of magnitude without changing the learning algorithm. The submission format and inference path stay the same.'
- What this solution (achieved 1148.22788) has done: 'I fix the pandas `Int64` dtype replacement error by mapping the weekday integers to string labels without using an in-place replace on a nullable integer column. I also fix the scikit-learn pipeline crash caused by mixed-type column names (weekday dummy columns are strings, others are not) by forcing all feature column names to strings for both train and test, which allows `StandardScaler` to fit. After that, the model fit successfully, predictions run, and a valid `submission.csv` with columns exactly `key,fare_amount` be written. These changes are score-neutral except that they unblock the intended training/inference and avoid feature-name mismatch errors.'
- What this solution (achieved 1148.22788) has done: 'Your RMSE is far worse than the target, so we should *improve* performance with the smallest changes that don’t alter the model/feature core. The biggest likely issue here is that the training set is being fit on a very noisy 10M-row slice without robustly removing extreme/invalid rows caused by coordinate zeros/out-of-NYC points and missing/invalid datetime-derived features; these outliers can blow up a linear model and lead to huge test errors. I keep your exact feature set and LinearRegression+StandardScaler pipeline, but add a couple of standard “NYC taxi fare” sanity filters (zero coordinates, obvious water/out-of-range points, and invalid pickuptime) and ensure the train/test one-hot encoding stays aligned after any rows are dropped. I also clip negative predictions to 0.0 (a legitimate post-processing for a nonnegative target that often reduces RMSE without changing training).'
- What this solution (achieved 1148.60043) has done: 'Your RMSE is massively worse than the target, so we should improve it with the smallest changes that don’t alter your model or feature set. The biggest correctness issue left is that `pickuptime` is being modeled as a raw HHMM integer (e.g., 5 vs 2355), which makes the linear model treat time as non-cyclical and creates a huge discontinuity at midnight; converting it to “minutes since midnight” preserves the same intent but fixes the scale/ordering. I also add one standard (non-architectural) cleanup filter on the derived airport-distance features to remove rare extreme rows that can dominate a linear regression fit. Everything else (feature engineering, StandardScaler+LinearRegression pipeline, clipping/rounding, and submission format) stays the same.'
- What this solution (achieved 1148.58219) has done: 'Your RMSE is still catastrophically high, which usually means the model is learning a badly mis-scaled feature; here that’s caused by dividing “Difference_*” by the *variance* instead of the *standard deviation* (a units bug that can blow up linear coefficients and predictions). I keep your exact feature set and the same `StandardScaler + LinearRegression` pipeline, but change the Difference scaling to use train standard deviation (and keep train-stat application to test) to restore intended normalization. I also add a tiny safety filter to drop any remaining non-finite engineered features before training (doesn’t change logic, just prevents numerical explosions). Everything else (data filters, distance features, one-hot weekday alignment, clipping/rounding, and writing `submission.csv`) stays the same.'
- What this solution (achieved 1201.94739) has done: 'Your RMSE is still extremely far from the target, which strongly suggests a correctness issue rather than a modeling-capacity issue. The smallest high-impact fix that preserves your exact feature set and LinearRegression pipeline is to ensure train/test use identical preprocessing for *all* engineered numeric columns: right now you manually normalize `Difference_*` and then also StandardScale everything, which can create unstable coefficients when combined with outlier-heavy distributions. I remove the manual `Difference_*` normalization (let the existing `StandardScaler` do the scaling once, consistently across all numeric features) and add a single safety step to drop any remaining non-finite rows after *all* feature engineering. This keeps the core logic (same engineered features, same model, same training loop) while addressing a likely source of numerical blow-ups that drive huge test RMSE.'
- What this solution (achieved 36.72113) has done: 'Your RMSE is catastrophically above target (lower is better), which strongly indicates a correctness/alignment issue rather than model capacity. The smallest high-impact fix that preserves your exact features and LinearRegression+StandardScaler core is to ensure **test rows with invalid pickup_datetime don’t produce NaN weekday dummies/pickuptime**, because that silently creates bad inputs at inference; we impute those test datetime-derived fields from the training distribution (not using labels). We also apply the same “drop non-finite engineered numerics” safety check to the test set (without dropping rows; we fill with train medians), preventing extreme predictions from numerical junk. Finally, we add a light, metric-consistent post-processing clip to a realistic upper bound (e.g., 250) matching your training fare filter, which typically reduces RMSE when the linear model produces rare huge fares.'
- What this solution (achieved 36.72115) has done: 'You’re still far above the target RMSE (lower is better), so we should make small correctness-driven changes that reduce error without changing your model or feature set. The biggest likely culprit is that `LinearRegression` is being fit on a multi-million-row slice with a tiny 1% holdout; plain OLS is very sensitive to residual outliers, so a few bad rows can dominate coefficients even after basic filters. Keeping the exact same features and training flow, we add a standard NYC Taxi Fare cleaning step that removes rows with implausible fare-per-distance (using haversine distance *before rounding*) and we stop rounding distance features (rounding throws away predictive signal and can worsen RMSE). Finally, we ensure test engineered numerics are finite (already done) and keep the same clipping/postprocessing and submission schema.'
- What this solution (achieved 36.72091) has done: 'Your current RMSE (36.72) is still far above the target (5.689, lower is better), which usually means a few remaining bad training rows are dominating the linear regression fit. Keeping your exact feature set and the same `StandardScaler + LinearRegression` pipeline, I add one more standard NYC Taxi Fare “sanity” cleaning rule: drop rides where the pickup/dropoff points are an implausibly large distance apart given your existing lat/lon bounds (e.g., > ~30 miles), which removes extreme cross-boundary artifacts that linear models fit poorly. I also enforce a small, metric-consistent post-processing floor (>=2.5) since fares in this dataset essentially never go below the minimum metered fare, which tends to reduce RMSE from occasional under-predictions without changing training. Everything else (data reading, feature engineering, model, and submission schema) stays the same, and it still write a valid `submission.csv`.'
- What this solution (achieved 36.72091) has done: 'We keep your exact feature set and the same `StandardScaler + LinearRegression` pipeline, but fix one remaining high-impact correctness issue: `passenger_count` in this dataset contains many non-integer/garbled values that slip through your current filter and can badly distort OLS coefficients. We coerce `passenger_count` to numeric, drop invalid/NaN passenger counts, then re-apply the same 1–6 filter so training is cleaner while preserving the same semantics. We also ensure the same coercion is applied to test (with median imputation) so the feature distribution matches train and avoids inference-time surprises. These minimal changes are aimed at reducing RMSE from 36.7 toward your 5.689 target without changing the model/training approach.'
- What this solution (achieved 37.39318) has done: 'Your current RMSE (36.72) is still far above the 5.689 target, so we should improve it with minimal, correctness-focused changes that keep your feature set and `StandardScaler + LinearRegression` pipeline intact. The biggest remaining issue is that plain OLS is still being influenced by heavy-tailed noise/outliers even after your filters; the smallest way to reduce that sensitivity without changing the model class is to train on a *log-transformed* target and then invert predictions (this keeps the same LinearRegression training loop/semantics, but makes it much more robust). We also ensure train/test have identical numeric dtypes (especially for the one-hot weekday columns and `pickuptime`) to avoid subtle scaling/fit issues. Finally, we keep your existing clipping and submission format unchanged.'
- What this solution (achieved 9.10831) has done: 'Your RMSE is still far above target (lower is better), so we should improve it with minimal, correctness-focused changes while keeping your exact features and `StandardScaler + LinearRegression` pipeline. The biggest remaining issue is a train/test distribution mismatch: you apply distance/airport/fare-per-mile sanity filters only to train, but you never apply the same “finite + plausible range” constraints/imputations to the corresponding engineered features in test, so rare extreme engineered values can still drive huge errors. I add a small test-side cleanup that mirrors the train bounds for the engineered distance features (without dropping rows; we clip/impute using train medians) and also ensure `Weekday` mapping never creates NaNs (fill unknowns to train mode label). These changes don’t alter your model or feature set; they just make inference inputs consistent with training, which should move RMSE materially toward your target.'

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

train_data = train_data[
    ~(
        (train_data["pickup_longitude"] == 0)
        | (train_data["pickup_latitude"] == 0)
        | (train_data["dropoff_longitude"] == 0)
        | (train_data["dropoff_latitude"] == 0)
    )
]



## === cell 9
train_data["passenger_count"] = pd.to_numeric(
    train_data["passenger_count"], errors="coerce"
)
test_data["passenger_count"] = pd.to_numeric(
    test_data["passenger_count"], errors="coerce"
)

train_data = train_data.dropna(subset=["passenger_count"])



## === cell 10
train_data = train_data[
    (train_data["fare_amount"] > 0) & (train_data["fare_amount"] <= 250)
]
train_data = train_data[
    (train_data["passenger_count"] >= 1) & (train_data["passenger_count"] <= 6)
]
train_data = train_data[
    (train_data["pickup_longitude"].between(-75, -72))
    & (train_data["dropoff_longitude"].between(-75, -72))
    & (train_data["pickup_latitude"].between(40, 42))
    & (train_data["dropoff_latitude"].between(40, 42))
]



## === cell 11
plot = train_data[:2000].plot.scatter("Difference_longitude", "Difference_latitude")



## === cell 12
train_data = train_data[
    (train_data["Difference_longitude"] < 0.5)
    & (train_data["Difference_latitude"] < 0.5)
]



## === cell 13
train_dt = pd.to_datetime(
    train_data["pickup_datetime"].str.slice(0, 19), errors="coerce"
)
test_dt = pd.to_datetime(test_data["pickup_datetime"].str.slice(0, 19), errors="coerce")

train_data["pickuptime"] = (train_dt.dt.hour * 60 + train_dt.dt.minute).astype("Int64")
test_data["pickuptime"] = (test_dt.dt.hour * 60 + test_dt.dt.minute).astype("Int64")



## === cell 14
train_data["Weekday"] = train_dt.dt.weekday.astype("Int64")
test_data["Weekday"] = test_dt.dt.weekday.astype("Int64")

train_data = train_data.dropna(subset=["Weekday", "pickuptime"])



## === cell 15
train_data.head()



## === cell 16
test_data.head()



## === cell 17
train_data.drop("pickup_datetime", inplace=True, axis=1)
test_data.drop("pickup_datetime", inplace=True, axis=1)



## === cell 18
weekday_map = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}

train_pickuptime_median = int(
    train_data["pickuptime"].dropna().astype("int64").median()
)
train_weekday_mode = int(train_data["Weekday"].dropna().astype("int64").mode().iloc[0])

test_data["pickuptime"] = test_data["pickuptime"].fillna(train_pickuptime_median)
test_data["Weekday"] = test_data["Weekday"].fillna(train_weekday_mode)

train_passenger_median = float(train_data["passenger_count"].median())
test_data["passenger_count"] = test_data["passenger_count"].fillna(
    train_passenger_median
)

train_data["Weekday"] = train_data["Weekday"].map(weekday_map)

test_data["Weekday"] = test_data["Weekday"].map(weekday_map)
test_data["Weekday"] = test_data["Weekday"].fillna(weekday_map[train_weekday_mode])



## === cell 19
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
train_data["pickuptime"] = pd.to_numeric(
    train_data["pickuptime"], errors="coerce"
).astype("int64")
test_data["pickuptime"] = pd.to_numeric(
    test_data["pickuptime"], errors="coerce"
).astype("int64")

for df in (train_data, test_data):
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].astype(np.float64)

train_data.head()



## === cell 22
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



## === cell 23
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



## === cell 24
train_data = train_data[(train_data["Distance"] > 0) & (train_data["Distance"] < 100)]

fare_per_mile = train_data["fare_amount"] / (train_data["Distance"] + 1e-6)
train_data = train_data[(fare_per_mile > 0.5) & (fare_per_mile < 50)]

train_data = train_data[
    (train_data["Pickup_Distance_airport"].between(0, 100))
    & (train_data["Dropoff_Distance_airport"].between(0, 100))
]

train_data = train_data[train_data["Distance"] <= 30].copy()



## === cell 25
train_engineered_cols = [
    "Distance",
    "Pickup_Distance_airport",
    "Dropoff_Distance_airport",
]
train_engineered_medians = train_data[train_engineered_cols].median(numeric_only=True)

for col in train_engineered_cols:
    test_data[col] = pd.to_numeric(test_data[col], errors="coerce")
    test_data[col] = test_data[col].replace([np.inf, -np.inf], np.nan)

test_data["Distance"] = test_data["Distance"].clip(lower=0.0, upper=30.0)
test_data["Pickup_Distance_airport"] = test_data["Pickup_Distance_airport"].clip(
    lower=0.0, upper=100.0
)
test_data["Dropoff_Distance_airport"] = test_data["Dropoff_Distance_airport"].clip(
    lower=0.0, upper=100.0
)

for col in train_engineered_cols:
    test_data[col] = test_data[col].fillna(float(train_engineered_medians[col]))



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
numeric_cols_train = (
    train_data.drop(columns=["key"], errors="ignore")
    .select_dtypes(include=[np.number])
    .columns
)
mask_finite_all = np.isfinite(train_data[numeric_cols_train]).all(axis=1)
train_data = train_data.loc[mask_finite_all].copy()

numeric_cols_test = (
    test_data.drop(columns=["key"], errors="ignore")
    .select_dtypes(include=[np.number])
    .columns
)
train_medians = train_data[numeric_cols_train].median(numeric_only=True)

test_data[numeric_cols_test] = test_data[numeric_cols_test].replace(
    [np.inf, -np.inf], np.nan
)
test_data[numeric_cols_test] = test_data[numeric_cols_test].fillna(
    train_medians.reindex(numeric_cols_test)
)



## === cell 28
train_data.shape



## === cell 29
test_data.shape



## === cell 30
from sklearn.model_selection import train_test_split

X = train_data.drop(["key", "fare_amount"], axis=1)
y = train_data["fare_amount"]

X.columns = X.columns.astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.01, random_state=80
)



## === cell 31
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

lr = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", LinearRegression()),
    ]
)

y_train_log = np.log1p(y_train.values)
y_test_log = np.log1p(y_test.values)

lr.fit(X_train, y_train_log)
print("R^2 on log1p target:", lr.score(X_test, y_test_log))



## === cell 32
X_submit = test_data.drop("key", axis=1)
X_submit.columns = X_submit.columns.astype(str)
X_submit = X_submit.reindex(columns=X.columns, fill_value=0)

pred_log = lr.predict(X_submit)
pred = np.expm1(pred_log)

pred = np.clip(pred, 2.5, 250.0)
pred = np.round(pred, 2)



## === cell 33
pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
).head()



## === cell 34
Submission = pd.DataFrame({"key": test_data["key"].values, "fare_amount": pred})
Submission = Submission[["key", "fare_amount"]]
Submission.head()



## === cell 35
Submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", Submission.shape)
print(Submission.head())
