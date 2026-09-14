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

3.10

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

5.6891

# 6. Current score

29.8543

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 936.92806) has done: 'The crash happens because scikit-learn 1.2 removed the `normalize` argument from `LinearRegression`, so passing `normalize=True` raises a `TypeError`. The minimal fix is to instantiate `LinearRegression()` without `normalize`, preserving the same training/prediction flow and keeping `lr` defined for the next cell. No other logic, data, or evaluation semantics are changed; only the incompatible parameter is removed. This ensures cell 43 can still call `lr.predict(...)` as before.'
- What this solution (achieved 752.76803) has done: 'Your RMSE is extremely high because the model is producing invalid/unstable predictions (often from misaligned one-hot columns and inconsistent scaling between train and test), which makes the submission far from realistic fares. I keep your exact feature set and LinearRegression training, but make train/test preprocessing consistent by (1) forcing dummy columns to align exactly and (2) applying the same mean/variance normalization parameters learned from train to the test set. I also clip negative fare predictions to 0.0 (a minimal, metric-aligned post-process) to avoid catastrophic errors. These are small changes that typically move RMSE dramatically toward a reasonable band without changing the core approach.'
- What this solution (achieved 1275.34428) has done: 'Your RMSE is still catastrophically high mainly because the model is being trained on many invalid/outlier rows (negative/huge fares and impossible coordinates) which dominate a linear regression and explode predictions on the test set. I keep your exact feature engineering and LinearRegression approach, but add minimal, standard NYC Taxi Fare data cleaning filters (fare bounds, passenger_count bounds, and latitude/longitude bounding box + non-zero trip distance) to stabilize coefficients. I also apply the same mean/variance normalization to the distance-to-airport features (currently unscaled), which is consistent with what you already do for abs_diff_* and improves numerical conditioning without changing the model. Finally, I keep the submission schema identical and ensure the CSV is written.'
- What this solution (achieved 946.96906) has done: 'Your RMSE is still far above the target, so we need a small but high-impact fix that doesn’t change the model: the current scaling uses variance as a divisor (instead of standard deviation), which massively distorts feature magnitudes and can destabilize LinearRegression coefficients/predictions. I keep your exact feature set and LinearRegression training flow, but replace the variance-based normalization with standard-deviation scaling (using train-derived parameters and applying identically to test), which should dramatically reduce prediction blow-ups and move RMSE much closer to the target. I also add a minimal “remove extreme distances” filter (same spirit as your existing filters) to reduce leverage points that a linear model handles poorly, and keep your existing non-negative clipping and submission schema unchanged. The script still run end-to-end and write `Submission.csv` with `key,fare_amount`.'
- What this solution (achieved 946.96906) has done: 'Your RMSE is still catastrophically high, so we need a minimal “stabilize LinearRegression coefficients” change without changing the model or features: the remaining biggest issue is that `pickuptime` is an unscaled ~0–2359 feature while everything else is z-scored, which can dominate the fit and explode predictions. I add the same train-derived standard-deviation scaling to `pickuptime` (and apply identically to test) to keep feature magnitudes comparable, preserving your exact feature set and training flow. I also add a tiny numeric safety step (`nan_to_num`) right before prediction to avoid any accidental NaN/inf propagating into `lr.predict`, which can otherwise create extreme outputs and huge RMSE. The submission schema and filename remain unchanged (`Submission.csv`, columns `key,fare_amount`).'
- What this solution (achieved 946.96906) has done: 'Your RMSE is still catastrophically worse than the target, so we need a minimal fix that stabilizes predictions without changing your model or feature set. The biggest remaining issue is that `train_test_split` shuffles rows, but your `lr` is being fit on a DataFrame that can carry non-contiguous indices after many filters; this can silently produce misalignment in some pipelines and increases the risk of NaN/inf propagation and extreme predictions. I make the training data index contiguous right after filtering (no data change, just alignment), and I apply the same numeric safety (`replace inf -> nan`, `fillna(0)`) to `X_train/X_test` as you already do for `X_sub` to prevent unstable coefficients. These are minimal, semantics-preserving hygiene steps that typically reduce “blow-up” predictions and move RMSE sharply toward a reasonable band.'
- What this solution (achieved 12.59768) has done: 'Your RMSE is still far above target, which strongly suggests the model is learning from (or predicting on) mis-parsed time features rather than true numeric signal. I keep your exact feature set and LinearRegression training flow, but fix the pickup time and weekday extraction to use vectorized `pd.to_datetime` parsing (same information, just correctly/consistently derived) and ensure `pickuptime` is numeric from the start. This is a minimal semantics-preserving change that typically collapses “blow-up” predictions caused by string-slicing quirks and mixed types. I also add a tiny final safety clamp for absurdly large predictions (non-negative clip is already there) to prevent a small number of extreme values from dominating RMSE, without changing the model or features.'
- What this solution (achieved 12.61449) has done: 'Your current RMSE (12.59768) is still far above the target (5.6891), so we should make the smallest changes that improve generalization without changing your model or feature set. The biggest remaining issue is that LinearRegression is very sensitive to a small number of remaining high-leverage/outlier rows; adding one more standard NYC filter (remove trips with implausible speed given your computed distance and pickup time) typically reduces coefficient blow-ups while preserving your exact core approach. I also ensure any rows with unparseable datetimes are removed consistently (they currently survive until later as NaNs in time features), which otherwise injects noise. These are minimal data-cleaning steps that keep the same features, scaling, and LinearRegression training/prediction flow, but usually move RMSE materially toward the target band.'
- What this solution (achieved 13.33843) has done: 'We keep your exact LinearRegression model and the same engineered feature set, but add one minimal, high-impact cleanup that specifically helps RMSE: remove a small number of remaining “high-leverage” rows by filtering on a realistic fare-per-mile range using your already-computed `Distance`. This does not change the model, loss, or feature extraction; it only prevents extreme coefficient distortion caused by mislabeled/outlier rides that a linear model can’t handle well. We also apply the same `replace inf -> nan -> fillna(0)` safety to `X_sub` (already done) and keep submission formatting unchanged. These small data-quality constraints typically move RMSE materially downward toward your target without altering the core approach.'
- What this solution (achieved 15.10797) has done: 'Your RMSE is still far above the target, so we should make a minimal, metric-aligned improvement without changing your model or feature set: add one more standard NYC Taxi cleanup that removes extreme outliers the linear model can’t handle—filter training rows by bounding the straight-line trip distance **in degrees** (`sqrt(abs_diff_longitude^2+abs_diff_latitude^2)`) to exclude cross-country/bad GPS points that can still slip through your current filters. This keeps the exact same engineered columns and LinearRegression training/prediction flow, but typically reduces coefficient distortion and improves RMSE materially. I also keep train/test preprocessing consistent and add the same numeric safety (`replace inf -> nan -> fillna(0)`) earlier for train after feature creation (no semantic change for valid rows, just prevents rare NaN propagation). The script still runs end-to-end and writes `Submission.csv` with `key,fare_amount`.'
- What this solution (achieved 15.02696) has done: 'Your current RMSE (15.10797) is still far above the target (5.6891), so we should make a minimal, high-impact fix without changing the model or feature set: ensure the one-hot weekday columns are strictly identical and in the same order for train/test after all filtering (your current concat-based one-hot can still drift if the train filters remove some weekdays). I also add the same train-derived scaling safety to `passenger_count` (it’s currently unscaled while most other numeric features are z-scored), which helps LinearRegression conditioning without changing the approach. Finally, I apply the same `replace inf -> nan -> fillna(0)` cleanup consistently to `test_df` right after feature/scaling creation to prevent rare NaNs from causing unstable predictions. These are small, evaluation-aligned stability fixes that typically reduce RMSE materially toward your target.'
- What this solution (achieved 15.02696) has done: 'Your RMSE is still far above target, so we need a small, high-impact correction that keeps your LinearRegression + current features intact: the current pipeline **drops `key` from `test_df` by resetting its index after fillna**, which silently misaligns predictions to the wrong keys and typically destroys leaderboard RMSE. I preserve the exact feature engineering and model, but keep `test_df`’s original ordering/length and only clean numeric feature columns (not `key`) so every prediction maps to the correct row. I also ensure one-hot weekday columns are present and ordered identically (already mostly done) and add a final safety check that submission row count equals `sample_submission` to avoid hidden misalignment. These changes are minimal, semantics-preserving, and directly targeted to move RMSE down toward your target.'
- What this solution (achieved 29.8543) has done: 'Your RMSE is still far above the target, so we should make the smallest change that usually yields a big RMSE drop for this specific competition without changing your core LinearRegression + current features: **use a log-transform on the target during training and invert it at prediction time**. This keeps the same model, features, and training flow (still `LinearRegression().fit(...)`), but reduces the impact of high-fare outliers and typically improves RMSE substantially. I also ensure the train/test feature columns are *exactly* identical (same order) right before fitting/predicting to avoid any silent column drift. Finally, I keep your submission formatting and key alignment unchanged and still write `Submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import time
import os  # reading the input files we have access to

print(os.listdir("../input"))



## === cell 1
train_df = pd.read_csv("../input/train.csv", nrows=10_000_000)
train_df.dtypes



## === cell 2
train_df.head()



## === cell 3
train_df.shape



## === cell 4
train_df.info()



## === cell 5
test_df = pd.read_csv("../input/test.csv")
test_df.dtypes



## === cell 6
test_df.head()



## === cell 7
test_df.info()



## === cell 8
test_df.shape



## === cell 9
train_df.isna().sum()




## === cell 10
def add_travel_vector_features(df):
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs()


add_travel_vector_features(train_df)
add_travel_vector_features(test_df)



## === cell 11
print(f"Before Dropping null values: {len(train_df)}")
train_df.dropna(inplace=True)
print(f"After Dropping null values: {len(train_df)}")



## === cell 12
print(train_df.isnull().sum())



## === cell 13
print("Old size: %d" % len(train_df))
train_df = train_df.dropna(how="any", axis="rows")
print("New size: %d" % len(train_df))



## === cell 14
plot = train_df.iloc[:2000].plot.scatter("abs_diff_longitude", "abs_diff_latitude")



## === cell 15
print("Old size: %d" % len(train_df))
train_df = train_df[
    (train_df.abs_diff_longitude < 5.0) & (train_df.abs_diff_latitude < 5.0)
]
print("New size: %d" % len(train_df))



## === cell 16
print("Old size before fare/geo/passenger filters: %d" % len(train_df))

train_df = train_df[
    (train_df["fare_amount"] >= 0.0) & (train_df["fare_amount"] <= 200.0)
]

train_df = train_df[
    (train_df["passenger_count"] >= 1) & (train_df["passenger_count"] <= 6)
]

coord_mask = (
    (train_df["pickup_longitude"].between(-75.0, -72.0))
    & (train_df["dropoff_longitude"].between(-75.0, -72.0))
    & (train_df["pickup_latitude"].between(40.0, 42.0))
    & (train_df["dropoff_latitude"].between(40.0, 42.0))
)
train_df = train_df[coord_mask]

train_df = train_df[
    (train_df["abs_diff_longitude"] > 0) | (train_df["abs_diff_latitude"] > 0)
]

print("New size after fare/geo/passenger filters: %d" % len(train_df))




## === cell 17
def add_time_features(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=False)
    df["Weekday"] = dt.dt.weekday
    df["pickuptime"] = (
        dt.dt.hour.astype("Int64") * 100 + dt.dt.minute.astype("Int64")
    ).astype("float64")


add_time_features(train_df)
add_time_features(test_df)

print("Old size before dropping NaT datetimes: %d" % len(train_df))
train_df = train_df[train_df["Weekday"].notna() & train_df["pickuptime"].notna()]
print("New size after dropping NaT datetimes: %d" % len(train_df))



## === cell 18
train_df.head()



## === cell 19
test_df.head()




## === cell 20
def creating_weekdays(df):
    return


creating_weekdays(train_df)
creating_weekdays(test_df)



## === cell 21
train_df.head()



## === cell 22
test_df.head()



## === cell 23
train_df.drop("pickup_datetime", inplace=True, axis=1)
test_df.drop("pickup_datetime", inplace=True, axis=1)




## === cell 24
def replace_weekday(df):
    df["Weekday"].replace(
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


replace_weekday(train_df)
replace_weekday(test_df)



## === cell 25
train_df.head()



## === cell 26
test_df.head()



## === cell 27
weekday_categories = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

train_weekday = pd.Categorical(train_df["Weekday"], categories=weekday_categories)
test_weekday = pd.Categorical(test_df["Weekday"], categories=weekday_categories)

train_one_hot = pd.get_dummies(train_weekday)
test_one_hot = pd.get_dummies(test_weekday)

train_one_hot = train_one_hot.reindex(columns=weekday_categories, fill_value=0)
test_one_hot = test_one_hot.reindex(columns=weekday_categories, fill_value=0)

train_df = pd.concat(
    [train_df.reset_index(drop=True), train_one_hot.reset_index(drop=True)], axis=1
)
test_df = pd.concat(
    [test_df.reset_index(drop=True), test_one_hot.reset_index(drop=True)], axis=1
)



## === cell 28
train_df.head()



## === cell 29
test_df.head()



## === cell 30
train_df.drop("Weekday", axis=1, inplace=True)
test_df.drop("Weekday", axis=1, inplace=True)




## === cell 31
def creating_pickupdate(df):
    return


creating_pickupdate(train_df)
creating_pickupdate(test_df)



## === cell 32
train_df.head()



## === cell 33
test_df.head()




## === cell 34
def finding_distance(df):
    R = 6373.0
    lat1 = np.asarray(np.radians(df["pickup_latitude"]))
    lon1 = np.asarray(np.radians(df["pickup_longitude"]))
    lat2 = np.asarray(np.radians(df["dropoff_latitude"]))
    lon2 = np.asarray(np.radians(df["dropoff_longitude"]))

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c

    df["Distance"] = np.asarray(distance) * 0.621


finding_distance(train_df)
finding_distance(test_df)




## === cell 35
def creating_pickup_dropoff_distance(df):
    R = 6373.0
    lat1 = np.asarray(np.radians(df["pickup_latitude"]))
    lon1 = np.asarray(np.radians(df["pickup_longitude"]))
    lat2 = np.asarray(np.radians(df["dropoff_latitude"]))
    lon2 = np.asarray(np.radians(df["dropoff_longitude"]))

    lat3 = np.zeros(len(df)) + np.radians(40.6413111)
    lon3 = np.zeros(len(df)) + np.radians(-73.7781391)
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
    df["Pickup_Distance_airport"] = np.asarray(distance1) * 0.621

    a2 = (
        np.sin(d_lat_dropoff / 2) ** 2
        + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
    )
    c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
    distance2 = R * c2

    df["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621


creating_pickup_dropoff_distance(train_df)
creating_pickup_dropoff_distance(test_df)



## === cell 36
train_df["Distance"] = np.round(train_df["Distance"], 2)
train_df["Pickup_Distance_airport"] = np.round(train_df["Pickup_Distance_airport"], 2)
train_df["Dropoff_Distance_airport"] = np.round(train_df["Dropoff_Distance_airport"], 2)
test_df["Distance"] = np.round(test_df["Distance"], 2)
test_df["Pickup_Distance_airport"] = np.round(test_df["Pickup_Distance_airport"], 2)
test_df["Dropoff_Distance_airport"] = np.round(test_df["Dropoff_Distance_airport"], 2)



## === cell 37
print("Old size before distance filter: %d" % len(train_df))
train_df = train_df[(train_df["Distance"] > 0.0) & (train_df["Distance"] <= 100.0)]
print("New size after distance filter: %d" % len(train_df))



## === cell 38
hh = np.floor(train_df["pickuptime"] / 100.0)
mm = train_df["pickuptime"] - hh * 100.0
duration_hours = (
    hh + mm / 60.0
)  # time-of-day in hours, not duration, so we must derive duration differently

print("Old size before stricter distance cap: %d" % len(train_df))
train_df = train_df[train_df["Distance"] <= 60.0]
print("New size after stricter distance cap: %d" % len(train_df))



## === cell 39
print("Old size before fare-per-mile filter: %d" % len(train_df))
fare_per_mile = train_df["fare_amount"] / np.maximum(train_df["Distance"], 0.1)
train_df = train_df[(fare_per_mile >= 0.5) & (fare_per_mile <= 50.0)]
print("New size after fare-per-mile filter: %d" % len(train_df))



## === cell 40
print("Old size before coordinate-distance cleanup: %d" % len(train_df))
coord_dist = np.sqrt(
    train_df["abs_diff_longitude"] ** 2 + train_df["abs_diff_latitude"] ** 2
)
train_df = train_df[
    coord_dist <= 1.0
]  # ~<= 70 miles in NYC latitudes; removes extreme GPS outliers
print("New size after coordinate-distance cleanup: %d" % len(train_df))



## === cell 41
train_df.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)
test_df.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)



## === cell 42
abs_lon_mean = np.mean(train_df["abs_diff_longitude"])
abs_lon_std = np.std(train_df["abs_diff_longitude"])
if abs_lon_std == 0:
    abs_lon_std = 1.0

train_df["abs_diff_longitude"] = (
    train_df["abs_diff_longitude"] - abs_lon_mean
) / abs_lon_std



## === cell 43
abs_lat_mean = np.mean(train_df["abs_diff_latitude"])
abs_lat_std = np.std(train_df["abs_diff_latitude"])
if abs_lat_std == 0:
    abs_lat_std = 1.0

train_df["abs_diff_latitude"] = (
    train_df["abs_diff_latitude"] - abs_lat_mean
) / abs_lat_std



## === cell 44
test_df["abs_diff_longitude"] = (
    test_df["abs_diff_longitude"] - abs_lon_mean
) / abs_lon_std
test_df["abs_diff_latitude"] = (
    test_df["abs_diff_latitude"] - abs_lat_mean
) / abs_lat_std



## === cell 45
dist_mean = np.mean(train_df["Distance"])
dist_std = np.std(train_df["Distance"])
if dist_std == 0:
    dist_std = 1.0
train_df["Distance"] = (train_df["Distance"] - dist_mean) / dist_std
test_df["Distance"] = (test_df["Distance"] - dist_mean) / dist_std

pda_mean = np.mean(train_df["Pickup_Distance_airport"])
pda_std = np.std(train_df["Pickup_Distance_airport"])
if pda_std == 0:
    pda_std = 1.0
train_df["Pickup_Distance_airport"] = (
    train_df["Pickup_Distance_airport"] - pda_mean
) / pda_std
test_df["Pickup_Distance_airport"] = (
    test_df["Pickup_Distance_airport"] - pda_mean
) / pda_std

dda_mean = np.mean(train_df["Dropoff_Distance_airport"])
dda_std = np.std(train_df["Dropoff_Distance_airport"])
if dda_std == 0:
    dda_std = 1.0
train_df["Dropoff_Distance_airport"] = (
    train_df["Dropoff_Distance_airport"] - dda_mean
) / dda_std
test_df["Dropoff_Distance_airport"] = (
    test_df["Dropoff_Distance_airport"] - dda_mean
) / dda_std



## === cell 46
pt_mean = np.mean(train_df["pickuptime"])
pt_std = np.std(train_df["pickuptime"])
if pt_std == 0:
    pt_std = 1.0
train_df["pickuptime"] = (train_df["pickuptime"] - pt_mean) / pt_std
test_df["pickuptime"] = (test_df["pickuptime"] - pt_mean) / pt_std



## === cell 47
pc_mean = np.mean(train_df["passenger_count"])
pc_std = np.std(train_df["passenger_count"])
if pc_std == 0:
    pc_std = 1.0
train_df["passenger_count"] = (train_df["passenger_count"] - pc_mean) / pc_std
test_df["passenger_count"] = (test_df["passenger_count"] - pc_mean) / pc_std



## === cell 48
train_df = train_df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

test_keys = test_df["key"].copy()
test_df = test_df.replace([np.inf, -np.inf], np.nan)
feature_cols = [c for c in test_df.columns if c != "key"]
test_df[feature_cols] = test_df[feature_cols].fillna(0.0)
test_df["key"] = test_keys  # ensure key is preserved exactly

print(train_df.shape)
print(test_df.shape)



## === cell 49
from sklearn.model_selection import train_test_split

X = train_df.drop(["key", "fare_amount"], axis=1)
y = train_df["fare_amount"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.01, random_state=80
)



## === cell 50
from sklearn.linear_model import LinearRegression

y_train_log = np.log1p(y_train)

X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0.0)
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0.0)

common_cols = X_train.columns.tolist()
X_test = X_test.reindex(columns=common_cols, fill_value=0.0)

lr = LinearRegression()
lr.fit(X_train, y_train_log)

pred_test = np.expm1(lr.predict(X_test))
pred_test = np.clip(pred_test, 0.0, 200.0)
rmse = float(np.sqrt(np.mean((pred_test - y_test.values) ** 2)))
print("Holdout RMSE (approx):", rmse)



## === cell 51
X_sub = test_df.drop("key", axis=1)
X_sub = X_sub.replace([np.inf, -np.inf], np.nan).fillna(0.0)
X_sub = X_sub.reindex(columns=common_cols, fill_value=0.0)

pred = np.expm1(lr.predict(X_sub))
pred = np.round(pred, 2)
pred = np.clip(pred, 0.0, 200.0)

print(pred)



## === cell 52
Submission = pd.DataFrame(data=pred, columns=["fare_amount"])
Submission["key"] = test_df["key"]
Submission = Submission[["key", "fare_amount"]]



## === cell 53
sample_sub = pd.read_csv("../input/sample_submission.csv")
assert len(Submission) == len(
    sample_sub
), f"Row count mismatch: Submission={len(Submission)} vs sample={len(sample_sub)}"

Submission.set_index("key", inplace=True)



## === cell 54
Submission.head()



## === cell 55
Submission.to_csv("Submission.csv")
