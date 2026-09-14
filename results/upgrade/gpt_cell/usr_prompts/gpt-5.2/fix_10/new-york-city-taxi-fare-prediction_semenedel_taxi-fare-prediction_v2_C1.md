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

5.69073

# 6. Current score

37.42517

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 937.00007) has done: 'The crash happens because scikit-learn 1.2 removed the `normalize` parameter from `LinearRegression`, so passing `normalize=True` raises a `TypeError`. The minimal fix is to construct `LinearRegression()` without that argument while keeping the same training and scoring flow. This change is localized to cell 23 and preserves the `lr` variable used in cell 24. No other logic, features, or evaluation semantics are changed.'
- What this solution (achieved 751.68646) has done: 'Your current RMSE (937) indicates the model is producing invalidly large predictions for many rows; the biggest minimal fix is to make train and test feature columns identical after one-hot encoding weekdays (right now they can differ), and to apply the exact same scaling statistics from train to test (you currently normalize test using test mean/var and even delay one division to the next cell). These are small, semantics-preserving corrections that keep your exact model (LinearRegression) and feature set, but remove train/test skew that explodes errors. I also add a deterministic split (`random_state`) to stabilize the learned coefficients and avoid score swings. The script still write `Submission.csv` in the required `key,fare_amount` format.'
- What this solution (achieved 1058.18974) has done: 'Your RMSE is still extremely high because the model is being trained on rows containing many invalid/outlier coordinates and fares (common in this dataset), and linear regression gets distorted by those extremes; the smallest legitimate fix is to add the standard NYC Taxi competition cleaning filters (bounding-box for NYC coordinates, passenger_count sanity, and fare_amount range) while keeping your same features and same LinearRegression training loop. I’m also correcting the scaling bug in cell 20: using variance (and abs) is not proper normalization and can magnify values; switching to standard deviation with the same train-derived statistics reduces feature scale issues without changing the model family or training approach. Finally, I keep the existing train/test column alignment and ensure the submission rows remain correctly keyed.'
- What this solution (achieved 15.27007) has done: 'Your very high RMSE suggests predictions are still occasionally exploding and dominating the metric. To move the score toward your target with minimal logic changes, I (1) keep your exact features and LinearRegression training, but add a small post-processing clamp on predictions to a realistic fare range and ensure all predictions are finite, and (2) fix a subtle submission-format risk by writing the CSV with an explicit `key,fare_amount` header (no index). These changes don’t alter the model/feature engineering, but they prevent invalid/extreme outputs from ruining RMSE and guarantee the file matches Kaggle’s required format.'
- What this solution (achieved 15.27374) has done: 'Your current RMSE is still far from the target (15.27 vs 5.69, lower is better), so we need a small, legitimate improvement that doesn’t change your model family or training loop. The biggest remaining issue is that the one-hot weekday columns can still be misaligned between train and test because you use `align(join="left")`, which can drop test-only columns and skew the regression at inference; switching to `join="outer"` makes the feature spaces identical. Also, your `lr.score()` check is R² (not RMSE), so I add an RMSE sanity check (doesn’t affect submission) to catch problems early. Finally, I add a minimal additional train-cleaning rule to remove rows with zero distance but non-trivial fare (common GPS/recording errors) to reduce distortion in linear regression without changing features or the model.'
- What this solution (achieved 16.50371) has done: 'Your RMSE (15.27) is still far from the target (5.69, lower is better), so we need a small but high-impact fix without changing the model family or training loop. The biggest remaining issue is that you’re training a plain LinearRegression on a distribution with heavy outliers; a minimal, competition-standard improvement is to train the same LinearRegression on a log1p-transformed target and invert with expm1 at prediction time, which stabilizes outlier influence while preserving the same approach. I also add a tiny consistency clamp on validation predictions (same as submission clamp) so your sanity-check RMSE reflects what you actually submit. Everything else (features, cleaning, one-hot alignment, scaling, model) stays the same and it still writes a valid `Submission.csv`.'
- What this solution (achieved 16.51293) has done: 'Your current RMSE (16.50 vs target 5.69, lower is better) is still dominated by a few systematic issues that are cheap to fix without changing your model family or training flow. I (1) fix a bug where you build `train_X/train_y` before feature engineering and then later ignore it (wasted work) by removing that unused matrix creation, (2) add two minimal, competition-standard cleaning rules that strongly help linear regression/log-target stability (remove extreme `abs_diff_*` outliers more tightly and remove very small-distance rides with high fare), and (3) ensure the train/test one-hot alignment and scaling remain identical while keeping your LinearRegression-on-log1p target logic unchanged. These adjustments should improve generalization and reduce occasional large errors, moving RMSE downward toward the target band while preserving your core approach and producing the same submission format.'
- What this solution (achieved 16.51293) has done: 'Your RMSE is still far above the target, so we should make a small, legitimate improvement that reduces large residuals without changing your model family or training loop. The highest-impact minimal change is to add one more standard NYC Taxi cleaning rule: remove records where pickup and dropoff coordinates are identical (or nearly identical) but the fare is non-trivial; those rows heavily distort linear regression even with a log target. I implement this using your already-created `abs_diff_longitude/abs_diff_latitude` features and keep all existing feature engineering, scaling, log1p training, and submission formatting the same. This should move RMSE downward toward the target band while remaining stable and fast.'
- What this solution (achieved 37.42517) has done: 'Your current RMSE (16.51 vs 5.69, lower is better) suggests the main remaining issue is still data quality and a few extreme-but-valid patterns that distort a plain linear model even with a log1p target. I keep your exact model (LinearRegression), features, log1p training/inversion, scaling, and submission format, but tighten training-set cleaning with a couple of competition-standard rules: remove unrealistic long trips, enforce a sane passenger_count upper bound (0–6), and drop extreme per-mile fares and near-zero fares. These are minimal, localized filters that reduce outlier leverage (the typical cause of large RMSE) without changing the learning approach. The pipeline still run end-to-end and write `Submission.csv` with `key,fare_amount`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
train_df = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", nrows=10_000_000
)
train_df.dtypes



## === cell 2
test_df = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")
test_df.dtypes



## === cell 3
print("Train DF shape ", train_df.shape)
print("Test DF Shape: ", test_df.shape)




## === cell 4
def add_travel_vector_features(df):
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs()


add_travel_vector_features(train_df)
add_travel_vector_features(test_df)



## === cell 5
test_df.head()



## === cell 6
print(train_df.isnull().sum())



## === cell 7
print("Old size: %d" % len(train_df))
train_df = train_df.dropna(how="any", axis="rows")
print("New size: %d" % len(train_df))



## === cell 8
print(test_df.isnull().sum())



## === cell 9
print("Old size: %d" % len(train_df))
train_df = train_df[(train_df["fare_amount"] > 0) & (train_df["fare_amount"] <= 250)]
train_df = train_df[
    (train_df["passenger_count"] >= 1) & (train_df["passenger_count"] <= 6)
]
train_df = train_df[
    (train_df["pickup_longitude"].between(-74.3, -72.9))
    & (train_df["dropoff_longitude"].between(-74.3, -72.9))
    & (train_df["pickup_latitude"].between(40.5, 41.8))
    & (train_df["dropoff_latitude"].between(40.5, 41.8))
]
train_df = train_df[
    (train_df.abs_diff_longitude < 1.0) & (train_df.abs_diff_latitude < 1.0)
]
print("New size: %d" % len(train_df))



## === cell 10
ls1 = list(train_df["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][11:-7:]
train_df["pickuptime"] = ls1

ls1 = list(test_df["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][11:-7:]
test_df["pickuptime"] = ls1



## === cell 11
ls1 = list(train_df["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][:-4:]
    ls1[i] = pd.Timestamp(ls1[i])
    ls1[i] = ls1[i].weekday()
train_df["Weekday"] = ls1

ls1 = list(test_df["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][:-4:]
    ls1[i] = pd.Timestamp(ls1[i])
    ls1[i] = ls1[i].weekday()
test_df["Weekday"] = ls1



## === cell 12
train_df.drop("pickup_datetime", inplace=True, axis=1)
test_df.drop("pickup_datetime", inplace=True, axis=1)



## === cell 13
train_df["Weekday"].replace(
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
test_df["Weekday"].replace(
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

train_one_hot = pd.get_dummies(train_df["Weekday"])
test_one_hot = pd.get_dummies(test_df["Weekday"])

train_df = pd.concat([train_df, train_one_hot], axis=1)
test_df = pd.concat([test_df, test_one_hot], axis=1)

train_df.drop("Weekday", axis=1, inplace=True)
test_df.drop("Weekday", axis=1, inplace=True)

ls1 = list(train_df["pickuptime"])
for i in range(len(ls1)):
    z = ls1[i].split(":")
    ls1[i] = int(z[0]) * 100 + int(z[1])
train_df["pickuptime"] = ls1

ls1 = list(test_df["pickuptime"])
for i in range(len(ls1)):
    z = ls1[i].split(":")
    ls1[i] = int(z[0]) * 100 + int(z[1])
test_df["pickuptime"] = ls1



## === cell 14
train_df.head()



## === cell 15
R = 6373.0
lat1 = np.asarray(np.radians(train_df["pickup_latitude"]))
lon1 = np.asarray(np.radians(train_df["pickup_longitude"]))
lat2 = np.asarray(np.radians(train_df["dropoff_latitude"]))
lon2 = np.asarray(np.radians(train_df["dropoff_longitude"]))

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
train_df["Distance"] = np.asarray(distance) * 0.621

lat1 = np.asarray(np.radians(test_df["pickup_latitude"]))
lon1 = np.asarray(np.radians(test_df["pickup_longitude"]))
lat2 = np.asarray(np.radians(test_df["dropoff_latitude"]))
lon2 = np.asarray(np.radians(test_df["dropoff_longitude"]))

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
test_df["Distance"] = np.asarray(distance) * 0.621



## === cell 16
R = 6373.0
lat1 = np.asarray(np.radians(train_df["pickup_latitude"]))
lon1 = np.asarray(np.radians(train_df["pickup_longitude"]))
lat2 = np.asarray(np.radians(train_df["dropoff_latitude"]))
lon2 = np.asarray(np.radians(train_df["dropoff_longitude"]))

lat3 = np.zeros(len(train_df)) + np.radians(40.6413111)
lon3 = np.zeros(len(train_df)) + np.radians(-73.7781391)

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
train_df["Pickup_Distance_airport"] = np.asarray(distance1) * 0.621

a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
train_df["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621

lat1 = np.asarray(np.radians(test_df["pickup_latitude"]))
lon1 = np.asarray(np.radians(test_df["pickup_longitude"]))
lat2 = np.asarray(np.radians(test_df["dropoff_latitude"]))
lon2 = np.asarray(np.radians(test_df["dropoff_longitude"]))

lat3 = np.zeros(len(test_df)) + np.radians(40.6413111)
lon3 = np.zeros(len(test_df)) + np.radians(-73.7781391)

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
test_df["Pickup_Distance_airport"] = np.asarray(distance1) * 0.621

a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
test_df["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621



## === cell 17
train_df["Distance"] = np.round(train_df["Distance"], 2)
train_df["Pickup_Distance_airport"] = np.round(train_df["Pickup_Distance_airport"], 2)
train_df["Dropoff_Distance_airport"] = np.round(train_df["Dropoff_Distance_airport"], 2)
test_df["Distance"] = np.round(test_df["Distance"], 2)
test_df["Pickup_Distance_airport"] = np.round(test_df["Pickup_Distance_airport"], 2)
test_df["Dropoff_Distance_airport"] = np.round(test_df["Dropoff_Distance_airport"], 2)



## === cell 18
print("Before additional cleaning size:", len(train_df))

train_df = train_df[(train_df["Distance"] > 0) & (train_df["Distance"] <= 50)].copy()

train_df = train_df[
    (train_df["passenger_count"] >= 1) & (train_df["passenger_count"] <= 6)
].copy()

train_df = train_df[train_df["fare_amount"] >= 2.5].copy()

ppm = train_df["fare_amount"] / train_df["Distance"].clip(lower=0.1)
train_df = train_df[(ppm >= 1.0) & (ppm <= 50.0)].copy()

print("After additional cleaning size:", len(train_df))



## === cell 19
train_df = train_df[
    ~((train_df["Distance"] <= 0.01) & (train_df["fare_amount"] > 5.0))
].copy()
train_df = train_df[
    ~((train_df["Distance"] <= 0.10) & (train_df["fare_amount"] > 30.0))
].copy()

train_df = train_df[
    ~(
        (train_df["abs_diff_longitude"] < 1e-4)
        & (train_df["abs_diff_latitude"] < 1e-4)
        & (train_df["fare_amount"] > 7.0)
    )
].copy()

print("After distance/fare + same-point sanity filters size:", len(train_df))



## === cell 20
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



## === cell 21
drop_cols = ["fare_amount"]
train_features_tmp = train_df.drop(columns=drop_cols, errors="ignore")
train_features_tmp, test_features_tmp = train_features_tmp.align(
    test_df, join="outer", axis=1, fill_value=0
)
train_df = pd.concat([train_df[["fare_amount"]], train_features_tmp], axis=1)
test_df = test_features_tmp

for col in [
    "abs_diff_longitude",
    "abs_diff_latitude",
    "Distance",
    "Pickup_Distance_airport",
    "Dropoff_Distance_airport",
]:
    if col in train_df.columns and col in test_df.columns:
        mu = train_df[col].mean()
        std = train_df[col].std()
        if std == 0 or np.isnan(std):
            std = 1.0
        train_df[col] = (train_df[col] - mu) / std
        test_df[col] = (test_df[col] - mu) / std



## === cell 22
from sklearn.model_selection import train_test_split

X = train_df.drop(["key", "fare_amount"], axis=1)
y = train_df["fare_amount"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.01, random_state=42
)



## === cell 23
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lr = LinearRegression()
lr.fit(X_train, np.log1p(y_train))

val_pred_log = lr.predict(X_test)
val_pred = np.expm1(val_pred_log)
val_pred = np.nan_to_num(val_pred, nan=11.35, posinf=250.0, neginf=0.0)
val_pred = np.clip(val_pred, 0.0, 250.0)

rmse = mean_squared_error(y_test, val_pred, squared=False)
print("Validation RMSE (sanity check, clipped):", rmse)
print("Validation R^2 (log-space):", lr.score(X_test, np.log1p(y_test)))



## === cell 24
raw_pred_log = lr.predict(test_df.drop("key", axis=1))

raw_pred = np.expm1(raw_pred_log)

pred = np.asarray(raw_pred, dtype=np.float64)
pred = np.nan_to_num(pred, nan=11.35, posinf=250.0, neginf=0.0)
pred = np.clip(pred, 0.0, 250.0)
pred = np.round(pred, 2)

Submission = pd.DataFrame({"key": test_df["key"].values, "fare_amount": pred})
Submission.head()



## === cell 25
Submission.to_csv("Submission.csv", index=False)
print("Wrote Submission.csv with shape:", Submission.shape)
print("fare_amount stats:", Submission["fare_amount"].describe())
