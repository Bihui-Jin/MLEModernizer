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

3.12

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

5.53411

# 6. Current score

14.09462

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1070.83058) has done: 'Your RMSE is exploding because the training filter `idx = train_data_set.passenger_count != 0 & (...)` is using the wrong operator precedence, so you end up training on a badly selected (almost arbitrary) subset. I fix that boolean mask with proper parentheses (minimal change, same model/feature logic) and also apply the same basic sanity cleaning to the test set (drop NaNs and clip negative predictions), which typically prevents huge errors. I also make the train/test split deterministic via `random_state` to stabilize the score while keeping the same training approach. These changes should move RMSE drastically down toward your target without changing the core modeling logic.'
- What this solution (achieved 1070.83058) has done: 'Your current RMSE is so large because the submission rows likely don’t align 1:1 with the required `test.csv` keys: you drop NaNs in the test set, which changes the row count/order and causes Kaggle to score mismatched predictions. I keep the same linear regression, same features, and same training logic, but I stop dropping test rows and instead fill missing feature values using medians computed from the filtered training data (so every test key gets a prediction). I also apply the same NYC bounding-box filter to training only (as you already do) and keep deterministic splitting; this should drastically reduce RMSE toward your target without changing the core approach. Finally, I ensure the submission has exactly the same number of rows as `test.csv` and the correct columns.'
- What this solution (achieved 37.89764) has done: 'I make the smallest changes needed to (1) ensure the notebook runs in your environment paths and always writes a valid `submission.csv`, and (2) reduce RMSE from “not yielded” toward your 5.534 target by preventing catastrophic training noise and out-of-distribution artifacts. Concretely, I fix the input path to match the provided `/kaggle/input/...` structure, add two standard NYC Taxi Fare sanity filters (valid passenger_count range and reasonable fare upper bound) to stabilize the linear regression without changing the model/feature logic, and ensure all numeric columns are safely coerced with consistent median-imputation for train/test. These are minimal, metric-aligned data-cleaning adjustments that typically improve RMSE dramatically while preserving your exact feature set, log1p target transform, and LinearRegression training approach.'
- What this solution (achieved 18.57277) has done: 'Your current RMSE (37.90) is far worse than the 5.53 target, so we should improve but with the smallest changes that don’t alter the core model/features. The biggest low-risk win here is to apply the same NYC bounding-box sanity filter to the **training mask** used for fitting (you currently compute the box filter earlier, but the final `idx` mask doesn’t explicitly include it), which reduces extreme/invalid coordinates that a linear model can’t handle well. I also add a very small additional training-only filter to remove near-zero/implausibly huge trip distances (computed from your existing distance feature), which typically eliminates the worst outliers without changing feature engineering or the regression itself. Finally, I keep the test set unfiltered (to preserve 1:1 key alignment) and keep your existing median imputation and clipping.'
- What this solution (achieved 18.39255) has done: 'Your RMSE (18.57) is still far above the 5.53 target, so we should improve with minimal, low-risk changes that keep the same model (LinearRegression on the same engineered features with log1p target). The biggest remaining stability issue is that the linear model is trained on a filtered subset, but the test set can still contain out-of-distribution coordinates/passenger counts; instead of dropping test rows (which would break alignment), we “snap” clearly invalid test coordinates into the NYC bounding box and clamp passenger_count to [1, 6] so the feature distribution better matches training. We also compute train medians on the final numeric training matrix and use them consistently, plus add a tiny numeric safety for distance computations (inf/NaN) without changing feature definitions. These changes usually reduce large errors from pathological test rows while preserving your exact training approach and submission semantics.'
- What this solution (achieved 16.49598) has done: 'Your current RMSE (18.39) is far worse than the 5.53 target, so we should improve with the smallest changes that keep your exact model (LinearRegression on the same engineered features with log1p target). The biggest remaining issue is distribution mismatch between train and test: you “snap” test coordinates/passengers into reasonable ranges, but you *don’t* apply the same snapping before computing training features (you drop out-of-box rows instead), and you keep some “bad but in-box” rows (e.g., distance_to_downtown < 15) that make the linear fit unstable. I (1) add a minimal training-side coordinate/passenger clamp identical to test before feature computation, and (2) tighten the training mask slightly by excluding clearly invalid times (NaT-derived features) and extreme `distance_to_downtown` values rather than only `<15`, which typically reduces big residual outliers without changing features or the learning algorithm. This should move RMSE materially down toward the target band while preserving your core logic and submission alignment.'
- What this solution (achieved 16.75352) has done: 'Your current RMSE (16.50) is far above the 5.53 target, so we should improve with the smallest changes that keep the same LinearRegression-on-log1p target and the same feature set. The biggest low-risk gain is to remove remaining heavy outliers in the *training* set that a linear model can’t fit (especially “high fare for short distance” and “low fare for long distance”), by adding two simple, training-only ratio-based filters using your existing `distance` and `fare_amount` columns. I also ensure we don’t accidentally train on rows with non-positive/NaN `distance` after recomputation, and keep the test pipeline unchanged so submission alignment stays 1:1 with `test.csv`. These changes typically reduce RMSE materially without altering model architecture, features, loss, or the log-transform semantics.'
- What this solution (achieved 16.75352) has done: 'Your current RMSE (16.75) is still far above the 5.53 target, so we should improve with the smallest, safest changes that keep your exact model (LinearRegression on log1p target) and the same engineered features. The biggest remaining issue is that LinearRegression is very sensitive to feature scale and multicollinearity (you have several highly correlated geospatial/time features), which can yield unstable coefficients and large errors even after filtering; switching to Ridge is a minimal change that preserves the same linear modeling logic while stabilizing the fit. I keep the entire feature engineering and training pipeline intact, and only replace the estimator with a lightly-regularized Ridge plus a tiny fixed alpha to reduce overfitting/outlier sensitivity. Submission alignment remains 1:1 with test keys and the same post-processing (expm1 + clipping) is preserved.'
- What this solution (achieved 16.75349) has done: 'Your current RMSE (16.75) is still far above the 5.53 target, so we should improve it with the smallest changes that keep the same overall linear-on-engineered-features approach. The most likely remaining driver of large error is that you’re fitting a plain linear model on heterogeneously-scaled features (year vs km vs binary), so I add a `StandardScaler` on the feature matrix (train+test consistently) to stabilize the linear fit without changing the feature set or training loop. I also make the Ridge regularization slightly stronger (small alpha bump) to reduce sensitivity to remaining outliers/multicollinearity while preserving the same model family and log1p/expm1 semantics. Submission alignment and row count remain 1:1 with `test.csv`, and we still write `submission.csv`.'
- What this solution (achieved 14.09462) has done: 'Your current RMSE (16.75) is far above the 5.53 target, so we should improve it with the smallest changes that preserve your exact linear-on-engineered-features approach. The main low-risk gain now is to make the *training* objective match RMSE more closely: instead of log1p-transforming the target (which changes the loss geometry), fit Ridge directly on raw `fare_amount` while keeping the same features, scaling, filtering, and prediction pipeline. I also keep your existing sanity filters and 1:1 test alignment intact, and only adjust the final clipping upper bound to match your training fare filter (250) consistently. This should move RMSE materially downward without changing the feature set, train/test handling, or overall model family.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # ploting library with python
from sklearn.linear_model import LinearRegression  # Library for linear regression model
from sklearn.linear_model import Ridge  # minimal regularization to stabilize linear fit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler,
)  # CHANGE: stabilize linear fit via feature scaling

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
TRAIN_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
TEST_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"

train_data_set = pd.read_csv(
    TRAIN_PATH,
    nrows=2_000_000,
    parse_dates=["pickup_datetime"],
)
train_data_set.head(5)



## === cell 2
print(train_data_set.dtypes)
train_data_set.describe()



## === cell 3
old_len = len(train_data_set)
train_data_set = train_data_set[train_data_set.fare_amount >= 0.1]
train_data_set = train_data_set[train_data_set.fare_amount <= 250.0]
new_len = len(train_data_set)
print(f"Removed {(old_len-new_len)} entities from the dataset")
train_data_set.describe()



## === cell 4
old_len = len(train_data_set)
train_data_set = train_data_set.dropna(how="any", axis="rows")
new_len = len(train_data_set)
print(f"Removed {(old_len-new_len)} entities from the dataset")



## === cell 5
train_data_set.fare_amount.hist(bins=100, figsize=(14, 3))
plt.xlabel("fare $USD")
plt.title("Histogram")




## === cell 6
def select_within_boundingbox(df, box):
    return (
        (df.pickup_longitude >= box[0])
        & (df.pickup_longitude <= box[1])
        & (df.pickup_latitude >= box[2])
        & (df.pickup_latitude <= box[3])
        & (df.dropoff_longitude >= box[0])
        & (df.dropoff_longitude <= box[1])
        & (df.dropoff_latitude >= box[2])
        & (df.dropoff_latitude <= box[3])
    )


new_york_box = (-74.763379, -72.856164, 40.502009, 41.915509)

old_len = len(train_data_set)
train_data_set = train_data_set[select_within_boundingbox(train_data_set, new_york_box)]
new_len = len(train_data_set)
print(f"Removed {(old_len-new_len)} entities from the dataset")




## === cell 7
def distance_on_the_sphere(lat1, lon1, lat2, lon2):
    earth_radius = 6371  # earth radius in km
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2.0) * np.sin(delta_phi / 2.0) + np.cos(phi1) * np.cos(
        phi2
    ) * np.sin(delta_lambda / 2.0) * np.sin(delta_lambda / 2.0)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return earth_radius * c


train_data_set["distance"] = distance_on_the_sphere(
    train_data_set["pickup_latitude"],
    train_data_set["pickup_longitude"],
    train_data_set["dropoff_latitude"],
    train_data_set["dropoff_longitude"],
)

train_data_set.head(5)



## === cell 8
train_data_set["pickup_datetime"] = pd.to_datetime(train_data_set["pickup_datetime"])
train_data_set["hour"] = train_data_set["pickup_datetime"].dt.hour
train_data_set["year"] = train_data_set["pickup_datetime"].dt.year
train_data_set["day_of_week"] = train_data_set["pickup_datetime"].dt.dayofweek

h = train_data_set["hour"]
train_data_set["is_rush_hour"] = (
    ((h >= 7) & (h <= 10)) | ((h >= 16) & (h <= 19))
).astype(np.int8)

train_data_set.head(5)



## === cell 9
nyc_down_town = (-74.0063889, 40.7141667)

train_data_set["distance_to_downtown"] = distance_on_the_sphere(
    nyc_down_town[1],
    nyc_down_town[0],
    train_data_set.pickup_latitude,
    train_data_set.pickup_longitude,
)

train_data_set.head(5)



## === cell 10
lon_min, lon_max, lat_min, lat_max = new_york_box
for col, lo, hi in [
    ("pickup_longitude", lon_min, lon_max),
    ("dropoff_longitude", lon_min, lon_max),
    ("pickup_latitude", lat_min, lat_max),
    ("dropoff_latitude", lat_min, lat_max),
]:
    train_data_set[col] = pd.to_numeric(train_data_set[col], errors="coerce")
    train_data_set[col] = train_data_set[col].clip(lo, hi)

train_data_set["passenger_count"] = pd.to_numeric(
    train_data_set["passenger_count"], errors="coerce"
).clip(1, 6)

train_data_set["distance"] = distance_on_the_sphere(
    train_data_set["pickup_latitude"],
    train_data_set["pickup_longitude"],
    train_data_set["dropoff_latitude"],
    train_data_set["dropoff_longitude"],
)
train_data_set["distance_to_downtown"] = distance_on_the_sphere(
    nyc_down_town[1],
    nyc_down_town[0],
    train_data_set.pickup_latitude,
    train_data_set.pickup_longitude,
)

bbox_mask = select_within_boundingbox(train_data_set, new_york_box)

idx = (
    bbox_mask
    & (train_data_set.passenger_count != 0)
    & (train_data_set.passenger_count <= 6)
    & (train_data_set.hour.notna())
    & (train_data_set.year.notna())
    & (train_data_set.day_of_week.notna())
    & (train_data_set.distance_to_downtown > 0.01)
    & (train_data_set.distance_to_downtown < 30.0)
    & (
        train_data_set.distance > 0.05
    )  # ~50m minimum: removes zero-distance / bad coords
    & (train_data_set.distance < 80.0)  # removes extreme outliers within the box
)

safe_dist = train_data_set["distance"].replace([np.inf, -np.inf], np.nan)
safe_fare = train_data_set["fare_amount"].replace([np.inf, -np.inf], np.nan)

fare_per_km = safe_fare / (safe_dist + 1e-3)

idx = (
    idx
    & safe_dist.notna()
    & (safe_dist > 0.05)
    & safe_fare.notna()
    & (fare_per_km >= 1.0)  # too-cheap long rides are often bad rows
    & (fare_per_km <= 80.0)  # too-expensive short rides are often bad rows
)

features = [
    "hour",
    "year",
    "distance",
    "passenger_count",
    "is_rush_hour",
    "day_of_week",
    "distance_to_downtown",
]
target = "fare_amount"

train_features_df = train_data_set.loc[idx, features].copy()

for c in features:
    train_features_df[c] = pd.to_numeric(train_features_df[c], errors="coerce")
train_features_df = train_features_df.replace([np.inf, -np.inf], np.nan)

train_feature_medians = train_features_df.median(numeric_only=True)
train_features_df = train_features_df.fillna(train_feature_medians)

X = train_features_df.values
y = train_data_set.loc[idx, target].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

linear_model = Ridge(alpha=3.0, random_state=42)
linear_model.fit(X_train_scaled, y_train)



## === cell 11
test_data_set = pd.read_csv(TEST_PATH)
test_data_set = test_data_set.copy()

test_keys = test_data_set["key"].values

lon_min, lon_max, lat_min, lat_max = new_york_box
for col, lo, hi in [
    ("pickup_longitude", lon_min, lon_max),
    ("dropoff_longitude", lon_min, lon_max),
    ("pickup_latitude", lat_min, lat_max),
    ("dropoff_latitude", lat_min, lat_max),
]:
    test_data_set[col] = pd.to_numeric(test_data_set[col], errors="coerce")
    test_data_set[col] = test_data_set[col].clip(lo, hi)

test_data_set["passenger_count"] = pd.to_numeric(
    test_data_set["passenger_count"], errors="coerce"
).clip(1, 6)

test_data_set["distance"] = distance_on_the_sphere(
    test_data_set["pickup_latitude"],
    test_data_set["pickup_longitude"],
    test_data_set["dropoff_latitude"],
    test_data_set["dropoff_longitude"],
)
test_data_set["distance_to_downtown"] = distance_on_the_sphere(
    nyc_down_town[1],
    nyc_down_town[0],
    test_data_set.pickup_latitude,
    test_data_set.pickup_longitude,
)
test_data_set["pickup_datetime"] = pd.to_datetime(
    test_data_set["pickup_datetime"], errors="coerce"
)
test_data_set["hour"] = test_data_set["pickup_datetime"].dt.hour
test_data_set["year"] = test_data_set["pickup_datetime"].dt.year
test_data_set["day_of_week"] = test_data_set["pickup_datetime"].dt.dayofweek

h_test = test_data_set["hour"]
test_data_set["is_rush_hour"] = (
    ((h_test >= 7) & (h_test <= 10)) | ((h_test >= 16) & (h_test <= 19))
).astype(np.int8)

for c in features:
    test_data_set[c] = pd.to_numeric(test_data_set[c], errors="coerce")
test_data_set[features] = test_data_set[features].replace([np.inf, -np.inf], np.nan)
for c in features:
    test_data_set[c] = test_data_set[c].fillna(train_feature_medians[c])



## === cell 12
XTEST = test_data_set[features].values
XTEST_scaled = scaler.transform(XTEST)

y_pred_final = linear_model.predict(XTEST_scaled)

y_pred_final = np.clip(y_pred_final, 0.0, 250.0)

submission = pd.DataFrame(
    {
        "key": test_keys,
        "fare_amount": y_pred_final.astype(np.float32),
    },
    columns=["key", "fare_amount"],
)

assert submission.shape[0] == len(
    test_keys
), "Submission rowcount must match test.csv rowcount"
assert (
    submission["key"].values == test_keys
).all(), "Submission keys must be aligned with test.csv keys"

print("Test shape:", test_data_set.shape, "Submission shape:", submission.shape)

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
