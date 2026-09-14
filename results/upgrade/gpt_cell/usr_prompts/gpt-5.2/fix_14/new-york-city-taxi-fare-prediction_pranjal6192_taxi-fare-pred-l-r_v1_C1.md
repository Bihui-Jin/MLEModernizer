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

5.68914

# 6. Current score

11.44829

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 936.92806) has done: 'Diagnosis: The crash occurs because `LinearRegression(normalize=True)` is no longer a valid argument in scikit-learn 1.2.2; the `normalize` parameter was deprecated and then removed, so passing it raises `TypeError`. The rest of the code expects `lr` to be a fitted `LinearRegression` instance and uses `.score()` and later `.predict()` in cell 34. Removing the unsupported argument fixes execution while preserving the same regression model class and training/evaluation semantics.

Patch summary: In cell 33, instantiate `LinearRegression()` without the removed `normalize` keyword and keep the subsequent `.fit()` and `.score()` calls unchanged.

Updated cells: (only cell 33)

Compatibility notes for cell k+1: `lr` remains a trained `LinearRegression` model, so `lr.predict(...)` in cell 34 work exactly as before.

Assumptions: No other cells rely on the removed `normalize` behavior; since no preprocessing pipeline is defined elsewhere, the minimal fix is to drop the invalid parameter rather than add new scaling logic.'
- What this solution (achieved 936.92806) has done: 'Your RMSE is extremely high because the model is being trained with numeric weekday one-hot columns but the test set ends up with a different set/order of weekday dummy columns (due to `get_dummies` being applied separately), so predictions are computed on misaligned features. I minimally fix this by creating weekday dummies with a fixed 7-day category order for both train and test, and then explicitly reindexing the test feature matrix to exactly match the training feature columns before calling `predict()`. This preserves your exact feature engineering and the same `LinearRegression` core logic, but removes the train/test feature mismatch that is causing the blow-up in error. The submission writing remains the same, still producing `Submission.csv` with `key,fare_amount`.'
- What this solution (achieved 752.76803) has done: 'Your score is far from the target (lower is better), and the biggest likely cause is inconsistent feature scaling: you normalize `Difference_longitude/latitude` using each dataset’s own mean/variance, which makes train vs test distributions mismatched and can explode RMSE. I make the smallest change that preserves your exact model/feature set: compute the mean/variance from the training set only, then apply that same scaling to both train and test. I also add a minimal safety clamp on negative fare predictions (fares can’t be negative), which typically reduces RMSE without changing the core model. The rest of your pipeline (features, LinearRegression, submission format) stays the same.'
- What this solution (achieved 868.88676) has done: 'Your RMSE is still huge because the training data likely contains many invalid/outlier fares and coordinates (including negative/zero fares and trips far outside NYC), which a plain LinearRegression fit poorly and then generalize badly. I make minimal, competition-standard data cleaning filters (fares within a sensible range, passenger_count valid, and coordinates within an NYC bounding box) while keeping your exact feature engineering and LinearRegression model unchanged. I also replace the variance-based scaling with standard-deviation scaling (still train-derived, same feature, same semantics) to avoid overly shrinking features, which can destabilize coefficients. Finally, I keep the train/test dummy alignment and submission format exactly as required.'
- What this solution (achieved 998.07345) has done: 'Your score is still extremely far from the target (lower is better), so we need a small-but-impactful correction rather than tuning. The biggest remaining issue is that you scale `Difference_longitude/latitude` incorrectly by taking an absolute deviation from the mean, which destroys sign/direction information and makes the features nonlinearly distorted for a LinearRegression. I change that to standard z-score scaling `(x-mean)/std` using train-derived stats (same feature, same model) and keep your train/test column alignment and submission format unchanged. This should substantially reduce RMSE while preserving the same overall approach and runtime.'
- What this solution (achieved 998.07262) has done: 'Your RMSE is still catastrophically high because the model is trained on a large random slice that likely still includes hard outliers (despite your bounds), and LinearRegression is extremely sensitive to those. With minimal change and without altering your feature set or training loop, I switch the regressor to `Ridge` (same linear model family, same `.fit/.predict` semantics) to stabilize coefficients via L2 regularization; this typically reduces blow-ups and moves RMSE dramatically toward the target. I also add a small, safe cleanup: ensure all feature columns are numeric and replace any inf/NaN created by transformations before fitting/predicting, which otherwise can silently corrupt training. Submission format and all feature engineering steps remain unchanged.'
- What this solution (achieved 998.07418) has done: 'Your current RMSE is still extremely far from the target, so we need a small but high-impact correction while keeping your linear model and engineered features intact. The biggest remaining issue is that the `Distance` and airport-distance features are computed with `R=6373` (km) and then multiplied by `0.621` (km→miles), but `R=6373` is already km and should be `6371`, and more importantly your conversion is inconsistent with typical fare scaling; this can distort the learned coefficients badly. I minimally fix the distance computations to use a consistent earth radius in kilometers (`6371.0`) and convert to miles using the standard factor (`0.621371`), leaving everything else (feature set, Ridge training, split, cleaning, submission) unchanged. This should materially reduce prediction scale errors and move RMSE strongly toward the target.'
- What this solution (achieved 1126.35552) has done: 'Your RMSE is still astronomically worse than the target, so we need a minimal correction that preserves your linear model and engineered features but fixes a key semantic bug: you’re taking `abs()` for the coordinate deltas, which destroys directionality and makes the model systematically mis-estimate fares. I remove the `abs()` so the deltas keep their sign (then the existing z-score scaling remains meaningful), and I add a very small safety filter that removes physically impossible trips (zero/near-zero distance with positive fare) that can destabilize a linear fit. Everything else (Ridge model, distance features, weekday one-hot alignment, train-derived scaling, submission writing) stays the same to keep core logic intact and runtime within limits.'
- What this solution (achieved 1130.5415) has done: 'Your current RMSE (1126) is far worse than the target (5.689), so we need a small, high-impact correction without changing your overall linear-model approach. The biggest remaining issue is that you rounded the distance-based features to 2 decimals before training/predicting, which throws away essential signal and can severely hurt a linear model’s fit; removing this rounding keeps the same features but restores precision. I also make the train/test feature alignment stricter by reindexing the submit matrix to the full training feature set `X.columns` (not the split `X_train.columns`) to avoid any rare column-order edge cases. Everything else (cleaning, features, Ridge regression, submission format) stays the same.'
- What this solution (achieved 1127.01348) has done: 'Your RMSE is still enormous because the `Distance` and airport distance features are in miles while the model is learning dollar fares; that mismatch plus removing the raw coordinates makes the linear model poorly conditioned and leads to extreme errors. I make a minimal, semantics-preserving fix by keeping the exact same haversine computations but expressing all distances consistently in kilometers (no miles conversion) to bring feature scales closer to the fare signal. I also stop rounding predictions (rounding can only add noise for RMSE) while keeping the non-negativity clip, and I ensure the submission is written with the required `key,fare_amount` columns (no index) for maximum compatibility. Core model family (Ridge) and all existing feature engineering steps remain intact.'
- What this solution (achieved 1127.01348) has done: 'Your current RMSE is catastrophically worse than the target, which strongly suggests a train/test mismatch or a bad training sample rather than a “tuning” issue. I keep your exact feature set and Ridge model, but make two minimal corrections that typically cause huge RMSE when wrong: (1) ensure the one-hot weekday columns are created in a fixed 7-day schema and are identical between train and test (including missing days), and (2) fix the datetime parsing to use vectorized `pd.to_datetime` (your string slicing is brittle and can silently produce wrong weekdays/times). I also keep your existing cleaning and scaling logic, but add a tiny safety step to drop any rows that became invalid after feature creation (NaN/inf in features), which otherwise can destabilize coefficients and explode predictions. The rest of the pipeline (10M rows, Ridge, same engineered columns, submission format/path) stays the same.'
- What this solution (achieved 865.28611) has done: 'Your current RMSE (1127) is so far from the target (5.689) that the main issue is almost certainly a semantic mismatch that makes predictions wildly wrong rather than “slightly suboptimal.” The smallest high-impact fix while preserving your linear model approach is to stop dropping the raw lat/long coordinates: a linear model can’t reconstruct geography well from just deltas and distances, and this often causes extreme errors. I keep all your existing cleaning, datetime/weekday one-hot, distance features, scaling, Ridge training, and submission format; the only functional change is retaining the coordinate columns as additional features and ensuring the submit matrix aligns to `X.columns`. This should drastically reduce RMSE toward the target without changing the overall pipeline.'
- What this solution (achieved 11.44829) has done: 'Your RMSE is still massively worse than the target, so we need one high-impact correction while keeping the same overall feature set and linear Ridge approach. The biggest remaining semantic issue is that `pickuptime` is encoded as `HH*100+MM` (e.g., 930, 1730), which is a discontinuous scale and hurts linear models; converting it to “minutes since midnight” keeps the same idea but makes the relationship linear and stable. I also apply the same train-derived z-score scaling you already do for deltas to the `Distance` and airport-distance features (using train stats), so train/test are on the same scale and Ridge coefficients don’t blow up. Finally, I add a minimal sanity clamp for unrealistically large predicted fares (keeps validity and typically reduces RMSE explosions) while still writing the same `Submission.csv` format.'

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



## === cell 8
print(f"Before Dropping null values: {len(train_data)}")
train_data.dropna(inplace=True)
print(f"After Dropping null values: {len(train_data)}")



## === cell 9
plot = train_data[:2000].plot.scatter("Difference_longitude", "Difference_latitude")



## === cell 10
train_data = train_data[
    (train_data["fare_amount"] > 0) & (train_data["fare_amount"] <= 250)
]
train_data = train_data[
    (train_data["passenger_count"] >= 1) & (train_data["passenger_count"] <= 6)
]

train_data = train_data[
    (train_data["pickup_longitude"].between(-74.5, -72.8))
    & (train_data["dropoff_longitude"].between(-74.5, -72.8))
    & (train_data["pickup_latitude"].between(40.3, 41.8))
    & (train_data["dropoff_latitude"].between(40.3, 41.8))
]

train_data = train_data[
    (train_data["Difference_longitude"].abs() < 5.0)
    & (train_data["Difference_latitude"].abs() < 5.0)
]



## === cell 11
train_dt = pd.to_datetime(train_data["pickup_datetime"], errors="coerce", utc=True)
test_dt = pd.to_datetime(test_data["pickup_datetime"], errors="coerce", utc=True)

train_data["pickuptime"] = (train_dt.dt.hour * 60 + train_dt.dt.minute).astype(
    "float64"
)
test_data["pickuptime"] = (test_dt.dt.hour * 60 + test_dt.dt.minute).astype("float64")

train_data["Weekday"] = train_dt.dt.weekday.astype("float64")
test_data["Weekday"] = test_dt.dt.weekday.astype("float64")



## === cell 12
train_data.head()



## === cell 13
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



## === cell 14
weekday_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
train_data["Weekday"] = pd.Categorical(train_data["Weekday"], categories=weekday_order)
test_data["Weekday"] = pd.Categorical(test_data["Weekday"], categories=weekday_order)

train_one_hot = pd.get_dummies(train_data["Weekday"])
test_one_hot = pd.get_dummies(test_data["Weekday"])

train_one_hot = train_one_hot.reindex(columns=weekday_order, fill_value=0)
test_one_hot = test_one_hot.reindex(columns=weekday_order, fill_value=0)

train_data = pd.concat([train_data, train_one_hot], axis=1)
test_data = pd.concat([test_data, test_one_hot], axis=1)



## === cell 15
train_data.drop("Weekday", axis=1, inplace=True)
test_data.drop("Weekday", axis=1, inplace=True)



## === cell 16
train_data.drop("pickup_datetime", inplace=True, axis=1)
test_data.drop("pickup_datetime", inplace=True, axis=1)



## === cell 17
train_data.head()



## === cell 18
test_data.head()



## === cell 19
R = 6371.0  # kilometers

lat1 = np.asarray(np.radians(train_data["pickup_latitude"]))
lon1 = np.asarray(np.radians(train_data["pickup_longitude"]))
lat2 = np.asarray(np.radians(train_data["dropoff_latitude"]))
lon2 = np.asarray(np.radians(train_data["dropoff_longitude"]))

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
train_data["Distance"] = np.asarray(distance)  # km

lat1 = np.asarray(np.radians(test_data["pickup_latitude"]))
lon1 = np.asarray(np.radians(test_data["pickup_longitude"]))
lat2 = np.asarray(np.radians(test_data["dropoff_latitude"]))
lon2 = np.asarray(np.radians(test_data["dropoff_longitude"]))

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
test_data["Distance"] = np.asarray(distance)  # km



## === cell 20
R = 6371.0  # kilometers

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
train_data["Pickup_Distance_airport"] = np.asarray(distance1)  # km

a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
train_data["Dropoff_Distance_airport"] = np.asarray(distance2)  # km

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
test_data["Pickup_Distance_airport"] = np.asarray(distance1)  # km

a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
test_data["Dropoff_Distance_airport"] = np.asarray(distance2)  # km



## === cell 21
pass



## === cell 22
train_data = train_data[
    ~((train_data["Distance"] < 0.01) & (train_data["fare_amount"] > 2.5))
]



## === cell 23
dlon_mean = np.mean(train_data["Difference_longitude"])
dlon_std = np.std(train_data["Difference_longitude"])
dlat_mean = np.mean(train_data["Difference_latitude"])
dlat_std = np.std(train_data["Difference_latitude"])

if dlon_std == 0:
    dlon_std = 1.0
if dlat_std == 0:
    dlat_std = 1.0

dist_mean = np.mean(train_data["Distance"])
dist_std = np.std(train_data["Distance"])
pda_mean = np.mean(train_data["Pickup_Distance_airport"])
pda_std = np.std(train_data["Pickup_Distance_airport"])
dda_mean = np.mean(train_data["Dropoff_Distance_airport"])
dda_std = np.std(train_data["Dropoff_Distance_airport"])

for _m, _s_name in [
    (dist_mean, "dist_std"),
    (pda_mean, "pda_std"),
    (dda_mean, "dda_std"),
]:
    pass

if dist_std == 0:
    dist_std = 1.0
if pda_std == 0:
    pda_std = 1.0
if dda_std == 0:
    dda_std = 1.0



## === cell 24
train_data["Difference_longitude"] = (
    train_data["Difference_longitude"] - dlon_mean
) / dlon_std
train_data["Difference_latitude"] = (
    train_data["Difference_latitude"] - dlat_mean
) / dlat_std

train_data["Distance"] = (train_data["Distance"] - dist_mean) / dist_std
train_data["Pickup_Distance_airport"] = (
    train_data["Pickup_Distance_airport"] - pda_mean
) / pda_std
train_data["Dropoff_Distance_airport"] = (
    train_data["Dropoff_Distance_airport"] - dda_mean
) / dda_std



## === cell 25
test_data["Difference_longitude"] = (
    test_data["Difference_longitude"] - dlon_mean
) / dlon_std
test_data["Difference_latitude"] = (
    test_data["Difference_latitude"] - dlat_mean
) / dlat_std

test_data["Distance"] = (test_data["Distance"] - dist_mean) / dist_std
test_data["Pickup_Distance_airport"] = (
    test_data["Pickup_Distance_airport"] - pda_mean
) / pda_std
test_data["Dropoff_Distance_airport"] = (
    test_data["Dropoff_Distance_airport"] - dda_mean
) / dda_std



## === cell 26
train_data.shape



## === cell 27
test_data.shape



## === cell 28
from sklearn.model_selection import train_test_split

X = train_data.drop(["key", "fare_amount"], axis=1)
y = train_data["fare_amount"]

X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
mask_valid = X.notna().all(axis=1) & y.notna()
X = X.loc[mask_valid].fillna(0)
y = y.loc[mask_valid]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.01, random_state=80
)



## === cell 29
from sklearn.linear_model import Ridge

lr = Ridge(alpha=1.0, random_state=80)
lr.fit(X_train, y_train)
print(lr.score(X_test, y_test))



## === cell 30
X_test_submit = test_data.drop("key", axis=1)

X_test_submit = X_test_submit.reindex(columns=X.columns, fill_value=0)

X_test_submit = (
    X_test_submit.apply(pd.to_numeric, errors="coerce")
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)

pred = lr.predict(X_test_submit)

pred = np.clip(pred, 0, 250)



## === cell 31
pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
).head()



## === cell 32
Submission = pd.DataFrame({"key": test_data["key"], "fare_amount": pred})
Submission = Submission[["key", "fare_amount"]]
Submission.head()



## === cell 33
Submission.to_csv("Submission.csv", index=False)
print("Wrote Submission.csv with shape:", Submission.shape)
