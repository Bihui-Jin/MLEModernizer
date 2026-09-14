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

5.52554

# 6. Current score

1091.98307

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1070.83058) has done: 'Your RMSE is exploding mainly because the boolean filter in cell 10 is wrong due to operator precedence (`!=` binds before `&`), so you’re effectively selecting the wrong rows (including lots of garbage) for training. I fix that mask with proper parentheses (core logic unchanged) and apply the same basic cleaning you already use (NYC bounding box + nonzero passenger_count) before training so the regression isn’t dominated by outliers. I also make the train/test split deterministic to stabilize results, and ensure the submission path is valid and predictions are clipped to non-negative fares (a safe constraint that typically reduces RMSE). These are minimal, directly score-relevant fixes and still run quickly on 2M rows.'
- What this solution (achieved 734.55375) has done: 'Your current RMSE (1070) is far worse than the target (5.53), so we should make the smallest safe changes that reduce obvious label-noise/outlier influence without changing the model or feature set. The biggest practical issue here is that you train only on pickups within 15 km of downtown, but you do not apply the same filter to test data; that distribution mismatch can blow up errors, so we apply the same downtown-distance rule to test predictions in a neutral way by falling back to a robust “typical fare” for out-of-scope rows. We also add two standard NYC Taxi Fare data-cleaning filters that keep core logic intact but remove extreme/garbage rows: valid passenger_count range and a reasonable upper bound for fare and distance. Finally, we keep the exact same LinearRegression and features, ensure alignment to the submission format, and still write `submission.csv`.'
- What this solution (achieved 7.39365) has done: 'Your RMSE is still orders of magnitude above target, so we should make the smallest changes that remove obvious train/test feature mismatch and label-noise without changing the model or feature set. The biggest remaining issue is that you filter training rows to an NYC bounding box but you do not apply the same box filter to test-time logic; this creates extreme “distance” values in test that linear regression extrapolates badly, exploding RMSE. I compute the same bounding-box mask for test and route out-of-box rows to the same robust fallback fare you already use (just like your downtown-distance rule), keeping core logic intact. I also make the downtown “rush hour” boolean unambiguous with parentheses (semantics-preserving but avoids precedence surprises) and ensure fallback_fare is computed after all training filters so it matches the prediction domain.'
- What this solution (achieved 7.39717) has done: 'Your current RMSE (7.39) is worse than the target (5.53), so we make the smallest changes that typically improve LinearRegression on this competition without changing the model or features. The biggest remaining score driver is label noise/outliers: we add two standard, lightweight filters (reasonable coordinate sanity and “fare-per-km” plausibility) to remove bad training rows that cause bad extrapolation. We also make the train/test “in_domain” definition consistent with the training subset by requiring the same passenger_count range (1–6) instead of just `!= 0`. Finally, we compute the fallback fare from the same cleaned training subset and use the same feature preparation, keeping the rest intact.'
- What this solution (achieved 1091.98307) has done: 'We need to reduce RMSE from 7.397 toward 5.526 (lower is better), so the smallest safe move is to make training/test domain handling more consistent without changing the model or feature set. Right now you train only on `distance_to_downtown < 15` but you *predict* for all test rows and then hard-fallback out-of-domain to a constant median; that discontinuity can hurt RMSE near the boundary. I keep the same LinearRegression and features, but (1) compute a simple “domain” flag for the training subset and fit on all cleaned rows with an added binary feature is not allowed, so instead (2) learn a second LinearRegression on the *same features* but trained on all cleaned data (no downtown filter) and use it only for out-of-domain test rows—this preserves core logic (same model type, same features, same loss) and typically reduces errors on out-of-domain rows compared to a constant fallback. I also compute `fallback_fare` from the *in-domain* subset only (since it is only used there as safety), keeping it representative, and keep all existing filters and submission format unchanged.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # ploting library with python
from sklearn.linear_model import LinearRegression  # Library for linear regression model
from sklearn.model_selection import train_test_split

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
train_data_set = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/train.csv",
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
    earth_radius = 6371  # radius in km
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

train_data_set["is_rush_hour"] = train_data_set["hour"].apply(
    lambda x: 1 if ((x >= 7 and x <= 10) or (x >= 16 and x <= 19)) else 0
)

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
old_len = len(train_data_set)

coord_sane = (
    (train_data_set["pickup_latitude"].between(40.0, 42.0))
    & (train_data_set["dropoff_latitude"].between(40.0, 42.0))
    & (train_data_set["pickup_longitude"].between(-75.0, -72.0))
    & (train_data_set["dropoff_longitude"].between(-75.0, -72.0))
)

basic_ok = (
    (train_data_set.passenger_count >= 1)
    & (train_data_set.passenger_count <= 6)
    & (train_data_set.fare_amount <= 250.0)
    & (train_data_set.distance > 0.0)
    & (train_data_set.distance <= 100.0)
)

fare_per_km = train_data_set["fare_amount"] / train_data_set["distance"]
fare_km_ok = fare_per_km.between(0.5, 60.0)

train_data_set = train_data_set[coord_sane & basic_ok & fare_km_ok]

new_len = len(train_data_set)
print(f"Removed {(old_len-new_len)} entities from the dataset (outlier filter)")

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

idx_in_domain = (
    (train_data_set.passenger_count >= 1)
    & (train_data_set.passenger_count <= 6)
    & (train_data_set.distance_to_downtown < 15)
)

X_in = train_data_set.loc[idx_in_domain, features].values
y_in = train_data_set.loc[idx_in_domain, target].values

X_all = train_data_set.loc[:, features].values
y_all = train_data_set.loc[:, target].values

X_train, X_test, y_train, y_test = train_test_split(
    X_in, y_in, test_size=0.25, random_state=42
)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

linear_model_all = LinearRegression()
linear_model_all.fit(X_all, y_all)

fallback_fare = float(np.median(y_in))
print("Fallback fare (median of in-domain training subset):", fallback_fare)
print("In-domain training rows:", len(y_in), "All-cleaned training rows:", len(y_all))



## === cell 11
test_data_set = pd.read_csv("../input/new-york-city-taxi-fare-prediction/test.csv")

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
test_data_set["pickup_datetime"] = pd.to_datetime(test_data_set["pickup_datetime"])
test_data_set["hour"] = test_data_set["pickup_datetime"].dt.hour
test_data_set["year"] = test_data_set["pickup_datetime"].dt.year
test_data_set["day_of_week"] = test_data_set["pickup_datetime"].dt.dayofweek

test_data_set["is_rush_hour"] = test_data_set["hour"].apply(
    lambda x: 1 if ((x >= 7 and x <= 10) or (x >= 16 and x <= 19)) else 0
)



## === cell 12
XTEST = test_data_set[features].values

y_pred_in = linear_model.predict(XTEST)

y_pred_all = linear_model_all.predict(XTEST)

in_box = select_within_boundingbox(test_data_set, new_york_box)

in_domain = (
    (test_data_set["passenger_count"] >= 1)
    & (test_data_set["passenger_count"] <= 6)
    & (test_data_set["distance_to_downtown"] < 15)
    & in_box
)

y_pred_final = np.where(in_domain.values, y_pred_in, y_pred_all)

y_pred_final = np.where(np.isfinite(y_pred_final), y_pred_final, fallback_fare)
y_pred_final = np.clip(y_pred_final, 0.0, None)

submission = pd.DataFrame(
    {"key": test_data_set.key, "fare_amount": y_pred_final},
    columns=["key", "fare_amount"],
)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with rows:", len(submission))
