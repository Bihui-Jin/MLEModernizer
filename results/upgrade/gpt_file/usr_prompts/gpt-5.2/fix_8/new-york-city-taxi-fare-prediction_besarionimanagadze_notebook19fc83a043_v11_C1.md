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

5.51851

# 6. Current score

13.33217

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 28.91348) has done: 'Your score is extremely far from the target, which strongly suggests the submission is being evaluated as catastrophically wrong (often due to row/key misalignment, NaNs/infs, or wildly out-of-distribution predictions). I make minimal, score-relevant fixes: (1) enforce numeric stability and validity for the haversine distance (handle bad coordinates and NaNs), (2) add the same NYC bounding-box filtering logic to test-time feature generation only as *clipping* (not dropping rows) to avoid extreme distances, and (3) clip final predictions to a reasonable non-negative range to avoid RMSE blow-ups from negative/huge fares. These changes keep your linear regression core logic intact and should dramatically reduce RMSE toward the target band while still producing a valid `submission.csv`.'
- What this solution (achieved 12.86831) has done: 'Your RMSE (28.9) is far above the target (5.52), and in this competition that typically happens when the model is systematically mis-specified rather than just “a bit weak”. With minimal changes while keeping your LinearRegression core logic, I (1) fix a key feature bug: you create `day_of_week` and `is_rush_hour` but never use them; adding them to `features` improves signal without changing the modeling approach, (2) apply the same NYC bounding-box logic to the test set as *clipping* (not dropping) to prevent extreme distances from bad coordinates from blowing up predictions, and (3) prevent outlier targets from dominating the linear fit by trimming only extreme `fare_amount` values (a standard cleanup for this dataset) while preserving the same training loop and model. These changes are directly score-relevant and should move RMSE substantially toward the target band while still producing a valid `submission.csv`.'
- What this solution (achieved 15.5249) has done: 'Your current RMSE (12.87) is still far from the 5.52 target, so we should make small, score-relevant feature improvements without changing the model type or training loop. The biggest low-risk gain here is adding standard NYC Taxi engineered features that linear models benefit from: absolute lat/lon deltas and a simple Manhattan-distance proxy, while keeping your existing haversine distance feature. We also ensure train/test feature generation is consistent (same added columns on both) and keep your existing cleaning/clipping so predictions stay sane. These changes typically reduce RMSE materially for this competition while preserving your LinearRegression core logic and still producing a valid `submission.csv`.'
- What this solution (achieved 15.50281) has done: 'Your current RMSE (15.52) is still far above the target (5.52), so the smallest score-relevant fix is to correct a likely feature scaling mismatch for linear regression: longitude/latitude deltas are in degrees, while `distance` is in km, and the current `manhattan` feature (sum of degree deltas) can confuse the fit. I keep your exact model (LinearRegression) and training loop, but replace the degree-based “manhattan” with a km-based Manhattan proxy (lat_km + lon_km using cos(latitude)) and use the km-scaled deltas as features instead of raw degree deltas. This preserves your feature-engineering approach (distance + simple deltas) but makes the features physically consistent, which typically yields a large RMSE drop on this competition. I apply the exact same feature generation to train and test and keep your existing cleaning/clipping and submission writing unchanged.'
- What this solution (achieved 15.52585) has done: 'Your RMSE (15.50) is still far above the target (5.52), so we should make a small but high-impact, still-linear adjustment that typically reduces error on this competition: add an intercept-like “base fare” feature (a constant 1.0 column). This keeps the exact same LinearRegression model and training loop, but helps the linear fit represent the fixed components of taxi pricing (flag drop + surcharges) without forcing them to be explained by distance/time features. I also clamp passenger_count to a reasonable range (1–6) in both train and test to prevent rare values from skewing coefficients, without changing the overall approach. Everything else (feature engineering, cleaning, model type, submission writing) stays the same and it still produce a valid `submission.csv`.'
- What this solution (achieved 13.33217) has done: 'I fix the runtime error in `distance_on_the_sphere` so it correctly handles scalar lat/lon inputs (like JFK’s coordinates) as well as pandas Series, which currently prevents the airport/Manhattan distance features from being created. Once those columns exist, the downstream KeyErrors in feature selection and `dropna` disappear and the script can train and generate predictions. I also make the CSV paths robust for this Kaggle filesystem (keeping your existing relative path working, but falling back to `/kaggle/input/...` if needed) and ensure the output is written as `submission.csv` with the required `key,fare_amount` columns. These changes are execution/validity fixes and preserve your core model and feature logic.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
TRAIN_PATH_CANDIDATES = [
    "../input/new-york-city-taxi-fare-prediction/train.csv",
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
]
train_path = next((p for p in TRAIN_PATH_CANDIDATES if os.path.exists(p)), None)
if train_path is None:
    raise FileNotFoundError(
        f"Could not find train.csv in candidates: {TRAIN_PATH_CANDIDATES}"
    )

train_data_set = pd.read_csv(
    train_path,
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
    earth_radius = 6371.0  # km

    lat1 = np.asarray(pd.to_numeric(lat1, errors="coerce"), dtype="float64")
    lon1 = np.asarray(pd.to_numeric(lon1, errors="coerce"), dtype="float64")
    lat2 = np.asarray(pd.to_numeric(lat2, errors="coerce"), dtype="float64")
    lon2 = np.asarray(pd.to_numeric(lon2, errors="coerce"), dtype="float64")

    lat1 = np.clip(lat1, -90.0, 90.0)
    lat2 = np.clip(lat2, -90.0, 90.0)
    lon1 = np.clip(lon1, -180.0, 180.0)
    lon2 = np.clip(lon2, -180.0, 180.0)

    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = (np.sin(delta_phi / 2.0) ** 2) + (
        np.cos(phi1) * np.cos(phi2) * (np.sin(delta_lambda / 2.0) ** 2)
    )
    a = np.clip(a, 0.0, 1.0)  # numerical safety
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    d = earth_radius * c
    d = np.where(np.isfinite(d), d, np.nan)
    return d


train_data_set["distance"] = distance_on_the_sphere(
    train_data_set["pickup_latitude"],
    train_data_set["pickup_longitude"],
    train_data_set["dropoff_latitude"],
    train_data_set["dropoff_longitude"],
)

train_data_set = train_data_set.dropna(subset=["distance"])
train_data_set.head(5)



## === cell 8
train_data_set["pickup_datetime"] = pd.to_datetime(train_data_set["pickup_datetime"])
train_data_set["hour"] = train_data_set["pickup_datetime"].dt.hour
train_data_set["year"] = train_data_set["pickup_datetime"].dt.year
train_data_set["day_of_week"] = train_data_set["pickup_datetime"].dt.dayofweek
train_data_set["is_rush_hour"] = train_data_set["hour"].apply(
    lambda x: 1 if x >= 7 and x <= 10 or x >= 16 and x <= 19 else 0
)

lat_km_per_deg = 111.32
mean_lat_rad = np.radians(
    (train_data_set["pickup_latitude"] + train_data_set["dropoff_latitude"]) / 2.0
)
cos_mean_lat = np.cos(mean_lat_rad)

train_data_set["abs_lat_diff_km"] = (
    train_data_set["dropoff_latitude"] - train_data_set["pickup_latitude"]
).abs() * lat_km_per_deg
train_data_set["abs_lon_diff_km"] = (
    (train_data_set["dropoff_longitude"] - train_data_set["pickup_longitude"]).abs()
    * lat_km_per_deg
    * cos_mean_lat
)
train_data_set["manhattan_km"] = (
    train_data_set["abs_lat_diff_km"] + train_data_set["abs_lon_diff_km"]
)

train_data_set["log1p_distance"] = np.log1p(train_data_set["distance"].clip(lower=0.0))

JFK = (40.6413, -73.7781)
LGA = (40.7769, -73.8740)
MANHATTAN = (40.7580, -73.9855)  # Times Square-ish

train_data_set["pickup_dist_to_jfk"] = distance_on_the_sphere(
    train_data_set["pickup_latitude"],
    train_data_set["pickup_longitude"],
    JFK[0],
    JFK[1],
)
train_data_set["dropoff_dist_to_jfk"] = distance_on_the_sphere(
    train_data_set["dropoff_latitude"],
    train_data_set["dropoff_longitude"],
    JFK[0],
    JFK[1],
)
train_data_set["pickup_dist_to_lga"] = distance_on_the_sphere(
    train_data_set["pickup_latitude"],
    train_data_set["pickup_longitude"],
    LGA[0],
    LGA[1],
)
train_data_set["dropoff_dist_to_lga"] = distance_on_the_sphere(
    train_data_set["dropoff_latitude"],
    train_data_set["dropoff_longitude"],
    LGA[0],
    LGA[1],
)
train_data_set["pickup_dist_to_manhattan"] = distance_on_the_sphere(
    train_data_set["pickup_latitude"],
    train_data_set["pickup_longitude"],
    MANHATTAN[0],
    MANHATTAN[1],
)
train_data_set["dropoff_dist_to_manhattan"] = distance_on_the_sphere(
    train_data_set["dropoff_latitude"],
    train_data_set["dropoff_longitude"],
    MANHATTAN[0],
    MANHATTAN[1],
)

train_data_set.head(5)



## === cell 9
old_len = len(train_data_set)
train_data_set = train_data_set[
    (train_data_set.fare_amount >= 0.1) & (train_data_set.fare_amount <= 250.0)
]
new_len = len(train_data_set)
print(f"Removed {(old_len-new_len)} extreme-fare entities from the dataset")

train_data_set["passenger_count"] = (
    pd.to_numeric(train_data_set["passenger_count"], errors="coerce")
    .fillna(1)
    .clip(1, 6)
    .astype(int)
)

train_data_set["const"] = 1.0

idx = train_data_set.passenger_count != 0

features = [
    "const",
    "hour",
    "year",
    "day_of_week",
    "is_rush_hour",
    "distance",
    "log1p_distance",
    "abs_lat_diff_km",
    "abs_lon_diff_km",
    "manhattan_km",
    "pickup_dist_to_jfk",
    "dropoff_dist_to_jfk",
    "pickup_dist_to_lga",
    "dropoff_dist_to_lga",
    "pickup_dist_to_manhattan",
    "dropoff_dist_to_manhattan",
    "passenger_count",
]
target = "fare_amount"

train_data_set = train_data_set.dropna(subset=features + [target])

X = train_data_set[idx][features].values
y = train_data_set[idx][target].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)



## === cell 10
TEST_PATH_CANDIDATES = [
    "../input/new-york-city-taxi-fare-prediction/test.csv",
    "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
]
test_path = next((p for p in TEST_PATH_CANDIDATES if os.path.exists(p)), None)
if test_path is None:
    raise FileNotFoundError(
        f"Could not find test.csv in candidates: {TEST_PATH_CANDIDATES}"
    )

test_data_set = pd.read_csv(test_path)

for col, lo, hi in [
    ("pickup_longitude", new_york_box[0], new_york_box[1]),
    ("dropoff_longitude", new_york_box[0], new_york_box[1]),
    ("pickup_latitude", new_york_box[2], new_york_box[3]),
    ("dropoff_latitude", new_york_box[2], new_york_box[3]),
]:
    test_data_set[col] = pd.to_numeric(test_data_set[col], errors="coerce").astype(
        "float64"
    )
    test_data_set[col] = test_data_set[col].clip(lo, hi)

test_data_set["distance"] = distance_on_the_sphere(
    test_data_set["pickup_latitude"],
    test_data_set["pickup_longitude"],
    test_data_set["dropoff_latitude"],
    test_data_set["dropoff_longitude"],
)

test_data_set["pickup_datetime"] = pd.to_datetime(test_data_set["pickup_datetime"])
test_data_set["hour"] = test_data_set["pickup_datetime"].dt.hour
test_data_set["year"] = test_data_set["pickup_datetime"].dt.year
test_data_set["day_of_week"] = test_data_set["pickup_datetime"].dt.dayofweek
test_data_set["is_rush_hour"] = test_data_set["hour"].apply(
    lambda x: 1 if x >= 7 and x <= 10 or x >= 16 and x <= 19 else 0
)

lat_km_per_deg = 111.32
mean_lat_rad = np.radians(
    (test_data_set["pickup_latitude"] + test_data_set["dropoff_latitude"]) / 2.0
)
cos_mean_lat = np.cos(mean_lat_rad)

test_data_set["abs_lat_diff_km"] = (
    test_data_set["dropoff_latitude"] - test_data_set["pickup_latitude"]
).abs() * lat_km_per_deg
test_data_set["abs_lon_diff_km"] = (
    (test_data_set["dropoff_longitude"] - test_data_set["pickup_longitude"]).abs()
    * lat_km_per_deg
    * cos_mean_lat
)
test_data_set["manhattan_km"] = (
    test_data_set["abs_lat_diff_km"] + test_data_set["abs_lon_diff_km"]
)

train_distance_median = float(train_data_set["distance"].median())
test_data_set["distance"] = test_data_set["distance"].fillna(train_distance_median)

test_data_set["log1p_distance"] = np.log1p(test_data_set["distance"].clip(lower=0.0))

JFK = (40.6413, -73.7781)
LGA = (40.7769, -73.8740)
MANHATTAN = (40.7580, -73.9855)

test_data_set["pickup_dist_to_jfk"] = distance_on_the_sphere(
    test_data_set["pickup_latitude"], test_data_set["pickup_longitude"], JFK[0], JFK[1]
)
test_data_set["dropoff_dist_to_jfk"] = distance_on_the_sphere(
    test_data_set["dropoff_latitude"],
    test_data_set["dropoff_longitude"],
    JFK[0],
    JFK[1],
)
test_data_set["pickup_dist_to_lga"] = distance_on_the_sphere(
    test_data_set["pickup_latitude"], test_data_set["pickup_longitude"], LGA[0], LGA[1]
)
test_data_set["dropoff_dist_to_lga"] = distance_on_the_sphere(
    test_data_set["dropoff_latitude"],
    test_data_set["dropoff_longitude"],
    LGA[0],
    LGA[1],
)
test_data_set["pickup_dist_to_manhattan"] = distance_on_the_sphere(
    test_data_set["pickup_latitude"],
    test_data_set["pickup_longitude"],
    MANHATTAN[0],
    MANHATTAN[1],
)
test_data_set["dropoff_dist_to_manhattan"] = distance_on_the_sphere(
    test_data_set["dropoff_latitude"],
    test_data_set["dropoff_longitude"],
    MANHATTAN[0],
    MANHATTAN[1],
)

for c in [
    "pickup_dist_to_jfk",
    "dropoff_dist_to_jfk",
    "pickup_dist_to_lga",
    "dropoff_dist_to_lga",
    "pickup_dist_to_manhattan",
    "dropoff_dist_to_manhattan",
]:
    test_data_set[c] = test_data_set[c].fillna(float(train_data_set[c].median()))

test_data_set["passenger_count"] = (
    pd.to_numeric(test_data_set["passenger_count"], errors="coerce")
    .fillna(1)
    .clip(1, 6)
    .astype(int)
)
test_data_set["const"] = 1.0



## === cell 11
filename = "./output/baseline_linear"

XTEST = test_data_set[features].values
y_pred_final = linear_model.predict(XTEST)

y_pred_final = np.clip(y_pred_final, 0.0, 500.0)

submission = pd.DataFrame(
    {"key": test_data_set.key, "fare_amount": y_pred_final},
    columns=["key", "fare_amount"],
)
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with", len(submission), "rows")
