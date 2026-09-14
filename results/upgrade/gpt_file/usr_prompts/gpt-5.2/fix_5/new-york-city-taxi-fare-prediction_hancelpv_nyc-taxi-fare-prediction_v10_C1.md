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

3.7

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

4.22488

# 6. Current score

5.23349

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.34176) has done: 'I fix the runtime errors caused by pandas API changes by replacing the deprecated `weekday_name` accessor with the supported `day_name()` and by ensuring `pickup_datetime` does not remain as a Timestamp in the model matrix. I also make the train/test concatenation consistent by reading only the needed test columns (including `key`) and dropping `key` from the feature set so it can’t leak into `X`. These changes are minimal and preserve the original feature engineering + RandomForest training flow, while making the pipeline run end-to-end and write a valid `submission.csv` with `key,fare_amount`. This should also improve score versus the broken run because the intended time features actually be created and used.'
- What this solution (achieved 5.72582) has done: 'Your current gap is 5.34176 − 4.22488 = 1.11688 (lower is better), so we should improve the RMSE but with minimal, low-risk changes that preserve your RandomForest + same engineered features pipeline. The biggest controllable issues here are (1) the model is using default hyperparameters and (2) a few strong, standard NYC Taxi data cleaning filters (NYC bounding box + fare upper cap) are missing, which typically reduces label noise a lot without changing the core approach. I add those minimal cleaning filters and set conservative RandomForest hyperparameters (more trees, fixed seed, sensible depth/min leaf) to move the score toward the target while keeping the same model family and feature set. I also ensure the train/test one-hot alignment stays identical (still via concatenation) and keep the submission schema unchanged.'
- What this solution (achieved 5.12493) has done: 'Your score gap is 5.72582 − 4.22488 = 1.50094 RMSE (lower is better), so we should improve accuracy with minimal, low-risk changes that keep the same RandomForest + feature engineering pipeline. The biggest issue is you only train on `nrows=10000`, which is far too small for this problem; increasing this to a still-manageable sample size usually yields a large RMSE drop without changing core logic. I also apply the same NYC bounding-box/valid-lat-lon filters to the test set (for safer feature distribution) and handle those filtered rows by falling back to the training mean fare so the submission stays complete and valid. Finally, I ensure any remaining missing engineered features (from invalid datetimes) are filled deterministically before `get_dummies`, preventing silent row drops or NaNs affecting the model.'
- What this solution (achieved 5.23349) has done: 'We should move RMSE down (lower is better) from 5.12493 toward 4.22488, so the smallest safe lever is better data quality and slightly stronger RandomForest generalization without changing the model family or feature set. I (1) add one standard, low-risk cleaning rule that removes “zero distance” trips (pickup==dropoff), which are mostly label noise, and (2) add the missing numeric geo feature (`squared_long`/`squared_lat`) that your code already computes but never feeds to the model. I also use `max_features="sqrt"` (a common RF default for better generalization) while keeping the same training loop and core pipeline, and keep the submission schema identical.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import os

for p in [
    "../input",
    "/kaggle/input",
    "/kaggle/data",
    "/kaggle/data/new-york-city-taxi-fare-prediction",
]:
    if os.path.exists(p):
        print(p, "->", os.listdir(p)[:10])



## === cell 1
types = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}

cols_train = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

cols_test = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]




## === cell 2
def resolve_path(fname):
    candidates = [
        f"../input/{fname}",
        f"/kaggle/input/{fname}",
        f"/kaggle/data/{fname}",
        f"/kaggle/data/new-york-city-taxi-fare-prediction/{fname}",
        f"/kaggle/input/new-york-city-taxi-fare-prediction/{fname}",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return f"../input/{fname}"


train_path = resolve_path("train.csv")
test_path = resolve_path("test.csv")
samp_path = resolve_path("sample_submission.csv")

NROWS_TRAIN = 300000  # chosen to stay within typical Kaggle notebook time/memory

train = pd.read_csv(train_path, nrows=NROWS_TRAIN, usecols=cols_train, dtype=types)
test = pd.read_csv(test_path, usecols=cols_test)
samp = pd.read_csv(samp_path)



## === cell 3
train.dropna(how="any", axis="rows", inplace=True)
train = train[train.fare_amount > 0]
train = train[train["passenger_count"] <= 6]
train = train[train["fare_amount"] <= 250]



## === cell 4
latitude_mask_pickup = (train.pickup_latitude > -90) & (train.pickup_latitude < 90)
train = train[latitude_mask_pickup]

latitude_mask_dropoff = (train.dropoff_latitude > -90) & (train.dropoff_latitude < 90)
train = train[latitude_mask_dropoff]



## === cell 5
longitude_mask_pickup = (train.pickup_longitude > -180) & (train.pickup_longitude < 180)
train = train[longitude_mask_pickup]

longitude_mask_dropoff = (train.dropoff_longitude > -180) & (
    train.dropoff_longitude < 180
)
train = train[longitude_mask_dropoff]

nyc_lat_min, nyc_lat_max = 40.5, 41.0
nyc_lon_min, nyc_lon_max = -74.3, -73.6

train = train[
    (train.pickup_latitude.between(nyc_lat_min, nyc_lat_max))
    & (train.dropoff_latitude.between(nyc_lat_min, nyc_lat_max))
    & (train.pickup_longitude.between(nyc_lon_min, nyc_lon_max))
    & (train.dropoff_longitude.between(nyc_lon_min, nyc_lon_max))
]

train = train[
    ~(
        (train["pickup_latitude"] == train["dropoff_latitude"])
        & (train["pickup_longitude"] == train["dropoff_longitude"])
    )
]

test_id = test.key
test_features_only = test.drop(columns=["key"]).copy()

test_valid = (
    test_features_only.pickup_latitude.between(-90, 90)
    & test_features_only.dropoff_latitude.between(-90, 90)
    & test_features_only.pickup_longitude.between(-180, 180)
    & test_features_only.dropoff_longitude.between(-180, 180)
    & test_features_only.pickup_latitude.between(nyc_lat_min, nyc_lat_max)
    & test_features_only.dropoff_latitude.between(nyc_lat_min, nyc_lat_max)
    & test_features_only.pickup_longitude.between(nyc_lon_min, nyc_lon_max)
    & test_features_only.dropoff_longitude.between(nyc_lon_min, nyc_lon_max)
)



## === cell 6
all_data = pd.concat(
    (train.drop(columns=["fare_amount"]), test_features_only.loc[test_valid]),
    axis=0,
    ignore_index=True,
)

y = train.fare_amount.values
n_train = len(train)
n_test_valid = int(test_valid.sum())

fallback_fare = float(np.mean(y))




## === cell 7
def week_num(day):
    """
    given the day of the month, return the week number of the month
    """
    if day <= 7:
        return "first"
    if (day > 7) and (day <= 14):
        return "second"
    if (day > 14) and (day <= 21):
        return "third"
    if (day > 21) and (day <= 28):
        return "fourth"
    return "fifth"




## === cell 8
def add_time_features(data):
    data = data.copy()
    data["pickup_datetime"] = pd.to_datetime(data["pickup_datetime"], errors="coerce")

    data["hour"] = data["pickup_datetime"].dt.hour
    data["day_of_week"] = data["pickup_datetime"].dt.day_name()
    data["day_of_month"] = data["pickup_datetime"].dt.day
    data["week_of_month"] = data["day_of_month"].map(week_num)
    data["month"] = data["pickup_datetime"].dt.month
    data["year"] = data["pickup_datetime"].dt.year

    data["hour"] = data["hour"].astype("Int64").astype(str)
    data["month"] = data["month"].astype("Int64").astype(str)
    data["year"] = data["year"].astype("Int64").astype(str)

    data["day_of_week"] = data["day_of_week"].fillna("Unknown")
    data["week_of_month"] = data["week_of_month"].fillna("Unknown")
    data["hour"] = data["hour"].replace({"<NA>": "Unknown"})
    data["month"] = data["month"].replace({"<NA>": "Unknown"})
    data["year"] = data["year"].replace({"<NA>": "Unknown"})

    data.drop(["pickup_datetime", "day_of_month"], axis=1, inplace=True)
    return data




## === cell 9
def add_geo_features(data):
    data = data.copy()
    data["abs_diff_longitude"] = (data.dropoff_longitude - data.pickup_longitude).abs()
    data["abs_diff_latitude"] = (data.dropoff_latitude - data.pickup_latitude).abs()

    data["manhattan_distance"] = data["abs_diff_longitude"] + data["abs_diff_latitude"]

    data["squared_long"] = np.power(data["abs_diff_longitude"], 2)
    data["squared_lat"] = np.power(data["abs_diff_latitude"], 2)
    return data




## === cell 10
all_data = add_time_features(all_data)
all_data = add_geo_features(all_data)



## === cell 11
features = [
    "passenger_count",
    "hour",
    "day_of_week",
    "week_of_month",
    "month",
    "year",
    "abs_diff_longitude",
    "abs_diff_latitude",
    "manhattan_distance",
    "squared_long",
    "squared_lat",
]

all_data = all_data[features]

num_cols = [
    "abs_diff_longitude",
    "abs_diff_latitude",
    "manhattan_distance",
    "squared_long",
    "squared_lat",
]
all_data[num_cols] = all_data[num_cols].fillna(0.0)

all_data = pd.get_dummies(all_data)



## === cell 12
x = all_data.iloc[:n_train, :]
x_test_valid = all_data.iloc[n_train:, :]



## === cell 13
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    max_depth=18,
    min_samples_leaf=2,
    max_features="sqrt",
)



## === cell 14
model.fit(x, y)

test_pred_valid = model.predict(x_test_valid)
test_pred_valid = np.clip(test_pred_valid, 0, None)

test_pred_full = np.full(shape=(len(test),), fill_value=fallback_fare, dtype=np.float64)
test_pred_full[test_valid.values] = test_pred_valid.astype(np.float64)

sub = pd.DataFrame({"key": test_id, "fare_amount": test_pred_full})
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Valid test rows used for model prediction:", n_test_valid, "out of", len(test))
print("Wrote submission.csv with shape:", sub.shape)
