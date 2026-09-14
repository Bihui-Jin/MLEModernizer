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

5.57556

# 6. Current score

None

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 761.7631) has done: 'I remove the obsolete `normalize` argument from the LinearRegression constructor, add an RMSE calculation (the competition metric) to monitor performance, and keep the subsequent prediction and submission steps unchanged so they run correctly and produce a valid Submission.csv file.'
- What this solution (achieved 761.58172) has done: 'The update adds feature scaling and a ridge‑regression model (still a linear model) to improve numerical stability and reduce the validation RMSE, bringing the score closer to the target while keeping the overall workflow unchanged.'
- What this solution (achieved 740.47979) has done: 'I add a modest outlier‑filter on `fare_amount` before the train/validation split, align the test feature columns with the training columns (filling missing dummy columns with 0), and clip any negative predictions to 0. These small data‑cleaning tweaks keep the same Ridge model while moving the validation RMSE much closer to the target 5.58.'
- What this solution (achieved 727.62593) has done: 'I add two simple distance‑based features (Manhattan distance and distance × passenger count) before dropping the raw coordinate columns, and I use a slightly weaker ridge regularization (α = 0.1). These lightweight changes keep the same linear‑model pipeline while giving the model more predictive signal, which should lower the validation RMSE and move it nearer to the target.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error




## === cell 1
train_path = "../input/new-york-city-taxi-fare-prediction/train.csv"
test_path = "../input/new-york-city-taxi-fare-prediction/test.csv"

train_df = pd.read_csv(train_path, nrows=10_000_000)
test_df = pd.read_csv(test_path)




## === cell 2
def add_travel_vector_features(df):
    df["abs_diff_longitude"] = (df["dropoff_longitude"] - df["pickup_longitude"]).abs()
    df["abs_diff_latitude"] = (df["dropoff_latitude"] - df["pickup_latitude"]).abs()


add_travel_vector_features(train_df)
add_travel_vector_features(test_df)




## === cell 3
train_df = train_df[
    (train_df["abs_diff_longitude"] < 5.0) & (train_df["abs_diff_latitude"] < 5.0)
]




## === cell 4
train_df["Weekday"] = pd.to_datetime(train_df["pickup_datetime"]).dt.weekday
test_df["Weekday"] = pd.to_datetime(test_df["pickup_datetime"]).dt.weekday

train_df["Hour"] = pd.to_datetime(train_df["pickup_datetime"]).dt.hour
test_df["Hour"] = pd.to_datetime(test_df["pickup_datetime"]).dt.hour

train_df["Month"] = pd.to_datetime(train_df["pickup_datetime"]).dt.month
test_df["Month"] = pd.to_datetime(test_df["pickup_datetime"]).dt.month




## === cell 5
train_weekday_ohe = pd.get_dummies(train_df["Weekday"], prefix="weekday")
test_weekday_ohe = pd.get_dummies(test_df["Weekday"], prefix="weekday")

train_df = pd.concat([train_df, train_weekday_ohe], axis=1)
test_df = pd.concat([test_df, test_weekday_ohe], axis=1)




## === cell 6
train_df.drop("Weekday", axis=1, inplace=True)
test_df.drop("Weekday", axis=1, inplace=True)




## === cell 7
R = 6373.0  # Earth radius in km


def haversine(df):
    lat1 = np.radians(df["pickup_latitude"])
    lon1 = np.radians(df["pickup_longitude"])
    lat2 = np.radians(df["dropoff_latitude"])
    lon2 = np.radians(df["dropoff_longitude"])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance_km = R * c
    return distance_km * 0.621371  # convert to miles


train_df["Distance"] = haversine(train_df)
test_df["Distance"] = haversine(test_df)

airport_lat = np.radians(40.6413111)
airport_lon = np.radians(-73.7781391)


def airport_dist(lat_series, lon_series):
    lat = np.radians(lat_series)
    lon = np.radians(lon_series)

    dlat = airport_lat - lat
    dlon = airport_lon - lon
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat) * np.cos(airport_lat) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c * 0.621371


train_df["Pickup_Distance_airport"] = airport_dist(
    train_df["pickup_latitude"], train_df["pickup_longitude"]
)
train_df["Dropoff_Distance_airport"] = airport_dist(
    train_df["dropoff_latitude"], train_df["dropoff_longitude"]
)
test_df["Pickup_Distance_airport"] = airport_dist(
    test_df["pickup_latitude"], test_df["pickup_longitude"]
)
test_df["Dropoff_Distance_airport"] = airport_dist(
    test_df["dropoff_latitude"], test_df["dropoff_longitude"]
)




## === cell 8
dist_cols = ["Distance", "Pickup_Distance_airport", "Dropoff_Distance_airport"]
train_df[dist_cols] = train_df[dist_cols].round(2)
test_df[dist_cols] = test_df[dist_cols].round(2)




## === cell 9
train_df["Manhattan"] = train_df["abs_diff_longitude"] + train_df["abs_diff_latitude"]
train_df["Dist_pass"] = train_df["Distance"] * train_df["passenger_count"]
test_df["Manhattan"] = test_df["abs_diff_longitude"] + test_df["abs_diff_latitude"]
test_df["Dist_pass"] = test_df["Distance"] * test_df["passenger_count"]

coord_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]
train_df.drop(columns=coord_cols, inplace=True)
test_df.drop(columns=coord_cols, inplace=True)




## === cell 10
lower_q = train_df["fare_amount"].quantile(0.001)
upper_q = train_df["fare_amount"].quantile(0.999)
train_df = train_df[
    (train_df["fare_amount"] >= lower_q) & (train_df["fare_amount"] <= upper_q)
]

X = train_df.drop(["key", "fare_amount", "pickup_datetime"], axis=1)
y = train_df["fare_amount"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.01, random_state=42)




## === cell 11
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

y_train_log = np.log1p(y_train)

ridge = Ridge(alpha=0.1, random_state=42)
ridge.fit(X_train_scaled, y_train_log)




## === cell 12
y_val_pred_log = ridge.predict(X_val_scaled)
y_val_pred = np.expm1(y_val_pred_log)

rmse = mean_squared_error(y_val, y_val_pred, squared=False)
print(f"Validation RMSE: {rmse:.4f}")




## === cell 13
test_features = test_df.drop(["key", "pickup_datetime"], axis=1)
test_features = test_features.reindex(columns=X_train.columns, fill_value=0)

test_features_scaled = scaler.transform(test_features)

test_pred_log = ridge.predict(test_features_scaled)
test_pred = np.expm1(test_pred_log)

test_pred = np.clip(test_pred, a_min=0, a_max=None)
test_pred = np.round(test_pred, 2)




## === cell 14
submission = pd.DataFrame({"key": test_df["key"], "fare_amount": test_pred})
submission.to_csv("Submission.csv", index=False)




## === cell 15
print("Submission file written. First rows:")
print(submission.head())
