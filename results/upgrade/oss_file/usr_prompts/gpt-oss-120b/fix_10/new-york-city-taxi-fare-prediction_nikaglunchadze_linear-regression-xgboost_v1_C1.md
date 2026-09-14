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
joblib==1.5.2
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
xgboost==2.0.3

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

3.56321

# 6. Current score

5.81204

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 9.15682) has done: 'I keep the overall pipeline unchanged but replace the XGBoost model with a better‑tuned version. By lowering the learning rate, increasing the number of trees, and adding modest regularisation (subsample, colsample_bytree, deeper trees) we can substantially reduce the RMSE while still using the same features and data‑processing steps. This small change moves the validation score toward the target without altering the core logic.'
- What this solution (achieved 5.2376) has done: 'I keep the overall pipeline but train the XGBoost model on the log‑transformed fare amount (log1p) and then exponentiate the predictions back to the original scale. This reduces the impact of large outliers and tends to lower RMSE. I also filter out extreme ride distances (>200 km) after computing them, which removes noisy samples. The XGBoost hyper‑parameters are modestly increased (more trees, smaller learning rate) to give the model extra capacity while still using early stopping. These changes keep the core logic intact but should move the validation RMSE much closer to the target.'
- What this solution (achieved 5.11768) has done: 'I add a log‑transformed ride distance feature (`log_ride_distance`) to both training and test data and include it in the feature list used by the XGBoost model. This inexpensive feature often improves model performance on distance‑based problems and should lower the validation RMSE, moving the score closer to the target while keeping the original pipeline unchanged.'
- What this solution (achieved 5.13604) has done: 'The main slowdown comes from expensive cross‑validation steps (LinearRegression CV, XGBoost hyper‑parameter search) that are not required for the final model. Those cells are replaced with lightweight no‑ops or short placeholders, while keeping all definitions needed later. The core XGBoost training (the final model) and the feature engineering remain unchanged, ensuring identical predictions and submission format.'
- What this solution (achieved 5.81204) has done: 'I trim the training sample size (while keeping the same preprocessing, feature engineering, and model architecture) and enable a modest `max_bin` setting for the histogram algorithm, which speeds up XGBoost without altering its core logic. All other steps remain unchanged, preserving the exact feature set and training/evaluation workflow.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

dtype_spec = {
    "key": "object",
    "fare_amount": "float32",
    "pickup_datetime": "object",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}

df = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
    nrows=1_000_000,  # smaller sample for faster fit, same preprocessing
    dtype=dtype_spec,
    low_memory=False,
)

test_df = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
    dtype={k: v for k, v in dtype_spec.items() if k != "fare_amount"},
    low_memory=False,
)

df.dtypes


## === cell 1
df.describe()




## === cell 2
def filter_column(df, column, range_min, range_max):
    return df[(df[column] >= range_min) & (df[column] <= range_max)]


ny_latitude_min, ny_latitude_max = 40.4772, 45.0153
ny_longitude_min, ny_longitude_max = -79.7624, -71.7517

df = df.dropna()

df = filter_column(df, "pickup_longitude", ny_longitude_min, ny_longitude_max)
df = filter_column(df, "pickup_latitude", ny_latitude_min, ny_latitude_max)
df = filter_column(df, "dropoff_longitude", ny_longitude_min, ny_longitude_max)
df = filter_column(df, "dropoff_latitude", ny_latitude_min, ny_latitude_max)
df = filter_column(df, "passenger_count", 1, 6)


## === cell 3
df[df["fare_amount"] > 200].describe()


## === cell 4
df = filter_column(df, "fare_amount", 1, 200)




## === cell 5
def refactor_datetime(df):
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["year"] = df["pickup_datetime"].dt.year
    df["month"] = df["pickup_datetime"].dt.month
    df["day"] = df["pickup_datetime"].dt.day
    df["weekday"] = df["pickup_datetime"].dt.weekday
    df["hour"] = df["pickup_datetime"].dt.hour
    df.drop(columns=["pickup_datetime"], inplace=True)


refactor_datetime(df)
refactor_datetime(test_df)
df.head()




## === cell 6
def haversine_series(lat1, lon1, lat2, lon2):
    """
    Vectorised haversine distance for pandas Series / numpy arrays.
    Returns distance in kilometres.
    """
    lat1_r, lon1_r, lat2_r, lon2_r = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2_r - lon1_r
    dlat = lat2_r - lat1_r
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


ny_center = ("ny_center", (40.7128, -74.0060))
jfk_airport = ("jfk_airport", (40.6446, -73.7797))
lga_airport = ("lga_airport", (40.7733, -73.8718))
ewr_airport = ("ewr_airport", (40.6895, -74.1745))

locs = [ny_center, jfk_airport, lga_airport, ewr_airport]




## === cell 7
def insert_haversine_dists(df, locations):
    df["ride_distance"] = haversine_series(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )
    for loc_name, (lat_pt, lon_pt) in locations:
        df["pickup_dist_to_" + loc_name] = haversine_series(
            df["pickup_latitude"], df["pickup_longitude"], lat_pt, lon_pt
        )
        df["dropoff_dist_to_" + loc_name] = haversine_series(
            df["dropoff_latitude"], df["dropoff_longitude"], lat_pt, lon_pt
        )


insert_haversine_dists(df, locs)
insert_haversine_dists(test_df, locs)

df["log_ride_distance"] = np.log1p(df["ride_distance"])
test_df["log_ride_distance"] = np.log1p(test_df["ride_distance"])


## === cell 8
df.describe()


## === cell 9
df = df[df["ride_distance"] > 0]
df = df[df["ride_distance"] <= 200]  # filter out extreme outliers
df.describe()


## === cell 10
from sklearn.model_selection import train_test_split

train_df, validation_df = train_test_split(df, test_size=0.2, random_state=42)


## === cell 11
features = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "year",
    "month",
    "day",
    "hour",
    "weekday",
    "ride_distance",
    "log_ride_distance",
]
features += ["pickup_dist_to_" + x[0] for x in locs]
features += ["dropoff_dist_to_" + x[0] for x in locs]
fare_amount = "fare_amount"

train_features = train_df[features].astype(np.float32).values
train_fare_amount = train_df[fare_amount].values

validation_features = validation_df[features].astype(np.float32).values
validation_fare_amount = validation_df[fare_amount].values

test_features = test_df[features].astype(np.float32).values

print("Train features shape:", train_features.shape)
print("Validation features shape:", validation_features.shape)


## === cell 12
from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()


## === cell 13
from sklearn.model_selection import cross_val_predict, cross_val_score


def estimate_model(model, df):
    X = df[features]
    y = df[fare_amount]
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")
    rmse_scores = np.sqrt(-cv_scores)
    print("RMSE scores for each fold:", rmse_scores)
    print("Mean RMSE:", rmse_scores.mean())
    print("Standard Deviation of RMSE:", rmse_scores.std())




## === cell 14
train_log_fare = np.log1p(train_fare_amount)
validation_log_fare = np.log1p(validation_fare_amount)

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

xgb_model = XGBRegressor(
    objective="reg:squarederror",
    learning_rate=0.02,
    n_estimators=3000,
    max_depth=8,
    min_child_weight=1,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=1.0,
    n_jobs=-1,
    random_state=42,
    verbosity=0,
    tree_method="hist",
    max_bin=256,  # reduces histogram bins for faster training without changing model form
)

xgb_model.fit(
    train_features,
    train_log_fare,
    eval_set=[(validation_features, validation_log_fare)],
    early_stopping_rounds=50,
    verbose=False,
)

validation_log_pred = xgb_model.predict(validation_features)
validation_pred = np.expm1(validation_log_pred)

validation_pred = np.clip(validation_pred, 1, 200)

val_rmse = mean_squared_error(validation_fare_amount, validation_pred, squared=False)
print("Validation RMSE:", val_rmse)


## === cell 15
train_log_pred = xgb_model.predict(train_features)
train_pred = np.expm1(train_log_pred)
train_pred = np.clip(train_pred, 1, 200)
train_rmse = mean_squared_error(train_fare_amount, train_pred, squared=False)
print("Training RMSE:", train_rmse)


## === cell 16
test_log_pred = xgb_model.predict(test_features)
test_predictions = np.expm1(test_log_pred)

test_predictions = np.clip(test_predictions, 1, 200)

submission = pd.DataFrame({"key": test_df["key"], "fare_amount": test_predictions})
submission.to_csv("submission.csv", index=False)
