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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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

4.15448

# 6. Current score

6.44604

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 135.78344) has done: 'The script is cleaned up by removing the stray text block that caused a syntax error, making the data‑path handling robust (works whether the files sit in `./` or under `/kaggle/input/`), and adding a small safety step that clips negative predictions to zero. No core modeling logic is changed; the same features, models, and ensemble weighting are kept, ensuring the solution runs end‑to‑end and outputs a valid CSV submission.'
- What this solution (achieved 139.34034) has done: 'I add a more informative geographic feature (haversine distance in kilometers) and include the month feature, then expand the feature list used by all three models. These changes keep the original models and workflow unchanged while providing stronger signals for fare amount, which should move the RMSE from the very high current value toward the target of ~4.15.'
- What this solution (achieved 5.79688) has done: 'I tighten the feature set to the most predictive columns (hour, month, passenger_count, and haversine distance) and give the XGBoost model a larger share in the final ensemble, because XGBoost usually outperforms the simple linear and random‑forest models on this data. These small adjustments keep the overall pipeline unchanged while expected to lower the RMSE toward the target.'
- What this solution (achieved 58.72299) has done: 'I add the more predictive temporal and Euclidean distance features to the model input and give the XGBoost model a slightly larger weight in the final ensemble. These small, targeted changes should lower the RMSE toward the target without altering the core pipeline.'
- What this solution (achieved 6.50407) has done: 'I add the raw latitude/longitude columns as features (they are highly predictive for fare zones), increase the XGBoost boosting rounds to let the model learn more, and simplify the ensemble by dropping the linear model which contributes little. These modest changes keep the overall pipeline intact while expected to pull the RMSE down significantly toward the target.'
- What this solution (achieved 6.37815) has done: 'I add two simple distance‑based features (Manhattan distance and passenger‑count × haversine distance) and include them in the model input, then give the XGBoost model a few more boosting rounds (400) and a slightly larger RandomForest (300 trees). These minimal extensions keep the original pipeline intact while providing extra predictive signal that should lower the RMSE toward the target.'
- What this solution (achieved 6.44604) has done: 'I add a simple linear calibration step that learns a scaling from the ensemble’s raw predictions to the true fare amounts on the training data, then apply the same adjustment to the test predictions. This modest post‑processing usually reduces bias and lowers RMSE without altering the core models. I also increase the XGBoost boosting rounds slightly (to 600) for a bit more learning capacity, which together should move the score closer to the target while keeping the overall pipeline unchanged.'

# 9. Code solution

## === cell 0
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

if Path("/kaggle/input/train.csv").exists():
    DATA_ROOT = Path("/kaggle/input")
elif Path("train.csv").exists():
    DATA_ROOT = Path(".")
else:
    raise FileNotFoundError("train.csv not found in expected locations")

TRAIN_PATH = DATA_ROOT / "train.csv"
TEST_PATH = DATA_ROOT / "test.csv"
SAMPLE_SUB_PATH = DATA_ROOT / "sample_submission.csv"




## === cell 1
train = pd.read_csv(TRAIN_PATH, nrows=300_000)
test = pd.read_csv(TEST_PATH)




## === cell 2
train["pickup_datetime"] = pd.to_datetime(train["pickup_datetime"])
train["hour"] = train["pickup_datetime"].dt.hour
train["day"] = train["pickup_datetime"].dt.day
train["week"] = train["pickup_datetime"].dt.isocalendar().week
train["month"] = train["pickup_datetime"].dt.month
train["day_of_year"] = train["pickup_datetime"].dt.dayofyear
train["week_of_year"] = train["pickup_datetime"].dt.isocalendar().week




## === cell 3
test["pickup_datetime"] = pd.to_datetime(test["pickup_datetime"])
test["hour"] = test["pickup_datetime"].dt.hour
test["day"] = test["pickup_datetime"].dt.day
test["week"] = test["pickup_datetime"].dt.isocalendar().week
test["month"] = test["pickup_datetime"].dt.month
test["day_of_year"] = test["pickup_datetime"].dt.dayofyear
test["week_of_year"] = test["pickup_datetime"].dt.isocalendar().week




## === cell 4
train = train.dropna(how="any", axis="rows")
train = train.loc[(train["fare_amount"] > 0) & (train["fare_amount"] < 200)]
train = train.loc[(train["pickup_longitude"] > -75) & (train["pickup_longitude"] < 75)]
train = train.loc[(train["pickup_latitude"] > 40) & (train["pickup_latitude"] < 45)]
train = train.loc[
    (train["dropoff_longitude"] > -75) & (train["dropoff_longitude"] < 75)
]
train = train.loc[(train["dropoff_latitude"] > 40) & (train["dropoff_latitude"] < 45)]
train = train.loc[train["passenger_count"] <= 8]




## === cell 5
for df in (train, test):
    df["abs_diff_longitude"] = (df["pickup_longitude"] - df["dropoff_longitude"]).abs()
    df["abs_diff_latitude"] = (df["pickup_latitude"] - df["dropoff_latitude"]).abs()
    df["distance"] = np.sqrt(
        (df["pickup_longitude"] - df["dropoff_longitude"]) ** 2
        + (df["pickup_latitude"] - df["dropoff_latitude"]) ** 2
    )
    df["manhattan_dist"] = df["abs_diff_longitude"] + df["abs_diff_latitude"]
    df["passenger_dist_product"] = (
        df["passenger_count"] * df["haversine_dist"] if "haversine_dist" in df else 0
    )


def haversine_np(lon1, lat1, lon2, lat2):
    R = 6371.0  # Earth radius in km
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


for df in (train, test):
    df["haversine_dist"] = haversine_np(
        df["pickup_longitude"],
        df["pickup_latitude"],
        df["dropoff_longitude"],
        df["dropoff_latitude"],
    )
    df["pickup_longitude_raw"] = df["pickup_longitude"]
    df["pickup_latitude_raw"] = df["pickup_latitude"]
    df["dropoff_longitude_raw"] = df["dropoff_longitude"]
    df["dropoff_latitude_raw"] = df["dropoff_latitude"]
    df["passenger_dist_product"] = df["passenger_count"] * df["haversine_dist"]




## === cell 6
feature_names = [
    "hour",
    "day",
    "month",
    "day_of_year",
    "week_of_year",
    "passenger_count",
    "haversine_dist",
    "distance",
    "manhattan_dist",
    "passenger_dist_product",
    "pickup_longitude_raw",
    "pickup_latitude_raw",
    "dropoff_longitude_raw",
    "dropoff_latitude_raw",
]
label_name = "fare_amount"

X_train = train[feature_names]
y_train = train[label_name]
X_test = test[feature_names]




## === cell 7
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
lin_pred = lin_reg.predict(X_test)

rf = RandomForestRegressor(random_state=42, n_estimators=300, n_jobs=5)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)
xgb_params = {
    "max_depth": 7,
    "eta": 0.05,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "verbosity": 0,
}
xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=600, verbose_eval=False)
xgb_pred = xgb_model.predict(dtest)




## === cell 8
test_ensemble = (rf_pred + 9 * xgb_pred) / 10

rf_train_pred = rf.predict(X_train)
xgb_train_pred = xgb_model.predict(dtrain)
train_ensemble = (rf_train_pred + 9 * xgb_train_pred) / 10

calibrator = LinearRegression()
calibrator.fit(train_ensemble.reshape(-1, 1), y_train)

predictions = calibrator.predict(test_ensemble.reshape(-1, 1))
predictions = np.clip(predictions, 0, None)




## === cell 9
submission = pd.read_csv(SAMPLE_SUB_PATH)
submission["fare_amount"] = predictions
submission.head()




## === cell 10
output_path = Path("./simplenewyorktaxi.csv")
submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path.resolve()}")
