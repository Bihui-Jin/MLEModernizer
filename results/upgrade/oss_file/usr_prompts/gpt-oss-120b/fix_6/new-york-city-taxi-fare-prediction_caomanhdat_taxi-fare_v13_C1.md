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

3.74883

# 6. Current score

4.38422

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.86889) has done: 'I fixed the date‑handling code (`dt.week` and `dt.weekofyear` are removed in recent pandas) and updated the XGBoost call so it accepts the processed data without dtype errors. The model now trains on the engineered features and writes a proper `submission.csv` with the required columns.'
- What this solution (achieved 4.38844) has done: 'I add a more accurate haversine distance feature and modestly increase the XGBoost model capacity (depth, estimators, subsample) while using early stopping on the validation split. These changes keep the overall pipeline and feature set intact but should reduce the RMSE from 4.87 toward the target 3.75.'
- What this solution (achieved 4.26175) has done: 'I slightly boost the XGBoost model capacity (deeper trees, more estimators, a smaller learning rate) so it can capture more of the patterns in the engineered features without altering the overall pipeline. These modest changes are expected to lower the RMSE and move the score from 4.388 → closer to the target 3.749 while keeping the same data handling and submission logic.'
- What this solution (achieved 4.21207) has done: 'I add a simple Manhattan distance feature (sum of absolute lat/lon differences) because it captures another notion of trip length that often helps taxi fare models, and I modestly increase the XGBoost capacity (deeper trees, more rounds, slightly lower learning rate) while keeping early stopping. These small, targeted tweaks should lower the RMSE toward the target without altering the overall pipeline.'
- What this solution (achieved 4.38422) has done: 'I added cyclical time features (hour sin/cos and weekday sin/cos) to give the model a better sense of periodic patterns, switched to training on the log‑transformed fare (log1p) so the regressor focuses on relative errors, and modestly increased XGBoost capacity (deeper trees, more estimators, smaller learning rate) while keeping early stopping. After prediction the values are inverse‑transformed (expm1) to obtain the final fare amounts. These small, targeted changes aim to lower the RMSE toward the target without altering the overall pipeline.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O
import os
import xgboost as xgb

print(os.listdir("../input"))




## === cell 1
train = pd.read_csv("../input/train.csv", nrows=5_000_000)
test = pd.read_csv("../input/test.csv")




## === cell 2
def handle_date(df):
    df["pickup_datetime"] = df["pickup_datetime"].str.replace(" UTC", "")
    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"], format="%Y-%m-%d %H:%M:%S"
    )
    df["hour_of_day"] = df["pickup_datetime"].dt.hour
    df["week"] = df["pickup_datetime"].dt.isocalendar().week
    df["month"] = df["pickup_datetime"].dt.month
    df["year"] = df["pickup_datetime"].dt.year
    df["day_of_year"] = df["pickup_datetime"].dt.dayofyear
    df["weekday"] = df["pickup_datetime"].dt.weekday
    df["quarter"] = df["pickup_datetime"].dt.quarter
    df["day_of_month"] = df["pickup_datetime"].dt.day

    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)

    df = df.drop("pickup_datetime", axis=1)
    return df




## === cell 3
train = handle_date(train)
test = handle_date(test)




## === cell 4
def handle_distance(df):
    df["longitude_distance"] = abs(df["pickup_longitude"] - df["dropoff_longitude"])
    df["latitude_distance"] = abs(df["pickup_latitude"] - df["dropoff_latitude"])
    df["distance_travelled"] = (
        df["longitude_distance"] ** 2 + df["latitude_distance"] ** 2
    ) ** 0.5
    df = df.drop(["longitude_distance", "latitude_distance"], axis=1)
    return df




## === cell 5
train = handle_distance(train)
test = handle_distance(test)




## === cell 6
def add_haversine_distance(df):
    r = 6371.0
    lat1 = np.radians(df["pickup_latitude"])
    lat2 = np.radians(df["dropoff_latitude"])
    lon1 = np.radians(df["pickup_longitude"])
    lon2 = np.radians(df["dropoff_longitude"])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    df["distance_haversine"] = r * c
    return df




## === cell 7
train = add_haversine_distance(train)
test = add_haversine_distance(test)




## === cell 8
def add_manhattan_distance(df):
    df["distance_manhattan"] = abs(
        df["pickup_longitude"] - df["dropoff_longitude"]
    ) + abs(df["pickup_latitude"] - df["dropoff_latitude"])
    return df




## === cell 9
train = add_manhattan_distance(train)
test = add_manhattan_distance(test)




## === cell 10
def clean_up_train(df):
    df = df.dropna()
    df = df[df["fare_amount"] > 0]
    df = df[df["passenger_count"] > 0]
    df = df[df["passenger_count"] < 7]
    return df




## === cell 11
train = clean_up_train(train)




## === cell 12
def get_samples_output(train_df):
    """
    Align train columns with test columns (minus the key) and return
    features matrix and target series.
    """
    feature_cols = test.drop("key", axis=1).columns
    return train_df[feature_cols], train_df["fare_amount"]


from sklearn.model_selection import train_test_split

samples_train, samples_label = get_samples_output(train.drop("key", axis=1))

samples_label = np.log1p(samples_label)

X_train, X_valid, y_train, y_valid = train_test_split(
    samples_train, samples_label, test_size=0.3, random_state=0
)




## === cell 13
model = xgb.XGBRegressor(
    max_depth=10,  # a bit deeper
    n_estimators=1200,  # more rounds, early stopping will cap
    learning_rate=0.02,  # finer steps
    subsample=0.9,
    colsample_bytree=0.9,
    objective="reg:squarederror",
    n_jobs=4,
    random_state=0,
    verbosity=0,
    reg_alpha=0.0,
    reg_lambda=1.0,
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    early_stopping_rounds=30,
    verbose=False,
)




## === cell 14
preds = np.expm1(model.predict(test.drop("key", axis=1)))

submission = pd.DataFrame(
    {
        "key": test["key"],
        "fare_amount": preds,
    },
    columns=["key", "fare_amount"],
)

submission.to_csv("submission.csv", index=False)
submission.head(20)
