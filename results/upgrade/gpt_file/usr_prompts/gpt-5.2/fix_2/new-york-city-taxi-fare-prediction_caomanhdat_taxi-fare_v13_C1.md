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

4.86501

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 4.86501) has done: 'I fix the datetime feature engineering to work with modern pandas by replacing deprecated `.dt.week`/`.dt.weekofyear` with ISO week extraction, and I ensure `pickup_datetime` is fully dropped so XGBoost only sees numeric columns. I also correct the `get_samples_output()` call so it doesn’t incorrectly pass a DataFrame with `key` already removed (which can cause column misalignment). Finally, I update the XGBoost parameter `silent` (deprecated) to `verbosity` while keeping the same model type and training approach, and I write a valid `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import xgboost as xgb

print(os.listdir("../input"))



## === cell 1
train = pd.read_csv("../input/train.csv", nrows=5_000_000)
test = pd.read_csv("../input/test.csv")



## === cell 2
train.head()



## === cell 3
test.head()



## === cell 4
train.describe()



## === cell 5
test.describe()



## === cell 6
train.isnull().sum()



## === cell 7
test.isnull().sum()




## === cell 8
def handle_date(df):
    df = df.copy()
    df["pickup_datetime"] = (
        df["pickup_datetime"].astype(str).str.replace(" UTC", "", regex=False)
    )
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")

    dt = df["pickup_datetime"].dt
    df["hour_of_day"] = dt.hour
    df["week"] = dt.isocalendar().week.astype("int16")
    df["month"] = dt.month.astype("int16")
    df["year"] = dt.year.astype("int16")
    df["day_of_year"] = dt.dayofyear.astype("int16")
    df["week_of_year"] = df["week"]  # keep same semantics as prior code intent
    df["weekday"] = dt.weekday.astype("int16")
    df["quarter"] = dt.quarter.astype("int16")
    df["day_of_month"] = dt.day.astype("int16")

    df = df.drop("pickup_datetime", axis=1)
    return df


train = handle_date(train)
test = handle_date(test)




## === cell 9
def handle_distance(df):
    df = df.copy()
    df["longitude_distance"] = (df["pickup_longitude"] - df["dropoff_longitude"]).abs()
    df["latitude_distance"] = (df["pickup_latitude"] - df["dropoff_latitude"]).abs()
    df["distance_travelled"] = (
        df["longitude_distance"] ** 2 + df["latitude_distance"] ** 2
    ) ** 0.5
    df = df.drop(["longitude_distance", "latitude_distance"], axis=1)
    return df


train = handle_distance(train)
test = handle_distance(test)



## === cell 10
train.describe()




## === cell 11
def clean_up_train(train_df):
    train_df = train_df.dropna()
    train_df = train_df[train_df["fare_amount"] > 0]
    train_df = train_df[train_df["passenger_count"] > 0]
    train_df = train_df[train_df["passenger_count"] < 7]
    return train_df


train = clean_up_train(train)
train.describe()




## === cell 12
def get_samples_output(train_df):
    feature_cols = test.drop("key", axis=1).columns
    return (train_df[feature_cols], train_df["fare_amount"])




## === cell 13
from sklearn.model_selection import train_test_split

samples_train, samples_label = get_samples_output(train.drop("key", axis=1))
X_train, X_test, y_train, y_test = train_test_split(
    samples_train, samples_label, test_size=0.3, random_state=0
)



## === cell 14
samples_train = samples_train.apply(pd.to_numeric, errors="coerce")
test_features = test.drop("key", axis=1).apply(pd.to_numeric, errors="coerce")

model = xgb.XGBRegressor(max_depth=2, n_estimators=50, verbosity=0)
model.fit(samples_train, samples_label)

submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": model.predict(test_features)},
    columns=["key", "fare_amount"],
)
submission.to_csv("submission.csv", index=False)
submission.head(20)
