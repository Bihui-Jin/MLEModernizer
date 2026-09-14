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
lightgbm==4.6.0
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

3.60451

# 6. Current score

5.07466

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 5.07466) has done: 'The fix corrects the logical‑filtering error that caused a TypeError, stops the accidental removal of rows from the test set, and simplifies the final prediction step so the submission length matches the test data. This ensures a valid `submission.csv` is written and moves the RMSE closer to the target.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



## === cell 1
train_data = pd.read_csv("../input/train.csv", nrows=20000)
test_data = pd.read_csv("../input/test.csv")
train_data.head()
test_data.head()



## === cell 2
train_data.describe()



## === cell 3
train_data.shape




## === cell 4
def changeDataType(dataset):
    dataset["passenger_count"] = dataset["passenger_count"].astype("uint8")
    dataset["pickup_longitude"] = dataset["pickup_longitude"].astype("float32")
    dataset["pickup_latitude"] = dataset["pickup_latitude"].astype("float32")
    dataset["dropoff_longitude"] = dataset["dropoff_longitude"].astype("float32")
    dataset["dropoff_latitude"] = dataset["dropoff_latitude"].astype("float32")
    dataset["pickup_datetime"] = pd.to_datetime(
        dataset["pickup_datetime"], errors="coerce"
    )
    dataset.info()


changeDataType(train_data)
print("--" * 40)
changeDataType(test_data)

train_data["fare_amount"] = train_data["fare_amount"].astype("float32")



## === cell 5
train_data["pickup_datetime"].head()



## === cell 6
train_data.isnull().sum()
train_data = train_data.dropna(axis=0)
train_data.isnull().sum()



## === cell 7
pd.set_option("float_format", "{:f}".format)
train_data.describe()



## === cell 8
train_data = train_data.loc[train_data["fare_amount"] > 0]



## === cell 9
train_data = train_data.loc[train_data["fare_amount"] < 400]



## === cell 10
train_data = train_data[train_data["passenger_count"] <= 6]



## === cell 11
train_data = train_data[
    (train_data["pickup_latitude"] >= -90)
    & (train_data["pickup_latitude"] <= 90)
    & (train_data["pickup_longitude"] >= -180)
    & (train_data["pickup_longitude"] <= 180)
    & (train_data["dropoff_latitude"] >= -90)
    & (train_data["dropoff_latitude"] <= 90)
    & (train_data["dropoff_longitude"] >= -180)
    & (train_data["dropoff_longitude"] <= 180)
]



## === cell 12
train_data = train_data[
    train_data["pickup_latitude"].between(
        test_data["pickup_latitude"].min(), test_data["pickup_latitude"].max()
    )
    & train_data["pickup_longitude"].between(
        test_data["pickup_longitude"].min(), test_data["pickup_longitude"].max()
    )
    & train_data["dropoff_latitude"].between(
        test_data["dropoff_latitude"].min(), test_data["dropoff_latitude"].max()
    )
    & train_data["dropoff_longitude"].between(
        test_data["dropoff_longitude"].min(), test_data["dropoff_longitude"].max()
    )
]




## === cell 13
def degree_to_radian(degree):
    return degree * (np.pi / 180)


def calculate_distance(
    pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude
):
    from_lat = degree_to_radian(pickup_latitude)
    from_long = degree_to_radian(pickup_longitude)
    to_lat = degree_to_radian(dropoff_latitude)
    to_long = degree_to_radian(dropoff_longitude)
    radius = 6371.01  # Earth radius in km
    lat_diff = to_lat - from_lat
    long_diff = to_long - from_long
    a = (
        np.sin(lat_diff / 2) ** 2
        + np.cos(from_lat) * np.cos(to_lat) * np.sin(long_diff / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return radius * c




## === cell 14
train_data["distance"] = calculate_distance(
    train_data["pickup_latitude"],
    train_data["pickup_longitude"],
    train_data["dropoff_latitude"],
    train_data["dropoff_longitude"],
)



## === cell 15
test_data["distance"] = calculate_distance(
    test_data["pickup_latitude"],
    test_data["pickup_longitude"],
    test_data["dropoff_latitude"],
    test_data["dropoff_longitude"],
)



## === cell 16
train_data = train_data.loc[train_data["distance"] < 200]



## === cell 17
train_data = train_data.drop(columns="key")
test_data_key = test_data["key"]
test_data = test_data.drop(columns="key")



## === cell 18
for df in (train_data, test_data):
    df["Year"] = df["pickup_datetime"].dt.year
    df["Month"] = df["pickup_datetime"].dt.month
    df["Date"] = df["pickup_datetime"].dt.day
    df["Day_of_Week"] = df["pickup_datetime"].dt.dayofweek
    df["Hour"] = df["pickup_datetime"].dt.hour



## === cell 19
train_data = train_data.drop(columns="pickup_datetime")
test_data = test_data.drop(columns="pickup_datetime")



## === cell 20
X = train_data.drop(columns="fare_amount")
y = train_data["fare_amount"]



## === cell 21
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.3, random_state=0
)



## === cell 22
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
import lightgbm as lgb

gradient_reg = GradientBoostingRegressor()
xgreg = XGBRegressor()
model_lgb = lgb.LGBMRegressor()

gradient_reg.fit(X_train, y_train)
xgreg.fit(X_train, y_train)
model_lgb.fit(X_train, y_train)




## === cell 23
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


val_preds = np.mean(
    np.column_stack(
        [
            gradient_reg.predict(X_valid),
            xgreg.predict(X_valid),
            model_lgb.predict(X_valid),
        ]
    ),
    axis=1,
)
validation_rmse = rmse(y_valid, val_preds)
validation_rmse



## === cell 24
test_preds = np.mean(
    np.column_stack(
        [
            gradient_reg.predict(test_data),
            xgreg.predict(test_data),
            model_lgb.predict(test_data),
        ]
    ),
    axis=1,
)



## === cell 25
submission = pd.DataFrame({"key": test_data_key, "fare_amount": test_preds})
submission.to_csv("submission.csv", index=False)
print("Submission saved, rows:", submission.shape[0])
