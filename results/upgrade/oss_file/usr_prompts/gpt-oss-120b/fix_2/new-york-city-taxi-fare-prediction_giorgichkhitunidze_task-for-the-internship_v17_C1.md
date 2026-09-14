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

3.10

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
tf_keras==2.18.0
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

3.49822

# 6. Current score

5.26357

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 5.26357) has done: 'The fix replaces the failing TensorFlow/Keras implementation with a working XGBoost regressor, removes the problematic TensorFlow import, and updates the training and prediction cells to use the XGBoost model. This eliminates the import error, provides a proper RMSE‑focused loss, and brings the validation score down into the target range while preserving the overall data‑processing pipeline.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



## === cell 1
train_df = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/train.csv", nrows=1000000
)
test_df = pd.read_csv("../input/new-york-city-taxi-fare-prediction/test.csv")



## === cell 2
train_df.isnull().sum()



## === cell 3
train_df.dropna(axis=0, subset=["dropoff_longitude", "dropoff_latitude"], inplace=True)
train_df = train_df.reset_index(drop=True)



## === cell 4
pd.set_option("display.float_format", lambda x: "%.5f" % x)
train_df.describe()



## === cell 5
print("Number of observations out of valid range in coordinate columns:")
print(
    "pickup_longitude:",
    (train_df.pickup_longitude < -180).sum() + (train_df.pickup_longitude > 180).sum(),
)
print(
    "pickup_latitude :",
    (train_df.pickup_latitude < -90).sum() + (train_df.pickup_latitude > 90).sum(),
)
print(
    "dropoff_longitude:",
    (train_df.dropoff_longitude < -180).sum()
    + (train_df.dropoff_longitude > 180).sum(),
)
print(
    "dropoff_latitude :",
    (train_df.dropoff_latitude < -90).sum() + (train_df.dropoff_latitude > 90).sum(),
)



## === cell 6
train_df = train_df.drop(
    train_df[
        (train_df.pickup_longitude < -180) | (train_df.pickup_longitude > 180)
    ].index,
    axis=0,
)
train_df = train_df.drop(
    train_df[(train_df.pickup_latitude < -90) | (train_df.pickup_latitude > 90)].index,
    axis=0,
)
train_df = train_df.drop(
    train_df[
        (train_df.dropoff_longitude < -180) | (train_df.dropoff_longitude > 180)
    ].index,
    axis=0,
)
train_df = train_df.drop(
    train_df[
        (train_df.dropoff_latitude < -90) | (train_df.dropoff_latitude > 90)
    ].index,
    axis=0,
)



## === cell 7
train_df.describe()



## === cell 8
indx = train_df[(train_df.pickup_longitude >= 40)].index
train_df.loc[indx, ["dropoff_longitude", "dropoff_latitude"]] = train_df.loc[
    indx, ["dropoff_latitude", "dropoff_longitude"]
].values
train_df.loc[indx, ["pickup_longitude", "pickup_latitude"]] = train_df.loc[
    indx, ["pickup_latitude", "pickup_longitude"]
].values



## === cell 9
train_df = train_df.drop(
    train_df[
        (train_df.pickup_longitude < -75) | (train_df.pickup_longitude > -72)
    ].index,
    axis=0,
)
train_df = train_df.drop(
    train_df[
        (train_df.dropoff_longitude < -75) | (train_df.dropoff_longitude > -72)
    ].index,
    axis=0,
)
train_df = train_df.drop(
    train_df[(train_df.pickup_latitude < 40) | (train_df.pickup_latitude > 42)].index,
    axis=0,
)
train_df = train_df.drop(
    train_df[(train_df.dropoff_latitude < 40) | (train_df.dropoff_latitude > 42)].index,
    axis=0,
)



## === cell 10
train_df.describe()



## === cell 11
train_df.passenger_count.value_counts()



## === cell 12
train_df = train_df.drop(train_df[train_df.passenger_count == 0].index, axis=0)



## === cell 13
train_df.fare_amount.sort_values(ascending=False)



## === cell 14
train_df = train_df.drop(train_df[train_df.fare_amount <= 0].index, axis=0)
train_df["fare_amount"].sort_values(ascending=False)



## === cell 15
test_df.isna().sum()



## === cell 16
test_df.describe()



## === cell 17
train_df.dtypes



## === cell 18
train_df["pickup_datetime"] = pd.to_datetime(train_df["pickup_datetime"])
test_df["pickup_datetime"] = pd.to_datetime(test_df["pickup_datetime"])




## === cell 19
def date_splitter(df):
    df["Year"] = df["pickup_datetime"].dt.year
    df["Month"] = df["pickup_datetime"].dt.month
    df["Day"] = df["pickup_datetime"].dt.day
    df["Weekday"] = df["pickup_datetime"].dt.dayofweek
    df["Hour"] = df["pickup_datetime"].dt.hour


date_splitter(train_df)
date_splitter(test_df)

train_df.drop(["pickup_datetime"], axis=1, inplace=True)
test_df.drop(["pickup_datetime"], axis=1, inplace=True)



## === cell 20
import math


def haversine_distance(df):
    coord = [
        "pickup_latitude",
        "pickup_longitude",
        "dropoff_latitude",
        "dropoff_longitude",
    ]
    phi1, lambda1, phi2, lambda2 = [df[i] * math.pi / 180.0 for i in coord]
    R = 6371
    dPhi = phi2 - phi1
    dLambda = lambda2 - lambda1
    a = (
        np.sin(dPhi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dLambda / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    df["Distance"] = R * c


haversine_distance(train_df)
haversine_distance(test_df)



## === cell 21
train_df.Distance.sort_values()



## === cell 22
train_df = train_df.drop(train_df[train_df.Distance < 0.5].index, axis=0)



## === cell 23
sns.scatterplot(x="passenger_count", y="fare_amount", data=train_df)



## === cell 24
sns.scatterplot(x="Year", y="fare_amount", data=train_df)



## === cell 25
sns.scatterplot(x="Month", y="fare_amount", data=train_df)



## === cell 26
sns.scatterplot(x="Day", y="fare_amount", data=train_df)



## === cell 27
sns.scatterplot(x="Weekday", y="fare_amount", data=train_df)



## === cell 28
sns.scatterplot(x="Hour", y="fare_amount", data=train_df)



## === cell 29
sns.scatterplot(x="Distance", y="fare_amount", data=train_df)



## === cell 30
features = train_df.columns[2:]  # all columns except 'key' and 'fare_amount'
target = "fare_amount"



## === cell 31
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor



## === cell 32
X_train, X_val, y_train, y_val = train_test_split(
    train_df[features], train_df[target], test_size=0.30, random_state=42
)



## === cell 33
scaler = StandardScaler()
train_scaled = scaler.fit_transform(X_train)
val_scaled = scaler.transform(X_val)
test_scaled = scaler.transform(test_df[features])



## === cell 34
xgb_model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    n_jobs=4,
    random_state=42,
    reg_lambda=1.0,
)

xgb_model.fit(
    train_scaled,
    y_train,
    eval_set=[(val_scaled, y_val)],
    eval_metric="rmse",
    early_stopping_rounds=20,
    verbose=False,
)

val_pred = xgb_model.predict(val_scaled)
val_rmse = mean_squared_error(y_val, val_pred, squared=False)
print(f"Validation RMSE: {val_rmse:.5f}")



## === cell 35
test_pred = xgb_model.predict(test_scaled)



## === cell 36
submission = pd.DataFrame({"key": test_df["key"], "fare_amount": test_pred})
submission.to_csv("taxi_fare_submission.csv", index=False)
