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

3.43988

# 6. Current score

4.96289

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.21188) has done: 'I increase the training sample size, add the pickup hour as a useful feature, keep the year as a numeric column (removing the one‑hot encoding that added sparse noise), train the model on the log‑transformed fare to handle skewness, and convert the predictions back with `expm1`. These minimal, targeted changes are expected to lower the RMSE from ~8.84 toward the target 3.44 while preserving the original workflow.'
- What this solution (achieved 4.91009) has done: 'I increase the training sample size (read 1 000 000 rows instead of 500 000) to give the model more data and tweak the XGBoost parameters by adding a smaller learning rate and modest regularisation. These two lightweight changes are expected to improve generalisation and move the RMSE closer to the target without altering the core workflow.'
- What this solution (achieved 5.64678) has done: 'I slightly increase the training sample size, add cyclical hour features (sin / cos) that capture daily patterns, and modestly strengthen the XGBoost model (max_depth and more rounds) while removing the unnecessary rounding of predictions. These adjustments keep the original workflow intact but should reduce RMSE, moving the score closer to the target.'
- What this solution (achieved 4.96977) has done: 'I increase the training sample to 3 000 000 rows, make the train‑validation split deterministic, and slightly regularize the XGBoost model (reduce max_depth and add a small `gamma`). These minimal adjustments keep the original workflow while giving the model more data and a bit less capacity, which should lower the RMSE toward the target.'
- What this solution (achieved 4.98767) has done: 'I add a “month” feature derived from the pickup datetime (both train and test) so the model can capture seasonal patterns that the current “day” feature misses. This change is limited to the feature‑engineering cells and retains the exact training‑validation split and XGBoost settings, providing a modest expected gain that moves the RMSE closer to the target without altering the core workflow.'
- What this solution (achieved 4.96289) has done: 'I add two modest features – a log‑scaled distance and distance per passenger – which often help linearize the relationship with fare and give the model a bit more signal without changing any core logic. I also slightly regularise XGBoost by lowering the learning rate, which lets early‑stopping find a better iteration while keeping the original training procedure intact. These minimal edits should lower the RMSE toward the target while preserving the overall workflow.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn.model_selection import train_test_split
from math import sin, cos, sqrt, atan2, radians
import xgboost
from sklearn.preprocessing import StandardScaler

import os

print(os.listdir("../input"))




## === cell 1
df = pd.read_csv(
    "../input/train.csv", nrows=3000000
)  # raised from 1 500 000 to 3 000 000 rows




## === cell 2
test = pd.read_csv("../input/test.csv")




## === cell 3
testkey = test.key




## === cell 4
df = df.dropna(how="any", axis="rows")




## === cell 5
len(df)




## === cell 6
df.head()




## === cell 7
df.describe()




## === cell 8
l = df[
    (df.pickup_latitude > 42.5)
    | (df.pickup_latitude < 40.0)
    | (df.dropoff_latitude > 42.5)
    | (df.dropoff_latitude < 40.0)
    | (df.pickup_longitude > -73.0)
    | (df.pickup_longitude < -75.0)
    | (df.dropoff_longitude > -73.0)
    | (df.dropoff_longitude < -75.0)
].index




## === cell 9
df = df.drop(l, axis=0)




## === cell 10
z = df[
    (df.fare_amount > 350.0)
    | (df.fare_amount < 0.0)
    | (df.passenger_count > 7.0)
    | (df.passenger_count < 0.0)
].index




## === cell 11
df = df.drop(z, axis=0)




## === cell 12
len(df)




## === cell 13
def distlatlong(lon1, lat1, lon2, lat2):
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    lon1 = radians(lon1)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (sin(dlat / 2)) ** 2 + cos(lat1) * cos(lat2) * (sin(dlon / 2)) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = 6373.0 * c
    return distance




## === cell 14
df["dist"] = [
    distlatlong(
        df.pickup_longitude[i],
        df.pickup_latitude[i],
        df.dropoff_longitude[i],
        df.dropoff_latitude[i],
    )
    for i in df.index
]
df["log_dist"] = np.log1p(df["dist"])
df["dist_per_passenger"] = df["dist"] / (df["passenger_count"] + 1)




## === cell 15
test["dist"] = [
    distlatlong(
        test.pickup_longitude[i],
        test.pickup_latitude[i],
        test.dropoff_longitude[i],
        test.dropoff_latitude[i],
    )
    for i in test.index
]
test["log_dist"] = np.log1p(test["dist"])
test["dist_per_passenger"] = test["dist"] / (test["passenger_count"] + 1)




## === cell 16
test.head()




## === cell 17
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])




## === cell 18
test["pickup_datetime"] = pd.to_datetime(test["pickup_datetime"])




## === cell 19
df.info()




## === cell 20
df["latenights"] = [1 if df.pickup_datetime[i].hour < 5 else 0 for i in df.index]
test["latenights"] = [1 if test.pickup_datetime[i].hour < 5 else 0 for i in test.index]

df["hour"] = df["pickup_datetime"].dt.hour
test["hour"] = test["pickup_datetime"].dt.hour

df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
test["hour_sin"] = np.sin(2 * np.pi * test["hour"] / 24)
test["hour_cos"] = np.cos(2 * np.pi * test["hour"] / 24)




## === cell 21
df.pickup_datetime[0].weekday()




## === cell 22
df["weekday"] = [1 if df.pickup_datetime[i].weekday() > 4 else 0 for i in df.index]
test["weekday"] = [
    1 if test.pickup_datetime[i].weekday() > 4 else 0 for i in test.index
]




## === cell 23
df.head()




## === cell 24
df["year"] = df["pickup_datetime"].dt.year
df["month"] = df["pickup_datetime"].dt.month




## === cell 25
test["year"] = test["pickup_datetime"].dt.year
test["month"] = test["pickup_datetime"].dt.month




## === cell 26
df["day"] = [df.pickup_datetime[i].day for i in df.index]




## === cell 27
test["day"] = [test.pickup_datetime[i].day for i in test.index]




## === cell 28
df.head()




## === cell 29
test.head()




## === cell 30
feat = df.drop(["key", "pickup_datetime"], axis=1)
test = test.drop(["key", "pickup_datetime"], axis=1)




## === cell 31
test.year.unique()




## === cell 32
feat.head()




## === cell 33
test.head()




## === cell 34
label = np.log1p(feat["fare_amount"])




## === cell 35
feat = feat.drop("fare_amount", axis=1)




## === cell 36
xtr, xts, ytr, yts = train_test_split(feat, label, test_size=0.2, random_state=42)




## === cell 37
xgbtrain = xgboost.DMatrix(xtr, ytr)
xgbtest = xgboost.DMatrix(xts, yts)
xgbfinaltest = xgboost.DMatrix(test)




## === cell 38
params = {
    "eval_metric": "rmse",
    "objective": "reg:squarederror",
    "learning_rate": 0.03,  # slightly lower LR for smoother convergence
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "max_depth": 6,  # keep capacity modest
    "lambda": 1.0,
    "alpha": 0.0,
    "gamma": 0.1,
}




## === cell 39
xgbmodel = xgboost.train(
    params,
    dtrain=xgbtrain,
    num_boost_round=2000,
    early_stopping_rounds=50,
    evals=[(xgbtest, "test")],
)




## === cell 40
pred_log = xgbmodel.predict(xgbfinaltest)




## === cell 41
pred = np.expm1(pred_log)  # removed rounding to keep predictions more precise




## === cell 42
finalset = pd.DataFrame({"key": testkey, "fare_amount": pred})




## === cell 43
finalset = finalset[["key", "fare_amount"]]




## === cell 44
finalset.head()




## === cell 45
finalset.to_csv("finaloutput.csv", index=False)
