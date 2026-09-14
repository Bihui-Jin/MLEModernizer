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

3.33042

# 6. Current score

5.47019

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.39566) has done: 'I add a simple but effective feature (the pickup hour) and train the model on the log‑transformed fare amount, converting predictions back with `expm1`. These tweaks keep the overall pipeline unchanged while targeting a lower RMSE, moving the score from 6.42 closer to the target 3.33.'
- What this solution (achieved 5.84016) has done: 'I improve the model by (1) strengthening the XGBoost training – using a smaller learning rate, a larger number of boosting rounds and modest subsampling – which usually lowers RMSE, and (2) removing the unnecessary rounding of predictions so the output stays as precise as possible. These tweaks keep the overall pipeline unchanged while moving the score closer to the target.'
- What this solution (achieved 5.14362) has done: 'The update adds two helpful engineered features—`log_dist` (log‑scaled haversine distance) and `manhattan` (sum of absolute latitude/longitude differences)—which often improve fare prediction without changing the core model. It also gives the model a bit more training data by using a larger train split (80 % vs 75 %). These minimal tweaks keep the original pipeline intact while aiming to lower RMSE toward the target score.'
- What this solution (achieved 6.17838) has done: 'I added a few targeted tweaks that keep the overall pipeline unchanged while aiming to lower the RMSE toward the target value.  
1. Load a larger training sample (2 million rows) to give the model more data.  
2. One‑hot encode the `hour` feature (like `year` and `day`) to capture time‑of‑day effects.  
3. Train the XGBoost model with a smaller learning rate (`eta=0.05`) and more boosting rounds (2000) so it can learn finer patterns; early stopping still prevent over‑fitting.  

These minimal changes preserve the core logic and should move the score closer to the target.'
- What this solution (achieved 5.47019) has done: 'Implemented minimal enhancements to push the RMSE closer to the target:  
- Load a larger training sample (3 M rows) for richer learning.  
- Extract the `month` from the datetime and one‑hot encode it alongside existing time features.  
- Increase XGBoost capacity by raising `max_depth` to 8 and allowing up to 3000 boosting rounds (early stopping still controls over‑fit).  

These tweaks keep the original pipeline intact while providing additional signal and model strength to reduce error.'

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
df = pd.read_csv("../input/train.csv", nrows=3000000)




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
l = df[
    (df.pickup_latitude > 42.0)
    | (df.pickup_latitude < 40.0)
    | (df.dropoff_latitude > 42.0)
    | (df.dropoff_latitude < 40.0)
    | (df.pickup_longitude > -73.0)
    | (df.pickup_longitude < -75.0)
    | (df.dropoff_longitude > -73.0)
    | (df.dropoff_longitude < -75.0)
].index




## === cell 8
df = df.drop(l, axis=0)




## === cell 9
z = df[
    (df.fare_amount > 300.0)
    | (df.fare_amount < 0.0)
    | (df.passenger_count > 7.0)
    | (df.passenger_count < 0.0)
].index




## === cell 10
df = df.drop(z, axis=0)




## === cell 11
len(df)




## === cell 12
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




## === cell 13
df["dist"] = [
    distlatlong(
        df.pickup_longitude[i],
        df.pickup_latitude[i],
        df.dropoff_longitude[i],
        df.dropoff_latitude[i],
    )
    for i in df.index
]




## === cell 14
test["dist"] = [
    distlatlong(
        test.pickup_longitude[i],
        test.pickup_latitude[i],
        test.dropoff_longitude[i],
        test.dropoff_latitude[i],
    )
    for i in test.index
]




## === cell 15
df["log_dist"] = np.log1p(df["dist"])
df["manhattan"] = abs(df["pickup_latitude"] - df["dropoff_latitude"]) + abs(
    df["pickup_longitude"] - df["dropoff_longitude"]
)

test["log_dist"] = np.log1p(test["dist"])
test["manhattan"] = abs(test["pickup_latitude"] - test["dropoff_latitude"]) + abs(
    test["pickup_longitude"] - test["dropoff_longitude"]
)




## === cell 16
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])




## === cell 17
test["pickup_datetime"] = pd.to_datetime(test["pickup_datetime"])




## === cell 18
df["latenights"] = (df["pickup_datetime"].dt.hour < 5).astype(int)
test["latenights"] = (test["pickup_datetime"].dt.hour < 5).astype(int)

df["hour"] = df["pickup_datetime"].dt.hour
test["hour"] = test["pickup_datetime"].dt.hour




## === cell 19
df["weekday"] = (df["pickup_datetime"].dt.weekday > 4).astype(int)
test["weekday"] = (test["pickup_datetime"].dt.weekday > 4).astype(int)




## === cell 20
df["year"] = df["pickup_datetime"].dt.year
test["year"] = test["pickup_datetime"].dt.year




## === cell 21
df["day"] = df["pickup_datetime"].dt.day
test["day"] = test["pickup_datetime"].dt.day

df["month"] = df["pickup_datetime"].dt.month
test["month"] = test["pickup_datetime"].dt.month




## === cell 22
df.head()




## === cell 23
test.head()




## === cell 24
feat = df.drop(["key", "pickup_datetime"], axis=1)
test = test.drop(["key", "pickup_datetime"], axis=1)




## === cell 25
feat = pd.concat([feat, pd.get_dummies(feat.year, prefix="year")], axis=1)
test = pd.concat([test, pd.get_dummies(test.year, prefix="year")], axis=1)




## === cell 26
feat = feat.drop("year", axis=1)
test = test.drop("year", axis=1)




## === cell 27
feat = pd.concat([feat, pd.get_dummies(feat.day, prefix="day")], axis=1)
test = pd.concat([test, pd.get_dummies(test.day, prefix="day")], axis=1)




## === cell 28
feat = feat.drop("day", axis=1)
test = test.drop("day", axis=1)

feat = pd.concat([feat, pd.get_dummies(feat.month, prefix="month")], axis=1)
test = pd.concat([test, pd.get_dummies(test.month, prefix="month")], axis=1)

feat = feat.drop("month", axis=1)
test = test.drop("month", axis=1)

feat = pd.concat([feat, pd.get_dummies(feat.hour, prefix="hour")], axis=1)
test = pd.concat([test, pd.get_dummies(test.hour, prefix="hour")], axis=1)

feat = feat.drop("hour", axis=1)
test = test.drop("hour", axis=1)




## === cell 29
feat.head()




## === cell 30
test.head()




## === cell 31
label = np.log1p(feat["fare_amount"])




## === cell 32
feat = feat.drop("fare_amount", axis=1)




## === cell 33
xtr, xts, ytr, yts = train_test_split(feat, label, test_size=0.20, random_state=42)




## === cell 34
xgbtrain = xgboost.DMatrix(xtr, ytr)
xgbtest = xgboost.DMatrix(xts, yts)
xgbfinaltest = xgboost.DMatrix(test)




## === cell 35
params = {
    "eval_metric": "rmse",
    "objective": "reg:squarederror",
    "eta": 0.05,  # smaller learning rate for finer fitting
    "max_depth": 8,  # increased depth for richer modeling
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}




## === cell 36
xgbmodel = xgboost.train(
    params,
    dtrain=xgbtrain,
    num_boost_round=3000,  # allow more rounds; early stopping will choose optimum
    early_stopping_rounds=50,
    evals=[(xgbtest, "test")],
)




## === cell 37
pred = np.expm1(xgbmodel.predict(xgbfinaltest))




## === cell 38
finalset = pd.DataFrame({"key": testkey, "fare_amount": pred})




## === cell 39
finalset = finalset[["key", "fare_amount"]]




## === cell 40
finalset.head()




## === cell 41
finalset.to_csv("finaloutput.csv", index=False)
