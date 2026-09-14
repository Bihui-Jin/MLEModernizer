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

6.63306

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 6.63306) has done: 'I keep your XGBoost approach and existing features, but fix two issues that are strongly hurting RMSE: (1) your train/test feature columns are not guaranteed to align after `get_dummies` (different year/day values in train vs test), and (2) you are rounding predictions to cents, which adds extra error under RMSE. I also remove early stopping (it violates your “no early stopping” requirement) while keeping the same number of boosting rounds so the training approach stays the same. Finally, I update the model objective to the non-deprecated equivalent and set a fixed random seed for stable splits and training.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from math import sin, cos, sqrt, atan2, radians
import xgboost
from sklearn.preprocessing import StandardScaler
import os

print(os.listdir("../input"))



## === cell 1
df = pd.read_csv("../input/train.csv", nrows=1000000)



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
        df.at[i, "pickup_longitude"],
        df.at[i, "pickup_latitude"],
        df.at[i, "dropoff_longitude"],
        df.at[i, "dropoff_latitude"],
    )
    for i in df.index
]



## === cell 14
test["dist"] = [
    distlatlong(
        test.at[i, "pickup_longitude"],
        test.at[i, "pickup_latitude"],
        test.at[i, "dropoff_longitude"],
        test.at[i, "dropoff_latitude"],
    )
    for i in test.index
]



## === cell 15
test.head()



## === cell 16
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
test["pickup_datetime"] = pd.to_datetime(test["pickup_datetime"])



## === cell 17
df.info()



## === cell 18
df["latenights"] = [1 if df.at[i, "pickup_datetime"].hour < 5 else 0 for i in df.index]
test["latenights"] = [
    1 if test.at[i, "pickup_datetime"].hour < 5 else 0 for i in test.index
]



## === cell 19
df.pickup_datetime.iloc[0].weekday()



## === cell 20
df["weekday"] = [
    1 if df.at[i, "pickup_datetime"].weekday() > 4 else 0 for i in df.index
]
test["weekday"] = [
    1 if test.at[i, "pickup_datetime"].weekday() > 4 else 0 for i in test.index
]



## === cell 21
df.head()



## === cell 22
df["year"] = [df.at[i, "pickup_datetime"].year for i in df.index]
test["year"] = [test.at[i, "pickup_datetime"].year for i in test.index]



## === cell 23
df["day"] = [df.at[i, "pickup_datetime"].day for i in df.index]
test["day"] = [test.at[i, "pickup_datetime"].day for i in test.index]



## === cell 24
df.head()



## === cell 25
test.head()



## === cell 26
feat = df.drop(["key", "pickup_datetime"], axis=1)
test_feat = test.drop(["key", "pickup_datetime"], axis=1)



## === cell 27
test_feat.year.unique()



## === cell 28
combined = pd.concat(
    [feat.drop(["fare_amount"], axis=1), test_feat], axis=0, ignore_index=True
)

combined = pd.concat(
    [combined, pd.get_dummies(combined["year"], prefix="year")], axis=1
)
combined = combined.drop("year", axis=1)

combined = pd.concat([combined, pd.get_dummies(combined["day"], prefix="day")], axis=1)
combined = combined.drop("day", axis=1)

X = combined.iloc[: len(feat), :].copy()
X_test = combined.iloc[len(feat) :, :].copy()



## === cell 29
label = feat["fare_amount"].copy()



## === cell 30
xtr, xts, ytr, yts = train_test_split(X, label, test_size=0.25, random_state=42)



## === cell 31
xgbtrain = xgboost.DMatrix(xtr, label=ytr)
xgbtest = xgboost.DMatrix(xts, label=yts)
xgbfinaltest = xgboost.DMatrix(X_test)



## === cell 32
params = {"eval_metric": "rmse", "objective": "reg:squarederror", "seed": 42}



## === cell 33
xgbmodel = xgboost.train(
    params,
    dtrain=xgbtrain,
    num_boost_round=150,
    evals=[(xgbtest, "test")],
    verbose_eval=False,
)



## === cell 34
pred = xgbmodel.predict(xgbfinaltest)



## === cell 35
finalset = pd.DataFrame({"key": testkey, "fare_amount": pred})



## === cell 36
finalset = finalset[["key", "fare_amount"]]



## === cell 37
finalset.head()



## === cell 38
finalset.to_csv("finaloutput.csv", index=False)
print("Wrote submission:", "finaloutput.csv", "rows:", len(finalset))
