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

3.21447

# 6. Current score

8.18774

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 7.37431) has done: 'I fix the Haversine distance feature to work on NumPy arrays (your current `math.radians/sin/cos` only accept scalars), which unblocks feature creation for both train and test. Then I fix the XGBoost feature mismatch by ensuring the test DMatrix is built from the exact same feature columns used for training (the error shows `fare_amount` accidentally ended up in the test matrix). Finally, I make sure a valid submission CSV with the required `key,fare_amount` header is written end-to-end.'
- What this solution (achieved 8.18774) has done: 'Your current RMSE (7.37431) is far worse than the target (3.21447), so we should legitimately improve the model with minimal, metric-aligned changes. The biggest win without changing the overall approach is to train XGBoost on `log1p(fare_amount)` (standard for this competition to handle heavy-tailed fares) and then invert with `expm1` at prediction time—this keeps the same model family/training loop/features but usually drops RMSE substantially. I also remove early stopping (it can stop too early and underfit here) while keeping the same boosting approach/params structure, and I clip negative predictions to 0 to avoid RMSE penalties from impossible fares. The script still writes a valid `key,fare_amount` submission CSV end-to-end.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn.model_selection import train_test_split
import xgboost
from sklearn.preprocessing import StandardScaler

import os

print(os.listdir("../input"))



## === cell 1
df = pd.read_csv("../input/train.csv", nrows=1005000)



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
    lon1 = np.asarray(lon1, dtype="float64")
    lat1 = np.asarray(lat1, dtype="float64")
    lon2 = np.asarray(lon2, dtype="float64")
    lat2 = np.asarray(lat2, dtype="float64")

    lat1 = np.deg2rad(lat1)
    lat2 = np.deg2rad(lat2)
    lon1 = np.deg2rad(lon1)
    lon2 = np.deg2rad(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (np.sin(dlat / 2.0) ** 2) + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    distance = 6373.0 * c
    return distance




## === cell 14
df["dist"] = distlatlong(
    df["pickup_longitude"].values,
    df["pickup_latitude"].values,
    df["dropoff_longitude"].values,
    df["dropoff_latitude"].values,
)



## === cell 15
test["dist"] = distlatlong(
    test["pickup_longitude"].values,
    test["pickup_latitude"].values,
    test["dropoff_longitude"].values,
    test["dropoff_latitude"].values,
)



## === cell 16
test.head()



## === cell 17
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])



## === cell 18
test["pickup_datetime"] = pd.to_datetime(test["pickup_datetime"])



## === cell 19
df.info()



## === cell 20
df["latenights"] = (df["pickup_datetime"].dt.hour < 5).astype(int)
test["latenights"] = (test["pickup_datetime"].dt.hour < 5).astype(int)



## === cell 21
df.pickup_datetime.iloc[0].weekday()



## === cell 22
df["weekday"] = (df["pickup_datetime"].dt.weekday > 4).astype(int)
test["weekday"] = (test["pickup_datetime"].dt.weekday > 4).astype(int)



## === cell 23
df.head()



## === cell 24
df["year"] = df["pickup_datetime"].dt.year



## === cell 25
test["year"] = test["pickup_datetime"].dt.year



## === cell 26
df["day"] = df["pickup_datetime"].dt.day



## === cell 27
test["day"] = test["pickup_datetime"].dt.day



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
for _df in (feat, test):
    _df["abs_lon_diff"] = (_df["pickup_longitude"] - _df["dropoff_longitude"]).abs()
    _df["abs_lat_diff"] = (_df["pickup_latitude"] - _df["dropoff_latitude"]).abs()



## === cell 33
feat_year = pd.get_dummies(feat["year"], prefix="year")
test_year = pd.get_dummies(test["year"], prefix="year")
feat = pd.concat([feat.drop("year", axis=1), feat_year], axis=1)
test = pd.concat([test.drop("year", axis=1), test_year], axis=1)
feat, test = feat.align(test, join="left", axis=1, fill_value=0)



## === cell 34
feat.head()



## === cell 35
test.head()



## === cell 36
label = feat["fare_amount"].astype("float64")
label_log = np.log1p(label.clip(lower=0.0))



## === cell 37
feat = feat.drop("fare_amount", axis=1)



## === cell 38
xtr, xts, ytr, yts = train_test_split(feat, label_log, test_size=0.2, random_state=42)



## === cell 39
test = test.reindex(columns=feat.columns, fill_value=0)

xgbtrain = xgboost.DMatrix(xtr, label=ytr, feature_names=list(feat.columns))
xgbtest = xgboost.DMatrix(xts, label=yts, feature_names=list(feat.columns))
xgbfinaltest = xgboost.DMatrix(test, feature_names=list(feat.columns))



## === cell 40
params = {"eval_metric": "rmse", "objective": "reg:squarederror", "seed": 42}



## === cell 41
xgbmodel = xgboost.train(
    params,
    dtrain=xgbtrain,
    num_boost_round=350,
    evals=[(xgbtest, "test")],
)



## === cell 42
pred_log = xgbmodel.predict(xgbfinaltest)



## === cell 43
pred = np.expm1(pred_log)
pred = np.clip(pred, 0.0, None)



## === cell 44
finalset = pd.DataFrame({"key": testkey, "fare_amount": pred})



## === cell 45
finalset = finalset[["key", "fare_amount"]]



## === cell 46
finalset.head()



## === cell 47
finalset.to_csv("finaloutput.csv", index=False)
print("Wrote submission:", os.path.abspath("finaloutput.csv"), "rows:", len(finalset))
