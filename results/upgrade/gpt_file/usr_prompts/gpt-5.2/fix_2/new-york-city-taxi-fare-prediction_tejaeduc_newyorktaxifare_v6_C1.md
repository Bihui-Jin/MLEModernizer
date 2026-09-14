# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import xgboost
import os

print(os.listdir("../input"))



## === cell 1
df = pd.read_csv(
    "../input/train.csv",
    nrows=105000,
    dtype={
        "fare_amount": "float32",
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "int16",
    },
)



## === cell 2
test = pd.read_csv(
    "../input/test.csv",
    dtype={
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "int16",
    },
)



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
def haversine_km(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1.astype("float64"))
    lat1 = np.radians(lat1.astype("float64"))
    lon2 = np.radians(lon2.astype("float64"))
    lat2 = np.radians(lat2.astype("float64"))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return 6371.0 * c  # Earth's mean radius in km




## === cell 14
df["dist"] = haversine_km(
    df["pickup_longitude"],
    df["pickup_latitude"],
    df["dropoff_longitude"],
    df["dropoff_latitude"],
).astype("float32")
test["dist"] = haversine_km(
    test["pickup_longitude"],
    test["pickup_latitude"],
    test["dropoff_longitude"],
    test["dropoff_latitude"],
).astype("float32")

df["abs_lon_diff"] = (
    (df["pickup_longitude"] - df["dropoff_longitude"]).abs().astype("float32")
)
df["abs_lat_diff"] = (
    (df["pickup_latitude"] - df["dropoff_latitude"]).abs().astype("float32")
)
test["abs_lon_diff"] = (
    (test["pickup_longitude"] - test["dropoff_longitude"]).abs().astype("float32")
)
test["abs_lat_diff"] = (
    (test["pickup_latitude"] - test["dropoff_latitude"]).abs().astype("float32")
)

df["manhattan_dist"] = (df["abs_lon_diff"] + df["abs_lat_diff"]).astype("float32")
test["manhattan_dist"] = (test["abs_lon_diff"] + test["abs_lat_diff"]).astype("float32")



## === cell 15
test.head()



## === cell 16
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
test["pickup_datetime"] = pd.to_datetime(test["pickup_datetime"], errors="coerce")



## === cell 17
df = df.dropna(subset=["pickup_datetime"])



## === cell 18
df.info()



## === cell 19
df["latenights"] = (df["pickup_datetime"].dt.hour < 5).astype("int8")
test["latenights"] = (test["pickup_datetime"].dt.hour < 5).astype("int8")



## === cell 20
df.pickup_datetime.iloc[0].weekday()



## === cell 21
df["weekday"] = (df["pickup_datetime"].dt.weekday > 4).astype("int8")
test["weekday"] = (test["pickup_datetime"].dt.weekday > 4).astype("int8")



## === cell 22
df.head()



## === cell 23
df["year"] = df["pickup_datetime"].dt.year.astype("int16")
test["year"] = test["pickup_datetime"].dt.year.astype("int16")



## === cell 24
df["day"] = df["pickup_datetime"].dt.day.astype("int8")
test["day"] = test["pickup_datetime"].dt.day.astype("int8")



## === cell 25
df.head()



## === cell 26
test.head()



## === cell 27
feat = df.drop(["key", "pickup_datetime"], axis=1)
test_feat = test.drop(["key", "pickup_datetime"], axis=1)



## === cell 28
test_feat.year.unique()



## === cell 29
feat = pd.get_dummies(feat, columns=["year"], prefix="year")
test_feat = pd.get_dummies(test_feat, columns=["year"], prefix="year")
feat, test_feat = feat.align(test_feat, join="left", axis=1, fill_value=0)



## === cell 30
feat.head()



## === cell 31
test_feat.head()



## === cell 32
label = feat["fare_amount"].astype("float32")



## === cell 33
feat = feat.drop("fare_amount", axis=1)



## === cell 34
xtr, xts, ytr, yts = train_test_split(feat, label, test_size=0.2, random_state=42)



## === cell 35
xgbtrain = xgboost.DMatrix(xtr, label=ytr)
xgbvalid = xgboost.DMatrix(xts, label=yts)
xgbfinaltest = xgboost.DMatrix(test_feat)



## === cell 36
params = {
    "eval_metric": "rmse",
    "objective": "reg:squarederror",
    "seed": 42,
    "verbosity": 0,
}



## === cell 37
xgbmodel = xgboost.train(
    params,
    dtrain=xgbtrain,
    num_boost_round=350,
    early_stopping_rounds=30,
    evals=[(xgbvalid, "valid")],
)



## === cell 38
pred = xgbmodel.predict(xgbfinaltest)



## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3403936747.py in <cell line: 0>()
----> 1 pred = xgbmodel.predict(xgbfinaltest)
      2 

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in predict(self, data, output_margin, pred_leaf, pred_contribs, approx_contribs, pred_interactions, validate_features, training, iteration_range, strict_shape)
   2270         if validate_features:
   2271             fn = data.feature_names
-> 2272             self._validate_features(fn)
   2273         args = {
   2274             "type": 0,

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in _validate_features(self, feature_names)
   2968                 )
   2969 
-> 2970             raise ValueError(msg.format(self.feature_names, feature_names))
   2971 
   2972     def get_split_value_histogram(

ValueError: feature_names mismatch: ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count', 'dist', 'abs_lon_diff', 'abs_lat_diff', 'manhattan_dist', 'latenights', 'weekday', 'day', 'year_2009', 'year_2010', 'year_2011', 'year_2012', 'year_2013', 'year_2014', 'year_2015'] ['fare_amount', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count', 'dist', 'abs_lon_diff', 'abs_lat_diff', 'manhattan_dist', 'latenights', 'weekday', 'day', 'year_2009', 'year_2010', 'year_2011', 'year_2012', 'year_2013', 'year_2014', 'year_2015']
training data did not have the following fields: fare_amount

## === cell 39
pred = np.clip(pred, 0.0, None)



## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2770739726.py in <cell line: 0>()
      1 # Do NOT round predictions: rounding increases RMSE and moves score away from the target (lower is better).
----> 2 pred = np.clip(pred, 0.0, None)
      3 

NameError: name 'pred' is not defined

## === cell 40
finalset = pd.DataFrame({"key": testkey, "fare_amount": pred})



## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/40697884.py in <cell line: 0>()
----> 1 finalset = pd.DataFrame({"key": testkey, "fare_amount": pred})
      2 

NameError: name 'pred' is not defined

## === cell 41
finalset = finalset[["key", "fare_amount"]]



## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1566441625.py in <cell line: 0>()
----> 1 finalset = finalset[["key", "fare_amount"]]
      2 

NameError: name 'finalset' is not defined

## === cell 42
finalset.head()



## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2648070237.py in <cell line: 0>()
----> 1 finalset.head()
      2 

NameError: name 'finalset' is not defined

## === cell 43
finalset.to_csv("finaloutput.csv", index=False)
print("Wrote finaloutput.csv with shape:", finalset.shape)

## --- ERROR in cell 43, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/750667728.py in <cell line: 0>()
      1 # Required: write a valid submission .csv with the correct columns.
----> 2 finalset.to_csv("finaloutput.csv", index=False)
      3 print("Wrote finaloutput.csv with shape:", finalset.shape)

NameError: name 'finalset' is not defined
