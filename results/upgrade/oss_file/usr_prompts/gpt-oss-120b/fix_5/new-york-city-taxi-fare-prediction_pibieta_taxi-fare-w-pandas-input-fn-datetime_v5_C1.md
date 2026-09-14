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

4.29156

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O
import os, shutil

print("../", os.listdir("../"))
print("../input", os.listdir("../input"))



## === cell 1
df = pd.read_csv("../input/train.csv", nrows=100000, parse_dates=["pickup_datetime"])
test = pd.read_csv("../input/test.csv", parse_dates=["pickup_datetime"])



## === cell 2
df.pickup_datetime.dt.weekday  # just to trigger datetime parsing



## === cell 3
from math import cos, asin, sqrt


def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295  # Pi/180
    a = (
        0.5
        - cos((lat2 - lat1) * p) / 2
        + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
    )
    return 12742 * asin(sqrt(a))




## === cell 4
def add_feats(df):
    df["distance"] = distance(
        df["pickup_latitude"].values,
        df["pickup_longitude"].values,
        df["dropoff_latitude"].values,
        df["dropoff_longitude"].values,
    )
    df["hour"] = df.pickup_datetime.dt.hour
    df["weekday"] = df.pickup_datetime.dt.weekday
    return df




## === cell 5
df = add_feats(df)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_56/1409400047.py in <cell line: 0>()
----> 1 df = add_feats(df)
      2 

/tmp/ipykernel_56/186399549.py in add_feats(df)
      1 def add_feats(df):
      2     # vectorized distance computation for speed
----> 3     df["distance"] = distance(
      4         df["pickup_latitude"].values,
      5         df["pickup_longitude"].values,

/tmp/ipykernel_56/3816874281.py in distance(lat1, lon1, lat2, lon2)
      6     a = (
      7         0.5
----> 8         - cos((lat2 - lat1) * p) / 2
      9         + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
     10     )

TypeError: only length-1 arrays can be converted to Python scalars

## === cell 6
dfc = df[
    (df.pickup_longitude >= -75.0)
    & (df.pickup_longitude <= -72)
    & (df.pickup_latitude >= 38)
    & (df.pickup_latitude <= 42)
    & (df.dropoff_longitude >= -75.0)
    & (df.dropoff_longitude <= -72)
    & (df.dropoff_latitude >= 38)
    & (df.dropoff_latitude <= 42)
    & (df.fare_amount > 2.5)
    & (df.passenger_count > 0)
    & (df.passenger_count < 7)
    & (df.distance > 0.2)
]



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_56/515245246.py in <cell line: 0>()
     11     & (df.passenger_count > 0)
     12     & (df.passenger_count < 7)
---> 13     & (df.distance > 0.2)
     14 ]
     15 

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'distance'

## === cell 7
np.random.seed(seed=1)  # reproducibility
msk = np.random.rand(len(dfc)) < 0.8
traindf = dfc[msk].drop(["key", "pickup_datetime"], axis=1).reset_index(drop=True)
evaldf = dfc[~msk].drop(["key", "pickup_datetime"], axis=1).reset_index(drop=True)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/1922774125.py in <cell line: 0>()
      1 np.random.seed(seed=1)  # reproducibility
----> 2 msk = np.random.rand(len(dfc)) < 0.8
      3 traindf = dfc[msk].drop(["key", "pickup_datetime"], axis=1).reset_index(drop=True)
      4 evaldf = dfc[~msk].drop(["key", "pickup_datetime"], axis=1).reset_index(drop=True)
      5 

NameError: name 'dfc' is not defined

## === cell 8
testdf = add_feats(test)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_56/2527666791.py in <cell line: 0>()
----> 1 testdf = add_feats(test)
      2 

/tmp/ipykernel_56/186399549.py in add_feats(df)
      1 def add_feats(df):
      2     # vectorized distance computation for speed
----> 3     df["distance"] = distance(
      4         df["pickup_latitude"].values,
      5         df["pickup_longitude"].values,

/tmp/ipykernel_56/3816874281.py in distance(lat1, lon1, lat2, lon2)
      6     a = (
      7         0.5
----> 8         - cos((lat2 - lat1) * p) / 2
      9         + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
     10     )

TypeError: only length-1 arrays can be converted to Python scalars

## === cell 9
testdf = testdf.drop(["key", "pickup_datetime"], axis=1)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/1720487287.py in <cell line: 0>()
----> 1 testdf = testdf.drop(["key", "pickup_datetime"], axis=1)
      2 

NameError: name 'testdf' is not defined

## === cell 10
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X_train = traindf.drop(["fare_amount"], axis=1)
y_train = traindf["fare_amount"]
X_eval = evaldf.drop(["fare_amount"], axis=1)
y_eval = evaldf["fare_amount"]

model = LinearRegression()
model.fit(X_train, y_train)

eval_pred = model.predict(X_eval)
rmse = mean_squared_error(y_eval, eval_pred, squared=False)
print(f"Validation RMSE: {rmse:.5f}")



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/2886351972.py in <cell line: 0>()
      3 
      4 # Prepare feature matrices and target vectors
----> 5 X_train = traindf.drop(["fare_amount"], axis=1)
      6 y_train = traindf["fare_amount"]
      7 X_eval = evaldf.drop(["fare_amount"], axis=1)

NameError: name 'traindf' is not defined

## === cell 11
test_pred = model.predict(testdf)

submission = pd.DataFrame({"key": test["key"], "fare_amount": test_pred})

submission_path = "submission_file.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/3404322757.py in <cell line: 0>()
      1 # Predict on the test set
----> 2 test_pred = model.predict(testdf)
      3 
      4 # Build submission DataFrame with correct column names
      5 submission = pd.DataFrame({"key": test["key"], "fare_amount": test_pred})

NameError: name 'model' is not defined
