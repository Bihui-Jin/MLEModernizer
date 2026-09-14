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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
sklearn-pandas==2.2.0

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

5.57567

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 973.18686) has done: 'The changes fix the import error by using `SimpleImputer` instead of the removed `Imputer`, adjust the test feature columns to match the training columns, and ensure all variables are defined before use. Minor clean‑ups (removing the IPython magic and correcting the cell numbering) make the notebook run end‑to‑end and produce a valid `submission.csv` ready for Kaggle.'

# 9. Code solution

## === cell 0
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D

from sklearn.linear_model import Ridge  # use Ridge instead of plain LinearRegression

print(os.listdir("../input"))




## === cell 1
def chunck_generator(filename, header=False, chunk_size=10**6, max_rows=None):
    usecols = [
        "fare_amount",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]
    dtype = {
        "fare_amount": np.float32,
        "pickup_longitude": np.float32,
        "pickup_latitude": np.float32,
        "dropoff_longitude": np.float32,
        "dropoff_latitude": np.float32,
        "passenger_count": np.int8,
    }
    return pd.read_csv(
        filename,
        delimiter=",",
        iterator=True,
        chunksize=chunk_size,
        parse_dates=["pickup_datetime"],
        usecols=usecols,
        dtype=dtype,
        nrows=max_rows,
    )




## === cell 2
alpha_ang = 0.506




## === cell 3
def distance_travel(df):
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs() * 50
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs() * 69
    df["displacement_vector"] = (
        df.abs_diff_latitude**2 + df.abs_diff_longitude**2
    ) ** 0.5
    df["actual_long"] = (
        df.displacement_vector
        * np.sin(np.arctan(df.abs_diff_longitude / df.abs_diff_latitude) - alpha_ang)
    ).abs()
    df["actual_lat"] = (
        df.displacement_vector
        * np.cos(np.arctan(df.abs_diff_longitude / df.abs_diff_latitude) - alpha_ang)
    ).abs()
    df["distance_travel"] = df.actual_long + df.actual_lat
    return df




## === cell 4
def add_time_features(df):
    df["hour"] = df.pickup_datetime.dt.hour
    df["dow"] = df.pickup_datetime.dt.dayofweek
    return df




## === cell 5
def data_clean(df):
    if "fare_amount" in df.columns:
        df["fare_amount"] = pd.to_numeric(df["fare_amount"], errors="coerce")
        df = df[df.fare_amount > 0]
    df = df[df.passenger_count > 0]
    if "distance_travel" in df.columns:
        df = df[df.distance_travel > 0]
    return df




## === cell 6
def remove_outliers(df):
    df = df[df.distance_travel < 30]
    df = df[df.fare_amount < 60]
    return df




## === cell 7
def data_preprocessing(df):
    df = df[df.passenger_count <= 6]
    df = distance_travel(df)  # compute distance once
    df = add_time_features(df)  # add hour & dow
    df = data_clean(df)  # numeric conversion & basic filters
    df = remove_outliers(df)  # final outlier removal
    return df




## === cell 8
df = pd.read_csv("../input/train.csv", nrows=1_000_000, parse_dates=["pickup_datetime"])
df["fare_amount"] = pd.to_numeric(df["fare_amount"], errors="coerce")
df = data_preprocessing(df)
df.distance_travel.hist(bins=50, figsize=(12, 4))
plt.xlabel("distance miles")
plt.title("Histogram ride distances in miles")
df.distance_travel.describe()




## === cell 9
df = pd.read_csv("../input/train.csv", nrows=1_000_000, parse_dates=["pickup_datetime"])
df["fare_amount"] = pd.to_numeric(df["fare_amount"], errors="coerce")
df = data_preprocessing(df)
df.fare_amount.hist(bins=50, figsize=(12, 4))
plt.xlabel("fare_amount")
df.fare_amount.describe()




## === cell 10
filename = r"../input/train.csv"
gen = chunck_generator(filename=filename, max_rows=5_000_000)

ridge_regr = Ridge(alpha=1.0, fit_intercept=True, solver="auto")

for t, df_chunk in enumerate(gen, start=1):
    print("Chunk", t)
    df_chunk = data_preprocessing(df_chunk)
    if df_chunk.empty:
        print("  empty after preprocessing – skipping")
        continue

    l = len(df_chunk)
    split = int(0.9 * l)
    df_train = df_chunk.iloc[:split]
    df_test = df_chunk.iloc[split:]

    train_X = np.column_stack(
        (
            df_train.distance_travel,
            df_train.passenger_count,
            df_train.pickup_longitude,
            df_train.dropoff_longitude,
            df_train.pickup_latitude,
            df_train.dropoff_latitude,
            df_train.hour,
            df_train.dow,
        )
    )
    test_X = np.column_stack(
        (
            df_test.distance_travel,
            df_test.passenger_count,
            df_test.pickup_longitude,
            df_test.dropoff_longitude,
            df_test.pickup_latitude,
            df_test.dropoff_latitude,
            df_test.hour,
            df_test.dow,
        )
    )
    train_y = np.array(df_train.fare_amount, dtype=np.float32)
    test_y = np.array(df_test.fare_amount, dtype=np.float32)

    ridge_regr.fit(train_X, train_y)
    print("  Ridge R^2 on chunk:", ridge_regr.score(test_X, test_y))

regr = ridge_regr  # keep the final model for prediction




## === cell 11
test_df = pd.read_csv("../input/test.csv", parse_dates=["pickup_datetime"])
test_df = data_preprocessing(test_df)  # adds distance & time features, cleans
test_df.head()




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/587374390.py in <cell line: 0>()
      1 test_df = pd.read_csv("../input/test.csv", parse_dates=["pickup_datetime"])
----> 2 test_df = data_preprocessing(test_df)  # adds distance & time features, cleans
      3 test_df.head()
      4 
      5 

/tmp/ipykernel_11/1679608730.py in data_preprocessing(df)
      4     df = add_time_features(df)  # add hour & dow
      5     df = data_clean(df)  # numeric conversion & basic filters
----> 6     df = remove_outliers(df)  # final outlier removal
      7     return df
      8 

/tmp/ipykernel_11/2933893557.py in remove_outliers(df)
      1 def remove_outliers(df):
      2     df = df[df.distance_travel < 30]
----> 3     df = df[df.fare_amount < 60]
      4     return df
      5 

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'fare_amount'

## === cell 12
test_X = np.column_stack(
    (
        test_df.distance_travel,
        test_df.passenger_count,
        test_df.pickup_longitude,
        test_df.dropoff_longitude,
        test_df.pickup_latitude,
        test_df.dropoff_latitude,
        test_df.hour,
        test_df.dow,
    )
)
predicted_fare = regr.predict(test_X)
print(predicted_fare[:5])
print("Mean predicted fare:", np.mean(predicted_fare))




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1654077328.py in <cell line: 0>()
      1 test_X = np.column_stack(
      2     (
----> 3         test_df.distance_travel,
      4         test_df.passenger_count,
      5         test_df.pickup_longitude,

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'distance_travel'

## === cell 13
my_submission = pd.DataFrame({"key": test_df.key, "fare_amount": predicted_fare})
my_submission.to_csv("submission.csv", index=False)
my_submission.head()

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/800656253.py in <cell line: 0>()
----> 1 my_submission = pd.DataFrame({"key": test_df.key, "fare_amount": predicted_fare})
      2 my_submission.to_csv("submission.csv", index=False)
      3 my_submission.head()

NameError: name 'predicted_fare' is not defined
