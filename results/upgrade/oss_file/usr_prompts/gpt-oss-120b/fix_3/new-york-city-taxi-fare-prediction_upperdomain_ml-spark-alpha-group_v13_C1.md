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

4.22828

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 6.55718) has done: 'The main fixes are:
* Replace the removed `Imputer` with `SimpleImputer`.
* Ensure the import error no longer stops the notebook, so all functions (including `chunck_generator`) are defined.
* Adjust the imputer initialization to use `np.nan`.
* Keep the overall logic unchanged while making the script runnable and capable of writing a proper `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor

print(os.listdir("../input"))


def chunck_generator(filename, header=False, chunk_size=10**6):
    """Yield CSV chunks for large files."""
    for chunk in pd.read_csv(
        filename,
        delimiter=",",
        iterator=True,
        chunksize=chunk_size,
        parse_dates=[1],  # pickup_datetime column
    ):
        yield chunk




## === cell 1
alpha_ang = 0.506


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




## === cell 2
def add_time_features(df):
    """Extract simple time based features from pickup_datetime."""
    df["hour"] = df.pickup_datetime.dt.hour
    df["dayofweek"] = df.pickup_datetime.dt.dayofweek
    df["month"] = df.pickup_datetime.dt.month
    return df


def data_clean(df):
    df = df[df.passenger_count > 0]
    df.fare_amount = df.fare_amount.astype(np.float64)
    df = df[df.fare_amount > 0]
    distance_travel(df)  # adds travel columns in‑place
    add_time_features(df)  # adds hour, dayofweek, month
    df = df[df.distance_travel > 0]
    return df




## === cell 3
def remove_outliers(df):
    df = df[df.distance_travel < 30]
    df = df[df.fare_amount < 100]
    return df




## === cell 4
def graph_presesnt(df):
    test = df[df.passenger_count == 1]
    _ = test.iloc[: len(test)].plot.scatter("distance_travel", "fare_amount")
    _ = df.iloc[:100000].plot.scatter("distance_travel", "fare_amount")




## === cell 5
def incremental_training(train_X, train_y, regr):
    regr.fit(train_X, train_y)
    return regr




## === cell 6
filename = r"../input/train.csv"
gen = chunck_generator(filename=filename)

regr = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    warm_start=True,
    random_state=42,
)

imp = SimpleImputer(missing_values=np.nan, strategy="mean")

t = 56
while t > 0:
    print(f"Chunk {57 - t} / 56")
    df = next(gen)

    df = data_clean(df)
    df = remove_outliers(df)

    l = len(df)
    if l == 0:
        t -= 1
        continue

    df_train = df[: int(0.9 * l)]
    df_test = df[int(0.9 * l) :]

    train_X = np.column_stack(
        (
            df_train.distance_travel,
            df_train.passenger_count,
            df_train.hour,
            df_train.dayofweek,
            df_train.month,
            np.ones(len(df_train)),  # bias term retained from original code
        )
    )
    test_X = np.column_stack(
        (
            df_test.distance_travel,
            df_test.passenger_count,
            df_test.hour,
            df_test.dayofweek,
            df_test.month,
            np.ones(len(df_test)),
        )
    )
    train_y = np.array(df_train.fare_amount)
    test_y = np.array(df_test.fare_amount)

    imp = imp.fit(train_X)
    train_X = imp.transform(train_X)
    test_X = imp.transform(test_X)

    regr = incremental_training(train_X, train_y, regr)
    print(
        "Chunk RMSE (approx):",
        np.sqrt(((regr.predict(test_X) - test_y) ** 2).mean()),
    )
    t -= 1




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1008015736.py in <cell line: 0>()
     17     df = next(gen)
     18 
---> 19     df = data_clean(df)
     20     df = remove_outliers(df)
     21 

/tmp/ipykernel_11/734834694.py in data_clean(df)
     12     df = df[df.fare_amount > 0]
     13     distance_travel(df)  # adds travel columns in‑place
---> 14     add_time_features(df)  # adds hour, dayofweek, month
     15     df = df[df.distance_travel > 0]
     16     return df

/tmp/ipykernel_11/734834694.py in add_time_features(df)
      1 def add_time_features(df):
      2     """Extract simple time based features from pickup_datetime."""
----> 3     df["hour"] = df.pickup_datetime.dt.hour
      4     df["dayofweek"] = df.pickup_datetime.dt.dayofweek
      5     df["month"] = df.pickup_datetime.dt.month

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

/usr/local/lib/python3.11/dist-packages/pandas/core/accessor.py in __get__(self, obj, cls)
    222             # we're accessing the attribute of the class, i.e., Dataset.geo
    223             return self._accessor
--> 224         accessor_obj = self._accessor(obj)
    225         # Replace the property with the accessor object. Inspired by:
    226         # https://www.pydanny.com/cached-property.html

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/accessors.py in __new__(cls, data)
    641             return PeriodProperties(data, orig)
    642 
--> 643         raise AttributeError("Can only use .dt accessor with datetimelike values")

AttributeError: Can only use .dt accessor with datetimelike values

## === cell 7
test_df = pd.read_csv("../input/test.csv", parse_dates=[1])
distance_travel(test_df)
add_time_features(test_df)

test_X = np.column_stack(
    (
        test_df.distance_travel,
        test_df.passenger_count,
        test_df.hour,
        test_df.dayofweek,
        test_df.month,
        np.ones(len(test_df)),
    )
)
test_X = imp.transform(test_X)

predicted_fare = regr.predict(test_X)
print("Sample predictions:", predicted_fare[:5])




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/4112998947.py in <cell line: 0>()
     13     )
     14 )
---> 15 test_X = imp.transform(test_X)
     16 
     17 predicted_fare = regr.predict(test_X)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py in wrapped(self, X, *args, **kwargs)
    138     @wraps(f)
    139     def wrapped(self, X, *args, **kwargs):
--> 140         data_to_wrap = f(self, X, *args, **kwargs)
    141         if isinstance(data_to_wrap, tuple):
    142             # only wrap the first output for cross decomposition

/usr/local/lib/python3.11/dist-packages/sklearn/impute/_base.py in transform(self, X)
    547             `X` with imputed values.
    548         """
--> 549         check_is_fitted(self)
    550 
    551         X = self._validate_input(X, in_fit=False)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_is_fitted(estimator, attributes, msg, all_or_any)
   1388 
   1389     if not fitted:
-> 1390         raise NotFittedError(msg % {"name": type(estimator).__name__})
   1391 
   1392 

NotFittedError: This SimpleImputer instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.

## === cell 8
my_submission = pd.DataFrame({"key": test_df.key, "fare_amount": predicted_fare})
my_submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv (first rows):")
print(my_submission.head())

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3845590556.py in <cell line: 0>()
----> 1 my_submission = pd.DataFrame({"key": test_df.key, "fare_amount": predicted_fare})
      2 my_submission.to_csv("submission.csv", index=False)
      3 print("Submission saved to submission.csv (first rows):")
      4 print(my_submission.head())

NameError: name 'predicted_fare' is not defined
