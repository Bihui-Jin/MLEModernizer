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

4.07608

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, glob
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

print("Available top‑level items:", os.listdir("."))

train_candidates = glob.glob(os.path.join("data", "**", "train.csv"), recursive=True)
test_candidates = glob.glob(os.path.join("data", "**", "test.csv"), recursive=True)

if not train_candidates:
    raise FileNotFoundError("train.csv not found under ./data/")
if not test_candidates:
    raise FileNotFoundError("test.csv not found under ./data/")

train_path = train_candidates[0]
test_path = test_candidates[0]

print("Using train path:", train_path)
print("Using test path :", test_path)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2238055224.py in <cell line: 0>()
     13 
     14 if not train_candidates:
---> 15     raise FileNotFoundError("train.csv not found under ./data/")
     16 if not test_candidates:
     17     raise FileNotFoundError("test.csv not found under ./data/")

FileNotFoundError: train.csv not found under ./data/

## === cell 1
df = pd.read_csv(train_path, nrows=1_000_000)
df.head()




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3349002108.py in <cell line: 0>()
      1 # read a subset of the training data (1 000 000 rows) to keep memory use modest
----> 2 df = pd.read_csv(train_path, nrows=1_000_000)
      3 df.head()
      4 
      5 

NameError: name 'train_path' is not defined

## === cell 2
df = df[df.passenger_count > 0]
df = df[df.fare_amount > 0]
df.head()




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1392046181.py in <cell line: 0>()
      1 # basic cleaning: keep only realistic rows
----> 2 df = df[df.passenger_count > 0]
      3 df = df[df.fare_amount > 0]
      4 df.head()
      5 

NameError: name 'df' is not defined

## === cell 3
alpha_ang = 0.506  # constant used in distance computation


def distance_travel(df):
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs() * 50
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs() * 69
    df["displacement_vector"] = np.sqrt(
        df.abs_diff_latitude**2 + df.abs_diff_longitude**2
    )
    df["actual_long"] = (
        df.displacement_vector
        * np.sin(np.arctan(df.abs_diff_longitude / df.abs_diff_latitude) - alpha_ang)
    ).abs()
    df["actual_lat"] = (
        df.displacement_vector
        * np.cos(np.arctan(df.abs_diff_longitude / df.abs_diff_latitude) - alpha_ang)
    ).abs()
    df["distance_travel"] = df["actual_long"] + df["actual_lat"]
    return df


df = distance_travel(df)
df = df[df.distance_travel > 0]
df.head()




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/774705502.py in <cell line: 0>()
     22 
     23 
---> 24 df = distance_travel(df)
     25 df = df[df.distance_travel > 0]
     26 df.head()

NameError: name 'df' is not defined

## === cell 4
df = df[df.distance_travel < 30]
df = df[df.fare_amount < 100]
df.head()




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2062563770.py in <cell line: 0>()
      1 # remove extreme outliers that would hurt the simple model
----> 2 df = df[df.distance_travel < 30]
      3 df = df[df.fare_amount < 100]
      4 df.head()
      5 

NameError: name 'df' is not defined

## === cell 5
l = len(df)
print("Total rows after filtering:", l)

train_df = df.iloc[: int(0.7 * l)]
val_df = df.iloc[int(0.7 * l) :]

train_X = np.column_stack(
    (train_df.distance_travel, train_df.passenger_count, np.ones(len(train_df)))
)
val_X = np.column_stack(
    (val_df.distance_travel, val_df.passenger_count, np.ones(len(val_df)))
)
train_y = train_df.fare_amount.values
val_y = val_df.fare_amount.values




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/141024166.py in <cell line: 0>()
----> 1 l = len(df)
      2 print("Total rows after filtering:", l)
      3 
      4 train_df = df.iloc[: int(0.7 * l)]
      5 val_df = df.iloc[int(0.7 * l) :]

NameError: name 'df' is not defined

## === cell 6
imp = SimpleImputer(strategy="mean")
train_X = imp.fit_transform(train_X)
val_X = imp.transform(val_X)

regr = GradientBoostingRegressor(random_state=21, n_estimators=800, learning_rate=0.05)
regr.fit(train_X, train_y)

val_pred = regr.predict(val_X)
rmse = mean_squared_error(val_y, val_pred, squared=False)
print(f"Validation RMSE: {rmse:.4f}")




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2451612015.py in <cell line: 0>()
      1 # impute possible missing values (none expected but kept for safety)
      2 imp = SimpleImputer(strategy="mean")
----> 3 train_X = imp.fit_transform(train_X)
      4 val_X = imp.transform(val_X)
      5 

NameError: name 'train_X' is not defined

## === cell 7
tdf = pd.read_csv(test_path)
tdf = distance_travel(tdf)  # add the same engineered feature
tdf.head()




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2602843661.py in <cell line: 0>()
      1 # load the official test set (full size – about 10 k rows)
----> 2 tdf = pd.read_csv(test_path)
      3 tdf = distance_travel(tdf)  # add the same engineered feature
      4 tdf.head()
      5 

NameError: name 'test_path' is not defined

## === cell 8
ttrain_X = np.column_stack(
    (tdf.distance_travel, tdf.passenger_count, np.ones(len(tdf)))
)
ttrain_X = imp.transform(ttrain_X)
output = regr.predict(ttrain_X)
print("First 5 predictions:", output[:5])




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/27500209.py in <cell line: 0>()
      1 ttrain_X = np.column_stack(
----> 2     (tdf.distance_travel, tdf.passenger_count, np.ones(len(tdf)))
      3 )
      4 ttrain_X = imp.transform(ttrain_X)
      5 output = regr.predict(ttrain_X)

NameError: name 'tdf' is not defined

## === cell 9
submission = pd.DataFrame({"key": tdf["key"], "fare_amount": output})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
submission.head()

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/633882563.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"key": tdf["key"], "fare_amount": output})
      2 submission_path = "submission.csv"
      3 submission.to_csv(submission_path, index=False)
      4 print(f"Submission saved to {submission_path}")
      5 submission.head()

NameError: name 'tdf' is not defined
