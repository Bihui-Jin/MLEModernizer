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

4.08766

# 6. Current score

5.52155

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 5.52155) has done: 'I replace the obsolete `Imputer` with `SimpleImputer`, ensure all libraries are imported before use, and reorganize the cells so every variable is defined before it is referenced. The core feature engineering (distance calculation) and model (GradientBoostingRegressor) remain unchanged, only the missing‑value handling and a modest increase in `n_estimators` are applied to push the RMSE a bit closer to the target while still producing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

print("input folder contents:", os.listdir("../input"))




## === cell 1
train_path = "../input/train.csv"
df = pd.read_csv(train_path, nrows=1_000_000)  # use 1 M rows to stay within limits
df.head()




## === cell 2
df = df[df.passenger_count > 0]
df = df[df.fare_amount > 0]

alpha_ang = 0.506


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


distance_travel(df)

df = df[df.distance_travel > 0]
df = df[df.distance_travel < 30]  # keep realistic trips
df = df[df.fare_amount < 100]  # cap extreme fares




## === cell 4
l = len(df)
df_train = df.iloc[: int(0.7 * l)]
df_valid = df.iloc[int(0.7 * l) :]

train_X = np.column_stack(
    (df_train.distance_travel, df_train.passenger_count, np.ones(len(df_train)))
)
valid_X = np.column_stack(
    (df_valid.distance_travel, df_valid.passenger_count, np.ones(len(df_valid)))
)

train_y = df_train.fare_amount.values
valid_y = df_valid.fare_amount.values




## === cell 5
imputer = SimpleImputer(strategy="mean")
train_X_imp = imputer.fit_transform(train_X)
valid_X_imp = imputer.transform(valid_X)

regr = GradientBoostingRegressor(n_estimators=200, random_state=42)
regr.fit(train_X_imp, train_y)

pred_valid = regr.predict(valid_X_imp)
rmse = mean_squared_error(valid_y, pred_valid, squared=False)
print(f"Validation RMSE: {rmse:.4f}")




## === cell 6
test_path = "../input/test.csv"
tdf = pd.read_csv(test_path, nrows=1_000_000)  # full test is small; limit is safe
distance_travel(tdf)

test_X = np.column_stack((tdf.distance_travel, tdf.passenger_count, np.ones(len(tdf))))
test_X_imp = imputer.transform(test_X)
output = regr.predict(test_X_imp)




## === cell 7
submission = pd.DataFrame({"key": tdf["key"], "fare_amount": output})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
print(submission.head())
