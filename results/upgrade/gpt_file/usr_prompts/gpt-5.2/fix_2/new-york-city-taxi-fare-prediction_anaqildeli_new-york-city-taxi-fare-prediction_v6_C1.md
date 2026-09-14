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

3.10

# 3. Installed packages

folium==0.20.0
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
seaborn==0.12.2
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

3.41572

# 6. Current score

5.93093

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 5.93093) has done: 'Your notebook doesn’t currently train any model or write a `submission.csv`, so no Kaggle score can be produced. I make the smallest end-to-end additions: keep your existing feature engineering/cleaning, retain the `key` from the test set (needed for submission), and add a simple scikit-learn regression model plus RMSE validation to ensure it’s working. I also add a minimal, safe missing-value drop and fare clipping to non-negative at prediction time (RMSE-appropriate and prevents invalid negatives) without changing your core feature logic. Finally, I generate `submission.csv` with exactly `key,fare_amount` and the correct row alignment.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import seaborn as sns
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap



## === cell 1
fields = [
    "pickup_datetime",
    "fare_amount",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

train = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
    nrows=1000000,
    skipinitialspace=True,
    usecols=fields,
    parse_dates=["pickup_datetime"],
)
print(f"{train.shape} shape")
train.head()



## === cell 2
print(train.info())
train.describe()



## === cell 3
test = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
    parse_dates=["pickup_datetime"],
)
print(f"{test.shape} shape")
test.head()



## === cell 4
test.info()



## === cell 5
test.describe()



## === cell 6
train.describe()



## === cell 7
test_key = test["key"].copy()



## === cell 8
train.head()



## === cell 9
sns.countplot(x="passenger_count", data=train)



## === cell 10
train[train["passenger_count"] > 6]



## === cell 11
train = train.drop(train[train["passenger_count"] == 208].index)



## === cell 12
train = train.drop(train[train["fare_amount"] < 0].index)



## === cell 13
train["year"] = train["pickup_datetime"].dt.year
train["month"] = train["pickup_datetime"].dt.month
train["day"] = train["pickup_datetime"].dt.day
train["hour"] = train["pickup_datetime"].dt.hour
train["minute"] = train["pickup_datetime"].dt.minute

test["year"] = test["pickup_datetime"].dt.year
test["month"] = test["pickup_datetime"].dt.month
test["day"] = test["pickup_datetime"].dt.day
test["hour"] = test["pickup_datetime"].dt.hour
test["minute"] = test["pickup_datetime"].dt.minute



## === cell 14
train = train.drop(columns=["pickup_datetime"])
test = test.drop(columns=["pickup_datetime"])



## === cell 15
train.describe()



## === cell 16
variables = ["year", "month", "day", "hour"]
for var in variables:
    plt.figure()
    sns.countplot(x=var, data=train).set(title=f"Count plot of {var}")



## === cell 17
train.shape



## === cell 18
train.describe()



## === cell 19
train = train[(train["pickup_latitude"] > 40) & (train["pickup_latitude"] < 45)]
train = train[(train["dropoff_latitude"] > 40) & (train["dropoff_latitude"] < 45)]

train = train[(train["pickup_longitude"] < -71) & (train["pickup_longitude"] > -79)]
train = train[(train["dropoff_longitude"] < -71) & (train["dropoff_longitude"] > -79)]



## === cell 20
train.describe()



## === cell 21
train.shape



## === cell 22
ny_map = folium.Map(location=[40.71, -74.00], tiles="cartodbpositron", zoom_start=20)
HeatMap(data=train[["pickup_latitude", "pickup_longitude"]], radius=10).add_to(ny_map)
ny_map



## === cell 23
train = train.dropna().reset_index(drop=True)
test = test.dropna().reset_index(drop=True)

train = train[
    (train["passenger_count"] >= 1) & (train["passenger_count"] <= 6)
].reset_index(drop=True)
test["passenger_count"] = test["passenger_count"].clip(1, 6)

train.shape, test.shape



## === cell 24
from sklearn.ensemble import RandomForestRegressor

X = train.drop(columns=["fare_amount"])
y = train["fare_amount"].astype(float)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(
    n_estimators=120, random_state=42, n_jobs=-1, min_samples_leaf=2
)
model.fit(X_train, y_train)

val_pred = model.predict(X_val)
rmse = mean_squared_error(y_val, val_pred, squared=False)
print("Validation RMSE:", rmse)



## === cell 25
test_pred = model.predict(
    test.drop(columns=["key"], errors="ignore") if "key" in test.columns else test
)
test_pred = np.clip(test_pred, 0, None)

submission = pd.DataFrame({"key": test_key.values, "fare_amount": test_pred})

print(submission.head())
print("Submission shape:", submission.shape)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
