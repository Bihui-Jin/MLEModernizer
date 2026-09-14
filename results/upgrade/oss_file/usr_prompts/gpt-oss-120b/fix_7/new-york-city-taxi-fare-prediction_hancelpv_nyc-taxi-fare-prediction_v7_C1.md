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

3.89954

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 5.05247) has done: 'I fixed the deprecated pandas attribute, removed the remaining datetime column after feature extraction, and ensured the feature list matches the updated dataframe so the model can be trained and a valid `submission.csv` is written.'
- What this solution (achieved 5.83364) has done: 'I add a geographic haversine distance feature (which better captures true travel distance) and filter out obviously invalid coordinate rows, then modestly increase the forest size and use the default “sqrt” max_features. These small, targeted tweaks should lower the RMSE toward the target without altering the overall model pipeline.'
- What this solution (achieved 5.24621) has done: 'I filter out rows with non‑positive fare amounts before creating the log‑target, which removes the NaNs that caused the fit to fail. This minimal change lets the model train, produce predictions, and write a valid `submission.csv` while preserving the existing pipeline and keeping the score as close as possible to the target.'

# 9. Code solution

## === cell 0
import os, math
import numpy as np
import pandas as pd

print(os.listdir("../input"))




## === cell 1
train = pd.read_csv("../input/train.csv", nrows=2_000_000)
test = pd.read_csv("../input/test.csv")
sample_sub = pd.read_csv("../input/sample_submission.csv")




## === cell 2
train = train.dropna(how="any", axis="rows")

train = train[train["fare_amount"] > 0]

coord_mask = (
    train["pickup_longitude"].between(-75, -73)
    & train["pickup_latitude"].between(40, 42)
    & train["dropoff_longitude"].between(-75, -73)
    & train["dropoff_latitude"].between(40, 42)
)
train = train[coord_mask]




## === cell 3
all_data = pd.concat((train, test)).reset_index(drop=True)

all_data.drop(["fare_amount"], axis=1, inplace=True)

y = train["fare_amount"].values
y_log = np.log1p(y)

n_train = len(train)
test_id = test["key"]




## === cell 4
def week_num(day):
    """Return the week number label of a month for a given day of month."""
    if day <= 7:
        return "first"
    if day <= 14:
        return "second"
    if day <= 21:
        return "third"
    if day <= 28:
        return "fourth"
    return "fifth"




## === cell 5
def add_time_features(df):
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

    df["hour"] = df["pickup_datetime"].dt.hour.astype(str)
    df["day_of_week"] = df["pickup_datetime"].dt.day_name()
    df["day_of_month"] = df["pickup_datetime"].dt.day
    df["week_of_month"] = df["day_of_month"].map(week_num)
    df["month"] = df["pickup_datetime"].dt.month.astype(str)
    df["year"] = df["pickup_datetime"].dt.year.astype(str)

    df.drop(["day_of_month", "pickup_datetime"], axis=1, inplace=True)
    return df




## === cell 6
def add_geo_features(df):
    df["abs_diff_longitude"] = (df["dropoff_longitude"] - df["pickup_longitude"]).abs()
    df["abs_diff_latitude"] = (df["dropoff_latitude"] - df["pickup_latitude"]).abs()

    df["manhattan_distance"] = df["abs_diff_longitude"] + df["abs_diff_latitude"]

    df["squared_long"] = np.power(df["abs_diff_longitude"], 2)
    df["squared_lat"] = np.power(df["abs_diff_latitude"], 2)

    df["euclid_disance"] = np.sqrt(df["squared_long"] + df["squared_lat"])

    def haversine_np(lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        )
        c = 2 * np.arcsin(np.sqrt(a))
        km = 6371 * c
        return km

    df["haversine_distance"] = haversine_np(
        df["pickup_longitude"],
        df["pickup_latitude"],
        df["dropoff_longitude"],
        df["dropoff_latitude"],
    )
    return df




## === cell 7
all_data = add_time_features(all_data)
all_data = add_geo_features(all_data)




## === cell 8
features = [
    "passenger_count",
    "hour",
    "day_of_week",
    "week_of_month",
    "month",
    "year",
    "abs_diff_longitude",
    "abs_diff_latitude",
    "euclid_disance",
    "manhattan_distance",
    "haversine_distance",
]

all_data = all_data[features]

all_data = pd.get_dummies(all_data)




## === cell 9
X_train = all_data[:n_train]
X_test = all_data[n_train:]




## === cell 10
from sklearnex import patch

patch(maximize=True)

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=1200,  # more trees for a stronger ensemble
    max_depth=None,  # allow deeper trees to capture complex patterns
    max_features=0.8,  # use a larger subset of features per split
    min_samples_leaf=1,  # smaller leaves for finer granularity
    min_samples_split=2,  # split as soon as possible
    bootstrap=True,
    random_state=42,
    n_jobs=-1,
)




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/1515705238.py in <cell line: 0>()
      1 # Enable Intel(R) Extension for Scikit‑Learn to accelerate RandomForest without changing any algorithmic parameters.
----> 2 from sklearnex import patch
      3 
      4 patch(maximize=True)
      5 

ImportError: cannot import name 'patch' from 'sklearnex' (/usr/local/lib/python3.11/dist-packages/sklearnex/__init__.py)

## === cell 11
model.fit(X_train, y_log)




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1563105408.py in <cell line: 0>()
----> 1 model.fit(X_train, y_log)
      2 
      3 

NameError: name 'model' is not defined

## === cell 12
test_pred_log = model.predict(X_test)
test_pred = np.expm1(test_pred_log)




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3193900741.py in <cell line: 0>()
----> 1 test_pred_log = model.predict(X_test)
      2 test_pred = np.expm1(test_pred_log)
      3 
      4 

NameError: name 'model' is not defined

## === cell 13
submission = pd.DataFrame({"key": test_id, "fare_amount": test_pred})
submission.to_csv("submission.csv", index=False)

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1193234245.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"key": test_id, "fare_amount": test_pred})
      2 submission.to_csv("submission.csv", index=False)

NameError: name 'test_pred' is not defined
