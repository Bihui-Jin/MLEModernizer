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

3.84109

# 6. Current score

5.35679

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.15099) has done: 'I fix the runtime error caused by deprecated `np.object` by using the built-in `object` dtype check, which is score-neutral but unblocks execution. I also fix the `date_extraction` function bug where `drop(..., inplace=True)` returns `None` and inadvertently overwrites the dataframe, which otherwise breaks downstream feature engineering and training. Finally, I correct the Haversine feature computation (wrong longitude variable and swapped diffs, plus missing return), which is a minimal logic bugfix directly improving the distance feature and should move RMSE down toward your target without changing the model family or training approach. I keep the RandomForest setup intact and ensure a valid `key,fare_amount` submission CSV is written.'
- What this solution (achieved 5.35679) has done: 'Your current RMSE (5.15099) is worse than the target (3.84109), so we should make small, safe fixes that improve generalization without changing the model family or overall approach. The biggest score drag in this script is that it trains on many physically impossible/outlier trips (bad lat/long ranges, absurd passenger_count, extreme fares/distances), which RandomForest overfit and which hurts leaderboard RMSE; adding standard NYC Taxi Fare cleaning filters is a minimal, competition-relevant adjustment. I also fix a minor date feature bug (your “weekday” is actually day-of-month) and ensure the simple distance feature is actually written back to the DataFrames (right now you’re not assigning the returned DataFrame in the loop). These are narrow changes to preprocessing/feature correctness that typically move RMSE down toward ~3–4 without altering the core RandomForest training loop.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing
import os

INPUT_DIR = "../input"
if not os.path.exists(INPUT_DIR):
    INPUT_DIR = "/kaggle/input"

print("Listing input dir:", INPUT_DIR)
print(os.listdir(INPUT_DIR))



## === cell 1
train = pd.read_csv(f"{INPUT_DIR}/train.csv", nrows=1000000)
test = pd.read_csv(f"{INPUT_DIR}/test.csv")
train.head()



## === cell 2
test.head()



## === cell 3
train.shape



## === cell 4
test.shape



## === cell 5
train.dtypes.value_counts()



## === cell 6
test.dtypes.value_counts()



## === cell 7
train.isnull().sum()



## === cell 8
train = train.dropna()
train.isnull().sum()



## === cell 9
(train == 0).astype(int).sum()



## === cell 10
train = train.loc[~(train == 0).any(axis=1)]



## === cell 11
(train == 0).astype(int).sum()



## === cell 12
train.shape



## === cell 13
train.describe()



## === cell 14
train.describe()



## === cell 15
train.dtypes.value_counts()



## === cell 16
object_data = train.dtypes == object
categoricals = train.columns[object_data]
categoricals



## === cell 17
train.drop("key", axis=1, inplace=True)
train.head()



## === cell 18
import datetime as dt


def date_extraction(data):
    data["pickup_datetime"] = pd.to_datetime(data["pickup_datetime"], errors="coerce")
    data["year"] = data["pickup_datetime"].dt.year
    data["month"] = data["pickup_datetime"].dt.month
    data["weekday"] = data["pickup_datetime"].dt.weekday
    data["hour"] = data["pickup_datetime"].dt.hour
    data.drop("pickup_datetime", axis=1, inplace=True)
    return data


date_extraction(train)



## === cell 19
train.head()



## === cell 20
date_extraction(test)
test.head()




## === cell 21
def long_lat_distance(x):
    x["Longitude_distance"] = np.radians(x["pickup_longitude"] - x["dropoff_longitude"])
    x["Latitude_distance"] = np.radians(x["pickup_latitude"] - x["dropoff_latitude"])
    x["distance_travelled/10e3"] = (
        (x["Longitude_distance"] ** 2 + x["Latitude_distance"] ** 2) ** 0.5
    ) * 1000
    return x




## === cell 22
train = long_lat_distance(train)
test = long_lat_distance(test)

train.head()




## === cell 23
def harvesine(x):
    r = 6371000  # meters
    theta_1 = np.radians(x["pickup_latitude"])
    theta_2 = np.radians(x["dropoff_latitude"])
    lambda_1 = np.radians(x["pickup_longitude"])
    lambda_2 = np.radians(x["dropoff_longitude"])

    theta_diff = theta_2 - theta_1
    lambda_diff = lambda_2 - lambda_1

    a = (
        np.sin(theta_diff / 2) ** 2
        + np.cos(theta_1) * np.cos(theta_2) * np.sin(lambda_diff / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    x["harvesine/km"] = (r * c) / 1000.0
    return x




## === cell 24
train = harvesine(train)
test = harvesine(test)

train.head()



## === cell 25
train.dtypes.value_counts()



## === cell 26
train.head()



## === cell 27
test.head()



## === cell 28
train.describe()



## === cell 29
print("Are there any nulls\nan in the train data: ")
print(train.isnull().sum())

print("\nAre there any nulls\nans in the test data: ")
print(test.isnull().sum())



## === cell 30
train["harvesine/km"] = train["harvesine/km"].fillna(train["harvesine/km"].median())
test["harvesine/km"] = test["harvesine/km"].fillna(train["harvesine/km"].median())



## === cell 31
train = train[(train["fare_amount"] > 0) & (train["fare_amount"] <= 250)].copy()

train = train[(train["passenger_count"] >= 1) & (train["passenger_count"] <= 6)].copy()

train = train[
    (train["pickup_longitude"].between(-74.5, -72.8))
    & (train["dropoff_longitude"].between(-74.5, -72.8))
    & (train["pickup_latitude"].between(40.5, 41.8))
    & (train["dropoff_latitude"].between(40.5, 41.8))
].copy()

train = train[(train["harvesine/km"] >= 0) & (train["harvesine/km"] <= 200)].copy()

train.shape



## === cell 32
from sklearn.ensemble import RandomForestRegressor

feature_cols = [x for x in train.columns if x != "fare_amount"]
X = train[feature_cols]
y = train["fare_amount"]



## === cell 33
correlations = X.corrwith(y)
correlations = abs(correlations * 100)
correlations.sort_values(ascending=False, inplace=True)

correlations



## === cell 34
ax = correlations.plot(kind="bar")
ax.set(ylim=[-1, 1], ylabel="pearson correlation")



## === cell 35
train.head()



## === cell 36
train_1 = train.drop(
    [
        "pickup_longitude",
        "dropoff_longitude",
        "pickup_latitude",
        "dropoff_latitude",
        "Longitude_distance",
        "Latitude_distance",
    ],
    axis=1,
)

train_1.head()



## === cell 37
train_1["harvesine/km"] = train_1["harvesine/km"].round(2)
train_1["distance_travelled/10e3"] = train_1["distance_travelled/10e3"].round(2)

train_1.head()



## === cell 38
train_1.describe()



## === cell 39
test_1 = test.drop(
    [
        "pickup_longitude",
        "dropoff_longitude",
        "pickup_latitude",
        "dropoff_latitude",
        "Longitude_distance",
        "Latitude_distance",
    ],
    axis=1,
)

test_1.head()



## === cell 40
from sklearn.model_selection import train_test_split

feat_cols = [x for x in train_1.columns if x != "fare_amount"]
X_1 = train_1[feat_cols]
y_1 = train_1["fare_amount"]
X_train, X_test, y_train, y_test = train_test_split(
    X_1, y_1, test_size=0.25, random_state=42
)



## === cell 41
rf = RandomForestRegressor(n_estimators=100, max_features=5, random_state=42, n_jobs=-1)
rf = rf.fit(X_train, y_train)



## === cell 42
test.head()



## === cell 43
final_prediction = rf.predict(X_test)



## === cell 44
test_1.drop("key", axis=1, inplace=True)
test_1.head()



## === cell 45
final_prediction = rf.predict(test_1)

NYCtaxiFare_submission = pd.DataFrame(
    {"key": test.key, "fare_amount": final_prediction}
)
NYCtaxiFare_submission.to_csv("NYCtaxiFare_prediction.csv", index=False)



## === cell 46
NYCtaxiFare_submission.head()
