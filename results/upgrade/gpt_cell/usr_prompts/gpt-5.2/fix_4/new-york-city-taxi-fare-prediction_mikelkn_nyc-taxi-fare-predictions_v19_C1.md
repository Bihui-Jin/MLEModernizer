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

9.29087

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.1871) has done: 'Diagnosis: Cell 18 crashes because `np.object` was removed in NumPy 1.26, so comparing dtypes to `np.object` raises `AttributeError`. The intended behavior is simply to identify columns with Python `object` dtype (string-like/categorical).  
Patch summary: Replace `np.object` with the builtin `object` so the dtype comparison works on modern NumPy without changing semantics. This keeps `object_data` and `categoricals` exactly as expected for downstream cells.  
Updated cells: Only cell 18 is modified.  
Compatibility notes for cell k+1: Cell 19 expects `train` to be unchanged except for prior processing; this patch only defines `object_data`/`categoricals` and does not alter `train`, so cell 19 remains compatible.  
Assumptions: The notebook’s goal in cell 18 is to detect object-typed columns (e.g., `pickup_datetime`) and not to use any NumPy-specific dtype alias.'
- What this solution (achieved 5.19638) has done: 'Your current pipeline is already close to correct, but two small feature-engineering bugs are likely hurting RMSE: `date_extraction()` mistakenly returns `None` (because it assigns the result of an inplace drop), and `harvesine()` mistakenly sets `lambda_2` to the dropoff longitude instead of the pickup longitude, distorting the distance feature. Fixing those preserves your core approach (same features + RandomForest) while making the engineered features valid and consistent between train/test, which should improve the score toward your 3.84 target. I also mirror the existing train NaN handling by filling `harvesine/km` NaNs in the test set to avoid unpredictable behavior at inference time. All paths and the submission schema remain unchanged, and the script still writes a valid `.csv`.'
- What this solution (achieved 9.29087) has done: 'Your current score (5.196) is worse than the target (3.841), so we should improve RMSE with minimal, low-risk fixes that preserve your RandomForest + engineered-distance core logic. The biggest remaining issue is that your Haversine implementation is mathematically inconsistent (it uses precomputed “distance” deltas in the wrong place and never uses the computed lat/lon radians properly), which makes `harvesine/km` noisy and hurts generalization; I correct it while keeping the same feature name and usage. I also ensure `harvesine/km` is always created (the current function doesn’t explicitly return, relying on in-place mutation) and clamp clearly invalid fare values/coordinates out of the 1M-row training slice to reduce label noise (a standard NYC Taxi Fare cleanup that stays within your existing “drop bad rows” approach). Finally, I keep the submission schema/path the same and guarantee alignment of `key` with predictions.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

print(os.listdir("../input"))



## === cell 1
train = pd.read_csv("../input/train.csv", nrows=1000000)
test = pd.read_csv("../input/test.csv")
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
    data["pickup_datetime"] = pd.to_datetime(data["pickup_datetime"])
    data["year"] = data["pickup_datetime"].dt.year
    data["month"] = data["pickup_datetime"].dt.month
    data["weekday"] = data["pickup_datetime"].dt.day
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
for x in [train, test]:
    long_lat_distance(x)
train.head()




## === cell 23
def harvesine(x):
    r = 6371.0  # km

    lat1 = np.radians(x["pickup_latitude"])
    lat2 = np.radians(x["dropoff_latitude"])
    lon1 = np.radians(x["pickup_longitude"])
    lon2 = np.radians(x["dropoff_longitude"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    x["harvesine/km"] = r * c
    return x




## === cell 24
for x in [train, test]:
    harvesine(x)
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
train = train[(train["fare_amount"] > 0) & (train["fare_amount"] <= 250)]
train = train[(train["passenger_count"] >= 1) & (train["passenger_count"] <= 6)]

train = train[
    (train["pickup_longitude"].between(-74.3, -73.6))
    & (train["dropoff_longitude"].between(-74.3, -73.6))
    & (train["pickup_latitude"].between(40.5, 41.0))
    & (train["dropoff_latitude"].between(40.5, 41.0))
]



## === cell 31
train["harvesine/km"] = train["harvesine/km"].fillna(train["harvesine/km"].median())
test["harvesine/km"] = test["harvesine/km"].fillna(train["harvesine/km"].median())



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
rf = RandomForestRegressor(n_estimators=100, max_features=5)
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
