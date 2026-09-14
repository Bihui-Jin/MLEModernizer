# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # CSV file I/O (e.g. pd.read_csv)
import os  # reading the input files we have access to

print(os.listdir("../input"))



## === cell 1
train_path = "../input/train.csv"
train_df = pd.read_csv(
    train_path,
    skiprows=lambda i: i > 0
    and np.random.RandomState(42).rand() > (1_000_000 / 55_423_856),
    nrows=1_000_000,
)
train_df.dtypes




## === cell 2
def add_travel_vector_features(df):
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs()


add_travel_vector_features(train_df)



## === cell 3
print(train_df.isnull().sum())



## === cell 4
t = len(train_df)
print(f"Old size {t}")
train_df = train_df.dropna(how="any", axis="rows")
print("New size: %d" % len(train_df))



## === cell 5
plot = train_df.iloc[:2000].plot.scatter("abs_diff_longitude", "abs_diff_latitude")



## === cell 6
print("Old size: %d" % len(train_df))

train_df = train_df[(train_df["fare_amount"] > 0) & (train_df["fare_amount"] <= 500)]

train_df = train_df[
    (train_df["passenger_count"] >= 1) & (train_df["passenger_count"] <= 6)
]

nyc_bbox = {
    "pickup_longitude": (-74.3, -72.9),
    "dropoff_longitude": (-74.3, -72.9),
    "pickup_latitude": (40.5, 41.0),
    "dropoff_latitude": (40.5, 41.0),
}
for col, (lo, hi) in nyc_bbox.items():
    train_df = train_df[(train_df[col] >= lo) & (train_df[col] <= hi)]

train_df = train_df.loc[
    (train_df.abs_diff_longitude < 5.0) & (train_df.abs_diff_latitude < 5.0)
]

print("New size: %d" % len(train_df))



## === cell 7
test_df = pd.read_csv("../input/test.csv")
test_df.dtypes



## === cell 8
add_travel_vector_features(test_df)



## === cell 9
print("Test rows kept:", len(test_df))



## === cell 10
train_df["pickup_datetime"].head()



## === cell 11
test_df["pickup_datetime"].head()



## === cell 12
ls1 = list(train_df["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][11:-7]
train_df["pickup_time"] = ls1

ls2 = list(test_df["pickup_datetime"])
for i in range(len(ls2)):
    ls2[i] = ls2[i][11:-7]
test_df["pickup_time"] = ls2



## === cell 13
ls1 = list(train_df["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][:-4:]
    ls1[i] = pd.Timestamp(ls1[i])
    ls1[i] = ls1[i].weekday()
train_df["Weekday"] = ls1

ls = list(test_df["pickup_datetime"])
for i in range(len(ls)):
    ls[i] = ls[i][:-4:]
    ls[i] = pd.Timestamp.weekday(pd.Timestamp(ls[i]))
test_df["Weekday"] = ls



## === cell 14
train_df.head()



## === cell 15
train_df.info()



## === cell 16
train_df.drop("pickup_datetime", inplace=True, axis=1)
test_df.drop("pickup_datetime", inplace=True, axis=1)



## === cell 17
train_df.head()



## === cell 18
train_df["Weekday"].replace(
    to_replace=[i for i in range(0, 7)],
    value=[
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ],
    inplace=True,
)
test_df["Weekday"].replace(
    to_replace=[i for i in range(0, 7)],
    value=[
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ],
    inplace=True,
)



## === cell 19
train_df.head()



## === cell 20
train_onehot = pd.get_dummies(train_df["Weekday"])
test_onehot = pd.get_dummies(test_df["Weekday"])
train_onehot, test_onehot = train_onehot.align(
    test_onehot, join="outer", axis=1, fill_value=0
)

train_df = pd.concat([train_df, train_onehot], axis=1)
test_df = pd.concat([test_df, test_onehot], axis=1)



## === cell 21
train_df.drop("Weekday", axis=1, inplace=True)
test_df.drop("Weekday", axis=1, inplace=True)



## === cell 22
train_df.head()



## === cell 23
type(train_df["pickup_time"][0])



## === cell 24
ls1 = list(train_df["pickup_time"])
for i in range(len(ls1)):
    z = ls1[i].split(":")
    ls1[i] = int(z[0]) * 100 + int(z[1])
train_df["pickup_time"] = ls1

ls1 = list(test_df["pickup_time"])
for i in range(len(ls1)):
    z = ls1[i].split(":")
    ls1[i] = int(z[0]) * 100 + int(z[1])
test_df["pickup_time"] = ls1



## === cell 25
m = len(train_df)
print(m)



## === cell 26
train_df["pickup_time"].head()



## === cell 27
type(train_df["pickup_time"])



## === cell 28
ls = list(train_df["pickup_time"])
m = len(ls)
for i in range(m):
    if ls[i] > 700 and ls[i] < 1000:
        ls[i] = "peak"
    elif ls[i] > 1600 and ls[i] < 2000:
        ls[i] = "peak"
    else:
        ls[i] = "not Peak"
train_df["Peak_hour"] = ls



## === cell 29
train_df.head()



## === cell 30
ls = list(test_df["pickup_time"])
m = len(ls)
for i in range(m):
    if ls[i] > 700 and ls[i] < 1000:
        ls[i] = "peak"
    elif ls[i] > 1600 and ls[i] < 2000:
        ls[i] = "peak"
    else:
        ls[i] = "not Peak"
test_df["Peak_hour"] = ls



## === cell 31
trainoh = pd.get_dummies(train_df["Peak_hour"])
testoh = pd.get_dummies(test_df["Peak_hour"])
trainoh, testoh = trainoh.align(testoh, join="outer", axis=1, fill_value=0)

train_df = pd.concat([train_df, trainoh], axis=1)
test_df = pd.concat([test_df, testoh], axis=1)



## === cell 32
test_df.tail()



## === cell 33
train_df.drop("Peak_hour", inplace=True, axis=1)
test_df.drop("Peak_hour", inplace=True, axis=1)

train_df.drop("pickup_time", inplace=True, axis=1)
test_df.drop("pickup_time", inplace=True, axis=1)



## === cell 34
train_df.head()



## === cell 35
R = 6373.0
lat1 = np.asarray(np.radians(train_df["pickup_latitude"]))
lon1 = np.asarray(np.radians(train_df["pickup_longitude"]))
lat2 = np.asarray(np.radians(train_df["dropoff_latitude"]))
lon2 = np.asarray(np.radians(train_df["dropoff_longitude"]))

dlat = lat2 - lat1
dlon = lon2 - lon1  # FIXED
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
train_df["Distance"] = np.asarray(distance) * 0.621

lat1 = np.asarray(np.radians(test_df["pickup_latitude"]))
lon1 = np.asarray(np.radians(test_df["pickup_longitude"]))
lat2 = np.asarray(np.radians(test_df["dropoff_latitude"]))
lon2 = np.asarray(np.radians(test_df["dropoff_longitude"]))

dlat = lat2 - lat1
dlon = lon2 - lon1  # FIXED
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
test_df["Distance"] = np.asarray(distance) * 0.621



## === cell 36
R = 6373.0
lat1 = np.asarray(np.radians(train_df["pickup_latitude"]))
lon1 = np.asarray(np.radians(train_df["pickup_longitude"]))
lat2 = np.asarray(np.radians(train_df["dropoff_latitude"]))
lon2 = np.asarray(np.radians(train_df["dropoff_longitude"]))

lat3 = np.zeros(len(train_df)) + np.radians(40.6413111)
lon3 = np.zeros(len(train_df)) + np.radians(-73.7781391)

dlat_pickup = lat3 - lat1
dlon_pickup = lon3 - lon1
dlat_dropoff = lat3 - lat2
dlon_dropoff = lon3 - lon2

a1 = (
    np.sin(dlat_pickup / 2) ** 2
    + np.cos(lat1) * np.cos(lat3) * np.sin(dlon_pickup / 2) ** 2
)
c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
distance1 = R * c1
train_df["pickup_Distance_airport"] = np.asarray(distance1) * 0.621

a2 = (
    np.sin(dlat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(dlon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
train_df["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621



## === cell 37
R = 6373.0
lat1 = np.asarray(np.radians(test_df["pickup_latitude"]))
lon1 = np.asarray(np.radians(test_df["pickup_longitude"]))
lat2 = np.asarray(np.radians(test_df["dropoff_latitude"]))
lon2 = np.asarray(np.radians(test_df["dropoff_longitude"]))

lat3 = np.zeros(len(test_df)) + np.radians(40.6413111)
lon3 = np.zeros(len(test_df)) + np.radians(-73.7781391)

dlat_pickup = lat3 - lat1
dlon_pickup = lon3 - lon1
dlat_dropoff = lat3 - lat2
dlon_dropoff = lon3 - lon2

a1 = (
    np.sin(dlat_pickup / 2) ** 2
    + np.cos(lat1) * np.cos(lat3) * np.sin(dlon_pickup / 2) ** 2
)
c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
distance1 = R * c1
test_df["pickup_Distance_airport"] = np.asarray(distance1) * 0.621

a2 = (
    np.sin(dlat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(dlon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
test_df["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621



## === cell 38
train_df["Distance"] = np.round(train_df["Distance"], 2)
train_df["pickup_Distance_airport"] = np.round(train_df["pickup_Distance_airport"], 2)
train_df["Dropoff_Distance_airport"] = np.round(train_df["Dropoff_Distance_airport"], 2)
test_df["Distance"] = np.round(test_df["Distance"], 2)
test_df["pickup_Distance_airport"] = np.round(test_df["pickup_Distance_airport"], 2)
test_df["Dropoff_Distance_airport"] = np.round(test_df["Dropoff_Distance_airport"], 2)



## === cell 39
train_df.head()



## === cell 40
_train_mean = float(train_df["abs_diff_longitude"].mean())
_train_var = float(train_df["abs_diff_longitude"].var())
_train_std = (
    float(np.sqrt(_train_var)) if _train_var > 0 and not np.isnan(_train_var) else 1.0
)

train_df["abs_diff_longitude"] = (
    train_df["abs_diff_longitude"] - _train_mean
) / _train_std
test_df["abs_diff_longitude"] = (
    test_df["abs_diff_longitude"] - _train_mean
) / _train_std



## === cell 41
train_df.shape



## === cell 42
test_df.shape



## === cell 43
train_df.head()



## === cell 44
from sklearn.model_selection import train_test_split

feature_cols = [c for c in train_df.columns if c not in ["key", "fare_amount"]]
train_df[feature_cols] = train_df[feature_cols].apply(pd.to_numeric, errors="coerce")
test_df[feature_cols] = test_df[feature_cols].apply(pd.to_numeric, errors="coerce")

train_df = train_df.dropna(subset=feature_cols)

X = train_df.drop(["key", "fare_amount"], axis=1)
y = train_df["fare_amount"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=80
)



## === cell 45
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)



## === cell 46
print(lr.score(X_test, y_test))



## === cell 47
X_submit = test_df.drop("key", axis=1).reindex(columns=X.columns, fill_value=0)

pred = lr.predict(X_submit)
pred = np.clip(pred, 0.0, None)
pred = np.round(pred, 2)



## === cell 48
submission = pd.DataFrame(data=pred, columns=["fare_amount"])
submission["key"] = test_df["key"]
submission = submission[["key", "fare_amount"]]



## === cell 49
submission



## === cell 50
submission.to_csv("submission.csv", index=False)

print(
    "Wrote submission.csv with rows:",
    len(submission),
    "and columns:",
    list(submission.columns),
)
print(submission.head())
