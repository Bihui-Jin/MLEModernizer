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

3.9

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

5.52625

# 6. Current score

963.32329

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 762.64707) has done: 'Your code currently won’t produce a valid Kaggle submission because you filter `test_df` rows (cell 11), which changes the required row count and drops keys; Kaggle expects predictions for every key in the original test set. I keep your feature engineering and LinearRegression pipeline intact, but remove the test-row filtering and instead only filter the training set (to avoid training on extreme outliers without breaking submission format). I also make the one-hot weekday columns consistent by fitting categories from train and reindexing test accordingly, and standardize your manual scaling to use train statistics for both train/test (same semantics, but prevents train/test mismatch that hurts RMSE). Finally, I ensure the submission filename is `submission.csv` and contains exactly all original test keys in order.'
- What this solution (achieved 963.71492) has done: 'Your current RMSE (762) indicates the model is learning from many bad training examples (invalid coordinates/fares) rather than a submission-format issue. To move toward the target (~5.5 RMSE) without changing your core model/pipeline, I only add standard NYC Taxi sanity filters on the training data (reasonable fare range, passenger_count, and NYC bounding box + nonzero trip distance) while keeping the test set completely untouched. This keeps your exact feature engineering and LinearRegression+StandardScaler approach, but removes label/feature corruption that dominates error. I also clip negative predictions to 0.0 (a minimal, metric-consistent post-process) to avoid extreme outliers hurting RMSE.'
- What this solution (achieved 963.32329) has done: 'We need to move RMSE down from ~963 toward ~5.53, so the model must stop being dominated by corrupted/out-of-distribution training rows and a subtle scaling bug. I keep your exact feature engineering and the same LinearRegression+StandardScaler pipeline, but (1) fix the manual “variance scaling” that currently divides by variance instead of standard deviation (this alone can blow up coefficients/predictions), and (2) add one more minimal, standard sanity filter on the training set to remove obviously wrong target values that survive the current filters (e.g., fare > 0 but unrealistically low given any trip). I also make the datetime parsing vectorized (same semantics, much faster) to ensure the notebook finishes within the time limit without changing the model logic.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # CSV file I/O (e.g. pd.read_csv)
import os  # reading the input files we have access to

print(os.listdir("../input"))



## === cell 1
train_df = pd.read_csv("../input/train.csv", nrows=10_000_000)
train_df.dtypes



## === cell 2
train_df.head()



## === cell 3
test_df = pd.read_csv("../input/test.csv")
test_df.dtypes



## === cell 4
test_df.head()




## === cell 5
def add_travel_vector_features(df):
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs()


add_travel_vector_features(train_df)
add_travel_vector_features(test_df)



## === cell 6
print(train_df.isnull().sum())



## === cell 7
print(test_df.isnull().sum())



## === cell 8
print("Old size: %d" % len(train_df))
train_df = train_df.dropna(how="any", axis="rows")
print("New size: %d" % len(train_df))



## === cell 9
plot = train_df.iloc[:2000].plot.scatter("abs_diff_longitude", "abs_diff_latitude")



## === cell 10
print("Old size: %d" % len(train_df))
train_df = train_df[
    (train_df.abs_diff_longitude < 5.0) & (train_df.abs_diff_latitude < 5.0)
]
print("New size: %d" % len(train_df))



## === cell 11
print("Old size (before NYC sanity filters): %d" % len(train_df))

train_df = train_df[
    (train_df["fare_amount"] > 0.0)
    & (train_df["fare_amount"] <= 500.0)
    & (train_df["passenger_count"] >= 1)
    & (train_df["passenger_count"] <= 6)
]

train_df = train_df[
    (train_df["pickup_longitude"] >= -74.3)
    & (train_df["pickup_longitude"] <= -72.9)
    & (train_df["dropoff_longitude"] >= -74.3)
    & (train_df["dropoff_longitude"] <= -72.9)
    & (train_df["pickup_latitude"] >= 40.5)
    & (train_df["pickup_latitude"] <= 41.8)
    & (train_df["dropoff_latitude"] >= 40.5)
    & (train_df["dropoff_latitude"] <= 41.8)
]

train_df = train_df[
    (train_df["abs_diff_longitude"] > 0.0) | (train_df["abs_diff_latitude"] > 0.0)
]

train_df = train_df[train_df["fare_amount"] >= 2.5]

print("New size (after NYC sanity filters): %d" % len(train_df))



## === cell 12
print("Test size (kept unchanged): %d" % len(test_df))



## === cell 13
train_df["pickup_time"] = train_df["pickup_datetime"].str.slice(11, -7)
test_df["pickup_time"] = test_df["pickup_datetime"].str.slice(11, -7)



## === cell 14
train_df.head()



## === cell 15
test_df.head()



## === cell 16
train_df["Weekday"] = pd.to_datetime(
    train_df["pickup_datetime"].str.slice(0, -4), errors="coerce"
).dt.weekday
test_df["Weekday"] = pd.to_datetime(
    test_df["pickup_datetime"].str.slice(0, -4), errors="coerce"
).dt.weekday



## === cell 17
train_df.head()



## === cell 18
test_df.head()



## === cell 19
train_df.drop("pickup_datetime", inplace=True, axis=1)
test_df.drop("pickup_datetime", inplace=True, axis=1)



## === cell 20
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



## === cell 21
train_df.head()



## === cell 22
test_df.head()



## === cell 23
train_one_hot = pd.get_dummies(train_df["Weekday"])
test_one_hot = pd.get_dummies(test_df["Weekday"]).reindex(
    columns=train_one_hot.columns, fill_value=0
)

train_df = pd.concat([train_df, train_one_hot], axis=1)
test_df = pd.concat([test_df, test_one_hot], axis=1)



## === cell 24
train_df.head()



## === cell 25
test_df.head()



## === cell 26
train_df.drop("Weekday", axis=1, inplace=True)
test_df.drop("Weekday", axis=1, inplace=True)



## === cell 27
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



## === cell 28
train_df.head()



## === cell 29
test_df.head()



## === cell 30
R = 6373.0
lat1 = np.asarray(np.radians(train_df["pickup_latitude"]))
lon1 = np.asarray(np.radians(train_df["pickup_longitude"]))
lat2 = np.asarray(np.radians(train_df["dropoff_latitude"]))
lon2 = np.asarray(np.radians(train_df["dropoff_longitude"]))

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
train_df["Distance"] = np.asarray(distance) * 0.621

lat1 = np.asarray(np.radians(test_df["pickup_latitude"]))
lon1 = np.asarray(np.radians(test_df["pickup_longitude"]))
lat2 = np.asarray(np.radians(test_df["dropoff_latitude"]))
lon2 = np.asarray(np.radians(test_df["dropoff_longitude"]))

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
test_df["Distance"] = np.asarray(distance) * 0.621



## === cell 31
R = 6373.0
lat1 = np.asarray(np.radians(train_df["pickup_latitude"]))
lon1 = np.asarray(np.radians(train_df["pickup_longitude"]))
lat2 = np.asarray(np.radians(train_df["dropoff_latitude"]))
lon2 = np.asarray(np.radians(train_df["dropoff_longitude"]))

lat3 = np.zeros(len(train_df["pickup_latitude"])) + np.radians(40.6413)
lon3 = np.zeros(len(train_df["pickup_longitude"])) + np.radians(-73.7781)
dlon_pickup = lon3 - lon1
dlat_pickup = lat3 - lat1
d_lon_dropoff = lon3 - lon2
d_lat_dropoff = lat3 - lat2

a1 = (
    np.sin(dlat_pickup / 2) ** 2
    + np.cos(lat1) * np.cos(lat3) * np.sin(dlon_pickup / 2) ** 2
)
c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
distance1 = R * c1
train_df["Pickup_Distance_airport"] = np.asarray(distance1) * 0.621

a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
train_df["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621

lat1 = np.asarray(np.radians(test_df["pickup_latitude"]))
lon1 = np.asarray(np.radians(test_df["pickup_longitude"]))
lat2 = np.asarray(np.radians(test_df["dropoff_latitude"]))
lon2 = np.asarray(np.radians(test_df["dropoff_longitude"]))

lat3 = np.zeros(len(test_df["pickup_latitude"])) + np.radians(40.6413)
lon3 = np.zeros(len(test_df["pickup_latitude"])) + np.radians(-73.7781)
dlon_pickup = lon3 - lon1
dlat_pickup = lat3 - lat1
d_lon_dropoff = lon3 - lon2
d_lat_dropoff = lat3 - lat2

a1 = (
    np.sin(dlat_pickup / 2) ** 2
    + np.cos(lat1) * np.cos(lat3) * np.sin(dlon_pickup / 2) ** 2
)
c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
distance1 = R * c1
test_df["Pickup_Distance_airport"] = np.asarray(distance1) * 0.621

a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
test_df["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621



## === cell 32
train_df["Distance"] = np.round(train_df["Distance"], 2)
train_df["Pickup_Distance_airport"] = np.round(train_df["Pickup_Distance_airport"], 2)
train_df["Dropoff_Distance_airport"] = np.round(train_df["Dropoff_Distance_airport"], 2)
test_df["Distance"] = np.round(test_df["Distance"], 2)
test_df["Pickup_Distance_airport"] = np.round(test_df["Pickup_Distance_airport"], 2)
test_df["Dropoff_Distance_airport"] = np.round(test_df["Dropoff_Distance_airport"], 2)



## === cell 33
train_df.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)
test_df.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)



## === cell 34
abs_long_mean = np.mean(np.abs(train_df["abs_diff_longitude"]))
abs_long_std = np.std(train_df["abs_diff_longitude"])
abs_long_std = abs_long_std if abs_long_std > 0 else 1.0
train_df["abs_diff_longitude"] = (
    np.abs(train_df["abs_diff_longitude"]) - abs_long_mean
) / abs_long_std
test_df["abs_diff_longitude"] = (
    np.abs(test_df["abs_diff_longitude"]) - abs_long_mean
) / abs_long_std



## === cell 35
abs_lat_mean = np.mean(np.abs(train_df["abs_diff_latitude"]))
abs_lat_std = np.std(train_df["abs_diff_latitude"])
abs_lat_std = abs_lat_std if abs_lat_std > 0 else 1.0
train_df["abs_diff_latitude"] = (
    np.abs(train_df["abs_diff_latitude"]) - abs_lat_mean
) / abs_lat_std
test_df["abs_diff_latitude"] = (
    np.abs(test_df["abs_diff_latitude"]) - abs_lat_mean
) / abs_lat_std



## === cell 36
train_df.shape



## === cell 37
test_df.shape



## === cell 38
train_df.head()



## === cell 39
from sklearn.model_selection import train_test_split

X = train_df.drop(["key", "fare_amount"], axis=1)
y = train_df["fare_amount"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.01, random_state=80
)



## === cell 40
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

lr = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("lr", LinearRegression()),
    ]
)

lr.fit(X_train, y_train)
print(lr.score(X_test, y_test))



## === cell 41
X_full = train_df.drop(["key", "fare_amount"], axis=1)
X_test_full = test_df.drop(["key"], axis=1)

X_test_full = X_test_full.reindex(columns=X_full.columns, fill_value=0)



## === cell 42
pred = lr.predict(X_test_full)
pred = np.clip(pred, 0.0, None)
pred = np.round(pred, 2)



## === cell 43
pd.read_csv("../input/sample_submission.csv").head()



## === cell 44
Submission = pd.DataFrame({"key": test_df["key"].values, "fare_amount": pred})
Submission = Submission[["key", "fare_amount"]]
Submission.head()



## === cell 45
Submission.to_csv("submission.csv", index=False)
print("Wrote submission to submission.csv with shape:", Submission.shape)
print("Expected test rows:", pd.read_csv("../input/test.csv", usecols=["key"]).shape[0])
