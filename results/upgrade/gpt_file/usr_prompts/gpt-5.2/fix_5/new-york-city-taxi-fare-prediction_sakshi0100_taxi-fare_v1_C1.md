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

5.68932

# 6. Current score

864.32864

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 936.93312) has done: 'I fix the runtime error caused by the removed `normalize` parameter in scikit-learn’s `LinearRegression` by replacing it with an equivalent `Pipeline(StandardScaler(with_mean=False), LinearRegression())`, which preserves the intended normalization behavior without changing the core linear model approach. I also address a likely training-time crash due to the huge `for` loops over 10M rows by switching the datetime feature extraction to vectorized pandas operations (same semantics, but feasible within the time limit). Finally, I make sure train/test one-hot columns are aligned and the submission is written as a valid `Submission.csv` with columns exactly `key,fare_amount`.'
- What this solution (achieved 752.76918) has done: 'Your huge RMSE is primarily coming from inconsistent/incorrect scaling and column alignment: the “manual normalization” for Difference_* is fit separately on train vs test (and uses test’s own mean/var), and the weekday alignment code builds `df = df[fixed]` without actually reassigning back to the original dataframes, so train/test column order can silently differ. I (1) compute Difference_* normalization stats on the training data and apply them to both train and test, (2) fix the weekday column alignment by explicitly reindexing both dataframes to the same ordered set of columns, and (3) ensure the final feature matrix for prediction matches the exact training feature columns. These are minimal changes that preserve your linear regression approach and feature set, but should drastically reduce the RMSE toward the target.'
- What this solution (achieved 864.32864) has done: 'Your current RMSE is massively worse than the target, so we need a small but decisive correction rather than tuning: the linear model is being trained on unfiltered/invalid targets (negative/zero fares and extreme outliers) and on rows with bad coordinates, which makes predictions explode and ruins RMSE. I add the standard NYC Taxi fare competition sanity filters (fare range, passenger_count range, and lat/lon bounding box) while keeping your exact feature set and the same LinearRegression-in-a-StandardScaler pipeline. I also clip negative predictions to 0 (a valid fare can’t be negative), which typically reduces RMSE without changing the model/training logic. Everything else (feature engineering, one-hot weekday, distance features, pipeline, and submission format) remains the same.'
- What this solution (achieved 864.32864) has done: 'Your RMSE is still extremely high versus the target, which strongly suggests a remaining correctness issue rather than “model quality”. The smallest high-impact fix is to ensure train/test feature columns are aligned in *exactly the same order* after weekday one-hot creation (your current code adds missing columns but never reorders the full dataframe to a common column order). I add a single, explicit reindex of both `train_data` and `test_data` to the same `all_feature_cols` right after dummy creation, and I also guard against any non-finite feature values (NaN/inf) leaking into the linear regression, which can destabilize predictions and RMSE. Everything else (features, LinearRegression + StandardScaler pipeline, filters, clipping, and submission schema) stays the same.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
train_data = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", nrows=10_000_000
)
train_data.head()



## === cell 2
train_data.shape



## === cell 3
train_data.info()



## === cell 4
test_data = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")
test_data.head()



## === cell 5
test_data.info()



## === cell 6
train_data["Difference_longitude"] = np.abs(
    np.asarray(train_data["pickup_longitude"] - train_data["dropoff_longitude"])
)
train_data["Difference_latitude"] = np.abs(
    np.asarray(train_data["pickup_latitude"] - train_data["dropoff_latitude"])
)

test_data["Difference_longitude"] = np.abs(
    np.asarray(test_data["pickup_longitude"] - test_data["dropoff_longitude"])
)
test_data["Difference_latitude"] = np.abs(
    np.asarray(test_data["pickup_latitude"] - test_data["dropoff_latitude"])
)



## === cell 7
print(f"Before Dropping null values: {len(train_data)}")
train_data.dropna(inplace=True)
print(f"After Dropping null values: {len(train_data)}")



## === cell 8
try:
    plot = train_data[:2000].plot.scatter("Difference_longitude", "Difference_latitude")
except Exception as e:
    print("Plot skipped:", repr(e))



## === cell 9
train_data = train_data[
    (train_data["Difference_longitude"] < 5.0)
    & (train_data["Difference_latitude"] < 5.0)
]



## === cell 10
fare_min, fare_max = 2.5, 250.0
train_data = train_data[
    (train_data["fare_amount"] >= fare_min) & (train_data["fare_amount"] <= fare_max)
]

train_data = train_data[
    (train_data["passenger_count"] >= 1) & (train_data["passenger_count"] <= 6)
]

train_data = train_data[
    (train_data["pickup_longitude"].between(-74.3, -72.7))
    & (train_data["dropoff_longitude"].between(-74.3, -72.7))
    & (train_data["pickup_latitude"].between(40.5, 41.8))
    & (train_data["dropoff_latitude"].between(40.5, 41.8))
]

train_data.reset_index(drop=True, inplace=True)
train_data.shape



## === cell 11
train_data["pickuptime"] = train_data["pickup_datetime"].str.slice(11, 16)
test_data["pickuptime"] = test_data["pickup_datetime"].str.slice(11, 16)



## === cell 12
train_data.head()



## === cell 13
train_dt = pd.to_datetime(
    train_data["pickup_datetime"].str.slice(0, 19), errors="coerce"
)
test_dt = pd.to_datetime(test_data["pickup_datetime"].str.slice(0, 19), errors="coerce")

train_data["Weekday"] = train_dt.dt.weekday
test_data["Weekday"] = test_dt.dt.weekday
test_data.head()



## === cell 14
train_data.drop("pickup_datetime", inplace=True, axis=1)
test_data.drop("pickup_datetime", inplace=True, axis=1)



## === cell 15
train_data["Weekday"].replace(
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
test_data["Weekday"].replace(
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



## === cell 16
train_one_hot = pd.get_dummies(train_data["Weekday"])
test_one_hot = pd.get_dummies(test_data["Weekday"])

train_data = pd.concat([train_data, train_one_hot], axis=1)
test_data = pd.concat([test_data, test_one_hot], axis=1)



## === cell 17
train_data.drop("Weekday", axis=1, inplace=True)
test_data.drop("Weekday", axis=1, inplace=True)



## === cell 18
weekday_cols = sorted(list(set(train_one_hot.columns).union(set(test_one_hot.columns))))
for col in weekday_cols:
    if col not in train_data.columns:
        train_data[col] = 0
    if col not in test_data.columns:
        test_data[col] = 0

non_target_cols_train = [c for c in train_data.columns if c != "fare_amount"]
all_cols = sorted(list(set(non_target_cols_train).union(set(test_data.columns))))
train_data = train_data.reindex(
    columns=(["fare_amount"] + [c for c in all_cols if c != "fare_amount"])
)
test_data = test_data.reindex(columns=[c for c in all_cols if c != "fare_amount"])



## === cell 19
hhmm_train = train_data["pickuptime"].str.split(":", expand=True)
train_data["pickuptime"] = hhmm_train[0].astype("int32") * 100 + hhmm_train[1].astype(
    "int32"
)

hhmm_test = test_data["pickuptime"].str.split(":", expand=True)
test_data["pickuptime"] = hhmm_test[0].astype("int32") * 100 + hhmm_test[1].astype(
    "int32"
)



## === cell 20
train_data.head()



## === cell 21
R = 6373.0
lat1 = np.asarray(np.radians(train_data["pickup_latitude"]))
lon1 = np.asarray(np.radians(train_data["pickup_longitude"]))
lat2 = np.asarray(np.radians(train_data["dropoff_latitude"]))
lon2 = np.asarray(np.radians(train_data["dropoff_longitude"]))

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
train_data["Distance"] = np.asarray(distance) * 0.621

lat1 = np.asarray(np.radians(test_data["pickup_latitude"]))
lon1 = np.asarray(np.radians(test_data["pickup_longitude"]))
lat2 = np.asarray(np.radians(test_data["dropoff_latitude"]))
lon2 = np.asarray(np.radians(test_data["dropoff_longitude"]))

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
test_data["Distance"] = np.asarray(distance) * 0.621



## === cell 22
R = 6373.0
lat1 = np.asarray(np.radians(train_data["pickup_latitude"]))
lon1 = np.asarray(np.radians(train_data["pickup_longitude"]))
lat2 = np.asarray(np.radians(train_data["dropoff_latitude"]))
lon2 = np.asarray(np.radians(train_data["dropoff_longitude"]))

lat3 = np.zeros(len(train_data)) + np.radians(40.6413111)
lon3 = np.zeros(len(train_data)) + np.radians(-73.7781391)

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
train_data["Pickup_Distance_airport"] = np.asarray(distance1) * 0.621

a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
train_data["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621

lat1 = np.asarray(np.radians(test_data["pickup_latitude"]))
lon1 = np.asarray(np.radians(test_data["pickup_longitude"]))
lat2 = np.asarray(np.radians(test_data["dropoff_latitude"]))
lon2 = np.asarray(np.radians(test_data["dropoff_longitude"]))

lat3 = np.zeros(len(test_data)) + np.radians(40.6413111)
lon3 = np.zeros(len(test_data)) + np.radians(-73.7781391)

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
test_data["Pickup_Distance_airport"] = np.asarray(distance1) * 0.621

a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
test_data["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621



## === cell 23
train_data["Distance"] = np.round(train_data["Distance"], 2)
train_data["Pickup_Distance_airport"] = np.round(
    train_data["Pickup_Distance_airport"], 2
)
train_data["Dropoff_Distance_airport"] = np.round(
    train_data["Dropoff_Distance_airport"], 2
)

test_data["Distance"] = np.round(test_data["Distance"], 2)
test_data["Pickup_Distance_airport"] = np.round(test_data["Pickup_Distance_airport"], 2)
test_data["Dropoff_Distance_airport"] = np.round(
    test_data["Dropoff_Distance_airport"], 2
)



## === cell 24
train_data.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)
test_data.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)



## === cell 25
eps = 1e-12
dl_mean = float(train_data["Difference_longitude"].mean())
dl_var = float(train_data["Difference_longitude"].var())
dlat_mean = float(train_data["Difference_latitude"].mean())
dlat_var = float(train_data["Difference_latitude"].var())

train_data["Difference_longitude"] = np.abs(
    train_data["Difference_longitude"] - dl_mean
) / (dl_var + eps)
train_data["Difference_latitude"] = np.abs(
    train_data["Difference_latitude"] - dlat_mean
) / (dlat_var + eps)

test_data["Difference_longitude"] = np.abs(
    test_data["Difference_longitude"] - dl_mean
) / (dl_var + eps)
test_data["Difference_latitude"] = np.abs(
    test_data["Difference_latitude"] - dlat_mean
) / (dlat_var + eps)

train_data.replace([np.inf, -np.inf], np.nan, inplace=True)
test_data.replace([np.inf, -np.inf], np.nan, inplace=True)
train_data.dropna(inplace=True)
test_data.fillna(0.0, inplace=True)



## === cell 26
train_data.shape



## === cell 27
test_data.shape



## === cell 28
from sklearn.model_selection import train_test_split

X = train_data.drop(["key", "fare_amount"], axis=1)
y = train_data["fare_amount"]

feature_cols = sorted(X.columns.tolist())
X = X.reindex(columns=feature_cols)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.01, random_state=80
)



## === cell 29
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

lr = Pipeline(
    steps=[
        (
            "scaler",
            StandardScaler(with_mean=False),
        ),
        ("model", LinearRegression()),
    ]
)

lr.fit(X_train, y_train)
print(lr.score(X_test, y_test))



## === cell 30
X_submit = test_data.drop("key", axis=1).reindex(columns=feature_cols, fill_value=0)

pred = lr.predict(X_submit)
pred = np.clip(pred, 0.0, None)
pred = np.round(pred, 2)



## === cell 31
pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
).head()



## === cell 32
Submission = pd.DataFrame(data=pred, columns=["fare_amount"])
Submission["key"] = test_data["key"]
Submission = Submission[["key", "fare_amount"]]



## === cell 33
Submission.to_csv("Submission.csv", index=False)
print("Wrote Submission.csv with shape:", Submission.shape)
print(Submission.head())
