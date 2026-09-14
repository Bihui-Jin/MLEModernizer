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

5.68059

# 6. Current score

1019.32628

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 939.03804) has done: 'I fix the runtime error caused by the removed `normalize` argument in scikit-learn’s `LinearRegression` by using an equivalent `Pipeline` with `StandardScaler` (same linear regression core, just the supported way to normalize now). I also ensure the one-hot weekday columns are aligned between train and test (missing dummy columns can silently break inference or degrade accuracy). Finally, I make the submission generation robust (no `set_index` needed for Kaggle) and guarantee a valid `Submission.csv` is written to the working directory with the exact required columns.'
- What this solution (achieved 760.05194) has done: 'Your huge RMSE indicates the model is producing wildly mis-scaled predictions; the main cause is inconsistent manual “normalization” of Difference_longitude/latitude where test is centered/scaled using its own mean/variance instead of train’s, shifting the feature distribution at inference. I keep the same features and the same LinearRegression-in-a-StandardScaler pipeline, but change those two manual steps to use train statistics for both train and test (minimal semantic fix). I also add a tiny sanity post-process to clip negative fares to 0 (fares can’t be negative) which usually reduces RMSE blow-ups without changing the core model. Everything else (data size, filtering, feature engineering, model) remains the same and it still writes `Submission.csv`.'
- What this solution (achieved 869.50897) has done: 'Your RMSE is still massively off because a few extreme/out-of-distribution rows in the 10M training sample are poisoning the linear fit (even with scaling), causing huge prediction errors on the test distribution. I keep your exact feature set and the same LinearRegression+StandardScaler pipeline, but add the standard NYC Taxi Fare competition sanity filters on training rows (valid lat/lon ranges around NYC, passenger_count bounds, and fare bounds), which is a minimal data-cleaning change that typically drops RMSE by orders of magnitude. I also ensure the same filters do not touch the test set and keep your submission writing unchanged. These changes should move the score much closer to the target band without changing the model/training logic.'
- What this solution (achieved 1021.01909) has done: 'Your current RMSE is still far above the target, so we need a minimal change that materially reduces error without changing the model or feature set. The biggest remaining issue is that the training fit is being dominated by a heavy tail of long trips even within your broad NYC bounding box; this makes a linear model extrapolate badly on the test distribution. I keep the exact same features and LinearRegression+StandardScaler pipeline, but tighten the training-only sanity filters with a standard “NYC-ish” lat/lon box and a realistic maximum trip distance (based on your already-computed `Distance`), which typically collapses RMSE dramatically. I also ensure any inf/overflow rows are removed before fitting, while leaving the test set untouched and keeping submission writing identical.'
- What this solution (achieved 1019.32628) has done: 'Your current RMSE is orders of magnitude worse than the target, so we need a small change that fixes a likely “unit mismatch” bug without changing your model, loss, or overall approach. The biggest issue in your feature engineering is that your haversine distances are computed with an Earth radius in kilometers (6373) and then multiplied by 0.621 (km→miles), but you label and filter them like miles; this systematically shrinks distances and breaks the fare–distance relationship. I switch to a miles-based radius (3958.7613) and remove the extra 0.621 multiplier for both `Distance` and airport distances (same features, just correct units), keeping all filtering/LinearRegression/StandardScaler logic intact. This should materially reduce RMSE toward the target while preserving the core pipeline and still writing a valid `Submission.csv`.'
- What this solution (achieved 1019.32628) has done: 'Your current pipeline likely fails to produce a valid submission for Kaggle because you filter rows out of the test set (lat/lon bounding box + dropna), which changes the required row count and keys; Kaggle expects predictions for every row in `test.csv`. To move RMSE toward the target, I keep your exact model (StandardScaler + LinearRegression) and feature set, but apply all sanity filters only to the training data while leaving the test set intact. I also ensure any NaNs/inf in test features are imputed from training feature medians (minimal, metric-aligned stability fix) so prediction always succeeds without dropping test rows. Finally, the submission be written with exactly the original test keys and 9914 rows.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

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
train_data.isna().sum()



## === cell 7
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



## === cell 8
print(f"Before Dropping null values: {len(train_data)}")
train_data.dropna(inplace=True)
print(f"After Dropping null values: {len(train_data)}")



## === cell 9
train_data = train_data[
    (train_data["fare_amount"] > 0) & (train_data["fare_amount"] <= 200)
]
train_data = train_data[
    (train_data["passenger_count"] >= 1) & (train_data["passenger_count"] <= 6)
]

train_data = train_data[
    (train_data["pickup_longitude"].between(-74.3, -72.9))
    & (train_data["dropoff_longitude"].between(-74.3, -72.9))
    & (train_data["pickup_latitude"].between(40.5, 41.8))
    & (train_data["dropoff_latitude"].between(40.5, 41.8))
]

train_data["Difference_longitude"] = np.abs(
    np.asarray(train_data["pickup_longitude"] - train_data["dropoff_longitude"])
)
train_data["Difference_latitude"] = np.abs(
    np.asarray(train_data["pickup_latitude"] - train_data["dropoff_latitude"])
)



## === cell 10
try:
    plot = train_data[:2000].plot.scatter("Difference_longitude", "Difference_latitude")
except Exception as e:
    print("Plot skipped due to:", repr(e))



## === cell 11
train_data = train_data[
    (train_data["Difference_longitude"] < 5.0)
    & (train_data["Difference_latitude"] < 5.0)
]



## === cell 12
ls1 = list(train_data["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][11:-7:]
train_data["pickuptime"] = ls1

ls1 = list(test_data["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][11:-7:]
test_data["pickuptime"] = ls1



## === cell 13
train_data.head()



## === cell 14
ls1 = list(train_data["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][:-4:]
    ls1[i] = pd.Timestamp(ls1[i])
    ls1[i] = ls1[i].weekday()
train_data["Weekday"] = ls1

ls1 = list(test_data["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][:-4:]
    ls1[i] = pd.Timestamp(ls1[i])
    ls1[i] = ls1[i].weekday()
test_data["Weekday"] = ls1



## === cell 15
train_data.head()



## === cell 16
test_data.head()



## === cell 17
train_data.drop("pickup_datetime", inplace=True, axis=1)
test_data.drop("pickup_datetime", inplace=True, axis=1)



## === cell 18
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



## === cell 19
train_one_hot = pd.get_dummies(train_data["Weekday"])
test_one_hot = pd.get_dummies(test_data["Weekday"])
train_one_hot, test_one_hot = train_one_hot.align(
    test_one_hot, join="outer", axis=1, fill_value=0
)

train_data = pd.concat([train_data, train_one_hot], axis=1)
test_data = pd.concat([test_data, test_one_hot], axis=1)



## === cell 20
train_data.drop("Weekday", axis=1, inplace=True)
test_data.drop("Weekday", axis=1, inplace=True)



## === cell 21
ls1 = list(train_data["pickuptime"])
for i in range(len(ls1)):
    z = ls1[i].split(":")
    ls1[i] = int(z[0]) * 100 + int(z[1])
train_data["pickuptime"] = ls1

ls1 = list(test_data["pickuptime"])
for i in range(len(ls1)):
    z = ls1[i].split(":")
    ls1[i] = int(z[0]) * 100 + int(z[1])
test_data["pickuptime"] = ls1



## === cell 22
train_data.head()



## === cell 23
R = 3958.7613  # Earth radius in miles

lat1 = np.asarray(np.radians(train_data["pickup_latitude"]))
lon1 = np.asarray(np.radians(train_data["pickup_longitude"]))
lat2 = np.asarray(np.radians(train_data["dropoff_latitude"]))
lon2 = np.asarray(np.radians(train_data["dropoff_longitude"]))

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
train_data["Distance"] = np.asarray(distance)

lat1 = np.asarray(np.radians(test_data["pickup_latitude"]))
lon1 = np.asarray(np.radians(test_data["pickup_longitude"]))
lat2 = np.asarray(np.radians(test_data["dropoff_latitude"]))
lon2 = np.asarray(np.radians(test_data["dropoff_longitude"]))

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
test_data["Distance"] = np.asarray(distance)



## === cell 24
R = 3958.7613  # miles

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
train_data["Pickup_Distance_airport"] = np.asarray(distance1)

a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
train_data["Dropoff_Distance_airport"] = np.asarray(distance2)

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
test_data["Pickup_Distance_airport"] = np.asarray(distance1)

a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
test_data["Dropoff_Distance_airport"] = np.asarray(distance2)



## === cell 25
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



## === cell 26
train_data = train_data[
    (train_data["Distance"] >= 0.0) & (train_data["Distance"] <= 60.0)
]



## === cell 27
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



## === cell 28
dlon_mean = float(np.mean(train_data["Difference_longitude"]))
dlon_var = (
    float(np.var(train_data["Difference_longitude"]))
    if float(np.var(train_data["Difference_longitude"])) != 0.0
    else 1.0
)
dlat_mean = float(np.mean(train_data["Difference_latitude"]))
dlat_var = (
    float(np.var(train_data["Difference_latitude"]))
    if float(np.var(train_data["Difference_latitude"])) != 0.0
    else 1.0
)

train_data["Difference_longitude"] = np.abs(
    train_data["Difference_longitude"] - dlon_mean
)
train_data["Difference_longitude"] = train_data["Difference_longitude"] / dlon_var

train_data["Difference_latitude"] = np.abs(
    train_data["Difference_latitude"] - dlat_mean
)
train_data["Difference_latitude"] = train_data["Difference_latitude"] / dlat_var

test_data["Difference_longitude"] = np.abs(
    test_data["Difference_longitude"] - dlon_mean
)
test_data["Difference_longitude"] = test_data["Difference_longitude"] / dlon_var

test_data["Difference_latitude"] = np.abs(test_data["Difference_latitude"] - dlat_mean)
test_data["Difference_latitude"] = test_data["Difference_latitude"] / dlat_var



## === cell 29
train_data.shape



## === cell 30
test_data.shape



## === cell 31
from sklearn.model_selection import train_test_split

train_data = train_data.replace([np.inf, -np.inf], np.nan).dropna()

X = train_data.drop(["key", "fare_amount"], axis=1)
y = train_data["fare_amount"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=80
)

test_key = test_data["key"].copy()
X_test_submit = test_data.drop("key", axis=1).replace([np.inf, -np.inf], np.nan)

train_feature_medians = X.median(numeric_only=True)
X_test_submit = X_test_submit.fillna(train_feature_medians)



## === cell 32
X_train.shape



## === cell 33
X_test.shape



## === cell 34
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

lr = Pipeline(
    [
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", LinearRegression()),
    ]
)
lr.fit(X_train, y_train)
print(lr.score(X_test, y_test))



## === cell 35
pred = lr.predict(X_test_submit)
pred = np.clip(pred, 0, None)
pred = np.round(pred, 2)



## === cell 36
pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
).head()



## === cell 37
Submission = pd.DataFrame({"key": test_key.values, "fare_amount": pred})
Submission = Submission[["key", "fare_amount"]]
Submission.head()



## === cell 38
Submission.to_csv("Submission.csv", index=False)
print("Wrote Submission.csv with shape:", Submission.shape)
print(Submission.head())
print("Unique keys in submission:", Submission["key"].nunique())
