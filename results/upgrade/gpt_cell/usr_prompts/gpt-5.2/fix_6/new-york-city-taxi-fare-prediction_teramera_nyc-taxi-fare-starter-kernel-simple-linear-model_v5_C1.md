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

3.8

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

5.45383

# 6. Current score

986.696

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 752.62216) has done: 'Diagnosis: The crash happens because scikit-learn 1.2.2 removed the `normalize` parameter from `sklearn.linear_model.LinearRegression`, so passing `normalize=True` raises `TypeError`. The rest of the cell (fit/score) is fine once the estimator is constructed correctly. We should remove the unsupported argument while keeping the same LinearRegression model and downstream variable names.

Patch summary: In cell 41, instantiate `LinearRegression()` without `normalize=True` and keep the `lr` variable plus the subsequent `fit` and `score` calls unchanged.

Updated cells: Only cell 41 is modified.

Compatibility notes for cell k+1: Cell 42 expects `lr` to exist and be a fitted `LinearRegression` object; this remains true with the patch.

Assumptions: No additional preprocessing (e.g., external scaling) is required for execution; the goal is to fix the API incompatibility without changing the pipeline structure beyond removing the invalid parameter.'
- What this solution (achieved 986.59596) has done: 'Your RMSE is extremely high because the training data still contains many invalid/outlier trips (e.g., fares <= 0, passenger_count outside a reasonable range, and coordinates far outside NYC), which makes a linear model fit poorly and then generalize badly. With minimal changes and without altering the core model/training loop, I add a small, standard set of sanity filters on the training set only (keep test untouched) to remove the most harmful rows and bring the score much closer to your target. I also ensure train/test have identical one-hot weekday columns (reindex) to prevent silent feature misalignment that can seriously degrade predictions. Everything else (feature engineering + LinearRegression + submission writing) stays the same.'
- What this solution (achieved 986.59596) has done: 'Diagnosis: Cell 24 fails with `KeyError: "['weekday'] not found in axis"` because the prior alignment/reindexing step in cell 23 already removed the `weekday` column from `train_df` (it reindexed to all columns except `"weekday"`). Therefore attempting to drop it again errors in pandas when the column is absent.  
Patch summary: Make the drop operation tolerant to the column already being removed by using `errors="ignore"` so execution continues deterministically without changing the dataset content.  
Updated cells: Only cell 24 is changed.  
Compatibility notes for cell k+1: `train_df` remains unchanged (it already had no `weekday` column), and cell 25 can still run as-is.  
Assumptions: It is acceptable to treat `weekday` as optional at this point because earlier code may already have removed it, and downstream expects it not to exist.'
- What this solution (achieved 986.59596) has done: 'Diagnosis: Cell 25 crashes because `test_df` no longer has a `weekday` column at that point (it was already removed in cell 23 via `reindex(columns=[col for col in test_df.columns if col != "weekday"] ...)`). Calling `test_df.drop("weekday", ...)` without `errors="ignore"` raises a `KeyError`. The analogous operation in cell 24 already uses `errors="ignore"`, indicating the intent is to drop the column if present.

Patch summary: Make the drop in cell 25 tolerant to the column already being absent by adding `errors="ignore"`. This is the smallest change that preserves identical semantics when the column exists and prevents the crash when it does not.

Updated cells: Only cell 25 is modified.

Compatibility notes for cell k+1: `test_df` remains unchanged in all cases where `weekday` is already absent (the current situation), so downstream cells continue to see the same `test_df` schema and data.

Assumptions: The intended behavior is “drop `weekday` if it exists”, consistent with cell 24 and the earlier `reindex` step that may remove it.'
- What this solution (achieved 986.696) has done: 'Your current RMSE is far from the target, so the smallest score-improving change is to fix a feature bug that makes the model effectively lose location information: you are mean-centering `abs_diff_longitude/latitude` separately on train and test, which shifts test features into a different space than what the model learned. I keep the same feature set and LinearRegression training, but compute those means on the training data once and apply the same centering to both train and test. This is a minimal, metric-aligned fix that typically reduces RMSE substantially without changing the model/loop. The rest of the pipeline (filters, one-hot alignment, distance feature, submission writing) remains unchanged and still output a valid `submission.csv`.'

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
def add_travel_vector_features(df):
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs()


add_travel_vector_features(train_df)



## === cell 3
print(train_df.isnull().sum())



## === cell 4
print("Old size: %d" % len(train_df))
train_df = train_df.dropna(how="any", axis="rows")
print("New size: %d" % len(train_df))



## === cell 5
plot = train_df.iloc[:2000].plot.scatter("abs_diff_longitude", "abs_diff_latitude")



## === cell 6
print("Old size: %d" % len(train_df))
train_df = train_df[
    (train_df.abs_diff_longitude < 5.0) & (train_df.abs_diff_latitude < 5.0)
]
print("New size: %d" % len(train_df))



## === cell 7
print("Old size (pre-sanity-filters): %d" % len(train_df))
train_df = train_df[
    (train_df.fare_amount > 0)
    & (train_df.fare_amount <= 250)
    & (train_df.passenger_count >= 1)
    & (train_df.passenger_count <= 6)
    & (train_df.pickup_longitude.between(-74.5, -72.8))
    & (train_df.dropoff_longitude.between(-74.5, -72.8))
    & (train_df.pickup_latitude.between(40.5, 41.8))
    & (train_df.dropoff_latitude.between(40.5, 41.8))
].copy()
print("New size (post-sanity-filters): %d" % len(train_df))



## === cell 8
ls1 = list(train_df["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][11:-7:]
train_df["pickup_time"] = ls1



## === cell 9
train_df["pickup_time"].head(5)



## === cell 10
test_df = pd.read_csv("../input/test.csv")
test_df.head()



## === cell 11
add_travel_vector_features(test_df)



## === cell 12
test_df.shape



## === cell 13
ls1 = list(test_df["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][11:-7:]
test_df["pickup_time"] = ls1



## === cell 14
ls1 = list(test_df["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][:-4:]
    ls1[i] = pd.Timestamp(ls1[i])
    ls1[i] = ls1[i].weekday()
test_df["weekday"] = ls1



## === cell 15
ls1 = list(train_df["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][:-4:]
    ls1[i] = pd.Timestamp(ls1[i])
    ls1[i] = ls1[i].weekday()
train_df["weekday"] = ls1



## === cell 16
train_df.shape



## === cell 17
train_df.drop("pickup_datetime", inplace=True, axis=1)



## === cell 18
test_df.drop("pickup_datetime", inplace=True, axis=1)



## === cell 19
train_df["weekday"].replace(
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



## === cell 20
test_df["weekday"].replace(
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
train_one_hot = pd.get_dummies(train_df["weekday"])
train_df = pd.concat([train_df, train_one_hot], axis=1)



## === cell 22
test_one_hot = pd.get_dummies(test_df["weekday"])
test_df = pd.concat([test_df, test_one_hot], axis=1)



## === cell 23
weekday_cols = sorted(set(train_one_hot.columns).union(set(test_one_hot.columns)))
for c in weekday_cols:
    if c not in train_df.columns:
        train_df[c] = 0
    if c not in test_df.columns:
        test_df[c] = 0
train_df = train_df.reindex(
    columns=[col for col in train_df.columns if col != "weekday"] + [], copy=False
)
test_df = test_df.reindex(
    columns=[col for col in test_df.columns if col != "weekday"] + [], copy=False
)



## === cell 24
train_df.drop("weekday", inplace=True, axis=1, errors="ignore")



## === cell 25
test_df.drop("weekday", inplace=True, axis=1, errors="ignore")



## === cell 26
ls1 = list(train_df["pickup_time"])
for i in range(len(ls1)):
    z = ls1[i].split(":")
    ls1[i] = int(z[0]) * 100 + int(z[1])
train_df["pickup_time"] = ls1



## === cell 27
ls1 = list(test_df["pickup_time"])
for i in range(len(ls1)):
    z = ls1[i].split(":")
    ls1[i] = int(z[0]) * 100 + int(z[1])
test_df["pickup_time"] = ls1



## === cell 28
train_df.shape



## === cell 29
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



## === cell 30
R = 6373.0
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
train_df["Distance"] = np.round(train_df["Distance"], 2)
test_df["Distance"] = np.round(test_df["Distance"], 2)



## === cell 32
train_df.shape



## === cell 33
abs_diff_longitude_mean_ = float(train_df["abs_diff_longitude"].mean())
abs_diff_latitude_mean_ = float(train_df["abs_diff_latitude"].mean())

train_df["abs_diff_longitude"] = np.abs(
    train_df["abs_diff_longitude"] - abs_diff_longitude_mean_
)
train_df["abs_diff_latitude"] = np.abs(
    train_df["abs_diff_latitude"] - abs_diff_latitude_mean_
)



## === cell 34
test_df["abs_diff_longitude"] = np.abs(
    test_df["abs_diff_longitude"] - abs_diff_longitude_mean_
)
test_df["abs_diff_latitude"] = np.abs(
    test_df["abs_diff_latitude"] - abs_diff_latitude_mean_
)



## === cell 35
train_df.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    inplace=True,
    axis=1,
)



## === cell 36
test_df.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    inplace=True,
    axis=1,
)



## === cell 37
train_df.head()



## === cell 38
test_df.shape



## === cell 39
from sklearn.model_selection import train_test_split



## === cell 40
X = train_df.drop(["key", "fare_amount"], axis=1)
y = train_df["fare_amount"]



## === cell 41
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.01, random_state=80
)



## === cell 42
X_train.shape



## === cell 43
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
print(lr.score(X_test, y_test))



## === cell 44
pred = np.round(lr.predict(test_df.drop(["key"], axis=1)), 2)



## === cell 45
submission = pd.DataFrame(data=pred, columns=["fare_amount"])



## === cell 46
submission["key"] = test_df["key"]



## === cell 47
submission.set_index("key", inplace=True)



## === cell 48
submission.to_csv("submission.csv")



## === cell 49
submission.head()
