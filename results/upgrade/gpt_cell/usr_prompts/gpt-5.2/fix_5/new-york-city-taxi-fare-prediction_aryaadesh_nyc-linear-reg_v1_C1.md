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

6.14494

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 762.61302) has done: 'I make the pipeline always produce a valid submission with the full 9,914 test keys by *not filtering out any test rows* (filtering test changes row count and breaks the required submission alignment). I also fix feature standardization leakage/shift by computing scaling statistics on the training data once and applying them to both train and test (your current code standardizes test using its own mean/var, causing train/test mismatch and worse RMSE). Finally, I align one-hot weekday columns between train and test so the model sees identical feature columns at prediction time, preventing silent column mismatches or dropped categories. These are minimal changes that preserve your model and overall feature set while improving score stability toward the target.'
- What this solution (achieved 993.30022) has done: 'Your current RMSE (762) is dominated by a small number of pathological training rows (bad coordinates, impossible passenger counts, and extreme fares) that make a linear regression generalize terribly. To move toward the target score with minimal changes and identical core model/training, I add standard, competition-safe training-only data cleaning (coordinate bounding box around NYC, valid passenger_count, and a reasonable fare_amount range) and keep the rest of your pipeline intact. I also clip negative predictions to 0 (fares can’t be negative), which reduces large-error outliers without changing the learning approach. These changes are small, fast, and should materially reduce RMSE while preserving your feature engineering and LinearRegression fit/predict flow.'
- What this solution (achieved 6.14494) has done: 'Your current RMSE is far from the target, so we should make small but high-impact fixes that keep your LinearRegression + existing feature set intact. The biggest issue is that you’re dividing by variance (not standard deviation) when “standardizing” abs diffs, which badly distorts feature scale and can explode errors; switching to std is a minimal correction that preserves semantics. Next, add the same NYC coordinate / passenger bounds cleaning to the test set only by clipping (not filtering) so the submission keeps all 9,914 rows but avoids extreme distances that create huge errors. Finally, train on all cleaned training rows (not a 1% holdout) to improve generalization without changing the model type or training approach.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

INPUT_DIR = "../input"

print(os.listdir(INPUT_DIR))



## === cell 1
train_df = pd.read_csv(f"{INPUT_DIR}/train.csv", nrows=10_000_000)
train_df.dtypes



## === cell 2
train_df.head()



## === cell 3
test_df = pd.read_csv(f"{INPUT_DIR}/test.csv")
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
print("Old size (before NYC/outlier cleaning): %d" % len(train_df))

coord_mask = (
    (train_df["pickup_longitude"].between(-74.5, -72.5))
    & (train_df["dropoff_longitude"].between(-74.5, -72.5))
    & (train_df["pickup_latitude"].between(40.0, 41.8))
    & (train_df["dropoff_latitude"].between(40.0, 41.8))
)

passenger_mask = train_df["passenger_count"].between(1, 6)

fare_mask = train_df["fare_amount"].between(2.5, 250.0)

train_df = train_df[coord_mask & passenger_mask & fare_mask].copy()
print("New size (after NYC/outlier cleaning): %d" % len(train_df))



## === cell 12
test_df["pickup_longitude"] = test_df["pickup_longitude"].clip(-74.5, -72.5)
test_df["dropoff_longitude"] = test_df["dropoff_longitude"].clip(-74.5, -72.5)
test_df["pickup_latitude"] = test_df["pickup_latitude"].clip(40.0, 41.8)
test_df["dropoff_latitude"] = test_df["dropoff_latitude"].clip(40.0, 41.8)
test_df["passenger_count"] = test_df["passenger_count"].clip(1, 6)

print("Test size kept (no filtering applied): %d" % len(test_df))



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
test_one_hot = pd.get_dummies(test_df["Weekday"])

all_cols = sorted(set(train_one_hot.columns).union(set(test_one_hot.columns)))
train_one_hot = train_one_hot.reindex(columns=all_cols, fill_value=0)
test_one_hot = test_one_hot.reindex(columns=all_cols, fill_value=0)

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
train_df["pickup_time"] = train_df["pickup_time"].str.split(":", expand=True)[0].astype(
    int
) * 100 + train_df["pickup_time"].str.split(":", expand=True)[1].astype(int)
test_df["pickup_time"] = test_df["pickup_time"].str.split(":", expand=True)[0].astype(
    int
) * 100 + test_df["pickup_time"].str.split(":", expand=True)[1].astype(int)



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
abs_lon_mean = float(np.mean(np.abs(train_df["abs_diff_longitude"])))
abs_lon_std = float(np.std(np.abs(train_df["abs_diff_longitude"])))
if abs_lon_std == 0.0:
    abs_lon_std = 1.0

train_df["abs_diff_longitude"] = (
    np.abs(train_df["abs_diff_longitude"]) - abs_lon_mean
) / abs_lon_std
test_df["abs_diff_longitude"] = (
    np.abs(test_df["abs_diff_longitude"]) - abs_lon_mean
) / abs_lon_std



## === cell 35
abs_lat_mean = float(np.mean(np.abs(train_df["abs_diff_latitude"])))
abs_lat_std = float(np.std(np.abs(train_df["abs_diff_latitude"])))
if abs_lat_std == 0.0:
    abs_lat_std = 1.0

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

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.01, random_state=80)



## === cell 40
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
print(lr.score(X_val, y_val))

lr.fit(X, y)



## === cell 41
X_test_submit = test_df.drop("key", axis=1).reindex(columns=X.columns, fill_value=0)

pred = lr.predict(X_test_submit)
pred = np.clip(pred, 0.0, None)
pred = np.round(pred, 2)



## === cell 42
pd.read_csv(f"{INPUT_DIR}/sample_submission.csv").head()



## === cell 43
Submission = pd.DataFrame({"key": test_df["key"].values, "fare_amount": pred})
Submission = Submission[["key", "fare_amount"]]



## === cell 44
Submission.head()



## === cell 45
submission_path = "Submission.csv"
Submission.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print("Submission shape:", Submission.shape)
print(Submission.head())
