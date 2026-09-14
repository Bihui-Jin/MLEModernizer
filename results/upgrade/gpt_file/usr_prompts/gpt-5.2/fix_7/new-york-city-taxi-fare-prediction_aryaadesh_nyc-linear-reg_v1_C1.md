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

1187.85105

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 762.64707) has done: 'Your notebook currently can’t produce a valid Kaggle submission because you filter rows out of `test_df` (based on abs diffs), which changes the required row count/keys; that makes the submission invalid and prevents getting any score. I keep the exact same feature engineering/model, but I stop dropping test rows and instead only apply the outlier filter to training data. I also fix the scaling bug where test features are normalized using their own mean/variance (data inconsistency); instead, I normalize both train/test using training mean/variance to improve RMSE while preserving the same linear regression approach. Finally, I ensure the submission filename ends with `.csv` and the output includes all original test keys in the original order.'
- What this solution (achieved 1047.88755) has done: 'Your current RMSE (762) indicates a data/label quality issue rather than a modeling issue, so the smallest improvement is to clean the training target and coordinates while keeping your exact feature engineering + LinearRegression pipeline intact. I (1) apply standard NYC Taxi sanity filters to the training set only (valid lat/lon ranges, passenger_count, and reasonable fare_amount), which typically collapses catastrophic errors, and (2) fix the scaling bug you have: you divide by variance instead of standard deviation for the two abs-diff features, while leaving the rest of the preprocessing/model unchanged. I also keep test rows/keys untouched and ensure the submission uses the original test order and exact required columns. These changes are minimal, legitimate, and should move RMSE dramatically toward your target band without changing the core approach.'
- What this solution (achieved 984.85853) has done: 'Your RMSE is catastrophically high relative to the target, which usually means the training distribution is badly mismatched (or features are malformed), not that LinearRegression itself is “underpowered”. I keep your exact core pipeline (same engineered features, same StandardScaler+LinearRegression) but fix the biggest leakage/mismatch: the distance features are computed using an Earth radius in km and then multiplied by 0.621 (miles conversion), which makes distances ~1000× too large; using miles directly corrects the feature scale and should move RMSE dramatically toward your target. I also fix the weekday/time extraction to use vectorized datetime parsing (same semantics, avoids subtle string-slicing inconsistencies) and clip negative predictions to 0.0 to avoid huge squared-error contributions from impossible negative fares. Submission format/row order stays identical and a valid `submission.csv` is always written.'
- What this solution (achieved 1187.85105) has done: 'Your RMSE is still catastrophically far from the target, so the most likely cause is not the LinearRegression itself but that a small fraction of “bad” training rows (corrupt coordinates/timestamps or label noise) are dominating the fit and exploding test errors. I keep your exact feature set and model pipeline, but add one more minimal, standard NYC-taxi cleaning step on the training set only: remove rides with essentially-zero distance but non-trivial fare, and remove a tiny top-end tail of extreme distances/fare-per-mile that LinearRegression can’t handle well. I also ensure the engineered numeric columns are all finite after datetime parsing and distance computations (dropping only bad training rows), while leaving the test set untouched so the submission remains valid and aligned.'
- What this solution (achieved 1187.85105) has done: 'You’re failing in cell 28 because you try to `dropna(subset=["pickup_time","Weekday"])` after you already dropped the `Weekday` column in cell 22, causing a `KeyError` and preventing any submission from being written. I fix this by changing the subset to only columns that still exist, and I also apply the same non-finite/NaN guard to the test set’s engineered numeric columns (without dropping any rows) to prevent inference-time crashes/NaNs. These changes are execution/stability fixes and keep your core feature engineering + LinearRegression pipeline unchanged, while allowing the notebook to run end-to-end and write a valid `submission.csv`.'

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
    (train_df.abs_diff_longitude < 5.0)
    & (train_df.abs_diff_latitude < 5.0)
    & (train_df.pickup_longitude.between(-75, -72))
    & (train_df.dropoff_longitude.between(-75, -72))
    & (train_df.pickup_latitude.between(40, 42))
    & (train_df.dropoff_latitude.between(40, 42))
    & (train_df.passenger_count.between(1, 6))
    & (train_df.fare_amount.between(2.5, 200))
]
print("New size: %d" % len(train_df))



## === cell 11
print("Old size: %d" % len(test_df))
print("New size: %d" % len(test_df))



## === cell 12
train_dt = pd.to_datetime(train_df["pickup_datetime"], errors="coerce")
test_dt = pd.to_datetime(test_df["pickup_datetime"], errors="coerce")

train_df["pickup_time"] = train_dt.dt.hour * 100 + train_dt.dt.minute
test_df["pickup_time"] = test_dt.dt.hour * 100 + test_dt.dt.minute

train_df["Weekday"] = train_dt.dt.weekday
test_df["Weekday"] = test_dt.dt.weekday



## === cell 13
train_df.head()



## === cell 14
test_df.head()



## === cell 15
train_df.drop("pickup_datetime", inplace=True, axis=1)
test_df.drop("pickup_datetime", inplace=True, axis=1)



## === cell 16
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



## === cell 17
train_df.head()



## === cell 18
test_df.head()



## === cell 19
train_one_hot = pd.get_dummies(train_df["Weekday"])
test_one_hot = pd.get_dummies(test_df["Weekday"])
test_one_hot = test_one_hot.reindex(columns=train_one_hot.columns, fill_value=0)

train_df = pd.concat([train_df, train_one_hot], axis=1)
test_df = pd.concat([test_df, test_one_hot], axis=1)



## === cell 20
train_df.head()



## === cell 21
test_df.head()



## === cell 22
train_df.drop("Weekday", axis=1, inplace=True)
test_df.drop("Weekday", axis=1, inplace=True)



## === cell 23
train_df.head()



## === cell 24
test_df.head()



## === cell 25
R_MILES = 3958.7613

lat1 = np.asarray(np.radians(train_df["pickup_latitude"]))
lon1 = np.asarray(np.radians(train_df["pickup_longitude"]))
lat2 = np.asarray(np.radians(train_df["dropoff_latitude"]))
lon2 = np.asarray(np.radians(train_df["dropoff_longitude"]))

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R_MILES * c
train_df["Distance"] = np.asarray(distance)

lat1 = np.asarray(np.radians(test_df["pickup_latitude"]))
lon1 = np.asarray(np.radians(test_df["pickup_longitude"]))
lat2 = np.asarray(np.radians(test_df["dropoff_latitude"]))
lon2 = np.asarray(np.radians(test_df["dropoff_longitude"]))

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R_MILES * c
test_df["Distance"] = np.asarray(distance)



## === cell 26
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
distance1 = R_MILES * c1
train_df["Pickup_Distance_airport"] = np.asarray(distance1)

a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R_MILES * c2
train_df["Dropoff_Distance_airport"] = np.asarray(distance2)

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
distance1 = R_MILES * c1
test_df["Pickup_Distance_airport"] = np.asarray(distance1)

a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R_MILES * c2
test_df["Dropoff_Distance_airport"] = np.asarray(distance2)



## === cell 27
train_df["Distance"] = np.round(train_df["Distance"], 2)
train_df["Pickup_Distance_airport"] = np.round(train_df["Pickup_Distance_airport"], 2)
train_df["Dropoff_Distance_airport"] = np.round(train_df["Dropoff_Distance_airport"], 2)
test_df["Distance"] = np.round(test_df["Distance"], 2)
test_df["Pickup_Distance_airport"] = np.round(test_df["Pickup_Distance_airport"], 2)
test_df["Dropoff_Distance_airport"] = np.round(test_df["Dropoff_Distance_airport"], 2)



## === cell 28
numeric_check_cols = [
    "pickup_time",
    "Distance",
    "Pickup_Distance_airport",
    "Dropoff_Distance_airport",
]

before = len(train_df)
train_df = train_df.dropna(subset=["pickup_time"])
train_df = train_df[np.isfinite(train_df[numeric_check_cols]).all(axis=1)]
print(
    "Dropped non-finite TRAIN rows after feature engineering:", before - len(train_df)
)

for col in numeric_check_cols:
    if col in test_df.columns:
        test_df[col] = pd.to_numeric(test_df[col], errors="coerce")
test_bad = ~np.isfinite(test_df[numeric_check_cols]).all(axis=1)
if test_bad.any():
    fill_vals = train_df[numeric_check_cols].median(numeric_only=True)
    test_df.loc[test_bad, numeric_check_cols] = test_df.loc[
        test_bad, numeric_check_cols
    ].fillna(fill_vals)
    test_df[numeric_check_cols] = (
        test_df[numeric_check_cols].replace([np.inf, -np.inf], np.nan).fillna(fill_vals)
    )



## === cell 29
before = len(train_df)
train_df = train_df[~((train_df["Distance"] < 0.01) & (train_df["fare_amount"] > 5.0))]
train_df = train_df[
    train_df["Distance"] <= 50.0
]  # conservative cap; doesn't touch test
fare_per_mile = train_df["fare_amount"] / np.maximum(train_df["Distance"], 0.1)
train_df = train_df[fare_per_mile.between(0.5, 100.0)]
print("Dropped additional pathological TRAIN rows:", before - len(train_df))



## === cell 30
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



## === cell 31
train_abs_long_mean = np.mean(np.abs(train_df["abs_diff_longitude"]))
train_abs_long_std = np.std(train_df["abs_diff_longitude"])
if train_abs_long_std == 0:
    train_abs_long_std = 1.0

train_df["abs_diff_longitude"] = (
    np.abs(train_df["abs_diff_longitude"]) - train_abs_long_mean
)
train_df["abs_diff_longitude"] = train_df["abs_diff_longitude"] / train_abs_long_std

test_df["abs_diff_longitude"] = (
    np.abs(test_df["abs_diff_longitude"]) - train_abs_long_mean
)
test_df["abs_diff_longitude"] = test_df["abs_diff_longitude"] / train_abs_long_std



## === cell 32
train_abs_lat_mean = np.mean(np.abs(train_df["abs_diff_latitude"]))
train_abs_lat_std = np.std(train_df["abs_diff_latitude"])
if train_abs_lat_std == 0:
    train_abs_lat_std = 1.0

train_df["abs_diff_latitude"] = (
    np.abs(train_df["abs_diff_latitude"]) - train_abs_lat_mean
)
train_df["abs_diff_latitude"] = train_df["abs_diff_latitude"] / train_abs_lat_std

test_df["abs_diff_latitude"] = np.abs(test_df["abs_diff_latitude"]) - train_abs_lat_mean
test_df["abs_diff_latitude"] = test_df["abs_diff_latitude"] / train_abs_lat_std



## === cell 33
train_df.shape



## === cell 34
test_df.shape



## === cell 35
train_df.head()



## === cell 36
from sklearn.model_selection import train_test_split

X = train_df.drop(["key", "fare_amount"], axis=1)
y = train_df["fare_amount"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.01, random_state=80
)



## === cell 37
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

lr = make_pipeline(StandardScaler(with_mean=True, with_std=True), LinearRegression())
lr.fit(X_train, y_train)
print(lr.score(X_test, y_test))



## === cell 38
test_X = test_df.drop("key", axis=1)
test_X = test_X.reindex(columns=X.columns, fill_value=0)

pred = lr.predict(test_X)
pred = np.clip(pred, 0.0, None)
pred = np.round(pred, 2)



## === cell 39
pd.read_csv("../input/sample_submission.csv").head()



## === cell 40
Submission = pd.DataFrame({"key": test_df["key"].values, "fare_amount": pred})
Submission = Submission[["key", "fare_amount"]]



## === cell 41
Submission.to_csv("submission.csv", index=False)
print("Wrote submission to submission.csv with shape:", Submission.shape)
print(Submission.head())
