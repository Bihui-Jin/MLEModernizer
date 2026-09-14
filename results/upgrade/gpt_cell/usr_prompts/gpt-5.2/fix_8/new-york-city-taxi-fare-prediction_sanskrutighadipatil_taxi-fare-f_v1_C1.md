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

5.57615

# 6. Current score

989.43731

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 762.58297) has done: 'Diagnosis: The crash happens because scikit-learn 1.2 removed the `normalize` parameter from `sklearn.linear_model.LinearRegression`, so passing `normalize=True` raises a `TypeError`. The rest of the cell (fit/score) is fine and should remain unchanged.  
Patch summary: Remove the unsupported `normalize=True` argument and instantiate `LinearRegression()` with default parameters to restore compatibility with scikit-learn 1.2.  
Updated cells: Only cell 33 is modified.  
Compatibility notes for cell k+1: The variable `lr` remains a fitted `LinearRegression` model with the same `.predict()` interface, so cell 34 work unchanged.  
Assumptions: No explicit feature scaling/normalization pipeline is required elsewhere; we only fix the API incompatibility to unblock execution while preserving the same overall modeling approach.'
- What this solution (achieved 1048.26773) has done: 'Your current RMSE (762) is far worse than the target (5.576), which usually happens when predictions are badly mis-scaled or the submission rows/keys are misaligned. I make minimal changes that keep your model/feature logic intact but fix two likely root causes: (1) ensure train/test one-hot weekday columns are aligned to the same set and order, and (2) add basic, standard NYC Taxi Fare sanity filtering on the training target and coordinates so the linear regression isn’t dominated by extreme outliers. I also ensure the submission file matches Kaggle’s expected format (keep `key` as a column, don’t set it as index) and write to a lowercase `submission.csv`. These are small, safe changes that typically drop RMSE dramatically without changing your core approach.'
- What this solution (achieved 988.5211) has done: 'Your RMSE is extremely worse than the target, which most commonly happens here due to a train/test feature mismatch or pathological numeric values leaking into the fit/prediction. I keep your exact feature set and LinearRegression approach, but (1) rebuild `train_X/train_y` after *all* filtering/feature engineering (your early `train_y` currently comes from the pre-filtered frame and is inconsistent), (2) enforce numeric dtypes and impute any remaining NaNs/Infs consistently between train/test right before fitting, and (3) clip negative predictions to 0 (fares can’t be negative), which usually improves RMSE substantially without changing the model. These are minimal changes that directly target bad-scale predictions while preserving your overall pipeline and submission semantics.'
- What this solution (achieved 989.46913) has done: 'Your huge RMSE is consistent with a single feature-engineering bug that injects wrong airport-distance values into the test set: in cell 24 the Haversine “c2” computation uses `sqrt(1 - a1)` instead of `sqrt(1 - a2)` only for the test dropoff-airport distance, which can distort predictions massively. I fix that one-line typo while keeping your exact model/feature set/training flow the same, and I add a tiny numeric safety clip to keep the Haversine “a” terms within \[0,1\] to prevent any rare NaNs from floating-point drift. Everything else (LinearRegression, filters, one-hot alignment, submission format/path) remains unchanged so the score should move sharply down toward the target.'
- What this solution (achieved 989.46913) has done: 'Your RMSE is still catastrophically high, which is most consistent with the model being trained on a stale/incorrect target array: you build `train_y` in cell 9 before later filtering (cell 11) and heavy feature engineering, so the model later fits `y` from the filtered frame but some earlier steps (and diagnostics) still operate on mismatched arrays. I make a minimal, core-logic-preserving fix by rebuilding the simple `train_X/train_y` *after* all filtering/feature creation (so everything is aligned), and I also ensure the final `X`/`test_X` columns are strictly aligned and numeric (guarding against any silent object columns). These changes don’t alter your model choice (LinearRegression), features, or training approach; they only prevent misalignment/NaN-driven blowups that inflate RMSE. The script still write a valid `submission.csv` with exactly `key,fare_amount`.'
- What this solution (achieved 989.43731) has done: 'Your RMSE is still far from the target, so the most likely remaining issue is that the model is being trained on raw features with very different scales (e.g., distances vs one-hot weekday vs time), which can make `LinearRegression` numerically unstable and yield wildly bad predictions. I keep your exact feature set and `LinearRegression` core logic, but add a `StandardScaler` fit on the training features and apply it to both validation and test features to stabilize coefficients and bring predictions back to a realistic scale. I also ensure `X_train`, `X_test`, and `test_X` are all aligned *before* scaling and that scaling happens after your existing NaN/Inf handling so the submission remains valid. This is a minimal, metric-aligned change that should move RMSE sharply down toward the target band without changing architecture or loss.'
- What this solution (achieved 989.43731) has done: 'Your RMSE is still catastrophically far from the target, which strongly suggests a basic feature/target mismatch rather than “model quality.” The smallest fix that preserves your core LinearRegression approach is to ensure you train on the *same feature matrix you later scale and predict with*: right now `get_input_matrix()` only uses `abs_diff_longitude/abs_diff_latitude`, while your actual model uses the full engineered dataframe columns, and those two pipelines are inconsistent. I (1) update `get_input_matrix()` to return the full engineered feature matrix (same columns as used for training/prediction) and (2) rebuild `train_X/train_y` only after all filtering/feature engineering so arrays are aligned; everything else (filters, feature engineering, StandardScaler + LinearRegression, submission format) remains the same. This should move the score sharply down toward the target band without changing the modeling family or training semantics.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

print(os.listdir("../input"))


## === cell 1
train_df = pd.read_csv("../input/train.csv", nrows=10_000_000)
train_df.dtypes


## === cell 2
test_df = pd.read_csv("../input/test.csv")
test_df.dtypes




## === cell 3
def add_travel_vector_features(df):
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs()


add_travel_vector_features(train_df)
add_travel_vector_features(test_df)


## === cell 4
test_df.head()


## === cell 5
print(train_df.isnull().sum())


## === cell 6
print("Old size: %d" % len(train_df))
train_df = train_df.dropna(how="any", axis="rows")
print("New size: %d" % len(train_df))


## === cell 7
plot = train_df.iloc[:2000].plot.scatter("abs_diff_longitude", "abs_diff_latitude")


## === cell 8
print("Old size: %d" % len(train_df))
train_df = train_df[
    (train_df.abs_diff_longitude < 5.0) & (train_df.abs_diff_latitude < 5.0)
]
print("New size: %d" % len(train_df))




## === cell 9
def get_input_matrix(df):
    feature_cols = [c for c in df.columns if c not in ["key", "fare_amount"]]
    X = df[feature_cols].to_numpy()
    return X


train_X = get_input_matrix(train_df)
train_y = np.array(train_df["fare_amount"])

print(train_X.shape)
print(train_y.shape)


## === cell 10
train_df.head()


## === cell 11
print("Old size (before sanity filter): %d" % len(train_df))
train_df = train_df[
    (train_df["fare_amount"] > 0)
    & (train_df["fare_amount"] <= 250)
    & (train_df["passenger_count"] >= 1)
    & (train_df["passenger_count"] <= 6)
    & (train_df["pickup_longitude"].between(-75, -72))
    & (train_df["dropoff_longitude"].between(-75, -72))
    & (train_df["pickup_latitude"].between(40, 42))
    & (train_df["dropoff_latitude"].between(40, 42))
].copy()
print("New size (after sanity filter): %d" % len(train_df))


## === cell 12
ls1 = list(train_df["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][11:-7:]
train_df["pickuptime"] = ls1

ls1 = list(test_df["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][11:-7:]
test_df["pickuptime"] = ls1


## === cell 13
train_df.head()


## === cell 14
ls1 = list(train_df["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][:-4:]
    ls1[i] = pd.Timestamp(ls1[i])
    ls1[i] = ls1[i].weekday()
train_df["Weekday"] = ls1

ls1 = list(test_df["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][:-4:]
    ls1[i] = pd.Timestamp(ls1[i])
    ls1[i] = ls1[i].weekday()
test_df["Weekday"] = ls1


## === cell 15
train_df.head()


## === cell 16
test_df.head()


## === cell 17
train_df.drop("pickup_datetime", inplace=True, axis=1)
test_df.drop("pickup_datetime", inplace=True, axis=1)


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
train_one_hot = pd.get_dummies(train_df["Weekday"])
test_one_hot = pd.get_dummies(test_df["Weekday"])

all_weekdays = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
train_one_hot = train_one_hot.reindex(columns=all_weekdays, fill_value=0)
test_one_hot = test_one_hot.reindex(columns=all_weekdays, fill_value=0)

train_df = pd.concat([train_df, train_one_hot], axis=1)
test_df = pd.concat([test_df, test_one_hot], axis=1)


## === cell 20
train_df.drop("Weekday", axis=1, inplace=True)
test_df.drop("Weekday", axis=1, inplace=True)


## === cell 21
ls1 = list(train_df["pickuptime"])
for i in range(len(ls1)):
    z = ls1[i].split(":")
    ls1[i] = int(z[0]) * 100 + int(z[1])
train_df["pickuptime"] = ls1

ls1 = list(test_df["pickuptime"])
for i in range(len(ls1)):
    z = ls1[i].split(":")
    ls1[i] = int(z[0]) * 100 + int(z[1])
test_df["pickuptime"] = ls1


## === cell 22
train_df.head()


## === cell 23
R = 6373.0
lat1 = np.asarray(np.radians(train_df["pickup_latitude"]))
lon1 = np.asarray(np.radians(train_df["pickup_longitude"]))
lat2 = np.asarray(np.radians(train_df["dropoff_latitude"]))
lon2 = np.asarray(np.radians(train_df["dropoff_longitude"]))

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
a = np.clip(a, 0.0, 1.0)
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
a = np.clip(a, 0.0, 1.0)
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
test_df["Distance"] = np.asarray(distance) * 0.621


## === cell 24
R = 6373.0
lat1 = np.asarray(np.radians(train_df["pickup_latitude"]))
lon1 = np.asarray(np.radians(train_df["pickup_longitude"]))
lat2 = np.asarray(np.radians(train_df["dropoff_latitude"]))
lon2 = np.asarray(np.radians(train_df["dropoff_longitude"]))

lat3 = np.zeros(len(train_df)) + np.radians(40.6413111)
lon3 = np.zeros(len(train_df)) + np.radians(-73.7781391)
dlon_pickup = lon3 - lon1
dlat_pickup = lat3 - lat1
d_lon_dropoff = lon3 - lon2
d_lat_dropoff = lat3 - lat2

a1 = (
    np.sin(dlat_pickup / 2) ** 2
    + np.cos(lat1) * np.cos(lat3) * np.sin(dlon_pickup / 2) ** 2
)
a1 = np.clip(a1, 0.0, 1.0)
c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
distance1 = R * c1
train_df["Pickup_Distance_airport"] = np.asarray(distance1) * 0.621

a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
a2 = np.clip(a2, 0.0, 1.0)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
train_df["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621

lat1 = np.asarray(np.radians(test_df["pickup_latitude"]))
lon1 = np.asarray(np.radians(test_df["pickup_longitude"]))
lat2 = np.asarray(np.radians(test_df["dropoff_latitude"]))
lon2 = np.asarray(np.radians(test_df["dropoff_longitude"]))

lat3 = np.zeros(len(test_df)) + np.radians(40.6413111)
lon3 = np.zeros(len(test_df)) + np.radians(-73.7781391)
dlon_pickup = lon3 - lon1
dlat_pickup = lat3 - lat1
d_lon_dropoff = lon3 - lon2
d_lat_dropoff = lat3 - lat2

a1 = (
    np.sin(dlat_pickup / 2) ** 2
    + np.cos(lat1) * np.cos(lat3) * np.sin(dlon_pickup / 2) ** 2
)
a1 = np.clip(a1, 0.0, 1.0)
c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
distance1 = R * c1
test_df["Pickup_Distance_airport"] = np.asarray(distance1) * 0.621

a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
a2 = np.clip(a2, 0.0, 1.0)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
test_df["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621


## === cell 25
train_df["Distance"] = np.round(train_df["Distance"], 2)
train_df["Pickup_Distance_airport"] = np.round(train_df["Pickup_Distance_airport"], 2)
train_df["Dropoff_Distance_airport"] = np.round(train_df["Dropoff_Distance_airport"], 2)

test_df["Distance"] = np.round(test_df["Distance"], 2)
test_df["Pickup_Distance_airport"] = np.round(test_df["Pickup_Distance_airport"], 2)
test_df["Dropoff_Distance_airport"] = np.round(test_df["Dropoff_Distance_airport"], 2)


## === cell 26
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


## === cell 27
train_df.head()


## === cell 28
test_df.head()


## === cell 29
train_df.head()


## === cell 30
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

train_X = get_input_matrix(train_df)
train_y = np.array(train_df["fare_amount"])
print("Rebuilt aligned train_X/train_y:", train_X.shape, train_y.shape)

feature_cols = [c for c in train_df.columns if c not in ["key", "fare_amount"]]

for c in feature_cols:
    train_df[c] = pd.to_numeric(train_df[c], errors="coerce")
    test_df[c] = pd.to_numeric(test_df[c], errors="coerce")

train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
test_df.replace([np.inf, -np.inf], np.nan, inplace=True)

medians = train_df[feature_cols].median(numeric_only=True)
train_df[feature_cols] = train_df[feature_cols].fillna(medians)
test_df[feature_cols] = test_df[feature_cols].fillna(medians)

X = train_df.drop(["key", "fare_amount"], axis=1)
y = train_df["fare_amount"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.01, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


## === cell 31
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
print(lr.score(X_test_scaled, y_test))


## === cell 32
test_X = test_df.drop("key", axis=1).reindex(columns=X.columns, fill_value=0)
test_X = test_X.apply(pd.to_numeric, errors="coerce").fillna(medians)
test_X_scaled = scaler.transform(test_X)

pred = lr.predict(test_X_scaled)
pred = np.clip(pred, 0, None)  # fares can't be negative; improves RMSE stability
pred = np.round(pred, 2)


## === cell 33
Submission = pd.DataFrame({"key": test_df["key"], "fare_amount": pred})
Submission.head()


## === cell 34
Submission.to_csv("submission.csv", index=False)


## === cell 35
print(Submission.shape)
print(Submission.columns)
print(Submission.head())
