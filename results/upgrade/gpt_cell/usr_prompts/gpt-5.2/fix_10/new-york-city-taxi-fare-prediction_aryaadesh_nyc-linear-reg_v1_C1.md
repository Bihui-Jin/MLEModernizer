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

8.55749

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 762.61302) has done: 'Diagnosis: Cell 22 concatenates the one-hot weekday columns into `train_df`/`test_df`, then immediately tries to `join(train_df[weekday_cols])` back onto the same DataFrame after dropping only `Weekday`. Since those weekday dummy columns already exist in `train_df`, `DataFrame.join` raises `ValueError: columns overlap but no suffix specified`. The intent appears to be just to ensure both train and test have the same dummy columns and (optionally) reorder them consistently.  
Patch summary: Replace the failing `.drop(...).join(...)` lines with a minimal, deterministic reordering step that does not re-join existing columns. This keeps the same columns and values while avoiding overlap.  
Updated cells: Only cell 22 is changed.  
Compatibility notes for cell k+1: `train_df` and `test_df` still exist with the same content as intended (weekday one-hot columns present; `Weekday` removed), so `train_df.head()` in cell 23 works unchanged.  
Assumptions: The goal of the final two lines in cell 22 is column ordering consistency rather than duplicating data; therefore, reindexing columns is the correct minimal equivalent.'
- What this solution (achieved 989.59268) has done: 'Your score is extremely far from the target (lower is better), which strongly suggests a data/label quality problem rather than model capacity. The smallest legitimate improvement is to filter obviously bad training rows (negative/zero fares, extreme fares, invalid passenger_count, and out-of-range NYC coordinates) before fitting the same LinearRegression on the same features; this usually collapses RMSE from “hundreds” down to single digits for this competition. I keep your exact feature engineering and model, but add minimal, standard cleaning right after `dropna` and before feature creation continues, and I also clip negative predictions to 0 (fares can’t be negative), which typically reduces RMSE a bit without changing the core approach. The script still run end-to-end and write `submission.csv` in the required format.'
- What this solution (achieved 960.85149) has done: 'Your RMSE is still astronomically worse than the target, which indicates the model is being trained on many extreme/outlier/noisy rows that dominate a LinearRegression fit. To move the score much closer to the target without changing your core feature engineering or model, I only tighten the training-data cleaning (still simple rule-based filtering) to remove clearly invalid coordinates/fare outliers and to also filter on the engineered distance features (which you already compute). I keep the same pipeline and LinearRegression, and I also ensure test rows with invalid coordinates get harmless filled features rather than causing pathological predictions. This should reduce the huge-error tail and bring RMSE down by orders of magnitude toward the target band.'
- What this solution (achieved 960.85149) has done: 'Your current RMSE is vastly worse than the target (lower is better), which strongly indicates a train/test feature mismatch rather than a model-capacity issue. The smallest fix that preserves your core logic is to ensure `train_df` and `test_df` end up with *identical feature columns in the same order* before fitting/predicting, because `pd.get_dummies` plus later column drops can easily leave subtle differences that make LinearRegression coefficients apply to the wrong features at inference. I add a minimal “column alignment” step right before the train/test split and prediction: reindex test features to match the training feature set (filling missing columns with 0) and enforce identical ordering. This keeps your model, features, and training approach unchanged while addressing the most likely cause of the huge score.'
- What this solution (achieved 8.55749) has done: 'Your code already has the right overall structure, but because you’re training `LinearRegression` on `log1p(fare_amount)`, the model can produce very large or very small values after `expm1`, and large outliers in the *training labels* can still dominate the fit and inflate RMSE. To move RMSE closer to the target without changing the model or feature set, I add one minimal extra cleaning step to remove extreme log-fare outliers using an IQR rule (applied after your existing cleaning), and I also cap the final predictions to your training fare range to avoid rare extreme predictions dominating RMSE. These changes preserve your core logic (same features, same `LinearRegression`, same log-transform training objective) while reducing the impact of pathological rows/predictions. The script still run end-to-end and write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

print(os.listdir("../input")[:20])



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
print("Old size (pre-clean): %d" % len(train_df))

train_df = train_df[(train_df.fare_amount > 0) & (train_df.fare_amount <= 200)]

train_df = train_df[(train_df.passenger_count >= 1) & (train_df.passenger_count <= 6)]

train_df = train_df[
    (train_df.pickup_longitude.between(-74.5, -72.8))
    & (train_df.dropoff_longitude.between(-74.5, -72.8))
    & (train_df.pickup_latitude.between(40.5, 41.8))
    & (train_df.dropoff_latitude.between(40.5, 41.8))
]

train_df = train_df[
    (train_df.abs_diff_longitude > 0) | (train_df.abs_diff_latitude > 0)
]
train_df = train_df[
    (train_df.abs_diff_longitude < 2.0) & (train_df.abs_diff_latitude < 2.0)
]

print("New size (post-clean): %d" % len(train_df))



## === cell 10
try:
    plot = train_df.iloc[:2000].plot.scatter("abs_diff_longitude", "abs_diff_latitude")
except Exception as e:
    print("Plot skipped:", repr(e))



## === cell 11
print("Old size: %d" % len(train_df))
train_df = train_df[
    (train_df.abs_diff_longitude < 5.0) & (train_df.abs_diff_latitude < 5.0)
]
print("New size: %d" % len(train_df))



## === cell 12
print("Old size: %d" % len(test_df))
print("New size: %d (unchanged to preserve all keys)" % len(test_df))



## === cell 13
train_dt = pd.to_datetime(train_df["pickup_datetime"], errors="coerce")
test_dt = pd.to_datetime(test_df["pickup_datetime"], errors="coerce")

train_df["pickup_time"] = train_dt.dt.hour * 100 + train_dt.dt.minute
test_df["pickup_time"] = test_dt.dt.hour * 100 + test_dt.dt.minute

train_df["Weekday"] = train_dt.dt.dayofweek
test_df["Weekday"] = test_dt.dt.dayofweek



## === cell 14
train_df.head()



## === cell 15
test_df.head()



## === cell 16
train_df.head()



## === cell 17
test_df.head()



## === cell 18
train_df.drop("pickup_datetime", inplace=True, axis=1)
test_df.drop("pickup_datetime", inplace=True, axis=1)



## === cell 19
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



## === cell 20
train_df.head()



## === cell 21
test_df.head()



## === cell 22
train_one_hot = pd.get_dummies(train_df["Weekday"])
test_one_hot = pd.get_dummies(test_df["Weekday"])

train_df = pd.concat([train_df, train_one_hot], axis=1)
test_df = pd.concat([test_df, test_one_hot], axis=1)

missing_in_test = set(train_one_hot.columns) - set(test_one_hot.columns)
for col in missing_in_test:
    test_df[col] = 0
missing_in_train = set(test_one_hot.columns) - set(train_one_hot.columns)
for col in missing_in_train:
    train_df[col] = 0

weekday_cols = sorted(list(set(train_one_hot.columns) | set(test_one_hot.columns)))

train_df = train_df.drop(columns=["Weekday"])
test_df = test_df.drop(columns=["Weekday"])

train_df = train_df.reindex(
    columns=[c for c in train_df.columns if c not in weekday_cols] + weekday_cols
)
test_df = test_df.reindex(
    columns=[c for c in test_df.columns if c not in weekday_cols] + weekday_cols
)



## === cell 23
train_df.head()



## === cell 24
test_df.head()



## === cell 25
pass



## === cell 26
train_df["pickup_time"] = (
    pd.to_numeric(train_df["pickup_time"], errors="coerce").fillna(0).astype(int)
)
test_df["pickup_time"] = (
    pd.to_numeric(test_df["pickup_time"], errors="coerce").fillna(0).astype(int)
)



## === cell 27
train_df.head()



## === cell 28
test_df.head()



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



## === cell 30
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



## === cell 31
print("Old size (pre-distance-clean): %d" % len(train_df))
train_df = train_df[(train_df["Distance"] >= 0) & (train_df["Distance"] <= 100)]
train_df = train_df[
    (train_df["Pickup_Distance_airport"] >= 0)
    & (train_df["Pickup_Distance_airport"] <= 150)
    & (train_df["Dropoff_Distance_airport"] >= 0)
    & (train_df["Dropoff_Distance_airport"] <= 150)
]
print("New size (post-distance-clean): %d" % len(train_df))



## === cell 32
log_fare = np.log1p(train_df["fare_amount"].astype(float))
q1, q3 = np.percentile(log_fare, [25, 75])
iqr = q3 - q1
low, high = q1 - 3.0 * iqr, q3 + 3.0 * iqr  # conservative fence to avoid over-pruning
before = len(train_df)
train_df = train_df[(log_fare >= low) & (log_fare <= high)]
print(
    "Post log-fare IQR-clean size: %d (removed %d rows)"
    % (len(train_df), before - len(train_df))
)



## === cell 33
train_df["Distance"] = np.round(train_df["Distance"], 2)
train_df["Pickup_Distance_airport"] = np.round(train_df["Pickup_Distance_airport"], 2)
train_df["Dropoff_Distance_airport"] = np.round(train_df["Dropoff_Distance_airport"], 2)
test_df["Distance"] = np.round(test_df["Distance"], 2)
test_df["Pickup_Distance_airport"] = np.round(test_df["Pickup_Distance_airport"], 2)
test_df["Dropoff_Distance_airport"] = np.round(test_df["Dropoff_Distance_airport"], 2)



## === cell 34
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



## === cell 35
abs_lon_mean = float(np.mean(train_df["abs_diff_longitude"]))
abs_lon_var = (
    float(np.var(train_df["abs_diff_longitude"]))
    if float(np.var(train_df["abs_diff_longitude"])) != 0.0
    else 1.0
)

train_df["abs_diff_longitude"] = np.abs(train_df["abs_diff_longitude"]) - abs_lon_mean
train_df["abs_diff_longitude"] = train_df["abs_diff_longitude"] / abs_lon_var



## === cell 36
abs_lat_mean = float(np.mean(train_df["abs_diff_latitude"]))
abs_lat_var = (
    float(np.var(train_df["abs_diff_latitude"]))
    if float(np.var(train_df["abs_diff_latitude"])) != 0.0
    else 1.0
)

train_df["abs_diff_latitude"] = np.abs(train_df["abs_diff_latitude"]) - abs_lat_mean
train_df["abs_diff_latitude"] = train_df["abs_diff_latitude"] / abs_lat_var



## === cell 37
test_df["abs_diff_longitude"] = np.abs(test_df["abs_diff_longitude"]) - abs_lon_mean
test_df["abs_diff_longitude"] = test_df["abs_diff_longitude"] / abs_lon_var



## === cell 38
test_df["abs_diff_latitude"] = np.abs(test_df["abs_diff_latitude"]) - abs_lat_mean
test_df["abs_diff_latitude"] = test_df["abs_diff_latitude"] / abs_lat_var



## === cell 39
train_df = train_df.replace([np.inf, -np.inf], np.nan)
test_df = test_df.replace([np.inf, -np.inf], np.nan)

train_df = train_df.dropna(axis=0, how="any")



## === cell 40
train_df.shape



## === cell 41
test_df.shape



## === cell 42
train_df.head()



## === cell 43
from sklearn.model_selection import train_test_split

X = train_df.drop(["key", "fare_amount"], axis=1)
y = train_df["fare_amount"]

test_X = test_df.drop(["key"], axis=1)

test_X = test_X.reindex(columns=X.columns, fill_value=0)
X = X.reindex(columns=test_X.columns, fill_value=0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.01, random_state=80
)



## === cell 44
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, np.log1p(y_train))

val_pred = np.expm1(lr.predict(X_test))
val_pred = np.clip(val_pred, 0, float(train_df["fare_amount"].max()))
rmse = float(np.sqrt(np.mean((val_pred - y_test.values) ** 2)))
print("Validation RMSE (original space):", rmse)



## === cell 45
pred = np.expm1(lr.predict(test_X))
pred = np.clip(pred, 0, float(train_df["fare_amount"].max()))
pred = np.round(pred, 2)



## === cell 46
pd.read_csv("../input/sample_submission.csv").head()



## === cell 47
Submission = pd.DataFrame({"key": test_df["key"].values, "fare_amount": pred})

Submission = Submission[["key", "fare_amount"]]
print(Submission.head())
print("Submission rows:", len(Submission), "Test rows:", len(test_df))



## === cell 48
Submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv")
