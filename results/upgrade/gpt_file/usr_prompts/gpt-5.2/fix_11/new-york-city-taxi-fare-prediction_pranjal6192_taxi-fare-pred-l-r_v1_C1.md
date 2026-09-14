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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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

5.68914

# 6. Current score

15.31827

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 936.94064) has done: 'I fix the runtime error caused by the deprecated `normalize` argument in `sklearn.linear_model.LinearRegression` by switching to an equivalent `Pipeline(StandardScaler, LinearRegression)` so the model trains successfully in scikit-learn 1.2.2. I also keep the original feature engineering and train/test split logic intact, only adding a small alignment step to ensure train/test have identical one-hot weekday columns (prevents potential column-mismatch errors). Finally, I ensure a valid submission file named with a `.csv` suffix is always written with exactly the required columns `key,fare_amount` and in the same order as the test file.'
- What this solution (achieved 922.05621) has done: 'Your RMSE is extremely high because the model is being trained on noisy/outlier rows (including invalid coordinates and implausible fares) and because the manual “normalization” in cells 26–28 incorrectly normalizes the test set using its own mean/variance (distribution shift). To move the score sharply toward the 5.68914 target without changing the core model or feature set, I’m only adding standard NYC Taxi baseline data cleaning (valid lat/lon bounds, passenger_count bounds, positive/upper-bounded fares) and fixing the scaling bug by applying the train mean/variance to both train and test for those two features. Everything else (feature engineering, LinearRegression-in-a-Pipeline, train/test split, and submission format) remains the same.'
- What this solution (achieved 1013.07766) has done: 'Your current RMSE (922) is far above the target (5.689), so we should improve performance with minimal, metric-aligned fixes that don’t change the core model (LinearRegression) or the feature set conceptually. The biggest issue left is a “double scaling” bug: you manually normalize `Difference_*` in cell 27 and then the Pipeline’s `StandardScaler` scales everything again, which can badly distort those two features. I keep your exact feature engineering and training approach, but remove the manual normalization and instead add a small, standard baseline cleanup that prevents extreme/unrealistic distances from dominating the linear fit. This should move RMSE sharply downward toward the target without altering the model architecture or training loop semantics.'
- What this solution (achieved 1013.07763) has done: 'Your current RMSE is catastrophically high relative to the 5.68914 target, which strongly suggests a submission alignment/format issue rather than just model quality. The smallest fix that can move the score sharply toward the target is to ensure the predictions are paired with the correct `key` in exactly the same row order as `test.csv` (avoid any chance of index misalignment after feature engineering), and to avoid rounding predictions before submission (rounding adds error under RMSE). I keep your exact feature engineering and LinearRegression-in-StandardScaler pipeline, but I (1) hard-anchor the original test key order right after loading, (2) construct the submission using that anchored key series, and (3) remove the `np.round(..., 2)` at prediction time. This should materially reduce RMSE without changing the model/training approach.'
- What this solution (achieved 993.79406) has done: 'Your current RMSE is so far from the target that it almost certainly comes from a silent train/test feature mismatch rather than just “model quality”. The smallest, core-logic-preserving fix is to force the *exact same feature columns in the same order* for training and test right before fitting/predicting (your `key` anchoring is good, but column-order drift after `drop()`/`concat()` can still happen). I also remove the rounding of engineered distance features (rounding throws away signal and increases RMSE) while keeping the same features and the same LinearRegression+StandardScaler pipeline. Finally, I add a strict alignment/assertion check so the notebook fails loudly if columns ever diverge again, preventing another catastrophic submission.'
- What this solution (achieved 15.02976) has done: 'Your RMSE is still catastrophically far from the target, so we should focus on the smallest fixes that can plausibly remove a “bad prediction scale” failure mode without changing your core model/feature set. The biggest remaining issue is that plain LinearRegression on this problem is extremely sensitive to a few remaining high-leverage points; clipping the training target to the same bounds you already filter to (0–250) and clipping the *predictions* to that same valid fare range is a minimal, metric-aligned post-processing step that often collapses huge RMSE down dramatically when the model occasionally outputs extreme values. I’m also adding a tiny safety step to ensure all model inputs are numeric (no object dtypes sneaking in from datetime-derived columns), because silent object->NaN coercions can yield nonsense coefficients and gigantic errors. Everything else (features, LinearRegression+StandardScaler pipeline, split, and submission schema/key order) remains unchanged.'
- What this solution (achieved inf) has done: 'Your current RMSE (15.03) is still far above the 5.69 target, so we should make the smallest fixes that reduce systematic error without changing the core model (LinearRegression + StandardScaler) or your feature set. The biggest remaining issue is that the model is trained with plain MSE on raw dollars, which overweights large fares; applying a log1p transform to the target (and inverting it at prediction time) is a minimal, standard regression calibration step that often drops RMSE substantially for this competition while keeping the same model/training approach. I also remove the prediction clipping to 250 (it can only increase RMSE for true fares >250) and instead only clip to non-negative after inverse-transform. Finally, I keep your key-order anchoring and strict column alignment so the submission stays correctly paired.'
- What this solution (achieved 27991239.96438) has done: 'The `inf` RMSE almost always happens when the submission contains non-finite predictions (NaN/inf), which can be triggered here by extreme `pred_log` values from a few problematic rows or by non-finite engineered distances. To move the score back from `inf` toward your target while keeping the same feature engineering and the same `StandardScaler + LinearRegression` pipeline, I’m only adding strict finite-value cleaning for engineered numeric columns and a safe guardrail on the log-predictions before `expm1`. I also ensure the final submission has finite `fare_amount` for every test `key` (fallback to the training median only if something still slips through), which guarantees Kaggle can compute RMSE. These are minimal, metric-aligned stability fixes and do not change the model architecture, loss, or training loop.'
- What this solution (achieved 15.02976) has done: 'Your current RMSE is catastrophically worse than the target (lower is better), which is consistent with a broken scale/feature bug rather than “just” a weak model. The smallest core-logic-preserving fix is to stop using the log1p/expm1 target transform, because it changes the training objective away from raw-dollar RMSE and can easily create badly mis-scaled predictions if the linear fit is slightly off. I also add a minimal, metric-aligned safeguard by clipping predictions to the same fare range you already train-filter to (0–250), which prevents a few extreme outputs from exploding RMSE. Everything else (your feature engineering, cleaning, LinearRegression+StandardScaler pipeline, column alignment, and submission formatting) remains the same.'
- What this solution (achieved 15.31827) has done: 'Your current RMSE (15.03) is still far above the target (5.689), so we should make a small, metric-aligned adjustment that improves model fit without changing the model type, feature set, or training loop. The biggest remaining systematic issue is that raw LinearRegression is highly sensitive to remaining outliers/high-leverage points; switching the estimator to `HuberRegressor` (still a linear model, same features, same scaling pipeline) is a minimal change that typically reduces RMSE substantially on NYC Taxi. I keep your exact feature engineering/cleaning, train/val split, column alignment, and submission formatting, and only adjust the linear estimator + ensure the prediction clipping remains to avoid occasional extreme outputs inflating RMSE. This should move the score downward toward the target band while remaining stable and within Kaggle constraints.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import time
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
test_keys = test_data["key"].copy()
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
plot = train_data[:2000].plot.scatter("Difference_longitude", "Difference_latitude")



## === cell 10
train_data = train_data[
    (train_data["Difference_longitude"] < 5.0)
    & (train_data["Difference_latitude"] < 5.0)
]



## === cell 11
train_data = train_data[
    (train_data["fare_amount"] > 0) & (train_data["fare_amount"] <= 250)
]
train_data = train_data[
    (train_data["passenger_count"] >= 1) & (train_data["passenger_count"] <= 6)
]

train_data = train_data[
    (train_data["pickup_longitude"].between(-80, -70))
    & (train_data["dropoff_longitude"].between(-80, -70))
    & (train_data["pickup_latitude"].between(35, 45))
    & (train_data["dropoff_latitude"].between(35, 45))
]

train_data.head()



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

all_cols = sorted(set(train_one_hot.columns).union(set(test_one_hot.columns)))
train_one_hot = train_one_hot.reindex(columns=all_cols, fill_value=0)
test_one_hot = test_one_hot.reindex(columns=all_cols, fill_value=0)

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



## === cell 24
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



## === cell 25
train_data = train_data[
    (train_data["Distance"] > 0) & (train_data["Distance"] <= 100)
].copy()
train_data = train_data[
    (train_data["Pickup_Distance_airport"] <= 200)
    & (train_data["Dropoff_Distance_airport"] <= 200)
].copy()



## === cell 26
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



## === cell 27
train_data.shape



## === cell 28
test_data.shape



## === cell 29
from sklearn.model_selection import train_test_split

X = train_data.drop(["key", "fare_amount"], axis=1)
y = train_data["fare_amount"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.01, random_state=80)

X_test_final = test_data.drop("key", axis=1)
X_test_final = X_test_final.reindex(columns=X.columns, fill_value=0)
assert list(X.columns) == list(
    X_test_final.columns
), "Train/test feature columns misaligned."

X_train = (
    X_train.apply(pd.to_numeric, errors="coerce")
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
)
X_val = (
    X_val.apply(pd.to_numeric, errors="coerce")
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
)
X_test_final = (
    X_test_final.apply(pd.to_numeric, errors="coerce")
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
)

y_train = (
    pd.to_numeric(y_train, errors="coerce")
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
    .clip(lower=0.0, upper=250.0)
)
y_val = (
    pd.to_numeric(y_val, errors="coerce")
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
    .clip(lower=0.0, upper=250.0)
)

y_train_target = y_train.values.astype(np.float64)
y_val_target = y_val.values.astype(np.float64)

y_train_median = float(np.median(y_train_target))



## === cell 30
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

lr = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("lr", HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=200)),
    ]
)
lr.fit(X_train, y_train_target)
print("R^2 on fare val:", lr.score(X_val, y_val_target))



## === cell 31
pred = lr.predict(X_test_final)
pred = np.where(np.isfinite(pred), pred, y_train_median)
pred = np.clip(pred, 0.0, 250.0)



## === cell 32
pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
).head()



## === cell 33
Submission = pd.DataFrame({"key": test_keys.values, "fare_amount": pred})



## === cell 34
Submission.head()



## === cell 35
Submission.to_csv("Submission.csv", index=False)
print("Wrote submission to Submission.csv with shape:", Submission.shape)
print("Submission columns:", Submission.columns.tolist())
print(
    "Non-finite fare_amount count:",
    int((~np.isfinite(Submission["fare_amount"].values)).sum()),
)
print("First/last key check:", Submission["key"].iloc[0], Submission["key"].iloc[-1])
