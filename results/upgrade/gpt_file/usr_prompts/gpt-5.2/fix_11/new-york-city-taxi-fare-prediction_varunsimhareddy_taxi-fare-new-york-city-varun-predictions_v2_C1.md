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

5.68916

# 6. Current score

1169.51772

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 936.94064) has done: 'I fix the runtime error caused by `LinearRegression(normalize=True)` which is no longer supported in scikit-learn 1.2.2, by replacing it with an equivalent `Pipeline(StandardScaler, LinearRegression)` (same linear model, just explicit normalization). I also ensure the one-hot encoded weekday columns are aligned between train and test so inference never fails due to missing weekday dummy columns. Finally, I make the submission generation robust (keep `key` as a column, enforce numeric predictions, and write a `.csv` file) so the notebook runs end-to-end and produces a valid Kaggle submission.'
- What this solution (achieved 752.78354) has done: 'Your huge RMSE is mainly coming from a feature scaling bug: you normalize `Difference_longitude/latitude` in the test set using the test set’s own mean/variance, but the model was trained on train-normalized values, so inference is on a different scale. I keep your exact feature set and LinearRegression pipeline, but compute the mean/variance on the training data once and apply those same parameters to both train and test (no change to core model logic). I also clip negative fare predictions to 0 (valid fares) which usually reduces RMSE for linear models without changing the training approach. Everything else (data reading, feature engineering, model, submission format/path) stays the same and still writes a valid `Submission.csv`.'
- What this solution (achieved 869.38972) has done: 'Your current RMSE is still extremely high versus the 5.689 target, which strongly suggests the model is being trained on many invalid/outlier rows (e.g., negative/huge fares, bad coordinates, passenger_count out of range) that dominate the squared error. I keep your exact feature engineering and the same `StandardScaler + LinearRegression` pipeline, but add a minimal set of standard NYC Taxi Fare sanity filters on the training data only (no leakage) so the linear model fits the typical fare regime. I also ensure the train/test feature columns are strictly aligned right before fitting/predicting to prevent any silent column mismatch. These are small, safe changes that usually move RMSE down by orders of magnitude for this specific competition without changing the core modeling approach.'
- What this solution (achieved 1112.2776) has done: 'Your RMSE is still massively above the target, and the most likely remaining cause is that the model is being trained with a few extreme-distance/outlier trips that survive the current basic filters and dominate squared error. To move RMSE sharply down while preserving your exact model/feature logic, I add one minimal, competition-standard filter on the *engineered* haversine `Distance` (and airport distances) after they’re computed, keeping only realistic NYC trip ranges. I also fix a small scaling bug: you divide by variance instead of standard deviation, which can badly distort those two features; switching to std keeps the same “manual normalization” idea but makes it correct and stable. Finally, I keep the train/test column alignment exactly once and ensure the submission is written as before.'
- What this solution (achieved 1112.2776) has done: 'Your score is far worse than the target (lower-is-better), so the smallest high-impact fix is to correct a likely feature/row misalignment between `test_dt` and `X_test_aligned`: you currently align `X` to the test columns but never reorder `test_dt` to match the filtered/aligned `X_test_aligned` rows, which can silently pair predictions with the wrong `key` values and explode RMSE. I keep your exact feature engineering and the same `StandardScaler + LinearRegression` pipeline, but build `X_test_aligned` from `test_dt` and then construct the submission from `X_test_aligned.index` to guarantee `key`/prediction row alignment. I also enforce identical column order for safety (same columns, same order) right before predicting. These are minimal changes that preserve your core logic while directly targeting the most common cause of extremely large RMSE in this competition.'
- What this solution (achieved 1125.87256) has done: 'Your RMSE is still extremely far from the target (lower is better), which is most consistent with a train/test feature mismatch rather than model capacity. The smallest high-impact fix is to remove the manual “abs-mean/std” normalization for `Difference_longitude/latitude` because it distorts those features (and then you *also* StandardScale everything), effectively double/incorrect scaling; instead we keep the same features but let the existing `StandardScaler` handle scaling consistently. I also add a minimal guard to ensure `X_test_aligned` is built from `X.columns` after all processing and that `key` alignment is strictly 1:1 with the test row order at submission time. These are minimal changes that preserve your feature set and the same `StandardScaler + LinearRegression` pipeline while directly addressing the most common cause of RMSE explosions.'
- What this solution (achieved 1125.87256) has done: 'Your RMSE is still astronomically high versus the target, which most strongly suggests the submission `key` order is not guaranteed to match the prediction row order after all the intermediate dataframe ops. I make the submission strictly index-aligned by carrying `key` as the index from the start of `test_dt` feature building, building `X_test_aligned` from that indexed frame, and then constructing the submission from `X_test_aligned.index` (so predictions and keys are 1:1 in the same order). I also hard-enforce identical column order for train/test right before fit/predict to eliminate any silent mismatch, without changing your model, features, or training loop. These are minimal changes aimed directly at fixing the most common cause of huge RMSE in this competition (key/prediction misalignment), and still produce a valid `Submission.csv`.'
- What this solution (achieved 1169.78967) has done: 'Your RMSE is catastrophically high for this competition, which is most consistent with invalid/overflowing engineered features (NaN/inf distances) slipping into training and/or test and then producing extreme linear predictions. I keep your exact feature set and the same `StandardScaler + LinearRegression` pipeline, but add a minimal “finite-only” filter after distance feature creation (train) and replace any non-finite test features with safe finite values (test) so the scaler/regression behave normally. I also add a small, competition-standard fare-vs-distance sanity filter (train only) to reduce squared-error domination by remaining outliers without changing the modeling approach. Finally, I keep your strict column/index alignment and submission generation unchanged except for enforcing numeric, finite predictions.'
- What this solution (achieved 1169.51772) has done: 'Your current RMSE is orders of magnitude worse than the target (lower is better), so the smallest likely high-impact fix is to stop rounding the engineered distance features and predictions, because that destroys information and can destabilize a linear model’s generalization on this competition. I keep your exact feature set and the same `StandardScaler + LinearRegression` pipeline, but remove the rounding of `Distance`/airport distances and keep full-precision predictions (still clipping negatives to 0 for valid fares). I also add a minimal guard to ensure there are no NaN/inf values in *any* feature columns right before fitting/predicting (not just the distance columns), since any remaining non-finite values can explode predictions and RMSE. Submission generation, column alignment, and file path/name remain unchanged and still produce `Submission.csv`.'
- What this solution (achieved 1169.51772) has done: 'Your RMSE is still far above the target (lower is better), which most strongly indicates the model is being trained on mislabeled/shifted targets rather than “just” noisy features. The smallest high-impact fix that preserves your model and feature engineering is to prevent `train_test_split` from breaking index alignment between `X` and `y` by making `y` explicitly share `X`’s index (after all row filtering) and adding a safety assert before fitting. I also remove the unused split variables (they’re fine) but keep the exact same split ratio/seed and the same `StandardScaler + LinearRegression` pipeline. This should move the score sharply down toward the target without changing the core approach.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
train_dt = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", nrows=10_000_000
)
train_dt.head()



## === cell 2
train_dt.shape



## === cell 3
train_dt.info()



## === cell 4
test_dt = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")
test_dt.head()



## === cell 5
test_dt.info()



## === cell 6
train_dt.isna().sum()



## === cell 7
train_dt["Difference_longitude"] = np.abs(
    np.asarray(train_dt["pickup_longitude"] - train_dt["dropoff_longitude"])
)
train_dt["Difference_latitude"] = np.abs(
    np.asarray(train_dt["pickup_latitude"] - train_dt["dropoff_latitude"])
)

test_dt["Difference_longitude"] = np.abs(
    np.asarray(test_dt["pickup_longitude"] - test_dt["dropoff_longitude"])
)
test_dt["Difference_latitude"] = np.abs(
    np.asarray(test_dt["pickup_latitude"] - test_dt["dropoff_latitude"])
)



## === cell 8
print(f"Before Dropping null values: {len(train_dt)}")
train_dt.dropna(inplace=True)
print(f"After Dropping null values: {len(train_dt)}")



## === cell 9
plot = train_dt[:2000].plot.scatter("Difference_longitude", "Difference_latitude")



## === cell 10
train_dt = train_dt[
    (train_dt["Difference_longitude"] < 5.0) & (train_dt["Difference_latitude"] < 5.0)
]



## === cell 11
train_dt = train_dt[
    (train_dt["fare_amount"] > 0.0)
    & (train_dt["fare_amount"] <= 500.0)
    & (train_dt["passenger_count"] >= 1)
    & (train_dt["passenger_count"] <= 6)
    & (train_dt["pickup_longitude"].between(-75.0, -72.0))
    & (train_dt["dropoff_longitude"].between(-75.0, -72.0))
    & (train_dt["pickup_latitude"].between(40.0, 42.0))
    & (train_dt["dropoff_latitude"].between(40.0, 42.0))
].copy()
train_dt.shape



## === cell 12
ls1 = list(train_dt["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][11:-7:]
train_dt["pickuptime"] = ls1

ls1 = list(test_dt["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][11:-7:]
test_dt["pickuptime"] = ls1



## === cell 13
train_dt.head()



## === cell 14
ls1 = list(train_dt["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][:-4:]
    ls1[i] = pd.Timestamp(ls1[i])
    ls1[i] = ls1[i].weekday()
train_dt["Weekday"] = ls1

ls1 = list(test_dt["pickup_datetime"])
for i in range(len(ls1)):
    ls1[i] = ls1[i][:-4:]
    ls1[i] = pd.Timestamp(ls1[i])
    ls1[i] = ls1[i].weekday()
test_dt["Weekday"] = ls1



## === cell 15
train_dt.head()



## === cell 16
test_dt.head()



## === cell 17
train_dt.drop("pickup_datetime", inplace=True, axis=1)
test_dt.drop("pickup_datetime", inplace=True, axis=1)



## === cell 18
train_dt["Weekday"].replace(
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
test_dt["Weekday"].replace(
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
train_one_hot = pd.get_dummies(train_dt["Weekday"])
test_one_hot = pd.get_dummies(test_dt["Weekday"])
train_one_hot, test_one_hot = train_one_hot.align(
    test_one_hot, join="outer", axis=1, fill_value=0
)

train_dt = pd.concat([train_dt, train_one_hot], axis=1)
test_dt = pd.concat([test_dt, test_one_hot], axis=1)



## === cell 20
train_dt.drop("Weekday", axis=1, inplace=True)
test_dt.drop("Weekday", axis=1, inplace=True)



## === cell 21
ls1 = list(train_dt["pickuptime"])
for i in range(len(ls1)):
    z = ls1[i].split(":")
    ls1[i] = int(z[0]) * 100 + int(z[1])
train_dt["pickuptime"] = ls1

ls1 = list(test_dt["pickuptime"])
for i in range(len(ls1)):
    z = ls1[i].split(":")
    ls1[i] = int(z[0]) * 100 + int(z[1])
test_dt["pickuptime"] = ls1



## === cell 22
train_dt.head()



## === cell 23
R = 6373.0
lat1 = np.asarray(np.radians(train_dt["pickup_latitude"]))
lon1 = np.asarray(np.radians(train_dt["pickup_longitude"]))
lat2 = np.asarray(np.radians(train_dt["dropoff_latitude"]))
lon2 = np.asarray(np.radians(train_dt["dropoff_longitude"]))

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
train_dt["Distance"] = np.asarray(distance) * 0.621

lat1 = np.asarray(np.radians(test_dt["pickup_latitude"]))
lon1 = np.asarray(np.radians(test_dt["pickup_longitude"]))
lat2 = np.asarray(np.radians(test_dt["dropoff_latitude"]))
lon2 = np.asarray(np.radians(test_dt["dropoff_longitude"]))

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
test_dt["Distance"] = np.asarray(distance) * 0.621



## === cell 24
R = 6373.0
lat1 = np.asarray(np.radians(train_dt["pickup_latitude"]))
lon1 = np.asarray(np.radians(train_dt["pickup_longitude"]))
lat2 = np.asarray(np.radians(train_dt["dropoff_latitude"]))
lon2 = np.asarray(np.radians(train_dt["dropoff_longitude"]))

lat3 = np.zeros(len(train_dt)) + np.radians(40.6413111)
lon3 = np.zeros(len(train_dt)) + np.radians(-73.7781391)
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
train_dt["Pickup_Distance_airport"] = np.asarray(distance1) * 0.621

a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
train_dt["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621

lat1 = np.asarray(np.radians(test_dt["pickup_latitude"]))
lon1 = np.asarray(np.radians(test_dt["pickup_longitude"]))
lat2 = np.asarray(np.radians(test_dt["dropoff_latitude"]))
lon2 = np.asarray(np.radians(test_dt["dropoff_longitude"]))

lat3 = np.zeros(len(test_dt)) + np.radians(40.6413111)
lon3 = np.zeros(len(test_dt)) + np.radians(-73.7781391)
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
test_dt["Pickup_Distance_airport"] = np.asarray(distance1) * 0.621

a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
test_dt["Dropoff_Distance_airport"] = np.asarray(distance2) * 0.621



## === cell 25
finite_cols = ["Distance", "Pickup_Distance_airport", "Dropoff_Distance_airport"]
train_dt = train_dt[np.isfinite(train_dt[finite_cols]).all(axis=1)].copy()

for c in finite_cols:
    med = np.nanmedian(test_dt[c].to_numpy(dtype=float))
    test_dt[c] = pd.to_numeric(test_dt[c], errors="coerce")
    test_dt[c] = test_dt[c].replace([np.inf, -np.inf], np.nan).fillna(med)



## === cell 26
train_dt = train_dt[
    (train_dt["Distance"] >= 0.0)
    & (train_dt["Distance"] <= 60.0)
    & (train_dt["Pickup_Distance_airport"] >= 0.0)
    & (train_dt["Pickup_Distance_airport"] <= 60.0)
    & (train_dt["Dropoff_Distance_airport"] >= 0.0)
    & (train_dt["Dropoff_Distance_airport"] <= 60.0)
].copy()
train_dt.shape



## === cell 27
train_dt = train_dt[
    (train_dt["fare_amount"] <= 3.0 + 15.0 * train_dt["Distance"])
].copy()
train_dt.shape



## === cell 28
train_dt.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)
test_dt.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)



## === cell 29
train_dt["Difference_longitude"] = train_dt["Difference_longitude"].astype(float)
train_dt["Difference_latitude"] = train_dt["Difference_latitude"].astype(float)
test_dt["Difference_longitude"] = test_dt["Difference_longitude"].astype(float)
test_dt["Difference_latitude"] = test_dt["Difference_latitude"].astype(float)



## === cell 30
train_dt.shape



## === cell 31
test_dt.shape



## === cell 32
from sklearn.model_selection import train_test_split

X = train_dt.drop(["key", "fare_amount"], axis=1)
y = train_dt.loc[X.index, "fare_amount"]

test_dt = test_dt.set_index("key", drop=False)

X_test_raw = test_dt.drop("key", axis=1).copy()
X_test_aligned = X_test_raw.reindex(columns=X.columns, fill_value=0)
X = X.reindex(columns=X_test_aligned.columns)

X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
X_test_aligned = X_test_aligned.apply(pd.to_numeric, errors="coerce").replace(
    [np.inf, -np.inf], np.nan
)

train_medians = X.median(numeric_only=True)
X = X.fillna(train_medians)
X_test_aligned = X_test_aligned.fillna(train_medians)

assert len(X) == len(y) and X.index.equals(
    y.index
), "X/y misalignment would destroy RMSE."

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.01, random_state=80
)



## === cell 33
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



## === cell 34
pred = lr.predict(X_test_aligned)

pred = np.asarray(pred, dtype=float)
pred[~np.isfinite(pred)] = np.nan
pred = (
    pd.Series(pred)
    .fillna(pred[np.isfinite(pred)].mean() if np.isfinite(pred).any() else 0.0)
    .to_numpy()
)

pred = np.clip(pred, 0.0, None)



## === cell 35
pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
).head()



## === cell 36
Submission = pd.DataFrame(
    {
        "key": X_test_aligned.index.astype(str).values,
        "fare_amount": pred.astype(float),
    }
)
Submission = Submission[["key", "fare_amount"]]
Submission.head()



## === cell 37
Submission.to_csv("Submission.csv", index=False)
print("Wrote Submission.csv with shape:", Submission.shape)
print("Columns:", list(Submission.columns))
print("Any NaNs in fare_amount:", Submission["fare_amount"].isna().any())
print("First 3 keys:", Submission["key"].head(3).tolist())
print("Last 3 keys:", Submission["key"].tail(3).tolist())
