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

5.689

# 6. Current score

33.99946

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 936.94064) has done: 'I fix the runtime error caused by the deprecated `normalize` argument in `sklearn.linear_model.LinearRegression` (removed in scikit-learn 1.2+), while keeping the same linear-regression core approach. To preserve the intended “normalized” behavior with minimal change, I add a `Pipeline` with `StandardScaler` before `LinearRegression`. I also ensure train/test feature columns are aligned after one-hot encoding weekdays (so missing weekday columns in test won’t break prediction). Finally, I make sure a valid `Submission.csv` with columns `key,fare_amount` is always written.'
- What this solution (achieved 752.76916) has done: 'Your RMSE is exploding because the test feature normalization is being computed using the *test* mean/variance (data leakage in preprocessing, but in the wrong direction), so train/test are on different scales and the linear model outputs huge errors. To move the score strongly toward the target while preserving your core model/training, I make the scaling consistent by computing normalization parameters on the training set and applying them to both train and test (a minimal semantic fix). I also clip obviously-invalid negative fare predictions to 0 (a common post-process for this competition that reduces extreme errors without changing the model). Everything else (feature engineering, LinearRegression in a scaler pipeline, submission schema/path) stays the same.'
- What this solution (achieved 1117.2198) has done: 'Your gap to the target RMSE is huge (752.8 vs 5.689, lower is better), so we need a meaningful but still “core-logic-preserving” fix: the main issue is that the model is learning from many clearly bad training rows (negative/zero fares, out-of-range lat/long, and extreme outliers) which makes linear regression generalize terribly. I add standard NYC Taxi competition-style data cleaning filters (fare range, passenger_count range, bounding-box coordinates, and a max distance cap) while keeping your exact feature engineering and the same `StandardScaler + LinearRegression` pipeline. I also switch your manual “variance” normalization to true standard deviation (still using train-only stats) to avoid shrinking features by ~var instead of std, which otherwise destabilizes coefficients. Finally, I keep your clipping and submission-writing logic intact, ensuring a valid `Submission.csv` is produced.'
- What this solution (achieved 1117.2198) has done: 'Your current RMSE is still orders of magnitude above the target, which strongly suggests a submission-format/alignment problem rather than a modeling-capacity issue. The biggest minimal-risk fix is to stop setting `key` as the index (some leaderboard evaluators mishandle index vs column) and instead write `key` as an explicit first column, in the exact row order of `test.csv`. I also ensure the `key` is treated as a string (to prevent any parsing/coercion changes) and add a quick sanity check that the submission row count and keys match the test set before writing. Everything else (your feature engineering, cleaning, StandardScaler+LinearRegression pipeline, and clipping) stays the same.'
- What this solution (achieved 15.32192) has done: 'Your RMSE is still far from the target, which strongly suggests the feature matrix is numerically unstable (multicollinearity + outliers) for plain OLS `LinearRegression`, leading to extreme predictions. Keeping the exact same training approach (a linear model in a `StandardScaler` pipeline), the smallest meaningful fix is to switch to `Ridge` regression (still linear regression, just L2-regularized) to stabilize coefficients and reduce blow-ups. I also add a very light upper clip on predictions (fare can’t realistically exceed the training cap you already used) to prevent a few extreme values from dominating RMSE, while preserving your existing non-negative clipping and submission format. Everything else (data read, cleaning, feature engineering, scaling-on-train, submission writing and key alignment) remains the same.'
- What this solution (achieved 35.34766) has done: 'Your current RMSE (15.32) is still far above the target (5.689), so we should make a small, legitimate change that reliably improves generalization without changing the overall “linear model + scaling” approach. The biggest remaining issue is that Ridge is still fit with only linear terms, while taxi fare depends much more on interactions (e.g., distance × time-of-day, distance × airport proximity), so I add a `PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)` step inside the same sklearn `Pipeline`. I keep the same cleaning, the same engineered features, the same Ridge model, and the same submission writing; the only other minimal fix is to use a solver-compatible Ridge configuration (remove `random_state`, which is ignored/invalid for default solvers). This should move RMSE materially down toward the target band while preserving your core logic and producing the same valid `Submission.csv`.'
- What this solution (achieved 35.34765) has done: 'We’re far above the target (RMSE 35.35 vs 5.689, lower is better), so we need a small but meaningful generalization improvement while keeping your same overall pipeline (scaling → interaction-only polynomial features → Ridge). The biggest remaining issue is that interaction-only degree-2 on uncentered latitude/longitude-derived features can still be poorly conditioned and sensitive to outliers; the smallest stabilizing tweak is to increase Ridge regularization slightly and ensure the solver is robust for high-dimensional poly features. I also make the fare clipping consistent with your training filter (0–250) but avoid rounding predictions before saving (rounding hurts RMSE slightly and is unnecessary for submission). Everything else—data loading, cleaning rules, feature engineering, and submission key alignment—stays the same and it still write a valid `Submission.csv`.'
- What this solution (achieved 35.34765) has done: 'Your current RMSE (35.35) is far above the target (5.689, lower is better), so the smallest legitimate move toward the target is to fix a key feature issue without changing your overall pipeline (scaler → interaction-only poly → Ridge). Right now, the weekday one-hot columns can mismatch between train/test (because `align(join="left")` keeps only train columns and can silently drop weekday columns that appear only in test), which degrades generalization; switching to a stable categorical dtype with fixed categories and then aligning with `join="outer"` makes train/test feature spaces consistent. I also replace the slow Python loops that parse datetime with vectorized pandas parsing to avoid timeout risk while keeping identical semantics (weekday + HHMM). Everything else—cleaning rules, engineered distance features, pipeline steps, regularization strength, clipping, and submission-writing—stays the same.'
- What this solution (achieved 36.30061) has done: 'We’re still far above the target RMSE (35.35 vs 5.689, lower is better), so we need a minimal change that improves generalization without changing your overall approach (cleaning + engineered features + scaler → interaction-only poly → Ridge). The biggest remaining issue is that after adding `PolynomialFeatures`, your model has no explicit intercept because `include_bias=False` and Ridge is fit on transformed features; adding a `LinearRegression`-style intercept via `Ridge(fit_intercept=True)` (and ensuring the pipeline doesn’t implicitly remove it) is a small, legitimate fix that typically reduces systematic bias and RMSE. Additionally, your manual normalization currently uses `abs(x - mean)` which discards sign and can hurt linear modeling; switching to standard z-score `(x-mean)/std` is still the same “normalize with train stats” logic but preserves directionality and usually helps RMSE. Everything else (data size, filters, feature engineering, model family, and submission writing) stays the same and still produces `Submission.csv`.'
- What this solution (achieved 33.99946) has done: 'Your current RMSE (36.30) is still far above the target (5.689, lower is better), so we should make a small, legitimate improvement that preserves your pipeline and feature logic. The biggest low-risk gain for this competition is to model the target in log-space (fit on `log1p(fare_amount)` and invert with `expm1`), which reduces the impact of large-fare outliers that dominate RMSE while keeping the same Ridge + scaling + interaction-only polynomial pipeline. I also remove the manual z-scoring of just two features (Difference_latitude/longitude) because you already standardize all features via `StandardScaler`, and double-scaling can hurt conditioning and generalization. Everything else (data cleaning rules, feature engineering, train/test alignment, model family, and submission writing) stays the same and still writes a valid `Submission.csv`.'

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
test_data = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
    dtype={"key": "string"},
)
test_data.head()



## === cell 5
test_data.info()



## === cell 6
test_data.shape



## === cell 7
train_data.isna().sum()



## === cell 8
train_data.isnull().sum()



## === cell 9
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



## === cell 10
print(f"Before Dropping null values: {len(train_data)}")
train_data.dropna(inplace=True)
print(f"After Dropping null values: {len(train_data)}")



## === cell 11
plot = train_data[:2000].plot.scatter("Difference_longitude", "Difference_latitude")



## === cell 12
train_data = train_data[
    (train_data["Difference_longitude"] < 5.0)
    & (train_data["Difference_latitude"] < 5.0)
]



## === cell 13
train_data = train_data[
    (train_data["fare_amount"] > 0.0)
    & (train_data["fare_amount"] <= 250.0)
    & (train_data["passenger_count"] >= 1)
    & (train_data["passenger_count"] <= 6)
    & (train_data["pickup_longitude"].between(-74.3, -72.9))
    & (train_data["dropoff_longitude"].between(-74.3, -72.9))
    & (train_data["pickup_latitude"].between(40.4, 41.2))
    & (train_data["dropoff_latitude"].between(40.4, 41.2))
].copy()



## === cell 14
train_dt = pd.to_datetime(train_data["pickup_datetime"], errors="coerce", utc=False)
test_dt = pd.to_datetime(test_data["pickup_datetime"], errors="coerce", utc=False)

train_data = train_data.loc[train_dt.notna()].copy()
train_dt = train_dt.loc[train_dt.notna()]

train_data["pickuptime"] = (
    train_dt.dt.hour.astype(np.int16) * 100 + train_dt.dt.minute.astype(np.int16)
).astype(np.int32)
test_data["pickuptime"] = (
    test_dt.dt.hour.astype(np.int16) * 100 + test_dt.dt.minute.astype(np.int16)
).astype(np.int32)

weekday_names = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
train_data["Weekday"] = pd.Categorical(
    train_dt.dt.weekday.map(lambda i: weekday_names[int(i)]),
    categories=weekday_names,
    ordered=False,
)
test_data["Weekday"] = pd.Categorical(
    test_dt.dt.weekday.map(lambda i: weekday_names[int(i)]),
    categories=weekday_names,
    ordered=False,
)



## === cell 15
train_data.head()



## === cell 16
test_data.head()



## === cell 17
train_data.drop("pickup_datetime", inplace=True, axis=1)
test_data.drop("pickup_datetime", inplace=True, axis=1)



## === cell 18
train_one_hot = pd.get_dummies(train_data["Weekday"], dtype=np.float32)
test_one_hot = pd.get_dummies(test_data["Weekday"], dtype=np.float32)
train_data = pd.concat([train_data, train_one_hot], axis=1)
test_data = pd.concat([test_data, test_one_hot], axis=1)



## === cell 19
train_data.drop("Weekday", axis=1, inplace=True)
test_data.drop("Weekday", axis=1, inplace=True)



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
train_data = train_data[train_data["Distance"].between(0.0, 100.0)].copy()



## === cell 23
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



## === cell 24
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



## === cell 25
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



## === cell 26
train_data.shape



## === cell 27
test_data.shape



## === cell 28
train_features = train_data.drop(["key", "fare_amount"], axis=1)
test_features = test_data.drop(["key"], axis=1)
train_features, test_features = train_features.align(
    test_features, join="outer", axis=1, fill_value=0.0
)

from sklearn.model_selection import train_test_split

X = train_features
y = np.log1p(train_data["fare_amount"].astype(np.float64))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.01, random_state=80
)



## === cell 29
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

lr = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        (
            "poly",
            PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
        ),
        ("lr", Ridge(alpha=10.0, solver="auto", fit_intercept=True)),
    ]
)
lr.fit(X_train, y_train)
print(lr.score(X_test, y_test))



## === cell 30
pred_log = lr.predict(test_features)
pred = np.expm1(pred_log)
pred = np.clip(pred, 0.0, 250.0)



## === cell 31
pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
).head()



## === cell 32
submission = pd.DataFrame(
    {
        "key": test_data["key"].astype("string").values,
        "fare_amount": pred,
    }
)

assert len(submission) == len(test_data), "Submission row count must match test.csv"
assert (
    submission["key"].iloc[0] == test_data["key"].iloc[0]
), "Key order mismatch vs test.csv"
assert submission["key"].isna().sum() == 0, "Submission key contains NaN"

submission.head()



## === cell 33
submission.to_csv("Submission.csv", index=False)
print("Wrote Submission.csv with shape:", submission.shape)
print("Columns:", list(submission.columns))
