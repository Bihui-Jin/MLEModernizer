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

3.7

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
xgboost==2.0.3

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

4.73089

# 6. Current score

5.42652

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 192.36618) has done: 'I fix the runtime error caused by deprecated pandas datetime accessors (`.dt.week` / `.dt.weekofyear`) by using the supported ISO calendar week extraction, keeping the same feature intent. I also fix the input file paths to match your provided Kaggle environment (`/kaggle/input/...`) so the notebook runs end-to-end. To move RMSE drastically toward your target with minimal logic change, I keep the same models and ensemble but add the already-created datetime features into `feature_names` (they were computed but never used), which is the main reason the current score is extremely poor. Finally, I ensure predictions align exactly to the `key` order in `sample_submission.csv` and write a valid `.csv` submission file.'
- What this solution (achieved 419.40651) has done: 'Your current RMSE is far worse than the target, so we need a real but still minimal improvement without changing the overall approach (feature engineering + simple model ensemble). The biggest lift with minimal logic change is to fix the obviously broken coordinate filters (longitude bounds mistakenly allow +75, which keeps invalid points) and to add one robust geographic feature (haversine distance) while keeping the same models and averaging scheme. I also make the datetime parsing consistent and drop rows with missing engineered features to avoid silent bad training. These changes should drastically reduce the error toward your target while preserving the same training/prediction pipeline and producing the same submission format.'
- What this solution (achieved 419.39374) has done: 'Your current RMSE is far worse than the target, so we need a meaningful but still “same approach” fix: the main culprit is that unscaled KNN on mixed-scale features (day_of_year/haversine/etc.) typically behaves very badly and can dominate the ensemble. To preserve the exact same model set and averaging scheme, I only add a `StandardScaler` for KNN via an sklearn `Pipeline`, leaving LinearRegression/RandomForest untouched. I also add a tiny bit of robustness by filling any remaining NaNs in test engineered features with training medians (to avoid silent NaNs causing pathological predictions), while keeping the same feature set and submission alignment. These changes should substantially reduce RMSE toward your target without changing the overall logic.'
- What this solution (achieved 5.61387) has done: 'Your current RMSE is far worse than the target (lower is better), so we need a meaningful fix while keeping the same overall pipeline (same engineered features and the same 3-model ensemble/averaging). The smallest high-impact change is to prevent KNN from harming the ensemble by using distance weighting and a more appropriate neighbor count, which preserves the same KNeighborsRegressor approach but makes it behave sensibly on this task. I also add a very light clip of extreme haversine distances (training + test) to reduce the effect of rare bad coordinates that slip through the bounding-box filter, without changing feature definitions. Everything else (paths, features, models used, and submission writing/alignment) stays the same and still produces a valid `.csv`.'
- What this solution (achieved 5.61144) has done: 'To reduce RMSE from 5.61 toward your 4.73 target (lower is better) without changing the overall feature set or the 3-model averaging approach, I make two minimal, high-impact adjustments. First, I fix the RandomForest hyperparameters to be more sensible for this dataset size by setting a moderate number of trees and a non-trivial `min_samples_leaf` (still the same RandomForest model, just less noisy and typically better generalization). Second, I add a very small, metric-aligned post-processing step: clip predictions to a realistic fare range based on the cleaned training labels (instead of only lower-bounding at 0), which usually trims a few extreme errors that hurt RMSE. Everything else (data loading, features, models used, ensemble formula, and submission alignment) stays intact and it still writes a valid `.csv`.'
- What this solution (achieved 5.42652) has done: 'To move your RMSE down from 5.611 toward the 4.7309 target with minimal disruption, I keep your exact feature set and 3-model ensemble, but make the RandomForest less biased and better at fitting continuous fares by (1) increasing `n_estimators` modestly and (2) removing the current `min_samples_leaf=2` constraint (which can underfit on this task). I also clamp `passenger_count` in the test set to the same valid range used in training so the models don’t see out-of-distribution passenger counts at inference. Finally, I replace the broad `[0, 200]` prediction clip with a slightly tighter, training-derived clip (using robust percentiles) to reduce RMSE impact from rare extreme predictions without changing the learning objective.'

# 9. Code solution

## === cell 0
import pandas as pd

TRAIN_PATH = "/kaggle/input/train.csv"
TEST_PATH = "/kaggle/input/test.csv"
SAMPLE_SUB_PATH = "/kaggle/input/sample_submission.csv"

train = pd.read_csv(TRAIN_PATH, nrows=300_000)
test = pd.read_csv(TEST_PATH)



## === cell 1
train.shape



## === cell 2
train.head()



## === cell 3
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt



## === cell 4
train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], errors="coerce", utc=True
)
train["hour"] = train["pickup_datetime"].dt.hour
train["day"] = train["pickup_datetime"].dt.day
train["week"] = train["pickup_datetime"].dt.isocalendar().week.astype("int16")
train["month"] = train["pickup_datetime"].dt.month
train["day_of_year"] = train["pickup_datetime"].dt.dayofyear
train["week_of_year"] = train["pickup_datetime"].dt.isocalendar().week.astype("int16")



## === cell 5
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], errors="coerce", utc=True
)
test["hour"] = test["pickup_datetime"].dt.hour
test["day"] = test["pickup_datetime"].dt.day
test["week"] = test["pickup_datetime"].dt.isocalendar().week.astype("int16")
test["month"] = test["pickup_datetime"].dt.month
test["day_of_year"] = test["pickup_datetime"].dt.dayofyear
test["week_of_year"] = test["pickup_datetime"].dt.isocalendar().week.astype("int16")



## === cell 6
train.head()
train = train.dropna(how="any", axis="rows")



## === cell 7
train = train.loc[(train["fare_amount"] > 0) & (train["fare_amount"] < 200)]
train = train.loc[(train["pickup_longitude"] > -75) & (train["pickup_longitude"] < -72)]
train = train.loc[(train["pickup_latitude"] > 40) & (train["pickup_latitude"] < 45)]
train = train.loc[
    (train["dropoff_longitude"] > -75) & (train["dropoff_longitude"] < -72)
]
train = train.loc[(train["dropoff_latitude"] > 40) & (train["dropoff_latitude"] < 45)]
train = train.loc[train["passenger_count"].between(1, 8)]

test["passenger_count"] = pd.to_numeric(test["passenger_count"], errors="coerce")
test["passenger_count"] = (
    test["passenger_count"].clip(lower=1, upper=8).fillna(1).astype(int)
)



## === cell 8
train["abs_diff_longitude"] = (
    train["pickup_longitude"] - train["dropoff_longitude"]
).abs()
train["abs_diff_latitude"] = (
    train["pickup_latitude"] - train["dropoff_latitude"]
).abs()



## === cell 9
test["abs_diff_longitude"] = (
    test["pickup_longitude"] - test["dropoff_longitude"]
).abs()
test["abs_diff_latitude"] = (test["pickup_latitude"] - test["dropoff_latitude"]).abs()



## === cell 10
import numpy as np


def haversine_km(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1.astype(float))
    lat1 = np.radians(lat1.astype(float))
    lon2 = np.radians(lon2.astype(float))
    lat2 = np.radians(lat2.astype(float))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    R = 6371.0
    return R * c


train["haversine_km"] = haversine_km(
    train["pickup_longitude"],
    train["pickup_latitude"],
    train["dropoff_longitude"],
    train["dropoff_latitude"],
)
test["haversine_km"] = haversine_km(
    test["pickup_longitude"],
    test["pickup_latitude"],
    test["dropoff_longitude"],
    test["dropoff_latitude"],
)

train["haversine_km"] = train["haversine_km"].clip(lower=0.0, upper=200.0)
test["haversine_km"] = test["haversine_km"].clip(lower=0.0, upper=200.0)

train = train.replace([np.inf, -np.inf], np.nan).dropna(axis=0)



## === cell 11
train.head()



## === cell 12
train.head()



## === cell 13
_ = sns.barplot(data=train, x="passenger_count", y="fare_amount")



## === cell 14
feature_names = [
    "abs_diff_longitude",
    "abs_diff_latitude",
    "haversine_km",
    "passenger_count",
    "hour",
    "day",
    "week",
    "month",
    "day_of_year",
    "week_of_year",
]
feature_names



## === cell 15
label_name = "fare_amount"
label_name



## === cell 16
X_train = train[feature_names]
y_train = train[label_name]
X_test = test[feature_names]

X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)
medians = X_train.median(numeric_only=True)
X_train = X_train.fillna(medians)
X_test = X_test.fillna(medians)



## === cell 17
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb



## === cell 18
regr = LinearRegression()
regr.fit(X_train, y_train)
regr_prediction = regr.predict(X_test)



## === cell 19
knr = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "knn",
            KNeighborsRegressor(
                n_neighbors=30, weights="distance", metric="minkowski", p=2
            ),
        ),
    ]
)
knr.fit(X_train, y_train)
knr_prediction = knr.predict(X_test)



## === cell 20
rfr = RandomForestRegressor(
    random_state=42,
    n_jobs=-1,
    n_estimators=350,
    min_samples_leaf=1,
)
rfr.fit(X_train, y_train)
rfr_prediction = rfr.predict(X_test)



## === cell 21
predictions = (regr_prediction * 2 + rfr_prediction * 2 + knr_prediction) / 5



## === cell 22
predictions



## === cell 23
submission = pd.read_csv(SAMPLE_SUB_PATH)
pred_df = pd.DataFrame({"key": test["key"].values, "fare_amount": predictions})

submission = submission[["key"]].merge(pred_df, on="key", how="left")
submission["fare_amount"] = submission["fare_amount"].fillna(float(y_train.mean()))

lo = float(y_train.quantile(0.001))
hi = float(y_train.quantile(0.999))
submission["fare_amount"] = submission["fare_amount"].clip(lower=lo, upper=hi)



## === cell 24
submission.head()



## === cell 25
submission.to_csv("./simplenewyorktaxi.csv", index=False)
print("Wrote submission:", "./simplenewyorktaxi.csv", "rows:", len(submission))
