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

3.14

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

4.736901452993294

# 6. Current score

360.22256

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 360.22067) has done: 'Your huge RMSE is coming from a feature mismatch between train and test: in training you include `pickup_datetime` (as raw strings) but in test you drop it and also shift columns (`iloc[:,2:]`), so the model is trained on a different set/order of features than it sees at inference. I make train/test use the exact same feature columns in the same order, and I convert `pickup_datetime` into numeric time features (year/month/day/hour/weekday) in both sets so scikit-learn can learn from it without breaking. I also ensure the `key` column is preserved for submission while never used as a feature, and keep the rest of your filtering, haversine feature, and VotingRegressor core logic intact. These minimal fixes should move RMSE dramatically down toward your target (lower is better) without changing your modeling approach.'
- What this solution (achieved 360.22256) has done: 'Your current score is far worse than the target (lower-is-better), so we should improve performance while keeping your core model/feature logic intact. The biggest safe gain here is to ensure the datetime-derived features are numeric (not pandas nullable Int dtypes that can cause dtype/object promotion in some sklearn paths) and that train/test feature matrices have identical dtypes and no NaNs after datetime parsing. I also make the BaggingRegressor parameter compatible with sklearn 1.2.2 (`base_estimator` instead of `estimator`) to avoid silent incompatibilities, and I keep your filtering, haversine feature, and VotingRegressor approach unchanged. These are minimal, execution-safe changes that should substantially reduce RMSE toward your target by removing feature/dtype issues without changing the modeling approach.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

warnings.filterwarnings("ignore")



## === cell 2
TRAIN_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
TEST_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"



## === cell 3
df_train = pd.read_csv(TRAIN_PATH, nrows=2_000_000, parse_dates=["pickup_datetime"])
df_train.head()



## === cell 4
df_train.head()



## === cell 5
df_train.info()



## === cell 6
df_train.shape



## === cell 7
df_train.describe()



## === cell 8
df_train.isna().sum()



## === cell 9
df_train = df_train.dropna()



## === cell 10
df_train = df_train[(df_train["fare_amount"] > 1) & (df_train["fare_amount"] < 100)]



## === cell 11
df_train = df_train[
    (df_train["passenger_count"] >= 1) & (df_train["passenger_count"] <= 6)
]



## === cell 12
df_train = df_train[
    (df_train["pickup_longitude"] > -75)
    & (df_train["pickup_longitude"] < -72)
    & (df_train["dropoff_longitude"] > -75)
    & (df_train["dropoff_longitude"] < -72)
    & (df_train["pickup_latitude"] > 40)
    & (df_train["pickup_latitude"] < 42)
    & (df_train["dropoff_latitude"] > 40)
    & (df_train["dropoff_latitude"] < 42)
]




## === cell 13
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c  # kilometers




## === cell 14
df_train["distance_km"] = haversine(
    df_train["pickup_latitude"],
    df_train["pickup_longitude"],
    df_train["dropoff_latitude"],
    df_train["dropoff_longitude"],
)
df_train = df_train[(df_train["distance_km"] > 0.1) & (df_train["distance_km"] < 30)]




## === cell 15
def add_time_features(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df = df.copy()
    df["pickup_year"] = dt.dt.year.astype(np.int16)
    df["pickup_month"] = dt.dt.month.astype(np.int8)
    df["pickup_day"] = dt.dt.day.astype(np.int8)
    df["pickup_hour"] = dt.dt.hour.astype(np.int8)
    df["pickup_weekday"] = dt.dt.weekday.astype(np.int8)
    return df


df_train = add_time_features(df_train)
df_train = df_train.dropna(
    subset=[
        "pickup_year",
        "pickup_month",
        "pickup_day",
        "pickup_hour",
        "pickup_weekday",
    ]
)



## === cell 16
feature_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "distance_km",
    "pickup_year",
    "pickup_month",
    "pickup_day",
    "pickup_hour",
    "pickup_weekday",
]

X = df_train[feature_cols]
y = df_train["fare_amount"]

X.head(), y.head()



## === cell 17
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=2,
    bootstrap=True,
    max_samples=5000,
    n_jobs=1,
)



## === cell 18
from sklearn.linear_model import LinearRegression

lr = LinearRegression()



## === cell 19
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor

sr = SVR(kernel="rbf")
br = BaggingRegressor(
    n_estimators=10, base_estimator=sr, max_samples=5000, bootstrap=True
)



## === cell 20
from sklearn.ensemble import VotingRegressor

vr = VotingRegressor([("lr", lr), ("rf", rf), ("svr", br)], verbose=True, n_jobs=-1)

vr.fit(X, y)



## === cell 21
df_test = pd.read_csv(TEST_PATH, parse_dates=["pickup_datetime"])
df_test.head()



## === cell 22
key = df_test["key"].copy()



## === cell 23
df_test["distance_km"] = haversine(
    df_test["pickup_latitude"],
    df_test["pickup_longitude"],
    df_test["dropoff_latitude"],
    df_test["dropoff_longitude"],
)

df_test = add_time_features(df_test)

for c in ["pickup_year", "pickup_month", "pickup_day", "pickup_hour", "pickup_weekday"]:
    if df_test[c].isna().any():
        df_test[c] = (
            df_test[c]
            .fillna(df_test[c].mode(dropna=True).iloc[0])
            .astype(df_test[c].dtype)
        )



## === cell 24
X_test = df_test[feature_cols]



## === cell 25
y_pred = vr.predict(X_test)



## === cell 26
y_pred = np.clip(y_pred, 0, None)



## === cell 27
results = pd.DataFrame({"key": key, "fare_amount": y_pred})
print(results.head())
print("Submission rows:", len(results))



## === cell 28
results.to_csv("submission.csv", index=False)
print("Wrote submission.csv")
