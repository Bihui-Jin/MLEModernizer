# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

No external packages required in the script and installed.

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

4.203419222523976

# 6. Current score

1096.2251

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 1096.2251) has done: 'I remove the unavailable `feather` dependency and stop trying to write/read a feather file, loading the training data directly instead. I fix the Pandas `.str.split()` API error by using keyword arguments so datetime parsing works on the current Kaggle/Pandas version. I also make the notebook run end-to-end by ensuring `df` is defined before it’s used, fixing indentation/cell-order issues, and keeping feature names consistent (`diff_lon`/`diff_long`) between train and test. Finally, I ensure a valid submission CSV with exactly `key,fare_amount` is written (keeping the same core linear regression and random forest modeling logic).'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import math
from scipy import stats as st
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestRegressor as rf
from sklearn.linear_model import LinearRegression

plt.style.use("seaborn-whitegrid")

TRAIN_PATH = "../input/new-york-city-taxi-fare-prediction/train.csv"
TEST_PATH = "../input/new-york-city-taxi-fare-prediction/test.csv"



## === cell 1
df = pd.read_csv(TRAIN_PATH, nrows=1_000_000, low_memory=True)
print(df.head())
print(df.shape)



## === cell 2
df = df[df.passenger_count > 0]

df = df[df.dropoff_latitude != 0]
df = df[df.pickup_longitude != 0]
df = df[df.pickup_latitude != 0]
df = df[df.dropoff_longitude != 0]

df = df[df.fare_amount > 2.5]
df = df[df.fare_amount < 100]

df = df.dropna()

dt_parts = df["pickup_datetime"].astype(str).str.split(" ", n=1, expand=True)
df["year"] = dt_parts[0].str[:4]
df["hour"] = dt_parts[1].str[:2]

df = df.dropna()



## === cell 3
df[["year", "hour"]] = df[["year", "hour"]].apply(pd.to_numeric, errors="coerce")
df = df.dropna(subset=["year", "hour"])
print(df.head())




## === cell 4
def select_within_newYork(df_in, BB):
    return (
        (df_in.pickup_longitude >= BB[0])
        & (df_in.pickup_longitude <= BB[1])
        & (df_in.pickup_latitude >= BB[2])
        & (df_in.pickup_latitude <= BB[3])
        & (df_in.dropoff_longitude >= BB[0])
        & (df_in.dropoff_longitude <= BB[1])
        & (df_in.dropoff_latitude >= BB[2])
        & (df_in.dropoff_latitude <= BB[3])
    )


NYC = (-74.5, -72.8, 40.5, 41.8)
df = df[select_within_newYork(df, NYC)].copy()
print(df.shape)




## === cell 5
def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    miles = 6367 * c * 0.62137
    return miles


df["distance"] = haversine_np(
    df.pickup_longitude, df.pickup_latitude, df.dropoff_longitude, df.dropoff_latitude
)

print(df[["fare_amount", "distance"]].head())



## === cell 6
print("Co-relation b/w Fare and Distance")
print(st.pearsonr(df.distance, df.fare_amount))
print(df["distance"].corr(df["fare_amount"], method="pearson"))
df = df[df.distance <= 30].copy()
print(df.shape)



## === cell 7
try:
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    con = (
        (df.distance < 30)
        & (df.distance > 0.5)
        & (df.fare_amount > 0)
        & (df.fare_amount < 200)
    )
    axs[0].scatter(df[con].distance, df[con].fare_amount, alpha=0.3)
    axs[0].set_xlabel("Distance")
    axs[0].set_ylabel("Fare")
    axs[0].set_title("Distance vs Fare")
    plt.close(fig)
except Exception as e:
    print("Plotting skipped:", repr(e))



## === cell 8
df["diff_lon"] = (df.dropoff_longitude - df.pickup_longitude).abs()
df["diff_lat"] = (df.dropoff_latitude - df.pickup_latitude).abs()

df = df.dropna(
    subset=["distance", "diff_lat", "diff_lon", "passenger_count", "fare_amount"]
).copy()
print(df[["diff_lon", "diff_lat", "distance"]].head())



## === cell 9
test = pd.read_csv(TEST_PATH, low_memory=True)
print(test.head())

test["diff_lat"] = (test.dropoff_latitude - test.pickup_latitude).abs()
test["diff_long"] = (test.dropoff_longitude - test.pickup_longitude).abs()
test["diff_lon"] = test["diff_long"]

test["distance"] = haversine_np(
    test.pickup_longitude,
    test.pickup_latitude,
    test.dropoff_longitude,
    test.dropoff_latitude,
)

dt_parts_t = test["pickup_datetime"].astype(str).str.split(" ", n=1, expand=True)
test["year"] = dt_parts_t[0].str[:4]
test["hour"] = dt_parts_t[1].str[:2]
test[["year", "hour"]] = test[["year", "hour"]].apply(pd.to_numeric, errors="coerce")

test_id = list(test["key"])
print(test.describe(include="all").transpose().head(20))



## === cell 10
lr = LinearRegression()
lr_features = [
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
    "distance",
    "passenger_count",
]

lr.fit(df[lr_features], df["fare_amount"])

print("Intercept", round(lr.intercept_, 4))
print("Coef sample:", [round(c, 6) for c in lr.coef_])

preds_lr = lr.predict(test[lr_features])

sub_lr = pd.DataFrame({"key": test_id, "fare_amount": preds_lr})
sub_lr.to_csv("output_lr.csv", index=False)
print("Wrote output_lr.csv", sub_lr.shape, sub_lr.head())



## === cell 11
random_forest = rf(
    n_estimators=20,
    max_depth=20,
    max_features=None,
    oob_score=True,
    bootstrap=True,
    verbose=0,
    n_jobs=-1,
    random_state=42,  # determinism; score-neutral in expectation
)

rf_features_train = ["distance", "diff_lat", "diff_lon", "passenger_count"]
random_forest.fit(df[rf_features_train], df["fare_amount"])

rf_features_test = ["distance", "diff_lat", "diff_long", "passenger_count"]
preds_rf = random_forest.predict(test[rf_features_test])

sub_rf = pd.DataFrame({"key": test_id, "fare_amount": preds_rf})
sub_rf.to_csv("submission.csv", index=False)
print("Wrote submission.csv", sub_rf.shape, sub_rf.head())

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/4143798894.py in <cell line: 0>()
     15 
     16 rf_features_test = ["distance", "diff_lat", "diff_long", "passenger_count"]
---> 17 preds_rf = random_forest.predict(test[rf_features_test])
     18 
     19 sub_rf = pd.DataFrame({"key": test_id, "fare_amount": preds_rf})

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_forest.py in predict(self, X)
    979         check_is_fitted(self)
    980         # Check data
--> 981         X = self._validate_X_predict(X)
    982 
    983         # Assign chunk of trees to jobs

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_forest.py in _validate_X_predict(self, X)
    600         Validate X whenever one tries to predict, apply, predict_proba."""
    601         check_is_fitted(self)
--> 602         X = self._validate_data(X, dtype=DTYPE, accept_sparse="csr", reset=False)
    603         if issparse(X) and (X.indices.dtype != np.intc or X.indptr.dtype != np.intc):
    604             raise ValueError("No support for np.int64 index based sparse matrices")

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    546             validated.
    547         """
--> 548         self._check_feature_names(X, reset=reset)
    549 
    550         if y is None and self._get_tags()["requires_y"]:

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _check_feature_names(self, X, reset)
    479                 )
    480 
--> 481             raise ValueError(message)
    482 
    483     def _validate_data(

ValueError: The feature names should match those that were passed during fit.
Feature names unseen at fit time:
- diff_long
Feature names seen at fit time, yet now missing:
- diff_lon
