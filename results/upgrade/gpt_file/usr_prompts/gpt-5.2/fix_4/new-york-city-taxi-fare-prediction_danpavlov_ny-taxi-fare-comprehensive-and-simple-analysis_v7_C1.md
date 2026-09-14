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
geopy==2.4.1
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

9.1952

# 6. Current score

524.29125

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 18.86722) has done: 'I (1) remove/guard notebook-only and plotting cells that currently crash in Kaggle’s script environment (e.g., `%matplotlib inline` and the `sns.lmplot` call with the old positional-args signature), so the pipeline runs end-to-end. I (2) fix the extremely slow/buggy distance feature creation that iterates row-by-row and likely leaves `distance` missing or takes too long, by computing the same great-circle (haversine) distance vectorized (same feature intent, but correct and fast). I (3) ensure train/test feature columns match (including `month`/`year`) and keep the same LinearRegression training approach, but stop dropping the useful time features so the RMSE moves toward the 9.1952 target. Finally, I write a valid `submission.csv` with the required columns and `.csv` suffix.'
- What this solution (achieved 524.29125) has done: 'Your current RMSE (18.86722) is much worse than the target (9.1952), so we should improve score with minimal, low-risk changes that preserve the same LinearRegression approach and features. The biggest likely issue is that LinearRegression can produce negative/too-small fares for short/odd trips, which heavily hurts RMSE; we clamp predictions to a sensible lower bound based on the training distribution (still the same model, just safer post-processing). We also add one minimal but highly standard geometry feature (`abs_delta_lon`, `abs_delta_lat`) derived from existing columns; this keeps the training loop and model unchanged while giving the linear model more signal than distance alone. Finally, we ensure train/test imputation uses the same medians from training features to avoid distribution mismatches.'
- What this solution (achieved 524.29125) has done: 'Your current public score (RMSE 524) indicates the submission is likely misaligned (predictions not matching the correct `key` rows), because this is far too large for a reasonable linear baseline. I make the smallest change that directly targets this: explicitly align the submission to the `sample_submission.csv` key order (and fill any missing predictions by key) to ensure row/key matching is correct. This preserves your exact model, features, and training logic, but prevents catastrophic scoring from any accidental ordering mismatch or hidden formatting issues. I also add a one-line sanity check on shape/NaNs before writing the CSV to avoid invalid submissions.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)



## === cell 1
print(os.listdir("../input"))



## === cell 2
test = pd.read_csv("../input/test.csv")
test.dtypes



## === cell 3
types = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}



## === cell 4
train = pd.read_csv("../input/train.csv", nrows=500000, dtype=types)
train.head()



## === cell 5
train.describe()



## === cell 6
try:
    _ = sns.histplot(train["fare_amount"], bins=100)
    plt.close()
except Exception:
    pass



## === cell 7
try:
    _ = sns.histplot(train["passenger_count"], bins=10)
    plt.close()
except Exception:
    pass



## === cell 8
train.isnull().sum()



## === cell 9
train.dropna(inplace=True)



## === cell 10
train = train[train["fare_amount"] > 0]
train = train[train["pickup_longitude"] < -72]
train = train[(train["pickup_latitude"] > 40) & (train["pickup_latitude"] < 44)]
train = train[train["dropoff_longitude"] < -72]
train = train[(train["dropoff_latitude"] > 40) & (train["dropoff_latitude"] < 44)]
train = train[(train["passenger_count"] > 0) & (train["passenger_count"] < 10)]



## === cell 11
train.describe()




## === cell 12
def add_distance_km(df: pd.DataFrame) -> pd.DataFrame:
    lat1 = np.radians(df["pickup_latitude"].astype("float64"))
    lon1 = np.radians(df["pickup_longitude"].astype("float64"))
    lat2 = np.radians(df["dropoff_latitude"].astype("float64"))
    lon2 = np.radians(df["dropoff_longitude"].astype("float64"))

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(a))
    R = 6371.0  # Earth radius in km
    df["distance"] = (R * c).astype("float32")
    return df


train = add_distance_km(train)
test = add_distance_km(test)



## === cell 13
train["pickup_datetime"] = (
    train["pickup_datetime"].astype(str).str.replace(" UTC", "", regex=False)
)
train["pickup_datetime"] = pd.to_datetime(train["pickup_datetime"], errors="coerce")

test["pickup_datetime"] = (
    test["pickup_datetime"].astype(str).str.replace(" UTC", "", regex=False)
)
test["pickup_datetime"] = pd.to_datetime(test["pickup_datetime"], errors="coerce")

train = train.dropna(subset=["pickup_datetime"])



## === cell 14
train["dayofweek"] = train.pickup_datetime.dt.dayofweek.astype("int16")
train["hour"] = train.pickup_datetime.dt.hour.astype("int16")
train["month"] = train.pickup_datetime.dt.month.astype("int16")
train["year"] = train.pickup_datetime.dt.year.astype("int16")

test["dayofweek"] = test.pickup_datetime.dt.dayofweek.astype("int16")
test["hour"] = test.pickup_datetime.dt.hour.astype("int16")
test["month"] = test.pickup_datetime.dt.month.astype("int16")
test["year"] = test.pickup_datetime.dt.year.astype("int16")

test.head()



## === cell 15
try:
    sns.lmplot(x="year", y="fare_amount", data=train[["year", "fare_amount"]])
    plt.close()
except Exception:
    pass



## === cell 16
try:
    _ = sns.heatmap(
        train.drop(
            [
                "key",
                "pickup_datetime",
                "pickup_longitude",
                "pickup_latitude",
                "dropoff_longitude",
                "dropoff_latitude",
            ],
            axis=1,
        ).corr(numeric_only=True)
    )
    plt.close()
except Exception:
    pass



## === cell 17
train["abs_delta_lon"] = (
    (train["dropoff_longitude"] - train["pickup_longitude"]).abs().astype("float32")
)
train["abs_delta_lat"] = (
    (train["dropoff_latitude"] - train["pickup_latitude"]).abs().astype("float32")
)
test["abs_delta_lon"] = (
    (test["dropoff_longitude"] - test["pickup_longitude"]).abs().astype("float32")
)
test["abs_delta_lat"] = (
    (test["dropoff_latitude"] - test["pickup_latitude"]).abs().astype("float32")
)

feature_cols = [
    "passenger_count",
    "distance",
    "abs_delta_lon",
    "abs_delta_lat",
    "dayofweek",
    "hour",
    "month",
    "year",
]

X = train[feature_cols].copy()
y = train["fare_amount"].astype("float32")

X = X.replace([np.inf, -np.inf], np.nan)

train_feature_medians = X.median(numeric_only=True)
X = X.fillna(train_feature_medians)



## === cell 18
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

lm = LinearRegression()
lm.fit(X_train, y_train)

print(lm.score(X_train, y_train))
print(lm.score(X_valid, y_valid))



## === cell 19
y_pred = lm.predict(X_valid)
lrmse = float(np.sqrt(metrics.mean_squared_error(y_valid, y_pred)))
lrmse




## === cell 20
def get_score(prediction, lables):
    print("R2: {}".format(r2_score(lables, prediction)))
    print("RMSE: {}".format(np.sqrt(mean_squared_error(lables, prediction))))


def train_test(estimator, x_trn, x_tst, y_trn, y_tst):
    prediction_train = estimator.predict(x_trn)
    print(estimator)
    get_score(prediction_train, y_trn)
    prediction_test = estimator.predict(x_tst)
    print("Test")
    get_score(prediction_test, y_tst)


train_test(lm, X_train, X_valid, y_train, y_valid)



## === cell 21
Xtest = test[feature_cols].copy()
Xtest = Xtest.replace([np.inf, -np.inf], np.nan)
Xtest = Xtest.fillna(train_feature_medians)

LinearPredictions = lm.predict(Xtest)

min_reasonable_fare = float(max(2.5, np.percentile(y.values, 0.5)))
LinearPredictions = np.maximum(LinearPredictions, min_reasonable_fare)

LinearPredictions = np.round(LinearPredictions, decimals=2)
LinearPredictions.size



## === cell 22
sample_sub = pd.read_csv("../input/sample_submission.csv")

pred_df = pd.DataFrame({"key": test["key"].values, "fare_amount": LinearPredictions})
pred_df = pred_df.drop_duplicates(subset=["key"], keep="last")

submission = sample_sub[["key"]].merge(pred_df, on="key", how="left")

fallback = float(np.round(np.median(y.values), 2))
submission["fare_amount"] = submission["fare_amount"].fillna(fallback).astype("float32")

assert (
    submission.shape[0] == sample_sub.shape[0]
), "Submission row count mismatch vs sample_submission."
assert (
    submission["fare_amount"].isna().sum() == 0
), "NaNs in fare_amount would invalidate submission."

submission.head()



## === cell 23
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print("fare_amount summary:", submission["fare_amount"].describe())
