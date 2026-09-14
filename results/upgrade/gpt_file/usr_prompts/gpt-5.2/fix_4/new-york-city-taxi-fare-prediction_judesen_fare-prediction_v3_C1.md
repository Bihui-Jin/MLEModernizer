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

47.55866

# 6. Current score

793.90146

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 846.01221) has done: 'I fix the pandas `DataFrame.drop` call to use keyword arguments compatible with pandas 2.2, which removes the first runtime error. I also correct the datetime feature engineering bug where you overwrite both train and test `pickup_datetime` using the test column, and ensure the model only sees numeric features (eliminating the Timestamp-related `fit` failure). Finally, I keep the same linear regression approach but make the pipeline robust (drop NaNs consistently, align columns) and write a valid `submission.csv` with the required `key,fare_amount` columns.'
- What this solution (achieved 702.92653) has done: 'Your RMSE is huge mainly because you train on only 10 rows (`nrows=10`), so the linear regression can’t learn a meaningful mapping and outputs wildly wrong fares. Keeping the exact same model and feature engineering, the smallest change is to train on a larger (but still manageable) subset of `train.csv` and add a couple of standard NYC Taxi sanity filters (valid lat/lon/passenger/fare ranges) that prevent extreme outliers from dominating RMSE. This preserves the core logic (linear regression on simple engineered numeric features) while moving the score sharply downward toward your target band. The submission format and paths remain the same, and it still run within the time limit.'
- What this solution (achieved 793.90146) has done: 'Your current score is far worse than the target (lower-is-better), so we should improve it with minimal, metric-aligned fixes while keeping the same linear regression and feature set. The biggest remaining issue is that your “distance_travelled” is not geographically meaningful (Euclidean degrees * 100), which makes the model extrapolate poorly; switching to a proper Haversine distance keeps the exact same feature (distance) but makes it predictive. We also add minimal, standard NYC coordinate sanity filters on the training subset (without changing the learning approach) so outliers don’t dominate the RMSE. Finally, we clip negative predictions to 0 (fares can’t be negative), which typically reduces RMSE on this competition without changing the model.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import datetime as dt

from sklearn.linear_model import LinearRegression

DATA_DIR = "/kaggle/input"

TRAIN_NROWS = (
    2_000_000  # keep same approach (subset training) but with enough data to learn
)

train = pd.read_csv(f"{DATA_DIR}/train.csv", nrows=TRAIN_NROWS)
test = pd.read_csv(f"{DATA_DIR}/test.csv")

combine = [train, test]

test.head(3)
test.dtypes




## === cell 1
def haversine_km(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1.astype(float))
    lat1 = np.radians(lat1.astype(float))
    lon2 = np.radians(lon2.astype(float))
    lat2 = np.radians(lat2.astype(float))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return 6371.0 * c  # Earth radius in km


for dataset in combine:
    dataset["distance_travelled"] = haversine_km(
        dataset["pickup_longitude"],
        dataset["pickup_latitude"],
        dataset["dropoff_longitude"],
        dataset["dropoff_latitude"],
    )

    dataset["pickup_datetime"] = pd.to_datetime(
        dataset["pickup_datetime"], errors="coerce", utc=False
    )
    dataset["hour_of_day"] = dataset["pickup_datetime"].dt.hour
    dataset["month"] = dataset["pickup_datetime"].dt.month

train_features_to_keep = [
    "key",
    "passenger_count",
    "distance_travelled",
    "hour_of_day",
    "month",
    "fare_amount",
]
train.drop(columns=train.columns.difference(train_features_to_keep), inplace=True)

test_features_to_keep = [
    "key",
    "passenger_count",
    "distance_travelled",
    "hour_of_day",
    "month",
]
test.drop(columns=test.columns.difference(test_features_to_keep), inplace=True)

train = train.dropna().copy()

train = train[
    (train["fare_amount"] > 0)
    & (train["fare_amount"] <= 300)
    & (train["passenger_count"] >= 1)
    & (train["passenger_count"] <= 6)
    & (train["distance_travelled"] >= 0)
    & (train["distance_travelled"] <= 200)  # km; generous upper bound for NYC area
].copy()

test = test.dropna().copy()



## === cell 2
x_train = train.drop(["fare_amount", "key"], axis=1)
y_train = train["fare_amount"].astype(float)

x_test = test.drop("key", axis=1)

x_train = x_train.apply(pd.to_numeric, errors="coerce")
x_test = x_test.apply(pd.to_numeric, errors="coerce")

x_train = x_train.dropna()
y_train = y_train.loc[x_train.index]

x_test = x_test.fillna(x_train.median(numeric_only=True))



## === cell 3
regr = LinearRegression()
regr.fit(x_train, y_train)
prediction = regr.predict(x_test)

prediction = np.clip(prediction, 0, None)



## === cell 4
submission = pd.DataFrame({"key": test["key"], "fare_amount": np.round(prediction, 2)})

submission.to_csv("submission.csv", index=False)

submission.head()
