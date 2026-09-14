# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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
lightgbm==4.6.0
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

# 5. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # CSV file I/O
import os
import time
import sys

import lightgbm as lgbm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

print(os.listdir("../input"))




## === cell 1
def incremental_learning(data, model):
    start = time.time()
    data["pickup_datetime"] = pd.to_datetime(
        data["pickup_datetime"], infer_datetime_format=True
    )
    print("Conversion took", time.time() - start, "s")
    add_travel_vector_features(data)
    data = pd.concat([data, timeExpansion(data["pickup_datetime"])], axis=1)

    X_tr, X_te, y_tr, y_te = train_test_split(
        data.drop(["key", "pickup_datetime", "fare_amount"], axis=1),
        data["fare_amount"],
        test_size=0.2,
        random_state=42,
    )

    params = {
        "boosting_type": "gbdt",
        "objective": "regression",  # use RMSE objective
        "metric": {"rmse"},  # evaluate with RMSE
        "num_leaves": 31,
        "learning_rate": 0.1,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }

    dset = lgbm.Dataset(X_tr, y_tr, free_raw_data=True)
    model = lgbm.train(
        params,
        init_model=model,
        train_set=dset,
        keep_training_booster=True,
        num_boost_round=50,  # more boosting rounds per chunk
    )
    mae = mean_absolute_error(y_te, model.predict(X_te))
    print("Mean Absolute Error (validation):", mae)
    return model




## === cell 2
def partial_import(filename, skip, rows):
    return pd.read_csv("../input/train.csv", skiprows=skip, nrows=rows)


def add_travel_vector_features(df):
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs()


def month_translation(number):
    month_list = [
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    ]
    return month_list[number - 1]


def quarter_translation(number):
    if 0 <= number < 6:
        return "Q1"
    elif 6 <= number < 12:
        return "Q2"
    elif 12 <= number < 18:
        return "Q3"
    else:
        return "Q4"


def week_translation(number):
    week_border = [0, 7, 14, 21, 31]
    week_list = ["1W", "2W", "3W", "4W"]
    for idx in range(1, len(week_border)):
        if week_border[idx - 1] < number <= week_border[idx]:
            return week_list[idx - 1]
    return "4W"


def timeExpansion(timeSeries):
    additional_cols = ["year", "month", "quarter", "day"]
    dummy0 = pd.DataFrame(
        np.zeros((len(timeSeries), len(additional_cols)), "int"),
        columns=additional_cols,
        index=timeSeries.index,
    )
    dummy0["year"] = [i.year for i in timeSeries]
    dummy0["month"] = [month_translation(i.month) for i in timeSeries]
    dummy0["day"] = [week_translation(i.day) for i in timeSeries]
    dummy0["quarter"] = [quarter_translation(i.hour) for i in timeSeries]
    return pd.get_dummies(dummy0)


def predict_and_submit(estimator, submit=True):
    test_df = pd.read_csv("../input/test.csv")
    add_travel_vector_features(test_df)
    test_df["pickup_datetime"] = pd.to_datetime(test_df["pickup_datetime"])
    test_X = pd.concat([test_df, timeExpansion(test_df["pickup_datetime"])], axis=1)
    test_X.drop(["key", "pickup_datetime"], axis=1, inplace=True)

    print("Test Shape:", test_X.shape)

    test_y_predictions = estimator.predict(test_X)

    if submit:
        submission = pd.DataFrame(
            {"key": test_df.key, "fare_amount": test_y_predictions},
            columns=["key", "fare_amount"],
        )
        submission.to_csv("submission.csv", index=False)




## === cell 3
training_step = list(range(0, 56000000, 5000000))  # cover the full file (~55 M rows)
estimator = None
origin_cols = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

for i in range(1, len(training_step)):
    start = time.time()
    print(f"Chunk {i}: rows {training_step[i-1]} to {training_step[i]}")
    train_df = pd.read_csv(
        "../input/train.csv", skiprows=training_step[i - 1], nrows=training_step[i]
    )
    train_df.columns = origin_cols
    estimator = incremental_learning(train_df, estimator)
    print("Chunk", i, "took", time.time() - start, "seconds\n")

predict_and_submit(estimator)
