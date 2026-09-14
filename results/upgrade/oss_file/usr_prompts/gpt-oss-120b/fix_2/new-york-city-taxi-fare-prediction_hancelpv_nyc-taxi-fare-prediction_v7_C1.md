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

3.89954

# 6. Current score

5.05247

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 5.05247) has done: 'I fixed the deprecated pandas attribute, removed the remaining datetime column after feature extraction, and ensured the feature list matches the updated dataframe so the model can be trained and a valid `submission.csv` is written.'

# 9. Code solution

## === cell 0
import os, math
import numpy as np
import pandas as pd

print(os.listdir("../input"))




## === cell 1
train = pd.read_csv("../input/train.csv", nrows=1_000_000)
test = pd.read_csv("../input/test.csv")
sample_sub = pd.read_csv("../input/sample_submission.csv")




## === cell 2
train = train.dropna(how="any", axis="rows")




## === cell 3
all_data = pd.concat((train, test)).reset_index(drop=True)

all_data.drop(["fare_amount"], axis=1, inplace=True)

y = train["fare_amount"].values
n_train = len(train)
test_id = test["key"]




## === cell 4
def week_num(day):
    """Return the week number label of a month for a given day of month."""
    if day <= 7:
        return "first"
    if day <= 14:
        return "second"
    if day <= 21:
        return "third"
    if day <= 28:
        return "fourth"
    return "fifth"




## === cell 5
def add_time_features(df):
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

    df["hour"] = df["pickup_datetime"].dt.hour.astype(str)
    df["day_of_week"] = df["pickup_datetime"].dt.day_name()
    df["day_of_month"] = df["pickup_datetime"].dt.day
    df["week_of_month"] = df["day_of_month"].map(week_num)
    df["month"] = df["pickup_datetime"].dt.month.astype(str)
    df["year"] = df["pickup_datetime"].dt.year.astype(str)

    df.drop(["day_of_month", "pickup_datetime"], axis=1, inplace=True)
    return df




## === cell 6
def add_geo_features(df):
    df["abs_diff_longitude"] = (df["dropoff_longitude"] - df["pickup_longitude"]).abs()
    df["abs_diff_latitude"] = (df["dropoff_latitude"] - df["pickup_latitude"]).abs()

    df["manhattan_distance"] = df["abs_diff_longitude"] + df["abs_diff_latitude"]

    df["squared_long"] = np.power(df["abs_diff_longitude"], 2)
    df["squared_lat"] = np.power(df["abs_diff_latitude"], 2)

    df["euclid_disance"] = np.sqrt(df["squared_long"] + df["squared_lat"])
    return df




## === cell 7
all_data = add_time_features(all_data)
all_data = add_geo_features(all_data)




## === cell 8
features = [
    "passenger_count",
    "hour",
    "day_of_week",
    "week_of_month",
    "month",
    "year",
    "abs_diff_longitude",
    "abs_diff_latitude",
    "euclid_disance",
    "manhattan_distance",
]

all_data = all_data[features]

all_data = pd.get_dummies(all_data)




## === cell 9
X_train = all_data[:n_train]
X_test = all_data[n_train:]




## === cell 10
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=100,
    max_features=3,
    min_samples_leaf=4,
    min_samples_split=10,
    bootstrap=True,
    random_state=42,
    n_jobs=-1,
)




## === cell 11
model.fit(X_train, y)




## === cell 12
test_pred = model.predict(X_test)




## === cell 13
submission = pd.DataFrame({"key": test_id, "fare_amount": test_pred})
submission.to_csv("submission.csv", index=False)
