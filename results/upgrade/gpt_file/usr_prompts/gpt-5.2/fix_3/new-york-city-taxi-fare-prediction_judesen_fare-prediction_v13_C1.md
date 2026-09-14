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

4.68065

# 6. Current score

5.65815

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.48136) has done: 'I fix the pandas datetime feature extraction that fails on newer pandas (`.dt.week`), and correct a bug that overwrote the train datetime using the test column. I also fix dataframe API changes (`drop(..., axis=1)` positional argument), ensure correlation plotting only uses numeric columns, and make sure only numeric features are passed into XGBoost so DMatrix construction succeeds. Finally, I update the deprecated XGBoost objective (`reg:linear`) to `reg:squarederror` (same RMSE intent) and ensure the notebook always writes a valid `sub_fare.csv` submission with `key,fare_amount`.'
- What this solution (achieved 5.65815) has done: 'To move RMSE down toward 4.68065 (lower is better), the smallest safe gain is to fix a bug in your engineered “distance_travelled” feature (it currently computes \|dx\|\*\|dy\| instead of Euclidean distance) while preserving the rest of your pipeline and XGBoost training approach. I also add the common NYC Taxi baseline improvement of predicting on `log1p(fare_amount)` and converting back with `expm1` (same regressor, same loss family, just a target transform), which typically reduces RMSE without changing your core modeling setup. Finally, I remove early stopping (it can underfit and also violates your “no early stopping” constraint) and instead train the fixed number of boosting rounds you already specify, keeping everything else stable and still writing `sub_fare.csv` in the required format.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import datetime as dt

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import xgboost as xgb

TRAIN_PATH = "/kaggle/input/train.csv"
TEST_PATH = "/kaggle/input/test.csv"

train = pd.read_csv(TRAIN_PATH, nrows=10_000_000)
test = pd.read_csv(TEST_PATH)

train.dtypes



## === cell 1
print("Sum of NaN values for each column")
print(train.isnull().sum())

train = train.dropna()
print("Sum of NaN values for each column after dropping NaN")
print(train.isnull().sum())



## === cell 2
train.describe()



## === cell 3
train = train.loc[(train["fare_amount"] > 0) & (train["fare_amount"] < 200)]
train = train.loc[(train["pickup_longitude"] > -150) & (train["pickup_longitude"] < 0)]
train = train.loc[(train["pickup_latitude"] > 0) & (train["pickup_latitude"] < 80)]
train = train.loc[
    (train["dropoff_longitude"] > -150) & (train["dropoff_longitude"] < 0)
]
train = train.loc[(train["dropoff_latitude"] > 0) & (train["dropoff_latitude"] < 80)]
train = train.loc[train["passenger_count"] <= 8]
train.describe()



## === cell 4
combine = [train, test]
for dataset in combine:
    dataset["longitude_distance"] = (
        dataset["pickup_longitude"] - dataset["dropoff_longitude"]
    ).abs()
    dataset["latitude_distance"] = (
        dataset["pickup_latitude"] - dataset["dropoff_latitude"]
    ).abs()

    dist_basic = np.sqrt(
        dataset["longitude_distance"] ** 2 + dataset["latitude_distance"] ** 2
    )
    dataset["distance_travelled"] = dist_basic
    dataset["distance_travelled_sin"] = np.sin(dist_basic)
    dataset["distance_travelled_cos"] = np.cos(dist_basic)
    dataset["distance_travelled_sin_sqrd"] = np.sin(dist_basic) ** 2
    dataset["distance_travelled_cos_sqrd"] = np.cos(dist_basic) ** 2

    R = 6371e3  # metres
    phi1 = np.radians(dataset["pickup_latitude"])
    phi2 = np.radians(dataset["dropoff_latitude"])
    dphi = np.radians(dataset["dropoff_latitude"] - dataset["pickup_latitude"])
    dlambda = np.radians(dataset["dropoff_longitude"] - dataset["pickup_longitude"])

    a = (np.sin(dphi / 2) ** 2) + np.cos(phi1) * np.cos(phi2) * (
        np.sin(dlambda / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    dataset["haversine"] = R * c

    y = np.sin(dlambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlambda)
    dataset["bearing"] = np.arctan2(y, x)

    dataset["pickup_datetime"] = pd.to_datetime(
        dataset["pickup_datetime"], errors="coerce", utc=True
    )
    dataset["hour_of_day"] = dataset["pickup_datetime"].dt.hour
    dataset["day"] = dataset["pickup_datetime"].dt.day
    dataset["month"] = dataset["pickup_datetime"].dt.month
    dataset["day_of_year"] = dataset["pickup_datetime"].dt.dayofyear

    isocal = dataset["pickup_datetime"].dt.isocalendar()
    dataset["week"] = isocal.week.astype("int16")
    dataset["week_of_year"] = isocal.week.astype("int16")

train.head(3)



## === cell 5
colormap = plt.cm.RdBu
plt.figure(figsize=(20, 20))
plt.title("Pearson Correlation of Features", y=1.05, size=15)
numeric_train = train.select_dtypes(include=[np.number])
sns.heatmap(
    numeric_train.corr(),
    linewidths=0.1,
    vmax=1.0,
    square=True,
    cmap=colormap,
    linecolor="white",
    annot=False,  # keep runtime reasonable
)
plt.show()



## === cell 6
train_features_to_keep = ["haversine", "distance_travelled_sin", "fare_amount"]
train.drop(columns=train.columns.difference(train_features_to_keep), inplace=True)

test_features_to_keep = ["haversine", "distance_travelled_sin", "key"]
test.drop(columns=test.columns.difference(test_features_to_keep), inplace=True)

train.head()



## === cell 7
x_pred = test.drop("key", axis=1)

y = np.log1p(train["fare_amount"].astype(float))

x_train, x_test, y_train, y_test = train_test_split(
    train.drop("fare_amount", axis=1),
    y,
    random_state=123,
    test_size=0.2,
)


def XGBmodel(x_train, x_test, y_train, y_test):
    matrix_train = xgb.DMatrix(x_train, label=y_train)
    matrix_test = xgb.DMatrix(x_test, label=y_test)

    model = xgb.train(
        params={"objective": "reg:squarederror", "eval_metric": "rmse"},
        dtrain=matrix_train,
        num_boost_round=100,
        evals=[(matrix_test, "test")],
        verbose_eval=False,
    )
    return model


model = XGBmodel(x_train, x_test, y_train, y_test)

prediction_log = model.predict(xgb.DMatrix(x_pred))
prediction = np.expm1(prediction_log)

prediction = np.maximum(prediction, 0)

prediction[:10]



## === cell 8
submission = pd.DataFrame(
    {"key": test["key"].astype(str), "fare_amount": np.round(prediction, 2)}
)

submission.to_csv("sub_fare.csv", index=False)
print("Wrote sub_fare.csv with shape:", submission.shape)
submission.head()



## === cell 9
submission
