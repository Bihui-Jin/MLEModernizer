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

12.50832

# 6. Current score

9.31901

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 10.11841) has done: 'The changes fix the holiday‑feature creation by using a proper datetime dtype (`datetime64[ns]`) and simplify the XGBoost prediction call to use the correct `iteration_range` argument instead of the removed `ntree_limit`. These fixes eliminate the runtime errors, allow the script to produce a valid `sub_fare.csv` submission, and keep the original modeling logic unchanged.'
- What this solution (achieved 9.75289) has done: 'I lower the number of boosting rounds in the XGBoost model from 400 to 80. This modest reduction slightly under‑fit the data, raising the RMSE from the current 10.12 toward the target 12.51 while keeping the overall pipeline, features, and evaluation logic unchanged.'
- What this solution (achieved 9.37779) has done: 'I slightly reduce the model’s capacity to raise the RMSE toward the target value. In **cell 6** I change the XGBoost `num_boost_round` from 80 to 20 (the rest of the pipeline and feature engineering stay unchanged). Fewer boosting rounds under‑fit the data, increasing the validation error and moving the score into the desired range while preserving all core logic.'
- What this solution (achieved 9.31901) has done: 'I slightly decrease the XGBoost boosting rounds from 20 to 5 in the model‑training cell. Fewer rounds under‑fit the data more, raising the validation RMSE and moving the score upward toward the target range (≈12.5) while keeping the overall pipeline unchanged. All other code stays the same, and the script still writes a valid `sub_fare.csv` submission.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import datetime as dt
import random
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import xgboost as xgb

from pandas.tseries.holiday import USFederalHolidayCalendar as calendar



## === cell 1
train = pd.read_csv("../input/train.csv", nrows=1_000_000)
print("train shape:", train.shape)
train.head()



## === cell 2
test = pd.read_csv("../input/test.csv")
test.head()



## === cell 3
combine = [train, test]

for dataset in combine:
    dataset["longitude_distance"] = abs(
        dataset["pickup_longitude"] - dataset["dropoff_longitude"]
    )
    dataset["latitude_distance"] = abs(
        dataset["pickup_latitude"] - dataset["dropoff_latitude"]
    )
    dataset["distance_travelled"] = (
        dataset["longitude_distance"] ** 2 * dataset["latitude_distance"] ** 2
    ) ** 0.5

    R = 6371e3  # metres
    phi1 = np.radians(dataset["pickup_latitude"])
    phi2 = np.radians(dataset["dropoff_latitude"])
    phi_chg = np.radians(dataset["pickup_latitude"] - dataset["dropoff_latitude"])
    delta_chg = np.radians(dataset["pickup_longitude"] - dataset["dropoff_longitude"])
    a = np.sin(phi_chg / 2) + np.cos(phi1) * np.cos(phi2) * np.sin(delta_chg / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    dataset["haversine"] = R * c

    y = np.sin(delta_chg * np.cos(phi2))
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_chg)
    dataset["bearing"] = np.degrees(np.arctan2(y, x))

    psi_chg = np.log(np.tan(np.pi / 4 + phi2 / 2) / np.tan(np.pi / 4 + phi1 / 2))
    q = phi_chg / psi_chg
    d = (phi_chg + q**2 * delta_chg**2) ** 0.5 * R
    dataset["rhumb_lines"] = d

    dataset["pickup_datetime"] = pd.to_datetime(
        dataset["pickup_datetime"], infer_datetime_format=True
    )

    dataset["hour_of_day"] = dataset["pickup_datetime"].dt.hour
    dataset["day"] = dataset["pickup_datetime"].dt.day
    dataset["week"] = dataset["pickup_datetime"].dt.isocalendar().week
    dataset["month"] = dataset["pickup_datetime"].dt.month
    dataset["dayofweek"] = dataset["pickup_datetime"].dt.dayofweek
    dataset["day_of_year"] = dataset["pickup_datetime"].dt.dayofyear
    dataset["week_of_year"] = dataset["pickup_datetime"].dt.isocalendar().week

    cal = calendar()
    holidays = cal.holidays()
    dataset["usFedHoliday"] = dataset["pickup_datetime"].dt.normalize().isin(holidays)

train.head(3)



## === cell 4
numeric_corr = train.select_dtypes(include=[np.number]).corr()
colormap = plt.cm.RdBu
plt.figure(figsize=(20, 20))
plt.title("Pearson Correlation of Numeric Features", y=1.05, size=15)
sns.heatmap(
    numeric_corr,
    linewidths=0.1,
    vmax=1.0,
    square=True,
    cmap=colormap,
    linecolor="white",
    annot=True,
)



## === cell 5
train.drop(["key", "pickup_datetime"], axis=1, inplace=True)
train.dropna(inplace=True)

test.drop(["pickup_datetime"], axis=1, inplace=True)  # keep 'key' for submission



## === cell 6
x_pred = test.drop("key", axis=1)

X = train.drop("fare_amount", axis=1)
y = train["fare_amount"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)


def XGBmodel(x_train, x_test, y_train, y_test):
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)
    params = {
        "objective": "reg:squarederror",  # modern equivalent of reg:linear
        "eval_metric": "rmse",
    }
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=5,  # reduced further to increase validation error
        early_stopping_rounds=30,
        evals=[(dtest, "test")],
        verbose_eval=False,
    )
    return model


model = XGBmodel(x_train, x_test, y_train, y_test)

best_ntree = (
    model.best_iteration if hasattr(model, "best_iteration") else model.best_ntree_limit
)

prediction = model.predict(xgb.DMatrix(x_pred), iteration_range=(0, best_ntree))



## === cell 7
submission = pd.DataFrame({"key": test["key"], "fare_amount": np.round(prediction, 2)})

submission.to_csv("sub_fare.csv", index=False)



## === cell 8
submission.head()
