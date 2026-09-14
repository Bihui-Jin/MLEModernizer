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

8.30512

# 6. Current score

5.08038

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.07383) has done: 'I fixed the XGBoost predict call (removed the unsupported ntree_limit argument and used iteration_range instead), corrected the feature list so fare_amount is not duplicated in the training dataframe, and added a safe fallback when best_iteration is not set. These changes resolve the runtime errors, ensure a proper prediction variable is created, and allow the script to write a valid sub_fare.csv submission file.'
- What this solution (achieved 5.08038) has done: 'I slightly weaken the XGBoost model by adding a shallow tree depth (`max_depth: 3`) to raise the RMSE toward the target value while keeping all other logic unchanged. This minimal change adjusts only the model parameters, preserving the overall pipeline and ensuring a valid CSV submission is still produced.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import xgboost as xgb

train = pd.read_csv("../input/train.csv", nrows=10_000_000)
test = pd.read_csv("../input/test.csv")
combine = [train, test]

print(test.dtypes)




## === cell 1
for dataset in combine:
    dataset["longitude_distance"] = np.abs(
        dataset["pickup_longitude"] - dataset["dropoff_longitude"]
    )
    dataset["latitude_distance"] = np.abs(
        dataset["pickup_latitude"] - dataset["dropoff_latitude"]
    )
    dataset["distance_travelled"] = np.sqrt(
        dataset["longitude_distance"] ** 2 + dataset["latitude_distance"] ** 2
    )
    dataset["distance_travelled_sin"] = np.sin(dataset["distance_travelled"])
    dataset["distance_travelled_cos"] = np.cos(dataset["distance_travelled"])
    dataset["distance_travelled_sin_sqrd"] = np.sin(dataset["distance_travelled"]) ** 2
    dataset["distance_travelled_cos_sqrd"] = np.cos(dataset["distance_travelled"]) ** 2

    R = 6371e3  # metres
    phi1 = np.radians(dataset["pickup_latitude"])
    phi2 = np.radians(dataset["dropoff_latitude"])
    delta_phi = np.radians(dataset["dropoff_latitude"] - dataset["pickup_latitude"])
    delta_lambda = np.radians(
        dataset["dropoff_longitude"] - dataset["pickup_longitude"]
    )
    a = (
        np.sin(delta_phi / 2) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    dataset["haversine"] = R * c

    y = np.sin(delta_lambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_lambda)
    dataset["bearing"] = np.degrees(np.arctan2(y, x))

    psi = np.log(np.tan(np.pi / 4 + phi2 / 2) / np.tan(np.pi / 4 + phi1 / 2))
    q = np.where(np.abs(psi) > 1e-12, delta_phi / psi, np.cos(phi1))
    d = np.sqrt(delta_phi**2 + (q * delta_lambda) ** 2) * R
    dataset["rhumb_lines"] = d

    dataset["pickup_datetime"] = pd.to_datetime(dataset["pickup_datetime"])
    dataset["hour_of_day"] = dataset["pickup_datetime"].dt.hour
    dataset["day"] = dataset["pickup_datetime"].dt.day
    dataset["week"] = dataset["pickup_datetime"].dt.isocalendar().week.astype(int)
    dataset["month"] = dataset["pickup_datetime"].dt.month
    dataset["day_of_year"] = dataset["pickup_datetime"].dt.dayofyear
    dataset["week_of_year"] = (
        dataset["pickup_datetime"].dt.isocalendar().week.astype(int)
    )

print(train.head(3))




## === cell 2
numeric_corr = train.select_dtypes(include=[np.number]).corr()
colormap = plt.cm.RdBu
plt.figure(figsize=(12, 10))
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
plt.show()




## === cell 3
features_to_keep = [
    "haversine",
    "distance_travelled",
    "bearing",
    "rhumb_lines",
    "hour_of_day",
    "day",
    "month",
    "day_of_year",
    "passenger_count",
]

train = train[features_to_keep + ["fare_amount"]].dropna()

test_features = features_to_keep + ["key"]
test = test[test_features].dropna()




## === cell 4
X = train.drop("fare_amount", axis=1)
y = train["fare_amount"]

x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=123)


def XGBmodel(x_train, x_val, y_train, y_val):
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dval = xgb.DMatrix(x_val, label=y_val)
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "seed": 42,
        "max_depth": 3,
    }
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dval, "validation")],
        early_stopping_rounds=30,
        verbose_eval=False,
    )
    return model


model = XGBmodel(x_train, x_val, y_train, y_val)

x_test_pred = xgb.DMatrix(test.drop("key", axis=1))

if hasattr(model, "best_iteration") and model.best_iteration is not None:
    prediction = model.predict(
        x_test_pred, iteration_range=(0, model.best_iteration + 1)
    )
else:
    prediction = model.predict(x_test_pred)




## === cell 5
submission = pd.DataFrame({"key": test["key"], "fare_amount": np.round(prediction, 2)})
submission.to_csv("sub_fare.csv", index=False)
print("Submission saved to sub_fare.csv")
