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

4.05769

# 6. Current score

4.82037

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.69351) has done: 'I fix the import errors, update the XGBoost parameters to the current API, and simplify the prediction call so it works without accessing a missing attribute. I also ensure the submission DataFrame is written to a proper “.csv” file and add the missing `operator` import for the feature‑importance plot. These changes resolve the runtime errors while preserving the original modeling logic.'
- What this solution (achieved 5.63939) has done: 'The changes focus on eliminating the slow per‑row datetime parsing by converting the column once and extracting hour, year, month, and weekday via the vectorized `.dt` accessor, and on speeding up XGBoost training by using the fast `'hist'` tree method (which keeps the same algorithmic behavior). Both tweaks preserve exact feature values and model semantics while substantially cutting runtime.'
- What this solution (achieved 5.63071) has done: 'I add cyclic hour features (sin hour, cos hour) to give the model a better representation of time‑of‑day, and I slightly strengthen the XGBoost model (lower learning rate, deeper trees, and more boosting rounds) while keeping the original logic unchanged. These modest changes should lower the RMSE toward the target without altering the overall pipeline.'
- What this solution (achieved 5.63743) has done: 'I add a few easy‑to‑compute numeric features (log‑distance, distance‑squared, passenger‑count‑squared) and slightly strengthen the XGBoost hyper‑parameters (lower learning rate, deeper trees, more boosting rounds) while keeping the same pipeline and data size. These changes preserve the core logic but give the model more informative signals, which should move the RMSE closer to the target value.'
- What this solution (achieved 5.70025) has done: 'I add two inexpensive temporal features (day of month and weekend flag) after the datetime parsing, include them in the feature list, and slightly tweak the XGBoost parameters (lower learning rate, deeper trees, more boosting rounds) so the model can better capture patterns without changing its core logic. These changes keep the original pipeline intact while aiming to lower the RMSE toward the target.'
- What this solution (achieved 4.82037) has done: 'I added a few cheap but informative features (raw coordinates, cyclic month and day encodings) and slightly strengthened the XGBoost hyper‑parameters (lower learning rate, deeper trees, more boosting rounds). These changes keep the original pipeline intact while giving the model more signal, which should reduce the RMSE toward the target.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import operator

sns.set_style("whitegrid")



## === cell 1
train_df = pd.read_csv("../input/train.csv", nrows=1000000)



## === cell 2
train_df.shape



## === cell 3
test_df = pd.read_csv("../input/test.csv")



## === cell 4
test_df.shape



## === cell 5
train_df.head(5)



## === cell 6
train_df.isnull().sum()



## === cell 7
train_df.dropna(inplace=True)



## === cell 8
train_df.describe()



## === cell 9
train_df = train_df[train_df["fare_amount"] > 0]



## === cell 10
train_df.shape




## === cell 11
def distance(lat1, lon1, lat2, lon2):
    a = (
        0.5
        - np.cos((lat2 - lat1) * 0.017453292519943295) / 2
        + np.cos(lat1 * 0.017453292519943295)
        * np.cos(lat2 * 0.017453292519943295)
        * (1 - np.cos((lon2 - lon1) * 0.017453292519943295))
        / 2
    )
    res = 0.6213712 * 12742 * np.arcsin(np.sqrt(a))
    return res




## === cell 12
train_df["distance"] = distance(
    train_df.pickup_latitude,
    train_df.pickup_longitude,
    train_df.dropoff_latitude,
    train_df.dropoff_longitude,
)



## === cell 13
test_df["distance"] = distance(
    test_df.pickup_latitude,
    test_df.pickup_longitude,
    test_df.dropoff_latitude,
    test_df.dropoff_longitude,
)



## === cell 14
train_df = train_df[train_df["distance"] < 15]



## === cell 15
train_df.describe()



## === cell 16
train_df = train_df[
    (train_df["passenger_count"] != 0) & (train_df["passenger_count"] < 10)
]



## === cell 17
train_df["log_distance"] = np.log1p(train_df["distance"])
train_df["distance_sq"] = train_df["distance"] ** 2
train_df["passenger_cnt_sq"] = train_df["passenger_count"] ** 2

test_df["log_distance"] = np.log1p(test_df["distance"])
test_df["distance_sq"] = test_df["distance"] ** 2
test_df["passenger_cnt_sq"] = test_df["passenger_count"] ** 2



## === cell 18
train_df["pickup_datetime"] = pd.to_datetime(train_df["pickup_datetime"])
test_df["pickup_datetime"] = pd.to_datetime(test_df["pickup_datetime"])

train_df["hour"] = train_df["pickup_datetime"].dt.hour
train_df["year"] = train_df["pickup_datetime"].dt.year
train_df["month"] = train_df["pickup_datetime"].dt.month
train_df["weekday"] = train_df["pickup_datetime"].dt.weekday

test_df["hour"] = test_df["pickup_datetime"].dt.hour
test_df["year"] = test_df["pickup_datetime"].dt.year
test_df["month"] = test_df["pickup_datetime"].dt.month
test_df["weekday"] = test_df["pickup_datetime"].dt.weekday

train_df["hour_sin"] = np.sin(2 * np.pi * train_df["hour"] / 24)
train_df["hour_cos"] = np.cos(2 * np.pi * train_df["hour"] / 24)

test_df["hour_sin"] = np.sin(2 * np.pi * test_df["hour"] / 24)
test_df["hour_cos"] = np.cos(2 * np.pi * test_df["hour"] / 24)

train_df["month_sin"] = np.sin(2 * np.pi * train_df["month"] / 12)
train_df["month_cos"] = np.cos(2 * np.pi * train_df["month"] / 12)
test_df["month_sin"] = np.sin(2 * np.pi * test_df["month"] / 12)
test_df["month_cos"] = np.cos(2 * np.pi * test_df["month"] / 12)

train_df["day"] = train_df["pickup_datetime"].dt.day
test_df["day"] = test_df["pickup_datetime"].dt.day

train_df["day_sin"] = np.sin(2 * np.pi * train_df["day"] / 31)
train_df["day_cos"] = np.cos(2 * np.pi * train_df["day"] / 31)
test_df["day_sin"] = np.sin(2 * np.pi * test_df["day"] / 31)
test_df["day_cos"] = np.cos(2 * np.pi * test_df["day"] / 31)

train_df["is_weekend"] = (train_df["weekday"] >= 5).astype(int)
test_df["is_weekend"] = (test_df["weekday"] >= 5).astype(int)



## === cell 19
feat_cols_s = [
    "distance",
    "log_distance",
    "distance_sq",
    "passenger_count",
    "passenger_cnt_sq",
    "hour",
    "hour_sin",
    "hour_cos",
    "year",
    "month",
    "month_sin",
    "month_cos",
    "weekday",
    "day",
    "day_sin",
    "day_cos",
    "is_weekend",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]
X = train_df[feat_cols_s]
y = train_df["fare_amount"]



## === cell 20
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)



## === cell 21
import xgboost as xgb




## === cell 22
def XGBoost(X_train, X_test, y_train, y_test, num_rounds=5000):
    dtrain = xgb.DMatrix(
        X_train.values.astype(np.float32), label=y_train.values.astype(np.float32)
    )
    dtest = xgb.DMatrix(
        X_test.values.astype(np.float32), label=y_test.values.astype(np.float32)
    )
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "seed": 42,
        "eta": 0.015,  # slightly lower learning rate
        "max_depth": 14,  # deeper trees
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "alpha": 0.0,
        "lambda": 1.0,
    }
    return xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_rounds,
        early_stopping_rounds=50,
        evals=[(dtest, "test")],
        verbose_eval=False,
    )




## === cell 23
xgbm = XGBoost(X_train, X_test, y_train, y_test)



## === cell 24
xgbm_pred = xgbm.predict(xgb.DMatrix(test_df[feat_cols_s].values.astype(np.float32)))



## === cell 25
submission = pd.DataFrame(
    {"key": test_df["key"], "fare_amount": xgbm_pred},
    columns=["key", "fare_amount"],
)
submission.to_csv("submission.csv", index=False)



## === cell 26
importance = xgbm.get_score()
importance = sorted(importance.items(), key=operator.itemgetter(1))
df_imp = pd.DataFrame(importance, columns=["feature", "score"])
plt.figure(figsize=(10, 6))
df_imp.plot(kind="barh", x="feature", y="score", legend=False)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()
