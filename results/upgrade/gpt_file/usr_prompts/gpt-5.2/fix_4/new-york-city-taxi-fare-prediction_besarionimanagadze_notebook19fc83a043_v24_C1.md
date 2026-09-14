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

3.12

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

5.0139

# 6. Current score

7.30319

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.44167) has done: 'Your score is extremely far from the target (lower-is-better RMSE), and the biggest cause is a bug in the training filter: `idx = train_data_set.passenger_count != 0 & (...)` is missing parentheses, producing a mostly-wrong mask and badly training the linear model. I fix that mask (minimal change), make the train/test feature engineering consistent and numeric for XGBoost (your current XGB block accidentally feeds object `pickup_datetime` and misses engineered features), and remove rounding/clipping issues that can inflate RMSE. I keep your two-model structure intact and still write a valid `submission.csv`, while also writing your `taxi_fare_submission.csv` as before. These changes should dramatically reduce RMSE toward your target without changing the core modeling approaches.'
- What this solution (achieved 1688.12171) has done: 'Your current RMSE (6.44167, lower-is-better) is still meaningfully above the target (5.0139), so we should improve the model while keeping your exact two-model structure and feature set. The smallest high-impact change is to fix the XGBoost training so it no longer uses early stopping (which violates your “no early stopping” constraint and also can underfit in this setup) and instead trains for a fixed number of boosting rounds. I also make XGBoost deterministic by adding `seed` and clip negative predictions to 0.0 (fares can’t be negative), which typically reduces RMSE without changing core logic. The LinearRegression path and all feature engineering stay the same, and both `submission.csv` and `taxi_fare_submission.csv` still be written.'
- What this solution (achieved 7.30319) has done: 'Your current score (1688 RMSE) is far above the target (5.0139), so we should make a small but high-impact correctness fix rather than “tuning.” The biggest remaining issue is that the XGBoost model is trained/predicted on raw `.values` arrays, which can silently include wrong dtypes/order and makes it easy for train/test feature alignment to drift; using explicit numeric DataFrames and identical column ordering for both train and test stabilizes learning and typically drops RMSE drastically without changing your model approach. I also make `passenger_count` cleaning consistent (remove zeros and extreme counts) and clip extreme fares to a reasonable upper bound (standard NYC Taxi baseline practice) to reduce outlier-driven RMSE, while keeping your exact two-model structure, feature set, and fixed boosting rounds. The script still writes both `submission.csv` (linear) and `taxi_fare_submission.csv` (xgboost) in the required format.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
train_data_set = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/train.csv",
    nrows=100_000,
    parse_dates=["pickup_datetime"],
)
train_data_set.head(5)



## === cell 2
print(train_data_set.dtypes)
train_data_set.describe()



## === cell 3
old_len = len(train_data_set)
train_data_set = train_data_set[
    (train_data_set.fare_amount >= 0.1) & (train_data_set.fare_amount <= 250.0)
]
new_len = len(train_data_set)
print(f"Removed {(old_len-new_len)} entities from the dataset")
train_data_set.describe()



## === cell 4
old_len = len(train_data_set)
train_data_set = train_data_set.dropna(how="any", axis="rows")
new_len = len(train_data_set)
print(f"Removed {(old_len-new_len)} entities from the dataset")



## === cell 5
train_data_set.fare_amount.hist(bins=100, figsize=(14, 3))
plt.xlabel("fare $USD")
plt.title("Histogram")




## === cell 6
def select_within_boundingbox(df, box):
    return (
        (df.pickup_longitude >= box[0])
        & (df.pickup_longitude <= box[1])
        & (df.pickup_latitude >= box[2])
        & (df.pickup_latitude <= box[3])
        & (df.dropoff_longitude >= box[0])
        & (df.dropoff_longitude <= box[1])
        & (df.dropoff_latitude >= box[2])
        & (df.dropoff_latitude <= box[3])
    )


new_york_box = (-74.763379, -72.856164, 40.502009, 41.915509)

old_len = len(train_data_set)
train_data_set = train_data_set[select_within_boundingbox(train_data_set, new_york_box)]
new_len = len(train_data_set)
print(f"Removed {(old_len-new_len)} entities from the dataset")




## === cell 7
def distance_on_the_sphere(lat1, lon1, lat2, lon2):
    earth_radius = 6371
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2.0) * np.sin(delta_phi / 2.0) + np.cos(phi1) * np.cos(
        phi2
    ) * np.sin(delta_lambda / 2.0) * np.sin(delta_lambda / 2.0)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return earth_radius * c


train_data_set["distance"] = distance_on_the_sphere(
    train_data_set["pickup_latitude"],
    train_data_set["pickup_longitude"],
    train_data_set["dropoff_latitude"],
    train_data_set["dropoff_longitude"],
)

train_data_set.head(5)



## === cell 8
train_data_set["pickup_datetime"] = pd.to_datetime(train_data_set["pickup_datetime"])
train_data_set["hour"] = train_data_set["pickup_datetime"].dt.hour
train_data_set["year"] = train_data_set["pickup_datetime"].dt.year
train_data_set["day_of_week"] = train_data_set["pickup_datetime"].dt.dayofweek
train_data_set["is_rush_hour"] = train_data_set["hour"].apply(
    lambda x: 1 if (7 <= x <= 10) or (16 <= x <= 19) else 0
)

train_data_set.head(5)



## === cell 9
nyc_down_town = (-74.0063889, 40.7141667)
jfk_airport = (-73.7822222222, 40.6441666667)

train_data_set["distance_to_downtown"] = distance_on_the_sphere(
    nyc_down_town[1],
    nyc_down_town[0],
    train_data_set.pickup_latitude,
    train_data_set.pickup_longitude,
)
train_data_set["distance_to_jfk_airport"] = distance_on_the_sphere(
    jfk_airport[1],
    jfk_airport[0],
    train_data_set.pickup_latitude,
    train_data_set.pickup_longitude,
)

train_data_set.head(5)



## === cell 10
idx = (
    (train_data_set.passenger_count > 0)
    & (train_data_set.passenger_count <= 6)
    & (train_data_set.distance_to_downtown < 15)
)

features = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "distance",
    "hour",
    "year",
    "day_of_week",
    "is_rush_hour",
    "distance_to_downtown",
    "distance_to_jfk_airport",
]
target = "fare_amount"

X = train_data_set.loc[idx, features].copy()
y = train_data_set.loc[idx, target].astype(np.float32).values

for c in features:
    X[c] = pd.to_numeric(X[c], errors="coerce")
X = X.dropna(axis=0, how="any")
y = train_data_set.loc[X.index, target].astype(np.float32).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=2666
)

linear_model = Pipeline(
    (
        ("standard_scaler", StandardScaler()),
        ("lin_reg", LinearRegression()),
    )
)

linear_model.fit(X_train, y_train)



## === cell 11
test_data_set = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/test.csv",
    parse_dates=["pickup_datetime"],
)

test_data_set["distance"] = distance_on_the_sphere(
    test_data_set["pickup_latitude"],
    test_data_set["pickup_longitude"],
    test_data_set["dropoff_latitude"],
    test_data_set["dropoff_longitude"],
)
test_data_set["pickup_datetime"] = pd.to_datetime(test_data_set["pickup_datetime"])
test_data_set["hour"] = test_data_set["pickup_datetime"].dt.hour
test_data_set["year"] = test_data_set["pickup_datetime"].dt.year
test_data_set["day_of_week"] = test_data_set["pickup_datetime"].dt.dayofweek
test_data_set["is_rush_hour"] = test_data_set["hour"].apply(
    lambda x: 1 if (7 <= x <= 10) or (16 <= x <= 19) else 0
)
test_data_set["distance_to_downtown"] = distance_on_the_sphere(
    nyc_down_town[1],
    nyc_down_town[0],
    test_data_set.pickup_latitude,
    test_data_set.pickup_longitude,
)
test_data_set["distance_to_jfk_airport"] = distance_on_the_sphere(
    jfk_airport[1],
    jfk_airport[0],
    test_data_set.pickup_latitude,
    test_data_set.pickup_longitude,
)



## === cell 12
XTEST = test_data_set[features].copy()
for c in features:
    XTEST[c] = pd.to_numeric(XTEST[c], errors="coerce")
XTEST = XTEST.fillna(0.0)

y_pred_final = linear_model.predict(XTEST)

y_pred_final = np.clip(y_pred_final, 0.0, None)

submission = pd.DataFrame(
    {"key": test_data_set.key, "fare_amount": y_pred_final},
    columns=["key", "fare_amount"],
)
submission.to_csv("submission.csv", index=False)

submission.head()



## === cell 13
train_xgb = train_data_set.loc[idx, ["key", "fare_amount"] + features].copy()

for c in features:
    train_xgb[c] = pd.to_numeric(train_xgb[c], errors="coerce")
train_xgb["fare_amount"] = pd.to_numeric(train_xgb["fare_amount"], errors="coerce")
train_xgb = train_xgb.dropna(axis=0, how="any")

y = train_xgb["fare_amount"].astype(np.float32).values
X_xgb = train_xgb[features].astype(np.float32)

x_train, x_test, y_train, y_test = train_test_split(
    X_xgb, y, random_state=2666, test_size=0.05
)

params = {
    "max_depth": 7,
    "subsample": 0.9,
    "eta": 0.03,
    "colsample_bytree": 0.9,
    "random_state": 2666,
    "seed": 2666,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
}


def XGBmodel(x_train, x_test, y_train, y_test, params):
    matrix_train = xgb.DMatrix(x_train, label=y_train, feature_names=features)
    matrix_test = xgb.DMatrix(x_test, label=y_test, feature_names=features)

    model = xgb.train(
        params=params,
        dtrain=matrix_train,
        num_boost_round=2000,
        evals=[(matrix_test, "test")],
        verbose_eval=False,
    )
    return model


model = XGBmodel(x_train, x_test, y_train, y_test, params)

test_key = test_data_set["key"].values
x_pred = test_data_set[features].copy()
for c in features:
    x_pred[c] = pd.to_numeric(x_pred[c], errors="coerce")
x_pred = x_pred.fillna(0.0).astype(np.float32)

prediction = model.predict(xgb.DMatrix(x_pred, feature_names=features))

prediction = np.clip(prediction, 0.0, None)

submission = pd.DataFrame(
    {"key": test_key, "fare_amount": prediction}, columns=["key", "fare_amount"]
)
submission.to_csv("taxi_fare_submission.csv", index=False)
submission.head()
