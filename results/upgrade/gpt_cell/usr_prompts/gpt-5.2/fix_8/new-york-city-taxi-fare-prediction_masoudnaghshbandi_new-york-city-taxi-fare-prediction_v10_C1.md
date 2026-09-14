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

3.10

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

3.77552

# 6. Current score

5.14613

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.34119) has done: 'Your code likely didn’t yield a Kaggle score because the submission is missing rows: you filter test rows by nulls and geographic bounds, which drops keys and produces an invalid/incomplete submission file. To move toward the target RMSE, keep your training/modeling exactly the same but ensure the submission contains *all* test keys by predicting for every row (imputing any missing values and not dropping out-of-bound rows). Additionally, the current training filter drops `passenger_count == 208`, which looks like a typo and can remove valid signal; changing it to `passenger_count == 0` (already present) is a minimal bug-fix that should slightly improve generalization without changing core logic. These changes are directly aimed at producing a valid submission and improving score stability toward the target.'
- What this solution (achieved 6.0534) has done: 'Your current gap is 5.34119 − 3.77552 = 1.56567 RMSE (lower is better), so we should make small, safe changes that typically improve generalization without changing the core model/training loop. The biggest score drag in this competition is usually bad labels and outliers, so we add minimal, standard NYC Taxi cleaning on the training subset only: remove non-positive/huge fares and obvious coordinate outliers, and add a single strong feature (haversine distance) while keeping the same XGBoost regressor and scaling workflow. We not drop any test rows (still predict for every test key), and we also align the XGBRegressor objective to the modern equivalent (`reg:squarederror`) to avoid deprecated behavior differences. These changes are small but commonly move RMSE materially toward your target without altering the overall approach.'
- What this solution (achieved 6.50194) has done: 'Your current RMSE (6.0534) is worse than the target (3.77552), so we should make small, low-risk improvements that usually reduce error without changing the overall XGBoost + scaling approach. The largest easy gain here is to add a few standard time-based features from `pickup_datetime` (hour, dayofweek, month, year) for both train and test, while keeping the same model type and training flow. We also clip negative test predictions to 0 (fares can’t be negative), which typically improves RMSE slightly without affecting submission validity. Everything else (data loading, cleaning, scaler, XGBRegressor training, and full-row test submission) remains the same.'
- What this solution (achieved 5.14613) has done: 'Your current RMSE (6.50194) is much worse than the target (3.77552), so we should make a small, standard generalization improvement without changing the overall approach (same XGBoost regressor, same scaling, same train/valid split, same feature set). The biggest low-risk gain here is to log-transform the target (`fare_amount`) during training and invert the transform at prediction time, which typically reduces the impact of high-fare outliers under RMSE while preserving the same model/loop semantics. I keep all cleaning/feature engineering the same, only adjusting `y_train/y_valid` used in `fit` and the post-processing for both validation and test predictions. Submission creation remains identical and still includes all test keys.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as MSE



## === cell 2
train = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/train.csv", nrows=1000000
)
test_raw = pd.read_csv("../input/new-york-city-taxi-fare-prediction/test.csv")



## === cell 3
train.shape, test_raw.shape



## === cell 4
train.head()



## === cell 5
train.isnull().sum()



## === cell 6
train = train.dropna(how="any", axis="rows")



## === cell 7
test_raw.isnull().sum()



## === cell 8
train.head()



## === cell 9
train["fare_amount"].describe()



## === cell 10
train = train[(train["fare_amount"] > 0) & (train["fare_amount"] < 250)].copy()

train.drop(train[train["pickup_longitude"] == 0].index, axis=0, inplace=True)
train.drop(train[train["pickup_latitude"] == 0].index, axis=0, inplace=True)
train.drop(train[train["dropoff_longitude"] == 0].index, axis=0, inplace=True)
train.drop(train[train["dropoff_latitude"] == 0].index, axis=0, inplace=True)
train.drop(train[train["passenger_count"] > 5].index, axis=0, inplace=True)
train.drop(train[train["passenger_count"] == 0].index, axis=0, inplace=True)



## === cell 11
train.head()




## === cell 12
def add_time_features(df, dt_col="pickup_datetime"):
    dt = pd.to_datetime(df[dt_col], utc=True, errors="coerce")
    df["pickup_hour"] = dt.dt.hour.astype("float32")
    df["pickup_dayofweek"] = dt.dt.dayofweek.astype("float32")
    df["pickup_month"] = dt.dt.month.astype("float32")
    df["pickup_year"] = dt.dt.year.astype("float32")
    return df


train = add_time_features(train, "pickup_datetime")



## === cell 13
train.drop(["key"], axis=1, inplace=True)



## === cell 14
train.drop(["pickup_datetime"], axis=1, inplace=True)



## === cell 15
train.dropna(inplace=True)

train.drop(
    train.index[
        (train.pickup_longitude < -75)
        | (train.pickup_longitude > -72)
        | (train.pickup_latitude < 40)
        | (train.pickup_latitude > 42)
    ],
    inplace=True,
)
train.drop(
    train.index[
        (train.dropoff_longitude < -75)
        | (train.dropoff_longitude > -72)
        | (train.dropoff_latitude < 40)
        | (train.dropoff_latitude > 42)
    ],
    inplace=True,
)




## === cell 16
def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c


train["haversine_km"] = haversine_km(
    train["pickup_longitude"].values,
    train["pickup_latitude"].values,
    train["dropoff_longitude"].values,
    train["dropoff_latitude"].values,
)

train = train[(train["haversine_km"] >= 0) & (train["haversine_km"] <= 100)].copy()



## === cell 17
train.head()



## === cell 18
X, y = train.drop("fare_amount", axis=1), train["fare_amount"]
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=12
)



## === cell 19
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)



## === cell 20
xgb_r = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=200, seed=123)



## === cell 21
y_train_log = np.log1p(y_train.values)
xgb_r.fit(X_train_scaled, y_train_log)



## === cell 22
y_pred_log = xgb_r.predict(X_valid_scaled)
y_pred = np.expm1(y_pred_log)
y_pred = np.clip(y_pred, 0.0, None)



## === cell 23
rmse = np.sqrt(MSE(y_valid, y_pred))
print("RMSE : % f" % (rmse))



## === cell 24
test_keys = test_raw[["key"]].copy()
test = test_raw.drop(["key"], axis=1)



## === cell 25
test.head()



## === cell 26
test = add_time_features(test.copy(), "pickup_datetime")

test.drop(["pickup_datetime"], axis=1, inplace=True)

test["haversine_km"] = haversine_km(
    test["pickup_longitude"].values,
    test["pickup_latitude"].values,
    test["dropoff_longitude"].values,
    test["dropoff_latitude"].values,
)

train_feature_medians = X.median(numeric_only=True)
for col in test.columns:
    if col in train_feature_medians.index:
        test[col] = test[col].fillna(train_feature_medians[col])
    else:
        test[col] = test[col].fillna(0.0)



## === cell 27
test.head()



## === cell 28
test_scaled = scaler.transform(test)



## === cell 29
new_pred_log = xgb_r.predict(test_scaled)
new_pred = np.expm1(new_pred_log)
new_pred = np.clip(new_pred, 0.0, None)



## === cell 30
submission = test_keys.copy()
submission["fare_amount"] = new_pred



## === cell 31
submission.head()



## === cell 32
submission = submission[["key", "fare_amount"]]
submission.to_csv("submission1.csv", index=False)



## === cell 33
print("Wrote submission1.csv with shape:", submission.shape)
print("Expected test rows:", test_raw.shape[0])
print(submission.head())
print("Any missing keys in submission:", submission["key"].isnull().any())
print("Any missing fares in submission:", submission["fare_amount"].isnull().any())
