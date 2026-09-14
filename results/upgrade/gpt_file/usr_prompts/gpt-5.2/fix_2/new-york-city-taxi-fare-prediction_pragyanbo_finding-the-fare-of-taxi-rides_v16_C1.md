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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
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

4.20766

# 6. Current score

6.07789

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 6.07789) has done: 'I fix the hard runtime errors so the notebook runs end-to-end and always writes a valid `submission.csv` in the required `key,fare_amount` format. The main scoring model be the existing XGBoost approach (kept the same training loop/semantics), but updated for XGBoost 2.0+ API (no deprecated `reg:linear` and no `best_ntree_limit` attribute). I also remove the TensorFlow Estimator section (it cannot run in TF 2.18 where `tf.estimator` is not available) so it doesn’t break execution after a valid submission is already created. Finally, I keep feature engineering identical but make datetime parsing vectorized (score-neutral) and add a small non-negative clamp on predictions (prevents obviously invalid fares and typically improves RMSE slightly).'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")



## === cell 1
import os

TRAIN_PATH = "../input/train.csv"
TEST_PATH = "../input/test.csv"
SAMPLE_PATH = "../input/sample_submission.csv"

if not os.path.exists(TRAIN_PATH):
    TRAIN_PATH = "/kaggle/input/train.csv"
    TEST_PATH = "/kaggle/input/test.csv"
    SAMPLE_PATH = "/kaggle/input/sample_submission.csv"



## === cell 2
train_df = pd.read_csv(TRAIN_PATH, nrows=10000)
train_df.shape



## === cell 3
test_df = pd.read_csv(TEST_PATH)
test_df.shape



## === cell 4
train_df.head(5)



## === cell 5
train_df.isnull().sum()



## === cell 6
train_df.dropna(inplace=True)



## === cell 7
train_df.describe()



## === cell 8
train_df = train_df[train_df["fare_amount"] > 0]
train_df.shape




## === cell 9
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




## === cell 10
train_df["distance"] = distance(
    train_df.pickup_latitude,
    train_df.pickup_longitude,
    train_df.dropoff_latitude,
    train_df.dropoff_longitude,
)

test_df["distance"] = distance(
    test_df.pickup_latitude,
    test_df.pickup_longitude,
    test_df.dropoff_latitude,
    test_df.dropoff_longitude,
)



## === cell 11
train_df = train_df[train_df["distance"] < 15]
train_df.describe()



## === cell 12
train_df = train_df[
    (train_df["passenger_count"] != 0) & (train_df["passenger_count"] < 10)
]
train_df.shape



## === cell 13
train_dt = pd.to_datetime(train_df["pickup_datetime"], errors="coerce", utc=False)
test_dt = pd.to_datetime(test_df["pickup_datetime"], errors="coerce", utc=False)

train_df["hour"] = train_dt.dt.hour
train_df["year"] = train_dt.dt.year

test_df["hour"] = test_dt.dt.hour
test_df["year"] = test_dt.dt.year

train_df = train_df.dropna(subset=["hour", "year"])



## === cell 14
feat_cols_s = ["distance", "passenger_count", "hour", "year"]
X = train_df[feat_cols_s]
y = train_df["fare_amount"]



## === cell 15
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.3, random_state=42
)



## === cell 16
from sklearn.ensemble import RandomForestRegressor

r_reg = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
r_reg.fit(X_train, y_train)

rf_pred = r_reg.predict(test_df[feat_cols_s])
rf_pred = np.maximum(rf_pred, 0.0)

rf_submission = pd.DataFrame(
    {"key": test_df.key, "fare_amount": rf_pred}, columns=["key", "fare_amount"]
)
rf_submission.to_csv("Random Forest regression.csv", index=False)



## === cell 17
import xgboost as xgb




## === cell 18
def XGBoost(X_train, X_valid, y_train, y_valid, num_rounds=300):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.1,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
    }

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_rounds,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=20,
        verbose_eval=False,
    )
    return booster




## === cell 19
xgbm = XGBoost(X_train, X_valid, y_train, y_valid)

dtest = xgb.DMatrix(test_df[feat_cols_s])

if hasattr(xgbm, "best_iteration") and xgbm.best_iteration is not None:
    xgbm_pred = xgbm.predict(dtest, iteration_range=(0, xgbm.best_iteration + 1))
else:
    xgbm_pred = xgbm.predict(dtest)

xgbm_pred = np.maximum(xgbm_pred, 0.0)

submission = pd.DataFrame(
    {"key": test_df.key, "fare_amount": xgbm_pred}, columns=["key", "fare_amount"]
)
submission.to_csv("submission.csv", index=False)

submission.to_csv("XGboost regression.csv", index=False)

submission.head()



## === cell 20
import operator

importance = xgbm.get_score(importance_type="weight")
importance = sorted(importance.items(), key=operator.itemgetter(1))
df_imp = pd.DataFrame(importance, columns=["feature", "score"])

plt.figure()
ax = df_imp.plot(kind="barh", x="feature", y="score", legend=False, figsize=(8, 4))
ax.set_title("Feature Importance (XGBoost)")
plt.tight_layout()



## === cell 21
print(
    "TensorFlow Estimator section skipped: tf.estimator is not available in this environment. "
    "Submission written to 'submission.csv'."
)
