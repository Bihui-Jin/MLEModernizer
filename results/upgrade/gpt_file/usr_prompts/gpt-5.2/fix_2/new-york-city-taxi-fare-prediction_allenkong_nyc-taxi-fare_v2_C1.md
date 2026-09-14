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

3.9

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
scipy==1.15.3
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

5.27254

# 6. Current score

6.42329

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 6.42329) has done: 'Your current submission is harmed mainly by (1) using the *last* fold’s `rf_model` to predict test (not a model trained on all training data) and (2) a train/test metric bug (`r2_score` arguments are swapped), which can hide problems. To move RMSE down toward the 5.27 target with minimal changes and identical core modeling, I (a) keep your exact RandomForest settings and log1p target, but retrain a final `rf_model` on the full cleaned dataset before generating test predictions, and (b) add a small, competition-standard coordinate bounding-box filter to remove obvious GPS outliers (this is still the same feature set; just cleaner data). I also clip negative fares after `expm1` to 0 to avoid invalid negatives from log-space noise, which usually helps RMSE slightly. These are minimal, safe changes that typically improve NYC Taxi Fare RMSE without changing the model class or training approach.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from scipy import stats
from scipy.stats import norm, skew
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet



## === cell 2
PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
df = pd.read_csv(PATH, nrows=300000)
df.head()



## === cell 3
df.shape



## === cell 4
df.isnull().sum().sort_index() / len(df)



## === cell 5
df.describe()



## === cell 6
df.dropna(subset=["dropoff_latitude", "dropoff_longitude"], inplace=True)



## === cell 7
df.drop(df[df["fare_amount"] < 0].index, axis=0, inplace=True)



## === cell 8
df[df["passenger_count"] > 5].sort_values("passenger_count")



## === cell 9
df.drop(df[df["pickup_longitude"] == 0].index, axis=0, inplace=True)
df.drop(df[df["pickup_latitude"] == 0].index, axis=0, inplace=True)
df.drop(df[df["dropoff_longitude"] == 0].index, axis=0, inplace=True)
df.drop(df[df["dropoff_latitude"] == 0].index, axis=0, inplace=True)
df.drop(df[df["passenger_count"] == 208].index, axis=0, inplace=True)



## === cell 10
df[df["passenger_count"] > 5].sort_values("passenger_count")



## === cell 11
df[df["passenger_count"] > 6].sort_values("passenger_count")



## === cell 12
df["key"] = pd.to_datetime(df["key"])
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])



## === cell 13
df = df[
    (df["pickup_longitude"].between(-74.3, -73.7))
    & (df["dropoff_longitude"].between(-74.3, -73.7))
    & (df["pickup_latitude"].between(40.5, 41.0))
    & (df["dropoff_latitude"].between(40.5, 41.0))
    & (df["passenger_count"].between(1, 6))
].copy()



## === cell 14
df["Year"] = df["pickup_datetime"].dt.year
df["Month"] = df["pickup_datetime"].dt.month
df["Date"] = df["pickup_datetime"].dt.day
df["Day of Week"] = df["pickup_datetime"].dt.dayofweek
df["Hour"] = df["pickup_datetime"].dt.hour
df.drop("pickup_datetime", axis=1, inplace=True)
df.drop("key", axis=1, inplace=True)



## === cell 15
df.head()



## === cell 16
plt.figure(figsize=(10, 8))
sns.heatmap(df.drop("fare_amount", axis=1).corr(), square=True)
plt.suptitle("Pearson Correlation Heatmap")
plt.show()



## === cell 17
X, y = df.drop("fare_amount", axis=1), df["fare_amount"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=24
)



## === cell 18
knn_model = KNeighborsRegressor(n_neighbors=3, n_jobs=-1)
knn_model.fit(X_train, y_train)
y_train_pred = knn_model.predict(X_train)
y_pred = knn_model.predict(X_test)
print("Train r2 score: ", r2_score(y_train, y_train_pred))
print("Test r2 score: ", r2_score(y_test, y_pred))
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Train RMSE: {train_rmse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")



## === cell 19
(mu, sigma) = norm.fit(np.log1p(df["fare_amount"]))
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(19, 5))
ax1 = sns.distplot(np.log1p(df["fare_amount"]), fit=norm, ax=ax1)
ax1.legend(
    [f"Normal distribution ($\\mu=$ {mu:.3f} and $\\sigma=$ {sigma:.3f})"], loc="best"
)
ax1.set_ylabel("Frequency")
ax1.set_title("Log(1+Fare) Distribution")
ax2 = stats.probplot(np.log1p(df["fare_amount"]), plot=plt)
f.show()



## === cell 20
log_y_train = np.log1p(y_train)
log_y_test = np.log1p(y_test)



## === cell 21
train_results, test_results = [], []
n_estimators_test = [16, 32, 64, 128, 256]
for i in range(len(n_estimators_test)):
    rf_model = RandomForestRegressor(
        n_estimators=n_estimators_test[i],
        max_depth=6,
        max_features=0.5,
        n_jobs=-1,
        oob_score=False,
        random_state=24,
    )
    rf_model.fit(X_train, log_y_train)
    y_train_pred = rf_model.predict(X_train)
    y_pred = rf_model.predict(X_test)
    train_rmse = np.sqrt(mean_squared_error(y_train_pred, log_y_train))
    test_rmse = np.sqrt(mean_squared_error(log_y_test, y_pred))
    train_results.append(train_rmse)
    test_results.append(test_rmse)
line_trn = plt.plot(n_estimators_test, train_results, "r")
line_test = plt.plot(n_estimators_test, test_results, "b")
plt.ylabel("Accuracy score")
plt.xlabel("# Estimators")
plt.show()



## === cell 22
kf = KFold(n_splits=5, shuffle=True, random_state=24)
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=6,
    max_features=0.5,
    n_jobs=-1,
    oob_score=True,
    random_state=24,
)
for train_index, test_index in kf.split(X):
    X_tr, X_te = X.iloc[train_index], X.iloc[test_index]
    y_tr, y_te = y.iloc[train_index], y.iloc[test_index]
    y_tr = np.log1p(y_tr)
    y_te = np.log1p(y_te)
    rf_model.fit(X_tr, y_tr)
    y_train_pred = rf_model.predict(X_tr)
    y_pred = rf_model.predict(X_te)
    print("Train r2 score: ", r2_score(y_tr, y_train_pred))
    print("Test r2 score: ", r2_score(y_te, y_pred))
    train_rmse = np.sqrt(mean_squared_error(y_tr, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}\n")



## === cell 23
final_rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=6,
    max_features=0.5,
    n_jobs=-1,
    oob_score=True,
    random_state=24,
)
final_rf_model.fit(X, np.log1p(y))



## === cell 24
test_df = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")



## === cell 25
key = test_df.key



## === cell 26
test_df["key"] = pd.to_datetime(test_df["key"])
test_df["pickup_datetime"] = pd.to_datetime(test_df["pickup_datetime"])



## === cell 27
test_df["Year"] = test_df["pickup_datetime"].dt.year
test_df["Month"] = test_df["pickup_datetime"].dt.month
test_df["Date"] = test_df["pickup_datetime"].dt.day
test_df["Day of Week"] = test_df["pickup_datetime"].dt.dayofweek
test_df["Hour"] = test_df["pickup_datetime"].dt.hour
test_df.drop("pickup_datetime", axis=1, inplace=True)
test_df.drop("key", axis=1, inplace=True)



## === cell 28
test_preds = final_rf_model.predict(test_df)



## === cell 29
fare_pred = np.expm1(test_preds)
fare_pred = np.clip(fare_pred, 0, None)

submission = pd.DataFrame(
    {"key": key, "fare_amount": fare_pred}, columns=["key", "fare_amount"]
)
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with", len(submission), "rows")
