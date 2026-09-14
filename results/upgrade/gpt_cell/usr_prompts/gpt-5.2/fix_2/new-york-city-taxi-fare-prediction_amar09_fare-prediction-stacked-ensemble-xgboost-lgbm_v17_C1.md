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

No external packages required in the script and installed.

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

3.93504

# 6. Current score

4.72377

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 4.72377) has done: 'Diagnosis: The crash in cell 39 happens because `X_train` (and/or `X_test`) contains NaN values, and `RandomForestRegressor.fit()` rejects NaNs. These NaNs are introduced by feature engineering (e.g., datetime-derived fields or distance computations) and weren’t removed after those new columns were added; the earlier `dropna()` (cell 15) occurred before these features existed. The smallest safe fix is to remove rows with NaNs from the train/test split outputs just before fitting, keeping the same model/training logic.

Patch summary: In cell 39, filter out any samples where `X_train` or `X_test` has NaNs (and align `y_train`/`y_test` accordingly) before calling `.fit()` and `.predict()`. This avoids changing earlier preprocessing and preserves the same estimator and evaluation semantics.

Updated cells: (cell 39 only)

Compatibility notes for cell k+1: The variables `X_train`, `X_test`, `y_train`, and `y_test` remain defined as numpy arrays and be used by cell 40 unchanged; they simply have fewer rows after NaN removal, which is compatible with `XGBRegressor.fit()`.

Assumptions: It is acceptable to drop NaN-containing samples at this stage (rather than imputing), since the notebook already uses row filtering/drop operations and this is the minimal change needed to unblock `RandomForestRegressor`.'

# 9. Code solution

## === cell 0
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns # for plot visualization 

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from math import sqrt

import os
print(os.listdir("../input"))


## === cell 1
sns.set_style("darkgrid")


## === cell 2
test_dataset = pd.read_csv('../input/test.csv')
train_dataset = pd.read_csv('../input/train.csv', nrows=5000)


## === cell 3
train_dataset.head(5)


## === cell 4
train_dataset.tail(5)


## === cell 5
train_dataset.dtypes


## === cell 6
train_dataset.info(memory_usage='deep')


## === cell 7
for dtype in ['float','int','object']:
    selected_dtype = train_dataset.select_dtypes(include=[dtype])
    mean_usage_b = selected_dtype.memory_usage(deep=True).mean()
    mean_usage_mb = mean_usage_b / 1024 ** 2
    print("Average memory usage for {} columns: {:03.2f} MB".format(dtype,mean_usage_mb))


## === cell 8
train_dataset.drop(labels='key', axis=1, inplace=True)
test_dataset.drop(labels='key', axis=1, inplace=True)


## === cell 9
train_dataset.info(memory_usage='deep')


## === cell 10
train_dataset.passenger_count = train_dataset.passenger_count.astype(dtype = 'uint8')


## === cell 11
train_dataset.pickup_longitude = train_dataset.pickup_longitude.astype(dtype = 'float32')
train_dataset.pickup_latitude = train_dataset.pickup_latitude.astype(dtype = 'float32')
train_dataset.dropoff_longitude = train_dataset.dropoff_longitude.astype(dtype = 'float32')
train_dataset.dropoff_latitude = train_dataset.dropoff_latitude.astype(dtype = 'float32')
train_dataset.fare_amount = train_dataset.fare_amount.astype(dtype = 'float32')


## === cell 12
train_dataset.info(memory_usage='deep')


## === cell 13
train_dataset.isnull().sum()


## === cell 14
len(train_dataset)


## === cell 15
print(f'Row count before drop-null operation - {train_dataset.shape[0]}')
train_dataset.dropna(inplace = True)
print(f'Row count after drop-null operation - {train_dataset.shape[0]}')


## === cell 16
train_dataset['pickup_datetime'] = pd.to_datetime(arg=train_dataset['pickup_datetime'], infer_datetime_format=True)
test_dataset['pickup_datetime'] = pd.to_datetime(arg=test_dataset['pickup_datetime'], infer_datetime_format=True)


## === cell 17
train_dataset.dtypes


## === cell 18
def add_new_date_time_features(dataset):
    dataset['hour'] = dataset.pickup_datetime.dt.hour
    dataset['day'] = dataset.pickup_datetime.dt.day
    dataset['month'] = dataset.pickup_datetime.dt.month
    dataset['year'] = dataset.pickup_datetime.dt.year
    dataset['day_of_week'] = dataset.pickup_datetime.dt.dayofweek
    return dataset

train_dataset = add_new_date_time_features(train_dataset)
test_dataset = add_new_date_time_features(test_dataset)


## === cell 19
train_dataset.describe()


## === cell 20
print(f'Rows before removing coordinate outliers - {train_dataset.shape[0]}')

train_dataset = train_dataset[train_dataset.pickup_longitude.between(test_dataset.pickup_longitude.min(), test_dataset.pickup_longitude.max())]
train_dataset = train_dataset[train_dataset.pickup_latitude.between(test_dataset.pickup_latitude.min(), test_dataset.pickup_latitude.max())]
train_dataset = train_dataset[train_dataset.dropoff_longitude.between(test_dataset.dropoff_longitude.min(), test_dataset.dropoff_longitude.max())]
train_dataset = train_dataset[train_dataset.dropoff_latitude.between(test_dataset.dropoff_latitude.min(), test_dataset.dropoff_latitude.max())]



print(f'Rows after removing coordinate outliers - {train_dataset.shape[0]}')


## === cell 21
train_dataset.describe()


## === cell 22
 train_dataset.fare_amount[train_dataset.fare_amount <= 0].count()


## === cell 23
print(f'Row count before elimination - {train_dataset.shape[0]}')
train_dataset = train_dataset[train_dataset.fare_amount > 0]
print(f'Row count after elimination - {train_dataset.shape[0]}')


## === cell 24
train_dataset.passenger_count[(train_dataset.passenger_count < 1) | (train_dataset.passenger_count > 8)].count()


## === cell 25
print(f'Row count before elimination - {train_dataset.shape[0]}')
train_dataset = train_dataset[train_dataset.passenger_count.between(0, 7)]
print(f'Row count after elimination - {train_dataset.shape[0]}')


## === cell 26
def degree_to_radion(degree):
    return degree*(np.pi/180)

def calculate_distance(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude):
    
    from_lat = degree_to_radion(pickup_latitude)
    from_long = degree_to_radion(pickup_longitude)
    to_lat = degree_to_radion(dropoff_latitude)
    to_long = degree_to_radion(dropoff_longitude)
    
    radius = 6371.01
    
    lat_diff = to_lat - from_lat
    long_diff = to_long - from_long

    a = np.sin(lat_diff / 2)**2 + np.cos(degree_to_radion(from_lat)) * np.cos(degree_to_radion(to_lat)) * np.sin(long_diff / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return radius * c

    


## === cell 27
train_dataset['distance'] = calculate_distance(train_dataset.pickup_latitude, train_dataset.pickup_longitude, train_dataset.dropoff_latitude, train_dataset.dropoff_longitude)
test_dataset['distance'] = calculate_distance(test_dataset.pickup_latitude, test_dataset.pickup_longitude, test_dataset.dropoff_latitude, test_dataset.dropoff_longitude)


## === cell 28
train_dataset.sort_values(by='distance')


## === cell 29
train_dataset[(train_dataset.distance == 0)].count()


## === cell 30
train_dataset[(train_dataset.pickup_latitude != train_dataset.dropoff_latitude) &
              (train_dataset.pickup_longitude != train_dataset.dropoff_latitude) &
              (train_dataset.distance == 0)].count()


## === cell 31
def add_distances_from_airport(dataset):
    jfk_coords = (40.639722, -73.778889)
    ewr_coords = (40.6925, -74.168611)
    lga_coords = (40.77725, -73.872611)
    print(dataset)
    dataset['pickup_jfk_distance'] = calculate_distance(jfk_coords[0], jfk_coords[1], dataset.pickup_latitude, dataset.pickup_longitude)
    dataset['dropof_jfk_distance'] = calculate_distance(jfk_coords[0], jfk_coords[1], dataset.dropoff_latitude, dataset.dropoff_longitude)
    
    dataset['pickup_ewr_distance'] = calculate_distance(ewr_coords[0], ewr_coords[1], dataset.pickup_latitude, dataset.pickup_longitude)
    dataset['dropof_ewr_distance'] = calculate_distance(ewr_coords[0], ewr_coords[1], dataset.dropoff_latitude, dataset.dropoff_longitude)
    
    dataset['pickup_lga_distance'] = calculate_distance(lga_coords[0], lga_coords[1], dataset.pickup_latitude, dataset.pickup_longitude)
    dataset['dropof_lga_distance'] = calculate_distance(lga_coords[0], lga_coords[1], dataset.dropoff_latitude, dataset.dropoff_longitude)
    
    return dataset


train_dataset = add_distances_from_airport(train_dataset)
test_dataset = add_distances_from_airport(test_dataset)


## === cell 32
sns.distplot(train_dataset.fare_amount)


## === cell 33
sns.jointplot(x='distance', y='fare_amount', data=train_dataset)


## === cell 36
sns.countplot(x='day_of_week', data=train_dataset)


## === cell 37
tc = train_dataset.pivot_table(index='day_of_week', columns='month', values='fare_amount')
sns.heatmap(data = tc)


## === cell 38
selected_predictors = [
    'pickup_longitude', 
    'pickup_latitude', 
    'dropoff_longitude', 
    'dropoff_latitude',
    'pickup_jfk_distance',
    'dropof_jfk_distance',
    'pickup_ewr_distance',
    'dropof_ewr_distance',
    'pickup_lga_distance',
    'dropof_lga_distance',
    'passenger_count',
    'hour',
    'day',
    'month',
    'year',
    'day_of_week',
    'distance'
]

X = train_dataset.loc[:, selected_predictors].values
y = train_dataset.iloc[:, 0].values
X_test_dataset = test_dataset.loc[:, selected_predictors].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/5)
print(f'X_test size - {len(X_test)}')


## === cell 39
train_mask = ~np.isnan(X_train).any(axis=1)
test_mask = ~np.isnan(X_test).any(axis=1)

X_train = X_train[train_mask]
y_train = y_train[train_mask]
X_test = X_test[test_mask]
y_test = y_test[test_mask]

rand_forest_regressor = RandomForestRegressor()
rand_forest_regressor.fit(X_train, y_train)

y_rand_forest_predict = rand_forest_regressor.predict(X_test)
print(f"random forest size - {len(y_rand_forest_predict)}")
random_forest_model_error = sqrt(mean_squared_error(y_test, y_rand_forest_predict))
print(f" Random Forest Mean Squared Error - {random_forest_model_error}")


## === cell 40
XGB_regressor = XGBRegressor()
XGB_regressor.fit(X_train, y_train)

y_XGB_predict = XGB_regressor.predict(X_test)
XGB_model_error = sqrt(mean_squared_error(y_test, y_XGB_predict))
print(f' XGBoost Mean Squared Error - {XGB_model_error}')


## === cell 41
sns.barplot(y=list(train_dataset.loc[:, selected_predictors].columns), x=list(XGB_regressor.feature_importances_))


## === cell 42
selected_predictors = [
    'pickup_longitude', 
    'pickup_latitude', 
    'dropoff_longitude', 
    'dropoff_latitude',
    'pickup_jfk_distance',
    'dropof_jfk_distance',
    'pickup_ewr_distance',
    'dropof_ewr_distance',
    'pickup_lga_distance',
    'dropof_lga_distance',
    'hour',
    'month',
    'year',
    'distance'
]

X = train_dataset.loc[:, selected_predictors].values
y = train_dataset.iloc[:, 0].values
X_test_dataset = test_dataset.loc[:, selected_predictors].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/5)


## === cell 45
XGB_regressor = XGBRegressor(learning_rate=0.07, max_depth=6, n_estimators=300)
XGB_regressor.fit(X_train, y_train)
y_XGB_predict = XGB_regressor.predict(X_test)

XGB_model_error = sqrt(mean_squared_error(y_test, y_XGB_predict))

print(f' XGBoost Mean Squared Error - {XGB_model_error}')


## === cell 46
sns.barplot(y=list(train_dataset.loc[:, selected_predictors].columns), x=list(XGB_regressor.feature_importances_))


## === cell 47
y_XGB_predict = XGB_regressor.predict(X_test_dataset)
test_dataset['fare_amount'] = y_XGB_predict

submission = pd.read_csv('../input/sample_submission.csv')
submission['fare_amount'] = y_XGB_predict
submission.head(10)


## === cell 48
sns.jointplot(x='distance', y='fare_amount', data=test_dataset)


## === cell 49
test_dataset[test_dataset.distance > 100]


## === cell 50
test_dataset[test_dataset.fare_amount > 150]


## === cell 51
submission.to_csv('submission_new.csv', index=False)
