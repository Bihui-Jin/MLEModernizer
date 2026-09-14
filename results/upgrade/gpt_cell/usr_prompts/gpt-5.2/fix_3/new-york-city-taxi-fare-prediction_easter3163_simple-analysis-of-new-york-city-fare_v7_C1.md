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

4.15448

# 6. Current score

81.03676

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 81.03051) has done: 'Diagnosis: The crash happens because `pandas==2.2.3` removed the deprecated `.dt.week` and `.dt.weekofyear` accessors, so `DatetimeProperties` no longer has a `week` attribute. The code in cell 4 uses these removed accessors when extracting time features from `pickup_datetime`.  
Patch summary: Replace `.dt.week` and `.dt.weekofyear` with the supported ISO week extraction via `.dt.isocalendar().week`, converting to a plain integer dtype to match prior behavior. Keep all other feature logic unchanged.  
Updated cells: Only cell 4 is modified.  
Compatibility notes for cell k+1: Cell 5 still uses `.dt.week`/`.dt.weekofyear` and raise the same error next; this patch keeps column names (`week`, `week_of_year`) and dtypes compatible so applying the same accessor replacement in cell 5 would be consistent.  
Assumptions: Using ISO week numbers is acceptable as the closest equivalent to the removed accessors and matches pandas’ recommended replacement.'
- What this solution (achieved 81.03676) has done: 'The crash happens because `Series.dt.week` and `Series.dt.weekofyear` were removed in recent pandas versions, so accessing them raises an `AttributeError`. In cell 4 you already use the supported replacement `dt.isocalendar().week`, so we should apply the same approach for `test` in cell 5. This keeps the produced feature semantics consistent between train and test, and preserves the same column names expected by later cells. The fix is limited to replacing the deprecated attributes with `isocalendar().week.astype("int64")`.'

# 9. Code solution

## === cell 0
import pandas as pd
train = pd.read_csv('../input/train.csv', nrows=300_000)
test = pd.read_csv('../input/test.csv')


## === cell 1
train.shape


## === cell 2
train.head()


## === cell 3
%matplotlib inline

import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt


## === cell 4
train["pickup_datetime"] = pd.to_datetime(train["pickup_datetime"])
train["hour"] = train["pickup_datetime"].dt.hour
train["day"] = train["pickup_datetime"].dt.day
train["week"] = train["pickup_datetime"].dt.isocalendar().week.astype("int64")
train["month"] = train["pickup_datetime"].dt.month
train["day_of_year"] = train["pickup_datetime"].dt.dayofyear
train["week_of_year"] = train["pickup_datetime"].dt.isocalendar().week.astype("int64")


## === cell 5
test["pickup_datetime"] = pd.to_datetime(test["pickup_datetime"])
test["hour"] = test["pickup_datetime"].dt.hour
test["day"] = test["pickup_datetime"].dt.day
test["week"] = test["pickup_datetime"].dt.isocalendar().week.astype("int64")
test["month"] = test["pickup_datetime"].dt.month
test["day_of_year"] = test["pickup_datetime"].dt.dayofyear
test["week_of_year"] = test["pickup_datetime"].dt.isocalendar().week.astype("int64")


## === cell 6
train.head()
train = train.dropna(how = 'any', axis='rows')


## === cell 7
train = train.loc[(train['fare_amount'] > 0) & (train['fare_amount'] < 200)]
train = train.loc[(train['pickup_longitude'] > -75) & (train['pickup_longitude'] < 75)]
train = train.loc[(train['pickup_latitude'] > 40) & (train['pickup_latitude'] < 45)]
train = train.loc[(train['dropoff_longitude'] > -75) & (train['dropoff_longitude'] < 75)]
train = train.loc[(train['dropoff_latitude'] > 40) & (train['dropoff_latitude'] < 45)]
train = train.loc[train['passenger_count'] <= 8]


## === cell 8
train['abs_diff_longitude'] = (train['pickup_longitude'] - train['dropoff_longitude']).abs()
train['abs_diff_latitude'] = (train['pickup_latitude'] - train['dropoff_latitude']).abs()


## === cell 9
test['abs_diff_longitude'] = (test['pickup_longitude'] - test['dropoff_longitude']).abs()
test['abs_diff_latitude'] = (test['pickup_latitude'] - test['dropoff_latitude']).abs()


## === cell 10
train.head()


## === cell 11
train.head()


## === cell 12
sns.barplot(data=train, x="passenger_count", y="fare_amount")


## === cell 13
feature_names = ['hour', 'passenger_count','abs_diff_longitude', 'abs_diff_latitude']
feature_names


## === cell 14
label_name = 'fare_amount'
label_name


## === cell 15
X_train = train[feature_names]
y_train = train[label_name]
X_test = test[feature_names]


## === cell 16
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import xgboost as xgb


## === cell 17
regr = LinearRegression()
regr.fit(X_train, y_train)
regr_prediction = regr.predict(X_test)


## === cell 18
knr = KNeighborsRegressor()
knr.fit(X_train, y_train)
knr_prediction = knr.predict(X_test)


## === cell 19
rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)
rfr_prediction = rfr.predict(X_test)


## === cell 20
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)


## === cell 21
params = {'max_depth':7,
          'eta':1,
          'silent':1,
          'objective':'reg:linear',
          'eval_metric':'rmse',
          'learning_rate':0.05
         }
num_rounds = 50


## === cell 22
xb = xgb.train(params, dtrain, num_rounds)


## === cell 23
y_pred_xgb = xb.predict(dtest)
print(y_pred_xgb)


## === cell 24
predictions = (regr_prediction + rfr_prediction + knr_prediction + 3 * y_pred_xgb) / 6


## === cell 25
predictions


## === cell 26
submission = pd.read_csv('../input/sample_submission.csv')
submission['fare_amount'] = predictions


## === cell 27
submission.head()


## === cell 28
submission.to_csv('./simplenewyorktaxi.csv', index=False)
