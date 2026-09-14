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

3.8

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

4.41355

# 6. Current score

5.94612

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 5.94612) has done: 'Diagnosis: In xgboost==2.0.3, the `Booster` attribute `best_ntree_limit` no longer exists (it was used in older versions with early stopping). The model still tracks the best iteration, and prediction should use `iteration_range` (or omit the limit) to be compatible with newer XGBoost. The crash occurs specifically at `model.predict(..., ntree_limit=model.best_ntree_limit)` in cell 32.

Patch summary: Update the prediction call to use `iteration_range=(0, model.best_iteration + 1)` when `best_iteration` is available; otherwise fall back to a plain `predict`. This preserves early-stopping semantics while avoiding the removed attribute.

Updated cells: Only cell 32 is modified.

Compatibility notes for cell k+1: The variable `prediction` remains a 1D numpy array of predictions with the same length/order as `df_test`, so cell 33 can build the submission exactly as before.

Assumptions: `best_iteration` is present when early stopping is used in this XGBoost version; if not, predicting without an iteration cap is an acceptable fallback and avoids crashing.'

# 9. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
plt.style.use('seaborn-whitegrid')


## === cell 1
df_train = pd.read_csv("../input/new-york-city-taxi-fare-prediction/train.csv", nrows=1_000_000,parse_dates=["pickup_datetime"])
df_train.head()


## === cell 2
df_train.describe()


## === cell 3
df_test = pd.read_csv("../input/new-york-city-taxi-fare-prediction/test.csv",parse_dates=["pickup_datetime"])
df_test.head()


## === cell 4
df_test.describe()


## === cell 5
print('Old size: %d' % len(df_train))
df_train = df_train[df_train.fare_amount>=0]
print('New size: %d' % len(df_train))


## === cell 6
print('Old size: %d' % len(df_train))
df_train = df_train.dropna(how = 'any', axis = 'rows')
print('New size: %d' % len(df_train))


## === cell 7
df_train[df_train.fare_amount < 80].fare_amount.hist(bins=100)
plt.xlabel('fare $USD')


## === cell 8
df_train['diff_long'] = (df_train.dropoff_longitude - df_train.pickup_longitude).abs()
df_train['diff_long'].describe()


## === cell 9
df_train['diff_lat'] = (df_train.dropoff_latitude - df_train.pickup_latitude).abs()
df_train['diff_lat'].describe()


## === cell 10
print('Old size: %d' % len(df_train))
df_train = df_train[(df_train.diff_long < 5.0) & (df_train.diff_lat < 5.0)]
print('New size: %d' % len(df_train))


## === cell 11
df_train['year'] = df_train.pickup_datetime.apply(lambda t: t.year)
df_train['weekday'] = df_train.pickup_datetime.apply(lambda t: t.weekday())
df_train['hour'] = df_train.pickup_datetime.apply(lambda t: t.hour)


## === cell 12
df_train.describe()


## === cell 13
df_train[['fare_amount', 'hour']].groupby(['hour'], as_index=False).mean().sort_values(by='fare_amount', ascending=False)


## === cell 14
df_train[['fare_amount', 'weekday']].groupby(['weekday'], as_index=False).mean().sort_values(by='fare_amount', ascending=False)


## === cell 15
df_train[['fare_amount', 'year']].groupby(['year'], as_index=False).mean().sort_values(by='fare_amount', ascending=False)


## === cell 16
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 12742 * np.arcsin(np.sqrt(a)) # 2*R*asin...


## === cell 17
df_train['distance'] = distance(df_train.pickup_latitude, df_train.pickup_longitude, \
                                      df_train.dropoff_latitude, df_train.dropoff_longitude)


## === cell 18
plt.figure(figsize=(15,8))
sns.heatmap(df_train.drop(['key','pickup_datetime'],axis=1).corr(),annot=True,fmt='.4f')


## === cell 19
plot = df_train.plot.scatter('distance', 'fare_amount')


## === cell 20
plot = df_train[(df_train.distance < 50) & (df_train.fare_amount < 100)].plot.scatter('distance', 'fare_amount',alpha=0.1)


## === cell 21
print('Old size: %d' % len(df_train))
df_train = df_train[(df_train.distance >= 0.1)]
print('New size: %d' % len(df_train))


## === cell 22
plot = df_train[(df_train.distance < 50) & (df_train.fare_amount < 100)].plot.scatter('distance', 'fare_amount',alpha=0.1)


## === cell 23
print('Old size: %d' % len(df_train))
df_train = df_train[(df_train.distance <= 50)]
print('New size: %d' % len(df_train))


## === cell 24
plot = df_train.plot.scatter('distance', 'fare_amount',alpha=0.1)


## === cell 25
print('Old size: %d' % len(df_train))
df_train = df_train[(df_train.fare_amount <= 200)]
print('New size: %d' % len(df_train))


## === cell 26
plot = df_train.plot.scatter('distance', 'fare_amount',alpha=0.1)


## === cell 27
features = ['year', 'hour', 'distance','passenger_count']
X = df_train[features].values
y = df_train['fare_amount'].values


## === cell 29
df_test['year'] = df_test.pickup_datetime.apply(lambda t: t.year)
df_test['hour'] = df_test.pickup_datetime.apply(lambda t: t.hour)
df_test['distance'] = distance(df_test.pickup_latitude, df_test.pickup_longitude, \
                                      df_test.dropoff_latitude, df_test.dropoff_longitude)


## === cell 30
X_kaggle_test = df_test[features].values


## === cell 32
from sklearn.model_selection import train_test_split
import xgboost as xgb

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=10, test_size=0.3
)


def XGBmodel(x_train, x_test, y_train, y_test):
    matrix_train = xgb.DMatrix(x_train, label=y_train)
    matrix_test = xgb.DMatrix(x_test, label=y_test)
    model = xgb.train(
        params={"objective": "reg:linear", "eval_metric": "rmse"},
        dtrain=matrix_train,
        num_boost_round=200,
        early_stopping_rounds=100,
        evals=[(matrix_test, "test")],
    )
    return model


model = XGBmodel(X_train, X_test, y_train, y_test)

dtest = xgb.DMatrix(X_kaggle_test)
if hasattr(model, "best_iteration") and model.best_iteration is not None:
    prediction = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))
else:
    prediction = model.predict(dtest)


## === cell 33
submission = pd.DataFrame({
        "key": df_test['key'],
        "fare_amount": prediction.round(2)
})

submission.to_csv('submission.csv',index=False)
submission
