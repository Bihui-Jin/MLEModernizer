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
sklearn-pandas==2.2.0

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

3.50273

# 6. Current score

4.44151

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
print(os.listdir("../input"))



## === cell 1
n_train = 100_000
n_features = 2
dtype_map = {
    'fare_amount': np.float32,
    'pickup_longitude': np.float64,
    'pickup_latitude': np.float64,
    'dropoff_longitude': np.float64,
    'dropoff_latitude': np.float64,
    'passenger_count': np.int8
}

df = pd.read_csv("../input/train.csv", nrows = n_train, parse_dates=["pickup_datetime"], dtype=dtype_map)

df_test =  pd.read_csv('../input/test.csv', parse_dates=["pickup_datetime"], dtype=dtype_map)


## === cell 2
df.describe()


## === cell 3
df.dtypes


## === cell 4
def add_travel_vector_features(df):
    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()

add_travel_vector_features(df)
add_travel_vector_features(df_test)
print(df.isnull().sum())
print('Old size: %d' % len(df))
df = df.dropna(how='any', axis='rows')
print('New size: %d' % len(df))


## === cell 5
min_year = df.pickup_datetime.apply(lambda t: t.year).min()
df['pickup_year'] = df.pickup_datetime.apply(lambda t: t.year - min_year)
df['pickup_hour'] = df.pickup_datetime.apply(lambda t: t.hour)
df['pickup_day'] = df.pickup_datetime.apply(lambda t: t.timetuple().tm_yday)

df_test['pickup_year'] = df_test.pickup_datetime.apply(lambda t: t.year - min_year)
df_test['pickup_hour'] = df_test.pickup_datetime.apply(lambda t: t.hour)
df_test['pickup_day'] = df_test.pickup_datetime.apply(lambda t: t.timetuple().tm_yday)


## === cell 6
from sklearn.model_selection import train_test_split

df_train, df_val = train_test_split(df, test_size=0.1)
len(df_val)


## === cell 7
def get_input_matrix(df):
    return np.column_stack((df.abs_diff_longitude, df.abs_diff_latitude,
                            df.dropoff_longitude, df.dropoff_latitude,
                            df.pickup_longitude, df.pickup_latitude,
                            df.passenger_count, df.pickup_year,
                            df.pickup_hour, df.pickup_day))

x_train, x_val = get_input_matrix(df_train), get_input_matrix(df_val)
y_train, y_val = np.array(df_train.fare_amount), np.array(df_val.fare_amount)


## === cell 8
from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(max_depth=25, n_estimators=500, oob_score=True, n_jobs=-1,min_samples_split=10, verbose=1)


## === cell 9
reg.fit(x_train, y_train)


## === cell 10
reg.oob_score_


## === cell 11
from sklearn.metrics import r2_score
y_pred = reg.predict(x_val)


## === cell 12
score = r2_score(y_val, y_pred)
score


## === cell 13
%matplotlib inline
import matplotlib.pyplot as plt

plt.scatter(y_val, y_pred, alpha=0.02)


## === cell 14
reg.score(x_val, y_val)


## === cell 15
from sklearn.metrics import mean_squared_error
mean_squared_error(y_val, y_pred)


## === cell 17
x_test = get_input_matrix(df_test)


## === cell 18
predictions = reg.predict(x_test)


## === cell 19
RFSubmission = pd.DataFrame({ 'key': df_test.key.ravel(),
                            'fare_amount': predictions })
RFSubmission.to_csv("RFSubmission.csv", index=False)
