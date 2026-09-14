# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

3.39438

# 6. Current score

4.62418

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

train_df =  pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/train.csv', nrows = 10_000_000)
test_df =  pd.read_csv('/kaggle/input/new-york-city-taxi-fare-prediction/test.csv')


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_df.head()


## === cell 2
test_df.head()


## === cell 3
train_df.dtypes


## === cell 4
train_df.describe()


## === cell 5
num_rows = len(train_df)
train_df = train_df[(train_df['fare_amount'] > 0)]
print(f'Drop {num_rows - len(train_df)} rows')


## === cell 6
def change_outliers_by_range(df, column_name, min_range, max_range):
    before_len = df.shape[0]
    mask = (df[column_name].between(min_range,max_range))
    selected_rows = df[mask]
    changed_rows = before_len - selected_rows.shape[0]
    
    dtype_of_column = df[column_name].dtype
    mean_of_column = selected_rows[column_name].mean()
    if dtype_of_column == np.int64:
        mean_of_column = round(mean_of_column)
    
    df.loc[~mask, column_name] = mean_of_column
    return changed_rows
    

def change_outliers(df):
    print("Change", change_outliers_by_range(df, 'pickup_latitude', 40.5, 41.0), "rows by pickup lat")
    print("Change", change_outliers_by_range(df, 'dropoff_latitude', 40.5, 41.0), "rows by dropoff lat")
    print("Change", change_outliers_by_range(df, 'pickup_longitude', -74.3, -73.60), "rows by pickup long")
    print("Change", change_outliers_by_range(df, 'dropoff_longitude', -74.3, -73.6), "rows by dropoff long")
    print("Change", change_outliers_by_range(df, 'passenger_count', 1, 10), "rows by passenger cnt")
    

print("Training data outliers: ")
change_outliers(train_df)
print("\nTest data outliers: ")
change_outliers(test_df)


## === cell 7
train_df.describe()


## === cell 8
train_df.isnull().sum()


## === cell 9
def preprocess_data(df):
    airport_lat_long = (40.644600, -73.779700)
    la_guardia_airport_lat_long = (40.7733, -73.8718)
    near_airport = (((df["pickup_latitude"] <= airport_lat_long[0] + 0.005) & 
                (df["pickup_latitude"] >= airport_lat_long[0] - 0.005) & 
                (df["pickup_longitude"] <= airport_lat_long[1] + 0.005) & 
                (df["pickup_longitude"] >= airport_lat_long[1] - 0.005)) |
                
                ((df["pickup_latitude"] <= la_guardia_airport_lat_long[0] + 0.002) & 
                (df["pickup_latitude"] >= la_guardia_airport_lat_long[0] - 0.003) & 
                (df["pickup_longitude"] <= la_guardia_airport_lat_long[1] + 0.005) & 
                (df["pickup_longitude"] >= la_guardia_airport_lat_long[1] - 0.005))).astype(int)

    df['near_airport'] = near_airport
    
    df['manhattan_distance'] = (abs(df['pickup_longitude'] - df['dropoff_longitude']) +
                                 abs(df['pickup_latitude'] - df['dropoff_latitude']))
    
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['pickup_year'] = df['pickup_datetime'].dt.year
    df['pickup_month'] = df['pickup_datetime'].dt.month
    df['pickup_hour'] = df['pickup_datetime'].dt.hour
    df['pickup_day'] = df['pickup_datetime'].dt.dayofweek
    is_weekend = ((df["pickup_day"] >=5) & 
                    (df["pickup_day"] <=6)).astype(int)
    df['is_weekend'] = is_weekend
    
    is_holiday = (
        ((df['pickup_month'] == 12) & (df['pickup_datetime'].dt.day == 25)) |  
        ((df['pickup_month'] == 12) & (df['pickup_datetime'].dt.day == 26)) | 
        ((df['pickup_month'] == 12) & (df['pickup_datetime'].dt.day == 31)) |  
        ((df['pickup_month'] == 1) & (df['pickup_datetime'].dt.day == 1)) | 
        ((df['pickup_month'] == 7) & (df['pickup_datetime'].dt.day == 4))
    ).astype(int)

    df['is_holiday'] = is_holiday

preprocess_data(train_df)
preprocess_data(test_df)


## === cell 10
train_df.head()


## === cell 11
train_df["near_airport"].sum()


## === cell 12
train_df["is_holiday"].sum()


## === cell 13
train_df["is_weekend"].sum()


## === cell 14
features = ['pickup_latitude', 'pickup_longitude','dropoff_latitude', 'dropoff_longitude','near_airport', 'manhattan_distance',  'passenger_count','pickup_year', 'pickup_hour', 'is_weekend', "is_holiday"]
X = train_df[features].values
y = train_df['fare_amount'].values


## === cell 15
X_test = test_df[features].values


## === cell 17
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=69)


## === cell 18
print(X.shape)
print(y.shape)

print(X_train.shape)
print(y_train.shape)

print(X_val.shape)
print(y_val.shape)


## === cell 19
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)


## === cell 20
validation_predictions_lr = lr_model.predict(X_val)
validation_rmse_lr = np.sqrt(mean_squared_error(y_val, validation_predictions_lr))
print("Validation RMSE (Linear Regression):", validation_rmse_lr)

train_predictions_lr = lr_model.predict(X)
train_rmse_lr = np.sqrt(mean_squared_error(y, train_predictions_lr))
print("Training RMSE (Linear Regression):", train_rmse_lr)


## === cell 21
test_predictions_lr = lr_model.predict(X_test)

submission_df_lr =  pd.DataFrame(
    {'key': test_df.key, 'fare_amount': test_predictions_lr},
    columns = ['key', 'fare_amount'])
submission_df_lr.to_csv('lr_submission.csv', index=False)


## === cell 22
import xgboost as xgb

xgb_params = {
    'objective': 'reg:squarederror',  # Regression task
    'eval_metric': 'rmse',  # Evaluation metric (Root Mean Squared Error)
    'max_depth': 10,  # Maximum depth of the decision trees
    'subsample': 0.8,  # Subsample ratio of the training instances
    'colsample_bytree': 0.7,  # Subsample ratio of columns when constructing each tree
    'eta': 0.05,  # Learning rate
    'min_child_weight': 3,  # Minimum sum of instance weight needed in a child
    'gamma': 0.1,  # Minimum loss reduction required to make a further partition
    'seed': 42,  # Random seed for reproducibility
    'tree_method': 'hist',  # Use the histogram-based algorithm for better performance
    'nthread': -1,  # Use all available CPU cores
}

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val)
dtest = xgb.DMatrix(X_test)
dallTrain = xgb.DMatrix(X)

xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=100)


## === cell 23
validation_predictions_xgb = xgb_model.predict(dval)
validation_rmse_xgb = np.sqrt(mean_squared_error(y_val, validation_predictions_xgb))
print("Validation RMSE (XGBoost):", validation_rmse_xgb)

train_predictions_xgb = xgb_model.predict(dallTrain)
train_rmse_xgb = np.sqrt(mean_squared_error(y, train_predictions_xgb))
print("Train RMSE (XGBoost):", train_rmse_xgb)


## === cell 26
test_predictions_xgb = xgb_model.predict(dtest)

submission_df_xgb =  pd.DataFrame(
    {'key': test_df.key, 'fare_amount': test_predictions_xgb},
    columns = ['key', 'fare_amount'])
submission_df_xgb.to_csv('submission.csv', index=False)
