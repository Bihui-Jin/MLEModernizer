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

4.01327

# 6. Current score

4.6571

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.66245) has done: 'I fixed the XGBoost prediction call (removed the wrong keyword), flattened the target array, added a few reasonable default hyper‑parameters, and ensured the submission file is written with the required `key` and `fare_amount` columns to `submission.csv`. These changes resolve the runtime errors and produce a valid Kaggle submission while preserving the original preprocessing and model logic.'
- What this solution (achieved 4.65935) has done: 'The changes add a Euclidean “distance” feature to capture travel length, split the training data into a validation set, and enable XGBoost early‑stopping with a slightly larger tree ensemble and lower learning rate. These tweaks keep the original preprocessing and model structure while improving generalisation, which should lower the RMSE toward the target value. The submission file generation is unchanged apart from preserving the required `key` column.'
- What this solution (achieved 4.60122) has done: 'I keep the overall model and training pipeline unchanged but add more informative geographic features (the Haversine distance) and retain the original latitude/longitude and passenger count columns, which were previously dropped. These additional features usually improve fare prediction modestly. I also raise the maximum number of trees and early‑stopping rounds so the model can fully exploit the richer feature set, while early stopping prevents over‑fitting. These minimal adjustments should lower the RMSE toward the target without altering the core logic.'
- What this solution (achieved 4.65271) has done: 'I keep the overall pipeline unchanged and only adjust the XGBoost regressor to better capture the relationships in the data. By increasing the tree depth slightly and lowering the learning rate, the model can fit the richer distance‑based features more precisely, which is expected to reduce the RMSE and move the score closer to the target. The rest of the code—including feature engineering, train/validation split, and submission creation—remains the same.'
- What this solution (achieved 4.6571) has done: 'I keep the overall pipeline unchanged while adding a few modest features that often help taxi‑fare models (Manhattan distance and sine/cosine encoding of the hour) and slightly tuning the XGBoost hyper‑parameters (more trees, a bit deeper, lower learning rate). These changes are small, preserve the original logic, and are expected to lower the RMSE toward the target value.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

print(os.listdir("../input"))




## === cell 1
training_data = pd.read_csv("../input/train.csv", nrows=2000000)
test_data = pd.read_csv("../input/test.csv")




## === cell 2
X_train = training_data.copy()
X_test = test_data.copy()
Y_train = training_data.copy()




## === cell 3
X_train["pickup_datetime"] = pd.to_datetime(X_train["pickup_datetime"])
X_train["hour"] = X_train["pickup_datetime"].dt.hour

X_train["latitude_distance"] = abs(
    X_train["dropoff_latitude"] - X_train["pickup_latitude"]
)
X_train["longitude_distance"] = abs(
    X_train["dropoff_longitude"] - X_train["pickup_longitude"]
)
X_train["euclidean_distance"] = np.sqrt(
    X_train["latitude_distance"] ** 2 + X_train["longitude_distance"] ** 2
)

X_train["manhattan_distance"] = (
    X_train["latitude_distance"] + X_train["longitude_distance"]
)

X_train["hour_sin"] = np.sin(2 * np.pi * X_train["hour"] / 24)
X_train["hour_cos"] = np.cos(2 * np.pi * X_train["hour"] / 24)

R = 6371.0  # Earth radius in kilometers
lat1 = np.radians(X_train["pickup_latitude"])
lon1 = np.radians(X_train["pickup_longitude"])
lat2 = np.radians(X_train["dropoff_latitude"])
lon2 = np.radians(X_train["dropoff_longitude"])
dlat = lat2 - lat1
dlon = lon2 - lon1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arcsin(np.sqrt(a))
X_train["haversine_distance"] = R * c

X_train = X_train.drop(
    columns=[
        "key",
        "fare_amount",
        "pickup_datetime",
    ]
)




## === cell 4
X_test["pickup_datetime"] = pd.to_datetime(X_test["pickup_datetime"])
X_test["hour"] = X_test["pickup_datetime"].dt.hour

X_test["latitude_distance"] = abs(
    X_test["dropoff_latitude"] - X_test["pickup_latitude"]
)
X_test["longitude_distance"] = abs(
    X_test["dropoff_longitude"] - X_test["pickup_longitude"]
)
X_test["euclidean_distance"] = np.sqrt(
    X_test["latitude_distance"] ** 2 + X_test["longitude_distance"] ** 2
)

X_test["manhattan_distance"] = (
    X_test["latitude_distance"] + X_test["longitude_distance"]
)

X_test["hour_sin"] = np.sin(2 * np.pi * X_test["hour"] / 24)
X_test["hour_cos"] = np.cos(2 * np.pi * X_test["hour"] / 24)

lat1_t = np.radians(X_test["pickup_latitude"])
lon1_t = np.radians(X_test["pickup_longitude"])
lat2_t = np.radians(X_test["dropoff_latitude"])
lon2_t = np.radians(X_test["dropoff_longitude"])
dlat_t = lat2_t - lat1_t
dlon_t = lon2_t - lon1_t
a_t = (
    np.sin(dlat_t / 2) ** 2 + np.cos(lat1_t) * np.cos(lat2_t) * np.sin(dlon_t / 2) ** 2
)
c_t = 2 * np.arcsin(np.sqrt(a_t))
X_test["haversine_distance"] = R * c_t

X_test = X_test.drop(
    columns=[
        "key",
        "pickup_datetime",
    ]
)




## === cell 5
Y_train = Y_train.drop(
    columns=[
        "key",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]
)




## === cell 6
"""
# The Keras neural‑network code is kept for reference but not used.
from keras import models, layers, optimizers
from keras.layers import Dropout

model_nn = models.Sequential()
model_nn.add(layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)))
model_nn.add(Dropout(0.2))
model_nn.add(layers.Dense(512, activation='relu'))
model_nn.add(Dropout(0.2))
model_nn.add(layers.Dense(1))

rmsprop = optimizers.RMSprop(lr=0.001)

model_nn.compile(optimizer=rmsprop, loss='mse', metrics=['mae'])
model_nn.fit(X_train, Y_train["fare_amount"], epochs=4, batch_size=512)
Y_pred = model_nn.predict(X_test).ravel()
"""




## === cell 7
import xgboost as xgb
from sklearn.model_selection import train_test_split

y_array = Y_train["fare_amount"].values.ravel()

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_array, test_size=0.2, random_state=42
)

model = xgb.XGBRegressor(
    objective="reg:squarederror",
    eval_metric="rmse",
    n_estimators=3000,  # increased pool; early stopping will cap actual trees
    max_depth=9,  # slightly deeper trees
    learning_rate=0.02,  # lower learning rate for smoother fitting
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=4,
    random_state=42,
    verbosity=0,
)

model.fit(
    X_tr,
    y_tr,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=False,
)

Y_pred = model.predict(X_test)




## === cell 8
from matplotlib import pyplot as plt
from xgboost import plot_importance

plot_importance(model)
plt.show()




## === cell 9
submission = test_data[["key"]].copy()
submission["fare_amount"] = Y_pred




## === cell 10
submission.to_csv("submission.csv", index=False)
