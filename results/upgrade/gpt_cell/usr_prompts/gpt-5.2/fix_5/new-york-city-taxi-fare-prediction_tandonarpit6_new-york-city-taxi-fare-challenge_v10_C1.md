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

8.83144

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.70062) has done: 'Diagnosis: The crash happens in cell 8 because `xgboost.XGBRegressor.predict()` (xgboost 2.0.3) does not accept a `data=` keyword argument; it expects the feature matrix as the first positional argument (`X`). This is an API mismatch (confusing the scikit-learn wrapper with the lower-level `xgb.DMatrix` API).  

Patch summary: Update the prediction call to pass `X_test` positionally (`model.predict(X_test)`), keeping the same model, training call, and output variable (`Y_pred`) so downstream cells continue to work unchanged.  

Updated cells: Only cell 8 is modified.  

Compatibility notes for cell k+1: Cell 9 uses `plot_importance(model)` and expects `model` to be a fitted XGBoost model; this remains true. `Y_pred` is still created for any later cells that might use it.  

Assumptions: `X_train`, `Y_train`, and `X_test` are valid numeric pandas objects as prepared in earlier cells, and `model.fit()` succeeds as before.'
- What this solution (achieved 9.67126) has done: 'Your current score (4.70062) is worse than the target (4.01327), so we should improve it with minimal, safe changes that keep the same overall approach (XGBoost regressor on simple engineered features). The biggest score gain for this competition typically comes from removing obviously bad training rows (invalid coordinates/passenger counts and extreme fares) and using a slightly more appropriate XGBoost configuration (objective/trees/learning rate) without changing the modeling family. I also ensure `Y_train` is a 1D vector (not a 2D DataFrame) and clip negative predictions to 0, which improves RMSE stability. Paths and the submission format remain unchanged, and the script still writes a valid `.csv`.'
- What this solution (achieved 9.75042) has done: 'To move RMSE down toward the 4.01327 target (your current 9.67126 is worse, and lower is better), the smallest high-impact change is to add one more standard, lightweight feature: haversine distance (great-circle distance), while keeping the same XGBoost regressor approach and training loop. I also ensure the test set gets identical feature engineering and apply the same geographic validity filter to training (already present) while keeping your existing columns and submission format unchanged. This preserves your model family and core workflow but typically yields a large RMSE improvement compared to raw lat/long + abs deltas. The script still run end-to-end within time by keeping `nrows=2_000_000` and writing a valid `sample_submission.csv`.'
- What this solution (achieved 8.83144) has done: 'Your RMSE (9.75042) is much worse than the target (4.01327), so we should make a small, high-impact improvement without changing the core model family (XGBRegressor) or training loop. The biggest issue is that the current feature set accidentally drops `passenger_count` and also drops the raw pickup coordinates, both of which are very informative for this task; adding them back is minimal feature-engineering change and typically reduces RMSE substantially. I also add simple, standard datetime parts (dayofweek, month) while keeping the same approach and submission semantics. All paths remain unchanged, runtime stays within limits, and the script still writes a valid `sample_submission.csv` with `key,fare_amount`.'

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
training_data



## === cell 3
X_train = training_data.copy()
X_test = test_data.copy()
Y_train = training_data.copy()




## === cell 4
def _clean_train(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(
        subset=[
            "fare_amount",
            "pickup_datetime",
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
            "passenger_count",
        ]
    ).copy()

    df = df[(df["fare_amount"] > 0) & (df["fare_amount"] < 250)]
    df = df[(df["passenger_count"] >= 1) & (df["passenger_count"] <= 6)]

    df = df[
        (df["pickup_longitude"].between(-75, -72))
        & (df["dropoff_longitude"].between(-75, -72))
    ]
    df = df[
        (df["pickup_latitude"].between(40, 42))
        & (df["dropoff_latitude"].between(40, 42))
    ]

    same_coord = (df["pickup_longitude"] == df["dropoff_longitude"]) & (
        df["pickup_latitude"] == df["dropoff_latitude"]
    )
    df = df[~same_coord]
    return df


training_data = _clean_train(training_data)

X_train = training_data.copy()
Y_train = training_data.copy()




## === cell 5
def _haversine_km(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1.astype(float))
    lat1 = np.radians(lat1.astype(float))
    lon2 = np.radians(lon2.astype(float))
    lat2 = np.radians(lat2.astype(float))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c


X_train["pickup_datetime"] = pd.to_datetime(X_train["pickup_datetime"])

X_train["hour"] = X_train["pickup_datetime"].dt.hour
X_train["dayofweek"] = X_train["pickup_datetime"].dt.dayofweek
X_train["month"] = X_train["pickup_datetime"].dt.month

X_train["latitude_distance"] = abs(
    X_train["dropoff_latitude"] - X_train["pickup_latitude"]
)
X_train["longitude_distance"] = abs(
    X_train["dropoff_longitude"] - X_train["pickup_longitude"]
)
X_train["haversine_km"] = _haversine_km(
    X_train["pickup_longitude"],
    X_train["pickup_latitude"],
    X_train["dropoff_longitude"],
    X_train["dropoff_latitude"],
)

X_train = X_train.drop(
    columns=[
        "key",
        "fare_amount",
        "pickup_datetime",
    ]
)



## === cell 6
X_test["pickup_datetime"] = pd.to_datetime(X_test["pickup_datetime"])

X_test["hour"] = X_test["pickup_datetime"].dt.hour
X_test["dayofweek"] = X_test["pickup_datetime"].dt.dayofweek
X_test["month"] = X_test["pickup_datetime"].dt.month

X_test["latitude_distance"] = abs(
    X_test["dropoff_latitude"] - X_test["pickup_latitude"]
)
X_test["longitude_distance"] = abs(
    X_test["dropoff_longitude"] - X_test["pickup_longitude"]
)
X_test["haversine_km"] = _haversine_km(
    X_test["pickup_longitude"],
    X_test["pickup_latitude"],
    X_test["dropoff_longitude"],
    X_test["dropoff_latitude"],
)

X_test = X_test.drop(
    columns=[
        "key",
        "pickup_datetime",
    ]
)



## === cell 7
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
Y_train = Y_train["fare_amount"]



## === cell 8
"""
from keras import models
from keras import layers
from keras import optimizers
from keras.layers import Dropout

model=models.Sequential()
model.add(layers.Dense(512,activation='relu',input_shape=(X_train.shape[1],)))
model.add(Dropout(0.2))
model.add(layers.Dense(512,activation='relu'))
model.add(Dropout(0.2))
model.add(layers.Dense(1))

rmsprop=optimizers.RMSprop(lr=0.001)

model.compile(optimizer=rmsprop,loss='mse',metrics=['mae'])

model.fit(X_train,Y_train,epochs=4,batch_size=512)

Y_pred=model.predict(X_test)
"""



## === cell 9
import xgboost as xgb

model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=600,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

Y_pred = np.clip(Y_pred, 0, None)



## === cell 10
from matplotlib import pyplot as plt
from xgboost import plot_importance

plot_importance(model)
plt.show()



## === cell 11
submission = test_data.copy()
submission = submission.drop(
    columns=[
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]
)
submission["fare_amount"] = Y_pred



## === cell 12
submission.to_csv("sample_submission.csv", index=False)
