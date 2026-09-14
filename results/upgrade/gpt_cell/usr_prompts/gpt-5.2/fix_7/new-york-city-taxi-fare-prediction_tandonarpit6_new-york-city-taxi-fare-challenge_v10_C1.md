# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.7

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

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
        (df["pickup_longitude"].between(-74.5, -72.8))
        & (df["dropoff_longitude"].between(-74.5, -72.8))
        & (df["pickup_latitude"].between(40.5, 41.8))
        & (df["dropoff_latitude"].between(40.5, 41.8))
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


def _bearing(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1.astype(float))
    lat1 = np.radians(lat1.astype(float))
    lon2 = np.radians(lon2.astype(float))
    lat2 = np.radians(lat2.astype(float))
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


X_train["pickup_datetime"] = pd.to_datetime(X_train["pickup_datetime"])

X_train["hour"] = X_train["pickup_datetime"].dt.hour
X_train["dayofweek"] = X_train["pickup_datetime"].dt.dayofweek
X_train["month"] = X_train["pickup_datetime"].dt.month
X_train["year"] = X_train["pickup_datetime"].dt.year
X_train["dayofyear"] = X_train["pickup_datetime"].dt.dayofyear

X_train["latitude_distance"] = (
    X_train["dropoff_latitude"] - X_train["pickup_latitude"]
).abs()
X_train["longitude_distance"] = (
    X_train["dropoff_longitude"] - X_train["pickup_longitude"]
).abs()

X_train["manhattan_approx"] = (
    X_train["latitude_distance"] + X_train["longitude_distance"]
)

X_train["haversine_km"] = _haversine_km(
    X_train["pickup_longitude"],
    X_train["pickup_latitude"],
    X_train["dropoff_longitude"],
    X_train["dropoff_latitude"],
)

X_train["bearing"] = _bearing(
    X_train["pickup_longitude"],
    X_train["pickup_latitude"],
    X_train["dropoff_longitude"],
    X_train["dropoff_latitude"],
)

X_train["_fare_amount_tmp"] = training_data["fare_amount"].values
X_train = X_train[(X_train["haversine_km"] > 0.05) & (X_train["haversine_km"] < 100.0)]

fare_per_km = X_train["_fare_amount_tmp"] / (X_train["haversine_km"] + 1e-6)
X_train = X_train[(fare_per_km > 0.5) & (fare_per_km < 100.0)]

training_data = training_data.loc[X_train.index].copy()
Y_train = training_data.copy()

X_train = X_train.drop(
    columns=[
        "key",
        "pickup_datetime",
        "_fare_amount_tmp",
    ]
)



## === cell 6
X_test["pickup_datetime"] = pd.to_datetime(X_test["pickup_datetime"])

X_test["hour"] = X_test["pickup_datetime"].dt.hour
X_test["dayofweek"] = X_test["pickup_datetime"].dt.dayofweek
X_test["month"] = X_test["pickup_datetime"].dt.month
X_test["year"] = X_test["pickup_datetime"].dt.year
X_test["dayofyear"] = X_test["pickup_datetime"].dt.dayofyear

X_test["latitude_distance"] = (
    X_test["dropoff_latitude"] - X_test["pickup_latitude"]
).abs()
X_test["longitude_distance"] = (
    X_test["dropoff_longitude"] - X_test["pickup_longitude"]
).abs()
X_test["manhattan_approx"] = X_test["latitude_distance"] + X_test["longitude_distance"]

X_test["haversine_km"] = _haversine_km(
    X_test["pickup_longitude"],
    X_test["pickup_latitude"],
    X_test["dropoff_longitude"],
    X_test["dropoff_latitude"],
)

X_test["bearing"] = _bearing(
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



## --- ERROR in cell 9, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3890959764.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     15[0m [0mmodel[0m[0;34m.[0m[0mfit[0m[0;34m([0m[0mX_train[0m[0;34m,[0m [0mY_train[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     16[0m [0;34m[0m[0m
[0;32m---> 17[0;31m [0mY_pred[0m [0;34m=[0m [0mmodel[0m[0;34m.[0m[0mpredict[0m[0;34m([0m[0mX_test[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     18[0m [0;34m[0m[0m
[1;32m     19[0m [0;31m# Keep the same safe post-processing for RMSE stability.[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py[0m in [0;36mpredict[0;34m(self, X, output_margin, validate_features, base_margin, iteration_range)[0m
[1;32m   1166[0m             [0;32mif[0m [0mself[0m[0;34m.[0m[0m_can_use_inplace_predict[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1167[0m                 [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1168[0;31m                     predts = self.get_booster().inplace_predict(
[0m[1;32m   1169[0m                         [0mdata[0m[0;34m=[0m[0mX[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1170[0m                         [0miteration_range[0m[0;34m=[0m[0miteration_range[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/core.py[0m in [0;36minplace_predict[0;34m(self, data, iteration_range, predict_type, missing, validate_features, base_margin, strict_shape)[0m
[1;32m   2416[0m             [0mdata[0m[0;34m,[0m [0mfns[0m[0;34m,[0m [0m_[0m [0;34m=[0m [0m_transform_pandas_df[0m[0;34m([0m[0mdata[0m[0;34m,[0m [0menable_categorical[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   2417[0m             [0;32mif[0m [0mvalidate_features[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 2418[0;31m                 [0mself[0m[0;34m.[0m[0m_validate_features[0m[0;34m([0m[0mfns[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   2419[0m         [0;32mif[0m [0m_is_list[0m[0;34m([0m[0mdata[0m[0;34m)[0m [0;32mor[0m [0m_is_tuple[0m[0;34m([0m[0mdata[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   2420[0m             [0mdata[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0marray[0m[0;34m([0m[0mdata[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/core.py[0m in [0;36m_validate_features[0;34m(self, feature_names)[0m
[1;32m   2968[0m                 )
[1;32m   2969[0m [0;34m[0m[0m
[0;32m-> 2970[0;31m             [0;32mraise[0m [0mValueError[0m[0;34m([0m[0mmsg[0m[0;34m.[0m[0mformat[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mfeature_names[0m[0;34m,[0m [0mfeature_names[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   2971[0m [0;34m[0m[0m
[1;32m   2972[0m     def get_split_value_histogram(

[0;31mValueError[0m: feature_names mismatch: ['fare_amount', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count', 'hour', 'dayofweek', 'month', 'year', 'dayofyear', 'latitude_distance', 'longitude_distance', 'manhattan_approx', 'haversine_km', 'bearing'] ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count', 'hour', 'dayofweek', 'month', 'year', 'dayofyear', 'latitude_distance', 'longitude_distance', 'manhattan_approx', 'haversine_km', 'bearing']
expected fare_amount in input data

## === cell 10
from matplotlib import pyplot as plt
from xgboost import plot_importance

plot_importance(model)
plt.show()
