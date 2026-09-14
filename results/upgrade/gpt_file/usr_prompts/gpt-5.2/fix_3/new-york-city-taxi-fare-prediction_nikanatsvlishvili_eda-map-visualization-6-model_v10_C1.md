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

3.10

# 3. Installed packages

folium==0.20.0
geopandas==0.14.4
haversine==2.9.0
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
tf_keras==2.18.0

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

4.46748

# 6. Current score

439.94995

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 439.94995) has done: 'I fix the TensorFlow/protobuf import crash by switching all Keras/TensorFlow imports to the already-installed `tf_keras` package (this avoids the `MessageFactory.GetPrototype` error in this environment). Then I fix the feature mismatch causing the `KeyError` by creating `fare_to_dist_ratio` for the test set too (since it exists in train and is used in `feature_cols`). Finally, I keep the same model/training logic but ensure the pipeline runs end-to-end and always writes a valid `/kaggle/working/submission.csv` with columns `key,fare_amount`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

try:
    get_ipython().run_line_magic("matplotlib", "inline")
except Exception:
    pass

import seaborn as sns
import math
from math import sqrt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import tf_keras as keras
from tf_keras import Sequential
from tf_keras.layers import Dense, Dropout, BatchNormalization
from tf_keras.optimizers import Adam

try:
    keras.utils.set_random_seed(42)
except Exception:
    pass
np.random.seed(42)

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:50]:
        print(os.path.join(dirname, filename))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/train.csv",
    nrows=100000,
    parse_dates=["pickup_datetime"],
)
test = pd.read_csv("../input/new-york-city-taxi-fare-prediction/test.csv")



## === cell 2
print(train.shape)
print(test.shape)



## === cell 3
train.head()



## === cell 4
train.dtypes



## === cell 5
train.describe()



## === cell 6
print(train.isnull().sum())



## === cell 7
train = train.dropna(how="any", axis="rows")



## === cell 8
print("Old size: %d" % len(train))



## === cell 9
train = train.drop(train[train.fare_amount < 2.5].index, axis=0)
train = train.drop(train[train.fare_amount > 300].index, axis=0)



## === cell 10
train = train.drop(train[train["passenger_count"] > 6].index, axis=0)
train = train.drop(train[train["passenger_count"] < 0].index, axis=0)



## === cell 11
train = train.drop(train[train["pickup_latitude"] < -90].index, axis=0)
train = train.drop(train[train["pickup_latitude"] > 90].index, axis=0)



## === cell 12
train = train.drop(train[train["pickup_longitude"] < -180].index, axis=0)
train = train.drop(train[train["pickup_longitude"] > 180].index, axis=0)



## === cell 13
train = train.drop(train[train["dropoff_latitude"] < -90].index, axis=0)
train = train.drop(train[train["dropoff_latitude"] > 90].index, axis=0)

train = train.drop(train[train["dropoff_longitude"] < -180].index, axis=0)
train = train.drop(train[train["dropoff_longitude"] > 180].index, axis=0)




## === cell 14
def select_outside_boundingbox(df, BB):
    filter_df = df.loc[
        (df["pickup_longitude"] < BB[0])
        | (df["pickup_longitude"] > BB[1])
        | (df["pickup_latitude"] < BB[2])
        | (df["pickup_latitude"] > BB[3])
        | (df["dropoff_longitude"] < BB[0])
        | (df["dropoff_longitude"] > BB[1])
        | (df["dropoff_latitude"] < BB[2])
        | (df["dropoff_latitude"] > BB[3])
    ]
    return filter_df


NYC_BB = (-74.5, -72.8, 40.5, 41.8)



## === cell 15
outliers = select_outside_boundingbox(train, NYC_BB)
outliers.head()



## === cell 16
train = train.drop(outliers.index, axis=0)



## === cell 17
print("New size: %d" % len(train))



## === cell 18
pass



## === cell 19
test.dtypes



## === cell 20
train["loc1"] = train[["pickup_latitude", "pickup_longitude"]].apply(tuple, axis=1)
train["loc2"] = train[["dropoff_latitude", "dropoff_longitude"]].apply(tuple, axis=1)



## === cell 21
import haversine as hs

train["H_Distance"] = train.apply(lambda row: hs.haversine(row.loc1, row.loc2), axis=1)




## === cell 22
def chebyshev(pickup_long, dropoff_long, pickup_lat, dropoff_lat):
    return np.maximum(
        np.absolute(pickup_long - dropoff_long), np.absolute(pickup_lat - dropoff_lat)
    )


train["Chebyshev"] = chebyshev(
    train["pickup_longitude"],
    train["dropoff_longitude"],
    train["pickup_latitude"],
    train["dropoff_latitude"],
)



## === cell 23
train.head()



## === cell 24
iso_week = train["pickup_datetime"].dt.isocalendar().week.astype(np.int16)

train["hour"] = train.pickup_datetime.dt.hour
train["day_of_week"] = train.pickup_datetime.dt.weekday
train["day_of_month"] = train.pickup_datetime.dt.day
train["week"] = iso_week
train["month"] = train.pickup_datetime.dt.month
train["year"] = train.pickup_datetime.dt.year - 2000

train["minute"] = train["pickup_datetime"].dt.minute
train["second"] = train["pickup_datetime"].dt.second
train["dayofyear"] = train["pickup_datetime"].dt.dayofyear



## === cell 25
train.head()



## === cell 26
train["fare_to_dist_ratio"] = train["fare_amount"] / (train["H_Distance"] + 0.0001)



## === cell 27
train = train.drop(train[train["loc1"] == train["loc2"]].index, axis=0)




## === cell 28
def add_distances_from_airport(dataset):
    jfk_coords = (40.639722, -73.778889)
    ewr_coords = (40.6925, -74.168611)
    lga_coords = (40.77725, -73.872611)

    dataset["pickup_jfk_distance"] = dataset.apply(
        lambda row: hs.haversine(jfk_coords, row.loc1), axis=1
    )
    dataset["dropof_jfk_distance"] = dataset.apply(
        lambda row: hs.haversine(jfk_coords, row.loc2), axis=1
    )

    dataset["pickup_ewr_distance"] = dataset.apply(
        lambda row: hs.haversine(ewr_coords, row.loc1), axis=1
    )
    dataset["dropof_ewr_distance"] = dataset.apply(
        lambda row: hs.haversine(ewr_coords, row.loc2), axis=1
    )

    dataset["pickup_lga_distance"] = dataset.apply(
        lambda row: hs.haversine(lga_coords, row.loc1), axis=1
    )
    dataset["dropof_lga_distance"] = dataset.apply(
        lambda row: hs.haversine(lga_coords, row.loc2), axis=1
    )

    return dataset


train = add_distances_from_airport(train)



## === cell 29
test["pickup_datetime"] = pd.to_datetime(test["pickup_datetime"])

test["loc1"] = test[["pickup_latitude", "pickup_longitude"]].apply(tuple, axis=1)
test["loc2"] = test[["dropoff_latitude", "dropoff_longitude"]].apply(tuple, axis=1)

test["H_Distance"] = test.apply(lambda row: hs.haversine(row.loc1, row.loc2), axis=1)

test["Chebyshev"] = chebyshev(
    test["pickup_longitude"],
    test["dropoff_longitude"],
    test["pickup_latitude"],
    test["dropoff_latitude"],
)

test_iso_week = test["pickup_datetime"].dt.isocalendar().week.astype(np.int16)

test["hour"] = test.pickup_datetime.dt.hour
test["day_of_week"] = test.pickup_datetime.dt.weekday
test["day_of_month"] = test.pickup_datetime.dt.day
test["week"] = test_iso_week
test["month"] = test.pickup_datetime.dt.month
test["year"] = test.pickup_datetime.dt.year - 2000

test["minute"] = test["pickup_datetime"].dt.minute
test["second"] = test["pickup_datetime"].dt.second
test["dayofyear"] = test["pickup_datetime"].dt.dayofyear

test = add_distances_from_airport(test)

test["fare_to_dist_ratio"] = 0.0 / (test["H_Distance"] + 0.0001)




## === cell 30
def downcast(df):
    df_int = df.select_dtypes(include=["int64", "int32", "int16", "int8", "int"])
    df[df_int.columns] = df_int.apply(pd.to_numeric, downcast="unsigned")

    df_float = df.select_dtypes(include=["float64", "float32", "float16", "float"])
    df[df_float.columns] = df_float.apply(pd.to_numeric, downcast="float")

    return df


downcast(train)
downcast(test)
train.dtypes



## === cell 31
train.plot(y="pickup_latitude", x="pickup_longitude", kind="scatter", alpha=0.7, s=0.02)
city_long_border = (-74.03, -73.75)
city_lat_border = (40.63, 40.85)
plt.title("Pickups Data")

plt.ylim(city_lat_border)
plt.xlim(city_long_border)
plt.show()



## === cell 32
train.plot(
    y="dropoff_latitude", x="dropoff_longitude", kind="scatter", alpha=0.5, s=0.02
)
city_long_border = (-74.03, -73.75)
city_lat_border = (40.63, 40.85)
plt.title("Dropoff Data")

plt.ylim(city_lat_border)
plt.xlim(city_long_border)
plt.show()



## === cell 33
import folium

long_trips = train[train["H_Distance"] >= 10]
drop_map = folium.Map(location=[40.730610, -73.935242], zoom_start=12)

for index, row in long_trips.iterrows():
    folium.CircleMarker(
        [row["dropoff_latitude"], row["dropoff_longitude"]],
        radius=3,
        color="green",
        fill_opacity=0.9,
    ).add_to(drop_map)

for index, row in long_trips.iterrows():
    folium.CircleMarker(
        [row["pickup_latitude"], row["pickup_longitude"]],
        radius=3,
        color="blue",
        fill_opacity=0.9,
    ).add_to(drop_map)

drop_map



## === cell 34
drop_cols = ["key", "fare_amount", "pickup_datetime", "loc1", "loc2"]
feature_cols = [c for c in train.columns if c not in drop_cols]

X = train[feature_cols].copy()
y = train["fare_amount"].astype(np.float32).copy()

X_test = test[feature_cols].copy()

X = X.apply(pd.to_numeric, errors="coerce")
X_test = X_test.apply(pd.to_numeric, errors="coerce")

X = X.fillna(0.0)
X_test = X_test.fillna(0.0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_scaled, y.values, test_size=0.2, random_state=42
)



## === cell 35
model = Sequential(
    [
        Dense(256, activation="relu", input_shape=(X_tr.shape[1],)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dense(1, activation="linear"),
    ]
)

model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")

history = model.fit(
    X_tr,
    y_tr,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=1024,
    verbose=2,
)

val_pred = model.predict(X_val, batch_size=4096).reshape(-1)
rmse = mean_squared_error(y_val, val_pred, squared=False)
print("Validation RMSE:", rmse)



## === cell 36
test_pred = model.predict(X_test_scaled, batch_size=4096).reshape(-1)

test_pred = np.clip(test_pred, 0.0, None)

submission = pd.DataFrame({"key": test["key"], "fare_amount": test_pred})
submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
print(submission.head())
