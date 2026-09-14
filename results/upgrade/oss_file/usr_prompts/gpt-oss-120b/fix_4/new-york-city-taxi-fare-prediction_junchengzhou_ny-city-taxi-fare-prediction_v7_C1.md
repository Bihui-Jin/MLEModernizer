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

3.7

# 3. Installed packages

geopandas==0.14.4
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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
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

4.12199

# 6. Current score

6.23608

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 15.27775) has done: 'I fixed the feature‑leakage issue by dropping the target column before scaling, corrected the scaler mismatch, and replaced the failing TensorFlow import with a plain Keras import that works in the given environment. These minimal changes let the pipeline run end‑to‑end and generate a proper `submission.csv` while keeping the original neural‑network architecture unchanged.'
- What this solution (achieved 6.23608) has done: 'I fixed the import errors by switching to TensorFlow’s Keras API, replaced the custom RMSE metric with TensorFlow’s built‑in `RootMeanSquaredError`, and updated the model compilation to use this metric. These changes unblock training so the model actually learns from the data, which lower the RMSE toward the target score while keeping the original architecture untouched.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

print(os.listdir("../input"))




## === cell 1
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from math import radians, cos, sin, asin, sqrt
import warnings

warnings.filterwarnings("ignore")




## === cell 2
train = pd.read_csv("../input/train.csv", nrows=10_000_000)




## === cell 3
train.head()




## === cell 4
test = pd.read_csv("../input/test.csv")




## === cell 5
test.head()




## === cell 6
train.dtypes




## === cell 7
def haversine(
    lon1, lat1, lon2, lat2
):  # longitude1, latitude1, longitude2, latitude2 (decimal degrees)
    """
    Calculate the great‑circle distance between two points on the earth (specified in decimal degrees)
    """
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Earth radius in kilometers
    return c * r




## === cell 8
def add_travel_distance_vector_features(df):
    df["distance"] = haversine(
        df["dropoff_longitude"],
        df["dropoff_latitude"],
        df["pickup_longitude"],
        df["pickup_latitude"],
    )
    df["log_distance"] = np.log1p(df["distance"])


add_travel_distance_vector_features(train)
add_travel_distance_vector_features(test)




## === cell 9
train.dtypes




## === cell 10
train.drop(
    [
        "dropoff_longitude",
        "dropoff_latitude",
        "pickup_longitude",
        "pickup_latitude",
        "pickup_datetime",
    ],
    axis=1,
    inplace=True,
)
test.drop(
    [
        "dropoff_longitude",
        "dropoff_latitude",
        "pickup_longitude",
        "pickup_latitude",
        "pickup_datetime",
    ],
    axis=1,
    inplace=True,
)




## === cell 11
train.head()




## === cell 12
train.isnull().sum()




## === cell 13
train.dropna(how="any", axis="rows", inplace=True)




## === cell 14
train.describe().astype("float16")




## === cell 15
sns.kdeplot(train.distance, shade=True)




## === cell 16
train.key = pd.to_datetime(train.key).values.astype(np.int64)
test.key = pd.to_datetime(test.key).values.astype(np.int64)




## === cell 17
train_features = train.drop(["key", "fare_amount"], axis=1)
test_features = test.drop("key", axis=1)




## === cell 18
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(train_features)
test_scaled = scaler.transform(test_features)




## === cell 19
y = train["fare_amount"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)




## === cell 20
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K




## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 21
def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))




## === cell 22
def nn(n_feature, k=10):
    model_in = layers.Input(shape=(n_feature,))
    x = layers.Dense(k)(model_in)
    x = layers.LeakyReLU(alpha=0.15)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(k * 4)(x)
    x = layers.LeakyReLU(alpha=0.15)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(k * 16)(x)
    x = layers.LeakyReLU(alpha=0.15)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(k * 16)(x)
    x = layers.LeakyReLU(alpha=0.15)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(k * 4)(x)
    x = layers.LeakyReLU(alpha=0.15)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(k)(x)
    x = layers.LeakyReLU(alpha=0.15)(x)
    x = layers.Dropout(0.2)(x)

    output = layers.Dense(1, activation="linear")(x)

    model = models.Model(inputs=model_in, outputs=output)
    model.compile(
        loss="mse",
        optimizer="adam",
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model




## === cell 23
model = nn(X.shape[1])




## === cell 24
history = model.fit(
    X_train,
    y_train,
    batch_size=1024,
    epochs=20,
    verbose=1,
    validation_data=(X_val, y_val),
)




## === cell 25
plt.plot(history.history["rmse"], label="train")
plt.plot(history.history["val_rmse"], label="val")
plt.title("Model RMSE")
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.legend()
plt.show()




## === cell 26
pres = model.predict(test_scaled)




## === cell 27
test_original = pd.read_csv("../input/test.csv")




## === cell 28
submission = pd.DataFrame(
    {"key": test_original["key"], "fare_amount": pres.reshape(-1)},
    columns=["key", "fare_amount"],
)




## === cell 29
submission.to_csv("submission.csv", index=False)




## === cell 30
print(os.listdir("."))
