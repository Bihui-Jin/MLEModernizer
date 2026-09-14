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

19.42538

# 6. Current score

8.27543

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 10.19109) has done: 'I fix the Keras import/backend incompatibility causing the `MessageFactory` and missing `backend.sqrt` errors by switching to `tf_keras` and using TensorFlow ops in the custom RMSE metric (score-neutral, correctness fix). I also fix a subtle but important preprocessing bug where you fit a separate `MinMaxScaler` on the test set; instead, fit on train and transform both train and test (this should modestly improve generalization and move your RMSE up toward the higher target, without changing the model). Finally, I keep the model architecture/training loop the same and ensure the submission CSV is created with the required `key,fare_amount` columns aligned to the raw test keys.'
- What this solution (achieved 6.32108) has done: 'You’re hitting the `MessageFactory.GetPrototype` protobuf/Keras compatibility crash at the point where TensorFlow is imported, so the pipeline never reaches training or submission writing. I fix this by forcing the runtime to use the TensorFlow-bundled Keras (`tf.keras`) consistently (and by setting the protobuf implementation to the pure-Python fallback before importing TF), which is a correctness/stability fix and should be score-neutral. I also keep your existing model architecture, feature engineering, scaling, and training loop unchanged, only adjusting the import order and Keras imports so the notebook runs end-to-end. Finally, I keep the submission formatting exactly `key,fare_amount` aligned to the raw test keys and ensure `submission.csv` is written.'
- What this solution (achieved 12.71248) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf runtime *before* any TensorFlow/Keras-related imports and by using the TensorFlow-bundled Keras (`tf.keras`) consistently. I keep your model, features, scaling, and training loop unchanged, only adjusting import order to ensure execution reaches training and submission writing. I also make the `key` parsing more robust by using `errors="coerce"` and dropping resulting NaNs, preventing failures from malformed keys without changing the learning logic. Finally, the script always write a valid `submission.csv` with the required `key,fare_amount` columns aligned to the original raw test keys.'
- What this solution (achieved 7.35478) has done: 'I fix the TensorFlow/Keras import crash by forcing the pure-Python protobuf implementation before any TF import and by using the TensorFlow-bundled Keras API consistently (`tf.keras`) so the notebook runs end-to-end. I also fix a logic bug where the test set scaling currently includes the `key` column (but the train set does not), which can silently break feature alignment and hurt RMSE; the fix is to keep `key` separate and scale only the real feature columns for both train and test. Finally, I preserve your model architecture, loss, optimizer, epochs, and feature engineering (including the `distance` feature) and ensure the submission is written as `submission.csv` with `key,fare_amount` aligned to the original raw test keys.'
- What this solution (achieved 8.27543) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by enforcing the pure-Python protobuf runtime before any TF import and by using the TensorFlow-bundled Keras API consistently via `tf.keras` (this is a stability/correctness fix so training and submission generation can run). I also ensure we read data from the correct Kaggle path (`/kaggle/input/...`) while keeping your dataset size, features, scaling, model architecture, and training loop unchanged. Finally, I keep `key` handling consistent and guarantee the submission is written as `submission.csv` with exactly `key,fare_amount`, aligned to the raw test file order.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

DATA_DIR = "/kaggle/input/new-york-city-taxi-fare-prediction"
print("DATA_DIR exists:", os.path.isdir(DATA_DIR))
print(os.listdir("/kaggle/input")[:20])



## === cell 1
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")



## === cell 2
train = pd.read_csv(f"{DATA_DIR}/train.csv", nrows=10_000_000)



## === cell 3
train.head()



## === cell 4
test = pd.read_csv(f"{DATA_DIR}/test.csv")



## === cell 5
test.head()



## === cell 6
train.dtypes




## === cell 7
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points on the earth (in km),
    specified in decimal degrees. Vectorized for numpy/pandas.
    """
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371
    return c * r




## === cell 8
def add_travel_distance_vector_features(df):
    df["distance"] = haversine(
        df.dropoff_longitude,
        df.dropoff_latitude,
        df.pickup_longitude,
        df.pickup_latitude,
    )


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
train_key_dt = pd.to_datetime(train.key, errors="coerce")
test_key_dt = pd.to_datetime(test.key, errors="coerce")

train = train.loc[train_key_dt.notna()].copy()
test = test.loc[test_key_dt.notna()].copy()

train.key = train_key_dt.loc[train_key_dt.notna()].values.astype(np.int64)
test.key = test_key_dt.loc[test_key_dt.notna()].values.astype(np.int64)



## === cell 17
train.head()



## === cell 18
train.distance.describe().astype("float16")



## === cell 19
y = train.pop("fare_amount")
X = train



## === cell 20
from sklearn.preprocessing import MinMaxScaler



## === cell 21
feature_cols = ["passenger_count", "distance"]
X_features = X[feature_cols]
test_features = test[feature_cols]

scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X_features)
test_scaled = scaler.transform(test_features)



## === cell 22
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.33, random_state=42
)



## === cell 23
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Model


def rmse(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))




## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 24
def nn(n_feature, k=10):
    model_in = Input(shape=(n_feature,))
    model = Dense(k, activation="relu")(model_in)
    model = Dropout(0.2)(model)
    model = Dense(k * 4, activation="relu")(model)
    model = Dropout(0.2)(model)
    model = Dense(k, activation="relu")(model)
    model = Dropout(0.2)(model)
    model = Dense(1, activation="linear")(model)

    model = Model(inputs=model_in, outputs=model)
    model.compile(loss="mse", optimizer="adam", metrics=[rmse])
    return model




## === cell 25
model = nn(X_train.shape[1])



## === cell 26
history = model.fit(X_train, y_train, batch_size=1000, epochs=10, verbose=1)



## === cell 27
plt.plot(history.history.get("rmse", []))
plt.title("model rmse")
plt.ylabel("rmse")
plt.xlabel("epoch")
plt.legend(["train"], loc="upper left")
plt.show()



## === cell 28
pres = model.predict(test_scaled, verbose=0)



## === cell 29
test_raw = pd.read_csv(f"{DATA_DIR}/test.csv")



## === cell 30
submission = pd.DataFrame(
    {"key": test_raw["key"].values, "fare_amount": pres.reshape(-1)},
    columns=["key", "fare_amount"],
)



## === cell 31
submission.to_csv("submission.csv", index=False)

print(os.listdir("."))
print(submission.head())
print("submission.csv rows:", len(submission))
