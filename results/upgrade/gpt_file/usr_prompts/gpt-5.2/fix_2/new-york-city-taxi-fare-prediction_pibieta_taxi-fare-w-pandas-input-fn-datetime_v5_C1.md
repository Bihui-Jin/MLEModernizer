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

4.29156

# 6. Current score

558.61119

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 558.61119) has done: 'I fix the environment/runtime errors by removing the broken TensorFlow Estimator/`tf.contrib` usage (not available in TF 2.18 here) and replacing it with an equivalent Keras model trained on the same engineered/tabular features, so the pipeline runs end-to-end. I also fix a logic bug where you computed `add_feats(test)` but then overwrote it by dropping columns from the original `test` (losing engineered features), and I vectorize the distance feature computation so it finishes quickly. Finally, I ensure the script always writes a valid `submission_file.csv` with the required `key,fare_amount` columns.'

# 9. Code solution

## === cell 0
import os
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf

print("../", os.listdir("../")[:20])
print("../input", os.listdir("../input")[:20])
print("tf version: ", tf.__version__)

np.random.seed(1)
tf.random.set_seed(1)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
df = pd.read_csv("../input/train.csv", nrows=100000, parse_dates=["pickup_datetime"])
test = pd.read_csv("../input/test.csv", parse_dates=["pickup_datetime"])




## === cell 2
def haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1.astype(np.float64))
    lon1 = np.radians(lon1.astype(np.float64))
    lat2 = np.radians(lat2.astype(np.float64))
    lon2 = np.radians(lon2.astype(np.float64))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c




## === cell 3
def add_feats(dfin):
    dfout = dfin.copy()
    dfout["distance"] = haversine_km(
        dfout["pickup_latitude"].values,
        dfout["pickup_longitude"].values,
        dfout["dropoff_latitude"].values,
        dfout["dropoff_longitude"].values,
    )
    dfout["hour"] = dfout["pickup_datetime"].dt.hour.astype(np.int16)
    dfout["weekday"] = dfout["pickup_datetime"].dt.weekday.astype(np.int16)
    return dfout




## === cell 4
df = add_feats(df)
test = add_feats(test)



## === cell 5
dfc = df[
    ((df.pickup_longitude >= -75.0) & (df.pickup_longitude <= -72))
    & ((df.pickup_latitude >= 38) & (df.pickup_latitude <= 42))
    & ((df.dropoff_longitude >= -75.0) & (df.dropoff_longitude <= -72))
    & ((df.dropoff_latitude >= 38) & (df.dropoff_latitude <= 42))
    & (df.fare_amount > 2.5)
    & (df.passenger_count > 0)
    & (df.passenger_count < 7)
    & (df.distance > 0.2)
].copy()

print("Filtered rows:", len(dfc), "of", len(df))



## === cell 6
msk = np.random.rand(len(dfc)) < 0.8
traindf = dfc[msk].drop(["key", "pickup_datetime"], axis=1)
evaldf = dfc[~msk].drop(["key", "pickup_datetime"], axis=1)

testdf = test.drop(["key", "pickup_datetime"], axis=1)

print(
    "Train shape:",
    traindf.shape,
    "Eval shape:",
    evaldf.shape,
    "Test shape:",
    testdf.shape,
)



## === cell 7

FEATURES = [c for c in traindf.columns if c != "fare_amount"]
X_train = traindf[FEATURES].astype(np.float32).values
y_train = traindf["fare_amount"].astype(np.float32).values
X_eval = evaldf[FEATURES].astype(np.float32).values
y_eval = evaldf["fare_amount"].astype(np.float32).values
X_test = testdf[FEATURES].astype(np.float32).values

mu = X_train.mean(axis=0)
sigma = X_train.std(axis=0)
sigma[sigma == 0] = 1.0
X_train_s = (X_train - mu) / sigma
X_eval_s = (X_eval - mu) / sigma
X_test_s = (X_test - mu) / sigma



## === cell 8
inputs = tf.keras.Input(shape=(X_train_s.shape[1],), dtype=tf.float32)
x = tf.keras.layers.Dense(128, activation="relu")(inputs)
x = tf.keras.layers.Dense(32, activation="relu")(x)
x = tf.keras.layers.Dense(4, activation="relu")(x)
outputs = tf.keras.layers.Dense(1, activation="linear")(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="mse",
    metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
)

model.summary()



## === cell 9
BATCH_SIZE = 512
EPOCHS = 100

history = model.fit(
    X_train_s,
    y_train,
    validation_data=(X_eval_s, y_eval),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2,
)

eval_metrics = model.evaluate(X_eval_s, y_eval, batch_size=BATCH_SIZE, verbose=0)
print(dict(zip(model.metrics_names, eval_metrics)))



## === cell 10
pred = model.predict(X_test_s, batch_size=1024, verbose=0).reshape(-1)

pred = np.clip(pred, 0.0, None)

submission = pd.DataFrame(
    {"key": test["key"].values, "fare_amount": pred.astype(np.float32)}
)

submission_path = "submission_file.csv"
submission.to_csv(submission_path, index=False)
print("Wrote:", submission_path, "shape:", submission.shape)
print(submission.head())
