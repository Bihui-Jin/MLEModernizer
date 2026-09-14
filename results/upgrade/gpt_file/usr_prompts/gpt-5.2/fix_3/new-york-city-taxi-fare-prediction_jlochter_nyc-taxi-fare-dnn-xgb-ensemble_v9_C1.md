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

4.20678

# 6. Current score

7.81816

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 5.90741) has done: 'I fix the import/runtime failure caused by the `keras` package conflict by switching to `tf_keras`, which is compatible in this environment and preserves the exact same network architecture and training loop. I also fix the pandas `.corr()` crash by limiting the correlation computation to numeric columns only (score-neutral and only affects plotting). Then I correct the XGBoost pipeline bugs: use the same fitted scaler for test data (was incorrectly refit), update deprecated XGBoost params, and use `best_iteration` safely instead of the missing `best_ntree_limit`. Finally, I fix the accidental use of the DNN predictions in the “xgb” submission and ensure we always write a valid `submission_ensemble.csv`.'
- What this solution (achieved 7.81816) has done: 'I fix the immediate runtime crash coming from importing `tf_keras` (it’s a known protobuf/TensorFlow import mismatch symptom) by switching the DNN imports to `tensorflow.keras`, keeping the exact same Sequential/Dense/Dropout architecture and training loop. I also make the feature engineering safe by not dropping `pickup_datetime` (since it’s already non-numeric here) and explicitly converting the model inputs to `float32` to avoid mixed dtype issues that can silently hurt training. To move RMSE toward your target (lower is better) with minimal semantic change, I add the standard NYC taxi baseline feature `passenger_count`- and `datetime`-derived time features (hour, dayofweek, month) using the existing `pickup_datetime` column, which is a small, common improvement without changing the model family. Submission writing remains identical, and all three submission CSVs are still produced.'

# 9. Code solution

## === cell 0
import os
import math
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

np.random.seed(70)
tf.random.set_seed(70)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
TRAIN_PATH_1 = "../input/train.csv"
TEST_PATH_1 = "../input/test.csv"
TRAIN_PATH_2 = "../input/new-york-city-taxi-fare-prediction/train.csv"
TEST_PATH_2 = "../input/new-york-city-taxi-fare-prediction/test.csv"

train_path = TRAIN_PATH_1 if os.path.exists(TRAIN_PATH_1) else TRAIN_PATH_2
test_path = TEST_PATH_1 if os.path.exists(TEST_PATH_1) else TEST_PATH_2

train_df = pd.read_csv(train_path, nrows=1000000)
test_df = pd.read_csv(test_path)

print("Loaded:", train_path, test_path)
print("Train shape:", train_df.shape, "Test shape:", test_df.shape)



## === cell 2
datasets = [train_df, test_df]
for name, df in zip(["train", "test"], datasets):
    missing_values = (
        df.isnull().sum().to_frame("missing").sort_values("missing", ascending=False)
    )
    print(f"\nMissing values ({name}) top 10:")
    print(missing_values.head(10))



## === cell 3
print("Train before cleaning:")
print(train_df.describe(include="all"))

train_df = train_df.dropna(how="any", axis="rows")
train_df = train_df[
    (train_df.pickup_longitude > -75.0) & (train_df.pickup_longitude < -73.0)
]
train_df = train_df[
    (train_df.pickup_latitude > 40.0) & (train_df.pickup_latitude < 42.0)
]
train_df = train_df[
    (train_df.dropoff_longitude > -75.0) & (train_df.dropoff_longitude < -73.0)
]
train_df = train_df[
    (train_df.dropoff_latitude > 40.0) & (train_df.dropoff_latitude < 42.0)
]
train_df = train_df[
    (train_df.passenger_count > 0.0) & (train_df.passenger_count <= 6.0)
]

print("\nTrain after cleaning:")
print(train_df.describe(include="all"))

print("\nTest for comparison:")
print(test_df.describe(include="all"))




## === cell 4
def add_time_features(df):
    df = df.copy()
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)
    df["pickup_hour"] = dt.dt.hour.astype("float32")
    df["pickup_dayofweek"] = dt.dt.dayofweek.astype("float32")
    df["pickup_month"] = dt.dt.month.astype("float32")
    return df


def calc_haversine(df):
    df = df.copy()
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs()

    df["dlat"] = np.radians(df.dropoff_latitude - df.pickup_latitude)
    df["dlon"] = np.radians(df.dropoff_longitude - df.pickup_longitude)
    df["haversine_a"] = np.sin(df.dlat / 2) * np.sin(df.dlat / 2) + np.cos(
        np.radians(df.pickup_latitude)
    ) * np.cos(np.radians(df.dropoff_latitude)) * np.sin(df.dlon / 2) * np.sin(
        df.dlon / 2
    )
    df["haversine"] = (
        6371 * 2 * np.arctan2(np.sqrt(df.haversine_a), np.sqrt(1 - df.haversine_a))
    )
    return df


train_df = add_time_features(train_df)
test_df = add_time_features(test_df)

train_df = calc_haversine(train_df)
test_df = calc_haversine(test_df)



## === cell 5
numeric_cols = train_df.select_dtypes(include=[np.number]).columns
corr = train_df[numeric_cols].corr()

f, ax = plt.subplots(figsize=(10, 10))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(
    corr,
    cmap=cmap,
    vmax=1.0,
    square=True,
    linewidths=0.3,
    cbar_kws={"shrink": 0.5},
    ax=ax,
)
plt.show()



## === cell 6
train_y = np.array(train_df["fare_amount"], dtype=np.float32)
train_X = train_df.drop(columns=["fare_amount", "key", "pickup_datetime"])

print("Shape for X:", train_X.shape)
print("Shape for Y:", train_y.shape)

test_X = test_df.drop(columns=["key", "pickup_datetime"])
print("Shape for test X:", test_X.shape)

train_X = train_X.astype(np.float32)
test_X = test_X.astype(np.float32)




## === cell 7
def run_model(X, Y, dnn_layers_size, dropout_value, batch_size, epochs):
    input_size = X.shape[1]
    model = Sequential()

    for l in dnn_layers_size:
        model.add(
            Dense(
                l, input_dim=input_size, kernel_initializer="normal", activation="selu"
            )
        )
        model.add(Dropout(dropout_value))
        input_size = l

    model.add(Dense(1, kernel_initializer="normal"))
    model.compile(loss="mean_squared_error", optimizer="adam")

    train_history = model.fit(
        X,
        Y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        shuffle=True,
        verbose=1,
    )
    return train_history, model


def build_layers(layers, n_features):
    if len(layers) == 0:
        n_features = int(n_features * 2.5)
    else:
        n_features = int(math.sqrt(n_features))

    if n_features < 3:
        return layers
    layers.append(n_features)
    return build_layers(layers, n_features)


def plot_build(train_history):
    plt.figure(0)
    axes = plt.gca()
    axes.set_ylim([0, 90])
    plt.plot(train_history.history["loss"], "g")
    plt.plot(train_history.history["val_loss"], "b")
    plt.rcParams["figure.figsize"] = (8, 6)
    plt.xlabel("Num of Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Validation Loss")
    plt.grid()
    plt.legend(["train", "validation"])
    plt.show()




## === cell 8
layers = build_layers([], train_X.shape[1])
print("Layers:", layers)
print("-" * 15)

train_history, model = run_model(
    train_X.values, train_y, layers, 0.2, batch_size=32, epochs=10
)



## === cell 9
plot_build(train_history)



## === cell 10
pred_y = model.predict(test_X.values, verbose=0).reshape(-1)
test_df["pred"] = pred_y

submission = pd.DataFrame(
    {"key": test_df.key, "fare_amount": test_df.pred}, columns=["key", "fare_amount"]
)
submission.to_csv("submission_dnn.csv", index=False)

print("Wrote submission_dnn.csv")
print("Working dir files:", os.listdir("."))



## === cell 11
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb

stdscaler = StandardScaler()
xgb_train_X = stdscaler.fit_transform(train_X.values)
xgb_test_X = stdscaler.transform(test_X.values)

x_train, x_val, y_train, y_val = train_test_split(
    xgb_train_X, train_y, random_state=70, test_size=0.2
)


def XGBmodel(x_train, x_val, y_train, y_val):
    matrix_train = xgb.DMatrix(x_train, label=y_train)
    matrix_val = xgb.DMatrix(x_val, label=y_val)

    params = {"objective": "reg:squarederror", "eval_metric": "rmse"}

    model = xgb.train(
        params=params,
        dtrain=matrix_train,
        num_boost_round=100,
        early_stopping_rounds=10,
        evals=[(matrix_val, "val")],
        verbose_eval=False,
    )
    return model


xgb_model = XGBmodel(x_train, x_val, y_train, y_val)

test_dmat = xgb.DMatrix(xgb_test_X)
if hasattr(xgb_model, "best_iteration") and xgb_model.best_iteration is not None:
    xgb_pred = xgb_model.predict(
        test_dmat, iteration_range=(0, xgb_model.best_iteration + 1)
    )
else:
    xgb_pred = xgb_model.predict(test_dmat)



## === cell 12
test_df["pred_xgb"] = xgb_pred

submission = pd.DataFrame(
    {"key": test_df.key, "fare_amount": test_df.pred_xgb},
    columns=["key", "fare_amount"],
)
submission.to_csv("submission_xgb.csv", index=False)

print("Wrote submission_xgb.csv")
print("Working dir files:", os.listdir("."))



## === cell 13
test_df["ensemble"] = (test_df.pred + test_df.pred_xgb) / 2.0

submission = pd.DataFrame(
    {"key": test_df.key, "fare_amount": test_df.ensemble},
    columns=["key", "fare_amount"],
)
submission.to_csv("submission_ensemble.csv", index=False)

print("Wrote submission_ensemble.csv")
print("Working dir files:", os.listdir("."))
