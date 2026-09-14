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

5.84951

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 5.84951) has done: 'I fix the import error by using `tensorflow.keras` instead of the standalone Keras package, adjust the correlation computation to ignore non‑numeric columns, correct the XGBoost prediction handling (use `best_iteration` and the proper prediction array), and ensure the DNN and XGBoost predictions are stored correctly before creating the ensemble submission. These minimal changes resolve the runtime errors and should modestly improve the RMSE, moving the score closer to the target.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import math

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

from sklearn.preprocessing import PolynomialFeatures




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_df = pd.read_csv("../input/train.csv", nrows=1000000)
test_df = pd.read_csv("../input/test.csv")




## === cell 2
datasets = [train_df, test_df]

for df in datasets:
    missing_values = df.isnull().sum().to_frame().sort_values(0, ascending=False)
    display(missing_values.head())




## === cell 3
print("Train before cleaning:")
display(train_df.describe())

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

print("Train after cleaning:")
display(train_df.describe())

print("Test for comparison:")
display(test_df.describe())




## === cell 4
def calc_haversine(df):
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

    return df.drop(columns=["pickup_datetime"])


train_df = calc_haversine(train_df)
test_df = calc_haversine(test_df)




## === cell 5
numeric_corr = train_df.select_dtypes(include=np.number).corr()
f, ax = plt.subplots(figsize=(10, 10))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(
    numeric_corr,
    cmap=cmap,
    vmax=1.0,
    square=True,
    linewidths=0.3,
    cbar_kws={"shrink": 0.5},
    ax=ax,
)
plt.show()




## === cell 6
train_y = np.array(train_df["fare_amount"])
train_X = train_df.drop(columns=["fare_amount", "key"])

print("Shape for X:")
print(train_X.shape)
print("Shape for Y:")
print(train_y.shape)

test_X = test_df.drop(columns=["key"])
print("Shape for test X:")
print(test_X.shape)




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
    else:
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
    train_X, train_y, layers, 0.2, batch_size=32, epochs=10
)




## === cell 9
plot_build(train_history)




## === cell 10
pred_y = model.predict(test_X).reshape(-1)
test_df["pred"] = pred_y

submission = pd.DataFrame(
    {"key": test_df["key"], "fare_amount": test_df["pred"]},
    columns=["key", "fare_amount"],
)
submission.to_csv("submission_dnn.csv", index=False)

print(os.listdir("."))




## === cell 11
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb

stdscaler = StandardScaler()
xgb_train_X = stdscaler.fit_transform(train_X)
xgb_test_X = stdscaler.transform(test_X)  # use transform, not fit, on test data

x_train, x_val, y_train, y_val = train_test_split(
    xgb_train_X, train_y, random_state=70, test_size=0.2
)


def XGBmodel(x_train, x_val, y_train, y_val):
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dval = xgb.DMatrix(x_val, label=y_val)
    model = xgb.train(
        params={"objective": "reg:squarederror", "eval_metric": "rmse"},
        dtrain=dtrain,
        num_boost_round=500,
        early_stopping_rounds=20,
        evals=[(dval, "validation")],
        verbose_eval=False,
    )
    return model


xgb_model = XGBmodel(x_train, x_val, y_train, y_val)
xgb_pred = xgb_model.predict(xgb.DMatrix(xgb_test_X))




## === cell 12
test_df["pred_xgb"] = xgb_pred

submission_xgb = pd.DataFrame(
    {"key": test_df["key"], "fare_amount": test_df["pred_xgb"]},
    columns=["key", "fare_amount"],
)
submission_xgb.to_csv("submission_xgb.csv", index=False)

print(os.listdir("."))




## === cell 13
test_df["ensemble"] = (test_df["pred"] + test_df["pred_xgb"]) / 2.0

submission_ens = pd.DataFrame(
    {"key": test_df["key"], "fare_amount": test_df["ensemble"]},
    columns=["key", "fare_amount"],
)
submission_ens.to_csv("submission_ensemble.csv", index=False)

print(os.listdir("."))
