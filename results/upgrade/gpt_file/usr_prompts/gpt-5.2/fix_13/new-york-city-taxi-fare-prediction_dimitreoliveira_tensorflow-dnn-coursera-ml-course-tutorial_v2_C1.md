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

11.1707

# 6. Current score

7.54692

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 7.54692) has done: 'I fix the protobuf / TF-DF import crash by setting the correct `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` environment variable before importing TensorFlow/TF-DF. Then I fix the TF-DF dataset creation error by explicitly telling `pd_dataframe_to_tf_dataset` that this is a regression task (otherwise it infers classification and complains about too many “classes”). Finally, I keep the rest of your feature engineering, cleaning, model, and training flow the same, and ensure a valid `submission.csv` with `key,fare_amount` is always written.'
- What this solution (achieved 7.54692) has done: 'I fix the TensorFlow import crash by changing the protobuf implementation environment variable to the pure-Python backend before importing TensorFlow/TF-DF, which avoids the missing `_message` symbol in this Kaggle image. Because cell 0 currently fails, downstream imports (like `train_test_split`) never happen, so I also ensure those imports occur after the environment fix and keep them available for later cells. Then I keep your exact data cleaning/feature engineering/model/training flow the same, only making minimal robustness tweaks (parse datetime once via `pd.to_datetime` to avoid slow `.apply`) while preserving semantics. Finally, I guarantee a valid `submission.csv` with columns `key,fare_amount` is always written.'
- What this solution (achieved 7.54692) has done: 'I fix the TensorFlow / TF-DF import crash by pinning protobuf to the pure-Python implementation and (critically) disabling C++ fast-paths via the `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2` setting before importing TensorFlow—this avoids the `MessageFactory.GetPrototype` error in this environment. I also make the TF-DF dataset creation robust by ensuring the label is float and the datetime-derived integer features are plain numeric dtypes (not pandas nullable Int64), which prevents silent type issues. Since your current score (7.54692) is already better than the target (11.1707) for a lower-is-better metric, I not make any modeling/feature changes that would intentionally move score; the edits are correctness/stability only. The script run end-to-end and always write a valid `submission.csv` with `key,fare_amount`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import tensorflow_decision_forests as tfdf

tf.get_logger().setLevel("INFO")
np.random.seed(1)
tf.random.set_seed(1)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def add_engineered(features):
    lat1 = features["pickup_latitude"]
    lat2 = features["dropoff_latitude"]
    lon1 = features["pickup_longitude"]
    lon2 = features["dropoff_longitude"]
    latdiff = lat1 - lat2
    londiff = lon1 - lon2
    euclidean = (latdiff**2 + londiff**2) ** 0.5

    features["latdiff"] = latdiff
    features["londiff"] = londiff
    features["euclidean"] = euclidean

    return features


def output_submission(df, prediction_df, id_column, prediction_column, file_name):
    df = df.copy()
    df[prediction_column] = prediction_df[prediction_column].astype(float)
    df[[id_column, prediction_column]].to_csv(file_name, index=False)
    print("Output complete:", file_name)




## === cell 2
TRAIN_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
TEST_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"
if not os.path.exists(TRAIN_PATH):
    TRAIN_PATH = "/kaggle/input/train.csv"
if not os.path.exists(TEST_PATH):
    TEST_PATH = "/kaggle/input/test.csv"

SUBMISSION_NAME = "submission.csv"



## === cell 3
CSV_COLUMNS = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "year",
    "month",
    "day",
    "hour",
]
LABEL_COLUMN = "fare_amount"

FEATURE_NAMES = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "year",
    "month",
    "day",
    "hour",
    "latdiff",
    "londiff",
    "euclidean",
]



## === cell 4
train = pd.read_csv(TRAIN_PATH, nrows=4000000)
test = pd.read_csv(TEST_PATH)




## === cell 5
def clean(df):
    df = df[(-76 <= df["pickup_longitude"]) & (df["pickup_longitude"] <= -72)]
    df = df[(-76 <= df["dropoff_longitude"]) & (df["dropoff_longitude"] <= -72)]
    df = df[(38 <= df["pickup_latitude"]) & (df["pickup_latitude"] <= 42)]
    df = df[(38 <= df["dropoff_latitude"]) & (df["dropoff_latitude"] <= 42)]
    df = df[(1 <= df["passenger_count"]) & (df["passenger_count"] <= 6)]
    df = df[df["fare_amount"] > 0]
    return df


def process(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=False)

    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["hour"] = dt.dt.hour
    df[["year", "month", "day", "hour"]] = df[["year", "month", "day", "hour"]].fillna(
        0
    )

    df["year"] = df["year"].astype(np.int32)
    df["month"] = df["month"].astype(np.int32)
    df["day"] = df["day"].astype(np.int32)
    df["hour"] = df["hour"].astype(np.int32)

    return df




## === cell 6
train = clean(train)



## === cell 7
train = process(train)
train[["fare_amount"]] = train[["fare_amount"]].astype("float64")
test = process(test)



## === cell 8
train = add_engineered(train)
test = add_engineered(test)

train.head(5)



## === cell 9
train_df, validation_df = train_test_split(train, test_size=0.2, random_state=1)




## === cell 10
def _prep_for_tfdf(df, is_train=True):
    cols = (["fare_amount"] if is_train else []) + FEATURE_NAMES
    out = df[cols].copy()

    float_cols = [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "latdiff",
        "londiff",
        "euclidean",
        "fare_amount",
    ]
    for c in float_cols:
        if c in out.columns:
            out[c] = out[c].astype(np.float32)

    int_cols = ["passenger_count", "year", "month", "day", "hour"]
    for c in int_cols:
        if c in out.columns:
            out[c] = out[c].astype(np.int32)

    return out


train_tfdf = _prep_for_tfdf(train_df, is_train=True)
valid_tfdf = _prep_for_tfdf(validation_df, is_train=True)
test_tfdf = _prep_for_tfdf(test, is_train=False)

train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
    train_tfdf, label="fare_amount", task=tfdf.keras.Task.REGRESSION
)
valid_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
    valid_tfdf, label="fare_amount", task=tfdf.keras.Task.REGRESSION
)

model = tfdf.keras.GradientBoostedTreesModel(task=tfdf.keras.Task.REGRESSION)
model.compile(metrics=["mse"])

model.fit(train_ds, validation_data=valid_ds)

test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
    test_tfdf, task=tfdf.keras.Task.REGRESSION
)
pred = model.predict(test_ds, verbose=0).reshape(-1)

prediction_df = pd.DataFrame({"fare_amount": pred.astype(float)})
output_submission(test, prediction_df, "key", "fare_amount", SUBMISSION_NAME)

sub = pd.read_csv(SUBMISSION_NAME)
print(sub.head())
print("Submission shape:", sub.shape)
print("Columns:", list(sub.columns))
print("Saved to:", os.path.abspath(SUBMISSION_NAME))
