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

5.45772

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 10.0291) has done: 'I fixed the TensorFlow feature‑column usage, removed the problematic Estimator input functions, and ensured the list of feature names is available for both training and test inference. This resolves the import and attribute errors, lets the Keras model train on the engineered features, and correctly writes a `submission.csv` file with the required columns.'
- What this solution (achieved 50.02613) has done: 'Implemented a protobuf compatibility fix and replaced TensorFlow feature‑column objects with lightweight placeholders that expose only the required `.key` attribute. This avoids the protobuf `MessageFactory` error while preserving the original model architecture, training loop, and submission logic. No changes were made to the core learning algorithm, ensuring the existing RMSE remains essentially unchanged (still better than the target).'
- What this solution (achieved 5.45772) has done: 'The fix removes the TensorFlow dependency that crashes due to protobuf incompatibility and replaces it with a Scikit‑Learn HistGradientBoostingRegressor, which works with the engineered features.  The model is trained on the same split and the prediction → submission steps stay unchanged, so the script now runs end‑to‑end and produces a valid `submission.csv`.  This change also improves the RMSE, moving the score toward the target.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor


class SimpleColumn:
    def __init__(self, key):
        self.key = key


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


def build_estimator(nbuckets, hidden_units, input_columns):
    feature_names = [col.key for col in input_columns]
    model = HistGradientBoostingRegressor(
        max_iter=200, max_depth=None, learning_rate=0.1, random_state=42
    )
    return model, feature_names


def output_submission(df, prediction_df, id_column, prediction_column, file_name):
    df[prediction_column] = prediction_df["predictions"]
    df[[id_column, prediction_column]].to_csv(file_name, index=False)
    print("Output complete")




## === cell 1
TRAIN_PATH = "../input/train.csv"
TEST_PATH = "../input/test.csv"
SUBMISSION_NAME = "submission.csv"

if not os.path.exists(TRAIN_PATH):
    TRAIN_PATH = "data/train.csv"
if not os.path.exists(TEST_PATH):
    TEST_PATH = "data/test.csv"


## === cell 2
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

INPUT_COLUMNS = [
    SimpleColumn("pickup_longitude"),
    SimpleColumn("pickup_latitude"),
    SimpleColumn("dropoff_longitude"),
    SimpleColumn("dropoff_latitude"),
    SimpleColumn("passenger_count"),
    SimpleColumn("year"),
    SimpleColumn("month"),
    SimpleColumn("day"),
    SimpleColumn("hour"),
    SimpleColumn("latdiff"),
    SimpleColumn("londiff"),
    SimpleColumn("euclidean"),
]


## === cell 3
train = pd.read_csv(TRAIN_PATH, nrows=4_000_000)
test = pd.read_csv(TEST_PATH)




## === cell 4
def clean(df):
    df = df[(-76 <= df["pickup_longitude"]) & (df["pickup_longitude"] <= -72)]
    df = df[(-76 <= df["dropoff_longitude"]) & (df["dropoff_longitude"] <= -72)]
    df = df[(38 <= df["pickup_latitude"]) & (df["pickup_latitude"] <= 42)]
    df = df[(38 <= df["dropoff_latitude"]) & (df["dropoff_latitude"] <= 42)]
    df = df[(1 <= df["passenger_count"]) & (df["passenger_count"] <= 6)]
    df = df[df["fare_amount"] > 0]
    return df


def process(df):
    df["year"] = df["pickup_datetime"].apply(lambda x: int(x[:4]))
    df["month"] = df["pickup_datetime"].apply(lambda x: int(x[5:7]))
    df["day"] = df["pickup_datetime"].apply(lambda x: int(x[8:10]))
    df["hour"] = df["pickup_datetime"].apply(lambda x: int(x[11:13]))
    return df




## === cell 5
train = clean(train)


## === cell 6
train = process(train)
train[["fare_amount"]] = train[["fare_amount"]].astype("float64")
test = process(test)


## === cell 7
add_engineered(train)
add_engineered(test)


## === cell 8
train_df, validation_df = train_test_split(train, test_size=0.2, random_state=1)

model, feature_names = build_estimator(
    nbuckets=16, hidden_units=[64, 64, 64, 8], input_columns=INPUT_COLUMNS
)

X_train = train_df[feature_names].values
y_train = train_df[LABEL_COLUMN].values
X_val = validation_df[feature_names].values
y_val = validation_df[LABEL_COLUMN].values

model.fit(X_train, y_train)

val_pred = model.predict(X_val)
val_rmse = np.sqrt(((val_pred - y_val) ** 2).mean())
print(f"Validation RMSE: {val_rmse:.4f}")


## === cell 9
X_test = test[feature_names].values
prediction = model.predict(X_test)
prediction_df = pd.DataFrame({"predictions": prediction})


## === cell 10
output_submission(test, prediction_df, "key", "fare_amount", SUBMISSION_NAME)
