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

9.42792

# 6. Current score

954.75865

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 951.13814) has done: 'The fix removes the unavailable TensorFlow Estimator API and replaces it with a lightweight scikit‑learn Ridge regression model while keeping the original feature engineering and data‑cleaning steps. This allows the notebook to run end‑to‑end, creates a valid `submission.csv`, and the simpler linear model is expected to reach an RMSE close to the target without altering the core preprocessing logic.'
- What this solution (achieved 951.14127) has done: 'I add a standard‑scaler to the numeric features so the Ridge model sees data on a comparable scale, which typically lowers the RMSE dramatically without changing the overall model type or training logic. This small preprocessing step is expected to move the validation score much closer to the target (≈9.4) while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 954.75865) has done: 'The current script points to non‑existent `train.csv`/`test.csv` files, causing the model to train on an empty or incorrect dataset and yielding a huge RMSE (~951). The fix updates the paths to the actual data files (`labels.csv` for training and `test.csv` for testing) while keeping all preprocessing, modeling, and submission logic unchanged, so the validation RMSE should now be close to the target ≈ 9.4.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline




## === cell 1
def add_engineered(features):
    lat1 = features["pickup_latitude"]
    lat2 = features["dropoff_latitude"]
    lon1 = features["pickup_longitude"]
    lon2 = features["dropoff_longitude"]
    latdiff = lat1 - lat2
    londiff = lon1 - lon2
    euclidean = np.sqrt(latdiff**2 + londiff**2)

    features["latdiff"] = latdiff
    features["londiff"] = londiff
    features["euclidean"] = euclidean
    return features


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


def output_submission(df, predictions, id_column, prediction_column, file_name):
    df[prediction_column] = predictions
    df[[id_column, prediction_column]].to_csv(file_name, index=False)
    print("Submission written to", file_name)




## === cell 2
TRAIN_PATH = "../input/new-york-city-taxi-fare-prediction/labels.csv"
TEST_PATH = "../input/new-york-city-taxi-fare-prediction/test.csv"
SUBMISSION_NAME = "submission.csv"




## === cell 3
train = pd.read_csv(TRAIN_PATH, nrows=4_000_000)
test = pd.read_csv(TEST_PATH)




## === cell 4
train = clean(train)
train = process(train)
test = process(test)

train = add_engineered(train)
test = add_engineered(test)

train["fare_amount"] = train["fare_amount"].astype("float64")




## === cell 5
NUMERIC_FEATURES = [
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

X = train[NUMERIC_FEATURES]
y = train["fare_amount"]




## === cell 6
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)




## === cell 7
model = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=1))
model.fit(X_train, y_train)

val_pred = model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f"Validation RMSE: {val_rmse:.4f}")




## === cell 8
test_features = test[NUMERIC_FEATURES]
test_pred = model.predict(test_features)




## === cell 9
output_submission(
    df=test,
    predictions=test_pred,
    id_column="key",
    prediction_column="fare_amount",
    file_name=SUBMISSION_NAME,
)
