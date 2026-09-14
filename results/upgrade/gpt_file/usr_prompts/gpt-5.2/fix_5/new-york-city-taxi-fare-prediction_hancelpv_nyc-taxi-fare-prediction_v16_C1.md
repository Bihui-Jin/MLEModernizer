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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

3.97644

# 6. Current score

27.78105

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 5.65568) has done: 'I fix the Pandas datetime feature extraction by replacing the removed `weekday_name` accessor with the supported `dt.day_name()` and by ensuring `pickup_datetime` is parsed consistently. Then I prevent datetime columns from leaking into the model matrices (which caused the RandomForest and Keras dtype errors) by selecting only the engineered numeric/categorical feature columns and forcing the final design matrices to `float32`. Finally, I fix the Keras import/runtime issue in this environment by using `tf_keras` (already installed) so the neural net trains and produces `submission_nn.csv` with the required `key,fare_amount` columns.'
- What this solution (achieved 177.63536) has done: 'I fix the runtime crash in the neural-network part by avoiding the `tf_keras` import path that triggers the protobuf `MessageFactory.GetPrototype` error in this Kaggle environment, and instead use `tensorflow.keras` (which is compatible with TF’s bundled protobuf). I keep the same model architecture, loss, optimizer, and training call, only changing the import to unblock execution. To move the RMSE down toward your target with minimal logic change, I also add a small, standard training-data filter to remove extreme/out-of-NYC coordinate outliers (a common source of large errors in this competition) while keeping your existing feature engineering and model choices intact. The script still write valid `submission_rf.csv` and `submission_nn.csv` with the required `key,fare_amount` columns.'
- What this solution (achieved 79.80632) has done: 'I fix the TensorFlow/Keras import crash caused by an incompatible protobuf runtime (the `MessageFactory.GetPrototype` error) by switching the neural-network section to use the installed `tf_keras` package instead of `tensorflow.keras`, keeping the exact same model architecture, loss, optimizer, and fit/predict flow. I also make the train/test read paths robust to Kaggle’s dataset directory structure without changing filenames, so the notebook runs end-to-end regardless of whether files live in `../input/` or `../input/new-york-city-taxi-fare-prediction/`. Finally, I ensure the NN training data contains no NaNs after datetime parsing (dropping only rows with invalid `pickup_datetime`), which prevents silent NaNs from poisoning training and should move RMSE down toward your target without altering core logic. The script still write valid `submission_rf.csv` and `submission_nn.csv` with columns `key,fare_amount`.'
- What this solution (achieved 27.78105) has done: 'I fix the crash in the neural-network section by avoiding the `tf_keras` import path that triggers the protobuf `MessageFactory.GetPrototype` error in this environment, and instead import Keras from TensorFlow (`tensorflow.keras`), keeping the same model architecture, loss, optimizer, and fit/predict flow. I also ensure `pickup_datetime` is parsed with `utc=True` so time-feature extraction is consistent and avoids timezone-related NaNs. Finally, I keep the existing feature engineering and filtering intact and ensure the submission files are written with the required `key,fare_amount` columns and `.csv` suffix.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing
import math
import os

INPUT_ROOT = "../input"
CANDIDATE_DIRS = [
    INPUT_ROOT,
    os.path.join(INPUT_ROOT, "new-york-city-taxi-fare-prediction"),
]


def _find_input_file(fname: str) -> str:
    for d in CANDIDATE_DIRS:
        p = os.path.join(d, fname)
        if os.path.exists(p):
            return p
    return os.path.join(INPUT_ROOT, fname)


print("Listing ../input:", os.listdir(INPUT_ROOT))

train_path = _find_input_file("train.csv")
test_path = _find_input_file("test.csv")
sample_path = _find_input_file("sample_submission.csv")

print("Using paths:")
print(" train:", train_path)
print(" test :", test_path)
print(" samp :", sample_path)



## === cell 1
types = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}

cols = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]



## === cell 2
train = pd.read_csv(train_path, nrows=1000000, usecols=cols, dtype=types)
test = pd.read_csv(test_path)
samp = pd.read_csv(sample_path)



## === cell 3
train.dropna(how="any", axis="rows", inplace=True)
train = train[train.fare_amount > 0]
train = train[train["passenger_count"] <= 6]



## === cell 4
latitude_mask_pickup = (train.pickup_latitude > -90) & (train.pickup_latitude < 90)
train = train[latitude_mask_pickup]

latitude_mask_dropoff = (train.dropoff_latitude > -90) & (train.dropoff_latitude < 90)
train = train[latitude_mask_dropoff]



## === cell 5
longitude_mask_pickup = (train.pickup_longitude > -180) & (train.pickup_longitude < 180)
train = train[longitude_mask_pickup]

longitude_mask_dropoff = (train.dropoff_longitude > -180) & (
    train.dropoff_longitude < 180
)
train = train[longitude_mask_dropoff]



## === cell 6
nyc_geo = (
    train["pickup_longitude"].between(-74.5, -72.8)
    & train["dropoff_longitude"].between(-74.5, -72.8)
    & train["pickup_latitude"].between(40.0, 41.8)
    & train["dropoff_latitude"].between(40.0, 41.8)
)
train = train[nyc_geo].copy()



## === cell 7
all_data = pd.concat((train, test), ignore_index=True)

all_data.drop(["fare_amount"], axis=1, inplace=True)
y = train.fare_amount.values
n_train = len(train)
n_test = len(test)
test_id = test.key




## === cell 8
def week_num(day):
    """
    given the day of the month, return the week number of the month
    """
    if day <= 7:
        return "first"
    if (day > 7) and (day <= 14):
        return "second"
    if (day > 14) and (day <= 21):
        return "third"
    if (day > 21) and (day <= 28):
        return "fourth"
    return "fifth"




## === cell 9
def add_time_features(data):
    data = data.copy()
    data["pickup_datetime"] = pd.to_datetime(
        data["pickup_datetime"], errors="coerce", utc=True
    )

    data["hour"] = data["pickup_datetime"].dt.hour
    data["day_of_week"] = data["pickup_datetime"].dt.day_name()
    data["day_of_month"] = data["pickup_datetime"].dt.day
    data["week_of_month"] = data["day_of_month"].map(week_num)
    data["month"] = data["pickup_datetime"].dt.month
    data["year"] = data["pickup_datetime"].dt.year

    data["hour"] = data["hour"].astype("Int64").astype(str)
    data["month"] = data["month"].astype("Int64").astype(str)
    data["year"] = data["year"].astype("Int64").astype(str)

    data.drop(["day_of_month"], axis=1, inplace=True)
    return data




## === cell 10
def add_geo_features(data):
    data = data.copy()
    data["abs_diff_longitude"] = (data.dropoff_longitude - data.pickup_longitude).abs()
    data["abs_diff_latitude"] = (data.dropoff_latitude - data.pickup_latitude).abs()

    data["manhattan_distance"] = data["abs_diff_longitude"] + data["abs_diff_latitude"]

    data["squared_long"] = np.power(data["abs_diff_longitude"], 2)
    data["squared_lat"] = np.power(data["abs_diff_latitude"], 2)

    data["euclid_distance"] = np.sqrt(data["squared_long"] + data["squared_lat"])
    return data




## === cell 11
all_data = add_time_features(all_data)



## === cell 12
all_data = add_geo_features(all_data)



## === cell 13
features = [
    "passenger_count",
    "hour",
    "day_of_week",
    "week_of_month",
    "month",
    "year",
    "abs_diff_longitude",
    "abs_diff_latitude",
    "manhattan_distance",
    "euclid_distance",
]

all_data = all_data[features]
all_data = pd.get_dummies(all_data)
all_data = all_data.astype(np.float32)



## === cell 14
x = all_data.iloc[:n_train].copy()
x_test = all_data.iloc[n_train:].copy()

train_nan_mask = ~np.isfinite(x.to_numpy(dtype=np.float32)).all(axis=1)
if train_nan_mask.any():
    x = x.loc[~train_nan_mask].copy()
    y = y[~train_nan_mask]
    n_train = len(x)
    print(
        f"Dropped {train_nan_mask.sum()} training rows with NaN/inf engineered features."
    )



## === cell 15
from sklearn.ensemble import RandomForestRegressor

model_1 = RandomForestRegressor(random_state=42, n_jobs=-1)



## === cell 16
model_1.fit(x, y)

model_1_pred = model_1.predict(x_test)
sub_1 = pd.DataFrame({"key": test_id, "fare_amount": model_1_pred})
sub_1.to_csv("submission_rf.csv", index=False)



## === cell 17
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 18
num_features = x.shape[1]



## === cell 19
model = Sequential()
model.add(Dense(30, input_dim=num_features, activation="relu"))
model.add(Dense(15, activation="relu"))
model.add(Dense(7, activation="relu"))
model.add(Dense(3, activation="relu"))
model.add(Dense(1, activation="linear"))



## === cell 20
model.summary()



## === cell 21
model.compile(loss="mean_squared_error", optimizer="adam")



## === cell 22
model.fit(x.to_numpy(dtype=np.float32), y, verbose=1)



## === cell 23
test_pred = model.predict(x_test.to_numpy(dtype=np.float32), verbose=0).reshape(-1)

sub = pd.DataFrame({"key": test_id, "fare_amount": test_pred})
sub.to_csv("submission_nn.csv", index=False)

print("Wrote: submission_rf.csv and submission_nn.csv")
print(sub.head())
