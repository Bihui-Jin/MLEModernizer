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
scipy==1.15.3
sklearn-pandas==2.2.0

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

4.20022

# 6. Current score

5.62072

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.37892) has done: 'I replace the deprecated **Imputer** with **SimpleImputer**, fix the import error that prevented GradientBoostingRegressor from being defined, and adjust the code to use the new imputer. No core modeling logic is changed; the script now run end‑to‑end and correctly create a submission.csv file.'
- What this solution (achieved 5.72373) has done: 'The fix speeds up the pipeline by (1) reading only the necessary columns, (2) pre‑allocating NumPy arrays instead of appending to Python lists, (3) converting feature columns to `float32` to reduce memory traffic, and (4) skipping the non‑essential histogram plot. All model‑related logic remains unchanged, so the predictions stay identical (aside from negligible floating‑point rounding).'
- What this solution (achieved 5.62072) has done: 'The script now limits the training data to 500 000 rows (instead of 2 000 000) and builds the feature matrix using a pre‑allocated NumPy array, which removes the overhead of repeatedly calling `np.column_stack`. These changes keep the exact same preprocessing, feature engineering, and model architecture while cutting the GradientBoosting training time enough to finish well inside the 600‑second limit.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearnex import patch_sklearn  # accelerate sklearn

patch_sklearn()  # apply acceleration

print(os.listdir("../input"))




## === cell 1
def chunck_generator(filename, header=False, chunk_size=2_000_000):
    """
    Generates DataFrame chunks from a CSV file, correctly parsing the
    `pickup_datetime` column as datetime objects.
    Only the needed columns are read to lower I/O and memory use.
    """
    usecols = [
        "key",
        "fare_amount",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]
    for chunk in pd.read_csv(
        filename,
        delimiter=",",
        iterator=True,
        chunksize=chunk_size,
        parse_dates=["pickup_datetime"],
        usecols=usecols,
    ):
        yield chunk




## === cell 2
alpha_ang = 0.506


def distance_travel(df):
    """
    Compute Haversine distance (in miles) between pickup and dropoff points.
    """
    lat1 = np.radians(df.pickup_latitude)
    lon1 = np.radians(df.pickup_longitude)
    lat2 = np.radians(df.dropoff_latitude)
    lon2 = np.radians(df.dropoff_longitude)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    earth_radius_miles = 3959.0
    df["distance_travel"] = earth_radius_miles * c
    return df




## === cell 3
def data_clean(df):
    df = df[df.passenger_count > 0]
    df.fare_amount = df.fare_amount.astype(np.float64)
    df = df[df.fare_amount > 0]
    distance_travel(df)  # distance computed once here
    df = df[df.distance_travel > 0]
    return df




## === cell 4
def remove_outliers(df):
    df = df[df.distance_travel < 30]
    df = df[df.fare_amount < 60]
    return df




## === cell 5
pass




## === cell 6
def data_preprocessing(df):
    df = data_clean(df)
    df = remove_outliers(df)
    df["hour"] = df["pickup_datetime"].dt.hour
    df["weekday"] = df["pickup_datetime"].dt.weekday
    return df




## === cell 7
pass




## === cell 8
MAX_TRAIN_ROWS = 500_000  # decreased from 2_000_000
CHUNK_SIZE = 500_000  # keep chunk size reasonable

rows_collected = 0
chunks = []
for chunk in chunck_generator("../input/train.csv", chunk_size=CHUNK_SIZE):
    processed = data_preprocessing(chunk)
    if not processed.empty:
        chunks.append(processed)
        rows_collected += len(processed)
    if rows_collected >= MAX_TRAIN_ROWS:
        break

train_df = pd.concat(chunks, ignore_index=True)
if len(train_df) > MAX_TRAIN_ROWS:
    train_df = train_df.iloc[:MAX_TRAIN_ROWS]

num_rows = len(train_df)
X_all = np.empty((num_rows, 5), dtype=np.float32)
X_all[:, 0] = train_df.distance_travel.astype(np.float32)
X_all[:, 1] = train_df.passenger_count.astype(np.float32)
X_all[:, 2] = train_df["hour"].astype(np.float32)
X_all[:, 3] = train_df["weekday"].astype(np.float32)
X_all[:, 4] = 1.0  # bias term

y_all = train_df.fare_amount.values.astype(np.float64)

imp = SimpleImputer(missing_values=np.nan, strategy="mean")
X_all = imp.fit_transform(X_all)

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=0.1, random_state=42
)

regr = GradientBoostingRegressor(
    n_estimators=400,
    learning_rate=0.05,
    subsample=0.8,
    max_depth=3,
    random_state=42,
    warm_start=False,
)

regr.fit(X_train, y_train)

val_pred = regr.predict(X_val)
rmse = mean_squared_error(y_val, val_pred, squared=False)
print(f"Validation RMSE: {rmse:.5f}")




## === cell 9
test_df = pd.read_csv("../input/test.csv", parse_dates=["pickup_datetime"])
distance_travel(test_df)
test_df["hour"] = test_df["pickup_datetime"].dt.hour
test_df["weekday"] = test_df["pickup_datetime"].dt.weekday




## === cell 10
test_X = np.empty((len(test_df), 5), dtype=np.float32)
test_X[:, 0] = test_df.distance_travel.astype(np.float32)
test_X[:, 1] = test_df.passenger_count.astype(np.float32)
test_X[:, 2] = test_df.hour.astype(np.float32)
test_X[:, 3] = test_df.weekday.astype(np.float32)
test_X[:, 4] = 1.0

test_X = imp.transform(test_X)
predicted_fare = regr.predict(test_X)
print(predicted_fare[:5])




## === cell 11
my_submission = pd.DataFrame({"key": test_df.key, "fare_amount": predicted_fare})
my_submission.to_csv("submission.csv", index=False)
my_submission.head()
