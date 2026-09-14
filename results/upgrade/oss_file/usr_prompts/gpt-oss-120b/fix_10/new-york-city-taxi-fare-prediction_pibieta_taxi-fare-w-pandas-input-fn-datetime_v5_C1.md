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

6.818282725949152e+101

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 988.00752) has done: 'The main issues were the distance function using scalar‑only `math` calls, which caused a TypeError when given NumPy arrays, and the subsequent cascade of errors because the “distance” column never got created. I replaced the function with a fully vectorized NumPy implementation, updated the feature‑addition helper, and reordered the cells so that the dataframe is filtered only after the new column exists. All variable names are kept the same, the model training runs, and a proper `submission_file.csv` is written.'
- What this solution (achieved 1045.86701) has done: 'I add a small step to fill missing feature values with column medians and tighten the training‑data filter by dropping extreme fare and distance outliers. These changes keep the overall pipeline and linear model unchanged but prevent huge errors from bad rows, which should pull the RMSE down toward the target value.'
- What this solution (achieved 1045.86701) has done: 'I fix the `fillna` call that crashes because it tries to compute a median on non‑numeric columns (the `key` and datetime fields). By using `numeric_only=True` we fill only the numeric features, leaving the categorical/date columns untouched. This resolves the runtime error and lets the rest of the pipeline run, producing a proper submission CSV and a realistic RMSE closer to the target.'
- What this solution (achieved 6.818282725949152e+101) has done: 'I increase the training sample size (to 500 k rows) and add a small, useful feature – the log‑scaled distance – which works well with the log‑target regression already used. These minimal changes keep the linear‑regression core unchanged, improve the model’s ability to capture non‑linear distance effects, and should lower the validation RMSE toward the target while still producing a correct `submission_file.csv`.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O
import os, shutil

print("../", os.listdir("../"))
print("../input", os.listdir("../input"))




## === cell 1
df = pd.read_csv("../input/train.csv", nrows=500000, parse_dates=["pickup_datetime"])
test = pd.read_csv("../input/test.csv", parse_dates=["pickup_datetime"])




## === cell 2
def distance(lat1, lon1, lat2, lon2):
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = lat2_rad - lat1_rad
    dlon = np.radians(lon2 - lon1)

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    earth_radius_km = 6371.0
    return earth_radius_km * c




## === cell 3
def add_feats(df):
    df["distance"] = distance(
        df["pickup_latitude"].values,
        df["pickup_longitude"].values,
        df["dropoff_latitude"].values,
        df["dropoff_longitude"].values,
    )
    df["log_distance"] = np.log1p(df["distance"])
    df["hour"] = df.pickup_datetime.dt.hour
    df["weekday"] = df.pickup_datetime.dt.weekday
    return df


df = add_feats(df)
testdf = add_feats(test)

df = df.fillna(df.median(numeric_only=True))
testdf = testdf.fillna(testdf.median(numeric_only=True))




## === cell 4
dfc = df[
    (df.pickup_longitude >= -75.0)
    & (df.pickup_longitude <= -72)
    & (df.pickup_latitude >= 38)
    & (df.pickup_latitude <= 42)
    & (df.dropoff_longitude >= -75.0)
    & (df.dropoff_longitude <= -72)
    & (df.dropoff_latitude >= 38)
    & (df.dropoff_latitude <= 42)
    & (df.fare_amount > 0)  # allow all positive fares
    & (df.fare_amount < 200)  # filter extreme fare outliers
    & (df.passenger_count > 0)
    & (df.passenger_count < 7)
    & (df.distance > 0)  # keep any non‑zero distance
    & (df.distance < 100)  # filter unrealistic distances
]




## === cell 5
np.random.seed(seed=1)  # reproducibility
msk = np.random.rand(len(dfc)) < 0.8
traindf = dfc[msk].drop(["key", "pickup_datetime"], axis=1).reset_index(drop=True)
evaldf = dfc[~msk].drop(["key", "pickup_datetime"], axis=1).reset_index(drop=True)




## === cell 6
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X_train = traindf.drop(["fare_amount"], axis=1)
y_train_log = np.log1p(traindf["fare_amount"])
X_eval = evaldf.drop(["fare_amount"], axis=1)
y_eval = evaldf["fare_amount"]

model = LinearRegression()
model.fit(X_train, y_train_log)

eval_pred_log = model.predict(X_eval)
eval_pred = np.expm1(eval_pred_log)

rmse = mean_squared_error(y_eval, eval_pred, squared=False)
print(f"Validation RMSE: {rmse:.5f}")




## === cell 7
testdf = testdf.drop(["key", "pickup_datetime"], axis=1)

test_pred_log = model.predict(testdf)
test_pred = np.expm1(test_pred_log)

submission = pd.DataFrame({"key": test["key"], "fare_amount": test_pred})
submission_path = "submission_file.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
