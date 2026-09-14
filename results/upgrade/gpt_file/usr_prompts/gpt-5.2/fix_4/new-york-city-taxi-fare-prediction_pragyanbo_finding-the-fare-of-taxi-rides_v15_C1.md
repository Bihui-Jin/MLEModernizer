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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

4.58218

# 6. Current score

1033.37678

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1033.37678) has done: 'I remove the notebook-only magic and fix the TensorFlow import crash by avoiding `tf.estimator` (not available in TF 2.18 here) and replacing the broken DNNRegressor section with a minimal Keras regressor using the exact same features and MinMax scaling. I also fix a score-hurting logic issue: your linear model was fit on 4 features but the TF part only used 2; the Keras model use all 4 (`distance`, `passenger_count`, `hour`, `year`) to improve RMSE toward the target. Finally, I ensure the script always writes a valid `submission.csv` with required columns and correct row counts.'
- What this solution (achieved 1033.37678) has done: 'You’re hitting a TensorFlow import-time crash caused by an incompatibility between `tensorflow==2.18.0` and `protobuf==6.x` (`MessageFactory.GetPrototype` missing). Since this prevents the pipeline from finishing and writing a valid `submission.csv`, the minimal safe fix is to remove the TensorFlow/Keras block and instead produce the final submission using one of the already-trained scikit-learn models (keeping the same engineered features and train/test processing). To move RMSE strongly toward the target (and far away from the current ~1033), we use the RandomForest predictions as the final `submission.csv` output (the earlier file name had spaces and wasn’t the required name). All other core feature logic and filters are kept intact; the only logic change is which already-computed predictions are used for the final submission.'
- What this solution (achieved 1033.37678) has done: 'Your RMSE is catastrophically high because the `train.csv` you read does not come from the NYC Taxi Fare competition dataset in `/kaggle/input`; it’s likely a different competition’s file, so the model learns nonsense and outputs wildly wrong fares. The smallest change that moves you strongly toward the target is to point `train_path`/`test_path` (and sample submission, for sanity checks) to the actual NYC Taxi Fare folder that exists in your environment, while keeping your exact feature engineering and RandomForest/LinearRegression logic unchanged. I also keep writing `submission.csv` (required) and add a quick key-alignment check against `sample_submission.csv` to prevent silent row-order issues that can hurt score. No model architecture/training loop changes are made.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

np.random.seed(42)



## === cell 1
BASE_INPUT_DIR = "/kaggle/input"
if not os.path.exists(BASE_INPUT_DIR):
    BASE_INPUT_DIR = "../input"

COMP_DIR = os.path.join(BASE_INPUT_DIR, "new-york-city-taxi-fare-prediction")
if os.path.exists(COMP_DIR):
    INPUT_DIR = COMP_DIR
else:
    INPUT_DIR = BASE_INPUT_DIR

train_path = os.path.join(INPUT_DIR, "train.csv")
test_path = os.path.join(INPUT_DIR, "test.csv")
sample_path = os.path.join(INPUT_DIR, "sample_submission.csv")

print("Using INPUT_DIR:", INPUT_DIR)
print("Train path exists:", os.path.exists(train_path), train_path)
print("Test path exists:", os.path.exists(test_path), test_path)

train_df = pd.read_csv(train_path, nrows=100000)
train_df.shape



## === cell 2
test_df = pd.read_csv(test_path)
test_df.shape



## === cell 3
train_df.head(5)



## === cell 4
train_df.isnull().sum()



## === cell 5
train_df.dropna(inplace=True)



## === cell 6
train_df.describe()



## === cell 7
train_df = train_df[train_df["fare_amount"] > 0]
train_df.shape




## === cell 8
def distance(lat1, lon1, lat2, lon2):
    a = (
        0.5
        - np.cos((lat2 - lat1) * 0.017453292519943295) / 2
        + np.cos(lat1 * 0.017453292519943295)
        * np.cos(lat2 * 0.017453292519943295)
        * (1 - np.cos((lon2 - lon1) * 0.017453292519943295))
        / 2
    )
    res = 0.6213712 * 12742 * np.arcsin(np.sqrt(a))
    return res




## === cell 9
train_df["distance"] = distance(
    train_df.pickup_latitude,
    train_df.pickup_longitude,
    train_df.dropoff_latitude,
    train_df.dropoff_longitude,
)

test_df["distance"] = distance(
    test_df.pickup_latitude,
    test_df.pickup_longitude,
    test_df.dropoff_latitude,
    test_df.dropoff_longitude,
)



## === cell 10
train_df = train_df[train_df["distance"] < 15]
train_df.describe()



## === cell 11
train_df = train_df[
    (train_df["passenger_count"] != 0) & (train_df["passenger_count"] < 10)
]

train_df["pickup_datetime"] = pd.to_datetime(
    train_df["pickup_datetime"], errors="coerce", utc=True
)
test_df["pickup_datetime"] = pd.to_datetime(
    test_df["pickup_datetime"], errors="coerce", utc=True
)

train_df = train_df.dropna(subset=["pickup_datetime"])

train_df["hour"] = train_df["pickup_datetime"].dt.hour
train_df["year"] = train_df["pickup_datetime"].dt.year

test_df["hour"] = test_df["pickup_datetime"].dt.hour
test_df["year"] = test_df["pickup_datetime"].dt.year



## === cell 12
feat_cols_s = ["distance", "passenger_count", "hour", "year"]
X = train_df[feat_cols_s]
y = train_df["fare_amount"]



## === cell 13
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.3, random_state=42
)



## === cell 14
from sklearn.ensemble import RandomForestRegressor

r_reg = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
r_reg.fit(X_train, y_train)

y_pred_final = r_reg.predict(test_df[feat_cols_s])

submission_rf = pd.DataFrame(
    {"key": test_df.key, "fare_amount": y_pred_final}, columns=["key", "fare_amount"]
)
submission_rf.to_csv("Random Forest regression.csv", index=False)



## === cell 15
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

model_l = Pipeline(
    (("standard_scaler", StandardScaler()), ("lin_reg", LinearRegression()))
)
model_l.fit(X_train, y_train)

y_pred_final = model_l.predict(test_df[feat_cols_s])

submission_lr = pd.DataFrame(
    {"key": test_df.key, "fare_amount": y_pred_final}, columns=["key", "fare_amount"]
)
submission_lr.to_csv("linear_reg.csv", index=False)



## === cell 16
if os.path.exists(sample_path):
    sample_sub = pd.read_csv(sample_path)
    if len(sample_sub) == len(test_df) and "key" in sample_sub.columns:
        if not np.array_equal(sample_sub["key"].values, test_df["key"].values):
            test_df = (
                test_df.set_index("key").loc[sample_sub["key"].values].reset_index()
            )

final_pred = submission_rf["fare_amount"].to_numpy(dtype=np.float64)
final_pred = np.clip(final_pred, 0.0, None)

submission = pd.DataFrame(
    {"key": test_df["key"].values, "fare_amount": final_pred},
    columns=["key", "fare_amount"],
)

assert len(submission) == len(test_df), "Submission length mismatch."
assert list(submission.columns) == ["key", "fare_amount"], "Wrong submission columns."

submission.to_csv("submission.csv", index=False)

submission.head()
