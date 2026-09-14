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

5.76416

# 6. Current score

8.43775

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1036.90176) has done: 'I remove the TensorFlow Estimator section that crashes under TF 2.18 (no `tf.estimator`) and keep the working scikit-learn LinearRegression pipeline as the core model to generate the submission. I also fix the notebook-only magic (`%matplotlib inline`) so the script runs as a plain Kaggle Python script, and make the input paths robust to both `/kaggle/input/...` and the user’s `../input/...` layout. Finally, I add a small, score-improving but still “same core logic” fix: clip extreme coordinate outliers/invalid rows (NYC bounding box + sane fare cap) and ensure distances are computed in kilometers (your current function returns miles), which is the main reason your RMSE is exploding.'
- What this solution (achieved 8.43775) has done: 'Your public score is exploding because the model is trained on heavily filtered data (distance < 15 km) but you still predict on unfiltered test rows, which creates extreme extrapolation errors; the smallest fix is to apply the exact same coordinate cleaning and distance cap logic to the test set before predicting. To keep the same core model/logic, we not change the model or features; we only (1) ensure train/test go through the same cleaning pipeline, and (2) safely handle the removed test rows by filling them with a reasonable fallback (the training mean fare) so the submission remains complete and valid. This should dramatically reduce RMSE toward your target without altering architecture or training semantics.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

RANDOM_STATE = 42


def _resolve_input_path(filename: str) -> str:
    candidates = [
        os.path.join("../input", filename),
        os.path.join("/kaggle/input", filename),
        os.path.join("/kaggle/input/new-york-city-taxi-fare-prediction", filename),
        os.path.join("../input/new-york-city-taxi-fare-prediction", filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


TRAIN_PATH = _resolve_input_path("train.csv")
TEST_PATH = _resolve_input_path("test.csv")
SAMPLE_SUB_PATH = _resolve_input_path("sample_submission.csv")



## === cell 1
train_df = pd.read_csv(TRAIN_PATH, nrows=1_000_000)



## === cell 2
train_df.shape



## === cell 3
test_df = pd.read_csv(TEST_PATH)



## === cell 4
test_df.shape



## === cell 5
train_df.head(5)



## === cell 6
train_df.isnull().sum()



## === cell 7
train_df.dropna(inplace=True)



## === cell 8
train_df.describe()



## === cell 9
train_df = train_df[train_df["fare_amount"] > 0]
train_df = train_df[train_df["fare_amount"] < 250]



## === cell 10
train_df.shape




## === cell 11
def distance_km(lat1, lon1, lat2, lon2):
    lat1 = lat1.astype(float)
    lon1 = lon1.astype(float)
    lat2 = lat2.astype(float)
    lon2 = lon2.astype(float)
    dlat = (lat2 - lat1) * (np.pi / 180.0)
    dlon = (lon2 - lon1) * (np.pi / 180.0)
    a = (
        0.5
        - np.cos(dlat) / 2.0
        + np.cos(lat1 * (np.pi / 180.0))
        * np.cos(lat2 * (np.pi / 180.0))
        * (1.0 - np.cos(dlon))
        / 2.0
    )
    return 12742.0 * np.arcsin(np.sqrt(a))  # 2*R where R=6371km




## === cell 12
def clean_coords(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lon_min, lon_max = -74.5, -72.8
    lat_min, lat_max = 40.5, 41.8
    for c in ["pickup_longitude", "dropoff_longitude"]:
        df = df[(df[c] >= lon_min) & (df[c] <= lon_max)]
    for c in ["pickup_latitude", "dropoff_latitude"]:
        df = df[(df[c] >= lat_min) & (df[c] <= lat_max)]
    return df


train_df = clean_coords(train_df)



## === cell 13
train_df["distance"] = distance_km(
    train_df.pickup_latitude,
    train_df.pickup_longitude,
    train_df.dropoff_latitude,
    train_df.dropoff_longitude,
)



## === cell 14
test_df["distance"] = distance_km(
    test_df.pickup_latitude,
    test_df.pickup_longitude,
    test_df.dropoff_latitude,
    test_df.dropoff_longitude,
)



## === cell 15
test_df["_row_id"] = np.arange(len(test_df))
test_df_clean = clean_coords(test_df)



## === cell 16
train_df = train_df[train_df["distance"] < 15]



## === cell 17
train_df.describe()



## === cell 18
train_df = train_df[
    (train_df["passenger_count"] != 0) & (train_df["passenger_count"] < 10)
]



## === cell 19
test_df_clean = test_df_clean[
    (test_df_clean["passenger_count"] != 0) & (test_df_clean["passenger_count"] < 10)
]
test_df_clean = test_df_clean[test_df_clean["distance"] < 15]



## === cell 20
feat_cols = ["distance", "passenger_count"]
X = train_df[feat_cols]
y = train_df["fare_amount"]



## === cell 21
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.1, random_state=RANDOM_STATE
)



## === cell 22
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

model_lin = Pipeline(
    (("standard_scaler", StandardScaler()), ("lin_reg", LinearRegression()))
)
model_lin.fit(X_train, y_train)



## === cell 23
from sklearn.metrics import mean_squared_error

valid_pred = model_lin.predict(X_valid)
rmse = mean_squared_error(y_valid, valid_pred, squared=False)
rmse



## === cell 24
fallback = float(y.mean())

pred_series = pd.Series(index=test_df["_row_id"], data=fallback, dtype=float)

if len(test_df_clean) > 0:
    clean_preds = model_lin.predict(test_df_clean[feat_cols])
    clean_preds = np.maximum(clean_preds, 0.0)
    pred_series.loc[test_df_clean["_row_id"].values] = clean_preds

submission = pd.DataFrame({"key": test_df["key"], "fare_amount": pred_series.values})
submission.to_csv("linear_reg.csv", index=False)
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote: linear_reg.csv and submission.csv with shape:", submission.shape)
print(
    "Fallback used for rows removed by cleaning:", int((pred_series == fallback).sum())
)
