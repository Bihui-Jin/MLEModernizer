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

3.12

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
sklearn-pandas==2.2.0
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

4.99506822338208

# 6. Current score

983.79213

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 983.79213) has done: 'The main runtime failure is in the XGBoost submission path: `x_pred` and `test_key` are undefined, and the train/test feature engineering is inconsistent because you drop `key`/`pickup_datetime` only for train, not for test. I fix this by applying the same feature-engineering steps to both train and test, then defining `x_pred` and `test_key` from the processed test set before predicting. I also correct a boolean indexing precedence bug in `idx` (missing parentheses) that can silently select the wrong rows and ruin RMSE. Finally, I ensure the notebook always writes a valid `submission.csv` with columns `key,fare_amount` using the XGBoost model output (the path most likely to move RMSE from ~1440 toward the ~5 target).'
- What this solution (achieved 14.74141) has done: 'Your RMSE is extremely high because the XGBoost model is being trained on unfiltered/ill-conditioned rows (e.g., passenger_count=0, unrealistically long trips, extreme fares) and because early stopping is enabled (which violates your “no early stopping” requirement and also makes training behavior unstable). I keep your exact model/feature logic, but apply the same simple row-quality filters already implied by your notebook (nonzero passengers, reasonable distances, and fare caps) before splitting/training XGBoost. I also remove early stopping while keeping the same boosting rounds to satisfy constraints and make the model converge normally. Finally, I ensure test feature columns are finite (replace inf with NaN then fill with medians from train) so prediction doesn’t get corrupted.'
- What this solution (achieved 983.79213) has done: 'Your current score (14.74 RMSE, lower-is-better) is far from the target (~4.995), so we should legitimately improve generalization without changing your core approach (same features + XGBoost training loop). The biggest minimal win is to remove a remaining source of label/feature noise by applying the same bounding-box geographic filter logic you already use earlier to the XGBoost training subset (right now XGB keeps some out-of-NYC points that the linear path filtered out), while keeping your existing distance/fare/passenger filters intact. I also add a small, metric-aligned post-processing step to clip unrealistically high predictions (using the same 0–250 range you trained on), which typically reduces RMSE on this competition without changing the model itself. Finally, I keep the submission writing path identical and ensure test feature columns are clean and aligned.'
- What this solution (achieved 983.79213) has done: 'Your current RMSE (983.79, lower-is-better) is catastrophically far from the target (~4.995), which almost always indicates a submission formatting/alignment issue or invalid predictions (NaNs/infs/wrong row order), not “model quality.” I make the smallest changes to ensure the XGBoost submission is strictly aligned to `sample_submission.csv` keys (correct order and exact row count), and I harden the test feature preparation so any missing/invalid datetime-derived features are safely imputed from train medians. I also add one minimal, metric-friendly post-processing step (rounding to cents) which tends to slightly reduce RMSE on this fare dataset without changing the model. Core feature engineering, model type, parameters, and training loop remain the same; the goal is to remove submission corruption and get RMSE back into a plausible range toward the 4.995 target.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # ploting library with python
from sklearn.linear_model import LinearRegression  # Library for linear regression model
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
TRAIN_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
TEST_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"
SAMPLE_SUB_PATH = (
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
)



## === cell 2
train_data_set = pd.read_csv(TRAIN_PATH, nrows=100_000, parse_dates=["pickup_datetime"])
train_data_set.head(5)




## === cell 3
def select_within_boundingbox(df, box):
    return (
        (df.pickup_longitude >= box[0])
        & (df.pickup_longitude <= box[1])
        & (df.pickup_latitude >= box[2])
        & (df.pickup_latitude <= box[3])
        & (df.dropoff_longitude >= box[0])
        & (df.dropoff_longitude <= box[1])
        & (df.dropoff_latitude >= box[2])
        & (df.dropoff_latitude <= box[3])
    )


def distance_on_the_sphere(lat1, lon1, lat2, lon2):
    earth_radius = 6371
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = (np.sin(delta_phi / 2.0) ** 2) + np.cos(phi1) * np.cos(phi2) * (
        np.sin(delta_lambda / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return earth_radius * c


new_york_box = (-74.763379, -72.856164, 40.502009, 41.915509)
nyc_down_town = (-74.0, 40.6)  # (lon, lat)
jfk_airport = (-73.8, 40.5)  # (lon, lat)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df["hour"] = df["pickup_datetime"].dt.hour
    df["year"] = df["pickup_datetime"].dt.year
    df["day_of_week"] = df["pickup_datetime"].dt.dayofweek
    df["is_rush_hour"] = df["hour"].apply(
        lambda x: 1 if (x >= 7 and x <= 10) or (x >= 16 and x <= 19) else 0
    )

    df["distance"] = distance_on_the_sphere(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )
    df["distance_to_downtown"] = distance_on_the_sphere(
        nyc_down_town[1],
        nyc_down_town[0],
        df["pickup_latitude"],
        df["pickup_longitude"],
    )
    df["distance_to_jfk_airport"] = distance_on_the_sphere(
        jfk_airport[1], jfk_airport[0], df["pickup_latitude"], df["pickup_longitude"]
    )
    return df


features = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "distance",
    "hour",
    "year",
    "day_of_week",
    "is_rush_hour",
    "distance_to_downtown",
    "distance_to_jfk_airport",
]
target = "fare_amount"



## === cell 4
print(train_data_set.dtypes)
train_data_set.describe()



## === cell 5
old_len = len(train_data_set)
train_data_set = train_data_set[train_data_set.fare_amount >= 0.1]
new_len = len(train_data_set)
print(f"Removed {(old_len-new_len)} entities from the dataset")
train_data_set.describe()



## === cell 6
old_len = len(train_data_set)
train_data_set = train_data_set.dropna(how="any", axis="rows")
new_len = len(train_data_set)
print(f"Removed {(old_len-new_len)} entities from the dataset")



## === cell 7
train_data_set.fare_amount.hist(bins=100, figsize=(14, 3))
plt.xlabel("fare $USD")
plt.title("Histogram")



## === cell 8
old_len = len(train_data_set)
train_data_set = train_data_set[select_within_boundingbox(train_data_set, new_york_box)]
new_len = len(train_data_set)
print(f"Removed {(old_len-new_len)} entities from the dataset")



## === cell 9
train_data_set = add_features(train_data_set)
train_data_set.head(5)



## === cell 10
idx = (train_data_set.passenger_count != 0) & (train_data_set.distance_to_downtown < 15)

X = train_data_set.loc[idx, features].values
y_lr = train_data_set.loc[idx, target].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y_lr, test_size=0.25, random_state=42
)

linear_model = LinearRegression()



## === cell 11
linear_model.fit(X_train, y_train)



## === cell 12
test_data_set = pd.read_csv(TEST_PATH, parse_dates=["pickup_datetime"])
test_data_set = add_features(test_data_set)

XTEST = test_data_set[features].values
y_pred_final = linear_model.predict(XTEST)

submission_lr = pd.DataFrame(
    {"key": test_data_set["key"], "fare_amount": y_pred_final},
    columns=["key", "fare_amount"],
)
submission_lr.to_csv("submission_linear.csv", index=False)



## === cell 13
train_xgb = train_data_set.copy()
train_xgb = train_xgb.dropna(subset=features + [target])

train_xgb = train_xgb[select_within_boundingbox(train_xgb, new_york_box)]

train_xgb = train_xgb[
    (train_xgb["passenger_count"] >= 1) & (train_xgb["passenger_count"] <= 6)
]
train_xgb = train_xgb[(train_xgb["distance"] > 0) & (train_xgb["distance"] < 80)]
train_xgb = train_xgb[(train_xgb["fare_amount"] > 0) & (train_xgb["fare_amount"] < 250)]

train_xgb[features] = train_xgb[features].replace([np.inf, -np.inf], np.nan)
train_xgb = train_xgb.dropna(subset=features + [target])

x_train, x_test, y_train, y_test = train_test_split(
    train_xgb[features], train_xgb[target], random_state=2666, test_size=0.05
)

params = {
    "max_depth": 7,
    "subsample": 0.9,
    "eta": 0.03,
    "colsample_bytree": 0.9,
    "random_state": 2666,
    "tree_method": "hist",
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
}


def XGBmodel(x_train, x_test, y_train, y_test, params):
    matrix_train = xgb.DMatrix(x_train, label=y_train)
    matrix_test = xgb.DMatrix(x_test, label=y_test)

    model = xgb.train(
        params=params,
        dtrain=matrix_train,
        num_boost_round=5000,
        evals=[(matrix_test, "test")],
        verbose_eval=False,
    )
    return model


model = XGBmodel(x_train, x_test, y_train, y_test, params)



## === cell 14
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
sample_keys = sample_sub["key"].astype(str)

test_data_set["key"] = test_data_set["key"].astype(str)

test_key = test_data_set["key"].copy()
x_pred = test_data_set[features].copy()

x_pred = x_pred.replace([np.inf, -np.inf], np.nan)

train_feature_medians = train_xgb[features].median(numeric_only=True)
x_pred = x_pred.fillna(train_feature_medians)

prediction = model.predict(xgb.DMatrix(x_pred))

submission = pd.DataFrame({"key": test_key.values, "fare_amount": prediction})
submission["fare_amount"] = submission["fare_amount"].clip(lower=0.0, upper=250.0)

submission["fare_amount"] = submission["fare_amount"].round(2)

submission = submission.set_index("key").reindex(sample_keys)
submission["fare_amount"] = submission["fare_amount"].fillna(
    sample_sub["fare_amount"].values
)
submission = submission.reset_index()

submission = submission[["key", "fare_amount"]]
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print(
    "Keys match sample_submission:",
    submission["key"].equals(sample_sub["key"].astype(str)),
)

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2869796191.py in <cell line: 0>()
     30 # Any missing keys (shouldn't happen) fall back to the sample's default fare.
     31 submission = submission.set_index("key").reindex(sample_keys)
---> 32 submission["fare_amount"] = submission["fare_amount"].fillna(
     33     sample_sub["fare_amount"].values
     34 )

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in fillna(self, value, method, axis, inplace, limit, downcast)
   7341                     pass
   7342                 else:
-> 7343                     raise TypeError(
   7344                         '"value" parameter must be a scalar, dict '
   7345                         "or Series, but you passed a "

TypeError: "value" parameter must be a scalar, dict or Series, but you passed a "ndarray"
