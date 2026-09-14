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

3.9

# 3. Installed packages

geopandas==0.14.4
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

4.19963

# 6. Current score

4.96153

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.87932) has done: 'The fix removes the erroneous conversion of the `key` column to a numeric feature, so the model is trained without it and the test data matches the training feature set. We also drop `key` from the test matrix before creating the `DMatrix`. This resolves the feature‑name mismatch, restores the `prediction` variable, and enables creation of a proper submission CSV.'
- What this solution (achieved 6.74807) has done: 'I add lightweight feature engineering (geodesic distance, hour, day‑of‑week and month extracted from the datetime) before the train/validation split so the XGBoost model receives more informative inputs. These new numeric columns are compatible with the existing model parameters and should lower the RMSE, moving the score from 5.88 closer to the target 4.20 without altering the core architecture or training loop. The script is renumbered into sequential cells and ensures the final CSV is written correctly.'
- What this solution (achieved 5.86724) has done: 'I increase the training sample size (from 400 k to 800 k rows) to give the model more data and clip the predicted fares to a realistic range (3 – 200 USD) to avoid extreme errors. These minimal adjustments keep the original architecture and training loop unchanged while aiming to lower the RMSE toward the target.'
- What this solution (achieved 6.14415) has done: 'I add a few lightweight location‑difference features (Δlongitude, Δlatitude) inside the feature‑engineering step and raise the maximum boost rounds to 1000 (early‑stopping still pick the optimal iteration). These changes keep the model architecture unchanged while giving XGBoost slightly richer inputs, which should lower the RMSE toward the target without risking over‑fitting.'
- What this solution (achieved 4.96153) has done: 'I keep the overall pipeline unchanged but switch the target to a log‑scale, which better fits the skewed fare distribution and usually reduces RMSE after back‑transforming. I add a small custom evaluation metric that computes the true RMSE on the original scale so early‑stopping is guided correctly. I also lower the tree depth slightly (from 10 to 8) to curb possible over‑fitting and increase the sampled training rows to 1.2 million for a bit more data. All other steps, feature engineering and submission format, remain the same.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 1
df_train = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", nrows=1_200_000
)
df_train.head()




## === cell 2
df_test = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")
df_test.head()




## === cell 3
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["hour"] = df["pickup_datetime"].dt.hour
    df["dayofweek"] = df["pickup_datetime"].dt.dayofweek
    df["month"] = df["pickup_datetime"].dt.month

    lon1 = np.radians(df["pickup_longitude"].values)
    lat1 = np.radians(df["pickup_latitude"].values)
    lon2 = np.radians(df["dropoff_longitude"].values)
    lat2 = np.radians(df["dropoff_latitude"].values)

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371.0 * c  # Earth radius in km
    df["distance"] = km

    df["delta_longitude"] = df["dropoff_longitude"] - df["pickup_longitude"]
    df["delta_latitude"] = df["dropoff_latitude"] - df["pickup_latitude"]
    return df


df_train = add_features(df_train)
df_test = add_features(df_test)




## === cell 4
df_train.drop(df_train[df_train["passenger_count"] > 5].index, axis=0, inplace=True)
df_train.drop(df_train[df_train["passenger_count"] == 0].index, axis=0, inplace=True)

df_train.drop(df_train[df_train["fare_amount"] < 3].index, axis=0, inplace=True)
df_train.drop(df_train[df_train["fare_amount"] > 200].index, axis=0, inplace=True)

df_train.drop(
    df_train.index[
        (df_train.pickup_longitude < -75.0)
        | (df_train.pickup_longitude > -72.0)
        | (df_train.pickup_latitude < 40.0)
        | (df_train.pickup_latitude > 42.0)
    ],
    inplace=True,
)

df_train.drop(
    df_train.index[
        (df_train.dropoff_longitude < -75.0)
        | (df_train.dropoff_longitude > -72.0)
        | (df_train.dropoff_latitude < 40.0)
        | (df_train.dropoff_latitude > 42.0)
    ],
    inplace=True,
)

df_train.dropna(inplace=True)




## === cell 5
from sklearn.model_selection import train_test_split

X = df_train.drop(["fare_amount", "pickup_datetime", "key"], axis=1)
y = df_train["fare_amount"]

X_test = df_test.drop(["pickup_datetime", "key"], axis=1)

train_x, valid_x, train_y, valid_y = train_test_split(
    X, y, test_size=0.2, random_state=12
)

train_y_log = np.log1p(train_y)
valid_y_log = np.log1p(valid_y)




## === cell 6
import xgboost as xgb

dtrain = xgb.DMatrix(train_x, label=train_y_log)
dvalid = xgb.DMatrix(valid_x, label=valid_y_log)
dtest = xgb.DMatrix(X_test)

watchlist = [(dtrain, "train"), (dvalid, "valid")]


def rmse_original(preds, dmatrix):
    """RMSE computed on the original (non‑log) scale."""
    labels = dmatrix.get_label()
    preds_exp = np.expm1(preds)
    labels_exp = np.expm1(labels)
    return "rmse_original", np.sqrt(((preds_exp - labels_exp) ** 2).mean())


params = {
    "min_child_weight": 1,
    "learning_rate": 0.005,
    "colsample_bytree": 0.7,
    "max_depth": 8,  # slightly reduced depth for better generalisation
    "subsample": 0.7,
    "n_jobs": -1,
    "booster": "gbtree",
    "eval_metric": "rmse",  # auxiliary metric on log scale
}

model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=watchlist,
    early_stopping_rounds=100,
    feval=rmse_original,
    maximize=False,
    verbose_eval=50,
)




## === cell 7
prediction_log = model.predict(dtest)
prediction = np.expm1(prediction_log)
prediction = np.clip(prediction, 3.0, 200.0)




## === cell 8
submission = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
)




## === cell 9
output = pd.DataFrame({"key": submission["key"], "fare_amount": prediction})
output.to_csv("my_submission.csv", index=False)
