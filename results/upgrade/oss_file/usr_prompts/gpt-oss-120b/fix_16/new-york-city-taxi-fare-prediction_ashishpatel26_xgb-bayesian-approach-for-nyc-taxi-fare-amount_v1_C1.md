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
geopy==2.4.1
lightgbm==4.6.0
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
seaborn==0.12.2
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

4.03634

# 6. Current score

5.56483

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.37619) has done: 'I fixed the notebook so it runs from start to finish and creates a proper `submission.csv`. The changes • comment out the IPython magic, • replace the removed `VincentyDistance` with `geodesic`, • ensure the distance column is kept, • select only numeric features for training/prediction, • remove non‑numeric columns before fitting LightGBM, and • add the final prediction‑to‑CSV step. These fixes resolve the runtime errors and let the model train and output a valid submission file.'
- What this solution (achieved 5.36537) has done: 'Implemented a minimal fix to allow the LightGBM model to train and generate predictions. Removed the unsupported `verbose` argument from `LGBMRegressor.fit()` which caused the fit to fail, consequently preventing validation, test predictions, and submission creation. With the model now fitting correctly, subsequent cells execute without errors and produce a valid `submission.csv` file containing the required columns.'
- What this solution (achieved 5.47147) has done: 'I keep the overall pipeline unchanged but train the model on a log‑transformed target, then back‑transform predictions. This often lowers RMSE for skewed fare amounts, moving the score closer to the target while preserving all existing features and model settings.'
- What this solution (achieved 5.3963) has done: 'The fix adds handling for missing datetime‑derived features, drops any rows that still contain NaNs before training, and fills missing test values with the training medians. A simple engineered feature `distance_per_passenger` is introduced and included in the model, and LightGBM’s capacity is increased slightly (more leaves and estimators). These changes eliminate the NaN error, improve the model’s predictive power, and keep the overall pipeline unchanged while moving the RMSE closer to the target.'
- What this solution (achieved 5.43608) has done: 'I filter out rows with non‑positive fare amounts right after filling missing values so that the log‑transform never receives invalid (negative) values, which caused NaNs in the validation set and broke the RMSE calculation. This small change removes the NaNs without altering the model architecture or training procedure, and it should improve the RMSE toward the target.'
- What this solution (achieved 5.41927) has done: 'I add a modest outlier filter on the fare amount, enrich the datetime‑derived features with minute and a weekend flag, and include these new columns in the model’s feature set. These adjustments keep the original pipeline intact while providing extra predictive signals and removing extreme values that can inflate the RMSE, moving the score closer to the target.'
- What this solution (achieved 5.33399) has done: 'I add cyclical time features, adjust the LightGBM hyper‑parameters, split the data before applying the log‑transform so we can train both a log‑target model and a raw‑target model, and then ensemble their predictions (average of the back‑transformed log predictions and the raw predictions). This modest feature‑engineering and model‑averaging is expected to reduce RMSE toward the target without changing the core pipeline.'
- What this solution (achieved 5.68214) has done: 'I tighten the pipeline by (1) discarding implausibly long trips, (2) adding log‑scaled distance and passenger‑count features, (3) including these new features in the model, and (4) modestly expanding LightGBM’s capacity (more leaves) while keeping early stopping. These targeted changes keep the original architecture and training flow but give the model richer signals, which should lower the validation RMSE toward the target.'
- What this solution (achieved 5.68987) has done: 'I keep the overall pipeline unchanged but replace the ensemble averaging of the log‑transformed and raw LightGBM predictions with a single prediction based on the log‑target model (back‑transformed). Using only the log‑target model generally yields lower RMSE for skewed fare amounts, moving the validation score closer to the target while preserving all existing features and training settings.'
- What this solution (achieved 5.68214) has done: 'I keep the overall pipeline unchanged but improve the prediction by averaging the log‑target model (back‑transformed) with the raw‑target model instead of using only the log model. This simple ensemble usually lowers RMSE and moves the score closer to the target while respecting all constraints.'
- What this solution (achieved 5.67707) has done: 'The changes add a simple distance‑squared feature, keep only the log‑target LightGBM model (which is generally more stable for skewed fares), and adjust the validation and test prediction steps to use this single model; this modest refinement is expected to lower the RMSE and move the score nearer to the target while preserving the original pipeline logic.'
- What this solution (achieved 5.72996) has done: 'I add a few simple interaction features (hour‑distance and night flag) and train both a log‑target and a raw‑target LightGBM model, then average their predictions. This modest enrichment keeps the original pipeline intact while providing extra signal, and averaging the two models usually lowers RMSE, moving the score closer to the target.'
- What this solution (achieved 5.56483) has done: 'I simplify the model by keeping only the log‑target LightGBM model (which usually performs better for skewed fare amounts) and use its back‑transformed predictions for validation and test. This removes the raw‑target model that was degrading performance, so the validation RMSE should move closer to the target while preserving the overall pipeline.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("fivethirtyeight")
import geopy.distance
import os, gc

print("Available input files:", os.listdir("../input"))
gc.collect()




## === cell 1
def load_Data():
    train = pd.read_csv("../input/train.csv", nrows=2_000_000, low_memory=True)
    test = pd.read_csv("../input/test.csv", nrows=2_000_000, low_memory=True)
    return train, test




## === cell 2
train, test = load_Data()




## === cell 3
train = train.fillna(0)
train = train[train["fare_amount"] > 0]  # filter non‑positive fares (avoid log issues)
train = train[train["fare_amount"] < 200]  # modest upper bound
test = test.fillna(0)




## === cell 4
train["key_dt"] = pd.to_datetime(train["key"], errors="coerce")
test["key_dt"] = pd.to_datetime(test["key"], errors="coerce")




## === cell 5
conds = (
    (train["pickup_latitude"].between(-90, 90))
    & (train["dropoff_latitude"].between(-90, 90))
    & (train["pickup_longitude"].between(-180, 180))
    & (train["dropoff_longitude"].between(-180, 180))
)
train = train[conds]




## === cell 6
def compute_distance(df):
    return df.apply(
        lambda row: geopy.distance.geodesic(
            (row["pickup_latitude"], row["pickup_longitude"]),
            (row["dropoff_latitude"], row["dropoff_longitude"]),
        ).km,
        axis=1,
    )


train["distance"] = compute_distance(train)
test["distance"] = compute_distance(test)

train = train[train["distance"] <= 100]  # remove unrealistic long trips




## === cell 7
train.drop(
    [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "key_dt",
    ],
    axis=1,
    inplace=True,
)
test.drop(
    [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "key_dt",
    ],
    axis=1,
    inplace=True,
)




## === cell 8
for df in (train, test):
    df["year"] = pd.to_datetime(df["key"]).dt.year
    df["month"] = pd.to_datetime(df["key"]).dt.month
    df["day"] = pd.to_datetime(df["key"]).dt.day
    df["day_of_week"] = pd.to_datetime(df["key"]).dt.weekday
    df["hour"] = pd.to_datetime(df["key"]).dt.hour
    df["minute"] = pd.to_datetime(df["key"]).dt.minute
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    df["distance_hour"] = df["distance"] * df["hour"]
    df["is_night"] = ((df["hour"] < 6) | (df["hour"] > 22)).astype(int)

train["distance_per_passenger"] = train["distance"] / train["passenger_count"].replace(
    0, np.nan
)
test["distance_per_passenger"] = test["distance"] / test["passenger_count"].replace(
    0, np.nan
)

train["distance_per_passenger"] = train["distance_per_passenger"].fillna(0)
test["distance_per_passenger"] = test["distance_per_passenger"].fillna(0)

train["log_distance"] = np.log1p(train["distance"])
test["log_distance"] = np.log1p(test["distance"])

train["log_passenger_count"] = np.log1p(train["passenger_count"])
test["log_passenger_count"] = np.log1p(test["passenger_count"])

train["distance_squared"] = train["distance"] ** 2
test["distance_squared"] = test["distance"] ** 2




## === cell 9
feature_cols = [
    "passenger_count",
    "distance",
    "distance_squared",
    "distance_per_passenger",
    "year",
    "month",
    "day",
    "day_of_week",
    "hour",
    "minute",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
    "log_distance",
    "log_passenger_count",
    "distance_hour",
    "is_night",
]

train = train.dropna(subset=feature_cols)

X = train[feature_cols]
y = train["fare_amount"]




## === cell 10
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)




## === cell 11
median_vals = X_train.median()
X_train = X_train.fillna(median_vals)
X_val = X_val.fillna(median_vals)
test_features = test[feature_cols].fillna(median_vals)




## === cell 12
from lightgbm import LGBMRegressor, early_stopping

lgb_params = dict(
    boosting_type="gbdt",
    colsample_bytree=0.9,
    learning_rate=0.03,
    max_depth=-1,
    min_child_samples=55,
    min_child_weight=0.001,
    min_split_gain=0.1,
    n_estimators=5000,
    n_jobs=-1,
    num_leaves=500,
    reg_alpha=5.0,
    reg_lambda=3.0,
    subsample=1.0,
    subsample_for_bin=200000,
    subsample_freq=1,
    silent=True,
)

lgb_log = LGBMRegressor(**lgb_params)
lgb_log.fit(
    X_train,
    y_train_log,
    eval_set=[(X_val, y_val_log)],
    eval_metric="rmse",
    callbacks=[early_stopping(stopping_rounds=100, verbose=False)],
)





## === cell 13
from sklearn.metrics import mean_squared_error

val_pred = np.expm1(lgb_log.predict(X_val))
rmse = mean_squared_error(y_val, val_pred, squared=False)
print(f"Validation RMSE (log‑model only): {rmse:.4f}")




## === cell 14
test_pred = np.expm1(lgb_log.predict(test_features))




## === cell 15
submission = pd.DataFrame({"key": test["key"], "fare_amount": test_pred})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
