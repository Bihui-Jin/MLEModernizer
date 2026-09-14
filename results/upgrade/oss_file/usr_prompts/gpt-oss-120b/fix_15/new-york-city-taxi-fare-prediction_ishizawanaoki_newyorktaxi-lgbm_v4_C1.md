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

3.10

# 3. Installed packages

geopandas==0.14.4
lightgbm==4.6.0
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

3.43515

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.92248) has done: 'I drop the non‑numeric **key** column before feeding the data to LightGBM, which fixes the dtype error and allows the models to train so that predictions are generated and a valid submission CSV is written. This minimal change keeps the original modeling pipeline intact while enabling a proper RMSE calculation and submission file.'
- What this solution (achieved 4.53261) has done: 'I add simple yet effective feature engineering: parse the pickup datetime to create hour, day‑of‑week and month features, and compute the haversine distance between pickup and drop‑off points. These numeric features are well‑suited for LightGBM and should reduce RMSE toward the target. I also switch LightGBM’s evaluation metric to `rmse` (root‑mean‑square error) so early stopping optimises the same loss we finally report. The rest of the pipeline stays unchanged.'
- What this solution (achieved 4.53046) has done: 'The script was using relative paths like “../input/…” which do not point to the actual data location in the Kaggle environment, causing a file‑not‑found error and preventing the creation of a valid submission. I switched the paths to the absolute “/kaggle/input/…” location while leaving the rest of the pipeline untouched, ensuring the data loads correctly and the submission CSV is generated.'
- What this solution (achieved 4.68351) has done: 'I improve the model by training LightGBM on the log‑transformed fare amount (which often stabilises variance and improves RMSE) and by slightly increasing model capacity (more leaves, a small `min_data_in_leaf`). After prediction the values are exponentiated back to the original scale, keeping the rest of the pipeline unchanged.'
- What this solution (achieved 4.3675) has done: 'I replace the log‑target handling with a direct fare‑amount target, add a squared‑distance feature, and adjust variable names so the model trains and predicts on the original scale. This keeps the LightGBM pipeline unchanged while aligning the training loss with the competition RMSE, which should move the score closer to the target 3.43515 and also ensures a proper submission file is written.'

# 9. Code solution

## === cell 0
import random, os
import pandas as pd
import numpy as np

random.seed(0)
np.random.seed(0)
os.environ["PYTHONHASHSEED"] = "0"

train = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv", nrows=2_000_000
)
test = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")
sample_submission = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
)



## === cell 1
data = pd.concat([train, test], sort=False)



## === cell 2
data["pickup_datetime"] = pd.to_datetime(data["pickup_datetime"])
data["hour"] = data["pickup_datetime"].dt.hour
data["dayofweek"] = data["pickup_datetime"].dt.dayofweek
data["month"] = data["pickup_datetime"].dt.month

data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)
data["dow_sin"] = np.sin(2 * np.pi * data["dayofweek"] / 7)
data["dow_cos"] = np.cos(2 * np.pi * data["dayofweek"] / 7)
data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12)
data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12)

lon1 = np.radians(data["pickup_longitude"])
lat1 = np.radians(data["pickup_latitude"])
lon2 = np.radians(data["dropoff_longitude"])
lat2 = np.radians(data["dropoff_latitude"])
dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arcsin(np.sqrt(a))
earth_radius_km = 6371.0
data["distance_km"] = earth_radius_km * c
data["distance_km_sq"] = data["distance_km"] ** 2

data = data.drop("pickup_datetime", axis=1)

data["key"] = data["key"].str.replace("[- :]", "", regex=True)



## === cell 3
orig_train_len = len(train)

train = data[:orig_train_len].reset_index(drop=True)
test = data[orig_train_len:].reset_index(drop=True)

mask = (
    (train["fare_amount"] > 0)
    & (train["fare_amount"] < 200)
    & (train["distance_km"] < 100)
)
train = train[mask].reset_index(drop=True)

y_train_original = train["fare_amount"]
X_train = train.drop(["fare_amount", "key"], axis=1)
X_test = test.drop(["key", "fare_amount"], axis=1, errors="ignore")

X_train_np = X_train.astype(np.float32).values
X_test_np = X_test.astype(np.float32).values
y_train_np = y_train_original.values.astype(np.float32)



## === cell 4
from sklearn.model_selection import KFold
import lightgbm as lgb
import gc

cv = KFold(n_splits=5, shuffle=True, random_state=0)
categorical_features = []  # no categorical features for this baseline

params = {
    "objective": "regression",
    "max_bin": 500,
    "learning_rate": 0.02,
    "num_leaves": 512,
    "min_data_in_leaf": 30,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "metric": "rmse",
    "verbose": -1,
    "seed": 0,
}

oof_train = np.zeros(len(X_train_np))
y_preds = []  # test predictions on original scale
models = []

for fold_id, (train_idx, valid_idx) in enumerate(cv.split(X_train_np, y_train_np)):
    X_tr, X_val = X_train_np[train_idx], X_train_np[valid_idx]
    y_tr, y_val = y_train_np[train_idx], y_train_np[valid_idx]

    lgb_train = lgb.Dataset(
        X_tr, y_tr, categorical_feature=categorical_features, free_raw_data=False
    )
    lgb_val = lgb.Dataset(
        X_val,
        y_val,
        reference=lgb_train,
        categorical_feature=categorical_features,
        free_raw_data=False,
    )

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=5000,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )

    oof_train[valid_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    y_pred = model.predict(X_test_np, num_iteration=model.best_iteration)

    y_preds.append(y_pred)
    models.append(model)

    del X_tr, X_val, y_tr, y_val, lgb_train, lgb_val, model
    gc.collect()



## === cell 5
pd.DataFrame(oof_train, columns=["oof_pred"]).to_csv("oof_train_kfold.csv", index=False)

scores = [m.best_score["valid_0"]["rmse"] for m in models]
score = np.mean(scores)
print("===CV scores (RMSE)===")
print(scores)
print("Mean CV RMSE:", score)



## === cell 6
from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(y_train_original, oof_train))
print("OOF RMSE (original scale):", rmse)



## === cell 7
print("Number of test folds predictions:", len(y_preds))



## === cell 8
y_sub = np.mean(np.column_stack(y_preds), axis=1)
print("First 10 ensemble predictions:", y_sub[:10])



## === cell 9
sub_lgb = sample_submission.copy()
sub_lgb["fare_amount"] = y_sub
sub_lgb.to_csv("submission_lightgbm.csv", index=False)
print("Submission saved to submission_lightgbm.csv")
sub_lgb.head()
