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
geopy==2.4.1
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

3.27817

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from geopy import distance

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
train_path = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
test_path = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"
sample_path = "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"

train = pd.read_csv(train_path, nrows=1_000_000)  # use a subset for speed
test = pd.read_csv(test_path)
sample_submission = pd.read_csv(sample_path)



## === cell 2
print("Missing values per column:")
print(train.isnull().sum())



## === cell 3
train.dropna(inplace=True)



## === cell 4
print(train.describe())



## === cell 5
print("Rows with passenger_count > 6:", train.query("passenger_count > 6").shape[0])
print("Rows with passenger_count < 1:", train.query("passenger_count < 1").shape[0])
print("Rows with fare_amount < 0:", train.query("fare_amount < 0").shape[0])
print(
    "Rows with invalid longitudes/latitudes:",
    train.query("pickup_longitude < -180 or pickup_longitude > 180").shape[0],
    train.query("dropoff_longitude < -180 or dropoff_longitude > 180").shape[0],
    train.query("pickup_latitude < -90 or pickup_latitude > 90").shape[0],
    train.query("dropoff_latitude < -90 or dropoff_latitude > 90").shape[0],
)



## === cell 6
train = train.query(
    "1 <= passenger_count <= 6 and "
    "0 <= fare_amount and "
    "-180 <= pickup_longitude <= 180 and "
    "-180 <= dropoff_longitude <= 180 and "
    "-90 <= pickup_latitude <= 90 and "
    "-90 <= dropoff_latitude <= 90"
)
print(train.describe())



## === cell 7
train.reset_index(drop=True, inplace=True)
print(train.head())



## === cell 8
data = pd.concat([train, test], sort=False)



## === cell 9
data = data.drop("pickup_datetime", axis=1)
data["key"] = data["key"].astype(str)
print(data.head())



## === cell 10
data["distance"] = data.apply(
    lambda x: distance.distance(
        (x["pickup_latitude"], x["pickup_longitude"]),
        (x["dropoff_latitude"], x["dropoff_longitude"]),
    ).miles,
    axis=1,
)
print(data.head())



## === cell 11
train = data[: len(train)]
test = data[len(train) :]

y_train = train["fare_amount"]
X_train = train.drop(["fare_amount", "key"], axis=1)
X_test = test.drop(["fare_amount", "key"], axis=1)

print("Training shape:", X_train.shape, "Test shape:", X_test.shape)



## === cell 12
cv = KFold(n_splits=5, shuffle=True, random_state=0)
categorical_features = []  # none in this dataset



## === cell 13
params = {
    "objective": "regression",
    "max_bin": 300,
    "learning_rate": 0.05,
    "num_leaves": 40,
    "metric": "l2",
}

y_preds = []
models = []
oof_train = np.zeros(len(X_train))

for fold_id, (train_idx, valid_idx) in enumerate(cv.split(X_train, y_train)):
    X_tr, X_val = X_train.loc[train_idx], X_train.loc[valid_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]

    lgb_train = lgb.Dataset(X_tr, y_tr, categorical_feature=categorical_features)
    lgb_valid = lgb.Dataset(
        X_val, y_val, reference=lgb_train, categorical_feature=categorical_features
    )

    callbacks = [lgb.early_stopping(stopping_rounds=10, verbose=False)]

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_train, lgb_valid],
        callbacks=callbacks,
    )

    oof_train[valid_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)

    y_preds.append(y_pred)
    models.append(model)



## === cell 14
pd.DataFrame(oof_train, columns=["fare_amount"]).to_csv(
    "oof_train_kfold.csv", index=False
)

scores = [m.best_score["valid_1"]["l2"] for m in models]
mean_mse = np.mean(scores)
rmse = np.sqrt(mean_mse)
print("=== CV RMSE ===")
print(rmse)



## === cell 15
rmse_oof = np.sqrt(mean_squared_error(y_train, oof_train))
print("OOF RMSE:", rmse_oof)



## === cell 16
print("Number of folds predictions collected:", len(y_preds))



## === cell 17
if y_preds:
    y_sub = np.mean(np.column_stack(y_preds), axis=1)
else:
    y_sub = np.zeros(len(X_test))

print("First 10 predictions:", y_sub[:10])



## === cell 18
sub_lgb = sample_submission.copy()
sub_lgb["fare_amount"] = y_sub
sub_lgb.to_csv("submission_lightgbm.csv", index=False)
print("Submission saved to submission_lightgbm.csv")
print(sub_lgb.head())
