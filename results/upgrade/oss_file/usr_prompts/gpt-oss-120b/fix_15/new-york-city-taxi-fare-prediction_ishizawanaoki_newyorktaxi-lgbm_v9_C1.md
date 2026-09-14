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

3.40091

# 6. Current score

5.26845

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 15.25184) has done: 'The fixes address three breaking points:  
1. **Key handling** – the script no longer tries to convert the string `key` column to float; it drops the column from the feature set, keeping it only for the final submission file.  
2. **LightGBM API** – `verbose_eval` is removed (not supported in the installed version) and replaced with `verbose=-1`.  
3. **Safety checks** – added guards to avoid division‑by‑zero when no models are trained and to ensure predictions are computed before creating the submission file.'
- What this solution (achieved 15.25184) has done: 'The changes fix the LightGBM `train` call (removing the unsupported `verbose` argument) and add useful engineered features – haversine distance between pickup and drop‑off points and simple datetime components (hour, day‑of‑week, month). These features improve the model’s predictive power, moving the RMSE toward the target while preserving the original training logic. The script now runs end‑to‑end and writes a proper `submission_lightgbm.csv` file.'
- What this solution (achieved 15.25184) has done: 'Implemented a minimal fix for the LightGBM training call by replacing the unsupported `verbose_eval` argument with `verbose=-1`. Adjusted the cell numbering to start at 1 as required while preserving the original workflow. No other logic changes were made, so the model now trains correctly and should produce a valid `submission_lightgbm.csv` with a substantially lower RMSE.'
- What this solution (achieved 4.62459) has done: 'The fix removes the unsupported `early_stopping_rounds` and `verbose_eval` arguments from the LightGBM training call, replacing them with the proper callback‑based early stopping and silencing output. Cell numbers are renumbered to start at 1, preserving the original workflow while ensuring the model trains correctly and a valid `submission_lightgbm.csv` is written. This correction should dramatically lower the RMSE toward the target score.'
- What this solution (achieved 4.63746) has done: 'I added a few lightweight feature engineering steps (night flag and cyclical hour sin/cos) and increased the boosting capacity by raising `num_boost_round` to 2000 while giving early stopping a slightly larger patience (30 rounds). These changes keep the original workflow intact but give the LightGBM model a bit more expressive power and richer time‑of‑day information, which should lower the RMSE toward the target. I also renumbered the cells to start at 1 as required.'
- What this solution (achieved 4.63387) has done: 'I added a modest amount of extra data, a simple outlier filter, and a categorical hint for `passenger_count`, then slightly enlarged the LightGBM tree capacity. These tweaks keep the original workflow untouched while giving the model a bit more useful signal, which should lower the RMSE and move it closer to the target score.'
- What this solution (achieved 5.26845) has done: 'The changes keep the same feature engineering and LightGBM training pipeline but reduce costly work: we lower the number of CV folds, tighten histogram bins, use a larger learning rate and fewer leaves, and cut the maximum boost rounds. These hyper‑parameter tweaks preserve the model type and loss while dramatically cutting training time, ensuring the script finishes well within the 600‑second limit. All other logic, data paths, and preprocessing remain unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 1
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
dtypes = {
    "key": "object",
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
train_path = "../input/new-york-city-taxi-fare-prediction/train.csv"
test_path = "../input/new-york-city-taxi-fare-prediction/test.csv"
sample_sub_path = "../input/new-york-city-taxi-fare-prediction/sample_submission.csv"

train = pd.read_csv(
    train_path,
    nrows=5_000_000,
    usecols=usecols,
    dtype=dtypes,
    low_memory=False,
)
test = pd.read_csv(
    test_path,
    usecols=[c for c in usecols if c != "fare_amount"],  # test has no target
    dtype={k: v for k, v in dtypes.items() if k != "fare_amount"},
    low_memory=False,
)
sample_submission = pd.read_csv(sample_sub_path)

test_keys = test["key"].copy()




## === cell 2
train.dropna(inplace=True)




## === cell 3
train = train.query(
    "1 <= passenger_count <= 6 and "
    "0 <= fare_amount <= 200 and "
    "-180 <= pickup_longitude <= 180 and "
    "-180 <= dropoff_longitude <= 180 and "
    "-90 <= pickup_latitude <= 90 and "
    "-90 <= dropoff_latitude <= 90"
)
train.reset_index(drop=True, inplace=True)




## === cell 4
for df in (train, test):
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["hour"] = df["pickup_datetime"].dt.hour.astype(np.int8)
    df["dayofweek"] = df["pickup_datetime"].dt.dayofweek.astype(np.int8)
    df["month"] = df["pickup_datetime"].dt.month.astype(np.int8)

    df["is_night"] = df["hour"].isin([0, 1, 2, 3, 4, 5]).astype(np.int8)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24).astype(np.float32)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24).astype(np.float32)




## === cell 5
def haversine_np(lon1, lat1, lon2, lat2):
    """Vectorized haversine distance in kilometres."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c




## === cell 6
for df in (train, test):
    df["distance"] = haversine_np(
        df["pickup_longitude"],
        df["pickup_latitude"],
        df["dropoff_longitude"],
        df["dropoff_latitude"],
    ).astype(np.float32)

    df["lat_diff"] = (
        (df["pickup_latitude"] - df["dropoff_latitude"]).abs().astype(np.float32)
    )
    df["lon_diff"] = (
        (df["pickup_longitude"] - df["dropoff_longitude"]).abs().astype(np.float32)
    )

    df.drop(["pickup_datetime", "key"], axis=1, inplace=True)

train_processed = train.copy()
test_processed = test.copy()

train_mask = (train_processed["distance"] > 0) & (train_processed["distance"] <= 200)
train_processed = train_processed[train_mask].reset_index(drop=True)

y_train = train_processed["fare_amount"]
X_train = train_processed.drop("fare_amount", axis=1)
X_test = test_processed.copy()

X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

X_train["passenger_count"] = X_train["passenger_count"].astype("category")
X_test["passenger_count"] = X_test["passenger_count"].astype("category")




## === cell 7
from sklearn.model_selection import KFold

cv = KFold(n_splits=3, shuffle=True, random_state=0)
categorical_features = ["passenger_count"]
oof_train = np.zeros(len(X_train))
y_preds = []
models = []




## === cell 8
import lightgbm as lgb

y_train_log = np.log1p(y_train)

params = {
    "objective": "regression",
    "metric": "rmse",
    "max_bin": 255,  # tighter bins → faster histogram building
    "learning_rate": 0.07,  # larger step size reduces needed rounds
    "num_leaves": 256,  # fewer leaves = less computation per tree
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_data_in_leaf": 100,  # modest leaf size speeds up training
    "verbosity": -1,
    "force_col_wise": True,
    "num_threads": 0,
}

for fold_id, (train_idx, valid_idx) in enumerate(cv.split(X_train, y_train_log)):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
    y_tr, y_val = y_train_log.iloc[train_idx], y_train_log.iloc[valid_idx]

    lgb_train = lgb.Dataset(
        X_tr,
        y_tr,
        categorical_feature=categorical_features,
        free_raw_data=False,
    )
    lgb_valid = lgb.Dataset(
        X_val,
        y_val,
        reference=lgb_train,
        categorical_feature=categorical_features,
        free_raw_data=False,
    )

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=3000,  # lower max rounds; early stopping will stop earlier if needed
        valid_sets=[lgb_valid],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    oof_train[valid_idx] = np.expm1(
        model.predict(X_val, num_iteration=model.best_iteration)
    )

    y_pred_fold = np.expm1(model.predict(X_test, num_iteration=model.best_iteration))
    y_preds.append(y_pred_fold)
    models.append(model)




## === cell 9
pd.DataFrame(oof_train, columns=["oof_pred"]).to_csv("oof_train_kfold.csv", index=False)

if models:
    cv_scores = [m.best_score["valid_0"]["rmse"] for m in models]
    avg_score = np.mean(cv_scores)
    print("=== CV scores per fold (log RMSE) ===")
    print(cv_scores)
    print("Average CV RMSE (log space):", avg_score)
else:
    print("No models were trained.")




## === cell 10
from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(y_train, oof_train))
print("OOF RMSE on original scale:", rmse)




## === cell 11
y_sub = np.mean(y_preds, axis=0)




## === cell 12
submission = pd.DataFrame({"key": test_keys, "fare_amount": y_sub})
submission.to_csv("submission_lightgbm.csv", index=False)
print("Submission saved to submission_lightgbm.csv")
