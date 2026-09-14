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

3.7695

# 6. Current score

5.101

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 5.101) has done: 'Your notebook already has solid core logic (LightGBM + haversine + 5-fold CV), but it likely didn’t yield a Kaggle score because the submission can silently end up misaligned: you overwrite `test` with feature-engineered data, then later rebuild `test_keys` from that modified frame, and you also concatenate/split after `dropna`/filtering which can shift indices if you’re not careful. I make minimal changes to preserve the exact modeling approach while ensuring the submission `key` ordering matches the original raw test file, and I add a small, standard coordinate-range filter to remove clearly invalid GPS rows (this usually improves RMSE without changing the model). Finally, I keep file paths the same and guarantee `submission.csv` is written with the required columns and row count.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
train = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/train.csv", nrows=1_000_000
)
test = pd.read_csv("../input/new-york-city-taxi-fare-prediction/test.csv")
sample_submission = pd.read_csv(
    "../input/new-york-city-taxi-fare-prediction/sample_submission.csv"
)

test_keys_raw = test["key"].astype(str).reset_index(drop=True)



## === cell 2
train.isnull().sum()



## === cell 3
train.dropna(inplace=True)



## === cell 4
train.describe()



## === cell 5
train.select_dtypes(include=[np.number]).quantile(0.99)



## === cell 6
train.select_dtypes(include=[np.number]).quantile(0.01)



## === cell 7
train = train.query("1 <= passenger_count <= 6 and 3.3 <= fare_amount <= 52.33")
train.describe()



## === cell 8
train.reset_index(drop=True, inplace=True)
train



## === cell 9
n_train = len(train)
data = pd.concat([train, test], sort=False, ignore_index=True)



## === cell 10
data.head()




## === cell 11
def haversine_np(lon1, lat1, lon2, lat2):
    lon1 = np.radians(lon1.astype(float))
    lat1 = np.radians(lat1.astype(float))
    lon2 = np.radians(lon2.astype(float))
    lat2 = np.radians(lat2.astype(float))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371.0 * c
    return km


data["distance_km"] = haversine_np(
    data["pickup_longitude"],
    data["pickup_latitude"],
    data["dropoff_longitude"],
    data["dropoff_latitude"],
)

coord_ok = (
    data["pickup_longitude"].between(-180, 180)
    & data["dropoff_longitude"].between(-180, 180)
    & data["pickup_latitude"].between(-90, 90)
    & data["dropoff_latitude"].between(-90, 90)
)
data.loc[: n_train - 1, "coord_ok"] = coord_ok.iloc[:n_train].values
data.loc[n_train:, "coord_ok"] = True  # keep all test rows

data = data[data["coord_ok"]].drop(columns=["coord_ok"]).reset_index(drop=True)

n_train = data["fare_amount"].notna().sum()

data = data.drop("pickup_datetime", axis=1)

data["key"] = data["key"].astype(str)

data.head()



## === cell 12
train = data.iloc[:n_train].copy()
test = data.iloc[n_train:].copy()

y_train = train["fare_amount"].astype(float)

X_train = train.drop(["fare_amount", "key"], axis=1)
X_test = test.drop(
    ["fare_amount", "key"], axis=1, errors="ignore"
)  # test has no fare_amount

X_train.head()



## === cell 13
from sklearn.model_selection import KFold

y_preds = []
models = []
oof_train = np.zeros((len(X_train),), dtype=np.float32)
cv = KFold(n_splits=5, shuffle=True, random_state=0)

categorical_features = []



## === cell 14
import lightgbm as lgb

params = {
    "objective": "regression",
    "max_bin": 300,
    "learning_rate": 0.05,
    "num_leaves": 40,
    "metric": "rmse",
    "verbosity": -1,
}

for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train, y_train)):
    X_tr = X_train.iloc[train_index, :]
    X_val = X_train.iloc[valid_index, :]
    y_tr = y_train.iloc[train_index]
    y_val = y_train.iloc[valid_index]

    lgb_train = lgb.Dataset(
        X_tr, y_tr, categorical_feature=categorical_features, free_raw_data=False
    )
    lgb_eval = lgb.Dataset(
        X_val,
        y_val,
        reference=lgb_train,
        categorical_feature=categorical_features,
        free_raw_data=False,
    )

    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_eval],
        valid_names=["train", "valid"],
        num_boost_round=1000,
        callbacks=[
            lgb.log_evaluation(period=50),
        ],
    )

    oof_train[valid_index] = model.predict(X_val, num_iteration=model.best_iteration)
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)

    y_preds.append(y_pred)
    models.append(model)



## === cell 15
pd.DataFrame(oof_train).to_csv("oof_train_kfold.csv", index=False)

if len(models) == 0:
    print("WARNING: No models were trained; cannot compute CV scores.")
else:
    scores = []
    for m in models:
        valid_scores = m.best_score.get("valid", {})
        if "rmse" in valid_scores:
            scores.append(valid_scores["rmse"])
    if len(scores) > 0:
        score = sum(scores) / len(scores)
        print("===CV scores (RMSE)===")
        print(scores)
        print(score)
    else:
        print(
            "WARNING: Could not read rmse from model.best_score; see sklearn RMSE below."
        )



## === cell 16
from sklearn.metrics import mean_squared_error

y_pred_oof = oof_train
np.sqrt(mean_squared_error(y_train, y_pred_oof))



## === cell 17
len(y_preds)



## === cell 18
if len(y_preds) == 0:
    y_sub = np.full(
        shape=(len(test_keys_raw),),
        fill_value=float(train["fare_amount"].mean()),
        dtype=np.float32,
    )
else:
    y_sub = sum(y_preds) / len(y_preds)

if len(y_sub) != len(test_keys_raw):
    fallback = float(train["fare_amount"].mean())
    y_sub = np.full(shape=(len(test_keys_raw),), fill_value=fallback, dtype=np.float32)

y_sub[:10]



## === cell 19
sub = pd.DataFrame(
    {
        "key": test_keys_raw,
        "fare_amount": pd.to_numeric(y_sub, errors="coerce"),
    }
)

if sub["fare_amount"].isna().any():
    fallback = float(train["fare_amount"].mean())
    sub["fare_amount"] = sub["fare_amount"].fillna(fallback)

sub = sub[["key", "fare_amount"]]

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
sub.head()
