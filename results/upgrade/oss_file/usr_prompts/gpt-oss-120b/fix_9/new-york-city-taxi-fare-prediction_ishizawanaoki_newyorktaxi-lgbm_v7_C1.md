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

3.42426

# 6. Current score

10.02944

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 4.57325) has done: 'I fixed the data preprocessing (removed the problematic `key` conversion, parsed the datetime for useful time features, added a haversine distance column, and dropped non‑numeric columns), updated the LightGBM training call to the current API (using callbacks instead of `verbose_eval`), ensured predictions are collected correctly, and finally wrote a proper submission CSV with the required `key` and `fare_amount` columns. These changes resolve the runtime errors and produce a valid submission while keeping the original model logic, which should achieve an RMSE close to the target.'
- What this solution (achieved 4.74592) has done: 'I apply a log‑transform to the target variable, train LightGBM on the transformed values, and back‑transform predictions for the OOF score and final submission. This small change often reduces RMSE for skewed targets like fare amounts while keeping the core model and workflow unchanged.'
- What this solution (achieved 4.76876) has done: 'I add a few lightweight feature engineering steps and a small outlier filter that usually improve LightGBM’s RMSE without changing the core modeling pipeline. Specifically I will:
1. Remove rows with impossible fare amounts (≤0 or ≥200) after dropping NaNs.
2. Create cyclic hour and weekday encodings (`sin`/`cos`) and a log‑distance column.
3. Slightly increase LightGBM’s `num_leaves` to give the model more flexibility.
These changes are minimal, keep the original workflow intact, and are expected to move the validation RMSE closer to the target 3.42426.'
- What this solution (achieved 4.6799) has done: 'I add a lightweight “distance per passenger” feature, increase model capacity (more leaves) and give LightGBM a larger early‑stopping patience so it can train a few more boosting rounds. These minimal changes keep the original pipeline intact while expectedly lowering the RMSE, moving the score closer to the target.'
- What this solution (achieved 4.466) has done: 'I add a few inexpensive engineered features (latitude/longitude deltas and Manhattan distance) and expose them to the model, plus a modest increase in model capacity and regularisation tweaks (more leaves, feature/bagging fractions, lower min_data_in_leaf and a longer early‑stopping patience). These changes keep the overall LightGBM workflow intact while giving the model more useful signals, which should lower the RMSE toward the target.'
- What this solution (achieved 4.50681) has done: 'I add a few inexpensive engineered features (binary weekend flag and cyclic month encoding) that give the model extra temporal signal, and increase LightGBM capacity slightly (more leaves, lower learning rate, slightly stronger bagging) while keeping the same training‑validation split and log‑target handling. These changes are minimal, preserve the core pipeline, and are expected to lower the RMSE, moving the score closer to the target.'
- What this solution (achieved 10.02944) has done: 'The changes focus on speeding up data handling and LightGBM training without altering the modeling approach.  
- Convert all feature columns to `float32` to reduce memory bandwidth and accelerate LightGBM’s native computations.  
- Explicitly cast the target log‑values to `float32` as well.  
These type reductions are lossless for model training and preserve the exact algorithmic steps and predictions.'

# 9. Code solution

## === cell 0
import os, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 1
train_path = "../input/new-york-city-taxi-fare-prediction/train.csv"
test_path = "../input/new-york-city-taxi-fare-prediction/test.csv"
sample_sub_path = "../input/new-york-city-taxi-fare-prediction/sample_submission.csv"

train = pd.read_csv(train_path, nrows=2_000_000)
test = pd.read_csv(test_path)
sample_submission = pd.read_csv(sample_sub_path)




## === cell 2
train.dropna(inplace=True)

train = train[(train["fare_amount"] > 0) & (train["fare_amount"] < 200)]




## === cell 3
def haversine_np(lon1, lat1, lon2, lat2):
    """Vectorized haversine distance (km)."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    km = 6371.0 * 2 * np.arcsin(np.sqrt(a))
    return km


def add_time_features(df):
    dt = pd.to_datetime(df["pickup_datetime"])
    df["pickup_hour"] = dt.dt.hour
    df["pickup_dayofweek"] = dt.dt.dayofweek
    df["pickup_month"] = dt.dt.month
    df["is_weekend"] = (df["pickup_dayofweek"] >= 5).astype(int)
    return df


def add_cyclic_features(df):
    df["hour_sin"] = np.sin(2 * np.pi * df["pickup_hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["pickup_hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["pickup_dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["pickup_dayofweek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["pickup_month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["pickup_month"] / 12)
    return df


train = add_time_features(train)
test = add_time_features(test)

train["distance"] = haversine_np(
    train["pickup_longitude"],
    train["pickup_latitude"],
    train["dropoff_longitude"],
    train["dropoff_latitude"],
)
test["distance"] = haversine_np(
    test["pickup_longitude"],
    test["pickup_latitude"],
    test["dropoff_longitude"],
    test["dropoff_latitude"],
)

train["distance_per_passenger"] = train["distance"] / train["passenger_count"]
test["distance_per_passenger"] = test["distance"] / test["passenger_count"]

train["delta_lat"] = train["dropoff_latitude"] - train["pickup_latitude"]
test["delta_lat"] = test["dropoff_latitude"] - test["pickup_latitude"]
train["delta_lon"] = train["dropoff_longitude"] - train["pickup_longitude"]
test["delta_lon"] = test["dropoff_longitude"] - test["pickup_longitude"]
train["manhattan"] = np.abs(train["delta_lat"]) + np.abs(train["delta_lon"])
test["manhattan"] = np.abs(test["delta_lat"]) + np.abs(test["delta_lon"])
train["log_manhattan"] = np.log1p(train["manhattan"])
test["log_manhattan"] = np.log1p(test["manhattan"])

train = add_cyclic_features(train)
test = add_cyclic_features(test)

train["log_distance"] = np.log1p(train["distance"])
test["log_distance"] = np.log1p(test["distance"])

train["log_distance_per_passenger"] = np.log1p(train["distance_per_passenger"])
test["log_distance_per_passenger"] = np.log1p(test["distance_per_passenger"])

train["log_passenger"] = np.log1p(train["passenger_count"])
test["log_passenger"] = np.log1p(test["passenger_count"])

train["is_night"] = ((train["pickup_hour"] >= 22) | (train["pickup_hour"] <= 5)).astype(
    int
)
test["is_night"] = ((test["pickup_hour"] >= 22) | (test["pickup_hour"] <= 5)).astype(
    int
)

cols_to_drop = ["key", "pickup_datetime"]
train = train.drop(columns=cols_to_drop)
test = test.drop(columns=cols_to_drop)




## === cell 4
y_train = train["fare_amount"]
y_train_log = np.log1p(y_train).astype(np.float32)  # cast to float32 for speed

X_train = train.drop("fare_amount", axis=1).astype(np.float32)  # float32 conversion
X_test = test.copy().astype(np.float32)




## === cell 5
cv = KFold(n_splits=5, shuffle=True, random_state=0)

oof_preds = np.zeros(len(X_train))
test_preds = []

params = {
    "objective": "regression",
    "max_bin": 400,
    "learning_rate": 0.02,
    "num_leaves": 1024,
    "feature_fraction": 0.90,
    "bagging_fraction": 0.80,
    "bagging_freq": 5,
    "metric": "rmse",
    "verbosity": -1,
    "min_data_in_leaf": 5,
}

for fold, (train_idx, valid_idx) in enumerate(cv.split(X_train, y_train_log)):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
    y_tr_log, y_val_log = y_train_log[train_idx], y_train_log[valid_idx]

    lgb_train = lgb.Dataset(X_tr, y_tr_log)
    lgb_valid = lgb.Dataset(X_val, y_val_log, reference=lgb_train)

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=5000,
        valid_sets=[lgb_train, lgb_valid],
        callbacks=[
            lgb.log_evaluation(period=10),
            lgb.early_stopping(stopping_rounds=200, verbose=False),
        ],
    )

    oof_preds[valid_idx] = np.expm1(
        model.predict(X_val, num_iteration=model.best_iteration)
    )
    test_preds.append(
        np.expm1(model.predict(X_test, num_iteration=model.best_iteration))
    )




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1944827981.py in <cell line: 0>()
     19 for fold, (train_idx, valid_idx) in enumerate(cv.split(X_train, y_train_log)):
     20     X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
---> 21     y_tr_log, y_val_log = y_train_log[train_idx], y_train_log[valid_idx]
     22 
     23     lgb_train = lgb.Dataset(X_tr, y_tr_log)

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in __getitem__(self, key)
   1151             return self._get_rows_with_mask(key)
   1152 
-> 1153         return self._get_with(key)
   1154 
   1155     def _get_with(self, key):

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in _get_with(self, key)
   1178             #  (i.e. self.iloc) or label-based (i.e. self.loc)
   1179             if not self.index._should_fallback_to_positional:
-> 1180                 return self.loc[key]
   1181             else:
   1182                 warnings.warn(

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in __getitem__(self, key)
   1189             maybe_callable = com.apply_if_callable(key, self.obj)
   1190             maybe_callable = self._check_deprecated_callable_usage(key, maybe_callable)
-> 1191             return self._getitem_axis(maybe_callable, axis=axis)
   1192 
   1193     def _is_scalar_access(self, key: tuple):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_axis(self, key, axis)
   1418                     raise ValueError("Cannot index with multidimensional key")
   1419 
-> 1420                 return self._getitem_iterable(key, axis=axis)
   1421 
   1422             # nested tuple slicing

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_iterable(self, key, axis)
   1358 
   1359         # A collection of keys
-> 1360         keyarr, indexer = self._get_listlike_indexer(key, axis)
   1361         return self.obj._reindex_with_indexers(
   1362             {axis: [keyarr, indexer]}, copy=True, allow_dups=True

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _get_listlike_indexer(self, key, axis)
   1556         axis_name = self.obj._get_axis_name(axis)
   1557 
-> 1558         keyarr, indexer = ax._get_indexer_strict(key, axis_name)
   1559 
   1560         return keyarr, indexer

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: '[2486, 10002, 13032, 27891, 28373, 28839, 42337, 97838, 101885, 102938, 105051, 120227, 130460, 142550, 149769, 168218, 175352, 179311, 182341, 182346, 182993, 196990, 202499, 207267, 211455, 215662, 224160, 225249, 233874, 245696, 249649, 266485, 285659, 287638, 288960, 298412, 309769, 323637, 329010, 331597, 339882, 340533, 351584, 361793, 386734, 399785, 416989, 427602, 428108, 431819, 436658, 471472, 479052, 481419, 489767, 494480, 512494, 519532, 524834, 534751, 549210, 561786, 577725, 578135, 578919, 580338, 605427, 612853, 670254, 681342, 686022, 689250, 698287, 738404, 743726, 748552, 760662, 762802, 772737, 786490, 788466, 794694, 857263, 857721, 888472, 888596, 888904, 895361, 895400, 896067, 897211, 930680, 938020, 942215, 951810, 957590, 963989, 979151, 981032, 1004275, 1006261, 1018618, 1032448, 1054606, 1078812, 1083722, 1104435, 1106549, 1107618, 1108362, 1109000, 1132401, 1144706, 1155672, 1161016, 1164725, 1195270, 1201970, 1210867, 1220978, 1221438, 1237866, 1239525, 1249746, 1261888, 1267941, 1277772, 1278246, 1290023, 1291130, 1296011, 1341448, 1355941, 1357181, 1379748, 1380855, 1474125, 1476796, 1484329, 1484989, 1507225, 1519941, 1521628, 1530388, 1544910, 1577286, 1581706, 1614540, 1617051, 1622032, 1634444, 1634619, 1662545, 1686259, 1688692, 1709610, 1712467, 1714213, 1722272, 1729099, 1788799, 1833544, 1848497, 1865880, 1878992, 1882440, 1889603, 1902135, 1909448, 1920170, 1936801, 1950951, 1961485, 1974467] not in index'

## === cell 6
rmse = np.sqrt(mean_squared_error(y_train, oof_preds))
print(f"OOF RMSE: {rmse:.5f}")

pd.DataFrame(oof_preds, columns=["oof_pred"]).to_csv("oof_train_lgb.csv", index=False)




## === cell 7
if len(test_preds) > 0:
    final_test_pred = np.mean(test_preds, axis=0)
else:
    final_test_pred = np.full(len(X_test), y_train.mean())

submission = sample_submission.copy()
submission["fare_amount"] = final_test_pred
submission_path = "submission_lightgbm.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
