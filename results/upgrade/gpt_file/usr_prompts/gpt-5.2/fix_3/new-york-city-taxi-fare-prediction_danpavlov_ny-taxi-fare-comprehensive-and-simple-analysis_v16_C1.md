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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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

3.623

# 6. Current score

6.51135

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 7.33786) has done: 'I fix the XGBoost inference crash caused by `best_ntree_limit` no longer existing in xgboost 2.x by switching to the supported `iteration_range` API while keeping the same training approach (early stopping on a held-out set). I also remove notebook-only `%matplotlib inline` so the script runs as a plain Python program, and replace deprecated `sns.distplot` with `sns.histplot` to avoid runtime errors on newer seaborn. Finally, I ensure we always write a valid submission CSV with the required columns (`key,fare_amount`) and a `.csv` suffix.'
- What this solution (achieved 6.51135) has done: 'You’re far from the target (7.34 vs 3.623 RMSE; lower is better), so we need small, legitimate changes that materially improve generalization without changing the overall approach (same feature set idea + XGBoost training + early stopping). The biggest score drag is the extremely slow per-row `iterrows()` distance engineering: it limits you to only 100k rows and also risks inconsistent feature creation; vectorizing the exact same Haversine math lets you train on more rows within the same time budget, which should move RMSE down toward your target. I also keep the same model objective and early-stopping workflow, but add a fixed `random_state` and a slightly stronger parameter set (depth/eta/subsampling) that is standard for this competition and should improve RMSE without changing the “XGBoost regressor with early stopping” core logic. Finally, I stop rounding predictions (rounding hurts RMSE) and keep the submission format identical.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn import metrics  # evaluating models
from sklearn.model_selection import train_test_split  # set splitting and validation
from sklearn.linear_model import LinearRegression
import xgboost as xgb  # XGBoost
import matplotlib.pyplot as plt  # plotting
import seaborn as sns  # plotting



## === cell 1
print(os.listdir("../input"))



## === cell 2
test = pd.read_csv("../input/test.csv")



## === cell 3
test.dtypes



## === cell 4
types = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}



## === cell 5
train = pd.read_csv("../input/train.csv", nrows=2_000_000, dtype=types)



## === cell 6
train.head()



## === cell 7
train.describe()



## === cell 8
sns.histplot(train["fare_amount"], kde=True)



## === cell 9
sns.histplot(train["passenger_count"], kde=False)



## === cell 10
train.isnull().sum()



## === cell 11
train.dropna(inplace=True)



## === cell 12
train = train[train["fare_amount"] > 0]
train = train[train["pickup_longitude"] < -72]
train = train[(train["pickup_latitude"] > 40) & (train["pickup_latitude"] < 44)]
train = train[train["dropoff_longitude"] < -72]
train = train[(train["dropoff_latitude"] > 40) & (train["dropoff_latitude"] < 44)]
train = train[(train["passenger_count"] > 0) & (train["passenger_count"] < 10)]



## === cell 13
train.describe()




## === cell 14
def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6373.0
    lat1 = np.radians(lat1.astype("float64"))
    lon1 = np.radians(lon1.astype("float64"))
    lat2 = np.radians(lat2.astype("float64"))
    lon2 = np.radians(lon2.astype("float64"))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


def quick_dist_calc(df):
    df["distance"] = _haversine_km(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    ).astype("float32")




## === cell 15
def quick_dist_calc_loc(df, c1, c2, cname):
    df[cname + "_pickup_dist"] = _haversine_km(
        df["pickup_latitude"], df["pickup_longitude"], c1, c2
    ).astype("float32")
    df[cname + "_dropoff_dist"] = _haversine_km(
        df["dropoff_latitude"], df["dropoff_longitude"], c1, c2
    ).astype("float32")




## === cell 16
quick_dist_calc(train)
quick_dist_calc(test)



## === cell 17
jfk_airport = (-73.785193, 40.645972)
laguardia_airport = (-73.872925, 40.773335)
newark_airport = (-74.184156, 40.692764)
manhattan = (-73.983132, 40.759006)

quick_dist_calc_loc(train, jfk_airport[1], jfk_airport[0], "jfk_airport")
quick_dist_calc_loc(
    train, laguardia_airport[1], laguardia_airport[0], "laguardia_airport"
)
quick_dist_calc_loc(train, newark_airport[1], newark_airport[0], "newark_airport")
quick_dist_calc_loc(train, manhattan[1], manhattan[0], "manhattan")

quick_dist_calc_loc(test, jfk_airport[1], jfk_airport[0], "jfk_airport")
quick_dist_calc_loc(
    test, laguardia_airport[1], laguardia_airport[0], "laguardia_airport"
)
quick_dist_calc_loc(test, newark_airport[1], newark_airport[0], "newark_airport")
quick_dist_calc_loc(test, manhattan[1], manhattan[0], "manhattan")



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3353574106.py in <cell line: 0>()
      4 manhattan = (-73.983132, 40.759006)
      5 
----> 6 quick_dist_calc_loc(train, jfk_airport[1], jfk_airport[0], "jfk_airport")
      7 quick_dist_calc_loc(
      8     train, laguardia_airport[1], laguardia_airport[0], "laguardia_airport"

/tmp/ipykernel_11/1769715196.py in quick_dist_calc_loc(df, c1, c2, cname)
      1 def quick_dist_calc_loc(df, c1, c2, cname):
      2     # c1=lat, c2=lon as in original call sites
----> 3     df[cname + "_pickup_dist"] = _haversine_km(
      4         df["pickup_latitude"], df["pickup_longitude"], c1, c2
      5     ).astype("float32")

/tmp/ipykernel_11/987865667.py in _haversine_km(lat1, lon1, lat2, lon2)
      5     lat1 = np.radians(lat1.astype("float64"))
      6     lon1 = np.radians(lon1.astype("float64"))
----> 7     lat2 = np.radians(lat2.astype("float64"))
      8     lon2 = np.radians(lon2.astype("float64"))
      9     dlon = lon2 - lon1

AttributeError: 'float' object has no attribute 'astype'

## === cell 18
train["jfk_distance"] = pd.concat(
    [train["jfk_airport_pickup_dist"], train["jfk_airport_dropoff_dist"]], axis=1
).min(axis=1)
train["laguardia_distance"] = pd.concat(
    [train["laguardia_airport_pickup_dist"], train["laguardia_airport_dropoff_dist"]],
    axis=1,
).min(axis=1)
train["newark_distance"] = pd.concat(
    [train["newark_airport_pickup_dist"], train["newark_airport_dropoff_dist"]], axis=1
).min(axis=1)
train["manhattan_distance"] = pd.concat(
    [train["manhattan_pickup_dist"], train["manhattan_dropoff_dist"]], axis=1
).min(axis=1)

test["jfk_distance"] = pd.concat(
    [test["jfk_airport_pickup_dist"], test["jfk_airport_dropoff_dist"]], axis=1
).min(axis=1)
test["laguardia_distance"] = pd.concat(
    [test["laguardia_airport_pickup_dist"], test["laguardia_airport_dropoff_dist"]],
    axis=1,
).min(axis=1)
test["newark_distance"] = pd.concat(
    [test["newark_airport_pickup_dist"], test["newark_airport_dropoff_dist"]], axis=1
).min(axis=1)
test["manhattan_distance"] = pd.concat(
    [test["manhattan_pickup_dist"], test["manhattan_dropoff_dist"]], axis=1
).min(axis=1)



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'jfk_airport_pickup_dist'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3398218307.py in <cell line: 0>()
      1 train["jfk_distance"] = pd.concat(
----> 2     [train["jfk_airport_pickup_dist"], train["jfk_airport_dropoff_dist"]], axis=1
      3 ).min(axis=1)
      4 train["laguardia_distance"] = pd.concat(
      5     [train["laguardia_airport_pickup_dist"], train["laguardia_airport_dropoff_dist"]],

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'jfk_airport_pickup_dist'

## === cell 19
train.drop("jfk_airport_pickup_dist", inplace=True, axis=1)
train.drop("jfk_airport_dropoff_dist", inplace=True, axis=1)
train.drop("laguardia_airport_pickup_dist", inplace=True, axis=1)
train.drop("laguardia_airport_dropoff_dist", inplace=True, axis=1)
train.drop("newark_airport_pickup_dist", inplace=True, axis=1)
train.drop("newark_airport_dropoff_dist", inplace=True, axis=1)
train.drop("manhattan_pickup_dist", inplace=True, axis=1)
train.drop("manhattan_dropoff_dist", inplace=True, axis=1)

test.drop("jfk_airport_pickup_dist", inplace=True, axis=1)
test.drop("jfk_airport_dropoff_dist", inplace=True, axis=1)
test.drop("laguardia_airport_pickup_dist", inplace=True, axis=1)
test.drop("laguardia_airport_dropoff_dist", inplace=True, axis=1)
test.drop("newark_airport_pickup_dist", inplace=True, axis=1)
test.drop("newark_airport_dropoff_dist", inplace=True, axis=1)
test.drop("manhattan_pickup_dist", inplace=True, axis=1)
test.drop("manhattan_dropoff_dist", inplace=True, axis=1)



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1339159034.py in <cell line: 0>()
----> 1 train.drop("jfk_airport_pickup_dist", inplace=True, axis=1)
      2 train.drop("jfk_airport_dropoff_dist", inplace=True, axis=1)
      3 train.drop("laguardia_airport_pickup_dist", inplace=True, axis=1)
      4 train.drop("laguardia_airport_dropoff_dist", inplace=True, axis=1)
      5 train.drop("newark_airport_pickup_dist", inplace=True, axis=1)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in drop(self, labels, axis, index, columns, level, inplace, errors)
   5579                 weight  1.0     0.8
   5580         """
-> 5581         return super().drop(
   5582             labels=labels,
   5583             axis=axis,

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in drop(self, labels, axis, index, columns, level, inplace, errors)
   4786         for axis, labels in axes.items():
   4787             if labels is not None:
-> 4788                 obj = obj._drop_axis(labels, axis, level=level, errors=errors)
   4789 
   4790         if inplace:

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in _drop_axis(self, labels, axis, level, errors, only_slice)
   4828                 new_axis = axis.drop(labels, level=level, errors=errors)
   4829             else:
-> 4830                 new_axis = axis.drop(labels, errors=errors)
   4831             indexer = axis.get_indexer(new_axis)
   4832 

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in drop(self, labels, errors)
   7068         if mask.any():
   7069             if errors != "ignore":
-> 7070                 raise KeyError(f"{labels[mask].tolist()} not found in axis")
   7071             indexer = indexer[~mask]
   7072         return self.delete(indexer)

KeyError: "['jfk_airport_pickup_dist'] not found in axis"

## === cell 20
train["pickup_datetime"] = train["pickup_datetime"].str.replace(" UTC", "", regex=False)
train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], format="%Y-%m-%d %H:%M:%S"
)

test["pickup_datetime"] = test["pickup_datetime"].str.replace(" UTC", "", regex=False)
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], format="%Y-%m-%d %H:%M:%S"
)



## === cell 21
train["hour"] = train.pickup_datetime.dt.hour
train["weekday"] = train.pickup_datetime.dt.weekday
train["month"] = train.pickup_datetime.dt.month
train["year"] = train.pickup_datetime.dt.year

test["hour"] = test.pickup_datetime.dt.hour
test["weekday"] = test.pickup_datetime.dt.weekday
test["month"] = test.pickup_datetime.dt.month
test["year"] = test.pickup_datetime.dt.year



## === cell 22
train.head()



## === cell 23
test.head()



## === cell 24
plt.figure(figsize=(20, 12))
sns.heatmap(
    train.drop(["key", "pickup_datetime"], axis=1).corr(), annot=True, fmt=".4f"
)



## === cell 25
X = train.drop(["key", "fare_amount", "pickup_datetime"], axis=1)
y = train["fare_amount"]



## === cell 26
X.head()



## === cell 27
y.head()



## === cell 28
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



## === cell 29
test_pred = test.drop(["key", "pickup_datetime"], axis=1)



## === cell 30
lm = LinearRegression()
lm.fit(X_train, y_train)
print(lm.score(X_train, y_train))
print(lm.score(X_test, y_test))



## === cell 31
y_pred = lm.predict(X)
lrmse = np.sqrt(metrics.mean_squared_error(y_pred, y))
lrmse



## === cell 32
LinearPredictions = lm.predict(test_pred)



## === cell 33
LinearPredictions.size



## === cell 34
linear_submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": LinearPredictions},
    columns=["key", "fare_amount"],
)



## === cell 35
linear_submission.head()




## === cell 36
def XGBoost(X_train, X_test, y_train, y_test):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.1,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1.0,
        "gamma": 0.0,
        "lambda": 1.0,
        "alpha": 0.0,
        "seed": 42,
        "tree_method": "hist",
    }

    return xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=2000,
        early_stopping_rounds=50,
        evals=[(dtest, "test")],
        verbose_eval=False,
    )




## === cell 37
xgbm = XGBoost(X_train, X_test, y_train, y_test)

dtest_submit = xgb.DMatrix(test_pred)
if getattr(xgbm, "best_iteration", None) is not None:
    XGBPredictions = xgbm.predict(
        dtest_submit, iteration_range=(0, xgbm.best_iteration + 1)
    )
else:
    XGBPredictions = xgbm.predict(dtest_submit)



## === cell 38
XGBPredictions



## === cell 39
XGBPredictions = np.clip(XGBPredictions, 0.0, None)



## === cell 40
XGB_submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": XGBPredictions}, columns=["key", "fare_amount"]
)
XGB_submission.head()



## === cell 41
submission = XGB_submission



## === cell 42
submission.to_csv("XGBSubmission23082018_2M.csv", index=False)
print(
    "Wrote submission:",
    os.path.abspath("XGBSubmission23082018_2M.csv"),
    "rows:",
    len(submission),
)
