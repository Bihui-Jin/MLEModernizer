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

eli5==0.13.0
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

4.57128

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 17.57254) has done: 'Implemented a robust fix for the feature‑selection step that caused the pipeline to break.  
The code now safely handles cases where there are fewer than 10 categorical columns (or none) by selecting all available columns, ensuring `selected` is always defined. This enables downstream steps (train/validation split, model fitting, prediction, and CSV export) to run without errors and produce a valid `submission.csv`. No other logic was changed, preserving the original modeling approach.'
- What this solution (achieved 8.44315) has done: 'Implemented three key fixes: replaced the L2 Normalizer with a StandardScaler for more appropriate feature scaling, removed premature rounding of predictions during validation to obtain a true RMSE, and wrapped the optional eli5 inspection in a safe try/except that merely skips on failure. These changes keep the original model architecture intact while substantially improving validation performance, moving the RMSE toward the target score. The script now reliably writes a correctly‑formatted `submission.csv`.'
- What this solution (achieved 7.63489) has done: 'Implemented three focused fixes:   
1️⃣ Added the time‑based and passenger count features to the numerical set and scaling pipeline so the model can learn from hour, weekday, day, year and passenger count.   
2️⃣ Expanded the feature list used for training to include these new columns.   
3️⃣ Strengthened the XGBoost model by increasing trees, depth and adding early stopping on the validation split for better generalisation, which should lower the RMSE toward the target.'

# 9. Code solution

## === cell 0
import numpy as np, pandas as pd, os, math, datetime as dt
import seaborn as sns, matplotlib.pyplot as plt

print(os.listdir("/kaggle/input"))



## === cell 1
train = pd.read_csv("../input/train.csv", nrows=1000000)
test = pd.read_csv("../input/test.csv")
train.head()



## === cell 2
train.isnull().sum()



## === cell 3
train = train.dropna(how="any", axis=0)



## === cell 4
train["abs_diff_longitude"] = np.abs(
    train["dropoff_longitude"] - train["pickup_longitude"]
)
train["abs_diff_latitude"] = np.abs(
    train["dropoff_latitude"] - train["pickup_latitude"]
)
test["abs_diff_longitude"] = np.abs(
    test["dropoff_longitude"] - test["pickup_longitude"]
)
test["abs_diff_latitude"] = np.abs(test["dropoff_latitude"] - test["pickup_latitude"])



## === cell 5
train = train.loc[train["fare_amount"] > 0, :]
train = train.loc[(train["passenger_count"] <= 6) & (train["passenger_count"] > 0), :]
train = train.loc[
    (train["abs_diff_latitude"] < 2) & (train["abs_diff_longitude"] < 2), :
]
train = train.loc[
    (train["abs_diff_latitude"] > 0) & (train["abs_diff_longitude"] > 0), :
]



## === cell 6
train["timestamp_with_key"] = train["key"]
test["timestamp_with_key"] = test["key"]
train["key"] = train["key"].str.split(".").str[1].astype("int")
test["key"] = test["key"].str.split(".").str[1].astype("int")



## === cell 7
train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], infer_datetime_format=True, utc=True
)
test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], infer_datetime_format=True, utc=True
)
train["hour_no"] = train["pickup_datetime"].dt.hour.astype("int")
test["hour_no"] = test["pickup_datetime"].dt.hour.astype("int")
train["weekday_no"] = train["pickup_datetime"].dt.weekday.astype("int")
test["weekday_no"] = test["pickup_datetime"].dt.weekday.astype("int")
train["day_no"] = train["pickup_datetime"].dt.day.astype("int")
test["day_no"] = test["pickup_datetime"].dt.day.astype("int")
train["year_no"] = train["pickup_datetime"].dt.year.astype("int")
test["year_no"] = test["pickup_datetime"].dt.year.astype("int")




## === cell 8
def dist_haversine(x):
    R = 6371  # km
    picklat = math.radians(x[1])
    droplat = math.radians(x[3])
    latdiff = abs(droplat - picklat)
    picklon = math.radians(x[0])
    droplon = math.radians(x[2])
    londiff = abs(droplon - picklon)
    a = (
        math.sin(latdiff / 2) ** 2
        + math.cos(picklat) * math.cos(droplat) * math.sin(londiff / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


train["dist_haversine_km"] = pd.DataFrame(
    list(
        map(
            dist_haversine,
            train[
                [
                    "pickup_longitude",
                    "pickup_latitude",
                    "dropoff_longitude",
                    "dropoff_latitude",
                ]
            ].values,
        )
    ),
    index=train.index,
)

test["dist_haversine_km"] = pd.DataFrame(
    list(
        map(
            dist_haversine,
            test[
                [
                    "pickup_longitude",
                    "pickup_latitude",
                    "dropoff_longitude",
                    "dropoff_latitude",
                ]
            ].values,
        )
    ),
    index=test.index,
)



## === cell 9
train["fare_per_km"] = train["fare_amount"] / train["dist_haversine_km"]
train["fare_per_km_passenger"] = train["fare_amount"] / (
    train["dist_haversine_km"] * train["passenger_count"]
)



## === cell 11
orig_train = train.copy()
orig_test = test.copy()



## === cell 12
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

train = shuffle(train).reset_index(drop=True)
val = train.iloc[int(0.9 * len(train)) :, :].reset_index(drop=True)
train = train.iloc[: int(0.9 * len(train)), :].reset_index(drop=True)

cols_to_normalize = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "abs_diff_longitude",
    "abs_diff_latitude",
    "dist_haversine_km",
    "passenger_count",
    "hour_no",
    "weekday_no",
    "day_no",
    "year_no",
]

scaler = StandardScaler().fit(train[cols_to_normalize])
train[cols_to_normalize] = scaler.transform(train[cols_to_normalize])
val[cols_to_normalize] = scaler.transform(val[cols_to_normalize])
test[cols_to_normalize] = scaler.transform(test[cols_to_normalize])



## === cell 13
categorical_cols = [
    c
    for c in train.columns
    if c
    not in [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "abs_diff_longitude",
        "abs_diff_latitude",
        "dist_haversine_km",
        "passenger_count",
        "hour_no",
        "weekday_no",
        "day_no",
        "year_no",
        "key",
        "pickup_datetime",
        "timestamp_with_key",
        "fare_amount",
    ]
]

numerical_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "abs_diff_longitude",
    "abs_diff_latitude",
    "dist_haversine_km",
    "passenger_count",
    "hour_no",
    "weekday_no",
    "day_no",
    "year_no",
    "fare_per_km",
    "fare_per_km_passenger",
    "fare_amount",
]

object_cols = ["key", "pickup_datetime", "timestamp_with_key"]

cor = train[numerical_cols]
f, ax = plt.subplots(1, 1, figsize=(12, 6))
sns.heatmap(cor.corr(), annot=True, ax=ax)



## === cell 14
from sklearn.feature_selection import SelectKBest, f_regression

if len(categorical_cols) == 0:
    selected = []
else:
    k_val = min(10, len(categorical_cols))
    selector = SelectKBest(f_regression, k=k_val)
    selector.fit(train[categorical_cols], train["fare_amount"])
    selected = [
        col for col, support in zip(categorical_cols, selector.get_support()) if support
    ]



## === cell 15
train_y = np.log1p(train["fare_amount"])
val_y = np.log1p(val["fare_amount"])
feature_cols = [
    c for c in (selected + numerical_cols) if c not in object_cols + ["fare_amount"]
]
train_x = train[feature_cols]
val_x = val[feature_cols]
test_x = test[feature_cols]



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2741315093.py in <cell line: 0>()
      7 train_x = train[feature_cols]
      8 val_x = val[feature_cols]
----> 9 test_x = test[feature_cols]
     10 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

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

KeyError: "['fare_per_km', 'fare_per_km_passenger'] not in index"

## === cell 16
import xgboost as xgb
from xgboost import XGBRegressor

xgbr = XGBRegressor(
    n_estimators=1200,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    n_jobs=5,
    random_state=42,
    reg_lambda=1.0,
)

xgb_train = xgbr.fit(
    train_x,
    train_y,
    eval_set=[(val_x, val_y)],
    early_stopping_rounds=50,
    verbose=False,
)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/894074554.py in <cell line: 0>()
     14 )
     15 
---> 16 xgb_train = xgbr.fit(
     17     train_x,
     18     train_y,

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in inner_f(*args, **kwargs)
    728             for k, arg in zip(sig.parameters, args):
    729                 kwargs[k] = arg
--> 730             return func(**kwargs)
    731 
    732         return inner_f

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in fit(self, X, y, sample_weight, base_margin, eval_set, eval_metric, early_stopping_rounds, verbose, xgb_model, sample_weight_eval_set, base_margin_eval_set, feature_weights, callbacks)
   1053         with config_context(verbosity=self.verbosity):
   1054             evals_result: TrainingCallback.EvalsLog = {}
-> 1055             train_dmatrix, evals = _wrap_evaluation_matrices(
   1056                 missing=self.missing,
   1057                 X=X,

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in _wrap_evaluation_matrices(missing, X, y, group, qid, sample_weight, base_margin, feature_weights, eval_set, sample_weight_eval_set, base_margin_eval_set, eval_group, eval_qid, create_dmatrix, enable_categorical, feature_types)
    519     """Convert array_like evaluation matrices into DMatrix.  Perform validation on the
    520     way."""
--> 521     train_dmatrix = create_dmatrix(
    522         data=X,
    523         label=y,

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in _create_dmatrix(self, ref, **kwargs)
    956         if _can_use_qdm(self.tree_method) and self.booster != "gblinear":
    957             try:
--> 958                 return QuantileDMatrix(
    959                     **kwargs, ref=ref, nthread=self.n_jobs, max_bin=self.max_bin
    960                 )

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in inner_f(*args, **kwargs)
    728             for k, arg in zip(sig.parameters, args):
    729                 kwargs[k] = arg
--> 730             return func(**kwargs)
    731 
    732         return inner_f

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in __init__(self, data, label, weight, base_margin, missing, silent, feature_names, feature_types, nthread, max_bin, ref, group, qid, label_lower_bound, label_upper_bound, feature_weights, enable_categorical, data_split_mode)
   1527                 )
   1528 
-> 1529         self._init(
   1530             data,
   1531             ref=ref,

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in _init(self, data, ref, enable_categorical, **meta)
   1586             ctypes.byref(handle),
   1587         )
-> 1588         it.reraise()
   1589         # delay check_call to throw intermediate exception first
   1590         _check_call(ret)

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in reraise(self)
    574             exc = self._exception
    575             self._exception = None
--> 576             raise exc  # pylint: disable=raising-bad-type
    577 
    578     def __del__(self) -> None:

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in _handle_exception(self, fn, dft_ret)
    555 
    556         try:
--> 557             return fn()
    558         except Exception as e:  # pylint: disable=broad-except
    559             # Defer the exception in order to return 0 and stop the iteration.

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in <lambda>()
    639 
    640         # pylint: disable=not-callable
--> 641         return self._handle_exception(lambda: self.next(input_data), 0)
    642 
    643     @abstractmethod

/usr/local/lib/python3.11/dist-packages/xgboost/data.py in next(self, input_data)
   1278             return 0
   1279         self.it += 1
-> 1280         input_data(**self.kwargs)
   1281         return 1
   1282 

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in inner_f(*args, **kwargs)
    728             for k, arg in zip(sig.parameters, args):
    729                 kwargs[k] = arg
--> 730             return func(**kwargs)
    731 
    732         return inner_f

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in input_data(data, feature_names, feature_types, **kwargs)
    631             self._temporary_data = (new, cat_codes, feature_names, feature_types)
    632             dispatch_proxy_set_data(self.proxy, new, cat_codes, self._allow_host)
--> 633             self.proxy.set_info(
    634                 feature_names=feature_names,
    635                 feature_types=feature_types,

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in inner_f(*args, **kwargs)
    728             for k, arg in zip(sig.parameters, args):
    729                 kwargs[k] = arg
--> 730             return func(**kwargs)
    731 
    732         return inner_f

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in set_info(self, label, weight, base_margin, group, qid, label_lower_bound, label_upper_bound, feature_names, feature_types, feature_weights)
    944             self.set_float_info("label_upper_bound", label_upper_bound)
    945         if feature_names is not None:
--> 946             self.feature_names = feature_names
    947         if feature_types is not None:
    948             self.feature_types = feature_types

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in feature_names(self, feature_names)
   1311             )
   1312             duplicates = [name for name, cnt in zip(values, counts) if cnt > 1]
-> 1313             raise ValueError(
   1314                 f"feature_names must be unique. Duplicates found: {duplicates}"
   1315             )

ValueError: feature_names must be unique. Duplicates found: ['fare_per_km', 'fare_per_km_passenger']

## === cell 17
pred_train_log = xgbr.predict(train_x)
pred_val_log = xgbr.predict(val_x)
pred_test_log = xgbr.predict(test_x)

pred_train = np.expm1(pred_train_log)
pred_val = np.expm1(pred_val_log)
pred_test = np.expm1(pred_test_log)



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/778338350.py in <cell line: 0>()
----> 1 pred_train_log = xgbr.predict(train_x)
      2 pred_val_log = xgbr.predict(val_x)
      3 pred_test_log = xgbr.predict(test_x)
      4 
      5 # Back‑transform to original scale

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in predict(self, X, output_margin, validate_features, base_margin, iteration_range)
   1166             if self._can_use_inplace_predict():
   1167                 try:
-> 1168                     predts = self.get_booster().inplace_predict(
   1169                         data=X,
   1170                         iteration_range=iteration_range,

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in get_booster(self)
    723             from sklearn.exceptions import NotFittedError
    724 
--> 725             raise NotFittedError("need to call fit or load_model beforehand")
    726         return self._Booster
    727 

NotFittedError: need to call fit or load_model beforehand

## === cell 18
from sklearn.metrics import mean_squared_error

rmse_train = np.sqrt(mean_squared_error(train["fare_amount"], pred_train))
rmse_val = np.sqrt(mean_squared_error(val["fare_amount"], pred_val))
print("RMSE Train:", rmse_train, "RMSE Val:", rmse_val)



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1034580383.py in <cell line: 0>()
      1 from sklearn.metrics import mean_squared_error
      2 
----> 3 rmse_train = np.sqrt(mean_squared_error(train["fare_amount"], pred_train))
      4 rmse_val = np.sqrt(mean_squared_error(val["fare_amount"], pred_val))
      5 print("RMSE Train:", rmse_train, "RMSE Val:", rmse_val)

NameError: name 'pred_train' is not defined

## === cell 19
try:
    import eli5
    from eli5.sklearn import PermutationImportance

    perm = PermutationImportance(xgb_train, random_state=1).fit(val_x, val_y)
    eli5.show_weights(perm, feature_names=val_x.columns.tolist())
except Exception as e:
    print("eli5 inspection skipped:", e)



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 20
final = pd.DataFrame({"key": test["timestamp_with_key"], "fare_amount": pred_test})
final.to_csv("submission.csv", index=False)



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1800704791.py in <cell line: 0>()
----> 1 final = pd.DataFrame({"key": test["timestamp_with_key"], "fare_amount": pred_test})
      2 final.to_csv("submission.csv", index=False)
      3 

NameError: name 'pred_test' is not defined

## === cell 21
final.head()

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3254143557.py in <cell line: 0>()
----> 1 final.head()

NameError: name 'final' is not defined
