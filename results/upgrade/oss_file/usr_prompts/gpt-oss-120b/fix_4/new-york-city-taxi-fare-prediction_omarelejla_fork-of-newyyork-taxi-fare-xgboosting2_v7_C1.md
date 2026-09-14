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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
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

4.19822

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV I/O
from sklearn.ensemble import RandomForestRegressor  # (unused but kept as originally)
import matplotlib.pyplot as plt
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_iop_path = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
test_iop_path = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"

dataset_train = pd.read_csv(train_iop_path, nrows=10_000, index_col="key")
dataset_test = pd.read_csv(test_iop_path, nrows=10_000, index_col="key")



## === cell 1
print("dataset_train old size", len(dataset_train))
dataset_train = dataset_train[dataset_train.dropoff_longitude != 0]
print("new size", len(dataset_train))
dataset_train.head(5)



## === cell 2
print("dataset_test old size", len(dataset_test))
dataset_test = dataset_test[dataset_test.dropoff_longitude != 0]
print("new size", len(dataset_test))
dataset_test.head(5)




## === cell 3
def get_year(pickup_date):
    return pickup_date.year


def get_month(pickup_date):
    return pickup_date.month


def get_day(pickup_date):
    return pickup_date.day


def get_hour(pickup_date):
    return pickup_date.hour




## === cell 4
from datetime import datetime as dt
import warnings


def preparedataset2(datasetname):
    warnings.filterwarnings("ignore")
    datasetname.info()
    datasetname["pickup_datetime"] = pd.to_datetime(datasetname["pickup_datetime"])
    datasetname["pickup_year"] = datasetname.apply(
        lambda x: get_year(x["pickup_datetime"]), axis=1
    )
    datasetname["pickup_month"] = datasetname.apply(
        lambda x: get_month(x["pickup_datetime"]), axis=1
    )
    datasetname["pickup_day"] = datasetname.apply(
        lambda x: get_day(x["pickup_datetime"]), axis=1
    )
    datasetname["pickup_hour"] = datasetname.apply(
        lambda x: get_hour(x["pickup_datetime"]), axis=1
    )
    datasetname["x_dis"] = (
        datasetname["dropoff_longitude"] - datasetname["pickup_longitude"]
    )
    datasetname["y_dis"] = (
        datasetname["dropoff_latitude"] - datasetname["pickup_latitude"]
    )
    datasetname["dis"] = np.sqrt(datasetname["x_dis"] ** 2 + datasetname["y_dis"] ** 2)
    datasetname = datasetname.drop(
        [
            "pickup_datetime",
            "pickup_longitude",
            "dropoff_latitude",
            "dropoff_longitude",
            "pickup_latitude",
        ],
        axis=1,
    )
    return datasetname




## === cell 5
df = preparedataset2(dataset_train)
df.head(5)



## === cell 6
test_df = preparedataset2(dataset_test)
test_df.head(5)



## === cell 7
from sklearn.model_selection import train_test_split

y = np.log1p(df["fare_amount"])
X = df.drop("fare_amount", axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)



## === cell 8
import warnings

warnings.filterwarnings("ignore")
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

best_n_estimators = 0
best_learning_rate = 0
best_rmse = float("inf")

for lr in [X / 100 for X in range(10, 50, 5)]:  # 0.10, 0.15, ..., 0.45
    for ns in range(200, 701, 50):  # 200, 250, ..., 700
        model = XGBRegressor(
            n_estimators=ns,
            learning_rate=lr,
            n_jobs=4,
            objective="reg:squarederror",
            random_state=42,
        )
        model.fit(X_train, y_train)

        preds_log = model.predict(X_valid)
        preds = np.expm1(preds_log)
        true = np.expm1(y_valid)
        rmse = mean_squared_error(true, preds, squared=False)

        if rmse < best_rmse:
            best_rmse = rmse
            best_n_estimators = ns
            best_learning_rate = lr
            print(f"New best -> n_estimators: {ns}, lr: {lr}, RMSE: {rmse:.4f}")

print("\nBest hyper‑parameters found:")
print(
    f"n_estimators = {best_n_estimators}, learning_rate = {best_learning_rate}, RMSE = {best_rmse:.4f}"
)

final_model = XGBRegressor(
    n_estimators=best_n_estimators,
    learning_rate=best_learning_rate,
    n_jobs=4,
    objective="reg:squarederror",
    random_state=42,
)
final_model.fit(X_train, y_train)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
XGBoostError                              Traceback (most recent call last)
/tmp/ipykernel_55/1307224693.py in <cell line: 0>()
     18             random_state=42,
     19         )
---> 20         model.fit(X_train, y_train)
     21 
     22         # Predict in log‑space, then back‑transform for RMSE computation

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
    930 
    931         if label is not None:
--> 932             self.set_label(label)
    933         if weight is not None:
    934             self.set_weight(weight)

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in set_label(self, label)
   1068         from .data import dispatch_meta_backend
   1069 
-> 1070         dispatch_meta_backend(self, label, "label", "float")
   1071 
   1072     def set_weight(self, weight: ArrayLike) -> None:

/usr/local/lib/python3.11/dist-packages/xgboost/data.py in dispatch_meta_backend(matrix, data, name, dtype)
   1223         return
   1224     if _is_pandas_series(data):
-> 1225         _meta_from_pandas_series(data, name, dtype, handle)
   1226         return
   1227     if _is_dlpack(data):

/usr/local/lib/python3.11/dist-packages/xgboost/data.py in _meta_from_pandas_series(data, name, dtype, handle)
    543         data = data.to_dense()  # type: ignore
    544     assert len(data.shape) == 1 or data.shape[1] == 0 or data.shape[1] == 1
--> 545     _meta_from_numpy(data, name, dtype, handle)
    546 
    547 

/usr/local/lib/python3.11/dist-packages/xgboost/data.py in _meta_from_numpy(data, field, dtype, handle)
   1157         raise ValueError("Masked array is not supported.")
   1158     interface_str = _array_interface(data)
-> 1159     _check_call(_LIB.XGDMatrixSetInfoFromInterface(handle, c_str(field), interface_str))
   1160 
   1161 

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in _check_call(ret)
    280     """
    281     if ret != 0:
--> 282         raise XGBoostError(py_str(_LIB.XGBGetLastError()))
    283 
    284 

XGBoostError: [00:46:05] /workspace/src/data/data.cc:507: Check failed: valid: Label contains NaN, infinity or a value too large.
Stack trace:
  [bt] (0) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x3588ca) [0x7fca2f77a8ca]
  [bt] (1) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x38a21d) [0x7fca2f7ac21d]
  [bt] (2) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x38ab51) [0x7fca2f7acb51]
  [bt] (3) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(XGDMatrixSetInfoFromInterface+0xb0) [0x7fca2f5803a0]
  [bt] (4) /lib/x86_64-linux-gnu/libffi.so.8(+0x7e2e) [0x7fcab9f3be2e]
  [bt] (5) /lib/x86_64-linux-gnu/libffi.so.8(+0x4493) [0x7fcab9f38493]
  [bt] (6) /usr/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so(+0xa4d8) [0x7fcab8db94d8]
  [bt] (7) /usr/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so(+0x9c8e) [0x7fcab8db8c8e]
  [bt] (8) /usr/bin/python3(_PyObject_MakeTpCall+0x27c) [0x52f85c]



## === cell 9
valid_preds_log = final_model.predict(X_valid)
valid_preds = np.expm1(valid_preds_log)
valid_true = np.expm1(y_valid)
rmse = mean_squared_error(valid_true, valid_preds, squared=False)
print(f"Validation RMSE: {rmse:.4f}")

test_preds_log = final_model.predict(test_df)
test_preds = np.expm1(test_preds_log)

submission = pd.DataFrame({"key": test_df.index, "fare_amount": test_preds})

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2973162659.py in <cell line: 0>()
      1 # Evaluate on validation set using back‑transformed values
----> 2 valid_preds_log = final_model.predict(X_valid)
      3 valid_preds = np.expm1(valid_preds_log)
      4 valid_true = np.expm1(y_valid)
      5 rmse = mean_squared_error(valid_true, valid_preds, squared=False)

NameError: name 'final_model' is not defined
