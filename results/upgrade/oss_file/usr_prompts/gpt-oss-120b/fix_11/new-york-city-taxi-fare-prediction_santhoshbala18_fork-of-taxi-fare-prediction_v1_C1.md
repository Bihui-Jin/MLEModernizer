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

3.55848

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 6.22767) has done: 'I expand the file‑search to include the usual Kaggle input directory, fix the airport‑coordinate check (the longitudes are negative in this dataset), and tidy the flow so every variable is defined before it’s used. These minimal changes resolve the “file not found” and subsequent NameError issues, and the corrected features should bring the RMSE toward the target while preserving the original model logic.'
- What this solution (achieved 10.04272) has done: 'I increase the amount of training data (from 500 k to 2 M rows) and keep a larger proportion for training (validation size 5 %). I also cast the computed distance feature to float32 to stay memory‑efficient. These small adjustments keep the original model unchanged while giving it more data and a slightly stronger training‑set, which should lower the RMSE toward the target.'
- What this solution (achieved 5.96011) has done: 'I add modest data‑cleaning (remove extreme fares and zero‑distance trips) and train the XGBoost model on a log‑transformed target, then invert the predictions for evaluation and submission. These tweaks keep the original model and features while addressing skew and outliers, which should lower the RMSE toward the target without altering the overall pipeline.'
- What this solution (achieved 5.4066) has done: 'I add a few cheap engineered features (latitude/longitude differences and a rough Manhattan distance) that often help fare prediction, and I slightly adjust the XGBoost hyper‑parameters (shallower trees, lower learning rate, more estimators and a regularisation term) to improve generalisation. These changes keep the original pipeline and model untouched while providing the model with extra useful information, which should lower the RMSE and move the score toward the target.'

# 9. Code solution

## === cell 0
import os
import math
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


def locate_file(filename: str) -> Path:
    for p in Path(".").rglob(filename):
        return p
    raise FileNotFoundError(f"{filename} not found in the current directory tree.")


TRAIN_PATH = locate_file("train.csv")
TEST_PATH = locate_file("test.csv")
SAMPLE_SUBMISSION_PATH = locate_file("sample_submission.csv")



## === cell 1
dtype_map = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}

df = pd.read_csv(
    TRAIN_PATH,
    dtype=dtype_map,
    usecols=list(dtype_map.keys()) + ["key", "pickup_datetime"],
    nrows=5_000_000,
)

test_set = pd.read_csv(
    TEST_PATH,
    dtype={k: v for k, v in dtype_map.items() if k != "fare_amount"},
    usecols=[
        c
        for c in ["key", "pickup_datetime"] + list(dtype_map.keys())
        if c != "fare_amount"
    ],
)



## === cell 2
X = df.drop(["key", "fare_amount", "pickup_datetime"], axis=1)
y = df["fare_amount"]

y_log = np.log1p(y)

X_train, X_val, y_train_log, y_val = train_test_split(
    X, y_log, test_size=0.05, random_state=42
)

regressor = XGBRegressor(
    max_depth=10,
    learning_rate=0.03,
    n_estimators=1200,
    objective="reg:squarederror",
    n_jobs=4,
    random_state=42,
    tree_method="hist",
    eval_metric="rmse",
    reg_lambda=1.5,
)

regressor.fit(X_train, y_train_log)

val_pred_log = regressor.predict(X_val)
val_pred = np.expm1(val_pred_log)
rmse = math.sqrt(mean_squared_error(np.expm1(y_val), val_pred))
print(f"Validation RMSE: {rmse:.5f}")



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
XGBoostError                              Traceback (most recent call last)
/tmp/ipykernel_12/1066525353.py in <cell line: 0>()
     25 
     26 # Train on log‑target
---> 27 regressor.fit(X_train, y_train_log)
     28 
     29 # Validation predictions and RMSE

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

XGBoostError: [23:38:06] /workspace/src/data/data.cc:507: Check failed: valid: Label contains NaN, infinity or a value too large.
Stack trace:
  [bt] (0) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x3588ca) [0x7fff8386c8ca]
  [bt] (1) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x38a21d) [0x7fff8389e21d]
  [bt] (2) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x38ab51) [0x7fff8389eb51]
  [bt] (3) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(XGDMatrixSetInfoFromInterface+0xb0) [0x7fff836723a0]
  [bt] (4) /lib/x86_64-linux-gnu/libffi.so.8(+0x7e2e) [0x7ffff63ace2e]
  [bt] (5) /lib/x86_64-linux-gnu/libffi.so.8(+0x4493) [0x7ffff63a9493]
  [bt] (6) /usr/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so(+0xa4d8) [0x7ffff63bc4d8]
  [bt] (7) /usr/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so(+0x9c8e) [0x7ffff63bbc8e]
  [bt] (8) /usr/bin/python3(_PyObject_MakeTpCall+0x27c) [0x52f85c]



## === cell 3
X_test = test_set.drop(["key", "pickup_datetime"], axis=1)

test_pred_log = regressor.predict(X_test)
test_pred = np.expm1(test_pred_log)

submission = pd.DataFrame(
    {"key": test_set["key"], "fare_amount": test_pred.astype("float32")}
)

submission_path = Path("submission.csv")
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path.resolve()}")

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_12/2582774704.py in <cell line: 0>()
      3 
      4 # Predict and invert log‑transform
----> 5 test_pred_log = regressor.predict(X_test)
      6 test_pred = np.expm1(test_pred_log)
      7 

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
