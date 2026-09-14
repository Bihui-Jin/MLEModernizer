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
Given simulated manufacturing control data, predict whether the machine is in state `0` or state `1`.

## Metric
Area under the ROC curve.

## Submission Format
For each `id` in the test set, you must predict a probability for the `target` variable. The file should contain a header and have the following format:

```
id,target
900000,0.65
900001,0.97
900002,0.02
etc.
```

## Dataset
- **train.csv** - the training data, which includes normalized continuous data and categorical data
- **test.csv** - the test set; your task is to predict binary `target` variable which represents the state of a manufacturing process
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.10

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
            description.md (79 lines)
            sample_submission.csv (100001 lines)
            sample_submission.csv.zip (224.9 kB)
            test.csv (100001 lines)
            test.csv.zip (16.6 MB)
            train.csv (800001 lines)
            train.csv.zip (133.3 MB)
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
        input/
            description.md (79 lines)
            sample_submission.csv (100001 lines)
            sample_submission.csv.zip (224.9 kB)
            test.csv (100001 lines)
            test.csv.zip (16.6 MB)
            train.csv (800001 lines)
            train.csv.zip (133.3 MB)
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
        working/
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
```

-> data/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> data/tabular-playground-series-may-2022/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> data/tabular-playground-series-may-2022/test.csv has 100000 rows and 32 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 17 more columns

-> data/tabular-playground-series-may-2022/train.csv has 800000 rows and 33 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 18 more columns

-> data/test.csv has 100000 rows and 32 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 17 more columns

-> data/train.csv has 800000 rows and 33 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 18 more columns

-> input/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> (stopped after 10 files for performance)

# 5. Target score

0.92974

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV I/O
import os, warnings

warnings.filterwarnings("ignore")
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 1
pd.options.display.float_format = "{:,.2f}".format
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 100)




## === cell 2
train_path = "/kaggle/input/tabular-playground-series-may-2022/train.csv"
test_path = "/kaggle/input/tabular-playground-series-may-2022/test.csv"
sub_path = "/kaggle/input/tabular-playground-series-may-2022/sample_submission.csv"

trn_data = pd.read_csv(train_path)
tst_data = pd.read_csv(test_path)
sub = pd.read_csv(sub_path)




## === cell 3
print("Train shape:", trn_data.shape)
print("Test shape :", tst_data.shape)
print("Sample submission head:")
print(sub.head())




## === cell 4
from sklearn.preprocessing import LabelEncoder

cat_cols = trn_data.select_dtypes(include=["object"]).columns.tolist()
encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    trn_data[col + "_enc"] = le.fit_transform(trn_data[col].astype(str))
    tst_data[col + "_enc"] = le.transform(tst_data[col].astype(str))
    encoders[col] = le

trn_data.drop(columns=cat_cols, inplace=True)
tst_data.drop(columns=cat_cols, inplace=True)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/sklearn/utils/_encode.py in _encode(values, uniques, check_unknown)
    223         try:
--> 224             return _map_to_integer(values, uniques)
    225         except KeyError as e:

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_encode.py in _map_to_integer(values, uniques)
    163     table = _nandict({val: i for i, val in enumerate(uniques)})
--> 164     return np.array([table[v] for v in values])
    165 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_encode.py in <listcomp>(.0)
    163     table = _nandict({val: i for i, val in enumerate(uniques)})
--> 164     return np.array([table[v] for v in values])
    165 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_encode.py in __missing__(self, key)
    157             return self.nan_value
--> 158         raise KeyError(key)
    159 

KeyError: 'AABBBBDHCC'

During handling of the above exception, another exception occurred:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3362194634.py in <cell line: 0>()
     10     trn_data[col + "_enc"] = le.fit_transform(trn_data[col].astype(str))
     11     # transform test data using the same encoder
---> 12     tst_data[col + "_enc"] = le.transform(tst_data[col].astype(str))
     13     encoders[col] = le
     14 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py in wrapped(self, X, *args, **kwargs)
    138     @wraps(f)
    139     def wrapped(self, X, *args, **kwargs):
--> 140         data_to_wrap = f(self, X, *args, **kwargs)
    141         if isinstance(data_to_wrap, tuple):
    142             # only wrap the first output for cross decomposition

/usr/local/lib/python3.11/dist-packages/sklearn/preprocessing/_label.py in transform(self, y)
    137             return np.array([])
    138 
--> 139         return _encode(y, uniques=self.classes_)
    140 
    141     def inverse_transform(self, y):

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_encode.py in _encode(values, uniques, check_unknown)
    224             return _map_to_integer(values, uniques)
    225         except KeyError as e:
--> 226             raise ValueError(f"y contains previously unseen labels: {str(e)}")
    227     else:
    228         if check_unknown:

ValueError: y contains previously unseen labels: 'AABBBBDHCC'

## === cell 5
ignore = ["id", "target"]  # columns not used as features
features = [c for c in trn_data.columns if c not in ignore]
target_feature = "target"




## === cell 6
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(
    trn_data[features],
    trn_data[target_feature],
    test_size=0.10,
    random_state=42,
    stratify=trn_data[target_feature],
)




## === cell 7
from xgboost import XGBClassifier

params = {
    "n_estimators": 4096,
    "max_depth": 8,
    "learning_rate": 0.15,
    "subsample": 0.95,
    "colsample_bytree": 0.95,
    "reg_lambda": 1.5,
    "reg_alpha": 1.5,
    "gamma": 1.5,
    "random_state": 46,
    "objective": "binary:logistic",
    "tree_method": "hist",  # CPU‑friendly
    "eval_metric": "auc",
}

xgb = XGBClassifier(**params)
xgb.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    early_stopping_rounds=128,
    verbose=False,
)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/221356135.py in <cell line: 0>()
     18 
     19 xgb = XGBClassifier(**params)
---> 20 xgb.fit(
     21     X_train,
     22     y_train,

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in inner_f(*args, **kwargs)
    728             for k, arg in zip(sig.parameters, args):
    729                 kwargs[k] = arg
--> 730             return func(**kwargs)
    731 
    732         return inner_f

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in fit(self, X, y, sample_weight, base_margin, eval_set, eval_metric, early_stopping_rounds, verbose, xgb_model, sample_weight_eval_set, base_margin_eval_set, feature_weights, callbacks)
   1498                 xgb_model, eval_metric, params, early_stopping_rounds, callbacks
   1499             )
-> 1500             train_dmatrix, evals = _wrap_evaluation_matrices(
   1501                 missing=self.missing,
   1502                 X=X,

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
    622                 new, cat_codes, feature_names, feature_types = self._temporary_data
    623             else:
--> 624                 new, cat_codes, feature_names, feature_types = _proxy_transform(
    625                     data,
    626                     feature_names,

/usr/local/lib/python3.11/dist-packages/xgboost/data.py in _proxy_transform(data, feature_names, feature_types, enable_categorical)
   1313         data = pd.DataFrame(data)
   1314     if _is_pandas_df(data):
-> 1315         arr, feature_names, feature_types = _transform_pandas_df(
   1316             data, enable_categorical, feature_names, feature_types
   1317         )

/usr/local/lib/python3.11/dist-packages/xgboost/data.py in _transform_pandas_df(data, enable_categorical, feature_names, feature_types, meta, meta_type)
    488             or is_pa_ext_dtype(dtype)
    489         ):
--> 490             _invalid_dataframe_dtype(data)
    491         if is_pa_ext_dtype(dtype):
    492             pyarrow_extension = True

/usr/local/lib/python3.11/dist-packages/xgboost/data.py in _invalid_dataframe_dtype(data)
    306     type_err = "DataFrame.dtypes for data must be int, float, bool or category."
    307     msg = f"""{type_err} {_ENABLE_CAT_ERR} {err}"""
--> 308     raise ValueError(msg)
    309 
    310 

ValueError: DataFrame.dtypes for data must be int, float, bool or category. When categorical type is supplied, The experimental DMatrix parameter`enable_categorical` must be set to `True`.  Invalid columns:f_27: object

## === cell 8
import seaborn as sns
import matplotlib.pyplot as plt


def plot_feature_importance(importance, names, model_type, max_features=10):
    fi_df = (
        pd.DataFrame({"feature_names": names, "feature_importance": importance})
        .sort_values("feature_importance", ascending=False)
        .head(max_features)
    )

    plt.figure(figsize=(8, 6))
    sns.barplot(x="feature_importance", y="feature_names", data=fi_df)
    plt.title(f"{model_type} Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()


plot_feature_importance(
    xgb.feature_importances_, X_train.columns, "XGBoost", max_features=15
)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/2057049753.py in <cell line: 0>()
     21 
     22 plot_feature_importance(
---> 23     xgb.feature_importances_, X_train.columns, "XGBoost", max_features=15
     24 )
     25 

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in feature_importances_(self)
   1312 
   1313         """
-> 1314         b: Booster = self.get_booster()
   1315 
   1316         def dft() -> str:

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in get_booster(self)
    723             from sklearn.exceptions import NotFittedError
    724 
--> 725             raise NotFittedError("need to call fit or load_model beforehand")
    726         return self._Booster
    727 

NotFittedError: need to call fit or load_model beforehand

## === cell 9
preds = xgb.predict_proba(tst_data[features])[:, 1]
sub["target"] = preds
submission_path = "my_submission_043022.csv"
sub.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2094020507.py in <cell line: 0>()
      1 # Predict on test set and create submission file
----> 2 preds = xgb.predict_proba(tst_data[features])[:, 1]
      3 sub["target"] = preds
      4 submission_path = "my_submission_043022.csv"
      5 sub.to_csv(submission_path, index=False)

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

KeyError: "['f_27_enc'] not in index"
