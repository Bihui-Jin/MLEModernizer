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

0.95972

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os, warnings

warnings.filterwarnings("ignore")

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
pd.options.display.float_format = "{:,.2f}".format
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_rows", 10)



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



## === cell 4
numeric_corr = trn_data.select_dtypes(include=[np.number]).corr()
print("Top correlations with target:")
print(numeric_corr["target"].sort_values(ascending=False).head())



## === cell 5
letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def count_sequence(df, field):
    for letter in letters:
        df[f"{letter}_count"] = df[field].str.count(letter)
    return df


trn_data = count_sequence(trn_data, "f_27")
tst_data = count_sequence(tst_data, "f_27")



## === cell 6
from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()


def encode_train(df, col):
    df[col + "_enc"] = enc.fit_transform(df[col])
    return df


def encode_test(df, col):
    df[col + "_enc"] = enc.transform(df[col])
    return df


trn_data = encode_train(trn_data, "f_27")
tst_data = encode_test(tst_data, "f_27")



## --- ERROR in cell 6, traceback:
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
/tmp/ipykernel_11/3660714580.py in <cell line: 0>()
     16 
     17 trn_data = encode_train(trn_data, "f_27")
---> 18 tst_data = encode_test(tst_data, "f_27")
     19 

/tmp/ipykernel_11/3660714580.py in encode_test(df, col)
     11 
     12 def encode_test(df, col):
---> 13     df[col + "_enc"] = enc.transform(df[col])
     14     return df
     15 

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

## === cell 7
ignore = ["id", "target", "f_27"]  # drop original string column & ID/target
features = [c for c in trn_data.columns if c not in ignore]
target_feature = "target"



## === cell 8
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(
    trn_data[features],
    trn_data[target_feature],
    test_size=0.20,
    random_state=42,
    stratify=trn_data[target_feature],
)



## === cell 9
from xgboost import XGBClassifier



## === cell 10
params = {
    "n_estimators": 4096,
    "max_depth": 6,
    "learning_rate": 0.15,
    "subsample": 0.95,
    "colsample_bytree": 0.95,
    "reg_lambda": 1.5,
    "reg_alpha": 1.5,
    "gamma": 1.5,
    "random_state": 46,
    "objective": "binary:logistic",
    "tree_method": "hist",  # CPU friendly
}



## === cell 11
xgb = XGBClassifier(**params)
xgb.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="auc",
    early_stopping_rounds=256,
    verbose=250,
)



## === cell 12
from sklearn.metrics import roc_auc_score

val_preds = xgb.predict_proba(X_valid)[:, 1]
val_auc = roc_auc_score(y_valid, val_preds)
print(f"Validation AUC: {val_auc:.5f}")



## === cell 13
import seaborn as sns
import matplotlib.pyplot as plt


def plot_feature_importance(importances, names, model_type, max_features=15):
    fi_df = pd.DataFrame({"feature_name": names, "importance": importances})
    fi_df = fi_df.sort_values("importance", ascending=False).head(max_features)
    plt.figure(figsize=(8, 6))
    sns.barplot(x="importance", y="feature_name", data=fi_df)
    plt.title(f"{model_type} FEATURE IMPORTANCE")
    plt.tight_layout()
    plt.show()


plot_feature_importance(xgb.feature_importances_, X_train.columns, "XGBoost")



## === cell 14
test_preds = xgb.predict_proba(tst_data[features])[:, 1]
sub["target"] = test_preds
submission_path = "/kaggle/working/submission.csv"
sub.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1379165961.py in <cell line: 0>()
      1 # Predict on test set and create submission
----> 2 test_preds = xgb.predict_proba(tst_data[features])[:, 1]
      3 sub["target"] = test_preds
      4 submission_path = "/kaggle/working/submission.csv"
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

## === cell 15
sub.head()
