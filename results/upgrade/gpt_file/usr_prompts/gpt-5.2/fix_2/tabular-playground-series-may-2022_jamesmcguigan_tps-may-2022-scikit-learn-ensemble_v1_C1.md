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

0.65336

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
import sklearn
from operator import itemgetter

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import warnings

warnings.filterwarnings("ignore")



## === cell 1
col_dtypes_train = {
    "f_00": "float32",
    "f_01": "float32",
    "f_02": "float32",
    "f_03": "float32",
    "f_04": "float32",
    "f_05": "float32",
    "f_06": "float32",
    "f_07": "int32",
    "f_08": "int32",
    "f_09": "int32",
    "f_10": "int32",
    "f_11": "int32",
    "f_12": "int32",
    "f_13": "int32",
    "f_14": "int32",
    "f_15": "int32",
    "f_16": "int32",
    "f_17": "int32",
    "f_18": "int32",
    "f_19": "float32",
    "f_20": "float32",
    "f_21": "float32",
    "f_22": "float32",
    "f_23": "float32",
    "f_24": "float32",
    "f_25": "float32",
    "f_26": "float32",
    "f_27": "category",
    "f_28": "float32",
    "f_29": "int32",
    "f_30": "int32",
    "target": "int32",
}
col_dtypes_test = {k: v for k, v in col_dtypes_train.items() if k != "target"}


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    if "f_27" in df.columns:
        del df["f_27"]
    return df


train_path = "../input/tabular-playground-series-may-2022/train.csv"
test_path = "../input/tabular-playground-series-may-2022/test.csv"
sub_path = "../input/tabular-playground-series-may-2022/sample_submission.csv"

train_df = pd.read_csv(train_path, index_col="id", dtype=col_dtypes_train)
test_df = pd.read_csv(test_path, index_col="id", dtype=col_dtypes_test)

train_df = preprocess_df(train_df)
test_df = preprocess_df(test_df)

columns = test_df.columns
X_raw = train_df[columns]
Y = train_df["target"].astype(np.int32)

X_train_raw, X_valid_raw, Y_train, Y_valid = train_test_split(
    X_raw, Y, test_size=0.05, random_state=42
)




## === cell 2
def fit_preprocessors(X_train: pd.DataFrame, degree: int = 1):
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_p = poly.fit_transform(X_train)
    X_train_s = scaler.fit_transform(X_train_p)
    return poly, scaler, X_train_s


def transform_with_preprocessors(X: pd.DataFrame, poly, scaler):
    X_p = poly.transform(X)
    X_s = scaler.transform(X_p)
    return X_s


degree = 1  # preserve your default behavior

poly, scaler, X_train = fit_preprocessors(X_train_raw, degree=degree)
X_valid = transform_with_preprocessors(X_valid_raw, poly, scaler)
X_test = transform_with_preprocessors(test_df[columns], poly, scaler)

print("X_train.shape", X_train.shape)
print("Y_train.shape", Y_train.shape)



## === cell 3
models = [
    (sklearn.linear_model.ARDRegression, {}),
    (sklearn.linear_model.BayesianRidge, {}),
    (sklearn.linear_model.ElasticNet, {}),
    (sklearn.linear_model.HuberRegressor, {"max_iter": 1000}),
    (sklearn.linear_model.Lars, {}),
    (sklearn.linear_model.LarsCV, {}),
    (sklearn.linear_model.Lasso, {}),
    (sklearn.linear_model.LassoCV, {"max_iter": 10_000}),
    (sklearn.linear_model.LassoLars, {}),
    (sklearn.linear_model.LassoLarsCV, {}),
    (sklearn.linear_model.LassoLarsIC, {}),
    (sklearn.linear_model.LinearRegression, {}),
    (sklearn.linear_model.OrthogonalMatchingPursuit, {}),
    (sklearn.linear_model.OrthogonalMatchingPursuitCV, {}),
    (sklearn.linear_model.PassiveAggressiveRegressor, {}),
    (sklearn.linear_model.RANSACRegressor, {}),
    (sklearn.linear_model.Ridge, {}),
    (sklearn.linear_model.RidgeCV, {}),
    (sklearn.linear_model.SGDRegressor, {}),
]




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/380285057.py in <cell line: 0>()
      1 # Core logic preserved: large list of sklearn linear models.
      2 models = [
----> 3     (sklearn.linear_model.ARDRegression, {}),
      4     (sklearn.linear_model.BayesianRidge, {}),
      5     (sklearn.linear_model.ElasticNet, {}),

AttributeError: module 'sklearn' has no attribute 'linear_model'

## === cell 4
def fit_predict(model_class, kwargs, verbose=True):
    name = model_class.__name__
    if verbose:
        print(name)

    model = model_class(**kwargs)
    model.fit(X_train, Y_train)

    valid_pred = model.predict(X_valid)
    valid_pred = np.clip(valid_pred, 0.0, 1.0)

    auc = roc_auc_score(Y_valid, valid_pred)

    test_pred = model.predict(X_test)
    test_pred = np.clip(test_pred, 0.0, 1.0)
    return name, auc, test_pred


scores = {}
predictions = {}
for model_class, kwargs in models:
    try:
        name, auc, prediction = fit_predict(model_class, kwargs, verbose=True)
        scores[name] = auc
        predictions[name] = prediction
    except Exception as e:
        print("ERROR", model_class.__name__, "->", str(e)[:200])



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4106394273.py in <cell line: 0>()
     21 scores = {}
     22 predictions = {}
---> 23 for model_class, kwargs in models:
     24     try:
     25         name, auc, prediction = fit_predict(model_class, kwargs, verbose=True)

NameError: name 'models' is not defined

## === cell 5
scores = dict(sorted(scores.items(), key=itemgetter(1), reverse=True))
print("Top models by valid ROC-AUC:")
for i, (k, v) in enumerate(scores.items()):
    if i >= 10:
        break
    print(f"{i+1:2d}. {k:30s} {v:.6f}")



## === cell 6
auc_values = np.array(list(scores.values()), dtype=float)
best_auc = float(np.max(auc_values)) if len(auc_values) else 0.5
threshold = best_auc - 0.01

selected = [name for name, auc in scores.items() if auc >= threshold]
if len(selected) == 0:
    selected = [next(iter(scores.keys()))]  # fallback

print("Selected models:", selected)

Y_test = np.mean([predictions[name] for name in selected], axis=0)
Y_test = np.clip(Y_test, 0.0, 1.0)
print(
    "Y_test.shape", Y_test.shape, "min/max:", float(Y_test.min()), float(Y_test.max())
)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
StopIteration                             Traceback (most recent call last)
/tmp/ipykernel_11/1509225942.py in <cell line: 0>()
      8 selected = [name for name, auc in scores.items() if auc >= threshold]
      9 if len(selected) == 0:
---> 10     selected = [next(iter(scores.keys()))]  # fallback
     11 
     12 print("Selected models:", selected)

StopIteration: 

## === cell 7
submission_df = pd.read_csv(sub_path, index_col="id")
submission_df["target"] = Y_test.astype(np.float32)
submission_df.to_csv("submission.csv")

print(submission_df.head())
print("Wrote submission.csv with shape:", submission_df.shape)

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3480151655.py in <cell line: 0>()
      1 # Change: remove shell magic; always write a valid submission.csv with correct index alignment.
      2 submission_df = pd.read_csv(sub_path, index_col="id")
----> 3 submission_df["target"] = Y_test.astype(np.float32)
      4 submission_df.to_csv("submission.csv")
      5 

NameError: name 'Y_test' is not defined
