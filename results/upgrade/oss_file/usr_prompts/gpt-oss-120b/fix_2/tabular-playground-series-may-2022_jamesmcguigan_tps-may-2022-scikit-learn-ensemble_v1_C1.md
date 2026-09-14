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

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import sklearn
from operator import itemgetter
from sklearn.linear_model import *
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import sklearn.metrics
import warnings

warnings.filterwarnings("ignore")
pp = None  # placeholder to keep original variable name if accessed later



## === cell 1
train_path = "/kaggle/input/tabular-playground-series-may-2022/train.csv"
test_path = "/kaggle/input/tabular-playground-series-may-2022/test.csv"
sample_path = "/kaggle/input/tabular-playground-series-may-2022/sample_submission.csv"

col_dtypes = {
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


def preprocess_df(df):
    if "f_27" in df.columns:
        df = df.drop(columns=["f_27"])
    return df


train_df = pd.read_csv(train_path, index_col="id", dtype=col_dtypes)
test_df = pd.read_csv(test_path, index_col="id", dtype=col_dtypes)

train_df = preprocess_df(train_df)
test_df = preprocess_df(test_df)

print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)




## === cell 2
def preprocess_X(X, degree=1):
    """Polynomial features + standard scaling."""
    X_poly = PolynomialFeatures(degree=degree, include_bias=False).fit_transform(X)
    X_scaled = StandardScaler().fit_transform(X_poly)
    return X_scaled


feature_cols = test_df.columns  # same columns as in train after dropping f_27
X = preprocess_X(train_df[feature_cols])
Y = train_df["target"]
X_test = preprocess_X(test_df[feature_cols])

X_train, X_valid, Y_train, Y_valid = sklearn.model_selection.train_test_split(
    X, Y, test_size=0.05, random_state=42
)

print("Processed X shape:", X.shape)



## === cell 3
models = [
    (sklearn.linear_model.ARDRegression, {}),
    (sklearn.linear_model.BayesianRidge, {}),
    (sklearn.linear_model.ElasticNet, {}),
    (sklearn.linear_model.HuberRegressor, {"max_iter": 1000}),
    (sklearn.linear_model.LassoCV, {"max_iter": 10_000}),
    (sklearn.linear_model.Ridge, {}),
    (sklearn.linear_model.SGDRegressor, {}),
    (sklearn.linear_model.LogisticRegression, {"max_iter": 1000, "solver": "lbfgs"}),
]




## === cell 4
def fit_predict(model_class, kwargs, verbose=True):
    """Fit a model on the training split, evaluate RMSE on validation,
    and predict on the test set."""
    name = model_class.__name__
    if verbose:
        print("Training:", name)
    model = model_class(**kwargs)
    model.fit(X_train, Y_train)
    preds_valid = model.predict(X_valid)
    rmse = sklearn.metrics.mean_squared_error(Y_valid, preds_valid, squared=False)
    preds_test = model.predict(X_test)
    return name, rmse, preds_test


scores = {}
predictions = {}
for model_class, kwargs in models:
    try:
        name, rmse, pred = fit_predict(model_class, kwargs)
        scores[name] = rmse
        predictions[name] = pred
    except Exception as e:
        print(f"ERROR with {model_class.__name__}: {e}")



## === cell 5
scores = dict(sorted(scores.items(), key=itemgetter(1)))
print("Model RMSE scores:", scores)

selected = [name for name, sc in scores.items() if sc <= 0.8]
if not selected:
    selected = list(scores.keys())

Y_test = np.mean([predictions[name] for name in selected], axis=0)
print("Blended prediction shape:", Y_test.shape)



## === cell 6
submission_df = pd.read_csv(sample_path, index_col="id")
submission_df["target"] = Y_test
output_path = "submission.csv"
submission_df.to_csv(output_path, index=True)
print(f"Submission written to {output_path}")
