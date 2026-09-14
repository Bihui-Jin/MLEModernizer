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
plotly==5.24.1
plotly-express==0.4.1
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tqdm==4.67.1
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

0.93763

# 6. Current score

None

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    get_ipython().run_line_magic("matplotlib", "inline")
except Exception:
    pass

import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import os
import pathlib
import gc
import sys
import re
import math
import random
import time
import datetime as dt
from tqdm import tqdm

import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance

from sklearn.model_selection import StratifiedKFold

import warnings

warnings.filterwarnings("ignore")

print("import done!")




## === cell 1
def load_csv(path_str):
    """Try several common locations for a dataset file, including Kaggle's default input folder."""
    p = pathlib.Path(path_str)
    if p.is_file():
        return pd.read_csv(p)
    candidates = [
        pathlib.Path("data") / p,
        pathlib.Path("kaggle") / "data" / p,
        pathlib.Path("/kaggle/input") / p,
        pathlib.Path("input") / p,
    ]
    for alt in candidates:
        if alt.is_file():
            return pd.read_csv(alt)
    raise FileNotFoundError(f"Cannot find {path_str} in known locations.")


global_seed = 42
random.seed(global_seed)
np.random.seed(global_seed)
os.environ["PYTHONHASHSEED"] = str(global_seed)
print("Seeds setted!")

data_config = {
    "train_csv_path": "tabular-playground-series-may-2022/train.csv",
    "test_csv_path": "tabular-playground-series-may-2022/test.csv",
    "sample_submission_path": "tabular-playground-series-may-2022/sample_submission.csv",
}

train_df = load_csv(data_config["train_csv_path"])
test_df = load_csv(data_config["test_csv_path"])
submission_df = load_csv(data_config["sample_submission_path"])

print(f"train_length: {len(train_df)}")
print(f"test_lenght: {len(test_df)}")
print(f"submission_length: {len(submission_df)}")




## === cell 2
train_df.head()




## === cell 3
print("train_df.info()")
print(train_df.info(), "\n")




## === cell 4
numeric_df = train_df.drop(["id"], axis=1).select_dtypes(include=[np.number])
fig = px.imshow(
    numeric_df.corr(),
    color_continuous_scale="RdBu_r",
    color_continuous_midpoint=0,
    aspect="auto",
)
fig.update_layout(height=750, title="Heatmap", showlegend=False)
fig.show()




## === cell 5
train_pos_df = train_df.query("target==1")
train_neg_df = train_df.query("target==0")

numerical_columns = [
    "f_00",
    "f_01",
    "f_02",
    "f_03",
    "f_04",
    "f_05",
    "f_06",
    "f_19",
    "f_20",
    "f_21",
    "f_22",
    "f_23",
    "f_24",
    "f_25",
    "f_26",
    "f_28",
]
categorical_columns = [
    "f_07",
    "f_08",
    "f_09",
    "f_10",
    "f_11",
    "f_12",
    "f_13",
    "f_14",
    "f_15",
    "f_16",
    "f_17",
    "f_18",
    "f_29",
    "f_30",
]
obj_columns = ["f_27"]

print(
    f"numerical_columns: {len(numerical_columns)},  categorical_columns: {len(categorical_columns)},  obj_columns: {len(obj_columns)}"
)




## === cell 6
train_df[numerical_columns].describe()




## === cell 7
fig = plt.figure(figsize=(16, 10))
for i, c in enumerate(numerical_columns):
    ax = fig.add_subplot(4, 4, i + 1)
    ax.hist(train_pos_df[c], color="b", alpha=0.5, bins=50)
    ax.hist(train_neg_df[c], color="r", alpha=0.5, bins=50)
    ax.set_title(numerical_columns[i])

fig.suptitle(
    'Distributions of Numerical Features (Blue: "target=1", red: "target=0")',
    fontsize=20,
)
fig.tight_layout()
plt.show()




## === cell 8
train_df[categorical_columns].describe()




## === cell 9
fig = plt.figure(figsize=(16, 10))
for i, c in enumerate(categorical_columns):
    ax = fig.add_subplot(4, 4, i + 1)
    x_range = (train_df[c].min(), train_df[c].max())
    bins = int(train_df[c].max() - train_df[c].min() + 1)
    ax.hist(train_pos_df[c], color="b", alpha=0.5, range=x_range, bins=bins)
    ax.hist(train_neg_df[c], color="r", alpha=0.5, range=x_range, bins=bins)
    ax.set_title(categorical_columns[i])

fig.suptitle(
    'Distributions of Categorical Features (Blue: "target=1", red: "target=0")',
    fontsize=20,
)
fig.tight_layout()
plt.show()




## === cell 10
f_27_df = train_df[["f_27", "target"]]
f_27_feature_df = f_27_df.drop(["f_27"], axis=1)
f_27_feature_df["n_char"] = f_27_df["f_27"].map(lambda x: len(x))

for i in range(65, 91):  # ASCII A‑Z
    f_27_feature_df[chr(i)] = f_27_df["f_27"].map(lambda x: x.count(chr(i)))

f_27_feature_df.describe()




## === cell 11
tmp_df = f_27_feature_df.groupby(["target"]).sum()
tmp_df = tmp_df.drop(["n_char"], axis=1)

fig = make_subplots(
    rows=2,
    cols=1,
    subplot_titles=["target=0", "target=1"],
    shared_xaxes="all",
    shared_yaxes="all",
)
for row in range(2):
    data = go.Bar(
        x=tmp_df.columns.astype(str).values,
        y=tmp_df.query(f"target=={row}").values.squeeze(),
    )
    fig.add_trace(data, row=row + 1, col=1)
fig.update_layout(title="Count of Characters", showlegend=False)
fig.show()




## === cell 12
f_27_feature_df = f_27_feature_df.drop(["U", "V", "W", "X", "Y", "Z", "n_char"], axis=1)
fig = px.imshow(
    f_27_feature_df.corr(),
    color_continuous_scale="RdBu_r",
    color_continuous_midpoint=0,
    aspect="auto",
)
fig.update_layout(height=750, title="Heatmap", showlegend=False)
fig.show()




## === cell 13
f_27_feature_df["A_B"] = f_27_feature_df["A"] / (f_27_feature_df["B"] + 1)
f_27_feature_df["A_C"] = f_27_feature_df["A"] / (f_27_feature_df["C"] + 1)
f_27_feature_df["A_D"] = f_27_feature_df["A"] / (f_27_feature_df["D"] + 1)
f_27_feature_df["B_C"] = f_27_feature_df["B"] / (f_27_feature_df["C"] + 1)
f_27_feature_df["B_D"] = f_27_feature_df["B"] / (f_27_feature_df["D"] + 1)
f_27_feature_df["B_E"] = f_27_feature_df["B"] / (f_27_feature_df["E"] + 1)
f_27_feature_df["C_D"] = f_27_feature_df["C"] / (f_27_feature_df["D"] + 1)
f_27_feature_df["C_E"] = f_27_feature_df["C"] / (f_27_feature_df["E"] + 1)
f_27_feature_df["D_E"] = f_27_feature_df["D"] / (f_27_feature_df["E"] + 1)

tmp_df = (
    f_27_feature_df[
        ["target", "A_B", "A_C", "A_D", "B_C", "B_D", "B_E", "C_D", "C_E", "D_E"]
    ]
    .groupby(["target"])
    .sum()
)
fig = make_subplots(
    rows=2,
    cols=1,
    subplot_titles=["target=0", "target=1"],
    shared_xaxes="all",
    shared_yaxes="all",
)
for row in range(2):
    data = go.Bar(
        x=tmp_df.columns.astype(str).values,
        y=tmp_df.query(f"target=={row}").values.squeeze(),
    )
    fig.add_trace(data, row=row + 1, col=1)
fig.update_layout(title="Count of Character Ratio", showlegend=False)
fig.show()




## === cell 14
train_df = train_df.drop(["f_27"], axis=1)
f_27_feature_df = f_27_feature_df.drop(["target"], axis=1)
train = pd.merge(train_df, f_27_feature_df, left_index=True, right_index=True)
print("Columns after merge:", train.columns.tolist())




## === cell 15
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=global_seed)
train["k_folds"] = -1
for fold, (train_idx, valid_idx) in enumerate(skf.split(X=train, y=train["target"])):
    train.loc[valid_idx, "k_folds"] = fold

models = []
for fold in range(n_splits):
    print(f"======fold {fold}======")
    valid_tmp = train[train["k_folds"] == fold]
    train_tmp = train[train["k_folds"] != fold]

    y_train = train_tmp["target"]
    X_train = train_tmp.drop(["id", "target", "k_folds"], axis=1)

    y_valid = valid_tmp["target"]
    X_valid = valid_tmp.drop(["id", "target", "k_folds"], axis=1)

    model = XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        seed=global_seed,
        eval_metric="auc",
    )
    model.fit(
        X_train,
        y_train,
        verbose=False,
        early_stopping_rounds=10,
        eval_set=[(X_valid, y_valid)],
    )
    models.append(model)




## === cell 16
if models:
    model = models[-1]  # use the most recent fitted model
    fig, ax = plt.subplots(1, 1, figsize=(20, 12))
    plot_importance(model, ax=ax, xlabel=None)
    plt.title("XGB Feature importance", fontsize=20)
    plt.show()
else:
    print("No models were trained; skipping importance plot.")




## === cell 17
for i in range(65, 91):  # ASCII A‑Z
    test_df[chr(i)] = test_df["f_27"].map(lambda x: x.count(chr(i)))

test_df["A_B"] = test_df["A"] / (test_df["B"] + 1)
test_df["A_C"] = test_df["A"] / (test_df["C"] + 1)
test_df["A_D"] = test_df["A"] / (test_df["D"] + 1)
test_df["B_C"] = test_df["B"] / (test_df["C"] + 1)
test_df["B_D"] = test_df["B"] / (test_df["D"] + 1)
test_df["B_E"] = test_df["B"] / (test_df["E"] + 1)
test_df["C_D"] = test_df["C"] / (test_df["D"] + 1)
test_df["C_E"] = test_df["C"] / (test_df["E"] + 1)
test_df["D_E"] = test_df["D"] / (test_df["E"] + 1)

test_df = test_df.drop(["id", "f_27"], axis=1)
print("Test columns:", test_df.columns.tolist())




## === cell 18
for i, model in enumerate(models):
    submission_df[f"pred_{i}"] = model.predict_proba(test_df)[:, 1]
submission_df.head()




## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/343046266.py in <cell line: 0>()
      1 for i, model in enumerate(models):
----> 2     submission_df[f"pred_{i}"] = model.predict_proba(test_df)[:, 1]
      3 submission_df.head()
      4 
      5 

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in predict_proba(self, X, validate_features, base_margin, iteration_range)
   1630             class_prob = softmax(raw_predt, axis=1)
   1631             return class_prob
-> 1632         class_probs = super().predict(
   1633             X=X,
   1634             validate_features=validate_features,

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in predict(self, X, output_margin, validate_features, base_margin, iteration_range)
   1166             if self._can_use_inplace_predict():
   1167                 try:
-> 1168                     predts = self.get_booster().inplace_predict(
   1169                         data=X,
   1170                         iteration_range=iteration_range,

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in inplace_predict(self, data, iteration_range, predict_type, missing, validate_features, base_margin, strict_shape)
   2416             data, fns, _ = _transform_pandas_df(data, enable_categorical)
   2417             if validate_features:
-> 2418                 self._validate_features(fns)
   2419         if _is_list(data) or _is_tuple(data):
   2420             data = np.array(data)

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in _validate_features(self, feature_names)
   2968                 )
   2969 
-> 2970             raise ValueError(msg.format(self.feature_names, feature_names))
   2971 
   2972     def get_split_value_histogram(

ValueError: feature_names mismatch: ['f_00', 'f_01', 'f_02', 'f_03', 'f_04', 'f_05', 'f_06', 'f_07', 'f_08', 'f_09', 'f_10', 'f_11', 'f_12', 'f_13', 'f_14', 'f_15', 'f_16', 'f_17', 'f_18', 'f_19', 'f_20', 'f_21', 'f_22', 'f_23', 'f_24', 'f_25', 'f_26', 'f_28', 'f_29', 'f_30', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'A_B', 'A_C', 'A_D', 'B_C', 'B_D', 'B_E', 'C_D', 'C_E', 'D_E'] ['f_00', 'f_01', 'f_02', 'f_03', 'f_04', 'f_05', 'f_06', 'f_07', 'f_08', 'f_09', 'f_10', 'f_11', 'f_12', 'f_13', 'f_14', 'f_15', 'f_16', 'f_17', 'f_18', 'f_19', 'f_20', 'f_21', 'f_22', 'f_23', 'f_24', 'f_25', 'f_26', 'f_28', 'f_29', 'f_30', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'A_B', 'A_C', 'A_D', 'B_C', 'B_D', 'B_E', 'C_D', 'C_E', 'D_E']
training data did not have the following fields: U, V, Y, X, W, Z

## === cell 19
pred_cols = [col for col in submission_df.columns if col.startswith("pred_")]
submission_df["target"] = submission_df[pred_cols].mean(axis=1)
submission_df = submission_df.drop(columns=pred_cols)
submission_df.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
submission_df.head()
