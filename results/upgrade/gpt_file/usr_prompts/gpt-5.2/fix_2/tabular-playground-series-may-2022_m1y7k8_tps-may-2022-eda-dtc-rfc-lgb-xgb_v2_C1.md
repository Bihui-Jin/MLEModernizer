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

0.93489

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

import matplotlib.pyplot as plt
import seaborn as sns

from collections import OrderedDict

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import roc_curve, auc

from xgboost import XGBClassifier
import xgboost as xgb

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 1
train = pd.read_csv("/kaggle/input/tabular-playground-series-may-2022/train.csv")
test = pd.read_csv("/kaggle/input/tabular-playground-series-may-2022/test.csv")



## === cell 2
train.head().T




## === cell 3
def check(df):
    col_list = df.columns.values
    rows = []
    for col in col_list:
        tmp = (
            col,
            df[col].dtype,
            df[col].isnull().sum(),
            df[col].count(),
            df[col].nunique(),
            df[col].unique(),
        )
        rows.append(tmp)
    df2 = pd.DataFrame(rows)
    df2.columns = ["feature", "dtype", "nan", "count", "nunique", "unique"]
    return df2




## === cell 4
check(train)




## === cell 5
def color_negative_red(val):
    color = "red" if val < 0 else "black"
    return "color: %s" % color




## === cell 6
cm = sns.light_palette("green", as_cmap=True)
train.drop("id", axis=1).describe().T.style.background_gradient(cmap=cm).applymap(
    color_negative_red
)



## === cell 7
test.head().T



## === cell 8
check(test)



## === cell 9
test.drop("id", axis=1).describe().T.style.background_gradient(cmap=cm).applymap(
    color_negative_red
)



## === cell 10
target_count = train["target"].value_counts()
target_count



## === cell 11
train["target"].describe()



## === cell 12
sns.set_theme(context="talk", style="darkgrid", palette="spring")
fig, axs = plt.subplots(ncols=2, figsize=(12, 5))

sns.countplot(x=train["target"], data=train, ax=axs[0])

labels = ["1", "0"]
axs[1].pie(target_count, labels=labels, autopct="%.0f%%")

plt.show()



## === cell 13
train["f_27"].value_counts()



## === cell 14
test["f_27"].value_counts()



## === cell 15
from collections import OrderedDict


def encord(input_str):
    d = OrderedDict.fromkeys(input_str, 0)
    for ch in input_str:
        d[ch] += 1
    output = ""
    for k, v in d.items():
        output = output + k + str(v)
    return output




## === cell 16
f_27_en = []
for i in range(len(train["f_27"])):
    a = train["f_27"].iloc[i]
    st = encord(a)
    f_27_en.append(st)

train["f_27_en"] = f_27_en
train["f_27_en"].value_counts()



## === cell 17
f_27_ent = []
for i in range(len(test["f_27"])):
    a = test["f_27"].iloc[i]
    st = encord(a)
    f_27_ent.append(st)

test["f_27_ent"] = f_27_ent
test["f_27_ent"].value_counts()



## === cell 18
label_f27 = LabelEncoder()
all_f27 = pd.concat([train["f_27"], test["f_27"]], axis=0).astype(str)
label_f27.fit(all_f27)

train["en_27"] = label_f27.transform(train["f_27"].astype(str))
test["en_27"] = label_f27.transform(test["f_27"].astype(str))

label_f27_en = LabelEncoder()
all_f27_en = pd.concat([train["f_27_en"], test["f_27_ent"]], axis=0).astype(str)
label_f27_en.fit(all_f27_en)

train["f_27_enc"] = label_f27_en.transform(train["f_27_en"].astype(str))
test["f_27_enc"] = label_f27_en.transform(test["f_27_ent"].astype(str))

display(train["en_27"].head(10))
display(train["f_27_enc"].head(10))
display(test["f_27_enc"].head(10))



## === cell 19
train.head().T



## === cell 20
test.head().T



## === cell 21
clf = DecisionTreeClassifier(max_depth=2, random_state=42)
clf.fit(train[["f_26"]], train["target"])
_, ax = plt.subplots(figsize=(20, 10))

plot_tree(
    clf,
    feature_names=["f_26"],
    class_names=train["target"].unique().astype(str),
    filled=True,
    ax=ax,
    fontsize=15,
    rounded=True,
)

plt.show()



## === cell 22
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(train[["f_28"]], train["target"])
_, ax = plt.subplots(figsize=(20, 10))

plot_tree(
    clf,
    feature_names=["f_28"],
    class_names=train["target"].unique().astype(str),
    filled=True,
    ax=ax,
    fontsize=15,
    rounded=True,
)

plt.show()



## === cell 23
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(train[["f_29"]], train["target"])
_, ax = plt.subplots(figsize=(20, 10))

plot_tree(
    clf,
    feature_names=["f_29"],
    class_names=train["target"].unique().astype(str),
    filled=True,
    ax=ax,
    fontsize=15,
    rounded=True,
)

plt.show()



## === cell 24
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(train[["f_30"]], train["target"])
_, ax = plt.subplots(figsize=(20, 10))

plot_tree(
    clf,
    feature_names=["f_30"],
    class_names=train["target"].unique().astype(str),
    filled=True,
    ax=ax,
    fontsize=15,
    rounded=True,
)

plt.show()



## === cell 25
for col in train.columns:
    if train[col].dtype == "float64":
        train[col] = pd.to_numeric(train[col], downcast="float")
    if train[col].dtype == "int64":
        train[col] = pd.to_numeric(train[col], downcast="integer")

for col in test.columns:
    if test[col].dtype == "float64":
        test[col] = pd.to_numeric(test[col], downcast="float")
    if test[col].dtype == "int64":
        test[col] = pd.to_numeric(test[col], downcast="integer")



## === cell 26
train.info(), test.info()



## === cell 27
X = train.drop(["id", "target", "f_27", "en_27", "f_27_en"], axis=1).copy()
y = train["target"].copy()
X_test = test.drop(["id", "f_27", "f_27_ent"], axis=1).copy()

del train
del test




## === cell 28
def _xgb_device_params():
    try:
        devices = xgb.core._py_version.get(
            "use_cuda", None
        )  # not reliable across versions
    except Exception:
        devices = None

    return


params = {
    "n_estimators": 10000,
    "colsample_bytree": 0.5,
    "subsample": 0.5,
    "learning_rate": 0.02,
    "max_depth": 6,
}



## === cell 29
splits = 5
seed = 42
skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)

preds = []
scores = []

use_gpu = True

for fold, (idx_train, idx_valid) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]
    X_valid, y_valid = X.iloc[idx_valid], y.iloc[idx_valid]

    if use_gpu:
        model = XGBClassifier(
            **params,
            booster="gbtree",
            eval_metric="auc",
            tree_method="gpu_hist",
            predictor="gpu_predictor",
            gpu_id=0,
            use_label_encoder=False,
            random_state=seed,
        )
    else:
        model = XGBClassifier(
            **params,
            booster="gbtree",
            eval_metric="auc",
            tree_method="hist",
            predictor="auto",
            use_label_encoder=False,
            random_state=seed,
        )

    try:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            early_stopping_rounds=100,
            verbose=False,
        )
    except xgb.core.XGBoostError as e:
        if use_gpu and (
            "Must have at least one device" in str(e)
            or "gpu_id" in str(e)
            or "gpu_hist" in str(e)
        ):
            use_gpu = False
            model = XGBClassifier(
                **params,
                booster="gbtree",
                eval_metric="auc",
                tree_method="hist",
                predictor="auto",
                use_label_encoder=False,
                random_state=seed,
            )
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                early_stopping_rounds=100,
                verbose=False,
            )
        else:
            raise

    pred_valid = model.predict_proba(X_valid)[:, 1]
    fpr, tpr, _ = roc_curve(y_valid, pred_valid)
    score = auc(fpr, tpr)
    scores.append(score)

    test_preds_fold = model.predict_proba(X_test)[:, 1]
    preds.append(test_preds_fold)

    print("fold : ", fold, "score : ", score)



## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3591226575.py in <cell line: 0>()
     75     scores.append(score)
     76 
---> 77     test_preds_fold = model.predict_proba(X_test)[:, 1]
     78     preds.append(test_preds_fold)
     79 

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

ValueError: feature_names mismatch: ['f_00', 'f_01', 'f_02', 'f_03', 'f_04', 'f_05', 'f_06', 'f_07', 'f_08', 'f_09', 'f_10', 'f_11', 'f_12', 'f_13', 'f_14', 'f_15', 'f_16', 'f_17', 'f_18', 'f_19', 'f_20', 'f_21', 'f_22', 'f_23', 'f_24', 'f_25', 'f_26', 'f_28', 'f_29', 'f_30', 'f_27_enc'] ['f_00', 'f_01', 'f_02', 'f_03', 'f_04', 'f_05', 'f_06', 'f_07', 'f_08', 'f_09', 'f_10', 'f_11', 'f_12', 'f_13', 'f_14', 'f_15', 'f_16', 'f_17', 'f_18', 'f_19', 'f_20', 'f_21', 'f_22', 'f_23', 'f_24', 'f_25', 'f_26', 'f_28', 'f_29', 'f_30', 'en_27', 'f_27_enc']
training data did not have the following fields: en_27

## === cell 30
print(scores)
print("CV mean AUC:", float(np.mean(scores)), "std:", float(np.std(scores)))



## === cell 31
sub = pd.read_csv(
    "/kaggle/input/tabular-playground-series-may-2022/sample_submission.csv"
)

test_pred_mean = np.mean(np.vstack(preds), axis=0)
sub["target"] = test_pred_mean.astype(float)

sub.to_csv("submission.csv", index=False)
sub.head()

## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2140755582.py in <cell line: 0>()
      4 )
      5 
----> 6 test_pred_mean = np.mean(np.vstack(preds), axis=0)
      7 sub["target"] = test_pred_mean.astype(float)
      8 

/usr/local/lib/python3.11/dist-packages/numpy/core/shape_base.py in vstack(tup, dtype, casting)
    287     if not isinstance(arrs, list):
    288         arrs = [arrs]
--> 289     return _nx.concatenate(arrs, 0, dtype=dtype, casting=casting)
    290 
    291 

ValueError: need at least one array to concatenate
