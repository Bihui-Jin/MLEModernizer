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
%matplotlib inline

from collections import OrderedDict

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc

from xgboost import XGBClassifier
import xgboost as xgb

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


## === cell 1
train = pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/train.csv')
test = pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/test.csv')


## === cell 2
train.head().T


## === cell 3
def check(df):
    col_list = df.columns.values
    rows = []
    for col in col_list:
        tmp = (col,
              df[col].dtype,
              df[col].isnull().sum(),
              df[col].count(),
              df[col].nunique(),
              df[col].unique())
        rows.append(tmp)
    df = pd.DataFrame(rows) 
    df.columns = ['feature','dtype','nan','count','nunique','unique']
    return df


## === cell 4
check(train)


## === cell 5
def color_negative_red(val):
    color = 'red' if val < 0 else 'black'
    return 'color: %s' % color


## === cell 6
cm = sns.light_palette('green', as_cmap = True)
train.drop('id', axis = 1).describe().T.style.background_gradient(cmap = cm).applymap(color_negative_red)


## === cell 7
test.head().T


## === cell 8
check(test)


## === cell 9
test.drop('id', axis = 1).describe().T.style.background_gradient(cmap = cm).applymap(color_negative_red)


## === cell 10
target_count = train['target'].value_counts()
target_count


## === cell 11
train['target'].describe()


## === cell 12
sns.set('talk', 'dark','spring_r')

fig,axs = plt.subplots(ncols = 2, )

sns.countplot(x=train['target'], data=train, ax = axs[0])

labels = ['1','0']
plt.pie(target_count, labels = labels, autopct = '%.0f%%')

plt.show()


## === cell 13
train['f_27'].value_counts()


## === cell 14
test['f_27'].value_counts()


## === cell 15
from collections import OrderedDict

def encord(input):
    dict = OrderedDict.fromkeys(input,0)
        
    for ch in input:
        dict[ch] += 1
        
    output = ''
    for k, v in dict.items():
        output = output + k + str(v)
    return output
        


## === cell 16
f_27_en=[]
for i in range(len(train['f_27'])):
    a = train['f_27'][i]
    st = encord(a)
    f_27_en.append(st)
    
train['f_27_en'] = f_27_en
train['f_27_en'].value_counts() 


## === cell 17
f_27_ent=[]
for i in range(len(test['f_27'])):
    a = test['f_27'][i]
    st = encord(a)
    f_27_ent.append(st)
    
test['f_27_ent'] = f_27_ent
test['f_27_ent'].value_counts() 


## === cell 18
label = LabelEncoder()

en_27 = pd.DataFrame(label.fit_transform(train['f_27']))
train['en_27'] = en_27

enc_27 = pd.DataFrame(label.fit_transform(train['f_27_en']))
train['f_27_enc'] = enc_27

enct_27 = pd.DataFrame(label.fit_transform(test['f_27_ent']))
test['f_27_enc'] = enct_27

display(train['en_27'].head(10))
display(train['f_27_enc'].head(10))
display(test['f_27_enc'].head(10))


## === cell 19
train.head().T


## === cell 20
test.head().T


## === cell 21
clf = DecisionTreeClassifier(max_depth=2)
clf.fit( train[["f_26"]], train["target"])
_, ax = plt.subplots(figsize=(20, 10))

plot_tree(
    clf,
    feature_names=["f_26"],
    class_names=train["target"].unique().astype(str),
    filled=True,
    ax=ax,
    fontsize=15,
    rounded =  True,
)

plt.show()


## === cell 22
lf = DecisionTreeClassifier(max_depth=5)
clf.fit( train[["f_28"]], train["target"])
_, ax = plt.subplots(figsize=(20, 10))

plot_tree(
    clf,
    feature_names=["f_28"],
    class_names=train["target"].unique().astype(str),
    filled=True,
    ax=ax,
    fontsize=15,
    rounded = True
)

plt.show()


## === cell 23
lf = DecisionTreeClassifier(max_depth=3)
clf.fit( train[["f_29"]], train["target"])
_, ax = plt.subplots(figsize=(20, 10))

plot_tree(
    clf,
    feature_names=["f_29"],
    class_names=train["target"].unique().astype(str),
    filled=True,
    ax=ax,
    fontsize=15,
    rounded = True
)

plt.show()


## === cell 24
lf = DecisionTreeClassifier(max_depth=3)
clf.fit( train[["f_30"]], train["target"])
_, ax = plt.subplots(figsize=(20, 10))

plot_tree(
    clf,
    feature_names=["f_30"],
    class_names=train["target"].unique().astype(str),
    filled=True,
    ax=ax,
    fontsize=15,
    rounded = True
)

plt.show()


## === cell 25
for col in train.columns:
    if train[col].dtype == "float64":
        train[col]=pd.to_numeric(train[col], downcast="float")
    if train[col].dtype == "int64":
        train[col]=pd.to_numeric(train[col], downcast="integer")
        
for col in test.columns:
    if test[col].dtype == "float64":
        test[col]=pd.to_numeric(test[col], downcast="float")
    if test[col].dtype == "int64":
        test[col]=pd.to_numeric(test[col], downcast="integer")


## === cell 26
train.info(),test.info()


## === cell 27
X = train.drop(['id','target','f_27','en_27','f_27_en'], axis = 1).copy()

y = train['target'].copy()
X_test = test.drop(['id','f_27','f_27_ent' ], axis = 1).copy()

del train
del test


## === cell 28
params = {'tree_method':'gpu_hist',
          'n_estimators': 10000,
          'colsample_bytree': 0.5, 
          'subsample': 0.5, 
          'learning_rate': 0.02, 
          'max_depth': 6, 
         }


## === cell 29
splits = 5
seed = 42
skf = StratifiedKFold(n_splits = splits, shuffle=True, random_state=seed)

preds = []
scores = []

for fold, (idx_train, idx_valid) in enumerate(skf.split(X, y)):
    X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]
    X_valid, y_valid = X.iloc[idx_valid], y.iloc[idx_valid]

    model = XGBClassifier(**params,
                            booster= 'gbtree',
                            eval_metric = 'auc',
                            gpu_id=0,
                            predictor="gpu_predictor",
                            use_label_encoder=False)
    
    model.fit(X_train,y_train,
              eval_set=[(X_valid,y_valid)],
              early_stopping_rounds=100,
              verbose=False)
    
    pred_valid = model.predict_proba(X_valid)[:,1]
    fpr, tpr, _ = roc_curve(y_valid, pred_valid)
    score = auc(fpr, tpr)
    scores.append(score)

    test_preds = model.predict_proba(X_test)[:,1]
    preds.append(test_preds)
    
    print("fold : ", fold , "score : ", score)


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
XGBoostError                              Traceback (most recent call last)
/tmp/ipykernel_11/1867500236.py in <cell line: 0>()
     17                             use_label_encoder=False)
     18 
---> 19     model.fit(X_train,y_train,
     20               eval_set=[(X_valid,y_valid)],
     21               early_stopping_rounds=100,

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in inner_f(*args, **kwargs)
    728             for k, arg in zip(sig.parameters, args):
    729                 kwargs[k] = arg
--> 730             return func(**kwargs)
    731 
    732         return inner_f

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in fit(self, X, y, sample_weight, base_margin, eval_set, eval_metric, early_stopping_rounds, verbose, xgb_model, sample_weight_eval_set, base_margin_eval_set, feature_weights, callbacks)
   1517             )
   1518 
-> 1519             self._Booster = train(
   1520                 params,
   1521                 train_dmatrix,

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in inner_f(*args, **kwargs)
    728             for k, arg in zip(sig.parameters, args):
    729                 kwargs[k] = arg
--> 730             return func(**kwargs)
    731 
    732         return inner_f

/usr/local/lib/python3.11/dist-packages/xgboost/training.py in train(params, dtrain, num_boost_round, evals, obj, feval, maximize, early_stopping_rounds, evals_result, verbose_eval, xgb_model, callbacks, custom_metric)
    179         if cb_container.before_iteration(bst, i, dtrain, evals):
    180             break
--> 181         bst.update(dtrain, i, obj)
    182         if cb_container.after_iteration(bst, i, dtrain, evals):
    183             break

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in update(self, dtrain, iteration, fobj)
   2048 
   2049         if fobj is None:
-> 2050             _check_call(
   2051                 _LIB.XGBoosterUpdateOneIter(
   2052                     self.handle, ctypes.c_int(iteration), dtrain.handle

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in _check_call(ret)
    280     """
    281     if ret != 0:
--> 282         raise XGBoostError(py_str(_LIB.XGBGetLastError()))
    283 
    284 

XGBoostError: [01:20:33] /workspace/src/tree/updater_gpu_hist.cu:781: Exception in gpu_hist: [01:20:33] /workspace/src/tree/updater_gpu_hist.cu:787: Check failed: ctx_->gpu_id >= 0 (-1 vs. 0) : Must have at least one device
Stack trace:
  [bt] (0) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb27f2a) [0x7fb6fcda8f2a]
  [bt] (1) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb3e95a) [0x7fb6fcdbf95a]
  [bt] (2) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb483cd) [0x7fb6fcdc93cd]
  [bt] (3) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x460c79) [0x7fb6fc6e1c79]
  [bt] (4) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x46176c) [0x7fb6fc6e276c]
  [bt] (5) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x4c54f7) [0x7fb6fc7464f7]
  [bt] (6) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(XGBoosterUpdateOneIter+0x70) [0x7fb6fc3e2ef0]
  [bt] (7) /lib/x86_64-linux-gnu/libffi.so.8(+0x7e2e) [0x7fb77000ce2e]
  [bt] (8) /lib/x86_64-linux-gnu/libffi.so.8(+0x4493) [0x7fb770009493]



Stack trace:
  [bt] (0) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb27f2a) [0x7fb6fcda8f2a]
  [bt] (1) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb485c9) [0x7fb6fcdc95c9]
  [bt] (2) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x460c79) [0x7fb6fc6e1c79]
  [bt] (3) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x46176c) [0x7fb6fc6e276c]
  [bt] (4) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x4c54f7) [0x7fb6fc7464f7]
  [bt] (5) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(XGBoosterUpdateOneIter+0x70) [0x7fb6fc3e2ef0]
  [bt] (6) /lib/x86_64-linux-gnu/libffi.so.8(+0x7e2e) [0x7fb77000ce2e]
  [bt] (7) /lib/x86_64-linux-gnu/libffi.so.8(+0x4493) [0x7fb770009493]
  [bt] (8) /usr/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so(+0xa4d8) [0x7fb77001c4d8]



## === cell 30
print(scores)


## === cell 32
sub = pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/sample_submission.csv')


## === cell 33
sub['target'] = test_preds
sub.to_csv('submission.csv', index = False)
sub.head()  


## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/908491362.py in <cell line: 0>()
----> 1 sub['target'] = test_preds
      2 sub.to_csv('submission.csv', index = False)
      3 sub.head()

NameError: name 'test_preds' is not defined
