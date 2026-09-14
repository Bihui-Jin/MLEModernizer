# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.10

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
%matplotlib inline

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
warnings.filterwarnings('ignore')

print('import done!')


## === cell 1
def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    os.environ['PYTHONHASHSEED'] = str(s) 
    print('Seeds setted!')
    
global_seed = 42
seed_all(global_seed)


## === cell 2
data_config = {'train_csv_path': '../input/tabular-playground-series-may-2022/train.csv',
               'test_csv_path': '../input/tabular-playground-series-may-2022/test.csv',
               'sample_submission_path': '../input/tabular-playground-series-may-2022/sample_submission.csv',
              }

train_df = pd.read_csv(data_config['train_csv_path'])
test_df = pd.read_csv(data_config['test_csv_path'])
submission_df = pd.read_csv(data_config['sample_submission_path'])

print(f'train_length: {len(train_df)}')
print(f'test_lenght: {len(test_df)}')
print(f'submission_length: {len(submission_df)}')


## === cell 3
train_df.head()


## === cell 4
print('train_df.info()'); print(train_df.info(), '\n')


## === cell 5
target_count = train_df.groupby(['target'])['id'].count()
target_percent = target_count / target_count.sum()

fig = go.Figure()

data = go.Bar(x=target_count.index.astype(str).values, 
              y=target_count.values)

fig.add_trace(data)

fig.update_layout(title = dict(text='target distribution'),
                  xaxis = dict(title='target values'),
                  yaxis = dict(title='counts'))

fig.show()


## === cell 6
fig = px.imshow(
    train_df.drop(["id"], axis=1).corr(numeric_only=True),
    color_continuous_scale="RdBu_r",
    color_continuous_midpoint=0,
    aspect="auto",
)
fig.update_layout(height=750, title="Heatmap", showlegend=False)
fig.show()


## === cell 7
train_pos_df = train_df.query('target==1')
train_neg_df = train_df.query('target==0')

numerical_columns = ['f_00', 'f_01', 'f_02', 'f_03', 'f_04', 'f_05', 'f_06',
                 'f_19', 'f_20', 'f_21', 'f_22', 'f_23', 'f_24', 'f_25',
                 'f_26', 'f_28']
categorical_columns = ['f_07', 'f_08', 'f_09', 'f_10', 'f_11', 'f_12',
               'f_13', 'f_14', 'f_15', 'f_16', 'f_17', 'f_18',
               'f_29', 'f_30']
obj_columns = ['f_27']

print(f'numerical_columns: {len(numerical_columns)},  categorical_columns: {len(categorical_columns)},  obj_columns: {len(obj_columns)}')


## === cell 8
train_df[numerical_columns].describe()


## === cell 9
fig = plt.figure(figsize=(16, 10))
for i, c in enumerate(numerical_columns):
    ax = fig.add_subplot(4, 4, i+1)
    ax.hist(train_pos_df[c], color='b', alpha=0.5, bins=50)
    ax.hist(train_neg_df[c], color='r', alpha=0.5, bins=50)
    ax.set_title(numerical_columns[i])
    
fig.suptitle('Distributions of Numerical Features (Blue: "target=1", red: "target=0")', fontsize=20)
fig.tight_layout()
plt.show()


## === cell 10
train_df[categorical_columns].describe()


## === cell 11
fig = plt.figure(figsize=(16, 10))
for i, c in enumerate(categorical_columns):
    ax = fig.add_subplot(4, 4, i+1)
    x_range = (train_df[c].min(), train_df[c].max())
    bins = train_df[c].max() - train_df[c].min() + 1
    ax.hist(train_pos_df[c], color='b', alpha=0.5, range=x_range, bins=bins)
    ax.hist(train_neg_df[c], color='r', alpha=0.5, range=x_range, bins=bins)
    ax.set_title(categorical_columns[i])
    
fig.suptitle('Distributions of Categorical Features (Blue: "target=1", red: "target=0")', fontsize=20)
fig.tight_layout()
plt.show()


## === cell 12
f_27_df = train_df[['f_27', 'target']]
f_27_feature_df = f_27_df.drop(['f_27'], axis=1)
f_27_feature_df['n_char'] = f_27_df['f_27'].map(lambda x: len(x))

for i in range(65, 91): ## ASCII of A to Z
    f_27_feature_df[chr(i)] = f_27_df['f_27'].map(lambda x: x.count(chr(i)))
    
f_27_feature_df.describe()


## === cell 13
tmp_df = f_27_feature_df.groupby(['target']).sum()
tmp_df = tmp_df.drop(['n_char'], axis=1)

fig = make_subplots(rows=2, cols=1,
                    subplot_titles=['target=0', 'target=1'],
                    shared_xaxes='all',
                    shared_yaxes='all')
for row in range(2):
    for col in range(1):
        data = go.Bar(x=tmp_df.columns.astype(str).values,
                      y=tmp_df.query(f'target=={row}').values.squeeze())
        fig.add_trace(data, row=row+1, col=col+1)
fig.update_layout(title='Count of Characters',
                  showlegend=False)
fig.show()


## === cell 14
f_27_feature_df = f_27_feature_df.drop(['U', 'V', 'W', 'X', 'Y', 'Z', 'n_char'], axis=1)
fig = px.imshow(f_27_feature_df.corr(),
                color_continuous_scale='RdBu_r',
                color_continuous_midpoint=0,
                aspect='auto')
fig.update_layout(height=750, 
                  title = "Heatmap",                  
                  showlegend=False)
fig.show()


## === cell 15
f_27_feature_df['A_B'] = f_27_feature_df['A'] / (f_27_feature_df['B'] + 1)
f_27_feature_df['A_C'] = f_27_feature_df['A'] / (f_27_feature_df['C'] + 1)
f_27_feature_df['A_D'] = f_27_feature_df['A'] / (f_27_feature_df['D'] + 1)
f_27_feature_df['B_C'] = f_27_feature_df['B'] / (f_27_feature_df['C'] + 1)
f_27_feature_df['B_D'] = f_27_feature_df['B'] / (f_27_feature_df['D'] + 1)
f_27_feature_df['B_E'] = f_27_feature_df['B'] / (f_27_feature_df['E'] + 1)
f_27_feature_df['C_D'] = f_27_feature_df['C'] / (f_27_feature_df['D'] + 1)
f_27_feature_df['C_E'] = f_27_feature_df['C'] / (f_27_feature_df['E'] + 1)
f_27_feature_df['D_E'] = f_27_feature_df['D'] / (f_27_feature_df['E'] + 1)

tmp_df = f_27_feature_df[['target', 'A_B', 'A_C', 'A_D', 'B_C', 'B_D', 'B_E', 'C_D', 'C_E', 'D_E']].groupby(['target']).sum()
fig = make_subplots(rows=2, cols=1,
                    subplot_titles=['target=0', 'target=1'],
                    shared_xaxes='all',
                    shared_yaxes='all')
for row in range(2):
    for col in range(1):
        data = go.Bar(x=tmp_df.columns.astype(str).values,
                      y=tmp_df.query(f'target=={row}').values.squeeze())
        fig.add_trace(data, row=row+1, col=col+1)
fig.update_layout(title='Count of Character Raito',
                  showlegend=False)
fig.show()


## === cell 16
train_df = train_df.drop(['f_27'], axis=1)
f_27_feature_df = f_27_feature_df.drop(['target'], axis=1)
train = pd.merge(train_df, f_27_feature_df, left_index=True, right_index=True)
train.columns


## === cell 17
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits)
train["k_folds"] = -1
for fold, (train_idx, valid_idx) in enumerate(skf.split(X=train, y=train["target"])):
    train["k_folds"][valid_idx] = fold

tree_method = "gpu_hist" if xgb.core._has_cuda_support() else "hist"

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
        objective="binary:logistic", tree_method=tree_method, seed=global_seed
    )
    model.fit(
        X_train,
        y_train,
        verbose=True,
        early_stopping_rounds=10,
        eval_metric="auc",
        eval_set=[(X_valid, y_valid)],
    )
    models.append(model)


## --- ERROR in cell 17, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2330607744.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      6[0m [0;34m[0m[0m
[1;32m      7[0m [0;31m# Fix: 'gpu_hist' requires a CUDA-capable device; fall back to CPU when no GPU is available.[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 8[0;31m [0mtree_method[0m [0;34m=[0m [0;34m"gpu_hist"[0m [0;32mif[0m [0mxgb[0m[0;34m.[0m[0mcore[0m[0;34m.[0m[0m_has_cuda_support[0m[0;34m([0m[0;34m)[0m [0;32melse[0m [0;34m"hist"[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      9[0m [0;34m[0m[0m
[1;32m     10[0m [0mmodels[0m [0;34m=[0m [0;34m[[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m

[0;31mAttributeError[0m: module 'xgboost.core' has no attribute '_has_cuda_support'

## === cell 18
fig, ax = plt.subplots(1,1,figsize=(20,12))
plot_importance(model, ax=ax, xlabel=None)
plt.title('XGB Feature importance', fontsize=20)
plt.show()
