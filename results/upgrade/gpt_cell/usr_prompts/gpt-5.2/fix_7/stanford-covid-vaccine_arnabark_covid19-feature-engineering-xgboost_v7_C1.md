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

3.8

# 2. Installed packages

catboost==1.2.8
geopandas==0.14.4
imbalanced-learn==0.13.0
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
nltk==3.9.2
numpy==1.26.4
optuna==4.5.0
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
seaborn==0.12.2
sklearn-pandas==2.2.0
statsmodels==0.14.5
tqdm==4.67.1
xgboost==2.0.3

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (125 lines)
            sample_submission.csv (25681 lines)
            sample_submission.csv.zip (74.8 kB)
            test.json (240 lines)
            train.json (2160 lines)
            stanford-covid-vaccine/
                description.md (125 lines)
                sample_submission.csv (25681 lines)
                ... and 3 other files
                stanford-covid-vaccine/
        input/
            description.md (125 lines)
            sample_submission.csv (25681 lines)
            sample_submission.csv.zip (74.8 kB)
            test.json (240 lines)
            train.json (2160 lines)
            stanford-covid-vaccine/
                description.md (125 lines)
                sample_submission.csv (25681 lines)
                ... and 3 other files
                stanford-covid-vaccine/
        working/
            stanford-covid-vaccine/
                description.md (125 lines)
                sample_submission.csv (25681 lines)
                ... and 3 other files
                stanford-covid-vaccine/
```

-> data/sample_submission.csv has 25680 rows and 6 columns.
The columns are: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

-> data/stanford-covid-vaccine/sample_submission.csv has 25680 rows and 6 columns.
The columns are: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

-> data/stanford-covid-vaccine/test.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "index": {
      "type": "integer"
    },
    "id": {
      "type": "string"
    },
    "sequence": {
      "type": "string"
    },
    "structure": {
      "type": "string"
    },
    "predicted_loop_type": {
      "type": "string"
    },
    "seq_length": {
      "type": "integer"
    },
    "seq_scored": {
      "type": "integer"
    }
  },
  "required": [
    "id",
    "index",
    "predicted_loop_type",
    "seq_length",
    "seq_scored",
    "sequence",
    "structure"
  ]
}

-> data/stanford-covid-vaccine/train.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "index": {
      "type": "integer"
    },
    "id": {
      "type": "string"
    },
    "sequence": {
      "type": "string"
    },
    "structure": {
      "type": "string"
    },
    "predicted_loop_type": {
      "type": "string"
    },
    "signal_to_noise": {
      "type": "number"
    },
    "SN_filter": {
      "type": "integer"
    },
    "seq_length": {
      "type": "integer"
    },
    "seq_scored": {
      "type": "integer"
    },
    "reactivity_error": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_Mg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_Mg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "reactivity": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_Mg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_Mg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    }
  },
  "required": [
    "SN_filter",
    "deg_50C",
    "deg_Mg_50C",
    "deg_Mg_pH10",
    "deg_error_50C",
    "deg_error_Mg_50C",
    "deg_error_Mg_pH10",
    "deg_error_pH10",
    "deg_pH10",
    "id",
    "index",
    "predicted_loop_type",
    "reactivity",
    "reactivity_error",
    "seq_length",
    "seq_scored",
    "sequence",
    "signal_to_noise",
    "structure"
  ]
}

-> data/test.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "index": {
      "type": "integer"
    },
    "id": {
      "type": "string"
    },
    "sequence": {
      "type": "string"
    },
    "structure": {
      "type": "string"
    },
    "predicted_loop_type": {
      "type": "string"
    },
    "seq_length": {
      "type": "integer"
    },
    "seq_scored": {
      "type": "integer"
    }
  },
  "required": [
    "id",
    "index",
    "predicted_loop_type",
    "seq_length",
    "seq_scored",
    "sequence",
    "structure"
  ]
}

-> data/train.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "index": {
      "type": "integer"
    },
    "id": {
      "type": "string"
    },
    "sequence": {
      "type": "string"
    },
    "structure": {
      "type": "string"
    },
    "predicted_loop_type": {
      "type": "string"
    },
    "signal_to_noise": {
      "type": "number"
    },
    "SN_filter": {
      "type": "integer"
    },
    "seq_length": {
      "type": "integer"
    },
    "seq_scored": {
      "type": "integer"
    },
    "reactivity_error": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_Mg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_Mg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "reactivity": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_Mg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_Mg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    }
  },
  "required": [
    "SN_filter",
    "deg_50C",
    "deg_Mg_50C",
    "deg_Mg_pH10",
    "deg_error_50C",
    "deg_error_Mg_50C",
    "deg_error_Mg_pH10",
    "deg_error_pH10",
    "deg_pH10",
    "id",
    "index",
    "predicted_loop_type",
    "reactivity",
    "reactivity_error",
    "seq_length",
    "seq_scored",
    "sequence",
    "signal_to_noise",
    "structure"
  ]
}

-> input/sample_submission.csv has 25680 rows and 6 columns.
The columns are: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import catboost
import optuna

try:
    import imblearn
    from imblearn.under_sampling import RandomUnderSampler
except Exception:
    imblearn = None
    RandomUnderSampler = None

from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
from catboost import *
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import Pool
from datetime import datetime
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from scipy.stats import norm, skew
from scipy import stats
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
import pandas as pd
import nltk
import operator
import re
import sys
from scipy import stats
from nltk.corpus import stopwords
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download("stopwords")
nltk.download("punkt")
import statsmodels.api as sm
from statsmodels.formula.api import ols
import time


## === cell 2
train = pd.read_json('../input/stanford-covid-vaccine/train.json',lines=True)
test = pd.read_json('../input/stanford-covid-vaccine/test.json', lines=True)
ss = pd.read_csv('../input/stanford-covid-vaccine/sample_submission.csv')
train.shape, test.shape, ss.shape


## === cell 3
train


## === cell 4
ss


## === cell 5
test['structure'].value_counts()


## === cell 6
train.columns


## === cell 7
train['deg_Mg_pH10']


## === cell 8
ss.columns


## === cell 10
test['E']=[sum([i=='E' for i in j])/len(j) for j in test['predicted_loop_type']]
test['S']=[sum([i=='S' for i in j])/len(j) for j in test['predicted_loop_type']]
test['B']=[sum([i=='B' for i in j])/len(j) for j in test['predicted_loop_type']]
test['H']=[sum([i=='H' for i in j])/len(j) for j in test['predicted_loop_type']]
test['I']=[sum([i=='I' for i in j])/len(j) for j in test['predicted_loop_type']]

test['G']=[sum([i=='G' for i in j])/len(j) for j in test['sequence']]
test['A']=[sum([i=='A' for i in j])/len(j) for j in test['sequence']]
test['C']=[sum([i=='C' for i in j])/len(j) for j in test['sequence']]
test['U']=[sum([i=='U' for i in j])/len(j) for j in test['sequence']]
test['Paired']=[sum([i=='(' or i==')' for i in j]) for j in test['structure']]
test['Unpaired']=[sum([i=='.' for i in j]) for j in test['structure']]


## === cell 11
train['E']=[sum([i=='E' for i in j])/len(j) for j in train['predicted_loop_type']]
train['S']=[sum([i=='S' for i in j])/len(j) for j in train['predicted_loop_type']]
train['B']=[sum([i=='B' for i in j])/len(j) for j in train['predicted_loop_type']]
train['H']=[sum([i=='H' for i in j])/len(j) for j in train['predicted_loop_type']]
train['I']=[sum([i=='I' for i in j])/len(j) for j in train['predicted_loop_type']]

train['G']=[sum([i=='G' for i in j])/len(j) for j in train['sequence']]
train['A']=[sum([i=='A' for i in j])/len(j) for j in train['sequence']]
train['C']=[sum([i=='C' for i in j])/len(j) for j in train['sequence']]
train['U']=[sum([i=='U' for i in j])/len(j) for j in train['sequence']]
train['Paired']=[sum([i=='(' or i==')' for i in j]) for j in train['structure']]
train['Unpaired']=[sum([i=='.' for i in j]) for j in train['structure']]


## === cell 12
train.columns


## === cell 16
for a in [ 'G', 'A', 'C', 'U']:
    train[a+'_position']=[np.sum([i for i in range(len(j)) if j[i]==a])/len([i for i in range(len(j)) if j[i]==a]) for j in train['sequence']]
    test[a+'_position']=[np.sum([i for i in range(len(j)) if j[i]==a])/len([i for i in range(len(j)) if j[i]==a]) for j in test['sequence']]


## === cell 17
for a in [ 'E', 'S', 'H',]:
    train[a+'_position']=[np.sum([i for i in range(len(j)) if j[i]==a])/len([i for i in range(len(j)) if j[i]==a]) for j in train['predicted_loop_type']]
    test[a+'_position']=[np.sum([i for i in range(len(j)) if j[i]==a])/len([i for i in range(len(j)) if j[i]==a]) for j in test['predicted_loop_type']]


## === cell 18
for a in [ 'E', 'S', 'H',]:
    train[a+'']=[np.sum([i for i in range(len(j)) if j[i]==a])/len([i for i in range(len(j)) if j[i]==a]) for j in train['predicted_loop_type']]
    test[a+'_position']=[np.sum([i for i in range(len(j)) if j[i]==a])/len([i for i in range(len(j)) if j[i]==a]) for j in test['predicted_loop_type']]


## === cell 19
a='S'
[np.sum([i for i in range(len(j)) if j[i]==a])/len([i for i in range(len(j)) if j[i]==a]) for j in train['predicted_loop_type']]


## === cell 21
train.head()


## === cell 22
test['predicted_loop_type'][0:100]


## === cell 23
train.columns
    


## === cell 24
train.head()


## === cell 25
test.columns


## === cell 26
train['seq_length'][0]


## === cell 27
train_ex = pd.DataFrame()
for index in train.index:
    temp = pd.DataFrame()
    temp["id_seqpos"] = [
        str(str(train["id"][index]) + "_" + str(i))
        for i in range(train["seq_scored"][index])
    ]
    temp["sequence"] = [
        train["sequence"][index][i] for i in range(train["seq_scored"][index])
    ]
    temp["structure"] = [
        train["structure"][index][i] for i in range(train["seq_scored"][index])
    ]
    temp["predicted_loop_type"] = [
        train["predicted_loop_type"][index][i]
        for i in range(train["seq_scored"][index])
    ]
    temp["E"] = train["E"][index]
    temp["S"] = train["S"][index]
    temp["B"] = train["B"][index]
    temp["H"] = train["H"][index]
    temp["I"] = train["I"][index]
    temp["G"] = train["G"][index]
    temp["A"] = train["A"][index]
    temp["C"] = train["C"][index]
    temp["U"] = train["U"][index]
    temp["Paired"] = train["Paired"][index]
    temp["Unpaired"] = train["Unpaired"][index]
    temp["G_position"] = train["G_position"][index]
    temp["A_position"] = train["A_position"][index]
    temp["C_position"] = train["C_position"][index]
    temp["U_position"] = train["U_position"][index]
    temp["E_position"] = train["E_position"][index]
    temp["S_position"] = train["S_position"][index]
    temp["H_position"] = train["H_position"][index]
    temp["G"] = train["G"][index]
    temp["A"] = train["A"][index]
    temp["C"] = train["C"][index]
    temp["U"] = train["U"][index]
    temp["reactivity"] = [
        train["reactivity"][index][i] for i in range(train["seq_scored"][index])
    ]
    temp["deg_Mg_pH10"] = [
        train["deg_Mg_pH10"][index][i] for i in range(train["seq_scored"][index])
    ]
    temp["deg_pH10"] = [
        train["deg_pH10"][index][i] for i in range(train["seq_scored"][index])
    ]
    temp["deg_Mg_50C"] = [
        train["deg_Mg_50C"][index][i] for i in range(train["seq_scored"][index])
    ]
    temp["deg_50C"] = [
        train["deg_50C"][index][i] for i in range(train["seq_scored"][index])
    ]
    train_ex = pd.concat([train_ex, temp], ignore_index=True)


## === cell 29
test_ex = pd.DataFrame()
for index in test.index:
    temp = pd.DataFrame()
    temp["id_seqpos"] = [
        str(str(test["id"][index]) + "_" + str(i))
        for i in range(test["seq_length"][index])
    ]
    temp["sequence"] = [
        test["sequence"][index][i] for i in range(test["seq_length"][index])
    ]
    temp["structure"] = [
        test["structure"][index][i] for i in range(test["seq_length"][index])
    ]
    temp["predicted_loop_type"] = [
        test["predicted_loop_type"][index][i] for i in range(test["seq_length"][index])
    ]
    temp["E"] = test["E"][index]
    temp["S"] = test["S"][index]
    temp["B"] = test["B"][index]
    temp["H"] = test["H"][index]
    temp["I"] = test["I"][index]
    temp["G"] = test["G"][index]
    temp["A"] = test["A"][index]
    temp["C"] = test["C"][index]
    temp["U"] = test["U"][index]
    temp["Paired"] = test["Paired"][index]
    temp["Unpaired"] = test["Unpaired"][index]
    temp["G_position"] = test["G_position"][index]
    temp["A_position"] = test["A_position"][index]
    temp["C_position"] = test["C_position"][index]
    temp["U_position"] = test["U_position"][index]
    temp["E_position"] = test["E_position"][index]
    temp["S_position"] = test["S_position"][index]
    temp["H_position"] = test["H_position"][index]
    test_ex = pd.concat([test_ex, temp], ignore_index=True)


## === cell 30
train_ex=train_ex.replace(-1,0)


## === cell 32
result=test_ex


## === cell 33
x_test=test_ex[[i for i in test_ex.columns if i!='id_seqpos']]
x_t=train_ex[x_test.columns]
x_test=pd.get_dummies(x_test)
x_t=pd.get_dummies(x_t)
y_t=train_ex[['reactivity', 'deg_Mg_pH10',
       'deg_pH10', 'deg_Mg_50C', 'deg_50C']]
x_t=x_t[x_test.columns]


## === cell 34
x_t.columns


## === cell 37
x_train,x_valid,y_train,y_valid=train_test_split(x_t,y_t,test_size=0.1,shuffle=True)


## === cell 38
import optuna
import xgboost as xgb
import sklearn
column='reactivity'
def objective(trial):
    dtrain = xgb.DMatrix(x_train.values, label=y_train[column])
    dvalid = xgb.DMatrix(x_valid.values, label=y_valid[column])

    param = {
        "silent": 1,
          "eval_metric": "rmse",
        "booster": "gbtree",
        "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
        "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),
        'tree_method' : 'gpu_hist'
        
    }

    if param["booster"] == "gbtree" or param["booster"] == "dart":
        param["max_depth"] = trial.suggest_int("max_depth", 1, 9)


    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, str("validation-"+param["eval_metric"]))
    bst = xgb.train(param, dtrain, evals=[(dvalid, "validation")],callbacks=[pruning_callback])
    preds = bst.predict(dvalid)
    rmse=np.sqrt(sklearn.metrics.mean_squared_error(y_valid[column], preds))
    return rmse


## === cell 39
try:
    from optuna_integration.xgboost import XGBoostPruningCallback  # type: ignore
except Exception:
    XGBoostPruningCallback = None

column = "reactivity"


def _pick_tree_method():
    try:
        has_cuda = bool(getattr(xgb.core, "_has_cuda_support", lambda: False)())
    except Exception:
        has_cuda = False
    return "gpu_hist" if has_cuda else "hist"


def objective(trial):
    dtrain = xgb.DMatrix(x_train.values, label=y_train[column])
    dvalid = xgb.DMatrix(x_valid.values, label=y_valid[column])

    param = {
        "silent": 1,
        "eval_metric": "rmse",
        "booster": "gbtree",
        "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
        "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),
        "tree_method": _pick_tree_method(),
    }

    if param["booster"] == "gbtree" or param["booster"] == "dart":
        param["max_depth"] = trial.suggest_int("max_depth", 1, 9)

    callbacks = []
    if XGBoostPruningCallback is not None:
        callbacks.append(
            XGBoostPruningCallback(trial, "validation-" + param["eval_metric"])
        )

    bst = xgb.train(param, dtrain, evals=[(dvalid, "validation")], callbacks=callbacks)
    preds = bst.predict(dvalid)
    rmse = np.sqrt(sklearn.metrics.mean_squared_error(y_valid[column], preds))
    return rmse


study = optuna.create_study()
study.optimize(objective, n_trials=100)


## === cell 40
print(study.best_params)


## === cell 42
dtrain = xgb.DMatrix(x_train.values, label=y_train[column])
dvalid = xgb.DMatrix(x_valid.values, label=y_valid[column])
dtest = xgb.DMatrix(x_test.values)
bst = xgb.train( study.best_params,dtrain, evals=[(dvalid, "validation")])
preds = bst.predict(dtest)
result[column]=preds


## === cell 43
import optuna
import xgboost as xgb
import sklearn
column='deg_Mg_pH10'
def objective(trial):
    column='deg_Mg_pH10'
    dtrain = xgb.DMatrix(x_train.values, label=y_train[column])
    dvalid = xgb.DMatrix(x_valid.values, label=y_valid[column])

    param = {
        "silent": 1,
          "eval_metric": "rmse",
        "booster": "gbtree",
        "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
        "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),
        'tree_method' : 'gpu_hist'
        
    }

    if param["booster"] == "gbtree" or param["booster"] == "dart":
        param["max_depth"] = trial.suggest_int("max_depth", 1, 9)


    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, str("validation-"+param["eval_metric"]))
    bst = xgb.train(param, dtrain, evals=[(dvalid, "validation")],callbacks=[pruning_callback])
    preds = bst.predict(dvalid)
    rmse=np.sqrt(sklearn.metrics.mean_squared_error(y_valid[column], preds))
    return rmse


## === cell 44
try:
    from optuna_integration.xgboost import XGBoostPruningCallback  # type: ignore
except Exception:
    XGBoostPruningCallback = None

study = optuna.create_study()


def _pick_tree_method():
    try:
        has_cuda = bool(getattr(xgb.core, "_has_cuda_support", lambda: False)())
    except Exception:
        has_cuda = False
    return "gpu_hist" if has_cuda else "hist"


def objective(trial):
    column = "deg_Mg_pH10"
    dtrain = xgb.DMatrix(x_train.values, label=y_train[column])
    dvalid = xgb.DMatrix(x_valid.values, label=y_valid[column])

    param = {
        "silent": 1,
        "eval_metric": "rmse",
        "booster": "gbtree",
        "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
        "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),
        "tree_method": _pick_tree_method(),
    }

    if param["booster"] == "gbtree" or param["booster"] == "dart":
        param["max_depth"] = trial.suggest_int("max_depth", 1, 9)

    callbacks = []
    if XGBoostPruningCallback is not None:
        callbacks.append(
            XGBoostPruningCallback(trial, "validation-" + param["eval_metric"])
        )

    bst = xgb.train(param, dtrain, evals=[(dvalid, "validation")], callbacks=callbacks)
    preds = bst.predict(dvalid)
    rmse = np.sqrt(sklearn.metrics.mean_squared_error(y_valid[column], preds))
    return rmse


study.optimize(objective, n_trials=100)


## === cell 45
study.best_params


## === cell 46
dtrain = xgb.DMatrix(x_train.values, label=y_train[column])
dvalid = xgb.DMatrix(x_valid.values, label=y_valid[column])
dtest = xgb.DMatrix(x_test.values)
bst = xgb.train( study.best_params,dtrain, evals=[(dvalid, "validation")])
preds = bst.predict(dtest)
result[column]=preds


## === cell 47
import optuna
import xgboost as xgb
import sklearn
column='deg_Mg_50C'
def objective(trial):
    column='deg_Mg_50C'
    dtrain = xgb.DMatrix(x_train.values, label=y_train[column])
    dvalid = xgb.DMatrix(x_valid.values, label=y_valid[column])

    param = {
        "silent": 1,
          "eval_metric": "rmse",
        "booster": "gbtree",
        "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
        "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),
        'tree_method' : 'gpu_hist'
        
    }

    if param["booster"] == "gbtree" or param["booster"] == "dart":
        param["max_depth"] = trial.suggest_int("max_depth", 1, 9)


    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, str("validation-"+param["eval_metric"]))
    bst = xgb.train(param, dtrain, evals=[(dvalid, "validation")],callbacks=[pruning_callback])
    preds = bst.predict(dvalid)
    rmse=np.sqrt(sklearn.metrics.mean_squared_error(y_valid[column], preds))
    return rmse


## === cell 48
study = optuna.create_study()
study.optimize(objective, n_trials=100)


## --- ERROR in cell 48, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
[0;32m/usr/local/lib/python3.11/dist-packages/optuna/integration/xgboost.py[0m in [0;36m<module>[0;34m[0m
[1;32m      4[0m [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 5[0;31m     [0;32mfrom[0m [0moptuna_integration[0m[0;34m.[0m[0mxgboost[0m [0;32mimport[0m [0mXGBoostPruningCallback[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      6[0m [0;32mexcept[0m [0mModuleNotFoundError[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;31mModuleNotFoundError[0m: No module named 'optuna_integration'

During handling of the above exception, another exception occurred:

[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
[0;32m/usr/local/lib/python3.11/dist-packages/optuna/integration/__init__.py[0m in [0;36m_get_module[0;34m(self, module_name)[0m
[1;32m    131[0m             [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 132[0;31m                 [0;32mreturn[0m [0mimportlib[0m[0;34m.[0m[0mimport_module[0m[0;34m([0m[0;34m"."[0m [0;34m+[0m [0mmodule_name[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0m__name__[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    133[0m             [0;32mexcept[0m [0mModuleNotFoundError[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/importlib/__init__.py[0m in [0;36mimport_module[0;34m(name, package)[0m
[1;32m    125[0m             [0mlevel[0m [0;34m+=[0m [0;36m1[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 126[0;31m     [0;32mreturn[0m [0m_bootstrap[0m[0;34m.[0m[0m_gcd_import[0m[0;34m([0m[0mname[0m[0;34m[[0m[0mlevel[0m[0;34m:[0m[0;34m][0m[0;34m,[0m [0mpackage[0m[0;34m,[0m [0mlevel[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    127[0m [0;34m[0m[0m

[0;32m/usr/lib/python3.11/importlib/_bootstrap.py[0m in [0;36m_gcd_import[0;34m(name, package, level)[0m

[0;32m/usr/lib/python3.11/importlib/_bootstrap.py[0m in [0;36m_find_and_load[0;34m(name, import_)[0m

[0;32m/usr/lib/python3.11/importlib/_bootstrap.py[0m in [0;36m_find_and_load_unlocked[0;34m(name, import_)[0m

[0;32m/usr/lib/python3.11/importlib/_bootstrap.py[0m in [0;36m_load_unlocked[0;34m(spec)[0m

[0;32m/usr/lib/python3.11/importlib/_bootstrap_external.py[0m in [0;36mexec_module[0;34m(self, module)[0m

[0;32m/usr/lib/python3.11/importlib/_bootstrap.py[0m in [0;36m_call_with_frames_removed[0;34m(f, *args, **kwds)[0m

[0;32m/usr/local/lib/python3.11/dist-packages/optuna/integration/xgboost.py[0m in [0;36m<module>[0;34m[0m
[1;32m      6[0m [0;32mexcept[0m [0mModuleNotFoundError[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 7[0;31m     [0;32mraise[0m [0mModuleNotFoundError[0m[0;34m([0m[0m_INTEGRATION_IMPORT_ERROR_TEMPLATE[0m[0;34m.[0m[0mformat[0m[0;34m([0m[0;34m"xgboost"[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      8[0m [0;34m[0m[0m

[0;31mModuleNotFoundError[0m: 
Could not find `optuna-integration` for `xgboost`.
Please run `pip install optuna-integration[xgboost]`.

During handling of the above exception, another exception occurred:

[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3066454264.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      1[0m [0mstudy[0m [0;34m=[0m [0moptuna[0m[0;34m.[0m[0mcreate_study[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 2[0;31m [0mstudy[0m[0;34m.[0m[0moptimize[0m[0;34m([0m[0mobjective[0m[0;34m,[0m [0mn_trials[0m[0;34m=[0m[0;36m100[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;32m/usr/local/lib/python3.11/dist-packages/optuna/study/study.py[0m in [0;36moptimize[0;34m(self, func, n_trials, timeout, n_jobs, catch, callbacks, gc_after_trial, show_progress_bar)[0m
[1;32m    488[0m                 [0mIf[0m [0mnested[0m [0minvocation[0m [0mof[0m [0mthis[0m [0mmethod[0m [0moccurs[0m[0;34m.[0m[0;34m[0m[0;34m[0m[0m
[1;32m    489[0m         """
[0;32m--> 490[0;31m         _optimize(
[0m[1;32m    491[0m             [0mstudy[0m[0;34m=[0m[0mself[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    492[0m             [0mfunc[0m[0;34m=[0m[0mfunc[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/optuna/study/_optimize.py[0m in [0;36m_optimize[0;34m(study, func, n_trials, timeout, n_jobs, catch, callbacks, gc_after_trial, show_progress_bar)[0m
[1;32m     61[0m     [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     62[0m         [0;32mif[0m [0mn_jobs[0m [0;34m==[0m [0;36m1[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 63[0;31m             _optimize_sequential(
[0m[1;32m     64[0m                 [0mstudy[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     65[0m                 [0mfunc[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/optuna/study/_optimize.py[0m in [0;36m_optimize_sequential[0;34m(study, func, n_trials, timeout, catch, callbacks, gc_after_trial, reseed_sampler_rng, time_start, progress_bar)[0m
[1;32m    158[0m [0;34m[0m[0m
[1;32m    159[0m         [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 160[0;31m             [0mfrozen_trial_id[0m [0;34m=[0m [0m_run_trial[0m[0;34m([0m[0mstudy[0m[0;34m,[0m [0mfunc[0m[0;34m,[0m [0mcatch[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    161[0m         [0;32mfinally[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    162[0m             [0;31m# The following line mitigates memory problems that can be occurred in some[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/optuna/study/_optimize.py[0m in [0;36m_run_trial[0;34m(study, func, catch)[0m
[1;32m    256[0m         [0;32mand[0m [0;32mnot[0m [0misinstance[0m[0;34m([0m[0mfunc_err[0m[0;34m,[0m [0mcatch[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    257[0m     ):
[0;32m--> 258[0;31m         [0;32mraise[0m [0mfunc_err[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    259[0m     [0;32mreturn[0m [0mtrial[0m[0;34m.[0m[0m_trial_id[0m[0;34m[0m[0;34m[0m[0m
[1;32m    260[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/optuna/study/_optimize.py[0m in [0;36m_run_trial[0;34m(study, func, catch)[0m
[1;32m    199[0m     [0;32mwith[0m [0mget_heartbeat_thread[0m[0;34m([0m[0mtrial[0m[0;34m.[0m[0m_trial_id[0m[0;34m,[0m [0mstudy[0m[0;34m.[0m[0m_storage[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    200[0m         [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 201[0;31m             [0mvalue_or_values[0m [0;34m=[0m [0mfunc[0m[0;34m([0m[0mtrial[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    202[0m         [0;32mexcept[0m [0mexceptions[0m[0;34m.[0m[0mTrialPruned[0m [0;32mas[0m [0me[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    203[0m             [0;31m# TODO(mamu): Handle multi-objective cases.[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/3947453071.py[0m in [0;36mobjective[0;34m(trial)[0m
[1;32m     27[0m [0;34m[0m[0m
[1;32m     28[0m     [0;31m# Add a callback for pruning.[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 29[0;31m     [0mpruning_callback[0m [0;34m=[0m [0moptuna[0m[0;34m.[0m[0mintegration[0m[0;34m.[0m[0mXGBoostPruningCallback[0m[0;34m([0m[0mtrial[0m[0;34m,[0m [0mstr[0m[0;34m([0m[0;34m"validation-"[0m[0;34m+[0m[0mparam[0m[0;34m[[0m[0;34m"eval_metric"[0m[0;34m][0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     30[0m     [0mbst[0m [0;34m=[0m [0mxgb[0m[0;34m.[0m[0mtrain[0m[0;34m([0m[0mparam[0m[0;34m,[0m [0mdtrain[0m[0;34m,[0m [0mevals[0m[0;34m=[0m[0;34m[[0m[0;34m([0m[0mdvalid[0m[0;34m,[0m [0;34m"validation"[0m[0;34m)[0m[0;34m][0m[0;34m,[0m[0mcallbacks[0m[0;34m=[0m[0;34m[[0m[0mpruning_callback[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     31[0m     [0mpreds[0m [0;34m=[0m [0mbst[0m[0;34m.[0m[0mpredict[0m[0;34m([0m[0mdvalid[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/optuna/integration/__init__.py[0m in [0;36m__getattr__[0;34m(self, name)[0m
[1;32m    118[0m                 [0mvalue[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_get_module[0m[0;34m([0m[0mname[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    119[0m             [0;32melif[0m [0mname[0m [0;32min[0m [0mself[0m[0;34m.[0m[0m_class_to_module[0m[0;34m.[0m[0mkeys[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 120[0;31m                 [0mmodule[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_get_module[0m[0;34m([0m[0mself[0m[0;34m.[0m[0m_class_to_module[0m[0;34m[[0m[0mname[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    121[0m                 [0mvalue[0m [0;34m=[0m [0mgetattr[0m[0;34m([0m[0mmodule[0m[0;34m,[0m [0mname[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    122[0m             [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/optuna/integration/__init__.py[0m in [0;36m_get_module[0;34m(self, module_name)[0m
[1;32m    132[0m                 [0;32mreturn[0m [0mimportlib[0m[0;34m.[0m[0mimport_module[0m[0;34m([0m[0;34m"."[0m [0;34m+[0m [0mmodule_name[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0m__name__[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    133[0m             [0;32mexcept[0m [0mModuleNotFoundError[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 134[0;31m                 [0;32mraise[0m [0mModuleNotFoundError[0m[0;34m([0m[0m_INTEGRATION_IMPORT_ERROR_TEMPLATE[0m[0;34m.[0m[0mformat[0m[0;34m([0m[0mmodule_name[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    135[0m [0;34m[0m[0m
[1;32m    136[0m     [0msys[0m[0;34m.[0m[0mmodules[0m[0;34m[[0m[0m__name__[0m[0;34m][0m [0;34m=[0m [0m_IntegrationModule[0m[0;34m([0m[0m__name__[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mModuleNotFoundError[0m: 
Could not find `optuna-integration` for `xgboost`.
Please run `pip install optuna-integration[xgboost]`.

## === cell 49
study.best_params
