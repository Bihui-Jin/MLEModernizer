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

geopandas==0.14.4
lightgbm==4.6.0
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
import gc
import os
import random

import lightgbm as lgb
import numpy as np
import pandas as pd
import seaborn as sns
import itertools

from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
from sklearn.cluster import KMeans

sns.set(style='darkgrid')
SEEDS = 42


## === cell 1
def rmse(y_true, y_pred):
    return (mean_squared_error(y_true, y_pred))** .5


## === cell 2
class TreeModel:
    def __init__(self, model_type):
        self.model_type = model_type
        self.tr_data = None
        self.vl_data = None
        self.model = None
    
    def train(self, params, train_x, train_y, valid_x=None, valid_y=None, num_round=None, early_stopping=None, verbose=None):
        if self.model_type == 'lgb':
            self.tr_data = lgb.Dataset(train_x, label=train_y)
            self.vl_data = lgb.Dataset(valid_x, label=valid_y)
            self.model = lgb.train(params, self.tr_data, valid_sets=[self.tr_data, self.vl_data],
                                   num_boost_round=num_round, early_stopping_rounds=early_stopping,verbose_eval=verbose)
            
        if self.model_type == 'rf_reg':
            self.train_x = train_x
            self.train_y = train_y
            self.model = RandomForestRegressor(**params).fit(self.train_x, self.train_y)
            
        if self.model_type == 'xgb':
            self.tr_data = xgb.DMatrix(train_x, train_y)
            self.vl_data = xgb.DMatrix(valid_x, valid_y)
            self.model = xgb.train(params, self.tr_data, num_boost_round=num_round,
                                   evals=[(self.tr_data, 'train'), (self.vl_data, 'val')], 
                                   verbose_eval=verbose, early_stopping_rounds=early_stopping)
            
        if self.model_type == 'cat':
            params['num_boost_round'] = num_round
            self.cat_cols = list(train_x.select_dtypes(include='object').columns)
            self.tr_data = Pool(train_x, train_y, cat_features=self.cat_cols)
            self.vl_data = Pool(valid_x, valid_y, cat_features=self.cat_cols)
            self.model = CatBoost(params).fit(self.tr_data, eval_set=self.vl_data,
                                                early_stopping_rounds=early_stopping, verbose=verbose, use_best_model=True)
            
            return self.model
            
    
    def predict(self,X):
        if self.model_type == 'lgb':
            return self.model.predict(X, num_iteration=self.model.best_iteration)
        
        if self.model_type == 'rf_reg':
            return self.model.predict(X)
        
        if self.model_type == 'xgb':
            X_DM = xgb.DMatrix(X)
            return self.model.predict(X_DM)
        
        if self.model_type == 'cat':
            X_pool = Pool(X, cat_features=self.cat_cols)
            return self.model.predict(X_pool)
    
    @property
    def feature_names_(self):
        if self.model_type == 'lgb':
            return self.model.feature_name()
        
        if self.model_type == 'rf_reg':
            return self.train_x.columns
        
        if self.model_type == 'xgb':
            return list(self.model.get_score(importance_type='gain').keys())
        
        if self.model_type == 'cat':
            return self.model.feature_names_
    
    @property
    def feature_importances_(self):
        if self.model_type == 'lgb':
            return self.model.feature_importance(importance_type='gain')
        
        if self.model_type == 'rf_reg':
            return self.model.feature_importances_
        
        if self.model_type == 'xgb':
            return list(self.model.get_score(importance_type='gain').values())
        
        if self.model_type == 'cat':
            return self.model.feature_importances_


## === cell 3
train = pd.read_json('../input/stanford-covid-vaccine/train.json',lines=True)
test = pd.read_json('../input/stanford-covid-vaccine/test.json', lines=True)
submission = pd.read_csv('/kaggle/input/stanford-covid-vaccine/sample_submission.csv')


## === cell 4
train_data = []
for mol_id in train['id'].unique():
    sample_data = train.loc[train['id'] == mol_id]
    sample_seq_length = sample_data.seq_length.values[0]
    
    for i in range(68):
        sample_dict = {'id' : sample_data['id'].values[0],
                       'id_seqpos' : sample_data['id'].values[0] + '_' + str(i),
                       'sequence' : sample_data['sequence'].values[0][i],
                       'structure' : sample_data['structure'].values[0][i],
                       'predicted_loop_type' : sample_data['predicted_loop_type'].values[0][i],
                       'reactivity' : sample_data['reactivity'].values[0][i],
                       'reactivity_error' : sample_data['reactivity_error'].values[0][i],
                       'deg_Mg_pH10' : sample_data['deg_Mg_pH10'].values[0][i],
                       'deg_error_Mg_pH10' : sample_data['deg_error_Mg_pH10'].values[0][i],
                       'deg_pH10' : sample_data['deg_pH10'].values[0][i],
                       'deg_error_pH10' : sample_data['deg_error_pH10'].values[0][i],
                       'deg_Mg_50C' : sample_data['deg_Mg_50C'].values[0][i],
                       'deg_error_Mg_50C' : sample_data['deg_error_Mg_50C'].values[0][i],
                       'deg_50C' : sample_data['deg_50C'].values[0][i],
                       'deg_error_50C' : sample_data['deg_error_50C'].values[0][i]}
        
        
        shifts = [1,2,3,4,5]
        shift_cols = ['sequence', 'structure', 'predicted_loop_type']
        for shift,col in itertools.product(shifts, shift_cols):
            if i - shift >= 0:
                sample_dict['b'+str(shift)+'_'+col] = sample_data[col].values[0][i-shift]
            else:
                sample_dict['b'+str(shift)+'_'+col] = -1
            
            if i + shift <= sample_seq_length - 1:
                sample_dict['a'+str(shift)+'_'+col] = sample_data[col].values[0][i+shift]
            else:
                sample_dict['a'+str(shift)+'_'+col] = -1
        
        
        train_data.append(sample_dict)
train_data = pd.DataFrame(train_data)
train_data.head()


## === cell 5
test_data = []
for mol_id in test['id'].unique():
    sample_data = test.loc[test['id'] == mol_id]
    sample_seq_length = sample_data.seq_length.values[0]
    for i in range(sample_seq_length):
        sample_dict = {'id' : sample_data['id'].values[0],
                       'id_seqpos' : sample_data['id'].values[0] + '_' + str(i),
                       'sequence' : sample_data['sequence'].values[0][i],
                       'structure' : sample_data['structure'].values[0][i],
                       'predicted_loop_type' : sample_data['predicted_loop_type'].values[0][i]}
        
        shifts = [1,2,3,4,5]
        shift_cols = ['sequence', 'structure', 'predicted_loop_type']
        for shift,col in itertools.product(shifts, shift_cols):
            if i - shift >= 0:
                sample_dict['b'+str(shift)+'_'+col] = sample_data[col].values[0][i-shift]
            else:
                sample_dict['b'+str(shift)+'_'+col] = -1
            
            if i + shift <= sample_seq_length - 1:
                sample_dict['a'+str(shift)+'_'+col] = sample_data[col].values[0][i+shift]
            else:
                sample_dict['a'+str(shift)+'_'+col] = -1
        
        test_data.append(sample_dict)
test_data = pd.DataFrame(test_data)
test_data.head()


## === cell 6
sequence_encmap = {'A': 0, 'G' : 1, 'C' : 2, 'U' : 3}
structure_encmap = {'.' : 0, '(' : 1, ')' : 2}
looptype_encmap = {'S':0, 'E':1, 'H':2, 'I':3, 'X':4, 'M':5, 'B':6}

enc_targets = ['sequence', 'structure', 'predicted_loop_type']
enc_maps = [sequence_encmap, structure_encmap, looptype_encmap]

for t,m in zip(enc_targets, enc_maps):
    for c in [c for c in train_data.columns if t in c]:
        train_data[c] = train_data[c].replace(m)
        test_data[c] = test_data[c].replace(m)


## === cell 7
not_use_cols = ['id', 'id_seqpos']
features = [c for c in test_data.columns if c not in not_use_cols]
targets = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']


## === cell 8
FOLD_N = 5
gkf = GroupKFold(n_splits=FOLD_N)


## === cell 9
params = {'objective': 'regression',
          'boosting': 'gbdt',
          'metric': 'rmse',
          'learning_rate': 0.1,
          'seed' : SEEDS}


## === cell 10
class TreeModel:
    def __init__(self, model_type):
        self.model_type = model_type
        self.tr_data = None
        self.vl_data = None
        self.model = None

    def train(
        self,
        params,
        train_x,
        train_y,
        valid_x=None,
        valid_y=None,
        num_round=None,
        early_stopping=None,
        verbose=None,
    ):
        if self.model_type == "lgb":
            self.tr_data = lgb.Dataset(train_x, label=train_y)
            self.vl_data = lgb.Dataset(valid_x, label=valid_y)

            callbacks = []
            if early_stopping is not None:
                callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping))
            if verbose is not None:
                callbacks.append(lgb.log_evaluation(period=verbose))

            self.model = lgb.train(
                params,
                self.tr_data,
                valid_sets=[self.tr_data, self.vl_data],
                num_boost_round=num_round,
                callbacks=callbacks,
            )

        if self.model_type == "rf_reg":
            self.train_x = train_x
            self.train_y = train_y
            self.model = RandomForestRegressor(**params).fit(self.train_x, self.train_y)

        if self.model_type == "xgb":
            self.tr_data = xgb.DMatrix(train_x, train_y)
            self.vl_data = xgb.DMatrix(valid_x, valid_y)
            self.model = xgb.train(
                params,
                self.tr_data,
                num_boost_round=num_round,
                evals=[(self.tr_data, "train"), (self.vl_data, "val")],
                verbose_eval=verbose,
                early_stopping_rounds=early_stopping,
            )

        if self.model_type == "cat":
            params["num_boost_round"] = num_round
            self.cat_cols = list(train_x.select_dtypes(include="object").columns)
            self.tr_data = Pool(train_x, train_y, cat_features=self.cat_cols)
            self.vl_data = Pool(valid_x, valid_y, cat_features=self.cat_cols)
            self.model = CatBoost(params).fit(
                self.tr_data,
                eval_set=self.vl_data,
                early_stopping_rounds=early_stopping,
                verbose=verbose,
                use_best_model=True,
            )

            return self.model

    def predict(self, X):
        if self.model_type == "lgb":
            return self.model.predict(X, num_iteration=self.model.best_iteration)

        if self.model_type == "rf_reg":
            return self.model.predict(X)

        if self.model_type == "xgb":
            X_DM = xgb.DMatrix(X)
            return self.model.predict(X_DM)

        if self.model_type == "cat":
            X_pool = Pool(X, cat_features=self.cat_cols)
            return self.model.predict(X_pool)

    @property
    def feature_names_(self):
        if self.model_type == "lgb":
            return self.model.feature_name()

        if self.model_type == "rf_reg":
            return self.train_x.columns

        if self.model_type == "xgb":
            return list(self.model.get_score(importance_type="gain").keys())

        if self.model_type == "cat":
            return self.model.feature_names_

    @property
    def feature_importances_(self):
        if self.model_type == "lgb":
            return self.model.feature_importance(importance_type="gain")

        if self.model_type == "rf_reg":
            return self.model.feature_importances_

        if self.model_type == "xgb":
            return list(self.model.get_score(importance_type="gain").values())

        if self.model_type == "cat":
            return self.model.feature_importances_


## === cell 11
if "result" in globals():
    display(result)
    display(f"total : {np.mean(list(result.values()))}")
else:
    display("result is not defined (previous evaluation cell may not have been run).")


## === cell 12
if "feature_importances" not in globals():
    feature_importances = pd.DataFrame(columns=["target", "feature", "importance"])

for target in targets:
    tmp = feature_importances[feature_importances.target == target]
    if tmp.empty:
        continue

    order = list(
        tmp.groupby("feature")
        .mean(numeric_only=True)
        .sort_values("importance", ascending=False)
        .index
    )

    plt.figure(figsize=(10, 5))
    sns.barplot(x="importance", y="feature", data=tmp, order=order)
    plt.title(target)
    plt.tight_layout()


## === cell 13
if "oof_df" in globals():
    display(oof_df.head())
else:
    display(
        "oof_df is not defined (OOF generation/training cell may not have been run)."
    )


## === cell 14
submission.head()


## === cell 15
if "oof_df" in globals():
    display(oof_df.shape)
else:
    display(
        "oof_df is not defined (OOF generation/training cell may not have been run)."
    )

display(submission.shape)


## === cell 16
oof_df.to_csv('oof_df.csv', index=False)
submission.to_csv('submission.csv', index=False)


## --- ERROR in cell 16, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mNameError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2882390106.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0moof_df[0m[0;34m.[0m[0mto_csv[0m[0;34m([0m[0;34m'oof_df.csv'[0m[0;34m,[0m [0mindex[0m[0;34m=[0m[0;32mFalse[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0msubmission[0m[0;34m.[0m[0mto_csv[0m[0;34m([0m[0;34m'submission.csv'[0m[0;34m,[0m [0mindex[0m[0;34m=[0m[0;32mFalse[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mNameError[0m: name 'oof_df' is not defined
