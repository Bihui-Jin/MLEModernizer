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

catboost==1.2.8
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

0.91948

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'I fix the correlation heatmap crash by computing correlations on numeric columns only (the dataset contains a string categorical feature). I fix the CatBoost GPU crash by switching training to CPU while keeping the same model/core training loop, so it runs in this Kaggle environment. I also ensure we generate test predictions by averaging the 5 fold models (same CV approach, just making inference consistent) and then write a valid `submission.csv` with `id,target`. These changes are necessary for end-to-end execution and should yield a reasonable AUC without altering the intended modeling approach.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

RANDOM_STATE = 42



## === cell 1
train = pd.read_csv(r"../input/tabular-playground-series-may-2022/train.csv")
train.head()



## === cell 2
test = pd.read_csv(r"../input/tabular-playground-series-may-2022/test.csv")
test.head()



## === cell 3
sub = pd.read_csv(r"../input/tabular-playground-series-may-2022/sample_submission.csv")
sub.head()



## === cell 4
train_id = train["id"].copy()
test_id = test["id"].copy()

train.drop("id", axis=1, inplace=True)
test.drop("id", axis=1, inplace=True)



## === cell 5
print(f"train set have {train.shape[0]} rows and {train.shape[1]} columns.")
print(f"test set have {test.shape[0]} rows and {test.shape[1]} columns.")
print(f"sample_submission set have {sub.shape[0]} rows and {sub.shape[1]} columns.")



## === cell 6
train.isnull().sum()



## === cell 7
train.describe().T



## === cell 8
plt.figure(figsize=(24, 20))
corr = train.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, annot=False, cmap="YlGnBu")
plt.show()



## === cell 9
cat = ["f_27"]
X = train.drop("target", axis=1)
y = train["target"]



## === cell 10
from sklearn.model_selection import KFold
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

folds = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

oof = np.zeros(len(X), dtype=float)
test_pred = np.zeros(len(test), dtype=float)

for fold, (trn_idx, val_idx) in enumerate(folds.split(X)):
    print(f"Fold: {fold}")
    X_train, X_valid = X.iloc[trn_idx], X.iloc[val_idx]
    y_train, y_valid = y.iloc[trn_idx], y.iloc[val_idx]

    model = CatBoostClassifier(
        n_estimators=1500,
        cat_features=cat,
        task_type="CPU",
        bootstrap_type="Poisson",
        random_seed=RANDOM_STATE,
        loss_function="Logloss",
        eval_metric="AUC",
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=400,
        verbose=False,
    )

    y_pred = model.predict_proba(X_valid)[:, 1]
    oof[val_idx] = y_pred
    roc = roc_auc_score(y_valid, y_pred)
    print(f" roc_auc_score: {roc}")
    print("-" * 50)

    test_pred += model.predict_proba(test)[:, 1] / folds.n_splits

print(f"Overall OOF ROC AUC: {roc_auc_score(y, oof)}")



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
CatBoostError                             Traceback (most recent call last)
/tmp/ipykernel_11/2731246160.py in <cell line: 0>()
     24     )
     25 
---> 26     model.fit(
     27         X_train,
     28         y_train,

/usr/local/lib/python3.11/dist-packages/catboost/core.py in fit(self, X, y, cat_features, text_features, embedding_features, graph, sample_weight, baseline, use_best_model, eval_set, verbose, logging_level, plot, plot_file, column_description, verbose_eval, metric_period, silent, early_stopping_rounds, save_snapshot, snapshot_file, snapshot_interval, init_model, callbacks, log_cout, log_cerr)
   5243             CatBoostClassifier._check_is_compatible_loss(params['loss_function'])
   5244 
-> 5245         self._fit(X, y, cat_features, text_features, embedding_features, None, graph, sample_weight, None, None, None, None, baseline, use_best_model,
   5246                   eval_set, verbose, logging_level, plot, plot_file, column_description, verbose_eval, metric_period,
   5247                   silent, early_stopping_rounds, save_snapshot, snapshot_file, snapshot_interval, init_model, callbacks, log_cout, log_cerr)

/usr/local/lib/python3.11/dist-packages/catboost/core.py in _fit(self, X, y, cat_features, text_features, embedding_features, pairs, graph, sample_weight, group_id, group_weight, subgroup_id, pairs_weight, baseline, use_best_model, eval_set, verbose, logging_level, plot, plot_file, column_description, verbose_eval, metric_period, silent, early_stopping_rounds, save_snapshot, snapshot_file, snapshot_interval, init_model, callbacks, log_cout, log_cerr)
   2393                 raise CatBoostError("y may be None only when X is an instance of catboost.Pool or string")
   2394 
-> 2395             train_params = self._prepare_train_params(
   2396                 X=X, y=y, cat_features=cat_features, text_features=text_features, embedding_features=embedding_features,
   2397                 pairs=pairs, graph=graph, sample_weight=sample_weight, group_id=group_id, group_weight=group_weight,

/usr/local/lib/python3.11/dist-packages/catboost/core.py in _prepare_train_params(self, X, y, cat_features, text_features, embedding_features, pairs, graph, sample_weight, group_id, group_weight, subgroup_id, pairs_weight, baseline, use_best_model, eval_set, verbose, logging_level, plot, plot_file, column_description, verbose_eval, metric_period, silent, early_stopping_rounds, save_snapshot, snapshot_file, snapshot_interval, init_model, callbacks)
   2319         _check_param_types(params)
   2320         params = _params_type_cast(params)
-> 2321         _check_train_params(params)
   2322 
   2323         if params.get('eval_fraction', 0.0) != 0.0:

_catboost.pyx in _catboost._check_train_params()

_catboost.pyx in _catboost._check_train_params()

CatBoostError: catboost/private/libs/options/bootstrap_options.cpp:29: Error: poisson bootstrap is not supported on CPU

## === cell 11
pred = test_pred



## === cell 12
sub = sub.copy()
sub["id"] = test_id.values  # keep exact ordering from test read
sub["target"] = pred.astype(float)
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
