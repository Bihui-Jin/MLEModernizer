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
google-api-python-client==2.177.0
imbalanced-learn==0.13.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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
setuptools==75.2.0
setuptools-scm==9.2.2
sklearn-pandas==2.2.0
tqdm==4.67.1
types-setuptools==80.9.0.20250529
ydata-profiling==4.17.0

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (59 lines)
            sample_submission.csv (400001 lines)
            sample_submission.csv.zip (1.6 MB)
            test.csv (400001 lines)
            test.csv.zip (10.7 MB)
            train.csv (3600001 lines)
            train.csv.zip (97.9 MB)
            tabular-playground-series-dec-2021/
                description.md (59 lines)
                sample_submission.csv (400001 lines)
                ... and 5 other files
                tabular-playground-series-dec-2021/
        input/
            description.md (59 lines)
            sample_submission.csv (400001 lines)
            sample_submission.csv.zip (1.6 MB)
            test.csv (400001 lines)
            test.csv.zip (10.7 MB)
            train.csv (3600001 lines)
            train.csv.zip (97.9 MB)
            tabular-playground-series-dec-2021/
                description.md (59 lines)
                sample_submission.csv (400001 lines)
                ... and 5 other files
                tabular-playground-series-dec-2021/
        working/
            tabular-playground-series-dec-2021/
                description.md (59 lines)
                sample_submission.csv (400001 lines)
                ... and 5 other files
                tabular-playground-series-dec-2021/
```

-> data/sample_submission.csv has 400000 rows and 2 columns.
The columns are: Id, Cover_Type

-> data/tabular-playground-series-dec-2021/sample_submission.csv has 400000 rows and 2 columns.
The columns are: Id, Cover_Type

-> data/tabular-playground-series-dec-2021/test.csv has 400000 rows and 55 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 40 more columns

-> data/tabular-playground-series-dec-2021/train.csv has 3600000 rows and 56 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 41 more columns

-> data/test.csv has 400000 rows and 55 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 40 more columns

-> data/train.csv has 3600000 rows and 56 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 41 more columns

-> input/sample_submission.csv has 400000 rows and 2 columns.
The columns are: Id, Cover_Type

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)


import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import gc
import random

from IPython import display as ipd

from pandas_profiling import ProfileReport as profile

import pkg_resources as pkg

print(f"pandas_profiling version: {pkg.get_distribution('pandas_profiling').version}")

from tqdm import tqdm
import lightgbm as lgb

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import roc_curve, auc, cohen_kappa_score
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    f1_score,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestRegressor


import matplotlib.pyplot as plt
import seaborn as sns

try:
    import imblearn

    print(imblearn.__version__)
except ModuleNotFoundError as e:
    print(f"imblearn import skipped due to environment incompatibility: {e}")


## === cell 1
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        if col != 'time':
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)  
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def get_stats(df):
    stats = pd.DataFrame(index=df.columns, columns=['na_count', 'n_unique', 'type', 'memory_usage'])
    for col in df.columns:
        stats.loc[col] = [df[col].isna().sum(), df[col].nunique(dropna=False), df[col].dtypes, df[col].memory_usage(deep=True, index=False) / 1024**2]
    stats.loc['Overall'] = [stats['na_count'].sum(), stats['n_unique'].sum(), None, df.memory_usage(deep=True).sum() / 1024**2]
    return stats

def seeding(SEED, use_tf=False):
    np.random.seed(SEED)
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['TF_CUDNN_DETERMINISTIC'] = str(SEED)
    if use_tf:
        tf.random.set_seed(SEED)
    print('seeding done!!!')


## === cell 2
RANDOM_SEED = 42
DEBUG = True
PROFILE = False

seeding(RANDOM_SEED)

train = pd.read_csv('/kaggle/input/tabular-playground-series-dec-2021/train.csv')
test = pd.read_csv('/kaggle/input/tabular-playground-series-dec-2021/test.csv')
submission = pd.read_csv('/kaggle/input/tabular-playground-series-dec-2021/sample_submission.csv')

train = train.sample(frac=1).reset_index(drop=True)

target = train.Cover_Type
train.drop(['Id','Cover_Type'], axis=1, inplace=True)
test.drop(['Id'], axis=1, inplace=True)


## === cell 3
train = reduce_mem_usage(train)
test = reduce_mem_usage(test)
gc.collect()


## === cell 4
import numpy as np
import pandas as pd


def random_oversample_auto(X: pd.DataFrame, y: pd.Series, random_state: int = 42):
    y = pd.Series(y).reset_index(drop=True)
    X = X.reset_index(drop=True)

    counts = y.value_counts(dropna=False)
    max_count = int(counts.max())

    rng = np.random.RandomState(random_state)
    parts_X = []
    parts_y = []

    for cls, n in counts.items():
        idx = np.flatnonzero(y.values == cls)
        if idx.size == 0:
            continue

        if n < max_count:
            extra = rng.choice(idx, size=max_count - int(n), replace=True)
            sel = np.concatenate([idx, extra])
        else:
            sel = idx

        parts_X.append(X.iloc[sel])
        parts_y.append(y.iloc[sel])

    X_res = pd.concat(parts_X, axis=0, ignore_index=True)
    y_res = pd.concat(parts_y, axis=0, ignore_index=True)

    perm = rng.permutation(len(y_res))
    X_res = X_res.iloc[perm].reset_index(drop=True)
    y_res = y_res.iloc[perm].reset_index(drop=True)

    return X_res, y_res


X_sample, y_sample = random_oversample_auto(train, target, random_state=RANDOM_SEED)


## === cell 5
def remap_target_classes(y):
    for i in range(1,8):
        y[y==i] = i-1
    return y

y_sample = remap_target_classes(y_sample)
y_sample.hist()


## === cell 6

if DEBUG:
    temp = pd.concat( [X_sample, y_sample], axis=1 )
    temp = temp.sample(frac=1.0)
    temp = temp[:300000]
    y = temp.Cover_Type
    X = temp.drop(['Cover_Type'], axis=1)
else:
    y = y_sample
    X = X_sample


## === cell 7
%%time
def run_train(X, y, run_params, splits, num_boost_round, verbose_eval, early_stopping_rounds ):
    scores = []
    models = []
    evals_results = {}  # to record eval results for plotting
    folds = KFold(n_splits=splits,shuffle=True, random_state=RANDOM_SEED)
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):
        print(f'Fold {fold_n+1} started')
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        model = lgb.train(
            run_params, valid_names=["train", "valid"], 
            train_set=lgb.Dataset(X_train, y_train ), 
            num_boost_round = num_boost_round,
            valid_sets = [lgb.Dataset(X_valid, y_valid)],
            verbose_eval = verbose_eval,
            evals_result=evals_results,
            early_stopping_rounds = early_stopping_rounds,
        )
        
        y_predicted = np.argmax(model.predict(X_valid), axis=1)
        score = f1_score(y_predicted, y_valid, average="macro")
        print(f"F1 Macro Score: ", score)
        models.append(model)
        scores.append(score)
    return scores, models, evals_results


LEARNING_RATE = 0.009
MAX_DEPTH = -1
NUM_LEAVES = 31
TOTAL_SPLITS = 3
NUM_BOOST_ROUND = 400
EARLY_STOPPING_ROUNDS = 10
VERBOSE_EVAL = 100
    
run_params = {
    'verbose': -1, 
    'boosting_type': 'gbdt', 
    'objective': 'multiclass', 
    'metric': ['multi_logloss'],
    'learning_rate': LEARNING_RATE, 
    'num_leaves': NUM_LEAVES, 
    'max_depth': MAX_DEPTH, 
    'num_class' : 7
}

_, models, evals_results = run_train(X, y, run_params, TOTAL_SPLITS, NUM_BOOST_ROUND, VERBOSE_EVAL, EARLY_STOPPING_ROUNDS)


## --- ERROR in cell 7, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mTypeError[0m                                 Traceback (most recent call last)
[0;32m<timed exec>[0m in [0;36m<module>[0;34m[0m

[0;32m<timed exec>[0m in [0;36mrun_train[0;34m(X, y, run_params, splits, num_boost_round, verbose_eval, early_stopping_rounds)[0m

[0;31mTypeError[0m: train() got an unexpected keyword argument 'verbose_eval'

## === cell 8
ax = lgb.plot_metric(evals_results, metric='multi_logloss')
plt.show()
