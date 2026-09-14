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

3.11

# 2. Installed packages

geopandas==0.14.4
imbalanced-learn==0.13.0
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

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (1485 lines)
            sample_submission.csv.zip (4.4 kB)
            test.csv (1485 lines)
            test.csv.zip (147.8 kB)
            train.csv (13355 lines)
            train.csv.zip (1.4 MB)
            playground-series-s3e18/
                description.md (56 lines)
                sample_submission.csv (1485 lines)
                ... and 5 other files
                playground-series-s3e18/
        input/
            description.md (56 lines)
            sample_submission.csv (1485 lines)
            sample_submission.csv.zip (4.4 kB)
            test.csv (1485 lines)
            test.csv.zip (147.8 kB)
            train.csv (13355 lines)
            train.csv.zip (1.4 MB)
            playground-series-s3e18/
                description.md (56 lines)
                sample_submission.csv (1485 lines)
                ... and 5 other files
                playground-series-s3e18/
        working/
            playground-series-s3e18/
                description.md (56 lines)
                sample_submission.csv (1485 lines)
                ... and 5 other files
                playground-series-s3e18/
```

-> data/playground-series-s3e18/sample_submission.csv has 1484 rows and 3 columns.
The columns are: id, EC1, EC2

-> data/playground-series-s3e18/test.csv has 1484 rows and 32 columns.
The columns are: id, BertzCT, Chi1, Chi1n, Chi1v, Chi2n, Chi2v, Chi3v, Chi4n, EState_VSA1, EState_VSA2, ExactMolWt, FpDensityMorgan1, FpDensityMorgan2, FpDensityMorgan3... and 17 more columns

-> data/playground-series-s3e18/train.csv has 13354 rows and 38 columns.
The columns are: id, BertzCT, Chi1, Chi1n, Chi1v, Chi2n, Chi2v, Chi3v, Chi4n, EState_VSA1, EState_VSA2, ExactMolWt, FpDensityMorgan1, FpDensityMorgan2, FpDensityMorgan3... and 23 more columns

-> data/sample_submission.csv has 1484 rows and 3 columns.
The columns are: id, EC1, EC2

-> data/test.csv has 1484 rows and 32 columns.
The columns are: id, BertzCT, Chi1, Chi1n, Chi1v, Chi2n, Chi2v, Chi3v, Chi4n, EState_VSA1, EState_VSA2, ExactMolWt, FpDensityMorgan1, FpDensityMorgan2, FpDensityMorgan3... and 17 more columns

-> data/train.csv has 13354 rows and 38 columns.
The columns are: id, BertzCT, Chi1, Chi1n, Chi1v, Chi2n, Chi2v, Chi3v, Chi4n, EState_VSA1, EState_VSA2, ExactMolWt, FpDensityMorgan1, FpDensityMorgan2, FpDensityMorgan3... and 23 more columns

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
!pip install klib


## === cell 2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import klib
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


## === cell 3
df_train = pd.read_csv('/kaggle/input/playground-series-s3e18/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s3e18/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s3e18/sample_submission.csv')


## === cell 4
klib.missingval_plot(df_train)
klib.missingval_plot(df_test)
klib.missingval_plot(sub)


## === cell 5
train = klib.data_cleaning(df_train)
test = klib.data_cleaning(df_test)


## === cell 6
train=train.drop(['ec3','ec4', 'ec5', 'ec6'], axis = 1)
train


## === cell 7
from sklearn.feature_selection import mutual_info_classif, SelectKBest
target = ['ec1','ec2']
dic = {}
for i in target:
    mutual_info=mutual_info_classif(train.drop([i],axis=1),train[i])
    mutual_info=pd.Series(mutual_info)
    mutual_info.index=train.drop([i],axis=1).columns
    columns=mutual_info.sort_values(ascending=False)
    columns.plot.bar(title=i,figsize=(20,8))
    plt.show()
    select_cols=SelectKBest(mutual_info_classif,k=10)
    select_cols.fit(train.drop([i],axis=1),train[i])
    dic[i]=train.drop([i],axis=1).columns[select_cols.get_support()]


## === cell 8
dic['ec1'].union(dic['ec2'])


## === cell 9
dic_mutual = {}
target_mutual_col = ['ec1', 'ec2']
for i in target_mutual_col:
    mutual_info = mutual_info_classif(train.drop([i], axis = 1), train[i])
    mutual_info = pd.Series(mutual_info)
    mutual_info.index = train.drop([i], axis = 1).columns
    mutual_info = mutual_info.sort_values(ascending = False)
    mutual_info = mutual_info.index[0:round(len(mutual_info)/2)]
    print('most mutual 50% features for '+ i + ': \n')
    print(mutual_info)
    dic_mutual[i] = mutual_info


## === cell 11
dic_mutual = dic_mutual['ec1'].union(dic_mutual['ec2'])
dic_mutual


## === cell 12
dic_mutual = dic_mutual.drop(['ec1'])


## === cell 13
dic_mutual


## === cell 14
test


## === cell 15
klib.corr_plot(train)


## === cell 16
train.info()


## === cell 17
train.describe().T


## === cell 18
df_train_test = pd.concat([train, test])
df_train_test = df_train_test.drop(['ec1', 'ec2', 'id'], axis = 1)


## === cell 19
df_train_test = df_train_test[dic_mutual]


## === cell 20
col = df_train_test.columns
segments = ['low', 'low-med', 'high-med', 'high']
for col_name in col:
    df_train_test[col_name+'_class'] = pd.cut(df_train_test[col_name], 4, labels = segments )


## === cell 21
df_train_test = pd.get_dummies(df_train_test, columns = df_train_test.select_dtypes('category').columns)
df_train_test


## === cell 22
scaler = MinMaxScaler(feature_range = (0,1))
df_train_test_scale = scaler.fit_transform(df_train_test)
df_train_test_scale = pd.DataFrame(df_train_test_scale)
df_train_test_scale.columns = df_train_test.columns
df_train_test_scale


## === cell 23
train = df_train_test_scale[0:len(train)]
test = df_train_test_scale[len(train):len(df_train_test)]


## === cell 24
train


## === cell 25
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
    cross_val_score,
)
from sklearn import metrics
from xgboost import XGBClassifier
import xgboost as xgb

try:
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import SMOTE
except ModuleNotFoundError:
    RandomUnderSampler = None
    SMOTE = None

from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, precision_score


## === cell 26
y_train = df_train[['EC1','EC2']]
X_train = train
X_test = test
print(f"X_train shape is = {X_train.shape}" )
print(f"y_train shape is = {y_train.shape}" )
print(f"X_test shape is = {X_test.shape}" )


## === cell 27
y_train.head(5)


## === cell 29
kfold = KFold (n_splits=5, shuffle=True, random_state=42)


## === cell 31
xgb = XGBClassifier(n_estimators=2500, random_state = 46, learning_rate=0.009,max_depth=7, max_leaves=15, tree_method="gpu_hist")
gb = GradientBoostingClassifier(random_state = 44, learning_rate=0.009, n_estimators=500, max_depth=10,
                                min_samples_split=20, min_samples_leaf=15)


## === cell 33
xgb_clf = MultiOutputClassifier(xgb)


## === cell 34
oof_preds_xgb = np.zeros(y_train.shape)
oof_preds_lgbm = np.zeros(y_train.shape)
oof_losses_xgb = []
oof_losses_lgbm = []
for fn, (trn_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
    print('Starting fold:', fn)
    X_train_kf, X_val_kf = X_train.iloc[trn_idx], X_train.iloc[val_idx]
    y_train_kf, y_val_kf = y_train.iloc[trn_idx], y_train.iloc[val_idx]
    xgb_clf.fit(X_train_kf, y_train_kf)

    val_preds_xgb = xgb_clf.predict_proba(X_val_kf)
    
    qual_pred = metrics.mean_squared_error(np.ravel(np.array(val_preds_xgb)[:, :, 1].T), np.ravel(y_val_kf))
    
    print('\n metrics.mean_squared_error', qual_pred, '\n')


## --- ERROR in cell 34, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mXGBoostError[0m                              Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/171951142.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      7[0m     [0mX_train_kf[0m[0;34m,[0m [0mX_val_kf[0m [0;34m=[0m [0mX_train[0m[0;34m.[0m[0miloc[0m[0;34m[[0m[0mtrn_idx[0m[0;34m][0m[0;34m,[0m [0mX_train[0m[0;34m.[0m[0miloc[0m[0;34m[[0m[0mval_idx[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m      8[0m     [0my_train_kf[0m[0;34m,[0m [0my_val_kf[0m [0;34m=[0m [0my_train[0m[0;34m.[0m[0miloc[0m[0;34m[[0m[0mtrn_idx[0m[0;34m][0m[0;34m,[0m [0my_train[0m[0;34m.[0m[0miloc[0m[0;34m[[0m[0mval_idx[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 9[0;31m     [0mxgb_clf[0m[0;34m.[0m[0mfit[0m[0;34m([0m[0mX_train_kf[0m[0;34m,[0m [0my_train_kf[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     10[0m [0;34m[0m[0m
[1;32m     11[0m     [0mval_preds_xgb[0m [0;34m=[0m [0mxgb_clf[0m[0;34m.[0m[0mpredict_proba[0m[0;34m([0m[0mX_val_kf[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/multioutput.py[0m in [0;36mfit[0;34m(self, X, Y, sample_weight, **fit_params)[0m
[1;32m    448[0m             [0mReturns[0m [0ma[0m [0mfitted[0m [0minstance[0m[0;34m.[0m[0;34m[0m[0;34m[0m[0m
[1;32m    449[0m         """
[0;32m--> 450[0;31m         [0msuper[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0mfit[0m[0;34m([0m[0mX[0m[0;34m,[0m [0mY[0m[0;34m,[0m [0msample_weight[0m[0;34m,[0m [0;34m**[0m[0mfit_params[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    451[0m         [0mself[0m[0;34m.[0m[0mclasses_[0m [0;34m=[0m [0;34m[[0m[0mestimator[0m[0;34m.[0m[0mclasses_[0m [0;32mfor[0m [0mestimator[0m [0;32min[0m [0mself[0m[0;34m.[0m[0mestimators_[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m    452[0m         [0;32mreturn[0m [0mself[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/multioutput.py[0m in [0;36mfit[0;34m(self, X, y, sample_weight, **fit_params)[0m
[1;32m    214[0m         [0mfit_params_validated[0m [0;34m=[0m [0m_check_fit_params[0m[0;34m([0m[0mX[0m[0;34m,[0m [0mfit_params[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    215[0m [0;34m[0m[0m
[0;32m--> 216[0;31m         self.estimators_ = Parallel(n_jobs=self.n_jobs)(
[0m[1;32m    217[0m             delayed(_fit_estimator)(
[1;32m    218[0m                 [0mself[0m[0;34m.[0m[0mestimator[0m[0;34m,[0m [0mX[0m[0;34m,[0m [0my[0m[0;34m[[0m[0;34m:[0m[0;34m,[0m [0mi[0m[0;34m][0m[0;34m,[0m [0msample_weight[0m[0;34m,[0m [0;34m**[0m[0mfit_params_validated[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/utils/parallel.py[0m in [0;36m__call__[0;34m(self, iterable)[0m
[1;32m     61[0m             [0;32mfor[0m [0mdelayed_func[0m[0;34m,[0m [0margs[0m[0;34m,[0m [0mkwargs[0m [0;32min[0m [0miterable[0m[0;34m[0m[0;34m[0m[0m
[1;32m     62[0m         )
[0;32m---> 63[0;31m         [0;32mreturn[0m [0msuper[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0m__call__[0m[0;34m([0m[0miterable_with_config[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     64[0m [0;34m[0m[0m
[1;32m     65[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/joblib/parallel.py[0m in [0;36m__call__[0;34m(self, iterable)[0m
[1;32m   1984[0m             [0moutput[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_get_sequential_output[0m[0;34m([0m[0miterable[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1985[0m             [0mnext[0m[0;34m([0m[0moutput[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1986[0;31m             [0;32mreturn[0m [0moutput[0m [0;32mif[0m [0mself[0m[0;34m.[0m[0mreturn_generator[0m [0;32melse[0m [0mlist[0m[0;34m([0m[0moutput[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1987[0m [0;34m[0m[0m
[1;32m   1988[0m         [0;31m# Let's create an ID that uniquely identifies the current call. If the[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/joblib/parallel.py[0m in [0;36m_get_sequential_output[0;34m(self, iterable)[0m
[1;32m   1912[0m                 [0mself[0m[0;34m.[0m[0mn_dispatched_batches[0m [0;34m+=[0m [0;36m1[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1913[0m                 [0mself[0m[0;34m.[0m[0mn_dispatched_tasks[0m [0;34m+=[0m [0;36m1[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1914[0;31m                 [0mres[0m [0;34m=[0m [0mfunc[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1915[0m                 [0mself[0m[0;34m.[0m[0mn_completed_tasks[0m [0;34m+=[0m [0;36m1[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1916[0m                 [0mself[0m[0;34m.[0m[0mprint_progress[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/utils/parallel.py[0m in [0;36m__call__[0;34m(self, *args, **kwargs)[0m
[1;32m    121[0m             [0mconfig[0m [0;34m=[0m [0;34m{[0m[0;34m}[0m[0;34m[0m[0;34m[0m[0m
[1;32m    122[0m         [0;32mwith[0m [0mconfig_context[0m[0;34m([0m[0;34m**[0m[0mconfig[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 123[0;31m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0mfunction[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;32m/usr/local/lib/python3.11/dist-packages/sklearn/multioutput.py[0m in [0;36m_fit_estimator[0;34m(estimator, X, y, sample_weight, **fit_params)[0m
[1;32m     47[0m         [0mestimator[0m[0;34m.[0m[0mfit[0m[0;34m([0m[0mX[0m[0;34m,[0m [0my[0m[0;34m,[0m [0msample_weight[0m[0;34m=[0m[0msample_weight[0m[0;34m,[0m [0;34m**[0m[0mfit_params[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     48[0m     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 49[0;31m         [0mestimator[0m[0;34m.[0m[0mfit[0m[0;34m([0m[0mX[0m[0;34m,[0m [0my[0m[0;34m,[0m [0;34m**[0m[0mfit_params[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     50[0m     [0;32mreturn[0m [0mestimator[0m[0;34m[0m[0;34m[0m[0m
[1;32m     51[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/core.py[0m in [0;36minner_f[0;34m(*args, **kwargs)[0m
[1;32m    728[0m             [0;32mfor[0m [0mk[0m[0;34m,[0m [0marg[0m [0;32min[0m [0mzip[0m[0;34m([0m[0msig[0m[0;34m.[0m[0mparameters[0m[0;34m,[0m [0margs[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    729[0m                 [0mkwargs[0m[0;34m[[0m[0mk[0m[0;34m][0m [0;34m=[0m [0marg[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 730[0;31m             [0;32mreturn[0m [0mfunc[0m[0;34m([0m[0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    731[0m [0;34m[0m[0m
[1;32m    732[0m         [0;32mreturn[0m [0minner_f[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py[0m in [0;36mfit[0;34m(self, X, y, sample_weight, base_margin, eval_set, eval_metric, early_stopping_rounds, verbose, xgb_model, sample_weight_eval_set, base_margin_eval_set, feature_weights, callbacks)[0m
[1;32m   1517[0m             )
[1;32m   1518[0m [0;34m[0m[0m
[0;32m-> 1519[0;31m             self._Booster = train(
[0m[1;32m   1520[0m                 [0mparams[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1521[0m                 [0mtrain_dmatrix[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/core.py[0m in [0;36minner_f[0;34m(*args, **kwargs)[0m
[1;32m    728[0m             [0;32mfor[0m [0mk[0m[0;34m,[0m [0marg[0m [0;32min[0m [0mzip[0m[0;34m([0m[0msig[0m[0;34m.[0m[0mparameters[0m[0;34m,[0m [0margs[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    729[0m                 [0mkwargs[0m[0;34m[[0m[0mk[0m[0;34m][0m [0;34m=[0m [0marg[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 730[0;31m             [0;32mreturn[0m [0mfunc[0m[0;34m([0m[0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    731[0m [0;34m[0m[0m
[1;32m    732[0m         [0;32mreturn[0m [0minner_f[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/training.py[0m in [0;36mtrain[0;34m(params, dtrain, num_boost_round, evals, obj, feval, maximize, early_stopping_rounds, evals_result, verbose_eval, xgb_model, callbacks, custom_metric)[0m
[1;32m    179[0m         [0;32mif[0m [0mcb_container[0m[0;34m.[0m[0mbefore_iteration[0m[0;34m([0m[0mbst[0m[0;34m,[0m [0mi[0m[0;34m,[0m [0mdtrain[0m[0;34m,[0m [0mevals[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    180[0m             [0;32mbreak[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 181[0;31m         [0mbst[0m[0;34m.[0m[0mupdate[0m[0;34m([0m[0mdtrain[0m[0;34m,[0m [0mi[0m[0;34m,[0m [0mobj[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    182[0m         [0;32mif[0m [0mcb_container[0m[0;34m.[0m[0mafter_iteration[0m[0;34m([0m[0mbst[0m[0;34m,[0m [0mi[0m[0;34m,[0m [0mdtrain[0m[0;34m,[0m [0mevals[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    183[0m             [0;32mbreak[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/core.py[0m in [0;36mupdate[0;34m(self, dtrain, iteration, fobj)[0m
[1;32m   2048[0m [0;34m[0m[0m
[1;32m   2049[0m         [0;32mif[0m [0mfobj[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 2050[0;31m             _check_call(
[0m[1;32m   2051[0m                 _LIB.XGBoosterUpdateOneIter(
[1;32m   2052[0m                     [0mself[0m[0;34m.[0m[0mhandle[0m[0;34m,[0m [0mctypes[0m[0;34m.[0m[0mc_int[0m[0;34m([0m[0miteration[0m[0;34m)[0m[0;34m,[0m [0mdtrain[0m[0;34m.[0m[0mhandle[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/core.py[0m in [0;36m_check_call[0;34m(ret)[0m
[1;32m    280[0m     """
[1;32m    281[0m     [0;32mif[0m [0mret[0m [0;34m!=[0m [0;36m0[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 282[0;31m         [0;32mraise[0m [0mXGBoostError[0m[0;34m([0m[0mpy_str[0m[0;34m([0m[0m_LIB[0m[0;34m.[0m[0mXGBGetLastError[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    283[0m [0;34m[0m[0m
[1;32m    284[0m [0;34m[0m[0m

[0;31mXGBoostError[0m: [06:20:01] /workspace/src/tree/updater_gpu_hist.cu:781: Exception in gpu_hist: [06:20:01] /workspace/src/tree/updater_gpu_hist.cu:787: Check failed: ctx_->gpu_id >= 0 (-1 vs. 0) : Must have at least one device
Stack trace:
  [bt] (0) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb27f2a) [0x7fff6bf49f2a]
  [bt] (1) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb3e95a) [0x7fff6bf6095a]
  [bt] (2) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb483cd) [0x7fff6bf6a3cd]
  [bt] (3) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x460c79) [0x7fff6b882c79]
  [bt] (4) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x46176c) [0x7fff6b88376c]
  [bt] (5) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x4c54f7) [0x7fff6b8e74f7]
  [bt] (6) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(XGBoosterUpdateOneIter+0x70) [0x7fff6b583ef0]
  [bt] (7) /lib/x86_64-linux-gnu/libffi.so.8(+0x7e2e) [0x7ffff63ace2e]
  [bt] (8) /lib/x86_64-linux-gnu/libffi.so.8(+0x4493) [0x7ffff63a9493]



Stack trace:
  [bt] (0) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb27f2a) [0x7fff6bf49f2a]
  [bt] (1) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb485c9) [0x7fff6bf6a5c9]
  [bt] (2) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x460c79) [0x7fff6b882c79]
  [bt] (3) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x46176c) [0x7fff6b88376c]
  [bt] (4) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x4c54f7) [0x7fff6b8e74f7]
  [bt] (5) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(XGBoosterUpdateOneIter+0x70) [0x7fff6b583ef0]
  [bt] (6) /lib/x86_64-linux-gnu/libffi.so.8(+0x7e2e) [0x7ffff63ace2e]
  [bt] (7) /lib/x86_64-linux-gnu/libffi.so.8(+0x4493) [0x7ffff63a9493]
  [bt] (8) /usr/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so(+0xa4d8) [0x7ffff63bc4d8]



## === cell 39
model = xgb_clf
