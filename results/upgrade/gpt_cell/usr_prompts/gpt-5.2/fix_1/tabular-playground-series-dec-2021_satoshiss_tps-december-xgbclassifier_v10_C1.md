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
optuna==4.5.0
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

!pip install scikit-learn-intelex
from sklearnex import patch_sklearn
patch_sklearn()


## === cell 1
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.preprocessing import LabelEncoder


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from functools import partial
import optuna
import warnings
warnings.filterwarnings('ignore')


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


## === cell 2
df_train = pd.read_csv('../input/tabular-playground-series-dec-2021/train.csv')
df_test = pd.read_csv('../input/tabular-playground-series-dec-2021/test.csv')


## === cell 3
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
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
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


## === cell 4
df_train = reduce_mem_usage(df_train)
df_test = reduce_mem_usage(df_test)


## === cell 5
df_train.head(5)


## === cell 6
df_train.isna().sum()


## === cell 7
df_train.dtypes


## === cell 8
df_train.describe()


## === cell 9
def is_categorical(data,column):
    if len(data[column].unique()) <= 15:
        print(str(column) + ": " + str(data[column].unique()))
    return None
    
    


## === cell 10
columns = df_train.columns.to_list()
for column in columns:
    is_categorical(df_train,column)
    


## === cell 11

df_train = df_train.drop(['Soil_Type7','Soil_Type15'],axis=1)
df_test = df_test.drop(['Soil_Type7','Soil_Type15'],axis=1)


## === cell 12
plt.figure(figsize=(8,6))

plt.title('Target Distribution')
ax=sns.countplot(x=df_train['Cover_Type'],data=df_train)
total= len(df_train['Cover_Type'])
for p in ax.patches:
    percentage = f'{100 * p.get_height() / total:.1f}%\n'
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center', va='center')


## === cell 13
df_train = df_train[df_train.Cover_Type != 5]
df_train =df_train[df_train.Cover_Type !=4]
df_train=df_train[df_train.Cover_Type !=6]


## === cell 14
targets = df_train.Cover_Type
df_train = df_train.drop(['Cover_Type'],axis=1)


## === cell 15

df_train["Aspect"][df_train["Aspect"] <0] +=360
df_train["Aspect"][df_train["Aspect"] <359]-=360

df_test["Aspect"][df_test["Aspect"] <0] +=360
df_test["Aspect"][df_test["Aspect"] <359] -=360


df_train.loc[df_train["Hillshade_9am"] < 0, "Hillshade_9am"] = 0
df_test.loc[df_test["Hillshade_9am"] < 0, "Hillshade_9am"] = 0

df_train.loc[df_train["Hillshade_Noon"] < 0, "Hillshade_Noon"] = 0
df_test.loc[df_test["Hillshade_Noon"] < 0, "Hillshade_Noon"] = 0

df_train.loc[df_train["Hillshade_3pm"] < 0, "Hillshade_3pm"] = 0
df_test.loc[df_test["Hillshade_3pm"] < 0, "Hillshade_3pm"] = 0

df_train.loc[df_train["Hillshade_9am"] > 255, "Hillshade_9am"] = 255
df_test.loc[df_test["Hillshade_9am"] > 255, "Hillshade_9am"] = 255

df_train.loc[df_train["Hillshade_Noon"] > 255, "Hillshade_Noon"] = 255
df_test.loc[df_test["Hillshade_Noon"] > 255, "Hillshade_Noon"] = 255

df_train.loc[df_train["Hillshade_3pm"] > 255, "Hillshade_3pm"] = 255
df_test.loc[df_test["Hillshade_3pm"] > 255, "Hillshade_3pm"] = 255


## === cell 16
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()

numeric_features = df_train.columns[1:11].to_list()

df_train[numeric_features] = scaler.fit_transform(df_train[numeric_features])
df_test[numeric_features] = scaler.transform(df_test[numeric_features])


## === cell 17
features = df_train.columns

soil_features = [x for x in features if x.startswith("Soil_Type")]
df_train['soil_type_count'] = df_train[soil_features].sum(axis=1)
df_test['soil_type_count'] =df_test[soil_features].sum(axis=1)

wilderness_features= [x for x in features if x.startswith('Wilderness')]
df_train['wilderness_area_count']=df_train[wilderness_features].sum(axis=1)
df_test['wilderness_area_count'] = df_test[wilderness_features].sum(axis=1)


## === cell 18
df_train[['soil_type_count','wilderness_area_count']] = scaler.fit_transform(df_train[['soil_type_count','wilderness_area_count']])
df_test[['soil_type_count','wilderness_area_count']] = scaler.transform(df_test[['soil_type_count','wilderness_area_count']])


## === cell 19

targets[:]=[a - 1 for a in targets]


## === cell 20
targets.value_counts()


## === cell 21
encoder = LabelEncoder()
targets[:] = encoder.fit_transform(targets.values)


## === cell 22
train_X,val_X,train_y,val_y = train_test_split(df_train,targets,random_state=1)


## === cell 23
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

def objective(trial,X,y, name='xgb'):
    params = param = {
        'objective':'multi:softmax',
        'tree_method':'gpu_hist',  
        'lambda': trial.suggest_loguniform(
            'lambda', 1e-3, 10.0
        ),
        'alpha': trial.suggest_loguniform(
            'alpha', 1e-3, 10.0
        ),
        'colsample_bytree': trial.suggest_categorical(
            'colsample_bytree', [0.5,0.6,0.7,0.8,0.9,1.0]
        ),
        'subsample': trial.suggest_categorical(
            'subsample', [0.6,0.7,0.8,1.0]
        ),
        'learning_rate': trial.suggest_categorical(
            'learning_rate', [0.25,0.3,0.35,0.4,0.2,0.1,0.09, 0.01]
        ),
        'n_estimators': trial.suggest_categorical(
            "n_estimators", [150, 200, 300, 3000]
        ),
        'max_depth': trial.suggest_categorical(
            'max_depth', [4,5,7,9,11,13,15,17]
        ),
        'random_state': 42,
        'min_child_weight': trial.suggest_int(
            'min_child_weight', 1, 300
        ),
        'eval_metric':'auc',
        
    }

    model =  XGBClassifier(**params)
    model.fit(train_X,train_y,eval_set=[(val_X,val_y)],early_stopping_rounds=50,verbose=False)


    train_score = np.round(accuracy_score(train_y, model.predict(train_X)), 5)
    test_score = np.round(accuracy_score(val_y, model.predict(val_X)), 5)
                  
    print(f'TRAIN ROC : {train_score} || TEST ROC : {test_score}')
                  
    return test_score


## === cell 26
params = {

'lambda': 0.015064001717372249,
'alpha': 0.009466225483502358,
'colsample_bytree': 0.6,
'subsample': 1.0,
'learning_rate': 0.2,
'n_estimators': 200,
'max_depth': 9,
'min_child_weight': 54,
}


## === cell 27
model = XGBClassifier(**params,tree_method='gpu_hist')
model.fit(train_X,train_y,verbose=False)
pred=model.predict(val_X)

acc= accuracy_score(val_y,pred)

print(f"accuracy_score:{acc}")


## --- ERROR in cell 27, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mXGBoostError[0m                              Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2345624015.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      1[0m [0mmodel[0m [0;34m=[0m [0mXGBClassifier[0m[0;34m([0m[0;34m**[0m[0mparams[0m[0;34m,[0m[0mtree_method[0m[0;34m=[0m[0;34m'gpu_hist'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 2[0;31m [0mmodel[0m[0;34m.[0m[0mfit[0m[0;34m([0m[0mtrain_X[0m[0;34m,[0m[0mtrain_y[0m[0;34m,[0m[0mverbose[0m[0;34m=[0m[0;32mFalse[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      3[0m [0mpred[0m[0;34m=[0m[0mmodel[0m[0;34m.[0m[0mpredict[0m[0;34m([0m[0mval_X[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0;34m[0m[0m
[1;32m      5[0m [0macc[0m[0;34m=[0m [0maccuracy_score[0m[0;34m([0m[0mval_y[0m[0;34m,[0m[0mpred[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

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

[0;31mXGBoostError[0m: [01:23:50] /workspace/src/tree/updater_gpu_hist.cu:781: Exception in gpu_hist: [01:23:50] /workspace/src/tree/updater_gpu_hist.cu:787: Check failed: ctx_->gpu_id >= 0 (-1 vs. 0) : Must have at least one device
Stack trace:
  [bt] (0) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb27f2a) [0x7fff244f5f2a]
  [bt] (1) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb3e95a) [0x7fff2450c95a]
  [bt] (2) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb483cd) [0x7fff245163cd]
  [bt] (3) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x460c79) [0x7fff23e2ec79]
  [bt] (4) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x461d09) [0x7fff23e2fd09]
  [bt] (5) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x4c54f7) [0x7fff23e934f7]
  [bt] (6) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(XGBoosterUpdateOneIter+0x70) [0x7fff23b2fef0]
  [bt] (7) /lib/x86_64-linux-gnu/libffi.so.8(+0x7e2e) [0x7ffff63ace2e]
  [bt] (8) /lib/x86_64-linux-gnu/libffi.so.8(+0x4493) [0x7ffff63a9493]



Stack trace:
  [bt] (0) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb27f2a) [0x7fff244f5f2a]
  [bt] (1) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb485c9) [0x7fff245165c9]
  [bt] (2) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x460c79) [0x7fff23e2ec79]
  [bt] (3) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x461d09) [0x7fff23e2fd09]
  [bt] (4) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x4c54f7) [0x7fff23e934f7]
  [bt] (5) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(XGBoosterUpdateOneIter+0x70) [0x7fff23b2fef0]
  [bt] (6) /lib/x86_64-linux-gnu/libffi.so.8(+0x7e2e) [0x7ffff63ace2e]
  [bt] (7) /lib/x86_64-linux-gnu/libffi.so.8(+0x4493) [0x7ffff63a9493]
  [bt] (8) /usr/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so(+0xa4d8) [0x7ffff63bc4d8]



## === cell 28
index = pd.read_csv("../input/tabular-playground-series-dec-2021/sample_submission.csv")
prediction = model.predict(df_test)
prediction[:]=[a + 1 for a in prediction]

index['Cover_Type'] = prediction
index.to_csv('submission.csv',index=False)
