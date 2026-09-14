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
pydicom==3.0.1
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
xgboost==2.0.3

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (122 lines)
            sample_submission.csv (1909 lines)
            sample_submission.csv.zip (5.7 kB)
            test.csv (19 lines)
            test.csv.zip (748 Bytes)
            test.zip (1.2 GB)
            train.csv (1395 lines)
            train.csv.zip (23.6 kB)
            train.zip (12.7 GB)
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
            test/
                ID00014637202177757139317/
                    1.dcm (1.5 MB)
                    10.dcm (1.5 MB)
                    ... and 29 other files
                ID00019637202178323708467/
                    1.dcm (525.5 kB)
                    10.dcm (525.5 kB)
                    ... and 27 other files
                ... and 17 other folders
            train/
                ID00007637202177411956430/
                    1.dcm (525.6 kB)
                    10.dcm (525.6 kB)
                    ... and 28 other files
                ID00009637202177434476278/
                    1.dcm (1.2 MB)
                    10.dcm (1.2 MB)
                    ... and 392 other files
                ... and 157 other folders
        input/
            description.md (122 lines)
            sample_submission.csv (1909 lines)
            sample_submission.csv.zip (5.7 kB)
            test.csv (19 lines)
            test.csv.zip (748 Bytes)
            test.zip (1.2 GB)
            train.csv (1395 lines)
            train.csv.zip (23.6 kB)
            train.zip (12.7 GB)
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
            test/
                ID00014637202177757139317/
                    1.dcm (1.5 MB)
                    10.dcm (1.5 MB)
                    ... and 29 other files
                ID00019637202178323708467/
                    1.dcm (525.5 kB)
                    10.dcm (525.5 kB)
                    ... and 27 other files
                ... and 17 other folders
            train/
                ID00007637202177411956430/
                    1.dcm (525.6 kB)
                    10.dcm (525.6 kB)
                    ... and 28 other files
                ID00009637202177434476278/
                    1.dcm (1.2 MB)
                    10.dcm (1.2 MB)
                    ... and 392 other files
                ... and 157 other folders
        working/
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
```

-> data/osic-pulmonary-fibrosis-progression/sample_submission.csv has 1908 rows and 3 columns.
The columns are: Patient_Week, FVC, Confidence

-> data/osic-pulmonary-fibrosis-progression/test.csv has 18 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/osic-pulmonary-fibrosis-progression/train.csv has 1394 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/sample_submission.csv has 1908 rows and 3 columns.
The columns are: Patient_Week, FVC, Confidence

-> data/test.csv has 18 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/train.csv has 1394 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0
import pandas as pd
import numpy as np
import typing as tp
import pydicom
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb
from sklearn.linear_model import HuberRegressor


## === cell 1
train_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
test_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
df = pd.concat([train_df, test_df], ignore_index=True)
df['Patient_Week'] = df['Patient'].astype(str) + '_ '+ df['Weeks'].astype(str)


## === cell 3
print('Shape of Training data: ', train_df.shape)
print('Shape of Test data: ', test_df.shape)


## === cell 4
def add_height(data)->'dataframe':
    data['Height'] = 0
    data['Height'] = data.apply(lambda x: x.FVC / (21.78 - (0.101 * x.Age)) if x.Height == 1 else x.FVC / (27.63 - (0.112 * x.Age)), axis=1)
    
def add_norm(data)->'dataframe':
    return (data - data.mean()) / data.std()


## === cell 5
df['Sex'] = df['Sex'].map({'Female': 0, 'Male': 1})
df['SmokingStatus'] = df['SmokingStatus'].map({'Currently smokes': 0, 'Never smoked': 1, 'Ex-smoker': 2})
df = df.drop('Patient_Week', axis=1)
df = df.set_index('Patient')
add_height(df)
df[df.columns[~df.columns.isin(['FVC', 'Sex'])]] = add_norm(df[df.columns[~df.columns.isin(['FVC', 'Sex'])]])
df.head()


## === cell 6
sub_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')
sub_df.drop(['FVC'], axis=1, inplace=True)
sub_df['Patient'] = sub_df['Patient_Week'].apply(lambda x: x.split('_')[0])
sub_df['pred_Weeks'] = sub_df['Patient_Week'].apply(lambda x: x.split('_')[1]).astype(int)

sub_df.head()


## === cell 7
test_FVC = pd.merge(sub_df, df, how="left", on=["Patient"])
test_FVC = test_FVC.rename(columns={"Patient_Week_x": "Patient_Week"})
test_FVC = test_FVC.drop(["pred_Weeks", "FVC", "Confidence"], axis=1)
test_FVC = test_FVC.groupby("Patient_Week").mean(numeric_only=True)
test_FVC[test_FVC.columns[~test_FVC.columns.isin(["Sex"])]] = add_norm(
    test_FVC[test_FVC.columns[~test_FVC.columns.isin(["Sex"])]]
)

test_FVC.head()


## === cell 8
test_conf = pd.merge(sub_df, df, how="left", on=["Patient"])
test_conf = test_conf.rename(columns={"Patient_Week_x": "Patient_Week"})
test_conf = test_conf.drop(["pred_Weeks", "Percent", "Confidence"], axis=1)

test_conf = test_conf.groupby("Patient_Week").mean(numeric_only=True)

test_conf[test_conf.columns[~test_conf.columns.isin(["Sex", "FVC"])]] = add_norm(
    test_conf[test_conf.columns[~test_conf.columns.isin(["Sex", "FVC"])]]
)

test_conf.head()


## === cell 9
print(test_FVC.shape)
print(test_conf.shape)


## === cell 10
X = df.iloc[:, df.columns != "FVC"]
y = df["FVC"]

X_train = X[:-5]
y_train = y[:-5]
X_val = X[-5:]
y_val = y[-5:]

print(X.shape)
print(y.shape)
print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)


## === cell 11
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
train_df.boxplot('FVC', by='SmokingStatus', ax = axs[0, 0])
train_df.boxplot('Percent', by='SmokingStatus', ax = axs[0, 1])
train_df.boxplot('FVC', by='Sex', ax = axs[1, 0])
train_df.boxplot('Percent', by='Sex', ax = axs[1, 1])

plt.show()


## === cell 12
img = "../input/osic-pulmonary-fibrosis-progression/train/ID00009637202177434476278/100.dcm"
ds = pydicom.dcmread(img)
plt.figure(figsize = (7,7))
plt.imshow(ds.pixel_array, cmap=plt.cm.bone)


## === cell 13
img_1 = "../input/osic-pulmonary-fibrosis-progression/train/ID00009637202177434476278/100.dcm"
img_2 = "../input/osic-pulmonary-fibrosis-progression/train/ID00012637202177665765362/10.dcm"

fig, ax = plt.subplots(1, 2, figsize=(10, 10))
ds = pydicom.dcmread(img_1)
ax[0].set_title('Patient 1: Ex-Smoker')
ax[0].imshow(ds.pixel_array, cmap=plt.cm.bone)

ds = pydicom.dcmread(img_2)
ax[1].set_title('Patient 2: Never smoked')
ax[1].imshow(ds.pixel_array, cmap=plt.cm.bone)

plt.show


## === cell 14
def baseline_loss_metric(trueFVC, predFVC, predSTD=100):
    clipSTD = np.clip(predSTD, 70 , 9e9)  
    deltaFVC = np.clip(np.abs(trueFVC - predFVC), 0 , 1000)  
    error = np.mean(-1 * (np.sqrt(2) * deltaFVC / clipSTD) - np.log(np.sqrt(2) * clipSTD))
    return error


## === cell 15
class model_selection(): 
    
    def __init__(self): 
        self.y_pred_FVC = pd.DataFrame()
        self.my_scorer = make_scorer(baseline_loss_metric, greater_is_better=False)
        self.best_param = None
        self.scoring = 0
    
    
    def xgboost(self, X, y, X_val, y_val,test): 
        parameters = {'learning_rate': [0.0015, 0.002], 'n_estimators':[3400, 3500], 'max_depth':[2, 3], 'reg_alpha':[0.005]}
        clf = GridSearchCV(XGBRegressor(min_child_weight=0, gamma=0, 
                                        colsample_bytree=0.7, objective='reg:linear', nthread=-1,
                                        scale_pos_weight=1, subsample=.7, seed=27), 
                           param_grid=parameters, 
                           scoring=self.my_scorer)
        clf.fit(X, y)
        self.best_param = clf.best_params_
        self.scoring = clf.score(X_val, y_val)
        y_pred_xgb_FVC = clf.predict(test)
        
        self.y_pred_FVC = pd.concat([pd.Series(test.index), pd.Series(y_pred_xgb_FVC)], axis=1)
        return self.y_pred_FVC, self.best_param, self.scoring
        
    
    def lightgbm(self, X, y, X_val, y_val,test): 
        parameters = {'learning_rate': [0.005, 0.01, 0.05], 'n_estimators':[500, 700, 1000], 'num_leaves':[30]}
        clf = GridSearchCV(LGBMRegressor(boosting_type='rf', 
                                         objective='regression',
                                         bagging_fraction=0.8,
                                         bagging_freq=1, 
                                         verbose=-1,), 
                           param_grid=parameters, 
                           scoring=self.my_scorer)
        clf.fit(X, y)
        self.best_param = clf.best_params_
        self.scoring = clf.score(X_val, y_val)
        y_pred_lgb_FVC = clf.predict(test)
        
        self.y_pred_FVC = pd.concat([pd.Series(test.index), pd.Series(y_pred_lgb_FVC)], axis=1)
        return self.y_pred_FVC, self.best_param, self.scoring
    
    
    def HuberRegressor(self, X, y, test): 
        hbr = HuberRegressor(max_iter=200)
        hbr.fit(X, y)
        y_pred_hbr_FVC = hbr.predict(test)
        
        self.y_pred_FVC = pd.concat([pd.Series(test.index), pd.Series(y_pred_hbr_FVC)], axis=1)
        return self.y_pred_FVC


## === cell 16
model = model_selection()
output = model.lightgbm(X_train, y_train, X_val, y_val,test_FVC)


## === cell 17
print(output[1]) # best params on previous model {'learning_rate': 0.05, 'n_estimators': 500, 'num_leaves': 30}
print(output[2]) # error on validation 5.434446656836348


## === cell 18
y_pred_FVC = output[0]
y_pred_FVC.rename(columns={0:'FVC'}, inplace=True)
y_pred_FVC


## === cell 19
def competition_metric(trueFVC, predFVC, predSTD):
    clipSTD = np.clip(predSTD, 70 , 9e9)  
    deltaFVC = np.clip(np.abs(trueFVC - predFVC), 0 , 700)  
    error = np.mean(-1 * (np.sqrt(2) * deltaFVC / clipSTD) - np.log(np.sqrt(2) * clipSTD))
    return deltaFVC


## === cell 20
test_conf = pd.merge(sub_df, df, how='left', on=['Patient'])
test_conf = test_conf.rename(columns={'Patient_Week_x':'Patient_Week'})
test_conf = test_conf.drop(['pred_Weeks', 'Percent', 'Confidence'], axis=1)
test_conf = test_conf.groupby('Patient_Week').mean()
test_conf[test_conf.columns[~test_conf.columns.isin(['Sex','FVC'])]] = add_norm(test_conf[test_conf.columns[~test_conf.columns.isin(['Sex','FVC'])]])

test_conf = pd.merge(y_pred_FVC, test_conf, how='left', on=['Patient_Week']).rename(columns={'FVC_x':'FVC_pred', 'FVC_y':'FVC'})
test_conf.set_index('Patient_Week', inplace=True)
test_conf.head()


## --- ERROR in cell 20, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mTypeError[0m                                 Traceback (most recent call last)
[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py[0m in [0;36m_agg_py_fallback[0;34m(self, how, values, ndim, alt)[0m
[1;32m   1941[0m         [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1942[0;31m             [0mres_values[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_grouper[0m[0;34m.[0m[0magg_series[0m[0;34m([0m[0mser[0m[0;34m,[0m [0malt[0m[0;34m,[0m [0mpreserve_dtype[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1943[0m         [0;32mexcept[0m [0mException[0m [0;32mas[0m [0merr[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/ops.py[0m in [0;36magg_series[0;34m(self, obj, func, preserve_dtype)[0m
[1;32m    863[0m [0;34m[0m[0m
[0;32m--> 864[0;31m         [0mresult[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_aggregate_series_pure_python[0m[0;34m([0m[0mobj[0m[0;34m,[0m [0mfunc[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    865[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/ops.py[0m in [0;36m_aggregate_series_pure_python[0;34m(self, obj, func)[0m
[1;32m    884[0m         [0;32mfor[0m [0mi[0m[0;34m,[0m [0mgroup[0m [0;32min[0m [0menumerate[0m[0;34m([0m[0msplitter[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 885[0;31m             [0mres[0m [0;34m=[0m [0mfunc[0m[0;34m([0m[0mgroup[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    886[0m             [0mres[0m [0;34m=[0m [0mextract_result[0m[0;34m([0m[0mres[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py[0m in [0;36m<lambda>[0;34m(x)[0m
[1;32m   2453[0m                 [0;34m"mean"[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 2454[0;31m                 [0malt[0m[0;34m=[0m[0;32mlambda[0m [0mx[0m[0;34m:[0m [0mSeries[0m[0;34m([0m[0mx[0m[0;34m,[0m [0mcopy[0m[0;34m=[0m[0;32mFalse[0m[0;34m)[0m[0;34m.[0m[0mmean[0m[0;34m([0m[0mnumeric_only[0m[0;34m=[0m[0mnumeric_only[0m[0;34m)[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   2455[0m                 [0mnumeric_only[0m[0;34m=[0m[0mnumeric_only[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/series.py[0m in [0;36mmean[0;34m(self, axis, skipna, numeric_only, **kwargs)[0m
[1;32m   6548[0m     ):
[0;32m-> 6549[0;31m         [0;32mreturn[0m [0mNDFrame[0m[0;34m.[0m[0mmean[0m[0;34m([0m[0mself[0m[0;34m,[0m [0maxis[0m[0;34m,[0m [0mskipna[0m[0;34m,[0m [0mnumeric_only[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   6550[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py[0m in [0;36mmean[0;34m(self, axis, skipna, numeric_only, **kwargs)[0m
[1;32m  12419[0m     ) -> Series | float:
[0;32m> 12420[0;31m         return self._stat_function(
[0m[1;32m  12421[0m             [0;34m"mean"[0m[0;34m,[0m [0mnanops[0m[0;34m.[0m[0mnanmean[0m[0;34m,[0m [0maxis[0m[0;34m,[0m [0mskipna[0m[0;34m,[0m [0mnumeric_only[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py[0m in [0;36m_stat_function[0;34m(self, name, func, axis, skipna, numeric_only, **kwargs)[0m
[1;32m  12376[0m [0;34m[0m[0m
[0;32m> 12377[0;31m         return self._reduce(
[0m[1;32m  12378[0m             [0mfunc[0m[0;34m,[0m [0mname[0m[0;34m=[0m[0mname[0m[0;34m,[0m [0maxis[0m[0;34m=[0m[0maxis[0m[0;34m,[0m [0mskipna[0m[0;34m=[0m[0mskipna[0m[0;34m,[0m [0mnumeric_only[0m[0;34m=[0m[0mnumeric_only[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/series.py[0m in [0;36m_reduce[0;34m(self, op, name, axis, skipna, numeric_only, filter_type, **kwds)[0m
[1;32m   6456[0m                 )
[0;32m-> 6457[0;31m             [0;32mreturn[0m [0mop[0m[0;34m([0m[0mdelegate[0m[0;34m,[0m [0mskipna[0m[0;34m=[0m[0mskipna[0m[0;34m,[0m [0;34m**[0m[0mkwds[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   6458[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/nanops.py[0m in [0;36mf[0;34m(values, axis, skipna, **kwds)[0m
[1;32m    146[0m             [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 147[0;31m                 [0mresult[0m [0;34m=[0m [0malt[0m[0;34m([0m[0mvalues[0m[0;34m,[0m [0maxis[0m[0;34m=[0m[0maxis[0m[0;34m,[0m [0mskipna[0m[0;34m=[0m[0mskipna[0m[0;34m,[0m [0;34m**[0m[0mkwds[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    148[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/nanops.py[0m in [0;36mnew_func[0;34m(values, axis, skipna, mask, **kwargs)[0m
[1;32m    403[0m [0;34m[0m[0m
[0;32m--> 404[0;31m         [0mresult[0m [0;34m=[0m [0mfunc[0m[0;34m([0m[0mvalues[0m[0;34m,[0m [0maxis[0m[0;34m=[0m[0maxis[0m[0;34m,[0m [0mskipna[0m[0;34m=[0m[0mskipna[0m[0;34m,[0m [0mmask[0m[0;34m=[0m[0mmask[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    405[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/nanops.py[0m in [0;36mnanmean[0;34m(values, axis, skipna, mask)[0m
[1;32m    719[0m     [0mthe_sum[0m [0;34m=[0m [0mvalues[0m[0;34m.[0m[0msum[0m[0;34m([0m[0maxis[0m[0;34m,[0m [0mdtype[0m[0;34m=[0m[0mdtype_sum[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 720[0;31m     [0mthe_sum[0m [0;34m=[0m [0m_ensure_numeric[0m[0;34m([0m[0mthe_sum[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    721[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/nanops.py[0m in [0;36m_ensure_numeric[0;34m(x)[0m
[1;32m   1700[0m             [0;31m# GH#44008, GH#36703 avoid casting e.g. strings to numeric[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1701[0;31m             [0;32mraise[0m [0mTypeError[0m[0;34m([0m[0;34mf"Could not convert string '{x}' to numeric"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1702[0m         [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;31mTypeError[0m: Could not convert string 'ID00014637202177757139317' to numeric

The above exception was the direct cause of the following exception:

[0;31mTypeError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2503227371.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      2[0m [0mtest_conf[0m [0;34m=[0m [0mtest_conf[0m[0;34m.[0m[0mrename[0m[0;34m([0m[0mcolumns[0m[0;34m=[0m[0;34m{[0m[0;34m'Patient_Week_x'[0m[0;34m:[0m[0;34m'Patient_Week'[0m[0;34m}[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m [0mtest_conf[0m [0;34m=[0m [0mtest_conf[0m[0;34m.[0m[0mdrop[0m[0;34m([0m[0;34m[[0m[0;34m'pred_Weeks'[0m[0;34m,[0m [0;34m'Percent'[0m[0;34m,[0m [0;34m'Confidence'[0m[0;34m][0m[0;34m,[0m [0maxis[0m[0;34m=[0m[0;36m1[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 4[0;31m [0mtest_conf[0m [0;34m=[0m [0mtest_conf[0m[0;34m.[0m[0mgroupby[0m[0;34m([0m[0;34m'Patient_Week'[0m[0;34m)[0m[0;34m.[0m[0mmean[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      5[0m [0mtest_conf[0m[0;34m[[0m[0mtest_conf[0m[0;34m.[0m[0mcolumns[0m[0;34m[[0m[0;34m~[0m[0mtest_conf[0m[0;34m.[0m[0mcolumns[0m[0;34m.[0m[0misin[0m[0;34m([0m[0;34m[[0m[0;34m'Sex'[0m[0;34m,[0m[0;34m'FVC'[0m[0;34m][0m[0;34m)[0m[0;34m][0m[0;34m][0m [0;34m=[0m [0madd_norm[0m[0;34m([0m[0mtest_conf[0m[0;34m[[0m[0mtest_conf[0m[0;34m.[0m[0mcolumns[0m[0;34m[[0m[0;34m~[0m[0mtest_conf[0m[0;34m.[0m[0mcolumns[0m[0;34m.[0m[0misin[0m[0;34m([0m[0;34m[[0m[0;34m'Sex'[0m[0;34m,[0m[0;34m'FVC'[0m[0;34m][0m[0;34m)[0m[0;34m][0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      6[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py[0m in [0;36mmean[0;34m(self, numeric_only, engine, engine_kwargs)[0m
[1;32m   2450[0m             )
[1;32m   2451[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 2452[0;31m             result = self._cython_agg_general(
[0m[1;32m   2453[0m                 [0;34m"mean"[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m   2454[0m                 [0malt[0m[0;34m=[0m[0;32mlambda[0m [0mx[0m[0;34m:[0m [0mSeries[0m[0;34m([0m[0mx[0m[0;34m,[0m [0mcopy[0m[0;34m=[0m[0;32mFalse[0m[0;34m)[0m[0;34m.[0m[0mmean[0m[0;34m([0m[0mnumeric_only[0m[0;34m=[0m[0mnumeric_only[0m[0;34m)[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py[0m in [0;36m_cython_agg_general[0;34m(self, how, alt, numeric_only, min_count, **kwargs)[0m
[1;32m   1996[0m             [0;32mreturn[0m [0mresult[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1997[0m [0;34m[0m[0m
[0;32m-> 1998[0;31m         [0mnew_mgr[0m [0;34m=[0m [0mdata[0m[0;34m.[0m[0mgrouped_reduce[0m[0;34m([0m[0marray_func[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1999[0m         [0mres[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_wrap_agged_manager[0m[0;34m([0m[0mnew_mgr[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   2000[0m         [0;32mif[0m [0mhow[0m [0;32min[0m [0;34m[[0m[0;34m"idxmin"[0m[0;34m,[0m [0;34m"idxmax"[0m[0;34m][0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py[0m in [0;36mgrouped_reduce[0;34m(self, func)[0m
[1;32m   1467[0m                 [0;31m#  while others do not.[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1468[0m                 [0;32mfor[0m [0msb[0m [0;32min[0m [0mblk[0m[0;34m.[0m[0m_split[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1469[0;31m                     [0mapplied[0m [0;34m=[0m [0msb[0m[0;34m.[0m[0mapply[0m[0;34m([0m[0mfunc[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1470[0m                     [0mresult_blocks[0m [0;34m=[0m [0mextend_blocks[0m[0;34m([0m[0mapplied[0m[0;34m,[0m [0mresult_blocks[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1471[0m             [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/internals/blocks.py[0m in [0;36mapply[0;34m(self, func, **kwargs)[0m
[1;32m    391[0m         [0mone[0m[0;34m[0m[0;34m[0m[0m
[1;32m    392[0m         """
[0;32m--> 393[0;31m         [0mresult[0m [0;34m=[0m [0mfunc[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mvalues[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    394[0m [0;34m[0m[0m
[1;32m    395[0m         [0mresult[0m [0;34m=[0m [0mmaybe_coerce_values[0m[0;34m([0m[0mresult[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py[0m in [0;36marray_func[0;34m(values)[0m
[1;32m   1993[0m [0;34m[0m[0m
[1;32m   1994[0m             [0;32massert[0m [0malt[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1995[0;31m             [0mresult[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_agg_py_fallback[0m[0;34m([0m[0mhow[0m[0;34m,[0m [0mvalues[0m[0;34m,[0m [0mndim[0m[0;34m=[0m[0mdata[0m[0;34m.[0m[0mndim[0m[0;34m,[0m [0malt[0m[0;34m=[0m[0malt[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1996[0m             [0;32mreturn[0m [0mresult[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1997[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py[0m in [0;36m_agg_py_fallback[0;34m(self, how, values, ndim, alt)[0m
[1;32m   1944[0m             [0mmsg[0m [0;34m=[0m [0;34mf"agg function failed [how->{how},dtype->{ser.dtype}]"[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1945[0m             [0;31m# preserve the kind of exception that raised[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1946[0;31m             [0;32mraise[0m [0mtype[0m[0;34m([0m[0merr[0m[0;34m)[0m[0;34m([0m[0mmsg[0m[0;34m)[0m [0;32mfrom[0m [0merr[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1947[0m [0;34m[0m[0m
[1;32m   1948[0m         [0;32mif[0m [0mser[0m[0;34m.[0m[0mdtype[0m [0;34m==[0m [0mobject[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;31mTypeError[0m: agg function failed [how->mean,dtype->object]

## === cell 21
y_pred_conf = pd.DataFrame(competition_metric(test_conf['FVC'], test_conf['FVC_pred'], 100))
y_pred_conf.rename(columns={0:'Confidence'}, inplace=True)
y_pred_conf
