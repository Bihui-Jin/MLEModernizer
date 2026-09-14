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
statsmodels==0.14.5

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
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


## === cell 1
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn import linear_model
import statsmodels.api as sm


## === cell 2
BASE_PATH = '../input/osic-pulmonary-fibrosis-progression'

data_train_dir = f'{BASE_PATH}/train'
data_test_dir = f'{BASE_PATH}/test'

train = pd.read_csv(f'{BASE_PATH}/train.csv')
test = pd.read_csv(f'{BASE_PATH}/test.csv')


## === cell 3
sample_submission = pd.read_csv(f'{BASE_PATH}/sample_submission.csv')


## === cell 4
patient_dict = {}
def init_fvc(row):
    if row['Patient'] not in patient_dict.keys():
        patient_dict[row['Patient']] = row['FVC']
        return row['FVC']
    else:
        return patient_dict[row['Patient']]

train['InitFVC'] = train.apply(lambda row: init_fvc(row), axis=1)
train.head(20)


## === cell 5
patient_dict = {}
def init_week(row):
    if row['Patient'] not in patient_dict.keys():
        patient_dict[row['Patient']] = row['Weeks']
        return row['Weeks']
    else:
        return patient_dict[row['Patient']]

train['InitWeeks'] = train.apply(lambda row: init_week(row), axis=1)
train.head(20)


## === cell 7
patient_dict = {}
def init_percent(row):
    if row['Patient'] not in patient_dict.keys():
        patient_dict[row['Patient']] = row['Percent']
        return row['Percent']
    else:
        return patient_dict[row['Patient']]

train['InitPercent'] = train.apply(lambda row: init_percent(row), axis=1)
train.head(20)


## === cell 8
train_df = pd.get_dummies(train, columns=['Sex', 'SmokingStatus'], prefix=['Sex', 'SmokingStatus'])


## === cell 9
train_df.head()


## === cell 10
test.head()


## === cell 11
data = []
for i in range(-12, 133+1):
    for index, row in test.iterrows():
        new_cols = list(test.columns)
        new_cols.append('InitWeeks')
        new_vals = [row['Patient'], i, row['FVC'], row['Percent'],row['Age'],row['Sex'],row['SmokingStatus'], row['Weeks']]
        data.append(dict(zip(new_cols, new_vals)))
test_df = pd.DataFrame(data)
test_df.head(10)


## === cell 12
X = train[['Weeks']]
Y = train['FVC']

regr = linear_model.LinearRegression()
regr.fit(X, Y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

model = sm.OLS(Y, X).fit()
print(model.summary())


## === cell 13
sns.lmplot(x='Weeks', y='FVC', data=train.sample(frac=0.8))


## === cell 14
X_test = test[['Weeks']]
Y_test = test['FVC']
print('Predicted FVCs: \n', regr.predict(X_test))
print('Actual FVCs: \n', Y_test)


## === cell 15
X = train[['Weeks', 'InitFVC']]
Y = train['FVC']

regr = linear_model.LinearRegression()
regr.fit(X, Y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

model = sm.OLS(Y, X).fit()
print(model.summary())


## === cell 16
sns.lmplot(x='Weeks', y='FVC', hue='InitFVC', data=train.head(98))

display(test.head())


## === cell 17
X_test = test_df[["Weeks", "FVC"]].rename(columns={"FVC": "InitFVC"})
Y_test = test_df["FVC"]

Y_pred = regr.predict(X_test)
print("Predicted FVCs: \n", Y_pred)
print("Actual FVCs: \n", Y_test)


## === cell 18
X = train[['Weeks', 'InitFVC', 'InitWeeks']]
Y = train['FVC']

regr = linear_model.LinearRegression()
regr.fit(X, Y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

model = sm.OLS(Y, X).fit()
print(model.summary())


## === cell 19
X_test = test_df[["Weeks", "FVC", "InitWeeks"]].rename(columns={"FVC": "InitFVC"})
Y_test = test_df["FVC"]

Y_pred = regr.predict(X_test)
print("Predicted FVCs: \n", Y_pred)
print("Actual FVCs: \n", Y_test)


## === cell 20
data = []
for i in range(test_df.shape[0]):
    new_cols = ['Patient', 'Weeks', 'FVC', 'Confidence']
    new_vals = [test_df.iloc[i]['Patient'], test_df.iloc[i]['Weeks'], Y_pred[i], 100]
    data.append(dict(zip(new_cols, new_vals)))
viz = pd.DataFrame(data)
sns.lmplot(x='Weeks', y='FVC', hue='Patient', data=viz)


## === cell 21
train.head()


## === cell 22
X = train[['Weeks', 'InitFVC','InitWeeks']]
Y = train['FVC']

regr = linear_model.LinearRegression()
regr.fit(X, Y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

model = sm.OLS(Y, X).fit()
print(model.summary())


## === cell 24
X = train[['Weeks', 'InitFVC', 'InitWeeks', 'Age']]
Y = train['FVC']

regr = linear_model.LinearRegression()
regr.fit(X, Y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

model = sm.OLS(Y, X).fit()
print(model.summary())


## === cell 25
X_test = test_df[["Weeks", "FVC", "InitWeeks", "Age"]].rename(
    columns={"FVC": "InitFVC"}
)
Y_test = test_df["FVC"]

Y_pred = regr.predict(X_test)
print("Predicted FVCs: \n", Y_pred)
print("Actual FVCs: \n", Y_test)


## === cell 27
train_df.head()


## === cell 28
X = train_df[["Weeks", "InitFVC", "InitWeeks", "Age", "SmokingStatus_Currently smokes"]]
Y = train_df["FVC"]

regr = linear_model.LinearRegression()
regr.fit(X, Y)

print("Intercept: \n", regr.intercept_)
print("Coefficients: \n", regr.coef_)

X_sm = X.apply(pd.to_numeric, errors="raise")
Y_sm = pd.to_numeric(Y, errors="raise")

model = sm.OLS(Y_sm, X_sm).fit()
print(model.summary())


## --- ERROR in cell 28, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2914136550.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     13[0m [0mY_sm[0m [0;34m=[0m [0mpd[0m[0;34m.[0m[0mto_numeric[0m[0;34m([0m[0mY[0m[0;34m,[0m [0merrors[0m[0;34m=[0m[0;34m"raise"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     14[0m [0;34m[0m[0m
[0;32m---> 15[0;31m [0mmodel[0m [0;34m=[0m [0msm[0m[0;34m.[0m[0mOLS[0m[0;34m([0m[0mY_sm[0m[0;34m,[0m [0mX_sm[0m[0;34m)[0m[0;34m.[0m[0mfit[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     16[0m [0mprint[0m[0;34m([0m[0mmodel[0m[0;34m.[0m[0msummary[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/statsmodels/regression/linear_model.py[0m in [0;36m__init__[0;34m(self, endog, exog, missing, hasconst, **kwargs)[0m
[1;32m    919[0m                    "An exception will be raised in the next version.")
[1;32m    920[0m             [0mwarnings[0m[0;34m.[0m[0mwarn[0m[0;34m([0m[0mmsg[0m[0;34m,[0m [0mValueWarning[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 921[0;31m         super().__init__(endog, exog, missing=missing,
[0m[1;32m    922[0m                                   hasconst=hasconst, **kwargs)
[1;32m    923[0m         [0;32mif[0m [0;34m"weights"[0m [0;32min[0m [0mself[0m[0;34m.[0m[0m_init_keys[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/statsmodels/regression/linear_model.py[0m in [0;36m__init__[0;34m(self, endog, exog, weights, missing, hasconst, **kwargs)[0m
[1;32m    744[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    745[0m             [0mweights[0m [0;34m=[0m [0mweights[0m[0;34m.[0m[0msqueeze[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 746[0;31m         super().__init__(endog, exog, missing=missing,
[0m[1;32m    747[0m                                   weights=weights, hasconst=hasconst, **kwargs)
[1;32m    748[0m         [0mnobs[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mexog[0m[0;34m.[0m[0mshape[0m[0;34m[[0m[0;36m0[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/statsmodels/regression/linear_model.py[0m in [0;36m__init__[0;34m(self, endog, exog, **kwargs)[0m
[1;32m    198[0m     """
[1;32m    199[0m     [0;32mdef[0m [0m__init__[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mendog[0m[0;34m,[0m [0mexog[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 200[0;31m         [0msuper[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0m__init__[0m[0;34m([0m[0mendog[0m[0;34m,[0m [0mexog[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    201[0m         [0mself[0m[0;34m.[0m[0mpinv_wexog[0m[0;34m:[0m [0mFloat64Array[0m [0;34m|[0m [0;32mNone[0m [0;34m=[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[1;32m    202[0m         [0mself[0m[0;34m.[0m[0m_data_attr[0m[0;34m.[0m[0mextend[0m[0;34m([0m[0;34m[[0m[0;34m'pinv_wexog'[0m[0;34m,[0m [0;34m'wendog'[0m[0;34m,[0m [0;34m'wexog'[0m[0;34m,[0m [0;34m'weights'[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/statsmodels/base/model.py[0m in [0;36m__init__[0;34m(self, endog, exog, **kwargs)[0m
[1;32m    268[0m [0;34m[0m[0m
[1;32m    269[0m     [0;32mdef[0m [0m__init__[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mendog[0m[0;34m,[0m [0mexog[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 270[0;31m         [0msuper[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0m__init__[0m[0;34m([0m[0mendog[0m[0;34m,[0m [0mexog[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    271[0m         [0mself[0m[0;34m.[0m[0minitialize[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    272[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/statsmodels/base/model.py[0m in [0;36m__init__[0;34m(self, endog, exog, **kwargs)[0m
[1;32m     93[0m         [0mmissing[0m [0;34m=[0m [0mkwargs[0m[0;34m.[0m[0mpop[0m[0;34m([0m[0;34m'missing'[0m[0;34m,[0m [0;34m'none'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     94[0m         [0mhasconst[0m [0;34m=[0m [0mkwargs[0m[0;34m.[0m[0mpop[0m[0;34m([0m[0;34m'hasconst'[0m[0;34m,[0m [0;32mNone[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 95[0;31m         self.data = self._handle_data(endog, exog, missing, hasconst,
[0m[1;32m     96[0m                                       **kwargs)
[1;32m     97[0m         [0mself[0m[0;34m.[0m[0mk_constant[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mdata[0m[0;34m.[0m[0mk_constant[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/statsmodels/base/model.py[0m in [0;36m_handle_data[0;34m(self, endog, exog, missing, hasconst, **kwargs)[0m
[1;32m    133[0m [0;34m[0m[0m
[1;32m    134[0m     [0;32mdef[0m [0m_handle_data[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mendog[0m[0;34m,[0m [0mexog[0m[0;34m,[0m [0mmissing[0m[0;34m,[0m [0mhasconst[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 135[0;31m         [0mdata[0m [0;34m=[0m [0mhandle_data[0m[0;34m([0m[0mendog[0m[0;34m,[0m [0mexog[0m[0;34m,[0m [0mmissing[0m[0;34m,[0m [0mhasconst[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    136[0m         [0;31m# kwargs arrays could have changed, easier to just attach here[0m[0;34m[0m[0;34m[0m[0m
[1;32m    137[0m         [0;32mfor[0m [0mkey[0m [0;32min[0m [0mkwargs[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/statsmodels/base/data.py[0m in [0;36mhandle_data[0;34m(endog, exog, missing, hasconst, **kwargs)[0m
[1;32m    673[0m [0;34m[0m[0m
[1;32m    674[0m     [0mklass[0m [0;34m=[0m [0mhandle_data_class_factory[0m[0;34m([0m[0mendog[0m[0;34m,[0m [0mexog[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 675[0;31m     return klass(endog, exog=exog, missing=missing, hasconst=hasconst,
[0m[1;32m    676[0m                  **kwargs)

[0;32m/usr/local/lib/python3.11/dist-packages/statsmodels/base/data.py[0m in [0;36m__init__[0;34m(self, endog, exog, missing, hasconst, **kwargs)[0m
[1;32m     82[0m             [0mself[0m[0;34m.[0m[0morig_endog[0m [0;34m=[0m [0mendog[0m[0;34m[0m[0;34m[0m[0m
[1;32m     83[0m             [0mself[0m[0;34m.[0m[0morig_exog[0m [0;34m=[0m [0mexog[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 84[0;31m             [0mself[0m[0;34m.[0m[0mendog[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0mexog[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_convert_endog_exog[0m[0;34m([0m[0mendog[0m[0;34m,[0m [0mexog[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     85[0m [0;34m[0m[0m
[1;32m     86[0m         [0mself[0m[0;34m.[0m[0mconst_idx[0m [0;34m=[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/statsmodels/base/data.py[0m in [0;36m_convert_endog_exog[0;34m(self, endog, exog)[0m
[1;32m    507[0m         [0mexog[0m [0;34m=[0m [0mexog[0m [0;32mif[0m [0mexog[0m [0;32mis[0m [0;32mNone[0m [0;32melse[0m [0mnp[0m[0;34m.[0m[0masarray[0m[0;34m([0m[0mexog[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    508[0m         [0;32mif[0m [0mendog[0m[0;34m.[0m[0mdtype[0m [0;34m==[0m [0mobject[0m [0;32mor[0m [0mexog[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m [0;32mand[0m [0mexog[0m[0;34m.[0m[0mdtype[0m [0;34m==[0m [0mobject[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 509[0;31m             raise ValueError("Pandas data cast to numpy dtype of object. "
[0m[1;32m    510[0m                              "Check input data with np.asarray(data).")
[1;32m    511[0m         [0;32mreturn[0m [0msuper[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0m_convert_endog_exog[0m[0;34m([0m[0mendog[0m[0;34m,[0m [0mexog[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: Pandas data cast to numpy dtype of object. Check input data with np.asarray(data).

## === cell 29
test.head()
