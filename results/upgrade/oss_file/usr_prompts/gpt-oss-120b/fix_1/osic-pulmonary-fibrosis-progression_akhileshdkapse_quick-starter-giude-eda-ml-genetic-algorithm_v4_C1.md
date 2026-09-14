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
Predict a patient’s severity of decline in lung function based on a CT scan of their lungs. Lung function is assessed based on output from a spirometer, which measures the forced vital capacity (`FVC`), i.e. the volume of air exhaled.

## Metric
A modified version of the Laplace Log Likelihood. 

For each true FVC measurement, you will predict both an FVC and a confidence measure (standard deviation 𝜎𝜎). The metric is computed as:

$$
\begin{gathered}
\sigma_{\text {clipped }}=\max (\sigma, 70), \\
\Delta=\min \left(\left|F V C_{\text {true }}-F V C_{\text {predicted }}\right|, 1000\right), \\
\text { metric }=-\frac{\sqrt{2} \Delta}{\sigma_{\text {clipped }}}-\ln \left(\sqrt{2} \sigma_{\text {clipped }}\right) .
\end{gathered}
$$

The error is thresholded at 1000 ml to avoid large errors adversely penalizing results, while the confidence values are clipped at 70 ml to reflect the approximate measurement uncertainty in FVC. The final score is calculated by averaging the metric across all test set `Patient_Week`s (three per patient). 

Metric values will be negative and higher is better.

## Submission Format
For each `Patient_Week`, you must predict the `FVC` and a confidence. You are asked to predict every patient's `FVC` measurement for every possible week. Those weeks which are not in the final three visits are ignored in scoring.

The file should contain a header and have the following format:

```
Patient_Week,FVC,Confidence
ID00002637202176704235138_1,2000,100
ID00002637202176704235138_2,2000,100
ID00002637202176704235138_3,2000,100
etc.

```

## Dataset
In the dataset, you are provided with a baseline chest CT scan and associated clinical information for a set of patients. A patient has an image acquired at time `Week = 0` and has numerous follow up visits over the course of approximately 1-2 years, at which time their `FVC` is measured.

- In the training set, you are provided with an anonymized, baseline CT scan and the entire history of FVC measurements.
- In the test set, you are provided with a baseline CT scan and only the initial FVC measurement. **You are asked to predict the final three `FVC` measurements for each patient, as well as a confidence value in your prediction.**

- **train.csv** - the training set, contains full history of clinical information
- **test.csv** - the test set, contains only the baseline measurement
- **train/** - contains the training patients' baseline CT scan in DICOM format
- **test/** - contains the test patients' baseline CT scan in DICOM format
- **sample_submission.csv** - demonstrates the submission format

**train.csv and test.csv**

- `Patient`a unique Id for each patient (also the name of the patient's DICOM folder)
- `Weeks`the relative number of weeks pre/post the baseline CT (may be negative)
- `FVC` - the recorded lung capacity in ml
- `Percent`a computed field which approximates the patient's FVC as a percent of the typical FVC for a person of similar characteristics
- `Age`
- `Sex`
- `SmokingStatus`

**sample submission.csv**

- `Patient_Week` - a unique Id formed by concatenating the `Patient` and `Weeks` columns (i.e. ABC_22 is a prediction for patient ABC at week 22)
- `FVC` - the predicted FVC in ml
- `Confidence` - a confidence value of your prediction (also has units of ml)

# 2. Python version

3.8

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
plotly==5.24.1
plotly-express==0.4.1
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
TPOT==0.12.1

# 4. Data file paths

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

# 5. Target score

-8.1672

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np # linear algebra
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
        

import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import os

from sklearn.preprocessing import StandardScaler
from tpot import TPOTRegressor
import sklearn
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score, ShuffleSplit, train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error


## === cell 1
df= pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
df.head()


## === cell 2
len(df.Patient.value_counts())


## === cell 3
smk_stats= pd.DataFrame(df.groupby(['SmokingStatus','Sex']).Patient.unique())
smk_stats= smk_stats.reset_index()
smk_stats['Patient']= smk_stats.Patient.apply(lambda x:len(x))

fig = px.bar(smk_stats, x='SmokingStatus', y='Patient', color='Sex',
             barmode='group', title='Smoking Status distribution of 176 patient.', height=500)
fig.show()


## === cell 4
a=df.groupby('Sex').FVC.unique()
a
x1 = a['Male']
x2 = a['Female']

hist_data = [x1, x2]
group_labels = ['Male', 'Female']

fig = ff.create_distplot(hist_data, group_labels, show_hist=False)
fig.update_layout(title_text='FVC distribution')
fig.update_xaxes(title_text='FVC')
fig.show()


## === cell 5
a=df.groupby('SmokingStatus').FVC.unique()
a
x1 = a['Currently smokes']
x2 = a['Ex-smoker']
x3= a['Never smoked']

hist_data = [x1, x2, x3]
group_labels = ['Currently smokes', 'Ex-smoker', 'Never smoked']

fig = ff.create_distplot(hist_data, group_labels, show_hist=False)

fig.update_layout(title_text='FCV distribution')
fig.update_xaxes(title_text='FCV')
fig.show()


## === cell 6
a=df.groupby('Sex').Weeks.unique()
a
x1 = a['Male']
x2 = a['Female']

hist_data = [x1, x2]
group_labels = ['Male', 'Female']

fig = ff.create_distplot(hist_data, group_labels, show_hist=False)

fig.update_layout(title_text='CT scan--->General check-up Week-gap distribution')
fig.update_xaxes(title_text='Week')
fig.show()


## === cell 7
f, ax= plt.subplots(1, 2, figsize=(20,6))

a=df.groupby('SmokingStatus').Age.unique()
a
x1 = a['Currently smokes']
x2 = a['Ex-smoker']
x3= a['Never smoked']
sns.kdeplot(x1, label='Currently smokes', ax=ax[0]); sns.kdeplot(x2, label='Ex-smoker', ax=ax[0])
sns.kdeplot(x3, label='Never smoked', ax=ax[0])

ax[0].set_xlabel('Age'); ax[1].set_xlabel('Age')
ax[0].set_ylabel('Prob. Distribution'); ax[1].set_ylabel('Prob. Distribution')

a=df.groupby('Sex').Age.unique()
a
x1 = a['Male']
x2 = a['Female']
sns.kdeplot(x1, label='Male', ax=ax[1]); sns.kdeplot(x2, label='Female', ax=ax[1])

plt.suptitle("AGE Distribution", size=16)
plt.show()


## === cell 8
f, ax= plt.subplots(1, 2, figsize=(20,6))

sns.scatterplot(df.FVC, df.Percent, hue=df.Sex, ax=ax[0])
sns.scatterplot(df.Weeks, df.FVC, hue=df.Sex, ax=ax[1])

plt.suptitle("Sex Distribution", size=16)
plt.show()


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1902435782.py in <cell line: 0>()
      1 f, ax= plt.subplots(1, 2, figsize=(20,6))
      2 
----> 3 sns.scatterplot(df.FVC, df.Percent, hue=df.Sex, ax=ax[0])
      4 sns.scatterplot(df.Weeks, df.FVC, hue=df.Sex, ax=ax[1])
      5 

TypeError: scatterplot() takes from 0 to 1 positional arguments but 2 positional arguments (and 2 keyword-only arguments) were given

## === cell 9
df.head()


## === cell 10
df=pd.concat([df.drop(['Sex'], axis=1), pd.get_dummies(df['Sex'])], axis=1)
df=pd.concat([df.drop(['SmokingStatus'], axis=1), pd.get_dummies(df['SmokingStatus'])], axis=1)
df.head()


## === cell 11
df.drop(['Female', 'Currently smokes', 'Patient'], inplace=True, axis=1)
df.head()


## === cell 14
df.head()


## === cell 15
df.values.shape


## === cell 16
xtrain, xtest, ytrain, ytest= train_test_split(df.drop(['FVC'], axis=1).values, df.FVC.values, test_size=0.12, random_state=45)
xtrain.shape, xtest.shape, ytrain.shape, ytest.shape


## === cell 18


model= RandomForestRegressor(max_depth=30, max_features='log2', min_samples_leaf=1,
                                                min_samples_split=2, n_estimators=180, random_state=11)                           
model.fit(xtrain, ytrain)
y_pred=model.predict(xtest)

print('---------------------------')
print(mean_squared_error(ytest,y_pred))
print('---------------------------')



## === cell 20
plt.figure(figsize=(20,5))
plt.plot(y_pred, label='Train')
plt.plot(ytest, label='Predictions')
plt.legend()


## === cell 21
plt.figure(figsize=(20,8))
plt.plot(ytrain, label='Train')
plt.plot(model.predict(xtrain), label='Predictions')
plt.legend()


## === cell 22
df2= pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
df2.head()


## === cell 23
df2=pd.concat([df2.drop(['Sex'], axis=1), pd.get_dummies(df2['Sex'])], axis=1)
df2=pd.concat([df2.drop(['SmokingStatus'], axis=1), pd.get_dummies(df2['SmokingStatus'])], axis=1)

id_test= df2.Patient
df2.drop(['Patient'], inplace=True, axis=1)
df2.head()


## === cell 25
plt.figure(figsize=(20,5))
plt.plot(df2.FVC.values, label='test')
plt.plot(model.predict(df2.drop(['FVC'], axis=1).values), label='Prediction')
plt.legend()


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2947244600.py in <cell line: 0>()
      1 plt.figure(figsize=(20,5))
      2 plt.plot(df2.FVC.values, label='test')
----> 3 plt.plot(model.predict(df2.drop(['FVC'], axis=1).values), label='Prediction')
      4 plt.legend()

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_forest.py in predict(self, X)
    979         check_is_fitted(self)
    980         # Check data
--> 981         X = self._validate_X_predict(X)
    982 
    983         # Assign chunk of trees to jobs

/usr/local/lib/python3.11/dist-packages/sklearn/ensemble/_forest.py in _validate_X_predict(self, X)
    600         Validate X whenever one tries to predict, apply, predict_proba."""
    601         check_is_fitted(self)
--> 602         X = self._validate_data(X, dtype=DTYPE, accept_sparse="csr", reset=False)
    603         if issparse(X) and (X.indices.dtype != np.intc or X.indptr.dtype != np.intc):
    604             raise ValueError("No support for np.int64 index based sparse matrices")

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    586 
    587         if not no_val_X and check_params.get("ensure_2d", True):
--> 588             self._check_n_features(X, reset=reset)
    589 
    590         return out

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _check_n_features(self, X, reset)
    387 
    388         if n_features != self.n_features_in_:
--> 389             raise ValueError(
    390                 f"X has {n_features} features, but {self.__class__.__name__} "
    391                 f"is expecting {self.n_features_in_} features as input."

ValueError: X has 8 features, but RandomForestRegressor is expecting 6 features as input.

## === cell 26
submission = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/sample_submission.csv")
submission[['Patient','Weeks']] = submission.Patient_Week.str.split("_",expand=True)
submission.head()


## === cell 27
submission.drop(['FVC', 'Confidence'], axis=1, inplace=True)
df2['Patient'] = id_test


## === cell 28
submission=pd.merge(submission,df2.drop('Weeks', 1),on='Patient',how='left')


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2885523916.py in <cell line: 0>()
----> 1 submission=pd.merge(submission,df2.drop('Weeks', 1),on='Patient',how='left')

TypeError: DataFrame.drop() takes from 1 to 2 positional arguments but 3 were given

## === cell 29
submission.head()


## === cell 30
result = submission.iloc[:, 2:]
result= result.drop('FVC', axis=1)
submission['FVC'] = model.predict(result).astype('int32')


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1697812112.py in <cell line: 0>()
      1 result = submission.iloc[:, 2:]
----> 2 result= result.drop('FVC', axis=1)
      3 submission['FVC'] = model.predict(result).astype('int32')

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in drop(self, labels, axis, index, columns, level, inplace, errors)
   5579                 weight  1.0     0.8
   5580         """
-> 5581         return super().drop(
   5582             labels=labels,
   5583             axis=axis,

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in drop(self, labels, axis, index, columns, level, inplace, errors)
   4786         for axis, labels in axes.items():
   4787             if labels is not None:
-> 4788                 obj = obj._drop_axis(labels, axis, level=level, errors=errors)
   4789 
   4790         if inplace:

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in _drop_axis(self, labels, axis, level, errors, only_slice)
   4828                 new_axis = axis.drop(labels, level=level, errors=errors)
   4829             else:
-> 4830                 new_axis = axis.drop(labels, errors=errors)
   4831             indexer = axis.get_indexer(new_axis)
   4832 

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in drop(self, labels, errors)
   7068         if mask.any():
   7069             if errors != "ignore":
-> 7070                 raise KeyError(f"{labels[mask].tolist()} not found in axis")
   7071             indexer = indexer[~mask]
   7072         return self.delete(indexer)

KeyError: "['FVC'] not found in axis"

## === cell 31
submission.head()


## === cell 32
submission['tot'] =submission.groupby(['Ex-smoker','Male','Age'])['FVC'].transform('mean')


## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3597369503.py in <cell line: 0>()
----> 1 submission['tot'] =submission.groupby(['Ex-smoker','Male','Age'])['FVC'].transform('mean')

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in groupby(self, by, axis, level, as_index, sort, group_keys, observed, dropna)
   9181             raise TypeError("You have to supply one of 'by' and 'level'")
   9182 
-> 9183         return DataFrameGroupBy(
   9184             obj=self,
   9185             keys=by,

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py in __init__(self, obj, keys, axis, level, grouper, exclusions, selection, as_index, sort, group_keys, observed, dropna)
   1327 
   1328         if grouper is None:
-> 1329             grouper, exclusions, obj = get_grouper(
   1330                 obj,
   1331                 keys,

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/grouper.py in get_grouper(obj, key, axis, level, sort, observed, validate, dropna)
   1041                 in_axis, level, gpr = False, gpr, None
   1042             else:
-> 1043                 raise KeyError(gpr)
   1044         elif isinstance(gpr, Grouper) and gpr.key is not None:
   1045             # Add key to exclusions

KeyError: 'Ex-smoker'

## === cell 33
submission['Confidence'] = 100*submission['FVC']/submission['tot']
submission['Confidence']= submission['Confidence'].astype('int32')


## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'FVC'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3462178019.py in <cell line: 0>()
----> 1 submission['Confidence'] = 100*submission['FVC']/submission['tot']
      2 submission['Confidence']= submission['Confidence'].astype('int32')

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'FVC'

## === cell 34
submission.head()


## === cell 35
sub_final = submission[['Patient_Week','FVC','Confidence']]
sub_final


## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/4144494213.py in <cell line: 0>()
----> 1 sub_final = submission[['Patient_Week','FVC','Confidence']]
      2 sub_final

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['FVC', 'Confidence'] not in index"

## === cell 36
sub_final.to_csv("/kaggle/working/submission.csv",index=False)


## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4159153885.py in <cell line: 0>()
----> 1 sub_final.to_csv("/kaggle/working/submission.csv",index=False)

NameError: name 'sub_final' is not defined
