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
pymc3==3.11.4
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0

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

import numpy as np  # linear algebra

if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore[attr-defined]

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

try:
    import pymc3 as pm
except AttributeError:
    pm = None

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        break


## === cell 1
train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
train_raw = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

train.drop(train[train.Patient == 'ID00197637202246865691526'].index, inplace=True)



train = pd.concat([train, test], axis=0, ignore_index=True)\
    .drop_duplicates()
le_id = LabelEncoder()
train['PatientID'] = le_id.fit_transform(train['Patient'])

train.head()


## === cell 2
def add_baselines(data):
    aux = data[['Patient', 'Weeks']].groupby('Patient')\
        .min().reset_index()
    aux = pd.merge(aux, data[['Patient', 'Weeks', 'FVC']], how='left', 
                   on=['Patient', 'Weeks'])
    aux = aux.groupby('Patient').mean().reset_index()
    aux['Weeks'] = aux['Weeks'].astype(int)
    aux['FVC'] = aux['FVC'].astype(int)
    data = pd.merge(data, aux, how='left', on='Patient', suffixes=('', '_base'))
    return data

train = add_baselines(train)
test = add_baselines(test)
train.head()


## === cell 3
def patient_class(row):
    if row['Sex'] == 'Male':
        if row['SmokingStatus'] == 'Currently smokes':
            return 0
        elif row['SmokingStatus'] == 'Ex-smoker':
            return 1
        elif row['SmokingStatus'] == 'Never smoked':
            return 2
    else:
        if row['SmokingStatus'] == 'Currently smokes':
            return 3
        elif row['SmokingStatus'] == 'Ex-smoker':
            return 4
        elif row['SmokingStatus'] == 'Never smoked':
            return 5

train['Class'] = train.apply(patient_class, axis=1)
test['Class'] = test.apply(patient_class, axis=1)
test.head()
train.loc[train['Patient']=='ID00007637202177411956430']['Class'].max()


## === cell 4
PatientID = train['Patient'].values
fvc_b = train.groupby('Patient').first()['FVC_base']
fvc_b.values


## === cell 5
def model_fit(data, examine=True):
    n_patients = data['Patient'].nunique()
    FVC_obs = data['FVC'].values
    Weeks = data['Weeks'].values
    PatientID = data['PatientID'].values
    patient_class = data['Class'].values
    FVC_b = data.groupby('PatientID').first()['FVC_base']
    w_b = data.groupby('PatientID').first()['Weeks_base']

    with pm.Model() as model:
        FVC_obs_shared = pm.Data("FVC_obs_shared", FVC_obs)
        Weeks_shared = pm.Data('Weeks_shared', Weeks)
        PatientID_shared = pm.Data('PatientID_shared', PatientID)
        patient_class_shared = pm.Data('patient_class_shared', patient_class)
        FVC_b_shared = pm.Data('FVC_b_shared', FVC_b)
        w_b_shared= pm.Data('w_b_shared', w_b)

        mu_a_base = pm.Normal('mu_a_base', mu=0., sigma=100)
        mu_a = FVC_b_shared + mu_a_base * w_b_shared
        
        sigma_a = pm.HalfNormal('sigma_a', 1000.)
        mu_b = pm.Normal('mu_b', mu=-4., sigma=1)
        sigma_b = pm.HalfNormal('sigma_b', 5.)

        a = pm.Normal('a', mu=mu_a, sigma=sigma_a, shape=n_patients)
        b = pm.Normal('b', mu=mu_b, sigma=sigma_b, shape=n_patients)

        sigma = pm.HalfNormal('sigma', 150., shape=6)
        
        FVC_est = a[PatientID_shared] + b[PatientID_shared] * Weeks_shared

        FVC_like = pm.Normal('FVC_like', mu=FVC_est,
                             sigma=sigma[patient_class_shared], observed=FVC_obs_shared)

        trace = pm.sample(2000, tune=2000, target_accept=.9, init="adapt_diag")
        
    if examine:
        with model:
            pm.traceplot(trace);
            
    return model, trace


## === cell 7
def generate_template(data):
    pred_template = []
    for i, patient in enumerate(data['Patient'].unique()):
        df = pd.DataFrame(columns=['PatientID', 'Weeks', 'Patient', 'Class'])
        df['Weeks'] = np.arange(-12, 134)
        df['Patient'] = patient
        df['Class'] = data.loc[data['Patient']==patient]['Class'].max()
        pred_template.append(df)
    pred_template = pd.concat(pred_template, ignore_index=True)
    pred_template['PatientID'] = le_id.transform(pred_template['Patient'])
    return pred_template


## === cell 8
template_train_test = generate_template(test)
template_train_test.head()


## === cell 10
def model_predict(model, trace, template):
    with model:
        pm.set_data({
            "PatientID_shared": template['PatientID'].values.astype(int),
            "Weeks_shared": template['Weeks'].values.astype(int),
            "FVC_obs_shared": np.zeros(len(template)).astype(int),
            "patient_class_shared": template['Class'].values.astype(int),
        })
        post_pred = pm.sample_posterior_predictive(trace)
    df = pd.DataFrame(columns=['Patient', 'Weeks', 'FVC_pred', 'sigma'])
    df['Patient'] = le_id.inverse_transform(template['PatientID'])
    df['Weeks'] = template['Weeks']
    df['FVC_pred'] = post_pred['FVC_like'].T.mean(axis=1)
    df['sigma'] = (post_pred['FVC_like'].T.std(axis=1))*1.5
    df['FVC_inf'] = df['FVC_pred'] - df['sigma']
    df['FVC_sup'] = df['FVC_pred'] + df['sigma']
    df = pd.merge(df, train[['Patient', 'Weeks', 'FVC']], how='left', on=['Patient', 'Weeks'])
    df = df.rename(columns={'FVC': 'FVC_true'})
    return df


## === cell 11
def examine_predictions(data):
    n = (data['Patient'].nunique())+1
    f, axes = plt.subplots((n//3)+1, 3, figsize=(15, 5*((n//3)+1)))
    for i, patient in enumerate(data['Patient'].unique()):
        ax = axes[i//3, i%3]
        df = data[data['Patient'] == patient]
        x = df['Weeks']
        ax.set_title(patient)
        ax.plot(x, df['FVC_true'], 'o')
        ax.plot(x, df['FVC_pred'])
        ax = sns.regplot(x, df['FVC_true'], ax=ax, ci=None, line_kws={'color':'red'})
        ax.fill_between(x, df["FVC_inf"], df["FVC_sup"],alpha=0.5, color='#ffcd3c')
        ax.set_ylabel('FVC')
    axes[n//3,n%3].plot()


## === cell 12
def evaluate_predictions(df, use_only_last_3_measures=False, examine=False):
    if use_only_last_3_measures:
        y = df.dropna().groupby('Patient').tail(3)
    else:
        y = df.dropna()

    rmse = ((y['FVC_pred'] - y['FVC_true']) ** 2).mean() ** (1/2)
    mae = (y['FVC_pred'] - y['FVC_true']).abs()
    mae_mean = (np.sqrt((y['FVC_pred'] - y['FVC_true']) ** 2)).mean()
    mae_sd = (np.sqrt((y['FVC_pred'] - y['FVC_true']) ** 2)).mean()
    mae_max = (np.sqrt((y['FVC_pred'] - y['FVC_true']) ** 2)).mean()
    sigma_c = y['sigma'].values
    sigma_c[sigma_c < 70] = 70
    delta = (y['FVC_pred'] - y['FVC_true']).abs()
    delta[delta > 1000] = 1000
    lll = - np.sqrt(2) * delta / sigma_c - np.log(np.sqrt(2) * sigma_c)

    y['sigma_c'] = y['sigma']
    y['sigma_c'].values[y['sigma_c'].values < 70] = 70
    y['delta_c'] = (y['FVC_pred'] - y['FVC_true']).abs()
    y['delta_c'].values[y['delta_c'].values > 1000] = 1000
    y['main_loss'] = y['delta_c']/y['sigma_c']
    if examine:
        plt.hist(y['main_loss'], bins=100)


    return lll.mean()


## === cell 13
def evaluation_cycle(train, valid, examine_trace=True, examine_preds=True):
    print("Fit model ...")
    model,trace = model_fit(train, examine=examine_trace)

    print("Examine true vs predictions for training data ...")
    template_train = generate_template(train)
    template_train.head()
    pred_train = model_predict(model,trace,template_train)
    if examine_preds:
        examine_predictions(pred_train)
    lll_train = evaluate_predictions(pred_train)

    if valid is not None:
        print("Examine true vs predictions for validation data ...")
        template_valid = generate_template(valid)
        pred_valid = model_predict(model,trace,template_valid)
        if examine_preds:
            examine_predictions(pred_valid)
        lll_valid = evaluate_predictions(pred_valid)
    return pred_train, pred_valid, lll_train, lll_valid


## === cell 14
for fold in range(0):
    examine_lll = True

    all_patients = train['Patient'].unique()
    validation_patients = np.random.choice(all_patients, size=20, replace=False)
    df_valid = train[train['Patient'].isin(validation_patients)]
    df_train = train[~train['Patient'].isin(validation_patients)]

    df_valid_first_readings = df_valid.groupby('Patient').head(1)
    df_train = pd.concat([df_train, df_valid_first_readings], axis=0, ignore_index=True)

    print(f'Fold: {fold}')
    pred_train, pred_valid, lll_train, lll_valid = evaluation_cycle(df_train, df_valid, examine_trace=True, examine_preds=True)

    print(f'Laplace Log Likelihoods for fold: {fold}')
    print(f'Training:     {lll_train:.4f}')
    print(f'Validation:   {lll_valid:.4f}')
    print("")

    evaluate_predictions(pred_train, use_only_last_3_measures=True, examine=examine_lll)
    evaluate_predictions(pred_valid, use_only_last_3_measures=True, examine=examine_lll)


## === cell 15
if pm is None:
    raise RuntimeError(
        "pymc3 could not be imported (pm is None), so Bayesian model fitting cannot run in this environment. "
        "Install/enable a compatible PyMC3/Theano stack or run in an environment where pymc3 imports successfully."
    )

print("Fit model ...")
model, trace = model_fit(train, examine=False)
print("")

print("Make predictions for test data ...")
template_test = generate_template(test)
template_test.head()
pred_test = model_predict(model, trace, template_test)

final = pd.DataFrame(columns=["Patient_Week", "FVC", "Confidence"])
final["Patient_Week"] = pred_test["Patient"] + "_" + pred_test["Weeks"].astype(str)
final["FVC"] = pred_test["FVC_pred"]
final["Confidence"] = pred_test["sigma"]
final.head()
final.to_csv("submission.csv", index=False)
print(final.shape)
final.head()


## --- ERROR in cell 15, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mRuntimeError[0m                              Traceback (most recent call last)
[0;32m/tmp/ipykernel_10/2666341638.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      2[0m [0;31m# Without this guard, pm is None and model_fit crashes with: 'NoneType' object has no attribute 'Model'.[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m [0;32mif[0m [0mpm[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 4[0;31m     raise RuntimeError(
[0m[1;32m      5[0m         [0;34m"pymc3 could not be imported (pm is None), so Bayesian model fitting cannot run in this environment. "[0m[0;34m[0m[0;34m[0m[0m
[1;32m      6[0m         [0;34m"Install/enable a compatible PyMC3/Theano stack or run in an environment where pymc3 imports successfully."[0m[0;34m[0m[0;34m[0m[0m

[0;31mRuntimeError[0m: pymc3 could not be imported (pm is None), so Bayesian model fitting cannot run in this environment. Install/enable a compatible PyMC3/Theano stack or run in an environment where pymc3 imports successfully.
