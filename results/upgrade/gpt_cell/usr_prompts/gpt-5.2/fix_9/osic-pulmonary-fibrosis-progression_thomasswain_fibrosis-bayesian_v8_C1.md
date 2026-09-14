# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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
pymc3==3.11.4
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0

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

-6.9351

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import numpy as np  # linear algebra

if not hasattr(np, "bool"):
    np.bool = np.bool_

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

os.environ.setdefault(
    "THEANO_FLAGS",
    "gcc__cxxflags=,cxx=,optimizer=fast_compile,device=cpu,mode=FAST_RUN",
)

import theano  # must be imported before pymc3 so we can patch config safely

if not hasattr(theano.config, "gcc__cxxflags"):
    theano.config.gcc__cxxflags = ""

try:
    import theano.printing as _theano_printing  # noqa: F401

    if not hasattr(_theano_printing, "Node"):
        try:
            import pydot as _pydot  # noqa: F401

            _theano_printing.Node = _pydot.Node
        except Exception:

            class _TheanoPrintingNodeStub:  # pragma: no cover
                pass

            _theano_printing.Node = _TheanoPrintingNodeStub
except Exception:
    pass

try:
    import theano.sandbox.rng_mrg as _rng_mrg  # noqa: F401

    if not hasattr(_rng_mrg, "MRG_RandomStream") and hasattr(
        _rng_mrg, "MRG_RandomStreams"
    ):
        _rng_mrg.MRG_RandomStream = _rng_mrg.MRG_RandomStreams
except Exception:
    pass

try:
    import pymc3 as pm
except Exception as e:
    pm = None
    print(f"WARNING: pymc3 could not be imported and will be unavailable: {e}")

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))


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
def model_fit(data, examine=True):
    n_patients = data['Patient'].nunique()
    FVC_obs = data['FVC'].values
    Weeks = data['Weeks'].values
    PatientID = data['PatientID'].values

    with pm.Model() as model:
        FVC_obs_shared = pm.Data("FVC_obs_shared", FVC_obs)
        Weeks_shared = pm.Data('Weeks_shared', Weeks)
        PatientID_shared = pm.Data('PatientID_shared', PatientID)

        mu_a = pm.Normal('mu_a', mu=1700., sigma=400)
        sigma_a = pm.HalfNormal('sigma_a', 1000.)
        mu_b = pm.Normal('mu_b', mu=-4., sigma=1)
        sigma_b = pm.HalfNormal('sigma_b', 5.)

        a = pm.Normal('a', mu=mu_a, sigma=sigma_a, shape=n_patients)
        b = pm.Normal('b', mu=mu_b, sigma=sigma_b, shape=n_patients)

        sigma = pm.HalfNormal('sigma', 150.)
        
        FVC_est = a[PatientID_shared] + b[PatientID_shared] * Weeks_shared

        FVC_like = pm.Normal('FVC_like', mu=FVC_est,
                             sigma=sigma, observed=FVC_obs_shared)

        trace = pm.sample(2000, tune=2000, target_accept=.9, init="adapt_diag")
    if examine:
        with model:
            pm.traceplot(trace);
    return model, trace


## === cell 4
def generate_template(data):
    pred_template = []
    for i, patient in enumerate(data['Patient'].unique()):
        df = pd.DataFrame(columns=['PatientID', 'Weeks', 'Patient'])
        df['Weeks'] = np.arange(-12, 134)
        df['Patient'] = patient
        pred_template.append(df)
    pred_template = pd.concat(pred_template, ignore_index=True)
    pred_template['PatientID'] = le_id.transform(pred_template['Patient'])
    return pred_template


## === cell 5
def model_predict(model, trace, template):
    with model:
        pm.set_data({
            "PatientID_shared": template['PatientID'].values.astype(int),
            "Weeks_shared": template['Weeks'].values.astype(int),
            "FVC_obs_shared": np.zeros(len(template)).astype(int),
        })
        post_pred = pm.sample_posterior_predictive(trace)
    df = pd.DataFrame(columns=['Patient', 'Weeks', 'FVC_pred', 'sigma'])
    df['Patient'] = le_id.inverse_transform(template['PatientID'])
    df['Weeks'] = template['Weeks']
    df['FVC_pred'] = post_pred['FVC_like'].T.mean(axis=1)
    df['sigma'] = post_pred['FVC_like'].T.std(axis=1)
    df['FVC_inf'] = df['FVC_pred'] - df['sigma']
    df['FVC_sup'] = df['FVC_pred'] + df['sigma']
    df = pd.merge(df, train[['Patient', 'Weeks', 'FVC']], how='left', on=['Patient', 'Weeks'])
    df = df.rename(columns={'FVC': 'FVC_true'})
    return df


## === cell 6
def examine_predictions(data):
    f, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, patient in enumerate(np.random.choice(data['Patient'].unique(), size=3, replace=False)):
        ax = axes[i]
        df = data[data['Patient'] == patient]
        x = df['Weeks']
        ax.set_title(patient)
        ax.plot(x, df['FVC_true'], 'o')
        ax.plot(x, df['FVC_pred'])
        ax = sns.regplot(x, df['FVC_true'], ax=ax, ci=None, line_kws={'color':'red'})
        ax.fill_between(x, df["FVC_inf"], df["FVC_sup"],alpha=0.5, color='#ffcd3c')
        ax.set_ylabel('FVC')


## === cell 7
def evaluate_predictions(df, use_only_last_3_measures=True, examine=False):
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
        sns.distplot(y['main_loss'], bins=100)


    return lll.mean()


## === cell 8
def evaluation_cycle(train, valid, examine_trace=True, examine_preds=True):
    print("Fit model ...")
    model,trace = model_fit(train, examine=examine_trace)
    print("")

    print("Examine true vs predictions for training data ...")
    template_train = generate_template(train)
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


## === cell 10
if pm is None:
    train_only = train_raw.copy()
    per_patient_params = {}
    per_patient_sigma = {}

    for patient, g in train_only.groupby("Patient"):
        g = g.dropna(subset=["Weeks", "FVC"])
        x = g["Weeks"].astype(float).values
        y = g["FVC"].astype(float).values

        if len(g) >= 2 and np.std(x) > 0:
            m, c = np.polyfit(x, y, 1)
            y_hat = m * x + c
            rmse = float(np.sqrt(np.mean((y_hat - y) ** 2)))
            per_patient_params[patient] = (c, m)
            per_patient_sigma[patient] = max(rmse, 70.0)
        else:
            c = float(np.mean(y)) if len(y) else 2000.0
            m = 0.0
            per_patient_params[patient] = (c, m)
            per_patient_sigma[patient] = 70.0

    preds = []
    for patient in test["Patient"].unique():
        weeks = np.arange(-12, 134)
        c, m = per_patient_params.get(patient, (2000.0, 0.0))
        fvc_pred = c + m * weeks
        sigma = per_patient_sigma.get(patient, 70.0)
        dfp = pd.DataFrame(
            {"Patient": patient, "Weeks": weeks, "FVC_pred": fvc_pred, "sigma": sigma}
        )
        preds.append(dfp)
    pred_test = pd.concat(preds, ignore_index=True)

    final = pd.DataFrame(columns=["Patient_Week", "FVC", "Confidence"])
    final["Patient_Week"] = pred_test["Patient"] + "_" + pred_test["Weeks"].astype(str)
    final["FVC"] = pred_test["FVC_pred"]
    final["Confidence"] = pred_test["sigma"]
    final.to_csv("submission.csv", index=False)
    print(final.shape)
    final.head()
else:
    print("Fit model ...")
    model, trace = model_fit(train, examine=False)
    print("")

    print("Examine true vs predictions for validation data ...")
    template_test = generate_template(test)
    pred_test = model_predict(model, trace, template_test)

    final = pd.DataFrame(columns=["Patient_Week", "FVC", "Confidence"])
    final["Patient_Week"] = pred_test["Patient"] + "_" + pred_test["Weeks"].astype(str)
    final["FVC"] = pred_test["FVC_pred"]
    final["Confidence"] = pred_test["sigma"]
    final.head()
    final.to_csv("submission.csv", index=False)
    print(final.shape)
    final.head()
