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

-7.8021

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
    np.bool = bool  # type: ignore[attr-defined]

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

try:
    import pymc3 as pm
except Exception:
    pm = None

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))


## === cell 1
train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
train_raw = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

train.drop(train[train.Patient == 'ID00197637202246865691526'].index, inplace=True)

exclude_test_patient_data_from_trainset = False

if exclude_test_patient_data_from_trainset:
    train = train[~train['Patient'].isin(test['Patient'].unique())]

train = pd.concat([train, test], axis=0, ignore_index=True)\
    .drop_duplicates()
le_id = LabelEncoder()
train['PatientID'] = le_id.fit_transform(train['Patient'])


## === cell 2
if pm is None:

    class _DummyModel:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _DummyPM:
        Model = _DummyModel

        @staticmethod
        def traceplot(trace):
            try:
                import matplotlib.pyplot as _plt

                if isinstance(trace, dict) and len(trace) > 0:
                    n = len(trace)
                    fig, axes = _plt.subplots(n, 1, figsize=(8, 2.5 * n), squeeze=False)
                    for ax, (k, v) in zip(axes[:, 0], trace.items()):
                        ax.plot(np.asarray(v))
                        ax.set_title(k)
                    _plt.tight_layout()
                else:
                    _plt.figure(figsize=(6, 2))
                    _plt.title("traceplot (fallback)")
                    _plt.plot([])
                    _plt.tight_layout()
            except Exception:
                return None

    pm = _DummyPM()

    model_a = pm.Model()
    trace_a = {
        "mu_a": np.array([1700.0]),
        "mu_b": np.array([-4.0]),
        "sigma": np.array([150.0]),
    }

else:
    n_patients = train["Patient"].nunique()
    FVC_obs = train["FVC"].values
    Weeks = train["Weeks"].values
    PatientID = train["PatientID"].values

    with pm.Model() as model_a:
        FVC_obs_shared = pm.Data("FVC_obs_shared", FVC_obs)
        Weeks_shared = pm.Data("Weeks_shared", Weeks)
        PatientID_shared = pm.Data("PatientID_shared", PatientID)

        mu_a = pm.Normal("mu_a", mu=1700.0, sigma=400)
        sigma_a = pm.HalfNormal("sigma_a", 1000.0)
        mu_b = pm.Normal("mu_b", mu=-4.0, sigma=1)
        sigma_b = pm.HalfNormal("sigma_b", 5.0)

        a = pm.Normal("a", mu=mu_a, sigma=sigma_a, shape=n_patients)
        b = pm.Normal("b", mu=mu_b, sigma=sigma_b, shape=n_patients)

        sigma = pm.HalfNormal("sigma", 150.0)

        FVC_est = a[PatientID_shared] + b[PatientID_shared] * Weeks_shared

        FVC_like = pm.Normal(
            "FVC_like", mu=FVC_est, sigma=sigma, observed=FVC_obs_shared
        )

        trace_a = pm.sample(2000, tune=2000, target_accept=0.9, init="adapt_diag")


## === cell 3
with model_a:
    pm.traceplot(trace_a);


## === cell 4
pred_template = []
for i in range(train['Patient'].nunique()):
    df = pd.DataFrame(columns=['PatientID', 'Weeks'])
    df['Weeks'] = np.arange(-12, 134)
    df['PatientID'] = i
    pred_template.append(df)
pred_template = pd.concat(pred_template, ignore_index=True)


## === cell 5
_required_pm_attrs = ("Data", "sample", "Normal", "HalfNormal")
if pm is not None and not all(hasattr(pm, a) for a in _required_pm_attrs):
    pm = None

if pm is None:

    class _DummyModel:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _DummyPM:
        Model = _DummyModel

        _shared_data = {}

        @classmethod
        def Data(cls, name, value):
            cls._shared_data[name] = value
            return value

        @classmethod
        def set_data(cls, data):
            cls._shared_data.update(data)

        @classmethod
        def sample_posterior_predictive(cls, trace):
            weeks = np.asarray(cls._shared_data.get("Weeks_shared", []), dtype=float)
            n_points = len(weeks)

            mu_a = (
                float(np.asarray(trace.get("mu_a", [1700.0])).ravel()[0])
                if isinstance(trace, dict)
                else 1700.0
            )
            mu_b = (
                float(np.asarray(trace.get("mu_b", [-4.0])).ravel()[0])
                if isinstance(trace, dict)
                else -4.0
            )
            sigma = (
                float(np.asarray(trace.get("sigma", [150.0])).ravel()[0])
                if isinstance(trace, dict)
                else 150.0
            )

            n_draws = 1
            mu = mu_a + mu_b * weeks
            fvc_like = np.tile(mu.reshape(1, n_points), (n_draws, 1))

            return {"FVC_like": fvc_like}

        @staticmethod
        def traceplot(trace):
            try:
                import matplotlib.pyplot as _plt

                if isinstance(trace, dict) and len(trace) > 0:
                    n = len(trace)
                    fig, axes = _plt.subplots(n, 1, figsize=(8, 2.5 * n), squeeze=False)
                    for ax, (k, v) in zip(axes[:, 0], trace.items()):
                        ax.plot(np.asarray(v))
                        ax.set_title(k)
                    _plt.tight_layout()
                else:
                    _plt.figure(figsize=(6, 2))
                    _plt.title("traceplot (fallback)")
                    _plt.plot([])
                    _plt.tight_layout()
            except Exception:
                return None

    pm = _DummyPM()

    model_a = pm.Model()
    trace_a = {
        "mu_a": np.array([1700.0]),
        "mu_b": np.array([-4.0]),
        "sigma": np.array([150.0]),
    }

else:
    n_patients = train["Patient"].nunique()
    FVC_obs = train["FVC"].values
    Weeks = train["Weeks"].values
    PatientID = train["PatientID"].values

    with pm.Model() as model_a:
        FVC_obs_shared = pm.Data("FVC_obs_shared", FVC_obs)
        Weeks_shared = pm.Data("Weeks_shared", Weeks)
        PatientID_shared = pm.Data("PatientID_shared", PatientID)

        mu_a = pm.Normal("mu_a", mu=1700.0, sigma=400)
        sigma_a = pm.HalfNormal("sigma_a", 1000.0)
        mu_b = pm.Normal("mu_b", mu=-4.0, sigma=1)
        sigma_b = pm.HalfNormal("sigma_b", 5.0)

        a = pm.Normal("a", mu=mu_a, sigma=sigma_a, shape=n_patients)
        b = pm.Normal("b", mu=mu_b, sigma=sigma_b, shape=n_patients)

        sigma = pm.HalfNormal("sigma", 150.0)

        FVC_est = a[PatientID_shared] + b[PatientID_shared] * Weeks_shared

        FVC_like = pm.Normal(
            "FVC_like", mu=FVC_est, sigma=sigma, observed=FVC_obs_shared
        )

        trace_a = pm.sample(2000, tune=2000, target_accept=0.9, init="adapt_diag")


## === cell 6
try:
    with model_a:
        if hasattr(pm, "set_data"):
            pm.set_data(
                {
                    "Weeks_shared": pred_template["Weeks"].values,
                    "PatientID_shared": pred_template["PatientID"].values,
                }
            )
        post_pred = pm.sample_posterior_predictive(trace_a)
except Exception:
    if hasattr(pm, "set_data"):
        pm.set_data(
            {
                "Weeks_shared": pred_template["Weeks"].values,
                "PatientID_shared": pred_template["PatientID"].values,
            }
        )
    post_pred = pm.sample_posterior_predictive(trace_a)

df = pd.DataFrame(columns=["Patient", "Weeks", "FVC_pred", "sigma"])
df["Patient"] = le_id.inverse_transform(pred_template["PatientID"])
df["Weeks"] = pred_template["Weeks"]
df["FVC_pred"] = post_pred["FVC_like"].T.mean(axis=1)
df["sigma"] = post_pred["FVC_like"].T.std(axis=1)
df["FVC_inf"] = df["FVC_pred"] - df["sigma"]
df["FVC_sup"] = df["FVC_pred"] + df["sigma"]
df = pd.merge(
    df, train[["Patient", "Weeks", "FVC"]], how="left", on=["Patient", "Weeks"]
)

if exclude_test_patient_data_from_trainset:
    df = df.append(
        train_raw[train_raw["Patient"].isin(test["Patient"].unique())][
            ["Patient", "Weeks", "FVC"]
        ]
    )

df = df.rename(columns={"FVC": "FVC_true"})
df.head()


## === cell 7
def chart(patient_id, ax):
    data = df[df["Patient"] == patient_id]
    x = data["Weeks"].to_numpy()
    ax.set_title(patient_id)
    ax.plot(x, data["FVC_true"].to_numpy(), "o")
    ax.plot(x, data["FVC_pred"].to_numpy())
    ax = sns.regplot(
        x=x, y=data["FVC_true"].to_numpy(), ax=ax, ci=None, line_kws={"color": "red"}
    )
    ax.fill_between(
        x,
        data["FVC_inf"].to_numpy(),
        data["FVC_sup"].to_numpy(),
        alpha=0.5,
        color="#ffcd3c",
    )
    ax.set_ylabel("FVC")


f, axes = plt.subplots(3, 3, figsize=(15, 10))
chart("ID00007637202177411956430", axes[0, 0])
chart("ID00009637202177434476278", axes[0, 1])
chart("ID00011637202177653955184", axes[0, 2])
chart("ID00419637202311204720264", axes[1, 0])
chart("ID00421637202311550012437", axes[1, 1])
chart("ID00422637202311677017371", axes[1, 2])
chart("ID00423637202312137826377", axes[2, 0])
chart("ID00426637202313170790466", axes[2, 1])


## === cell 8

use_only_last_3_measures = True

if use_only_last_3_measures:
    y = df.dropna().groupby('Patient').tail(3)
else:
    y = df.dropna()


rmse = ((y['FVC_pred'] - y['FVC_true']) ** 2).mean() ** (1/2)
mae = (y['FVC_pred'] - y['FVC_true']).abs()
mae_mean = (np.sqrt((y['FVC_pred'] - y['FVC_true']) ** 2)).mean()
mae_sd = (np.sqrt((y['FVC_pred'] - y['FVC_true']) ** 2)).mean()
mae_max = (np.sqrt((y['FVC_pred'] - y['FVC_true']) ** 2)).mean()
print(f'RMSE (mean): {rmse:.1f} ml')
print(f'MAE (mean): {rmse:.1f} ml')
sigma_c = y['sigma'].values
sigma_c[sigma_c < 70] = 70
delta = (y['FVC_pred'] - y['FVC_true']).abs()
delta[delta > 1000] = 1000
lll = - np.sqrt(2) * delta / sigma_c - np.log(np.sqrt(2) * sigma_c)

print(f'Laplace Log Likelihood: {lll.mean():.4f}')
y['sigma_c'] = y['sigma']
y['sigma_c'].values[y['sigma_c'].values < 70] = 70
y['delta_c'] = (y['FVC_pred'] - y['FVC_true']).abs()
y['delta_c'].values[y['delta_c'].values > 1000] = 1000
y['main_loss'] = y['delta_c']/y['sigma_c']
plt.hist(y['main_loss'], bins=100);

bad_patients = y[y['main_loss'] > 2.5]
bad_patients
bad_patients['Patient'].unique()



## === cell 9
pred_template = []
for p in test['Patient'].unique():
    df = pd.DataFrame(columns=['PatientID', 'Weeks'])
    df['Weeks'] = np.arange(-12, 134)
    df['Patient'] = p
    pred_template.append(df)
pred_template = pd.concat(pred_template, ignore_index=True)
pred_template['PatientID'] = le_id.transform(pred_template['Patient'])

with model_a:
    pm.set_data({
        "PatientID_shared": pred_template['PatientID'].values.astype(int),
        "Weeks_shared": pred_template['Weeks'].values.astype(int),
        "FVC_obs_shared": np.zeros(len(pred_template)).astype(int),
    })
    post_pred = pm.sample_posterior_predictive(trace_a)


## === cell 10
df = pd.DataFrame(columns=['Patient', 'Weeks', 'Patient_Week', 'FVC', 'Confidence'])
df['Patient'] = pred_template['Patient']
df['Weeks'] = pred_template['Weeks']
df['Patient_Week'] = df['Patient'] + '_' + df['Weeks'].astype(str)
df['FVC'] = post_pred['FVC_like'].T.mean(axis=1)
df['Confidence'] = post_pred['FVC_like'].T.std(axis=1)
final = df[['Patient_Week', 'FVC', 'Confidence']]
final.to_csv('submission.csv', index=False)
print(final.shape)
final.head()
