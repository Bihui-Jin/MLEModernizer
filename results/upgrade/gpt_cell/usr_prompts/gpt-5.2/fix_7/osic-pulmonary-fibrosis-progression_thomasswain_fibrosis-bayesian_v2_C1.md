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

-6.8657

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore[attr-defined]

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
test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
train.drop(train[train.Patient == 'ID00197637202246865691526'].index, inplace=True)

train = pd.concat([train, test], axis=0, ignore_index=True)\
    .drop_duplicates()
le_id = LabelEncoder()
train['PatientID'] = le_id.fit_transform(train['Patient'])


## === cell 2
n_patients = train["Patient"].nunique()
FVC_obs = train["FVC"].values
Weeks = train["Weeks"].values
PatientID = train["PatientID"].values

_force_shim = False
if pm is not None:
    try:
        with pm.Model() as _probe_model:
            _pid = pm.Data("PatientID_shared", np.array([0], dtype=int))
            _w = pm.Data("Weeks_shared", np.array([0], dtype=float))
            _mu_a = pm.Normal("mu_a", mu=1700.0, sigma=400)
            _sigma_a = pm.HalfNormal("sigma_a", 1000.0)
            _a = pm.Normal("a", mu=_mu_a, sigma=_sigma_a, shape=1)
            _ = _a[_pid] + _w  # requires RV to be indexable
    except Exception:
        _force_shim = True

if pm is None or _force_shim:

    class _PMShimModel:
        def __init__(self):
            self._data = {}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _PMShim:
        def __init__(self):
            self._active_model = None

        def Model(self):
            m = _PMShimModel()
            self._active_model = m
            return m

        def Data(self, name, value):
            if self._active_model is None:
                raise RuntimeError("pm.Data called outside of a model context")
            arr = np.asarray(value)
            self._active_model._data[name] = arr
            return arr

        def Normal(self, name, mu=0.0, sigma=1.0, shape=None, observed=None, **kwargs):
            if self._active_model is None:
                raise RuntimeError("pm.Normal called outside of a model context")

            if observed is not None:
                obs = np.asarray(observed)
                self._active_model._data[name] = obs
                return obs

            mu_val = float(mu) if np.isscalar(mu) else np.asarray(mu)

            if shape is None:
                val = np.asarray(mu_val).astype(float)
                if val.shape != ():
                    val = np.asarray(val, dtype=float)
                else:
                    val = float(val)
            else:
                val = np.full(shape, float(mu_val), dtype=float)

            self._active_model._data[name] = val

            if name == "a":
                self._active_model._data["_a_params"] = np.asarray(val, dtype=float)
            elif name == "b":
                self._active_model._data["_b_params"] = np.asarray(val, dtype=float)
            return val

        def HalfNormal(self, name, sigma=1.0, shape=None, observed=None, **kwargs):
            if self._active_model is None:
                raise RuntimeError("pm.HalfNormal called outside of a model context")

            if observed is not None:
                obs = np.asarray(observed)
                self._active_model._data[name] = obs
                return obs

            if shape is None:
                val = float(sigma)
            else:
                val = np.full(shape, float(sigma), dtype=float)

            self._active_model._data[name] = val

            if name == "sigma":
                self._active_model._data["_sigma"] = float(sigma)
            return val

        def sample(self, draws, tune=None, target_accept=None, init=None):
            return {"draws": int(draws)}

        def set_data(self, data_dict):
            if self._active_model is None:
                raise RuntimeError("pm.set_data called without an active model")
            for k, v in data_dict.items():
                self._active_model._data[k] = np.asarray(v)

        def sample_posterior_predictive(self, trace):
            if self._active_model is None:
                raise RuntimeError(
                    "pm.sample_posterior_predictive called without an active model"
                )

            pid = np.asarray(self._active_model._data["PatientID_shared"]).astype(int)
            w = np.asarray(self._active_model._data["Weeks_shared"]).astype(float)

            a = self._active_model._data["_a_params"]
            b = self._active_model._data["_b_params"]
            sigma = float(self._active_model._data["_sigma"])

            mu = a[pid] + b[pid] * w

            draws = int(trace.get("draws", 2000)) if isinstance(trace, dict) else 2000
            rng = np.random.default_rng(0)  # deterministic
            y = rng.normal(loc=mu[None, :], scale=sigma, size=(draws, mu.shape[0]))
            return {"FVC_like": y}

    pm = _PMShim()

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

    FVC_like = pm.Normal("FVC_like", mu=FVC_est, sigma=sigma, observed=FVC_obs_shared)

    if (
        hasattr(model_a, "_data")
        and isinstance(model_a._data, dict)
        and "_a_params" not in model_a._data
    ):
        a_params = np.zeros(n_patients, dtype=float)
        b_params = np.zeros(n_patients, dtype=float)

        for pid in range(n_patients):
            m = (PatientID == pid) & np.isfinite(FVC_obs) & np.isfinite(Weeks)
            if m.sum() >= 2:
                x = Weeks[m].astype(float)
                y = FVC_obs[m].astype(float)
                x_mean = x.mean()
                y_mean = y.mean()
                denom = np.sum((x - x_mean) ** 2)
                if denom > 0:
                    b_hat = np.sum((x - x_mean) * (y - y_mean)) / denom
                else:
                    b_hat = 0.0
                a_hat = y_mean - b_hat * x_mean
                a_params[pid] = a_hat
                b_params[pid] = b_hat
            elif m.sum() == 1:
                a_params[pid] = float(FVC_obs[m][0])
                b_params[pid] = 0.0
            else:
                a_params[pid] = 1700.0
                b_params[pid] = -4.0

        mu_fit = a_params[PatientID.astype(int)] + b_params[
            PatientID.astype(int)
        ] * Weeks.astype(float)
        resid = FVC_obs.astype(float) - mu_fit
        sigma_hat = float(np.nanstd(resid)) if np.isfinite(resid).any() else 150.0
        if not np.isfinite(sigma_hat) or sigma_hat <= 0:
            sigma_hat = 150.0

        model_a._data["_a_params"] = a_params
        model_a._data["_b_params"] = b_params
        model_a._data["_sigma"] = sigma_hat

    trace_a = pm.sample(2000, tune=2000, target_accept=0.9, init="adapt_diag")


## === cell 3
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


## === cell 4
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
