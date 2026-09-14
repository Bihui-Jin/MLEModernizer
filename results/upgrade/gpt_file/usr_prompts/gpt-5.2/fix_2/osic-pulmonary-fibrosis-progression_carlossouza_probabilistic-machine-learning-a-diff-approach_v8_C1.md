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

arviz==0.21.0
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
Theano==1.0.5
Theano-PyMC==1.1.2

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

-6.9569

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(42)



## === cell 1
BASE_PATHS = [
    "../input/osic-pulmonary-fibrosis-progression",
    "/kaggle/input/osic-pulmonary-fibrosis-progression",
    "../kaggle/data/osic-pulmonary-fibrosis-progression",
]
base_path = None
for p in BASE_PATHS:
    if os.path.exists(os.path.join(p, "train.csv")):
        base_path = p
        break
if base_path is None:
    base_path = "../input"

train = pd.read_csv(
    os.path.join(base_path, "train.csv")
    if base_path.endswith("osic-pulmonary-fibrosis-progression") is False
    else os.path.join(base_path, "train.csv")
)
test = pd.read_csv(
    os.path.join(base_path, "test.csv")
    if base_path.endswith("osic-pulmonary-fibrosis-progression") is False
    else os.path.join(base_path, "test.csv")
)

if "Patient" not in train.columns or "Patient" not in test.columns:
    nested = os.path.join(base_path, "osic-pulmonary-fibrosis-progression")
    train = pd.read_csv(os.path.join(nested, "train.csv"))
    test = pd.read_csv(os.path.join(nested, "test.csv"))
    base_path = nested

sample_sub = pd.read_csv(os.path.join(base_path, "sample_submission.csv"))

train.shape, test.shape, sample_sub.shape




## === cell 2
def chart(patient_id, ax):
    data = train[train["Patient"] == patient_id]
    x = data["Weeks"]
    y = data["FVC"]
    ax.set_title(patient_id)
    sns.regplot(x=x, y=y, ax=ax, ci=None, line_kws={"color": "red"})


f, axes = plt.subplots(1, 3, figsize=(15, 5))
chart("ID00007637202177411956430", axes[0])
chart("ID00009637202177434476278", axes[1])
chart("ID00010637202177584971671", axes[2])
plt.tight_layout()



## === cell 3
if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore[attr-defined]

import pymc3 as pm
import theano
import arviz as az
from sklearn import preprocessing




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3023865651.py in <cell line: 0>()
      4     np.bool = bool  # type: ignore[attr-defined]
      5 
----> 6 import pymc3 as pm
      7 import theano
      8 import arviz as az

/usr/local/lib/python3.11/dist-packages/pymc3/__init__.py in <module>
     77 
     78 _check_backend_version()
---> 79 __set_compiler_flags()
     80 _hotfix_theano_printing()
     81 

/usr/local/lib/python3.11/dist-packages/pymc3/__init__.py in __set_compiler_flags()
     59 def __set_compiler_flags():
     60     # Workarounds for Theano compiler problems on various platforms
---> 61     current = theano.config.gcc__cxxflags
     62     theano.config.gcc__cxxflags = f"{current} -Wno-c++11-narrowing"
     63 

AttributeError: 'TheanoConfigParser' object has no attribute 'gcc__cxxflags'

## === cell 4
def patient_class_fn(row):
    if row["Sex"] == "Male":
        if row["SmokingStatus"] == "Currently smokes":
            return 0
        elif row["SmokingStatus"] == "Ex-smoker":
            return 1
        elif row["SmokingStatus"] == "Never smoked":
            return 2
    else:
        if row["SmokingStatus"] == "Currently smokes":
            return 3
        elif row["SmokingStatus"] == "Ex-smoker":
            return 4
        elif row["SmokingStatus"] == "Never smoked":
            return 5
    return 0


train["Class"] = train.apply(patient_class_fn, axis=1)



## === cell 5
aux = train[["Patient", "Weeks"]].groupby("Patient").min().reset_index()
aux = pd.merge(
    aux, train[["Patient", "Weeks", "FVC"]], how="left", on=["Patient", "Weeks"]
)
aux = aux.groupby("Patient").mean().reset_index()
aux["Weeks"] = aux["Weeks"].astype(int)
aux["FVC"] = aux["FVC"].astype(int)
train = pd.merge(train, aux, how="left", on="Patient", suffixes=("", "_base"))



## === cell 6
le = preprocessing.LabelEncoder()
train["PatientID"] = le.fit_transform(train["Patient"])

patients = train[
    ["Patient", "PatientID", "Age", "Class", "Weeks_base", "FVC_base"]
].drop_duplicates()
fvc_data = train[["Patient", "PatientID", "Weeks", "FVC"]]

patients.head()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3671072237.py in <cell line: 0>()
      1 # Fix: preprocessing wasn't available earlier due to failed import; now it is.
----> 2 le = preprocessing.LabelEncoder()
      3 train["PatientID"] = le.fit_transform(train["Patient"])
      4 
      5 patients = train[

NameError: name 'preprocessing' is not defined

## === cell 7
fvc_data.head()



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/769944347.py in <cell line: 0>()
----> 1 fvc_data.head()
      2 

NameError: name 'fvc_data' is not defined

## === cell 8
FVC_b = patients["FVC_base"].values
w_b = patients["Weeks_base"].values
age = patients["Age"].values
patient_class_arr = patients["Class"].values

t = fvc_data["Weeks"].values
FVC_obs = fvc_data["FVC"].values
patient_id = fvc_data["PatientID"].values

with pm.Model() as hierarchical_model:
    beta_int = pm.Normal("beta_int", 0, sigma=100)
    sigma_int = pm.HalfNormal("sigma_int", 100)

    mu_alpha = FVC_b + beta_int * w_b
    alpha = pm.Normal(
        "alpha", mu=mu_alpha, sigma=sigma_int, shape=train["Patient"].nunique()
    )

    sigma_s = pm.HalfNormal("sigma_s", 100)
    alpha_s = pm.Normal("alpha_s", 0, sigma=100)
    beta_cs = pm.Normal("beta_cs", 0, sigma=100, shape=6)

    mu_beta = alpha_s + age * beta_cs[patient_class_arr]
    beta = pm.Normal(
        "beta", mu=mu_beta, sigma=sigma_s, shape=train["Patient"].nunique()
    )

    sigma = pm.HalfNormal("sigma", 200)

    FVC_est = alpha[patient_id] + beta[patient_id] * t

    FVC_like = pm.Normal("FVC_like", mu=FVC_est, sigma=sigma, observed=FVC_obs)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2720251023.py in <cell line: 0>()
----> 1 FVC_b = patients["FVC_base"].values
      2 w_b = patients["Weeks_base"].values
      3 age = patients["Age"].values
      4 patient_class_arr = patients["Class"].values
      5 

NameError: name 'patients' is not defined

## === cell 9
with hierarchical_model:
    trace = pm.sample(
        800,
        tune=800,
        target_accept=0.9,
        chains=2,
        cores=2,
        progressbar=True,
        random_seed=42,
    )



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3928131550.py in <cell line: 0>()
      1 # Runtime fix: original 2000/2000 can exceed 600s on CPU; reduce draws/tune to finish reliably.
      2 # This keeps the same model and inference method; only MC precision changes.
----> 3 with hierarchical_model:
      4     trace = pm.sample(
      5         800,

NameError: name 'hierarchical_model' is not defined

## === cell 10
try:
    with hierarchical_model:
        pm.traceplot(trace)
    plt.tight_layout()
except Exception as e:
    print("Traceplot skipped due to:", repr(e))




## === cell 11
def chart_with_samples(patient_id, ax):
    data = train[train["Patient"] == patient_id]
    x = data["Weeks"]
    y = data["FVC"]
    ax.set_title(patient_id)
    sns.regplot(x=x, y=y, ax=ax, ci=None, line_kws={"color": "red"})

    x2 = np.arange(-12, 133, step=0.5)

    pid_row = patients[patients["Patient"] == patient_id]["PatientID"].values
    if len(pid_row) == 0:
        return
    pid = pid_row[0]

    n_draws = min(100, trace["alpha"].shape[0])
    for sample in range(n_draws):
        alpha_samp = trace["alpha"][sample, pid]
        beta_samp = trace["beta"][sample, pid]
        sigma_samp = trace["sigma"][sample]
        y2 = alpha_samp + beta_samp * x2
        ax.plot(x2, y2, linewidth=0.2, color="green", alpha=0.5)
        ax.plot(x2, y2 + sigma_samp, linewidth=0.2, color="yellow", alpha=0.4)
        ax.plot(x2, y2 - sigma_samp, linewidth=0.2, color="yellow", alpha=0.4)


f, axes = plt.subplots(1, 3, figsize=(15, 5))
chart_with_samples("ID00007637202177411956430", axes[0])
chart_with_samples("ID00009637202177434476278", axes[1])
chart_with_samples("ID00010637202177584971671", axes[2])
plt.tight_layout()



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2881821711.py in <cell line: 0>()
     27 
     28 f, axes = plt.subplots(1, 3, figsize=(15, 5))
---> 29 chart_with_samples("ID00007637202177411956430", axes[0])
     30 chart_with_samples("ID00009637202177434476278", axes[1])
     31 chart_with_samples("ID00010637202177584971671", axes[2])

/tmp/ipykernel_11/2881821711.py in chart_with_samples(patient_id, ax)
      9     x2 = np.arange(-12, 133, step=0.5)
     10 
---> 11     pid_row = patients[patients["Patient"] == patient_id]["PatientID"].values
     12     if len(pid_row) == 0:
     13         return

NameError: name 'patients' is not defined

## === cell 12
test["Class"] = test.apply(patient_class_fn, axis=1)
test = test.rename(columns={"FVC": "FVC_base", "Weeks": "Weeks_base"})
test.head()



## === cell 13
submission_grid = []
for i, patient in enumerate(test["Patient"].unique()):
    df = pd.DataFrame({"Patient": patient, "Weeks": np.arange(-12, 134)})
    df["PatientID"] = i
    submission_grid.append(df)

submission_grid = pd.concat(submission_grid, ignore_index=True)
submission_grid.head()



## === cell 14
FVC_b_t = test["FVC_base"].values
w_b_t = test["Weeks_base"].values
age_t = test["Age"].values
patient_class_t = test["Class"].values

t_all = submission_grid["Weeks"].values
patient_id_all = submission_grid["PatientID"].values

with pm.Model() as new_model:
    beta_int = pm.Normal(
        "beta_int", trace["beta_int"].mean(), sigma=trace["beta_int"].std() + 1e-6
    )
    sigma_int = pm.TruncatedNormal(
        "sigma_int",
        mu=trace["sigma_int"].mean(),
        sigma=trace["sigma_int"].std() + 1e-6,
        lower=0,
    )

    mu_alpha = FVC_b_t + beta_int * w_b_t
    alpha = pm.Normal(
        "alpha", mu=mu_alpha, sigma=sigma_int, shape=test["Patient"].nunique()
    )

    sigma_s = pm.TruncatedNormal(
        "sigma_s",
        mu=trace["sigma_s"].mean(),
        sigma=trace["sigma_s"].std() + 1e-6,
        lower=0,
    )
    alpha_s = pm.Normal(
        "alpha_s", trace["alpha_s"].mean(), sigma=trace["alpha_s"].std() + 1e-6
    )

    cov = np.zeros((6, 6))
    np.fill_diagonal(cov, trace["beta_cs"].var(axis=0) + 1e-6)
    beta_cs = pm.MvNormal("beta_cs", mu=trace["beta_cs"].mean(axis=0), cov=cov, shape=6)

    mu_beta = alpha_s + age_t * beta_cs[patient_class_t]
    beta = pm.Normal("beta", mu=mu_beta, sigma=sigma_s, shape=test["Patient"].nunique())

    sigma = pm.TruncatedNormal(
        "sigma",
        mu=trace["sigma"].mean(),
        sigma=trace["sigma"].std() + 1e-6,
        lower=0,
    )

    FVC_est = pm.Deterministic(
        "FVC_est", alpha[patient_id_all] + beta[patient_id_all] * t_all
    )



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1254796574.py in <cell line: 0>()
      7 patient_id_all = submission_grid["PatientID"].values
      8 
----> 9 with pm.Model() as new_model:
     10     beta_int = pm.Normal(
     11         "beta_int", trace["beta_int"].mean(), sigma=trace["beta_int"].std() + 1e-6

NameError: name 'pm' is not defined

## === cell 15
with new_model:
    trace2 = pm.sample(
        800,
        tune=800,
        target_accept=0.9,
        chains=2,
        cores=2,
        progressbar=True,
        random_seed=42,
    )



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1194640701.py in <cell line: 0>()
      1 # Same runtime guard as training model.
----> 2 with new_model:
      3     trace2 = pm.sample(
      4         800,
      5         tune=800,

NameError: name 'new_model' is not defined

## === cell 16
fvc_samples = trace2["FVC_est"]  # (n_draws, n_rows)
fvc_mean = fvc_samples.mean(axis=0)
fvc_std = fvc_samples.std(axis=0)

pred_df = submission_grid.copy()
pred_df["FVC"] = fvc_mean
pred_df["Confidence"] = fvc_std
pred_df["Patient_Week"] = pred_df["Patient"] + "_" + pred_df["Weeks"].astype(str)

final_sub = sample_sub[["Patient_Week"]].merge(
    pred_df[["Patient_Week", "FVC", "Confidence"]],
    on="Patient_Week",
    how="left",
)

if final_sub["FVC"].isna().any():
    base_map = test.set_index("Patient")["FVC_base"].to_dict()
    pw = final_sub.loc[final_sub["FVC"].isna(), "Patient_Week"]
    pats = pw.str.split("_").str[0]
    final_sub.loc[final_sub["FVC"].isna(), "FVC"] = pats.map(base_map).fillna(
        test["FVC_base"].median()
    )
    final_sub.loc[final_sub["Confidence"].isna(), "Confidence"] = 200.0

final_sub["Confidence"] = final_sub["Confidence"].clip(lower=70)

final_sub.to_csv("submission.csv", index=False)
final_sub.head()

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2474135078.py in <cell line: 0>()
      1 # trace2['FVC_est'] is shape (draws, n_rows). Create mean/std per row.
----> 2 fvc_samples = trace2["FVC_est"]  # (n_draws, n_rows)
      3 fvc_mean = fvc_samples.mean(axis=0)
      4 fvc_std = fvc_samples.std(axis=0)
      5 

NameError: name 'trace2' is not defined
