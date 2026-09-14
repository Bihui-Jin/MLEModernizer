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
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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

-6.956703284743935

# 6. Current score

-8.33163

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -11.9005) has done: 'I fix the TensorFlow/protobuf import crash by pinning the pure-Python protobuf implementation via environment variables set before importing TensorFlow. Then I make the pretrained-weights dependency robust: if the referenced Kaggle dataset/paths aren’t present (your environment doesn’t have `../input/osic-linear-decay-and-quant-reg-base/...`), the code fall back to a simple, deterministic per-patient linear model trained from `train.csv` clinical history to generate FVC and a reasonable confidence. Finally, I ensure the submission is always produced with the exact required columns and `.csv` suffix, aligned to `sample_submission.csv`’s `Patient_Week` ordering.'
- What this solution (achieved -8.33163) has done: 'I fix the TensorFlow import crash caused by the protobuf 6.x API mismatch by removing the hard dependency on TensorFlow for this notebook (since the current run is already designed to fall back to the deterministic per-patient linear model when pretrained weights aren’t available). Then I make the feature-prep more robust for `sub`/`test` by ensuring `Percent` is present (it’s needed by `transform_features`) and by preventing `init_week` from becoming all-NaN due to missing `min_week`. Finally, I keep the existing fallback modeling logic intact but slightly improve confidence calibration (still clipped at 70) to move score toward the target while preserving the same modeling approach and producing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")



## === cell 1
import sys
import numpy as np
import pandas as pd

from IPython.display import display

pd.set_option("display.max_columns", 50)

print("Python:", sys.version.split()[0])
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)



## === cell 2
np.random.seed(42)



## === cell 3
input_path = "../input/osic-pulmonary-fibrosis-progression"
pretrained_path = "../input/osic-linear-decay-and-quant-reg-base/pretrained_weights"

print("input_path exists:", os.path.exists(input_path))
print("pretrained_path exists:", os.path.exists(pretrained_path))




## === cell 4
def height_proxy(fvc_e, age, sex):
    if sex == "Female":
        h = fvc_e / (21.78 - 0.101 * age)
    else:
        h = fvc_e / (27.63 - 0.112 * age)
    return h


def process_init_week(df, train_df=False):
    if train_df:
        df["min_week"] = df.groupby("Patient")["Weeks"].transform("min")
    else:
        if "min_week" not in df.columns or df["min_week"].isna().all():
            df["min_week"] = df.groupby("Patient")["Weeks"].transform("min")

    base = df.loc[df.Weeks == df.min_week][["Patient", "FVC", "Percent", "Age", "Sex"]]
    base["FVC_init_avg"] = base.groupby("Patient")["FVC"].transform("mean").astype(int)
    base["Percent_init"] = base.groupby("Patient")["Percent"].transform("mean")
    base = base[
        ["Patient", "FVC_init_avg", "Percent_init", "Age", "Sex"]
    ].drop_duplicates()
    base["FVC_expected"] = base["FVC_init_avg"] / (base["Percent_init"] / 100)
    base["Height_proxy"] = base.apply(
        lambda x: height_proxy(x.FVC_expected, x.Age, x.Sex), axis=1
    )
    base = base[["Patient", "Height_proxy", "FVC_init_avg", "Percent_init"]]

    df = df.merge(base, on="Patient", how="left")
    df["init_week"] = df["Weeks"] - df["min_week"]
    return df




## === cell 5
train = pd.read_csv(input_path + "/train.csv")
train = process_init_week(train, train_df=True)
train.drop_duplicates(keep="first", inplace=True, subset=["Patient", "Weeks"])
train.head(3)



## === cell 6
sub = pd.read_csv(input_path + "/sample_submission.csv")
test = pd.read_csv(input_path + "/test.csv")

sub["Patient"] = sub["Patient_Week"].apply(lambda x: x.split("_")[0])
sub["Weeks"] = sub["Patient_Week"].apply(lambda x: int(x.split("_")[-1]))
sub = sub[["Patient", "Weeks", "Confidence", "Patient_Week"]]

test_for_merge = test.rename(columns={"Weeks": "min_week"}).copy()
test_for_merge = test_for_merge[
    ["Patient", "min_week", "FVC", "Percent", "Age", "Sex", "SmokingStatus"]
]

sub = sub.merge(test_for_merge, on="Patient", how="left")

sub = process_init_week(sub, train_df=False)
sub.head(3)




## === cell 7
def scale_fn(var_name):
    col = train[var_name]
    denom = col.max() - col.min()
    if denom == 0 or pd.isna(denom):
        return lambda x: 0.0
    return lambda x: (x - col.min()) / denom


scale_age = scale_fn("Age")
scale_height = scale_fn("Height_proxy")
scale_percent = scale_fn("Percent")
scale_fvc = scale_fn("FVC_init_avg")

scale_week = lambda x: (x - (-12)) / (133 - (-12))


def transform_features(df):
    df = df.assign(sex_code=np.where(df["Sex"] == "Female", 1, 0))
    df = df.assign(ex_smoker=np.where(df["SmokingStatus"] == "Ex-smoker", 1, 0))
    df = df.assign(never_smoked=np.where(df["SmokingStatus"] == "Never smoked", 1, 0))
    df = df.assign(
        current_smoker=np.where(df["SmokingStatus"] == "Currently smokes", 1, 0)
    )
    df["has_smoked"] = df["ex_smoker"] + df["current_smoker"]

    df["age"] = df["Age"].map(scale_age)
    df["height"] = df["Height_proxy"].map(scale_height)
    df["percent"] = df["Percent"].map(scale_percent)
    df["percent_init"] = df["Percent_init"].map(scale_percent)
    df["week"] = df["Weeks"].map(scale_week)
    df["fvc_init"] = df["FVC_init_avg"].map(scale_fvc)
    return df




## === cell 8
train = transform_features(train)
train.reset_index(inplace=True, drop=True)
train.head(3)



## === cell 9
sub = transform_features(sub)
sub.head(3)




## === cell 10
def _exists_all(paths):
    return all(os.path.exists(p) for p in paths)




## === cell 11
linear_decay_features = [
    "age",
    "sex_code",
    "has_smoked",
    "current_smoker",
    "height",
    "percent_init",
    "fvc_init",
]


def get_patient_tab(df):  # df is either train or sub
    patients_init = df[df["init_week"] == 0].copy()
    patients_init = patients_init[["Patient"] + linear_decay_features]
    patients_init.set_index("Patient", inplace=True)
    return patients_init


patients_tab_train = get_patient_tab(train)
patients_tab_test = get_patient_tab(sub)
print(patients_tab_train.shape)
display(patients_tab_train.head(3))
patients_tab_test.head(3)



## === cell 12
required_sub_cols = ["Patient", "Weeks", "Patient_Week", "FVC_init_avg", "init_week"]
missing = [c for c in required_sub_cols if c not in sub.columns]
if missing:
    raise RuntimeError(f"Missing columns in sub after processing: {missing}")



## === cell 13
PREDICTIONS = sub[["Patient", "Weeks", "Patient_Week"]].copy()
PREDICTIONS.head(5)




## === cell 14
def fit_patient_linear_models(train_df):
    g = train_df[["Patient", "Weeks", "FVC"]].dropna().groupby("Patient", sort=False)

    rows = []
    for pid, d in g:
        x = d["Weeks"].to_numpy(dtype=np.float64)
        y = d["FVC"].to_numpy(dtype=np.float64)

        if len(d) >= 2 and np.var(x) > 1e-12:
            b, a = np.polyfit(x, y, 1)  # y = b*x + a
            yhat = a + b * x
            resid = y - yhat
            sigma = float(np.sqrt(np.mean(resid**2))) if len(resid) else 200.0
        else:
            a = float(np.mean(y)) if len(y) else 2500.0
            b = 0.0
            sigma = 250.0

        rows.append((pid, a, b, sigma, len(d)))

    out = pd.DataFrame(rows, columns=["Patient", "a", "b", "sigma", "n_obs"])
    return out


patient_models = fit_patient_linear_models(train)

global_a = float(train["FVC"].mean())
global_b = 0.0
global_sigma = float(train.groupby("Patient")["FVC"].std().median())
if not np.isfinite(global_sigma):
    global_sigma = 250.0

print("Fitted patient models:", patient_models.shape)
patient_models.head(3)



## === cell 15
ld_inf_path = os.path.join(pretrained_path, "inference_linear_decay_2020Sep19.csv")
qr_inf_path = os.path.join(pretrained_path, "inference_quant_reg_2020Sep23.csv")

use_pretrained = os.path.exists(pretrained_path) and _exists_all(
    [ld_inf_path, qr_inf_path]
)
print("use_pretrained:", use_pretrained)

use_pretrained = False
LD_inference = None
QR_inference = None



## === cell 16
base = sub[["Patient", "Weeks", "Patient_Week", "init_week"]].merge(
    patient_models[["Patient", "a", "b", "sigma", "n_obs"]], on="Patient", how="left"
)
base["a"] = base["a"].fillna(global_a)
base["b"] = base["b"].fillna(global_b)
base["sigma"] = base["sigma"].fillna(global_sigma)
base["n_obs"] = base["n_obs"].fillna(1)

base["FVC"] = base["a"] + base["b"] * base["Weeks"]

week_dist = base["init_week"].abs().astype(float)
sigma0 = base["sigma"].astype(float)

obs_factor = 1.0 / np.sqrt(base["n_obs"].astype(float).clip(lower=1.0))
base["Confidence"] = np.maximum(sigma0 + (12.0 * week_dist * obs_factor), 70.0)

to_submit = base[["Patient_Week", "FVC", "Confidence"]].copy()
to_submit["FVC"] = np.round(to_submit["FVC"]).astype(int)
to_submit["Confidence"] = to_submit["Confidence"].astype(float)

to_submit = to_submit.merge(
    pd.read_csv(input_path + "/sample_submission.csv")[["Patient_Week"]],
    on="Patient_Week",
    how="right",
)
to_submit = to_submit[["Patient_Week", "FVC", "Confidence"]]
to_submit.head()



## === cell 17
to_submit.describe(include="all").T



## === cell 18
to_submit.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", to_submit.shape)



## === cell 19
print(pd.read_csv("submission.csv").head(3))
