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
protobuf==6.33.0
pydicom==3.0.1
scikit-image==0.25.2
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

-6.8423

# 6. Current score

-7.9655

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -11.05123) has done: 'I fix the two blockers that prevent an end-to-end run and a valid submission: (1) TensorFlow import crash caused by an incompatible protobuf version by avoiding TF entirely (it was only used for a dense net, but we keep the modeling approach by switching to scikit-learn’s RandomForestRegressor which you already import), and (2) the OneHotEncoder crash by ensuring SmokingStatus is treated as string categories consistently (fit/transform on the same dtype), then ensuring the test feature columns match train. I also fix a couple of logic bugs in building the test dictionaries (using `.iloc[0]` instead of casting a Series to float/int). Finally, I guarantee the saved CSV matches `sample_submission.csv` columns exactly: `Patient_Week,FVC,Confidence`.'
- What this solution (achieved -8.0821) has done: 'Your biggest gap to the target comes from the model being asked to predict `confidence` from an all-zero target, which makes sigma uninformative and hurts the Laplace log-likelihood. I keep the same RandomForest setup and feature logic, but compute a per-row training confidence from each patient’s within-patient FVC variability (a legitimate proxy for uncertainty), then train the sigma forest on that instead of zeros. I also apply the scaler you already fit (you currently compute scaled frames but don’t use them for training/inference), which is a minimal consistency fix that usually improves generalization without changing the approach. Finally, I keep the required submission schema and clip sigma at 70 as per the metric.'
- What this solution (achieved -7.9655) has done: 'Your current gap to the target (-8.0821 vs -6.8423, higher is better) suggests we should improve the LaplaceLL mainly by calibrating `Confidence` (sigma), because overly-small/overly-large sigma is heavily penalized by the `-log(sigma)` term. I keep your RandomForest models and all feature engineering intact, but (1) generate a more metric-aligned training target for sigma using each patient’s residual spread after removing a simple per-patient linear week trend (still computed only from train), and (2) apply a single global multiplicative calibration to predicted sigma chosen to maximize the validation metric, then use that calibrated factor for test. These are minimal changes that preserve the approach while typically moving the score upward toward your target band.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_DIR = "../input/osic-pulmonary-fibrosis-progression"



## === cell 1
train_csv = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
train_csv.head()



## === cell 2
base_week = train_csv.groupby("Patient")["Weeks"].min()

base_week_list = []
for i in range(len(train_csv)):
    base_week_list.append(base_week[train_csv.iloc[i, 0]])
train_csv["base_week"] = base_week_list

count_from_base_week = []
for i in range(len(train_csv)):
    count_from_base_week.append(train_csv.iloc[i, 1] - base_week[train_csv.iloc[i, 0]])
train_csv["count_from_base_week"] = count_from_base_week


def _patient_resid_std(df_pat: pd.DataFrame) -> float:
    w = df_pat["Weeks"].values.astype(np.float64)
    y = df_pat["FVC"].values.astype(np.float64)
    if len(df_pat) < 2:
        return 70.0
    wc = w - w.mean()
    X = np.vstack([wc, np.ones_like(wc)]).T
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ coef
    resid = y - yhat
    s = float(np.std(resid, ddof=1)) if len(resid) > 1 else float(np.std(resid))
    if not np.isfinite(s):
        s = 70.0
    return max(s, 70.0)


patient_resid_std = train_csv.groupby("Patient", sort=False).apply(_patient_resid_std)
train_csv["confidence"] = train_csv["Patient"].map(patient_resid_std).astype(np.float32)
train_csv["confidence"] = np.maximum(train_csv["confidence"].values, 70.0)

base_fvc_dict = {}
for pid in train_csv["Patient"].unique():
    base_fvc_dict[pid] = np.array(
        train_csv[
            (train_csv["Patient"] == pid) & (train_csv["Weeks"] == base_week[pid])
        ]["FVC"]
    )[0]

base_fvc = []
for i in range(len(train_csv)):
    base_fvc.append(base_fvc_dict[train_csv.iloc[i, 0]])
train_csv["base_fvc"] = base_fvc

base_fev1_dict = {}
for pid in train_csv["Patient"].unique():
    A = train_csv[train_csv["Patient"] == pid]["base_fvc"].unique()[0]
    B = train_csv[train_csv["Patient"] == pid]["Age"].unique()[0]
    if train_csv[train_csv["Patient"] == pid]["Sex"].unique()[0] == "Male":
        base_fev1_dict[pid] = 0.77 * A + 0.32 + 0.0069 * B
    else:
        base_fev1_dict[pid] = 0.77 * A + 0.28 + 0.0052 * B

base_fev1 = []
for i in range(len(train_csv)):
    base_fev1.append(base_fev1_dict[train_csv.iloc[i, 0]])
train_csv["base_fev1"] = base_fev1

base_week_percent_dict = {}
for pid in train_csv["Patient"].unique():
    base_week_percent_dict[pid] = np.array(
        train_csv[
            (train_csv["Patient"] == pid) & (train_csv["Weeks"] == base_week[pid])
        ]["Percent"]
    )[0]

base_week_percent = []
for i in range(len(train_csv)):
    base_week_percent.append(base_week_percent_dict[train_csv.iloc[i, 0]])
train_csv["base_week_percent"] = base_week_percent

train_csv["base fev1/base fvc"] = train_csv["base_fev1"] / train_csv["base_fvc"]
train_csv["base_height"] = (train_csv["base_fvc"] + 9030) / 77.0

base_weight_dict = {}
for pid in train_csv["Patient"].unique():
    FVC = train_csv[train_csv["Patient"] == pid]["base_fvc"].unique()[0]
    A = train_csv[train_csv["Patient"] == pid]["Age"].unique()[0]
    H = train_csv[train_csv["Patient"] == pid]["base_height"].unique()[0]
    if train_csv[train_csv["Patient"] == pid]["Sex"].unique()[0] == "Male":
        base_weight_dict[pid] = (FVC + 5458 - 49 * H + 8 * A) / 12.0
    else:
        base_weight_dict[pid] = (FVC + 3863 - 37 * H + 6 * A) / 14.0

base_weight = []
for i in range(len(train_csv)):
    base_weight.append(base_weight_dict[train_csv.iloc[i, 0]])
train_csv["base_weight"] = base_weight

train_csv["base_bmi"] = train_csv["base_weight"] / (
    (train_csv["base_height"] / 100) ** 2
)

train_csv.head()



## === cell 3
lb = LabelEncoder()  # Sex
train_csv.iloc[:, 5] = lb.fit_transform(train_csv.iloc[:, 5])

lb2 = LabelEncoder()  # SmokingStatus
train_csv.iloc[:, 6] = lb2.fit_transform(train_csv.iloc[:, 6])

train_csv[["Sex", "SmokingStatus"]].head()



## === cell 4
oh1 = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

train_csv["SmokingStatus_str"] = train_csv["SmokingStatus"].astype(str)
smoke_cat = pd.DataFrame(
    oh1.fit_transform(train_csv[["SmokingStatus_str"]]),
    columns=["smoking cat 0", "smoking cat 1", "smoking cat 2"],
)
train_csv = pd.concat([train_csv, smoke_cat], axis=1)

train_csv[
    [
        "SmokingStatus",
        "SmokingStatus_str",
        "smoking cat 0",
        "smoking cat 1",
        "smoking cat 2",
    ]
].head()



## === cell 5
sc = StandardScaler()
num_cols = [
    "Weeks",
    "Age",
    "base_week",
    "count_from_base_week",
    "base_fvc",
    "base_fev1",
    "base_week_percent",
    "base fev1/base fvc",
    "base_height",
    "base_weight",
    "base_bmi",
]

train_scaled = pd.DataFrame(sc.fit_transform(train_csv[num_cols]), columns=num_cols)
train_scaled["Sex"] = train_csv["Sex"]
train_scaled["smoking cat 0"] = train_csv["smoking cat 0"]
train_scaled["smoking cat 1"] = train_csv["smoking cat 1"]
train_scaled["smoking cat 2"] = train_csv["smoking cat 2"]

train_scaled.head()



## === cell 6
sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
test_csv = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

sub.head(), test_csv.head()



## === cell 7
test_week = []
patient_id = []
for i in range(len(sub)):
    test_week.append(int(sub.iloc[i, 0].split("_")[-1]))
    patient_id.append(sub.iloc[i, 0].split("_")[0])

sub["Patient"] = patient_id
sub["Weeks"] = test_week
sub.drop(["FVC", "Confidence"], axis=1, inplace=True)

base_fvc = test_csv.groupby("Patient")["FVC"].min()
fvc = []
for i in range(len(sub)):
    fvc.append(base_fvc[sub.iloc[i, 1]])
sub["base_fvc"] = fvc

base_fev1_dict_test = {}
for pid in sub["Patient"].unique():
    A = sub[sub["Patient"] == pid]["base_fvc"].unique()[0]
    B = test_csv[test_csv["Patient"] == pid]["Age"].iloc[0]
    if test_csv[test_csv["Patient"] == pid]["Sex"].iloc[0] == "Male":
        base_fev1_dict_test[pid] = 0.77 * A + 0.32 + 0.0069 * B
    else:
        base_fev1_dict_test[pid] = 0.77 * A + 0.28 + 0.0052 * B

base_fev1_test = []
for i in range(len(sub)):
    base_fev1_test.append(base_fev1_dict_test[sub.iloc[i, 1]])
sub["base_fev1"] = base_fev1_test

sub["base_height"] = (sub["base_fvc"] + 9030) / 77.0

base_weight_dict_test = {}
for pid in sub["Patient"].unique():
    FVC = sub[sub["Patient"] == pid]["base_fvc"].unique()[0]
    A = test_csv[test_csv["Patient"] == pid]["Age"].iloc[0]
    H = sub[sub["Patient"] == pid]["base_height"].unique()[0]
    if test_csv[test_csv["Patient"] == pid]["Sex"].iloc[0] == "Male":
        base_weight_dict_test[pid] = (FVC + 5458 - 49 * H + 8 * A) / 12.0
    else:
        base_weight_dict_test[pid] = (FVC + 3863 - 37 * H + 6 * A) / 14.0

base_weight_test = []
for i in range(len(sub)):
    base_weight_test.append(base_weight_dict_test[sub.iloc[i, 1]])
sub["base_weight"] = base_weight_test

test_csv.iloc[:, 5] = lb.transform(test_csv.iloc[:, 5])
test_csv.iloc[:, 6] = lb2.transform(test_csv.iloc[:, 6])

percent_dict = {}
sex_dict = {}
age_dict = {}
ss_dict = {}
for pid in test_csv["Patient"].unique():
    percent_dict[pid] = float(
        test_csv.loc[test_csv["Patient"] == pid, "Percent"].iloc[0]
    )
    sex_dict[pid] = int(test_csv.loc[test_csv["Patient"] == pid, "Sex"].iloc[0])
    age_dict[pid] = int(test_csv.loc[test_csv["Patient"] == pid, "Age"].iloc[0])
    ss_dict[pid] = int(
        test_csv.loc[test_csv["Patient"] == pid, "SmokingStatus"].iloc[0]
    )

percent = []
sex = []
age = []
ss = []
for i in range(len(sub)):
    percent.append(percent_dict[sub.iloc[i, 1]])
    sex.append(sex_dict[sub.iloc[i, 1]])
    age.append(age_dict[sub.iloc[i, 1]])
    ss.append(ss_dict[sub.iloc[i, 1]])

sub["base_week_percent"] = percent
sub["Age"] = age
sub["Sex"] = sex
sub["SmokingStatus"] = ss

base_week_test = test_csv.groupby("Patient")["Weeks"].min()
count_from_base_week_test = []
base_week_list2 = []
for i in range(len(sub)):
    count_from_base_week_test.append(sub.iloc[i, 2] - base_week_test[sub.iloc[i, 1]])
    base_week_list2.append(base_week_test[sub.iloc[i, 1]])
sub["count_from_base_week"] = count_from_base_week_test
sub["base_week"] = base_week_list2

sub["base fev1/base fvc"] = sub["base_fev1"] / sub["base_fvc"]
sub["base_bmi"] = sub["base_weight"] / ((sub["base_height"] / 100.0) ** 2)

sub.head()



## === cell 8
sub["SmokingStatus_str"] = sub["SmokingStatus"].astype(str)

smoke_cat_test = pd.DataFrame(
    oh1.transform(sub[["SmokingStatus_str"]]),
    columns=["smoking cat 0", "smoking cat 1", "smoking cat 2"],
)
sub = pd.concat([sub, smoke_cat_test], axis=1)

sub[
    [
        "SmokingStatus",
        "SmokingStatus_str",
        "smoking cat 0",
        "smoking cat 1",
        "smoking cat 2",
    ]
].head()



## === cell 9
sub_scaled = pd.DataFrame(sc.transform(sub[num_cols]), columns=num_cols)
sub_scaled["Sex"] = sub["Sex"]
sub_scaled["smoking cat 0"] = sub["smoking cat 0"]
sub_scaled["smoking cat 1"] = sub["smoking cat 1"]
sub_scaled["smoking cat 2"] = sub["smoking cat 2"]

sub_scaled.head()



## === cell 10
feature_cols = [
    "Weeks",
    "Age",
    "Sex",
    "base_week",
    "count_from_base_week",
    "base_fvc",
    "base_fev1",
    "base_week_percent",
    "base fev1/base fvc",
    "base_height",
    "base_weight",
    "base_bmi",
    "smoking cat 0",
    "smoking cat 1",
]  # keep identical to original

x = np.array(train_scaled[feature_cols], dtype=np.float32)
y = np.array(train_csv[["FVC", "confidence"]], dtype=np.float32)

xtrain, xvalid, ytrain, yvalid = train_test_split(
    x, y, test_size=0.2, random_state=RANDOM_STATE
)

xtrain.shape, ytrain.shape



## === cell 11
rf_fvc = RandomForestRegressor(n_estimators=600, random_state=RANDOM_STATE, n_jobs=-1)
rf_sig = RandomForestRegressor(
    n_estimators=600, random_state=RANDOM_STATE + 1, n_jobs=-1
)

rf_fvc.fit(xtrain, ytrain[:, 0])
rf_sig.fit(xtrain, ytrain[:, 1])

val_fvc = rf_fvc.predict(xvalid)
val_sig = rf_sig.predict(xvalid)

val_fvc[:5], val_sig[:5]




## === cell 12
def metric(actual_fvc, predicted_fvc, confidence, return_values=False):
    sd_clipped = np.maximum(confidence, 70)
    delta = np.minimum(np.abs(actual_fvc - predicted_fvc), 1000)
    m = -np.sqrt(2) * delta / sd_clipped - np.log(np.sqrt(2) * sd_clipped)
    return m if return_values else np.mean(m)


print(
    "Validation metric (using clipped sigma):",
    metric(yvalid[:, 0], val_fvc, np.abs(val_sig)),
)

val_sig_abs = np.abs(val_sig).astype(np.float64)

factors = np.array(
    [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6, 1.8, 2.0, 2.3, 2.6, 3.0],
    dtype=np.float64,
)
best_factor = 1.0
best_score = -1e18
for f in factors:
    s = metric(yvalid[:, 0], val_fvc, np.maximum(val_sig_abs * f, 70.0))
    if s > best_score:
        best_score = s
        best_factor = float(f)

print("Best sigma factor on validation:", best_factor)
print(
    "Validation metric (after sigma factor):",
    metric(yvalid[:, 0], val_fvc, np.maximum(val_sig_abs * best_factor, 70.0)),
)



## === cell 13
xtest = np.array(sub_scaled[feature_cols], dtype=np.float32)

pred_fvc = rf_fvc.predict(xtest)
pred_sig = np.abs(rf_sig.predict(xtest)).astype(np.float64)

pred_sig = pred_sig * best_factor
pred_sig = np.maximum(pred_sig, 70.0)

submission = pd.DataFrame(
    {
        "Patient_Week": sub["Patient_Week"].values,
        "FVC": pred_fvc.astype(np.float32),
        "Confidence": pred_sig.astype(np.float32),
    }
)

submission.head()



## === cell 14
submission = submission[["Patient_Week", "FVC", "Confidence"]]
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.columns.tolist())
print(submission.head())
