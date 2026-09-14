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
lightgbm==4.6.0
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pydicom==3.0.1
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

-7.8671

# 6. Current score

-9.28627

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -13.09808) has done: 'I make the pipeline produce a valid submission end-to-end by fixing the cell numbering, correcting a shape bug in how `Y_train`/`Y_test` are appended (it currently becomes 1D and can break LightGBM), and ensuring the submission loop indexes `testY` consistently with the way `X_test` is constructed. To move the score up (closer to the target) without changing the core modeling approach, I add a minimal, metric-aligned post-processing step: predict `Confidence` from the model’s per-patient training residuals (clipped to Kaggle’s minimum 70), instead of using a constant. I also clip predicted `FVC` to a reasonable physiological range to avoid extreme values that hurt the Laplace log-likelihood, while keeping the same LightGBM training logic and features.'
- What this solution (achieved -13.38149) has done: 'We keep your LightGBM training and the week-wise feature construction exactly as-is, but fix the main score limiter: your `Confidence` is currently a constant global value, which is suboptimal for the Laplace log-likelihood. We compute a per-test-patient confidence by matching each test patient to the most similar train patient using baseline clinical features (Age/Percent/Sex/SmokingStatus), then reusing that matched train patient’s residual-based sigma (already computed) for the confidence output. This is a minimal post-processing change (no change to model, loss, or features) that usually improves the metric by better calibrating σ. We also ensure the week indexing uses the actual `Week` grid (`Week[0] == -12`) so it’s robust and avoids accidental off-by-12 issues.'
- What this solution (achieved -9.44722) has done: 'We keep your LightGBM training and week-grid feature construction unchanged, but fix two score-limiters in the post-processing that affect the Laplace log-likelihood directly. First, we produce a per-(patient, week) Confidence that grows as you extrapolate farther from the observed baseline week (instead of a single constant per patient), while still being anchored to your residual-derived per-patient sigma and clipped at 70 as required. Second, we adjust predicted FVC with a tiny per-test-patient bias correction using the known baseline FVC at week 0 (the only label-like value available in test), aligning predictions to the patient’s provided baseline without changing the model. These are minimal, metric-aligned post-processing steps and should move your score upward toward the target without altering core model logic.'
- What this solution (achieved -9.28627) has done: 'To move your score upward toward the target (higher is better) with minimal disruption, I keep the same LightGBM training and the same week-grid feature construction, but make two metric-aligned post-processing fixes. First, instead of using a nearest-neighbor patient’s sigma for test Confidence (which can be noisy/mismatched), we compute a per-test-patient sigma directly from your model’s *own* train residual sigma via a lightweight regression on baseline clinical features (same information you already use), then clip to Kaggle’s minimum 70. Second, we change the baseline FVC bias correction to anchor at each test patient’s provided baseline week (not hard-coded week 0), which is a small but meaningful alignment improvement when a test baseline week isn’t 0. These changes are only in the prediction/Confidence calibration layer and should improve the Laplace log-likelihood without changing your model/feature/training core logic.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import re
import cv2



## === cell 1
train_df = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
train_df



## === cell 2
import pydicom


def plot_pixel_array(dataset, figsize=(5, 5)):
    plt.figure(figsize=figsize)
    plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)
    plt.show()


file_path = (
    "../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/1.dcm"
)
dataset = pydicom.dcmread(file_path)
plot_pixel_array(dataset)




## === cell 3
def extract_num(s, p, ret=0):
    search = p.search(s)
    if search:
        return int(search.groups()[0])
    else:
        return ret




## === cell 4
filepath = []
ID = "ID00007637202177411956430"

for file in glob.glob(
    "../input/osic-pulmonary-fibrosis-progression/train/" + ID + "/*.dcm"
):
    filepath.append(file)

p = re.compile(ID + "/" + "(\d+)")
filepath = sorted(
    filepath, key=lambda s: extract_num(s, p, float("inf"))
)  # 画像を数字順にsort



## === cell 5
fig = plt.figure(figsize=(16, 7))

for i in range(18):
    plt.subplot(3, 6, i + 1)
    file_path = filepath[i]
    dataset = pydicom.dcmread(file_path)
    plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)
    plt.title(file_path[77:])
    plt.tick_params(
        labelbottom=False, labelleft=False, labelright=False, labeltop=False
    )



## === cell 6
train_df.loc[train_df.Patient == ID]



## === cell 7
Patient_list = list(train_df.Patient.unique())
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

a = 0
b = 0
c = 0

for ID in Patient_list:
    grp = train_df.loc[train_df.Patient == ID]
    grp = grp[["Weeks", "FVC", "SmokingStatus"]]

    if grp.iloc[0, 2] == "Currently smokes" and a <= 10:
        ax1.plot(grp.Weeks, grp.FVC, marker="o", color="red")
        ax1.set_title("Currently smokes")
        a = a + 1
    elif grp.iloc[0, 2] == "Ex-smoker" and b <= 10:
        ax2.plot(grp.Weeks, grp.FVC, marker="x", color="green")
        ax2.set_title("Ex-smoker")
        b = b + 1
    elif grp.iloc[0, 2] == "Never smoked" and c <= 10:
        ax3.plot(grp.Weeks, grp.FVC, marker="s", color="blue")
        ax3.set_title("Never smoked")
        c = c + 1
    else:
        pass



## === cell 8
Week = np.arange(-12, 134)
train_df2 = pd.DataFrame(Week, columns=["Weeks"])
train_df2.insert(1, "FVC", np.nan)
train_df2.insert(2, "Percent", np.nan)
train_df2.insert(3, "Age", np.nan)
train_df2.insert(4, "Sex", np.nan)
train_df2.insert(5, "SmokingStatus", np.nan)

train_id = train_df.loc[train_df.Patient == Patient_list[1]]
train_id = train_id.reset_index()

for i, D in enumerate(train_id.Weeks):
    D = D + 12
    train_df2.at[D, "FVC"] = train_id.FVC[i]
    train_df2.at[D, "Percent"] = train_id.Percent[i]

train_df2.loc[:, "Age"] = train_id.Age[0]
train_df2.loc[:, "Sex"] = train_id.Sex[0]
train_df2.loc[:, "SmokingStatus"] = train_id.SmokingStatus[0]

train_df2 = train_df2.interpolate("linear", order=2, limit_direction="both")
train_df2



## === cell 9
plt.figure(figsize=(18, 6))
grp = train_df2

plt.xlabel("Weeks")
plt.ylabel("FVC")
plt.plot(grp.Weeks, grp.FVC, marker="x")
plt.plot(train_id.Weeks, train_id.FVC, marker="o", markersize=8)



## === cell 10
plt.figure(figsize=(18, 6))
grp = train_df2

plt.xlabel("Weeks")
plt.ylabel("Percent")
plt.plot(grp.Weeks, grp.Percent, marker="^")
plt.plot(train_id.Weeks, train_id.Percent, marker="o", markersize=8)



## === cell 11
Week = np.arange(-12, 134)


def train_layer(ID_N):
    train_df2 = pd.DataFrame(Week, columns=["Weeks"])
    train_df_Y = pd.DataFrame(Week, columns=["Weeks"])
    train_df_Y.insert(1, "FVC", np.nan)
    train_df2.insert(1, "Percent", np.nan)
    train_df2.insert(2, "Age", np.nan)
    train_df2.insert(3, "Sex_Male", 0)
    train_df2.insert(4, "Sex_Female", 0)
    train_df2.insert(5, "Currently smokes", 0)
    train_df2.insert(6, "Ex-smoker", 0)
    train_df2.insert(7, "Never smoked", 0)

    train_id = train_df.loc[train_df.Patient == Patient_list[ID_N]]
    train_id = train_id.reset_index()

    for i, D in enumerate(train_id.Weeks):
        D = D + 12
        if D <= 133:
            train_df_Y.at[D, "FVC"] = train_id.FVC[i]
            train_df2.at[D, "Percent"] = train_id.Percent[i]

    train_df2.loc[:, "Age"] = train_id.Age[0]

    if train_id.Sex[0] == "Male":
        train_df2.loc[:, "Sex_Male"] = 1
    else:
        train_df2.loc[:, "Sex_Female"] = 1

    if train_id.SmokingStatus[0] == "Currently smokes":
        train_df2.loc[:, "Currently smokes"] = 1
    elif train_id.SmokingStatus[0] == "Ex-smoker":
        train_df2.loc[:, "Ex-smoker"] = 1
    else:
        train_df2.loc[:, "Never smoked"] = 1

    train_df2 = train_df2.interpolate("linear", order=2, limit_direction="both")
    train_df_Y = train_df_Y.interpolate("linear", order=2, limit_direction="both")
    train_df_Y = train_df_Y.astype("int")
    train_df_Y = train_df_Y.drop(["Weeks"], axis=1)

    return train_df2, train_df_Y




## === cell 12
train_layer(0)[0]



## === cell 13
train_layer(0)[1]



## === cell 14
X_train = train_layer(0)[0].to_numpy()
Y_train = train_layer(0)[1].to_numpy()

for i in range(1, len(Patient_list)):
    a = train_layer(i)[0].to_numpy()
    X_train = np.append(X_train, a, axis=0)

    b = train_layer(i)[1].to_numpy()
    Y_train = np.append(Y_train, b, axis=0)



## === cell 15
X_train.shape



## === cell 16
Y_train.shape



## === cell 17
test_df = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
test_df



## === cell 18
Week = np.arange(-12, 134)
Patient_list_test = list(test_df.Patient.unique())


def test_layer(ID_N):
    test_df2 = pd.DataFrame(Week, columns=["Weeks"])
    test_df_Y = pd.DataFrame(Week, columns=["Weeks"])
    test_df_Y.insert(1, "FVC", np.nan)
    test_df2.insert(1, "Percent", np.nan)
    test_df2.insert(2, "Age", np.nan)
    test_df2.insert(3, "Sex_Male", 0)
    test_df2.insert(4, "Sex_Female", 0)
    test_df2.insert(5, "Currently smokes", 0)
    test_df2.insert(6, "Ex-smoker", 0)
    test_df2.insert(7, "Never smoked", 0)

    test_id = test_df.loc[test_df.Patient == Patient_list_test[ID_N]]
    test_id = test_id.reset_index()

    for i, D in enumerate(test_id.Weeks):
        D = D + 12
        if D <= 133:
            test_df_Y.at[D, "FVC"] = test_id.FVC[i]
            test_df2.at[D, "Percent"] = test_id.Percent[i]

    test_df2.loc[:, "Age"] = test_id.Age[0]

    if test_id.Sex[0] == "Male":
        test_df2.loc[:, "Sex_Male"] = 1
    else:
        test_df2.loc[:, "Sex_Female"] = 1

    if test_id.SmokingStatus[0] == "Currently smokes":
        test_df2.loc[:, "Currently smokes"] = 1
    elif test_id.SmokingStatus[0] == "Ex-smoker":
        test_df2.loc[:, "Ex-smoker"] = 1
    else:
        test_df2.loc[:, "Never smoked"] = 1

    test_df2 = test_df2.interpolate("linear", order=2, limit_direction="both")
    test_df_Y = test_df_Y.interpolate("linear", order=2, limit_direction="both")
    test_df_Y = test_df_Y.astype("int")
    test_df_Y = test_df_Y.drop(["Weeks"], axis=1)

    return test_df2, test_df_Y




## === cell 19
test_layer(0)[0]



## === cell 20
X_test = test_layer(0)[0].to_numpy()
Y_test = test_layer(0)[1].to_numpy()  # Do not use Y_test for training

for i in range(1, len(Patient_list_test)):
    a = test_layer(i)[0].to_numpy()
    X_test = np.append(X_test, a, axis=0)

    b = test_layer(i)[1].to_numpy()
    Y_test = np.append(Y_test, b, axis=0)



## === cell 21
X_test.shape



## === cell 22
Y_test.shape



## === cell 23
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn import tree
import lightgbm as lgb



## === cell 24
trainY = []
testY = []

params = {
    "metric": "rmse",
    "num_leaves": 80,
    "seed": 42,
    "feature_fraction_seed": 42,
    "bagging_seed": 42,
}
lgb_train = lgb.Dataset(X_train, Y_train)

model = lgb.train(params, lgb_train)
trainY.append(model.predict(X_train))
testY.append(model.predict(X_test))



## === cell 25
plt.figure(figsize=(18, 6))

Y_train_Graph = pd.DataFrame(trainY[-1])
plt.plot(Y_train)
plt.plot(Y_train_Graph, label="Predict")
plt.legend()



## === cell 26
"""
plt.figure(figsize=(18,8))

ID_NUM = 0

for i in range(5):
    Y_test_Graph = pd.DataFrame(testY[i])
    plt.plot(Y_test, linestyle = "dashed")
    plt.plot(Y_test_Graph, label = "Predict:{0}".format(i))
    plt.xlim(145*ID_NUM, 145*(ID_NUM+1))
    
plt.legend()
"""



## === cell 27
pred_train = np.asarray(trainY[-1]).reshape(-1)
true_train = np.asarray(Y_train).reshape(-1)

train_resid = np.abs(true_train - pred_train)

n_weeks = len(Week)  # 146
n_train_patients = len(Patient_list)

resid_by_patient = train_resid.reshape(n_train_patients, n_weeks)
sigma_by_patient = resid_by_patient.mean(axis=1) * np.sqrt(2.0)  # Laplace scale proxy
sigma_by_patient = np.clip(sigma_by_patient, 70.0, 1000.0)  # keep in sensible range

global_sigma = float(np.clip(np.mean(sigma_by_patient), 70.0, 1000.0))



## === cell 28
sample_sub = pd.read_csv(
    "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)
pred_test = np.asarray(testY[-1]).reshape(-1)

pred_test = np.clip(pred_test, 500.0, 6000.0)

patient_to_idx = {p: i for i, p in enumerate(Patient_list_test)}


def _baseline_row(df_one_patient: pd.DataFrame) -> pd.Series:
    dfp = df_one_patient.sort_values("Weeks")
    w0 = dfp[dfp["Weeks"] == 0]
    if len(w0) > 0:
        return w0.iloc[0]
    return dfp.iloc[0]


train_base = (
    train_df.groupby("Patient", sort=False).apply(_baseline_row).reset_index(drop=True)
)
test_base = (
    test_df.groupby("Patient", sort=False).apply(_baseline_row).reset_index(drop=True)
)

train_base = train_base.set_index("Patient").loc[Patient_list].reset_index()
test_base = test_base.set_index("Patient").loc[Patient_list_test].reset_index()


def _encode_base(df_base: pd.DataFrame) -> np.ndarray:
    sex_male = (df_base["Sex"] == "Male").astype(np.float32).to_numpy()
    smoke_cur = (
        (df_base["SmokingStatus"] == "Currently smokes").astype(np.float32).to_numpy()
    )
    smoke_ex = (df_base["SmokingStatus"] == "Ex-smoker").astype(np.float32).to_numpy()
    smoke_nev = (
        (df_base["SmokingStatus"] == "Never smoked").astype(np.float32).to_numpy()
    )
    age = df_base["Age"].astype(np.float32).to_numpy()
    pct = df_base["Percent"].astype(np.float32).to_numpy()
    return np.stack(
        [age / 100.0, pct / 100.0, sex_male, smoke_cur, smoke_ex, smoke_nev], axis=1
    )


train_vec = _encode_base(train_base)
test_vec = _encode_base(test_base)

ridge = sklearn.linear_model.Ridge(alpha=1.0, random_state=42)
ridge.fit(train_vec, np.log1p(sigma_by_patient.astype(np.float32)))
sigma_test_by_patient = np.expm1(ridge.predict(test_vec)).astype(np.float32)
sigma_test_by_patient = np.clip(sigma_test_by_patient, 70.0, 1000.0)

week0 = int(Week[0])  # Week starts at -12
baseline_fvc_test = test_base.set_index("Patient")["FVC"].astype(np.float32).to_dict()
baseline_week_test = test_base.set_index("Patient")["Weeks"].astype(int).to_dict()

bias_by_patient = np.zeros(len(Patient_list_test), dtype=np.float32)
for i, p in enumerate(Patient_list_test):
    bw = int(baseline_week_test.get(p, 0))
    j_bw = bw - week0  # index for that baseline week within the 146-grid
    j_bw = int(np.clip(j_bw, 0, n_weeks - 1))
    pred_bw = pred_test[i * n_weeks + j_bw]
    true_bw = float(baseline_fvc_test.get(p, pred_bw))
    bias_by_patient[i] = true_bw - pred_bw

base_week_test = baseline_week_test

fvc_out = np.zeros(len(sample_sub), dtype=np.float32)
conf_out = np.zeros(len(sample_sub), dtype=np.float32)

alpha = 0.004  # per-week relative increase

for k, pw in enumerate(sample_sub["Patient_Week"].values):
    patient, week_str = pw.split("_")
    week = int(week_str)
    j = week - week0  # maps Week[0]..Week[-1] to 0..n_weeks-1

    if j < 0 or j >= n_weeks or patient not in patient_to_idx:
        fvc_out[k] = float(np.median(true_train))
        conf_out[k] = global_sigma
        continue

    i = patient_to_idx[patient]

    fvc_out[k] = pred_test[i * n_weeks + j] + bias_by_patient[i]

    bw = int(base_week_test.get(patient, 0))
    dist = abs(week - bw)
    conf = float(sigma_test_by_patient[i]) * (1.0 + alpha * float(dist))
    conf_out[k] = float(np.clip(conf, 70.0, 1000.0))

fvc_out = np.clip(fvc_out, 500.0, 6000.0)

submission = pd.DataFrame(
    {"Patient_Week": sample_sub["Patient_Week"], "FVC": fvc_out, "Confidence": conf_out}
)



## === cell 29
submission.to_csv("submission.csv", index=False)



## === cell 30
submission.head(40)
