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

-6.9798

# 6. Current score

-9.29009

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -9.32017) has done: 'I fix the TensorFlow import crash caused by an incompatible `protobuf` version by forcing the pure-Python protobuf implementation before importing TensorFlow. Then I fix the `Invalid dtype: object` training error by ensuring the model inputs/targets are strictly numeric `float32` (some columns become `object` after concatenations/encodings). Finally, I ensure `Patient_Week` is preserved through feature engineering and that the submission is written as a valid `submission.csv` with the exact required columns and row alignment.'
- What this solution (achieved -9.29009) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by pinning protobuf to the pure-Python implementation early and also forcing the compatible TF/protobuf behavior via environment flags before importing TensorFlow. Then I make the prediction Confidence numerically safe and closer to the competition’s evaluation clipping by enforcing a minimum of 70 at submission-time (this is metric-aligned and typically improves score vs. letting the model output tiny sigmas). Finally, I add a small determinism setup to make training/inference reproducible without changing the model architecture or training loop semantics, and ensure `submission.csv` is written with the required columns and row alignment.'
- What this solution (achieved -9.29009) has done: 'I fix the TensorFlow import crash by setting the protobuf-related environment flags before any TensorFlow-related imports and by ensuring we don’t accidentally trigger TF imports earlier in the notebook. Then I make the in-notebook validation metric consistent with the competition’s clipping rules (clip sigma at 70 and also cap delta at 1000) so the “best epoch” callback selects weights that better match the Kaggle evaluation, which should improve score toward your target without changing the model architecture or training loop. Finally, I preserve the exact submission row alignment by explicitly carrying `Patient_Week` through `sub` and ensuring numeric dtypes are enforced for both predictions and Confidence before writing `submission.csv`.'
- What this solution (achieved -9.29009) has done: 'I fix the TensorFlow import crash by switching protobuf to the pure-Python implementation (the current `"cpp"` setting triggers the missing `_message` error in this environment), and I ensure these environment flags are set before any TensorFlow import occurs. Then I keep the model/training logic unchanged but make sure the script can proceed past the failed import so the callbacks, training, and prediction cells run. Finally, I keep the submission creation logic intact while ensuring the output is written as a valid `submission.csv` with the required columns and aligned to `sample_submission.csv`.'
- What this solution (achieved -9.29009) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing TensorFlow to use the pure-Python protobuf runtime and by pinning a protobuf version known to be compatible with TF 2.18 at runtime before importing TensorFlow. This unblocks the training and inference cells without changing your model architecture, loss, or training loop. I also ensure `confidence` in the training target is numeric float32 (currently it’s all zeros, which can make training unstable) by setting it to the metric-aligned floor (70) while keeping the same two-output target structure; this typically nudges the score upward toward your target without altering the core approach. Finally, I keep submission alignment with `sample_submission.csv` and guarantee a valid `submission.csv` with required columns and safe numeric types.'
- What this solution (achieved -9.29009) has done: 'I fix the TensorFlow import crash by removing the runtime protobuf downgrade/reload logic (which is what triggers the `MessageFactory.GetPrototype` mismatch) and instead forcing TensorFlow to use the pure-Python protobuf implementation before importing TensorFlow. This keeps your model, loss, training loop, and feature pipeline identical, but makes the notebook run end-to-end reliably in this Kaggle environment. I also make the metric callback explicitly write into `logs` so `best_weights` consistently sees `val_metric` every epoch (score-neutral but stabilizes selection). Finally, I keep the submission creation exactly aligned to `sample_submission.csv` and ensure numeric dtypes and Confidence floor at 70 remain enforced.'
- What this solution (achieved -9.29009) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by enforcing the pure-Python protobuf runtime and preventing any accidental C++ protobuf usage before TensorFlow is imported; this is the root cause of the current runtime failure. I also add a safe fallback to import TensorFlow after importing `google.protobuf` to ensure the runtime selection has taken effect, without changing any model/training logic. Finally, I keep the existing metric-aligned Confidence floor at 70 and preserve submission alignment with `sample_submission.csv`, ensuring a valid `submission.csv` is always written.'
- What this solution (achieved -9.29009) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf runtime *and* applying a small compatibility monkey-patch to `google.protobuf.message_factory.MessageFactory` before importing TensorFlow; this addresses the exact missing method that TF expects in some protobuf variants. I keep your model, loss, feature pipeline, and training loop unchanged, only touching the import/runtime compatibility layer so the notebook runs end-to-end. Since your current score is below the target, I keep the metric-aligned Confidence floor at 70 as-is (already beneficial) and otherwise avoid score-changing edits beyond making the code execute reliably. The script still write a valid `submission.csv` with the required columns aligned to `sample_submission.csv`.'
- What this solution (achieved -9.29009) has done: 'I fix the TensorFlow import crash by applying the protobuf `MessageFactory.GetPrototype` compatibility patch correctly (patching the *instance* used by TF, not just the class) before importing TensorFlow. This unblocks the rest of the notebook without touching your model architecture, loss, feature engineering, or training loop semantics. I also keep the metric-aligned confidence floor at 70 and ensure the pipeline still writes a valid `submission.csv` with the exact required columns and row alignment. No score-degrading changes are introduced; this is primarily an execution fix.'
- What this solution (achieved -9.29009) has done: 'I fix the TensorFlow import crash by applying a more robust protobuf compatibility patch that guarantees `MessageFactory.GetPrototype` exists on both the class and the default instance that TensorFlow touches in this environment. This is an execution-only change that keeps your model, loss, training loop, and feature pipeline unchanged. After that, the notebook run end-to-end again and produce a valid `submission.csv` with the required columns and alignment. No additional score-tuning changes are introduced beyond restoring correct execution.'
- What this solution (achieved -9.29009) has done: 'I fix the TensorFlow import crash by replacing the brittle protobuf `MessageFactory.GetPrototype` monkey-patch with a safe, minimal compatibility shim that defines `GetPrototype` on `google.protobuf.message_factory.MessageFactory` when missing (this is the root cause of the current runtime error). I keep your model, loss, feature engineering, and training loop unchanged, so score behavior stays comparable while execution becomes reliable. I also keep the metric-aligned confidence floor at 70 for both training targets and submission post-processing (already beneficial for this metric). Finally, I ensure the pipeline always reaches the CSV write and produces `submission.csv` with the exact required columns and alignment to `sample_submission.csv`.'
- What this solution (achieved -9.29009) has done: 'I fix the TensorFlow import crash by making the protobuf `GetPrototype` compatibility shim apply both to the `MessageFactory` class and the default *instance* TensorFlow actually uses (the current patch only touches the class, so TF still hits an unpatched instance). This is execution-critical and should be score-neutral, restoring the exact same training/inference pipeline you already had. I also keep the metric-aligned confidence flooring at 70 (already beneficial for this competition metric) and ensure the script always reaches the `submission.csv` write with correct columns and alignment. No model architecture, loss, or feature logic is changed.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

np.random.seed(42)



## === cell 1
DATA_DIR = "../input/osic-pulmonary-fibrosis-progression"
train_csv = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))



## === cell 2
base_week = train_csv.groupby("Patient")["Weeks"].min()

base_week_list = []
for i in range(len(train_csv)):
    base_week_list.append(base_week.loc[train_csv.iloc[i, 0]])
train_csv["base_week"] = base_week_list

count_from_base_week = []
for i in range(len(train_csv)):
    count_from_base_week.append(
        train_csv.iloc[i, 1] - base_week.loc[train_csv.iloc[i, 0]]
    )
train_csv["count_from_base_week"] = count_from_base_week

train_csv["confidence"] = np.full(train_csv.shape[0], 70.0, dtype=np.float32)

base_fvc_dict = {}
for pid in train_csv["Patient"].unique():
    base_fvc_dict[pid] = np.array(
        train_csv[
            (train_csv["Patient"] == pid) & (train_csv["Weeks"] == base_week.loc[pid])
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
            (train_csv["Patient"] == pid) & (train_csv["Weeks"] == base_week.loc[pid])
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



## === cell 3
lb = LabelEncoder()  # Sex
train_csv.iloc[:, 5] = lb.fit_transform(train_csv.iloc[:, 5])

lb2 = LabelEncoder()  # SmokingStatus (kept, but do NOT use for one-hot later)
train_csv.iloc[:, 6] = lb2.fit_transform(train_csv.iloc[:, 6])



## === cell 4
train_csv["SmokingStatus_str"] = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))[
    "SmokingStatus"
].astype(str)

oh1 = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
smoke_cat = pd.DataFrame(
    oh1.fit_transform(train_csv[["SmokingStatus_str"]]),
    columns=["smoking cat 0", "smoking cat 1", "smoking cat 2"],
)
train_csv = pd.concat([train_csv, smoke_cat], axis=1)



## === cell 5
sc = StandardScaler()
train_scaled = pd.DataFrame(
    sc.fit_transform(
        train_csv[
            [
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
        ]
    ),
    columns=[
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
    ],
)

train_scaled["Sex"] = train_csv["Sex"]
train_scaled["smoking cat 0"] = train_csv["smoking cat 0"]
train_scaled["smoking cat 1"] = train_csv["smoking cat 1"]

train_scaled = train_scaled.apply(pd.to_numeric, errors="coerce").fillna(0.0)



## === cell 6
sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
test_csv = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))



## === cell 7
test_week = []
patient_id = []
for i in range(len(sub)):
    test_week.append(int(sub.iloc[i, 0].split("_")[-1]))
    patient_id.append(sub.iloc[i, 0].split("_")[0])

sub["Patient_Week"] = sub["Patient_Week"].astype(str)

sub["Patient"] = patient_id
sub["Weeks"] = test_week
sub.drop(["FVC", "Confidence"], axis=1, inplace=True)

base_fvc = test_csv.groupby("Patient")["FVC"].min()
fvc = []
for i in range(len(sub)):
    fvc.append(base_fvc.loc[sub.iloc[i, 1]])
sub["base_fvc"] = fvc

base_fev1_dict_test = {}
for pid in sub["Patient"].unique():
    A = sub[sub["Patient"] == pid]["base_fvc"].unique()[0]
    B = test_csv[test_csv["Patient"] == pid]["Age"].unique()[0]
    if test_csv[test_csv["Patient"] == pid]["Sex"].unique()[0] == "Male":
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
    A = test_csv[test_csv["Patient"] == pid]["Age"].unique()[0]
    H = sub[sub["Patient"] == pid]["base_height"].unique()[0]
    if test_csv[test_csv["Patient"] == pid]["Sex"].unique()[0] == "Male":
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
    row = test_csv[test_csv["Patient"] == pid].iloc[0]
    percent_dict[pid] = float(row["Percent"])
    sex_dict[pid] = int(row["Sex"])
    age_dict[pid] = int(row["Age"])
    ss_dict[pid] = int(row["SmokingStatus"])

percent = []
sex = []
age = []
ss = []
for i in range(len(sub)):
    pid = sub.iloc[i, 1]
    percent.append(percent_dict[pid])
    sex.append(sex_dict[pid])
    age.append(age_dict[pid])
    ss.append(ss_dict[pid])

sub["base_week_percent"] = percent
sub["Age"] = age
sub["Sex"] = sex
sub["SmokingStatus"] = ss

base_week_test = test_csv.groupby("Patient")["Weeks"].min()
count_from_base_week_test = []
base_week_list = []
for i in range(len(sub)):
    pid = sub.iloc[i, 1]
    count_from_base_week_test.append(sub.iloc[i, 2] - base_week_test.loc[pid])
    base_week_list.append(base_week_test.loc[pid])
sub["count_from_base_week"] = count_from_base_week_test
sub["base_week"] = base_week_list

sub["base fev1/base fvc"] = sub["base_fev1"] / sub["base_fvc"]
sub["base_bmi"] = sub["base_weight"] / ((sub["base_height"] / 100) ** 2)



## === cell 8
test_smoking_str = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))[
    ["Patient", "SmokingStatus"]
]
test_smoking_str["SmokingStatus"] = test_smoking_str["SmokingStatus"].astype(str)
pid_to_smoke = dict(zip(test_smoking_str["Patient"], test_smoking_str["SmokingStatus"]))

sub["SmokingStatus_str"] = sub["Patient"].map(pid_to_smoke).astype(str)

smoke_cat_test = pd.DataFrame(
    oh1.transform(sub[["SmokingStatus_str"]]),
    columns=["smoking cat 0", "smoking cat 1", "smoking cat 2"],
)
sub = pd.concat([sub, smoke_cat_test], axis=1)



## === cell 9
sub_scaled = pd.DataFrame(
    sc.transform(
        sub[
            [
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
        ]
    ),
    columns=[
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
    ],
)
sub_scaled["Sex"] = sub["Sex"]
sub_scaled["smoking cat 0"] = sub["smoking cat 0"]
sub_scaled["smoking cat 1"] = sub["smoking cat 1"]

sub_scaled = sub_scaled.apply(pd.to_numeric, errors="coerce").fillna(0.0)



## === cell 10
x = train_scaled.to_numpy(dtype=np.float32)
y = train_csv[["FVC", "confidence"]].to_numpy(dtype=np.float32)

xtrain, xvalid, ytrain, yvalid = train_test_split(x, y, test_size=0.2, random_state=42)



## === cell 11
import google.protobuf  # noqa: F401
from google.protobuf import message_factory as _message_factory


def _ensure_getprototype_on_factory(factory_obj_or_cls):
    if hasattr(factory_obj_or_cls, "GetPrototype"):
        return
    if hasattr(factory_obj_or_cls, "GetMessageClass"):

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        try:
            setattr(factory_obj_or_cls, "GetPrototype", _GetPrototype)
        except Exception:
            pass


_ensure_getprototype_on_factory(_message_factory.MessageFactory)
try:
    _ensure_getprototype_on_factory(_message_factory._DEFAULT)
except Exception:
    pass

import tensorflow as tf

tf.keras.utils.set_random_seed(42)


def metric(actual_fvc, predicted_fvc, confidence, return_values=False):
    sd_clipped = np.maximum(confidence, 70.0)
    delta = np.minimum(np.abs(actual_fvc - predicted_fvc), 1000.0)
    m = -np.sqrt(2.0) * delta / sd_clipped - np.log(np.sqrt(2.0) * sd_clipped)
    if return_values:
        return m
    return float(np.mean(m))


def model_loss(ytrue, ypred):
    fvc_pred = ypred[:, 0]
    sigmas = tf.maximum(ypred[:, 1], 70.0)
    ans = tf.math.log(sigmas)
    ans = ans + ((ytrue[:, 0] - fvc_pred) ** 2) / (2.0 * sigmas**2)
    return tf.reduce_mean(ans)




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 12
class best_weights(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.metric_op = -30.0
        self.weights_op = None
        self.epoch_op = -1

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if logs.get("val_metric", -1e9) >= self.metric_op:
            self.metric_op = logs["val_metric"]
            self.epoch_op = epoch
            self.weights_op = self.model.get_weights()

    def on_train_end(self, logs=None):
        if self.weights_op is not None:
            self.model.set_weights(self.weights_op)
        print(
            "BEST_EPOCH = {}   BEST_SCORE_ON_VALID_SET = {}".format(
                self.epoch_op + 1, self.metric_op
            )
        )


class metrics_call(tf.keras.callbacks.Callback):
    def __init__(self, mertic, xtrain, ytrain, xvalid, yvalid):
        super().__init__()
        self.metric = mertic
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xvalid = xvalid
        self.yvalid = yvalid

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        val_preds = self.model.predict(self.xvalid, verbose=0)
        val_preds = np.asarray(val_preds, dtype=np.float32)
        logs["val_metric"] = self.metric(
            self.yvalid[:, 0].astype(np.float32),
            val_preds[:, 0].astype(np.float32),
            val_preds[:, 1].astype(np.float32),
        )


def run_model(xtrain, ytrain, xvalid, yvalid, epoch=50):
    input_layer = tf.keras.layers.Input(shape=xtrain.shape[1:])
    noisy = tf.keras.layers.GaussianNoise(0.3)(input_layer)

    d1 = tf.keras.layers.Dense(128, activation="relu")(noisy)
    d2 = tf.keras.layers.Dense(128, activation="relu")(d1)
    d3 = tf.keras.layers.Dense(128, activation="relu")(d2)
    mean_out1 = tf.keras.layers.Dense(1)(d3)
    std_den1 = tf.keras.layers.Dense(1)(d3)

    d4 = tf.keras.layers.Dense(128, activation="relu")(noisy)
    d5 = tf.keras.layers.Dense(128, activation="relu")(d4)
    d6 = tf.keras.layers.Dense(128, activation="relu")(d5)
    mean_out2 = tf.keras.layers.Dense(1)(d6)
    std_den2 = tf.keras.layers.Dense(1)(d6)

    d7 = tf.keras.layers.Dense(128, activation="relu")(noisy)
    d8 = tf.keras.layers.Dense(128, activation="relu")(d7)
    d9 = tf.keras.layers.Dense(128, activation="relu")(d8)
    mean_out3 = tf.keras.layers.Dense(1)(d9)
    std_den3 = tf.keras.layers.Dense(1)(d9)

    mean_combine = tf.keras.layers.Concatenate()([mean_out1, mean_out2, mean_out3])
    std_combine = tf.keras.layers.Concatenate()([std_den1, std_den2, std_den3])
    mean_final = tf.keras.layers.Dense(1)(mean_combine)
    std_final_den = tf.keras.layers.Dense(1)(std_combine)
    std_final = tf.keras.layers.Activation("relu")(std_final_den)
    output = tf.keras.layers.Concatenate()([mean_final, std_final])

    model = tf.keras.models.Model(inputs=input_layer, outputs=output)

    model.compile(
        loss=lambda ytrue, ypred: model_loss(ytrue, ypred),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    )

    history = model.fit(
        xtrain,
        ytrain,
        epochs=epoch,
        batch_size=256,
        validation_data=(xvalid, yvalid),
        verbose=0,
        callbacks=[
            metrics_call(metric, xtrain, ytrain, xvalid, yvalid),
            best_weights(),
        ],
    )

    try:
        pd.DataFrame(history.history).plot(figsize=(8, 5))
        plt.ylim(-10, 10)
        plt.grid(True)
        plt.show()
    except Exception:
        pass

    return model




## === cell 13
model = run_model(xtrain, ytrain, xvalid, yvalid, epoch=300)



## === cell 14
xtest = sub_scaled.to_numpy(dtype=np.float32)
yans = model.predict(xtest, verbose=0)
yans = np.asarray(yans, dtype=np.float32)

pred_fvc = yans[:, 0].astype(np.float32)
pred_conf = np.maximum(np.abs(yans[:, 1]).astype(np.float32), 70.0)

submission = pd.DataFrame(
    {
        "Patient_Week": sub["Patient_Week"].astype(str).values,
        "FVC": pred_fvc,
        "Confidence": pred_conf,
    }
)

sample = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))[["Patient_Week"]]
submission = sample.merge(submission, on="Patient_Week", how="left")
submission["FVC"] = pd.to_numeric(submission["FVC"], errors="coerce").fillna(0.0)
submission["Confidence"] = pd.to_numeric(
    submission["Confidence"], errors="coerce"
).fillna(70.0)
submission["Confidence"] = np.maximum(submission["Confidence"].to_numpy(), 70.0)

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
