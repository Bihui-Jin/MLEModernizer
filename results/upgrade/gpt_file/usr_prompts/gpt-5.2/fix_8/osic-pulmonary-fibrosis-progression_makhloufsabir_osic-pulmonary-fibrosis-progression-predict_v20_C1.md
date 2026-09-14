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

category_encoders==2.7.0
geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
pillow==11.3.0
protobuf==6.33.0
pydicom==3.0.1
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
seaborn==0.12.2
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
tf_keras==2.18.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1

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

-11.0734

# 6. Current score

-8.18866

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -9.11728) has done: 'I fix the import/runtime crash caused by the protobuf incompatibility (triggered indirectly by Keras/TensorFlow imports) by setting the protobuf implementation to pure-Python before any TF/Keras-related import. I also remove hard GPU assertions so the notebook runs reliably on CPU-only sessions, while keeping the modeling logic unchanged. Next I update the Seaborn plotting calls and the correlation heatmap to match newer seaborn/pandas APIs so EDA cells don’t stop execution. Finally, I fix the model compile (it currently provides two losses for a single-output model) and make label encoding consistent between train/test, ensuring the pipeline trains, predicts, and writes a valid `submission.csv` with correct columns.'
- What this solution (achieved -9.12538) has done: 'I fix the TensorFlow import crash by pinning protobuf to the pure-Python implementation and also forcing TensorFlow to use the Python protobuf runtime (this is the direct cause of the `MessageFactory.GetPrototype` error in this environment). Since your current score (-9.11728) is better than the target (-11.0734), I avoid any model/training/prediction changes that could further improve score; changes be score-neutral and focused on stability. I also ensure the saved checkpoint format is compatible with TF/Keras 2.18+ by saving weights-only (to avoid any `.h5` save/load edge cases). Finally, I keep the submission format identical but add a quick sanity check to guarantee the CSV has the required columns and row count.'
- What this solution (achieved -19.09388) has done: 'I fix the TensorFlow import crash by ensuring the protobuf Python implementation environment variables are set before any TensorFlow/Keras import (and by avoiding any earlier TF-triggering imports). I also keep the existing model/training logic intact, but remove the per-row `scipy.optimize.minimize` confidence search (it is extremely slow and unnecessary for a valid submission) and replace it with the same constant confidence used earlier so the notebook finishes within the time limit. Finally, I keep the submission schema identical and add a small safety clip/cast so `FVC` is numeric and the CSV is always written successfully. These changes are primarily stability/runtime fixes and may slightly change score (your current score is already better than target, so we avoid intentional improvements).'
- What this solution (achieved -13.93695) has done: 'I fix the TensorFlow import crash (`MessageFactory` / protobuf incompatibility) by setting the protobuf env vars before any TensorFlow-related import and by ensuring we never import TF in a way that bypasses those settings. Then I remove the unintended `* 1000.0` scaling applied to the predicted FVC at submission time, which is a logic bug that severely harms the Laplace log-likelihood score (it makes predictions wildly off-scale). I also make the submission write step robust by clipping/casting numeric types without changing the model/training loop or architecture. These changes should run end-to-end and improve score toward the target while preserving the core model logic.'
- What this solution (achieved -13.93763) has done: 'I fix the TensorFlow/protobuf crash by setting the required environment variables before any TF/Keras import and by avoiding importing TensorFlow until it’s actually needed for the LSTM model. Then I keep the model/training logic identical but correct a scoring-impacting mismatch: the model is trained to predict `FVC` directly, yet the current quick “score” sanity-check compares predictions to `base_FVC` (baseline), not the true target `FVC`; I change that check to evaluate on validation labels only (score-neutral for submission, but prevents misleading debugging). Finally, I keep the submission generation the same while ensuring predictions are aligned to `Patient_Week` and written as a valid `submission.csv`.'
- What this solution (achieved -8.18899) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by setting the protobuf environment variables *before any TensorFlow-related import* and importing TensorFlow in a minimal/safer way for this Kaggle image. Then I make the categorical encoding robust by fitting the label encoders on the union of train+test categories so `.transform()` can’t crash on unseen test labels (this is score-neutral but stabilizes runtime). Finally, to move the score upward toward your target (current -13.94 is worse than target -11.07), I change only the submission-time `Confidence` from a hardcoded 100 to a constant calibrated from validation residuals (still a constant, so core modeling is unchanged), which typically improves the Laplace log-likelihood without altering FVC predictions.'
- What this solution (achieved -8.18866) has done: 'I fix the TensorFlow/protobuf crash by enforcing the pure-Python protobuf runtime *and* importing TensorFlow only after those environment variables are set, plus clearing any previously loaded protobuf modules to prevent the C++ runtime from being used. This change is runtime-only and should not alter your model logic or training semantics. I also keep your current constant-confidence calibration (median absolute residuals) unchanged, since your current score is already better than the target and we should avoid score-changing modifications. Finally, I keep the submission generation intact and ensure `submission.csv` is always written with the required columns.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["TF_USE_C_API_PROBUF"] = "0"

import sys

for _m in list(sys.modules.keys()):
    if _m.startswith(("google.protobuf", "tensorflow")):
        sys.modules.pop(_m, None)

import numpy as np
import pandas as pd
import pydicom
import seaborn as sns
import seaborn as sb
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import category_encoders as ce

import cv2
import lightgbm as lgb

import warnings

warnings.filterwarnings("ignore")

plt.style.use("seaborn-darkgrid")



## === cell 1
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("There are %d GPU(s) available." % torch.cuda.device_count())
    print("We will use the GPU:", torch.cuda.get_device_name(0))
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")



## === cell 2
train = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")
test = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")



## === cell 3
sub = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/sample_submission.csv")
sub["Patient"] = sub["Patient_Week"].apply(lambda x: x.split("_")[0])
sub["Weeks"] = sub["Patient_Week"].apply(lambda x: int(x.split("_")[-1]))
sub = sub[["Patient", "Weeks", "Confidence", "Patient_Week"]]
sub = sub.merge(test.drop("Weeks", axis=1), on="Patient")



## === cell 4
sub



## === cell 5
train.head(100)



## === cell 6
test.head()



## === cell 7
print(train.shape, test.shape)



## === cell 8
print(train.isnull().sum(), "\n")
print(test.isnull().sum())



## === cell 9
train.info()



## === cell 10
plt.figure(figsize=(16, 10))



## === cell 11
sns.barplot(x=train["Sex"].value_counts().index, y=train["Sex"].value_counts().values)



## === cell 12
sns.barplot(
    x=train["SmokingStatus"].value_counts().index,
    y=train["SmokingStatus"].value_counts().values,
)



## === cell 13
sa = pd.crosstab(train["SmokingStatus"], train["Sex"])
sa.plot(kind="bar", title="No of passengers survived")
plt.show()



## === cell 14
plt.rcParams["figure.figsize"] = 10, 5
ax = train["Age"].hist(bins=15, alpha=0.9, color="green")
ax.set(xlabel="Age", ylabel="Count", title="Visualization of Ages")
plt.show()



## === cell 15
plt.rcParams["figure.figsize"] = 10, 5
ax = train["Weeks"].hist(bins=15, alpha=0.9, color="green")
ax.set(xlabel="Weeks", ylabel="Count", title="Visualization of Ages")
plt.show()



## === cell 16
plt.scatter(train["Weeks"], train["FVC"])



## === cell 17
plt.scatter(train["Age"], train["FVC"])



## === cell 18
plt.rcParams["figure.figsize"] = 10, 10
sb.heatmap(
    train.corr(numeric_only=True),
    annot=True,
    square=True,
    linewidths=2,
    linecolor="black",
)



## === cell 19
train[train["FVC"] == train["FVC"].max()]



## === cell 20
train[train["FVC"] == train["FVC"].min()]



## === cell 21
imdir_max = (
    "/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00219637202258203123958"
)
imdir_min = (
    "/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00225637202259339837603"
)

if os.path.isdir(imdir_max):
    fig = plt.figure(figsize=(12, 12))
    columns = 4
    rows = 5
    for i in range(1, columns * rows + 1):
        filename = os.path.join(imdir_max, f"{i}.dcm")
        if os.path.exists(filename):
            ds = pydicom.dcmread(filename)
            fig.add_subplot(rows, columns, i)
            plt.imshow(ds.pixel_array, cmap="jet")
    plt.show()
else:
    print(f"Skipping visualization; folder not found: {imdir_max}")



## === cell 22
if os.path.isdir(imdir_min):
    fig = plt.figure(figsize=(12, 12))
    columns = 4
    rows = 5
    for i in range(1, columns * rows + 1):
        filename = os.path.join(imdir_min, f"{i}.dcm")
        if os.path.exists(filename):
            ds = pydicom.dcmread(filename)
            fig.add_subplot(rows, columns, i)
            plt.imshow(ds.pixel_array, cmap="jet")
    plt.show()
else:
    print(f"Skipping visualization; folder not found: {imdir_min}")



## === cell 23
train["Patient_Week"] = train["Patient"].astype(str) + "_" + train["Weeks"].astype(str)
train.head()



## === cell 24
test["Patient_Week"] = test["Patient"].astype(str) + "_" + test["Weeks"].astype(str)
test.head()



## === cell 25
print(train.shape)
print(test.shape)
train



## === cell 26
output = pd.DataFrame()
gb = train.groupby("Patient")
tk0 = tqdm(gb, total=len(gb))
for _, usr_df in tk0:
    usr_output = pd.DataFrame()
    for week, tmp in usr_df.groupby("Weeks"):
        rename_cols = {
            "Weeks": "base_Week",
            "FVC": "base_FVC",
            "Percent": "base_Percent",
            "Age": "base_Age",
        }
        tmp = tmp.drop(columns="Patient_Week").rename(columns=rename_cols)
        drop_cols = ["Age", "Sex", "SmokingStatus", "Percent"]
        _usr_output = (
            usr_df.drop(columns=drop_cols)
            .rename(columns={"Weeks": "predict_Week"})
            .merge(tmp, on="Patient")
        )
        _usr_output["Week_passed"] = (
            _usr_output["predict_Week"] - _usr_output["base_Week"]
        )
        usr_output = pd.concat([usr_output, _usr_output])
    output = pd.concat([output, usr_output])

train = output[output["Week_passed"] != 0].reset_index(drop=True)
train.head()



## === cell 27
test = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv").rename(
    columns={
        "Weeks": "base_Week",
        "FVC": "base_FVC",
        "Percent": "base_Percent",
        "Age": "base_Age",
    }
)
submission = pd.read_csv(
    "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)
submission["Patient"] = submission["Patient_Week"].apply(lambda x: x.split("_")[0])
submission["predict_Week"] = (
    submission["Patient_Week"].apply(lambda x: x.split("_")[1]).astype(int)
)
test = submission.drop(columns=["FVC", "Confidence"]).merge(test, on="Patient")
test["Week_passed"] = test["predict_Week"] - test["base_Week"]
test.head()



## === cell 28
print(submission.shape)
print(test.shape)
print(train.shape)



## === cell 29
print(train.isnull().sum(), "\n")
print(test.isnull().sum())



## === cell 30
train.set_index(["Patient_Week"], inplace=True)
test.set_index(["Patient_Week"], inplace=True)



## === cell 31
train



## === cell 32
y = train["FVC"]
X = train.drop(["FVC"], axis=1)
X = X.drop(["Patient"], axis=1)

test_X = test.drop(["Patient"], axis=1)



## === cell 33
X



## === cell 34
enc_sex = LabelEncoder()
enc_smoke = LabelEncoder()

sex_all = pd.concat([X["Sex"].astype(str), test_X["Sex"].astype(str)], axis=0)
smoke_all = pd.concat(
    [X["SmokingStatus"].astype(str), test_X["SmokingStatus"].astype(str)], axis=0
)

enc_sex.fit(sex_all)
enc_smoke.fit(smoke_all)

X["Sex"] = enc_sex.transform(X["Sex"].astype(str))
X["SmokingStatus"] = enc_smoke.transform(X["SmokingStatus"].astype(str))



## === cell 35
X



## === cell 36
test_X["Sex"] = enc_sex.transform(test_X["Sex"].astype(str))
test_X["SmokingStatus"] = enc_smoke.transform(test_X["SmokingStatus"].astype(str))



## === cell 37
test_X



## === cell 38
y



## === cell 39
X = normalize(X)
test_X = normalize(test_X)



## === cell 40
kf = KFold(n_splits=5, random_state=2020, shuffle=True)

for train_index, val_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", val_index)
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]



## === cell 41
print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)

X_train = X_train.reshape(-1, 1, 8)
X_val = X_val.reshape(-1, 1, 8)
y_train = y_train.values
y_train = y_train.reshape(-1, 1)
y_val = y_val.values
y_val = y_val.reshape(-1, 1)

print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)



## === cell 42
import tensorflow as tf

device_name = tf.test.gpu_device_name()
if device_name == "/device:GPU:0":
    print("Found GPU at: {}".format(device_name))
else:
    print("GPU device not found; running on CPU.")



## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 43
from tensorflow.keras.layers import LSTM, LeakyReLU, Input, Dropout, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import Model

input_layer = Input(shape=(1, 8))
main_rnn_layer = LSTM(64, return_sequences=True, recurrent_dropout=0.2)(input_layer)
rnn = LSTM(32)(main_rnn_layer)
dense = Dense(128)(rnn)
dropout_c = Dropout(0.3)(dense)
classes = Dense(1, activation=LeakyReLU(alpha=0.1), name="class")(dropout_c)

model = Model(input_layer, classes)

callbacks = [
    ReduceLROnPlateau(monitor="val_loss", patience=4, verbose=1, factor=0.6),
    EarlyStopping(monitor="val_loss", patience=20),
    ModelCheckpoint(
        filepath="best_model.weights.h5",
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
    ),
]

model.compile(loss=tf.keras.losses.MeanSquaredLogarithmicError(), optimizer="adam")

model.summary()
history = model.fit(
    X_train,
    y_train,
    epochs=250,
    batch_size=16,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
)



## === cell 44
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Loss over epochs")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"], loc="best")
plt.show()



## === cell 45
model.load_weights("best_model.weights.h5")

test_X = test_X.reshape(-1, 1, 8)
predictions = model.predict(test_X, verbose=0)



## === cell 46
print(predictions.shape)
print(predictions[:5])



## === cell 47
test



## === cell 48
submis = test.copy()



## === cell 49
import math

val_pred = model.predict(X_val, verbose=0)[:, 0].astype(float)
val_true = y_val.reshape(-1).astype(float)

abs_err = np.abs(val_true - val_pred)
sigma_cal = float(np.maximum(70.0, np.median(abs_err) * math.sqrt(2)))

sigma = sigma_cal
sigma_clipped = max(sigma, 70.0)
delta = np.minimum(np.abs(val_true - val_pred), 1000.0)
val_score = (
    -math.sqrt(2) * delta / sigma_clipped - np.log(math.sqrt(2) * sigma_clipped)
).mean()
print("Validation proxy Laplace score (sigma=calibrated):", float(val_score))
print("Chosen constant Confidence (sigma):", float(sigma_cal))



## === cell 50
pass



## === cell 51
pass



## === cell 52
submis["FVC_pred"] = predictions[:, 0].astype(float)
submis["Confidence"] = float(sigma_cal)



## === cell 53
submis = submis.reset_index()
submis



## === cell 54
submis_final = submis[["Patient_Week", "FVC_pred", "Confidence"]].copy()
submis_final = submis_final.rename(columns={"FVC_pred": "FVC"})

submis_final["FVC"] = pd.to_numeric(submis_final["FVC"], errors="coerce")
submis_final["FVC"] = submis_final["FVC"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
submis_final["FVC"] = submis_final["FVC"].clip(lower=0.0, upper=20000.0)

required_cols = ["Patient_Week", "FVC", "Confidence"]
assert (
    list(submis_final.columns) == required_cols
), f"Bad columns: {submis_final.columns}"
assert len(submis_final) == len(
    submission
), f"Row mismatch: {len(submis_final)} vs {len(submission)}"
assert submis_final["Patient_Week"].isna().sum() == 0

submis_final.to_csv("submission.csv", index=False)
submis_final
