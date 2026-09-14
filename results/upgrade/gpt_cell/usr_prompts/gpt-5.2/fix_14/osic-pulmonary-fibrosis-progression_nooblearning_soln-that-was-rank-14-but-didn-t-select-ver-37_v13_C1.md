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

-6.8484

# 6. Current score

-9.88415

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -7.64759) has done: 'Diagnosis: Cell 15 crashes because `model` was never created in the executed path. The notebook defines `run_model(...)` in cells 12/13, but no cell shown actually calls it, so `model.predict(...)` raises `NameError`. To keep the same model architecture/training semantics, we should instantiate `model` by calling the already-defined `run_model` only if it does not exist, then proceed with the existing prediction/submission logic.

Patch summary: In cell 15, add a small guard that checks for `model` in globals and, if missing, trains it via `run_model(xtrain, ytrain, xvalid, yvalid)` using the default epoch argument. This is the minimal localized change to ensure `model` exists before `predict`, without changing features, architecture, or loss.

Updated cells:'
- What this solution (achieved -9.88415) has done: 'To move the score upward toward the target (your current -7.64759 is worse than -6.8484), I make two minimal, metric-aligned fixes without changing the model architecture or training loop: (1) use the competition’s clinically meaningful baseline as `Weeks==0` when available (instead of `min Weeks`) for both train and test feature engineering, and (2) post-process predictions to enforce the metric’s constraints by clipping `Confidence` to at least 70 and rounding `FVC` to integer ml. These changes improve alignment between features and what the test setup represents (baseline CT at Week 0) and prevent low predicted sigmas from being heavily penalized by the log-likelihood score. I also remove the redundant second `run_model` definition to avoid accidental divergence, keeping exactly the same core model/training semantics. The script still run end-to-end and write `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pydicom

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from skimage import morphology
from skimage import measure
from skimage.transform import resize

from sklearn.cluster import KMeans
import matplotlib.patches as patches



## === cell 1
train_csv = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")



## === cell 2
train_csv



## === cell 3
min_week = train_csv.groupby("Patient")["Weeks"].min()
has_week0 = train_csv["Weeks"].eq(0).groupby(train_csv["Patient"]).any()
baseline_week = pd.Series(
    {p: (0 if has_week0.loc[p] else int(min_week.loc[p])) for p in min_week.index}
)

train_csv["base_week"] = train_csv["Patient"].map(baseline_week).astype(np.int32)
train_csv["count_from_base_week"] = (
    train_csv["Weeks"] - train_csv["base_week"]
).astype(np.int32)

train_csv["confidence"] = 0.0

base_fvc_dict = {}
for pid in train_csv["Patient"].unique():
    bw = baseline_week[pid]
    base_fvc_dict[pid] = float(
        train_csv[(train_csv["Patient"] == pid) & (train_csv["Weeks"] == bw)][
            "FVC"
        ].iloc[0]
    )
train_csv["base_fvc"] = train_csv["Patient"].map(base_fvc_dict).astype(np.float32)

base_fev1_dict = {}
for pid in train_csv["Patient"].unique():
    A = float(train_csv[train_csv["Patient"] == pid]["base_fvc"].unique()[0])
    B = float(train_csv[train_csv["Patient"] == pid]["Age"].unique()[0])
    if train_csv[train_csv["Patient"] == pid]["Sex"].unique()[0] == "Male":
        base_fev1_dict[pid] = 0.77 * A + 0.32 + 0.0069 * B
    else:
        base_fev1_dict[pid] = 0.77 * A + 0.28 + 0.0052 * B
train_csv["base_fev1"] = train_csv["Patient"].map(base_fev1_dict).astype(np.float32)

base_week_percent_dict = {}
for pid in train_csv["Patient"].unique():
    bw = baseline_week[pid]
    base_week_percent_dict[pid] = float(
        train_csv[(train_csv["Patient"] == pid) & (train_csv["Weeks"] == bw)][
            "Percent"
        ].iloc[0]
    )
train_csv["base_week_percent"] = (
    train_csv["Patient"].map(base_week_percent_dict).astype(np.float32)
)

train_csv["base fev1/base fvc"] = (
    train_csv["base_fev1"] / train_csv["base_fvc"]
).astype(np.float32)
train_csv["base_height"] = ((train_csv["base_fvc"] + 9030) / 77.0).astype(np.float32)



## === cell 4
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()  # sex
train_csv.iloc[:, 5] = lb.fit_transform(train_csv.iloc[:, 5])
lb2 = LabelEncoder()  # ss
train_csv.iloc[:, 6] = lb2.fit_transform(train_csv.iloc[:, 6])



## === cell 5
train_csv



## === cell 6
sub = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/sample_submission.csv")
test_csv = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")



## === cell 7
test_week = []
patient_id = []
for i in range(len(sub)):
    test_week.append(int(sub.iloc[i, 0].split("_")[-1]))
    patient_id.append(sub.iloc[i, 0].split("_")[0])
sub["Patient"] = patient_id
sub["Weeks"] = test_week
sub.drop(["FVC", "Confidence"], axis=1, inplace=True)

test_min_week = test_csv.groupby("Patient")["Weeks"].min()
test_has_week0 = test_csv["Weeks"].eq(0).groupby(test_csv["Patient"]).any()
test_baseline_week = pd.Series(
    {
        p: (0 if test_has_week0.loc[p] else int(test_min_week.loc[p]))
        for p in test_min_week.index
    }
)

base_fvc_test_dict = {}
for pid in test_csv["Patient"].unique():
    bw = test_baseline_week[pid]
    base_fvc_test_dict[pid] = float(
        test_csv[(test_csv["Patient"] == pid) & (test_csv["Weeks"] == bw)]["FVC"].iloc[
            0
        ]
    )
sub["base_fvc"] = sub["Patient"].map(base_fvc_test_dict).astype(np.float32)

base_fev1_dict_test = {}
for pid in sub["Patient"].unique():
    A = float(sub[sub["Patient"] == pid]["base_fvc"].unique()[0])
    B = float(test_csv[test_csv["Patient"] == pid]["Age"].unique()[0])
    if test_csv[test_csv["Patient"] == pid]["Sex"].unique()[0] == "Male":
        base_fev1_dict_test[pid] = 0.77 * A + 0.32 + 0.0069 * B
    else:
        base_fev1_dict_test[pid] = 0.77 * A + 0.28 + 0.0052 * B
sub["base_fev1"] = sub["Patient"].map(base_fev1_dict_test).astype(np.float32)

test_csv.iloc[:, 5] = lb.transform(test_csv.iloc[:, 5])
test_csv.iloc[:, 6] = lb2.transform(test_csv.iloc[:, 6])

percent_dict = {}
sex_dict = {}
age_dict = {}
ss_dict = {}
for pid in test_csv["Patient"].unique():
    percent_dict[pid] = float(test_csv[test_csv["Patient"] == pid]["Percent"].iloc[0])
    sex_dict[pid] = int(test_csv[test_csv["Patient"] == pid]["Sex"].iloc[0])
    age_dict[pid] = int(test_csv[test_csv["Patient"] == pid]["Age"].iloc[0])
    ss_dict[pid] = int(test_csv[test_csv["Patient"] == pid]["SmokingStatus"].iloc[0])

sub["base_week_percent"] = sub["Patient"].map(percent_dict).astype(np.float32)
sub["Age"] = sub["Patient"].map(age_dict).astype(np.int32)
sub["Sex"] = sub["Patient"].map(sex_dict).astype(np.int32)
sub["SmokingStatus"] = sub["Patient"].map(ss_dict).astype(np.int32)

sub["base_week"] = sub["Patient"].map(test_baseline_week).astype(np.int32)
sub["count_from_base_week"] = (sub["Weeks"] - sub["base_week"]).astype(np.int32)

sub["base fev1/base fvc"] = (sub["base_fev1"] / sub["base_fvc"]).astype(np.float32)
sub["base_height"] = ((sub["base_fvc"] + 9030) / 77.0).astype(np.float32)



## === cell 8
sub



## === cell 9
x = np.array(
    train_csv[
        [
            "Weeks",
            "Age",
            "Sex",
            "SmokingStatus",
            "base_week",
            "count_from_base_week",
            "base_fvc",
            "base_fev1",
            "base_week_percent",
            "base fev1/base fvc",
            "base_height",
        ]
    ]
)
y = np.array(train_csv[["FVC", "confidence"]])

from sklearn.model_selection import train_test_split

xtrain, xvalid, ytrain, yvalid = train_test_split(x, y, test_size=0.2)




## === cell 10
def metric(actual_fvc, predicted_fvc, confidence, return_values=False):
    """
    Calculates the modified Laplace Log Likelihood score for this competition.
    """
    sd_clipped = np.maximum(confidence, 70)
    delta = np.minimum(np.abs(actual_fvc - predicted_fvc), 1000)
    metric_val = -np.sqrt(2) * delta / sd_clipped - np.log(np.sqrt(2) * sd_clipped)

    if return_values:
        return metric_val
    else:
        return np.mean(metric_val)


def model_loss(
    ytrue, ypred
):  # this loss penalises both prediction and confidence value
    eps = 1.0
    fvc_pred = ypred[:, 0]
    sigmas = ypred[:, 1] + eps  # avoid log(0); predicted confidences
    ans = tf.math.log(sigmas)
    ans = ans + (tf.abs(ytrue[:, 0] - fvc_pred)) / (sigmas)
    return tf.reduce_mean(ans)




## === cell 11
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

import sys
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "protobuf<5"])

import tensorflow as tf


class best_weights(tf.keras.callbacks.Callback):
    def __init__(self):
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
            "\n\n BEST_EPOCH = {}   BEST_SCORE_ON_VALID_SET = {}".format(
                self.epoch_op + 1, self.metric_op
            )
        )


class metrics_call(tf.keras.callbacks.Callback):
    def __init__(self, mertic, xtrain, ytrain, xvalid, yvalid):
        self.metric = metric
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xvalid = xvalid
        self.yvalid = yvalid

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_preds = self.model.predict(self.xtrain, verbose=0)
        val_preds = self.model.predict(self.xvalid, verbose=0)
        print(
            "\r  metric on train set: ",
            self.metric(self.ytrain[:, 0], train_preds[:, 0], train_preds[:, 1]),
            end="",
        )
        logs["val_metric"] = self.metric(
            self.yvalid[:, 0], val_preds[:, 0], val_preds[:, 1]
        )
        print(
            "  metric on valid set: ",
            self.metric(self.yvalid[:, 0], val_preds[:, 0], val_preds[:, 1]),
        )


def run_model(xtrain, ytrain, xvalid, yvalid, epoch=50):
    input_layer = tf.keras.layers.Input(shape=xtrain.shape[1:])
    d1 = tf.keras.layers.Dense(128, activation="relu")(input_layer)
    d2 = tf.keras.layers.Dense(128, activation="relu")(d1)
    d3 = tf.keras.layers.Dense(128, activation="relu")(d2)
    mean_out = tf.keras.layers.Dense(1)(d3)
    std_den = tf.keras.layers.Dense(1)(d3)
    std_out = tf.keras.layers.Activation("relu")(std_den)  # confidence positive
    output = tf.keras.layers.Concatenate()([mean_out, std_out])
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
        callbacks=[
            metrics_call(metric, xtrain, ytrain, xvalid, yvalid),
            best_weights(),
        ],
        verbose=0,
    )
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.ylim(-10, 10)
    plt.grid(True)

    return model




## === cell 12
pass



## === cell 13
x = train_csv[
    [
        "Weeks",
        "Age",
        "Sex",
        "SmokingStatus",
        "base_week",
        "count_from_base_week",
        "base_fvc",
        "base_fev1",
        "base_week_percent",
        "base fev1/base fvc",
        "base_height",
    ]
].to_numpy(dtype=np.float32)

y = train_csv[["FVC", "confidence"]].to_numpy(dtype=np.float32)

from sklearn.model_selection import train_test_split

xtrain, xvalid, ytrain, yvalid = train_test_split(x, y, test_size=0.2)



## === cell 14
if "model" not in globals():
    model = run_model(xtrain, ytrain, xvalid, yvalid)

xtest = sub[
    [
        "Weeks",
        "Age",
        "Sex",
        "SmokingStatus",
        "base_week",
        "count_from_base_week",
        "base_fvc",
        "base_fev1",
        "base_week_percent",
        "base fev1/base fvc",
        "base_height",
    ]
].to_numpy(dtype=np.float32)

yans = model.predict(xtest, verbose=0)

sub["FVC"] = yans[:, 0]
sub["Confidence"] = yans[:, 1]

sub["Confidence"] = np.maximum(sub["Confidence"].to_numpy(dtype=np.float32), 70.0)
sub["FVC"] = np.rint(sub["FVC"].to_numpy(dtype=np.float32)).astype(np.int32)

sub.drop(
    [
        "Patient",
        "Weeks",
        "Age",
        "Sex",
        "SmokingStatus",
        "base_week",
        "count_from_base_week",
        "base_fvc",
        "base_fev1",
        "base_week_percent",
        "base fev1/base fvc",
        "base_height",
    ],
    axis=1,
    inplace=True,
)



## === cell 15
sub



## === cell 16
sub.to_csv("submission.csv", index=False)
