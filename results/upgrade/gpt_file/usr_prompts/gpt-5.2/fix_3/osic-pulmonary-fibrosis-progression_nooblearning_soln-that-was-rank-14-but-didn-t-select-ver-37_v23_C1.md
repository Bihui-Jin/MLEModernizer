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

-6.9199

# 6. Current score

-9.455

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -9.65882) has done: 'I fix the runtime blockers so the notebook runs end-to-end in the Kaggle environment and writes a valid `submission.csv` with the required columns. Concretely: remove/avoid the TensorFlow import crash caused by an incompatible protobuf runtime (it isn’t needed for your tabular baseline anyway), fix the Adam optimizer argument (`lr` → `learning_rate`) to match TF/Keras 2.18, and correct the test feature-building bugs (several dict extractions were returning Series instead of scalars). Finally, I ensure the submission keeps `Patient_Week` and includes properly named `FVC` and `Confidence` columns (your current code was dropping `FVC`/`Confidence` and then later dropping `Patient_Week` implicitly), so Kaggle accepts the file.'
- What this solution (achieved -9.455) has done: 'I fix the TensorFlow import crash that comes from a protobuf API mismatch by forcing the pure-Python protobuf implementation *before any protobuf-dependent imports*, and by defensively falling back to a lightweight non-TF baseline only if TF still can’t import (so you always get a valid `submission.csv`). To improve score toward the target, I also make the training target confidence non-zero (the current all-zero sigma encourages pathological training and poor Laplace score) by setting it to a reasonable constant consistent with the metric’s clipping behavior; this keeps the model/loop/architecture intact while aligning the loss with the evaluation. Finally, I fix a small bug in the callback constructor argument (`mertic`) to avoid confusion and ensure validation metric tracking works as intended. The rest of the pipeline, features, and submission format are preserved.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pydicom  # unused but kept to preserve original intent/environment parity

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

np.random.seed(42)



## === cell 1
train_csv = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")



## === cell 2
train_csv



## === cell 3
base_week = train_csv.groupby("Patient")["Weeks"].min()
base_week_list = []
for i in range(len(train_csv)):
    base_week_list.append(base_week[train_csv.iloc[i, 0]])
train_csv["base_week"] = base_week_list

base_week = train_csv.groupby("Patient")["Weeks"].min()
count_from_base_week = []
for i in range(len(train_csv)):
    count_from_base_week.append(train_csv.iloc[i, 1] - base_week[train_csv.iloc[i, 0]])
train_csv["count_from_base_week"] = count_from_base_week

confidence = np.full(train_csv.shape[0], 200.0, dtype=np.float32)
train_csv["confidence"] = confidence

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



## === cell 4
lb = LabelEncoder()  # sex
train_csv.iloc[:, 5] = lb.fit_transform(train_csv.iloc[:, 5])
lb2 = LabelEncoder()  # smoking status
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
sub = sub.drop(["FVC", "Confidence"], axis=1)

base_fvc = test_csv.groupby("Patient")["FVC"].min()
fvc = []
for i in range(len(sub)):
    fvc.append(float(base_fvc[sub.iloc[i, 1]]))
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
    FVC0 = sub[sub["Patient"] == pid]["base_fvc"].unique()[0]
    A = test_csv[test_csv["Patient"] == pid]["Age"].unique()[0]
    H = sub[sub["Patient"] == pid]["base_height"].unique()[0]
    if test_csv[test_csv["Patient"] == pid]["Sex"].unique()[0] == "Male":
        base_weight_dict_test[pid] = (FVC0 + 5458 - 49 * H + 8 * A) / 12.0
    else:
        base_weight_dict_test[pid] = (FVC0 + 3863 - 37 * H + 6 * A) / 14.0

base_weight_test = []
for i in range(len(sub)):
    base_weight_test.append(base_weight_dict_test[sub.iloc[i, 1]])
sub["base_weight"] = base_weight_test

test_csv = test_csv.copy()
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
base_week_vals = []
for i in range(len(sub)):
    pid = sub.iloc[i, 1]
    count_from_base_week_test.append(sub.iloc[i, 2] - base_week_test[pid])
    base_week_vals.append(base_week_test[pid])
sub["count_from_base_week"] = count_from_base_week_test
sub["base_week"] = base_week_vals

sub["base fev1/base fvc"] = sub["base_fev1"] / sub["base_fvc"]
sub["base_bmi"] = sub["base_weight"] / ((sub["base_height"] / 100) ** 2)



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
            "base_weight",
            "base_bmi",
        ]
    ]
).astype(np.float32)
y = np.array(train_csv[["FVC", "confidence"]]).astype(np.float32)

xtrain, xvalid, ytrain, yvalid = train_test_split(x, y, test_size=0.2, random_state=42)




## === cell 10
def metric(actual_fvc, predicted_fvc, confidence, return_values=False):
    sd_clipped = np.maximum(confidence, 70)
    delta = np.minimum(np.abs(actual_fvc - predicted_fvc), 1000)
    m = -np.sqrt(2) * delta / sd_clipped - np.log(np.sqrt(2) * sd_clipped)
    if return_values:
        return m
    return np.mean(m)




## === cell 11
USE_TF = True
try:
    import tensorflow as tf
except Exception as e:
    USE_TF = False
    tf_import_error = repr(e)
    print("TensorFlow import failed; will use baseline predictor instead.")
    print("TF import error:", tf_import_error)

if USE_TF:

    def model_loss(ytrue, ypred):
        eps = 1.0
        fvc_pred = ypred[:, 0]
        sigmas = ypred[:, 1] + eps
        ans = tf.math.log(sigmas)
        ans = ans + ((ytrue[:, 0] - fvc_pred) ** 2) / (2 * sigmas**2)
        return tf.reduce_mean(ans)

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
                "BEST_EPOCH = {}   BEST_SCORE_ON_VALID_SET = {}".format(
                    self.epoch_op + 1, self.metric_op
                )
            )

    class metrics_call(tf.keras.callbacks.Callback):
        def __init__(self, metric_fn, xtrain, ytrain, xvalid, yvalid):
            self.metric = metric_fn
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
        input_ = tf.keras.layers.Input(shape=xtrain.shape[1:])
        noisy = tf.keras.layers.GaussianNoise(0.3)(input_)

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

        model = tf.keras.models.Model(inputs=input_, outputs=output)

        model.compile(
            loss=lambda ytrue, ypred: model_loss(ytrue, ypred),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        )
        print(model.summary())

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
        plt.show()

        return model




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 12
if USE_TF:
    model = run_model(xtrain, ytrain, xvalid, yvalid, epoch=400)
else:
    model = None



## === cell 13
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
        "base_weight",
        "base_bmi",
    ]
].astype(np.float32)

if USE_TF:
    yans = model.predict(xtest.values, verbose=0)
    pred_fvc = yans[:, 0]
    pred_conf = np.abs(yans[:, 1])
else:
    pred_fvc = sub["base_fvc"].values.astype(np.float32)
    pred_conf = np.full(len(sub), 200.0, dtype=np.float32)

submission = pd.DataFrame(
    {
        "Patient_Week": sub["Patient_Week"].values,
        "FVC": pred_fvc,
        "Confidence": pred_conf,
    }
)



## === cell 14
submission



## === cell 15
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print("Columns:", submission.columns.tolist())
print(submission.head())
