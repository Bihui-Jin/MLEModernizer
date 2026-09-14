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

-6.8474

# 6. Current score

-9.46938

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved -9.46938) has done: 'I fix the environment-breaking TensorFlow import error by forcing the pure‑Python protobuf implementation before importing TensorFlow. Then I correct the OneHotEncoder usage: you fit it on string `SmokingStatus` but later transform an integer-coded column, which causes the `isnan` TypeError and the missing “smoking cat” columns; the fix is to keep `SmokingStatus` as strings for OHE on both train and test/sub. Next I update the Adam optimizer argument from deprecated `lr` to `learning_rate` so the model compiles under TF/Keras 2.18. Finally, I ensure the submission keeps/creates exactly `Patient_Week,FVC,Confidence` columns and is written to `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pydicom  # kept (not used later but part of original imports)
from sklearn.linear_model import LinearRegression  # kept
from sklearn.preprocessing import PolynomialFeatures  # kept
from sklearn.ensemble import RandomForestRegressor  # kept
from skimage import morphology  # kept
from skimage import measure  # kept
from skimage.transform import resize  # kept
import tensorflow as tf
from sklearn.cluster import KMeans  # kept
import matplotlib.patches as patches  # kept
import tensorflow.keras.backend as k  # kept

np.random.seed(42)
tf.random.set_seed(42)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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

confidence = np.zeros(train_csv.shape[0])
train_csv["confidence"] = confidence

base_fvc_dict = {}
for id in train_csv["Patient"].unique():
    base_fvc_dict[id] = np.array(
        train_csv[(train_csv["Patient"] == id) & (train_csv["Weeks"] == base_week[id])][
            "FVC"
        ]
    )[0]
base_fvc = []
for i in range(len(train_csv)):
    base_fvc.append(base_fvc_dict[train_csv.iloc[i, 0]])
train_csv["base_fvc"] = base_fvc

base_fev1_dict = {}
for id in train_csv["Patient"].unique():
    A = train_csv[train_csv["Patient"] == id]["base_fvc"].unique()[0]
    B = train_csv[train_csv["Patient"] == id]["Age"].unique()[0]
    if train_csv[train_csv["Patient"] == id]["Sex"].unique()[0] == "Male":
        base_fev1_dict[id] = 0.77 * A + 0.32 + 0.0069 * B
    else:
        base_fev1_dict[id] = 0.77 * A + 0.28 + 0.0052 * B

base_fev1 = []
for i in range(len(train_csv)):
    base_fev1.append(base_fev1_dict[train_csv.iloc[i, 0]])
train_csv["base_fev1"] = base_fev1

base_week_percent_dict = {}
for id in train_csv["Patient"].unique():
    base_week_percent_dict[id] = np.array(
        train_csv[(train_csv["Patient"] == id) & (train_csv["Weeks"] == base_week[id])][
            "Percent"
        ]
    )[0]

base_week_percent = []
for i in range(len(train_csv)):
    base_week_percent.append(base_week_percent_dict[train_csv.iloc[i, 0]])
train_csv["base_week_percent"] = base_week_percent

train_csv["base fev1/base fvc"] = train_csv["base_fev1"] / train_csv["base_fvc"]
train_csv["base_height"] = (train_csv["base_fvc"] + 9030) / 77.0

base_weight_dict = {}
for id in train_csv["Patient"].unique():
    FVC = train_csv[train_csv["Patient"] == id]["base_fvc"].unique()[0]
    A = train_csv[train_csv["Patient"] == id]["Age"].unique()[0]
    H = train_csv[train_csv["Patient"] == id]["base_height"].unique()[0]
    if train_csv[train_csv["Patient"] == id]["Sex"].unique()[0] == "Male":
        base_weight_dict[id] = (FVC + 5458 - 49 * H + 8 * A) / 12.0
    else:
        base_weight_dict[id] = (FVC + 3863 - 37 * H + 6 * A) / 14.0
base_weight = []
for i in range(len(train_csv)):
    base_weight.append(base_weight_dict[train_csv.iloc[i, 0]])
train_csv["base_weight"] = base_weight

train_csv["base_bmi"] = train_csv["base_weight"] / (
    (train_csv["base_height"] / 100) ** 2
)



## === cell 4
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()  # sex
train_csv.iloc[:, 5] = lb.fit_transform(train_csv.iloc[:, 5])



## === cell 5
from sklearn.preprocessing import OneHotEncoder

oh1 = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
smoke_arr = oh1.fit_transform(train_csv[["SmokingStatus"]])

smoke_cols = [f"smoking cat {i}" for i in range(smoke_arr.shape[1])]
smoke_cat = pd.DataFrame(smoke_arr, columns=smoke_cols, index=train_csv.index)
train_csv = pd.concat([train_csv, smoke_cat], axis=1)



## === cell 6
train_csv



## === cell 7
sub = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/sample_submission.csv")
test_csv = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")



## === cell 8
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
for id in sub["Patient"].unique():
    A = sub[sub["Patient"] == id]["base_fvc"].unique()[0]
    B = test_csv[test_csv["Patient"] == id]["Age"].unique()[0]
    if test_csv[test_csv["Patient"] == id]["Sex"].unique()[0] == "Male":
        base_fev1_dict_test[id] = 0.77 * A + 0.32 + 0.0069 * B
    else:
        base_fev1_dict_test[id] = 0.77 * A + 0.28 + 0.0052 * B

base_fev1_test = []
for i in range(len(sub)):
    base_fev1_test.append(base_fev1_dict_test[sub.iloc[i, 1]])
sub["base_fev1"] = base_fev1_test

sub["base_height"] = (sub["base_fvc"] + 9030) / 77.0

base_weight_dict_test = {}
for id in sub["Patient"].unique():
    FVC = sub[sub["Patient"] == id]["base_fvc"].unique()[0]
    A = test_csv[test_csv["Patient"] == id]["Age"].unique()[0]
    H = sub[sub["Patient"] == id]["base_height"].unique()[0]
    if test_csv[test_csv["Patient"] == id]["Sex"].unique()[0] == "Male":
        base_weight_dict_test[id] = (FVC + 5458 - 49 * H + 8 * A) / 12.0
    else:
        base_weight_dict_test[id] = (FVC + 3863 - 37 * H + 6 * A) / 14.0
base_weight_test = []
for i in range(len(sub)):
    base_weight_test.append(base_weight_dict_test[sub.iloc[i, 1]])
sub["base_weight"] = base_weight_test

test_csv.iloc[:, 5] = lb.transform(test_csv.iloc[:, 5])

percent_dict = {}
for id in test_csv["Patient"].unique():
    percent_dict[id] = float(test_csv[test_csv["Patient"] == id]["Percent"].iloc[0])

sex_dict = {}
for id in test_csv["Patient"].unique():
    sex_dict[id] = int(test_csv[test_csv["Patient"] == id]["Sex"].iloc[0])

age_dict = {}
for id in test_csv["Patient"].unique():
    age_dict[id] = int(test_csv[test_csv["Patient"] == id]["Age"].iloc[0])

ss_dict = {}
for id in test_csv["Patient"].unique():
    ss_dict[id] = str(test_csv[test_csv["Patient"] == id]["SmokingStatus"].iloc[0])

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
base_week = []
for i in range(len(sub)):
    count_from_base_week_test.append(sub.iloc[i, 2] - base_week_test[sub.iloc[i, 1]])
    base_week.append(base_week_test[sub.iloc[i, 1]])
sub["count_from_base_week"] = count_from_base_week_test
sub["base_week"] = base_week

sub["base fev1/base fvc"] = sub["base_fev1"] / sub["base_fvc"]
sub["base_bmi"] = sub["base_weight"] / ((sub["base_height"] / 100) ** 2)



## === cell 9
smoke_arr_test = oh1.transform(sub[["SmokingStatus"]])
smoke_cat_test = pd.DataFrame(smoke_arr_test, columns=smoke_cols, index=sub.index)
sub = pd.concat([sub, smoke_cat_test], axis=1)

for col in ["smoking cat 0", "smoking cat 1", "smoking cat 2"]:
    if col not in sub.columns:
        sub[col] = 0.0
for col in ["smoking cat 0", "smoking cat 1", "smoking cat 2"]:
    if col not in train_csv.columns:
        train_csv[col] = 0.0



## === cell 10
sub



## === cell 11
x = np.array(
    train_csv[
        [
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
        ]
    ],
    dtype=np.float32,
)
y = np.array(train_csv[["FVC", "confidence"]], dtype=np.float32)

from sklearn.model_selection import train_test_split

xtrain, xvalid, ytrain, yvalid = train_test_split(x, y, test_size=0.2, random_state=42)




## === cell 12
def metric(actual_fvc, predicted_fvc, confidence, return_values=False):
    """
    Calculates the modified Laplace Log Likelihood score for this competition.
    """
    sd_clipped = np.maximum(confidence, 70)
    delta = np.minimum(np.abs(actual_fvc - predicted_fvc), 1000)
    m = -np.sqrt(2) * delta / sd_clipped - np.log(np.sqrt(2) * sd_clipped)
    if return_values:
        return m
    else:
        return np.mean(m)


def model_loss(
    ytrue, ypred
):  # this loss penalises both prediction and confidence value
    fvc_pred = ypred[:, 0]
    sigmas = tf.maximum(
        ypred[:, 1], 70.0
    )  # avoid log(0) + match metric clipping lower bound
    ans = tf.math.log(sigmas)
    ans = ans + ((ytrue[:, 0] - fvc_pred) ** 2) / (2.0 * sigmas**2)
    return tf.reduce_mean(ans)




## === cell 13
class best_weights(tf.keras.callbacks.Callback):
    def __init__(self):
        self.metric_op = -30.0
        self.weights_op = None
        self.epoch_op = -1

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if logs.get("val_metric", -np.inf) >= self.metric_op:
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
        self.metric = metric
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xvalid = xvalid
        self.yvalid = yvalid

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_preds = self.model.predict(self.xvalid, verbose=0)
        logs["val_metric"] = self.metric(
            self.yvalid[:, 0], val_preds[:, 0], val_preds[:, 1]
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
    print(model.summary())
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
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.ylim(-10, 10)
    plt.grid(True)
    plt.show()

    return model




## === cell 14
model = run_model(xtrain, ytrain, xvalid, yvalid, epoch=1000)



## === cell 15
xtest = np.array(
    sub[
        [
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
        ]
    ],
    dtype=np.float32,
)

yans = model.predict(xtest, verbose=0)

sub["FVC"] = yans[:, 0]
sub["Confidence"] = np.abs(yans[:, 1])

sub = sub[["Patient_Week", "FVC", "Confidence"]]



## === cell 16
sub



## === cell 17
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
