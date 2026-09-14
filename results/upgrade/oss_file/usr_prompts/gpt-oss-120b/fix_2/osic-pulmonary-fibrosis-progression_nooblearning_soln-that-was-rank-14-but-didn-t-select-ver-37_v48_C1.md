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

-6.9076

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pydicom
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from skimage import morphology, measure
from skimage.transform import resize
import tensorflow as tf
from sklearn.cluster import KMeans
import matplotlib.patches as patches
import tensorflow.keras.backend as k



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
base_week_list = [base_week[pid] for pid in train_csv["Patient"]]
train_csv["base_week"] = base_week_list

count_from_base_week = train_csv["Weeks"] - train_csv["Patient"].map(base_week)
train_csv["count_from_base_week"] = count_from_base_week

train_csv["confidence"] = np.zeros(train_csv.shape[0])

base_fvc_dict = {
    pid: train_csv[
        (train_csv["Patient"] == pid) & (train_csv["Weeks"] == base_week[pid])
    ]["FVC"].values[0]
    for pid in train_csv["Patient"].unique()
}
train_csv["base_fvc"] = train_csv["Patient"].map(base_fvc_dict)

base_fev1_dict = {}
for pid in train_csv["Patient"].unique():
    A = base_fvc_dict[pid]
    B = train_csv[train_csv["Patient"] == pid]["Age"].unique()[0]
    if train_csv[train_csv["Patient"] == pid]["Sex"].unique()[0] == "Male":
        base_fev1_dict[pid] = 0.77 * A + 0.32 + 0.0069 * B
    else:
        base_fev1_dict[pid] = 0.77 * A + 0.28 + 0.0052 * B
train_csv["base_fev1"] = train_csv["Patient"].map(base_fev1_dict)

base_week_percent_dict = {
    pid: train_csv[
        (train_csv["Patient"] == pid) & (train_csv["Weeks"] == base_week[pid])
    ]["Percent"].values[0]
    for pid in train_csv["Patient"].unique()
}
train_csv["base_week_percent"] = train_csv["Patient"].map(base_week_percent_dict)

train_csv["base fev1/base fvc"] = train_csv["base_fev1"] / train_csv["base_fvc"]
train_csv["base_height"] = (train_csv["base_fvc"] + 9030) / 77.0

base_weight_dict = {}
for pid in train_csv["Patient"].unique():
    FVC = base_fvc_dict[pid]
    A = train_csv[train_csv["Patient"] == pid]["Age"].unique()[0]
    H = train_csv[train_csv["Patient"] == pid]["base_height"].unique()[0]
    if train_csv[train_csv["Patient"] == pid]["Sex"].unique()[0] == "Male":
        base_weight_dict[pid] = (FVC + 5458 - 49 * H + 8 * A) / 12.0
    else:
        base_weight_dict[pid] = (FVC + 3863 - 37 * H + 6 * A) / 14.0
train_csv["base_weight"] = train_csv["Patient"].map(base_weight_dict)

train_csv["base_bmi"] = train_csv["base_weight"] / (
    (train_csv["base_height"] / 100) ** 2
)



## === cell 4
from sklearn.preprocessing import LabelEncoder

lb_sex = LabelEncoder()
train_csv["Sex"] = lb_sex.fit_transform(train_csv["Sex"])
lb_smoke = LabelEncoder()
train_csv["SmokingStatus"] = lb_smoke.fit_transform(train_csv["SmokingStatus"])



## === cell 5
from sklearn.preprocessing import OneHotEncoder

oh1 = OneHotEncoder(handle_unknown="ignore", sparse=False)
smoke_cat = pd.DataFrame(
    oh1.fit_transform(train_csv[["SmokingStatus"]]),
    columns=[f"smoking cat {i}" for i in range(oh1.categories_[0].size)],
)
train_csv = pd.concat([train_csv, smoke_cat], axis=1)



## === cell 6
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
numeric_cols = [
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
train_scaled_vals = sc.fit_transform(train_csv[numeric_cols])
train_scaled = pd.DataFrame(train_scaled_vals, columns=numeric_cols)
train_scaled["Sex"] = train_csv["Sex"]
train_scaled["smoking cat 0"] = train_csv["smoking cat 0"]
train_scaled["smoking cat 1"] = train_csv["smoking cat 1"]



## === cell 7
train_scaled



## === cell 8
train_csv



## === cell 9
sub = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/sample_submission.csv")
test_csv = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")

sub["Patient"] = sub["Patient_Week"].apply(lambda x: x.split("_")[0])
sub["Weeks"] = sub["Patient_Week"].apply(lambda x: int(x.split("_")[-1]))
sub.drop(["FVC", "Confidence"], axis=1, inplace=True)

base_fvc = test_csv.groupby("Patient")["FVC"].min()
sub["base_fvc"] = sub["Patient"].map(base_fvc)

base_fev1_dict_test = {}
for pid in sub["Patient"].unique():
    A = sub[sub["Patient"] == pid]["base_fvc"].iloc[0]
    B = test_csv[test_csv["Patient"] == pid]["Age"].iloc[0]
    sex = test_csv[test_csv["Patient"] == pid]["Sex"].iloc[0]
    if sex == "Male":
        base_fev1_dict_test[pid] = 0.77 * A + 0.32 + 0.0069 * B
    else:
        base_fev1_dict_test[pid] = 0.77 * A + 0.28 + 0.0052 * B
sub["base_fev1"] = sub["Patient"].map(base_fev1_dict_test)

sub["base_height"] = (sub["base_fvc"] + 9030) / 77.0

base_weight_dict_test = {}
for pid in sub["Patient"].unique():
    FVC = sub[sub["Patient"] == pid]["base_fvc"].iloc[0]
    A = test_csv[test_csv["Patient"] == pid]["Age"].iloc[0]
    H = sub[sub["Patient"] == pid]["base_height"].iloc[0]
    sex = test_csv[test_csv["Patient"] == pid]["Sex"].iloc[0]
    if sex == "Male":
        base_weight_dict_test[pid] = (FVC + 5458 - 49 * H + 8 * A) / 12.0
    else:
        base_weight_dict_test[pid] = (FVC + 3863 - 37 * H + 6 * A) / 14.0
sub["base_weight"] = sub["Patient"].map(base_weight_dict_test)

sub["Sex"] = lb_sex.transform(test_csv.set_index("Patient").loc[sub["Patient"], "Sex"])

sub["SmokingStatus"] = (
    test_csv.set_index("Patient").loc[sub["Patient"], "SmokingStatus"].values
)

sub["Age"] = test_csv.set_index("Patient").loc[sub["Patient"], "Age"].values
sub["Percent"] = test_csv.set_index("Patient").loc[sub["Patient"], "Percent"].values

base_week_test = test_csv.groupby("Patient")["Weeks"].min()
sub["base_week"] = sub["Patient"].map(base_week_test)
sub["count_from_base_week"] = sub["Weeks"] - sub["base_week"]

sub["base fev1/base fvc"] = sub["base_fev1"] / sub["base_fvc"]
sub["base_bmi"] = sub["base_weight"] / ((sub["base_height"] / 100.0) ** 2)



## === cell 10
smoke_cat_test = pd.DataFrame(
    oh1.transform(sub[["SmokingStatus"]]),
    columns=[f"smoking cat {i}" for i in range(oh1.categories_[0].size)],
)
sub = pd.concat([sub, smoke_cat_test], axis=1)



## === cell 11
sub_scaled_vals = sc.transform(sub[numeric_cols])
sub_scaled = pd.DataFrame(sub_scaled_vals, columns=numeric_cols)
sub_scaled["Sex"] = sub["Sex"]
sub_scaled["smoking cat 0"] = sub["smoking cat 0"]
sub_scaled["smoking cat 1"] = sub["smoking cat 1"]



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/3265709812.py in <cell line: 0>()
----> 1 sub_scaled_vals = sc.transform(sub[numeric_cols])
      2 sub_scaled = pd.DataFrame(sub_scaled_vals, columns=numeric_cols)
      3 sub_scaled["Sex"] = sub["Sex"]
      4 sub_scaled["smoking cat 0"] = sub["smoking cat 0"]
      5 sub_scaled["smoking cat 1"] = sub["smoking cat 1"]

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['base_week_percent'] not in index"

## === cell 12
sub_scaled



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/883724888.py in <cell line: 0>()
----> 1 sub_scaled
      2 

NameError: name 'sub_scaled' is not defined

## === cell 13
x = np.array(
    train_scaled[
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
    ]
)
y = np.array(train_csv[["FVC", "confidence"]])

from sklearn.model_selection import train_test_split

xtrain, xvalid, ytrain, yvalid = train_test_split(x, y, test_size=0.2, random_state=42)




## === cell 14
def metric(actual_fvc, predicted_fvc, confidence, return_values=False):
    sd_clipped = np.maximum(confidence, 70)
    delta = np.minimum(np.abs(actual_fvc - predicted_fvc), 1000)
    metric = -np.sqrt(2) * delta / sd_clipped - np.log(np.sqrt(2) * sd_clipped)
    return np.mean(metric) if not return_values else metric


def model_loss(ytrue, ypred):
    eps = 1.0
    fvc_pred = ypred[:, 0]
    sigmas = ypred[:, 1] + eps
    ans = tf.math.log(sigmas) + ((ytrue[:, 0] - fvc_pred) ** 2) / (2 * sigmas**2)
    return tf.reduce_mean(ans)




## === cell 15
C1 = tf.constant(70, dtype=tf.float32)
C2 = tf.constant(1000, dtype=tf.float32)


def score(y_true, y_pred):
    sigma = y_pred[:, 1]
    fvc_pred = y_pred[:, 0]
    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true[:, 0] - fvc_pred)
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt(tf.constant(2.0, dtype=tf.float32))
    metric = (delta / sigma_clip) * sq2 + tf.math.log(sigma_clip * sq2)
    return tf.reduce_mean(metric)


def huber_loss(y_true, y_pred):
    error = y_true[:, 0] - y_pred[:, 0]
    is_small = tf.abs(error) <= C2
    quad = tf.square(error) / 2.0
    linear = C2 * tf.abs(error) - tf.square(C2) / 2.0
    return tf.reduce_mean(tf.where(is_small, quad, linear))


def custom_loss(y_true, y_pred):
    return huber_loss(y_true, y_pred) + score(y_true, y_pred)




## === cell 16
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    factor=0.2, monitor="val_loss", mode="min", patience=150, verbose=0
)


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
            f"BEST_EPOCH = {self.epoch_op+1}   BEST_SCORE_ON_VALID_SET = {self.metric_op}"
        )


class metrics_call(tf.keras.callbacks.Callback):
    def __init__(self, metric, xtrain, ytrain, xvalid, yvalid):
        super().__init__()
        self.metric = metric
        self.xvalid = xvalid
        self.yvalid = yvalid

    def on_epoch_end(self, epoch, logs=None):
        val_preds = self.model.predict(self.xvalid, verbose=0)
        logs = logs or {}
        logs["val_metric"] = self.metric(
            self.yvalid[:, 0], val_preds[:, 0], val_preds[:, 1]
        )


def run_model(xtrain, ytrain, xvalid, yvalid, epoch=50):
    inp = tf.keras.layers.Input(shape=xtrain.shape[1:])
    noisy = tf.keras.layers.GaussianNoise(0.6)(inp)

    def branch(x):
        d = tf.keras.layers.Dense(128, activation="relu")(x)
        d = tf.keras.layers.Dense(128, activation="relu")(d)
        d = tf.keras.layers.Dense(128, activation="relu")(d)
        mean = tf.keras.layers.Dense(1)(d)
        std = tf.keras.layers.Dense(1)(d)
        return mean, std

    m1, s1 = branch(noisy)
    m2, s2 = branch(noisy)
    m3, s3 = branch(noisy)

    mean_comb = tf.keras.layers.Concatenate()([m1, m2, m3])
    std_comb = tf.keras.layers.Concatenate()([s1, s2, s3])

    mean_final = tf.keras.layers.Dense(1)(mean_comb)
    std_final_den = tf.keras.layers.Dense(1)(std_comb)
    std_final = tf.keras.layers.Lambda(lambda x: tf.abs(x))(std_final_den)

    output = tf.keras.layers.Concatenate()([mean_final, std_final])

    model = tf.keras.models.Model(inputs=inp, outputs=output)
    model.compile(
        loss=lambda yt, yp: custom_loss(yt, yp),
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
            lr_scheduler,
            metrics_call(metric, xtrain, ytrain, xvalid, yvalid),
            best_weights(),
        ],
    )
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.ylim(-10, 10)
    plt.grid(True)
    return model




## === cell 17
model = run_model(xtrain, ytrain, xvalid, yvalid, epoch=200)



## === cell 18
xtest = np.array(
    sub_scaled[
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
    ]
)
yans = model.predict(xtest, verbose=0)

sub["FVC"] = yans[:, 0]
sub["Confidence"] = yans[:, 1]

cols_to_drop = [
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
    "base_weight",
    "base_bmi",
    "smoking cat 0",
    "smoking cat 1",
    "smoking cat 2",
]
sub.drop(columns=cols_to_drop, inplace=True)



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2465796052.py in <cell line: 0>()
      1 xtest = np.array(
----> 2     sub_scaled[
      3         [
      4             "Weeks",
      5             "Age",

NameError: name 'sub_scaled' is not defined

## === cell 19
sub.head()



## === cell 20
sub.to_csv("submission.csv", index=False)

## --- ERROR in outputing the csv:
Invalid submission: Submission DataFrame must have a 'FVC' column.
