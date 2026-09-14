# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
pillow==11.3.0
protobuf==6.33.0
pydicom==3.0.1
scikit-image==0.25.2
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

# 5. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

import pydicom
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import mean_absolute_error

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    AveragePooling2D,
    Flatten,
    Dropout,
    Dense,
    concatenate,
    Input,
)



## === cell 1
ROOT = "../input/osic-pulmonary-fibrosis-progression"
DESIRED_SIZE = 128



## === cell 2
tr = pd.read_csv(f"{ROOT}/train.csv")
tr.drop_duplicates(keep=False, inplace=True, subset=["Patient", "Weeks"])
chunk = pd.read_csv(f"{ROOT}/test.csv")

print("add infos")
sub = pd.read_csv(f"{ROOT}/sample_submission.csv")

pw = sub["Patient_Week"].str.rsplit("_", n=1, expand=True)
sub["Patient"] = pw[0]
sub["Weeks"] = pw[1].astype(np.int32)

sub = sub[["Patient", "Weeks", "Confidence", "Patient_Week"]]
sub = sub.merge(chunk.drop("Weeks", axis=1), on="Patient")



## === cell 3
tr["WHERE"] = "train"
chunk["WHERE"] = "val"
sub["WHERE"] = "test"
data = pd.concat([tr, chunk, sub], axis=0, ignore_index=True)



## === cell 4
data["min_week"] = data["Weeks"]
data.loc[data.WHERE == "test", "min_week"] = np.nan
data["min_week"] = data.groupby("Patient")["min_week"].transform("min")



## === cell 5
base = data.loc[data.Weeks == data.min_week]
base = base[["Patient", "FVC"]].copy()
base.columns = ["Patient", "min_FVC"]
base["nb"] = 1
base["nb"] = base.groupby("Patient")["nb"].transform("cumsum")
base = base[base.nb == 1]
base.drop("nb", axis=1, inplace=True)



## === cell 6
data = data.merge(base, on="Patient", how="left")
data["base_week"] = data["Weeks"] - data["min_week"]
del base



## === cell 7
COLS = ["Sex", "SmokingStatus"]
FE = []
for col in COLS:
    for mod in data[col].dropna().unique():
        FE.append(mod)
        data[mod] = (data[col] == mod).astype(int)



## === cell 8
data["age"] = (data["Age"] - data["Age"].min()) / (
    data["Age"].max() - data["Age"].min()
)
data["BASE"] = (data["min_FVC"] - data["min_FVC"].min()) / (
    data["min_FVC"].max() - data["min_FVC"].min()
)
data["week"] = (data["base_week"] - data["base_week"].min()) / (
    data["base_week"].max() - data["base_week"].min()
)
data["percent"] = (data["Percent"] - data["Percent"].min()) / (
    data["Percent"].max() - data["Percent"].min()
)
FE += ["age", "percent", "week", "BASE"]



## === cell 9
tr = data.loc[data.WHERE == "train"].copy()
chunk = data.loc[data.WHERE == "val"].copy()
sub = data.loc[data.WHERE == "test"].copy()
del data



## === cell 10
tr.shape, chunk.shape, sub.shape, len(FE)



## === cell 11
_patient_mid_dcm_cache = {}


def _get_patient_dicom_path(patient_folder):
    cached = _patient_mid_dcm_cache.get(patient_folder, None)
    if cached is not None:
        return cached

    if not os.path.isdir(patient_folder):
        _patient_mid_dcm_cache[patient_folder] = None
        return None

    dcm_paths = []
    with os.scandir(patient_folder) as it:
        for entry in it:
            if entry.is_file() and entry.name.lower().endswith(".dcm"):
                dcm_paths.append(entry.path)

    if not dcm_paths:
        _patient_mid_dcm_cache[patient_folder] = None
        return None

    dss = []
    for pth in dcm_paths:
        try:
            ds = pydicom.dcmread(
                pth, stop_before_pixels=True, specific_tags=["InstanceNumber"]
            )
            inst = getattr(ds, "InstanceNumber", None)
            dss.append((inst if inst is not None else 10**9, pth))
        except Exception:
            continue

    if not dss:
        _patient_mid_dcm_cache[patient_folder] = None
        return None

    dss.sort(key=lambda x: x[0])
    mid = dss[len(dss) // 2][1]
    _patient_mid_dcm_cache[patient_folder] = mid
    return mid


def get_images(df, how="train"):
    xo = []
    p = []
    w = []

    patients = df["Patient"].to_numpy()
    weeks = df["Weeks"].to_numpy()

    for patient, week in tqdm(list(zip(patients, weeks)), total=len(df)):
        try:
            patient_folder = f"{ROOT}/{how}/{patient}"
            img_path = _get_patient_dicom_path(patient_folder)
            if img_path is None:
                continue

            ds = pydicom.dcmread(img_path)
            im = ds.pixel_array
            if im.ndim != 2:
                im = np.squeeze(im)
                if im.ndim != 2:
                    continue

            im = Image.fromarray(im.astype(np.int16))
            im = im.resize((DESIRED_SIZE, DESIRED_SIZE))
            im = np.asarray(im)

            xo.append(im[np.newaxis, :, :])
            p.append(patient)
            w.append(week)
        except Exception:
            continue

    data_out = pd.DataFrame({"Patient": p, "Weeks": w})
    if len(xo) == 0:
        return np.zeros((0, DESIRED_SIZE, DESIRED_SIZE), dtype=np.float32), data_out
    return np.concatenate(xo, axis=0), data_out




## === cell 12
x, df_tr = get_images(tr, how="train")



## === cell 13
x.shape, df_tr.shape



## === cell 14
if x.shape[0] > 0:
    idx = np.random.randint(x.shape[0])
    plt.imshow(x[idx], cmap=plt.cm.bone)
    plt.show()



## === cell 15
df_tr = df_tr.merge(tr, how="left", on=["Patient", "Weeks"])



## === cell 16
y = df_tr["FVC"].values.astype(np.float32)
z = df_tr[FE].values.astype(np.float32)



## === cell 17
z.shape, y.shape



## === cell 18
y2 = y.reshape(-1, 1)



## === cell 19
C1, C2 = tf.constant(70, dtype="float32"), tf.constant(1000, dtype="float32")


@tf.function
def kloss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    sigma = y_pred[:, 1]
    fvc_pred = y_pred[:, 0]

    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true[:, 0] - fvc_pred)
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt(tf.cast(2.0, dtype=tf.float32))
    metric = (delta / sigma_clip) * sq2 + tf.math.log(sigma_clip * sq2)
    return K.mean(metric)


@tf.function
def kmae(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    spread = tf.abs((y_true[:, 0] - y_pred[:, 0]) / (y_true[:, 0] + 1e-6))
    return K.mean(spread)


def mloss(_lambda):
    @tf.function
    def loss(y_true, y_pred):
        return _lambda * kloss(y_true, y_pred) + (1 - _lambda) * kmae(y_true, y_pred)

    return loss




## === cell 20
def VGG19(x):
    x = Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same",
        name="block1_conv1",
        kernel_initializer="he_normal",
    )(x)
    x = Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same",
        name="block1_conv2",
        kernel_initializer="he_normal",
    )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")(x)

    x = Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same",
        name="block2_conv1",
        kernel_initializer="he_normal",
    )(x)
    x = Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same",
        name="block2_conv2",
        kernel_initializer="he_normal",
    )(x)
    return x




## === cell 21
LRN2D_NORM = True
DATA_FORMAT = "channels_last"
USE_BN = True
DROPOUT = 0.25


def conv2D_lrn2d(
    x,
    filters,
    kernel_size,
    strides=(1, 1),
    padding="same",
    dilation_rate=(1, 1),
    activation="relu",
    use_bias=True,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    lrn2d_norm=LRN2D_NORM,
    weight_decay=None,
):
    if weight_decay:
        kernel_regularizer = tf.keras.regularizers.L1L2(
            l1=weight_decay, l2=weight_decay
        )
        bias_regularizer = tf.keras.regularizers.L1L2(l1=weight_decay, l2=weight_decay)
    else:
        kernel_regularizer = None
        bias_regularizer = None
    x = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
    )(x)
    if lrn2d_norm:
        x = BatchNormalization()(x)
    return x


def inception_module(
    x,
    params,
    concat_axis,
    padding="same",
    dilation_rate=(1, 1),
    activation="relu",
    use_bias=True,
    kernel_initializer="he_normal",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    lrn2d_norm=LRN2D_NORM,
    weight_decay=None,
):
    (branch1, branch2, branch3, branch4) = params
    if weight_decay:
        kernel_regularizer = tf.keras.regularizers.L1L2(
            l1=weight_decay, l2=weight_decay
        )
        bias_regularizer = tf.keras.regularizers.L1L2(l1=weight_decay, l2=weight_decay)
    else:
        kernel_regularizer = None
        bias_regularizer = None
    pathway1 = Conv2D(
        filters=branch1[0],
        kernel_size=(1, 1),
        strides=1,
        padding=padding,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
    )(x)
    pathway2 = Conv2D(
        filters=branch2[0],
        kernel_size=(1, 1),
        strides=1,
        padding=padding,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
    )(x)
    pathway2 = Conv2D(
        filters=branch2[1],
        kernel_size=(3, 3),
        strides=1,
        padding=padding,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
    )(pathway2)
    pathway3 = Conv2D(
        filters=branch3[0],
        kernel_size=(1, 1),
        strides=1,
        padding=padding,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
    )(x)
    pathway3 = Conv2D(
        filters=branch3[1],
        kernel_size=(5, 5),
        strides=1,
        padding=padding,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
    )(pathway3)
    pathway4 = MaxPooling2D(
        pool_size=(3, 3), strides=1, padding=padding, data_format=DATA_FORMAT
    )(x)
    pathway4 = Conv2D(
        filters=branch4[0],
        kernel_size=(1, 1),
        strides=1,
        padding=padding,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
    )(pathway4)
    return concatenate([pathway1, pathway2, pathway3, pathway4], axis=concat_axis)




## === cell 22
CONCAT_AXIS = 3


def make_model(n_features):
    inp = Input((DESIRED_SIZE, DESIRED_SIZE, 1), name="input")
    z = Input((n_features,), name="Patient")

    x = VGG19(inp)
    x = conv2D_lrn2d(x, 64, (7, 7), 2, padding="same", lrn2d_norm=False)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding="same")(x)
    x = BatchNormalization()(x)

    x = conv2D_lrn2d(x, 64, (1, 1), 1, padding="same", lrn2d_norm=False)

    x = conv2D_lrn2d(x, 192, (3, 3), 1, padding="same", lrn2d_norm=True)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding="same")(x)

    x = inception_module(
        x, params=[(64,), (96, 128), (16, 32), (32,)], concat_axis=CONCAT_AXIS
    )
    x = inception_module(
        x, params=[(128,), (128, 192), (32, 96), (64,)], concat_axis=CONCAT_AXIS
    )
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding="same")(x)

    x = inception_module(
        x, params=[(192,), (96, 208), (16, 48), (64,)], concat_axis=CONCAT_AXIS
    )
    x = inception_module(
        x, params=[(160,), (112, 224), (24, 64), (64,)], concat_axis=CONCAT_AXIS
    )
    x = inception_module(
        x, params=[(128,), (128, 256), (24, 64), (64,)], concat_axis=CONCAT_AXIS
    )
    x = inception_module(
        x, params=[(112,), (144, 288), (32, 64), (64,)], concat_axis=CONCAT_AXIS
    )
    x = inception_module(
        x, params=[(256,), (160, 320), (32, 128), (128,)], concat_axis=CONCAT_AXIS
    )
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding="same")(x)

    x = inception_module(
        x, params=[(256,), (160, 320), (32, 128), (128,)], concat_axis=CONCAT_AXIS
    )
    x = inception_module(
        x, params=[(384,), (192, 384), (48, 128), (128,)], concat_axis=CONCAT_AXIS
    )
    x = AveragePooling2D(pool_size=(1, 1), strides=1, padding="valid")(x)

    x = Flatten()(x)
    x = Dropout(DROPOUT)(x)
    preds = Dense(2, activation="relu", name="preds")(x)

    model = tf.keras.Model([inp, z], preds, name="CNN")
    model.compile(loss=mloss(0.5), optimizer="adam", metrics=[kloss])
    return model




## === cell 23
net = make_model(n_features=z.shape[1])
print(net.summary())



## === cell 24
x = x.astype(np.float32)
x_min = np.min(x) if x.size else 0.0
x_max = np.max(x) if x.size else 1.0
den = (x_max - x_min) if (x_max - x_min) != 0 else 1.0
xs = (x - x_min) / den



## === cell 25
xs.shape, y2.shape, x_min



## === cell 26
xs = np.expand_dims(xs, axis=3)
xs.shape



## === cell 27
if xs.shape[0] > 0:
    net.fit([xs, z], y2, batch_size=32, epochs=120, verbose=2)



## === cell 28
if xs.shape[0] > 0:
    pred = net.predict([xs, z], batch_size=100, verbose=1)
else:
    pred = np.zeros((0, 2), dtype=np.float32)



## === cell 29
if pred.shape[0] > 0:
    sigma_opt = mean_absolute_error(y, pred[:, 0])
    sigma_mean = np.mean(pred[:, 1])
else:
    sigma_opt = 200.0
    sigma_mean = 200.0
print(sigma_opt, sigma_mean)



## === cell 30
if pred.shape[0] > 0:
    plt.plot(y)
    plt.plot(pred[:, 0])
    plt.show()



## === cell 31
if pred.shape[0] > 0:
    pred[:, 1].min(), pred[:, 1].max()
else:
    (0.0, 0.0)



## === cell 32
if pred.shape[0] > 0:
    plt.hist(pred[:, 1])
    plt.title("uncertainty in prediction")
    plt.show()



## === cell 33
xe, df_te = get_images(sub, how="test")
df_te = df_te.merge(sub, how="left", on=["Patient", "Weeks"])



## === cell 34
xe = xe.astype(np.float32)
den = (x_max - x_min) if (x_max - x_min) != 0 else 1.0
x_te = (xe - x_min) / den
x_te = np.expand_dims(x_te, axis=3)
ze = df_te[FE].values.astype(np.float32)
if x_te.shape[0] > 0:
    pe = net.predict([x_te, ze], batch_size=100, verbose=1)
else:
    pe = np.zeros((0, 2), dtype=np.float32)



## === cell 35
df_te["FVC1"] = pe[:, 0] if pe.shape[0] else np.nan
df_te["Confidence1"] = pe[:, 1] if pe.shape[0] else np.nan



## === cell 36
sub = sub.merge(
    df_te[["Patient", "Weeks", "FVC1", "Confidence1"]],
    how="left",
    on=["Patient", "Weeks"],
)



## === cell 37
sub.head()



## === cell 38
subm = sub[["Patient_Week", "FVC", "Confidence", "FVC1", "Confidence1"]].copy()



## === cell 39
subm.loc[~subm.FVC1.isnull()].head(10)



## === cell 40
subm.loc[~subm.FVC1.isnull(), "FVC"] = subm.loc[~subm.FVC1.isnull(), "FVC1"]
if sigma_mean < 70:
    subm["Confidence"] = sigma_opt
else:
    subm.loc[~subm.FVC1.isnull(), "Confidence"] = subm.loc[
        ~subm.FVC1.isnull(), "Confidence1"
    ]
    subm.loc[subm.FVC1.isnull(), "Confidence"] = sigma_opt

subm["Confidence"] = (
    pd.to_numeric(subm["Confidence"], errors="coerce").fillna(sigma_opt).clip(lower=70)
)
subm["FVC"] = pd.to_numeric(subm["FVC"], errors="coerce").fillna(subm["FVC"].median())



## === cell 41
subm.head()



## === cell 42
subm.describe().T



## === cell 43
subm[["Patient_Week", "FVC", "Confidence"]].to_csv("submission.csv", index=False)
print(
    "Wrote submission.csv with shape:",
    subm[["Patient_Week", "FVC", "Confidence"]].shape,
)
print(subm[["Patient_Week", "FVC", "Confidence"]].head())
