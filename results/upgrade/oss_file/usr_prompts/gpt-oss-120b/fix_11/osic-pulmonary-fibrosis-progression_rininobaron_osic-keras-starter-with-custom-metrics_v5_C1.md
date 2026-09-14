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

# 5. Target score

-8.7484

# 6. Current score

-24.65932

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -24.65932) has done: 'The fix adds a protobuf compatibility flag before importing TensorFlow to prevent the `MessageFactory` error, and corrects the final merge step so the required `FVC` and `Confidence` columns are always present before creating the submission file.'
- What this solution (achieved -24.65932) has done: 'I fix the image‑loading routine so that every row gets an image (real one when the DICOM file exists, otherwise a zero‑filled placeholder). This prevents rows from being dropped, keeps the training data aligned, and lets the model learn from the patient features even when images are missing, which should raise the validation score toward the target. The rest of the pipeline remains unchanged.'
- What this solution (achieved -24.65932) has done: 'I increase the loss weighting toward the competition‐specific Laplace loss (kloss) by changing the lambda from 0.5 to 0.9 when compiling the model. This small tweak keeps the overall architecture unchanged while directing training more toward the metric we need to improve, helping move the score toward the target.'
- What this solution (achieved -24.65932) has done: 'The fix moves the protobuf compatibility flag to the very top, adjusts the loss‑weighting to focus more on the competition‑specific Laplace loss (λ = 0.95) and trains a bit longer (200 epochs) to let the model better capture the data while keeping the original architecture unchanged.'
- What this solution (achieved -24.65932) has done: 'The fix moves the protobuf compatibility flag to the very first cell so TensorFlow can import without error, adds a small adjustment to the loss weighting (λ = 0.9) to better match the competition metric, and replaces the model‑predicted confidence with a constant confidence derived from the training MAE (clipped at 70). This keeps the original architecture unchanged, ensures a valid `submission.csv` is written, and nudges the score toward the target.'
- What this solution (achieved -24.65932) has done: 'The fix moves the protobuf‐compatibility flag into the first import cell (so it’s set before any library loads) and changes the confidence prediction to use the model’s own sigma output (clipped at 70) instead of a constant derived from MAE. This resolves the TensorFlow import error and provides a more realistic confidence, nudging the validation score toward the target while keeping the original model architecture unchanged.'
- What this solution (achieved -24.65932) has done: 'I fix the loss function by returning the negative Laplace metric (so training maximizes the competition score instead of minimizing it) and increase its weight slightly to better match the target metric. I also replace the model‑predicted confidence with the constant confidence derived from the training MAE, which is more stable. These changes keep the overall architecture unchanged while addressing the scoring mis‑alignment and confidence handling.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pydicom
from PIL import Image
import tensorflow as tf
from tensorflow.keras import backend as K
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
from tensorflow.keras.models import Model
from sklearn.metrics import mean_absolute_error




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
ROOT = "../input/osic-pulmonary-fibrosis-progression"
DESIRED_SIZE = 128




## === cell 2
tr = pd.read_csv(f"{ROOT}/train.csv")
tr.drop_duplicates(keep="first", inplace=True, subset=["Patient", "Weeks"])
chunk = pd.read_csv(f"{ROOT}/test.csv")

sub = pd.read_csv(f"{ROOT}/sample_submission.csv")
sub["Patient"] = sub["Patient_Week"].apply(lambda x: x.split("_")[0])
sub["Weeks"] = sub["Patient_Week"].apply(lambda x: int(x.split("_")[-1]))
sub = sub[["Patient", "Weeks", "Confidence", "Patient_Week"]]
sub = sub.merge(chunk.drop("Weeks", axis=1), on="Patient")
sub = sub.drop(columns=["FVC"], errors="ignore")




## === cell 3
tr["WHERE"] = "train"
chunk["WHERE"] = "val"
sub["WHERE"] = "test"
data = pd.concat([tr, chunk, sub], ignore_index=True)




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




## === cell 7
COLS = ["Sex", "SmokingStatus"]
FE = []
for col in COLS:
    for mod in data[col].unique():
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
tr = data.loc[data.WHERE == "train"]
chunk = data.loc[data.WHERE == "val"]
sub = data.loc[data.WHERE == "test"]
del data




## === cell 10
print(tr.shape, chunk.shape, sub.shape)




## === cell 11
def get_images(df, how="train"):
    imgs = []
    patients = []
    weeks = []
    zero_image = np.zeros((DESIRED_SIZE, DESIRED_SIZE), dtype=np.uint8)
    for _, row in tqdm(df.iterrows(), total=len(df)):
        patient = row["Patient"]
        week = row["Weeks"]
        img_path = f"{ROOT}/{how}/{patient}/{week}.dcm"
        try:
            ds = pydicom.dcmread(img_path)
            im = Image.fromarray(ds.pixel_array)
            im = im.resize((DESIRED_SIZE, DESIRED_SIZE))
            im = np.array(im)
        except Exception:
            im = zero_image.copy()
        if im.shape != (DESIRED_SIZE, DESIRED_SIZE):
            im = Image.fromarray(im).resize((DESIRED_SIZE, DESIRED_SIZE))
            im = np.array(im)
        imgs.append(im[np.newaxis, :, :])
        patients.append(patient)
        weeks.append(week)
    if len(imgs) == 0:
        return np.empty((0, DESIRED_SIZE, DESIRED_SIZE)), pd.DataFrame(
            columns=["Patient", "Weeks"]
        )
    imgs = np.concatenate(imgs, axis=0)
    meta = pd.DataFrame({"Patient": patients, "Weeks": weeks})
    return imgs, meta




## === cell 12
x, df_tr = get_images(tr, how="train")




## === cell 13
print("Train images shape:", x.shape, "metadata shape:", df_tr.shape)




## === cell 14
df_tr = df_tr.merge(tr, how="left", on=["Patient", "Weeks"])
y = df_tr["FVC"].values
z = df_tr[FE].values




## === cell 15
C1 = tf.constant(70, dtype=tf.float32)
C2 = tf.constant(1000, dtype=tf.float32)


def kloss(y_true, y_pred):
    sigma = y_pred[:, 1]
    fvc_pred = y_pred[:, 0]
    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(tf.squeeze(y_true) - fvc_pred)
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt(tf.constant(2.0, dtype=tf.float32))
    metric = (delta / sigma_clip) * sq2 + tf.math.log(sigma_clip * sq2)
    return -K.mean(metric)


def kmae(y_true, y_pred):
    spread = tf.abs((tf.squeeze(y_true) - y_pred[:, 0]) / tf.squeeze(y_true))
    return K.mean(spread)


def mloss(_lambda):
    def loss(y_true, y_pred):
        return _lambda * kloss(y_true, y_pred) + (1 - _lambda) * kmae(y_true, y_pred)

    return loss




## === cell 16
def VGG19(x):
    x = Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        name="block1_conv1",
    )(x)
    x = Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        name="block1_conv2",
    )(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")(x)

    x = Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        name="block2_conv1",
    )(x)
    x = Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        name="block2_conv2",
    )(x)
    return x




## === cell 17
LRN2D_NORM = True
DATA_FORMAT = "channels_last"
USE_BN = True
DROPOUT = 0.25
CONCAT_AXIS = 3


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
    x = Conv2D(
        filters,
        kernel_size,
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
    pathway1 = Conv2D(
        branch1[0],
        (1, 1),
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
        branch2[0],
        (1, 1),
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
        branch2[1],
        (3, 3),
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
        branch3[0],
        (1, 1),
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
        branch3[1],
        (5, 5),
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
        branch4[0],
        (1, 1),
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




## === cell 18
def make_model():
    img_input = Input((DESIRED_SIZE, DESIRED_SIZE, 1), name="image")
    patient_input = Input((len(FE),), name="patient")
    x = VGG19(img_input)
    x = conv2D_lrn2d(x, 64, (7, 7), strides=2, padding="same", lrn2d_norm=False)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding="same")(x)
    x = BatchNormalization()(x)

    x = conv2D_lrn2d(x, 64, (1, 1), strides=1, padding="same", lrn2d_norm=False)
    x = conv2D_lrn2d(x, 192, (3, 3), strides=1, padding="same", lrn2d_norm=True)
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
    model = Model([img_input, patient_input], preds, name="CNN")
    model.compile(loss=mloss(0.95), optimizer="adam", metrics=[kloss])
    return model




## === cell 19
net = make_model()
net.summary()




## === cell 20
x_min = np.min(x)
x_max = np.max(x)
xs = (x - x_min) / (x_max - x_min)




## === cell 21
print(
    "Normalized train images shape:", xs.shape, "y shape:", y.shape, "z shape:", z.shape
)




## === cell 22
xs = np.expand_dims(xs, axis=3)




## === cell 23
net.fit([xs, z], y, batch_size=32, epochs=200, verbose=2)




## === cell 24
pred_train = net.predict([xs, z], batch_size=100, verbose=0)




## === cell 25
sigma_opt = mean_absolute_error(y, pred_train[:, 0])
sigma_const = max(sigma_opt, 70.0)
print("sigma_opt:", sigma_opt, "using constant Confidence:", sigma_const)




## === cell 26
xe, df_te = get_images(sub, how="test")
df_te = df_te.merge(sub, how="left", on=["Patient", "Weeks"])




## === cell 27
x_te = (xe - x_min) / (x_max - x_min)
x_te = np.expand_dims(x_te, axis=3)
z_te = df_te[FE].values
pred_test = net.predict([x_te, z_te], batch_size=100, verbose=0)




## === cell 28
df_te["FVC"] = pred_test[:, 0]
df_te["Confidence"] = sigma_const

sub_final = sub.merge(
    df_te[["Patient", "Weeks", "FVC", "Confidence"]],
    on=["Patient", "Weeks"],
    how="left",
)
if "FVC" not in sub_final.columns:
    sub_final["FVC"] = 0.0
if "Confidence" not in sub_final.columns:
    sub_final["Confidence"] = 70.0
sub_final["FVC"] = sub_final["FVC"].fillna(0)
sub_final["Confidence"] = sub_final["Confidence"].fillna(70)




## === cell 29
submission = sub_final[["Patient_Week", "FVC", "Confidence"]].copy()
submission.to_csv("submission.csv", index=False)
print("submission.csv written with", submission.shape[0], "rows")
