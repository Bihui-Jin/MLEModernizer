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

No external packages required in the script and installed.

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

-9.035029120532077

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import pydicom
from PIL import Image
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model, Input
import keras

if not hasattr(keras.backend, "sigmoid"):
    keras.backend.sigmoid = tf.nn.sigmoid

try:
    import tensorflow_addons as tfa
except Exception:

    class _FakeSWA:
        def __init__(self, optimizer):
            self.optimizer = optimizer

        def __call__(self, *args, **kwargs):
            return self.optimizer

    tfa = type("tfa", (), {"optimizers": type("opt", (), {"SWA": _FakeSWA})})



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
import os



## === cell 2
ROOT = "../input/osic-pulmonary-fibrosis-progression"
DESIRED_SIZE = 64



## === cell 3
tr = pd.read_csv(f"{ROOT}/train.csv")
tr.drop_duplicates(keep=False, inplace=True, subset=["Patient", "Weeks"])
chunk = pd.read_csv(f"{ROOT}/test.csv")

print("add infos")
sub = pd.read_csv(f"{ROOT}/sample_submission.csv")
sub["Patient"] = sub["Patient_Week"].apply(lambda x: x.split("_")[0])
sub["Weeks"] = sub["Patient_Week"].apply(lambda x: int(x.split("_")[-1]))
sub = sub[["Patient", "Weeks", "Confidence", "Patient_Week"]]
sub = sub.merge(chunk.drop("Weeks", axis=1), on="Patient")



## === cell 4
tr["WHERE"] = "train"
chunk["WHERE"] = "val"
sub["WHERE"] = "test"
data = pd.concat([tr, chunk, sub], ignore_index=True)



## === cell 5
data["min_week"] = data["Weeks"]
data.loc[data.WHERE == "test", "min_week"] = np.nan
data["min_week"] = data.groupby("Patient")["min_week"].transform("min")



## === cell 6
base = data.loc[data.Weeks == data.min_week]
base = base[["Patient", "FVC"]].copy()
base.columns = ["Patient", "min_FVC"]
base["nb"] = 1
base["nb"] = base.groupby("Patient")["nb"].transform("cumsum")
base = base[base.nb == 1]
base.drop("nb", axis=1, inplace=True)



## === cell 7
data = data.merge(base, on="Patient", how="left")
data["base_week"] = data["Weeks"] - data["min_week"]
del base



## === cell 8
COLS = ["Sex", "SmokingStatus"]
FE = []
for col in COLS:
    for mod in data[col].unique():
        FE.append(mod)
        data[mod] = (data[col] == mod).astype(int)



## === cell 9
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



## === cell 10
tr = data.loc[data.WHERE == "train"]
chunk = data.loc[data.WHERE == "val"]
sub = data.loc[data.WHERE == "test"]
del data



## === cell 11
print(tr.shape, chunk.shape, sub.shape)




## === cell 12
def get_images(df, how="train"):
    xo = []
    p = []
    w = []
    for i in tqdm(range(df.shape[0])):
        patient = df.iloc[i, 0]
        week = df.iloc[i, 1]
        try:
            img_path = f"{ROOT}/{how}/{patient}/{week}.dcm"
            ds = pydicom.dcmread(img_path)
            im = Image.fromarray(ds.pixel_array)
            im = im.resize((DESIRED_SIZE, DESIRED_SIZE))
            im = np.array(im)
            xo.append(im[np.newaxis, :, :])
            p.append(patient)
            w.append(week)
        except:
            pass
    data = pd.DataFrame({"Patient": p, "Weeks": w})
    return np.concatenate(xo, axis=0), data




## === cell 13
x, df_tr = get_images(tr, how="train")



## === cell 14
print(x.shape, df_tr.shape)



## === cell 15
idx = np.random.randint(x.shape[0])
plt.imshow(x[idx], cmap=plt.cm.bone)
plt.show()



## === cell 16
df_tr = df_tr.merge(tr, how="left", on=["Patient", "Weeks"])



## === cell 17
y = df_tr["FVC"].values
z = df_tr[FE].values



## === cell 18
print(z.shape)



## === cell 19
C1, C2 = tf.constant(70, dtype="float32"), tf.constant(1000, dtype="float32")


def kloss(y_true, y_pred):
    sigma = y_pred[:, 1]
    fvc_pred = y_pred[:, 0]
    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true[:, 0] - fvc_pred)
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt(tf.constant(2.0, dtype=tf.float32))
    metric = (delta / sigma_clip) * sq2 + tf.math.log(sigma_clip * sq2)
    return K.mean(metric)


def kmae(y_true, y_pred):
    spread = tf.abs((y_true[:, 0] - y_pred[:, 0]) / y_true[:, 0])
    return K.mean(spread)


def mloss(_lambda):
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
weight_decay = None


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
    weight_decay=weight_decay,
):
    if weight_decay:
        kernel_regularizer = tf.keras.regularizers.l1(weight_decay)
        bias_regularizer = tf.keras.regularizers.l1(weight_decay)
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
    weight_decay=weight_decay,
):
    (branch1, branch2, branch3, branch4) = params
    if weight_decay:
        kernel_regularizer = tf.keras.regularizers.l1(weight_decay)
        bias_regularizer = tf.keras.regularizers.l1(weight_decay)
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
def conv2d_bn(x, filters, num_row, num_col, padding="same", strides=(1, 1), name=None):
    if name is not None:
        bn_name = name + "_bn"
        conv_name = name + "_conv"
    else:
        bn_name = None
        conv_name = None
    x = Conv2D(
        filters,
        (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name,
    )(x)
    x = BatchNormalization(scale=False, name=bn_name)(x)
    x = Activation("relu", name=name)(x)
    return x




## === cell 23
channel_axis = 3


def inceptionV3_module(x):
    x = conv2d_bn(x, 32, 3, 3, strides=(2, 2), padding="valid")
    x = conv2d_bn(x, 32, 3, 3, padding="valid")
    x = conv2d_bn(x, 64, 3, 3)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = conv2d_bn(x, 80, 1, 1, padding="valid")
    x = conv2d_bn(x, 192, 3, 3, padding="valid")
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    branch1x1 = conv2d_bn(x, 64, 1, 1)
    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)
    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding="same")(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
    x = concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name="mixed0",
    )
    return x




## === cell 24
CONCAT_AXIS = 3


def InceptionV1():
    inp = Input((DESIRED_SIZE, DESIRED_SIZE, 1), name="input")
    x = inception_module(
        inp, params=[(128,), (128, 192), (32, 96), (64,)], concat_axis=CONCAT_AXIS
    )
    x = Flatten()(x)
    x = Dropout(DROPOUT)(x)
    preds = Dense(2, activation="relu", name="preds")(x)
    model = tf.keras.Model(inp, preds, name="CNN")
    opt = tfa.optimizers.SWA(tf.keras.optimizers.Adam(lr=0.001))
    model.compile(loss=mloss(0.5), optimizer=opt, metrics=[kloss])
    return model




## === cell 25
def GoogleNet():
    inp = Input((DESIRED_SIZE, DESIRED_SIZE, 1), name="input")
    x = conv2D_lrn2d(inp, 64, (7, 7), 2, padding="same", lrn2d_norm=False)
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
    model = tf.keras.Model(inp, preds, name="CNN")
    opt = tfa.optimizers.SWA(tf.keras.optimizers.Adam(lr=0.001))
    model.compile(loss=mloss(0.5), optimizer=opt, metrics=[kloss])
    return model




## === cell 26
def InceptionV3():
    inp = Input((DESIRED_SIZE, DESIRED_SIZE, 1), name="input")
    x = inceptionV3_module(inp)
    x = GlobalAveragePooling2D(name="avg_pool")(x)
    preds = Dense(2, activation="relu", name="preds")(x)
    model = tf.keras.Model(inp, preds, name="CNN")
    opt = tfa.optimizers.SWA(tf.keras.optimizers.Adam(lr=0.001))
    model.compile(loss=mloss(0.5), optimizer=opt, metrics=[kloss])
    return model




## === cell 28
net = InceptionV3()
net.compile(
    loss=mloss(0.5),
    optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001),
    metrics=[kloss],
)



## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2152684385.py in <cell line: 0>()
      1 # Build a simple model using the InceptionV3 architecture (no EfficientNet)
----> 2 net = InceptionV3()
      3 net.compile(
      4     loss=mloss(0.5),
      5     optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001),

/tmp/ipykernel_11/3424939461.py in InceptionV3()
      5     preds = Dense(2, activation="relu", name="preds")(x)
      6     model = tf.keras.Model(inp, preds, name="CNN")
----> 7     opt = tfa.optimizers.SWA(tf.keras.optimizers.Adam(lr=0.001))
      8     model.compile(loss=mloss(0.5), optimizer=opt, metrics=[kloss])
      9     return model

/usr/local/lib/python3.11/dist-packages/keras/src/optimizers/adam.py in __init__(self, learning_rate, beta_1, beta_2, epsilon, amsgrad, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, **kwargs)
     60         **kwargs,
     61     ):
---> 62         super().__init__(
     63             learning_rate=learning_rate,
     64             name=name,

/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/optimizer.py in __init__(self, *args, **kwargs)
     19 class TFOptimizer(KerasAutoTrackable, base_optimizer.BaseOptimizer):
     20     def __init__(self, *args, **kwargs):
---> 21         super().__init__(*args, **kwargs)
     22         self._distribution_strategy = tf.distribute.get_strategy()
     23 

/usr/local/lib/python3.11/dist-packages/keras/src/optimizers/base_optimizer.py in __init__(self, learning_rate, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, **kwargs)
     88             )
     89         if kwargs:
---> 90             raise ValueError(f"Argument(s) not recognized: {kwargs}")
     91 
     92         if name is None:

ValueError: Argument(s) not recognized: {'lr': 0.001}

## === cell 29
net.summary()



## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/521331300.py in <cell line: 0>()
----> 1 net.summary()
      2 

NameError: name 'net' is not defined

## === cell 30
x_min = np.min(x)
x_max = np.max(x)
xs = (x - x_min) / (x_max - x_min + 1e-8)



## === cell 31
print(xs.shape, y.shape, x_min)



## === cell 32
xs = np.expand_dims(xs, axis=3)



## === cell 33
temp = np.repeat(xs, 3, axis=3)
xs = temp



## === cell 34
y_target = np.column_stack([y, np.zeros_like(y)])

X_train, X_val, y_train, y_val = train_test_split(
    xs, y_target, test_size=0.1, random_state=42
)



## === cell 35
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="./best.weights.h5",
    save_weights_only=True,
    monitor="val_kloss",
    mode="min",
    save_best_only=True,
)



## === cell 36
try:
    net.load_weights("../input/efnb7-gnet-64x64/efnB7_gnet_64x64.h5")
except Exception as e:
    print("Pre‑trained weights not loaded:", e)



## === cell 37
net.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[model_checkpoint_callback],
    verbose=2,
)



## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/440797063.py in <cell line: 0>()
      1 # Quick training (few epochs to keep runtime low)
----> 2 net.fit(
      3     X_train,
      4     y_train,
      5     epochs=5,

NameError: name 'net' is not defined

## === cell 38
pred = net.predict(xs, batch_size=100, verbose=1)



## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/98570857.py in <cell line: 0>()
      1 # Predict on the training set (for demonstration / confidence estimation)
----> 2 pred = net.predict(xs, batch_size=100, verbose=1)
      3 

NameError: name 'net' is not defined

## === cell 39
sigma_opt = mean_absolute_error(y, pred[:, 0])
sigma_mean = np.mean(pred[:, 1])
print("sigma_opt:", sigma_opt, "sigma_mean:", sigma_mean)



## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2080293755.py in <cell line: 0>()
----> 1 sigma_opt = mean_absolute_error(y, pred[:, 0])
      2 sigma_mean = np.mean(pred[:, 1])
      3 print("sigma_opt:", sigma_opt, "sigma_mean:", sigma_mean)
      4 

NameError: name 'pred' is not defined

## === cell 40
xe, df_te = get_images(sub, how="test")
df_te = df_te.merge(sub, how="left", on=["Patient", "Weeks"])



## === cell 41
x_te = (xe - x_min) / (x_max - x_min + 1e-8)
x_te = np.expand_dims(x_te, axis=3)
x_te = np.repeat(x_te, 3, axis=3)



## === cell 42
pe = net.predict(x_te, batch_size=100, verbose=1)



## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1672602514.py in <cell line: 0>()
----> 1 pe = net.predict(x_te, batch_size=100, verbose=1)
      2 

NameError: name 'net' is not defined

## === cell 43
df_te["FVC1"] = pe[:, 0]
df_te["Confidence1"] = pe[:, 1]



## --- ERROR in cell 43, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1530317875.py in <cell line: 0>()
----> 1 df_te["FVC1"] = pe[:, 0]
      2 df_te["Confidence1"] = pe[:, 1]
      3 

NameError: name 'pe' is not defined

## === cell 44
sub = sub.merge(
    df_te[["Patient", "Weeks", "FVC1", "Confidence1"]],
    how="left",
    on=["Patient", "Weeks"],
)



## --- ERROR in cell 44, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1630072185.py in <cell line: 0>()
      1 sub = sub.merge(
----> 2     df_te[["Patient", "Weeks", "FVC1", "Confidence1"]],
      3     how="left",
      4     on=["Patient", "Weeks"],
      5 )

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

KeyError: "['FVC1', 'Confidence1'] not in index"

## === cell 45
subm = sub[["Patient_Week", "FVC", "Confidence", "FVC1", "Confidence1"]].copy()



## --- ERROR in cell 45, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3793820747.py in <cell line: 0>()
----> 1 subm = sub[["Patient_Week", "FVC", "Confidence", "FVC1", "Confidence1"]].copy()
      2 

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

KeyError: "['FVC1', 'Confidence1'] not in index"

## === cell 46
subm.loc[~subm.FVC1.isnull(), "FVC"] = subm.loc[~subm.FVC1.isnull(), "FVC1"]
if sigma_mean < 70:
    subm["Confidence"] = sigma_opt
else:
    subm.loc[~subm.FVC1.isnull(), "Confidence"] = subm.loc[
        ~subm.FVC1.isnull(), "Confidence1"
    ]
    subm.loc[subm.FVC1.isnull(), "Confidence"] = sigma_opt



## --- ERROR in cell 46, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4045831225.py in <cell line: 0>()
----> 1 subm.loc[~subm.FVC1.isnull(), "FVC"] = subm.loc[~subm.FVC1.isnull(), "FVC1"]
      2 if sigma_mean < 70:
      3     subm["Confidence"] = sigma_opt
      4 else:
      5     subm.loc[~subm.FVC1.isnull(), "Confidence"] = subm.loc[

NameError: name 'subm' is not defined

## === cell 47
subm[["Patient_Week", "FVC", "Confidence"]].to_csv("submission.csv", index=False)

## --- ERROR in cell 47, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2447075011.py in <cell line: 0>()
----> 1 subm[["Patient_Week", "FVC", "Confidence"]].to_csv("submission.csv", index=False)

NameError: name 'subm' is not defined
