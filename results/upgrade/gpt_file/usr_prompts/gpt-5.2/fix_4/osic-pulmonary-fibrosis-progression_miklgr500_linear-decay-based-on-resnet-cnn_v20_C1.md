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
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
pydicom==3.0.1
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

-6.9229

# 6. Current score

-9.03792

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -10.78684) has done: 'I fix the TensorFlow/Keras runtime errors caused by API changes in TF 2.18 (replace deprecated `fit_generator` with `fit` and make the `Sequence` robust), and fix the protobuf-related crash by forcing the Python protobuf implementation before importing TensorFlow. I also correct a couple of silent logic issues that hurt score: `np.linalg.lstsq` needs `rcond=None` for modern NumPy, and your `Confidence` calculation is currently based on `Percent`/week distance instead of a proper standard deviation in ml (using your already-computed per-patient `STD` from slice predictions, clipped to the metric’s minimum). These changes preserve your core modeling approach (same CNN, same training loop semantics, same per-slice inference aggregation) but should move the score upward toward the target band. The script run end-to-end and write `submission.csv` with the required columns.'
- What this solution (achieved -9.54054) has done: 'You’re hitting two execution blockers: (1) TensorFlow 2.18 + protobuf 6 can crash at import time, so we force protobuf<->TF compatibility by using the pure-Python protobuf implementation *and* disabling the C++ fast-path before importing TensorFlow; (2) Keras 3/TF 2.18 can’t wrap a `Sequence` that returns Python `list`s, so we make the generator return a tuple structure `((x, tab), y)` instead of `[x, tab], y`. These are minimal runtime fixes that preserve your model and training semantics, and they allow training/inference to run end-to-end and write a valid `submission.csv`. I also ensure the tabular “Sex” mapping matches the dataset casing (`Male`/`Female`) to avoid silent feature corruption, which should improve score toward your target without changing the approach.'
- What this solution (achieved -9.03792) has done: 'We fix the immediate runtime crash on TensorFlow import caused by the protobuf 6.x / TF 2.18 incompatibility by additionally disabling the C++ protobuf implementation *before* importing TensorFlow (this is score-neutral and purely a stability fix). To make the pipeline robust in Kaggle’s environment, we also ensure the data root path resolves correctly whether the notebook is run under `../input/...` or `/kaggle/input/...` without changing any modeling logic. Finally, we keep your existing training/inference logic intact, only adding small guards around DICOM folder listing and ensuring the submission is always written as a valid `.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_C"] = "1"

import cv2
import pydicom
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

np.random.seed(42)
tf.random.set_seed(42)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_ROOT_CANDIDATES = [
    "../input/osic-pulmonary-fibrosis-progression",
    "/kaggle/input/osic-pulmonary-fibrosis-progression",
    "/kaggle/data/osic-pulmonary-fibrosis-progression",
]

DATA_ROOT = None
for cand in DATA_ROOT_CANDIDATES:
    if os.path.exists(os.path.join(cand, "train.csv")):
        DATA_ROOT = cand
        break

if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not locate osic-pulmonary-fibrosis-progression dataset. "
        f"Tried: {DATA_ROOT_CANDIDATES}"
    )

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")

print("Using DATA_ROOT:", DATA_ROOT)



## === cell 2
train = pd.read_csv(TRAIN_CSV)



## === cell 3
train.head()



## === cell 4
train.SmokingStatus.unique()




## === cell 5
def get_tab(df):
    vector = [(df.Age.values[0] - 30) / 30]

    sex = str(df.Sex.values[0]).strip().lower()
    if sex == "male":
        vector.append(0.0)
    else:
        vector.append(1.0)

    if df.SmokingStatus.values[0] == "Never smoked":
        vector.extend([0.0, 0.0])
    elif df.SmokingStatus.values[0] == "Ex-smoker":
        vector.extend([1.0, 1.0])
    elif df.SmokingStatus.values[0] == "Currently smokes":
        vector.extend([0.0, 1.0])
    else:
        vector.extend([1.0, 0.0])
    return np.array(vector, dtype=np.float32)




## === cell 6
A = {}
TAB = {}
P = []
for i, p in tqdm(enumerate(train.Patient.unique()), total=train.Patient.nunique()):
    subp = train.loc[train.Patient == p, :]
    fvc = subp.FVC.values
    weeks = subp.Weeks.values
    c = np.vstack([weeks, np.ones(len(weeks))]).T

    a, b = np.linalg.lstsq(c, fvc, rcond=None)[0]

    A[p] = a
    TAB[p] = get_tab(subp)
    P.append(p)




## === cell 7
def get_img(path):
    d = pydicom.dcmread(path)
    img = d.pixel_array.astype(np.float32) / (2**11)
    return cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)




## === cell 8
from tensorflow.keras.utils import Sequence


class IGenerator(Sequence):
    BAD_ID = ["ID00011637202177653955184", "ID00052637202186188008618"]

    def __init__(self, keys, a, tab, batch_size=32):
        self.keys = [k for k in keys if k not in self.BAD_ID]
        self.a = a
        self.tab = tab
        self.batch_size = batch_size

        self.train_data = {}
        for p in train.Patient.unique():
            folder = os.path.join(TRAIN_DIR, p)
            if os.path.isdir(folder):
                try:
                    self.train_data[p] = os.listdir(folder)
                except Exception:
                    self.train_data[p] = []

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        x = []
        a_out, tab_out = [], []
        keys = np.random.choice(self.keys, size=self.batch_size, replace=True)
        for k in keys:
            if k not in self.train_data or len(self.train_data[k]) == 0:
                continue
            i = np.random.choice(self.train_data[k], size=1)[0]
            try:
                img = get_img(os.path.join(TRAIN_DIR, k, i))
            except Exception:
                continue
            x.append(img)
            a_out.append(self.a[k])
            tab_out.append(self.tab[k])

        if len(x) == 0:
            x = [np.zeros((512, 512), dtype=np.float32)]
            a_out = [0.0]
            tab_out = [np.zeros((4,), dtype=np.float32)]

        while len(x) < self.batch_size:
            j = np.random.randint(0, len(x))
            x.append(x[j])
            a_out.append(a_out[j])
            tab_out.append(tab_out[j])

        x = np.array(x, dtype=np.float32)
        a_out = np.array(a_out, dtype=np.float32)
        tab_out = np.array(tab_out, dtype=np.float32)
        x = np.expand_dims(x, axis=-1)

        return (x, tab_out), a_out




## === cell 9
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Input,
    BatchNormalization,
    GlobalAveragePooling2D,
    Add,
    Conv2D,
    AveragePooling2D,
    LeakyReLU,
    Concatenate,
)
from tensorflow.keras import Model


def get_model(shape=(512, 512, 1)):
    def res_block(x, n_features):
        _x = x
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Conv2D(n_features, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
        x = Add()([_x, x])
        return x

    inp = Input(shape=shape)

    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding="same")(inp)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Conv2D(8, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
    for _ in range(2):
        x = res_block(x, 8)
    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
    for _ in range(2):
        x = res_block(x, 16)
    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
    for _ in range(3):
        x = res_block(x, 32)
    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
    for _ in range(3):
        x = res_block(x, 64)
    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
    for _ in range(3):
        x = res_block(x, 128)

    x = GlobalAveragePooling2D()(x)

    inp2 = Input(shape=(4,))
    x = Concatenate()([x, inp2])
    x = Dropout(0.6)(x)
    x = Dense(1)(x)
    return Model([inp, inp2], x)




## === cell 10
model = get_model()
model.summary()



## === cell 11
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=["mae"])




## === cell 12
def build_lrfn(
    lr_start=5e-4,
    lr_max=4e-3,
    lr_min=1e-4,
    lr_rampup_epochs=7,
    lr_sustain_epochs=2,
    lr_exp_decay=0.85,
):

    def lrfn(epoch):
        if epoch < lr_rampup_epochs:
            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) * lr_exp_decay ** (
                epoch - lr_rampup_epochs - lr_sustain_epochs
            ) + lr_min
        return lr

    return lrfn


plt.figure(figsize=(10, 7))
_lrfn = build_lrfn()
plt.plot([i for i in range(35)], [_lrfn(i) for i in range(35)])



## === cell 13
from sklearn.model_selection import train_test_split

tr_p, vl_p = train_test_split(P, shuffle=True, train_size=0.8, random_state=42)



## === cell 14
er = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=1e-3,
    patience=5,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
)
lr_schedule = tf.keras.callbacks.LearningRateScheduler(_lrfn, verbose=1)



## === cell 15
model.fit(
    IGenerator(keys=tr_p, a=A, tab=TAB),
    steps_per_epoch=50,
    validation_data=IGenerator(keys=vl_p, a=A, tab=TAB),
    validation_steps=20,
    callbacks=[er, lr_schedule],
    epochs=30,
    verbose=1,
)



## === cell 16
sub = pd.read_csv(SAMPLE_SUB)
sub.head()



## === cell 17
test = pd.read_csv(TEST_CSV)
test.head()



## === cell 18
A_test, B_test, P_test, STD, WEEK = {}, {}, {}, {}, {}
for p in test.Patient.unique():
    x = []
    tab = []
    folder = os.path.join(TEST_DIR, p)
    if not os.path.isdir(folder):
        continue

    files = []
    try:
        files = os.listdir(folder)
    except Exception:
        files = []

    for i in files:
        x.append(get_img(os.path.join(folder, i)))
        tab.append(get_tab(test.loc[test.Patient == p, :]))

    tab = np.array(tab, dtype=np.float32)
    x = np.expand_dims(np.array(x, dtype=np.float32), axis=-1)

    _a = model.predict([x, tab], verbose=0).reshape(-1)
    a = np.quantile(_a, 0.75)
    std = float(np.std(_a))

    A_test[p] = float(a)
    B_test[p] = float(
        test.FVC.values[test.Patient == p][0]
        - a * test.Weeks.values[test.Patient == p][0]
    )
    P_test[p] = float(test.Percent.values[test.Patient == p][0])
    STD[p] = std
    WEEK[p] = int(test.Weeks.values[test.Patient == p][0])



## === cell 19
for k in sub.Patient_Week.values:
    p, w = k.split("_")
    w = int(w)

    fvc = A_test[p] * w + B_test[p]
    sub.loc[sub.Patient_Week == k, "FVC"] = fvc

    conf = max(STD[p], 70.0)
    sub.loc[sub.Patient_Week == k, "Confidence"] = conf



## === cell 20
sub.head()



## === cell 21
sub_out = sub[["Patient_Week", "FVC", "Confidence"]].copy()
sub_out["FVC"] = sub_out["FVC"].astype(np.float32)
sub_out["Confidence"] = sub_out["Confidence"].astype(np.float32)
sub_out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_out.shape)
print(sub_out.head())
