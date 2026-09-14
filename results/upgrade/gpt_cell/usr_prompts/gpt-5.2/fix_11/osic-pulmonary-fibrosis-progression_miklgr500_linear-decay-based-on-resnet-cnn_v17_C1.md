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

# 5. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import cv2
import pydicom
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

SEED = 42
os.environ.setdefault("PYTHONHASHSEED", str(SEED))
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass
try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

BASE_PATH = "../input/osic-pulmonary-fibrosis-progression"
TRAIN_IMG_DIR = f"{BASE_PATH}/train"
TEST_IMG_DIR = f"{BASE_PATH}/test"
IMG_CACHE_DIR = "../working/dcm_cache_512"
os.makedirs(IMG_CACHE_DIR, exist_ok=True)

_IMG_MEMO = {}
_PATH2CACHE = {}



## === cell 1
train = pd.read_csv(f"{BASE_PATH}/train.csv")



## === cell 2
train.head()



## === cell 3
train.SmokingStatus.unique()




## === cell 4
def get_tab(df):
    vector = [(df.Age.values[0] - 30) / 30]

    if df.Sex.values[0] == "male":
        vector.append(0)
    else:
        vector.append(1)

    if df.SmokingStatus.values[0] == "Never smoked":
        vector.extend([0, 0])
    elif df.SmokingStatus.values[0] == "Ex-smoker":
        vector.extend([1, 1])
    elif df.SmokingStatus.values[0] == "Currently smokes":
        vector.extend([0, 1])
    else:
        vector.extend([1, 0])
    return np.array(vector)




## === cell 5
A = {}
TAB = {}
P = []
for i, p in tqdm(enumerate(train.Patient.unique())):
    sub = train.loc[train.Patient == p, :]
    fvc = sub.FVC.values
    weeks = sub.Weeks.values
    c = np.vstack([weeks, np.ones(len(weeks))]).T
    a, b = np.linalg.lstsq(c, fvc, rcond=None)[0]

    A[p] = a
    TAB[p] = get_tab(sub)
    P.append(p)



## === cell 6
import hashlib


def _cache_path_for(path: str) -> str:
    cp = _PATH2CACHE.get(path)
    if cp is not None:
        return cp
    h = hashlib.md5(path.encode("utf-8")).hexdigest()
    cp = os.path.join(IMG_CACHE_DIR, f"{h}.npy")
    _PATH2CACHE[path] = cp
    return cp


def get_img(path):
    cached = _IMG_MEMO.get(path)
    if cached is not None:
        return cached

    cache_path = _cache_path_for(path)
    if os.path.exists(cache_path):
        arr = np.load(cache_path, mmap_mode=None)
        _IMG_MEMO[path] = arr
        return arr

    d = pydicom.dcmread(path, force=True)
    arr = d.pixel_array  # triggers pixel decoding only once
    arr = cv2.resize(arr / (2**11), (512, 512))
    arr = arr.astype(np.float32, copy=False)

    np.save(cache_path, arr)
    _IMG_MEMO[path] = arr
    return arr




## === cell 7
from tensorflow.keras.utils import Sequence

_TRAIN_FILES_BY_PATIENT = None


class IGenerator(Sequence):
    BAD_ID = ["ID00011637202177653955184", "ID00052637202186188008618"]

    def __init__(self, keys, a, tab, batch_size=32):
        global _TRAIN_FILES_BY_PATIENT
        self.keys = [k for k in keys if k not in self.BAD_ID]
        self.a = a
        self.tab = tab
        self.batch_size = batch_size

        if _TRAIN_FILES_BY_PATIENT is None:
            files = {}
            for p in train.Patient.unique():
                files[p] = os.listdir(f"{TRAIN_IMG_DIR}/{p}/")
            _TRAIN_FILES_BY_PATIENT = files
        self.train_data = _TRAIN_FILES_BY_PATIENT

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        x = np.empty((self.batch_size, 512, 512, 1), dtype=np.float32)
        tab = np.empty((self.batch_size, 4), dtype=np.float32)
        a_out = np.empty((self.batch_size,), dtype=np.float32)

        keys = np.random.choice(self.keys, size=self.batch_size)
        for j, k in enumerate(keys):
            try:
                i = np.random.choice(self.train_data[k], size=1)[0]
                img = get_img(f"{TRAIN_IMG_DIR}/{k}/{i}")
                x[j, :, :, 0] = img
                a_out[j] = self.a[k]
                tab[j, :] = self.tab[k].astype(np.float32, copy=False)
            except Exception:
                print(k)

        return (x, tab), a_out




## === cell 8
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




## === cell 9
model = get_model()
model.summary()



## === cell 10
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=["mae"])



## === cell 11
from sklearn.model_selection import train_test_split

tr_p, vl_p = train_test_split(P, shuffle=True, train_size=0.8, random_state=SEED)



## === cell 12
er = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=1e-3,
    patience=5,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
)



## === cell 13
train_gen = IGenerator(keys=tr_p, a=A, tab=TAB)
val_gen = IGenerator(keys=vl_p, a=A, tab=TAB)

history = model.fit(
    train_gen,
    steps_per_epoch=200,
    validation_data=val_gen,
    validation_steps=20,
    callbacks=[er],
    epochs=30,
)



## === cell 14
sub = pd.read_csv(f"{BASE_PATH}/sample_submission.csv")
sub.head()



## === cell 15
test = pd.read_csv(f"{BASE_PATH}/test.csv")
test.head()



## === cell 16
A_test, B_test, P_test, W, FVC = {}, {}, {}, {}, {}
STD, WEEK = {}, {}

test_tab_by_patient = {
    p: get_tab(test.loc[test.Patient == p, :]) for p in test.Patient.unique()
}

for p in test.Patient.unique():
    files = os.listdir(f"{TEST_IMG_DIR}/{p}/")
    x = [get_img(f"{TEST_IMG_DIR}/{p}/{fn}") for fn in files]
    x = np.expand_dims(np.asarray(x, dtype=np.float32), axis=-1)

    tab = np.repeat(
        test_tab_by_patient[p][None, :].astype(np.float32), repeats=len(files), axis=0
    )

    _a = model.predict([x, tab], verbose=0)
    a = np.quantile(_a, 0.75)
    std = np.std(_a)

    A_test[p] = a
    B_test[p] = (
        test.FVC.values[test.Patient == p] - a * test.Weeks.values[test.Patient == p]
    )
    P_test[p] = test.Percent.values[test.Patient == p]
    STD[p] = std
    WEEK[p] = test.Weeks.values[test.Patient == p]



## === cell 17
A



## === cell 18
A_test



## === cell 19
pw = sub["Patient_Week"].values
p_arr = np.array([s.split("_")[0] for s in pw], dtype=object)
w_arr = np.array([int(s.split("_")[1]) for s in pw], dtype=np.int32)

a_vec = np.array([A_test[p] for p in p_arr], dtype=np.float32)
b_vec = np.array([B_test[p] for p in p_arr], dtype=np.float32)
pperc_vec = np.array([P_test[p] for p in p_arr], dtype=np.float32)
week0_vec = np.array([WEEK[p] for p in p_arr], dtype=np.float32)

sub["FVC"] = a_vec * w_arr.astype(np.float32) + b_vec
sub["Confidence"] = pperc_vec - a_vec * np.abs(week0_vec - w_arr.astype(np.float32))



## === cell 20
sub.head()



## === cell 21
sub[["Patient_Week", "FVC", "Confidence"]].to_csv("submission.csv", index=False)
