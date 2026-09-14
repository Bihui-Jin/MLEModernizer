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

-8.2838

# 6. Current score

-7.31242

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -9.6804) has done: 'Diagnosis: The crash happens during `import tensorflow as tf` in cell 0 because your environment has `protobuf==6.33.0`, which is incompatible with TensorFlow 2.18’s expected protobuf runtime API (it looks for `MessageFactory.GetPrototype`, removed/changed in newer protobuf). This is a known protobuf/TensorFlow version mismatch issue that prevents TensorFlow from importing at all. The minimal in-notebook workaround is to force TensorFlow to use the pure-Python protobuf implementation via an environment variable before importing TensorFlow, which avoids the failing C++ path.  

Patch summary: In cell 0, set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (and version `2`) before importing TensorFlow, then import TensorFlow as originally intended. No other logic is changed.  

Updated cells: (only cell 0 updated below)  

Compatibility notes for cell k+1: Cell 1 expects `pd` to be imported; this remains unchanged. TensorFlow now import successfully as `tf` for any later cells that use it.  

Assumptions: This environment allows setting `os.environ` before importing TensorFlow, and using the Python protobuf implementation is acceptable for this notebook (may be slower but is deterministic and unblocks execution).'
- What this solution (achieved -9.35884) has done: 'Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.
The crash happens in cell 0 during `import tensorflow as tf` because the environment has `protobuf==6.33.0`, which is incompatible with TensorFlow 2.18 in a way that triggers `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. The two `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION*` environment variables are being set **after** importing `protobuf`-dependent packages (and in practice are insufficient for this mismatch), so the import still fails. The minimal fix is to force protobuf to use the pure-Python implementation **before any protobuf-using imports**, and to ensure TensorFlow is imported only after that is set. This change is localized to cell 0 and preserves all downstream variables and interfaces used by cell 1+.'
- What this solution (achieved -9.832) has done: 'Diagnosis: The crash happens during TensorFlow import in cell 0 due to an incompatibility between TensorFlow 2.18.0 and the installed protobuf 6.33.0; TensorFlow expects protobuf <5 and triggers `'MessageFactory' object has no attribute 'GetPrototype'`. The current workaround forcing `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"` is insufficient with protobuf 6.x. The minimal deterministic fix is to downgrade protobuf at runtime (via pip) to a TensorFlow-compatible version before importing TensorFlow.

Patch summary: In cell 0 only, add a small pre-import guard that checks the installed protobuf version and, if it is >=5, installs `protobuf==4.25.3` and clears any already-imported `google.protobuf` modules so the new version is used. Keep everything else (imports, logic, env vars) unchanged.

Updated cells: cell 0 only.

Compatibility notes for cell k+1: No variables or interfaces used by cell 1 change; `pd`, `np`, etc. are still imported and TensorFlow import succeed, allowing subsequent cells to run.

Assumptions: The environment allows `pip` installs at runtime (common in notebook/Kaggle-like environments) and has network/package access to fetch protobuf; if offline, the only alternative would be changing the base image, which is outside the constraints.'
- What this solution (achieved -9.97001) has done: 'Diagnosis: TensorFlow/Keras 2.18 removed the deprecated `Model.fit_generator` API, so calling it raises `AttributeError`. Your generator `IGenerator` already subclasses `tf.keras.utils.Sequence`, which is fully supported by `Model.fit`, so the training logic can be preserved by switching to `fit` with the same arguments. No other cells need changes because the returned data shapes/types and callbacks remain identical.

Patch summary: In cell 14 only, replace `model.fit_generator(...)` with `model.fit(...)`, keeping all parameters the same to maintain training/evaluation semantics.

Updated cells:'
- What this solution (achieved -9.84197) has done: 'The timeout is dominated by repeated DICOM decoding/resizing inside the Python `Sequence` and the overhead of wrapping it into a `tf.data` generator, plus extra work in `IGenerator.__init__` building `train_data` by iterating over every row in `train`. I keep the exact same model/fit setup and the same number of steps/epochs, but make the data pipeline provably equivalent and much faster by (1) caching resized slices in-memory (and optionally on-disk) keyed by file path, (2) precomputing per-patient file lists once using `os.scandir` instead of per-row `os.listdir`, (3) returning float32 arrays directly (avoids redundant conversions), and (4) feeding the `Sequence` directly to `model.fit(..., workers=..., use_multiprocessing=True, max_queue_size=...)` to enable parallel CPU decoding without changing training semantics. I also vectorize submission filling to remove slow per-row `sub.loc` updates (same math, same outputs).'
- What this solution (achieved -9.21261) has done: 'Diagnosis: The crash in cell 14 is due to a Keras/TensorFlow API change in TF 2.18 where `Model.fit()` no longer accepts the legacy `workers`, `use_multiprocessing`, and `max_queue_size` arguments for Python `Sequence` inputs under the new `TensorFlowTrainer`. This causes an immediate `TypeError` before any training starts. The generator itself remains compatible with `fit()`, so we only need to remove the unsupported arguments while keeping the same training data, steps, callbacks, and epochs.

Patch summary: In cell 14, drop the `workers=...`, `use_multiprocessing=...`, and `max_queue_size=...` parameters from `model.fit()` to match the TF 2.18 signature. Keep everything else (generators, steps, callbacks, epochs) identical to preserve training/evaluation semantics.

Updated cells: (cell 14 only)

Compatibility notes for cell k+1: Cell 15 only reads the submission CSV and does not depend on `_workers` or multiprocessing settings, so removing these fit-arguments does not affect later variables or interfaces.

Assumptions: Training is intended to run in the current TF 2.18 environment, and it is acceptable to use single-process data loading (default behavior) since the removed arguments are no longer supported for this trainer path.'
- What this solution (achieved -9.86921) has done: 'The timeout is dominated by repeatedly decoding/resizing DICOMs inside a Python `Sequence` wrapped by a slow `from_generator` loop, plus repeated directory scans and per-item Python overhead. I keep the same model and training semantics, but speed up data feeding by (1) pre-indexing each patient’s DICOM paths once, (2) switching to a pure `tf.data` pipeline with parallel map and prefetch so image decoding happens concurrently, and (3) making DICOM reads faster by using `stop_before_pixels=True`, `force=True`, and minimal metadata parsing while keeping identical pixel arrays. I also remove extra fixed work in the generator (`__len__=1000`) by driving epochs purely via `steps_per_epoch`, which preserves training behavior because the model already trains on random patient/slice sampling each step.'
- What this solution (achieved -7.31242) has done: 'Your current gap to the target is about 1.59 points (−9.869 → −8.284), so we should improve score rather than refactor. The biggest issue affecting this metric is that your `Confidence` values can become negative or too small, which gets clipped to 70 in scoring and heavily penalizes the log term; we make Confidence strictly positive and tune its scale using only training-derived calibration. Concretely, we (1) build a simple global confidence calibration from the distribution of residuals of your per-patient linear fit (same core logic), and (2) set per-row Confidence to `sqrt(calib_sigma^2 + model_std^2)` with a small floor, which usually improves LaplaceLL without changing your FVC prediction logic. This is minimal, fast, and directly aligned to the competition metric.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

import sys
import subprocess

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as _pb_ver

    _pb_major = int(_pb_ver.split(".", 1)[0])
except Exception:
    _pb_major = None

if _pb_major is not None and _pb_major >= 5:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
    )
    for _m in list(sys.modules.keys()):
        if _m.startswith("google.protobuf"):
            del sys.modules[_m]

import cv2
import pydicom
import pandas as pd
import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

np.random.seed(42)
tf.random.set_seed(42)

try:
    cv2.setNumThreads(max(1, (os.cpu_count() or 2) - 1))
except Exception:
    pass

tf.data.experimental.enable_debug_mode = False
AUTOTUNE = tf.data.AUTOTUNE



## === cell 1
train = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")



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

RESID_SIGMA = {}

for i, p in tqdm(enumerate(train.Patient.unique())):
    sub = train.loc[train.Patient == p, :]
    fvc = sub.FVC.values.astype(np.float32)
    weeks = sub.Weeks.values.astype(np.float32)
    c = np.vstack([weeks, np.ones(len(weeks), dtype=np.float32)]).T
    a, b = np.linalg.lstsq(c, fvc, rcond=None)[0]

    A[p] = a
    TAB[p] = get_tab(sub)
    P.append(p)

    pred = a * weeks + b
    mae = float(np.mean(np.abs(fvc - pred))) if len(fvc) > 0 else 200.0
    RESID_SIGMA[p] = np.float32(np.sqrt(2.0) * mae)

_GLOBAL_SIGMA = np.float32(np.median(list(RESID_SIGMA.values())))



## === cell 6
from functools import lru_cache

_IMG_SIZE = (512, 512)


@lru_cache(maxsize=8192)
def _get_img_cached(path: str):
    d = pydicom.dcmread(path, force=True, stop_before_pixels=False, specific_tags=None)
    arr = d.pixel_array.astype(np.float32) / (2**11)
    arr = cv2.resize(arr, _IMG_SIZE)
    return arr


def get_img(path):
    return _get_img_cached(path)




## === cell 7
from tensorflow.keras.utils import Sequence


class IGenerator(Sequence):
    BAD_ID = ["ID00011637202177653955184", "ID00052637202186188008618"]

    def __init__(self, keys, a, tab, batch_size=32):
        self.keys = [k for k in keys if k not in self.BAD_ID]
        self.a = a
        self.tab = tab
        self.batch_size = batch_size

        base = "../input/osic-pulmonary-fibrosis-progression/train"
        self.train_data = {}
        for p in train.Patient.unique():
            pdir = f"{base}/{p}/"
            try:
                with os.scandir(pdir) as it:
                    self.train_data[p] = [e.name for e in it if e.is_file()]
            except FileNotFoundError:
                self.train_data[p] = []

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        x = []
        a, tab = [], []
        keys = np.random.choice(self.keys, size=self.batch_size)
        for k in keys:
            try:
                i = np.random.choice(self.train_data[k], size=1)[0]
                img = get_img(
                    f"../input/osic-pulmonary-fibrosis-progression/train/{k}/{i}"
                )
                x.append(img)
                a.append(self.a[k])
                tab.append(self.tab[k])
            except Exception:
                print(k, i)

        x = np.asarray(x, dtype=np.float32)
        a = np.asarray(a, dtype=np.float32)
        tab = np.asarray(tab, dtype=np.float32)
        x = np.expand_dims(x, axis=-1)
        return [x, tab], a




## === cell 8
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Activation,
    Flatten,
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
from tensorflow.keras.optimizers import Nadam


def get_model(shape=(512, 512, 1)):
    def res_block(x, n_features):
        _x = x
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv2D(n_features, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
        x = Add()([_x, x])
        return x

    inp = Input(shape=shape)

    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
    for _ in range(2):
        x = res_block(x, 32)
    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
    for _ in range(2):
        x = res_block(x, 32)
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
    x = Dropout(0.35)(x)

    x = Dense(16)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    inp2 = Input(shape=(4,))

    x2 = Dense(4)(inp2)
    x2 = BatchNormalization()(x2)
    x2 = Activation("relu")(x2)

    x = Concatenate()([x, x2])

    x = Dropout(0.2)(x)
    x = Dense(1)(x)
    return Model([inp, inp2], x)




## === cell 9
model = get_model()
model.summary()



## === cell 10
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.003), loss=["mae"])




## === cell 11
def build_lrfn(
    lr_start=5e-4,
    lr_max=1e-3,
    lr_min=1e-4,
    lr_rampup_epochs=3,
    lr_sustain_epochs=0,
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


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 7))

_lrfn = build_lrfn()
plt.plot([i for i in range(35)], [_lrfn(i) for i in range(35)])



## === cell 12
from sklearn.model_selection import train_test_split

tr_p, vl_p = train_test_split(P, shuffle=True, train_size=0.8)



## === cell 13
er = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=1e-3,
    patience=7,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
)
lr_schedule = tf.keras.callbacks.LearningRateScheduler(_lrfn, verbose=1)



## === cell 14
from typing import Dict, List

TRAIN_IMG_BASE = "../input/osic-pulmonary-fibrosis-progression/train"

BAD_ID = set(["ID00011637202177653955184", "ID00052637202186188008618"])

_train_files: Dict[str, np.ndarray] = {}
for p in train.Patient.unique():
    if p in BAD_ID:
        continue
    pdir = f"{TRAIN_IMG_BASE}/{p}/"
    try:
        with os.scandir(pdir) as it:
            files = [f"{pdir}{e.name}" for e in it if e.is_file()]
    except FileNotFoundError:
        files = []
    if len(files) > 0:
        _train_files[p] = np.array(files, dtype=object)

tr_keys = np.array([p for p in tr_p if p in _train_files], dtype=object)
vl_keys = np.array([p for p in vl_p if p in _train_files], dtype=object)

A_arr = {k: np.float32(A[k]) for k in _train_files.keys()}
TAB_arr = {k: TAB[k].astype(np.float32) for k in _train_files.keys()}


def _py_sample_batch(keys_np: np.ndarray, batch_size: int):
    keys = np.random.choice(keys_np, size=batch_size, replace=True)
    x = np.empty((batch_size, 512, 512, 1), dtype=np.float32)
    tab = np.empty((batch_size, 4), dtype=np.float32)
    y = np.empty((batch_size,), dtype=np.float32)
    for j, k in enumerate(keys):
        flist = _train_files[k]
        fpath = flist[np.random.randint(0, len(flist))]
        img = get_img(str(fpath))
        x[j, ..., 0] = img
        tab[j] = TAB_arr[k]
        y[j] = A_arr[k]
    return x, tab, y


def make_dataset(keys_np: np.ndarray, batch_size: int):
    def _gen():
        while True:
            x, tab, y = _py_sample_batch(keys_np, batch_size)
            yield (x, tab), y

    output_signature = (
        (
            tf.TensorSpec(shape=(batch_size, 512, 512, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(batch_size, 4), dtype=tf.float32),
        ),
        tf.TensorSpec(shape=(batch_size,), dtype=tf.float32),
    )
    ds = tf.data.Dataset.from_generator(_gen, output_signature=output_signature)
    return ds.prefetch(AUTOTUNE)


BATCH_SIZE = 32
train_ds = make_dataset(tr_keys, BATCH_SIZE)
val_ds = make_dataset(vl_keys, BATCH_SIZE)

model.fit(
    train_ds,
    steps_per_epoch=50,
    validation_data=val_ds,
    validation_steps=20,
    callbacks=[er, lr_schedule],
    epochs=30,
)



## === cell 15
sub = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/sample_submission.csv")
sub.head()



## === cell 16
test = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
test.head()



## === cell 17
A_test, B_test, P_test, W, FVC = {}, {}, {}, {}, {}
STD, WEEK = {}, {}

TEST_IMG_BASE = "../input/osic-pulmonary-fibrosis-progression/test"

for p in test.Patient.unique():
    pdir = f"{TEST_IMG_BASE}/{p}/"
    files = []
    with os.scandir(pdir) as it:
        for e in it:
            if e.is_file():
                files.append(f"{pdir}{e.name}")

    x = [get_img(f) for f in files]

    tab_one = get_tab(test.loc[test.Patient == p, :]).astype(np.float32)
    tab = np.repeat(tab_one[None, :], repeats=len(x), axis=0)

    x = np.expand_dims(np.asarray(x, dtype=np.float32), axis=-1)
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



## === cell 18
pw = sub["Patient_Week"].values
patients = np.array([k.split("_", 1)[0] for k in pw], dtype=object)
weeks = np.array([int(k.split("_", 1)[1]) for k in pw], dtype=np.int32)

fvc_pred = np.empty(len(sub), dtype=np.float32)
conf_pred = np.empty(len(sub), dtype=np.float32)

CONF_FLOOR = np.float32(
    70.0
)  # metric clips at 70; keeping >=70 avoids extra log penalty from low sigma
for p in np.unique(patients):
    m = patients == p
    w = weeks[m].astype(np.float32)
    fvc_pred[m] = (A_test[p] * w + B_test[p]).astype(np.float32)

    model_sigma = np.float32(
        np.abs(STD[p]) * 1000.0
    )  # keep scale similar to ml; std(_a) is in slope units
    calib_sigma = _GLOBAL_SIGMA  # global training-derived fallback in ml

    sigma = np.sqrt(calib_sigma * calib_sigma + model_sigma * model_sigma).astype(
        np.float32
    )
    if sigma < CONF_FLOOR:
        sigma = CONF_FLOOR
    conf_pred[m] = sigma

sub["FVC"] = fvc_pred
sub["Confidence"] = conf_pred



## === cell 19
sub.head()



## === cell 20
sub[["Patient_Week", "FVC", "Confidence"]].to_csv("submission.csv", index=False)
