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

-6.9229

# 6. Current score

-8.18296

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -10.78684) has done: 'I fix the TensorFlow/Keras runtime errors caused by API changes in TF 2.18 (replace deprecated `fit_generator` with `fit` and make the `Sequence` robust), and fix the protobuf-related crash by forcing the Python protobuf implementation before importing TensorFlow. I also correct a couple of silent logic issues that hurt score: `np.linalg.lstsq` needs `rcond=None` for modern NumPy, and your `Confidence` calculation is currently based on `Percent`/week distance instead of a proper standard deviation in ml (using your already-computed per-patient `STD` from slice predictions, clipped to the metric’s minimum). These changes preserve your core modeling approach (same CNN, same training loop semantics, same per-slice inference aggregation) but should move the score upward toward the target band. The script run end-to-end and write `submission.csv` with the required columns.'
- What this solution (achieved -9.54054) has done: 'You’re hitting two execution blockers: (1) TensorFlow 2.18 + protobuf 6 can crash at import time, so we force protobuf<->TF compatibility by using the pure-Python protobuf implementation *and* disabling the C++ fast-path before importing TensorFlow; (2) Keras 3/TF 2.18 can’t wrap a `Sequence` that returns Python `list`s, so we make the generator return a tuple structure `((x, tab), y)` instead of `[x, tab], y`. These are minimal runtime fixes that preserve your model and training semantics, and they allow training/inference to run end-to-end and write a valid `submission.csv`. I also ensure the tabular “Sex” mapping matches the dataset casing (`Male`/`Female`) to avoid silent feature corruption, which should improve score toward your target without changing the approach.'
- What this solution (achieved -9.03792) has done: 'We fix the immediate runtime crash on TensorFlow import caused by the protobuf 6.x / TF 2.18 incompatibility by additionally disabling the C++ protobuf implementation *before* importing TensorFlow (this is score-neutral and purely a stability fix). To make the pipeline robust in Kaggle’s environment, we also ensure the data root path resolves correctly whether the notebook is run under `../input/...` or `/kaggle/input/...` without changing any modeling logic. Finally, we keep your existing training/inference logic intact, only adding small guards around DICOM folder listing and ensuring the submission is always written as a valid `.csv` with the required columns.'
- What this solution (achieved -9.23053) has done: 'The current crash happens before any training because TensorFlow import is failing due to an incompatibility between TF 2.18 and protobuf 6.x (the `MessageFactory.GetPrototype` AttributeError). I fix this in the smallest way by forcing a compatible protobuf version at runtime (via pip inside the notebook) before importing TensorFlow, while keeping your model, generator, training loop, and inference logic unchanged. I also add a small safety fallback for reading DICOM pixel data (some DICOMs can fail `pixel_array` decoding without `force=True`) so inference doesn’t silently skip too many slices and hurt score. Finally, I keep the submission writing exactly the same but ensure the required columns are present and numeric.'
- What this solution (achieved -8.82181) has done: 'The timeout is dominated by repeated, slow DICOM decoding + resizing inside the `Sequence` (done synchronously with `workers=1`) and by expensive full-volume test-time inference (predicting on *all* slices per patient). To keep core logic identical, we (1) keep the exact same model/training loop semantics but make data loading faster via `stop_before_pixels`, sorting/fast filtering of DICOM filenames, and a lightweight in-memory LRU for decoded+resized slices; (2) enable Keras `Sequence` multiprocessing with safe worker settings so CPU can decode/resize while GPU/TF trains; and (3) keep the same slice-aggregation logic (median + std over slices) but cap test-time slices deterministically (same as already done for sigma-floor estimation) to remove the biggest inference bottleneck without changing evaluation semantics (since Kaggle score only uses 3 weeks/patient and the aggregation remains identical on the selected slices). These changes reduce disk I/O, Python overhead, and make training/inference finish well under 600s while preserving the algorithm’s intent and outputs up to negligible floating-point differences.'
- What this solution (achieved -8.18296) has done: 'We fix the runtime error in TF/Keras 2.18 where `model.fit()` no longer accepts `workers/use_multiprocessing/max_queue_size` by removing those arguments (this is purely an API compatibility fix; it doesn’t change the model or training objective). To keep the pipeline stable and within time, we optionally enable `tf.data`-style prefetching is not applicable here, so we simply keep the same `Sequence` and the same `steps_per_epoch/epochs`. Everything else (data loading, model architecture, inference aggregation, confidence computation, and submission formatting) remains unchanged. The script then run end-to-end and write a valid `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_C"] = "1"

import sys
import subprocess


def _ensure_protobuf_compatible():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver

        major = int(pb_ver.split(".")[0])
        if major >= 5:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
            )
            for m in list(sys.modules.keys()):
                if m.startswith("google.protobuf"):
                    del sys.modules[m]
    except Exception:
        pass


_ensure_protobuf_compatible()

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
    tf.config.threading.set_intra_op_parallelism_threads(max(1, (os.cpu_count() or 2)))
    tf.config.threading.set_inter_op_parallelism_threads(max(1, (os.cpu_count() or 2)))
except Exception:
    pass

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")




## === cell 1
DATA_ROOT_CANDIDATES = [
    "../input/osic-pulmonary-fibrosis-progression",
    "/kaggle/input/osic-pulmonary-fibrosis-progression",
    "/kaggle/data/osic-pulmonary-fibrosis-progression",
    "/kaggle/data/osic-pulmonary-fibrosis-progression/osic-pulmonary-fibrosis-progression",
    "/kaggle/input/osic-pulmonary-fibrosis-progression/osic-pulmonary-fibrosis-progression",
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
train_patients = train.Patient.unique().tolist()
train_by_patient = {
    p: df.reset_index(drop=True) for p, df in train.groupby("Patient", sort=False)
}
train_firstrow_by_patient = {
    p: df.iloc[[0]] for p, df in train_by_patient.items() if len(df) > 0
}




## === cell 6
def get_tab(df, week_value=None):
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

    if week_value is None:
        week_value = float(df.Weeks.values[0])
    vector.append(float(week_value) / 100.0)

    return np.array(vector, dtype=np.float32)




## === cell 7
P = list(train.Patient.unique())




## === cell 8
from functools import lru_cache

try:
    pydicom.config.convert_wrong_length_to_UN = True
except Exception:
    pass

_DICOM_READ_KW = dict(force=True, stop_before_pixels=False)


@lru_cache(maxsize=8192)
def _read_dicom_pixels(path: str):
    d = pydicom.dcmread(path, **_DICOM_READ_KW)
    return d.pixel_array


@lru_cache(maxsize=8192)
def get_img(path: str):
    img = _read_dicom_pixels(path).astype(np.float32) / (2**11)
    return cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)




## === cell 9
def _list_dicom_paths(root_dir: str, patient_ids):
    out = {}
    for p in patient_ids:
        folder = os.path.join(root_dir, p)
        if not os.path.isdir(folder):
            out[p] = tuple()
            continue
        try:
            files = os.listdir(folder)
        except Exception:
            out[p] = tuple()
            continue

        dcm = [f for f in files if f.lower().endswith(".dcm")]
        use = dcm if len(dcm) > 0 else files

        def _key(fn):
            base = os.path.splitext(fn)[0]
            try:
                return int(base)
            except Exception:
                return base

        use.sort(key=_key)
        out[p] = tuple(os.path.join(folder, f) for f in use)
    return out


TRAIN_PATHS_BY_PATIENT = _list_dicom_paths(TRAIN_DIR, train_patients)




## === cell 10
from tensorflow.keras.utils import Sequence


class IGenerator(Sequence):
    BAD_ID = ["ID00011637202177653955184", "ID00052637202186188008618"]

    def __init__(self, keys, batch_size=32):
        self.keys = [k for k in keys if k not in self.BAD_ID]
        self.batch_size = batch_size

        self.patient_rows = {
            p: train_by_patient[p] for p in self.keys if p in train_by_patient
        }
        self.patient_base = {
            p: train_firstrow_by_patient[p]
            for p in self.keys
            if p in train_firstrow_by_patient
        }

        self.train_paths = {
            p: TRAIN_PATHS_BY_PATIENT.get(p, tuple()) for p in self.keys
        }

        self.valid_patients = [
            p
            for p in self.keys
            if len(self.train_paths.get(p, ())) > 0
            and len(self.patient_rows.get(p, ())) > 0
        ]
        if len(self.valid_patients) == 0:
            self.valid_patients = list(self.keys)

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        x = []
        y_out = []
        tab_out = []

        keys = np.random.choice(self.valid_patients, size=self.batch_size, replace=True)
        for k in keys:
            paths = self.train_paths.get(k, ())
            subp = self.patient_rows.get(k, None)
            base = self.patient_base.get(k, None)
            if (
                (subp is None)
                or (base is None)
                or (len(paths) == 0)
                or (len(subp) == 0)
            ):
                continue

            ridx = np.random.randint(0, len(subp))
            week = float(subp.loc[ridx, "Weeks"])
            fvc = float(subp.loc[ridx, "FVC"])

            pth = paths[np.random.randint(0, len(paths))]
            try:
                img = get_img(pth)
            except Exception:
                continue

            x.append(img)
            y_out.append(fvc)
            tab_out.append(get_tab(base, week_value=week))

        if len(x) == 0:
            x = [np.zeros((512, 512), dtype=np.float32)]
            y_out = [2000.0]
            tab_out = [np.zeros((5,), dtype=np.float32)]

        while len(x) < self.batch_size:
            j = np.random.randint(0, len(x))
            x.append(x[j])
            y_out.append(y_out[j])
            tab_out.append(tab_out[j])

        x = np.array(x, dtype=np.float32)
        y_out = np.array(y_out, dtype=np.float32)
        tab_out = np.array(tab_out, dtype=np.float32)
        x = np.expand_dims(x, axis=-1)

        return (x, tab_out), y_out




## === cell 11
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


def get_model(shape=(512, 512, 1), tab_dim=5):
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

    inp2 = Input(shape=(tab_dim,))
    x = Concatenate()([x, inp2])
    x = Dropout(0.6)(x)
    x = Dense(1)(x)
    return Model([inp, inp2], x)




## === cell 12
model = get_model(tab_dim=5)
model.summary()




## === cell 13
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=["mae"])




## === cell 14
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




## === cell 15
from sklearn.model_selection import train_test_split

tr_p, vl_p = train_test_split(P, shuffle=True, train_size=0.8, random_state=42)




## === cell 16
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




## === cell 17
model.fit(
    IGenerator(keys=tr_p),
    steps_per_epoch=50,
    validation_data=IGenerator(keys=vl_p),
    validation_steps=20,
    callbacks=[er, lr_schedule],
    epochs=30,
    verbose=1,
)




## === cell 18
sub = pd.read_csv(SAMPLE_SUB)
sub.head()




## === cell 19
test = pd.read_csv(TEST_CSV)
test.head()




## === cell 20
test_by_patient = {
    p: test.loc[test.Patient == p, :].iloc[0:1] for p in test.Patient.unique()
}

TEST_PATHS_BY_PATIENT = _list_dicom_paths(TEST_DIR, test.Patient.unique())

from functools import lru_cache

TEST_MAX_SLICES = 12


def _select_paths_deterministic(paths, k=TEST_MAX_SLICES):
    if len(paths) <= k:
        return paths
    idx = np.linspace(0, len(paths) - 1, num=k, dtype=int)
    return tuple(paths[i] for i in idx)


@lru_cache(maxsize=128)
def _get_test_patient_x(patient_id: str):
    paths = TEST_PATHS_BY_PATIENT.get(patient_id, ())
    if len(paths) == 0:
        return None
    paths = _select_paths_deterministic(paths, k=TEST_MAX_SLICES)

    imgs = []
    for pth in paths:
        try:
            imgs.append(get_img(pth))
        except Exception:
            continue
    if len(imgs) == 0:
        return None
    x = np.expand_dims(np.array(imgs, dtype=np.float32), axis=-1)
    return x


@lru_cache(maxsize=4096)
def _predict_patient_week_fvc_and_std(patient_id: str, week: int):
    base = test_by_patient.get(patient_id, None)
    x = _get_test_patient_x(patient_id)

    if (base is None) or (len(base) == 0) or (x is None):
        if base is None or len(base) == 0:
            tab = np.zeros((5,), dtype=np.float32)
        else:
            tab = get_tab(base, week_value=week)

        pred = float(
            model.predict(
                [np.zeros((1, 512, 512, 1), np.float32), tab.reshape(1, -1)], verbose=0
            ).reshape(-1)[0]
        )
        return pred, 0.0

    tab_vec = get_tab(base, week_value=week)
    tab = np.repeat(tab_vec.reshape(1, -1), repeats=len(x), axis=0).astype(
        np.float32, copy=False
    )
    preds = model.predict([x, tab], verbose=0).reshape(-1)
    return float(np.quantile(preds, 0.50)), float(np.std(preds))


def _estimate_global_residual_floor(
    vl_patients, n_patients=25, n_rows_per_patient=5, n_slices_per_patient=12
):
    rng = np.random.RandomState(42)
    pts = [p for p in vl_patients if os.path.isdir(os.path.join(TRAIN_DIR, p))]
    if len(pts) == 0:
        return 70.0
    if len(pts) > n_patients:
        pts = list(rng.choice(pts, size=n_patients, replace=False))

    abs_errs = []
    for p in pts:
        subp = train_by_patient.get(p, None)
        base = train_firstrow_by_patient.get(p, None)
        if subp is None or base is None or len(subp) == 0:
            continue

        paths_all = TRAIN_PATHS_BY_PATIENT.get(p, tuple())
        if len(paths_all) == 0:
            continue

        ridxs = rng.choice(
            np.arange(len(subp)), size=min(n_rows_per_patient, len(subp)), replace=False
        )

        if len(paths_all) > n_slices_per_patient:
            sidx = rng.choice(
                np.arange(len(paths_all)), size=n_slices_per_patient, replace=False
            )
            paths = [paths_all[i] for i in sidx]
        else:
            paths = list(paths_all)

        imgs = []
        for pth in paths:
            try:
                imgs.append(get_img(pth))
            except Exception:
                continue
        if len(imgs) == 0:
            continue
        x = np.expand_dims(np.array(imgs, dtype=np.float32), axis=-1)

        for ridx in ridxs:
            week = float(subp.loc[ridx, "Weeks"])
            y_true = float(subp.loc[ridx, "FVC"])
            tab_vec = get_tab(base, week_value=week)
            tab = np.repeat(tab_vec.reshape(1, -1), repeats=len(x), axis=0).astype(
                np.float32, copy=False
            )
            y_pred_slices = model.predict([x, tab], verbose=0).reshape(-1)
            y_pred = float(np.quantile(y_pred_slices, 0.50))
            abs_errs.append(abs(y_true - y_pred))

    if len(abs_errs) == 0:
        return 70.0

    floor = float(np.quantile(np.array(abs_errs, dtype=np.float32), 0.60))
    return max(70.0, floor)


GLOBAL_SIGMA_FLOOR = _estimate_global_residual_floor(vl_p)
print("Estimated GLOBAL_SIGMA_FLOOR:", GLOBAL_SIGMA_FLOOR)




## === cell 21
sub = sub.copy()
patient_week = sub.Patient_Week.values
out_fvc = np.empty(len(patient_week), dtype=np.float32)
out_conf = np.empty(len(patient_week), dtype=np.float32)

for i, k in enumerate(patient_week):
    p, w = k.split("_")
    w = int(w)

    pred_fvc, pred_std = _predict_patient_week_fvc_and_std(p, w)
    out_fvc[i] = pred_fvc
    out_conf[i] = max(float(pred_std), float(GLOBAL_SIGMA_FLOOR))

sub["FVC"] = out_fvc
sub["Confidence"] = out_conf
sub.head()




## === cell 22
required_cols = ["Patient_Week", "FVC", "Confidence"]
sub_out = sub[required_cols].copy()
sub_out["FVC"] = pd.to_numeric(sub_out["FVC"], errors="coerce").astype(np.float32)
sub_out["Confidence"] = pd.to_numeric(sub_out["Confidence"], errors="coerce").astype(
    np.float32
)

sub_out["FVC"] = sub_out["FVC"].fillna(2000.0).astype(np.float32)
sub_out["Confidence"] = sub_out["Confidence"].fillna(70.0).astype(np.float32)

sub_out["Confidence"] = np.maximum(sub_out["Confidence"].values, 70.0).astype(
    np.float32
)

sub_out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_out.shape)
print(sub_out.head())
