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

-7.6716

# 6. Current score

-10.32044

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -9.7172) has done: 'I first fix the environment/runtime issues: the protobuf-related import crash (caused by an incompatible protobuf version with something in the TF stack) by forcing the pure-Python protobuf implementation before importing TensorFlow-related modules. Then I update the deprecated `fit_generator` call to `fit` (Keras now trains on `Sequence` via `fit`) so training runs. Finally, I fix two logic bugs that hurt score: the sex string case mismatch in `get_tab` and the `Confidence` computation using the wrong variable (`P_test` dict instead of per-row Percent), replacing it with a stable per-patient confidence derived from the model’s slice-prediction spread and clipped to the competition’s minimum-meaningful scale (>=70), which should move the score upward toward the target.'
- What this solution (achieved -10.47239) has done: 'The timeout is dominated by extremely expensive DICOM decoding/resizing inside the Python `Sequence` for every batch (no caching, no parallelism), plus Python-level loops and repeated filesystem scans. I keep the exact model, loss, and training semantics, but make data loading dramatically cheaper by (1) pre-indexing DICOM file lists once, (2) caching decoded+resized slices in memory per worker process (so repeated sampling of the same slice becomes near-free), and (3) using Keras `Sequence` multiprocessing (`workers`, `use_multiprocessing`, `max_queue_size`) to overlap CPU decoding with GPU/TF compute. I also speed up submission creation by vectorizing the loop over `Patient_Week` (identical math), and avoid repeated `test.loc[...]` lookups by precomputing per-patient scalars once.'
- What this solution (achieved -10.40373) has done: 'The timeout is dominated by (1) an unnecessary `pip install protobuf` at runtime, (2) very expensive DICOM decoding + 512×512 resizing done inside the Python `Sequence` without prefetching, and (3) test-time inference running `model.predict` on large per-patient stacks without batching control. I keep the exact same model, loss, training loop semantics, and prediction logic, but speed up data loading by switching to `stop_before_pixels=True` where possible and—critically—decoding pixels via `pydicom.dcmread(..., specific_tags=...)` to minimize header parsing, plus a faster LRU cache and removing per-sample exception printing. I also enable `Sequence` multiprocessing workers/prefetch during `fit` (same sampling logic; just pipelined IO), and explicitly control test-time prediction batch size to avoid slow single-batch execution and reduce peak memory stalls. Finally, I eliminate the runtime protobuf downgrade (it can consume minutes) since your environment already has TensorFlow 2.18 with protobuf 6; forcing pure-Python protobuf also slows TF.'
- What this solution (achieved -10.32044) has done: 'The timeout is dominated by repeatedly decoding/resizing DICOMs inside the Python `Sequence` (training) and then loading *all* test slices for each patient at once (inference), causing heavy I/O + CPU + large temporary arrays. I keep the same model, loss, and training semantics, but make data loading faster and less wasteful by (1) pre-indexing slice file lists only for the patients actually used by each generator, (2) switching DICOM decoding to `pydicom.dcmread(..., force=False)` without PixelData tag filtering overhead (still exact same pixels), (3) using a larger `max_queue_size/workers` for Keras to overlap preprocessing with training, and (4) streaming test inference in batches per patient instead of materializing all slices in memory at once. These changes are equivalent in outputs (same random sampling, same preprocessing math, same predictions) while substantially reducing overhead and peak memory.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import sys
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
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")




## === cell 2
train.head()




## === cell 3
train.SmokingStatus.unique()




## === cell 4
def get_tab(df):
    vector = [(df.Age.values[0] - 30) / 30]

    sex = str(df.Sex.values[0]).strip().lower()
    if sex == "male":
        vector.append(0)
    else:
        vector.append(1)

    ss = str(df.SmokingStatus.values[0]).strip()
    if ss == "Never smoked":
        vector.extend([0, 0])
    elif ss == "Ex-smoker":
        vector.extend([1, 1])
    elif ss == "Currently smokes":
        vector.extend([0, 1])
    else:
        vector.extend([1, 0])
    return np.array(vector, dtype=np.float32)




## === cell 5
A = {}
TAB = {}
P = []

for p, sub in tqdm(train.groupby("Patient", sort=False), total=train.Patient.nunique()):
    fvc = sub.FVC.values
    weeks = sub.Weeks.values
    c = np.vstack([weeks, np.ones(len(weeks))]).T
    a, b = np.linalg.lstsq(c, fvc, rcond=None)[0]

    A[p] = a
    TAB[p] = get_tab(sub)
    P.append(p)




## === cell 6
from functools import lru_cache


@lru_cache(maxsize=16384)
def _get_img_cached(path: str) -> np.ndarray:
    d = pydicom.dcmread(
        path,
        force=True,  # keep robustness identical to original
        stop_before_pixels=False,  # must read pixels
    )
    arr = d.pixel_array.astype(np.float32)
    arr = arr / (2**11)
    arr = cv2.resize(arr, (512, 512), interpolation=cv2.INTER_AREA).astype(np.float32)
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

        self.train_data = {}
        base = "../input/osic-pulmonary-fibrosis-progression/train"
        for p in self.keys:
            try:
                self.train_data[p] = tuple(os.listdir(f"{base}/{p}/"))
            except FileNotFoundError:
                self.train_data[p] = tuple()

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        x = []
        a, tab = [], []
        keys = np.random.choice(self.keys, size=self.batch_size)
        base = "../input/osic-pulmonary-fibrosis-progression/train"
        for k in keys:
            files = self.train_data.get(k, ())
            if not files:
                continue
            try:
                i = np.random.choice(files, size=1)[0]
                img = get_img(f"{base}/{k}/{i}")
                x.append(img)
                a.append(self.a[k])
                tab.append(self.tab[k])
            except Exception:
                continue

        x = np.array(x, dtype=np.float32)
        a = np.array(a, dtype=np.float32)
        tab = np.array(tab, dtype=np.float32)

        x = np.expand_dims(x, axis=-1)

        return (x, tab), a




## === cell 8
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Activation,
    Input,
    BatchNormalization,
    GlobalAveragePooling2D,
    Add,
    Conv2D,
    AveragePooling2D,
    Concatenate,
)
from tensorflow.keras import Model


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


plt.figure(figsize=(10, 7))
_lrfn = build_lrfn()
plt.plot([i for i in range(35)], [_lrfn(i) for i in range(35)])




## === cell 12
from sklearn.model_selection import train_test_split

tr_p, vl_p = train_test_split(P, shuffle=True, train_size=0.8, random_state=42)




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
train_gen = IGenerator(keys=tr_p, a=A, tab=TAB)
val_gen = IGenerator(keys=vl_p, a=A, tab=TAB)

model.fit(
    train_gen,
    steps_per_epoch=50,
    validation_data=val_gen,
    validation_steps=20,
    callbacks=[er, lr_schedule],
    epochs=30,
    workers=2,
    use_multiprocessing=False,
    max_queue_size=16,
)




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1737131477.py in <cell line: 0>()
      4 # Speed fix: overlap CPU DICOM decode/resize with GPU/TF training via workers+prefetch.
      5 # This does not change sampling, batches, model, or loss; it only pipelines input.
----> 6 model.fit(
      7     train_gen,
      8     steps_per_epoch=50,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    117             return fn(*args, **kwargs)
    118         except Exception as e:
--> 119             filtered_tb = _process_traceback_frames(e.__traceback__)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`

TypeError: TensorFlowTrainer.fit() got an unexpected keyword argument 'workers'

## === cell 15
sub = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/sample_submission.csv")
sub.head()




## === cell 16
test = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
test.head()




## === cell 17
test_idx = test.set_index("Patient")
test_tab = {p: get_tab(test.loc[test.Patient == p, :]) for p in test.Patient.unique()}

A_test, B_test, P_test = {}, {}, {}
STD, WEEK = {}, {}

base_test = "../input/osic-pulmonary-fibrosis-progression/test"

for p in test.Patient.unique():
    files = os.listdir(f"{base_test}/{p}/")

    tab_vec = test_tab[p].astype(np.float32)

    preds = []
    bs = 32
    for i0 in range(0, len(files), bs):
        batch_files = files[i0 : i0 + bs]
        xb = [get_img(f"{base_test}/{p}/{fn}") for fn in batch_files]
        xb = np.expand_dims(np.array(xb, dtype=np.float32), axis=-1)

        tab = np.repeat(tab_vec[None, :], repeats=xb.shape[0], axis=0).astype(
            np.float32
        )

        pb = model.predict([xb, tab], verbose=0, batch_size=32).reshape(-1)
        preds.append(pb)

    _a = np.concatenate(preds, axis=0) if preds else np.array([], dtype=np.float32)
    a = np.quantile(_a, 0.75) if _a.size else 0.0
    std = float(np.std(_a)) if _a.size else 0.0

    weeks0 = int(test_idx.loc[p, "Weeks"])
    fvc0 = float(test_idx.loc[p, "FVC"])
    pct0 = float(test_idx.loc[p, "Percent"])

    A_test[p] = float(a)
    B_test[p] = float(fvc0 - a * weeks0)
    P_test[p] = pct0
    STD[p] = std
    WEEK[p] = weeks0




## === cell 18
pw = sub["Patient_Week"].astype(str)
split = pw.str.split("_", n=1, expand=True)
p_arr = split[0].values
w_arr = split[1].astype(np.int32).values

A_arr = np.array([A_test[p] for p in p_arr], dtype=np.float32)
B_arr = np.array([B_test[p] for p in p_arr], dtype=np.float32)
W0_arr = np.array([WEEK[p] for p in p_arr], dtype=np.int32)
STD_arr = np.array([STD[p] for p in p_arr], dtype=np.float32)

fvc_pred = A_arr * w_arr.astype(np.float32) + B_arr
dt = np.abs(w_arr - W0_arr).astype(np.float32)
conf = 70.0 + (STD_arr * dt * 10.0)
conf = np.where(np.isfinite(conf), conf, 200.0).astype(np.float32)
conf = np.clip(conf, 70.0, 1000.0)

sub["FVC"] = fvc_pred
sub["Confidence"] = conf




## === cell 19
sub.head()




## === cell 20
out = sub[["Patient_Week", "FVC", "Confidence"]].copy()
out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", out.shape)
print(out.head())
