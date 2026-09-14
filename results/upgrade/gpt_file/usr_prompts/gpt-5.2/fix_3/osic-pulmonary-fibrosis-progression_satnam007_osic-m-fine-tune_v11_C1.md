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

-6.930245896468549

# 6. Current score

-8.76216

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -7.79218) has done: 'I fix the environment/runtime issues that currently prevent the notebook from running: TensorFlow import/protobuf crash, deprecated `DataFrame.append`, and Keras optimizer argument changes. I also fix shape/type bugs in the custom loss/metric (your model outputs 3 values but training labels were 1D), so training and inference run end-to-end without changing the model architecture or training loop semantics. Then I ensure the submission is created strictly in the required format with columns `Patient_Week,FVC,Confidence`, and remove/guard the extra cell that reads a non-existent external submission file. These changes are necessary for producing a valid `submission.csv` and should move the score from “not yielded” to a real score without altering the intended approach.'
- What this solution (achieved -8.76216) has done: 'I fix the TensorFlow import crash by switching the protobuf implementation setting to `cpp` (and falling back to `python` only if needed), which resolves the `MessageFactory.GetPrototype` error in Kaggle’s environment. I also add a small path fallback so `ROOT` works whether the dataset is mounted under `../input/...` or `/kaggle/input/...`, without changing any modeling logic. Finally, I make the submission generation a bit more robust by ensuring predictions produce non-negative confidence (via absolute diff) and by sorting/deduplicating `Patient_Week` before writing, which is score-neutral but prevents format/alignment issues.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = os.environ.get(
    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "cpp"
)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")



## === cell 1
import numpy as np
import pandas as pd
import pydicom
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold



## === cell 2
try:
    import tensorflow as tf
except Exception as e:
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    import tensorflow as tf

import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
import tensorflow.keras.models as M




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/4259247738.py in <cell line: 0>()
      2 try:
----> 3     import tensorflow as tf
      4 except Exception as e:

/usr/local/lib/python3.11/dist-packages/tensorflow/__init__.py in <module>
     48 
---> 49 from tensorflow._api.v2 import __internal__
     50 from tensorflow._api.v2 import __operators__

/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/__init__.py in <module>
      7 
----> 8 from tensorflow._api.v2.__internal__ import autograph
      9 from tensorflow._api.v2.__internal__ import decorator

/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/autograph/__init__.py in <module>
      7 
----> 8 from tensorflow.python.autograph.core.ag_ctx import control_status_ctx # line: 34
      9 from tensorflow.python.autograph.impl.api import tf_convert # line: 493

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/ag_ctx.py in <module>
     20 
---> 21 from tensorflow.python.autograph.utils import ag_logging
     22 from tensorflow.python.util.tf_export import tf_export

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/__init__.py in <module>
     16 
---> 17 from tensorflow.python.autograph.utils.context_managers import control_dependency_on_returns
     18 from tensorflow.python.autograph.utils.misc import alias_tensors

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/context_managers.py in <module>
     18 
---> 19 from tensorflow.python.framework import ops
     20 from tensorflow.python.ops import tensor_array_ops

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in <module>
     32 from google.protobuf import message
---> 33 from tensorflow.core.framework import attr_value_pb2
     34 from tensorflow.core.framework import full_type_pb2

/usr/local/lib/python3.11/dist-packages/tensorflow/core/framework/attr_value_pb2.py in <module>
      4 """Generated protocol buffer code."""
----> 5 from google.protobuf.internal import builder as _builder
      6 from google.protobuf import descriptor as _descriptor

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/builder.py in <module>
     17 from google.protobuf.internal import enum_type_wrapper
---> 18 from google.protobuf.internal import python_message
     19 from google.protobuf import message as _message

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/python_message.py in <module>
     37 
---> 38 from google.protobuf import descriptor as descriptor_mod
     39 from google.protobuf import message as message_mod

/usr/local/lib/python3.11/dist-packages/google/protobuf/descriptor.py in <module>
     28   if _message is None:
---> 29     from google.protobuf.pyext import _message
     30   _USE_C_DESCRIPTORS = True

ImportError: cannot import name '_message' from 'google.protobuf.pyext' (/usr/local/lib/python3.11/dist-packages/google/protobuf/pyext/__init__.py)

During handling of the above exception, another exception occurred:

ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/4259247738.py in <cell line: 0>()
      5     # Last resort fallback; keeps execution going on images where cpp impl is unavailable.
      6     os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
----> 7     import tensorflow as tf
      8 
      9 import tensorflow.keras.backend as K

/usr/local/lib/python3.11/dist-packages/tensorflow/__init__.py in <module>
     47 _tf2.enable()
     48 
---> 49 from tensorflow._api.v2 import __internal__
     50 from tensorflow._api.v2 import __operators__
     51 from tensorflow._api.v2 import audio

/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/__init__.py in <module>
      6 import sys as _sys
      7 
----> 8 from tensorflow._api.v2.__internal__ import autograph
      9 from tensorflow._api.v2.__internal__ import decorator
     10 from tensorflow._api.v2.__internal__ import dispatch

/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/autograph/__init__.py in <module>
      6 import sys as _sys
      7 
----> 8 from tensorflow.python.autograph.core.ag_ctx import control_status_ctx # line: 34
      9 from tensorflow.python.autograph.impl.api import tf_convert # line: 493

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/ag_ctx.py in <module>
     19 import threading
     20 
---> 21 from tensorflow.python.autograph.utils import ag_logging
     22 from tensorflow.python.util.tf_export import tf_export
     23 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/__init__.py in <module>
     15 """Utility module that contains APIs usable in the generated code."""
     16 
---> 17 from tensorflow.python.autograph.utils.context_managers import control_dependency_on_returns
     18 from tensorflow.python.autograph.utils.misc import alias_tensors
     19 from tensorflow.python.autograph.utils.tensor_list import dynamic_list_append

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/context_managers.py in <module>
     17 import contextlib
     18 
---> 19 from tensorflow.python.framework import ops
     20 from tensorflow.python.ops import tensor_array_ops
     21 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in <module>
     31 
     32 from google.protobuf import message
---> 33 from tensorflow.core.framework import attr_value_pb2
     34 from tensorflow.core.framework import full_type_pb2
     35 from tensorflow.core.framework import function_pb2

/usr/local/lib/python3.11/dist-packages/tensorflow/core/framework/attr_value_pb2.py in <module>
      3 # source: tensorflow/core/framework/attr_value.proto
      4 """Generated protocol buffer code."""
----> 5 from google.protobuf.internal import builder as _builder
      6 from google.protobuf import descriptor as _descriptor
      7 from google.protobuf import descriptor_pool as _descriptor_pool

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/builder.py in <module>
     16 
     17 from google.protobuf.internal import enum_type_wrapper
---> 18 from google.protobuf.internal import python_message
     19 from google.protobuf import message as _message
     20 from google.protobuf import reflection as _reflection

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/python_message.py in <module>
     36 import weakref
     37 
---> 38 from google.protobuf import descriptor as descriptor_mod
     39 from google.protobuf import message as message_mod
     40 from google.protobuf import text_format

/usr/local/lib/python3.11/dist-packages/google/protobuf/descriptor.py in <module>
     27   # TODO: Remove this import after fix api_implementation
     28   if _message is None:
---> 29     from google.protobuf.pyext import _message
     30   _USE_C_DESCRIPTORS = True
     31 

ImportError: cannot import name '_message' from 'google.protobuf.pyext' (/usr/local/lib/python3.11/dist-packages/google/protobuf/pyext/__init__.py)

## === cell 3
def seed_everything(seed=2020):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    return seed




## === cell 4
ROOT = "../input/osic-pulmonary-fibrosis-progression"
if not os.path.exists(ROOT):
    ROOT = "/kaggle/input/osic-pulmonary-fibrosis-progression"

tr = pd.read_csv(f"{ROOT}/train.csv")
tr.drop_duplicates(keep=False, inplace=True, subset=["Patient", "Weeks"])
chunk = pd.read_csv(f"{ROOT}/test.csv")

print("add infos")
sub = pd.read_csv(f"{ROOT}/sample_submission.csv")
sub["Patient"] = sub["Patient_Week"].apply(lambda x: x.split("_")[0])
sub["Weeks"] = sub["Patient_Week"].apply(lambda x: int(x.split("_")[-1]))
sub = sub[["Patient", "Weeks", "Confidence", "Patient_Week"]]
sub = sub.merge(chunk.drop("Weeks", axis=1), on="Patient")



## === cell 5
tr["WHERE"] = "train"
chunk["WHERE"] = "val"
sub["WHERE"] = "test"
data = pd.concat([tr, chunk, sub], axis=0, ignore_index=True)



## === cell 6
print(tr.shape, chunk.shape, sub.shape, data.shape)
print(
    tr.Patient.nunique(),
    chunk.Patient.nunique(),
    sub.Patient.nunique(),
    data.Patient.nunique(),
)



## === cell 7
data["min_week"] = data["Weeks"]
data.loc[data.WHERE == "test", "min_week"] = np.nan
data["min_week"] = data.groupby("Patient")["min_week"].transform("min")



## === cell 8
base = data.loc[data.Weeks == data.min_week]
base = base[["Patient", "FVC"]].copy()
base.columns = ["Patient", "min_FVC"]
base["nb"] = 1
base["nb"] = base.groupby("Patient")["nb"].transform("cumsum")
base = base[base.nb == 1]
base.drop("nb", axis=1, inplace=True)



## === cell 9
data = data.merge(base, on="Patient", how="left")
data["base_week"] = data["Weeks"] - data["min_week"]
del base



## === cell 10
COLS = ["Sex", "SmokingStatus"]
FE = []
for col in COLS:
    mods = pd.Series(data[col].fillna("Unknown").unique()).tolist()
    for mod in mods:
        FE.append(mod)
        data[mod] = (data[col].fillna("Unknown") == mod).astype(int)




## === cell 11
def _minmax(s):
    smin, smax = s.min(), s.max()
    denom = (smax - smin) if (smax - smin) != 0 else 1.0
    return (s - smin) / denom


data["age"] = _minmax(data["Age"])
data["BASE"] = _minmax(data["min_FVC"])
data["week"] = _minmax(data["base_week"])
data["percent"] = _minmax(data["Percent"])
FE += ["age", "percent", "week", "BASE"]



## === cell 12
tr = data.loc[data.WHERE == "train"].copy()
chunk = data.loc[data.WHERE == "val"].copy()
sub = data.loc[data.WHERE == "test"].copy()
del data



## === cell 13
if len(FE) != 9:
    raise ValueError(f"Expected 9 features for model input, got {len(FE)}: {FE}")



## === cell 14
SEED = seed_everything(42)
NFOLD = 5
EPOCHS = 800
BATCH_SIZE = 128

M_LOSS = 0.775
LR = 0.1005
DECAY = 0.01

kf = KFold(n_splits=NFOLD, shuffle=True, random_state=SEED)




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2523130216.py in <cell line: 0>()
----> 1 SEED = seed_everything(42)
      2 NFOLD = 5
      3 EPOCHS = 800
      4 BATCH_SIZE = 128
      5 

/tmp/ipykernel_11/594212371.py in seed_everything(seed)
      3     os.environ["PYTHONHASHSEED"] = str(seed)
      4     np.random.seed(seed)
----> 5     tf.random.set_seed(seed)
      6     return seed
      7 

NameError: name 'tf' is not defined

## === cell 15
def make_y_fvc(y1d):
    y1d = np.asarray(y1d).reshape(-1, 1).astype("float32")
    return np.repeat(y1d, 3, axis=1)




## === cell 16
y = make_y_fvc(tr["FVC"].values)
z = tr[FE].values.astype("float32")
ze = sub[FE].values.astype("float32")
pe = np.zeros((ze.shape[0], 3), dtype="float32")
pred = np.zeros((z.shape[0], 3), dtype="float32")
delta = np.zeros((z.shape[0], 3), dtype="float32")



## === cell 17
C1, C2 = tf.constant(70.0, dtype="float32"), tf.constant(1000.0, dtype="float32")


def score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]

    sigma_clip = tf.maximum(sigma, C1)
    delta_ = tf.abs(y_true[:, 0] - fvc_pred)
    delta_ = tf.minimum(delta_, C2)
    sq2 = tf.sqrt(tf.constant(2.0, dtype=tf.float32))
    metric = (delta_ / sigma_clip) * sq2 + tf.math.log(sigma_clip * sq2)
    return K.mean(metric)


def qloss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    qs = [0.2, 0.5, 0.8]
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q * e, (q - 1) * e)
    return K.mean(v)


def mloss(_lambda):
    def loss(y_true, y_pred):
        return _lambda * qloss(y_true, y_pred) + (1 - _lambda) * score(y_true, y_pred)

    return loss


def make_model():
    z_in = L.Input((9,), name="Patient")
    x = L.Dense(100, activation="relu", name="d1")(z_in)
    x = L.Dense(100, activation="relu", name="d2")(x)
    p1 = L.Dense(3, activation="linear", name="p1")(x)
    p2 = L.Dense(3, activation="relu", name="p2")(x)
    preds = L.Lambda(lambda xx: xx[0] + tf.cumsum(xx[1], axis=1), name="preds")(
        [p1, p2]
    )

    model = M.Model(z_in, preds, name="ANN")

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=LR,
        decay_steps=1000,
        decay_rate=(1.0 / (1.0 + DECAY)),
        staircase=False,
    )
    opt = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, amsgrad=False
    )
    model.compile(loss=mloss(M_LOSS), optimizer=opt, metrics=[score])
    return model




## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/34727258.py in <cell line: 0>()
----> 1 C1, C2 = tf.constant(70.0, dtype="float32"), tf.constant(1000.0, dtype="float32")
      2 
      3 
      4 def score(y_true, y_pred):
      5     y_true = tf.cast(y_true, tf.float32)

NameError: name 'tf' is not defined

## === cell 18
net = make_model()
print(net.summary())
print(net.count_params())



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1637008597.py in <cell line: 0>()
----> 1 net = make_model()
      2 print(net.summary())
      3 print(net.count_params())
      4 

NameError: name 'make_model' is not defined

## === cell 19
import time

t0 = time.time()

cnt = 0
train = []
val = []

for tr_idx, val_idx in kf.split(z):
    cnt += 1
    print(f"FOLD {cnt}")

    net = make_model()

    net.fit(
        z[tr_idx],
        y[tr_idx],
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(z[val_idx], y[val_idx]),
        verbose=0,
    )

    print("train", net.evaluate(z[tr_idx], y[tr_idx], verbose=0, batch_size=BATCH_SIZE))
    print("val", net.evaluate(z[val_idx], y[val_idx], verbose=0, batch_size=BATCH_SIZE))
    pred[val_idx] = net.predict(z[val_idx], batch_size=BATCH_SIZE, verbose=0)
    print()

    pe += net.predict(ze, batch_size=BATCH_SIZE, verbose=0) / NFOLD
    delta += net.predict(z, batch_size=BATCH_SIZE, verbose=0) / NFOLD

print(f"Training+Inference time: {time.time()-t0:.1f}s")



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/783474039.py in <cell line: 0>()
      7 val = []
      8 
----> 9 for tr_idx, val_idx in kf.split(z):
     10     cnt += 1
     11     print(f"FOLD {cnt}")

NameError: name 'kf' is not defined

## === cell 20
y1d = tr["FVC"].values.astype("float32")
sigma_opt = mean_absolute_error(y1d, pred[:, 1])
unc = pred[:, 2] - pred[:, 0]
sigma_mean = np.mean(unc)
print(sigma_opt, sigma_mean)



## === cell 21
o_clipped = np.maximum(delta[:, 2] - delta[:, 0], 70)
delta_abs = np.minimum(np.abs(delta[:, 1] - y1d), 1000)
sqrt2 = np.sqrt(2.0)
score_arr = (-(sqrt2 * delta_abs) / (o_clipped)) - np.log(sqrt2 * o_clipped)
logL_Score = float(np.mean(score_arr))



## === cell 22
print(
    "we are using fix seed value always to avoid RANDOMIZATION (NEED TO GET SAME RESULT)"
)
print("Seed value          =", SEED)
print("Batch size          =", BATCH_SIZE)
print("Number of epochs    =", EPOCHS)
print("\nmean_absolute_error =", sigma_opt)
print("unc_mean            =", float(unc.mean()))
print()
print("Log_laplace_Scores  =", logL_Score)
print()
print("unc_min             =", float(unc.min()))
print("unc_max             =", float(unc.max()))
print("unc_nonneg_frac     =", float((unc >= 0).mean()))



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/941333596.py in <cell line: 0>()
      2     "we are using fix seed value always to avoid RANDOMIZATION (NEED TO GET SAME RESULT)"
      3 )
----> 4 print("Seed value          =", SEED)
      5 print("Batch size          =", BATCH_SIZE)
      6 print("Number of epochs    =", EPOCHS)

NameError: name 'SEED' is not defined

## === cell 23
stats = pd.DataFrame()
index = 0



## === cell 24
data_stats = [
    [
        index,
        logL_Score,
        sigma_opt,
        float(unc.mean()),
        float(unc.min()),
        float(unc.max()),
        float((unc >= 0).mean()),
        BATCH_SIZE,
        EPOCHS,
        NFOLD,
        M_LOSS,
        LR,
        DECAY,
        SEED,
    ]
]
columns = [
    "S.No",
    "Score",
    "meanAbseErr",
    "unc.mean",
    "unc.min",
    "unc.max",
    "(unc>=0)",
    "Bsize",
    "epoch",
    "NFOLD",
    "M_LOSS",
    "LR",
    "DECAY",
    "seed",
]
kernal_stats = pd.DataFrame(data_stats, columns=columns)
kernal_stats



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4179587551.py in <cell line: 0>()
      8         float(unc.max()),
      9         float((unc >= 0).mean()),
---> 10         BATCH_SIZE,
     11         EPOCHS,
     12         NFOLD,

NameError: name 'BATCH_SIZE' is not defined

## === cell 25
stats = pd.concat([stats, kernal_stats], ignore_index=True)
stats.to_csv("kernal.csv", index=False)
index += 1
stats



## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1530296469.py in <cell line: 0>()
----> 1 stats = pd.concat([stats, kernal_stats], ignore_index=True)
      2 stats.to_csv("kernal.csv", index=False)
      3 index += 1
      4 stats
      5 

NameError: name 'kernal_stats' is not defined

## === cell 26
plt.hist(unc, bins=50)
plt.title("uncertainty in prediction")
plt.show()



## === cell 27
try:
    import seaborn as sns

    sns.histplot(unc, bins=50, kde=True)
    plt.title("uncertainty in prediction (kde)")
    plt.show()
except Exception as e:
    print("Skipping seaborn plots:", repr(e))



## === cell 28
sub["FVC1"] = pe[:, 1]

sub["Confidence1"] = np.abs(pe[:, 2] - pe[:, 0])



## === cell 29
subm = sub[["Patient_Week", "FVC", "Confidence", "FVC1", "Confidence1"]].copy()



## === cell 30
subm.loc[~subm.FVC1.isnull()].head(1)



## === cell 31
subm.loc[~subm.FVC1.isnull(), "FVC"] = subm.loc[~subm.FVC1.isnull(), "FVC1"]
if sigma_mean < 70:
    subm["Confidence"] = sigma_opt
else:
    subm.loc[~subm.FVC1.isnull(), "Confidence"] = subm.loc[
        ~subm.FVC1.isnull(), "Confidence1"
    ]

subm["Confidence"] = np.maximum(subm["Confidence"].astype("float32"), 70.0)



## === cell 32
subm.head(1)



## === cell 33
plt.hist(subm.FVC, bins=50)
plt.title("FVC")
plt.show()

plt.hist(subm.FVC1, bins=50)
plt.title("FVC1")
plt.show()



## === cell 34
plt.hist(subm.Confidence, bins=50)
plt.title("Confidence")
plt.show()

plt.hist(subm.Confidence1, bins=50)
plt.title("Confidence1")
plt.show()



## === cell 35
subm.describe().T



## === cell 36
otest = pd.read_csv(f"{ROOT}/test.csv")
for i in range(len(otest)):
    pw = otest.Patient[i] + "_" + str(otest.Weeks[i])
    subm.loc[subm["Patient_Week"] == pw, "FVC"] = float(otest.FVC[i])
    subm.loc[subm["Patient_Week"] == pw, "Confidence"] = 70.0



## === cell 37
submission = subm[["Patient_Week", "FVC", "Confidence"]].copy()
submission["FVC"] = submission["FVC"].astype("float32")
submission["Confidence"] = submission["Confidence"].astype("float32")

submission = (
    submission.drop_duplicates(subset=["Patient_Week"], keep="first")
    .sort_values("Patient_Week")
    .reset_index(drop=True)
)

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())



## === cell 38
pass
