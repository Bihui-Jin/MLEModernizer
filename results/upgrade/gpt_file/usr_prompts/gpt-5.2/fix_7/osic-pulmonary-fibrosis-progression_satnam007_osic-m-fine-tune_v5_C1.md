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

-6.950892286071913

# 6. Current score

-24.65932

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -7.99068) has done: 'I fix the environment/runtime issues that prevent TensorFlow/Keras from importing and the pipeline from completing (protobuf/TensorFlow import crash, pandas `append` removal, Keras optimizer argument changes, missing `sns`, and references to nonexistent external files). I keep the model, loss, and training loop logic the same, but make the minimum compatibility edits so it runs on the current Kaggle Python stack and writes a valid `submission.csv` with the exact required columns. I also ensure the submission is built by merging predictions back onto the original `sample_submission.csv` order so `Patient_Week` is guaranteed present and aligned. No score-tuning changes are introduced beyond making the intended model actually train and generate predictions.'
- What this solution (achieved -7.99068) has done: 'I fix the TensorFlow import crash by removing the protobuf “python” implementation override and instead pinning a compatible implementation at runtime (and fall back cleanly if TF is unavailable). Then I keep the same model/loss/training loop but correct a key inference-time issue: baseline test rows should keep the true baseline FVC while keeping a valid confidence (>=70) rather than forcing Confidence=0.1 (which is clipped to 70 in the metric anyway and can hurt numerical stability). Finally, I ensure the script always writes a valid `submission.csv` with the required columns in the sample submission order.'
- What this solution (achieved -7.99068) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing protobuf to use the pure-Python implementation before importing TensorFlow, which is the minimal change needed to run end-to-end in the Kaggle environment. I also make the TF import guarded and fail-fast with a clear error if it still can’t load, instead of crashing mid-notebook. The rest of the pipeline (features, model, loss, training loop, and submission building) is preserved to keep evaluation semantics the same and to let the model train properly (which should improve score versus the current broken run). Finally, I keep the submission writing and alignment logic unchanged so a valid `submission.csv` is always produced.'
- What this solution (achieved -7.99068) has done: 'I fix the TensorFlow import crash by removing the protobuf pure-Python override that triggers the `MessageFactory.GetPrototype` error in this Kaggle image, and instead just set safe TF runtime flags plus deterministic seeds. I keep the model, loss, folds, epochs, and submission-building logic the same, but add a small, score-neutral safeguard to ensure predicted sigma is always valid (Confidence already clipped to 70 for the metric). Finally, I keep the output exactly aligned to `sample_submission.csv` and always write a valid `submission.csv` with the required columns.'
- What this solution (achieved -7.99068) has done: 'I fix the TensorFlow import crash caused by an incompatible protobuf API by forcing protobuf to use the pure-Python implementation *before* importing TensorFlow, and by clearing any already-imported protobuf modules to ensure the setting takes effect. I also add a safe fallback to choose the correct OSIC dataset root path that exists in this environment (without changing any filenames or I/O outputs). The model architecture, loss, training loop, and submission-building logic remain the same. These changes should both unblock end-to-end execution and allow the intended training/inference to run, improving score from the current broken state toward your target.'
- What this solution (achieved -24.65932) has done: 'I fix the TensorFlow import crash by removing the protobuf pure-Python override (which triggers the `MessageFactory.GetPrototype` error on this Kaggle stack) and instead forcing protobuf to use the C++ implementation, then importing TF normally. I keep the model, loss, folds, and training loop unchanged, but I add a minimal learning-rate decay callback using your existing `DECAY` variable (currently unused) to nudge performance upward toward your target without changing the core approach. I also add a tiny numeric safeguard to ensure predicted uncertainty is strictly positive before clipping to 70 (score-neutral in most cases, but prevents rare invalid sigmas). The pipeline still write a valid `submission.csv` with the exact required columns and order.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)



## === cell 1
import sys
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

for m in list(sys.modules.keys()):
    if m.startswith("google.protobuf"):
        del sys.modules[m]



## === cell 2
try:
    import tensorflow as tf
    import tensorflow.keras.backend as K
    import tensorflow.keras.layers as L
    import tensorflow.keras.models as M
except Exception as e:
    raise RuntimeError(
        "TensorFlow failed to import in this Kaggle environment. "
        "This script attempts to avoid common TF/protobuf incompatibilities by forcing "
        "the protobuf C++ implementation before importing TensorFlow. Full exception attached."
    ) from e




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/3618347994.py in <cell line: 0>()
      1 try:
----> 2     import tensorflow as tf
      3     import tensorflow.keras.backend as K

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

The above exception was the direct cause of the following exception:

RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/3618347994.py in <cell line: 0>()
      5     import tensorflow.keras.models as M
      6 except Exception as e:
----> 7     raise RuntimeError(
      8         "TensorFlow failed to import in this Kaggle environment. "
      9         "This script attempts to avoid common TF/protobuf incompatibilities by forcing "

RuntimeError: TensorFlow failed to import in this Kaggle environment. This script attempts to avoid common TF/protobuf incompatibilities by forcing the protobuf C++ implementation before importing TensorFlow. Full exception attached.

## === cell 3
def seed_everything(seed=2020):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    return seed




## === cell 4
ROOT_CANDIDATES = [
    "../input/osic-pulmonary-fibrosis-progression",
    "/kaggle/input/osic-pulmonary-fibrosis-progression",
]
ROOT = None
for c in ROOT_CANDIDATES:
    if os.path.exists(os.path.join(c, "train.csv")):
        ROOT = c
        break
if ROOT is None:
    ROOT = "../input/osic-pulmonary-fibrosis-progression"

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

data = pd.concat([tr, chunk, sub], ignore_index=True, sort=False)



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
    for mod in data[col].dropna().unique():
        FE.append(mod)
        data[mod] = (data[col] == mod).astype(int)




## === cell 11
def _minmax(s):
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return s * 0.0
    return (s - mn) / (mx - mn)


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
print("Number of engineered features:", len(FE))
print("Features:", FE)



## === cell 14
SEED = seed_everything(42)
NFOLD = 5
EPOCHS = 1020
BATCH_SIZE = 128

M_LOSS = 0.775
LR = 0.1
DECAY = 0.01

kf = KFold(n_splits=NFOLD, shuffle=True, random_state=SEED)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1317564279.py in <cell line: 0>()
----> 1 SEED = seed_everything(42)
      2 NFOLD = 5
      3 EPOCHS = 1020
      4 BATCH_SIZE = 128
      5 

/tmp/ipykernel_11/3724837847.py in seed_everything(seed)
      3     os.environ["PYTHONHASHSEED"] = str(seed)
      4     np.random.seed(seed)
----> 5     tf.random.set_seed(seed)
      6     try:
      7         tf.config.experimental.enable_op_determinism()

NameError: name 'tf' is not defined

## === cell 15
y = tr["FVC"].values.astype("float32")
z = tr[FE].values.astype("float32")
ze = sub[FE].values.astype("float32")

pe = np.zeros((ze.shape[0], 3), dtype="float32")
pred = np.zeros((z.shape[0], 3), dtype="float32")
delta_pred_all = np.zeros((z.shape[0], 3), dtype="float32")



## === cell 16
C1, C2 = tf.constant(70, dtype="float32"), tf.constant(1000, dtype="float32")


def score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]

    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true[:, 0] - fvc_pred)
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt(tf.cast(2.0, dtype=tf.float32))
    metric = (delta / sigma_clip) * sq2 + tf.math.log(sigma_clip * sq2)
    return K.mean(metric)


def qloss(y_true, y_pred):
    qs = [0.2, 0.5, 0.8]
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q * e, (q - 1) * e)
    return K.mean(v)


def mloss(_lambda):
    def loss(y_true, y_pred):
        return _lambda * qloss(y_true, y_pred) + (1 - _lambda) * score(y_true, y_pred)

    return loss


def make_model(input_dim):
    z_in = L.Input((input_dim,), name="Patient")
    x = L.Dense(100, activation="relu", name="d1")(z_in)
    x = L.Dense(100, activation="relu", name="d2")(x)
    p1 = L.Dense(3, activation="linear", name="p1")(x)
    p2 = L.Dense(3, activation="relu", name="p2")(x)
    preds = L.Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1), name="preds")([p1, p2])

    model = M.Model(z_in, preds, name="ANN")

    opt = tf.keras.optimizers.Adam(
        learning_rate=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False
    )
    model.compile(loss=mloss(M_LOSS), optimizer=opt, metrics=[score])
    return model




## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1628892078.py in <cell line: 0>()
----> 1 C1, C2 = tf.constant(70, dtype="float32"), tf.constant(1000, dtype="float32")
      2 
      3 
      4 def score(y_true, y_pred):
      5     y_true = tf.cast(y_true, tf.float32)

NameError: name 'tf' is not defined

## === cell 17
net = make_model(input_dim=z.shape[1])
print(net.summary())
print(net.count_params())



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2276884122.py in <cell line: 0>()
----> 1 net = make_model(input_dim=z.shape[1])
      2 print(net.summary())
      3 print(net.count_params())
      4 

NameError: name 'make_model' is not defined

## === cell 18
y_train = y.reshape(-1, 1).astype("float32")




## === cell 19
def lr_schedule(epoch, lr):
    return float(lr / (1.0 + DECAY * epoch))


lr_cb = tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=0)

cnt = 0
for tr_idx, val_idx in kf.split(z):
    cnt += 1
    print(f"FOLD {cnt}")

    net = make_model(input_dim=z.shape[1])

    net.fit(
        z[tr_idx],
        y_train[tr_idx],
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(z[val_idx], y_train[val_idx]),
        verbose=0,
        callbacks=[lr_cb],
    )

    print(
        "train",
        net.evaluate(z[tr_idx], y_train[tr_idx], verbose=0, batch_size=BATCH_SIZE),
    )
    print(
        "val",
        net.evaluate(z[val_idx], y_train[val_idx], verbose=0, batch_size=BATCH_SIZE),
    )

    pred[val_idx] = net.predict(z[val_idx], batch_size=BATCH_SIZE, verbose=0)
    pe += net.predict(ze, batch_size=BATCH_SIZE, verbose=0) / NFOLD
    delta_pred_all += net.predict(z, batch_size=BATCH_SIZE, verbose=0) / NFOLD
    print()



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2050209119.py in <cell line: 0>()
      5 
      6 
----> 7 lr_cb = tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=0)
      8 
      9 cnt = 0

NameError: name 'tf' is not defined

## === cell 20
sigma_opt = mean_absolute_error(y, pred[:, 1])
unc = pred[:, 2] - pred[:, 0]
sigma_mean = np.mean(unc)
print("sigma_opt (MAE):", sigma_opt, "sigma_mean:", sigma_mean)



## === cell 21
o_clipped = np.maximum(delta_pred_all[:, 2] - delta_pred_all[:, 0], 70)
delta_abs = np.minimum(np.abs(delta_pred_all[:, 1] - y), 1000)
sqrt2 = np.sqrt(2.0)
score_arr = (-(sqrt2 * delta_abs) / (o_clipped)) - np.log(sqrt2 * o_clipped)
logL_Score = float(np.mean(score_arr))

print("CV metric estimate:", logL_Score)
print(
    "unc stats:",
    float(unc.min()),
    float(unc.mean()),
    float(unc.max()),
    float((unc >= 0).mean()),
)



## === cell 22
print(
    "we are using fix seed value always to avoid RANDOMIZATION (NEED TO GET SAME RESULT)"
)
print("Seed value          =", SEED)
print("Batch size          =", BATCH_SIZE)
print("Number of epochs    =", EPOCHS)
print("NFOLD               =", NFOLD)

print("\nmean_absolute_error =", sigma_opt)
print("unc_mean            =", float(unc.mean()))
print("Log_laplace_Scores  =", logL_Score)



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4170872671.py in <cell line: 0>()
      2     "we are using fix seed value always to avoid RANDOMIZATION (NEED TO GET SAME RESULT)"
      3 )
----> 4 print("Seed value          =", SEED)
      5 print("Batch size          =", BATCH_SIZE)
      6 print("Number of epochs    =", EPOCHS)

NameError: name 'SEED' is not defined

## === cell 23
plt.hist(unc, bins=50)
plt.title("uncertainty in prediction")
plt.show()



## === cell 24
sub_pred = sub[["Patient_Week"]].copy()
sub_pred["FVC"] = pe[:, 1].astype(np.float32)

conf = (pe[:, 2] - pe[:, 0]).astype(np.float32)

conf = np.where(np.isfinite(conf), conf, 0.0).astype(np.float32)
sub_pred["Confidence"] = np.abs(conf)

sub_pred.loc[sub_pred["Confidence"] < 70, "Confidence"] = 70.0



## === cell 25
otest = pd.read_csv(f"{ROOT}/test.csv")
otest["Patient_Week"] = otest["Patient"] + "_" + otest["Weeks"].astype(str)

sub_pred = sub_pred.merge(
    otest[["Patient_Week", "FVC"]].rename(columns={"FVC": "FVC_true_baseline"}),
    on="Patient_Week",
    how="left",
)

mask = sub_pred["FVC_true_baseline"].notna()
sub_pred.loc[mask, "FVC"] = sub_pred.loc[mask, "FVC_true_baseline"].astype(np.float32)
sub_pred.loc[mask, "Confidence"] = 70.0
sub_pred.drop(columns=["FVC_true_baseline"], inplace=True)

sample = pd.read_csv(f"{ROOT}/sample_submission.csv")[["Patient_Week"]]
submission = sample.merge(sub_pred, on="Patient_Week", how="left")

submission["FVC"] = submission["FVC"].astype(np.float32)
submission["Confidence"] = submission["Confidence"].astype(np.float32)
submission.loc[submission["Confidence"] < 70, "Confidence"] = 70.0

submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Columns:", submission.columns.tolist())
assert submission.shape[0] == sample.shape[0]
assert submission.columns.tolist() == ["Patient_Week", "FVC", "Confidence"]
assert submission["FVC"].notna().all() and submission["Confidence"].notna().all()
