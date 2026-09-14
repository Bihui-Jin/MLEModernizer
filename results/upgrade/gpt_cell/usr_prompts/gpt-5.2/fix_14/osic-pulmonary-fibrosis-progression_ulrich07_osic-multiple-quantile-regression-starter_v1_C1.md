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
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
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

-7.4661

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -8.76214) has done: 'Diagnosis: The crash happens because `DataFrame.append()` was removed in pandas 2.x, so `tr.append([chunk, sub])` raises `AttributeError`. The intended behavior is to vertically concatenate `tr`, `chunk`, and `sub` into a single `data` DataFrame. Using `pd.concat([...], axis=0)` is the direct drop-in replacement and preserves the resulting columns/rows needed by cell 5.

Patch summary: Replace the deprecated `DataFrame.append()` call with `pd.concat()` while keeping the same three input frames and returning a single combined `data` DataFrame. No other logic or data transformations are changed.

Updated cells: cell 4 only.

Compatibility notes for cell k+1: `data` remains a pandas DataFrame with the same columns and is available under the same variable name, so `print(..., data.shape)` and `data.Patient.nunique()` in cell 5 continue to work.

Assumptions: Row order does not matter for later steps (both `append` and `concat` preserve input order by default), and preserving the existing indices is acceptable (default `ignore_index=False`, matching prior `append` behavior).'
- What this solution (achieved -8.76216) has done: 'Diagnosis: The crash happens during `import tensorflow as tf` in cell 17, before any model code runs. With TensorFlow 2.18.0 and protobuf 6.x installed, TensorFlow can fail at import time with `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` due to an incompatibility between TF’s generated protobuf bindings and newer protobuf runtime APIs. The minimal deterministic fix is to force TensorFlow to use the pure-Python protobuf implementation (which still exposes the expected API surface) before importing TensorFlow.

Patch summary: In cell 17 only, set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (and version=2 for determinism) before importing TensorFlow/Keras. This avoids the protobuf C++ runtime path that triggers the missing `GetPrototype` attribute and allows the rest of the notebook (cell 18+) to run unchanged.

Updated cells:  
Compatibility notes for cell k+1: `tf`, `K`, `L`, and `M` are still imported with the same names and modules as before, so cell 18 work unchanged.  
Assumptions: Environment allows setting `os.environ` at runtime and TensorFlow import occurs after this cell (i.e., TensorFlow hasn’t been imported earlier).'
- What this solution (achieved -8.76215) has done: 'Diagnosis: The crash happens during `import tensorflow` because cell 17 forces `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="cpp"`, which makes TensorFlow try to use the C++ protobuf runtime. In this environment, `google.protobuf.pyext._message` is not available (protobuf 6.x), so the import fails. TensorFlow 2.18 works with the Python protobuf implementation here, so we should not force the C++ implementation.

Patch summary: Modify only cell 17 to stop forcing the protobuf C++ backend and instead force the safe Python backend before importing TensorFlow. This avoids the missing `_message` symbol and allows TensorFlow/Keras imports to succeed deterministically.

Updated cells: cell 17 only.

Compatibility notes for cell k+1: `tf`, `K`, `L`, and `M` are still imported with the same names and be available for cell 18 unchanged.

Assumptions: The runtime has TensorFlow 2.18 installed and functional when using the Python protobuf implementation (standard for protobuf>=4/6 environments).'
- What this solution (achieved -8.76215) has done: 'Diagnosis: Cell 17 crashes while importing TensorFlow because it forces `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="cpp"`, but the installed `protobuf==6.33.0` does not ship a compatible C-extension (`google.protobuf.pyext._message`), leading to `ImportError: cannot import name '_message'`. The current try/except only handles an `AttributeError` containing `"GetPrototype"`, so this ImportError is unhandled and the notebook stops. The safest minimal fix is to avoid forcing the C++ protobuf implementation and instead force the pure-Python implementation, which is compatible with protobuf 6 and TensorFlow in this environment.

Patch summary: In cell 17, change the protobuf env var to use `"python"` instead of `"cpp"` and remove the dead/unsafe auto-pip-downgrade logic. Keep the TensorFlow/Keras imports and exposed symbols (`tf`, `K`, `L`, `M`) unchanged so downstream cells work the same.

Updated cells: Only cell 17 is modified.

Compatibility notes for cell k+1: Cell 18 expects `tf`, `K`, `L`, and `M` to be defined; the patched cell 17 still defines them with the same names and modules (`tensorflow as tf`, `tensorflow.keras.backend as K`, etc.).

Assumptions: This runtime allows importing TensorFlow 2.18.0 successfully when protobuf uses the pure-Python implementation; no internet/package installation is required or permitted at runtime.'
- What this solution (achieved -8.76213) has done: 'Diagnosis: The crash happens while importing TensorFlow in cell 17 because this notebook forces `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`. With the installed `protobuf==6.33.0` and `tensorflow==2.18.0`, forcing the pure-Python protobuf runtime triggers an incompatibility (`MessageFactory` missing `GetPrototype`) during TF/protobuf initialization. The fix is to stop forcing the Python implementation and let TensorFlow use the default C++/upb protobuf runtime.

Patch summary: In cell 17, remove the environment overrides that force the Python protobuf implementation (and the related pop call). Keep the TensorFlow/Keras imports unchanged so downstream cells (cell 18+) see the same `tf`, `K`, `L`, `M` symbols.

Updated cells: Only cell 17 is changed.

Compatibility notes for cell k+1: Cell 18 expects `tf`, `K`, `L`, and `M` to be successfully imported; this patch preserves those names and their APIs exactly, only preventing the protobuf runtime mismatch during import.

Assumptions: The environment’s default protobuf backend (without forcing `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`) is available and compatible with TensorFlow 2.18.0 in this runtime.'
- What this solution (achieved -8.76215) has done: 'Diagnosis: The crash happens while importing TensorFlow in cell 17 due to an incompatibility between TensorFlow 2.18 and the installed `protobuf==6.33.0`, which triggers `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` during TensorFlow’s internal protobuf usage. The earlier attempt to “fix” protobuf by unsetting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION*` does not resolve this in protobuf 6. The most reliable minimal fix is to force TensorFlow to use the pure-Python protobuf implementation via environment variables set **before** importing TensorFlow.

Patch summary: In cell 17, set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (and version=2) instead of popping those variables, then import TensorFlow as originally intended. This keeps all downstream TensorFlow/Keras symbols (`tf`, `K`, `L`, `M`) identical for cell 18+.

Updated cells: Only cell 17 is modified.

Compatibility notes for cell k+1: Cell 18 still receives the same `tf`, `K`, `L`, and `M` objects and can build/compile the model unchanged. No interfaces, shapes, or semantics are altered—only the protobuf backend used during TensorFlow import.

Assumptions: The environment allows setting `os.environ` at runtime (standard in notebooks) and TensorFlow can operate with the Python protobuf implementation in this setup.'
- What this solution (achieved -8.76215) has done: 'Diagnosis: The crash happens while importing TensorFlow in cell 17. The cell forces `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`, which makes TensorFlow use the pure-Python protobuf runtime; with protobuf 6.33.0 this leads to `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` during TF/protobuf initialization. TensorFlow 2.18 is compatible with the default C++ protobuf runtime, so forcing the Python implementation is the root cause.

Patch summary: Remove/avoid forcing the pure-Python protobuf runtime by unsetting those environment variables (or setting implementation to `cpp`) before importing TensorFlow. Keep the rest of the cell (imports and aliases `tf`, `K`, `L`, `M`) unchanged so downstream code in cell 18 continues to work.

Updated cells: Only cell 17 is modified.

Compatibility notes for cell k+1: Cell 18 expects `tf`, `K`, `L`, and `M` to be defined exactly as in the original; this patch preserves those names and TensorFlow import behavior.

Assumptions: No other part of the notebook relies on `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`; using TensorFlow’s default protobuf runtime is acceptable in this environment.'
- What this solution (achieved -8.76216) has done: 'Diagnosis: The crash happens when importing TensorFlow in cell 17, raising `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. This is a known incompatibility between TensorFlow and newer `protobuf` (here `protobuf==6.33.0`), where TensorFlow expects older protobuf APIs. The cell currently *removes* the environment variables that would force TensorFlow to use the pure-Python protobuf implementation, which is the usual workaround for this mismatch. We should instead set those environment variables before importing TensorFlow.

Patch summary: Modify only cell 17 to set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (and its version) before importing TensorFlow, rather than popping them. This avoids the protobuf C++ implementation API mismatch and makes the TensorFlow import deterministic.

Updated cells: only cell 17 is changed.

Compatibility notes for cell k+1: All symbols (`tf`, `K`, `L`, `M`) are still defined exactly as before, so cell 18 runs unchanged.

Assumptions: Using the Python protobuf implementation is acceptable for this notebook and does not change model logic/semantics (only the backend protobuf runtime used during import/graph construction).'
- What this solution (achieved -8.76214) has done: 'Diagnosis: Cell 17 crashes while importing TensorFlow because it forces `google.protobuf` to use the C++ implementation (`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp`). In this environment, `protobuf==6.33.0` does not ship a compatible `_message` C-extension module, so `from google.protobuf.pyext import _message` fails during TensorFlow import. The fix is to avoid forcing the C++ protobuf runtime and instead use the pure-Python protobuf implementation, which is compatible here and still works with TensorFlow.

Patch summary: In cell 17 only, remove the hard-forcing of `cpp` and set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` before importing TensorFlow. Keep the module cleanup of `google.protobuf*` to ensure the environment variable takes effect.

Updated cells: Only cell 17 is changed.

Compatibility notes for cell k+1: The patch preserves `import tensorflow as tf` and the subsequent `K`, `L`, `M` imports exactly as expected by cell 18, so `tf`, `K`, `L`, and `M` remain available with the same interfaces.

Assumptions: Using protobuf’s pure-Python implementation is acceptable for this notebook (it is a standard compatibility workaround) and does not change model/training semantics beyond negligible performance differences.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import pydicom
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import mean_absolute_error


## === cell 2
ROOT = "../input/osic-pulmonary-fibrosis-progression"
DESIRED_SIZE = 128


## === cell 3
tr = pd.read_csv(f"{ROOT}/train.csv")
tr.drop_duplicates(keep=False, inplace=True, subset=['Patient','Weeks'])
chunk = pd.read_csv(f"{ROOT}/test.csv")

print("add infos")
sub = pd.read_csv(f"{ROOT}/sample_submission.csv")
sub['Patient'] = sub['Patient_Week'].apply(lambda x:x.split('_')[0])
sub['Weeks'] = sub['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))
sub =  sub[['Patient','Weeks','Confidence','Patient_Week']]
sub = sub.merge(chunk.drop('Weeks', axis=1), on="Patient")


## === cell 4
tr["WHERE"] = "train"
chunk["WHERE"] = "val"
sub["WHERE"] = "test"

data = pd.concat([tr, chunk, sub], axis=0)


## === cell 5
print(tr.shape, chunk.shape, sub.shape, data.shape)
print(tr.Patient.nunique(), chunk.Patient.nunique(), sub.Patient.nunique(), 
      data.Patient.nunique())


## === cell 6
data['min_week'] = data['Weeks']
data.loc[data.WHERE=='test','min_week'] = np.nan
data['min_week'] = data.groupby('Patient')['min_week'].transform('min')


## === cell 7
base = data.loc[data.Weeks == data.min_week]
base = base[['Patient','FVC']].copy()
base.columns = ['Patient','min_FVC']
base['nb'] = 1
base['nb'] = base.groupby('Patient')['nb'].transform('cumsum')
base = base[base.nb==1]
base.drop('nb', axis=1, inplace=True)


## === cell 8
data = data.merge(base, on='Patient', how='left')
data['base_week'] = data['Weeks'] - data['min_week']
del base


## === cell 9
COLS = ['Sex','SmokingStatus']
FE = []
for col in COLS:
    for mod in data[col].unique():
        FE.append(mod)
        data[mod] = (data[col] == mod).astype(int)


## === cell 10
data['age'] = (data['Age'] - data['Age'].min() ) / ( data['Age'].max() - data['Age'].min() )
data['BASE'] = (data['min_FVC'] - data['min_FVC'].min() ) / ( data['min_FVC'].max() - data['min_FVC'].min() )
data['week'] = (data['base_week'] - data['base_week'].min() ) / ( data['base_week'].max() - data['base_week'].min() )
data['percent'] = (data['Percent'] - data['Percent'].min() ) / ( data['Percent'].max() - data['Percent'].min() )
FE += ['age','percent','week','BASE']


## === cell 12
tr = data.loc[data.WHERE=='train']
chunk = data.loc[data.WHERE=='val']
sub = data.loc[data.WHERE=='test']
del data


## === cell 13
tr.shape, chunk.shape, sub.shape


## === cell 14
def get_images(df, how="train"):
    xo = []
    p = []
    w  = []
    for i in tqdm(range(df.shape[0])):
        patient = df.iloc[i,0]
        week = df.iloc[i,1]
        try:
            img_path = f"{ROOT}/{how}/{patient}/{week}.dcm"
            ds = pydicom.dcmread(img_path)
            im = Image.fromarray(ds.pixel_array)
            im = im.resize((DESIRED_SIZE,DESIRED_SIZE)) 
            im = np.array(im)
            xo.append(im[np.newaxis,:,:])
            p.append(patient)
            w.append(week)
        except:
            pass
    data = pd.DataFrame({"Patient":p,"Weeks":w})
    return np.concatenate(xo, axis=0), data


## === cell 17
import os
import sys
import subprocess


def _ensure_compatible_protobuf_and_restart_once():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver
    except Exception:
        pb_ver = None

    def _major(ver):
        try:
            return int(str(ver).split(".", 1)[0])
        except Exception:
            return None

    if (
        _major(pb_ver) is not None
        and _major(pb_ver) >= 5
        and os.environ.get("_RESTARTED_FOR_PROTOBUF", "0") != "1"
    ):
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"]
        )
        os.environ["_RESTARTED_FOR_PROTOBUF"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)


_ensure_compatible_protobuf_and_restart_once()

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

for m in list(sys.modules.keys()):
    if m.startswith("google.protobuf"):
        del sys.modules[m]

import tensorflow as tf

import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
import tensorflow.keras.models as M


## === cell 18
C1, C2 = tf.constant(70, dtype='float32'), tf.constant(1000, dtype="float32")
def score(y_true, y_pred):
    tf.dtypes.cast(y_true, tf.float32)
    tf.dtypes.cast(y_pred, tf.float32)
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]
    
    sigma_clip = tf.maximum(sigma, C1)
    delta = tf.abs(y_true[:, 0] - fvc_pred)
    delta = tf.minimum(delta, C2)
    sq2 = tf.sqrt( tf.dtypes.cast(2, dtype=tf.float32) )
    metric = (delta / sigma_clip)*sq2 + tf.math.log(sigma_clip* sq2)
    return K.mean(metric)
def qloss(y_true, y_pred):
    qs = [0.25, 0.50, 0.75]
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q*e, (q-1)*e)
    return K.mean(v)
def mloss(_lambda):
    def loss(y_true, y_pred):
        return _lambda * qloss(y_true, y_pred) + (1 - _lambda)*score(y_true, y_pred)
    return loss
def make_model():
    z = L.Input((9,), name="Patient")
    x = L.Dense(100, activation="relu", name="d1")(z)
    x = L.Dense(100, activation="relu", name="d2")(x)
    preds = L.Dense(3, activation="relu", name="preds")(x)
    
    model = M.Model(z, preds, name="CNN")
    model.compile(loss=mloss(0.5), optimizer="adam", metrics=[score])
    return model


## === cell 19
net = make_model()
print(net.count_params())


## === cell 20
y = tr['FVC'].values
z = tr[FE].values


## === cell 21
net.fit(z, y, batch_size=500, epochs=800) #, validation_split=0.1


## === cell 22
net.evaluate(z, y)


## === cell 23
pred = net.predict(z, batch_size=100, verbose=1)


## === cell 24
sigma_opt = mean_absolute_error(y, pred[:, 1])
sigma_mean = np.mean(pred[:, 2] - pred[:, 0])
print(sigma_opt, sigma_mean)


## === cell 25
idxs = np.random.randint(0, y.shape[0], 100)
plt.plot(y[idxs])
plt.plot(pred[idxs, 0], label="q25")
plt.plot(pred[idxs, 1], label="q50")
plt.plot(pred[idxs, 2], label="q75")
plt.legend(loc="best")
plt.show()


## === cell 27
unc = pred[:,2] - pred[:, 0]
unc.min(), unc.mean(), unc.max()


## === cell 28
plt.hist(unc)
plt.title("uncertainty in prediction")
plt.show()


## === cell 29
sub.head()


## === cell 30
ze = sub[FE].values
pe = net.predict(ze, batch_size=100, verbose=1)


## === cell 31
sub['FVC1'] = pe[:, 1]
sub['Confidence1'] = pe[:, 2] - pe[:, 0]


## === cell 32
subm = sub[['Patient_Week','FVC','Confidence','FVC1','Confidence1']].copy()


## === cell 34
subm.loc[~subm.FVC1.isnull()].head(10)


## === cell 35
subm.loc[~subm.FVC1.isnull(),'FVC'] = subm.loc[~subm.FVC1.isnull(),'FVC1']
if sigma_mean<70:
    subm['Confidence'] = sigma_opt
else:
    subm.loc[~subm.FVC1.isnull(),'Confidence'] = subm.loc[~subm.FVC1.isnull(),'Confidence1']


## === cell 36
subm.head()


## === cell 37
subm.describe().T


## === cell 39
subm[["Patient_Week","FVC","Confidence"]].to_csv("submission.csv", index=False)
