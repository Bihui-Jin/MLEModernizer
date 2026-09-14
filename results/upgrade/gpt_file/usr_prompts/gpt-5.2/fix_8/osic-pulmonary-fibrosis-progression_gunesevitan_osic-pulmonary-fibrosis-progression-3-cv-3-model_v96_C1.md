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

-6.851255533448338

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -9.74653) has done: 'I fix the TensorFlow/protobuf crash by disabling XLA JIT and avoiding the forced pure-Python protobuf setting that triggers the `MessageFactory.GetPrototype` error in this Kaggle image. Then I fix the training shape bug by making `y_train` 2D (so `y_true[:,0]` works) while keeping the same model/loss semantics, and ensure QR training also receives a 2D target to avoid silent broadcasting issues. Finally, I make submission generation robust by ensuring prediction columns exist (training completes) and by clipping/cleaning `Confidence` per metric constraints before writing a `submission.csv`.'
- What this solution (achieved -9.74653) has done: 'I fix the TensorFlow/protobuf crash by forcing the Python protobuf implementation early (before importing TensorFlow), which is the most reliable workaround for the `MessageFactory.GetPrototype` error in Kaggle’s OSIC image. Then I keep the model/loss/training logic intact and only make stability fixes that don’t change semantics (deterministic seeding and safe protobuf env ordering). Finally, I ensure submission generation always produces the required `Patient_Week,FVC,Confidence` columns and writes a `submission.csv` file in the working directory.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"

import random
import gc

import numpy as np
import pandas as pd

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

from scipy.stats import skew, mode
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

import cv2

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, GaussianDropout

SEED = 1337


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


pydicom = None

seed_everything(SEED)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

print("TensorFlow version:", tf.__version__)
print("Eager execution:", tf.executing_eagerly())




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/249806697.py in <cell line: 0>()
     33 from sklearn.linear_model import LinearRegression
     34 
---> 35 import tensorflow as tf
     36 import tensorflow.keras.backend as K
     37 from tensorflow.keras.models import Model

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

## === cell 1
def _read_csv_fallback(rel_path_from_input_root: str) -> pd.DataFrame:
    """
    Fix: Some Kaggle environments mount inputs at /kaggle/input, others use ../input.
    Keep I/O semantics the same but try both locations for robustness.
    """
    p1 = os.path.join("../input", rel_path_from_input_root)
    p2 = os.path.join("/kaggle/input", rel_path_from_input_root)
    if os.path.exists(p1):
        return pd.read_csv(p1)
    return pd.read_csv(p2)


df_train = _read_csv_fallback("osic-pulmonary-fibrosis-progression/train.csv")
df_test = _read_csv_fallback("osic-pulmonary-fibrosis-progression/test.csv")
df_submission = _read_csv_fallback(
    "osic-pulmonary-fibrosis-progression/sample_submission.csv"
)

print(
    f'Training Set Shape = {df_train.shape} - Patients = {df_train["Patient"].nunique()}'
)
print(f"Training Set Memory Usage = {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB")
print(f'Test Set Shape = {df_test.shape} - Patients = {df_test["Patient"].nunique()}')
print(f"Test Set Memory Usage = {df_test.memory_usage().sum() / 1024 ** 2:.2f} MB")
print(f"Sample Submission Shape = {df_submission.shape}")
print(
    f"Sample Submission Memory Usage = {df_submission.memory_usage().sum() / 1024 ** 2:.2f} MB"
)




## === cell 2
class Preprocessor:
    def __init__(
        self, df_train, df_test, df_submission, n_folds, shuffle, resize_shape
    ):
        self.df_train = df_train.copy(deep=True)
        self.df_train.sort_values(by=["Patient", "Weeks"], inplace=True)
        self.df_test = df_test.copy(deep=True)
        self.df_submission = df_submission.copy(deep=True)

        self.n_folds = n_folds
        self.shuffle = shuffle
        self.resize_shape = resize_shape

    def _drop_duplicates(self):
        self.df_train["FVC"] = self.df_train.groupby(["Patient", "Weeks"])[
            "FVC"
        ].transform("mean")
        self.df_train["Percent"] = self.df_train.groupby(["Patient", "Weeks"])[
            "Percent"
        ].transform("mean")
        self.df_train.drop_duplicates(inplace=True)
        self.df_train.reset_index(drop=True, inplace=True)

    def _label_encode(self):
        for df in [self.df_train, self.df_test]:
            df["Sex"] = df["Sex"].map({"Male": 0, "Female": 1})
            df["SmokingStatus"] = df["SmokingStatus"].map(
                {"Never smoked": 0, "Ex-smoker": 1, "Currently smokes": 2}
            )

    def _create_folds(self):
        self.df_train["Sex_SmokingStatus"] = (
            self.df_train["Sex"].astype(str)
            + "_"
            + self.df_train["SmokingStatus"].astype(str)
        )
        for group in self.df_train["Sex_SmokingStatus"].unique():
            patients = self.df_train[self.df_train["Sex_SmokingStatus"] == group][
                "Patient"
            ].unique()

            if self.shuffle:
                np.random.seed(SEED)
                np.random.shuffle(patients)

            for fold, patient_group in enumerate(
                np.array_split(patients, self.n_folds), 1
            ):
                self.df_train.loc[
                    self.df_train["Patient"].isin(patient_group), "CV1_Fold"
                ] = fold

        for patient_name in self.df_train["Patient"].unique():
            last2 = self.df_train[(self.df_train["Patient"] == patient_name)][
                "FVC"
            ].values[-2:]
            std = last2.std()
            if std == 0:
                z = np.zeros_like(last2, dtype=np.float32)
            else:
                z = (last2 - last2.mean()) / std

            reg = LinearRegression().fit(
                self.df_train[(self.df_train["Patient"] == patient_name)]["Weeks"]
                .values[-2:]
                .reshape(-1, 1),
                z,
            )
            self.df_train.loc[self.df_train["Patient"] == patient_name, "Intercept"] = (
                reg.intercept_
            )
            self.df_train.loc[self.df_train["Patient"] == patient_name, "Coef"] = (
                reg.coef_[0]
            )

        self.df_train.loc[self.df_train["Coef"] > 0.4, "Cluster"] = 1
        self.df_train.loc[
            (self.df_train["Coef"] < 0.4) & (self.df_train["Coef"] > -0.4), "Cluster"
        ] = 2
        self.df_train.loc[self.df_train["Coef"] < -0.4, "Cluster"] = 3

        for group in self.df_train["Cluster"].unique():
            patients = self.df_train[self.df_train["Cluster"] == group][
                "Patient"
            ].unique()

            if self.shuffle:
                np.random.seed(SEED)
                np.random.shuffle(patients)

            for fold, patient_group in enumerate(
                np.array_split(patients, self.n_folds), 1
            ):
                self.df_train.loc[
                    self.df_train["Patient"].isin(patient_group), "CV2_Fold"
                ] = fold

        patients = self.df_train["Patient"].unique()
        np.random.seed(SEED)
        np.random.shuffle(patients)

        for fold, patient_group in enumerate(np.array_split(patients, self.n_folds), 1):
            self.df_train.loc[
                self.df_train["Patient"].isin(patient_group), "CV3_Fold"
            ] = fold

        self.df_train.drop(
            columns=["Sex_SmokingStatus", "Intercept", "Coef", "Cluster"], inplace=True
        )

    def load_scan(self, dataset, patient_name):
        global pydicom
        if pydicom is None:
            import pydicom as _pydicom

            pydicom = _pydicom

        base_rel = f"osic-pulmonary-fibrosis-progression/{dataset}/{patient_name}"
        base1 = f"../input/{base_rel}"
        base2 = f"/kaggle/input/{base_rel}"
        base = base1 if os.path.exists(base1) else base2

        patient_directory = [pydicom.dcmread(f"{base}/{s}") for s in os.listdir(base)]

        try:
            patient_directory.sort(key=lambda s: float(s.ImagePositionPatient[2]))
            slice_positions = np.round(
                [s.ImagePositionPatient[2] for s in patient_directory], 4
            )
            non_duplicate_idx = np.unique(
                [
                    np.where(slice_position == slice_positions)[0][0]
                    for slice_position in slice_positions
                ]
            )
        except Exception:
            patient_directory.sort(key=lambda s: int(s.InstanceNumber))
            instance_numbers = np.array(
                [int(s.InstanceNumber) for s in patient_directory]
            )
            non_duplicate_idx = np.unique(
                [
                    np.where(instance_number == instance_numbers)[0][0]
                    for instance_number in instance_numbers
                ]
            )

        patient_directory = list(np.array(patient_directory)[non_duplicate_idx])

        metadata = {}
        pixel_spacings = np.zeros((len(patient_directory), 2))
        slice_positions = np.zeros((len(patient_directory)))

        for i, s in enumerate(patient_directory):
            try:
                pixel_spacings[i, :] = np.array(s.PixelSpacing)
            except Exception:
                pixel_spacings[i, :] = np.nan

            try:
                slice_positions[i] = s.ImagePositionPatient[2]
            except Exception:
                pass

        metadata["PixelSpacing"] = list(np.round(np.nanmean(pixel_spacings, axis=0), 3))

        if patient_name == "ID00128637202219474716089":
            metadata["SliceSpacing"] = 5.0
        elif patient_name == "ID00132637202222178761324":
            metadata["SliceSpacing"] = 0.7
        else:
            try:
                metadata["SliceSpacing"] = list(
                    mode(np.abs(np.diff(np.round(slice_positions, 3))))
                )[0][0]
            except Exception:
                metadata["SliceSpacing"] = 1.0

        scan = np.zeros(
            (len(patient_directory), self.resize_shape[0], self.resize_shape[1]),
            dtype=np.int16,
        )
        for i, s in enumerate(patient_directory):
            s_processed = self.crop(s.pixel_array)
            s_processed = self.resize(s_processed)
            s_processed = self.window(
                s_processed, s.RescaleSlope, s.RescaleIntercept, 1500, -500, 0, 256
            )
            if not np.all(s_processed == 0):
                scan[i] = np.int16(s_processed)

        del patient_directory
        scan = scan[~np.all(scan == 0, axis=(-1, -2))]
        return scan, metadata

    def crop(self, s):
        if np.all(s == 0):
            return s
        if s.shape[0] != self.resize_shape[0] and s.shape[1] != self.resize_shape[1]:
            s_cropped = s[~np.all(s == 0, axis=1)]
            s_cropped = s_cropped[:, ~np.all(s_cropped == 0, axis=0)]
        else:
            s_cropped = s
        return s_cropped

    def resize(self, s):
        if s.shape[0] != self.resize_shape[0] and s.shape[1] != self.resize_shape[1]:
            s_resized = cv2.resize(
                s, self.resize_shape, interpolation=cv2.INTER_NEAREST
            )
        else:
            s_resized = s
        return s_resized

    def window(self, s, slope, intercept, window_width, window_center, y_min, y_max):
        x = s * slope + intercept
        y = np.zeros_like(x)
        y[x <= (window_center - 0.5 - (window_width - 1) / 2)] = y_min
        y[x > (window_center - 0.5 + (window_width - 1) / 2)] = y_max
        m = (x > (window_center - 0.5 - (window_width - 1) / 2)) & (
            x <= (window_center - 0.5 + (window_width - 1) / 2)
        )
        y[m] = ((x[m] - (window_center - 0.5)) / (window_width - 1) + 0.5) * (
            y_max - y_min
        ) + y_min
        return y

    def _create_baseline_features(self):
        self.df_submission["Type"] = "Test"
        pw = (
            self.df_submission["Patient_Week"]
            .astype(str)
            .str.split("_", n=1, expand=True)
        )
        self.df_submission["Patient"] = pw[0].astype(str)
        self.df_submission["Weeks"] = pw[1].astype(int)
        self.df_submission.drop(
            columns=["Patient_Week", "FVC", "Confidence"], inplace=True
        )

        self.df_train["Type"] = "Train"
        self.df_train["Weeks_Passed"] = self.df_train["Weeks"] - self.df_train.groupby(
            "Patient"
        )["Weeks"].transform("min")
        self.df_train["FVC_Baseline"] = self.df_train.groupby("Patient")[
            "FVC"
        ].transform("first")

        base_test = self.df_test.drop_duplicates("Patient").set_index("Patient")[
            ["FVC", "Percent", "Age", "Sex", "SmokingStatus"]
        ]
        self.df_submission = self.df_submission.join(
            base_test, on="Patient", how="left"
        )
        self.df_submission = self.df_submission.rename(columns={"FVC": "FVC_Baseline"})
        self.df_submission["Weeks_Passed"] = self.df_submission["Weeks"]

        self.df_all = pd.concat(
            [self.df_train, self.df_submission], ignore_index=True, axis=0
        )
        self.df_all["Age"] += np.int8(np.floor(self.df_all["Weeks_Passed"] / 52))
        self.df_all["Age"] = self.df_all["Age"].astype(np.float32)
        self.df_all["FVC_Baseline"] = self.df_all["FVC_Baseline"].astype(np.float32)
        self.df_all["Percent"] = self.df_all["Percent"].astype(np.float32)
        self.df_all["Weeks_Passed"] = self.df_all["Weeks_Passed"].astype(np.float32)
        self.df_all["Weeks"] = self.df_all["Weeks"].astype(np.int16)
        self.df_all["Sex"] = self.df_all["Sex"].astype(np.uint8)
        self.df_all["SmokingStatus"] = self.df_all["SmokingStatus"].astype(np.uint8)
        self.df_all["FVC"] = self.df_all["FVC"].astype(np.float32)

        self.df_train = self.df_all.loc[self.df_all["Type"] == "Train", :].drop(
            columns=["Type"]
        )
        for i in range(1, 4):
            self.df_train[f"CV{i}_Fold"] = self.df_train[f"CV{i}_Fold"].astype(np.uint8)

        self.df_test = self.df_all.loc[self.df_all["Type"] == "Test", :].drop(
            columns=["Type", "FVC", "CV1_Fold", "CV2_Fold", "CV3_Fold"]
        )

    def _create_image_features(self):
        patients_tr = self.df_train["Patient"].unique()
        patients_te = self.df_test["Patient"].unique()

        df_train_features = pd.DataFrame({"Patient": patients_tr})
        df_test_features = pd.DataFrame({"Patient": patients_te})

        df_train_features["Scan_Skew"] = 0.0
        df_train_features["Scan_Mean"] = 0.0
        df_train_features["Scan_Std"] = 1.0
        df_train_features["Std_Slice_Skew"] = 0.0
        df_train_features["Var_Slice_Skew"] = 0.0

        df_test_features["Scan_Skew"] = 0.0
        df_test_features["Scan_Mean"] = 0.0
        df_test_features["Scan_Std"] = 1.0
        df_test_features["Std_Slice_Skew"] = 0.0
        df_test_features["Var_Slice_Skew"] = 0.0

        scale_features = ["Scan_Std", "Scan_Mean"]
        scaler = StandardScaler()
        scaler.fit(df_train_features.loc[:, scale_features])
        df_train_features.loc[:, scale_features] = scaler.transform(
            df_train_features.loc[:, scale_features]
        )

        self.df_train = self.df_train.merge(df_train_features, how="left", on="Patient")
        self.df_test = self.df_test.merge(df_test_features, how="left", on="Patient")

        self.df_test.loc[:, scale_features] = scaler.transform(
            self.df_test.loc[:, scale_features]
        )

    def get_data(self):
        self._drop_duplicates()
        self._label_encode()
        self._create_folds()
        self._create_baseline_features()
        self._create_image_features()

        print(f"Preprocessed Training Set Shape = {self.df_train.shape}")
        print(
            f"Preprocessed Training Set Memory Usage = {self.df_train.memory_usage().sum() / 1024 ** 2:.2f} MB"
        )
        print(f"Preprocessed Test Set Shape = {self.df_test.shape}")
        print(
            f"Preprocessed Test Set Memory Usage = {self.df_test.memory_usage().sum() / 1024 ** 2:.2f} MB"
        )

        return self.df_train.copy(deep=True), self.df_test.copy(deep=True)




## === cell 3
preprocessor_parameters = {
    "df_train": df_train,
    "df_test": df_test,
    "df_submission": df_submission,
    "n_folds": 2,
    "shuffle": True,
    "resize_shape": (512, 512),
}

preprocessor = Preprocessor(**preprocessor_parameters)
df_train, df_test = preprocessor.get_data()




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3717416800.py in <cell line: 0>()
      9 
     10 preprocessor = Preprocessor(**preprocessor_parameters)
---> 11 df_train, df_test = preprocessor.get_data()
     12 
     13 

/tmp/ipykernel_11/476254013.py in get_data(self)
    315         self._drop_duplicates()
    316         self._label_encode()
--> 317         self._create_folds()
    318         self._create_baseline_features()
    319         self._create_image_features()

/tmp/ipykernel_11/476254013.py in _create_folds(self)
     41 
     42             if self.shuffle:
---> 43                 np.random.seed(SEED)
     44                 np.random.shuffle(patients)
     45 

NameError: name 'SEED' is not defined

## === cell 4
class QuantileRegressorMLP:
    def __init__(self, model, predictors, mlp_parameters, qr_parameters):
        self.model = model
        self.predictors = predictors
        self.mlp_parameters = mlp_parameters
        self.qr_parameters = qr_parameters

        self.df_train_ = None  # set in train()

    def laplace_log_likelihood_metric(self, y_true, y_pred, sigma):
        sigma_clipped = np.maximum(sigma, 70)
        delta_clipped = np.minimum(np.abs(y_true - y_pred), 1000)
        score = -np.sqrt(2) * delta_clipped / sigma_clipped - np.log(
            np.sqrt(2) * sigma_clipped
        )
        return np.mean(score)

    def laplace_log_likelihood_loss(self, y_true, y_pred):
        y_true = K.cast(y_true, "float32")
        y_pred = K.cast(y_pred, "float32")

        sigma_lower_bound = K.constant(70, dtype="float32")
        delta_upper_bound = K.constant(1000, dtype="float32")

        sigma = y_pred[:, 1]
        fvc_pred = y_pred[:, 0]

        sigma_clipped = K.maximum(sigma, sigma_lower_bound)
        delta = K.abs(y_true[:, 0] - fvc_pred)
        delta_clipped = K.minimum(delta, delta_upper_bound)

        score = (delta_clipped / sigma_clipped) * K.sqrt(K.cast(2, "float32")) + K.log(
            sigma_clipped * K.sqrt(K.cast(2, "float32"))
        )
        return K.mean(score)

    def tilted_loss(self, y_true, y_pred):
        quantiles = K.constant(
            np.array([self.qr_parameters["quantiles"]]), dtype="float32"
        )
        error = y_true - y_pred
        return K.mean(K.maximum(quantiles * error, (quantiles - 1) * error))

    def get_model(self, input_shape, m):
        if m == "MLP":
            input_layer = Input(shape=(input_shape,))
            x = Dense(2**7, activation="relu")(input_layer)
            x = GaussianDropout(0.01)(x)
            x = Dense(2**7, activation="relu")(x)
            x = GaussianDropout(0.01)(x)
            p1 = Dense(2, activation="linear")(x)
            p2 = Dense(2, activation="relu")(x)
            output_layer = Lambda(lambda x: x[0] + K.cumsum(x[1], axis=1))([p1, p2])
            model = Model(input_layer, output_layer)
            model.compile(
                loss=self.laplace_log_likelihood_loss,
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=self.mlp_parameters["lr"]
                ),
                metrics=[self.laplace_log_likelihood_loss],
                jit_compile=False,
            )
            return model

        if m == "QR":
            input_layer = Input(shape=(input_shape,))
            x = Dense(2**7, activation="relu")(input_layer)
            x = GaussianDropout(0.01)(x)
            x = Dense(2**7, activation="relu")(
                input_layer if False else x
            )  # keep exact structure
            x = GaussianDropout(0.01)(x)
            p1 = Dense(3, activation="linear")(x)
            p2 = Dense(3, activation="relu")(x)
            output_layer = Lambda(lambda x: x[0] + K.cumsum(x[1], axis=1))([p1, p2])
            model = Model(input_layer, output_layer)
            model.compile(
                loss=self.tilted_loss,
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=self.qr_parameters["lr"]
                ),
                metrics=[self.laplace_log_likelihood_loss],
                jit_compile=False,
            )
            return model

        raise ValueError(f"Unknown model kind: {m}")

    def _final3_scores(self, df, y_true_col, y_pred_col, sigma_col):
        tmp = df.groupby("Patient", sort=False).nth([-1, -2, -3])
        scores = tmp.groupby(level=0, sort=False).apply(
            lambda g: self.laplace_log_likelihood_metric(
                g[y_true_col].to_numpy(),
                g[y_pred_col].to_numpy(),
                g[sigma_col].to_numpy(),
            )
        )
        return scores.to_numpy(dtype=np.float64, copy=False)

    def train(self, X_train, y_train, df_train_ref):
        self.df_train_ = df_train_ref

        self.mlp_oof = pd.DataFrame(np.zeros((len(y_train), 2)), index=X_train.index)
        self.qr_oof = pd.DataFrame(
            np.zeros((len(y_train), len(self.qr_parameters["quantiles"]))),
            index=X_train.index,
        )

        self.mlp_models = {"CV1": [], "CV2": [], "CV3": []}
        self.qr_models = {"CV1": [], "CV2": [], "CV3": []}

        models = [self.model] if self.model != "Stack" else ["MLP", "QR"]
        for m in models:
            print(f'\nRunning {m.upper()} Model\n{("-") * (14 + (len(m)))}')

            for cv in range(1, 4):
                fold_col = f"CV{cv}_Fold"
                for fold in sorted(X_train[fold_col].unique()):
                    trn_idx = X_train.index[X_train[fold_col] != fold]
                    val_idx = X_train.index[X_train[fold_col] == fold]

                    X_trn_np = X_train.loc[trn_idx, self.predictors].to_numpy(
                        np.float32, copy=False
                    )
                    y_trn_np = y_train.loc[trn_idx].to_numpy(np.float32, copy=False)
                    X_val_np = X_train.loc[val_idx, self.predictors].to_numpy(
                        np.float32, copy=False
                    )
                    y_val_np = y_train.loc[val_idx].to_numpy(np.float32, copy=False)

                    model = self.get_model(input_shape=X_trn_np.shape[1], m=m)

                    if m == "MLP":
                        model.fit(
                            X_trn_np,
                            y_trn_np,
                            epochs=self.mlp_parameters["epochs"],
                            batch_size=self.mlp_parameters["batch_size"],
                            verbose=0,
                        )
                        self.mlp_models[f"CV{cv}"].append(model)
                    else:
                        model.fit(
                            X_trn_np,
                            y_trn_np,
                            epochs=self.qr_parameters["epochs"],
                            batch_size=self.qr_parameters["batch_size"],
                            verbose=0,
                        )
                        self.qr_models[f"CV{cv}"].append(model)

                    predictions = model.predict(X_val_np, verbose=0)

                    if m == "MLP":
                        oof_predictions = predictions[:, 0]
                        oof_confidence = predictions[:, 1]
                        self.mlp_oof.loc[val_idx, 0] = oof_predictions
                        self.mlp_oof.loc[val_idx, 1] = oof_confidence
                        self.df_train_.loc[val_idx, f"CV{cv}_MLP_FVC_Predictions"] = (
                            oof_predictions
                        )
                        self.df_train_.loc[
                            val_idx, f"CV{cv}_MLP_Confidence_Predictions"
                        ] = oof_confidence

                        fold_final_scores = self._final3_scores(
                            self.df_train_.loc[val_idx],
                            "FVC",
                            f"CV{cv}_MLP_FVC_Predictions",
                            f"CV{cv}_MLP_Confidence_Predictions",
                        )

                    else:
                        oof_predictions = predictions[:, 1]
                        oof_confidence = predictions[:, 2] - predictions[:, 0]
                        for i, quantile in enumerate(self.qr_parameters["quantiles"]):
                            self.qr_oof.loc[val_idx, i] = predictions[:, i]
                            self.df_train_.loc[
                                val_idx, f"CV{cv}_QR_{quantile}_Predictions"
                            ] = predictions[:, i]

                        q0, q1, q2 = self.qr_parameters["quantiles"]
                        tmp = (
                            self.df_train_.loc[val_idx]
                            .groupby("Patient", sort=False)
                            .nth([-1, -2, -3])
                        )
                        sig = (
                            tmp[f"CV{cv}_QR_{q2}_Predictions"].to_numpy()
                            - tmp[f"CV{cv}_QR_{q0}_Predictions"].to_numpy()
                        )
                        ytrue = tmp["FVC"].to_numpy()
                        ypred = tmp[f"CV{cv}_QR_{q1}_Predictions"].to_numpy()
                        pids = tmp.index.get_level_values(0).to_numpy()
                        order = np.argsort(pids, kind="mergesort")
                        p_sorted = pids[order]
                        ytrue_s, ypred_s, sig_s = ytrue[order], ypred[order], sig[order]
                        _, starts = np.unique(p_sorted, return_index=True)
                        starts = np.r_[starts, len(p_sorted)]
                        fold_final_scores = np.empty(len(starts) - 1, dtype=np.float64)
                        for i in range(len(starts) - 1):
                            a, b = starts[i], starts[i + 1]
                            fold_final_scores[i] = self.laplace_log_likelihood_metric(
                                ytrue_s[a:b], ypred_s[a:b], sig_s[a:b]
                            )

                    oof_score_all = self.laplace_log_likelihood_metric(
                        y_val_np[:, 0], oof_predictions, oof_confidence
                    )
                    print(
                        f"CV {cv} {m} Fold {int(fold)} - X_train: {X_trn_np.shape} X_val: {X_val_np.shape} - "
                        f"All Measurement Score: {oof_score_all:.6} - Final 3 Measurement Score {np.mean(fold_final_scores):.6} "
                        f"[Std: {np.std(fold_final_scores):.6}]"
                    )

                if m == "MLP":
                    oof_final_scores = self._final3_scores(
                        self.df_train_,
                        "FVC",
                        f"CV{cv}_MLP_FVC_Predictions",
                        f"CV{cv}_MLP_Confidence_Predictions",
                    )
                    oof_all_score = self.laplace_log_likelihood_metric(
                        y_train.to_numpy(np.float32, copy=False)[:, 0],
                        self.mlp_oof.iloc[:, 0].to_numpy(np.float32, copy=False),
                        self.mlp_oof.iloc[:, 1].to_numpy(np.float32, copy=False),
                    )
                else:
                    q0, q1, q2 = self.qr_parameters["quantiles"]
                    tmp = self.df_train_.groupby("Patient", sort=False).nth(
                        [-1, -2, -3]
                    )
                    pids = tmp.index.get_level_values(0).to_numpy()
                    sig = (
                        tmp[f"CV{cv}_QR_{q2}_Predictions"].to_numpy()
                        - tmp[f"CV{cv}_QR_{q0}_Predictions"].to_numpy()
                    )
                    ytrue = tmp["FVC"].to_numpy()
                    ypred = tmp[f"CV{cv}_QR_{q1}_Predictions"].to_numpy()
                    order = np.argsort(pids, kind="mergesort")
                    p_sorted = pids[order]
                    ytrue_s, ypred_s, sig_s = ytrue[order], ypred[order], sig[order]
                    _, starts = np.unique(p_sorted, return_index=True)
                    starts = np.r_[starts, len(p_sorted)]
                    oof_final_scores = np.empty(len(starts) - 1, dtype=np.float64)
                    for i in range(len(starts) - 1):
                        a, b = starts[i], starts[i + 1]
                        oof_final_scores[i] = self.laplace_log_likelihood_metric(
                            ytrue_s[a:b], ypred_s[a:b], sig_s[a:b]
                        )

                    oof_all_score = self.laplace_log_likelihood_metric(
                        y_train.to_numpy(np.float32, copy=False)[:, 0],
                        self.qr_oof.iloc[:, 1].to_numpy(np.float32, copy=False),
                        (
                            self.qr_oof.iloc[:, 2].to_numpy(np.float32, copy=False)
                            - self.qr_oof.iloc[:, 0].to_numpy(np.float32, copy=False)
                        ),
                    )

                print(
                    f'{"-" * 30}\nCV {cv} {m} All Measurement OOF Score {oof_all_score:.6} - '
                    f'Final 3 Measurement OOF Score {np.mean(oof_final_scores):.6} [Std: {np.std(oof_final_scores):.6}]\n{"-" * 30}\n'
                )

        return self.df_train_

    def predict(self, X_test):
        X_np = X_test[self.predictors].to_numpy(np.float32, copy=False)
        for cv in range(1, 4):
            mlp_predictions = np.zeros((len(X_test), 2), dtype=np.float32)
            for model in self.mlp_models[f"CV{cv}"]:
                mlp_predictions += model.predict(X_np, verbose=0) / len(
                    self.mlp_models[f"CV{cv}"]
                )
            X_test[f"CV{cv}_MLP_FVC_Predictions"] = mlp_predictions[:, 0]
            X_test[f"CV{cv}_MLP_Confidence_Predictions"] = mlp_predictions[:, 1]

            qr_predictions = np.zeros(
                (len(X_test), len(self.qr_parameters["quantiles"])), dtype=np.float32
            )
            for model in self.qr_models[f"CV{cv}"]:
                qr_predictions += model.predict(X_np, verbose=0) / len(
                    self.qr_models[f"CV{cv}"]
                )
            for i, quantile in enumerate(self.qr_parameters["quantiles"]):
                X_test[f"CV{cv}_QR_{quantile}_Predictions"] = qr_predictions[:, i]

    def plot_predictions(self, df, patient):
        mlp_prediction_columns = [
            f"CV{cv}_MLP_{target}_Predictions"
            for cv in range(1, 4)
            for target in ["FVC", "Confidence"]
        ]
        mlp_scores = []
        for cv in range(1, 4):
            score = self.laplace_log_likelihood_metric(
                df["FVC"],
                df[f"CV{cv}_MLP_FVC_Predictions"],
                df[f"CV{cv}_MLP_Confidence_Predictions"],
            )
            mlp_scores.append(round(score, 5))

        qr_prediction_columns = [
            f"CV{cv}_QR_{quantile}_Predictions"
            for cv in range(1, 4)
            for quantile in self.qr_parameters["quantiles"]
        ]
        qr_scores = []
        for i, cv in enumerate(range(1, 4)):
            score = self.laplace_log_likelihood_metric(
                df["FVC"],
                df[qr_prediction_columns[1 + (i * 3)]],
                (
                    df[qr_prediction_columns[2 + (i * 3)]]
                    - df[qr_prediction_columns[0 + (i * 3)]]
                ),
            )
            qr_scores.append(round(score, 5))

        ax = (
            df[
                ["Weeks", "FVC"]
                + mlp_prediction_columns[::2]
                + qr_prediction_columns[1::3]
            ]
            .set_index("Weeks")
            .plot(figsize=(30, 6), style=["-b", "r--", "g--", "b--", "r:", "g:", "b:"])
        )
        ax.fill_between(
            df["Weeks"],
            df[qr_prediction_columns[2]],
            df[qr_prediction_columns[0]],
            alpha=0.1,
            color="red",
        )
        ax.fill_between(
            df["Weeks"],
            df[qr_prediction_columns[5]],
            df[qr_prediction_columns[3]],
            alpha=0.1,
            color="green",
        )
        ax.fill_between(
            df["Weeks"],
            df[qr_prediction_columns[8]],
            df[qr_prediction_columns[6]],
            alpha=0.1,
            color="blue",
        )

        ax.tick_params(axis="x", labelsize=20)
        ax.tick_params(axis="y", labelsize=20)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(
            f"Patient: {patient} - MLP Scores: {mlp_scores} QR Scores: {qr_scores}",
            size=25,
            pad=25,
        )
        ax.legend(prop={"size": 18})
        plt.show()




## === cell 5
seed_everything(SEED)

X_train = df_train.drop(columns=["FVC"])
y_train = df_train[["FVC"]].copy(deep=True)

model_parameters = {
    "model": "Stack",
    "predictors": [
        "Age",
        "Sex",
        "SmokingStatus",
        "FVC_Baseline",
        "Percent",
        "Weeks_Passed",
        "Std_Slice_Skew",
    ],
    "mlp_parameters": {"lr": 0.0005, "epochs": 350, "batch_size": 2**5},
    "qr_parameters": {
        "quantiles": [0.25, 0.5, 0.75],
        "lr": 0.0005,
        "epochs": 150,
        "batch_size": 2**5,
    },
}

qr_mlp = QuantileRegressorMLP(**model_parameters)
df_train = qr_mlp.train(X_train, y_train, df_train_ref=df_train)
qr_mlp.predict(df_test)

print(
    "Train columns now include prediction columns:",
    [c for c in df_train.columns if "Predictions" in c][:10],
)
print(
    "Test columns now include prediction columns:",
    [c for c in df_test.columns if "Predictions" in c][:10],
)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1813747170.py in <cell line: 0>()
----> 1 seed_everything(SEED)
      2 
      3 X_train = df_train.drop(columns=["FVC"])
      4 y_train = df_train[["FVC"]].copy(deep=True)
      5 

NameError: name 'seed_everything' is not defined

## === cell 6
DO_PLOTS = False
if DO_PLOTS:
    for patient, dfp in list(df_train.groupby("Patient"))[:2]:
        if all(
            col in dfp.columns
            for col in ["CV1_MLP_FVC_Predictions", "CV1_MLP_Confidence_Predictions"]
        ):
            qr_mlp.plot_predictions(dfp, patient)




## === cell 7
class SubmissionPipeline:
    def __init__(self, df_train, df_test):
        self.df_train = df_train
        self.df_test = df_test

    def laplace_log_likelihood_metric(self, y_true, y_pred, sigma):
        sigma_clipped = np.maximum(sigma, 70)
        delta_clipped = np.minimum(np.abs(y_true - y_pred), 1000)
        score = -np.sqrt(2) * delta_clipped / sigma_clipped - np.log(
            np.sqrt(2) * sigma_clipped
        )
        return np.mean(score)

    def blend(self, by, model, cv=None):
        self.df_test["Patient_Week"] = (
            self.df_test["Patient"].astype(str)
            + "_"
            + self.df_test["Weeks"].astype(str)
        )

        if by == "model":
            if model == "MLP":
                for df in [self.df_train, self.df_test]:
                    df[f"{model}_FVC"] = (
                        (df["CV1_MLP_FVC_Predictions"] * 0.34)
                        + (df["CV2_MLP_FVC_Predictions"] * 0.33)
                        + (df["CV3_MLP_FVC_Predictions"] * 0.33)
                    )
                    df[f"{model}_Confidence"] = (
                        (df["CV1_MLP_Confidence_Predictions"] * 0.34)
                        + (df["CV2_MLP_Confidence_Predictions"] * 0.33)
                        + (df["CV3_MLP_Confidence_Predictions"] * 0.33)
                    )

                score = self.laplace_log_likelihood_metric(
                    self.df_train["FVC"],
                    self.df_train[f"{model}_FVC"],
                    self.df_train[f"{model}_Confidence"],
                )
                print(f"MLP Blend (across CVs) Score: {score:.6}")

                self.df_test["FVC"] = self.df_test[f"{model}_FVC"]
                self.df_test["Confidence"] = self.df_test[f"{model}_Confidence"]

            elif model == "QR":
                quantiles = [0.25, 0.50, 0.75]
                for df in [self.df_train, self.df_test]:
                    for q in quantiles:
                        df[f"{model}_{q}_FVC"] = (
                            (df[f"CV1_QR_{q}_Predictions"] * 0.34)
                            + (df[f"CV2_QR_{q}_Predictions"] * 0.33)
                            + (df[f"CV3_QR_{q}_Predictions"] * 0.33)
                        )

                score = self.laplace_log_likelihood_metric(
                    self.df_train["FVC"],
                    self.df_train[f"{model}_{quantiles[1]}_FVC"],
                    (
                        self.df_train[f"{model}_{quantiles[2]}_FVC"]
                        - self.df_train[f"{model}_{quantiles[0]}_FVC"]
                    ),
                )
                print(f"QR Blend (across CVs) Score: {score:.6}")

                self.df_test["FVC"] = self.df_test[f"{model}_{quantiles[1]}_FVC"]
                self.df_test["Confidence"] = (
                    self.df_test[f"{model}_{quantiles[2]}_FVC"]
                    - self.df_test[f"{model}_{quantiles[0]}_FVC"]
                )
            else:
                raise ValueError("model must be 'MLP' or 'QR'")

        elif by == "cv":
            if cv is None:
                raise ValueError("cv must be provided when by='cv'")
            quantiles = [0.25, 0.50, 0.75]
            for df in [self.df_train, self.df_test]:
                df[f"CV{cv}_FVC"] = (df[f"CV{cv}_MLP_FVC_Predictions"] * 0.5) + (
                    df[f"CV{cv}_QR_{quantiles[1]}_Predictions"] * 0.5
                )
                df[f"CV{cv}_Confidence"] = (
                    df[f"CV{cv}_MLP_Confidence_Predictions"] * 0.5
                ) + (
                    (
                        df[f"CV{cv}_QR_{quantiles[2]}_Predictions"]
                        - df[f"CV{cv}_QR_{quantiles[0]}_Predictions"]
                    )
                    * 0.5
                )

            score = self.laplace_log_likelihood_metric(
                self.df_train["FVC"],
                self.df_train[f"CV{cv}_FVC"],
                self.df_train[f"CV{cv}_Confidence"],
            )
            print(f"CV{cv} Blend (MLP+QR) Score: {score:.6}")

            self.df_test["FVC"] = self.df_test[f"CV{cv}_FVC"]
            self.df_test["Confidence"] = self.df_test[f"CV{cv}_Confidence"]

        else:
            raise ValueError("by must be one of: 'model', 'cv'")

        return self.df_test[["Patient_Week", "FVC", "Confidence"]].copy(deep=True)




## === cell 8
sub = SubmissionPipeline(df_train, df_test)
df_sub_out = sub.blend(by="model", model="MLP")

df_sub_out["FVC"] = pd.to_numeric(df_sub_out["FVC"], errors="coerce").astype(np.float32)
df_sub_out["Confidence"] = pd.to_numeric(
    df_sub_out["Confidence"], errors="coerce"
).astype(np.float32)

df_sub_out["Confidence"] = (
    df_sub_out["Confidence"].replace([np.inf, -np.inf], np.nan).fillna(70.0)
)
df_sub_out["Confidence"] = df_sub_out["Confidence"].clip(lower=1.0)

sample = _read_csv_fallback("osic-pulmonary-fibrosis-progression/sample_submission.csv")
df_sub_out = sample[["Patient_Week"]].merge(df_sub_out, on="Patient_Week", how="left")
df_sub_out["FVC"] = df_sub_out["FVC"].fillna(sample["FVC"]).astype(np.float32)
df_sub_out["Confidence"] = (
    df_sub_out["Confidence"].fillna(sample["Confidence"]).astype(np.float32)
)

df_sub_out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_sub_out.shape)
print(df_sub_out.head())
print(df_sub_out.tail())

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'CV1_MLP_FVC_Predictions'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3723582532.py in <cell line: 0>()
      1 sub = SubmissionPipeline(df_train, df_test)
----> 2 df_sub_out = sub.blend(by="model", model="MLP")
      3 
      4 df_sub_out["FVC"] = pd.to_numeric(df_sub_out["FVC"], errors="coerce").astype(np.float32)
      5 df_sub_out["Confidence"] = pd.to_numeric(

/tmp/ipykernel_11/1542086091.py in blend(self, by, model, cv)
     23                 for df in [self.df_train, self.df_test]:
     24                     df[f"{model}_FVC"] = (
---> 25                         (df["CV1_MLP_FVC_Predictions"] * 0.34)
     26                         + (df["CV2_MLP_FVC_Predictions"] * 0.33)
     27                         + (df["CV3_MLP_FVC_Predictions"] * 0.33)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'CV1_MLP_FVC_Predictions'
