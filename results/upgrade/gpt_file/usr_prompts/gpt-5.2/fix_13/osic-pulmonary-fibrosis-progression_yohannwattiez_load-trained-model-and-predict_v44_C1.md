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

-6.989372728592168

# 6. Current score

-20.70018

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -12.66732) has done: 'I fix the early import crash by preventing the `pydicom` import from triggering the protobuf `MessageFactory.GetPrototype` error, and I make the pipeline robust by falling back to a tabular-only submission when CT/model loading isn’t available. I also fix the `_read()` path-join bug (it was joining a Series instead of a string) so image loading works when available. For the SavedModel load failure, I add a safe fallback that uses `tf.saved_model.load()` and calls the serving signature directly (instead of `TFSMLayer`), without changing the model itself. Finally, I ensure `submission.csv` is always written with the correct columns and aligned exactly to `sample_submission.csv`.'
- What this solution (achieved -11.2611) has done: 'I fix the immediate crash in the first import cell by preventing the `pydicom` import from triggering the protobuf `MessageFactory.GetPrototype` error (which happens in this Kaggle environment) and cleanly switching to tabular-only mode when DICOM tooling can’t be loaded. I also fix a small but score-relevant data bug in `data_preparation`: it standardizes a non-existent `Base_percent` column instead of the real `Percent`, which can degrade the tabular features and predictions. Finally, I keep the existing inference/core logic intact, but improve the tabular-only fallback to use a simple per-patient linear trend fitted from train history (same semantics: predict FVC + confidence), which should move the score substantially toward the target without changing the CNN path when it’s available. The script always write a valid `submission.csv` with the exact required columns and sample submission alignment.'
- What this solution (achieved -11.2611) has done: 'I fix the crash happening before any cells run by setting protobuf to use the pure-Python implementation *before* importing TensorFlow/pydicom, which avoids the `MessageFactory.GetPrototype` AttributeError in this Kaggle environment. I also make the pydicom import more robust by catching the protobuf-related failure and cleanly falling back to the tabular-only path (your current score-improving linear-per-patient fallback remains unchanged). Finally, I keep the submission-writing logic intact but ensure it always produces a correctly ordered, complete `submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved -11.2611) has done: 'I fix the immediate import crash by ensuring the protobuf environment variables are set early enough and forcing TensorFlow to use the pure-Python protobuf implementation before TensorFlow loads. This unblocks the notebook so it runs end-to-end and still keeps your existing “CNN path if available, otherwise tabular-linear fallback” logic unchanged. I also make the pydicom/skimage imports lazy/optional (only used when needed) so the fallback path can always run even if medical imaging deps break. Finally, I keep the submission-writing logic but guarantee the output `submission.csv` is always produced with the exact required columns and aligned to `sample_submission.csv`.'
- What this solution (achieved -11.2611) has done: 'I fix the early protobuf/TensorFlow import crash by forcing the pure-Python protobuf implementation *before* any TensorFlow import and by retrying TensorFlow import with a clean, score-neutral fallback if it still fails. I also make the script robust to missing TensorFlow by providing tiny stubs for the Keras `Sequence` base class so the tabular-only path can still run end-to-end without changing your modeling logic. Finally, I keep your existing per-patient linear fallback (the score-improving path you’re currently using) unchanged, and ensure the submission CSV is always written with the correct columns and aligned to `sample_submission.csv`.'
- What this solution (achieved -15.9108) has done: 'I fix the import-time crash (`MessageFactory.GetPrototype`) by forcing protobuf’s pure-Python implementation and disabling the C++ implementation *before* any TensorFlow/pydicom-related imports, and by hardening the fallback so the script still runs even if TF/pydicom can’t load. I also make the fallback tabular model output the correct “Confidence” semantics (sigma, not interval width): your current code accidentally uses the full width `(upper-lower)` as confidence, which hurts the Laplace metric; changing it to half-width (sigma) is a minimal, metric-aligned fix that should move the score toward the target. Finally, I keep the CNN/SavedModel path unchanged when available, and ensure `submission.csv` is written with correct columns and aligned to `sample_submission.csv`.'
- What this solution (achieved -15.9108) has done: 'I fix the import-time crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation *and* retrying TensorFlow import in a safe way; if it still fails, the script cleanly fall back to the tabular-only path instead of stopping. I also harden the optional `pydicom` import so that even if it triggers protobuf issues, the rest of the pipeline still runs and writes `submission.csv`. Finally, I keep your existing model/fallback logic intact (including the sigma-as-halfwidth confidence fix) and only adjust the early-import robustness so you can actually get a valid submission and recover the better score behavior.'
- What this solution (achieved -15.9108) has done: 'I fix the import-time crash (`MessageFactory.GetPrototype`) by forcing a compatible protobuf runtime *before* any TensorFlow/pydicom import and by cleanly falling back to the tabular-only path when those imports still fail. I also remove the incorrect `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_C` setting (it can trigger protobuf/TensorFlow breakage in Kaggle) while keeping the rest of your pipeline unchanged. This is execution-critical and score-neutral except that it restores the intended (and better-scoring) behavior instead of crashing. The submission-writing logic remain the same and always produce a valid `submission.csv`.'
- What this solution (achieved -15.9108) has done: 'I fix the TensorFlow/protobuf import crash that prevents the notebook from running by forcing the pure-Python protobuf implementation early and isolating the failing TF import so the tabular fallback can still execute. Then I make the fallback linear-per-patient model slightly more metric-aligned by using a safer, higher-confidence (sigma) calibration derived from train residuals rather than an overly optimistic sigma, which should move the score up toward your target without changing the core approach. Finally, I ensure the submission is always fully aligned to `sample_submission.csv` and contains finite numeric `FVC` and `Confidence` values clipped per competition rules.'
- What this solution (achieved -18.1227) has done: 'I fix the immediate import-time crash (`MessageFactory.GetPrototype`) by forcing protobuf’s pure-Python backend **before any other imports** and by avoiding the TensorFlow/pydicom import path entirely when that protobuf error still occurs, so the notebook always runs end-to-end. Then I keep your existing modeling logic (CNN if available, otherwise per-patient linear fallback) but make the fallback slightly more metric-aligned by calibrating `sigma` from train residuals in a robust way and using it directly as `Confidence` (still clipped to 70). Finally, I ensure the submission is always written as `submission.csv` with exact required columns and aligned to `sample_submission.csv` ordering.'
- What this solution (achieved -20.70018) has done: 'I fix the import-time `MessageFactory.GetPrototype` crash by forcing protobuf’s pure-Python implementation and disabling the C++ implementation *before any TensorFlow/pydicom import*, which is the root cause of the current failure in cell 0. Then I keep your existing “CNN if available else tabular per-patient linear fallback” logic intact, but make the fallback sigma calibration slightly less over-conservative (remove the extra slope penalty and use a robust global sigma blended with per-patient residual sigma) to move score upward toward the target. Finally, I ensure the pipeline always reaches the submission-writing cell and produces a valid `submission.csv` aligned to `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_C", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import random
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

_TF_AVAILABLE = False
tf = None
K = None
L = None

try:
    import tensorflow as tf  # noqa: F401
    from tensorflow import keras as K  # noqa: F401
    from tensorflow.keras import layers as L  # noqa: F401

    _TF_AVAILABLE = True
    try:
        tf.random.set_seed(SEED)
    except Exception:
        pass
except Exception as e:
    print(
        "Warning: tensorflow import failed; will run without CNN/SavedModel path.\n",
        repr(e),
    )
    _TF_AVAILABLE = False
    tf = None
    K = None
    L = None

_HAVE_PYDICOM = False
pydicom = None

try:
    import pydicom as _pydicom  # noqa: F401

    pydicom = _pydicom
    _HAVE_PYDICOM = True
except Exception as e:
    print(
        "Warning: pydicom import failed; will run in tabular-only fallback mode.\n",
        repr(e),
    )
    _HAVE_PYDICOM = False
    pydicom = None

_HAVE_SKIMAGE = False
try:
    from scipy.ndimage import zoom
    import scipy.ndimage as ndimage
    from skimage import measure, morphology, segmentation  # noqa: F401

    _HAVE_SKIMAGE = True
except Exception as e:
    print(
        "Warning: skimage/scipy import failed; will run in tabular-only fallback mode.\n",
        repr(e),
    )
    _HAVE_SKIMAGE = False

from math import ceil

if not _TF_AVAILABLE:

    class _SequenceStub:
        pass

    class _KStub:
        class utils:
            Sequence = _SequenceStub

    K = _KStub()

print(
    "Env status:",
    {"TF": _TF_AVAILABLE, "pydicom": _HAVE_PYDICOM, "skimage": _HAVE_SKIMAGE},
)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
raw_test = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv")
raw_train = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv")
X_prediction = pd.read_csv(
    "/kaggle/input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)



## === cell 2
TEST_PATH = "/kaggle/input/osic-pulmonary-fibrosis-progression/test"

DESIRED_SIZE = (30, 256, 256)
BATCH_SIZE = 256

clip_bounds = (-1000, 200)
pre_calculated_mean = 0.02865046213070556



## === cell 3
X_prediction["Patient"] = X_prediction["Patient_Week"].str.extract(r"(.*)_.*")
X_prediction["Weeks"] = X_prediction["Patient_Week"].str.extract(r".*_(.*)").astype(int)
X_prediction = X_prediction[["Patient", "Weeks", "Patient_Week"]]

rename_cols = {"Weeks_y": "Min_week", "Weeks_x": "Weeks", "FVC": "Base_FVC"}

X_prediction = (
    X_prediction.merge(raw_test, how="left", left_on="Patient", right_on="Patient")
    .rename(columns=rename_cols)[
        [
            "Patient",
            "Min_week",
            "Base_FVC",
            "Percent",
            "Age",
            "Sex",
            "SmokingStatus",
            "Weeks",
            "Patient_Week",
        ]
    ]
    .reset_index(drop=True)
)

X_prediction["Base_week"] = X_prediction["Weeks"] - X_prediction["Min_week"]



## === cell 4
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import NotFittedError


class OneHotEncoder(SklearnOneHotEncoder):
    def __init__(self, **kwargs):
        super(OneHotEncoder, self).__init__(**kwargs)
        self.fit_flag = False

    def fit(self, X, **kwargs):
        out = super().fit(X)
        self.fit_flag = True
        return out

    def transform(self, X, categories, index="", name="", **kwargs):
        sparse_matrix = super(OneHotEncoder, self).transform(X)
        new_columns = self.get_new_columns(X=X, name=name, categories=categories)
        d_out = pd.DataFrame(sparse_matrix.toarray(), columns=new_columns, index=index)
        return d_out

    def fit_transform(self, X, categories, index, name, **kwargs):
        self.fit(X)
        return self.transform(X, categories=categories, index=index, name=name)

    def get_new_columns(self, X, name, categories):
        new_columns = []
        for j in range(len(categories)):
            new_columns.append("{}_{}".format(name, categories[j]))
        return new_columns


def standardisation(x, u, s):
    return (x - u) / s


def normalization(x, ma, mi):
    denom = ma - mi
    if denom == 0:
        return x * 0.0
    return (x - mi) / denom


class data_preparation:
    def __init__(self, bool_normalization=True, bool_standard=False):
        self.enc_sex = LabelEncoder()
        self.enc_smok = LabelEncoder()
        self.onehotenc_smok = OneHotEncoder()
        self.standardisation = bool_standard
        self.normalization = bool_normalization

    def __call__(self, data_untransformed):
        data = data_untransformed.copy(deep=True)

        try:
            data["Sex"] = self.enc_sex.transform(data["Sex"].values)
            data["SmokingStatus"] = self.enc_smok.transform(
                data["SmokingStatus"].values
            )
            data = pd.concat(
                [
                    data.drop(columns=["SmokingStatus"]),
                    self.onehotenc_smok.transform(
                        data["SmokingStatus"].values.reshape(-1, 1),
                        categories=self.enc_smok.classes_,
                        name="",
                        index=data.index,
                    ).astype(int),
                ],
                axis=1,
            )

            if self.standardisation:
                data["Base_week"] = standardisation(
                    data["Base_week"], self.base_week_mean, self.base_week_std
                )
                data["Base_FVC"] = standardisation(
                    data["Base_FVC"], self.base_fvc_mean, self.base_fvc_std
                )
                data["Percent"] = standardisation(
                    data["Percent"], self.base_percent_mean, self.base_percent_std
                )
                data["Age"] = standardisation(data["Age"], self.age_mean, self.age_std)
                data["Weeks"] = standardisation(
                    data["Weeks"], self.weeks_mean, self.weeks_std
                )

            if self.normalization:
                data["Base_week"] = normalization(
                    data["Base_week"], self.base_week_max, self.base_week_min
                )
                data["Base_FVC"] = normalization(
                    data["Base_FVC"], self.base_fvc_max, self.base_fvc_min
                )
                data["Percent"] = normalization(
                    data["Percent"], self.base_percent_max, self.base_percent_min
                )
                data["Age"] = normalization(data["Age"], self.age_max, self.age_min)
                data["Weeks"] = normalization(
                    data["Weeks"], self.weeks_max, self.weeks_min
                )
                data["Min_week"] = normalization(
                    data["Min_week"], self.min_week_max, self.min_week_min
                )

        except NotFittedError:
            data["Sex"] = self.enc_sex.fit_transform(data["Sex"].values)
            data["SmokingStatus"] = self.enc_smok.fit_transform(
                data["SmokingStatus"].values
            )
            data = pd.concat(
                [
                    data.drop(columns=["SmokingStatus"]),
                    self.onehotenc_smok.fit_transform(
                        data["SmokingStatus"].values.reshape(-1, 1),
                        categories=self.enc_smok.classes_,
                        name="",
                        index=data.index,
                    ).astype(int),
                ],
                axis=1,
            )

            if self.standardisation:
                self.base_week_mean = data["Base_week"].mean()
                self.base_week_std = data["Base_week"].std()
                data["Base_week"] = standardisation(
                    data["Base_week"], self.base_week_mean, self.base_week_std
                )

                self.base_fvc_mean = data["Base_FVC"].mean()
                self.base_fvc_std = data["Base_FVC"].std()
                data["Base_FVC"] = standardisation(
                    data["Base_FVC"], self.base_fvc_mean, self.base_fvc_std
                )

                self.base_percent_mean = data["Percent"].mean()
                self.base_percent_std = data["Percent"].std()
                data["Percent"] = standardisation(
                    data["Percent"], self.base_percent_mean, self.base_percent_std
                )

                self.age_mean = data["Age"].mean()
                self.age_std = data["Age"].std()
                data["Age"] = standardisation(data["Age"], self.age_mean, self.age_std)

                self.weeks_mean = data["Weeks"].mean()
                self.weeks_std = data["Weeks"].std()
                data["Weeks"] = standardisation(
                    data["Weeks"], self.weeks_mean, self.weeks_std
                )

            if self.normalization:
                self.base_week_min = data["Base_week"].min()
                self.base_week_max = data["Base_week"].max()
                data["Base_week"] = normalization(
                    data["Base_week"], self.base_week_max, self.base_week_min
                )

                self.base_fvc_min = data["Base_FVC"].min()
                self.base_fvc_max = data["Base_FVC"].max()
                data["Base_FVC"] = normalization(
                    data["Base_FVC"], self.base_fvc_max, self.base_fvc_min
                )

                self.base_percent_min = data["Percent"].min()
                self.base_percent_max = data["Percent"].max()
                data["Percent"] = normalization(
                    data["Percent"], self.base_percent_max, self.base_percent_min
                )

                self.age_min = data["Age"].min()
                self.age_max = data["Age"].max()
                data["Age"] = normalization(data["Age"], self.age_max, self.age_min)

                self.weeks_min = data["Weeks"].min()
                self.weeks_max = data["Weeks"].max()
                data["Weeks"] = normalization(
                    data["Weeks"], self.weeks_max, self.weeks_min
                )

                self.min_week_min = data["Min_week"].min()
                self.min_week_max = data["Min_week"].max()
                data["Min_week"] = normalization(
                    data["Min_week"], self.min_week_max, self.min_week_min
                )

        return data




## === cell 5
tr = raw_train.copy()
base = tr.sort_values(["Patient", "Weeks"]).groupby("Patient").first().reset_index()
base = base.rename(columns={"Weeks": "Min_week", "FVC": "Base_FVC"})
base["Weeks"] = base["Min_week"]
base["Base_week"] = base["Weeks"] - base["Min_week"]  # = 0

fit_cols = [
    "Patient",
    "Min_week",
    "Base_FVC",
    "Percent",
    "Age",
    "Sex",
    "SmokingStatus",
    "Weeks",
    "Base_week",
]
fit_df = base[fit_cols].copy()

data_prep = data_preparation(bool_normalization=True, bool_standard=False)
_ = data_prep(fit_df)  # fit

X_prediction = data_prep(X_prediction).sort_values("Patient").reset_index(drop=True)




## === cell 6
class ConvertToHU:
    def __call__(self, imgs, dicom):
        intercept = getattr(dicom, "RescaleIntercept", 0.0)
        slope = getattr(dicom, "RescaleSlope", 1.0)
        imgs = (np.array(imgs.to_list()) * slope + intercept).astype(np.int16)
        return imgs


convertohu = ConvertToHU()


class Clip:
    def __init__(self, bounds=(-1000, 500)):
        self.min = min(bounds)
        self.max = max(bounds)

    def __call__(self, image):
        image = image.copy()
        image[image < self.min] = self.min
        image[image > self.max] = self.max
        return image


clip = Clip(clip_bounds)


class MaskWatershed:
    def __init__(self, min_hu, iterations):
        self.min_hu = min_hu
        self.iterations = iterations

    def __call__(self, image, dicom):
        stack = []
        for slice_idx in range(image.shape[0]):
            sliced = image[slice_idx]
            stack.append(self.seperate_lungs(sliced, self.min_hu, self.iterations))
        return np.stack(stack)

    @staticmethod
    def seperate_lungs(image, min_hu, iterations):
        h, w = image.shape[0], image.shape[1]
        marker_internal, marker_external, marker_watershed = (
            MaskWatershed.generate_markers(image)
        )

        sobel_filtered_dx = ndimage.sobel(image, 1)
        sobel_filtered_dy = ndimage.sobel(image, 0)
        sobel_gradient = np.hypot(sobel_filtered_dx, sobel_filtered_dy)
        mx = np.max(sobel_gradient)
        if mx > 0:
            sobel_gradient *= 255.0 / mx

        watershed = segmentation.watershed(sobel_gradient, marker_watershed)

        outline = ndimage.morphological_gradient(watershed, size=(3, 3))
        outline = outline.astype(bool)

        blackhat_struct = [
            [0, 0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 0, 0],
        ]

        blackhat_struct = ndimage.iterate_structure(blackhat_struct, iterations)
        outline = outline | ndimage.black_tophat(outline, structure=blackhat_struct)

        lungfilter = np.bitwise_or(marker_internal, outline)
        lungfilter = ndimage.binary_closing(
            lungfilter, structure=np.ones((5, 5)), iterations=3
        )

        segmented = np.where(lungfilter == 1, image, min_hu * np.ones((h, w)))
        return segmented

    @staticmethod
    def generate_markers(image, threshold=-400):
        h, w = image.shape[0], image.shape[1]
        marker_internal = image < threshold
        marker_internal = segmentation.clear_border(marker_internal)
        marker_internal_labels = measure.label(marker_internal)

        areas = [r.area for r in measure.regionprops(marker_internal_labels)]
        areas.sort()

        if len(areas) > 2:
            for region in measure.regionprops(marker_internal_labels):
                if region.area < areas[-2]:
                    for coordinates in region.coords:
                        marker_internal_labels[coordinates[0], coordinates[1]] = 0

        marker_internal = marker_internal_labels > 0

        external_a = ndimage.binary_dilation(marker_internal, iterations=10)
        external_b = ndimage.binary_dilation(marker_internal, iterations=55)
        marker_external = external_b ^ external_a

        marker_watershed = np.zeros((h, w), dtype=int)
        marker_watershed += marker_internal * 255
        marker_watershed += marker_external * 128
        return marker_internal, marker_external, marker_watershed


maskwatershed = MaskWatershed(min_hu=min(clip_bounds), iterations=2)


class Normalize:
    def __init__(self, bounds=(-1000, 500)):
        self.min = min(bounds)
        self.max = max(bounds)

    def __call__(self, image):
        image = image.astype(np.float32)
        image = (image - self.min) / (self.max - self.min)
        return image


class ZeroCenter:
    def __init__(self, pre_calculated_mean):
        self.pre_calculated_mean = pre_calculated_mean

    def __call__(self, image):
        return image - self.pre_calculated_mean


normalize = Normalize(bounds=clip_bounds)
zerocenter = ZeroCenter(pre_calculated_mean=pre_calculated_mean)


def sort_function(x):
    return int(x.split(".")[0])


def _read(path, patients=[], desired_size=(60, 512, 512)):
    X = np.empty(
        np.concatenate(([len(patients), 1], np.array(desired_size))), dtype=np.float32
    )
    i = 0
    for patient in patients:
        patient_dir = os.path.join(path, patient)
        files = os.listdir(patient_dir)
        if len(files) == 0:
            X[i, 0, :, :, :] = 0.0
            i += 1
            continue

        list_patient_files = sorted(files, key=sort_function)
        first_path = os.path.join(patient_dir, list_patient_files[0])
        dicom0 = pydicom.dcmread(first_path)

        arrs = []
        for fn in list_patient_files:
            fp = os.path.join(patient_dir, fn)
            dcm = pydicom.dcmread(fp)
            arrs.append(dcm.pixel_array)
        arrs = pd.Series(arrs)
        df = convertohu(arrs, dicom0)

        df = zoom(df, np.array(desired_size) / np.array(df.shape), mode="nearest")
        X[i, 0, :, :, :] = zerocenter(normalize(maskwatershed(clip(df), dicom0)))
        i += 1
    return X




## === cell 7
class DataGenerator(K.utils.Sequence):
    def on_epoch_end(self):
        self.indices = np.arange(len(self.list_IDs))

    def __len__(self):
        return int(ceil(len(self.indices) / self.batch_size))

    def __init__(
        self,
        train,
        list_IDs,
        batch_size=1,
        desired_size=(10, 512, 512),
        img_path=TEST_PATH,
        *args,
        **kwargs,
    ):
        self.train = train
        self.list_IDs = list_IDs
        self.batch_size = batch_size
        self.desired_size = desired_size
        self.img_path = img_path
        self.on_epoch_end()

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indices]

        patients = self.train.loc[list_IDs_temp, "Patient"].unique()
        imgs = _read(self.img_path, patients=patients, desired_size=self.desired_size)
        return self.train.loc[list_IDs_temp, :].reset_index(drop=True), np.transpose(
            imgs, (0, 2, 3, 4, 1)
        )


pred_generator = None
if _TF_AVAILABLE and _HAVE_PYDICOM and _HAVE_SKIMAGE:
    pred_generator = DataGenerator(
        X_prediction,
        X_prediction.index,
        batch_size=BATCH_SIZE,
        desired_size=DESIRED_SIZE,
        img_path=TEST_PATH,
    )



## === cell 8
MODEL_DIR = "/kaggle/input/3d-cnn-mlp/model_9"
CNN_DIR = "/kaggle/input/3d-cnn-mlp/CNN_9"


def _load_savedmodel_callable(savedmodel_dir):
    obj = tf.saved_model.load(savedmodel_dir)
    sig = None
    if hasattr(obj, "signatures") and isinstance(obj.signatures, dict):
        for k in ("serving_default", "serve", "call"):
            if k in obj.signatures:
                sig = obj.signatures[k]
                return obj, sig, k
        if len(obj.signatures) > 0:
            k = sorted(obj.signatures.keys())[0]
            sig = obj.signatures[k]
            return obj, sig, k
    if callable(obj):
        return obj, None, "direct_call"
    raise ValueError(f"Could not load callable SavedModel from {savedmodel_dir}.")


model_obj = None
model_sig = None
model_sig_name = None

if _TF_AVAILABLE and _HAVE_PYDICOM and _HAVE_SKIMAGE:
    try:
        model_obj, model_sig, model_sig_name = _load_savedmodel_callable(MODEL_DIR)
        print("Loaded model SavedModel:", MODEL_DIR, "signature:", model_sig_name)
    except Exception as e:
        print(
            "Warning: failed to load CNN+MLP SavedModel; falling back to tabular-only.\n",
            repr(e),
        )
        model_obj = None
        model_sig = None



## === cell 9
SELECTED_COLUMNS = [
    "Weeks",
    "Percent",
    "Age",
    "Sex",
    "Min_week",
    "Base_FVC",
    "Base_week",
    "_Currently smokes",
    "_Ex-smoker",
    "_Never smoked",
]


def _ensure_selected_columns(df, selected):
    out = df.copy()
    for c in selected:
        if c not in out.columns:
            out[c] = 0
    return out[selected]


def _run_model_on_batch(model_obj, model_sig, X_tab, X_img):
    if model_sig is not None:
        try:
            out = model_sig(tabular=X_tab, image=X_img)
            return out
        except Exception:
            pass
        try:
            in_keys = list(model_sig.structured_input_signature[1].keys())
            if len(in_keys) == 2:
                out = model_sig(**{in_keys[0]: X_tab, in_keys[1]: X_img})
                return out
        except Exception:
            pass
        try:
            out = model_sig(X_tab, X_img)
            return out
        except Exception:
            pass

    out = model_obj([X_tab, X_img]) if callable(model_obj) else model_obj
    return out


def _fit_patient_linear_models(train_df: pd.DataFrame):
    """
    Tabular-only fallback: per-patient linear trend FVC = a + b*Weeks.
    """
    d = {}
    for pid, g in train_df.groupby("Patient"):
        gg = g[["Weeks", "FVC"]].dropna().sort_values("Weeks")
        if len(gg) == 0:
            continue
        x = gg["Weeks"].to_numpy(dtype=np.float32)
        y = gg["FVC"].to_numpy(dtype=np.float32)
        if len(gg) == 1 or np.allclose(x.var(), 0.0):
            a = float(y.mean())
            b = 0.0
            resid_std = float(np.std(y - a)) if len(gg) > 1 else 200.0
            d[pid] = (a, b, resid_std, int(len(gg)))
            continue
        b, a = np.polyfit(x, y, deg=1)  # y = b*x + a
        yhat = a + b * x
        resid = y - yhat
        resid_std = float(np.std(resid)) if resid.size > 1 else 200.0
        d[pid] = (float(a), float(b), resid_std, int(len(gg)))
    return d


def _calibrate_sigma_from_train(train_df: pd.DataFrame, patient_models: dict):
    resids = []
    for pid, g in train_df.groupby("Patient"):
        if pid not in patient_models:
            continue
        a, b, _, _ = patient_models[pid]
        gg = g[["Weeks", "FVC"]].dropna()
        if len(gg) == 0:
            continue
        x = gg["Weeks"].to_numpy(dtype=np.float32)
        y = gg["FVC"].to_numpy(dtype=np.float32)
        yhat = a + b * x
        r = (y - yhat).astype(np.float32)
        if r.size:
            resids.append(r)
    if not resids:
        return 250.0
    r = np.concatenate(resids)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return 250.0
    mad = np.median(np.abs(r - np.median(r)))
    mad_sigma = 1.4826 * mad
    q68 = np.quantile(np.abs(r), 0.68)
    sigma = float(max(mad_sigma, q68, 70.0))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 250.0
    return sigma


y_prediction = None

if (
    (_TF_AVAILABLE and _HAVE_PYDICOM and _HAVE_SKIMAGE)
    and (pred_generator is not None)
    and (model_obj is not None)
):
    preds = []
    for X1, X2 in pred_generator:
        X_tab = tf.convert_to_tensor(
            np.asarray(_ensure_selected_columns(X1, SELECTED_COLUMNS)), dtype=tf.float32
        )
        X_img = tf.convert_to_tensor(X2, dtype=tf.float32)
        out = _run_model_on_batch(model_obj, model_sig, X_tab, X_img)
        if isinstance(out, dict):
            out = list(out.values())[0]
        out = tf.convert_to_tensor(out).numpy()
        preds.append(out)
    y_prediction = np.vstack(preds)
else:
    patient_models = _fit_patient_linear_models(raw_train)
    global_sigma = _calibrate_sigma_from_train(raw_train, patient_models)

    n = len(X_prediction)
    weeks = X_prediction["Weeks"].to_numpy(dtype=np.float32)
    base_fvc = X_prediction["Base_FVC"].to_numpy(dtype=np.float32)
    pids = X_prediction["Patient"].values

    pred_fvc = np.empty(n, dtype=np.float32)
    pred_sigma = np.empty(n, dtype=np.float32)

    for i in range(n):
        pid = pids[i]
        w = float(weeks[i])
        if pid in patient_models:
            a, b, resid_std, nobs = patient_models[pid]
            pred_fvc[i] = a + b * w
            nobs_eff = max(int(nobs), 1)
            w_global = 1.0 / np.sqrt(nobs_eff)
            sigma = (1.0 - w_global) * float(max(resid_std, 70.0)) + w_global * float(
                max(global_sigma, 70.0)
            )
            pred_sigma[i] = float(max(sigma, 70.0))
        else:
            pred_fvc[i] = float(base_fvc[i])
            pred_sigma[i] = float(max(global_sigma, 70.0))

    y_prediction = np.zeros((n, 3), dtype=np.float32)
    y_prediction[:, 1] = pred_fvc
    y_prediction[:, 0] = pred_fvc - pred_sigma
    y_prediction[:, 2] = pred_fvc + pred_sigma



## === cell 10
fvc_pred = y_prediction[:, 1].astype(np.float32)

conf = ((y_prediction[:, 2] - y_prediction[:, 0]) * 0.5).astype(np.float32)
conf = np.where(np.isfinite(conf), conf, 200.0)
conf = np.maximum(conf, 70.0)

sub = pd.DataFrame(
    {
        "Patient_Week": X_prediction["Patient_Week"].values,
        "FVC": fvc_pred,
        "Confidence": conf,
    }
)

sample = pd.read_csv(
    "/kaggle/input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)[["Patient_Week"]]
sub = sample.merge(sub, on="Patient_Week", how="left")

sub["FVC"] = sub["FVC"].fillna(raw_test["FVC"].median()).astype(float)
sub["Confidence"] = sub["Confidence"].fillna(200.0).astype(float)
sub["Confidence"] = sub["Confidence"].clip(lower=70.0)

sub["FVC"] = np.where(
    np.isfinite(sub["FVC"].values), sub["FVC"].values, raw_test["FVC"].median()
)
sub["Confidence"] = np.where(
    np.isfinite(sub["Confidence"].values), sub["Confidence"].values, 200.0
)
sub["Confidence"] = sub["Confidence"].clip(lower=70.0)

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("Columns:", list(sub.columns))
print("Any NA?:", sub.isna().any().to_dict())
