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
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
protobuf==6.33.0
scipy==1.15.3
seaborn==0.12.2
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
tf_keras==2.18.0
tqdm==4.67.1
wandb==0.21.0

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

-6.8426

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -8.39439) has done: 'I fix the root import/runtime failure coming from `wandb`/protobuf by making W&B optional and safely disabled by default in this Kaggle environment, which also prevents the downstream `NameError`s (since cell 0 never finished). I also remove the hard dependency on an external `pfutils` module by inlining minimal equivalents (data loading/feature engineering, KFold splitting, model builder, and a Keras `Sequence` generator) that preserve the same core semantics your notebook expects. Finally, I ensure the script always writes a valid `submission.csv` with the exact required columns and row alignment with `sample_submission.csv`, so you get a valid Kaggle submission end-to-end.'
- What this solution (achieved -8.39073) has done: 'I fix the runtime crash in the first cell caused by an incompatible `wandb`/protobuf import by fully disabling W&B by default and never importing it unless explicitly enabled, which keeps the rest of the pipeline intact. Then I make a minimal, score-improving correction to the submission post-processing: the current code incorrectly takes `abs()` of the FVC predictions (which can flip negative predictions to positive and harm accuracy); instead we only enforce positivity/clipping on the uncertainty (sigma) while leaving the FVC mean unconstrained. This change preserves the trained model and loss, but aligns inference with the metric and should move the score upward toward the target band. The script still train the same way and always write a valid `submission.csv` with the required columns and row order.'
- What this solution (achieved -8.5417) has done: 'I fix the early runtime crash (`MessageFactory.GetPrototype`) by fully preventing `wandb` (and any protobuf-dependent W&B code) from importing unless explicitly enabled, since it’s not needed for training/submission here. Then I make a minimal, score-improving calibration tweak aligned with the competition metric: learn a single global multiplier for the predicted uncertainty (`Confidence`) using out-of-fold validation predictions, and apply it to test predictions (with the required `>=70` clipping). This keeps your model, loss, folds, and training loop intact while nudging the Laplace log-likelihood upward toward the target by better matching sigma scale to residuals. The script still run end-to-end and always write a valid `submission.csv` with the correct columns and row order.'
- What this solution (achieved -8.65993) has done: 'I fix the runtime crash in the first cell by preventing `wandb` (which is incompatible with the current protobuf runtime in this environment) from being imported at all unless explicitly enabled, instead of attempting the import and failing before the notebook can proceed. Then I keep your existing training, CV, and sigma-multiplier calibration logic intact, but make fold assignment deterministic-yet-shuffled (seeded) at the patient level so each fold is a more representative mix; this is a minimal change that typically improves generalization and should move the score upward toward your target. Finally, I ensure the pipeline always reaches the submission-writing step and emits a valid `submission.csv` with the correct columns and row order.'
- What this solution (achieved -8.56153) has done: 'I fix the immediate runtime crash in cell 0 by hard-disabling `wandb` imports (the protobuf incompatibility is triggered even by attempting to import W&B in this environment), so the notebook runs end-to-end again. Then, to move the score upward toward your target with minimal semantic change, I correct the fold construction used by `get_fold_indices()` to match the intended deterministic-but-shuffled patient-level assignment already used during training; previously `fold_pos` could be inconsistent with the actual fold splits (and would break non-generator mode). Finally, I keep your existing sigma-multiplier calibration intact and ensure the submission is always written as `submission.csv` with the required columns and row order.'
- What this solution (achieved -8.61206) has done: 'I fix the immediate runtime crash by fully preventing `wandb` from being imported unless explicitly enabled, because the protobuf error is triggered during import in this environment. Then I make a minimal score-improving correction to the sigma calibration by expanding the candidate multiplier grid so it can better match the scale of residuals without changing the model, folds, training loop, or loss. Finally, I keep the submission-writing logic intact but add a small safety check to guarantee the output CSV has the exact required columns and row count matching `sample_submission.csv`.'
- What this solution (achieved -8.66568) has done: 'I fix the immediate runtime crash by never attempting to import `wandb` unless it is explicitly enabled, because in this Kaggle environment the protobuf/W&B stack can throw `MessageFactory.GetPrototype` even before your try/except can recover. Then, to move the score upward toward your target with minimal semantic change, I slightly widen the sigma-multiplier calibration grid (still a single global scalar learned from OOF predictions) so the confidence scaling can better match residuals under the Laplace metric. Finally, I keep the existing training/inference logic intact while adding a small safety cast to ensure `test_data` is always a DataFrame before `.to_numpy()`, guaranteeing the submission is written as a valid `submission.csv`.'
- What this solution (achieved -8.67027) has done: 'I fix the immediate runtime crash coming from importing `wandb` (which triggers a protobuf `MessageFactory.GetPrototype` error in this environment) by hard-disabling W&B at the environment level and never importing it unless explicitly enabled, so the notebook can run end-to-end. Then, to move the score upward toward your target with minimal semantic change, I tighten the sigma calibration to optimize the *same* Laplace metric more precisely by searching a slightly richer 1D multiplier grid (still a single global scalar learned from OOF predictions). Finally, I add a small safety normalization for `test_data` to guarantee it is always a DataFrame and that the submission is written with the exact required columns and row count.'
- What this solution (achieved -8.68542) has done: 'I fix the immediate runtime crash by ensuring `wandb` is never imported (the protobuf `MessageFactory.GetPrototype` error happens at import time in this environment), while keeping your training loop and model unchanged. I also make the fold split consistent between `fold_pos` computation and the actual `fold_ids` used during CV (currently `get_fold_indices()` is called on the wrong dataframe, which can desync indices if generator mode is off or later changed). Finally, I keep your existing sigma-multiplier calibration logic but ensure all paths use the same deterministic patient-level fold assignment to slightly improve generalization (and thus move score upward toward the target), without changing architecture/loss/training regimen.'
- What this solution (achieved -8.67428) has done: 'I fix the immediate runtime crash in cell 1 by preventing any `wandb` import path from being triggered (the `MessageFactory.GetPrototype` error is a known protobuf/W&B incompatibility) while keeping all training/inference logic unchanged. Then I correct a logic bug hurting score: `get_train_data()` currently normalizes features but stores `mu/sig` on `train_pairs` and later you read them from `train` (the DataFrame returned), so test normalization silently doesn’t happen; I attach `mu/sig` to the returned `train` DataFrame so train/test use the same scaling. This is a minimal, metric-aligned fix (better calibrated inputs at inference) that should improve score toward your target without changing architecture, loss, folds, or training regimen. Finally, I keep the submission writer intact and ensure it always writes `submission.csv` with the correct columns and row count.'
- What this solution (achieved -8.70391) has done: 'I fix the immediate runtime crash in cell 0 by hard-blocking any accidental `wandb` import before TensorFlow/Keras can trigger it via callbacks/loggers, which resolves the protobuf `MessageFactory.GetPrototype` incompatibility. Then I fix the normalization logic bug so the computed `mu/sig` used for input scaling is attached to the returned `train` DataFrame (the one later read in cell 4), ensuring test features are scaled identically to training features. These changes are minimal, preserve your model/training loop/loss semantics, and should improve the score by making inference use correctly normalized inputs. The script still write a valid `submission.csv` with the required columns and row alignment.'
- What this solution (achieved -8.6586) has done: 'I fix the immediate runtime crash happening before any training starts by preventing the incompatible protobuf codepath from being imported at all (the `MessageFactory.GetPrototype` error comes from TF/Keras pulling in protobuf internals early in this environment). Then I make the seed/determinism and submission-writing path robust so the script always reaches `submission.csv` creation with the required columns and row order. These changes are score-neutral except for allowing the model to actually train and submit; the modeling/training loop/loss remain unchanged. I also keep W&B fully disabled but do it in a way that does not trigger the protobuf issue.'
- What this solution (achieved -8.64869) has done: 'I fix the immediate runtime crash in cell 1 (`MessageFactory` missing `GetPrototype`) by switching the protobuf implementation back to the default C++ runtime (and avoiding forced “python” protobuf), which is the common cause of this exact TensorFlow/protobuf mismatch in Kaggle images. I keep W&B fully disabled but remove any environment settings that inadvertently push protobuf into an incompatible mode. These changes are execution/stability-only and won’t change your model/training semantics, but allow the pipeline to run end-to-end and write `submission.csv`. No score-tuning changes are added beyond restoring correct execution.'

# 9. Code solution

## === cell 0
import os
import math
import numpy as np
import pandas as pd

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "cpp")

os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_START_METHOD"] = "thread"
os.environ["WANDB_API_KEY"] = ""
os.environ["WANDB_ENTITY"] = ""
os.environ["WANDB_PROJECT"] = ""

import sys
import types

if "wandb" not in sys.modules:
    dummy = types.ModuleType("wandb")

    def _noop(*args, **kwargs):
        return None

    dummy.init = _noop
    dummy.login = _noop
    dummy.finish = _noop
    dummy.log = _noop
    dummy.config = {}
    sys.modules["wandb"] = dummy

import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.utils import Sequence

from tqdm import tqdm

os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(0)
tf.random.set_seed(0)

WANDB = False
WandbCallback = None
wandb = None

SUBMIT = True
DATA_GENERATOR = True
TRAIN_ON_BACKWARD_WEEKS = False
PSEUDO_TEST_PATIENTS = 0

BASE_PATH = "/kaggle/input/osic-pulmonary-fibrosis-progression"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
TEST_CSV = os.path.join(BASE_PATH, "test.csv")
SAMPLE_SUB_CSV = os.path.join(BASE_PATH, "sample_submission.csv")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/2281841307.py in <cell line: 0>()
     36     sys.modules["wandb"] = dummy
     37 
---> 38 import tensorflow as tf
     39 from tensorflow.keras import layers, regularizers
     40 from tensorflow.keras.utils import Sequence

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
if SUBMIT:
    PSEUDO_TEST_PATIENTS = 0
    WANDB = False



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/55447505.py in <cell line: 0>()
----> 1 if SUBMIT:
      2     PSEUDO_TEST_PATIENTS = 0
      3     WANDB = False
      4 

NameError: name 'SUBMIT' is not defined

## === cell 2
FOLDS = 5
BATCH_SIZE = 128
NUMBER_FEATURES = 9
HIDDEN_LAYERS = [64, 64]
PREDICT_SLOPE = False

VALUE_GAUSSIAN_NOISE_ON_FVC = 140
GAUSSIAN_NOISE_CORRELATED = True
VALUE_GAUSSIAN_NOISE_ON_META = 0.3

ACTIVATION_FUNCTION = "swish"
MODIFIED_LOSS = True

DROP_OUT_RATE = 0
DROP_OUT_LAYERS = []

EPOCHS = 250

L2_REGULARIZATION = False
REGULARIZATION_CONSTANT = 0.0001

INPUT_NORMALIZATION = True
OUTPUT_NORMALIZATION = True

LEARNING_RATE_SCHEDULER = "exp"  # 'exp', 'cos' or None
MAX_LEARNING_RATE = 0.001
COSINE_CYCLES = 5
EPOCHS_PER_OOM_DECAY = 150

MODEL_NAME = "Submitalotoflosd"

config = dict(
    NUMBER_FEATURES=NUMBER_FEATURES,
    L2_REGULARIZATION=L2_REGULARIZATION,
    INPUT_NORMALIZATION=INPUT_NORMALIZATION,
    ACTIVATION_FUNCTION=ACTIVATION_FUNCTION,
    DROP_OUT_RATE=DROP_OUT_RATE,
    OUTPUT_NORMALIZATION=OUTPUT_NORMALIZATION,
    EPOCHS=EPOCHS,
    MAX_LEARNING_RATE=MAX_LEARNING_RATE,
    MODIFIED_LOSS=MODIFIED_LOSS,
    VALUE_GAUSSIAN_NOISE_ON_META=VALUE_GAUSSIAN_NOISE_ON_META,
    COSINE_CYCLES=COSINE_CYCLES,
    MODEL_NAME=MODEL_NAME,
    LEARNING_RATE_SCHEDULER=LEARNING_RATE_SCHEDULER,
    VALUE_GAUSSIAN_NOISE_ON_FVC=VALUE_GAUSSIAN_NOISE_ON_FVC,
    PREDICT_SLOPE=PREDICT_SLOPE,
    HIDDEN_LAYERS=HIDDEN_LAYERS,
    REGULARIZATION_CONSTANT=REGULARIZATION_CONSTANT,
    EPOCHS_PER_OOM_DECAY=EPOCHS_PER_OOM_DECAY,
    DROP_OUT_LAYERS=DROP_OUT_LAYERS,
    BATCH_SIZE=BATCH_SIZE,
    GAUSSIAN_NOISE_CORRELATED=GAUSSIAN_NOISE_CORRELATED,
)




## === cell 3
def _onehot_smoking(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["Currently smokes", "Ex-smoker", "Never smoked"]:
        if col not in df.columns:
            df[col] = 0.0
    if "SmokingStatus" in df.columns:
        df["Currently smokes"] = (df["SmokingStatus"] == "Currently smokes").astype(
            float
        )
        df["Ex-smoker"] = (df["SmokingStatus"] == "Ex-smoker").astype(float)
        df["Never smoked"] = (df["SmokingStatus"] == "Never smoked").astype(float)
    return df


def _encode_sex(df: pd.DataFrame) -> pd.DataFrame:
    if "Sex" in df.columns:
        df["Sex"] = (df["Sex"].astype(str).str.lower() == "male").astype(float)
    return df


def get_fold_indices(folds: int, train_df: pd.DataFrame, seed: int = 0):
    pats = train_df["Patient"].values
    unique_pats = pd.unique(pats)

    rng = np.random.RandomState(seed)
    shuffled = unique_pats.copy()
    rng.shuffle(shuffled)
    fold_id_by_patient = {p: (i % folds) for i, p in enumerate(shuffled)}
    fold_ids = np.array([fold_id_by_patient[p] for p in pats])

    fold_pos = [0]
    for f in range(folds):
        fold_rows = np.where(fold_ids == f)[0]
        fold_pos.append(fold_pos[-1] + len(fold_rows))
    return fold_pos, fold_ids


def get_exponential_decay_lr_callback(cfg: dict):
    max_lr = float(cfg["MAX_LEARNING_RATE"])
    decay_epochs = int(cfg.get("EPOCHS_PER_OOM_DECAY", 150))

    def schedule(epoch, lr):
        return max_lr * (10 ** (-(epoch / max(decay_epochs, 1))))

    return tf.keras.callbacks.LearningRateScheduler(schedule, verbose=0)


def get_cosine_annealing_lr_callback(cfg: dict):
    max_lr = float(cfg["MAX_LEARNING_RATE"])
    epochs = int(cfg["EPOCHS"])
    cycles = int(cfg.get("COSINE_CYCLES", 5))

    def schedule(epoch, lr):
        if epochs <= 1:
            return max_lr
        t = epoch / (epochs - 1)
        return max_lr * 0.5 * (1 + math.cos(2 * math.pi * cycles * t))

    return tf.keras.callbacks.LearningRateScheduler(schedule, verbose=0)


def _laplace_nll_loss(y_true, y_pred):
    fvc_true = y_true[:, 0:1]
    fvc_pred = y_pred[:, 0:1]
    sigma = y_pred[:, 1:2]
    sigma = tf.maximum(tf.abs(sigma), 70.0)
    delta = tf.minimum(tf.abs(fvc_true - fvc_pred), 1000.0)
    sq2 = tf.constant(np.sqrt(2.0), dtype=tf.float32)
    metric = -(sq2 * delta / sigma) - tf.math.log(sq2 * sigma)
    return -tf.reduce_mean(metric)


def build_model(cfg: dict):
    n_in = int(cfg["NUMBER_FEATURES"])
    inp = layers.Input(shape=(n_in,), name="input_features")

    x = inp
    if cfg.get("INPUT_NORMALIZATION", True):
        x = layers.LayerNormalization()(x)

    if (
        cfg.get("VALUE_GAUSSIAN_NOISE_ON_META", 0)
        and cfg["VALUE_GAUSSIAN_NOISE_ON_META"] > 0
    ):
        x = layers.GaussianNoise(float(cfg["VALUE_GAUSSIAN_NOISE_ON_META"]))(x)

    if (
        cfg.get("VALUE_GAUSSIAN_NOISE_ON_FVC", 0)
        and cfg["VALUE_GAUSSIAN_NOISE_ON_FVC"] > 0
    ):

        def add_fvc_noise(t):
            noise = tf.random.normal(
                shape=(tf.shape(t)[0], 1),
                stddev=float(cfg["VALUE_GAUSSIAN_NOISE_ON_FVC"]),
            )
            left = t[:, :1]
            mid = t[:, 1:2] + noise
            right = t[:, 2:]
            return tf.concat([left, mid, right], axis=1)

        x = layers.Lambda(add_fvc_noise)(x)

    reg = None
    if cfg.get("L2_REGULARIZATION", False):
        reg = regularizers.l2(float(cfg.get("REGULARIZATION_CONSTANT", 1e-4)))

    for i, units in enumerate(cfg["HIDDEN_LAYERS"]):
        x = layers.Dense(
            int(units), activation=cfg["ACTIVATION_FUNCTION"], kernel_regularizer=reg
        )(x)
        if i in cfg.get("DROP_OUT_LAYERS", []):
            x = layers.Dropout(float(cfg.get("DROP_OUT_RATE", 0.0)))(x)

    out = layers.Dense(2, activation=None, name="pred")(x)

    model = tf.keras.Model(inputs=inp, outputs=out)
    opt = tf.keras.optimizers.Adam(learning_rate=float(cfg["MAX_LEARNING_RATE"]))

    if cfg.get("MODIFIED_LOSS", True):
        loss = _laplace_nll_loss
    else:
        loss = "mse"

    model.compile(optimizer=opt, loss=loss)
    return model


def get_train_data(
    train_csv_path: str,
    pseudo_test_patients: int,
    input_normalization: bool,
    train_on_backward_weeks: bool,
):
    train = pd.read_csv(train_csv_path)
    train = train.sort_values(["Patient", "Weeks"]).reset_index(drop=True)
    train = _encode_sex(train)
    train = _onehot_smoking(train)

    g = train.groupby("Patient")
    rows = []
    for p, pdf in g:
        pdf = pdf.sort_values("Weeks")
        base = pdf.iloc[0]
        for _, r in pdf.iterrows():
            weekdiff = float(r["Weeks"] - base["Weeks"])
            if (not train_on_backward_weeks) and weekdiff < 0:
                continue
            rows.append(
                dict(
                    Patient=p,
                    Weeks=float(base["Weeks"]),
                    FVC=float(base["FVC"]),
                    Percent=float(base["Percent"]),
                    Age=float(base["Age"]),
                    Sex=float(base["Sex"]),
                    **{
                        "Currently smokes": float(base["Currently smokes"]),
                        "Ex-smoker": float(base["Ex-smoker"]),
                        "Never smoked": float(base["Never smoked"]),
                    },
                    Weekdiff_target=weekdiff,
                    TargetFVC=float(r["FVC"]),
                )
            )
    train_pairs = pd.DataFrame(rows)

    feats = train_pairs[
        [
            "Weeks",
            "FVC",
            "Percent",
            "Age",
            "Sex",
            "Currently smokes",
            "Ex-smoker",
            "Never smoked",
            "Weekdiff_target",
        ]
    ].astype(float)

    if input_normalization:
        mu = feats.mean(axis=0)
        sig = feats.std(axis=0).replace(0, 1.0)
        feats = (feats - mu) / sig
        train_pairs.attrs["mu"] = mu
        train_pairs.attrs["sig"] = sig

    data = {"input_features": feats}
    labels = pd.DataFrame(
        {
            "TargetFVC": train_pairs["TargetFVC"].astype(float),
            "Dummy": np.zeros(len(train_pairs), dtype=float),
        }
    )
    return train_pairs, data, labels


def get_test_data(test_csv_path: str, input_normalization: bool):
    test = pd.read_csv(test_csv_path)
    test = test.sort_values(["Patient", "Weeks"]).reset_index(drop=True)
    test = _encode_sex(test)
    test = _onehot_smoking(test)

    submission = pd.read_csv(SAMPLE_SUB_CSV)
    pw = submission["Patient_Week"].str.split("_", expand=True)
    submission_pat = pw[0].values
    submission_week = pw[1].astype(int).values

    base_map = test.set_index("Patient").to_dict(orient="index")

    rows = []
    for p, w in zip(submission_pat, submission_week):
        base = base_map[p]
        weekdiff = float(w - base["Weeks"])
        rows.append(
            dict(
                Patient=p,
                Weeks=float(base["Weeks"]),
                FVC=float(base["FVC"]),
                Percent=float(base["Percent"]),
                Age=float(base["Age"]),
                Sex=float(base["Sex"]),
                **{
                    "Currently smokes": float(base.get("Currently smokes", 0.0)),
                    "Ex-smoker": float(base.get("Ex-smoker", 0.0)),
                    "Never smoked": float(base.get("Never smoked", 0.0)),
                },
                Weekdiff_target=weekdiff,
            )
        )
    test_pairs = pd.DataFrame(rows)
    feats = test_pairs[
        [
            "Weeks",
            "FVC",
            "Percent",
            "Age",
            "Sex",
            "Currently smokes",
            "Ex-smoker",
            "Never smoked",
            "Weekdiff_target",
        ]
    ].astype(float)

    return feats, submission


def get_pseudo_test_data(
    train_csv_path: str, pseudo_test_patients: int, input_normalization: bool
):
    train = (
        pd.read_csv(train_csv_path)
        .sort_values(["Patient", "Weeks"])
        .reset_index(drop=True)
    )
    pats = pd.unique(train["Patient"])
    pts = pats[: int(pseudo_test_patients)]
    pseudo = train[train["Patient"].isin(pts)].copy()
    rows = []
    checks = []
    for p, pdf in pseudo.groupby("Patient"):
        pdf = pdf.sort_values("Weeks")
        base = pdf.iloc[0]
        target = pdf.iloc[-1]
        weekdiff = float(target["Weeks"] - base["Weeks"])
        rows.append(
            dict(
                Patient=p,
                Weeks=float(base["Weeks"]),
                FVC=float(base["FVC"]),
                Percent=float(base["Percent"]),
                Age=float(base["Age"]),
                Sex=float((str(base["Sex"]).lower() == "male")),
                SmokingStatus=str(base["SmokingStatus"]),
                Weekdiff_target=weekdiff,
            )
        )
        checks.append(dict(Patient=p, TargetFVC=float(target["FVC"])))
    test_pairs = pd.DataFrame(rows)
    test_pairs = _encode_sex(test_pairs)
    test_pairs = _onehot_smoking(test_pairs)
    feats = test_pairs[
        [
            "Weeks",
            "FVC",
            "Percent",
            "Age",
            "Sex",
            "Currently smokes",
            "Ex-smoker",
            "Never smoked",
            "Weekdiff_target",
        ]
    ].astype(float)
    check = pd.DataFrame(checks)
    return feats, check


class DataGenerator(Sequence):
    def __init__(self, list_IDs, cfg, validation=False):
        self.list_IDs = list_IDs
        self.cfg = cfg
        self.validation = validation
        self.X = np.load("train_data.npy")
        self.y = np.load("train_labels.npy")
        self.batch_size = int(cfg["BATCH_SIZE"])

    def __len__(self):
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        inds = self.list_IDs[index * self.batch_size : (index + 1) * self.batch_size]
        Xb = self.X[inds].astype(np.float32)
        yb = self.y[inds].astype(np.float32)
        return Xb, yb




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3402768736.py in <cell line: 0>()
    300 
    301 
--> 302 class DataGenerator(Sequence):
    303     def __init__(self, list_IDs, cfg, validation=False):
    304         self.list_IDs = list_IDs

NameError: name 'Sequence' is not defined

## === cell 4
if SUBMIT:
    test_data, submission = get_test_data(TEST_CSV, INPUT_NORMALIZATION)

train, data, labels = get_train_data(
    TRAIN_CSV, PSEUDO_TEST_PATIENTS, INPUT_NORMALIZATION, TRAIN_ON_BACKWARD_WEEKS
)

if SUBMIT and not isinstance(test_data, pd.DataFrame):
    test_data = pd.DataFrame(test_data)

if INPUT_NORMALIZATION:
    mu = train.attrs.get("mu", None)
    sig = train.attrs.get("sig", None)
    if mu is not None and sig is not None and SUBMIT:
        test_data = ((test_data - mu) / sig).astype(np.float32)

if PSEUDO_TEST_PATIENTS > 0:
    test_data, test_check = get_pseudo_test_data(
        TRAIN_CSV, PSEUDO_TEST_PATIENTS, INPUT_NORMALIZATION
    )



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3811765401.py in <cell line: 0>()
----> 1 if SUBMIT:
      2     test_data, submission = get_test_data(TEST_CSV, INPUT_NORMALIZATION)
      3 
      4 train, data, labels = get_train_data(
      5     TRAIN_CSV, PSEUDO_TEST_PATIENTS, INPUT_NORMALIZATION, TRAIN_ON_BACKWARD_WEEKS

NameError: name 'SUBMIT' is not defined

## === cell 5
model = build_model(config)
model.summary()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1116904041.py in <cell line: 0>()
----> 1 model = build_model(config)
      2 model.summary()
      3 

/tmp/ipykernel_11/3402768736.py in build_model(cfg)
     72 def build_model(cfg: dict):
     73     n_in = int(cfg["NUMBER_FEATURES"])
---> 74     inp = layers.Input(shape=(n_in,), name="input_features")
     75 
     76     x = inp

NameError: name 'layers' is not defined

## === cell 6
fold_pos, fold_ids = get_fold_indices(FOLDS, train, seed=0)
print(fold_pos)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/752043331.py in <cell line: 0>()
----> 1 fold_pos, fold_ids = get_fold_indices(FOLDS, train, seed=0)
      2 print(fold_pos)
      3 

NameError: name 'train' is not defined

## === cell 7
if DATA_GENERATOR:
    train_data = train[
        [
            "Weeks",
            "FVC",
            "Percent",
            "Age",
            "Sex",
            "Currently smokes",
            "Ex-smoker",
            "Never smoked",
            "Weekdiff_target",
        ]
    ]
    train_labels = labels[["TargetFVC", "Dummy"]]
    np.save("train_data.npy", train_data.to_numpy(dtype=np.float32))
    np.save("train_labels.npy", train_labels.to_numpy(dtype=np.float32))




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3853716603.py in <cell line: 0>()
----> 1 if DATA_GENERATOR:
      2     train_data = train[
      3         [
      4             "Weeks",
      5             "FVC",

NameError: name 'DATA_GENERATOR' is not defined

## === cell 8
def _laplace_metric_np(fvc_true, fvc_pred, sigma):
    sigma_clip = np.maximum(np.abs(sigma), 70.0)
    delta = np.minimum(np.abs(fvc_true - fvc_pred), 1000.0)
    sq2 = np.sqrt(2.0)
    metric = -(sq2 * delta / sigma_clip) - np.log(sq2 * sigma_clip)
    return metric


def _calibrate_sigma_multiplier(fvc_true, fvc_pred, sigma_pred):
    base_sigma = np.maximum(np.abs(sigma_pred), 70.0)
    fvc_true = np.asarray(fvc_true, dtype=np.float32)
    fvc_pred = np.asarray(fvc_pred, dtype=np.float32)

    ks_coarse = np.array(
        [
            0.15,
            0.2,
            0.25,
            0.3,
            0.35,
            0.4,
            0.45,
            0.5,
            0.55,
            0.6,
            0.65,
            0.7,
            0.75,
            0.8,
            0.85,
            0.9,
            0.95,
            1.0,
            1.05,
            1.1,
            1.15,
            1.2,
            1.25,
            1.3,
            1.35,
            1.4,
            1.45,
            1.5,
            1.6,
            1.65,
            1.7,
            1.8,
            1.9,
            2.0,
            2.1,
            2.2,
            2.3,
            2.5,
            2.7,
            3.0,
            3.3,
            3.6,
            4.0,
        ],
        dtype=np.float32,
    )

    best_k = 1.0
    best_score = -1e18
    for k in ks_coarse:
        m = _laplace_metric_np(fvc_true, fvc_pred, base_sigma * k).mean()
        if m > best_score:
            best_score = m
            best_k = float(k)

    lo = max(0.05, best_k * 0.7)
    hi = best_k * 1.3
    ks_fine = np.linspace(lo, hi, 41, dtype=np.float32)
    for k in ks_fine:
        m = _laplace_metric_np(fvc_true, fvc_pred, base_sigma * k).mean()
        if m > best_score:
            best_score = m
            best_k = float(k)

    return best_k, float(best_score)


predictions = []

oof_fvc_pred = np.zeros(len(train), dtype=np.float32)
oof_sigma_pred = np.zeros(len(train), dtype=np.float32)

for fold in range(FOLDS):
    if DATA_GENERATOR:
        train_ID = np.where(fold_ids != fold)[0].tolist()
        val_ID = np.where(fold_ids == fold)[0].tolist()
        training_generator = DataGenerator(train_ID, config)
        validation_generator = DataGenerator(val_ID, config, validation=True)
    else:
        feats = data["input_features"]
        x_train = pd.concat(
            [feats.iloc[: fold_pos[fold]], feats.iloc[fold_pos[fold + 1] :]], axis=0
        )
        y_train = pd.concat(
            [labels.iloc[: fold_pos[fold]], labels.iloc[fold_pos[fold + 1] :]], axis=0
        )
        x_val = feats.iloc[fold_pos[fold] : fold_pos[fold + 1]]
        y_val = labels.iloc[fold_pos[fold] : fold_pos[fold + 1]]

    model = build_model(config)

    sv = tf.keras.callbacks.ModelCheckpoint(
        f"fold-{fold}.weights.h5",
        monitor="val_loss",
        verbose=0,
        save_best_only=True,
        save_weights_only=True,
        mode="min",
        save_freq="epoch",
    )
    callbacks = [sv]
    if LEARNING_RATE_SCHEDULER == "exp":
        callbacks.append(get_exponential_decay_lr_callback(config))
    if LEARNING_RATE_SCHEDULER == "cos":
        callbacks.append(get_cosine_annealing_lr_callback(config))

    print(fold + 1, "of", FOLDS)

    if DATA_GENERATOR:
        _ = model.fit(
            training_generator,
            validation_data=validation_generator,
            epochs=EPOCHS,
            verbose=0,
            callbacks=callbacks,
        )
    else:
        _ = model.fit(
            x_train.to_numpy(dtype=np.float32),
            y_train.to_numpy(dtype=np.float32),
            validation_data=(
                x_val.to_numpy(dtype=np.float32),
                y_val.to_numpy(dtype=np.float32),
            ),
            epochs=EPOCHS,
            verbose=0,
            callbacks=callbacks,
        )

    model.load_weights(f"fold-{fold}.weights.h5")

    val_idx = np.where(fold_ids == fold)[0]
    val_feats = data["input_features"].iloc[val_idx].to_numpy(dtype=np.float32)
    val_preds = model.predict(val_feats, batch_size=256, verbose=0).astype(np.float32)
    oof_fvc_pred[val_idx] = val_preds[:, 0]
    oof_sigma_pred[val_idx] = val_preds[:, 1]

    if SUBMIT or PSEUDO_TEST_PATIENTS > 0:
        preds = model.predict(
            test_data.to_numpy(dtype=np.float32), batch_size=256, verbose=0
        )
        predictions.append(preds.astype(np.float32))

sigma_multiplier = 1.0
if MODIFIED_LOSS:
    fvc_true_oof = labels["TargetFVC"].to_numpy(dtype=np.float32)
    sigma_multiplier, oof_metric = _calibrate_sigma_multiplier(
        fvc_true_oof, oof_fvc_pred, oof_sigma_pred
    )
    print("OOF sigma_multiplier:", sigma_multiplier, "OOF mean metric:", oof_metric)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1420916831.py in <cell line: 0>()
     83 predictions = []
     84 
---> 85 oof_fvc_pred = np.zeros(len(train), dtype=np.float32)
     86 oof_sigma_pred = np.zeros(len(train), dtype=np.float32)
     87 

NameError: name 'train' is not defined

## === cell 9
if SUBMIT:
    if len(predictions) == 0:
        raise RuntimeError(
            "No predictions were generated; check training loop and SUBMIT flag."
        )

    if PREDICT_SLOPE:
        preds = np.mean(predictions, axis=0)
        weekdiff = test_data["Weekdiff_target"].values.astype(np.float32)
        base_fvc = test_data["FVC"].values.astype(np.float32)
        submission["FVC"] = (
            base_fvc + preds[:, 0].astype(np.float32) * weekdiff
        ).astype(np.float32)
        submission["Confidence"] = np.abs(
            preds[:, 1].astype(np.float32) * weekdiff
        ).astype(np.float32)
    else:
        preds = np.array(predictions, dtype=np.float32)  # [F, N, 2]

        fvc_mean = np.mean(preds[:, :, 0], axis=0)  # [N]
        sigma_raw = np.abs(preds[:, :, 1])  # [F, N]
        sigma_var = np.power(sigma_raw, 2)  # average variances
        sigma = np.sqrt(np.mean(sigma_var, axis=0))  # [N]

        submission["FVC"] = fvc_mean.astype(np.float32)
        submission["Confidence"] = sigma.astype(np.float32)

    submission["Confidence"] = (
        submission["Confidence"].values.astype(np.float32) * float(sigma_multiplier)
    ).astype(np.float32)
    submission["Confidence"] = np.maximum(submission["Confidence"].values, 70.0)

    submission = submission[["Patient_Week", "FVC", "Confidence"]]
    if len(submission) != len(pd.read_csv(SAMPLE_SUB_CSV)):
        raise RuntimeError("Submission row count mismatch vs sample_submission.csv")

    submission.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", submission.shape)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2848504174.py in <cell line: 0>()
----> 1 if SUBMIT:
      2     if len(predictions) == 0:
      3         raise RuntimeError(
      4             "No predictions were generated; check training loop and SUBMIT flag."
      5         )

NameError: name 'SUBMIT' is not defined

## === cell 10
import matplotlib.pyplot as plt
from scipy.stats import gmean

if PSEUDO_TEST_PATIENTS > 0 and len(predictions) > 0:
    result = []
    for i in range(-20, 20):
        postprocess = np.abs(np.array(predictions))
        if i == 0:
            postprocess[:, :, 1] = gmean(postprocess[:, :, 1], axis=0)
            postprocess = np.mean(postprocess, axis=0)
        else:
            postprocess[:, :, 1] = np.power(postprocess[:, :, 1], i)
            postprocess = np.mean(postprocess, axis=0)
            postprocess[:, 1] = np.power(postprocess[:, 1], 1 / i)

        FVC_true = test_check["TargetFVC"].values
        FVC_pred = postprocess[:, 0]
        sigma = postprocess[:, 1]

        sigma_clip = np.maximum(np.abs(sigma), 70)
        delta = np.abs(FVC_true - FVC_pred)
        delta = np.minimum(delta, 1000)

        sq2 = np.sqrt(2)
        loss = (delta / sigma_clip) * sq2 + np.log(sigma_clip * sq2)
        result.append(np.mean(loss))

    plt.plot(np.arange(-20, 20), result)
    plt.show()

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2869701167.py in <cell line: 0>()
      2 from scipy.stats import gmean
      3 
----> 4 if PSEUDO_TEST_PATIENTS > 0 and len(predictions) > 0:
      5     result = []
      6     for i in range(-20, 20):

NameError: name 'PSEUDO_TEST_PATIENTS' is not defined
