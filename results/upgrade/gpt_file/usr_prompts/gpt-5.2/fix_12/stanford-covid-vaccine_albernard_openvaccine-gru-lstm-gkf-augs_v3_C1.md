# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Overview
Predict likely degradation rates at each base of an RNA molecule.

## Metric
Mean columnwise root mean squared error:

$\textrm{MCRMSE} = \frac{1}{N_{t}}\sum_{j=1}^{N_{t}}\sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_{ij} - \hat{y}_{ij})^2}$

where $N_{t}$ is the number of scored ground truth target columns, and $y$ and $\hat{y}$ are the actual and predicted values, respectively.

There are multiple ground truth values provided in the training data. While the submission format requires all 5 to be predicted, only the following are scored: reactivity, deg_Mg_pH10, and deg_Mg_50C.

## Submission Formats
For each sample `id` in the test set, you must predict targets for *each* sequence position (`seqpos`), one per row. If the length of the `sequence` of an `id` is, e.g., 107, then you should make 107 predictions. Positions greater than the `seq_scored` value of a sample are not scored, but still need a value in the solution file.

```csv
id_seqpos,reactivity,deg_Mg_pH10,deg_pH10,deg_Mg_50C,deg_50C
id_d190610e8_0,0.1,0.3,0.2,0.5,0.4
id_d190610e8_1,0.3,0.2,0.5,0.4,0.2
id_d190610e8_2,0.5,0.4,0.2,0.1,0.2
etc.
```

## Dataset 
- **train.json** - the training data
- **test.json** - the test set, without any columns associated with the ground truth.
- **sample_submission.csv** - a sample submission file in the correct format

#### Columns
- `id` - An arbitrary identifier for each sample.
- `seq_scored` - (68 in Train and Public Test, 68 in Private Test) Integer value denoting the number of positions used in scoring with predicted values. This should match the length of `reactivity`, `deg_*` and `*_error_*` columns.
- `seq_length` - (107 in Train and Public Test, 107 in Private Test) Integer values, denotes the length of `sequence`.
- `sequence` - (1x107 string in Train and Public Test, 107 in Private Test) Describes the RNA sequence, a combination of `A`, `G`, `U`, and `C` for each sample. Should be 107 characters long, and the first 68 bases should correspond to the 68 positions specified in `seq_scored` (note: indexed starting at 0).
- `structure` - (1x107 string in Train and Public Test, 107 in Private Test) An array of `(`, `)`, and `.` characters that describe whether a base is estimated to be paired or unpaired. Paired bases are denoted by opening and closing parentheses e.g. (....) means that base 0 is paired to base 5, and bases 1-4 are unpaired.
- `reactivity` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likely secondary structure of the RNA sample.
- `deg_pH10` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likelihood of degradation at the base/linkage after incubating without magnesium at high pH (pH 10).
- `deg_Mg_pH10` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likelihood of degradation at the base/linkage after incubating with magnesium in high pH (pH 10).
- `deg_50C` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likelihood of degradation at the base/linkage after incubating without magnesium at high temperature (50 degrees Celsius).
- `deg_Mg_50C` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likelihood of degradation at the base/linkage after incubating with magnesium at high temperature (50 degrees Celsius).
- `*_error_*` - An array of floating point numbers, should have the same length as the corresponding `reactivity` or `deg_*` columns, calculated errors in experimental values obtained in `reactivity` and `deg_*` columns.
- `predicted_loop_type` - (1x107 string) Describes the structural context (also referred to as 'loop type')of each character in `sequence`. Loop types assigned by bpRNA from Vienna RNAfold 2 structure. From the bpRNA_documentation: S: paired "Stem" M: Multiloop I: Internal loop B: Bulge H: Hairpin loop E: dangling End X: eXternal loop
    - `S/N filter` Indicates if the sample passed filters described below in `Additional Notes`.

#### Additional Notes
At the beginning of the competition, Stanford scientists have data on 2400 RNA sequences of length 107. For technical reasons, measurements cannot be carried out on the final bases of these RNA sequences, so we have experimental data (ground truth) in 5 conditions for the first 68 bases.

We have split out 240 of these 2400 sequences for a public test set to allow for continuous evaluation through the competition, on the public leaderboard. These sequences, in `test.json`, have been additionally filtered based on three criteria detailed below to ensure that this subset is not dominated by any large cluster of RNA molecules with poor data, which might bias the public leaderboard. The remaining 2160 sequences for which we have data are in `train.json`.

For our final and most important scoring (the Private Leaderbooard), Stanford scientists are carrying out measurements on 240 new RNAs. For these data, we expect to have measurements for the first 68 bases, again missing the ends of the RNA. These sequences constitute the 240 sequences in `test.json`.

For those interested in how the sequences in `test.json` were filtered, here were the steps to ensure a diverse and high quality test set for public leaderboard scoring:

1. Minimum value across all 5 conditions must be greater than -0.5.
2. Mean signal/noise across all 5 conditions must be greater than 1.0. [Signal/noise is defined as mean( measurement value over 68 nts )/mean( statistical error in measurement value over 68 nts)]
3. To help ensure sequence diversity, the resulting sequences were clustered into clusters with less than 50% sequence similarity, and the 240 test set sequences were chosen from clusters with 3 or fewer members. That is, any sequence in the test set should be sequence similar to at most 2 other sequences.

Note that these filters have not been applied to the 2160 RNAs in the public training data `train.json` -- some of those measurements have negative values or poor signal-to-noise, or some RNA sequences have near-identical sequences in that set. But we are providing all those data in case competitors can squeeze out more signal.

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
protobuf==6.33.0
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
            description.md (125 lines)
            sample_submission.csv (25681 lines)
            sample_submission.csv.zip (74.8 kB)
            test.json (240 lines)
            train.json (2160 lines)
            stanford-covid-vaccine/
                description.md (125 lines)
                sample_submission.csv (25681 lines)
                ... and 3 other files
                stanford-covid-vaccine/
        input/
            description.md (125 lines)
            sample_submission.csv (25681 lines)
            sample_submission.csv.zip (74.8 kB)
            test.json (240 lines)
            train.json (2160 lines)
            stanford-covid-vaccine/
                description.md (125 lines)
                sample_submission.csv (25681 lines)
                ... and 3 other files
                stanford-covid-vaccine/
        working/
            stanford-covid-vaccine/
                description.md (125 lines)
                sample_submission.csv (25681 lines)
                ... and 3 other files
                stanford-covid-vaccine/
```

-> data/sample_submission.csv has 25680 rows and 6 columns.
The columns are: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

-> data/stanford-covid-vaccine/sample_submission.csv has 25680 rows and 6 columns.
The columns are: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

-> data/stanford-covid-vaccine/test.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "index": {
      "type": "integer"
    },
    "id": {
      "type": "string"
    },
    "sequence": {
      "type": "string"
    },
    "structure": {
      "type": "string"
    },
    "predicted_loop_type": {
      "type": "string"
    },
    "seq_length": {
      "type": "integer"
    },
    "seq_scored": {
      "type": "integer"
    }
  },
  "required": [
    "id",
    "index",
    "predicted_loop_type",
    "seq_length",
    "seq_scored",
    "sequence",
    "structure"
  ]
}

-> data/stanford-covid-vaccine/train.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "index": {
      "type": "integer"
    },
    "id": {
      "type": "string"
    },
    "sequence": {
      "type": "string"
    },
    "structure": {
      "type": "string"
    },
    "predicted_loop_type": {
      "type": "string"
    },
    "signal_to_noise": {
      "type": "number"
    },
    "SN_filter": {
      "type": "integer"
    },
    "seq_length": {
      "type": "integer"
    },
    "seq_scored": {
      "type": "integer"
    },
    "reactivity_error": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_Mg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_Mg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "reactivity": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_Mg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_Mg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    }
  },
  "required": [
    "SN_filter",
    "deg_50C",
    "deg_Mg_50C",
    "deg_Mg_pH10",
    "deg_error_50C",
    "deg_error_Mg_50C",
    "deg_error_Mg_pH10",
    "deg_error_pH10",
    "deg_pH10",
    "id",
    "index",
    "predicted_loop_type",
    "reactivity",
    "reactivity_error",
    "seq_length",
    "seq_scored",
    "sequence",
    "signal_to_noise",
    "structure"
  ]
}

-> data/test.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "index": {
      "type": "integer"
    },
    "id": {
      "type": "string"
    },
    "sequence": {
      "type": "string"
    },
    "structure": {
      "type": "string"
    },
    "predicted_loop_type": {
      "type": "string"
    },
    "seq_length": {
      "type": "integer"
    },
    "seq_scored": {
      "type": "integer"
    }
  },
  "required": [
    "id",
    "index",
    "predicted_loop_type",
    "seq_length",
    "seq_scored",
    "sequence",
    "structure"
  ]
}

-> data/train.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "index": {
      "type": "integer"
    },
    "id": {
      "type": "string"
    },
    "sequence": {
      "type": "string"
    },
    "structure": {
      "type": "string"
    },
    "predicted_loop_type": {
      "type": "string"
    },
    "signal_to_noise": {
      "type": "number"
    },
    "SN_filter": {
      "type": "integer"
    },
    "seq_length": {
      "type": "integer"
    },
    "seq_scored": {
      "type": "integer"
    },
    "reactivity_error": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_Mg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_Mg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "reactivity": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_Mg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_Mg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    }
  },
  "required": [
    "SN_filter",
    "deg_50C",
    "deg_Mg_50C",
    "deg_Mg_pH10",
    "deg_error_50C",
    "deg_error_Mg_50C",
    "deg_error_Mg_pH10",
    "deg_error_pH10",
    "deg_pH10",
    "id",
    "index",
    "predicted_loop_type",
    "reactivity",
    "reactivity_error",
    "seq_length",
    "seq_scored",
    "sequence",
    "signal_to_noise",
    "structure"
  ]
}

-> input/sample_submission.csv has 25680 rows and 6 columns.
The columns are: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

-> (stopped after 10 files for performance)

# 5. Target score

0.39105

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.24784) has done: 'I fix the initial runtime crash by removing the incompatible `tensorflow_addons` import/usage that triggers the protobuf `MessageFactory.GetPrototype` error in this environment. Then I fix the Keras Functional error by replacing the raw `tf.reshape` call with a Keras layer (`Reshape`) so the model can be built and trained. Finally, I make inference/submission robust to the actual test set (only length 107 here) by removing the broken public/private split logic and always generating 107 predictions per id, then aligning them to `sample_submission.csv` and writing a valid `submission.csv`.'
- What this solution (achieved 0.24797) has done: 'I fix the immediate runtime crash by avoiding TensorFlow’s protobuf “GetPrototype” path via a safe import order and by forcing the pure-Python protobuf implementation before importing TensorFlow. Then I keep the model/training logic intact but make the inference model architecture consistent with training by reusing the same model and padding predictions from 68 to 107 instead of rebuilding a different “pred_len=107” head (which can degrade accuracy and is unnecessary). Finally, I ensure submission rows align exactly with `sample_submission.csv` and that all required columns are present, producing a valid `submission.csv` end-to-end.'
- What this solution (achieved 0.24873) has done: 'I fix the protobuf/TensorFlow crash in the first cell by forcing the Python protobuf implementation *before* any protobuf/TensorFlow-related imports and by importing `google.protobuf` after setting the env var (this avoids the `MessageFactory.GetPrototype` path that breaks in this environment). Then I keep the existing model/training/inference logic the same, but remove the unnecessary rebuild of separate inference models by directly reusing the trained models for prediction (score-neutral but reduces weight/graph mismatch risk). Finally, I keep the submission alignment via `sample_submission.csv` but add strict checks to guarantee every `id_seqpos` is produced and the output CSV is valid.'
- What this solution (achieved 0.24909) has done: 'I fix the immediate runtime crash caused by the protobuf/TensorFlow incompatibility by enforcing the pure-Python protobuf implementation and then importing TensorFlow in a way that avoids the `MessageFactory.GetPrototype` path that’s failing in this Kaggle image. I also make the environment deterministic and safer by disabling XLA (which can trigger protobuf descriptor paths) while leaving your model/training/inference logic unchanged. The rest of the pipeline (data loading, model definitions, training loops, prediction padding to 107, blending, and sample_submission alignment) remain the same to preserve your achieved score behavior while ensuring the notebook runs end-to-end and writes `submission.csv`. This should be score-neutral (or extremely close) and primarily restores execution stability.'
- What this solution (achieved 0.24118) has done: 'I fix the current runtime crash by removing the protobuf implementation forcing that is incompatible with the TensorFlow 2.18 + protobuf 6.x stack in this environment, and instead rely on the default C++ protobuf (which avoids the `MessageFactory.GetPrototype` AttributeError). I also make the TF import safer by enabling the legacy Keras mode for TF 2.18 to reduce version-mismatch issues without changing your model/training logic. Everything else (data loading, preprocessing, model architecture, training loop, prediction padding to 107, blending, and submission alignment) remain unchanged to preserve your current score behavior (already better than the target). The result run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.24134) has done: 'I fix the immediate runtime crash occurring at TensorFlow import by forcing the pure-Python protobuf implementation before any TensorFlow/protobuf imports, which avoids the `MessageFactory.GetPrototype` incompatibility seen in this Kaggle environment. I keep your model, training loop, inference, padding-to-107, blending, and submission alignment logic unchanged to preserve evaluation semantics and keep the score behavior as close as possible. I also make the import order explicit and add a small safety check around the protobuf setup so the notebook runs end-to-end reliably and always writes a valid `submission.csv` with the required columns and row alignment.'
- What this solution (achieved 0.24138) has done: 'The crash happens before training because forcing the pure-Python protobuf implementation is incompatible with TensorFlow 2.18 + protobuf 6.x in this environment and triggers the `MessageFactory.GetPrototype` error. I remove that protobuf override and keep the rest of your pipeline (data loading, preprocessing, model definitions, training, padding to 107, blending, and submission alignment) unchanged so it runs end-to-end. Since your current score (0.24134, lower-is-better) is already better than the target (0.39105), I won’t make any score-improving changes—only the minimal stability fix needed to execute and write a valid `submission.csv`.'
- What this solution (achieved 0.24125) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by pinning the protobuf runtime to the pure-Python implementation *before* any TensorFlow/protobuf imports, which is the safest way to avoid the broken C++ protobuf descriptor path in this environment. I keep your model/training/inference logic intact (same architecture, epochs, blending, padding, and submission alignment), changing only the import/bootstrap to restore end-to-end execution. I also add a tiny fallback to automatically retry TF import with the other protobuf backend if the first attempt fails, so the notebook reliably runs regardless of the underlying Kaggle image. Since your current score (0.24138, lower-is-better) is already better than the target (0.39105), I not make any score-improving changes.'
- What this solution (achieved 0.24098) has done: 'I fix the TensorFlow import crash by removing the forced pure-Python protobuf override (which is what triggers `MessageFactory.GetPrototype` with TF 2.18 + protobuf 6.x here) and instead rely on the default protobuf backend, keeping only the safe TF/XLA/legacy-keras environment settings. I also simplify the “safe import” helper to avoid reloading TensorFlow (which is unreliable) and just fail fast with a clear error if TF can’t import. No model/training/prediction logic is changed, so your score behavior should remain essentially the same (already better than the target), but the notebook run end-to-end and produce a valid `submission.csv`.'
- What this solution (achieved 0.24126) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by adding a small, robust “protobuf backend fallback” import routine that tries the default protobuf runtime first and, only if that fails, retries with the pure-Python protobuf runtime in a fresh process. This is the minimal change that unblocks execution while keeping all model/training/inference logic intact, so your score behavior should remain essentially unchanged (and you’re already better than the target since lower is better). I also keep the existing deterministic seeding and submission alignment checks, and ensure the script always writes a valid `submission.csv` in the working directory. No architecture, loss, epochs, blending weights, or feature extraction are changed.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault(
    "TF_XLA_FLAGS", "--tf_xla_auto_jit=0 --tf_xla_enable_xla_devices=false"
)
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import warnings

warnings.filterwarnings("ignore")

import sys
import subprocess


def import_tensorflow_robust():
    if "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION" not in os.environ:
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"

    try:
        import tensorflow as tf  # noqa: F401

        return tf
    except Exception as e:
        msg = str(e)
        protobuf_related = ("protobuf" in msg.lower()) or ("GetPrototype" in msg)
        if (not protobuf_related) or os.environ.get(
            "__TF_PROTOBUF_RETRY__", "0"
        ) == "1":
            raise

        env = os.environ.copy()
        env["__TF_PROTOBUF_RETRY__"] = "1"
        current = env.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "cpp").lower()
        env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = (
            "python" if current != "python" else "cpp"
        )
        env.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

        cmd = [sys.executable] + sys.argv
        raise SystemExit(subprocess.call(cmd, env=env))


tf = import_tensorflow_robust()

import gc, random, math, json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

import tensorflow.keras.backend as K
import tensorflow.keras.layers as L

from sklearn.model_selection import train_test_split, KFold

SEED = 34
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

print("TF version:", tf.__version__)
print("TF_USE_LEGACY_KERAS:", os.environ.get("TF_USE_LEGACY_KERAS"))
print(
    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:",
    os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"),
)
print("GPU devices:", tf.config.list_physical_devices("GPU"))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/3664794983.py in import_tensorflow_robust()
     28     try:
---> 29         import tensorflow as tf  # noqa: F401
     30 

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

SystemExit                                Traceback (most recent call last)
    [... skipping hidden 1 frame]

/tmp/ipykernel_11/3664794983.py in <cell line: 0>()
     53 
---> 54 tf = import_tensorflow_robust()
     55 

/tmp/ipykernel_11/3664794983.py in import_tensorflow_robust()
     50         cmd = [sys.executable] + sys.argv
---> 51         raise SystemExit(subprocess.call(cmd, env=env))
     52 

SystemExit: 1

During handling of the above exception, another exception occurred:

TypeError                                 Traceback (most recent call last)
    [... skipping hidden 1 frame]

/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py in showtraceback(self, exc_tuple, filename, tb_offset, exception_only, running_compiled_code)
   2090                     stb = ['An exception has occurred, use %tb to see '
   2091                            'the full traceback.\n']
-> 2092                     stb.extend(self.InteractiveTB.get_exception_only(etype,
   2093                                                                      value))
   2094                 else:

/usr/local/lib/python3.11/dist-packages/IPython/core/ultratb.py in get_exception_only(self, etype, value)
    752         value : exception value
    753         """
--> 754         return ListTB.structured_traceback(self, etype, value)
    755 
    756     def show_exception_only(self, etype, evalue):

/usr/local/lib/python3.11/dist-packages/IPython/core/ultratb.py in structured_traceback(self, etype, evalue, etb, tb_offset, context)
    627             chained_exceptions_tb_offset = 0
    628             out_list = (
--> 629                 self.structured_traceback(
    630                     etype, evalue, (etb, chained_exc_ids),
    631                     chained_exceptions_tb_offset, context)

/usr/local/lib/python3.11/dist-packages/IPython/core/ultratb.py in structured_traceback(self, etype, value, tb, tb_offset, number_of_lines_of_context)
   1365         else:
   1366             self.tb = tb
-> 1367         return FormattedTB.structured_traceback(
   1368             self, etype, value, tb, tb_offset, number_of_lines_of_context)
   1369 

/usr/local/lib/python3.11/dist-packages/IPython/core/ultratb.py in structured_traceback(self, etype, value, tb, tb_offset, number_of_lines_of_context)
   1265         if mode in self.verbose_modes:
   1266             # Verbose modes need a full traceback
-> 1267             return VerboseTB.structured_traceback(
   1268                 self, etype, value, tb, tb_offset, number_of_lines_of_context
   1269             )

/usr/local/lib/python3.11/dist-packages/IPython/core/ultratb.py in structured_traceback(self, etype, evalue, etb, tb_offset, number_of_lines_of_context)
   1122         """Return a nice text document describing the traceback."""
   1123 
-> 1124         formatted_exception = self.format_exception_as_a_whole(etype, evalue, etb, number_of_lines_of_context,
   1125                                                                tb_offset)
   1126 

/usr/local/lib/python3.11/dist-packages/IPython/core/ultratb.py in format_exception_as_a_whole(self, etype, evalue, etb, number_of_lines_of_context, tb_offset)
   1080 
   1081 
-> 1082         last_unique, recursion_repeat = find_recursion(orig_etype, evalue, records)
   1083 
   1084         frames = self.format_records(records, last_unique, recursion_repeat)

/usr/local/lib/python3.11/dist-packages/IPython/core/ultratb.py in find_recursion(etype, value, records)
    380     # first frame (from in to out) that looks different.
    381     if not is_recursion_error(etype, value, records):
--> 382         return len(records), 0
    383 
    384     # Select filename, lineno, func_name to track frames with

TypeError: object of type 'NoneType' has no len()

## === cell 1
train = pd.read_json("/kaggle/input/stanford-covid-vaccine/train.json", lines=True)
test = pd.read_json("/kaggle/input/stanford-covid-vaccine/test.json", lines=True)
sample_sub = pd.read_csv("/kaggle/input/stanford-covid-vaccine/sample_submission.csv")

print(train.shape, test.shape, sample_sub.shape)
print("test seq_length unique:", sorted(test["seq_length"].unique().tolist()))



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1276076435.py in <cell line: 0>()
----> 1 train = pd.read_json("/kaggle/input/stanford-covid-vaccine/train.json", lines=True)
      2 test = pd.read_json("/kaggle/input/stanford-covid-vaccine/test.json", lines=True)
      3 sample_sub = pd.read_csv("/kaggle/input/stanford-covid-vaccine/sample_submission.csv")
      4 
      5 print(train.shape, test.shape, sample_sub.shape)

NameError: name 'pd' is not defined

## === cell 2
target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]



## === cell 3
token2int = {x: i for i, x in enumerate("().ACGUBEHIMSX")}
print("Vocab size:", len(token2int))




## === cell 4
def preprocess_inputs(df, cols=("sequence", "structure", "predicted_loop_type")):
    """
    Returns int32 array of shape (n_samples, seq_len, 3)
    """
    arr = (
        df.loc[:, list(cols)]
        .applymap(lambda seq: [token2int[x] for x in seq])
        .values.tolist()
    )
    x = np.array(arr, dtype=np.int32)  # (n, 3, seq_len)
    x = np.transpose(x, (0, 2, 1))  # (n, seq_len, 3)
    return x




## === cell 5
train_filt = train[train.signal_to_noise > 1].copy()

train_inputs = preprocess_inputs(train_filt)
train_labels = np.array(
    train_filt[target_cols].values.tolist(), dtype=np.float32
).transpose(
    (0, 2, 1)
)  # (n, 68, 5)

print("train_inputs:", train_inputs.shape, train_inputs.dtype)
print("train_labels:", train_labels.shape, train_labels.dtype)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3094398645.py in <cell line: 0>()
----> 1 train_filt = train[train.signal_to_noise > 1].copy()
      2 
      3 train_inputs = preprocess_inputs(train_filt)
      4 train_labels = np.array(
      5     train_filt[target_cols].values.tolist(), dtype=np.float32

NameError: name 'train' is not defined

## === cell 6
def gru_layer(hidden_dim, dropout):
    return tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(
            hidden_dim,
            dropout=dropout,
            return_sequences=True,
            kernel_initializer="orthogonal",
        )
    )


def lstm_layer(hidden_dim, dropout):
    return tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            hidden_dim,
            dropout=dropout,
            return_sequences=True,
            kernel_initializer="orthogonal",
        )
    )


def build_model(
    gru=False, seq_len=107, pred_len=68, dropout=0.4, embed_dim=75, hidden_dim=96
):

    inputs = tf.keras.layers.Input(shape=(seq_len, 3), dtype=tf.int32)

    embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=embed_dim)(
        inputs
    )

    reshaped = tf.keras.layers.Reshape((seq_len, 3 * embed_dim))(embed)

    reshaped = tf.keras.layers.SpatialDropout1D(0.2)(reshaped)

    if gru:
        hidden = gru_layer(hidden_dim, dropout)(reshaped)
        hidden = gru_layer(hidden_dim, dropout)(hidden)
        hidden = gru_layer(hidden_dim, dropout)(hidden)
    else:
        hidden = lstm_layer(hidden_dim, dropout)(reshaped)
        hidden = lstm_layer(hidden_dim, dropout)(hidden)
        hidden = lstm_layer(hidden_dim, dropout)(hidden)

    truncated = hidden[:, :pred_len]
    out = tf.keras.layers.Dense(5, activation="linear")(truncated)

    model = tf.keras.Model(inputs=inputs, outputs=out)

    adam = tf.optimizers.Adam()
    model.compile(optimizer=adam, loss="mse")

    return model




## === cell 7
train_inputs, val_inputs, train_labels, val_labels = train_test_split(
    train_inputs, train_labels, test_size=0.1, random_state=SEED
)

print("train split:", train_inputs.shape, train_labels.shape)
print("val split:", val_inputs.shape, val_labels.shape)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2949584448.py in <cell line: 0>()
----> 1 train_inputs, val_inputs, train_labels, val_labels = train_test_split(
      2     train_inputs, train_labels, test_size=0.1, random_state=SEED
      3 )
      4 
      5 print("train split:", train_inputs.shape, train_labels.shape)

NameError: name 'train_test_split' is not defined

## === cell 8
if len(tf.config.list_physical_devices("GPU")) > 0:
    print("Training on GPU")
else:
    print("Training on CPU")

lr_callback = tf.keras.callbacks.ReduceLROnPlateau()



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2994654021.py in <cell line: 0>()
----> 1 if len(tf.config.list_physical_devices("GPU")) > 0:
      2     print("Training on GPU")
      3 else:
      4     print("Training on CPU")
      5 

NameError: name 'tf' is not defined

## === cell 9
gru = build_model(gru=True, seq_len=107, pred_len=68)
sv_gru = tf.keras.callbacks.ModelCheckpoint(
    "model_gru.weights.h5", save_weights_only=True, save_best_only=False
)

history_gru = gru.fit(
    train_inputs,
    train_labels,
    validation_data=(val_inputs, val_labels),
    batch_size=64,
    epochs=72,
    callbacks=[lr_callback, sv_gru],
    verbose=2,
)

print(
    f"Min training loss={min(history_gru.history['loss'])}, min validation loss={min(history_gru.history['val_loss'])}"
)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/433153080.py in <cell line: 0>()
----> 1 gru = build_model(gru=True, seq_len=107, pred_len=68)
      2 sv_gru = tf.keras.callbacks.ModelCheckpoint(
      3     "model_gru.weights.h5", save_weights_only=True, save_best_only=False
      4 )
      5 

/tmp/ipykernel_11/4025110096.py in build_model(gru, seq_len, pred_len, dropout, embed_dim, hidden_dim)
     25 ):
     26 
---> 27     inputs = tf.keras.layers.Input(shape=(seq_len, 3), dtype=tf.int32)
     28 
     29     embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=embed_dim)(

NameError: name 'tf' is not defined

## === cell 10
lstm = build_model(gru=False, seq_len=107, pred_len=68)
sv_lstm = tf.keras.callbacks.ModelCheckpoint(
    "model_lstm.weights.h5", save_weights_only=True, save_best_only=False
)

history_lstm = lstm.fit(
    train_inputs,
    train_labels,
    validation_data=(val_inputs, val_labels),
    batch_size=64,
    epochs=72,
    callbacks=[lr_callback, sv_lstm],
    verbose=2,
)

print(
    f"Min training loss={min(history_lstm.history['loss'])}, min validation loss={min(history_lstm.history['val_loss'])}"
)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1953731830.py in <cell line: 0>()
----> 1 lstm = build_model(gru=False, seq_len=107, pred_len=68)
      2 sv_lstm = tf.keras.callbacks.ModelCheckpoint(
      3     "model_lstm.weights.h5", save_weights_only=True, save_best_only=False
      4 )
      5 

/tmp/ipykernel_11/4025110096.py in build_model(gru, seq_len, pred_len, dropout, embed_dim, hidden_dim)
     25 ):
     26 
---> 27     inputs = tf.keras.layers.Input(shape=(seq_len, 3), dtype=tf.int32)
     28 
     29     embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=embed_dim)(

NameError: name 'tf' is not defined

## === cell 11
fig, ax = plt.subplots(1, 2, figsize=(20, 6))

ax[0].plot(history_gru.history["loss"])
ax[0].plot(history_gru.history["val_loss"])
ax[0].set_title("GRU")
ax[0].legend(["train", "validation"], loc="upper right")
ax[0].set_ylabel("Loss")
ax[0].set_xlabel("Epoch")

ax[1].plot(history_lstm.history["loss"])
ax[1].plot(history_lstm.history["val_loss"])
ax[1].set_title("LSTM")
ax[1].legend(["train", "validation"], loc="upper right")
ax[1].set_ylabel("Loss")
ax[1].set_xlabel("Epoch")

plt.show()



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4045418568.py in <cell line: 0>()
----> 1 fig, ax = plt.subplots(1, 2, figsize=(20, 6))
      2 
      3 ax[0].plot(history_gru.history["loss"])
      4 ax[0].plot(history_gru.history["val_loss"])
      5 ax[0].set_title("GRU")

NameError: name 'plt' is not defined

## === cell 12
test_df = test.copy()
assert (
    test_df["seq_length"].nunique() == 1 and int(test_df["seq_length"].iloc[0]) == 107
), "Unexpected test seq_length; update inference handling."

test_inputs = preprocess_inputs(test_df)
print("test_inputs:", test_inputs.shape)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3646382736.py in <cell line: 0>()
----> 1 test_df = test.copy()
      2 assert (
      3     test_df["seq_length"].nunique() == 1 and int(test_df["seq_length"].iloc[0]) == 107
      4 ), "Unexpected test seq_length; update inference handling."
      5 

NameError: name 'test' is not defined

## === cell 13
gru.load_weights("model_gru.weights.h5")
lstm.load_weights("model_lstm.weights.h5")

gru_preds_68 = gru.predict(test_inputs, batch_size=64, verbose=1)  # (n_test, 68, 5)
lstm_preds_68 = lstm.predict(test_inputs, batch_size=64, verbose=1)  # (n_test, 68, 5)

print("gru_preds_68:", gru_preds_68.shape, "lstm_preds_68:", lstm_preds_68.shape)




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1505594205.py in <cell line: 0>()
----> 1 gru.load_weights("model_gru.weights.h5")
      2 lstm.load_weights("model_lstm.weights.h5")
      3 
      4 gru_preds_68 = gru.predict(test_inputs, batch_size=64, verbose=1)  # (n_test, 68, 5)
      5 lstm_preds_68 = lstm.predict(test_inputs, batch_size=64, verbose=1)  # (n_test, 68, 5)

NameError: name 'gru' is not defined

## === cell 14
def pad_to_107(preds_68, seq_len=107):
    n, L, c = preds_68.shape
    assert L == 68 and c == 5
    out = np.zeros((n, seq_len, c), dtype=np.float32)
    out[:, :68, :] = preds_68.astype(np.float32)
    return out


gru_preds = pad_to_107(gru_preds_68, seq_len=107)
lstm_preds = pad_to_107(lstm_preds_68, seq_len=107)

print("padded gru_preds:", gru_preds.shape, "padded lstm_preds:", lstm_preds.shape)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1092801449.py in <cell line: 0>()
      7 
      8 
----> 9 gru_preds = pad_to_107(gru_preds_68, seq_len=107)
     10 lstm_preds = pad_to_107(lstm_preds_68, seq_len=107)
     11 

NameError: name 'gru_preds_68' is not defined

## === cell 15
preds_gru = []
for i, uid in enumerate(test_df.id.values):
    single_pred = gru_preds[i]  # (107, 5)
    single_df = pd.DataFrame(single_pred, columns=target_cols)
    single_df["id_seqpos"] = [f"{uid}_{x}" for x in range(single_df.shape[0])]
    preds_gru.append(single_df)

preds_gru_df = pd.concat(preds_gru, axis=0, ignore_index=True)
print(preds_gru_df.head(), preds_gru_df.shape)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1764526261.py in <cell line: 0>()
      1 preds_gru = []
----> 2 for i, uid in enumerate(test_df.id.values):
      3     single_pred = gru_preds[i]  # (107, 5)
      4     single_df = pd.DataFrame(single_pred, columns=target_cols)
      5     single_df["id_seqpos"] = [f"{uid}_{x}" for x in range(single_df.shape[0])]

NameError: name 'test_df' is not defined

## === cell 16
preds_lstm = []
for i, uid in enumerate(test_df.id.values):
    single_pred = lstm_preds[i]  # (107, 5)
    single_df = pd.DataFrame(single_pred, columns=target_cols)
    single_df["id_seqpos"] = [f"{uid}_{x}" for x in range(single_df.shape[0])]
    preds_lstm.append(single_df)

preds_lstm_df = pd.concat(preds_lstm, axis=0, ignore_index=True)
print(preds_lstm_df.head(), preds_lstm_df.shape)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1924283768.py in <cell line: 0>()
      1 preds_lstm = []
----> 2 for i, uid in enumerate(test_df.id.values):
      3     single_pred = lstm_preds[i]  # (107, 5)
      4     single_df = pd.DataFrame(single_pred, columns=target_cols)
      5     single_df["id_seqpos"] = [f"{uid}_{x}" for x in range(single_df.shape[0])]

NameError: name 'test_df' is not defined

## === cell 17
blend_preds_df = pd.DataFrame()
blend_preds_df["id_seqpos"] = preds_gru_df["id_seqpos"]
blend_preds_df["reactivity"] = (
    0.4 * preds_gru_df["reactivity"] + 0.6 * preds_lstm_df["reactivity"]
)
blend_preds_df["deg_Mg_pH10"] = (
    0.4 * preds_gru_df["deg_Mg_pH10"] + 0.6 * preds_lstm_df["deg_Mg_pH10"]
)
blend_preds_df["deg_pH10"] = (
    0.4 * preds_gru_df["deg_pH10"] + 0.6 * preds_lstm_df["deg_pH10"]
)
blend_preds_df["deg_Mg_50C"] = (
    0.4 * preds_gru_df["deg_Mg_50C"] + 0.6 * preds_lstm_df["deg_Mg_50C"]
)
blend_preds_df["deg_50C"] = (
    0.4 * preds_gru_df["deg_50C"] + 0.6 * preds_lstm_df["deg_50C"]
)

print(blend_preds_df.head(), blend_preds_df.shape)



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/231359592.py in <cell line: 0>()
----> 1 blend_preds_df = pd.DataFrame()
      2 blend_preds_df["id_seqpos"] = preds_gru_df["id_seqpos"]
      3 blend_preds_df["reactivity"] = (
      4     0.4 * preds_gru_df["reactivity"] + 0.6 * preds_lstm_df["reactivity"]
      5 )

NameError: name 'pd' is not defined

## === cell 18
submission = sample_sub[["id_seqpos"]].merge(blend_preds_df, on="id_seqpos", how="left")

missing = int(submission[target_cols].isna().any(axis=1).sum())
print("Missing rows after merge:", missing)

submission[target_cols] = submission[target_cols].fillna(0.0)
submission = submission[["id_seqpos"] + target_cols]

assert (
    submission.shape[0] == sample_sub.shape[0]
), "Row count mismatch vs sample_submission"
assert (
    submission.columns.tolist() == ["id_seqpos"] + target_cols
), "Submission columns mismatch"
assert submission["id_seqpos"].is_unique, "id_seqpos must be unique"
assert set(submission["id_seqpos"]) == set(
    sample_sub["id_seqpos"]
), "id_seqpos set mismatch vs sample_submission"

print(submission.head(), submission.shape)



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3651547001.py in <cell line: 0>()
----> 1 submission = sample_sub[["id_seqpos"]].merge(blend_preds_df, on="id_seqpos", how="left")
      2 
      3 missing = int(submission[target_cols].isna().any(axis=1).sum())
      4 print("Missing rows after merge:", missing)
      5 

NameError: name 'sample_sub' is not defined

## === cell 19
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
print("Saved columns:", submission.columns.tolist())
print("Saved rows:", len(submission))

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/865138129.py in <cell line: 0>()
----> 1 submission.to_csv("submission.csv", index=False)
      2 print("Submission saved to submission.csv")
      3 print("Saved columns:", submission.columns.tolist())
      4 print("Saved rows:", len(submission))

NameError: name 'submission' is not defined
