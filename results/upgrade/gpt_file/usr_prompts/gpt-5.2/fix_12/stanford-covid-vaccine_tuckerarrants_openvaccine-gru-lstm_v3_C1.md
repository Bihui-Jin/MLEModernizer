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

0.40244

# 6. Current score

0.28619

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.27739) has done: 'The immediate blocker is a TensorFlow import crash caused by an incompatible protobuf version in this environment; the safest minimal fix is to force Python protobuf implementation before importing TensorFlow. Next, Keras 3 removed `by_name`/`skip_mismatch` from `load_weights`, so the inference cell must load weights without those arguments and keep the same model shape (pred_len=68) to match trained weights. Finally, the submission-building logic should generate predictions only for the required 68 scored positions and then merge onto `sample_submission.csv` to guarantee exact row order and presence of all `id_seqpos` rows. These fixes are execution/stability oriented and preserve your model/training logic, while ensuring a valid `submission.csv` is produced end-to-end.'
- What this solution (achieved 0.28502) has done: 'I fix the TensorFlow import crash by forcing protobuf to use the pure-Python implementation *before* any TensorFlow-related imports, and by importing TensorFlow in a safer order for this Kaggle environment. I also correct a minor logic bug in the missing-value checks (`~` vs `not`) without changing modeling behavior. Finally, I make submission generation robust by ensuring predictions align to `sample_submission.csv` row order via a merge (as you already intended), and I keep the model shapes identical so saved weights load cleanly and scores remain in the same performance band (your current score is already better than the target for a lower-is-better metric). The core model, training loop, loss, epochs, and blending remain unchanged.'
- What this solution (achieved 0.28471) has done: 'Your current issue is that no Kaggle score was yielded, so the smallest improvement is to guarantee the notebook completes reliably and writes a valid `submission.csv`. In this environment, downgrading `protobuf` via `pip` is fragile and often breaks TensorFlow 2.18; instead we keep the installed protobuf and force the pure-Python protobuf runtime *before* importing TensorFlow to avoid the common TF/protobuf crash. We also explicitly reset the ReduceLROnPlateau callback per model (so GRU and LSTM don’t share state), which keeps the training semantics the same but improves stability/reproducibility. Finally, we keep your merge-based submission construction (already correct) and add a small defensive check to ensure predictions cover exactly the scored length (68) as expected by the model/weights.'
- What this solution (achieved 0.28619) has done: 'The immediate blocker is the TensorFlow import crash caused by a protobuf API mismatch; the minimal, environment-safe fix is to force the pure-Python protobuf implementation and disable the C++ one *before* any protobuf/TensorFlow imports, and also pin protobuf to use the Python codegen. Next, to keep runtime stable across TF/Keras versions, we set deterministic seeds and reduce TF logging without changing the model/training core logic. Finally, we keep your existing prediction/blending and merge-based submission construction, but add a small defensive check to ensure the merged submission has no missing predictions and is written as a valid `submission.csv`. These changes are execution/stability oriented and should preserve your score behavior (already within the target tolerance band for a lower-is-better metric).'
- What this solution (achieved 0.28619) has done: 'We fix the current runtime blocker happening before training by addressing the protobuf/TensorFlow incompatibility that triggers `MessageFactory.GetPrototype` errors in this Kaggle environment. The minimal, environment-safe approach is to force the pure-Python protobuf runtime and additionally ensure TensorFlow sees a compatible protobuf API by importing protobuf early and patching the missing method when needed (this is score-neutral but unblocks execution). All model/training/inference logic, shapes, epochs, and blending remain unchanged to keep your score behavior in the same band. Finally, we keep the existing merge-based submission construction and add a small sanity check that predictions cover the expected 68 positions so a valid `submission.csv` is always produced.'
- What this solution (achieved 0.28619) has done: 'The crash happens before TensorFlow fully imports because the protobuf compatibility patch is checking for `GetPrototype` on the class, but the failure is raised on an instance during import; the minimal safe fix is to patch both the `MessageFactory` class and any instantiated factory objects with a compatible `GetPrototype` implementation before importing TensorFlow. I also ensure this patch runs unconditionally (but defensively) and only touches the import/runtime compatibility layer, keeping your model/training/inference logic unchanged so score behavior stays in the same band (already within ±10% of the target for a lower-is-better metric). The rest of the pipeline remain the same, including pred_len=68, blending, and the merge onto `sample_submission.csv`. Finally, the script still write a valid `submission.csv` in the working directory.'
- What this solution (achieved 0.28619) has done: 'I fix the runtime crash in the first cell by making the protobuf compatibility patch apply to both `MessageFactory` and its base `google.protobuf.message_factory.MessageFactory`, and by safely adding a `GetPrototype` shim when only `GetMessageClass` exists; this unblocks TensorFlow import in this environment without touching modeling logic. I also make the patch robust against protobuf internals differences by attempting to patch any default/registered factory instances if present. Since your current score (0.28619, lower-is-better) is already better than the target (0.40244) and within the ±10% target band requirement, I won’t make any score-changing changes to training, architecture, or post-processing beyond these stability fixes. The rest of the pipeline remains identical and still write a valid `submission.csv` with the required columns and row order.'
- What this solution (achieved 0.28619) has done: 'The only failing part is the TensorFlow import due to protobuf’s `MessageFactory` lacking `GetPrototype` in this environment; I fix this by patching the *instance* methods (including `_DEFAULT_FACTORY`) before importing TensorFlow, rather than patching only the class. This is an execution-only compatibility fix and does not touch your model, training loop, epochs, loss, blending, or submission logic, so it should be score-neutral (and your current score is already well within the ±10% target band for a lower-is-better metric). I also keep the submission merge/order checks intact so the notebook always writes a valid `submission.csv` with the exact required columns and row count.'
- What this solution (achieved 0.28619) has done: 'I fix the TensorFlow import crash by applying a more robust protobuf compatibility shim that patches both the `MessageFactory` class and the default/active factory instance(s), including `google.protobuf.internal.python_message.MessageFactory`, which is where this environment’s missing `GetPrototype` often originates. This change is execution-only and happens before importing TensorFlow, so it preserves your model/training/inference logic and should be score-neutral (your current score is already within the ±10% band around the target for a lower-is-better metric). I also keep everything else the same, only adding a tiny defensive import ordering so the patch is guaranteed to run first. The pipeline then train, predict, and write a valid `submission.csv` with correct columns and row order.'

# 9. Code solution

## === cell 0
import os, sys, warnings

warnings.filterwarnings("ignore")

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def _patch_protobuf_getprototype():
    """
    Some Kaggle images ship protobuf where MessageFactory lacks GetPrototype,
    but TensorFlow (via older generated protos) still calls it.
    We patch both classes and default factory instances BEFORE importing TF.
    """
    try:
        import types
        from google.protobuf import message_factory as mf

        def _ensure(obj):
            if obj is None:
                return
            if (not hasattr(obj, "GetPrototype")) and hasattr(obj, "GetMessageClass"):

                def _GetPrototype(self, descriptor):
                    return self.GetMessageClass(descriptor)

                try:
                    setattr(obj, "GetPrototype", _GetPrototype)
                except Exception:
                    try:
                        obj.GetPrototype = types.MethodType(_GetPrototype, obj)
                    except Exception:
                        pass

        _ensure(getattr(mf, "MessageFactory", None))
        try:
            from google.protobuf.message_factory import MessageFactory as BaseMF

            _ensure(BaseMF)
        except Exception:
            pass

        _ensure(getattr(mf, "_DEFAULT_FACTORY", None))
        _ensure(getattr(mf, "default_factory", None))

        try:
            from google.protobuf.internal import python_message as pm

            _ensure(getattr(pm, "MessageFactory", None))
            _ensure(getattr(pm, "_DEFAULT_FACTORY", None))
        except Exception:
            pass

        try:
            from google.protobuf import symbol_database as _symbol_database

            _ensure(getattr(_symbol_database, "Default", None))
            _ensure(getattr(_symbol_database, "_DEFAULT", None))
        except Exception:
            pass

    except Exception:
        return


_patch_protobuf_getprototype()

import gc, random, math, json  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from tqdm import tqdm  # noqa: E402

import tensorflow as tf  # noqa: E402
import tensorflow.keras.backend as K  # noqa: E402
import tensorflow.keras.layers as L  # noqa: E402

from sklearn.model_selection import train_test_split, KFold  # noqa: E402

SEED = 34
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TF version:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train = pd.read_json("/kaggle/input/stanford-covid-vaccine/train.json", lines=True)
test = pd.read_json("/kaggle/input/stanford-covid-vaccine/test.json", lines=True)
sample_sub = pd.read_csv("/kaggle/input/stanford-covid-vaccine/sample_submission.csv")



## === cell 2
print(train.shape)
if not train.isnull().values.any():
    print("No missing values")
train.head()



## === cell 3
print(test.shape)
if not test.isnull().values.any():
    print("No missing values")
test.head()



## === cell 4
print(sample_sub.shape)
if not sample_sub.isnull().values.any():
    print("No missing values")
sample_sub.head()



## === cell 5
target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]



## === cell 6
token2int = {x: i for i, x in enumerate("().ACGUBEHIMSX")}
PAD_TOKEN = "."  # safe pad token that exists in token2int
PAD_ID = token2int[PAD_TOKEN]




## === cell 7
def preprocess_inputs(
    df, cols=["sequence", "structure", "predicted_loop_type"], seq_len=107
):
    out = np.zeros((len(df), seq_len, len(cols)), dtype=np.int32)
    for i, (_, row) in enumerate(df[cols].iterrows()):
        for j, c in enumerate(cols):
            s = row[c]
            if len(s) < seq_len:
                s = s + (PAD_TOKEN * (seq_len - len(s)))
            else:
                s = s[:seq_len]
            out[i, :, j] = [token2int[ch] for ch in s]
    return out




## === cell 8
train_inputs = preprocess_inputs(train, seq_len=107)
train_labels = (
    np.array(train[target_cols].values.tolist()).transpose((0, 2, 1)).astype(np.float32)
)

print("train_inputs:", train_inputs.shape, train_inputs.dtype)
print("train_labels:", train_labels.shape, train_labels.dtype)




## === cell 9
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
    gru=False, seq_len=107, pred_len=68, dropout=0.5, embed_dim=75, hidden_dim=128
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




## === cell 10
train_inputs_tr, val_inputs, train_labels_tr, val_labels = train_test_split(
    train_inputs, train_labels, test_size=0.1, random_state=SEED
)

print(train_inputs_tr.shape, val_inputs.shape, train_labels_tr.shape, val_labels.shape)



## === cell 11
if tf.config.list_physical_devices("GPU"):
    print("Training on GPU")
else:
    print("Training on CPU")




## === cell 12
def make_lr_callback():
    return tf.keras.callbacks.ReduceLROnPlateau()




## === cell 13
gru = build_model(gru=True, seq_len=107, pred_len=68)
sv_gru = tf.keras.callbacks.ModelCheckpoint(
    "model_gru.weights.h5",
    save_weights_only=True,
    save_best_only=True,
    monitor="val_loss",
)

history_gru = gru.fit(
    train_inputs_tr,
    train_labels_tr,
    validation_data=(val_inputs, val_labels),
    batch_size=64,
    epochs=70,
    callbacks=[make_lr_callback(), sv_gru],
    verbose=2,
)

print(
    f"Min training loss={min(history_gru.history['loss'])}, min validation loss={min(history_gru.history['val_loss'])}"
)



## === cell 14
lstm = build_model(gru=False, seq_len=107, pred_len=68)
sv_lstm = tf.keras.callbacks.ModelCheckpoint(
    "model_lstm.weights.h5",
    save_weights_only=True,
    save_best_only=True,
    monitor="val_loss",
)

history_lstm = lstm.fit(
    train_inputs_tr,
    train_labels_tr,
    validation_data=(val_inputs, val_labels),
    batch_size=64,
    epochs=70,
    callbacks=[make_lr_callback(), sv_lstm],
    verbose=2,
)

print(
    f"Min training loss={min(history_lstm.history['loss'])}, min validation loss={min(history_lstm.history['val_loss'])}"
)



## === cell 15
fig, ax = plt.subplots(1, 2, figsize=(20, 10))

ax[0].plot(history_gru.history["loss"])
ax[0].plot(history_gru.history["val_loss"])

ax[1].plot(history_lstm.history["loss"])
ax[1].plot(history_lstm.history["val_loss"])

ax[0].set_title("GRU")
ax[1].set_title("LSTM")

ax[0].legend(["train", "validation"], loc="upper right")
ax[1].legend(["train", "validation"], loc="upper right")

ax[0].set_ylabel("Loss")
ax[0].set_xlabel("Epoch")
ax[1].set_ylabel("Loss")
ax[1].set_xlabel("Epoch")



## === cell 16
test_inputs = preprocess_inputs(test, seq_len=107)

gru_infer = build_model(gru=True, seq_len=107, pred_len=68)
lstm_infer = build_model(gru=False, seq_len=107, pred_len=68)

gru_infer.load_weights("model_gru.weights.h5")
lstm_infer.load_weights("model_lstm.weights.h5")

gru_test_preds = gru_infer.predict(
    test_inputs, batch_size=64, verbose=1
)  # (n_test, 68, 5)
lstm_test_preds = lstm_infer.predict(
    test_inputs, batch_size=64, verbose=1
)  # (n_test, 68, 5)

assert (
    gru_test_preds.shape[1] == 68 and lstm_test_preds.shape[1] == 68
), "pred_len mismatch."

print("gru_test_preds:", gru_test_preds.shape)
print("lstm_test_preds:", lstm_test_preds.shape)



## === cell 17
preds_gru = []
for i, uid in enumerate(test.id.values):
    single_pred = gru_test_preds[i]  # (68, 5)
    single_df = pd.DataFrame(single_pred, columns=target_cols)
    single_df["id_seqpos"] = [f"{uid}_{x}" for x in range(single_df.shape[0])]
    preds_gru.append(single_df)
preds_gru_df = pd.concat(preds_gru, ignore_index=True)

preds_lstm = []
for i, uid in enumerate(test.id.values):
    single_pred = lstm_test_preds[i]  # (68, 5)
    single_df = pd.DataFrame(single_pred, columns=target_cols)
    single_df["id_seqpos"] = [f"{uid}_{x}" for x in range(single_df.shape[0])]
    preds_lstm.append(single_df)
preds_lstm_df = pd.concat(preds_lstm, ignore_index=True)

print(preds_gru_df.shape, preds_lstm_df.shape)
preds_gru_df.head()



## === cell 18
blend_preds_df = pd.DataFrame()
blend_preds_df["id_seqpos"] = preds_gru_df["id_seqpos"].values

for c in target_cols:
    blend_preds_df[c] = 0.5 * preds_gru_df[c].values + 0.5 * preds_lstm_df[c].values

blend_preds_df.head()



## === cell 19
submission = sample_sub[["id_seqpos"]].merge(blend_preds_df, on="id_seqpos", how="left")

for c in target_cols:
    submission[c] = submission[c].astype(np.float32).fillna(0.0)

print(submission.shape)
submission.head()



## === cell 20
assert (
    submission.shape[0] == sample_sub.shape[0]
), "Row count mismatch vs sample_submission."
assert (
    list(submission.columns) == ["id_seqpos"] + target_cols
), "Column mismatch vs required format."
assert submission["id_seqpos"].isna().sum() == 0, "Missing id_seqpos."
missing_pred_rows = submission[target_cols].isna().any(axis=1).sum()
assert missing_pred_rows == 0, "Unexpected NaNs in predictions after fillna."

_scored_count = test["seq_scored"].iloc[0] if "seq_scored" in test.columns else 68
assert _scored_count == 68, "Unexpected seq_scored; model pred_len is fixed to 68."

submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
print("File size (bytes):", os.path.getsize("submission.csv"))
print(submission.head())
