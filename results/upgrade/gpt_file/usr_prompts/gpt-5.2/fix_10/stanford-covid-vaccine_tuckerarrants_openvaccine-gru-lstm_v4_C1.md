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

0.40469

# 6. Current score

0.28413

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.28749) has done: 'I remove the import causing the protobuf/tensorflow_addons crash and switch to a built-in optimizer to keep training identical in spirit while making it run in TF 2.18. I fix the missing `train_test_split` by ensuring sklearn is imported correctly and make the `tf.reshape` Keras-safe by replacing it with a Keras `Reshape` layer. I also fix inference so it works with the actual test lengths (only 107 here) and make preprocessing robust to empty inputs, then ensure predictions are expanded to the required 107 rows per id and merged correctly with `sample_submission.csv`. Finally, the script always write a valid `submission.csv` with the exact required columns.'
- What this solution (achieved 0.28483) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation before importing TensorFlow (this is the root cause of the `MessageFactory.GetPrototype` error in TF 2.18 with protobuf 6.x). I also correct a small but real logic bug in the missing-values checks (`~...` should be `not ...`) so the notebook doesn’t behave incorrectly. The rest of the pipeline (preprocessing, model architecture, training loops, inference, blending, and submission formatting) be kept identical to preserve score behavior; since your current score is already better than the target (lower is better), I avoid score-changing modifications beyond stability fixes. The script run end-to-end and always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.28357) has done: 'I fix the TensorFlow/protobuf crash that prevents the notebook from importing TensorFlow by pinning protobuf to the compatible 3.x runtime via environment variables (the known TF 2.18 + protobuf 6.x issue) and by importing TensorFlow only after setting those variables. I also make the import section robust in Kaggle by adding a small fallback that forces the pure-Python protobuf and disables the C++ implementation to avoid the `MessageFactory.GetPrototype` AttributeError. These changes are execution/stability fixes and should keep the model/training/prediction logic identical, so your score should remain close to the current (already better than target). The rest of the pipeline is kept unchanged to preserve evaluation semantics and ensure a valid `submission.csv` is always written.'
- What this solution (achieved 0.28589) has done: 'I fix the TensorFlow import crash caused by the protobuf 6.x runtime by ensuring the pure-Python protobuf implementation is used before TensorFlow loads, and by importing TensorFlow through a guarded helper that retries after forcing the environment variables. I also make the `build_model` slicing Keras-safe by replacing `hidden[:, :pred_len]` with a `Lambda` layer, which avoids graph/eager incompatibilities in TF 2.18 without changing the model’s semantics. Finally, I keep the training/inference/submission logic unchanged (to avoid score drift since your current score is already better than the target) while ensuring a valid `submission.csv` is always written with the correct columns and row alignment.'
- What this solution (achieved 0.28575) has done: 'I fix the TensorFlow import crash by forcing the pure-Python protobuf implementation *before* any TensorFlow import and by importing TensorFlow only through a guarded helper (the current code still imports TF unconditionally, which triggers the crash). I keep the model/training/inference logic identical, only changing the import/initialization path so the notebook runs end-to-end in TF 2.18 + protobuf 6.x Kaggle images. I also add a tiny safety check to ensure we always load the best checkpointed weights (if present) before predicting, without changing the training procedure. Finally, the script still write a valid `submission.csv` with the exact required columns and row alignment.'
- What this solution (achieved 0.28324) has done: 'We fix the runtime crash in TF2.18+protobuf6 by forcing a compatible protobuf runtime mode before TensorFlow is imported, and by adding a robust import guard that falls back to the pure-Python protobuf implementation. This is an execution/stability fix and does not change the model architecture, training loop, loss, or inference semantics, so it should keep your score behavior essentially unchanged (you’re already better than the target since lower is better). We also make the long/short test split robust to the actual dataset (only 107 here) without changing predictions, and ensure `submission.csv` is always written with correct columns and row alignment.'
- What this solution (achieved 0.28736) has done: 'We fix the immediate runtime crash by importing TensorFlow in a way that’s compatible with protobuf 6.x in the Kaggle image (the current env-var approach alone isn’t sufficient because TF can still pull in the C++ protobuf bindings). To keep score behavior stable (you’re already better than the target and lower is better), we not change the model, training loop, loss, or prediction logic—only the import/compatibility path and a couple of small safety guards that don’t alter outputs when things are healthy. Finally, we ensure the pipeline always reaches the CSV write step and produces a valid `submission.csv` with the exact required columns and row alignment.'
- What this solution (achieved 0.28453) has done: 'I fix the TensorFlow import crash caused by the protobuf 6.x runtime by forcing TensorFlow to use the pure-Python protobuf implementation *before* importing TF, and by adding a guarded import that retries after clearing protobuf/TF modules if needed. This is an execution-only change that does not touch your model architecture, training loop, loss, or inference logic, so it should keep score behavior essentially unchanged (you’re already better than the target since lower is better). I also adjust the cell numbering to start at 1 (your current script starts at cell 0) so it matches the required “cells” format, without changing code execution. The notebook then run end-to-end and always write a valid `submission.csv` with the correct columns and row alignment.'
- What this solution (achieved 0.28413) has done: 'I fix the TensorFlow/protobuf import crash (`MessageFactory.GetPrototype` in TF 2.18 + protobuf 6.x) by forcing the pure-Python protobuf implementation *before anything protobuf-related is imported*, and by adding a guarded TensorFlow import that clears already-loaded protobuf/TF modules on retry. I keep the model, training loop, and inference logic identical to preserve score behavior (your current score is already better than the target for a lower-is-better metric, so we should not intentionally “improve” it). I also renumber cells starting at 1 to match the required format and add a small safety check to ensure tokenization doesn’t break if unexpected characters appear (score-neutral; prevents runtime failure). The pipeline run end-to-end and always write a valid `submission.csv` with the correct columns and row alignment.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import warnings

warnings.filterwarnings("ignore")

import sys
import gc, random, math, json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

SEED = 34
os.environ["PYTHONHASHSEED"] = str(SEED)


def import_tensorflow_safely():
    """
    Fix for TF2.18 + protobuf6 Kaggle images:
    - Force pure-Python protobuf via env vars (must be done before TF import).
    - If TF import still fails due to a partially-loaded protobuf/TF, clear modules and retry.
    Execution-only fix; does not touch model/training/inference logic.
    """
    import importlib

    def _clear_modules():
        for m in list(sys.modules.keys()):
            if m.startswith("tensorflow") or m.startswith("google.protobuf"):
                del sys.modules[m]
        gc.collect()

    try:
        tf = importlib.import_module("tensorflow")
        return tf
    except Exception as e1:
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
        _clear_modules()
        try:
            tf = importlib.import_module("tensorflow")
            return tf
        except Exception as e2:
            raise RuntimeError(
                "TensorFlow import failed even after forcing pure-python protobuf.\n"
                f"First error: {repr(e1)}\nSecond error: {repr(e2)}"
            )


tf = import_tensorflow_safely()

import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
from sklearn.model_selection import train_test_split, KFold

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

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




## === cell 7
def preprocess_inputs(df, cols=["sequence", "structure", "predicted_loop_type"]):
    if df is None or len(df) == 0:
        return np.zeros((0, 107, 3), dtype=np.int32)

    def _encode(seq):
        return [token2int.get(x, token2int["."]) for x in seq]

    arr = df[cols].applymap(_encode).values.tolist()
    arr = np.array(arr, dtype=np.int32)  # (n, 3, seq_len)
    return np.transpose(arr, (0, 2, 1))  # (n, seq_len, 3)




## === cell 8
train_inputs = preprocess_inputs(train)
train_labels = np.array(train[target_cols].values.tolist(), dtype=np.float32).transpose(
    (0, 2, 1)
)  # (n, 68, 5)

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

    truncated = tf.keras.layers.Lambda(lambda x: x[:, :pred_len], name="truncate")(
        hidden
    )
    out = tf.keras.layers.Dense(5, activation="linear")(truncated)

    model = tf.keras.Model(inputs=inputs, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
    return model




## === cell 10
train_inputs, val_inputs, train_labels, val_labels = train_test_split(
    train_inputs, train_labels, test_size=0.1, random_state=SEED
)

print(train_inputs.shape, val_inputs.shape, train_labels.shape, val_labels.shape)



## === cell 11
gpus = tf.config.list_physical_devices("GPU")
if gpus is not None and len(gpus) > 0:
    print("Training on GPU")
else:
    print("Training on CPU")



## === cell 12
lr_callback = tf.keras.callbacks.ReduceLROnPlateau()



## === cell 13
gru = build_model(gru=True)
sv_gru = tf.keras.callbacks.ModelCheckpoint(
    "model_gru.weights.h5",
    save_weights_only=True,
    save_best_only=True,
    monitor="val_loss",
)

history_gru = gru.fit(
    train_inputs,
    train_labels,
    validation_data=(val_inputs, val_labels),
    batch_size=64,
    epochs=75,
    callbacks=[lr_callback, sv_gru],
    verbose=2,
)

print(
    f"Min training loss={min(history_gru.history['loss'])}, min validation loss={min(history_gru.history['val_loss'])}"
)



## === cell 14
lstm = build_model(gru=False)
sv_lstm = tf.keras.callbacks.ModelCheckpoint(
    "model_lstm.weights.h5",
    save_weights_only=True,
    save_best_only=True,
    monitor="val_loss",
)

history_lstm = lstm.fit(
    train_inputs,
    train_labels,
    validation_data=(val_inputs, val_labels),
    batch_size=64,
    epochs=75,
    callbacks=[lr_callback, sv_lstm],
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
public_df = test.query("seq_length == 107").copy()
private_df = test.query("seq_length == 130").copy()

public_inputs = preprocess_inputs(public_df)
private_inputs = preprocess_inputs(private_df)

print("public_df:", public_df.shape, "public_inputs:", public_inputs.shape)
print("private_df:", private_df.shape, "private_inputs:", private_inputs.shape)




## === cell 17
def load_weights_if_exists(model, path):
    if os.path.exists(path):
        model.load_weights(path)
    else:
        print(f"WARNING: weights file not found: {path}")


gru_short = build_model(gru=True, seq_len=107, pred_len=107)
lstm_short = build_model(gru=False, seq_len=107, pred_len=107)

load_weights_if_exists(gru_short, "model_gru.weights.h5")
load_weights_if_exists(lstm_short, "model_lstm.weights.h5")

gru_public_preds = gru_short.predict(
    public_inputs, batch_size=64, verbose=0
)  # (n, 107, 5)
lstm_public_preds = lstm_short.predict(public_inputs, batch_size=64, verbose=0)

if len(private_df) > 0:
    gru_long = build_model(gru=True, seq_len=130, pred_len=130)
    lstm_long = build_model(gru=False, seq_len=130, pred_len=130)
    load_weights_if_exists(gru_long, "model_gru.weights.h5")
    load_weights_if_exists(lstm_long, "model_lstm.weights.h5")
    gru_private_preds = gru_long.predict(private_inputs, batch_size=64, verbose=0)
    lstm_private_preds = lstm_long.predict(private_inputs, batch_size=64, verbose=0)
else:
    gru_private_preds = np.zeros((0, 130, 5), dtype=np.float32)
    lstm_private_preds = np.zeros((0, 130, 5), dtype=np.float32)

print(
    "gru_public_preds:",
    gru_public_preds.shape,
    "lstm_public_preds:",
    lstm_public_preds.shape,
)
print(
    "gru_private_preds:",
    gru_private_preds.shape,
    "lstm_private_preds:",
    lstm_private_preds.shape,
)




## === cell 18
def preds_to_long_df(df, preds, target_cols):
    out = []
    for i, uid in enumerate(df.id.values):
        single_pred = preds[i]
        single_df = pd.DataFrame(single_pred, columns=target_cols)
        single_df["id_seqpos"] = [f"{uid}_{x}" for x in range(single_df.shape[0])]
        out.append(single_df)
    if len(out) == 0:
        return pd.DataFrame(columns=["id_seqpos"] + target_cols)
    return pd.concat(out, axis=0, ignore_index=True)


preds_gru_df = pd.concat(
    [
        preds_to_long_df(public_df, gru_public_preds, target_cols),
        preds_to_long_df(private_df, gru_private_preds, target_cols),
    ],
    axis=0,
    ignore_index=True,
)

preds_lstm_df = pd.concat(
    [
        preds_to_long_df(public_df, lstm_public_preds, target_cols),
        preds_to_long_df(private_df, lstm_private_preds, target_cols),
    ],
    axis=0,
    ignore_index=True,
)

print(preds_gru_df.shape, preds_lstm_df.shape)
preds_gru_df.head()



## === cell 19
blend_preds_df = pd.DataFrame()
blend_preds_df["id_seqpos"] = preds_gru_df["id_seqpos"].values

for c in target_cols:
    blend_preds_df[c] = 0.5 * preds_gru_df[c].values + 0.5 * preds_lstm_df[c].values

blend_preds_df.head()



## === cell 20
submission = sample_sub[["id_seqpos"]].merge(blend_preds_df, on="id_seqpos", how="left")
submission[target_cols] = submission[target_cols].fillna(0.0)
submission = submission[["id_seqpos"] + target_cols]

print(submission.shape)
submission.head()



## === cell 21
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
print("Saved columns:", submission.columns.tolist())
print("Any NaNs?", submission.isna().any().any())
print("submission.csv rows:", len(submission))
