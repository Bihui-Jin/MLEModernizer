# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.3878

# 6. Current score

0.26133

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.26094) has done: 'I remove the `tensorflow_addons` dependency (it’s failing to import in this environment due to a protobuf incompatibility) while keeping the same model/training logic by using Adam as before. I fix the Functional-model reshape bug by replacing raw `tf.reshape` calls on KerasTensors with an equivalent Keras `Reshape` layer (preserving shapes and semantics). I make preprocessing robust for any sequence length (so it won’t crash if the test set only contains length 107) and simplify test handling to match the provided data. Finally, I ensure predictions are expanded to the full `seq_length` (107) required by submission and write a valid `submission.csv` with correct columns and row order matching `sample_submission.csv`.'
- What this solution (achieved 0.26254) has done: 'The crash happens before any data loads because importing TensorFlow triggers a protobuf incompatibility (`MessageFactory.GetPrototype`), so the notebook can’t run end-to-end or produce a submission. The minimal fix is to force TensorFlow to use the pure-Python protobuf implementation (and disable C++ fast path) **before** importing TensorFlow, which avoids the missing method. I’m keeping all model/training/inference logic identical, only adjusting import order and environment variables to restore runtime correctness; this should allow you to reproduce (or improve toward) the previously achieved score by actually completing training and writing `submission.csv`. I also add a couple of safety checks to ensure the submission columns/order match `sample_submission.csv` exactly (score-neutral).'
- What this solution (achieved 0.26176) has done: 'We fix the TensorFlow import crash caused by a protobuf API mismatch by ensuring the pure-Python protobuf runtime is forced *before* any TensorFlow-related import happens, and we do it in a way that’s robust in Kaggle by also clearing any previously-imported `google.protobuf` modules. Then we fix a Keras functional-graph bug where slicing a `KerasTensor` (`hidden[:, :pred_len]`) can error under TF 2.18 by replacing it with an equivalent `Lambda` layer (same semantics). Finally, we keep the training/inference logic and blend unchanged, and add small safety checks to guarantee the submission rows/ordering exactly match `sample_submission.csv` and that a valid `submission.csv` is always written.'
- What this solution (achieved 0.26331) has done: 'We fix the runtime crash happening at TensorFlow import by forcing protobuf’s pure-Python implementation early and also proactively importing `google.protobuf.message_factory` before importing TensorFlow (this avoids the missing `GetPrototype` call in this environment). We keep the model/training/inference logic identical to preserve scoring behavior, only making these import/order changes plus a small safety fallback for data paths. Finally, we ensure the submission file is always written as `submission.csv` with the exact required columns and row order matching `sample_submission.csv` (score-neutral).'
- What this solution (achieved 0.26252) has done: 'The notebook is currently failing immediately at TensorFlow import due to a protobuf runtime mismatch (`MessageFactory.GetPrototype` missing). I force the pure-Python protobuf implementation even more robustly by setting env vars before *any* protobuf import, and by preloading the right protobuf symbol (`GetMessageClass`) that TensorFlow expects in this environment. I keep the model/training/inference logic unchanged to preserve the solution’s behavior and score trend, only touching the import/bootstrap section so everything runs end-to-end. Finally, I keep the submission-writing logic the same but ensure it always writes a valid `submission.csv` in the working directory.'
- What this solution (achieved 0.26133) has done: 'We fix the immediate runtime crash at TensorFlow import by pinning protobuf to the pure-Python implementation *and* providing a small compatibility shim that adds back `MessageFactory.GetPrototype` (which TF 2.18 expects in some environments). This is a minimal, score-neutral change that unblocks the entire pipeline without changing your model/training/inference logic. I also keep the existing safety checks for paths and submission formatting so a valid `submission.csv` is always written. No training hyperparameters, architecture, or blending weights are changed, so the score behavior should remain consistent (and in this case you’re already better than the target, so we avoid score-changing edits).'

# 9. Code solution

## === cell 0
import os, sys, warnings

warnings.filterwarnings("ignore")

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

for m in list(sys.modules.keys()):
    if m.startswith("google.protobuf"):
        del sys.modules[m]

import google.protobuf  # noqa: F401
from google.protobuf import message_factory as _message_factory  # noqa: F401

if not hasattr(_message_factory.MessageFactory, "GetPrototype"):

    def _GetPrototype(self, descriptor):
        gm = getattr(_message_factory, "GetMessageClass", None)
        if gm is not None:
            return gm(descriptor)
        return self.GetMessageClass(descriptor)

    _message_factory.MessageFactory.GetPrototype = _GetPrototype

_ = getattr(_message_factory, "GetMessageClass", None)

import gc, random, math, json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split, KFold

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("set up complete!", "TF:", tf.__version__)



## === cell 1
TRAIN_PATH = "/kaggle/input/stanford-covid-vaccine/train.json"
TEST_PATH = "/kaggle/input/stanford-covid-vaccine/test.json"
SAMPLE_PATH = "/kaggle/input/stanford-covid-vaccine/sample_submission.csv"

if not os.path.exists(TRAIN_PATH):
    TRAIN_PATH = "/kaggle/input/train.json"
if not os.path.exists(TEST_PATH):
    TEST_PATH = "/kaggle/input/test.json"
if not os.path.exists(SAMPLE_PATH):
    SAMPLE_PATH = "/kaggle/input/sample_submission.csv"

train = pd.read_json(TRAIN_PATH, lines=True)
test = pd.read_json(TEST_PATH, lines=True)
sample_sub = pd.read_csv(SAMPLE_PATH)

print("Data Load Complete")



## === cell 2
print(train.shape)
if ~train.isnull().values.any():
    print("No missing values")
train.head()



## === cell 3
print(test.shape)
if ~test.isnull().values.any():
    print("No missing values")
test.head()



## === cell 4
print(sample_sub.shape)
if ~sample_sub.isnull().values.any():
    print("No missing values")
sample_sub.head()



## === cell 5
target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]



## === cell 6
token2int = {x: i for i, x in enumerate("().ACGUBEHIMSX")}
token2int["U"]  # keep as in original



## === cell 7
cols = ["sequence", "structure", "predicted_loop_type"]
train[cols].applymap(lambda seq: [token2int[x] for x in seq]).head()




## === cell 8
def preprocess_inputs(df, cols=["sequence", "structure", "predicted_loop_type"]):
    if df is None or len(df) == 0:
        return np.zeros((0, 0, len(cols)), dtype=np.int32)

    arr = df[cols].applymap(lambda seq: [token2int[x] for x in seq]).values.tolist()
    x = np.array(arr, dtype=np.int32)  # (n, 3, seq_len)
    if x.ndim != 3:
        raise ValueError(
            f"Unexpected preprocessed input ndim={x.ndim}, shape={x.shape}"
        )
    return np.transpose(x, (0, 2, 1))  # (n, seq_len, 3)




## === cell 9
train_filt = train[train.signal_to_noise > 1].copy()
train_inputs = preprocess_inputs(train_filt)
train_y = np.array(train_filt[target_cols].values.tolist(), dtype=np.float32).transpose(
    (0, 2, 1)
)

print(train_inputs.shape)
print(train_y.shape)



## === cell 10
print("Ready to build models.")




## === cell 11
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
    gru=False, seq_len=107, pred_len=68, dropout=0.5, embed_dim=100, hidden_dim=128
):
    inputs = tf.keras.layers.Input(shape=(seq_len, 3))

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

    truncated = tf.keras.layers.Lambda(lambda x: x[:, :pred_len, :], name="truncate")(
        hidden
    )
    out = tf.keras.layers.Dense(5, activation="linear")(truncated)

    model = tf.keras.Model(inputs=inputs, outputs=out)

    adam = tf.optimizers.Adam()
    model.compile(optimizer=adam, loss="mse")
    return model


print("Model structure defined")




## === cell 12
def lstm_model(seq_len=107, output_dim=100, dropout=0.5, pred_len=68):
    inputs = tf.keras.layers.Input(shape=(seq_len, 3))
    embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=output_dim)(
        inputs
    )
    reshaped = tf.keras.layers.Reshape((seq_len, 3 * output_dim))(embed)

    hidden = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            128, dropout=dropout, kernel_initializer="orthogonal", return_sequences=True
        )
    )(reshaped)
    hidden = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            128, dropout=dropout, kernel_initializer="orthogonal", return_sequences=True
        )
    )(hidden)
    hidden = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            128, dropout=dropout, kernel_initializer="orthogonal", return_sequences=True
        )
    )(hidden)

    truncated = tf.keras.layers.Lambda(lambda x: x[:, :pred_len, :], name="truncate")(
        hidden
    )
    output = tf.keras.layers.Dense(5, activation="linear")(truncated)

    model = tf.keras.Model(inputs=inputs, outputs=output)

    adam = tf.optimizers.Adam(learning_rate=0.01, decay=0.0001)
    model.compile(loss="mse", optimizer=adam)
    return model




## === cell 13
train_data, val_data, train_labels, val_labels = train_test_split(
    train_inputs, train_y, test_size=0.2, random_state=4
)

print(train_data.shape, val_data.shape, train_labels.shape, val_labels.shape)



## === cell 14
lr_callback = tf.keras.callbacks.ReduceLROnPlateau()
smpl_lstm = lstm_model(seq_len=107, output_dim=100, dropout=0.5, pred_len=68)
sv_smpl_lstm = tf.keras.callbacks.ModelCheckpoint(
    "model_smpl_lstm.weights.h5", save_weights_only=True
)

smpl_lstm.summary()



## === cell 15
history_smpl_lstm = smpl_lstm.fit(
    train_data,
    train_labels,
    validation_data=(val_data, val_labels),
    batch_size=64,
    epochs=80,
    callbacks=[lr_callback, sv_smpl_lstm],
    verbose=2,
)

print(
    f"Min training loss={min(history_smpl_lstm.history['loss'])}, min validation loss={min(history_smpl_lstm.history['val_loss'])}"
)



## === cell 16
gru = build_model(gru=True, seq_len=107, pred_len=68)
sv_gru = tf.keras.callbacks.ModelCheckpoint(
    "model_gru.weights.h5", save_weights_only=True
)

gru.summary()



## === cell 17
history_gru = gru.fit(
    train_data,
    train_labels,
    validation_data=(val_data, val_labels),
    batch_size=64,
    epochs=70,
    callbacks=[lr_callback, sv_gru],
    verbose=2,
)

print(
    f"Min training loss={min(history_gru.history['loss'])}, min validation loss={min(history_gru.history['val_loss'])}"
)



## === cell 18
fig, ax = plt.subplots(1, 2, figsize=(20, 10))

ax[0].plot(history_gru.history["loss"])
ax[0].plot(history_gru.history["val_loss"])

ax[1].plot(history_smpl_lstm.history["loss"])
ax[1].plot(history_smpl_lstm.history["val_loss"])

ax[0].set_title("GRU")
ax[1].set_title("SMPL_LSTM")

ax[0].legend(["train", "validation"], loc="upper right")
ax[1].legend(["train", "validation"], loc="upper right")

ax[0].set_ylabel("Loss")
ax[0].set_xlabel("Epoch")
ax[1].set_ylabel("Loss")
ax[1].set_xlabel("Epoch")
plt.show()



## === cell 19
test_df = test.copy()
test_inputs = preprocess_inputs(test_df)

print("Test data prepared!", test_df.shape, test_inputs.shape)



## === cell 20
gru_pred = build_model(gru=True, seq_len=107, pred_len=68)
lstm_pred = lstm_model(seq_len=107, output_dim=100, dropout=0.5, pred_len=68)

gru_pred.load_weights("model_gru.weights.h5")
lstm_pred.load_weights("model_smpl_lstm.weights.h5")

print("Models reloaded for inference")



## === cell 21
gru_preds_68 = gru_pred.predict(test_inputs, batch_size=64, verbose=1)  # (n, 68, 5)
lstm_preds_68 = lstm_pred.predict(test_inputs, batch_size=64, verbose=1)  # (n, 68, 5)

print(gru_preds_68.shape, lstm_preds_68.shape)



## === cell 22
SEQ_LEN = int(test_df.seq_length.iloc[0])  # expected 107
PRED_LEN = gru_preds_68.shape[1]  # 68
assert SEQ_LEN == 107, f"Unexpected seq_length={SEQ_LEN} for this dataset snapshot"


def pad_to_seq_len(preds_68, seq_len=107):
    n, pred_len, c = preds_68.shape
    if pred_len == seq_len:
        return preds_68
    if pred_len > seq_len:
        return preds_68[:, :seq_len, :]
    pad_len = seq_len - pred_len
    last = preds_68[:, -1:, :]
    pad = np.repeat(last, pad_len, axis=1)
    return np.concatenate([preds_68, pad], axis=1)


gru_preds = pad_to_seq_len(gru_preds_68, SEQ_LEN)
lstm_preds = pad_to_seq_len(lstm_preds_68, SEQ_LEN)

print(gru_preds.shape, lstm_preds.shape)



## === cell 23
preds_gru = []
for i, uid in enumerate(test_df.id.values):
    single_pred = gru_preds[i]  # (107, 5)
    single_df = pd.DataFrame(single_pred, columns=target_cols)
    single_df["id_seqpos"] = [f"{uid}_{x}" for x in range(single_df.shape[0])]
    preds_gru.append(single_df)

preds_gru_df = pd.concat(preds_gru, ignore_index=True)
preds_gru_df.head()



## === cell 24
preds_lstm = []
for i, uid in enumerate(test_df.id.values):
    single_pred = lstm_preds[i]  # (107, 5)
    single_df = pd.DataFrame(single_pred, columns=target_cols)
    single_df["id_seqpos"] = [f"{uid}_{x}" for x in range(single_df.shape[0])]
    preds_lstm.append(single_df)

preds_lstm_df = pd.concat(preds_lstm, ignore_index=True)
preds_lstm_df.head()



## === cell 25
lstm_weight = 0.7
gru_weight = 0.3

blend_preds_df = pd.DataFrame()
blend_preds_df["id_seqpos"] = preds_gru_df["id_seqpos"]
for c in target_cols:
    blend_preds_df[c] = (
        gru_weight * preds_gru_df[c].values + lstm_weight * preds_lstm_df[c].values
    )

blend_preds_df.head()



## === cell 26
submission = sample_sub[["id_seqpos"]].merge(blend_preds_df, on="id_seqpos", how="left")

for c in target_cols:
    submission[c] = submission[c].fillna(0.0).astype(np.float32)

submission = submission[["id_seqpos"] + target_cols]

assert submission.shape[0] == sample_sub.shape[0], (submission.shape, sample_sub.shape)
assert list(submission.columns) == list(sample_sub.columns), (
    submission.columns,
    sample_sub.columns,
)

print(submission.head())
print("Submission rows:", len(submission), "Expected:", len(sample_sub))



## === cell 27
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
