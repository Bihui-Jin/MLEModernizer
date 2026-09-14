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

0.40727

# 6. Current score

0.26534

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.25845) has done: 'I fix the TensorFlow import crash caused by an incompatible protobuf version by forcing the pure-Python protobuf implementation before importing TensorFlow. Then I fix the Keras Functional graph error by replacing raw `tf.transpose`/tensor slicing with Keras-safe layers (`Permute` and `Lambda`) while keeping the same model topology and outputs. Finally, I ensure the data shapes match what the model expects (fixed-length 107 tokens per feature) and correct the submission creation to pad predictions to `seq_length` (107) while only using the first 68 positions the model predicts. These changes are required for the notebook to run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.26121) has done: 'I fix the TensorFlow import crash by ensuring the protobuf pure-Python environment variables are set before any TensorFlow-related import happens (the current cell ordering can still trigger the `MessageFactory.GetPrototype` error in some Kaggle runtimes). Then I make the training target (`y_train`) match the model output shape by padding the 68-length target arrays to length 107 (so the loss compares aligned tensors rather than relying on incompatible shapes). These changes are execution-critical and should also improve the score by correctly training on all positions the model outputs (instead of broadcasting/mismatching). Finally, I keep the submission-writing logic intact but add a small assertion to guarantee the output matches the sample submission format.'
- What this solution (achieved 0.26054) has done: 'I fix the TensorFlow/protobuf crash by moving the protobuf environment variables to the very top of the script and forcing the pure-Python protobuf implementation before any TensorFlow import can occur. This is execution-critical and should restore end-to-end training/inference. I also add a safe fallback to set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` again immediately before importing TensorFlow in case Kaggle pre-imports protobuf in the runtime. The rest of the model/training/submission logic be kept the same to avoid score-changing edits, and the script still write a valid `submission.csv` matching the sample submission format.'
- What this solution (achieved 0.25909) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by ensuring the protobuf environment variables are set before any TensorFlow-related import and by force-reloading protobuf modules so the setting actually takes effect in Kaggle’s preloaded runtime. This is an execution-critical change and is score-neutral (it just makes the notebook run). I keep your model, loss, training loop, and submission logic the same, only adjusting the import order to guarantee TensorFlow can be imported reliably. The script then train, predict, and write a valid `submission.csv` matching `sample_submission.csv`.'
- What this solution (achieved 0.26534) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation *before any protobuf/tensorflow import*, and by removing any already-loaded `google.protobuf` modules then importing `google.protobuf` first to “lock in” the pure-Python backend before importing TensorFlow. This is execution-critical and should be score-neutral (it just ensures the same training/inference can run). I also add a small safety guard so the submission builder always produces exactly the sample submission’s row order/shape, without changing the model or training logic. The rest of the model, loss, training loop, and post-processing are kept identical to preserve the achieved score behavior while restoring end-to-end execution.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

import random
import numpy as np
import pandas as pd

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)



## === cell 1
target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
scored_cols = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]




## === cell 2
def read_json(filename):
    """
    reads in train/test json data as pandas DataFrame
    """
    with open(filename, "r") as f:
        df = pd.read_json(path_or_buf=f, orient="records", lines=True)
    return df




## === cell 3
TRAIN_PATH = "../input/stanford-covid-vaccine/train.json"
TEST_PATH = "../input/stanford-covid-vaccine/test.json"
SAMPLE_SUB_PATH = "../input/stanford-covid-vaccine/sample_submission.csv"

train_df = read_json(TRAIN_PATH)
test_df = read_json(TEST_PATH)

print(train_df["id"].nunique())
print(train_df.columns)



## === cell 4
print("Features only in training set (not including target columns):")
print(set(train_df.columns) - set(test_df.columns) - set(target_cols))




## === cell 5
def unpack_df_lists(df, col_names):
    """
    turn list-like elements of dataframe into tabular data
    """
    if isinstance(col_names, str):
        col_names = [col_names]

    all_series = [df[c] for c in col_names]
    unpacked = [ser.explode() for ser in all_series]
    data = pd.concat(unpacked, axis=1)

    original = df.drop(col_names, axis=1)
    data = original.join(data)

    return data




## === cell 6
class CharTokenizer:
    def __init__(self, vocab, oov_token=None):
        self.vocab = list(vocab)
        self.oov_token = oov_token
        self.char_to_id = {ch: i + 1 for i, ch in enumerate(self.vocab)}  # start at 1
        self.word_index = {ch: i + 1 for i, ch in enumerate(self.vocab)}
        if oov_token is not None and oov_token not in self.char_to_id:
            self.word_index[oov_token] = len(self.word_index) + 1
            self.char_to_id[oov_token] = self.word_index[oov_token]

    def texts_to_sequences(self, texts):
        out = []
        for t in texts:
            seq = []
            for ch in t:
                if ch in self.char_to_id:
                    seq.append(self.char_to_id[ch])
                elif self.oov_token is not None:
                    seq.append(self.char_to_id[self.oov_token])
                else:
                    seq.append(0)
            out.append(seq)
        return out


tokenize_cols = ["sequence", "structure", "predicted_loop_type"]

tokenizer = CharTokenizer(vocab="().ACGUBEHIMSX")


def tokenize_df(df, tokenizer, cols=tokenize_cols):
    data = df.copy()
    for c in cols:
        data[c] = tokenizer.texts_to_sequences(data[c].astype(str).tolist())
    return data




## === cell 7
train_df = train_df[train_df["SN_filter"] == 1].reset_index(drop=True)
train_df = tokenize_df(train_df, tokenizer)

test_df = tokenize_df(test_df, tokenizer)

print(train_df.shape, test_df.shape)



## === cell 8
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

import sys

for m in list(sys.modules.keys()):
    if m.startswith("google.protobuf"):
        sys.modules.pop(m, None)

import google.protobuf  # noqa: F401

import tensorflow as tf


def mcrmse_tf(y_true, y_pred, scored_idx=(0, 1, 3)):
    """
    y_true,y_pred: (batch, 5, L_scored) per this notebook's model output.
    Return scalar loss.
    """
    idx = tf.constant(list(scored_idx), dtype=tf.int32)
    yt = tf.gather(y_true, idx, axis=1)
    yp = tf.gather(y_pred, idx, axis=1)
    mse = tf.reduce_mean(tf.square(yt - yp), axis=[0, 2])  # shape (3,)
    rmse = tf.sqrt(mse + 1e-8)
    return tf.reduce_mean(rmse)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 9
train_only_cols = [
    "reactivity_error",
    "deg_error_Mg_pH10",
    "deg_error_pH10",
    "deg_error_Mg_50C",
    "deg_error_50C",
]
signal_cols = ["signal_to_noise", "SN_filter"]
drop_cols = ["seq_length", "seq_scored", "index", "id"]
train_drop_cols = drop_cols + target_cols + train_only_cols + signal_cols


def to_fixed_int_seq(x, L=107):
    x = list(x)
    if len(x) >= L:
        return np.asarray(x[:L], dtype=np.int32)
    return np.asarray(x + [0] * (L - len(x)), dtype=np.int32)


def to_fixed_float_seq(x, L=107):
    x = list(x)
    if len(x) >= L:
        return np.asarray(x[:L], dtype=np.float32)
    return np.asarray(x + [0.0] * (L - len(x)), dtype=np.float32)


X_train = (
    train_df.drop(train_drop_cols, axis=1)
    .apply(lambda row: [to_fixed_int_seq(e, 107) for e in row], axis=1)
    .apply(lambda e: np.array(e, dtype=np.int32))
)
X_train = np.stack(X_train.values, axis=0)

y_train = (
    train_df[target_cols]
    .apply(lambda row: [to_fixed_float_seq(e, 107) for e in row], axis=1)
    .apply(lambda e: np.array(e, dtype=np.float32))
)
y_train = np.stack(y_train.values, axis=0)

print("X_train:", X_train.shape, X_train.dtype)
print("y_train:", y_train.shape, y_train.dtype)



## === cell 10
sw = train_df["signal_to_noise"].values.astype(np.float32)
sw = np.log1p(sw + 5) / 2.0



## === cell 11
import tensorflow.keras.layers as layers


def make_model():
    EMBEDDING_PARAMS = {
        "input_dim": len(tokenizer.word_index) + 1,  # +1 for padding idx 0
        "output_dim": 100,
    }

    inputs = tf.keras.Input(shape=(3, 107), dtype=tf.int32)
    embed = layers.Embedding(**EMBEDDING_PARAMS)(inputs)

    def rnn_layer():
        return layers.Bidirectional(layers.LSTM(30, return_sequences=True))

    rnn_layers = []
    for i in range(3):
        r = rnn_layer()(embed[:, i])  # (batch, 107, hidden)
        r = rnn_layer()(r)
        rnn_layers.append(r)

    x = layers.Concatenate()(rnn_layers)  # (batch, 107, feat)
    x = layers.Dense(100, activation="relu")(x)
    x = layers.Dense(5, activation="linear")(x)  # (batch, 107, 5)

    x = layers.Permute((2, 1))(x)  # (batch, 5, 107)
    x = layers.Lambda(lambda t: t[:, :, :-39], name="crop_to_68")(x)  # (batch, 5, 68)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(
        optimizer="adam",
        loss=mcrmse_tf,
        metrics=[tf.keras.metrics.MeanSquaredError(name="mse")],
    )
    return model


model = make_model()
model.summary()



## === cell 12
TF_FITPARAMS = {"epochs": 100, "batch_size": 100, "verbose": 2}

y_train_crop = y_train[:, :, :68]

history = model.fit(X_train, y_train_crop, sample_weight=sw, **TF_FITPARAMS)



## === cell 13
test_public = test_df[test_df["seq_length"] == 107].reset_index(drop=True)
test_private = test_df[test_df["seq_length"] == 130].reset_index(drop=True)

X_test_public = (
    test_public.drop(drop_cols, axis=1)
    .apply(lambda row: [to_fixed_int_seq(e, 107) for e in row], axis=1)
    .apply(lambda e: np.array(e, dtype=np.int32))
)
X_test_public = (
    np.stack(X_test_public.values, axis=0)
    if len(X_test_public)
    else np.empty((0, 3, 107), dtype=np.int32)
)

if len(test_private) > 0:
    X_test_private = (
        test_private.drop(drop_cols, axis=1)
        .apply(lambda row: [to_fixed_int_seq(e, 107) for e in row], axis=1)
        .apply(lambda e: np.array(e, dtype=np.int32))
    )
    X_test_private = np.stack(X_test_private.values, axis=0)
else:
    X_test_private = np.empty((0, 3, 107), dtype=np.int32)

print("X_test_public:", X_test_public.shape)
print("X_test_private:", X_test_private.shape)



## === cell 14
test_pred_public = (
    model.predict(X_test_public, verbose=0)
    if X_test_public.shape[0]
    else np.empty((0, 5, 68), dtype=np.float32)
)
test_pred_private = (
    model.predict(X_test_private, verbose=0)
    if X_test_private.shape[0]
    else np.empty((0, 5, 68), dtype=np.float32)
)

print("pred public:", test_pred_public.shape)
print("pred private:", test_pred_private.shape)




## === cell 15
def create_sub_df(test_df_part, predictions, length):
    sub_df = test_df_part.drop(tokenize_cols + ["index", "seq_scored"], axis=1)
    sub_df["seqpos"] = sub_df.apply(
        lambda row: list(range(int(row["seq_length"]))), axis=1
    )
    sub_df = unpack_df_lists(sub_df, "seqpos")
    sub_df["id_seqpos"] = sub_df.apply(
        lambda row: f"{row['id']}_{int(row['seqpos'])}", axis=1
    )

    def pad_pred(p, final_len):
        start = list(p)
        if len(start) >= final_len:
            return start[:final_len]
        return start + [0.0] * (final_len - len(start))

    pred_df = pd.DataFrame(
        [
            [
                pad_pred(predictions[i, j], int(test_df_part.iloc[i]["seq_length"]))
                for j in range(predictions.shape[1])
            ]
            for i in range(predictions.shape[0])
        ],
        columns=target_cols,
    )
    pred_df = unpack_df_lists(pred_df, target_cols)

    pred_df.index = sub_df.index
    out = pd.DataFrame({"id_seqpos": sub_df["id_seqpos"].values})
    out = out.join(pred_df.reset_index(drop=True))
    return out




## === cell 16
sub_parts = []
if len(test_public) > 0:
    sub_parts.append(create_sub_df(test_public, test_pred_public, 107))
if len(test_private) > 0:
    sub_parts.append(create_sub_df(test_private, test_pred_private, 130))

sub_df = (
    pd.concat(sub_parts, axis=0).reset_index(drop=True)
    if sub_parts
    else pd.DataFrame(columns=["id_seqpos"] + target_cols)
)

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

sub_df = sub_df.set_index("id_seqpos").reindex(sample_sub["id_seqpos"]).reset_index()

for c in target_cols:
    sub_df[c] = sub_df[c].astype(np.float32)
sub_df[target_cols] = sub_df[target_cols].fillna(0.0)

sub_df = sub_df[["id_seqpos"] + target_cols]

assert list(sub_df.columns) == list(sample_sub.columns), (
    sub_df.columns,
    sample_sub.columns,
)
assert len(sub_df) == len(sample_sub), (len(sub_df), len(sample_sub))

print(sub_df.shape)
print(sub_df.head())



## === cell 17
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with columns:", list(sub_df.columns))
print("Preview:")
print(sub_df.head())
