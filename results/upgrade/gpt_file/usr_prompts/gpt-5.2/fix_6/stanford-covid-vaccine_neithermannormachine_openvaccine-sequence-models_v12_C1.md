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

0.71003

# 6. Current score

0.26401

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.2592) has done: 'I fix the TensorFlow/Keras runtime errors so the model can be built and trained (mainly replacing raw `tf.transpose` on a KerasTensor with a Keras layer op), and I remove the broken protobuf/Tokenizer dependency that triggers the `MessageFactory.GetPrototype` error by switching to a simple character-to-integer mapping with the same “char-level tokenization” semantics. I also make the BPP loading optional/robust because your provided dataset tree doesn’t include the `bpps/` files; when missing, the pipeline fall back to the sequence-only model so it still runs end-to-end. Finally, I ensure the submission is created with the exact `sample_submission.csv` row order and columns, writing a valid `submission.csv` every time.'
- What this solution (achieved 0.26366) has done: 'I remove the protobuf-triggering dependency that causes the `MessageFactory.GetPrototype` crash by forcing pure-Python protobuf and clearing any TensorFlow Text/Hub proto usage paths, so the notebook imports cleanly under TF 2.18. I also fix a logic bug in how `sample_weight` is passed (it currently doesn’t apply to the validation set and can be misaligned), while preserving the same model, loss, and training loop semantics. Finally, I make the data root resolution robust to both `/kaggle/input/...` and your provided `/kaggle/data/...` layout and ensure the submission is always written as `submission.csv` matching `sample_submission.csv` row order and required columns.'
- What this solution (achieved 0.26359) has done: 'To fix the `MessageFactory.GetPrototype` crash under TF 2.18 / protobuf 6, I force the pure-Python protobuf runtime and (crucially) do it before importing TensorFlow, while also ensuring any pre-imported protobuf modules don’t keep the C++ implementation. I keep your model/training logic intact and only touch the minimal environment/bootstrap code needed for stable imports. I also add a small safety check so the submission always matches `sample_submission.csv` row count and order (without changing predictions), ensuring Kaggle accepts the file. No score-changing modeling/training edits are introduced beyond restoring the pipeline to run end-to-end.'
- What this solution (achieved 0.26883) has done: 'I fix the protobuf/TensorFlow import crash by pinning protobuf to the pure-Python runtime before TensorFlow loads, and by installing a small compatibility shim for `MessageFactory.GetPrototype` (needed by some TF/TFDS/TFTXT paths under protobuf 6). This is a correctness/stability fix only and won’t change your model/training logic or submission formatting. I also remove any unnecessary protobuf-module deletion that can re-trigger the bad state after TensorFlow starts importing submodules. After that, the pipeline run end-to-end and still write `submission.csv` with the exact required columns and row order.'
- What this solution (achieved 0.26401) has done: 'I fix the protobuf compatibility shim so it no longer throws `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` under protobuf 6 / TF 2.18 by patching the correct class (`google.protobuf.message_factory.MessageFactory`) and doing it safely before importing TensorFlow. This is a runtime/stability fix only and won’t change your model, training loop, or submission formatting, so it should preserve the current score behavior aside from negligible nondeterminism. I also keep the pure-Python protobuf forcing in place (it’s required in this environment) and make the shim robust across protobuf versions. Everything else (data loading, tokenization, model, training, inference, and CSV writing) remains the same so the pipeline runs end-to-end and produces `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import sys
import numpy as np
import pandas as pd

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import google.protobuf  # noqa: E402
from google.protobuf import message_factory as _message_factory  # noqa: E402

_MF = getattr(_message_factory, "MessageFactory", None)
if _MF is not None:
    if (not hasattr(_MF, "GetPrototype")) and hasattr(_MF, "GetMessageClass"):

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        setattr(_MF, "GetPrototype", _GetPrototype)

import tensorflow as tf  # noqa: E402
import tensorflow.keras.layers as layers  # noqa: E402
from tensorflow.keras.optimizers import Adam  # noqa: E402

np.random.seed(42)
tf.random.set_seed(42)

print("TF:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
tokenize_cols = ["sequence", "structure", "predicted_loop_type"]

DATA_ROOT_CANDIDATES = [
    "/kaggle/input/stanford-covid-vaccine",
    "/kaggle/data/stanford-covid-vaccine",
    "../input/stanford-covid-vaccine",
    "../data/stanford-covid-vaccine",
]


def resolve_data_root():
    for p in DATA_ROOT_CANDIDATES:
        if os.path.exists(p) and os.path.exists(os.path.join(p, "train.json")):
            return p
    for p in ["/kaggle/input", "/kaggle/data", "../input", "../data"]:
        cand = os.path.join(p, "stanford-covid-vaccine")
        if os.path.exists(cand) and os.path.exists(os.path.join(cand, "train.json")):
            return cand
    raise FileNotFoundError("Could not locate stanford-covid-vaccine dataset folder.")


DATA_ROOT = resolve_data_root()
TRAIN_PATH = os.path.join(DATA_ROOT, "train.json")
TEST_PATH = os.path.join(DATA_ROOT, "test.json")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

print("DATA_ROOT:", DATA_ROOT)




## === cell 2
def read_json(filename):
    """
    reads in train/test json data as pandas DataFrame
    """
    return pd.read_json(path_or_buf=filename, orient="records", lines=True)


train_df = read_json(TRAIN_PATH)
test_df = read_json(TEST_PATH)

print("train rows:", len(train_df), "unique ids:", train_df["id"].nunique())
print("test rows:", len(test_df), "unique ids:", test_df["id"].nunique())



## === cell 3
VOCAB = list("().ACGUBEHIMSX")
char2idx = {c: i + 1 for i, c in enumerate(VOCAB)}  # reserve 0 for unknown/pad


def encode_string(s, mapping=char2idx):
    return [mapping.get(ch, 0) for ch in s]


def tokenize_df(df, cols=tokenize_cols):
    data = df.copy()
    for c in cols:
        data[c] = data[c].map(encode_string)
    return data




## === cell 4
train_df = train_df[train_df["SN_filter"] == 1].reset_index(drop=True)
train_df = tokenize_df(train_df)

test_df_tok = tokenize_df(test_df)

print("Filtered train rows:", len(train_df))




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
def score(raw_values=False, use_tf=False, **kwargs):
    """
    Competition metric (MCRMSE over scored columns). In training we use a TF version.
    Note: scored columns: reactivity, deg_Mg_pH10, deg_Mg_50C.
    y_true/y_pred are shaped (batch, 5, seq_scored).
    """
    scored_idx = [0, 1, 3]

    def loss_np(y_true, y_pred):
        from sklearn.metrics import mean_squared_error

        y_true = np.asarray(y_true)[:, scored_idx]
        y_pred = np.asarray(y_pred)[:, scored_idx]
        multi = "raw_values" if raw_values else "uniform_average"
        return mean_squared_error(y_true, y_pred, squared=False, multioutput=multi)

    def loss_tf(y_true, y_pred):
        y_true_s = tf.gather(y_true, scored_idx, axis=1)
        y_pred_s = tf.gather(y_pred, scored_idx, axis=1)
        mse = tf.reduce_mean(tf.square(y_true_s - y_pred_s), axis=2)  # (batch, 3)
        rmse = tf.sqrt(mse + 1e-8)  # (batch, 3)
        return tf.reduce_mean(rmse, axis=1)  # (batch,)

    return loss_tf if use_tf else loss_np




## === cell 7
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

X_train = (
    train_df.drop(train_drop_cols, axis=1)
    .apply(lambda row: [e for e in row], axis=1)
    .apply(lambda e: np.array(e, dtype=np.int32))
)
X_train = np.stack(X_train.values, axis=0)  # (n, 3, seq_length=107)

y_train = (
    train_df[target_cols]
    .apply(lambda row: [e for e in row], axis=1)
    .apply(lambda e: np.array(e, dtype=np.float32))
)
y_train = np.stack(y_train.values, axis=0)  # (n, 5, seq_scored=68)

print("X_train:", X_train.shape, X_train.dtype)
print("y_train:", y_train.shape, y_train.dtype)



## === cell 8
sw = train_df["signal_to_noise"].values.astype(np.float32)
sw = np.log1p(sw + 5) / 2.0
print("sample_weight stats:", float(sw.min()), float(sw.max()), float(sw.mean()))



## === cell 9
TF_COMPILEPARAMS = {
    "optimizer": Adam(learning_rate=0.01),
    "loss": score(use_tf=True),
    "metrics": ["mse"],
}


def make_model():
    """
    Creates a tensorflow keras sequence model.
    Core logic preserved; only change is using Keras ops to avoid tf.transpose on KerasTensor.
    """
    EMBEDDING_PARAMS = {
        "input_dim": len(char2idx) + 1,  # +1 for 0
        "output_dim": 100,
    }

    shape = (3, None)  # (3, seq_length)
    seq_inputs = tf.keras.Input(shape=shape, dtype=tf.int32)
    embed = layers.Embedding(**EMBEDDING_PARAMS)(seq_inputs)  # (n, 3, L, D)

    def rnn_layer(num_neurons):
        return layers.Bidirectional(
            layers.LSTM(num_neurons, return_sequences=True, dropout=0.2)
        )

    rnn_layers = []
    for i in range(3):
        r = rnn_layer(30)(embed[:, i])  # (n, L, 60)
        r = rnn_layer(30)(r)  # (n, L, 60)
        rnn_layers.append(r)

    x = layers.Concatenate()(rnn_layers)  # (n, L, 180)
    x = layers.Dense(100, activation="relu")(x)  # (n, L, 100)
    x = layers.Dense(5, activation="linear")(x)  # (n, L, 5)

    x = layers.Permute((2, 1))(x)  # (n, 5, L)
    x = layers.Lambda(lambda t: t[:, :, :-39], name="crop_to_scored")(x)  # (n, 5, 68)

    model = tf.keras.Model(inputs=seq_inputs, outputs=x)
    model.compile(**TF_COMPILEPARAMS)
    return model




## === cell 10
def read_bpp(ids):
    base_path = os.path.join(DATA_ROOT, "bpps")
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"bpps folder not found at {base_path}")
    filepaths = [os.path.join(base_path, f"{rna_id}.npy") for rna_id in ids]
    bpps = [np.load(fp) for fp in filepaths]
    return np.stack(bpps, axis=0)


def make_model_bpp(seq_model):
    """
    Preserved for compatibility, but will only be used if bpps exist.
    """
    seq_input = seq_model.input
    seq_layers = seq_model.layers
    intercept_layer = 12
    for l in seq_layers[:intercept_layer]:
        l.trainable = False

    bpp_shape = (107, 107)
    bpp_input = tf.keras.Input(shape=bpp_shape, dtype=tf.float32)

    x = seq_layers[intercept_layer].output
    x = layers.Concatenate()([x, bpp_input])
    x = layers.Dense(250, activation="relu")(x)
    x = layers.Dense(100, activation="relu")(x)

    for l in seq_layers[intercept_layer + 1 :]:
        x = l(x)

    model = tf.keras.Model(inputs=[seq_input, bpp_input], outputs=x)
    model.compile(**TF_COMPILEPARAMS)
    return model




## === cell 11
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from sklearn.model_selection import train_test_split

TF_FITPARAMS = {"epochs": 150, "batch_size": 100, "validation_batch_size": 50}
fp = TF_FITPARAMS


def schedule_func(epoch, lr):
    if epoch < 50:
        return lr
    return lr * np.exp(-0.13)


callbacks = [
    LearningRateScheduler(schedule_func),
    EarlyStopping(
        monitor="val_loss",
        mode="min",
        min_delta=5e-5,
        patience=10,
        restore_best_weights=True,
    ),
]

idx_tr, idx_val = train_test_split(
    np.arange(len(X_train)), test_size=0.1, random_state=42
)

X_tr, y_tr = X_train[idx_tr], y_train[idx_tr]
X_val, y_val = X_train[idx_val], y_train[idx_val]
sw_tr, sw_val = sw[idx_tr], sw[idx_val]

model = make_model()
model.summary()

history = model.fit(
    X_tr,
    y_tr,
    validation_data=(X_val, y_val, sw_val),
    sample_weight=sw_tr,
    callbacks=callbacks,
    **fp,
)



## === cell 12
use_bpp = False
model_bpp = None
history_bpp = None

try:
    _ = read_bpp(train_df.loc[idx_tr, "id"])
    use_bpp = True
except Exception as e:
    print("BPP features not available; using sequence-only model. Reason:", repr(e))

if use_bpp:
    bpp_train = read_bpp(train_df["id"])
    bpp_tr, bpp_val = bpp_train[idx_tr], bpp_train[idx_val]
    model_bpp = make_model_bpp(model)
    model_bpp.summary()
    history_bpp = model_bpp.fit(
        [X_tr, bpp_tr],
        y_tr,
        validation_data=([X_val, bpp_val], y_val, sw_val),
        sample_weight=sw_tr,
        callbacks=callbacks,
        **fp,
    )



## === cell 13
test_public = test_df_tok[test_df_tok["seq_length"] == 107].reset_index(drop=True)
test_private = test_df_tok[test_df_tok["seq_length"] == 130].reset_index(drop=True)

X_test_public = (
    test_public.drop(drop_cols, axis=1)
    .apply(lambda row: [e for e in row], axis=1)
    .apply(lambda e: np.array(e, dtype=np.int32))
)
X_test_public = np.stack(X_test_public.values, axis=0)

X_test_private = None
if len(test_private) > 0:
    X_test_private = (
        test_private.drop(drop_cols, axis=1)
        .apply(lambda row: [e for e in row], axis=1)
        .apply(lambda e: np.array(e, dtype=np.int32))
    )
    X_test_private = np.stack(X_test_private.values, axis=0)

print("X_test_public:", X_test_public.shape)
print("private n:", len(test_private))




## === cell 14
def predict_for_df(df_tok, X_tok):
    """
    Returns predictions shaped (n, 5, 68) for each id, using BPP model if available.
    """
    if use_bpp and model_bpp is not None:
        bpp = read_bpp(df_tok["id"])
        return model_bpp.predict([X_tok, bpp], verbose=0)
    return model.predict(X_tok, verbose=0)


test_pred_public = predict_for_df(test_public, X_test_public)

test_pred_private = None
if X_test_private is not None:
    Xp = X_test_private[:, :, :107]
    test_pred_private = predict_for_df(test_private.assign(seq_length=107), Xp)




## === cell 15
def preds_to_long_df(test_df_raw, preds_scored_68):
    """
    Create long-format predictions for a given test_df (one row per id_seqpos),
    padding positions > 68 with zeros up to seq_length.
    """
    rows = []
    for i, r in test_df_raw.iterrows():
        rid = r["id"]
        seq_len = int(r["seq_length"])
        p = preds_scored_68[i]
        full = np.zeros((5, seq_len), dtype=np.float32)
        L = min(68, seq_len)
        full[:, :L] = p[:, :L]
        for pos in range(seq_len):
            rows.append([f"{rid}_{pos}"] + [float(full[j, pos]) for j in range(5)])
    return pd.DataFrame(rows, columns=["id_seqpos"] + target_cols)




## === cell 16
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

sub_parts = []
sub_parts.append(
    preds_to_long_df(
        test_public[["id", "seq_length"]].assign(seq_length=107), test_pred_public
    )
)

if test_pred_private is not None and len(test_private) > 0:
    sub_parts.append(
        preds_to_long_df(test_private[["id", "seq_length"]], test_pred_private)
    )

pred_long = pd.concat(sub_parts, axis=0, ignore_index=True)

sub_df = sample_sub[["id_seqpos"]].merge(pred_long, on="id_seqpos", how="left")

for c in target_cols:
    sub_df[c] = sub_df[c].fillna(0.0)

sub_df = sub_df[["id_seqpos"] + target_cols]

assert sub_df.shape[0] == sample_sub.shape[0], (sub_df.shape, sample_sub.shape)

print("submission shape:", sub_df.shape)
print(sub_df.head())



## === cell 17
sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print("Wrote:", sub_path)
print("File size bytes:", os.path.getsize(sub_path))
print("Columns:", list(sub_df.columns))
