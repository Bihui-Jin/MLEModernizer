# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

No external packages required in the script and installed.

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

# 5. Code solution

## === cell 0
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"



## === cell 1
import sys

_bad_paths = [p for p in list(sys.path) if ("site-packages" in p and "protobuf" in p)]
for p in _bad_paths:
    try:
        sys.path.remove(p)
    except ValueError:
        pass

import json, math
import pandas as pd
import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K  # noqa: F401
import tensorflow.keras.layers as L

import warnings

warnings.filterwarnings("ignore")

from itertools import product
from sklearn.model_selection import GroupKFold


class _NoColor:
    YELLOW = ""
    RESET_ALL = ""


Fore = _NoColor()
Style = _NoColor()



## === cell 2
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass



## === cell 3
SEED = 53

n_folds = 5
debug = True
Window_features = True

model_name = "GG"
epochs = 100
BATCH_SIZE = 32
n_layers = 2
layers = ["GRU", "GRU"]
hidden_dim = [128, 128]
dropout = [0.5, 0.5]
sp_dropout = 0.2
embed_dim = 150
num_hidden_units = 8

Cosine_Schedule = True
Rampup_decy_lr = False




## === cell 4
def seed_everything(seed=1234):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


seed_everything(SEED)



## === cell 5
target_cols = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C", "deg_pH10", "deg_50C"]
window_columns = ["sequence", "structure", "predicted_loop_type"]

categorical_features = ["sequence", "structure", "predicted_loop_type"]

cat_feature = len(categorical_features)
if Window_features:
    cat_feature += len(window_columns)

numerical_features = [
    "BPPS_Max",
    "BPPS_nb",
    "BPPS_sum",
    "positional_entropy",
    "stems",
    "interior_loops",
    "multiloops",
    "A_percent",
    "G_percent",
    "C_percent",
    "U_percent",
    "pair_map",
    "pair_distance",
]

num_features = len(numerical_features)
feature_cols = categorical_features + numerical_features
pred_col_names = ["pred_" + c_name for c_name in target_cols]

target_eval_col = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
pred_eval_col = ["pred_" + c_name for c_name in target_eval_col]

SEQ_LEN_SHORT = 107
SEQ_LEN_LONG = 130
PRED_LEN = 68  # scored length



## === cell 6
data_dir = "/kaggle/input/stanford-covid-vaccine/"

train = pd.read_json(os.path.join(data_dir, "train.json"), lines=True)
test = pd.read_json(os.path.join(data_dir, "test.json"), lines=True)
sample_sub = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))

print("train:", train.shape, "test:", test.shape, "sample_sub:", sample_sub.shape)
print("sample_sub columns:", list(sample_sub.columns))



## === cell 7
train = train[train["signal_to_noise"] >= 0.5].reset_index(drop=True)
print("train after SNR filter:", train.shape)

if "cnt" not in train.columns:
    train["cnt"] = 1




## === cell 8
def _add_minimal_numerical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    seq = df["sequence"].astype(str)
    struct = df["structure"].astype(str)

    df["A_percent"] = seq.apply(lambda s: s.count("A") / max(1, len(s)))
    df["C_percent"] = seq.apply(lambda s: s.count("C") / max(1, len(s)))
    df["G_percent"] = seq.apply(lambda s: s.count("G") / max(1, len(s)))
    df["U_percent"] = seq.apply(lambda s: s.count("U") / max(1, len(s)))

    df["stems"] = struct.apply(lambda s: (s.count("(") + s.count(")")) / max(1, len(s)))
    df["interior_loops"] = 0.0
    df["multiloops"] = 0.0

    df["BPPS_Max"] = 0.0
    df["BPPS_nb"] = 0.0
    df["BPPS_sum"] = 0.0
    df["positional_entropy"] = 0.0
    df["pair_map"] = 0.0
    df["pair_distance"] = 0.0

    for c in numerical_features:
        if c not in df.columns:
            df[c] = 0.0

    return df


train = _add_minimal_numerical_features(train)
test = _add_minimal_numerical_features(test)




## === cell 9
def pair_feature(row):
    arr = list(row)
    its = [iter(["_"] + arr[:]), iter(arr[1:] + ["_"])]
    list_touple = list(zip(*its))
    return list(map("".join, list_touple))




## === cell 10
def _pad_or_truncate_tokens(tokens, seq_len, pad_token):
    if len(tokens) >= seq_len:
        return tokens[:seq_len]
    return tokens + [pad_token] * (seq_len - len(tokens))


def preprocess_categorical_inputs(
    df,
    cols=categorical_features,
    Window_features=Window_features,
    seq_len=SEQ_LEN_SHORT,
):
    cols_local = list(cols)

    if Window_features:
        df = df.copy()
        for c in window_columns:
            pc = "pair_" + c
            if pc not in df.columns:
                df[pc] = df[c].apply(pair_feature)
            cols_local.append(pc)

    cols_local = list(dict.fromkeys(cols_local))  # stable unique

    pad_token = "_"
    pad_idx = token2int.get(pad_token, 0)

    X = np.zeros((len(df), seq_len, len(cols_local)), dtype=np.int32)
    for i in range(len(df)):
        for j, c in enumerate(cols_local):
            seq = df.iloc[i][c]
            if isinstance(seq, str):
                tokens = list(seq)
            else:
                tokens = list(seq)
            tokens = _pad_or_truncate_tokens(tokens, seq_len, pad_token)
            X[i, :, j] = [token2int.get(t, pad_idx) for t in tokens]

    return X




## === cell 11
def preprocess_numerical_inputs(df, cols=numerical_features, seq_len=SEQ_LEN_SHORT):
    X = df[cols].astype(np.float32).values  # (n, num_features)
    X = np.repeat(X[:, None, :], repeats=seq_len, axis=1)  # (n, seq_len, num_features)
    return X




## === cell 12
base_tokens = set(list("().ACGU"))  # structure + sequence
loop_tokens = set(list("BEHIMSX"))  # bpRNA loop types seen in this dataset
extra_tokens = set(list("pshftim"))  # tokens present in original notebook token list

token_set = base_tokens | loop_tokens | extra_tokens | set(["_"])

if Window_features:
    pair_alphabet = sorted(list(token_set))
    token_set |= set([a + b for a, b in product(pair_alphabet, repeat=2)])

token_list = sorted(list(token_set))
token2int = {x: i for i, x in enumerate(token_list)}

print("token_list Size :", len(token_list))

train_inputs_all_cat = preprocess_categorical_inputs(
    train, cols=categorical_features, seq_len=SEQ_LEN_SHORT
)
train_inputs_all_num = preprocess_numerical_inputs(
    train, cols=numerical_features, seq_len=SEQ_LEN_SHORT
)
train_labels_all = np.array(
    train[target_cols].values.tolist(), dtype=np.float32
).transpose((0, 2, 1))

print("Train categorical Features Shape : ", train_inputs_all_cat.shape)
print("Train numerical Features Shape : ", train_inputs_all_num.shape)
print("Train labels Shape : ", train_labels_all.shape)



## === cell 13
public_df = test.query(f"seq_length == {SEQ_LEN_SHORT}").reset_index(drop=True)
private_df = test.query(f"seq_length == {SEQ_LEN_LONG}").reset_index(drop=True)

print("public_df : ", public_df.shape)
print("private_df : ", private_df.shape)

public_inputs_cat = preprocess_categorical_inputs(
    public_df, cols=categorical_features, seq_len=SEQ_LEN_SHORT
)
private_inputs_cat = preprocess_categorical_inputs(
    private_df, cols=categorical_features, seq_len=SEQ_LEN_LONG
)

public_inputs_num = preprocess_numerical_inputs(
    public_df, cols=numerical_features, seq_len=SEQ_LEN_SHORT
)
private_inputs_num = preprocess_numerical_inputs(
    private_df, cols=numerical_features, seq_len=SEQ_LEN_LONG
)

print("Public categorical Features Shape : ", public_inputs_cat.shape)
print("Public numerical Features Shape : ", public_inputs_num.shape)
print("Private categorical Features Shape : ", private_inputs_cat.shape)
print("Private numerical Features Shape : ", private_inputs_num.shape)




## === cell 14
def MCRMSE(y_true, y_pred):
    colwise_mse = tf.reduce_mean(tf.square(y_true[:, :, :3] - y_pred[:, :, :3]), axis=1)
    return tf.reduce_mean(tf.sqrt(colwise_mse), axis=1)




## === cell 15
def get_lr_callback(batch_size=8):
    lr_start = 0.00001
    lr_max = 0.004
    lr_min = 0.00005
    lr_ramp_ep = 45
    lr_sus_ep = 2
    lr_decay = 0.8

    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) * lr_decay ** (
                epoch - lr_ramp_ep - lr_sus_ep
            ) + lr_min
        return lr

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)
    return lr_callback




## === cell 16
def get_cosine_schedule_with_warmup(
    lr, num_warmup_steps, num_training_steps, num_cycles=3.5
):
    def lrfn(epoch):
        if epoch < num_warmup_steps:
            return (float(epoch) / float(max(1, num_warmup_steps))) * lr
        progress = float(epoch - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0,
            0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)) * lr,
        )

    return tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)




## === cell 17
def lstm_layer(hidden_dim, dropout):
    return tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            hidden_dim,
            dropout=dropout,
            return_sequences=True,
            kernel_initializer="orthogonal",
        )
    )


def gru_layer(hidden_dim, dropout):
    return tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(
            hidden_dim,
            dropout=dropout,
            return_sequences=True,
            kernel_initializer="orthogonal",
        )
    )




## === cell 18
def build_model(
    embed_size,
    seq_len=107,
    pred_len=68,
    dropout=dropout,
    sp_dropout=sp_dropout,
    num_features=num_features,
    num_hidden_units=num_hidden_units,
    embed_dim=embed_dim,
    layers=layers,
    hidden_dim=hidden_dim,
    n_layers=n_layers,
    cat_feature=cat_feature,
):
    inputs = L.Input(shape=(seq_len, cat_feature), name="category_input")
    embed = L.Embedding(input_dim=embed_size, output_dim=embed_dim)(inputs)

    reshaped = L.Reshape((seq_len, cat_feature * embed_dim))(embed)
    reshaped_conv = L.Conv1D(
        filters=512, kernel_size=3, strides=1, padding="same", activation="elu"
    )(reshaped)

    numerical_input = L.Input(shape=(seq_len, num_features), name="numeric_input")
    n_Dense_1 = L.Dense(64)(numerical_input)
    n_Dense_2 = L.Dense(128)(n_Dense_1)
    numerical_conv = L.Conv1D(
        filters=256, kernel_size=4, strides=1, padding="same", activation="elu"
    )(n_Dense_2)

    hidden = L.concatenate([reshaped_conv, numerical_conv])
    hidden = L.SpatialDropout1D(sp_dropout)(hidden)

    for x in range(n_layers):
        if layers[x] == "GRU":
            hidden = gru_layer(hidden_dim[x], dropout[x])(hidden)
        else:
            hidden = lstm_layer(hidden_dim[x], dropout[x])(hidden)

    truncated = hidden[:, :pred_len]
    out = L.Dense(5)(truncated)

    model = tf.keras.Model(inputs=[inputs, numerical_input], outputs=out)
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss=MCRMSE)
    return model




## === cell 19
model = build_model(embed_size=len(token2int))
model.summary()



## === cell 20
if "cnt" not in train.columns or train["cnt"].isna().any():
    train["cnt"] = 1

id_counts = train["id"].value_counts()
train["cnt"] = train["id"].map(id_counts).astype(int)




## === cell 21
def get_stratify_group(row):
    snf = row["SN_filter"]
    snr = row["signal_to_noise"]
    id_ = row["id"]

    if snf == 0:
        if snr < 0:
            snr_c = 0
        elif 0 <= snr < 2:
            snr_c = 1
        elif 2 <= snr < 4:
            snr_c = 2
        elif 4 <= snr < 5.5:
            snr_c = 3
        elif 5.5 <= snr < 10:
            snr_c = 4
        else:
            snr_c = 5
    else:
        if snr < 0:
            snr_c = 6
        elif 0 <= snr < 1:
            snr_c = 7
        elif 1 <= snr < 2:
            snr_c = 8
        elif 2 <= snr < 3:
            snr_c = 9
        elif 3 <= snr < 4:
            snr_c = 10
        elif 4 <= snr < 5:
            snr_c = 11
        elif 5 <= snr < 6:
            snr_c = 12
        elif 6 <= snr < 7:
            snr_c = 13
        elif 7 <= snr < 8:
            snr_c = 14
        elif 8 <= snr < 10:
            snr_c = 15
        else:
            snr_c = 16
    return f"{id_}_{snr_c}"


train["stratify_group"] = train.apply(get_stratify_group, axis=1)
train["stratify_group"] = train["stratify_group"].astype("category").cat.codes



## === cell 22
gkf = GroupKFold(n_splits=n_folds)



## === cell 23
submission = pd.DataFrame(index=sample_sub.index, columns=target_cols).fillna(0.0)

val_losses = []
historys = []
oof_preds_all = []

for Fold, (train_index, val_index) in enumerate(
    gkf.split(train_inputs_all_cat, groups=train["stratify_group"])
):
    print(Fore.YELLOW + ("#" * 45))
    print("###  Fold : ", str(Fold + 1))
    print(("#" * 45) + Style.RESET_ALL)
    print(
        f"|| Batch_size: {BATCH_SIZE} \n|| n_layers: {n_layers} \n|| embed_dim: {embed_dim}"
    )
    print(f"|| cat_feature: {cat_feature} \n|| num_features: {num_features}")
    print(
        f"|| layers : {layers} \n|| hidden_dim: {hidden_dim} \n|| dropout: {dropout} \n|| sp_dropout: {sp_dropout}"
    )

    train_data = train.iloc[train_index].reset_index(drop=True)
    val_data = train.iloc[val_index].reset_index(drop=True)

    print(
        "|| number Augmented data Present in Val Data : ",
        len(val_data[val_data["cnt"] != 1]),
    )
    print(
        "|| number Augmented data Present in Train Data : ",
        len(train_data[train_data["cnt"] != 1]),
    )
    print("|| Data Lekage : ", len(val_data[val_data["id"].isin(train_data["id"])]))

    val_data = val_data[val_data["cnt"] == 1].reset_index(drop=True)

    model_train = build_model(
        embed_size=len(token2int), seq_len=SEQ_LEN_SHORT, pred_len=PRED_LEN
    )
    model_short = build_model(
        embed_size=len(token2int), seq_len=SEQ_LEN_SHORT, pred_len=SEQ_LEN_SHORT
    )
    model_long = build_model(
        embed_size=len(token2int), seq_len=SEQ_LEN_LONG, pred_len=SEQ_LEN_LONG
    )

    train_inputs_cat = preprocess_categorical_inputs(
        train_data, cols=categorical_features, seq_len=SEQ_LEN_SHORT
    )
    train_inputs_num = preprocess_numerical_inputs(
        train_data, cols=numerical_features, seq_len=SEQ_LEN_SHORT
    )
    train_labels = np.array(
        train_data[target_cols].values.tolist(), dtype=np.float32
    ).transpose((0, 2, 1))

    val_inputs_cat = preprocess_categorical_inputs(
        val_data, cols=categorical_features, seq_len=SEQ_LEN_SHORT
    )
    val_inputs_num = preprocess_numerical_inputs(
        val_data, cols=numerical_features, seq_len=SEQ_LEN_SHORT
    )
    val_labels = np.array(
        val_data[target_cols].values.tolist(), dtype=np.float32
    ).transpose((0, 2, 1))

    csv_logger = tf.keras.callbacks.CSVLogger(
        f"Fold_{Fold}_log.csv", separator=",", append=False
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        f"{model_name}_Fold_{Fold}.weights.h5",
        monitor="val_loss",
        verbose=0,
        mode="min",
        save_best_only=True,
        save_weights_only=True,
    )

    if Cosine_Schedule:
        lr_schedule = get_cosine_schedule_with_warmup(
            lr=0.001, num_warmup_steps=20, num_training_steps=epochs
        )
        callbacks = [lr_schedule, checkpoint, csv_logger]
    elif Rampup_decy_lr:
        lr_schedule = get_lr_callback(BATCH_SIZE)
        callbacks = [lr_schedule, checkpoint, csv_logger]
    else:
        lr_plateau = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, verbose=0
        )
        callbacks = [lr_plateau, checkpoint, csv_logger]

    history = model_train.fit(
        {"numeric_input": train_inputs_num, "category_input": train_inputs_cat},
        train_labels,
        validation_data=(
            {"numeric_input": val_inputs_num, "category_input": val_inputs_cat},
            val_labels,
        ),
        batch_size=BATCH_SIZE,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1 if debug else 0,
    )

    print("Min Validation Loss : ", float(np.min(history.history["val_loss"])))
    print("Min Validation Epoch : ", int(np.argmin(history.history["val_loss"]) + 1))
    val_losses.append(float(np.min(history.history["val_loss"])))
    historys.append(history)

    model_short.load_weights(f"{model_name}_Fold_{Fold}.weights.h5")
    model_long.load_weights(f"{model_name}_Fold_{Fold}.weights.h5")

    public_preds = model_short.predict(
        {"numeric_input": public_inputs_num, "category_input": public_inputs_cat},
        verbose=0,
    )
    private_preds = model_long.predict(
        {"numeric_input": private_inputs_num, "category_input": private_inputs_cat},
        verbose=0,
    )

    oof_preds = model_train.predict(
        {"numeric_input": val_inputs_num, "category_input": val_inputs_cat}, verbose=0
    )

    preds_model = []
    for df, preds in [(public_df, public_preds), (private_df, private_preds)]:
        for i, uid in enumerate(df.id):
            single_pred = preds[i]
            single_df = pd.DataFrame(single_pred, columns=target_cols)
            single_df["id_seqpos"] = [f"{uid}_{x}" for x in range(single_df.shape[0])]
            preds_model.append(single_df)

    preds_model_df = pd.concat(preds_model, ignore_index=True)
    preds_model_df = preds_model_df.groupby(["id_seqpos"], sort=False).mean(
        numeric_only=True
    )

    preds_aligned = sample_sub[["id_seqpos"]].merge(
        preds_model_df.reset_index(), on="id_seqpos", how="left"
    )
    preds_aligned[target_cols] = preds_aligned[target_cols].fillna(0.0)
    submission[target_cols] += preds_aligned[target_cols].values / n_folds

    for i, uid in enumerate(val_data.id):
        single_pred = oof_preds[i]
        single_label = val_labels[i]
        single_label_df = pd.DataFrame(single_label, columns=target_cols)
        single_label_df["id_seqpos"] = [
            f"{uid}_{x}" for x in range(single_label_df.shape[0])
        ]
        single_label_df["id"] = uid

        single_pred_df = pd.DataFrame(single_pred, columns=pred_col_names)
        single_pred_df["id_seqpos"] = [
            f"{uid}_{x}" for x in range(single_pred_df.shape[0])
        ]

        single_df = pd.merge(
            single_label_df, single_pred_df, on="id_seqpos", how="left"
        )
        oof_preds_all.append(single_df)



## === cell 24
submission = pd.concat([sample_sub[["id_seqpos"]], submission[target_cols]], axis=1)

for c in target_cols:
    submission[c] = submission[c].astype(np.float32)

submission = sample_sub[["id_seqpos"]].merge(submission, on="id_seqpos", how="left")
for c in target_cols:
    submission[c] = submission[c].fillna(0.0).astype(np.float32)

submission = submission[["id_seqpos"] + target_cols]

print(submission.head())
print("submission shape:", submission.shape)



## === cell 25
if len(oof_preds_all) > 0:
    OOF = pd.concat(oof_preds_all, ignore_index=True)
    OOF = (
        OOF.groupby(["id_seqpos", "id"], sort=False)
        .mean(numeric_only=True)
        .reset_index()
    )

    OOF_score = MCRMSE(
        np.expand_dims(OOF[target_eval_col].values, axis=0),
        np.expand_dims(OOF[pred_eval_col].values, axis=0),
    ).numpy()[0]
    print("Overall OOF Score :", float(OOF_score))

    OOF.to_csv("OOf.csv", index=False)
else:
    OOF = None
    OOF_score = np.nan
    print("OOF skipped: no OOF objects collected.")



## === cell 26
if OOF is not None:
    OOF_filter_1 = pd.merge(train[["SN_filter", "id"]], OOF, on="id")
    OOF_filter_1 = OOF_filter_1[OOF_filter_1["SN_filter"] == 1]

    OOF_filter_1_score = MCRMSE(
        np.expand_dims(OOF_filter_1[target_eval_col].values, axis=0),
        np.expand_dims(OOF_filter_1[pred_eval_col].values, axis=0),
    ).numpy()[0]
    print("OOF_public Score :", float(OOF_filter_1_score))
    OOF_filter_1.to_csv("OOF_filter_1.csv", index=False)
else:
    OOF_filter_1_score = np.nan



## === cell 27
if OOF is not None:
    OOF_filter_0 = pd.merge(train[["SN_filter", "id"]], OOF, on="id")
    OOF_filter_0 = OOF_filter_0[OOF_filter_0["SN_filter"] == 0]

    OOF_filter_0_score = MCRMSE(
        np.expand_dims(OOF_filter_0[target_eval_col].values, axis=0),
        np.expand_dims(OOF_filter_0[pred_eval_col].values, axis=0),
    ).numpy()[0]
    print("OOF_filter_0 Score :", float(OOF_filter_0_score))
    OOF_filter_0.to_csv("OOF_filter_0.csv", index=False)
else:
    OOF_filter_0_score = np.nan



## === cell 28
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)



## === cell 29
print(
    "|No|OOF_score|OOF_filter_1|OOF_filter_0|LB|n_folds|Window_features|cat_feature|num_features|epochs|BATCH_SIZE|"
)
print("|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|")
print(
    f"|-|{float(OOF_score) if np.isfinite(OOF_score) else float('nan'):.6f}|"
    f"{float(OOF_filter_1_score) if np.isfinite(OOF_filter_1_score) else float('nan'):.6f}|"
    f"{float(OOF_filter_0_score) if np.isfinite(OOF_filter_0_score) else float('nan'):.6f}|"
    f"-|{n_folds}|{Window_features}|{cat_feature}|{num_features}|{epochs}|{BATCH_SIZE}|"
)

## --- ERROR in outputing the csv:
Invalid submission: Expected submission to be the same length as answers, but got 43 instead of 25680.
