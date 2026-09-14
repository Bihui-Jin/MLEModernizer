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
import os, math, json, warnings

warnings.filterwarnings("ignore")

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.layers as L

from itertools import combinations_with_replacement
from sklearn.model_selection import GroupKFold

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
try:
    import seaborn as sns
except Exception:
    sns = None
try:
    from colorama import Fore, Style
except Exception:

    class _Dummy:
        YELLOW = ""
        RESET_ALL = ""

    Fore = _Dummy()
    Style = _Dummy()

HAS_TFA = False
tfa = None

try:
    from keras.utils import plot_model
except Exception:
    plot_model = None

print("TensorFlow:", tf.__version__, "| TFA:", HAS_TFA)



## === cell 1
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
embed_dim = 250
num_hidden_units = 8  # kept for config compatibility (not used in model definition)

Cosine_Schedule = True
Rampup_decy_lr = False




## === cell 2
def seed_everything(seed=1234):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"


seed_everything(SEED)



## === cell 3
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
    "U-G",
    "C-G",
    "U-A",
    "G-C",
    "A-U",
    "G-U",
    "pair_map",
    "pair_distance",
]
num_features = len(numerical_features)

feature_cols = categorical_features + numerical_features
pred_col_names = ["pred_" + c_name for c_name in target_cols]

target_eval_col = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
pred_eval_col = ["pred_" + c_name for c_name in target_eval_col]



## === cell 4
data_dir = "/kaggle/input/stanford-covid-vaccine/"

train = pd.read_json(os.path.join(data_dir, "train.json"), lines=True)
test = pd.read_json(os.path.join(data_dir, "test.json"), lines=True)
sample_sub = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))

print("train:", train.shape, "| test:", test.shape, "| sample_sub:", sample_sub.shape)
print("train cols:", list(train.columns)[:10], "...")
print("test cols:", list(test.columns))



## === cell 5
train = train[train["signal_to_noise"] >= 0.5].reset_index(drop=True)
print("train after SNR filter:", train.shape)

if "cnt" not in train.columns:
    train["cnt"] = 1




## === cell 6
def _make_engineered_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    seqs = out["sequence"].astype(str).values
    Ls = out["seq_length"].astype(int).values

    def frac(ch):
        return np.array([s.count(ch) / max(1, len(s)) for s in seqs], dtype=np.float32)

    out["A_percent"] = frac("A")
    out["C_percent"] = frac("C")
    out["G_percent"] = frac("G")
    out["U_percent"] = frac("U")

    structures = out["structure"].astype(str).values
    out["stems"] = np.array(
        [s.count("(") + s.count(")") for s in structures], dtype=np.float32
    ) / np.maximum(1, Ls)
    out["interior_loops"] = np.array(
        [s.count(".") for s in structures], dtype=np.float32
    ) / np.maximum(1, Ls)

    loops = out["predicted_loop_type"].astype(str).values
    out["multiloops"] = np.array(
        [lt.count("M") for lt in loops], dtype=np.float32
    ) / np.maximum(1, Ls)

    out["pair_map"] = 0.0
    out["pair_distance"] = 0.0
    out["BPPS_Max"] = 0.0
    out["BPPS_nb"] = 0.0
    out["BPPS_sum"] = 0.0
    out["positional_entropy"] = 0.0

    def dinuc_frac(dn):
        return np.array(
            [s.count(dn) / max(1, len(s) - 1) for s in seqs], dtype=np.float32
        )

    for dn in ["UG", "CG", "UA", "GC", "AU", "GU"]:
        out[dn[0] + "-" + dn[1]] = dinuc_frac(dn)

    for c in numerical_features:
        if c not in out.columns:
            out[c] = 0.0

    for c in numerical_features:
        if not isinstance(out.iloc[0][c], (list, np.ndarray)):
            out[c] = [
                ([float(v)] * int(sl))
                for v, sl in zip(
                    out[c].astype(float).values, out["seq_length"].astype(int).values
                )
            ]
        else:
            fixed = []
            for v, sl in zip(out[c].values, out["seq_length"].astype(int).values):
                arr = list(v)
                if len(arr) < sl:
                    arr = arr + [float(arr[-1]) if len(arr) else 0.0] * (sl - len(arr))
                elif len(arr) > sl:
                    arr = arr[:sl]
                fixed.append(arr)
            out[c] = fixed

    return out


train = _make_engineered_numeric_features(train)
test = _make_engineered_numeric_features(test)

print("Engineered numeric features ready.")



## === cell 7


def _pair_feature_list(seq_list):
    L = len(seq_list)
    out = [""] * L
    prev = "_"
    for i in range(L - 1):
        cur = seq_list[i]
        nxt = seq_list[i + 1]
        out[i] = prev + cur
        prev = cur
    out[L - 1] = prev + "_"
    out = ["_"] + seq_list
    out2 = seq_list[1:] + ["_"]
    return [a + b for a, b in zip(out, out2)]


def _tokens_to_ids(tokens, token2int, unk_id):
    return np.fromiter(
        (token2int.get(t, unk_id) for t in tokens), dtype=np.int32, count=len(tokens)
    )


def preprocess_categorical_inputs_fast(df, cols=None, Window_features=Window_features):
    if cols is None:
        cols = list(categorical_features)
    else:
        cols = list(cols)

    unk = token2int.get("<UNK>", 0)

    base_cols = []
    for c in cols:
        base_cols.append([list(s) for s in df[c].astype(str).values])

    if Window_features:
        for c in window_columns:
            base_cols.append(
                [_pair_feature_list(list(s)) for s in df[c].astype(str).values]
            )

    n = len(df)
    seq_len = int(df["seq_length"].iloc[0]) if "seq_length" in df.columns and n else 107
    n_feat = len(base_cols)

    out = np.empty((n, seq_len, n_feat), dtype=np.int32)
    for j, col_list in enumerate(base_cols):
        for i in range(n):
            out[i, :, j] = _tokens_to_ids(col_list[i], token2int, unk)
    return out


def preprocess_numerical_inputs(df, cols=numerical_features):
    arr = np.array(df[cols].values.tolist())
    return np.transpose(arr, (0, 2, 1)).astype(np.float32)




## === cell 8
token_list = (
    list("().ACGU") + list("SMIBHEX") + list("_")
)  # base tokens + padding/unknown
token_list += list("bshftim")  # keep original extra tokens for compatibility

if Window_features:
    comb = combinations_with_replacement(token_list, 2)
    token_list += list(set(list(map("".join, comb))))

token_list = list(set(token_list))
if "<UNK>" not in token_list:
    token_list.append("<UNK>")

token2int = {x: i for i, x in enumerate(token_list)}
print("token_list size:", len(token_list))

train_inputs_all_cat = preprocess_categorical_inputs_fast(
    train, cols=categorical_features
)
train_inputs_all_num = preprocess_numerical_inputs(train, cols=numerical_features)
train_labels_all = np.array(
    train[target_cols].values.tolist(), dtype=np.float32
).transpose((0, 2, 1))

print("Train categorical:", train_inputs_all_cat.shape)
print("Train numerical:", train_inputs_all_num.shape)
print("Train labels:", train_labels_all.shape)



## === cell 9
public_df = test.query("seq_length == 107").reset_index(drop=True)
private_df = test.query("seq_length != 107").reset_index(drop=True)
print("public_df:", public_df.shape)
print("private_df:", private_df.shape)

public_inputs_cat = preprocess_categorical_inputs_fast(
    public_df, cols=categorical_features
)
public_inputs_num = preprocess_numerical_inputs(public_df, cols=numerical_features)

if len(private_df):
    private_inputs_cat = preprocess_categorical_inputs_fast(
        private_df, cols=categorical_features
    )
    private_inputs_num = preprocess_numerical_inputs(
        private_df, cols=numerical_features
    )
else:
    private_inputs_cat = None
    private_inputs_num = None

print("Public cat/num:", public_inputs_cat.shape, public_inputs_num.shape)
if len(private_df):
    print("Private cat/num:", private_inputs_cat.shape, private_inputs_num.shape)




## === cell 10
def MCRMSE(y_true, y_pred):
    colwise_mse = tf.reduce_mean(tf.square(y_true[:, :, :3] - y_pred[:, :, :3]), axis=1)
    return tf.reduce_mean(tf.sqrt(colwise_mse), axis=1)




## === cell 11
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

    return tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)


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




## === cell 12
def lstm_layer(hidden_dim, dropout):
    return L.Bidirectional(
        L.LSTM(
            hidden_dim,
            dropout=dropout,
            return_sequences=True,
            kernel_initializer="orthogonal",
        )
    )


def gru_layer(hidden_dim, dropout):
    return L.Bidirectional(
        L.GRU(
            hidden_dim,
            dropout=dropout,
            return_sequences=True,
            kernel_initializer="orthogonal",
        )
    )




## === cell 13
def build_model(
    embed_size,
    seq_len=107,
    pred_len=68,
    dropout=dropout,
    sp_dropout=sp_dropout,
    num_features=num_features,
    num_hidden_units=num_hidden_units,  # kept (unused) to preserve signature
    embed_dim=embed_dim,
    layers=layers,
    hidden_dim=hidden_dim,
    n_layers=n_layers,
    cat_feature=cat_feature,
):
    inputs = L.Input(shape=(seq_len, cat_feature), name="category_input")
    embed = L.Embedding(input_dim=embed_size, output_dim=embed_dim)(inputs)

    reshaped = L.Reshape((seq_len, cat_feature * embed_dim))(embed)
    reshaped = L.SpatialDropout1D(sp_dropout)(reshaped)
    reshaped_conv = L.Conv1D(
        filters=512, kernel_size=3, strides=1, padding="same", activation="elu"
    )(reshaped)

    numerical_input = L.Input(shape=(seq_len, num_features), name="numeric_input")
    hidden = L.Concatenate()([reshaped_conv, numerical_input])

    hidden_1 = L.Conv1D(
        filters=256, kernel_size=4, strides=1, padding="same", activation="elu"
    )(hidden)
    hidden = gru_layer(128, 0.5)(hidden_1)
    hidden = L.Concatenate()([hidden, hidden_1])

    for x in range(n_layers):
        if layers[x] == "GRU":
            hidden = gru_layer(hidden_dim[x], dropout[x])(hidden)
        else:
            hidden = lstm_layer(hidden_dim[x], dropout[x])(hidden)
        hidden = L.Concatenate()([hidden, hidden_1])

    truncated = hidden[:, :pred_len]
    out = L.Dense(5)(truncated)

    model = tf.keras.Model(inputs=[inputs, numerical_input], outputs=out)

    opt = tf.keras.optimizers.Adam()
    model.compile(optimizer=opt, loss=MCRMSE)
    return model




## === cell 14
model = build_model(embed_size=len(token_list))
print(model.count_params())
if plot_model is not None:
    try:
        plot_model(
            model, to_file="model_plot.png", show_shapes=True, show_layer_names=True
        )
        print("Saved model_plot.png")
    except Exception as e:
        print("plot_model failed (non-fatal):", repr(e))




## === cell 15
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

gkf = GroupKFold(n_splits=n_folds)



## === cell 16

id_seqpos_to_row = {k: i for i, k in enumerate(sample_sub["id_seqpos"].values)}
sub_sum = np.zeros((len(sample_sub), len(target_cols)), dtype=np.float32)

oof_chunks = []
stacking_chunks = []

val_losses = []


def _write_preds_to_sub_sum(preds, ids, sub_sum):
    for i, uid in enumerate(ids):
        base = f"{uid}_"
        for pos in range(preds.shape[1]):
            key = base + str(pos)
            row = id_seqpos_to_row.get(key, None)
            if row is not None:
                sub_sum[row, :] += preds[i, pos, :]


for Fold, (train_index, val_index) in enumerate(
    gkf.split(train_inputs_all_cat, groups=train["stratify_group"])
):
    print(Fore.YELLOW + "#" * 45)
    print(f"###  Fold : {Fold+1}")
    print("#" * 45 + Style.RESET_ALL)

    tr_cat = train_inputs_all_cat[train_index]
    tr_num = train_inputs_all_num[train_index]
    tr_y = train_labels_all[train_index]

    val_index_raw = val_index
    val_mask = train.loc[val_index_raw, "cnt"].values == 1
    val_index2 = val_index_raw[val_mask]

    va_cat = train_inputs_all_cat[val_index2]
    va_num = train_inputs_all_num[val_index2]
    va_y = train_labels_all[val_index2]

    model_train = build_model(embed_size=len(token_list))
    model_short = build_model(embed_size=len(token_list), seq_len=107, pred_len=107)

    model_long = None
    if len(private_df):
        model_long = build_model(
            embed_size=len(token_list),
            seq_len=int(private_df["seq_length"].iloc[0]),
            pred_len=int(private_df["seq_length"].iloc[0]),
        )

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
    elif Rampup_decy_lr:
        lr_schedule = get_lr_callback(BATCH_SIZE)
    else:
        lr_schedule = tf.keras.callbacks.ReduceLROnPlateau()

    history = model_train.fit(
        {"numeric_input": tr_num, "category_input": tr_cat},
        tr_y,
        validation_data=(
            {"numeric_input": va_num, "category_input": va_cat},
            va_y,
        ),
        batch_size=BATCH_SIZE,
        epochs=epochs,
        callbacks=[lr_schedule, checkpoint, csv_logger],
        verbose=1 if debug else 0,
    )

    min_v = float(np.min(history.history["val_loss"]))
    val_losses.append(min_v)
    print(
        "Min Validation Loss:",
        min_v,
        "| at epoch",
        int(np.argmin(history.history["val_loss"]) + 1),
    )

    model_short.load_weights(f"{model_name}_Fold_{Fold}.weights.h5")
    if model_long is not None:
        model_long.load_weights(f"{model_name}_Fold_{Fold}.weights.h5")

    public_preds = model_short.predict(
        {"numeric_input": public_inputs_num, "category_input": public_inputs_cat},
        verbose=0,
    )
    private_preds = None
    if model_long is not None and private_inputs_num is not None:
        private_preds = model_long.predict(
            {"numeric_input": private_inputs_num, "category_input": private_inputs_cat},
            verbose=0,
        )

    _write_preds_to_sub_sum(public_preds, public_df.id.values, sub_sum)
    if private_preds is not None:
        _write_preds_to_sub_sum(private_preds, private_df.id.values, sub_sum)

    oof_preds = model_train.predict(
        {"numeric_input": va_num, "category_input": va_cat}, verbose=0
    )
    stacking_pred = model_short.predict(
        {"numeric_input": va_num, "category_input": va_cat}, verbose=0
    )

    val_ids = train.loc[val_index2, "id"].values
    seq_len_oof = oof_preds.shape[1]
    id_seqpos = np.concatenate(
        [[f"{uid}_{p}" for p in range(seq_len_oof)] for uid in val_ids]
    )
    ids_rep = np.repeat(val_ids, seq_len_oof)
    s_id = np.tile(np.arange(seq_len_oof), len(val_ids))

    y_true_flat = va_y.reshape(-1, va_y.shape[-1])
    y_pred_flat = oof_preds.reshape(-1, oof_preds.shape[-1])
    stack_flat = stacking_pred.reshape(-1, stacking_pred.shape[-1])

    oof_df = pd.DataFrame(y_true_flat, columns=target_cols)
    oof_df["id_seqpos"] = id_seqpos
    oof_df["id"] = ids_rep
    oof_df["s_id"] = s_id
    for j, c in enumerate(pred_col_names):
        oof_df[c] = y_pred_flat[:, j]
    oof_chunks.append(oof_df)

    stack_df = pd.DataFrame(stack_flat, columns=pred_col_names)
    stack_df["id_seqpos"] = id_seqpos
    stack_df["id"] = ids_rep
    stacking_chunks.append(stack_df)



## === cell 17
sub_avg = sub_sum / float(n_folds)
submission = pd.DataFrame(sub_avg, columns=target_cols)
submission["id_seqpos"] = sample_sub["id_seqpos"].values
submission = submission[["id_seqpos"] + target_cols]
submission[target_cols] = submission[target_cols].fillna(0.0)

OOF = pd.concat(oof_chunks, axis=0) if len(oof_chunks) else pd.DataFrame()
stacking_df = (
    pd.concat(stacking_chunks, axis=0) if len(stacking_chunks) else pd.DataFrame()
)

print("submission shape:", submission.shape)
print(submission.head())



## === cell 18
if len(OOF):
    OOF = OOF.groupby(["id_seqpos", "id", "s_id"], as_index=False).mean()
    OOF = OOF.sort_values(["id", "s_id"], ascending=[True, True])

    OOF_score = MCRMSE(
        np.expand_dims(OOF[target_eval_col].values, axis=0),
        np.expand_dims(OOF[pred_eval_col].values, axis=0),
    ).numpy()[0]
    print("Overall OOF Score:", float(OOF_score))
    OOF.to_csv("OOf.csv", index=False)

    OOF_filter_1 = pd.merge(train[["SN_filter", "id"]], OOF, on="id")
    OOF_filter_1 = OOF_filter_1[OOF_filter_1["SN_filter"] == 1].sort_values(
        ["id", "s_id"]
    )
    OOF_filter_1_score = MCRMSE(
        np.expand_dims(OOF_filter_1[target_eval_col].values, axis=0),
        np.expand_dims(OOF_filter_1[pred_eval_col].values, axis=0),
    ).numpy()[0]
    print("OOF SN_filter==1 Score:", float(OOF_filter_1_score))
    OOF_filter_1.to_csv("OOF_filter_1.csv", index=False)

    OOF_filter_0 = pd.merge(train[["SN_filter", "id"]], OOF, on="id")
    OOF_filter_0 = OOF_filter_0[OOF_filter_0["SN_filter"] == 0].sort_values(
        ["id", "s_id"]
    )
    OOF_filter_0_score = MCRMSE(
        np.expand_dims(OOF_filter_0[target_eval_col].values, axis=0),
        np.expand_dims(OOF_filter_0[pred_eval_col].values, axis=0),
    ).numpy()[0]
    print("OOF SN_filter==0 Score:", float(OOF_filter_0_score))
    OOF_filter_0.to_csv("OOF_filter_0.csv", index=False)

    if len(stacking_df):
        stacking_df.to_csv("stacking.csv", index=False)



## === cell 19
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())

## --- ERROR in outputing the csv:
Invalid submission: Expected submission to be the same length as answers, but got 22 instead of 25680.
