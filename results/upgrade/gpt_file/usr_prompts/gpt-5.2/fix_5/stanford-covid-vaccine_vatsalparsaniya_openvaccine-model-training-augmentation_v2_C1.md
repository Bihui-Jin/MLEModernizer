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
import os, math
import numpy as np
import pandas as pd

plt = None
sns = None

import tensorflow as tf
import tensorflow.keras.layers as L

import warnings

warnings.filterwarnings("ignore")

TFA_AVAILABLE = False
tfa = None
PLOT_MODEL_AVAILABLE = False

try:
    from itertools import combinations_with_replacement
    from sklearn.model_selection import KFold, StratifiedKFold
except Exception as e:
    raise RuntimeError("Required sklearn/itertools import failed: " + repr(e))

try:
    from colorama import Fore, Back, Style
except Exception:

    class _NoColor:
        def __getattr__(self, name):
            return ""

    Fore = Back = Style = _NoColor()

print("TF version:", tf.__version__)
print("Num GPUs available:", len(tf.config.list_physical_devices("GPU")))



## === cell 1
SEED = 53

n_folds = 5
debug = False  # score-neutral: reduces notebook log overhead; does not change training procedure/epochs

Window_features = False

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



## === cell 2
pass




## === cell 3
def seed_everything(seed=1234):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"


seed_everything(SEED)



## === cell 4
pass



## === cell 5
target_cols = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C", "deg_pH10", "deg_50C"]
window_columns = ["sequence", "structure", "predicted_loop_type"]

categorical_features = ["sequence", "structure", "predicted_loop_type"]

cat_feature = len(categorical_features)
if Window_features:
    cat_feature += len(window_columns)

numerical_features = ["BPPS_Max", "BPPS_nb", "BPPS_sum"]
num_features = len(numerical_features)

feature_cols = categorical_features + numerical_features
pred_col_names = ["pred_" + c_name for c_name in target_cols]

target_eval_col = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
pred_eval_col = ["pred_" + c_name for c_name in target_eval_col]



## === cell 6
pass



## === cell 7
DATA_DIR_CANDIDATES = [
    "/kaggle/input/stanford-covid-vaccine/",
    "/kaggle/input/stanford-covid-vaccine/stanford-covid-vaccine/",
    "/kaggle/data/stanford-covid-vaccine/",
    "/kaggle/data/stanford-covid-vaccine/stanford-covid-vaccine/",
    "/kaggle/input/",
    "/kaggle/data/",
]

data_dir = None
for p in DATA_DIR_CANDIDATES:
    if os.path.exists(os.path.join(p, "train.json")) and os.path.exists(
        os.path.join(p, "test.json")
    ):
        data_dir = p
        break
if data_dir is None:
    raise FileNotFoundError(
        "Could not find train.json/test.json under expected Kaggle input paths."
    )

train = pd.read_json(os.path.join(data_dir, "train.json"), lines=True)
test = pd.read_json(os.path.join(data_dir, "test.json"), lines=True)
sample_sub = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))

print("Data dir:", data_dir)
print(
    "train shape:",
    train.shape,
    "test shape:",
    test.shape,
    "sample_sub shape:",
    sample_sub.shape,
)
print("sample_sub columns:", list(sample_sub.columns))




## === cell 8
def _ensure_bpps_features(df, seq_len_col="seq_length"):
    Ls = df[seq_len_col].astype(int).to_numpy()
    for col in numerical_features:
        if col not in df.columns:
            df[col] = [[0.0] * L_ for L_ in Ls]
        else:
            fixed = []
            vals = df[col].values
            for x, L_ in zip(vals, Ls):
                if isinstance(x, (list, np.ndarray)) and len(x) == L_:
                    fixed.append(list(x))
                else:
                    fixed.append([0.0] * L_)
            df[col] = fixed
    return df


train = _ensure_bpps_features(train)
test = _ensure_bpps_features(test)




## === cell 9
def pair_feature(row):
    arr = list(row)
    its = [iter(["_"] + arr[:]), iter(arr[1:] + ["_"])]
    list_touple = list(zip(*its))
    return list(map("".join, list_touple))




## === cell 10
def preprocess_categorical_inputs(
    df, cols=categorical_features, Window_features=Window_features
):
    cols_local = list(cols)
    if Window_features:
        for c in window_columns:
            df["pair_" + c] = df[c].apply(pair_feature)
            cols_local.append("pair_" + c)
    cols_local = list(dict.fromkeys(cols_local))

    def _encode_seq(seq):
        return [token2int.get(x, 0) for x in seq]

    encoded_cols = []
    for c in cols_local:
        encoded_cols.append(df[c].map(_encode_seq).to_list())
    arr = np.stack([np.asarray(col, dtype=np.int32) for col in encoded_cols], axis=-1)
    return arr




## === cell 11
def preprocess_numerical_inputs(df, cols=numerical_features):
    vals = [
        np.asarray(v, dtype=np.float32) for v in df[cols].values.tolist()
    ]  # (N, C) each is list length seq_len
    arr = np.asarray(vals, dtype=np.float32)
    return np.transpose(arr, (0, 2, 1))




## === cell 12
def _collect_tokens_from_df(df, cols):
    s = set()
    for c in cols:
        for seq in df[c].astype(str).values:
            s.update(list(seq))
    return s


base_tokens = set()
base_tokens |= _collect_tokens_from_df(train, categorical_features)
base_tokens |= _collect_tokens_from_df(test, categorical_features)

base_tokens.add("_")

token_list = sorted(list(base_tokens))

if Window_features:
    comb = combinations_with_replacement(list("_" + "".join(token_list)) * 2, 2)
    token_list += list(set(list(map("".join, comb))))
    token_list = sorted(list(set(token_list)))

if "_" in token_list:
    token_list = ["_"] + [t for t in token_list if t != "_"]

token2int = {x: i for i, x in enumerate(token_list)}
print("token_list Size :", len(token_list))
print("Example tokens:", token_list[:20])

train_inputs_all_cat = preprocess_categorical_inputs(train, cols=categorical_features)
train_inputs_all_num = preprocess_numerical_inputs(train, cols=numerical_features)

train_labels_all = np.array(
    train[target_cols].values.tolist(), dtype=np.float32
).transpose((0, 2, 1))

print("Train categorical Features Shape : ", train_inputs_all_cat.shape)
print("Train numerical Features Shape : ", train_inputs_all_num.shape)
print("Train labels Shape : ", train_labels_all.shape)



## === cell 13
pass



## === cell 14
pass



## === cell 15
pass



## === cell 16
public_df = test.query("seq_length == 107").copy()
private_df = test.query("seq_length == 130").copy()
print("public_df : ", public_df.shape)
print("private_df : ", private_df.shape)

public_inputs_cat = preprocess_categorical_inputs(public_df)
public_inputs_num = preprocess_numerical_inputs(public_df, cols=numerical_features)

if len(private_df) > 0:
    private_inputs_cat = preprocess_categorical_inputs(private_df)
    private_inputs_num = preprocess_numerical_inputs(
        private_df, cols=numerical_features
    )
else:
    private_inputs_cat = np.zeros((0, 130, cat_feature), dtype=np.int32)
    private_inputs_num = np.zeros((0, 130, num_features), dtype=np.float32)

print("Public categorical Features Shape : ", public_inputs_cat.shape)
print("Public numerical Features Shape : ", public_inputs_num.shape)
print("Private categorical Features Shape : ", private_inputs_cat.shape)
print("Private numerical Features Shape : ", private_inputs_num.shape)



## === cell 17
pass




## === cell 18
def MCRMSE(y_true, y_pred):
    colwise_mse = tf.reduce_mean(tf.square(y_true[:, :, :3] - y_pred[:, :, :3]), axis=1)
    return tf.reduce_mean(tf.sqrt(colwise_mse), axis=1)




## === cell 19
pass




## === cell 20
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




## === cell 21
pass




## === cell 22
def get_cosine_schedule_with_warmup(
    lr, num_warmup_steps, num_training_steps, num_cycles=3.5
):
    def lrfn(epoch):
        if epoch < num_warmup_steps:
            return (float(epoch) / float(max(1, num_warmup_steps))) * lr
        progress = float(epoch - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return (
            max(
                0.0,
                0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)),
            )
            * lr
        )

    return tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)




## === cell 23
pass




## === cell 24
def lstm_layer(hidden_dim, dropout):
    return tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            hidden_dim,
            dropout=dropout,
            return_sequences=True,
            kernel_initializer="orthogonal",
        )
    )




## === cell 25
def gru_layer(hidden_dim, dropout):
    return L.Bidirectional(
        L.GRU(
            hidden_dim,
            dropout=dropout,
            return_sequences=True,
            kernel_initializer="orthogonal",
        )
    )




## === cell 26
pass




## === cell 27
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

    reshaped_conv = tf.keras.layers.Conv1D(
        filters=512, kernel_size=3, strides=1, padding="same", activation="elu"
    )(reshaped)

    numerical_input = L.Input(shape=(seq_len, num_features), name="numeric_input")
    n_Dense_1 = L.Dense(64)(numerical_input)
    n_Dense_2 = L.Dense(128)(n_Dense_1)
    numerical_conv = tf.keras.layers.Conv1D(
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

    if TFA_AVAILABLE:
        optimizer = tfa.optimizers.RectifiedAdam()
    else:
        optimizer = tf.keras.optimizers.Adam()

    model.compile(optimizer=optimizer, loss=MCRMSE)
    return model




## === cell 28
pass



## === cell 29
model = build_model(embed_size=len(token_list))
print(model.summary())



## === cell 30
pass




## === cell 31
def get_stratify_group(row):
    snf = row["SN_filter"]
    snr = row["signal_to_noise"]

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
        elif snr >= 10:
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
        elif 8 <= snr < 9:
            snr_c = 15
        elif 9 <= snr < 10:
            snr_c = 15
        elif snr >= 10:
            snr_c = 16

    return "{}".format(snr_c)


train["stratify_group"] = train.apply(get_stratify_group, axis=1)
train["stratify_group"] = train["stratify_group"].astype("category").cat.codes

skf = StratifiedKFold(n_folds, shuffle=True, random_state=SEED)

print("StratifiedKFold prepared with n_folds =", n_folds)




## === cell 32
def build_id_seqpos_for_ids(ids, seq_len):
    ids = np.asarray(ids, dtype=object)
    suffix = np.asarray([f"_{i}" for i in range(seq_len)], dtype=object)
    out = np.empty(ids.size * seq_len, dtype=object)
    k = 0
    for uid in ids:
        out[k : k + seq_len] = uid + suffix
        k += seq_len
    return out


public_id_seqpos = build_id_seqpos_for_ids(public_df.id.values, 107)
private_id_seqpos = (
    build_id_seqpos_for_ids(private_df.id.values, 130)
    if len(private_df) > 0
    else np.empty((0,), dtype=object)
)

sample_id_seqpos = sample_sub["id_seqpos"].values.astype(object)
id2subidx = {k: i for i, k in enumerate(sample_id_seqpos)}

public_sub_idx = np.fromiter(
    (id2subidx[x] for x in public_id_seqpos),
    count=public_id_seqpos.size,
    dtype=np.int32,
)
private_sub_idx = (
    np.fromiter(
        (id2subidx[x] for x in private_id_seqpos),
        count=private_id_seqpos.size,
        dtype=np.int32,
    )
    if private_id_seqpos.size
    else np.empty((0,), dtype=np.int32)
)

oof_id = train["id"].values.astype(object)



## === cell 33
submission = np.zeros((sample_sub.shape[0], len(target_cols)), dtype=np.float32)

val_losses = []
historys = []

oof_store = []

skf = StratifiedKFold(n_folds, shuffle=True, random_state=SEED)

for Fold, (train_index, val_index) in enumerate(
    skf.split(train_inputs_all_cat, train["stratify_group"])
):
    print(Fore.YELLOW)
    print("#" * 45)
    print("###  Fold : ", str(Fold + 1))
    print("#" * 45)
    print(Style.RESET_ALL)

    model_train = build_model(embed_size=len(token_list))
    model_short = build_model(embed_size=len(token_list), seq_len=107, pred_len=107)
    model_long = build_model(embed_size=len(token_list), seq_len=130, pred_len=130)

    train_inputs_cat, train_labels = (
        train_inputs_all_cat[train_index],
        train_labels_all[train_index],
    )
    val_inputs_cat, val_labels = (
        train_inputs_all_cat[val_index],
        train_labels_all[val_index],
    )
    train_inputs_num, val_inputs_num = (
        train_inputs_all_num[train_index],
        train_inputs_all_num[val_index],
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
        {"numeric_input": train_inputs_num, "category_input": train_inputs_cat},
        train_labels,
        validation_data=(
            {"numeric_input": val_inputs_num, "category_input": val_inputs_cat},
            val_labels,
        ),
        batch_size=BATCH_SIZE,
        epochs=epochs,
        callbacks=[lr_schedule, checkpoint, csv_logger],
        verbose=1 if debug else 0,
    )

    val_losses.append(float(np.min(history.history["val_loss"])))
    historys.append(history)

    model_short.load_weights(f"{model_name}_Fold_{Fold}.weights.h5")
    model_long.load_weights(f"{model_name}_Fold_{Fold}.weights.h5")

    public_preds = model_short.predict(
        {"numeric_input": public_inputs_num, "category_input": public_inputs_cat},
        verbose=0,
    ).astype(np.float32)

    if len(private_df) > 0:
        private_preds = model_long.predict(
            {"numeric_input": private_inputs_num, "category_input": private_inputs_cat},
            verbose=0,
        ).astype(np.float32)
    else:
        private_preds = np.zeros((0, 130, 5), dtype=np.float32)

    oof_preds = model_train.predict(
        {"numeric_input": val_inputs_num, "category_input": val_inputs_cat},
        verbose=0,
    ).astype(np.float32)

    pub_flat = public_preds.reshape(-1, 5)
    submission[public_sub_idx] += pub_flat / n_folds

    if private_preds.shape[0] > 0:
        prv_flat = private_preds.reshape(-1, 5)
        submission[private_sub_idx] += prv_flat / n_folds

    val_ids = oof_id[val_index]
    id_seqpos = build_id_seqpos_for_ids(val_ids, val_labels.shape[1])  # pred_len=68
    oof_store.append(
        {
            "id_seqpos": id_seqpos,
            "id": np.repeat(val_ids, val_labels.shape[1]),
            "y": val_labels.reshape(-1, 5),
            "p": oof_preds.reshape(-1, 5),
        }
    )



## === cell 34
pass



## === cell 35
pass



## === cell 36
pass



## === cell 37
if len(oof_store) == 0:
    raise RuntimeError(
        "OOF predictions list is empty; training loop did not append any fold predictions."
    )

oof_id_seqpos = np.concatenate([d["id_seqpos"] for d in oof_store], axis=0)
oof_id_col = np.concatenate([d["id"] for d in oof_store], axis=0)
oof_y = np.concatenate([d["y"] for d in oof_store], axis=0)
oof_p = np.concatenate([d["p"] for d in oof_store], axis=0)

OOF = pd.DataFrame(
    {
        "id_seqpos": oof_id_seqpos,
        "id": oof_id_col,
        **{c: oof_y[:, i] for i, c in enumerate(target_cols)},
        **{c: oof_p[:, i] for i, c in enumerate(pred_col_names)},
    }
)

OOF_mean = OOF.groupby(["id_seqpos", "id"]).mean(numeric_only=True)
OOF_max = OOF.groupby(["id_seqpos", "id"]).max(numeric_only=True)
OOF_min = OOF.groupby(["id_seqpos", "id"]).min(numeric_only=True)



## === cell 38
pass



## === cell 39
OOF_mean_score = MCRMSE(
    np.expand_dims(OOF_mean[target_eval_col].values, axis=0),
    np.expand_dims(OOF_mean[pred_eval_col].values, axis=0),
).numpy()[0]
OOF_max_score = MCRMSE(
    np.expand_dims(OOF_max[target_eval_col].values, axis=0),
    np.expand_dims(OOF_max[pred_eval_col].values, axis=0),
).numpy()[0]
OOF_min_score = MCRMSE(
    np.expand_dims(OOF_min[target_eval_col].values, axis=0),
    np.expand_dims(OOF_min[pred_eval_col].values, axis=0),
).numpy()[0]



## === cell 40
print(OOF_mean.head())



## === cell 41
print("Overall OOF_mean_score  :", float(OOF_mean_score))
print("Overall OOF_max_score  :", float(OOF_max_score))
print("Overall OOF_min_score  :", float(OOF_min_score))
OOF_mean.to_csv("OOF_mean.csv")
OOF_max.to_csv("OOF_max.csv")
OOF_min.to_csv("OOF_min.csv")
OOF.reset_index().to_csv("OOF.csv", index=False)



## === cell 42
pass



## === cell 43
OOF_mean_filter_1 = pd.merge(
    train[["SN_filter", "id"]], OOF_mean.reset_index(), on="id"
)
OOF_mean_filter_1 = OOF_mean_filter_1[OOF_mean_filter_1["SN_filter"] == 1]
OOF_mean_filter_1_score = MCRMSE(
    np.expand_dims(OOF_mean_filter_1[target_eval_col].values, axis=0),
    np.expand_dims(OOF_mean_filter_1[pred_eval_col].values, axis=0),
).numpy()[0]
print("OOF_mean_filter_1 Score :", float(OOF_mean_filter_1_score))
OOF_mean_filter_1.to_csv("OOF_mean_filter_1.csv", index=False)

OOF_max_filter_1 = pd.merge(train[["SN_filter", "id"]], OOF_max.reset_index(), on="id")
OOF_max_filter_1 = OOF_max_filter_1[OOF_max_filter_1["SN_filter"] == 1]
OOF_max_filter_1_score = MCRMSE(
    np.expand_dims(OOF_max_filter_1[target_eval_col].values, axis=0),
    np.expand_dims(OOF_max_filter_1[pred_eval_col].values, axis=0),
).numpy()[0]
print("OOF_max_filter_1 Score :", float(OOF_max_filter_1_score))
OOF_max_filter_1.to_csv("OOF_max_filter_1.csv", index=False)

OOF_min_filter_1 = pd.merge(train[["SN_filter", "id"]], OOF_min.reset_index(), on="id")
OOF_min_filter_1 = OOF_min_filter_1[OOF_min_filter_1["SN_filter"] == 1]
OOF_min_filter_1_score = MCRMSE(
    np.expand_dims(OOF_min_filter_1[target_eval_col].values, axis=0),
    np.expand_dims(OOF_min_filter_1[pred_eval_col].values, axis=0),
).numpy()[0]
print("OOF_min_filter_1 Score :", float(OOF_min_filter_1_score))
OOF_min_filter_1.to_csv("OOF_min_filter_1.csv", index=False)



## === cell 44
pass



## === cell 45
final_sub = pd.DataFrame({"id_seqpos": sample_sub["id_seqpos"].values})
for j, c in enumerate(target_cols):
    final_sub[c] = submission[:, j].astype(np.float32)

final_sub = final_sub[["id_seqpos"] + target_cols]
final_sub.to_csv("submission.csv", index=False)
print(final_sub.head())
print("Wrote submission.csv with shape:", final_sub.shape)



## === cell 46
pass



## === cell 47
print(
    "|No|n_folds|Window_features|cat_feature|num_features|epochs|BATCH_SIZE|OOF_mean_score|OOF_max_score|OOF_min_score|OOF_mean_filter_1_score|OOF_max_filter_1_score|OOF_min_filter_1_score|LB|"
)
print("|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|")
print(
    f"|-|{n_folds}|{Window_features}|{cat_feature}|{num_features}|{epochs}|{BATCH_SIZE}|"
    f"{float(OOF_mean_score):.6f}|{float(OOF_max_score):.6f}|{float(OOF_min_score):.6f}|"
    f"{float(OOF_mean_filter_1_score):.6f}|{float(OOF_max_filter_1_score):.6f}|{float(OOF_min_filter_1_score):.6f}|-|"
)

## --- ERROR in outputing the csv:
Invalid submission: Expected submission to be the same length as answers, but got 43 instead of 25680.
