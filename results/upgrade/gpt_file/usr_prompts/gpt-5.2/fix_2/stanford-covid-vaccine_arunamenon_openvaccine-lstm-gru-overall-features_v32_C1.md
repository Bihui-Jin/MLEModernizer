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

3.9

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

# 5. Target score

0.3763091623635361

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import json
import numpy as np
import pandas as pd

os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(0)



## === cell 1
import keras
from keras import layers, ops



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
DATA_DIR = "/kaggle/input/stanford-covid-vaccine"
WORK_DIR = "/kaggle/working"
os.makedirs(WORK_DIR, exist_ok=True)

train_path = f"{DATA_DIR}/train.json"
test_path = f"{DATA_DIR}/test.json"
sub_path = f"{DATA_DIR}/sample_submission.csv"

train_data = pd.read_json(train_path, lines=True)
test_data = pd.read_json(test_path, lines=True)
submission_format = pd.read_csv(sub_path, encoding="utf-8-sig")

print(train_data.shape, test_data.shape, submission_format.shape)



## === cell 3
token2int = {x: i for i, x in enumerate("().ACGUBEHIMSX")}
target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]




## === cell 4
def _safe_bpps_feature(df, reducer="sum"):
    out = []
    for seq_len in df["seq_length"].to_numpy():
        out.append(np.zeros((seq_len,), dtype=np.float32))
    return out


def _safe_bpps_nb(df):
    out = []
    for seq_len in df["seq_length"].to_numpy():
        out.append(np.zeros((seq_len,), dtype=np.float32))
    return out


train_data["bpps_sum"] = _safe_bpps_feature(train_data, "sum")
test_data["bpps_sum"] = _safe_bpps_feature(test_data, "sum")
train_data["bpps_max"] = _safe_bpps_feature(train_data, "max")
test_data["bpps_max"] = _safe_bpps_feature(test_data, "max")
train_data["bpps_nb"] = _safe_bpps_nb(train_data)
test_data["bpps_nb"] = _safe_bpps_nb(test_data)



## === cell 5
from collections import Counter as count


def get_bases(data):
    bases = []
    for j in range(len(data)):
        counts = dict(count(data.iloc[j]["sequence"]))
        bases.append(
            (
                counts.get("A", 0) / 107,
                counts.get("G", 0) / 107,
                counts.get("C", 0) / 107,
                counts.get("U", 0) / 107,
            )
        )
    return pd.DataFrame(
        bases, columns=["A_percent", "G_percent", "C_percent", "U_percent"]
    )


def get_pairs_rate(data):
    pairs_rate = []
    for j in range(len(data)):
        res = dict(count(data.iloc[j]["structure"]))
        pairs_rate.append(res.get("(", 0) / 53.5)
    return pd.DataFrame(pairs_rate, columns=["pairs_rate"])


def get_pairs(data):
    pairs = []
    for j in range(len(data)):
        pairs_dict = {}
        queue = []
        structure = data.iloc[j]["structure"]
        sequence = data.iloc[j]["sequence"]
        for i in range(len(structure)):
            if structure[i] == "(":
                queue.append(i)
            elif structure[i] == ")":
                first = queue.pop()
                key = (sequence[first], sequence[i])
                pairs_dict[key] = pairs_dict.get(key, 0) + 1

        pairs_num = sum(pairs_dict.values()) if len(pairs_dict) else 0
        pairs_unique = [
            ("U", "G"),
            ("C", "G"),
            ("U", "A"),
            ("G", "C"),
            ("A", "U"),
            ("G", "U"),
        ]
        add_tuple = []
        for item in pairs_unique:
            if pairs_num == 0:
                add_tuple.append(0.0)
            else:
                add_tuple.append(pairs_dict.get(item, 0) / pairs_num)
        pairs.append(add_tuple)
    return pd.DataFrame(pairs, columns=["U-G", "C-G", "U-A", "G-C", "A-U", "G-U"])


def get_loops(data):
    loops = []
    available = ["E", "S", "H", "B", "X", "I", "M"]
    for j in range(len(data)):
        counts = dict(count(data.iloc[j]["predicted_loop_type"]))
        row = []
        for item in available:
            row.append(counts.get(item, 0) / 107)
        loops.append(row)
    return pd.DataFrame(loops, columns=available)




## === cell 6
def get_structure_adj(df):
    Ss = []
    for i in range(len(df)):
        seq_length = int(df["seq_length"].iloc[i])
        structure = df["structure"].iloc[i]
        sequence = df["sequence"].iloc[i]

        cue = []
        a_structures = {
            ("A", "U"): np.zeros([seq_length, seq_length], dtype=np.float32),
            ("C", "G"): np.zeros([seq_length, seq_length], dtype=np.float32),
            ("U", "G"): np.zeros([seq_length, seq_length], dtype=np.float32),
            ("U", "A"): np.zeros([seq_length, seq_length], dtype=np.float32),
            ("G", "C"): np.zeros([seq_length, seq_length], dtype=np.float32),
            ("G", "U"): np.zeros([seq_length, seq_length], dtype=np.float32),
        }

        for j in range(seq_length):
            if structure[j] == "(":
                cue.append(j)
            elif structure[j] == ")":
                start = cue.pop()
                a_structures[(sequence[start], sequence[j])][start, j] = 1.0
                a_structures[(sequence[j], sequence[start])][j, start] = 1.0

        a_strc = np.stack(list(a_structures.values()), axis=2)
        a_strc = np.sum(a_strc, axis=2, keepdims=True)  # (L, L, 1)
        Ss.append(a_strc)

    return np.array(Ss, dtype=np.float32)


def get_distance_matrix(n_samples, seq_len):
    idx = np.arange(seq_len)
    Ds = []
    for i in range(len(idx)):
        d = np.abs(idx[i] - idx)
        Ds.append(d)
    Ds = np.array(Ds, dtype=np.float32) + 1.0
    Ds = 1.0 / Ds  # (L, L)
    Ds = Ds[None, :, :]  # (1, L, L)
    Ds = np.repeat(Ds, n_samples, axis=0)  # (N, L, L)

    Dss = []
    for p in [1, 2, 4]:
        Dss.append(Ds**p)
    Ds = np.stack(Dss, axis=3)  # (N, L, L, 3)
    return Ds.astype(np.float32)




## === cell 7
bases = get_bases(train_data)
pairs = get_pairs(train_data)
loops = get_loops(train_data)
pairs_rate = get_pairs_rate(train_data)
train_data = pd.concat([train_data, bases, pairs, loops, pairs_rate], axis=1)

bases = get_bases(test_data)
pairs = get_pairs(test_data)
loops = get_loops(test_data)
pairs_rate = get_pairs_rate(test_data)
test_data = pd.concat([test_data, bases, pairs, loops, pairs_rate], axis=1)




## === cell 8
def preprocess_inputs(
    df, cols=("sequence", "structure", "predicted_loop_type"), seq_length=107
):
    base_fea = np.transpose(
        np.array(
            df[list(cols)]
            .applymap(lambda seq: [token2int[x] for x in seq])
            .values.tolist(),
            dtype=np.int32,
        ),
        (0, 2, 1),
    )

    bpps_sum_fea = np.array(df["bpps_sum"].to_list(), dtype=np.float32)[
        :, :, np.newaxis
    ]
    bpps_max_fea = np.array(df["bpps_max"].to_list(), dtype=np.float32)[
        :, :, np.newaxis
    ]
    bpps_nb_fea = np.array(df["bpps_nb"].to_list(), dtype=np.float32)[:, :, np.newaxis]

    Ss = get_structure_adj(df)  # (N, L, L, 1)
    Ss = Ss.sum(axis=1)  # sum over source positions -> (N, L, 1)

    Ds = get_distance_matrix(len(df), seq_length)  # (N, L, L, 3)
    Ds = Ds.sum(axis=1)  # (N, L, 3)

    data = np.concatenate(
        [base_fea, bpps_sum_fea, bpps_max_fea, bpps_nb_fea, Ss, Ds], axis=2
    )

    global_cols = [
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
        "E",
        "S",
        "H",
        "B",
        "X",
        "I",
        "M",
        "pairs_rate",
    ]
    for col in global_cols:
        v = df[col].to_numpy(dtype=np.float32)[:, None, None]  # (N,1,1)
        v = np.repeat(v, seq_length, axis=1)  # (N,L,1)
        data = np.concatenate([data, v], axis=2)

    return data.astype(np.float32)




## === cell 9
train_filtered = train_data.loc[train_data["signal_to_noise"] > 1].copy()

train_inputs = preprocess_inputs(train_filtered, seq_length=107)
train_labels = np.array(
    train_filtered[target_cols].values.tolist(), dtype=np.float32
).transpose((0, 2, 1))

print(train_inputs.shape, train_labels.shape)




## === cell 10
def MCRMSE(y_true, y_pred):
    colwise_mse = ops.mean(ops.square(y_true - y_pred), axis=1)
    return ops.mean(ops.sqrt(colwise_mse), axis=1)


def lstm_layer(hidden_dim, dropout):
    return layers.Bidirectional(
        layers.LSTM(
            hidden_dim,
            dropout=dropout,
            return_sequences=True,
            kernel_initializer="orthogonal",
        )
    )


def gru_layer(hidden_dim, dropout):
    return layers.Bidirectional(
        layers.GRU(
            hidden_dim,
            dropout=dropout,
            return_sequences=True,
            kernel_initializer="orthogonal",
        )
    )


def build_model(
    n_layers=2,
    seq_len=107,
    num_features=28,
    embed_dim=200,
    sp_dropout=0.2,
    hidden_dim=512,
    dropout=0.5,
    pred_len=68,
    gru_flag=False,
):
    inputs = layers.Input(shape=(seq_len, num_features))

    categorical_feats = inputs[:, :, :3]
    numerical_feats = inputs[:, :, 3:10]
    overall_gene_feats = inputs[:, :, 10:]

    embed = layers.Embedding(input_dim=len(token2int), output_dim=embed_dim)(
        categorical_feats
    )

    reshaped = layers.Reshape((seq_len, 3 * embed_dim))(embed)

    reshaped_1 = layers.Concatenate(axis=2)([reshaped, numerical_feats])
    normalized_layer_1 = layers.BatchNormalization()(reshaped_1)
    normalized_layer_1 = layers.SpatialDropout1D(sp_dropout)(normalized_layer_1)

    if gru_flag:
        for _ in range(n_layers):
            normalized_layer_1 = gru_layer(hidden_dim, dropout)(normalized_layer_1)
    else:
        for _ in range(n_layers):
            normalized_layer_1 = lstm_layer(hidden_dim, dropout)(normalized_layer_1)

    normalized_layer_2 = layers.BatchNormalization()(normalized_layer_1)
    concat_layer = layers.Concatenate(axis=2)([normalized_layer_2, overall_gene_feats])

    dense_layer_1 = layers.Dense(100, activation="linear")(concat_layer)
    normalized_layer_3 = layers.BatchNormalization()(dense_layer_1)
    dropout_layer_1 = layers.SpatialDropout1D(sp_dropout)(normalized_layer_3)

    dense_layer_2 = layers.Dense(100, activation="linear")(dropout_layer_1)
    normalized_layer_4 = layers.BatchNormalization()(dense_layer_2)
    dropout_layer_2 = layers.SpatialDropout1D(sp_dropout)(normalized_layer_4)

    truncated = dropout_layer_2[:, :pred_len]
    out = layers.Dense(5, activation="linear")(truncated)

    model = keras.Model(inputs=inputs, outputs=out)
    model.compile(optimizer=keras.optimizers.Adam(), loss=MCRMSE)
    return model




## === cell 11
public_df = test_data.query("seq_length == 107").copy()
private_df = test_data.query("seq_length == 130").copy()

public_inputs = preprocess_inputs(public_df, seq_length=107)
if len(private_df) > 0:
    private_inputs = preprocess_inputs(private_df, seq_length=130)
else:
    private_inputs = np.zeros((0, 130, public_inputs.shape[2]), dtype=np.float32)

print(public_inputs.shape, private_inputs.shape)



## === cell 12
WEIGHTS_DIR = "/kaggle/input/openvaccine-covid-model-weights"
lstm_w_path = f"{WEIGHTS_DIR}/LSTM model.h5"
gru_w_path = f"{WEIGHTS_DIR}/GRU model (1).h5"

if not (os.path.exists(lstm_w_path) and os.path.exists(gru_w_path)):
    raise FileNotFoundError(
        "Pretrained weight files not found under /kaggle/input/openvaccine-covid-model-weights. "
        "Please attach that dataset to the notebook."
    )

model_LSTM_on_test_data_public = build_model(seq_len=107, pred_len=107, gru_flag=False)
model_LSTM_on_test_data_public.load_weights(lstm_w_path)
pred_test_data_public_LSTM = model_LSTM_on_test_data_public.predict(
    public_inputs, verbose=0
)

model_GRU_on_test_data_public = build_model(seq_len=107, pred_len=107, gru_flag=True)
model_GRU_on_test_data_public.load_weights(gru_w_path)
pred_test_data_public_GRU = model_GRU_on_test_data_public.predict(
    public_inputs, verbose=0
)

if len(private_df) > 0:
    model_LSTM_on_test_data_private = build_model(
        seq_len=130, pred_len=130, gru_flag=False
    )
    model_LSTM_on_test_data_private.load_weights(lstm_w_path)
    pred_test_data_private_LSTM = model_LSTM_on_test_data_private.predict(
        private_inputs, verbose=0
    )

    model_GRU_on_test_data_private = build_model(
        seq_len=130, pred_len=130, gru_flag=True
    )
    model_GRU_on_test_data_private.load_weights(gru_w_path)
    pred_test_data_private_GRU = model_GRU_on_test_data_private.predict(
        private_inputs, verbose=0
    )
else:
    pred_test_data_private_LSTM = np.zeros((0, 130, 5), dtype=np.float32)
    pred_test_data_private_GRU = np.zeros((0, 130, 5), dtype=np.float32)

print(pred_test_data_public_LSTM.shape, pred_test_data_public_GRU.shape)




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_12/2462762095.py in <cell line: 0>()
      5 
      6 if not (os.path.exists(lstm_w_path) and os.path.exists(gru_w_path)):
----> 7     raise FileNotFoundError(
      8         "Pretrained weight files not found under /kaggle/input/openvaccine-covid-model-weights. "
      9         "Please attach that dataset to the notebook."

FileNotFoundError: Pretrained weight files not found under /kaggle/input/openvaccine-covid-model-weights. Please attach that dataset to the notebook.

## === cell 13
def format_predictions(public_df, private_df, public_preds, private_preds):
    preds = []
    for df, preds_ in [(public_df, public_preds), (private_df, private_preds)]:
        for i, uid in enumerate(df.id.to_list()):
            single_pred = preds_[i]
            single_df = pd.DataFrame(single_pred, columns=target_cols)
            single_df["id_seqpos"] = [f"{uid}_{x}" for x in range(single_df.shape[0])]
            preds.append(single_df)
    if len(preds) == 0:
        return pd.DataFrame(columns=["id_seqpos"] + target_cols)
    return pd.concat(preds).reset_index(drop=True)


lstm_preds = format_predictions(
    public_df, private_df, pred_test_data_public_LSTM, pred_test_data_private_LSTM
)
gru_preds = format_predictions(
    public_df, private_df, pred_test_data_public_GRU, pred_test_data_private_GRU
)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/162826071.py in <cell line: 0>()
     13 
     14 lstm_preds = format_predictions(
---> 15     public_df, private_df, pred_test_data_public_LSTM, pred_test_data_private_LSTM
     16 )
     17 gru_preds = format_predictions(

NameError: name 'pred_test_data_public_LSTM' is not defined

## === cell 14
submission_LSTM = submission_format[["id_seqpos"]].merge(
    lstm_preds, how="left", on="id_seqpos"
)
submission_GRU = submission_format[["id_seqpos"]].merge(
    gru_preds, how="left", on="id_seqpos"
)

submission_LSTM[target_cols] = submission_LSTM[target_cols].fillna(0.0)
submission_GRU[target_cols] = submission_GRU[target_cols].fillna(0.0)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/639981254.py in <cell line: 0>()
      1 # Align to sample submission rows; use left merge to guarantee all required id_seqpos exist.
      2 submission_LSTM = submission_format[["id_seqpos"]].merge(
----> 3     lstm_preds, how="left", on="id_seqpos"
      4 )
      5 submission_GRU = submission_format[["id_seqpos"]].merge(

NameError: name 'lstm_preds' is not defined

## === cell 15
submission_lstm_gru_combined = submission_format[["id_seqpos"]].copy()
gru_weight = 0.5
lstm_weight = 0.5

for c in target_cols:
    submission_lstm_gru_combined[c] = (
        submission_GRU[c].to_numpy(dtype=np.float32) * gru_weight
        + submission_LSTM[c].to_numpy(dtype=np.float32) * lstm_weight
    )

print(submission_lstm_gru_combined.shape)
print(submission_lstm_gru_combined.columns.tolist()[:6])



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/1257471110.py in <cell line: 0>()
      5 for c in target_cols:
      6     submission_lstm_gru_combined[c] = (
----> 7         submission_GRU[c].to_numpy(dtype=np.float32) * gru_weight
      8         + submission_LSTM[c].to_numpy(dtype=np.float32) * lstm_weight
      9     )

NameError: name 'submission_GRU' is not defined

## === cell 16
os.chdir(WORK_DIR)
submission_LSTM.to_csv("submission_LSTM.csv", index=False)
submission_GRU.to_csv("submission_GRU.csv", index=False)
submission_lstm_gru_combined.to_csv("submission_lstm_gru_combined.csv", index=False)

print("Wrote:", os.path.join(WORK_DIR, "submission_lstm_gru_combined.csv"))
print(submission_lstm_gru_combined.head())

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/922742928.py in <cell line: 0>()
      1 os.chdir(WORK_DIR)
----> 2 submission_LSTM.to_csv("submission_LSTM.csv", index=False)
      3 submission_GRU.to_csv("submission_GRU.csv", index=False)
      4 submission_lstm_gru_combined.to_csv("submission_lstm_gru_combined.csv", index=False)
      5 

NameError: name 'submission_LSTM' is not defined
