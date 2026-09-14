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

# 8. Previous improvement plans

- What this solution (achieved 0.45245) has done: 'I fix the runtime crash caused by importing `keras` in this Kaggle environment by switching to `tf.keras` while keeping the exact same model/loss logic. I also remove the hard dependency on an external Kaggle dataset for pretrained weights (which isn’t attached) and instead train the same model architecture briefly on the provided `train.json`, then predict on `test.json`. Finally, I make the test preprocessing robust to varying `seq_length` (107 vs 130) so concatenation shapes don’t break, and I guarantee the submission is aligned to `sample_submission.csv` and written as a valid `.csv` file.'
- What this solution (achieved 0.91474) has done: 'I fix the TensorFlow import crash (`MessageFactory` / protobuf incompatibility) by forcing TensorFlow to use the pure-Python protobuf implementation before importing TF, which is the standard Kaggle-safe workaround and is score-neutral. I also make data path selection more robust by falling back to `/kaggle/input` and `/kaggle/data` if the initially configured dataset directory is not present, without changing preprocessing/model logic. Finally, I keep the exact same training/inference pipeline but ensure the submission is always fully populated and written to `/kaggle/working/submission.csv` with the required columns.'
- What this solution (achieved 0.96762) has done: 'I fix the TensorFlow/protobuf crash by moving the protobuf environment variables to the very top of the script (before any TensorFlow-related import can occur) and by enforcing the pure-Python protobuf backend early, which is the direct root cause of the `MessageFactory.GetPrototype` failure. Then I correct an input-shape logic bug in preprocessing (`Ss` was incorrectly allocated as `(N, L, 1)` while it should match the `(N, L, L, 1)` adjacency aggregation), which currently degrades learning and leads to a much worse score than expected. Finally, I ensure test predictions are generated for *all* test rows (not only `seq_length==107/130` subsets) and are aligned 1:1 to `sample_submission.csv`, guaranteeing a complete, valid `submission.csv` and improving score toward the target without changing the model architecture or training approach.'

# 9. Code solution

## === cell 0
import os

os.environ["PYTHONHASHSEED"] = "0"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import numpy as np
import pandas as pd

np.random.seed(0)



## === cell 1
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(0)


class _Ops:
    @staticmethod
    def mean(x, axis=None):
        return tf.reduce_mean(x, axis=axis)

    @staticmethod
    def square(x):
        return tf.square(x)

    @staticmethod
    def sqrt(x):
        return tf.sqrt(x)


ops = _Ops()



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
CANDIDATE_DATA_DIRS = [
    "/kaggle/input/stanford-covid-vaccine",
    "/kaggle/input/stanford-covid-vaccine/stanford-covid-vaccine",
    "/kaggle/data/stanford-covid-vaccine",
    "/kaggle/data/stanford-covid-vaccine/stanford-covid-vaccine",
    "/kaggle/input",
    "/kaggle/data",
]
DATA_DIR = None
for d in CANDIDATE_DATA_DIRS:
    if os.path.exists(os.path.join(d, "train.json")) and os.path.exists(
        os.path.join(d, "test.json")
    ):
        DATA_DIR = d
        break
if DATA_DIR is None:
    raise FileNotFoundError(
        "Could not find train.json/test.json in expected Kaggle directories. "
        f"Tried: {CANDIDATE_DATA_DIRS}"
    )

WORK_DIR = "/kaggle/working"
os.makedirs(WORK_DIR, exist_ok=True)

train_path = f"{DATA_DIR}/train.json"
test_path = f"{DATA_DIR}/test.json"
sub_path = f"{DATA_DIR}/sample_submission.csv"
if not os.path.exists(sub_path):
    for d in [
        DATA_DIR,
        "/kaggle/input",
        "/kaggle/data",
        "/kaggle/input/stanford-covid-vaccine",
        "/kaggle/data/stanford-covid-vaccine",
    ]:
        cand = os.path.join(d, "sample_submission.csv")
        if os.path.exists(cand):
            sub_path = cand
            break

train_data = pd.read_json(train_path, lines=True)
test_data = pd.read_json(test_path, lines=True)
submission_format = pd.read_csv(sub_path, encoding="utf-8-sig")

print("DATA_DIR:", DATA_DIR)
print(train_data.shape, test_data.shape, submission_format.shape)



## === cell 3
token2int = {x: i for i, x in enumerate("().ACGUBEHIMSX")}
target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]




## === cell 4
def _safe_bpps_feature(df, reducer="sum"):
    out = []
    for seq_len in df["seq_length"].to_numpy():
        out.append(np.zeros((int(seq_len),), dtype=np.float32))
    return out


def _safe_bpps_nb(df):
    out = []
    for seq_len in df["seq_length"].to_numpy():
        out.append(np.zeros((int(seq_len),), dtype=np.float32))
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
def get_structure_adj(df, seq_length=107):
    Ss = []
    for i in range(len(df)):
        L_i = int(df["seq_length"].iloc[i])
        structure = df["structure"].iloc[i]
        sequence = df["sequence"].iloc[i]

        cue = []
        a_structures = {
            ("A", "U"): np.zeros([L_i, L_i], dtype=np.float32),
            ("C", "G"): np.zeros([L_i, L_i], dtype=np.float32),
            ("U", "G"): np.zeros([L_i, L_i], dtype=np.float32),
            ("U", "A"): np.zeros([L_i, L_i], dtype=np.float32),
            ("G", "C"): np.zeros([L_i, L_i], dtype=np.float32),
            ("G", "U"): np.zeros([L_i, L_i], dtype=np.float32),
        }

        for j in range(L_i):
            if structure[j] == "(":
                cue.append(j)
            elif structure[j] == ")":
                start = cue.pop()
                a_structures[(sequence[start], sequence[j])][start, j] = 1.0
                a_structures[(sequence[j], sequence[start])][j, start] = 1.0

        a_strc = np.stack(list(a_structures.values()), axis=2)
        a_strc = np.sum(a_strc, axis=2, keepdims=True)  # (Li, Li, 1)

        out = np.zeros((seq_length, seq_length, 1), dtype=np.float32)
        m = min(seq_length, a_strc.shape[0])
        out[:m, :m, :] = a_strc[:m, :m, :]
        Ss.append(out)

    return np.asarray(Ss, dtype=np.float32)  # (N, L, L, 1)


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
def _pad_1d(arr, L, value=0.0, dtype=np.float32):
    arr = np.asarray(arr, dtype=dtype)
    if arr.shape[0] >= L:
        return arr[:L]
    out = np.full((L,), value, dtype=dtype)
    out[: arr.shape[0]] = arr
    return out


def _seq_to_ints(seq, L):
    ints = [token2int.get(x, 0) for x in seq]  # default to 0 if unexpected token
    return _pad_1d(ints, L, value=0, dtype=np.int32)


def preprocess_inputs(
    df, cols=("sequence", "structure", "predicted_loop_type"), seq_length=107
):
    N = len(df)

    base_fea = np.zeros((N, seq_length, 3), dtype=np.int32)
    for i in range(N):
        for j, c in enumerate(cols):
            base_fea[i, :, j] = _seq_to_ints(df.iloc[i][c], seq_length)

    bpps_sum_fea = np.zeros((N, seq_length, 1), dtype=np.float32)
    bpps_max_fea = np.zeros((N, seq_length, 1), dtype=np.float32)
    bpps_nb_fea = np.zeros((N, seq_length, 1), dtype=np.float32)
    for i in range(N):
        bpps_sum_fea[i, :, 0] = _pad_1d(df.iloc[i]["bpps_sum"], seq_length, value=0.0)
        bpps_max_fea[i, :, 0] = _pad_1d(df.iloc[i]["bpps_max"], seq_length, value=0.0)
        bpps_nb_fea[i, :, 0] = _pad_1d(df.iloc[i]["bpps_nb"], seq_length, value=0.0)

    per_pos_num = np.concatenate([bpps_sum_fea, bpps_max_fea, bpps_nb_fea], axis=2)

    per_pos_num_2d = np.repeat(
        per_pos_num[:, :, None, :], seq_length, axis=2
    )  # (N,L,L,3)

    Ss = get_structure_adj(df, seq_length=seq_length)

    Ds = get_distance_matrix(N, seq_length)

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
    global_stack = []
    for col in global_cols:
        v = df[col].to_numpy(dtype=np.float32)[:, None, None, None]  # (N,1,1,1)
        v = np.repeat(v, seq_length, axis=1)
        v = np.repeat(v, seq_length, axis=2)  # (N,L,L,1)
        global_stack.append(v)
    global_2d = (
        np.concatenate(global_stack, axis=3)
        if global_stack
        else np.zeros((N, seq_length, seq_length, 0), dtype=np.float32)
    )

    numeric_2d = np.concatenate([per_pos_num_2d, Ss, Ds, global_2d], axis=3).astype(
        np.float32
    )

    cat_2d = np.repeat(base_fea[:, :, None, :], seq_length, axis=2).astype(
        np.float32
    )  # (N,L,L,3)

    data = np.concatenate([cat_2d, numeric_2d], axis=3).astype(
        np.float32
    )  # (N,L,L, 3+24=27)
    return data




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
    num_features=27,  # updated to match preprocess_inputs
    embed_dim=200,
    sp_dropout=0.2,
    hidden_dim=512,
    dropout=0.5,
    pred_len=68,
    gru_flag=False,
):
    inputs = layers.Input(shape=(seq_len, seq_len, num_features))

    x = layers.Lambda(lambda t: tf.reduce_mean(t, axis=2))(inputs)  # (N, L, F)

    categorical_feats = x[:, :, :3]
    numerical_feats = x[:, :, 3:10]
    overall_gene_feats = x[:, :, 10:]

    embed = layers.Embedding(input_dim=len(token2int), output_dim=embed_dim)(
        tf.cast(categorical_feats, tf.int32)
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
idx = np.arange(len(train_inputs))
rng = np.random.RandomState(0)
rng.shuffle(idx)

val_size = max(1, int(0.1 * len(idx)))
val_idx = idx[:val_size]
tr_idx = idx[val_size:]

X_tr, y_tr = train_inputs[tr_idx], train_labels[tr_idx]
X_va, y_va = train_inputs[val_idx], train_labels[val_idx]

model = build_model(
    seq_len=107, pred_len=68, gru_flag=False, num_features=train_inputs.shape[-1]
)
history = model.fit(
    X_tr,
    y_tr,
    validation_data=(X_va, y_va),
    epochs=3,
    batch_size=16,
    verbose=2,
)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2358360886.py in <cell line: 0>()
     10 X_va, y_va = train_inputs[val_idx], train_labels[val_idx]
     11 
---> 12 model = build_model(
     13     seq_len=107, pred_len=68, gru_flag=False, num_features=train_inputs.shape[-1]
     14 )

/tmp/ipykernel_11/532122209.py in build_model(n_layers, seq_len, num_features, embed_dim, sp_dropout, hidden_dim, dropout, pred_len, gru_flag)
     48 
     49     embed = layers.Embedding(input_dim=len(token2int), output_dim=embed_dim)(
---> 50         tf.cast(categorical_feats, tf.int32)
     51     )
     52     reshaped = layers.Reshape((seq_len, 3 * embed_dim))(embed)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/keras_tensor.py in __tf_tensor__(self, dtype, name)
    136 
    137     def __tf_tensor__(self, dtype=None, name=None):
--> 138         raise ValueError(
    139             "A KerasTensor cannot be used as input to a TensorFlow function. "
    140             "A KerasTensor is a symbolic placeholder for a shape and dtype, "

ValueError: A KerasTensor cannot be used as input to a TensorFlow function. A KerasTensor is a symbolic placeholder for a shape and dtype, used when constructing Keras Functional models or Keras Functions. You can only use it as input to a Keras layer or a Keras operation (from the namespaces `keras.layers` and `keras.operations`). You are likely doing something like:

```
x = Input(...)
...
tf_fn(x)  # Invalid.
```

What you should do instead is wrap `tf_fn` in a layer:

```
class MyLayer(Layer):
    def call(self, x):
        return tf_fn(x)

x = MyLayer()(x)
```


## === cell 12
test_inputs = preprocess_inputs(test_data, seq_length=107)
pred_68 = model.predict(test_inputs, verbose=0)  # (N,68,5)

pred_all = np.zeros((len(test_data), 107, 5), dtype=np.float32)
pred_all[:, :68, :] = pred_68

print("pred_all:", pred_all.shape)




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/215590841.py in <cell line: 0>()
      1 test_inputs = preprocess_inputs(test_data, seq_length=107)
----> 2 pred_68 = model.predict(test_inputs, verbose=0)  # (N,68,5)
      3 
      4 pred_all = np.zeros((len(test_data), 107, 5), dtype=np.float32)
      5 pred_all[:, :68, :] = pred_68

NameError: name 'model' is not defined

## === cell 13
def format_predictions(df, preds_107):
    preds = []
    for i, uid in enumerate(df.id.to_list()):
        single_pred = preds_107[i]  # (107,5)
        single_df = pd.DataFrame(single_pred, columns=target_cols)
        single_df["id_seqpos"] = [f"{uid}_{x}" for x in range(single_df.shape[0])]
        preds.append(single_df)
    if len(preds) == 0:
        return pd.DataFrame(columns=["id_seqpos"] + target_cols)
    return pd.concat(preds).reset_index(drop=True)


model_preds = format_predictions(test_data, pred_all)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1488858021.py in <cell line: 0>()
     11 
     12 
---> 13 model_preds = format_predictions(test_data, pred_all)
     14 

NameError: name 'pred_all' is not defined

## === cell 14
submission = submission_format[["id_seqpos"]].merge(
    model_preds, how="left", on="id_seqpos"
)
submission[target_cols] = submission[target_cols].astype(np.float32).fillna(0.0)

print(submission.shape)
print(submission.head())



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2077945862.py in <cell line: 0>()
      1 submission = submission_format[["id_seqpos"]].merge(
----> 2     model_preds, how="left", on="id_seqpos"
      3 )
      4 submission[target_cols] = submission[target_cols].astype(np.float32).fillna(0.0)
      5 

NameError: name 'model_preds' is not defined

## === cell 15
os.chdir(WORK_DIR)
out_path = os.path.join(WORK_DIR, "submission.csv")
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
print("Columns:", submission.columns.tolist())
print("Nulls per target:", submission[target_cols].isna().sum().to_dict())
print("Any NaN in file:", submission.isna().any().to_dict())

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3014531068.py in <cell line: 0>()
      1 os.chdir(WORK_DIR)
      2 out_path = os.path.join(WORK_DIR, "submission.csv")
----> 3 submission.to_csv(out_path, index=False)
      4 print("Wrote:", out_path)
      5 print("Columns:", submission.columns.tolist())

NameError: name 'submission' is not defined
