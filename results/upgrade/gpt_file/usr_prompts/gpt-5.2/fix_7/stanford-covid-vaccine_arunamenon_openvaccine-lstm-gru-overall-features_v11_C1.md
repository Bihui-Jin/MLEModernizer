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

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
tf_keras==2.18.0
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

# 5. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_ENABLE_PROTOBUF_CPP", "0")
os.environ.setdefault(
    "CUDA_VISIBLE_DEVICES", ""
)  # CPU for determinism/stability in Kaggle

import sys
import subprocess


def _ensure_protobuf_compatible():
    try:
        import google.protobuf  # noqa: F401
        import google.protobuf.__version__ as _pbver  # type: ignore
    except Exception:
        _pbver = None

    def _major(ver):
        try:
            return int(str(ver).split(".")[0])
        except Exception:
            return None

    maj = _major(_pbver)
    if maj is None or maj >= 5:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
            )
        except Exception as e:
            print("WARNING: Could not pip-install protobuf==4.25.3:", repr(e))


_ensure_protobuf_compatible()

import numpy as np
import pandas as pd

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))




## === cell 1
import json
import tensorflow as tf

print("TensorFlow:", tf.__version__)
print("Eager:", tf.executing_eagerly())

tf.keras.utils.set_random_seed(42)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass
try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF decide
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass




## === cell 2
os.chdir("/kaggle/")
os.getcwd()




## === cell 3
DATA_ROOT_CANDIDATES = [
    "/kaggle/input/stanford-covid-vaccine",
    "/kaggle/data/stanford-covid-vaccine",
    "/kaggle/input",
    "/kaggle/data",
]
DATA_ROOT = None
for p in DATA_ROOT_CANDIDATES:
    if os.path.exists(os.path.join(p, "train.json")) and os.path.exists(
        os.path.join(p, "test.json")
    ):
        DATA_ROOT = p
        break
if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not locate train.json/test.json under expected Kaggle paths."
    )

train_data = pd.read_json(os.path.join(DATA_ROOT, "train.json"), lines=True)
test_data = pd.read_json(os.path.join(DATA_ROOT, "test.json"), lines=True)
submission_format = pd.read_csv(
    os.path.join(DATA_ROOT, "sample_submission.csv"), encoding="utf-8-sig"
)

print("DATA_ROOT:", DATA_ROOT)




## === cell 4
train_data.head()




## === cell 5
train_data.shape




## === cell 6
train_data.groupby(["SN_filter"]).size()




## === cell 7
test_data.head()




## === cell 8
test_data.shape




## === cell 9
submission_format.head()




## === cell 10
print(train_data.shape)
print(test_data.shape)
print(submission_format.shape)




## === cell 11
print("Training data:\n", train_data["seq_scored"].value_counts())
print("Test data:\n", test_data["seq_scored"].value_counts())
len(train_data["reactivity"].iloc[0])




## === cell 12
len(train_data["sequence"].iloc[0])




## === cell 13
err_cols = [
    "reactivity_error",
    "deg_error_Mg_pH10",
    "deg_error_pH10",
    "deg_error_Mg_50C",
    "deg_error_50C",
]
flag = False
for c in err_cols:
    arr = np.asarray(train_data[c].tolist(), dtype=np.float32)
    if (arr < 0).any():
        flag = True
        break
print(flag)




## === cell 14
arr_reactivity = np.asarray(train_data["reactivity"].tolist(), dtype=np.float32)
arr_deg_Mg_pH10 = np.asarray(train_data["deg_Mg_pH10"].tolist(), dtype=np.float32)
arr_deg_pH10 = np.asarray(train_data["deg_pH10"].tolist(), dtype=np.float32)
arr_deg_Mg_50C = np.asarray(train_data["deg_Mg_50C"].tolist(), dtype=np.float32)
arr_deg_50C = np.asarray(train_data["deg_50C"].tolist(), dtype=np.float32)

print(
    float(arr_reactivity.min()),
    float(arr_deg_Mg_pH10.min()),
    float(arr_deg_pH10.min()),
    float(arr_deg_Mg_50C.min()),
    float(arr_deg_50C.min()),
)




## === cell 15
train_data.columns




## === cell 16
token2int = {x: i for i, x in enumerate("().ACGUBEHIMSX")}
target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]




## === cell 17
token2int




## === cell 18
BPPS_DIR = os.path.join(DATA_ROOT, "bpps")

_BPPS_CACHE = {}


def _safe_load_bpps(mol_id, seq_len):
    path = os.path.join(BPPS_DIR, f"{mol_id}.npy")
    key = (mol_id, seq_len)
    if key in _BPPS_CACHE:
        return _BPPS_CACHE[key]
    if os.path.exists(path):
        bpps = np.load(path, mmap_mode="r")
    else:
        bpps = np.zeros((seq_len, seq_len), dtype=np.float32)
    _BPPS_CACHE[key] = bpps
    return bpps


def read_bpps_sum(df):
    bpps_arr = []
    for mol_id, seq_len in zip(df.id.to_list(), df.seq_length.to_list()):
        bpps = _safe_load_bpps(mol_id, seq_len)
        bpps_arr.append(np.asarray(bpps).sum(axis=1, dtype=np.float32))
    return bpps_arr


def read_bpps_max(df):
    bpps_arr = []
    for mol_id, seq_len in zip(df.id.to_list(), df.seq_length.to_list()):
        bpps = _safe_load_bpps(mol_id, seq_len)
        bpps_arr.append(np.asarray(bpps).max(axis=1).astype(np.float32, copy=False))
    return bpps_arr


def read_bpps_nb(df):
    bpps_nb_mean = 0.077522
    bpps_nb_std = 0.08914
    bpps_arr = []
    for mol_id, seq_len in zip(df.id.to_list(), df.seq_length.to_list()):
        bpps = _safe_load_bpps(mol_id, seq_len)
        bpps = np.asarray(bpps)
        bpps_nb = (bpps > 0).sum(axis=0).astype(np.float32) / float(bpps.shape[0])
        bpps_nb = (bpps_nb - bpps_nb_mean) / bpps_nb_std
        bpps_arr.append(bpps_nb.astype(np.float32, copy=False))
    return bpps_arr


os.chdir("/kaggle/working/")
train_data["bpps_sum"] = read_bpps_sum(train_data)
test_data["bpps_sum"] = read_bpps_sum(test_data)
train_data["bpps_max"] = read_bpps_max(train_data)
test_data["bpps_max"] = read_bpps_max(test_data)
train_data["bpps_nb"] = read_bpps_nb(train_data)
test_data["bpps_nb"] = read_bpps_nb(test_data)

train_data.head()




## === cell 19
_token_map = np.full(256, -1, dtype=np.int16)
for k, v in token2int.items():
    _token_map[ord(k)] = v


def _encode_str_array(str_list):
    b = np.frombuffer(("".join(str_list)).encode("ascii"), dtype=np.uint8)
    L = len(str_list[0])
    return _token_map[b].reshape(len(str_list), L)


def preprocess_inputs(df, cols=["sequence", "structure", "predicted_loop_type"]):
    seq = _encode_str_array(df[cols[0]].tolist())
    struct = _encode_str_array(df[cols[1]].tolist())
    loop = _encode_str_array(df[cols[2]].tolist())
    base_fea = np.stack([seq, struct, loop], axis=2).astype(np.float32, copy=False)

    bpps_sum_fea = np.asarray(df["bpps_sum"].to_list(), dtype=np.float32)[
        :, :, np.newaxis
    ]
    bpps_max_fea = np.asarray(df["bpps_max"].to_list(), dtype=np.float32)[
        :, :, np.newaxis
    ]
    bpps_nb_fea = np.asarray(df["bpps_nb"].to_list(), dtype=np.float32)[
        :, :, np.newaxis
    ]

    return np.concatenate(
        [base_fea, bpps_sum_fea, bpps_max_fea, bpps_nb_fea], 2
    ).astype(np.float32, copy=False)




## === cell 20
from tqdm import tqdm


def get_structure_adj(train):
    Ss = []
    for i in tqdm(range(len(train))):
        seq_length = int(train["seq_length"].iloc[i])
        structure = train["structure"].iloc[i]

        paired = np.zeros((seq_length, 1), dtype=np.float32)
        cue = []
        for j, ch in enumerate(structure):
            if ch == "(":
                cue.append(j)
            elif ch == ")":
                start = cue.pop()
                paired[start, 0] = 1.0
                paired[j, 0] = 1.0
        a = np.zeros((seq_length, seq_length, 1), dtype=np.float32)
        a[np.arange(seq_length), np.arange(seq_length), 0] = paired[:, 0]
        Ss.append(a)

    Ss = np.array(Ss, dtype=np.float32)
    print(Ss.shape)
    return Ss




## === cell 21
train_filtered = train_data.loc[train_data["signal_to_noise"] > 1].reset_index(
    drop=True
)
train_inputs = preprocess_inputs(train_filtered)
train_labels = np.array(
    train_filtered[target_cols].values.tolist(), dtype=np.float32
).transpose((0, 2, 1))

Ss = get_structure_adj(train_filtered)
Ss = Ss.sum(axis=1)  # (N, seq_len, 1)
train_inputs = np.concatenate([train_inputs, Ss], 2)  # adds 1 numerical feature

print("train_inputs:", train_inputs.shape)
print("train_labels:", train_labels.shape)




## === cell 22
preprocess_inputs(train_data.loc[[0]]).shape




## === cell 23
test_data.head()




## === cell 24
def MCRMSE(y_true, y_pred):
    colwise_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)  # (batch, 5)
    return tf.reduce_mean(tf.sqrt(colwise_mse + 1e-12), axis=1)  # (batch,)


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


def build_model(
    seq_len=107,
    num_features=7,
    embed_dim=100,
    hidden_dim=256,
    dropout=0.2,
    pred_len=68,
    gru_flag=False,
):
    inputs = tf.keras.layers.Input(shape=(seq_len, num_features))

    categorical_feats = inputs[:, :, :3]
    numerical_feats = inputs[:, :, 3:]

    embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=embed_dim)(
        categorical_feats
    )

    reshaped = tf.keras.layers.Reshape((seq_len, 3 * embed_dim))(embed)

    reshaped = tf.keras.layers.Concatenate(axis=2)([reshaped, numerical_feats])
    normalized_layer_1 = tf.keras.layers.BatchNormalization()(reshaped)

    if gru_flag:
        rnn_out = gru_layer(hidden_dim, dropout)(normalized_layer_1)
    else:
        rnn_out = lstm_layer(hidden_dim, dropout)(normalized_layer_1)

    normalized_layer_2 = tf.keras.layers.BatchNormalization()(rnn_out)
    truncated = normalized_layer_2[:, :pred_len]
    out = tf.keras.layers.Dense(5, activation="linear")(truncated)

    model = tf.keras.Model(inputs=inputs, outputs=out)
    adam = tf.optimizers.Adam()
    model.compile(optimizer=adam, loss=MCRMSE)
    return model




## === cell 25
SEQ_LEN = train_inputs.shape[1]
NUM_FEATURES = train_inputs.shape[2]
PRED_LEN = train_labels.shape[1]

EPOCHS = 60
BATCH_SIZE = 32

model_GRU_on_train_data = build_model(
    seq_len=SEQ_LEN, num_features=NUM_FEATURES, pred_len=PRED_LEN, gru_flag=True
)
model_GRU_on_train_data.summary()
model_GRU_callback = tf.keras.callbacks.ModelCheckpoint(
    "GRU_model.weights.h5",
    save_weights_only=True,
    monitor="loss",
    mode="min",
    save_best_only=True,
)

history_GRU = model_GRU_on_train_data.fit(
    train_inputs,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=2,
    callbacks=[model_GRU_callback],
)




## === cell 26
model_LSTM_on_train_data = build_model(
    seq_len=SEQ_LEN, num_features=NUM_FEATURES, pred_len=PRED_LEN, gru_flag=False
)
model_LSTM_on_train_data.summary()
model_LSTM_callback = tf.keras.callbacks.ModelCheckpoint(
    "LSTM_model.weights.h5",
    save_weights_only=True,
    monitor="loss",
    mode="min",
    save_best_only=True,
)

history_LSTM = model_LSTM_on_train_data.fit(
    train_inputs,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=2,
    callbacks=[model_LSTM_callback],
)




## === cell 27
from matplotlib import pyplot as plt

print(f" LSTM loss: {min(history_LSTM.history['loss'])}")
print(f" GRU loss: {min(history_GRU.history['loss'])}")

fig, ax = plt.subplots(1, 1, figsize=(20, 10))
ax.plot(history_LSTM.history["loss"], label="LSTM")
ax.plot(history_GRU.history["loss"], label="GRU")
ax.set_title("Model - LSTM vs GRU")
ax.set_ylabel("Loss")
ax.set_xlabel("Epoch")
ax.legend()
plt.show()




## === cell 28
seq_lens = sorted(test_data["seq_length"].unique().tolist())
print("Test seq_length unique:", seq_lens)

public_df = test_data.query("seq_length == 107").copy()
private_df = test_data.query("seq_length != 107").copy()  # may be empty

public_inputs = preprocess_inputs(public_df)
Ss_pub = get_structure_adj(public_df).sum(axis=1)
public_inputs = np.concatenate([public_inputs, Ss_pub], 2).astype(np.float32)

if len(private_df) > 0:
    private_inputs = preprocess_inputs(private_df)
    Ss_pri = get_structure_adj(private_df).sum(axis=1)
    private_inputs = np.concatenate([private_inputs, Ss_pri], 2).astype(np.float32)
else:
    private_inputs = None

print(
    "public_inputs:",
    public_inputs.shape,
    "private_inputs:",
    None if private_inputs is None else private_inputs.shape,
)




## === cell 29
TRAIN_PRED_LEN = int(PRED_LEN)  # 68

model_LSTM_on_test_data_public = build_model(
    seq_len=public_inputs.shape[1],
    num_features=public_inputs.shape[2],
    pred_len=TRAIN_PRED_LEN,
    gru_flag=False,
)
model_LSTM_on_test_data_public.load_weights("LSTM_model.weights.h5")
pred_test_data_public_LSTM = model_LSTM_on_test_data_public.predict(
    public_inputs, batch_size=64, verbose=0
)

model_GRU_on_test_data_public = build_model(
    seq_len=public_inputs.shape[1],
    num_features=public_inputs.shape[2],
    pred_len=TRAIN_PRED_LEN,
    gru_flag=True,
)
model_GRU_on_test_data_public.load_weights("GRU_model.weights.h5")
pred_test_data_public_GRU = model_GRU_on_test_data_public.predict(
    public_inputs, batch_size=64, verbose=0
)

if private_inputs is not None:
    model_LSTM_on_test_data_private = build_model(
        seq_len=private_inputs.shape[1],
        num_features=private_inputs.shape[2],
        pred_len=TRAIN_PRED_LEN,
        gru_flag=False,
    )
    model_LSTM_on_test_data_private.load_weights("LSTM_model.weights.h5")
    pred_test_data_private_LSTM = model_LSTM_on_test_data_private.predict(
        private_inputs, batch_size=64, verbose=0
    )

    model_GRU_on_test_data_private = build_model(
        seq_len=private_inputs.shape[1],
        num_features=private_inputs.shape[2],
        pred_len=TRAIN_PRED_LEN,
        gru_flag=True,
    )
    model_GRU_on_test_data_private.load_weights("GRU_model.weights.h5")
    pred_test_data_private_GRU = model_GRU_on_test_data_private.predict(
        private_inputs, batch_size=64, verbose=0
    )
else:
    pred_test_data_private_LSTM = np.zeros((0, TRAIN_PRED_LEN, 5), dtype=np.float32)
    pred_test_data_private_GRU = np.zeros((0, TRAIN_PRED_LEN, 5), dtype=np.float32)

print(pred_test_data_public_LSTM.shape, pred_test_data_public_GRU.shape)




## === cell 30
def _pad_to_seq_len(pred_68, seq_len=107):
    """Pad (N, pred_len, 5) to (N, seq_len, 5) with zeros after pred_len."""
    n, pred_len, c = pred_68.shape
    if pred_len == seq_len:
        return pred_68
    out = np.zeros((n, seq_len, c), dtype=pred_68.dtype)
    out[:, :pred_len, :] = pred_68
    return out


def format_predictions(public_preds, private_preds):
    parts = []
    for df, preds_ in [(public_df, public_preds), (private_df, private_preds)]:
        if len(df) == 0:
            continue
        seq_len = int(df["seq_length"].iloc[0])
        preds_seq = _pad_to_seq_len(preds_, seq_len=seq_len)  # (N, seq_len, 5)

        ids = df["id"].to_numpy()
        n = len(ids)
        pos = np.arange(seq_len, dtype=np.int32)

        id_seqpos = np.char.add(np.repeat(ids.astype(str), seq_len), "_")
        id_seqpos = np.char.add(id_seqpos, np.tile(pos.astype(str), n))

        flat = preds_seq.reshape(n * seq_len, 5)
        out = pd.DataFrame(flat, columns=target_cols)
        out.insert(0, "id_seqpos", id_seqpos)
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


lstm_preds = format_predictions(pred_test_data_public_LSTM, pred_test_data_private_LSTM)
gru_preds = format_predictions(pred_test_data_public_GRU, pred_test_data_private_GRU)

print(lstm_preds.shape, gru_preds.shape)
lstm_preds.head()




## === cell 31
submission_LSTM = submission_format[["id_seqpos"]].merge(
    lstm_preds, how="left", on="id_seqpos"
)
submission_GRU = submission_format[["id_seqpos"]].merge(
    gru_preds, how="left", on="id_seqpos"
)

submission_LSTM[target_cols] = submission_LSTM[target_cols].fillna(0.0)
submission_GRU[target_cols] = submission_GRU[target_cols].fillna(0.0)

print(submission_LSTM.shape, submission_GRU.shape)
submission_LSTM.head()




## === cell 32
submission_lstm_gru_combined = submission_GRU.merge(
    submission_LSTM, how="inner", on="id_seqpos", suffixes=("_gru", "_lstm")
)

gru_weight = 0.6
lstm_weight = 0.4
for c in target_cols:
    submission_lstm_gru_combined[c] = (
        submission_lstm_gru_combined[f"{c}_gru"] * gru_weight
        + submission_lstm_gru_combined[f"{c}_lstm"] * lstm_weight
    )

submission_lstm_gru_combined = submission_lstm_gru_combined[["id_seqpos"] + target_cols]
submission_lstm_gru_combined.head()




## === cell 33
os.chdir("/kaggle/working/")
submission_LSTM.to_csv("submission_LSTM.csv", index=False)
submission_GRU.to_csv("submission_GRU.csv", index=False)
submission_lstm_gru_combined.to_csv("submission.csv", index=False)

print("Wrote:")
print("/kaggle/working/submission.csv")
print("/kaggle/working/submission_LSTM.csv")
print("/kaggle/working/submission_GRU.csv")
print(submission_lstm_gru_combined.shape)
print(submission_lstm_gru_combined.columns.tolist())

assert submission_lstm_gru_combined.shape[0] == submission_format.shape[0]
assert list(submission_lstm_gru_combined.columns) == ["id_seqpos"] + target_cols
assert submission_lstm_gru_combined[target_cols].isna().sum().sum() == 0
