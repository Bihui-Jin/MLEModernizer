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

os.environ.setdefault(
    "CUDA_VISIBLE_DEVICES", ""
)  # CPU for determinism/stability in Kaggle
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")

import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf

print("TensorFlow:", tf.__version__)
print("Eager:", tf.executing_eagerly())

tf.keras.utils.set_random_seed(42)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    ncpu = os.cpu_count() or 4
    intra = min(8, max(1, ncpu))
    inter = 2
    tf.config.threading.set_intra_op_parallelism_threads(intra)
    tf.config.threading.set_inter_op_parallelism_threads(inter)
    os.environ.setdefault("OMP_NUM_THREADS", str(intra))
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(intra))
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", str(inter))
    print("Threads intra/inter:", intra, inter)
except Exception:
    pass




## === cell 1
os.chdir("/kaggle/")
os.getcwd()




## === cell 2
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




## === cell 3
train_data.head()




## === cell 4
train_data.shape




## === cell 5
train_data.groupby(["SN_filter"]).size()




## === cell 6
test_data.head()




## === cell 7
test_data.shape




## === cell 8
submission_format.head()




## === cell 9
print(train_data.shape)
print(test_data.shape)
print(submission_format.shape)




## === cell 10
print("Training data:\n", train_data["seq_scored"].value_counts())
print("Test data:\n", test_data["seq_scored"].value_counts())
len(train_data["reactivity"].iloc[0])




## === cell 11
len(train_data["sequence"].iloc[0])




## === cell 12
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




## === cell 13
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




## === cell 14
train_data.columns




## === cell 15
token2int = {x: i for i, x in enumerate("().ACGUBEHIMSX")}
target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]




## === cell 16
token2int




## === cell 17
BPPS_DIR = os.path.join(DATA_ROOT, "bpps")

_BPPS_CACHE = {}


def _safe_load_bpps(mol_id, seq_len):
    path = os.path.join(BPPS_DIR, f"{mol_id}.npy")
    bpps = _BPPS_CACHE.get(mol_id)
    if bpps is not None:
        return bpps
    if os.path.exists(path):
        bpps = np.load(path, mmap_mode="r")
    else:
        bpps = np.zeros((int(seq_len), int(seq_len)), dtype=np.float32)
    _BPPS_CACHE[mol_id] = bpps
    return bpps


def read_bpps_features(df):
    bpps_nb_mean = 0.077522
    bpps_nb_std = 0.08914

    ids = df["id"].astype(str).to_list()
    lens = df["seq_length"].astype(int).to_list()
    n = len(ids)
    L0 = int(lens[0])

    sums = np.empty((n, L0), dtype=np.float32)
    maxs = np.empty((n, L0), dtype=np.float32)
    nbs = np.empty((n, L0), dtype=np.float32)

    for i, (mol_id, seq_len) in enumerate(zip(ids, lens)):
        bpps = _safe_load_bpps(mol_id, seq_len)
        s = bpps.sum(axis=1, dtype=np.float32)
        m = bpps.max(axis=1).astype(np.float32, copy=False)
        bpps_nb = (bpps > 0).sum(axis=0).astype(np.float32, copy=False) / float(seq_len)
        bpps_nb = (bpps_nb - bpps_nb_mean) / bpps_nb_std

        if s.shape[0] != L0:
            tmp = np.zeros((L0,), dtype=np.float32)
            tmp[: min(L0, s.shape[0])] = s[: min(L0, s.shape[0])]
            s = tmp
            tmp = np.zeros((L0,), dtype=np.float32)
            tmp[: min(L0, m.shape[0])] = m[: min(L0, m.shape[0])]
            m = tmp
            tmp = np.zeros((L0,), dtype=np.float32)
            tmp[: min(L0, bpps_nb.shape[0])] = bpps_nb[: min(L0, bpps_nb.shape[0])]
            bpps_nb = tmp

        sums[i] = s
        maxs[i] = m
        nbs[i] = bpps_nb.astype(np.float32, copy=False)

    return sums, maxs, nbs


os.chdir("/kaggle/working/")
train_s, train_m, train_nb = read_bpps_features(train_data)
test_s, test_m, test_nb = read_bpps_features(test_data)

train_data["bpps_sum"] = list(train_s)
train_data["bpps_max"] = list(train_m)
train_data["bpps_nb"] = list(train_nb)
test_data["bpps_sum"] = list(test_s)
test_data["bpps_max"] = list(test_m)
test_data["bpps_nb"] = list(test_nb)

train_data.head()




## === cell 18
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

    bpps_sum_fea = np.asarray(df["bpps_sum"].to_list(), dtype=np.float32)[:, :, None]
    bpps_max_fea = np.asarray(df["bpps_max"].to_list(), dtype=np.float32)[:, :, None]
    bpps_nb_fea = np.asarray(df["bpps_nb"].to_list(), dtype=np.float32)[:, :, None]

    out = np.concatenate([base_fea, bpps_sum_fea, bpps_max_fea, bpps_nb_fea], axis=2)
    return np.ascontiguousarray(out, dtype=np.float32)




## === cell 19
_STRUCT_MASK_CACHE = {}


def _paired_mask_from_structure_numpy(struct: str, seq_len: int) -> np.ndarray:
    L = len(struct)
    mask = np.zeros((L,), dtype=np.float32)
    stack = []
    for i, ch in enumerate(struct):
        if ch == "(":
            stack.append(i)
        elif ch == ")":
            if stack:
                j = stack.pop()
                mask[j] = 1.0
                mask[i] = 1.0
    if seq_len == L:
        return mask
    if seq_len < L:
        return mask[:seq_len]
    out = np.zeros((seq_len,), dtype=np.float32)
    out[:L] = mask
    return out


def get_structure_adj(df):
    seq_len0 = int(df["seq_length"].iloc[0])
    ids = df["id"].astype(str).to_numpy()
    structs = df["structure"].astype(str).to_numpy()

    out = np.empty((len(df), seq_len0), dtype=np.float32)
    cache = _STRUCT_MASK_CACHE
    for i in range(len(df)):
        key = (ids[i], structs[i], seq_len0)
        m = cache.get(key)
        if m is None:
            m = _paired_mask_from_structure_numpy(structs[i], seq_len0)
            cache[key] = m
        out[i] = m
    return out[:, :, None]




## === cell 20
train_filtered = train_data.loc[train_data["signal_to_noise"] > 1].reset_index(
    drop=True
)
train_inputs = preprocess_inputs(train_filtered)
train_labels = np.array(
    train_filtered[target_cols].values.tolist(), dtype=np.float32
).transpose((0, 2, 1))

Ss = get_structure_adj(train_filtered)  # already (N, seq_len, 1)
train_inputs = np.ascontiguousarray(
    np.concatenate([train_inputs, Ss], 2), dtype=np.float32
)

print("train_inputs:", train_inputs.shape)
print("train_labels:", train_labels.shape)




## === cell 21
preprocess_inputs(train_data.loc[[0]]).shape




## === cell 22
test_data.head()




## === cell 23
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

    model.compile(optimizer=adam, loss=MCRMSE, jit_compile=True)
    return model




## === cell 24
SEQ_LEN = train_inputs.shape[1]
NUM_FEATURES = train_inputs.shape[2]
PRED_LEN = train_labels.shape[1]

EPOCHS = 60
BATCH_SIZE = 32

options = tf.data.Options()
options.deterministic = True

train_ds = tf.data.Dataset.from_tensor_slices((train_inputs, train_labels))
train_ds = train_ds.with_options(options)
train_ds = train_ds.cache()
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)


class SaveWeightsAtEnd(tf.keras.callbacks.Callback):
    def __init__(self, path):
        super().__init__()
        self.path = path

    def on_train_end(self, logs=None):
        self.model.save_weights(self.path)


STEPS_PER_EPOCH = int(np.ceil(len(train_inputs) / BATCH_SIZE))
SPE = min(32, STEPS_PER_EPOCH)  # same math/updates, fewer python/dispatch overhead
print("steps_per_epoch:", STEPS_PER_EPOCH, "steps_per_execution:", SPE)

x0, y0 = next(iter(train_ds.take(1)))




## === cell 25
model_GRU_on_train_data = build_model(
    seq_len=SEQ_LEN, num_features=NUM_FEATURES, pred_len=PRED_LEN, gru_flag=True
)
model_GRU_on_train_data.summary()

model_GRU_on_train_data.compile(
    optimizer=model_GRU_on_train_data.optimizer,
    loss=model_GRU_on_train_data.loss,
    jit_compile=True,
    steps_per_execution=SPE,
)

_ = model_GRU_on_train_data.train_on_batch(x0, y0)

history_GRU = model_GRU_on_train_data.fit(
    train_ds,
    epochs=EPOCHS,
    verbose=2,
    callbacks=[SaveWeightsAtEnd("GRU_model.weights.h5")],
)




## === cell 26
model_LSTM_on_train_data = build_model(
    seq_len=SEQ_LEN, num_features=NUM_FEATURES, pred_len=PRED_LEN, gru_flag=False
)
model_LSTM_on_train_data.summary()

model_LSTM_on_train_data.compile(
    optimizer=model_LSTM_on_train_data.optimizer,
    loss=model_LSTM_on_train_data.loss,
    jit_compile=True,
    steps_per_execution=SPE,
)

_ = model_LSTM_on_train_data.train_on_batch(x0, y0)

history_LSTM = model_LSTM_on_train_data.fit(
    train_ds,
    epochs=EPOCHS,
    verbose=2,
    callbacks=[SaveWeightsAtEnd("LSTM_model.weights.h5")],
)




## === cell 27
print(f" LSTM loss: {min(history_LSTM.history['loss'])}")
print(f" GRU loss: {min(history_GRU.history['loss'])}")




## === cell 28
seq_lens = sorted(test_data["seq_length"].unique().tolist())
print("Test seq_length unique:", seq_lens)

public_df = test_data.query("seq_length == 107").copy()
private_df = test_data.query("seq_length != 107").copy()  # may be empty

public_inputs = preprocess_inputs(public_df)
Ss_pub = get_structure_adj(public_df)  # already (N, L, 1)
public_inputs = np.ascontiguousarray(
    np.concatenate([public_inputs, Ss_pub], 2), dtype=np.float32
)

if len(private_df) > 0:
    private_inputs = preprocess_inputs(private_df)
    Ss_pri = get_structure_adj(private_df)
    private_inputs = np.ascontiguousarray(
        np.concatenate([private_inputs, Ss_pri], 2), dtype=np.float32
    )
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

pred_test_data_public_LSTM = model_LSTM_on_train_data.predict(
    public_inputs, batch_size=64, verbose=0
)
pred_test_data_public_GRU = model_GRU_on_train_data.predict(
    public_inputs, batch_size=64, verbose=0
)

if private_inputs is not None:
    pred_test_data_private_LSTM = model_LSTM_on_train_data.predict(
        private_inputs, batch_size=64, verbose=0
    )
    pred_test_data_private_GRU = model_GRU_on_train_data.predict(
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


def build_submission_in_sample_order(
    pred_public, pred_private, w_public_df, w_private_df
):
    id_to_preds = {}

    if len(w_public_df) > 0:
        seq_len = int(w_public_df["seq_length"].iloc[0])
        preds_seq = _pad_to_seq_len(pred_public, seq_len=seq_len)
        for mol_id, arr in zip(w_public_df["id"].astype(str).to_numpy(), preds_seq):
            id_to_preds[mol_id] = arr

    if len(w_private_df) > 0:
        seq_len = int(w_private_df["seq_length"].iloc[0])
        preds_seq = _pad_to_seq_len(pred_private, seq_len=seq_len)
        for mol_id, arr in zip(w_private_df["id"].astype(str).to_numpy(), preds_seq):
            id_to_preds[mol_id] = arr

    out = submission_format[["id_seqpos"]].copy()
    preds_flat = np.zeros((len(out), 5), dtype=np.float32)

    ids_pos = out["id_seqpos"].astype(str).to_numpy()
    for i, s in enumerate(ids_pos):
        mol_id, pos_str = s.rsplit("_", 1)
        preds_flat[i] = id_to_preds[mol_id][int(pos_str)]
    out[target_cols] = preds_flat
    return out


submission_LSTM = build_submission_in_sample_order(
    pred_test_data_public_LSTM, pred_test_data_private_LSTM, public_df, private_df
)
submission_GRU = build_submission_in_sample_order(
    pred_test_data_public_GRU, pred_test_data_private_GRU, public_df, private_df
)

print(submission_LSTM.shape, submission_GRU.shape)
submission_LSTM.head()




## === cell 31
submission_lstm_gru_combined = submission_format[["id_seqpos"]].copy()
gru_weight = 0.6
lstm_weight = 0.4

combined = (submission_GRU[target_cols].to_numpy(dtype=np.float32) * gru_weight) + (
    submission_LSTM[target_cols].to_numpy(dtype=np.float32) * lstm_weight
)
submission_lstm_gru_combined[target_cols] = combined
submission_lstm_gru_combined.head()




## === cell 32
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
