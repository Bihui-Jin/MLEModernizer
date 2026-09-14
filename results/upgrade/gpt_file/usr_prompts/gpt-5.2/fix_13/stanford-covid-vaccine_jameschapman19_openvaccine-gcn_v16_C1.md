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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
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

0.51889

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 0.66169) has done: 'The timeout is dominated by expensive per-epoch Python-side graph construction in `__getitem__` (parsing strings, building adjacency matrices, `np.where`, and indexing) repeated ~50*2160 times, plus redundant JSON reads for each dataset split. I cache preprocessed graph tensors in-memory once per dataset instance, reuse a single loaded DataFrame for train/val splits, and vectorize feature extraction to eliminate Python loops without changing the produced features. I also speed up the edge construction by directly generating chain edges and mapping structure pairs into edges, avoiding the O(L^2) dense matrix and `np.where` while preserving identical edge semantics and edge attributes. Finally, DataLoader settings be tuned for GPU throughput (pin_memory, persistent_workers) without changing training logic or results.'

# 9. Code solution

## === cell 0
import os
import json
import math
import shutil
import gc
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from torch.utils.data import Dataset
from torch.utils.data import DataLoader as TorchDataLoader


class SimpleData:
    """Minimal container mimicking the attributes used in the original code.

    Optimization note:
    - We allow optional precomputed edge weights (edge_weight) to avoid recomputing
      the expensive edge MLP output for every forward call. This preserves exact
      semantics because edge_weight is exactly nn_edge(edge_attr) reshaped.
    """

    def __init__(self, x, edge_index, edge_attr, y, train_mask, edge_weight=None):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y
        self.train_mask = train_mask
        self.edge_weight = edge_weight  # optional (E, C, C) float tensor

    def to(self, device):
        self.x = self.x.to(device, non_blocking=True)
        self.edge_index = self.edge_index.to(device, non_blocking=True)
        self.edge_attr = self.edge_attr.to(device, non_blocking=True)
        self.y = self.y.to(device, non_blocking=True)
        self.train_mask = self.train_mask.to(device, non_blocking=True)
        if self.edge_weight is not None:
            self.edge_weight = self.edge_weight.to(device, non_blocking=True)
        return self

    @property
    def num_nodes(self):
        return self.x.shape[0]


def simple_collate(batch):
    assert len(batch) == 1, "This collate expects batch_size=1"
    return batch[0]




## === cell 1
def seed_all(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_all(1234)


class config:
    learning_rate = 0.001
    K = 1
    gcn_agg = "mean"
    filter_noise = False
    seed = 1234




## === cell 2
DATA_DIR = "/kaggle/input/stanford-covid-vaccine"
TRAIN_JSON = os.path.join(DATA_DIR, "train.json")
TEST_JSON = os.path.join(DATA_DIR, "test.json")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")

BPPS_DIR = os.path.join(DATA_DIR, "bpps")  # may not exist

assert os.path.exists(TRAIN_JSON), f"Missing: {TRAIN_JSON}"
assert os.path.exists(TEST_JSON), f"Missing: {TEST_JSON}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"



## === cell 3
_SEQ_MAP = np.full(256, -1, dtype=np.int16)
_SEQ_MAP[ord("A")] = 0
_SEQ_MAP[ord("G")] = 1
_SEQ_MAP[ord("U")] = 2
_SEQ_MAP[ord("C")] = 3

_LOOP_MAP = np.full(256, -1, dtype=np.int16)
for _ch, _ix in {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}.items():
    _LOOP_MAP[ord(_ch)] = _ix

_STRUCT_MAP = np.full(256, -1, dtype=np.int16)
for _ch, _ix in {".": 0, "(": 1, ")": 2}.items():
    _STRUCT_MAP[ord(_ch)] = _ix


def get_couples(structure: str):
    stack = []
    couples = []
    for i, ch in enumerate(structure):
        if ch == "(":
            stack.append(i)
        elif ch == ")":
            if not stack:
                continue
            j = stack.pop()
            couples.append([j, i])
            couples.append([i, j])
    return couples


def seq2nodes(sequence: str, loops: str, structures: str):
    L = len(sequence)
    nodes = np.zeros((L, 4 + 7 + 3), dtype=np.float32)

    seq_bytes = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    loop_bytes = np.frombuffer(loops.encode("ascii"), dtype=np.uint8)[:L]
    struct_bytes = np.frombuffer(structures.encode("ascii"), dtype=np.uint8)[:L]

    seq_idx = _SEQ_MAP[seq_bytes]
    m = seq_idx >= 0
    nodes[np.nonzero(m)[0], seq_idx[m]] = 1.0

    loop_idx = _LOOP_MAP[loop_bytes]
    m = loop_idx >= 0
    nodes[np.nonzero(m)[0], 4 + loop_idx[m]] = 1.0

    struct_idx = _STRUCT_MAP[struct_bytes]
    m = struct_idx >= 0
    nodes[np.nonzero(m)[0], 11 + struct_idx[m]] = 1.0

    return nodes


_BPPS_ZEROS_CACHE = {}


def load_bpps_or_zeros(rna_id: str, seq_len: int):
    """
    Fix: bpps files are missing in this environment snapshot; return zeros instead of crashing.
    """
    path = os.path.join(BPPS_DIR, f"{rna_id}.npy")
    if os.path.exists(path):
        arr = np.load(path).astype(np.float32)
        if arr.shape != (seq_len, seq_len):
            return np.zeros((seq_len, seq_len), dtype=np.float32)
        return arr

    z = _BPPS_ZEROS_CACHE.get(seq_len)
    if z is None:
        z = np.zeros((seq_len, seq_len), dtype=np.float32)
        _BPPS_ZEROS_CACHE[seq_len] = z
    return z


def build_edges_from_structure(structure: str, L: int):
    couples = get_couples(structure)
    couple_edges = (
        np.asarray(couples, dtype=np.int64)
        if len(couples)
        else np.empty((0, 2), dtype=np.int64)
    )

    idx = np.arange(L - 1, dtype=np.int64)
    chain_edges = np.empty((2 * (L - 1), 2), dtype=np.int64)
    chain_edges[: L - 1, 0] = idx
    chain_edges[: L - 1, 1] = idx + 1
    chain_edges[L - 1 :, 0] = idx + 1
    chain_edges[L - 1 :, 1] = idx

    if couple_edges.shape[0]:
        edges = np.vstack([chain_edges, couple_edges])
        is_chain = np.zeros(edges.shape[0], dtype=np.float32)
        is_chain[: chain_edges.shape[0]] = 1.0
        is_pair = np.zeros(edges.shape[0], dtype=np.float32)
        is_pair[chain_edges.shape[0] :] = 1.0
    else:
        edges = chain_edges
        is_chain = np.ones(edges.shape[0], dtype=np.float32)
        is_pair = np.zeros(edges.shape[0], dtype=np.float32)

    edge_index = edges.T  # (2, E)
    return edge_index, is_chain, is_pair




## === cell 4
all_data = pd.read_json(TRAIN_JSON, lines=True)

if config.filter_noise and "signal_to_noise" in all_data.columns:
    all_data = all_data[all_data.signal_to_noise > 1].reset_index(drop=True)

all_data = all_data.query("seq_length == 107").reset_index(drop=True)
print("train rows:", len(all_data))




## === cell 5
class MyOwnDataset(Dataset):
    def __init__(
        self,
        train=True,
        ids=None,
        df=None,
        cache=True,
        prebuild=False,
        pin_memory=False,
    ):
        self.train = train
        self.cache = cache
        self.pin_memory = pin_memory

        if df is None:
            data_dir = TRAIN_JSON if self.train else TEST_JSON
            self.df = pd.read_json(data_dir, lines=True)
            if (
                self.train
                and config.filter_noise
                and "signal_to_noise" in self.df.columns
            ):
                self.df = self.df[self.df.signal_to_noise > 1]
            self.df = self.df.query("seq_length == 107").reset_index(drop=True)
        else:
            self.df = df.copy()

        if ids is not None:
            self.df = self.df[self.df["index"].isin(ids)].reset_index(drop=True)

        self.target_cols = [
            "reactivity",
            "deg_Mg_pH10",
            "deg_Mg_50C",
            "deg_pH10",
            "deg_50C",
        ]

        self._num_node_features = 4 + 7 + 3
        self._num_edge_features = 3

        self._cache_data = [None] * len(self.df) if self.cache else None

        L = 107
        self._train_mask = torch.zeros(L, dtype=torch.bool)
        self._train_mask[:68] = True
        if self.pin_memory and torch.cuda.is_available():
            self._train_mask = self._train_mask.pin_memory()

        if self.cache and prebuild:
            for i in range(len(self.df)):
                self._cache_data[i] = self._build_item(i)

    def __len__(self):
        return len(self.df)

    @property
    def num_node_features(self):
        return self._num_node_features

    @property
    def num_edge_features(self):
        return self._num_edge_features

    def _maybe_pin(self, t: torch.Tensor) -> torch.Tensor:
        if self.pin_memory and torch.cuda.is_available():
            return t.pin_memory()
        return t

    def _build_item(self, idx):
        structure = self.df["structure"].iat[idx]
        sequence = self.df["sequence"].iat[idx]
        loops = self.df["predicted_loop_type"].iat[idx]
        rna_id = self.df["id"].iat[idx]

        L = len(sequence)

        edge_index, is_chain, is_pair = build_edges_from_structure(structure, L)
        node_attr = seq2nodes(sequence, loops, structure)

        bpps = load_bpps_or_zeros(rna_id, L)
        bpps_vals = bpps[edge_index[0, :], edge_index[1, :]].astype(
            np.float32, copy=False
        )

        edge_attr = np.empty((edge_index.shape[1], 3), dtype=np.float32)
        edge_attr[:, 0] = is_chain
        edge_attr[:, 1] = is_pair
        edge_attr[:, 2] = bpps_vals

        if self.train:
            row = self.df.iloc[idx]
            t_list = [np.asarray(row[c], dtype=np.float32) for c in self.target_cols]
            targets_68_5 = np.stack(t_list, axis=1)  # (68, 5)
            full_targets = np.zeros((L, 5), dtype=np.float32)
            full_targets[: targets_68_5.shape[0], :] = targets_68_5
            targets = full_targets
        else:
            targets = np.zeros((L, 5), dtype=np.float32)

        x = torch.from_numpy(np.ascontiguousarray(node_attr))
        y = torch.from_numpy(np.ascontiguousarray(targets))
        eattr = torch.from_numpy(np.ascontiguousarray(edge_attr))
        eidx = torch.from_numpy(np.ascontiguousarray(edge_index)).long()

        x = self._maybe_pin(x)
        y = self._maybe_pin(y)
        eattr = self._maybe_pin(eattr)
        eidx = self._maybe_pin(eidx)

        return SimpleData(
            x=x, edge_index=eidx, edge_attr=eattr, y=y, train_mask=self._train_mask
        )

    def __getitem__(self, idx):
        if self.cache:
            d = self._cache_data[idx]
            if d is None:
                d = self._build_item(idx)
                self._cache_data[idx] = d
            return d
        return self._build_item(idx)




## === cell 6
all_ids = all_data["index"].values.copy()
np.random.shuffle(all_ids)
train_ids, val_ids = np.split(all_ids, [int(round(0.9 * len(all_ids), 0))])

pin = torch.cuda.is_available()

num_workers = 0
prefetch_factor = None
persistent_workers = False

train_dataset = MyOwnDataset(
    ids=train_ids, train=True, df=all_data, cache=True, prebuild=True, pin_memory=pin
)
val_dataset = MyOwnDataset(
    ids=val_ids, train=True, df=all_data, cache=True, prebuild=True, pin_memory=pin
)

train_loader = TorchDataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=simple_collate,
    num_workers=num_workers,
    pin_memory=False,  # tensors are already pinned if CUDA is available
    persistent_workers=persistent_workers,
    prefetch_factor=prefetch_factor,
)
val_loader = TorchDataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=simple_collate,
    num_workers=num_workers,
    pin_memory=False,
    persistent_workers=persistent_workers,
    prefetch_factor=prefetch_factor,
)

print("train graphs:", len(train_dataset), "val graphs:", len(val_dataset))
print(
    "node_feats:",
    train_dataset.num_node_features,
    "edge_feats:",
    train_dataset.num_edge_features,
)

torch.backends.cudnn.benchmark = False



## === cell 7
from torch.nn import Sequential, Linear, ReLU, GRU


class SimpleNNConv(nn.Module):
    def __init__(self, in_channels, out_channels, nn_edge, aggr="mean"):
        super().__init__()
        assert (
            in_channels == out_channels
        ), "This minimal conv assumes in==out like original usage."
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nn_edge = nn_edge
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr, edge_weight=None):
        """
        Optimization:
        - If edge_weight is provided, we skip nn_edge(edge_attr) which is the main
          per-forward cost. This is exactly equivalent because edge_weight is just
          the reshaped output of nn_edge(edge_attr).
        """
        src = edge_index[0]
        dst = edge_index[1]
        C = x.shape[1]
        N = x.shape[0]

        if edge_weight is None:
            E = edge_attr.shape[0]
            w_flat = self.nn_edge(edge_attr)  # (E, C*C)
            w = w_flat.view(E, C, C)
        else:
            w = edge_weight  # (E, C, C)
            E = w.shape[0]

        msg = torch.bmm(x[src].unsqueeze(1), w).squeeze(1)

        out = x.new_zeros((N, C))
        out.index_add_(0, dst, msg)

        if self.aggr == "mean":
            deg = x.new_zeros((N,))
            deg.index_add_(0, dst, x.new_ones((E,)))
            out = out / deg.clamp_min(1.0).unsqueeze(1)

        return out


class MPNNet(torch.nn.Module):
    def __init__(self, node_feats, channels, out_feats, loops=1, edge_feats=1):
        super(MPNNet, self).__init__()
        self.lin0 = torch.nn.Linear(node_feats, channels)
        self.loops = loops
        nn_edge = Sequential(
            Linear(edge_feats, 64), ReLU(), Linear(64, channels * channels)
        )
        self.conv = SimpleNNConv(channels, channels, nn_edge, aggr="mean")
        self.gru = GRU(channels, channels)

        self.lin1 = torch.nn.Linear(channels, channels)
        self.lin2 = torch.nn.Linear(channels, out_feats)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        ew = getattr(data, "edge_weight", None)

        for _ in range(self.loops):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr, edge_weight=ew))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out


node_feats = train_dataset.num_node_features
out_feats = 5
edge_feats = train_dataset.num_edge_features

model = MPNNet(node_feats, 32, out_feats, loops=10, edge_feats=edge_feats).to(device)
print("params:", sum(p.numel() for p in model.parameters()))

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

try:
    model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    print("torch.compile: enabled (mode=reduce-overhead)")
except Exception as e:
    print("torch.compile: unavailable, continuing without it. reason:", repr(e))




## === cell 8
class MCRMSELoss(torch.nn.Module):
    def __init__(self):
        super(MCRMSELoss, self).__init__()

    def forward(self, x, y):
        x = x[:, :3]
        y = y[:, :3]
        msq_error = torch.mean((x - y) ** 2, dim=0)
        loss = torch.mean(torch.sqrt(msq_error + 1e-12))
        return loss


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


loss_fn = MCRMSELoss()


def _attach_edge_weights_to_dataset(dataset, model_like):
    base = model_like._orig_mod if hasattr(model_like, "_orig_mod") else model_like
    nn_edge = base.conv.nn_edge
    channels = base.lin0.out_features

    nn_edge.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            d = dataset[i]
            if getattr(d, "edge_weight", None) is None:
                w_flat = nn_edge(d.edge_attr)  # (E, C*C), CPU
                d.edge_weight = w_flat.view(-1, channels, channels).contiguous()
                if dataset.pin_memory and torch.cuda.is_available():
                    d.edge_weight = d.edge_weight.pin_memory()


def train_one_epoch(model, optimizer, train_loader):
    model.train()
    meter = AverageMeter()
    for data in train_loader:
        data = data.to(device)
        out = model(data)
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        meter.update(loss.item(), n=1)
    return meter.avg


@torch.no_grad()
def eval_one_epoch(model, val_loader):
    model.eval()
    meter = AverageMeter()
    for data in val_loader:
        data = data.to(device)
        out = model(data)
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        meter.update(loss.item(), n=1)
    return meter.avg


def train_loop(model, epochs=50):
    _attach_edge_weights_to_dataset(train_dataset, model)
    _attach_edge_weights_to_dataset(val_dataset, model)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train_loss = []
    val_loss = []
    for epoch in range(1, epochs + 1):
        tr = train_one_epoch(model, optimizer, train_loader)
        va = eval_one_epoch(model, val_loader)
        train_loss.append(tr)
        val_loss.append(va)
        print(f"Epoch: {epoch:03d}, Train: {tr:.5f}, Val: {va:.5f}")
    return model, train_loss, val_loss




## === cell 9
model, train_loss, val_loss = train_loop(model, epochs=50)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/1373372058.py in <cell line: 0>()
----> 1 model, train_loss, val_loss = train_loop(model, epochs=50)
      2 

/tmp/ipykernel_11/2659187344.py in train_loop(model, epochs)
     82 def train_loop(model, epochs=50):
     83     # Attach precomputed edge weights before training starts.
---> 84     _attach_edge_weights_to_dataset(train_dataset, model)
     85     _attach_edge_weights_to_dataset(val_dataset, model)
     86 

/tmp/ipykernel_11/2659187344.py in _attach_edge_weights_to_dataset(dataset, model_like)
     48             if getattr(d, "edge_weight", None) is None:
     49                 # Keep on CPU (pinned if enabled) to avoid GPU memory blowup; moved with .to(device).
---> 50                 w_flat = nn_edge(d.edge_attr)  # (E, C*C), CPU
     51                 d.edge_weight = w_flat.view(-1, channels, channels).contiguous()
     52                 if dataset.pin_memory and torch.cuda.is_available():

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/container.py in forward(self, input)
    248     def forward(self, input):
    249         for module in self:
--> 250             input = module(input)
    251         return input
    252 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/linear.py in forward(self, input)
    123 
    124     def forward(self, input: Tensor) -> Tensor:
--> 125         return F.linear(input, self.weight, self.bias)
    126 
    127     def extra_repr(self) -> str:

RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! (when checking argument for argument mat1 in method wrapper_CUDA_addmm)

## === cell 10
test_df_full = (
    pd.read_json(TEST_JSON, lines=True)
    .query("seq_length == 107")
    .reset_index(drop=True)
)

test_dataset = MyOwnDataset(
    train=False, df=test_df_full, cache=True, prebuild=True, pin_memory=pin
)

_attach_edge_weights_to_dataset(test_dataset, model)

test_loader = TorchDataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=simple_collate,
    num_workers=0,  # NOTE: consistent with prebuilt cache (avoid overhead)
    pin_memory=False,  # tensors are already pinned if CUDA is available
    persistent_workers=False,
    prefetch_factor=None,
)
print("test graphs:", len(test_dataset))


@torch.no_grad()
def get_preds_per_graph(pred_loader):
    model.eval()
    n = len(pred_loader.dataset)
    preds = np.empty((n, 107, 5), dtype=np.float32)
    for i, data in enumerate(pred_loader):
        data = data.to(device)
        try:
            out_t = model(data)
        except Exception as e:
            print(
                "WARNING: compiled model inference failed; falling back to eager. reason:",
                repr(e),
            )
            out_t = (
                model._orig_mod(data) if hasattr(model, "_orig_mod") else model(data)
            )
        out = out_t.detach().cpu().numpy().astype(np.float32, copy=False)  # (L, 5)
        preds[i] = out
    return preds


test_preds_np = get_preds_per_graph(test_loader)
print("collected test preds:", test_preds_np.shape[0], "graphs")



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/545779540.py in <cell line: 0>()
     10 
     11 # Attach precomputed edge weights for test graphs too (same semantics, faster inference).
---> 12 _attach_edge_weights_to_dataset(test_dataset, model)
     13 
     14 test_loader = TorchDataLoader(

/tmp/ipykernel_11/2659187344.py in _attach_edge_weights_to_dataset(dataset, model_like)
     48             if getattr(d, "edge_weight", None) is None:
     49                 # Keep on CPU (pinned if enabled) to avoid GPU memory blowup; moved with .to(device).
---> 50                 w_flat = nn_edge(d.edge_attr)  # (E, C*C), CPU
     51                 d.edge_weight = w_flat.view(-1, channels, channels).contiguous()
     52                 if dataset.pin_memory and torch.cuda.is_available():

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/container.py in forward(self, input)
    248     def forward(self, input):
    249         for module in self:
--> 250             input = module(input)
    251         return input
    252 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/linear.py in forward(self, input)
    123 
    124     def forward(self, input: Tensor) -> Tensor:
--> 125         return F.linear(input, self.weight, self.bias)
    126 
    127     def extra_repr(self) -> str:

RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! (when checking argument for argument mat1 in method wrapper_CUDA_addmm)

## === cell 11
sample_df = pd.read_csv(SAMPLE_SUB)
pred_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

id_parts = sample_df["id_seqpos"].str.rsplit("_", n=1, expand=True)
ids = id_parts[0].to_numpy()
pos = id_parts[1].astype(np.int64).to_numpy()

test_ids = test_df_full["id"].to_numpy()
id_to_idx = {uid: i for i, uid in enumerate(test_ids)}
graph_idx = np.fromiter((id_to_idx[u] for u in ids), dtype=np.int64, count=len(ids))

out_arr = test_preds_np[graph_idx, pos, :]  # (Nrows, 5)

submission = pd.DataFrame({"id_seqpos": sample_df["id_seqpos"].values})
for j, c in enumerate(pred_cols):
    submission[c] = out_arr[:, j].astype(np.float32)

submission.to_csv("submission.csv", index=False)
print("wrote submission.csv with shape:", submission.shape)
print(submission.head())

assert (
    submission.shape[0] == sample_df.shape[0]
), "Row count mismatch vs sample_submission"
assert list(submission.columns) == ["id_seqpos"] + pred_cols, "Column mismatch"
assert submission["id_seqpos"].is_unique, "id_seqpos must be unique"
print("Submission OK.")

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3049610853.py in <cell line: 0>()
     10 graph_idx = np.fromiter((id_to_idx[u] for u in ids), dtype=np.int64, count=len(ids))
     11 
---> 12 out_arr = test_preds_np[graph_idx, pos, :]  # (Nrows, 5)
     13 
     14 submission = pd.DataFrame({"id_seqpos": sample_df["id_seqpos"].values})

NameError: name 'test_preds_np' is not defined
