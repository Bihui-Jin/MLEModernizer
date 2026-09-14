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

# 5. Code solution

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



## === cell 1

from torch.utils.data import Dataset
from torch.utils.data import DataLoader as TorchDataLoader


class SimpleData:
    """Minimal container mimicking the attributes used in the original code."""

    def __init__(self, x, edge_index, edge_attr, y, train_mask):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y
        self.train_mask = train_mask

    def to(self, device):
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.edge_attr = self.edge_attr.to(device)
        self.y = self.y.to(device)
        self.train_mask = self.train_mask.to(device)
        return self

    @property
    def num_nodes(self):
        return self.x.shape[0]


def simple_collate(batch):
    assert len(batch) == 1, "This collate expects batch_size=1"
    return batch[0]




## === cell 2
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




## === cell 3
DATA_DIR = "/kaggle/input/stanford-covid-vaccine"
TRAIN_JSON = os.path.join(DATA_DIR, "train.json")
TEST_JSON = os.path.join(DATA_DIR, "test.json")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")

BPPS_DIR = os.path.join(DATA_DIR, "bpps")  # may not exist

assert os.path.exists(TRAIN_JSON), f"Missing: {TRAIN_JSON}"
assert os.path.exists(TEST_JSON), f"Missing: {TEST_JSON}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"




## === cell 4
def get_couples(structure: str):
    """
    Original logic preserved; minor robustness: handle empty or malformed structures gracefully.
    """
    opened = [idx for idx, ch in enumerate(structure) if ch == "("]
    closed = [idx for idx, ch in enumerate(structure) if ch == ")"]

    if len(opened) != len(closed):
        return []

    assigned = set()
    couples = []

    for close_idx in closed:
        candidate = None
        for open_idx in opened:
            if open_idx < close_idx:
                if open_idx not in assigned:
                    candidate = open_idx
            else:
                break
        if candidate is None:
            continue
        assigned.add(candidate)
        assigned.add(close_idx)
        couples.append([candidate, close_idx])
        couples.append([close_idx, candidate])

    return couples


def build_matrix(couples, size: int):
    mat = np.zeros((size, size), dtype=np.float32)

    for i in range(size):
        if i < size - 1:
            mat[i, i + 1] = 1.0
        if i > 0:
            mat[i, i - 1] = 1.0

    for i, j in couples:
        if 0 <= i < size and 0 <= j < size:
            mat[i, j] = 2.0
            mat[j, i] = 2.0

    return mat


def seq2nodes(sequence: str, loops: str, structures: str):
    type_dict = {"A": 0, "G": 1, "U": 2, "C": 3}
    loop_dict = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}
    struct_dict = {".": 0, "(": 1, ")": 2}

    nodes = np.zeros((len(sequence), 4 + 7 + 3), dtype=np.float32)

    for i, s in enumerate(sequence):
        if s in type_dict:
            nodes[i, type_dict[s]] = 1.0

    for i, s in enumerate(loops):
        if i >= len(sequence):
            break
        if s in loop_dict:
            nodes[i, 4 + loop_dict[s]] = 1.0

    for i, s in enumerate(structures):
        if i >= len(sequence):
            break
        if s in struct_dict:
            nodes[i, 11 + struct_dict[s]] = 1.0

    return nodes


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
    return np.zeros((seq_len, seq_len), dtype=np.float32)




## === cell 5
all_data = pd.read_json(TRAIN_JSON, lines=True)

if config.filter_noise and "signal_to_noise" in all_data.columns:
    all_data = all_data[all_data.signal_to_noise > 1].reset_index(drop=True)

all_data = all_data.query("seq_length == 107").reset_index(drop=True)
print("train rows:", len(all_data))



## === cell 6


class MyOwnDataset(Dataset):
    def __init__(self, train=True, ids=None):
        self.train = train
        self.data_dir = TRAIN_JSON if self.train else TEST_JSON
        self.df = pd.read_json(self.data_dir, lines=True)

        if self.train and config.filter_noise and "signal_to_noise" in self.df.columns:
            self.df = self.df[self.df.signal_to_noise > 1]

        self.df = self.df.query("seq_length == 107").reset_index(drop=True)

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

    def __len__(self):
        return len(self.df)

    @property
    def num_node_features(self):
        return self._num_node_features

    @property
    def num_edge_features(self):
        return self._num_edge_features

    def __getitem__(self, idx):
        structure = self.df["structure"].iloc[idx]
        sequence = self.df["sequence"].iloc[idx]
        loops = self.df["predicted_loop_type"].iloc[idx]
        rna_id = self.df["id"].iloc[idx]

        L = len(sequence)
        matrix = build_matrix(get_couples(structure), L)
        bpps = load_bpps_or_zeros(rna_id, L)

        edge_index = np.stack(np.where(matrix > 0)).astype(np.int64)  # (2, E)

        node_attr = seq2nodes(sequence, loops, structure)
        edge_attr = np.zeros((edge_index.shape[1], 3), dtype=np.float32)
        edge_attr[:, 0] = (matrix == 1)[edge_index[0, :], edge_index[1, :]].astype(
            np.float32
        )
        edge_attr[:, 1] = (matrix == 2)[edge_index[0, :], edge_index[1, :]].astype(
            np.float32
        )
        edge_attr[:, 2] = bpps[edge_index[0, :], edge_index[1, :]].astype(np.float32)

        if self.train:
            targets = np.stack(self.df[self.target_cols].iloc[idx]).T.astype(
                np.float32
            )  # (68,5)
            full_targets = np.zeros((L, 5), dtype=np.float32)
            full_targets[: targets.shape[0], :] = targets
            targets = full_targets
        else:
            targets = np.zeros((L, 5), dtype=np.float32)

        x = torch.from_numpy(node_attr).float()
        y = torch.from_numpy(targets).float()
        eattr = torch.from_numpy(edge_attr).float()
        eidx = torch.from_numpy(edge_index).long()

        train_mask = torch.zeros(L, dtype=torch.bool)
        train_mask[:68] = True

        return SimpleData(
            x=x, edge_index=eidx, edge_attr=eattr, y=y, train_mask=train_mask
        )




## === cell 7
all_ids = all_data["index"].values.copy()
np.random.shuffle(all_ids)
train_ids, val_ids = np.split(all_ids, [int(round(0.9 * len(all_ids), 0))])

train_dataset = MyOwnDataset(ids=train_ids, train=True)
val_dataset = MyOwnDataset(ids=val_ids, train=True)

train_loader = TorchDataLoader(
    train_dataset, batch_size=1, shuffle=True, collate_fn=simple_collate
)
val_loader = TorchDataLoader(
    val_dataset, batch_size=1, shuffle=False, collate_fn=simple_collate
)

print("train graphs:", len(train_dataset), "val graphs:", len(val_dataset))
print(
    "node_feats:",
    train_dataset.num_node_features,
    "edge_feats:",
    train_dataset.num_edge_features,
)



## === cell 8
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

    def forward(self, x, edge_index, edge_attr):
        src = edge_index[0]
        dst = edge_index[1]
        E = edge_attr.shape[0]
        C = x.shape[1]

        w = self.nn_edge(edge_attr).view(E, C, C)  # (E, C, C)
        msg = torch.bmm(x[src].unsqueeze(1), w).squeeze(1)  # (E, C)

        out = torch.zeros_like(x)
        out.index_add_(0, dst, msg)

        if self.aggr == "mean":
            deg = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
            deg.index_add_(0, dst, torch.ones(E, device=x.device, dtype=x.dtype))
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

        for _ in range(self.loops):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
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




## === cell 9
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




## === cell 10
model, train_loss, val_loss = train_loop(model, epochs=50)



## === cell 11
test_dataset = MyOwnDataset(train=False)
test_loader = TorchDataLoader(
    test_dataset, batch_size=1, shuffle=False, collate_fn=simple_collate
)
print("test graphs:", len(test_dataset))


@torch.no_grad()
def get_preds_per_graph(pred_loader):
    model.eval()
    preds = []
    for data in pred_loader:
        data = data.to(device)
        out = model(data).detach().cpu().numpy()  # (L, 5)
        preds.append(out)
    return preds


test_preds_list = get_preds_per_graph(test_loader)



## === cell 12
test_df = (
    pd.read_json(TEST_JSON, lines=True)
    .query("seq_length == 107")
    .reset_index(drop=True)
)
sample_df = pd.read_csv(SAMPLE_SUB)

pred_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

preds_ls = []
for i, uid in enumerate(test_df["id"].values):
    seq = test_df["sequence"].iloc[i]
    L = len(seq)  # expected 107
    pred = test_preds_list[i]
    if pred.shape[0] != L:
        if pred.shape[0] > L:
            pred = pred[:L, :]
        else:
            pad = np.zeros((L - pred.shape[0], pred.shape[1]), dtype=pred.dtype)
            pred = np.vstack([pred, pad])

    single_df = pd.DataFrame(pred, columns=pred_cols)
    single_df["id_seqpos"] = [f"{uid}_{x}" for x in range(L)]
    preds_ls.append(single_df)

preds_df = pd.concat(preds_ls, axis=0, ignore_index=True)

submission = sample_df[["id_seqpos"]].merge(preds_df, on="id_seqpos", how="left")

for c in pred_cols:
    submission[c] = submission[c].fillna(0.0).astype(np.float32)

submission.to_csv("submission.csv", index=False)
print("wrote submission.csv with shape:", submission.shape)
print(submission.head())

assert (
    submission.shape[0] == sample_df.shape[0]
), "Row count mismatch vs sample_submission"
assert list(submission.columns) == ["id_seqpos"] + pred_cols, "Column mismatch"
assert submission["id_seqpos"].is_unique, "id_seqpos must be unique"
print("Submission OK.")
