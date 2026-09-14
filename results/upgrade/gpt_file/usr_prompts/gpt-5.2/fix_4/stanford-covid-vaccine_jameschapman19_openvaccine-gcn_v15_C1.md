# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.56679

# 6. Current score

0.47468

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.67497) has done: 'I remove the hard dependency on `torch_geometric` (not installed in this environment) by providing a small, compatible fallback that mimics the needed `Data`, `InMemoryDataset`, `DataLoader`, and `NNConv` behaviors using plain PyTorch. I also fix the downstream `NameError`s that were caused by the early crash (paths/datasets/model never got defined) and ensure the pipeline runs end-to-end. To keep the core model logic intact, the message passing still uses learned edge-conditioned weights and the same GRU update loop; the only change is implementing that operator without `torch_geometric`. Finally, I ensure `submission.csv` is produced with the exact required columns and row order aligned to `sample_submission.csv`.'
- What this solution (achieved 0.47468) has done: 'I fix the shape mismatch that causes training to crash by ensuring each graph’s target tensor `y` always has length `seq_length` (107) while still masking loss to the first `seq_scored` positions. This preserves the core model and loss semantics, but aligns tensor shapes so boolean masking works. I also make the split reproducible by using the configured seed when shuffling ids (score-neutral but improves stability). Finally, I keep the submission-building logic intact and ensure the CSV is always written with the correct columns/order.'

# 9. Code solution

## === cell 0
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())



## === cell 1
try:
    import torch_geometric  # noqa: F401

    HAS_PYG = True
except Exception:
    HAS_PYG = False

print("HAS_PYG:", HAS_PYG)

import shutil
from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT_CANDIDATES = [
    "/kaggle/input/stanford-covid-vaccine",
    "/kaggle/input",
    "../input/stanford-covid-vaccine",
    "../input",
    "/kaggle/data/stanford-covid-vaccine",
    "/kaggle/data",
    "../data/stanford-covid-vaccine",
    "../data",
]
DATA_ROOT = None
for cand in DATA_ROOT_CANDIDATES:
    if os.path.exists(os.path.join(cand, "train.json")) and os.path.exists(
        os.path.join(cand, "test.json")
    ):
        DATA_ROOT = cand
        break
if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not locate train.json/test.json under expected Kaggle input paths."
    )

TRAIN_PATH = os.path.join(DATA_ROOT, "train.json")
TEST_PATH = os.path.join(DATA_ROOT, "test.json")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

print("Using DATA_ROOT:", DATA_ROOT)
print("TRAIN_PATH:", TRAIN_PATH)
print("TEST_PATH:", TEST_PATH)
print("SAMPLE_SUB_PATH:", SAMPLE_SUB_PATH)


class Data:
    def __init__(self, x, edge_index, edge_attr, y):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y
        self.train_mask = None

    @property
    def num_nodes(self):
        return int(self.x.shape[0])

    def to(self, device):
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.edge_attr = self.edge_attr.to(device)
        self.y = self.y.to(device)
        if self.train_mask is not None:
            self.train_mask = self.train_mask.to(device)
        return self


class InMemoryDataset:
    def __init__(self, root="", transform=None, pre_transform=None):
        self.root = root or ""
        self.transform = transform
        self.pre_transform = pre_transform

        self._processed_dir = os.path.join(self.root, "processed") if self.root else ""
        if self.root:
            os.makedirs(self._processed_dir, exist_ok=True)

        if not self.root:
            self.data_list = None
        else:
            if not os.path.exists(self.processed_paths[0]):
                self.process()

    @property
    def processed_paths(self):
        if not self.root:
            return [""]
        return [os.path.join(self._processed_dir, self.processed_file_names)]

    def collate(self, data_list):
        return data_list, None

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        idxs = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(idxs)
        for i in idxs:
            yield self.dataset[int(i)]

    def __len__(self):
        return len(self.dataset)


class NNConv(nn.Module):
    """
    Pure PyTorch edge-conditioned convolution compatible with torch_geometric.nn.NNConv usage:
      m_i = sum_{j->i} W(e_{j,i}) x_j , aggregated by mean/sum
    edge_index is shape [2, E] with rows [src, dst].
    edge_attr is shape [E, edge_feats].
    """

    def __init__(self, in_channels, out_channels, nn_edge, aggr="mean"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nn_edge = nn_edge
        if aggr not in ("mean", "sum"):
            raise ValueError("aggr must be 'mean' or 'sum'")
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        src = edge_index[0].long()
        dst = edge_index[1].long()
        E = edge_attr.shape[0]
        N = x.shape[0]

        w = self.nn_edge(edge_attr).view(E, self.out_channels, self.in_channels)

        x_src = x[src]  # [E, in]
        msg = torch.bmm(w, x_src.unsqueeze(-1)).squeeze(-1)  # [E, out]

        out = x.new_zeros((N, self.out_channels))
        out.index_add_(0, dst, msg)

        if self.aggr == "mean":
            deg = x.new_zeros((N,))
            ones = x.new_ones((E,))
            deg.index_add_(0, dst, ones)
            out = out / deg.clamp_min(1.0).unsqueeze(-1)

        return out




## === cell 2
HAS_FORGI = False
HAS_RNA = False
try:
    import forgi.graph.bulge_graph as fgb  # noqa: F401
    import forgi.visual.mplotlib as fvm  # noqa: F401

    HAS_FORGI = True
except Exception:
    HAS_FORGI = False

try:
    import RNA  # noqa: F401

    HAS_RNA = True
except Exception:
    HAS_RNA = False

print("HAS_FORGI:", HAS_FORGI, "HAS_RNA:", HAS_RNA)




## === cell 3
def seed_all(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_all()




## === cell 4
class config:
    learning_rate = 0.001
    K = 1
    gcn_agg = "mean"
    filter_noise = True
    seed = 1234




## === cell 5
def get_couples(structure):
    """
    Robustly match parentheses in dot-bracket notation using a stack.
    Returns directed edges (i,j) for paired bases.
    """
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


def build_matrix(couples, size):
    mat = np.zeros((size, size), dtype=np.float64)

    for i in range(size):
        if i < size - 1:
            mat[i, i + 1] = 1.0
        if i > 0:
            mat[i, i - 1] = 1.0

    for i, j in couples:
        mat[i, j] = 2.0
        mat[j, i] = 2.0

    return mat


def seq2nodes(sequence, loops, structures):
    type_dict = {"A": 0, "G": 1, "U": 2, "C": 3}
    loop_dict = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}
    struct_dict = {".": 0, "(": 1, ")": 2}

    nodes = np.zeros((len(sequence), 4 + 7 + 3), dtype=np.float64)
    for i, s in enumerate(sequence):
        nodes[i, type_dict[s]] = 1.0
    for i, s in enumerate(loops):
        nodes[i, 4 + loop_dict.get(s, 6)] = 1.0
    for i, s in enumerate(structures):
        nodes[i, 11 + struct_dict.get(s, 0)] = 1.0
    return nodes




## === cell 6
all_data = pd.read_json(TRAIN_PATH, lines=True)
if config.filter_noise:
    all_data = all_data[all_data.signal_to_noise > 1].reset_index(drop=True)

print("Train rows after filter:", len(all_data))
all_data.head(2)




## === cell 7
def dotbracket_bpps(structure, size, paired_value=1.0):
    """
    Replacement for missing bpps/*.npy files.
    Build a simple symmetric matrix with 1.0 at paired positions, 0 otherwise.
    This keeps the original edge feature shape (3) used by the message passing NN.
    """
    bpps = np.zeros((size, size), dtype=np.float64)
    couples = get_couples(structure)
    for i, j in couples:
        bpps[i, j] = paired_value
    return bpps




## === cell 8
class MyOwnDataset(InMemoryDataset):
    def __init__(
        self,
        root="",
        train=True,
        public=True,
        ids=None,
        transform=None,
        pre_transform=None,
    ):
        if (
            root
            and os.path.isdir(root)
            and os.path.abspath(root).startswith(os.path.abspath("."))
        ):
            try:
                shutil.rmtree(root)
            except Exception:
                pass

        self.train = train
        self.data_dir = TRAIN_PATH if self.train else TEST_PATH
        self.df = pd.read_json(self.data_dir, lines=True)

        if self.train and config.filter_noise:
            self.df = self.df[self.df.signal_to_noise > 1]

        if ids is not None:
            self.df = self.df[self.df["index"].isin(ids)]

        if public:
            self.df = self.df.query("seq_length == 107")
        else:
            self.df = self.df.query("seq_length == 130")
            if len(self.df) == 0:
                self.df = pd.read_json(self.data_dir, lines=True).query(
                    "seq_length == 107"
                )

        self.df = self.df.reset_index(drop=True)
        self.target_cols = [
            "reactivity",
            "deg_Mg_pH10",
            "deg_Mg_50C",
            "deg_pH10",
            "deg_50C",
        ]

        super().__init__(root, transform, pre_transform)

        if self.root and os.path.exists(self.processed_paths[0]):
            obj = torch.load(self.processed_paths[0], weights_only=False)
            self.data_list = obj[0] if isinstance(obj, (tuple, list)) else obj
        else:
            if getattr(self, "data_list", None) is None:
                self.process()

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return "data.pt"

    def download(self):
        pass

    def process(self):
        data_list = []
        for idx in tqdm(
            range(len(self.df)), desc=f"Processing {'train' if self.train else 'test'}"
        ):
            structure = self.df["structure"].iloc[idx]
            sequence = self.df["sequence"].iloc[idx]
            loops = self.df["predicted_loop_type"].iloc[idx]

            size = len(sequence)
            matrix = build_matrix(get_couples(structure), size)
            bpps = dotbracket_bpps(structure, size, paired_value=1.0)

            edge_index = np.stack(np.where((matrix) > 0))
            node_attr = seq2nodes(sequence, loops, structure)

            edge_attr = np.zeros((edge_index.shape[1], 3), dtype=np.float64)
            edge_attr[:, 0] = (matrix == 1)[edge_index[0, :], edge_index[1, :]].astype(
                np.float64
            )
            edge_attr[:, 1] = (matrix == 2)[edge_index[0, :], edge_index[1, :]].astype(
                np.float64
            )
            edge_attr[:, 2] = bpps[edge_index[0, :], edge_index[1, :]].astype(
                np.float64
            )

            if self.train:
                scored = (
                    int(self.df["seq_scored"].iloc[idx])
                    if "seq_scored" in self.df.columns
                    else 68
                )
                targets_scored = np.stack(self.df[self.target_cols].iloc[idx]).T.astype(
                    np.float64
                )  # [scored, 5]
                targets = np.zeros((size, 5), dtype=np.float64)
                targets[:scored] = targets_scored
            else:
                scored = (
                    int(self.df["seq_scored"].iloc[idx])
                    if "seq_scored" in self.df.columns
                    else 68
                )
                targets = np.zeros((size, 5), dtype=np.float64)

            x = torch.from_numpy(node_attr).double()
            y = torch.from_numpy(targets).double()
            edge_attr_t = torch.from_numpy(edge_attr).double()
            edge_index_t = torch.tensor(edge_index, dtype=torch.long)

            data = Data(x=x, edge_index=edge_index_t, edge_attr=edge_attr_t, y=y)
            data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            data.train_mask[:scored] = True

            data_list.append(data)

        if self.root:
            torch.save((data_list, None), self.processed_paths[0])
        self.data_list = data_list

    @property
    def num_node_features(self):
        return int(self.data_list[0].x.shape[1])

    @property
    def num_edge_features(self):
        return int(self.data_list[0].edge_attr.shape[1])




## === cell 9
seed_all(config.seed)

all_ids = all_data["index"].values.copy()
np.random.shuffle(all_ids)
split = int(round(0.9 * len(all_ids), 0))
train_ids, val_ids = np.split(all_ids, [split])

train_dataset = MyOwnDataset(ids=train_ids, root="train_ds", train=True, public=True)
val_dataset = MyOwnDataset(ids=val_ids, root="val_ds", train=True, public=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

print("Train graphs:", len(train_dataset), "Val graphs:", len(val_dataset))
print(
    "Node feats:",
    train_dataset.num_node_features,
    "Edge feats:",
    train_dataset.num_edge_features,
)



## === cell 10
from torch.nn import Sequential, Linear, ReLU, GRU


class MPNNet(torch.nn.Module):
    def __init__(self, node_feats, channels, out_feats, loops=1, edge_feats=1):
        super(MPNNet, self).__init__()
        self.lin0 = torch.nn.Linear(node_feats, channels)
        self.loops = loops
        nn_edge = Sequential(
            Linear(edge_feats, 5), ReLU(), Linear(5, channels * channels)
        )
        self.conv = NNConv(channels, channels, nn_edge, aggr="mean")
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
out_feats = 5  # fixed by target cols
edge_feats = train_dataset.num_edge_features

model = (
    MPNNet(node_feats, 100, out_feats, loops=3, edge_feats=edge_feats)
    .double()
    .to(device)
)
print("Params:", sum(p.numel() for p in model.parameters()))




## === cell 11
class MCRMSELoss(torch.nn.Module):
    def __init__(self):
        super(MCRMSELoss, self).__init__()

    def forward(self, x, y):
        x = x[:, :3]
        y = y[:, :3]
        msq_error = torch.mean((x - y) ** 2, 0)
        loss = torch.mean(torch.sqrt(msq_error))
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
        self.avg = self.sum / max(1, self.count)


loss_fn = MCRMSELoss()


def train_one_epoch(model, optimizer, train_loader):
    model.train()
    train_loss = AverageMeter()
    for data in train_loader:
        data = data.to(device)
        out = model(data)
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        train_loss.update(loss.item())
    return train_loss.avg


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    val_loss = AverageMeter()
    for data in val_loader:
        data = data.to(device)
        out = model(data)
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        val_loss.update(loss.item())
    return val_loss.avg




## === cell 12
def train_loop(model, epochs=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train_loss_hist, val_loss_hist = [], []
    for epoch in range(1, epochs + 1):
        tr = train_one_epoch(model, optimizer, train_loader)
        va = evaluate(model, val_loader)
        train_loss_hist.append(tr)
        val_loss_hist.append(va)
        print(f"Epoch: {epoch:03d}, Train Loss: {tr:.4f}, Val Loss: {va:.4f}")
    return model, train_loss_hist, val_loss_hist


model, train_loss, val_loss = train_loop(model, epochs=50)



## === cell 13
test_df = pd.read_json(TEST_PATH, lines=True)
public_df = test_df.query("seq_length == 107").reset_index(drop=True)
private_df = test_df.query("seq_length == 130").reset_index(drop=True)

public_ds = MyOwnDataset(root="public_ds", train=False, public=True)
public_loader = DataLoader(public_ds, batch_size=8, shuffle=False)

private_preds = None
if len(private_df) > 0:
    private_ds = MyOwnDataset(root="private_ds", train=False, public=False)
    private_loader = DataLoader(private_ds, batch_size=8, shuffle=False)
else:
    private_loader = None


@torch.no_grad()
def get_preds(pred_loader):
    model.eval()
    all_out = []
    for data in pred_loader:
        data = data.to(device)
        out = model(data)
        all_out.append(out.detach().cpu())
    return torch.cat(all_out, dim=0).numpy()


public_preds = get_preds(public_loader)
if private_loader is not None:
    private_preds = get_preds(private_loader)

print(
    "public_preds shape:",
    public_preds.shape,
    "private_preds:",
    None if private_preds is None else private_preds.shape,
)



## === cell 14
sample_df = pd.read_csv(SAMPLE_SUB_PATH)
pred_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]


def pack_preds(df, preds):
    offset = 0
    out_rows = []
    for i, uid in enumerate(df.id):
        seq = df.sequence.iloc[i]
        L = len(seq)
        single_pred = preds[offset : offset + L, :]
        offset += L

        single_df = pd.DataFrame(single_pred, columns=pred_cols)
        single_df["id_seqpos"] = [f"{uid}_{x}" for x in range(L)]
        out_rows.append(single_df)
    return pd.concat(out_rows, axis=0, ignore_index=True)


preds_df = pack_preds(public_df, public_preds)
if private_preds is not None and len(private_df) > 0:
    preds_df = pd.concat(
        [preds_df, pack_preds(private_df, private_preds)], axis=0, ignore_index=True
    )

submission = sample_df[["id_seqpos"]].merge(preds_df, on="id_seqpos", how="left")

for c in pred_cols:
    submission[c] = submission[c].astype(np.float64).fillna(0.0)

submission = submission[["id_seqpos"] + pred_cols]
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print("Columns:", submission.columns.tolist())
submission.head(5)
