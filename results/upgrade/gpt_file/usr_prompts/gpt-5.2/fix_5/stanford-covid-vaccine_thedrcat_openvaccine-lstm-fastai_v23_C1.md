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

fastai==2.8.5
geopandas==0.14.4
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

0.37023

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

from fastai.data.core import DataLoaders
from fastai.learner import Learner
from fastai.callback.core import Callback
from fastai.torch_core import tensor

from fastai.callback.schedule import fit_one_cycle


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



## === cell 1
path = "/kaggle/input/stanford-covid-vaccine"
train = pd.read_json(f"{path}/train.json", lines=True)
test = pd.read_json(f"{path}/test.json", lines=True)
sub = pd.read_csv(f"{path}/sample_submission.csv")

train.shape, train["id"].nunique(), test.shape, sub.shape, sub.columns.tolist()



## === cell 2
(
    train["seq_length"].unique(),
    train["seq_scored"].unique(),
    test["seq_length"].unique(),
    test["seq_scored"].unique(),
)




## === cell 3
def _bpps_path(mol_id: str) -> str:
    return os.path.join(path, "bpps", f"{mol_id}.npy")


def read_bpps_sum(df):
    bpps_arr = []
    for mol_id in df.id.to_list():
        fp = _bpps_path(mol_id)
        if os.path.exists(fp):
            bpps_arr.append(np.load(fp).sum(axis=1))
        else:
            bpps_arr.append(np.zeros(107, dtype=np.float32))
    return bpps_arr


def read_bpps_max(df):
    bpps_arr = []
    for mol_id in df.id.to_list():
        fp = _bpps_path(mol_id)
        if os.path.exists(fp):
            bpps_arr.append(np.load(fp).max(axis=1))
        else:
            bpps_arr.append(np.zeros(107, dtype=np.float32))
    return bpps_arr


def read_bpps_nb(df):
    bpps_nb_mean = 0.077522  # from original notebook
    bpps_nb_std = 0.08914
    bpps_arr = []
    for mol_id in df.id.to_list():
        fp = _bpps_path(mol_id)
        if os.path.exists(fp):
            bpps = np.load(fp)
            bpps_nb = (bpps > 0).sum(axis=0) / bpps.shape[0]
            bpps_nb = (bpps_nb - bpps_nb_mean) / bpps_nb_std
            bpps_arr.append(bpps_nb.astype(np.float32))
        else:
            bpps_arr.append(np.zeros(107, dtype=np.float32))
    return bpps_arr


train["bpps_sum"] = read_bpps_sum(train)
test["bpps_sum"] = read_bpps_sum(test)
train["bpps_max"] = read_bpps_max(train)
test["bpps_max"] = read_bpps_max(test)
train["bpps_nb"] = read_bpps_nb(train)
test["bpps_nb"] = read_bpps_nb(test)

train[["bpps_sum", "bpps_max", "bpps_nb"]].iloc[0].apply(
    lambda x: (type(x), len(x))
).to_dict()



## === cell 4
train = train.sample(frac=1, random_state=42).reset_index(drop=True)
train.shape



## === cell 5
all1, all2, all3 = [], [], []
for i in range(len(train)):
    all1.extend(train["sequence"].iloc[i])
    all2.extend(train["structure"].iloc[i])
    all3.extend(train["predicted_loop_type"].iloc[i])

len(all1), len(all2), len(all3)



## === cell 6
vocab1 = list(pd.unique(all1))
vocab2 = list(pd.unique(all2))
vocab3 = list(pd.unique(all3))

word2idx1 = {w: i for i, w in enumerate(vocab1)}
word2idx2 = {w: i for i, w in enumerate(vocab2)}
word2idx3 = {w: i for i, w in enumerate(vocab3)}

len(vocab1), len(vocab2), len(vocab3)




## === cell 7
def joiner(row):
    l1 = list(row[0])
    l2 = list(row[1])
    l3 = list(row[2])
    l4 = list(row[3])
    l5 = list(row[4])
    l6 = list(row[5])
    out = [
        [word2idx1[l1[i]], word2idx2[l2[i]], word2idx3[l3[i]], l4[i], l5[i], l6[i]]
        for i in range(len(l1))
    ]
    return out


train["seqs"] = train[
    ["sequence", "structure", "predicted_loop_type", "bpps_sum", "bpps_max", "bpps_nb"]
].apply(joiner, axis=1)
test["seqs"] = test[
    ["sequence", "structure", "predicted_loop_type", "bpps_sum", "bpps_max", "bpps_nb"]
].apply(joiner, axis=1)

train["seqs"].iloc[0][0], len(train["seqs"].iloc[0])



## === cell 8
train = train[train["SN_filter"] == 1].reset_index(drop=True)
train.shape



## === cell 9
txts = train["seqs"].values
tgts1 = train["reactivity"].values
tgts2 = train["deg_Mg_pH10"].values
tgts3 = train["deg_pH10"].values
tgts4 = train["deg_Mg_50C"].values
tgts5 = train["deg_50C"].values

seqs = [
    (tensor(txts[i]), tensor([tgts1[i], tgts2[i], tgts3[i], tgts4[i], tgts5[i]]))
    for i in range(len(txts))
]
len(seqs), seqs[0][0].shape, seqs[0][1].shape



## === cell 10
test_seqs = [(tensor(x[:107]), torch.zeros(5, 68)) for x in test["seqs"].values]
len(test_seqs), test_seqs[0][0].shape, test_seqs[0][1].shape



## === cell 11
BS = 32  # batch size
ES = 16  # embedding size
NH = 256  # number hidden units
NL = 2  # number layers
DO = 0.5  # dropout
EP = 30  # epochs
LR = 0.0029906294233063486  # learning rate
WD = 0.0  # weight decay

sl = 107




## === cell 12
class OVModel(nn.Module):
    def __init__(
        self,
        vocab1_sz,
        vocab2_sz,
        vocab3_sz,
        emb_sz,
        n_hidden,
        n_layers,
        p,
        y_range=None,
    ):
        super().__init__()
        self.y_range = y_range
        self.n_layers = n_layers
        self.n_hidden = n_hidden

        self.i_h1 = nn.Embedding(vocab1_sz, emb_sz)
        self.i_h2 = nn.Embedding(vocab2_sz, emb_sz)
        self.i_h3 = nn.Embedding(vocab3_sz, emb_sz)
        self.rnn = nn.LSTM(
            emb_sz * 3 + 3, n_hidden, n_layers, batch_first=True, bidirectional=True
        )
        self.drop = nn.Dropout(p)
        self.h_o = nn.Linear(n_hidden * 2, 5)

        self.h = None  # initialized on first forward with correct batch size

    def _init_h(self, bs, device_):
        h0 = torch.zeros(self.n_layers * 2, bs, self.n_hidden, device=device_)
        c0 = torch.zeros(self.n_layers * 2, bs, self.n_hidden, device=device_)
        self.h = (h0, c0)

    def forward(self, x):
        if (
            self.h is None
            or self.h[0].shape[1] != x.shape[0]
            or self.h[0].device != x.device
        ):
            self._init_h(x.shape[0], x.device)

        e1 = self.i_h1(x[:, :, 0].long())
        e2 = self.i_h2(x[:, :, 1].long())
        e3 = self.i_h3(x[:, :, 2].long())
        bp = x[:, :, 3:].float()
        e = torch.cat((e1, e2, e3, bp), dim=2)

        raw, h = self.rnn(e, self.h)
        do = self.drop(raw)
        out = self.h_o(do)

        if self.y_range is None:
            self.h = (h[0].detach(), h[1].detach())
            return out, raw, do

        out = torch.sigmoid(out) * (self.y_range[1] - self.y_range[0]) + self.y_range[0]
        self.h = (h[0].detach(), h[1].detach())
        return out, raw, do

    def reset(self):
        if self.h is not None:
            self.h[0].zero_()
            self.h[1].zero_()


class ModelResetter(Callback):
    def before_batch(self):
        if hasattr(self.learn.model, "reset"):
            self.learn.model.reset()




## === cell 13
def loss_func(inp, targ):
    inp = inp[0]
    inp = inp[:, :68, :]
    l1 = F.mse_loss(inp[:, :, 0], targ[:, 0, :])
    l2 = F.mse_loss(inp[:, :, 1], targ[:, 1, :])
    l3 = F.mse_loss(inp[:, :, 2], targ[:, 2, :])
    l4 = F.mse_loss(inp[:, :, 3], targ[:, 3, :])
    l5 = F.mse_loss(inp[:, :, 4], targ[:, 4, :])
    return torch.sqrt((l1 + l2 + l3 + l4 + l5) / 5)




## === cell 14

spltidx = np.arange(len(seqs))
kf = KFold(n_splits=5, shuffle=True, random_state=42)
splts = list(kf.split(spltidx))
len(splts), [len(v[0]) for v in splts], [len(v[1]) for v in splts]



## === cell 15
all_preds = []

for i in range(5):
    tr_idx, va_idx = splts[i]
    dls = DataLoaders.from_dsets(
        [seqs[j] for j in tr_idx],
        [seqs[j] for j in va_idx],
        bs=BS,
        drop_last=True,
        shuffle=True,
    )
    dls = dls.to(device)

    test_dl_fa = dls.test_dl(test_seqs, bs=BS)

    net = OVModel(
        len(vocab1), len(vocab2), len(vocab3), ES, NH, NL, DO, y_range=None
    ).to(device)

    learn = Learner(dls, net, loss_func=loss_func, cbs=[ModelResetter()])

    learn.fit_one_cycle(EP, LR, wd=WD)

    preds_out = learn.get_preds(dl=test_dl_fa, reorder=False)
    preds_tensor = preds_out[0] if isinstance(preds_out, (tuple, list)) else preds_out
    all_preds.append(preds_tensor.detach().cpu())

len(all_preds), all_preds[0].shape



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/327764995.py in <cell line: 0>()
     25     preds_out = learn.get_preds(dl=test_dl_fa, reorder=False)
     26     preds_tensor = preds_out[0] if isinstance(preds_out, (tuple, list)) else preds_out
---> 27     all_preds.append(preds_tensor.detach().cpu())
     28 
     29 len(all_preds), all_preds[0].shape

AttributeError: 'tuple' object has no attribute 'detach'

## === cell 16
predictions = sum(all_preds) / len(all_preds)
predictions.shape  # (n_test, 107, 5)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
ZeroDivisionError                         Traceback (most recent call last)
/tmp/ipykernel_11/3403832115.py in <cell line: 0>()
----> 1 predictions = sum(all_preds) / len(all_preds)
      2 predictions.shape  # (n_test, 107, 5)
      3 

ZeroDivisionError: division by zero

## === cell 17
pred_scored = predictions[:, :68, :].reshape(-1, 5).numpy()

sub_out = sub.copy()
target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
sub_out[target_cols] = pred_scored

assert list(sub_out.columns) == ["id_seqpos"] + target_cols
assert sub_out.shape[0] == sub.shape[0] == 25680
assert np.isfinite(sub_out[target_cols].to_numpy()).all()

sub_out.head(), sub_out.shape



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1073142086.py in <cell line: 0>()
----> 1 pred_scored = predictions[:, :68, :].reshape(-1, 5).numpy()
      2 
      3 sub_out = sub.copy()
      4 target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
      5 sub_out[target_cols] = pred_scored

NameError: name 'predictions' is not defined

## === cell 18
out_path = "submission.csv"
sub_out.to_csv(out_path, index=False)

assert os.path.exists(out_path)
assert out_path.endswith(".csv")
out_path, sub_out.head()

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2112445772.py in <cell line: 0>()
      1 out_path = "submission.csv"
----> 2 sub_out.to_csv(out_path, index=False)
      3 
      4 assert os.path.exists(out_path)
      5 assert out_path.endswith(".csv")

NameError: name 'sub_out' is not defined
