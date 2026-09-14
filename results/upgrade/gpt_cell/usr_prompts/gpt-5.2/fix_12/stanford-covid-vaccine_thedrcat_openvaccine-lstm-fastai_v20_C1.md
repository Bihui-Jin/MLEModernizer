# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.8

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

## === cell 0
!pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html -q
!pip install fastai==2.0.13 -q


## === cell 1
pass


## === cell 2
import sys
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numpy==1.26.4"])

import numpy as np  # noqa: F401
import pandas as pd

path = "/kaggle/data/stanford-covid-vaccine"
train = pd.read_json(f"{path}/train.json", lines=True)
test = pd.read_json(f"{path}/test.json", lines=True)
sub = pd.read_csv(f"{path}/sample_submission.csv")


## === cell 3
train.shape, train['id'].nunique(), test.shape, sub.shape


## === cell 4

import os

_possible_bpps_dirs = [
    os.path.join(path, "bpps") if "path" in globals() else None,  # from cell 2
    "/kaggle/data/stanford-covid-vaccine/bpps",
    "/kaggle/input/stanford-covid-vaccine/bpps",
    "../input/stanford-covid-vaccine/bpps",
]
_possible_bpps_dirs = [p for p in _possible_bpps_dirs if p is not None]
BPPS_DIR = next((p for p in _possible_bpps_dirs if os.path.isdir(p)), None)


def _zeros_for_row(row):
    L = (
        int(row["seq_length"])
        if "seq_length" in row and not pd.isna(row["seq_length"])
        else len(row["sequence"])
    )
    return np.zeros(L, dtype=np.float32)


def read_bpps_sum(df):
    bpps_arr = []
    for _, row in df.iterrows():
        mol_id = row["id"]
        if BPPS_DIR is None:
            bpps_arr.append(_zeros_for_row(row))
            continue
        fpath = os.path.join(BPPS_DIR, f"{mol_id}.npy")
        if not os.path.isfile(fpath):
            bpps_arr.append(_zeros_for_row(row))
            continue
        bpps_arr.append(np.load(fpath).sum(axis=1))
    return bpps_arr


def read_bpps_max(df):
    bpps_arr = []
    for _, row in df.iterrows():
        mol_id = row["id"]
        if BPPS_DIR is None:
            bpps_arr.append(_zeros_for_row(row))
            continue
        fpath = os.path.join(BPPS_DIR, f"{mol_id}.npy")
        if not os.path.isfile(fpath):
            bpps_arr.append(_zeros_for_row(row))
            continue
        bpps_arr.append(np.load(fpath).max(axis=1))
    return bpps_arr


def read_bpps_nb(df):
    bpps_nb_mean = 0.077522  # mean of bpps_nb across all training data
    bpps_nb_std = 0.08914  # std of bpps_nb across all training data
    bpps_arr = []
    for _, row in df.iterrows():
        mol_id = row["id"]
        if BPPS_DIR is None:
            bpps_arr.append(_zeros_for_row(row))
            continue
        fpath = os.path.join(BPPS_DIR, f"{mol_id}.npy")
        if not os.path.isfile(fpath):
            bpps_arr.append(_zeros_for_row(row))
            continue
        bpps = np.load(fpath)
        bpps_nb = (bpps > 0).sum(axis=0) / bpps.shape[0]
        bpps_nb = (bpps_nb - bpps_nb_mean) / bpps_nb_std
        bpps_arr.append(bpps_nb)
    return bpps_arr


train["bpps_sum"] = read_bpps_sum(train)
test["bpps_sum"] = read_bpps_sum(test)
train["bpps_max"] = read_bpps_max(train)
test["bpps_max"] = read_bpps_max(test)
train["bpps_nb"] = read_bpps_nb(train)
test["bpps_nb"] = read_bpps_nb(test)


## === cell 5
all1 = []
all2 = []
all3 = []
for i in range(len(train)):
    all1.extend(train['sequence'].loc[i])
    all2.extend(train['structure'].loc[i])
    all3.extend(train['predicted_loop_type'].loc[i])
    


## === cell 6
from fastcore.foundation import L

all1 = L(all1)
all2 = L(all2)
all3 = L(all3)


## === cell 7
vocab1 = all1.unique()
vocab2 = all2.unique()
vocab3 = all3.unique()


## === cell 8
word2idx1 = {w:i for i,w in enumerate(vocab1)}
word2idx2 = {w:i for i,w in enumerate(vocab2)}
word2idx3 = {w:i for i,w in enumerate(vocab3)}


## === cell 9
def joiner(row):
    l1 =  list(row[0])
    l2 =  list(row[1])
    l3 =  list(row[2])
    l4 =  list(row[3])
    l5 =  list(row[4])
    l6 =  list(row[5])
    out = [[word2idx1[l1[i]], word2idx2[l2[i]], word2idx3[l3[i]], l4[i], l5[i], l6[i]] for i in range(len(l1))]
    return out


## === cell 10
train['seqs'] = train[['sequence', 'structure', 'predicted_loop_type', 'bpps_sum', 'bpps_max', 'bpps_nb']].apply(joiner, axis=1)


## === cell 11
train = train[train['SN_filter'] == 1]


## === cell 12
txts = L([x for x in train['seqs'].values])
tgts1 = L([x for x in train['reactivity'].values])
tgts2 = L([x for x in train['deg_Mg_pH10'].values])
tgts3 = L([x for x in train['deg_pH10'].values])
tgts4 = L([x for x in train['deg_Mg_50C'].values])
tgts5 = L([x for x in train['deg_50C'].values])


## === cell 13
from torch import tensor

seqs = L(
    (tensor(txts[i]), tensor([tgts1[i], tgts2[i], tgts3[i], tgts4[i], tgts5[i]]))
    for i in range(len(txts))
)


## === cell 14
BS = 32 # batch size 
ES = 32 # embedding size
NH = 512 # number hidden units
NL = 2 # number layers
DO = 0.5 # dropout
EP = 18 # epochs
LR = 3e-3 # learning rate
WD = 0.1 # weight decay


## === cell 15
import torch
import torch.nn as nn

try:
    from fastai.torch_basics import Module
except Exception:
    from torch.nn import Module  # fallback if fastai import isn't available

sl = 107


class OVModel(Module):
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
        self.y_range = y_range
        self.i_h1 = nn.Embedding(vocab1_sz, emb_sz)
        self.i_h2 = nn.Embedding(vocab2_sz, emb_sz)
        self.i_h3 = nn.Embedding(vocab3_sz, emb_sz)
        self.rnn = nn.LSTM(
            emb_sz * 3 + 3, n_hidden, n_layers, batch_first=True, bidirectional=True
        )
        self.drop = nn.Dropout(p)
        self.h_o = nn.Linear(n_hidden * 2, 5)
        self.h = [torch.zeros(n_layers * 2, BS, n_hidden).to("cuda") for _ in range(2)]

    def forward(self, x):
        e1 = self.i_h1(x[:, :, 0].long())
        e2 = self.i_h2(x[:, :, 1].long())
        e3 = self.i_h3(x[:, :, 2].long())
        bp = x[:, :, 3:]
        e = torch.cat((e1, e2, e3, bp), dim=2)
        raw, h = self.rnn(e, self.h)
        do = self.drop(raw)
        out = self.h_o(do)
        if self.y_range is None:
            self.h = [h_.detach() for h_ in h]
            return out, raw, do
        out = torch.sigmoid(out) * (self.y_range[1] - self.y_range[0]) + self.y_range[0]
        self.h = [h_.detach() for h_ in h]
        return out, raw, do

    def reset(self):
        for h in self.h:
            h.zero_()


## === cell 16
def loss_func(inp, targ):
    inp = inp[0]
    inp = inp[:,:68,:]
    l1 = F.mse_loss(inp[:,:,0], targ[:,0,:])
    l2 = F.mse_loss(inp[:,:,1], targ[:,1,:])
    l3 = F.mse_loss(inp[:,:,2], targ[:,2,:])
    l4 = F.mse_loss(inp[:,:,3], targ[:,3,:])
    l5 = F.mse_loss(inp[:,:,4], targ[:,4,:])
    return torch.sqrt((l1 + l2 + l3 + l4 +l5)/5)


## === cell 17
cut = int(len(seqs) * 0.8)
cut


## === cell 18
dls = DataLoaders.from_dsets(seqs[:cut], seqs[cut:], bs=BS, drop_last=True, shuffle=True)
dls.cuda()


## --- ERROR in cell 18, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mNameError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1409643339.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0mdls[0m [0;34m=[0m [0mDataLoaders[0m[0;34m.[0m[0mfrom_dsets[0m[0;34m([0m[0mseqs[0m[0;34m[[0m[0;34m:[0m[0mcut[0m[0;34m][0m[0;34m,[0m [0mseqs[0m[0;34m[[0m[0mcut[0m[0;34m:[0m[0;34m][0m[0;34m,[0m [0mbs[0m[0;34m=[0m[0mBS[0m[0;34m,[0m [0mdrop_last[0m[0;34m=[0m[0;32mTrue[0m[0;34m,[0m [0mshuffle[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0mdls[0m[0;34m.[0m[0mcuda[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mNameError[0m: name 'DataLoaders' is not defined

## === cell 19
net = OVModel(len(vocab1), len(vocab2), len(vocab3), ES, NH, NL, DO, y_range=None)
