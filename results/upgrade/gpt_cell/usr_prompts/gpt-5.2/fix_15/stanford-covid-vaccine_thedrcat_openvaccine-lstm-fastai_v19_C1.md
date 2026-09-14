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

for m in list(sys.modules):
    if m == "numpy" or m.startswith("numpy."):
        sys.modules.pop(m, None)

import torch, torchvision, fastai, numpy, pandas

print(
    "Using:",
    "torch",
    torch.__version__,
    "| torchvision",
    torchvision.__version__,
    "| fastai",
    fastai.__version__,
    "| numpy",
    numpy.__version__,
    "| pandas",
    pandas.__version__,
)


## === cell 3
import pandas as pd

base_path = "/kaggle/input/stanford-covid-vaccine/"
train = pd.read_json(base_path + "train.json", lines=True)
test = pd.read_json(base_path + "test.json", lines=True)
sub = pd.read_csv(base_path + "sample_submission.csv")

train.shape, train["id"].nunique(), test.shape, sub.shape


## === cell 4
all1 = []
all2 = []
all3 = []
for i in range(len(train)):
    all1.extend(train['sequence'].loc[i])
    all2.extend(train['structure'].loc[i])
    all3.extend(train['predicted_loop_type'].loc[i])
    


## === cell 5
from fastcore.foundation import L

all1 = L(all1)
all2 = L(all2)
all3 = L(all3)


## === cell 6
vocab1 = all1.unique()
vocab2 = all2.unique()
vocab3 = all3.unique()


## === cell 7
word2idx1 = {w:i for i,w in enumerate(vocab1)}
word2idx2 = {w:i for i,w in enumerate(vocab2)}
word2idx3 = {w:i for i,w in enumerate(vocab3)}


## === cell 8
def joiner(row):
    l1 =  list(row[0])
    l2 =  list(row[1])
    l3 =  list(row[2])
    out = [[word2idx1[l1[i]], word2idx2[l2[i]], word2idx3[l3[i]]] for i in range(len(l1))]
    return out


## === cell 9
train['seqs'] = train[['sequence', 'structure', 'predicted_loop_type']].apply(joiner, axis=1)


## === cell 10
train = train[train['SN_filter'] == 1]


## === cell 11
txts = L([x for x in train['seqs'].values])
tgts1 = L([x for x in train['reactivity'].values])
tgts2 = L([x for x in train['deg_Mg_pH10'].values])
tgts3 = L([x for x in train['deg_pH10'].values])
tgts4 = L([x for x in train['deg_Mg_50C'].values])
tgts5 = L([x for x in train['deg_50C'].values])


## === cell 12
from torch import tensor

seqs = L(
    (tensor(txts[i]), tensor([tgts1[i], tgts2[i], tgts3[i], tgts4[i], tgts5[i]]))
    for i in range(len(txts))
)


## === cell 13
BS = 32 # batch size 
ES = 32 # embedding size
NH = 512 # number hidden units
NL = 2 # number layers
DO = 0.5 # dropout
EP = 18 # epochs
LR = 3e-3 # learning rate
WD = 0.1 # weight decay


## === cell 14
import torch
from torch import nn

sl = 107


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
        self.i_h1 = nn.Embedding(vocab1_sz, emb_sz)
        self.i_h2 = nn.Embedding(vocab2_sz, emb_sz)
        self.i_h3 = nn.Embedding(vocab3_sz, emb_sz)
        self.rnn = nn.LSTM(
            emb_sz * 3, n_hidden, n_layers, batch_first=True, bidirectional=True
        )
        self.drop = nn.Dropout(p)
        self.h_o = nn.Linear(n_hidden * 2, 5)
        self.h = [torch.zeros(n_layers * 2, BS, n_hidden).to("cuda") for _ in range(2)]

    def forward(self, x):
        e1 = self.i_h1(x[:, :, 0])
        e2 = self.i_h2(x[:, :, 1])
        e3 = self.i_h3(x[:, :, 2])
        e = torch.cat((e1, e2, e3), dim=2)
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


## === cell 15
def loss_func(inp, targ):
    inp = inp[0]
    inp = inp[:,:68,:]
    l1 = F.mse_loss(inp[:,:,0], targ[:,0,:])
    l2 = F.mse_loss(inp[:,:,1], targ[:,1,:])
    l3 = F.mse_loss(inp[:,:,2], targ[:,2,:])
    l4 = F.mse_loss(inp[:,:,3], targ[:,3,:])
    l5 = F.mse_loss(inp[:,:,4], targ[:,4,:])
    return torch.sqrt((l1 + l2 + l3 + l4 +l5)/5)


## === cell 16
cut = int(len(seqs) * 0.8)
cut


## === cell 17
import torch
from torch.utils.data import DataLoader


class DataLoaders:
    def __init__(self, train_dl, valid_dl, device=None):
        self.train = train_dl
        self.valid = valid_dl
        self.device = device

    @classmethod
    def from_dsets(
        cls, train_ds, valid_ds, bs=64, drop_last=False, shuffle=False, **kwargs
    ):
        train_dl = DataLoader(
            train_ds, batch_size=bs, shuffle=shuffle, drop_last=drop_last
        )
        valid_dl = DataLoader(valid_ds, batch_size=bs, shuffle=False, drop_last=False)
        return cls(train_dl, valid_dl)

    def cuda(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self


dls = DataLoaders.from_dsets(
    seqs[:cut], seqs[cut:], bs=BS, drop_last=True, shuffle=True
)
dls.cuda()


## === cell 18
net = OVModel(len(vocab1), len(vocab2), len(vocab3), ES, NH, NL, DO, y_range=None)


## === cell 19
learn = None


## === cell 20
model = OVModel(len(vocab1), len(vocab2), len(vocab3), ES, NH, NL, DO).cuda()
batch = dls.one_batch()
batch[0].shape, batch[1].shape
out = model(batch[0])
out[0].shape, batch[1].shape, batch[0].shape


## --- ERROR in cell 20, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/267021659.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      1[0m [0;31m# Testing shapes[0m[0;34m[0m[0;34m[0m[0m
[1;32m      2[0m [0mmodel[0m [0;34m=[0m [0mOVModel[0m[0;34m([0m[0mlen[0m[0;34m([0m[0mvocab1[0m[0;34m)[0m[0;34m,[0m [0mlen[0m[0;34m([0m[0mvocab2[0m[0;34m)[0m[0;34m,[0m [0mlen[0m[0;34m([0m[0mvocab3[0m[0;34m)[0m[0;34m,[0m [0mES[0m[0;34m,[0m [0mNH[0m[0;34m,[0m [0mNL[0m[0;34m,[0m [0mDO[0m[0;34m)[0m[0;34m.[0m[0mcuda[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 3[0;31m [0mbatch[0m [0;34m=[0m [0mdls[0m[0;34m.[0m[0mone_batch[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      4[0m [0mbatch[0m[0;34m[[0m[0;36m0[0m[0;34m][0m[0;34m.[0m[0mshape[0m[0;34m,[0m [0mbatch[0m[0;34m[[0m[0;36m1[0m[0;34m][0m[0;34m.[0m[0mshape[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0mout[0m [0;34m=[0m [0mmodel[0m[0;34m([0m[0mbatch[0m[0;34m[[0m[0;36m0[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mAttributeError[0m: 'DataLoaders' object has no attribute 'one_batch'

## === cell 22
learn.fit_one_cycle(EP, LR, wd=WD)
