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

0.39223

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import json, os
import pandas as pd, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from fastai.torch_core import L, tensor
from fastai.data.core import DataLoaders
from fastai.learner import Learner
from fastai.callback.all import ModelResetter



## === cell 1
base_path = "/kaggle/input/stanford-covid-vaccine"
train = pd.read_json(os.path.join(base_path, "train.json"), lines=True)
test = pd.read_json(os.path.join(base_path, "test.json"), lines=True)
sub = pd.read_csv(os.path.join(base_path, "sample_submission.csv"))



## === cell 2
train = train[train["SN_filter"] == 1].reset_index(drop=True)

all_seq = list("".join(train["sequence"].values))
all_str = list("".join(train["structure"].values))
all_loop = list("".join(train["predicted_loop_type"].values))

vocab1 = sorted(set(all_seq))
vocab2 = sorted(set(all_str))
vocab3 = sorted(set(all_loop))

word2idx1 = {c: i for i, c in enumerate(vocab1)}
word2idx2 = {c: i for i, c in enumerate(vocab2)}
word2idx3 = {c: i for i, c in enumerate(vocab3)}




## === cell 3
def joiner(row):
    seq = list(row["sequence"])
    stru = list(row["structure"])
    loop = list(row["predicted_loop_type"])
    return [
        [word2idx1[seq[i]], word2idx2[stru[i]], word2idx3[loop[i]]]
        for i in range(len(seq))
    ]


train["seqs"] = train.apply(joiner, axis=1)



## === cell 4
tgts1 = L(train["reactivity"].values)
tgts2 = L(train["deg_Mg_pH10"].values)
tgts3 = L(train["deg_pH10"].values)
tgts4 = L(train["deg_Mg_50C"].values)
tgts5 = L(train["deg_50C"].values)



## === cell 5
seqs = L()
for i in range(len(train)):
    inp = tensor(train["seqs"][i])  # (107,3)  long tensor
    tgt = tensor(
        [tgts1[i], tgts2[i], tgts3[i], tgts4[i], tgts5[i]], dtype=torch.float32
    )
    seqs.append((inp, tgt))



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_55/3092545729.py in <cell line: 0>()
      5     # stack the five target vectors → shape (5,68)
      6     tgt = tensor(
----> 7         [tgts1[i], tgts2[i], tgts3[i], tgts4[i], tgts5[i]], dtype=torch.float32
      8     )
      9     seqs.append((inp, tgt))

/usr/local/lib/python3.11/dist-packages/fastcore/foundation.py in __getitem__(self, idx)
    118     def _new(self, items, *args, **kwargs): return type(self)(items, *args, use_list=None, **kwargs)
    119     def __getitem__(self, idx):
--> 120         if isinstance(idx,int) and not hasattr(self.items,'iloc'): return self.items[idx]
    121         return self._get(idx) if is_indexer(idx) else L(self._get(idx), use_list=None)
    122     def copy(self): return self._new(self.items.copy())

IndexError: list index out of range

## === cell 6
cut = int(0.8 * len(seqs))
train_ds, valid_ds = seqs[:cut], seqs[cut:]

dls = DataLoaders.from_dsets(train_ds, valid_ds, bs=32, drop_last=True, shuffle=True)
dls.cuda() if torch.cuda.is_available() else dls.cpu()




## === cell 7
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
        self.emb1 = nn.Embedding(vocab1_sz, emb_sz)
        self.emb2 = nn.Embedding(vocab2_sz, emb_sz)
        self.emb3 = nn.Embedding(vocab3_sz, emb_sz)
        self.rnn = nn.LSTM(
            emb_sz * 3, n_hidden, n_layers, batch_first=True, bidirectional=True
        )
        self.drop = nn.Dropout(p)
        self.head = nn.Linear(n_hidden * 2, 5)
        self.n_layers = n_layers
        self.n_hidden = n_hidden

    def forward(self, x):
        e1 = self.emb1(x[:, :, 0])
        e2 = self.emb2(x[:, :, 1])
        e3 = self.emb3(x[:, :, 2])
        e = torch.cat((e1, e2, e3), dim=2)  # (B, L, 3*emb)
        h0 = torch.zeros(self.n_layers * 2, x.size(0), self.n_hidden, device=x.device)
        c0 = torch.zeros(self.n_layers * 2, x.size(0), self.n_hidden, device=x.device)
        rnn_out, _ = self.rnn(e, (h0, c0))
        dropped = self.drop(rnn_out)
        out = self.head(dropped)  # (B, L, 5)
        if self.y_range is not None:
            out = (
                torch.sigmoid(out) * (self.y_range[1] - self.y_range[0])
                + self.y_range[0]
            )
        return out, rnn_out, dropped




## === cell 8
def loss_func(pred, targ):
    out = pred[0][:, :68, :]  # only first 68 positions are scored
    targ = targ.permute(0, 2, 1)
    mse = F.mse_loss(out, targ, reduction="none")  # (B,68,5)
    return torch.sqrt(mse.mean())




## === cell 9
net = OVModel(
    len(vocab1),
    len(vocab2),
    len(vocab3),
    emb_sz=32,
    n_hidden=512,
    n_layers=2,
    p=0.5,
    y_range=None,
)

learn = Learner(dls, net, loss_func=loss_func, cbs=ModelResetter)
learn.fit_one_cycle(10, 9e-3)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/1478139932.py in <cell line: 0>()
     11 
     12 learn = Learner(dls, net, loss_func=loss_func, cbs=ModelResetter)
---> 13 learn.fit_one_cycle(10, 9e-3)
     14 
     15 

/usr/local/lib/python3.11/dist-packages/fastai/callback/schedule.py in fit_one_cycle(self, n_epoch, lr_max, div, div_final, pct_start, wd, moms, cbs, reset_opt, start_epoch)
    119     scheds = {'lr': combined_cos(pct_start, lr_max/div, lr_max, lr_max/div_final),
    120               'mom': combined_cos(pct_start, *(self.moms if moms is None else moms))}
--> 121     self.fit(n_epoch, cbs=ParamScheduler(scheds)+L(cbs), reset_opt=reset_opt, wd=wd, start_epoch=start_epoch)
    122 
    123 # %% ../../nbs/14_callback.schedule.ipynb 50

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in fit(self, n_epoch, lr, wd, cbs, reset_opt, start_epoch)
    270             self.opt.set_hypers(lr=self.lr if lr is None else lr)
    271             self.n_epoch = n_epoch
--> 272             self._with_events(self._do_fit, 'fit', CancelFitException, self._end_cleanup)
    273 
    274     def _end_cleanup(self): self.dl,self.xb,self.yb,self.pred,self.loss = None,(None,),(None,),None,None

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _with_events(self, f, event_type, ex, final)
    205 
    206     def _with_events(self, f, event_type, ex, final=noop):
--> 207         try: self(f'before_{event_type}');  f()
    208         except ex: self(f'after_cancel_{event_type}')
    209         self(f'after_{event_type}');  final()

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _do_fit(self)
    259         for epoch in range(self.n_epoch):
    260             self.epoch=epoch
--> 261             self._with_events(self._do_epoch, 'epoch', CancelEpochException)
    262 
    263     def fit(self, n_epoch, lr=None, wd=None, cbs=None, reset_opt=False, start_epoch=0):

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _with_events(self, f, event_type, ex, final)
    205 
    206     def _with_events(self, f, event_type, ex, final=noop):
--> 207         try: self(f'before_{event_type}');  f()
    208         except ex: self(f'after_cancel_{event_type}')
    209         self(f'after_{event_type}');  final()

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _do_epoch(self)
    253 
    254     def _do_epoch(self):
--> 255         self._do_epoch_train()
    256         self._do_epoch_validate()
    257 

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _do_epoch_train(self)
    245     def _do_epoch_train(self):
    246         self.dl = self.dls.train
--> 247         self._with_events(self.all_batches, 'train', CancelTrainException)
    248 
    249     def _do_epoch_validate(self, ds_idx=1, dl=None):

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _with_events(self, f, event_type, ex, final)
    205 
    206     def _with_events(self, f, event_type, ex, final=noop):
--> 207         try: self(f'before_{event_type}');  f()
    208         except ex: self(f'after_cancel_{event_type}')
    209         self(f'after_{event_type}');  final()

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in __call__(self, event_name)
    178 
    179     def ordered_cbs(self, event): return [cb for cb in self.cbs.sorted('order') if hasattr(cb, event)]
--> 180     def __call__(self, event_name): L(event_name).map(self._call_one)
    181 
    182     def _call_one(self, event_name):

/usr/local/lib/python3.11/dist-packages/fastcore/foundation.py in map(self, f, *args, **kwargs)
    166     def range(cls, a, b=None, step=None): return cls(range_of(a, b=b, step=step))
    167 
--> 168     def map(self, f, *args, **kwargs): return self._new(map_ex(self, f, *args, gen=False, **kwargs))
    169     def argwhere(self, f, negate=False, **kwargs): return self._new(argwhere(self, f, negate, **kwargs))
    170     def argfirst(self, f, negate=False):

/usr/local/lib/python3.11/dist-packages/fastcore/basics.py in map_ex(iterable, f, gen, *args, **kwargs)
    949     res = map(g, iterable)
    950     if gen: return res
--> 951     return list(res)
    952 
    953 # %% ../nbs/01_basics.ipynb

/usr/local/lib/python3.11/dist-packages/fastcore/basics.py in __call__(self, *args, **kwargs)
    934             if isinstance(v,_Arg): kwargs[k] = args.pop(v.i)
    935         fargs = [args[x.i] if isinstance(x, _Arg) else x for x in self.pargs] + args[self.maxi+1:]
--> 936         return self.func(*fargs, **kwargs)
    937 
    938 # %% ../nbs/01_basics.ipynb

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _call_one(self, event_name)
    182     def _call_one(self, event_name):
    183         if not hasattr(event, event_name): raise Exception(f'missing {event_name}')
--> 184         for cb in self.cbs.sorted('order'): cb(event_name)
    185 
    186     def _bn_bias_state(self, with_bias): return norm_bias_params(self.model, with_bias).map(self.opt.state)

/usr/local/lib/python3.11/dist-packages/fastai/callback/core.py in __call__(self, event_name)
     62             try: res = getcallable(self, event_name)()
     63             except (CancelBatchException, CancelBackwardException, CancelEpochException, CancelFitException, CancelStepException, CancelTrainException, CancelValidException): raise
---> 64             except Exception as e: raise modify_exception(e, f'Exception occured in `{self.__class__.__name__}` when calling event `{event_name}`:\n\t{e.args[0]}', replace=True)
     65         if event_name=='after_fit': self.run=True #Reset self.run to True at each end of fit
     66         return res

/usr/local/lib/python3.11/dist-packages/fastai/callback/core.py in __call__(self, event_name)
     60         res = None
     61         if self.run and _run:
---> 62             try: res = getcallable(self, event_name)()
     63             except (CancelBatchException, CancelBackwardException, CancelEpochException, CancelFitException, CancelStepException, CancelTrainException, CancelValidException): raise
     64             except Exception as e: raise modify_exception(e, f'Exception occured in `{self.__class__.__name__}` when calling event `{event_name}`:\n\t{e.args[0]}', replace=True)

/usr/local/lib/python3.11/dist-packages/fastai/callback/rnn.py in before_train(self)
     14 class ModelResetter(Callback):
     15     "`Callback` that resets the model at each validation/training step"
---> 16     def before_train(self):    self.model.reset()
     17     def before_validate(self): self.model.reset()
     18     def after_fit(self):       self.model.reset()

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in __getattr__(self, name)
   1926             if name in modules:
   1927                 return modules[name]
-> 1928         raise AttributeError(
   1929             f"'{type(self).__name__}' object has no attribute '{name}'"
   1930         )

AttributeError: Exception occured in `ModelResetter` when calling event `before_train`:
	'OVModel' object has no attribute 'reset'

## === cell 10
def joiner_test(row):
    seq = list(row["sequence"])
    stru = list(row["structure"])
    loop = list(row["predicted_loop_type"])
    return [
        [word2idx1[seq[i]], word2idx2[stru[i]], word2idx3[loop[i]]]
        for i in range(len(seq))
    ]


test["seqs"] = test.apply(joiner_test, axis=1)

test_seqs = [(tensor(x), torch.zeros(5, 68)) for x in test["seqs"].values]
while len(test_seqs) % 32 != 0:
    test_seqs.append((torch.zeros((107, 3), dtype=torch.long), torch.zeros(5, 68)))

test_dl = DataLoader(dataset=test_seqs, batch_size=32, shuffle=False, drop_last=False)
test_dl = test_dl.to("cuda") if torch.cuda.is_available() else test_dl



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/3394929066.py in <cell line: 0>()
     18 
     19 test_dl = DataLoader(dataset=test_seqs, batch_size=32, shuffle=False, drop_last=False)
---> 20 test_dl = test_dl.to("cuda") if torch.cuda.is_available() else test_dl
     21 

AttributeError: 'DataLoader' object has no attribute 'to'

## === cell 11
preds = learn.get_preds(dl=test_dl, reorder=False)[0]  # (N,5,68)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/fastai/learner.py in get_preds(self, ds_idx, dl, with_input, with_decoded, with_loss, act, inner, reorder, cbs, **kwargs)
    315         with ContextManagers(ctx_mgrs):
--> 316             self._do_epoch_validate(dl=dl)
    317             if act is None: act = getcallable(self.loss_func, 'activation')

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _do_epoch_validate(self, ds_idx, dl)
    251         self.dl = dl
--> 252         with torch.no_grad(): self._with_events(self.all_batches, 'validate', CancelValidException)
    253 

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _with_events(self, f, event_type, ex, final)
    206     def _with_events(self, f, event_type, ex, final=noop):
--> 207         try: self(f'before_{event_type}');  f()
    208         except ex: self(f'after_cancel_{event_type}')

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in __call__(self, event_name)
    179     def ordered_cbs(self, event): return [cb for cb in self.cbs.sorted('order') if hasattr(cb, event)]
--> 180     def __call__(self, event_name): L(event_name).map(self._call_one)
    181 

/usr/local/lib/python3.11/dist-packages/fastcore/foundation.py in map(self, f, *args, **kwargs)
    167 
--> 168     def map(self, f, *args, **kwargs): return self._new(map_ex(self, f, *args, gen=False, **kwargs))
    169     def argwhere(self, f, negate=False, **kwargs): return self._new(argwhere(self, f, negate, **kwargs))

/usr/local/lib/python3.11/dist-packages/fastcore/basics.py in map_ex(iterable, f, gen, *args, **kwargs)
    950     if gen: return res
--> 951     return list(res)
    952 

/usr/local/lib/python3.11/dist-packages/fastcore/basics.py in __call__(self, *args, **kwargs)
    935         fargs = [args[x.i] if isinstance(x, _Arg) else x for x in self.pargs] + args[self.maxi+1:]
--> 936         return self.func(*fargs, **kwargs)
    937 

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _call_one(self, event_name)
    183         if not hasattr(event, event_name): raise Exception(f'missing {event_name}')
--> 184         for cb in self.cbs.sorted('order'): cb(event_name)
    185 

/usr/local/lib/python3.11/dist-packages/fastai/callback/core.py in __call__(self, event_name)
     63             except (CancelBatchException, CancelBackwardException, CancelEpochException, CancelFitException, CancelStepException, CancelTrainException, CancelValidException): raise
---> 64             except Exception as e: raise modify_exception(e, f'Exception occured in `{self.__class__.__name__}` when calling event `{event_name}`:\n\t{e.args[0]}', replace=True)
     65         if event_name=='after_fit': self.run=True #Reset self.run to True at each end of fit

/usr/local/lib/python3.11/dist-packages/fastai/callback/core.py in __call__(self, event_name)
     61         if self.run and _run:
---> 62             try: res = getcallable(self, event_name)()
     63             except (CancelBatchException, CancelBackwardException, CancelEpochException, CancelFitException, CancelStepException, CancelTrainException, CancelValidException): raise

/usr/local/lib/python3.11/dist-packages/fastai/callback/rnn.py in before_validate(self)
     16     def before_train(self):    self.model.reset()
---> 17     def before_validate(self): self.model.reset()
     18     def after_fit(self):       self.model.reset()

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in __getattr__(self, name)
   1927                 return modules[name]
-> 1928         raise AttributeError(
   1929             f"'{type(self).__name__}' object has no attribute '{name}'"

AttributeError: Exception occured in `ModelResetter` when calling event `before_validate`:
	'OVModel' object has no attribute 'reset'

During handling of the above exception, another exception occurred:

AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/3216039035.py in <cell line: 0>()
----> 1 preds = learn.get_preds(dl=test_dl, reorder=False)[0]  # (N,5,68)
      2 

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in get_preds(self, ds_idx, dl, with_input, with_decoded, with_loss, act, inner, reorder, cbs, **kwargs)
    313         ctx_mgrs = self.validation_context(cbs=L(cbs)+[cb], inner=inner)
    314         if with_loss: ctx_mgrs.append(self.loss_not_reduced())
--> 315         with ContextManagers(ctx_mgrs):
    316             self._do_epoch_validate(dl=dl)
    317             if act is None: act = getcallable(self.loss_func, 'activation')

/usr/local/lib/python3.11/dist-packages/fastcore/xtras.py in __exit__(self, *args, **kwargs)
    710     def __init__(self, mgrs): self.default,self.stack = L(mgrs),ExitStack()
    711     def __enter__(self): self.default.map(self.stack.enter_context)
--> 712     def __exit__(self, *args, **kwargs): self.stack.__exit__(*args, **kwargs)
    713 
    714 # %% ../nbs/03_xtras.ipynb

/usr/lib/python3.11/contextlib.py in __exit__(self, *exc_details)
    599                 # set-up context
    600                 fixed_ctx = exc_details[1].__context__
--> 601                 raise exc_details[1]
    602             except BaseException:
    603                 exc_details[1].__context__ = fixed_ctx

/usr/lib/python3.11/contextlib.py in __exit__(self, *exc_details)
    584             assert is_sync
    585             try:
--> 586                 if cb(*exc_details):
    587                     suppressed_exc = True
    588                     pending_raise = False

/usr/local/lib/python3.11/dist-packages/fastcore/xtras.py in __exit__(self, *args, **kwargs)
    710     def __init__(self, mgrs): self.default,self.stack = L(mgrs),ExitStack()
    711     def __enter__(self): self.default.map(self.stack.enter_context)
--> 712     def __exit__(self, *args, **kwargs): self.stack.__exit__(*args, **kwargs)
    713 
    714 # %% ../nbs/03_xtras.ipynb

/usr/lib/python3.11/contextlib.py in __exit__(self, *exc_details)
    599                 # set-up context
    600                 fixed_ctx = exc_details[1].__context__
--> 601                 raise exc_details[1]
    602             except BaseException:
    603                 exc_details[1].__context__ = fixed_ctx

/usr/lib/python3.11/contextlib.py in __exit__(self, *exc_details)
    584             assert is_sync
    585             try:
--> 586                 if cb(*exc_details):
    587                     suppressed_exc = True
    588                     pending_raise = False

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in __exit__(self, exc_type, exc_value, tb)
    274     def _end_cleanup(self): self.dl,self.xb,self.yb,self.pred,self.loss = None,(None,),(None,),None,None
    275     def __enter__(self): self(_before_epoch); return self
--> 276     def __exit__(self, exc_type, exc_value, tb): self(_after_epoch)
    277 
    278     def validation_context(self, cbs=None, inner=False):

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in __call__(self, event_name)
    178 
    179     def ordered_cbs(self, event): return [cb for cb in self.cbs.sorted('order') if hasattr(cb, event)]
--> 180     def __call__(self, event_name): L(event_name).map(self._call_one)
    181 
    182     def _call_one(self, event_name):

/usr/local/lib/python3.11/dist-packages/fastcore/foundation.py in map(self, f, *args, **kwargs)
    166     def range(cls, a, b=None, step=None): return cls(range_of(a, b=b, step=step))
    167 
--> 168     def map(self, f, *args, **kwargs): return self._new(map_ex(self, f, *args, gen=False, **kwargs))
    169     def argwhere(self, f, negate=False, **kwargs): return self._new(argwhere(self, f, negate, **kwargs))
    170     def argfirst(self, f, negate=False):

/usr/local/lib/python3.11/dist-packages/fastcore/basics.py in map_ex(iterable, f, gen, *args, **kwargs)
    949     res = map(g, iterable)
    950     if gen: return res
--> 951     return list(res)
    952 
    953 # %% ../nbs/01_basics.ipynb

/usr/local/lib/python3.11/dist-packages/fastcore/basics.py in __call__(self, *args, **kwargs)
    934             if isinstance(v,_Arg): kwargs[k] = args.pop(v.i)
    935         fargs = [args[x.i] if isinstance(x, _Arg) else x for x in self.pargs] + args[self.maxi+1:]
--> 936         return self.func(*fargs, **kwargs)
    937 
    938 # %% ../nbs/01_basics.ipynb

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _call_one(self, event_name)
    182     def _call_one(self, event_name):
    183         if not hasattr(event, event_name): raise Exception(f'missing {event_name}')
--> 184         for cb in self.cbs.sorted('order'): cb(event_name)
    185 
    186     def _bn_bias_state(self, with_bias): return norm_bias_params(self.model, with_bias).map(self.opt.state)

/usr/local/lib/python3.11/dist-packages/fastai/callback/core.py in __call__(self, event_name)
     62             try: res = getcallable(self, event_name)()
     63             except (CancelBatchException, CancelBackwardException, CancelEpochException, CancelFitException, CancelStepException, CancelTrainException, CancelValidException): raise
---> 64             except Exception as e: raise modify_exception(e, f'Exception occured in `{self.__class__.__name__}` when calling event `{event_name}`:\n\t{e.args[0]}', replace=True)
     65         if event_name=='after_fit': self.run=True #Reset self.run to True at each end of fit
     66         return res

/usr/local/lib/python3.11/dist-packages/fastai/callback/core.py in __call__(self, event_name)
     60         res = None
     61         if self.run and _run:
---> 62             try: res = getcallable(self, event_name)()
     63             except (CancelBatchException, CancelBackwardException, CancelEpochException, CancelFitException, CancelStepException, CancelTrainException, CancelValidException): raise
     64             except Exception as e: raise modify_exception(e, f'Exception occured in `{self.__class__.__name__}` when calling event `{event_name}`:\n\t{e.args[0]}', replace=True)

/usr/local/lib/python3.11/dist-packages/fastai/callback/rnn.py in after_fit(self)
     16     def before_train(self):    self.model.reset()
     17     def before_validate(self): self.model.reset()
---> 18     def after_fit(self):       self.model.reset()
     19     _docs = dict(before_train="Reset the model before training",
     20                  before_validate="Reset the model before validation",

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in __getattr__(self, name)
   1926             if name in modules:
   1927                 return modules[name]
-> 1928         raise AttributeError(
   1929             f"'{type(self).__name__}' object has no attribute '{name}'"
   1930         )

AttributeError: Exception occured in `ModelResetter` when calling event `after_fit`:
	'OVModel' object has no attribute 'reset'

## === cell 12
ids = []
react = []
deg_mg_pH10 = []
deg_pH10 = []
deg_mg_50C = []
deg_50C = []

for idx, row in test.iterrows():
    for pos in range(107):
        ids.append(f"{row['id']}_{pos}")

flat = preds.cpu().numpy().reshape(-1, 5)  # (N*68,5)
pad_len = len(ids) - flat.shape[0]
if pad_len > 0:
    flat = np.vstack([flat, np.zeros((pad_len, 5))])

react = flat[:, 0].tolist()
deg_mg_pH10 = flat[:, 1].tolist()
deg_pH10 = flat[:, 2].tolist()
deg_mg_50C = flat[:, 3].tolist()
deg_50C = flat[:, 4].tolist()

submission = pd.DataFrame(
    {
        "id_seqpos": ids,
        "reactivity": react,
        "deg_Mg_pH10": deg_mg_pH10,
        "deg_pH10": deg_pH10,
        "deg_Mg_50C": deg_mg_50C,
        "deg_50C": deg_50C,
    }
)

submission = submission.head(sub.shape[0])

submission.to_csv("submission.csv", index=False)

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/175729410.py in <cell line: 0>()
     12 
     13 # flatten predictions in the same order as ids (batch order is preserved)
---> 14 flat = preds.cpu().numpy().reshape(-1, 5)  # (N*68,5)
     15 # repeat each 68‑step prediction for the missing positions (set to 0)
     16 pad_len = len(ids) - flat.shape[0]

NameError: name 'preds' is not defined
