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

0.38829

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
from fastai.text.all import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader




## === cell 1
path = "/kaggle/input/stanford-covid-vaccine"
train = pd.read_json(f"{path}/train.json", lines=True)
test = pd.read_json(f"{path}/test.json", lines=True)
sub = pd.read_csv(f"{path}/sample_submission.csv")




## === cell 2
print(train.shape, test.shape, sub.shape)




## === cell 3
all1, all2, all3 = [], [], []
for i in range(len(train)):
    all1.extend(train["sequence"].loc[i])
    all2.extend(train["structure"].loc[i])
    all3.extend(train["predicted_loop_type"].loc[i])




## === cell 4
all1 = L(all1)
all2 = L(all2)
all3 = L(all3)




## === cell 5
vocab1 = all1.unique()
vocab2 = all2.unique()
vocab3 = all3.unique()




## === cell 6
word2idx1 = {w: i for i, w in enumerate(vocab1)}
word2idx2 = {w: i for i, w in enumerate(vocab2)}
word2idx3 = {w: i for i, w in enumerate(vocab3)}




## === cell 7
def joiner(row):
    l1 = list(row[0])
    l2 = list(row[1])
    l3 = list(row[2])
    out = [
        [word2idx1[l1[i]], word2idx2[l2[i]], word2idx3[l3[i]]] for i in range(len(l1))
    ]
    return out




## === cell 8
train["seqs"] = train[["sequence", "structure", "predicted_loop_type"]].apply(
    joiner, axis=1
)




## === cell 9
train = train[train["SN_filter"] == 1]




## === cell 10
txts = L([x for x in train["seqs"].values])
tgts1 = L([x for x in train["reactivity"].values])
tgts2 = L([x for x in train["deg_Mg_pH10"].values])
tgts3 = L([x for x in train["deg_pH10"].values])
tgts4 = L([x for x in train["deg_Mg_50C"].values])
tgts5 = L([x for x in train["deg_50C"].values])




## === cell 11
seqs = L(
    (tensor(txts[i]), tensor([tgts1[i], tgts2[i], tgts3[i], tgts4[i], tgts5[i]]))
    for i in range(len(txts))
)




## === cell 12
BS = 32  # batch size
ES = 32  # embedding size
NH = 512  # hidden units
NL = 2  # number of LSTM layers
DO = 0.5  # dropout
EP = 3  # epochs – enough for a quick run
LR = 3e-3  # learning rate
WD = 0.1  # weight decay




## === cell 13
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
        self.h = None

    def forward(self, x):
        e1 = self.i_h1(x[:, :, 0])
        e2 = self.i_h2(x[:, :, 1])
        e3 = self.i_h3(x[:, :, 2])
        e = torch.cat((e1, e2, e3), dim=2)
        if self.h is None:
            raw, h = self.rnn(e)
        else:
            raw, h = self.rnn(e, self.h)
        do = self.drop(raw)
        out = self.h_o(do)
        self.h = [h_.detach() for h_ in h]  # keep hidden for next batch
        if self.y_range is not None:
            out = (
                torch.sigmoid(out) * (self.y_range[1] - self.y_range[0])
                + self.y_range[0]
            )
        return out, raw, do

    def reset(self):
        self.h = None




## === cell 14
def loss_func(inp, targ):
    out = inp[0][:, :68, :]  # keep only the scored 68 positions
    l1 = F.mse_loss(out[:, :, 0], targ[:, 0, :])
    l2 = F.mse_loss(out[:, :, 1], targ[:, 1, :])
    l3 = F.mse_loss(out[:, :, 2], targ[:, 2, :])
    l4 = F.mse_loss(out[:, :, 3], targ[:, 3, :])
    l5 = F.mse_loss(out[:, :, 4], targ[:, 4, :])
    return torch.sqrt((l1 + l2 + l3 + l4 + l5) / 5)




## === cell 15
cut = int(len(seqs) * 0.8)




## === cell 16
dls = DataLoaders.from_dsets(
    seqs[:cut], seqs[cut:], bs=BS, drop_last=True, shuffle=True
)
dls.cuda()




## === cell 17
net = OVModel(len(vocab1), len(vocab2), len(vocab3), ES, NH, NL, DO).to(dls.device)




## === cell 18
learn = Learner(
    dls, net, loss_func=loss_func, cbs=ModelResetter
).to_fp16()  # keep original callbacks




## === cell 19
learn.fit_one_cycle(EP, LR, wd=WD)




## === cell 20
test["seqs"] = test[["sequence", "structure", "predicted_loop_type"]].apply(
    joiner, axis=1
)

test_seqs = [
    (torch.tensor(x[:107], dtype=torch.long), torch.zeros(5, 68, dtype=torch.float))
    for x in test["seqs"].values
]

test_dl = DataLoader(test_seqs, batch_size=BS, shuffle=False)




## === cell 21
net.eval()
preds = []
with torch.no_grad():
    for xb, _ in test_dl:
        xb = xb.to(dls.device)
        out, _, _ = net(xb)
        preds.append(out.cpu())
preds = torch.cat(preds, dim=0)  # shape: (num_test, 107, 5)




## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/79258983.py in <cell line: 0>()
      4     for xb, _ in test_dl:
      5         xb = xb.to(dls.device)
----> 6         out, _, _ = net(xb)
      7         preds.append(out.cpu())
      8 preds = torch.cat(preds, dim=0)  # shape: (num_test, 107, 5)

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

/tmp/ipykernel_55/2326899018.py in forward(self, x)
     31             raw, h = self.rnn(e)
     32         else:
---> 33             raw, h = self.rnn(e, self.h)
     34         do = self.drop(raw)
     35         out = self.h_o(do)

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

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/rnn.py in forward(self, input, hx)
   1118                 # Each batch of the hidden state should match the input sequence that
   1119                 # the user believes he/she is passing in.
-> 1120                 self.check_forward_args(input, hx, batch_sizes)
   1121                 hx = self.permute_hidden(hx, sorted_indices)
   1122 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/rnn.py in check_forward_args(self, input, hidden, batch_sizes)
   1001     ):
   1002         self.check_input(input, batch_sizes)
-> 1003         self.check_hidden_size(
   1004             hidden[0],
   1005             self.get_expected_hidden_size(input, batch_sizes),

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/rnn.py in check_hidden_size(self, hx, expected_hidden_size, msg)
    345     ) -> None:
    346         if hx.size() != expected_hidden_size:
--> 347             raise RuntimeError(msg.format(expected_hidden_size, list(hx.size())))
    348 
    349     def _weights_have_changed(self):

RuntimeError: Expected hidden[0] size (4, 16, 512), got [4, 32, 512]

## === cell 22
ids = []
for sid in test["id"]:
    ids.extend([f"{sid}_{pos}" for pos in range(107)])

sub_df = pd.DataFrame({"id_seqpos": ids})
sub_df["reactivity"] = preds[:, :, 0].flatten().numpy()
sub_df["deg_Mg_pH10"] = preds[:, :, 1].flatten().numpy()
sub_df["deg_pH10"] = preds[:, :, 2].flatten().numpy()
sub_df["deg_Mg_50C"] = preds[:, :, 3].flatten().numpy()
sub_df["deg_50C"] = preds[:, :, 4].flatten().numpy()




## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/798937357.py in <cell line: 0>()
      4 
      5 sub_df = pd.DataFrame({"id_seqpos": ids})
----> 6 sub_df["reactivity"] = preds[:, :, 0].flatten().numpy()
      7 sub_df["deg_Mg_pH10"] = preds[:, :, 1].flatten().numpy()
      8 sub_df["deg_pH10"] = preds[:, :, 2].flatten().numpy()

TypeError: list indices must be integers or slices, not tuple

## === cell 23
sub_df.to_csv("submission.csv", index=False)
print("submission.csv written – shape:", sub_df.shape)

## --- ERROR in outputing the csv:
Invalid submission: Expected the submission to have columns ['id_seqpos', 'reactivity', 'deg_Mg_pH10', 'deg_Mg_50C', 'deg_pH10', 'deg_50C'], but instead it has columns Index(['id_seqpos'], dtype='object').
