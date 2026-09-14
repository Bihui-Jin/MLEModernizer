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

0.37567

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
from tqdm.auto import tqdm

from fastai.text.all import *
from fastai.data.core import DataLoaders

torch.backends.cudnn.benchmark = True

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



## === cell 1
path = "/kaggle/input/stanford-covid-vaccine"
if not os.path.exists(path):
    path = "/kaggle/data/stanford-covid-vaccine"
if not os.path.exists(path):
    path = "/kaggle/input/stanford-covid-vaccine/stanford-covid-vaccine"

train = pd.read_json(f"{path}/train.json", lines=True)
test = pd.read_json(f"{path}/test.json", lines=True)
sub = pd.read_csv(f"{path}/sample_submission.csv")

train.shape, train["id"].nunique(), test.shape, sub.shape



## === cell 2
assert "id_seqpos" in sub.columns
assert train["seq_length"].iloc[0] == 107 and test["seq_length"].iloc[0] == 107
assert train["seq_scored"].iloc[0] == 68 and test["seq_scored"].iloc[0] == 68
train.head(2)



## === cell 3

BPPS_DIR = os.path.join(path, "bpps")
if not os.path.exists(BPPS_DIR):
    BPPS_DIR = "/kaggle/input/stanford-covid-vaccine/bpps"


def read_bpps_sum(df):
    bpps_arr = []
    for mol_id in tqdm(df.id.to_list(), desc="bpps_sum"):
        bpps_arr.append(np.load(f"{BPPS_DIR}/{mol_id}.npy").sum(axis=1))
    return bpps_arr


def read_bpps_max(df):
    bpps_arr = []
    for mol_id in tqdm(df.id.to_list(), desc="bpps_max"):
        bpps_arr.append(np.load(f"{BPPS_DIR}/{mol_id}.npy").max(axis=1))
    return bpps_arr


def read_bpps_nb(df):
    bpps_nb_mean = 0.077522  # mean of bpps_nb across all training data
    bpps_nb_std = 0.08914  # std of bpps_nb across all training data
    bpps_arr = []
    for mol_id in tqdm(df.id.to_list(), desc="bpps_nb"):
        bpps = np.load(f"{BPPS_DIR}/{mol_id}.npy")
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

train[["bpps_sum", "bpps_max", "bpps_nb"]].head(1)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1060480205.py in <cell line: 0>()
     34 
     35 
---> 36 train["bpps_sum"] = read_bpps_sum(train)
     37 test["bpps_sum"] = read_bpps_sum(test)
     38 train["bpps_max"] = read_bpps_max(train)

/tmp/ipykernel_11/1060480205.py in read_bpps_sum(df)
     11     bpps_arr = []
     12     for mol_id in tqdm(df.id.to_list(), desc="bpps_sum"):
---> 13         bpps_arr.append(np.load(f"{BPPS_DIR}/{mol_id}.npy").sum(axis=1))
     14     return bpps_arr
     15 

/usr/local/lib/python3.11/dist-packages/numpy/lib/npyio.py in load(file, mmap_mode, allow_pickle, fix_imports, encoding, max_header_size)
    425             own_fid = False
    426         else:
--> 427             fid = stack.enter_context(open(os_fspath(file), "rb"))
    428             own_fid = True
    429 

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/stanford-covid-vaccine/bpps/id_001f94081.npy'

## === cell 4
all1, all2, all3 = [], [], []
for i in range(len(train)):
    all1.extend(train["sequence"].iloc[i])
    all2.extend(train["structure"].iloc[i])
    all3.extend(train["predicted_loop_type"].iloc[i])

all1 = L(all1)
all2 = L(all2)
all3 = L(all3)

vocab1 = all1.unique()
vocab2 = all2.unique()
vocab3 = all3.unique()

word2idx1 = {w: i for i, w in enumerate(vocab1)}
word2idx2 = {w: i for i, w in enumerate(vocab2)}
word2idx3 = {w: i for i, w in enumerate(vocab3)}

len(vocab1), len(vocab2), len(vocab3), vocab1, vocab2, vocab3




## === cell 5
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

train["seqs"].iloc[0][:2], len(train["seqs"].iloc[0])



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/623020634.py in <cell line: 0>()
     14 
     15 
---> 16 train["seqs"] = train[
     17     ["sequence", "structure", "predicted_loop_type", "bpps_sum", "bpps_max", "bpps_nb"]
     18 ].apply(joiner, axis=1)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['bpps_sum', 'bpps_max', 'bpps_nb'] not in index"

## === cell 6
train = train[train["SN_filter"] == 1].reset_index(drop=True)
train.shape



## === cell 7
txts = L([x for x in train["seqs"].values])
tgts1 = L([x for x in train["reactivity"].values])
tgts2 = L([x for x in train["deg_Mg_pH10"].values])
tgts3 = L([x for x in train["deg_pH10"].values])
tgts4 = L([x for x in train["deg_Mg_50C"].values])
tgts5 = L([x for x in train["deg_50C"].values])

seqs = L(
    (tensor(txts[i]), tensor([tgts1[i], tgts2[i], tgts3[i], tgts4[i], tgts5[i]]))
    for i in range(len(txts))
)
len(seqs), seqs[0][0].shape, seqs[0][1].shape



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'seqs'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1230213656.py in <cell line: 0>()
      1 # Prepare training tensors; keep original target organization (5 x 68).
----> 2 txts = L([x for x in train["seqs"].values])
      3 tgts1 = L([x for x in train["reactivity"].values])
      4 tgts2 = L([x for x in train["deg_Mg_pH10"].values])
      5 tgts3 = L([x for x in train["deg_pH10"].values])

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'seqs'

## === cell 8
BS = 32  # batch size
ES = 32  # embedding size
NH = 512  # number hidden units
NL = 2  # number layers
DO = 0.5  # dropout
EP = 18  # epochs
LR = 3e-3  # learning rate
WD = 0.1  # weight decay

sl = 107



## === cell 9


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
        bs=BS,
    ):
        self.y_range = y_range
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.bs = bs

        self.i_h1 = nn.Embedding(vocab1_sz, emb_sz)
        self.i_h2 = nn.Embedding(vocab2_sz, emb_sz)
        self.i_h3 = nn.Embedding(vocab3_sz, emb_sz)
        self.rnn = nn.LSTM(
            emb_sz * 3 + 3, n_hidden, n_layers, batch_first=True, bidirectional=True
        )
        self.drop = nn.Dropout(p)
        self.h_o = nn.Linear(n_hidden * 2, 5)

        self.h = None

    def _init_hidden(self, bs, device_):
        return [
            torch.zeros(self.n_layers * 2, bs, self.n_hidden, device=device_)
            for _ in range(2)
        ]

    def forward(self, x):
        bs = x.size(0)
        dev = x.device
        if (self.h is None) or (self.h[0].size(1) != bs) or (self.h[0].device != dev):
            self.h = self._init_hidden(bs, dev)

        e1 = self.i_h1(x[:, :, 0].long())
        e2 = self.i_h2(x[:, :, 1].long())
        e3 = self.i_h3(x[:, :, 2].long())
        bp = x[:, :, 3:]
        e = torch.cat((e1, e2, e3, bp), dim=2)

        raw, h = self.rnn(e, self.h)
        do = self.drop(raw)
        out = self.h_o(do)

        if self.y_range is not None:
            out = (
                torch.sigmoid(out) * (self.y_range[1] - self.y_range[0])
                + self.y_range[0]
            )

        self.h = [h_.detach() for h_ in h]
        return out, raw, do

    def reset(self):
        if self.h is None:
            return
        for h in self.h:
            h.zero_()




## === cell 10
def loss_func(inp, targ):
    inp = inp[0]
    inp = inp[:, :68, :]
    l1 = F.mse_loss(inp[:, :, 0], targ[:, 0, :])
    l2 = F.mse_loss(inp[:, :, 1], targ[:, 1, :])
    l3 = F.mse_loss(inp[:, :, 2], targ[:, 2, :])
    l4 = F.mse_loss(inp[:, :, 3], targ[:, 3, :])
    l5 = F.mse_loss(inp[:, :, 4], targ[:, 4, :])
    return torch.sqrt((l1 + l2 + l3 + l4 + l5) / 5)




## === cell 11
cut = int(len(seqs) * 0.8)
dls = DataLoaders.from_dsets(
    seqs[:cut], seqs[cut:], bs=BS, drop_last=True, shuffle=True
)
dls = dls.to(device)
cut, len(dls.train), len(dls.valid)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1800727399.py in <cell line: 0>()
      1 # Split and dataloaders; keep original behavior (drop_last=True for training stability).
----> 2 cut = int(len(seqs) * 0.8)
      3 dls = DataLoaders.from_dsets(
      4     seqs[:cut], seqs[cut:], bs=BS, drop_last=True, shuffle=True
      5 )

NameError: name 'seqs' is not defined

## === cell 12
net = OVModel(
    len(vocab1), len(vocab2), len(vocab3), ES, NH, NL, DO, y_range=None, bs=BS
).to(device)
learn = Learner(dls, net, loss_func=loss_func, cbs=ModelResetter)

batch = dls.one_batch()
out = net(batch[0])
(out[0].shape, batch[1].shape, batch[0].shape)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1555983143.py in <cell line: 0>()
      2     len(vocab1), len(vocab2), len(vocab3), ES, NH, NL, DO, y_range=None, bs=BS
      3 ).to(device)
----> 4 learn = Learner(dls, net, loss_func=loss_func, cbs=ModelResetter)
      5 
      6 # quick shape test

NameError: name 'dls' is not defined

## === cell 13
learn.fit_one_cycle(EP, LR, wd=WD)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/480904053.py in <cell line: 0>()
      1 # Train as original
----> 2 learn.fit_one_cycle(EP, LR, wd=WD)
      3 

NameError: name 'learn' is not defined

## === cell 14
test_seqs = [(tensor(x[:107]), torch.zeros(5, 68)) for x in test["seqs"].values]
test_seqs = L(test_seqs)

test_dl = DataLoader(dataset=test_seqs, batch_size=BS, shuffle=False, drop_last=False)

preds = learn.get_preds(dl=test_dl, reorder=False)
predictions = preds[0][0]  # (n_test, 107, 5)
predictions.shape



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'seqs'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2278799111.py in <cell line: 0>()
      1 # Build test DataLoader WITHOUT drop_last and WITHOUT dummy padding,
      2 # so predictions align exactly to test rows and submission template.
----> 3 test_seqs = [(tensor(x[:107]), torch.zeros(5, 68)) for x in test["seqs"].values]
      4 test_seqs = L(test_seqs)
      5 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'seqs'

## === cell 15
pred_np = predictions.detach().cpu().numpy()  # (n_test,107,5)

id_to_pred = {test["id"].iloc[i]: pred_np[i] for i in range(len(test))}


def parse_id_seqpos(id_seqpos):
    mol_id, pos = id_seqpos.rsplit("_", 1)
    return mol_id, int(pos)


out = sub[["id_seqpos"]].copy()
mol_ids, poss = zip(*out["id_seqpos"].map(parse_id_seqpos))
mol_ids = np.array(mol_ids, dtype=object)
poss = np.array(poss, dtype=int)

targets = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
pred_cols = {t: np.zeros(len(out), dtype=np.float32) for t in targets}

for i in range(len(out)):
    mid = mol_ids[i]
    p = poss[i]
    pred_mat = id_to_pred.get(mid, None)
    if pred_mat is None:
        continue
    pred_cols["reactivity"][i] = pred_mat[p, 0]
    pred_cols["deg_Mg_pH10"][i] = pred_mat[p, 1]
    pred_cols["deg_pH10"][i] = pred_mat[p, 2]
    pred_cols["deg_Mg_50C"][i] = pred_mat[p, 3]
    pred_cols["deg_50C"][i] = pred_mat[p, 4]

for t in targets:
    out[t] = pred_cols[t]

out = out[["id_seqpos"] + targets]
out.head(), out.shape



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/373629447.py in <cell line: 0>()
      1 # Create submission by merging into sample_submission order exactly.
      2 # This fixes earlier mismatched/incorrect id_seqpos construction.
----> 3 pred_np = predictions.detach().cpu().numpy()  # (n_test,107,5)
      4 
      5 # Map each test id to its prediction matrix

NameError: name 'predictions' is not defined

## === cell 16
out.to_csv("submission.csv", index=False)

assert os.path.exists("submission.csv")
assert list(out.columns) == [
    "id_seqpos",
    "reactivity",
    "deg_Mg_pH10",
    "deg_pH10",
    "deg_Mg_50C",
    "deg_50C",
]
out.tail()

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1493886628.py in <cell line: 0>()
      1 # Write valid submission
----> 2 out.to_csv("submission.csv", index=False)
      3 
      4 # sanity checks
      5 assert os.path.exists("submission.csv")

NameError: name 'out' is not defined
