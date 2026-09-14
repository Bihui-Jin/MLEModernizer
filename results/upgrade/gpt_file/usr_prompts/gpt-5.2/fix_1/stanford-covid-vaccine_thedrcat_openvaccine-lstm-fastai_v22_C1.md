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

0.3717

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
!pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html -q
!pip install fastai==2.0.13 -q


## === cell 1
from fastai.text.all import *
import pandas as pd
import numpy as np
from tqdm.autonotebook import tqdm
from torch import nn
from sklearn.model_selection import KFold


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/numpy/_core/__init__.py in <module>
     20 
     21 try:
---> 22     from . import multiarray
     23 except ImportError as exc:
     24     import sys

/usr/local/lib/python3.11/dist-packages/numpy/_core/multiarray.py in <module>
      9 import functools
     10 
---> 11 from . import _multiarray_umath, overrides
     12 from ._multiarray_umath import *  # noqa: F403
     13 

ImportError: cannot load module more than once per process

## === cell 2
path = '/kaggle/input/stanford-covid-vaccine'
train = pd.read_json(f'{path}/train.json',lines=True)
test = pd.read_json(f'{path}/test.json', lines=True)
sub = pd.read_csv(f'{path}/sample_submission.csv')


## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1215796067.py in <cell line: 0>()
      1 path = '/kaggle/input/stanford-covid-vaccine'
----> 2 train = pd.read_json(f'{path}/train.json',lines=True)
      3 test = pd.read_json(f'{path}/test.json', lines=True)
      4 sub = pd.read_csv(f'{path}/sample_submission.csv')

NameError: name 'pd' is not defined

## === cell 3
train.shape, train['id'].nunique(), test.shape, sub.shape


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2149844321.py in <cell line: 0>()
----> 1 train.shape, train['id'].nunique(), test.shape, sub.shape

NameError: name 'train' is not defined

## === cell 4

def read_bpps_sum(df):
    bpps_arr = []
    for mol_id in df.id.to_list():
        bpps_arr.append(np.load(f"../input/stanford-covid-vaccine/bpps/{mol_id}.npy").sum(axis=1))
    return bpps_arr

def read_bpps_max(df):
    bpps_arr = []
    for mol_id in df.id.to_list():
        bpps_arr.append(np.load(f"../input/stanford-covid-vaccine/bpps/{mol_id}.npy").max(axis=1))
    return bpps_arr

def read_bpps_nb(df):
    bpps_nb_mean = 0.077522 # mean of bpps_nb across all training data
    bpps_nb_std = 0.08914   # std of bpps_nb across all training data
    bpps_arr = []
    for mol_id in df.id.to_list():
        bpps = np.load(f"../input/stanford-covid-vaccine/bpps/{mol_id}.npy")
        bpps_nb = (bpps > 0).sum(axis=0) / bpps.shape[0]
        bpps_nb = (bpps_nb - bpps_nb_mean) / bpps_nb_std
        bpps_arr.append(bpps_nb)
    return bpps_arr 

train['bpps_sum'] = read_bpps_sum(train)
test['bpps_sum'] = read_bpps_sum(test)
train['bpps_max'] = read_bpps_max(train)
test['bpps_max'] = read_bpps_max(test)
train['bpps_nb'] = read_bpps_nb(train)
test['bpps_nb'] = read_bpps_nb(test)


## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1286438174.py in <cell line: 0>()
     26     return bpps_arr
     27 
---> 28 train['bpps_sum'] = read_bpps_sum(train)
     29 test['bpps_sum'] = read_bpps_sum(test)
     30 train['bpps_max'] = read_bpps_max(train)

NameError: name 'train' is not defined

## === cell 5
train = train.sample(frac=1, random_state=42)


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/857898435.py in <cell line: 0>()
----> 1 train = train.sample(frac=1, random_state=42)

NameError: name 'train' is not defined

## === cell 6
all1 = []
all2 = []
all3 = []
for i in range(len(train)):
    all1.extend(train['sequence'].loc[i])
    all2.extend(train['structure'].loc[i])
    all3.extend(train['predicted_loop_type'].loc[i])


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3061529500.py in <cell line: 0>()
      2 all2 = []
      3 all3 = []
----> 4 for i in range(len(train)):
      5     all1.extend(train['sequence'].loc[i])
      6     all2.extend(train['structure'].loc[i])

NameError: name 'train' is not defined

## === cell 7
all1 = L(all1)
all2 = L(all2)
all3 = L(all3)


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1138642764.py in <cell line: 0>()
----> 1 all1 = L(all1)
      2 all2 = L(all2)
      3 all3 = L(all3)

NameError: name 'L' is not defined

## === cell 8
vocab1 = all1.unique()
vocab2 = all2.unique()
vocab3 = all3.unique()


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2215064394.py in <cell line: 0>()
----> 1 vocab1 = all1.unique()
      2 vocab2 = all2.unique()
      3 vocab3 = all3.unique()

AttributeError: 'list' object has no attribute 'unique'

## === cell 9
word2idx1 = {w:i for i,w in enumerate(vocab1)}
word2idx2 = {w:i for i,w in enumerate(vocab2)}
word2idx3 = {w:i for i,w in enumerate(vocab3)}


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3152211321.py in <cell line: 0>()
----> 1 word2idx1 = {w:i for i,w in enumerate(vocab1)}
      2 word2idx2 = {w:i for i,w in enumerate(vocab2)}
      3 word2idx3 = {w:i for i,w in enumerate(vocab3)}

NameError: name 'vocab1' is not defined

## === cell 10
def joiner(row):
    l1 =  list(row[0])
    l2 =  list(row[1])
    l3 =  list(row[2])
    l4 =  list(row[3])
    l5 =  list(row[4])
    l6 =  list(row[5])
    out = [[word2idx1[l1[i]], word2idx2[l2[i]], word2idx3[l3[i]], l4[i], l5[i], l6[i]] for i in range(len(l1))]
    return out


## === cell 11
train['seqs'] = train[['sequence', 'structure', 'predicted_loop_type', 'bpps_sum', 'bpps_max', 'bpps_nb']].apply(joiner, axis=1)


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4200957343.py in <cell line: 0>()
----> 1 train['seqs'] = train[['sequence', 'structure', 'predicted_loop_type', 'bpps_sum', 'bpps_max', 'bpps_nb']].apply(joiner, axis=1)

NameError: name 'train' is not defined

## === cell 12
train = train[train['SN_filter'] == 1]


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3401125560.py in <cell line: 0>()
----> 1 train = train[train['SN_filter'] == 1]

NameError: name 'train' is not defined

## === cell 13
txts = L([x for x in train['seqs'].values])
tgts1 = L([x for x in train['reactivity'].values])
tgts2 = L([x for x in train['deg_Mg_pH10'].values])
tgts3 = L([x for x in train['deg_pH10'].values])
tgts4 = L([x for x in train['deg_Mg_50C'].values])
tgts5 = L([x for x in train['deg_50C'].values])


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/525619739.py in <cell line: 0>()
----> 1 txts = L([x for x in train['seqs'].values])
      2 tgts1 = L([x for x in train['reactivity'].values])
      3 tgts2 = L([x for x in train['deg_Mg_pH10'].values])
      4 tgts3 = L([x for x in train['deg_pH10'].values])
      5 tgts4 = L([x for x in train['deg_Mg_50C'].values])

NameError: name 'L' is not defined

## === cell 14
seqs = L((tensor(txts[i]), tensor([tgts1[i], tgts2[i], tgts3[i], tgts4[i], tgts5[i]])) for i in range(len(txts)))


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1857580970.py in <cell line: 0>()
----> 1 seqs = L((tensor(txts[i]), tensor([tgts1[i], tgts2[i], tgts3[i], tgts4[i], tgts5[i]])) for i in range(len(txts)))

NameError: name 'L' is not defined

## === cell 15
test['seqs'] = test[['sequence', 'structure', 'predicted_loop_type', 'bpps_sum', 'bpps_max', 'bpps_nb']].apply(joiner, axis=1)


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4077121197.py in <cell line: 0>()
----> 1 test['seqs'] = test[['sequence', 'structure', 'predicted_loop_type', 'bpps_sum', 'bpps_max', 'bpps_nb']].apply(joiner, axis=1)

NameError: name 'test' is not defined

## === cell 16
test_ids = pd.DataFrame()
test_ids['id'] = test['id']
for i in range(14):
    test_ids.loc[len(test_ids)] = 'id_dummy'
test_ids['seqnum'] = ''    
test_ids['seqnum'] = test_ids['seqnum'].astype(object)
sn = np.array(list(range(107)))
for i in range(len(test_ids)):
    test_ids['seqnum'].loc[i] = sn


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/605996572.py in <cell line: 0>()
----> 1 test_ids = pd.DataFrame()
      2 test_ids['id'] = test['id']
      3 for i in range(14):
      4     test_ids.loc[len(test_ids)] = 'id_dummy'
      5 test_ids['seqnum'] = ''

NameError: name 'pd' is not defined

## === cell 17
test_ids = test_ids.explode('seqnum').reset_index(drop=True)
test_ids['id_seqpos'] = test_ids.apply(lambda r: str(r[0]) + '_' + str(r[1]), axis=1)


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3342797605.py in <cell line: 0>()
----> 1 test_ids = test_ids.explode('seqnum').reset_index(drop=True)
      2 test_ids['id_seqpos'] = test_ids.apply(lambda r: str(r[0]) + '_' + str(r[1]), axis=1)

NameError: name 'test_ids' is not defined

## === cell 18
test_seqs = [(tensor(x[:107]), torch.zeros(5,68)) for x in test['seqs'].values]
len(test_seqs)
test_seqs_empty = [(torch.zeros((107,6), dtype=torch.long), torch.zeros(5,68)) for i in range(14)]
test_seqs += test_seqs_empty
test_seqs = L(test_seqs)
len(test_seqs), len(test_seqs) % 32


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2161757190.py in <cell line: 0>()
      1 #need to fix this later - predict full sequence 130 vs 107 only
----> 2 test_seqs = [(tensor(x[:107]), torch.zeros(5,68)) for x in test['seqs'].values]
      3 len(test_seqs)
      4 #14 empty seqs to fill up the batch :/
      5 test_seqs_empty = [(torch.zeros((107,6), dtype=torch.long), torch.zeros(5,68)) for i in range(14)]

NameError: name 'test' is not defined

## === cell 19
test_seqs = [(a.to('cuda'), b.to('cuda')) for (a,b) in test_seqs]


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3905554898.py in <cell line: 0>()
----> 1 test_seqs = [(a.to('cuda'), b.to('cuda')) for (a,b) in test_seqs]

NameError: name 'test_seqs' is not defined

## === cell 20
BS = 32 # batch size 
ES = 32 # embedding size
NH = 512 # number hidden units
NL = 2 # number layers
DO = 0.5 # dropout
EP = 18 # epochs
LR = 3e-3 # learning rate
WD = 0.1 # weight decay


## === cell 21
sl = 107

class OVModel(Module):
    def __init__(self, vocab1_sz, vocab2_sz, vocab3_sz, emb_sz, n_hidden, n_layers, p, y_range=None):
        self.y_range = y_range
        self.i_h1 = nn.Embedding(vocab1_sz, emb_sz)
        self.i_h2 = nn.Embedding(vocab2_sz, emb_sz)
        self.i_h3 = nn.Embedding(vocab3_sz, emb_sz)
        self.rnn = nn.LSTM(emb_sz*3+3, n_hidden, n_layers, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(p)
        self.h_o = nn.Linear(n_hidden*2, 5)
        self.h = [torch.zeros(n_layers*2, BS, n_hidden).to('cuda') for _ in range(2)]
        
    def forward(self, x):
        e1 = self.i_h1(x[:,:,0].long())
        e2 = self.i_h2(x[:,:,1].long())
        e3 = self.i_h3(x[:,:,2].long())
        bp = x[:,:,3:]
        e = torch.cat((e1, e2, e3, bp), dim=2)
        raw,h = self.rnn(e, self.h)
        do = self.drop(raw)
        out = self.h_o(do)
        if self.y_range is None: 
            self.h = [h_.detach() for h_ in h]
            return out, raw, do        
        out = torch.sigmoid(out) * (self.y_range[1]-self.y_range[0]) + self.y_range[0]
        self.h = [h_.detach() for h_ in h]
        return out, raw, do
    
    def reset(self): 
        for h in self.h: h.zero_()


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1394780409.py in <cell line: 0>()
      1 sl = 107
      2 
----> 3 class OVModel(Module):
      4     def __init__(self, vocab1_sz, vocab2_sz, vocab3_sz, emb_sz, n_hidden, n_layers, p, y_range=None):
      5         self.y_range = y_range

NameError: name 'Module' is not defined

## === cell 22
def loss_func(inp, targ):
    inp = inp[0]
    inp = inp[:,:68,:]
    l1 = F.mse_loss(inp[:,:,0], targ[:,0,:])
    l2 = F.mse_loss(inp[:,:,1], targ[:,1,:])
    l3 = F.mse_loss(inp[:,:,2], targ[:,2,:])
    l4 = F.mse_loss(inp[:,:,3], targ[:,3,:])
    l5 = F.mse_loss(inp[:,:,4], targ[:,4,:])
    return torch.sqrt((l1 + l2 + l3 + l4 +l5)/5)


## === cell 23
test_dl = DataLoader(dataset=test_seqs, bs=BS, shuffle=False, drop_last=True)


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/368942647.py in <cell line: 0>()
----> 1 test_dl = DataLoader(dataset=test_seqs, bs=BS, shuffle=False, drop_last=True)

NameError: name 'DataLoader' is not defined

## === cell 24
spltidx = np.array(range(len(seqs)))
kf = KFold(n_splits=5)
splts = list(kf.split(spltidx))


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2616323908.py in <cell line: 0>()
----> 1 spltidx = np.array(range(len(seqs)))
      2 kf = KFold(n_splits=5)
      3 splts = list(kf.split(spltidx))

NameError: name 'np' is not defined

## === cell 25
all_preds = []

for i in range(5):
    dls = DataLoaders.from_dsets(seqs[splts[i][0]], seqs[splts[i][1]], bs=BS, drop_last=True, shuffle=True).cuda()
    net = OVModel(len(vocab1), len(vocab2), len(vocab3), ES, NH, NL, DO, y_range=None)
    learn = Learner(dls, net, loss_func=loss_func, cbs=ModelResetter)
    learn.fit_one_cycle(EP, LR, wd=WD)
    preds = learn.get_preds(dl=test_dl, reorder=False)
    all_preds.append(preds[0][0])


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1219382184.py in <cell line: 0>()
      2 
      3 for i in range(5):
----> 4     dls = DataLoaders.from_dsets(seqs[splts[i][0]], seqs[splts[i][1]], bs=BS, drop_last=True, shuffle=True).cuda()
      5     net = OVModel(len(vocab1), len(vocab2), len(vocab3), ES, NH, NL, DO, y_range=None)
      6     learn = Learner(dls, net, loss_func=loss_func, cbs=ModelResetter)

NameError: name 'DataLoaders' is not defined

## === cell 26
predictions = sum(all_preds) / len(all_preds)
predictions.shape


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
ZeroDivisionError                         Traceback (most recent call last)
/tmp/ipykernel_11/4199935768.py in <cell line: 0>()
----> 1 predictions = sum(all_preds) / len(all_preds)
      2 predictions.shape

ZeroDivisionError: division by zero

## === cell 27
s = pd.DataFrame()
s['id_seqpos'] = test_ids['id_seqpos']
s['reactivity'] = predictions[:,:,0].flatten().numpy().tolist()
s['deg_Mg_pH10'] = predictions[:,:,1].flatten().numpy().tolist()
s['deg_pH10'] = predictions[:,:,2].flatten().numpy().tolist()
s['deg_Mg_50C'] = predictions[:,:,3].flatten().numpy().tolist()
s['deg_50C'] = predictions[:,:,4].flatten().numpy().tolist()
s = s.iloc[:-14*107]


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/610467869.py in <cell line: 0>()
----> 1 s = pd.DataFrame()
      2 s['id_seqpos'] = test_ids['id_seqpos']
      3 s['reactivity'] = predictions[:,:,0].flatten().numpy().tolist()
      4 s['deg_Mg_pH10'] = predictions[:,:,1].flatten().numpy().tolist()
      5 s['deg_pH10'] = predictions[:,:,2].flatten().numpy().tolist()

NameError: name 'pd' is not defined

## === cell 28
sub['seqpos'] = sub['id_seqpos'].apply(lambda r: int(r.rsplit('_',1)[1]))
sub = sub[sub['seqpos'] > 106]
sub = sub[s.columns]
s = pd.concat([s, sub], axis=0)


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4133300136.py in <cell line: 0>()
----> 1 sub['seqpos'] = sub['id_seqpos'].apply(lambda r: int(r.rsplit('_',1)[1]))
      2 sub = sub[sub['seqpos'] > 106]
      3 sub = sub[s.columns]
      4 s = pd.concat([s, sub], axis=0)

NameError: name 'sub' is not defined

## === cell 29
s.to_csv('submission.csv', index=False)


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3133415782.py in <cell line: 0>()
----> 1 s.to_csv('submission.csv', index=False)

NameError: name 's' is not defined

## === cell 30
s.head()


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2430826787.py in <cell line: 0>()
----> 1 s.head()

NameError: name 's' is not defined
