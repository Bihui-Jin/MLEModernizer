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

3.9

# 3. Installed packages

No external packages required in the script and installed.

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

0.3763091623635361

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 6

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os



## === cell 7
import json
import tensorflow as tf
from matplotlib import pyplot as plt

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 8
os.chdir('/kaggle/')
os.getcwd()

## === cell 10
train_data = pd.read_json('/kaggle/input/stanford-covid-vaccine/train.json', lines = True)
test_data = pd.read_json('/kaggle/input/stanford-covid-vaccine/test.json', lines = True)
submission_format = pd.read_csv('/kaggle/input/stanford-covid-vaccine/sample_submission.csv', encoding = 'utf-8-sig')

## === cell 12
train_data.head()

## === cell 13
train_data.shape

## === cell 14
train_data.groupby(['SN_filter']).size()

## === cell 15
test_data.head()

## === cell 16
test_data.shape

## === cell 17
submission_format.head()

## === cell 18
print(train_data.shape)
print(test_data.shape)
print(submission_format.shape)

## === cell 19
print('Training data:\n',train_data['seq_scored'].value_counts())
print('Test data:\n',test_data['seq_scored'].value_counts())
len(train_data['reactivity'].iloc[0])

## === cell 20
len(train_data['sequence'].iloc[0])

## === cell 21
flag = False
for i in range(0,len(train_data)):
    if(([x<0 for x in train_data['reactivity_error'].iloc[i]].count(True) > 0) |
       ([x<0 for x in train_data['deg_error_Mg_pH10'].iloc[i]].count(True) > 0) |
       ([x<0 for x in train_data['deg_error_pH10'].iloc[i]].count(True) > 0) |
       ([x<0 for x in train_data['deg_error_Mg_50C'].iloc[i]].count(True) > 0) |
       ([x<0 for x in train_data['deg_error_50C'].iloc[i]].count(True) > 0)):
        flag = True
print(flag)

## === cell 22
min_reactivity_value = min(train_data['reactivity'].iloc[0])
min_deg_Mg_pH10_value = min(train_data['deg_Mg_pH10'].iloc[0])
min_deg_pH10_value = min(train_data['deg_pH10'].iloc[0])
min_deg_Mg_50C_value = min(train_data['deg_Mg_50C'].iloc[0])
min_deg_Mg_50C_value = min(train_data['deg_50C'].iloc[0])

for i in range(0,len(train_data)):
    if(min(train_data['reactivity'].iloc[i]) < min_reactivity_value):
        min_reactivity_value = min(train_data['reactivity'].iloc[i])   

    if(min(train_data['deg_Mg_pH10'].iloc[i]) < min_deg_Mg_pH10_value):
        min_deg_Mg_pH10_value = min(train_data['deg_Mg_pH10'].iloc[i])

    if(min(train_data['deg_pH10'].iloc[i]) < min_deg_pH10_value):
        min_deg_pH10_value = min(train_data['deg_pH10'].iloc[i])

    if(min(train_data['deg_Mg_50C'].iloc[i]) < min_deg_Mg_50C_value):
        min_deg_Mg_50C_value = min(train_data['deg_Mg_50C'].iloc[i])

    if(min(train_data['deg_50C'].iloc[i]) < min_deg_Mg_50C_value):
        min_deg_Mg_50C_value = min(train_data['deg_50C'].iloc[i])
        
print(min_reactivity_value, min_deg_Mg_pH10_value, min_deg_pH10_value, min_deg_Mg_50C_value, min_deg_Mg_50C_value)

## === cell 23
train_data.head()

## === cell 27
train_data.columns

## === cell 28
token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')}
target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']

## === cell 29
token2int

## === cell 31
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
    bpps_nb_mean = 0.077522
    bpps_nb_std = 0.08914
    bpps_arr = []
    for mol_id in df.id.to_list():
        bpps = np.load(f"../input/stanford-covid-vaccine/bpps/{mol_id}.npy")
        bpps_nb = (bpps > 0).sum(axis=0) / bpps.shape[0]
        bpps_nb = (bpps_nb - bpps_nb_mean) / bpps_nb_std
        bpps_arr.append(bpps_nb)
    return bpps_arr 

os.chdir("/kaggle/working/")
train_data['bpps_sum'] = read_bpps_sum(train_data)
test_data['bpps_sum'] = read_bpps_sum(test_data)
train_data['bpps_max'] = read_bpps_max(train_data)
test_data['bpps_max'] = read_bpps_max(test_data)
train_data['bpps_nb'] = read_bpps_nb(train_data)
test_data['bpps_nb'] = read_bpps_nb(test_data)

train_data.head()

## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1296318261.py in <cell line: 0>()
     24 
     25 os.chdir("/kaggle/working/")
---> 26 train_data['bpps_sum'] = read_bpps_sum(train_data)
     27 test_data['bpps_sum'] = read_bpps_sum(test_data)
     28 train_data['bpps_max'] = read_bpps_max(train_data)

/tmp/ipykernel_11/1296318261.py in read_bpps_sum(df)
      2     bpps_arr = []
      3     for mol_id in df.id.to_list():
----> 4         bpps_arr.append(np.load(f"../input/stanford-covid-vaccine/bpps/{mol_id}.npy").sum(axis=1))
      5     return bpps_arr
      6 

/usr/local/lib/python3.11/dist-packages/numpy/lib/npyio.py in load(file, mmap_mode, allow_pickle, fix_imports, encoding, max_header_size)
    425             own_fid = False
    426         else:
--> 427             fid = stack.enter_context(open(os_fspath(file), "rb"))
    428             own_fid = True
    429 

FileNotFoundError: [Errno 2] No such file or directory: '../input/stanford-covid-vaccine/bpps/id_001f94081.npy'

## === cell 32
import plotly.express as px
from collections import Counter as count

def get_bases(data):
    bases = []

    for j in range(len(data)):
        counts = dict(count(data.iloc[j]['sequence']))
        bases.append((
            counts['A'] / 107,
            counts['G'] / 107,
            counts['C'] / 107,
            counts['U'] / 107
        ))

    bases = pd.DataFrame(bases, columns=['A_percent', 'G_percent', 'C_percent', 'U_percent'])
    return bases

## === cell 33
def get_pairs_rate(data):
    pairs_rate = []

    for j in range(len(data)):
        res = dict(count(data.iloc[j]['structure']))
        pairs_rate.append(res['('] / 53.5)

    pairs_rate = pd.DataFrame(pairs_rate, columns=['pairs_rate'])
    return pairs_rate

## === cell 34
def get_pairs(data):
    pairs = []
    all_partners = []
    for j in range(len(data)):
        partners = [-1 for i in range(130)]
        pairs_dict = {}
        queue = []
        for i in range(0, len(data.iloc[j]['structure'])):
            if data.iloc[j]['structure'][i] == '(':
                queue.append(i)
            if data.iloc[j]['structure'][i] == ')':
                first = queue.pop()
                try:
                    pairs_dict[(data.iloc[j]['sequence'][first], data.iloc[j]['sequence'][i])] += 1
                except:
                    pairs_dict[(data.iloc[j]['sequence'][first], data.iloc[j]['sequence'][i])] = 1

                partners[first] = i
                partners[i] = first

        all_partners.append(partners)

        pairs_num = 0
        pairs_unique = [('U', 'G'), ('C', 'G'), ('U', 'A'), ('G', 'C'), ('A', 'U'), ('G', 'U')]
        for item in pairs_dict:
            pairs_num += pairs_dict[item]
        add_tuple = list()
        for item in pairs_unique:
            try:
                add_tuple.append(pairs_dict[item]/pairs_num)
            except:
                add_tuple.append(0)
        pairs.append(add_tuple)

    pairs = pd.DataFrame(pairs, columns=['U-G', 'C-G', 'U-A', 'G-C', 'A-U', 'G-U'])
    return pairs

## === cell 35
def get_loops(data):
    loops = []
    for j in range(len(data)):
        counts = dict(count(data.iloc[j]['predicted_loop_type']))
        available = ['E', 'S', 'H', 'B', 'X', 'I', 'M']
        row = []
        for item in available:
            try:
                row.append(counts[item] / 107)
            except:
                row.append(0)
        loops.append(row)

    loops = pd.DataFrame(loops, columns=available)
    return loops

## === cell 36
from tqdm.notebook import tqdm

def get_structure_adj(train):
    
    Ss = []
    for i in tqdm(range(len(train))):
        seq_length = train["seq_length"].iloc[i]
        structure = train["structure"].iloc[i]
        sequence = train["sequence"].iloc[i]

        cue = []
        a_structures = {
            ("A", "U") : np.zeros([seq_length, seq_length]),
            ("C", "G") : np.zeros([seq_length, seq_length]),
            ("U", "G") : np.zeros([seq_length, seq_length]),
            ("U", "A") : np.zeros([seq_length, seq_length]),
            ("G", "C") : np.zeros([seq_length, seq_length]),
            ("G", "U") : np.zeros([seq_length, seq_length]),
        }
        a_structure = np.zeros([seq_length, seq_length])
        for j in range(seq_length):
            if structure[j] == "(":
                cue.append(j)
            elif structure[j] == ")":
                start = cue.pop()
                a_structures[(sequence[start], sequence[j])][start, j] = 1
                a_structures[(sequence[j], sequence[start])][j, start] = 1
        
        a_strc = np.stack([a for a in a_structures.values()], axis = 2)
        a_strc = np.sum(a_strc, axis = 2, keepdims = True)
        Ss.append(a_strc)
    
    Ss = np.array(Ss)
    print(Ss.shape)
    return Ss

## === cell 37
As = []
data = train_data[train_data['signal_to_noise'] > 1].copy()
for id in tqdm(data['id']):
    a = np.load(f"/kaggle/input/stanford-covid-vaccine/bpps/{id}.npy")
    As.append(a)
As = np.array(As)

## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/4263429728.py in <cell line: 0>()
      2 data = train_data[train_data['signal_to_noise'] > 1].copy()
      3 for id in tqdm(data['id']):
----> 4     a = np.load(f"/kaggle/input/stanford-covid-vaccine/bpps/{id}.npy")
      5     As.append(a)
      6 As = np.array(As)

/usr/local/lib/python3.11/dist-packages/numpy/lib/npyio.py in load(file, mmap_mode, allow_pickle, fix_imports, encoding, max_header_size)
    425             own_fid = False
    426         else:
--> 427             fid = stack.enter_context(open(os_fspath(file), "rb"))
    428             own_fid = True
    429 

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/stanford-covid-vaccine/bpps/id_001f94081.npy'

## === cell 38
def get_distance_matrix(As):
    
    idx = np.arange(As.shape[1])
    Ds = []
    for i in range(len(idx)):
        d = np.abs(idx[i] - idx)
        Ds.append(d)

    Ds = np.array(Ds) + 1
    Ds = 1/Ds
    Ds = Ds[None, :,:]
    Ds = np.repeat(Ds, len(As), axis = 0)
    
    Dss = []
    for i in [1, 2, 4]: 
        Dss.append(Ds ** i)
    Ds = np.stack(Dss, axis = 3)
    print(Ds.shape)
    return Ds

## === cell 39
def preprocess_inputs(df, cols=['sequence', 'structure', 'predicted_loop_type'], seq_length = 107, flag = 'train'):
    base_fea = np.transpose(
        np.array(
            df[cols]
            .applymap(lambda seq: [token2int[x] for x in seq])
            .values
            .tolist()
        ),
        (0, 2, 1)
    )

    bpps_sum_fea = np.array(df['bpps_sum'].to_list())[:,:,np.newaxis]
    bpps_max_fea = np.array(df['bpps_max'].to_list())[:,:,np.newaxis]
    bpps_nb_fea = np.array(df['bpps_nb'].to_list())[:,:,np.newaxis]

    Ss = get_structure_adj(df)
    Ss = Ss.sum(axis = 1)
    
    if flag == 'train':
        Ds = get_distance_matrix(As)
    elif flag == 'test_private':
        Ds = get_distance_matrix(As_private)
    elif flag == 'test_public':
        Ds = get_distance_matrix(As_public)
    Ds = Ds.sum(axis = 1)
    
    data = np.concatenate([base_fea,bpps_sum_fea,bpps_max_fea,bpps_nb_fea, Ss, Ds], 2)
    
    array_data = (np.reshape(([([list(df['A_percent'])[0]] * seq_length)]),(1,seq_length,1)))
    for i in range(1,len(df)):
        array_data_i = (np.reshape(([([list(df['A_percent'])[i]] * seq_length)]),(1,seq_length,1)))
        array_data = np.concatenate([array_data,array_data_i], axis = 0)

    data = np.concatenate([data, array_data], 2)

    for col in ['G_percent','C_percent','U_percent','U-G','C-G','U-A','G-C','A-U','G-U',
                'E','S','H','B','X','I','M','pairs_rate']:
        array_data = (np.reshape(([([list(df[col])[0]] * seq_length)]),(1,seq_length,1)))
        for i in range(1,len(df)):
            arraydom_data_i = (np.reshape(([([list(df[col])[i]] * seq_length)]),(1,seq_length,1)))
            array_data = np.concatenate([array_data,array_data_i], axis = 0)

        data = np.concatenate([data, array_data], 2)

    return data

## === cell 40
bases = get_bases(train_data)
pairs = get_pairs(train_data)
loops = get_loops(train_data)
pairs_rate = get_pairs_rate(train_data)
train_data = pd.concat([train_data, bases, pairs, loops, pairs_rate], axis=1)

bases = get_bases(test_data)
pairs = get_pairs(test_data)
loops = get_loops(test_data)
pairs_rate = get_pairs_rate(test_data)
test_data = pd.concat([test_data, bases, pairs, loops, pairs_rate], axis=1)

## === cell 41
train_inputs = preprocess_inputs(train_data.loc[train_data['signal_to_noise'] > 1], seq_length = 107, flag = 'train')
train_labels = np.array(train_data.loc[train_data['signal_to_noise'] > 1][target_cols].values.tolist()).transpose((0, 2, 1))

## --- ERROR in cell 41, traceback:
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

KeyError: 'bpps_sum'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1443857701.py in <cell line: 0>()
----> 1 train_inputs = preprocess_inputs(train_data.loc[train_data['signal_to_noise'] > 1], seq_length = 107, flag = 'train')
      2 train_labels = np.array(train_data.loc[train_data['signal_to_noise'] > 1][target_cols].values.tolist()).transpose((0, 2, 1))

/tmp/ipykernel_11/3217117268.py in preprocess_inputs(df, cols, seq_length, flag)
     10     )
     11 
---> 12     bpps_sum_fea = np.array(df['bpps_sum'].to_list())[:,:,np.newaxis]
     13     bpps_max_fea = np.array(df['bpps_max'].to_list())[:,:,np.newaxis]
     14     bpps_nb_fea = np.array(df['bpps_nb'].to_list())[:,:,np.newaxis]

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

KeyError: 'bpps_sum'

## === cell 42
train_inputs.shape

## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4107804673.py in <cell line: 0>()
----> 1 train_inputs.shape

NameError: name 'train_inputs' is not defined

## === cell 43
from keras.losses import mean_squared_error

def root_mean_squared_error(y_true, y_pred):
    return tf.sqrt(mean_squared_error(y_true, y_pred))

def MCRMSE(y_true, y_pred):
    colwise_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)
    return tf.reduce_mean(tf.sqrt(colwise_mse), axis=1)

def lstm_layer(hidden_dim, dropout):
    return tf.keras.layers.Bidirectional(
                                tf.keras.layers.LSTM(hidden_dim,
                                dropout=dropout,
                                return_sequences=True,
                                kernel_initializer = 'orthogonal'))

def gru_layer(hidden_dim, dropout):
    return tf.keras.layers.Bidirectional(
                                tf.keras.layers.GRU(hidden_dim,
                                dropout=dropout,
                                return_sequences=True,
                                kernel_initializer='orthogonal'))

def build_model(n_layers = 2, seq_len = 107, num_features = 28, embed_dim = 200, sp_dropout = 0.2, hidden_dim = 512, dropout = 0.5, pred_len = 68, gru_flag = False):
    
    inputs = tf.keras.layers.Input(shape=(seq_len, num_features))
    categorical_feats = inputs[:, :, :3]
    numerical_feats = inputs[:, :, 3:10]
    overall_gene_feats = inputs[:, :, 10:]

    embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=embed_dim)(categorical_feats)
   
    reshaped = tf.reshape(embed, shape=(-1, embed.shape[1],  embed.shape[2] * embed.shape[3]))
    
    reshaped_1 = tf.keras.layers.concatenate([reshaped, numerical_feats], axis=2)
          
    normalized_layer_1 = tf.keras.layers.BatchNormalization()(reshaped_1)
    
    spatial_dropout = tf.keras.layers.SpatialDropout1D(sp_dropout)(normalized_layer_1)
  
    if gru_flag:
        for x in range(n_layers):
            normalized_layer_1 = gru_layer(hidden_dim, dropout)(normalized_layer_1)
        normalized_layer_2 = tf.keras.layers.BatchNormalization()(normalized_layer_1)
    else:
        for x in range(n_layers):
            normalized_layer_1 = lstm_layer(hidden_dim, dropout)(normalized_layer_1)
        normalized_layer_2 = tf.keras.layers.BatchNormalization()(normalized_layer_1)
    
    concat_layer = tf.keras.layers.concatenate([normalized_layer_2, overall_gene_feats], axis=2)
    
    dense_layer_1 = tf.keras.layers.Dense(100, activation = 'linear')(concat_layer)
    normalized_layer_3 = tf.keras.layers.BatchNormalization()(dense_layer_1)
    dropout_layer_1 = tf.keras.layers.SpatialDropout1D(sp_dropout)(normalized_layer_3)
   
    dense_layer_2 = tf.keras.layers.Dense(100, activation = 'linear')(dropout_layer_1)
    normalized_layer_4 = tf.keras.layers.BatchNormalization()(dense_layer_2)
    dropout_layer_2 = tf.keras.layers.SpatialDropout1D(sp_dropout)(normalized_layer_4)
    
    truncated = dropout_layer_2[:, :pred_len]

    out = tf.keras.layers.Dense(5, activation='linear')(truncated)

    model = tf.keras.Model(inputs=inputs, outputs=out)

    adam = tf.optimizers.Adam()

    model.compile(optimizer = adam, loss = MCRMSE)
    
    return model

## === cell 44
train_inputs.shape

## --- ERROR in cell 44, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4107804673.py in <cell line: 0>()
----> 1 train_inputs.shape

NameError: name 'train_inputs' is not defined

## === cell 49
public_df = test_data.query("seq_length == 107").copy()
private_df = test_data.query("seq_length == 130").copy()

## === cell 50
As_public = []
for id in tqdm(public_df["id"]):
    a = np.load(f"/kaggle/input/stanford-covid-vaccine/bpps/{id}.npy")
    As_public.append(a)
As_public = np.array(As_public)
As_private = []
for id in tqdm(private_df["id"]):
    a = np.load(f"/kaggle/input/stanford-covid-vaccine/bpps/{id}.npy")
    As_private.append(a)
As_private = np.array(As_private)

## --- ERROR in cell 50, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3384915679.py in <cell line: 0>()
      1 As_public = []
      2 for id in tqdm(public_df["id"]):
----> 3     a = np.load(f"/kaggle/input/stanford-covid-vaccine/bpps/{id}.npy")
      4     As_public.append(a)
      5 As_public = np.array(As_public)

/usr/local/lib/python3.11/dist-packages/numpy/lib/npyio.py in load(file, mmap_mode, allow_pickle, fix_imports, encoding, max_header_size)
    425             own_fid = False
    426         else:
--> 427             fid = stack.enter_context(open(os_fspath(file), "rb"))
    428             own_fid = True
    429 

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/stanford-covid-vaccine/bpps/id_00b436dec.npy'

## === cell 51
public_inputs = preprocess_inputs(public_df, seq_length = 107, flag = 'test_public')
private_inputs = preprocess_inputs(private_df, seq_length = 130, flag = 'test_private')

## --- ERROR in cell 51, traceback:
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

KeyError: 'bpps_sum'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2144260428.py in <cell line: 0>()
----> 1 public_inputs = preprocess_inputs(public_df, seq_length = 107, flag = 'test_public')
      2 private_inputs = preprocess_inputs(private_df, seq_length = 130, flag = 'test_private')

/tmp/ipykernel_11/3217117268.py in preprocess_inputs(df, cols, seq_length, flag)
     10     )
     11 
---> 12     bpps_sum_fea = np.array(df['bpps_sum'].to_list())[:,:,np.newaxis]
     13     bpps_max_fea = np.array(df['bpps_max'].to_list())[:,:,np.newaxis]
     14     bpps_nb_fea = np.array(df['bpps_nb'].to_list())[:,:,np.newaxis]

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

KeyError: 'bpps_sum'

## === cell 52
model_LSTM_on_test_data_public = build_model(seq_len=107, pred_len=107, gru_flag = False)
model_LSTM_on_test_data_public.load_weights('../input/openvaccine-covid-model-weights/LSTM model.h5')
pred_test_data_public_LSTM = model_LSTM_on_test_data_public.predict(public_inputs)

model_GRU_on_test_data_public = build_model(seq_len=107, pred_len=107, gru_flag = True)
model_GRU_on_test_data_public.load_weights('../input/openvaccine-covid-model-weights/GRU model (1).h5')
pred_test_data_public_GRU = model_GRU_on_test_data_public.predict(public_inputs)

## --- ERROR in cell 52, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2651245333.py in <cell line: 0>()
----> 1 model_LSTM_on_test_data_public = build_model(seq_len=107, pred_len=107, gru_flag = False)
      2 model_LSTM_on_test_data_public.load_weights('../input/openvaccine-covid-model-weights/LSTM model.h5')
      3 pred_test_data_public_LSTM = model_LSTM_on_test_data_public.predict(public_inputs)
      4 
      5 model_GRU_on_test_data_public = build_model(seq_len=107, pred_len=107, gru_flag = True)

/tmp/ipykernel_11/1431818362.py in build_model(n_layers, seq_len, num_features, embed_dim, sp_dropout, hidden_dim, dropout, pred_len, gru_flag)
     31     embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=embed_dim)(categorical_feats)
     32 
---> 33     reshaped = tf.reshape(embed, shape=(-1, embed.shape[1],  embed.shape[2] * embed.shape[3]))
     34 
     35     reshaped_1 = tf.keras.layers.concatenate([reshaped, numerical_feats], axis=2)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/weak_tensor_ops.py in wrapper(*args, **kwargs)
     86   def wrapper(*args, **kwargs):
     87     if not ops.is_auto_dtype_conversion_enabled():
---> 88       return op(*args, **kwargs)
     89     bound_arguments = signature.bind(*args, **kwargs)
     90     bound_arguments.apply_defaults()

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/keras_tensor.py in __tf_tensor__(self, dtype, name)
    136 
    137     def __tf_tensor__(self, dtype=None, name=None):
--> 138         raise ValueError(
    139             "A KerasTensor cannot be used as input to a TensorFlow function. "
    140             "A KerasTensor is a symbolic placeholder for a shape and dtype, "

ValueError: A KerasTensor cannot be used as input to a TensorFlow function. A KerasTensor is a symbolic placeholder for a shape and dtype, used when constructing Keras Functional models or Keras Functions. You can only use it as input to a Keras layer or a Keras operation (from the namespaces `keras.layers` and `keras.operations`). You are likely doing something like:

```
x = Input(...)
...
tf_fn(x)  # Invalid.
```

What you should do instead is wrap `tf_fn` in a layer:

```
class MyLayer(Layer):
    def call(self, x):
        return tf_fn(x)

x = MyLayer()(x)
```


## === cell 53
model_LSTM_on_test_data_private = build_model(seq_len=130, pred_len=130, gru_flag = False)
model_LSTM_on_test_data_private.load_weights('../input/openvaccine-covid-model-weights/LSTM model.h5')
pred_test_data_private_LSTM = model_LSTM_on_test_data_private.predict(private_inputs)

model_GRU_on_test_data_private = build_model(seq_len=130, pred_len=130, gru_flag = True)
model_GRU_on_test_data_private.load_weights('../input/openvaccine-covid-model-weights/GRU model (1).h5')
pred_test_data_private_GRU = model_GRU_on_test_data_private.predict(private_inputs)

## --- ERROR in cell 53, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1467740276.py in <cell line: 0>()
----> 1 model_LSTM_on_test_data_private = build_model(seq_len=130, pred_len=130, gru_flag = False)
      2 model_LSTM_on_test_data_private.load_weights('../input/openvaccine-covid-model-weights/LSTM model.h5')
      3 pred_test_data_private_LSTM = model_LSTM_on_test_data_private.predict(private_inputs)
      4 
      5 model_GRU_on_test_data_private = build_model(seq_len=130, pred_len=130, gru_flag = True)

/tmp/ipykernel_11/1431818362.py in build_model(n_layers, seq_len, num_features, embed_dim, sp_dropout, hidden_dim, dropout, pred_len, gru_flag)
     31     embed = tf.keras.layers.Embedding(input_dim=len(token2int), output_dim=embed_dim)(categorical_feats)
     32 
---> 33     reshaped = tf.reshape(embed, shape=(-1, embed.shape[1],  embed.shape[2] * embed.shape[3]))
     34 
     35     reshaped_1 = tf.keras.layers.concatenate([reshaped, numerical_feats], axis=2)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/weak_tensor_ops.py in wrapper(*args, **kwargs)
     86   def wrapper(*args, **kwargs):
     87     if not ops.is_auto_dtype_conversion_enabled():
---> 88       return op(*args, **kwargs)
     89     bound_arguments = signature.bind(*args, **kwargs)
     90     bound_arguments.apply_defaults()

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/keras_tensor.py in __tf_tensor__(self, dtype, name)
    136 
    137     def __tf_tensor__(self, dtype=None, name=None):
--> 138         raise ValueError(
    139             "A KerasTensor cannot be used as input to a TensorFlow function. "
    140             "A KerasTensor is a symbolic placeholder for a shape and dtype, "

ValueError: A KerasTensor cannot be used as input to a TensorFlow function. A KerasTensor is a symbolic placeholder for a shape and dtype, used when constructing Keras Functional models or Keras Functions. You can only use it as input to a Keras layer or a Keras operation (from the namespaces `keras.layers` and `keras.operations`). You are likely doing something like:

```
x = Input(...)
...
tf_fn(x)  # Invalid.
```

What you should do instead is wrap `tf_fn` in a layer:

```
class MyLayer(Layer):
    def call(self, x):
        return tf_fn(x)

x = MyLayer()(x)
```


## === cell 54
def format_predictions(public_preds, private_preds):
    preds = []
    
    for df, preds_ in [(public_df, public_preds), (private_df, private_preds)]:
        for i, uid in enumerate(df.id):
            single_pred = preds_[i]

            single_df = pd.DataFrame(single_pred, columns=target_cols)
            single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]

            preds.append(single_df)
    return pd.concat(preds).reset_index(drop = True)

## === cell 55
lstm_preds = format_predictions(pred_test_data_public_LSTM, pred_test_data_private_LSTM)
gru_preds = format_predictions(pred_test_data_public_GRU, pred_test_data_private_GRU)

## --- ERROR in cell 55, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/537825428.py in <cell line: 0>()
----> 1 lstm_preds = format_predictions(pred_test_data_public_LSTM, pred_test_data_private_LSTM)
      2 gru_preds = format_predictions(pred_test_data_public_GRU, pred_test_data_private_GRU)

NameError: name 'pred_test_data_public_LSTM' is not defined

## === cell 56
lstm_preds.head()

## --- ERROR in cell 56, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/187545891.py in <cell line: 0>()
----> 1 lstm_preds.head()

NameError: name 'lstm_preds' is not defined

## === cell 57
gru_preds.head()

## --- ERROR in cell 57, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3192410269.py in <cell line: 0>()
----> 1 gru_preds.head()

NameError: name 'gru_preds' is not defined

## === cell 58
submission_LSTM = submission_format[['id_seqpos']].merge(lstm_preds, how = 'inner', on = 'id_seqpos')
submission_GRU = submission_format[['id_seqpos']].merge(gru_preds, how = 'inner', on = 'id_seqpos')

## --- ERROR in cell 58, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/987302572.py in <cell line: 0>()
----> 1 submission_LSTM = submission_format[['id_seqpos']].merge(lstm_preds, how = 'inner', on = 'id_seqpos')
      2 submission_GRU = submission_format[['id_seqpos']].merge(gru_preds, how = 'inner', on = 'id_seqpos')

NameError: name 'lstm_preds' is not defined

## === cell 59
print(submission_LSTM.shape)
submission_LSTM.head()

## --- ERROR in cell 59, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/314558667.py in <cell line: 0>()
----> 1 print(submission_LSTM.shape)
      2 submission_LSTM.head()

NameError: name 'submission_LSTM' is not defined

## === cell 60
print(submission_GRU.shape)
submission_GRU.head()

## --- ERROR in cell 60, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/662349557.py in <cell line: 0>()
----> 1 print(submission_GRU.shape)
      2 submission_GRU.head()

NameError: name 'submission_GRU' is not defined

## === cell 61
target_cols

## === cell 62
submission_lstm_gru_combined = submission_GRU.merge(submission_LSTM, how = 'inner', on = 'id_seqpos')

gru_weight = 0.5
lstm_weight = 0.5
for i in range(len(target_cols)):
    submission_lstm_gru_combined[target_cols[i]] = submission_lstm_gru_combined[target_cols[i]+'_x']*gru_weight + submission_lstm_gru_combined[target_cols[i]+'_y']*lstm_weight

## --- ERROR in cell 62, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/10698350.py in <cell line: 0>()
----> 1 submission_lstm_gru_combined = submission_GRU.merge(submission_LSTM, how = 'inner', on = 'id_seqpos')
      2 
      3 gru_weight = 0.5
      4 lstm_weight = 0.5
      5 for i in range(len(target_cols)):

NameError: name 'submission_GRU' is not defined

## === cell 63
submission_lstm_gru_combined = submission_lstm_gru_combined[['id_seqpos'] + target_cols]

## --- ERROR in cell 63, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/556028319.py in <cell line: 0>()
----> 1 submission_lstm_gru_combined = submission_lstm_gru_combined[['id_seqpos'] + target_cols]

NameError: name 'submission_lstm_gru_combined' is not defined

## === cell 64
submission_lstm_gru_combined.head()

## --- ERROR in cell 64, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/57021014.py in <cell line: 0>()
----> 1 submission_lstm_gru_combined.head()

NameError: name 'submission_lstm_gru_combined' is not defined

## === cell 65
os.chdir("/kaggle/working/")
submission_LSTM.to_csv('submission_LSTM.csv', index = False)
submission_GRU.to_csv('submission_GRU.csv', index = False)
submission_lstm_gru_combined.to_csv('submission_lstm_gru_combined.csv', index = False)

## --- ERROR in cell 65, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3481409283.py in <cell line: 0>()
      1 os.chdir("/kaggle/working/")
----> 2 submission_LSTM.to_csv('submission_LSTM.csv', index = False)
      3 submission_GRU.to_csv('submission_GRU.csv', index = False)
      4 submission_lstm_gru_combined.to_csv('submission_lstm_gru_combined.csv', index = False)

NameError: name 'submission_LSTM' is not defined
