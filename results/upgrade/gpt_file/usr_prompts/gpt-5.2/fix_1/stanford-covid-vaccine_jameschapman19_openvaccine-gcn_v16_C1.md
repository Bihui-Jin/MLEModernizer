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

0.51889

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import torch
print(torch.__version__)
print(torch.version.cuda)


## === cell 1
!pip install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
!pip install torch-sparse==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
!pip install torch-cluster==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
!pip install torch-spline-conv==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
!pip install torch-geometric


## === cell 3
import warnings
warnings.filterwarnings('ignore')

import os
import shutil

import pandas as pd, numpy as np, seaborn as sns
import math, json
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from matplotlib import cm
import seaborn as sns
import colorsys
from tqdm import tqdm

from sklearn.model_selection import train_test_split, KFold

import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


## === cell 4
!conda install -y -c bioconda forgi
!conda install -y -c bioconda viennarna

import forgi.graph.bulge_graph as fgb
import forgi.visual.mplotlib as fvm
import forgi.threedee.utilities.vector as ftuv
import forgi

import RNA


## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/2932686769.py in <cell line: 0>()
      2 get_ipython().system('conda install -y -c bioconda viennarna')
      3 
----> 4 import forgi.graph.bulge_graph as fgb
      5 import forgi.visual.mplotlib as fvm
      6 import forgi.threedee.utilities.vector as ftuv

ModuleNotFoundError: No module named 'forgi'

## === cell 5
def seed_all(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
seed_all()


## === cell 6
class config:
    learning_rate = 0.001
    K = 1 # number of aggregation loop (also means number of GCN layers)
    gcn_agg = 'mean' # aggregator function: mean, conv, lstm, pooling
    filter_noise = False
    seed = 1234


## === cell 7
def get_couples(structure):
    """
    For each closing parenthesis, I find the matching opening one and store their index in the couples list.
    The assigned list is used to keep track of the assigned opening parenthesis
    """
    opened = [idx for idx, i in enumerate(structure) if i == '(']
    closed = [idx for idx, i in enumerate(structure) if i == ')']

    assert len(opened) == len(closed)


    assigned = []
    couples = []

    for close_idx in closed:
        for open_idx in opened:
            if open_idx < close_idx:
                if open_idx not in assigned:
                    candidate = open_idx
            else:
                break
        assigned.append(candidate)
        couples.append([candidate, close_idx])
        assigned.append(close_idx)
        couples.append([close_idx, candidate])
        
    assert len(couples) == 2*len(opened)
    
    return couples


## === cell 8
def build_matrix(couples,size):
    mat = np.zeros((size, size))
    
    for i in range(size):  # neigbouring bases are linked as well
        if i < size - 1:
            mat[i, i + 1] = 1
        if i > 0:
            mat[i, i - 1] = 1
    
    for i, j in couples:
        mat[i, j] = 2
        mat[j, i] = 2
        
    return mat


## === cell 9
def seq2nodes(sequence,loops,structures):
    type_dict={'A':0,'G':1,'U':2,'C':3}
    loop_dict={'S':0,'M':1,'I':2,'B':3,'H':4,'E':5,'X':6}
    struct_dict={'.':0,'(':1,')':2}
    nodes=np.zeros((len(sequence),4+7+3))
    for i,s in enumerate(sequence):
        nodes[i,type_dict[s]]=1
    for i,s in enumerate(loops):
        nodes[i,4+loop_dict[s]]=1
    for i,s in enumerate(structures):
        nodes[i,11+struct_dict[s]]=1
    return nodes


## === cell 10
all_data=pd.read_json('../input/stanford-covid-vaccine/train.json',lines=True)
all_data.head(5)


## === cell 11
idx = 0
id_=all_data.iloc[idx].id
sequence = all_data.iloc[idx].sequence
structure = all_data.iloc[idx].structure
loops=all_data.iloc[idx].predicted_loop_type
reactivity = all_data.iloc[idx].reactivity
bg = fgb.BulgeGraph.from_fasta_text(f'>rna1\n{structure}\n{sequence}')[0]
fig = plt.figure(figsize=(6, 6))

fvm.plot_rna(bg, lighten=0.5, text_kwargs={"fontweight":None})
plt.show()


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3441451981.py in <cell line: 0>()
      5 loops=all_data.iloc[idx].predicted_loop_type
      6 reactivity = all_data.iloc[idx].reactivity
----> 7 bg = fgb.BulgeGraph.from_fasta_text(f'>rna1\n{structure}\n{sequence}')[0]
      8 fig = plt.figure(figsize=(6, 6))
      9 

NameError: name 'fgb' is not defined

## === cell 12
matrix=build_matrix(get_couples(structure),len(sequence))
bpps_dir='../input/stanford-covid-vaccine/bpps/'
bpps=np.load(bpps_dir+id_+'.npy')
edge_index=np.stack(np.where((matrix+bpps)>0))
node_attr=seq2nodes(sequence,loops,structure)
edge_attr=np.zeros((edge_index.shape[1],3))
edge_attr[:,0]=(matrix==1)[edge_index[0,:],edge_index[1,:]]
edge_attr[:,1]=(matrix==2)[edge_index[0,:],edge_index[1,:]]
edge_attr[:,2]=bpps[edge_index[0,:],edge_index[1,:]]


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1925657703.py in <cell line: 0>()
      1 matrix=build_matrix(get_couples(structure),len(sequence))
      2 bpps_dir='../input/stanford-covid-vaccine/bpps/'
----> 3 bpps=np.load(bpps_dir+id_+'.npy')
      4 edge_index=np.stack(np.where((matrix+bpps)>0))
      5 #adjacents=np.stack(np.where(matrix==1))

/usr/local/lib/python3.11/dist-packages/numpy/lib/npyio.py in load(file, mmap_mode, allow_pickle, fix_imports, encoding, max_header_size)
    425             own_fid = False
    426         else:
--> 427             fid = stack.enter_context(open(os_fspath(file), "rb"))
    428             own_fid = True
    429 

FileNotFoundError: [Errno 2] No such file or directory: '../input/stanford-covid-vaccine/bpps/id_001f94081.npy'

## === cell 13
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data

class MyOwnDataset(InMemoryDataset):
    def __init__(self, root='',train=True, public=True, ids=None,transform=None, pre_transform=None):
        try:
            shutil.rmtree('./'+root)
        except:
            print("doesn't exist")
        self.train=train
        if self.train:
            self.data_dir = '../input/stanford-covid-vaccine/train.json'
        else:
            self.data_dir = '/kaggle/input/stanford-covid-vaccine/test.json'
        self.bpps_dir='../input/stanford-covid-vaccine/bpps/'
        self.df=pd.read_json(self.data_dir,lines=True)
        if self.train:
            if config.filter_noise:
                self.df = self.df[self.df.signal_to_noise > 1]
        if ids is not None:
            self.df=self.df[self.df['index'].isin(ids)]
        if public:
            self.df=self.df.query("seq_length == 107")
        else:
            self.df=self.df.query("seq_length == 130")
        self.target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C','deg_pH10', 'deg_50C']
        
        super(MyOwnDataset, self).__init__(root,transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    
    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass

    def process(self):
        data_list = []
        for idx in range(len(self.df)):
            structure=self.df['structure'].iloc[idx]
            sequence=self.df['sequence'].iloc[idx]
            loops=self.df['predicted_loop_type'].iloc[idx]
            matrix=build_matrix(get_couples(structure),len(sequence))
            id_=self.df['id'].iloc[idx]
            bpps=np.load(self.bpps_dir+id_+'.npy')
            edge_index=np.stack(np.where((matrix)>0))
            node_attr=seq2nodes(sequence,loops,structure)
            edge_attr=np.zeros((edge_index.shape[1],3))
            edge_attr[:,0]=(matrix==1)[edge_index[0,:],edge_index[1,:]]
            edge_attr[:,1]=(matrix==2)[edge_index[0,:],edge_index[1,:]]
            edge_attr[:,2]=bpps[edge_index[0,:],edge_index[1,:]]
            if self.train:
                targets=np.stack(self.df[self.target_cols].iloc[idx]).T
            else:
                targets=np.zeros((130,5))
            x = torch.from_numpy(node_attr)
            y = torch.from_numpy(targets)
            edge_attr=torch.from_numpy(edge_attr)
            edge_index=torch.tensor(edge_index,dtype=torch.long)
            data = Data(x=x, edge_index=edge_index,edge_attr=edge_attr, y=y)
            data.train_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
            data.train_mask[:68] = 1
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


## === cell 14
if config.filter_noise:
    all_data = all_data[all_data.signal_to_noise > 1]
all_ids=np.arange(len(all_data))
np.random.shuffle(all_ids)
train_ids,val_ids=np.split(all_ids, [int(round(0.9 * len(all_ids), 0))])

train_dataset=MyOwnDataset(ids=train_ids, root='train')
val_dataset=MyOwnDataset(ids=val_ids, root='val')

from torch_geometric.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2421535975.py in <cell line: 0>()
      5 train_ids,val_ids=np.split(all_ids, [int(round(0.9 * len(all_ids), 0))])
      6 
----> 7 train_dataset=MyOwnDataset(ids=train_ids, root='train')
      8 val_dataset=MyOwnDataset(ids=val_ids, root='val')
      9 

/tmp/ipykernel_11/1803442915.py in __init__(self, root, train, public, ids, transform, pre_transform)
     26         self.target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C','deg_pH10', 'deg_50C']
     27 
---> 28         super(MyOwnDataset, self).__init__(root,transform, pre_transform)
     29         self.data, self.slices = torch.load(self.processed_paths[0])
     30 

/usr/local/lib/python3.11/dist-packages/torch_geometric/data/in_memory_dataset.py in __init__(self, root, transform, pre_transform, pre_filter, log, force_reload)
     79         force_reload: bool = False,
     80     ) -> None:
---> 81         super().__init__(root, transform, pre_transform, pre_filter, log,
     82                          force_reload)
     83 

/usr/local/lib/python3.11/dist-packages/torch_geometric/data/dataset.py in __init__(self, root, transform, pre_transform, pre_filter, log, force_reload)
    113 
    114         if self.has_process:
--> 115             self._process()
    116 
    117     def indices(self) -> Sequence:

/usr/local/lib/python3.11/dist-packages/torch_geometric/data/dataset.py in _process(self)
    263 
    264         fs.makedirs(self.processed_dir, exist_ok=True)
--> 265         self.process()
    266 
    267         path = osp.join(self.processed_dir, 'pre_transform.pt')

/tmp/ipykernel_11/1803442915.py in process(self)
     52             # nodes x features
     53             id_=self.df['id'].iloc[idx]
---> 54             bpps=np.load(self.bpps_dir+id_+'.npy')
     55             edge_index=np.stack(np.where((matrix)>0))
     56             node_attr=seq2nodes(sequence,loops,structure)

/usr/local/lib/python3.11/dist-packages/numpy/lib/npyio.py in load(file, mmap_mode, allow_pickle, fix_imports, encoding, max_header_size)
    425             own_fid = False
    426         else:
--> 427             fid = stack.enter_context(open(os_fspath(file), "rb"))
    428             own_fid = True
    429 

FileNotFoundError: [Errno 2] No such file or directory: '../input/stanford-covid-vaccine/bpps/id_001f94081.npy'

## === cell 15
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNNet(torch.nn.Module):
    def __init__(self, node_feats,channels,out_feats,edge_feats=1):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(node_feats, channels)
        self.conv2 = GCNConv(channels, channels)
        self.conv3 = GCNConv(channels, channels)
        self.conv4 = GCNConv(channels, channels)
        self.conv5 = GCNConv(channels, channels)
        self.conv9 = GCNConv(channels, out_feats)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv5(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv9(x, edge_index)
        return x
    

import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU
from torch_geometric.nn import NNConv, Set2Set    

class MPNNet(torch.nn.Module):
    def __init__(self, node_feats,channels,out_feats,loops=1,edge_feats=1):
        super(MPNNet, self).__init__()
        self.lin0 = torch.nn.Linear(node_feats,channels)
        self.loops=loops
        nn = Sequential(Linear(edge_feats, 64), ReLU(), Linear(64, channels * channels))
        self.conv = NNConv(channels, channels, nn, aggr='mean')
        self.gru = GRU(channels, channels)

        self.lin1 = torch.nn.Linear(channels, channels)
        self.lin2 = torch.nn.Linear(channels, out_feats)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(self.loops):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out


## === cell 16
node_feats=train_dataset.num_node_features
out_feats=train_dataset.num_classes
edge_feats=train_dataset.num_edge_features


model = MPNNet(node_feats,32,out_feats,loops=10,edge_feats=edge_feats).double()
print(sum(p.numel() for p in model.parameters()))
optimizer = torch.optim.Adam(model.parameters())


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1369947319.py in <cell line: 0>()
----> 1 node_feats=train_dataset.num_node_features
      2 out_feats=train_dataset.num_classes
      3 edge_feats=train_dataset.num_edge_features
      4 
      5 

NameError: name 'train_dataset' is not defined

## === cell 17
class MCRMSELoss(torch.nn.Module):
    def __init__(self):
        super(MCRMSELoss,self).__init__()

    def forward(self,x,y):
        x=x[:,:3]
        y=y[:,:3]
        msq_error=torch.mean((x-y)**2,0)
        loss=torch.mean(torch.sqrt(msq_error))
        return loss


## === cell 18
class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


## === cell 19
from torch.nn import MSELoss
import gc
loss_fn = MCRMSELoss()

def train(model,optimizer,train_loader):
    model.train()
    train_loss = AverageMeter()
    for batch_idx,data in enumerate(train_loader):# Iterate in batches over the training dataset.
        out = model(data.to(device))  # Perform a single forward pass.
        loss = loss_fn(out[data.train_mask], data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()
        train_loss.update(loss.item())
    return train_loss.avg

def test(model,val_loader):
    model.eval()
    val_loss = AverageMeter()
    for batch_idx,data in enumerate(val_loader):  # Iterate in batches over the training/test dataset.
        out = model(data.to(device))
        loss=loss_fn(out[data.train_mask], data.y)
        val_loss.update(loss.item())  # Compute the loss. # Check against ground-truth labels.
    return val_loss.avg


## === cell 20
def train_loop(model,epochs=1):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=config.learning_rate)
    train_loss = []
    val_loss = []
    for epoch in range(1, epochs+1):
        train_acc = train(model,optimizer,train_loader)
        val_acc = test(model,val_loader)
        train_loss.append(train_acc)
        val_loss.append(val_acc)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {val_acc:.4f}')
    return model, train_loss, val_loss


## === cell 21
model, train_loss, val_loss = train_loop(model,epochs=50)


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1190105618.py in <cell line: 0>()
----> 1 model, train_loss, val_loss = train_loop(model,epochs=50)

NameError: name 'model' is not defined

## === cell 22
def custom_plot_rna(cg, coloring, ax=None):
    '''
    Edited from https://github.com/ViennaRNA/forgi/blob/master/forgi/visual/mplotlib.py
    '''
    RNA.cvar.rna_plot_type = 1
    coords = []
    bp_string = cg.to_dotbracket_string()
    if ax is None:
        ax = plt.gca()
    vrna_coords = RNA.get_xy_coordinates(bp_string)
    
    for i, _ in enumerate(bp_string):
        coord = (vrna_coords.get(i).X, vrna_coords.get(i).Y)
        coords.append(coord)
    coords = np.array(coords)
    
    for i, coord in enumerate(coords):
        if i < len(coloring):
            c = cm.coolwarm(coloring[i])
        else: 
            c = 'grey'
        h,l,s = colorsys.rgb_to_hls(*mc.to_rgb(c))
        c=colorsys.hls_to_rgb(h,l,s)
        circle = plt.Circle((coord[0], coord[1]),color=c)
        ax.add_artist(circle)

    datalim = ((min(list(coords[:, 0]) + [ax.get_xlim()[0]]),
                min(list(coords[:, 1]) + [ax.get_ylim()[0]])),
               (max(list(coords[:, 0]) + [ax.get_xlim()[1]]),
                max(list(coords[:, 1]) + [ax.get_ylim()[1]])))

    width = datalim[1][0] - datalim[0][0]
    height = datalim[1][1] - datalim[0][1]

    ax.set_aspect('equal', 'datalim')
    ax.update_datalim(datalim)
    ax.autoscale_view()
    ax.set_axis_off()

    return (ax, coords)

def plot_structure_with_target_var(idx):
    sequence = all_data.iloc[idx].sequence
    structure = all_data.iloc[idx].structure

    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(16, 4))
    coloring = all_data.iloc[idx].reactivity
    coloring = [(c-min(all_data.reactivity[idx]))/(max(all_data.reactivity[idx])-min(all_data.reactivity[idx])) for c in coloring] 
    bg = fgb.BulgeGraph.from_fasta_text(f'>rna1\n{structure}\n{sequence}')[0]
    custom_plot_rna(bg, coloring, ax=ax[0])
    ax[0].set_title('reactivity', fontsize=16)

    coloring = all_data.iloc[idx].deg_Mg_pH10
    coloring = [(c-min(all_data.deg_Mg_pH10[idx]))/(max(all_data.deg_Mg_pH10[idx])-min(all_data.deg_Mg_pH10[idx])) for c in coloring] 
    custom_plot_rna(bg, coloring, ax=ax[1])
    ax[1].set_title('deg_Mg_pH10', fontsize=16)

    coloring = all_data.iloc[idx].deg_pH10
    coloring = [(c-min(all_data.deg_pH10[idx]))/(max(all_data.deg_pH10[idx])-min(all_data.deg_pH10[idx])) for c in coloring] 
    custom_plot_rna(bg, coloring, ax=ax[2])
    ax[2].set_title('deg_pH10', fontsize=16)

    coloring = all_data.iloc[idx].deg_Mg_50C
    coloring = [(c-min(all_data.deg_Mg_50C[idx]))/(max(all_data.deg_Mg_50C[idx])-min(all_data.deg_Mg_50C[idx])) for c in coloring] 
    custom_plot_rna(bg, coloring, ax=ax[3])
    ax[3].set_title('deg_Mg_50C', fontsize=16)

    coloring = all_data.iloc[idx].deg_50C
    coloring = [(c-min(all_data.deg_50C[idx]))/(max(all_data.deg_50C[idx])-min(all_data.deg_50C[idx])) for c in coloring] 
    custom_plot_rna(bg, coloring, ax=ax[4])
    ax[4].set_title('deg_50C', fontsize=16)

    plt.show()
    
def plot_structure_with_predicted_var(idx):
    sequence = all_data.iloc[idx].sequence
    structure = all_data.iloc[idx].structure
    try:
        data=train_dataset[idx]
    except:
        data=val_dataset[idx]
    preds = model(data.to(device)).detach().cpu().numpy()
    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(16, 4))

    coloring = preds[:,0].tolist()
    coloring = [(c-min(all_data.reactivity[idx]))/(max(all_data.reactivity[idx])-min(all_data.reactivity[idx])) for c in coloring] 
    bg = fgb.BulgeGraph.from_fasta_text(f'>rna1\n{structure}\n{sequence}')[0]
    custom_plot_rna(bg, coloring, ax=ax[0])
    ax[0].set_title('reactivity', fontsize=16)

    coloring = preds[:,1].tolist()
    coloring = [(c-min(all_data.deg_Mg_pH10[idx]))/(max(all_data.deg_Mg_pH10[idx])-min(all_data.deg_Mg_pH10[idx])) for c in coloring] 
    custom_plot_rna(bg, coloring, ax=ax[1])
    ax[1].set_title('deg_Mg_pH10', fontsize=16)

    coloring = preds[:,2].tolist()
    coloring = [(c-min(all_data.deg_pH10[idx]))/(max(all_data.deg_pH10[idx])-min(all_data.deg_pH10[idx])) for c in coloring] 
    custom_plot_rna(bg, coloring, ax=ax[2])
    ax[2].set_title('deg_pH10', fontsize=16)

    coloring = preds[:,3].tolist()
    coloring = [(c-min(all_data.deg_Mg_50C[idx]))/(max(all_data.deg_Mg_50C[idx])-min(all_data.deg_Mg_50C[idx])) for c in coloring] 
    custom_plot_rna(bg, coloring, ax=ax[3])
    ax[3].set_title('deg_Mg_50C', fontsize=16)

    coloring = preds[:,4].tolist()
    coloring = [(c-min(all_data.deg_50C[idx]))/(max(all_data.deg_50C[idx])-min(all_data.deg_50C[idx])) for c in coloring] 
    custom_plot_rna(bg, coloring, ax=ax[4])
    ax[4].set_title('deg_50C', fontsize=16)

    plt.show()


## === cell 23
idx=val_ids[0]

plot_structure_with_target_var(5)
plot_structure_with_predicted_var(5)


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2263165377.py in <cell line: 0>()
      2 idx=val_ids[0]
      3 
----> 4 plot_structure_with_target_var(5)
      5 plot_structure_with_predicted_var(5)

/tmp/ipykernel_11/1617167474.py in plot_structure_with_target_var(idx)
     48     coloring = all_data.iloc[idx].reactivity
     49     coloring = [(c-min(all_data.reactivity[idx]))/(max(all_data.reactivity[idx])-min(all_data.reactivity[idx])) for c in coloring]
---> 50     bg = fgb.BulgeGraph.from_fasta_text(f'>rna1\n{structure}\n{sequence}')[0]
     51     custom_plot_rna(bg, coloring, ax=ax[0])
     52     ax[0].set_title('reactivity', fontsize=16)

NameError: name 'fgb' is not defined

## === cell 24
import matplotlib.pyplot as plt
plt.plot(train_loss,label='train')
plt.plot(val_loss,label='val')
plt.title('Plot training and validation losses')
plt.legend()


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2582333629.py in <cell line: 0>()
      1 import matplotlib.pyplot as plt
----> 2 plt.plot(train_loss,label='train')
      3 plt.plot(val_loss,label='val')
      4 plt.title('Plot training and validation losses')
      5 plt.legend()

NameError: name 'train_loss' is not defined

## === cell 25
import gc
del train_dataset
del train_loader
del val_dataset
del val_loader
gc.collect()


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4010307493.py in <cell line: 0>()
      1 import gc
----> 2 del train_dataset
      3 del train_loader
      4 del val_dataset
      5 del val_loader

NameError: name 'train_dataset' is not defined

## === cell 26
public_leaderboard_dataset=MyOwnDataset(root='public/',train=False,public=True)
private_leaderboard_dataset=MyOwnDataset(root='private/',train=False,public=False)

public_leaderboard_loader = DataLoader(public_leaderboard_dataset, batch_size=4, shuffle=False)
private_leaderboard_loader = DataLoader(private_leaderboard_dataset, batch_size=4, shuffle=False)


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1294642573.py in <cell line: 0>()
----> 1 public_leaderboard_dataset=MyOwnDataset(root='public/',train=False,public=True)
      2 private_leaderboard_dataset=MyOwnDataset(root='private/',train=False,public=False)
      3 
      4 public_leaderboard_loader = DataLoader(public_leaderboard_dataset, batch_size=4, shuffle=False)
      5 private_leaderboard_loader = DataLoader(private_leaderboard_dataset, batch_size=4, shuffle=False)

/tmp/ipykernel_11/1803442915.py in __init__(self, root, train, public, ids, transform, pre_transform)
     26         self.target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C','deg_pH10', 'deg_50C']
     27 
---> 28         super(MyOwnDataset, self).__init__(root,transform, pre_transform)
     29         self.data, self.slices = torch.load(self.processed_paths[0])
     30 

/usr/local/lib/python3.11/dist-packages/torch_geometric/data/in_memory_dataset.py in __init__(self, root, transform, pre_transform, pre_filter, log, force_reload)
     79         force_reload: bool = False,
     80     ) -> None:
---> 81         super().__init__(root, transform, pre_transform, pre_filter, log,
     82                          force_reload)
     83 

/usr/local/lib/python3.11/dist-packages/torch_geometric/data/dataset.py in __init__(self, root, transform, pre_transform, pre_filter, log, force_reload)
    113 
    114         if self.has_process:
--> 115             self._process()
    116 
    117     def indices(self) -> Sequence:

/usr/local/lib/python3.11/dist-packages/torch_geometric/data/dataset.py in _process(self)
    263 
    264         fs.makedirs(self.processed_dir, exist_ok=True)
--> 265         self.process()
    266 
    267         path = osp.join(self.processed_dir, 'pre_transform.pt')

/tmp/ipykernel_11/1803442915.py in process(self)
     52             # nodes x features
     53             id_=self.df['id'].iloc[idx]
---> 54             bpps=np.load(self.bpps_dir+id_+'.npy')
     55             edge_index=np.stack(np.where((matrix)>0))
     56             node_attr=seq2nodes(sequence,loops,structure)

/usr/local/lib/python3.11/dist-packages/numpy/lib/npyio.py in load(file, mmap_mode, allow_pickle, fix_imports, encoding, max_header_size)
    425             own_fid = False
    426         else:
--> 427             fid = stack.enter_context(open(os_fspath(file), "rb"))
    428             own_fid = True
    429 

FileNotFoundError: [Errno 2] No such file or directory: '../input/stanford-covid-vaccine/bpps/id_00b436dec.npy'

## === cell 27
def get_preds(pred_loader,public=False):
    model.eval()
    batch_preds=[]
    for batch_idx,data in enumerate(pred_loader):
        out = model(data.to(device))
        if public:
            out=out
        else:
            out=out
        batch_preds.append(out.cpu().detach())
    return batch_preds


## === cell 28
public_preds=get_preds(public_leaderboard_loader,public=True)
private_preds=get_preds(private_leaderboard_loader,public=False)


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/994035557.py in <cell line: 0>()
----> 1 public_preds=get_preds(public_leaderboard_loader,public=True)
      2 private_preds=get_preds(private_leaderboard_loader,public=False)

NameError: name 'public_leaderboard_loader' is not defined

## === cell 29
public_preds=torch.cat(public_preds,dim=0).numpy()
private_preds=torch.cat(private_preds,dim=0).numpy()


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3504220691.py in <cell line: 0>()
----> 1 public_preds=torch.cat(public_preds,dim=0).numpy()
      2 private_preds=torch.cat(private_preds,dim=0).numpy()

NameError: name 'public_preds' is not defined

## === cell 30
all_df=pd.read_json('/kaggle/input/stanford-covid-vaccine/test.json',lines=True)
public_df = all_df.query("seq_length == 107")
private_df = all_df.query("seq_length == 130")


## === cell 31
pred_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
preds_ls = []

for df, preds in [(public_df, public_preds), (private_df, private_preds)]:
    for i, uid in enumerate(df.id):
        sequence=df.sequence.iloc[i]
        single_pred = preds[i*len(sequence):i*len(sequence)+len(sequence),:]

        single_df = pd.DataFrame(single_pred, columns=pred_cols)
        single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]

        preds_ls.append(single_df)

preds_df = pd.concat(preds_ls)
preds_df.head(10)


## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2024928008.py in <cell line: 0>()
      2 preds_ls = []
      3 
----> 4 for df, preds in [(public_df, public_preds), (private_df, private_preds)]:
      5     for i, uid in enumerate(df.id):
      6         sequence=df.sequence.iloc[i]

NameError: name 'public_preds' is not defined

## === cell 32
sample_df = pd.read_csv('../input/stanford-covid-vaccine/sample_submission.csv')
submission = sample_df[['id_seqpos']].merge(preds_df, on=['id_seqpos'])
submission.to_csv('submission.csv', index=False)


## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3260412657.py in <cell line: 0>()
      1 sample_df = pd.read_csv('../input/stanford-covid-vaccine/sample_submission.csv')
----> 2 submission = sample_df[['id_seqpos']].merge(preds_df, on=['id_seqpos'])
      3 submission.to_csv('submission.csv', index=False)

NameError: name 'preds_df' is not defined

## === cell 33
submission.head(10)


## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3300208274.py in <cell line: 0>()
----> 1 submission.head(10)

NameError: name 'submission' is not defined

## === cell 34
print(len(submission))


## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/536198159.py in <cell line: 0>()
----> 1 print(len(submission))

NameError: name 'submission' is not defined
