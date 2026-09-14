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
import warnings

try:
    import forgi.graph.bulge_graph as fgb
    import forgi.visual.mplotlib as fvm
    import forgi.threedee.utilities.vector as ftuv
    import forgi
except ModuleNotFoundError as e:
    warnings.warn(
        f"Optional dependency not available ({e}). "
        "forgi-related features will be disabled in this environment."
    )
    fgb = None
    fvm = None
    ftuv = None
    forgi = None

try:
    import RNA
except ModuleNotFoundError as e:
    warnings.warn(
        f"Optional dependency not available ({e}). "
        "ViennaRNA (RNA) features will be disabled in this environment."
    )
    RNA = None


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
    filter_noise = True
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
id_ = all_data.iloc[idx].id
sequence = all_data.iloc[idx].sequence
structure = all_data.iloc[idx].structure
loops = all_data.iloc[idx].predicted_loop_type
reactivity = all_data.iloc[idx].reactivity

if fgb is not None and fvm is not None:
    bg = fgb.BulgeGraph.from_fasta_text(f">rna1\n{structure}\n{sequence}")[0]
    fig = plt.figure(figsize=(6, 6))
    fvm.plot_rna(bg, lighten=0.5, text_kwargs={"fontweight": None})
    plt.show()
else:
    print("forgi not available; skipping RNA visualization in this environment.")


## === cell 12
matrix = build_matrix(get_couples(structure), len(sequence))

candidate_bpps_dirs = [
    "/kaggle/input/stanford-covid-vaccine/bpps",
    "/kaggle/data/stanford-covid-vaccine/bpps",
    "../input/stanford-covid-vaccine/bpps",
    "../data/stanford-covid-vaccine/bpps",
]
bpps_dir = next((d for d in candidate_bpps_dirs if os.path.isdir(d)), None)

bpps = None
if bpps_dir is not None:
    bpps_path = os.path.join(bpps_dir, f"{id_}.npy")
    if os.path.isfile(bpps_path):
        bpps = np.load(bpps_path)

if bpps is None:
    bpps = np.zeros((len(sequence), len(sequence)), dtype=np.float32)

edge_index = np.stack(np.where((matrix + bpps) > 0))
node_attr = seq2nodes(sequence, loops, structure)
edge_attr = np.zeros((edge_index.shape[1], 3))
edge_attr[:, 0] = (matrix == 1)[edge_index[0, :], edge_index[1, :]]
edge_attr[:, 1] = (matrix == 2)[edge_index[0, :], edge_index[1, :]]
edge_attr[:, 2] = bpps[edge_index[0, :], edge_index[1, :]]


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
[0;31m---------------------------------------------------------------------------[0m
[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2421535975.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      5[0m [0mtrain_ids[0m[0;34m,[0m[0mval_ids[0m[0;34m=[0m[0mnp[0m[0;34m.[0m[0msplit[0m[0;34m([0m[0mall_ids[0m[0;34m,[0m [0;34m[[0m[0mint[0m[0;34m([0m[0mround[0m[0;34m([0m[0;36m0.9[0m [0;34m*[0m [0mlen[0m[0;34m([0m[0mall_ids[0m[0;34m)[0m[0;34m,[0m [0;36m0[0m[0;34m)[0m[0;34m)[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      6[0m [0;34m[0m[0m
[0;32m----> 7[0;31m [0mtrain_dataset[0m[0;34m=[0m[0mMyOwnDataset[0m[0;34m([0m[0mids[0m[0;34m=[0m[0mtrain_ids[0m[0;34m,[0m [0mroot[0m[0;34m=[0m[0;34m'train'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      8[0m [0mval_dataset[0m[0;34m=[0m[0mMyOwnDataset[0m[0;34m([0m[0mids[0m[0;34m=[0m[0mval_ids[0m[0;34m,[0m [0mroot[0m[0;34m=[0m[0;34m'val'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      9[0m [0;34m[0m[0m

[0;32m/tmp/ipykernel_11/1803442915.py[0m in [0;36m__init__[0;34m(self, root, train, public, ids, transform, pre_transform)[0m
[1;32m     26[0m         [0mself[0m[0;34m.[0m[0mtarget_cols[0m [0;34m=[0m [0;34m[[0m[0;34m'reactivity'[0m[0;34m,[0m [0;34m'deg_Mg_pH10'[0m[0;34m,[0m [0;34m'deg_Mg_50C'[0m[0;34m,[0m[0;34m'deg_pH10'[0m[0;34m,[0m [0;34m'deg_50C'[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m     27[0m [0;34m[0m[0m
[0;32m---> 28[0;31m         [0msuper[0m[0;34m([0m[0mMyOwnDataset[0m[0;34m,[0m [0mself[0m[0;34m)[0m[0;34m.[0m[0m__init__[0m[0;34m([0m[0mroot[0m[0;34m,[0m[0mtransform[0m[0;34m,[0m [0mpre_transform[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     29[0m         [0mself[0m[0;34m.[0m[0mdata[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0mslices[0m [0;34m=[0m [0mtorch[0m[0;34m.[0m[0mload[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mprocessed_paths[0m[0;34m[[0m[0;36m0[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     30[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch_geometric/data/in_memory_dataset.py[0m in [0;36m__init__[0;34m(self, root, transform, pre_transform, pre_filter, log, force_reload)[0m
[1;32m     79[0m         [0mforce_reload[0m[0;34m:[0m [0mbool[0m [0;34m=[0m [0;32mFalse[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     80[0m     ) -> None:
[0;32m---> 81[0;31m         super().__init__(root, transform, pre_transform, pre_filter, log,
[0m[1;32m     82[0m                          force_reload)
[1;32m     83[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch_geometric/data/dataset.py[0m in [0;36m__init__[0;34m(self, root, transform, pre_transform, pre_filter, log, force_reload)[0m
[1;32m    113[0m [0;34m[0m[0m
[1;32m    114[0m         [0;32mif[0m [0mself[0m[0;34m.[0m[0mhas_process[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 115[0;31m             [0mself[0m[0;34m.[0m[0m_process[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    116[0m [0;34m[0m[0m
[1;32m    117[0m     [0;32mdef[0m [0mindices[0m[0;34m([0m[0mself[0m[0;34m)[0m [0;34m->[0m [0mSequence[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch_geometric/data/dataset.py[0m in [0;36m_process[0;34m(self)[0m
[1;32m    263[0m [0;34m[0m[0m
[1;32m    264[0m         [0mfs[0m[0;34m.[0m[0mmakedirs[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mprocessed_dir[0m[0;34m,[0m [0mexist_ok[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 265[0;31m         [0mself[0m[0;34m.[0m[0mprocess[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    266[0m [0;34m[0m[0m
[1;32m    267[0m         [0mpath[0m [0;34m=[0m [0mosp[0m[0;34m.[0m[0mjoin[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mprocessed_dir[0m[0;34m,[0m [0;34m'pre_transform.pt'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/1803442915.py[0m in [0;36mprocess[0;34m(self)[0m
[1;32m     52[0m             [0;31m# nodes x features[0m[0;34m[0m[0;34m[0m[0m
[1;32m     53[0m             [0mid_[0m[0;34m=[0m[0mself[0m[0;34m.[0m[0mdf[0m[0;34m[[0m[0;34m'id'[0m[0;34m][0m[0;34m.[0m[0miloc[0m[0;34m[[0m[0midx[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 54[0;31m             [0mbpps[0m[0;34m=[0m[0mnp[0m[0;34m.[0m[0mload[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mbpps_dir[0m[0;34m+[0m[0mid_[0m[0;34m+[0m[0;34m'.npy'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     55[0m             [0medge_index[0m[0;34m=[0m[0mnp[0m[0;34m.[0m[0mstack[0m[0;34m([0m[0mnp[0m[0;34m.[0m[0mwhere[0m[0;34m([0m[0;34m([0m[0mmatrix[0m[0;34m)[0m[0;34m>[0m[0;36m0[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     56[0m             [0mnode_attr[0m[0;34m=[0m[0mseq2nodes[0m[0;34m([0m[0msequence[0m[0;34m,[0m[0mloops[0m[0;34m,[0m[0mstructure[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/numpy/lib/npyio.py[0m in [0;36mload[0;34m(file, mmap_mode, allow_pickle, fix_imports, encoding, max_header_size)[0m
[1;32m    425[0m             [0mown_fid[0m [0;34m=[0m [0;32mFalse[0m[0;34m[0m[0;34m[0m[0m
[1;32m    426[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 427[0;31m             [0mfid[0m [0;34m=[0m [0mstack[0m[0;34m.[0m[0menter_context[0m[0;34m([0m[0mopen[0m[0;34m([0m[0mos_fspath[0m[0;34m([0m[0mfile[0m[0;34m)[0m[0;34m,[0m [0;34m"rb"[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    428[0m             [0mown_fid[0m [0;34m=[0m [0;32mTrue[0m[0;34m[0m[0;34m[0m[0m
[1;32m    429[0m [0;34m[0m[0m

[0;31mFileNotFoundError[0m: [Errno 2] No such file or directory: '../input/stanford-covid-vaccine/bpps/id_001f94081.npy'

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
        nn = Sequential(Linear(edge_feats, 5), ReLU(), Linear(5, channels * channels))
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
