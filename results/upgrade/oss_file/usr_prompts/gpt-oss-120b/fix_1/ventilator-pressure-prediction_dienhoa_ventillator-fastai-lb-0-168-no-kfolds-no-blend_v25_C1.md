# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Given time series of breaths, predict the airway pressure in the respiratory circuit during the breath, given the time series of control inputs.

The best submissions will take lung attributes compliance and resistance into account.

## Metric
Mean absolute error between the predicted and actual pressures during the inspiratory phase of each breath. The expiratory phase is not scored.

## Submission Format
For each `id` in the test set, you must predict a value for the `pressure` variable. The file should contain a header and have the following format:

```
id,pressure
1,20
2,23
3,24
etc.
```

## Dataset
The ventilator data used in this competition was produced using a modified [open-source ventilator](https://pvp.readthedocs.io/) connected to an [artificial bellows test lung](https://www.ingmarmed.com/product/quicklung/) via a respiratory circuit. The diagram below illustrates the setup, with the two control inputs highlighted in green and the state variable (airway pressure) to predict in blue. The first control input is a continuous variable from 0 to 100 representing the percentage the inspiratory solenoid valve is open to let air into the lung (i.e., 0 is completely closed and no air is let in and 100 is completely open). The second control input is a binary variable representing whether the exploratory valve is open (1) or closed (0) to let air out.

![Ventilator diagram](https://raw.githubusercontent.com/google/deluca-lung/main/assets/2020-10-02%20Ventilator%20diagram.svg)

Each time series represents an approximately 3-second breath. The files are organized such that each row is a time step in a breath and gives the two control signals, the resulting airway pressure, and relevant attributes of the lung, described below.

### Files
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

### Columns
- `id` - globally-unique time step identifier across an entire file
- `breath_id` - globally-unique time step for breaths
- `R` - lung attribute indicating how restricted the airway is (in cmH2O/L/S). Physically, this is the change in pressure per change in flow (air volume per time). Intuitively, one can imagine blowing up a balloon through a straw. We can change `R` by changing the diameter of the straw, with higher `R` being harder to blow.
- `C` - lung attribute indicating how compliant the lung is (in mL/cmH2O). Physically, this is the change in volume per change in pressure. Intuitively, one can imagine the same balloon example. We can change `C` by changing the thickness of the balloon’s latex, with higher `C` having thinner latex and easier to blow.
- `time_step` - the actual time stamp.
- `u_in` - the control input for the inspiratory solenoid valve. Ranges from 0 to 100.
- `u_out` - the control input for the exploratory solenoid valve. Either 0 or 1.
- `pressure` - the airway pressure measured in the respiratory circuit, measured in cmH2O.

# 2. Python version

3.10

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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (82 lines)
            sample_submission.csv (603601 lines)
            sample_submission.csv.zip (1.3 MB)
            test.csv (603601 lines)
            test.csv.zip (8.5 MB)
            train.csv (5432401 lines)
            train.csv.zip (93.8 MB)
            ventilator-pressure-prediction/
                description.md (82 lines)
                sample_submission.csv (603601 lines)
                ... and 5 other files
                ventilator-pressure-prediction/
        input/
            description.md (82 lines)
            sample_submission.csv (603601 lines)
            sample_submission.csv.zip (1.3 MB)
            test.csv (603601 lines)
            test.csv.zip (8.5 MB)
            train.csv (5432401 lines)
            train.csv.zip (93.8 MB)
            ventilator-pressure-prediction/
                description.md (82 lines)
                sample_submission.csv (603601 lines)
                ... and 5 other files
                ventilator-pressure-prediction/
        working/
            ventilator-pressure-prediction/
                description.md (82 lines)
                sample_submission.csv (603601 lines)
                ... and 5 other files
                ventilator-pressure-prediction/
```

-> data/sample_submission.csv has 603600 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (int64) has 1 unique values: [0]

-> data/test.csv has 603600 rows and 7 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [20, 50, 10]
R (int64) has 3 unique values: [50, 5, 20]
breath_id (int64) has range: 2436.00 - 124535.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> data/train.csv has 5432400 rows and 8 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [10, 20, 50]
R (int64) has 3 unique values: [5, 50, 20]
breath_id (int64) has range: 1194.00 - 124492.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (float64) has range: 3.52 - 41.48, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> data/ventilator-pressure-prediction/sample_submission.csv has 603600 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (int64) has 1 unique values: [0]

-> data/ventilator-pressure-prediction/test.csv has 603600 rows and 7 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [20, 50, 10]
R (int64) has 3 unique values: [50, 5, 20]
breath_id (int64) has range: 2436.00 - 124535.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> data/ventilator-pressure-prediction/train.csv has 5432400 rows and 8 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [10, 20, 50]
R (int64) has 3 unique values: [5, 50, 20]
breath_id (int64) has range: 1194.00 - 124492.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (float64) has range: 3.52 - 41.48, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> input/sample_submission.csv has 603600 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (int64) has 1 unique values: [0]

-> (stopped after 10 files for performance)

# 5. Target score

0.6766

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
!pip install -Uqq fastai


## === cell 1
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from fastai.data.core import DataLoaders
from fastai.learner import Learner
from fastai.callback.progress import ProgressCallback
from fastai.optimizer import OptimWrapper
from torch import optim
from fastai.losses import MSELossFlat, L1LossFlat, CrossEntropyLossFlat
from fastai.callback.schedule import Learner
from fastai.callback.tracker import EarlyStoppingCallback, ReduceLROnPlateau
from fastai.data.transforms import IndexSplitter
from fastai.metrics import accuracy
from sklearn.preprocessing import RobustScaler, normalize
from sklearn.model_selection import KFold
import random
import gc

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
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

AttributeError: module 'numpy._globals' has no attribute '_signature_descriptor'

## === cell 2
df = pd.read_csv('/kaggle/input/ventilator-pressure-prediction/train.csv')
df_test = pd.read_csv('../input/ventilator-pressure-prediction/test.csv')


## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2539638951.py in <cell line: 0>()
----> 1 df = pd.read_csv('/kaggle/input/ventilator-pressure-prediction/train.csv')
      2 df_test = pd.read_csv('../input/ventilator-pressure-prediction/test.csv')

NameError: name 'pd' is not defined

## === cell 3
target_dic = {v:i for i, v in enumerate(sorted(df['pressure'].unique().tolist()))}
target_dic_inv = {v: k for k, v in target_dic.items()}


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2500128432.py in <cell line: 0>()
----> 1 target_dic = {v:i for i, v in enumerate(sorted(df['pressure'].unique().tolist()))}
      2 target_dic_inv = {v: k for k, v in target_dic.items()}

NameError: name 'df' is not defined

## === cell 6
def add_features(df):
    df['area'] = df['time_step'] * df['u_in']
    df['area'] = df.groupby('breath_id')['area'].cumsum()
    df['cross']= df['u_in']*df['u_out']
    df['cross2']= df['time_step']*df['u_out']
    
    df['u_in_cumsum'] = (df['u_in']).groupby(df['breath_id']).cumsum()
    df['one'] = 1
    df['count'] = (df['one']).groupby(df['breath_id']).cumsum()
    df['u_in_cummean'] =df['u_in_cumsum'] /df['count']
    df['breath_id_lag']=df['breath_id'].shift(1).fillna(0)
    df['breath_id_lag2']=df['breath_id'].shift(2).fillna(0)
    df['breath_id_lagsame']=np.select([df['breath_id_lag']==df['breath_id']],[1],0)
    df['breath_id_lag2same']=np.select([df['breath_id_lag2']==df['breath_id']],[1],0)
    df['u_in_lag'] = df['u_in'].shift(1).fillna(0)
    df['u_in_lag'] = df['u_in_lag']*df['breath_id_lagsame']
    df['u_in_lag2'] = df['u_in'].shift(2).fillna(0)
    df['u_in_lag2'] = df['u_in_lag2']*df['breath_id_lag2same']
    df['u_out_lag2'] = df['u_out'].shift(2).fillna(0)
    df['u_out_lag2'] = df['u_out_lag2']*df['breath_id_lag2same']
    
    df['R'] = df['R'].astype(str)
    df['C'] = df['C'].astype(str)
    df['RC'] = df['R']+df['C']
    df = pd.get_dummies(df)
    return df



train = add_features(df)
test = add_features(df_test)
del df
gc.collect()


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3357446740.py in <cell line: 0>()
     29 
     30 
---> 31 train = add_features(df)
     32 test = add_features(df_test)
     33 del df

NameError: name 'df' is not defined

## === cell 7
train.pressure = train.pressure.map(target_dic)


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3958826123.py in <cell line: 0>()
----> 1 train.pressure = train.pressure.map(target_dic)

NameError: name 'train' is not defined

## === cell 8
train.pressure


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3242538751.py in <cell line: 0>()
----> 1 train.pressure

NameError: name 'train' is not defined

## === cell 9
train.shape


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1612428091.py in <cell line: 0>()
----> 1 train.shape

NameError: name 'train' is not defined

## === cell 10
targets = train[['pressure']].to_numpy().reshape(-1, 80)
train.drop(['pressure','id', 'breath_id','one','count','breath_id_lag','breath_id_lag2','breath_id_lagsame','breath_id_lag2same','u_out_lag2'], axis=1, inplace=True)
test = test.drop(['id', 'breath_id','one','count','breath_id_lag','breath_id_lag2','breath_id_lagsame','breath_id_lag2same','u_out_lag2'], axis=1)


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3421453840.py in <cell line: 0>()
----> 1 targets = train[['pressure']].to_numpy().reshape(-1, 80)
      2 train.drop(['pressure','id', 'breath_id','one','count','breath_id_lag','breath_id_lag2','breath_id_lagsame','breath_id_lag2same','u_out_lag2'], axis=1, inplace=True)
      3 test = test.drop(['id', 'breath_id','one','count','breath_id_lag','breath_id_lag2','breath_id_lagsame','breath_id_lag2same','u_out_lag2'], axis=1)

NameError: name 'train' is not defined

## === cell 11
train


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3436609520.py in <cell line: 0>()
----> 1 train

NameError: name 'train' is not defined

## === cell 12
RS = RobustScaler()
train = RS.fit_transform(train)
test = RS.transform(test)


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1058649458.py in <cell line: 0>()
----> 1 RS = RobustScaler()
      2 train = RS.fit_transform(train)
      3 test = RS.transform(test)

NameError: name 'RobustScaler' is not defined

## === cell 13
train = train.reshape(-1, 80, train.shape[-1])
test = test.reshape(-1, 80, train.shape[-1])


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2074315604.py in <cell line: 0>()
----> 1 train = train.reshape(-1, 80, train.shape[-1])
      2 test = test.reshape(-1, 80, train.shape[-1])

NameError: name 'train' is not defined

## === cell 14
idx = list(range(len(train)))


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2044617495.py in <cell line: 0>()
----> 1 idx = list(range(len(train)))

NameError: name 'train' is not defined

## === cell 16
train.shape[-2:]


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/901771965.py in <cell line: 0>()
----> 1 train.shape[-2:]

NameError: name 'train' is not defined

## === cell 17
class VentilatorDataset(Dataset):
    def __init__(self, data, target, label_dic=None):
        self.data = torch.from_numpy(data).float()
        self.label_dic = label_dic
        if target is not None:
            self.targets = torch.from_numpy(target).long()
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if hasattr(self, 'targets'): 
            x = self.data[idx]
            y = self.targets[idx]
            return x, y
        else: return self.data[idx]


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3363039736.py in <cell line: 0>()
----> 1 class VentilatorDataset(Dataset):
      2     def __init__(self, data, target, label_dic=None):
      3         self.data = torch.from_numpy(data).float()
      4         self.label_dic = label_dic
      5         if target is not None:

NameError: name 'Dataset' is not defined

## === cell 18
class config:
    EXP_NAME = "exp080_conti_rc"
    
    INPUT = "/kaggle/input/ventilator-pressure-prediction"
    OUTPUT = "/kaggle/working"
    N_FOLD = 5
    SEED = 0
    
    LR = 5e-3
    N_EPOCHS = 50
    EMBED_SIZE = 64
    HIDDEN_SIZE = 256
    BS = 512
    WEIGHT_DECAY = 1e-3

    USE_LAG = 4
    CONT_FEATURES = ['u_in', 'u_out', 'time_step'] + ['u_in_cumsum', 'u_in_cummean', 'area', 'cross', 'cross2'] + ['R_cate', 'C_cate']
    LAG_FEATURES = ['breath_time']
    LAG_FEATURES += [f'u_in_lag_{i}' for i in range(1, USE_LAG+1)]
    LAG_FEATURES += [f'u_in_time{i}' for i in range(1, USE_LAG+1)]
    LAG_FEATURES += [f'u_out_lag_{i}' for i in range(1, USE_LAG+1)]
    ALL_FEATURES = CONT_FEATURES + LAG_FEATURES
    
    NOT_WATCH_PARAM = ['INPUT']


## === cell 19
class VentilatorModel(nn.Module):
    def __init__(self, input_size=25):
        hidden = [400, 300, 200, 100]
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden[0],
                             batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(2 * hidden[0], hidden[1],
                             batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM(2 * hidden[1], hidden[2],
                             batch_first=True, bidirectional=True)
        self.lstm4 = nn.LSTM(2 * hidden[2], hidden[3],
                             batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(2 * hidden[3], 4 * hidden[3])
        self.selu = nn.SELU()
        self.fc2 = nn.Linear( 4 * hidden[3], 950)
        self._reinitialize()

    def _reinitialize(self):
        """
        Tensorflow/Keras-like initialization
        """
        for name, p in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(p.data)
                elif 'bias_ih' in name:
                    p.data.fill_(0)
                    n = p.size(0)
                    p.data[(n // 4):(n // 2)].fill_(1)
                elif 'bias_hh' in name:
                    p.data.fill_(0)
            elif 'fc' in name:
                if 'weight' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'bias' in name:
                    p.data.fill_(0)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x, _ = self.lstm4(x)
        x = self.fc1(x)
        x = self.selu(x)
        x = self.fc2(x)

        return x


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2735974318.py in <cell line: 0>()
----> 1 class VentilatorModel(nn.Module):
      2     def __init__(self, input_size=25):
      3         hidden = [400, 300, 200, 100]
      4         super().__init__()
      5         self.lstm1 = nn.LSTM(input_size, hidden[0],

NameError: name 'nn' is not defined

## === cell 21
batch_size = 512
submission = pd.read_csv('../input/ventilator-pressure-prediction/sample_submission.csv')
test_dataset = VentilatorDataset(test, None)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False)


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3878544985.py in <cell line: 0>()
      1 batch_size = 512
----> 2 submission = pd.read_csv('../input/ventilator-pressure-prediction/sample_submission.csv')
      3 test_dataset = VentilatorDataset(test, None)
      4 test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False)

NameError: name 'pd' is not defined

## === cell 23
train_index=list(range(int(0.95*len(train)))) ## Change to have reasonable train/valid dataset
valid_index=list(range(int(0.95*len(train)), len(train)))


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2197867989.py in <cell line: 0>()
----> 1 train_index=list(range(int(0.95*len(train)))) ## Change to have reasonable train/valid dataset
      2 valid_index=list(range(int(0.95*len(train)), len(train)))

NameError: name 'train' is not defined

## === cell 24
train_input, valid_input = train[train_index], train[valid_index]
train_targets, valid_targets = targets[train_index], targets[valid_index]

train_dataset = VentilatorDataset(train_input, train_targets)
valid_dataset = VentilatorDataset(valid_input, valid_targets)


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3906699560.py in <cell line: 0>()
----> 1 train_input, valid_input = train[train_index], train[valid_index]
      2 train_targets, valid_targets = targets[train_index], targets[valid_index]
      3 
      4 train_dataset = VentilatorDataset(train_input, train_targets)
      5 valid_dataset = VentilatorDataset(valid_input, valid_targets)

NameError: name 'train' is not defined

## === cell 25
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size = batch_size, shuffle=False)
dls = DataLoaders(train_loader, valid_loader)


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2073227999.py in <cell line: 0>()
----> 1 train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
      2 valid_loader = DataLoader(valid_dataset, batch_size = batch_size, shuffle=False)
      3 dls = DataLoaders(train_loader, valid_loader)

NameError: name 'DataLoader' is not defined

## === cell 26
model = VentilatorModel()


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4126008178.py in <cell line: 0>()
----> 1 model = VentilatorModel()

NameError: name 'VentilatorModel' is not defined

## === cell 27
def mae_loss_discrete(y_pred, y):
    y_clone = y.clone().detach().to(dtype=torch.double)
    y_pred_clone = y_pred.clone().detach().to(dtype=torch.double)
    y_clone = y_clone.to('cpu')
    y_pred_clone = y_pred_clone.to('cpu')
    y_clone.cpu().apply_(lambda element: target_dic_inv[element])
    y_pred_clone = torch.max(y_pred_clone, axis=2)[1].to(dtype=torch.float)
    y_pred_clone.apply_(lambda element: target_dic_inv[element])
    return nn.L1Loss()(y_pred_clone, y_clone)


## === cell 28
learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), metrics=[accuracy, mae_loss_discrete])


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/595173623.py in <cell line: 0>()
----> 1 learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), metrics=[accuracy, mae_loss_discrete])

NameError: name 'Learner' is not defined

## === cell 29
learn.lr_find()


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2515604160.py in <cell line: 0>()
----> 1 learn.lr_find()

NameError: name 'learn' is not defined

## === cell 30
learn.fit_one_cycle(10, lr_max=1e-3, cbs=ReduceLROnPlateau(monitor='valid_loss', min_delta=0.5, patience=10))


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/200930985.py in <cell line: 0>()
----> 1 learn.fit_one_cycle(10, lr_max=1e-3, cbs=ReduceLROnPlateau(monitor='valid_loss', min_delta=0.5, patience=10))

NameError: name 'learn' is not defined

## === cell 31
del train_dataset, valid_dataset, train_loader, valid_loader, train_index, valid_index, train_input, valid_input, train_targets, valid_targets, test, dls, learn, train


## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1003322877.py in <cell line: 0>()
----> 1 del train_dataset, valid_dataset, train_loader, valid_loader, train_index, valid_index, train_input, valid_input, train_targets, valid_targets, test, dls, learn, train

NameError: name 'train_dataset' is not defined

## === cell 32
gc.collect()


## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/774317657.py in <cell line: 0>()
----> 1 gc.collect()

NameError: name 'gc' is not defined

## === cell 33
def get_pred(data):
    pred = model(data.to('cuda'))
    
    pred = torch.max(pred, axis=2)[1].to(dtype=torch.float).to('cpu')
    pred.apply_(lambda element: target_dic_inv[element])
    
    pred = pred.squeeze(-1).flatten().detach().numpy()
    return pred


## === cell 34
with torch.no_grad():
    preds = [get_pred(data) for data in test_loader]
preds = np.concatenate(preds)


## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3193946450.py in <cell line: 0>()
----> 1 with torch.no_grad():
      2     preds = [get_pred(data) for data in test_loader]
      3 preds = np.concatenate(preds)

NameError: name 'torch' is not defined

## === cell 35
del model, test_loader
gc.collect()


## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4266477484.py in <cell line: 0>()
----> 1 del model, test_loader
      2 gc.collect()

NameError: name 'model' is not defined

## === cell 36
df_test['pressure'] = preds
df_test[['id', 'pressure']].to_csv('submission.csv', index=False)


## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1097825756.py in <cell line: 0>()
----> 1 df_test['pressure'] = preds
      2 df_test[['id', 'pressure']].to_csv('submission.csv', index=False)

NameError: name 'preds' is not defined
