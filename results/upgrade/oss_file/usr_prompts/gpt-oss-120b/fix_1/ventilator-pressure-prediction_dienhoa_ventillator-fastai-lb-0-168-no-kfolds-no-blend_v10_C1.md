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

0.877

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
!pip install --upgrade fastai


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
from fastai.losses import MSELossFlat
from fastai.callback.schedule import Learner
from sklearn.preprocessing import RobustScaler, normalize

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

## === cell 5
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


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2446934803.py in <cell line: 0>()
     43     return df
     44 
---> 45 train = add_features(df)
     46 test = add_features(df_test)

NameError: name 'df' is not defined

## === cell 6
targets = train[['pressure']].to_numpy().reshape(-1, 80)
train.drop(['pressure','id', 'breath_id','one','count','breath_id_lag','breath_id_lag2','breath_id_lagsame','breath_id_lag2same','u_out_lag2'], axis=1, inplace=True)
test = test.drop(['id', 'breath_id','one','count','breath_id_lag','breath_id_lag2','breath_id_lagsame','breath_id_lag2same','u_out_lag2'], axis=1)


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3421453840.py in <cell line: 0>()
----> 1 targets = train[['pressure']].to_numpy().reshape(-1, 80)
      2 train.drop(['pressure','id', 'breath_id','one','count','breath_id_lag','breath_id_lag2','breath_id_lagsame','breath_id_lag2same','u_out_lag2'], axis=1, inplace=True)
      3 test = test.drop(['id', 'breath_id','one','count','breath_id_lag','breath_id_lag2','breath_id_lagsame','breath_id_lag2same','u_out_lag2'], axis=1)

NameError: name 'train' is not defined

## === cell 7
RS = RobustScaler()
train = RS.fit_transform(train)
test = RS.transform(test)


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1058649458.py in <cell line: 0>()
----> 1 RS = RobustScaler()
      2 train = RS.fit_transform(train)
      3 test = RS.transform(test)

NameError: name 'RobustScaler' is not defined

## === cell 8
train = train.reshape(-1, 80, train.shape[-1])
test = test.reshape(-1, 80, train.shape[-1])


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2074315604.py in <cell line: 0>()
----> 1 train = train.reshape(-1, 80, train.shape[-1])
      2 test = test.reshape(-1, 80, train.shape[-1])

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
len(train)


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/101669503.py in <cell line: 0>()
----> 1 len(train)

NameError: name 'train' is not defined

## === cell 11
train_input, valid_input = train[:int(len(train)*0.8)], train[int(len(train)*0.8):]
train_targets, valid_targets = targets[:int(len(train)*0.8)], targets[int(len(train)*0.8):]


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/911076242.py in <cell line: 0>()
----> 1 train_input, valid_input = train[:int(len(train)*0.8)], train[int(len(train)*0.8):]
      2 train_targets, valid_targets = targets[:int(len(train)*0.8)], targets[int(len(train)*0.8):]

NameError: name 'train' is not defined

## === cell 13
train.shape[-2:]


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/901771965.py in <cell line: 0>()
----> 1 train.shape[-2:]

NameError: name 'train' is not defined

## === cell 14
class VentilatorDataset(Dataset):
    def __init__(self, data, target):
        self.data = torch.from_numpy(data).float()
        if target is not None:
            self.targets = torch.from_numpy(target).float()
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if hasattr(self, 'targets'): return self.data[idx], self.targets[idx]
        else: return self.data[idx]


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3516231332.py in <cell line: 0>()
----> 1 class VentilatorDataset(Dataset):
      2     def __init__(self, data, target):
      3         self.data = torch.from_numpy(data).float()
      4         if target is not None:
      5             self.targets = torch.from_numpy(target).float()

NameError: name 'Dataset' is not defined

## === cell 15
train_dataset = VentilatorDataset(train_input, train_targets)
valid_dataset = VentilatorDataset(valid_input, valid_targets)


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3361874181.py in <cell line: 0>()
----> 1 train_dataset = VentilatorDataset(train_input, train_targets)
      2 valid_dataset = VentilatorDataset(valid_input, valid_targets)

NameError: name 'VentilatorDataset' is not defined

## === cell 16
batch_size = 8


## === cell 17
def worker_init_fn(worker_id):
    """
    Handles PyTorch x Numpy seeding issues.

    Args:
        worker_id (int): Id of the worker.
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)


## === cell 18
class RNNModel(nn.Module):
    def __init__(
        self,
        input_dim=25,
        lstm_dim=512,
        dense_dim=512,
        logit_dim=512,
        num_classes=1,
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, dense_dim // 2),
            nn.ReLU(),
            nn.Linear(dense_dim // 2, dense_dim),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(dense_dim, lstm_dim, batch_first=True, bidirectional=True)
        self.logits = nn.Sequential(
            nn.Linear(lstm_dim * 2, logit_dim),
            nn.ReLU(),
            nn.Linear(logit_dim, num_classes),
        )

    def forward(self, x):
        features = self.mlp(x)
        features, _ = self.lstm(features)
        pred = self.logits(features)
        return pred


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1608745488.py in <cell line: 0>()
----> 1 class RNNModel(nn.Module):
      2     def __init__(
      3         self,
      4         input_dim=25,
      5         lstm_dim=512,

NameError: name 'nn' is not defined

## === cell 19
model = RNNModel()


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1317953640.py in <cell line: 0>()
----> 1 model = RNNModel()

NameError: name 'RNNModel' is not defined

## === cell 20
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size = batch_size, shuffle=False)


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/592518789.py in <cell line: 0>()
----> 1 train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
      2 valid_loader = DataLoader(valid_dataset, batch_size = batch_size, shuffle=False)

NameError: name 'DataLoader' is not defined

## === cell 21
x, y = next(iter(train_loader))


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3997623068.py in <cell line: 0>()
----> 1 x, y = next(iter(train_loader))

NameError: name 'train_loader' is not defined

## === cell 22
dls = DataLoaders(train_loader, valid_loader)


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1684595834.py in <cell line: 0>()
----> 1 dls = DataLoaders(train_loader, valid_loader)

NameError: name 'DataLoaders' is not defined

## === cell 23
learn = Learner(dls, model, loss_func=MSELossFlat())


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2003238097.py in <cell line: 0>()
----> 1 learn = Learner(dls, model, loss_func=MSELossFlat())

NameError: name 'Learner' is not defined

## === cell 24
learn.lr_find()


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2515604160.py in <cell line: 0>()
----> 1 learn.lr_find()

NameError: name 'learn' is not defined

## === cell 25
learn.fit_one_cycle(2, lr_max=0.0109)


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2574599692.py in <cell line: 0>()
----> 1 learn.fit_one_cycle(2, lr_max=0.0109)

NameError: name 'learn' is not defined

## === cell 26
submission = pd.read_csv('../input/ventilator-pressure-prediction/sample_submission.csv')


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2619557148.py in <cell line: 0>()
----> 1 submission = pd.read_csv('../input/ventilator-pressure-prediction/sample_submission.csv')

NameError: name 'pd' is not defined

## === cell 27
test.shape


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/269475441.py in <cell line: 0>()
----> 1 test.shape

NameError: name 'test' is not defined

## === cell 28
test_dataset = VentilatorDataset(test, None)


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/312847402.py in <cell line: 0>()
----> 1 test_dataset = VentilatorDataset(test, None)

NameError: name 'VentilatorDataset' is not defined

## === cell 29
test_dataset[1]


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3078004527.py in <cell line: 0>()
----> 1 test_dataset[1]

NameError: name 'test_dataset' is not defined

## === cell 30
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False)


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1558102503.py in <cell line: 0>()
----> 1 test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False)

NameError: name 'DataLoader' is not defined

## === cell 31
one_batch = next(iter(test_loader))


## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/613671030.py in <cell line: 0>()
----> 1 one_batch = next(iter(test_loader))

NameError: name 'test_loader' is not defined

## === cell 32
preds = []
with torch.no_grad():
    for data in test_loader:
        pred = model(data.to('cuda')).squeeze(-1).flatten()
        preds.extend(pred.detach().cpu().numpy())



## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4196383972.py in <cell line: 0>()
      1 preds = []
----> 2 with torch.no_grad():
      3     for data in test_loader:
      4         pred = model(data.to('cuda')).squeeze(-1).flatten()
      5         preds.extend(pred.detach().cpu().numpy())

NameError: name 'torch' is not defined

## === cell 33
df_test['pressure'] = preds


## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2403236272.py in <cell line: 0>()
----> 1 df_test['pressure'] = preds

NameError: name 'df_test' is not defined

## === cell 34
df_test


## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2435861599.py in <cell line: 0>()
----> 1 df_test

NameError: name 'df_test' is not defined

## === cell 35
df_test[['id', 'pressure']].to_csv('submission.csv', index=False)


## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/827391536.py in <cell line: 0>()
----> 1 df_test[['id', 'pressure']].to_csv('submission.csv', index=False)

NameError: name 'df_test' is not defined
