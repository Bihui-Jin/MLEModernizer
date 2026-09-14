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
sentence-transformers==4.1.0
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
transformers==4.53.3
wandb==0.21.0

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

0.1580959161450108

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 2
import gc
import os
import random
import wandb
import math

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AdamW
from transformers import get_cosine_schedule_with_warmup
from sklearn.preprocessing import RobustScaler

device = torch.device("cuda")


## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/4232217695.py in <cell line: 0>()
     16 from torch.utils.data import Dataset, DataLoader
     17 
---> 18 from transformers import AdamW
     19 from transformers import get_cosine_schedule_with_warmup
     20 from sklearn.preprocessing import RobustScaler

ImportError: cannot import name 'AdamW' from 'transformers' (/usr/local/lib/python3.11/dist-packages/transformers/__init__.py)

## === cell 3
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

## === cell 4
class VentilatorDataset(Dataset):
    
    def __init__(self, df, label_dic=None):
        self.dfs = [_df for _, _df in df.groupby("breath_id")]
        self.label_dic = label_dic
        
    def __len__(self):
        return len(self.dfs)
    
    def __getitem__(self, item):
        df = self.dfs[item]
        X = df[config.ALL_FEATURES].values
        y = df['pressure'].values
        if self.label_dic is None:
            label = [-1]
        else:
            label = [self.label_dic[i] for i in y]

        d = {
            "X": torch.tensor(X).float(),
            "y" : torch.tensor(label).long(),
        }
        return d

## === cell 5
class VentilatorModel(nn.Module):
    
    def __init__(self):
        super(VentilatorModel, self).__init__()
        self.seq_emb = nn.Sequential(
            nn.Linear(len(config.CONT_FEATURES)+len(config.LAG_FEATURES), config.EMBED_SIZE),
            nn.LayerNorm(config.EMBED_SIZE),
        )
        
        self.lstm = nn.LSTM(config.EMBED_SIZE, config.HIDDEN_SIZE, batch_first=True, bidirectional=True, dropout=0.0, num_layers=4)

        self.head = nn.Sequential(
            nn.Linear(config.HIDDEN_SIZE * 2, config.HIDDEN_SIZE * 2),
            nn.LayerNorm(config.HIDDEN_SIZE * 2),
            nn.ReLU(),
            nn.Linear(config.HIDDEN_SIZE * 2, 950),
        )
        
        
        for n, m in self.named_modules():
            if isinstance(m, nn.LSTM):
                print(f'init {m}')
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        nn.init.orthogonal_(param.data)
                    else:
                        nn.init.normal_(param.data)

    def forward(self, X, y=None):
        
        seq_x = X
        emb_x = self.seq_emb(seq_x)
        
        out, _ = self.lstm(emb_x, None) 
        logits = self.head(out)

        if y is None:
            loss = None
        else:
            loss = self.loss_fn(logits, y)
            
        return logits, loss
    
    def loss_fn(self, y_pred, y_true):
        loss = nn.CrossEntropyLoss()(y_pred.reshape(-1, 950), y_true.reshape(-1))
        return loss
    
    
model = VentilatorModel()

## === cell 6
def test_loop(model, loader, target_dic_inv):
    predicts = []
    model.eval()
    for d in loader:
        with torch.no_grad():
            out, _ = model(d['X'].to(device))
        out = torch.tensor([[target_dic_inv[j.item()] for j in i] for i in out.argmax(2)])
        predicts.append(out.cpu())

    return torch.vstack(predicts).numpy().reshape(-1)

## === cell 7
def add_feature(df):
    df['time_delta'] = df.groupby('breath_id')['time_step'].diff().fillna(0)
    df['delta'] = df['time_delta'] * df['u_in']
    df['area'] = df.groupby('breath_id')['delta'].cumsum()

    df['cross']= df['u_in']*df['u_out']
    df['cross2']= df['time_step']*df['u_out']
    
    df['u_in_cumsum'] = (df['u_in']).groupby(df['breath_id']).cumsum()
    df['one'] = 1
    df['count'] = (df['one']).groupby(df['breath_id']).cumsum()
    df['u_in_cummean'] =df['u_in_cumsum'] / df['count']
    
    df = df.drop(['count','one'], axis=1)
    return df

def add_lag_feature(df):
    for lag in range(1, config.USE_LAG+1):
        df[f'breath_id_lag{lag}']=df['breath_id'].shift(lag).fillna(0)
        df[f'breath_id_lag{lag}same']=np.select([df[f'breath_id_lag{lag}']==df['breath_id']], [1], 0)

        df[f'u_in_lag_{lag}'] = df['u_in'].shift(lag).fillna(0) * df[f'breath_id_lag{lag}same']
        df[f'u_in_time{lag}'] = df['u_in'] - df[f'u_in_lag_{lag}']
        df[f'u_out_lag_{lag}'] = df['u_out'].shift(lag).fillna(0) * df[f'breath_id_lag{lag}same']

    df['time_step_lag'] = df['time_step'].shift(1).fillna(0) * df[f'breath_id_lag{lag}same']
    df['breath_time'] = df['time_step'] - df['time_step_lag']

    drop_columns = ['time_step_lag']
    drop_columns += [f'breath_id_lag{i}' for i in range(1, config.USE_LAG+1)]
    drop_columns += [f'breath_id_lag{i}same' for i in range(1, config.USE_LAG+1)]
    df = df.drop(drop_columns, axis=1)

    df = df.fillna(0)
    return df

c_dic = {10: 0, 20: 1, 50:2}
r_dic = {5: 0, 20: 1, 50:2}
rc_sum_dic = {v: i for i, v in enumerate([15, 25, 30, 40, 55, 60, 70, 100])}
rc_dot_dic = {v: i for i, v in enumerate([50, 100, 200, 250, 400, 500, 2500, 1000])}    

def add_category_features(df):
    df['C_cate'] = df['C'].map(c_dic)
    df['R_cate'] = df['R'].map(r_dic)
    df['RC_sum'] = (df['R'] + df['C']).map(rc_sum_dic)
    df['RC_dot'] = (df['R'] * df['C']).map(rc_dot_dic)
    return df

norm_features = config.CONT_FEATURES + config.LAG_FEATURES
def norm_scale(train_df, test_df):
    scaler = RobustScaler()
    all_u_in = np.vstack([train_df[norm_features].values, test_df[norm_features].values])
    scaler.fit(all_u_in)
    train_df[norm_features] = scaler.transform(train_df[norm_features].values)
    test_df[norm_features] = scaler.transform(test_df[norm_features].values)
    return train_df, test_df

## === cell 8
train_df = pd.read_csv(f"{config.INPUT}/train.csv")
test_df = pd.read_csv(f"{config.INPUT}/test.csv")
sub_df = pd.read_csv(f"{config.INPUT}/sample_submission.csv")
oof = np.zeros(len(train_df))
test_preds_lst = []

target_dic = {v:i for i, v in enumerate(sorted(train_df['pressure'].unique().tolist()))}
target_dic_inv = {v: k for k, v in target_dic.items()}

gkf = GroupKFold(n_splits=config.N_FOLD).split(train_df, train_df.pressure, groups=train_df.breath_id)
for fold, (_, valid_idx) in enumerate(gkf):
    train_df.loc[valid_idx, 'fold'] = fold

train_df = add_feature(train_df)
test_df = add_feature(test_df)
train_df = add_lag_feature(train_df)
test_df = add_lag_feature(test_df)
train_df = add_category_features(train_df)
test_df = add_category_features(test_df)
train_df, test_df = norm_scale(train_df, test_df)

test_df['pressure'] = -1
test_dset = VentilatorDataset(test_df)
test_loader = DataLoader(test_dset, batch_size=config.BS,
                         pin_memory=True, shuffle=False, drop_last=False, num_workers=os.cpu_count())



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/95914824.py in <cell line: 0>()
     18 train_df = add_category_features(train_df)
     19 test_df = add_category_features(test_df)
---> 20 train_df, test_df = norm_scale(train_df, test_df)
     21 
     22 test_df['pressure'] = -1

/tmp/ipykernel_11/2827651790.py in norm_scale(train_df, test_df)
     56 norm_features = config.CONT_FEATURES + config.LAG_FEATURES
     57 def norm_scale(train_df, test_df):
---> 58     scaler = RobustScaler()
     59     all_u_in = np.vstack([train_df[norm_features].values, test_df[norm_features].values])
     60     scaler.fit(all_u_in)

NameError: name 'RobustScaler' is not defined

## === cell 9
unique_pressures = train_df["pressure"].unique()
sorted_pressures = np.sort(unique_pressures)
total_pressures_len = len(sorted_pressures)

## === cell 10

def find_nearest(prediction):
    insert_idx = np.searchsorted(sorted_pressures, prediction)
    if insert_idx == total_pressures_len:
        return sorted_pressures[-1]
    elif insert_idx == 0:
        return sorted_pressures[0]
    lower_val = sorted_pressures[insert_idx - 1]
    upper_val = sorted_pressures[insert_idx]
    return lower_val if abs(lower_val - prediction) < abs(upper_val - prediction) else upper_val


## === cell 13
from glob import glob 

models = []
for model_path in glob('/kaggle/input/ventilator-train-classification/exp080_conti_rc/ventilator_f*_best_model.bin'):
    model = VentilatorModel()
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    models.append(model)

## === cell 14
classes, class_freq = np.unique(train_df['pressure'].map(target_dic), return_counts=True)

## === cell 15
class_proba = class_freq / np.sum(class_freq)

## === cell 17
def test_loop_pred(models, loader, target_dic_inv, class_proba=None):
    predicts = []
    for model in models:
        model.eval()
    if class_proba is not None:
        class_proba = torch.unsqueeze(torch.unsqueeze(torch.log(class_proba), dim=0), dim=0) # (1, 1, 950)
    for d in tqdm(loader):
        outs = []
        with torch.no_grad():
            for model in models:
                out, _ = model(d['X'].to(device))  
                out = torch.log(torch.nn.functional.softmax(out, dim=2))   # B, S, 950                
                outs.append(out)
        if class_proba is not None:
            outs.append(class_proba.expand(outs[-1].shape))            
        out = torch.stack(outs, dim=2)  # B, S, n_folds + 1, 950            
        out = torch.sum(out, dim=2)
        out = torch.tensor([[target_dic_inv[j.item()] for j in i] for i in out.argmax(2)])
        predicts.append(out.cpu())

    return torch.vstack(predicts).numpy().reshape(-1)



## === cell 18
df_submission = pd.read_csv('/kaggle/input/ventilator-pressure-prediction/sample_submission.csv')
df_submission['pressure'] = test_loop_pred(models, test_loader, target_dic_inv)
df_submission['pressure'] = df_submission['pressure'].apply(find_nearest)
df_submission.to_csv('submission.csv', index=False)

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/677450093.py in <cell line: 0>()
      1 df_submission = pd.read_csv('/kaggle/input/ventilator-pressure-prediction/sample_submission.csv')
----> 2 df_submission['pressure'] = test_loop_pred(models, test_loader, target_dic_inv)
      3 df_submission['pressure'] = df_submission['pressure'].apply(find_nearest)
      4 df_submission.to_csv('submission.csv', index=False)

NameError: name 'test_loader' is not defined
