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
import os
import time
import gc
import random
from tqdm._tqdm_notebook import tqdm_notebook as tqdm

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import BatchSampler, SequentialSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


## === cell 1
def seed_everything(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()


## === cell 2
train = pd.read_json('/kaggle/input/stanford-covid-vaccine/train.json', lines=True)
test = pd.read_json('/kaggle/input/stanford-covid-vaccine/test.json', lines=True)
sample_df = pd.read_csv('/kaggle/input/stanford-covid-vaccine/sample_submission.csv')


## === cell 3
cols = ['sequence', 'structure', 'predicted_loop_type']
pred_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')}

def preprocess_inputs(data):
    '''
    Credits goes to @xhlulu: https://www.kaggle.com/xhlulu/openvaccine-simple-gru-model
    '''
    return np.transpose(
        np.array(
            data[cols]
            .applymap(lambda seq: [token2int[x] for x in seq])
            .values
            .tolist()
        ),
        (0, 2, 1)
    )


## === cell 4
class OpenVaccineDataset(Dataset):
    def __init__(self, data, labels):
        super(OpenVaccineDataset, self).__init__()
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'x': torch.tensor(self.data[idx]),
            'y': torch.tensor(self.labels[idx])
        }


## === cell 5
max_features = None
max_features = max_features or len(token2int)
pred_len = 68
EMBEDDING_DIM = 100
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS
NUM_TARGETS = 5
LR = 1e-3
BATCH_SIZE = 32
EPOCHS = 150


## === cell 6
class SpatialDropout(nn.Dropout2d):
    
    def forward(self, x):
        x = x.permute(0, 3, 2, 1)  
        x = super(SpatialDropout, self).forward(x)  
        x = x.permute(0, 3, 2, 1)  
        return x
    
class NeuralNet(nn.Module):
    
    def __init__(self, embed_size, num_targets):
        super(NeuralNet, self).__init__()
        
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding_dropout = SpatialDropout(0.3)
        
        self.lstm1 = nn.LSTM(embed_size * 3, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)
        
        self.linear_out = nn.Linear(LSTM_UNITS * 2, num_targets)

    def forward(self, x):
        h_embedding = self.embedding(x)
        h_embedding = self.embedding_dropout(h_embedding)
        
        h_reshaped = torch.reshape(
            h_embedding, 
            shape=(-1, h_embedding.shape[1],  h_embedding.shape[2] * h_embedding.shape[3])
        )
        
        h_lstm1, _ = self.lstm1(h_reshaped)
        h_lstm2, _ = self.lstm2(h_lstm1)
        h_truncated = h_lstm2[:, :pred_len]
        
        return self.linear_out(h_truncated)


## === cell 7
def loss_fn(outputs, targets):
    colwise_mse = torch.mean(torch.square(targets - outputs), dim=(0, 1))
    loss = torch.mean(torch.sqrt(colwise_mse), dim=-1)
    return loss


## === cell 8
def get_model_optimizer(model):
    def is_linear(name):
        return "linear" in name
    
    optimizer_grouped_parameters = [
       {'params': [param for name, param in model.named_parameters() if not is_linear(name)], 'lr': LR},
       {'params': [param for name, param in model.named_parameters() if is_linear(name)], 'lr': LR*3} 
    ]
    
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=LR, weight_decay=1e-2
    )
    
    return optimizer


## === cell 9
class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


## === cell 10
def train_loop_fn(train_loader, model, optimizer, device, scheduler, epoch=None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':2.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses],
        prefix="[TRAIN] Epoch: [{}]".format(epoch)
    )
    model.train()
    end = time.time()
    for i, data in enumerate(train_loader):
        inputs = data['x']
        targets = data['y']
        inputs = inputs.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), BATCH_SIZE)
        scheduler.step()
        batch_time.update(time.time() - end)
        end = time.time()
        if i % 37 == 0 and i !=0:
            progress.display(i)


## === cell 11
def _run():
    print('Starting Training ... ')
    
    print('  Loading Data ... ')
    train_data = preprocess_inputs(data=train)
    train_labels = np.array(train[pred_cols].values.tolist()).transpose((0, 2, 1))
    train_dataset = OpenVaccineDataset(
        train_data,
        train_labels
    )
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=4
    )
    print('  Data Loading Completed ... ')
    
    print('  Loading Model Configurations ... ')
    num_train_steps = int(len(train_dataset)) / BATCH_SIZE
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = NeuralNet(
        embed_size=EMBEDDING_DIM, 
        num_targets=NUM_TARGETS
    )
    model = model.to(device)
    optimizer = get_model_optimizer(model)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_train_steps*EPOCHS)
    print('  Model Configuration Completed ... ')
    
    print('Training Started ... ')
    for epoch in range(EPOCHS):
        train_loop_fn(
            train_data_loader,
            model,
            optimizer,
            device,
            scheduler,
            epoch
        )
        if epoch == EPOCHS-1:
            print('  Saving Model ...')
            torch.save(model.state_dict(), 'model.bin')
            print('  Model Saved ...')
    
    print('Training Completed.')


## === cell 12
if __name__ == "__main__":
    _run()


## === cell 13
class OpenVaccineTestDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'x': self.data[idx]
        }


## === cell 14
def test_loop_fn(test_loader, model, device):
    model.eval()
    end = time.time()
    preds = []
    for i, data in tqdm(enumerate(test_loader),total=len(test_loader)):
        inputs = data['x']
        inputs = inputs.to(device, dtype=torch.long)
        outputs = model(inputs)
        preds.append(outputs.detach().cpu().numpy())
    return preds


## === cell 15
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

pred_len = 107
model_short = NeuralNet(embed_size=EMBEDDING_DIM, num_targets=NUM_TARGETS).to(device)
model_short.load_state_dict(
    torch.load('model.bin')
)

pred_len = 130
model_long = NeuralNet(embed_size=EMBEDDING_DIM, num_targets=NUM_TARGETS).to(device)
model_long.load_state_dict(
    torch.load('model.bin')
)

public_df = test.query("seq_length == 107").copy()
private_df = test.query("seq_length == 130").copy()

public_inputs = preprocess_inputs(public_df)
private_inputs = preprocess_inputs(private_df)

public_dataset = OpenVaccineTestDataset(public_inputs)
private_dataset = OpenVaccineTestDataset(private_inputs)

public_data_loader = DataLoader(
    public_dataset,
    shuffle=False,
    batch_size=16,
    pin_memory=False,
    drop_last=False,
    num_workers=0
)
private_data_loader = DataLoader(
    private_dataset,
    shuffle=False,
    batch_size=16,
    pin_memory=False,
    drop_last=False,
    num_workers=0
)


## --- ERROR in cell 15, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/387038985.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     17[0m [0;34m[0m[0m
[1;32m     18[0m [0mpublic_inputs[0m [0;34m=[0m [0mpreprocess_inputs[0m[0;34m([0m[0mpublic_df[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 19[0;31m [0mprivate_inputs[0m [0;34m=[0m [0mpreprocess_inputs[0m[0;34m([0m[0mprivate_df[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     20[0m [0;34m[0m[0m
[1;32m     21[0m [0mpublic_dataset[0m [0;34m=[0m [0mOpenVaccineTestDataset[0m[0;34m([0m[0mpublic_inputs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/1615858004.py[0m in [0;36mpreprocess_inputs[0;34m(data)[0m
[1;32m      7[0m     [0mCredits[0m [0mgoes[0m [0mto[0m [0;34m@[0m[0mxhlulu[0m[0;34m:[0m [0mhttps[0m[0;34m:[0m[0;34m//[0m[0mwww[0m[0;34m.[0m[0mkaggle[0m[0;34m.[0m[0mcom[0m[0;34m/[0m[0mxhlulu[0m[0;34m/[0m[0mopenvaccine[0m[0;34m-[0m[0msimple[0m[0;34m-[0m[0mgru[0m[0;34m-[0m[0mmodel[0m[0;34m[0m[0;34m[0m[0m
[1;32m      8[0m     '''
[0;32m----> 9[0;31m     return np.transpose(
[0m[1;32m     10[0m         np.array(
[1;32m     11[0m             [0mdata[0m[0;34m[[0m[0mcols[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/numpy/core/fromnumeric.py[0m in [0;36mtranspose[0;34m(a, axes)[0m
[1;32m    653[0m [0;34m[0m[0m
[1;32m    654[0m     """
[0;32m--> 655[0;31m     [0;32mreturn[0m [0m_wrapfunc[0m[0;34m([0m[0ma[0m[0;34m,[0m [0;34m'transpose'[0m[0;34m,[0m [0maxes[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    656[0m [0;34m[0m[0m
[1;32m    657[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/numpy/core/fromnumeric.py[0m in [0;36m_wrapfunc[0;34m(obj, method, *args, **kwds)[0m
[1;32m     57[0m [0;34m[0m[0m
[1;32m     58[0m     [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 59[0;31m         [0;32mreturn[0m [0mbound[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwds[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     60[0m     [0;32mexcept[0m [0mTypeError[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     61[0m         [0;31m# A TypeError occurs if the object does have such a method in its[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: axes don't match array

## === cell 16
public_preds = np.vstack(test_loop_fn(public_data_loader, model_short, device))
private_preds = np.vstack(test_loop_fn(private_data_loader, model_long, device))
