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
Predict values for synthetic data.

### Description
## Metric
Area under the ROC curve for each target, with the final score being the average of the individual AUCs of each predicted column.

## Submission Format
For each `id` in the test set, you must predict the value for the targets `EC1` and `EC2`. The file should contain a header and have the following format:

```
id,EC1,EC2
14838,0.22,0.71
14839,0.78,0.43
14840,0.53,0.11
etc.
```

## Dataset 
- **train.csv** - the training dataset; `[EC1 - EC6]` are the (binary) targets, although you are only asked to predict `EC1` and `EC2`.
- **test.csv** - the test dataset; your objective is to predict the probability of the two targets `EC1` and `EC2`
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.11

# 3. Installed packages

geopandas==0.14.4
imbalanced-learn==0.13.0
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
plotly==5.24.1
plotly-express==0.4.1
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
ydata-profiling==4.17.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (1485 lines)
            sample_submission.csv.zip (4.4 kB)
            test.csv (1485 lines)
            test.csv.zip (147.8 kB)
            train.csv (13355 lines)
            train.csv.zip (1.4 MB)
            playground-series-s3e18/
                description.md (56 lines)
                sample_submission.csv (1485 lines)
                ... and 5 other files
                playground-series-s3e18/
        input/
            description.md (56 lines)
            sample_submission.csv (1485 lines)
            sample_submission.csv.zip (4.4 kB)
            test.csv (1485 lines)
            test.csv.zip (147.8 kB)
            train.csv (13355 lines)
            train.csv.zip (1.4 MB)
            playground-series-s3e18/
                description.md (56 lines)
                sample_submission.csv (1485 lines)
                ... and 5 other files
                playground-series-s3e18/
        working/
            playground-series-s3e18/
                description.md (56 lines)
                sample_submission.csv (1485 lines)
                ... and 5 other files
                playground-series-s3e18/
```

-> data/playground-series-s3e18/sample_submission.csv has 1484 rows and 3 columns.
The columns are: id, EC1, EC2

-> data/playground-series-s3e18/test.csv has 1484 rows and 32 columns.
The columns are: id, BertzCT, Chi1, Chi1n, Chi1v, Chi2n, Chi2v, Chi3v, Chi4n, EState_VSA1, EState_VSA2, ExactMolWt, FpDensityMorgan1, FpDensityMorgan2, FpDensityMorgan3... and 17 more columns

-> data/playground-series-s3e18/train.csv has 13354 rows and 38 columns.
The columns are: id, BertzCT, Chi1, Chi1n, Chi1v, Chi2n, Chi2v, Chi3v, Chi4n, EState_VSA1, EState_VSA2, ExactMolWt, FpDensityMorgan1, FpDensityMorgan2, FpDensityMorgan3... and 23 more columns

-> data/sample_submission.csv has 1484 rows and 3 columns.
The columns are: id, EC1, EC2

-> data/test.csv has 1484 rows and 32 columns.
The columns are: id, BertzCT, Chi1, Chi1n, Chi1v, Chi2n, Chi2v, Chi3v, Chi4n, EState_VSA1, EState_VSA2, ExactMolWt, FpDensityMorgan1, FpDensityMorgan2, FpDensityMorgan3... and 17 more columns

-> data/train.csv has 13354 rows and 38 columns.
The columns are: id, BertzCT, Chi1, Chi1n, Chi1v, Chi2n, Chi2v, Chi3v, Chi4n, EState_VSA1, EState_VSA2, ExactMolWt, FpDensityMorgan1, FpDensityMorgan2, FpDensityMorgan3... and 23 more columns

-> (stopped after 10 files for performance)

# 5. Target score

0.63645

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
I removed the stray non‑code text that caused a syntax error, moved the `BinaryAccuracy` metric onto the same device as the model outputs to fix the device‑mismatch error, and cleaned up the final cell that contained stray markdown fences. These changes allow the training loops for both targets to run without crashing and produce a proper `submission.csv` file, moving the score closer to the target.

```


## --- ERROR in cell 0, traceback:
  File "/tmp/ipykernel_55/1045337348.py", line 1
    I removed the stray non‑code text that caused a syntax error, moved the `BinaryAccuracy` metric onto the same device as the model outputs to fix the device‑mismatch error, and cleaned up the final cell that contained stray markdown fences. These changes allow the training loops for both targets to run without crashing and produce a proper `submission.csv` file, moving the score closer to the target.
                           ^
SyntaxError: invalid character '‑' (U+2011)


## === cell 1
%%capture
!pip install ydata-profiling -q



## === cell 2
%%capture
!pip install torchsampler -q



## === cell 3
%%capture
!pip install torchmetrics -q



## === cell 4
import random
import numpy as np 
import pandas as pd 
import os
import seaborn as sns
from tqdm.notebook import tqdm
from ydata_profiling import ProfileReport
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torchmetrics import ROC
from torchmetrics.classification import BinaryAccuracy

import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')



## === cell 5
for dirname, _, filenames in os.walk('/kaggle/input/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 6
class Datapreparation(object):
    
    def __init__(self,root_path):
        self.root_path = root_path
        
    def get_dataframe(self,filename):
        return pd.read_csv(os.path.join(self.root_path,filename))
    
    def summary(self,text, df):
        summary = pd.DataFrame(df.dtypes, columns=['dtypes'])
        summary['null'] = df.isnull().sum()
        summary['unique'] = df.nunique()
        summary['min'] = df.min()
        summary['median'] = df.median()
        summary['max'] = df.max()
        summary['mean'] = df.mean()
        summary['std'] = df.std()
        summary['duplicate'] = df.duplicated().sum()
        return summary
    
    def random_split_data(self,X,y):
        return train_test_split(X, y, test_size=0.20, random_state=42)

    def standardization_data(self,X_data):
        scaler = StandardScaler()
        return scaler.fit_transform(X_data)

data = Datapreparation('/kaggle/input/playground-series-s3e18')
train = data.get_dataframe('train.csv')



## === cell 8
y1 = train['EC1']
y2 = train['EC2']
train.drop(columns=['id','EC1','EC2','EC3','EC4','EC5','EC6'],axis=1,inplace=True)
X = train.copy()



## === cell 9
X_train,X_val,y1_train,y1_val = data.random_split_data(X,y1)
print('EC1 split:', X_train.shape, X_val.shape, y1_train.shape, y1_val.shape)



## === cell 10
X_train,X_val,y2_train,y2_val = data.random_split_data(X,y2)
print('EC2 split:', X_train.shape, X_val.shape, y2_train.shape, y2_val.shape)



## === cell 11
std_X_train = data.standardization_data(X_train)
std_X_val = data.standardization_data(X_val)



## === cell 12
class Tensoroperations():
    
    def convert_to_tensor(self,X,y=None):
        X_tensor = torch.from_numpy(np.array(X)).float()
        if y is not None:
            y_tensor = torch.from_numpy(np.array(y)).float()
            return X_tensor, y_tensor
        return X_tensor, None
        
    def convert_to_test_tensor(self,X):
        return torch.from_numpy(np.array(X)).float()
    
tenops = Tensoroperations()



## === cell 13
class CustomDataset(Dataset):
    
    def __init__(self,X_data,y_data=None):
        self.X_data = X_data
        self.y_data = y_data
    
    def __getitem__(self,index):
        if self.y_data is not None:
            return self.X_data[index], self.y_data[index]
        return self.X_data[index]
    
    def __len__(self):
        return len(self.X_data)



## === cell 14
X_tensor_train, y1_tensor_train = tenops.convert_to_tensor(std_X_train, y1_train)
X_tensor_val,   y1_tensor_val   = tenops.convert_to_tensor(std_X_val,   y1_val)

X_tensor_train_ec2, y2_tensor_train = tenops.convert_to_tensor(std_X_train, y2_train)
X_tensor_val_ec2,   y2_tensor_val   = tenops.convert_to_tensor(std_X_val,   y2_val)



## === cell 15
false_pos, true_pos = [], []



## === cell 16
class EnzymeClassificationBase(torch.nn.Module):
    
    def _accuracy(self, outputs, labels):
        metric = BinaryAccuracy().to(device)
        return metric(outputs, labels.unsqueeze(1))
    
    def training_step(self,batch):
        features, labels = batch
        out = self(features)
        loss = F.binary_cross_entropy(out, labels.unsqueeze(1))
        return loss
    
    def validation_step(self, batch):
        features, labels = batch 
        out = self(features)
        loss = F.binary_cross_entropy(out, labels.unsqueeze(1))
        acc = self._accuracy(out, labels)
        self._get_roc(out, labels)
        return {'Validation_loss': loss.detach(), 'Validation_acc': acc}
    
    def _get_roc(self, output, labels):
        roc = ROC(task="binary")
        fpr, tpr, _ = roc(output, (labels.long()).unsqueeze(1))
        false_pos.append(fpr)
        true_pos.append(tpr)
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['Validation_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['Validation_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'Validation_loss': epoch_loss.item(), 'Validation_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        if epoch % 10 == 0:
            print(f"Epoch [{epoch}], Train_loss: {result['Train_loss']:.4f}, "
                  f"Validation_loss: {result['Validation_loss']:.4f}, "
                  f"Validation_acc: {result['Validation_acc']:.4f}")



## === cell 17
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)



## === cell 18
n_input_dim = X_train.shape[1]
n_output = 1

class MultiClassificationNN(EnzymeClassificationBase):
    def __init__(self):
        super(MultiClassificationNN, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(n_input_dim, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(16, 8),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(8, n_output),
            torch.nn.Sigmoid()
        )
    
    def forward(self, xb):
        return self.network(xb)

model_EC1 = MultiClassificationNN().to(device)
model_EC2 = MultiClassificationNN().to(device)



## === cell 19
train1_dataset = CustomDataset(X_tensor_train, y1_tensor_train)
val1_dataset   = CustomDataset(X_tensor_val,   y1_tensor_val)

counts = np.bincount(y1_train.values)
weights = (1. / counts)[y1_train.values]
sampler_train_EC1 = torch.utils.data.WeightedRandomSampler(weights, len(weights))
sampler_val_EC1   = torch.utils.data.WeightedRandomSampler(weights, len(weights))

train_dataloader_EC1 = DataLoader(train1_dataset, batch_size=64, sampler=sampler_train_EC1)
val_dataloader_EC1   = DataLoader(val1_dataset,   batch_size=64, sampler=sampler_val_EC1)

train2_dataset = CustomDataset(X_tensor_train_ec2, y2_tensor_train)
val2_dataset   = CustomDataset(X_tensor_val_ec2,   y2_tensor_val)

counts2 = np.bincount(y2_train.values)
weights2 = (1. / counts2)[y2_train.values]
sampler_train_EC2 = torch.utils.data.WeightedRandomSampler(weights2, len(weights2))
sampler_val_EC2   = torch.utils.data.WeightedRandomSampler(weights2, len(weights2))

train_dataloader_EC2 = DataLoader(train2_dataset, batch_size=64, sampler=sampler_train_EC2)
val_dataloader_EC2   = DataLoader(val2_dataset,   batch_size=64, sampler=sampler_val_EC2)



## === cell 20
class Trainer:
    
    @torch.no_grad()
    def _evaluate(self, model, val_loader):
        model.eval()
        outputs = []
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs.append(model.validation_step((xb, yb)))
        return model.validation_epoch_end(outputs)

    def fit(self, epochs, lr, model, train_loader, val_loader, opt_func):
        history = []
        optimizer = opt_func(model.parameters(), lr, weight_decay=2e-5)
        for epoch in tqdm(range(epochs)):
            model.train()
            train_losses = []
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = model.training_step((xb, yb))
                train_losses.append(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            result = self._evaluate(model, val_loader)
            result['Train_loss'] = torch.stack(train_losses).mean().item()
            model.epoch_end(epoch, result)
            history.append(result)
        return history



## === cell 21
num_epochs = 30
lr = 1e-4
optimizer_fn = torch.optim.Adam
trainer = Trainer()



## === cell 22
history_EC1 = trainer.fit(num_epochs, lr, model_EC1,
                          train_dataloader_EC1, val_dataloader_EC1,
                          optimizer_fn)



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_55/2060122490.py in <cell line: 0>()
----> 1 history_EC1 = trainer.fit(num_epochs, lr, model_EC1,
      2                           train_dataloader_EC1, val_dataloader_EC1,
      3                           optimizer_fn)
      4 

/tmp/ipykernel_55/399488159.py in fit(self, epochs, lr, model, train_loader, val_loader, opt_func)
     23                 optimizer.step()
     24                 optimizer.zero_grad()
---> 25             result = self._evaluate(model, val_loader)
     26             result['Train_loss'] = torch.stack(train_losses).mean().item()
     27             model.epoch_end(epoch, result)

/usr/local/lib/python3.11/dist-packages/torch/utils/_contextlib.py in decorate_context(*args, **kwargs)
    114     def decorate_context(*args, **kwargs):
    115         with ctx_factory():
--> 116             return func(*args, **kwargs)
    117 
    118     return decorate_context

/tmp/ipykernel_55/399488159.py in _evaluate(self, model, val_loader)
      5         model.eval()
      6         outputs = []
----> 7         for xb, yb in val_loader:
      8             xb, yb = xb.to(device), yb.to(device)
      9             outputs.append(model.validation_step((xb, yb)))

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
    762     def _next_data(self):
    763         index = self._next_index()  # may raise StopIteration
--> 764         data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
    765         if self._pin_memory:
    766             data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in fetch(self, possibly_batched_index)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in <listcomp>(.0)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/tmp/ipykernel_55/3762267917.py in __getitem__(self, index)
      7     def __getitem__(self,index):
      8         if self.y_data is not None:
----> 9             return self.X_data[index], self.y_data[index]
     10         return self.X_data[index]
     11 

IndexError: index 5840 is out of bounds for dimension 0 with size 2671

## === cell 23
def plot_accuracies(history):
    accuracies = [x['Validation_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy vs. Epoch')
    plt.show()
    
plot_accuracies(history_EC1)



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2387654310.py in <cell line: 0>()
      7     plt.show()
      8 
----> 9 plot_accuracies(history_EC1)
     10 

NameError: name 'history_EC1' is not defined

## === cell 24
def plot_losses(history):
    train_losses = [x.get('Train_loss') for x in history]
    val_losses   = [x['Validation_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. Epoch')
    plt.show()
    
plot_losses(history_EC1)



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3027762041.py in <cell line: 0>()
     10     plt.show()
     11 
---> 12 plot_losses(history_EC1)
     13 

NameError: name 'history_EC1' is not defined

## === cell 25
fpr_ec1, tpr_ec1 = [], []
for fp, tp in zip(false_pos, true_pos):
    fpr_ec1.append(fp.mean().cpu().numpy())
    tpr_ec1.append(tp.mean().cpu().numpy())

with torch.no_grad():
    plt.plot(fpr_ec1, tpr_ec1)
    plt.title("ROC Curve (EC1)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()



## === cell 26
history_EC2 = trainer.fit(num_epochs, lr, model_EC2,
                          train_dataloader_EC2, val_dataloader_EC2,
                          optimizer_fn)



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_55/89750894.py in <cell line: 0>()
      1 # Train EC2
----> 2 history_EC2 = trainer.fit(num_epochs, lr, model_EC2,
      3                           train_dataloader_EC2, val_dataloader_EC2,
      4                           optimizer_fn)
      5 

/tmp/ipykernel_55/399488159.py in fit(self, epochs, lr, model, train_loader, val_loader, opt_func)
     23                 optimizer.step()
     24                 optimizer.zero_grad()
---> 25             result = self._evaluate(model, val_loader)
     26             result['Train_loss'] = torch.stack(train_losses).mean().item()
     27             model.epoch_end(epoch, result)

/usr/local/lib/python3.11/dist-packages/torch/utils/_contextlib.py in decorate_context(*args, **kwargs)
    114     def decorate_context(*args, **kwargs):
    115         with ctx_factory():
--> 116             return func(*args, **kwargs)
    117 
    118     return decorate_context

/tmp/ipykernel_55/399488159.py in _evaluate(self, model, val_loader)
      5         model.eval()
      6         outputs = []
----> 7         for xb, yb in val_loader:
      8             xb, yb = xb.to(device), yb.to(device)
      9             outputs.append(model.validation_step((xb, yb)))

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
    762     def _next_data(self):
    763         index = self._next_index()  # may raise StopIteration
--> 764         data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
    765         if self._pin_memory:
    766             data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in fetch(self, possibly_batched_index)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in <listcomp>(.0)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/tmp/ipykernel_55/3762267917.py in __getitem__(self, index)
      7     def __getitem__(self,index):
      8         if self.y_data is not None:
----> 9             return self.X_data[index], self.y_data[index]
     10         return self.X_data[index]
     11 

IndexError: index 2861 is out of bounds for dimension 0 with size 2671

## === cell 27
plot_accuracies(history_EC2)
plot_losses(history_EC2)



## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/188872238.py in <cell line: 0>()
----> 1 plot_accuracies(history_EC2)
      2 plot_losses(history_EC2)
      3 

NameError: name 'history_EC2' is not defined

## === cell 28
fpr_ec2, tpr_ec2 = [], []
for fp, tp in zip(false_pos, true_pos):
    fpr_ec2.append(fp.mean().cpu().numpy())
    tpr_ec2.append(tp.mean().cpu().numpy())

with torch.no_grad():
    plt.plot(fpr_ec2, tpr_ec2)
    plt.title("ROC Curve (EC2)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()



## === cell 29
test = data.get_dataframe('test.csv')



## === cell 30
test_update = test.loc[:, test.columns != 'id']
std_X_test = data.standardization_data(test_update)



## === cell 31
X_tensor_test = tenops.convert_to_test_tensor(std_X_test)



## === cell 32
class CustomDataTest(Dataset):
    def __init__(self, X_data):
        self.X_data = X_data
    def __getitem__(self, index):
        return self.X_data[index]
    def __len__(self):
        return len(self.X_data)

test_dataset = CustomDataTest(X_tensor_test)
test_dataloader = DataLoader(test_dataset, batch_size=64)



## === cell 33
class Evaluate:
    def eval_test_data(self, model, test_loader):
        preds = []
        model.eval()
        with torch.no_grad():
            for X_batch in test_loader:
                X_batch = X_batch.to(device)
                y_pred = model(X_batch)
                preds.append(y_pred.cpu())
        return [p.squeeze().tolist() for p in preds]

evaluator = Evaluate()



## === cell 34
ec1_preds = evaluator.eval_test_data(model_EC1, test_dataloader)
ec2_preds = evaluator.eval_test_data(model_EC2, test_dataloader)



## === cell 35
def flatten(lis):
    for item in lis:
        if isinstance(item, (list, tuple)):
            for sub in flatten(item):
                yield sub
        else:
            yield item



## === cell 36
submission = pd.DataFrame({
    'id': test['id'],
    'EC1': list(flatten(ec1_preds)),
    'EC2': list(flatten(ec2_preds))
})
submission.to_csv('submission.csv', index=False)
print('Submission file saved as submission.csv')
```

## --- ERROR in cell 36, traceback:
  File "/tmp/ipykernel_55/1995243434.py", line 8
    ```
    ^
SyntaxError: invalid syntax
