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

0.54533

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
I removed the stray non‑code text that caused a syntax error, defined the execution device before creating the models, made the tensor conversion robust to pandas objects, and corrected the weighted‑sampler construction to use numpy arrays. These fixes eliminate the runtime “NameError” and datatype issues, allowing the training loops to run and the script to generate a proper `submission.csv` with the required columns.

```


## --- ERROR in cell 0, traceback:
  File "/tmp/ipykernel_55/4144079278.py", line 1
    I removed the stray non‑code text that caused a syntax error, defined the execution device before creating the models, made the tensor conversion robust to pandas objects, and corrected the weighted‑sampler construction to use numpy arrays. These fixes eliminate the runtime “NameError” and datatype issues, allowing the training loops to run and the script to generate a proper `submission.csv` with the required columns.
                           ^
SyntaxError: invalid character '‑' (U+2011)


## === cell 2
%%capture
!pip install ydata-profiling -q



## === cell 3
%%capture
!pip install torchsampler -q



## === cell 4
%%capture
!pip install torchmetrics -q



## === cell 5
import random
import numpy as np 
import pandas as pd 
import os
import datetime
import seaborn as sns
from tqdm.notebook import tqdm
from ydata_profiling import ProfileReport
from collections import Counter

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve

from torchsampler import ImbalancedDatasetSampler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split, SubsetRandomSampler, ConcatDataset
from torch.nn import functional as F
from torchmetrics import ROC
from torchmetrics.classification import BinaryAccuracy

import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('fivethirtyeight')
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')
import itertools



## === cell 6
for dirname, _, filenames in os.walk('/kaggle/input/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 7
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
        return train_test_split(X, y,test_size=0.20,random_state=42)

 
    def standardization_data(self,X_data):
        scaler = StandardScaler()
        std_X_data = scaler.fit_transform(X_data)
        return std_X_data
    

    
data = Datapreparation('/kaggle/input/playground-series-s3e18')
train=data.get_dataframe('train.csv')



## === cell 8
data.summary('train',train)



## === cell 9
pass



## === cell 10
y1 = train['EC1']
y2 = train['EC2']
train.drop(columns=['id','EC1','EC2','EC3','EC4','EC5','EC6'],axis=1,inplace=True)
X = train.copy()



## === cell 11
X_train,X_val,y1_train,y1_val = data.random_split_data(X,y1)
print('Data splits for EC1 \n',X_train.shape,X_val.shape,y1_train.shape,y1_val.shape)



## === cell 12
X_train,X_val,y2_train,y2_val = data.random_split_data(X,y2)
print('Data splits for EC2 \n',X_train.shape,X_val.shape,y2_train.shape,y2_val.shape)



## === cell 13
std_X_train = data.standardization_data(X_train)
std_X_val = data.standardization_data(X_val)
print(std_X_train[0],std_X_val[0])



## === cell 14
class Tensoroperations():
    
    def __init__(self):
        super(Tensoroperations,self).__init__()
    
    def convert_to_tensor(self,X,y=None):
        X_tensor = torch.from_numpy(np.array(X)).float()   
        if y is not None:
            y_tensor = torch.from_numpy(np.array(y)).float() 
            return X_tensor,y_tensor
        return X_tensor, None
        
    def convert_to_test_tensor(self,X):
        X_tensor =  torch.from_numpy(np.array(X)).float()
        return X_tensor
    
    def get_dataloaders(self,train_dataset,val_dataset):
        train_loaders = DataLoader(train_dataset,batch_size=32,shuffle=True)
        val_loaders = DataLoader(val_dataset,batch_size=32)
        return train_loaders,val_loaders
    
    def get_test_dataloaders(self,test_dataset,X_test):
        test_loaders = DataLoader(test_dataset,batch_size=X_test.shape[0])
        return test_loaders
        
        
tenops = Tensoroperations()



## === cell 15
class CustomDataset(Dataset):
    
    def __init__(self,X_data,y_data=None,is_train=True):
        super().__init__()
        self.is_train = is_train
        self.X_data = X_data
        self.y_data = y_data  # may be None for test
    
    def __getitem__(self,index):
        if self.is_train:
            return (self.X_data[index], self.y_data[index])
        else:
            return self.X_data[index]
    
    def __len__(self):
        return len(self.X_data)



## === cell 16
X_tensor_train,y1_tensor_train = tenops.convert_to_tensor(std_X_train,y1_train)
X_tensor_val,y1_tensor_val = tenops.convert_to_tensor(std_X_val,y1_val)
print('The training tensor for EC1\n',X_tensor_train.shape,y1_tensor_train.shape)
print('The validation tensor for EC1\n',X_tensor_val.shape,y1_tensor_val.shape)



## === cell 17
X_tensor_train,y2_tensor_train = tenops.convert_to_tensor(std_X_train,y2_train)
X_tensor_val,y2_tensor_val = tenops.convert_to_tensor(std_X_val,y2_val)
print('The training tensor EC2\n',X_tensor_train.shape,y2_tensor_train.shape)
print('The validation tensor EC2\n',X_tensor_val.shape,y2_tensor_val.shape)



## === cell 18
false_pos,true_pos = [],[]



## === cell 19
class EnzymeClassificationBase(torch.nn.Module):
    
    def _accuracy(self,outputs, labels):
        metric = BinaryAccuracy()
        return metric(outputs, labels.unsqueeze(1))
    
    
    def training_step(self,batch):
        features,labels = batch
        out = self(features)
        loss = F.binary_cross_entropy(out,labels.unsqueeze(1))
        return loss
    
    def validation_step(self, batch):
        features, labels = batch 
        out = self(features)                    # Generate predictions
        loss = F.binary_cross_entropy(out, labels.unsqueeze(1))   # Calculate loss
        acc = self._accuracy(out, labels)           # Calculate accuracy
        self._get_roc(out, labels)
        return {'Validation_loss': loss.detach(), 'Validation_acc': acc}
    
    def _get_roc(self,output,labels):
        roc = ROC(task="binary")
        fpr, tpr, thresholds = roc(output, (labels.long()).unsqueeze(1))
        false_pos.append(fpr)
        true_pos.append(tpr)
        
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['Validation_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['Validation_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'Validation_loss': epoch_loss.item(), 'Validation_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        if epoch%10 == 0:
            print("Epoch [{}], Train_loss: {:.4f}, Validation_loss: {:.4f}, Validation_acc: {:.4f}".format(
                epoch, result['Train_loss'], result['Validation_loss'], result['Validation_acc']))



## === cell 20
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)



## === cell 21
n_input_dim = X_train.shape[1]
n_output =  1   # binary output

class MultiClassificationNN(EnzymeClassificationBase):
    
    def __init__(self):
        super(MultiClassificationNN, self).__init__()
        
        self.network = torch.nn.Sequential(
            torch.nn.Linear(n_input_dim,32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32,16),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(16,8),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(8,n_output),
            torch.nn.Sigmoid()
        )
    
    def forward(self, xb):
        return self.network(xb)

model_EC1 = MultiClassificationNN().to(device)
model_EC2 = MultiClassificationNN().to(device)



## === cell 22
print(model_EC1)



## === cell 23
print(model_EC2)



## === cell 24
train1_dataset = CustomDataset(X_tensor_train,y1_tensor_train)
val1_dataset = CustomDataset(X_tensor_val,y1_tensor_val)

train2_dataset = CustomDataset(X_tensor_train,y2_tensor_train)
val2_dataset = CustomDataset(X_tensor_val,y2_tensor_val)



## === cell 25
from torch.utils.data.sampler import WeightedRandomSampler

counts = np.bincount(y1_train.values)
labels_weights = 1. / counts
weights = labels_weights[y1_train.values]
sampler_train_EC1 = WeightedRandomSampler(weights, len(weights))



## === cell 26
counts = np.bincount(y1_val.values)
labels_weights = 1. / counts
weights = labels_weights[y1_val.values]
sampler_val_EC1 = WeightedRandomSampler(weights, len(weights))



## === cell 27
train_dataloader = DataLoader(train1_dataset,64,sampler=sampler_train_EC1)
val_dataloader = DataLoader(val1_dataset,64,sampler=sampler_val_EC1)



## === cell 28
counts = np.bincount(y2_train.values)
labels_weights = 1. / counts
weights = labels_weights[y2_train.values]
sampler_train_EC2 = WeightedRandomSampler(weights, len(weights))



## === cell 29
counts = np.bincount(y2_val.values)
labels_weights = 1. / counts
weights = labels_weights[y2_val.values]
sampler_val_EC2 = WeightedRandomSampler(weights, len(weights))



## === cell 30
train_dataloader2 = DataLoader(train2_dataset,64,sampler=sampler_train_EC2)
val_dataloader2 = DataLoader(val2_dataset,64,sampler=sampler_val_EC2)



## === cell 31
class Trainer:
    
    @torch.no_grad()
    def _evaluate(self,model, val_loader):
        model.eval()
        outputs = []
        for batch in val_loader:
            xb, yb = batch
            xb, yb = xb.to(device), yb.to(device)
            outputs.append(model.validation_step((xb, yb)))
        return model.validation_epoch_end(outputs)

  
    def fit(self,epochs, lr, model, train_loader, val_loader, opt_func):
    
        history = []
        optimizer = opt_func(model.parameters(),lr,weight_decay=2e-5)
        for epoch in tqdm(range(epochs)):
        
            model.train()
            train_losses = []
            for batch in train_loader:
                xb, yb = batch
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



## === cell 32
num_epochs = 30          # reduced from 100 to keep runtime reasonable
lr = 1e-4
opt_func = torch.optim.Adam



## === cell 33
trainer = Trainer()
history_EC1 = trainer.fit(num_epochs, lr, model_EC1, train_dataloader, val_dataloader, opt_func)



## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/torchmetrics/metric.py in wrapped_func(*args, **kwargs)
    548                 try:
--> 549                     update(*args, **kwargs)
    550                 except RuntimeError as err:

/usr/local/lib/python3.11/dist-packages/torchmetrics/classification/stat_scores.py in update(self, preds, target)
    189         tp, fp, tn, fn = _binary_stat_scores_update(preds, target, self.multidim_average)
--> 190         self._update_state(tp, fp, tn, fn)
    191 

/usr/local/lib/python3.11/dist-packages/torchmetrics/classification/stat_scores.py in _update_state(self, tp, fp, tn, fn)
     76         else:
---> 77             self.tp = self.tp + tp if not isinstance(self.tp, list) else [*self.tp, tp]
     78             self.fp = self.fp + fp if not isinstance(self.fp, list) else [*self.fp, fp]

RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!

The above exception was the direct cause of the following exception:

RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/2663417825.py in <cell line: 0>()
      1 trainer = Trainer()
----> 2 history_EC1 = trainer.fit(num_epochs, lr, model_EC1, train_dataloader, val_dataloader, opt_func)
      3 

/tmp/ipykernel_55/2021796789.py in fit(self, epochs, lr, model, train_loader, val_loader, opt_func)
     29                 optimizer.zero_grad()
     30 
---> 31             result = self._evaluate(model, val_loader)
     32             result['Train_loss'] = torch.stack(train_losses).mean().item()
     33             model.epoch_end(epoch, result)

/usr/local/lib/python3.11/dist-packages/torch/utils/_contextlib.py in decorate_context(*args, **kwargs)
    114     def decorate_context(*args, **kwargs):
    115         with ctx_factory():
--> 116             return func(*args, **kwargs)
    117 
    118     return decorate_context

/tmp/ipykernel_55/2021796789.py in _evaluate(self, model, val_loader)
      8             xb, yb = batch
      9             xb, yb = xb.to(device), yb.to(device)
---> 10             outputs.append(model.validation_step((xb, yb)))
     11         return model.validation_epoch_end(outputs)
     12 

/tmp/ipykernel_55/3672676204.py in validation_step(self, batch)
     16         out = self(features)                    # Generate predictions
     17         loss = F.binary_cross_entropy(out, labels.unsqueeze(1))   # Calculate loss
---> 18         acc = self._accuracy(out, labels)           # Calculate accuracy
     19         self._get_roc(out, labels)
     20         return {'Validation_loss': loss.detach(), 'Validation_acc': acc}

/tmp/ipykernel_55/3672676204.py in _accuracy(self, outputs, labels)
      3     def _accuracy(self,outputs, labels):
      4         metric = BinaryAccuracy()
----> 5         return metric(outputs, labels.unsqueeze(1))
      6 
      7 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torchmetrics/metric.py in forward(self, *args, **kwargs)
    313             self._forward_cache = self._forward_full_state_update(*args, **kwargs)
    314         else:
--> 315             self._forward_cache = self._forward_reduce_state_update(*args, **kwargs)
    316 
    317         return self._forward_cache

/usr/local/lib/python3.11/dist-packages/torchmetrics/metric.py in _forward_reduce_state_update(self, *args, **kwargs)
    382 
    383         # calculate batch state and compute batch value
--> 384         self.update(*args, **kwargs)
    385         batch_val = self.compute()
    386 

/usr/local/lib/python3.11/dist-packages/torchmetrics/metric.py in wrapped_func(*args, **kwargs)
    550                 except RuntimeError as err:
    551                     if "Expected all tensors to be on" in str(err):
--> 552                         raise RuntimeError(
    553                             "Encountered different devices in metric calculation (see stacktrace for details)."
    554                             " This could be due to the metric class not being on the same device as input."

RuntimeError: Encountered different devices in metric calculation (see stacktrace for details). This could be due to the metric class not being on the same device as input. Instead of `metric=BinaryAccuracy(...)` try to do `metric=BinaryAccuracy(...).to(device)` where device corresponds to the device of the input.

## === cell 34
def plot_accuracies(history):
    """ Plot the history of accuracies"""
    accuracies = [x['Validation_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    plt.title('Validation Accuracy vs. Epoch');
    
plot_accuracies(history_EC1)



## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3799099402.py in <cell line: 0>()
      7     plt.title('Validation Accuracy vs. Epoch');
      8 
----> 9 plot_accuracies(history_EC1)
     10 

NameError: name 'history_EC1' is not defined

## === cell 35
def plot_losses(history):
    """ Plot the losses in each epoch"""
    train_losses = [x.get('Train_loss') for x in history]
    val_losses = [x['Validation_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. Epoch')
    
plot_losses(history_EC1)



## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3029708354.py in <cell line: 0>()
     10     plt.title('Loss vs. Epoch')
     11 
---> 12 plot_losses(history_EC1)
     13 

NameError: name 'history_EC1' is not defined

## === cell 36
fpr,tpr = [],[]
for i in range(len(false_pos)):
    fpr.append(np.mean(false_pos[i].cpu().numpy()))
for j in range(len(true_pos)):
    tpr.append(np.mean(true_pos[j].cpu().numpy()))



## === cell 37
with torch.no_grad():
    plt.plot(fpr,tpr) # ROC curve = TPR vs FPR
    plt.title("Receiver Operating Characteristics")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()



## === cell 38
history_EC2 = trainer.fit(num_epochs, lr, model_EC2, train_dataloader2, val_dataloader2, opt_func)



## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/torchmetrics/metric.py in wrapped_func(*args, **kwargs)
    548                 try:
--> 549                     update(*args, **kwargs)
    550                 except RuntimeError as err:

/usr/local/lib/python3.11/dist-packages/torchmetrics/classification/stat_scores.py in update(self, preds, target)
    189         tp, fp, tn, fn = _binary_stat_scores_update(preds, target, self.multidim_average)
--> 190         self._update_state(tp, fp, tn, fn)
    191 

/usr/local/lib/python3.11/dist-packages/torchmetrics/classification/stat_scores.py in _update_state(self, tp, fp, tn, fn)
     76         else:
---> 77             self.tp = self.tp + tp if not isinstance(self.tp, list) else [*self.tp, tp]
     78             self.fp = self.fp + fp if not isinstance(self.fp, list) else [*self.fp, fp]

RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!

The above exception was the direct cause of the following exception:

RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/2875191281.py in <cell line: 0>()
----> 1 history_EC2 = trainer.fit(num_epochs, lr, model_EC2, train_dataloader2, val_dataloader2, opt_func)
      2 

/tmp/ipykernel_55/2021796789.py in fit(self, epochs, lr, model, train_loader, val_loader, opt_func)
     29                 optimizer.zero_grad()
     30 
---> 31             result = self._evaluate(model, val_loader)
     32             result['Train_loss'] = torch.stack(train_losses).mean().item()
     33             model.epoch_end(epoch, result)

/usr/local/lib/python3.11/dist-packages/torch/utils/_contextlib.py in decorate_context(*args, **kwargs)
    114     def decorate_context(*args, **kwargs):
    115         with ctx_factory():
--> 116             return func(*args, **kwargs)
    117 
    118     return decorate_context

/tmp/ipykernel_55/2021796789.py in _evaluate(self, model, val_loader)
      8             xb, yb = batch
      9             xb, yb = xb.to(device), yb.to(device)
---> 10             outputs.append(model.validation_step((xb, yb)))
     11         return model.validation_epoch_end(outputs)
     12 

/tmp/ipykernel_55/3672676204.py in validation_step(self, batch)
     16         out = self(features)                    # Generate predictions
     17         loss = F.binary_cross_entropy(out, labels.unsqueeze(1))   # Calculate loss
---> 18         acc = self._accuracy(out, labels)           # Calculate accuracy
     19         self._get_roc(out, labels)
     20         return {'Validation_loss': loss.detach(), 'Validation_acc': acc}

/tmp/ipykernel_55/3672676204.py in _accuracy(self, outputs, labels)
      3     def _accuracy(self,outputs, labels):
      4         metric = BinaryAccuracy()
----> 5         return metric(outputs, labels.unsqueeze(1))
      6 
      7 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torchmetrics/metric.py in forward(self, *args, **kwargs)
    313             self._forward_cache = self._forward_full_state_update(*args, **kwargs)
    314         else:
--> 315             self._forward_cache = self._forward_reduce_state_update(*args, **kwargs)
    316 
    317         return self._forward_cache

/usr/local/lib/python3.11/dist-packages/torchmetrics/metric.py in _forward_reduce_state_update(self, *args, **kwargs)
    382 
    383         # calculate batch state and compute batch value
--> 384         self.update(*args, **kwargs)
    385         batch_val = self.compute()
    386 

/usr/local/lib/python3.11/dist-packages/torchmetrics/metric.py in wrapped_func(*args, **kwargs)
    550                 except RuntimeError as err:
    551                     if "Expected all tensors to be on" in str(err):
--> 552                         raise RuntimeError(
    553                             "Encountered different devices in metric calculation (see stacktrace for details)."
    554                             " This could be due to the metric class not being on the same device as input."

RuntimeError: Encountered different devices in metric calculation (see stacktrace for details). This could be due to the metric class not being on the same device as input. Instead of `metric=BinaryAccuracy(...)` try to do `metric=BinaryAccuracy(...).to(device)` where device corresponds to the device of the input.

## === cell 39
plot_accuracies(history_EC2)



## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1564203759.py in <cell line: 0>()
----> 1 plot_accuracies(history_EC2)
      2 

NameError: name 'history_EC2' is not defined

## === cell 40
plot_losses(history_EC2)



## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2034444355.py in <cell line: 0>()
----> 1 plot_losses(history_EC2)
      2 

NameError: name 'history_EC2' is not defined

## === cell 41
fpr2,tpr2 = [],[]
for i in range(len(false_pos)):
    fpr2.append(np.mean(false_pos[i].cpu().numpy()))
for j in range(len(true_pos)):
    tpr2.append(np.mean(true_pos[j].cpu().numpy()))



## === cell 42
with torch.no_grad():
    plt.plot(fpr2, tpr2)
    plt.title("Receiver Operating Characteristics (EC2)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()



## === cell 43
test=data.get_dataframe('test.csv')



## === cell 44
data.summary('test',test)



## === cell 45
pass



## === cell 46
test_update = test.loc[:, test.columns != 'id']
std_X_test = data.standardization_data(test_update)
print(std_X_test[0])



## === cell 47
X_tensor_test = tenops.convert_to_test_tensor(std_X_test)
print(X_tensor_test.shape)



## === cell 48
class CustomDataTest(Dataset):
    def __init__(self, X_data):
        self.X_data = X_data

    def __getitem__(self, index):
        return self.X_data[index]
    
    def __len__(self):
        return len(self.X_data)



## === cell 49
test_dataset = CustomDataTest(X_tensor_test)
test_dataloader = DataLoader(test_dataset,64)



## === cell 50
class Evaluate:
        
    def eval_test_data(self,model,test_data_dl):
        preds = []
        model.eval()
        with torch.no_grad():
            for X_batch_test in test_data_dl:
                X_batch_test = X_batch_test.to(device)
                y_test_pred = model(X_batch_test)
                y_pred_tag = torch.sigmoid(y_test_pred)
                preds.append(y_pred_tag.cpu())
        return [a.squeeze().tolist() for a in preds]
    
    
eva = Evaluate()



## === cell 51
from collections.abc import Iterable
def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:        
            yield item
            
ec1 = eva.eval_test_data(model_EC1,test_dataloader)
ec2 = eva.eval_test_data(model_EC2,test_dataloader)



## === cell 52
class Submit:
    
    def submit_predictions(self):        
        df_submit = pd.DataFrame(data={'id': test['id'],
                                       'EC1': list(flatten(ec1)),
                                       'EC2': list(flatten(ec2))})
        df_submit.to_csv('submission.csv',index=False)
        print('Submission Completed!!')
        return df_submit
        
        
submit = Submit()
df_submit = submit.submit_predictions()



## === cell 53
df_submit.head()
```

## --- ERROR in cell 53, traceback:
  File "/tmp/ipykernel_55/2627412113.py", line 2
    ```
    ^
SyntaxError: invalid syntax
