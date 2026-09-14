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

3.11

# 2. Installed packages

array_record==0.7.2
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
ray==2.51.1
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
xarray==2025.7.1
xarray-einstats==0.9.1
ydata-profiling==4.17.0

# 3. Data file paths

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

# 4. Code solution

## === cell 0
%%capture
!pip install ydata-profiling


## === cell 1
%%capture
!pip install torchsampler


## === cell 2
%%capture
!pip install torchmetrics


## === cell 3
%%capture
!pip install ray


## === cell 4
import random
import numpy as np
import pandas as pd
import os
import datetime
import seaborn as sns
from tqdm.notebook import tqdm
from ydata_profiling import ProfileReport
from collections import Counter

try:
    from imblearn.over_sampling import SMOTE, SMOTEN
except Exception:
    SMOTE, SMOTEN = None, None

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve

from torchsampler import ImbalancedDatasetSampler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import (
    Dataset,
    DataLoader,
    TensorDataset,
    random_split,
    SubsetRandomSampler,
    ConcatDataset,
)
from torch.nn import functional as F
from torchmetrics import ROC
from torchmetrics.classification import BinaryAccuracy


import ray
from ray import tune
from ray.air import session

try:
    from ray.train import Checkpoint
except Exception:
    from ray.air import Checkpoint
from ray.tune.schedulers import ASHAScheduler


import matplotlib.pyplot as plt

get_ipython().run_line_magic("matplotlib", "inline")
plt.style.use("fivethirtyeight")
import plotly.express as px

import warnings

warnings.filterwarnings("ignore")
import itertools


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
        return train_test_split(X, y,test_size=0.20,random_state=42)

 
    def standardization_data(self,X_data):
        scaler = StandardScaler()
        std_X_data = scaler.fit_transform(X_data)
        return std_X_data
    

    
data = Datapreparation('/kaggle/input/playground-series-s3e18')
train=data.get_dataframe('train.csv')


## === cell 7
data.summary('train',train)


## === cell 9
y1 = train['EC1']
y2 = train['EC2']
train.drop(columns=['id','EC1','EC2','EC3','EC4','EC5','EC6'],axis=1,inplace=True)
X = train.copy()


## === cell 10
X_train,X_val,y1_train,y1_val = data.random_split_data(X,y1)
print('Data splits for EC1 \n',X_train.shape,X_val.shape,y1_train.shape,y1_val.shape)


## === cell 11
X_train,X_val,y2_train,y2_val = data.random_split_data(X,y2)
print('Data splits for EC2 \n',X_train.shape,X_val.shape,y2_train.shape,y2_val.shape)


## === cell 12
std_X_train = data.standardization_data(X_train)
std_X_val = data.standardization_data(X_val)
print(std_X_train[0],std_X_val[0])


## === cell 13
class Tensoroperations():
    
    def __init__(self):
        super(Tensoroperations,self).__init__()
    
    def convert_to_tensor(self,X,y=None):
        X_tensor =  torch.from_numpy(X).float()   
        y_tensor = torch.from_numpy(y).float() 
        return X_tensor,y_tensor
        
    def convert_to_test_tensor(self,X):
        X_tensor =  torch.from_numpy(X).float()
        return X_tensor
    
    def get_dataloaders(self,train_dataset,val_dataset):
        train_loaders = DataLoader(train_dataset,batch_size=32,shuffle=True)
        val_loaders = DataLoader(val_dataset,batch_size=32)
        return train_loaders,val_loaders
    
    def get_test_dataloaders(self,test_dataset,X_test):
        test_loaders = DataLoader(test_dataset,batch_size=X_test.shape[0])
        return test_loaders
        
        
    
tenops = Tensoroperations()


## === cell 14
class CustomDataset(Dataset):
    
    def __init__(self,X_data,y_data=None,is_train=True):
        super().__init__()
        if is_train:
            self.X_data = X_data
            self.y_data = y_data
        else:
            self.X_data=X_train
            
    def __getitem__(self,index):
        return (self.X_data[index],self.y_data[index])
    
    def __len__(self):
        return len(self.X_data)


## === cell 15
X_tensor_train,y1_tensor_train = tenops.convert_to_tensor(std_X_train,y1_train.values)
X_tensor_val,y1_tensor_val = tenops.convert_to_tensor(std_X_val,y1_val.values)
print('The training tensor for EC1\n',X_tensor_train,y1_tensor_train)
print('The validation tensor for EC1\n',X_tensor_val,y1_tensor_val)


## === cell 16
X_tensor_train,y2_tensor_train = tenops.convert_to_tensor(std_X_train,y2_train.values)
X_tensor_val,y2_tensor_val = tenops.convert_to_tensor(std_X_val,y2_val.values)
print('The training tensor EC2\n',X_tensor_train,y2_tensor_train)
print('The validation tensor Ec2\n',X_tensor_val,y2_tensor_val)


## === cell 17
false_pos,true_pos = [],[]


## === cell 18
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
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['Validation_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'Validation_loss': epoch_loss.item(), 'Validation_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        if epoch%10 == 0:
            print("Epoch [{}], Train_loss: {:.4f}, Validation_loss: {:.4f}, Validation_acc: {:.4f}".format(
                epoch, result['Train_loss'], result['Validation_loss'], result['Validation_acc']))


## === cell 19
n_input_dim = X_train.shape[1]
n_output =  1   # Number of output nodes = for binary classifier

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
            torch.nn.Linear(8,4),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(4,n_output),
            torch.nn.Sigmoid()
        )
    
    def forward(self, xb):
        return self.network(xb)


model_EC1 = MultiClassificationNN()
model_EC2 = MultiClassificationNN()


## === cell 20
print(model_EC1)


## === cell 21
print(model_EC2)


## === cell 22
train1_dataset = CustomDataset(X_tensor_train,y1_tensor_train)
val1_dataset = CustomDataset(X_tensor_val,y1_tensor_val)

train2_dataset = CustomDataset(X_tensor_train,y2_tensor_train)
val2_dataset = CustomDataset(X_tensor_val,y2_tensor_val)


## === cell 23
train_dataloader = DataLoader(train1_dataset,64,shuffle=True)
val_dataloader = DataLoader(val1_dataset,64)


## === cell 24
train_dataloader2 = DataLoader(train2_dataset,64,shuffle=True)
val_dataloader2 = DataLoader(val2_dataset,64)


## === cell 25
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)


## === cell 26
class Trainer:
    
    @torch.no_grad()
    def _evaluate(self,model, val_loader):
        model.eval()
        outputs = [model.validation_step(batch) for batch in val_loader]
        return model.validation_epoch_end(outputs)

  
    def fit(self,epochs, lr, model, train_loader, val_loader, opt_func):
    
        history = []
        optimizer = opt_func(model.parameters(),lr,weight_decay=2e-5)
        for epoch in tqdm(range(epochs)):
        
            model.train()
            train_losses = []
            for batch in train_loader:
                loss = model.training_step(batch)
                train_losses.append(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            result = self._evaluate(model, val_loader)
            result['Train_loss'] = torch.stack(train_losses).mean().item()
            model.epoch_end(epoch, result)
            history.append(result)
    
        return history


## === cell 27
num_epochs=100
lr=1e-4
opt_func= torch.optim.Adam


## === cell 28
tariner=Trainer()
history_EC1 = tariner.fit(num_epochs, lr, model_EC1, train_dataloader, val_dataloader, opt_func)


## === cell 29
def plot_accuracies(history_EC1):
    """ Plot the history of accuracies"""
    accuracies = [x['Validation_acc'] for x in history_EC1]
    plt.plot(accuracies, '-x')
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    plt.title('EC1 Accuracy vs. No. of epochs');
    
plot_accuracies(history_EC1)


## === cell 30
def plot_losses(history_EC1):
    """ Plot the losses in each epoch"""
    train_losses = [x.get('Train_loss') for x in history_EC1]
    val_losses = [x['Validation_loss'] for x in history_EC1]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('EC1 Loss vs. No. of Epochs')

plot_losses(history_EC1)


## === cell 31
fpr,tpr =[],[]
for i in range(len(false_pos)):
        fpr.append(np.mean(false_pos[i].tolist()))

for j in range(len(true_pos)):
        tpr.append(np.mean(false_pos[j].tolist()))


## === cell 32
with torch.no_grad():
    plt.plot(fpr,tpr) # ROC curve = TPR vs FPR
    plt.title("Receiver Operating Characteristics")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()


## === cell 33
history_EC2 = tariner.fit(num_epochs, lr, model_EC2, train_dataloader2, val_dataloader2, opt_func)


## === cell 34
def plot_accuracies(history_EC2):
    """ Plot the history of accuracies"""
    accuracies = [x['Validation_acc'] for x in history_EC2]
    plt.plot(accuracies, '-x')
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    plt.title('EC2 Accuracy vs. No. of epochs');
    
plot_accuracies(history_EC2)


## === cell 35
def plot_losses(history_EC2):
    """ Plot the losses in each epoch"""
    train_losses = [x.get('Train_loss') for x in history_EC2]
    val_losses = [x['Validation_loss'] for x in history_EC2]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('EC2 Loss vs. No. of Epochs')

plot_losses(history_EC2)


## === cell 36
fpr,tpr =[],[]
for i in range(len(false_pos)):
        fpr.append(np.mean(false_pos[i].tolist()))

for j in range(len(true_pos)):
        tpr.append(np.mean(false_pos[j].tolist()))


## === cell 37
with torch.no_grad():
    plt.plot(fpr, tpr) # ROC curve = TPR vs FPR
    plt.title("Receiver Operating Characteristics")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()


## === cell 38
test=data.get_dataframe('test.csv')


## === cell 39
data.summary('test',test)


## === cell 41
test_update = test.loc[:, test.columns != 'id']
std_X_test = data.standardization_data(test_update)
std_X_test[0]


## === cell 42
X_tensor_test = tenops.convert_to_test_tensor(std_X_test)
X_tensor_test[0]


## === cell 43
class CustomDataTest(Dataset):
    def __init__(self, X_data):
        self.X_data = X_data

        
    def __getitem__(self, index):
            return self.X_data[index]
        
    def __len__ (self):
        return len(self.X_data)


## === cell 44
test_dataset = CustomDataTest(X_tensor_test)
test_dataloader = DataLoader(test_dataset,64)


## === cell 45
class Evaluate:
        
    def eval_test_data(self,model,test_data_dl):
        age_target = []
        model.eval()
        with torch.no_grad():
            for X_batch_test in test_data_dl:
                X_batch_test = X_batch_test.to(device)
                y_test_pred = model(X_batch_test)
                y_pred_tag = torch.sigmoid(y_test_pred)
                age_target.append(y_pred_tag.cpu())
        return [a.squeeze().tolist() for a in age_target]
    
    
eva = Evaluate()


## === cell 46
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


## --- ERROR in cell 46, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mRuntimeError[0m                              Traceback (most recent call last)
[0;32m/tmp/ipykernel_12/1880297972.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      8[0m             [0;32myield[0m [0mitem[0m[0;34m[0m[0;34m[0m[0m
[1;32m      9[0m [0;34m[0m[0m
[0;32m---> 10[0;31m [0mec1[0m [0;34m=[0m [0meva[0m[0;34m.[0m[0meval_test_data[0m[0;34m([0m[0mmodel_EC1[0m[0;34m,[0m[0mtest_dataloader[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     11[0m [0mec2[0m [0;34m=[0m [0meva[0m[0;34m.[0m[0meval_test_data[0m[0;34m([0m[0mmodel_EC2[0m[0;34m,[0m[0mtest_dataloader[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_12/2716177355.py[0m in [0;36meval_test_data[0;34m(self, model, test_data_dl)[0m
[1;32m      7[0m             [0;32mfor[0m [0mX_batch_test[0m [0;32min[0m [0mtest_data_dl[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m      8[0m                 [0mX_batch_test[0m [0;34m=[0m [0mX_batch_test[0m[0;34m.[0m[0mto[0m[0;34m([0m[0mdevice[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 9[0;31m                 [0my_test_pred[0m [0;34m=[0m [0mmodel[0m[0;34m([0m[0mX_batch_test[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     10[0m                 [0my_pred_tag[0m [0;34m=[0m [0mtorch[0m[0;34m.[0m[0msigmoid[0m[0;34m([0m[0my_test_pred[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     11[0m                 [0mage_target[0m[0;34m.[0m[0mappend[0m[0;34m([0m[0my_pred_tag[0m[0;34m.[0m[0mcpu[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py[0m in [0;36m_wrapped_call_impl[0;34m(self, *args, **kwargs)[0m
[1;32m   1737[0m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_compiled_call_impl[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m  [0;31m# type: ignore[misc][0m[0;34m[0m[0;34m[0m[0m
[1;32m   1738[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1739[0;31m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_call_impl[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1740[0m [0;34m[0m[0m
[1;32m   1741[0m     [0;31m# torchrec tests the code consistency with the following code[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py[0m in [0;36m_call_impl[0;34m(self, *args, **kwargs)[0m
[1;32m   1748[0m                 [0;32mor[0m [0m_global_backward_pre_hooks[0m [0;32mor[0m [0m_global_backward_hooks[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1749[0m                 or _global_forward_hooks or _global_forward_pre_hooks):
[0;32m-> 1750[0;31m             [0;32mreturn[0m [0mforward_call[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1751[0m [0;34m[0m[0m
[1;32m   1752[0m         [0mresult[0m [0;34m=[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_12/34986389.py[0m in [0;36mforward[0;34m(self, xb)[0m
[1;32m     21[0m [0;34m[0m[0m
[1;32m     22[0m     [0;32mdef[0m [0mforward[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mxb[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 23[0;31m         [0;32mreturn[0m [0mself[0m[0;34m.[0m[0mnetwork[0m[0;34m([0m[0mxb[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     24[0m [0;34m[0m[0m
[1;32m     25[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py[0m in [0;36m_wrapped_call_impl[0;34m(self, *args, **kwargs)[0m
[1;32m   1737[0m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_compiled_call_impl[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m  [0;31m# type: ignore[misc][0m[0;34m[0m[0;34m[0m[0m
[1;32m   1738[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1739[0;31m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_call_impl[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1740[0m [0;34m[0m[0m
[1;32m   1741[0m     [0;31m# torchrec tests the code consistency with the following code[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py[0m in [0;36m_call_impl[0;34m(self, *args, **kwargs)[0m
[1;32m   1748[0m                 [0;32mor[0m [0m_global_backward_pre_hooks[0m [0;32mor[0m [0m_global_backward_hooks[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1749[0m                 or _global_forward_hooks or _global_forward_pre_hooks):
[0;32m-> 1750[0;31m             [0;32mreturn[0m [0mforward_call[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1751[0m [0;34m[0m[0m
[1;32m   1752[0m         [0mresult[0m [0;34m=[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/container.py[0m in [0;36mforward[0;34m(self, input)[0m
[1;32m    248[0m     [0;32mdef[0m [0mforward[0m[0;34m([0m[0mself[0m[0;34m,[0m [0minput[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    249[0m         [0;32mfor[0m [0mmodule[0m [0;32min[0m [0mself[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 250[0;31m             [0minput[0m [0;34m=[0m [0mmodule[0m[0;34m([0m[0minput[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    251[0m         [0;32mreturn[0m [0minput[0m[0;34m[0m[0;34m[0m[0m
[1;32m    252[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py[0m in [0;36m_wrapped_call_impl[0;34m(self, *args, **kwargs)[0m
[1;32m   1737[0m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_compiled_call_impl[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m  [0;31m# type: ignore[misc][0m[0;34m[0m[0;34m[0m[0m
[1;32m   1738[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1739[0;31m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_call_impl[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1740[0m [0;34m[0m[0m
[1;32m   1741[0m     [0;31m# torchrec tests the code consistency with the following code[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py[0m in [0;36m_call_impl[0;34m(self, *args, **kwargs)[0m
[1;32m   1748[0m                 [0;32mor[0m [0m_global_backward_pre_hooks[0m [0;32mor[0m [0m_global_backward_hooks[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1749[0m                 or _global_forward_hooks or _global_forward_pre_hooks):
[0;32m-> 1750[0;31m             [0;32mreturn[0m [0mforward_call[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1751[0m [0;34m[0m[0m
[1;32m   1752[0m         [0mresult[0m [0;34m=[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/linear.py[0m in [0;36mforward[0;34m(self, input)[0m
[1;32m    123[0m [0;34m[0m[0m
[1;32m    124[0m     [0;32mdef[0m [0mforward[0m[0;34m([0m[0mself[0m[0;34m,[0m [0minput[0m[0;34m:[0m [0mTensor[0m[0;34m)[0m [0;34m->[0m [0mTensor[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 125[0;31m         [0;32mreturn[0m [0mF[0m[0;34m.[0m[0mlinear[0m[0;34m([0m[0minput[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0mweight[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0mbias[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    126[0m [0;34m[0m[0m
[1;32m    127[0m     [0;32mdef[0m [0mextra_repr[0m[0;34m([0m[0mself[0m[0;34m)[0m [0;34m->[0m [0mstr[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;31mRuntimeError[0m: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0! (when checking argument for argument mat1 in method wrapper_CUDA_addmm)

## === cell 47
class Submit:
    
    def submit_predictions(self):        
        df_submit = pd.DataFrame(data={'id': test['id'],'EC1':  list(flatten(ec1)),'EC2':list(flatten(ec2))})
        df_submit.to_csv('submission.csv',index=False)
        print('Submission Completed!!')
        return df_submit
        
        
submit = Submit()
df_submit=submit.submit_predictions()
