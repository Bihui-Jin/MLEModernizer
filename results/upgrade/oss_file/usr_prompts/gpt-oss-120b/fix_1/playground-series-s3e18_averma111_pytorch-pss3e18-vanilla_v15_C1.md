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

0.64272

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

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
import random
import numpy as np 
import pandas as pd 
import os
import datetime
import seaborn as sns
from tqdm.notebook import tqdm
from ydata_profiling import ProfileReport
from collections import Counter

from imblearn.over_sampling import SMOTE,SMOTEN
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve

from torchsampler import ImbalancedDatasetSampler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
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


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/2936491713.py in <cell line: 0>()
      9 from collections import Counter
     10 
---> 11 from imblearn.over_sampling import SMOTE,SMOTEN
     12 from sklearn.model_selection import KFold
     13 from sklearn.model_selection import train_test_split

/usr/local/lib/python3.11/dist-packages/imblearn/__init__.py in <module>
     50     # process, as it may not be compiled yet
     51 else:
---> 52     from . import (
     53         combine,
     54         ensemble,

/usr/local/lib/python3.11/dist-packages/imblearn/combine/__init__.py in <module>
      3 """
      4 
----> 5 from ._smote_enn import SMOTEENN
      6 from ._smote_tomek import SMOTETomek
      7 

/usr/local/lib/python3.11/dist-packages/imblearn/combine/_smote_enn.py in <module>
     10 from sklearn.utils import check_X_y
     11 
---> 12 from ..base import BaseSampler
     13 from ..over_sampling import SMOTE
     14 from ..over_sampling.base import BaseOverSampler

/usr/local/lib/python3.11/dist-packages/imblearn/base.py in <module>
     10 from sklearn.base import BaseEstimator, OneToOneFeatureMixin
     11 from sklearn.preprocessing import label_binarize
---> 12 from sklearn.utils._metadata_requests import METHODS
     13 from sklearn.utils.multiclass import check_classification_targets
     14 

ModuleNotFoundError: No module named 'sklearn.utils._metadata_requests'

## === cell 4
for dirname, _, filenames in os.walk('/kaggle/input/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


## === cell 5
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


## === cell 6
data.summary('train',train)


## === cell 8
y1 = train['EC1']
y2 = train['EC2']
train.drop(columns=['id','EC1','EC2','EC3','EC4','EC5','EC6'],axis=1,inplace=True)
X = train.copy()


## === cell 9
X_train,X_val,y1_train,y1_val = data.random_split_data(X,y1)
print('Data splits for EC1 \n',X_train.shape,X_val.shape,y1_train.shape,y1_val.shape)


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/94059866.py in <cell line: 0>()
----> 1 X_train,X_val,y1_train,y1_val = data.random_split_data(X,y1)
      2 print('Data splits for EC1 \n',X_train.shape,X_val.shape,y1_train.shape,y1_val.shape)

/tmp/ipykernel_11/2904229419.py in random_split_data(self, X, y)
     21 
     22     def random_split_data(self,X,y):
---> 23         return train_test_split(X, y,test_size=0.20,random_state=42)
     24 
     25 

NameError: name 'train_test_split' is not defined

## === cell 10
X_train,X_val,y2_train,y2_val = data.random_split_data(X,y2)
print('Data splits for EC2 \n',X_train.shape,X_val.shape,y2_train.shape,y2_val.shape)


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1322148836.py in <cell line: 0>()
----> 1 X_train,X_val,y2_train,y2_val = data.random_split_data(X,y2)
      2 print('Data splits for EC2 \n',X_train.shape,X_val.shape,y2_train.shape,y2_val.shape)

/tmp/ipykernel_11/2904229419.py in random_split_data(self, X, y)
     21 
     22     def random_split_data(self,X,y):
---> 23         return train_test_split(X, y,test_size=0.20,random_state=42)
     24 
     25 

NameError: name 'train_test_split' is not defined

## === cell 11
std_X_train = data.standardization_data(X_train)
std_X_val = data.standardization_data(X_val)
print(std_X_train[0],std_X_val[0])


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1066238445.py in <cell line: 0>()
----> 1 std_X_train = data.standardization_data(X_train)
      2 std_X_val = data.standardization_data(X_val)
      3 print(std_X_train[0],std_X_val[0])

NameError: name 'X_train' is not defined

## === cell 12
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


## === cell 13
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


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4074897332.py in <cell line: 0>()
----> 1 class CustomDataset(Dataset):
      2 
      3     def __init__(self,X_data,y_data=None,is_train=True):
      4         super().__init__()
      5         if is_train:

NameError: name 'Dataset' is not defined

## === cell 14
X_tensor_train,y1_tensor_train = tenops.convert_to_tensor(std_X_train,y1_train.values)
X_tensor_val,y1_tensor_val = tenops.convert_to_tensor(std_X_val,y1_val.values)
print('The training tensor for EC1\n',X_tensor_train,y1_tensor_train)
print('The validation tensor for EC1\n',X_tensor_val,y1_tensor_val)


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/501646945.py in <cell line: 0>()
----> 1 X_tensor_train,y1_tensor_train = tenops.convert_to_tensor(std_X_train,y1_train.values)
      2 X_tensor_val,y1_tensor_val = tenops.convert_to_tensor(std_X_val,y1_val.values)
      3 print('The training tensor for EC1\n',X_tensor_train,y1_tensor_train)
      4 print('The validation tensor for EC1\n',X_tensor_val,y1_tensor_val)

NameError: name 'std_X_train' is not defined

## === cell 15
X_tensor_train,y2_tensor_train = tenops.convert_to_tensor(std_X_train,y2_train.values)
X_tensor_val,y2_tensor_val = tenops.convert_to_tensor(std_X_val,y2_val.values)
print('The training tensor EC2\n',X_tensor_train,y2_tensor_train)
print('The validation tensor Ec2\n',X_tensor_val,y2_tensor_val)


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3236619678.py in <cell line: 0>()
----> 1 X_tensor_train,y2_tensor_train = tenops.convert_to_tensor(std_X_train,y2_train.values)
      2 X_tensor_val,y2_tensor_val = tenops.convert_to_tensor(std_X_val,y2_val.values)
      3 print('The training tensor EC2\n',X_tensor_train,y2_tensor_train)
      4 print('The validation tensor Ec2\n',X_tensor_val,y2_tensor_val)

NameError: name 'std_X_train' is not defined

## === cell 16
false_pos,true_pos = [],[]


## === cell 17
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


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2092293466.py in <cell line: 0>()
----> 1 class EnzymeClassificationBase(torch.nn.Module):
      2 
      3     def _accuracy(self,outputs, labels):
      4         #return torch.tensor(outputs.round() == labels).float().mean()
      5         metric = BinaryAccuracy()

NameError: name 'torch' is not defined

## === cell 18
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
            torch.nn.Linear(8,n_output),
            torch.nn.Sigmoid()
        )
    
    def forward(self, xb):
        return self.network(xb)


model_EC1 = MultiClassificationNN()
model_EC2 = MultiClassificationNN()


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1804402915.py in <cell line: 0>()
----> 1 n_input_dim = X_train.shape[1]
      2 n_output =  1   # Number of output nodes = for binary classifier
      3 
      4 class MultiClassificationNN(EnzymeClassificationBase):
      5 

NameError: name 'X_train' is not defined

## === cell 19
print(model_EC1)


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1277040629.py in <cell line: 0>()
----> 1 print(model_EC1)

NameError: name 'model_EC1' is not defined

## === cell 20
print(model_EC2)


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/320784261.py in <cell line: 0>()
----> 1 print(model_EC2)

NameError: name 'model_EC2' is not defined

## === cell 21
train1_dataset = CustomDataset(X_tensor_train,y1_tensor_train)
val1_dataset = CustomDataset(X_tensor_val,y1_tensor_val)

train2_dataset = CustomDataset(X_tensor_train,y2_tensor_train)
val2_dataset = CustomDataset(X_tensor_val,y2_tensor_val)


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1061073250.py in <cell line: 0>()
----> 1 train1_dataset = CustomDataset(X_tensor_train,y1_tensor_train)
      2 val1_dataset = CustomDataset(X_tensor_val,y1_tensor_val)
      3 
      4 train2_dataset = CustomDataset(X_tensor_train,y2_tensor_train)
      5 val2_dataset = CustomDataset(X_tensor_val,y2_tensor_val)

NameError: name 'CustomDataset' is not defined

## === cell 22
train_dataloader = DataLoader(train1_dataset,64,shuffle=True)
val_dataloader = DataLoader(val1_dataset,64)


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/709400041.py in <cell line: 0>()
----> 1 train_dataloader = DataLoader(train1_dataset,64,shuffle=True)
      2 val_dataloader = DataLoader(val1_dataset,64)

NameError: name 'DataLoader' is not defined

## === cell 23
train_dataloader2 = DataLoader(train2_dataset,64,shuffle=True)
val_dataloader2 = DataLoader(val2_dataset,64)


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3340902438.py in <cell line: 0>()
----> 1 train_dataloader2 = DataLoader(train2_dataset,64,shuffle=True)
      2 val_dataloader2 = DataLoader(val2_dataset,64)

NameError: name 'DataLoader' is not defined

## === cell 24
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1723146003.py in <cell line: 0>()
----> 1 device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      2 torch.manual_seed(42)

NameError: name 'torch' is not defined

## === cell 25
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


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3707648762.py in <cell line: 0>()
----> 1 class Trainer:
      2 
      3     @torch.no_grad()
      4     def _evaluate(self,model, val_loader):
      5         model.eval()

/tmp/ipykernel_11/3707648762.py in Trainer()
      1 class Trainer:
      2 
----> 3     @torch.no_grad()
      4     def _evaluate(self,model, val_loader):
      5         model.eval()

NameError: name 'torch' is not defined

## === cell 26
num_epochs=100
lr=1e-4
opt_func= torch.optim.Adam


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2655760665.py in <cell line: 0>()
      1 num_epochs=100
      2 lr=1e-4
----> 3 opt_func= torch.optim.Adam

NameError: name 'torch' is not defined

## === cell 27
tariner=Trainer()
history_EC1 = tariner.fit(num_epochs, lr, model_EC1, train_dataloader, val_dataloader, opt_func)


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/617061179.py in <cell line: 0>()
----> 1 tariner=Trainer()
      2 history_EC1 = tariner.fit(num_epochs, lr, model_EC1, train_dataloader, val_dataloader, opt_func)

NameError: name 'Trainer' is not defined

## === cell 28
def plot_accuracies(history_EC1):
    """ Plot the history of accuracies"""
    accuracies = [x['Validation_acc'] for x in history_EC1]
    plt.plot(accuracies, '-x')
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    plt.title('EC1 Accuracy vs. No. of epochs');
    
plot_accuracies(history_EC1)


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3662779122.py in <cell line: 0>()
      7     plt.title('EC1 Accuracy vs. No. of epochs');
      8 
----> 9 plot_accuracies(history_EC1)

NameError: name 'history_EC1' is not defined

## === cell 29
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


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1742559759.py in <cell line: 0>()
     10     plt.title('EC1 Loss vs. No. of Epochs')
     11 
---> 12 plot_losses(history_EC1)

NameError: name 'history_EC1' is not defined

## === cell 30
fpr,tpr =[],[]
for i in range(len(false_pos)):
        fpr.append(np.mean(false_pos[i].tolist()))

for j in range(len(true_pos)):
        tpr.append(np.mean(false_pos[j].tolist()))


## === cell 31
with torch.no_grad():
    plt.plot(fpr,tpr) # ROC curve = TPR vs FPR
    plt.title("Receiver Operating Characteristics")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()


## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/616036740.py in <cell line: 0>()
----> 1 with torch.no_grad():
      2     plt.plot(fpr,tpr) # ROC curve = TPR vs FPR
      3     plt.title("Receiver Operating Characteristics")
      4     plt.xlabel("False Positive Rate")
      5     plt.ylabel("True Positive Rate")

NameError: name 'torch' is not defined

## === cell 32
history_EC2 = tariner.fit(num_epochs, lr, model_EC2, train_dataloader2, val_dataloader2, opt_func)


## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2863001020.py in <cell line: 0>()
----> 1 history_EC2 = tariner.fit(num_epochs, lr, model_EC2, train_dataloader2, val_dataloader2, opt_func)

NameError: name 'tariner' is not defined

## === cell 33
def plot_accuracies(history_EC2):
    """ Plot the history of accuracies"""
    accuracies = [x['Validation_acc'] for x in history_EC2]
    plt.plot(accuracies, '-x')
    plt.xlabel('Epoch')
    plt.ylabel('accuracy')
    plt.title('EC2 Accuracy vs. No. of epochs');
    
plot_accuracies(history_EC2)


## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3913821715.py in <cell line: 0>()
      7     plt.title('EC2 Accuracy vs. No. of epochs');
      8 
----> 9 plot_accuracies(history_EC2)

NameError: name 'history_EC2' is not defined

## === cell 34
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


## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1872981222.py in <cell line: 0>()
     10     plt.title('EC2 Loss vs. No. of Epochs')
     11 
---> 12 plot_losses(history_EC2)

NameError: name 'history_EC2' is not defined

## === cell 35
fpr,tpr =[],[]
for i in range(len(false_pos)):
        fpr.append(np.mean(false_pos[i].tolist()))

for j in range(len(true_pos)):
        tpr.append(np.mean(false_pos[j].tolist()))


## === cell 36
with torch.no_grad():
    plt.plot(fpr, tpr) # ROC curve = TPR vs FPR
    plt.title("Receiver Operating Characteristics")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()


## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3089573012.py in <cell line: 0>()
----> 1 with torch.no_grad():
      2     plt.plot(fpr, tpr) # ROC curve = TPR vs FPR
      3     plt.title("Receiver Operating Characteristics")
      4     plt.xlabel("False Positive Rate")
      5     plt.ylabel("True Positive Rate")

NameError: name 'torch' is not defined

## === cell 37
test=data.get_dataframe('test.csv')


## === cell 38
data.summary('test',test)


## === cell 40
test_update = test.loc[:, test.columns != 'id']
std_X_test = data.standardization_data(test_update)
std_X_test[0]


## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3126544239.py in <cell line: 0>()
      1 test_update = test.loc[:, test.columns != 'id']
----> 2 std_X_test = data.standardization_data(test_update)
      3 std_X_test[0]

/tmp/ipykernel_11/2904229419.py in standardization_data(self, X_data)
     25 
     26     def standardization_data(self,X_data):
---> 27         scaler = StandardScaler()
     28         std_X_data = scaler.fit_transform(X_data)
     29         return std_X_data

NameError: name 'StandardScaler' is not defined

## === cell 41
X_tensor_test = tenops.convert_to_test_tensor(std_X_test)
X_tensor_test[0]


## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1294000201.py in <cell line: 0>()
----> 1 X_tensor_test = tenops.convert_to_test_tensor(std_X_test)
      2 X_tensor_test[0]

NameError: name 'std_X_test' is not defined

## === cell 42
class CustomDataTest(Dataset):
    def __init__(self, X_data):
        self.X_data = X_data

        
    def __getitem__(self, index):
            return self.X_data[index]
        
    def __len__ (self):
        return len(self.X_data)


## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2295821276.py in <cell line: 0>()
----> 1 class CustomDataTest(Dataset):
      2     def __init__(self, X_data):
      3         self.X_data = X_data
      4 
      5 

NameError: name 'Dataset' is not defined

## === cell 43
test_dataset = CustomDataTest(X_tensor_test)
test_dataloader = DataLoader(test_dataset,64)


## --- ERROR in cell 43, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1918755712.py in <cell line: 0>()
----> 1 test_dataset = CustomDataTest(X_tensor_test)
      2 test_dataloader = DataLoader(test_dataset,64)

NameError: name 'CustomDataTest' is not defined

## === cell 44
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


## === cell 45
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


## --- ERROR in cell 45, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1880297972.py in <cell line: 0>()
      8             yield item
      9 
---> 10 ec1 = eva.eval_test_data(model_EC1,test_dataloader)
     11 ec2 = eva.eval_test_data(model_EC2,test_dataloader)

NameError: name 'model_EC1' is not defined

## === cell 46
class Submit:
    
    def submit_predictions(self):        
        df_submit = pd.DataFrame(data={'id': test['id'],'EC1':  list(flatten(ec1)),'EC2':list(flatten(ec2))})
        df_submit.to_csv('submission.csv',index=False)
        print('Submission Completed!!')
        return df_submit
        
        
submit = Submit()
df_submit=submit.submit_predictions()


## --- ERROR in cell 46, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4015976665.py in <cell line: 0>()
      9 
     10 submit = Submit()
---> 11 df_submit=submit.submit_predictions()

/tmp/ipykernel_11/4015976665.py in submit_predictions(self)
      2 
      3     def submit_predictions(self):
----> 4         df_submit = pd.DataFrame(data={'id': test['id'],'EC1':  list(flatten(ec1)),'EC2':list(flatten(ec2))})
      5         df_submit.to_csv('submission.csv',index=False)
      6         print('Submission Completed!!')

NameError: name 'ec1' is not defined

## === cell 47
df_submit


## --- ERROR in cell 47, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3093292827.py in <cell line: 0>()
----> 1 df_submit

NameError: name 'df_submit' is not defined
