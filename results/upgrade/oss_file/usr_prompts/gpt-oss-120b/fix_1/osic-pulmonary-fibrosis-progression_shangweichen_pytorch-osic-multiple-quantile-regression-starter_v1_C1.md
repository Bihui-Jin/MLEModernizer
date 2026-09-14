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
Predict a patient’s severity of decline in lung function based on a CT scan of their lungs. Lung function is assessed based on output from a spirometer, which measures the forced vital capacity (`FVC`), i.e. the volume of air exhaled.

## Metric
A modified version of the Laplace Log Likelihood. 

For each true FVC measurement, you will predict both an FVC and a confidence measure (standard deviation 𝜎𝜎). The metric is computed as:

$$
\begin{gathered}
\sigma_{\text {clipped }}=\max (\sigma, 70), \\
\Delta=\min \left(\left|F V C_{\text {true }}-F V C_{\text {predicted }}\right|, 1000\right), \\
\text { metric }=-\frac{\sqrt{2} \Delta}{\sigma_{\text {clipped }}}-\ln \left(\sqrt{2} \sigma_{\text {clipped }}\right) .
\end{gathered}
$$

The error is thresholded at 1000 ml to avoid large errors adversely penalizing results, while the confidence values are clipped at 70 ml to reflect the approximate measurement uncertainty in FVC. The final score is calculated by averaging the metric across all test set `Patient_Week`s (three per patient). 

Metric values will be negative and higher is better.

## Submission Format
For each `Patient_Week`, you must predict the `FVC` and a confidence. You are asked to predict every patient's `FVC` measurement for every possible week. Those weeks which are not in the final three visits are ignored in scoring.

The file should contain a header and have the following format:

```
Patient_Week,FVC,Confidence
ID00002637202176704235138_1,2000,100
ID00002637202176704235138_2,2000,100
ID00002637202176704235138_3,2000,100
etc.

```

## Dataset
In the dataset, you are provided with a baseline chest CT scan and associated clinical information for a set of patients. A patient has an image acquired at time `Week = 0` and has numerous follow up visits over the course of approximately 1-2 years, at which time their `FVC` is measured.

- In the training set, you are provided with an anonymized, baseline CT scan and the entire history of FVC measurements.
- In the test set, you are provided with a baseline CT scan and only the initial FVC measurement. **You are asked to predict the final three `FVC` measurements for each patient, as well as a confidence value in your prediction.**

- **train.csv** - the training set, contains full history of clinical information
- **test.csv** - the test set, contains only the baseline measurement
- **train/** - contains the training patients' baseline CT scan in DICOM format
- **test/** - contains the test patients' baseline CT scan in DICOM format
- **sample_submission.csv** - demonstrates the submission format

**train.csv and test.csv**

- `Patient`a unique Id for each patient (also the name of the patient's DICOM folder)
- `Weeks`the relative number of weeks pre/post the baseline CT (may be negative)
- `FVC` - the recorded lung capacity in ml
- `Percent`a computed field which approximates the patient's FVC as a percent of the typical FVC for a person of similar characteristics
- `Age`
- `Sex`
- `SmokingStatus`

**sample submission.csv**

- `Patient_Week` - a unique Id formed by concatenating the `Patient` and `Weeks` columns (i.e. ABC_22 is a prediction for patient ABC at week 22)
- `FVC` - the predicted FVC in ml
- `Confidence` - a confidence value of your prediction (also has units of ml)

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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
            description.md (122 lines)
            sample_submission.csv (1909 lines)
            sample_submission.csv.zip (5.7 kB)
            test.csv (19 lines)
            test.csv.zip (748 Bytes)
            test.zip (1.2 GB)
            train.csv (1395 lines)
            train.csv.zip (23.6 kB)
            train.zip (12.7 GB)
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
            test/
                ID00014637202177757139317/
                    1.dcm (1.5 MB)
                    10.dcm (1.5 MB)
                    ... and 29 other files
                ID00019637202178323708467/
                    1.dcm (525.5 kB)
                    10.dcm (525.5 kB)
                    ... and 27 other files
                ... and 17 other folders
            train/
                ID00007637202177411956430/
                    1.dcm (525.6 kB)
                    10.dcm (525.6 kB)
                    ... and 28 other files
                ID00009637202177434476278/
                    1.dcm (1.2 MB)
                    10.dcm (1.2 MB)
                    ... and 392 other files
                ... and 157 other folders
        input/
            description.md (122 lines)
            sample_submission.csv (1909 lines)
            sample_submission.csv.zip (5.7 kB)
            test.csv (19 lines)
            test.csv.zip (748 Bytes)
            test.zip (1.2 GB)
            train.csv (1395 lines)
            train.csv.zip (23.6 kB)
            train.zip (12.7 GB)
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
            test/
                ID00014637202177757139317/
                    1.dcm (1.5 MB)
                    10.dcm (1.5 MB)
                    ... and 29 other files
                ID00019637202178323708467/
                    1.dcm (525.5 kB)
                    10.dcm (525.5 kB)
                    ... and 27 other files
                ... and 17 other folders
            train/
                ID00007637202177411956430/
                    1.dcm (525.6 kB)
                    10.dcm (525.6 kB)
                    ... and 28 other files
                ID00009637202177434476278/
                    1.dcm (1.2 MB)
                    10.dcm (1.2 MB)
                    ... and 392 other files
                ... and 157 other folders
        working/
            osic-pulmonary-fibrosis-progression/
                description.md (122 lines)
                sample_submission.csv (1909 lines)
                ... and 7 other files
                osic-pulmonary-fibrosis-progression/
                test/
                    ID00014637202177757139317/
                        1.dcm (1.5 MB)
                        10.dcm (1.5 MB)
                        ... and 29 other files
                    ID00019637202178323708467/
                        1.dcm (525.5 kB)
                        10.dcm (525.5 kB)
                        ... and 27 other files
                    ... and 17 other folders
                train/
                    ID00007637202177411956430/
                        1.dcm (525.6 kB)
                        10.dcm (525.6 kB)
                        ... and 28 other files
                    ID00009637202177434476278/
                        1.dcm (1.2 MB)
                        10.dcm (1.2 MB)
                        ... and 392 other files
                    ... and 157 other folders
```

-> data/osic-pulmonary-fibrosis-progression/sample_submission.csv has 1908 rows and 3 columns.
The columns are: Patient_Week, FVC, Confidence

-> data/osic-pulmonary-fibrosis-progression/test.csv has 18 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/osic-pulmonary-fibrosis-progression/train.csv has 1394 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/sample_submission.csv has 1908 rows and 3 columns.
The columns are: Patient_Week, FVC, Confidence

-> data/test.csv has 18 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> data/train.csv has 1394 rows and 7 columns.
The columns are: Patient, Weeks, FVC, Percent, Age, Sex, SmokingStatus

-> (stopped after 10 files for performance)

# 5. Target score

-6.9708

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
import gc
import os
import math
import random
from tqdm import tqdm
from IPython.core.interactiveshell import InteractiveShell
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torch.optim import Adam, AdamW
%matplotlib inline

warnings.filterwarnings("ignore")
InteractiveShell.ast_node_interactivity = "all"


## === cell 1
BATCH_SIZE = 128
EPOCHS = 800
LR = 1e-1
FOLDER = 5
SEED = 42
DEVICE = 'cuda'
QUANTILE = [0.2, 0.5, 0.8]
SAVE_AND_LOAD_BEST_MODEL = True    # You can turn on this for higher score
LR_Scheduler = False               # You can turn on this for higher score

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)


## === cell 2
ROOT = "../input/osic-pulmonary-fibrosis-progression"

tr = pd.read_csv(f"{ROOT}/train.csv")
tr.drop_duplicates(keep=False, inplace=True, subset=['Patient','Weeks'])
chunk = pd.read_csv(f"{ROOT}/test.csv")

sub = pd.read_csv(f"{ROOT}/sample_submission.csv")
sub['Patient'] = sub['Patient_Week'].apply(lambda x:x.split('_')[0])
sub['Weeks'] = sub['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))
sub =  sub[['Patient','Weeks','Confidence','Patient_Week']]
sub = sub.merge(chunk.drop('Weeks', axis=1), on="Patient", how='left')

tr['WHERE'] = 'train'
chunk['WHERE'] = 'val'
sub['WHERE'] = 'test'
data = tr.append([chunk, sub])

print("Origin Shape: ")
print(tr.shape, chunk.shape, sub.shape, data.shape)
print(tr.Patient.nunique(), chunk.Patient.nunique(), sub.Patient.nunique(), data.Patient.nunique())

data['min_week'] = data['Weeks']
data.loc[data.WHERE=='test','min_week'] = np.nan
data['min_week'] = data.groupby('Patient')['min_week'].transform('min')

base = data.loc[data.Weeks == data.min_week]
base = base[['Patient','FVC']].copy()
base.columns = ['Patient','min_FVC']
base['nb'] = 1
base['nb'] = base.groupby('Patient')['nb'].transform('cumsum')
base = base[base.nb==1]
base.drop('nb', axis=1, inplace=True)

data = data.merge(base, on='Patient', how='left')
data['base_week'] = data['Weeks'] - data['min_week']
del base

COLS = ['Sex','SmokingStatus']
FE = []
for col in COLS:
    for mod in data[col].unique():
        FE.append(mod)
        data[mod] = (data[col] == mod).astype(int)

data['age'] = (data['Age'] - data['Age'].min() ) / ( data['Age'].max() - data['Age'].min() )
data['BASE'] = (data['min_FVC'] - data['min_FVC'].min() ) / ( data['min_FVC'].max() - data['min_FVC'].min() )
data['week'] = (data['base_week'] - data['base_week'].min() ) / ( data['base_week'].max() - data['base_week'].min() )
data['percent'] = (data['Percent'] - data['Percent'].min() ) / ( data['Percent'].max() - data['Percent'].min() )
FE += ['age','percent','week','BASE']
INPUT_FEATURES = len(FE)
print("\nFE: ", FE, 
      "\nINPUT_FEATURES: ", INPUT_FEATURES)

tr = data.loc[data.WHERE=='train']
chunk = data.loc[data.WHERE=='val']
sub = data.loc[data.WHERE=='test']
del data, chunk

X_train = tr[FE].values
Y_train = tr['FVC'].values
X_test = sub[FE].values
print("\nAfter Shape: ")
print(X_train.shape, Y_train.shape, X_test.shape)
gc.collect()


## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3126382469.py in <cell line: 0>()
     14 chunk['WHERE'] = 'val'
     15 sub['WHERE'] = 'test'
---> 16 data = tr.append([chunk, sub])
     17 
     18 print("Origin Shape: ")

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'append'

## === cell 3
class DatasetRetriever(Dataset):
    def __init__(self, x_data, y_data=None):
        self.x_data = x_data
        self.y_data = y_data
    
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        if self.y_data is not None:
            return torch.tensor(self.x_data[idx]), torch.tensor(self.y_data[idx])
        else:
            return torch.tensor(self.x_data[idx])


class QuantileRegression(nn.Module):
    def __init__(self, input_features=INPUT_FEATURES):
        super(QuantileRegression, self).__init__()
        
        self.nn1 = nn.Linear(input_features, 100)
        self.nn2 = nn.Linear(100, 100)
        self.nn3_1 = nn.Linear(100, 3)
        self.nn3_2 = nn.Linear(100, 3)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.constant_(self.nn1.bias, 0)
        torch.nn.init.xavier_uniform_(self.nn2.weight)
        torch.nn.init.constant_(self.nn2.bias, 0)
        torch.nn.init.xavier_uniform_(self.nn3_1.weight)
        torch.nn.init.constant_(self.nn3_1.bias, 0)
        torch.nn.init.xavier_uniform_(self.nn3_2.weight)
        torch.nn.init.constant_(self.nn3_2.bias, 0)
    
    def forward(self, inputs):
        X = F.relu(self.nn1(inputs))
        X = F.relu(self.nn2(X))
        X_1 = self.nn3_1(X)
        X_2 = F.relu(self.nn3_2(X))
        output = X_1 + torch.cumsum(X_2, dim=1)
        return output


class LossMeter(object):
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ScoreMeter(object):
    def __init__(self):
        self.sum = 0
        self.count = 0
        self.avg = 0
    
    def compute_score(self, y_pred, y_true):
        sigma = y_pred[:, 2] - y_pred[:, 0]
        fvc_pred = y_pred[:, 1]
        sigma_clip = np.maximum(sigma, 70.0)
        delta = np.minimum(np.abs(y_true[:, 0]-fvc_pred), 1000.0)
        metric = (delta / sigma_clip) * np.sqrt(2.0) + np.log(sigma_clip * np.sqrt(2.0))
        return np.mean(metric)
    
    def update(self, preds, labels):
        batch_size = preds.size(0)
        preds = preds.data.cpu().numpy()
        labels = labels.data.cpu().numpy()
        val = self.compute_score(preds, labels)
        self.sum += (val * batch_size)
        self.count += batch_size
        self.avg = self.sum / self.count


class QuantileRegressionLoss(nn.Module):
    def __init__(self):
        super(QuantileRegressionLoss, self).__init__()
        self.quantile = torch.tensor(QUANTILE).to(DEVICE, dtype=torch.float)

    def forward(self, preds, labels):
        error = labels - preds
        vector = torch.max(self.quantile*error, (self.quantile-1)*error)
        return vector.mean()


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/439389067.py in <cell line: 0>()
     14 
     15 
---> 16 class QuantileRegression(nn.Module):
     17     def __init__(self, input_features=INPUT_FEATURES):
     18         super(QuantileRegression, self).__init__()

/tmp/ipykernel_11/439389067.py in QuantileRegression()
     15 
     16 class QuantileRegression(nn.Module):
---> 17     def __init__(self, input_features=INPUT_FEATURES):
     18         super(QuantileRegression, self).__init__()
     19 

NameError: name 'INPUT_FEATURES' is not defined

## === cell 4
class Fitter:
    def __init__(self, model, device, fold):
        self.model = model
        self.device = device
        self.fold = fold
        self.optimizer = Adam(self.model.parameters(), lr=LR, weight_decay=0.01)  # default weight_decay=0
        self.criterion = QuantileRegressionLoss()
        if LR_Scheduler:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                                                                        mode='min', 
                                                                        factor=0.5, 
                                                                        patience=20,
                                                                        min_lr=1e-4,
                                                                        verbose=True)
        print(f'Fitter prepared. Device is {self.device}')
    
    def fit(self, train_loader, valid_loader):
        min_valid_score = 999
        plot_rec = {"train_loss": [],
                    "train_score":[],
                    "valid_loss": [],
                    "valid_score":[],
                    "best_epoch": 0,
                    "final": [],
                    "best": []
                    }
        for epoch in range(EPOCHS):
            train_loss, train_score = self.train_one_epoch(train_loader)
            plot_rec["train_loss"].append(train_loss)
            plot_rec["train_score"].append(train_score)
            
            valid_loss, valid_score = self.validation(valid_loader)
            plot_rec["valid_loss"].append(valid_loss)
            plot_rec["valid_score"].append(valid_score)
            
            if LR_Scheduler:
                self.scheduler.step(valid_score)
            
            if SAVE_AND_LOAD_BEST_MODEL:
                if valid_score < min_valid_score:
                    min_valid_score = valid_score
                    torch.save(self.model.state_dict(), f"Folder-{self.fold}.bin")
                    plot_rec["best_epoch"] = epoch + 1
                    plot_rec["best"] = [train_loss, train_score, valid_loss, valid_score]
                    print(f'****************************** Epoch {epoch+1} Model Is Best ******************************')
            
        if SAVE_AND_LOAD_BEST_MODEL:
            temp = plot_rec["best"]
            print(f"\nBest: ", 
                  f"\ntrain_loss: {temp[0]:.4f}, train_score: {temp[1]:.4f}, valid_loss: {temp[2]:.4f}, valid_score: {temp[3]:.4f}")
            
        plot_rec["final"] = [train_loss, train_score, valid_loss, valid_score]
        print(f"Final: ", 
              f"\ntrain_loss: {train_loss:.4f}, train_score: {train_score:.4f}, valid_loss: {valid_loss:.4f}, valid_score: {valid_score:.4f}")
        
        return plot_rec
    
    def train_one_epoch(self, train_loader):
        losses = LossMeter()
        scores = ScoreMeter()
        self.model.train()
        for step, (ipt, lbl) in enumerate(train_loader):
            ipt = ipt.to(self.device, dtype=torch.float)
            lbl = lbl.to(self.device, dtype=torch.float).view(-1, 1)
            self.optimizer.zero_grad()
            opt = self.model(ipt)
            loss = self.criterion(opt, lbl)
            losses.update(loss.detach().item(), ipt.size(0))
            scores.update(opt, lbl)
            loss.backward()
            self.optimizer.step()
        return losses.avg, scores.avg
    
    def validation(self, validation_loader):
        losses = LossMeter()
        scores = ScoreMeter()
        self.model.eval()
        for step, (ipt, lbl) in enumerate(validation_loader):
            with torch.no_grad():
                ipt = ipt.to(self.device, dtype=torch.float)
                lbl = lbl.to(self.device, dtype=torch.float).view(-1, 1)
                opt = self.model(ipt)
                loss = self.criterion(opt, lbl)
                losses.update(loss.detach().item(), ipt.size(0))
                scores.update(opt, lbl)
        return losses.avg, scores.avg
    
    def run_inference(self, data_loader):
        self.model.eval()
        temp = np.empty((0, 3))
        for step, ipt in enumerate(data_loader):
            with torch.no_grad():
                ipt = ipt.to(self.device, dtype=torch.float)
                opt = self.model(ipt)
                temp = np.append(temp, opt.cpu().detach().numpy(), axis=0)
        return temp


## === cell 5
def TrainAndPred(x_train, y_train, x_valid, y_valid, x_test, fold):
    device = DEVICE
    model = QuantileRegression()
    model.to(device)
    
    train_data = DatasetRetriever(x_train, y_train)
    train_data_for_pred = DatasetRetriever(x_train)
    valid_data = DatasetRetriever(x_valid, y_valid)
    valid_data_for_pred = DatasetRetriever(x_valid)
    test_data = DatasetRetriever(x_test)
    
    train_data_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        drop_last=False,
        num_workers=0,
        shuffle=True
    )
    
    train_data_loader_for_pred = torch.utils.data.DataLoader(
        train_data_for_pred,
        batch_size=BATCH_SIZE,
        drop_last=False,
        num_workers=0,
        shuffle=False
    )
    
    valid_data_loader = torch.utils.data.DataLoader(
        valid_data,
        batch_size=BATCH_SIZE,
        drop_last=False,
        num_workers=0,
        shuffle=False
    )
    
    valid_data_loader_for_pred = torch.utils.data.DataLoader(
        valid_data_for_pred,
        batch_size=BATCH_SIZE,
        drop_last=False,
        num_workers=0,
        shuffle=False
    )
    
    test_data_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        drop_last=False,
        num_workers=0,
        shuffle=False
    )
    
    fitter = Fitter(model=model, device=device, fold=fold)
    plot_rec = fitter.fit(train_data_loader, valid_data_loader)
    
    if SAVE_AND_LOAD_BEST_MODEL:
        bestModel = QuantileRegression()
        bestModel.load_state_dict(torch.load(f"Folder-{fold}.bin"))
        bestModel.to(device)
        fitter = Fitter(model=bestModel, device=device, fold=fold)
        
    pred_for_train = fitter.run_inference(train_data_loader_for_pred)
    pred_for_valid = fitter.run_inference(valid_data_loader_for_pred)
    pred_for_test = fitter.run_inference(test_data_loader)
    
    return pred_for_train, pred_for_valid, pred_for_test, plot_rec


## === cell 6
for_train = np.zeros((len(X_train), 3))
for_valid = np.zeros((len(X_train), 3))
for_test = np.zeros((len(X_test), 3))

plot_recs = []
split_record = []
kfold = KFold(FOLDER)

for fold, (xx, yy) in enumerate(kfold.split(X_train)):
    split_record.append([fold, xx, yy])

for item in split_record:
    fold, xx, yy = item
    print(f"\n========================================fold:{fold+1}============================================")
    temp_x_train = X_train[xx]
    temp_y_train = Y_train[xx]
    temp_x_valid = X_train[yy]
    temp_y_valid = Y_train[yy]
    print("Shape: ", temp_x_train.shape, temp_y_train.shape, temp_x_valid.shape, temp_y_valid.shape)
    pred_for_train, pred_for_valid, pred_for_test, plot_rec = TrainAndPred(temp_x_train, temp_y_train, temp_x_valid, temp_y_valid, X_test, fold+1)
    plot_recs.append(plot_rec)
    for_train[xx] += pred_for_train / (FOLDER - 1)
    for_valid[yy] = pred_for_valid
    for_test[:] += pred_for_test / FOLDER

np.save("for_train", for_train)
np.save("for_valid", for_valid)
np.save("for_test", for_test)


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4124698062.py in <cell line: 0>()
----> 1 for_train = np.zeros((len(X_train), 3))
      2 for_valid = np.zeros((len(X_train), 3))
      3 for_test = np.zeros((len(X_test), 3))
      4 
      5 plot_recs = []

NameError: name 'X_train' is not defined

## === cell 7
if SAVE_AND_LOAD_BEST_MODEL:
    print("Best: ")
    plot_recs_copy = np.array([plot_rec["best"] for plot_rec in plot_recs]).mean(axis=0)
    print("train_loss_avg: ", round(plot_recs_copy[0], 4), "  train_score_avg: ", round(plot_recs_copy[1], 4))
    print("valid_loss_avg: ", round(plot_recs_copy[2], 4), "  valid_score_avg: ", round(plot_recs_copy[3], 4))
else:
    print("Final: ")
    plot_recs_copy = np.array([plot_rec["final"] for plot_rec in plot_recs]).mean(axis=0)
    print("train_loss_avg: ", round(plot_recs_copy[0], 4), "  train_score_avg: ", round(plot_recs_copy[1], 4))
    print("valid_loss_avg: ", round(plot_recs_copy[2], 4), "  valid_score_avg: ", round(plot_recs_copy[3], 4))


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3315594510.py in <cell line: 0>()
      1 if SAVE_AND_LOAD_BEST_MODEL:
      2     print("Best: ")
----> 3     plot_recs_copy = np.array([plot_rec["best"] for plot_rec in plot_recs]).mean(axis=0)
      4     print("train_loss_avg: ", round(plot_recs_copy[0], 4), "  train_score_avg: ", round(plot_recs_copy[1], 4))
      5     print("valid_loss_avg: ", round(plot_recs_copy[2], 4), "  valid_score_avg: ", round(plot_recs_copy[3], 4))

NameError: name 'plot_recs' is not defined

## === cell 8
if SAVE_AND_LOAD_BEST_MODEL:
    for fold, adict in enumerate(plot_recs):
        best_epoch = adict["best_epoch"]

        best_epoch_train_loss = round(adict['best'][0],4)
        best_epoch_train_score = round(adict['best'][1],4)
        best_epoch_valid_loss = round(adict['best'][2],4)
        best_epoch_valid_score = round(adict['best'][3],4)

        min_loss = min(min(adict["train_loss"]), min(adict["valid_loss"]))
        max_loss = max(max(adict["train_loss"]), max(adict["valid_loss"]))
        min_score = min(min(adict["train_score"]), min(adict["valid_score"]))
        max_score = max(max(adict["train_score"]), max(adict["valid_score"]))
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        ax[0].plot(range(1, len(adict["train_loss"])+1), adict["train_loss"], label="train_loss")
        ax[0].plot(range(1, len(adict["train_loss"])+1), adict["valid_loss"], label="valid_loss")
        ax[0].plot([best_epoch, best_epoch], [min_loss, max_loss], label="best_epoch")
        ax[1].plot(range(1, len(adict["train_loss"])+1), adict["train_score"], label="train_score")
        ax[1].plot(range(1, len(adict["train_loss"])+1), adict["valid_score"], label="valid_score")
        ax[1].plot([best_epoch, best_epoch], [min_score, max_score], label="best_epoch")
        ax[0].legend()
        ax[1].legend()
        ax[0].grid()
        ax[1].grid()
        ax[0].set_title(f"Fold{fold+1}-Loss-Epoch{best_epoch}-train{best_epoch_train_loss}-valid{best_epoch_valid_loss}")
        ax[1].set_title(f"Fold{fold+1}-Score-Epoch{best_epoch}-train{best_epoch_train_score}-valid{best_epoch_valid_score}")
        ax[0].set_ylim(20, 100)
        ax[1].set_ylim(6, 8)
        plt.show()


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4068104277.py in <cell line: 0>()
      1 # The place where the green line is is the epochs of the best model
      2 if SAVE_AND_LOAD_BEST_MODEL:
----> 3     for fold, adict in enumerate(plot_recs):
      4         best_epoch = adict["best_epoch"]
      5 

NameError: name 'plot_recs' is not defined

## === cell 9
sigma_opt = mean_absolute_error(Y_train, for_valid[:, 1])
unc = for_valid[:, 2] - for_valid[:, 0]
sigma_mean = np.mean(unc)
print(sigma_opt, sigma_mean)
print("Min: ", unc.min(), "Mean: ", unc.mean(), "Max: ", unc.max(), "Mean(>=0): ", (unc>=0).mean())
plt.figure(figsize=(12, 6))
plt.hist(unc)
plt.title("uncertainty in prediction")
plt.show()


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3780937394.py in <cell line: 0>()
----> 1 sigma_opt = mean_absolute_error(Y_train, for_valid[:, 1])
      2 unc = for_valid[:, 2] - for_valid[:, 0]
      3 sigma_mean = np.mean(unc)
      4 print(sigma_opt, sigma_mean)
      5 print("Min: ", unc.min(), "Mean: ", unc.mean(), "Max: ", unc.max(), "Mean(>=0): ", (unc>=0).mean())

NameError: name 'Y_train' is not defined

## === cell 10
plt.figure(figsize=(15, 8))
idxs = np.random.randint(0, Y_train.shape[0], 100)
plt.plot(Y_train[idxs], label="ground truth")
plt.plot(for_valid[idxs, 0], label=f"q{int(QUANTILE[0]*100)}")
plt.plot(for_valid[idxs, 1], label=f"q{int(QUANTILE[1]*100)}")
plt.plot(for_valid[idxs, 2], label=f"q{int(QUANTILE[2]*100)}")
plt.legend(loc="best")
plt.show()


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4117835786.py in <cell line: 0>()
      1 plt.figure(figsize=(15, 8))
----> 2 idxs = np.random.randint(0, Y_train.shape[0], 100)
      3 plt.plot(Y_train[idxs], label="ground truth")
      4 plt.plot(for_valid[idxs, 0], label=f"q{int(QUANTILE[0]*100)}")
      5 plt.plot(for_valid[idxs, 1], label=f"q{int(QUANTILE[1]*100)}")

NameError: name 'Y_train' is not defined

## === cell 11
sub['FVC1'] = 0.996 * for_test[:, 1]
sub['Confidence1'] = for_test[:, 2] - for_test[:, 0]
subm = sub[['Patient_Week', 'FVC', 'Confidence', 'FVC1', 'Confidence1']].copy()


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1603629970.py in <cell line: 0>()
----> 1 sub['FVC1'] = 0.996 * for_test[:, 1]
      2 sub['Confidence1'] = for_test[:, 2] - for_test[:, 0]
      3 subm = sub[['Patient_Week', 'FVC', 'Confidence', 'FVC1', 'Confidence1']].copy()

NameError: name 'for_test' is not defined

## === cell 12
subm.loc[~subm.FVC1.isnull(), 'FVC'] = subm.loc[~subm.FVC1.isnull(), 'FVC1']
if sigma_mean < 70:
    subm['Confidence'] = sigma_opt
else:
    subm.loc[~subm.FVC1.isnull(), 'Confidence'] = subm.loc[~subm.FVC1.isnull(), 'Confidence1']

subm.describe().T


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/391367086.py in <cell line: 0>()
----> 1 subm.loc[~subm.FVC1.isnull(), 'FVC'] = subm.loc[~subm.FVC1.isnull(), 'FVC1']
      2 if sigma_mean < 70:
      3     subm['Confidence'] = sigma_opt
      4 else:
      5     subm.loc[~subm.FVC1.isnull(), 'Confidence'] = subm.loc[~subm.FVC1.isnull(), 'Confidence1']

NameError: name 'subm' is not defined

## === cell 13
otest = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
for i in range(len(otest)):
    subm.loc[subm['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'FVC'] = otest.FVC[i]
    subm.loc[subm['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'Confidence'] = 0.1

subm[["Patient_Week", "FVC", "Confidence"]].to_csv("submission.csv", index=False)


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2574850780.py in <cell line: 0>()
      1 otest = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
      2 for i in range(len(otest)):
----> 3     subm.loc[subm['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'FVC'] = otest.FVC[i]
      4     subm.loc[subm['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'Confidence'] = 0.1
      5 

NameError: name 'subm' is not defined
