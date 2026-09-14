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

category_encoders==2.7.0
geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
lightgbm==4.6.0
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
plotly==5.24.1
plotly-express==0.4.1
pydicom==3.0.1
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

# 4. Code solution

## === cell 0
from IPython.display import HTML
HTML('')


## === cell 1
import os
import pydicom as dcm
import glob
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.linear_model import Ridge
import random
from tqdm.notebook import tqdm
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from sklearn.metrics import mean_squared_error
import category_encoders as ce
from PIL import Image
import cv2
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

import seaborn as sns
p = sns.color_palette()
import plotly.express as px


## === cell 2
train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
train['ID'] = train['Patient'].astype(str) + '_' + train['Weeks'].astype(str)
print(train.shape)
train.head()


## === cell 3
test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')


## === cell 4
patient_sizes = [len(os.listdir('../input/osic-pulmonary-fibrosis-progression/train/' + d)) for d in os.listdir('../input/osic-pulmonary-fibrosis-progression/train/')]
plt.hist(patient_sizes, color=p[2])
plt.ylabel('Number of patients')
plt.xlabel('DICOM files')
plt.title('Histogram of DICOM count per patient');


## === cell 5
patient_sizes_arr = np.array(patient_sizes) 
for x in [200,300,400,500,600,800,1000]:
    print("We have ",sum(patient_sizes_arr > x),"patients with more than",x,"DICOM files")


## === cell 6
print("Total entries",len(train['Patient']))
print("Total patients",len(np.unique(train['Patient'])))


## === cell 7
train.info()


## === cell 8
train.isnull().sum()


## === cell 9
train[train['Patient'] == 'ID00007637202177411956430']


## === cell 10
train_patient_data = train[['Patient','Age','Sex','SmokingStatus']].drop_duplicates().reset_index(drop=True)
train_patient_data.head()


## === cell 11
print("Different smoking status:",np.unique(train_patient_data.SmokingStatus))


## === cell 12
plt.hist(train_patient_data.SmokingStatus)
plt.ylabel('Number of patients')
plt.xlabel('SmokingStatus')
plt.title('Histogram of SmokingStatus of patients');


## === cell 13
labels = 'Ex-smoker', 'Never smoked', 'Currently smokes'
values = [np.sum(train_patient_data.SmokingStatus.eq("Ex-smoker")), np.sum(train_patient_data.SmokingStatus.eq("Never smoked")), np.sum(train_patient_data.SmokingStatus.eq("Currently smokes"))]
fig1, ax1 = plt.subplots()
ax1.pie(values,  labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')
plt.title('Distribution of patients by SmokingStatus')


## === cell 14
plt.figure(figsize=(30,15))
plt.hist(train_patient_data.Age,bins=50)
plt.ylabel('Number of patients')
plt.xlabel('Age')
plt.title('Histogram of Age of patients');


## === cell 15
plt.figure(figsize=(16, 6))
sns.kdeplot(train_patient_data.loc[train_patient_data['Sex'] == 'Male', 'Age'], label = 'Male',shade=True)
sns.kdeplot(train_patient_data.loc[train_patient_data['Sex'] == 'Female', 'Age'], label = 'Female',shade=True)
plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages over gender');


## === cell 16
plt.figure(figsize=(16, 6))
sns.kdeplot(train_patient_data.loc[train_patient_data['SmokingStatus'] == 'Ex-smoker', 'Age'], label = 'Ex-smoker',shade=True)
sns.kdeplot(train_patient_data.loc[train_patient_data['SmokingStatus'] == 'Never smoked', 'Age'], label = 'Never smoked',shade=True)
sns.kdeplot(train_patient_data.loc[train_patient_data['SmokingStatus'] == 'Currently smokes', 'Age'], label = 'Currently smokes', shade=True)

plt.xlabel('Age (years)');
plt.ylabel('Density');
plt.title('Distribution of Ages over SmokingStatus');


## === cell 17
plt.figure(figsize=(16, 6))
ax = sns.violinplot(x = train_patient_data['SmokingStatus'], y = train_patient_data['Age'], palette = 'Reds')
ax.set_xlabel(xlabel = 'Smoking habit', fontsize = 15)
ax.set_ylabel(ylabel = 'Age', fontsize = 15)
ax.set_title(label = 'Distribution of Smokers over Age', fontsize = 20)
plt.show()


## === cell 18
plt.hist(train_patient_data.Sex)
plt.ylabel('Number of patients')
plt.xlabel('Sex')
plt.title('Histogram of Sex of patients');


## === cell 19
labels = 'Male', 'Female'
values = [np.sum(train_patient_data.Sex.eq("Male")), np.sum(train_patient_data.Sex.eq("Female"))]
fig1, ax1 = plt.subplots()
ax1.pie(values,  labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')
plt.title('Distribution of patients by Gender')


## === cell 20
cond1_colname = 'Sex'
cond1_vales = np.unique(train_patient_data[[cond1_colname]])

cond2_colname = 'SmokingStatus'
cond2_vales = np.unique(train_patient_data[[cond2_colname]])
count_list = []
for x in cond1_vales:
    tmp = []
    for y in cond2_vales:
        tmp.append(np.sum((train_patient_data[cond1_colname] == x) & (train_patient_data[cond2_colname] == y)))
    count_list.append(tmp)
count_list = pd.DataFrame(count_list,columns=cond2_vales,index=cond1_vales)

count_list.plot(kind='bar',stacked = True, figsize=(10,6))
plt.legend(bbox_to_anchor=(1,1), title = cond2_colname)
plt.title(cond2_colname+' by '+cond1_colname)
plt.xlabel(cond1_colname)
plt.ylabel(cond2_colname)
warnings.filterwarnings('ignore')
plt.show()


## === cell 21
for x in cond1_vales:
    fig1, ax1 = plt.subplots()
    ax1.pie(count_list.loc[x,:],  labels=cond2_vales, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')
    plt.title('Distribution of SmokingStatus in '+x)
    plt.show()


## === cell 22
train_patient_data.boxplot(column='Age', by='Sex')
plt.title("Boxplot of Age by Sex")
plt.ylabel("Age")
plt.suptitle("")
plt.show()


## === cell 23
train['Patient'].value_counts().max()


## === cell 24
plt.figure(figsize=(30,15))
plt.hist(train.Weeks,bins=150)
plt.ylabel('Number of Weeks')
plt.xlabel('Week')
plt.title('Distribution of Weeks');


## === cell 25
week_arr = np.array(train.Weeks)
print("We have",sum(week_arr < 0),"records before the baseline CT")
for x in [60,70,80,90,100,110,120]:
    print("We have",sum(week_arr > x),"records after",x,"weeks")


## === cell 26
plt.scatter(x=list(train['FVC'].value_counts().keys()),y=list(train['FVC'].value_counts().values))
plt.ylabel('Count of FVC')
plt.xlabel('FVC')
plt.title('Distribution of FVC')


## === cell 27
plt.figure(figsize=(30,15))
plt.hist(train['FVC'],bins=250)
plt.ylabel('Count of FVC')
plt.xlabel('FVC')
plt.title('Distribution of FVC')


## === cell 28
print("so FVC values range from",min(train['FVC']),"to",max(train['FVC']),"and do not seem to be repeating often")


## === cell 29
fig = px.scatter(train, x="FVC", y="Percent", color='Age')
fig.show()


## === cell 30
np.unique(train[train['FVC'] > 5100].Patient)


## === cell 31
sum(train[train['Patient'] == 'ID00219637202258203123958'].FVC < 5100)


## === cell 32
fig = px.scatter(train, x="Weeks", y="Age", color='Sex')
fig.show()


## === cell 33
fig = px.scatter(train, x="FVC", y="Age", color='Sex')
fig.show()


## === cell 34
high_FVC_female_patients = np.unique(train[(train['FVC'] > 2800) & (train['Sex']=='Female')].Patient)
high_FVC_female_patients


## === cell 35
sum(train['Patient'].isin(high_FVC_female_patients) & (train['FVC'] < 2800))


## === cell 36
tmp_patients = np.unique(train[train['Patient'].isin(high_FVC_female_patients) & (train['FVC'] < 2800)].Patient)
tmp_patients


## === cell 37
set(high_FVC_female_patients) - set(tmp_patients)


## === cell 38
fig = px.scatter(train, x="FVC", y="Weeks", color='SmokingStatus')
fig.show()


## === cell 39
plt.figure(figsize=(30,15))
plt.hist(train['Percent'],bins=25)
plt.ylabel('Count of Percent')
plt.xlabel('Percent')
plt.title('Distribution of Percent')


## === cell 40
fig = px.violin(train, y='Percent', x='SmokingStatus', box=True, color='Sex', points="all",
          hover_data=train.columns)
fig.show()


## === cell 41
plt.figure(figsize=(16, 6))
ax = sns.violinplot(x = train['SmokingStatus'], y = train['Percent'], palette = 'Reds')
ax.set_xlabel(xlabel = 'Smoking Habit', fontsize = 15)
ax.set_ylabel(ylabel = 'Percent', fontsize = 15)
ax.set_title(label = 'Distribution of Smoking Status Over Percentage', fontsize = 20)
plt.show()


## === cell 42
fig = px.scatter(train, x="Age", y="Percent", color='SmokingStatus')
fig.show()


## === cell 43
files = folders = 0

path = "/kaggle/input/osic-pulmonary-fibrosis-progression/train"

for _, dirnames, filenames in os.walk(path):
    files += len(filenames)
    folders += len(dirnames)
print(files,"files/images\n",folders,'folders/patients')


## === cell 44
files = glob.glob('../input/osic-pulmonary-fibrosis-progression/train/*/*.dcm')
def dicom_to_image(filename):
    im = dcm.dcmread(filename)
    img = im.pixel_array
    img[img == -2000] = 0
    return img
f, plots = plt.subplots(4, 5, sharex='col', sharey='row', figsize=(10, 8))
for i in range(20):
    plots[i // 5, i % 5].axis('off')
    plots[i // 5, i % 5].imshow(dicom_to_image(files[i]), cmap=plt.cm.bone)


## === cell 45
f, plots = plt.subplots(4, 5, sharex='col', sharey='row', figsize=(10, 8))
for i in range(20):
    plots[i // 5, i % 5].axis('off')
    plots[i // 5, i % 5].imshow(dicom_to_image(np.random.choice(files)), cmap=plt.cm.bone)


## === cell 46
import copy
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

SummaryWriter = None  # type: ignore

if SummaryWriter is None:

    class SummaryWriter:  # noqa: N801
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def add_scalars(self, *args, **kwargs):
            pass

        def add_image(self, *args, **kwargs):
            pass

        def add_images(self, *args, **kwargs):
            pass

        def add_histogram(self, *args, **kwargs):
            pass

        def add_text(self, *args, **kwargs):
            pass

        def flush(self):
            pass

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()
            return False


from tqdm.notebook import trange
from time import time


## === cell 47
root_dir = Path('/kaggle/input/osic-pulmonary-fibrosis-progression')
model_dir = Path('/kaggle/working')
num_kfolds = 5
batch_size = 32
learning_rate = 3e-3
num_epochs = 1000
es_patience = 20
quantiles = (0.2, 0.5, 0.8)
model_name ='descartes'
tensorboard_dir = Path('/kaggle/working/runs')


## === cell 48
class ClinicalDataset(Dataset):
    def __init__(self, root_dir, mode, transform=None):
        self.transform = transform
        self.mode = mode

        tr = pd.read_csv(Path(root_dir)/"train.csv")
        tr.drop_duplicates(keep=False, inplace=True, subset=['Patient', 'Weeks'])
        chunk = pd.read_csv(Path(root_dir)/"test.csv")

        sub = pd.read_csv(Path(root_dir)/"sample_submission.csv")
        sub['Patient'] = sub['Patient_Week'].apply(lambda x: x.split('_')[0])
        sub['Weeks'] = sub['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))
        sub = sub[['Patient', 'Weeks', 'Confidence', 'Patient_Week']]
        sub = sub.merge(chunk.drop('Weeks', axis=1), on="Patient")

        tr['WHERE'] = 'train'
        chunk['WHERE'] = 'val'
        sub['WHERE'] = 'test'
        data = tr.append([chunk, sub])

        data['min_week'] = data['Weeks']
        data.loc[data.WHERE == 'test', 'min_week'] = np.nan
        data['min_week'] = data.groupby('Patient')['min_week'].transform('min')

        base = data.loc[data.Weeks == data.min_week]
        base = base[['Patient', 'FVC']].copy()
        base.columns = ['Patient', 'min_FVC']
        base['nb'] = 1
        base['nb'] = base.groupby('Patient')['nb'].transform('cumsum')
        base = base[base.nb == 1]
        base.drop('nb', axis=1, inplace=True)

        data = data.merge(base, on='Patient', how='left')
        data['base_week'] = data['Weeks'] - data['min_week']
        del base

        COLS = ['Sex', 'SmokingStatus']
        self.FE = []
        for col in COLS:
            for mod in data[col].unique():
                self.FE.append(mod)
                data[mod] = (data[col] == mod).astype(int)

        data['age'] = (data['Age'] - data['Age'].min()) / \
                      (data['Age'].max() - data['Age'].min())
        data['BASE'] = (data['min_FVC'] - data['min_FVC'].min()) / \
                       (data['min_FVC'].max() - data['min_FVC'].min())
        data['week'] = (data['base_week'] - data['base_week'].min()) / \
                       (data['base_week'].max() - data['base_week'].min())
        data['percent'] = (data['Percent'] - data['Percent'].min()) / \
                          (data['Percent'].max() - data['Percent'].min())
        self.FE += ['age', 'percent', 'week', 'BASE']

        self.raw = data.loc[data.WHERE == mode].reset_index()
        del data

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {
            'patient_id': self.raw['Patient'].iloc[idx],
            'features': self.raw[self.FE].iloc[idx].values,
            'target': self.raw['FVC'].iloc[idx]
        }
        if self.transform:
            sample = self.transform(sample)

        return sample

    def group_kfold(self, n_splits):
        gkf = GroupKFold(n_splits=n_splits)
        groups = self.raw['Patient']
        for train_idx, val_idx in gkf.split(self.raw, self.raw, groups):
            train = Subset(self, train_idx)
            val = Subset(self, val_idx)
            yield train, val

    def group_split(self, test_size=0.2):
        """To test no-kfold
        """
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size)
        groups = self.raw['Patient']
        idx = list(gss.split(self.raw, self.raw, groups))
        train = Subset(self, idx[0][0])
        val = Subset(self, idx[0][1])
        return train, val


## === cell 49
class QuantModel(nn.Module):
    def __init__(self, in_tabular_features=9, out_quantiles=3):
        super(QuantModel, self).__init__()
        self.fc1 = nn.Linear(in_tabular_features, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, out_quantiles)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x


def quantile_loss(preds, target, quantiles):
    assert not target.requires_grad
    assert preds.size(0) == target.size(0)
    losses = []
    for i, q in enumerate(quantiles):
        errors = target - preds[:, i]
        losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))
    loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
    return loss


## === cell 50
class Monitor:
    def __init__(self, model, es_patience, experiment_name, tensorboard_dir,
                 num_epochs, dataset_sizes, model_file):

        self.model = model
        self.model_file = model_file
        self.es_patience = es_patience
        self.tensorboard_dir = tensorboard_dir
        self.dataset_sizes = dataset_sizes
        date_time = datetime.now().strftime("%Y%m%d-%H%M")
        log_dir = tensorboard_dir / f'{experiment_name}-{date_time}'
        self.w = SummaryWriter(log_dir)

        self.bar = trange(num_epochs, desc=experiment_name)

        self.epoch_loss = {'train': np.inf, 'val': np.inf}
        self.epoch_metric = {'train': -np.inf, 'val': -np.inf}
        self.best_loss = np.inf
        self.best_model_wts = None

        self.e = {'train': 0, 'val': 0}  # epoch counter
        self.t = {'train': 0, 'val': 0}  # global time-step (never resets)
        self.running_loss = 0.0
        self.running_metric = 0.0
        self.es_counter = 0

    def reset_epoch(self):
        self.running_loss = 0.0
        self.running_metric = 0.0

    def step(self, loss, inputs, preds, targets, phase):
        self.running_loss += loss.item() * inputs.size(0)
        self.running_metric += self.metric(preds, targets).sum()
        self.t[phase] += 1

    def log_epoch(self, phase):
        self.epoch_loss[phase] = self.running_loss / self.dataset_sizes[phase]
        self.epoch_metric[phase] = self.running_metric / self.dataset_sizes[phase]
        self.bar.set_postfix(
            a_train_loss=f'{self.epoch_loss["train"]:0.1f}',
            b_val_loss=f'{self.epoch_loss["val"]:0.1f}',
            c_train_metric=f'{self.epoch_metric["train"]:0.4f}',
            d_val_metric=f'{self.epoch_metric["val"]:0.4f}',
            es_counter=self.es_counter
        )
        self.w.add_scalar(
            f'Loss/{phase}', self.epoch_loss[phase], self.e[phase])
        self.w.add_scalar(
            f'Accuracy/{phase}', self.epoch_metric[phase], self.e[phase])

        self.e[phase] += 1

        early_stop = False
        if phase == 'val':
            if self.epoch_loss['val'] < self.best_loss:
                self.best_loss = self.epoch_loss['val']
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                torch.save(self.best_model_wts, self.model_file)
                self.es_counter = 0
            else:
                self.es_counter += 1
                if self.es_counter >= self.es_patience:
                    early_stop = True
                    self.bar.close()

        return early_stop

    @staticmethod
    def metric(preds, targets):
        sigma = preds[:, 2] - preds[:, 0]
        sigma[sigma < 70] = 70
        delta = (preds[:, 1] - targets).abs()
        delta[delta > 1000] = 1000
        return -np.sqrt(2) * delta / sigma - torch.log(np.sqrt(2) * sigma)


## === cell 51
class ClinicalDataset(Dataset):
    def __init__(self, root_dir, mode, transform=None):
        self.transform = transform
        self.mode = mode

        tr = pd.read_csv(Path(root_dir) / "train.csv")
        tr.drop_duplicates(keep=False, inplace=True, subset=["Patient", "Weeks"])
        chunk = pd.read_csv(Path(root_dir) / "test.csv")

        sub = pd.read_csv(Path(root_dir) / "sample_submission.csv")
        sub["Patient"] = sub["Patient_Week"].apply(lambda x: x.split("_")[0])
        sub["Weeks"] = sub["Patient_Week"].apply(lambda x: int(x.split("_")[-1]))
        sub = sub[["Patient", "Weeks", "Confidence", "Patient_Week"]]
        sub = sub.merge(chunk.drop("Weeks", axis=1), on="Patient")

        tr["WHERE"] = "train"
        chunk["WHERE"] = "val"
        sub["WHERE"] = "test"
        data = pd.concat([tr, chunk, sub], ignore_index=False)

        data["min_week"] = data["Weeks"]
        data.loc[data.WHERE == "test", "min_week"] = np.nan
        data["min_week"] = data.groupby("Patient")["min_week"].transform("min")

        base = data.loc[data.Weeks == data.min_week]
        base = base[["Patient", "FVC"]].copy()
        base.columns = ["Patient", "min_FVC"]
        base["nb"] = 1
        base["nb"] = base.groupby("Patient")["nb"].transform("cumsum")
        base = base[base.nb == 1]
        base.drop("nb", axis=1, inplace=True)

        data = data.merge(base, on="Patient", how="left")
        data["base_week"] = data["Weeks"] - data["min_week"]
        del base

        COLS = ["Sex", "SmokingStatus"]
        self.FE = []
        for col in COLS:
            for mod in data[col].unique():
                self.FE.append(mod)
                data[mod] = (data[col] == mod).astype(int)

        data["age"] = (data["Age"] - data["Age"].min()) / (
            data["Age"].max() - data["Age"].min()
        )
        data["BASE"] = (data["min_FVC"] - data["min_FVC"].min()) / (
            data["min_FVC"].max() - data["min_FVC"].min()
        )
        data["week"] = (data["base_week"] - data["base_week"].min()) / (
            data["base_week"].max() - data["base_week"].min()
        )
        data["percent"] = (data["Percent"] - data["Percent"].min()) / (
            data["Percent"].max() - data["Percent"].min()
        )
        self.FE += ["age", "percent", "week", "BASE"]

        self.raw = data.loc[data.WHERE == mode].reset_index()
        del data

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {
            "patient_id": self.raw["Patient"].iloc[idx],
            "features": self.raw[self.FE].iloc[idx].values,
            "target": self.raw["FVC"].iloc[idx],
        }
        if self.transform:
            sample = self.transform(sample)

        return sample

    def group_kfold(self, n_splits):
        gkf = GroupKFold(n_splits=n_splits)
        groups = self.raw["Patient"]
        for train_idx, val_idx in gkf.split(self.raw, self.raw, groups):
            train = Subset(self, train_idx)
            val = Subset(self, val_idx)
            yield train, val

    def group_split(self, test_size=0.2):
        """To test no-kfold"""
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size)
        groups = self.raw["Patient"]
        idx = list(gss.split(self.raw, self.raw, groups))
        train = Subset(self, idx[0][0])
        val = Subset(self, idx[0][1])
        return train, val


## === cell 52
data = ClinicalDataset(root_dir, mode='test')
avg_preds = np.zeros((len(data), len(quantiles)))

for model in models:
    dataloader = DataLoader(data, batch_size=batch_size,
                            shuffle=False, num_workers=2)
    preds = []
    for batch in dataloader:
        inputs = batch['features'].float()
        with torch.no_grad():
            x = model(inputs)
            preds.append(x)

    preds = torch.cat(preds, dim=0).numpy()
    avg_preds += preds

avg_preds /= len(models)
df = pd.DataFrame(data=avg_preds, columns=list(quantiles))
df['Patient_Week'] = data.raw['Patient_Week']
df['FVC'] = df[quantiles[1]]
df['Confidence'] = df[quantiles[2]] - df[quantiles[0]]
df = df.drop(columns=list(quantiles))
df.to_csv('submission.csv', index=False)


## --- ERROR in cell 52, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mNameError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/4248399250.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      2[0m [0mavg_preds[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0mzeros[0m[0;34m([0m[0;34m([0m[0mlen[0m[0;34m([0m[0mdata[0m[0;34m)[0m[0;34m,[0m [0mlen[0m[0;34m([0m[0mquantiles[0m[0;34m)[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m [0;34m[0m[0m
[0;32m----> 4[0;31m [0;32mfor[0m [0mmodel[0m [0;32min[0m [0mmodels[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      5[0m     dataloader = DataLoader(data, batch_size=batch_size,
[1;32m      6[0m                             shuffle=False, num_workers=2)

[0;31mNameError[0m: name 'models' is not defined

## === cell 53
df.head()
