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
pydicom==3.0.1
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-image==0.25.2
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
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


import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import torch as pt
from torch.utils.data import Dataset, DataLoader, RandomSampler
import torch.optim as optim
import matplotlib.pyplot as plt
import pydicom
from sklearn.model_selection import KFold
import seaborn as sns
import pydicom
from glob import glob
import scipy.ndimage
from skimage import morphology
from skimage import measure
from skimage.filters import threshold_otsu, median
from scipy.ndimage import binary_fill_holes
from skimage.segmentation import clear_border
from scipy.stats import describe

trainImagesPath = "/kaggle/input/osic-pulmonary-fibrosis-progression/train/"
dtype = pt.float
use_cuda = pt.cuda.is_available()
device = pt.device("cuda:0" if use_cuda else "cpu")


inputs = [
    "PercentIn",
    "AgeIn",
    "WeekIn",
    "min_FVC",
    "SmokingStatus_Currently smokes",
    "SmokingStatus_Ex-smoker",
    "SmokingStatus_Never smoked",
    "Sex_Male",
    "Sex_Female",
]

test = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv")
train = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv")
subms = pd.read_csv(
    "../input/osic-pulmonary-fibrosis-progression/sample_submission.csv"
)
subms["Patient"] = subms.Patient_Week.apply(lambda x: x.split("_")[0])
subms["Weeks"] = subms.Patient_Week.apply(lambda x: int(x.split("_")[-1]))
train["Split"] = "train"
test["Split"] = "test"
test["PatientDir"] = test.Patient.apply(
    lambda x: "../input/osic-pulmonary-fibrosis-progression/test/" + x
)
train["PatientDir"] = train.Patient.apply(
    lambda x: "../input/osic-pulmonary-fibrosis-progression/train/" + x
)
subms["PatientDir"] = subms.Patient.apply(
    lambda x: "../input/osic-pulmonary-fibrosis-progression/train/" + x
)
subms = subms[["Patient", "Weeks", "Patient_Week"]]
subms = subms.merge(test.drop("Weeks", axis=1), on="Patient")
subms["Split"] = "subm"

data = pd.concat([test, subms, train], ignore_index=True, sort=False)

data["first_week"] = data.Weeks
data["first_week"] = data.groupby("Patient")["first_week"].transform(
    "min"
)  # This assumes all patient ids in submission set exist in test set later. Set missing to train set mean?
data["first_week"] = data.Weeks - data.first_week
data["min_FVC"] = data.groupby("Patient")["FVC"].transform("min")  # Same here...
data["WeekIn"] = (data.first_week - data.first_week.min()) / (
    data.first_week.max() - data.first_week.min()
)
data = pd.concat(
    [data, pd.get_dummies(data.SmokingStatus, prefix="SmokingStatus")], axis=1
)
data = pd.concat([data, pd.get_dummies(data.Sex, prefix="Sex")], axis=1)
train = data.loc[data.Split == "train"].copy()
subms = data.loc[data.Split == "subm"].copy()
test = data.loc[data.Split == "test"].copy()

data.corr(numeric_only=True)


## === cell 1
print(data['PatientDir'])
print((data.nunique()))


## === cell 2
class OSICDataSet(Dataset):
    def __init__(self, data, mode = "train"):
        self.data = data
        self.data['Smoke'] = self.data.SmokingStatus.replace({'Ex-smoker' : 0.5, 'Never smoked' : 0, 'Currently smokes' : 1})
        self.data['Gender'] = self.data.Sex.replace({'Male' : 1, 'Female' : 0})
        self.data['min_FVC'] = (self.data.min_FVC - data.min_FVC.min()) / (data.min_FVC.max() - data.min_FVC.min())
        self.data['WeekIn'] = (self.data.first_week - data.first_week.min()) / (data.first_week.max() - data.first_week.min())
        self.data['AgeIn'] = (self.data.Age - data.Age.min()) / (data.Age.max() - data.Age.min())
        self.data['PercentIn'] = (self.data.Percent - data.Percent.min()) / (data.Percent.max() - data.Percent.min())
        self.mode = mode
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        otherData = pt.from_numpy(self.data.loc[idx, inputs].values.astype(np.float32))
        if self.mode == "train":
            targets = pt.from_numpy(self.data.loc[idx, ['FVC']].values.astype(np.float32))
            return otherData, targets
        return otherData #TODO: Add image data
    
class Model(pt.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.start = pt.nn.Sequential(
            pt.nn.Linear(len(inputs), 128),
            pt.nn.ReLU(),
            pt.nn.Linear(128, 256),
            pt.nn.ReLU(),
            pt.nn.Linear(256, 512),
            pt.nn.ReLU(),
            pt.nn.Linear(512, 256),
            pt.nn.ReLU(),
        )
        self.left = pt.nn.Sequential(
            pt.nn.Linear(256, 128)
        )
        self.sigmoid = pt.nn.Sigmoid()
        self.right = pt.nn.Sequential(
            pt.nn.Linear(256,128)
        )
        self.last = pt.nn.Sequential(
            pt.nn.Linear(128, 3),
        )
        self.lastRelu = pt.nn.Sequential(
            pt.nn.Linear(128, 3),
            pt.nn.ReLU()
        )
        
        
    def forward(self, x):
        h = self.start(x)
        l = self.left(h)
        r = self.right(h)
        h = l * self.sigmoid(r)
        p1 = self.last(h)
        p2 = self.lastRelu(h)
        out = p1 + pt.cumsum(p2, 1)
        return out

model = Model()

C1, C2 = pt.FloatTensor([70]), pt.FloatTensor([1000])
def score(y_pred, y_true):
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]

    sigma_clip = pt.max(sigma, C1.expand_as(sigma))
    delta = pt.abs(y_true - fvc_pred)
    delta = pt.min(delta, C2)
    sq2 = pt.sqrt(pt.FloatTensor([2]))
    metric = (delta / sigma_clip)*sq2 + pt.log(sigma_clip* sq2)
    return pt.mean(metric)

def pinballLoss(pred, label, quant):
    err = label - pred
    m = pt.mean(pt.max(quant * err, (quant-1)*err))
    return m

def loss(pred, label):
    quantiles = pt.FloatTensor([0.2, 0.5, 0.8])
    return 0.8 * pinballLoss(pred, label, quantiles) + 0.2 * score(pred, label)


## === cell 3
trainDataSet = OSICDataSet(train) 
optimizer = optim.Adam(model.parameters(), lr=0.075, weight_decay = 0.1, eps=0.001)
losses = []
valLosses = []
nSplits = 5
kf = KFold(n_splits=5)
c = 0
submDataSet = OSICDataSet(subms, "submission")
inp = pt.from_numpy(submDataSet.data[inputs].values.astype(np.float32))
stopping = 0
for i in range(100):
    for train_index, val_index in kf.split(trainDataSet):
        train_x, train_y = trainDataSet[train_index]
        val_x, val_y = trainDataSet[val_index]
        optimizer.zero_grad()
        pred = model(train_x)
        los = loss(pred, train_y)
        los.backward()
        if c % 10 == 0:
            pred = model(val_x)
            valLos = loss(pred, val_y)
            losses.append(los)
            valLosses.append(valLos)
            if len(valLosses) > 1 and valLosses[-1] < valLosses[-2]:
                stopping += 1
                if stopping > nSplits:
                    i = 800
                break
        stopping = 0
        optimizer.step()
        c+=1
    


## --- ERROR in cell 3, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mKeyError[0m                                  Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1035674195.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     11[0m [0;32mfor[0m [0mi[0m [0;32min[0m [0mrange[0m[0;34m([0m[0;36m100[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     12[0m     [0;32mfor[0m [0mtrain_index[0m[0;34m,[0m [0mval_index[0m [0;32min[0m [0mkf[0m[0;34m.[0m[0msplit[0m[0;34m([0m[0mtrainDataSet[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 13[0;31m         [0mtrain_x[0m[0;34m,[0m [0mtrain_y[0m [0;34m=[0m [0mtrainDataSet[0m[0;34m[[0m[0mtrain_index[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     14[0m         [0mval_x[0m[0;34m,[0m [0mval_y[0m [0;34m=[0m [0mtrainDataSet[0m[0;34m[[0m[0mval_index[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m     15[0m         [0moptimizer[0m[0;34m.[0m[0mzero_grad[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/763861143.py[0m in [0;36m__getitem__[0;34m(self, idx)[0m
[1;32m     15[0m     [0;32mdef[0m [0m__getitem__[0m[0;34m([0m[0mself[0m[0;34m,[0m [0midx[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     16[0m         [0;31m#image = self.data.loc[idx, ['Patient']].map(lambda filename: pydicom.dcmread(trainImagesPath + filename + "/1.dcm")).item().pixel_array.astype(np.float64)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 17[0;31m         [0motherData[0m [0;34m=[0m [0mpt[0m[0;34m.[0m[0mfrom_numpy[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mdata[0m[0;34m.[0m[0mloc[0m[0;34m[[0m[0midx[0m[0;34m,[0m [0minputs[0m[0;34m][0m[0;34m.[0m[0mvalues[0m[0;34m.[0m[0mastype[0m[0;34m([0m[0mnp[0m[0;34m.[0m[0mfloat32[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     18[0m         [0;32mif[0m [0mself[0m[0;34m.[0m[0mmode[0m [0;34m==[0m [0;34m"train"[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     19[0m             [0mtargets[0m [0;34m=[0m [0mpt[0m[0;34m.[0m[0mfrom_numpy[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mdata[0m[0;34m.[0m[0mloc[0m[0;34m[[0m[0midx[0m[0;34m,[0m [0;34m[[0m[0;34m'FVC'[0m[0;34m][0m[0;34m][0m[0;34m.[0m[0mvalues[0m[0;34m.[0m[0mastype[0m[0;34m([0m[0mnp[0m[0;34m.[0m[0mfloat32[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py[0m in [0;36m__getitem__[0;34m(self, key)[0m
[1;32m   1182[0m             [0;32mif[0m [0mself[0m[0;34m.[0m[0m_is_scalar_access[0m[0;34m([0m[0mkey[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1183[0m                 [0;32mreturn[0m [0mself[0m[0;34m.[0m[0mobj[0m[0;34m.[0m[0m_get_value[0m[0;34m([0m[0;34m*[0m[0mkey[0m[0;34m,[0m [0mtakeable[0m[0;34m=[0m[0mself[0m[0;34m.[0m[0m_takeable[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1184[0;31m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_getitem_tuple[0m[0;34m([0m[0mkey[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1185[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1186[0m             [0;31m# we by definition only have the 0th axis[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py[0m in [0;36m_getitem_tuple[0;34m(self, tup)[0m
[1;32m   1373[0m         [0;31m# ugly hack for GH #836[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1374[0m         [0;32mif[0m [0mself[0m[0;34m.[0m[0m_multi_take_opportunity[0m[0;34m([0m[0mtup[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1375[0;31m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_multi_take[0m[0;34m([0m[0mtup[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1376[0m [0;34m[0m[0m
[1;32m   1377[0m         [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_getitem_tuple_same_dim[0m[0;34m([0m[0mtup[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py[0m in [0;36m_multi_take[0;34m(self, tup)[0m
[1;32m   1324[0m         """
[1;32m   1325[0m         [0;31m# GH 836[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1326[0;31m         d = {
[0m[1;32m   1327[0m             [0maxis[0m[0;34m:[0m [0mself[0m[0;34m.[0m[0m_get_listlike_indexer[0m[0;34m([0m[0mkey[0m[0;34m,[0m [0maxis[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1328[0m             [0;32mfor[0m [0;34m([0m[0mkey[0m[0;34m,[0m [0maxis[0m[0;34m)[0m [0;32min[0m [0mzip[0m[0;34m([0m[0mtup[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0mobj[0m[0;34m.[0m[0m_AXIS_ORDERS[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py[0m in [0;36m<dictcomp>[0;34m(.0)[0m
[1;32m   1325[0m         [0;31m# GH 836[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1326[0m         d = {
[0;32m-> 1327[0;31m             [0maxis[0m[0;34m:[0m [0mself[0m[0;34m.[0m[0m_get_listlike_indexer[0m[0;34m([0m[0mkey[0m[0;34m,[0m [0maxis[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1328[0m             [0;32mfor[0m [0;34m([0m[0mkey[0m[0;34m,[0m [0maxis[0m[0;34m)[0m [0;32min[0m [0mzip[0m[0;34m([0m[0mtup[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0mobj[0m[0;34m.[0m[0m_AXIS_ORDERS[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1329[0m         }

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py[0m in [0;36m_get_listlike_indexer[0;34m(self, key, axis)[0m
[1;32m   1556[0m         [0maxis_name[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mobj[0m[0;34m.[0m[0m_get_axis_name[0m[0;34m([0m[0maxis[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1557[0m [0;34m[0m[0m
[0;32m-> 1558[0;31m         [0mkeyarr[0m[0;34m,[0m [0mindexer[0m [0;34m=[0m [0max[0m[0;34m.[0m[0m_get_indexer_strict[0m[0;34m([0m[0mkey[0m[0;34m,[0m [0maxis_name[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1559[0m [0;34m[0m[0m
[1;32m   1560[0m         [0;32mreturn[0m [0mkeyarr[0m[0;34m,[0m [0mindexer[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py[0m in [0;36m_get_indexer_strict[0;34m(self, key, axis_name)[0m
[1;32m   6198[0m             [0mkeyarr[0m[0;34m,[0m [0mindexer[0m[0;34m,[0m [0mnew_indexer[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_reindex_non_unique[0m[0;34m([0m[0mkeyarr[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   6199[0m [0;34m[0m[0m
[0;32m-> 6200[0;31m         [0mself[0m[0;34m.[0m[0m_raise_if_missing[0m[0;34m([0m[0mkeyarr[0m[0;34m,[0m [0mindexer[0m[0;34m,[0m [0maxis_name[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   6201[0m [0;34m[0m[0m
[1;32m   6202[0m         [0mkeyarr[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mtake[0m[0;34m([0m[0mindexer[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py[0m in [0;36m_raise_if_missing[0;34m(self, key, indexer, axis_name)[0m
[1;32m   6247[0m         [0;32mif[0m [0mnmissing[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   6248[0m             [0;32mif[0m [0mnmissing[0m [0;34m==[0m [0mlen[0m[0;34m([0m[0mindexer[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 6249[0;31m                 [0;32mraise[0m [0mKeyError[0m[0;34m([0m[0;34mf"None of [{key}] are in the [{axis_name}]"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   6250[0m [0;34m[0m[0m
[1;32m   6251[0m             [0mnot_found[0m [0;34m=[0m [0mlist[0m[0;34m([0m[0mensure_index[0m[0;34m([0m[0mkey[0m[0;34m)[0m[0;34m[[0m[0mmissing_mask[0m[0;34m.[0m[0mnonzero[0m[0;34m([0m[0;34m)[0m[0;34m[[0m[0;36m0[0m[0;34m][0m[0;34m][0m[0;34m.[0m[0munique[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mKeyError[0m: "None of [Index([ 279,  280,  281,  282,  283,  284,  285,  286,  287,  288,\n       ...\n       1384, 1385, 1386, 1387, 1388, 1389, 1390, 1391, 1392, 1393],\n      dtype='int64', length=1115)] are in the [index]"

## === cell 4
sns.set(style="white", palette="muted", color_codes=True)

plt.plot(losses, label="Train loss")
plt.plot(valLosses, label="Val loss")
plt.legend()
plt.show()

asd = pt.from_numpy(trainDataSet.data[inputs].values.astype(np.float32))
pred = model(asd).detach()
idxs = np.random.randint(0, len(trainDataSet), 100)
plt.plot(trainDataSet.data.loc[idxs, ['FVC']].values.astype(np.float32), label="ground truth")
plt.plot(pred[idxs, 0], label="q25")
plt.plot(pred[idxs, 1], label="q50")
plt.plot(pred[idxs, 2], label="q75")
plt.legend(loc="best")
plt.show()
c = pred[:,2] - pred[:,0]
f = pred[:,1]


fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
sns.distplot(c, color="g", kde=False, kde_kws={"shade": True}, ax=axes[0]).set_title("Predicted Confidence on train set")
sns.distplot(f, color="g", kde=False, kde_kws={"shade": True}, ax=axes[1]).set_title("Predicted FVC on train set")
plt.show()

submDataSet = OSICDataSet(subms, "submission")
inp = pt.from_numpy(submDataSet.data[inputs].values.astype(np.float32))
out = model(inp).detach()
confidence = out[:,2] - out[:,0]
fvc = out[:,1]
fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
sns.distplot(confidence, color="g", kde=False, kde_kws={"shade": True}, ax=axes[0]).set_title("Confidence on submission set")
sns.distplot(fvc, color="g", kde=False, kde_kws={"shade": True}, ax=axes[1]).set_title("FVC on submision set")
plt.show()

submDataSet.data['pFVC'] = fvc
submDataSet.data['cConf'] = confidence

submission = pd.DataFrame({'Patient_Week' : submDataSet.data.Patient_Week, 'FVC' : fvc, 'Confidence' : confidence})
otest = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
for i in range(len(otest)):
    submission.loc[submission['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'FVC'] = otest.FVC[i]
    submission.loc[submission['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'Confidence'] = 0.1

submission.to_csv("submission.csv", index=False)
null_columns=submission.columns[submission.isnull().any()]
print(submission.head())
print(len(submission))
