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
subms = subms[["Patient", "Weeks", "Patient_Week"]]
subms = subms.merge(test.drop("Weeks", axis=1), on="Patient")
subms["Split"] = "subm"

data = pd.concat([train, subms, test], ignore_index=True, sort=False)

data["first_week"] = data.Weeks
data["first_week"] = data.groupby("Patient")["first_week"].transform(
    "min"
)  # This assumes all patient ids in submission set exist in test set later. Set missing to train set mean?
data["first_week"] = data.Weeks - data.first_week
data["min_FVC"] = data.groupby("Patient")["FVC"].transform("min")  # Same here...
print(len(data))
data["WeekIn"] = (data.first_week - data.first_week.min()) / (
    data.first_week.max() - data.first_week.min()
)
data = pd.concat(
    [data, pd.get_dummies(data.SmokingStatus, prefix="SmokingStatus")], axis=1
)
data = pd.concat([data, pd.get_dummies(data.Sex, prefix="Sex")], axis=1)

print(len(data))

train = data.loc[data.Split == "train"].copy()
subms = data.loc[data.Split == "subm"].copy()
test = data.loc[data.Split == "test"].copy()

f, axes = plt.subplots(1, 2, figsize=(7, 7), sharey=True, sharex=True)
sns.distplot(
    train.Age.values, color="g", kde=False, kde_kws={"shade": True}, ax=axes[0]
).set_title("Training age")
sns.distplot(
    subms.Age.values, color="g", kde=False, kde_kws={"shade": True}, ax=axes[1]
).set_title("Submission age")
plt.legend()
plt.show()




## === cell 1
class OSICDataSet(Dataset):
    def __init__(self, data, mode="train"):
        self.data = data
        self.data["Smoke"] = self.data.SmokingStatus.replace(
            {"Ex-smoker": 0.5, "Never smoked": 0, "Currently smokes": 1}
        )
        self.data["Gender"] = self.data.Sex.replace({"Male": 1, "Female": 0})
        self.data["min_FVC"] = (self.data.min_FVC - data.min_FVC.min()) / (
            data.min_FVC.max() - data.min_FVC.min()
        )
        self.data["WeekIn"] = (self.data.first_week - data.first_week.min()) / (
            data.first_week.max() - data.first_week.min()
        )
        self.data["AgeIn"] = (self.data.Age - data.Age.min()) / (
            data.Age.max() - data.Age.min()
        )
        self.data["PercentIn"] = (self.data.Percent - data.Percent.min()) / (
            data.Percent.max() - data.Percent.min()
        )
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        otherData = pt.from_numpy(self.data.loc[idx, inputs].values.astype(np.float32))
        if self.mode == "train":
            targets = pt.from_numpy(
                self.data.loc[idx, ["FVC"]].values.astype(np.float32)
            )
            return otherData, targets
        return otherData  # TODO: Add image data


class Model(pt.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.start = pt.nn.Sequential(
            pt.nn.Linear(len(inputs), 100),
            pt.nn.ReLU(),
            pt.nn.Linear(100, 100),
            pt.nn.ReLU(),
        )
        self.left = pt.nn.Sequential(pt.nn.Linear(100, 100))
        self.sigmoid = pt.nn.Sigmoid()
        self.right = pt.nn.Sequential(pt.nn.Linear(100, 100))
        self.last = pt.nn.Sequential(
            pt.nn.Linear(100, 3),
        )
        self.lastRelu = pt.nn.Sequential(pt.nn.Linear(100, 3), pt.nn.ReLU())

    def forward(self, x):
        h = self.start(x)
        p1 = self.last(h)
        p2 = self.lastRelu(h)
        out = p1 + pt.cumsum(p2, 1)
        return out


model = Model()

C1, C2 = pt.FloatTensor([70]).to(device), pt.FloatTensor([1000]).to(device)


def score(y_pred, y_true):
    sigma = y_pred[:, 2] - y_pred[:, 0]
    fvc_pred = y_pred[:, 1]

    sigma_clip = pt.max(sigma, C1.expand_as(sigma))
    delta = pt.abs(y_true - fvc_pred)
    delta = pt.min(delta, C2)
    sq2 = pt.sqrt(pt.FloatTensor([2]).to(y_pred.device))
    metric = (delta / sigma_clip) * sq2 + pt.log(sigma_clip * sq2)
    return pt.mean(metric)


def pinballLoss(pred, label, quant):
    err = label - pred
    m = pt.mean(pt.max(quant * err, (quant - 1) * err))
    return m


def loss(pred, label):
    quantiles = pt.FloatTensor([0.2, 0.5, 0.8]).to(pred.device)
    return 0.8 * pinballLoss(pred, label, quantiles) + 0.2 * score(pred, label)




## === cell 2
model = model.to(device)

trainDataSet = OSICDataSet(train)
optimizer = optim.Adam(model.parameters(), lr=0.075, weight_decay=0.1, eps=0.00)
losses = []
valLosses = []
kf = KFold(n_splits=5)
c = 0
submDataSet = OSICDataSet(subms, "submission")
inp = pt.from_numpy(submDataSet.data[inputs].values.astype(np.float32)).to(device)

for train_index, val_index in kf.split(trainDataSet):
    for i in range(100):
        train_x, train_y = trainDataSet[train_index]
        val_x, val_y = trainDataSet[val_index]

        train_x = train_x.to(device)
        train_y = train_y.to(device)
        val_x = val_x.to(device)
        val_y = val_y.to(device)

        optimizer.zero_grad()
        pred = model(train_x)
        los = loss(pred, train_y)
        los.backward()
        optimizer.step()
        if c % 10 == 0:
            pred = model(val_x)
            valLos = loss(pred, val_y)
            losses.append(los)
            valLosses.append(valLos)
        c += 1


## === cell 3
sns.set(style="white", palette="muted", color_codes=True)

plot_losses = [float(l.detach().cpu()) if pt.is_tensor(l) else float(l) for l in losses]
plot_valLosses = [
    float(l.detach().cpu()) if pt.is_tensor(l) else float(l) for l in valLosses
]

plt.plot(plot_losses, label="Train loss")
plt.plot(plot_valLosses, label="Val loss")
plt.legend()
plt.show()

asd = pt.from_numpy(trainDataSet.data[inputs].values.astype(np.float32))
pred = model(asd).detach()
idxs = np.random.randint(0, len(trainDataSet), 100)
plt.plot(
    trainDataSet.data.loc[idxs, ["FVC"]].values.astype(np.float32), label="ground truth"
)
plt.plot(pred[idxs, 0], label="q25")
plt.plot(pred[idxs, 1], label="q50")
plt.plot(pred[idxs, 2], label="q75")
plt.legend(loc="best")
plt.show()
c = pred[:, 2] - pred[:, 0]
f = pred[:, 1]


fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
sns.distplot(c, color="g", kde=False, kde_kws={"shade": True}, ax=axes[0]).set_title(
    "Predicted Confidence on train set"
)
sns.distplot(f, color="g", kde=False, kde_kws={"shade": True}, ax=axes[1]).set_title(
    "Predicted FVC on train set"
)
plt.show()

submDataSet = OSICDataSet(subms, "submission")
inp = pt.from_numpy(submDataSet.data[inputs].values.astype(np.float32))
out = model(inp).detach()
confidence = out[:, 2] - out[:, 0]
fvc = out[:, 1]
fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
sns.distplot(
    confidence, color="g", kde=False, kde_kws={"shade": True}, ax=axes[0]
).set_title("Confidence on submission set")
sns.distplot(fvc, color="g", kde=False, kde_kws={"shade": True}, ax=axes[1]).set_title(
    "FVC on submision set"
)
plt.show()

submDataSet.data["pFVC"] = fvc
submDataSet.data["cConf"] = confidence

confidence_subm = pt.clamp(confidence, min=70.0).cpu().numpy().astype(np.float32)
fvc_subm = fvc.cpu().numpy().astype(np.float32)

submission = pd.DataFrame(
    {
        "Patient_Week": submDataSet.data.Patient_Week.values,
        "FVC": fvc_subm,
        "Confidence": confidence_subm,
    }
)

otest = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/test.csv")
for i in range(len(otest)):
    submission.loc[
        submission["Patient_Week"] == otest.Patient[i] + "_" + str(otest.Weeks[i]),
        "FVC",
    ] = otest.FVC[i]
    submission.loc[
        submission["Patient_Week"] == otest.Patient[i] + "_" + str(otest.Weeks[i]),
        "Confidence",
    ] = 70.0  # keep consistent with metric clipping floor

submission.to_csv("submission.csv", index=False)
null_columns = submission.columns[submission.isnull().any()]
print(submission.head())
print(len(submission))

## --- ERROR in cell 3, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mRuntimeError[0m                              Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2053399746.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     12[0m [0;34m[0m[0m
[1;32m     13[0m [0masd[0m [0;34m=[0m [0mpt[0m[0;34m.[0m[0mfrom_numpy[0m[0;34m([0m[0mtrainDataSet[0m[0;34m.[0m[0mdata[0m[0;34m[[0m[0minputs[0m[0;34m][0m[0;34m.[0m[0mvalues[0m[0;34m.[0m[0mastype[0m[0;34m([0m[0mnp[0m[0;34m.[0m[0mfloat32[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 14[0;31m [0mpred[0m [0;34m=[0m [0mmodel[0m[0;34m([0m[0masd[0m[0;34m)[0m[0;34m.[0m[0mdetach[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     15[0m [0midxs[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0mrandom[0m[0;34m.[0m[0mrandint[0m[0;34m([0m[0;36m0[0m[0;34m,[0m [0mlen[0m[0;34m([0m[0mtrainDataSet[0m[0;34m)[0m[0;34m,[0m [0;36m100[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     16[0m plt.plot(

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

[0;32m/tmp/ipykernel_11/3413043759.py[0m in [0;36mforward[0;34m(self, x)[0m
[1;32m     51[0m [0;34m[0m[0m
[1;32m     52[0m     [0;32mdef[0m [0mforward[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mx[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 53[0;31m         [0mh[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mstart[0m[0;34m([0m[0mx[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     54[0m         [0mp1[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mlast[0m[0;34m([0m[0mh[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     55[0m         [0mp2[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mlastRelu[0m[0;34m([0m[0mh[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

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

[0;31mRuntimeError[0m: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! (when checking argument for argument mat1 in method wrapper_CUDA_addmm)
