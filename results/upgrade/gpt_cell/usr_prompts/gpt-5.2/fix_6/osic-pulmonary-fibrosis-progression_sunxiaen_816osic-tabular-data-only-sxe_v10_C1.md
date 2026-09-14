# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

-7.1255

# 6. Current score

-24.65924

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved -24.65924) has done: 'Diagnosis: Cell 31 fails immediately because it expects a trained `model` or a list `models` to already exist in the notebook state, but the provided cells do not define either before inference. Since inference can proceed with a deterministically initialized model when no checkpoint is available, we can unblock execution by instantiating `QuantModel` and wrapping it in `models`. We also need to ensure `batch_size` exists (it is referenced but not defined in the provided cells), otherwise a second NameError occur.

Patch summary: In cell 31, replace the hard failure with a minimal fallback that (1) defines `batch_size` if missing, (2) creates a `QuantModel` instance if neither `model` nor `models` exists, and (3) moves model(s) to the selected device and switches them to `eval()` before generating predictions. This preserves the existing inference loop and output format unchanged.

Updated cells: Only cell 31 is modified.

Compatibility notes for cell k+1: Cell 31 is the last provided cell; the patch keeps the variables `models`, `avg_preds`, and `df` semantics unchanged and still writes `submission.csv` in the same format.

Assumptions: No trained checkpoint path/filename is available in the provided cells; therefore the safest minimal unblock is to run inference with a freshly initialized `QuantModel`. If a trained `model` or `models` is defined earlier in the full notebook, this patch continue to use it unchanged.'

# 9. Code solution

## === cell 0
import copy
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Subset

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR


class SummaryWriter:  # noqa: N801
    def __init__(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

    def add_histogram(self, *args, **kwargs):
        pass

    def add_image(self, *args, **kwargs):
        pass

    def add_text(self, *args, **kwargs):
        pass

    def flush(self):
        pass

    def close(self):
        pass


from tqdm.notebook import trange
from time import time

root_dir = Path("/kaggle/input/osic-pulmonary-fibrosis-progression")
model_dir = Path("/kaggle/working")


## === cell 17
class ClinicalDataset(Dataset):
    def __init__(self, root_dir, mode, transform=None):
        self.transform = transform   #tabular的东西transform先挂着，感觉一般不用
        self.mode = mode             #train or test
        tst = pd.read_csv(Path(root_dir)/"test.csv")#test_data
        tst['minweek']=tst['Weeks']
        tst['minweek_FVC']=tst['FVC']
        sub = pd.read_csv(Path(root_dir)/"sample_submission.csv")
        sub['Patient'] = sub['Patient_Week'].apply(lambda x: x.split('_')[0])
        sub['Weeks'] = sub['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))
        sub = sub[['Patient', 'Weeks', 'Confidence', 'Patient_Week']]
        sub = sub.merge(tst.drop('Weeks', axis=1), on="Patient",how="inner")#样子就是之前单元格中的样子
        sub['base_week']=sub['Weeks']-sub['minweek']

        tr = pd.read_csv(Path(root_dir)/"train.csv")#train_data
        tr.drop_duplicates(keep="first", inplace=True, subset=['Patient', 'Weeks']) #暂且留其中一个把 keep=“first”
        tst = pd.read_csv(Path(root_dir)/"test.csv")#test_data
        tr['minweek']=tr.groupby('Patient')['Weeks'].transform(min)
        base = tr.loc[tr.Weeks == tr.minweek] #找基准
        base = base[['Patient', 'FVC']]
        base.columns = ['Patient', 'minweek_FVC']#改名
        tr=tr.merge(base, on='Patient', how='inner')#train部分数据不脏，left和inner都没问题
        tr['base_week']=tr['Weeks']-tr['minweek']
        data=tr.append(sub)
        COLS = ['Sex', 'SmokingStatus']
        self.FE = []
        for col in COLS:
            for mod in data[col].unique():
                self.FE.append(mod)
                data[mod] = (data[col] == mod).astype(int)
    
        data['N_age'] = (data['Age'] - data['Age'].min()) / \
                      (data['Age'].max() - data['Age'].min())
        data['N_base_minweek_FVC'] = (data['minweek_FVC'] - data['minweek_FVC'].min()) / \
                       (data['minweek_FVC'].max() - data['minweek_FVC'].min())
        
        data['N_week'] = (data['base_week'] - data['base_week'].min()) / \
                       (data['base_week'].max() - data['base_week'].min())
        
        data['N_percent'] = (data['Percent'] - data['Percent'].min()) / \
                          (data['Percent'].max() - data['Percent'].min())
        
        self.FE += ['N_age', 'N_percent', 'N_week', 'N_base_minweek_FVC']
        
        if self.mode=='train':
            self.raw = data[data['Patient_Week'].isna()].reset_index()
        elif self.mode=='test':
            self.raw =data[~data['Patient_Week'].isna()].reset_index()
        
        del base
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
    def group_split(self, test_size=0.1):
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size)
        groups = self.raw['Patient']
        idx = list(gss.split(self.raw, self.raw, groups))
        train = Subset(self, idx[0][0])
        val = Subset(self, idx[0][1])
        return train, val


## === cell 20
import torch
import torch.nn as nn
import torch.nn.functional as F
class QuantModel(nn.Module):
    def __init__(self, in_tabular_features=9, out_quantiles=3):
        super(QuantModel, self).__init__()
        self.fc1 = nn.Linear(in_tabular_features, 100)
        self.fc2 = nn.Linear(100, 80)
        self.fc3 = nn.Linear(80, 3)
        self.fc4 = nn.Linear(3,out_quantiles)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.fc4(x)
        return x

quantiles = (0.2, 0.5, 0.8)
def quantile_loss(preds, target, quantiles):
    assert not target.requires_grad
    assert preds.size(0) == target.size(0)
    losses = []
    for i, q in enumerate(quantiles):
        errors = target - preds[:, i]
        losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))#添加维度   （x,1）
    loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
    return loss


## === cell 27
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


## === cell 28
class ClinicalDataset(Dataset):
    def __init__(self, root_dir, mode, transform=None):
        self.transform = transform  # tabular的东西transform先挂着，感觉一般不用
        self.mode = mode  # train or test
        tst = pd.read_csv(Path(root_dir) / "test.csv")  # test_data
        tst["minweek"] = tst["Weeks"]
        tst["minweek_FVC"] = tst["FVC"]
        sub = pd.read_csv(Path(root_dir) / "sample_submission.csv")
        sub["Patient"] = sub["Patient_Week"].apply(lambda x: x.split("_")[0])
        sub["Weeks"] = sub["Patient_Week"].apply(lambda x: int(x.split("_")[-1]))
        sub = sub[["Patient", "Weeks", "Confidence", "Patient_Week"]]
        sub = sub.merge(
            tst.drop("Weeks", axis=1), on="Patient", how="inner"
        )  # 样子就是之前单元格中的样子
        sub["base_week"] = sub["Weeks"] - sub["minweek"]

        tr = pd.read_csv(Path(root_dir) / "train.csv")  # train_data
        tr.drop_duplicates(
            keep="first", inplace=True, subset=["Patient", "Weeks"]
        )  # 暂且留其中一个把 keep=“first”
        tst = pd.read_csv(Path(root_dir) / "test.csv")  # test_data
        tr["minweek"] = tr.groupby("Patient")["Weeks"].transform(min)
        base = tr.loc[tr.Weeks == tr.minweek]  # 找基准
        base = base[["Patient", "FVC"]]
        base.columns = ["Patient", "minweek_FVC"]  # 改名
        tr = tr.merge(
            base, on="Patient", how="inner"
        )  # train部分数据不脏，left和inner都没问题
        tr["base_week"] = tr["Weeks"] - tr["minweek"]

        data = pd.concat([tr, sub], ignore_index=True, sort=False)

        COLS = ["Sex", "SmokingStatus"]
        self.FE = []
        for col in COLS:
            for mod in data[col].unique():
                self.FE.append(mod)
                data[mod] = (data[col] == mod).astype(int)

        data["N_age"] = (data["Age"] - data["Age"].min()) / (
            data["Age"].max() - data["Age"].min()
        )
        data["N_base_minweek_FVC"] = (
            data["minweek_FVC"] - data["minweek_FVC"].min()
        ) / (data["minweek_FVC"].max() - data["minweek_FVC"].min())

        data["N_week"] = (data["base_week"] - data["base_week"].min()) / (
            data["base_week"].max() - data["base_week"].min()
        )

        data["N_percent"] = (data["Percent"] - data["Percent"].min()) / (
            data["Percent"].max() - data["Percent"].min()
        )

        self.FE += ["N_age", "N_percent", "N_week", "N_base_minweek_FVC"]

        if self.mode == "train":
            self.raw = data[data["Patient_Week"].isna()].reset_index()
        elif self.mode == "test":
            self.raw = data[~data["Patient_Week"].isna()].reset_index()

        del base
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

    def group_split(self, test_size=0.1):
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size)
        groups = self.raw["Patient"]
        idx = list(gss.split(self.raw, self.raw, groups))
        train = Subset(self, idx[0][0])
        val = Subset(self, idx[0][1])
        return train, val


## === cell 31
if "batch_size" not in globals() or batch_size is None:
    batch_size = 32

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if "models" not in globals() or models is None:
    if "model" in globals() and model is not None:
        models = [model]
    else:
        model = QuantModel().to(device)
        models = [model]

data = ClinicalDataset(root_dir, mode="test")
avg_preds = np.zeros((len(data), len(quantiles)))

for model in models:
    model = model.to(device)
    model.eval()

    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=2)
    preds = []
    for batch in dataloader:
        inputs = batch["features"].float().to(device)
        with torch.no_grad():
            x = model(inputs)
            preds.append(x)

    preds = torch.cat(preds, dim=0).cpu().numpy()
    avg_preds += preds

avg_preds /= len(models)
df = pd.DataFrame(data=avg_preds, columns=list(quantiles))
df["Patient_Week"] = data.raw["Patient_Week"]
df["FVC"] = df[quantiles[1]]
df["Confidence"] = df[quantiles[2]] - df[quantiles[0]]
df = df.drop(columns=list(quantiles))
df.to_csv("submission.csv", index=False)
