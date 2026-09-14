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

0.64128

# 6. Current score

0.54077

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.54077) has done: 'I fix the import/runtime failures by removing incompatible/unused dependencies (the imblearn import is broken with the installed scikit-learn) and consolidating all required imports so later cells don’t lose symbols like `torch`, `Dataset`, `DataLoader`, `train_test_split`, and `StandardScaler`. I also fix critical logic bugs that prevent valid inference/submission: the scaler must be fit on train and reused for val/test (previously it refit each time), the test-time sigmoid was applied twice, and the model/data were never moved to the same device. Finally, I keep your model architecture and training loop intact but make it run end-to-end and write a correct `submission.csv` with columns `id,EC1,EC2`.'

# 9. Code solution

## === cell 0
import sys, os



## === cell 1
pass



## === cell 2
pass



## === cell 3
pass



## === cell 4
import random
import numpy as np
import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from torchmetrics.classification import BinaryAccuracy


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)



## === cell 5
for dirname, _, filenames in os.walk("/kaggle/input/"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 6
class Datapreparation(object):
    def __init__(self, root_path):
        self.root_path = root_path
        self.scaler = None  # will be fit on train only

    def get_dataframe(self, filename):
        return pd.read_csv(os.path.join(self.root_path, filename))

    def summary(self, text, df):
        summary = pd.DataFrame(df.dtypes, columns=["dtypes"])
        summary["null"] = df.isnull().sum()
        summary["unique"] = df.nunique()
        num_df = df.select_dtypes(include=[np.number])
        if num_df.shape[1] > 0:
            summary["min"] = num_df.min()
            summary["median"] = num_df.median()
            summary["max"] = num_df.max()
            summary["mean"] = num_df.mean()
            summary["std"] = num_df.std()
        else:
            summary["min"] = np.nan
            summary["median"] = np.nan
            summary["max"] = np.nan
            summary["mean"] = np.nan
            summary["std"] = np.nan
        summary["duplicate"] = df.duplicated().sum()
        return summary

    def random_split_data(self, X, y):
        return train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    def fit_standardization(self, X_train):
        self.scaler = StandardScaler()
        return self.scaler.fit_transform(X_train)

    def transform_standardization(self, X_data):
        if self.scaler is None:
            raise RuntimeError(
                "Scaler not fit. Call fit_standardization on training data first."
            )
        return self.scaler.transform(X_data)


data = Datapreparation("/kaggle/input/playground-series-s3e18")
train = data.get_dataframe("train.csv")



## === cell 7
data.summary("train", train).head()



## === cell 8
y1 = train["EC1"].astype(np.float32)
y2 = train["EC2"].astype(np.float32)
train.drop(
    columns=["id", "EC1", "EC2", "EC3", "EC4", "EC5", "EC6"], axis=1, inplace=True
)
X = train.copy()



## === cell 9
X_train, X_val, y1_train, y1_val = data.random_split_data(X, y1)
print(
    "Data splits for EC1 \n", X_train.shape, X_val.shape, y1_train.shape, y1_val.shape
)



## === cell 10
_, _, y2_train, y2_val = train_test_split(
    X, y2, test_size=0.20, random_state=42, stratify=y2
)
print(
    "Data splits for EC2 \n", X_train.shape, X_val.shape, y2_train.shape, y2_val.shape
)



## === cell 11
std_X_train = data.fit_standardization(X_train)
std_X_val = data.transform_standardization(X_val)
print(std_X_train[0], std_X_val[0])




## === cell 12
class Tensoroperations:
    def __init__(self):
        super().__init__()

    def convert_to_tensor(self, X, y=None):
        X_tensor = torch.from_numpy(np.asarray(X)).float()
        if y is None:
            return X_tensor
        y_tensor = torch.from_numpy(np.asarray(y)).float()
        return X_tensor, y_tensor

    def convert_to_test_tensor(self, X):
        X_tensor = torch.from_numpy(np.asarray(X)).float()
        return X_tensor


tenops = Tensoroperations()




## === cell 13
class CustomDataset(Dataset):
    def __init__(self, X_data, y_data):
        super().__init__()
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return (self.X_data[index], self.y_data[index])

    def __len__(self):
        return len(self.X_data)




## === cell 14
X_tensor_train, y1_tensor_train = tenops.convert_to_tensor(
    std_X_train, y1_train.values.astype(np.float32)
)
X_tensor_val, y1_tensor_val = tenops.convert_to_tensor(
    std_X_val, y1_val.values.astype(np.float32)
)
print("The training tensor for EC1\n", X_tensor_train.shape, y1_tensor_train.shape)
print("The validation tensor for EC1\n", X_tensor_val.shape, y1_tensor_val.shape)



## === cell 15
_, y2_tensor_train = tenops.convert_to_tensor(
    std_X_train, y2_train.values.astype(np.float32)
)
_, y2_tensor_val = tenops.convert_to_tensor(std_X_val, y2_val.values.astype(np.float32))
print("The training tensor EC2\n", X_tensor_train.shape, y2_tensor_train.shape)
print("The validation tensor Ec2\n", X_tensor_val.shape, y2_tensor_val.shape)



## === cell 16
false_pos, true_pos = [], []




## === cell 17
class EnzymeClassificationBase(torch.nn.Module):
    def _accuracy(self, outputs, labels):
        metric = BinaryAccuracy()
        return metric(outputs, labels.unsqueeze(1))

    def training_step(self, batch):
        features, labels = batch
        out = self(features)
        loss = F.binary_cross_entropy(out, labels.unsqueeze(1))
        return loss

    def validation_step(self, batch):
        features, labels = batch
        out = self(features)
        loss = F.binary_cross_entropy(out, labels.unsqueeze(1))
        acc = self._accuracy(out, labels)
        return {"Validation_loss": loss.detach(), "Validation_acc": acc.detach()}

    def validation_epoch_end(self, outputs):
        batch_losses = [x["Validation_loss"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x["Validation_acc"] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {
            "Validation_loss": epoch_loss.item(),
            "Validation_acc": epoch_acc.item(),
        }

    def epoch_end(self, epoch, result):
        if epoch % 10 == 0:
            print(
                "Epoch [{}], Train_loss: {:.4f}, Validation_loss: {:.4f}, Validation_acc: {:.4f}".format(
                    epoch,
                    result["Train_loss"],
                    result["Validation_loss"],
                    result["Validation_acc"],
                )
            )




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
            torch.nn.Linear(8, 4),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(4, n_output),
            torch.nn.Sigmoid(),
        )

    def forward(self, xb):
        return self.network(xb)


model_EC1 = MultiClassificationNN().to(device)
model_EC2 = MultiClassificationNN().to(device)



## === cell 19
print(model_EC1)



## === cell 20
print(model_EC2)



## === cell 21
train1_dataset = CustomDataset(X_tensor_train, y1_tensor_train)
val1_dataset = CustomDataset(X_tensor_val, y1_tensor_val)

train2_dataset = CustomDataset(X_tensor_train, y2_tensor_train)
val2_dataset = CustomDataset(X_tensor_val, y2_tensor_val)



## === cell 22
train_dataloader = DataLoader(train1_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val1_dataset, batch_size=64, shuffle=False)



## === cell 23
train_dataloader2 = DataLoader(train2_dataset, batch_size=64, shuffle=True)
val_dataloader2 = DataLoader(val2_dataset, batch_size=64, shuffle=False)



## === cell 24
print("Using device:", device)




## === cell 25
class Trainer:
    @torch.no_grad()
    def _evaluate(self, model, val_loader):
        model.eval()
        outputs = []
        for batch in val_loader:
            features, labels = batch
            features = features.to(device)
            labels = labels.to(device)
            outputs.append(model.validation_step((features, labels)))
        return model.validation_epoch_end(outputs)

    def fit(self, epochs, lr, model, train_loader, val_loader, opt_func):
        history = []
        optimizer = opt_func(model.parameters(), lr=lr, weight_decay=2e-5)
        for epoch in range(epochs):
            model.train()
            train_losses = []
            for batch in train_loader:
                features, labels = batch
                features = features.to(device)
                labels = labels.to(device)

                loss = model.training_step((features, labels))
                train_losses.append(loss.detach())

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            result = self._evaluate(model, val_loader)
            result["Train_loss"] = torch.stack(train_losses).mean().item()
            model.epoch_end(epoch, result)
            history.append(result)
        return history




## === cell 26
num_epochs = 100
lr = 1e-4
opt_func = torch.optim.Adam



## === cell 27
tariner = Trainer()
history_EC1 = tariner.fit(
    num_epochs, lr, model_EC1, train_dataloader, val_dataloader, opt_func
)




## --- ERROR in cell 27, traceback:
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
/tmp/ipykernel_12/526435594.py in <cell line: 0>()
      1 tariner = Trainer()
----> 2 history_EC1 = tariner.fit(
      3     num_epochs, lr, model_EC1, train_dataloader, val_dataloader, opt_func
      4 )
      5 

/tmp/ipykernel_12/2397505151.py in fit(self, epochs, lr, model, train_loader, val_loader, opt_func)
     29                 optimizer.zero_grad()
     30 
---> 31             result = self._evaluate(model, val_loader)
     32             result["Train_loss"] = torch.stack(train_losses).mean().item()
     33             model.epoch_end(epoch, result)

/usr/local/lib/python3.11/dist-packages/torch/utils/_contextlib.py in decorate_context(*args, **kwargs)
    114     def decorate_context(*args, **kwargs):
    115         with ctx_factory():
--> 116             return func(*args, **kwargs)
    117 
    118     return decorate_context

/tmp/ipykernel_12/2397505151.py in _evaluate(self, model, val_loader)
      8             features = features.to(device)
      9             labels = labels.to(device)
---> 10             outputs.append(model.validation_step((features, labels)))
     11         return model.validation_epoch_end(outputs)
     12 

/tmp/ipykernel_12/1938062211.py in validation_step(self, batch)
     15         out = self(features)
     16         loss = F.binary_cross_entropy(out, labels.unsqueeze(1))
---> 17         acc = self._accuracy(out, labels)
     18         return {"Validation_loss": loss.detach(), "Validation_acc": acc.detach()}
     19 

/tmp/ipykernel_12/1938062211.py in _accuracy(self, outputs, labels)
      3         metric = BinaryAccuracy()
      4         # outputs are probabilities (sigmoid already in model)
----> 5         return metric(outputs, labels.unsqueeze(1))
      6 
      7     def training_step(self, batch):

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

## === cell 28
def plot_accuracies(history):
    accuracies = [x["Validation_acc"] for x in history]
    plt.figure(figsize=(7, 4))
    plt.plot(accuracies, "-x")
    plt.xlabel("Epoch")
    plt.ylabel("accuracy")
    plt.title("Accuracy vs. No. of epochs")
    plt.show()


plot_accuracies(history_EC1)




## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/1358727296.py in <cell line: 0>()
      9 
     10 
---> 11 plot_accuracies(history_EC1)
     12 
     13 

NameError: name 'history_EC1' is not defined

## === cell 29
def plot_losses(history):
    train_losses = [x.get("Train_loss") for x in history]
    val_losses = [x["Validation_loss"] for x in history]
    plt.figure(figsize=(7, 4))
    plt.plot(train_losses, "-bx")
    plt.plot(val_losses, "-rx")
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.legend(["Training", "Validation"])
    plt.title("Loss vs. No. of Epochs")
    plt.show()


plot_losses(history_EC1)



## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/2005594533.py in <cell line: 0>()
     12 
     13 
---> 14 plot_losses(history_EC1)
     15 

NameError: name 'history_EC1' is not defined

## === cell 30
fpr, tpr = [], []



## === cell 31
pass



## === cell 32
history_EC2 = tariner.fit(
    num_epochs, lr, model_EC2, train_dataloader2, val_dataloader2, opt_func
)



## --- ERROR in cell 32, traceback:
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
/tmp/ipykernel_12/501250328.py in <cell line: 0>()
----> 1 history_EC2 = tariner.fit(
      2     num_epochs, lr, model_EC2, train_dataloader2, val_dataloader2, opt_func
      3 )
      4 

/tmp/ipykernel_12/2397505151.py in fit(self, epochs, lr, model, train_loader, val_loader, opt_func)
     29                 optimizer.zero_grad()
     30 
---> 31             result = self._evaluate(model, val_loader)
     32             result["Train_loss"] = torch.stack(train_losses).mean().item()
     33             model.epoch_end(epoch, result)

/usr/local/lib/python3.11/dist-packages/torch/utils/_contextlib.py in decorate_context(*args, **kwargs)
    114     def decorate_context(*args, **kwargs):
    115         with ctx_factory():
--> 116             return func(*args, **kwargs)
    117 
    118     return decorate_context

/tmp/ipykernel_12/2397505151.py in _evaluate(self, model, val_loader)
      8             features = features.to(device)
      9             labels = labels.to(device)
---> 10             outputs.append(model.validation_step((features, labels)))
     11         return model.validation_epoch_end(outputs)
     12 

/tmp/ipykernel_12/1938062211.py in validation_step(self, batch)
     15         out = self(features)
     16         loss = F.binary_cross_entropy(out, labels.unsqueeze(1))
---> 17         acc = self._accuracy(out, labels)
     18         return {"Validation_loss": loss.detach(), "Validation_acc": acc.detach()}
     19 

/tmp/ipykernel_12/1938062211.py in _accuracy(self, outputs, labels)
      3         metric = BinaryAccuracy()
      4         # outputs are probabilities (sigmoid already in model)
----> 5         return metric(outputs, labels.unsqueeze(1))
      6 
      7     def training_step(self, batch):

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

## === cell 33
plot_accuracies(history_EC2)



## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/1564203759.py in <cell line: 0>()
----> 1 plot_accuracies(history_EC2)
      2 

NameError: name 'history_EC2' is not defined

## === cell 34
plot_losses(history_EC2)



## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/2034444355.py in <cell line: 0>()
----> 1 plot_losses(history_EC2)
      2 

NameError: name 'history_EC2' is not defined

## === cell 35
pass



## === cell 36
pass



## === cell 37
test = data.get_dataframe("test.csv")



## === cell 38
data.summary("test", test).head()



## === cell 39
test_update = test.loc[:, test.columns != "id"]



## === cell 40
std_X_test = data.transform_standardization(test_update)
std_X_test[0]



## === cell 41
X_tensor_test = tenops.convert_to_test_tensor(std_X_test)
X_tensor_test[0]




## === cell 42
class CustomDataTest(Dataset):
    def __init__(self, X_data):
        self.X_data = X_data

    def __getitem__(self, index):
        return self.X_data[index]

    def __len__(self):
        return len(self.X_data)




## === cell 43
test_dataset = CustomDataTest(X_tensor_test)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)




## === cell 44
class Evaluate:
    def eval_test_data(self, model, test_data_dl):
        preds = []
        model.eval()
        with torch.no_grad():
            for X_batch_test in test_data_dl:
                X_batch_test = X_batch_test.to(device)
                y_pred_prob = model(X_batch_test)
                preds.append(y_pred_prob.detach().cpu().numpy().reshape(-1))
        return np.concatenate(preds, axis=0)


eva = Evaluate()



## === cell 45
ec1 = eva.eval_test_data(model_EC1, test_dataloader)
ec2 = eva.eval_test_data(model_EC2, test_dataloader)
print(ec1.shape, ec2.shape)




## === cell 46
class Submit:
    def submit_predictions(self):
        df_submit = pd.DataFrame(
            {
                "id": test["id"].values,
                "EC1": ec1.astype(np.float64),
                "EC2": ec2.astype(np.float64),
            }
        )
        df_submit = df_submit[["id", "EC1", "EC2"]]
        df_submit.to_csv("submission.csv", index=False)
        print("Submission Completed!! -> submission.csv")
        return df_submit


submit = Submit()
df_submit = submit.submit_predictions()



## === cell 47
df_submit.head()
