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

0.52964

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.52964) has done: 'I remove the incompatible `imblearn` import (it fails due to your sklearn version) and also stop relying on optional pip installs, since the needed packages are already present. Then I fix the train/val split + scaling pipeline so the `StandardScaler` is fit on train and applied to val/test (the current code incorrectly fits separately, hurting AUC and causing leakage/instability). Finally I fix dataset/test inference bugs (wrong tensor returned, double-sigmoid, and missing `.to(device)`), ensure models are trained on GPU/CPU consistently, and write a valid `submission.csv` with columns `id,EC1,EC2` and correct row alignment.'

# 9. Code solution

## === cell 1
import os
import random
import warnings
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torchmetrics.classification import BinaryAccuracy

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)



## === cell 2
ROOT_PATH = "/kaggle/input/playground-series-s3e18"

train_path = os.path.join(ROOT_PATH, "train.csv")
test_path = os.path.join(ROOT_PATH, "test.csv")
sample_path = os.path.join(ROOT_PATH, "sample_submission.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_path)

print(train_df.shape, test_df.shape, sample_sub.shape)
print("Sample submission columns:", list(sample_sub.columns))




## === cell 3
class Datapreparation(object):
    def __init__(self, root_path):
        self.root_path = root_path

    def get_dataframe(self, filename):
        return pd.read_csv(os.path.join(self.root_path, filename))

    def random_split_data(self, X, y):
        return train_test_split(X, y, test_size=0.20, random_state=SEED, stratify=y)


data = Datapreparation(ROOT_PATH)



## === cell 4
y1 = train_df["EC1"].astype(int)
y2 = train_df["EC2"].astype(int)

drop_cols = ["id", "EC1", "EC2", "EC3", "EC4", "EC5", "EC6"]
X = train_df.drop(columns=drop_cols).copy()
X_test = test_df.drop(columns=["id"]).copy()

print("X:", X.shape, "X_test:", X_test.shape)



## === cell 5
X_train_1, X_val_1, y1_train, y1_val = data.random_split_data(X, y1)
X_train_2, X_val_2, y2_train, y2_val = data.random_split_data(X, y2)

print("EC1 split:", X_train_1.shape, X_val_1.shape, y1_train.shape, y1_val.shape)
print("EC2 split:", X_train_2.shape, X_val_2.shape, y2_train.shape, y2_val.shape)



## === cell 6
scaler1 = StandardScaler()
std_X_train_1 = scaler1.fit_transform(X_train_1)
std_X_val_1 = scaler1.transform(X_val_1)
std_X_test_1 = scaler1.transform(X_test)

scaler2 = StandardScaler()
std_X_train_2 = scaler2.fit_transform(X_train_2)
std_X_val_2 = scaler2.transform(X_val_2)
std_X_test_2 = scaler2.transform(X_test)

print(std_X_train_1.shape, std_X_val_1.shape, std_X_test_1.shape)




## === cell 7
class Tensoroperations:
    def convert_to_tensor(self, X, y):
        X_tensor = torch.from_numpy(np.asarray(X)).float()
        y_tensor = torch.from_numpy(np.asarray(y)).float()
        return X_tensor, y_tensor

    def convert_to_test_tensor(self, X):
        return torch.from_numpy(np.asarray(X)).float()


tenops = Tensoroperations()

X_tensor_train_1, y1_tensor_train = tenops.convert_to_tensor(
    std_X_train_1, y1_train.values
)
X_tensor_val_1, y1_tensor_val = tenops.convert_to_tensor(std_X_val_1, y1_val.values)

X_tensor_train_2, y2_tensor_train = tenops.convert_to_tensor(
    std_X_train_2, y2_train.values
)
X_tensor_val_2, y2_tensor_val = tenops.convert_to_tensor(std_X_val_2, y2_val.values)

X_tensor_test_1 = tenops.convert_to_test_tensor(std_X_test_1)
X_tensor_test_2 = tenops.convert_to_test_tensor(std_X_test_2)

print("EC1 train tensors:", X_tensor_train_1.shape, y1_tensor_train.shape)




## === cell 8
class CustomDataset(Dataset):
    def __init__(self, X_data, y_data):
        super().__init__()
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


train1_dataset = CustomDataset(X_tensor_train_1, y1_tensor_train)
val1_dataset = CustomDataset(X_tensor_val_1, y1_tensor_val)

train2_dataset = CustomDataset(X_tensor_train_2, y2_tensor_train)
val2_dataset = CustomDataset(X_tensor_val_2, y2_tensor_val)



## === cell 9
from torch.utils.data.sampler import WeightedRandomSampler


def make_weighted_sampler(y_series_or_array):
    y_arr = np.asarray(y_series_or_array, dtype=int)
    counts = np.bincount(y_arr, minlength=2)
    counts = np.maximum(counts, 1)
    labels_weights = 1.0 / counts
    weights = labels_weights[y_arr]
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


sampler_train_EC1 = make_weighted_sampler(y1_train.values)
sampler_val_EC1 = make_weighted_sampler(y1_val.values)

sampler_train_EC2 = make_weighted_sampler(y2_train.values)
sampler_val_EC2 = make_weighted_sampler(y2_val.values)

train_dataloader_1 = DataLoader(
    train1_dataset, batch_size=64, sampler=sampler_train_EC1
)
val_dataloader_1 = DataLoader(val1_dataset, batch_size=64, sampler=sampler_val_EC1)

train_dataloader_2 = DataLoader(
    train2_dataset, batch_size=64, sampler=sampler_train_EC2
)
val_dataloader_2 = DataLoader(val2_dataset, batch_size=64, sampler=sampler_val_EC2)



## === cell 10
false_pos, true_pos = [], []


class EnzymeClassificationBase(torch.nn.Module):
    def _accuracy(self, outputs, labels):
        metric = BinaryAccuracy()
        return metric(outputs, labels.unsqueeze(1))

    def training_step(self, batch):
        features, labels = batch
        features = features.to(device)
        labels = labels.to(device)
        out = self(features)
        loss = F.binary_cross_entropy(out, labels.unsqueeze(1))
        return loss

    def validation_step(self, batch):
        features, labels = batch
        features = features.to(device)
        labels = labels.to(device)
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




## === cell 11
n_input_dim = X.shape[1]
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
            torch.nn.Sigmoid(),
        )

    def forward(self, xb):
        return self.network(xb)


model_EC1 = MultiClassificationNN().to(device)
model_EC2 = MultiClassificationNN().to(device)

print(model_EC1)




## === cell 12
class Trainer:
    @torch.no_grad()
    def _evaluate(self, model, val_loader):
        model.eval()
        outputs = [model.validation_step(batch) for batch in val_loader]
        return model.validation_epoch_end(outputs)

    def fit(self, epochs, lr, model, train_loader, val_loader, opt_func):
        history = []
        optimizer = opt_func(model.parameters(), lr=lr, weight_decay=2e-5)

        for epoch in range(epochs):
            model.train()
            train_losses = []
            for batch in train_loader:
                loss = model.training_step(batch)
                train_losses.append(loss.detach().cpu())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            result = self._evaluate(model, val_loader)
            result["Train_loss"] = torch.stack(train_losses).mean().item()
            model.epoch_end(epoch, result)
            history.append(result)

        return history




## === cell 13
num_epochs = 100
lr = 1e-4
opt_func = torch.optim.Adam

trainer = Trainer()

history_EC1 = trainer.fit(
    num_epochs, lr, model_EC1, train_dataloader_1, val_dataloader_1, opt_func
)
history_EC2 = trainer.fit(
    num_epochs, lr, model_EC2, train_dataloader_2, val_dataloader_2, opt_func
)




## --- ERROR in cell 13, traceback:
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
/tmp/ipykernel_11/1988541098.py in <cell line: 0>()
      5 trainer = Trainer()
      6 
----> 7 history_EC1 = trainer.fit(
      8     num_epochs, lr, model_EC1, train_dataloader_1, val_dataloader_1, opt_func
      9 )

/tmp/ipykernel_11/550608909.py in fit(self, epochs, lr, model, train_loader, val_loader, opt_func)
     20                 optimizer.zero_grad()
     21 
---> 22             result = self._evaluate(model, val_loader)
     23             result["Train_loss"] = torch.stack(train_losses).mean().item()
     24             model.epoch_end(epoch, result)

/usr/local/lib/python3.11/dist-packages/torch/utils/_contextlib.py in decorate_context(*args, **kwargs)
    114     def decorate_context(*args, **kwargs):
    115         with ctx_factory():
--> 116             return func(*args, **kwargs)
    117 
    118     return decorate_context

/tmp/ipykernel_11/550608909.py in _evaluate(self, model, val_loader)
      3     def _evaluate(self, model, val_loader):
      4         model.eval()
----> 5         outputs = [model.validation_step(batch) for batch in val_loader]
      6         return model.validation_epoch_end(outputs)
      7 

/tmp/ipykernel_11/550608909.py in <listcomp>(.0)
      3     def _evaluate(self, model, val_loader):
      4         model.eval()
----> 5         outputs = [model.validation_step(batch) for batch in val_loader]
      6         return model.validation_epoch_end(outputs)
      7 

/tmp/ipykernel_11/1154058216.py in validation_step(self, batch)
     22         out = self(features)
     23         loss = F.binary_cross_entropy(out, labels.unsqueeze(1))
---> 24         acc = self._accuracy(out, labels)
     25         return {"Validation_loss": loss.detach(), "Validation_acc": acc.detach()}
     26 

/tmp/ipykernel_11/1154058216.py in _accuracy(self, outputs, labels)
      6         metric = BinaryAccuracy()
      7         # outputs expected shape: [B,1], labels: [B]
----> 8         return metric(outputs, labels.unsqueeze(1))
      9 
     10     def training_step(self, batch):

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

## === cell 14
class CustomDataTest(Dataset):
    def __init__(self, X_data):
        self.X_data = X_data

    def __getitem__(self, index):
        return self.X_data[index]

    def __len__(self):
        return len(self.X_data)


test_loader_1 = DataLoader(
    CustomDataTest(X_tensor_test_1), batch_size=256, shuffle=False
)
test_loader_2 = DataLoader(
    CustomDataTest(X_tensor_test_2), batch_size=256, shuffle=False
)




## === cell 15
class Evaluate:
    def eval_test_data(self, model, test_data_dl):
        preds = []
        model.eval()
        with torch.no_grad():
            for X_batch_test in test_data_dl:
                X_batch_test = X_batch_test.to(device)
                y_pred = model(X_batch_test).squeeze(1)
                preds.append(y_pred.detach().cpu().numpy())
        return np.concatenate(preds, axis=0)


eva = Evaluate()

ec1_pred = eva.eval_test_data(model_EC1, test_loader_1)
ec2_pred = eva.eval_test_data(model_EC2, test_loader_2)

print(ec1_pred.shape, ec2_pred.shape, "min/max EC1:", ec1_pred.min(), ec1_pred.max())



## === cell 16
sub = sample_sub.copy()
sub["EC1"] = ec1_pred
sub["EC2"] = ec2_pred

sub["id"] = test_df["id"].values
assert sub.shape[0] == test_df.shape[0]
assert list(sub.columns) == ["id", "EC1", "EC2"]

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
sub.head()
