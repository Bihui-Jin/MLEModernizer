# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import random
import warnings
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from torchmetrics.classification import BinaryAccuracy, BinaryROC

warnings.filterwarnings("ignore")
plt.style.use("fivethirtyeight")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device



## === cell 1
DATA_ROOT = "/kaggle/input/playground-series-s3e18"
if not os.path.exists(DATA_ROOT):
    alt = "/kaggle/input"
    if os.path.exists(os.path.join(alt, "playground-series-s3e18")):
        DATA_ROOT = os.path.join(alt, "playground-series-s3e18")
DATA_ROOT




## === cell 2
class Datapreparation(object):
    def __init__(self, root_path):
        self.root_path = root_path

    def get_dataframe(self, filename):
        return pd.read_csv(os.path.join(self.root_path, filename))

    def summary(self, text, df):
        summary = pd.DataFrame(df.dtypes, columns=["dtypes"])
        summary["null"] = df.isnull().sum()
        summary["unique"] = df.nunique()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            summary["min"] = df.min(numeric_only=True)
            summary["median"] = df.median(numeric_only=True)
            summary["max"] = df.max(numeric_only=True)
            summary["mean"] = df.mean(numeric_only=True)
            summary["std"] = df.std(numeric_only=True)
        summary["duplicate"] = df.duplicated().sum()
        return summary

    def random_split_data(self, X, y):
        return train_test_split(X, y, test_size=0.20, random_state=42)


data = Datapreparation(DATA_ROOT)
train = data.get_dataframe("train.csv")
test = data.get_dataframe("test.csv")
sample_sub = data.get_dataframe("sample_submission.csv")

(train.shape, test.shape, sample_sub.shape)



## === cell 3
assert "id" in train.columns and "id" in test.columns
assert {"EC1", "EC2"}.issubset(train.columns)
assert list(sample_sub.columns) == ["id", "EC1", "EC2"]
data.summary("train", train).head()



## === cell 4
y1 = train["EC1"].astype(np.float32)
y2 = train["EC2"].astype(np.float32)

target_cols = [c for c in train.columns if c.startswith("EC")]
X = train.drop(columns=["id"] + target_cols).copy()

X_test_df = test.drop(columns=["id"]).copy()
assert list(X.columns) == list(X_test_df.columns)

(X.shape, X_test_df.shape)



## === cell 5
idx = np.arange(len(X))
idx_train, idx_val = train_test_split(
    idx, test_size=0.20, random_state=42, shuffle=True
)

X_train = X.iloc[idx_train].reset_index(drop=True)
X_val = X.iloc[idx_val].reset_index(drop=True)

y1_train = y1.iloc[idx_train].reset_index(drop=True)
y1_val = y1.iloc[idx_val].reset_index(drop=True)
y2_train = y2.iloc[idx_train].reset_index(drop=True)
y2_val = y2.iloc[idx_val].reset_index(drop=True)

print("Data splits\n", X_train.shape, X_val.shape, y1_train.shape, y1_val.shape)



## === cell 6
scaler = StandardScaler()
std_X_train = scaler.fit_transform(X_train)
std_X_val = scaler.transform(X_val)
std_X_test = scaler.transform(X_test_df)

(std_X_train.shape, std_X_val.shape, std_X_test.shape)




## === cell 7
class Tensoroperations:
    def convert_to_tensor(self, X, y=None):
        X_tensor = torch.from_numpy(np.asarray(X)).float()
        if y is None:
            return X_tensor
        y_tensor = torch.from_numpy(np.asarray(y)).float()
        return X_tensor, y_tensor


tenops = Tensoroperations()

X_tensor_train, y1_tensor_train = tenops.convert_to_tensor(std_X_train, y1_train.values)
X_tensor_val, y1_tensor_val = tenops.convert_to_tensor(std_X_val, y1_val.values)

_, y2_tensor_train = tenops.convert_to_tensor(std_X_train, y2_train.values)
_, y2_tensor_val = tenops.convert_to_tensor(std_X_val, y2_val.values)

X_tensor_test = tenops.convert_to_tensor(std_X_test)

(X_tensor_train.shape, y1_tensor_train.shape, X_tensor_test.shape)




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


class CustomDataTest(Dataset):
    def __init__(self, X_data):
        super().__init__()
        self.X_data = X_data

    def __getitem__(self, index):
        return self.X_data[index]

    def __len__(self):
        return len(self.X_data)


train1_dataset = CustomDataset(X_tensor_train, y1_tensor_train)
val1_dataset = CustomDataset(X_tensor_val, y1_tensor_val)

train2_dataset = CustomDataset(X_tensor_train, y2_tensor_train)
val2_dataset = CustomDataset(X_tensor_val, y2_tensor_val)

test_dataset = CustomDataTest(X_tensor_test)

train_dataloader = DataLoader(train1_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val1_dataset, batch_size=64, shuffle=False)

train_dataloader2 = DataLoader(train2_dataset, batch_size=64, shuffle=True)
val_dataloader2 = DataLoader(val2_dataset, batch_size=64, shuffle=False)

test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)



## === cell 9
false_pos, true_pos = [], []


class EnzymeClassificationBase(nn.Module):
    def __init__(self):
        super().__init__()
        self._acc_metric = BinaryAccuracy().to(device)
        self._roc_metric = BinaryROC().to(device)

    def _accuracy(self, outputs, labels):
        return self._acc_metric(outputs, labels.unsqueeze(1))

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
        self._get_roc(out.detach(), labels.detach())
        return {
            "Validation_loss": loss.detach().cpu(),
            "Validation_acc": acc.detach().cpu(),
        }

    def _get_roc(self, output, labels):
        fpr, tpr, _ = self._roc_metric(output, labels.long().unsqueeze(1))
        false_pos.append(fpr.detach().cpu())
        true_pos.append(tpr.detach().cpu())

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




## === cell 10
n_output = 1


class MultiClassificationNN(EnzymeClassificationBase):
    def __init__(self, n_input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_input_dim, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 8),
            nn.LeakyReLU(),
            nn.Linear(8, n_output),
            nn.Sigmoid(),
        )

    def forward(self, xb):
        return self.network(xb)


model_EC1 = MultiClassificationNN(X_train.shape[1]).to(device)
model_EC2 = MultiClassificationNN(X_train.shape[1]).to(device)

print(model_EC1)




## === cell 11
class Trainer:
    @torch.no_grad()
    def _evaluate(self, model, val_loader):
        model.eval()
        outputs = [model.validation_step(batch) for batch in val_loader]
        return model.validation_epoch_end(outputs)

    def fit(self, epochs, lr, model, train_loader, val_loader, opt_func):
        history = []
        optimizer = opt_func(model.parameters(), lr, weight_decay=2e-5)

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


num_epochs = 150
lr = 1e-4
opt_func = optim.Adam

tariner = Trainer()



## === cell 12
history_EC1 = tariner.fit(
    num_epochs, lr, model_EC1, train_dataloader, val_dataloader, opt_func
)



## === cell 13
history_EC2 = tariner.fit(
    num_epochs, lr, model_EC2, train_dataloader2, val_dataloader2, opt_func
)




## === cell 14
def plot_accuracies(history):
    accuracies = [x["Validation_acc"] for x in history]
    plt.figure(figsize=(8, 4))
    plt.plot(accuracies, "-x")
    plt.xlabel("Epoch")
    plt.ylabel("accuracy")
    plt.title("Validation Accuracy vs Epoch")
    plt.show()


def plot_losses(history):
    train_losses = [x.get("Train_loss") for x in history]
    val_losses = [x["Validation_loss"] for x in history]
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, "-bx")
    plt.plot(val_losses, "-rx")
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.legend(["Training", "Validation"])
    plt.title("Loss vs Epochs")
    plt.show()


plot_accuracies(history_EC1)
plot_losses(history_EC1)



## === cell 15
if len(false_pos) > 0 and len(true_pos) > 0:
    fpr = [float(torch.mean(fp)) for fp in false_pos if len(fp) > 0]
    tpr = [float(torch.mean(tp)) for tp in true_pos if len(tp) > 0]
    if len(fpr) == len(tpr) and len(fpr) > 1:
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr)
        plt.title("Receiver Operating Characteristics (mean per epoch)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.show()




## === cell 16
class Evaluate:
    def eval_test_data(self, model, test_data_dl):
        preds = []
        model.eval()
        with torch.no_grad():
            for X_batch_test in test_data_dl:
                X_batch_test = X_batch_test.to(device)
                y_prob = model(X_batch_test)
                preds.append(y_prob.detach().cpu().numpy().reshape(-1))
        return np.concatenate(preds, axis=0)


eva = Evaluate()

ec1 = eva.eval_test_data(model_EC1, test_dataloader)
ec2 = eva.eval_test_data(model_EC2, test_dataloader)

(ec1.shape, ec2.shape)



## === cell 17
submission = pd.DataFrame({"id": test["id"].values, "EC1": ec1, "EC2": ec2})

assert submission.shape[0] == sample_sub.shape[0]
assert list(submission.columns) == ["id", "EC1", "EC2"]
assert submission["id"].equals(sample_sub["id"])

submission.to_csv("submission.csv", index=False)
submission.head()
