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
Given simulated manufacturing control data, predict whether the machine is in state `0` or state `1`.

## Metric
Area under the ROC curve.

## Submission Format
For each `id` in the test set, you must predict a probability for the `target` variable. The file should contain a header and have the following format:

```
id,target
900000,0.65
900001,0.97
900002,0.02
etc.
```

## Dataset
- **train.csv** - the training data, which includes normalized continuous data and categorical data
- **test.csv** - the test set; your task is to predict binary `target` variable which represents the state of a manufacturing process
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.10

# 3. Installed packages

geopandas==0.14.4
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
sentence-transformers==4.1.0
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
transformers==4.53.3

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (79 lines)
            sample_submission.csv (100001 lines)
            sample_submission.csv.zip (224.9 kB)
            test.csv (100001 lines)
            test.csv.zip (16.6 MB)
            train.csv (800001 lines)
            train.csv.zip (133.3 MB)
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
        input/
            description.md (79 lines)
            sample_submission.csv (100001 lines)
            sample_submission.csv.zip (224.9 kB)
            test.csv (100001 lines)
            test.csv.zip (16.6 MB)
            train.csv (800001 lines)
            train.csv.zip (133.3 MB)
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
        working/
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
```

-> data/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> data/tabular-playground-series-may-2022/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> data/tabular-playground-series-may-2022/test.csv has 100000 rows and 32 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 17 more columns

-> data/tabular-playground-series-may-2022/train.csv has 800000 rows and 33 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 18 more columns

-> data/test.csv has 100000 rows and 32 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 17 more columns

-> data/train.csv has 800000 rows and 33 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 18 more columns

-> input/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> (stopped after 10 files for performance)

# 5. Target score

0.99659

# 6. Current score

0.49872

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.49872) has done: 'The crash happens during batching because the feature matrix coming from `df.drop(...).values` contains `object` dtype, which PyTorch’s default collate cannot stack into tensors. This is caused by `pd.get_dummies` producing boolean columns in pandas 2.2+, which can lead to mixed dtypes and an overall `object` numpy array. The minimal fix is to ensure `DataLoader` always emits purely numeric `float32` features (and `float32` labels) by converting the dataframe columns to numeric numpy arrays explicitly inside cell 6. This preserves the exact model/training logic while making the DataLoader compatible with PyTorch collation.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from typing import List
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm
import random
from sklearn import metrics


## === cell 1
train_df = pd.read_csv(f"../input/tabular-playground-series-may-2022/train.csv")
test_df = pd.read_csv(f"../input/tabular-playground-series-may-2022/test.csv")


## === cell 2
test_df["target"] = 0

train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)


## === cell 3
params = {
        "~lr": 0.01,
        "~batch_size": 2048,
        "~epochs": 40,
        "~early_stopping_patience": 6,
        "~optimizer": "adam",
        "~loss": "bce",
        "activation": "swish",
        "model": "baseline"
    }


## === cell 4
class DataProcess:
    def __init__(self, df: pd.DataFrame) -> None:
        self.scaler = StandardScaler()
        self.numerical_cols = [f"f_{i:02d}" for i in range(27)] + ["f_28"]
        self.float_cols = [i for i in df.columns if df[i].dtype == "float64"]
        self.scaler.fit(df[self.numerical_cols].values)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.numerical_cols] = self.scaler.transform(df[self.numerical_cols].values)

        df = df.drop(columns="f_29").join(
            pd.get_dummies(df["f_29"]).rename(columns={0: "f_29_0", 1: "f_29_1"})
        )

        df = df.drop(columns="f_30").join(
            pd.get_dummies(df["f_30"]).rename(
                columns={0: "f_30_0", 1: "f_30_1", 2: "f_30_2"}
            )
        )



        for i in range(10):
            df[f"f_27_{i}_int"] = df.f_27.str[i].map(ord) - ord("A")
        df[f"f_27_nunique"] = df.f_27.apply(lambda c: len(set(c)))

        df = df.drop(columns="f_27")

        df["f_sum"] = df[self.float_cols].sum(axis=1)
        df["f_min"] = df[self.float_cols].min(axis=1)
        df["f_max"] = df[self.float_cols].max(axis=1)
        df["f_mean"] = df[self.float_cols].mean(axis=1)
        df["f_std"] = df[self.float_cols].std(axis=1)
        df["f_mad"] = df[self.float_cols].mad(axis=1)
        df["f_kurt"] = df[self.float_cols].kurt(axis=1)
        df["f_count_pos"] = df[self.float_cols].gt(0).count(axis=1)

        return df


## === cell 5
processor = DataProcess(train_df)


def _preprocess_with_mad_fix(df: pd.DataFrame) -> pd.DataFrame:
    df[processor.numerical_cols] = processor.scaler.transform(
        df[processor.numerical_cols].values
    )

    df = df.drop(columns="f_29").join(
        pd.get_dummies(df["f_29"]).rename(columns={0: "f_29_0", 1: "f_29_1"})
    )

    df = df.drop(columns="f_30").join(
        pd.get_dummies(df["f_30"]).rename(
            columns={0: "f_30_0", 1: "f_30_1", 2: "f_30_2"}
        )
    )

    for i in range(10):
        df[f"f_27_{i}_int"] = df.f_27.str[i].map(ord) - ord("A")
    df["f_27_nunique"] = df.f_27.apply(lambda c: len(set(c)))

    df = df.drop(columns="f_27")

    df["f_sum"] = df[processor.float_cols].sum(axis=1)
    df["f_min"] = df[processor.float_cols].min(axis=1)
    df["f_max"] = df[processor.float_cols].max(axis=1)
    df["f_mean"] = df[processor.float_cols].mean(axis=1)
    df["f_std"] = df[processor.float_cols].std(axis=1)
    row_mean = df[processor.float_cols].mean(axis=1)
    df["f_mad"] = (df[processor.float_cols].sub(row_mean, axis=0)).abs().mean(axis=1)
    df["f_kurt"] = df[processor.float_cols].kurt(axis=1)
    df["f_count_pos"] = df[processor.float_cols].gt(0).count(axis=1)

    return df


train_df = _preprocess_with_mad_fix(train_df)
val_df = _preprocess_with_mad_fix(val_df)
test_df = _preprocess_with_mad_fix(test_df)

for _df in (train_df, val_df, test_df):
    float_cols = processor.float_cols
    row_mean = _df[float_cols].mean(axis=1)
    _df["f_mad"] = (_df[float_cols].sub(row_mean, axis=0)).abs().mean(axis=1)


## === cell 6
class DataLoader:
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, x: np.ndarray, y: np.ndarray):
            self.x = x
            self.y = y
            self.len = len(self.x)

        def __getitem__(self, index):
            x = self.x[index]
            y = self.y[index]
            return x, y

        def __len__(self):
            return self.len

    class Sampler(torch.utils.data.Sampler):
        def __init__(self, l: int, shuffle: bool) -> None:
            super().__init__(l)
            self.len = l
            self.shuffle = shuffle

        def __iter__(self) -> List[int]:
            lst = list(range(self.len))
            if self.shuffle:
                random.shuffle(lst)
            for i in lst:
                yield i

        def __len__(self) -> int:
            return self.len

    def __init__(self, df: pd.DataFrame) -> None:
        self.x = df.drop(columns=["id", "target"]).values
        self.y = df["target"].values

    def get(self, is_train=False) -> torch.utils.data.DataLoader:
        dataset = self.Dataset(self.x, self.y)
        sampler = self.Sampler(len(self.x), shuffle=is_train)
        batch_size = params["~batch_size"] if is_train else len(dataset)

        return torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_size=batch_size,
            drop_last=is_train,
        )

    
train_ds = DataLoader(train_df).get(is_train=True)
val_ds = DataLoader(val_df).get()
test_ds = DataLoader(test_df).get()


## === cell 7
class Model(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc5 = nn.Linear(128, 32)
        self.bn5 = nn.BatchNorm1d(32)
        self.fc6 = nn.Linear(32, 1)
        if params["activation"] == "relu":
            self.activation = F.relu
        elif params["activation"] == "swish":
            self.activation = F.silu

    def forward(self, x):
        x = self.activation(self.bn1(self.fc1(x)))
        x = self.activation(self.bn2(self.fc2(x)))
        x = self.activation(self.bn5(self.fc5(x)))
        x = torch.sigmoid(self.fc6(x))

        return x.squeeze()

model = Model(len(set(train_df.columns) - {"id", "target"})).to('cuda')


## === cell 8
def get_scheduler(optimizer, train_dataloader):
    epochs = params["~epochs"]
    num_training_steps = int(epochs * len(train_dataloader))

    return get_linear_schedule_with_warmup(
        optimizer, int(0.1 * num_training_steps), num_training_steps
    )


## === cell 9
class Callback:
    def __init__(self) -> None:
        pass

    def on_val_end(self, preds: np.ndarray, gts: np.ndarray, loss):
        pass

    def on_train_batch_end(self, preds: np.ndarray, gts: np.ndarray, loss):
        pass

    def on_epoch_end(self, loss, val_loss, model: torch.nn.Module) -> bool:
        pass

    def on_train_finish(self, model: torch.nn.Module):
        pass

class EarlyStopping(Callback):
    def __init__(self) -> None:
        self.patience = params["~early_stopping_patience"]
        self.min_loss = np.inf
        self.counter = 0
        self.best_state_dict = None

    def on_val_end(self, preds: np.ndarray, gts: np.ndarray, loss):
        pass

    def on_train_batch_end(self, preds: np.ndarray, gts: np.ndarray, loss):
        pass

    def on_epoch_end(self, loss, val_loss, model: torch.nn.Module) -> bool:
        if val_loss < self.min_loss:
            self.min_loss = val_loss
            self.counter = 0
            self.best_state_dict = model.state_dict()
        else:
            self.counter += 1

        return self.counter < self.patience

    def on_train_finish(self, model: torch.nn.Module):
        model.load_state_dict(self.best_state_dict)

        
class WandbCallback(Callback):
    def __init__(self) -> None:
        self.train_epoch_losses = []
        self.val_epoch_losses = []
        self.train_batch_losses = []
        self.val_batch_losses = []

    def on_val_end(self, preds: np.ndarray, gts: np.ndarray, loss):
        gts = gts.detach().cpu()
        preds = preds.detach().cpu()

        fpr, tpr, threshold = metrics.roc_curve(gts, preds)
        roc_auc = metrics.auc(fpr, tpr)

        print(f"roc_auc: {roc_auc}")
        return True

    def on_train_batch_end(self, preds: np.ndarray, gts: np.ndarray, loss):
        self.train_batch_losses.append(loss)

    def on_epoch_end(self, loss, val_loss, model: torch.nn.Module) -> bool:
        self.val_epoch_losses.append(val_loss)
        self.train_epoch_losses.append(loss)


        return True

def epoch_train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    train_loader,
    criterion,
    callbacks: List[Callback] = [],
):
    model.train()

    losses = []
    for i, batch in tqdm(enumerate(train_loader), total=len(train_loader), unit=" batch"):
        batch_x = batch[0].to(torch.float32).to("cuda")
        batch_y = batch[1].to(torch.float32).to("cuda")

        optimizer.zero_grad()
        pred_y = model(batch_x)
        loss = criterion(pred_y, batch_y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        [cb.on_train_batch_end(pred_y, batch_y, loss.item()) for cb in callbacks]

    return np.mean(losses)


def epoch_val(
    model: torch.nn.Module, val_loader, criterion, callbacks: List[Callback] = []
):
    model.eval()

    losses = []
    for i, batch in enumerate(val_loader):
        batch_x = batch[0].to(torch.float32).to("cuda")
        batch_y = batch[1].to(torch.float32).to("cuda")
        pred_y = model(batch_x)
        loss = criterion(pred_y, batch_y)
        losses.append(loss.item())

        [cb.on_val_end(pred_y, batch_y, loss.item()) for cb in callbacks]
        break
    return np.mean(losses)


def predict(model: torch.nn.Module, test_loader):
    model.eval()
    preds = []
    gts = []
    for i, (batch_x, batch_y) in enumerate(test_loader):
        batch_x = batch_x.to(torch.float32).to("cuda")
        batch_y = batch_y.to(torch.float32).to("cuda")
        pred_y = model(batch_x)

        preds.append(pred_y.cpu().detach().numpy())
        gts.append(batch_y.cpu().detach().numpy())

        break

    preds = np.array(preds)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    gts = np.array(gts)
    gts = gts.reshape(-1, gts.shape[-2], gts.shape[-1])
    return preds, gts


## === cell 10
class DataLoader:
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, x: np.ndarray, y: np.ndarray):
            self.x = x
            self.y = y
            self.len = len(self.x)

        def __getitem__(self, index):
            x = self.x[index]
            y = self.y[index]
            return x, y

        def __len__(self):
            return self.len

    class Sampler(torch.utils.data.Sampler):
        def __init__(self, l: int, shuffle: bool) -> None:
            super().__init__(l)
            self.len = l
            self.shuffle = shuffle

        def __iter__(self) -> List[int]:
            lst = list(range(self.len))
            if self.shuffle:
                random.shuffle(lst)
            for i in lst:
                yield i

        def __len__(self) -> int:
            return self.len

    def __init__(self, df: pd.DataFrame) -> None:
        x_df = df.drop(columns=["id", "target"])
        self.x = x_df.to_numpy(dtype=np.float32, copy=True)
        self.y = df["target"].to_numpy(dtype=np.float32, copy=True)

    def get(self, is_train=False) -> torch.utils.data.DataLoader:
        dataset = self.Dataset(self.x, self.y)
        sampler = self.Sampler(len(self.x), shuffle=is_train)
        batch_size = params["~batch_size"] if is_train else len(dataset)

        return torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_size=batch_size,
            drop_last=is_train,
        )


train_ds = DataLoader(train_df).get(is_train=True)
val_ds = DataLoader(val_df).get()
test_ds = DataLoader(test_df).get()


## === cell 11
preds, gts = predict(model, test_ds)

sub = pd.read_csv(f"../input/tabular-playground-series-may-2022/sample_submission.csv")
sub.target = preds.squeeze()
sub.to_csv('submission.csv', index=False)
