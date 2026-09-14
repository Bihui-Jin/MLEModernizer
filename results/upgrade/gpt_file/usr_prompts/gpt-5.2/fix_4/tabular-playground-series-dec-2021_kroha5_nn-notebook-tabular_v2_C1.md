# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Predict the class of a given image from a synthetic dataset.

## MetricMulti-class classification accuracy.

## Submission FormatFor each `Id` in the test set, you must predict the `Cover_Type` class. The file should contain a header and have the following format:
```
Id,Cover_Type
4000000,2
4000001,1
4000001,3
etc.
```

## Dataset 
- train.csv - the training data with the target `Cover_Type` column
- test.csv - the test set; you will be predicting the `Cover_Type` for each row in this file (the target integer class)
- sample_submission.csv - a sample submission file in the correct format

# 2. Python version

3.10

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
            description.md (59 lines)
            sample_submission.csv (400001 lines)
            sample_submission.csv.zip (1.6 MB)
            test.csv (400001 lines)
            test.csv.zip (10.7 MB)
            train.csv (3600001 lines)
            train.csv.zip (97.9 MB)
            tabular-playground-series-dec-2021/
                description.md (59 lines)
                sample_submission.csv (400001 lines)
                ... and 5 other files
                tabular-playground-series-dec-2021/
        input/
            description.md (59 lines)
            sample_submission.csv (400001 lines)
            sample_submission.csv.zip (1.6 MB)
            test.csv (400001 lines)
            test.csv.zip (10.7 MB)
            train.csv (3600001 lines)
            train.csv.zip (97.9 MB)
            tabular-playground-series-dec-2021/
                description.md (59 lines)
                sample_submission.csv (400001 lines)
                ... and 5 other files
                tabular-playground-series-dec-2021/
        working/
            tabular-playground-series-dec-2021/
                description.md (59 lines)
                sample_submission.csv (400001 lines)
                ... and 5 other files
                tabular-playground-series-dec-2021/
```

-> data/sample_submission.csv has 400000 rows and 2 columns.
The columns are: Id, Cover_Type

-> data/tabular-playground-series-dec-2021/sample_submission.csv has 400000 rows and 2 columns.
The columns are: Id, Cover_Type

-> data/tabular-playground-series-dec-2021/test.csv has 400000 rows and 55 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 40 more columns

-> data/tabular-playground-series-dec-2021/train.csv has 3600000 rows and 56 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 41 more columns

-> data/test.csv has 400000 rows and 55 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 40 more columns

-> data/train.csv has 3600000 rows and 56 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 41 more columns

-> input/sample_submission.csv has 400000 rows and 2 columns.
The columns are: Id, Cover_Type

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



## === cell 1
input_dir = "/kaggle/input/tabular-playground-series-dec-2021/"
train = pd.read_csv(os.path.join(input_dir, "train.csv"), index_col="Id")
test = pd.read_csv(os.path.join(input_dir, "test.csv"), index_col="Id")
sub = pd.read_csv(os.path.join(input_dir, "sample_submission.csv"))



## === cell 2
cont_cols = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Horizontal_Distance_To_Fire_Points",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
]

binary_cols = [f"Wilderness_Area{i}" for i in range(1, 5)] + [
    f"Soil_Type{i}" for i in range(1, 41)
]
target_col = "Cover_Type"



## === cell 3
_ = train[target_col].value_counts()



## === cell 4
train[target_col] = train[target_col].astype(np.int64) - 1

min_y = int(train[target_col].min())
max_y = int(train[target_col].max())
if min_y < 0 or max_y > 6:
    raise ValueError(
        f"Encoded labels out of expected range [0,6]. Got min={min_y}, max={max_y}."
    )



## === cell 5
from sklearn.model_selection import train_test_split

vc = train[target_col].value_counts()
rare_classes = vc[vc < 2].index.tolist()
if len(rare_classes) > 0:
    train = train[~train[target_col].isin(rare_classes)].copy()

train_df, val_df = train_test_split(
    train, test_size=0.1, stratify=train[target_col], random_state=42, shuffle=True
)



## === cell 6
from sklearn.preprocessing import MinMaxScaler

all_cols = cont_cols + binary_cols

missing_train = [c for c in all_cols + [target_col] if c not in train_df.columns]
missing_val = [c for c in all_cols + [target_col] if c not in val_df.columns]
missing_test = [c for c in all_cols if c not in test.columns]
if missing_train:
    raise KeyError(f"Missing columns in train_df: {missing_train}")
if missing_val:
    raise KeyError(f"Missing columns in val_df: {missing_val}")
if missing_test:
    raise KeyError(f"Missing columns in test: {missing_test}")

scaler = MinMaxScaler()
train_df = train_df.copy()
val_df = val_df.copy()
test = test.copy()

train_df.loc[:, cont_cols] = scaler.fit_transform(train_df[cont_cols])
val_df.loc[:, cont_cols] = scaler.transform(val_df[cont_cols])
test.loc[:, cont_cols] = scaler.transform(test[cont_cols])

n_classes = 7

for name, df_ in [("train_df", train_df), ("val_df", val_df)]:
    y_ = df_[target_col].to_numpy()
    if y_.min() < 0 or y_.max() >= n_classes:
        raise ValueError(
            f"{name} labels out of range [0,{n_classes-1}]. min={y_.min()} max={y_.max()}"
        )



## === cell 7
import time
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import optim

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class ForestDataset(Dataset):
    def __init__(self, df, feature_cols, target_col):
        self.feature_cols = feature_cols
        self.target_col = target_col

        if target_col in df.columns:
            self.X = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
            self.y = df[target_col].to_numpy(dtype=np.int64, copy=True)
        else:
            self.X = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
            self.y = np.zeros((len(df),), dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = ForestDataset(train_df, all_cols, target_col)
val_dataset = ForestDataset(val_df, all_cols, target_col)




## === cell 8
class MultiLayerPerceptron(nn.Module):
    def __init__(self, len_fc1, len_fc2):
        super().__init__()
        self.fc1 = nn.Linear(len(all_cols), len_fc1)
        self.act1 = nn.Tanh()
        self.fc2 = nn.Linear(len_fc1, len_fc2)
        self.act2 = nn.Tanh()
        self.fc3 = nn.Linear(len_fc2, n_classes)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return self.fc3(x)




## === cell 9
def train_epoch(model, criterion, optimizer, dataset, epoch, device):
    data_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        pin_memory=True if device.startswith("cuda") else False,
        persistent_workers=False,
    )
    dataset_size = len(dataset)
    print(f"Epoch#{epoch}. Train")
    start_time = time.time()
    model.train()
    running_loss = 0.0
    running_acc = 0.0

    for inputs, labels in tqdm(data_loader, leave=False):
        inputs = inputs.to(device).type(torch.float)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, dim=1)
        running_acc += torch.sum(preds == labels.data).item()

    epoch_loss = running_loss / dataset_size
    epoch_acc = running_acc / dataset_size
    print(f"Loss (cross-entropy): {epoch_loss}")
    print(f"Accuracy (multiclass): {epoch_acc}")
    print(f"Epoch#{epoch} (Train) completed. {round(time.time() - start_time, 3)}s")
    return model, epoch_loss, epoch_acc




## === cell 10
def valid_epoch(model, criterion, dataset, epoch, device):
    data_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.startswith("cuda") else False,
        persistent_workers=False,
    )
    dataset_size = len(dataset)
    print(f"Epoch#{epoch}. Validation")
    start_time = time.time()
    model.eval()
    running_loss = 0.0
    running_acc = 0.0

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, leave=False):
            inputs = inputs.to(device).type(torch.float)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, dim=1)
            running_acc += torch.sum(preds == labels.data).item()

    epoch_loss = running_loss / dataset_size
    epoch_acc = running_acc / dataset_size
    print(f"Loss (cross-entropy): {epoch_loss}")
    print(f"Accuracy (multiclass): {epoch_acc}")
    print(
        f"Epoch#{epoch} (Validation) completed. {round(time.time() - start_time, 3)}s"
    )
    return model, epoch_loss, epoch_acc




## === cell 11
def init_model_and_optim(device_):
    model_ = MultiLayerPerceptron(150, 150).to(device_)
    crit_ = nn.CrossEntropyLoss()
    opt_ = optim.Adam(model_.parameters(), lr=1e-4)
    return model_, crit_, opt_


device = "cuda" if torch.cuda.is_available() else "cpu"
mlp_model, criterion, optimizer = init_model_and_optim(device)

best_state_dict = None
best_epoch = 1
best_acc = -1.0
num_epochs = 5

train_loss_history = []
val_loss_history = []
train_acc_history = []
val_acc_history = []

try:
    for epoch in range(1, num_epochs + 1):
        mlp_model, train_loss, train_acc = train_epoch(
            mlp_model, criterion, optimizer, train_dataset, epoch, device
        )
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)

        mlp_model, val_loss, val_acc = valid_epoch(
            mlp_model, criterion, val_dataset, epoch, device
        )
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        if val_acc > best_acc:
            best_epoch = epoch
            best_acc = val_acc
            best_state_dict = {
                k: v.detach().cpu().clone() for k, v in mlp_model.state_dict().items()
            }

except RuntimeError as e:
    if "device-side assert triggered" in str(e) and device.startswith("cuda"):
        print(
            "Caught CUDA device-side assert; switching to CPU and restarting training for stability."
        )
        device = "cpu"
        torch.cuda.empty_cache()
        mlp_model, criterion, optimizer = init_model_and_optim(device)
        best_state_dict = None
        best_epoch = 1
        best_acc = -1.0
        train_loss_history = []
        val_loss_history = []
        train_acc_history = []
        val_acc_history = []

        for epoch in range(1, num_epochs + 1):
            mlp_model, train_loss, train_acc = train_epoch(
                mlp_model, criterion, optimizer, train_dataset, epoch, device
            )
            train_loss_history.append(train_loss)
            train_acc_history.append(train_acc)

            mlp_model, val_loss, val_acc = valid_epoch(
                mlp_model, criterion, val_dataset, epoch, device
            )
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)

            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                best_state_dict = {
                    k: v.detach().cpu().clone()
                    for k, v in mlp_model.state_dict().items()
                }
    else:
        raise

if best_state_dict is None:
    best_state_dict = {
        k: v.detach().cpu().clone() for k, v in mlp_model.state_dict().items()
    }

print(f"Best epoch: {best_epoch}  Best val acc: {best_acc}")



## === cell 12
output_model_file = "best_model.bin"
torch.save(best_state_dict, output_model_file)



## === cell 13
best_model = MultiLayerPerceptron(150, 150).to(device)
best_model.load_state_dict(torch.load(output_model_file, map_location=device))

test_dataset = ForestDataset(test, all_cols, target_col)
data_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=True if device.startswith("cuda") else False,
    persistent_workers=False,
)

best_model.eval()
preds_list = []
with torch.no_grad():
    for inputs, _labels in tqdm(data_loader, leave=False):
        inputs = inputs.to(device).type(torch.float)
        outputs = best_model(inputs)
        _, preds = torch.max(outputs, dim=1)
        preds_list.append(preds.cpu())

test_preds = torch.cat(preds_list).numpy()



## === cell 14
pred_df = pd.DataFrame(
    {
        "Id": test.index.values.astype(np.int64),
        "Cover_Type": (test_preds + 1).astype(np.int64),
    }
)

sub_out = sub.merge(pred_df, on="Id", how="left", suffixes=("", "_pred"))
if "Cover_Type_pred" in sub_out.columns:
    sub_out["Cover_Type"] = sub_out["Cover_Type_pred"]
    sub_out = sub_out[["Id", "Cover_Type"]]

if sub_out["Cover_Type"].isna().any():
    raise ValueError("Submission has missing predictions after alignment/merge.")

sub_out["Cover_Type"] = sub_out["Cover_Type"].astype(np.int64)
sub_out.to_csv("submission_mlp.csv", index=False)

print(sub_out.head())
print("Wrote submission_mlp.csv with shape:", sub_out.shape)
