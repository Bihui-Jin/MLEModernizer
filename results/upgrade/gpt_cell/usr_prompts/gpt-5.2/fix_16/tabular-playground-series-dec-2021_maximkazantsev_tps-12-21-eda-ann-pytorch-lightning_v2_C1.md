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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
    StratifiedShuffleSplit,
)
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.optim.lr_scheduler import ExponentialLR
from pytorch_lightning.callbacks import LearningRateMonitor
from sklearn.utils.class_weight import compute_class_weight
import time
import gc
import torchmetrics

pd.set_option("display.max_rows", 150)
pd.set_option("display.max_columns", 500)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.float_format", lambda x: "%.5f" % x)

import os

if False:
    for dirname, _, filenames in os.walk("/kaggle/input"):
        for filename in filenames:
            print(os.path.join(dirname, filename))

pl.seed_everything(42, workers=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

try:
    cpu = os.cpu_count() or 1
    torch.set_num_threads(min(8, cpu))
    torch.set_num_interop_threads(1)
except Exception:
    pass




## === cell 1
def _build_dtypes():
    dtypes = {"Id": np.int32}
    cont = [
        "Elevation",
        "Aspect",
        "Slope",
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Hillshade_9am",
        "Hillshade_Noon",
        "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points",
    ]
    for c in cont:
        dtypes[c] = np.float32
    for i in range(1, 5):
        dtypes[f"Wilderness_Area{i}"] = np.int8
    for i in range(1, 41):
        dtypes[f"Soil_Type{i}"] = np.int8
    dtypes["Cover_Type"] = np.int8
    return dtypes


DTYPES = _build_dtypes()

target = "Cover_Type"
_all_feature_cols = (
    [
        "Elevation",
        "Aspect",
        "Slope",
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Hillshade_9am",
        "Hillshade_Noon",
        "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points",
    ]
    + [f"Wilderness_Area{i}" for i in range(1, 5)]
    + [f"Soil_Type{i}" for i in range(1, 41)]
)
_usecols_train = ["Id"] + _all_feature_cols + [target]
_usecols_test = ["Id"] + _all_feature_cols

train = pd.read_csv(
    "/kaggle/input/tabular-playground-series-dec-2021/train.csv",
    engine="c",
    low_memory=False,
    usecols=_usecols_train,
    dtype={k: v for k, v in DTYPES.items() if k in _usecols_train},
)
test = pd.read_csv(
    "/kaggle/input/tabular-playground-series-dec-2021/test.csv",
    engine="c",
    low_memory=False,
    usecols=_usecols_test,
    dtype={k: v for k, v in DTYPES.items() if k in _usecols_test},
)




## === cell 2
def reduce_mem_usage(df, verbose=True):
    if verbose:
        start_mem = df.memory_usage().sum() / 1024**2
        end_mem = start_mem
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 0.0
            )
        )
    return df


train = reduce_mem_usage(train)
test = reduce_mem_usage(test)



## === cell 3
if False:
    train.info(memory_usage="deep")



## === cell 4
if False:
    test.info(memory_usage="deep")



## === cell 5
colors = [
    "lightcoral",
    "sandybrown",
    "darkorange",
    "mediumseagreen",
    "lightseagreen",
    "cornflowerblue",
    "mediumpurple",
    "palevioletred",
    "lightskyblue",
    "sandybrown",
    "yellowgreen",
    "indianred",
    "lightsteelblue",
    "mediumorchid",
    "deepskyblue",
]



## === cell 6
if False:
    train.head()



## === cell 7
features = list(train.columns[1:55])  # 54 feature cols



## === cell 8
if False:
    train[target].value_counts()



## === cell 9
if False:
    fig, ax = plt.subplots(figsize=(5, 6))
    pie = ax.pie(
        [len(train), len(test)],
        labels=["Train dataset", "Test dataset"],
        colors=["salmon", "teal"],
        textprops={"fontsize": 15},
        autopct="%1.1f%%",
    )
    ax.axis("equal")
    ax.set_title("Dataset length comparison", fontsize=18)
    fig.set_facecolor("white")
    plt.show()



## === cell 10
if False:
    fig, ax = plt.subplots(figsize=(14, 8))

    bars = ax.bar(
        train[target].value_counts().sort_index().index,
        train[target].value_counts().sort_index().values,
        color=colors,
        edgecolor="black",
    )
    ax.set_title("Target distribution", fontsize=20, pad=15)
    ax.set_ylabel("Count", fontsize=14, labelpad=15)
    ax.set_xlabel("Target label", fontsize=14, labelpad=20)
    ax.tick_params(axis="x", pad=20)
    ax.bar_label(
        bars, train[target].value_counts().sort_index().values, padding=3, fontsize=12
    )
    ax.bar_label(
        bars,
        [
            f"{x*100:2.1f}%"
            for x in train[target].value_counts().sort_index().values / len(train)
        ],
        padding=-20,
        fontsize=12,
    )
    ax.margins(0.025, 0.06)
    ax.grid(axis="y")

    plt.show()



## === cell 11
if False:
    train[features].describe()



## === cell 12
cat_features = [
    c for c in features if c.startswith("Wilderness_Area") or c.startswith("Soil_Type")
]
num_features = [c for c in features if c not in cat_features]

print(f"There are {len(cat_features)} categorical features: {pd.Index(cat_features)}")
print(f"\nThere are {len(num_features)} continuous features: {pd.Index(num_features)}")



## === cell 13
train.isna().sum().sum(), test.isna().sum().sum()



## === cell 14
if False:
    df = pd.concat([train[num_features], test[num_features]], axis=0)
    columns = df.columns.values

    cols = 3
    rows = len(columns) // cols + 1

    fig, axs = plt.subplots(ncols=cols, nrows=rows, figsize=(16, 20), sharex=False)

    plt.subplots_adjust(hspace=0.3)
    i = 0

    for r in np.arange(0, rows, 1):
        for c in np.arange(0, cols, 1):
            if i >= len(columns):
                axs[r, c].set_visible(False)
            else:
                hist1 = axs[r, c].hist(
                    train[columns[i]].values,
                    range=(df[columns[i]].min(), df[columns[i]].max()),
                    bins=40,
                    color="deepskyblue",
                    edgecolor="black",
                    alpha=0.7,
                    label="Train Dataset",
                )
                hist2 = axs[r, c].hist(
                    test[columns[i]].values,
                    range=(df[columns[i]].min(), df[columns[i]].max()),
                    bins=40,
                    color="palevioletred",
                    edgecolor="black",
                    alpha=0.7,
                    label="Test Dataset",
                )
                axs[r, c].set_title(columns[i], fontsize=12, pad=5)
                axs[r, c].set_yticks(axs[r, c].get_yticks())
                axs[r, c].set_yticklabels(
                    [str(int(i / 1000)) + "k" for i in axs[r, c].get_yticks()]
                )
                axs[r, c].tick_params(axis="y", labelsize=10)
                axs[r, c].tick_params(axis="x", labelsize=10)
                axs[r, c].grid(axis="y")
                if i == 0:
                    axs[r, c].legend(fontsize=10)

            i += 1
    plt.show()



## === cell 15
if False:
    df = pd.concat([train[cat_features], test[cat_features]], axis=0)
    columns = df.columns.values

    cols = 4
    rows = len(columns) // cols + 1

    fig, axs = plt.subplots(ncols=cols, nrows=rows, figsize=(16, 40), sharex=False)

    plt.subplots_adjust(hspace=0.3)
    i = 0

    for r in np.arange(0, rows, 1):
        for c in np.arange(0, cols, 1):
            if i >= len(columns):
                axs[r, c].set_visible(False)
            else:
                hist1 = axs[r, c].hist(
                    train[columns[i]].values,
                    range=(df[columns[i]].min(), df[columns[i]].max()),
                    bins=40,
                    color="deepskyblue",
                    edgecolor="black",
                    alpha=0.7,
                    label="Train Dataset",
                )
                hist2 = axs[r, c].hist(
                    test[columns[i]].values,
                    range=(df[columns[i]].min(), df[columns[i]].max()),
                    bins=40,
                    color="palevioletred",
                    edgecolor="black",
                    alpha=0.7,
                    label="Test Dataset",
                )
                axs[r, c].set_title(columns[i], fontsize=12, pad=5)
                axs[r, c].set_yticks(axs[r, c].get_yticks())
                axs[r, c].set_yticklabels(
                    [str(int(i / 1000)) + "k" for i in axs[r, c].get_yticks()]
                )
                axs[r, c].tick_params(axis="y", labelsize=10)
                axs[r, c].tick_params(axis="x", labelsize=10)
                axs[r, c].grid(axis="y")
                if i == 0:
                    axs[r, c].legend(fontsize=10)

            i += 1
    plt.show()



## === cell 16
print(
    f"Rows with soil type 7: {(train['Soil_Type7'] == 1).sum() + (test['Soil_Type7'] == 1).sum()}"
)
print(
    f"Rows with soil type 15: {(train['Soil_Type15'] == 1).sum() + (test['Soil_Type15'] == 1).sum()}"
)



## === cell 17
train.drop(["Soil_Type7", "Soil_Type15"], axis=1, inplace=True)
test.drop(["Soil_Type7", "Soil_Type15"], axis=1, inplace=True)
features.remove("Soil_Type7")
features.remove("Soil_Type15")
cat_features = [c for c in cat_features if c not in ("Soil_Type7", "Soil_Type15")]
num_features = [c for c in num_features if c not in ("Soil_Type7", "Soil_Type15")]



## === cell 18
if False:
    print("Numerical features with the least amount of unique values:")
    train[num_features].nunique().sort_values().head(5)



## === cell 19
train = train.loc[train[target] != 5].reset_index(drop=True)

label_enc = LabelEncoder()
NUM_CLASSES = train[target].nunique()



## === cell 20
s_scaler = StandardScaler(copy=False)

train_num = train[num_features].to_numpy(dtype=np.float32, copy=False)
test_num = test[num_features].to_numpy(dtype=np.float32, copy=False)

train_num_scaled = s_scaler.fit_transform(train_num)
test_num_scaled = s_scaler.transform(test_num)

X_nn_np = train[features].to_numpy(dtype=np.float32, copy=False)
X_test_np = test[features].to_numpy(dtype=np.float32, copy=False)

_feat_index = {c: i for i, c in enumerate(features)}
_num_idx = np.fromiter(
    (_feat_index[c] for c in num_features), dtype=np.int64, count=len(num_features)
)

X_nn_np[:, _num_idx] = train_num_scaled
X_test_np[:, _num_idx] = test_num_scaled

y_np = label_enc.fit_transform(train[target]).astype(np.int64, copy=False)

mm_scaler = MinMaxScaler(copy=False)
X_nn_np = mm_scaler.fit_transform(X_nn_np).astype(np.float32, copy=False)
X_test_np = mm_scaler.transform(X_test_np).astype(np.float32, copy=False)

y = pd.Series(y_np)

X_test_nn = torch.from_numpy(np.ascontiguousarray(X_test_np))

del (
    train_num,
    test_num,
    train_num_scaled,
    test_num_scaled,
    X_test_np,
    y_np,
    _feat_index,
    _num_idx,
)
gc.collect()



## === cell 21
BATCH_SIZE = 4096




## === cell 22
def prepare_datasets(X_nn_np, X_valid_np, y_nn_np, y_valid_np, batch_size=BATCH_SIZE):
    X_nn_t = torch.from_numpy(np.ascontiguousarray(X_nn_np, dtype=np.float32))
    y_nn_t = torch.from_numpy(np.ascontiguousarray(y_nn_np, dtype=np.int64))
    X_valid_nn_t = torch.from_numpy(np.ascontiguousarray(X_valid_np, dtype=np.float32))
    y_valid_nn_t = torch.from_numpy(np.ascontiguousarray(y_valid_np, dtype=np.int64))

    train_ds = TensorDataset(X_nn_t, y_nn_t)
    valid_ds = TensorDataset(X_valid_nn_t, y_valid_nn_t)

    pin = torch.cuda.is_available()

    cpu = os.cpu_count() or 1
    num_workers = min(4, max(1, cpu // 2))

    dl_kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = 8

    train_dl = DataLoader(train_ds, **dl_kwargs)

    valid_dl = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
        prefetch_factor=8 if num_workers > 0 else None,
    )
    return train_dl, valid_dl




## === cell 23
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data)




## === cell 24
class ParamsTracker(pl.callbacks.Callback):
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []
        self.lr_epoch_start = []

    def on_train_epoch_start(self, trainer, module):
        current_learning_rate = trainer.optimizers[0].state_dict()["param_groups"][0][
            "lr"
        ]
        self.lr_epoch_start.append(current_learning_rate)

    def _get_metric(self, trainer, key):
        v = trainer.callback_metrics.get(key)
        if v is None:
            v = trainer.logged_metrics.get(key)
        return v

    def on_validation_epoch_end(self, trainer, module):
        metrics_val_loss = self._get_metric(trainer, "val_loss")
        metrics_val_acc = self._get_metric(trainer, "val_acc")
        if metrics_val_loss is not None:
            self.val_loss.append(float(metrics_val_loss.detach().cpu().item()))
        if metrics_val_acc is not None:
            self.val_acc.append(float(metrics_val_acc.detach().cpu().item()))

    def on_train_epoch_end(self, trainer, module):
        metrics_loss = self._get_metric(trainer, "loss_epoch")
        if metrics_loss is None:
            metrics_loss = self._get_metric(trainer, "loss")
        metrics_train_acc = self._get_metric(trainer, "train_acc")

        if metrics_loss is not None:
            self.train_loss.append(float(metrics_loss.detach().cpu().item()))
        if metrics_train_acc is not None:
            self.train_acc.append(float(metrics_train_acc.detach().cpu().item()))

        if (
            self.verbose
            and (module.current_epoch % 5 == 0)
            and self.lr_epoch_start
            and self.train_loss
            and self.train_acc
            and self.val_loss
            and self.val_acc
        ):
            print(
                f"Epoch {module.current_epoch} start learning rate: {self.lr_epoch_start[-1]:.6f}, "
                f"train_loss: {self.train_loss[-1]:.4f}, "
                f"train_acc: {self.train_acc[-1]:.4f}, "
                f"val_loss: {self.val_loss[-1]:.4f}, "
                f"val_acc: {self.val_acc[-1]:.4f}"
            )




## === cell 25
class BestStateCallback(pl.callbacks.Callback):
    def __init__(self, monitor="val_acc", mode="max"):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.best = -float("inf") if mode == "max" else float("inf")
        self.best_state_dict = None

    def on_validation_epoch_end(self, trainer, pl_module):
        val = trainer.callback_metrics.get(self.monitor)
        if val is None:
            val = trainer.logged_metrics.get(self.monitor)
        if val is None:
            return
        try:
            v = float(val.item())
        except Exception:
            v = float(val)
        improved = (v > self.best) if self.mode == "max" else (v < self.best)
        if improved:
            self.best = v
            self.best_state_dict = {
                k: v.detach().cpu().clone() for k, v in pl_module.state_dict().items()
            }




## === cell 26
class Model(pl.LightningModule):
    def __init__(self, input_shape):
        super().__init__()

        self.input = nn.Linear(input_shape, 128)
        self.hidden1 = nn.Linear(128, 64)
        self.hidden2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 6)

        self.dr = 0.2
        self.swish = F.hardswish
        self.softmax = nn.Softmax(dim=1)
        self.train_acc_metric = torchmetrics.Accuracy(
            task="multiclass", num_classes=6, average="micro"
        )
        self.val_acc_metric = torchmetrics.Accuracy(
            task="multiclass", num_classes=6, average="micro"
        )
        self.loss = nn.CrossEntropyLoss()

        self.flag = False

    def forward(self, x):
        x = self.swish(self.input(x))
        x = F.dropout(x, p=self.dr, training=self.training)
        x = self.swish(self.hidden1(x))
        x = F.dropout(x, p=self.dr, training=self.training)
        x = self.swish(self.hidden2(x))
        x = F.dropout(x, p=self.dr, training=self.training)
        x = self.output(x)
        return x

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X).squeeze(1)
        loss = self.loss(y_hat, y)
        self.train_acc_metric(torch.argmax(y_hat, dim=1), y)
        self.log("loss", loss, prog_bar=True, on_epoch=True, on_step=False, logger=True)
        return {"loss": loss}

    def on_train_epoch_end(self):
        train_acc = self.train_acc_metric.compute()
        self.log(
            "train_acc",
            train_acc,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            logger=True,
        )
        self.train_acc_metric.reset()

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X).squeeze(1)
        val_loss = self.loss(y_hat, y)
        self.val_acc_metric(torch.argmax(y_hat, dim=1), y)
        self.log(
            "val_loss",
            val_loss,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            logger=True,
        )
        return {"val_loss": val_loss}

    def on_validation_epoch_end(self):
        val_acc = self.val_acc_metric.compute()
        self.log(
            "val_acc", val_acc, prog_bar=True, on_epoch=True, on_step=False, logger=True
        )
        self.val_acc_metric.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=1e-3, eps=1e-8, weight_decay=1e-2, amsgrad=False
        )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=7,
            min_lr=1e-04,
            eps=1e-08,
            verbose=False,
            threshold=0.005,
            threshold_mode="abs",
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val_acc",
        }




## === cell 27
def train_ann(train_ds, valid_ds, Model=Model, input_shape=None):
    model = Model(input_shape)
    model.apply(initialize_weights)

    try:
        if hasattr(torch, "compile"):
            pass
    except Exception:
        pass

    early_stop_callback = EarlyStopping(
        monitor="val_acc", min_delta=0.00, patience=20, verbose=False, mode="max"
    )

    params_tracker_callback = ParamsTracker(verbose=True)
    best_state_callback = BestStateCallback(monitor="val_acc", mode="max")

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = 1

    trainer = pl.Trainer(
        fast_dev_run=False,
        max_epochs=60,
        precision=32,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        val_check_interval=1.0,
        callbacks=[early_stop_callback, params_tracker_callback, best_state_callback],
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        accelerator=accelerator,
        devices=devices,
        deterministic=True,
        log_every_n_steps=200,
    )

    trainer.fit(model, train_ds, valid_ds)

    if best_state_callback.best_state_dict is not None:
        model.load_state_dict(best_state_callback.best_state_dict, strict=True)

    model.eval()
    return model, params_tracker_callback


sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, valid_idx = next(sss.split(X_nn_np, y.to_numpy()))
X_train_np = X_nn_np[train_idx]
X_valid_np = X_nn_np[valid_idx]
y_train_np = y.to_numpy()[train_idx]
y_valid_np = y.to_numpy()[valid_idx]

train_ds, valid_ds = prepare_datasets(
    X_train_np, X_valid_np, y_train_np, y_valid_np, batch_size=BATCH_SIZE
)

model, _ = train_ann(train_ds, valid_ds, Model=Model, input_shape=X_train_np.shape[1])

X_valid_t = torch.from_numpy(np.ascontiguousarray(X_valid_np, dtype=np.float32))
with torch.no_grad():
    np.argmax(model(X_valid_t).detach().cpu().numpy(), axis=1)[:10]


## === cell 28
X_train_t = torch.from_numpy(np.ascontiguousarray(X_train_np, dtype=np.float32))
with torch.no_grad():
    accuracy_score(
        y_train_np,
        np.argmax(model(X_train_t).detach().cpu().numpy(), axis=1),
    )



## === cell 29
with torch.no_grad():
    accuracy_score(
        y_valid_np,
        np.argmax(model(X_valid_t).detach().cpu().numpy(), axis=1),
    )



## === cell 30
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

pin = torch.cuda.is_available()
test_ds = TensorDataset(X_test_nn)  # CPU tensor

cpu = os.cpu_count() or 1
num_workers = min(4, max(1, cpu // 2))

dl_kwargs = dict(
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin,
    persistent_workers=(num_workers > 0),
)
if num_workers > 0:
    dl_kwargs["prefetch_factor"] = 8

test_dl = DataLoader(test_ds, **dl_kwargs)

all_pred = []
with torch.no_grad():
    for (xb,) in test_dl:
        xb = xb.to(device, non_blocking=pin)
        logits = model(xb)
        pred_idx = torch.argmax(logits, dim=1)
        all_pred.append(pred_idx.detach().cpu())
test_pred_idx = torch.cat(all_pred, dim=0).numpy()

test_pred_cover_type = label_enc.inverse_transform(test_pred_idx)

predictions = pd.DataFrame()
predictions["Id"] = test["Id"].astype(np.int64)
predictions["Cover_Type"] = test_pred_cover_type.astype(np.int64)

predictions.to_csv("submission.csv", index=False)
predictions.head()



## === cell 31
if False:
    predictions["Cover_Type"].hist()
