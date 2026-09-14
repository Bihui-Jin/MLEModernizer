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

3.10

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

# 4. Code solution

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
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
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



## --- ERROR in cell 27, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mInternalTorchDynamoError[0m                  Traceback (most recent call last)
[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/call.py[0m in [0;36m_call_and_handle_interrupt[0;34m(trainer, trainer_fn, *args, **kwargs)[0m
[1;32m     48[0m             [0;32mreturn[0m [0mtrainer[0m[0;34m.[0m[0mstrategy[0m[0;34m.[0m[0mlauncher[0m[0;34m.[0m[0mlaunch[0m[0;34m([0m[0mtrainer_fn[0m[0;34m,[0m [0;34m*[0m[0margs[0m[0;34m,[0m [0mtrainer[0m[0;34m=[0m[0mtrainer[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 49[0;31m         [0;32mreturn[0m [0mtrainer_fn[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     50[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/trainer.py[0m in [0;36m_fit_impl[0;34m(self, model, train_dataloaders, val_dataloaders, datamodule, ckpt_path)[0m
[1;32m    597[0m         )
[0;32m--> 598[0;31m         [0mself[0m[0;34m.[0m[0m_run[0m[0;34m([0m[0mmodel[0m[0;34m,[0m [0mckpt_path[0m[0;34m=[0m[0mckpt_path[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    599[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/trainer.py[0m in [0;36m_run[0;34m(self, model, ckpt_path)[0m
[1;32m   1010[0m         [0;31m# ----------------------------[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1011[0;31m         [0mresults[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_run_stage[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1012[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/trainer.py[0m in [0;36m_run_stage[0;34m(self)[0m
[1;32m   1054[0m             [0;32mwith[0m [0mtorch[0m[0;34m.[0m[0mautograd[0m[0;34m.[0m[0mset_detect_anomaly[0m[0;34m([0m[0mself[0m[0;34m.[0m[0m_detect_anomaly[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1055[0;31m                 [0mself[0m[0;34m.[0m[0mfit_loop[0m[0;34m.[0m[0mrun[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1056[0m             [0;32mreturn[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/fit_loop.py[0m in [0;36mrun[0;34m(self)[0m
[1;32m    215[0m                 [0mself[0m[0;34m.[0m[0mon_advance_start[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 216[0;31m                 [0mself[0m[0;34m.[0m[0madvance[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    217[0m                 [0mself[0m[0;34m.[0m[0mon_advance_end[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/fit_loop.py[0m in [0;36madvance[0;34m(self)[0m
[1;32m    457[0m             [0;32massert[0m [0mself[0m[0;34m.[0m[0m_data_fetcher[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 458[0;31m             [0mself[0m[0;34m.[0m[0mepoch_loop[0m[0;34m.[0m[0mrun[0m[0;34m([0m[0mself[0m[0;34m.[0m[0m_data_fetcher[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    459[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/training_epoch_loop.py[0m in [0;36mrun[0;34m(self, data_fetcher)[0m
[1;32m    151[0m             [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 152[0;31m                 [0mself[0m[0;34m.[0m[0madvance[0m[0;34m([0m[0mdata_fetcher[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    153[0m                 [0mself[0m[0;34m.[0m[0mon_advance_end[0m[0;34m([0m[0mdata_fetcher[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/training_epoch_loop.py[0m in [0;36madvance[0;34m(self, data_fetcher)[0m
[1;32m    347[0m                     [0;31m# in automatic optimization, there can only be one optimizer[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 348[0;31m                     [0mbatch_output[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mautomatic_optimization[0m[0;34m.[0m[0mrun[0m[0;34m([0m[0mtrainer[0m[0;34m.[0m[0moptimizers[0m[0;34m[[0m[0;36m0[0m[0;34m][0m[0;34m,[0m [0mbatch_idx[0m[0;34m,[0m [0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    349[0m                 [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/optimization/automatic.py[0m in [0;36mrun[0;34m(self, optimizer, batch_idx, kwargs)[0m
[1;32m    191[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 192[0;31m             [0mself[0m[0;34m.[0m[0m_optimizer_step[0m[0;34m([0m[0mbatch_idx[0m[0;34m,[0m [0mclosure[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    193[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/optimization/automatic.py[0m in [0;36m_optimizer_step[0;34m(self, batch_idx, train_step_and_backward_closure)[0m
[1;32m    269[0m         [0;31m# model hook[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 270[0;31m         call._call_lightning_module_hook(
[0m[1;32m    271[0m             [0mtrainer[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/call.py[0m in [0;36m_call_lightning_module_hook[0;34m(trainer, hook_name, pl_module, *args, **kwargs)[0m
[1;32m    176[0m     [0;32mwith[0m [0mtrainer[0m[0;34m.[0m[0mprofiler[0m[0;34m.[0m[0mprofile[0m[0;34m([0m[0;34mf"[LightningModule]{pl_module.__class__.__name__}.{hook_name}"[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 177[0;31m         [0moutput[0m [0;34m=[0m [0mfn[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    178[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_lightning/core/module.py[0m in [0;36moptimizer_step[0;34m(self, epoch, batch_idx, optimizer, optimizer_closure)[0m
[1;32m   1365[0m         """
[0;32m-> 1366[0;31m         [0moptimizer[0m[0;34m.[0m[0mstep[0m[0;34m([0m[0mclosure[0m[0;34m=[0m[0moptimizer_closure[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1367[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_lightning/core/optimizer.py[0m in [0;36mstep[0;34m(self, closure, **kwargs)[0m
[1;32m    153[0m         [0;32massert[0m [0mself[0m[0;34m.[0m[0m_strategy[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 154[0;31m         [0mstep_output[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_strategy[0m[0;34m.[0m[0moptimizer_step[0m[0;34m([0m[0mself[0m[0;34m.[0m[0m_optimizer[0m[0;34m,[0m [0mclosure[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    155[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_lightning/strategies/strategy.py[0m in [0;36moptimizer_step[0;34m(self, optimizer, closure, model, **kwargs)[0m
[1;32m    238[0m         [0;32massert[0m [0misinstance[0m[0;34m([0m[0mmodel[0m[0;34m,[0m [0mpl[0m[0;34m.[0m[0mLightningModule[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 239[0;31m         [0;32mreturn[0m [0mself[0m[0;34m.[0m[0mprecision_plugin[0m[0;34m.[0m[0moptimizer_step[0m[0;34m([0m[0moptimizer[0m[0;34m,[0m [0mmodel[0m[0;34m=[0m[0mmodel[0m[0;34m,[0m [0mclosure[0m[0;34m=[0m[0mclosure[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    240[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_lightning/plugins/precision/precision.py[0m in [0;36moptimizer_step[0;34m(self, optimizer, model, closure, **kwargs)[0m
[1;32m    122[0m         [0mclosure[0m [0;34m=[0m [0mpartial[0m[0;34m([0m[0mself[0m[0;34m.[0m[0m_wrap_closure[0m[0;34m,[0m [0mmodel[0m[0;34m,[0m [0moptimizer[0m[0;34m,[0m [0mclosure[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 123[0;31m         [0;32mreturn[0m [0moptimizer[0m[0;34m.[0m[0mstep[0m[0;34m([0m[0mclosure[0m[0;34m=[0m[0mclosure[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    124[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/optim/optimizer.py[0m in [0;36mwrapper[0;34m(*args, **kwargs)[0m
[1;32m    492[0m [0;34m[0m[0m
[0;32m--> 493[0;31m                 [0mout[0m [0;34m=[0m [0mfunc[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    494[0m                 [0mself[0m[0;34m.[0m[0m_optimizer_step_code[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/optim/optimizer.py[0m in [0;36m_use_grad[0;34m(self, *args, **kwargs)[0m
[1;32m     90[0m             [0mtorch[0m[0;34m.[0m[0m_dynamo[0m[0;34m.[0m[0mgraph_break[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 91[0;31m             [0mret[0m [0;34m=[0m [0mfunc[0m[0;34m([0m[0mself[0m[0;34m,[0m [0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     92[0m         [0;32mfinally[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/optim/adamw.py[0m in [0;36mstep[0;34m(self, closure)[0m
[1;32m    219[0m             [0;32mwith[0m [0mtorch[0m[0;34m.[0m[0menable_grad[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 220[0;31m                 [0mloss[0m [0;34m=[0m [0mclosure[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    221[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_lightning/plugins/precision/precision.py[0m in [0;36m_wrap_closure[0;34m(self, model, optimizer, closure)[0m
[1;32m    108[0m         """
[0;32m--> 109[0;31m         [0mclosure_result[0m [0;34m=[0m [0mclosure[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    110[0m         [0mself[0m[0;34m.[0m[0m_after_closure[0m[0;34m([0m[0mmodel[0m[0;34m,[0m [0moptimizer[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/optimization/automatic.py[0m in [0;36m__call__[0;34m(self, *args, **kwargs)[0m
[1;32m    145[0m     [0;32mdef[0m [0m__call__[0m[0;34m([0m[0mself[0m[0;34m,[0m [0;34m*[0m[0margs[0m[0;34m:[0m [0mAny[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m:[0m [0mAny[0m[0;34m)[0m [0;34m->[0m [0mOptional[0m[0;34m[[0m[0mTensor[0m[0;34m][0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 146[0;31m         [0mself[0m[0;34m.[0m[0m_result[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mclosure[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    147[0m         [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_result[0m[0;34m.[0m[0mloss[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/utils/_contextlib.py[0m in [0;36mdecorate_context[0;34m(*args, **kwargs)[0m
[1;32m    115[0m         [0;32mwith[0m [0mctx_factory[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 116[0;31m             [0;32mreturn[0m [0mfunc[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    117[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/optimization/automatic.py[0m in [0;36mclosure[0;34m(self, *args, **kwargs)[0m
[1;32m    130[0m     [0;32mdef[0m [0mclosure[0m[0;34m([0m[0mself[0m[0;34m,[0m [0;34m*[0m[0margs[0m[0;34m:[0m [0mAny[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m:[0m [0mAny[0m[0;34m)[0m [0;34m->[0m [0mClosureResult[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 131[0;31m         [0mstep_output[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_step_fn[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    132[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/optimization/automatic.py[0m in [0;36m_training_step[0;34m(self, kwargs)[0m
[1;32m    318[0m [0;34m[0m[0m
[0;32m--> 319[0;31m         [0mtraining_step_output[0m [0;34m=[0m [0mcall[0m[0;34m.[0m[0m_call_strategy_hook[0m[0;34m([0m[0mtrainer[0m[0;34m,[0m [0;34m"training_step"[0m[0;34m,[0m [0;34m*[0m[0mkwargs[0m[0;34m.[0m[0mvalues[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    320[0m         [0mself[0m[0;34m.[0m[0mtrainer[0m[0;34m.[0m[0mstrategy[0m[0;34m.[0m[0mpost_training_step[0m[0;34m([0m[0;34m)[0m  [0;31m# unused hook - call anyway for backward compatibility[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/call.py[0m in [0;36m_call_strategy_hook[0;34m(trainer, hook_name, *args, **kwargs)[0m
[1;32m    328[0m     [0;32mwith[0m [0mtrainer[0m[0;34m.[0m[0mprofiler[0m[0;34m.[0m[0mprofile[0m[0;34m([0m[0;34mf"[Strategy]{trainer.strategy.__class__.__name__}.{hook_name}"[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 329[0;31m         [0moutput[0m [0;34m=[0m [0mfn[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    330[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_lightning/strategies/strategy.py[0m in [0;36mtraining_step[0;34m(self, *args, **kwargs)[0m
[1;32m    390[0m                 [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_forward_redirection[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mmodel[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0mlightning_module[0m[0;34m,[0m [0;34m"training_step"[0m[0;34m,[0m [0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 391[0;31m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0mlightning_module[0m[0;34m.[0m[0mtraining_step[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    392[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/_dynamo/eval_frame.py[0m in [0;36m_fn[0;34m(*args, **kwargs)[0m
[1;32m    573[0m             [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 574[0;31m                 [0;32mreturn[0m [0mfn[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    575[0m             [0;32mfinally[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/130411005.py[0m in [0;36mtraining_step[0;34m(self, batch, batch_idx)[0m
[1;32m     32[0m [0;34m[0m[0m
[0;32m---> 33[0;31m     [0;32mdef[0m [0mtraining_step[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mbatch[0m[0;34m,[0m [0mbatch_idx[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     34[0m         [0mX[0m[0;34m,[0m [0my[0m [0;34m=[0m [0mbatch[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py[0m in [0;36m_wrapped_call_impl[0;34m(self, *args, **kwargs)[0m
[1;32m   1738[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1739[0;31m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_call_impl[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1740[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py[0m in [0;36m_call_impl[0;34m(self, *args, **kwargs)[0m
[1;32m   1749[0m                 or _global_forward_hooks or _global_forward_pre_hooks):
[0;32m-> 1750[0;31m             [0;32mreturn[0m [0mforward_call[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1751[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torchmetrics/metric.py[0m in [0;36mforward[0;34m(self, *args, **kwargs)[0m
[1;32m    314[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 315[0;31m             [0mself[0m[0;34m.[0m[0m_forward_cache[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_forward_reduce_state_update[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    316[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py[0m in [0;36m__call__[0;34m(self, frame, cache_entry, frame_state)[0m
[1;32m   1379[0m             [0;31m# skip=1: skip this frame[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1380[0;31m             return self._torchdynamo_orig_callable(
[0m[1;32m   1381[0m                 [0mframe[0m[0;34m,[0m [0mcache_entry[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0mhooks[0m[0;34m,[0m [0mframe_state[0m[0;34m,[0m [0mskip[0m[0;34m=[0m[0;36m1[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py[0m in [0;36m__call__[0;34m(self, frame, cache_entry, hooks, frame_state, skip)[0m
[1;32m   1163[0m         [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1164[0;31m             result = self._inner_convert(
[0m[1;32m   1165[0m                 [0mframe[0m[0;34m,[0m [0mcache_entry[0m[0;34m,[0m [0mhooks[0m[0;34m,[0m [0mframe_state[0m[0;34m,[0m [0mskip[0m[0;34m=[0m[0mskip[0m [0;34m+[0m [0;36m1[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py[0m in [0;36m__call__[0;34m(self, frame, cache_entry, hooks, frame_state, skip)[0m
[1;32m    546[0m         [0;32mwith[0m [0mcompile_context[0m[0;34m([0m[0mCompileContext[0m[0;34m([0m[0mcompile_id[0m[0;34m)[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 547[0;31m             return _compile(
[0m[1;32m    548[0m                 [0mframe[0m[0;34m.[0m[0mf_code[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py[0m in [0;36m_compile[0;34m(code, globals, locals, builtins, closure, compiler_fn, one_graph, export, export_constraints, hooks, cache_entry, cache_size, frame, frame_state, compile_id, skip)[0m
[1;32m   1035[0m                 [0;31m# Rewrap for clarity[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1036[0;31m                 raise InternalTorchDynamoError(
[0m[1;32m   1037[0m                     [0;34mf"{type(e).__qualname__}: {str(e)}"[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py[0m in [0;36m_compile[0;34m(code, globals, locals, builtins, closure, compiler_fn, one_graph, export, export_constraints, hooks, cache_entry, cache_size, frame, frame_state, compile_id, skip)[0m
[1;32m    985[0m         [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 986[0;31m             [0mguarded_code[0m [0;34m=[0m [0mcompile_inner[0m[0;34m([0m[0mcode[0m[0;34m,[0m [0mone_graph[0m[0;34m,[0m [0mhooks[0m[0;34m,[0m [0mtransform[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    987[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py[0m in [0;36mcompile_inner[0;34m(code, one_graph, hooks, transform)[0m
[1;32m    714[0m             [0mstack[0m[0;34m.[0m[0menter_context[0m[0;34m([0m[0mCompileTimeInstructionCounter[0m[0;34m.[0m[0mrecord[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 715[0;31m             [0;32mreturn[0m [0m_compile_inner[0m[0;34m([0m[0mcode[0m[0;34m,[0m [0mone_graph[0m[0;34m,[0m [0mhooks[0m[0;34m,[0m [0mtransform[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    716[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/_utils_internal.py[0m in [0;36mwrapper_function[0;34m(*args, **kwargs)[0m
[1;32m     94[0m             [0;32mif[0m [0;32mnot[0m [0mStrobelightCompileTimeProfiler[0m[0;34m.[0m[0menabled[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 95[0;31m                 [0;32mreturn[0m [0mfunction[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     96[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py[0m in [0;36m_compile_inner[0;34m(code, one_graph, hooks, transform)[0m
[1;32m    749[0m             [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 750[0;31m                 [0mout_code[0m [0;34m=[0m [0mtransform_code_object[0m[0;34m([0m[0mcode[0m[0;34m,[0m [0mtransform[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    751[0m                 [0;32mbreak[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/_dynamo/bytecode_transformation.py[0m in [0;36mtransform_code_object[0;34m(code, transformations, safe)[0m
[1;32m   1360[0m [0;34m[0m[0m
[0;32m-> 1361[0;31m     [0mtransformations[0m[0;34m([0m[0minstructions[0m[0;34m,[0m [0mcode_options[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1362[0m     [0;32mreturn[0m [0mclean_and_assemble_instructions[0m[0;34m([0m[0minstructions[0m[0;34m,[0m [0mkeys[0m[0;34m,[0m [0mcode_options[0m[0;34m)[0m[0;34m[[0m[0;36m1[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py[0m in [0;36m_fn[0;34m(*args, **kwargs)[0m
[1;32m    230[0m             [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 231[0;31m                 [0;32mreturn[0m [0mfn[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    232[0m             [0;32mfinally[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/_dynamo/convert_frame.py[0m in [0;36mtransform[0;34m(instructions, code_options)[0m
[1;32m    661[0m             [0;32mwith[0m [0mtracing[0m[0;34m([0m[0mtracer[0m[0;34m.[0m[0moutput[0m[0;34m.[0m[0mtracing_context[0m[0;34m)[0m[0;34m,[0m [0mtracer[0m[0;34m.[0m[0mset_current_tx[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 662[0;31m                 [0mtracer[0m[0;34m.[0m[0mrun[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    663[0m         [0;32mexcept[0m [0mexc[0m[0;34m.[0m[0mUnspecializeRestartAnalysis[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py[0m in [0;36mrun[0;34m(self)[0m
[1;32m   2867[0m     [0;32mdef[0m [0mrun[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 2868[0;31m         [0msuper[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0mrun[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   2869[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py[0m in [0;36mrun[0;34m(self)[0m
[1;32m   1051[0m                 [0mself[0m[0;34m.[0m[0moutput[0m[0;34m.[0m[0mpush_tx[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1052[0;31m                 [0;32mwhile[0m [0mself[0m[0;34m.[0m[0mstep[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1053[0m                     [0;32mpass[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py[0m in [0;36mstep[0;34m(self)[0m
[1;32m    961[0m         [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 962[0;31m             [0mself[0m[0;34m.[0m[0mdispatch_table[0m[0;34m[[0m[0minst[0m[0;34m.[0m[0mopcode[0m[0;34m][0m[0;34m([0m[0mself[0m[0;34m,[0m [0minst[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    963[0m             [0;32mreturn[0m [0;32mnot[0m [0mself[0m[0;34m.[0m[0moutput[0m[0;34m.[0m[0mshould_exit[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py[0m in [0;36mwrapper[0;34m(self, inst)[0m
[1;32m    658[0m             [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 659[0;31m                 [0;32mreturn[0m [0minner_fn[0m[0;34m([0m[0mself[0m[0;34m,[0m [0minst[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    660[0m             [0;32mexcept[0m [0mUnsupported[0m [0;32mas[0m [0mexcp[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py[0m in [0;36mCALL[0;34m(self, inst)[0m
[1;32m   2340[0m     [0;32mdef[0m [0mCALL[0m[0;34m([0m[0mself[0m[0;34m,[0m [0minst[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 2341[0;31m         [0mself[0m[0;34m.[0m[0m_call[0m[0;34m([0m[0minst[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   2342[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py[0m in [0;36m_call[0;34m(self, inst, call_kw)[0m
[1;32m   2334[0m             [0;31m# a subsequent call may have self.kw_names set to an old value[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 2335[0;31m             [0mself[0m[0;34m.[0m[0mcall_function[0m[0;34m([0m[0mfn[0m[0;34m,[0m [0margs[0m[0;34m,[0m [0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   2336[0m         [0;32mfinally[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py[0m in [0;36mcall_function[0;34m(self, fn, args, kwargs)[0m
[1;32m    896[0m             [0;32mraise[0m [0mAssertionError[0m[0;34m([0m[0;34mf"Attempt to trace forbidden callable {inner_fn}"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 897[0;31m         [0mself[0m[0;34m.[0m[0mpush[0m[0;34m([0m[0mfn[0m[0;34m.[0m[0mcall_function[0m[0;34m([0m[0mself[0m[0;34m,[0m [0margs[0m[0;34m,[0m [0mkwargs[0m[0;34m)[0m[0;34m)[0m  [0;31m# type: ignore[arg-type][0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    898[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/_dynamo/variables/functions.py[0m in [0;36mcall_function[0;34m(self, tx, args, kwargs)[0m
[1;32m    377[0m             [0;32mreturn[0m [0minvoke_and_store_as_constant[0m[0;34m([0m[0mtx[0m[0;34m,[0m [0mfn[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0mget_name[0m[0;34m([0m[0;34m)[0m[0;34m,[0m [0margs[0m[0;34m,[0m [0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 378[0;31m         [0;32mreturn[0m [0msuper[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0mcall_function[0m[0;34m([0m[0mtx[0m[0;34m,[0m [0margs[0m[0;34m,[0m [0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    379[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/_dynamo/variables/functions.py[0m in [0;36mcall_function[0;34m(self, tx, args, kwargs)[0m
[1;32m    316[0m                     [0;32mreturn[0m [0msuper[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0mcall_function[0m[0;34m([0m[0mtx[0m[0;34m,[0m [0margs[0m[0;34m,[0m [0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 317[0;31m         [0;32mreturn[0m [0msuper[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0mcall_function[0m[0;34m([0m[0mtx[0m[0;34m,[0m [0margs[0m[0;34m,[0m [0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    318[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/_dynamo/variables/functions.py[0m in [0;36mcall_function[0;34m(self, tx, args, kwargs)[0m
[1;32m    117[0m     ) -> "VariableTracker":
[0;32m--> 118[0;31m         [0;32mreturn[0m [0mtx[0m[0;34m.[0m[0minline_user_function_return[0m[0;34m([0m[0mself[0m[0;34m,[0m [0;34m[[0m[0;34m*[0m[0mself[0m[0;34m.[0m[0mself_args[0m[0;34m([0m[0;34m)[0m[0;34m,[0m [0;34m*[0m[0margs[0m[0;34m][0m[0;34m,[0m [0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    119[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py[0m in [0;36minline_user_function_return[0;34m(self, fn, args, kwargs)[0m
[1;32m    902[0m         """
[0;32m--> 903[0;31m         [0;32mreturn[0m [0mInliningInstructionTranslator[0m[0;34m.[0m[0minline_call[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mfn[0m[0;34m,[0m [0margs[0m[0;34m,[0m [0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    904[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py[0m in [0;36minline_call[0;34m(cls, parent, func, args, kwargs)[0m
[1;32m   3071[0m         [0;32mwith[0m [0mpatch[0m[0;34m.[0m[0mdict[0m[0;34m([0m[0mcounters[0m[0;34m,[0m [0;34m{[0m[0;34m"unimplemented"[0m[0;34m:[0m [0mcounters[0m[0;34m[[0m[0;34m"inline_call"[0m[0;34m][0m[0;34m}[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 3072[0;31m             [0;32mreturn[0m [0mcls[0m[0;34m.[0m[0minline_call_[0m[0;34m([0m[0mparent[0m[0;34m,[0m [0mfunc[0m[0;34m,[0m [0margs[0m[0;34m,[0m [0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   3073[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py[0m in [0;36minline_call_[0;34m(parent, func, args, kwargs)[0m
[1;32m   3197[0m             [0;32mwith[0m [0mstrict_ctx[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 3198[0;31m                 [0mtracer[0m[0;34m.[0m[0mrun[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   3199[0m         [0;32mexcept[0m [0mexc[0m[0;34m.[0m[0mObservedException[0m [0;32mas[0m [0me[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py[0m in [0;36mrun[0;34m(self)[0m
[1;32m   1051[0m                 [0mself[0m[0;34m.[0m[0moutput[0m[0;34m.[0m[0mpush_tx[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1052[0;31m                 [0;32mwhile[0m [0mself[0m[0;34m.[0m[0mstep[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1053[0m                     [0;32mpass[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py[0m in [0;36mstep[0;34m(self)[0m
[1;32m    961[0m         [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 962[0;31m             [0mself[0m[0;34m.[0m[0mdispatch_table[0m[0;34m[[0m[0minst[0m[0;34m.[0m[0mopcode[0m[0;34m][0m[0;34m([0m[0mself[0m[0;34m,[0m [0minst[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    963[0m             [0;32mreturn[0m [0;32mnot[0m [0mself[0m[0;34m.[0m[0moutput[0m[0;34m.[0m[0mshould_exit[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/_dynamo/symbolic_convert.py[0m in [0;36mSTORE_FAST[0;34m(self, inst)[0m
[1;32m   1121[0m         [0mloaded_vt[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mpop[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1122[0;31m         [0mloaded_vt[0m[0;34m.[0m[0mset_name_hint[0m[0;34m([0m[0mname[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1123[0m         [0mself[0m[0;34m.[0m[0msymbolic_locals[0m[0;34m[[0m[0mname[0m[0;34m][0m [0;34m=[0m [0mloaded_vt[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/_dynamo/variables/lazy.py[0m in [0;36mrealize_and_forward[0;34m(self, *args, **kwargs)[0m
[1;32m    169[0m     ) -> Any:
[0;32m--> 170[0;31m         [0;32mreturn[0m [0mgetattr[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mrealize[0m[0;34m([0m[0;34m)[0m[0;34m,[0m [0mname[0m[0;34m)[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    171[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/_dynamo/variables/lazy.py[0m in [0;36mrealize[0;34m(self)[0m
[1;32m     63[0m         [0;32mif[0m [0mself[0m[0;34m.[0m[0m_cache[0m[0;34m.[0m[0mvt[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 64[0;31m             [0mself[0m[0;34m.[0m[0m_cache[0m[0;34m.[0m[0mrealize[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     65[0m             [0;32massert[0m [0mself[0m[0;34m.[0m[0m_cache[0m[0;34m.[0m[0mvt[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/_dynamo/variables/lazy.py[0m in [0;36mrealize[0;34m(self)[0m
[1;32m     30[0m [0;34m[0m[0m
[0;32m---> 31[0;31m         [0mself[0m[0;34m.[0m[0mvt[0m [0;34m=[0m [0mVariableTracker[0m[0;34m.[0m[0mbuild[0m[0;34m([0m[0mtx[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0mvalue[0m[0;34m,[0m [0msource[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     32[0m         [0;32mdel[0m [0mself[0m[0;34m.[0m[0mvalue[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/_dynamo/variables/base.py[0m in [0;36mbuild[0;34m(tx, value, source)[0m
[1;32m    456[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 457[0;31m             [0;32mreturn[0m [0mbuilder[0m[0;34m.[0m[0mVariableBuilder[0m[0;34m([0m[0mtx[0m[0;34m,[0m [0msource[0m[0;34m)[0m[0;34m([0m[0mvalue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    458[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/_dynamo/variables/builder.py[0m in [0;36m__call__[0;34m(self, value)[0m
[1;32m    387[0m [0;34m[0m[0m
[0;32m--> 388[0;31m         [0mvt[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_wrap[0m[0;34m([0m[0mvalue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    389[0m         [0mvt[0m[0;34m.[0m[0msource[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0msource[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/_dynamo/variables/builder.py[0m in [0;36m_wrap[0;34m(self, value)[0m
[1;32m    566[0m         [0;32mif[0m [0mtype_dispatch[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 567[0;31m             [0;32mreturn[0m [0mtype_dispatch[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mvalue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    568[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/_dynamo/variables/builder.py[0m in [0;36mwrap_tensor[0;34m(self, value)[0m
[1;32m   1653[0m [0;34m[0m[0m
[0;32m-> 1654[0;31m         example_value = wrap_to_fake_tensor_and_record(
[0m[1;32m   1655[0m             [0mvalue[0m[0;34m,[0m [0mtx[0m[0;34m=[0m[0mself[0m[0;34m.[0m[0mtx[0m[0;34m,[0m [0mis_tensor[0m[0;34m=[0m[0;32mTrue[0m[0;34m,[0m [0msource[0m[0;34m=[0m[0msource[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/_dynamo/variables/builder.py[0m in [0;36mwrap_to_fake_tensor_and_record[0;34m(e, tx, source, is_tensor, parent_context)[0m
[1;32m   2859[0m         )
[0;32m-> 2860[0;31m         fake_e = wrap_fake_exception(
[0m[1;32m   2861[0m             lambda: tx.fake_mode.from_tensor(

[0;32m/usr/local/lib/python3.11/dist-packages/torch/_dynamo/utils.py[0m in [0;36mwrap_fake_exception[0;34m(fn)[0m
[1;32m   2016[0m     [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 2017[0;31m         [0;32mreturn[0m [0mfn[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   2018[0m     [0;32mexcept[0m [0mUnsupportedFakeTensorException[0m [0;32mas[0m [0me[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/_dynamo/variables/builder.py[0m in [0;36m<lambda>[0;34m()[0m
[1;32m   2860[0m         fake_e = wrap_fake_exception(
[0;32m-> 2861[0;31m             lambda: tx.fake_mode.from_tensor(
[0m[1;32m   2862[0m                 [0me[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/_subclasses/fake_tensor.py[0m in [0;36mfrom_tensor[0;34m(self, tensor, static_shapes, source, symbolic_context, trace)[0m
[1;32m   2609[0m             [0mshape_env[0m [0;34m=[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 2610[0;31m         return self.fake_tensor_converter.from_real_tensor(
[0m[1;32m   2611[0m             [0mself[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/_subclasses/fake_tensor.py[0m in [0;36mfrom_real_tensor[0;34m(self, fake_mode, t, make_constant, shape_env, source, symbolic_context, trace)[0m
[1;32m    387[0m [0;34m[0m[0m
[0;32m--> 388[0;31m         out = self.meta_converter(
[0m[1;32m    389[0m             [0mt[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/_subclasses/meta_utils.py[0m in [0;36m__call__[0;34m(self, t, shape_env, callback, source, symbolic_context, trace)[0m
[1;32m   1849[0m         [0;31m# to query them when figuring out what to put in here[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1850[0;31m         [0mt_desc[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mdescriber[0m[0;34m.[0m[0mdescribe_tensor[0m[0;34m([0m[0mt[0m[0;34m,[0m [0mtrace[0m[0;34m=[0m[0mtrace[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1851[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/_subclasses/meta_utils.py[0m in [0;36mdescribe_tensor[0;34m(self, t, recurse, trace)[0m
[1;32m    293[0m             [0;31m# put it in for accuracy[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 294[0;31m             [0mstorage[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mdescribe_storage[0m[0;34m([0m[0mt[0m[0;34m.[0m[0muntyped_storage[0m[0;34m([0m[0;34m)[0m[0;34m,[0m [0mtrace[0m[0;34m=[0m[0mtrace[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    295[0m             [0mstorage_offset[0m [0;34m=[0m [0mt[0m[0;34m.[0m[0mstorage_offset[0m[0;34m([0m[0;34m)[0m  [0;31m# type: ignore[assignment][0m[0;34m[0m[0;34m[0m[0m

[0;31mInternalTorchDynamoError[0m: RuntimeError: Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run. Stack trace: File "/usr/local/lib/python3.11/dist-packages/torchmetrics/metric.py", line 390, in torch_dynamo_resume_in__forward_reduce_state_update_at_385
    self._reduce_states(global_state)
  File "/usr/local/lib/python3.11/dist-packages/torchmetrics/metric.py", line 479, in _reduce_states
    reduced = global_state + local_state. To prevent overwriting, clone the tensor outside of torch.compile() or call torch.compiler.cudagraph_mark_step_begin() before each model invocation.

from user code:
   File "/usr/local/lib/python3.11/dist-packages/torchmetrics/metric.py", line 372, in _forward_reduce_state_update
    global_state = self._copy_state_dict()
  File "/usr/local/lib/python3.11/dist-packages/torchmetrics/metric.py", line 962, in _copy_state_dict
    current_value = getattr(self, attr)

Set TORCH_LOGS="+dynamo" and TORCHDYNAMO_VERBOSE=1 for more information


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True


During handling of the above exception, another exception occurred:

[0;31mRuntimeError[0m                              Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3921307745.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     62[0m )
[1;32m     63[0m [0;34m[0m[0m
[0;32m---> 64[0;31m [0mmodel[0m[0;34m,[0m [0m_[0m [0;34m=[0m [0mtrain_ann[0m[0;34m([0m[0mtrain_ds[0m[0;34m,[0m [0mvalid_ds[0m[0;34m,[0m [0mModel[0m[0;34m=[0m[0mModel[0m[0;34m,[0m [0minput_shape[0m[0;34m=[0m[0mX_train_np[0m[0;34m.[0m[0mshape[0m[0;34m[[0m[0;36m1[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     65[0m [0;34m[0m[0m
[1;32m     66[0m [0mX_valid_t[0m [0;34m=[0m [0mtorch[0m[0;34m.[0m[0mfrom_numpy[0m[0;34m([0m[0mnp[0m[0;34m.[0m[0mascontiguousarray[0m[0;34m([0m[0mX_valid_np[0m[0;34m,[0m [0mdtype[0m[0;34m=[0m[0mnp[0m[0;34m.[0m[0mfloat32[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/3921307745.py[0m in [0;36mtrain_ann[0;34m(train_ds, valid_ds, Model, input_shape)[0m
[1;32m     42[0m     )
[1;32m     43[0m [0;34m[0m[0m
[0;32m---> 44[0;31m     [0mtrainer[0m[0;34m.[0m[0mfit[0m[0;34m([0m[0mmodel[0m[0;34m,[0m [0mtrain_ds[0m[0;34m,[0m [0mvalid_ds[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     45[0m [0;34m[0m[0m
[1;32m     46[0m     [0;32mif[0m [0mbest_state_callback[0m[0;34m.[0m[0mbest_state_dict[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/trainer.py[0m in [0;36mfit[0;34m(self, model, train_dataloaders, val_dataloaders, datamodule, ckpt_path)[0m
[1;32m    558[0m         [0mself[0m[0;34m.[0m[0mtraining[0m [0;34m=[0m [0;32mTrue[0m[0;34m[0m[0;34m[0m[0m
[1;32m    559[0m         [0mself[0m[0;34m.[0m[0mshould_stop[0m [0;34m=[0m [0;32mFalse[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 560[0;31m         call._call_and_handle_interrupt(
[0m[1;32m    561[0m             [0mself[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0m_fit_impl[0m[0;34m,[0m [0mmodel[0m[0;34m,[0m [0mtrain_dataloaders[0m[0;34m,[0m [0mval_dataloaders[0m[0;34m,[0m [0mdatamodule[0m[0;34m,[0m [0mckpt_path[0m[0;34m[0m[0;34m[0m[0m
[1;32m    562[0m         )

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/call.py[0m in [0;36m_call_and_handle_interrupt[0;34m(trainer, trainer_fn, *args, **kwargs)[0m
[1;32m     68[0m     [0;32mexcept[0m [0mBaseException[0m [0;32mas[0m [0mexception[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     69[0m         [0m_interrupt[0m[0;34m([0m[0mtrainer[0m[0;34m,[0m [0mexception[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 70[0;31m         [0mtrainer[0m[0;34m.[0m[0m_teardown[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     71[0m         [0;31m# teardown might access the stage so we reset it after[0m[0;34m[0m[0;34m[0m[0m
[1;32m     72[0m         [0mtrainer[0m[0;34m.[0m[0mstate[0m[0;34m.[0m[0mstage[0m [0;34m=[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/trainer.py[0m in [0;36m_teardown[0;34m(self)[0m
[1;32m   1032[0m         """This is the Trainer's internal teardown, unrelated to the `teardown` hooks in LightningModule and Callback;
[1;32m   1033[0m         those are handled by :meth:`_call_teardown_hook`."""
[0;32m-> 1034[0;31m         [0mself[0m[0;34m.[0m[0mstrategy[0m[0;34m.[0m[0mteardown[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1035[0m         [0mloop[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_active_loop[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1036[0m         [0;31m# loop should never be `None` here but it can because we don't know the trainer stage with `ddp_spawn`[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_lightning/strategies/strategy.py[0m in [0;36mteardown[0;34m(self)[0m
[1;32m    534[0m         [0;32mif[0m [0mself[0m[0;34m.[0m[0mlightning_module[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    535[0m             [0mlog[0m[0;34m.[0m[0mdebug[0m[0;34m([0m[0;34mf"{self.__class__.__name__}: moving model to CPU"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 536[0;31m             [0mself[0m[0;34m.[0m[0mlightning_module[0m[0;34m.[0m[0mcpu[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    537[0m         [0mself[0m[0;34m.[0m[0mprecision_plugin[0m[0;34m.[0m[0mteardown[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    538[0m         [0;32massert[0m [0mself[0m[0;34m.[0m[0maccelerator[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/lightning_fabric/utilities/device_dtype_mixin.py[0m in [0;36mcpu[0;34m(self)[0m
[1;32m     80[0m         [0;34m"""See :meth:`torch.nn.Module.cpu`."""[0m[0;34m[0m[0;34m[0m[0m
[1;32m     81[0m         [0m_update_properties[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mdevice[0m[0;34m=[0m[0mtorch[0m[0;34m.[0m[0mdevice[0m[0;34m([0m[0;34m"cpu"[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 82[0;31m         [0;32mreturn[0m [0msuper[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0mcpu[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     83[0m [0;34m[0m[0m
[1;32m     84[0m     [0;34m@[0m[0moverride[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py[0m in [0;36mcpu[0;34m(self)[0m
[1;32m   1119[0m             [0mModule[0m[0;34m:[0m [0mself[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1120[0m         """
[0;32m-> 1121[0;31m         [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_apply[0m[0;34m([0m[0;32mlambda[0m [0mt[0m[0;34m:[0m [0mt[0m[0;34m.[0m[0mcpu[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1122[0m [0;34m[0m[0m
[1;32m   1123[0m     [0;32mdef[0m [0mtype[0m[0;34m([0m[0mself[0m[0;34m:[0m [0mT[0m[0;34m,[0m [0mdst_type[0m[0;34m:[0m [0mUnion[0m[0;34m[[0m[0mdtype[0m[0;34m,[0m [0mstr[0m[0;34m][0m[0;34m)[0m [0;34m->[0m [0mT[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py[0m in [0;36m_apply[0;34m(self, fn, recurse)[0m
[1;32m    901[0m         [0;32mif[0m [0mrecurse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    902[0m             [0;32mfor[0m [0mmodule[0m [0;32min[0m [0mself[0m[0;34m.[0m[0mchildren[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 903[0;31m                 [0mmodule[0m[0;34m.[0m[0m_apply[0m[0;34m([0m[0mfn[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    904[0m [0;34m[0m[0m
[1;32m    905[0m         [0;32mdef[0m [0mcompute_should_use_set_data[0m[0;34m([0m[0mtensor[0m[0;34m,[0m [0mtensor_applied[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py[0m in [0;36m_apply[0;34m(self, fn, recurse)[0m
[1;32m    965[0m             [0;32mif[0m [0mparam_grad[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    966[0m                 [0;32mwith[0m [0mtorch[0m[0;34m.[0m[0mno_grad[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 967[0;31m                     [0mgrad_applied[0m [0;34m=[0m [0mfn[0m[0;34m([0m[0mparam_grad[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    968[0m                 g_should_use_set_data = compute_should_use_set_data(
[1;32m    969[0m                     [0mparam_grad[0m[0;34m,[0m [0mgrad_applied[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py[0m in [0;36m<lambda>[0;34m(t)[0m
[1;32m   1119[0m             [0mModule[0m[0;34m:[0m [0mself[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1120[0m         """
[0;32m-> 1121[0;31m         [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_apply[0m[0;34m([0m[0;32mlambda[0m [0mt[0m[0;34m:[0m [0mt[0m[0;34m.[0m[0mcpu[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1122[0m [0;34m[0m[0m
[1;32m   1123[0m     [0;32mdef[0m [0mtype[0m[0;34m([0m[0mself[0m[0;34m:[0m [0mT[0m[0;34m,[0m [0mdst_type[0m[0;34m:[0m [0mUnion[0m[0;34m[[0m[0mdtype[0m[0;34m,[0m [0mstr[0m[0;34m][0m[0;34m)[0m [0;34m->[0m [0mT[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;31mRuntimeError[0m: Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run. Stack trace: File "/tmp/ipykernel_11/130411005.py", line 35, in training_step
    y_hat = self(X).squeeze(1)
  File "/tmp/ipykernel_11/130411005.py", line 24, in forward
    x = self.swish(self.input(x)). To prevent overwriting, clone the tensor outside of torch.compile() or call torch.compiler.cudagraph_mark_step_begin() before each model invocation.

## === cell 28
X_train_t = torch.from_numpy(np.ascontiguousarray(X_train_np, dtype=np.float32))
with torch.no_grad():
    accuracy_score(
        y_train_np,
        np.argmax(model(X_train_t).detach().cpu().numpy(), axis=1),
    )
