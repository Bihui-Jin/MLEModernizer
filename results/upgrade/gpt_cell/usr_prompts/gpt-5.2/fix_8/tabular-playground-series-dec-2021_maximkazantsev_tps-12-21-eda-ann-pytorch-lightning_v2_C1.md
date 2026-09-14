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
from sklearn.model_selection import StratifiedKFold, train_test_split
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

train = pd.read_csv(
    "/kaggle/input/tabular-playground-series-dec-2021/train.csv",
    low_memory=False,
    dtype=DTYPES,
)
test = pd.read_csv(
    "/kaggle/input/tabular-playground-series-dec-2021/test.csv",
    low_memory=False,
    dtype={k: v for k, v in DTYPES.items() if k != "Cover_Type"},
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
target = "Cover_Type"
features = list(train.columns[1:54])




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
train.drop(train[train[target] == 5].index, axis=0, inplace=True)
train.reset_index(drop=True, inplace=True)
label_enc = LabelEncoder()
NUM_CLASSES = train[target].nunique()




## === cell 20
s_scaler = StandardScaler()
train_num = train[num_features].to_numpy(dtype=np.float32, copy=False)
test_num = test[num_features].to_numpy(dtype=np.float32, copy=False)
train[num_features] = s_scaler.fit_transform(train_num)
test[num_features] = s_scaler.transform(test_num)




## === cell 21
X_nn = train[features]
X_test_nn_df = test[features]
y = pd.Series(label_enc.fit_transform(train[target]))




## === cell 22
mm_scaler = MinMaxScaler()
X_nn_np = mm_scaler.fit_transform(X_nn.to_numpy(dtype=np.float32, copy=False)).astype(
    np.float32, copy=False
)
X_test_np = mm_scaler.transform(
    X_test_nn_df.to_numpy(dtype=np.float32, copy=False)
).astype(np.float32, copy=False)

X_nn = pd.DataFrame(X_nn_np, columns=features)
X_test_nn = torch.from_numpy(X_test_np)  # float32 already

del X_test_nn_df, train_num, test_num, X_nn_np, X_test_np
gc.collect()




## === cell 23
BATCH_SIZE = 4096




## === cell 24
def prepare_datasets(X_nn, X_valid_nn, y_nn, y_valid_nn, batch_size=BATCH_SIZE):
    X_nn_t = torch.from_numpy(X_nn.to_numpy(dtype=np.float32, copy=False))
    y_nn_t = torch.from_numpy(y_nn.to_numpy(dtype=np.int64, copy=False))
    X_valid_nn_t = torch.from_numpy(X_valid_nn.to_numpy(dtype=np.float32, copy=False))
    y_valid_nn_t = torch.from_numpy(y_valid_nn.to_numpy(dtype=np.int64, copy=False))

    train_ds = TensorDataset(X_nn_t, y_nn_t)
    valid_ds = TensorDataset(X_valid_nn_t, y_valid_nn_t)

    nw = min(8, (os.cpu_count() or 2))
    pin = torch.cuda.is_available()
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,  # Lightning handles shuffling? keep False to preserve original behavior (it was default False for TensorDataset)
        drop_last=False,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=(nw > 0),
        prefetch_factor=2 if nw > 0 else None,
    )
    valid_dl = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=(nw > 0),
        prefetch_factor=2 if nw > 0 else None,
    )
    return train_dl, valid_dl




## === cell 25
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data)




## === cell 26
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

    def on_validation_epoch_end(self, trainer, module):
        metrics_logs = trainer.logged_metrics
        if "val_loss" in metrics_logs:
            self.val_loss.append(metrics_logs["val_loss"].item())
        if "val_acc" in metrics_logs:
            self.val_acc.append(metrics_logs["val_acc"].item())

    def on_train_epoch_end(self, trainer, module):
        metrics_logs = trainer.logged_metrics
        if "loss_epoch" in metrics_logs:
            self.train_loss.append(metrics_logs["loss_epoch"].item())
        if "train_acc" in metrics_logs:
            self.train_acc.append(metrics_logs["train_acc"].item())

        if (
            self.verbose
            and (module.current_epoch % 5 == 0)
            and self.train_loss
            and self.val_loss
        ):
            print(
                f"Epoch {module.current_epoch} start learning rate: {self.lr_epoch_start[-1]:.6f}, "
                f"train_loss: {self.train_loss[-1]:.4f}, "
                f"train_acc: {self.train_acc[-1]:.4f}, "
                f"val_loss: {self.val_loss[-1]:.4f}, "
                f"val_acc: {self.val_acc[-1]:.4f}"
            )




## === cell 27
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
        self.log("loss", loss, prog_bar=True, on_epoch=True, logger=True)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
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
        self.log("val_loss", val_loss, prog_bar=True, on_epoch=True, logger=True)
        return {"val_loss": val_loss}

    def validation_epoch_end(self, outputs):
        val_acc = self.val_acc_metric.compute()
        self.log(
            "val_acc", val_acc, prog_bar=True, on_epoch=True, on_step=False, logger=True
        )
        self.val_acc_metric.reset()
        return {"val_acc": val_acc}

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




## === cell 28
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


def train_ann(train_ds, valid_ds, Model=Model, input_shape=X_nn.shape[1]):

    model = Model(input_shape)
    model.apply(initialize_weights)

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
        accelerator=accelerator,
        devices=devices,
        deterministic=True,
    )

    trainer.fit(model, train_ds, valid_ds)

    if best_state_callback.best_state_dict is not None:
        model.load_state_dict(best_state_callback.best_state_dict, strict=True)

    model.eval()
    return model, params_tracker_callback


X_train, X_valid, y_train, y_valid = train_test_split(
    X_nn, y, test_size=0.2, random_state=42, stratify=y
)

train_ds, valid_ds = prepare_datasets(
    X_train, X_valid, y_train, y_valid, batch_size=BATCH_SIZE
)

model, _ = train_ann(train_ds, valid_ds, Model=Model, input_shape=X_nn.shape[1])

X_valid_t = torch.from_numpy(X_valid.to_numpy(dtype=np.float32, copy=False))
with torch.no_grad():
    np.argmax(model(X_valid_t).detach().cpu().numpy(), axis=1)[:10]




## --- ERROR in cell 28, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mNotImplementedError[0m                       Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2274763389.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     78[0m )
[1;32m     79[0m [0;34m[0m[0m
[0;32m---> 80[0;31m [0mmodel[0m[0;34m,[0m [0m_[0m [0;34m=[0m [0mtrain_ann[0m[0;34m([0m[0mtrain_ds[0m[0;34m,[0m [0mvalid_ds[0m[0;34m,[0m [0mModel[0m[0;34m=[0m[0mModel[0m[0;34m,[0m [0minput_shape[0m[0;34m=[0m[0mX_nn[0m[0;34m.[0m[0mshape[0m[0;34m[[0m[0;36m1[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     81[0m [0;34m[0m[0m
[1;32m     82[0m [0mX_valid_t[0m [0;34m=[0m [0mtorch[0m[0;34m.[0m[0mfrom_numpy[0m[0;34m([0m[0mX_valid[0m[0;34m.[0m[0mto_numpy[0m[0;34m([0m[0mdtype[0m[0;34m=[0m[0mnp[0m[0;34m.[0m[0mfloat32[0m[0;34m,[0m [0mcopy[0m[0;34m=[0m[0;32mFalse[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/2274763389.py[0m in [0;36mtrain_ann[0;34m(train_ds, valid_ds, Model, input_shape)[0m
[1;32m     60[0m     )
[1;32m     61[0m [0;34m[0m[0m
[0;32m---> 62[0;31m     [0mtrainer[0m[0;34m.[0m[0mfit[0m[0;34m([0m[0mmodel[0m[0;34m,[0m [0mtrain_ds[0m[0;34m,[0m [0mvalid_ds[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     63[0m [0;34m[0m[0m
[1;32m     64[0m     [0;31m# Restore best weights (equivalent to loading best checkpoint)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/trainer.py[0m in [0;36mfit[0;34m(self, model, train_dataloaders, val_dataloaders, datamodule, ckpt_path)[0m
[1;32m    558[0m         [0mself[0m[0;34m.[0m[0mtraining[0m [0;34m=[0m [0;32mTrue[0m[0;34m[0m[0;34m[0m[0m
[1;32m    559[0m         [0mself[0m[0;34m.[0m[0mshould_stop[0m [0;34m=[0m [0;32mFalse[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 560[0;31m         call._call_and_handle_interrupt(
[0m[1;32m    561[0m             [0mself[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0m_fit_impl[0m[0;34m,[0m [0mmodel[0m[0;34m,[0m [0mtrain_dataloaders[0m[0;34m,[0m [0mval_dataloaders[0m[0;34m,[0m [0mdatamodule[0m[0;34m,[0m [0mckpt_path[0m[0;34m[0m[0;34m[0m[0m
[1;32m    562[0m         )

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/call.py[0m in [0;36m_call_and_handle_interrupt[0;34m(trainer, trainer_fn, *args, **kwargs)[0m
[1;32m     47[0m         [0;32mif[0m [0mtrainer[0m[0;34m.[0m[0mstrategy[0m[0;34m.[0m[0mlauncher[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     48[0m             [0;32mreturn[0m [0mtrainer[0m[0;34m.[0m[0mstrategy[0m[0;34m.[0m[0mlauncher[0m[0;34m.[0m[0mlaunch[0m[0;34m([0m[0mtrainer_fn[0m[0;34m,[0m [0;34m*[0m[0margs[0m[0;34m,[0m [0mtrainer[0m[0;34m=[0m[0mtrainer[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 49[0;31m         [0;32mreturn[0m [0mtrainer_fn[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     50[0m [0;34m[0m[0m
[1;32m     51[0m     [0;32mexcept[0m [0m_TunerExitException[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/trainer.py[0m in [0;36m_fit_impl[0;34m(self, model, train_dataloaders, val_dataloaders, datamodule, ckpt_path)[0m
[1;32m    596[0m             [0mmodel_connected[0m[0;34m=[0m[0mself[0m[0;34m.[0m[0mlightning_module[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    597[0m         )
[0;32m--> 598[0;31m         [0mself[0m[0;34m.[0m[0m_run[0m[0;34m([0m[0mmodel[0m[0;34m,[0m [0mckpt_path[0m[0;34m=[0m[0mckpt_path[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    599[0m [0;34m[0m[0m
[1;32m    600[0m         [0;32massert[0m [0mself[0m[0;34m.[0m[0mstate[0m[0;34m.[0m[0mstopped[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/trainer.py[0m in [0;36m_run[0;34m(self, model, ckpt_path)[0m
[1;32m    959[0m         [0mself[0m[0;34m.[0m[0m_callback_connector[0m[0;34m.[0m[0m_attach_model_logging_functions[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    960[0m [0;34m[0m[0m
[0;32m--> 961[0;31m         [0m_verify_loop_configurations[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    962[0m [0;34m[0m[0m
[1;32m    963[0m         [0;31m# ----------------------------[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/configuration_validator.py[0m in [0;36m_verify_loop_configurations[0;34m(trainer)[0m
[1;32m     34[0m         [0;32mraise[0m [0mValueError[0m[0;34m([0m[0;34m"Unexpected: Trainer state fn must be set before validating loop configuration."[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     35[0m     [0;32mif[0m [0mtrainer[0m[0;34m.[0m[0mstate[0m[0;34m.[0m[0mfn[0m [0;34m==[0m [0mTrainerFn[0m[0;34m.[0m[0mFITTING[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 36[0;31m         [0m__verify_train_val_loop_configuration[0m[0;34m([0m[0mtrainer[0m[0;34m,[0m [0mmodel[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     37[0m         [0m__verify_manual_optimization_support[0m[0;34m([0m[0mtrainer[0m[0;34m,[0m [0mmodel[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     38[0m     [0;32melif[0m [0mtrainer[0m[0;34m.[0m[0mstate[0m[0;34m.[0m[0mfn[0m [0;34m==[0m [0mTrainerFn[0m[0;34m.[0m[0mVALIDATING[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/configuration_validator.py[0m in [0;36m__verify_train_val_loop_configuration[0;34m(trainer, model)[0m
[1;32m     75[0m     [0;31m# check legacy hooks are not present[0m[0;34m[0m[0;34m[0m[0m
[1;32m     76[0m     [0;32mif[0m [0mcallable[0m[0;34m([0m[0mgetattr[0m[0;34m([0m[0mmodel[0m[0;34m,[0m [0;34m"training_epoch_end"[0m[0;34m,[0m [0;32mNone[0m[0;34m)[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 77[0;31m         raise NotImplementedError(
[0m[1;32m     78[0m             [0;34mf"Support for `training_epoch_end` has been removed in v2.0.0. `{type(model).__name__}` implements this"[0m[0;34m[0m[0;34m[0m[0m
[1;32m     79[0m             [0;34m" method. You can use the `on_train_epoch_end` hook instead. To access outputs, save them in-memory as"[0m[0;34m[0m[0;34m[0m[0m

[0;31mNotImplementedError[0m: Support for `training_epoch_end` has been removed in v2.0.0. `Model` implements this method. You can use the `on_train_epoch_end` hook instead. To access outputs, save them in-memory as instance attributes. You can find migration examples in https://github.com/Lightning-AI/pytorch-lightning/pull/16520.

## === cell 29
X_train_t = torch.from_numpy(X_train.to_numpy(dtype=np.float32, copy=False))
with torch.no_grad():
    accuracy_score(
        y_train,
        np.argmax(model(X_train_t).detach().cpu().numpy(), axis=1),
    )
