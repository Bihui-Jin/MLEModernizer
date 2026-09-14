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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import time
import gc
import torchmetrics
import os

pd.set_option("display.max_rows", 150)
pd.set_option("display.max_columns", 500)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.float_format", lambda x: "%.5f" % x)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
pl.seed_everything(SEED, workers=True)

torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.set_num_threads(min(8, os.cpu_count() or 1))

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass




## === cell 1
def _make_dtypes_for_tps_dec2021_train_test():
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
        dtypes[c] = np.int32
    for i in range(1, 5):
        dtypes[f"Wilderness_Area{i}"] = np.int8
    for i in range(1, 41):
        dtypes[f"Soil_Type{i}"] = np.int8
    dtypes["Cover_Type"] = np.int8
    return dtypes


DTYPES = _make_dtypes_for_tps_dec2021_train_test()


def _fast_read_csv(path, dtype, low_memory=False):
    try:
        return pd.read_csv(path, dtype=dtype, low_memory=low_memory, engine="pyarrow")
    except Exception:
        return pd.read_csv(path, dtype=dtype, low_memory=low_memory, engine="c")


train = _fast_read_csv(
    "/kaggle/input/tabular-playground-series-dec-2021/train.csv",
    dtype=DTYPES,
    low_memory=False,
)
test = _fast_read_csv(
    "/kaggle/input/tabular-playground-series-dec-2021/test.csv",
    dtype={k: v for k, v in DTYPES.items() if k != "Cover_Type"},
    low_memory=False,
)




## === cell 2
def reduce_mem_usage(df, verbose=True):
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64", "int8"]
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes
        if col_type not in numerics:
            continue

        if col_type in (np.int8, np.int16, np.float32):
            continue

        c_min = df[col].min()
        c_max = df[col].max()

        if str(col_type)[:3] == "int":
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            else:
                df[col] = df[col].astype(np.int64)
        else:
            if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df




## === cell 3
pass



## === cell 4
pass



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
pass



## === cell 7
target = "Cover_Type"
features = [c for c in train.columns if c not in ["Id", target]]



## === cell 8
pass



## === cell 9
PASS_PLOTTING = True
if not PASS_PLOTTING:
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
if not PASS_PLOTTING:
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
pass



## === cell 12
cat_features = [c for c in features if ("Wilderness_Area" in c) or ("Soil_Type" in c)]
num_features = [c for c in features if c not in cat_features]

print(f"There are {len(cat_features)} categorical features: {cat_features}")
print(f"\nThere are {len(num_features)} continuous features: {num_features}")



## === cell 13
train.isna().sum().sum(), test.isna().sum().sum()



## === cell 14
if not PASS_PLOTTING:
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
                axs[r, c].hist(
                    train[columns[i]].values,
                    range=(df[columns[i]].min(), df[columns[i]].max()),
                    bins=40,
                    color="deepskyblue",
                    edgecolor="black",
                    alpha=0.7,
                    label="Train Dataset",
                )
                axs[r, c].hist(
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
if not PASS_PLOTTING:
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
                axs[r, c].hist(
                    train[columns[i]].values,
                    range=(df[columns[i]].min(), df[columns[i]].max()),
                    bins=40,
                    color="deepskyblue",
                    edgecolor="black",
                    alpha=0.7,
                    label="Train Dataset",
                )
                axs[r, c].hist(
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
if "Soil_Type7" in features:
    features.remove("Soil_Type7")
if "Soil_Type15" in features:
    features.remove("Soil_Type15")
cat_features = [c for c in cat_features if c not in ["Soil_Type7", "Soil_Type15"]]
num_features = [c for c in num_features if c not in ["Soil_Type7", "Soil_Type15"]]



## === cell 18
pass



## === cell 19
train.drop(train[train[target] == 5].index, axis=0, inplace=True)
train.reset_index(drop=True, inplace=True)
label_enc = LabelEncoder()
NUM_CLASSES = train[target].nunique()



## === cell 20
s_scaler = StandardScaler()

train_num_np = train[num_features].to_numpy(dtype=np.float32, copy=False)
test_num_np = test[num_features].to_numpy(dtype=np.float32, copy=False)

train_num_scaled = s_scaler.fit_transform(train_num_np).astype(np.float32, copy=False)
test_num_scaled = s_scaler.transform(test_num_np).astype(np.float32, copy=False)



## === cell 21
cat_np_train = train[cat_features].to_numpy(dtype=np.float32, copy=False)
cat_np_test = test[cat_features].to_numpy(dtype=np.float32, copy=False)

X_nn_arr_pre = np.concatenate([train_num_scaled, cat_np_train], axis=1).astype(
    np.float32, copy=False
)
X_test_arr_pre = np.concatenate([test_num_scaled, cat_np_test], axis=1).astype(
    np.float32, copy=False
)

y = pd.Series(label_enc.fit_transform(train[target]))



## === cell 22
mm_scaler = MinMaxScaler()
X_nn_arr = mm_scaler.fit_transform(X_nn_arr_pre).astype(np.float32, copy=False)
X_test_arr = mm_scaler.transform(X_test_arr_pre).astype(np.float32, copy=False)

X_nn_arr = np.ascontiguousarray(X_nn_arr, dtype=np.float32)
X_test_arr = np.ascontiguousarray(X_test_arr, dtype=np.float32)

X_all_t = torch.from_numpy(X_nn_arr)
y_all_t = torch.from_numpy(np.ascontiguousarray(y.to_numpy(dtype=np.int64, copy=False)))
X_test_nn = torch.from_numpy(X_test_arr)




## === cell 23
BATCH_SIZE = 4096



## === cell 24
from torch.utils.data import Subset


def _suggest_num_workers():
    cpu = os.cpu_count() or 1
    if cpu <= 2:
        return 0
    if cpu <= 4:
        return 2
    return 4


def prepare_datasets_from_tensors(
    X_all_t: torch.Tensor,
    y_all_t: torch.Tensor,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    batch_size: int = BATCH_SIZE,
):
    full_ds = TensorDataset(X_all_t, y_all_t)
    train_ds = Subset(full_ds, train_idx.tolist())
    valid_ds = Subset(full_ds, valid_idx.tolist())

    use_cuda = torch.cuda.is_available()
    num_workers = _suggest_num_workers()
    pin = use_cuda

    train_kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    valid_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    if num_workers > 0:
        train_kwargs.update(dict(persistent_workers=True, prefetch_factor=8))
        valid_kwargs.update(dict(persistent_workers=True, prefetch_factor=8))

    train_dl = DataLoader(train_ds, **train_kwargs)
    valid_dl = DataLoader(valid_ds, **valid_kwargs)
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
        elif "loss" in metrics_logs:
            self.train_loss.append(metrics_logs["loss"].item())

        if "train_acc" in metrics_logs:
            self.train_acc.append(metrics_logs["train_acc"].item())

        if self.verbose == True:
            last_train_loss = (
                self.train_loss[-1] if len(self.train_loss) else float("nan")
            )
            last_train_acc = self.train_acc[-1] if len(self.train_acc) else float("nan")
            last_val_loss = self.val_loss[-1] if len(self.val_loss) else float("nan")
            last_val_acc = self.val_acc[-1] if len(self.val_acc) else float("nan")
            last_lr = (
                self.lr_epoch_start[-1] if len(self.lr_epoch_start) else float("nan")
            )
            print(
                f"Epoch {module.current_epoch} start learning rate: {last_lr:.6f}, "
                f"train_loss: {last_train_loss:.4f}, "
                f"train_acc: {last_train_acc:.4f}, "
                f"val_loss: {last_val_loss:.4f}, "
                f"val_acc: {last_val_acc:.4f}"
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




## === cell 28
def train_ann(train_ds, valid_ds, Model=Model, input_shape=None):
    if input_shape is None:
        raise ValueError("input_shape must be provided")

    model = Model(input_shape)
    model.apply(initialize_weights)

    early_stop_callback = EarlyStopping(
        monitor="val_acc", min_delta=0.00, patience=20, verbose=False, mode="max"
    )
    params_tracker_callback = ParamsTracker(verbose=False)

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
        callbacks=[early_stop_callback, params_tracker_callback],
        enable_progress_bar=False,
        logger=False,
        enable_checkpointing=False,
        deterministic=True,
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=10_000_000,
        enable_model_summary=False,
        inference_mode=True,
    )

    trainer.fit(model, train_ds, valid_ds)
    model.eval()
    return model, params_tracker_callback




## === cell 29
def load_lightning_weights_into_model(model: pl.LightningModule, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    new_state = {}
    for k, v in state.items():
        if k.startswith("model."):
            new_state[k[len("model.") :]] = v
        else:
            new_state[k] = v
    model.load_state_dict(new_state, strict=True)
    return model




## === cell 30
splits = 10
skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)

X_all = X_nn_arr  # already contiguous float32
y_all = np.ascontiguousarray(y.to_numpy(dtype=np.int64, copy=False), dtype=np.int64)

nn_oof_preds = np.zeros((X_all.shape[0],), dtype=np.int64)
nn_test_preds_proba = np.zeros((X_test_nn.shape[0], 6), dtype=np.float32)
total_mean_acc = 0.0

trained_folds = 0

model = None
X_valid_arr = None
y_valid_arr = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def batched_predict_argmax_from_tensor(
    model, X_t: torch.Tensor, device, batch_size: int = 262144
) -> np.ndarray:
    preds_out = np.empty((X_t.shape[0],), dtype=np.int64)
    use_cuda = device.type == "cuda"
    for i in range(0, X_t.shape[0], batch_size):
        xb_cpu = X_t[i : i + batch_size]
        if use_cuda:
            xb_cpu = xb_cpu.pin_memory()
        xb = xb_cpu.to(device, non_blocking=True)
        logits = model(xb)
        preds_out[i : i + batch_size] = torch.argmax(logits, dim=1).cpu().numpy()
    return preds_out


@torch.no_grad()
def batched_predict_proba_from_tensor(
    model, X_t: torch.Tensor, device, batch_size: int = 262144
) -> np.ndarray:
    out = np.empty((X_t.shape[0], 6), dtype=np.float32)
    use_cuda = device.type == "cuda"
    for i in range(0, X_t.shape[0], batch_size):
        xb_cpu = X_t[i : i + batch_size]
        if use_cuda:
            xb_cpu = xb_cpu.pin_memory()
        xb = xb_cpu.to(device, non_blocking=True)
        logits = model(xb)
        out[i : i + batch_size] = torch.softmax(logits, dim=1).cpu().numpy()
    return out


for num, (train_idx, valid_idx) in enumerate(skf.split(X_all, y_all)):
    if num > 0:
        break
    print(f"\n\n===Training with fold {num}")

    X_valid_arr = X_all[valid_idx]
    y_valid_arr = y_all[valid_idx]

    train_ds, valid_ds = prepare_datasets_from_tensors(
        X_all_t, y_all_t, train_idx, valid_idx
    )

    model, tracked_values = train_ann(
        train_ds, valid_ds, Model, input_shape=X_all.shape[1]
    )
    model.eval()
    model = model.to(device)

    preds = batched_predict_argmax_from_tensor(
        model, X_all_t[valid_idx], device=device, batch_size=262144
    )

    fold_score = accuracy_score(y_valid_arr, preds)
    print(f"\n===Fold {num} valid data accuracy score is {fold_score}")

    test_proba_acc = batched_predict_proba_from_tensor(
        model, X_test_nn, device=device, batch_size=262144
    )
    nn_test_preds_proba += test_proba_acc

    nn_oof_preds[valid_idx] = preds
    total_mean_acc += fold_score
    trained_folds += 1

    del train_ds, valid_ds, test_proba_acc, preds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if trained_folds > 0:
    nn_test_preds_proba /= trained_folds
    total_mean_acc /= trained_folds

print(f"Average accuracy score of all models is {total_mean_acc}")



## === cell 31
if model is not None and X_valid_arr is not None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_valid_t_small = X_all_t[:2048]
    if device.type == "cuda":
        X_valid_t_small = X_valid_t_small.pin_memory()
    X_valid_t_small = X_valid_t_small.to(device, non_blocking=True)
    np.argmax(model(X_valid_t_small).detach().cpu().numpy(), axis=1)[:10]
else:
    print("Model not available for sanity check.")



## === cell 32
print("Skipped full train accuracy check to save time.")



## === cell 33
print("Skipped full valid accuracy check to save time.")



## === cell 34
if (
    nn_test_preds_proba is None
    or nn_test_preds_proba.shape[0] != len(test)
    or not np.isfinite(nn_test_preds_proba).all()
    or trained_folds == 0
):
    majority_cover_type = int(train[target].mode().iloc[0])
    test_pred_cover_type = np.full((len(test),), majority_cover_type, dtype=np.int32)
else:
    test_pred_encoded = np.argmax(nn_test_preds_proba, axis=1)
    test_pred_cover_type = label_enc.inverse_transform(test_pred_encoded).astype(
        np.int32
    )

predictions = pd.DataFrame()
predictions["Id"] = test["Id"].values
predictions["Cover_Type"] = test_pred_cover_type.astype(int)

predictions.to_csv("submission.csv", index=False)
print(predictions.head())
print("Wrote submission.csv with shape:", predictions.shape)



## === cell 35
if not PASS_PLOTTING:
    predictions["Cover_Type"].hist()
