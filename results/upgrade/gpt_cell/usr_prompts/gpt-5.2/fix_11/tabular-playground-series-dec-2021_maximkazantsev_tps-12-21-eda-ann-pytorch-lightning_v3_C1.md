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
import os
import gc
import time
import glob
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import torchmetrics

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


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
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.set_num_threads(min(8, os.cpu_count() or 1))



## === cell 1
TRAIN_PATH = "/kaggle/input/tabular-playground-series-dec-2021/train.csv"
TEST_PATH = "/kaggle/input/tabular-playground-series-dec-2021/test.csv"

_cols = pd.read_csv(TRAIN_PATH, nrows=0).columns.tolist()
target = "Cover_Type"
id_col = "Id"
feature_cols_all = [c for c in _cols if c not in [id_col, target]]

dtype_map_train = {id_col: np.int32, target: np.int8}
dtype_map_test = {id_col: np.int32}
for c in feature_cols_all:
    if c.startswith("Wilderness_Area") or c.startswith("Soil_Type"):
        dtype_map_train[c] = np.int8
        dtype_map_test[c] = np.int8
    else:
        dtype_map_train[c] = np.float32
        dtype_map_test[c] = np.float32

t0 = time.time()
train = pd.read_csv(TRAIN_PATH, dtype=dtype_map_train)
test = pd.read_csv(TEST_PATH, dtype=dtype_map_test)
print(
    f"Loaded train/test in {time.time()-t0:.1f}s: train={train.shape}, test={test.shape}"
)
gc.collect()



## === cell 2
pass



## === cell 3
pass



## === cell 4
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



## === cell 5
pass



## === cell 6
features = list(train.columns[1:54])



## === cell 7
pass



## === cell 8
pass



## === cell 9
pass



## === cell 10
cat_features = [
    c for c in features if c.startswith("Wilderness_Area") or c.startswith("Soil_Type")
]
num_features = [c for c in features if c not in cat_features]

print(
    f"There are {len(cat_features)} categorical features: {list(cat_features)[:5]}{'...' if len(cat_features)>5 else ''}"
)
print(
    f"\nThere are {len(num_features)} continuous features: {list(num_features)[:5]}{'...' if len(num_features)>5 else ''}"
)



## === cell 11
print(int(train.isna().sum().sum()), int(test.isna().sum().sum()))



## === cell 12
pass



## === cell 13
pass



## === cell 14
print(
    f"Rows with soil type 7: {(train['Soil_Type7'] == 1).sum() + (test['Soil_Type7'] == 1).sum()}"
)
print(
    f"Rows with soil type 15: {(train['Soil_Type15'] == 1).sum() + (test['Soil_Type15'] == 1).sum()}"
)



## === cell 15
train.drop(["Soil_Type7", "Soil_Type15"], axis=1, inplace=True)
test.drop(["Soil_Type7", "Soil_Type15"], axis=1, inplace=True)
features.remove("Soil_Type7")
features.remove("Soil_Type15")
if "Soil_Type7" in cat_features:
    cat_features = [c for c in cat_features if c not in ("Soil_Type7", "Soil_Type15")]
gc.collect()



## === cell 16
pass



## === cell 17
train.drop(train[train[target] == 5].index, axis=0, inplace=True)
train.reset_index(drop=True, inplace=True)
label_enc = LabelEncoder()
NUM_CLASSES = train[target].nunique()
gc.collect()



## === cell 18
s_scaler = StandardScaler()
train_num = train.loc[:, num_features].to_numpy(dtype=np.float32, copy=False)
test_num = test.loc[:, num_features].to_numpy(dtype=np.float32, copy=False)

train_scaled = s_scaler.fit_transform(train_num)
test_scaled = s_scaler.transform(test_num)

train.loc[:, num_features] = train_scaled
test.loc[:, num_features] = test_scaled
del train_num, test_num, train_scaled, test_scaled
gc.collect()



## === cell 19
X_nn = train[features]
X_test_nn_df = test[features]
y = pd.Series(label_enc.fit_transform(train[target]))
gc.collect()



## === cell 20
mm_scaler = MinMaxScaler()

X_nn_arr = X_nn.to_numpy(dtype=np.float32, copy=True)
X_test_arr = X_test_nn_df.to_numpy(dtype=np.float32, copy=True)

X_nn_arr = mm_scaler.fit_transform(X_nn_arr)
X_test_arr = mm_scaler.transform(X_test_arr)

del X_nn, X_test_nn_df
gc.collect()



## === cell 21
BATCH_SIZE = 4096




## === cell 22
def prepare_datasets(X_nn, X_valid_nn, y_nn, y_valid_nn, batch_size=BATCH_SIZE):
    X_nn_arr = (
        X_nn.to_numpy(dtype=np.float32, copy=False)
        if isinstance(X_nn, (pd.DataFrame, pd.Series))
        else np.asarray(X_nn, dtype=np.float32)
    )
    y_nn_arr = (
        y_nn.to_numpy(dtype=np.int64, copy=False)
        if isinstance(y_nn, (pd.DataFrame, pd.Series))
        else np.asarray(y_nn, dtype=np.int64)
    )
    X_valid_arr = (
        X_valid_nn.to_numpy(dtype=np.float32, copy=False)
        if isinstance(X_valid_nn, (pd.DataFrame, pd.Series))
        else np.asarray(X_valid_nn, dtype=np.float32)
    )
    y_valid_arr = (
        y_valid_nn.to_numpy(dtype=np.int64, copy=False)
        if isinstance(y_valid_nn, (pd.DataFrame, pd.Series))
        else np.asarray(y_valid_nn, dtype=np.int64)
    )

    X_nn_t = torch.from_numpy(np.ascontiguousarray(X_nn_arr))
    y_nn_t = torch.from_numpy(np.ascontiguousarray(y_nn_arr))
    X_valid_t = torch.from_numpy(np.ascontiguousarray(X_valid_arr))
    y_valid_t = torch.from_numpy(np.ascontiguousarray(y_valid_arr))

    train_ds = TensorDataset(X_nn_t, y_nn_t)
    valid_ds = TensorDataset(X_valid_t, y_valid_t)

    num_workers = min(4, os.cpu_count() or 1)
    common = dict(
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
        shuffle=False,  # keep identical to original semantics
    )
    common = {k: v for k, v in common.items() if v is not None}

    train_dl = DataLoader(train_ds, **common)
    valid_dl = DataLoader(valid_ds, **common)
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

    def on_validation_epoch_end(self, trainer, module):
        metrics_logs = trainer.callback_metrics
        if "val_loss" in metrics_logs and metrics_logs["val_loss"] is not None:
            self.val_loss.append(metrics_logs["val_loss"].detach().cpu().item())
        if "val_acc" in metrics_logs and metrics_logs["val_acc"] is not None:
            self.val_acc.append(metrics_logs["val_acc"].detach().cpu().item())

    def on_train_epoch_end(self, trainer, module):
        metrics_logs = trainer.callback_metrics
        if "loss_epoch" in metrics_logs and metrics_logs["loss_epoch"] is not None:
            self.train_loss.append(metrics_logs["loss_epoch"].detach().cpu().item())
        elif "loss" in metrics_logs and metrics_logs["loss"] is not None:
            self.train_loss.append(metrics_logs["loss"].detach().cpu().item())
        if "train_acc" in metrics_logs and metrics_logs["train_acc"] is not None:
            self.train_acc.append(metrics_logs["train_acc"].detach().cpu().item())

        if self.verbose is True:
            lr = self.lr_epoch_start[-1] if self.lr_epoch_start else float("nan")
            tl = self.train_loss[-1] if self.train_loss else float("nan")
            ta = self.train_acc[-1] if self.train_acc else float("nan")
            vl = self.val_loss[-1] if self.val_loss else float("nan")
            va = self.val_acc[-1] if self.val_acc else float("nan")
            print(
                f"Epoch {module.current_epoch} start learning rate: {lr:.6f}, "
                f"train_loss: {tl:.4f}, "
                f"train_acc: {ta:.4f}, "
                f"val_loss: {vl:.4f}, "
                f"val_acc: {va:.4f}"
            )




## === cell 25
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

        self._train_correct = 0
        self._train_total = 0
        self._val_correct = 0
        self._val_total = 0

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
        X = X.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)

        y_hat = self(X).squeeze(1)
        loss = self.loss(y_hat, y)

        preds = torch.argmax(y_hat, dim=1)
        self._train_correct += int((preds == y).sum().detach().cpu().item())
        self._train_total += int(y.numel())

        self.log(
            "loss", loss, prog_bar=False, on_epoch=True, on_step=False, logger=False
        )
        return {"loss": loss}

    def on_train_epoch_end(self):
        train_acc = (
            float(self._train_correct) / float(self._train_total)
            if self._train_total > 0
            else 0.0
        )
        self.log(
            "train_acc",
            torch.tensor(train_acc, device=self.device),
            prog_bar=False,
            on_epoch=True,
            on_step=False,
            logger=False,
        )
        if "loss" in self.trainer.callback_metrics:
            self.log(
                "loss_epoch",
                self.trainer.callback_metrics["loss"],
                prog_bar=False,
                on_epoch=True,
                on_step=False,
                logger=False,
            )
        self._train_correct = 0
        self._train_total = 0

    def validation_step(self, batch, batch_idx):
        X, y = batch
        X = X.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)

        y_hat = self(X).squeeze(1)
        val_loss = self.loss(y_hat, y)

        preds = torch.argmax(y_hat, dim=1)
        self._val_correct += int((preds == y).sum().detach().cpu().item())
        self._val_total += int(y.numel())

        self.log(
            "val_loss",
            val_loss,
            prog_bar=False,
            on_epoch=True,
            on_step=False,
            logger=False,
        )
        return {"val_loss": val_loss}

    def on_validation_epoch_end(self):
        val_acc = (
            float(self._val_correct) / float(self._val_total)
            if self._val_total > 0
            else 0.0
        )
        self.log(
            "val_acc",
            torch.tensor(val_acc, device=self.device),
            prog_bar=False,
            on_epoch=True,
            on_step=False,
            logger=False,
        )
        self._val_correct = 0
        self._val_total = 0

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




## === cell 26
def train_ann(train_ds, valid_ds, Model=Model, input_shape=None):
    if input_shape is None:
        input_shape = train_ds.dataset.tensors[0].shape[1]

    model = Model(input_shape)
    model.apply(initialize_weights)

    class BestStateCallback(pl.callbacks.Callback):
        def __init__(self):
            self.best = -float("inf")
            self.best_state = None

        def on_validation_epoch_end(self, trainer, pl_module):
            metrics = trainer.callback_metrics
            if "val_acc" in metrics and metrics["val_acc"] is not None:
                v = float(metrics["val_acc"].detach().cpu().item())
                if v > self.best:
                    self.best = v
                    self.best_state = {
                        k: vv.cpu().clone() for k, vv in pl_module.state_dict().items()
                    }

    early_stop_callback = EarlyStopping(
        monitor="val_acc",
        min_delta=0.00,
        patience=20,
        verbose=False,
        mode="max",
    )

    params_tracker_callback = ParamsTracker(verbose=False)
    best_state_cb = BestStateCallback()

    trainer = pl.Trainer(
        fast_dev_run=False,
        max_epochs=60,
        precision=32,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        val_check_interval=1.0,
        callbacks=[early_stop_callback, params_tracker_callback, best_state_cb],
        logger=False,
        accelerator="gpu" if DEVICE == "cuda" else "cpu",
        devices=1,
        enable_checkpointing=False,
        deterministic=True,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    trainer.fit(model, train_ds, valid_ds)
    model.eval()

    os.makedirs("models", exist_ok=True)
    best_model_path = os.path.join("models", "best_model_state_dict.pt")
    state_to_save = (
        best_state_cb.best_state
        if best_state_cb.best_state is not None
        else model.state_dict()
    )
    torch.save({"state_dict": state_to_save}, best_model_path)

    return best_model_path, params_tracker_callback




## === cell 27
pass



## === cell 28
if "X_valid" not in globals():
    X_train_arr, X_valid_arr, y_train_arr, y_valid_arr = train_test_split(
        X_nn_arr,
        y.to_numpy(dtype=np.int64, copy=False),
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

input_shape = X_nn_arr.shape[1]
os.makedirs("models", exist_ok=True)

ckpt_path = None
if (
    "best_model_path" in globals()
    and isinstance(best_model_path, str)
    and best_model_path
    and os.path.exists(best_model_path)
):
    ckpt_path = best_model_path
else:
    candidates = glob.glob(os.path.join("models", "*.pt")) + glob.glob(
        os.path.join("models", "*.ckpt")
    )
    if candidates:
        ckpt_path = max(candidates, key=os.path.getmtime)

if ckpt_path is None or not os.path.exists(ckpt_path):
    train_ds, valid_ds = prepare_datasets(
        X_train_arr, X_valid_arr, y_train_arr, y_valid_arr
    )
    best_model_path, _ = train_ann(
        train_ds, valid_ds, Model=Model, input_shape=input_shape
    )
    ckpt_path = best_model_path

model = Model(input_shape)
state = torch.load(ckpt_path, map_location="cpu")
if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]
model.load_state_dict(state, strict=True)
model.eval()
model.to(DEVICE)

X_train_t = torch.from_numpy(np.ascontiguousarray(X_train_arr)).to(
    DEVICE, non_blocking=True
)
X_valid_t = torch.from_numpy(np.ascontiguousarray(X_valid_arr)).to(
    DEVICE, non_blocking=True
)
X_test_t = torch.from_numpy(np.ascontiguousarray(X_test_arr)).to(
    DEVICE, non_blocking=True
)


@torch.inference_mode()
def predict_argmax_tensor(model, X_t, batch_size=262144):
    preds = torch.empty((X_t.shape[0],), dtype=torch.int64, device="cpu")
    offset = 0
    for i in range(0, X_t.shape[0], batch_size):
        xb = X_t[i : i + batch_size]
        logits = model(xb)
        out = torch.argmax(logits, dim=1).detach().cpu()
        preds[offset : offset + out.shape[0]] = out
        offset += out.shape[0]
    return preds.numpy()


valid_preview = predict_argmax_tensor(model, X_valid_t[:10], batch_size=10)
valid_preview[:10]



## === cell 29
yhat_train = predict_argmax_tensor(model, X_train_t, batch_size=262144)
accuracy_score(y_train_arr, yhat_train)



## === cell 30
yhat_valid = predict_argmax_tensor(model, X_valid_t, batch_size=262144)
accuracy_score(y_valid_arr, yhat_valid)



## === cell 31
test_preds = predict_argmax_tensor(model, X_test_t, batch_size=262144)

predictions = pd.DataFrame()
predictions["Id"] = test["Id"].astype(np.int32, copy=False)
predictions["Cover_Type"] = label_enc.inverse_transform(test_preds)

predictions.to_csv("submission.csv", index=False, header=predictions.columns)
predictions.head()



## === cell 32
pass
