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

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import time
import gc
import torchmetrics

pd.set_option("display.max_rows", 150)
pd.set_option("display.max_columns", 500)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.float_format", lambda x: "%.5f" % x)

if os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "") != "Batch":
    for dirname, _, filenames in os.walk("/kaggle/input"):
        for filename in filenames:
            print(os.path.join(dirname, filename))

pl.seed_everything(42, workers=True)
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_DATALOADER_WORKERS = 2 if torch.cuda.is_available() else 0
_PERSISTENT_WORKERS = bool(_DATALOADER_WORKERS > 0)



## === cell 1
try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

train_path = "/kaggle/input/tabular-playground-series-dec-2021/train.csv"
test_path = "/kaggle/input/tabular-playground-series-dec-2021/test.csv"

_cols = pd.read_csv(train_path, nrows=0).columns.tolist()
dtype_map = {}
for c in _cols:
    if c == "Id":
        dtype_map[c] = np.int32
    elif c == "Cover_Type":
        dtype_map[c] = np.int8
    else:
        dtype_map[c] = np.int16

train = pd.read_csv(train_path, low_memory=False, dtype=dtype_map)
dtype_map_test = {k: v for k, v in dtype_map.items() if k != "Cover_Type"}
test = pd.read_csv(test_path, low_memory=False, dtype=dtype_map_test)

del _cols, dtype_map, dtype_map_test
gc.collect()




## === cell 2
def reduce_mem_usage(df, verbose=True):
    if verbose:
        mem = df.memory_usage().sum() / 1024**2
        print(f"Mem. usage (already typed on read): {mem:5.2f} Mb (0.0% reduction)")
    return df


train = reduce_mem_usage(train)
test = reduce_mem_usage(test)



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
features = [c for c in train.columns if c not in ("Id", target)]



## === cell 8
pass



## === cell 9
pass



## === cell 10
pass



## === cell 11
pass



## === cell 12
df_cols = train[features]
nunique = df_cols.nunique()
cat_features = nunique[nunique < 10].index
num_features = nunique[nunique >= 10].index

print(f"There are {len(cat_features)} categorical features: {cat_features}")
print(f"\nThere are {len(num_features)} continuous features: {num_features}")



## === cell 13
_ = (train.isna().sum().sum(), test.isna().sum().sum())
print("Total NA (train, test):", _)



## === cell 14
pass



## === cell 15
pass



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

df_cols = train[features]
nunique = df_cols.nunique()
cat_features = nunique[nunique < 10].index
num_features = nunique[nunique >= 10].index



## === cell 18
print("Numerical features with the least amount of unique values:")
train[num_features].nunique().sort_values().head(5)



## === cell 19
train.drop(train[train[target] == 5].index, axis=0, inplace=True)
train.reset_index(drop=True, inplace=True)
label_enc = LabelEncoder()
NUM_CLASSES = train[target].nunique()



## === cell 20
s_scaler = StandardScaler(copy=False)

train_num_arr = train[num_features].to_numpy(dtype=np.float32, copy=True)
test_num_arr = test[num_features].to_numpy(dtype=np.float32, copy=True)

s_scaler.fit(train_num_arr)
s_scaler.transform(train_num_arr)
s_scaler.transform(test_num_arr)

del s_scaler
gc.collect()



## === cell 21
mm_scaler = MinMaxScaler(copy=False)

cat_cols = list(cat_features)
train_cat_arr = train[cat_cols].to_numpy(dtype=np.float32, copy=False)
test_cat_arr = test[cat_cols].to_numpy(dtype=np.float32, copy=False)

feat_pos = {c: i for i, c in enumerate(features)}
num_pos = [feat_pos[c] for c in num_features]
cat_pos = [feat_pos[c] for c in cat_cols]

X_nn_np = np.empty((train.shape[0], len(features)), dtype=np.float32)
X_test_np = np.empty((test.shape[0], len(features)), dtype=np.float32)

X_nn_np[:, num_pos] = train_num_arr
X_test_np[:, num_pos] = test_num_arr
X_nn_np[:, cat_pos] = train_cat_arr
X_test_np[:, cat_pos] = test_cat_arr

y = pd.Series(label_enc.fit_transform(train[target]))
y_np = y.to_numpy(dtype=np.int64, copy=False)

mm_scaler.fit(X_nn_np)
mm_scaler.transform(X_nn_np)
mm_scaler.transform(X_test_np)

del (
    mm_scaler,
    train_num_arr,
    test_num_arr,
    train_cat_arr,
    test_cat_arr,
    feat_pos,
    num_pos,
    cat_pos,
    y,
)
gc.collect()



## === cell 22
BATCH_SIZE = 4096




## === cell 23
class NumpyIndexedDataset(Dataset):
    def __init__(self, X_np: np.ndarray, y_np: np.ndarray, indices: np.ndarray):
        self.X = X_np
        self.y = y_np
        self.indices = indices

    def __len__(self):
        return self.indices.shape[0]

    def __getitem__(self, i):
        idx = int(self.indices[i])
        return torch.from_numpy(self.X[idx]), torch.tensor(
            self.y[idx], dtype=torch.long
        )


def prepare_datasets_from_indices(
    X_np, y_np, train_idx, valid_idx, batch_size=BATCH_SIZE
):
    train_ds = NumpyIndexedDataset(X_np, y_np, train_idx)
    valid_ds = NumpyIndexedDataset(X_np, y_np, valid_idx)

    pin = torch.cuda.is_available()
    common = dict(
        batch_size=batch_size,
        drop_last=False,
        num_workers=_DATALOADER_WORKERS,
        persistent_workers=_PERSISTENT_WORKERS,
        pin_memory=pin,
        prefetch_factor=4 if _DATALOADER_WORKERS > 0 else None,
    )

    train_dl = DataLoader(
        train_ds, shuffle=False, **{k: v for k, v in common.items() if v is not None}
    )
    valid_dl = DataLoader(
        valid_ds, shuffle=False, **{k: v for k, v in common.items() if v is not None}
    )
    return train_dl, valid_dl




## === cell 24
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data)




## === cell 25
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
        if "val_loss" in metrics_logs and "val_acc" in metrics_logs:
            self.val_loss.append(metrics_logs["val_loss"].item())
            self.val_acc.append(metrics_logs["val_acc"].item())

    def on_train_epoch_end(self, trainer, module):
        metrics_logs = trainer.logged_metrics
        if "loss" in metrics_logs:
            self.train_loss.append(metrics_logs["loss"].item())
        if "train_acc" in metrics_logs:
            self.train_acc.append(metrics_logs["train_acc"].item())

        if (
            False
            and self.verbose is True
            and len(self.val_loss) > 0
            and len(self.train_loss) > 0
        ):
            print(
                f"Epoch {module.current_epoch} start learning rate: {self.lr_epoch_start[-1]:.6f}, "
                f"train_loss: {self.train_loss[-1]:.4f}, "
                f"train_acc: {self.train_acc[-1]:.4f}, "
                f"val_loss: {self.val_loss[-1]:.4f}, "
                f"val_acc: {self.val_acc[-1]:.4f}"
            )




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
        return loss

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
        return val_loss

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
class BestStateCallback(pl.callbacks.Callback):
    def __init__(self, monitor="val_acc", mode="max"):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.best = -float("inf") if mode == "max" else float("inf")
        self.best_state_dict = None

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.logged_metrics
        if self.monitor not in metrics:
            return
        val = metrics[self.monitor].item()
        improved = val > self.best if self.mode == "max" else val < self.best
        if improved:
            self.best = val
            self.best_state_dict = {
                k: v.detach().cpu().clone() for k, v in pl_module.state_dict().items()
            }


def train_ann(train_ds, valid_ds, Model=Model, input_shape=None):
    if input_shape is None:
        xb0, _ = train_ds.dataset[0] if hasattr(train_ds, "dataset") else train_ds[0]
        input_shape = xb0.shape[0]

    model = Model(input_shape)
    model.apply(initialize_weights)

    early_stop_callback = EarlyStopping(
        monitor="val_acc", min_delta=0.00, patience=20, verbose=False, mode="max"
    )
    params_tracker_callback = ParamsTracker(verbose=True)
    best_state_callback = BestStateCallback(monitor="val_acc", mode="max")

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
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator=("gpu" if torch.cuda.is_available() else "cpu"),
        devices=1,
    )

    trainer.fit(model, train_ds, valid_ds)

    if best_state_callback.best_state_dict is None:
        best_state_callback.best_state_dict = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }

    model.eval()
    return best_state_callback.best_state_dict, params_tracker_callback




## === cell 28
@torch.inference_mode()
def predict_logits_batched(model, X_tensor, batch_size=BATCH_SIZE, device=DEVICE):
    model = model.to(device)
    model.eval()
    if not X_tensor.is_contiguous():
        X_tensor = X_tensor.contiguous()
    dl = DataLoader(
        TensorDataset(X_tensor),
        batch_size=batch_size,
        shuffle=False,
        num_workers=_DATALOADER_WORKERS,
        persistent_workers=_PERSISTENT_WORKERS,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=4 if _DATALOADER_WORKERS > 0 else None,
    )
    outs = []
    for (xb,) in dl:
        xb = xb.to(device, non_blocking=True)
        outs.append(model(xb).detach().cpu())
    return torch.cat(outs, dim=0).numpy()




## === cell 29
start_time = time.time()

splits = 10
skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)

X_test_tensor = torch.from_numpy(np.ascontiguousarray(X_test_np)).to(
    dtype=torch.float32
)

nn_oof_preds = np.zeros((X_nn_np.shape[0],), dtype=np.int64)
nn_test_logits_sum = np.zeros((X_test_tensor.shape[0], 6), dtype=np.float32)
total_mean_acc = 0.0
used_folds = 0

for num, (train_idx, valid_idx) in enumerate(skf.split(X_nn_np, y_np)):
    if num > 0:
        break
    used_folds += 1
    print(f"\n\n===Training with fold {num}")

    train_dl, valid_dl = prepare_datasets_from_indices(
        X_nn_np, y_np, train_idx, valid_idx
    )

    best_state_dict, tracked_values = train_ann(
        train_dl, valid_dl, Model, X_nn_np.shape[1]
    )

    model = Model(X_nn_np.shape[1])
    model.load_state_dict(best_state_dict)

    X_valid_np = X_nn_np[valid_idx]
    y_valid_np = y_np[valid_idx]

    X_valid_t = torch.from_numpy(np.ascontiguousarray(X_valid_np)).to(
        dtype=torch.float32
    )
    valid_logits = predict_logits_batched(model, X_valid_t, batch_size=BATCH_SIZE)
    preds = np.argmax(valid_logits, axis=1)

    fold_score = accuracy_score(y_valid_np, preds)
    print(f"\n===Fold {num} valid data accuracy score is {fold_score}")

    test_logits = predict_logits_batched(model, X_test_tensor, batch_size=BATCH_SIZE)

    nn_oof_preds[valid_idx] = preds
    nn_test_logits_sum += test_logits
    total_mean_acc += fold_score / splits

print(f"Average accuracy score of all models is {total_mean_acc}")
print(f"Elapsed seconds: {time.time() - start_time:.2f}")



## === cell 30
if "model" in globals():
    try:
        _xv = X_valid_t
    except NameError:
        _xv = torch.from_numpy(np.ascontiguousarray(X_valid_np)).to(dtype=torch.float32)
    print(np.argmax(model(_xv).detach().numpy(), axis=1)[:10])



## === cell 31
if "model" in globals():
    _train_idx0, _valid_idx0 = next(
        StratifiedKFold(n_splits=10, shuffle=True, random_state=42).split(X_nn_np, y_np)
    )
    X_train_np = X_nn_np[_train_idx0]
    y_train_np = y_np[_train_idx0]
    _xt = torch.from_numpy(np.ascontiguousarray(X_train_np)).to(dtype=torch.float32)
    train_logits = predict_logits_batched(model, _xt, batch_size=BATCH_SIZE)
    print(accuracy_score(y_train_np, np.argmax(train_logits, axis=1)))



## === cell 32
if "model" in globals():
    try:
        _xv = X_valid_t
    except NameError:
        _xv = torch.from_numpy(np.ascontiguousarray(X_valid_np)).to(dtype=torch.float32)
    valid_logits = predict_logits_batched(model, _xv, batch_size=BATCH_SIZE)
    print(accuracy_score(y_valid_np, np.argmax(valid_logits, axis=1)))



## === cell 33
denom = float(max(used_folds, 1))
avg_test_logits = nn_test_logits_sum / denom
test_pred_labels = np.argmax(avg_test_logits, axis=1)
test_pred_cover_type = label_enc.inverse_transform(test_pred_labels)

predictions = pd.DataFrame()
predictions["Id"] = test["Id"].astype(np.int64)
predictions["Cover_Type"] = test_pred_cover_type.astype(np.int64)

predictions.to_csv("submission.csv", index=False)
print(predictions.head())
print("Saved submission.csv with shape:", predictions.shape)
print("submission.csv columns:", predictions.columns.tolist())
