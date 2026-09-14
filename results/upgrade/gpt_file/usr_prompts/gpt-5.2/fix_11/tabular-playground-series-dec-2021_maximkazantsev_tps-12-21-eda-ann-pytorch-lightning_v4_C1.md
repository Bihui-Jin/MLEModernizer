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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
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

pl.seed_everything(42, workers=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

pd.set_option("display.max_rows", 150)
pd.set_option("display.max_columns", 500)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.float_format", lambda x: "%.5f" % x)

train_path = "/kaggle/input/tabular-playground-series-dec-2021/train.csv"
test_path = "/kaggle/input/tabular-playground-series-dec-2021/test.csv"


def _build_dtype_map_for_tps_dec2021(path, has_target: bool):
    header = pd.read_csv(path, nrows=0)
    cols = header.columns.tolist()
    dtype = {}
    for c in cols:
        if c == "Id":
            dtype[c] = np.int32
        elif c == "Cover_Type" and has_target:
            dtype[c] = np.int8
        else:
            if c in (
                "Elevation",
                "Horizontal_Distance_To_Hydrology",
                "Vertical_Distance_To_Hydrology",
                "Horizontal_Distance_To_Roadways",
                "Horizontal_Distance_To_Fire_Points",
            ):
                dtype[c] = np.int32
            else:
                dtype[c] = np.int16
    return dtype


def _read_csv_fast(path, dtype_map):
    try:
        return pd.read_csv(path, engine="pyarrow", dtype=dtype_map, memory_map=True)
    except Exception:
        return pd.read_csv(path, dtype=dtype_map, low_memory=False)


train = _read_csv_fast(
    train_path, _build_dtype_map_for_tps_dec2021(train_path, has_target=True)
)
test = _read_csv_fast(
    test_path, _build_dtype_map_for_tps_dec2021(test_path, has_target=False)
)

print(train.shape, test.shape)
print(train.columns[:5].tolist(), "...", train.columns[-5:].tolist())




## === cell 1
def reduce_mem_usage(df, verbose=True):
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
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


print("Skipped reduce_mem_usage (already read with compact dtypes).")



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
target = "Cover_Type"
features = [c for c in test.columns if c != "Id"]
print("Initial feature count:", len(features))



## === cell 7
pass



## === cell 8
pass



## === cell 9
pass



## === cell 10
pass



## === cell 11
cat_features = [
    c for c in features if c.startswith("Wilderness_Area") or c.startswith("Soil_Type")
]
num_features = [c for c in features if c not in cat_features]

print(
    f"There are {len(cat_features)} categorical features: {cat_features[:5]}...{cat_features[-5:]}"
)
print(f"\nThere are {len(num_features)} continuous features: {num_features}")



## === cell 12
_ = (train.isna().sum().sum(), test.isna().sum().sum())
print("Total NA (train, test):", _)



## === cell 13
pass



## === cell 14
pass



## === cell 15
print(
    f"Rows with soil type 7: {(train['Soil_Type7'] == 1).sum() + (test['Soil_Type7'] == 1).sum()}"
)
print(
    f"Rows with soil type 15: {(train['Soil_Type15'] == 1).sum() + (test['Soil_Type15'] == 1).sum()}"
)



## === cell 16
train.drop(["Soil_Type7", "Soil_Type15"], axis=1, inplace=True)
test.drop(["Soil_Type7", "Soil_Type15"], axis=1, inplace=True)
if "Soil_Type7" in features:
    features.remove("Soil_Type7")
if "Soil_Type15" in features:
    features.remove("Soil_Type15")
cat_features = [c for c in cat_features if c not in ("Soil_Type7", "Soil_Type15")]

print("Feature count after Soil_Type drops:", len(features))



## === cell 17
print("Numerical features with the least amount of unique values:")
print("(skipped expensive nunique sort on full train for speed)")



## === cell 18
train.drop(train[train[target] == 5].index, axis=0, inplace=True)
train.reset_index(drop=True, inplace=True)
label_enc = LabelEncoder()
NUM_CLASSES = train[target].nunique()
print("NUM_CLASSES after dropping Cover_Type==5:", NUM_CLASSES)



## === cell 19
s_scaler = StandardScaler()
train_num_arr = s_scaler.fit_transform(
    train[num_features].to_numpy(dtype=np.float32, copy=False)
)
test_num_arr = s_scaler.transform(
    test[num_features].to_numpy(dtype=np.float32, copy=False)
)



## === cell 20
train_cat_arr = train[cat_features].to_numpy(dtype=np.float32, copy=False)
test_cat_arr = test[cat_features].to_numpy(dtype=np.float32, copy=False)

num_pos = {c: i for i, c in enumerate(num_features)}
cat_pos = {c: i for i, c in enumerate(cat_features)}
num_indices = [num_pos[c] for c in features if c in num_pos]
cat_indices = [cat_pos[c] for c in features if c in cat_pos]

features_is_num_then_cat = all(
    c in num_pos for c in features[: len(num_features)]
) and all(c in cat_pos for c in features[len(num_features) :])

if features_is_num_then_cat:
    X_nn_np = np.concatenate(
        [train_num_arr[:, num_indices], train_cat_arr[:, cat_indices]], axis=1
    ).astype(np.float32, copy=False)
    X_test_np = np.concatenate(
        [test_num_arr[:, num_indices], test_cat_arr[:, cat_indices]], axis=1
    ).astype(np.float32, copy=False)
else:
    col_blocks_train = []
    col_blocks_test = []
    for c in features:
        if c in num_pos:
            col_blocks_train.append(train_num_arr[:, num_pos[c]][:, None])
            col_blocks_test.append(test_num_arr[:, num_pos[c]][:, None])
        else:
            col_blocks_train.append(train_cat_arr[:, cat_pos[c]][:, None])
            col_blocks_test.append(test_cat_arr[:, cat_pos[c]][:, None])
    X_nn_np = np.concatenate(col_blocks_train, axis=1).astype(np.float32, copy=False)
    X_test_np = np.concatenate(col_blocks_test, axis=1).astype(np.float32, copy=False)

y = pd.Series(label_enc.fit_transform(train[target].to_numpy(copy=False)))
y_np = y.to_numpy(dtype=np.int64, copy=False)

mm_scaler = MinMaxScaler()
X_nn_np = mm_scaler.fit_transform(X_nn_np).astype(np.float32, copy=False)
X_test_np = mm_scaler.transform(X_test_np).astype(np.float32, copy=False)

X_nn_np = np.ascontiguousarray(X_nn_np, dtype=np.float32)
X_test_np = np.ascontiguousarray(X_test_np, dtype=np.float32)

X_test_nn = torch.from_numpy(X_test_np).float()

del train_num_arr, test_num_arr, train_cat_arr, test_cat_arr, X_test_np, y
gc.collect()



## === cell 21
BATCH_SIZE = 4096




## === cell 22
def prepare_datasets(
    X_nn_np_train, X_nn_np_valid, y_nn_train, y_nn_valid, batch_size=BATCH_SIZE
):
    X_nn_t = torch.from_numpy(np.ascontiguousarray(X_nn_np_train)).float()
    y_nn_t = torch.from_numpy(y_nn_train).long()
    X_valid_t = torch.from_numpy(np.ascontiguousarray(X_nn_np_valid)).float()
    y_valid_t = torch.from_numpy(y_nn_valid).long()

    train_ds = TensorDataset(X_nn_t, y_nn_t)
    valid_ds = TensorDataset(X_valid_t, y_valid_t)

    pin = torch.cuda.is_available()
    num_workers = min(4, (os.cpu_count() or 0)) if (os.cpu_count() or 0) >= 4 else 0
    persistent_workers = num_workers > 0

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=persistent_workers,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    valid_dl = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=persistent_workers,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return train_dl, valid_dl




## === cell 23
class_weights = compute_class_weight(
    class_weight="balanced", classes=np.unique(y_np), y=y_np
)
class_weights




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
        if "val_loss" in metrics_logs:
            self.val_loss.append(metrics_logs["val_loss"].item())
        if "val_acc" in metrics_logs:
            self.val_acc.append(metrics_logs["val_acc"].item())

    def on_train_epoch_end(self, trainer, module):
        metrics_logs = trainer.logged_metrics
        if "loss" in metrics_logs:
            self.train_loss.append(metrics_logs["loss"].item())
        if "train_acc" in metrics_logs:
            self.train_acc.append(metrics_logs["train_acc"].item())

        if (
            self.verbose == True
            and len(self.val_loss) > 0
            and len(self.val_acc) > 0
            and len(self.train_loss) > 0
            and len(self.train_acc) > 0
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
    def __init__(self, input_shape, num_classes=NUM_CLASSES):
        super().__init__()

        self.input = nn.Linear(input_shape, 128)
        self.hidden1 = nn.Linear(128, 64)
        self.hidden2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, num_classes)

        self.dr = 0.2
        self.activation = F.relu

        self.train_acc_metric = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, average="micro"
        )
        self.val_acc_metric = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes, average="micro"
        )

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.activation(self.input(x))
        x = F.dropout(x, p=self.dr, training=self.training)
        x = self.activation(self.hidden1(x))
        x = F.dropout(x, p=self.dr, training=self.training)
        x = self.activation(self.hidden2(x))
        x = F.dropout(x, p=self.dr, training=self.training)
        x = self.output(x)
        return x

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X).squeeze(1)
        loss = self.loss(y_hat, y)
        self.train_acc_metric(torch.argmax(y_hat, dim=1), y)
        self.log(
            "loss", loss, prog_bar=False, on_epoch=True, on_step=False, logger=False
        )
        return {"loss": loss}

    def on_train_epoch_end(self):
        train_acc = self.train_acc_metric.compute()
        self.log(
            "train_acc",
            train_acc,
            prog_bar=False,
            on_epoch=True,
            on_step=False,
            logger=False,
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
            prog_bar=False,
            on_epoch=True,
            on_step=False,
            logger=False,
        )
        return {"val_loss": val_loss}

    def on_validation_epoch_end(self):
        val_acc = self.val_acc_metric.compute()
        self.log(
            "val_acc",
            val_acc,
            prog_bar=False,
            on_epoch=True,
            on_step=False,
            logger=False,
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
class BestStateDictSaver(pl.callbacks.Callback):
    def __init__(self, monitor="val_acc", mode="max"):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.best_score = None
        self.best_state_dict = None

    def _is_better(self, current, best):
        if best is None:
            return True
        return current > best if self.mode == "max" else current < best

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics_logs = trainer.logged_metrics
        if self.monitor not in metrics_logs:
            return
        current = metrics_logs[self.monitor]
        if hasattr(current, "item"):
            current = float(current.item())
        else:
            current = float(current)
        if self._is_better(current, self.best_score):
            self.best_score = current
            self.best_state_dict = {
                k: v.detach().cpu().clone() for k, v in pl_module.state_dict().items()
            }


def train_ann(train_dl, valid_dl, Model=Model, input_shape=None, max_epochs=60):
    if input_shape is None:
        input_shape = train_dl.dataset.tensors[0].shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(input_shape)
    model.apply(initialize_weights)
    model.to(device)

    opt_dict = model.configure_optimizers()
    optimizer = opt_dict["optimizer"]
    scheduler = opt_dict["lr_scheduler"]

    es_mode = "max"
    es_min_delta = 0.0002
    es_patience = 20
    best_es_score = None
    num_bad_epochs = 0

    tracked = ParamsTracker(verbose=True)

    best_score = None
    best_state_dict = None

    for epoch in range(max_epochs):
        model.train()

        tracked.lr_epoch_start.append(optimizer.param_groups[0]["lr"])

        running_loss = 0.0
        seen = 0

        for xb, yb in train_dl:
            xb = xb.to(device=device, non_blocking=True)
            yb = yb.to(device=device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb).squeeze(1)
            loss = model.loss(logits, yb)
            loss.backward()
            optimizer.step()

            bs = yb.shape[0]
            running_loss += float(loss.detach()) * bs
            seen += bs

            model.train_acc_metric(torch.argmax(logits.detach(), dim=1), yb)

        epoch_train_loss = running_loss / max(seen, 1)
        train_acc = float(model.train_acc_metric.compute().detach().cpu())
        model.train_acc_metric.reset()

        model.eval()
        val_running_loss = 0.0
        val_seen = 0
        with torch.inference_mode():
            for xb, yb in valid_dl:
                xb = xb.to(device=device, non_blocking=True)
                yb = yb.to(device=device, non_blocking=True)
                logits = model(xb).squeeze(1)
                vloss = model.loss(logits, yb)

                bs = yb.shape[0]
                val_running_loss += float(vloss.detach()) * bs
                val_seen += bs

                model.val_acc_metric(torch.argmax(logits, dim=1), yb)

        epoch_val_loss = val_running_loss / max(val_seen, 1)
        val_acc = float(model.val_acc_metric.compute().detach().cpu())
        model.val_acc_metric.reset()

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_acc)
        else:
            scheduler.step()

        if best_score is None or val_acc > best_score:
            best_score = val_acc
            best_state_dict = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

        tracked.train_loss.append(float(epoch_train_loss))
        tracked.train_acc.append(float(train_acc))
        tracked.val_loss.append(float(epoch_val_loss))
        tracked.val_acc.append(float(val_acc))

        if tracked.verbose:
            print(
                f"Epoch {epoch} start learning rate: {tracked.lr_epoch_start[-1]:.6f}, "
                f"train_loss: {tracked.train_loss[-1]:.4f}, "
                f"train_acc: {tracked.train_acc[-1]:.4f}, "
                f"val_loss: {tracked.val_loss[-1]:.4f}, "
                f"val_acc: {tracked.val_acc[-1]:.4f}"
            )

        if best_es_score is None:
            best_es_score = val_acc
            num_bad_epochs = 0
        else:
            improved = (
                val_acc > (best_es_score + es_min_delta)
                if es_mode == "max"
                else val_acc < (best_es_score - es_min_delta)
            )
            if improved:
                best_es_score = val_acc
                num_bad_epochs = 0
            else:
                num_bad_epochs += 1

        if num_bad_epochs > es_patience:
            break

    return best_state_dict, tracked




## === cell 28
import os

os.makedirs("models", exist_ok=True)

splits = 10
skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)

nn_oof_preds = np.zeros((X_nn_np.shape[0],), dtype=np.int64)
nn_test_logits = np.zeros((X_test_nn.shape[0], NUM_CLASSES), dtype=np.float32)
total_mean_acc = 0.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

trained_folds = 0

if torch.cuda.is_available():
    X_test_nn = X_test_nn.pin_memory()


def batched_logits(model, X_tensor, batch_size=BATCH_SIZE, device=device):
    ds = TensorDataset(X_tensor)
    pin = torch.cuda.is_available()
    num_workers = min(4, (os.cpu_count() or 0)) if (os.cpu_count() or 0) >= 4 else 0
    persistent_workers = num_workers > 0
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=persistent_workers,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    outs = []
    model.eval()
    with torch.inference_mode():
        for (xb,) in dl:
            xb = xb.to(device=device, non_blocking=True)
            outs.append(model(xb).detach().cpu())
    return torch.cat(outs, dim=0).numpy()


for num, (train_idx, valid_idx) in enumerate(skf.split(X_nn_np, y_np)):
    if num > 0:
        break

    print(f"\n\n===Training with fold {num}")
    X_train_np, X_valid_np = X_nn_np[train_idx], X_nn_np[valid_idx]
    y_train_np, y_valid_np = y_np[train_idx], y_np[valid_idx]

    train_dl, valid_dl = prepare_datasets(
        X_train_np, X_valid_np, y_train_np, y_valid_np
    )

    best_state_dict, tracked_values = train_ann(
        train_dl, valid_dl, Model, X_nn_np.shape[1]
    )

    model = Model(X_nn_np.shape[1])
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    model.to(device)
    model.eval()

    X_valid_t = torch.from_numpy(
        np.ascontiguousarray(X_valid_np, dtype=np.float32)
    ).float()
    if torch.cuda.is_available():
        X_valid_t = X_valid_t.pin_memory()

    valid_logits = batched_logits(
        model, X_valid_t, batch_size=BATCH_SIZE, device=device
    )
    preds = np.argmax(valid_logits, axis=1)

    fold_score = accuracy_score(y_valid_np, preds)
    print(f"\n===Fold {num} valid data accuracy score is {fold_score}")

    test_logits = batched_logits(model, X_test_nn, batch_size=BATCH_SIZE, device=device)

    nn_oof_preds[valid_idx] = preds
    nn_test_logits += test_logits
    total_mean_acc += fold_score
    trained_folds += 1

    del (
        X_train_np,
        X_valid_np,
        y_train_np,
        y_valid_np,
        train_dl,
        valid_dl,
        model,
        valid_logits,
        test_logits,
        X_valid_t,
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

nn_test_logits /= max(trained_folds, 1)
total_mean_acc /= max(trained_folds, 1)

print(f"Average accuracy score of all models is {total_mean_acc}")



## === cell 29
np.argmax(nn_test_logits, axis=1)[:10]



## === cell 30
pass



## === cell 31
pass



## === cell 32
test_pred_labels = np.argmax(nn_test_logits, axis=1)
predictions = pd.DataFrame(
    {
        "Id": test["Id"].values,
        "Cover_Type": label_enc.inverse_transform(test_pred_labels),
    }
)
predictions.to_csv("submission.csv", index=False)
predictions.head()



## === cell 33
pass
