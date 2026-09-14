# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

# 5. Target score

0.94695

# 6. Current score

0.36817

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.36817) has done: 'I fix the runtime error caused by the newer `torchmetrics.Accuracy` API (it now requires a `task=` argument), which currently prevents training from starting and therefore prevents any submission from being written. I also remove the notebook-only `%%time` magic so the script runs as plain Python in Kaggle. Finally, I fix a couple of inference/aggregation bugs (test prediction averaging and incorrect tensor wrapping) so predictions are produced deterministically and a valid `submission.csv` with the required `Id,Cover_Type` columns is always generated.'
- What this solution (achieved 0.36817) has done: 'Main bottlenecks are (1) extremely expensive EDA/plotting on 3.6M rows, (2) per-column loops that fit/transform scalers 2× over all features, and (3) slow PyTorch Lightning training caused by extra callback overhead, multi-worker DataLoader startup cost, and doing full-batch predictions without efficient DataLoader/inference batching. I keep the exact model/training logic and epochs, but skip non-essential visualization cells, vectorize the scaling to fit once on train and transform train/test in one shot, and speed up data pipelines (pin_memory/persistent_workers, fewer workers) plus batched inference to avoid huge single-tensor forward passes on CPU/GPU. I also avoid printing/walking the entire input directory and remove per-epoch verbose callback printing (still tracking metrics) to reduce overhead while preserving the same optimization objective and checkpoints. All paths, architecture, loss, epochs, and fold logic remain unchanged.'

# 9. Code solution

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


## === cell 1
train = pd.read_csv(
    "/kaggle/input/tabular-playground-series-dec-2021/train.csv", low_memory=False
)  # , nrows=10000)
test = pd.read_csv(
    "/kaggle/input/tabular-playground-series-dec-2021/test.csv", low_memory=False
)  # , nrows=10000)




## === cell 2
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
features = list(train.columns[1:54])


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
unique_values = df_cols.nunique() < 10
cat_features = unique_values[unique_values == True].index
unique_values = df_cols.nunique() >= 10
num_features = unique_values[unique_values == True].index

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
features.remove("Soil_Type7")
features.remove("Soil_Type15")


## === cell 18
print("Numerical features with the least amount of unique values:")
train[num_features].nunique().sort_values().head(5)


## === cell 19
train.drop(train[train[target] == 5].index, axis=0, inplace=True)
train.reset_index(drop=True, inplace=True)
label_enc = LabelEncoder()
NUM_CLASSES = train[target].nunique()


## === cell 20
s_scaler = StandardScaler()
train_num = s_scaler.fit_transform(train[num_features].to_numpy())
test_num = s_scaler.transform(test[num_features].to_numpy())
train.loc[:, num_features] = train_num
test.loc[:, num_features] = test_num
del train_num, test_num
gc.collect()


## === cell 21
X_nn = train[features].copy()
X_test_nn = test[features].copy()
y = pd.Series(label_enc.fit_transform(train[target]))


## === cell 22
mm_scaler = MinMaxScaler()
X_nn_np = mm_scaler.fit_transform(X_nn.to_numpy())
X_test_np = mm_scaler.transform(X_test_nn.to_numpy())

X_nn = pd.DataFrame(X_nn_np, columns=features)
X_test_nn = torch.tensor(X_test_np, dtype=torch.float32)

del X_nn_np, X_test_np
gc.collect()


## === cell 23
BATCH_SIZE = 4096




## === cell 24
def prepare_datasets(X_nn, X_valid_nn, y_nn, y_valid_nn, batch_size=BATCH_SIZE):
    X_nn_t = torch.tensor(X_nn.to_numpy(), dtype=torch.float32)
    y_nn_t = torch.tensor(y_nn.to_numpy(), dtype=torch.long)
    X_valid_nn_t = torch.tensor(X_valid_nn.to_numpy(), dtype=torch.float32)
    y_valid_nn_t = torch.tensor(y_valid_nn.to_numpy(), dtype=torch.long)

    train_ds = TensorDataset(X_nn_t, y_nn_t)
    valid_ds = TensorDataset(X_valid_nn_t, y_valid_nn_t)

    num_workers = 2  # reduces process startup overhead vs 4 for short epoch times
    pin = torch.cuda.is_available()
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )
    valid_dl = DataLoader(
        valid_ds,
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
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




## === cell 28
def train_ann(train_ds, valid_ds, Model=Model, input_shape=X_nn.shape[1]):

    model = Model(input_shape)
    model.apply(initialize_weights)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="models",
        filename=f"model_" + "{val_acc:.4}",
        monitor="val_acc",
        mode="max",
        save_weights_only=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_acc", min_delta=0.00, patience=20, verbose=False, mode="max"
    )

    params_tracker_callback = ParamsTracker(verbose=True)

    trainer = pl.Trainer(
        fast_dev_run=False,
        max_epochs=60,
        precision=32,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        val_check_interval=1.0,
        callbacks=[checkpoint_callback, early_stop_callback, params_tracker_callback],
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator=("gpu" if torch.cuda.is_available() else "cpu"),
        devices=1,
    )

    trainer.fit(model, train_ds, valid_ds)

    model.eval()
    best_model_path = checkpoint_callback.best_model_path
    return best_model_path, params_tracker_callback




## === cell 29
@torch.no_grad()
def predict_logits_batched(model, X_tensor, batch_size=BATCH_SIZE, device=DEVICE):
    model = model.to(device)
    model.eval()
    dl = DataLoader(
        TensorDataset(X_tensor),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    outs = []
    for (xb,) in dl:
        xb = xb.to(device, non_blocking=True)
        outs.append(model(xb).detach().cpu())
    return torch.cat(outs, dim=0).numpy()




## === cell 30
start_time = time.time()

splits = 10
skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)

nn_oof_preds = np.zeros((X_nn.shape[0],), dtype=np.int64)
nn_test_logits_sum = np.zeros((X_test_nn.shape[0], 6), dtype=np.float32)
total_mean_acc = 0.0

for num, (train_idx, valid_idx) in enumerate(skf.split(X_nn, y)):
    if num > 0:
        break
    print(f"\n\n===Training with fold {num}")
    X_train, X_valid = X_nn.loc[train_idx], X_nn.loc[valid_idx]
    y_train, y_valid = y.loc[train_idx], y.loc[valid_idx]

    train_ds, valid_ds = prepare_datasets(X_train, X_valid, y_train, y_valid)

    best_model_path, tracked_values = train_ann(
        train_ds, valid_ds, Model, X_nn.shape[1]
    )

    model = Model(X_nn.shape[1])
    ckpt = torch.load(best_model_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])

    X_valid_t = torch.tensor(X_valid.to_numpy(), dtype=torch.float32)
    valid_logits = predict_logits_batched(model, X_valid_t, batch_size=BATCH_SIZE)
    preds = np.argmax(valid_logits, axis=1)

    fold_score = accuracy_score(y_valid, preds)
    print(f"\n===Fold {num} valid data accuracy score is {fold_score}")

    test_logits = predict_logits_batched(model, X_test_nn, batch_size=BATCH_SIZE)

    nn_oof_preds[valid_idx] = preds
    nn_test_logits_sum += test_logits
    total_mean_acc += fold_score / splits

print(f"Average accuracy score of all models is {total_mean_acc}")
print(f"Elapsed seconds: {time.time() - start_time:.2f}")


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/4243045269.py in <cell line: 0>()
     17     train_ds, valid_ds = prepare_datasets(X_train, X_valid, y_train, y_valid)
     18 
---> 19     best_model_path, tracked_values = train_ann(
     20         train_ds, valid_ds, Model, X_nn.shape[1]
     21     )

/tmp/ipykernel_11/2092933517.py in train_ann(train_ds, valid_ds, Model, input_shape)
     38     )
     39 
---> 40     trainer.fit(model, train_ds, valid_ds)
     41 
     42     model.eval()

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/trainer.py in fit(self, model, train_dataloaders, val_dataloaders, datamodule, ckpt_path)
    558         self.training = True
    559         self.should_stop = False
--> 560         call._call_and_handle_interrupt(
    561             self, self._fit_impl, model, train_dataloaders, val_dataloaders, datamodule, ckpt_path
    562         )

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/call.py in _call_and_handle_interrupt(trainer, trainer_fn, *args, **kwargs)
     47         if trainer.strategy.launcher is not None:
     48             return trainer.strategy.launcher.launch(trainer_fn, *args, trainer=trainer, **kwargs)
---> 49         return trainer_fn(*args, **kwargs)
     50 
     51     except _TunerExitException:

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/trainer.py in _fit_impl(self, model, train_dataloaders, val_dataloaders, datamodule, ckpt_path)
    596             model_connected=self.lightning_module is not None,
    597         )
--> 598         self._run(model, ckpt_path=ckpt_path)
    599 
    600         assert self.state.stopped

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/trainer.py in _run(self, model, ckpt_path)
   1009         # RUN THE TRAINER
   1010         # ----------------------------
-> 1011         results = self._run_stage()
   1012 
   1013         # ----------------------------

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/trainer.py in _run_stage(self)
   1053                 self._run_sanity_check()
   1054             with torch.autograd.set_detect_anomaly(self._detect_anomaly):
-> 1055                 self.fit_loop.run()
   1056             return None
   1057         raise RuntimeError(f"Unexpected state {self.state}")

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/fit_loop.py in run(self)
    214             try:
    215                 self.on_advance_start()
--> 216                 self.advance()
    217                 self.on_advance_end()
    218             except StopIteration:

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/fit_loop.py in advance(self)
    456         with self.trainer.profiler.profile("run_training_epoch"):
    457             assert self._data_fetcher is not None
--> 458             self.epoch_loop.run(self._data_fetcher)
    459 
    460     def on_advance_end(self) -> None:

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/training_epoch_loop.py in run(self, data_fetcher)
    150         while not self.done:
    151             try:
--> 152                 self.advance(data_fetcher)
    153                 self.on_advance_end(data_fetcher)
    154             except StopIteration:

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/training_epoch_loop.py in advance(self, data_fetcher)
    346                 if trainer.lightning_module.automatic_optimization:
    347                     # in automatic optimization, there can only be one optimizer
--> 348                     batch_output = self.automatic_optimization.run(trainer.optimizers[0], batch_idx, kwargs)
    349                 else:
    350                     batch_output = self.manual_optimization.run(kwargs)

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/optimization/automatic.py in run(self, optimizer, batch_idx, kwargs)
    190         # gradient update with accumulated gradients
    191         else:
--> 192             self._optimizer_step(batch_idx, closure)
    193 
    194         result = closure.consume_result()

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/optimization/automatic.py in _optimizer_step(self, batch_idx, train_step_and_backward_closure)
    268 
    269         # model hook
--> 270         call._call_lightning_module_hook(
    271             trainer,
    272             "optimizer_step",

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/call.py in _call_lightning_module_hook(trainer, hook_name, pl_module, *args, **kwargs)
    175 
    176     with trainer.profiler.profile(f"[LightningModule]{pl_module.__class__.__name__}.{hook_name}"):
--> 177         output = fn(*args, **kwargs)
    178 
    179     # restore current_fx when nested context

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/core/module.py in optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure)
   1364 
   1365         """
-> 1366         optimizer.step(closure=optimizer_closure)
   1367 
   1368     def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer: Optimizer) -> None:

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/core/optimizer.py in step(self, closure, **kwargs)
    152 
    153         assert self._strategy is not None
--> 154         step_output = self._strategy.optimizer_step(self._optimizer, closure, **kwargs)
    155 
    156         self._on_after_step()

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/strategies/strategy.py in optimizer_step(self, optimizer, closure, model, **kwargs)
    237         # TODO(fabric): remove assertion once strategy's optimizer_step typing is fixed
    238         assert isinstance(model, pl.LightningModule)
--> 239         return self.precision_plugin.optimizer_step(optimizer, model=model, closure=closure, **kwargs)
    240 
    241     def _setup_model_and_optimizers(self, model: Module, optimizers: list[Optimizer]) -> tuple[Module, list[Optimizer]]:

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/plugins/precision/precision.py in optimizer_step(self, optimizer, model, closure, **kwargs)
    121         """Hook to run the optimizer step."""
    122         closure = partial(self._wrap_closure, model, optimizer, closure)
--> 123         return optimizer.step(closure=closure, **kwargs)
    124 
    125     def _clip_gradients(

/usr/local/lib/python3.11/dist-packages/torch/optim/optimizer.py in wrapper(*args, **kwargs)
    491                             )
    492 
--> 493                 out = func(*args, **kwargs)
    494                 self._optimizer_step_code()
    495 

/usr/local/lib/python3.11/dist-packages/torch/optim/optimizer.py in _use_grad(self, *args, **kwargs)
     89             torch.set_grad_enabled(self.defaults["differentiable"])
     90             torch._dynamo.graph_break()
---> 91             ret = func(self, *args, **kwargs)
     92         finally:
     93             torch._dynamo.graph_break()

/usr/local/lib/python3.11/dist-packages/torch/optim/adamw.py in step(self, closure)
    218         if closure is not None:
    219             with torch.enable_grad():
--> 220                 loss = closure()
    221 
    222         for group in self.param_groups:

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/plugins/precision/precision.py in _wrap_closure(self, model, optimizer, closure)
    107 
    108         """
--> 109         closure_result = closure()
    110         self._after_closure(model, optimizer)
    111         return closure_result

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/optimization/automatic.py in __call__(self, *args, **kwargs)
    144     @override
    145     def __call__(self, *args: Any, **kwargs: Any) -> Optional[Tensor]:
--> 146         self._result = self.closure(*args, **kwargs)
    147         return self._result.loss
    148 

/usr/local/lib/python3.11/dist-packages/torch/utils/_contextlib.py in decorate_context(*args, **kwargs)
    114     def decorate_context(*args, **kwargs):
    115         with ctx_factory():
--> 116             return func(*args, **kwargs)
    117 
    118     return decorate_context

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/optimization/automatic.py in closure(self, *args, **kwargs)
    138 
    139         if self._backward_fn is not None and step_output.closure_loss is not None:
--> 140             self._backward_fn(step_output.closure_loss)
    141 
    142         return step_output

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/optimization/automatic.py in backward_fn(loss)
    239 
    240         def backward_fn(loss: Tensor) -> None:
--> 241             call._call_strategy_hook(self.trainer, "backward", loss, optimizer)
    242 
    243         return backward_fn

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/call.py in _call_strategy_hook(trainer, hook_name, *args, **kwargs)
    327 
    328     with trainer.profiler.profile(f"[Strategy]{trainer.strategy.__class__.__name__}.{hook_name}"):
--> 329         output = fn(*args, **kwargs)
    330 
    331     # restore current_fx when nested context

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/strategies/strategy.py in backward(self, closure_loss, optimizer, *args, **kwargs)
    211         closure_loss = self.precision_plugin.pre_backward(closure_loss, self.lightning_module)
    212 
--> 213         self.precision_plugin.backward(closure_loss, self.lightning_module, optimizer, *args, **kwargs)
    214 
    215         closure_loss = self.precision_plugin.post_backward(closure_loss, self.lightning_module)

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/plugins/precision/precision.py in backward(self, tensor, model, optimizer, *args, **kwargs)
     71 
     72         """
---> 73         model.backward(tensor, *args, **kwargs)
     74 
     75     @override

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/core/module.py in backward(self, loss, *args, **kwargs)
   1133             self._fabric.backward(loss, *args, **kwargs)
   1134         else:
-> 1135             loss.backward(*args, **kwargs)
   1136 
   1137     def toggle_optimizer(self, optimizer: Union[Optimizer, LightningOptimizer]) -> None:

/usr/local/lib/python3.11/dist-packages/torch/_tensor.py in backward(self, gradient, retain_graph, create_graph, inputs)
    624                 inputs=inputs,
    625             )
--> 626         torch.autograd.backward(
    627             self, gradient, retain_graph, create_graph, inputs=inputs
    628         )

/usr/local/lib/python3.11/dist-packages/torch/autograd/__init__.py in backward(tensors, grad_tensors, retain_graph, create_graph, grad_variables, inputs)
    345     # some Python versions print out the first line of a multi-line function
    346     # calls in the traceback and some print out the last line
--> 347     _engine_run_backward(
    348         tensors,
    349         grad_tensors_,

/usr/local/lib/python3.11/dist-packages/torch/autograd/graph.py in _engine_run_backward(t_outputs, *args, **kwargs)
    821         unregister_hooks = _register_logging_hooks_on_whole_graph(t_outputs)
    822     try:
--> 823         return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
    824             t_outputs, *args, **kwargs
    825         )  # Calls into the C++ engine to run the backward pass

RuntimeError: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility

## === cell 31
if "model" in globals():
    print(
        np.argmax(
            model(torch.tensor(X_valid.to_numpy(), dtype=torch.float32))
            .detach()
            .numpy(),
            axis=1,
        )[:10]
    )


## === cell 32
if "model" in globals():
    print(
        accuracy_score(
            y_train,
            np.argmax(
                model(torch.tensor(X_train.to_numpy(), dtype=torch.float32))
                .detach()
                .numpy(),
                axis=1,
            ),
        )
    )


## === cell 33
if "model" in globals():
    print(
        accuracy_score(
            y_valid,
            np.argmax(
                model(torch.tensor(X_valid.to_numpy(), dtype=torch.float32))
                .detach()
                .numpy(),
                axis=1,
            ),
        )
    )


## === cell 34
avg_test_logits = nn_test_logits_sum / 1.0
test_pred_labels = np.argmax(avg_test_logits, axis=1)
test_pred_cover_type = label_enc.inverse_transform(test_pred_labels)

predictions = pd.DataFrame()
predictions["Id"] = test["Id"].astype(np.int64)
predictions["Cover_Type"] = test_pred_cover_type.astype(np.int64)

predictions.to_csv("submission.csv", index=False)
print(predictions.head())
print("Saved submission.csv with shape:", predictions.shape)


## === cell 35
pass
