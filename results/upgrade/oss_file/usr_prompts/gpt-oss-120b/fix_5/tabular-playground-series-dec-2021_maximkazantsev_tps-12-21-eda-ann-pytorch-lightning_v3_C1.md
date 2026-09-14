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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
import torchmetrics
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint



## === cell 1
train_path = "/kaggle/input/tabular-playground-series-dec-2021/train.csv"
test_path = "/kaggle/input/tabular-playground-series-dec-2021/test.csv"
train = pd.read_csv(train_path, low_memory=False)
test = pd.read_csv(test_path, low_memory=False)




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
                else:
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




## === cell 3
train = reduce_mem_usage(train)
test = reduce_mem_usage(test)



## === cell 4
target = "Cover_Type"
features = list(train.columns[1:54])  # keep original slice
train = train[train[target] != 5].reset_index(drop=True)
label_enc = LabelEncoder()
NUM_CLASSES = train[target].nunique()



## === cell 5
num_features = features
s_scaler = StandardScaler()
for col in num_features:
    train[col] = s_scaler.fit_transform(train[[col]].values)
    test[col] = s_scaler.transform(test[[col]].values)



## === cell 6
X_nn = train[features].copy()
X_test_nn = test[features].copy()
y = pd.Series(label_enc.fit_transform(train[target]))
mm_scaler = MinMaxScaler()
for col in X_nn.columns:
    X_nn[col] = mm_scaler.fit_transform(X_nn[[col]].values)
    X_test_nn[col] = mm_scaler.transform(X_test_nn[[col]].values)



## === cell 7
BATCH_SIZE = 4096




## === cell 8
def prepare_datasets(X_train, X_valid, y_train, y_valid, batch_size=BATCH_SIZE):
    X_train_t = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
    y_train_t = torch.tensor(y_train.to_numpy(), dtype=torch.long)
    X_valid_t = torch.tensor(X_valid.to_numpy(), dtype=torch.float32)
    y_valid_t = torch.tensor(y_valid.to_numpy(), dtype=torch.long)

    train_ds = TensorDataset(X_train_t, y_train_t)
    valid_ds = TensorDataset(X_valid_t, y_valid_t)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, drop_last=False, num_workers=4
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=batch_size, drop_last=False, num_workers=4
    )

    for data, label in train_loader:
        print(f"Train batch shape: {data.shape}, {label.shape}")
        break
    for data, label in valid_loader:
        print(f"Valid batch shape: {data.shape}, {label.shape}")
        break
    return train_loader, valid_loader




## === cell 9
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data)




## === cell 10
class ParamsTracker(pl.callbacks.Callback):
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []
        self.lr_epoch_start = []

    def on_train_epoch_start(self, trainer, pl_module):
        lr = trainer.optimizers[0].state_dict()["param_groups"][0]["lr"]
        self.lr_epoch_start.append(lr)

    def on_validation_epoch_end(self, trainer, pl_module):
        logs = trainer.logged_metrics
        self.val_loss.append(logs["val_loss"].item())
        self.val_acc.append(logs["val_acc"].item())

    def on_train_epoch_end(self, trainer, pl_module):
        logs = trainer.logged_metrics
        self.train_loss.append(logs["loss"].item())
        self.train_acc.append(logs["train_acc"].item())
        if self.verbose:
            print(
                f"Epoch {pl_module.current_epoch} lr: {self.lr_epoch_start[-1]:.6f}, "
                f"train_loss: {self.train_loss[-1]:.4f}, train_acc: {self.train_acc[-1]:.4f}, "
                f"val_loss: {self.val_loss[-1]:.4f}, val_acc: {self.val_acc[-1]:.4f}"
            )




## === cell 11
class Model(pl.LightningModule):
    def __init__(self, input_shape):
        super().__init__()
        self.input = nn.Linear(input_shape, 128)
        self.hidden1 = nn.Linear(128, 64)
        self.hidden2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 6)

        self.dr = 0.2
        self.swish = F.hardswish
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_acc_metric = torchmetrics.Accuracy(
            task="multiclass", num_classes=6, average="micro"
        )
        self.val_acc_metric = torchmetrics.Accuracy(
            task="multiclass", num_classes=6, average="micro"
        )

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
        y_hat = self(X)
        loss = self.loss_fn(y_hat, y)
        self.train_acc_metric(torch.argmax(y_hat, dim=1), y)
        self.log("loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        acc = self.train_acc_metric.compute()
        self.log("train_acc", acc, prog_bar=True, on_epoch=True)
        self.train_acc_metric.reset()

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        val_loss = self.loss_fn(y_hat, y)
        self.val_acc_metric(torch.argmax(y_hat, dim=1), y)
        self.log("val_loss", val_loss, prog_bar=True, on_epoch=True)
        return val_loss

    def on_validation_epoch_end(self):
        acc = self.val_acc_metric.compute()
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)
        self.val_acc_metric.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=1e-3, eps=1e-8, weight_decay=1e-2
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=7,
            min_lr=1e-4,
            eps=1e-8,
            threshold=0.005,
            threshold_mode="abs",
            verbose=False,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_acc",
        }




## === cell 12
def train_ann(train_loader, valid_loader, ModelClass=Model, input_shape=X_nn.shape[1]):
    os.makedirs("models", exist_ok=True)

    model = ModelClass(input_shape)
    model.apply(initialize_weights)

    checkpoint_cb = ModelCheckpoint(
        dirpath="models",
        filename="model_{val_acc:.4f}",
        monitor="val_acc",
        mode="max",
        save_weights_only=False,
    )
    early_stop_cb = EarlyStopping(monitor="val_acc", patience=5, mode="max")
    params_tracker_cb = ParamsTracker(verbose=False)

    trainer = pl.Trainer(
        max_epochs=5,  # reduced epochs for quick execution
        precision=32,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        logger=False,
        callbacks=[checkpoint_cb, early_stop_cb, params_tracker_cb],
    )
    trainer.fit(model, train_loader, valid_loader)
    best_path = checkpoint_cb.best_model_path
    return best_path, params_tracker_cb




## === cell 13
splits = 10
skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)

nn_oof_preds = np.zeros((X_nn.shape[0],))
total_mean_acc = 0.0

for fold, (train_idx, valid_idx) in enumerate(skf.split(X_nn, y)):
    if fold > 0:  # keep only first fold for speed (original code broke after first)
        break
    print(f"\n=== Training fold {fold} ===")
    X_train, X_valid = X_nn.iloc[train_idx], X_nn.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    train_loader, valid_loader = prepare_datasets(X_train, X_valid, y_train, y_valid)

    best_model_path, _ = train_ann(train_loader, valid_loader, Model, X_nn.shape[1])

    ckpt = torch.load(best_model_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    model = Model(X_nn.shape[1])
    model.load_state_dict(state_dict)
    model.eval()

    val_logits = model(torch.tensor(X_valid.to_numpy()).float())
    val_preds = torch.argmax(val_logits, dim=1).cpu().numpy()
    nn_oof_preds[valid_idx] = val_preds

    fold_acc = accuracy_score(y_valid, val_preds)
    print(f"Fold {fold} validation accuracy: {fold_acc:.4f}")
    total_mean_acc += fold_acc / splits

print(f"Average accuracy across processed folds: {total_mean_acc:.4f}")



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/362548905.py in <cell line: 0>()
     14     train_loader, valid_loader = prepare_datasets(X_train, X_valid, y_train, y_valid)
     15 
---> 16     best_model_path, _ = train_ann(train_loader, valid_loader, Model, X_nn.shape[1])
     17 
     18     # Load checkpoint (handles both dict formats)

/tmp/ipykernel_55/1420979965.py in train_ann(train_loader, valid_loader, ModelClass, input_shape)
     24         callbacks=[checkpoint_cb, early_stop_cb, params_tracker_cb],
     25     )
---> 26     trainer.fit(model, train_loader, valid_loader)
     27     best_path = checkpoint_cb.best_model_path
     28     return best_path, params_tracker_cb

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
   1051         if self.training:
   1052             with isolate_rng():
-> 1053                 self._run_sanity_check()
   1054             with torch.autograd.set_detect_anomaly(self._detect_anomaly):
   1055                 self.fit_loop.run()

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/trainer.py in _run_sanity_check(self)
   1080 
   1081             # run eval step
-> 1082             val_loop.run()
   1083 
   1084             call._call_callback_hooks(self, "on_sanity_check_end")

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/utilities.py in _decorator(self, *args, **kwargs)
    177             context_manager = torch.no_grad
    178         with context_manager():
--> 179             return loop_run(self, *args, **kwargs)
    180 
    181     return _decorator

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/evaluation_loop.py in run(self)
    150                 self.on_iteration_done()
    151         self._store_dataloader_outputs()
--> 152         return self.on_run_end()
    153 
    154     def setup_data(self) -> None:

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/evaluation_loop.py in on_run_end(self)
    293 
    294         # hook
--> 295         self._on_evaluation_epoch_end()
    296 
    297         logged_outputs, self._logged_outputs = self._logged_outputs, []  # free memory

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/evaluation_loop.py in _on_evaluation_epoch_end(self)
    372 
    373         hook_name = "on_test_epoch_end" if trainer.testing else "on_validation_epoch_end"
--> 374         call._call_callback_hooks(trainer, hook_name)
    375         call._call_lightning_module_hook(trainer, hook_name)
    376 

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/trainer/call.py in _call_callback_hooks(trainer, hook_name, monitoring_callbacks, *args, **kwargs)
    226         if callable(fn):
    227             with trainer.profiler.profile(f"[Callback]{callback.state_key}.{hook_name}"):
--> 228                 fn(trainer, trainer.lightning_module, *args, **kwargs)
    229 
    230     if pl_module:

/tmp/ipykernel_55/2494758041.py in on_validation_epoch_end(self, trainer, pl_module)
     15         logs = trainer.logged_metrics
     16         self.val_loss.append(logs["val_loss"].item())
---> 17         self.val_acc.append(logs["val_acc"].item())
     18 
     19     def on_train_epoch_end(self, trainer, pl_module):

KeyError: 'val_acc'

## === cell 14
predictions = pd.DataFrame()
predictions["Id"] = test["Id"]
test_logits = model(torch.tensor(X_test_nn.to_numpy()).float())
test_pred_labels = torch.argmax(test_logits, dim=1).cpu().numpy()
predictions["Cover_Type"] = label_enc.inverse_transform(test_pred_labels)

submission_path = "submission.csv"
predictions.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
predictions.head()



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2544772929.py in <cell line: 0>()
      1 predictions = pd.DataFrame()
      2 predictions["Id"] = test["Id"]
----> 3 test_logits = model(torch.tensor(X_test_nn.to_numpy()).float())
      4 test_pred_labels = torch.argmax(test_logits, dim=1).cpu().numpy()
      5 predictions["Cover_Type"] = label_enc.inverse_transform(test_pred_labels)

NameError: name 'model' is not defined

## === cell 15
predictions["Cover_Type"].hist()

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'Cover_Type'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/2956745796.py in <cell line: 0>()
----> 1 predictions["Cover_Type"].hist()

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'Cover_Type'
