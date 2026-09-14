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

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")




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
train.info(memory_usage="deep")




## === cell 4
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
train.head()




## === cell 7
target = "Cover_Type"

features = list(train.columns[1:54])




## === cell 8
train[target].value_counts()




## === cell 9
pass




## === cell 10
pass




## === cell 11
train[features].describe()




## === cell 12
df = pd.concat([train[features], test[features]], axis=0)
df.reset_index(inplace=True, drop=True)

unique_values = df[features].nunique() < 10
cat_features = unique_values[unique_values == True].index
unique_values = df[features].nunique() >= 10
num_features = unique_values[unique_values == True].index

print(f"There are {len(cat_features)} categorical features: {cat_features}")
print(f"\nThere are {len(num_features)} continuous features: {num_features}")




## === cell 13
train.isna().sum().sum(), test.isna().sum().sum()




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
train[num_features] = s_scaler.fit_transform(train[num_features])
test[num_features] = s_scaler.transform(test[num_features])

train[num_features] = train[num_features].astype(np.float32)
test[num_features] = test[num_features].astype(np.float32)




## === cell 21
X_nn = train[features].copy()
X_test_nn = test[features].copy()
y = pd.Series(label_enc.fit_transform(train[target]))




## === cell 22
mm_scaler = MinMaxScaler()
X_nn = pd.DataFrame(
    mm_scaler.fit_transform(X_nn), columns=X_nn.columns, index=X_nn.index
)
X_test_nn = pd.DataFrame(
    mm_scaler.transform(X_test_nn), columns=X_test_nn.columns, index=X_test_nn.index
)

X_test_nn = torch.tensor(X_test_nn.to_numpy()).float()




## === cell 23
BATCH_SIZE = 1_048_576  # fits in GPU memory for this model and speeds up training




## === cell 24
def prepare_datasets(X_nn, X_valid_nn, y_nn, y_valid_nn, batch_size=BATCH_SIZE):
    X_nn = torch.tensor(X_nn.to_numpy(), dtype=torch.float32)
    y_nn = torch.tensor(y_nn.to_numpy(), dtype=torch.long)
    X_valid_nn = torch.tensor(X_valid_nn.to_numpy(), dtype=torch.float32)
    y_valid_nn = torch.tensor(y_valid_nn.to_numpy(), dtype=torch.long)

    train_ds = TensorDataset(X_nn, y_nn)
    valid_ds = TensorDataset(X_valid_nn, y_valid_nn)

    train_loader = DataLoader(
        train_ds,
        batch_size,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
        shuffle=False,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
        shuffle=False,
    )

    return train_loader, valid_loader




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
        self.val_loss.append(metrics_logs["val_loss"].item())
        val_acc = metrics_logs.get("val_acc")
        if val_acc is not None:
            self.val_acc.append(val_acc.item())
        else:
            try:
                self.val_acc.append(module.val_acc_metric.compute().item())
            except Exception:
                pass

    def on_train_epoch_end(self, trainer, module):
        metrics_logs = trainer.logged_metrics
        loss = metrics_logs.get("loss")
        if loss is not None:
            self.train_loss.append(loss.item())
        else:
            self.train_loss.append(None)

        train_acc = metrics_logs.get("train_acc")
        if train_acc is not None:
            self.train_acc.append(train_acc.item())
        else:
            self.train_acc.append(None)

        if self.verbose:
            lr = self.lr_epoch_start[-1] if self.lr_epoch_start else float("nan")
            loss_str = (
                f"{self.train_loss[-1]:.4f}"
                if self.train_loss[-1] is not None
                else "N/A"
            )
            acc_str = (
                f"{self.train_acc[-1]:.4f}" if self.train_acc[-1] is not None else "N/A"
            )
            val_loss_str = f"{self.val_loss[-1]:.4f}" if self.val_loss else "N/A"
            val_acc_str = f"{self.val_acc[-1]:.4f}" if self.val_acc else "N/A"
            print(
                f"Epoch {module.current_epoch} start learning rate: {lr:.6f}, "
                f"train_loss: {loss_str}, "
                f"train_acc: {acc_str}, "
                f"val_loss: {val_loss_str}, "
                f"val_acc: {val_acc_str}"
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
        self.log("val_loss", val_loss, prog_bar=True, on_epoch=True, logger=True)
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
def train_ann(train_ds, valid_ds, Model=Model, input_shape=X_nn.shape[1]):

    model = Model(input_shape)
    if torch.cuda.is_available():
        model = torch.compile(model)

    model.apply(initialize_weights)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="models",
        filename="model_{val_acc:.4f}",
        monitor="val_acc",
        mode="max",
        save_weights_only=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_acc", min_delta=0.00, patience=10, verbose=False, mode="max"
    )

    params_tracker_callback = ParamsTracker(verbose=True)

    trainer = pl.Trainer(
        fast_dev_run=False,
        max_epochs=30,  # reduced from 60 to stay within time limit
        precision=16,  # mixed precision for speed
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        val_check_interval=1.0,
        logger=False,
        callbacks=[checkpoint_callback, early_stop_callback, params_tracker_callback],
        accelerator="auto",
        devices=1,
        enable_progress_bar=False,
    )

    trainer.fit(model, train_ds, valid_ds)

    best_model_path = (
        checkpoint_callback.best_model_path or checkpoint_callback.last_model_path
    )

    return best_model_path, params_tracker_callback




## === cell 29
splits = 10
skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)

nn_oof_preds = np.zeros((X_nn.shape[0],))
nn_test_preds = np.zeros((X_test_nn.shape[0], NUM_CLASSES))
total_mean_acc = 0

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
    model.load_state_dict(torch.load(best_model_path)["state_dict"])
    model.eval()

    preds = np.argmax(
        model(torch.tensor(X_valid.to_numpy()).float()).detach().numpy(), axis=1
    )

    fold_score = accuracy_score(y_valid, preds)
    print(f"\n===Fold {num} valid data accuracy score is {fold_score}")

    test_pred_probs = model(X_test_nn).detach().numpy()  # shape (n_test, NUM_CLASSES)
    nn_test_preds += test_pred_probs / splits

    nn_oof_preds[valid_idx] = preds
    total_mean_acc += fold_score / splits

print(f"Average accuracy score of all models is {total_mean_acc}")




## === cell 30
np.argmax(model(torch.tensor(X_valid.to_numpy()).float()).detach().numpy(), axis=1)[:10]




## === cell 31
accuracy_score(
    y_train,
    np.argmax(model(torch.tensor(X_train.to_numpy()).float()).detach().numpy(), axis=1),
)




## === cell 32
accuracy_score(
    y_valid,
    np.argmax(model(torch.tensor(X_valid.to_numpy()).float()).detach().numpy(), axis=1),
)




## === cell 33
pred_encoded = np.argmax(nn_test_preds, axis=1)  # encoded class indices
pred_original = label_enc.inverse_transform(pred_encoded)  # map back to original labels
predictions = pd.DataFrame({"Id": test["Id"], "Cover_Type": pred_original})
predictions.to_csv("submission.csv", index=False)
predictions.head()




## === cell 34
predictions["Cover_Type"].hist()
