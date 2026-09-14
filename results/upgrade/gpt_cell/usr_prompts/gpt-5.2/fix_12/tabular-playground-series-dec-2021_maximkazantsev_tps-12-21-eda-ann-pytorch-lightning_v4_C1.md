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

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
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
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
import time
import gc
import torchmetrics


pd.set_option('display.max_rows', 150)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
train = pd.read_csv("/kaggle/input/tabular-playground-series-dec-2021/train.csv", low_memory=False)#, nrows=10000)
test = pd.read_csv("/kaggle/input/tabular-playground-series-dec-2021/test.csv", low_memory=False)#, nrows=10000)


## === cell 2
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

train = reduce_mem_usage(train)
test = reduce_mem_usage(test)


## === cell 3
train.info(memory_usage="deep")


## === cell 4
test.info(memory_usage="deep")


## === cell 5
colors = ["lightcoral", "sandybrown", "darkorange", "mediumseagreen",
          "lightseagreen", "cornflowerblue", "mediumpurple", "palevioletred",
          "lightskyblue", "sandybrown", "yellowgreen", "indianred",
          "lightsteelblue", "mediumorchid", "deepskyblue"]


## === cell 6
train.head()


## === cell 7
target = "Cover_Type"

features = list(train.columns[1:54])


## === cell 8
train[target].value_counts()


## === cell 9
fig, ax = plt.subplots(figsize=(5, 6))
pie = ax.pie([len(train), len(test)],
             labels=["Train dataset", "Test dataset"],
             colors=["salmon", "teal"],
             textprops={"fontsize": 15},
             autopct='%1.1f%%')
ax.axis("equal")
ax.set_title("Dataset length comparison", fontsize=18)
fig.set_facecolor('white')
plt.show();


## === cell 10
fig, ax = plt.subplots(figsize=(14, 8))

bars = ax.bar(train[target].value_counts().sort_index().index,
                  train[target].value_counts().sort_index().values,
                  color=colors,
                  edgecolor="black")
ax.set_title("Target distribution", fontsize=20, pad=15)
ax.set_ylabel("Count", fontsize=14, labelpad=15)
ax.set_xlabel("Target label", fontsize=14, labelpad=20)
ax.tick_params(axis="x", pad=20)
ax.bar_label(bars, train[target].value_counts().sort_index().values,
                 padding=3, fontsize=12)
ax.bar_label(bars, [f"{x*100:2.1f}%" for x in train[target].value_counts().sort_index().values/len(train)],
                 padding=-20, fontsize=12)
ax.margins(0.025, 0.06)
ax.grid(axis="y")

plt.show();


## === cell 11
train[features].describe()


## === cell 12
df = pd.concat([train[features], test[features]], axis=0)
df.reset_index(inplace=True, drop=True)

unique_values = df[features].nunique() < 10
cat_features = unique_values[unique_values==True].index
unique_values = df[features].nunique() >= 10
num_features = unique_values[unique_values==True].index

print(f"There are {len(cat_features)} categorical features: {cat_features}")
print(f"\nThere are {len(num_features)} continuous features: {num_features}")


## === cell 13
train.isna().sum().sum(), test.isna().sum().sum()


## === cell 14
df = pd.concat([train[num_features], test[num_features]], axis=0)
columns = df.columns.values

cols = 3
rows = len(columns) // cols + 1

fig, axs = plt.subplots(ncols=cols, nrows=rows, figsize=(16,20), sharex=False)

plt.subplots_adjust(hspace = 0.3)
i=0

for r in np.arange(0, rows, 1):
    for c in np.arange(0, cols, 1):
        if i >= len(columns):
            axs[r, c].set_visible(False)
        else:
            hist1 = axs[r, c].hist(train[columns[i]].values,
                                   range=(df[columns[i]].min(),
                                          df[columns[i]].max()),
                                   bins=40,
                                   color="deepskyblue",
                                   edgecolor="black",
                                   alpha=0.7,
                                   label="Train Dataset")
            hist2 = axs[r, c].hist(test[columns[i]].values,
                                   range=(df[columns[i]].min(),
                                          df[columns[i]].max()),
                                   bins=40,
                                   color="palevioletred",
                                   edgecolor="black",
                                   alpha=0.7,
                                   label="Test Dataset")
            axs[r, c].set_title(columns[i], fontsize=12, pad=5)
            axs[r, c].set_yticks(axs[r, c].get_yticks())
            axs[r, c].set_yticklabels([str(int(i/1000))+"k" for i in axs[r, c].get_yticks()])
            axs[r, c].tick_params(axis="y", labelsize=10)
            axs[r, c].tick_params(axis="x", labelsize=10)
            axs[r, c].grid(axis="y")
            if i == 0:
                axs[r, c].legend(fontsize=10)
                                  
        i+=1
plt.show();


## === cell 15
df = pd.concat([train[cat_features], test[cat_features]], axis=0)
columns = df.columns.values

cols = 4
rows = len(columns) // cols + 1

fig, axs = plt.subplots(ncols=cols, nrows=rows, figsize=(16,40), sharex=False)

plt.subplots_adjust(hspace = 0.3)
i=0

for r in np.arange(0, rows, 1):
    for c in np.arange(0, cols, 1):
        if i >= len(columns):
            axs[r, c].set_visible(False)
        else:
            hist1 = axs[r, c].hist(train[columns[i]].values,
                                   range=(df[columns[i]].min(),
                                          df[columns[i]].max()),
                                   bins=40,
                                   color="deepskyblue",
                                   edgecolor="black",
                                   alpha=0.7,
                                   label="Train Dataset")
            hist2 = axs[r, c].hist(test[columns[i]].values,
                                   range=(df[columns[i]].min(),
                                          df[columns[i]].max()),
                                   bins=40,
                                   color="palevioletred",
                                   edgecolor="black",
                                   alpha=0.7,
                                   label="Test Dataset")
            axs[r, c].set_title(columns[i], fontsize=12, pad=5)
            axs[r, c].set_yticks(axs[r, c].get_yticks())
            axs[r, c].set_yticklabels([str(int(i/1000))+"k" for i in axs[r, c].get_yticks()])
            axs[r, c].tick_params(axis="y", labelsize=10)
            axs[r, c].tick_params(axis="x", labelsize=10)
            axs[r, c].grid(axis="y")
            if i == 0:
                axs[r, c].legend(fontsize=10)
                                  
        i+=1
plt.show();


## === cell 16
print(f"Rows with soil type 7: {(train['Soil_Type7'] == 1).sum() + (test['Soil_Type7'] == 1).sum()}")
print(f"Rows with soil type 15: {(train['Soil_Type15'] == 1).sum() + (test['Soil_Type15'] == 1).sum()}")


## === cell 17
train.drop(["Soil_Type7", "Soil_Type15"], axis=1, inplace=True)
test.drop(["Soil_Type7", "Soil_Type15"], axis=1, inplace=True)
features.remove("Soil_Type7")
features.remove("Soil_Type15")


## === cell 18
print("Numerical features with the least amount of unique values:")
train[num_features].nunique().sort_values().head(5)


## === cell 19
train.drop(train[train[target]==5].index, axis=0, inplace=True)
train.reset_index(drop=True, inplace=True)
label_enc = LabelEncoder()
NUM_CLASSES = train[target].nunique()


## === cell 20
s_scaler = StandardScaler()
for col in num_features:
    train[col] = s_scaler.fit_transform(np.array(train[col]).reshape(-1,1))
    test[col] = s_scaler.transform(np.array(test[col]).reshape(-1,1))


## === cell 21
X_nn = train[features].copy()
X_test_nn = test[features].copy()
y = pd.Series(label_enc.fit_transform(train[target]))



## === cell 22
mm_scaler = MinMaxScaler()
for col in X_nn.columns:
    X_nn[col] = mm_scaler.fit_transform(np.array(X_nn[col]).reshape(-1,1))
    X_test_nn[col] = mm_scaler.transform(np.array(X_test_nn[col]).reshape(-1,1))
    
X_test_nn = torch.tensor(X_test_nn.to_numpy()).float()


## === cell 23
BATCH_SIZE = 4096


## === cell 24
def prepare_datasets(X_nn, X_valid_nn, y_nn, y_valid_nn, batch_size=BATCH_SIZE):
    X_nn = torch.tensor(X_nn.to_numpy(), dtype=torch.float32)
    y_nn = torch.tensor(y_nn.to_numpy(), dtype=torch.long)
    X_valid_nn = torch.tensor(X_valid_nn.to_numpy(), dtype=torch.float32)
    y_valid_nn = torch.tensor(y_valid_nn.to_numpy(), dtype=torch.long)
    
    print("Using these datasets:")
    train_ds = TensorDataset(X_nn, y_nn)
    valid_ds = TensorDataset(X_valid_nn, y_valid_nn)
    print(f"Train_ds elements: {len(train_ds)}")
    print(f"Valid_ds elements: {len(valid_ds)}")

    
    train_ds = DataLoader(train_ds, batch_size, drop_last=False, num_workers=4)
    valid_ds = DataLoader(valid_ds, batch_size, drop_last=False, num_workers=4)

    
    for data, label in train_ds:
        print(f"Train_ds batch: {data.shape}, {label.shape}")
        break
    for data, label in valid_ds:
        print(f"Valid_ds batch: {data.shape}, {label.shape}")
        break
    return train_ds, valid_ds


## === cell 25
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
class_weights


## === cell 26
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data)


## === cell 27
class ParamsTracker(pl.callbacks.Callback):

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []
        self.lr_epoch_start = []

    
    def on_train_epoch_start(self, trainer, module):
        current_learning_rate = trainer.optimizers[0].state_dict()["param_groups"][0]["lr"]
        self.lr_epoch_start.append(current_learning_rate)
        
    def on_validation_epoch_end(self, trainer, module):
        metrics_logs = trainer.logged_metrics
        self.val_loss.append(metrics_logs["val_loss"].item())
        self.val_acc.append(metrics_logs["val_acc"].item())
    
    def on_train_epoch_end(self, trainer, module):
        metrics_logs = trainer.logged_metrics
        self.train_loss.append(metrics_logs["loss_epoch"].item())
        self.train_acc.append(metrics_logs["train_acc"].item())
        
        if self.verbose == True:
            print(f"Epoch {module.current_epoch} start learning rate: {self.lr_epoch_start[-1]:.6f}, "
                  f"train_loss: {self.train_loss[-1]:.4f}, "
                  f"train_acc: {self.train_acc[-1]:.4f}, "
                  f"val_loss: {self.val_loss[-1]:.4f}, "
                  f"val_acc: {self.val_acc[-1]:.4f}")  


## === cell 28
class Model(pl.LightningModule):
    def __init__(self, input_shape):
        super().__init__()
        
        self.input = nn.Linear(input_shape,128)
        self.hidden1 = nn.Linear(128,64)
        self.hidden2 = nn.Linear(64,32)
        self.output = nn.Linear(32,6)
        
        self.dr = 0.2
        self.activation = F.relu
        self.softmax = nn.Softmax(dim=1)
        self.train_acc_metric = torchmetrics.Accuracy(num_classes=6, average="micro")
        self.val_acc_metric = torchmetrics.Accuracy(num_classes=6, average="micro")
        self.loss = nn.CrossEntropyLoss()
        
        self.flag = False
    
    def forward(self, x):
        x = self.activation(self.input(x))
        x = F.dropout(x,p=self.dr,training=self.training)
        x = self.activation(self.hidden1(x))
        x = F.dropout(x,p=self.dr,training=self.training)
        x = self.activation(self.hidden2(x))
        x = F.dropout(x,p=self.dr,training=self.training)
        x = self.output(x)
        return x
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X).squeeze(1)
        loss = self.loss(y_hat, y)
        self.train_acc_metric(torch.argmax(y_hat, dim=1), y)
        self.log('loss', loss, prog_bar=True, on_epoch=True, logger=True)
        return {'loss': loss,}

    def training_epoch_end(self, outputs):
        train_acc = self.train_acc_metric.compute()
        self.log('train_acc', train_acc, prog_bar=True, on_epoch=True, on_step=False, logger=True)
        self.train_acc_metric.reset()



    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X).squeeze(1)
        val_loss = self.loss(y_hat, y)
        self.val_acc_metric(torch.argmax(y_hat, dim=1), y)
        self.log('val_loss',val_loss, prog_bar=True, on_epoch=True, logger=True)
        return {'val_loss': val_loss}

    def validation_epoch_end(self, outputs):
        val_acc = self.val_acc_metric.compute()
        self.log('val_acc', val_acc, prog_bar=True, on_epoch=True, on_step=False, logger=True)
        self.val_acc_metric.reset()
        return {'val_acc': val_acc}

       
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, eps=1e-8, weight_decay=1e-2, amsgrad=False)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                                                  patience=7, min_lr=1e-04, eps=1e-08,
                                                                  verbose=False, threshold=0.005, threshold_mode="abs")
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": "val_acc"}


## === cell 29
def train_ann(train_ds, valid_ds, Model=Model, input_shape=X_nn.shape[1]):
    
    model = Model(input_shape)
    model.apply(initialize_weights)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="models",
        filename=f'model_' + '{val_acc:.4}',
        monitor='val_acc',
        mode='max',
        save_weights_only=True)

    early_stop_callback = EarlyStopping(
        monitor='val_acc',
        min_delta=0.0002,
        patience=20,
        verbose=False,
        mode='max'
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
     )

    trainer.fit(model, train_ds, valid_ds)

    model.eval()

    best_model_path = checkpoint_callback.best_model_path
    
    return best_model_path, params_tracker_callback


## === cell 30
class Model(pl.LightningModule):
    def __init__(self, input_shape):
        super().__init__()

        self.input = nn.Linear(input_shape, 128)
        self.hidden1 = nn.Linear(128, 64)
        self.hidden2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 6)

        self.dr = 0.2
        self.activation = F.relu
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
        self.log("loss", loss, prog_bar=True, on_epoch=True, logger=True)
        return {
            "loss": loss,
        }

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


## === cell 31
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
        monitor="val_acc", min_delta=0.0002, patience=20, verbose=False, mode="max"
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
    )

    trainer.fit(model, train_ds, valid_ds)

    model.eval()

    best_model_path = checkpoint_callback.best_model_path

    return best_model_path, params_tracker_callback


## === cell 32
if all(name in globals() for name in ["y_train", "X_train", "model"]):
    accuracy_score(
        y_train,
        np.argmax(
            model(torch.tensor(X_train.to_numpy()).float()).detach().numpy(),
            axis=1,
        ),
    )


## === cell 33
if all(name in globals() for name in ["y_valid", "X_valid", "model"]):
    accuracy_score(
        y_valid,
        np.argmax(
            model(torch.tensor(X_valid.to_numpy()).float()).detach().numpy(),
            axis=1,
        ),
    )


## === cell 34
class Model(pl.LightningModule):
    def __init__(self, input_shape):
        super().__init__()

        self.input = nn.Linear(input_shape, 128)
        self.hidden1 = nn.Linear(128, 64)
        self.hidden2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 6)

        self.dr = 0.2
        self.activation = F.relu
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


if "model" not in globals():
    if (
        "best_model_path" in globals()
        and isinstance(best_model_path, str)
        and len(best_model_path) > 0
    ):
        model = Model(X_nn.shape[1])
        state_dict = torch.load(best_model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
    else:
        if not all(
            name in globals() for name in ["X_train", "X_valid", "y_train", "y_valid"]
        ):
            X_train, X_valid, y_train, y_valid = train_test_split(
                X_nn, y, test_size=0.2, random_state=42, stratify=y
            )

        train_ds, valid_ds = prepare_datasets(
            X_train, X_valid, y_train, y_valid, batch_size=BATCH_SIZE
        )
        best_model_path, params_tracker_callback = train_ann(train_ds, valid_ds)

        model = Model(X_nn.shape[1])
        state_dict = torch.load(best_model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()

predictions = pd.DataFrame()
predictions["Id"] = test["Id"]
predictions["Cover_Type"] = label_enc.inverse_transform(
    np.argmax(model(torch.tensor(X_test_nn).float()).detach().numpy(), axis=1)
)

predictions.to_csv("submission.csv", index=False, header=predictions.columns)
predictions.head()


## --- ERROR in cell 34, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mNotImplementedError[0m                       Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2854439082.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m    113[0m             [0mX_train[0m[0;34m,[0m [0mX_valid[0m[0;34m,[0m [0my_train[0m[0;34m,[0m [0my_valid[0m[0;34m,[0m [0mbatch_size[0m[0;34m=[0m[0mBATCH_SIZE[0m[0;34m[0m[0;34m[0m[0m
[1;32m    114[0m         )
[0;32m--> 115[0;31m         [0mbest_model_path[0m[0;34m,[0m [0mparams_tracker_callback[0m [0;34m=[0m [0mtrain_ann[0m[0;34m([0m[0mtrain_ds[0m[0;34m,[0m [0mvalid_ds[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    116[0m [0;34m[0m[0m
[1;32m    117[0m         [0mmodel[0m [0;34m=[0m [0mModel[0m[0;34m([0m[0mX_nn[0m[0;34m.[0m[0mshape[0m[0;34m[[0m[0;36m1[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/789031254.py[0m in [0;36mtrain_ann[0;34m(train_ds, valid_ds, Model, input_shape)[0m
[1;32m     34[0m     )
[1;32m     35[0m [0;34m[0m[0m
[0;32m---> 36[0;31m     [0mtrainer[0m[0;34m.[0m[0mfit[0m[0;34m([0m[0mmodel[0m[0;34m,[0m [0mtrain_ds[0m[0;34m,[0m [0mvalid_ds[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     37[0m [0;34m[0m[0m
[1;32m     38[0m     [0mmodel[0m[0;34m.[0m[0meval[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

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

## === cell 35
predictions["Cover_Type"].hist()
