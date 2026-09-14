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

0.01765

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

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

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




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
for col in num_features:
    train[col] = s_scaler.fit_transform(np.array(train[col]).reshape(-1, 1))
    test[col] = s_scaler.transform(np.array(test[col]).reshape(-1, 1))




## === cell 21
X_nn = train[features].copy()
X_test_nn = test[features].copy()
y = pd.Series(label_enc.fit_transform(train[target]))




## === cell 22
mm_scaler = MinMaxScaler()
for col in X_nn.columns:
    X_nn[col] = mm_scaler.fit_transform(np.array(X_nn[col]).reshape(-1, 1))
    X_test_nn[col] = mm_scaler.transform(np.array(X_test_nn[col]).reshape(-1, 1))

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
        self.val_acc.append(metrics_logs["val_acc"].item())

    def on_train_epoch_end(self, trainer, module):
        metrics_logs = trainer.logged_metrics
        self.train_loss.append(metrics_logs["loss_epoch"].item())
        self.train_acc.append(metrics_logs["train_acc"].item())

        if self.verbose == True:
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
        return {
            "loss": loss,
        }

    def on_train_epoch_end(self, outputs):
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

    def on_validation_epoch_end(self, outputs):
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
        logger=False,  # Disable TensorBoard logger to avoid import errors
        callbacks=[checkpoint_callback, early_stop_callback, params_tracker_callback],
    )

    trainer.fit(model, train_ds, valid_ds)

    model.eval()

    best_model_path = checkpoint_callback.best_model_path

    return best_model_path, params_tracker_callback




## === cell 29
splits = 10
skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)

nn_oof_preds = np.zeros((X_nn.shape[0],))
nn_test_preds = np.zeros((X_test_nn.shape[0],))
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

    test_pred_probs = model(X_test_nn).detach().numpy()
    nn_test_preds = test_pred_probs / splits

    nn_oof_preds[valid_idx] = preds
    total_mean_acc += fold_score / splits

print(f"Average accuracy score of all models is {total_mean_acc}")




## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/3802239586.py in <cell line: 0>()
     15     train_ds, valid_ds = prepare_datasets(X_train, X_valid, y_train, y_valid)
     16 
---> 17     best_model_path, tracked_values = train_ann(
     18         train_ds, valid_ds, Model, X_nn.shape[1]
     19     )

/tmp/ipykernel_55/18929307.py in train_ann(train_ds, valid_ds, Model, input_shape)
     31     )
     32 
---> 33     trainer.fit(model, train_ds, valid_ds)
     34 
     35     model.eval()

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
    151             try:
    152                 self.advance(data_fetcher)
--> 153                 self.on_advance_end(data_fetcher)
    154             except StopIteration:
    155                 break

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loops/training_epoch_loop.py in on_advance_end(self, data_fetcher)
    396                 call._call_lightning_module_hook(self.trainer, "on_validation_model_zero_grad")
    397 
--> 398             self.val_loop.run()
    399             self.trainer.training = True
    400             self.trainer._logger_connector._first_loop_iter = first_loop_iter

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

/tmp/ipykernel_55/1929815367.py in on_validation_epoch_end(self, trainer, module)
     18         metrics_logs = trainer.logged_metrics
     19         self.val_loss.append(metrics_logs["val_loss"].item())
---> 20         self.val_acc.append(metrics_logs["val_acc"].item())
     21 
     22     def on_train_epoch_end(self, trainer, module):

KeyError: 'val_acc'

## === cell 30
np.argmax(model(torch.tensor(X_valid.to_numpy()).float()).detach().numpy(), axis=1)[:10]




## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1589902521.py in <cell line: 0>()
----> 1 np.argmax(model(torch.tensor(X_valid.to_numpy()).float()).detach().numpy(), axis=1)[:10]
      2 
      3 

NameError: name 'model' is not defined

## === cell 31
accuracy_score(
    y_train,
    np.argmax(model(torch.tensor(X_train.to_numpy()).float()).detach().numpy(), axis=1),
)




## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2308745313.py in <cell line: 0>()
      1 accuracy_score(
      2     y_train,
----> 3     np.argmax(model(torch.tensor(X_train.to_numpy()).float()).detach().numpy(), axis=1),
      4 )
      5 

NameError: name 'model' is not defined

## === cell 32
accuracy_score(
    y_valid,
    np.argmax(model(torch.tensor(X_valid.to_numpy()).float()).detach().numpy(), axis=1),
)




## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/105046064.py in <cell line: 0>()
      1 accuracy_score(
      2     y_valid,
----> 3     np.argmax(model(torch.tensor(X_valid.to_numpy()).float()).detach().numpy(), axis=1),
      4 )
      5 

NameError: name 'model' is not defined

## === cell 33
predictions = pd.DataFrame()
predictions["Id"] = test["Id"]
pred_enc = np.argmax(nn_test_preds, axis=1)
predictions["Cover_Type"] = label_enc.inverse_transform(pred_enc.astype(int))

predictions.to_csv("submission.csv", index=False)
predictions.head()




## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
AxisError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1212268879.py in <cell line: 0>()
      2 predictions["Id"] = test["Id"]
      3 # Obtain class predictions from the averaged probability matrix
----> 4 pred_enc = np.argmax(nn_test_preds, axis=1)
      5 predictions["Cover_Type"] = label_enc.inverse_transform(pred_enc.astype(int))
      6 

/usr/local/lib/python3.11/dist-packages/numpy/core/fromnumeric.py in argmax(a, axis, out, keepdims)
   1227     """
   1228     kwds = {'keepdims': keepdims} if keepdims is not np._NoValue else {}
-> 1229     return _wrapfunc(a, 'argmax', axis=axis, out=out, **kwds)
   1230 
   1231 

/usr/local/lib/python3.11/dist-packages/numpy/core/fromnumeric.py in _wrapfunc(obj, method, *args, **kwds)
     57 
     58     try:
---> 59         return bound(*args, **kwds)
     60     except TypeError:
     61         # A TypeError occurs if the object does have such a method in its

AxisError: axis 1 is out of bounds for array of dimension 1

## === cell 34
predictions["Cover_Type"].hist()

## --- ERROR in cell 34, traceback:
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
