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

0.95056

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.36817) has done: 'I fix the runtime error caused by the TorchMetrics API change by specifying `task="multiclass"` for `torchmetrics.Accuracy`, which unblocks training. I also fix inference/loading so the best checkpoint weights are loaded correctly and predictions are accumulated properly (the current code overwrites instead of summing). Finally, I ensure the submission is generated from the averaged test logits across folds (even if you still only run one fold as currently coded) and written to `submission.csv` with the required `Id,Cover_Type` columns.'

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

pl.seed_everything(42, workers=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")

pd.set_option("display.max_rows", 150)
pd.set_option("display.max_columns", 500)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.float_format", lambda x: "%.5f" % x)

import os




## === cell 1
train = pd.read_csv(
    "/kaggle/input/tabular-playground-series-dec-2021/train.csv",
    low_memory=False,
    engine="pyarrow",
)
test = pd.read_csv(
    "/kaggle/input/tabular-playground-series-dec-2021/test.csv",
    low_memory=False,
    engine="pyarrow",
)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1451578269.py in <cell line: 0>()
      1 # Speed: use pyarrow engine (when available) + explicit dtypes off by default to reduce parse overhead.
----> 2 train = pd.read_csv(
      3     "/kaggle/input/tabular-playground-series-dec-2021/train.csv",
      4     low_memory=False,
      5     engine="pyarrow",

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1605         self._currow = 0
   1606 
-> 1607         options = self._get_options_with_defaults(engine)
   1608         options["storage_options"] = kwds.get("storage_options", None)
   1609 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _get_options_with_defaults(self, engine)
   1658                         pass
   1659                     else:
-> 1660                         raise ValueError(
   1661                             f"The {repr(argname)} option is not supported with the "
   1662                             f"{repr(engine)} engine"

ValueError: The 'low_memory' option is not supported with the 'pyarrow' engine

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


train = reduce_mem_usage(train, verbose=True)
test = reduce_mem_usage(test, verbose=True)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3143200294.py in <cell line: 0>()
     34 
     35 
---> 36 train = reduce_mem_usage(train, verbose=True)
     37 test = reduce_mem_usage(test, verbose=True)
     38 

NameError: name 'train' is not defined

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



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4051802249.py in <cell line: 0>()
      1 target = "Cover_Type"
----> 2 features = list(train.columns[1:54])
      3 

NameError: name 'train' is not defined

## === cell 8
pass



## === cell 9
pass



## === cell 10
pass



## === cell 11
pass



## === cell 12
df = pd.concat([train[features], test[features]], axis=0, ignore_index=True)

unique_values = df[features].nunique() < 10
cat_features = unique_values[unique_values == True].index
unique_values = df[features].nunique() >= 10
num_features = unique_values[unique_values == True].index

print(f"There are {len(cat_features)} categorical features: {list(cat_features)}")
print(f"\nThere are {len(num_features)} continuous features: {list(num_features)}")



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2485558324.py in <cell line: 0>()
      1 # Speed: compute nunique on concatenated df is OK, but avoid repeated work; keep as in original logic.
----> 2 df = pd.concat([train[features], test[features]], axis=0, ignore_index=True)
      3 
      4 unique_values = df[features].nunique() < 10
      5 cat_features = unique_values[unique_values == True].index

NameError: name 'train' is not defined

## === cell 13
_ = (train.isna().sum().sum(), test.isna().sum().sum())
print("Total NA (train, test):", _)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3939643670.py in <cell line: 0>()
      1 # Speed: NA checks kept minimal.
----> 2 _ = (train.isna().sum().sum(), test.isna().sum().sum())
      3 print("Total NA (train, test):", _)
      4 

NameError: name 'train' is not defined

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



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/345369426.py in <cell line: 0>()
      1 print(
----> 2     f"Rows with soil type 7: {(train['Soil_Type7'] == 1).sum() + (test['Soil_Type7'] == 1).sum()}"
      3 )
      4 print(
      5     f"Rows with soil type 15: {(train['Soil_Type15'] == 1).sum() + (test['Soil_Type15'] == 1).sum()}"

NameError: name 'train' is not defined

## === cell 17
train.drop(["Soil_Type7", "Soil_Type15"], axis=1, inplace=True)
test.drop(["Soil_Type7", "Soil_Type15"], axis=1, inplace=True)
features.remove("Soil_Type7")
features.remove("Soil_Type15")



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2451430694.py in <cell line: 0>()
----> 1 train.drop(["Soil_Type7", "Soil_Type15"], axis=1, inplace=True)
      2 test.drop(["Soil_Type7", "Soil_Type15"], axis=1, inplace=True)
      3 features.remove("Soil_Type7")
      4 features.remove("Soil_Type15")
      5 

NameError: name 'train' is not defined

## === cell 18
print("Numerical features with the least amount of unique values:")
print(train[num_features].nunique().sort_values().head(5))



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3353518377.py in <cell line: 0>()
      1 # Speed: keep as original; small print only.
      2 print("Numerical features with the least amount of unique values:")
----> 3 print(train[num_features].nunique().sort_values().head(5))
      4 

NameError: name 'train' is not defined

## === cell 19
train.drop(train[train[target] == 5].index, axis=0, inplace=True)
train.reset_index(drop=True, inplace=True)
label_enc = LabelEncoder()
NUM_CLASSES = train[target].nunique()



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1740087016.py in <cell line: 0>()
----> 1 train.drop(train[train[target] == 5].index, axis=0, inplace=True)
      2 train.reset_index(drop=True, inplace=True)
      3 label_enc = LabelEncoder()
      4 NUM_CLASSES = train[target].nunique()
      5 

NameError: name 'train' is not defined

## === cell 20
s_scaler = StandardScaler()
train_num = s_scaler.fit_transform(
    train[num_features].to_numpy(dtype=np.float32, copy=False)
)
test_num = s_scaler.transform(test[num_features].to_numpy(dtype=np.float32, copy=False))
train.loc[:, num_features] = train_num
test.loc[:, num_features] = test_num



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2826435374.py in <cell line: 0>()
      2 s_scaler = StandardScaler()
      3 train_num = s_scaler.fit_transform(
----> 4     train[num_features].to_numpy(dtype=np.float32, copy=False)
      5 )
      6 test_num = s_scaler.transform(test[num_features].to_numpy(dtype=np.float32, copy=False))

NameError: name 'train' is not defined

## === cell 21
X_nn = train[features].copy()
X_test_nn = test[features].copy()
y = pd.Series(label_enc.fit_transform(train[target]))



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2197971389.py in <cell line: 0>()
----> 1 X_nn = train[features].copy()
      2 X_test_nn = test[features].copy()
      3 y = pd.Series(label_enc.fit_transform(train[target]))
      4 

NameError: name 'train' is not defined

## === cell 22
mm_scaler = MinMaxScaler()
X_nn_np = mm_scaler.fit_transform(X_nn.to_numpy(dtype=np.float32, copy=False))
X_test_np = mm_scaler.transform(X_test_nn.to_numpy(dtype=np.float32, copy=False))

X_test_nn = torch.from_numpy(X_test_np).float()
y_np = y.to_numpy(dtype=np.int64, copy=False)



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/151241047.py in <cell line: 0>()
      2 # because MinMaxScaler scales each feature independently even when fit on multi-column data).
      3 mm_scaler = MinMaxScaler()
----> 4 X_nn_np = mm_scaler.fit_transform(X_nn.to_numpy(dtype=np.float32, copy=False))
      5 X_test_np = mm_scaler.transform(X_test_nn.to_numpy(dtype=np.float32, copy=False))
      6 

NameError: name 'X_nn' is not defined

## === cell 23
BATCH_SIZE = 4096




## === cell 24
def prepare_datasets(
    X_nn_np_train, X_nn_np_valid, y_nn_train, y_nn_valid, batch_size=BATCH_SIZE
):
    X_nn_t = torch.from_numpy(X_nn_np_train).float()
    y_nn_t = torch.from_numpy(y_nn_train).long()
    X_valid_t = torch.from_numpy(X_nn_np_valid).float()
    y_valid_t = torch.from_numpy(y_nn_valid).long()

    train_ds = TensorDataset(X_nn_t, y_nn_t)
    valid_ds = TensorDataset(X_valid_t, y_valid_t)

    pin = torch.cuda.is_available()
    num_workers = min(4, os.cpu_count() or 1)

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
class_weights = compute_class_weight(
    class_weight="balanced", classes=np.unique(y_np), y=y_np
)
class_weights




## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3994397121.py in <cell line: 0>()
      1 class_weights = compute_class_weight(
----> 2     class_weight="balanced", classes=np.unique(y_np), y=y_np
      3 )
      4 class_weights
      5 

NameError: name 'y_np' is not defined

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

        if self.verbose == True and len(self.val_loss) > 0 and len(self.val_acc) > 0:
            print(
                f"Epoch {module.current_epoch} start learning rate: {self.lr_epoch_start[-1]:.6f}, "
                f"train_loss: {self.train_loss[-1]:.4f}, "
                f"train_acc: {self.train_acc[-1]:.4f}, "
                f"val_loss: {self.val_loss[-1]:.4f}, "
                f"val_acc: {self.val_acc[-1]:.4f}"
            )




## === cell 28
class Model(pl.LightningModule):
    def __init__(self, input_shape):
        super().__init__()

        self.input = nn.Linear(input_shape, 128)
        self.hidden1 = nn.Linear(128, 64)
        self.hidden2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 6)

        self.dr = 0.2
        self.activation = F.relu

        self.train_acc_metric = torchmetrics.Accuracy(
            task="multiclass", num_classes=6, average="micro"
        )
        self.val_acc_metric = torchmetrics.Accuracy(
            task="multiclass", num_classes=6, average="micro"
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




## === cell 29
def train_ann(train_ds, valid_ds, Model=Model, input_shape=X_nn_np.shape[1]):

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
        enable_checkpointing=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=200,
        deterministic=True,
    )

    trainer.fit(model, train_ds, valid_ds)

    model.eval()
    best_model_path = checkpoint_callback.best_model_path
    return best_model_path, params_tracker_callback




## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2098340736.py in <cell line: 0>()
----> 1 def train_ann(train_ds, valid_ds, Model=Model, input_shape=X_nn_np.shape[1]):
      2 
      3     model = Model(input_shape)
      4     model.apply(initialize_weights)
      5 

NameError: name 'X_nn_np' is not defined

## === cell 30
import os

os.makedirs("models", exist_ok=True)

splits = 10
skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)

nn_oof_preds = np.zeros((X_nn_np.shape[0],), dtype=np.int64)
nn_test_logits = np.zeros((X_test_nn.shape[0], 6), dtype=np.float32)
total_mean_acc = 0.0

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

trained_folds = 0

for num, (train_idx, valid_idx) in enumerate(skf.split(X_nn_np, y_np)):
    if num > 0:
        break

    print(f"\n\n===Training with fold {num}")
    X_train_np, X_valid_np = X_nn_np[train_idx], X_nn_np[valid_idx]
    y_train_np, y_valid_np = y_np[train_idx], y_np[valid_idx]

    train_ds, valid_ds = prepare_datasets(
        X_train_np, X_valid_np, y_train_np, y_valid_np
    )

    best_model_path, tracked_values = train_ann(
        train_ds, valid_ds, Model, X_nn_np.shape[1]
    )

    model = Model(X_nn_np.shape[1])

    ckpt = torch.load(best_model_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    with torch.no_grad():
        valid_logits = (
            model(torch.from_numpy(X_valid_np).to(device=device, dtype=torch.float32))
            .cpu()
            .numpy()
        )
    preds = np.argmax(valid_logits, axis=1)

    fold_score = accuracy_score(y_valid_np, preds)
    print(f"\n===Fold {num} valid data accuracy score is {fold_score}")

    with torch.no_grad():
        test_logits = model(X_test_nn.to(device)).cpu().numpy()

    nn_oof_preds[valid_idx] = preds
    nn_test_logits += test_logits
    total_mean_acc += fold_score
    trained_folds += 1

nn_test_logits /= max(trained_folds, 1)
total_mean_acc /= max(trained_folds, 1)

print(f"Average accuracy score of all models is {total_mean_acc}")



## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1182270795.py in <cell line: 0>()
      6 skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
      7 
----> 8 nn_oof_preds = np.zeros((X_nn_np.shape[0],), dtype=np.int64)
      9 nn_test_logits = np.zeros((X_test_nn.shape[0], 6), dtype=np.float32)
     10 total_mean_acc = 0.0

NameError: name 'X_nn_np' is not defined

## === cell 31
np.argmax(valid_logits, axis=1)[:10]



## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1114036350.py in <cell line: 0>()
----> 1 np.argmax(valid_logits, axis=1)[:10]
      2 

NameError: name 'valid_logits' is not defined

## === cell 32
accuracy_score(
    y_train_np,
    np.argmax(
        model(torch.from_numpy(X_train_np).to(device=device, dtype=torch.float32))
        .detach()
        .cpu()
        .numpy(),
        axis=1,
    ),
)



## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4251161052.py in <cell line: 0>()
      1 accuracy_score(
----> 2     y_train_np,
      3     np.argmax(
      4         model(torch.from_numpy(X_train_np).to(device=device, dtype=torch.float32))
      5         .detach()

NameError: name 'y_train_np' is not defined

## === cell 33
accuracy_score(
    y_valid_np,
    np.argmax(
        model(torch.from_numpy(X_valid_np).to(device=device, dtype=torch.float32))
        .detach()
        .cpu()
        .numpy(),
        axis=1,
    ),
)



## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3506217087.py in <cell line: 0>()
      1 accuracy_score(
----> 2     y_valid_np,
      3     np.argmax(
      4         model(torch.from_numpy(X_valid_np).to(device=device, dtype=torch.float32))
      5         .detach()

NameError: name 'y_valid_np' is not defined

## === cell 34
test_pred_labels = np.argmax(nn_test_logits, axis=1)
predictions = pd.DataFrame(
    {
        "Id": test["Id"].values,
        "Cover_Type": label_enc.inverse_transform(test_pred_labels),
    }
)
predictions.to_csv("submission.csv", index=False)
predictions.head()



## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2367158964.py in <cell line: 0>()
----> 1 test_pred_labels = np.argmax(nn_test_logits, axis=1)
      2 predictions = pd.DataFrame(
      3     {
      4         "Id": test["Id"].values,
      5         "Cover_Type": label_enc.inverse_transform(test_pred_labels),

NameError: name 'nn_test_logits' is not defined

## === cell 35
pass
