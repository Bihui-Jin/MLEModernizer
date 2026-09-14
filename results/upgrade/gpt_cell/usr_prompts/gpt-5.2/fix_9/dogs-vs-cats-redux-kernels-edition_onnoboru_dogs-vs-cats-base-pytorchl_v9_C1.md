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

3.13

# 2. Installed packages

albumentations==2.0.8
geopandas==0.14.4
lightning-utilities==0.15.2
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
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
timm==1.0.19
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1
transformers==4.53.3

# 3. Data file paths

```
/
    kaggle/
        data/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
        input/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
        working/
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
```

-> data/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> working/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

# 4. Code solution

## === cell 0
import os
import gc
import re
import sys
import time
import copy
import random
import glob
import zipfile
import shutil

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

import transformers

from tqdm import tqdm
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


## === cell 1
class CFG:
    debug_one_epoch = True
    debug_one_fold = False
    only_infer = False
    num_workers = 16
    batch_size = 64
    num_epochs = 10
    lr = 1e-3
    early_stopping_round = 5
    warmup_prop = 0.1
    random_seed = 42
    n_splits = 5
    model_name = "resnet18"  # timm model name
    pretrained_path = None
    train_dir = None
    test_dir = None
    optimizer = torch.optim.AdamW
    criterion = nn.BCEWithLogitsLoss()
    scheduler = transformers.get_linear_schedule_with_warmup
    input_imgsize = 224
    data_dir = "../input/dogs-vs-cats-redux-kernels-edition/"
    kaggle_working_dir = "/kaggle/working/"


def seed_torch(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_torch(CFG.random_seed)

if CFG.debug_one_epoch:
    CFG.num_epochs = 1

print("KAGGLE_URL_BASE" in set(os.environ.keys()))



## === cell 2
submission = pd.read_csv(os.path.join(CFG.data_dir, "sample_submission.csv"))

if "KAGGLE_URL_BASE" in set(os.environ.keys()):
    kaggle_train_dir = os.path.join(CFG.kaggle_working_dir, "train")
    if not os.path.exists(kaggle_train_dir):
        shutil.unpack_archive(
            os.path.join(CFG.data_dir, "train.zip"), CFG.kaggle_working_dir
        )

    kaggle_test_dir = os.path.join(CFG.kaggle_working_dir, "test")
    if not os.path.exists(kaggle_test_dir):
        shutil.unpack_archive(
            os.path.join(CFG.data_dir, "test.zip"), CFG.kaggle_working_dir
        )

    CFG.data_dir = CFG.kaggle_working_dir

CFG.train_dir = os.path.join(CFG.data_dir, "train")
CFG.test_dir = os.path.join(CFG.data_dir, "test")

train_list = glob.glob(os.path.join(CFG.train_dir, "**", "*.jpg"), recursive=True)
test_list = glob.glob(os.path.join(CFG.test_dir, "**", "*.jpg"), recursive=True)

train_list = [p for p in train_list if os.path.isfile(p)]
test_list = [p for p in test_list if os.path.isfile(p)]

print(f"Found train images: {len(train_list)}")
print(f"Found test images:  {len(test_list)}")



## === cell 3
train_df = pd.DataFrame(train_list, columns=["path"])


def _infer_class_from_path(p):
    base = os.path.basename(p).lower()
    parent = os.path.basename(os.path.dirname(p)).lower()
    if parent in ("cat", "dog"):
        return parent
    return base.split(".")[0]


train_df["class_str"] = train_df["path"].apply(_infer_class_from_path)
train_df["class"] = train_df["class_str"].map({"dog": 1, "cat": 0}).astype(np.int64)

train_df = train_df[train_df["class"].isin([0, 1])].reset_index(drop=True)
train_df = train_df[["path", "class"]]

test_df = pd.DataFrame(test_list, columns=["path"])
test_df["class"] = -1

test_df["id"] = test_df["path"].apply(
    lambda x: int(os.path.splitext(os.path.basename(x))[0])
)
test_df = test_df.sort_values("id").reset_index(drop=True)

print(train_df.head())
print(test_df.head())



## === cell 4
train_transform = A.Compose(
    [
        A.Resize(CFG.input_imgsize, CFG.input_imgsize),
        A.HorizontalFlip(p=0.5),
        A.Normalize(),
        ToTensorV2(),
    ]
)
test_transform = A.Compose(
    [
        A.Resize(CFG.input_imgsize, CFG.input_imgsize),
        A.Normalize(),
        ToTensorV2(),
    ]
)




## === cell 5
class DogsCatsDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = Image.open(self.df.iloc[idx, 0]).convert("RGB")
        img = self.transform(image=np.array(img))["image"]

        label = np.float32(self.df.iloc[idx, 1])
        return img, label




## === cell 6
class DogCatModel(nn.Module):
    def __init__(self):
        super(DogCatModel, self).__init__()
        self.model = timm.create_model(CFG.model_name, pretrained=True, num_classes=1)

    def forward(self, x):
        return self.model(x)




## === cell 7
class dog_vs_cats_pl_model(pl.LightningModule):
    def __init__(self, model):
        super(dog_vs_cats_pl_model, self).__init__()
        self.model = model
        self.criterion = CFG.criterion

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        img, label = batch
        output = self(img)
        loss = self.criterion(output.squeeze(-1), label)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "lr",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        img, label = batch
        output = self(img)
        loss = self.criterion(output.squeeze(-1), label)
        self.log(
            "valid_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def predict_step(self, batch, batch_idx):
        img, _ = batch
        output = self(img)
        output = torch.sigmoid(output).cpu().numpy()
        return output

    def configure_optimizers(self):
        optimizer = CFG.optimizer(self.parameters(), lr=CFG.lr)
        num_training_steps = len(self.train_dataloader) * CFG.num_epochs
        num_warmup_steps = int(num_training_steps * CFG.warmup_prop)
        scheduler = {
            "scheduler": transformers.get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            ),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler]




## === cell 8
def run_train_cv_pl(train, test):
    def _flatten_predict_outputs(pred_out):
        if pred_out is None:
            return []
        if isinstance(pred_out, (np.ndarray, torch.Tensor)):
            return [pred_out]
        flat = []
        if isinstance(pred_out, (list, tuple)):
            for x in pred_out:
                flat.extend(_flatten_predict_outputs(x))
            return flat
        return [pred_out]

    def _to_numpy_2d(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x)
        if x.ndim == 0:
            x = x.reshape(1, 1)
        elif x.ndim == 1:
            x = x.reshape(-1, 1)
        return x

    def _predict_to_array(trainer, model, loader):
        pred_out = trainer.predict(model, loader)
        flat = _flatten_predict_outputs(pred_out)
        flat_np = [_to_numpy_2d(p) for p in flat]
        if len(flat_np) == 0:
            return np.zeros((0, 1), dtype=np.float32)
        return np.concatenate(flat_np, axis=0)

    kf = KFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.random_seed)
    oof = np.zeros((len(train), 1), dtype=np.float32)
    predictions = []

    for fold, (train_idx, valid_idx) in enumerate(kf.split(train)):
        print(f"====================fold : {fold}====================")
        train_df_fold = train.iloc[train_idx].reset_index(drop=True)
        valid_df_fold = train.iloc[valid_idx].reset_index(drop=True)

        train_dataset = DogsCatsDataset(train_df_fold, transform=train_transform)
        valid_dataset = DogsCatsDataset(valid_df_fold, transform=test_transform)
        test_dataset = DogsCatsDataset(test_df, transform=test_transform)

        train_loader = DataLoader(
            train_dataset,
            batch_size=CFG.batch_size,
            shuffle=True,
            num_workers=CFG.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=CFG.batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=CFG.batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
            pin_memory=True,
        )

        model = DogCatModel()
        lightning_model = dog_vs_cats_pl_model(model)

        lightning_model.train_dataloader = train_loader
        lightning_model.valid_dataloader = valid_loader

        early_stopping = EarlyStopping(
            monitor="valid_loss",
            mode="min",
            patience=CFG.early_stopping_round,
        )
        checkpoint = ModelCheckpoint(
            monitor="valid_loss",
            mode="min",
            dirpath="checkpoints",
            filename=f"{CFG.model_name}_fold{fold}",
            save_top_k=1,
        )

        seed_everything(CFG.random_seed)

        logger = pl.loggers.TensorBoardLogger(
            "logs", name=f"{CFG.model_name}_fold{fold}"
        )

        accelerator = "gpu" if torch.cuda.is_available() else "cpu"

        trainer = pl.Trainer(
            max_epochs=CFG.num_epochs,
            accelerator=accelerator,
            devices=1,
            precision=16 if torch.cuda.is_available() else 32,
            logger=logger,
            callbacks=[early_stopping, checkpoint],
            enable_checkpointing=True,
        )
        trainer.fit(lightning_model, train_loader, valid_loader)

        valid_preds_arr = _predict_to_array(trainer, lightning_model, valid_loader)
        oof[valid_idx] = valid_preds_arr

        test_preds_arr = _predict_to_array(trainer, lightning_model, test_loader)
        predictions.append(test_preds_arr)

        del model, lightning_model, trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if CFG.debug_one_fold:
            break

    predictions = np.mean(predictions, axis=0)

    return {
        "oof": oof,
        "predictions": predictions,
    }




## === cell 9
class _NoOpTBLogger:
    def __init__(self, save_dir=".", name=None, version=None, *args, **kwargs):
        self.save_dir = save_dir
        self.name = name
        self.version = version

    @property
    def log_dir(self):
        return self.save_dir

    def log_metrics(self, metrics, step=None):
        return

    def log_hyperparams(self, params, *args, **kwargs):
        return

    def finalize(self, status):
        return


pl.loggers.TensorBoardLogger = _NoOpTBLogger

_original_trainer_cls = pl.Trainer


class _PatchedTrainer(_original_trainer_cls):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("logger", False)
        super().__init__(*args, **kwargs)


pl.Trainer = _PatchedTrainer

result = run_train_cv_pl(train_df, test_df)

try:
    oof_pred = np.clip(result["oof"].reshape(-1), 1e-7, 1 - 1e-7)
    oof_y = train_df["class"].values.astype(np.float32)
    print("OOF logloss:", log_loss(oof_y, oof_pred))
except Exception as e:
    print("OOF metric computation skipped due to:", repr(e))

test_pred = result["predictions"].reshape(-1)

test_pred = np.clip(test_pred, 1e-7, 1 - 1e-7)

sub = submission.copy()

sub = sub.sort_values("id").reset_index(drop=True)

pred_df = pd.DataFrame({"id": test_df["id"].values, "label": test_pred})
sub = sub[["id"]].merge(pred_df, on="id", how="left")

sub["label"] = sub["label"].fillna(0.5).astype(np.float32)

out_path = os.path.join(
    CFG.kaggle_working_dir if "KAGGLE_URL_BASE" in set(os.environ.keys()) else ".",
    "submission.csv",
)
sub.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(sub.head())
print("Rows:", len(sub), "Cols:", list(sub.columns))


## --- ERROR in cell 9, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2951416090.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     36[0m [0mpl[0m[0;34m.[0m[0mTrainer[0m [0;34m=[0m [0m_PatchedTrainer[0m[0;34m[0m[0;34m[0m[0m
[1;32m     37[0m [0;34m[0m[0m
[0;32m---> 38[0;31m [0mresult[0m [0;34m=[0m [0mrun_train_cv_pl[0m[0;34m([0m[0mtrain_df[0m[0;34m,[0m [0mtest_df[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     39[0m [0;34m[0m[0m
[1;32m     40[0m [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/3977864894.py[0m in [0;36mrun_train_cv_pl[0;34m(train, test)[0m
[1;32m    104[0m             [0menable_checkpointing[0m[0;34m=[0m[0;32mTrue[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    105[0m         )
[0;32m--> 106[0;31m         [0mtrainer[0m[0;34m.[0m[0mfit[0m[0;34m([0m[0mlightning_model[0m[0;34m,[0m [0mtrain_loader[0m[0;34m,[0m [0mvalid_loader[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    107[0m [0;34m[0m[0m
[1;32m    108[0m         [0mvalid_preds_arr[0m [0;34m=[0m [0m_predict_to_array[0m[0;34m([0m[0mtrainer[0m[0;34m,[0m [0mlightning_model[0m[0;34m,[0m [0mvalid_loader[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

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
[1;32m    992[0m             [0mcall[0m[0;34m.[0m[0m_call_lightning_module_hook[0m[0;34m([0m[0mself[0m[0;34m,[0m [0;34m"on_fit_start"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    993[0m [0;34m[0m[0m
[0;32m--> 994[0;31m         [0m_log_hyperparams[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    995[0m [0;34m[0m[0m
[1;32m    996[0m         [0;32mif[0m [0mself[0m[0;34m.[0m[0mstrategy[0m[0;34m.[0m[0mrestore_checkpoint_after_setup[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pytorch_lightning/loggers/utilities.py[0m in [0;36m_log_hyperparams[0;34m(trainer)[0m
[1;32m     99[0m         [0;32mif[0m [0mhparams_initial[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    100[0m             [0mlogger[0m[0;34m.[0m[0mlog_hyperparams[0m[0;34m([0m[0mhparams_initial[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 101[0;31m         [0mlogger[0m[0;34m.[0m[0mlog_graph[0m[0;34m([0m[0mpl_module[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    102[0m         [0mlogger[0m[0;34m.[0m[0msave[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mAttributeError[0m: '_NoOpTBLogger' object has no attribute 'log_graph'
