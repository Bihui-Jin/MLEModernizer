# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Given a dataset of images of dogs and cats, predict if an image is a dog or a cat.

## Metric
Log loss.

## Submission Format
For each image in the test set, you must submit a probability that image is a dog. The file should have a header and be in the following format:

```
id,label
1,0.5
2,0.5
3,0.5
...
```

## Dataset
The train folder contains 25,000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12,500 images, named according to a numeric id.

# 2. Python version

3.13

# 3. Installed packages

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

# 4. Data file paths

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

# 5. Code solution

## === cell 0
import os
import gc
import random
import glob
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
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACCELERATOR = "gpu" if torch.cuda.is_available() else "cpu"
print(device)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

torch.set_float32_matmul_precision("high" if torch.cuda.is_available() else "highest")

try:
    torch.set_num_threads(min(8, os.cpu_count() or 8))
    torch.set_num_interop_threads(1)
except Exception:
    pass

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True




## === cell 1
class CFG:
    debug_one_epoch = False
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
    model_name = "resnet18"
    pretrained_path = None
    train_dir = None
    test_dir = None
    optimizer = torch.optim.AdamW
    criterion = nn.BCEWithLogitsLoss()
    scheduler = transformers.get_linear_schedule_with_warmup
    input_imgsize = 224

    data_dir = "/kaggle/data/dogs-vs-cats-redux-kernels-edition/"
    kaggle_working_dir = "/kaggle/working/"

    use_torch_compile = True

    infer_batch_size = 256


def seed_torch(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_torch(CFG.random_seed)

if CFG.debug_one_epoch:
    CFG.num_epochs = 1

print("KAGGLE_URL_BASE" in set(os.environ.keys()))




## === cell 2
def _pick_existing_path(candidates):
    for p in candidates:
        if p is not None and os.path.exists(p):
            return p
    return None


CFG.data_dir = _pick_existing_path(
    [
        CFG.data_dir,
        "/kaggle/input/dogs-vs-cats-redux-kernels-edition/",
        "/kaggle/data/dogs-vs-cats-redux-kernels-edition/",
        "../input/dogs-vs-cats-redux-kernels-edition/",
    ]
)

if CFG.data_dir is None:
    raise FileNotFoundError(
        "Could not locate dataset directory for dogs-vs-cats-redux-kernels-edition."
    )

submission_path = os.path.join(CFG.data_dir, "sample_submission.csv")
if not os.path.exists(submission_path):
    submission_path = _pick_existing_path(
        [
            os.path.join("/kaggle/data", "sample_submission.csv"),
            os.path.join("/kaggle/input", "sample_submission.csv"),
            os.path.join(CFG.data_dir, "sample_submission.csv"),
        ]
    )
if submission_path is None or (not os.path.exists(submission_path)):
    raise FileNotFoundError("Could not find sample_submission.csv in known locations.")

sample_submission = pd.read_csv(submission_path)

train_zip = os.path.join(CFG.data_dir, "train.zip")
test_zip = os.path.join(CFG.data_dir, "test.zip")

work_train_dir = os.path.join(CFG.kaggle_working_dir, "train")
work_test_dir = os.path.join(CFG.kaggle_working_dir, "test")

if os.path.exists(train_zip) and not os.path.exists(work_train_dir):
    shutil.unpack_archive(train_zip, CFG.kaggle_working_dir)
if os.path.exists(test_zip) and not os.path.exists(work_test_dir):
    shutil.unpack_archive(test_zip, CFG.kaggle_working_dir)


def _find_train_root(base_dir: str) -> str:
    candidates = [
        os.path.join(base_dir, "train"),
        os.path.join(base_dir, "train", "train"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return base_dir


def _find_test_root(base_dir: str) -> str:
    candidates = [
        os.path.join(base_dir, "test"),
        os.path.join(base_dir, "test", "test"),
        os.path.join(base_dir, "test", "unknown"),
        os.path.join(base_dir, "test", "test", "unknown"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return base_dir


CFG.train_dir = _find_train_root(
    CFG.kaggle_working_dir if os.path.exists(work_train_dir) else CFG.data_dir
)
CFG.test_dir = _find_test_root(
    CFG.kaggle_working_dir if os.path.exists(work_test_dir) else CFG.data_dir
)


def _glob_train_images(train_root: str):
    cat_glob = glob.glob(os.path.join(train_root, "cat", "*.jpg"))
    dog_glob = glob.glob(os.path.join(train_root, "dog", "*.jpg"))
    if len(cat_glob) + len(dog_glob) > 0:
        return cat_glob + dog_glob
    return glob.glob(os.path.join(train_root, "*.jpg"))


def _glob_test_images(test_root: str):
    direct = glob.glob(os.path.join(test_root, "*.jpg"))
    if len(direct) > 0:
        return direct
    nested = glob.glob(os.path.join(test_root, "*", "*.jpg"))
    return nested


train_list = _glob_train_images(CFG.train_dir)
test_list = _glob_test_images(CFG.test_dir)

print("CFG.data_dir:", CFG.data_dir)
print("CFG.train_dir:", CFG.train_dir, "num_train_images:", len(train_list))
print("CFG.test_dir:", CFG.test_dir, "num_test_images:", len(test_list))

if len(train_list) == 0 or len(test_list) == 0:
    raise FileNotFoundError(
        f"Images not found. train_list={len(train_list)}, test_list={len(test_list)}. "
        f"Checked train_dir={CFG.train_dir}, test_dir={CFG.test_dir}"
    )




## === cell 3
train_df = pd.DataFrame({"path": train_list})
base_names = pd.Series(train_list, dtype="string").map(os.path.basename)

first_token = base_names.str.split(".", n=1, expand=True)[0]
parent_name = pd.Series(train_list, dtype="string").map(
    lambda p: os.path.basename(os.path.dirname(p))
)

class_name = first_token.where(first_token.isin(["cat", "dog"]), parent_name)
train_df["class_name"] = class_name

train_df["class"] = (
    train_df["class_name"].map({"dog": 1.0, "cat": 0.0}).astype(np.float32)
)
if train_df["class"].isna().any():
    bad = train_df[train_df["class"].isna()].head()
    raise ValueError(
        f"Found unlabeled train images (could not map to dog/cat). Example:\n{bad}"
    )

test_df = pd.DataFrame({"path": test_list})
test_base = pd.Series(test_list, dtype="string").map(os.path.basename)
test_df["class"] = -1.0
test_df["id"] = test_base.str.split(".", n=1, expand=True)[0].astype(int)
test_df = test_df.sort_values("id").reset_index(drop=True)

submission = pd.DataFrame({"id": test_df["id"].astype(int).values, "label": 0.5})
submission = submission.sort_values("id").reset_index(drop=True)

print(train_df.head())
print(test_df.head())
print(
    "submission rows:",
    len(submission),
    "min/max id:",
    submission["id"].min(),
    submission["id"].max(),
)




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
try:
    from torchvision.io import read_image, ImageReadMode

    _HAS_TVIO = True
except Exception:
    _HAS_TVIO = False


class DogsCatsDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.paths = self.df["path"].values
        if "class" in self.df.columns:
            self.labels = self.df["class"].values.astype(np.float32)
        else:
            self.labels = np.zeros((len(self.df),), dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        p = self.paths[idx]
        try:
            if _HAS_TVIO:
                img = read_image(p, mode=ImageReadMode.RGB).permute(1, 2, 0).numpy()
            else:
                img = np.array(Image.open(p).convert("RGB"))
        except Exception as e:
            raise FileNotFoundError(f"Failed to read image: {p} ({e})")

        if self.transform is not None:
            img = self.transform(image=img)["image"]
        else:
            img = ToTensorV2()(image=img)["image"]

        label = float(self.labels[idx])
        return img, label




## === cell 6
class DogCatModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(CFG.model_name, pretrained=True, num_classes=1)
        self._compiled = False

        if CFG.use_torch_compile:
            try:
                self.model = torch.compile(self.model, mode="max-autotune")
                self._compiled = True
            except Exception:
                self._compiled = False

    def forward(self, x):
        return self.model(x)




## === cell 7
class dog_vs_cats_pl_model(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.criterion = CFG.criterion
        self._num_training_steps = None
        self._num_warmup_steps = None

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        img, label = batch
        label = label.to(device=img.device, dtype=torch.float32)

        output = self(img)
        loss = self.criterion(output.squeeze(-1), label)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        opt = self.trainer.optimizers[0]
        self.log(
            "lr",
            opt.param_groups[0]["lr"],
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        img, label = batch
        label = label.to(device=img.device, dtype=torch.float32)

        output = self(img)
        loss = self.criterion(output.squeeze(-1), label)
        self.log(
            "valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def predict_step(self, batch, batch_idx):
        img, _ = batch
        output = self(img)
        output = torch.sigmoid(output).detach().cpu().numpy()
        return output

    def on_fit_start(self):
        self._num_training_steps = int(self.trainer.estimated_stepping_batches)
        self._num_warmup_steps = int(self._num_training_steps * CFG.warmup_prop)

    def configure_optimizers(self):
        optimizer = CFG.optimizer(self.parameters(), lr=CFG.lr)
        num_training_steps = self._num_training_steps or 1000
        num_warmup_steps = self._num_warmup_steps or int(
            num_training_steps * CFG.warmup_prop
        )

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
def _preds_to_1d(preds_list):
    """Lightning predict can return arrays of shape (bs,1) or (bs,). Normalize to (N,)."""
    arr = np.concatenate(preds_list, axis=0)
    arr = np.asarray(arr)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    return arr.astype(np.float32)


def _dl_kwargs():
    nw = int(CFG.num_workers)
    cpu = os.cpu_count() or 8
    nw = max(2, min(nw, cpu // 2))
    kwargs = dict(
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(nw > 0),
    )
    if nw > 0:
        kwargs["prefetch_factor"] = 4
    return kwargs


def run_train_cv_pl(train, test):
    if len(train) < CFG.n_splits:
        raise ValueError(
            f"n_splits={CFG.n_splits} > n_samples={len(train)}; check data loading."
        )

    skf = StratifiedKFold(
        n_splits=CFG.n_splits, shuffle=True, random_state=CFG.random_seed
    )
    oof = np.zeros((len(train),), dtype=np.float32)
    predictions = []

    test_dataset = DogsCatsDataset(test_df, transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=max(CFG.batch_size, CFG.infer_batch_size),
        shuffle=False,
        **_dl_kwargs(),
    )

    y = train["class"].values.astype(int)

    for fold, (train_idx, valid_idx) in enumerate(skf.split(train, y)):
        print(f"====================fold : {fold}====================")
        tr_df = train.iloc[train_idx].reset_index(drop=True)
        va_df = train.iloc[valid_idx].reset_index(drop=True)

        train_dataset = DogsCatsDataset(tr_df, transform=train_transform)
        valid_dataset = DogsCatsDataset(va_df, transform=test_transform)

        train_loader = DataLoader(
            train_dataset,
            batch_size=CFG.batch_size,
            shuffle=True,
            drop_last=True,
            **_dl_kwargs(),
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=max(CFG.batch_size, CFG.infer_batch_size),
            shuffle=False,
            **_dl_kwargs(),
        )

        model = DogCatModel()
        lightning_model = dog_vs_cats_pl_model(model)

        early_stopping = EarlyStopping(
            monitor="valid_loss", mode="min", patience=CFG.early_stopping_round
        )

        os.makedirs("checkpoints", exist_ok=True)
        checkpoint = ModelCheckpoint(
            monitor="valid_loss",
            mode="min",
            dirpath="checkpoints",
            filename=f"{CFG.model_name}_fold{fold}" + "-{epoch:02d}-{valid_loss:.4f}",
            save_top_k=1,
            save_last=False,
            save_weights_only=True,
        )

        seed_everything(CFG.random_seed, workers=True)

        trainer = pl.Trainer(
            max_epochs=CFG.num_epochs,
            accelerator=ACCELERATOR,
            devices=1,
            precision="16-mixed" if torch.cuda.is_available() else 32,
            logger=False,
            callbacks=[early_stopping, checkpoint],
            enable_checkpointing=True,
            log_every_n_steps=200,
            enable_progress_bar=False,
            num_sanity_val_steps=0,
            inference_mode=True,
            deterministic=True,
            enable_model_summary=False,
        )

        trainer.fit(lightning_model, train_loader, valid_loader)

        best_path = checkpoint.best_model_path
        if best_path and os.path.exists(best_path):
            best_model = dog_vs_cats_pl_model.load_from_checkpoint(
                best_path, model=DogCatModel()
            )
        else:
            best_model = lightning_model

        valid_preds_list = trainer.predict(
            best_model, dataloaders=valid_loader, return_predictions=True
        )
        valid_preds_arr = _preds_to_1d(valid_preds_list)
        oof[valid_idx] = valid_preds_arr

        test_preds_list = trainer.predict(
            best_model, dataloaders=test_loader, return_predictions=True
        )
        test_preds_arr = _preds_to_1d(test_preds_list)
        predictions.append(test_preds_arr)

        del (
            model,
            lightning_model,
            trainer,
            train_dataset,
            valid_dataset,
            train_loader,
            valid_loader,
            best_model,
            valid_preds_list,
            test_preds_list,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if CFG.debug_one_fold:
            break

    predictions = np.mean(np.stack(predictions, axis=0), axis=0)
    return {"oof": oof, "predictions": predictions}


def main():
    if CFG.only_infer:
        test_dataset = DogsCatsDataset(test_df, transform=test_transform)
        test_loader = DataLoader(
            test_dataset,
            batch_size=max(CFG.batch_size, CFG.infer_batch_size),
            shuffle=False,
            **_dl_kwargs(),
        )

        predictions = []
        infer_trainer = pl.Trainer(
            accelerator=ACCELERATOR,
            devices=1,
            precision="16-mixed" if torch.cuda.is_available() else 32,
            logger=False,
            enable_progress_bar=False,
            inference_mode=True,
            deterministic=True,
            enable_model_summary=False,
        )

        for fold in range(CFG.n_splits):
            pattern = os.path.join("checkpoints", f"{CFG.model_name}_fold{fold}-*.ckpt")
            ckpts = sorted(glob.glob(pattern))
            if len(ckpts) == 0:
                raise FileNotFoundError(
                    f"No checkpoint found for fold={fold}. Looked for: {pattern}"
                )
            ckpt_path = ckpts[-1]

            lightning_model = dog_vs_cats_pl_model.load_from_checkpoint(
                ckpt_path, model=DogCatModel()
            )

            preds_list = infer_trainer.predict(
                lightning_model, dataloaders=test_loader, return_predictions=True
            )
            preds = _preds_to_1d(preds_list)
            predictions.append(preds)

        preds_mean = np.mean(np.stack(predictions, axis=0), axis=0).reshape(-1)
        preds_mean = np.clip(preds_mean, 1e-6, 1 - 1e-6)

        sub = submission.copy().sort_values("id").reset_index(drop=True)
        if len(preds_mean) != len(sub):
            raise ValueError(
                f"Predictions length mismatch: preds={len(preds_mean)} vs submission={len(sub)}"
            )
        sub["label"] = preds_mean
        sub = sub[["id", "label"]]
        sub.to_csv("submission.csv", index=False)
        print(
            "Wrote submission.csv with rows:", len(sub), "columns:", list(sub.columns)
        )
    else:
        result = run_train_cv_pl(train_df, test_df)
        oof_preds = result["oof"].reshape(-1)
        predictions = result["predictions"].reshape(-1)

        predictions = np.clip(predictions, 1e-6, 1 - 1e-6)

        sub = submission.copy().sort_values("id").reset_index(drop=True)
        if len(predictions) != len(sub):
            raise ValueError(
                f"Predictions length mismatch: preds={len(predictions)} vs submission={len(sub)}"
            )

        if not np.array_equal(sub["id"].values, test_df.sort_values("id")["id"].values):
            raise ValueError("Submission ids are not aligned with sorted test_df ids.")

        sub["label"] = predictions
        sub = sub[["id", "label"]]
        sub.to_csv("submission.csv", index=False)

        train_df_out = train_df.copy()
        train_df_out["oof_preds"] = oof_preds
        train_df_out.to_csv("oof_preds.csv", index=False)

        if CFG.debug_one_fold is False:
            oof_clip = np.clip(oof_preds, 1e-6, 1 - 1e-6)
            print(f"oof log loss : {log_loss(train_df_out['class'].values, oof_clip)}")

        print(
            "Wrote submission.csv with rows:", len(sub), "columns:", list(sub.columns)
        )


if __name__ == "__main__":
    main()
