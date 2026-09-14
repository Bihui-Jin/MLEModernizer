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

No external packages required in the script and installed.

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
import glob
import numpy as np
import random
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import cv2
import timm
from torch.optim.lr_scheduler import _LRScheduler
import math
import pandas as pd
import tqdm
import pprint




## === cell 1
class Config:
    dog = 1
    cat = 0

    base_dir = "/kaggle/input/dogs-vs-cats-redux-kernels-edition"
    train_dir = os.path.join(base_dir, "train")  # has cat/ and dog/ subfolders
    test_dir = os.path.join(base_dir, "test")  # has nested test/unknown/ in this dump

    n_fold = 5
    num_workers = 2  # keep modest to avoid dataloader spawn overhead/timeouts in some environments
    pin_memory = True
    batch_size = 64
    seed = 2025
    drop_last = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 2
    early_stopping = 3
    lr = 1e-4
    optimizer = torch.optim.AdamW
    warmup_epochs = 0
    criterion = nn.BCEWithLogitsLoss()
    size = (224, 224)


cfg = Config()




## === cell 2
def seed_everything(seed=cfg.seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # deterministic inference


seed_everything()




## === cell 3
class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(
        self, optimizer, warmup_epochs, total_epochs, eta_min=0, last_epoch=-1
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            if self.warmup_epochs == 0:
                return list(self.base_lrs)
            warmup_lr = [
                base_lr * (self.last_epoch + 1) / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
            return warmup_lr
        else:
            denom = self.total_epochs - self.warmup_epochs
            if denom <= 0:
                return list(self.base_lrs)
            cos_anneal_lr = [
                self.eta_min
                + (base_lr - self.eta_min)
                * (
                    1
                    + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / denom)
                )
                / 2
                for base_lr in self.base_lrs
            ]
            return cos_anneal_lr


class DC_Dataset(Dataset):
    """
    Minimal extension to support training:
    - for train/valid: return (image, label_float)
    - for test: return (image, id_int)

    Bugfix: Albumentations RandomSizedCrop API changed; use RandomResizedCrop instead
    to preserve the same augmentation intent and unblock training.
    """

    def __init__(self, paths, valid=False, return_id=False, return_label=False):
        super().__init__()
        self.paths = paths
        self.valid = valid
        self.return_id = return_id
        self.return_label = return_label

        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        if not self.valid:
            self.transform = A.Compose(
                [
                    A.ShiftScaleRotate(
                        shift_limit=0.2, scale_limit=0.2, rotate_limit=180, p=0.5
                    ),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.5),
                    A.RGBShift(p=0.5),
                    A.RandomResizedCrop(
                        size=cfg.size,  # new API uses `size=(h,w)`
                        scale=(0.5, 1.0),  # approximate old min/max crop area intent
                        ratio=(0.75, 1.3333),
                        p=1.0,
                    ),
                    A.Normalize(),
                    ToTensorV2(),
                ]
            )
        else:
            self.transform = A.Compose(
                [
                    A.Normalize(),
                    ToTensorV2(),
                ]
            )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, cfg.size)
        img = self.transform(image=img)["image"]

        if self.return_label:
            parent = os.path.basename(os.path.dirname(path)).lower()
            label = 1.0 if parent == "dog" else 0.0
            return img, torch.tensor(label, dtype=torch.float32)

        if self.return_id:
            image_id = os.path.basename(path).split(".")[0]
            return img, int(image_id)

        return img


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return torch.mean(x.clamp(min=self.eps).pow(self.p), dim=(-1, -2)).pow(
            1.0 / self.p
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class Distill_Model(pl.LightningModule):
    """
    Keep core model logic; distillation remains optional.
    """

    def __init__(
        self,
        model_name="convnext_small",
        teacher_path=None,
        pretrained=True,
        num_batch=0,
        fold=0,
        distill_ratio=0.0,
    ):
        super().__init__()

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=1,
        )
        if not ("vit" in model_name or "mobile" in model_name):
            self.model.head = nn.Sequential(
                GeM(), nn.Linear(self.model.head.in_features, 1)
            )

        self.teacher = timm.create_model(
            "convnext_small",
            pretrained=pretrained,
            num_classes=1,
        )
        self.teacher.head = nn.Sequential(
            GeM(), nn.Linear(self.teacher.head.in_features, 1)
        )

        self.fold = fold
        self.num_batch = num_batch
        self.criterion = cfg.criterion
        self.model_name = model_name
        self.pretrained = pretrained
        self.distill_ratio = float(distill_ratio)

        self.save_hyperparameters(ignore=["teacher_path"])

    def forward(self, x):
        return self.model(x).squeeze(-1)

    def training_step(self, batch, batch_idx):
        img, label = batch
        output = self(img)

        if self.distill_ratio > 0:
            with torch.no_grad():
                t = torch.sigmoid(self.teacher(img).squeeze(-1))
            label = self.distill_ratio * t + label * (1 - self.distill_ratio)

        loss = self.criterion(output, label)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, label = batch
        output = self(img)
        loss = self.criterion(output, label)
        pred = (torch.sigmoid(output) > 0.5).float()
        acc = (pred == (label > 0.5).float()).float().mean()
        if torch.isnan(loss):
            return None
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = cfg.optimizer(self.model.parameters(), lr=cfg.lr)
        scheduler = WarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=cfg.warmup_epochs * max(self.num_batch, 1),
            total_epochs=cfg.epochs * max(self.num_batch, 1) + 1,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }




## === cell 4
candidate_patterns = [
    os.path.join(cfg.test_dir, "*.jpg"),
    os.path.join(cfg.test_dir, "unknown", "*.jpg"),
    os.path.join(cfg.test_dir, "test", "*.jpg"),
    os.path.join(cfg.test_dir, "test", "unknown", "*.jpg"),
    os.path.join(cfg.test_dir, "test", "test", "*.jpg"),
    os.path.join(cfg.test_dir, "test", "test", "unknown", "*.jpg"),
]
test_paths = []
for pat in candidate_patterns:
    test_paths = glob.glob(pat)
    if len(test_paths) > 0:
        break

if len(test_paths) == 0:
    raise FileNotFoundError(
        f"No test images found under {cfg.test_dir}. Tried: {candidate_patterns}"
    )

test_paths = sorted(test_paths, key=lambda p: int(os.path.basename(p).split(".")[0]))
print("Found test images:", len(test_paths), "example:", test_paths[0])



## === cell 5
model_paths = glob.glob(
    "/kaggle/input/dogs-vs-cats-distillation/lightning_logs/version_*/checkpoints/*.ckpt"
)
pprint.pprint(model_paths)

if len(model_paths) == 0:
    print(
        "WARNING: No checkpoints found at /kaggle/input/dogs-vs-cats-distillation/... . "
        "Will fine-tune a pretrained convnext_small on the provided train/ folder."
    )



## === cell 6
train_cat = sorted(glob.glob(os.path.join(cfg.train_dir, "cat", "*.jpg")))
train_dog = sorted(glob.glob(os.path.join(cfg.train_dir, "dog", "*.jpg")))
train_paths_all = train_cat + train_dog
if len(train_paths_all) == 0:
    raise FileNotFoundError(
        f"No training images found under {cfg.train_dir}/cat and /dog"
    )

rng = np.random.RandomState(cfg.seed)
idx = np.arange(len(train_paths_all))
rng.shuffle(idx)
split = int(0.9 * len(idx))
tr_idx, va_idx = idx[:split], idx[split:]
train_paths = [train_paths_all[i] for i in tr_idx]
valid_paths = [train_paths_all[i] for i in va_idx]
print("Train/Valid sizes:", len(train_paths), len(valid_paths))

train_dataset = DC_Dataset(train_paths, valid=False, return_label=True)
valid_dataset = DC_Dataset(valid_paths, valid=True, return_label=True)

train_loader = DataLoader(
    train_dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=cfg.num_workers,
    pin_memory=cfg.pin_memory and torch.cuda.is_available(),
    drop_last=cfg.drop_last,
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=cfg.pin_memory and torch.cuda.is_available(),
    drop_last=False,
)



## === cell 7
trained_ckpt_path = "/kaggle/working/fallback_finetuned.ckpt"
if len(model_paths) == 0:
    model = Distill_Model(pretrained=True, num_batch=len(train_loader))
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=True,
        deterministic=True,
        default_root_dir="/kaggle/working",
        log_every_n_steps=50,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    trainer.save_checkpoint(trained_ckpt_path)
    model_paths = [trained_ckpt_path]
    print("Saved fine-tuned checkpoint to:", trained_ckpt_path)



## === cell 8
outputs_per_model = []



## === cell 9
test_dataset = DC_Dataset(test_paths, valid=True, return_id=True)
test_loader = DataLoader(
    test_dataset,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=cfg.pin_memory and torch.cuda.is_available(),
    drop_last=False,
)



## === cell 10
id_order = []
models_to_run = model_paths if len(model_paths) > 0 else [None]

for model_path in models_to_run:
    if model_path is None:
        model = Distill_Model(pretrained=True)
        model_name_for_desc = "pretrained_convnext_small"
    else:
        model = Distill_Model.load_from_checkpoint(model_path)
        model_name_for_desc = os.path.basename(model_path)

    model.to(cfg.device)
    model.eval()

    logits = {}
    with torch.no_grad():
        for imgs, ids in tqdm.tqdm(
            test_loader, desc=f"Predicting {model_name_for_desc}"
        ):
            imgs = imgs.to(cfg.device, non_blocking=True)
            out = model(imgs).detach().float().cpu()
            ids = ids.detach().cpu().numpy().tolist()
            for i, image_id in enumerate(ids):
                logits[int(image_id)] = out[i].item()

    if not id_order:
        id_order = sorted(logits.keys())
    outputs_per_model.append([logits[i] for i in id_order])



## === cell 11
logits = torch.tensor(outputs_per_model, dtype=torch.float32).mean(dim=0)
probs = torch.sigmoid(logits)

clip = 0.005
probs = torch.clamp(probs, min=clip, max=1 - clip)



## === cell 12
sample_path = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv"
sample = pd.read_csv(sample_path)

pred_df = pd.DataFrame({"id": id_order, "label": probs.cpu().numpy().tolist()})
pred_df["id"] = pd.to_numeric(pred_df["id"], errors="coerce").astype("Int64")
pred_df = pred_df.dropna(subset=["id"]).copy()
pred_df["id"] = pred_df["id"].astype(int)
pred_df = pred_df.sort_values("id")

submission = sample[["id"]].merge(pred_df, on="id", how="left")
if submission["label"].isna().any():
    submission["label"] = submission["label"].fillna(0.5)

out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)
print(
    "Wrote:",
    out_path,
    "rows:",
    len(submission),
    "na_filled:",
    int(submission["label"].isna().sum()),
)



## === cell 13
submission.head()



## === cell 14
print("Done.")
