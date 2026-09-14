# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

# 5. Target score

0.0339717376907435

# 6. Current score

0.03795

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 0.03795) has done: 'I cleaned up the syntax errors, added a safe fallback model using torchvision when `timm` is unavailable, created a proper training dataset with labels, trained a small pretrained ResNet model for a few epochs, and ensured the script always writes a correctly‑formatted `submission.csv`. These changes let the notebook run end‑to‑end, produce a valid submission, and improve the log‑loss toward the target while preserving the original architecture intent.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchvision import models as tv_models
import cv2

try:
    import timm
except Exception:
    timm = None
from torch.optim.lr_scheduler import _LRScheduler
import math
import pandas as pd
import tqdm
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
import pprint




## === cell 1
class Config:
    dog = 1
    cat = 0
    train_dir = "/kaggle/working/dogs-vs-cats-redux-kernels-edition/train"
    test_dir = "/kaggle/working/dogs-vs-cats-redux-kernels-edition/test"
    n_fold = 5
    num_workers = 2
    pin_memory = True
    batch_size = 32
    seed = 2025
    drop_last = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 3
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
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything()




## === cell 3
def square_pad_and_resize(image, size):
    h, w, _ = image.shape
    max_dim = max(h, w)
    top = (max_dim - h) // 2
    bottom = max_dim - h - top
    left = (max_dim - w) // 2
    right = max_dim - w - left
    padded_image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    resized_image = cv2.resize(padded_image, size)
    return resized_image


class TrainDataset(Dataset):
    """Dataset for training/validation with labels derived from folder name."""

    def __init__(self, paths, labels, valid=False):
        self.paths = paths
        self.labels = labels
        self.valid = valid
        if not self.valid:
            self.transform = A.Compose(
                [
                    A.ShiftScaleRotate(
                        shift_limit=0.2, scale_limit=0.2, rotate_limit=15, p=0.5
                    ),
                    A.HorizontalFlip(p=0.5),
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
        img = cv2.imread(self.paths[index])
        if img is None:
            img = np.zeros((cfg.size[1], cfg.size[0], 3), dtype=np.uint8)
        img = cv2.resize(img, cfg.size)
        img = self.transform(image=img)["image"]
        if self.valid:
            return img
        else:
            label = torch.tensor(self.labels[index], dtype=torch.float32)
            return img, label


class TestDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.transform = A.Compose(
            [
                A.Normalize(),
                ToTensorV2(),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = cv2.imread(self.paths[index])
        if img is None:
            img = np.zeros((cfg.size[1], cfg.size[0], 3), dtype=np.uint8)
        img = cv2.resize(img, cfg.size)
        img = self.transform(image=img)["image"]
        return img


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return torch.mean(x.clamp(min=self.eps).pow(self.p), dim=(-2, -1)).pow(
            1.0 / self.p
        )


class DC_Model(pl.LightningModule):
    def __init__(self, model_name="resnet18", pretrained=True, num_batch=0, fold=0):
        super().__init__()
        if timm is not None:
            self.model = timm.create_model(
                model_name, pretrained=pretrained, num_classes=0, global_pool=""
            )
            num_features = self.model.num_features
        else:
            backbone = tv_models.resnet18(pretrained=pretrained)
            modules = list(backbone.children())[:-2]  # up to conv5
            self.model = nn.Sequential(*modules)
            num_features = backbone.fc.in_features

        self.pool = GeM()
        self.head = nn.Linear(num_features, 1)
        self.fold = fold
        self.num_batch = num_batch
        self.criterion = cfg.criterion
        self.save_hyperparameters()

    def forward(self, x):
        x = self.model(x)
        x = self.pool(x)
        x = self.head(x).squeeze()
        return x

    def training_step(self, batch, batch_idx):
        img, label = batch
        output = self(img)
        loss = self.criterion(output, label)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, label = batch
        output = self(img)
        loss = self.criterion(output, label)
        pred = torch.sigmoid(output) > 0.5
        acc = (pred == label).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = cfg.optimizer(self.parameters(), lr=cfg.lr, weight_decay=0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs * self.num_batch
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


def collate_fn(batch):
    if isinstance(batch[0], tuple):
        imgs, labels = zip(*batch)
        return torch.stack(imgs), torch.stack(labels)
    else:
        return torch.stack(batch)




## === cell 4
train_cat_paths = glob.glob(os.path.join(cfg.train_dir, "cat", "*.jpg"))
train_dog_paths = glob.glob(os.path.join(cfg.train_dir, "dog", "*.jpg"))
train_paths = train_cat_paths + train_dog_paths
train_labels = [cfg.cat] * len(train_cat_paths) + [cfg.dog] * len(train_dog_paths)

train_idx, val_idx = train_test_split(
    np.arange(len(train_paths)),
    test_size=0.2,
    stratify=train_labels,
    random_state=cfg.seed,
)

train_dataset = TrainDataset(
    [train_paths[i] for i in train_idx],
    [train_labels[i] for i in train_idx],
    valid=False,
)
val_dataset = TrainDataset(
    [train_paths[i] for i in val_idx], [train_labels[i] for i in val_idx], valid=False
)




## === cell 5
train_loader = DataLoader(
    train_dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=cfg.num_workers,
    pin_memory=cfg.pin_memory,
    collate_fn=collate_fn,
    drop_last=cfg.drop_last,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=cfg.pin_memory,
    collate_fn=collate_fn,
    drop_last=False,
)




## === cell 6
model = DC_Model(
    model_name="resnet18", pretrained=True, num_batch=len(train_loader), fold=0
)

checkpoint_callback = ModelCheckpoint(
    dirpath="/kaggle/working/checkpoints",
    filename="best-checkpoint",
    save_top_k=1,
    monitor="val_loss",
    mode="min",
)

trainer = Trainer(
    max_epochs=cfg.epochs,
    callbacks=[
        checkpoint_callback,
        EarlyStopping(monitor="val_loss", patience=cfg.early_stopping, mode="min"),
    ],
    logger=False,
    enable_progress_bar=True,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
)

trainer.fit(model, train_loader, val_loader)




## === cell 7
best_ckpt = checkpoint_callback.best_model_path
print(f"Best checkpoint: {best_ckpt}")
model = DC_Model.load_from_checkpoint(best_ckpt, map_location=cfg.device)
model.eval()
model.to(cfg.device)




## === cell 8
test_paths = glob.glob(os.path.join(cfg.test_dir, "**", "*.jpg"), recursive=True)
image_ids = [os.path.splitext(os.path.basename(p))[0] for p in test_paths]

test_dataset = TestDataset(test_paths)
test_loader = DataLoader(
    test_dataset,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=cfg.pin_memory,
    collate_fn=lambda x: torch.stack(x),
)




## === cell 9
outputs = []
with torch.no_grad():
    for img in tqdm.tqdm(test_loader):
        img = img.to(cfg.device)
        out = model(img)
        outputs.append(out.cpu())
outputs = torch.cat(outputs)
outputs = torch.sigmoid(outputs).numpy()  # probability of class 1 (dog)




## === cell 10
submission = pd.DataFrame(
    {"id": pd.to_numeric(image_ids, errors="coerce"), "label": outputs}
).sort_values("id")
submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")




## === cell 11
submission.head()
