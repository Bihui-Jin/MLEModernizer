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

0.03249

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, zipfile, pathlib, shutil


def _unzip_if_needed(zip_path: str, out_dir: str, marker_dir: str):
    if os.path.isdir(marker_dir):
        return
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)


_unzip_if_needed(
    "/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip",
    "/kaggle/working",
    "/kaggle/working/train",
)
_unzip_if_needed(
    "/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip",
    "/kaggle/working",
    "/kaggle/working/test",
)



## === cell 1
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
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pprint




## === cell 2
class Config:
    dog = 1
    cat = 0
    train_dir = "/kaggle/working/train"
    test_dir = "/kaggle/working/test"
    n_fold = 5
    num_workers = 2  # safer default for Kaggle; avoid worker spawn issues
    pin_memory = True
    batch_size = 64
    seed = 2025
    drop_last = True
    device = "cuda"
    epochs = 2
    early_stopping = 3
    lr = 1e-5
    optimizer = torch.optim.AdamW
    warmup_epochs = 0
    criterion = nn.BCEWithLogitsLoss()
    size = (320, 320)


cfg = Config()


def seed_everything(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(cfg.seed)



## === cell 3
import math
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, last_epoch=-1):
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.total_epochs = max(1, int(total_epochs))
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if self.warmup_epochs > 0 and step <= self.warmup_epochs:
            warmup_factor = step / float(self.warmup_epochs)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        progress = (step - self.warmup_epochs) / float(
            max(1, self.total_epochs - self.warmup_epochs)
        )
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [base_lr * cosine for base_lr in self.base_lrs]




## === cell 4
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
    resized_image = cv2.resize(padded_image, (size))
    return resized_image


class DC_Dataset(Dataset):
    def __init__(self, paths, valid=False):
        super().__init__()
        self.paths = paths
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
        img_bgr = cv2.imread(self.paths[index])
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {self.paths[index]}")
        img = square_pad_and_resize(img_bgr, cfg.size)
        img = self.transform(image=img)["image"]
        return img


class DC_Model(pl.LightningModule):
    def __init__(
        self, model_name="convnext_small", pretrained=True, num_batch=0, fold=0
    ):
        super().__init__()
        if "vit" in model_name or "convnext" in model_name:
            self.model = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=1,
            )
        else:
            self.model = timm.create_model(model_name, pretrained=pretrained)
            self.model.classifier = nn.Linear(self.model.classifier.in_features, 1)

        self.fold = fold
        self.num_batch = num_batch
        self.criterion = cfg.criterion
        self.model_name = model_name
        self.pretrained = pretrained

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x).squeeze()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch
        logits = self(x)
        return logits.detach()

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
        return loss

    def test_step(self, batch, batch_idx):
        img, label = batch
        output = self(img)
        loss = self.criterion(output, label)
        pred = torch.sigmoid(output) > 0.5
        acc = (pred == label).float().mean()
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = cfg.optimizer(self.parameters(), lr=cfg.lr, weight_decay=0.1)
        scheduler = WarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=cfg.warmup_epochs * max(1, self.num_batch),
            total_epochs=cfg.epochs * max(1, self.num_batch) + 1,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }




## === cell 5
test_paths = sorted(
    glob.glob(os.path.join(cfg.test_dir, "**", "*.jpg"), recursive=True)
)
if len(test_paths) == 0:
    raise FileNotFoundError(
        f"No test images found under {cfg.test_dir}. Directory listing: {os.listdir(cfg.test_dir)}"
    )

image_ids = [os.path.basename(p).split(".")[0] for p in test_paths]

print("n_test_images:", len(test_paths), "example:", test_paths[0])



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/982001217.py in <cell line: 0>()
      6 if len(test_paths) == 0:
      7     raise FileNotFoundError(
----> 8         f"No test images found under {cfg.test_dir}. Directory listing: {os.listdir(cfg.test_dir)}"
      9     )
     10 

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/test'

## === cell 6
candidate_patterns = [
    "/kaggle/input/**/checkpoints/*.ckpt",
    "/kaggle/input/**/*.ckpt",
    "/kaggle/working/**/checkpoints/*.ckpt",
    "/kaggle/working/**/*.ckpt",
]
model_paths = []
for pat in candidate_patterns:
    model_paths.extend(glob.glob(pat, recursive=True))
model_paths = sorted(set(model_paths))

pprint.pprint(model_paths[:20])
print("n_checkpoints_found:", len(model_paths))

if len(model_paths) == 0:
    raise FileNotFoundError(
        "No .ckpt files found. Add your trained Lightning checkpoints as a Kaggle Dataset "
        "or place them under /kaggle/working, then rerun."
    )



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2774718464.py in <cell line: 0>()
     16 
     17 if len(model_paths) == 0:
---> 18     raise FileNotFoundError(
     19         "No .ckpt files found. Add your trained Lightning checkpoints as a Kaggle Dataset "
     20         "or place them under /kaggle/working, then rerun."

FileNotFoundError: No .ckpt files found. Add your trained Lightning checkpoints as a Kaggle Dataset or place them under /kaggle/working, then rerun.

## === cell 7
test_dataset = DC_Dataset(test_paths, valid=True)
test_loader = DataLoader(
    test_dataset,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=cfg.pin_memory,
)



## === cell 8
outputs_per_model = []

for model_path in model_paths:
    model = DC_Model.load_from_checkpoint(model_path)
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )
    preds = trainer.predict(model, test_loader)

    preds = [p.detach().cpu().flatten() for p in preds if p is not None]
    if len(preds) == 0:
        raise RuntimeError(f"Predict returned no outputs for checkpoint: {model_path}")
    logits = torch.cat(preds, dim=0)

    if logits.shape[0] != len(test_paths):
        raise RuntimeError(
            f"Prediction length mismatch for {model_path}: got {logits.shape[0]}, expected {len(test_paths)}"
        )

    outputs_per_model.append(logits)

logits_stack = torch.stack(outputs_per_model, dim=0)
probs = torch.sigmoid(logits_stack).mean(dim=0)

print("probs shape:", probs.shape, "min/max:", float(probs.min()), float(probs.max()))



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/2206601640.py in <cell line: 0>()
     28 
     29 # Shape: (n_models, n_images)
---> 30 logits_stack = torch.stack(outputs_per_model, dim=0)
     31 probs = torch.sigmoid(logits_stack).mean(dim=0)
     32 

RuntimeError: stack expects a non-empty TensorList

## === cell 9
out_dir = "/kaggle/working"
clips = [0.01, 0.005, 0.015, 0.0125, 0.0025, 0.0, 0.0075, 0.004]

submission = pd.DataFrame({"id": image_ids, "label": probs.numpy()})
submission["id"] = pd.to_numeric(submission["id"])
submission = submission.sort_values("id").reset_index(drop=True)
submission.to_csv(os.path.join(out_dir, "submission.csv"), index=False)

for clip in clips:
    clipped = torch.clamp(probs, min=clip, max=1 - clip).numpy()
    sub_clip = pd.DataFrame({"id": image_ids, "label": clipped})
    sub_clip["id"] = pd.to_numeric(sub_clip["id"])
    sub_clip = sub_clip.sort_values("id").reset_index(drop=True)
    sub_clip.to_csv(os.path.join(out_dir, f"submission-clip={clip}.csv"), index=False)

submission.head()



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/127094833.py in <cell line: 0>()
      5 
      6 # Always write a standard submission.csv (no clipping by default to avoid unnecessary score shift).
----> 7 submission = pd.DataFrame({"id": image_ids, "label": probs.numpy()})
      8 submission["id"] = pd.to_numeric(submission["id"])
      9 submission = submission.sort_values("id").reset_index(drop=True)

NameError: name 'image_ids' is not defined

## === cell 10
print(submission.shape)
print(submission.dtypes)
print(submission.head())



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/404179334.py in <cell line: 0>()
      1 # Display the main submission info
----> 2 print(submission.shape)
      3 print(submission.dtypes)
      4 print(submission.head())
      5 

NameError: name 'submission' is not defined

## === cell 11
import shutil

if os.path.isdir("/kaggle/working/train"):
    shutil.rmtree("/kaggle/working/train")
if os.path.isdir("/kaggle/working/test"):
    shutil.rmtree("/kaggle/working/test")
