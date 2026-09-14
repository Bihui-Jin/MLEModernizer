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

0.0324629077409772

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

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
import pandas as pd
import tqdm




## === cell 1
class Config:
    dog = 1
    cat = 0
    train_dir = "/kaggle/working/train"
    test_dir = "/kaggle/working/test"
    n_fold = 5
    num_workers = 2
    pin_memory = True
    batch_size = 64
    seed = 2025
    drop_last = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 2
    early_stopping = 3
    lr = 1e-4
    optimizer = torch.optim.AdamW
    warmup_epochs = 0
    criterion = nn.BCEWithLogitsLoss()
    size = (320, 320)


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
    torch.backends.cudnn.benchmark = False


seed_everything()




## === cell 3
class DC_Dataset(Dataset):
    def __init__(self, imgs):
        super().__init__()
        self.imgs = imgs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = self.imgs[index]
        if img is None:
            raise ValueError(f"Image at index {index} is None (failed to read).")
        x = torch.from_numpy(img).permute(2, 0, 1).to(torch.float32) / 255.0
        return x


class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, last_epoch=-1):
        self.warmup_epochs = max(int(warmup_epochs), 0)
        self.total_epochs = max(int(total_epochs), 1)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if self.warmup_epochs > 0 and step <= self.warmup_epochs:
            scale = step / float(self.warmup_epochs)
        else:
            t = min(
                max(step - self.warmup_epochs, 0),
                self.total_epochs - self.warmup_epochs,
            )
            T = max(self.total_epochs - self.warmup_epochs, 1)
            scale = 0.5 * (1.0 + math.cos(math.pi * (t / T)))
        return [base_lr * scale for base_lr in self.base_lrs]


import math


class DC_Model(pl.LightningModule):
    def __init__(self, model_name="efficientnetv2_rw_s", pretrained=True, num_batch=0):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, 1)

        self.num_batch = num_batch
        self.criterion = cfg.criterion
        self.model_name = model_name
        self.pretrained = pretrained
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x).squeeze(-1)

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
        pred = (torch.sigmoid(output) > 0.5).to(label.dtype)
        acc = (pred == label).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = cfg.optimizer(self.parameters(), lr=cfg.lr)
        scheduler = WarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=cfg.warmup_epochs * self.num_batch,
            total_epochs=cfg.epochs * self.num_batch + 1,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def collate(x):
    return torch.stack(x, dim=0)


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




## === cell 4
test_glob = os.path.join(cfg.test_dir, "test", "*.jpg")
test_paths = sorted(glob.glob(test_glob))

if len(test_paths) == 0:
    test_glob_alt = os.path.join(cfg.test_dir, "test", "test", "*.jpg")
    test_paths = sorted(glob.glob(test_glob_alt))

if len(test_paths) == 0:
    raise FileNotFoundError(
        f"No test images found. Tried: {test_glob} and nested test/test/*.jpg"
    )

test_images = []
image_ids = []
for path in test_paths:
    bgr = cv2.imread(path)
    if bgr is None:
        raise ValueError(f"cv2.imread failed for: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, cfg.size)
    test_images.append(rgb)
    image_ids.append(os.path.basename(path).split(".")[0])



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/328373524.py in <cell line: 0>()
     10 
     11 if len(test_paths) == 0:
---> 12     raise FileNotFoundError(
     13         f"No test images found. Tried: {test_glob} and nested test/test/*.jpg"
     14     )

FileNotFoundError: No test images found. Tried: /kaggle/working/test/test/*.jpg and nested test/test/*.jpg

## === cell 5
model_paths = glob.glob(
    "/kaggle/input/dogs-vs-cats-lightning/lightning_logs/version_*/checkpoints/*.ckpt"
)
if len(model_paths) == 0:
    raise FileNotFoundError(
        "No checkpoints found at /kaggle/input/dogs-vs-cats-lightning/..."
    )



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/526849760.py in <cell line: 0>()
      3 )
      4 if len(model_paths) == 0:
----> 5     raise FileNotFoundError(
      6         "No checkpoints found at /kaggle/input/dogs-vs-cats-lightning/..."
      7     )

FileNotFoundError: No checkpoints found at /kaggle/input/dogs-vs-cats-lightning/...

## === cell 6
outputs = []



## === cell 7
test_dataset = DC_Dataset(test_images)
test_loader = DataLoader(
    test_dataset,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=cfg.pin_memory,
    collate_fn=collate,
)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2697080552.py in <cell line: 0>()
----> 1 test_dataset = DC_Dataset(test_images)
      2 test_loader = DataLoader(
      3     test_dataset,
      4     batch_size=cfg.batch_size,
      5     shuffle=False,

NameError: name 'test_images' is not defined

## === cell 8
for model_path in model_paths:
    model = DC_Model.load_from_checkpoint(model_path, map_location=cfg.device)
    model.to(cfg.device)
    model.eval()
    outputs.append([])
    with torch.no_grad():
        for img in tqdm.tqdm(test_loader, desc=os.path.basename(model_path)):
            img = img.to(cfg.device, non_blocking=True)
            output = model(img)
            outputs[-1] += output.detach().cpu().tolist()



## === cell 9
outputs = torch.tensor(outputs)  # [n_models, n_test]
outputs = outputs.mean(dim=0)  # [n_test]
outputs = torch.sigmoid(outputs)  # probabilities of dog



## === cell 10
clip = 0.005
sub = pd.DataFrame(
    {"id": image_ids, "label": torch.clamp(outputs, min=clip, max=1 - clip).numpy()}
)
sub["id"] = pd.to_numeric(sub["id"])
sub = sub.sort_values("id").reset_index(drop=True)

out_path = "/kaggle/working/submission.csv"
sub.to_csv(out_path, index=False)
print(f"Wrote {out_path} with shape {sub.shape}")
sub.head()



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3237171123.py in <cell line: 0>()
      3 clip = 0.005
      4 sub = pd.DataFrame(
----> 5     {"id": image_ids, "label": torch.clamp(outputs, min=clip, max=1 - clip).numpy()}
      6 )
      7 sub["id"] = pd.to_numeric(sub["id"])

NameError: name 'image_ids' is not defined

## === cell 11
sub



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/270188417.py in <cell line: 0>()
----> 1 sub
      2 

NameError: name 'sub' is not defined

## === cell 12
import shutil

for p in ["/kaggle/working/train", "/kaggle/working/test"]:
    if os.path.isdir(p):
        shutil.rmtree(p)
