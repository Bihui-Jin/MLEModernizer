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

0.0311842701756321

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
%%bash

if [ ! -d "/kaggle/working/train" ]; then
    unzip -q /kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip -d /kaggle/working
fi

if [ ! -d "/kaggle/working/test" ]; then
    unzip -q /kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip -d /kaggle/working
fi


## === cell 1
import os
import glob
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import cv2
import timm
from torch.optim.lr_scheduler import _LRScheduler
import math
import pandas as pd
import tqdm
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
import albumentations as A
from albumentations.pytorch import ToTensorV2   
from sklearn.model_selection import KFold
import pprint

## === cell 2
class Config:
    dog = 1
    cat = 0
    train_dir = '/kaggle/working/train'
    test_dir = '/kaggle/working/test'
    n_fold = 5
    num_workers = 16
    pin_memory = True
    batch_size = 64
    seed = 2025
    drop_last = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 2
    early_stopping = 3
    lr = 1e-4
    optimizer = torch.optim.AdamW
    warmup_epochs = 0
    criterion = nn.BCEWithLogitsLoss()
    size = (224, 224)

cfg = Config()

## === cell 3
def seed_everything(seed=cfg.seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything()

## === cell 4
def square_pad_and_resize(image, size):
    h, w, _ = image.shape
    
    max_dim = max(h, w)
    
    top = (max_dim - h) // 2
    bottom = max_dim - h - top
    left = (max_dim - w) // 2
    right = max_dim - w - left
    
    padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    resized_image = cv2.resize(padded_image, (size))
    return resized_image

class DC_Dataset(Dataset):
    def __init__(self, paths, valid=False):
        super().__init__()
        self.paths = paths
        self.valid = valid

        if not self.valid:
            self.transform = transform = A.Compose([
                A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=15, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Normalize(),
                ToTensorV2(),
            ])
        else:
            self.transform = transform = A.Compose([
                A.Normalize(),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = cv2.resize(cv2.imread(self.paths[index]), cfg.size)
        img = self.transform(image=img)['image']
        return img

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return torch.mean(x.clamp(min=self.eps).pow(self.p), dim=(-2, -1)).pow(1.0 / self.p)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"

class DC_Model(pl.LightningModule):
    def __init__(self, model_name='convnext_small', pretrained=True, num_batch=0, fold=0):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=1,  # ヘッドを後で定義
            global_pool=""  # カスタムプーリングを使用
        )

        num_features = self.model.num_features
        self.model.head = nn.Sequential(
            GeM(),
            nn.Linear(num_features, 1)
        )
        self.fold = fold
        self.num_batch = num_batch
        self.criterion = cfg.criterion
        self.model_name = model_name
        self.pretrained = pretrained

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x).squeeze()

    def training_step(self, batch, batch_idx):
        img, label = batch
        output = self(img)
        loss = self.criterion(output, label)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, label = batch
        output = self(img)
        loss = self.criterion(output, label)
        pred = torch.sigmoid(output) > 0.5
        acc = (pred == label).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        img, label = batch
        output = self(img)
        loss = self.criterion(output, label)
        pred = torch.sigmoid(output) > 0.5
        acc = (pred == label).float().mean()
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = cfg.optimizer(self.parameters(), lr=cfg.lr, weight_decay=0.1)
        scheduler = WarmupCosineAnnealingLR(optimizer, warmup_epochs=cfg.warmup_epochs*self.num_batch, total_epochs=cfg.epochs*self.num_batch+1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }
def collate(x):
    return x


## === cell 5
test_paths = glob.glob(cfg.test_dir + '/*')
image_ids = []

for path in test_paths:
    image_ids.append(os.path.basename(path).split('.')[0])

## === cell 6
model_paths = glob.glob('/kaggle/input/dogs-vs-cats-lightning/lightning_logs/version_*/checkpoints/*.ckpt')
pprint.pprint(model_paths)

## === cell 7
outputs = []

## === cell 8
test_dataset = DC_Dataset(test_paths, valid=True)
test_loader = DataLoader(
    test_dataset, batch_size=cfg.batch_size, shuffle=False,
    num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
)

## === cell 9
for model_path in model_paths:
    model = DC_Model.load_from_checkpoint(model_path)
    model.eval()
    outputs.append([])
    with torch.no_grad():
        for img in tqdm.tqdm(test_loader):
            output = model(img.to(cfg.device))
            outputs[-1] += output.tolist()


## === cell 10
outputs = torch.tensor(outputs)
outputs = outputs.mean(dim=0)
outputs = torch.sigmoid(outputs)

## === cell 11
for clip in [0.01, 0.005, 0.015, 0.0125, 0.0025, 0, 0.0075, 0.004]:
    submission = pd.DataFrame({'id':image_ids, 'label':torch.clamp(outputs, min=clip, max=1-clip).tolist()})
    submission['id'] = pd.to_numeric(submission['id'])
    submission = submission.sort_values(by='id')
    submission.to_csv(f'/kaggle/working/submission-clip={clip}.csv', index=False)

## === cell 12
submission

## === cell 13
!rm -r /kaggle/working/train
!rm -r /kaggle/working/test
