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

0.0311964116562683

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
I will adjust the script so it reliably produces a submission CSV.  
Key fixes:  
- Recursively collect all test images (the original glob missed nested folders).  
- Detect when no pretrained checkpoints are found; in that case compute the overall dog‑class frequency from the training filenames and use it as a constant prediction for every test image.  
- Ensure the submission file is written as `submission.csv` with the correct column names and sorted IDs.  

These changes keep the original model‑based pipeline untouched but add a safe fallback that guarantees a valid submission, moving the workflow toward the target score.

```python


## --- ERROR in cell 0, traceback:
  File "/tmp/ipykernel_55/525454206.py", line 4
    - Detect when no pretrained checkpoints are found; in that case compute the overall dog‑class frequency from the training filenames and use it as a constant prediction for every test image.
                                                                                           ^
SyntaxError: invalid character '‑' (U+2011)


## === cell 1
%%bash

if [ ! -d "/kaggle/working/train" ]; then
    unzip -q /kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip -d /kaggle/working
fi

if [ ! -d "/kaggle/working/test" ]; then
    unzip -q /kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip -d /kaggle/working
fi




## === cell 2
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




## === cell 3
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
    distill_ratio = 0.0

cfg = Config()




## === cell 4
def seed_everything(seed=cfg.seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything()




## === cell 5
class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            warmup_lr = [
                base_lr * (self.last_epoch + 1) / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
            return warmup_lr
        else:
            cos_anneal_lr = [
                self.eta_min + (base_lr - self.eta_min) * 
                (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / 
                              (self.total_epochs - self.warmup_epochs))) / 2
                for base_lr in self.base_lrs
            ]
            return cos_anneal_lr

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
            self.transform = A.Compose([
                A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=180, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.RGBShift(p=0.5),
                A.RandomSizedCrop(min_max_height=(cfg.size[0], cfg.size[0]//2), height=cfg.size[0], width=cfg.size[1]),
                A.Normalize(),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
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
        return torch.mean(x.clamp(min=self.eps).pow(self.p), dim=(-1,-2)).pow(1.0 / self.p)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"

class Distill_Model(pl.LightningModule):
    def __init__(self, model_name='convnext_small', teacher_path=None, pretrained=True, num_batch=0, fold=0):
        super().__init__()
        
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=1,
        )
        if not ('vit' in model_name or 'mobile' in model_name):
            self.model.head = nn.Sequential(
                GeM(),
                nn.Linear(self.model.head.in_features, 1)
            )
        self.teacher = timm.create_model(
            'convnext_small',
            pretrained=pretrained,
            num_classes=1,
        )
        self.teacher.head = nn.Sequential(
                GeM(),
                nn.Linear(self.teacher.head.in_features, 1)
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
        label = cfg.distill_ratio*torch.sigmoid(self.teacher(img).squeeze()) + label*(1-cfg.distill_ratio)
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
        optimizer = cfg.optimizer(self.model.parameters(), lr=cfg.lr)
        scheduler = WarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=cfg.warmup_epochs*self.num_batch,
            total_epochs=cfg.epochs*self.num_batch+1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }

def collate(x):
    return




## === cell 6
test_paths = glob.glob(os.path.join(cfg.test_dir, '**', '*.jpg'), recursive=True)
image_ids = [os.path.basename(p).split('.')[0] for p in test_paths]




## === cell 7
model_paths = glob.glob('/kaggle/input/dogs-vs-cats-distillation/lightning_logs/version_*/checkpoints/*.ckpt')
pprint.pprint(model_paths)

if len(model_paths) == 0:
    train_cat_paths = glob.glob(os.path.join(cfg.train_dir, 'cat', '*.jpg'))
    train_dog_paths = glob.glob(os.path.join(cfg.train_dir, 'dog', '*.jpg'))
    total = len(train_cat_paths) + len(train_dog_paths)
    dog_ratio = len(train_dog_paths) / total if total > 0 else 0.5
    fallback_pred = torch.full((len(test_paths),), dog_ratio, dtype=torch.float32)
    outputs = [fallback_pred]
else:
    outputs = []  # will be filled in the next cell




## === cell 8
pass




## === cell 9
test_dataset = DC_Dataset(test_paths, valid=True)
test_loader = DataLoader(
    test_dataset, batch_size=cfg.batch_size, shuffle=False,
    num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
)




## === cell 10
if len(model_paths) > 0:
    for model_path in model_paths:
        model = Distill_Model.load_from_checkpoint(model_path)
        model.eval()
        outputs.append([])
        with torch.no_grad():
            for img in tqdm.tqdm(test_loader):
                output = model(img.to(cfg.device))
                outputs[-1] += output.tolist()




## === cell 11
if isinstance(outputs, list) and len(outputs) > 0 and isinstance(outputs[0], list):
    outputs = torch.tensor(outputs)          # (models, samples)
    outputs = outputs.mean(dim=0)            # average across models
else:
    outputs = outputs[0] if isinstance(outputs, list) else outputs

outputs = torch.sigmoid(outputs)




## === cell 12
submission = pd.DataFrame({
    'id': pd.to_numeric(image_ids, errors='coerce'),
    'label': outputs.tolist()
})
submission = submission.sort_values('id')
submission_path = '/kaggle/working/submission.csv'
submission.to_csv(submission_path, index=False)
print(f'Submission written to {submission_path}')




## === cell 13
submission.head()




## === cell 14
!rm -r /kaggle/working/train
!rm -r /kaggle/working/test
```

## --- ERROR in cell 14, traceback:
  File "/tmp/ipykernel_55/3402149409.py", line 3
    ```
    ^
SyntaxError: invalid syntax


## --- ERROR in outputing the csv:
Invalid submission: Submission and answers have different id's
