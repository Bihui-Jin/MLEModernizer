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
Determine which of the images have hidden messages embedded using one of three steganography algorithms (JMiPOD, JUNIWARD, UERD).

## Metric
Weighted AUC. Each region of the ROC curve is weighted according to these chosen parameters:

```
tpr_thresholds = [0.0, 0.4, 1.0]
weights = [2, 1]
```

In other words, the area between the true positive rate of 0 and 0.4 is weighted 2X, the area between 0.4 and 1 is now weighed (1X). The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.

## Submission Format
For each `Id` (image) in the test set, you must provide a score that indicates how likely this image contains hidden data: the higher the score, the more it is assumed that image contains secret data. The file should contain a header and have the following format:

```
Id,Label
0001.jpg,0.1
0002.jpg,0.99
0003.jpg,1.2
0004.jpg,-2.2
etc.
```
## Dataset
The only available information on the test set is:

1. Each embedding algorithm is used with the same probability.
2. The payload (message length) is adjusted such that the "difficulty" is approximately the same regardless the content of the image. Images with smooth content are used to hide shorter messages while highly textured images will be used to hide more secret bits. The payload is adjusted in the same manner for testing and training sets.
3. The average message length is 0.4 bit per non-zero AC DCT coefficient.
4. The images are all compressed with one of the three following JPEG quality factors: 95, 90 or 75.

### Files
- `Cover/` contains 75k unaltered images meant for use in training.
- `JMiPOD/` contains 75k examples of the JMiPOD algorithm applied to the cover images.
- `JUNIWARD/`contains 75k examples of the JUNIWARD algorithm applied to the cover images.
- `UERD/` contains 75k examples of the UERD algorithm applied to the cover images.
- `Test/` contains 5k test set images. These are the images for which you are predicting.
- `sample_submission.csv` contains an example submission in the correct format.

# 2. Python version

3.13

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-image==0.25.2
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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

# 4. Data file paths

```
/
    kaggle/
        data/
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        input/
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        working/
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
```

-> data/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> data/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> working/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

# 5. Target score

0.8876821205467409

# 6. Current score

0.57913

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.59505) has done: 'I remove the unavailable `torchsampler` import and replace its usage with a `WeightedRandomSampler`, which resolves the import error and the subsequent NameErrors. This also restores the data loaders so training can run, after which inference and the ensemble creation work and produce a valid `submission.csv`. The core model architecture and training logic remain unchanged.'
- What this solution (achieved 0.59417) has done: 'I add all required imports and definitions at the start, fixing the NameError issues for torch, random, os, numpy, pandas, glob, cv2, albumentations, torch modules, and sklearn utilities. This makes the pipeline runnable end‑to‑end and ensures a proper `submission.csv` is created.'
- What this solution (achieved 0.57913) has done: 'Implemented fixes to address the runtime errors:
- Updated Albumentations `RandomResizedCrop` usage to the current API (`size=(224,224)`) so transforms initialize correctly.
- Ensured that the training/validation/test transform objects (`train_tf`, `val_tf`, `test_tf`) are defined after the fix, eliminating the subsequent NameErrors.
- Added a small import guard for `torch.backends.cudnn` determinism (no impact on core logic) and clarified comments.

These minimal changes restore the full end‑to‑end pipeline, allowing model training, inference, and generation of a valid `submission.csv` file.'

# 9. Code solution

## === cell 0
import os
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class LabelSmoothing(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target):
        target = target * (1 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(logits.squeeze(), target.squeeze())


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count != 0 else 0.0


class RocAucMeter:
    def __init__(self):
        self.targets = []
        self.preds = []
        self._score = None

    def update(self, targets, preds):
        self.targets.append(targets.detach().cpu().numpy())
        self.preds.append(preds.detach().cpu().numpy())

    @property
    def score(self):
        if self._score is None:
            y_true = np.concatenate(self.targets).ravel()
            y_score = np.concatenate(self.preds).ravel()
            if len(np.unique(y_true)) == 1:
                self._score = 0.5
            else:
                self._score = roc_auc_score(y_true, y_score)
        return self._score

    @score.setter
    def score(self, value):
        self._score = value




## === cell 1
SEED = 42


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(SEED)




## === cell 2
class Config:
    batch_size = 64
    n_epochs = 4  # a few more epochs to improve AUC
    num_workers = 8
    lr = 0.001
    verbose = True
    verbose_step = 1
    step_scheduler = False
    validation_scheduler = True
    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode="min",
        factor=0.5,
        patience=1,
        verbose=False,
        threshold=1e-4,
        threshold_mode="abs",
        cooldown=0,
        min_lr=1e-8,
        eps=1e-8,
    )




## === cell 3
import cv2
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm


class StegoDataset(Dataset):
    def __init__(self, items, transforms=None, has_label=True):
        """
        items: list of (image_path, label) if has_label else list of image_path
        """
        self.items = items
        self.transforms = transforms
        self.has_label = has_label

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        if self.has_label:
            path, label = self.items[idx]
        else:
            path = self.items[idx]
            label = -1  # dummy
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms:
            img = self.transforms(image=img)["image"]
        if self.has_label:
            return img, torch.tensor(label, dtype=torch.float32)
        else:
            return img, torch.tensor(0.0)  # placeholder for compatibility


train_tf = A.Compose(
    [
        A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)
val_tf = A.Compose(
    [
        A.Resize(height=224, width=224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)
test_tf = val_tf  # same as validation



## === cell 4
possible_roots = [
    os.path.join("input", "alaska2-image-steganalysis"),
    os.path.join("working", "alaska2-image-steganalysis"),
    os.path.join("/kaggle/input", "alaska2-image-steganalysis"),
    os.path.join("/kaggle/working", "alaska2-image-steganalysis"),
]
DATA_ROOT = None
for p in possible_roots:
    if os.path.isdir(p):
        DATA_ROOT = p
        break
if DATA_ROOT is None:
    raise FileNotFoundError("Could not locate the dataset root directory.")

cover_dir = os.path.join(DATA_ROOT, "Cover")
stego_dirs = [os.path.join(DATA_ROOT, d) for d in ["JMiPOD", "JUNIWARD", "UERD"]]

MAX_PER_CLASS = 25000  # adjust if memory permits

cover_imgs = sorted(
    [os.path.join(cover_dir, f) for f in os.listdir(cover_dir) if f.endswith(".jpg")]
)[:MAX_PER_CLASS]

stego_imgs = []
for d in stego_dirs:
    stego_imgs.extend([os.path.join(d, f) for f in os.listdir(d) if f.endswith(".jpg")])
stego_imgs = stego_imgs[:MAX_PER_CLASS]  # keep balanced amount

labels = [0] * len(cover_imgs) + [1] * len(stego_imgs)
paths = cover_imgs + stego_imgs

train_paths, val_paths, train_labels, val_labels = train_test_split(
    paths,
    labels,
    test_size=0.2,
    random_state=SEED,
    stratify=labels,
)

train_items = list(zip(train_paths, train_labels))
val_items = list(zip(val_paths, val_labels))



## === cell 5
train_dataset = StegoDataset(train_items, transforms=train_tf, has_label=True)
val_dataset = StegoDataset(val_items, transforms=val_tf, has_label=True)

label_counts = np.bincount(train_labels)
class_weights = 1.0 / label_counts
sample_weights = [class_weights[label] for label in train_labels]
sampler = WeightedRandomSampler(
    sample_weights, num_samples=len(sample_weights), replacement=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=Config.batch_size,
    sampler=sampler,
    num_workers=Config.num_workers,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=Config.batch_size,
    shuffle=False,
    num_workers=Config.num_workers,
    pin_memory=True,
)



## === cell 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model("resnet18", pretrained=True, num_classes=1)
model = model.to(device)

fitter = Fitter(model, device, Config)
fitter.fit(train_loader, val_loader)

ckpt_files = [
    f for f in os.listdir(".") if f.startswith("best-checkpoint") and f.endswith(".bin")
]
if ckpt_files:
    best_ckpt = max(
        ckpt_files,
        key=lambda x: int(x.split("-")[2].split("epoch")[0]),
    )
    fitter.load(best_ckpt)
else:
    print("No checkpoint found; using last model state.")



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/477451320.py in <cell line: 0>()
      3 model = model.to(device)
      4 
----> 5 fitter = Fitter(model, device, Config)
      6 fitter.fit(train_loader, val_loader)
      7 

NameError: name 'Fitter' is not defined

## === cell 7
test_dir = os.path.join(DATA_ROOT, "Test")
test_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".jpg")])
test_paths = [os.path.join(test_dir, f) for f in test_files]

test_dataset = StegoDataset(test_paths, transforms=test_tf, has_label=False)
test_loader = DataLoader(
    test_dataset,
    batch_size=Config.batch_size,
    shuffle=False,
    num_workers=Config.num_workers,
    pin_memory=True,
)

model.eval()
all_preds = []
with torch.no_grad():
    for imgs, _ in test_loader:
        imgs = imgs.to(device, non_blocking=True).float()
        with (
            torch.autocast(device_type=device.type, dtype=torch.float16)
            if device.type == "cuda"
            else torch.no_grad()
        ):
            logits = model(imgs)
        probs = torch.sigmoid(logits).cpu().numpy().ravel()
        all_preds.extend(probs)

submission = pd.DataFrame({"Id": test_files, "Label": all_preds})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path} ({submission.shape[0]} rows)")
