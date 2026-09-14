# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import warnings

warnings.filterwarnings("ignore", category=ResourceWarning)

import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import os
import cv2
import random
import time
from datetime import datetime

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
import torch.nn as nn
import torch.nn.functional as F
import timm
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.feature import hog
from sklearn import metrics
from sklearn.model_selection import GroupKFold
import re
from torch.cuda.amp import autocast, GradScaler  # added for mixed‑precision




## === cell 1
PATH = "/kaggle/input/alaska2-image-steganalysis"
SEED = 42


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


seed_everything(SEED)




## === cell 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")




## === cell 3
class DatasetRetriever(Dataset):
    def __init__(self, kinds, image_names, labels, transforms=None):
        super().__init__()
        self.kinds = kinds
        self.image_names = image_names
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, index: int):
        kind, image_name, label = (
            self.kinds[index],
            self.image_names[index],
            self.labels[index],
        )
        image = cv2.imread(f"{PATH}/{kind}/{image_name}", cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if self.transforms:
            sample = {"image": image}
            sample = self.transforms(**sample)
            image = sample["image"]
        target = ONE_HOT_CACHE[label]  # use cached one‑hot tensor
        return image, target

    def __len__(self) -> int:
        return len(self.image_names)

    def get_labels(self):
        return list(self.labels)




## === cell 4
CLASSES = ["Cover", "JMiPOD", "JUNIWARD", "UERD"]
N_SPLITS = 5

dataset = []
for label, kind in enumerate(CLASSES):
    image_paths = glob(os.path.join(PATH, kind, "*.jpg"))
    for path in image_paths:
        dataset.append(
            {"kind": kind, "image_name": os.path.basename(path), "label": label}
        )
random.shuffle(dataset)
dataset = pd.DataFrame(dataset)
dataset["fold"] = 0

gkf = GroupKFold(n_splits=N_SPLITS)
for fold_number, (train_idx, val_idx) in enumerate(
    gkf.split(X=dataset.index, y=dataset["label"], groups=dataset["image_name"])
):
    dataset.loc[dataset.iloc[val_idx].index, "fold"] = fold_number




## === cell 5
def get_train_transforms():
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(height=224, width=224, p=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )


def get_valid_transforms():
    return A.Compose(
        [
            A.Resize(height=224, width=224, p=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )




## === cell 6
ONE_HOT_CACHE = {i: torch.eye(4, dtype=torch.float32)[i] for i in range(4)}




## === cell 7
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




## === cell 8
class RocAucMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.y_true = np.array([0, 1])
        self.y_pred = np.array([0.5, 0.5])
        self.score = 0

    def update(self, y_true, y_pred):
        y_true = y_true.cpu().numpy().argmax(axis=1).clip(min=0, max=1).astype(int)
        y_pred = 1 - nn.functional.softmax(y_pred, dim=1).data.cpu().numpy()[:, 0]
        self.y_true = np.hstack((self.y_true, y_true))
        self.y_pred = np.hstack((self.y_pred, y_pred))
        self.score = alaska_weighted_auc(self.y_true, self.y_pred)

    @property
    def avg(self):
        return self.score




## === cell 9
def alaska_weighted_auc(y_true, y_valid):
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2, 1]
    fpr, tpr, _ = metrics.roc_curve(y_true, y_valid, pos_label=1)
    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])
    normalization = np.dot(areas, weights)
    competition_metric = 0
    for idx, weight in enumerate(weights):
        y_min = tpr_thresholds[idx]
        y_max = tpr_thresholds[idx + 1]
        mask = (y_min < tpr) & (tpr < y_max)
        if not np.any(mask):
            continue
        x_padding = np.linspace(fpr[mask][-1], 1, 100)
        x = np.concatenate([fpr[mask], x_padding])
        y = np.concatenate([tpr[mask], [y_max] * len(x_padding)])
        y = y - y_min
        competition_metric += weight * metrics.auc(x, y)
    return competition_metric / normalization




## === cell 10
class LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = target.float()
            logprobs = torch.nn.functional.log_softmax(x, dim=-1)
            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)
            smooth_loss = -logprobs.mean(dim=-1)
            loss = self.confidence * nll_loss + self.smoothing * smooth_loss
            return loss.mean()
        else:
            return torch.nn.functional.cross_entropy(x, target)




## === cell 11
class Config:
    batch_size = 64  # larger batch reduces iteration overhead
    n_epochs = 3  # increased epochs for better learning
    num_workers = 8  # more workers speeds up image loading
    lr = 0.001
    verbose = True
    verbose_step = 1
    step_scheduler = False
    validation_scheduler = True
    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode="max",  # we maximise weighted AUC
        factor=0.5,
        patience=1,
        verbose=False,
        threshold=0.0001,
        threshold_mode="abs",
        cooldown=0,
        min_lr=1e-8,
        eps=1e-08,
    )




## === cell 12
fold_number = 0

train_dataset = DatasetRetriever(
    kinds=dataset[dataset["fold"] != fold_number].kind.values,
    image_names=dataset[dataset["fold"] != fold_number].image_name.values,
    labels=dataset[dataset["fold"] != fold_number].label.values,
    transforms=get_train_transforms(),
)

validation_dataset = DatasetRetriever(
    kinds=dataset[dataset["fold"] == fold_number].kind.values,
    image_names=dataset[dataset["fold"] == fold_number].image_name.values,
    labels=dataset[dataset["fold"] == fold_number].label.values,
    transforms=get_valid_transforms(),
)

train_loader = DataLoader(
    train_dataset,
    sampler=RandomSampler(train_dataset),
    batch_size=Config.batch_size,
    num_workers=Config.num_workers,
    pin_memory=torch.cuda.is_available(),  # pin only when using CUDA
    drop_last=True,
    persistent_workers=True,  # keep workers alive between epochs
)

val_loader = DataLoader(
    validation_dataset,
    sampler=SequentialSampler(validation_dataset),
    batch_size=Config.batch_size,
    num_workers=Config.num_workers,
    pin_memory=torch.cuda.is_available(),
    shuffle=False,
    drop_last=False,
    persistent_workers=True,
)




## === cell 13
class EffNet(nn.Module):
    def __init__(self, out_dim):
        super(EffNet, self).__init__()
        self.net = timm.create_model(
            "efficientnet_b0", pretrained=True, num_classes=out_dim
        )

    def forward(self, x):
        return self.net(x)




## === cell 14
model = EffNet(4).to(DEVICE)




## === cell 15
class Fitter:
    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.cfg = config
        self.criterion = LabelSmoothing(smoothing=0.1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.scaler = GradScaler()  # mixed‑precision scaler
        self.scheduler = None
        if self.cfg.validation_scheduler:
            self.scheduler = self.cfg.SchedulerClass(
                self.optimizer, **self.cfg.scheduler_params
            )
        self.epoch = 0
        self.best_score = -np.inf

    def fit(self, train_loader, val_loader):
        for epoch in range(self.cfg.n_epochs):
            self.epoch = epoch
            self.train_one_epoch(train_loader)
            val_score = self.validate_one_epoch(val_loader)

            if val_score > self.best_score:
                self.best_score = val_score
                self.save(f"best-checkpoint-{epoch}epoch.bin")
                if self.cfg.verbose:
                    print(f"✅ New best score: {val_score:.5f}")

            if self.scheduler:
                self.scheduler.step(val_score)

    def train_one_epoch(self, loader):
        self.model.train()
        loss_meter = AverageMeter()
        for step, (images, targets) in enumerate(loader):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            self.optimizer.zero_grad()
            with autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            loss_meter.update(loss.item(), images.size(0))
            if self.cfg.verbose and (step + 1) % self.cfg.verbose_step == 0:
                print(f"Epoch {self.epoch} | Step {step+1} | Loss {loss_meter.avg:.5f}")

    def validate_one_epoch(self, loader):
        self.model.eval()
        auc_meter = RocAucMeter()
        with torch.no_grad():
            for images, targets in loader:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                with autocast():
                    outputs = self.model(images)
                auc_meter.update(targets, outputs)
        if self.cfg.verbose:
            print(f"Epoch {self.epoch} | Validation Weighted AUC: {auc_meter.avg:.5f}")
        return auc_meter.avg

    def save(self, path):
        torch.save(
            {"model_state_dict": self.model.state_dict(), "epoch": self.epoch}, path
        )

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.epoch = ckpt.get("epoch", 0)




## === cell 16
class TrainingSession:
    def __init__(
        self,
        model,
        config,
        train_loader,
        val_loader,
        ckpt_folder="/kaggle/input/alaska-checkpoint",
        output_log_path="/kaggle/working/log.txt",
    ):
        self.model = model
        self.device = DEVICE
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.ckpt_folder = ckpt_folder
        self.output_log_path = output_log_path

    def append_previous_log(self):
        prev_log = os.path.join(self.ckpt_folder, "log.txt")
        if os.path.exists(prev_log):
            with open(prev_log, "r") as f:
                old_content = f.read()
            with open(self.output_log_path, "a+") as f:
                f.write("\n\n# ==== Previous log ====\n")
                f.write(old_content)
                f.write("\n\n# ==== New session start ====\n")
            print(f"Appended old log from {prev_log}")
        else:
            print(f"No previous log at {prev_log}")

    def get_latest_best_checkpoint(self):
        pattern = re.compile(r"best-checkpoint-(\d+)epoch\.bin")
        max_epoch = -1
        best_path = None
        if not os.path.exists(self.ckpt_folder):
            return None
        for fname in os.listdir(self.ckpt_folder):
            match = pattern.match(fname)
            if match:
                epoch = int(match.group(1))
                if epoch > max_epoch:
                    max_epoch = epoch
                    best_path = os.path.join(self.ckpt_folder, fname)
        return best_path

    def run(self):
        fitter = Fitter(self.model, self.device, self.config)
        self.append_previous_log()
        best_ckpt = self.get_latest_best_checkpoint()
        if best_ckpt:
            print(f"Resuming from checkpoint: {best_ckpt}")
            fitter.load(best_ckpt)
        else:
            print("No checkpoint found, training from scratch")
        print(f"Starting training at epoch {fitter.epoch}")
        fitter.fit(self.train_loader, self.val_loader)




## === cell 17
session = TrainingSession(model, Config, train_loader, val_loader)




## === cell 18
session.run()




## === cell 19
def get_test_transforms(mode):
    if mode == 0:
        return A.Compose(
            [
                A.Resize(height=224, width=224, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        )
    elif mode == 1:
        return A.Compose(
            [
                A.HorizontalFlip(p=1),
                A.Resize(height=224, width=224, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        )
    elif mode == 2:
        return A.Compose(
            [
                A.VerticalFlip(p=1),
                A.Resize(height=224, width=224, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        )
    else:
        return A.Compose(
            [
                A.HorizontalFlip(p=1),
                A.VerticalFlip(p=1),
                A.Resize(height=224, width=224, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        )




## === cell 20
class DatasetSubmissionRaw(Dataset):
    """
    Returns raw (unnormalized) images as NumPy arrays.
    Transformations are applied later in the inference loop to avoid
    re‑reading the same image for each TTA mode.
    """

    def __init__(self, image_names):
        super().__init__()
        self.image_names = image_names

    def __getitem__(self, index: int):
        image_name = self.image_names[index]
        img = cv2.imread(f"{PATH}/Test/{image_name}", cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return image_name, img  # return NumPy array

    def __len__(self) -> int:
        return len(self.image_names)




## === cell 21
test_image_paths = glob(os.path.join(PATH, "Test", "*.jpg"))
test_image_names = np.array([os.path.basename(p) for p in test_image_paths])

test_dataset = DatasetSubmissionRaw(test_image_names)
test_loader = DataLoader(
    test_dataset,
    batch_size=32,  # larger batch reduces loader overhead
    shuffle=False,
    num_workers=8,
    pin_memory=torch.cuda.is_available(),
    drop_last=False,
    persistent_workers=True,
)

tta_transforms = [get_test_transforms(m) for m in range(4)]

all_ids = []
all_preds = np.zeros(len(test_image_names), dtype=np.float32)

model.eval()
with torch.no_grad():
    for batch_idx, (img_ids, imgs_np) in enumerate(test_loader):
        imgs_tensor = (
            torch.from_numpy(np.stack(imgs_np))
            .permute(0, 3, 1, 2)
            .to(DEVICE, non_blocking=True)
        )

        batch_preds = np.zeros(imgs_tensor.size(0), dtype=np.float32)
        for transform in tta_transforms:
            augmented = [transform(image=img)["image"].unsqueeze(0) for img in imgs_np]
            aug_tensor = torch.cat(augmented, dim=0).to(DEVICE, non_blocking=True)

            with autocast():
                outputs = model(aug_tensor)
            probs = 1 - nn.functional.softmax(outputs, dim=1).cpu().numpy()[:, 0]
            batch_preds += probs.astype(np.float32)

        batch_preds /= len(tta_transforms)  # average over TTA modes

        start_idx = batch_idx * test_loader.batch_size
        all_preds[start_idx : start_idx + len(img_ids)] = batch_preds
        all_ids.extend(img_ids)

final_submission = pd.DataFrame({"Id": all_ids, "Label": all_preds})
final_submission.to_csv("submission.csv", index=False)
print("✅ Submission file saved as submission.csv")
