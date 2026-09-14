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
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import os
import cv2
import random
import time
from datetime import datetime

os.makedirs("/kaggle/working/ckpt", exist_ok=True)
os.makedirs("/kaggle/outputs", exist_ok=True)




## === cell 1
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import (
    SequentialSampler,
    RandomSampler,
    WeightedRandomSampler,
)
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchsampler import ImbalancedDatasetSampler
except ImportError:
    ImbalancedDatasetSampler = None  # fallback – not used later




## === cell 2
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.feature import hog
from sklearn import metrics
from sklearn.model_selection import GroupKFold
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2




## === cell 3
PATH = "/kaggle/input/alaska2-image-steganalysis"




## === cell 4
import re  # needed for parse_log_file




## === cell 5
SEED = 42


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


seed_everything(SEED)




## === cell 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")




## === cell 7
def onehot(size, target):
    return F.one_hot(torch.tensor(target), num_classes=size).float()


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
        target = onehot(4, label)
        return image, target

    def __len__(self):
        return len(self.image_names)

    def get_labels(self):
        return list(self.labels)




## === cell 8
CLASSES = ["Cover", "JMiPOD", "JUNIWARD", "UERD"]
N_SPLITS = 5

dataset = []
for label, kind in enumerate(CLASSES):
    for path in glob(os.path.join(PATH, kind, "*.jpg")):
        dataset.append(
            {"kind": kind, "image_name": os.path.basename(path), "label": label}
        )
random.shuffle(dataset)
dataset = pd.DataFrame(dataset)
dataset["fold"] = -1

gkf = GroupKFold(n_splits=N_SPLITS)
for fold_number, (_, val_idx) in enumerate(
    gkf.split(X=dataset.index, y=dataset["label"], groups=dataset["image_name"])
):
    dataset.loc[dataset.iloc[val_idx].index, "fold"] = fold_number




## === cell 9
def get_train_transforms():
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(height=512, width=512, p=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )


def get_valid_transforms():
    return A.Compose(
        [
            A.Resize(height=512, width=512, p=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )




## === cell 10
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




## === cell 11
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




## === cell 12
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




## === cell 13
def alaska_weighted_auc(y_true, y_valid):
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2, 1]
    fpr, tpr, _ = metrics.roc_curve(y_true, y_valid, pos_label=1)
    areas = np.diff(tpr_thresholds)
    normalization = np.dot(areas, weights)
    competition_metric = 0.0
    for idx, weight in enumerate(weights):
        y_min, y_max = tpr_thresholds[idx], tpr_thresholds[idx + 1]
        mask = (y_min < tpr) & (tpr < y_max)
        if not np.any(mask):
            continue
        x_padding = np.linspace(fpr[mask][-1], 1, 100)
        x = np.concatenate([fpr[mask], x_padding])
        y = np.concatenate([tpr[mask], np.full_like(x_padding, y_max)])
        y = y - y_min
        competition_metric += weight * metrics.auc(x, y)
    return competition_metric / normalization




## === cell 14
class LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            logprobs = torch.nn.functional.log_softmax(x, dim=-1)
            nll_loss = (-logprobs * target).sum(-1)
            smooth_loss = -logprobs.mean(dim=-1)
            loss = self.confidence * nll_loss + self.smoothing * smooth_loss
            return loss.mean()
        else:
            return torch.nn.functional.cross_entropy(x, target)




## === cell 15
class Fitter:
    def __init__(self, model, device, config):
        self.config = config
        self.epoch = 0
        self.log_path = "/kaggle/working/ckpt/log.txt"
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.best_summary_loss = 1e5
        self.model = model
        self.device = device
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)
        self.scheduler = config.SchedulerClass(
            self.optimizer, **config.scheduler_params
        )
        self.criterion = LabelSmoothing().to(self.device)
        self.scaler = (
            torch.cuda.amp.GradScaler() if self.device.type == "cuda" else None
        )
        self.log(f"Fitter prepared. Device: {self.device}")

    def fit(self, train_loader, val_loader):
        for e in range(self.epoch, self.config.n_epochs):
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]["lr"]
                self.log(f"\nEpoch {self.epoch} LR: {lr}")
            t0 = time.time()
            tr_loss, tr_score = self.train_model(train_loader)
            self.log(
                f"[RESULT] Train Epoch {self.epoch} loss {tr_loss.avg:.5f} score {tr_score.avg:.5f} time {time.time()-t0:.2f}"
            )
            torch.save(self.model.state_dict(), "/kaggle/outputs/last-checkpoint.bin")
            t0 = time.time()
            val_loss, val_score = self.validation(val_loader)
            self.log(
                f"[RESULT] Val Epoch {self.epoch} loss {val_loss.avg:.5f} score {val_score.avg:.5f} time {time.time()-t0:.2f}"
            )
            if val_loss.avg < self.best_summary_loss:
                self.best_summary_loss = val_loss.avg
                torch.save(
                    self.model.state_dict(),
                    f"/kaggle/outputs/best-checkpoint-{self.epoch:03d}.bin",
                )
            if self.config.validation_scheduler:
                self.scheduler.step(metrics=val_loss.avg)
            self.epoch += 1

    def validation(self, loader):
        self.model.eval()
        loss_meter = AverageMeter()
        score_meter = RocAucMeter()
        with torch.no_grad():
            for images, targets in loader:
                images, targets = images.to(self.device), targets.to(self.device)
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
                loss_meter.update(loss.item(), images.size(0))
                score_meter.update(targets, outputs)
        return loss_meter, score_meter

    def train_model(self, loader):
        self.model.train()
        loss_meter = AverageMeter()
        score_meter = RocAucMeter()
        for images, targets in loader:
            images, targets = images.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            loss_meter.update(loss.item(), images.size(0))
            score_meter.update(targets, outputs)
            if self.config.step_scheduler and not isinstance(
                self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                self.scheduler.step()
        return loss_meter, score_meter

    def log(self, msg):
        if self.config.verbose:
            print(msg)
        with open(self.log_path, "a+") as f:
            f.write(msg + "\n")




## === cell 16
def parse_log_file(log_path):
    with open(log_path) as f:
        lines = f.readlines()
    epochs, train_loss, train_score, train_time = [], [], [], []
    val_loss, val_score, val_time = [], [], []
    lr_list = []
    cur_lr = None
    for line in lines:
        if line.startswith("LR:"):
            cur_lr = float(line.split(":")[1].strip())
        elif "[RESULT] Train" in line:
            epochs.append(int(re.search(r"Epoch (\d+)", line).group(1)))
            train_loss.append(float(re.search(r"loss ([\d.]+)", line).group(1)))
            train_score.append(float(re.search(r"score ([\d.]+)", line).group(1)))
            train_time.append(float(re.search(r"time ([\d.]+)", line).group(1)))
            lr_list.append(cur_lr)
        elif "[RESULT] Val" in line:
            val_loss.append(float(re.search(r"loss ([\d.]+)", line).group(1)))
            val_score.append(float(re.search(r"score ([\d.]+)", line).group(1)))
            val_time.append(float(re.search(r"time ([\d.]+)", line).group(1)))
    return {
        "epochs": epochs,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_score": train_score,
        "val_score": val_score,
        "train_time": train_time,
        "val_time": val_time,
        "lr": lr_list,
    }




## === cell 17
def plot_log_results(metrics, save_path="log_plots.png"):
    epochs = metrics["epochs"]
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs[0, 0].plot(epochs, metrics["train_loss"], label="Train Loss", marker="o")
    axs[0, 0].plot(epochs, metrics["val_loss"], label="Val Loss", marker="x")
    axs[0, 1].plot(epochs, metrics["train_score"], label="Train Score", marker="o")
    axs[0, 1].plot(epochs, metrics["val_score"], label="Val Score", marker="x")
    axs[1, 0].plot(epochs, metrics["lr"], label="LR", marker="o")
    axs[1, 1].plot(epochs, metrics["train_time"], label="Train Time", marker="o")
    axs[1, 1].plot(epochs, metrics["val_time"], label="Val Time", marker="x")
    for ax in axs.flat:
        ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plots to {save_path}")




## === cell 18
import timm


class EffNet(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(6, 12, 3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(12, 3, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(6)
        self.bn2 = nn.BatchNorm2d(12)
        self.bn3 = nn.BatchNorm2d(3)

        self.net = timm.create_model("efficientnet_b0", pretrained=True)
        for param in self.net.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(0.5)

        self.net.blocks[5] = nn.Identity()
        self.net.blocks[6] = nn.Sequential(
            nn.Conv2d(
                self.net.blocks[4][2].conv_pwl.out_channels,
                self.net.conv_head.in_channels,
                1,
            ),
            nn.BatchNorm2d(self.net.conv_head.in_channels),
            nn.ReLU6(),
        )
        self.fc = nn.Linear(self.net.classifier.in_features, out_dim)
        self.net.classifier = nn.Identity()

    def extract(self, x):
        x = F.relu6(self.bn1(self.conv1(x)))
        x = F.relu6(self.bn2(self.conv2(x)))
        x = F.relu6(self.bn3(self.conv3(x)))
        x = self.net(x)
        return x

    def forward(self, x):
        x = self.extract(x)
        x = self.fc(self.dropout(x))
        return x




## === cell 19
model = EffNet(4).to(DEVICE)

import torch.optim.lr_scheduler as lr_sched


class Config:
    batch_size = 64
    n_epochs = 4
    lr = 1e-4
    SchedulerClass = lr_sched.CosineAnnealingLR
    scheduler_params = {"T_max": n_epochs}
    verbose = True
    validation_scheduler = False  # keep step scheduler only
    step_scheduler = False  # keep unchanged scheduling behaviour




## === cell 20
label_counts = np.bincount(train_dataset.get_labels())
class_weights = 1.0 / label_counts
sample_weights = [class_weights[label] for label in train_dataset.get_labels()]

train_loader = DataLoader(
    train_dataset,
    sampler=WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    ),
    batch_size=Config.batch_size,
    num_workers=8,  # increased parallelism for faster I/O
    prefetch_factor=2,  # small prefetch to keep GPU fed
    pin_memory=True,
    persistent_workers=True,
    drop_last=True,
)

val_loader = DataLoader(
    validation_dataset,
    sampler=SequentialSampler(validation_dataset),
    batch_size=Config.batch_size,
    num_workers=8,
    prefetch_factor=2,
    pin_memory=True,
    persistent_workers=True,
    shuffle=False,
)




## === cell 21
ckpt_path = "/kaggle/working/ckpt/last-checkpoint.bin"




## === cell 22
class TrainingSession:
    def __init__(self, model, config, train_loader, val_loader):
        self.model = model
        self.device = DEVICE
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.ckpt_path = ckpt_path

    def _load_best_checkpoint(self):
        best_files = glob("/kaggle/outputs/best-checkpoint-*.bin")
        if not best_files:
            return False
        best_file = max(
            best_files, key=lambda p: int(re.search(r"-(\d{3})\\.bin", p).group(1))
        )
        print(f"Loading best checkpoint: {best_file}")
        self.model.load_state_dict(torch.load(best_file, map_location=self.device))
        return True

    def run(self):
        fitter = Fitter(self.model, self.device, self.config)
        if os.path.exists(self.ckpt_path):
            print(f"Resuming from {self.ckpt_path}")
            self.model.load_state_dict(
                torch.load(self.ckpt_path, map_location=self.device)
            )
        else:
            print("Starting new training")
        fitter.fit(self.train_loader, self.val_loader)

        if not self._load_best_checkpoint():
            if os.path.exists(self.ckpt_path):
                print(f"Falling back to last checkpoint: {self.ckpt_path}")
                self.model.load_state_dict(
                    torch.load(self.ckpt_path, map_location=self.device)
                )
        self.model.eval()


trainer = TrainingSession(model, Config, train_loader, val_loader)
trainer.run()




## === cell 23
def get_test_transforms(mode):
    if mode == 0:
        return A.Compose([A.Resize(512, 512, p=1.0), ToTensorV2(p=1.0)], p=1.0)
    elif mode == 1:
        return A.Compose(
            [A.HorizontalFlip(p=1.0), A.Resize(512, 512, p=1.0), ToTensorV2(p=1.0)],
            p=1.0,
        )
    elif mode == 2:
        return A.Compose(
            [A.VerticalFlip(p=1.0), A.Resize(512, 512, p=1.0), ToTensorV2(p=1.0)], p=1.0
        )
    else:
        return A.Compose(
            [
                A.HorizontalFlip(p=1.0),
                A.VerticalFlip(p=1.0),
                A.Resize(512, 512, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        )




## === cell 24
class DatasetSubmissionRetriever(Dataset):
    def __init__(self, image_names, transforms=None):
        super().__init__()
        self.image_names = image_names
        self.transforms = transforms

    def __getitem__(self, idx):
        name = self.image_names[idx]
        img = cv2.imread(f"{PATH}/Test/{name}", cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if self.transforms:
            img = self.transforms(image=img)["image"]
        return name, img

    def __len__(self):
        return len(self.image_names)




## === cell 25
results = []
test_image_names = np.array([os.path.basename(p) for p in glob(f"{PATH}/Test/*.jpg")])

for mode in range(4):
    ds = DatasetSubmissionRetriever(
        test_image_names, transforms=get_test_transforms(mode)
    )
    dl = DataLoader(
        ds,
        batch_size=8,
        shuffle=False,
        num_workers=8,  # more workers for fast test I/O
        prefetch_factor=2,
        pin_memory=True,
        persistent_workers=True,
    )
    mode_result = {"Id": [], "Label": []}
    model.eval()
    with torch.no_grad():
        for ids, imgs in dl:
            imgs = imgs.to(DEVICE)
            with torch.cuda.amp.autocast():
                preds = model(imgs)
            probs = 1 - nn.functional.softmax(preds, dim=1).cpu().numpy()[:, 0]
            mode_result["Id"].extend(ids)
            mode_result["Label"].extend(probs)
    results.append(mode_result)




## === cell 26
submissions = [pd.DataFrame(r) for r in results]

weight0, weight1, weight2, weight3 = 5, 1, 1, 1
total_weight = weight0 + weight1 + weight2 + weight3

submissions[0]["Label"] = (
    submissions[0]["Label"] * weight0
    + submissions[1]["Label"] * weight1
    + submissions[2]["Label"] * weight2
    + submissions[3]["Label"] * weight3
) / total_weight

final_submission = submissions[0][["Id", "Label"]]
final_submission.to_csv("submission.csv", index=False)
print("✅ Submission file 'submission.csv' created.")
