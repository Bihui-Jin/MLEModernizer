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
import re
from datetime import datetime




## === cell 1
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler

import timm




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
SEED = 42


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False  # allow faster kernels


seed_everything(SEED)




## === cell 5
def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.0
    return vec


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

    def __len__(self) -> int:
        return self.image_names.shape[0]

    def get_labels(self):
        return list(self.labels)


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




## === cell 6
print("One‑hot vectors for verification:")
for i in range(4):
    print(f"Class {i}: {onehot(4, i)}")




## === cell 7
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RocAucMeter(object):
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
        if np.sum(mask) == 0:
            continue
        x_padding = np.linspace(fpr[mask][-1], 1, 100)
        x = np.concatenate([fpr[mask], x_padding])
        y = np.concatenate([tpr[mask], [y_max] * len(x_padding)])
        y = y - y_min
        competition_metric += weight * metrics.auc(x, y)
    return competition_metric / normalization




## === cell 8
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




## === cell 9
class Fitter:
    def __init__(self, model, device, config):
        self.config = config
        self.epoch = 0
        self.base_dir = "./"
        self.log_path = f"{self.base_dir}/log.txt"
        self.best_summary_loss = 1e5
        self.model = model
        self.device = device
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)
        self.scheduler = config.SchedulerClass(
            self.optimizer, **config.scheduler_params
        )
        self.criterion = LabelSmoothing().to(self.device)
        self.log(f"Fitter prepared. Device is {self.device}")

    def fit(self, train_loader, validation_loader):
        for e in range(self.epoch, self.config.n_epochs):
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]["lr"]
                timestamp = datetime.utcnow().isoformat()
                self.log(f"\n{timestamp}\nLR: {lr}")
            t = time.time()
            summary_loss, final_scores = self.train_model(train_loader)
            self.log(
                f"[RESULT]: Train. Epoch: {self.epoch},summary_loss: {summary_loss.avg:.5f},final_score: {final_scores.avg:.5f},time: {(time.time()-t):.5f}"
            )
            self.save(f"{self.base_dir}/last-checkpoint.bin")
            t = time.time()
            summary_loss, final_scores = self.validation(validation_loader)
            self.log(
                f"[RESULT]: Val. Epoch: {self.epoch},summary_loss: {summary_loss.avg:.5f},final_score: {final_scores.avg:.5f},time: {(time.time()-t):.5f}"
            )
            if summary_loss.avg < self.best_summary_loss:
                self.best_summary_loss = summary_loss.avg
                self.model.eval()
                self.save(
                    f"{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin"
                )
                for path in sorted(glob(f"{self.base_dir}/best-checkpoint-*epoch.bin"))[
                    :-3
                ]:
                    os.remove(path)
            if self.config.validation_scheduler:
                self.scheduler.step(metrics=summary_loss.avg)
            self.epoch += 1

    def validation(self, val_loader):
        self.model.eval()
        summary_loss = AverageMeter()
        final_scores = RocAucMeter()
        amp_ctx = (
            torch.cuda.amp.autocast() if torch.cuda.is_available() else nullcontext()
        )
        for step, (images, targets) in enumerate(val_loader):
            with torch.no_grad():
                targets = targets.to(self.device).float()
                images = images.to(self.device).float()
                with amp_ctx:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
                final_scores.update(targets, outputs)
                summary_loss.update(loss.detach().item(), images.shape[0])
        return summary_loss, final_scores

    def train_model(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter()
        final_scores = RocAucMeter()
        amp_ctx = (
            torch.cuda.amp.autocast() if torch.cuda.is_available() else nullcontext()
        )
        for step, (images, targets) in enumerate(train_loader):
            targets = targets.to(self.device).float()
            images = images.to(self.device).float()
            self.optimizer.zero_grad()
            with amp_ctx:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
            loss.backward()
            final_scores.update(targets, outputs)
            summary_loss.update(loss.detach().item(), images.shape[0])
            self.optimizer.step()
            if self.config.step_scheduler:
                self.scheduler.step()
        return summary_loss, final_scores

    def save(self, path):
        self.model.eval()
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_summary_loss": self.best_summary_loss,
                "epoch": self.epoch,
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_summary_loss = checkpoint["best_summary_loss"]
        self.epoch = checkpoint["epoch"] + 1
        self.log(f"🔁 Loaded checkpoint from {path}, resume at epoch {self.epoch}")

    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, "a+") as logger:
            logger.write(f"{message}\n")




## === cell 10
class EffNet(nn.Module):
    def __init__(self, out_dim):
        super(EffNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(6, 12, 3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(12, 36, 3, stride=1, padding=1, bias=False)
        self.mybn1 = nn.BatchNorm2d(6)
        self.mybn2 = nn.BatchNorm2d(12)
        self.mybn3 = nn.BatchNorm2d(36)

        self.net = timm.create_model("efficientnet_b0", pretrained=True)
        self.net.conv_stem.weight = nn.Parameter(
            self.net.conv_stem.weight.repeat(1, 12, 1, 1)
        )

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
        self.myfc = nn.Linear(self.net.classifier.in_features, out_dim)
        self.net.classifier = nn.Identity()

    def extract(self, x):
        x = F.relu6(self.mybn1(self.conv1(x)))
        x = F.relu6(self.mybn2(self.conv2(x)))
        x = F.relu6(self.mybn3(self.conv3(x)))
        x = self.net(x)
        return x

    def forward(self, x):
        x = self.extract(x)
        x = self.myfc(self.dropout(x))
        return x




## === cell 11
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EffNet(4).to(DEVICE)




## === cell 12
class Config:
    batch_size = 64  # larger batch for better GPU throughput
    n_epochs = 3
    num_workers = 4  # more workers speed up image loading
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
        threshold=0.0001,
        threshold_mode="abs",
        cooldown=0,
        min_lr=1e-8,
        eps=1e-08,
    )
    device = DEVICE




## === cell 13
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
dataset["fold"] = -1

gkf = GroupKFold(n_splits=N_SPLITS)
for fold_number, (train_idx, val_idx) in enumerate(
    gkf.split(X=dataset.index, y=dataset["label"], groups=dataset["image_name"])
):
    dataset.loc[dataset.iloc[val_idx].index, "fold"] = fold_number




## === cell 14
fold_number = 0

validation_dataset = DatasetRetriever(
    kinds=dataset[dataset["fold"] == fold_number].kind.values,
    image_names=dataset[dataset["fold"] == fold_number].image_name.values,
    labels=dataset[dataset["fold"] == fold_number].label.values,
    transforms=get_valid_transforms(),
)

val_loader = DataLoader(
    validation_dataset,
    sampler=SequentialSampler(validation_dataset),
    batch_size=Config.batch_size,
    num_workers=Config.num_workers,
    pin_memory=True,
    persistent_workers=True,
    shuffle=False,
    drop_last=False,
)

MAX_TRAIN_SAMPLES = 50000
train_indices = np.where(dataset["fold"] != fold_number)[0]
if len(train_indices) > MAX_TRAIN_SAMPLES:
    np.random.seed(SEED)
    train_indices = np.random.choice(train_indices, MAX_TRAIN_SAMPLES, replace=False)

train_dataset = DatasetRetriever(
    kinds=dataset.iloc[train_indices].kind.values,
    image_names=dataset.iloc[train_indices].image_name.values,
    labels=dataset.iloc[train_indices].label.values,
    transforms=get_train_transforms(),
)

train_loader = DataLoader(
    train_dataset,
    sampler=RandomSampler(train_dataset),
    batch_size=Config.batch_size,
    num_workers=Config.num_workers,
    pin_memory=True,
    persistent_workers=True,
    drop_last=True,
)




## === cell 15
class TrainingSession:
    """
    Handles training / checkpoint loading.
    If an external pretrained checkpoint exists, it is loaded and training is skipped.
    Otherwise a short training run (few epochs on a sampled subset) is performed.
    """

    def __init__(
        self,
        model,
        config,
        train_loader,
        val_loader,
        ckpt_folder="/kaggle/input/alaska-checkpoint",
        output_log_path="/kaggle/working/log.txt",
        external_ckpt_path="/kaggle/input/alaska-checkpoint/best-checkpoint-010epoch.bin",
    ):
        self.model = model
        self.device = config.device
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.ckpt_folder = ckpt_folder
        self.output_log_path = output_log_path
        self.external_ckpt_path = external_ckpt_path

    def append_previous_log(self):
        prev_log = os.path.join(self.ckpt_folder, "log.txt")
        if os.path.exists(prev_log):
            with open(prev_log, "r") as f:
                old_content = f.read()
            with open(self.output_log_path, "a+") as f:
                f.write("\n\n# ==== Previous log ====\n")
                f.write(old_content)
                f.write("\n\n# ==== New session start ====\n")
            print(f"📜 Appended previous log from {prev_log}")
        else:
            print(f"⚠️ No previous log at {prev_log}")

    def load_external_checkpoint(self):
        if os.path.exists(self.external_ckpt_path):
            checkpoint = torch.load(self.external_ckpt_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            print("✅ Loaded external pretrained checkpoint; skipping training.")
            return True
        return False

    def run(self):
        if self.load_external_checkpoint():
            return

        print("⚠️ No external checkpoint found – starting short training.")
        fitter = Fitter(self.model, self.device, self.config)
        fitter.fit(self.train_loader, self.val_loader)
        print("✅ Training finished; using the newly trained model.")




## === cell 16
session = TrainingSession(model, Config, train_loader, val_loader)
session.run()




## === cell 17
sample_sub_path = os.path.join(PATH, "sample_submission.csv")
test_ids = pd.read_csv(sample_sub_path)["Id"].values

test_dataset = DatasetRetriever(
    kinds=np.array(["Test"] * len(test_ids)),
    image_names=test_ids,
    labels=np.zeros(len(test_ids), dtype=int),
    transforms=get_valid_transforms(),
)

test_loader = DataLoader(
    test_dataset,
    sampler=SequentialSampler(test_dataset),
    batch_size=Config.batch_size,
    num_workers=Config.num_workers,
    pin_memory=True,
    persistent_workers=True,
    shuffle=False,
    drop_last=False,
)

model.eval()

preds = []
with torch.no_grad():
    for imgs, _ in test_loader:
        imgs = imgs.to(DEVICE).float()
        logits = model(imgs)
        probs = F.softmax(logits, dim=1)
        stego_prob = probs[:, 1:].sum(dim=1)  # sum probabilities of hidden‑data classes
        preds.append(stego_prob.cpu().numpy())

preds = np.concatenate(preds)

submission = pd.DataFrame({"Id": test_ids, "Label": preds})
submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"✅ Submission file written to {submission_path}")
