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



## === cell 1
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



## === cell 2
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler
import torch.nn as nn
import torch.nn.functional as F
import timm



## === cell 3
from sklearn import metrics
from sklearn.model_selection import GroupKFold
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2



## === cell 4
PATH = "/kaggle/input/alaska2-image-steganalysis"
SAMPLE_SUB_PATH = os.path.join(PATH, "sample_submission.csv")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)



## === cell 5
SEED = 42


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(SEED)




## === cell 6
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
            int(self.labels[index]),
        )
        image = cv2.imread(f"{PATH}/{kind}/{image_name}", cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {PATH}/{kind}/{image_name}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

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




## === cell 7
CLASSES = ["Cover", "JMiPOD", "JUNIWARD", "UERD"]
N_SPLITS = 5

kinds_list = []
names_list = []
labels_list = []

for label, kind in enumerate(CASSES := CLASSES):
    image_paths = glob(os.path.join(PATH, kind, "*.jpg"))
    basenames = [os.path.basename(p) for p in image_paths]
    kinds_list.extend([kind] * len(basenames))
    names_list.extend(basenames)
    labels_list.extend([label] * len(basenames))

kinds_arr = np.array(kinds_list, dtype=object)
names_arr = np.array(names_list, dtype=object)
labels_arr = np.array(labels_list, dtype=np.int64)

rng = np.random.default_rng(SEED)
perm = rng.permutation(len(names_arr))
kinds_arr = kinds_arr[perm]
names_arr = names_arr[perm]
labels_arr = labels_arr[perm]

folds = np.zeros(len(names_arr), dtype=np.int64)
gkf = GroupKFold(n_splits=N_SPLITS)
for fold_number, (_, val_index) in enumerate(
    gkf.split(X=np.arange(len(names_arr)), y=labels_arr, groups=names_arr)
):
    folds[val_index] = fold_number

dataset_df = pd.DataFrame(
    {"kind": kinds_arr, "image_name": names_arr, "label": labels_arr, "fold": folds}
)
print(dataset_df.head())
print("Fold counts:", dataset_df["fold"].value_counts().to_dict())



## === cell 8
IMG_SIZE = 224


def get_train_transforms():
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(height=IMG_SIZE, width=IMG_SIZE, p=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )


def get_valid_transforms():
    return A.Compose(
        [
            A.Resize(height=IMG_SIZE, width=IMG_SIZE, p=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )




## === cell 9
class AverageMeter(object):
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
        self.avg = self.sum / max(self.count, 1)


def alaska_weighted_auc(y_true, y_valid):
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2, 1]

    fpr, tpr, _ = metrics.roc_curve(y_true, y_valid, pos_label=1)

    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])
    normalization = np.dot(areas, weights)

    competition_metric = 0.0
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

        score = metrics.auc(x, y)
        competition_metric += score * weight

    return competition_metric / normalization


class RocAucMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.y_true = np.array([0, 1])
        self.y_pred = np.array([0.5, 0.5])
        self.score = 0

    def update(self, y_true, y_pred):
        y_true = (
            y_true.detach().cpu().numpy().argmax(axis=1).clip(min=0, max=1).astype(int)
        )
        y_pred = 1 - nn.functional.softmax(y_pred, dim=1).detach().cpu().numpy()[:, 0]
        self.y_true = np.hstack((self.y_true, y_true))
        self.y_pred = np.hstack((self.y_pred, y_pred))
        self.score = alaska_weighted_auc(self.y_true, self.y_pred)

    @property
    def avg(self):
        return self.score




## === cell 10
class LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = target.float()
            logprobs = F.log_softmax(x, dim=-1)
            nll_loss = (-logprobs * target).sum(-1)
            smooth_loss = (-logprobs).mean(dim=-1)
            loss = self.confidence * nll_loss + self.smoothing * smooth_loss
            return loss.mean()
        else:
            target_idx = target.argmax(dim=1)
            return F.cross_entropy(x, target_idx)




## === cell 11
class Fitter:
    def __init__(self, model, device, config):
        self.config = config
        self.epoch = 0
        self.base_dir = "./"
        self.log_path = f"{self.base_dir}/log.txt"
        self.best_summary_loss = 10**5

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
                f"[RESULT]: Train. Epoch: {self.epoch},summary_loss: {summary_loss.avg:.5f},final_score: {final_scores.avg:.5f},time: {(time.time() - t):.5f}"
            )
            self.save(f"{self.base_dir}/last-checkpoint.bin")

            t = time.time()
            summary_loss, final_scores = self.validation(validation_loader)
            self.log(
                f"[RESULT]: Val. Epoch: {self.epoch},summary_loss: {summary_loss.avg:.5f},final_score: {final_scores.avg:.5f},time: {(time.time() - t):.5f}"
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
                    try:
                        os.remove(path)
                    except OSError:
                        pass

            if self.config.validation_scheduler:
                self.scheduler.step(metrics=summary_loss.avg)
            self.epoch += 1

    def validation(self, val_loader):
        self.model.eval()
        summary_loss = AverageMeter()
        final_scores = RocAucMeter()
        t = time.time()

        y_true_all = []
        y_pred_all = []

        for step, (images, targets) in enumerate(val_loader):
            if self.config.verbose and step % self.config.verbose_step == 0:
                print(
                    f"Val Step {step}/{len(val_loader)}, summary_loss: {summary_loss.avg:.5f}, final_score: {final_scores.avg:.5f}, time: {(time.time() - t):.5f}",
                    end="\r",
                )

            with torch.no_grad():
                targets = targets.to(self.device, non_blocking=True).float()
                images = images.to(self.device, non_blocking=True).float()
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

                y_true_all.append(targets.detach().cpu())
                y_pred_all.append(outputs.detach().cpu())
                summary_loss.update(loss.detach().item(), images.shape[0])

        y_true_cat = torch.cat(y_true_all, dim=0)
        y_pred_cat = torch.cat(y_pred_all, dim=0)
        final_scores.update(y_true_cat, y_pred_cat)

        print()
        return summary_loss, final_scores

    def train_model(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter()
        final_scores = RocAucMeter()
        t = time.time()

        y_true_all = []
        y_pred_all = []

        for step, (images, targets) in enumerate(train_loader):
            if self.config.verbose and step % self.config.verbose_step == 0:
                print(
                    f"Train Step {step}/{len(train_loader)}, summary_loss: {summary_loss.avg:.5f}, final_score: {final_scores.avg:.5f}, time: {(time.time() - t):.5f}",
                    end="\r",
                )

            targets = targets.to(self.device, non_blocking=True).float()
            images = images.to(self.device, non_blocking=True).float()

            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            loss.backward()

            summary_loss.update(loss.detach().item(), images.shape[0])
            self.optimizer.step()

            y_true_all.append(targets.detach().cpu())
            y_pred_all.append(outputs.detach().cpu())

            if self.config.metric_step and ((step + 1) % self.config.metric_step == 0):
                y_true_cat = torch.cat(y_true_all, dim=0)
                y_pred_cat = torch.cat(y_pred_all, dim=0)
                final_scores.update(y_true_cat, y_pred_cat)

            if self.config.step_scheduler:
                self.scheduler.step()

        y_true_cat = torch.cat(y_true_all, dim=0)
        y_pred_cat = torch.cat(y_pred_all, dim=0)
        final_scores.update(y_true_cat, y_pred_cat)

        print()
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
        self.log(f"Loaded checkpoint from {path}, resume from epoch {self.epoch}")

    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, "a+") as logger:
            logger.write(f"{message}\n")




## === cell 12
class EffNet(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
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




## === cell 13
model = EffNet(4).to(DEVICE)




## === cell 14
class Config:
    batch_size = 8 if DEVICE.type == "cuda" else 4
    n_epochs = 1  # keep as provided
    num_workers = min(8, os.cpu_count() or 2) if DEVICE.type == "cuda" else 0
    lr = 0.001
    verbose = True
    verbose_step = 50
    metric_step = 500  # logging-only; does not affect training
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




## === cell 15
fold_number = 0

train_dataset = DatasetRetriever(
    kinds=dataset_df[dataset_df["fold"] != fold_number].kind.values,
    image_names=dataset_df[dataset_df["fold"] != fold_number].image_name.values,
    labels=dataset_df[dataset_df["fold"] != fold_number].label.values,
    transforms=get_train_transforms(),
)

validation_dataset = DatasetRetriever(
    kinds=dataset_df[dataset_df["fold"] == fold_number].kind.values,
    image_names=dataset_df[dataset_df["fold"] == fold_number].image_name.values,
    labels=dataset_df[dataset_df["fold"] == fold_number].label.values,
    transforms=get_valid_transforms(),
)

loader_kwargs = dict(
    batch_size=Config.batch_size,
    num_workers=Config.num_workers,
    pin_memory=(DEVICE.type == "cuda"),
    drop_last=True,
)
if Config.num_workers > 0:
    loader_kwargs.update(dict(persistent_workers=True, prefetch_factor=4))

train_loader = DataLoader(
    train_dataset,
    shuffle=True,
    **loader_kwargs,
)

val_loader_kwargs = dict(
    batch_size=Config.batch_size,
    num_workers=Config.num_workers,
    pin_memory=(DEVICE.type == "cuda"),
    drop_last=False,
)
if Config.num_workers > 0:
    val_loader_kwargs.update(dict(persistent_workers=True, prefetch_factor=4))

val_loader = DataLoader(
    validation_dataset,
    sampler=SequentialSampler(validation_dataset),
    shuffle=False,
    **val_loader_kwargs,
)



## === cell 16
fitter = Fitter(model, DEVICE, Config)
fitter.fit(train_loader, val_loader)




## === cell 17
def get_test_transforms(mode):
    if mode == 0:
        return A.Compose(
            [A.Resize(height=IMG_SIZE, width=IMG_SIZE, p=1.0), ToTensorV2(p=1.0)], p=1.0
        )
    elif mode == 1:
        return A.Compose(
            [
                A.HorizontalFlip(p=1),
                A.Resize(height=IMG_SIZE, width=IMG_SIZE, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        )
    elif mode == 2:
        return A.Compose(
            [
                A.VerticalFlip(p=1),
                A.Resize(height=IMG_SIZE, width=IMG_SIZE, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        )
    else:
        return A.Compose(
            [
                A.HorizontalFlip(p=1),
                A.VerticalFlip(p=1),
                A.Resize(height=IMG_SIZE, width=IMG_SIZE, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        )


class DatasetSubmissionRetriever(Dataset):
    def __init__(self, image_names, transforms=None):
        super().__init__()
        self.image_names = image_names
        self.transforms = transforms

    def __getitem__(self, index: int):
        image_name = self.image_names[index]
        image = cv2.imread(f"{PATH}/Test/{image_name}", cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(
                f"Could not read test image: {PATH}/Test/{image_name}"
            )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        if self.transforms:
            sample = {"image": image}
            sample = self.transforms(**sample)
            image = sample["image"]
        return image_name, image

    def __len__(self) -> int:
        return self.image_names.shape[0]




## === cell 18
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
test_image_names = sample_sub["Id"].values

model.eval()
results = []

test_bs = 32 if DEVICE.type == "cuda" else 8
test_num_workers = min(8, os.cpu_count() or 2) if DEVICE.type == "cuda" else 0

for mode in range(0, 4):
    ds = DatasetSubmissionRetriever(
        image_names=test_image_names, transforms=get_test_transforms(mode)
    )
    dl_kwargs = dict(
        batch_size=test_bs,
        shuffle=False,
        num_workers=test_num_workers,
        drop_last=False,
        pin_memory=(DEVICE.type == "cuda"),
    )
    if test_num_workers > 0:
        dl_kwargs.update(dict(persistent_workers=True, prefetch_factor=4))
    dl = DataLoader(ds, **dl_kwargs)

    result = {"Id": [], "Label": []}
    with torch.no_grad():
        for step, (image_names, images) in enumerate(dl):
            if step % 50 == 0:
                print(f"TTA mode {mode} - step {step}/{len(dl)}", end="\r")
            images = images.to(DEVICE, non_blocking=True).float()
            y_pred = model(images)
            y_pred = (
                1 - nn.functional.softmax(y_pred, dim=1).detach().cpu().numpy()[:, 0]
            )
            result["Id"].extend(list(image_names))
            result["Label"].extend(list(y_pred))
    print()
    results.append(pd.DataFrame(result))



## === cell 19
weight0, weight1, weight2, weight3 = 5, 1, 1, 1
weight = weight0 + weight1 + weight2 + weight3

for i in range(4):
    results[i] = results[i].set_index("Id").loc[test_image_names].reset_index()

blend = results[0].copy()
blend["Label"] = (
    results[0]["Label"] * weight0
    + results[1]["Label"] * weight1
    + results[2]["Label"] * weight2
    + results[3]["Label"] * weight3
) / weight

submission = sample_sub[["Id"]].merge(blend[["Id", "Label"]], on="Id", how="left")
assert submission.shape[0] == sample_sub.shape[0]
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
