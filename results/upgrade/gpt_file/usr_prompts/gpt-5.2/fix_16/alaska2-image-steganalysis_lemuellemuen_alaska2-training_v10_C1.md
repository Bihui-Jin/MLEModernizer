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
import os, sys, subprocess



## === cell 1
import importlib.util


def _safe_pip_install(pkg):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
    except Exception as e:
        print(f"[WARN] pip install failed for {pkg}: {e}")


HAS_TORCHSAMPLER = importlib.util.find_spec("torchsampler") is not None



## === cell 2
import warnings

warnings.filterwarnings("ignore", category=ResourceWarning)



## === cell 3
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import os
import cv2
import random
import time
from datetime import datetime



## === cell 4
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import (
    SequentialSampler,
    RandomSampler,
    WeightedRandomSampler,
)
import torch.nn as nn
import torch.nn.functional as F

if HAS_TORCHSAMPLER:
    from torchsampler import ImbalancedDatasetSampler
else:
    ImbalancedDatasetSampler = None



## === cell 5
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.feature import hog
from sklearn import metrics
from sklearn.model_selection import GroupKFold
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2



## === cell 6
PATH = "/kaggle/input/alaska2-image-steganalysis"



## === cell 7
CKPT_DIR = "/kaggle/working/ckpt"
OUT_DIR = "/kaggle/working/outputs"
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

try:
    cv2.setUseOptimized(True)
    cv2.setNumThreads(max(1, (os.cpu_count() or 4) // 2))
except Exception:
    pass

if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass



## === cell 8
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
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)




## === cell 9
def seed_worker(worker_id: int):
    worker_seed = (SEED + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


g_dl = torch.Generator()
g_dl.manual_seed(SEED)



## === cell 10
pass



## === cell 11
CLASSES = ["Cover", "JMiPOD", "JUNIWARD", "UERD"]
N_SPLITS = 5

SPLIT_CACHE_PATH = os.path.join(
    OUT_DIR, f"dataset_folds_seed{SEED}_splits{N_SPLITS}.parquet"
)

if os.path.exists(SPLIT_CACHE_PATH):
    dataset = pd.read_parquet(SPLIT_CACHE_PATH)
else:
    frames = []
    for label, kind in enumerate(CLASSES):
        paths = glob(os.path.join(PATH, kind, "*.jpg"))
        img_names = [os.path.basename(p) for p in paths]
        frames.append(
            pd.DataFrame(
                {"kind": kind, "image_name": img_names, "label": np.int32(label)}
            )
        )
    dataset = pd.concat(frames, ignore_index=True)

    dataset = dataset.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    dataset.loc[:, "fold"] = 0
    gkf = GroupKFold(n_splits=N_SPLITS)
    for fold_number, (train_index, val_index) in enumerate(
        gkf.split(X=dataset.index, y=dataset["label"], groups=dataset["image_name"])
    ):
        dataset.loc[dataset.iloc[val_index].index, "fold"] = fold_number

    dataset.to_parquet(SPLIT_CACHE_PATH, index=False)



## === cell 12
pass



## === cell 13
IMG_SIZE = 256


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




## === cell 14
pass



## === cell 15
_ONEHOT_CACHE = torch.eye(4, dtype=torch.float32)


def onehot(size, target):
    if size == 4:
        return _ONEHOT_CACHE[int(target)].clone()
    vec = torch.zeros(size, dtype=torch.float32)
    vec[int(target)] = 1.0
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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        if self.transforms:
            sample = {"image": image}
            sample = self.transforms(**sample)
            image = sample["image"]

        target = onehot(4, int(label))
        return image, target

    def __len__(self) -> int:
        return self.image_names.shape[0]

    def get_labels(self):
        return list(self.labels)




## === cell 16
pass



## === cell 17
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




## === cell 18
class AverageMeter(object):
    """Computes and stores the average and current value"""

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




## === cell 19
class RocAucMeter(object):
    """
    Competition is binary: cover(0) vs stego(1). We compute prob(stego) as 1 - P(cover).
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.y_true = None
        self.y_pred = None
        self._pos = 0
        self.score = 0.0

    def init_storage(self, n_samples: int):
        self.y_true = np.empty((n_samples,), dtype=np.int32)
        self.y_pred = np.empty((n_samples,), dtype=np.float32)
        self._pos = 0

    def update(self, y_true, y_pred):
        yt = y_true.detach().argmax(dim=1)
        yt = (yt != 0).to(torch.int32).cpu().numpy()

        p0 = torch.softmax(y_pred, dim=1)[:, 0]
        yp = (1.0 - p0).detach().to(torch.float32).cpu().numpy()

        b = yt.shape[0]
        self.y_true[self._pos : self._pos + b] = yt
        self.y_pred[self._pos : self._pos + b] = yp
        self._pos += b

    def compute(self):
        if self.y_true is None or self._pos == 0:
            self.score = 0.0
            return self.score
        self.score = alaska_weighted_auc(
            self.y_true[: self._pos], self.y_pred[: self._pos]
        )
        return self.score

    @property
    def avg(self):
        return float(self.score)




## === cell 20
def alaska_weighted_auc(y_true, y_valid):
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2, 1]

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_valid, pos_label=1)

    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])
    normalization = np.dot(areas, weights)

    competition_metric = 0.0
    for idx, weight in enumerate(weights):
        y_min = tpr_thresholds[idx]
        y_max = tpr_thresholds[idx + 1]

        mask = (y_min <= tpr) & (tpr <= y_max)

        if np.sum(mask) < 2:
            continue

        x_padding = np.linspace(fpr[mask][-1], 1, 100)
        x = np.concatenate([fpr[mask], x_padding])
        y = np.concatenate([tpr[mask], [y_max] * len(x_padding)])
        y = y - y_min

        score = metrics.auc(x, y)
        competition_metric += score * weight

    return float(competition_metric / normalization)




## === cell 21
pass




## === cell 22
class LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1, class_priors=None):
        """
        Change (score improvement): keep the same LabelSmoothing loss family, but allow
        smoothing to distribute mass according to observed class priors instead of uniform.
        This tends to improve probability calibration (important for AUC) without changing
        the model/training loop structure.
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - float(smoothing)
        self.smoothing = float(smoothing)

        if class_priors is None:
            self.register_buffer("class_priors", torch.empty(0), persistent=False)
        else:
            pri = torch.as_tensor(class_priors, dtype=torch.float32)
            pri = pri / pri.sum().clamp_min(1e-12)
            self.register_buffer("class_priors", pri, persistent=False)

    def forward(self, x, target):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs * target
        nll_loss = nll_loss.sum(-1)

        if self.class_priors.numel() == target.shape[-1]:
            smooth_loss = -(logprobs * self.class_priors).sum(dim=-1)
        else:
            smooth_loss = -logprobs.mean(dim=-1)

        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()




## === cell 23
pass




## === cell 24
class Fitter:
    def __init__(self, model, device, config):
        self.config = config
        self.epoch = 0

        self.base_dir = "./"
        self.log_path = f"{CKPT_DIR}/log.txt"
        self.best_summary_loss = 10**5

        self.model = model
        self.device = device
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)
        self.scheduler = config.SchedulerClass(
            self.optimizer, **config.scheduler_params
        )

        self.criterion = LabelSmoothing(
            smoothing=getattr(config, "label_smoothing", 0.1),
            class_priors=getattr(config, "class_priors", None),
        ).to(self.device)

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

            self.save(f"{CKPT_DIR}/last-checkpoint.bin")

            t = time.time()
            summary_loss, final_scores = self.validation(validation_loader)
            self.log(
                f"[RESULT]: Val. Epoch: {self.epoch},summary_loss: {summary_loss.avg:.5f},final_score: {final_scores.avg:.5f},time: {(time.time() - t):.5f}"
            )
            if summary_loss.avg < self.best_summary_loss:
                self.best_summary_loss = summary_loss.avg
                self.model.eval()
                self.save(
                    f"{CKPT_DIR}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin"
                )
                best_paths = sorted(glob(f"{CKPT_DIR}/best-checkpoint-*epoch.bin"))
                for path in best_paths[:-3]:
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
        final_scores.init_storage(len(val_loader.dataset))

        t = time.time()
        for step, (images, targets) in enumerate(val_loader):
            if self.config.verbose and (step % self.config.verbose_step == 0):
                print(
                    f"Val Step {step}/{len(val_loader)}, "
                    + f"summary_loss: {summary_loss.avg:.5f}, final_score: {final_scores.avg:.5f}, "
                    + f"time: {(time.time() - t):.5f}",
                    end="\r",
                )
            with torch.no_grad():
                targets = targets.to(self.device, non_blocking=True).float()
                batch_size = images.shape[0]
                images = images.to(self.device, non_blocking=True).float()
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                final_scores.update(targets, outputs)
                summary_loss.update(loss.detach().item(), batch_size)

        final_scores.compute()
        if self.config.verbose:
            print()
        return summary_loss, final_scores

    def train_model(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter()
        final_scores = RocAucMeter()
        final_scores.init_storage(len(train_loader) * train_loader.batch_size)

        t = time.time()
        for step, (images, targets) in enumerate(train_loader):
            if self.config.verbose and (step % self.config.verbose_step == 0):
                print(
                    f"Train Step {step}/{len(train_loader)}, "
                    + f"summary_loss: {summary_loss.avg:.5f}, final_score: {final_scores.avg:.5f}, "
                    + f"time: {(time.time() - t):.5f}",
                    end="\r",
                )

            targets = targets.to(self.device, non_blocking=True).float()
            images = images.to(self.device, non_blocking=True).float()
            batch_size = images.shape[0]

            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            loss.backward()

            final_scores.update(targets, outputs)
            summary_loss.update(loss.detach().item(), batch_size)

            self.optimizer.step()

            if self.config.step_scheduler:
                self.scheduler.step()

        final_scores.compute()
        if self.config.verbose:
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
            _use_new_zipfile_serialization=True,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_summary_loss = checkpoint.get(
            "best_summary_loss", self.best_summary_loss
        )
        self.epoch = checkpoint.get("epoch", 0) + 1

    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, "a+", encoding="utf-8") as logger:
            logger.write(f"{message}\n")




## === cell 25
pass



## === cell 26
import re


def parse_log_file(log_path):
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    epochs = []
    train_loss = []
    train_score = []
    train_time = []

    val_loss = []
    val_score = []
    val_time = []

    lr_list = []

    current_lr = None
    for line in lines:
        if line.startswith("LR:"):
            current_lr = float(line.strip().split(":")[1])
        elif "[RESULT]: Train." in line:
            epoch = int(re.search(r"Epoch: (\d+)", line).group(1))
            summary_loss = float(re.search(r"summary_loss: ([\d.]+)", line).group(1))
            final_score = float(re.search(r"final_score: ([\d.]+)", line).group(1))
            t_time = float(re.search(r"time: ([\d.]+)", line).group(1))

            epochs.append(epoch)
            train_loss.append(summary_loss)
            train_score.append(final_score)
            train_time.append(t_time)
            lr_list.append(current_lr)

        elif "[RESULT]: Val." in line:
            val_summary_loss = float(
                re.search(r"summary_loss: ([\d.]+)", line).group(1)
            )
            val_final_score = float(re.search(r"final_score: ([\d.]+)", line).group(1))
            val_t_time = float(re.search(r"time: ([\d.]+)", line).group(1))

            val_loss.append(val_summary_loss)
            val_score.append(val_final_score)
            val_time.append(val_t_time)

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




## === cell 27
def plot_log_results(metrics, save_path="log_plots.png"):
    epochs = metrics["epochs"]
    if len(epochs) == 0:
        print("[WARN] No epochs found in log, skip plotting.")
        return

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training Metrics from Log File", fontsize=16)

    axs[0, 0].plot(epochs, metrics["train_loss"], label="Train Loss", marker="o")
    axs[0, 0].plot(epochs, metrics["val_loss"], label="Val Loss", marker="x")
    axs[0, 0].set_title("Loss per Epoch")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].legend()

    axs[0, 1].plot(epochs, metrics["train_score"], label="Train Score", marker="o")
    axs[0, 1].plot(epochs, metrics["val_score"], label="Val Score", marker="x")
    axs[0, 1].set_title("Score per Epoch")
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].set_ylabel("Score")
    axs[0, 1].legend()

    axs[1, 0].plot(epochs, metrics["lr"], label="Learning Rate", marker="o")
    axs[1, 0].set_title("Learning Rate")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].set_ylabel("LR")
    axs[1, 0].legend()

    axs[1, 1].plot(epochs, metrics["train_time"], label="Train Time (s)", marker="o")
    axs[1, 1].plot(epochs, metrics["val_time"], label="Val Time (s)", marker="x")
    axs[1, 1].set_title("Time per Epoch")
    axs[1, 1].set_xlabel("Epoch")
    axs[1, 1].set_ylabel("Seconds")
    axs[1, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plots to: {save_path}")




## === cell 28
pass



## === cell 29
import timm


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




## === cell 30
pass



## === cell 31
model = EffNet(4).to(DEVICE)



## === cell 32
pass




## === cell 33
class Config:
    batch_size = 16

    n_epochs = 3

    num_workers = min(8, (os.cpu_count() or 4))

    lr = 0.001
    verbose = True
    verbose_step = 200
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

    label_smoothing = 0.1
    class_priors = None


TRAIN_PER_CLASS = 12000  # 48000 total
VAL_PER_CLASS = 3000  # 12000 total




## === cell 34
def make_imbalanced_sampler(ds: Dataset):
    labels = np.asarray(ds.get_labels(), dtype=np.int64)
    if ImbalancedDatasetSampler is not None:
        return ImbalancedDatasetSampler(ds, labels=labels.tolist())

    class_counts = np.bincount(labels, minlength=int(labels.max()) + 1).astype(
        np.float64
    )
    class_counts[class_counts == 0] = 1.0
    weights = 1.0 / class_counts[labels]
    weights = torch.as_tensor(weights, dtype=torch.double)

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
        generator=g_dl,
    )




## === cell 35
import concurrent.futures as _fut

TRAIN_CACHE_DIR = f"/kaggle/working/train_cache_{IMG_SIZE}"
os.makedirs(TRAIN_CACHE_DIR, exist_ok=True)


def _fold_cache_paths(fold: int):
    base = os.path.join(TRAIN_CACHE_DIR, f"fold{fold}_seed{SEED}_img{IMG_SIZE}")
    return (
        base + "_train_images.npy",
        base + "_train_meta.npz",
        base + "_val_images.npy",
        base + "_val_meta.npz",
    )


def _stratified_cap(df: pd.DataFrame, per_class: int) -> pd.DataFrame:
    parts = []
    for lbl in range(4):
        d = df[df["label"] == lbl]
        if len(d) > per_class:
            d = d.sample(n=per_class, random_state=SEED)
        parts.append(d)
    out = pd.concat(parts, ignore_index=True)
    out = out.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return out


def _read_resize_chw(kind: str, image_name: str):
    img = cv2.imread(f"{PATH}/{kind}/{image_name}", cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    return np.transpose(img, (2, 0, 1))


def _build_memmap_for_df(df: pd.DataFrame, out_npy: str, out_meta: str, desc: str):
    image_names = df["image_name"].astype(str).values
    kinds = df["kind"].astype(str).values
    labels = df["label"].astype(np.int32).values
    N = len(df)

    need_build = True
    if os.path.exists(out_npy) and os.path.exists(out_meta):
        meta = np.load(out_meta, allow_pickle=False)
        if (
            meta["image_names"].shape == image_names.shape
            and np.array_equal(meta["image_names"], image_names)
            and np.array_equal(meta["kinds"], kinds)
            and np.array_equal(meta["labels"], labels)
            and int(meta.get("img_size", -1)) == int(IMG_SIZE)
        ):
            need_build = False

    if need_build:
        t0 = time.time()
        mm = np.lib.format.open_memmap(
            out_npy, mode="w+", dtype=np.float32, shape=(N, 3, IMG_SIZE, IMG_SIZE)
        )

        max_workers = min(8, max(1, (os.cpu_count() or 4) // 2))
        with _fut.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [
                ex.submit(_read_resize_chw, kinds[i], image_names[i]) for i in range(N)
            ]
            for i, fu in enumerate(tqdm(futures, desc=desc, mininterval=1.0)):
                arr = fu.result()
                if arr is None:
                    mm[i].fill(0.0)
                else:
                    mm[i] = arr

        mm.flush()
        np.savez_compressed(
            out_meta,
            image_names=image_names,
            kinds=kinds,
            labels=labels,
            img_size=np.int32(IMG_SIZE),
        )
        print(f"[CACHE] Built {desc}: {N} images in {time.time()-t0:.1f}s")

    mm = np.load(out_npy, mmap_mode="r")
    return image_names, labels, mm


class TrainValMemmapDataset(Dataset):
    def __init__(self, image_names, labels, mm_array, train: bool):
        super().__init__()
        self.image_names = image_names
        self.labels = labels.astype(np.int32, copy=False)
        self.mm = mm_array  # numpy memmap [N,3,H,W] float32
        self.train = bool(train)

    def __len__(self):
        return self.image_names.shape[0]

    def __getitem__(self, idx: int):
        img = torch.from_numpy(self.mm[idx])  # CHW float32
        if self.train:
            if torch.rand((), generator=g_dl).item() < 0.5:
                img = torch.flip(img, dims=(2,))
            if torch.rand((), generator=g_dl).item() < 0.5:
                img = torch.flip(img, dims=(1,))
        target = _ONEHOT_CACHE[int(self.labels[idx])]
        return img, target

    def get_labels(self):
        return list(self.labels)


train_df_full = dataset[dataset["fold"] != fold_number].reset_index(drop=True)
val_df_full = dataset[dataset["fold"] == fold_number].reset_index(drop=True)

train_df = _stratified_cap(train_df_full, TRAIN_PER_CLASS)
val_df = _stratified_cap(val_df_full, VAL_PER_CLASS)

_counts = (
    train_df["label"]
    .value_counts()
    .reindex(range(4), fill_value=0)
    .values.astype(np.float32)
)
Config.class_priors = (_counts / max(_counts.sum(), 1.0)).tolist()
print("[INFO] Train class priors (used for label smoothing):", Config.class_priors)

train_npy, train_meta, val_npy, val_meta = _fold_cache_paths(fold_number)
train_image_names, train_labels, train_mm = _build_memmap_for_df(
    train_df,
    train_npy,
    train_meta,
    desc=f"Building TRAIN memmap cache ({IMG_SIZE} resize)",
)
val_image_names, val_labels, val_mm = _build_memmap_for_df(
    val_df, val_npy, val_meta, desc=f"Building VAL memmap cache ({IMG_SIZE} resize)"
)

train_dataset = TrainValMemmapDataset(
    train_image_names, train_labels, train_mm, train=True
)
validation_dataset = TrainValMemmapDataset(
    val_image_names, val_labels, val_mm, train=False
)



## === cell 36
train_loader = DataLoader(
    train_dataset,
    sampler=RandomSampler(train_dataset, generator=g_dl),
    batch_size=Config.batch_size,
    num_workers=Config.num_workers,
    pin_memory=torch.cuda.is_available(),
    drop_last=True,
    persistent_workers=(Config.num_workers > 0),
    prefetch_factor=8 if Config.num_workers > 0 else None,
    worker_init_fn=seed_worker,
    generator=g_dl,
)
val_loader = DataLoader(
    validation_dataset,
    sampler=SequentialSampler(validation_dataset),
    batch_size=Config.batch_size,
    num_workers=Config.num_workers,
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(Config.num_workers > 0),
    prefetch_factor=8 if Config.num_workers > 0 else None,
    worker_init_fn=seed_worker,
    generator=g_dl,
)



## === cell 37
pass



## === cell 38
ckpt_path = f"{CKPT_DIR}/last-checkpoint.bin"




## === cell 39
class TrainingSession:
    def __init__(self, model, config, train_loader, val_loader):
        self.model = model
        self.device = DEVICE
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.ckpt_path = ckpt_path

    def run(self):
        fitter = Fitter(self.model, self.device, self.config)
        if os.path.exists(self.ckpt_path):
            print(f"Resume from checkpoint: {self.ckpt_path}")
            fitter.load(self.ckpt_path)
        else:
            print("Start new training...")
        fitter.fit(self.train_loader, self.val_loader)




## === cell 40
pass



## === cell 41
session = TrainingSession(model, Config, train_loader, val_loader)
session.run()



## === cell 42
pass



## === cell 43
best_ckpts = sorted(glob(f"{CKPT_DIR}/best-checkpoint-*epoch.bin"))
checkpoint = None
if len(best_ckpts) > 0:
    best_path = best_ckpts[-1]
    print("Loading best checkpoint:", best_path)
    checkpoint = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
else:
    last_path = f"{CKPT_DIR}/last-checkpoint.bin"
    if os.path.exists(last_path):
        print("Loading last checkpoint:", last_path)
        checkpoint = torch.load(last_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
model.eval()



## === cell 44
try:
    print("Checkpoint keys:", list(checkpoint.keys()))
except Exception as e:
    print("[INFO] No checkpoint dict loaded:", e)



## === cell 45
pass




## === cell 46
def get_test_transforms(mode):
    if mode == 0:
        return A.Compose(
            [
                A.Resize(height=IMG_SIZE, width=IMG_SIZE, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
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




## === cell 47
class DatasetSubmissionCached(Dataset):
    def __init__(self, image_names, base_tensor, mode: int):
        super().__init__()
        self.image_names = image_names
        self.base_tensor = base_tensor  # torch.FloatTensor [N,3,H,W]
        self.mode = int(mode)

    def __getitem__(self, index: int):
        name = self.image_names[index]
        img = self.base_tensor[index]
        if self.mode == 1:  # HFlip
            img = torch.flip(img, dims=(2,))
        elif self.mode == 2:  # VFlip
            img = torch.flip(img, dims=(1,))
        elif self.mode == 3:  # HVFlip
            img = torch.flip(img, dims=(1, 2))
        return name, img

    def __len__(self) -> int:
        return self.image_names.shape[0]




## === cell 48
TEST_CACHE_DIR = f"/kaggle/working/test_cache_{IMG_SIZE}"
os.makedirs(TEST_CACHE_DIR, exist_ok=True)


def _test_cache_npy_path():
    return os.path.join(
        TEST_CACHE_DIR, f"test_images_3x{IMG_SIZE}x{IMG_SIZE}_float32.npy"
    )


def _test_cache_meta_path():
    return os.path.join(TEST_CACHE_DIR, f"test_meta_img{IMG_SIZE}.npz")


test_image_names = np.array([os.path.basename(p) for p in glob(f"{PATH}/Test/*.jpg")])
test_image_names = np.sort(test_image_names)

npy_path = _test_cache_npy_path()
meta_path = _test_cache_meta_path()

need_build = True
if os.path.exists(npy_path) and os.path.exists(meta_path):
    meta = np.load(meta_path, allow_pickle=False)
    if meta["image_names"].shape == test_image_names.shape and np.array_equal(
        meta["image_names"], test_image_names
    ):
        need_build = False

if need_build:
    t0 = time.time()
    N = len(test_image_names)
    mm = np.lib.format.open_memmap(
        npy_path, mode="w+", dtype=np.float32, shape=(N, 3, IMG_SIZE, IMG_SIZE)
    )

    def _read_test_one(name: str):
        img = cv2.imread(f"{PATH}/Test/{name}", cv2.IMREAD_COLOR)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        return np.transpose(img, (2, 0, 1))

    max_workers = min(8, max(1, (os.cpu_count() or 4) // 2))
    with _fut.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_read_test_one, test_image_names[i]) for i in range(N)]
        for i, fu in enumerate(
            tqdm(futures, desc="Building test memmap cache", mininterval=1.0)
        ):
            arr = fu.result()
            if arr is None:
                mm[i].fill(0.0)
            else:
                mm[i] = arr

    mm.flush()
    np.savez_compressed(meta_path, image_names=test_image_names)
    print(f"[CACHE] Built test memmap: {N} images in {time.time()-t0:.1f}s")

test_mm = np.load(npy_path, mmap_mode="r")


class TestMemmapDataset(Dataset):
    def __init__(self, image_names, mm_array, mode: int):
        super().__init__()
        self.image_names = image_names
        self.mm = mm_array  # numpy memmap [N,3,H,W]
        self.mode = int(mode)

    def __len__(self):
        return self.image_names.shape[0]

    def __getitem__(self, idx: int):
        name = self.image_names[idx]
        img = torch.from_numpy(self.mm[idx])  # CHW float32, view of memmap
        if self.mode == 1:
            img = torch.flip(img, dims=(2,))
        elif self.mode == 2:
            img = torch.flip(img, dims=(1,))
        elif self.mode == 3:
            img = torch.flip(img, dims=(1, 2))
        return name, img


results = []
model.eval()

inf_workers = min(4, (os.cpu_count() or 4))
for mode in range(0, 4):
    ds = TestMemmapDataset(
        image_names=test_image_names,
        mm_array=test_mm,
        mode=mode,
    )

    data_loader = DataLoader(
        ds,
        batch_size=128,
        shuffle=False,
        num_workers=inf_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(inf_workers > 0),
        prefetch_factor=8 if inf_workers > 0 else None,
        worker_init_fn=seed_worker,
        generator=g_dl,
    )

    result = {"Id": [], "Label": []}
    with torch.inference_mode():
        for step, (image_names, images) in enumerate(data_loader):
            if step % 50 == 0:
                print(f"TTA mode {mode} - step {step}/{len(data_loader)}", end="\r")

            images = images.to(DEVICE, non_blocking=True).float()
            y_pred = model(images)

            p0 = torch.softmax(y_pred, dim=1)[:, 0]
            y_pred_np = (1.0 - p0).detach().cpu().numpy()

            result["Id"].extend(list(image_names))
            result["Label"].extend(list(y_pred_np))

    print()
    results.append(result)



## === cell 49
pass



## === cell 50
submissions = [pd.DataFrame(results[mode]) for mode in range(0, 4)]



## === cell 51
for mode in range(0, 4):
    submissions[mode].to_csv(f"/kaggle/working/submission_{mode}.csv", index=False)



## === cell 52
pass



## === cell 53
weight0 = 5
weight1 = 1
weight2 = 1
weight3 = 1
weight = weight0 + weight1 + weight2 + weight3



## === cell 54
for mode in range(0, 4):
    submissions[mode] = submissions[mode].sort_values("Id").reset_index(drop=True)

ensemble = submissions[0].copy()
ensemble["Label"] = (
    submissions[0]["Label"] * weight0
    + submissions[1]["Label"] * weight1
    + submissions[2]["Label"] * weight2
    + submissions[3]["Label"] * weight3
) / weight

sample_path = f"{PATH}/sample_submission.csv"
if os.path.exists(sample_path):
    sample = pd.read_csv(sample_path)
    sample_ids = sample["Id"].astype(str).values

    ensemble["Id"] = ensemble["Id"].astype(str)
    ensemble = ensemble.groupby("Id", as_index=False)["Label"].mean()

    ensemble = pd.DataFrame({"Id": sample_ids}).merge(ensemble, on="Id", how="left")

    fill_value = float(np.nanmean(ensemble["Label"].values))
    if not np.isfinite(fill_value):
        fill_value = 0.5
    ensemble["Label"] = ensemble["Label"].astype(np.float32).fillna(fill_value)

assert list(ensemble.columns) == ["Id", "Label"], f"Bad columns: {ensemble.columns}"
assert ensemble["Id"].notna().all(), "Some Id are NaN"
assert ensemble["Label"].notna().all(), "Some Label are NaN"
assert len(ensemble) == 5000, f"Unexpected submission rows: {len(ensemble)}"

out_path = "/kaggle/working/submission.csv"
ensemble.to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(ensemble), "cols:", list(ensemble.columns))
print(ensemble.head())
