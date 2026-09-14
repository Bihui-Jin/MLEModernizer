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

0.58472

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.58924) has done: 'Main bottlenecks are (1) iterating over the full ~240k training images for an epoch and then another ~60k for validation, which is infeasible in 600s, and (2) slow image decode/augment on CPU feeding the GPU. To keep the *same model, loss, transforms, and training loop semantics* but finish in time, the refactor makes the training/validation loaders stream from a fixed-size **deterministic subset** (still using the same fold split) sized to fit the 600s budget, and optimizes the input pipeline (OpenCV threading, faster tensor conversion, channels-last, persistent workers, and non-blocking H2D). Test-time inference is kept identical (same 4 TTA flips and weights), but made faster via channels-last and `inference_mode()`. All changes are deterministic (seeded) and do not alter architecture/loss/augmentations; they only remove redundant work and bound total work to meet the timeout.'
- What this solution (achieved 0.58328) has done: 'The timeout is dominated by (1) building a full 300k-image file list/DataFrame and (2) training on ~240k images plus validation, which is not feasible in 600 seconds. To preserve the exact model/training logic while cutting runtime, the refactor keeps the same architecture, loss, optimizer, scheduler, and loop semantics, but reuses the provided pretrained EfficientNet and skips the expensive training/val phase (and all associated dataset indexing) when generating the submission. Additionally, test-time inference is made faster without changing outputs by batching the 4 TTA flips into a single forward pass (same pixels, same weights), and by enabling fast DataLoader settings and avoiding repeated CPU↔GPU sync points.'
- What this solution (achieved 0.58472) has done: 'Your current score is far below the target, and the main reason is that the model is never loaded with any trained weights (so inference runs with essentially random head weights on top of ImageNet features). To move the score toward the target while keeping the same architecture and inference semantics, I (1) add a lightweight 1-epoch fine-tune on a small deterministic subset built from the already-available `Cover/` and one stego folder (to avoid huge indexing/timeouts), (2) keep your same transforms/loss/optimizer/scheduler/loop code intact, and (3) keep the same 4-flip TTA and submission formatting. This is the smallest change that legitimately improves signal without rewriting the approach, and it should still finish within the 600s budget.'

# 9. Code solution

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

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


seed_everything(SEED)

try:
    cv2.setUseOptimized(True)
    cv2.setNumThreads(max(1, (os.cpu_count() or 2) // 2))
except Exception:
    pass



## === cell 6
_ONEHOT_4 = torch.eye(4, dtype=torch.float32)


def onehot(size, target):
    if size == 4:
        return _ONEHOT_4[target]
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.0
    return vec


class DatasetRetriever(Dataset):
    def __init__(
        self,
        kinds,
        image_names,
        labels,
        transforms=None,
        cache_base=False,
        img_size=224,
    ):
        super().__init__()
        self.kinds = kinds
        self.image_names = image_names
        self.labels = labels
        self.transforms = transforms
        self.img_size = int(img_size)
        self._kind_to_folder = {k: os.path.join(PATH, k) for k in np.unique(kinds)}
        self.cache_base = bool(cache_base)

        self._base_cache = None
        if self.cache_base:
            self._base_cache = self._build_base_cache()

    def _build_base_cache(self):
        cache = {}
        for k, n in zip(self.kinds, self.image_names):
            key = (k, n)
            if key in cache:
                continue
            folder = self._kind_to_folder[k]
            img = cv2.imread(os.path.join(folder, n), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Could not read image: {folder}/{n}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(
                img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA
            )
            img = img.astype(np.float32) * (1.0 / 255.0)
            cache[key] = img.astype(np.float16)
        return cache

    def __getitem__(self, index: int):
        kind, image_name, label = (
            self.kinds[index],
            self.image_names[index],
            int(self.labels[index]),
        )
        if self._base_cache is not None:
            image = self._base_cache[(kind, image_name)].astype(np.float32)
        else:
            folder = self._kind_to_folder[kind]
            image = cv2.imread(os.path.join(folder, image_name), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Could not read image: {folder}/{image_name}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            image *= 1.0 / 255.0

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

dataset_df = None
print(
    "Using lightweight subset training (no full train/val indexing) to improve score within 600s."
)



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
        self.y_true_parts = []
        self.y_pred_parts = []
        self.score = 0.0

    def update(self, y_true, y_pred):
        y_true_np = (
            y_true.detach().cpu().numpy().argmax(axis=1).clip(min=0, max=1).astype(int)
        )
        y_pred_np = (
            1 - nn.functional.softmax(y_pred, dim=1).detach().cpu().numpy()[:, 0]
        )
        self.y_true_parts.append(y_true_np)
        self.y_pred_parts.append(y_pred_np)

    def finalize(self):
        if not self.y_true_parts:
            self.score = 0.0
            return self.score
        y_true = np.concatenate(self.y_true_parts, axis=0)
        y_pred = np.concatenate(self.y_pred_parts, axis=0)
        self.score = alaska_weighted_auc(y_true, y_pred)
        return self.score

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

        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.log(f"Fitter prepared. Device is {self.device}, AMP: {self.use_amp}")

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

        with torch.inference_mode():
            for step, (images, targets) in enumerate(val_loader):
                if self.config.verbose and step % self.config.verbose_step == 0:
                    print(
                        f"Val Step {step}/{len(val_loader)}, summary_loss: {summary_loss.avg:.5f}, final_score: {final_scores.avg:.5f}, time: {(time.time() - t):.5f}",
                        end="\r",
                    )

                targets = targets.to(self.device, non_blocking=True)
                if self.device.type == "cuda":
                    images = images.to(self.device, non_blocking=True).contiguous(
                        memory_format=torch.channels_last
                    )
                else:
                    images = images.to(self.device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)

                summary_loss.update(loss.detach().float().item(), images.shape[0])
                final_scores.update(targets, outputs)

        final_scores.finalize()

        print()
        return summary_loss, final_scores

    def train_model(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter()
        final_scores = RocAucMeter()
        t = time.time()

        for step, (images, targets) in enumerate(train_loader):
            if self.config.verbose and step % self.config.verbose_step == 0:
                print(
                    f"Train Step {step}/{len(train_loader)}, summary_loss: {summary_loss.avg:.5f}, final_score: {final_scores.avg:.5f}, time: {(time.time() - t):.5f}",
                    end="\r",
                )

            targets = targets.to(self.device, non_blocking=True)
            if self.device.type == "cuda":
                images = images.to(self.device, non_blocking=True).contiguous(
                    memory_format=torch.channels_last
                )
            else:
                images = images.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            summary_loss.update(loss.detach().float().item(), images.shape[0])
            final_scores.update(targets, outputs)

            if self.config.step_scheduler:
                self.scheduler.step()

        final_scores.finalize()

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

if DEVICE.type == "cuda":
    model = model.to(memory_format=torch.channels_last)




## === cell 14
class Config:
    batch_size = 16 if DEVICE.type == "cuda" else 4
    n_epochs = 1
    num_workers = min(8, os.cpu_count() or 2) if DEVICE.type == "cuda" else 0
    lr = 0.001
    verbose = True
    verbose_step = 50
    metric_step = 500
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
def _seed_worker(worker_id):
    worker_seed = (SEED + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _list_jpgs(folder):
    files = [f for f in os.listdir(folder) if f.lower().endswith(".jpg")]
    files.sort()
    return np.array(files)


def _build_small_train_val_subset(
    stego_kind="JMiPOD", n_train_each=1200, n_val_each=300
):
    cover_folder = os.path.join(PATH, "Cover")
    stego_folder = os.path.join(PATH, stego_kind)

    cover_files = _list_jpgs(cover_folder)
    stego_files = _list_jpgs(stego_folder)

    rng = np.random.RandomState(SEED)
    cover_idx = rng.permutation(len(cover_files))
    stego_idx = rng.permutation(len(stego_files))

    cover_sel = cover_files[cover_idx[: (n_train_each + n_val_each)]]
    stego_sel = stego_files[stego_idx[: (n_train_each + n_val_each)]]

    train_cover = cover_sel[:n_train_each]
    val_cover = cover_sel[n_train_each:]
    train_stego = stego_sel[:n_train_each]
    val_stego = stego_sel[n_train_each:]

    train_kinds = np.array(
        ["Cover"] * len(train_cover) + [stego_kind] * len(train_stego)
    )
    train_names = np.concatenate([train_cover, train_stego])
    stego_label = CLASSES.index(stego_kind)
    train_labels = np.array(
        [0] * len(train_cover) + [stego_label] * len(train_stego), dtype=np.int64
    )

    val_kinds = np.array(["Cover"] * len(val_cover) + [stego_kind] * len(val_stego))
    val_names = np.concatenate([val_cover, val_stego])
    val_labels = np.array(
        [0] * len(val_cover) + [stego_label] * len(val_stego), dtype=np.int64
    )

    tr_perm = rng.permutation(len(train_names))
    va_perm = rng.permutation(len(val_names))
    return (
        train_kinds[tr_perm],
        train_names[tr_perm],
        train_labels[tr_perm],
        val_kinds[va_perm],
        val_names[va_perm],
        val_labels[va_perm],
    )


do_train = True
print("Training enabled:", do_train)



## === cell 16
if do_train:
    tr_kinds, tr_names, tr_labels, va_kinds, va_names, va_labels = (
        _build_small_train_val_subset(
            stego_kind="JMiPOD",
            n_train_each=1200,
            n_val_each=300,
        )
    )

    train_ds = DatasetRetriever(
        kinds=tr_kinds,
        image_names=tr_names,
        labels=tr_labels,
        transforms=get_train_transforms(),
        cache_base=False,
        img_size=IMG_SIZE,
    )
    valid_ds = DatasetRetriever(
        kinds=va_kinds,
        image_names=va_names,
        labels=va_labels,
        transforms=get_valid_transforms(),
        cache_base=False,
        img_size=IMG_SIZE,
    )

    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=(DEVICE.type == "cuda"),
        drop_last=True,
        worker_init_fn=_seed_worker if Config.num_workers > 0 else None,
        persistent_workers=(Config.num_workers > 0),
        prefetch_factor=4 if Config.num_workers > 0 else None,
        generator=g,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=Config.batch_size * 2,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=(DEVICE.type == "cuda"),
        drop_last=False,
        worker_init_fn=_seed_worker if Config.num_workers > 0 else None,
        persistent_workers=(Config.num_workers > 0),
        prefetch_factor=4 if Config.num_workers > 0 else None,
    )

    fitter = Fitter(model=model, device=DEVICE, config=Config)
    fitter.fit(train_loader, valid_loader)

    best_ckpts = sorted(glob("./best-checkpoint-*epoch.bin"))
    if best_ckpts:
        fitter.load(best_ckpts[-1])
    else:
        if os.path.exists("./last-checkpoint.bin"):
            fitter.load("./last-checkpoint.bin")




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
    def __init__(self, image_names, transforms=None, cache_base=False, img_size=224):
        super().__init__()
        self.image_names = image_names
        self.transforms = transforms
        self._test_folder = os.path.join(PATH, "Test")
        self.cache_base = bool(cache_base)
        self.img_size = int(img_size)
        self._base_cache = None
        if self.cache_base:
            self._base_cache = self._build_base_cache()

    def _build_base_cache(self):
        cache = {}
        for n in self.image_names:
            if n in cache:
                continue
            img = cv2.imread(os.path.join(self._test_folder, n), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Could not read test image: {PATH}/Test/{n}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(
                img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA
            )
            img = img.astype(np.float32) * (1.0 / 255.0)
            cache[n] = img.astype(np.float16)
        return cache

    def __getitem__(self, index: int):
        image_name = self.image_names[index]
        if self._base_cache is not None:
            image = self._base_cache[image_name].astype(np.float32)
        else:
            image = cv2.imread(
                os.path.join(self._test_folder, image_name), cv2.IMREAD_COLOR
            )
            if image is None:
                raise FileNotFoundError(
                    f"Could not read test image: {PATH}/Test/{image_name}"
                )
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            image *= 1.0 / 255.0
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

base_test_tfms = A.Compose(
    [A.Resize(height=IMG_SIZE, width=IMG_SIZE, p=1.0), ToTensorV2(p=1.0)], p=1.0
)

ds = DatasetSubmissionRetriever(
    image_names=test_image_names,
    transforms=base_test_tfms,
    cache_base=False,
    img_size=IMG_SIZE,
)

test_bs = 128 if DEVICE.type == "cuda" else 8
test_num_workers = min(8, os.cpu_count() or 2) if DEVICE.type == "cuda" else 0
dl_kwargs = dict(
    batch_size=test_bs,
    shuffle=False,
    num_workers=test_num_workers,
    drop_last=False,
    pin_memory=(DEVICE.type == "cuda"),
    worker_init_fn=_seed_worker if test_num_workers > 0 else None,
)
if test_num_workers > 0:
    dl_kwargs.update(dict(persistent_workers=True, prefetch_factor=4))
dl = DataLoader(ds, **dl_kwargs)

weights = [5, 1, 1, 1]
weight_sum = sum(weights)


def _stego_prob_from_logits(logits: torch.Tensor) -> np.ndarray:
    return (1.0 - torch.softmax(logits, dim=1)[:, 0]).detach().cpu().numpy()


result_ids = []
result_labels = []

use_amp = DEVICE.type == "cuda"

with torch.inference_mode():
    for step, (image_names, images) in enumerate(dl):
        if step % 50 == 0:
            print(f"Test - step {step}/{len(dl)}", end="\r")

        if DEVICE.type == "cuda":
            images = images.to(DEVICE, non_blocking=True).contiguous(
                memory_format=torch.channels_last
            )
        else:
            images = images.to(DEVICE, non_blocking=True)

        imgs0 = images
        imgs1 = torch.flip(images, dims=[3])
        imgs2 = torch.flip(images, dims=[2])
        imgs3 = torch.flip(images, dims=[2, 3])
        big = torch.cat([imgs0, imgs1, imgs2, imgs3], dim=0)

        with torch.amp.autocast("cuda", enabled=use_amp):
            big_logits = model(big)

        bsz = images.shape[0]
        logits0, logits1, logits2, logits3 = big_logits.split(bsz, dim=0)

        p0 = _stego_prob_from_logits(logits0)
        p1 = _stego_prob_from_logits(logits1)
        p2 = _stego_prob_from_logits(logits2)
        p3 = _stego_prob_from_logits(logits3)

        blend_p = (
            p0 * weights[0] + p1 * weights[1] + p2 * weights[2] + p3 * weights[3]
        ) / weight_sum

        result_ids.extend(list(image_names))
        result_labels.extend(list(blend_p))

print()
blend = pd.DataFrame({"Id": result_ids, "Label": result_labels})

submission = sample_sub[["Id"]].merge(blend[["Id", "Label"]], on="Id", how="left")
assert submission.shape[0] == sample_sub.shape[0]
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
