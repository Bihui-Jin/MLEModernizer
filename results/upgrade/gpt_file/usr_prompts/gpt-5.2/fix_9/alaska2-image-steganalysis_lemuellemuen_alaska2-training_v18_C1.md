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

0.8870092884880383

# 6. Current score

0.5823

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.58048) has done: 'I fix the runtime failures by making the code device-agnostic (CPU fallback when no GPU driver is available) and by removing dependencies on missing external checkpoint/log folders. I keep your core model, training loop, and TTA logic intact, but ensure the model is always defined and moved to the selected device, and that inference uses the same device without `.cuda()` calls. I also fix the test file glob/path mismatch and ensure we always write a valid `submission.csv` with the exact `Id,Label` columns aligned to `sample_submission.csv`. Finally, I add safe fallbacks: if no checkpoint is available, the script still run end-to-end (optionally training for a short run is left as-is but not automatically executed to avoid time issues) and still produce a submission.'
- What this solution (achieved 0.58048) has done: 'Main bottlenecks are (1) training from scratch on ~240k images if no checkpoint is present, and (2) slow JPEG decode/resize repeated 4× during TTA inference. To finish under 600s without changing the model/loops/loss/feature logic, the refactor (a) guarantees “no-training” fast-path for submission by requiring a checkpoint (same inference semantics as your current checkpoint path) and failing fast if none exists, and (b) makes inference I/O-bound optimizations that are provably equivalent: caching the decoded+resized base tensor once per image and applying flips on the already-resized tensor, plus using more DataLoader workers and persistent workers for throughput. It also removes unnecessary per-sample one-hot allocation overhead by precomputing 4 one-hot tensors (exact same values), and enables `torch.inference_mode()` (equivalent to `no_grad()` but faster). These changes keep paths, architecture, and prediction math identical (up to negligible FP differences).'
- What this solution (achieved 0.68814) has done: 'I remove the hard failure when `/kaggle/input/alaska-checkpoint` is missing and instead run a short, time-bounded training fallback using the same model, loss, and training loop already present, then proceed to the exact same TTA inference/submission creation. This directly fixes the current runtime error and should increase score versus random/pretrained-only inference by producing a trained checkpoint within the 600s budget. I also ensure the log path is written to `/kaggle/working/log.txt` consistently (so it doesn’t try to write into read-only input), and keep all paths and submission formatting unchanged. No changes are made to the model architecture, loss, or prediction post-processing—only the control flow to guarantee an end-to-end run and a better-trained model.'
- What this solution (achieved 0.5823) has done: 'The timeout is dominated by CPU-side image decoding/resizing and Python loops in TTA inference (building per-batch lists and repeatedly flipping tensors), plus DataLoader/worker overhead that isn’t used for test-time here. I keep the exact same model, checkpoint-loading, TTA modes (0/1/2/3), and final weighted averaging, but make inference faster by (1) decoding+resizing once into a single contiguous tensor cache (instead of a Python dict of tensors), (2) using pinned-memory + non_blocking H2D transfers, and (3) vectorizing flips per batch (flip the whole batch tensor once, not per-image). These changes are mathematically equivalent (same pixels, same flips, same softmax/prob computation) and only remove redundant work and Python overhead.'

# 9. Code solution

## === cell 0
import os, warnings

warnings.filterwarnings("ignore", category=ResourceWarning)



## === cell 1
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import cv2
import random
import time
import re
from datetime import datetime

import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchsampler import ImbalancedDatasetSampler
except Exception:
    ImbalancedDatasetSampler = None

import timm

import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import GroupKFold
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2



## === cell 2
PATH = "/kaggle/input/alaska2-image-steganalysis"



## === cell 3
SEED = 42


def seed_everything(seed: int):
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

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)



## === cell 4
if not os.path.exists(PATH):
    alt = "/kaggle/input/alaska2-image-steganalysis/alaska2-image-steganalysis"
    if os.path.exists(alt):
        PATH = alt
print("Dataset PATH:", PATH)



## === cell 5
_ONEHOT_4 = torch.eye(4, dtype=torch.float32)


def onehot(size, target):
    if size == 4:
        return _ONEHOT_4[int(target)]
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
        kind = self.kinds[index]
        image_name = self.image_names[index]
        label = self.labels[index]

        image = cv2.imread(f"{PATH}/{kind}/{image_name}", cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {PATH}/{kind}/{image_name}")
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




## === cell 6
print("Chuyển tập dữ liệu thành các vector one-hot cho từng lớp:")
for i in range(4):
    print(f"Class {i}: {onehot(4, i)}")




## === cell 7
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * int(n)
        self.count += int(n)
        self.avg = self.sum / max(1, self.count)




## === cell 8
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




## === cell 9
class RocAucMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.y_true = np.array([0, 1], dtype=np.int64)
        self.y_pred = np.array([0.5, 0.5], dtype=np.float64)
        self.score = 0.0

    def update(self, y_true, y_pred):
        y_true_cls = y_true.detach().cpu().numpy().argmax(axis=1).astype(np.int64)
        y_true_bin = (y_true_cls != 0).astype(np.int64)

        y_pred_stego = (
            1 - nn.functional.softmax(y_pred, dim=1).detach().cpu().numpy()[:, 0]
        )

        self.y_true = np.hstack((self.y_true, y_true_bin))
        self.y_pred = np.hstack((self.y_pred, y_pred_stego))
        self.score = alaska_weighted_auc(self.y_true, self.y_pred)

    @property
    def avg(self):
        return float(self.score)




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
            if target.ndim == 2:
                target = target.argmax(dim=1)
            return torch.nn.functional.cross_entropy(x, target)




## === cell 11
class Fitter:
    def __init__(self, model, device, config):
        self.config = config
        self.epoch = 0

        self.base_dir = "/kaggle/working"
        os.makedirs(self.base_dir, exist_ok=True)

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
        for _ in range(self.epoch, self.config.n_epochs):
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
        for step, (images, targets) in enumerate(val_loader):
            if self.config.verbose and (step % self.config.verbose_step == 0):
                print(
                    f"Val Step {step}/{len(val_loader)}, "
                    + f"summary_loss: {summary_loss.avg:.5f}, final_score: {final_scores.avg:.5f}, "
                    + f"time: {(time.time() - t):.5f}",
                    end="\r",
                )
            with torch.no_grad():
                targets = targets.to(self.device).float()
                batch_size = images.shape[0]
                images = images.to(self.device).float()
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                final_scores.update(targets, outputs)
                summary_loss.update(loss.detach().item(), batch_size)

        if self.config.verbose:
            print()
        return summary_loss, final_scores

    def train_model(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter()
        final_scores = RocAucMeter()
        t = time.time()
        for step, (images, targets) in enumerate(train_loader):
            if self.config.verbose and (step % self.config.verbose_step == 0):
                print(
                    f"Train Step {step}/{len(train_loader)}, "
                    + f"summary_loss: {summary_loss.avg:.5f}, final_score: {final_scores.avg:.5f}, "
                    + f"time: {(time.time() - t):.5f}",
                    end="\r",
                )

            targets = targets.to(self.device).float()
            images = images.to(self.device).float()
            batch_size = images.shape[0]

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            loss.backward()

            final_scores.update(targets, outputs)
            summary_loss.update(loss.detach().item(), batch_size)

            self.optimizer.step()

            if self.config.step_scheduler:
                self.scheduler.step()

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
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(
            checkpoint.get("model_state_dict", checkpoint), strict=False
        )
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_summary_loss = checkpoint.get(
            "best_summary_loss", self.best_summary_loss
        )
        self.epoch = checkpoint.get("epoch", -1) + 1
        self.log(f"Đã load checkpoint từ {path}, resume từ epoch {self.epoch}")

    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, "a+", encoding="utf-8") as logger:
            logger.write(f"{message}\n")




## === cell 12
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




## === cell 13
def plot_log_results(metrics, save_path="log_plots.png"):
    epochs = metrics["epochs"]
    if len(epochs) == 0:
        print("No epochs found in log; skipping plot.")
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
    axs[0, 1].set_title("AUC score per Epoch")
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
    print(f"Saved plot to: {save_path}")




## === cell 14
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




## === cell 15
model = EffNet(4).to(DEVICE)
model.eval()




## === cell 16
class Config:
    batch_size = 16
    n_epochs = 11
    num_workers = 4
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




## === cell 17
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




## === cell 18
CLASSES = ["Cover", "JMiPOD", "JUNIWARD", "UERD"]
N_SPLITS = 5


def build_full_dataset_with_folds(
    path: str, seed: int, n_splits: int = 5
) -> pd.DataFrame:
    dataset_rows = []
    for label, kind in enumerate(CLASSES):
        image_paths = glob(os.path.join(path, kind, "*.jpg"))
        for p in image_paths:
            dataset_rows.append(
                {"kind": kind, "image_name": os.path.basename(p), "label": label}
            )
    random.Random(seed).shuffle(dataset_rows)
    df = pd.DataFrame(dataset_rows)
    df.loc[:, "fold"] = 0
    gkf = GroupKFold(n_splits=n_splits)
    for fold_number, (_, val_index) in enumerate(
        gkf.split(X=df.index, y=df["label"], groups=df["image_name"])
    ):
        df.loc[df.iloc[val_index].index, "fold"] = fold_number
    return df




## === cell 19
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
            with open(prev_log, "r", encoding="utf-8") as f:
                old_content = f.read()
            with open(self.output_log_path, "a+", encoding="utf-8") as f:
                f.write("\n\n# ==== Log từ phiên trước ====\n")
                f.write(old_content)
                f.write("\n\n# ==== Bắt đầu phiên mới ====\n")
            print(f"Ghi lại log cũ từ {prev_log} vào {self.output_log_path}")
        else:
            print(f"Không tìm thấy log.txt trong {self.ckpt_folder}")

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
        if best_ckpt is not None:
            print(f"Resume từ checkpoint: {best_ckpt}")
            fitter.load(best_ckpt)
        else:
            print("Không có checkpoint nào, bắt đầu từ đầu (epoch 0)")

        print(f"Bắt đầu training từ epoch {fitter.epoch}")
        fitter.fit(self.train_loader, self.val_loader)




## === cell 20
def try_load_checkpoint_into_model(
    model, device, ckpt_folder="/kaggle/input/alaska-checkpoint"
):
    if not os.path.exists(ckpt_folder):
        return False, None
    pattern = re.compile(r"best-checkpoint-(\d+)epoch\.bin")
    best = None
    best_epoch = -1
    for fname in os.listdir(ckpt_folder):
        m = pattern.match(fname)
        if m:
            ep = int(m.group(1))
            if ep > best_epoch:
                best_epoch = ep
                best = os.path.join(ckpt_folder, fname)
    if best is None:
        return False, None
    ckpt = torch.load(best, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return True, best


loaded, ckpt_path = try_load_checkpoint_into_model(model, DEVICE)
print("Checkpoint loaded:", loaded, ckpt_path)



## === cell 21
if not loaded:
    raise RuntimeError(
        "No external checkpoint found at /kaggle/input/alaska-checkpoint. "
        "Fallback training is disabled to meet the 600s timeout requirement. "
        "Please provide a checkpoint in that folder."
    )



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/4178754733.py in <cell line: 0>()
      1 if not loaded:
----> 2     raise RuntimeError(
      3         "No external checkpoint found at /kaggle/input/alaska-checkpoint. "
      4         "Fallback training is disabled to meet the 600s timeout requirement. "
      5         "Please provide a checkpoint in that folder."

RuntimeError: No external checkpoint found at /kaggle/input/alaska-checkpoint. Fallback training is disabled to meet the 600s timeout requirement. Please provide a checkpoint in that folder.

## === cell 22
log_path = "/kaggle/input/alaska-checkpoint/log.txt"
if os.path.exists(log_path):
    metrics_parsed = parse_log_file(log_path)
    plot_log_results(metrics_parsed, save_path="/kaggle/working/log_plots.png")
else:
    print("No external log file found at:", log_path)




## === cell 23
def get_test_transforms(mode):
    if mode == 0:
        return A.Compose(
            [
                A.Resize(height=512, width=512, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        )
    elif mode == 1:
        return A.Compose(
            [
                A.HorizontalFlip(p=1),
                A.Resize(height=512, width=512, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        )
    elif mode == 2:
        return A.Compose(
            [
                A.VerticalFlip(p=1),
                A.Resize(height=512, width=512, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        )
    else:
        return A.Compose(
            [
                A.HorizontalFlip(p=1),
                A.VerticalFlip(p=1),
                A.Resize(height=512, width=512, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        )




## === cell 24
class DatasetSubmissionRetrieverCached(Dataset):
    def __init__(self, image_names, base_cache, mode: int):
        super().__init__()
        self.image_names = image_names
        self.base_cache = base_cache
        self.mode = int(mode)

    def __getitem__(self, index: int):
        image_name = self.image_names[index]
        img = self.base_cache[image_name]  # torch.FloatTensor (C,H,W)

        if self.mode == 0:
            out = img
        elif self.mode == 1:
            out = torch.flip(img, dims=[2])  # horizontal flip (W)
        elif self.mode == 2:
            out = torch.flip(img, dims=[1])  # vertical flip (H)
        else:
            out = torch.flip(img, dims=[1, 2])  # both flips
        return image_name, out

    def __len__(self) -> int:
        return self.image_names.shape[0]


def _build_test_base_cache(image_names, path):
    n = len(image_names)
    cache = torch.empty((n, 3, 512, 512), dtype=torch.float32)
    for i, name in enumerate(
        tqdm(image_names, desc="Building test base cache (decode+resize once)")
    ):
        img = cv2.imread(f"{path}/Test/{name}", cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read test image: {path}/Test/{name}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        cache[i].copy_(torch.from_numpy(img).permute(2, 0, 1).contiguous())
    if torch.cuda.is_available():
        cache = cache.pin_memory()
    return cache




## === cell 25
sample_path = os.path.join(PATH, "sample_submission.csv")
if not os.path.exists(sample_path):
    sample_path = "/kaggle/input/sample_submission.csv"
sample = pd.read_csv(sample_path)

test_dir = os.path.join(PATH, "Test")
if os.path.exists(test_dir) and len(sample) == 5000:
    test_image_names = sample["Id"].values.astype(str)
else:
    test_paths = glob(os.path.join(PATH, "Test", "*.jpg"))
    test_image_names = np.array([os.path.basename(p) for p in test_paths])
    test_image_names = np.sort(test_image_names)

print("Num test images:", len(test_image_names))
print("First 5 test ids:", test_image_names[:5])



## === cell 26
test_base_cache = _build_test_base_cache(test_image_names, PATH)



## === cell 27
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # safe for inference

model.eval()
model.to(DEVICE)
model = model.to(memory_format=torch.channels_last)

results = []
batch_size = 64  # larger batch reduces overhead; does not change semantics

n_test = len(test_image_names)
total_steps = (n_test + batch_size - 1) // batch_size

for mode in range(0, 4):
    ids_out = []
    preds_out = []

    with torch.inference_mode():
        for step in range(total_steps):
            start = step * batch_size
            end = min(start + batch_size, n_test)

            images = test_base_cache[start:end]
            if mode == 1:
                images = torch.flip(images, dims=[3])  # horizontal flip (W)
            elif mode == 2:
                images = torch.flip(images, dims=[2])  # vertical flip (H)
            elif mode == 3:
                images = torch.flip(images, dims=[2, 3])  # both flips

            images = images.contiguous(memory_format=torch.channels_last)
            images = images.to(DEVICE, dtype=torch.float32, non_blocking=True)

            if step % 50 == 0:
                print("TTA mode", mode, "step", step, "/", total_steps, end="\r")

            y_pred = model(images)
            y_pred = (
                1 - nn.functional.softmax(y_pred, dim=1).detach().cpu().numpy()[:, 0]
            )

            batch_names = test_image_names[start:end]
            ids_out.extend(
                batch_names.tolist()
                if isinstance(batch_names, np.ndarray)
                else list(batch_names)
            )
            preds_out.extend(y_pred.tolist())

    results.append({"Id": ids_out, "Label": preds_out})

print("\nTTA inference done. Num modes:", len(results))



## === cell 28
submissions = []
for mode in range(0, 4):
    submission = pd.DataFrame(results[mode])
    submissions.append(submission)

for mode in range(0, 4):
    submissions[mode] = (
        submissions[mode].sort_values("Id", kind="mergesort").reset_index(drop=True)
    )

print(submissions[0].head())



## === cell 29
for mode in range(0, 4):
    submissions[mode].to_csv(f"submission_{mode}.csv", index=False)
print("Wrote per-TTA submissions: submission_0.csv ... submission_3.csv")



## === cell 30
weight0 = 5
weight1 = 1
weight2 = 1
weight3 = 3
weight = weight0 + weight1 + weight2 + weight3



## === cell 31
avg_df = submissions[0].copy()
avg_df["Label"] = (
    submissions[0]["Label"] * weight0
    + submissions[1]["Label"] * weight1
    + submissions[2]["Label"] * weight2
    + submissions[3]["Label"] * weight3
) / weight

avg_df = avg_df.set_index("Id").reindex(sample["Id"]).reset_index()
avg_df.columns = ["Id", "Label"]

if avg_df["Label"].isna().any():
    avg_df["Label"] = avg_df["Label"].fillna(avg_df["Label"].mean())

avg_df.to_csv("submission.csv", index=False)
print("Saved final submission to submission.csv with shape:", avg_df.shape)
print(avg_df.head())
