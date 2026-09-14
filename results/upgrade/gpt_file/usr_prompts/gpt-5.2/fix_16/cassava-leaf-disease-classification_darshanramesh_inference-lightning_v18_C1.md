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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.9

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.8612873980054397

# 6. Current score

0.7216

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11771) has done: 'I fix the immediate runtime errors by updating the deprecated `pytorch_lightning.metrics.functional.accuracy` import and ensuring `torchvision` is imported before `LitModel` is instantiated. Then I remove the dependency on a missing private checkpoint (`/kaggle/input/exp1-models/epoch0.ckpt`) and instead load standard ImageNet weights for ResNet50 using the current torchvision API, which keeps the same architecture but makes inference possible. Finally, I make the submission generation robust by reading the competition’s `sample_submission.csv` from the correct path, iterating exactly those image_ids, and writing a `submission.csv` with the required two columns and correct row count.'
- What this solution (achieved 0.66293) has done: 'Your score is low because the current code is effectively an untrained classifier head on top of a pretrained ResNet backbone, and the SWA wrapper averages only a single set of weights (so it changes nothing). To move accuracy up toward the 0.861 target while preserving your core architecture, we (1) load the training labels from `train.csv`, (2) train only the existing `self.logits` head for a small fixed number of epochs (no early stopping), and (3) keep inference the same but run it in batches for correctness/stability and speed. This keeps the model definition and loss (CrossEntropyLoss) intact and produces the same `submission.csv` format, but replaces “random head” with a minimally-trained head, which should substantially close the gap toward the target. We also set deterministic seeds and freeze the backbone so training is fast and finishes under the timeout.'
- What this solution (achieved 0.11771) has done: 'I remove the expensive fine-tuning loop that currently runs 10 full epochs over ~18k images (this is the primary timeout cause) and instead use the intended pretrained ResNet50 as a frozen feature extractor plus the same linear head, but load the head weights from a precomputed checkpoint if present; this preserves the model architecture and inference semantics while avoiding training-time compute. I also speed up data loading deterministically by enabling persistent DataLoader workers, prefetching, and using faster PIL decoding settings; these changes don’t alter the algorithm, only throughput. Finally, I keep inference identical but make it more efficient by using larger worker counts (bounded) and avoiding per-item filesystem checks inside `__getitem__`.'
- What this solution (achieved 0.73019) has done: 'Your current score is low because the linear `logits` head is effectively random whenever no cached `logits_head.pth` exists, so predictions collapse toward chance accuracy. To move toward the 0.861 target while preserving the exact same architecture and loss, I add a small, fixed training loop that trains only the existing `lit_model.logits` head on top of the frozen pretrained ResNet50 embeddings, using your existing optimizer/scheduler and transforms. I also correctly put the backbone in `eval()` during head training so BatchNorm statistics don’t drift while the backbone is frozen, which improves stability and accuracy without changing the model. Finally, I keep the submission writing identical but ensure the trained head is always used (either loaded or trained) before inference.'
- What this solution (achieved 0.68199) has done: 'You’re currently under the target (0.73019 vs 0.8613), so the smallest safe move is to improve generalization without changing the model architecture or loss. I keep the frozen ResNet50 + linear head exactly as-is, but add two minimal training fixes: (1) compute class weights from `train.csv` and pass them into `CrossEntropyLoss` to counter the strong class imbalance in Cassava, and (2) use a standard validation-driven checkpoint (best head on val accuracy) rather than always using the last epoch, which typically improves accuracy with no extra training. I also make inference use the best-saved head weights (or cached weights if provided), and keep the submission formatting identical.'
- What this solution (achieved 0.68199) has done: 'I keep your frozen ResNet50 + linear head setup intact, but make two minimal fixes that typically move accuracy up materially on Cassava without changing the modeling approach: (1) during head training, extract embeddings under `torch.no_grad()` so gradients don’t track the (frozen) backbone, reducing noise/instability and making optimization cleaner; and (2) use a stable validation split stratified by `label` only (the `source` column is not guaranteed and can create tiny strata), which improves validation selection of the “best head” checkpoint. I also load the best saved head reliably (prefer best over last), while leaving inference and submission formatting unchanged.'
- What this solution (achieved 0.69021) has done: 'Your current score (0.68199) is well below the target (0.8613), so we should improve generalization while keeping your frozen ResNet50 + linear head and the same loss intact. The smallest high-impact change is to make the feature extractor produce more stable embeddings by using the same resize/crop scheme for validation and test as for training (center-crop at 224 after resizing to 256), which reduces train–test mismatch without changing the model. Next, keep the exact training loop but train the head a bit longer with a standard StepLR schedule aligned to the longer run; this is still “head-only training” and finishes within the time budget. Finally, ensure the “best head” checkpoint is always preferred at inference (even when a cached head exists), so you don’t accidentally submit a suboptimal last-epoch head.'
- What this solution (achieved 0.71001) has done: 'Your current score (0.69021) is well below the target (0.86129), so we should make a small, safe improvement that preserves your frozen ResNet50 + linear head setup and CrossEntropyLoss training. The highest-impact minimal change is to remove dropout noise from the frozen backbone during head training/inference by keeping the backbone in `eval()` (as you already do) and additionally ensuring BatchNorm layers don’t run in train mode anywhere; then, modestly strengthen the head training signal by switching the LR schedule to a more appropriate longer StepLR (same optimizer, same loop) and slightly extending epochs while staying within the time budget. Finally, to better match test-time behavior without changing architecture, we use the same normalization/crop but increase test-time stability via simple deterministic test-time augmentation (horizontal flip) and average logits (no change to loss/labels, just inference post-processing), which typically lifts accuracy toward your target. All changes are localized to training duration/scheduler and inference averaging, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.72272) has done: 'The timeout is dominated by repeated expensive ResNet50 forward passes over the full training set for 18 epochs, even though the backbone is frozen; we can keep the exact same training logic while caching the frozen backbone embeddings once and training the same logits head on those cached tensors. This preserves the model architecture, loss, optimizer/scheduler, and number of optimization steps per epoch, but removes redundant compute and PIL decode costs during training. For inference, we keep identical test-time augmentation (original + horizontal flip) and predictions, but enable safe throughput settings (channels-last + cudnn autotune for fixed shapes) that do not change semantics. We also reduce Python overhead by using a small embedding dataset/dataloader for the head training loop and by keeping all randomness/determinism settings intact.'
- What this solution (achieved 0.7216) has done: 'Your current score (0.72272) is well below the target (0.86129), so we should make the smallest high-impact fixes that preserve your frozen ResNet50 + linear head, loss, and overall training/inference semantics. The main gap comes from a train/val mismatch: you train the head on features extracted from *augmented* (random flip) train images, but validate on non-augmented features; this makes “best head on val” selection less reliable and usually hurts test accuracy. I keep your exact architecture and head-only training, but (1) cache train features using the *same deterministic transform* as val/test (no random flip) to align distributions, and (2) apply the same simple TTA (original + horizontal flip) when extracting train/val features so training/selection better matches inference. These are localized changes inside the feature-caching path and should move accuracy upward toward the target without changing the model or adding extra training passes over ResNet beyond what you already do.'
- What this solution (achieved 0.7216) has done: 'Your current score (0.7216) is well below the target (0.8613), so we should make a small, safe improvement that preserves your frozen ResNet50 + linear head and the same loss/loops. The biggest low-risk accuracy gain here is to ensure you always use the best head weights you already trained: right now, if `logits_head_best.pth` doesn’t exist, you keep the last epoch head even though you tracked `best_state`. I (1) always load `best_state` into the head at the end of training (even if saving fails), and (2) force inference to prefer `logits_head_best.pth` if present, otherwise fall back to the trained-in-memory best. This keeps architecture/training semantics intact, but avoids submitting a worse “last epoch” head, which should move accuracy upward toward your target.'
- What this solution (achieved 0.7216) has done: 'I keep your frozen pretrained ResNet50 + linear head setup exactly as-is, but fix two small issues that are likely holding accuracy back: (1) the backbone’s `fc` contains dropout/BatchNorm, and you currently extract cached features in `eval()` (good) but you never explicitly freeze those `fc` BN stats against accidental mode changes, so I hard-lock the backbone to `eval()` and also set `requires_grad=False` for *all* backbone parameters once, including `fc`, to prevent any drift. (2) Your head training uses class-weighted CE, but the weights are computed on the train split only; I compute weights from the full `train.csv` distribution (still no leakage, since labels are only from train) for a slightly more stable prior that usually improves test accuracy on Cassava’s imbalance. These are minimal, localized changes that preserve architecture, loss, training loop, and TTA, and should move your 0.7216 upward toward the 0.861 target while still finishing within the timeout.'
- What this solution (achieved 0.7216) has done: 'I make two minimal, score-relevant fixes that keep your frozen ResNet50 + linear head architecture and the same CrossEntropy training loop intact. First, I apply the same test-time augmentation (original + horizontal flip averaged) during *head training* by caching train/val features using a deterministic (no-random) transform, so the head is trained/selected on the same feature distribution it see at inference. Second, I correct a mismatch in your code where the linear head expects 512-d embeddings but your current inference path still routes through the 5-class “classify=True” forward (which feeds images into the backbone and then the head); we keep that, but ensure the cached-feature pipeline always uses the backbone output and never accidentally uses augmented randomness. These changes are localized to the feature-caching/training path and should move accuracy upward toward your 0.8613 target without altering core modeling semantics.'

# 9. Code solution

## === cell 0
import os
import json
import time
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel

import torchvision
from torchvision import transforms
import pytorch_lightning as pl

from torchmetrics.functional import accuracy  # noqa: F401

from sklearn import metrics, model_selection, preprocessing  # noqa: F401
from PIL import Image

from albumentations import (
    Compose,
    HorizontalFlip,
    VerticalFlip,
    ShiftScaleRotate,
    RandomCrop,
    MultiplicativeNoise,
)  # noqa: F401

torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_num_threads(min(4, os.cpu_count() or 1))
torch.set_num_interop_threads(1)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

Image.MAX_IMAGE_PIXELS = None
try:
    from PIL import ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
except Exception:
    pass



## === cell 1
DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_CSV_PATH = os.path.join(DATA_DIR, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")

assert os.path.exists(
    SAMPLE_SUB_PATH
), f"Missing sample submission at {SAMPLE_SUB_PATH}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing test images dir at {TEST_IMG_DIR}"
assert os.path.exists(TRAIN_CSV_PATH), f"Missing train.csv at {TRAIN_CSV_PATH}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing train images dir at {TRAIN_IMG_DIR}"




## === cell 2
class LitModel(pl.LightningModule):
    def __init__(self, classify, n_cls=5, pretrained=False, t_data=None, v_data=None):
        super().__init__()
        self.classify = classify
        self.n_cls = n_cls
        self.pre_trained = pretrained
        self.model = self.modified_model()

        self.criterion = nn.CrossEntropyLoss()

        self.learning_rate = 0.1
        self.t_data = t_data
        self.v_data = v_data
        self.batch_size = 256

        self.logits = nn.Linear(512, self.n_cls)

    def forward(self, x):
        embeddings = self.model(x)
        if self.classify:
            logits = self.logits(embeddings)
            return logits
        else:
            return embeddings

    def modified_model(self):
        if self.pre_trained:
            weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        else:
            weights = None

        model = torchvision.models.resnet50(weights=weights)
        model.fc = nn.Sequential(
            nn.Dropout(p=0.8),
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
        )
        return model




## === cell 3
lit_model = LitModel(True, 5, pretrained=True)
lit_model.to(DEVICE)

lit_model.model.eval()
for p in lit_model.model.parameters():
    p.requires_grad = False
for p in lit_model.logits.parameters():
    p.requires_grad = True




## === cell 4
class SWAResnet(pl.LightningModule):
    def __init__(self, trained_model):
        super().__init__()
        self.model = trained_model
        self.swa_model = AveragedModel(self.model)

    def forward(self, x):
        logit = self.swa_model(x)
        return logit




## === cell 5
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_tfms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        normalize,
    ]
)

val_tfms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]
)

test_tfms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]
)

cache_tfms = val_tfms




## === cell 6
class CassavaTrainDataset(torch.utils.data.Dataset):
    def __init__(self, df, img_dir, tfms):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.tfms = tfms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_id"])
        img = Image.open(img_path).convert("RGB")
        x = self.tfms(img)
        y = int(row["label"])
        return x, y




## === cell 7
train_df = pd.read_csv(TRAIN_CSV_PATH)

trn_df, val_df = model_selection.train_test_split(
    train_df,
    test_size=0.1,
    random_state=42,
    stratify=train_df["label"],
)

n_cls = 5

counts = (
    train_df["label"]
    .value_counts()
    .reindex(range(n_cls), fill_value=0)
    .values.astype(np.float32)
)
counts = np.maximum(counts, 1.0)
inv = 1.0 / counts
cls_w = inv / inv.mean()
class_weights = torch.tensor(cls_w, dtype=torch.float32, device=DEVICE)
lit_model.criterion = nn.CrossEntropyLoss(weight=class_weights)

train_ds = CassavaTrainDataset(trn_df, TRAIN_IMG_DIR, train_tfms)
val_ds = CassavaTrainDataset(val_df, TRAIN_IMG_DIR, val_tfms)

batch_size = 64 if DEVICE == "cuda" else 32

num_workers = min(4, (os.cpu_count() or 2))
common_loader_kwargs = dict(
    num_workers=num_workers,
    pin_memory=(DEVICE == "cuda"),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
)

train_loader = torch.utils.data.DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    **common_loader_kwargs,
)
val_loader = torch.utils.data.DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    **common_loader_kwargs,
)



## === cell 8
base_lr_for_bs256 = 0.1
lr = base_lr_for_bs256 * (batch_size / 256.0)

optimizer = torch.optim.SGD(
    lit_model.logits.parameters(),
    lr=lr,
    momentum=0.9,
    weight_decay=1e-4,
)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.3)


def eval_acc(model_, loader_):
    model_.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader_:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            logits = model_(xb)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
    return correct / max(1, total)


head_ckpt_paths = [
    "/kaggle/input/cassava-leaf-disease-classification/logits_head.pth",
    "/kaggle/input/logits_head.pth",
    "/kaggle/working/logits_head.pth",
]
loaded = False
for pth in head_ckpt_paths:
    if os.path.exists(pth):
        state = torch.load(pth, map_location="cpu")
        try:
            lit_model.logits.load_state_dict(state)
            loaded = True
            break
        except Exception:
            if isinstance(state, dict) and "state_dict" in state:
                sd = state["state_dict"]
                sd2 = {
                    k.replace("logits.", ""): v
                    for k, v in sd.items()
                    if k.startswith("logits.")
                }
                if sd2:
                    lit_model.logits.load_state_dict(sd2)
                    loaded = True
                    break

best_head_path = "/kaggle/working/logits_head_best.pth"

if os.path.exists(best_head_path):
    try:
        lit_model.logits.load_state_dict(torch.load(best_head_path, map_location="cpu"))
        loaded = True
        print("Loaded cached best logits head weights.")
    except Exception:
        pass


class _FeatDataset(torch.utils.data.Dataset):
    def __init__(self, feats_cpu, labels_cpu):
        self.feats = feats_cpu
        self.labels = labels_cpu

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.feats[idx], int(self.labels[idx])


def _extract_backbone_features(ds, loader_bs, do_tta=True):
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=loader_bs,
        shuffle=False,
        **common_loader_kwargs,
    )
    lit_model.model.eval()
    feats_list = []
    labels_list = []
    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True
    with torch.no_grad():
        for xb, yb in loader:
            if DEVICE == "cuda":
                xb = xb.to(DEVICE, non_blocking=True).contiguous(
                    memory_format=torch.channels_last
                )
            else:
                xb = xb.to(DEVICE, non_blocking=True)

            f1 = lit_model.model(xb)
            if do_tta:
                xb_flip = torch.flip(xb, dims=[3])
                f2 = lit_model.model(xb_flip)
                feats = 0.5 * (f1 + f2)
            else:
                feats = f1

            feats_list.append(feats.detach().cpu())
            labels_list.append(yb.detach().cpu())

    feats_cpu = torch.cat(feats_list, dim=0).contiguous()
    labels_cpu = torch.cat(labels_list, dim=0).contiguous()
    return feats_cpu, labels_cpu


if not loaded:
    train_ds_for_cache = CassavaTrainDataset(trn_df, TRAIN_IMG_DIR, cache_tfms)
    val_ds_for_cache = CassavaTrainDataset(val_df, TRAIN_IMG_DIR, cache_tfms)

    feat_start = time.time()
    train_feats_cpu, train_labels_cpu = _extract_backbone_features(
        train_ds_for_cache, loader_bs=batch_size, do_tta=True
    )
    val_feats_cpu, val_labels_cpu = _extract_backbone_features(
        val_ds_for_cache, loader_bs=batch_size, do_tta=True
    )

    feat_elapsed = time.time() - feat_start
    print(
        f"Cached features: train={train_feats_cpu.shape} val={val_feats_cpu.shape} in {feat_elapsed:.1f}s"
    )

    feat_train_ds = _FeatDataset(train_feats_cpu, train_labels_cpu)
    feat_val_ds = _FeatDataset(val_feats_cpu, val_labels_cpu)

    feat_train_loader = torch.utils.data.DataLoader(
        feat_train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE == "cuda"),
    )
    feat_val_loader = torch.utils.data.DataLoader(
        feat_val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE == "cuda"),
    )

    def eval_acc_from_feats(loader_):
        lit_model.logits.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for feats_cpu, yb in loader_:
                feats = feats_cpu.to(DEVICE, non_blocking=True)
                yb = torch.as_tensor(yb, device=DEVICE)
                logits = lit_model.logits(feats)
                pred = logits.argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.numel()
        return correct / max(1, total)

    lit_model.logits.train()
    epochs = 18
    start = time.time()

    best_va = -1.0
    best_state = None

    for ep in range(epochs):
        running_loss = 0.0
        n_seen = 0

        for feats_cpu, yb in feat_train_loader:
            feats = feats_cpu.to(DEVICE, non_blocking=True)
            yb = torch.as_tensor(yb, device=DEVICE)

            optimizer.zero_grad(set_to_none=True)

            logits = lit_model.logits(feats)
            loss = lit_model.criterion(logits, yb)
            loss.backward()
            optimizer.step()

            bs = yb.size(0)
            running_loss += loss.item() * bs
            n_seen += bs

        scheduler.step()

        va = eval_acc_from_feats(feat_val_loader)

        if va > best_va:
            best_va = va
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in lit_model.logits.state_dict().items()
            }

        lit_model.logits.train()

        elapsed = time.time() - start
        print(
            f"epoch={ep+1}/{epochs} train_loss={running_loss/max(1,n_seen):.4f} "
            f"val_acc={va:.4f} best_val_acc={best_va:.4f} elapsed={elapsed:.1f}s"
        )

    torch.save(lit_model.logits.state_dict(), "/kaggle/working/logits_head.pth")
    if best_state is not None:
        try:
            torch.save(best_state, best_head_path)
        except Exception:
            pass
        lit_model.logits.load_state_dict(best_state)
        print(f"Using best head from training with val_acc={best_va:.4f}")
else:
    print("Loaded cached logits head weights (best if available).")

lit_model.eval()
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True



## === cell 9
final_model = lit_model
final_model.to(DEVICE)
final_model.eval()



## === cell 10
sample_submission_df = pd.read_csv(SAMPLE_SUB_PATH)


class CassavaTestDataset(torch.utils.data.Dataset):
    def __init__(self, image_ids, img_dir, tfms):
        self.image_ids = list(image_ids)
        self.img_dir = img_dir
        self.tfms = tfms

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, img_id)
        img = Image.open(img_path).convert("RGB")
        x = self.tfms(img)
        return img_id, x


test_ds = CassavaTestDataset(
    sample_submission_df["image_id"].tolist(), TEST_IMG_DIR, test_tfms
)
test_loader = torch.utils.data.DataLoader(
    test_ds,
    batch_size=256 if DEVICE == "cuda" else 32,
    shuffle=False,
    **common_loader_kwargs,
)

predictions = []
image_id = []

with torch.no_grad():
    for ids, xb in test_loader:
        if DEVICE == "cuda":
            xb = xb.to(DEVICE, non_blocking=True).contiguous(
                memory_format=torch.channels_last
            )
        else:
            xb = xb.to(DEVICE, non_blocking=True)

        logits1 = final_model(xb)

        xb_flip = torch.flip(xb, dims=[3])
        logits2 = final_model(xb_flip)

        logits = 0.5 * (logits1 + logits2)
        pred = logits.argmax(dim=1).detach().cpu().numpy().astype(int).tolist()

        predictions.extend(pred)
        image_id.extend(list(ids))



## === cell 11
assert len(predictions) == len(sample_submission_df), (
    len(predictions),
    len(sample_submission_df),
)
assert len(image_id) == len(sample_submission_df), (
    len(image_id),
    len(sample_submission_df),
)

my_submission = pd.DataFrame({"image_id": image_id, "label": predictions})
my_submission["label"] = my_submission["label"].astype(int)

my_submission = (
    my_submission.set_index("image_id")
    .loc[sample_submission_df["image_id"]]
    .reset_index()
)

out_path = "submission.csv"
my_submission.to_csv(out_path, index=False)

print(f"Wrote {out_path} with shape={my_submission.shape}")
print(my_submission.head())
