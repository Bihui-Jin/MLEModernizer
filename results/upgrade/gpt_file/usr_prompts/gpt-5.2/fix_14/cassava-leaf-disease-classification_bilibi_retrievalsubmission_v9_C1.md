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

geopandas==0.14.4
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

0.8573587186461167

# 6. Current score

0.76196

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.10762) has done: 'Your notebook fails because it relies on a missing external Kaggle Dataset (`baseline-infer`) that provided the `data/`, `model/`, `configs/`, etc. python packages, so the imports and `build_transforms`/`net` construction never work. To keep the same end-to-end intent (load a ResNet50-based model and run test-time inference to create `submission.csv`), I replace those missing modules with a minimal, equivalent PyTorch ResNet50 classifier head and standard image transforms, and make inference follow the exact row order in `sample_submission.csv` so the submission length/order always matches. I also fix pathing to use the provided `/kaggle/input/...` structure, ensure all required imports exist in the cells that use them, and write a valid `submission.csv` with exactly 2676 rows.'
- What this solution (achieved 0.1065) has done: 'Your current score is low because the notebook is almost certainly running with random weights (the `baseline-weights` dataset path isn’t present in the provided file tree), so predictions are effectively random. To move the score toward your target with minimal change, I switch the model to use ImageNet-pretrained ResNet50 weights when the competition checkpoint isn’t found, which preserves the same architecture and inference flow but gives a strong, legitimate baseline. I also apply standard torchvision ImageNet preprocessing (Resize/CenterCrop/Normalize) instead of manual OpenCV resizing to better match the pretrained model’s expected input distribution. Everything else (submission order, format, no extra training) stays the same.'
- What this solution (achieved 0.73468) has done: 'Your current score (0.1065) is far below the target (0.8574), and the main reason is that you’re effectively using a randomly-initialized 5-class head when the competition checkpoint is missing, so predictions are near-random. To move the score up with minimal core-logic changes, I keep the same ResNet50 inference pipeline but (1) replace the random head with a cheap, legitimate linear probe trained on the provided `train.csv` images using the frozen ImageNet backbone, and (2) ensure the exact same ImageNet preprocessing is used for both train and test. This preserves the architecture (ResNet50 + linear head), loss (cross-entropy), and straightforward training loop, while producing a much stronger submission within the time limit. If the external weights file is present, the code still load it and skip training.'
- What this solution (achieved 0.6988) has done: 'Your current score (0.73468) is below the target (0.85736), so we should improve accuracy with the smallest legitimate changes while keeping the same ResNet50 + linear-head training/inference logic. The biggest low-risk gain is to train the same frozen-backbone linear head on the full train set (instead of a 6000-image subset) and to use the standard ImageNet augmentation for training only (RandomResizedCrop + RandomHorizontalFlip) while keeping test preprocessing unchanged. This preserves the architecture, loss (cross-entropy), and training loop, but usually yields a sizable accuracy bump for this competition. I also add a small weighted sampler to reduce class-imbalance harm without changing the model or objective.'
- What this solution (achieved 0.6577) has done: 'Your current score (0.6988) is well below the target (0.85736), so we should improve accuracy with the smallest changes that keep the same “frozen ResNet50 backbone + trained linear head + cross-entropy + simple loop” core logic. The biggest issue is that the model is training the head from random initialization while the backbone is frozen, which often converges poorly in only 4 epochs; initializing the head with a better starting point and using a more stable optimizer setup for a linear probe typically yields a sizable bump without changing architecture or loss. I (1) run a single no-grad pass over train to compute backbone features and fit the exact same linear head with a closed-form/iterative solver (still cross-entropy-trained head, just initialized), then (2) continue the same AdamW training loop as before; additionally (3) switch the sampler to class-balanced *without replacement per epoch* behavior to reduce variance and help convergence. All paths, transforms, submission ordering/format, and the ResNet50 model structure remain unchanged.'
- What this solution (achieved 0.63266) has done: 'Your current score (0.6577) is far below the target (0.85736), so we should improve accuracy with minimal, low-risk changes that keep the same “ImageNet ResNet50 backbone + train only linear head + cross-entropy + simple loop” core logic. The main issue is the class-balanced `WeightedRandomSampler(replacement=False)` which causes the “epoch” to be a biased, non-iid permutation and can under-train/over-train classes in a way that hurts generalization for this task. I switch to standard shuffled training (no sampler) to better match the true training distribution, and I add a lightweight validation split only to select the best epoch for inference (no early stopping; still trains all epochs), which typically gives a solid bump without changing architecture or loss. Everything else (paths, transforms, model definition, loss, and writing `submission.csv` in sample order) remains the same.'
- What this solution (achieved 0.65247) has done: 'Your current score (0.63266) is well below the target (0.85736), so we should increase accuracy with the smallest changes that keep the same “ResNet50 + train only the linear head + cross-entropy + simple loop” core logic. The biggest low-risk win here is removing the unnecessary 512-resize preprocessing (which distorts the ImageNet-pretrained backbone’s expected input distribution) and always using standard ImageNet 224 preprocessing for both train/val/test, regardless of whether a competition checkpoint exists. Additionally, your current train/val split is stratified but not grouped; for cassava there are near-duplicates, so grouping by the image_id prefix (before “.jpg”) helps reduce leakage and select a better epoch without changing training semantics. Finally, we keep the same training loop/epochs, but switch the head optimizer LR back to your configured LR (3e-3) and use a cosine schedule across the fixed epochs to improve convergence of the linear probe without changing the model/loss or adding early stopping.'
- What this solution (achieved 0.76271) has done: 'Your current score (0.65247) is well below the target (0.85736), so we should improve accuracy with the smallest changes that keep the same “ImageNet ResNet50 backbone + train only linear head + cross-entropy + simple training loop” logic. The biggest low-risk issue is that BatchNorm layers remain in training mode during head training (because `model.train()` is called), which updates BN running stats even though the backbone is frozen; this commonly hurts transfer accuracy. I keep the exact same model and optimizer/loss/epochs, but force the frozen backbone (all layers except `fc`) to stay in `eval()` mode while still training `fc`, preventing BN drift. I also ensure the feature extractor uses the same frozen-eval backbone behavior for consistency.'
- What this solution (achieved 0.74253) has done: 'Your current score (0.76271) is below the target (0.85736), so we should improve accuracy with the smallest safe changes that keep the same “ImageNet ResNet50 backbone + train only linear head + cross-entropy + simple loop” core logic. The biggest likely remaining gap is that the linear head is being trained on heavily augmented views (RandomResizedCrop with scale down to 0.7), which is often too strong for a frozen-backbone linear probe and can reduce final test accuracy; switching to standard 224 training preprocessing (Resize+CenterCrop) keeps semantics identical and typically improves stability/accuracy. I also enable a small amount of label smoothing in CrossEntropyLoss (same loss family, same labels) to improve generalization without changing architecture or training loop structure. Everything else (frozen backbone with BN kept in eval, epochs, optimizer/scheduler, submission order/format) remains unchanged and still writes a valid `submission.csv`.'
- What this solution (achieved 0.67003) has done: 'Your current score (0.74253) is below the target (0.85736), so we should improve accuracy with the smallest, lowest-risk changes that keep the same “ImageNet ResNet50 backbone + train only linear head + cross-entropy + simple loop” core logic. The biggest likely gain is that the current head training sees a distribution mismatch (no color/lighting augmentation) and the head is initialized using a no-label-smoothing solver but then trained with label smoothing; we align these by (1) adding very light ColorJitter to training only (test/val unchanged) and (2) using label smoothing consistently during the head initialization pass. Additionally, we use class-weighted CrossEntropyLoss (still cross-entropy, same labels) to reduce class-imbalance error without changing the architecture or training loop. Everything else (frozen backbone with BN kept in eval, epochs, optimizer/scheduler, and submission ordering/format) remains unchanged and still writes `submission.csv`.'
- What this solution (achieved 0.67451) has done: 'Your score gap is large (0.67003 vs target 0.85736), so we should improve accuracy with minimal changes while keeping the same “ImageNet ResNet50 backbone + train only linear head + cross-entropy + simple loop” core logic. The biggest likely issue is that we’re training/evaluating a linear head with dropout/BN-safe backbone, but we still use fairly weak augmentation and we never leverage the known strong baseline for this competition: simple test-time augmentation (TTA) at inference. I keep the same model/training exactly, and only adjust inference to average predictions across two deterministic views (original + horizontal flip), which is a minimal, legitimate change that usually improves accuracy for leaf-disease classification. I also make the head-initialization feature pass use the same (non-augmented) preprocessing as validation (so the initializer is less noisy), without changing the model, loss family, or training loop structure.'
- What this solution (achieved 0.75972) has done: 'Your current score (0.6745) is far below the target (0.8574), so we should improve accuracy with the smallest safe changes while preserving the same ResNet50 + linear-head training/inference core logic. The biggest low-risk issue is that class-weighted loss plus label smoothing can hurt top-1 accuracy for this dataset when only training the head; we keep the exact same model, epochs, optimizer, scheduler, and TTA, but remove the class weights while keeping label smoothing (still CrossEntropyLoss). Additionally, we train the FC head on a slightly less jittered distribution by disabling ColorJitter (keeping flip), which typically helps a frozen-backbone linear probe. Finally, we make inference use softmax-probability averaging for TTA (same argmax labels, just better-calibrated averaging) without changing model or metric semantics.'
- What this solution (achieved 0.76196) has done: 'Your current score (0.75972) is below the target (0.85736), so we should increase accuracy with the smallest safe changes while preserving the same “ImageNet ResNet50 backbone + train only linear head + cross-entropy + simple loop + 2-view TTA” core logic. The main low-risk improvement is to train the FC head on the full training distribution rather than a split that discards 10% of data, and still use the existing validation split only for model selection (we keep the same number of epochs and still select the best epoch by val accuracy). This typically improves the final head fit without changing the model, loss, or training procedure. Additionally, we make the training DataLoader deterministically shuffled via a seeded generator (stability) without changing semantics.'

# 9. Code solution

## === cell 0
import os
import sys
import copy
import datetime
import random

import numpy as np
import pandas as pd

import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.models import ResNet50_Weights
import torchvision.transforms as T


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_CSV_PATH = os.path.join(DATA_ROOT, "train.csv")

assert os.path.isdir(TEST_IMG_DIR), f"Missing test_images dir: {TEST_IMG_DIR}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing train_images dir: {TRAIN_IMG_DIR}"
assert os.path.isfile(
    SAMPLE_SUB_PATH
), f"Missing sample_submission.csv: {SAMPLE_SUB_PATH}"
assert os.path.isfile(TRAIN_CSV_PATH), f"Missing train.csv: {TRAIN_CSV_PATH}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



## === cell 1
cfg = {
    "img_size": 512,
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
    "batch_size": 64,
    "num_workers": 2,
    "num_classes": 5,
    "lr": 3e-3,
    "epochs": 4,
    "val_frac": 0.1,
    "label_smoothing": 0.05,
}

weights_path = "../input/baseline-weights/epoch_22.pth"

candidate_weights = [
    weights_path,
    "/kaggle/input/baseline-weights/epoch_22.pth",
    "/kaggle/input/baseline-weights/epoch_22.pth.tar",
]

weights_path = next((p for p in candidate_weights if os.path.isfile(p)), None)
weights_path



## === cell 2
train_transform = T.Compose(
    [
        T.ToPILImage(),
        T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(224),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=cfg["mean"], std=cfg["std"]),
    ]
)

test_transform = T.Compose(
    [
        T.ToPILImage(),
        T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=cfg["mean"], std=cfg["std"]),
    ]
)


test_transform_hflip = T.Compose(
    [
        T.ToPILImage(),
        T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(224),
        T.RandomHorizontalFlip(p=1.0),
        T.ToTensor(),
        T.Normalize(mean=cfg["mean"], std=cfg["std"]),
    ]
)


class TrainSet(Dataset):
    def __init__(self, img_dir: str, df: pd.DataFrame, transform):
        self.img_dir = img_dir
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.df.loc[idx, "image_id"]
        label = int(self.df.loc[idx, "label"])
        path = os.path.join(self.img_dir, image_id)
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        x = self.transform(img_rgb)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


class InferSet(Dataset):
    def __init__(self, img_dir: str, image_ids, transform):
        self.img_dir = img_dir
        self.image_ids = list(image_ids)
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        path = os.path.join(self.img_dir, image_id)
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        x = self.transform(img_rgb)
        return x, image_id


sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
assert list(sample_sub.columns) == [
    "image_id",
    "label",
], "Unexpected sample_submission.csv columns"
image_ids = sample_sub["image_id"].tolist()

train_df = pd.read_csv(TRAIN_CSV_PATH)
assert list(train_df.columns) == ["image_id", "label"], "Unexpected train.csv columns"
train_df = train_df.reset_index(drop=True)

rng = np.random.RandomState(42)
stems = train_df["image_id"].astype(str).str.replace(".jpg", "", regex=False)
groups = stems.str.slice(0, 6).values

val_mask = np.zeros(len(train_df), dtype=bool)
for c in range(cfg["num_classes"]):
    idxs = np.where(train_df["label"].values == c)[0]
    g = groups[idxs]
    uniq_g = np.unique(g)
    rng.shuffle(uniq_g)
    n_val_g = max(1, int(round(cfg["val_frac"] * len(uniq_g))))
    val_groups = set(uniq_g[:n_val_g])
    val_mask[idxs[np.isin(g, list(val_groups))]] = True

train_df_tr = train_df.loc[~val_mask].reset_index(drop=True)
train_df_va = train_df.loc[val_mask].reset_index(drop=True)

train_set = TrainSet(TRAIN_IMG_DIR, train_df_tr, train_transform)
val_set = TrainSet(TRAIN_IMG_DIR, train_df_va, test_transform)

train_set_full = TrainSet(TRAIN_IMG_DIR, train_df, train_transform)

dl_gen = torch.Generator()
dl_gen.manual_seed(42)

train_loader = DataLoader(
    train_set_full,
    batch_size=cfg["batch_size"],
    shuffle=True,
    generator=dl_gen,
    num_workers=cfg["num_workers"],
    pin_memory=torch.cuda.is_available(),
)

val_loader = DataLoader(
    val_set,
    batch_size=cfg["batch_size"],
    shuffle=False,
    num_workers=cfg["num_workers"],
    pin_memory=torch.cuda.is_available(),
)

infer_set = InferSet(TEST_IMG_DIR, image_ids, test_transform)
infer_loader = DataLoader(
    infer_set,
    batch_size=cfg["batch_size"],
    shuffle=False,
    num_workers=cfg["num_workers"],
    pin_memory=torch.cuda.is_available(),
)

infer_set_flip = InferSet(TEST_IMG_DIR, image_ids, test_transform_hflip)
infer_loader_flip = DataLoader(
    infer_set_flip,
    batch_size=cfg["batch_size"],
    shuffle=False,
    num_workers=cfg["num_workers"],
    pin_memory=torch.cuda.is_available(),
)

xb, yb = next(iter(train_loader))
xb.shape, yb.shape



## === cell 3
if weights_path is None:
    backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
else:
    backbone = models.resnet50(weights=None)

in_features = backbone.fc.in_features
backbone.fc = nn.Linear(in_features, cfg["num_classes"])

if weights_path is not None:
    ckpt = torch.load(weights_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    new_state = {}
    for k, v in state.items():
        kk = k
        if kk.startswith("module."):
            kk = kk[len("module.") :]
        new_state[kk] = v

    missing, unexpected = backbone.load_state_dict(new_state, strict=False)
    print(f"Loaded weights from: {weights_path}")
    print(f"Missing keys: {len(missing)}; Unexpected keys: {len(unexpected)}")
else:
    print(
        "No competition weights found; using ImageNet backbone and training only the 5-class head on train.csv."
    )

model = backbone.to(device)




## === cell 4
def evaluate_acc(m: nn.Module, loader: DataLoader) -> float:
    m.eval()
    correct = 0
    n = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = m(xb)
            preds = torch.argmax(logits, dim=1)
            correct += int((preds == yb).sum().detach().cpu())
            n += xb.size(0)
    return correct / max(1, n)


def set_backbone_eval_keep_fc_mode(m: nn.Module, fc_train: bool = True):
    m.eval()
    if fc_train:
        m.fc.train()
    else:
        m.fc.eval()


if weights_path is None:
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True

    init_set = TrainSet(TRAIN_IMG_DIR, train_df, test_transform)
    init_loader = DataLoader(
        init_set,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    set_backbone_eval_keep_fc_mode(model, fc_train=False)
    feats_list = []
    y_list = []
    feature_extractor = nn.Sequential(*list(model.children())[:-1]).to(device)
    feature_extractor.eval()

    with torch.no_grad():
        for xb, yb in init_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            feats = feature_extractor(xb).flatten(1)  # (B, 2048)
            feats_list.append(feats.detach().cpu())
            y_list.append(yb.detach().cpu())

    X = torch.cat(feats_list, dim=0)  # (N, 2048)
    y = torch.cat(y_list, dim=0)  # (N,)
    X_mean = X.mean(dim=0, keepdim=True)
    X_std = X.std(dim=0, keepdim=True).clamp_min(1e-6)
    Xn = (X - X_mean) / X_std

    W = torch.zeros((cfg["num_classes"], Xn.shape[1]), dtype=torch.float32)
    b = torch.zeros((cfg["num_classes"],), dtype=torch.float32)
    lr_init = 0.5
    steps = 25
    reg = 1e-4

    Y = torch.nn.functional.one_hot(y, num_classes=cfg["num_classes"]).float()
    if cfg["label_smoothing"] > 0:
        eps = float(cfg["label_smoothing"])
        Y = (1.0 - eps) * Y + eps / cfg["num_classes"]

    for _ in range(steps):
        logits = Xn @ W.t() + b  # (N, C)
        logits = logits - logits.max(dim=1, keepdim=True).values
        exp = torch.exp(logits)
        probs = exp / exp.sum(dim=1, keepdim=True)
        grad_logits = (probs - Y) / max(1, Xn.shape[0])
        gW = grad_logits.t() @ Xn + reg * W
        gb = grad_logits.sum(dim=0)
        W -= lr_init * gW
        b -= lr_init * gb

    with torch.no_grad():
        W_eff = W / X_std.squeeze(0)
        b_eff = b - (X_mean.squeeze(0) / X_std.squeeze(0)) @ W.t()
        model.fc.weight.copy_(W_eff.to(model.fc.weight.dtype))
        model.fc.bias.copy_(b_eff.to(model.fc.bias.dtype))

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])

    optimizer = torch.optim.AdamW(
        model.fc.parameters(), lr=cfg["lr"], weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"]
    )
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val_acc = -1.0
    best_state = None

    for epoch in range(cfg["epochs"]):
        running_loss = 0.0
        running_correct = 0
        running_n = 0

        set_backbone_eval_keep_fc_mode(model, fc_train=True)

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(xb)
                loss = criterion(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.detach().cpu()) * xb.size(0)
            preds = torch.argmax(logits.detach(), dim=1)
            running_correct += int((preds == yb).sum().detach().cpu())
            running_n += xb.size(0)

        scheduler.step()

        val_acc = evaluate_acc(model, val_loader)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"epoch {epoch+1}/{cfg['epochs']} - lr {scheduler.get_last_lr()[0]:.6f} - loss {running_loss/max(running_n,1):.4f} - train_acc {running_correct/max(running_n,1):.4f} - val_acc {val_acc:.4f} (best {best_val_acc:.4f})"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

model.eval()



## === cell 5
out_path = "./submission.csv"

preds_all = []
ids_all = []

with torch.no_grad():
    model.eval()

    for (xb1, fns1), (xb2, fns2) in zip(infer_loader, infer_loader_flip):
        assert list(fns1) == list(fns2), "TTA loaders are misaligned."

        xb1 = xb1.to(device, non_blocking=True)
        xb2 = xb2.to(device, non_blocking=True)

        logits1 = model(xb1)
        logits2 = model(xb2)

        p1 = torch.softmax(logits1, dim=1)
        p2 = torch.softmax(logits2, dim=1)
        p = (p1 + p2) / 2.0

        preds = torch.argmax(p, dim=1).detach().cpu().numpy().astype(int).tolist()
        preds_all.extend(preds)
        ids_all.extend(list(fns1))

assert len(ids_all) == len(
    sample_sub
), f"Pred length {len(ids_all)} != sample {len(sample_sub)}"
assert (
    ids_all == image_ids
), "Order mismatch vs sample_submission (should not happen with this DataLoader)."

sub = pd.DataFrame({"image_id": ids_all, "label": preds_all})
sub.to_csv(out_path, index=False)

print(f"Wrote {out_path} with shape {sub.shape}")
sub.head()



## === cell 6
check = pd.read_csv("./submission.csv")
print(check.shape)
print(check.dtypes)
print(check.head())
assert check.shape[0] == sample_sub.shape[0], "Invalid submission: wrong number of rows"
assert list(check.columns) == ["image_id", "label"], "Invalid submission: wrong columns"
assert check["image_id"].isna().sum() == 0, "Invalid submission: missing image_id"
assert check["label"].isna().sum() == 0, "Invalid submission: missing labels"
assert check["label"].between(0, 4).all(), "Invalid submission: labels must be in [0,4]"
