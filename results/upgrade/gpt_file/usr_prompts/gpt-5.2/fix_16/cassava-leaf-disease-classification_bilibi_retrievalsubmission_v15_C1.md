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

0.8689936536718041

# 6. Current score

0.31801

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.10426) has done: 'The main issue is that this notebook depends on external Kaggle Dataset code (`utils`, `model`, `configs`, and a long config module) that is not available in your current environment, so imports fail and the pipeline never produces a valid submission. To keep the same “load a pretrained model and run inference on test_images” core approach, I replace those missing modules with a self-contained PyTorch/torchvision ResNet50 inference pipeline that runs end-to-end on the provided dataset paths. I also fix the submission-length error by writing predictions in exactly the order of `sample_submission.csv` and ensuring every `image_id` gets one `label`. The script writes `submission.csv` with the required two columns and a `.csv` suffix.'
- What this solution (achieved 0.33333) has done: 'Your score is extremely low because the model is effectively making random 5-class predictions: you’re loading an ImageNet backbone but your 5-class head is randomly initialized (and the external `baseline-weights` path doesn’t exist in this environment). To move the score toward the target with minimal core-logic change (still “pretrained model + inference on test_images”), I swap to a torchvision model that already has 5 output classes trained on Cassava (so the head is not random) and keep the same preprocessing/inference/submission pipeline. I also adjust the normalization/resize to exactly match the pretrained weights’ expected transforms to avoid avoidable accuracy loss. The rest (DataLoader order, argmax, submission mapping) stays the same to preserve evaluation semantics.'
- What this solution (achieved 0.74738) has done: 'Your current 0.33333 score strongly suggests the model is not actually using Cassava-trained weights and is effectively guessing (often by collapsing to one class). To move toward the 0.86899 target with minimal core-logic change (still “pretrained model + deterministic inference on test_images”), I switch to a torchvision classification backbone that has widely-available ImageNet weights and then load *actual Cassava 5-class finetuned weights* by training briefly on the provided `train.csv`/`train_images` (same model, same cross-entropy objective) before running test inference. I keep preprocessing consistent (ImageNet normalization, fixed resize) and ensure the submission order exactly matches `sample_submission.csv`. This stays within Kaggle constraints, runs end-to-end, and should materially increase accuracy from the random/near-random regime toward your target.'
- What this solution (achieved 0.6988) has done: 'Your current score is below target, so we should improve generalization with the smallest change that preserves the same “pretrained ResNet50 + replace fc + train briefly + argmax inference” core logic. The biggest issue is you train on the full training set without a validation split and stop after a fixed 3 epochs, which can easily overfit and hurts test accuracy; adding a simple stratified split and selecting the best epoch by validation accuracy keeps the same training objective/loop but improves model selection toward the target. I also switch the loss to class-weighted CrossEntropy using training-label frequencies (still CrossEntropy) to address Cassava’s class imbalance, which typically gives a noticeable accuracy bump with minimal risk. Finally, I keep preprocessing and submission ordering identical to avoid format/alignment regressions.'
- What this solution (achieved 0.76084) has done: 'Your current gap to target is large (0.6988 → 0.86899), so we should improve generalization while keeping your exact core approach (ResNet50 pretrained backbone, replace fc, train briefly with CrossEntropy, argmax inference). The biggest low-risk issue is that you only train the final FC layer; unfreezing and fine-tuning the last ResNet stage (`layer4`) (plus `fc`) is a minimal extension that usually yields a sizeable accuracy jump on Cassava without changing architecture or loss. To keep convergence stable without changing the loop structure, I also reduce the learning rate for fine-tuning and add a tiny amount of label smoothing (still CrossEntropy) to improve calibration/generalization. Finally, I switch resize interpolation to bicubic (closer to typical ImageNet preprocessing) while keeping normalization and submission ordering identical.'
- What this solution (achieved 0.16218) has done: 'The timeout is dominated by input pipeline overhead (OpenCV read + PIL conversion + heavy torchvision CPU transforms) and dataloader under-utilization, which starves the GPU. I keep the exact model, loss, optimizer, scheduler, epochs, and evaluation logic, but speed up loading/augmentation by (1) switching to torchvision’s fast tensor-based transforms (no PIL roundtrip) while preserving the same transform semantics, (2) enabling persistent workers + prefetching, and (3) using a faster OpenCV decode path plus minor dataset indexing optimizations. I also enable TF32 matmul on Ampere+ GPUs (numerically negligible) and compile the model when available for a large forward/backward speedup without changing the architecture.'
- What this solution (achieved 0.3707) has done: 'The timeout is most likely dominated by (1) slow per-sample image decoding/augmentation in Python and (2) `torch.compile(mode="max-autotune")` spending substantial time compiling/tuning relative to this small fine-tuning workload. I keep the exact same model, losses, epochs, and data augmentation semantics, but make the input pipeline faster by switching to `torchvision.io.read_image` (libjpeg-turbo) instead of OpenCV+color conversion, enabling `torch.inference_mode()` for eval/infer, and using `torch.backends.cudnn.benchmark` safely with deterministic behavior unchanged. I also change `torch.compile` to a much lower-overhead configuration (still compiled, same numerics) to avoid compilation timeouts. Everything else (paths, training loop, weights, transforms, optimizer/scheduler) remains the same.'
- What this solution (achieved 0.39163) has done: 'Your score is far below the target, so the smallest likely win is improving training signal without changing the core “ResNet50 backbone + replace fc + fine-tune layer4+fc + CrossEntropy + argmax inference” approach. The biggest bug-like issue here is that your validation dataset uses `train_aug=False` but still passes through `infer_tfms` (simple resize), which is mismatched with the training crop distribution; switching val to a deterministic center-crop pipeline better matches typical ImageNet/Cassava evaluation and gives a more reliable best-epoch selection. Separately, you are using `cudnn.deterministic=True` while also enabling `benchmark=True` (conflicting); making this consistent improves stability and avoids performance regressions that can hurt convergence. Finally, I make sure the model is moved to `eval()` before compilation/inference and that compilation never changes model state, keeping semantics identical.'
- What this solution (achieved 0.36323) has done: 'Your current score (0.39163) is far below the target (0.86899), so we should improve generalization while keeping your exact core approach (ResNet50 pretrained, replace `fc`, fine-tune `layer4+fc`, CrossEntropy, argmax inference). The biggest low-risk issue is that you’re training from an ImageNet checkpoint without adapting BatchNorm statistics to the cassava domain; enabling BatchNorm updates (while keeping only `layer4+fc` learnable) typically yields a large accuracy jump for minimal code change. Second, your RandomResizedCrop is relatively mild; slightly widening its scale/ratio improves invariance without changing the training paradigm. Finally, I keep submission ordering/format identical and avoid any new dependencies or major refactors.'
- What this solution (achieved 0.31801) has done: 'Your current score is far below the target, so we should raise accuracy with the smallest changes that keep your core “ResNet50 pretrained + replace fc + fine-tune layer4+fc + CrossEntropy + argmax inference” approach intact. The biggest accuracy limiter in your current code is that BatchNorm layers are being forced into `train()` even during validation/inference (because `model.eval()` is called but then your train-time BN handling can leave BN behavior inconsistent); we make BN update only during training and strictly freeze BN (eval mode) during validation/inference for stable evaluation. Second, we add a lightweight Exponential Moving Average (EMA) of model weights (same architecture, same loss/optimizer/epochs) and use the EMA weights for validation selection and test inference, which typically boosts generalization toward your target without changing evaluation semantics. Finally, we keep submission generation identical but ensure no “missing filled with 0” ever happens by asserting full coverage of test_ids (so we don’t silently lose score due to any rare dataloader/image read issue).'

# 9. Code solution

## === cell 0
import os
import sys
import copy
import datetime
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import cv2
from torchvision import models, transforms


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

try:
    cv2.setUseOptimized(True)
    cv2.setNumThreads(max(1, min(4, (os.cpu_count() or 4) // 2)))
except Exception:
    pass

DATA_ROOT = "../input/cassava-leaf-disease-classification"
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_CSV_PATH = os.path.join(DATA_ROOT, "train.csv")

assert os.path.exists(TEST_IMG_DIR), f"Missing test image dir: {TEST_IMG_DIR}"
assert os.path.exists(TRAIN_IMG_DIR), f"Missing train image dir: {TRAIN_IMG_DIR}"
assert os.path.exists(SAMPLE_SUB_PATH), f"Missing sample submission: {SAMPLE_SUB_PATH}"
assert os.path.exists(TRAIN_CSV_PATH), f"Missing train csv: {TRAIN_CSV_PATH}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



## === cell 1
NUM_CLASSES = 5
BATCH_SIZE = 64
IMG_SIZE = 224

weights = models.ResNet50_Weights.IMAGENET1K_V2
model = models.resnet50(weights=weights)

in_features = model.fc.in_features
model.fc = nn.Linear(in_features, NUM_CLASSES)

if torch.cuda.is_available():
    model = model.to(device, memory_format=torch.channels_last)
else:
    model = model.to(device)

if torch.cuda.is_available():
    try:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    except Exception:
        pass

model



## === cell 2
wmeta = getattr(weights, "meta", {}) if weights is not None else {}
mean = wmeta.get("mean", (0.485, 0.456, 0.406))
std = wmeta.get("std", (0.229, 0.224, 0.225))

from torchvision.transforms import v2 as T
from torchvision.io import read_image, ImageReadMode

_TENSOR_SAFE_INTERP = transforms.InterpolationMode.BILINEAR

train_tfms = T.Compose(
    [
        T.RandomResizedCrop(
            size=(IMG_SIZE, IMG_SIZE),
            scale=(0.60, 1.0),
            ratio=(0.80, 1.25),
            interpolation=_TENSOR_SAFE_INTERP,
            antialias=True,
        ),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply(
            [T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02)],
            p=0.5,
        ),
        T.RandomRotation(
            degrees=10,
            interpolation=_TENSOR_SAFE_INTERP,
            expand=False,
            fill=0,
        ),
        T.Normalize(mean=mean, std=std),
    ]
)

val_tfms = T.Compose(
    [
        T.Resize(
            size=int(round(IMG_SIZE * 256 / 224)),
            interpolation=_TENSOR_SAFE_INTERP,
            antialias=True,
        ),
        T.CenterCrop(size=(IMG_SIZE, IMG_SIZE)),
        T.Normalize(mean=mean, std=std),
    ]
)

infer_tfms = T.Compose(
    [
        T.Resize(
            size=int(round(IMG_SIZE * 256 / 224)),
            interpolation=_TENSOR_SAFE_INTERP,
            antialias=True,
        ),
        T.CenterCrop(size=(IMG_SIZE, IMG_SIZE)),
        T.Normalize(mean=mean, std=std),
    ]
)


def _read_rgb_uint8(fp: str) -> torch.Tensor:
    return read_image(fp, mode=ImageReadMode.RGB)


def preprocess_rgb_tensor(
    img_u8_chw: torch.Tensor, train_aug: bool = False, is_val: bool = False
) -> torch.Tensor:
    x = img_u8_chw.to(dtype=torch.float32).div_(255.0)
    if train_aug:
        x = train_tfms(x)
    else:
        x = val_tfms(x) if is_val else infer_tfms(x)
    return x


class CassavaTrainSet(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        train_aug: bool = True,
        is_val: bool = False,
    ):
        df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.train_aug = train_aug
        self.is_val = is_val
        self.image_ids = df["image_id"].tolist()
        self.labels = df["label"].astype(np.int64).to_numpy()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        fn = self.image_ids[idx]
        y = int(self.labels[idx])
        fp = os.path.join(self.image_dir, fn)
        try:
            img = _read_rgb_uint8(fp)
        except Exception as e:
            raise FileNotFoundError(f"Failed to read image: {fp}") from e
        x = preprocess_rgb_tensor(img, train_aug=self.train_aug, is_val=self.is_val)
        return x, y


class InferSet(Dataset):
    def __init__(self, image_dir: str, image_ids):
        self.image_dir = image_dir
        self.image_ids = list(image_ids)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        fn = self.image_ids[idx]
        fp = os.path.join(self.image_dir, fn)
        try:
            img = _read_rgb_uint8(fp)
        except Exception as e:
            raise FileNotFoundError(f"Failed to read image: {fp}") from e
        x = preprocess_rgb_tensor(img, train_aug=False, is_val=False)
        return x, fn


train_df = pd.read_csv(TRAIN_CSV_PATH)
train_df.head(), train_df.shape




## === cell 3
def stratified_split(
    df: pd.DataFrame, label_col: str, val_frac: float = 0.1, seed: int = 42
):
    rng = np.random.RandomState(seed)
    val_indices = []
    for lbl, g in df.groupby(label_col):
        idxs = g.index.to_numpy()
        rng.shuffle(idxs)
        n_val = max(1, int(round(len(idxs) * val_frac)))
        val_indices.extend(idxs[:n_val].tolist())
    val_mask = df.index.isin(val_indices)
    return df.loc[~val_mask].reset_index(drop=True), df.loc[val_mask].reset_index(
        drop=True
    )


tr_df, va_df = stratified_split(train_df, "label", val_frac=0.1, seed=42)
print("train/val sizes:", tr_df.shape, va_df.shape)
print("train label dist:\n", tr_df["label"].value_counts(normalize=True).sort_index())
print("val   label dist:\n", va_df["label"].value_counts(normalize=True).sort_index())




## === cell 4
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {}
        self.backup = {}

        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for name, p in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(d).add_(p.detach(), alpha=(1.0 - d))

    def apply_shadow(self, model: nn.Module):
        self.backup = {}
        for name, p in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = p.detach().clone()
                p.data.copy_(self.shadow[name].data)

    def restore(self, model: nn.Module):
        for name, p in model.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name].data)
        self.backup = {}


for p in model.parameters():
    p.requires_grad = False
for p in model.layer4.parameters():
    p.requires_grad = True
for p in model.fc.parameters():
    p.requires_grad = True

train_ds = CassavaTrainSet(tr_df, TRAIN_IMG_DIR, train_aug=True, is_val=False)
val_ds = CassavaTrainSet(va_df, TRAIN_IMG_DIR, train_aug=False, is_val=True)

cpu_cnt = os.cpu_count() or 2
nw = min(6, cpu_cnt)
pin = torch.cuda.is_available()


def _worker_init_fn(worker_id: int):
    seed = 42 + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass


def _collate_train(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs, 0), torch.tensor(ys, dtype=torch.int64)


def _collate_infer(batch):
    xs, fns = zip(*batch)
    return torch.stack(xs, 0), list(fns)


_pf = 4 if nw > 0 else None

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=nw,
    pin_memory=pin,
    persistent_workers=(nw > 0),
    prefetch_factor=_pf,
    worker_init_fn=_worker_init_fn if nw > 0 else None,
    collate_fn=_collate_train,
)
val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=nw,
    pin_memory=pin,
    persistent_workers=(nw > 0),
    prefetch_factor=_pf,
    worker_init_fn=_worker_init_fn if nw > 0 else None,
    collate_fn=_collate_train,
)

counts = tr_df["label"].value_counts().sort_index()
freq = counts.values.astype(np.float32)
cls_w = freq.sum() / (NUM_CLASSES * freq)
cls_w = torch.tensor(cls_w, dtype=torch.float32, device=device)

criterion = nn.CrossEntropyLoss(weight=cls_w, label_smoothing=0.05)

optimizer = torch.optim.AdamW(
    [
        {"params": model.layer4.parameters(), "lr": 5e-5, "weight_decay": 1e-2},
        {"params": model.fc.parameters(), "lr": 5e-4, "weight_decay": 1e-2},
    ]
)

EPOCHS = 6
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-6
)

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

ema = EMA(model, decay=0.999)

best_state = None
best_val_acc = -1.0
start = datetime.datetime.now()

for epoch in range(EPOCHS):
    running_loss = 0.0
    correct = 0
    total = 0

    model.train()

    for xb, yb in train_loader:
        xb = xb.to(device, non_blocking=True)
        if torch.cuda.is_available():
            xb = xb.contiguous(memory_format=torch.channels_last)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        ema.update(model)

        bs = xb.size(0)
        running_loss += float(loss.item()) * bs
        preds = logits.argmax(dim=1)
        correct += int((preds == yb).sum().item())
        total += int(bs)

    scheduler.step()

    train_loss = running_loss / max(1, total)
    train_acc = correct / max(1, total)

    model.eval()
    ema.apply_shadow(model)
    v_correct = 0
    v_total = 0
    with torch.inference_mode():
        for xb, yb in val_loader:
            xb = xb.to(device, non_blocking=True)
            if torch.cuda.is_available():
                xb = xb.contiguous(memory_format=torch.channels_last)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            preds = logits.argmax(dim=1)
            v_correct += int((preds == yb).sum().item())
            v_total += int(xb.size(0))
    val_acc = v_correct / max(1, v_total)
    ema.restore(model)

    print(
        f"epoch {epoch+1}/{EPOCHS} - loss: {train_loss:.4f} - train_acc: {train_acc:.4f} - val_acc: {val_acc:.4f}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        ema.apply_shadow(model)
        best_state = copy.deepcopy(model.state_dict())
        ema.restore(model)

print("training time:", datetime.datetime.now() - start)
print("best_val_acc:", best_val_acc)

if best_state is not None:
    model.load_state_dict(best_state)

model.eval()



## === cell 5
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
test_ids = sample_sub["image_id"].tolist()

infer_ds = InferSet(TEST_IMG_DIR, test_ids)
infer_loader = DataLoader(
    infer_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=nw,
    pin_memory=pin,
    persistent_workers=(nw > 0),
    prefetch_factor=_pf,
    worker_init_fn=_worker_init_fn if nw > 0 else None,
    collate_fn=_collate_infer,
)



## === cell 6
all_fns = []
all_preds = []

model.eval()
with torch.inference_mode():
    for xb, fns in infer_loader:
        xb = xb.to(device, non_blocking=True)
        if torch.cuda.is_available():
            xb = xb.contiguous(memory_format=torch.channels_last)
        logits = model(xb)
        preds = logits.argmax(dim=1).to("cpu", non_blocking=True).numpy()
        all_fns.extend(fns)
        all_preds.extend(preds.tolist())

len(all_fns), len(all_preds), sample_sub.shape



## === cell 7
pred_map = dict(zip(all_fns, all_preds))
sub = sample_sub.copy()

missing = int((~sub["image_id"].isin(pred_map)).sum())
assert (
    missing == 0
), f"Missing predictions for {missing} test images; fix data loading before submitting."

sub["label"] = sub["image_id"].map(pred_map).astype(int)

sub.to_csv("./submission.csv", index=False)

print("submission.csv written:", sub.shape, "missing:", missing)
sub.head()



## === cell 8
chk = pd.read_csv("./submission.csv")
print(chk.shape)
print(chk.columns.tolist())
print(chk["label"].value_counts().sort_index())
assert (
    chk.shape[0] == sample_sub.shape[0]
), "Submission length mismatch vs sample_submission"
assert chk.columns.tolist() == ["image_id", "label"], "Wrong submission columns"
assert chk["label"].between(0, 4).all(), "Labels must be integers in [0,4]"
print("Submission looks valid.")
