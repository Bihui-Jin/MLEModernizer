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

3.12

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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

0.8943789664551224

# 6. Current score

0.53251

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11659) has done: 'I fix the Albumentations API breakage by updating `RandomResizedCrop` calls to the new `size=(h,w)` signature so augmentations build correctly under albumentations==2.x. Then I fix the missing checkpoint path by auto-detecting the real dataset root (`/kaggle/input/...`) and adding a safe fallback: if the external ensemble weights aren’t present, the code still run end-to-end by using the untrained models (score be low, but it produce a valid submission). I also make inference deterministic and submission-order-stable by sorting `test_image_list`, and I ensure device handling doesn’t crash when CUDA is unavailable. All changes are minimal and keep the same core ensemble/inference logic when weights are found.'
- What this solution (achieved 0.5867) has done: 'I fix the albumentations==2.x API break that stops execution by replacing the removed `A.Cutout` with the closest equivalent `A.CoarseDropout`, keeping augmentation intent the same. I also ensure both models are moved to the same device used for inference (the current code can silently place DataParallel on a different device than `DEVICE`, hurting correctness or crashing). Finally, I enable `pretrained=True` as a minimal, standard change (same architecture/training loop) to move accuracy substantially toward the target; your current score strongly indicates you’re effectively using untrained weights (or mismatched inference). The submission writing and ordering stay unchanged and still produce a valid `submission.csv`.'
- What this solution (achieved 0.48804) has done: 'Your current score suggests the inference post-processing is hurting accuracy more than helping; the biggest issue is applying `F.normalize` (L2 normalization) to logits across classes, which distorts relative class confidence and tends to reduce argmax accuracy. I remove that normalization and instead ensemble in probability space (softmax), which matches the accuracy metric while keeping the same two-model ensemble/inference flow. I also fix a silent TTA bug (model 1 currently uses `range(1)` instead of `TTA`), making both models use the configured TTA consistently. These are minimal, metric-aligned changes and should move the score upward toward your target without changing architecture or training.'
- What this solution (achieved 0.49178) has done: 'Your current score is far below the target, which strongly suggests your intended trained checkpoints still aren’t being used at inference time (or are being loaded into the wrong module due to `DataParallel` key mismatches). I make the smallest change that materially improves accuracy: load checkpoints **after** wrapping the model with `nn.DataParallel`, and update `_try_load_weights` to correctly handle both `module.` and non-`module.` state dict keys depending on whether the target model is wrapped. I also resolve `INPUT_PATH` to common Kaggle dataset roots so the checkpoint files are actually found when present, while keeping the exact same ensemble, TTA, and probability-space averaging logic. These changes preserve your architecture and inference semantics, but should move the score substantially upward toward the target by ensuring you’re using the trained weights.'
- What this solution (achieved 0.4843) has done: 'Your score is far below the target, which strongly indicates the intended trained checkpoints still aren’t being found/loaded, so the ensemble is mostly using ImageNet-pretrained (or random-head) weights. I make the smallest changes that (1) robustly resolve the real checkpoint directory under `/kaggle/input` by auto-searching for the two `.pth` files, and (2) load common checkpoint formats (raw `state_dict`, `{"state_dict":...}`, `{"model":...}`) while correctly handling `DataParallel` `module.` prefixes. This preserves your exact model architectures, TTA flow, and probability-space ensembling; it just makes weight loading actually work when the files exist, which should move accuracy substantially toward your target. Submission writing, column names, and row ordering remain unchanged and valid.'
- What this solution (achieved 0.49514) has done: 'Your score is far below the target, so the most likely cause is still “not actually using the intended trained checkpoints.” I make the smallest changes that (1) correctly set both timm models’ classifier heads via `reset_classifier` (so the checkpoint keys match timm’s expected module names), and (2) make weight loading robust to checkpoints saved from DataParallel/EMA/Lightning by stripping common prefixes like `module.` and `model.` before loading. This keeps the same two-model ensemble, same TTA loop, same augmentations, and same softmax-probability averaging, but greatly increases the chance that your `.pth` weights load fully (which should move accuracy up toward your target). The script still run end-to-end and write a valid `submission.csv` even if checkpoints are missing.'
- What this solution (achieved 0.55717) has done: 'Your score is far below the target, so we should make a small, high-impact fix that improves accuracy without changing the model architectures or the overall inference approach. The biggest likely issue is that your checkpoints were trained at a different input resolution than 512, and using the wrong resolution at inference can heavily hurt accuracy even when weights load correctly. I keep your exact two-model, softmax-probability averaging ensemble and TTA loop, but I (1) infer the expected input size from each model (and/or checkpoint) and resize accordingly per model, and (2) ensure the final submission strictly follows `sample_submission.csv` ordering to avoid any accidental misalignment. These are minimal, score-relevant adjustments that typically move accuracy substantially upward when resolution mismatch is the culprit.'
- What this solution (achieved 0.55344) has done: 'Your current score is far below the target, so we should make a small fix that increases accuracy without changing your ensemble/TTA logic. The most likely remaining issue is a normalization mismatch: timm models expect the *model-specific* mean/std from `default_cfg`, but your pipeline always uses ImageNet mean/std; this can significantly hurt accuracy even when checkpoints are correctly loaded. I keep the same two-model setup and probability-averaging ensemble, but make test-time normalization per-model using `model.default_cfg` (with your current ImageNet values as fallback). I also ensure the submission rows exactly follow `sample_submission.csv` order (already mostly true) and add a strict alignment safeguard.'
- What this solution (achieved 0.53251) has done: 'Your current score is far below the target, so we should make a small, high-impact fix without changing the model architectures or the overall ensemble/TTA approach. The biggest remaining correctness issue is that the inference loop is running one image at a time and repeatedly re-opening files; this is slow and can cause timeouts/partial runs, and it also prevents using stable batched tensor shapes that DataParallel expects. I keep the exact same augmentations/TTA/probability-averaging logic, but switch inference to a minimal `Dataset`+`DataLoader` that batches images (still applying Albumentations per-sample) and preserves `sample_submission.csv` ordering. This should complete comfortably within the 600s budget and typically improves accuracy stability by ensuring both models actually run full inference consistently.'

# 9. Code solution

## === cell 0
import os
import math
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F

import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib.pyplot as plt
from tqdm import tqdm
import timm




## === cell 1
def _resolve_first_existing(*candidates: str) -> str:
    for p in candidates:
        if p is None:
            continue
        if os.path.exists(p):
            return p
    return candidates[0]


def _find_file_under_roots(filename: str, roots: list[str]) -> str | None:
    """Minimal robustness: find the first occurrence of filename under provided roots."""
    for root in roots:
        if not root or not os.path.exists(root):
            continue
        direct = os.path.join(root, filename)
        if os.path.exists(direct):
            return direct
        for dirpath, _, files in os.walk(root):
            if filename in files:
                return os.path.join(dirpath, filename)
    return None




## === cell 2
INPUT_PATH = "../input/ensemble-1023/"
TRAIN_CSV_PATH = "../input/cassava-leaf-disease-classification/train.csv"
TRAIN_IMAGE_PATH = "../input/cassava-leaf-disease-classification/train_images/"
TEST_IMAGE_PATH = "../input/cassava-leaf-disease-classification/test_images/"
SUBMISSION_PATH = "submission.csv"
RESNEXT_PATH = "1022_res50.pth"
B4_PATH = "1022_b4ns.pth"
DEVICES = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
OUT_FEATURES = 5
NUM_EPOCHS = 17
BATCH_SIZE = 32
IMAGE_SIZE = 512
OPTIMIZER = torch.optim.AdamW
SEED = 42
LR_START = 1e-5
LR_MAX = 2e-4
LR_FINAL = 1e-5
TTA = 3

INPUT_PATH = _resolve_first_existing(
    INPUT_PATH,
    "/kaggle/input/ensemble-1023/",
    "/kaggle/input/ensemble-1023",
    "/kaggle/input/ensemble1023/",
    "/kaggle/input/ensemble1023",
    "/kaggle/input/",
)

TRAIN_CSV_PATH = _resolve_first_existing(
    TRAIN_CSV_PATH,
    "/kaggle/input/cassava-leaf-disease-classification/train.csv",
)
TRAIN_IMAGE_PATH = _resolve_first_existing(
    TRAIN_IMAGE_PATH,
    "/kaggle/input/cassava-leaf-disease-classification/train_images/",
)
TEST_IMAGE_PATH = _resolve_first_existing(
    TEST_IMAGE_PATH,
    "/kaggle/input/cassava-leaf-disease-classification/test_images/",
)



## === cell 3
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if len(DEVICES) == 0:
    DEVICES = [DEVICE]




## === cell 4
def sigmoid_focal_cross_entropy(y_hat, y_true, alpha=0.25, gamma=2.0):
    def smooth(y, smooth_factor):
        assert len(y.shape) == 2
        y *= 1 - smooth_factor
        y += smooth_factor / y.shape[1]
        return y

    smooth_factor = 0.1

    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)
    if not isinstance(y_hat, torch.Tensor):
        y_hat = torch.tensor(y_hat)

    y_true = smooth(y_true, smooth_factor)

    cross_entropy = F.binary_cross_entropy_with_logits(y_hat, y_true, reduction="none")
    p_t = y_true * y_hat + (1 - y_true) * (1 - y_hat)
    alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
    modulating_factor = (1.0 - p_t).pow(gamma)

    return torch.sum(alpha_t * modulating_factor * cross_entropy, dim=-1)




## === cell 5
def _unwrap_state_dict(state):
    """
    Handle common checkpoint wrappers so trained weights actually load.
    Accepts:
      - raw state_dict
      - {"state_dict": ...}
      - {"model": ...}
      - {"model_state_dict": ...}
    """
    if not isinstance(state, dict):
        return None
    if len(state) > 0 and all(isinstance(k, str) for k in state.keys()):
        if any("." in k for k in state.keys()):
            return state
    for key in ("state_dict", "model", "model_state_dict", "net", "network"):
        if key in state and isinstance(state[key], dict):
            return state[key]
    return state  # best-effort fallback


def _strip_known_prefixes(state: dict) -> dict:
    """
    Strip common prefixes to match timm model keys.
    """
    if not isinstance(state, dict) or len(state) == 0:
        return state

    prefixes = ("module.", "model.", "net.", "network.", "ema.", "student.", "encoder.")
    for pref in prefixes:
        keys = list(state.keys())
        share = sum(k.startswith(pref) for k in keys)
        if share >= 0.8 * len(keys):
            state = {k[len(pref) :]: v for k, v in state.items() if k.startswith(pref)}
            return state
    return state


def _try_load_weights(model: nn.Module, ckpt_path: str) -> bool:
    if (ckpt_path is None) or (not os.path.exists(ckpt_path)):
        print(f"[WARN] Checkpoint not found: {ckpt_path}. Using fallback weights.")
        return False

    state = torch.load(ckpt_path, map_location="cpu")
    state = _unwrap_state_dict(state)
    if not isinstance(state, dict) or len(state) == 0:
        print(
            f"[WARN] Unexpected/empty checkpoint format at {ckpt_path}. Using fallback weights."
        )
        return False

    state = _strip_known_prefixes(state)

    target_is_dp = isinstance(model, nn.DataParallel)
    has_module_prefix = any(k.startswith("module.") for k in state.keys())

    if target_is_dp and not has_module_prefix:
        state = {f"module.{k}": v for k, v in state.items()}
    elif (not target_is_dp) and has_module_prefix:
        state = {k[7:]: v for k, v in state.items() if k.startswith("module.")}

    incompatible = model.load_state_dict(state, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if missing or unexpected:
        print(
            f"[WARN] Non-strict load. Missing: {len(missing)} Unexpected: {len(unexpected)}"
        )
    else:
        print("[INFO] Strict-equivalent load (no missing/unexpected keys).")

    print(f"[INFO] Loaded checkpoint: {ckpt_path}")
    return True




## === cell 6
def lr_tune(epoch, num_epochs=NUM_EPOCHS):
    lr_start = LR_START
    lr_max = LR_MAX
    lr_final = LR_FINAL
    lr_warmup_epoch = 4
    lr_sustain_epoch = 0
    lr_decay_epoch = num_epochs - lr_warmup_epoch - lr_sustain_epoch - 1

    if epoch <= lr_warmup_epoch:
        lr = lr_start + (lr_max - lr_start) * (epoch / lr_warmup_epoch) ** 2.5
    elif epoch < lr_warmup_epoch + lr_sustain_epoch:
        lr = lr_max
    else:
        epoch_diff = epoch - lr_warmup_epoch - lr_sustain_epoch
        decay_factor = (epoch_diff / lr_decay_epoch) * math.pi
        decay_factor = (torch.cos(torch.tensor(decay_factor)).numpy() + 1) / 2
        lr = lr_final + (lr_max - lr_final) * decay_factor
    return lr


x = [i for i in range(NUM_EPOCHS)]
y = [lr_tune(i) for i in x]
plt.plot(x, y)



## === cell 7
train_augs = A.Compose(
    [
        A.RandomResizedCrop(size=(IMAGE_SIZE, IMAGE_SIZE)),
        A.Transpose(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.HueSaturationValue(
            hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
        A.CoarseDropout(p=0.5),
        A.CoarseDropout(p=0.5),
        ToTensorV2(p=1.0),
    ],
    p=1.0,
)

valid_augs = A.Compose(
    [
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.CenterCrop(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)




## === cell 8
def _infer_model_input_size(model: nn.Module, fallback: int) -> int:
    m = model.module if isinstance(model, nn.DataParallel) else model
    cfg = getattr(m, "default_cfg", None)
    if isinstance(cfg, dict):
        isz = cfg.get("input_size", None)
        if isinstance(isz, (tuple, list)) and len(isz) == 3:
            h, w = int(isz[1]), int(isz[2])
            if h > 0 and w > 0:
                return int(min(h, w))
    return int(fallback)


def _infer_model_norm(model: nn.Module):
    """
    Change (score-relevant, minimal): use per-model timm default_cfg mean/std at inference.
    This keeps the same inference logic but fixes a common accuracy drop from normalization mismatch.
    """
    m = model.module if isinstance(model, nn.DataParallel) else model
    cfg = getattr(m, "default_cfg", None)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if isinstance(cfg, dict):
        cm = cfg.get("mean", None)
        cs = cfg.get("std", None)
        if isinstance(cm, (tuple, list)) and len(cm) == 3:
            mean = [float(x) for x in cm]
        if isinstance(cs, (tuple, list)) and len(cs) == 3:
            std = [float(x) for x in cs]
    return mean, std


def _make_test_augs(image_size: int, mean, std) -> A.Compose:
    return A.Compose(
        [
            A.OneOf(
                [
                    A.Resize(image_size, image_size, p=1.0),
                    A.CenterCrop(image_size, image_size, p=1.0),
                    A.RandomResizedCrop(size=(image_size, image_size), p=1.0),
                ],
                p=1.0,
            ),
            A.Transpose(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=mean,
                std=std,
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )




## === cell 9
def seed_everything(seed=42):
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



## === cell 10
model_name1 = "resnext50_32x4d"
my_model_1 = timm.create_model(model_name1, pretrained=True)
my_model_1.reset_classifier(num_classes=OUT_FEATURES)
if hasattr(my_model_1, "get_classifier"):
    head1 = my_model_1.get_classifier()
    if isinstance(head1, nn.Linear):
        nn.init.xavier_uniform_(head1.weight)
        if head1.bias is not None:
            nn.init.zeros_(head1.bias)

model_name2 = "tf_efficientnet_b4_ns"
my_model_2 = timm.create_model(model_name2, pretrained=True)
my_model_2.reset_classifier(num_classes=OUT_FEATURES)
if hasattr(my_model_2, "get_classifier"):
    head2 = my_model_2.get_classifier()
    if isinstance(head2, nn.Linear):
        nn.init.xavier_uniform_(head2.weight)
        if head2.bias is not None:
            nn.init.zeros_(head2.bias)



## === cell 11
torch.cuda.empty_cache()



## === cell 12
ckpt_search_roots = [
    INPUT_PATH,
    "/kaggle/input",
    "/kaggle/input/cassava-leaf-disease-classification",
]

resnext_ckpt = _find_file_under_roots(RESNEXT_PATH, ckpt_search_roots)
b4_ckpt = _find_file_under_roots(B4_PATH, ckpt_search_roots)

if resnext_ckpt is None:
    resnext_ckpt = os.path.join(INPUT_PATH, RESNEXT_PATH)
if b4_ckpt is None:
    b4_ckpt = os.path.join(INPUT_PATH, B4_PATH)

sample_sub_path = _resolve_first_existing(
    os.path.join(os.path.dirname(TRAIN_CSV_PATH), "sample_submission.csv"),
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv",
)
sample_sub = pd.read_csv(sample_sub_path)

dir_list = sorted(
    [f for f in os.listdir(TEST_IMAGE_PATH) if f.lower().endswith(".jpg")]
)

if "image_id" in sample_sub.columns and len(sample_sub) > 0:
    test_image_list = sample_sub["image_id"].astype(str).values
else:
    test_image_list = np.asarray(dir_list)

infer_device = (
    DEVICES[0] if (torch.cuda.is_available() and len(DEVICES) > 0) else DEVICE
)


def _logits_to_probs(logits_1d: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits_1d, dim=-1)




## === cell 13
class CassavaTestDataset(torch.utils.data.Dataset):
    def __init__(self, image_ids, image_dir: str, augs: A.Compose):
        self.image_ids = list(image_ids)
        self.image_dir = image_dir
        self.augs = augs

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, image_id)
        image = Image.open(img_path).convert("RGB")
        x = self.augs(image=np.array(image))["image"]
        return image_id, x


def _predict_probs_batched(
    model: nn.Module,
    image_ids,
    image_dir: str,
    test_augs: A.Compose,
    tta: int,
    out_features: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
):
    ds = CassavaTestDataset(image_ids=image_ids, image_dir=image_dir, augs=test_augs)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    model.eval()
    all_probs = []
    all_image_ids = []
    with torch.no_grad():
        for batch_ids, batch_x in tqdm(dl, desc="Batched inference", leave=False):
            batch_x = batch_x.float().to(device, non_blocking=True)
            prob_accum = torch.zeros((batch_x.shape[0], out_features), device=device)
            for _ in range(tta):
                logits = model(batch_x)
                prob_accum += F.softmax(logits, dim=-1)
            prob_accum /= float(tta)
            all_probs.append(prob_accum.detach().cpu())
            all_image_ids.extend(list(batch_ids))

    probs = torch.cat(all_probs, dim=0)
    return np.asarray(all_image_ids), probs




## === cell 14
my_model_1 = nn.DataParallel(my_model_1).to(infer_device)
_ = _try_load_weights(my_model_1, resnext_ckpt)

m1_size = _infer_model_input_size(my_model_1, IMAGE_SIZE)
m1_mean, m1_std = _infer_model_norm(my_model_1)
test_augs_1 = _make_test_augs(m1_size, m1_mean, m1_std)

infer_bs = BATCH_SIZE if torch.cuda.is_available() else max(4, min(16, BATCH_SIZE))

ids1, probs_1 = _predict_probs_batched(
    model=my_model_1,
    image_ids=test_image_list,
    image_dir=TEST_IMAGE_PATH,
    test_augs=test_augs_1,
    tta=TTA,
    out_features=OUT_FEATURES,
    device=infer_device,
    batch_size=infer_bs,
    num_workers=2,
)

torch.cuda.empty_cache()



## === cell 15
my_model_2 = nn.DataParallel(my_model_2).to(infer_device)
_ = _try_load_weights(my_model_2, b4_ckpt)

m2_size = _infer_model_input_size(my_model_2, IMAGE_SIZE)
m2_mean, m2_std = _infer_model_norm(my_model_2)
test_augs_2 = _make_test_augs(m2_size, m2_mean, m2_std)

ids2, probs_2 = _predict_probs_batched(
    model=my_model_2,
    image_ids=test_image_list,
    image_dir=TEST_IMAGE_PATH,
    test_augs=test_augs_2,
    tta=TTA,
    out_features=OUT_FEATURES,
    device=infer_device,
    batch_size=infer_bs,
    num_workers=2,
)

if not np.array_equal(ids1, test_image_list) or not np.array_equal(
    ids2, test_image_list
):
    raise RuntimeError(
        "Inference order mismatch: DataLoader must be shuffle=False and aligned."
    )

final_prob = (probs_1 * 0.4) + (probs_2 * 0.6)
label = final_prob.argmax(dim=-1).numpy().astype(int)

df_submission = pd.DataFrame({"image_id": test_image_list, "label": label})

if "image_id" in sample_sub.columns and len(sample_sub) == len(df_submission):
    df_submission = sample_sub[["image_id"]].merge(
        df_submission, on="image_id", how="left"
    )
    if df_submission["label"].isna().any():
        raise RuntimeError(
            "Submission alignment failed: some labels are missing after merge."
        )
    df_submission["label"] = df_submission["label"].astype(int)

df_submission.to_csv(SUBMISSION_PATH, index=False)

print(
    f"Wrote submission to: {os.path.abspath(SUBMISSION_PATH)}  rows={len(df_submission)}"
)
print(f"[INFO] INPUT_PATH used: {INPUT_PATH}")
print(
    f"[INFO] RESNEXT ckpt resolved to: {resnext_ckpt}  exists={os.path.exists(resnext_ckpt)}"
)
print(f"[INFO] B4 ckpt resolved to: {b4_ckpt}  exists={os.path.exists(b4_ckpt)}")
print(
    f"[INFO] Model1 inferred input size: {m1_size}  norm_mean={m1_mean}  norm_std={m1_std}"
)
print(
    f"[INFO] Model2 inferred input size: {m2_size}  norm_mean={m2_mean}  norm_std={m2_std}"
)
print(f"[INFO] Inference batch size: {infer_bs}  TTA={TTA}  device={infer_device}")
