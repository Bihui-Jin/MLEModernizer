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

0.8983076458144454

# 6. Current score

0.05867

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11323) has done: 'I fix the Albumentations v2 API break by updating `RandomResizedCrop` to use the new `size=(h, w)` argument so the augmentation pipelines construct correctly. I also fix the missing weight files issue by switching `INPUT_PATH` to the actual provided dataset directory and adding a safe fallback: if the `.pth` files aren’t present, run the same inference pipeline with randomly initialized models (still producing a valid submission). Finally, I ensure test images are ordered to match `sample_submission.csv` (avoids accidental misalignment) and make device handling robust for CPU-only environments while keeping the model/ensemble logic unchanged.'
- What this solution (achieved 0.1136) has done: 'I remove the hard failure when checkpoints can’t be found and make `try_load_state_dict` safely handle `None` paths, so the notebook always runs end-to-end and writes `submission.csv`. I also fix the focal-loss helper (even though it’s unused) to avoid incorrect math if it ever gets called, without changing the inference core logic. Finally, I make inference robust by only wrapping with `DataParallel` when multiple GPUs exist and by ensuring test image ordering matches `sample_submission.csv`, producing a valid submission in the required format.'
- What this solution (achieved 0.11286) has done: 'Your current score (~0.11) strongly suggests the checkpoints aren’t being loaded and you’re effectively doing random-weight inference; the smallest legitimate improvement is to reliably locate and load the `.pth` files if they exist in the dataset, otherwise keep the same fallback. I tighten checkpoint resolution to search specifically under the competition dataset directories first (where Kaggle normally stores uploaded weights), and I also ensure the model is moved to device before loading state dict to avoid any subtle key/device issues. Finally, I keep your exact ensemble/TTA/core inference logic unchanged, but make inference deterministic per-image (fixed TTA randomness) so results are stable run-to-run and not accidentally worse.'
- What this solution (achieved 0.11286) has done: 'Your score (~0.11) indicates the checkpoints still aren’t being loaded (random-weight inference). I make the smallest change that plausibly moves accuracy toward your target by resolving checkpoints more aggressively: first search by exact filename, then (if not found) search for any `.pth` under the competition dataset and pick the best match for each model. I also add a clear “weights_loaded” flag to confirm at runtime whether each model actually loaded weights (so you can verify you’re no longer submitting random predictions). Everything else (models, TTA, normalization, ensembling, submission ordering/format) stays the same.'
- What this solution (achieved 0.11472) has done: 'Your score (~0.11) is consistent with effectively random predictions, and the most likely cause is that no trained checkpoints are actually being loaded. I keep your ensemble/TTA/model definitions exactly the same, but make checkpoint resolution and loading robust to common Kaggle `.pth` formats (e.g., `model`, `model_state_dict`, `net`, Lightning `state_dict`, nested dicts) and to weight files being placed anywhere under `/kaggle/input`. I also add a safe key-renaming step to handle classifier head naming differences (`fc` vs `classifier`) without changing architecture, so weights that match in shape can load. These are minimal changes intended to move accuracy substantially upward toward your target without altering inference semantics or submission formatting.'
- What this solution (achieved 0.11286) has done: 'Your low accuracy strongly indicates the intended checkpoints still aren’t being loaded correctly (so you’re effectively predicting from random weights). I make the smallest changes focused on reliably finding and loading the two `.pth` files by searching across `/kaggle/input/**` and `/kaggle/data/**`, and I add robust state-dict cleanup (remove common prefixes like `model.`/`module.`, ignore non-tensor entries) so `load_state_dict` actually matches your current `timm` model keys without changing the model itself. I also ensure inference uses the loaded model object on the same device (and only wraps with `DataParallel` after loading), keeping your ensemble weights, TTA, augmentations, normalization, and submission formatting unchanged. These changes should move the score substantially upward toward your target if the correct weights exist in the environment; otherwise behavior remains identical to your current fallback.'
- What this solution (achieved 0.58558) has done: 'Your score (~0.11) is consistent with random predictions, so the most direct way to move toward the ~0.898 target is to ensure you’re actually using meaningful weights. I keep your exact models/ensemble/TTA logic, but (1) fix the likely bad dataset paths for this environment (`/kaggle/input/...`) so images and any checkpoints can be found, (2) make checkpoint resolution search the whole dataset tree under the cassava dataset directory first, and (3) if checkpoints still aren’t present, fall back to using `pretrained=True` (ImageNet weights) rather than random initialization—this preserves architecture and inference semantics while materially improving accuracy. I also fix one issue that can silently break submission alignment: `image_id` must exactly match `sample_submission.csv` order, which we keep.'
- What this solution (achieved 0.44245) has done: 'Your current score (0.58558) is far below the target (0.8983), and the code is still likely underperforming because (a) the classifier heads are being randomly re-initialized even when checkpoints are loaded, and (b) inference is effectively single-image, non-batched, and uses stochastic test augmentations that can hurt accuracy for ImageNet-pretrained fallbacks. I make the smallest changes that keep your exact model architectures and ensembling logic, but ensure that if checkpoints exist they actually take effect by not overwriting the loaded heads, and I make test-time preprocessing deterministic (Resize+CenterCrop) while preserving your TTA loop count. I also switch inference to a DataLoader (no change to model logic) so the same compute budget is spent on more consistent predictions rather than Python overhead. These changes are directly aimed at improving accuracy toward your target while still producing a valid `submission.csv`.'
- What this solution (achieved 0.05867) has done: 'Your current score is far below the target, and the biggest “minimal-change” lever is to ensure the custom checkpoints (if present) actually load into the exact same classifier-head structure they were trained with. I keep your two-model ensemble, TTA, normalization, and argmax submission exactly the same, but change model construction to create the 5-class heads via `timm.create_model(..., num_classes=5)` (instead of replacing `.fc`/`.classifier` afterward), which avoids silent head/key mismatches and improves checkpoint compatibility. I also make `try_load_state_dict` more robust to common timm checkpoint formats by filtering out mismatched tensor shapes (so the rest of the network loads cleanly) while still preserving the exact inference logic. Finally, I keep the test ordering locked to `sample_submission.csv` and still write `submission.csv` end-to-end.'

# 9. Code solution

## === cell 0
import os
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm import tqdm
import timm



## === cell 1
INPUT_PATH = "/kaggle/input/cassava-leaf-disease-classification/"
TRAIN_CSV_PATH = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
TRAIN_IMAGE_PATH = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
TEST_IMAGE_PATH = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
SUBMISSION_PATH = "submission.csv"

RESNEXT_PATH = "1022_res50.pth"
B4_PATH = "1022_b4ns.pth"

OUT_FEATURES = 5
NUM_EPOCHS = 17
BATCH_SIZE = 32
IMAGE_SIZE = 512
OPTIMIZER = torch.optim.AdamW
SEED = 42
LR_START = 1e-5
LR_MAX = 2e-4
LR_FINAL = 1e-5
TTA = 8



## === cell 2
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICES = (
    [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    if torch.cuda.is_available()
    else [torch.device("cpu")]
)




## === cell 3
def sigmoid_focal_cross_entropy(y_hat, y_true, alpha=0.25, gamma=2.0):
    def smooth(y, smooth_factor):
        assert len(y.shape) == 2
        y = y * (1 - smooth_factor)
        y = y + smooth_factor / y.shape[1]
        return y

    smooth_factor = 0.1

    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)
    if not isinstance(y_hat, torch.Tensor):
        y_hat = torch.tensor(y_hat)

    y_true = y_true.to(dtype=y_hat.dtype, device=y_hat.device)
    y_true = smooth(y_true, smooth_factor)

    prob = torch.sigmoid(y_hat)
    p_t = y_true * prob + (1 - y_true) * (1 - prob)
    alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)

    ce = F.binary_cross_entropy_with_logits(y_hat, y_true, reduction="none")
    modulating_factor = (1.0 - p_t).pow(gamma)

    return torch.sum(alpha_t * modulating_factor * ce, dim=-1)




## === cell 4
def resolve_checkpoint_path(
    ckpt_name: str, preferred_root: str = INPUT_PATH
) -> str | None:
    direct = os.path.join(preferred_root, ckpt_name)
    if os.path.exists(direct):
        return direct

    search_roots = [
        Path(preferred_root),
        Path("/kaggle/input/cassava-leaf-disease-classification"),
        Path("/kaggle/input"),
        Path("/kaggle/data/input"),
        Path("/kaggle/data"),
        Path("../input"),
        Path("."),
    ]

    seen = set()
    for root in search_roots:
        try:
            root = root.resolve()
        except Exception:
            continue
        if str(root) in seen or not root.exists():
            continue
        seen.add(str(root))
        try:
            hits = list(root.rglob(ckpt_name))
        except Exception:
            hits = []
        if hits:
            hits_sorted = sorted(hits, key=lambda p: (len(str(p)), str(p)))
            return str(hits_sorted[0])

    def tokenize(s: str) -> set[str]:
        return {
            t
            for t in "".join([c.lower() if c.isalnum() else " " for c in s]).split()
            if t
        }

    target_tokens = tokenize(ckpt_name)
    candidates: list[Path] = []
    for root in search_roots:
        try:
            root = root.resolve()
        except Exception:
            continue
        if not root.exists():
            continue
        try:
            candidates.extend(list(root.rglob("*.pth")))
        except Exception:
            continue

    if not candidates:
        return None

    def score(p: Path) -> tuple:
        pt = tokenize(p.name)
        overlap = len(target_tokens.intersection(pt))
        return (-overlap, len(str(p)), str(p))

    candidates_sorted = sorted(candidates, key=score)
    best = candidates_sorted[0]
    if len(target_tokens.intersection(tokenize(best.name))) == 0:
        return None
    return str(best)




## === cell 5
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



## === cell 6
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



## === cell 7
test_augs = A.Compose(
    [
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.CenterCrop(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
        ToTensorV2(p=1.0),
    ],
    p=1.0,
)




## === cell 8
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




## === cell 9
def _extract_state_dict(state_obj):
    if isinstance(state_obj, dict):
        for k in [
            "state_dict",
            "model",
            "model_state_dict",
            "net",
            "weights",
            "params",
        ]:
            if k in state_obj and isinstance(state_obj[k], dict):
                return state_obj[k]
    return state_obj


def _clean_state_dict_keys(sd: dict) -> dict:
    if not isinstance(sd, dict):
        return sd
    out = {}
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        nk = k
        for pref in ("module.", "model."):
            if nk.startswith(pref):
                nk = nk[len(pref) :]
        out[nk] = v
    return out


def _remap_head_keys_if_needed(model: nn.Module, sd: dict) -> dict:
    if not isinstance(sd, dict):
        return sd
    model_sd = model.state_dict()
    out = dict(sd)

    def maybe_map(src_prefix, dst_prefix):
        for suf in ["weight", "bias"]:
            src_k = f"{src_prefix}.{suf}"
            dst_k = f"{dst_prefix}.{suf}"
            if src_k in out and dst_k in model_sd and dst_k not in out:
                if out[src_k].shape == model_sd[dst_k].shape:
                    out[dst_k] = out[src_k]

    maybe_map("classifier", "fc")
    maybe_map("fc", "classifier")
    return out


def _filter_state_dict_by_shape(model: nn.Module, sd: dict) -> dict:
    if not isinstance(sd, dict):
        return sd
    model_sd = model.state_dict()
    out = {}
    dropped = 0
    for k, v in sd.items():
        if (
            k in model_sd
            and isinstance(v, torch.Tensor)
            and isinstance(model_sd[k], torch.Tensor)
        ):
            if v.shape == model_sd[k].shape:
                out[k] = v
            else:
                dropped += 1
    if dropped:
        print(
            f"[INFO] Dropped {dropped} tensors due to shape mismatch (kept backbone-compatible weights)."
        )
    return out


def try_load_state_dict(model: nn.Module, ckpt_path: str | None) -> bool:
    if ckpt_path is None:
        print("[WARN] Checkpoint path is None. Proceeding without loading weights.")
        return False
    if not os.path.exists(ckpt_path):
        print(
            f"[WARN] Checkpoint not found: {ckpt_path}. Proceeding without loading weights."
        )
        return False

    state = torch.load(ckpt_path, map_location="cpu")
    state = _extract_state_dict(state)

    if not isinstance(state, dict):
        print(
            f"[WARN] Unsupported checkpoint content type ({type(state)}). Proceeding without loading weights."
        )
        return False

    state = _clean_state_dict_keys(state)
    state = _remap_head_keys_if_needed(model, state)
    state = _filter_state_dict_by_shape(model, state)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(
            f"[WARN] Missing keys when loading {ckpt_path}: {missing[:8]}{'...' if len(missing) > 8 else ''}"
        )
    if unexpected:
        print(
            f"[WARN] Unexpected keys when loading {ckpt_path}: {unexpected[:8]}{'...' if len(unexpected) > 8 else ''}"
        )
    print(f"[INFO] Loaded checkpoint: {ckpt_path}")
    return True




## === cell 10
model_name1 = "resnext50_32x4d"
my_model_1 = timm.create_model(model_name1, pretrained=True, num_classes=OUT_FEATURES)
my_model_1



## === cell 11
model_name2 = "tf_efficientnet_b4_ns"
my_model_2 = timm.create_model(model_name2, pretrained=True, num_classes=OUT_FEATURES)
my_model_2



## === cell 12
if torch.cuda.is_available():
    torch.cuda.empty_cache()



## === cell 13
SAMPLE_SUB_PATH = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
test_image_list = sample_sub["image_id"].astype(str).values

missing_imgs = [
    img
    for img in test_image_list[:50]
    if not os.path.exists(os.path.join(TEST_IMAGE_PATH, img))
]
if missing_imgs:
    print(
        f"[WARN] Some test images not found under TEST_IMAGE_PATH. Example: {missing_imgs[0]}"
    )
else:
    print(f"[INFO] Found test images under: {TEST_IMAGE_PATH}")




## === cell 14
class CassavaTestDataset(torch.utils.data.Dataset):
    def __init__(self, image_ids, image_dir, augs):
        self.image_ids = list(image_ids)
        self.image_dir = image_dir
        self.augs = augs

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img = Image.open(os.path.join(self.image_dir, image_id)).convert("RGB")
        x = self.augs(image=np.array(img))["image"].float()
        return image_id, x


def predict_logits(model: nn.Module, image_ids, tta: int = 1, batch_size: int = 32):
    ds = CassavaTestDataset(image_ids, TEST_IMAGE_PATH, test_augs)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    all_logits = []
    model.eval()
    with torch.no_grad():
        for _ in range(tta):
            batch_logits = []
            for _, xb in loader:
                xb = xb.to(DEVICE, non_blocking=True)
                out = model(xb)
                batch_logits.append(out.detach().cpu())
            all_logits.append(torch.cat(batch_logits, dim=0))

    logits = torch.stack(all_logits, dim=0).mean(dim=0)
    return logits


ckpt1 = resolve_checkpoint_path(RESNEXT_PATH, preferred_root=INPUT_PATH)
ckpt2 = resolve_checkpoint_path(B4_PATH, preferred_root=INPUT_PATH)

print(f"Resolved ckpt1: {ckpt1}")
print(f"Resolved ckpt2: {ckpt2}")

if ckpt1 is None or ckpt2 is None:
    print(
        "[WARN] One or more checkpoints could not be resolved. "
        "Will proceed using ImageNet-pretrained backbones."
    )

my_model_1 = my_model_1.to(DEVICE)
loaded_1 = try_load_state_dict(my_model_1, ckpt1)
print(f"[INFO] model_1 weights_loaded={loaded_1}")
if torch.cuda.device_count() > 1:
    my_model_1 = nn.DataParallel(my_model_1)

predictions_1 = predict_logits(
    my_model_1, test_image_list, tta=1, batch_size=BATCH_SIZE
)
normalize_pred_1 = F.normalize(predictions_1.T, p=2, dim=0).T

if torch.cuda.is_available():
    torch.cuda.empty_cache()

my_model_2 = my_model_2.to(DEVICE)
loaded_2 = try_load_state_dict(my_model_2, ckpt2)
print(f"[INFO] model_2 weights_loaded={loaded_2}")
if torch.cuda.device_count() > 1:
    my_model_2 = nn.DataParallel(my_model_2)

predictions_2 = predict_logits(
    my_model_2, test_image_list, tta=TTA, batch_size=BATCH_SIZE
)
normalize_pred_2 = F.normalize(predictions_2.T, p=2, dim=0).T

final_pred = (normalize_pred_1 * 0.43) + (normalize_pred_2 * 0.57)

label = final_pred.argmax(dim=-1).numpy().astype(int)
df_submission = pd.DataFrame({"image_id": test_image_list, "label": label})
df_submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Wrote submission to: {SUBMISSION_PATH}  (rows={len(df_submission)})")
print(df_submission.head())
