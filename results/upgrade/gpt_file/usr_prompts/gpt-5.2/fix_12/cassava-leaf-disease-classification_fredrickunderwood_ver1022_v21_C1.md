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

0.8961922030825022

# 6. Current score

0.25635

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.10912) has done: 'I fix the Albumentations runtime errors by updating `RandomResizedCrop` calls to the v2 API (`size=(H,W)` and an explicit `scale` tuple) while keeping the same augmentation intent. I also fix the missing checkpoint path by pointing `INPUT_PATH` to the competition dataset directory and adding a safe fallback: if the `.pth` files are not present, the code still run end-to-end by using the untrained models (score be low but a valid submission be produced). Finally, I make inference deterministic and robust (sorted test filenames, correct device handling, and proper tensor types) without changing the core ensemble/prediction logic. This ensures a `submission.csv` is always written in the required format.'
- What this solution (achieved 0.11024) has done: 'I fix the immediate Albumentations v2 runtime error by replacing the removed `A.Cutout` with the supported equivalent (`A.CoarseDropout`) while keeping the same augmentation intent. I also make checkpoint loading robust to the common Kaggle layout by checking both the competition root and the `cassava-leaf-disease-classification/` subfolder, so your trained `.pth` files (if present) actually get picked up—this should move the accuracy score sharply upward toward your target. Finally, I correct the focal-loss helper’s `p_t` computation (it was mathematically wrong) without changing the rest of your pipeline; this is score-neutral here (loss isn’t used) but prevents future silent logic issues. The rest of the model/inference/ensemble logic and output submission format are preserved.'
- What this solution (achieved 0.10987) has done: 'Your current score (~0.11) strongly suggests the intended trained checkpoints are still not being loaded, so the models are effectively random at inference. I make a minimal, score-relevant change: expand checkpoint discovery to also search typical Kaggle “working” and repository subfolders, and print an explicit warning if either checkpoint is missing so you can immediately see why the score is low. I also ensure we run inference on the *same* model object we loaded weights into (by moving `.to(device)` before loading) to avoid any subtle state mismatches with DataParallel wrapping. Core model architectures, TTA, normalization/ensembling, and submission semantics remain unchanged.'
- What this solution (achieved 0.10949) has done: 'Your very low accuracy strongly indicates the checkpoints still aren’t being found/loaded at inference, so the smallest meaningful improvement is to make checkpoint discovery actually locate the `.pth` files wherever Kaggle placed them. I expand `_resolve_ckpt_path` to (a) search recursively under `../input` and `../working` for the exact filenames and (b) accept the case where the checkpoints live inside a dataset subfolder, while keeping the same model definitions and prediction/ensemble logic. I also make the loader handle the common “nested key” cases (`model`, `net`, etc.) so weights load even if the file isn’t a raw state_dict. These changes are directly score-relevant (they turn random inference into trained inference) and preserve your pipeline semantics; everything else stays the same and a valid `submission.csv` is still written.'
- What this solution (achieved 0.11024) has done: 'Your score (~0.11) is consistent with near-random predictions, which almost always happens when the trained checkpoints still aren’t actually being loaded into the exact inference modules. I make a minimal, score-relevant fix to checkpoint loading for `DataParallel`: load weights into the underlying `.module` (when wrapped) and strip both `module.` and `_orig_mod.` prefixes, which are common in newer PyTorch saves. I also make checkpoint discovery slightly more permissive by accepting `.pth/.pt/.bin` variants when the exact filename isn’t found, without changing your model architectures, TTA, normalization, or ensembling. These changes should move accuracy sharply upward toward your target if the correct weights exist anywhere under the Kaggle input/working trees.'
- What this solution (achieved 0.11136) has done: 'Your low score (~0.11) still looks like random guessing, so the most score-relevant minimal change is to ensure the test-time preprocessing matches what the checkpoints were trained with: remove stochastic augmentations from inference and use a deterministic resize/center-crop + normalize pipeline, while keeping your ensemble logic and model definitions intact. I also make checkpoint loading a bit more robust by mapping any `fc.*` keys to `classifier.*` (and vice versa) when the head naming differs between timm versions, which can silently prevent loading the trained head weights and crater accuracy. Finally, I keep your submission writing unchanged but add a strict check that all test images were found so you don’t accidentally submit an empty/misaligned file.'
- What this solution (achieved 0.11136) has done: 'I fix the immediate runtime failure by removing the hard stop when checkpoints aren’t found and instead generate a valid submission using the (deterministic) untrained ensemble as a fallback, so you always get a `submission.csv`. I also make checkpoint discovery actually look for these `.pth` files anywhere under the Kaggle `/kaggle/input` tree (including other attached datasets) and accept common filename variants, which is the smallest score-relevant change because the current low score is consistent with “weights not loaded”. Finally, I keep the same model definitions, preprocessing, TTA, normalization, and ensembling logic, but add clear prints that confirm whether weights truly loaded (so you can verify why the score is low/high).'
- What this solution (achieved 0.11136) has done: 'Your current score (~0.11) is effectively random for 5 classes, which almost always means the trained checkpoints still aren’t being loaded (or they load but don’t match the model keys). I make a minimal, score-relevant improvement to checkpoint handling by (1) searching recursively specifically for your checkpoint filenames under `/kaggle/input` and `/kaggle/working`, and (2) making the loader robust to more timm/PyTorch key mismatches (e.g., `head.*` vs `classifier.*`, and `fc.*` vs `head.*`) while still using `strict=False`. I also add a hard “loaded_effectively” verification based on the number of matched keys (not just a parameter fingerprint), so we don’t silently proceed with random weights. Core model definitions, inference flow, normalization, ensembling, and submission format remain unchanged; this only aims to ensure the intended trained weights actually get applied so accuracy moves toward your target.'
- What this solution (achieved 0.11136) has done: 'Your score (~0.11) is consistent with random guessing, so the most direct way to move toward the target is to ensure trained weights actually load and are compatible with the current timm model head names. I make checkpoint resolution more likely to find the files by also searching the competition dataset folder you actually have (`/kaggle/input/cassava-leaf-disease-classification/`) and by allowing “best/last/epoch*.pth” style matches when the exact filename isn’t present. I also strengthen state_dict key remapping beyond just `fc/classifier/head` to include common timm variants like `head.fc` and `classifier.fc`, and add a stricter “effective load” check that requires the classifier/head weights to be present (otherwise accuracy stays near-random). Core models, augmentations, inference loop, ensembling, and submission format remain the same.'
- What this solution (achieved 0.25635) has done: 'Your current ~0.11 accuracy is near-random for 5 classes, so the smallest score-relevant change is to stop silently running with random weights and instead load a known-good pretrained backbone when your `.pth` checkpoints are missing/ineffective. This keeps the same model architectures and inference/ensembling logic, but switches `pretrained=True` only as a fallback when checkpoint loading fails, which should move accuracy substantially toward your target. I also make the “effective load” check stricter by ensuring head weights are actually present and by reporting missing/unexpected keys counts, so you can confirm weights truly applied. Submission writing, paths, preprocessing, and the ensemble computation remain the same.'

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
import timm
from tqdm import tqdm



## === cell 1
INPUT_PATH = "../input/cassava-leaf-disease-classification/"
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
TTA = 10

device = DEVICES[0] if len(DEVICES) else torch.device("cpu")




## === cell 2
def sigmoid_focal_cross_entropy(y_hat, y_true, alpha=0.25, gamma=2.0):
    """
    Note: This function is not used in the current inference-only script,
    but keep it correct to avoid silent bugs if training is added back.
    """

    def smooth(y, smooth_factor):
        assert len(y.shape) == 2
        y = y * (1 - smooth_factor) + smooth_factor / y.shape[1]
        return y

    smooth_factor = 0.1

    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)
    if not isinstance(y_hat, torch.Tensor):
        y_hat = torch.tensor(y_hat)

    y_true = smooth(y_true, smooth_factor).to(dtype=y_hat.dtype, device=y_hat.device)

    bce = F.binary_cross_entropy_with_logits(y_hat, y_true, reduction="none")
    p = torch.sigmoid(y_hat)
    p_t = y_true * p + (1 - y_true) * (1 - p)
    alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
    modulating_factor = (1.0 - p_t).pow(gamma)

    return torch.sum(alpha_t * modulating_factor * bce, dim=-1)




## === cell 3
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
plt.title("LR schedule")
plt.show()



## === cell 4
train_augs = A.Compose(
    [
        A.RandomResizedCrop(size=(IMAGE_SIZE, IMAGE_SIZE), scale=(0.08, 1.0)),
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
        A.CoarseDropout(p=0.5),  # replacement for removed Cutout
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



## === cell 5
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




## === cell 6
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



## === cell 7
model_name1 = "resnext50_32x4d"
my_model_1 = timm.create_model(model_name1, pretrained=False)
my_model_1.fc = nn.Linear(my_model_1.fc.in_features, OUT_FEATURES)
nn.init.xavier_uniform_(my_model_1.fc.weight)
if my_model_1.fc.bias is not None:
    nn.init.zeros_(my_model_1.fc.bias)
my_model_1



## === cell 8
model_name2 = "tf_efficientnet_b4_ns"
my_model_2 = timm.create_model(model_name2, pretrained=False)
my_model_2.classifier = nn.Linear(my_model_2.classifier.in_features, OUT_FEATURES)
nn.init.xavier_uniform_(my_model_2.classifier.weight)
if my_model_2.classifier.bias is not None:
    nn.init.zeros_(my_model_2.classifier.bias)
my_model_2



## === cell 9
torch.cuda.empty_cache()




## === cell 10
def _extract_state_dict(ckpt_obj):
    """
    Score-relevant robustness:
    Kaggle checkpoints are often wrapped (state_dict/model/net/ema/student/...).
    Extracting correctly is essential; otherwise inference stays near-random.
    """
    if isinstance(ckpt_obj, dict) and len(ckpt_obj) > 0:
        for k in (
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "network",
            "module",
            "ema",
            "ema_state_dict",
            "model_ema",
            "student",
            "student_state_dict",
        ):
            v = ckpt_obj.get(k, None)
            if isinstance(v, dict) and len(v) > 0:
                nested = v.get("state_dict", None) if isinstance(v, dict) else None
                if isinstance(nested, dict) and len(nested) > 0:
                    return nested
                return v

        if all(isinstance(k, str) for k in ckpt_obj.keys()) and any(
            (".weight" in k) or (".bias" in k) for k in ckpt_obj.keys()
        ):
            return ckpt_obj

    return None


def _strip_common_prefixes(state: dict) -> dict:
    """
    Score-relevant fix: handle DataParallel/compile prefixes so keys match model.
    """
    if not isinstance(state, dict) or len(state) == 0:
        return state

    def strip_prefix(d, prefix):
        if any(k.startswith(prefix) for k in d.keys()):
            return {
                (k[len(prefix) :] if k.startswith(prefix) else k): v
                for k, v in d.items()
            }
        return d

    state = strip_prefix(state, "module._orig_mod.")
    state = strip_prefix(state, "_orig_mod.")
    state = strip_prefix(state, "module.")
    return state


def _maybe_remap_head_keys(state: dict, model: nn.Module) -> dict:
    """
    Score-relevant fix:
    timm head naming differs across families/versions (fc/classifier/head and nested head.fc etc.).
    If the head keys don't map, the classifier stays randomly initialized -> ~0.11 acc.
    """
    if not isinstance(state, dict) or len(state) == 0:
        return state

    target = model.module if isinstance(model, nn.DataParallel) else model
    model_keys = set(target.state_dict().keys())

    head_prefixes = ["fc.", "classifier.", "head.", "head.fc.", "classifier.fc."]

    def has_prefix(keys, pref):
        return any(k.startswith(pref) for k in keys)

    model_heads = [p for p in head_prefixes if has_prefix(model_keys, p)]
    state_heads = [p for p in head_prefixes if has_prefix(state.keys(), p)]

    if not model_heads or not state_heads:
        return state

    model_head = sorted(model_heads, key=len)[0]
    state_head = sorted(state_heads, key=len)[0]

    if model_head == state_head:
        return state

    remapped = dict(state)
    for k, v in list(state.items()):
        if k.startswith(state_head):
            remapped[model_head + k[len(state_head) :]] = v
    return remapped


def _head_key_present(state_keys: set, possible_prefixes: list[str]) -> bool:
    for p in possible_prefixes:
        if any(k.startswith(p) for k in state_keys):
            return True
    return False


def _load_state_dict_if_exists(model: nn.Module, ckpt_path: str) -> bool:
    """
    Tries to load a checkpoint if present. Returns True if loaded and *effective*.

    Score-relevant change (still minimal): print missing/unexpected counts so it's obvious
    whether a "load" actually matched keys; require head weights.
    """
    if not os.path.exists(ckpt_path):
        return False

    ckpt_obj = torch.load(ckpt_path, map_location="cpu")
    state = _extract_state_dict(ckpt_obj)
    if state is None:
        return False

    state = _strip_common_prefixes(state)
    state = _maybe_remap_head_keys(state, model)

    target = model.module if isinstance(model, nn.DataParallel) else model
    model_state = target.state_dict()
    model_keys = set(model_state.keys())
    state_keys = set(state.keys())

    matched = len(model_keys & state_keys)

    model_head_prefixes = ["fc.", "classifier.", "head.", "head.fc.", "classifier.fc."]
    model_has_head = any(
        any(k.startswith(p) for k in model_keys) for p in model_head_prefixes
    )
    head_ok = (
        _head_key_present(state_keys, model_head_prefixes) if model_has_head else True
    )

    incompatible = target.load_state_dict(state, strict=False)
    try:
        n_missing = len(incompatible.missing_keys)
        n_unexpected = len(incompatible.unexpected_keys)
    except Exception:
        n_missing, n_unexpected = -1, -1

    print(
        f"Loaded ckpt={ckpt_path} with strict=False: matched_keys={matched}/{len(model_keys)}, "
        f"missing={n_missing}, unexpected={n_unexpected}, head_present={head_ok}"
    )

    if (matched < max(50, int(0.20 * len(model_keys)))) or (not head_ok):
        return False
    return True


def _resolve_ckpt_path(input_path: str, ckpt_name: str) -> str:
    candidates = [
        os.path.join(input_path, ckpt_name),
        os.path.join(input_path, "cassava-leaf-disease-classification", ckpt_name),
        os.path.join("../input", ckpt_name),
        os.path.join("../input", "cassava-leaf-disease-classification", ckpt_name),
        os.path.join("/kaggle/input", ckpt_name),
        os.path.join("/kaggle/input/cassava-leaf-disease-classification", ckpt_name),
        os.path.join(
            "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
            ckpt_name,
        ),
        os.path.join("../working", ckpt_name),
        os.path.join("/kaggle/working", ckpt_name),
        os.path.join("./", ckpt_name),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    search_roots = [
        Path("/kaggle/input"),
        Path("/kaggle/working"),
        Path("../input"),
        Path("../working"),
        Path(input_path),
        Path("."),
    ]

    for root in search_roots:
        if not root.exists() or not root.is_dir():
            continue
        try:
            hits = list(root.rglob(ckpt_name))
            if hits:
                hits = sorted(hits, key=lambda p: len(str(p)))
                return str(hits[0])
        except Exception:
            pass

    stem = Path(ckpt_name).stem
    for ext in (".pth", ".pt", ".bin"):
        nv = stem + ext
        for root in search_roots:
            if not root.exists() or not root.is_dir():
                continue
            try:
                hits = list(root.rglob(nv))
                if hits:
                    hits = sorted(hits, key=lambda p: len(str(p)))
                    return str(hits[0])
            except Exception:
                pass

    patterns = [
        f"*{stem}*.pth",
        f"*{stem}*.pt",
        "*best*.pth",
        "*last*.pth",
        "*final*.pth",
        "*epoch*.pth",
    ]
    for root in search_roots:
        if not root.exists() or not root.is_dir():
            continue
        try:
            for pat in patterns:
                hits = list(root.rglob(pat))
                if hits:
                    hits = sorted(
                        hits,
                        key=lambda p: (
                            len(str(p)),
                            -(p.stat().st_size if p.exists() else 0),
                        ),
                    )
                    return str(hits[0])
        except Exception:
            pass

    return candidates[0]


test_image_list = np.asarray(
    sorted(
        [
            image_name
            for image_name in os.listdir(TEST_IMAGE_PATH)
            if image_name.lower().endswith(".jpg")
        ]
    )
)

if len(test_image_list) == 0:
    raise RuntimeError(f"No .jpg files found under TEST_IMAGE_PATH={TEST_IMAGE_PATH}")

my_model_1 = (
    nn.DataParallel(my_model_1).to(device)
    if torch.cuda.is_available() and len(DEVICES) > 1
    else my_model_1.to(device)
)
my_model_2 = (
    nn.DataParallel(my_model_2).to(device)
    if torch.cuda.is_available() and len(DEVICES) > 1
    else my_model_2.to(device)
)

ckpt1 = _resolve_ckpt_path(INPUT_PATH, RESNEXT_PATH)
ckpt2 = _resolve_ckpt_path(INPUT_PATH, B4_PATH)

loaded1 = _load_state_dict_if_exists(my_model_1, ckpt1)
loaded2 = _load_state_dict_if_exists(my_model_2, ckpt2)

if not loaded1:
    print(
        "Checkpoint for model1 not loaded effectively -> using pretrained backbone fallback."
    )
    target1 = (
        my_model_1.module if isinstance(my_model_1, nn.DataParallel) else my_model_1
    )
    target1_pre = timm.create_model(model_name1, pretrained=True)
    target1_pre.fc = nn.Linear(target1_pre.fc.in_features, OUT_FEATURES)
    target1.load_state_dict(target1_pre.state_dict(), strict=False)

if not loaded2:
    print(
        "Checkpoint for model2 not loaded effectively -> using pretrained backbone fallback."
    )
    target2 = (
        my_model_2.module if isinstance(my_model_2, nn.DataParallel) else my_model_2
    )
    target2_pre = timm.create_model(model_name2, pretrained=True)
    target2_pre.classifier = nn.Linear(target2_pre.classifier.in_features, OUT_FEATURES)
    target2.load_state_dict(target2_pre.state_dict(), strict=False)

if (not loaded1) and (not loaded2):
    print(
        "WARNING: Neither checkpoint could be loaded effectively.\n"
        "  - Pretrained-backbone fallbacks were used, so accuracy should be far above random.\n"
        f"  - Tried ckpt1={ckpt1}\n"
        f"  - Tried ckpt2={ckpt2}\n"
        "If you intended to use trained weights, add/attach the dataset containing these .pth files."
    )
elif (not loaded1) or (not loaded2):
    print(
        "WARNING: One checkpoint was not loaded effectively; that model used pretrained fallback."
    )
    print(f"  model1 loaded={loaded1}, path tried={ckpt1}")
    print(f"  model2 loaded={loaded2}, path tried={ckpt2}")
else:
    print("Both checkpoints loaded effectively.")
    print(f"  model1: {ckpt1}")
    print(f"  model2: {ckpt2}")

preds_1 = []
my_model_1.eval()

for single_image_name in tqdm(
    test_image_list,
    desc=f"Infer model1 ({'ckpt' if loaded1 else 'pretrained-fallback'})",
):
    with torch.no_grad():
        ans = torch.zeros(OUT_FEATURES, device=device)
        for _ in range(1):
            image = Image.open(
                os.path.join(TEST_IMAGE_PATH, single_image_name)
            ).convert("RGB")
            aug_image = test_augs(image=np.array(image))["image"]  # torch.Tensor CxHxW
            test_image = aug_image.unsqueeze(0).to(device=device, dtype=torch.float32)
            ans += my_model_1(test_image).view(ans.shape)
        preds_1.append(ans.detach().cpu())

predictions_1 = torch.stack(preds_1, dim=0)
normalize_pred_1 = F.normalize(predictions_1.T, p=2, dim=0).T
torch.cuda.empty_cache()

preds_2 = []
my_model_2.eval()

for single_image_name in tqdm(
    test_image_list,
    desc=f"Infer model2 ({'ckpt' if loaded2 else 'pretrained-fallback'})",
):
    with torch.no_grad():
        ans = torch.zeros(OUT_FEATURES, device=device)
        for _ in range(TTA):
            image = Image.open(
                os.path.join(TEST_IMAGE_PATH, single_image_name)
            ).convert("RGB")
            aug_image = test_augs(image=np.array(image))["image"]
            test_image = aug_image.unsqueeze(0).to(device=device, dtype=torch.float32)
            ans += my_model_2(test_image).view(ans.shape)
        ans /= TTA
        preds_2.append(ans.detach().cpu())

predictions_2 = torch.stack(preds_2, dim=0)
normalize_pred_2 = F.normalize(predictions_2.T, p=2, dim=0).T

final_pred = (normalize_pred_1 * 0.45) + (normalize_pred_2 * 0.55)
label = final_pred.argmax(dim=-1).detach().cpu().numpy().astype(int).tolist()

df_submission = pd.DataFrame({"image_id": test_image_list.tolist(), "label": label})
df_submission.to_csv(SUBMISSION_PATH, index=False)

print(f"Wrote submission to: {SUBMISSION_PATH}")
print(df_submission.head())
print(f"Checkpoints loaded: model1={loaded1} ({ckpt1}), model2={loaded2} ({ckpt2})")
print(f"Submission rows: {len(df_submission)} (expected {len(test_image_list)})")
if len(df_submission) != len(test_image_list):
    raise RuntimeError("Submission row count does not match number of test images.")
