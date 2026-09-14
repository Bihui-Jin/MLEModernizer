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

0.51794

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.11248) has done: 'I fix the Albumentations v2 API break by replacing the removed `A.Cutout` with `A.CoarseDropout`, which preserves the augmentation intent and unblocks execution. I also make the weight-loading logic robust: if the expected `.pth` files are not present in `/kaggle/input`, the script fall back to running the untrained models (still producing a valid submission CSV instead of crashing). Finally, I fix a small ordering bug by setting models to `.eval()` after moving them into `DataParallel`/device so inference is deterministic and correct. These changes are minimal, preserve the ensemble/inference core logic, and guarantee `submission.csv` is always written.'
- What this solution (achieved 0.11211) has done: 'Your current low score is consistent with the warning path where the expected `.pth` weights are not found, so the submission is produced from randomly initialized models. The smallest change that should move accuracy strongly toward your target (without changing the ensemble/inference logic) is to point `INPUT_PATH` at the actual Kaggle dataset directory you have available and make weight lookup prefer the competition dataset folder first. I keep the same models, TTA, normalization, and ensembling, and only adjust the weight path resolution to reliably load the trained checkpoints when they exist. This should increase score substantially toward the target band because it restores the intended trained ensemble behavior.'
- What this solution (achieved 0.11248) has done: 'Your current score is consistent with the “weights not found → random model” path, so the smallest change to move accuracy toward your 0.898 target is to reliably locate and load the intended `.pth` checkpoints. I keep the exact same models, TTA, normalization, and ensembling, but expand the weight search to also check `/kaggle/working` and the competition dataset subfolder, and add clear failure diagnostics (listing candidate `.pth` files) so we don’t silently submit random predictions. I also ensure the state-dict loader tolerates a common “model” wrapper key in saved checkpoints while keeping `strict=True` for actual parameter matching. These are minimal changes that preserve core inference semantics but should restore the trained ensemble behavior and substantially increase accuracy.'
- What this solution (achieved 0.11248) has done: 'Your score is extremely low because the script is almost certainly running in the “weights not found → random model” fallback, so the smallest change that moves accuracy toward your 0.8983 target is to reliably locate and load the intended `.pth` checkpoints. I keep the exact same models, TTA, normalization, and ensembling, but expand the weight search to also scan the current working directory tree (where Kaggle notebooks commonly place uploaded/created checkpoints) and fail loudly (instead of silently) if the `.pth` files still aren’t found. I also add a safe load that accepts common checkpoint wrappers (`state_dict`/`model`) while keeping parameter matching strict, so the trained weights actually get applied. These changes preserve core inference semantics and should move accuracy dramatically upward toward the target band.'
- What this solution (achieved 0.58782) has done: 'I fix the Albumentations v2 API break that currently prevents `train_augs`/`test_augs` from being created (the new API requires `size=(h,w)` instead of `height=`/`width=` for `RandomResizedCrop`). This also eliminate the downstream `NameError: test_augs is not defined` during inference, letting the notebook run end-to-end and always write `submission.csv`. I keep the exact model/ensemble/TTA logic unchanged and only adjust augmentation constructor arguments to their v2 equivalents. I also add a tiny safety check for `TEST_IMAGE_PATH` existence to fail early with a clear message if paths are wrong (score-neutral).'
- What this solution (achieved 0.5157) has done: 'Your current score (0.58782) is far below the target (0.8983), so we should improve accuracy with minimal, score-relevant fixes while keeping your ensemble, TTA, and inference flow intact. The largest issue hurting accuracy is that `F.normalize()` is being applied to raw logits before ensembling, which discards calibration/magnitude information and often degrades argmax accuracy; we remove that and ensemble logits directly (core logic unchanged: still same two models + weights + TTA + weighted sum + argmax). We also fix the (incorrect) focal-loss helper’s `p_t` definition to be mathematically correct (score-neutral here since it isn’t used, but it prevents future confusion and keeps semantics consistent). Finally, we speed and stabilize inference slightly (no approximation) by reusing the opened image per TTA loop and enabling cuDNN benchmark when not strictly deterministic.'
- What this solution (achieved 0.51794) has done: 'Your score gap is large (0.5157 vs target 0.8983), and the biggest likely cause in your current inference-only notebook is that the classification heads are randomly initialized when the `.pth` weights aren’t actually found/loaded, so accuracy collapses. I make the smallest score-relevant change: fail fast if the expected weight files are missing (instead of silently proceeding), and broaden the search to also pick up any `.pth` inside the competition directory tree (common Kaggle layout) while keeping strict state-dict loading. I also fix one inference-only detail that can improve accuracy without changing the core model/ensemble logic: apply TTA consistently to both models (resnext currently uses only 1 pass while effnet uses `TTA`). These changes preserve your architecture, augmentations, ensembling, and argmax semantics, but should move accuracy substantially toward your target when the checkpoints exist.'

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
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")

BASE_COMP_DIR = KAGGLE_INPUT_ROOT / "cassava-leaf-disease-classification"
if not BASE_COMP_DIR.exists():
    BASE_COMP_DIR = Path("../input/cassava-leaf-disease-classification")

TRAIN_CSV_PATH = str(BASE_COMP_DIR / "train.csv")
TRAIN_IMAGE_PATH = str(BASE_COMP_DIR / "train_images")
TEST_IMAGE_PATH = str(BASE_COMP_DIR / "test_images")

SUBMISSION_PATH = "submission.csv"

INPUT_PATH = str(BASE_COMP_DIR)

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
TTA = 8




## === cell 2
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

    y_true = smooth(y_true, smooth_factor)

    p = torch.sigmoid(y_hat)
    p_t = y_true * p + (1 - y_true) * (1 - p)

    cross_entropy = F.binary_cross_entropy_with_logits(y_hat, y_true, reduction="none")
    alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
    modulating_factor = (1.0 - p_t).pow(gamma)

    return torch.sum(alpha_t * modulating_factor * cross_entropy, dim=-1)




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



## === cell 4
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
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.CenterCrop(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)



## === cell 5
test_augs = A.Compose(
    [
        A.OneOf(
            [
                A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE, p=1.0),
                A.CenterCrop(height=IMAGE_SIZE, width=IMAGE_SIZE, p=1.0),
                A.RandomResizedCrop(size=(IMAGE_SIZE, IMAGE_SIZE), p=1.0),
            ],
            p=1.0,
        ),
        A.Transpose(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
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
USE_IMAGENET_PRETRAINED_FALLBACK = True



## === cell 8
model_name1 = "resnext50_32x4d"
my_model_1 = timm.create_model(model_name1, pretrained=USE_IMAGENET_PRETRAINED_FALLBACK)
my_model_1.fc = nn.Linear(my_model_1.fc.in_features, OUT_FEATURES)
nn.init.xavier_uniform_(my_model_1.fc.weight)
if my_model_1.fc.bias is not None:
    nn.init.zeros_(my_model_1.fc.bias)
my_model_1



## === cell 9
model_name2 = "tf_efficientnet_b4_ns"
my_model_2 = timm.create_model(model_name2, pretrained=USE_IMAGENET_PRETRAINED_FALLBACK)
my_model_2.classifier = nn.Linear(my_model_2.classifier.in_features, OUT_FEATURES)
nn.init.xavier_uniform_(my_model_2.classifier.weight)
if my_model_2.classifier.bias is not None:
    nn.init.zeros_(my_model_2.classifier.bias)
my_model_2



## === cell 10
torch.cuda.empty_cache()




## === cell 11
def _candidate_weight_roots(preferred_dir: str | None):
    roots = []
    if preferred_dir:
        roots.append(Path(preferred_dir))

    roots.append(BASE_COMP_DIR)
    roots.append(BASE_COMP_DIR / "cassava-leaf-disease-classification")

    roots.append(KAGGLE_INPUT_ROOT)
    roots.append(KAGGLE_WORKING_ROOT)
    roots.append(KAGGLE_WORKING_ROOT / "cassava-leaf-disease-classification")

    roots.append(Path("../input/cassava-leaf-disease-classification"))

    roots.append(Path("."))
    roots.append(Path(".."))

    seen = set()
    out = []
    for r in roots:
        r2 = r.resolve() if r.exists() else r
        if str(r2) not in seen:
            seen.add(str(r2))
            out.append(r2)
    return out


def _find_weight_file(preferred_dir: str | None, filename: str) -> str | None:
    for root in _candidate_weight_roots(preferred_dir):
        p = root / filename
        if p.exists():
            return str(p)

    for root in _candidate_weight_roots(preferred_dir):
        if root.exists():
            hits = list(root.rglob(filename))
            if len(hits) > 0:
                hits = sorted(hits, key=lambda x: str(x))
                return str(hits[0])

    return None


RESNEXT_WEIGHT = _find_weight_file(INPUT_PATH, RESNEXT_PATH)
B4_WEIGHT = _find_weight_file(INPUT_PATH, B4_PATH)

print("Weight lookup results:")
print(" RESNEXT_WEIGHT:", RESNEXT_WEIGHT)
print(" B4_WEIGHT:", B4_WEIGHT)

WEIGHTS_AVAILABLE = (RESNEXT_WEIGHT is not None) and (B4_WEIGHT is not None)

if not WEIGHTS_AVAILABLE:
    pth_candidates = []
    for root in _candidate_weight_roots(INPUT_PATH):
        if root.exists():
            pth_candidates.extend([str(p) for p in root.rglob("*.pth")])
    pth_candidates = sorted(set(pth_candidates))
    msg = (
        "Required ensemble weight files were not found, so predictions would be made with "
        "randomly initialized classification heads (very low accuracy).\n"
        f"Missing: {RESNEXT_PATH} and/or {B4_PATH}\n"
        "Searched roots:\n  - "
        + "\n  - ".join([str(r) for r in _candidate_weight_roots(INPUT_PATH)])
        + "\n"
        f"Found {len(pth_candidates)} *.pth files (showing up to 50):\n  - "
        + "\n  - ".join(pth_candidates[:50])
    )
    raise FileNotFoundError(msg)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/2965247867.py in <cell line: 0>()
     75         + "\n  - ".join(pth_candidates[:50])
     76     )
---> 77     raise FileNotFoundError(msg)
     78 

FileNotFoundError: Required ensemble weight files were not found, so predictions would be made with randomly initialized classification heads (very low accuracy).
Missing: 1022_res50.pth and/or 1022_b4ns.pth
Searched roots:
  - /kaggle/input/cassava-leaf-disease-classification
  - /kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification
  - /kaggle/input
  - /kaggle/working
  - /kaggle/working/cassava-leaf-disease-classification
  - /kaggle
Found 0 *.pth files (showing up to 50):
  - 

## === cell 12
if not Path(TEST_IMAGE_PATH).exists():
    raise FileNotFoundError(f"TEST_IMAGE_PATH does not exist: {TEST_IMAGE_PATH}")

device = DEVICES[0] if len(DEVICES) > 0 else torch.device("cpu")


def _load_dp_state_dict(model: nn.Module, weight_path: str | None):
    if weight_path is None:
        return False

    sd = torch.load(weight_path, map_location="cpu")

    if (
        isinstance(sd, dict)
        and "state_dict" in sd
        and isinstance(sd["state_dict"], dict)
    ):
        sd = sd["state_dict"]
    elif isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]

    if isinstance(sd, dict) and any(k.startswith("module.") for k in sd.keys()):
        sd = {k[7:]: v for k, v in sd.items() if k.startswith("module.")}

    model.load_state_dict(sd, strict=True)
    return True


test_image_list = sorted(
    [
        image_name
        for image_name in os.listdir(TEST_IMAGE_PATH)
        if image_name.lower().endswith(".jpg")
    ]
)

preds_1 = []
loaded1 = _load_dp_state_dict(my_model_1, RESNEXT_WEIGHT)
print("Loaded resnext weights:", loaded1)

my_model_1 = nn.DataParallel(my_model_1).to(device)
my_model_1.eval()

for single_image_name in tqdm(test_image_list, desc="Predict resnext"):
    with torch.no_grad():
        ans = torch.zeros(OUT_FEATURES, device=device)
        image = Image.open(os.path.join(TEST_IMAGE_PATH, single_image_name)).convert(
            "RGB"
        )
        image_np = np.array(image)
        for _ in range(TTA):
            aug_image = test_augs(image=image_np)["image"]
            test_image = aug_image.float().unsqueeze(0).to(device)
            ans += my_model_1(test_image).view(ans.shape)
        ans /= TTA
        preds_1.append(ans.detach().to("cpu"))

predictions_1 = torch.stack(preds_1, dim=0)
torch.cuda.empty_cache()

preds_2 = []
loaded2 = _load_dp_state_dict(my_model_2, B4_WEIGHT)
print("Loaded effnet_b4_ns weights:", loaded2)

my_model_2 = nn.DataParallel(my_model_2).to(device)
my_model_2.eval()

for single_image_name in tqdm(test_image_list, desc="Predict effnet_b4_ns"):
    with torch.no_grad():
        ans = torch.zeros(OUT_FEATURES, device=device)
        image = Image.open(os.path.join(TEST_IMAGE_PATH, single_image_name)).convert(
            "RGB"
        )
        image_np = np.array(image)
        for _ in range(TTA):
            aug_image = test_augs(image=image_np)["image"]
            test_image = aug_image.float().unsqueeze(0).to(device)
            ans += my_model_2(test_image).view(ans.shape)
        ans /= TTA
        preds_2.append(ans.detach().to("cpu"))

predictions_2 = torch.stack(preds_2, dim=0)

final_pred = (predictions_1 * 0.43) + (predictions_2 * 0.57)
label = final_pred.argmax(dim=-1).numpy().astype(int).tolist()

df_submission = pd.DataFrame({"image_id": test_image_list, "label": label})

sample_path = BASE_COMP_DIR / "sample_submission.csv"
if sample_path.exists():
    sample_df = pd.read_csv(sample_path)
    df_submission = sample_df[["image_id"]].merge(
        df_submission, on="image_id", how="left"
    )
    df_submission["label"] = df_submission["label"].fillna(0).astype(int)

df_submission.to_csv(SUBMISSION_PATH, index=False)
print("Wrote:", SUBMISSION_PATH, "rows:", len(df_submission))
print(df_submission.head())
