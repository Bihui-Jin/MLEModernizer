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

0.61099

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11248) has done: 'I fix the Albumentations v2 API break by replacing the removed `A.Cutout` with `A.CoarseDropout`, which preserves the augmentation intent and unblocks execution. I also make the weight-loading logic robust: if the expected `.pth` files are not present in `/kaggle/input`, the script fall back to running the untrained models (still producing a valid submission CSV instead of crashing). Finally, I fix a small ordering bug by setting models to `.eval()` after moving them into `DataParallel`/device so inference is deterministic and correct. These changes are minimal, preserve the ensemble/inference core logic, and guarantee `submission.csv` is always written.'
- What this solution (achieved 0.11211) has done: 'Your current low score is consistent with the warning path where the expected `.pth` weights are not found, so the submission is produced from randomly initialized models. The smallest change that should move accuracy strongly toward your target (without changing the ensemble/inference logic) is to point `INPUT_PATH` at the actual Kaggle dataset directory you have available and make weight lookup prefer the competition dataset folder first. I keep the same models, TTA, normalization, and ensembling, and only adjust the weight path resolution to reliably load the trained checkpoints when they exist. This should increase score substantially toward the target band because it restores the intended trained ensemble behavior.'
- What this solution (achieved 0.11248) has done: 'Your current score is consistent with the “weights not found → random model” path, so the smallest change to move accuracy toward your 0.898 target is to reliably locate and load the intended `.pth` checkpoints. I keep the exact same models, TTA, normalization, and ensembling, but expand the weight search to also check `/kaggle/working` and the competition dataset subfolder, and add clear failure diagnostics (listing candidate `.pth` files) so we don’t silently submit random predictions. I also ensure the state-dict loader tolerates a common “model” wrapper key in saved checkpoints while keeping `strict=True` for actual parameter matching. These are minimal changes that preserve core inference semantics but should restore the trained ensemble behavior and substantially increase accuracy.'
- What this solution (achieved 0.11248) has done: 'Your score is extremely low because the script is almost certainly running in the “weights not found → random model” fallback, so the smallest change that moves accuracy toward your 0.8983 target is to reliably locate and load the intended `.pth` checkpoints. I keep the exact same models, TTA, normalization, and ensembling, but expand the weight search to also scan the current working directory tree (where Kaggle notebooks commonly place uploaded/created checkpoints) and fail loudly (instead of silently) if the `.pth` files still aren’t found. I also add a safe load that accepts common checkpoint wrappers (`state_dict`/`model`) while keeping parameter matching strict, so the trained weights actually get applied. These changes preserve core inference semantics and should move accuracy dramatically upward toward the target band.'
- What this solution (achieved 0.58782) has done: 'I fix the Albumentations v2 API break that currently prevents `train_augs`/`test_augs` from being created (the new API requires `size=(h,w)` instead of `height=`/`width=` for `RandomResizedCrop`). This also eliminate the downstream `NameError: test_augs is not defined` during inference, letting the notebook run end-to-end and always write `submission.csv`. I keep the exact model/ensemble/TTA logic unchanged and only adjust augmentation constructor arguments to their v2 equivalents. I also add a tiny safety check for `TEST_IMAGE_PATH` existence to fail early with a clear message if paths are wrong (score-neutral).'
- What this solution (achieved 0.5157) has done: 'Your current score (0.58782) is far below the target (0.8983), so we should improve accuracy with minimal, score-relevant fixes while keeping your ensemble, TTA, and inference flow intact. The largest issue hurting accuracy is that `F.normalize()` is being applied to raw logits before ensembling, which discards calibration/magnitude information and often degrades argmax accuracy; we remove that and ensemble logits directly (core logic unchanged: still same two models + weights + TTA + weighted sum + argmax). We also fix the (incorrect) focal-loss helper’s `p_t` definition to be mathematically correct (score-neutral here since it isn’t used, but it prevents future confusion and keeps semantics consistent). Finally, we speed and stabilize inference slightly (no approximation) by reusing the opened image per TTA loop and enabling cuDNN benchmark when not strictly deterministic.'
- What this solution (achieved 0.51794) has done: 'Your score gap is large (0.5157 vs target 0.8983), and the biggest likely cause in your current inference-only notebook is that the classification heads are randomly initialized when the `.pth` weights aren’t actually found/loaded, so accuracy collapses. I make the smallest score-relevant change: fail fast if the expected weight files are missing (instead of silently proceeding), and broaden the search to also pick up any `.pth` inside the competition directory tree (common Kaggle layout) while keeping strict state-dict loading. I also fix one inference-only detail that can improve accuracy without changing the core model/ensemble logic: apply TTA consistently to both models (resnext currently uses only 1 pass while effnet uses `TTA`). These changes preserve your architecture, augmentations, ensembling, and argmax semantics, but should move accuracy substantially toward your target when the checkpoints exist.'
- What this solution (achieved 0.40097) has done: 'I remove the hard failure on missing `.pth` checkpoints (cell 11) so the notebook always runs end-to-end and writes a valid `submission.csv`. To improve score toward your target without changing your ensemble/model logic, I also auto-search for alternative checkpoint extensions commonly used on Kaggle (`.pt`, `.bin`) and load them if found, while keeping strict state-dict loading and the same architectures/TTA/weighted-logit ensembling. Finally, I make the fallback (when no weights exist anywhere) as strong as possible without changing your approach by using ImageNet pretrained backbones and disabling random re-initialization of the classification heads (so it’s not catastrophically random). These changes are minimal, directly address the current crash, and should move accuracy upward when checkpoints are present (or at least avoid the very-low-score random-head case when they aren’t).'
- What this solution (achieved 0.11584) has done: 'Your current score (0.40097) is far below the target (0.8983), and the most likely cause is still that the intended trained checkpoints are not being loaded (so you’re effectively submitting ImageNet-backbone + random head predictions). I make the smallest score-relevant change: when weights are missing, replace the random classification heads with a deterministic “healthy-leaf prior” head (biasing class 4), which is a much stronger fallback for Cassava than random heads and should move accuracy upward toward your target without changing the model architectures, TTA, or ensembling logic. I also fix a subtle but important state-dict loading issue: currently you load weights *before* wrapping with `DataParallel`, so checkpoints saved from a DP model (with `module.` keys) won’t load; I wrap first (when weights exist) and then load into the DP model, while still handling both DP/non-DP checkpoints. Finally, I keep submission formatting/alignment identical.'
- What this solution (achieved 0.11584) has done: 'Your score is still extremely low, which strongly suggests the intended trained checkpoints are not being loaded and you’re effectively predicting with ImageNet backbones plus a fallback head. The smallest change that moves accuracy toward your 0.8983 target is to (1) search for the actual `.pth` checkpoints by *model stem* (so if the files were renamed, we still find them), and (2) only accept a found checkpoint if its head shape matches `OUT_FEATURES=5` (to avoid accidentally loading a wrong model). If matching weights still aren’t found, we keep your deterministic prior-head fallback, but we make it slightly less extreme to avoid over-collapsing to a single class, which can tank accuracy if class-4 isn’t dominant. Core models, TTA, augmentations, ensembling, and argmax submission semantics remain unchanged.'
- What this solution (achieved 0.61099) has done: 'Your current score is far below the target, and the code strongly indicates you are still not loading the intended trained checkpoints (so you’re effectively submitting a weak fallback). The smallest score-relevant fix is to make weight discovery actually find the `.pth` files inside the Kaggle input tree by searching by *architecture name* (resnext/efficientnet) in addition to the old hardcoded filenames, while still verifying `OUT_FEATURES==5` so we don’t accidentally load the wrong head. Second, we keep your ensemble/TTA/logit-averaging logic identical, but make the “no-weights” fallback less collapse-prone by using the empirical class prior from `train.csv` (instead of forcing class 4), which should improve accuracy if weights are still missing. These changes preserve your model architectures, augmentations, TTA, and argmax submission semantics, and they directly address the likely cause of the 0.115 score.'
- What this solution (achieved 0.61099) has done: 'Your current score (0.61099) is still far from the target (0.8983), and the biggest remaining accuracy limiter (without changing your models/ensemble/TTA core logic) is that you are using heavy *random* test-time augmentation during inference (including `RandomResizedCrop` and `ShiftScaleRotate`), which often hurts top-1 accuracy for cassava unless it was trained with the exact same distribution. I keep your TTA loop and ensembling identical, but switch the *test* augmentation to a deterministic resize/centercrop + optional mild flips (still TTA, but no random crop/rotate), which typically increases accuracy substantially and should move you toward the target band. I also fix a subtle dtype issue by ensuring we feed `float32` tensors (ToTensorV2 already outputs float32, but making it explicit prevents occasional mixed-type overhead) and keep everything else (weight loading, prior fallback, logits ensembling, submission merge) unchanged. These changes are minimal, inference-only, and directly score-relevant for an accuracy metric.'
- What this solution (achieved 0.61099) has done: 'Your score gap is large (0.61099 vs target 0.8983), so we should improve accuracy with the smallest inference-only changes while keeping your ensemble, models, TTA loop, and argmax submission logic intact. The most likely remaining limiter is that your test-time augmentation is still relatively “strong” (Transpose/VerticalFlip) for cassava leaves and can hurt top-1 accuracy unless the model was trained to be invariant to those transforms; we make TTA milder and more label-preserving (NoOp + HorizontalFlip only) while keeping `TTA=8` and the same preprocessing/normalization. To avoid hurting performance when trained weights are available, we also only use the class-prior head initialization in the true no-weights fallback, leaving the trained-head behavior unchanged. Everything else (paths, checkpoint discovery/verification, ensembling weights 0.43/0.57, submission merge) stays the same and still writes `submission.csv`.'

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
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.CenterCrop(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.OneOf(
            [
                A.NoOp(p=1.0),
                A.HorizontalFlip(p=1.0),
            ],
            p=1.0,
        ),
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
my_model_1



## === cell 9
model_name2 = "tf_efficientnet_b4_ns"
my_model_2 = timm.create_model(model_name2, pretrained=USE_IMAGENET_PRETRAINED_FALLBACK)
my_model_2.classifier = nn.Linear(my_model_2.classifier.in_features, OUT_FEATURES)
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


def _extract_state_dict(maybe_ckpt):
    sd = maybe_ckpt
    if (
        isinstance(sd, dict)
        and "state_dict" in sd
        and isinstance(sd["state_dict"], dict)
    ):
        sd = sd["state_dict"]
    elif isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    return sd


def _state_dict_head_out_features(sd: dict) -> int | None:
    if not isinstance(sd, dict):
        return None

    keys_to_try = [
        "fc.weight",
        "module.fc.weight",
        "classifier.weight",
        "module.classifier.weight",
    ]
    for k in keys_to_try:
        if k in sd and isinstance(sd[k], torch.Tensor) and sd[k].ndim == 2:
            return int(sd[k].shape[0])
    return None


def _find_weight_file_multi(preferred_dir: str | None, filename: str) -> str | None:
    for root in _candidate_weight_roots(preferred_dir):
        p = root / filename
        if p.exists():
            return str(p)

    stem = Path(filename).stem
    candidate_names = [filename, f"{stem}.pth", f"{stem}.pt", f"{stem}.bin"]
    for root in _candidate_weight_roots(preferred_dir):
        for name in candidate_names:
            p = root / name
            if p.exists():
                return str(p)

    stem_globs = [f"{stem}*.pth", f"{stem}*.pt", f"{stem}*.bin"]
    for root in _candidate_weight_roots(preferred_dir):
        if root.exists():
            hits = []
            for g in stem_globs:
                hits.extend(list(root.rglob(g)))
            if hits:
                hits = sorted(set(hits), key=lambda x: str(x))
                return str(hits[0])

    for root in _candidate_weight_roots(preferred_dir):
        if root.exists():
            for name in candidate_names:
                hits = list(root.rglob(name))
                if len(hits) > 0:
                    hits = sorted(hits, key=lambda x: str(x))
                    return str(hits[0])

    return None


def _find_weight_file_multi_verified(
    preferred_dir: str | None, filename: str
) -> str | None:
    candidate = _find_weight_file_multi(preferred_dir, filename)
    if candidate is None:
        return None
    try:
        ckpt = torch.load(candidate, map_location="cpu")
        sd = _extract_state_dict(ckpt)
        out = _state_dict_head_out_features(sd)
        if out is None or out != OUT_FEATURES:
            return None
        return candidate
    except Exception:
        return None


def _find_weight_by_keywords_verified(
    preferred_dir: str | None, keywords: list[str]
) -> str | None:
    kws = [k.lower() for k in keywords if k]
    exts = (".pth", ".pt", ".bin")
    for root in _candidate_weight_roots(preferred_dir):
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in exts:
                continue
            name = p.name.lower()
            if all(k in name for k in kws):
                try:
                    ckpt = torch.load(str(p), map_location="cpu")
                    sd = _extract_state_dict(ckpt)
                    out = _state_dict_head_out_features(sd)
                    if out == OUT_FEATURES:
                        return str(p)
                except Exception:
                    continue
    return None


RESNEXT_WEIGHT = _find_weight_file_multi_verified(INPUT_PATH, RESNEXT_PATH)
if RESNEXT_WEIGHT is None:
    RESNEXT_WEIGHT = _find_weight_by_keywords_verified(
        INPUT_PATH, keywords=["resnext", "50"]
    )

B4_WEIGHT = _find_weight_file_multi_verified(INPUT_PATH, B4_PATH)
if B4_WEIGHT is None:
    B4_WEIGHT = _find_weight_by_keywords_verified(
        INPUT_PATH, keywords=["efficientnet", "b4"]
    )

print("Weight lookup results (verified head size):")
print(" RESNEXT_WEIGHT:", RESNEXT_WEIGHT)
print(" B4_WEIGHT:", B4_WEIGHT)

WEIGHTS_AVAILABLE = (RESNEXT_WEIGHT is not None) and (B4_WEIGHT is not None)

if not WEIGHTS_AVAILABLE:
    pth_candidates = []
    for root in _candidate_weight_roots(INPUT_PATH):
        if root.exists():
            for ext in ("*.pth", "*.pt", "*.bin"):
                pth_candidates.extend([str(p) for p in root.rglob(ext)])
    pth_candidates = sorted(set(pth_candidates))
    print(
        "WARNING: Required ensemble weights not found/verified. Proceeding with ImageNet-pretrained backbones.\n"
        "We keep a deterministic fallback head so submission is valid.\n"
        f"Missing/Unverified: {RESNEXT_PATH} and/or {B4_PATH}\n"
        f"Found {len(pth_candidates)} checkpoint-like files (showing up to 50):\n  - "
        + "\n  - ".join(pth_candidates[:50])
    )



## === cell 12
if not Path(TEST_IMAGE_PATH).exists():
    raise FileNotFoundError(f"TEST_IMAGE_PATH does not exist: {TEST_IMAGE_PATH}")

device = DEVICES[0] if len(DEVICES) > 0 else torch.device("cpu")


def _load_state_dict_flexible(model: nn.Module, weight_path: str | None) -> bool:
    if weight_path is None:
        return False

    sd = torch.load(weight_path, map_location="cpu")
    sd = _extract_state_dict(sd)

    try:
        model.load_state_dict(sd, strict=True)
        return True
    except RuntimeError:
        pass

    if (
        isinstance(model, nn.DataParallel)
        and isinstance(sd, dict)
        and not any(k.startswith("module.") for k in sd.keys())
    ):
        sd2 = {f"module.{k}": v for k, v in sd.items()}
        model.load_state_dict(sd2, strict=True)
        return True

    if (
        not isinstance(model, nn.DataParallel)
        and isinstance(sd, dict)
        and any(k.startswith("module.") for k in sd.keys())
    ):
        sd2 = {k[7:]: v for k, v in sd.items() if k.startswith("module.")}
        model.load_state_dict(sd2, strict=True)
        return True

    raise


def _train_class_prior_logits(train_csv_path: str, out_features: int) -> torch.Tensor:
    df = pd.read_csv(train_csv_path)
    counts = (
        df["label"].value_counts().reindex(range(out_features), fill_value=0).values
    )
    p = counts / max(counts.sum(), 1)
    p = np.clip(p, 1e-6, 1.0)
    logits = np.log(p)
    logits = logits - logits.mean()
    return torch.tensor(logits, dtype=torch.float32)


def _init_prior_head_linear_from_logits(linear: nn.Linear, class_logits: torch.Tensor):
    with torch.no_grad():
        linear.weight.zero_()
        linear.bias.copy_(
            class_logits.to(dtype=linear.bias.dtype, device=linear.bias.device)
        )


test_image_list = sorted(
    [
        image_name
        for image_name in os.listdir(TEST_IMAGE_PATH)
        if image_name.lower().endswith(".jpg")
    ]
)

prior_logits = _train_class_prior_logits(TRAIN_CSV_PATH, OUT_FEATURES)

my_model_1 = nn.DataParallel(my_model_1).to(device)
if WEIGHTS_AVAILABLE:
    loaded1 = _load_state_dict_flexible(my_model_1, RESNEXT_WEIGHT)
else:
    loaded1 = False
    _init_prior_head_linear_from_logits(my_model_1.module.fc, prior_logits)
print("Loaded resnext weights:", loaded1)
my_model_1.eval()

preds_1 = []
for single_image_name in tqdm(test_image_list, desc="Predict resnext"):
    with torch.no_grad():
        ans = torch.zeros(OUT_FEATURES, device=device)
        image = Image.open(os.path.join(TEST_IMAGE_PATH, single_image_name)).convert(
            "RGB"
        )
        image_np = np.array(image)
        for _ in range(TTA):
            aug_image = test_augs(image=image_np)["image"]
            test_image = aug_image.to(dtype=torch.float32).unsqueeze(0).to(device)
            ans += my_model_1(test_image).view(ans.shape)
        ans /= TTA
        preds_1.append(ans.detach().to("cpu"))

predictions_1 = torch.stack(preds_1, dim=0)
torch.cuda.empty_cache()

my_model_2 = nn.DataParallel(my_model_2).to(device)
if WEIGHTS_AVAILABLE:
    loaded2 = _load_state_dict_flexible(my_model_2, B4_WEIGHT)
else:
    loaded2 = False
    _init_prior_head_linear_from_logits(my_model_2.module.classifier, prior_logits)
print("Loaded effnet_b4_ns weights:", loaded2)
my_model_2.eval()

preds_2 = []
for single_image_name in tqdm(test_image_list, desc="Predict effnet_b4_ns"):
    with torch.no_grad():
        ans = torch.zeros(OUT_FEATURES, device=device)
        image = Image.open(os.path.join(TEST_IMAGE_PATH, single_image_name)).convert(
            "RGB"
        )
        image_np = np.array(image)
        for _ in range(TTA):
            aug_image = test_augs(image=image_np)["image"]
            test_image = aug_image.to(dtype=torch.float32).unsqueeze(0).to(device)
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
