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
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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

0.8482925355092172

# 6. Current score

0.20516

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I fix the runtime blockers that prevent the notebook from running end-to-end: (1) remove the failing TensorBoard `SummaryWriter` import/usage (it isn’t needed for inference), (2) replace the deprecated/removed albumentations imports with only the actually used transforms, and (3) remove the dependency on `efficientnet_pytorch` by switching to torchvision’s built-in EfficientNet-B7 with a matching 5-class head while still loading the provided checkpoint when possible. I also make inference device-safe (use `.to(device)` instead of `.cuda()`) and ensure the dataset returns a dummy label for test rows. Finally, the script always write a valid `submission.csv` with `image_id,label` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.22309) has done: 'Your current score (0.61099) is far below the target (0.84829), and the main reason is that in inference you instantiate EfficientNet-B7 with `weights=None` and then try to load a checkpoint that was likely trained on a different EfficientNet implementation (common for this dataset), so most keys don’t load and you effectively predict with a near-random model head. To move the score toward the target with minimal core-logic change, I keep the same EfficientNet-B7 architecture idea but switch the model construction to match the checkpoint format by using `torchvision.models.efficientnet_b7` with ImageNet weights as a safe fallback and enforce stricter checkpoint loading (raise if too few keys match). I also make inference use softmax+mean over a simple test-time augmentation (horizontal flip) which is a minimal, evaluation-consistent improvement for accuracy without changing training or loss. Submission writing and alignment to `sample_submission.csv` remain identical.'
- What this solution (achieved 0.22309) has done: 'Your current score is far below the target, so the most likely issue is still that the provided checkpoint isn’t being loaded correctly into the torchvision EfficientNet-B7 (you’re only loading the intersecting keys and leaving many weights at ImageNet init, which often behaves close to random for this task). I keep the same EfficientNet-B7 core logic, but change checkpoint loading to be “head-agnostic”: load all matching feature/backbone weights while intentionally skipping the classifier head if shapes differ, and require a higher match ratio for backbone keys (so we don’t silently run with an almost-untrained model). I also ensure deterministic inference ordering/alignment by iterating over `sample_submission.csv` exactly and turning on `model.eval()` + `torch.inference_mode()` (no semantic change, but reduces accidental nondeterminism). These are minimal, metric-aligned changes that should move accuracy upward toward your target without changing architecture/training/loss.'
- What this solution (achieved 0.20478) has done: 'Your score is far below the target, so the most likely remaining issue is still checkpoint incompatibility: you’re only loading weights if backbone key names/shapes match torchvision EfficientNet-B7, which often results in almost no useful loading and near-random predictions. I keep the same EfficientNet-B7 core approach, but make checkpoint loading robust to common naming differences by (1) trying multiple key-prefix rewrites and (2) falling back to loading a full `model = torch.load(...)` object if the checkpoint is a serialized model (both are legitimate and minimal). I also align inference preprocessing to the EfficientNet-B7 ImageNet weights’ canonical transforms (same resize/crop/normalize semantics) to improve accuracy without changing the model or training loop. Finally, I keep the same submission writing, but enforce sample_submission order and sanity-check that we used either a successfully loaded checkpoint or explicit ImageNet fallback.'
- What this solution (achieved 0.20478) has done: 'Your current score (0.20478) is far below the target (0.84829), which strongly suggests the checkpoint still isn’t being loaded into the model you’re running at inference (so you’re effectively predicting with mostly ImageNet/random head). I keep your EfficientNet-B7 + softmax + (orig+hflip) averaging exactly as-is, but make checkpoint loading robust to the most common Cassava EfficientNet training code: checkpoints saved as `model_state_dict`, `net`, `model`, or with classifier named `._fc`, `fc`, or `classifier.1`. I also remove the overly-strict “>=0.90 backbone match” gate and instead always load all shape-matching tensors (skipping only incompatible classifier tensors), and print a clear summary so you can confirm it actually loaded most weights. These are minimal changes focused on getting the trained weights into the same architecture so accuracy moves up toward the target band.'
- What this solution (achieved 0.20516) has done: 'Your score is far below the target, and the current code still looks like it’s running mostly on ImageNet weights because the checkpoint is not being loaded into the model in a way that matches how it was saved. I keep your EfficientNet-B7 inference + (orig+hflip) averaging unchanged, but I make checkpoint loading try a small set of common EfficientNet key patterns (including `features.*` vs `model.*` vs `backbone.*` and `_fc`/`fc` heads) and load the *best* remapping by shape-match count, skipping only incompatible head tensors. I also ensure that if the checkpoint is a full serialized `nn.Module`, we use it directly (and if it’s a dict, we search more keys like `ema`, `student`, etc.). These are minimal, score-relevant changes focused on actually using the trained weights; submission writing/ordering remains exactly aligned to `sample_submission.csv`.'
- What this solution (achieved 0.20516) has done: 'Your current score is far below the target, which strongly indicates the trained checkpoint still isn’t being loaded into the exact model it was trained with (so predictions are effectively near-random). I keep your EfficientNet-B7 inference core and TTA (orig+hflip) unchanged, but I (1) load the checkpoint more faithfully by instantiating an `EfficientNet` model from `torchvision.models.efficientnet.EfficientNet` using the checkpoint’s own `state_dict` shapes, then load that state dict strictly (or head-agnostic when needed), and (2) fix a subtle but critical bug: your remap logic currently can “double-prefix” keys with `features.features.*`, preventing backbone weights from loading. These are minimal, score-relevant changes aimed purely at making inference actually use the trained weights and should move accuracy sharply upward toward your target band while still producing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os, sys

print(
    "Using torchvision EfficientNet-B7; improving checkpoint compatibility by trying multiple key remaps + (if needed) building model from checkpoint shapes; and aligning inference transforms to EfficientNet weights."
)



## === cell 1
import os

WEIGHT_FILE = "../input/ramki-cassava-weights/weight-at-epoch-14-acc-0.85109.pth"
print("WEIGHT_FILE exists:", os.path.exists(WEIGHT_FILE), WEIGHT_FILE)



## === cell 2
import os

base_ls = "../input/cassava-leaf-disease-classification"
print("Base path exists:", os.path.exists(base_ls))
if os.path.exists(base_ls):
    print("Files:", sorted(os.listdir(base_ls))[:20])



## === cell 3
pass



## === cell 4
import os

base_ls = "../input/cassava-leaf-disease-classification"
if os.path.exists(base_ls):
    print("train_images exists:", os.path.exists(os.path.join(base_ls, "train_images")))
    print("test_images exists:", os.path.exists(os.path.join(base_ls, "test_images")))



## === cell 5
pass



## === cell 6
TRAINING = False
WEIGHT_FILE = "../input/ramki-cassava-weights/weight-at-epoch-14-acc-0.85109.pth"



## === cell 7
import numpy as np
import pandas as pd
import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import random

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

SummaryWriter = None



## === cell 8
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2



## === cell 9
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device



## === cell 10
if torch.cuda.is_available():
    print(torch.cuda.device_count())



## === cell 11
writer = None



## === cell 12
SEED = 42
N_FOLDS = 5
N_EPOCHS = 10
BATCH_SIZE = 16
IMG_SIZE = 224
LR = 5e-4
NUM_CLASSES = 5




## === cell 13
def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


seed_everything(SEED)



## === cell 14
base_path = "../input/cassava-leaf-disease-classification/"



## === cell 15
train_path = base_path + "train_images/"
test_path = base_path + "test_images/"

train_csv = pd.read_csv(base_path + "train.csv")
sample = pd.read_csv(base_path + "sample_submission.csv")



## === cell 16
train_csv.head()



## === cell 17
train_csv_disease = train_csv.label.map(
    {
        0: "Cassava Bacterial Blight (CBB)",
        1: "Cassava Brown Streak Disease (CBSD)",
        2: "Cassava Green Mottle (CGM)",
        3: "Cassava Mosaic Disease (CMD)",
        4: "Healthy",
    }
)
diseases = train_csv_disease.value_counts()



## === cell 18
diseases



## === cell 19
ax = diseases.plot.pie(figsize=(6, 6))
plt.show()



## === cell 20
pass




## === cell 21
class CasavaDataset(Dataset):
    def __init__(self, dataframe, transforms=None, test=False):
        self.df = dataframe.reset_index(drop=True)
        self.transforms = transforms
        self.test = test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if (not self.test) and ("label" in self.df.columns):
            label = int(self.df.iloc[idx].label)
        else:
            label = 0

        p = self.df.iloc[idx].image_id
        p_path = (test_path if self.test else train_path) + p

        image = Image.open(p_path).convert("RGB")
        image = np.array(image)

        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed["image"]

        return image, label




## === cell 22
pass



## === cell 23
_EFF_W = models.EfficientNet_B7_Weights.IMAGENET1K_V1
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

transforms_train = A.Compose(
    [
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.3),
        A.Normalize(
            mean=_IMAGENET_MEAN,
            std=_IMAGENET_STD,
            max_pixel_value=255.0,
            p=1.0,
        ),
        ToTensorV2(p=1.0),
    ],
    p=1.0,
)

transforms_valid = A.Compose(
    [
        A.Resize(256, 256),
        A.CenterCrop(IMG_SIZE, IMG_SIZE),
        A.Normalize(
            mean=_IMAGENET_MEAN,
            std=_IMAGENET_STD,
            max_pixel_value=255.0,
            p=1.0,
        ),
        ToTensorV2(p=1.0),
    ],
    p=1.0,
)

transforms_valid_hflip = A.Compose(
    [
        A.Resize(256, 256),
        A.CenterCrop(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=1.0),
        A.Normalize(
            mean=_IMAGENET_MEAN,
            std=_IMAGENET_STD,
            max_pixel_value=255.0,
            p=1.0,
        ),
        ToTensorV2(p=1.0),
    ],
    p=1.0,
)



## === cell 24
pass



## === cell 25
train_csv.shape



## === cell 26
model_name = "efficientnet-b7"

from torchvision.models.efficientnet import EfficientNet, MBConvConfig

model = models.efficientnet_b7(weights=_EFF_W)

in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)


def _clean_state_dict(sd: dict) -> dict:
    cleaned = {}
    for k, v in sd.items():
        nk = k
        for pfx in ["module.", "model.", "net.", "backbone.", "encoder."]:
            if nk.startswith(pfx):
                nk = nk[len(pfx) :]
        cleaned[nk] = v
    return cleaned


def _extract_state_dict(state):
    """
    Broad support for typical checkpoint dict structures (score-relevant only for loading).
    """
    if isinstance(state, torch.nn.Module):
        return state, "full_model_object"

    if not isinstance(state, dict):
        return None, "unknown_format"

    for key in [
        "state_dict",
        "model_state_dict",
        "model",
        "net",
        "network",
        "ema",
        "ema_state_dict",
        "teacher",
        "student",
        "swa_state_dict",
    ]:
        if key in state:
            v = state[key]
            if isinstance(v, torch.nn.Module):
                return v, f"module[{key}]"
            if isinstance(v, dict):
                looks_like_sd = any(
                    isinstance(kk, str) and torch.is_tensor(vv) for kk, vv in v.items()
                )
                if looks_like_sd:
                    return v, f"dict[{key}]"

    looks_like_sd = any(
        isinstance(k, str) and torch.is_tensor(v) for k, v in state.items()
    )
    if looks_like_sd:
        return state, "raw_state_dict"

    return None, "dict_unknown"


def _apply_key_rewrites(k: str, mode: str) -> str:
    """
    Try a few common EfficientNet naming conventions to load trained backbone weights.

    Important bugfix (score-relevant):
    - avoid producing 'features.features.*' when the key already starts with 'features.'.
    """
    nk = k

    if "._fc." in nk:
        nk = nk.replace("._fc.", ".classifier.1.")
    if "_fc." in nk:
        nk = nk.replace("_fc.", "classifier.1.")
    if nk.startswith("fc."):
        nk = "classifier.1." + nk[len("fc.") :]

    if nk in ["classifier.weight", "classifier.bias"]:
        nk = nk.replace("classifier.", "classifier.1.")

    if mode == "as_is":
        return nk

    if mode == "strip_features_prefix":
        if nk.startswith("features."):
            nk = nk[len("features.") :]
        return nk

    if mode == "add_features_prefix":
        if (not nk.startswith("features.")) and (not nk.startswith("classifier.")):
            nk = "features." + nk
        return nk

    return nk


def _score_remap(remapped: dict, model_sd: dict) -> tuple[dict, float, int, int]:
    matched = {
        k: v
        for k, v in remapped.items()
        if (k in model_sd and v.shape == model_sd[k].shape)
    }
    ratio = len(matched) / max(1, len(model_sd))
    bb = [k for k in model_sd.keys() if not k.startswith("classifier.")]
    bb_matched = sum(1 for k in bb if k in matched)
    return matched, ratio, len(matched), bb_matched


def _build_efficientnet_from_ckpt_shapes(cleaned_sd: dict) -> nn.Module | None:
    """
    Change (score-relevant, minimal, architecture-preserving):
    If the checkpoint was trained with torchvision's internal EfficientNet (not the wrapper),
    its keys often start with 'features.*' and include 'features.8.*' (classifier conv).
    We can build a matching EfficientNet-B7 with 5 classes using torchvision's official
    EfficientNet + MBConvConfig, then load the checkpoint strictly/head-agnostic.
    """
    has_features0 = any(
        k.startswith("features.0.") or k.startswith("features.0.0.")
        for k in cleaned_sd.keys()
    )
    has_classifier = any(k.startswith("classifier.") for k in cleaned_sd.keys())
    if not (has_features0 and has_classifier):
        return None

    bneck_conf = partial = MBConvConfig
    inverted_residual_setting = [
        partial(1, 3, 1, 32, 32, 2),
        partial(2, 3, 2, 32, 48, 4),
        partial(2, 5, 2, 48, 80, 4),
        partial(3, 3, 2, 80, 160, 6),
        partial(3, 5, 1, 160, 224, 6),
        partial(4, 5, 2, 224, 384, 8),
        partial(4, 3, 1, 384, 640, 8),
    ]
    m = EfficientNet(
        inverted_residual_setting=inverted_residual_setting,
        dropout=0.5,
        stochastic_depth_prob=0.2,
        num_classes=NUM_CLASSES,
        norm_layer=nn.BatchNorm2d,
        last_channel=2560,
    )
    return m


loaded_from_checkpoint = False

if (not TRAINING) and os.path.exists(WEIGHT_FILE):
    state = torch.load(WEIGHT_FILE, map_location="cpu")
    extracted, extracted_kind = _extract_state_dict(state)

    if isinstance(extracted, torch.nn.Module):
        model = extracted
        model.to(device)
        model.eval()
        loaded_from_checkpoint = True
        print("Loaded full serialized model object from checkpoint.")
    elif isinstance(extracted, dict):
        cleaned = _clean_state_dict(extracted)

        model_sd = model.state_dict()
        modes = ["as_is", "add_features_prefix", "strip_features_prefix"]
        best = None  # (mode, matched, ratio, n_matched, n_bb_matched)

        for mode in modes:
            remapped = {_apply_key_rewrites(k, mode): v for k, v in cleaned.items()}
            matched, ratio, n_matched, n_bb_matched = _score_remap(remapped, model_sd)
            cand = (mode, matched, ratio, n_matched, n_bb_matched)
            if best is None:
                best = cand
            else:
                if (cand[4], cand[3], cand[2]) > (best[4], best[3], best[2]):
                    best = cand

        best_mode, matched, match_ratio, n_matched, n_bb_matched = best

        cls_keys = [k for k in model_sd.keys() if k.startswith("classifier.")]
        cls_matched = sum(1 for k in cls_keys if k in matched)

        print(
            f"Checkpoint extracted as: {extracted_kind} | best remap mode: {best_mode} | "
            f"Total key-shape matches: {n_matched}/{len(model_sd)} ({match_ratio:.2%}) | "
            f"backbone matches: {n_bb_matched}/{len([k for k in model_sd.keys() if not k.startswith('classifier.')])} | "
            f"classifier.* matched: {cls_matched}/{len(cls_keys)}"
        )

        if n_matched > 0:
            missing, unexpected = model.load_state_dict(matched, strict=False)
            loaded_from_checkpoint = True
            print(
                f"Loaded {n_matched} tensors into torchvision wrapper. Missing keys: {len(missing)} Unexpected keys: {len(unexpected)}"
            )
        else:
            m2 = _build_efficientnet_from_ckpt_shapes(cleaned)
            if m2 is not None:
                m2_sd = m2.state_dict()
                shape_matched = {
                    k: v
                    for k, v in cleaned.items()
                    if (k in m2_sd and v.shape == m2_sd[k].shape)
                }
                print(
                    f"Fallback build EfficientNet from ckpt-shapes: matched {len(shape_matched)}/{len(m2_sd)} tensors."
                )
                if len(shape_matched) > 0:
                    m2.load_state_dict(shape_matched, strict=False)
                    model = m2
                    loaded_from_checkpoint = True
                    print(
                        "Using EfficientNet instance built from checkpoint-compatible shapes."
                    )
                else:
                    print(
                        "Fallback build found 0 matching tensors; using ImageNet weights fallback."
                    )
            else:
                print(
                    "Checkpoint keys not compatible with internal EfficientNet signature; using ImageNet weights fallback."
                )
    else:
        print("Checkpoint format not understood; using ImageNet weights fallback.")
else:
    if not TRAINING:
        print("Weight file not found; using ImageNet weights fallback.")

model.to(device)
print("Model ready on", device, "| loaded_from_checkpoint:", loaded_from_checkpoint)



## === cell 27
pass



## === cell 28
layer = 0
for child in model.children():
    layer += 1
print(layer)



## === cell 29
pass



## === cell 30
pass



## === cell 31
pass



## === cell 32
pass



## === cell 33
pass



## === cell 34
trainset = CasavaDataset(train_csv, transforms=transforms_train, test=False)
train_loader = DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=torch.cuda.is_available(),
)

testset = CasavaDataset(sample, transforms=transforms_valid, test=True)
test_loader = DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=torch.cuda.is_available(),
)

testset_hflip = CasavaDataset(sample, transforms=transforms_valid_hflip, test=True)
test_loader_hflip = DataLoader(
    testset_hflip,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=torch.cuda.is_available(),
)



## === cell 35
len(train_loader)



## === cell 36
len(test_loader)



## === cell 37
BATCH_SIZE




## === cell 38
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0.0




## === cell 39
def show_metrics(model, epoch, dataloader, criterion, optimizer):
    model.eval()

    losses = AverageMeter()
    accs = AverageMeter()

    all_preds = []
    all_labels = []

    tk = tqdm(dataloader, total=len(dataloader), position=0, leave=True)
    for idx, (imgs, labels) in enumerate(tk):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()

        with torch.no_grad():
            output = model(imgs)
            loss = criterion(output, labels)

        preds = output.argmax(1)
        all_preds.append(preds.detach().cpu())
        all_labels.append(labels.detach().cpu())

        correctly = (preds == labels).sum().item()
        n = imgs.size(0)
        accs.update(correctly / n, n)
        losses.update(loss.item(), n)

        tk.set_postfix(loss=losses.avg, acc=accs.avg)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    cf_matrix = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cf_matrix, annot=True, fmt="d", cmap="YlGnBu")
    plt.show()

    return losses.avg, all_preds, all_labels




## === cell 40
(
    "Cassava Bacterial Blight (CBB)",
    "Cassava Brown Streak Disease (CBSD)",
    "Cassava Green Mottle (CGM)",
    "Cassava Mosaic Disease (CMD)",
    "Healthy",
)




## === cell 41
def train_model(model, epoch, dataloader_train, criterion, optimizer):
    model.train()

    losses = AverageMeter()
    accs = AverageMeter()
    tk = tqdm(dataloader_train, total=len(dataloader_train), position=0, leave=True)
    for idx, (imgs, labels) in enumerate(tk):
        imgs_train = imgs.to(device, non_blocking=True)
        labels_train = labels.to(device, non_blocking=True).long()

        output_train = model(imgs_train)
        loss = criterion(output_train, labels_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predicted_classes = output_train.argmax(1)
        correctly_identified_sum = (predicted_classes == labels_train).sum().item()
        number_of_images = imgs_train.size(0)

        accs.update(correctly_identified_sum / number_of_images, number_of_images)
        losses.update(loss.item(), number_of_images)

        tk.set_postfix(loss=losses.avg, acc=accs.avg)

    return losses.avg, accs.avg


def test_model(model, dataloader_valid, criterion):
    model.eval()

    losses = AverageMeter()
    accs = AverageMeter()

    with torch.no_grad():
        tk = tqdm(dataloader_valid, total=len(dataloader_valid), position=0, leave=True)
        for idx, (imgs, labels) in enumerate(tk):
            imgs_valid = imgs.to(device, non_blocking=True)
            labels_valid = labels.to(device, non_blocking=True).long()

            output_valid = model(imgs_valid)
            loss = criterion(output_valid, labels_valid)

            losses.update(loss.item(), imgs_valid.size(0))
            accs.update(
                (output_valid.argmax(1) == labels_valid).sum().item()
                / imgs_valid.size(0),
                imgs_valid.size(0),
            )

            tk.set_postfix(loss=losses.avg, acc=accs.avg)

    return losses.avg, accs.avg




## === cell 42
X = train_csv.iloc[:, :-1]
y = train_csv.iloc[:, -1]

folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)



## === cell 43
if TRAINING:
    for i_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
        print("Fold {}/{}".format(i_fold + 1, N_FOLDS))

        train = train_csv.iloc[train_idx].reset_index(drop=True)
        valid = train_csv.iloc[valid_idx].reset_index(drop=True)

        dataset_train = CasavaDataset(train, transforms=transforms_train, test=False)
        dataset_valid = CasavaDataset(valid, transforms=transforms_valid, test=False)

        dataloader_train = DataLoader(
            dataset_train, batch_size=BATCH_SIZE, num_workers=4, shuffle=True
        )
        dataloader_valid = DataLoader(
            dataset_valid, batch_size=BATCH_SIZE, num_workers=4, shuffle=False
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=1, verbose=True, min_lr=1e-5
        )

        best_acc = 0.0
        for epoch in range(N_EPOCHS):
            train_loss, train_acc = train_model(
                model, epoch, dataloader_train, criterion, optimizer
            )
            val_loss, acc = test_model(model, dataloader_valid, criterion)

            scheduler.step(acc)

            if writer is not None:
                writer.add_scalar(
                    "training acc", train_acc, i_fold * N_EPOCHS + epoch + 1
                )
                writer.add_scalar(
                    "training loss", train_loss, i_fold * N_EPOCHS + epoch + 1
                )
                writer.add_scalar(
                    "validation loss", val_loss, i_fold * N_EPOCHS + epoch + 1
                )
                writer.add_scalar("validation Acc", acc, i_fold * N_EPOCHS + epoch + 1)
                writer.flush()

            if acc > best_acc:
                best_acc = acc
                os.makedirs("weights1", exist_ok=True)
                torch.save(
                    model.state_dict(),
                    "weights1/weight-at-fold-{}-epoch-{}-acc-{:.5}.pth".format(
                        i_fold + 1, epoch, best_acc
                    ),
                )

            print("current_val_acc:", acc, "best_val_acc:", best_acc)



## === cell 44
test_pred = []

model.eval()
all_probs = []

with torch.inference_mode():
    tk1 = tqdm(test_loader, position=0, leave=True, desc="Infer (orig)")
    for images, _ in tk1:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        all_probs.append(probs.detach().cpu())

all_probs_flip = []
with torch.inference_mode():
    tk2 = tqdm(test_loader_hflip, position=0, leave=True, desc="Infer (hflip)")
    for images, _ in tk2:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        all_probs_flip.append(probs.detach().cpu())

probs1 = torch.cat(all_probs, dim=0)
probs2 = torch.cat(all_probs_flip, dim=0)
assert probs1.shape == probs2.shape
probs = 0.5 * (probs1 + probs2)

test_pred = probs.argmax(1).numpy().astype(int).tolist()

assert len(test_pred) == len(sample), (len(test_pred), len(sample))

sample_out = sample.copy()
sample_out["label"] = test_pred

sample_out = sample_out[["image_id", "label"]]
sample_out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sample_out.shape)



## === cell 45
sample_out.head()



## === cell 46
with open("submission.csv", "r") as f:
    for _ in range(10):
        print(f.readline().rstrip())
